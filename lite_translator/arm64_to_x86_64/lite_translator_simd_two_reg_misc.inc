/*
 * Copyright (C) 2026 utzcoz
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Textual fragment of class LiteTranslator; included by lite_translator.h
// inside the class body. Not a standalone header -- no include guards on
// purpose (no #ifndef / #include / namespace). See lite_translator.h.
//
// Family: SIMD two-register-misc vector.
// Methods:
//   AdvSimdTwoRegMisc

  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& args) {
    // JIT for CMEQZ (cmeq Vd, Vn, #0) used by the
    // dynamic linker's calculate_gnu_hash_neon. Other opcodes fall
    // through to the interpreter.
    int32_t vn_off = VRegOffset(args.rn);
    int32_t vd_off = VRegOffset(args.rd);

    switch (args.opcode) {
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmeqZero: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xz = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xz, xz);
        switch (args.size) {
          case 0b00: as_.Pcmpeqb(xn, xz); break;
          case 0b01: as_.Pcmpeqw(xn, xz); break;
          case 0b10: as_.Pcmpeqd(xn, xz); break;
          default:  // 0b11 (.2D): PCMPEQQ is SSE4.1.
            if (!host_platform::kHasSSE4_1) { Undefined(); return; }
            as_.Pcmpeqq(xn, xz);
            break;
        }
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }

      // SHLL/SHLL2 (shift left long by element size). Each
      // source element is zero-extended to twice its width and shifted left by
      // the source element-size in bits (so the source lands in the high half).
      // size=00 .8B->.8H, 01 .4H->.4S, 10 .2S->.2D. Q=0 (SHLL) reads the low 8
      // source bytes, Q=1 (SHLL2) the high 8 (PSRLDQ first). Lowered as
      // PMOVZX widen + PSLL by 8/16/32.
      case Decoder::AdvSimdTwoRegMiscOpcode::kShll: {
        if (args.size > 0b10) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (args.q) as_.Psrldq(xn, int8_t{8});  // SHLL2: bring high 8 bytes low
        switch (args.size) {
          case 0b00: as_.Pmovzxbw(xn, xn); as_.Psllw(xn, int8_t{8}); break;
          case 0b01: as_.Pmovzxwd(xn, xn); as_.Pslld(xn, int8_t{16}); break;
          default:   as_.Pmovzxdq(xn, xn); as_.Psllq(xn, int8_t{32}); break;  // 0b10
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }

      // URECPE/URSQRTE (.2S/.4S): per-lane unsigned integer
      // reciprocal / reciprocal-sqrt estimate. The result is a pure function of
      // the 9-bit field (lane>>23)&0x1FF, so extract that index per lane and
      // load the precomputed estimate from the table (matches the interpreter
      // bit-for-bit). Q=0 zeroes the upper 64 bits (xres starts zeroed).
      case Decoder::AdvSimdTwoRegMiscOpcode::kUrecpe:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUrsqrte: {
        if (args.size != 0b10) { success_ = false; return; }  // 32-bit lanes only
        const bool is_rsqrt =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUrsqrte);
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xres = AllocTempSimdReg();
        Register tbl = AllocTempReg();
        Register idx = AllocTempReg();
        Register val = AllocTempReg();
        if (xn == no_simd_register || xres == no_simd_register ||
            tbl == no_register || idx == no_register || val == no_register) {
          success_ = false; return;
        }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xres, xres);
        as_.Movq(tbl, reinterpret_cast<int64_t>(UnsignedEstimateTable(is_rsqrt)));
        const int lanes = args.q ? 4 : 2;
        for (int i = 0; i < lanes; ++i) {
          as_.Pextrd(idx, xn, static_cast<int8_t>(i));
          as_.Shrl(idx, int8_t{23});
          as_.Andl(idx, int32_t{0x1FF});
          as_.Movl(val, {.base = tbl, .index = idx, .scale = Assembler::kTimesFour});
          as_.Pinsrd(xres, val, static_cast<int8_t>(i));
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xres);
        return;
      }

      // FCVTXN / FCVTXN2 (vector FP64->FP32, round-to-odd).
      // Per lane, mirror the validated scalar FCVTXN: one-shot MXCSR RC=RTZ +
      // CVTSD2SS, then OR the FP32 LSB if that lane's PE (inexact) bit was set.
      // (PE stays 0 for NaN/Inf/exact, so those keep CVTSD2SS's value.) Two
      // doubles -> two floats; Q=0 (FCVTXN) writes the low 64 of Vd (upper
      // zeroed), Q=1 (FCVTXN2) the high 64 (low preserved). Cumulative MXCSR
      // exception bits are preserved across the op.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtxn: {
        if (args.size != 0b01) { success_ = false; return; }  // FP64 source only
        SimdRegister xmm = AllocTempSimdReg();
        SimdRegister xres = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        Register mxcsr_save = AllocTempReg();
        Register mxcsr_after = AllocTempReg();
        if (xmm == no_simd_register || xres == no_simd_register ||
            tmp == no_register || mxcsr_save == no_register ||
            mxcsr_after == no_register) {
          success_ = false; return;
        }
        as_.Pxor(xres, xres);
        as_.Stmxcsr({.base = Assembler::rsp, .disp = 0});
        as_.Movl(mxcsr_save, {.base = Assembler::rsp, .disp = 0});
        for (int lane = 0; lane < 2; lane++) {
          as_.Movsd(xmm, {.base = Assembler::rbp, .disp = vn_off + lane * 8});
          // RC=RTZ (bits 13-14 = 11), clear the six exception flags.
          as_.Movl(mxcsr_after, mxcsr_save);
          as_.Andl(mxcsr_after, int32_t{~int32_t{0x3F}});
          as_.Orl(mxcsr_after, int32_t{0x6000});
          as_.Movl({.base = Assembler::rsp, .disp = 0}, mxcsr_after);
          as_.Ldmxcsr({.base = Assembler::rsp, .disp = 0});
          as_.Cvtsd2ss(xmm, xmm);
          as_.Stmxcsr({.base = Assembler::rsp, .disp = 0});
          as_.Movl(mxcsr_after, {.base = Assembler::rsp, .disp = 0});
          as_.Andl(mxcsr_after, int32_t{0x3F});
          as_.Orl(mxcsr_save, mxcsr_after);  // accumulate cumulative exceptions
          as_.Movd(tmp, xmm);
          Assembler::Label* done = as_.MakeLabel();
          as_.Testl(mxcsr_after, int32_t{0x20});  // PE (inexact)
          as_.Jcc(Assembler::Condition::kZero, *done);
          as_.Orl(tmp, int32_t{1});  // round-to-odd: force LSB
          as_.Bind(done);
          as_.Pinsrd(xres, tmp, static_cast<int8_t>(lane));
        }
        // Restore cumulative MXCSR (original RC + accumulated exceptions).
        as_.Movl({.base = Assembler::rsp, .disp = 0}, mxcsr_save);
        as_.Ldmxcsr({.base = Assembler::rsp, .disp = 0});
        if (!args.q) {
          SimdRegister xz = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Pxor(xz, xz);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xz);
          as_.Movq({.base = Assembler::rbp, .disp = vd_off}, xres);
        } else {
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
          as_.Pslldq(xres, int8_t{8});  // move the 2 floats into the high 64
          MaskLow64(xd);               // keep Vd's low 64, zero the high
          as_.Por(xd, xres);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
        }
        return;
      }

      // XTN / XTN2 (truncating extract narrow). args.size
      // selects the destination element width (00=.8B, 01=.4H, 10=.2S); each
      // source lane is twice as wide.  Q=0 writes the packed result to the low
      // 64 bits (upper zeroed); Q=1 (XTN2) writes the upper 64 bits, preserving
      // Vd's low 64.  Truncation via mask-then-unsigned-pack (the mask makes
      // PACKUS* saturation a no-op); .2D->.2S gathers the low dwords.
      case Decoder::AdvSimdTwoRegMiscOpcode::kXtn: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xz = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xz, xz);
        if (args.size == 0b00 || args.size == 0b01) {
          SimdRegister xm = AllocTempSimdReg();
          Register tmp = AllocTempReg();
          if (xm == no_simd_register || tmp == no_register) { success_ = false; return; }
          as_.Movq(tmp, args.size == 0b00 ? int64_t{0x00FF00FF00FF00FFLL}
                                          : int64_t{0x0000FFFF0000FFFFLL});
          as_.Movq(xm, tmp);
          as_.Punpcklqdq(xm, xm);
          as_.Pand(xn, xm);
          if (args.size == 0b00) {
            as_.Packuswb(xn, xz);  // 8 words -> 8 bytes, low 64
          } else {
            as_.Packusdw(xn, xz);  // 4 dwords -> 4 words, low 64
          }
        } else if (args.size == 0b10) {
          as_.Pshufd(xn, xn, int8_t{0b00001000});  // [d0, d2, *, *]
          MaskLow64(xn);                           // -> [d0, d2, 0, 0]
        } else {
          success_ = false;
          return;
        }
        if (!args.q) {
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        } else {
          // XTN2: shift packed result into the upper 64 bits, then merge with
          // Vd's preserved low 64.
          as_.Pslldq(xn, int8_t{8});
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
          MaskLow64(xd);
          as_.Por(xn, xd);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        }
        return;
      }

      // SQXTN / UQXTN / SQXTUN (saturating extract narrow).
      // Same dst-width selection and Q=0/Q2 layout as XTN; the pack differs by
      // saturation flavour:
      //   SQXTN  signed src -> signed dst   : PACKSSWB / PACKSSDW
      //   SQXTUN signed src -> unsigned dst : PACKUSWB / PACKUSDW
      //   UQXTN  unsigned src -> unsigned dst: clamp via PMINUW/PMINUD then
      //                                        PACKUSWB/PACKUSDW (the clamp
      //                                        keeps values in the positive
      //                                        signed range so PACKUS is exact).
      // .2D->.2S (size=10) has no x86 narrowing pack; it is handled below by
      // per-lane clamp + PSHUFD (SSE4.2).
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqxtn:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUqxtn:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun: {
        const auto opc = args.opcode;
        if (args.size == 0b10) {
          // .2D->.2S / .2D->.4S(2): 64->32 saturating narrow. x86 has no 64->32
          // pack, so clamp each 64-bit lane into the destination range, then
          // gather the two low dwords with PSHUFD. Signed clamps need PCMPGTQ
          // (SSE4.2); bail to the interpreter when the host lacks it.
          if (!host_platform::kHasSSE4_2) { success_ = false; return; }
          SimdRegister x = AllocTempSimdReg();
          SimdRegister c = AllocTempSimdReg();
          SimdRegister m = AllocTempSimdReg();
          SimdRegister t = AllocTempSimdReg();
          Register gp = AllocTempReg();
          if (x == no_simd_register || c == no_simd_register ||
              m == no_simd_register || t == no_simd_register ||
              gp == no_register) { success_ = false; return; }
          as_.Movdqu(x, {.base = Assembler::rbp, .disp = vn_off});
          auto set_const = [&](SimdRegister r, int64_t v) {
            as_.Movq(gp, v);
            as_.Movq(r, gp);
            as_.Punpcklqdq(r, r);  // replicate to both 64-bit lanes
          };
          // x = (x & ~mask) | (val & mask), via x ^= (x ^ val) & mask.
          auto blend = [&](SimdRegister val, SimdRegister mask) {
            as_.Movdqa(t, x);
            as_.Pxor(t, val);
            as_.Pand(t, mask);
            as_.Pxor(x, t);
          };
          if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtn) {
            set_const(c, int64_t{0x000000007FFFFFFFLL});      // INT32_MAX
            as_.Movdqa(m, x); as_.Pcmpgtq(m, c);              // x > MAX
            blend(c, m);
            set_const(c, static_cast<int64_t>(0xFFFFFFFF80000000ULL));  // INT32_MIN
            as_.Movdqa(m, c); as_.Pcmpgtq(m, x);              // x < MIN
            blend(c, m);
          } else if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun) {
            set_const(c, int64_t{0x00000000FFFFFFFFLL});      // UINT32_MAX
            as_.Movdqa(m, x); as_.Pcmpgtq(m, c);              // x > UMAX (x>=0 region)
            blend(c, m);
            as_.Pxor(c, c);                                    // 0
            as_.Movdqa(m, c); as_.Pcmpgtq(m, x);              // x < 0
            as_.Pandn(m, x); as_.Movdqa(x, m);                // negatives -> 0
          } else {  // kUqxtn: unsigned uint64 -> clamp to UINT32_MAX
            set_const(c, int64_t{0x00000000FFFFFFFFLL});      // UINT32_MAX
            as_.Movdqa(m, x); as_.Psrlq(m, int8_t{32});       // high 32 bits
            as_.Pxor(t, t); as_.Pcmpeqq(m, t);                // mask: high32 == 0 (keep x)
            as_.Movdqa(t, x); as_.Pxor(t, c); as_.Pandn(m, t); as_.Pxor(x, m);
          }
          as_.Pshufd(x, x, int8_t{0x08});  // {dword0, dword2} -> low 64
          if (!args.q) {
            MaskLow64(x);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, x);
          } else {
            as_.Pslldq(x, int8_t{8});
            SimdRegister xd = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
            MaskLow64(xd);
            as_.Por(x, xd);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, x);
          }
          return;
        }
        if (args.size != 0b00 && args.size != 0b01) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xz = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xz, xz);
        if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kUqxtn) {
          SimdRegister xm = AllocTempSimdReg();
          Register tmp = AllocTempReg();
          if (xm == no_simd_register || tmp == no_register) { success_ = false; return; }
          as_.Movq(tmp, args.size == 0b00 ? int64_t{0x00FF00FF00FF00FFLL}
                                          : int64_t{0x0000FFFF0000FFFFLL});
          as_.Movq(xm, tmp);
          as_.Punpcklqdq(xm, xm);
          if (args.size == 0b00) {
            as_.Pminuw(xn, xm);
            as_.Packuswb(xn, xz);
          } else {
            as_.Pminud(xn, xm);
            as_.Packusdw(xn, xz);
          }
        } else if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun) {
          if (args.size == 0b00) {
            as_.Packuswb(xn, xz);
          } else {
            as_.Packusdw(xn, xz);
          }
        } else {  // kSqxtn
          if (args.size == 0b00) {
            as_.Packsswb(xn, xz);
          } else {
            as_.Packssdw(xn, xz);
          }
        }
        if (!args.q) {
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        } else {
          as_.Pslldq(xn, int8_t{8});
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
          MaskLow64(xd);
          as_.Por(xn, xd);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        }
        return;
      }

      // FCVTL / FCVTN (FP32<->FP64, size=01).  FCVTL widens
      // 2 floats to 2 doubles (CVTPS2PD; Q=1/FCVTL2 takes the upper floats);
      // FCVTN narrows 2 doubles to 2 floats (CVTPD2PS; Q=0 low+zero-upper,
      // Q=1/FCVTN2 upper-half merge).  size=00 (FP16) is JIT-lowered via F16C
      // (VCVTPH2PS / VCVTPS2PH), bailing to the interpreter only without F16C.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtl: {
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (args.size == 0b01) {
          // FP32 -> FP64 widen.
          if (args.q) as_.Psrldq(xn, int8_t{8});  // FCVTL2: high 2 floats -> low
          as_.Cvtps2pd(xn, xn);
        } else if (args.size == 0b00) {
          // FP16 -> FP32 widen (4 lanes) via F16C; widening is exact.
          if (!host_platform::kHasF16C) { success_ = false; return; }
          if (args.q) as_.Psrldq(xn, int8_t{8});  // FCVTL2: high 4 halves -> low
          as_.Vcvtph2ps(xn, xn);
        } else {
          success_ = false;
          return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtn: {
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (args.size == 0b01) {
          as_.Cvtpd2ps(xn, xn);  // 2 doubles -> 2 floats in low 64, upper zeroed
        } else if (args.size == 0b00) {
          // FP32 -> FP16 narrow (4 lanes) via F16C, round-to-nearest-even
          // (imm8=0). Result in low 64, upper zeroed.
          if (!host_platform::kHasF16C) { success_ = false; return; }
          as_.Vcvtps2ph(xn, xn, int8_t{0});
        } else {
          success_ = false;
          return;
        }
        if (!args.q) {
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        } else {
          as_.Pslldq(xn, int8_t{8});
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
          MaskLow64(xd);
          as_.Por(xn, xd);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        }
        return;
      }

      // CMGT/CMGE/CMLE/CMLT Vd.<T>, Vn.<T>, #0 -- per-lane signed compare against 0.
      //   CMGT (Vn > 0)  ->  PCMPGTx(Vn, 0)
      //   CMLT (Vn < 0)  ->  PCMPGTx(0, Vn)
      //   CMGE (Vn >= 0) ->  ~PCMPGTx(0, Vn)   (XOR with all-ones)
      //   CMLE (Vn <= 0) ->  ~PCMPGTx(Vn, 0)
      // Integer signed compare only — FP16 form (args.is_fp16) bails to interp
      // via success_=false. .2D (size=11) uses PCMPGTQ and is gated on SSE4.2.
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmltZero: {
        if (args.is_fp16) { success_ = false; return; }
        if (args.size == 0b11 && !host_platform::kHasSSE4_2) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xz = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xz, xz);
        // (Vn > 0) for CMGT/CMLE; (0 > Vn) for CMLT/CMGE.
        const bool n_gt_z =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero);
        if (n_gt_z) {
          switch (args.size) {
            case 0b00: as_.Pcmpgtb(xn, xz); break;
            case 0b01: as_.Pcmpgtw(xn, xz); break;
            case 0b10: as_.Pcmpgtd(xn, xz); break;
            default: as_.Pcmpgtq(xn, xz); break;  // 0b11 (.2D)
          }
        } else {
          switch (args.size) {
            case 0b00: as_.Pcmpgtb(xz, xn); break;
            case 0b01: as_.Pcmpgtw(xz, xn); break;
            case 0b10: as_.Pcmpgtd(xz, xn); break;
            default: as_.Pcmpgtq(xz, xn); break;  // 0b11 (.2D)
          }
          as_.Movdqa(xn, xz);
        }
        // CMGE = NOT(0 > Vn); CMLE = NOT(Vn > 0). Invert via XOR with all-ones.
        const bool invert =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero);
        if (invert) {
          SimdRegister allones = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Pcmpeqd(allones, allones);
          as_.Pxor(xn, allones);
        }
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // FCMxx Vd.<T>, Vn.<T>, #0.0 — per-lane FP compare against +0.0.
      //   FCMGT: Vn >  0  -> Cmplt*p* xz, xn   (xz < Vn  == Vn > 0)
      //   FCMGE: Vn >= 0  -> Cmple*p* xz, xn   (xz <= Vn == Vn >= 0)
      //   FCMEQ: Vn == 0  -> Cmpeq*p* xn, xz
      //   FCMLE: Vn <= 0  -> Cmple*p* xn, xz
      //   FCMLT: Vn <  0  -> Cmplt*p* xn, xz
      // SSE legacy Cmp{eq,lt,le}p{s,d} are ordered — they write 0 (FALSE) on
      // any NaN operand, matching ARM FCMxx zero semantics.  Result lane is
      // 0xFFFFFFFF (FP32) / 0xFFFF..FFFF (FP64) on TRUE, zero on FALSE.
      // Decoder gates: bit23=1 required (size & 0b10), .2D (size=0b11) needs
      // Q=1, .1D (size=0b11 Q=0) is ARM-reserved.  args.is_fp16 is always
      // false for this encoding column — FP16 FCMxxZero lives in a different
      // encoding slot (interpreter handles).
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgtZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgeZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmeqZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmleZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmltZero: {
        if (args.is_fp16) { success_ = false; return; }
        // size encodes bit23:bit22. FP forms always have bit23=1 → size&0b10.
        if ((args.size & 0b10) == 0) { Undefined(); return; }
        const bool is_double = (args.size == 0b11);
        if (is_double && !args.q) { Undefined(); return; }  // .1D reserved
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xz = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xz, xz);
        using Op = Decoder::AdvSimdTwoRegMiscOpcode;
        switch (args.opcode) {
          case Op::kFcmeqZero:
            if (is_double) as_.Cmpeqpd(xn, xz);
            else           as_.Cmpeqps(xn, xz);
            break;
          case Op::kFcmgtZero:
            // Vn > 0  <=>  0 < Vn ; result lands in xz, move to xn.
            if (is_double) as_.Cmpltpd(xz, xn);
            else           as_.Cmpltps(xz, xn);
            as_.Movdqa(xn, xz);
            break;
          case Op::kFcmgeZero:
            // Vn >= 0  <=>  0 <= Vn
            if (is_double) as_.Cmplepd(xz, xn);
            else           as_.Cmpleps(xz, xn);
            as_.Movdqa(xn, xz);
            break;
          case Op::kFcmltZero:
            if (is_double) as_.Cmpltpd(xn, xz);
            else           as_.Cmpltps(xn, xz);
            break;
          case Op::kFcmleZero:
            if (is_double) as_.Cmplepd(xn, xz);
            else           as_.Cmpleps(xn, xz);
            break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // FRECPE Vd.<T>, Vn.<T> — per-lane FP reciprocal estimate.
      // FRSQRTE Vd.<T>, Vn.<T> — per-lane FP reciprocal-square-root estimate.
      // Vector form: bit23=1 always; size=0b10 -> FP32, size=0b11 -> FP64.
      // FP16 form (separate encoding column, args.is_fp16==true) bails to
      // the interpreter; .1D (size=0b11, Q=0) is ARM-reserved.
      //
      // ARM ARM C7.2.118 / C7.2.131 require only ~8-bit mantissa precision;
      // the exact 1/x via DIVP{s,d} (FRECPE) and 1/sqrt(x) via SQRTP{s,d} +
      // DIVP{s,d} (FRSQRTE) is well within bound and matches the scalar
      // FRECPE/FRSQRTE lowering for endpoint consistency.
      //
      // FRECPE endpoints fall out of SSE IEEE-754 divide natively:
      //   ±0 -> ±inf, ±inf -> ±0, NaN -> NaN propagated.
      // FRSQRTE special case: ARM specifies negative/zero/NaN input returns
      // default qNaN, but SSE SQRT propagates NaN / writes signed inf for
      // ±0.  Branchless blend:
      //   sqrt   = SQRT(src)
      //   recip  = 1.0 / sqrt
      //   mask   = CMPLE(src, 0) | CMPUNORD(src, src)
      //   result = (mask & qNaN) | (~mask & recip)
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrecpeV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrsqrteV: {
        if (args.is_fp16) { success_ = false; return; }
        if ((args.size & 0b10) == 0) { Undefined(); return; }
        const bool is_double = (args.size == 0b11);
        if (is_double && !args.q) { Undefined(); return; }  // .1D reserved
        const bool is_rsqrt =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFrsqrteV);

        if (is_rsqrt) {
          // FRSQRTE: needs the (src <= 0 || isNaN(src)) -> qNaN blend.
          SimdRegister src = AllocTempSimdReg();
          SimdRegister sqrt_xmm = AllocTempSimdReg();
          SimdRegister recip = AllocTempSimdReg();
          SimdRegister mask_xmm = AllocTempSimdReg();
          SimdRegister mask_nan = AllocTempSimdReg();
          Register tmp = AllocTempReg();
          if (src == no_simd_register || sqrt_xmm == no_simd_register ||
              recip == no_simd_register || mask_xmm == no_simd_register ||
              mask_nan == no_simd_register || tmp == no_register) {
            success_ = false; return;
          }
          as_.Movdqu(src, {.base = Assembler::rbp, .disp = vn_off});
          // sqrt = SQRT(src)
          if (is_double) as_.Sqrtpd(sqrt_xmm, src);
          else           as_.Sqrtps(sqrt_xmm, src);
          // recip = 1.0 / sqrt(src) via broadcasted 1.0 vector.
          if (is_double) {
            as_.Movq(tmp, int64_t{0x3FF0000000000000LL});  // 1.0 (FP64)
            as_.Movq(recip, tmp);
            as_.Pshufd(recip, recip, static_cast<int8_t>(0x44));
            as_.Divpd(recip, sqrt_xmm);
          } else {
            as_.Movl(tmp, int32_t{0x3F800000});  // 1.0f (FP32)
            as_.Movd(recip, tmp);
            as_.Pshufd(recip, recip, static_cast<int8_t>(0x00));
            as_.Divps(recip, sqrt_xmm);
          }
          // sqrt_xmm is dead — reuse it as the all-zero vector for CMPLE.
          as_.Pxor(sqrt_xmm, sqrt_xmm);
          as_.Movdqa(mask_xmm, src);
          if (is_double) as_.Cmplepd(mask_xmm, sqrt_xmm);
          else           as_.Cmpleps(mask_xmm, sqrt_xmm);
          // mask_nan = CMPUNORD(src, src) — true lanes where src is NaN.
          as_.Movdqa(mask_nan, src);
          if (is_double) as_.Cmpunordpd(mask_nan, mask_nan);
          else           as_.Cmpunordps(mask_nan, mask_nan);
          as_.Por(mask_xmm, mask_nan);
          // mask_nan is dead — reuse for the broadcast qNaN.
          if (is_double) {
            as_.Movq(tmp, int64_t{0x7FF8000000000000LL});  // default qNaN
            as_.Movq(mask_nan, tmp);
            as_.Pshufd(mask_nan, mask_nan, static_cast<int8_t>(0x44));
          } else {
            as_.Movl(tmp, int32_t{0x7FC00000});  // default qNaN (FP32)
            as_.Movd(mask_nan, tmp);
            as_.Pshufd(mask_nan, mask_nan, static_cast<int8_t>(0x00));
          }
          // result = (mask_xmm & qNaN) | (~mask_xmm & recip)
          as_.Pand(mask_nan, mask_xmm);
          as_.Pandn(mask_xmm, recip);
          as_.Por(mask_nan, mask_xmm);
          if (!args.q) MaskLow64(mask_nan);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, mask_nan);
          return;
        }
        // FRECPE: recip = 1.0 / src per lane. SSE DIV matches ARM endpoints.
        SimdRegister src = AllocTempSimdReg();
        SimdRegister recip = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        if (src == no_simd_register || recip == no_simd_register ||
            tmp == no_register) {
          success_ = false; return;
        }
        as_.Movdqu(src, {.base = Assembler::rbp, .disp = vn_off});
        if (is_double) {
          as_.Movq(tmp, int64_t{0x3FF0000000000000LL});
          as_.Movq(recip, tmp);
          as_.Pshufd(recip, recip, static_cast<int8_t>(0x44));
          as_.Divpd(recip, src);
        } else {
          as_.Movl(tmp, int32_t{0x3F800000});
          as_.Movd(recip, tmp);
          as_.Pshufd(recip, recip, static_cast<int8_t>(0x00));
          as_.Divps(recip, src);
        }
        if (!args.q) MaskLow64(recip);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, recip);
        return;
      }
      // SQABS Vd.<T>, Vn.<T> — per-lane saturating signed absolute value.
      // SQNEG Vd.<T>, Vn.<T> — per-lane saturating signed negate.
      //   size=00 .8B/.16B  -> byte ops via PCMPGTB / PSUBB / PCMPEQB
      //   size=01 .4H/.8H   -> word ops
      //   size=10 .2S/.4S   -> dword ops
      //   size=11 .2D       -> needs PCMPGTQ (SSE4.2) and PCMPEQQ (SSE4.1);
      //                        bail to interp (same boundary as kAbs/kNeg).
      // Per-lane semantics — INT_MIN_w is the only saturating input (|INT_MIN|
      // and -INT_MIN both overflow the signed range):
      //   if src == INT_MIN_w -> INT_MAX_w
      //   else SQABS -> |src|; SQNEG -> -src
      // Saturation as branchless XOR: after the abs/neg step, lanes whose
      // source was INT_MIN still hold INT_MIN (0x80..0).  XOR-ing INT_MIN
      // with all-1s yields INT_MAX (0x7F..F).  Build the saturation mask as
      // PCMPEQ(src, INT_MIN_bcast) and XOR into the result; non-INT_MIN lanes
      // XOR with zero and pass through unchanged.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqabs:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqneg: {
        if (args.size == 0b11) { success_ = false; return; }
        const bool is_neg =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSqneg);
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xres = AllocTempSimdReg();
        SimdRegister xtmp = AllocTempSimdReg();
        Register gp_min = AllocTempReg();
        if (xn == no_simd_register || xres == no_simd_register ||
            xtmp == no_simd_register || gp_min == no_register) {
          success_ = false; return;
        }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (is_neg) {
          // SQNEG: res = 0 - xn.
          as_.Pxor(xres, xres);
          switch (args.size) {
            case 0b00: as_.Psubb(xres, xn); break;
            case 0b01: as_.Psubw(xres, xn); break;
            default:   as_.Psubd(xres, xn); break;
          }
        } else {
          // SQABS: sign_mask = PCMPGT(0, xn); res = (xn ^ sign_mask) - sign_mask.
          as_.Pxor(xtmp, xtmp);
          switch (args.size) {
            case 0b00: as_.Pcmpgtb(xtmp, xn); break;
            case 0b01: as_.Pcmpgtw(xtmp, xn); break;
            default:   as_.Pcmpgtd(xtmp, xn); break;
          }
          as_.Movdqa(xres, xn);
          as_.Pxor(xres, xtmp);
          switch (args.size) {
            case 0b00: as_.Psubb(xres, xtmp); break;
            case 0b01: as_.Psubw(xres, xtmp); break;
            default:   as_.Psubd(xres, xtmp); break;
          }
        }
        // Build broadcasted INT_MIN_w in xtmp; xtmp is dead at this point
        // (sign mask consumed for SQABS; never set for SQNEG).  A 32-bit
        // broadcast pattern (Movd + Pshufd 0x00) covers all three widths
        // because the INT_MIN bit pattern repeats every 32 bits at
        // byte/half/word granularity.
        int32_t int_min_bcast;
        switch (args.size) {
          case 0b00: int_min_bcast = static_cast<int32_t>(0x80808080u); break;
          case 0b01: int_min_bcast = static_cast<int32_t>(0x80008000u); break;
          default:   int_min_bcast = static_cast<int32_t>(0x80000000u); break;
        }
        as_.Movl(gp_min, int_min_bcast);
        as_.Movd(xtmp, gp_min);
        as_.Pshufd(xtmp, xtmp, int8_t{0x00});
        // Saturation mask: 1s in lanes where src == INT_MIN_w.
        switch (args.size) {
          case 0b00: as_.Pcmpeqb(xtmp, xn); break;
          case 0b01: as_.Pcmpeqw(xtmp, xn); break;
          default:   as_.Pcmpeqd(xtmp, xn); break;
        }
        // Apply: lanes where src==INT_MIN flip from 0x80..0 (INT_MIN) to
        // 0x7F..F (INT_MAX); other lanes XOR with 0 (no-op).
        as_.Pxor(xres, xtmp);
        if (!args.q) MaskLow64(xres);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xres);
        return;
      }
      // SADDLP / UADDLP Vd.<Tb>, Vn.<Ta> — add adjacent pairs of elements,
      // producing a result with elements twice as wide.  S = signed, U = unsigned.
      // SADALP / UADALP — same shape but accumulate into Vd.
      //   size=00 src .8B/.16B  -> dst .4H/.8H
      //   size=01 src .4H/.8H   -> dst .2S/.4S
      //   size=10 src .2S/.4S   -> dst .1D/.2D
      //   size=11 reserved (decoder routes UNDEF).
      //
      // Idioms per width:
      //   byte->half  unsigned: lo = (xn & 0x00FF); hi = (xn >> 8); PADDW.
      //   byte->half  signed:   lo = (xn << 8) >> 8 (arith); hi = (xn >> 8) arith; PADDW.
      //   half->word  unsigned: lo = (xn & 0x0000FFFF); hi = (xn >> 16); PADDD.
      //   half->word  signed:   PMADDWD(xn, [0x0001 per half]) — exact signed pair sum.
      //   word->dword unsigned: lo = (xn & 0xFFFFFFFF per qword); hi = (xn >> 32); PADDQ.
      //   word->dword signed:   sign-extend each dword to qword via PMOVSXDQ on lo/hi
      //                         halves, then pair-sum via PUNPCKLQDQ/PUNPCKHQDQ + PADDQ.
      // Accumulate forms (SADALP/UADALP) PADD into the existing Vd contents.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSaddlp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUaddlp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSadalp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUadalp: {
        if (args.size == 0b11) { Undefined(); return; }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSaddlp ||
             args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSadalp);
        const bool is_accum =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSadalp ||
             args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUadalp);
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xres = AllocTempSimdReg();
        SimdRegister xtmp = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        switch (args.size) {
          case 0b00: {  // byte -> half
            if (is_signed) {
              as_.Movdqa(xres, xn);
              as_.Psllw(xres, int8_t{8});
              as_.Psraw(xres, int8_t{8});  // low byte sign-extended to 16
              as_.Movdqa(xtmp, xn);
              as_.Psraw(xtmp, int8_t{8});  // high byte sign-extended to 16
              as_.Paddw(xres, xtmp);
            } else {
              as_.Pcmpeqw(xtmp, xtmp);
              as_.Psrlw(xtmp, int8_t{8});  // 0x00FF per half
              as_.Movdqa(xres, xn);
              as_.Pand(xres, xtmp);         // low byte zero-extended
              as_.Movdqa(xtmp, xn);
              as_.Psrlw(xtmp, int8_t{8});  // high byte zero-extended
              as_.Paddw(xres, xtmp);
            }
            break;
          }
          case 0b01: {  // half -> word
            if (is_signed) {
              as_.Pcmpeqw(xtmp, xtmp);
              as_.Psrlw(xtmp, int8_t{15});  // 0x0001 per half
              as_.Movdqa(xres, xn);
              as_.Pmaddwd(xres, xtmp);       // signed pair sum -> 32-bit per dword
            } else {
              as_.Pcmpeqd(xtmp, xtmp);
              as_.Psrld(xtmp, int8_t{16});  // 0x0000FFFF per dword
              as_.Movdqa(xres, xn);
              as_.Pand(xres, xtmp);
              as_.Movdqa(xtmp, xn);
              as_.Psrld(xtmp, int8_t{16});
              as_.Paddd(xres, xtmp);
            }
            break;
          }
          case 0b10: {  // word -> dword
            if (is_signed) {
              SimdRegister xhi = AllocTempSimdReg();
              if (!success()) { return; }
              as_.Pmovsxdq(xres, xn);            // [sx(xn[0]), sx(xn[1])]
              as_.Movdqa(xhi, xn);
              as_.Psrldq(xhi, int8_t{8});         // [xn[2], xn[3], 0, 0]
              as_.Pmovsxdq(xhi, xhi);             // [sx(xn[2]), sx(xn[3])]
              as_.Movdqa(xtmp, xres);
              as_.Punpcklqdq(xtmp, xhi);          // [sx(0), sx(2)]
              as_.Punpckhqdq(xres, xhi);          // [sx(1), sx(3)]
              as_.Paddq(xres, xtmp);              // [sx(0)+sx(1), sx(2)+sx(3)]
            } else {
              as_.Pcmpeqd(xtmp, xtmp);
              as_.Psrlq(xtmp, int8_t{32});  // 0x00000000_FFFFFFFF per qword
              as_.Movdqa(xres, xn);
              as_.Pand(xres, xtmp);          // low 32 of each qword, zero-extended
              as_.Movdqa(xtmp, xn);
              as_.Psrlq(xtmp, int8_t{32});  // high 32 of each qword, zero-extended
              as_.Paddq(xres, xtmp);
            }
            break;
          }
        }
        if (is_accum) {
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
          switch (args.size) {
            case 0b00: as_.Paddw(xres, xd); break;
            case 0b01: as_.Paddd(xres, xd); break;
            default:   as_.Paddq(xres, xd); break;
          }
        }
        if (!args.q) MaskLow64(xres);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xres);
        return;
      }
      // REV64 Vd.<T>, Vn.<T> — reverse element order within each 64-bit lane.
      // size=00: byte reverse (8B / 16B) — BSWAPQ on each 64-bit half.
      // size=01: halfword reverse (4H / 8H) — PSHUFLW + PSHUFHW imm=0x1B.
      // size=10: word reverse (2S / 4S) — PSHUFD imm=0x01 (Q=0) or 0xB1 (Q=1).
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev64: {
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        switch (args.size) {
          case 0b00: {
            Register r1 = AllocTempReg();
            if (r1 == no_register) { Undefined(); return; }
            as_.Movq(r1, xn);
            as_.Bswapq(r1);
            if (args.q) {
              Register r2 = AllocTempReg();
              if (r2 == no_register) { Undefined(); return; }
              as_.Pextrq(r2, xn, int8_t{1});
              as_.Bswapq(r2);
              as_.Movq(xn, r1);
              as_.Pinsrq(xn, r2, int8_t{1});
            } else {
              as_.Movq(xn, r1);  // zeros upper 64 bits
            }
            break;
          }
          case 0b01:
            as_.Pshuflw(xn, xn, int8_t{0x1B});
            if (args.q) {
              as_.Pshufhw(xn, xn, int8_t{0x1B});
            } else {
              MaskLow64(xn);
            }
            break;
          case 0b10:
            if (args.q) {
              as_.Pshufd(xn, xn, static_cast<int8_t>(0xB1));
            } else {
              as_.Pshufd(xn, xn, static_cast<int8_t>(0x01));
              MaskLow64(xn);
            }
            break;
          default: Undefined(); return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // REV32 Vd.<T>, Vn.<T> — reverse element order within each 32-bit lane.
      // size=00: byte reverse (8B / 16B) — BSWAPL per 32-bit lane.
      // size=01: halfword reverse (4H / 8H) — PSHUFLW/HW imm=0xB1 (swap pairs).
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev32: {
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        switch (args.size) {
          case 0b00: {
            SimdRegister xd = AllocTempSimdReg();
            Register r1 = AllocTempReg();
            if (xd == no_simd_register || r1 == no_register) { Undefined(); return; }
            as_.Pxor(xd, xd);
            int lanes = args.q ? 4 : 2;
            for (int i = 0; i < lanes; ++i) {
              as_.Pextrd(r1, xn, static_cast<int8_t>(i));
              as_.Bswapl(r1);
              as_.Pinsrd(xd, r1, static_cast<int8_t>(i));
            }
            as_.Movdqa(xn, xd);
            break;
          }
          case 0b01:
            as_.Pshuflw(xn, xn, static_cast<int8_t>(0xB1));
            if (args.q) {
              as_.Pshufhw(xn, xn, static_cast<int8_t>(0xB1));
            } else {
              MaskLow64(xn);
            }
            break;
          default: Undefined(); return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // REV16 Vd.<T>, Vn.<T> — reverse byte order within each 16-bit lane.
      // size=00 only (8B / 16B). Compute: (Vn << 8) | (Vn >> 8) per halfword.
      // PSLLW shifts each 16-bit lane left by 8 (high byte was lost, low byte
      // moves to high). PSRLW shifts right by 8 in the saved copy (low byte
      // was lost, high byte moves to low). OR'ing combines the byte-swapped
      // result without needing a PSHUFB mask table.
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev16: {
        if (args.size != 0b00) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xt = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Movdqa(xt, xn);
        as_.Psllw(xn, int8_t{8});
        as_.Psrlw(xt, int8_t{8});
        as_.Por(xn, xt);
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // ABS V.<T>, V.<T> -- per-lane integer absolute value.
      //   size=00 .8B/.16B  -> mask = PCMPGTB(0, Vn); Vd = PSUBB(PXOR(Vn, mask), mask)
      //   size=01 .4H/.8H   -> same with PCMPGTW/PSUBW
      //   size=10 .2S/.4S   -> same with PCMPGTD/PSUBD
      //   size=11 .2D       -> needs PCMPGTQ (SSE4.2); JIT-bail to interp via success_=false.
      // Identity: (Vn ^ sign_mask) - sign_mask  where sign_mask = -1 if Vn<0 else 0.
      //   Vn>=0: (Vn ^ 0) - 0 = Vn
      //   Vn<0 : (Vn ^ -1) - (-1) = (~Vn) + 1 = -Vn
      // Avoids depending on SSSE3 PABSB/W/D (not currently in the assembler defs).
      // INT_MIN stays INT_MIN (matches ARM ABS semantics for the most-negative lane).
      case Decoder::AdvSimdTwoRegMiscOpcode::kAbs: {
        // .2D (size=0b11) uses PCMPGTQ for the sign mask and is gated on SSE4.2.
        if (args.size == 0b11 && !host_platform::kHasSSE4_2) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister mask = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(mask, mask);
        switch (args.size) {
          case 0b00: as_.Pcmpgtb(mask, xn); break;
          case 0b01: as_.Pcmpgtw(mask, xn); break;
          case 0b10: as_.Pcmpgtd(mask, xn); break;
          default: as_.Pcmpgtq(mask, xn); break;  // 0b11 (.2D)
        }
        as_.Pxor(xn, mask);
        switch (args.size) {
          case 0b00: as_.Psubb(xn, mask); break;
          case 0b01: as_.Psubw(xn, mask); break;
          case 0b10: as_.Psubd(xn, mask); break;
          default: as_.Psubq(xn, mask); break;  // 0b11 (.2D)
        }
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // NEG V.<T>, V.<T> -- per-lane integer negation: Vd = 0 - Vn.
      //   size=00 -> PSUBB; size=01 -> PSUBW; size=10 -> PSUBD; size=11 -> PSUBQ.
      // All four widths are SSE2-direct. size=11 with Q=0 (.1D) is reserved and
      // bails to Undefined to mirror the decoder's general size/Q reservation
      // pattern; the interpreter never sees that encoding either.
      case Decoder::AdvSimdTwoRegMiscOpcode::kNeg: {
        if (args.size == 0b11 && !args.q) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xz = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xz, xz);
        switch (args.size) {
          case 0b00: as_.Psubb(xz, xn); break;
          case 0b01: as_.Psubw(xz, xn); break;
          case 0b10: as_.Psubd(xz, xn); break;
          case 0b11: as_.Psubq(xz, xn); break;
          default: success_ = false; return;
        }
        if (!args.q) MaskLow64(xz);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xz);
        return;
      }
      // CNT V.16B / V.8B -- per-byte population count (set-bit count).
      // Lowering: Mula-Wojcik nibble-LUT via PSHUFB.
      //   table = {0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4}  (popcount per nibble)
      //   low_nibbles  = Vn & 0x0F
      //   high_nibbles = (Vn >> 4) & 0x0F        (PSRLW shifts by word; mask
      //                                           keeps just the nibble we want)
      //   Vd = PSHUFB(table, low_nibbles) + PSHUFB(table, high_nibbles)
      // CNT has only one encoded size (size=00 → 8B/16B); any other size is
      // unallocated and never reaches this case via the decoder, but bail
      // defensively just in case.
      case Decoder::AdvSimdTwoRegMiscOpcode::kCnt: {
        if (args.size != 0b00) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xt_hi = AllocTempSimdReg();
        SimdRegister xtable_lo = AllocTempSimdReg();
        SimdRegister xtable_hi = AllocTempSimdReg();
        SimdRegister xmask = AllocTempSimdReg();
        if (!success()) { return; }
        Register r1 = AllocTempReg();
        if (r1 == no_register) { success_ = false; return; }
        // Build the popcount nibble table:
        //   bytes  0..7 = {0,1,1,2,1,2,2,3} → 0x0302020102010100 (little-endian)
        //   bytes  8..15= {1,2,2,3,2,3,3,4} → 0x0403030203020201
        as_.Movq(r1, static_cast<int64_t>(0x0302020102010100LL));
        as_.Movq(xtable_lo, r1);
        as_.Movq(r1, static_cast<int64_t>(0x0403030203020201LL));
        as_.Pinsrq(xtable_lo, r1, int8_t{1});
        // We need the table twice (PSHUFB is destructive); keep a second copy.
        as_.Movdqa(xtable_hi, xtable_lo);
        // Build the 0x0F low-nibble mask, broadcast to all 16 bytes.
        as_.Movq(r1, static_cast<int64_t>(0x0F0F0F0F0F0F0F0FLL));
        as_.Movq(xmask, r1);
        as_.Punpcklqdq(xmask, xmask);
        // Load Vn; split into low and high nibbles.
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Movdqa(xt_hi, xn);
        as_.Psrlw(xt_hi, int8_t{4});   // shift 16-bit lanes right by 4
        as_.Pand(xt_hi, xmask);        // keep low nibble of each byte
        as_.Pand(xn, xmask);
        // PSHUFB(dst, src): for each byte i, dst[i] = dst[src[i] & 0x0F] (if
        // src[i] high bit clear). After PAND with 0x0F the high bits are 0, so
        // the table is indexed by the nibble value directly.
        as_.Pshufb(xtable_lo, xn);
        as_.Pshufb(xtable_hi, xt_hi);
        as_.Paddb(xtable_lo, xtable_hi);
        if (!args.q) MaskLow64(xtable_lo);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xtable_lo);
        return;
      }
      // NOT V.16B / V.8B (size=00) -- per-lane bitwise NOT: Vd = ~Vn.
      // RBIT V.16B / V.8B (size=01) -- per-byte bit reversal.
      // The decoder routes opcode=00101+U=1 to kNot for both encodings,
      // discriminated by size: size=00 → NOT, size=01 → RBIT.
      //
      // RBIT lowering: two-PSHUFB nibble-LUT.  For input byte b = (H<<4) | L:
      //   reverse_bits(b) = (reverse_4(L) << 4) | reverse_4(H)
      // where reverse_4 is the 4-bit bit-reverse.  Two lookup tables:
      //   low_table[i]  = reverse_4(i)              -- lookup result occupies low nibble
      //   high_table[i] = reverse_4(i) << 4         -- lookup result occupies high nibble
      // Then:
      //   t_lo = Vn & 0x0F       (low nibbles)
      //   t_hi = (Vn >> 4) & 0x0F (high nibbles, via 16-bit PSRLW + PAND)
      //   Vd   = PSHUFB(high_table, t_lo) | PSHUFB(low_table, t_hi)
      // The two PSHUFB results have disjoint nibble positions, so POR or
      // PADDB combine equivalently; we use POR to match the read-as-bits
      // shape of the operation.
      case Decoder::AdvSimdTwoRegMiscOpcode::kNot: {
        if (args.size == 0b00) {
          // NOT: trivial bitwise complement.
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister allones = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Pcmpeqd(allones, allones);
          as_.Pxor(xn, allones);
          if (!args.q) MaskLow64(xn);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        if (args.size == 0b01) {
          // RBIT: two-PSHUFB nibble-LUT bit reversal.
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xn_hi = AllocTempSimdReg();
          SimdRegister xlow_table = AllocTempSimdReg();
          SimdRegister xhigh_table = AllocTempSimdReg();
          SimdRegister xmask = AllocTempSimdReg();
          if (!success()) { return; }
          Register r1 = AllocTempReg();
          if (r1 == no_register) { success_ = false; return; }
          // Build low_table = {0,8,4,12,2,10,6,14, 1,9,5,13,3,11,7,15}.
          //   bytes  0..7 → 0x0E060A020C040800
          //   bytes 8..15 → 0x0F070B030D050901
          as_.Movq(r1, static_cast<int64_t>(0x0E060A020C040800LL));
          as_.Movq(xlow_table, r1);
          as_.Movq(r1, static_cast<int64_t>(0x0F070B030D050901LL));
          as_.Pinsrq(xlow_table, r1, int8_t{1});
          // Build high_table = {0,0x80,0x40,0xC0,0x20,0xA0,0x60,0xE0,
          //                     0x10,0x90,0x50,0xD0,0x30,0xB0,0x70,0xF0}.
          //   bytes  0..7 → 0xE060A020C0408000
          //   bytes 8..15 → 0xF070B030D0509010
          as_.Movq(r1, static_cast<int64_t>(0xE060A020C0408000ULL));
          as_.Movq(xhigh_table, r1);
          as_.Movq(r1, static_cast<int64_t>(0xF070B030D0509010ULL));
          as_.Pinsrq(xhigh_table, r1, int8_t{1});
          // Build 0x0F broadcast mask.
          as_.Movq(r1, static_cast<int64_t>(0x0F0F0F0F0F0F0F0FLL));
          as_.Movq(xmask, r1);
          as_.Punpcklqdq(xmask, xmask);
          // Load Vn; split into low and high nibbles.
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Movdqa(xn_hi, xn);
          as_.Psrlw(xn_hi, int8_t{4});  // 16-bit-lane shift; bits leak between
                                        // adjacent bytes but PAND below filters
                                        // each byte's high nibble cleanly.
          as_.Pand(xn_hi, xmask);
          as_.Pand(xn, xmask);
          // Dual PSHUFB: high_table looked up by low nibbles → high-nibble of
          // result; low_table looked up by high nibbles → low-nibble of result.
          // PSHUFB is destructive in its first operand; tables consumed here.
          as_.Pshufb(xhigh_table, xn);
          as_.Pshufb(xlow_table, xn_hi);
          as_.Por(xhigh_table, xlow_table);
          if (!args.q) MaskLow64(xhigh_table);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xhigh_table);
          return;
        }
        // size=10 / size=11 are reserved for this opcode group.
        success_ = false;
        return;
      }

      // SUQADD / USQADD (per-lane saturating accumulate of
      // mixed signedness) for byte/halfword lanes.
      //   SUQADD Vd, Vn: Vd[i] = SignedSat( int(Vd[i]) + uint(Vn[i]) )
      //   USQADD Vd, Vn: Vd[i] = UnsignedSat( uint(Vd[i]) + int(Vn[i]) )
      // x86 has no mixed-sign saturating add, but the per-lane sum of an N-bit
      // signed and an N-bit unsigned value always fits in 2N bits, so widen
      // both operands (Vd and Vn each with their own signedness), add in the
      // wider lane, then narrow back with the saturating pack that matches the
      // destination's signedness: SUQADD -> signed dst -> PACKSS*, USQADD ->
      // unsigned dst -> PACKUS*.  PACKUS* already clamps negatives to 0 and
      // over-range to max, which is exactly UnsignedSat.  Q=0 packs the low
      // half against zero (upper 64 zeroed); Q=1 packs low+high halves.
      // .2S/.4S (size=10) is handled by the per-lane scalar path below (32-bit
      // sums fit in a 64-bit GP register, so a simple clamp suffices).  .1D/.2D
      // (size=11) would need 65-bit saturation logic and bails to the
      // interpreter.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSuqadd:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUsqadd: {
        const bool usqadd =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUsqadd);
        if (args.size == 0b10) {
          // .2S/.4S: per 32-bit lane, compute the mixed-sign sum in a 64-bit GP
          // register (no overflow) and clamp to the destination range.
          //   SUQADD: sat_s32( sext(Vd) + zext(Vn) )
          //   USQADD: sat_u32( zext(Vd) + sext(Vn) )
          SimdRegister xres = AllocTempSimdReg();
          Register acc = AllocTempReg();
          Register addend = AllocTempReg();
          Register lim = AllocTempReg();
          if (xres == no_simd_register || acc == no_register ||
              addend == no_register || lim == no_register) {
            success_ = false; return;
          }
          as_.Pxor(xres, xres);
          const int lanes = args.q ? 4 : 2;
          for (int i = 0; i < lanes; ++i) {
            const int32_t d_off = vd_off + i * 4;
            const int32_t n_off = vn_off + i * 4;
            if (usqadd) {
              as_.Movl(acc, {.base = Assembler::rbp, .disp = d_off});      // zext Vd
              as_.Movsxlq(addend, {.base = Assembler::rbp, .disp = n_off});  // sext Vn
            } else {
              as_.Movsxlq(acc, {.base = Assembler::rbp, .disp = d_off});   // sext Vd
              as_.Movl(addend, {.base = Assembler::rbp, .disp = n_off});   // zext Vn
            }
            as_.Addq(acc, addend);
            if (usqadd) {
              // clamp to [0, UINT32_MAX]
              auto* ge0 = as_.MakeLabel();
              as_.Testq(acc, acc);
              as_.Jcc(Condition::kGreaterEqual, *ge0);
              as_.Xorl(acc, acc);
              as_.Bind(ge0);
              auto* le_max = as_.MakeLabel();
              as_.Movq(lim, int64_t{0xFFFFFFFFLL});
              as_.Cmpq(acc, lim);
              as_.Jcc(Condition::kLessEqual, *le_max);
              as_.Movq(acc, int64_t{0xFFFFFFFFLL});
              as_.Bind(le_max);
            } else {
              // clamp to [INT32_MIN, INT32_MAX]
              auto* le_max = as_.MakeLabel();
              as_.Movq(lim, int64_t{0x7FFFFFFFLL});
              as_.Cmpq(acc, lim);
              as_.Jcc(Condition::kLessEqual, *le_max);
              as_.Movq(acc, int64_t{0x7FFFFFFFLL});
              as_.Bind(le_max);
              auto* ge_min = as_.MakeLabel();
              as_.Movq(lim, static_cast<int64_t>(0xFFFFFFFF80000000ULL));  // INT32_MIN
              as_.Cmpq(acc, lim);
              as_.Jcc(Condition::kGreaterEqual, *ge_min);
              as_.Movq(acc, static_cast<int64_t>(0xFFFFFFFF80000000ULL));
              as_.Bind(ge_min);
            }
            as_.Pinsrd(xres, acc, static_cast<int8_t>(i));
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xres);
          return;
        }
        if (args.size != 0b00 && args.size != 0b01) { success_ = false; return; }
        SimdRegister xd = AllocTempSimdReg();
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister dlo = AllocTempSimdReg();
        SimdRegister nlo = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        // Widen the low 8 bytes of `src` into `dst` with the given signedness.
        auto widen_lo = [&](SimdRegister dst, bool is_signed) {
          if (args.size == 0b00) {
            if (is_signed) as_.Pmovsxbw(dst, dst); else as_.Pmovzxbw(dst, dst);
          } else {
            if (is_signed) as_.Pmovsxwd(dst, dst); else as_.Pmovzxwd(dst, dst);
          }
        };
        auto add_wide = [&](SimdRegister a, SimdRegister b) {
          if (args.size == 0b00) as_.Paddw(a, b); else as_.Paddd(a, b);
        };
        // Vd signedness is the destination flavour (SUQADD signed / USQADD
        // unsigned); Vn carries the opposite signedness.
        as_.Movdqa(dlo, xd);
        as_.Movdqa(nlo, xn);
        widen_lo(dlo, /*is_signed=*/!usqadd);
        widen_lo(nlo, /*is_signed=*/usqadd);
        add_wide(dlo, nlo);  // dlo = widened lane sums (low half)
        SimdRegister hi = AllocTempSimdReg();
        if (!success()) { return; }
        if (args.q) {
          SimdRegister nhi = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Psrldq(xd, int8_t{8});  // bring high 8 bytes into the low lanes
          as_.Psrldq(xn, int8_t{8});
          as_.Movdqa(hi, xd);
          as_.Movdqa(nhi, xn);
          widen_lo(hi, /*is_signed=*/!usqadd);
          widen_lo(nhi, /*is_signed=*/usqadd);
          add_wide(hi, nhi);  // hi = widened lane sums (high half)
        } else {
          as_.Pxor(hi, hi);
        }
        // Narrow with the destination-signedness saturating pack.
        if (args.size == 0b00) {
          if (usqadd) as_.Packuswb(dlo, hi); else as_.Packsswb(dlo, hi);
        } else {
          if (usqadd) as_.Packusdw(dlo, hi); else as_.Packssdw(dlo, hi);
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, dlo);
        return;
      }

      // CLZ V.<T>, V.<T> -- per-lane count leading zeros.
      //   size=00 .8B/.16B  -> 8-bit lane CLZ (result 0..8)
      //   size=01 .4H/.8H   -> 16-bit lane CLZ (result 0..16)
      //   size=10 .2S/.4S   -> 32-bit lane CLZ (result 0..32)
      //   size=11           -> reserved per ARM ARM; decoder never routes here,
      //                        but bail defensively via success_=false.
      // Per-lane lowering: PEXTR{B,W,D} → scalar CLZ via BSR (with zero-input
      // branch to lane_bits) → XOR with lane_bits-1 → PINSR{B,W,D}.  The
      // BSR+XOR trick matches the scalar CLZ lowering already used in this
      // file at the integer scalar dispatch (BSR returns the MSB position
      // 0..lane_bits-1; XOR with lane_bits-1 (= 2^k - 1, all-ones in low k
      // bits) flips to (lane_bits - 1) - BSR with no borrow).  The zero-input
      // case takes the forward branch and stores lane_bits unchanged.
      //
      // BSR is undefined on zero on Intel CPUs (AMD sets dst to operand size,
      // but we don't rely on that), so the TEST+JZ guard is necessary.
      //
      // Per-lane round-trip is acceptable for CLZ — rarely on the hot path.
      // An AVX-512-CD VPLZCNTD/Q gate is possible (size=10 only) but not
      // implemented here; the lowering is uniform across sizes and the
      // BSR-based scalar path is portable to any x86_64 host.
      case Decoder::AdvSimdTwoRegMiscOpcode::kClz: {
        if (args.size == 0b11) { success_ = false; return; }
        const uint8_t lane_bits = static_cast<uint8_t>(8u << args.size);
        const uint8_t bytes_per_lane = static_cast<uint8_t>(1u << args.size);
        const int lanes_per_vec = (args.q ? 16 : 8) / bytes_per_lane;
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        Register r1 = AllocTempReg();
        if (r1 == no_register) { success_ = false; return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xd, xd);
        for (int i = 0; i < lanes_per_vec; ++i) {
          switch (args.size) {
            case 0b00: as_.Pextrb(r1, xn, static_cast<int8_t>(i)); break;
            case 0b01: as_.Pextrw(r1, xn, static_cast<int8_t>(i)); break;
            case 0b10: as_.Pextrd(r1, xn, static_cast<int8_t>(i)); break;
          }
          auto* nonzero = as_.MakeLabel();
          auto* done = as_.MakeLabel();
          as_.Testl(r1, r1);
          as_.Jcc(Condition::kNotZero, *nonzero);
          as_.Movl(r1, static_cast<int32_t>(lane_bits));
          as_.Jmp(*done);
          as_.Bind(nonzero);
          as_.Bsrl(r1, r1);
          as_.Xorl(r1, static_cast<int8_t>(lane_bits - 1));
          as_.Bind(done);
          switch (args.size) {
            case 0b00: as_.Pinsrb(xd, r1, static_cast<int8_t>(i)); break;
            case 0b01: as_.Pinsrw(xd, r1, static_cast<int8_t>(i)); break;
            case 0b10: as_.Pinsrd(xd, r1, static_cast<int8_t>(i)); break;
          }
        }
        if (!args.q) MaskLow64(xd);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
        return;
      }
      // CLS V.<T>, V.<T> -- per-lane count leading sign bits (number of
      // consecutive bits below the MSB that equal the MSB; range 0..N-1).
      //   size=00 .8B/.16B  -> 8-bit lane CLS (result 0..7)
      //   size=01 .4H/.8H   -> 16-bit lane CLS (result 0..15)
      //   size=10 .2S/.4S   -> 32-bit lane CLS (result 0..31)
      //   size=11           -> reserved per ARM ARM.
      // Identity used: y = (x < 0) ? ~x : x (per lane); then
      //   CLS(x) = CLZ_N(y) - 1, where CLZ_N is N-bit count-leading-zeros.
      // Per-lane y is built once via a vector XOR with the per-lane sign
      // mask (PCMPGT{B,W,D}(0, xn) yields all-ones in negative lanes, 0
      // elsewhere — exactly arithmetic-shift-right-by-(N-1) per lane),
      // then scalarized through PEXTR{B,W,D} → BSR → XOR with (N-1) → SUB 1
      // (the BSR+XOR pair gives CLZ_N; the trailing -1 gives CLS).  The
      // y==0 branch substitutes (N-1) directly — that's CLS of an all-zero
      // or all-one lane (e.g. 0x00 or 0xFF), where every bit after the MSB
      // matches.
      case Decoder::AdvSimdTwoRegMiscOpcode::kCls: {
        if (args.size == 0b11) { success_ = false; return; }
        const uint8_t lane_bits = static_cast<uint8_t>(8u << args.size);
        const uint8_t bytes_per_lane = static_cast<uint8_t>(1u << args.size);
        const int lanes_per_vec = (args.q ? 16 : 8) / bytes_per_lane;
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        SimdRegister xsign = AllocTempSimdReg();
        if (!success()) { return; }
        Register r1 = AllocTempReg();
        if (r1 == no_register) { success_ = false; return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xsign, xsign);
        switch (args.size) {
          case 0b00: as_.Pcmpgtb(xsign, xn); break;
          case 0b01: as_.Pcmpgtw(xsign, xn); break;
          case 0b10: as_.Pcmpgtd(xsign, xn); break;
        }
        // xsign now has the per-lane sign-extension mask (all-ones in
        // negative lanes, zero elsewhere).  XOR collapses negative lanes to
        // (~x) while leaving positive lanes untouched — that's the
        // CLS-via-CLZ preprocess.
        as_.Pxor(xn, xsign);
        as_.Pxor(xd, xd);
        for (int i = 0; i < lanes_per_vec; ++i) {
          switch (args.size) {
            case 0b00: as_.Pextrb(r1, xn, static_cast<int8_t>(i)); break;
            case 0b01: as_.Pextrw(r1, xn, static_cast<int8_t>(i)); break;
            case 0b10: as_.Pextrd(r1, xn, static_cast<int8_t>(i)); break;
          }
          auto* nonzero = as_.MakeLabel();
          auto* done = as_.MakeLabel();
          as_.Testl(r1, r1);
          as_.Jcc(Condition::kNotZero, *nonzero);
          // y == 0  =>  CLS = lane_bits - 1.  (All bits below MSB match.)
          as_.Movl(r1, static_cast<int32_t>(lane_bits - 1));
          as_.Jmp(*done);
          as_.Bind(nonzero);
          // CLZ_N(y) = (lane_bits - 1) XOR BSR(y).  CLS = CLZ_N(y) - 1.
          as_.Bsrl(r1, r1);
          // CLS = CLZ - 1 = (lane_bits - 1) XOR BSR  XOR'd with (lane_bits - 2).
          // We collapse to a single XOR by noting that for a non-zero
          // bsr_value in [0..lane_bits-1], CLZ = (lane_bits - 1) - bsr_value,
          // hence CLS = (lane_bits - 2) - bsr_value.  But the host doesn't
          // have a "XOR with constant minus 1" — keep it as XOR + DECL.
          as_.Xorl(r1, static_cast<int8_t>(lane_bits - 1));
          as_.Decl(r1);
          as_.Bind(done);
          switch (args.size) {
            case 0b00: as_.Pinsrb(xd, r1, static_cast<int8_t>(i)); break;
            case 0b01: as_.Pinsrw(xd, r1, static_cast<int8_t>(i)); break;
            case 0b10: as_.Pinsrd(xd, r1, static_cast<int8_t>(i)); break;
          }
        }
        if (!args.q) MaskLow64(xd);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
        return;
      }
      // Vector FABS / FNEG (FP32 .2S/.4S, FP64 .2D, FP16 .4H/.8H).
      //   size=10 → FP32, size=11 → FP64.  FP64 requires Q=1.
      //   FABS: AND with broadcast mask 0x7FFFFFFF (FP32) or 0x7FFFFFFF_FFFFFFFF (FP64).
      //   FNEG: XOR with broadcast mask 0x80000000 (FP32) or 0x80000000_00000000 (FP64).
      //   FP16: broadcast 16-bit mask 0x7FFF (FABS) / 0x8000 (FNEG).  FP16
      //   FABS/FNEG are pure bit operations — no F16C round-trip required.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFabs:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFneg: {
        const bool is_fabs =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFabs);
        if (args.is_fp16) {
          // .4H (Q=0) and .8H (Q=1) — broadcast 16-bit mask then PAND/PXOR.
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister mask = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Pcmpeqd(mask, mask);
          if (is_fabs) {
            as_.Psrlw(mask, int8_t{1});   // each 16-bit lane = 0x7FFF
            as_.Pand(xn, mask);
          } else {
            as_.Psllw(mask, int8_t{15});  // each 16-bit lane = 0x8000
            as_.Pxor(xn, mask);
          }
          if (!args.q) MaskLow64(xn);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        if (args.size != 0b10 && args.size != 0b11) { Undefined(); return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister mask = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pcmpeqd(mask, mask);
        if (is_double) {
          // FABS mask 0x7FFFFFFFFFFFFFFF (allones >> 1); FNEG mask 0x8000000000000000.
          if (is_fabs) {
            as_.Psrlq(mask, int8_t{1});
            as_.Pand(xn, mask);
          } else {
            as_.Psllq(mask, int8_t{63});
            as_.Pxor(xn, mask);
          }
        } else {
          if (is_fabs) {
            as_.Psrld(mask, int8_t{1});
            as_.Pand(xn, mask);
          } else {
            as_.Pslld(mask, int8_t{31});
            as_.Pxor(xn, mask);
          }
        }
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // Vector FRINTN / FRINTM / FRINTP / FRINTZ / FRINTX / FRINTI
      // (FP32 .2S/.4S, FP64 .2D).
      //   FRINTN  -> ROUNDPS/PD imm=0 (round to nearest even).
      //   FRINTM  -> ROUNDPS/PD imm=1 (toward -inf, floor).
      //   FRINTP  -> ROUNDPS/PD imm=2 (toward +inf, ceil).
      //   FRINTZ  -> ROUNDPS/PD imm=3 (toward zero, trunc).
      //   FRINTX/FRINTI -> ROUNDPS/PD imm=4 (use MXCSR; default RNE matches ARM).
      // SSE4.1 ROUNDPS/PD natively accepts the imm; one instruction per vector.
      // The FP16 form (args.is_fp16) bails to the interpreter.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintxV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintiV: {
        int8_t round_imm;
        switch (args.opcode) {
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV: round_imm = 0x00; break;
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV: round_imm = 0x01; break;
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV: round_imm = 0x02; break;
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV: round_imm = 0x03; break;
          // FRINTX / FRINTI follow the current FPCR rounding mode; we treat
          // MXCSR (default RNE) as the canonical mode.  imm=0x04 sets the
          // ROUND* "use MXCSR" bit.
          default: round_imm = 0x04; break;
        }
        if (args.is_fp16) {
          // .4H (Q=0) and .8H (Q=1) — F16C round-trip with ROUNDPS imm.
          // Per standing rule: F16C round-trip is exact for
          // FP16 unary FRINT*. Pattern identical to FSQRT FP16 but with
          // ROUNDPS instead of SQRTPS.
          if (!host_platform::kHasF16C) { success_ = false; return; }
          SimdRegister xlo = AllocTempSimdReg();
          if (!success()) { return; }
          if (!args.q) {
            as_.Movq(xlo, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xlo, xlo);
            as_.Roundps(xlo, xlo, round_imm);
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            // Vcvtps2ph already zeroes the upper 64 bits of xlo.
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          } else {
            SimdRegister xhi = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqu(xhi, {.base = Assembler::rbp, .disp = vn_off});
            as_.Movdqa(xlo, xhi);
            as_.Vcvtph2ps(xlo, xlo);
            as_.Psrldq(xhi, int8_t{8});
            as_.Vcvtph2ps(xhi, xhi);
            as_.Roundps(xlo, xlo, round_imm);
            as_.Roundps(xhi, xhi, round_imm);
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            as_.Vcvtps2ph(xhi, xhi, int8_t{0});
            as_.Pslldq(xhi, int8_t{8});
            as_.Por(xlo, xhi);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          }
          return;
        }
        // args.size for the FP variant of two-reg misc carries
        // (bit23=a, bit22=sz).  For the FRINTN/M/P/Z/X/I family the
        // decoder dispatches both bit23 values (decoder.h lines
        // 4397-4414): FRINTN/M/X come from bit23=0 (args.size 0b00/0b01),
        // FRINTP/Z/I from bit23=1 (args.size 0b10/0b11).  All four
        // values are valid here; only bit22 (the low bit) selects
        // FP32 vs FP64.  Mirrors the interpreter's `args.size & 1`
        // dispatch in interpreter.h.
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (is_double) {
          as_.Roundpd(xn, xn, round_imm);
        } else {
          as_.Roundps(xn, xn, round_imm);
        }
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // Vector FRINTA (FP32 .2S/.4S, FP64 .2D, FP16 .4H/.8H) -- "round to
      // nearest, ties away".
      // x86 ROUNDPS/PD has no ties-away mode, so use the identity
      //     FRINTA(x) = trunc(x + copysign(0.5, x))
      // already used by the scalar FRINTA JIT path.  Per-lane:
      // build (x AND sign_mask) OR half_pattern  ->  +/-0.5, add to x, ROUND
      // imm=3.  Bit-exact against ARM for all finite/NaN/Inf inputs (the
      // +0.5 nudge is a no-op when |x| >= 2^p; signed zeros preserved by
      // ROUNDPS imm=3; NaN/Inf propagate through ADDPS/ADDPD).
      // FP16 .4H/.8H runs the same trick in FP32 space via the F16C
      // round-trip (Vcvtph2ps -> ADDPS + ROUNDPS imm=3 -> Vcvtps2ph
      // imm=0).  Bit-exact for FP16 because: the widen-add-trunc-narrow
      // sequence preserves ARM's RNA result at every FP16 input (the
      // +0.5 nudge is exact in FP32; trunc gives an integer that, when
      // |x| <= 2^11, is representable in FP16 directly; when |x| > 2^11,
      // x itself is already integer in FP16 and trunc(x + sign*0.5) == x).
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintaV: {
        if (args.is_fp16) {
          if (!host_platform::kHasF16C) { success_ = false; return; }
          SimdRegister xlo = AllocTempSimdReg();
          SimdRegister smask = AllocTempSimdReg();
          SimdRegister half = AllocTempSimdReg();
          SimdRegister tmp = AllocTempSimdReg();
          if (!success()) { return; }
          Register gp_half = AllocTempReg();
          if (gp_half == no_register) { Undefined(); return; }
          // Build FP32 sign-bit broadcast mask.
          as_.Pcmpeqd(smask, smask);
          as_.Pslld(smask, int8_t{31});
          // Build 0.5 (FP32) broadcast across all 4 dwords.
          as_.Movl(gp_half, int32_t{0x3F000000});
          as_.Movd(half, gp_half);
          as_.Pshufd(half, half, static_cast<int8_t>(0x00));
          if (!args.q) {
            // .4H: 4 FP16 lanes in low 64 bits of Vn.
            as_.Movq(xlo, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xlo, xlo);
            as_.Movdqa(tmp, xlo);
            as_.Pand(tmp, smask);   // tmp = sign(xlo) in FP32 sign-bit position
            as_.Por(tmp, half);     // tmp = copysign(0.5, xlo)
            as_.Addps(xlo, tmp);
            as_.Roundps(xlo, xlo, int8_t{0x03});
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            // Vcvtps2ph zeroes the upper 64 bits of xlo.
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          } else {
            SimdRegister xhi = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqu(xhi, {.base = Assembler::rbp, .disp = vn_off});
            // Low half: widen lanes 0-3 into xlo.
            as_.Vcvtph2ps(xlo, xhi);
            // High half: shift xhi right 8 bytes, then widen.
            as_.Psrldq(xhi, int8_t{8});
            as_.Vcvtph2ps(xhi, xhi);
            // FRINTA dance on low half; narrow immediately.
            as_.Movdqa(tmp, xlo);
            as_.Pand(tmp, smask);
            as_.Por(tmp, half);
            as_.Addps(xlo, tmp);
            as_.Roundps(xlo, xlo, int8_t{0x03});
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            // FRINTA dance on high half; narrow.
            as_.Movdqa(tmp, xhi);
            as_.Pand(tmp, smask);
            as_.Por(tmp, half);
            as_.Addps(xhi, tmp);
            as_.Roundps(xhi, xhi, int8_t{0x03});
            as_.Vcvtps2ph(xhi, xhi, int8_t{0});
            // Recombine: xhi << 8 bytes; xlo |= xhi.
            as_.Pslldq(xhi, int8_t{8});
            as_.Por(xlo, xhi);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          }
          return;
        }
        // args.size for the FP variant of two-reg misc carries
        // (bit23=a, bit22=sz).  The decoder only dispatches kFrintaV when
        // bit23==0, so args.size is 0b00 (FP32 .2S/.4S) or 0b01 (FP64 .2D).
        // Reject any sneaky out-of-range size defensively; mirrors the
        // interpreter's `args.size & 1` FP32/FP64 dispatch (interpreter.h
        // around line 5850).
        if (args.size > 1) { Undefined(); return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { Undefined(); return; }
        // FP64 magnitude gate uses PCMPGTQ (SSE4.2).  Bail to interpreter
        // on hosts that lack it (test only; Digitalis target CPUs have it).
        if (is_double && !host_platform::kHasSSE4_2) {
          success_ = false; return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister copysign = AllocTempSimdReg();
        SimdRegister half = AllocTempSimdReg();
        SimdRegister abs_bits = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        // Per-lane |bits(xn)| -- IEEE-754 bits compare as unsigned int for
        // non-negative values; clearing the sign bit gives |bits(xn)|.
        as_.Pcmpeqd(abs_bits, abs_bits);              // all-ones
        if (is_double) {
          as_.Psrlq(abs_bits, int8_t{1});             // 0x7FFF.. per lane
        } else {
          as_.Psrld(abs_bits, int8_t{1});             // 0x7FFFFFFF per lane
        }
        as_.Pand(abs_bits, xn);                       // |bits(xn)|
        // Build the sign-bit mask in copysign, then AND with xn to extract
        // per-lane sign bits.
        as_.Pcmpeqd(copysign, copysign);
        if (is_double) {
          as_.Psllq(copysign, int8_t{63});
        } else {
          as_.Pslld(copysign, int8_t{31});
        }
        as_.Pand(copysign, xn);
        // Build per-lane 0.5 broadcast in half (FP32: 0x3F000000, FP64:
        // 0x3FE0000000000000).  Move via a GP scratch -> Movd/Movq -> Pshufd.
        Register gp_half = AllocTempReg();
        if (gp_half == no_register) { Undefined(); return; }
        if (is_double) {
          as_.Movq(gp_half, int64_t{0x3FE0000000000000LL});
          as_.Movq(half, gp_half);
          // imm=0x44 = 01_00_01_00 -> broadcast low 64 bits across both
          // 64-bit lanes of the XMM.
          as_.Pshufd(half, half, static_cast<int8_t>(0x44));
        } else {
          as_.Movl(gp_half, int32_t{0x3F000000});
          as_.Movd(half, gp_half);
          // imm=0x00 broadcasts the low dword across all 4 dwords.
          as_.Pshufd(half, half, static_cast<int8_t>(0x00));
        }
        // copysign |= half  -> per-lane sign(xn) * 0.5.
        as_.Por(copysign, half);
        // Per-lane magnitude gate.  Broadcast the threshold bit pattern
        // (FP32 2^23 = 0x4B000000; FP64 2^52 = 0x4330000000000000) into
        // `half` (its 0.5 contents are no longer needed) and per-lane
        // signed-compare (threshold > abs_bits).  Both operands have
        // their sign bits clear (threshold is positive; abs_bits had its
        // sign bit masked), so signed and unsigned compare agree.
        //
        //   |x| <  threshold  ->  mask = all-ones  ->  addend preserved
        //                          (add-half-and-trunc rounds correctly)
        //   |x| >= threshold  ->  mask = 0         ->  addend zeroed
        //                          (xn+0 = xn; ROUNDPS/PD is no-op for
        //                          already-integer x, and propagates
        //                          NaN/Inf unchanged per Intel SDM).
        //
        // Without this gate, RNE round-half-to-even would bump
        // odd-mantissa integers >= 2^23 (FP32) / >= 2^52 (FP64) to the
        // next even because the 0.5 nudge lands a tie below the LSB.
        if (is_double) {
          as_.Movq(gp_half, int64_t{0x4330000000000000LL});  // 2^52
          as_.Movq(half, gp_half);
          as_.Pshufd(half, half, static_cast<int8_t>(0x44));
          as_.Pcmpgtq(half, abs_bits);
        } else {
          as_.Movl(gp_half, int32_t{0x4B000000});            // 2^23
          as_.Movd(half, gp_half);
          as_.Pshufd(half, half, static_cast<int8_t>(0x00));
          as_.Pcmpgtd(half, abs_bits);
        }
        as_.Pand(copysign, half);   // zero addend in already-integer lanes
        // xn += copysign(0.5, xn); truncate toward zero.
        if (is_double) {
          as_.Addpd(xn, copysign);
          as_.Roundpd(xn, xn, int8_t{0x03});
        } else {
          as_.Addps(xn, copysign);
          as_.Roundps(xn, xn, int8_t{0x03});
        }
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // Vector FSQRT (FP32 .2S/.4S, FP64 .2D, FP16 .4H/.8H) -- per-lane
      // square root.
      //   FP32/FP64: SSE SQRTPS / SQRTPD; default MXCSR rounding (RNE)
      //   matches ARM's default FPCR rounding. NaN, signed zeros and
      //   negative finites all behave identically to the ARM ARM
      //   semantics (SQRTPS/PD propagate NaN, produce -0 for -0 input
      //   and qNaN for negative finite input).
      //   FP16: F16C round-trip (Vcvtph2ps widen -> SQRTPS -> Vcvtps2ph
      //   narrow imm=0). Per the standing rule, F16C round-trip is
      //   exact for FP16 unary FSQRT. The .8H form splits the upper 4
      //   half-lanes into a second F16C round-trip (no AVX YMM path).
      // FCVTZS V (vector FP→signed int, truncating).  Handles .2S / .4S
      // (FP32 → S32) directly with CVTTPS2DQ plus an ARM-vs-x86 saturation
      // fix-up; .2D (FP64 → S64) via per-lane scalar Cvttsd2siq + scalar
      // saturation fix-up because x86 CVTTPD2DQ produces only 32-bit
      // outputs (a per-lane scalar path is the cleanest AVX-1-compatible
      // lowering — AVX-512 VCVTTPD2QQ would be the SIMD-native alternative).
      // FP16 bails to the interpreter.
      //
      // FP32 algorithm.  x86 CVTTPS2DQ returns 0x80000000 (INT32_MIN) for
      // NaN, ±Inf, and any out-of-range FP.  ARM wants: NaN → 0,
      // value ≥ 2^31 → INT32_MAX, value < -2^31 → INT32_MIN.
      //   1. Convert via CVTTPS2DQ.
      //   2. Build NaN mask (CMPUNORDPS src,src) and clear NaN lanes in
      //      the result (PANDN).
      //   3. Detect "result == INT32_MIN" (PCMPEQD).
      //   4. Detect "src non-negative" — PSRAD src,31 gives 0 for non-neg
      //      lanes; the FP sign bit IS the MSB.
      //   5. Positive-overflow mask = (result == INT32_MIN) AND (src non-neg).
      //      NaN-lanes-in-result are already 0 after step 2, so they don't
      //      match INT32_MIN — naturally excluded.
      //   6. Flip INT32_MIN → INT32_MAX in pos-overflow lanes via PXOR
      //      with the all-1s mask (0x80000000 ^ 0xFFFFFFFF = 0x7FFFFFFF).
      // Negative-overflow lanes need no fix-up — INT32_MIN is the correct
      // ARM-saturated value.
      //
      // FP64 (.2D) algorithm.  Emit the scalar FCVTZS-D fix-up twice (once
      // per lane).  For each lane: Movsd-load the FP64, Cvttsd2siq, then
      // classify by PF (NaN) and the FP sign bit (raw Movq to GP) — same
      // structure as the scalar FCVTZS path at the top of FpIntConversion.
      // .1D (size=11, Q=0) is reserved per ARM ARM; bail.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzsV: {
        if (args.is_fp16) { success_ = false; return; }
        if (args.size == 0b11) {
          if (!args.q) { success_ = false; return; }  // .1D reserved
          SimdRegister xmm = AllocTempSimdReg();
          Register tmp = AllocTempReg();
          Register sign_tmp = AllocTempReg();
          if (xmm == no_simd_register || tmp == no_register ||
              sign_tmp == no_register) {
            success_ = false; return;
          }
          for (int lane = 0; lane < 2; ++lane) {
            as_.Movsd(xmm,
                      {.base = Assembler::rbp, .disp = vn_off + lane * 8});
            as_.Cvttsd2siq(tmp, xmm);
            Assembler::Label* nan_path = as_.MakeLabel();
            Assembler::Label* done = as_.MakeLabel();
            as_.Ucomisd(xmm, xmm);
            as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
            as_.Movq(sign_tmp, xmm);
            as_.Testq(sign_tmp, sign_tmp);
            as_.Jcc(Assembler::Condition::kNegative, *done);
            as_.Testq(tmp, tmp);
            as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
            as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
            as_.Jmp(*done);
            as_.Bind(nan_path);
            as_.Xorq(tmp, tmp);
            as_.Bind(done);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + lane * 8},
                     tmp);
          }
          return;
        }
        if (args.size != 0b10) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister x_dst = AllocTempSimdReg();
        SimdRegister x_mask = AllocTempSimdReg();
        SimdRegister x_eqmin = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        // 1. Primary conversion.
        as_.Movdqa(x_dst, xn);
        as_.Cvttps2dq(x_dst, x_dst);
        // 2. NaN mask -> zero those lanes in x_dst.
        as_.Movdqa(x_mask, xn);
        as_.Cmpunordps(x_mask, x_mask);  // 1s in NaN lanes
        as_.Pandn(x_mask, x_dst);         // x_mask = ~nan_mask & x_dst
        as_.Movdqa(x_dst, x_mask);
        // 3. Broadcast INT32_MIN constant.
        as_.Pcmpeqd(x_mask, x_mask);
        as_.Pslld(x_mask, int8_t{31});    // 0x80000000 per lane
        // 4. result == INT32_MIN ?
        as_.Movdqa(x_eqmin, x_dst);
        as_.Pcmpeqd(x_eqmin, x_mask);
        // 5. src non-negative ?  PSRAD by 31: 0 if non-neg, all-1s if neg.
        as_.Movdqa(x_mask, xn);
        as_.Psrad(x_mask, int8_t{31});    // 1s = negative
        as_.Pandn(x_mask, x_eqmin);       // x_mask = ~neg & eqmin = pos-ovf
        // 6. Flip INT_MIN -> INT_MAX in pos-overflow lanes.
        as_.Pxor(x_dst, x_mask);
        if (!args.q) MaskLow64(x_dst);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, x_dst);
        return;
      }
      // FCVTZU V (vector FP->unsigned int, truncating).  Handles .2S /
      // .4S (FP32 -> U32) directly with a CVTTPS2DQ-based lowering and
      // a "subtract 2^31" offset trick; .2D (FP64 -> U64) via per-lane
      // scalar Cvttsd2siq with the scalar "subtract 2^63" offset trick
      // for the [2^63, 2^64) range.  FP16 bails to the interpreter.
      //
      // ARM FCVTZU saturation rules (for both FP32->U32 and FP64->U64
      // with the obvious bound substitution):
      //   NaN              -> 0
      //   FP < 0 (incl -0) -> 0
      //   FP >= 2^32       -> UINT32_MAX (0xFFFFFFFF)
      //   FP in [0, 2^32)  -> truncate toward zero
      //
      // Algorithm:
      //   1. Clamp negative and NaN inputs to 0 via MAXPS(src, zero).
      //      MAXPS returns the second operand when the first is NaN
      //      (and for +0/-0 pairs), so NaN and negative lanes collapse
      //      to 0 in a single instruction.
      //   2. Build the FP32 constant 2^31 (0x4F000000) broadcast per
      //      lane via a Movd + Pshufd from a GP scratch.
      //   3. needs_offset mask = (2^31 <= src_clamped) via CMPLEPS.
      //   4. offset_amount = 2^31 where needs_offset, else 0 (PAND).
      //   5. src_for_cvt = src_clamped - offset_amount.  The subtract is
      //      exact in FP32 in the [2^31, 2^32) range (both operands at
      //      the same exponent step).  Result lands in [0, 2^31] -- the
      //      [0, 2^31) inputs pass through unchanged because their
      //      needs_offset mask is 0.
      //   6. too_big mask = (2^31 <= src_for_cvt).  Equivalent to
      //      original >= 2^32 (since src_for_cvt = original - 2^31 for
      //      values that hit the offset, and src_for_cvt < 2^31 for
      //      values that didn't).  +Inf falls in this bucket.
      //   7. CVTTPS2DQ converts.  Inputs <= 2^31; the boundary case
      //      (2^31 exact) saturates to 0x80000000 = INT32_MIN, but the
      //      too_big mask catches that lane and overwrites with all-1s.
      //   8. Build 0x80000000 per lane via PCMPEQD + PSLLD 31.
      //      AND with needs_offset, then OR into the result -- this
      //      restores the high bit for values in [2^31, 2^32) whose
      //      true unsigned representation has bit 31 set.
      //   9. OR with too_big -- bits-1 lanes become 0xFFFFFFFF, lanes
      //      with too_big=0 are unchanged.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzuV: {
        if (args.is_fp16) { success_ = false; return; }
        if (args.size == 0b11) {
          if (!args.q) { success_ = false; return; }  // .1D reserved
          // Per-lane scalar FCVTZU-D.  Mirror of the sf=1 scalar
          // FCVTZU-D path at the top of FpIntConversion: NaN -> 0,
          // negative -> 0, FP < 2^63 -> direct cvtt-Q, FP in
          // [2^63, 2^64) -> subtract 2^63 (exact in FP64 at that
          // exponent step) + cvtt + bit 63 set, FP >= 2^64 ->
          // UINT64_MAX.
          SimdRegister xmm = AllocTempSimdReg();
          SimdRegister bound_xmm = AllocTempSimdReg();    // FP64(2^63)
          SimdRegister bound2_xmm = AllocTempSimdReg();   // FP64(2^64)
          Register tmp = AllocTempReg();
          Register sign_tmp = AllocTempReg();
          if (xmm == no_simd_register || bound_xmm == no_simd_register ||
              bound2_xmm == no_simd_register || tmp == no_register ||
              sign_tmp == no_register) {
            success_ = false; return;
          }
          // Pre-load 2^63 and 2^64 FP64 constants; survive across both
          // lane emits because we only Subsd into xmm.
          as_.Movq(tmp, static_cast<int64_t>(0x43E0000000000000LL));
          as_.Movq(bound_xmm, tmp);
          as_.Movq(tmp, static_cast<int64_t>(0x43F0000000000000LL));
          as_.Movq(bound2_xmm, tmp);
          for (int lane = 0; lane < 2; ++lane) {
            as_.Movsd(xmm,
                      {.base = Assembler::rbp, .disp = vn_off + lane * 8});
            Assembler::Label* zero_path = as_.MakeLabel();
            Assembler::Label* direct_path = as_.MakeLabel();
            Assembler::Label* sat_max = as_.MakeLabel();
            Assembler::Label* done = as_.MakeLabel();
            // NaN -> 0.
            as_.Ucomisd(xmm, xmm);
            as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
            // FP < 0 -> 0 (and -0.0 falls through to direct_path with
            // cvtt-Q(-0.0) = 0, also correct).
            as_.Movq(sign_tmp, xmm);
            as_.Testq(sign_tmp, sign_tmp);
            as_.Jcc(Assembler::Condition::kNegative, *zero_path);
            // FP < 2^63 -> direct cvtt-Q gives the exact u64.
            as_.Ucomisd(xmm, bound_xmm);
            as_.Jcc(Assembler::Condition::kBelow, *direct_path);
            // FP >= 2^64 -> UINT64_MAX.
            as_.Ucomisd(xmm, bound2_xmm);
            as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);
            // FP in [2^63, 2^64): subtract 2^63 (exact at this
            // exponent step), cvtt, OR bit 63 back in.
            as_.Subsd(xmm, bound_xmm);
            as_.Cvttsd2siq(tmp, xmm);
            as_.Btsq(tmp, int8_t{63});
            as_.Jmp(*done);

            as_.Bind(sat_max);
            as_.Movq(tmp, static_cast<int64_t>(-1));  // UINT64_MAX
            as_.Jmp(*done);

            as_.Bind(direct_path);
            as_.Cvttsd2siq(tmp, xmm);
            as_.Jmp(*done);

            as_.Bind(zero_path);
            as_.Xorq(tmp, tmp);

            as_.Bind(done);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + lane * 8},
                     tmp);
          }
          return;
        }
        if (args.size != 0b10) { success_ = false; return; }
        SimdRegister x_dst = AllocTempSimdReg();
        SimdRegister x_pow31 = AllocTempSimdReg();
        SimdRegister x_needs_off = AllocTempSimdReg();
        SimdRegister x_scratch = AllocTempSimdReg();
        if (!success()) { return; }
        Register gp_tmp = AllocTempReg();
        if (gp_tmp == no_register) { success_ = false; return; }
        // 1. Load src and clamp neg/NaN to 0 via MAXPS with zero.
        as_.Movdqu(x_dst, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(x_scratch, x_scratch);          // 0.0 per lane
        as_.Maxps(x_dst, x_scratch);             // NaN/neg -> 0
        // 2. Broadcast 2^31 = 0x4F000000 to all four FP32 lanes.
        as_.Movl(gp_tmp, int32_t{0x4F000000});
        as_.Movd(x_pow31, gp_tmp);
        as_.Pshufd(x_pow31, x_pow31, int8_t{0x00});
        // 3. needs_offset = (2^31 <= src_clamped).
        as_.Movdqa(x_needs_off, x_pow31);
        as_.Cmpleps(x_needs_off, x_dst);
        // 4. offset_amount = 2^31 where needs_offset (reuse x_scratch).
        as_.Movdqa(x_scratch, x_pow31);
        as_.Pand(x_scratch, x_needs_off);
        // 5. src_for_cvt = src_clamped - offset_amount.
        as_.Subps(x_dst, x_scratch);
        // 6. too_big = (2^31 <= src_for_cvt).  Reuse x_scratch for mask.
        as_.Movdqa(x_scratch, x_pow31);
        as_.Cmpleps(x_scratch, x_dst);
        // 7. CVTTPS2DQ; in-range lanes get correct s32, too_big lanes
        //    saturate to 0x80000000 (overwritten below).
        as_.Cvttps2dq(x_dst, x_dst);
        // 8. Build 0x80000000 per lane via self-PCMPEQD + PSLLD 31.
        //    Overwrite x_pow31; its compare-constant role is done.
        as_.Pcmpeqd(x_pow31, x_pow31);
        as_.Pslld(x_pow31, int8_t{31});
        as_.Pand(x_pow31, x_needs_off);          // 0x80000000 where subtracted
        as_.Por(x_dst, x_pow31);
        // 9. Saturate too-big lanes to 0xFFFFFFFF.
        as_.Por(x_dst, x_scratch);
        if (!args.q) MaskLow64(x_dst);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, x_dst);
        return;
      }
      // FCVTNS/PS/MS (signed) and FCVTNU/PU/MU (unsigned) vector FP -> int
      // with explicit rounding mode.  ARM rmode -> ROUNDSD imm:
      //   FCVTNS / FCVTNU  RNE             imm=0x08 (RNE + suppress-inexact)
      //   FCVTPS / FCVTPU  round +inf      imm=0x0A
      //   FCVTMS / FCVTMU  round -inf      imm=0x09
      //
      // Strategy mirrors the scalar FCVTNS/PS/MS path at lite_translator.h
      // ~2949: ROUNDSD the FP value first, then reuse the FCVTZS / FCVTZU
      // saturation classifier verbatim.  After ROUNDSD, finite values are
      // exact integers in FP domain, and NaN/+/-Inf/sign-of-zero pass through
      // unchanged so the classifier still distinguishes them correctly.
      //
      // .2D path: per-lane scalar Roundsd + Cvttsd2siq, identical scaffolding
      // to the FCVTZS V / FCVTZU V .2D paths above.  .2S/.4S (FP32) uses
      // ROUNDPS + CVTTPS2DQ + the FCVTZS V / FCVTZU V .2S/.4S vector
      // saturation fix-up.  FP16 bails to the interpreter.
      //
      // Size selector: bits[23:22] = (bit23='a', bit22='sz').  bit23 is
      // baked into the opcode dispatch (FCVTPS/PU have bit23=1, FCVTNS/NU/
      // MS/MU have bit23=0), so `args.size` carries both — use
      // `(args.size & 1)` as the canonical FP32-vs-FP64 selector that
      // works for both bit23 halves.  FP64 .2D corresponds to:
      //   FCVTPS/PU (bit23=1, sz=1): args.size == 0b11
      //   FCVTNS/NU/MS/MU (bit23=0, sz=1): args.size == 0b01
      // FP32 .2S/.4S corresponds to:
      //   FCVTPS/PU (bit23=1, sz=0): args.size == 0b10
      //   FCVTNS/NU/MS/MU (bit23=0, sz=0): args.size == 0b00
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmsV: {
        if (args.is_fp16) { success_ = false; return; }
        int8_t round_imm;
        if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnsV) {
          round_imm = int8_t{0x08};   // RNE + suppress-inexact
        } else if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpsV) {
          round_imm = int8_t{0x0A};   // toward +inf
        } else {
          round_imm = int8_t{0x09};   // toward -inf
        }
        if ((args.size & 1) == 1) {
          // FP64 .2D path: per-lane scalar Roundsd + Cvttsd2siq.
          if (!args.q) { success_ = false; return; }  // .1D reserved
          SimdRegister xmm = AllocTempSimdReg();
          Register tmp = AllocTempReg();
          Register sign_tmp = AllocTempReg();
          if (xmm == no_simd_register || tmp == no_register ||
              sign_tmp == no_register) {
            success_ = false; return;
          }
          for (int lane = 0; lane < 2; ++lane) {
            as_.Movsd(xmm,
                      {.base = Assembler::rbp, .disp = vn_off + lane * 8});
            as_.Roundsd(xmm, xmm, round_imm);
            as_.Cvttsd2siq(tmp, xmm);
            Assembler::Label* nan_path = as_.MakeLabel();
            Assembler::Label* done = as_.MakeLabel();
            as_.Ucomisd(xmm, xmm);
            as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
            as_.Movq(sign_tmp, xmm);
            as_.Testq(sign_tmp, sign_tmp);
            as_.Jcc(Assembler::Condition::kNegative, *done);
            as_.Testq(tmp, tmp);
            as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
            as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
            as_.Jmp(*done);
            as_.Bind(nan_path);
            as_.Xorq(tmp, tmp);
            as_.Bind(done);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + lane * 8},
                     tmp);
          }
          return;
        }
        // FP32 .2S/.4S path: ROUNDPS + CVTTPS2DQ + FCVTZS V .2S/.4S
        // saturation fix-up.  Roundps preserves NaN/+/-Inf/sign-of-zero,
        // so the fix-up classifier still distinguishes them correctly.
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister x_dst = AllocTempSimdReg();
        SimdRegister x_mask = AllocTempSimdReg();
        SimdRegister x_eqmin = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Roundps(xn, xn, round_imm);
        // 1. Primary conversion.
        as_.Movdqa(x_dst, xn);
        as_.Cvttps2dq(x_dst, x_dst);
        // 2. NaN mask -> zero those lanes in x_dst.
        as_.Movdqa(x_mask, xn);
        as_.Cmpunordps(x_mask, x_mask);  // 1s in NaN lanes
        as_.Pandn(x_mask, x_dst);         // x_mask = ~nan_mask & x_dst
        as_.Movdqa(x_dst, x_mask);
        // 3. Broadcast INT32_MIN constant.
        as_.Pcmpeqd(x_mask, x_mask);
        as_.Pslld(x_mask, int8_t{31});    // 0x80000000 per lane
        // 4. result == INT32_MIN ?
        as_.Movdqa(x_eqmin, x_dst);
        as_.Pcmpeqd(x_eqmin, x_mask);
        // 5. src non-negative ?  PSRAD by 31: 0 if non-neg, all-1s if neg.
        as_.Movdqa(x_mask, xn);
        as_.Psrad(x_mask, int8_t{31});    // 1s = negative
        as_.Pandn(x_mask, x_eqmin);       // x_mask = ~neg & eqmin = pos-ovf
        // 6. Flip INT_MIN -> INT_MAX in pos-overflow lanes.
        as_.Pxor(x_dst, x_mask);
        if (!args.q) MaskLow64(x_dst);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, x_dst);
        return;
      }
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmuV: {
        if (args.is_fp16) { success_ = false; return; }
        int8_t round_imm;
        if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV) {
          round_imm = int8_t{0x08};   // RNE + suppress-inexact
        } else if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV) {
          round_imm = int8_t{0x0A};   // toward +inf
        } else {
          round_imm = int8_t{0x09};   // toward -inf
        }
        if ((args.size & 1) == 1) {
          // FP64 .2D path: per-lane scalar Roundsd with offset-trick fix-up.
          if (!args.q) { success_ = false; return; }  // .1D reserved
          SimdRegister xmm = AllocTempSimdReg();
          SimdRegister bound_xmm = AllocTempSimdReg();    // FP64(2^63)
          SimdRegister bound2_xmm = AllocTempSimdReg();   // FP64(2^64)
          Register tmp = AllocTempReg();
          Register sign_tmp = AllocTempReg();
          if (xmm == no_simd_register || bound_xmm == no_simd_register ||
              bound2_xmm == no_simd_register || tmp == no_register ||
              sign_tmp == no_register) {
            success_ = false; return;
          }
          // Pre-load 2^63 and 2^64 FP64 constants; survive across both
          // lane emits because we only Subsd into xmm.  Same as FCVTZU V .2D.
          as_.Movq(tmp, static_cast<int64_t>(0x43E0000000000000LL));
          as_.Movq(bound_xmm, tmp);
          as_.Movq(tmp, static_cast<int64_t>(0x43F0000000000000LL));
          as_.Movq(bound2_xmm, tmp);
          for (int lane = 0; lane < 2; ++lane) {
            as_.Movsd(xmm,
                      {.base = Assembler::rbp, .disp = vn_off + lane * 8});
            as_.Roundsd(xmm, xmm, round_imm);
            Assembler::Label* zero_path = as_.MakeLabel();
            Assembler::Label* direct_path = as_.MakeLabel();
            Assembler::Label* sat_max = as_.MakeLabel();
            Assembler::Label* done = as_.MakeLabel();
            // NaN -> 0 (Roundsd passes NaN through).
            as_.Ucomisd(xmm, xmm);
            as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
            // FP < 0 (sign bit set) -> 0.  Roundsd preserves the sign of
            // zero, so e.g. ceil(-0.5) = -0.0 still classifies as negative
            // here and clamps to 0 -- matching ARM FCVT*U behavior.
            as_.Movq(sign_tmp, xmm);
            as_.Testq(sign_tmp, sign_tmp);
            as_.Jcc(Assembler::Condition::kNegative, *zero_path);
            // FP < 2^63 -> direct cvtt-Q gives the exact u64.
            as_.Ucomisd(xmm, bound_xmm);
            as_.Jcc(Assembler::Condition::kBelow, *direct_path);
            // FP >= 2^64 -> UINT64_MAX.
            as_.Ucomisd(xmm, bound2_xmm);
            as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);
            // FP in [2^63, 2^64): subtract 2^63 (exact at this exponent
            // step), cvtt, OR bit 63 back in.
            as_.Subsd(xmm, bound_xmm);
            as_.Cvttsd2siq(tmp, xmm);
            as_.Btsq(tmp, int8_t{63});
            as_.Jmp(*done);

            as_.Bind(sat_max);
            as_.Movq(tmp, static_cast<int64_t>(-1));  // UINT64_MAX
            as_.Jmp(*done);

            as_.Bind(direct_path);
            as_.Cvttsd2siq(tmp, xmm);
            as_.Jmp(*done);

            as_.Bind(zero_path);
            as_.Xorq(tmp, tmp);

            as_.Bind(done);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + lane * 8},
                     tmp);
          }
          return;
        }
        // FP32 .2S/.4S path: Roundps + FCVTZU V .2S/.4S offset-trick
        // saturation fix-up.  Same algorithm as FCVTZU V .2S/.4S above
        // with a Roundps insertion before the MAXPS clamp.  Roundps
        // preserves sign-of-zero, so ceil(-0.5) = -0.0 still classifies
        // as negative via MAXPS(src, 0) -> 0 (NaN/neg -> 0).
        SimdRegister x_dst = AllocTempSimdReg();
        SimdRegister x_pow31 = AllocTempSimdReg();
        SimdRegister x_needs_off = AllocTempSimdReg();
        SimdRegister x_scratch = AllocTempSimdReg();
        if (!success()) { return; }
        Register gp_tmp = AllocTempReg();
        if (gp_tmp == no_register) { success_ = false; return; }
        // Load and round.
        as_.Movdqu(x_dst, {.base = Assembler::rbp, .disp = vn_off});
        as_.Roundps(x_dst, x_dst, round_imm);
        // 1. Clamp neg/NaN to 0 via MAXPS with zero.
        as_.Pxor(x_scratch, x_scratch);          // 0.0 per lane
        as_.Maxps(x_dst, x_scratch);             // NaN/neg -> 0
        // 2. Broadcast 2^31 = 0x4F000000 to all four FP32 lanes.
        as_.Movl(gp_tmp, int32_t{0x4F000000});
        as_.Movd(x_pow31, gp_tmp);
        as_.Pshufd(x_pow31, x_pow31, int8_t{0x00});
        // 3. needs_offset = (2^31 <= src_clamped).
        as_.Movdqa(x_needs_off, x_pow31);
        as_.Cmpleps(x_needs_off, x_dst);
        // 4. offset_amount = 2^31 where needs_offset.
        as_.Movdqa(x_scratch, x_pow31);
        as_.Pand(x_scratch, x_needs_off);
        // 5. src_for_cvt = src_clamped - offset_amount.
        as_.Subps(x_dst, x_scratch);
        // 6. too_big = (2^31 <= src_for_cvt).
        as_.Movdqa(x_scratch, x_pow31);
        as_.Cmpleps(x_scratch, x_dst);
        // 7. CVTTPS2DQ.
        as_.Cvttps2dq(x_dst, x_dst);
        // 8. Build 0x80000000 per lane; AND with needs_offset; OR in.
        as_.Pcmpeqd(x_pow31, x_pow31);
        as_.Pslld(x_pow31, int8_t{31});
        as_.Pand(x_pow31, x_needs_off);
        as_.Por(x_dst, x_pow31);
        // 9. Saturate too-big lanes to 0xFFFFFFFF.
        as_.Por(x_dst, x_scratch);
        if (!args.q) MaskLow64(x_dst);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, x_dst);
        return;
      }
      // Vector FCVTAS / FCVTAU (round-to-nearest, ties-away-from-zero, FP -> int).
      //
      // x86 ROUNDPS/ROUNDPD have no ties-away rounding mode (only RNE/floor/
      // ceil/trunc/MXCSR).  Reuse the FRINTA V trick: per lane,
      //     addend = copysign(0.5, x)              (gated to 0 when |x| >= 2^p)
      //     FCVTA*(x) = trunc(x + addend)
      // where p is the FP mantissa precision (23 for FP32, 52 for FP64).
      // The magnitude gate is required to avoid the ties-to-even bump on
      // odd-mantissa integers >= 2^p: for |x| >= 2^p, x is
      // already integer in its FP type, so the addend must be zero.
      //
      // After the FRINTA add-and-trunc step, xn holds the round-to-nearest-
      // ties-away of the input as an integer-valued FP.  Saturation:
      //   FP64 .2D:        per-lane scalar Cvttsd2siq + signed/unsigned fix-up
      //   FP32 .2S/.4S:    vector CVTTPS2DQ + FCVTZ{S,U} V saturation fix-up
      // FP16 .4H/.8H bails to the interpreter (matches FCVTNS V etc.).
      //
      // ROUNDPS/PD imm=3 = truncate-toward-zero + suppress-inexact.  NaN/
      // +/-Inf/sign-of-zero pass through ADDPS/ADDPD and ROUNDPS/PD
      // unchanged, so the saturation classifiers still distinguish them.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtasV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV: {
        if (args.is_fp16) { success_ = false; return; }
        const bool is_unsigned =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV);
        if ((args.size & 1) == 1) {
          // ---------------- FP64 .2D path ----------------
          if (!args.q) { success_ = false; return; }   // .1D reserved
          if (!host_platform::kHasSSE4_2) { success_ = false; return; }
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister copysign = AllocTempSimdReg();
          SimdRegister half = AllocTempSimdReg();
          SimdRegister abs_bits = AllocTempSimdReg();
          Register gp_half = AllocTempReg();
          if (xn == no_simd_register || copysign == no_simd_register ||
              half == no_simd_register || abs_bits == no_simd_register ||
              gp_half == no_register) {
            success_ = false; return;
          }
          // FRINTA dance (vector form): build per-lane addend gated by
          // |x| < 2^52, add to xn, ROUNDPD imm=3.
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Pcmpeqd(abs_bits, abs_bits);
          as_.Psrlq(abs_bits, int8_t{1});                 // 0x7FFF.. per lane
          as_.Pand(abs_bits, xn);                          // |bits(xn)|
          as_.Pcmpeqd(copysign, copysign);
          as_.Psllq(copysign, int8_t{63});                 // 0x8000.. per lane
          as_.Pand(copysign, xn);                          // sign bit of xn
          as_.Movq(gp_half, int64_t{0x3FE0000000000000LL});  // 0.5 (FP64)
          as_.Movq(half, gp_half);
          as_.Pshufd(half, half, static_cast<int8_t>(0x44));
          as_.Por(copysign, half);                          // sign(xn) | 0.5
          as_.Movq(gp_half, int64_t{0x4330000000000000LL});  // 2^52 (FP64)
          as_.Movq(half, gp_half);
          as_.Pshufd(half, half, static_cast<int8_t>(0x44));
          as_.Pcmpgtq(half, abs_bits);                       // 1s where |x| < 2^52
          as_.Pand(copysign, half);                          // zero addend if already int
          as_.Addpd(xn, copysign);
          as_.Roundpd(xn, xn, int8_t{0x03});                  // trunc + suppress-inexact
          // Per-lane saturation.  Lane 0 reads xn directly; lane 1 uses
          // Pshufd(xmm, xn, 0xEE) to splat xn dwords[2:3] into xmm dword[0:1].
          SimdRegister xmm = AllocTempSimdReg();
          Register tmp = AllocTempReg();
          Register sign_tmp = AllocTempReg();
          if (xmm == no_simd_register || tmp == no_register ||
              sign_tmp == no_register) {
            success_ = false; return;
          }
          if (!is_unsigned) {
            for (int lane = 0; lane < 2; ++lane) {
              if (lane == 0) {
                as_.Movdqa(xmm, xn);
              } else {
                as_.Pshufd(xmm, xn, static_cast<int8_t>(0xEE));
              }
              as_.Cvttsd2siq(tmp, xmm);
              Assembler::Label* nan_path = as_.MakeLabel();
              Assembler::Label* done = as_.MakeLabel();
              as_.Ucomisd(xmm, xmm);
              as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
              as_.Movq(sign_tmp, xmm);
              as_.Testq(sign_tmp, sign_tmp);
              as_.Jcc(Assembler::Condition::kNegative, *done);
              as_.Testq(tmp, tmp);
              as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
              as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
              as_.Jmp(*done);
              as_.Bind(nan_path);
              as_.Xorq(tmp, tmp);
              as_.Bind(done);
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + lane * 8},
                       tmp);
            }
          } else {
            SimdRegister bound_xmm = AllocTempSimdReg();
            SimdRegister bound2_xmm = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movq(tmp, int64_t{0x43E0000000000000LL});   // 2^63 (FP64)
            as_.Movq(bound_xmm, tmp);
            as_.Movq(tmp, int64_t{0x43F0000000000000LL});   // 2^64 (FP64)
            as_.Movq(bound2_xmm, tmp);
            for (int lane = 0; lane < 2; ++lane) {
              if (lane == 0) {
                as_.Movdqa(xmm, xn);
              } else {
                as_.Pshufd(xmm, xn, static_cast<int8_t>(0xEE));
              }
              Assembler::Label* zero_path = as_.MakeLabel();
              Assembler::Label* direct_path = as_.MakeLabel();
              Assembler::Label* sat_max = as_.MakeLabel();
              Assembler::Label* done = as_.MakeLabel();
              as_.Ucomisd(xmm, xmm);
              as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
              as_.Movq(sign_tmp, xmm);
              as_.Testq(sign_tmp, sign_tmp);
              as_.Jcc(Assembler::Condition::kNegative, *zero_path);
              as_.Ucomisd(xmm, bound_xmm);
              as_.Jcc(Assembler::Condition::kBelow, *direct_path);
              as_.Ucomisd(xmm, bound2_xmm);
              as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);
              as_.Subsd(xmm, bound_xmm);
              as_.Cvttsd2siq(tmp, xmm);
              as_.Btsq(tmp, int8_t{63});
              as_.Jmp(*done);
              as_.Bind(sat_max);
              as_.Movq(tmp, static_cast<int64_t>(-1));
              as_.Jmp(*done);
              as_.Bind(direct_path);
              as_.Cvttsd2siq(tmp, xmm);
              as_.Jmp(*done);
              as_.Bind(zero_path);
              as_.Xorq(tmp, tmp);
              as_.Bind(done);
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + lane * 8},
                       tmp);
            }
          }
          return;
        }
        // ---------------- FP32 .2S / .4S path ----------------
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister copysign = AllocTempSimdReg();
        SimdRegister half = AllocTempSimdReg();
        SimdRegister abs_bits = AllocTempSimdReg();
        Register gp_half = AllocTempReg();
        if (xn == no_simd_register || copysign == no_simd_register ||
            half == no_simd_register || abs_bits == no_simd_register ||
            gp_half == no_register) {
          success_ = false; return;
        }
        // FRINTA dance (FP32 form): build per-lane addend gated by |x| < 2^23,
        // ADDPS, ROUNDPS imm=3.
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pcmpeqd(abs_bits, abs_bits);
        as_.Psrld(abs_bits, int8_t{1});                  // 0x7FFFFFFF per lane
        as_.Pand(abs_bits, xn);                           // |bits(xn)|
        as_.Pcmpeqd(copysign, copysign);
        as_.Pslld(copysign, int8_t{31});                  // 0x80000000 per lane
        as_.Pand(copysign, xn);                            // sign bit of xn
        as_.Movl(gp_half, int32_t{0x3F000000});            // 0.5 (FP32)
        as_.Movd(half, gp_half);
        as_.Pshufd(half, half, static_cast<int8_t>(0x00));
        as_.Por(copysign, half);                           // sign(xn) | 0.5
        as_.Movl(gp_half, int32_t{0x4B000000});            // 2^23 (FP32)
        as_.Movd(half, gp_half);
        as_.Pshufd(half, half, static_cast<int8_t>(0x00));
        as_.Pcmpgtd(half, abs_bits);                        // 1s where |x| < 2^23
        as_.Pand(copysign, half);                            // zero addend if already int
        as_.Addps(xn, copysign);
        as_.Roundps(xn, xn, int8_t{0x03});                    // trunc + suppress-inexact
        if (!is_unsigned) {
          // FCVTZS V .2S/.4S saturation fix-up.
          SimdRegister x_dst = AllocTempSimdReg();
          SimdRegister x_mask = AllocTempSimdReg();
          SimdRegister x_eqmin = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(x_dst, xn);
          as_.Cvttps2dq(x_dst, x_dst);
          as_.Movdqa(x_mask, xn);
          as_.Cmpunordps(x_mask, x_mask);                   // NaN lanes
          as_.Pandn(x_mask, x_dst);                          // ~NaN & x_dst
          as_.Movdqa(x_dst, x_mask);
          as_.Pcmpeqd(x_mask, x_mask);
          as_.Pslld(x_mask, int8_t{31});                     // INT32_MIN per lane
          as_.Movdqa(x_eqmin, x_dst);
          as_.Pcmpeqd(x_eqmin, x_mask);                      // result == INT32_MIN?
          as_.Movdqa(x_mask, xn);
          as_.Psrad(x_mask, int8_t{31});                     // 1s = src negative
          as_.Pandn(x_mask, x_eqmin);                        // ~neg & eqmin = pos-ovf
          as_.Pxor(x_dst, x_mask);                            // flip INT_MIN -> INT_MAX
          if (!args.q) MaskLow64(x_dst);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, x_dst);
        } else {
          // FCVTZU V .2S/.4S saturation fix-up (offset-by-2^31 trick).
          SimdRegister x_dst = AllocTempSimdReg();
          SimdRegister x_pow31 = AllocTempSimdReg();
          SimdRegister x_needs_off = AllocTempSimdReg();
          SimdRegister x_scratch = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(x_dst, xn);
          as_.Pxor(x_scratch, x_scratch);                    // 0.0 per lane
          as_.Maxps(x_dst, x_scratch);                        // NaN/neg -> 0
          as_.Movl(gp_half, int32_t{0x4F000000});             // 2^31 (FP32)
          as_.Movd(x_pow31, gp_half);
          as_.Pshufd(x_pow31, x_pow31, int8_t{0x00});
          as_.Movdqa(x_needs_off, x_pow31);
          as_.Cmpleps(x_needs_off, x_dst);                    // 2^31 <= clamped src
          as_.Movdqa(x_scratch, x_pow31);
          as_.Pand(x_scratch, x_needs_off);
          as_.Subps(x_dst, x_scratch);
          as_.Movdqa(x_scratch, x_pow31);
          as_.Cmpleps(x_scratch, x_dst);                       // 2^31 <= shifted src
          as_.Cvttps2dq(x_dst, x_dst);
          as_.Pcmpeqd(x_pow31, x_pow31);
          as_.Pslld(x_pow31, int8_t{31});                       // 0x80000000 per lane
          as_.Pand(x_pow31, x_needs_off);
          as_.Por(x_dst, x_pow31);
          as_.Por(x_dst, x_scratch);                            // saturate too-big
          if (!args.q) MaskLow64(x_dst);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, x_dst);
        }
        return;
      }
      // Vector SCVTF / UCVTF (signed/unsigned integer -> FP).
      //
      // Encoding: opcode=11101, bit23=0 (bit23=1 is FRECPE / FRSQRTE).  The
      // decoder forces bit23=0 for this case so `args.size & 0b10` is always
      // 0; `args.size & 1` selects FP32 (.2S/.4S) vs FP64 (.2D).
      //
      // FP32 .2S / .4S:
      //   SCVTF:  single Cvtdq2ps -- x86 signed int32 -> FP32 matches ARM.
      //   UCVTF:  Cvtdq2ps treats the input as signed, so values with bit31
      //           set come out as their negative two's-complement
      //           equivalent.  Recover the unsigned representation by
      //           per-lane adding 2^32 (FP32 bits 0x4F800000) wherever bit31
      //           was set:
      //             msb_mask = Psrad(xn, 31)      // 0 or all-1s
      //             signed_fp = Cvtdq2ps(xn)
      //             addend = (FP 2^32) & msb_mask   // 2^32 where MSB set
      //             result = signed_fp + addend
      //           For values < 2^31: msb_mask = 0, addend = 0, result =
      //           signed_fp (correct).  For values >= 2^31: signed
      //           interpretation = value - 2^32, so adding 2^32 recovers
      //           the unsigned value.  2^32 = 4294967296.0 is exactly
      //           representable in FP32 (single power of two).
      //
      // FP64 .2D (Q must be 1; .1D reserved):
      //   SCVTF:  per-lane Cvtsi2sdq from memory.
      //   UCVTF:  per-lane unsigned-int64 -> FP64 via the "halve | LSB"
      //           round-to-odd trick (matches the scalar UCVTF Xd Dn JIT
      //           lowering at the top of FpIntConversion).  For values
      //           with bit63 clear the conversion is direct; for bit63 set
      //           (negative as int64), halve the value (logical shr 1)
      //           preserving the LSB via OR, convert as signed, then double
      //           back via Addsd.  This avoids the Cvtsi2sdq overflow when
      //           the input has bit63 set, and the LSB preservation keeps
      //           the conversion correctly rounded.
      case Decoder::AdvSimdTwoRegMiscOpcode::kScvtfV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV: {
        if (args.is_fp16) { success_ = false; return; }
        const bool is_unsigned =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV);
        if ((args.size & 1) == 1) {
          // ---------------- FP64 .2D path ----------------
          if (!args.q) { success_ = false; return; }   // .1D reserved
          SimdRegister xmm = AllocTempSimdReg();
          if (!success()) { return; }
          if (!is_unsigned) {
            // SCVTF V .2D: per-lane direct convert from memory.
            for (int lane = 0; lane < 2; ++lane) {
              as_.Cvtsi2sdq(
                  xmm, {.base = Assembler::rbp, .disp = vn_off + lane * 8});
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + lane * 8},
                       xmm);
            }
            return;
          }
          // UCVTF V .2D: per-lane halve-OR-LSB trick for bit63-set values.
          Register tmp = AllocTempReg();
          Register low_bit = AllocTempReg();
          if (tmp == no_register || low_bit == no_register) {
            success_ = false; return;
          }
          for (int lane = 0; lane < 2; ++lane) {
            as_.Movq(tmp,
                     {.base = Assembler::rbp, .disp = vn_off + lane * 8});
            Assembler::Label* neg_path = as_.MakeLabel();
            Assembler::Label* done = as_.MakeLabel();
            as_.Testq(tmp, tmp);
            as_.Jcc(Assembler::Condition::kNegative, *neg_path);
            as_.Cvtsi2sdq(xmm, tmp);
            as_.Jmp(*done);
            as_.Bind(neg_path);
            as_.Movq(low_bit, tmp);
            as_.Andq(low_bit, static_cast<int32_t>(1));
            as_.Shrq(tmp, static_cast<int8_t>(1));
            as_.Orq(tmp, low_bit);
            as_.Cvtsi2sdq(xmm, tmp);
            as_.Addsd(xmm, xmm);
            as_.Bind(done);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + lane * 8},
                     xmm);
          }
          return;
        }
        // ---------------- FP32 .2S / .4S path ----------------
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (!is_unsigned) {
          // SCVTF V: signed int32 -> FP32 is native.
          as_.Cvtdq2ps(xn, xn);
          if (!args.q) MaskLow64(xn);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        // UCVTF V .2S/.4S: Cvtdq2ps + per-lane 2^32 addend for MSB-set
        // lanes.  Bit-level: addend = (broadcast 0x4F800000) & psrad(xn, 31).
        SimdRegister msb = AllocTempSimdReg();
        SimdRegister addend = AllocTempSimdReg();
        Register gp_tmp = AllocTempReg();
        if (msb == no_simd_register || addend == no_simd_register ||
            gp_tmp == no_register) {
          success_ = false; return;
        }
        as_.Movdqa(msb, xn);
        as_.Psrad(msb, int8_t{31});               // 0 or all-1s per lane
        as_.Cvtdq2ps(xn, xn);                      // signed convert
        as_.Movl(gp_tmp, int32_t{0x4F800000});      // 2^32 (FP32 bits)
        as_.Movd(addend, gp_tmp);
        as_.Pshufd(addend, addend, int8_t{0x00});
        as_.Pand(addend, msb);                      // 2^32 where MSB set
        as_.Addps(xn, addend);
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      case Decoder::AdvSimdTwoRegMiscOpcode::kFsqrtV: {
        if (args.is_fp16) {
          if (!host_platform::kHasF16C) { success_ = false; return; }
          SimdRegister xlo = AllocTempSimdReg();
          if (!success()) { return; }
          if (!args.q) {
            // .4H: 4 FP16 lanes in low 64 bits of Vn.
            as_.Movq(xlo, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xlo, xlo);
            as_.Sqrtps(xlo, xlo);
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            // Vcvtps2ph already zeroes the upper 64 bits of xlo.
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          } else {
            // .8H: 8 FP16 lanes; process low 4 then high 4.
            SimdRegister xhi = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqu(xhi, {.base = Assembler::rbp, .disp = vn_off});
            // Low half: copy then widen lanes 0-3.
            as_.Movdqa(xlo, xhi);
            as_.Vcvtph2ps(xlo, xlo);
            // High half: shift right 8 bytes so upper 4 FP16 -> low 64
            // bits of xhi, then widen.
            as_.Psrldq(xhi, int8_t{8});
            as_.Vcvtph2ps(xhi, xhi);
            as_.Sqrtps(xlo, xlo);
            as_.Sqrtps(xhi, xhi);
            // Narrow each back to 4 FP16 in the low 64 bits.
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            as_.Vcvtps2ph(xhi, xhi, int8_t{0});
            // Shift high result into the upper 64 bits and OR with low.
            as_.Pslldq(xhi, int8_t{8});
            as_.Por(xlo, xhi);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          }
          return;
        }
        if (args.size != 0b10 && args.size != 0b11) { Undefined(); return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (is_double) {
          as_.Sqrtpd(xn, xn);
        } else {
          as_.Sqrtps(xn, xn);
        }
        if (!args.q) MaskLow64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // ADDV (across-lanes integer sum). Result is a single
      // scalar in the lowest lane of Vd with all other bytes zeroed.
      //
      // Encoding (DDI 0487 §C7.2 Advanced SIMD across lanes):
      //   0 Q U 01110 size 11000 11011 10 Rn Rd, with U=0.
      // Valid lane forms:
      //   size=00 Q=0: ADDV B, V.8B    (sum 8 bytes  → 8-bit lane)
      //   size=00 Q=1: ADDV B, V.16B   (sum 16 bytes → 8-bit lane)
      //   size=01 Q=0: ADDV H, V.4H    (sum 4 hw     → 16-bit lane)
      //   size=01 Q=1: ADDV H, V.8H    (sum 8 hw     → 16-bit lane)
      //   size=10 Q=1: ADDV S, V.4S    (sum 4 dw     → 32-bit lane)
      //   (size=10 Q=0 and size=11 are reserved; rejected here.)
      //
      // Byte path uses PSADBW (sum-of-absolute-differences vs 0), which
      // produces 16-bit byte sums in qword-lane positions; halfword/dword
      // paths use cascading PHADDW / PHADDD to collapse the vector.
      case Decoder::AdvSimdTwoRegMiscOpcode::kAddv: {
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        switch (args.size) {
          case 0b00: {
            SimdRegister xz = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Pxor(xz, xz);
            // xn := [sum(bytes 0..7), 0..., sum(bytes 8..15), 0...] as
            // 16-bit values in qword-lane positions.
            as_.Psadbw(xn, xz);
            if (args.q) {
              // .16B: also fold in the high-qword sum.
              SimdRegister xt = AllocTempSimdReg();
              if (!success()) { return; }
              as_.Pshufd(xt, xn, static_cast<int8_t>(0xEE));  // hi qword → lo qword
              as_.Paddq(xn, xt);
            }
            // Keep only the low byte; zero the rest of Vd.
            as_.Pslldq(xn, int8_t{15});
            as_.Psrldq(xn, int8_t{15});
            break;
          }
          case 0b01: {
            // .4H: 2 cascading PHADDW collapses 4 halfwords → 1.
            // .8H: 3 cascading PHADDW collapses 8 halfwords → 1.
            as_.Phaddw(xn, xn);
            as_.Phaddw(xn, xn);
            if (args.q) as_.Phaddw(xn, xn);
            // Keep only the low halfword (2 bytes); zero the rest.
            as_.Pslldq(xn, int8_t{14});
            as_.Psrldq(xn, int8_t{14});
            break;
          }
          case 0b10: {
            if (!args.q) { Undefined(); return; }  // .2S reserved for ADDV.
            // .4S: 2 cascading PHADDD collapses 4 dwords → 1.
            as_.Phaddd(xn, xn);
            as_.Phaddd(xn, xn);
            // Keep only the low dword (4 bytes); zero the rest.
            as_.Pslldq(xn, int8_t{12});
            as_.Psrldq(xn, int8_t{12});
            break;
          }
          default:
            Undefined();
            return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // SADDLV / UADDLV (across-lanes long integer sum).
      // Each source element is widened to 2*esize before being summed; the
      // single 2*esize-wide result is written to the lowest lane of Vd with
      // all other bytes zeroed.
      //
      // Encoding (DDI 0487 §C7.2 Advanced SIMD across lanes):
      //   0 Q U 01110 size 11000 00011 10 Rn Rd
      // U=0 → SADDLV (signed), U=1 → UADDLV (unsigned).  Valid lane forms:
      //   size=00 Q=0: V.8B   → H result (8b lanes  → 16b sum)
      //   size=00 Q=1: V.16B  → H result
      //   size=01 Q=0: V.4H   → S result (16b lanes → 32b sum)
      //   size=01 Q=1: V.8H   → S result
      //   size=10 Q=1: V.4S   → D result (32b lanes → 64b sum)
      //   (size=10 Q=0 and size=11 are reserved; decoder rejects size=11.)
      //
      // Observed `uaddlv h0, v0.8b` (insn 0x2E303800) in WhatsApp's
      // libar-bundle3.so JNI_OnLoad path (was previously interpreter-only).
      //
      // For UADDLV at .8B/.16B, PSADBW against zero already produces a
      // 16-bit sum of bytes per qword lane (max 16*255 = 4080 << 2^16), so
      // the unsigned byte paths reuse the ADDV PSADBW idiom.  For SADDLV
      // and for halfword/dword inputs, PMOVxxBW / PMOVxxWD / PMOVxxDQ widen
      // each source lane to the result width first; the widened lanes are
      // then summed via PADDW / PADDD / PADDQ + PHADDW / PHADDD cascades.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSaddlv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUaddlv: {
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSaddlv);
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        switch (args.size) {
          case 0b00: {
            // Bytes → 16-bit sum.
            if (!is_signed) {
              // UADDLV: PSADBW with zero gives 16-bit unsigned byte sum
              // per qword (max 8*255 = 2040 ≤ 2^16).
              SimdRegister xz = AllocTempSimdReg();
              if (!success()) { return; }
              as_.Pxor(xz, xz);
              as_.Psadbw(xn, xz);
              if (args.q) {
                // .16B: fold high-qword sum into low qword.
                SimdRegister xt = AllocTempSimdReg();
                if (!success()) { return; }
                as_.Pshufd(xt, xn, static_cast<int8_t>(0xEE));
                as_.Paddq(xn, xt);
              }
            } else {
              // SADDLV: sign-extend bytes → halfwords, then horizontal
              // reduce via PHADDW.  Max signed sum 16*127 = 2032, fits.
              if (args.q) {
                // .16B: widen low 8 bytes and high 8 bytes separately,
                // then lane-wise sum (8 partial halfword sums) → 3 PHADDW.
                SimdRegister xt = AllocTempSimdReg();
                if (!success()) { return; }
                as_.Pmovsxbw(xt, xn);          // low  8b → 8 halfwords
                as_.Psrldq(xn, int8_t{8});
                as_.Pmovsxbw(xn, xn);          // high 8b → 8 halfwords
                as_.Paddw(xn, xt);             // 8 partial-sum halfwords
                as_.Phaddw(xn, xn);
                as_.Phaddw(xn, xn);
                as_.Phaddw(xn, xn);
              } else {
                // .8B: only widen low 8 bytes; 3 PHADDW cascades collapse.
                as_.Pmovsxbw(xn, xn);          // low 8b → 8 halfwords
                as_.Phaddw(xn, xn);
                as_.Phaddw(xn, xn);
                as_.Phaddw(xn, xn);
              }
            }
            // Keep only the low halfword (2 bytes); zero the rest.
            as_.Pslldq(xn, int8_t{14});
            as_.Psrldq(xn, int8_t{14});
            break;
          }
          case 0b01: {
            // Halfwords → 32-bit sum.
            if (args.q) {
              // .8H: widen low 4 halfwords and high 4 halfwords separately,
              // then lane-wise sum (4 partial dword sums) → 2 PHADDD.
              SimdRegister xt = AllocTempSimdReg();
              if (!success()) { return; }
              if (is_signed) {
                as_.Pmovsxwd(xt, xn);          // low 4 hw → 4 dwords
                as_.Psrldq(xn, int8_t{8});
                as_.Pmovsxwd(xn, xn);          // high 4 hw → 4 dwords
              } else {
                as_.Pmovzxwd(xt, xn);
                as_.Psrldq(xn, int8_t{8});
                as_.Pmovzxwd(xn, xn);
              }
              as_.Paddd(xn, xt);               // 4 partial-sum dwords
              as_.Phaddd(xn, xn);
              as_.Phaddd(xn, xn);
            } else {
              // .4H: only widen low 4 halfwords; 2 PHADDD cascades collapse.
              if (is_signed) {
                as_.Pmovsxwd(xn, xn);
              } else {
                as_.Pmovzxwd(xn, xn);
              }
              as_.Phaddd(xn, xn);
              as_.Phaddd(xn, xn);
            }
            // Keep only the low dword (4 bytes); zero the rest.
            as_.Pslldq(xn, int8_t{12});
            as_.Psrldq(xn, int8_t{12});
            break;
          }
          case 0b10: {
            if (!args.q) { Undefined(); return; }  // .2S reserved.
            // .4S → 64-bit sum.  Widen 4 dwords → 4 qwords across two
            // registers, sum lane-wise (2 partial qword sums), then fold
            // hi-qword to lo-qword and add.
            SimdRegister xt = AllocTempSimdReg();
            if (!success()) { return; }
            if (is_signed) {
              as_.Pmovsxdq(xt, xn);            // low 2 dw → 2 qwords
              as_.Psrldq(xn, int8_t{8});
              as_.Pmovsxdq(xn, xn);            // high 2 dw → 2 qwords
            } else {
              as_.Pmovzxdq(xt, xn);
              as_.Psrldq(xn, int8_t{8});
              as_.Pmovzxdq(xn, xn);
            }
            as_.Paddq(xn, xt);                 // 2 partial-sum qwords
            as_.Pshufd(xt, xn, static_cast<int8_t>(0xEE));
            as_.Paddq(xn, xt);
            // Keep only the low qword (8 bytes); zero the rest.
            as_.Pslldq(xn, int8_t{8});
            as_.Psrldq(xn, int8_t{8});
            break;
          }
          default:
            Undefined();
            return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // SMAXV / SMINV / UMAXV / UMINV (across-lanes
      // integer max/min reduce).  Scan all source lanes; write the single
      // scalar max/min to the lowest lane of Vd with all other bytes of Vd
      // zeroed.  Result width equals esize (unlike SADDLV/UADDLV which
      // widens to 2*esize).
      //
      // Encoding (DDI 0487 §C7.2 Advanced SIMD across lanes):
      //   0 Q U 01110 size 11000 opcode 10 Rn Rd
      // SMAXV: opcode=01010, U=0.  UMAXV: opcode=01010, U=1.
      // SMINV: opcode=11010, U=0.  UMINV: opcode=11010, U=1.
      // Valid lane forms (decoder rejects size=11; JIT rejects size=10 Q=0):
      //   size=00 Q=0: V.8B   size=00 Q=1: V.16B
      //   size=01 Q=0: V.4H   size=01 Q=1: V.8H
      //   size=10 Q=1: V.4S
      //
      // Strategy: load Vn into xn.  For Q=0, replicate the low qword across
      // both halves (Pshufd 0x44) so the don't-care upper half can't poison
      // the reduction — max(x, x) = x so duplicated lanes are idempotent.
      // Then cascading PMAX/PMIN against a shuffle-of-xn collapses the
      // vector by halving the surviving lane count each step.  Final shuttle
      // (Pslldq N; Psrldq N) keeps only the low result lane.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSminv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUminv: {
        const bool is_max =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUmaxv);
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSminv);
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        SimdRegister xt = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (!args.q) {
          // Replicate low qword to high qword: { dw0, dw1, dw0, dw1 } via
          // _MM_SHUFFLE(1,0,1,0) = 0x44.  Neutralizes the don't-care upper
          // half of Vn so it can't corrupt the PMAX/PMIN cascade.
          as_.Pshufd(xn, xn, static_cast<int8_t>(0x44));
        }
        auto EmitPmaxPmin = [&](unsigned width_bits) {
          switch (width_bits) {
            case 8:
              if (is_signed) {
                if (is_max) as_.Pmaxsb(xn, xt); else as_.Pminsb(xn, xt);
              } else {
                if (is_max) as_.Pmaxub(xn, xt); else as_.Pminub(xn, xt);
              }
              break;
            case 16:
              if (is_signed) {
                if (is_max) as_.Pmaxsw(xn, xt); else as_.Pminsw(xn, xt);
              } else {
                if (is_max) as_.Pmaxuw(xn, xt); else as_.Pminuw(xn, xt);
              }
              break;
            case 32:
              if (is_signed) {
                if (is_max) as_.Pmaxsd(xn, xt); else as_.Pminsd(xn, xt);
              } else {
                if (is_max) as_.Pmaxud(xn, xt); else as_.Pminud(xn, xt);
              }
              break;
          }
        };
        switch (args.size) {
          case 0b00: {
            // Bytes: 16 → 8 → 4 → 2 → 1 surviving lanes per step.
            as_.Pshufd(xt, xn, static_cast<int8_t>(0x4E));  // swap qwords
            EmitPmaxPmin(8);
            as_.Pshufd(xt, xn, static_cast<int8_t>(0xB1));  // swap dwords within qwords
            EmitPmaxPmin(8);
            as_.Movdqa(xt, xn);
            as_.Psrldq(xt, int8_t{2});  // bring xn bytes 2..3 to xt bytes 0..1
            EmitPmaxPmin(8);
            as_.Movdqa(xt, xn);
            as_.Psrldq(xt, int8_t{1});  // bring xn byte 1 to xt byte 0
            EmitPmaxPmin(8);
            // Keep low byte; zero rest.
            as_.Pslldq(xn, int8_t{15});
            as_.Psrldq(xn, int8_t{15});
            break;
          }
          case 0b01: {
            // Halfwords: 8 → 4 → 2 → 1 surviving lanes per step.
            as_.Pshufd(xt, xn, static_cast<int8_t>(0x4E));  // swap qwords
            EmitPmaxPmin(16);
            as_.Pshufd(xt, xn, static_cast<int8_t>(0xB1));  // swap dwords within qwords
            EmitPmaxPmin(16);
            as_.Movdqa(xt, xn);
            as_.Psrldq(xt, int8_t{2});  // bring xn halfword 1 to xt halfword 0
            EmitPmaxPmin(16);
            // Keep low halfword; zero rest.
            as_.Pslldq(xn, int8_t{14});
            as_.Psrldq(xn, int8_t{14});
            break;
          }
          case 0b10: {
            if (!args.q) { Undefined(); return; }  // .2S reserved.
            // Dwords: 4 → 2 → 1 surviving lanes per step.
            as_.Pshufd(xt, xn, static_cast<int8_t>(0x4E));  // swap qwords
            EmitPmaxPmin(32);
            as_.Pshufd(xt, xn, static_cast<int8_t>(0xB1));  // swap dwords within qwords
            EmitPmaxPmin(32);
            // Keep low dword; zero rest.
            as_.Pslldq(xn, int8_t{12});
            as_.Psrldq(xn, int8_t{12});
            break;
          }
          default:
            Undefined();
            return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // BFCVTN / BFCVTN2 (vector narrow FP32 -> BF16) (§H2).
      //
      // Q=0 BFCVTN  v.4h, v.4s:  writes 4 BF16 lanes into Vd.h[0..3], upper 64 bits zeroed.
      // Q=1 BFCVTN2 v.8h, v.4s:  writes 4 BF16 lanes into Vd.h[4..7], lower 64 bits preserved.
      // Decoder pins size=10 (FP32 source).  Matches interpreter at
      // `interpreter.h:7240` and the FloatToBf16 helper at `interpreter.h:1330`.
      //
      // Per-lane algorithm (4 FP32 lanes in parallel via SSE):
      //   bits = src as uint32_t
      //   if (exp == 0xFF && mant != 0):
      //     out = (bits >> 16) | 0x0040           ; quieted NaN
      //   else:
      //     lsb = (bits >> 16) & 1                ; RTNE tie bias
      //     out = (bits + 0x7FFF + lsb) >> 16     ; round-to-nearest-even narrow
      // ±0 and ±Inf pass through the RTNE path unchanged (the +0x7FFF+lsb cannot
      // carry from mantissa into exponent when mantissa is 0).
      //
      // Constants are synthesized via PCMPEQD + shifts (no rodata).
      // No AVX-512-BF16 dependency.
      case Decoder::AdvSimdTwoRegMiscOpcode::kBfcvtn: {
        if (args.size != 0b10) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xnan_val = AllocTempSimdReg();
        SimdRegister xtmp = AllocTempSimdReg();
        SimdRegister xconst = AllocTempSimdReg();
        SimdRegister xexp = AllocTempSimdReg();
        if (!success()) { return; }

        // Load 4 FP32 lanes.
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});

        // xnan_val = (n >> 16) | 0x0040 per lane.
        as_.Movdqa(xnan_val, xn);
        as_.Psrld(xnan_val, int8_t{16});
        as_.Pcmpeqd(xconst, xconst);
        as_.Psrld(xconst, int8_t{31});            // 0x00000001 per lane (kept for reuse).
        as_.Movdqa(xtmp, xconst);
        as_.Pslld(xtmp, int8_t{6});               // 0x00000040 per lane.
        as_.Por(xnan_val, xtmp);

        // xtmp = ((n >> 16) & 1) + 0x7FFF per lane (the RTNE bias).
        as_.Movdqa(xtmp, xn);
        as_.Psrld(xtmp, int8_t{16});
        as_.Pand(xtmp, xconst);                   // lsb = (n >> 16) & 1.
        as_.Pcmpeqd(xconst, xconst);
        as_.Psrld(xconst, int8_t{17});            // 0x00007FFF per lane.
        as_.Paddd(xtmp, xconst);                  // lsb + 0x7FFF.

        // xn = (n + bias) >> 16 = rounded BF16 in low 16 bits of each lane.
        as_.Paddd(xn, xtmp);
        as_.Psrld(xn, int8_t{16});

        // xexp = nan_mask = (exp == 0xFF) & (mant != 0).
        // Reload original n into xtmp (we mutated xn into rounded).
        as_.Movdqu(xtmp, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pcmpeqd(xconst, xconst);
        as_.Psrld(xconst, int8_t{24});            // 0x000000FF per lane.
        as_.Pslld(xconst, int8_t{23});            // 0x7F800000 per lane.
        as_.Movdqa(xexp, xtmp);
        as_.Pand(xexp, xconst);                   // n & 0x7F800000.
        as_.Pcmpeqd(xexp, xconst);                // exp == 0xFF per lane (mask).
        // mant != 0 -> ones per lane in xtmp.
        as_.Pcmpeqd(xconst, xconst);
        as_.Psrld(xconst, int8_t{9});             // 0x007FFFFF per lane.
        as_.Pand(xtmp, xconst);                   // mant.
        as_.Pxor(xconst, xconst);
        as_.Pcmpeqd(xtmp, xconst);                // mant == 0 mask.
        as_.Pcmpeqd(xconst, xconst);              // all-ones.
        as_.Pxor(xtmp, xconst);                   // mant != 0.
        as_.Pand(xexp, xtmp);                     // nan_mask = exp_max & mant_nz.

        // Blend: out = (nan_mask & nan_val) | (~nan_mask & rounded).
        as_.Pand(xnan_val, xexp);                 // nan_val & nan_mask.
        as_.Movdqa(xtmp, xexp);
        as_.Pandn(xtmp, xn);                      // ~nan_mask & rounded.
        as_.Por(xnan_val, xtmp);                  // final 4 dwords (low 16 of each).

        // Narrow 4 dwords -> 4 words: PACKUSDW duplicates result to high 64.
        as_.Packusdw(xnan_val, xnan_val);

        if (!args.q) {
          // BFCVTN: zero the upper 64 bits before writing Vd.
          as_.Pslldq(xnan_val, int8_t{8});
          as_.Psrldq(xnan_val, int8_t{8});
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xnan_val);
        } else {
          // BFCVTN2: write packed result into upper 64 of Vd, preserving lower 64.
          as_.Movdqu(xtmp, {.base = Assembler::rbp, .disp = vd_off});
          as_.Pslldq(xtmp, int8_t{8});
          as_.Psrldq(xtmp, int8_t{8});            // keep low 64 of Vd, zero high 64.
          as_.Pslldq(xnan_val, int8_t{8});        // move packed bytes to high 64.
          as_.Por(xnan_val, xtmp);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xnan_val);
        }
        return;
      }
      // FMAXV/FMINV/FMAXNMV/FMINNMV — across-lanes FP reduction. The FP32
      // form is Sd, Vn.4S; the Armv8.2-FP16 form is Hd, Vn.4H (Q=0) or
      // Hd, Vn.8H (Q=1). Everything folds in FP32 space: two log2(4) folding
      // steps, each PSHUFD-ing the operand and combining, so the result lands
      // in lane 0.
      //
      // The pair step differs by NaN policy, matching the interpreter:
      //   FMAXV/FMINV     — NaN-propagating. MAXPS/MINPS on x86 return the
      //                     SECOND operand whenever either input is NaN, so a
      //                     single MAXPS would silently drop a NaN depending on
      //                     operand order. Doing it both ways and POR-ing keeps
      //                     a NaN exponent if either input was NaN, and makes
      //                     +-0 order-independent.
      //   FMAXNMV/FMINNMV — NaN-suppressing. Substitute each NaN lane with the
      //                     other operand via a CMPUNORDPS self-compare mask
      //                     before MAX/MIN, so a NaN only survives if both are.
      //
      // The FP16 forms use the F16C round-trip (VCVTPH2PS to widen, VCVTPS2PH
      // imm=0 to narrow). This is exact for a reduction: max/min returns one of
      // its inputs unchanged, and FP16 -> FP32 widening is exact for every FP16
      // encoding, so the winning FP32 lane is the exact widening of some input
      // half and narrowing it reproduces that half bit-for-bit. NaNs survive
      // the round-trip as NaNs — a widened FP16 NaN is an FP32 NaN, the POR /
      // CMPUNORDPS idioms below only ever produce an all-ones FP32 exponent
      // when an input had one, and VCVTPS2PH maps any FP32 NaN to an FP16
      // QNaN, never to an infinity.
      //
      // .2S (Q=0) is ARM-reserved for the FP32 across-lanes reductions and the
      // decoder rejects it. Mirrors the heavy frontend's lowering of the same
      // opcodes (which routes its FP16 fold through the shared FP32 min/max
      // helper).
      case Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFminv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv: {
        if (!args.is_fp16 && !args.q) { success_ = false; return; }
        if (args.is_fp16 && !host_platform::kHasF16C) { success_ = false; return; }
        const bool is_max =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv);
        const bool is_nm =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv);
        // All temps are allocated once, before the folding steps: the pair step
        // runs twice (three times for .8H), and allocating inside it would take
        // a fresh register from the pool each time for no reason.
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xt = AllocTempSimdReg();
        SimdRegister xa = AllocTempSimdReg();
        SimdRegister xb = is_nm ? AllocTempSimdReg() : xa;
        SimdRegister a_sub = is_nm ? AllocTempSimdReg() : xa;
        SimdRegister b_sub = is_nm ? AllocTempSimdReg() : xa;
        // .8H widens its two half-lane groups into separate registers, so it
        // needs one temp more than .4S / .4H do.
        SimdRegister xh = (args.is_fp16 && args.q) ? AllocTempSimdReg() : xn;
        if (!success()) { return; }
        // Fold b into a with packed FP32 min/max under the selected NaN rule.
        auto emit_pair = [&](SimdRegister a, SimdRegister b) {
          if (!is_nm) {
            as_.Movdqa(xa, b);
            if (is_max) {
              as_.Maxps(xa, a);
              as_.Maxps(a, b);
            } else {
              as_.Minps(xa, a);
              as_.Minps(a, b);
            }
            as_.Por(a, xa);
          } else {
            as_.Movdqa(xa, a);
            as_.Movdqa(xb, b);
            as_.Cmpunordps(xa, xa);   // mask: lanes where a is NaN
            as_.Cmpunordps(xb, xb);   // mask: lanes where b is NaN
            as_.Movdqa(a_sub, xa);
            as_.Pand(a_sub, b);       // a NaN -> take b
            as_.Movdqa(b_sub, xb);
            as_.Pand(b_sub, a);       // b NaN -> take a
            as_.Pandn(xa, a);         // non-NaN lanes of a
            as_.Pandn(xb, b);         // non-NaN lanes of b
            as_.Por(xa, a_sub);
            as_.Por(xb, b_sub);
            if (is_max) {
              as_.Maxps(xa, xb);
            } else {
              as_.Minps(xa, xb);
            }
            as_.Movdqa(a, xa);
          }
        };
        if (!args.is_fp16) {
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        } else if (!args.q) {
          // .4H: the four half-lanes occupy the low 64 bits of Vn. VCVTPH2PS
          // reads exactly those and produces the four FP32 lanes the fold
          // below expects.
          as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Vcvtph2ps(xn, xn);
        } else {
          // .8H: widen the low and high half-lane groups into two registers,
          // then fold the high group into the low one with the same pair step.
          // That lane-wise fold reduces 8 halves to 4 FP32 lanes holding
          // max/min(lane i, lane i+4), and the two-step fold below finishes
          // the reduction.
          as_.Movdqu(xh, {.base = Assembler::rbp, .disp = vn_off});
          as_.Movdqa(xn, xh);
          as_.Vcvtph2ps(xn, xn);
          as_.Psrldq(xh, int8_t{8});
          as_.Vcvtph2ps(xh, xh);
          emit_pair(xn, xh);
        }
        // Step 1: fold lanes {2,3} into {0,1} (PSHUFD 0x4E swaps 64-bit halves).
        as_.Pshufd(xt, xn, static_cast<int8_t>(0x4E));
        emit_pair(xn, xt);
        // Step 2: fold lane 1 into lane 0 (PSHUFD 0xB1 swaps adjacent dwords).
        as_.Pshufd(xt, xn, static_cast<int8_t>(0xB1));
        emit_pair(xn, xt);
        if (args.is_fp16) {
          // Narrow the winner back to FP16. VCVTPS2PH packs the four FP32
          // lanes into the low 64 bits, so the scalar result is at byte 0;
          // shifting left then right by 14 bytes keeps those two bytes and
          // zeroes the upper 112 bits of Hd.
          as_.Vcvtps2ph(xn, xn, int8_t{0});
          as_.Pslldq(xn, int8_t{14});
          as_.Psrldq(xn, int8_t{14});
        } else {
          // Scalar result: keep lane 0, zero the upper 96 bits.
          as_.Pslldq(xn, int8_t{12});
          as_.Psrldq(xn, int8_t{12});
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }

      default:
        Undefined();
        return;
    }
  }

