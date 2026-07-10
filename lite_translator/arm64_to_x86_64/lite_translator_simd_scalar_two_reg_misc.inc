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
// Family: SIMD scalar two-register-misc.
// Methods:
//   AdvSimdScalarTwoRegMisc

  //
  // AdvSimd scalar two-register-misc JIT dispatch.
  //
  // Covers the AdvSimd-class single-lane FP/integer conversions whose
  // destination is a vector register's bottom S/D lane with the upper
  // 96/64 bits zeroed:
  //
  //   SCVTF Sd, Sn / SCVTF Dd, Dn        signed   int → FP   (size&1: 0=S, 1=D)
  //   UCVTF Sd, Sn / UCVTF Dd, Dn        unsigned int → FP
  //   FCVTZS Sd, Sn / FCVTZS Dd, Dn      FP → signed   int, round toward zero
  //   FCVTZU Sd, Sn / FCVTZU Dd, Dn      FP → unsigned int, round toward zero
  //   FCVTAS Sd, Sn / FCVTAS Dd, Dn      FP → signed   int, round-to-nearest ties-away
  //   FCVTAU Sd, Sn / FCVTAU Dd, Dn      FP → unsigned int, round-to-nearest ties-away
  //   FCVTXN Sd, Dn                      FP64 → FP32 narrow, round-to-odd
  //   SQABS  Bd/Hd/Sd/Dd, Bn/Hn/Sn/Dn    signed saturating absolute value
  //   SQNEG  Bd/Hd/Sd/Dd, Bn/Hn/Sn/Dn    signed saturating negate
  //   SQXTN  Bd, Hn / Hd, Sn / Sd, Dn    signed saturating extract narrow
  //
  // The FP-to-int and int-to-FP single-lane lowerings mirror the per-lane
  // variant of the corresponding vector kFcvtzsV / kFcvtzuV / kFcvtasV /
  // kFcvtauV / kScvtfV / kUcvtfV emission above; the only differences are
  // that we work on the bottom lane directly (Movsd/Movss) and explicitly
  // zero Vd above the lane before writing the single-precision/double
  // result.  FCVTXN uses a one-shot MXCSR round-toward-zero conversion
  // plus LSB-OR fix-up to realize the ARM "Round to Odd" mode.  SQABS /
  // SQNEG share a GP-register lowering: sign-extend the source, NEG into
  // a paired temp; SQABS picks abs(x) via CMOVS, SQNEG unconditionally
  // uses the negated value; then CMP+CMOVZ saturates INT_MIN → INT_MAX
  // (the single saturating input case for both ops).  SQXTN sign-extends
  // the 2× lane src to int64 and clamps with two CMOV arms (CMOVG to
  // INT_MAX_dst, CMOVL to INT_MIN_dst) before storing the dst-width
  // result at lane 0.
  //
  // Other constituents of this dispatch class (SQXTUN/UQXTN/FRECPE/
  // FRSQRTE scalar) bail to the interpreter via success_=false — they are
  // JIT follow-ups; the interpreter handles them correctly today.
  void AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs& args) {
    using Opcode = Decoder::AdvSimdScalarTwoRegMiscOpcode;

    const int32_t vn_off = VRegOffset(args.rn);
    const int32_t vd_off = VRegOffset(args.rd);

    switch (args.opcode) {
      case Opcode::kScvtf:
      case Opcode::kUcvtf: {
        // SCVTF / UCVTF: integer → FP, written to bottom S/D lane of Vd.
        const bool is_unsigned = (args.opcode == Opcode::kUcvtf);
        const bool is_double = ((args.size & 1) != 0);
        SimdRegister xmm = AllocTempSimdReg();
        SimdRegister xzero = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        if (xmm == no_simd_register || xzero == no_simd_register ||
            tmp == no_register) {
          success_ = false; return;
        }
        as_.Pxor(xmm, xmm);  // clear xmm so legacy CVTSI2{SS,SD} writes a clean lane.
        if (is_double) {
          // 64-bit integer source.
          as_.Movq(tmp, {.base = Assembler::rbp, .disp = vn_off});
          if (!is_unsigned) {
            // SCVTF Dd, Dn: signed int64 → FP64.
            as_.Cvtsi2sdq(xmm, tmp);
          } else {
            // UCVTF Dd, Dn: u64 → FP64.  Halve-OR-LSB round-to-odd trick
            // for inputs with bit63 set (mirrors the scalar UCVTF Xd Dn
            // and vector UCVTF V .2D lowerings).
            Register low_bit = AllocTempReg();
            if (low_bit == no_register) { success_ = false; return; }
            Assembler::Label* neg_path = as_.MakeLabel();
            Assembler::Label* done = as_.MakeLabel();
            as_.Testq(tmp, tmp);
            as_.Jcc(Assembler::Condition::kNegative, *neg_path);
            as_.Cvtsi2sdq(xmm, tmp);
            as_.Jmp(*done);
            as_.Bind(neg_path);
            as_.Movq(low_bit, tmp);
            as_.Andq(low_bit, int32_t{1});
            as_.Shrq(tmp, int8_t{1});
            as_.Orq(tmp, low_bit);
            as_.Cvtsi2sdq(xmm, tmp);
            as_.Addsd(xmm, xmm);
            as_.Bind(done);
          }
        } else {
          // 32-bit integer source.
          if (!is_unsigned) {
            // SCVTF Sd, Sn: signed int32 → FP32.  Movl zero-extends, then
            // Cvtsi2ssl reads the 32-bit subreg as int32.
            as_.Movl(tmp, {.base = Assembler::rbp, .disp = vn_off});
            as_.Cvtsi2ssl(xmm, tmp);
          } else {
            // UCVTF Sd, Sn: u32 → FP32.  Zero-extend via Movl and convert as
            // signed int64 (value ≤ UINT32_MAX < INT64_MAX, no overflow).
            as_.Movl(tmp, {.base = Assembler::rbp, .disp = vn_off});
            as_.Cvtsi2ssq(xmm, tmp);
          }
        }
        // Zero Vd above the result lane, then write the scalar lane.
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        if (is_double) {
          as_.Movsd({.base = Assembler::rbp, .disp = vd_off}, xmm);
        } else {
          as_.Movss({.base = Assembler::rbp, .disp = vd_off}, xmm);
        }
        return;
      }

      case Opcode::kFcvtzs: {
        // FCVTZS Sd, Sn / FCVTZS Dd, Dn: FP → signed int, round toward zero.
        // Per-lane mirror of the kFcvtzsV vector path with single-lane store
        // (lane 0 of Vd, upper bytes zeroed).
        //
        // ARM saturation rules:
        //   NaN              -> 0
        //   positive overflow -> INT_MAX
        //   negative overflow -> INT_MIN (matches x86 cvtt indefinite)
        //   in-range          -> trunc toward zero
        const bool is_double = ((args.size & 1) != 0);
        SimdRegister xmm = AllocTempSimdReg();
        SimdRegister xzero = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        Register sign_tmp = AllocTempReg();
        if (xmm == no_simd_register || xzero == no_simd_register ||
            tmp == no_register || sign_tmp == no_register) {
          success_ = false; return;
        }
        if (is_double) {
          as_.Movsd(xmm, {.base = Assembler::rbp, .disp = vn_off});
          as_.Cvttsd2siq(tmp, xmm);
        } else {
          as_.Movss(xmm, {.base = Assembler::rbp, .disp = vn_off});
          as_.Cvttss2sil(tmp, xmm);
        }
        Assembler::Label* nan_path = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        if (is_double) as_.Ucomisd(xmm, xmm);
        else as_.Ucomiss(xmm, xmm);
        as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
        if (is_double) {
          as_.Movq(sign_tmp, xmm);
          as_.Testq(sign_tmp, sign_tmp);
        } else {
          as_.Movd(sign_tmp, xmm);
          as_.Testl(sign_tmp, sign_tmp);
        }
        as_.Jcc(Assembler::Condition::kNegative, *done);
        if (is_double) as_.Testq(tmp, tmp);
        else as_.Testl(tmp, tmp);
        as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
        if (is_double) as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
        else as_.Movl(tmp, int32_t{INT32_MAX});
        as_.Jmp(*done);
        as_.Bind(nan_path);
        if (is_double) as_.Xorq(tmp, tmp);
        else as_.Xorl(tmp, tmp);
        as_.Bind(done);
        // Zero Vd above the result lane, then write.
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        if (is_double) {
          as_.Movq({.base = Assembler::rbp, .disp = vd_off}, tmp);
        } else {
          as_.Movl({.base = Assembler::rbp, .disp = vd_off}, tmp);
        }
        return;
      }

      case Opcode::kFcvtzu: {
        // FCVTZU Sd, Sn / FCVTZU Dd, Dn: FP → unsigned int, round toward zero.
        // ARM saturation rules:
        //   NaN              -> 0
        //   FP < 0 (incl -0) -> 0
        //   FP > UMAX        -> UMAX
        //   in-range         -> trunc(FP) as unsigned
        //
        // FP32 → u32 path: cvtt-Q gives int64; valid u32 outputs have upper
        // 32 bits zero, so an upper-32 test classifies overflow.  FP64 → u64
        // path: cvtt-Q covers [0, 2^63); for [2^63, 2^64) use the
        // "subtract 2^63, cvtt, OR bit63" trick; FP ≥ 2^64 saturates to
        // UINT64_MAX.
        const bool is_double = ((args.size & 1) != 0);
        SimdRegister xmm = AllocTempSimdReg();
        SimdRegister xzero = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        Register sign_tmp = AllocTempReg();
        if (xmm == no_simd_register || xzero == no_simd_register ||
            tmp == no_register || sign_tmp == no_register) {
          success_ = false; return;
        }
        if (is_double) {
          // FP64 → u64.
          SimdRegister bound_xmm = AllocTempSimdReg();
          SimdRegister bound2_xmm = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movsd(xmm, {.base = Assembler::rbp, .disp = vn_off});
          // 2^63, 2^64 constants in FP64.
          as_.Movq(tmp, int64_t{0x43E0000000000000LL});
          as_.Movq(bound_xmm, tmp);
          as_.Movq(tmp, int64_t{0x43F0000000000000LL});
          as_.Movq(bound2_xmm, tmp);

          Assembler::Label* zero_path = as_.MakeLabel();
          Assembler::Label* direct_path = as_.MakeLabel();
          Assembler::Label* sat_max = as_.MakeLabel();
          Assembler::Label* done = as_.MakeLabel();
          // NaN → 0.
          as_.Ucomisd(xmm, xmm);
          as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
          // FP < 0 → 0 (sign bit set; -0.0 falls through to direct_path
          // where cvtt(-0.0)=0 is also correct).
          as_.Movq(sign_tmp, xmm);
          as_.Testq(sign_tmp, sign_tmp);
          as_.Jcc(Assembler::Condition::kNegative, *zero_path);
          // FP < 2^63 → direct cvttsd2siq is exact.
          as_.Ucomisd(xmm, bound_xmm);
          as_.Jcc(Assembler::Condition::kBelow, *direct_path);
          // FP ≥ 2^64 → saturate to UINT64_MAX.
          as_.Ucomisd(xmm, bound2_xmm);
          as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);
          // [2^63, 2^64): subtract 2^63, cvtt, OR bit63.
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
        } else {
          // FP32 → u32 via cvtt-Q (int64), classify by upper-32 zero-ness.
          as_.Movss(xmm, {.base = Assembler::rbp, .disp = vn_off});
          Assembler::Label* zero_path = as_.MakeLabel();
          Assembler::Label* done = as_.MakeLabel();
          as_.Ucomiss(xmm, xmm);
          as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
          // FP sign-bit check via raw bits.  -0.0 falls through; cvtt(-0.0)=0.
          as_.Movd(sign_tmp, xmm);
          as_.Testl(sign_tmp, sign_tmp);
          as_.Jcc(Assembler::Condition::kNegative, *zero_path);
          // cvtt-Q to int64; valid u32 outputs have upper-32 = 0.
          as_.Cvttss2siq(tmp, xmm);
          // sign_tmp = upper-32 of tmp; nonzero ⇒ overflow.
          as_.Movq(sign_tmp, tmp);
          as_.Shrq(sign_tmp, int8_t{32});
          as_.Testl(sign_tmp, sign_tmp);
          Assembler::Label* sat_max = as_.MakeLabel();
          as_.Jcc(Assembler::Condition::kNotZero, *sat_max);
          // tmp's bottom 32 bits already correct; truncate via Movl into
          // tmp (auto-zero-extends to 64).
          as_.Movl(tmp, tmp);
          as_.Jmp(*done);
          as_.Bind(sat_max);
          as_.Movl(tmp, int32_t{-1});  // 0xFFFFFFFF
          as_.Jmp(*done);
          as_.Bind(zero_path);
          as_.Xorl(tmp, tmp);
          as_.Bind(done);
        }
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        if (is_double) {
          as_.Movq({.base = Assembler::rbp, .disp = vd_off}, tmp);
        } else {
          as_.Movl({.base = Assembler::rbp, .disp = vd_off}, tmp);
        }
        return;
      }

      case Opcode::kFcvtas:
      case Opcode::kFcvtau: {
        // FCVTAS / FCVTAU scalar: FP → int, round-to-nearest ties-away.
        // Strategy: FRINTA dance (add sign(x)*0.5 when |x| < threshold, then
        // ROUND/CVTT toward zero); then apply the same saturation fix-up as
        // FCVTZS/FCVTZU.
        //
        // SSE4.2 required for the ROUNDSS/ROUNDSD with imm=3 (truncate +
        // suppress-inexact) used after the addend.
        if (!host_platform::kHasSSE4_2) { success_ = false; return; }
        const bool is_unsigned = (args.opcode == Opcode::kFcvtau);
        const bool is_double = ((args.size & 1) != 0);

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
        if (is_double) {
          as_.Movsd(xn, {.base = Assembler::rbp, .disp = vn_off});
        } else {
          as_.Movss(xn, {.base = Assembler::rbp, .disp = vn_off});
        }
        // |bits(xn)| = xn AND 0x7FFF... (sign-bit-cleared).
        as_.Pcmpeqd(abs_bits, abs_bits);
        if (is_double) as_.Psrlq(abs_bits, int8_t{1});  // 0x7FFFFFFFFFFFFFFF
        else as_.Psrld(abs_bits, int8_t{1});             // 0x7FFFFFFF
        as_.Pand(abs_bits, xn);
        // sign(xn) | 0.5.
        as_.Pcmpeqd(copysign, copysign);
        if (is_double) as_.Psllq(copysign, int8_t{63});  // 0x8000000000000000
        else as_.Pslld(copysign, int8_t{31});             // 0x80000000
        as_.Pand(copysign, xn);
        if (is_double) {
          as_.Movq(gp_half, int64_t{0x3FE0000000000000LL});  // 0.5 (FP64)
        } else {
          as_.Movl(gp_half, int32_t{0x3F000000});             // 0.5 (FP32)
        }
        if (is_double) as_.Movq(half, gp_half);
        else as_.Movd(half, gp_half);
        as_.Por(copysign, half);
        // Mask gate: if |x| < threshold, keep addend; else zero it.
        // Threshold: 2^52 for FP64, 2^23 for FP32 — beyond that, the value
        // is already integer so the addend would corrupt it.
        if (is_double) {
          as_.Movq(gp_half, int64_t{0x4330000000000000LL});  // 2^52
        } else {
          as_.Movl(gp_half, int32_t{0x4B000000});             // 2^23
        }
        if (is_double) as_.Movq(half, gp_half);
        else as_.Movd(half, gp_half);
        // Compare lane: 1s where |x| < threshold (signed compare safe; both
        // are non-negative bit patterns at this point).
        if (is_double) as_.Pcmpgtq(half, abs_bits);
        else as_.Pcmpgtd(half, abs_bits);
        as_.Pand(copysign, half);
        // xn += copysign.
        if (is_double) as_.Addpd(xn, copysign);
        else as_.Addps(xn, copysign);
        // ROUNDSx imm=3: truncate toward zero, suppress inexact.
        if (is_double) as_.Roundsd(xn, xn, int8_t{0x03});
        else as_.Roundss(xn, xn, int8_t{0x03});

        // Now apply FCVTZS/FCVTZU-style saturation on the single-lane result.
        SimdRegister xzero = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        Register sign_tmp = AllocTempReg();
        if (xzero == no_simd_register || tmp == no_register ||
            sign_tmp == no_register) {
          success_ = false; return;
        }
        if (!is_unsigned) {
          // FCVTAS: signed.  Mirror of the FCVTZS lowering above.
          if (is_double) as_.Cvttsd2siq(tmp, xn);
          else as_.Cvttss2sil(tmp, xn);
          Assembler::Label* nan_path = as_.MakeLabel();
          Assembler::Label* done = as_.MakeLabel();
          if (is_double) as_.Ucomisd(xn, xn);
          else as_.Ucomiss(xn, xn);
          as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
          if (is_double) {
            as_.Movq(sign_tmp, xn);
            as_.Testq(sign_tmp, sign_tmp);
          } else {
            as_.Movd(sign_tmp, xn);
            as_.Testl(sign_tmp, sign_tmp);
          }
          as_.Jcc(Assembler::Condition::kNegative, *done);
          if (is_double) as_.Testq(tmp, tmp);
          else as_.Testl(tmp, tmp);
          as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
          if (is_double) as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
          else as_.Movl(tmp, int32_t{INT32_MAX});
          as_.Jmp(*done);
          as_.Bind(nan_path);
          if (is_double) as_.Xorq(tmp, tmp);
          else as_.Xorl(tmp, tmp);
          as_.Bind(done);
        } else if (is_double) {
          // FCVTAU Dd, Dn: unsigned u64.  Mirror of the FP64 FCVTZU path.
          SimdRegister bound_xmm = AllocTempSimdReg();
          SimdRegister bound2_xmm = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movq(tmp, int64_t{0x43E0000000000000LL});
          as_.Movq(bound_xmm, tmp);
          as_.Movq(tmp, int64_t{0x43F0000000000000LL});
          as_.Movq(bound2_xmm, tmp);

          Assembler::Label* zero_path = as_.MakeLabel();
          Assembler::Label* direct_path = as_.MakeLabel();
          Assembler::Label* sat_max = as_.MakeLabel();
          Assembler::Label* done = as_.MakeLabel();
          as_.Ucomisd(xn, xn);
          as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
          as_.Movq(sign_tmp, xn);
          as_.Testq(sign_tmp, sign_tmp);
          as_.Jcc(Assembler::Condition::kNegative, *zero_path);
          as_.Ucomisd(xn, bound_xmm);
          as_.Jcc(Assembler::Condition::kBelow, *direct_path);
          as_.Ucomisd(xn, bound2_xmm);
          as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);
          as_.Subsd(xn, bound_xmm);
          as_.Cvttsd2siq(tmp, xn);
          as_.Btsq(tmp, int8_t{63});
          as_.Jmp(*done);
          as_.Bind(sat_max);
          as_.Movq(tmp, static_cast<int64_t>(-1));
          as_.Jmp(*done);
          as_.Bind(direct_path);
          as_.Cvttsd2siq(tmp, xn);
          as_.Jmp(*done);
          as_.Bind(zero_path);
          as_.Xorq(tmp, tmp);
          as_.Bind(done);
        } else {
          // FCVTAU Sd, Sn: unsigned u32.  Mirror of the FP32 FCVTZU path.
          Assembler::Label* zero_path = as_.MakeLabel();
          Assembler::Label* done = as_.MakeLabel();
          as_.Ucomiss(xn, xn);
          as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
          as_.Movd(sign_tmp, xn);
          as_.Testl(sign_tmp, sign_tmp);
          as_.Jcc(Assembler::Condition::kNegative, *zero_path);
          as_.Cvttss2siq(tmp, xn);
          as_.Movq(sign_tmp, tmp);
          as_.Shrq(sign_tmp, int8_t{32});
          as_.Testl(sign_tmp, sign_tmp);
          Assembler::Label* sat_max = as_.MakeLabel();
          as_.Jcc(Assembler::Condition::kNotZero, *sat_max);
          as_.Movl(tmp, tmp);
          as_.Jmp(*done);
          as_.Bind(sat_max);
          as_.Movl(tmp, int32_t{-1});
          as_.Jmp(*done);
          as_.Bind(zero_path);
          as_.Xorl(tmp, tmp);
          as_.Bind(done);
        }
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        if (is_double) {
          as_.Movq({.base = Assembler::rbp, .disp = vd_off}, tmp);
        } else {
          as_.Movl({.base = Assembler::rbp, .disp = vd_off}, tmp);
        }
        return;
      }

      case Opcode::kFcvtxn: {
        // FCVTXN Sd, Dn — single-lane FP64 -> FP32 narrow with the ARMv8
        // "Round to Odd" mode.  Decoder pins args.size == 0b01 (FP64
        // source).  RtO maps: overflow -> +/-FP32_MAX, NaN -> quiet FP32
        // NaN, +/-Inf -> +/-Inf, +/-0 -> +/-0; otherwise rounds toward
        // zero then forces the FP32 LSB to 1 if any FP64 mantissa bits
        // were discarded.  See Interpreter::FpDoubleToFloatRtO for the
        // reference semantics.
        //
        // Strategy: clear MXCSR's six exception bits and set RC=RTZ,
        // run CVTSD2SS, then read the new PE bit.  Under RC=RTZ
        // CVTSD2SS produces the round-toward-zero FP32 (or +/-FP32_MAX
        // for overflow, since RTZ clamps overflow magnitude to the
        // largest finite), matching the RtO base value.  If the PE
        // bit is set the result was inexact -- force the FP32 LSB to
        // 1 to make it odd.  The overflow saturation +/-FP32_MAX has
        // LSB=1 already (0x{7F,FF}7FFFFF), so the OR there is a no-op.
        //
        // MXCSR cumulative state is preserved across this op: we save
        // the original MXCSR, run with cleared exceptions + RC=RTZ,
        // then OR the per-op exception bits back into the saved value
        // and reload.  MXCSR exception bits are caller-saved per
        // System V x86_64 ABI 3.2.1 (see CLAUDE.md), so the 4-byte
        // scratch slot at [rsp+0] used by the region-exit mirror is
        // safe to use here mid-region.
        SimdRegister xmm = AllocTempSimdReg();
        SimdRegister xzero = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        Register mxcsr_save = AllocTempReg();
        Register mxcsr_after = AllocTempReg();
        if (xmm == no_simd_register || xzero == no_simd_register ||
            tmp == no_register || mxcsr_save == no_register ||
            mxcsr_after == no_register) {
          success_ = false; return;
        }
        as_.Movsd(xmm, {.base = Assembler::rbp, .disp = vn_off});
        // Save MXCSR.
        as_.Stmxcsr({.base = Assembler::rsp, .disp = 0});
        as_.Movl(mxcsr_save, {.base = Assembler::rsp, .disp = 0});
        // Build modified MXCSR: clear bits 0-5 (exception flags),
        // set bits 13-14 (RC=11=Round-Toward-Zero).
        as_.Movl(mxcsr_after, mxcsr_save);
        as_.Andl(mxcsr_after, int32_t{~int32_t{0x3F}});
        as_.Orl(mxcsr_after, int32_t{0x6000});
        as_.Movl({.base = Assembler::rsp, .disp = 0}, mxcsr_after);
        as_.Ldmxcsr({.base = Assembler::rsp, .disp = 0});
        // FP64 -> FP32 conversion under RC=RTZ.
        as_.Cvtsd2ss(xmm, xmm);
        // Read post-op MXCSR; isolate this op's exception bits.
        as_.Stmxcsr({.base = Assembler::rsp, .disp = 0});
        as_.Movl(mxcsr_after, {.base = Assembler::rsp, .disp = 0});
        as_.Andl(mxcsr_after, int32_t{0x3F});
        // Restore cumulative MXCSR: saved | per-op exceptions.
        // This preserves the original RC and ORs the per-op
        // exception bits into the prior cumulative state, matching
        // what the region-exit mirror expects.
        as_.Orl(mxcsr_save, mxcsr_after);
        as_.Movl({.base = Assembler::rsp, .disp = 0}, mxcsr_save);
        as_.Ldmxcsr({.base = Assembler::rsp, .disp = 0});
        // If the per-op PE (bit 5) was set the result was inexact;
        // OR the FP32 LSB with 1.  (Overflow's FP32_MAX already has
        // LSB=1, so the OR is a no-op there; NaN/Inf/exact paths
        // keep PE=0 and skip the fix-up entirely.)
        Assembler::Label* done = as_.MakeLabel();
        as_.Testl(mxcsr_after, int32_t{0x20});
        as_.Jcc(Assembler::Condition::kZero, *done);
        as_.Movd(tmp, xmm);
        as_.Orl(tmp, int32_t{1});
        as_.Movd(xmm, tmp);
        as_.Bind(done);
        // Store with upper 96 bits of Vd zeroed.
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        as_.Movss({.base = Assembler::rbp, .disp = vd_off}, xmm);
        return;
      }

      case Opcode::kSqabs:
      case Opcode::kSqneg: {
        // SQABS / SQNEG scalar: saturating signed absolute value / negate.
        // size: 00=B (8-bit), 01=H (16-bit), 10=S (32-bit), 11=D (64-bit).
        // Per-lane semantics (both share the saturation step):
        //   if src == INT_MIN_for_width -> INT_MAX_for_width (saturation)
        //   else if SQNEG               -> -src
        //   else (SQABS)                -> |src|
        // Result is written to lane 0 of Vd, upper bytes zeroed.  See
        // Interpreter::AdvSimdScalarTwoRegMisc::{kSqabs,kSqneg} for the
        // reference semantics (matched by the host exec tests below).
        //
        // Strategy: sign-extend src to int64, compute negated = -src in a
        // sibling temp.  SQABS picks abs(x) via CMOVS on the sign of src;
        // SQNEG unconditionally takes negated.  Then CMPQ src,INT_MIN_w +
        // CMOVZ saturates the single saturating input case to INT_MAX_w.
        // The intermediate result fits in int64 even for D-width because
        // |INT64_MIN| (and -INT64_MIN) both overflow to INT64_MIN (i.e.
        // stays in the "saturate" bucket selected by the CMP).
        const bool is_neg = (args.opcode == Opcode::kSqneg);
        const uint8_t sz = static_cast<uint8_t>(args.size & 0x3);
        int64_t int_min;
        int64_t int_max;
        switch (sz) {
          case 0: int_min = INT8_MIN;  int_max = INT8_MAX;  break;
          case 1: int_min = INT16_MIN; int_max = INT16_MAX; break;
          case 2: int_min = INT32_MIN; int_max = INT32_MAX; break;
          default:
            // sz == 3 (D-width).
            int_min = INT64_MIN; int_max = INT64_MAX; break;
        }
        Register src = AllocTempReg();
        Register negated = AllocTempReg();
        Register result = AllocTempReg();
        Register int_min_reg = AllocTempReg();
        Register int_max_reg = AllocTempReg();
        SimdRegister xzero = AllocTempSimdReg();
        if (src == no_register || negated == no_register ||
            result == no_register || int_min_reg == no_register ||
            int_max_reg == no_register || xzero == no_simd_register) {
          success_ = false; return;
        }
        // Load src, sign-extended to 64-bit.
        switch (sz) {
          case 0:
            as_.Movsxbq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          case 1:
            as_.Movsxwq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          case 2:
            as_.Movsxlq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          default:
            as_.Movq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
        }
        // negated = -src.  Negq clobbers EFLAGS.
        as_.Movq(negated, src);
        as_.Negq(negated);
        if (is_neg) {
          // SQNEG: always take the negated value.
          as_.Movq(result, negated);
        } else {
          // SQABS: result starts as src; if src < 0, result = -src.  Testq
          // sets SF from src's sign bit.
          as_.Movq(result, src);
          as_.Testq(src, src);
          as_.Cmovq(Assembler::Condition::kNegative, result, negated);
        }
        // Saturation: if src == INT_MIN_for_width, result = INT_MAX_for_width.
        as_.Movq(int_min_reg, int_min);
        as_.Cmpq(src, int_min_reg);
        as_.Movq(int_max_reg, int_max);  // Movq imm does not touch EFLAGS.
        as_.Cmovq(Assembler::Condition::kZero, result, int_max_reg);
        // Zero Vd, then store result at lane 0 at the right width.
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        switch (sz) {
          case 0:
            as_.Movb({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          case 1:
            as_.Movw({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          case 2:
            as_.Movl({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          default:
            as_.Movq({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
        }
        return;
      }

      case Opcode::kUqxtn: {
        // UQXTN scalar: unsigned saturating extract narrow, single-lane.
        // size: 00 → 8-bit dst from 16-bit src, 01 → 16-bit dst from 32-bit
        // src, 10 → 32-bit dst from 64-bit src; size=11 is rejected by the
        // decoder.  Per-lane semantics: clamp unsigned src to [0, UMAX_dst]
        // and truncate to dst width.  Result is written to lane 0 of Vd with
        // the upper bytes zeroed.  See
        // Interpreter::AdvSimdScalarTwoRegMisc::kUqxtn for the reference
        // semantics (matched by the host exec tests below).
        //
        // Strategy: load src zero-extended to uint64 at the 2× lane (src)
        // width, then clamp with a single CMOV arm — CMOVA (unsigned above)
        // against UMAX_dst.  Lower bound is implicit (zero-extended load
        // produces a non-negative value).
        const uint8_t sz = static_cast<uint8_t>(args.size & 0x3);
        if (sz == 3) { success_ = false; return; }
        int64_t uint_max;
        switch (sz) {
          case 0: uint_max = 0xFF;       break;  // UINT8_MAX
          case 1: uint_max = 0xFFFF;     break;  // UINT16_MAX
          default:
            // sz == 2: dst is S (32-bit), src is D (64-bit).
            uint_max = 0xFFFFFFFFLL; break;     // UINT32_MAX
        }
        Register src = AllocTempReg();
        Register result = AllocTempReg();
        Register bound = AllocTempReg();
        SimdRegister xzero = AllocTempSimdReg();
        if (src == no_register || result == no_register ||
            bound == no_register || xzero == no_simd_register) {
          success_ = false; return;
        }
        // Load src zero-extended to 64-bit, at the 2× lane src width.
        switch (sz) {
          case 0:
            // src is H (16-bit).
            as_.Movzxwq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          case 1:
            // src is S (32-bit).  Movl on a 64-bit dst zero-extends.
            as_.Movl(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          default:
            // src is D (64-bit); already zero-extended (full width).
            as_.Movq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
        }
        // result = src (CMOV may override below).
        as_.Movq(result, src);
        // Upper-bound clamp: if (unsigned) src > UMAX_dst, result = UMAX_dst.
        as_.Movq(bound, uint_max);
        as_.Cmpq(src, bound);
        as_.Cmovq(Assembler::Condition::kAbove, result, bound);
        // Zero Vd, then store result at lane 0 at the dst width.
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        switch (sz) {
          case 0:
            as_.Movb({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          case 1:
            as_.Movw({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          default:
            as_.Movl({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
        }
        return;
      }

      case Opcode::kSqxtn: {
        // SQXTN scalar: signed saturating extract narrow, single-lane.
        // size: 00 → 8-bit dst from 16-bit src, 01 → 16-bit dst from 32-bit
        // src, 10 → 32-bit dst from 64-bit src; size=11 is rejected by the
        // decoder.  Per-lane semantics: clamp signed src to
        // [INT_MIN_dst, INT_MAX_dst] and truncate to dst width.  Result is
        // written to lane 0 of Vd with the upper bytes zeroed.  See
        // Interpreter::AdvSimdScalarTwoRegMisc::kSqxtn for the reference
        // semantics (matched by the host exec tests below).
        //
        // Strategy: load src sign-extended to int64 at the 2× lane (src)
        // width, then clamp with two CMOV arms — CMOVG against INT_MAX_dst
        // and CMOVL against INT_MIN_dst.  Movq imm64 between the CMP/CMOV
        // pairs does not touch EFLAGS, so the second comparison sees the
        // original src vs INT_MIN_dst correctly.  If src exceeds the
        // positive bound, the CMOVG sets result=INT_MAX_dst and the
        // subsequent CMOVL is harmless (src > INT_MAX_dst > INT_MIN_dst).
        const uint8_t sz = static_cast<uint8_t>(args.size & 0x3);
        if (sz == 3) { success_ = false; return; }
        int64_t int_min;
        int64_t int_max;
        switch (sz) {
          case 0: int_min = INT8_MIN;  int_max = INT8_MAX;  break;
          case 1: int_min = INT16_MIN; int_max = INT16_MAX; break;
          default:
            // sz == 2: dst is S (32-bit), src is D (64-bit).
            int_min = INT32_MIN; int_max = INT32_MAX; break;
        }
        Register src = AllocTempReg();
        Register result = AllocTempReg();
        Register bound = AllocTempReg();
        SimdRegister xzero = AllocTempSimdReg();
        if (src == no_register || result == no_register ||
            bound == no_register || xzero == no_simd_register) {
          success_ = false; return;
        }
        // Load src sign-extended to 64-bit, at the 2× lane src width.
        switch (sz) {
          case 0:
            // src is H (16-bit).
            as_.Movsxwq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          case 1:
            // src is S (32-bit).
            as_.Movsxlq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          default:
            // src is D (64-bit); already signed at int64 width.
            as_.Movq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
        }
        // result = src (clamping arms may override below).
        as_.Movq(result, src);
        // Upper-bound clamp: if src > INT_MAX_dst, result = INT_MAX_dst.
        as_.Movq(bound, int_max);
        as_.Cmpq(src, bound);
        as_.Cmovq(Assembler::Condition::kGreater, result, bound);
        // Lower-bound clamp: if src < INT_MIN_dst, result = INT_MIN_dst.
        // Movq imm64 does not affect EFLAGS, so we still have a valid
        // src-vs-INT_MAX_dst comparison until the next Cmpq sets new flags.
        as_.Movq(bound, int_min);
        as_.Cmpq(src, bound);
        as_.Cmovq(Assembler::Condition::kLess, result, bound);
        // Zero Vd, then store result at lane 0 at the dst width.
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        switch (sz) {
          case 0:
            as_.Movb({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          case 1:
            as_.Movw({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          default:
            as_.Movl({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
        }
        return;
      }

      case Opcode::kSqxtun: {
        // SQXTUN scalar: signed-source, unsigned-dst saturating extract
        // narrow, single-lane.  size: 00 → 8-bit dst from 16-bit src,
        // 01 → 16-bit dst from 32-bit src, 10 → 32-bit dst from 64-bit src;
        // size=11 is rejected by the decoder.  Per-lane semantics: treat src
        // as signed; clamp to [0, UMAX_dst] and truncate to dst width.
        // Result is written to lane 0 of Vd with the upper bytes zeroed.
        // See Interpreter::AdvSimdScalarTwoRegMisc::kSqxtun for the
        // reference semantics (matched by the host exec tests below).
        //
        // Strategy: load src sign-extended to int64 at the 2× lane (src)
        // width, then clamp with two CMOV arms — CMOVL against 0 (signed
        // less; negative src → 0) and CMOVG against UMAX_dst (signed
        // greater; src > UMAX_dst → UMAX_dst).  Movq imm64 between the
        // Cmpq/Cmovq pairs does not touch EFLAGS, so each Cmpq sets fresh
        // flags for its own arm.
        const uint8_t sz = static_cast<uint8_t>(args.size & 0x3);
        if (sz == 3) { success_ = false; return; }
        int64_t uint_max;
        switch (sz) {
          case 0: uint_max = 0xFF;       break;  // UINT8_MAX
          case 1: uint_max = 0xFFFF;     break;  // UINT16_MAX
          default:
            // sz == 2: dst is S (32-bit), src is D (64-bit).
            uint_max = 0xFFFFFFFFLL; break;     // UINT32_MAX
        }
        Register src = AllocTempReg();
        Register result = AllocTempReg();
        Register bound = AllocTempReg();
        SimdRegister xzero = AllocTempSimdReg();
        if (src == no_register || result == no_register ||
            bound == no_register || xzero == no_simd_register) {
          success_ = false; return;
        }
        // Load src sign-extended to 64-bit, at the 2× lane src width.
        switch (sz) {
          case 0:
            // src is H (16-bit).
            as_.Movsxwq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          case 1:
            // src is S (32-bit).
            as_.Movsxlq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
          default:
            // src is D (64-bit); already signed at int64 width.
            as_.Movq(src, {.base = Assembler::rbp, .disp = vn_off});
            break;
        }
        // result = src (clamping arms may override below).
        as_.Movq(result, src);
        // Lower-bound clamp: if src < 0, result = 0.
        as_.Movq(bound, int64_t{0});
        as_.Cmpq(src, bound);
        as_.Cmovq(Assembler::Condition::kLess, result, bound);
        // Upper-bound clamp: if src > UMAX_dst, result = UMAX_dst.
        as_.Movq(bound, uint_max);
        as_.Cmpq(src, bound);
        as_.Cmovq(Assembler::Condition::kGreater, result, bound);
        // Zero Vd, then store result at lane 0 at the dst width.
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        switch (sz) {
          case 0:
            as_.Movb({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          case 1:
            as_.Movw({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
          default:
            as_.Movl({.base = Assembler::rbp, .disp = vd_off}, result);
            break;
        }
        return;
      }

      case Opcode::kFrecpe: {
        // FRECPE scalar: single-lane FP reciprocal estimate.  Decoder pins
        // bit1 of size (size ∈ {10, 11}); bit0 selects S (0) vs D (1).
        // Per ARM ARM C7.2.150 the result needs only ~8 bits of mantissa
        // precision; the exact 1.0/x computed by DIVSS / DIVSD is well
        // within bound.  All special-case inputs fall out of x86 IEEE-754
        // divide semantics natively:
        //   ±0   -> ±inf   (matches FRECPE FPSR.DZC behaviour)
        //   ±inf -> ±0
        //   NaN  -> NaN    (single-rounded, propagates input)
        // Matches Interpreter::AdvSimdScalarTwoRegMisc::kFrecpe by-result
        // for every input class — verified by the host exec tests.
        const bool is_double = ((args.size & 1) != 0);
        SimdRegister result_xmm = AllocTempSimdReg();
        SimdRegister src_xmm = AllocTempSimdReg();
        SimdRegister xzero = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        if (result_xmm == no_simd_register || src_xmm == no_simd_register ||
            xzero == no_simd_register || tmp == no_register) {
          success_ = false; return;
        }
        // Build 1.0 in result_xmm; load src lane 0 into src_xmm; divide.
        if (is_double) {
          as_.Movq(tmp, int64_t{0x3FF0000000000000LL});  // bits of 1.0 (FP64)
          as_.Movq(result_xmm, tmp);
          as_.Movsd(src_xmm, {.base = Assembler::rbp, .disp = vn_off});
          as_.Divsd(result_xmm, src_xmm);
        } else {
          as_.Movl(tmp, int32_t{0x3F800000});  // bits of 1.0f (FP32)
          as_.Movd(result_xmm, tmp);
          as_.Movss(src_xmm, {.base = Assembler::rbp, .disp = vn_off});
          as_.Divss(result_xmm, src_xmm);
        }
        // Zero Vd, then write the scalar lane 0.
        as_.Pxor(xzero, xzero);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
        if (is_double) {
          as_.Movsd({.base = Assembler::rbp, .disp = vd_off}, result_xmm);
        } else {
          as_.Movss({.base = Assembler::rbp, .disp = vd_off}, result_xmm);
        }
        return;
      }

      case Opcode::kFrsqrte: {
        // FRSQRTE scalar: single-lane FP reciprocal-square-root estimate.
        // ARM ARM C7.2.155.  Decoder pins bit1 of size (size ∈ {10, 11});
        // bit0 selects S (0) vs D (1).
        //
        // Interpreter reference (interpreter.h kFrsqrte):
        //   r = (src <= 0 || isNaN(src)) ? default_qNaN : 1.0 / sqrt(src)
        //
        // The non-positive-or-NaN replacement uses the default qNaN
        // payload (0x7FC00000 / 0x7FF8000000000000) rather than the
        // input NaN — x86 SQRT/DIV would propagate the input NaN, so we
        // must blend it out explicitly.
        //
        // Branchless lowering: compute the normal-path 1.0/sqrt(src),
        // build the special-case mask (src <= 0 OR isNaN(src)), and blend
        // the default qNaN under the mask.
        //   sqrt   = SQRT(src)                  (SQRTSS / SQRTSD)
        //   recip  = 1.0 / sqrt                 (DIVSS / DIVSD)
        //   mask   = CMPLE(src, 0) | CMPUNORD(src, src)
        //   result = (mask & qNaN) | (~mask & recip)
        // Mantissa-precision requirement is only ~8 bits (same as FRECPE);
        // the exact 1.0/sqrt(x) computed by SQRT+DIV is well within bound
        // for every positive-finite input.
        const bool is_double = ((args.size & 1) != 0);
        SimdRegister src_xmm = AllocTempSimdReg();
        SimdRegister sqrt_xmm = AllocTempSimdReg();
        SimdRegister recip_xmm = AllocTempSimdReg();
        SimdRegister mask_xmm = AllocTempSimdReg();
        SimdRegister mask_nan_xmm = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        if (src_xmm == no_simd_register || sqrt_xmm == no_simd_register ||
            recip_xmm == no_simd_register || mask_xmm == no_simd_register ||
            mask_nan_xmm == no_simd_register || tmp == no_register) {
          success_ = false; return;
        }
        if (is_double) {
          // src_xmm = src (memory load zero-extends to 128 bits).
          as_.Movsd(src_xmm, {.base = Assembler::rbp, .disp = vn_off});
          // sqrt_xmm = sqrt(src) at lane 0.
          as_.Sqrtsd(sqrt_xmm, src_xmm);
          // recip_xmm = 1.0 / sqrt(src).
          as_.Movq(tmp, int64_t{0x3FF0000000000000LL});  // bits of 1.0d
          as_.Movq(recip_xmm, tmp);
          as_.Divsd(recip_xmm, sqrt_xmm);
          // Build the (src <= 0) mask in mask_xmm.  Reuse sqrt_xmm as
          // the 0.0 source (its lane 0 contents are no longer needed).
          as_.Pxor(sqrt_xmm, sqrt_xmm);
          as_.Movdqa(mask_xmm, src_xmm);
          as_.Cmplesd(mask_xmm, sqrt_xmm);
          // Build the isNaN(src) mask in mask_nan_xmm.
          as_.Movdqa(mask_nan_xmm, src_xmm);
          as_.Cmpunordsd(mask_nan_xmm, mask_nan_xmm);
          // mask_xmm = (src <= 0) OR isNaN(src).
          as_.Por(mask_xmm, mask_nan_xmm);
          // Default qNaN constant into src_xmm (src is no longer needed).
          as_.Movq(tmp, int64_t{0x7FF8000000000000LL});
          as_.Movq(src_xmm, tmp);
        } else {
          as_.Movss(src_xmm, {.base = Assembler::rbp, .disp = vn_off});
          as_.Sqrtss(sqrt_xmm, src_xmm);
          as_.Movl(tmp, int32_t{0x3F800000});  // bits of 1.0f
          as_.Movd(recip_xmm, tmp);
          as_.Divss(recip_xmm, sqrt_xmm);
          as_.Pxor(sqrt_xmm, sqrt_xmm);
          as_.Movdqa(mask_xmm, src_xmm);
          as_.Cmpless(mask_xmm, sqrt_xmm);
          as_.Movdqa(mask_nan_xmm, src_xmm);
          as_.Cmpunordss(mask_nan_xmm, mask_nan_xmm);
          as_.Por(mask_xmm, mask_nan_xmm);
          as_.Movl(tmp, int32_t{0x7FC00000});  // default qNaN FP32
          as_.Movd(src_xmm, tmp);
        }
        // Blend: result_in_src_xmm = (mask & qNaN) | (~mask & recip).
        as_.Pand(src_xmm, mask_xmm);
        as_.Pandn(mask_xmm, recip_xmm);
        as_.Por(src_xmm, mask_xmm);
        // Zero Vd, then write the scalar lane 0.
        as_.Pxor(sqrt_xmm, sqrt_xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, sqrt_xmm);
        if (is_double) {
          as_.Movsd({.base = Assembler::rbp, .disp = vd_off}, src_xmm);
        } else {
          as_.Movss({.base = Assembler::rbp, .disp = vd_off}, src_xmm);
        }
        return;
      }

      default:
        // All AdvSimdScalarTwoRegMisc constituents are JIT-emitted.  Any
        // remaining unhandled opcode bails to the interpreter for safety.
        success_ = false;
        return;
    }
  }

