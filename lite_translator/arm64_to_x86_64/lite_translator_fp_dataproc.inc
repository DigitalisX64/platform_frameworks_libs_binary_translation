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
// Family: scalar FP data-processing / compare / estimate tables.
// Methods:
//   FpDataProc1, FpDataProc2, FpCompare, FpConditionalCompare,
//   BuildUnsignedEstimateTable, UnsignedEstimateTable

  void FpDataProc1(const Decoder::FpDataProc1Args& args) {
    // JIT path for FP one-source ops across FP16/FP32/FP64.
    //
    //   FMOV          : direct copy (no F16C).
    //   FABS / FNEG   : sign-bit clear / flip via GP register mask
    //                   (0x7FFF / 0x8000 for FP16, 0x7FFFFFFF / 0x80000000
    //                   for FP32, sign byte at +7 for FP64).
    //   FSQRT         : SQRTSS / SQRTSD natively at FP32/FP64; F16C
    //                   round-trip with SQRTSS at FP16.
    //   FRINTN/M/P/Z/X/I :  ROUNDSS / ROUNDSD natively at FP32/FP64;
    //                   F16C round-trip with ROUNDSS at FP16.
    //   FRINTA        : ARM ties-to-away has no native x86 ROUND* imm.
    //                   Lowered as: dst = trunc(src + copysign(0.5, src)).
    //                   Half goes through F16C round-trip; FP32/FP64 use
    //                   ADDSS+ROUNDSS / ADDSD+ROUNDSD on a GP-built half.
    //   FCVT between precisions : interpreter (different dst layouts).
    //
    // ROUNDSS / ROUNDSD imm[3:0]:
    //   bit 3   : SAE (suppress all FP exceptions)
    //   bit 2   : 1 -> route rounding through MXCSR.RC (NEVER set here;
    //             Berberis doesn't sync MXCSR.RC with guest FPCR.RMode)
    //   bits[1:0] : rounding mode when bit 2 == 0:
    //     00 = RNE (FRINTN)
    //     01 = -inf (FRINTM)
    //     10 = +inf (FRINTP)
    //     11 = zero (FRINTZ)
    //   FRINTX uses current FPCR (assume default RNE) and signals Inexact
    //     -> imm 0x00.
    //   FRINTI uses current FPCR without raising Inexact -> imm 0x08
    //     (SAE + RNE).  Berberis doesn't track FPSR yet, so the SAE bit
    //     is documentary; setting it keeps the lowering spec-faithful.
    //
    // The F16C round-trip is exact for FP16 FSQRT / FRINT* because
    // binary32's 24-bit mantissa fully covers a single-rounding narrowing
    // from any FP16 unary result; the only rounding is at VCVTPS2PH (RNE,
    // matching ARM default FPCR.RMode=0).

    int32_t src_offset = VRegOffset(args.rn);
    int32_t dst_offset = VRegOffset(args.rd);

    if (args.ftype == 0b11) {
      // FP16 (half-precision) ops.
      if (args.opcode == 0b000000) {
        // FMOV Hd, Hn: zero dest, then copy 2 bytes via GP scratch.
        // No F16C dependency.
        SimdRegister xmm = AllocTempSimdReg();
        if (!success()) { return; }
        Register tmp = AllocTempReg();
        as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = src_offset});
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
        as_.Movw({.base = Assembler::rbp, .disp = dst_offset}, tmp);
        return;
      }

      if (args.opcode == 0b000001 || args.opcode == 0b000010) {
        // FABS / FNEG Hd, Hn: load 16 bits, mask/xor sign bit, store 2
        // bytes into a zero-filled dest.  No F16C dependency.
        SimdRegister xmm = AllocTempSimdReg();
        if (!success()) { return; }
        Register tmp = AllocTempReg();
        as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = src_offset});
        if (args.opcode == 0b000001) {
          as_.Andl(tmp, int32_t{0x00007FFF});  // FABS: clear sign bit
        } else {
          as_.Xorl(tmp, int32_t{0x00008000});  // FNEG: flip sign bit
        }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
        as_.Movw({.base = Assembler::rbp, .disp = dst_offset}, tmp);
        return;
      }

      // FSQRT, FRINT*, and FCVT-from-half need F16C; bail to interpreter
      // if absent.
      if (!host_platform::kHasF16C) { success_ = false; return; }

      // FCVT Sd, Hn (opcode=0b000100) and FCVT Dd, Hn (opcode=0b000101):
      // widen half->single via VCVTPH2PS, then optionally widen to double
      // via CVTSS2SD.  Result lives in lane 0 of a freshly-zeroed dst XMM,
      // so MOVDQU writes the AArch64 scalar layout [val_in_low_N, 0×rest]
      // verbatim.
      if (args.opcode == 0b000100 || args.opcode == 0b000101) {
        bool to_double = (args.opcode == 0b000101);
        SimdRegister xmm_src = AllocTempSimdReg();
        SimdRegister xmm_dst = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pxor(xmm_src, xmm_src);
        as_.Pinsrw(xmm_src, {.base = Assembler::rbp, .disp = src_offset},
                   int8_t{0});
        as_.Vcvtph2ps(xmm_src, xmm_src);   // half->single in lane 0
        if (to_double) {
          as_.Pxor(xmm_dst, xmm_dst);
          as_.Cvtss2sd(xmm_dst, xmm_src);  // single->double in lane 0; dst
                                           // upper 96 preserved (zero).
          as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_dst);
        } else {
          // VCVTPH2PS already produced [single, 0, 0, 0] in xmm_src
          // because lanes 1..3 of the source halves were zero.
          as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_src);
        }
        return;
      }

      // FRINTA Hd, Hn: ties-to-away in FP32 space via the canonical
      // add-copysign(0.5)-then-truncate trick.  Build the copysign value
      // in a GP register from the FP16 sign bit, widen src via F16C, do
      // the ADDSS + ROUNDSS imm=3 (truncate), and narrow back.
      if (args.opcode == 0b001100) {
        SimdRegister xmm_val = AllocTempSimdReg();
        SimdRegister xmm_half = AllocTempSimdReg();
        if (!success()) { return; }
        Register tmp = AllocTempReg();
        // bits-of-copysign(0.5_fp32, src_fp16): sign of FP16 src shifted to
        // FP32 sign-bit position, OR'd with the FP32 mantissa/exponent bits
        // of +0.5 (0x3F000000).
        as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = src_offset});
        as_.Andl(tmp, int32_t{0x00008000});
        as_.Shll(tmp, int8_t{16});
        as_.Orl(tmp, int32_t{0x3F000000});
        as_.Movd(xmm_half, tmp);
        as_.Pxor(xmm_val, xmm_val);
        as_.Pinsrw(xmm_val, {.base = Assembler::rbp, .disp = src_offset},
                   int8_t{0});
        as_.Vcvtph2ps(xmm_val, xmm_val);
        as_.Addss(xmm_val, xmm_half);
        as_.Roundss(xmm_val, xmm_val, int8_t{0x03});  // truncate toward 0
        as_.Vcvtps2ph(xmm_val, xmm_val, int8_t{0});
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
        return;
      }

      int8_t round_imm = 0;
      bool is_sqrt = false;
      switch (args.opcode) {
        case 0b000011: is_sqrt = true; break;       // FSQRT
        case 0b001000: round_imm = 0x00; break;     // FRINTN
        case 0b001001: round_imm = 0x02; break;     // FRINTP
        case 0b001010: round_imm = 0x01; break;     // FRINTM
        case 0b001011: round_imm = 0x03; break;     // FRINTZ
        case 0b001110: round_imm = 0x00; break;     // FRINTX
        case 0b001111: round_imm = 0x08; break;     // FRINTI
        default:
          // Unknown FP16 unary opcode -> interpreter.
          success_ = false;
          return;
      }

      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Pxor(xmm, xmm);
      as_.Pinsrw(xmm, {.base = Assembler::rbp, .disp = src_offset}, int8_t{0});
      as_.Vcvtph2ps(xmm, xmm);
      if (is_sqrt) {
        as_.Sqrtss(xmm, xmm);
      } else {
        as_.Roundss(xmm, xmm, round_imm);
      }
      as_.Vcvtps2ph(xmm, xmm, int8_t{0});
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
      return;
    }

    // FP32 (ftype=00) and FP64 (ftype=01) paths.
    if (args.ftype != 0b00 && args.ftype != 0b01) { Undefined(); return; }
    bool is_double = (args.ftype == 0b01);

    // FMOV, FABS, FNEG: GP-register mask/copy lowering.
    if (args.opcode == 0b000000 || args.opcode == 0b000001 ||
        args.opcode == 0b000010) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp_lo = AllocTempReg();

      if (!is_double) {
        // FP32: one 4-byte word.
        as_.Movl(tmp_lo, {.base = Assembler::rbp, .disp = src_offset});
        if (args.opcode == 0b000001) {
          as_.Andl(tmp_lo, int32_t{0x7FFFFFFF});  // FABS
        } else if (args.opcode == 0b000010) {
          as_.Xorl(tmp_lo, static_cast<int32_t>(0x80000000));  // FNEG
        }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
        as_.Movl({.base = Assembler::rbp, .disp = dst_offset}, tmp_lo);
        return;
      }

      // FP64: load low 4 bytes verbatim, load high 4 bytes and patch sign
      // bit there.  Splitting into two MOVL avoids needing a 64-bit
      // immediate (MOVABSQ) for the FABS/FNEG masks.
      if (args.opcode == 0b000000) {
        // FMOV Dd, Dn: copy all 8 bytes.
        as_.Movq(tmp_lo, {.base = Assembler::rbp, .disp = src_offset});
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
        as_.Movq({.base = Assembler::rbp, .disp = dst_offset}, tmp_lo);
        return;
      }
      Register tmp_hi = AllocTempReg();
      as_.Movl(tmp_lo, {.base = Assembler::rbp, .disp = src_offset});
      as_.Movl(tmp_hi, {.base = Assembler::rbp, .disp = src_offset + 4});
      if (args.opcode == 0b000001) {
        as_.Andl(tmp_hi, int32_t{0x7FFFFFFF});  // FABS Dd
      } else {
        as_.Xorl(tmp_hi, static_cast<int32_t>(0x80000000));  // FNEG Dd
      }
      as_.Pxor(xmm, xmm);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
      as_.Movl({.base = Assembler::rbp, .disp = dst_offset}, tmp_lo);
      as_.Movl({.base = Assembler::rbp, .disp = dst_offset + 4}, tmp_hi);
      return;
    }

    // FCVT between FP32 and FP64: zero a fresh dst XMM, then CVTSS2SD or
    // CVTSD2SS into lane 0.  Both ops preserve the upper 96 bits of the
    // destination, so the zero-then-convert sequence leaves
    // [val_in_low_N, 0×rest] which is the AArch64 scalar layout.
    if (!is_double && args.opcode == 0b000101) {
      // FCVT Dd, Sn — single -> double.  No F16C needed.
      SimdRegister xmm_src = AllocTempSimdReg();
      SimdRegister xmm_dst = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movss(xmm_src, {.base = Assembler::rbp, .disp = src_offset});
      as_.Pxor(xmm_dst, xmm_dst);
      as_.Cvtss2sd(xmm_dst, xmm_src);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_dst);
      return;
    }
    if (is_double && args.opcode == 0b000100) {
      // FCVT Sd, Dn — double -> single.  No F16C needed.
      SimdRegister xmm_src = AllocTempSimdReg();
      SimdRegister xmm_dst = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movsd(xmm_src, {.base = Assembler::rbp, .disp = src_offset});
      as_.Pxor(xmm_dst, xmm_dst);
      as_.Cvtsd2ss(xmm_dst, xmm_src);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_dst);
      return;
    }
    // FCVT Hd, Sn / FCVT Hd, Dn — narrow to half.  Needs F16C for VCVTPS2PH.
    // For FP64 source, first narrow double->single via CVTSD2SS into a
    // zeroed XMM, then VCVTPS2PH narrows single->half and zeroes upper 64.
    if (args.opcode == 0b000111) {
      if (!host_platform::kHasF16C) { success_ = false; return; }
      SimdRegister xmm_src = AllocTempSimdReg();
      SimdRegister xmm_dst = AllocTempSimdReg();
      if (!success()) { return; }
      if (is_double) {
        as_.Movsd(xmm_src, {.base = Assembler::rbp, .disp = src_offset});
        as_.Pxor(xmm_dst, xmm_dst);
        as_.Cvtsd2ss(xmm_dst, xmm_src);            // single in xmm_dst lane 0
        as_.Vcvtps2ph(xmm_dst, xmm_dst, int8_t{0}); // half in xmm_dst lane 0
      } else {
        as_.Movss(xmm_src, {.base = Assembler::rbp, .disp = src_offset});
        as_.Vcvtps2ph(xmm_dst, xmm_src, int8_t{0}); // half in xmm_dst lane 0
      }
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_dst);
      return;
    }

    // FRINTA Sd/Dd, Sn/Dn: ties-to-away has no native ROUND* imm.
    // Lowered as: dst = trunc(src + copysign(0.5, src)), gated on
    // magnitude.  For |x| >= 2^23 (FP32) / 2^52 (FP64) x is already an
    // exact integer (FP step >= 1 at that magnitude), so adding 0.5
    // lands a tie below the LSB and RNE round-half-to-even would bump
    // odd values to the next even — skip the add in that range and
    // trunc directly (trunc on an exact integer is a no-op).  NaN /
    // +/-Inf bits also exceed the threshold and skip the add; ROUNDSS/SD
    // preserves NaN and Inf per Intel SDM.
    if (args.opcode == 0b001100) {
      SimdRegister xmm_val = AllocTempSimdReg();
      SimdRegister xmm_half = AllocTempSimdReg();
      SimdRegister xmm_zero = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp = AllocTempReg();
      Register sign_tmp = AllocTempReg();
      // Load src into xmm_val first so the magnitude gate can read its
      // bits without a second memory round-trip.
      if (is_double) {
        as_.Movsd(xmm_val, {.base = Assembler::rbp, .disp = src_offset});
      } else {
        as_.Movss(xmm_val, {.base = Assembler::rbp, .disp = src_offset});
      }
      // Magnitude gate via integer-domain compare on |bits(x)|.  IEEE-754
      // bits compare as unsigned int for non-negative values; clearing
      // the sign bit gives |bits(x)|.  Compare against the bit pattern
      // of 2^23 (FP32) / 2^52 (FP64).
      Assembler::Label* skip_add = as_.MakeLabel();
      if (is_double) {
        as_.Movq(sign_tmp, xmm_val);
        as_.Movq(tmp, static_cast<int64_t>(0x7FFFFFFFFFFFFFFFLL));
        as_.Andq(sign_tmp, tmp);
        as_.Movq(tmp, static_cast<int64_t>(0x4330000000000000LL));  // 2^52
        as_.Cmpq(sign_tmp, tmp);
        as_.Jcc(Assembler::Condition::kAboveEqual, *skip_add);
      } else {
        as_.Movd(sign_tmp, xmm_val);
        as_.Andl(sign_tmp, int32_t{0x7FFFFFFF});
        as_.Cmpl(sign_tmp, int32_t{0x4B000000});  // 2^23
        as_.Jcc(Assembler::Condition::kAboveEqual, *skip_add);
      }
      // |x| < threshold: build copysign(0.5, x) and add to xmm_val.
      if (is_double) {
        // FP64: high 32 bits hold sign+exp; low 32 of 0.5_fp64 are zero.
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = src_offset + 4});
        as_.Andl(tmp, static_cast<int32_t>(0x80000000));  // sign bit only
        as_.Orl(tmp, int32_t{0x3FE00000});                 // |= high32(0.5d)
        as_.Pxor(xmm_half, xmm_half);
        as_.Pinsrd(xmm_half, tmp, int8_t{1});              // copysign(0.5d, src)
        as_.Addsd(xmm_val, xmm_half);
      } else {
        // FP32: one word holds sign+exp+mantissa.
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = src_offset});
        as_.Andl(tmp, static_cast<int32_t>(0x80000000));  // sign bit only
        as_.Orl(tmp, int32_t{0x3F000000});                 // |= bits of +0.5
        as_.Movd(xmm_half, tmp);                           // copysign(0.5, src)
        as_.Addss(xmm_val, xmm_half);
      }
      as_.Bind(skip_add);
      // Truncate toward zero (no-op for the skip-add path; rounds away
      // from zero for the add-half path because adding sign(x)*0.5
      // pushed |x| up by half).
      if (is_double) {
        as_.Roundsd(xmm_val, xmm_val, int8_t{0x03});
      } else {
        as_.Roundss(xmm_val, xmm_val, int8_t{0x03});
      }
      // Write back to v[d], zeroing the high 64 bits of the scalar
      // AArch64 layout.
      as_.Pxor(xmm_zero, xmm_zero);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_zero);
      if (is_double) {
        as_.Movsd({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
      } else {
        as_.Movss({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
      }
      return;
    }

    // BFCVT Hd, Sn (single -> BF16) (§H2).
    //
    // Encoding: ftype=01 (D-form discriminant), opcode=0b000110.  The
    // source register is read as FP32 (Sn) despite ftype=01 — the
    // discriminant is purely an opcode-namespace selector, not a "source
    // is double" hint.  Mirrors interpreter handling at
    // `interpreter.h:4934` and the FloatToBf16 helper at
    // `interpreter.h:1330`.
    //
    // Algorithm:
    //   bits = src as uint32_t                  ; raw FP32 bits
    //   if (exp == 0xFF && mant != 0):
    //     out = (bits >> 16) | 0x0040           ; quiet NaN with payload top
    //   else:
    //     lsb = (bits >> 16) & 1                ; tie-to-even bias
    //     out = (bits + 0x7FFF + lsb) >> 16     ; RTNE narrow
    //   Vd = 0; low 16 bits of Vd = out
    //
    // ±Inf and ±0 pass through structurally identical (RTNE on Inf:
    // exp=0xFF mant=0, the +0x7FFF+lsb doesn't carry into the exponent;
    // top-16 still encodes the same Inf).  No AVX-512-BF16 dependency.
    if (args.opcode == 0b000110 && is_double) {
      SimdRegister xmm_zero = AllocTempSimdReg();
      if (!success()) { return; }
      Register bits = AllocTempReg();
      Register tmp = AllocTempReg();
      Register lsb = AllocTempReg();

      as_.Movl(bits, {.base = Assembler::rbp, .disp = src_offset});

      Assembler::Label* nan_label = as_.MakeLabel();
      Assembler::Label* done_label = as_.MakeLabel();
      Assembler::Label* rtne_label = as_.MakeLabel();

      // NaN check: exponent == 0xFF AND mantissa != 0.
      as_.Movl(tmp, bits);
      as_.Andl(tmp, int32_t{0x7F800000});
      as_.Cmpl(tmp, int32_t{0x7F800000});
      as_.Jcc(Assembler::Condition::kNotEqual, *rtne_label);
      as_.Movl(tmp, bits);
      as_.Andl(tmp, int32_t{0x007FFFFF});
      as_.Jcc(Assembler::Condition::kNotZero, *nan_label);

      // Fall through (±Inf): treat as RTNE.  Top 16 bits are the BF16.
      as_.Bind(rtne_label);
      as_.Movl(lsb, bits);
      as_.Shrl(lsb, int8_t{16});
      as_.Andl(lsb, int32_t{1});
      as_.Addl(lsb, int32_t{0x7FFF});
      as_.Addl(bits, lsb);
      as_.Shrl(bits, int8_t{16});
      as_.Jmp(*done_label);

      // NaN fixup: quiet by setting top bit of BF16 mantissa.
      as_.Bind(nan_label);
      as_.Shrl(bits, int8_t{16});
      as_.Orl(bits, int32_t{0x0040});

      as_.Bind(done_label);
      as_.Pxor(xmm_zero, xmm_zero);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_zero);
      as_.Movw({.base = Assembler::rbp, .disp = dst_offset}, bits);
      return;
    }

    // FSQRT and FRINT*: SIMD lowering via SQRTSS/SQRTSD / ROUNDSS/ROUNDSD.
    int8_t round_imm = 0;
    bool is_sqrt = false;
    switch (args.opcode) {
      case 0b000011: is_sqrt = true; break;       // FSQRT
      case 0b001000: round_imm = 0x00; break;     // FRINTN
      case 0b001001: round_imm = 0x02; break;     // FRINTP
      case 0b001010: round_imm = 0x01; break;     // FRINTM
      case 0b001011: round_imm = 0x03; break;     // FRINTZ
      case 0b001110: round_imm = 0x00; break;     // FRINTX
      case 0b001111: round_imm = 0x08; break;     // FRINTI
      default:
        // Unknown FP unary opcode -> interpreter.
        success_ = false;
        return;
    }

    SimdRegister xmm_val = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_zero = AllocTempSimdReg();
    if (!success()) { return; }

    // Load src first (in case dst == src).
    if (is_double) {
      as_.Movsd(xmm_val, {.base = Assembler::rbp, .disp = src_offset});
    } else {
      as_.Movss(xmm_val, {.base = Assembler::rbp, .disp = src_offset});
    }
    // Zero the full 128-bit dst register.
    as_.Pxor(xmm_zero, xmm_zero);
    as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_zero);
    // Operate in place on lane 0 of xmm_val (upper lanes were zeroed by
    // MOVSS/MOVSD from memory; SQRTSS/SQRTSD and ROUNDSS/ROUNDSD preserve
    // the upper lanes of the dest).
    if (is_double) {
      if (is_sqrt) {
        as_.Sqrtsd(xmm_val, xmm_val);
      } else {
        as_.Roundsd(xmm_val, xmm_val, round_imm);
      }
      as_.Movsd({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
    } else {
      if (is_sqrt) {
        as_.Sqrtss(xmm_val, xmm_val);
      } else {
        as_.Roundss(xmm_val, xmm_val, round_imm);
      }
      as_.Movss({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
    }
  }

  // FP arithmetic JIT
  //
  // FP16 (ftype=0b11) lowering: F16C round-trip.
  //   PINSRW [src] -> XMM lane0       (load the 16-bit half, upper lanes zero)
  //   VCVTPH2PS XMM, XMM              (widen 4 halves->4 singles; lanes 1-3 are zero)
  //   <op>SS    XMM, XMM              (perform the binary op in single precision)
  //   VCVTPS2PH XMM, XMM, imm=0       (narrow back, RNE, ignore MXCSR)
  //   MOVDQU [dst], XMM               (store 128 bits: result in low 16, rest zero)
  //
  // The round-trip is exact for FADD/FSUB/FMUL/FDIV because binary32's 24-bit
  // mantissa fully covers a single-rounding narrowing from any FP16 op result.
  // RNE matches the ARM default rounding mode (FPCR.RMode=0, ties-to-even).
  // F16C is part of the IvyBridge+ baseline; bail to interpreter if absent.
  void FpDataProc2(const Decoder::FpDataProc2Args& args) {
    // Only handle single (ftype=00), double (ftype=01), and half (ftype=11).
    if (args.ftype == 0b10) { Undefined(); return; }
    if (args.ftype == 0b11 && !host_platform::kHasF16C) { success_ = false; return; }
    bool is_double = (args.ftype == 0b01);
    bool is_half = (args.ftype == 0b11);

    int32_t src_n_off = VRegOffset(args.rn);
    int32_t src_m_off = VRegOffset(args.rm);
    int32_t dst_off = VRegOffset(args.rd);

    SimdRegister xmm_n = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (!success()) { return; }

    // Load operands.
    if (is_double) {
      as_.Movsd(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movsd(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    } else if (is_half) {
      as_.Pxor(xmm_n, xmm_n);
      as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_n, xmm_n);
      as_.Pxor(xmm_m, xmm_m);
      as_.Pinsrw(xmm_m, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_m, xmm_m);
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    }

    // Perform operation: result in xmm_n. FP16 reuses the SS form because the
    // value already lives as binary32 in lane 0.
    switch (args.opcode) {
      case 0b0000:  // FMUL
        if (is_double) as_.Mulsd(xmm_n, xmm_m);
        else as_.Mulss(xmm_n, xmm_m);
        break;
      case 0b0001:  // FDIV
        if (is_double) as_.Divsd(xmm_n, xmm_m);
        else as_.Divss(xmm_n, xmm_m);
        break;
      case 0b0010:  // FADD
        if (is_double) as_.Addsd(xmm_n, xmm_m);
        else as_.Addss(xmm_n, xmm_m);
        break;
      case 0b0011:  // FSUB
        if (is_double) as_.Subsd(xmm_n, xmm_m);
        else as_.Subss(xmm_n, xmm_m);
        break;
      // scalar FMAX / FMIN / FMAXNM / FMINNM / FNMUL JIT.
      //
      // FMAX/FMIN (NaN-propagating per ARM ARM): symmetric MAX with POR.
      //   tmp = m; MAXP{S,D} tmp, n   ; tmp lane0 = NaN if any NaN else max
      //   MAXP{S,D} n, m              ; n lane0   = NaN if any NaN else max
      //   POR n, tmp                  ; bitwise OR keeps NaN exponent if any NaN
      //
      // FMAXNM/FMINNM (NaN-suppressing per ARM ARM): substitute NaN lanes
      // with the other operand, then MAXP{S,D}.
      //
      // Operates on the vector forms because the SS/SD scalar MAX/MIN
      // mnemonics are not exposed in the Berberis x86 assembler. For S/D
      // only lane 0 matters (Movss/Movsd writes lane 0 only); for H the
      // upper lanes are pre-zeroed by Pxor+Pinsrw, so MAXP{S,D} lanes 1..3
      // = max(0, 0) = 0 and the F16C narrow produces zero in FP16 lanes
      // 1..3 — matching the AArch64 Hd zero-extend semantic.
      case 0b0100:    // FMAX
      case 0b0101: {  // FMIN
        SimdRegister tmp = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqa(tmp, xmm_m);
        const bool is_max = (args.opcode == 0b0100);
        if (is_max) {
          if (is_double) {
            as_.Maxpd(tmp, xmm_n);
            as_.Maxpd(xmm_n, xmm_m);
          } else {
            as_.Maxps(tmp, xmm_n);
            as_.Maxps(xmm_n, xmm_m);
          }
        } else {
          if (is_double) {
            as_.Minpd(tmp, xmm_n);
            as_.Minpd(xmm_n, xmm_m);
          } else {
            as_.Minps(tmp, xmm_n);
            as_.Minps(xmm_n, xmm_m);
          }
        }
        as_.Por(xmm_n, tmp);
        break;
      }
      case 0b0110:    // FMAXNM
      case 0b0111: {  // FMINNM
        SimdRegister mask_a = AllocTempSimdReg();
        SimdRegister mask_b = AllocTempSimdReg();
        SimdRegister an_sub = AllocTempSimdReg();
        SimdRegister bn_sub = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqa(mask_a, xmm_n);
        if (is_double) as_.Cmpunordpd(mask_a, mask_a);
        else as_.Cmpunordps(mask_a, mask_a);
        as_.Movdqa(mask_b, xmm_m);
        if (is_double) as_.Cmpunordpd(mask_b, mask_b);
        else as_.Cmpunordps(mask_b, mask_b);
        as_.Movdqa(an_sub, mask_a);
        as_.Pand(an_sub, xmm_m);
        as_.Movdqa(bn_sub, mask_b);
        as_.Pand(bn_sub, xmm_n);
        as_.Pandn(mask_a, xmm_n);
        as_.Pandn(mask_b, xmm_m);
        as_.Por(mask_a, an_sub);
        as_.Por(mask_b, bn_sub);
        const bool is_max = (args.opcode == 0b0110);
        if (is_max) {
          if (is_double) as_.Maxpd(mask_a, mask_b);
          else as_.Maxps(mask_a, mask_b);
        } else {
          if (is_double) as_.Minpd(mask_a, mask_b);
          else as_.Minps(mask_a, mask_b);
        }
        as_.Movdqa(xmm_n, mask_a);
        break;
      }
      case 0b1000: {  // FNMUL: -(n * m)
        SimdRegister sign_xmm = AllocTempSimdReg();
        if (!success()) { return; }
        if (is_double) {
          as_.Mulsd(xmm_n, xmm_m);
          Register sign_gpr = AllocTempReg();
          as_.Movq(sign_gpr, int64_t{static_cast<int64_t>(0x8000000000000000ULL)});
          as_.Movq(sign_xmm, sign_gpr);
          as_.Xorpd(xmm_n, sign_xmm);
        } else {
          as_.Mulss(xmm_n, xmm_m);
          Register sign_gpr = AllocTempReg();
          as_.Movl(sign_gpr, int32_t{static_cast<int32_t>(0x80000000U)});
          as_.Movd(sign_xmm, sign_gpr);
          as_.Xorps(xmm_n, sign_xmm);
        }
        break;
      }
      default:
        // Any other opcode (reserved / future) — fall back to interpreter.
        Undefined();
        return;
    }

    if (is_half) {
      // Narrow FP32 result -> FP16 with RNE (imm=0). VCVTPS2PH zeroes the upper
      // 64 bits of the XMM dest; lanes 1-3 of the FP32 are zero so the FP16
      // lanes 1-3 are also zero. Storing all 128 bits gives the correct guest
      // register layout: result in low 16, rest zero.
      as_.Vcvtps2ph(xmm_n, xmm_n, int8_t{0});
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_n);
      return;
    }

    // Zero dest register, then store result.
    SimdRegister zero = AllocTempSimdReg();
    if (zero == no_simd_register) {
      // Can reuse xmm_m since we're done with it.
      as_.Pxor(xmm_m, xmm_m);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_m);
    } else {
      as_.Pxor(zero, zero);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, zero);
    }
    if (is_double) {
      as_.Movsd({.base = Assembler::rbp, .disp = dst_off}, xmm_n);
    } else {
      as_.Movss({.base = Assembler::rbp, .disp = dst_off}, xmm_n);
    }
  }

  void FpCompare(const Decoder::FpCompareArgs& args) {
    // FCMP Sn, Sm / FCMP Dn, Dm / FCMP Hn, Hm: compare and set NZCV flags.
    if (args.ftype == 0b10) { Undefined(); return; }
    if (args.ftype == 0b11 && !host_platform::kHasF16C) { success_ = false; return; }
    bool is_double = (args.ftype == 0b01);
    bool is_half = (args.ftype == 0b11);

    int32_t src_n_off = VRegOffset(args.rn);
    int32_t src_m_off = VRegOffset(args.rm);

    SimdRegister xmm_n = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (!success()) { return; }

    if (is_double) {
      as_.Movsd(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      if (args.with_zero) {
        as_.Pxor(xmm_m, xmm_m);
      } else {
        as_.Movsd(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      }
      as_.Ucomisd(xmm_n, xmm_m);
    } else if (is_half) {
      // Widen both operands FP16->FP32 via F16C, then UCOMISS. FP16 zero
      // (0x0000) round-trips through Vcvtph2ps to FP32 +0.0, so the with_zero
      // path simply leaves xmm_m as Pxor'd (FP32 zero in lane 0).
      as_.Pxor(xmm_n, xmm_n);
      as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_n, xmm_n);
      if (args.with_zero) {
        as_.Pxor(xmm_m, xmm_m);
      } else {
        as_.Pxor(xmm_m, xmm_m);
        as_.Pinsrw(xmm_m, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_m, xmm_m);
      }
      as_.Ucomiss(xmm_n, xmm_m);
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      if (args.with_zero) {
        as_.Pxor(xmm_m, xmm_m);
      } else {
        as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      }
      as_.Ucomiss(xmm_n, xmm_m);
    }

    // fix: emit the correct FP-specific ARM NZCV
    // mapping, not the integer-SUB EmitStoreArmNZCV.  UCOMISS/UCOMISD set
    // only ZF/PF/CF; SF and OF retain stale values, so the integer-flag
    // emission produced random N and V bits.  This silently broke any
    // FCMP-then-CSET-{ge,gt,lt,le} sequence (only EQ/NE worked because
    // they only depend on Z, which the integer mapping happened to set
    // correctly).  See note in EmitStoreArmFpNZCV below.
    EmitStoreArmFpNZCV();
  }

  // FCCMP / FCCMPE: if cond evaluates true, perform UCOMISS/UCOMISD and map
  // x86 EFLAGS -> ARM NZCV via EmitStoreArmFpNZCV (same as FCMP); otherwise
  // write the immediate NZCV field directly to ThreadState::cpu.flags.
  //
  // ftype 00 (S), 01 (D) and 11 (H, Armv8.2-FP16) are all encoded; 10 is
  // reserved.  FP16 widens both operands to FP32 via F16C (UCOMISS), matching
  // FpCompare; without host F16C the region bails to the interpreter.
  //
  // Layout mirrors the integer ConditionalSelect / FCSEL NZCV decoder:
  //   1. Read existing flags into flags_reg (Btl source).
  //   2. Write the nzcv immediate into ThreadState::cpu.flags (FALSE default).
  //   3. Condition switch: each arm Jcc-s to `done` if the condition is FALSE
  //      (so the imm just written wins).
  //   4. Fall-through = condition TRUE: do the UCOMIS compare, then
  //      EmitStoreArmFpNZCV reads x86 EFLAGS and overwrites cpu.flags.
  //   5. done.
  //
  // The 'signal_nans' (FCCMPE) bit only changes the FP-exception behaviour
  // (signal vs quiet) — the architectural NZCV output is identical for both
  // FCCMP and FCCMPE.  UCOMISS/UCOMISD on x86 already signal on SNaN by
  // setting #IA (matching FCCMPE), so we ignore the bit here; the host's
  // SIMD floating-point exception path follows the same trap behaviour as
  // a plain FpCompare.
  void FpConditionalCompare(const Decoder::FpConditionalCompareArgs& args) {
    if (args.ftype == 0b10) { Undefined(); return; }
    if (args.ftype == 0b11 && !host_platform::kHasF16C) { success_ = false; return; }
    const bool is_double = (args.ftype == 0b01);
    const bool is_half = (args.ftype == 0b11);

    SimdRegister xmm_n = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (!success()) { return; }
    Register flags_reg = AllocTempReg();
    if (flags_reg == no_register) { success_ = false; return; }
    Register imm_reg = AllocTempReg();
    if (imm_reg == no_register) { success_ = false; return; }

    const int32_t src_n_off = VRegOffset(args.rn);
    const int32_t src_m_off = VRegOffset(args.rm);
    const int32_t flags_off = offsetof(ThreadState, cpu.flags);

    // ARM NZCV layout matches CPUState::kFlag{Negative,Zero,Carry,Overflow}:
    //   N = bit 15, Z = bit 14, C = bit 8, V = bit 0.
    const int32_t imm_flags =
        ((args.nzcv & 0b1000) ? CPUState::kFlagNegative : 0) |
        ((args.nzcv & 0b0100) ? CPUState::kFlagZero : 0) |
        ((args.nzcv & 0b0010) ? CPUState::kFlagCarry : 0) |
        ((args.nzcv & 0b0001) ? CPUState::kFlagOverflow : 0);

    // Step 1: snapshot flags for the condition test.
    as_.Movzxwl(flags_reg, {.base = Assembler::rbp, .disp = flags_off});

    // Step 2: write nzcv imm as the FALSE-branch default.
    as_.Movl(imm_reg, imm_flags);
    as_.Movw({.base = Assembler::rbp, .disp = flags_off}, imm_reg);

    Assembler::Label* done = as_.MakeLabel();

    // Step 3: condition switch (same shape as FCSEL — each arm jumps to
    // `done` if the condition is FALSE).
    switch (args.cond) {
      case Decoder::Condition::kEq:
        as_.Btl(flags_reg, kFlagZeroBit);
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      case Decoder::Condition::kNe:
        as_.Btl(flags_reg, kFlagZeroBit);
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kCs:
        as_.Btl(flags_reg, kFlagCarryBit);
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      case Decoder::Condition::kCc:
        as_.Btl(flags_reg, kFlagCarryBit);
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kMi:
        as_.Btl(flags_reg, kFlagNegativeBit);
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      case Decoder::Condition::kPl:
        as_.Btl(flags_reg, kFlagNegativeBit);
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kVs:
        as_.Btl(flags_reg, kFlagOverflowBit);
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      case Decoder::Condition::kVc:
        as_.Btl(flags_reg, kFlagOverflowBit);
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kHi:
        as_.Btl(flags_reg, kFlagCarryBit);
        as_.Jcc(Condition::kNotCarry, *done);
        as_.Btl(flags_reg, kFlagZeroBit);
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kLs: {
        Assembler::Label* true_path = as_.MakeLabel();
        as_.Btl(flags_reg, kFlagCarryBit);
        as_.Jcc(Condition::kNotCarry, *true_path);
        as_.Btl(flags_reg, kFlagZeroBit);
        as_.Jcc(Condition::kNotCarry, *done);
        as_.Bind(true_path);
        break;
      }
      case Decoder::Condition::kGe: {
        Register tmp = AllocTempReg();
        if (tmp == no_register) { success_ = false; return; }
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, kFlagNegativeBit);
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, kFlagOverflowBit);
        as_.Jcc(Condition::kCarry, *done);
        break;
      }
      case Decoder::Condition::kLt: {
        Register tmp = AllocTempReg();
        if (tmp == no_register) { success_ = false; return; }
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, kFlagNegativeBit);
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, kFlagOverflowBit);
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      }
      case Decoder::Condition::kGt: {
        as_.Btl(flags_reg, kFlagZeroBit);
        as_.Jcc(Condition::kCarry, *done);
        Register tmp = AllocTempReg();
        if (tmp == no_register) { success_ = false; return; }
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, kFlagNegativeBit);
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, kFlagOverflowBit);
        as_.Jcc(Condition::kCarry, *done);
        break;
      }
      case Decoder::Condition::kLe: {
        Assembler::Label* true_path = as_.MakeLabel();
        as_.Btl(flags_reg, kFlagZeroBit);
        as_.Jcc(Condition::kCarry, *true_path);
        Register tmp = AllocTempReg();
        if (tmp == no_register) { success_ = false; return; }
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, kFlagNegativeBit);
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, kFlagOverflowBit);
        as_.Jcc(Condition::kNotCarry, *done);
        as_.Bind(true_path);
        break;
      }
      case Decoder::Condition::kAl:
      case Decoder::Condition::kNv:
        // Reserved on FCCMP per the ARM ARM; ConditionalSelect/FCSEL treat
        // these as always-true, so we fall through to the compare path too.
        break;
    }

    // Step 4: condition TRUE path — perform the FP compare.  UCOMIS sets
    // ZF/PF/CF; EmitStoreArmFpNZCV reads them and writes ARM NZCV.
    if (is_double) {
      as_.Movsd(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movsd(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      as_.Ucomisd(xmm_n, xmm_m);
    } else if (is_half) {
      // Widen both FP16 operands to FP32 via F16C, then UCOMISS (mirrors
      // FpCompare's half path).
      as_.Pxor(xmm_n, xmm_n);
      as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_n, xmm_n);
      as_.Pxor(xmm_m, xmm_m);
      as_.Pinsrw(xmm_m, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_m, xmm_m);
      as_.Ucomiss(xmm_n, xmm_m);
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      as_.Ucomiss(xmm_n, xmm_m);
    }
    EmitStoreArmFpNZCV();

    as_.Bind(done);
  }

  // URECPE / URSQRTE estimate tables. The result of these
  // unsigned integer reciprocal / reciprocal-sqrt estimates is a pure function
  // of the 9-bit field (a>>23)&0x1FF (the saturation condition — top bit clear
  // for URECPE, top two bits clear for URSQRTE — is encoded in that field's
  // high bits), so a 512-entry table per op lets the JIT do a per-lane lookup
  // that bit-matches the interpreter's UnsignedRecipEstimate/RSqrtEstimate.
  static const uint32_t* BuildUnsignedEstimateTable(bool is_rsqrt) {
    uint32_t* t = new uint32_t[512];
    for (int idx = 0; idx < 512; ++idx) {
      uint32_t r;
      if (!is_rsqrt) {
        if (idx < 256) {  // top bit (bit31 of input) clear -> saturate
          r = 0xFFFFFFFFu;
        } else {
          int a2 = idx * 2 + 1;
          int b = (1 << 19) / a2;
          int estimate = (b + 1) / 2;
          r = static_cast<uint32_t>(estimate) << 23;
        }
      } else {
        if (idx < 128) {  // top two bits clear -> saturate
          r = 0xFFFFFFFFu;
        } else {
          int aa;
          if (idx < 256) {
            aa = idx * 2 + 1;
          } else {
            aa = (idx >> 1) << 1;
            aa = (aa + 1) * 2;
          }
          int b = 512;
          while (static_cast<int64_t>(aa) * (b + 1) * (b + 1) < (1 << 28)) {
            b += 1;
          }
          int estimate = (b + 1) / 2;
          r = static_cast<uint32_t>(estimate) << 23;
        }
      }
      t[idx] = r;
    }
    return t;
  }
  static const uint32_t* UnsignedEstimateTable(bool is_rsqrt) {
    static const uint32_t* const kRecpe = BuildUnsignedEstimateTable(false);
    static const uint32_t* const kRsqrte = BuildUnsignedEstimateTable(true);
    return is_rsqrt ? kRsqrte : kRecpe;
  }

