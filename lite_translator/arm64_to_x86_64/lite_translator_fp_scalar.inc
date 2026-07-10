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
// Family: scalar floating-point.
// Methods:
//   FpCondSelect, FpFixedPointConversion, FpDataProc3, FpMovImmediate,
//   FpIntConversion

  // FCSEL Sd|Dd|Hd, Sn|Dn|Hn, Sm|Dm|Hm, cond.
  //   if ConditionHolds(cond) then result = V[rn] else result = V[rm];
  //   V[rd] = ZeroExtend(result, 128);
  // ftype: 00 = S (FP32), 01 = D (FP64), 11 = H (FP16).  Encoding 10 is
  // reserved on ARM64 -- bail to the interpreter for safety.
  //
  // The chosen value is loaded into an XMM with the high lanes zero-
  // extended: MOVSS / MOVSD do this naturally; for FP16 the register is
  // first cleared with PXOR before PINSRW.  A 128-bit MOVDQU then writes
  // V[rd], matching the interpreter's `state_->cpu.v[rd] = 0` followed
  // by a sized partial copy.
  //
  // The condition switch mirrors ConditionalSelect() exactly (bit 14=Z,
  // 15=N, 8=C, 0=V in ThreadState::cpu.flags) -- the convention is "jump
  // to `done` if the condition is FALSE" so the fall-through path loads
  // V[rn] (the true case).
  void FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm,
                    uint8_t ftype, Decoder::Condition cond) {
    if (ftype != 0b00 && ftype != 0b01 && ftype != 0b11) {
      success_ = false;
      return;
    }

    SimdRegister xmm = AllocTempSimdReg();
    if (!success()) { return; }
    Register flags_reg = AllocTempReg();
    if (flags_reg == no_register) { success_ = false; return; }

    const int32_t v_rn_off = VRegOffset(rn);
    const int32_t v_rm_off = VRegOffset(rm);
    const int32_t v_rd_off = VRegOffset(rd);
    const int32_t flags_off = offsetof(ThreadState, cpu.flags);

    auto load_fp = [&](int32_t off) {
      switch (ftype) {
        case 0b00:
          as_.Movss(xmm, {.base = Assembler::rbp, .disp = off});
          break;
        case 0b01:
          as_.Movsd(xmm, {.base = Assembler::rbp, .disp = off});
          break;
        case 0b11:
          as_.Pxor(xmm, xmm);
          as_.Pinsrw(xmm, {.base = Assembler::rbp, .disp = off}, int8_t{0});
          break;
      }
    };

    // Default: load V[rm] (the condition-FALSE result).
    load_fp(v_rm_off);

    // Read NZCV (low 16 bits of ThreadState::cpu.flags).
    as_.Movzxwl(flags_reg, {.base = Assembler::rbp, .disp = flags_off});

    Assembler::Label* done = as_.MakeLabel();

    switch (cond) {
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
        // Reserved encodings on FCSEL; ConditionalSelect treats them
        // as always-true (fall through to the V[rn] override).
        break;
    }

    // Condition TRUE: overwrite with V[rn].
    load_fp(v_rn_off);

    as_.Bind(done);

    as_.Movdqu({.base = Assembler::rbp, .disp = v_rd_off}, xmm);
  }

  // FP <-> fixed-point conversion (GPR↔FPR, FpFixedPoint encoding family).
  //
  //   SCVTF / UCVTF: signed/unsigned integer in Wn/Xn → FP in Sd/Dd,
  //                  divided by 2^fbits.
  //   FCVTZS / FCVTZU: FP in Sn/Dn → signed/unsigned integer in Wd/Xd,
  //                    multiplied by 2^fbits, truncated toward zero,
  //                    ARM-saturated on NaN / overflow / negative.
  //
  // Lowering mirrors the unscaled FpIntConversion SCVTF/UCVTF/FCVTZS/
  // FCVTZU paths above with an inline 2^±fbits MUL inserted.
  // Multiplication by a power of 2 is exact in IEEE-754 (it only biases
  // the exponent), so the single-rounding semantics of CVTSI2{SS,SD}
  // (int→FP) and the saturation classification post-CVTT (FP→int)
  // carry through verbatim.
  //
  // FP16 (ftype != 00 && != 01) falls back to interpreter.
  void FpFixedPointConversion(const Decoder::FpFixedPointArgs& args) {
    using Op = Decoder::FpFixedPointOp;
    if (args.ftype != 0b00 && args.ftype != 0b01) {
      success_ = false;
      return;
    }
    const uint8_t fbits = args.fbits;
    if (fbits == 0 || fbits > (args.sf ? 64u : 32u)) {
      success_ = false;
      return;
    }

    // ------------------------------------------------------------------
    // SCVTF / UCVTF: integer in GP → FP in Vd, /2^fbits.
    // ------------------------------------------------------------------
    if (args.op == Op::kScvtf || args.op == Op::kUcvtf) {
      const bool is_unsigned = (args.op == Op::kUcvtf);
      SimdRegister xmm = AllocTempSimdReg();
      SimdRegister xscale = AllocTempSimdReg();
      if (!success()) { return; }
      int32_t dst_off = VRegOffset(args.rd);
      as_.Pxor(xmm, xmm);
      // Pre-store the zeroed xmm; rn=31 (XZR/WZR) drops out here with a
      // zero FP result naturally, no further work needed.
      if (args.rn < 31) {
        Register gp_val = GetReg(args.rn);
        if (is_unsigned && args.sf) {
          // sf=1 UCVTF: uint64 source.  Bit-63 may be set; branch on
          // sign bit, take direct CVTSI2SDQ/SSQ for values < 2^63;
          // halve / round-to-odd / convert / double-back for >= 2^63
          // to preserve single-rounding (mirror of the existing
          // FpIntConversion UCVTF lowering at lines 3299-3328).
          Register tmp = AllocTempReg();
          if (tmp == no_register) { success_ = false; return; }
          as_.Movq(tmp, gp_val);
          Assembler::Label* neg_path = as_.MakeLabel();
          Assembler::Label* done = as_.MakeLabel();
          as_.Testq(tmp, tmp);
          as_.Jcc(Assembler::Condition::kNegative, *neg_path);
          if (args.ftype == 0b00) as_.Cvtsi2ssq(xmm, tmp);
          else as_.Cvtsi2sdq(xmm, tmp);
          as_.Jmp(*done);
          as_.Bind(neg_path);
          Register low_bit = AllocTempReg();
          if (low_bit == no_register) { success_ = false; return; }
          as_.Movq(low_bit, tmp);
          as_.Andq(low_bit, static_cast<int32_t>(1));
          as_.Shrq(tmp, static_cast<int8_t>(1));
          as_.Orq(tmp, low_bit);
          if (args.ftype == 0b00) {
            as_.Cvtsi2ssq(xmm, tmp);
            as_.Addss(xmm, xmm);
          } else {
            as_.Cvtsi2sdq(xmm, tmp);
            as_.Addsd(xmm, xmm);
          }
          as_.Bind(done);
        } else if (is_unsigned) {
          // sf=0 UCVTF: uint32 source.  Zero-extend via Movl then
          // signed-Q convert (value ≤ UINT32_MAX < INT64_MAX so exact).
          Register tmp = AllocTempReg();
          if (tmp == no_register) { success_ = false; return; }
          as_.Movl(tmp, gp_val);
          if (args.ftype == 0b00) as_.Cvtsi2ssq(xmm, tmp);
          else as_.Cvtsi2sdq(xmm, tmp);
        } else if (args.sf) {
          // sf=1 SCVTF: int64 source.
          if (args.ftype == 0b00) as_.Cvtsi2ssq(xmm, gp_val);
          else as_.Cvtsi2sdq(xmm, gp_val);
        } else {
          // sf=0 SCVTF: int32 source.  CVTSI2{SS,SD}L sign-extends.
          if (args.ftype == 0b00) as_.Cvtsi2ssl(xmm, gp_val);
          else as_.Cvtsi2sdl(xmm, gp_val);
        }
      }
      // Multiply by 2^-fbits.  Power-of-2 multiply is exact.
      Register gp_scale = AllocTempReg();
      if (gp_scale == no_register) { success_ = false; return; }
      if (args.ftype == 0b00) {
        const uint32_t scale_bits =
            static_cast<uint32_t>(127u - fbits) << 23;
        as_.Movl(gp_scale, static_cast<int32_t>(scale_bits));
        as_.Movd(xscale, gp_scale);
        as_.Mulss(xmm, xscale);
      } else {
        const uint64_t scale_bits =
            static_cast<uint64_t>(1023u - fbits) << 52;
        as_.Movq(gp_scale, static_cast<int64_t>(scale_bits));
        as_.Movq(xscale, gp_scale);
        as_.Mulsd(xmm, xscale);
      }
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm);
      return;
    }

    // ------------------------------------------------------------------
    // FCVTZS / FCVTZU: FP in Vn → integer in GP, *2^fbits, trunc.
    // ------------------------------------------------------------------
    const bool is_unsigned = (args.op == Op::kFcvtzu);
    SimdRegister xmm = AllocTempSimdReg();
    SimdRegister xscale = AllocTempSimdReg();
    Register tmp = AllocTempReg();
    Register sign_tmp = AllocTempReg();
    if (xmm == no_simd_register || xscale == no_simd_register ||
        tmp == no_register || sign_tmp == no_register) {
      success_ = false;
      return;
    }
    int32_t src_off = VRegOffset(args.rn);
    if (args.ftype == 0b00) {
      as_.Movss(xmm, {.base = Assembler::rbp, .disp = src_off});
    } else {
      as_.Movsd(xmm, {.base = Assembler::rbp, .disp = src_off});
    }
    // Pre-multiply by 2^+fbits (exact; biases the exponent only).
    if (args.ftype == 0b00) {
      const uint32_t scale_bits =
          static_cast<uint32_t>(127u + fbits) << 23;
      as_.Movl(tmp, static_cast<int32_t>(scale_bits));
      as_.Movd(xscale, tmp);
      as_.Mulss(xmm, xscale);
    } else {
      const uint64_t scale_bits =
          static_cast<uint64_t>(1023u + fbits) << 52;
      as_.Movq(tmp, static_cast<int64_t>(scale_bits));
      as_.Movq(xscale, tmp);
      as_.Mulsd(xmm, xscale);
    }

    if (!is_unsigned) {
      // FCVTZS path: cvtt to signed dest width, then ARM saturation
      // fixup (NaN→0; positive overflow → INT_MAX; negative overflow
      // → INT_MIN matches the x86 indefinite already).
      if (args.sf) {
        if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
        else as_.Cvttsd2siq(tmp, xmm);
      } else {
        if (args.ftype == 0b00) as_.Cvttss2sil(tmp, xmm);
        else as_.Cvttsd2sil(tmp, xmm);
      }
      Assembler::Label* nan_path = as_.MakeLabel();
      Assembler::Label* done = as_.MakeLabel();
      if (args.ftype == 0b00) as_.Ucomiss(xmm, xmm);
      else as_.Ucomisd(xmm, xmm);
      as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
      // Sign of the SCALED FP source (post-multiply).
      if (args.ftype == 0b00) {
        as_.Movd(sign_tmp, xmm);
        as_.Testl(sign_tmp, sign_tmp);
      } else {
        as_.Movq(sign_tmp, xmm);
        as_.Testq(sign_tmp, sign_tmp);
      }
      as_.Jcc(Assembler::Condition::kNegative, *done);
      if (args.sf) as_.Testq(tmp, tmp);
      else as_.Testl(tmp, tmp);
      as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
      if (args.sf) {
        as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
      } else {
        as_.Movl(tmp, int32_t{INT32_MAX});
      }
      as_.Jmp(*done);
      as_.Bind(nan_path);
      if (args.sf) as_.Xorq(tmp, tmp);
      else as_.Xorl(tmp, tmp);
      as_.Bind(done);
      if (args.rd < 31) {
        SetReg(args.rd, tmp);
      }
      return;
    }

    // FCVTZU path.
    Assembler::Label* zero_path = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (args.ftype == 0b00) as_.Ucomiss(xmm, xmm);
    else as_.Ucomisd(xmm, xmm);
    as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
    if (args.ftype == 0b00) {
      as_.Movd(sign_tmp, xmm);
      as_.Testl(sign_tmp, sign_tmp);
    } else {
      as_.Movq(sign_tmp, xmm);
      as_.Testq(sign_tmp, sign_tmp);
    }
    as_.Jcc(Assembler::Condition::kNegative, *zero_path);

    if (!args.sf) {
      // sf=0 uint32 dest: cvtt-Q to int64, classify by upper-32 zero.
      if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
      else as_.Cvttsd2siq(tmp, xmm);
      as_.Movq(sign_tmp, tmp);
      as_.Shrq(sign_tmp, int8_t{32});
      as_.Testq(sign_tmp, sign_tmp);
      as_.Jcc(Assembler::Condition::kZero, *done);
      as_.Movl(tmp, int32_t{-1});  // saturate to UINT32_MAX
      as_.Jmp(*done);
      as_.Bind(zero_path);
      as_.Xorq(tmp, tmp);
    } else {
      // sf=1 uint64 dest: bound by 2^63 / 2^64 ladder.
      SimdRegister bound_xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Assembler::Label* sat_max = as_.MakeLabel();
      Assembler::Label* direct_path = as_.MakeLabel();
      if (args.ftype == 0b00) {
        as_.Movl(tmp, int32_t{0x5F000000});      // FP32(2^63)
        as_.Movd(bound_xmm, tmp);
      } else {
        as_.Movq(tmp, static_cast<int64_t>(0x43E0000000000000LL));  // FP64(2^63)
        as_.Movq(bound_xmm, tmp);
      }
      if (args.ftype == 0b00) as_.Ucomiss(xmm, bound_xmm);
      else as_.Ucomisd(xmm, bound_xmm);
      as_.Jcc(Assembler::Condition::kBelow, *direct_path);
      SimdRegister bound2_xmm = AllocTempSimdReg();
      if (!success()) { return; }
      if (args.ftype == 0b00) {
        as_.Movl(tmp, int32_t{0x5F800000});      // FP32(2^64)
        as_.Movd(bound2_xmm, tmp);
      } else {
        as_.Movq(tmp, static_cast<int64_t>(0x43F0000000000000LL));  // FP64(2^64)
        as_.Movq(bound2_xmm, tmp);
      }
      if (args.ftype == 0b00) as_.Ucomiss(xmm, bound2_xmm);
      else as_.Ucomisd(xmm, bound2_xmm);
      as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);
      // FP in [2^63, 2^64): subtract 2^63 (exact), cvtt, set bit 63.
      if (args.ftype == 0b00) {
        as_.Subss(xmm, bound_xmm);
        as_.Cvttss2siq(tmp, xmm);
      } else {
        as_.Subsd(xmm, bound_xmm);
        as_.Cvttsd2siq(tmp, xmm);
      }
      as_.Btsq(tmp, int8_t{63});
      as_.Jmp(*done);
      as_.Bind(sat_max);
      as_.Movq(tmp, static_cast<int64_t>(-1));  // UINT64_MAX
      as_.Jmp(*done);
      as_.Bind(direct_path);
      if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
      else as_.Cvttsd2siq(tmp, xmm);
      as_.Jmp(*done);
      as_.Bind(zero_path);
      as_.Xorq(tmp, tmp);
    }
    as_.Bind(done);
    if (args.rd < 31) {
      SetReg(args.rd, tmp);
    }
  }

  // FP data-processing (3 source): FMADD, FMSUB, FNMADD, FNMSUB at S/D.
  //
  //   FMADD  (o1=0, o0=0) : Rd = Ra + Rn*Rm
  //   FMSUB  (o1=0, o0=1) : Rd = Ra - Rn*Rm
  //   FNMADD (o1=1, o0=0) : Rd = -(Ra + Rn*Rm)
  //   FNMSUB (o1=1, o0=1) : Rd = Rn*Rm - Ra
  //
  // The ARM ARM specifies the single multiply-add is computed without
  // intermediate rounding (one rounding for the whole fused operation).
  // x86 FMA3 (VFMADD231SS/SD and friends) has the same semantic, so this is
  // the only sound JIT lowering.  Hosts without FMA3 fall back to the
  // interpreter (which uses libc fma() / fmaf() — also single-rounded),
  // not to a MUL+ADD pair (which would double-round).
  //
  // Mapping to x86 FMA231 form (dest = src1*src2 ± dest, ± from mnemonic;
  // load Ra into the dest slot, Rn into src1, Rm into src2):
  //   FMADD  -> Vfmadd231(ss|sd)  : Ra +  Rn*Rm
  //   FMSUB  -> Vfnmadd231(ss|sd) : Ra + -(Rn*Rm)  = Ra - Rn*Rm
  //   FNMADD -> Vfnmsub231(ss|sd) : -(Rn*Rm) - Ra  = -(Ra + Rn*Rm)
  //   FNMSUB -> Vfmsub231(ss|sd)  :  Rn*Rm - Ra
  //
  // FP16 (ftype=11) is JIT-lifted via FP16 -> FP32 -> FP64 and dispatched
  // through the SD form of VFMADD/VFNMADD/VFMSUB/VFNMSUB.  The interpreter
  // does the multiply-add in binary64 then narrows once to FP16 (single
  // rounding); binary64's 53-bit mantissa holds (binary16 * binary16) +
  // binary16 exactly, so an FP32-only round-trip would double-round on
  // some inputs.
  void FpDataProc3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra,
                   uint8_t ftype, bool o1, bool o0) {
    if (ftype != 0b00 && ftype != 0b01 && ftype != 0b11) {
      success_ = false;
      return;
    }
    if (!host_platform::kHasFMA) {
      success_ = false;
      return;
    }
    if (ftype == 0b11 && !host_platform::kHasF16C) {
      success_ = false;
      return;
    }
    const bool is_double = (ftype == 0b01);
    const bool is_half = (ftype == 0b11);

    const int32_t src_n_off = VRegOffset(rn);
    const int32_t src_m_off = VRegOffset(rm);
    const int32_t src_a_off = VRegOffset(ra);
    const int32_t dst_off = VRegOffset(rd);

    SimdRegister xmm_n = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_a = AllocTempSimdReg();
    if (!success()) { return; }

    if (is_double) {
      as_.Movsd(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movsd(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      as_.Movsd(xmm_a, {.base = Assembler::rbp, .disp = src_a_off});
    } else if (is_half) {
      // FP16 -> FP32 -> FP64 lift.  Pxor + Pinsrw isolates the 16-bit
      // input in lane 0 with FP16 +0.0 in lanes 1..3; Vcvtph2ps then
      // produces FP32 [value, 0, 0, 0]; Vcvtps2pd narrows the low 2
      // FP32 lanes to 2 FP64 lanes [FP64(value), FP64(0)].  The
      // preserved FP64(0) in lane 1 is consumed by the Vcvtpd2ps narrow
      // below and produces an FP32 +0.0 lane that Vcvtps2ph rounds to
      // FP16 +0.0 — matching the AArch64 zero-extend semantic for Hd.
      as_.Pxor(xmm_n, xmm_n);
      as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_n, xmm_n);
      as_.Vcvtps2pd(xmm_n, xmm_n);
      as_.Pxor(xmm_m, xmm_m);
      as_.Pinsrw(xmm_m, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_m, xmm_m);
      as_.Vcvtps2pd(xmm_m, xmm_m);
      as_.Pxor(xmm_a, xmm_a);
      as_.Pinsrw(xmm_a, {.base = Assembler::rbp, .disp = src_a_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_a, xmm_a);
      as_.Vcvtps2pd(xmm_a, xmm_a);
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      as_.Movss(xmm_a, {.base = Assembler::rbp, .disp = src_a_off});
    }

    // FP16 dispatches through the SD form because the value is already
    // binary64 in lane 0 after the Vcvtph2ps + Vcvtps2pd lift.
    const bool use_double_fma = is_double || is_half;
    if (!o1 && !o0) {
      if (use_double_fma) as_.Vfmadd231sd(xmm_a, xmm_n, xmm_m);
      else as_.Vfmadd231ss(xmm_a, xmm_n, xmm_m);
    } else if (!o1 && o0) {
      if (use_double_fma) as_.Vfnmadd231sd(xmm_a, xmm_n, xmm_m);
      else as_.Vfnmadd231ss(xmm_a, xmm_n, xmm_m);
    } else if (o1 && !o0) {
      if (use_double_fma) as_.Vfnmsub231sd(xmm_a, xmm_n, xmm_m);
      else as_.Vfnmsub231ss(xmm_a, xmm_n, xmm_m);
    } else {
      if (use_double_fma) as_.Vfmsub231sd(xmm_a, xmm_n, xmm_m);
      else as_.Vfmsub231ss(xmm_a, xmm_n, xmm_m);
    }

    if (is_half) {
      // Narrow FP64 -> FP32 -> FP16.  Vcvtpd2ps writes 2 FP32 lanes into
      // the low 64 bits and zeroes the upper 64; lane 1 was FP64 +0.0
      // (preserved by the SD FMA above), so the resulting FP32 has +0.0
      // in lanes 1..3.  Vcvtps2ph rounds 4 FP32 -> 4 FP16 and zeroes the
      // upper 64 bits: lane 0 = FP16(FMA result), lanes 1..3 = FP16(+0.0)
      // = 0.  Storing the full 128 bits gives the correct Hd layout
      // (result in bits[15:0], all other bits zero).
      as_.Vcvtpd2ps(xmm_a, xmm_a);
      as_.Vcvtps2ph(xmm_a, xmm_a, int8_t{0});
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
      return;
    }

    // ARM zero-extends Vd above the result lane.  Zero the full 128 bits
    // (reusing xmm_n as a scratch — n is no longer needed), then write the
    // scalar lane on top.
    as_.Pxor(xmm_n, xmm_n);
    as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_n);
    if (is_double) {
      as_.Movsd({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
    } else {
      as_.Movss({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
    }
  }

  // FMOV (scalar, immediate): JIT - load a FP constant into SIMD register
  void FpMovImmediate(uint8_t rd, uint8_t imm8, uint8_t ftype) {
    int32_t vreg_offset = VRegOffset(rd);
    SimdRegister xmm = AllocTempSimdReg();
    if (xmm == no_simd_register) {
      success_ = false;  // fallback to interpreter
      return;
    }
    // Zero the full 128-bit register
    as_.Pxor(xmm, xmm);
    as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);

    if (ftype == 0b00) {
      // Single-precision: expand imm8 to 32-bit float
      uint32_t sign = (imm8 >> 7) & 1;
      uint32_t exp6 = (imm8 >> 6) & 1;
      uint32_t exp_top = exp6 ? 0 : 1;
      uint32_t exp_rep = exp6 ? 0b11111 : 0b00000;
      uint32_t exp_low = (imm8 >> 4) & 0b11;
      uint32_t exp = (exp_top << 7) | (exp_rep << 2) | exp_low;
      uint32_t frac = (imm8 & 0xF) << 19;
      uint32_t imm32 = (sign << 31) | (exp << 23) | frac;
      Register tmp = AllocTempReg();
      as_.Movl(tmp, imm32);
      as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
    } else if (ftype == 0b01) {
      // Double-precision: expand imm8 to 64-bit double
      uint64_t sign = (imm8 >> 7) & 1;
      uint64_t exp6 = (imm8 >> 6) & 1;
      uint64_t exp_top = exp6 ? 0 : 1;
      uint64_t exp_rep = exp6 ? 0xFF : 0x00;
      uint64_t exp_low = (imm8 >> 4) & 0b11;
      uint64_t exp = (exp_top << 10) | (exp_rep << 2) | exp_low;
      uint64_t frac = static_cast<uint64_t>(imm8 & 0xF) << 48;
      uint64_t imm64 = (sign << 63) | (exp << 52) | frac;
      Register tmp = AllocTempReg();
      as_.Movq(tmp, imm64);
      as_.Movq({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
    } else {
      success_ = false;  // fallback for half-precision
    }
  }

  void FpIntConversion(const Decoder::FpIntConvArgs& args) {
    // JIT support for FMOV GP↔FP conversions
    uint8_t rmode = args.rmode;
    uint8_t opcode = args.op;

    // FMOV GP → FP: opcode=0b111, rmode=0b00
    if (rmode == 0b00 && opcode == 0b111) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      int32_t vreg_offset = VRegOffset(args.rd);
      // Zero the full 128-bit SIMD register first
      as_.Pxor(xmm, xmm);
      as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      if (args.rn < 31) {
        // Use GetReg to read the latest GP value (may be in a mapped register)
        Register gp_val = GetReg(args.rn);
        if (args.ftype == 0b00) {
          // FMOV Sd, Wn: 32-bit
          as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, gp_val);
        } else {
          // FMOV Dd, Xn: 64-bit
          as_.Movq({.base = Assembler::rbp, .disp = vreg_offset}, gp_val);
        }
      }
      // else: rn=31 (XZR), register is already zeroed
      return;
    }

    // FMOV FP → GP: opcode=0b110, rmode=0b00
    if (rmode == 0b00 && opcode == 0b110 && args.rd < 31) {
      Register tmp = AllocTempReg();
      int32_t vreg_offset = VRegOffset(args.rn);
      if (args.ftype == 0b00) {
        // FMOV Wd, Sn: 32-bit (zero-extend to 64)
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
      } else {
        // FMOV Xd, Dn: 64-bit
        as_.Movq(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
      }
      // Use SetReg to update the mapped register (not just ThreadState)
      SetReg(args.rd, tmp);
      return;
    }

    // FMOV Vd.D[1], Xn: opcode=0b111, rmode=0b01
    if (rmode == 0b01 && opcode == 0b111) {
      int32_t vreg_offset = VRegOffset(args.rd);
      if (args.rn < 31) {
        // Use GetReg to read the latest GP value (may be in a mapped register)
        Register gp_val = GetReg(args.rn);
        // Write to upper 64 bits (offset + 8)
        as_.Movq({.base = Assembler::rbp, .disp = vreg_offset + 8}, gp_val);
      } else {
        // XZR: write zero to upper 64 bits
        Register tmp = AllocTempReg();
        as_.Xorq(tmp, tmp);
        as_.Movq({.base = Assembler::rbp, .disp = vreg_offset + 8}, tmp);
      }
      return;
    }

    // FMOV Xd, Vn.D[1]: opcode=0b110, rmode=0b01
    if (rmode == 0b01 && opcode == 0b110 && args.rd < 31) {
      Register tmp = AllocTempReg();
      int32_t vreg_offset = VRegOffset(args.rn);
      // Read upper 64 bits (offset + 8)
      as_.Movq(tmp, {.base = Assembler::rbp, .disp = vreg_offset + 8});
      // Use SetReg to update the mapped register (not just ThreadState)
      SetReg(args.rd, tmp);
      return;
    }

    // SCVTF / UCVTF (scalar, integer→FP, rmode=00): Vd = (FP)(s|u)Wn/Xn.
    //
    //   rmode=00, opcode=010 -> SCVTF: signed Wn/Xn -> Sd/Dd
    //   rmode=00, opcode=011 -> UCVTF: unsigned Wn/Xn -> Sd/Dd
    //
    // x86 CVTSI2SS/SD always reads a signed source.  For SCVTF this matches
    // ARM directly: the L-variant takes a 32-bit source and sign-extends it
    // into the FP unit's signed conversion, the Q-variant takes a 64-bit
    // source.  For UCVTF with sf=0 (32-bit unsigned source) we zero-extend
    // the source into a 64-bit GPR via MOVL and then use CVTSI2SS/SD Q-form
    // — the value always fits in int64 so the signed convert is exact.
    // UCVTF with sf=1 (64-bit unsigned) splits at bit 63: if the source is
    // < 2^63 the direct Q-form is exact; otherwise we apply the textbook
    // "halve, round-to-odd, convert, double" fix-up.  The round-to-odd
    // halve (`v >> 1) | (v & 1)`) preserves enough information that the
    // subsequent round-to-nearest-even after doubling lands on the same
    // bit pattern clang emits for `static_cast<float|double>(uint64_t)`.
    if (rmode == 0b00 && (opcode == 0b010 || opcode == 0b011) &&
        (args.ftype == 0b00 || args.ftype == 0b01)) {
      const bool is_unsigned = (opcode == 0b011);
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      int32_t dst_off = VRegOffset(args.rd);
      as_.Pxor(xmm, xmm);
      if (args.rn < 31) {
        Register gp_val = GetReg(args.rn);
        if (is_unsigned && args.sf) {
          // sf=1, opcode=011: 64-bit unsigned.  Branch on the sign bit:
          // direct convert for values < 2^63, halve/convert/double for the
          // upper half.  Forward jumps keep the positive (fast) path
          // straight-line; only the >= 2^63 case takes the fix-up.
          Register tmp = AllocTempReg();
          if (tmp == no_register) { success_ = false; return; }
          as_.Movq(tmp, gp_val);
          Assembler::Label* neg_path = as_.MakeLabel();
          Assembler::Label* done = as_.MakeLabel();
          as_.Testq(tmp, tmp);
          as_.Jcc(Assembler::Condition::kNegative, *neg_path);
          if (args.ftype == 0b00) as_.Cvtsi2ssq(xmm, tmp);
          else as_.Cvtsi2sdq(xmm, tmp);
          as_.Jmp(*done);
          as_.Bind(neg_path);
          Register low_bit = AllocTempReg();
          if (low_bit == no_register) { success_ = false; return; }
          as_.Movq(low_bit, tmp);
          as_.Andq(low_bit, static_cast<int32_t>(1));  // round-to-odd LSB
          as_.Shrq(tmp, static_cast<int8_t>(1));       // logical halve
          as_.Orq(tmp, low_bit);                       // tmp = halve|LSB
          if (args.ftype == 0b00) {
            as_.Cvtsi2ssq(xmm, tmp);
            as_.Addss(xmm, xmm);                       // double back
          } else {
            as_.Cvtsi2sdq(xmm, tmp);
            as_.Addsd(xmm, xmm);                       // double back
          }
          as_.Bind(done);
        } else if (is_unsigned) {
          // sf=0, opcode=011: 32-bit unsigned.  Zero-extend via MOVL then
          // convert as signed 64-bit (value <= UINT32_MAX < INT64_MAX).
          Register tmp = AllocTempReg();
          as_.Movl(tmp, gp_val);  // 32-bit MOV zero-extends to 64
          if (args.ftype == 0b00) as_.Cvtsi2ssq(xmm, tmp);
          else as_.Cvtsi2sdq(xmm, tmp);
        } else if (args.sf) {
          // sf=1, opcode=010: 64-bit signed, direct Q-form.
          if (args.ftype == 0b00) as_.Cvtsi2ssq(xmm, gp_val);
          else as_.Cvtsi2sdq(xmm, gp_val);
        } else {
          // sf=0, opcode=010: 32-bit signed, L-form.  CVTSI2{SS,SD}L reads
          // the 32-bit subreg as int32 (sign-extends to the FP convert).
          if (args.ftype == 0b00) as_.Cvtsi2ssl(xmm, gp_val);
          else as_.Cvtsi2sdl(xmm, gp_val);
        }
      }
      // Legacy CVTSI2{SS,SD} leaves xmm[127:32]/xmm[127:64] unchanged; the
      // pre-PXOR guarantees those bits are zero, so the single MOVDQU below
      // writes the lane plus ARM's required zero-extension above it in one
      // shot.  rn=31 (XZR/WZR) falls through here too — xmm is still zero,
      // matching the (FP)(0) result.
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm);
      return;
    }

    // FCVTZS (scalar, FP -> signed int, truncate toward zero):
    //   rmode=11, opcode=000, ftype in {00, 01}.
    //
    //   sf  ftype  insn               x86 lowering
    //    0   00    FCVTZS Wd, Sn      Cvttss2sil + ARM saturation fix-up
    //    0   01    FCVTZS Wd, Dn      Cvttsd2sil + ARM saturation fix-up
    //    1   00    FCVTZS Xd, Sn      Cvttss2siq + ARM saturation fix-up
    //    1   01    FCVTZS Xd, Dn      Cvttsd2siq + ARM saturation fix-up
    //
    // x86 cvtt{ss,sd}2si returns the destination type's INT_MIN ("indefinite")
    // for any out-of-range input (NaN, +/-Inf, overflow), so we must rebuild
    // ARM's by-sign saturation:
    //   NaN              -> 0
    //   positive overflow -> INT_MAX
    //   negative overflow -> INT_MIN (matches the indefinite already)
    // After truncate-convert we classify by (a) parity flag (PF=1 iff NaN)
    // and (b) the FP source's sign bit (extracted via Movd/Movq):
    //   NaN              -> branch to nan_path, write 0.
    //   non-NaN, FP < 0  -> keep tmp (correct in-range neg or INT_MIN).
    //   non-NaN, FP >= 0 -> if tmp >= 0 keep it (in-range pos); else
    //                       tmp == INT_MIN i.e. positive overflow, write INT_MAX.
    // -0.0 takes the non-NaN/FP<0 branch but tmp is 0 (cvtt returns 0 for
    // -0.0), so keeping tmp gives the ARM-correct 0.
    if (rmode == 0b11 && opcode == 0b000 &&
        (args.ftype == 0b00 || args.ftype == 0b01)) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp = AllocTempReg();
      if (tmp == no_register) { success_ = false; return; }
      Register sign_tmp = AllocTempReg();
      if (sign_tmp == no_register) { success_ = false; return; }
      int32_t src_off = VRegOffset(args.rn);
      if (args.ftype == 0b00) {
        as_.Movss(xmm, {.base = Assembler::rbp, .disp = src_off});
      } else {
        as_.Movsd(xmm, {.base = Assembler::rbp, .disp = src_off});
      }
      if (args.sf) {
        if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
        else as_.Cvttsd2siq(tmp, xmm);
      } else {
        if (args.ftype == 0b00) as_.Cvttss2sil(tmp, xmm);
        else as_.Cvttsd2sil(tmp, xmm);
      }
      Assembler::Label* nan_path = as_.MakeLabel();
      Assembler::Label* done = as_.MakeLabel();
      // NaN check.
      if (args.ftype == 0b00) as_.Ucomiss(xmm, xmm);
      else as_.Ucomisd(xmm, xmm);
      as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
      // Sign-of-FP check via raw bits.
      if (args.ftype == 0b00) {
        as_.Movd(sign_tmp, xmm);
        as_.Testl(sign_tmp, sign_tmp);
      } else {
        as_.Movq(sign_tmp, xmm);
        as_.Testq(sign_tmp, sign_tmp);
      }
      as_.Jcc(Assembler::Condition::kNegative, *done);
      // Non-negative FP: keep tmp if tmp >= 0 (in-range); else overwrite
      // with INT_MAX (positive overflow).
      if (args.sf) as_.Testq(tmp, tmp);
      else as_.Testl(tmp, tmp);
      as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
      if (args.sf) {
        as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
      } else {
        as_.Movl(tmp, int32_t{INT32_MAX});
      }
      as_.Jmp(*done);
      as_.Bind(nan_path);
      if (args.sf) as_.Xorq(tmp, tmp);
      else as_.Xorl(tmp, tmp);
      as_.Bind(done);
      if (args.rd < 31) {
        // sf=0: tmp's upper 32 bits are already zero (x86-64 writes to
        // 32-bit subregs auto-zero-extend), matching Wd-write semantics.
        SetReg(args.rd, tmp);
      }
      return;
    }

    // FCVTZU (scalar, FP -> unsigned int, truncate toward zero):
    //   rmode=11, opcode=001, ftype in {00, 01}.
    //
    //   sf  ftype  insn               Strategy
    //    0   00    FCVTZU Wd, Sn      Cvttss2siq + upper-32-saturation
    //    0   01    FCVTZU Wd, Dn      Cvttsd2siq + upper-32-saturation
    //    1   00    FCVTZU Xd, Sn      offset-trick for FP in [2^63, 2^64)
    //    1   01    FCVTZU Xd, Dn      offset-trick for FP in [2^63, 2^64)
    //
    // ARM saturation rules:
    //   NaN              -> 0
    //   FP < 0 (incl -0) -> 0 (FCVTZU(-0.0) = 0 because the cvtt result is 0)
    //   FP > UINT*_MAX   -> UINT*_MAX
    //   in-range         -> trunc(FP) as unsigned
    //
    // sf=0: x86 cvtt-Q form yields an int64 result.  Valid unsigned outputs
    // are in [0, UINT32_MAX]; the upper 32 bits are zero in that range and
    // non-zero for both INT64_MIN indefinite and any finite FP > UINT32_MAX
    // — so a single upper-32 zero test classifies in-range vs. overflow.
    //
    // sf=1: cvtt-Q covers FP in [0, 2^63) directly.  For FP >= 2^63, the
    // standard offset trick (FP - 2^63 in FP, cvtt to int64, OR bit 63
    // back in) recovers the unsigned result exactly.  The subtract is
    // exact because both 2^63 and FP - 2^63 land on representable values
    // at the same exponent step in FP32/FP64.  FP >= 2^64 saturates to
    // UINT64_MAX; +Inf falls into this bucket via the FP comparison.
    if (rmode == 0b11 && opcode == 0b001 &&
        (args.ftype == 0b00 || args.ftype == 0b01)) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp = AllocTempReg();
      if (tmp == no_register) { success_ = false; return; }
      Register sign_tmp = AllocTempReg();
      if (sign_tmp == no_register) { success_ = false; return; }
      int32_t src_off = VRegOffset(args.rn);
      if (args.ftype == 0b00) {
        as_.Movss(xmm, {.base = Assembler::rbp, .disp = src_off});
      } else {
        as_.Movsd(xmm, {.base = Assembler::rbp, .disp = src_off});
      }

      Assembler::Label* zero_path = as_.MakeLabel();
      Assembler::Label* done = as_.MakeLabel();

      // NaN check via Ucomi self (PF=1 iff NaN).
      if (args.ftype == 0b00) as_.Ucomiss(xmm, xmm);
      else as_.Ucomisd(xmm, xmm);
      as_.Jcc(Assembler::Condition::kParityEven, *zero_path);

      // FP sign bit check via raw bits.  ARM FCVTZU(-0.0) = 0, which falls
      // out of this branch naturally (cvtt(-0.0) = 0 in the zero_path).
      if (args.ftype == 0b00) {
        as_.Movd(sign_tmp, xmm);
        as_.Testl(sign_tmp, sign_tmp);
      } else {
        as_.Movq(sign_tmp, xmm);
        as_.Testq(sign_tmp, sign_tmp);
      }
      as_.Jcc(Assembler::Condition::kNegative, *zero_path);

      if (!args.sf) {
        // sf=0 (uint32 destination): cvtt-Q always fits in int64.  Either
        // the upper 32 bits of the result are zero (in-range, low 32 are
        // the answer) or non-zero (saturate to UINT32_MAX).
        if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
        else as_.Cvttsd2siq(tmp, xmm);
        // sign_tmp is dead after the FP-sign jump above; reuse as upper-32 scratch.
        as_.Movq(sign_tmp, tmp);
        as_.Shrq(sign_tmp, int8_t{32});
        as_.Testq(sign_tmp, sign_tmp);
        as_.Jcc(Assembler::Condition::kZero, *done);
        // Positive overflow: saturate to UINT32_MAX (low 32 ones, upper auto-zero).
        as_.Movl(tmp, int32_t{-1});
        as_.Jmp(*done);
        as_.Bind(zero_path);
        as_.Xorq(tmp, tmp);
      } else {
        // sf=1 (uint64 destination).
        SimdRegister bound_xmm = AllocTempSimdReg();
        if (!success()) { return; }
        Assembler::Label* sat_max = as_.MakeLabel();
        Assembler::Label* direct_path = as_.MakeLabel();

        // Load 2^63 as FP constant into bound_xmm via the existing GP scratch.
        // FP32(2^63) = 0x5F000000, FP64(2^63) = 0x43E0000000000000.
        if (args.ftype == 0b00) {
          as_.Movl(tmp, int32_t{0x5F000000});
          as_.Movd(bound_xmm, tmp);
        } else {
          as_.Movq(tmp, static_cast<int64_t>(0x43E0000000000000LL));
          as_.Movq(bound_xmm, tmp);
        }

        // FP < 2^63 -> direct cvtt-Q gives an exact non-negative int64.
        if (args.ftype == 0b00) as_.Ucomiss(xmm, bound_xmm);
        else as_.Ucomisd(xmm, bound_xmm);
        as_.Jcc(Assembler::Condition::kBelow, *direct_path);

        // FP >= 2^63: now check upper bound.  Load 2^64 into sign_tmp's
        // companion register, materialize as FP, and compare.  After the
        // compare we discard the 2^64 constant and reuse bound_xmm (still
        // holding 2^63) for the subtract.
        SimdRegister bound2_xmm = AllocTempSimdReg();
        if (!success()) { return; }
        if (args.ftype == 0b00) {
          as_.Movl(tmp, int32_t{0x5F800000});  // FP32(2^64)
          as_.Movd(bound2_xmm, tmp);
        } else {
          as_.Movq(tmp, static_cast<int64_t>(0x43F0000000000000LL));  // FP64(2^64)
          as_.Movq(bound2_xmm, tmp);
        }
        if (args.ftype == 0b00) as_.Ucomiss(xmm, bound2_xmm);
        else as_.Ucomisd(xmm, bound2_xmm);
        as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);

        // FP in [2^63, 2^64): subtract 2^63 (exact), cvtt, set bit 63.
        if (args.ftype == 0b00) {
          as_.Subss(xmm, bound_xmm);
          as_.Cvttss2siq(tmp, xmm);
        } else {
          as_.Subsd(xmm, bound_xmm);
          as_.Cvttsd2siq(tmp, xmm);
        }
        as_.Btsq(tmp, int8_t{63});
        as_.Jmp(*done);

        as_.Bind(sat_max);
        as_.Movq(tmp, static_cast<int64_t>(-1));  // UINT64_MAX
        as_.Jmp(*done);

        as_.Bind(direct_path);
        if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
        else as_.Cvttsd2siq(tmp, xmm);
        as_.Jmp(*done);

        as_.Bind(zero_path);
        as_.Xorq(tmp, tmp);
      }
      as_.Bind(done);
      if (args.rd < 31) {
        SetReg(args.rd, tmp);
      }
      return;
    }

    // FCVTNS/PS/MS (signed) and FCVTNU/PU/MU (unsigned) scalar FP -> int.
    //   rmode in {00, 01, 10}, opcode in {000, 001}, ftype in {00, 01}.
    //
    //   rmode  insn family               ROUNDSS/ROUNDSD imm
    //    00    FCVTNS / FCVTNU (RNE)     0x00
    //    01    FCVTPS / FCVTPU (ceil)    0x02  (round toward +inf)
    //    10    FCVTMS / FCVTMU (floor)   0x01  (round toward -inf)
    //
    // Strategy: round the FP value in FP domain via ROUNDSS/ROUNDSD, then
    // reuse the FCVTZS / FCVTZU saturation fix-up verbatim (the result of
    // ROUNDSS on a finite value is always a representable integer, so the
    // subsequent truncating cvtt is a no-op for in-range inputs and still
    // produces the x86 INT_MIN indefinite on out-of-range / Inf — exactly
    // the input the saturation fix-up expects).  NaN, ±Inf and the sign of
    // zero pass through ROUNDSS/ROUNDSD unchanged, so the NaN-check (PF=1
    // from Ucomi self) and the FP-sign-bit branch still classify them
    // correctly.  FCVTAS / FCVTAU (rmode=00, opcode in {100, 101}, ties-
    // away-from-zero) has no native x86 ROUND mode; see the dedicated
    // magnitude-gated add-half-and-trunc path further down.
    if ((rmode == 0b00 || rmode == 0b01 || rmode == 0b10) &&
        (opcode == 0b000 || opcode == 0b001) &&
        (args.ftype == 0b00 || args.ftype == 0b01)) {
      // round_imm: imm[1:0] = rounding mode, imm[3] = suppress inexact (we
      // don't model FPSR).
      int8_t round_imm;
      if (rmode == 0b00) round_imm = int8_t{0x08};       // RNE + suppress
      else if (rmode == 0b01) round_imm = int8_t{0x0A};  // round toward +inf + suppress
      else round_imm = int8_t{0x09};                     // round toward -inf + suppress
      const bool is_unsigned = (opcode == 0b001);

      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp = AllocTempReg();
      if (tmp == no_register) { success_ = false; return; }
      Register sign_tmp = AllocTempReg();
      if (sign_tmp == no_register) { success_ = false; return; }
      int32_t src_off = VRegOffset(args.rn);
      if (args.ftype == 0b00) {
        as_.Movss(xmm, {.base = Assembler::rbp, .disp = src_off});
        as_.Roundss(xmm, xmm, round_imm);
      } else {
        as_.Movsd(xmm, {.base = Assembler::rbp, .disp = src_off});
        as_.Roundsd(xmm, xmm, round_imm);
      }

      if (!is_unsigned) {
        // FCVTNS / FCVTPS / FCVTMS: signed saturation fix-up (mirror of
        // FCVTZS above, applied to the now-rounded xmm).
        if (args.sf) {
          if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
          else as_.Cvttsd2siq(tmp, xmm);
        } else {
          if (args.ftype == 0b00) as_.Cvttss2sil(tmp, xmm);
          else as_.Cvttsd2sil(tmp, xmm);
        }
        Assembler::Label* nan_path = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        if (args.ftype == 0b00) as_.Ucomiss(xmm, xmm);
        else as_.Ucomisd(xmm, xmm);
        as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
        if (args.ftype == 0b00) {
          as_.Movd(sign_tmp, xmm);
          as_.Testl(sign_tmp, sign_tmp);
        } else {
          as_.Movq(sign_tmp, xmm);
          as_.Testq(sign_tmp, sign_tmp);
        }
        as_.Jcc(Assembler::Condition::kNegative, *done);
        if (args.sf) as_.Testq(tmp, tmp);
        else as_.Testl(tmp, tmp);
        as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
        if (args.sf) {
          as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
        } else {
          as_.Movl(tmp, int32_t{INT32_MAX});
        }
        as_.Jmp(*done);
        as_.Bind(nan_path);
        if (args.sf) as_.Xorq(tmp, tmp);
        else as_.Xorl(tmp, tmp);
        as_.Bind(done);
      } else {
        // FCVTNU / FCVTPU / FCVTMU: unsigned saturation fix-up (mirror of
        // FCVTZU above, applied to the now-rounded xmm).
        Assembler::Label* zero_path = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        if (args.ftype == 0b00) as_.Ucomiss(xmm, xmm);
        else as_.Ucomisd(xmm, xmm);
        as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
        if (args.ftype == 0b00) {
          as_.Movd(sign_tmp, xmm);
          as_.Testl(sign_tmp, sign_tmp);
        } else {
          as_.Movq(sign_tmp, xmm);
          as_.Testq(sign_tmp, sign_tmp);
        }
        as_.Jcc(Assembler::Condition::kNegative, *zero_path);
        if (!args.sf) {
          // sf=0 (uint32): cvtt-Q gives int64 in [0, UINT32_MAX] iff the
          // upper 32 bits are zero; non-zero means overflow.
          if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
          else as_.Cvttsd2siq(tmp, xmm);
          as_.Movq(sign_tmp, tmp);
          as_.Shrq(sign_tmp, int8_t{32});
          as_.Testq(sign_tmp, sign_tmp);
          as_.Jcc(Assembler::Condition::kZero, *done);
          as_.Movl(tmp, int32_t{-1});  // UINT32_MAX
          as_.Jmp(*done);
          as_.Bind(zero_path);
          as_.Xorq(tmp, tmp);
        } else {
          // sf=1 (uint64): offset-trick for FP in [2^63, 2^64); saturate
          // to UINT64_MAX for FP >= 2^64 (incl +Inf).
          SimdRegister bound_xmm = AllocTempSimdReg();
          if (!success()) { return; }
          Assembler::Label* sat_max = as_.MakeLabel();
          Assembler::Label* direct_path = as_.MakeLabel();
          if (args.ftype == 0b00) {
            as_.Movl(tmp, int32_t{0x5F000000});               // FP32(2^63)
            as_.Movd(bound_xmm, tmp);
          } else {
            as_.Movq(tmp, static_cast<int64_t>(0x43E0000000000000LL));  // FP64(2^63)
            as_.Movq(bound_xmm, tmp);
          }
          if (args.ftype == 0b00) as_.Ucomiss(xmm, bound_xmm);
          else as_.Ucomisd(xmm, bound_xmm);
          as_.Jcc(Assembler::Condition::kBelow, *direct_path);
          SimdRegister bound2_xmm = AllocTempSimdReg();
          if (!success()) { return; }
          if (args.ftype == 0b00) {
            as_.Movl(tmp, int32_t{0x5F800000});               // FP32(2^64)
            as_.Movd(bound2_xmm, tmp);
          } else {
            as_.Movq(tmp, static_cast<int64_t>(0x43F0000000000000LL));  // FP64(2^64)
            as_.Movq(bound2_xmm, tmp);
          }
          if (args.ftype == 0b00) as_.Ucomiss(xmm, bound2_xmm);
          else as_.Ucomisd(xmm, bound2_xmm);
          as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);
          if (args.ftype == 0b00) {
            as_.Subss(xmm, bound_xmm);
            as_.Cvttss2siq(tmp, xmm);
          } else {
            as_.Subsd(xmm, bound_xmm);
            as_.Cvttsd2siq(tmp, xmm);
          }
          as_.Btsq(tmp, int8_t{63});
          as_.Jmp(*done);
          as_.Bind(sat_max);
          as_.Movq(tmp, static_cast<int64_t>(-1));  // UINT64_MAX
          as_.Jmp(*done);
          as_.Bind(direct_path);
          if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
          else as_.Cvttsd2siq(tmp, xmm);
          as_.Jmp(*done);
          as_.Bind(zero_path);
          as_.Xorq(tmp, tmp);
        }
        as_.Bind(done);
      }
      if (args.rd < 31) {
        SetReg(args.rd, tmp);
      }
      return;
    }

    // FCVTAS (signed) / FCVTAU (unsigned) scalar FP -> int, round to nearest
    // ties-AWAY-from-zero:
    //   rmode=00, opcode=100 (signed) or 101 (unsigned), ftype in {00, 01}.
    //
    // x86 has no native ROUND* imm for ties-away.  Strategy: the canonical
    // "trunc(x + copysign(0.5, x))" trick, gated on magnitude.  For
    // |x| < 2^23 (FP32) / 2^52 (FP64) the add lands a tie x=k+/-0.5 exactly
    // on k+/-1, and trunc rounds it away from zero.  For |x| >= threshold,
    // x is already an exact integer (FP step >= 1 at that magnitude), so we
    // skip the add and trunc directly — without the gate, odd integers x in
    // [threshold, dst_max) would be wrongly bumped to x+1 by RNE of x+0.5
    // (since 0.5 lies below the LSB and RNE rounds half-to-even).
    //
    // After the magnitude branch the saturation fix-up is identical to the
    // FCVTZS/FCVTZU paths above: classify by NaN (Ucomi self -> PF=1) and
    // by post-rounded FP sign, saturate to INT_MAX / INT_MIN / 0 /
    // UINT*_MAX accordingly.
    if (rmode == 0b00 && (opcode == 0b100 || opcode == 0b101) &&
        (args.ftype == 0b00 || args.ftype == 0b01)) {
      const bool is_unsigned = (opcode == 0b101);

      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister half_xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp = AllocTempReg();
      if (tmp == no_register) { success_ = false; return; }
      Register sign_tmp = AllocTempReg();
      if (sign_tmp == no_register) { success_ = false; return; }

      int32_t src_off = VRegOffset(args.rn);
      if (args.ftype == 0b00) {
        as_.Movss(xmm, {.base = Assembler::rbp, .disp = src_off});
      } else {
        as_.Movsd(xmm, {.base = Assembler::rbp, .disp = src_off});
      }

      // Magnitude gate via integer-domain compare on |bits(x)|.  IEEE-754
      // bits compare as unsigned int for non-negative values: larger
      // exponent -> larger bit value; mantissa breaks ties left-to-right.
      // NaN bits (exponent all 1) and +/-Inf both exceed any finite
      // threshold, so they skip the add (NaN+0.5=NaN passes through
      // ROUNDSS and the saturation fix-up classifies them correctly).
      Assembler::Label* skip_add = as_.MakeLabel();
      if (args.ftype == 0b00) {
        as_.Movd(sign_tmp, xmm);
        as_.Andl(sign_tmp, int32_t{0x7FFFFFFF});
        as_.Cmpl(sign_tmp, int32_t{0x4B000000});  // FP32 bits of 2^23
        as_.Jcc(Assembler::Condition::kAboveEqual, *skip_add);
      } else {
        as_.Movq(sign_tmp, xmm);
        as_.Movq(tmp, static_cast<int64_t>(0x7FFFFFFFFFFFFFFFLL));
        as_.Andq(sign_tmp, tmp);
        as_.Movq(tmp, static_cast<int64_t>(0x4330000000000000LL));  // FP64 bits of 2^52
        as_.Cmpq(sign_tmp, tmp);
        as_.Jcc(Assembler::Condition::kAboveEqual, *skip_add);
      }

      // |x| < threshold: build copysign(0.5, x) in half_xmm, add to xmm.
      // FP32 bits of copysign(0.5, x) = (bits(x) & 0x80000000) | 0x3F000000.
      // FP64 high 32 bits = (high32(bits(x)) & 0x80000000) | 0x3FE00000;
      // low 32 bits are 0.  Mirrors the FRINTA pattern at line ~5400.
      if (args.ftype == 0b00) {
        as_.Movd(tmp, xmm);
        as_.Andl(tmp, static_cast<int32_t>(0x80000000));
        as_.Orl(tmp, int32_t{0x3F000000});
        as_.Movd(half_xmm, tmp);
        as_.Addss(xmm, half_xmm);
      } else {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = src_off + 4});
        as_.Andl(tmp, static_cast<int32_t>(0x80000000));
        as_.Orl(tmp, int32_t{0x3FE00000});
        as_.Pxor(half_xmm, half_xmm);
        as_.Pinsrd(half_xmm, tmp, int8_t{1});
        as_.Addsd(xmm, half_xmm);
      }
      as_.Bind(skip_add);

      // Truncate toward zero.  For exact-integer skipped-add inputs this
      // is a no-op; for |x| < threshold inputs the add already produced a
      // (possibly half-)integer whose trunc is the ARM result.
      if (args.ftype == 0b00) {
        as_.Roundss(xmm, xmm, int8_t{0x03});
      } else {
        as_.Roundsd(xmm, xmm, int8_t{0x03});
      }

      if (!is_unsigned) {
        // Signed saturation fix-up (mirror of FCVTZS).
        if (args.sf) {
          if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
          else as_.Cvttsd2siq(tmp, xmm);
        } else {
          if (args.ftype == 0b00) as_.Cvttss2sil(tmp, xmm);
          else as_.Cvttsd2sil(tmp, xmm);
        }
        Assembler::Label* nan_path = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        if (args.ftype == 0b00) as_.Ucomiss(xmm, xmm);
        else as_.Ucomisd(xmm, xmm);
        as_.Jcc(Assembler::Condition::kParityEven, *nan_path);
        if (args.ftype == 0b00) {
          as_.Movd(sign_tmp, xmm);
          as_.Testl(sign_tmp, sign_tmp);
        } else {
          as_.Movq(sign_tmp, xmm);
          as_.Testq(sign_tmp, sign_tmp);
        }
        as_.Jcc(Assembler::Condition::kNegative, *done);
        if (args.sf) as_.Testq(tmp, tmp);
        else as_.Testl(tmp, tmp);
        as_.Jcc(Assembler::Condition::kPositiveOrZero, *done);
        if (args.sf) {
          as_.Movq(tmp, static_cast<int64_t>(INT64_MAX));
        } else {
          as_.Movl(tmp, int32_t{INT32_MAX});
        }
        as_.Jmp(*done);
        as_.Bind(nan_path);
        if (args.sf) as_.Xorq(tmp, tmp);
        else as_.Xorl(tmp, tmp);
        as_.Bind(done);
      } else {
        // Unsigned saturation fix-up (mirror of FCVTZU).
        Assembler::Label* zero_path = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        if (args.ftype == 0b00) as_.Ucomiss(xmm, xmm);
        else as_.Ucomisd(xmm, xmm);
        as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
        if (args.ftype == 0b00) {
          as_.Movd(sign_tmp, xmm);
          as_.Testl(sign_tmp, sign_tmp);
        } else {
          as_.Movq(sign_tmp, xmm);
          as_.Testq(sign_tmp, sign_tmp);
        }
        as_.Jcc(Assembler::Condition::kNegative, *zero_path);
        if (!args.sf) {
          // sf=0 (uint32): cvtt-Q + upper-32 zero classifies in-range.
          if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
          else as_.Cvttsd2siq(tmp, xmm);
          as_.Movq(sign_tmp, tmp);
          as_.Shrq(sign_tmp, int8_t{32});
          as_.Testq(sign_tmp, sign_tmp);
          as_.Jcc(Assembler::Condition::kZero, *done);
          as_.Movl(tmp, int32_t{-1});  // UINT32_MAX
          as_.Jmp(*done);
          as_.Bind(zero_path);
          as_.Xorq(tmp, tmp);
        } else {
          // sf=1 (uint64): offset-trick + 2^64 saturation.
          SimdRegister bound_xmm = AllocTempSimdReg();
          if (!success()) { return; }
          Assembler::Label* sat_max = as_.MakeLabel();
          Assembler::Label* direct_path = as_.MakeLabel();
          if (args.ftype == 0b00) {
            as_.Movl(tmp, int32_t{0x5F000000});               // FP32(2^63)
            as_.Movd(bound_xmm, tmp);
          } else {
            as_.Movq(tmp, static_cast<int64_t>(0x43E0000000000000LL));  // FP64(2^63)
            as_.Movq(bound_xmm, tmp);
          }
          if (args.ftype == 0b00) as_.Ucomiss(xmm, bound_xmm);
          else as_.Ucomisd(xmm, bound_xmm);
          as_.Jcc(Assembler::Condition::kBelow, *direct_path);
          SimdRegister bound2_xmm = AllocTempSimdReg();
          if (!success()) { return; }
          if (args.ftype == 0b00) {
            as_.Movl(tmp, int32_t{0x5F800000});               // FP32(2^64)
            as_.Movd(bound2_xmm, tmp);
          } else {
            as_.Movq(tmp, static_cast<int64_t>(0x43F0000000000000LL));  // FP64(2^64)
            as_.Movq(bound2_xmm, tmp);
          }
          if (args.ftype == 0b00) as_.Ucomiss(xmm, bound2_xmm);
          else as_.Ucomisd(xmm, bound2_xmm);
          as_.Jcc(Assembler::Condition::kAboveEqual, *sat_max);
          if (args.ftype == 0b00) {
            as_.Subss(xmm, bound_xmm);
            as_.Cvttss2siq(tmp, xmm);
          } else {
            as_.Subsd(xmm, bound_xmm);
            as_.Cvttsd2siq(tmp, xmm);
          }
          as_.Btsq(tmp, int8_t{63});
          as_.Jmp(*done);
          as_.Bind(sat_max);
          as_.Movq(tmp, static_cast<int64_t>(-1));  // UINT64_MAX
          as_.Jmp(*done);
          as_.Bind(direct_path);
          if (args.ftype == 0b00) as_.Cvttss2siq(tmp, xmm);
          else as_.Cvttsd2siq(tmp, xmm);
          as_.Jmp(*done);
          as_.Bind(zero_path);
          as_.Xorq(tmp, tmp);
        }
        as_.Bind(done);
      }
      if (args.rd < 31) {
        SetReg(args.rd, tmp);
      }
      return;
    }

    // (FJCVTZS — Armv8.3-JSCVT, double -> int32 ECMAScript ToInt32)
    //
    // FJCVTZS Wd, Dn: rmode=11, opcode=110, ftype=01, sf=0.
    //
    // ARM ARM C7.2.110 semantics: NaN/±Inf -> 0; otherwise apply ECMAScript
    // ToInt32 to trunc(d) -- i.e. take trunc(d) modulo 2^32, signed.  Sets
    // PSTATE.Z = 1 iff the input was a finite integer-valued double in
    // [INT32_MIN, INT32_MAX] (i.e., ToInt32 was exact); N = C = V = 0 always.
    //
    // Universal inline lowering (works for NaN, ±Inf, and every finite
    // input including the |d| > 2^31 ECMAScript modular-reduction case):
    //
    //   q   = trunc(d / 2^32)             ; ROUNDSD imm=0x03 (trunc + suppress)
    //   rem = d - q * 2^32                ; |rem| < 2^32 for finite d, NaN for ±Inf/NaN
    //   tmp = Cvttsd2siq(rem)             ; rem fits in int64 exactly; NaN -> INT64_MIN
    //   result_wd = (uint32_t)tmp         ; low 32 bits = ECMAScript ToInt32
    //
    // Why this is exact:
    //   - Finite d: ROUNDSD-trunc(d/2^32) computes the FP integer quotient
    //     in [-2^21, 2^21] (since |d|/2^32 <= 2^21 once |d| fits in FP64's
    //     2^1024 range and |trunc(d/2^32)| <= 2^21 when |d| < 2^53; for larger
    //     |d| both sides become multiples of higher powers of 2 and the
    //     subtraction is still exact at FP64 precision).  q * 2^32 is exact
    //     because 2^32 is a power of two and the product can't gain
    //     precision.  d - q*2^32 has magnitude < 2^32 so Cvttsd2siq is exact.
    //   - NaN: every arithmetic step propagates NaN; Cvttsd2siq(NaN) =
    //     INT64_MIN, whose low 32 bits are zero -- matching ARM's "NaN -> 0".
    //   - ±Inf: q = ±Inf, q * 2^32 = ±Inf, d - q*2^32 = Inf - Inf = NaN, then
    //     the NaN path above applies -- result = 0.
    //
    // Exactness for the Z flag:
    //   back = (int32_t)result, converted to FP64 (Cvtsi2sdl reads low 32 of tmp
    //   as int32 and produces the exact FP64).  Z = 1 iff Ucomisd(back, d) is
    //   ordered-equal (ZF=1 AND PF=0).  For NaN/Inf, ordered-equal is false;
    //   for in-range integers, back == d.  For out-of-range or non-integer
    //   inputs, back != d.
    if (rmode == 0b11 && opcode == 0b110 && args.ftype == 0b01 && !args.sf) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister q_xmm = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister scale_xmm = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister rem_xmm = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister back_xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp = AllocTempReg();
      if (tmp == no_register) { success_ = false; return; }
      Register flags_tmp = AllocTempReg();
      if (flags_tmp == no_register) { success_ = false; return; }

      int32_t src_off = VRegOffset(args.rn);

      // Load d into xmm.
      as_.Movsd(xmm, {.base = Assembler::rbp, .disp = src_off});

      // Materialize 2^32 as FP64 constant in scale_xmm via GP scratch.
      // FP64(2^32) bit pattern: exponent = 1023 + 32 = 1055 = 0x41F, mantissa = 0
      // => 0x41F0000000000000.
      as_.Movq(tmp, static_cast<int64_t>(0x41F0000000000000LL));
      as_.Movq(scale_xmm, tmp);

      // q_xmm = trunc(d / 2^32).
      as_.Movsd(q_xmm, xmm);
      as_.Divsd(q_xmm, scale_xmm);
      as_.Roundsd(q_xmm, q_xmm, int8_t{0x03});  // truncate-toward-zero + suppress

      // rem_xmm = d - q_xmm * 2^32.
      as_.Mulsd(q_xmm, scale_xmm);
      as_.Movsd(rem_xmm, xmm);
      as_.Subsd(rem_xmm, q_xmm);

      // tmp = (int64)trunc(rem); NaN -> INT64_MIN (0x8000000000000000), so
      // low 32 bits = 0 which is the ARM-mandated NaN/Inf result.
      as_.Cvttsd2siq(tmp, rem_xmm);

      // Exactness: back_xmm = (int32_t)tmp converted to FP64.  Cvtsi2sdl reads
      // the low 32 of tmp as int32 and sign-extends in the FP convert, so the
      // exactness comparison is against the same value that gets written to Wd.
      as_.Cvtsi2sdl(back_xmm, tmp);
      as_.Ucomisd(back_xmm, xmm);

      // flags = (ZF=1 AND PF=0) ? 0x4000 : 0 (only Z bit; N/C/V always 0).
      // The two MOVs and the two Jcc don't perturb the Ucomisd flags read.
      Assembler::Label* skip_exact = as_.MakeLabel();
      as_.Movl(flags_tmp, int32_t{0});           // default: not exact
      as_.Jcc(Assembler::Condition::kParityEven, *skip_exact);  // NaN
      as_.Jcc(Assembler::Condition::kNotEqual, *skip_exact);    // not equal
      as_.Movl(flags_tmp, int32_t{0x4000});      // exact: Z bit
      as_.Bind(skip_exact);
      int32_t flags_offset = offsetof(ThreadState, cpu.flags);
      as_.Movw({.base = Assembler::rbp, .disp = flags_offset}, flags_tmp);

      if (args.rd < 31) {
        // Zero-extend low 32 of tmp into the full 64-bit guest register
        // (Wd write semantics).  Movl on x86-64 auto-zeros the upper 32.
        as_.Movl(tmp, tmp);
        SetReg(args.rd, tmp);
      }
      return;
    }

    Undefined();
  }

