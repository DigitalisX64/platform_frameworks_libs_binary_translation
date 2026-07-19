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
// Family: SIMD scalar pairwise + shift-by-immediate.
// Methods:
//   AdvSimdScalarPairwise, AdvSimdShiftByImm

  // AdvSimd scalar pairwise JIT (ARM ARM "Advanced SIMD scalar pairwise").
  // Reads a pair of esize elements from Vn (lane[0] and lane[1] of the low
  // 64 bits) and combines them into a single scalar in Vd lane[0], with bits
  // above the lane zero-extended.
  //
  // Currently implemented:
  //   kAddp           — ADDP scalar D-form (Vd.D = Vn.D[0] + Vn.D[1]) via Paddq.
  //   kFaddpScalar    — FADDP scalar S/D via Addss / Addsd.
  //   kFmaxpScalar    — FMAXP scalar S/D via NaN-propagating max idiom
  //                     (Maxss/Maxsd × 2 + Por) + ±0 sign disambiguation.
  //   kFminpScalar    — FMINP scalar S/D via NaN-propagating min idiom.
  //   kFmaxnmpScalar  — FMAXNMP scalar S/D via NaN-suppressing substitution.
  //   kFminnmpScalar  — FMINNMP scalar S/D via NaN-suppressing substitution.
  //
  // FP16 path (Armv8.2-FP16 forms at U=0 + size[0]=0): F16C round-trip.
  // The two FP16 source lanes are lifted to FP32 via Pinsrw + Vcvtph2ps,
  // the existing FP32 reduction core runs on them unchanged, then the
  // FP32 result is narrowed back to FP16 via Vcvtps2ph.  Vcvtph2ps
  // preserves NaN-ness and ±0 sign, and the upper FP32 lanes stay 0
  // throughout the reduction (the inputs were 16-bit zero-extended and
  // x86 MAX/MIN/AND/OR on (0,0) returns 0), so Vcvtps2ph's narrowed
  // upper FP16 lanes are also 0; Movdqu then writes Vd[15:0]=result with
  // Vd[127:16]=0.  Falls back to interpreter when F16C is absent.
  //
  // FMAX/FMIN-vs-FMAXNM/FMINNM NaN policy (matches the vector lowering at
  // AdvSimdThreeSame{kFmaxV,kFminV,kFmaxnmV,kFminnmV}):
  //   - FMAX/FMIN: any NaN input -> NaN result (NaN propagation).
  //   - FMAXNM/FMINNM: single NaN -> the other operand; both NaN -> NaN.
  //
  // ±0 sign disambiguation (ARM ARM C7.2):
  //   - FMAX(+0,-0)=FMAX(-0,+0)=+0 regardless of operand order;
  //   - FMIN(+0,-0)=FMIN(-0,+0)=-0 regardless of operand order.
  // x86 MAXSS/MAXSD/MINSS/MINSD on (0,0) return the source operand bits,
  // which is order-sensitive, so we apply a corrective blend:
  // when both inputs are zero we replace the result with AND(a,b) for
  // FMAX-family (gives +0 if either is +0) or OR(a,b) for FMIN-family
  // (gives -0 if either is -0).  In the (-0,-0) case AND/OR both produce
  // -0, which matches ARM's "sign of -0 is preserved" for both families.
  void AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs& args) {
    using Op = Decoder::AdvSimdScalarPairwiseOpcode;
    if (args.is_fp16 && !host_platform::kHasF16C) { success_ = false; return; }
    const auto opc = args.opcode;
    const bool is_addp        = (opc == Op::kAddp);
    const bool is_faddp       = (opc == Op::kFaddpScalar);
    const bool is_fmax_family = (opc == Op::kFmaxpScalar ||
                                 opc == Op::kFmaxnmpScalar);
    const bool is_fmin_family = (opc == Op::kFminpScalar ||
                                 opc == Op::kFminnmpScalar);
    const bool is_minmax      = is_fmax_family || is_fmin_family;
    const bool is_nm          = (opc == Op::kFmaxnmpScalar ||
                                 opc == Op::kFminnmpScalar);
    if (!is_addp && !is_faddp && !is_minmax) {
      success_ = false; return;
    }

    int32_t src_n_off = VRegOffset(args.rn);
    int32_t dst_off   = VRegOffset(args.rd);

    // is_double also covers ADDP (D-form, 64-bit) for store-width selection.
    const bool is_double = is_addp || ((args.size & 1) != 0);

    SimdRegister xmm_a = AllocTempSimdReg();
    SimdRegister xmm_b = AllocTempSimdReg();
    if (!success()) { return; }

    // Load lane 0 (xmm_a) and lane 1 (xmm_b) of Vn.  Movsd/Movss from memory
    // zero-extend upper 96/64 bits — both source lanes are loaded before any
    // store to Vd so an in-place encoding (Vd == Vn) reads Vn's original pair.
    // FP16 path: each 16-bit source lane is loaded via Pxor + Pinsrw to
    // isolate it in xmm bits[15:0] with bits[127:16] zeroed, then
    // Vcvtph2ps widens 4 FP16 lanes to 4 FP32 lanes.  Pinsrw reads
    // Vn[15:0] and Vn[31:16] before any store to Vd, so an in-place
    // encoding (Vd == Vn) sees Vn's original pair.
    if (is_addp) {
      as_.Movq(xmm_a, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movq(xmm_b, {.base = Assembler::rbp, .disp = src_n_off + 8});
    } else if (is_double) {
      as_.Movsd(xmm_a, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movsd(xmm_b, {.base = Assembler::rbp, .disp = src_n_off + 8});
    } else if (args.is_fp16) {
      as_.Pxor(xmm_a, xmm_a);
      as_.Pinsrw(xmm_a, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_a, xmm_a);
      as_.Pxor(xmm_b, xmm_b);
      as_.Pinsrw(xmm_b, {.base = Assembler::rbp, .disp = src_n_off + 2}, int8_t{0});
      as_.Vcvtph2ps(xmm_b, xmm_b);
    } else {
      as_.Movss(xmm_a, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movss(xmm_b, {.base = Assembler::rbp, .disp = src_n_off + 4});
    }

    if (is_addp) {
      as_.Paddq(xmm_a, xmm_b);
    } else if (is_faddp) {
      if (is_double) as_.Addsd(xmm_a, xmm_b);
      else            as_.Addss(xmm_a, xmm_b);
    } else {
      // FMAXP / FMINP / FMAXNMP / FMINNMP scalar.
      //
      // Compute the ±0 corrective value and the "both zero" mask from the
      // original loads BEFORE the min/max op clobbers xmm_a.  After the
      // min/max we blend the corrective value over the result where the
      // mask says both inputs were zero.
      SimdRegister xmm_corr = AllocTempSimdReg();
      SimdRegister xmm_eq_a = AllocTempSimdReg();
      SimdRegister xmm_eq_b = AllocTempSimdReg();
      SimdRegister xmm_zero = AllocTempSimdReg();
      if (!success()) { return; }

      // Corrective value: AND(a,b) for FMAX family, OR(a,b) for FMIN family.
      as_.Movdqa(xmm_corr, xmm_a);
      if (is_fmax_family) as_.Pand(xmm_corr, xmm_b);
      else                as_.Por(xmm_corr, xmm_b);

      // "Both zero" mask in xmm_eq_a (low lane all-ones iff both ==0; IEEE
      // -0 compares equal to +0, so this catches both sign forms).
      as_.Pxor(xmm_zero, xmm_zero);
      as_.Movdqa(xmm_eq_a, xmm_a);
      if (is_double) as_.Cmpeqsd(xmm_eq_a, xmm_zero);
      else            as_.Cmpeqss(xmm_eq_a, xmm_zero);
      as_.Movdqa(xmm_eq_b, xmm_b);
      if (is_double) as_.Cmpeqsd(xmm_eq_b, xmm_zero);
      else            as_.Cmpeqss(xmm_eq_b, xmm_zero);
      as_.Pand(xmm_eq_a, xmm_eq_b);

      if (!is_nm) {
        // FMAX/FMIN: NaN-propagating idiom (mirrors the vector lowering at
        // AdvSimdThreeSame kFmaxV/kFminV).
        //   tmp = b; (Min|Max)ps tmp, a  -> tmp = a if any NaN, else min/max
        //   (Min|Max)ps a, b              -> a   = b if any NaN, else min/max
        //   POR a, tmp                    -> bitwise OR keeps all-1 exponent
        //                                    (NaN) if either was NaN; equals
        //                                    min/max otherwise.
        // We use the packed (PS/PD) variants because the assembler does not
        // expose scalar Maxss/Minss/Maxsd/Minsd; upper lanes hold zero (from
        // the Movsd/Movss zero-extending memory load) and stay zero through
        // the packed op since x86 MAX/MIN on (0,0) returns src2 = 0.
        SimdRegister xmm_tmp = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqa(xmm_tmp, xmm_b);
        if (is_fmax_family) {
          if (is_double) {
            as_.Maxpd(xmm_tmp, xmm_a);
            as_.Maxpd(xmm_a, xmm_b);
          } else {
            as_.Maxps(xmm_tmp, xmm_a);
            as_.Maxps(xmm_a, xmm_b);
          }
        } else {
          if (is_double) {
            as_.Minpd(xmm_tmp, xmm_a);
            as_.Minpd(xmm_a, xmm_b);
          } else {
            as_.Minps(xmm_tmp, xmm_a);
            as_.Minps(xmm_a, xmm_b);
          }
        }
        as_.Por(xmm_a, xmm_tmp);
      } else {
        // FMAXNM/FMINNM: substitute NaN lanes with the other operand, then
        // take min/max.  After substitution:
        //   - a NaN, b non-NaN -> a' = b, b' = b -> MAXSS/MINSS = b. ✓
        //   - b NaN, a non-NaN -> a' = a, b' = a -> MAXSS/MINSS = a. ✓
        //   - both NaN         -> a' = b (NaN), b' = a (NaN); the result
        //                          must be NaN per ARM ARM.  MAXSS/MINSS on
        //                          (NaN, NaN) returns the SRC2 operand bits,
        //                          which is still a NaN, satisfying ARM.
        //   - neither NaN      -> a' = a, b' = b -> normal MAXSS/MINSS.
        SimdRegister xmm_mask_a = AllocTempSimdReg();
        SimdRegister xmm_mask_b = AllocTempSimdReg();
        SimdRegister xmm_sub_a  = AllocTempSimdReg();
        SimdRegister xmm_sub_b  = AllocTempSimdReg();
        if (xmm_mask_a == no_simd_register || xmm_mask_b == no_simd_register ||
            xmm_sub_a == no_simd_register || xmm_sub_b == no_simd_register) {
          success_ = false; return;
        }
        as_.Movdqa(xmm_mask_a, xmm_a);
        if (is_double) as_.Cmpunordsd(xmm_mask_a, xmm_mask_a);
        else            as_.Cmpunordss(xmm_mask_a, xmm_mask_a);
        as_.Movdqa(xmm_mask_b, xmm_b);
        if (is_double) as_.Cmpunordsd(xmm_mask_b, xmm_mask_b);
        else            as_.Cmpunordss(xmm_mask_b, xmm_mask_b);
        as_.Movdqa(xmm_sub_a, xmm_mask_a);
        as_.Pand(xmm_sub_a, xmm_b);
        as_.Movdqa(xmm_sub_b, xmm_mask_b);
        as_.Pand(xmm_sub_b, xmm_a);
        as_.Pandn(xmm_mask_a, xmm_a);
        as_.Pandn(xmm_mask_b, xmm_b);
        as_.Por(xmm_mask_a, xmm_sub_a);
        as_.Por(xmm_mask_b, xmm_sub_b);
        if (is_fmax_family) {
          if (is_double) as_.Maxpd(xmm_mask_a, xmm_mask_b);
          else            as_.Maxps(xmm_mask_a, xmm_mask_b);
        } else {
          if (is_double) as_.Minpd(xmm_mask_a, xmm_mask_b);
          else            as_.Minps(xmm_mask_a, xmm_mask_b);
        }
        as_.Movdqa(xmm_a, xmm_mask_a);
      }

      // ±0 disambiguation: blend xmm_corr over xmm_a where both inputs were
      // zero.  Result: xmm_a := (mask & corr) | (~mask & xmm_a).
      as_.Pand(xmm_corr, xmm_eq_a);
      as_.Pandn(xmm_eq_a, xmm_a);
      as_.Por(xmm_corr, xmm_eq_a);
      as_.Movdqa(xmm_a, xmm_corr);
    }

    // Zero Vd above the result lane, then write the scalar at lane 0.
    // S-form writes 32 bits (lanes 1..3 stay zero); D-form and ADDP write
    // 64 bits (upper 64 bits stay zero).  FP16 path narrows xmm_a's FP32
    // lane 0 back to FP16 via Vcvtps2ph — the load lifted each FP16 lane
    // into an xmm with FP32 lanes 1..3 == 0, the reduction core preserves
    // those zeros (MAX/MIN/AND/OR on (0,0) returns 0; Addss/Cmpeqss only
    // touch lane 0), so the narrowed FP16 lanes 1..3 are also 0.  The
    // Vcvtps2ph instruction itself zeroes the upper 64 bits of the
    // destination xmm, so a full 128-bit Movdqu store delivers
    // Vd[15:0]=result and Vd[127:16]=0.
    if (args.is_fp16) {
      as_.Vcvtps2ph(xmm_a, xmm_a, int8_t{0});
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
    } else {
      as_.Pxor(xmm_b, xmm_b);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_b);
      if (is_double) {
        as_.Movsd({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
      } else {
        as_.Movss({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
      }
    }
  }

  void AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& args) {
    // Scalar shift-by-immediate dispatch.  The decoder forces D-form
    // (immh bit 3 set, esize=64) for the "simple" shift family — SSHR /
    // USHR / SSRA / USRA / SRSHR / URSHR / SRSRA / URSRA / SHL / SLI /
    // SRI — and sets args.q=false.  For those opcodes the existing
    // vector lowering below produces the correct scalar D-form result:
    //   - esize_bits is decoded as 64 (immh bit 3 set), so PSLLQ / PSRLQ
    //     are emitted for the low 64 bits of Vn.
    //   - args.q=false triggers the `Pslldq+Psrldq 8` upper-zero step
    //     at the store path, which delivers Vd[127:64]=0 — exactly
    //     ARM ARM scalar Vd[127:esize]=0.
    //   - SSHR / SSRA / SRSHR / SRSRA at .2D still bail inside their
    //     case arms (PSRAQ is AVX-512F-VL only); the interpreter picks
    //     them up.
    //
    // Saturating shifts (SQSHL / UQSHL / SQSHLU), narrow shifts
    // (SQSHRN / UQSHRN / SQRSHRN / UQRSHRN / SQSHRUN / SQRSHRUN), and
    // fixed-point conversions (SCVTF / UCVTF / FCVTZS / FCVTZU) accept
    // B/H/S scalar forms whose esize < 64; the vector path's upper-zero
    // only covers Vd[127:64], not Vd[esize:63], so it would leak wrong
    // values into Vd's low half.  Bail to interpreter for those.
    //
    // Exception: UQSHL and SQSHLU at scalar D (immh & 0b1000) light up
    // through the saturating-shift case below — the .D pipeline only
    // needs PSLLQ / PSRLQ / PCMPEQQ / PCMPGTQ (all SSE4.x), and the
    // upper-zero step at the store delivers Vd[127:64]=0 just like the
    // simple-shift family.  SQSHL .D still needs PSRAQ (AVX-512F-VL),
    // so it stays on the interpreter path for any size.  B/H/S
    // saturating scalar shifts also still bail — they need explicit
    // width-truncated stores that the .D path doesn't cover.
    if (args.scalar) {
      switch (args.opcode) {
        case Decoder::AdvSimdShiftImmOpcode::kShl:
        case Decoder::AdvSimdShiftImmOpcode::kSshr:
        case Decoder::AdvSimdShiftImmOpcode::kUshr:
        case Decoder::AdvSimdShiftImmOpcode::kSsra:
        case Decoder::AdvSimdShiftImmOpcode::kUsra:
        case Decoder::AdvSimdShiftImmOpcode::kSli:
        case Decoder::AdvSimdShiftImmOpcode::kSri:
        case Decoder::AdvSimdShiftImmOpcode::kSrshr:
        case Decoder::AdvSimdShiftImmOpcode::kUrshr:
        case Decoder::AdvSimdShiftImmOpcode::kSrsra:
        case Decoder::AdvSimdShiftImmOpcode::kUrsra:
          break;  // D-form-only ops fall through to vector path.
        case Decoder::AdvSimdShiftImmOpcode::kUqshl:
        case Decoder::AdvSimdShiftImmOpcode::kSqshlu:
          // Scalar D/S/H/B all fall through.  Scalar B (immh=0001)
          // takes a dedicated widen-then-clamp byte path inside the
          // saturating-shift case (no PSLLB in baseline SSE).
          break;
        case Decoder::AdvSimdShiftImmOpcode::kSqshl:
          // Scalar S/H/B fall through.  Scalar B uses the widen-via-
          // PMOVSXBW + PSLLW + signed clamp byte path.  Scalar D
          // (immh=1xxx) routes through a dedicated GPR-scalar
          // Shlq + Sarq + Cmpq overflow detector inside the case body
          // — the SSE vector path's PSRAQ signed-recover step is
          // AVX-512F-VL only, but the single-lane GPR fallback uses
          // baseline 64-bit GPR ops.  Vector .2D (args.scalar=false,
          // immh=1xxx) routes through the same GPR detector applied
          // twice (once per .2D lane) inside the case body.
          break;
        case Decoder::AdvSimdShiftImmOpcode::kScvtfFixed:
        case Decoder::AdvSimdShiftImmOpcode::kUcvtfFixed:
          // SCVTF / UCVTF scalar (signed/unsigned fixed-point integer →
          // FP).  The dedicated case body below handles its own bail
          // for UCVTF .D's high-bit-set int64 fixup; .S and SCVTF .D
          // fall through to the JIT.
          break;
        case Decoder::AdvSimdShiftImmOpcode::kFcvtzsFixed:
        case Decoder::AdvSimdShiftImmOpcode::kFcvtzuFixed:
          // FCVTZS / FCVTZU scalar (FP → signed/unsigned fixed-point
          // int).  Dedicated case body below handles .S and .D; .H
          // FP16 bails on the immh guard.
          break;
        case Decoder::AdvSimdShiftImmOpcode::kSqshrn:
        case Decoder::AdvSimdShiftImmOpcode::kUqshrn:
        case Decoder::AdvSimdShiftImmOpcode::kSqrshrn:
        case Decoder::AdvSimdShiftImmOpcode::kUqrshrn:
        case Decoder::AdvSimdShiftImmOpcode::kSqshrun:
        case Decoder::AdvSimdShiftImmOpcode::kSqrshrun:
          // Scalar narrow-shift saturating (ARM ARM C7.2 SQSHRN /
          // UQSHRN / SQRSHRN / UQRSHRN / SQSHRUN / SQRSHRUN scalar
          // B/H/S forms).  The shared vector lowering below performs
          // the lane-wide signed/unsigned arithmetic right shift,
          // saturation clamp, and PSHUFB narrow-pack on all lanes;
          // the scalar form differs only in needing the final write
          // to deliver `Vd[dst_bits-1:0] = narrow(Vn[src_bits-1:0])`
          // with `Vd[127:dst_bits] = 0`.  That width truncation is
          // applied at the store path by `Pslldq(16-dst_bytes) +
          // Psrldq(16-dst_bytes)`, mirroring the byte-granular
          // upper-zero pattern proven for SQSHL/UQSHL scalar B/H/S
          // in handoff-334.  Scalar D-source bails are inherited
          // from the vector-form guards (`uses_signed_shift &&
          // src_bits == 64` for SQ*/SQSU* scalar S, `is_rounding
          // && is_saturating_unsigned && src_bits == 64` for UQRSHRN
          // scalar S).
          break;
        default:
          success_ = false; return;
      }
    }
    // JIT for USHLL / SSHLL (unsigned/signed shift-left
    // long) at 8B→8H / 4H→4S / 2S→2D widening + Q=1 "long2" forms
    // (USHLL2/SSHLL2 reading the upper half of Vn).  Used by
    // calculate_gnu_hash_neon and many SIMD widening expansions.  Other
    // shift-imm opcodes fall back.
    int32_t vn_off = VRegOffset(args.rn);
    int32_t vd_off = VRegOffset(args.rd);

    switch (args.opcode) {
      case Decoder::AdvSimdShiftImmOpcode::kUshll:
      case Decoder::AdvSimdShiftImmOpcode::kSshll: {
        // USHLL/SSHLL Vd.<wide>, Vn.<narrow>, #shift.
        // ARM encoding: esize = 8 << highest-set-bit(immh).
        // immh=0001 → 8B→8H, shift = (immh:immb) - 8
        // immh=001x → 4H→4S, shift = (immh:immb) - 16
        // immh=01xx → 2S→2D, shift = (immh:immb) - 32
        // Q=0: read low 64 bits of Vn (offset 0).
        // Q=1: read upper 64 bits of Vn (offset 8) — "long2" form.
        // SSHLL sign-extends the narrow lanes via PMOVSX before the
        // shift; left-shift is bit-identical for signed/unsigned, so
        // PSLLW/PSLLD/PSLLQ is shared.
        uint8_t immh = args.immh;
        if (immh == 0) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSshll);
        const int32_t load_off = vn_off + (args.q ? 8 : 0);
        uint8_t shift_imm;
        // immh decode for SSHLL/USHLL (ARM ARM C7-1/C7-2):
        //   0001    → 8B (Q=0) / 16B (Q=1) source, dst=8H, shift=immh:immb-8
        //   001x    → 4H (Q=0) /  8H (Q=1) source, dst=4S, shift=immh:immb-16
        //   01xx    → 2S (Q=0) /  4S (Q=1) source, dst=2D, shift=immh:immb-32
        //   1xxx    → RESERVED for shift-left-long.
        // Test the high-set-bit first: priority is bit3 (reserved) → bit2
        // (2S→2D) → bit1 (4H→4S) → bit0 (8B→8H).
        if (immh & 0b1000) {
          // RESERVED — bail to undefined.
          Undefined(); return;
        } else if (immh & 0b0100) {
          // 2S → 2D
          shift_imm = ((immh << 3) | args.immb) - 32;
          if (is_signed) {
            as_.Pmovsxdq(xn, {.base = Assembler::rbp, .disp = load_off});
          } else {
            as_.Pmovzxdq(xn, {.base = Assembler::rbp, .disp = load_off});
          }
          if (shift_imm != 0) as_.Psllq(xn, shift_imm);
        } else if (immh & 0b0010) {
          // 4H → 4S
          shift_imm = ((immh << 3) | args.immb) - 16;
          if (is_signed) {
            as_.Pmovsxwd(xn, {.base = Assembler::rbp, .disp = load_off});
          } else {
            as_.Pmovzxwd(xn, {.base = Assembler::rbp, .disp = load_off});
          }
          if (shift_imm != 0) as_.Pslld(xn, shift_imm);
        } else if (immh & 0b0001) {
          // 8B → 8H
          shift_imm = ((immh << 3) | args.immb) - 8;
          if (is_signed) {
            as_.Pmovsxbw(xn, {.base = Assembler::rbp, .disp = load_off});
          } else {
            as_.Pmovzxbw(xn, {.base = Assembler::rbp, .disp = load_off});
          }
          if (shift_imm != 0) as_.Psllw(xn, shift_imm);
        } else {
          Undefined(); return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // SHL / USHR / SSHR vector shift-by-immediate JIT.
      //
      //   esize from immh:  bit3→8 (.2D), bit2→4 (.4S/.2S), bit1→2 (.8H/.4H),
      //                     bit0→1 (.16B/.8B — byte shift, no x86 equivalent,
      //                     bail).
      //   SHL  shift count = (immh:immb) - bits   ∈ [0, bits-1]
      //   USHR / SSHR rshift count = 2*bits - (immh:immb) ∈ [1, bits]
      //
      // x86 mapping:
      //   SHL  .8H/.4H → PSLLW   .4S/.2S → PSLLD   .2D → PSLLQ
      //   USHR .8H/.4H → PSRLW   .4S/.2S → PSRLD   .2D → PSRLQ
      //   SSHR .8H/.4H → PSRAW   .4S/.2S → PSRAD   .2D → bail (no PSRAQ pre AVX-512)
      //
      // Saturation semantics match ARM at the upper bound:
      //   PSLL{W,D,Q} with count ≥ bits → 0           (matches SHL spec
      //     boundary, but the encoding never gets there: SHL shift ≤ bits-1)
      //   PSRL{W,D,Q} with count ≥ bits → 0           (matches USHR at
      //     rshift==bits, which produces 0)
      //   PSRA{W,D}   with count ≥ bits → sign-fill   (matches SSHR at
      //     rshift==bits, which produces (signed)>>bits-1 i.e. sign-fill)
      //
      // !Q forms read the low 64 bits semantically; the source is read
      // as a 128-bit value but only the low half participates, and the
      // upper half of Vd is zeroed at the store.
      case Decoder::AdvSimdShiftImmOpcode::kShl:
      case Decoder::AdvSimdShiftImmOpcode::kUshr:
      case Decoder::AdvSimdShiftImmOpcode::kSshr: {
        const uint8_t immh = args.immh;
        if (immh == 0) { success_ = false; return; }
        const bool is_byte = (immh == 0b0001);  // esize = 1
        // SSHR .8B / .16B byte form JIT via
        // PMOVSXBW + PSRAW + PACKSSWB.
        //
        // SSHR's per-lane arithmetic right shift on bytes has no SSE
        // equivalent (no PSRAB). Widen each byte to a sign-extended
        // 16-bit word with PMOVSXBW, shift via PSRAW, then pack back
        // via PACKSSWB. The arithmetic-right-shift of an int8 cannot
        // grow magnitude, so every shifted word is in [-128, 127] —
        // PACKSSWB does not saturate.
        //
        // shift_count for byte form is 8 - immb ∈ [1, 8]. At cnt==8 on
        // a sign-extended-byte word, PSRAW by 8 shifts the sign byte
        // into bits[7:0] (byte-wide sign-fill), matching ARM SSHR's
        // sign-fill at shift==esize.
        //
        // .8B (!Q): widen low 8 bytes only, shift, pack with self, zero
        //   Vd[127:64] (D-register semantics).
        // .16B (Q): widen low and high halves separately into two SIMD
        //   temps, shift each, PACKSSWB(lo, hi) places low result in
        //   low 8 bytes and high result in high 8 bytes — matches ARM
        //   lane order.
        //
        // Scalar B form is not encoded by ARM (scalar SSHR only exists
        // at D). Defensive bail if args.scalar is set with is_byte.
        if (is_byte &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kSshr) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t byte_shift_count =
              static_cast<uint8_t>(8 - args.immb);
          SimdRegister xn_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xn_hi = no_simd_register;
          if (args.q) {
            xn_hi = AllocTempSimdReg();
            if (xn_hi == no_simd_register) {
              success_ = false; return;
            }
          }
          as_.Pmovsxbw(xn_lo,
                       {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Pmovsxbw(xn_hi,
                         {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          const int8_t cnt = static_cast<int8_t>(byte_shift_count);
          as_.Psraw(xn_lo, cnt);
          if (args.q) {
            as_.Psraw(xn_hi, cnt);
            as_.Packsswb(xn_lo, xn_hi);
          } else {
            as_.Packsswb(xn_lo, xn_lo);
            // Zero upper 64 bits of Vd (D-register semantics).
            as_.Pslldq(xn_lo, int8_t{8});
            as_.Psrldq(xn_lo, int8_t{8});
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn_lo);
          return;
        }
        // USHR .8B / .16B byte form JIT via
        // PMOVZXBW + PSRLW + PACKUSWB.
        //
        // Unsigned sibling of the SSHR .8B / .16B byte form above.
        // PMOVZXBW zero-extends each byte to a u16 word (high 8 bits =
        // 0); PSRLW logical-right-shifts (zero-fill); PACKUSWB
        // saturate-narrows words back to bytes.  The PSRLW result on a
        // zero-extended byte word always fits in u8 (logical right
        // shift cannot grow the value), so PACKUSWB does not actually
        // saturate.
        //
        // shift_count for byte form is 8 - immb ∈ [1, 8].  At cnt==8
        // on a zero-extended byte word, PSRLW by 8 shifts all 8 valid
        // bits off the right, yielding 0 — matches ARM USHR's zero
        // result at shift==esize.
        //
        // .8B (!Q): widen low 8 bytes only, shift, pack with self, zero
        //   Vd[127:64] (D-register semantics).
        // .16B (Q): widen low and high halves separately into two SIMD
        //   temps, shift each, PACKUSWB(lo, hi) places low result in
        //   low 8 bytes and high result in high 8 bytes — matches ARM
        //   lane order.
        //
        // Scalar B form is not encoded by ARM (scalar USHR only exists
        // at D — ARM ARM C7.2.371).  Defensive bail.
        if (is_byte &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kUshr) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t byte_shift_count =
              static_cast<uint8_t>(8 - args.immb);
          SimdRegister xn_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xn_hi = no_simd_register;
          if (args.q) {
            xn_hi = AllocTempSimdReg();
            if (xn_hi == no_simd_register) {
              success_ = false; return;
            }
          }
          as_.Pmovzxbw(xn_lo,
                       {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Pmovzxbw(xn_hi,
                         {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          const int8_t cnt = static_cast<int8_t>(byte_shift_count);
          as_.Psrlw(xn_lo, cnt);
          if (args.q) {
            as_.Psrlw(xn_hi, cnt);
            as_.Packuswb(xn_lo, xn_hi);
          } else {
            as_.Packuswb(xn_lo, xn_lo);
            // Zero upper 64 bits of Vd (D-register semantics).
            as_.Pslldq(xn_lo, int8_t{8});
            as_.Psrldq(xn_lo, int8_t{8});
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn_lo);
          return;
        }
        // SHL .8B / .16B byte form JIT via PSLLW + per-byte AND mask.
        //
        // x86 has no packed byte shift.  Shift the 16-bit words left by n
        // (PSLLW), which computes each byte's `<< n` but leaks the low
        // byte's high bits into the neighbouring high byte.  AND each byte
        // with (0xFF << n) & 0xFF strips those spilled bits (which are the
        // low n bits of each byte), yielding the per-byte `(Vn[i] << n) &
        // 0xFF` that ARM SHL specifies.  n = immh:immb - 8 = immb ∈ [0, 7].
        //
        // Scalar B SHL is not encoded (scalar SHL is D-form only — the
        // decoder forces immh bit 3), so is_byte && args.scalar can't
        // co-occur; bail defensively.
        if (is_byte && args.opcode == Decoder::AdvSimdShiftImmOpcode::kShl) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t n = args.immb;  // 0..7
          SimdRegister xn = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xmask = AllocTempSimdReg();
          if (!success()) { return; }
          Register tmp = AllocTempReg();
          if (tmp == no_register) { success_ = false; return; }
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          if (n != 0) {
            as_.Psllw(xn, static_cast<int8_t>(n));
          }
          const uint8_t mbyte = static_cast<uint8_t>((0xFFu << n) & 0xFFu);
          const uint64_t mword = 0x0101010101010101ULL * mbyte;
          as_.Movq(tmp, static_cast<int64_t>(mword));
          as_.Movq(xmask, tmp);               // low 64 = mword, high 64 = 0
          as_.Pinsrq(xmask, tmp, int8_t{1});  // high 64 = mword
          as_.Pand(xn, xmask);
          if (!args.q) {
            // Zero upper 64 bits of Vd (D-register semantics).
            as_.Pslldq(xn, int8_t{8});
            as_.Psrldq(xn, int8_t{8});
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        if (is_byte) { success_ = false; return; }
        const bool is_left = (args.opcode == Decoder::AdvSimdShiftImmOpcode::kShl);
        uint8_t esize_bits;
        uint8_t shift_count;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else /* immh & 0b0010 */ {
          esize_bits = 16;
        }
        const uint16_t immh_immb = static_cast<uint16_t>((immh << 3) | args.immb);
        if (is_left) {
          shift_count = static_cast<uint8_t>(immh_immb - esize_bits);  // 0..bits-1
        } else {
          shift_count = static_cast<uint8_t>(2 * esize_bits - immh_immb);  // 1..bits
        }
        // SSHR .2D / scalar D GPR fallback.
        //
        // SSHR's per-lane arithmetic right shift at .D needs PSRAQ,
        // which is AVX-512F-VL only.  For the .2D vector form and the
        // scalar D-source form (both single-or-double 64-bit lanes
        // operating on the low half of the SIMD register), the lanes
        // can be shipped through 64-bit GPR ops without PSRAQ:
        //
        //   Movq lane_i, Vn[64*i +: 64]        (signed int64)
        //   Sarq lane_i, cnt                   (cnt in [1, 64])
        //   Movq Vd[64*i +: 64], lane_i        (write back)
        //
        // Load BOTH lanes before storing — Vd may equal Vn.  shift_count
        // is in [1, 64]; x86 Sarq r64, imm masks the count to 6 bits,
        // so cnt==64 would hardware-decode as 0 (no shift) — wrong for
        // ARM's spec'd sign-fill at shift==esize.  Map cnt==64 → 63: a
        // Sarq r64, 63 replicates the sign bit across the whole lane,
        // which is the same sign-fill result the ARM spec demands for
        // SSHR at shift==64.
        //
        // Scalar D (args.scalar=true) writes a single lane and zeroes
        // Vd[127:64].  Vector .2D (args.scalar=false, args.q=true)
        // writes both lanes.  Vector .1D (args.scalar=false,
        // args.q=false) is reserved per ARM ARM C7.2.336 — bail so the
        // interpreter raises SIGILL via the decoder UNDEFINED path.
        if (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSshr &&
            esize_bits == 64) {
          const bool is_two_lane = !args.scalar && args.q;
          if (!args.scalar && !args.q) {
            // Vector .1D is reserved.
            success_ = false; return;
          }
          Register lane0 = AllocTempReg();
          Register lane1 =
              is_two_lane ? AllocTempReg() : Assembler::no_register;
          if (lane0 == Assembler::no_register ||
              (is_two_lane && lane1 == Assembler::no_register)) {
            success_ = false; return;
          }
          as_.Movq(lane0, {.base = Assembler::rbp, .disp = vn_off});
          if (is_two_lane) {
            as_.Movq(lane1,
                     {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          const int8_t sarq_cnt =
              (shift_count >= 64) ? int8_t{63}
                                  : static_cast<int8_t>(shift_count);
          as_.Sarq(lane0, sarq_cnt);
          if (is_two_lane) {
            as_.Sarq(lane1, sarq_cnt);
          }
          as_.Movq({.base = Assembler::rbp, .disp = vd_off}, lane0);
          if (is_two_lane) {
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                     lane1);
          } else {
            // Scalar D: zero Vd[127:64].
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                     int32_t{0});
          }
          return;
        }
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        const int8_t cnt = static_cast<int8_t>(shift_count);
        if (cnt != 0) {
          if (is_left) {
            switch (esize_bits) {
              case 16: as_.Psllw(xn, cnt); break;
              case 32: as_.Pslld(xn, cnt); break;
              case 64: as_.Psllq(xn, cnt); break;
            }
          } else if (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUshr) {
            switch (esize_bits) {
              case 16: as_.Psrlw(xn, cnt); break;
              case 32: as_.Psrld(xn, cnt); break;
              case 64: as_.Psrlq(xn, cnt); break;
            }
          } else /* kSshr */ {
            switch (esize_bits) {
              case 16: as_.Psraw(xn, cnt); break;
              case 32: as_.Psrad(xn, cnt); break;
            }
          }
        }
        if (!args.q) {
          // Zero upper 64 bits of Vd (D-register semantics).
          as_.Pslldq(xn, int8_t{8});
          as_.Psrldq(xn, int8_t{8});
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // SSRA / USRA / SLI / SRI vector shift-by-immediate JIT.
      //
      //   SSRA  Vd<i> = Vd<i> + (Vn<i> >> shift)        (arithmetic right shift, accumulate)
      //   USRA  Vd<i> = Vd<i> + (Vn<i> >> shift)        (logical right shift, accumulate)
      //   SLI   Vd<i> = (Vd<i> & low_mask)  | (Vn<i> << shift)   shift ∈ [0, esize-1]
      //   SRI   Vd<i> = (Vd<i> & high_mask) | (Vn<i> >> shift)   shift ∈ [1, esize]
      //
      //   low_mask  has the low  `shift`        bits per lane set (for SLI).
      //   high_mask has the high `shift`        bits per lane set (for SRI).
      //
      // For SLI/SRI the mask-and-OR is expressed via a two-shift bracket on Vd:
      //
      //   SLI:  PSLL(Vd, esize-shift); PSRL(Vd, esize-shift) — keeps low  `shift` bits.
      //   SRI:  PSRL(Vd, esize-shift); PSLL(Vd, esize-shift) — keeps high `shift` bits.
      //
      // x86 saturation matches ARM at the boundary in every case:
      //   - SLI shift==0: PSLL(Vd, esize) → 0 (count ≥ esize zeros the dest),
      //     then PSRL(0, esize) → 0, then POR(0, PSLL(Vn,0)=Vn) → Vn. Correct.
      //   - SRI shift==esize: PSRL(Vd, 0) → Vd, PSLL(Vd, 0) → Vd,
      //     POR(Vd, PSRL(Vn, esize)=0) → Vd. Correct (every bit preserved).
      //   - SSRA/USRA shift==esize: PSRA at count ≥ esize sign-fills (matches arch);
      //     PSRL at count ≥ esize zeros (matches arch).
      //
      // Bail cases:
      //   - byte (immh=0001): no PSLLB / PSRLB / PSRAB in SSE; interpreter handles.
      //   - SSRA at .2D (size=11): needs PSRAQ which is AVX-512F-VL only.
      case Decoder::AdvSimdShiftImmOpcode::kSsra:
      case Decoder::AdvSimdShiftImmOpcode::kUsra:
      case Decoder::AdvSimdShiftImmOpcode::kSli:
      case Decoder::AdvSimdShiftImmOpcode::kSri: {
        const uint8_t immh = args.immh;
        if (immh == 0) { success_ = false; return; }
        const bool is_byte = (immh == 0b0001);  // esize = 1 (8 bits)
        // SSRA .8B / .16B byte form JIT via
        // PMOVSXBW + PSRAW + PACKSSWB + PADDB.
        //
        // Sibling-promote of the SSHR .8B / .16B byte form above.
        // SSRA's accumulate step is `Vd += SSHR(Vn, cnt)` per byte.
        // Doing the accumulate at word width (PADDW before PACKSSWB)
        // would let PACKSSWB saturate the post-add words (e.g. 127 +
        // 63 = 190 saturates to 127); ARM SSRA wraps modulo 256, not
        // saturates. So pack first to int8 (arith-shifted bytes stay
        // in [-128, 127] — PACKSSWB does not saturate them), then
        // PADDB at byte width for native mod-256 wrap.
        //
        // .8B (!Q): widen low 8 bytes, shift, pack with self, zero
        //   upper half of the packed result, load Vd's low 8 bytes
        //   via Movq (zeros upper), PADDB, store. The Movq+upper-zero
        //   sequence guarantees Vd[127:64] is 0 after the store
        //   (D-register semantics).
        // .16B (Q): widen both halves of Vn, shift each, PACKSSWB(lo,
        //   hi) lays the result in ARM lane order, Movdqu full 16 of
        //   Vd, PADDB, store.
        //
        // Vd==Vn safety: PMOVSXBW reads from vn_off before the Movq /
        // Movdqu loads Vd, and the final Movdqu to vd_off happens
        // last — so writes to Vd don't disturb the earlier Vn reads.
        //
        // Scalar B form is not encoded by ARM (scalar SSRA only
        // exists at D — ARM ARM C7.2.339). Defensive bail.
        if (is_byte &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kSsra) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t byte_shift_count =
              static_cast<uint8_t>(8 - args.immb);
          SimdRegister xn_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xn_hi = no_simd_register;
          if (args.q) {
            xn_hi = AllocTempSimdReg();
            if (xn_hi == no_simd_register) {
              success_ = false; return;
            }
          }
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Pmovsxbw(xn_lo,
                       {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Pmovsxbw(xn_hi,
                         {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          const int8_t cnt = static_cast<int8_t>(byte_shift_count);
          as_.Psraw(xn_lo, cnt);
          if (args.q) {
            as_.Psraw(xn_hi, cnt);
            as_.Packsswb(xn_lo, xn_hi);
            as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
            as_.Paddb(xd, xn_lo);
          } else {
            as_.Packsswb(xn_lo, xn_lo);
            // PACKSSWB(xn, xn) duplicates the 8 result bytes into both
            // halves of xn_lo. Zero its upper half before PADDB so the
            // upper 8 result bytes of Vd stay 0 (D-reg semantics).
            as_.Pslldq(xn_lo, int8_t{8});
            as_.Psrldq(xn_lo, int8_t{8});
            // Movq m64→xmm loads only the low 8 bytes of Vd and zeros
            // xd[127:64].
            as_.Movq(xd, {.base = Assembler::rbp, .disp = vd_off});
            as_.Paddb(xd, xn_lo);
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
          return;
        }
        // USRA .8B / .16B byte form JIT via
        // PMOVZXBW + PSRLW + PACKUSWB + PADDB.
        //
        // Unsigned sibling of the SSRA .8B / .16B byte form above with
        // three instruction substitutions:
        //   PMOVSXBW → PMOVZXBW  (zero-extend instead of sign-extend)
        //   PSRAW    → PSRLW     (logical instead of arithmetic right shift)
        //   PACKSSWB → PACKUSWB  (unsigned-saturate-narrow)
        //
        // USRA's accumulate step is `Vd += USHR(Vn, cnt)` per byte.
        // PACKUSWB on the PSRLW'd zero-extended-byte words does not
        // actually saturate — logical right shift on a u16 word whose
        // high 8 bits are 0 cannot grow the value beyond 0xFF.
        // PADDB at byte width gives ARM's native mod-256 wrap on the
        // u8 accumulate (e.g. Vd=200, USHR=100 → 300 wraps to 44).
        //
        // .8B (!Q): widen low 8 bytes, shift, pack with self, zero
        //   upper half of the packed result, load Vd's low 8 bytes via
        //   Movq (zeros upper), PADDB, store. The Movq+upper-zero
        //   sequence guarantees Vd[127:64] is 0 after the store
        //   (D-register semantics).
        // .16B (Q): widen both halves of Vn, shift each, PACKUSWB(lo,
        //   hi) lays the result in ARM lane order, Movdqu full 16 of
        //   Vd, PADDB, store.
        //
        // Vd==Vn safety: PMOVZXBW reads from vn_off before the Movq /
        // Movdqu loads Vd, and the final Movdqu to vd_off happens
        // last — so writes to Vd don't disturb the earlier Vn reads.
        //
        // shift==esize (cnt==8) collapses to a per-lane no-op on Vd:
        // PSRLW by 8 on a zero-extended byte word produces 0, PACKUSWB
        // gives all-zero bytes, PADDB(Vd, 0) leaves Vd unchanged.
        // Matches ARM USRA at shift==esize where the USHR contribution
        // is zero.
        //
        // Scalar B form is not encoded by ARM (scalar USRA only exists
        // at D — ARM ARM C7.2.367). Defensive bail.
        if (is_byte &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kUsra) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t byte_shift_count =
              static_cast<uint8_t>(8 - args.immb);
          SimdRegister xn_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xn_hi = no_simd_register;
          if (args.q) {
            xn_hi = AllocTempSimdReg();
            if (xn_hi == no_simd_register) {
              success_ = false; return;
            }
          }
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Pmovzxbw(xn_lo,
                       {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Pmovzxbw(xn_hi,
                         {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          const int8_t cnt = static_cast<int8_t>(byte_shift_count);
          as_.Psrlw(xn_lo, cnt);
          if (args.q) {
            as_.Psrlw(xn_hi, cnt);
            as_.Packuswb(xn_lo, xn_hi);
            as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
            as_.Paddb(xd, xn_lo);
          } else {
            as_.Packuswb(xn_lo, xn_lo);
            // PACKUSWB(xn, xn) duplicates the 8 result bytes into both
            // halves of xn_lo. Zero its upper half before PADDB so the
            // upper 8 result bytes of Vd stay 0 (D-reg semantics).
            as_.Pslldq(xn_lo, int8_t{8});
            as_.Psrldq(xn_lo, int8_t{8});
            // Movq m64→xmm loads only the low 8 bytes of Vd and zeros
            // xd[127:64].
            as_.Movq(xd, {.base = Assembler::rbp, .disp = vd_off});
            as_.Paddb(xd, xn_lo);
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
          return;
        }
        // SLI .8B / .16B byte form JIT via PSLLW + per-byte insert mask.
        //
        // x86 has no packed byte shift.  Shift the 16-bit words left by n
        // (PSLLW), then keep only bits [n,7] of each byte via PAND with the
        // broadcast mask (0xFF << n) & 0xFF — this both drops the bits that
        // spilled across the byte boundary into the neighbouring high byte and
        // clears the shifted-in low n bits.  The destination supplies the low n
        // bits: PANDN(mask, Vd) = ~mask & Vd keeps Vd[n-1:0], and POR merges.
        // n = immh:immb - 8 = immb in [0, 7].  At n==0 the mask is 0xFF, so the
        // result is Vn and Vd contributes nothing — ARM SLI #0 = copy.
        //
        // Scalar B SLI is not ARM-encoded (scalar SLI is D-form only); bail.
        if (is_byte && args.opcode == Decoder::AdvSimdShiftImmOpcode::kSli) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t n = args.immb;  // 0..7
          SimdRegister xn = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xmask = AllocTempSimdReg();
          if (!success()) { return; }
          Register tmp = AllocTempReg();
          if (tmp == no_register) { success_ = false; return; }
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
          if (n != 0) {
            as_.Psllw(xn, static_cast<int8_t>(n));
          }
          const uint8_t mbyte = static_cast<uint8_t>((0xFFu << n) & 0xFFu);
          const uint64_t mword = 0x0101010101010101ULL * mbyte;
          as_.Movq(tmp, static_cast<int64_t>(mword));
          as_.Movq(xmask, tmp);               // low 64 = mword, high 64 = 0
          as_.Pinsrq(xmask, tmp, int8_t{1});  // high 64 = mword
          as_.Pand(xn, xmask);                // shifted bits [n,7] per byte
          as_.Pandn(xmask, xd);               // xmask = ~mask & Vd = Vd[n-1:0]
          as_.Por(xn, xmask);
          if (!args.q) {
            // Zero upper 64 bits of Vd (D-register semantics).
            as_.Pslldq(xn, int8_t{8});
            as_.Psrldq(xn, int8_t{8});
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        // SRI .8B / .16B byte form JIT via PSRLW + per-byte insert mask.
        //
        // Unsigned-shift sibling of the SLI byte form.  PSRLW logical-right-
        // shifts the 16-bit words by cnt, then PAND with the broadcast mask
        // (0xFF >> cnt) keeps only bits [0, 7-cnt] of each byte (dropping the
        // bits that spilled down from the neighbouring high byte).  The
        // destination supplies the top cnt bits: PANDN(mask, Vd) = ~mask & Vd
        // keeps Vd[7:8-cnt], POR merges.  cnt = 2*8 - immh:immb = 8 - immb in
        // [1, 8].  At cnt==8 the mask is 0 (shifted contributes nothing,
        // ~mask = all-ones keeps all of Vd) — ARM SRI at shift==esize leaves
        // Vd unchanged.
        //
        // Scalar B SRI is not ARM-encoded (scalar SRI is D-form only); bail.
        if (is_byte && args.opcode == Decoder::AdvSimdShiftImmOpcode::kSri) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t cnt_v = static_cast<uint8_t>(8 - args.immb);  // 1..8
          SimdRegister xn = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xmask = AllocTempSimdReg();
          if (!success()) { return; }
          Register tmp = AllocTempReg();
          if (tmp == no_register) { success_ = false; return; }
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
          as_.Psrlw(xn, static_cast<int8_t>(cnt_v));
          const uint8_t mbyte = static_cast<uint8_t>(0xFFu >> cnt_v);
          const uint64_t mword = 0x0101010101010101ULL * mbyte;
          as_.Movq(tmp, static_cast<int64_t>(mword));
          as_.Movq(xmask, tmp);
          as_.Pinsrq(xmask, tmp, int8_t{1});
          as_.Pand(xn, xmask);   // shifted bits [0,7-cnt] per byte
          as_.Pandn(xmask, xd);  // xmask = ~mask & Vd = Vd[7:8-cnt]
          as_.Por(xn, xmask);
          if (!args.q) {
            as_.Pslldq(xn, int8_t{8});
            as_.Psrldq(xn, int8_t{8});
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        if (is_byte) { success_ = false; return; }
        uint8_t esize_bits;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else /* immh & 0b0010 */ {
          esize_bits = 16;
        }
        const bool is_sli = (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSli);
        const uint16_t immh_immb = static_cast<uint16_t>((immh << 3) | args.immb);
        uint8_t shift_count;
        if (is_sli) {
          // SLI shift = immh:immb - bits, range [0, bits-1].
          shift_count = static_cast<uint8_t>(immh_immb - esize_bits);
        } else {
          // SSRA/USRA/SRI shift = 2*bits - immh:immb, range [1, bits].
          shift_count = static_cast<uint8_t>(2 * esize_bits - immh_immb);
        }
        // SSRA .2D / scalar D GPR fallback.
        //
        // SSRA's per-lane arithmetic right shift at .D needs PSRAQ,
        // which is AVX-512F-VL only.  Sibling of the SSHR .2D GPR
        // fallback above: load both Vn lanes into GPRs, Sarq each,
        // accumulate into the original Vd lane values, then write
        // back.  SSRA's accumulate step adds one Addq per lane on
        // top of the SSHR shape — Vd += SSHR(Vn, shift) per lane.
        //
        //   Movq vn0,  Vn[63:0]               (signed int64)
        //   Movq vn1,  Vn[127:64]             (only if two-lane)
        //   Movq vd0,  Vd[63:0]               (load orig Vd up front
        //   Movq vd1,  Vd[127:64]              so Vd==Vn stays safe
        //   Sarq vn0,  cnt                     across the writes)
        //   Sarq vn1,  cnt                    (only if two-lane)
        //   Addq vd0,  vn0                    (accumulate SSHR into Vd)
        //   Addq vd1,  vn1                    (only if two-lane)
        //   Movq Vd[63:0],   vd0
        //   Movq Vd[127:64], vd1              (two-lane)
        //   Movq Vd[127:64], imm32{0}         (scalar D: zero upper)
        //
        // cnt==64 maps to Sarq r64, 63 — same sign-fill the ARM
        // spec demands at shift==esize (see the SSHR fallback rationale
        // above).  Vector .1D (args.scalar=false, args.q=false) is
        // reserved per ARM ARM C7.2.337; bail to interpreter so the
        // decoder UNDEFINED path raises SIGILL.
        if (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSsra &&
            esize_bits == 64) {
          const bool is_two_lane = !args.scalar && args.q;
          if (!args.scalar && !args.q) {
            // Vector .1D is reserved.
            success_ = false; return;
          }
          Register vn0 = AllocTempReg();
          Register vn1 =
              is_two_lane ? AllocTempReg() : Assembler::no_register;
          Register vd0 = AllocTempReg();
          Register vd1 =
              is_two_lane ? AllocTempReg() : Assembler::no_register;
          if (vn0 == Assembler::no_register ||
              vd0 == Assembler::no_register ||
              (is_two_lane &&
               (vn1 == Assembler::no_register ||
                vd1 == Assembler::no_register))) {
            success_ = false; return;
          }
          as_.Movq(vn0, {.base = Assembler::rbp, .disp = vn_off});
          if (is_two_lane) {
            as_.Movq(vn1, {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          as_.Movq(vd0, {.base = Assembler::rbp, .disp = vd_off});
          if (is_two_lane) {
            as_.Movq(vd1, {.base = Assembler::rbp, .disp = vd_off + 8});
          }
          const int8_t sarq_cnt =
              (shift_count >= 64) ? int8_t{63}
                                  : static_cast<int8_t>(shift_count);
          as_.Sarq(vn0, sarq_cnt);
          if (is_two_lane) {
            as_.Sarq(vn1, sarq_cnt);
          }
          as_.Addq(vd0, vn0);
          if (is_two_lane) {
            as_.Addq(vd1, vn1);
          }
          as_.Movq({.base = Assembler::rbp, .disp = vd_off}, vd0);
          if (is_two_lane) {
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, vd1);
          } else {
            // Scalar D: zero Vd[127:64].
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                     int32_t{0});
          }
          return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
        const int8_t cnt = static_cast<int8_t>(shift_count);
        const int8_t inv = static_cast<int8_t>(esize_bits - shift_count);
        switch (args.opcode) {
          case Decoder::AdvSimdShiftImmOpcode::kSsra:
            // Vd += SSHR(Vn, shift). PSRA{W,D} only.
            switch (esize_bits) {
              case 16: as_.Psraw(xn, cnt); as_.Paddw(xd, xn); break;
              case 32: as_.Psrad(xn, cnt); as_.Paddd(xd, xn); break;
            }
            break;
          case Decoder::AdvSimdShiftImmOpcode::kUsra:
            // Vd += USHR(Vn, shift). PSRL{W,D,Q}.
            switch (esize_bits) {
              case 16: as_.Psrlw(xn, cnt); as_.Paddw(xd, xn); break;
              case 32: as_.Psrld(xn, cnt); as_.Paddd(xd, xn); break;
              case 64: as_.Psrlq(xn, cnt); as_.Paddq(xd, xn); break;
            }
            break;
          case Decoder::AdvSimdShiftImmOpcode::kSli:
            // Vd = (Vd & low_mask) | (Vn << shift).
            switch (esize_bits) {
              case 16:
                as_.Psllw(xd, inv); as_.Psrlw(xd, inv);
                as_.Psllw(xn, cnt);
                as_.Por(xd, xn);
                break;
              case 32:
                as_.Pslld(xd, inv); as_.Psrld(xd, inv);
                as_.Pslld(xn, cnt);
                as_.Por(xd, xn);
                break;
              case 64:
                as_.Psllq(xd, inv); as_.Psrlq(xd, inv);
                as_.Psllq(xn, cnt);
                as_.Por(xd, xn);
                break;
            }
            break;
          case Decoder::AdvSimdShiftImmOpcode::kSri:
            // Vd = (Vd & high_mask) | USHR(Vn, shift).
            switch (esize_bits) {
              case 16:
                as_.Psrlw(xd, inv); as_.Psllw(xd, inv);
                as_.Psrlw(xn, cnt);
                as_.Por(xd, xn);
                break;
              case 32:
                as_.Psrld(xd, inv); as_.Pslld(xd, inv);
                as_.Psrld(xn, cnt);
                as_.Por(xd, xn);
                break;
              case 64:
                as_.Psrlq(xd, inv); as_.Psllq(xd, inv);
                as_.Psrlq(xn, cnt);
                as_.Por(xd, xn);
                break;
            }
            break;
          default:
            // Unreachable — outer switch limits opcode to the four cases above.
            success_ = false; return;
        }
        if (!args.q) {
          // Zero upper 64 bits of Vd (D-register semantics).
          as_.Pslldq(xd, int8_t{8});
          as_.Psrldq(xd, int8_t{8});
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
        return;
      }
      // SRSHR / URSHR / SRSRA / URSRA vector
      // shift-by-immediate JIT (rounding right-shift family).
      //
      // Per-lane semantics:
      //   SRSHR  Vd<i> = floor((Vn<i> + 2^(shift-1)) / 2^shift)  signed
      //   URSHR  Vd<i> = floor((Vn<i> + 2^(shift-1)) / 2^shift)  unsigned
      //   SRSRA  Vd<i> = Vd<i> + SRSHR(Vn<i>, shift)
      //   URSRA  Vd<i> = Vd<i> + URSHR(Vn<i>, shift)
      //
      // shift ∈ [1, esize], esize ∈ {16, 32, 64} (byte form bails).
      //
      // The architectural addition `(x + roundbit)` is computed at infinite
      // precision; doing it in lane-narrow arithmetic would overflow near
      // INT_MAX (e.g. SRSHR INT16_MAX by 1).  To avoid this, we compute the
      // shifted value and the round bit separately, both in lane-narrow
      // arithmetic where neither overflows, and recombine with PADD:
      //
      //   shifted    = PSR{A,L}(Vn, shift)              arith for SRSHR/SRSRA,
      //                                                 logical for URSHR/URSRA
      //   round_bit  = PSRL(Vn, shift-1)                bit (shift-1) of Vn -> bit 0
      //   round_bit  = PSLL(round_bit, esize-1)         two-shift bracket: keep
      //   round_bit  = PSRL(round_bit, esize-1)         only bit 0 per lane
      //   result     = PADD(shifted, round_bit)         result == SRSHR/URSHR
      //
      // This identity is exact because in infinite precision:
      //   floor((x + 2^(n-1)) / 2^n) = floor(x / 2^n) + ((x >> (n-1)) & 1)
      // (the round bit decides whether to round up to the next quotient).
      //
      // x86 saturation matches ARM at every boundary:
      //   - URSHR / URSRA at shift==esize:
      //       PSRL(Vn, esize) → 0; round_bit = MSB(Vn).
      //       Architectural URSHR(x, esize) = MSB(x).  ✓
      //   - SRSHR / SRSRA at shift==esize:
      //       PSRA(Vn, esize) → sign-fill (0 if MSB=0, -1 if MSB=1);
      //       round_bit = MSB(Vn).
      //       Architectural SRSHR(x, esize): if MSB=0 → 0; if MSB=1 → -1+1 = 0.
      //       Always 0 — matches.  ✓
      //   - shift==1 (cnt_minus_1=0): PSRL(xr, 0) is a no-op; two-shift
      //       bracket then isolates bit 0 of Vn = the round bit at shift=1. ✓
      //
      // Bail cases:
      //   - byte (immh=0001): no PSR{A,L}B / PSLLB in SSE.
      //   - SRSHR / SRSRA at .2D (esize=64): needs PSRAQ, AVX-512F-VL only.
      //     URSHR / URSRA at .2D are fine — PSRLQ exists in SSE2.
      case Decoder::AdvSimdShiftImmOpcode::kSrshr:
      case Decoder::AdvSimdShiftImmOpcode::kUrshr:
      case Decoder::AdvSimdShiftImmOpcode::kSrsra:
      case Decoder::AdvSimdShiftImmOpcode::kUrsra: {
        const uint8_t immh = args.immh;
        if (immh == 0) { success_ = false; return; }
        const bool is_byte = (immh == 0b0001);  // esize = 8
        // URSHR .8B / .16B byte form JIT via
        // PMOVZXBW + PSRLW + two-shift bit-isolate + PADDW + PACKUSWB.
        //
        // Per-lane URSHR:  Vd<i> = floor((Vn<i> + 2^(cnt-1)) / 2^cnt)
        // with cnt ∈ [1, 8] for byte form.
        //
        // Computed at u16 word width on PMOVZXBW-widened bytes via the
        // identity URSHR(x, cnt) = (x >> cnt) + ((x >> (cnt-1)) & 1):
        //
        //   xn          = PMOVZXBW(Vn_half)       // u8 → u16 word (hi 8 bits = 0)
        //   round       = MOVDQA(xn)              // separate copy for round derivation
        //   round       = PSRLW(round, cnt-1)     // bit (cnt-1) of byte → bit 0 of word
        //   round       = PSLLW(round, 15)        // two-shift bracket: keep only bit 0,
        //   round       = PSRLW(round, 15)        //   yielding {0, 1} per word
        //   shifted     = PSRLW(xn, cnt)          // logical right shift, fits in u8
        //   shifted    += PADDW(shifted, round)   // shifted + round, fits in u8 too
        //   bytes       = PACKUSWB(shifted, ...)  // word → byte (no actual saturation)
        //   Vd          = bytes
        //
        // PACKUSWB never saturates: PSRLW on a zero-extended-byte word is
        // ≤ 0x7F for cnt ≥ 1; +round ≤ 1 makes the max 0x80, well below
        // 0xFF.
        //
        // shift==esize (cnt==8) boundary: PSRLW by 8 yields 0; round bit
        // = (byte >> 7) & 1 = MSB of original byte; result per lane is
        // MSB.  Matches ARM URSHR at shift==esize: (x + 2^(esize-1)) >>
        // esize = (x + 128) >> 8 = `x >= 128 ? 1 : 0` = MSB.  No
        // saturation issue (max result = 1).
        //
        // shift==1 boundary: cnt_minus_1 == 0, PSRLW(round, 0) is a
        // no-op, so the two-shift bracket isolates bit 0 of the original
        // word = bit 0 of the original byte = correct round bit at
        // shift=1.
        //
        // .8B (!Q): widen low 8 bytes only, compute one word lane, pack
        //   with self, zero upper duplicate via Pslldq+Psrldq (D-register
        //   semantics).
        // .16B (Q): widen both halves of Vn, compute both lanes in
        //   parallel, PACKUSWB(lo, hi) lays out ARM lane order (low half
        //   of result in low 8 bytes, high half in high 8 bytes).
        //
        // Scalar B form is not encoded by ARM (scalar URSHR only exists
        // at D — ARM ARM C7.2.358).  Defensive bail.
        if (is_byte &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kUrshr) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t byte_shift_count =
              static_cast<uint8_t>(8 - args.immb);
          SimdRegister xn_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister round_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xn_hi = no_simd_register;
          SimdRegister round_hi = no_simd_register;
          if (args.q) {
            xn_hi = AllocTempSimdReg();
            if (xn_hi == no_simd_register) { success_ = false; return; }
            round_hi = AllocTempSimdReg();
            if (round_hi == no_simd_register) {
              success_ = false; return;
            }
          }
          as_.Pmovzxbw(xn_lo,
                       {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Pmovzxbw(xn_hi,
                         {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          // Copy widened Vn into the round-bit derivation regs.
          as_.Movdqa(round_lo, xn_lo);
          if (args.q) {
            as_.Movdqa(round_hi, xn_hi);
          }
          const int8_t cnt = static_cast<int8_t>(byte_shift_count);
          const int8_t cnt_minus_1 =
              static_cast<int8_t>(byte_shift_count - 1);
          // bit (cnt-1) of each byte → bit 0 of each word.
          as_.Psrlw(round_lo, cnt_minus_1);
          if (args.q) {
            as_.Psrlw(round_hi, cnt_minus_1);
          }
          // Two-shift bracket: keep only bit 0 per word ({0, 1}).
          as_.Psllw(round_lo, int8_t{15});
          if (args.q) {
            as_.Psllw(round_hi, int8_t{15});
          }
          as_.Psrlw(round_lo, int8_t{15});
          if (args.q) {
            as_.Psrlw(round_hi, int8_t{15});
          }
          // Logical right shift the widened byte by cnt.
          as_.Psrlw(xn_lo, cnt);
          if (args.q) {
            as_.Psrlw(xn_hi, cnt);
          }
          // shifted + round (per-word; fits in u8).
          as_.Paddw(xn_lo, round_lo);
          if (args.q) {
            as_.Paddw(xn_hi, round_hi);
            as_.Packuswb(xn_lo, xn_hi);
          } else {
            as_.Packuswb(xn_lo, xn_lo);
            // Zero upper 64 bits (D-register semantics).
            as_.Pslldq(xn_lo, int8_t{8});
            as_.Psrldq(xn_lo, int8_t{8});
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn_lo);
          return;
        }
        // SRSHR .8B / .16B byte form JIT via
        // PMOVSXBW + PSRAW + round-bit isolate + PADDW + PACKSSWB.
        //
        // Signed sibling of the URSHR byte form above.  SRSHR(x,cnt) =
        // (x >>a cnt) + ((x >> (cnt-1)) & 1), computed at i16 word width on
        // PMOVSXBW-sign-extended bytes.  The round bit is derived with PSRLW
        // (bit (cnt-1) of the byte -> bit 0 of the word, then a PSLLW(15)/
        // PSRLW(15) bit-0 bracket), independent of the sign-extension bits.
        // The arith-shifted i16 stays in [-64, 64] (max |SRSHR(byte,cnt)| ~=
        // byte/2), so PACKSSWB never saturates.  cnt = 8 - immb in [1, 8]; at
        // cnt==8 PSRAW sign-fills and the round bit is the MSB, giving the
        // ARM-correct 0 per byte (e.g. -1 -> 0).
        //
        // Scalar B SRSHR is not ARM-encoded (scalar SRSHR is D-form only); bail.
        if (is_byte &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrshr) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t byte_shift_count =
              static_cast<uint8_t>(8 - args.immb);
          SimdRegister xn_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister round_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xn_hi = no_simd_register;
          SimdRegister round_hi = no_simd_register;
          if (args.q) {
            xn_hi = AllocTempSimdReg();
            if (xn_hi == no_simd_register) { success_ = false; return; }
            round_hi = AllocTempSimdReg();
            if (round_hi == no_simd_register) { success_ = false; return; }
          }
          as_.Pmovsxbw(xn_lo, {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Pmovsxbw(xn_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          as_.Movdqa(round_lo, xn_lo);
          if (args.q) { as_.Movdqa(round_hi, xn_hi); }
          const int8_t cnt = static_cast<int8_t>(byte_shift_count);
          const int8_t cnt_minus_1 = static_cast<int8_t>(byte_shift_count - 1);
          as_.Psrlw(round_lo, cnt_minus_1);
          if (args.q) { as_.Psrlw(round_hi, cnt_minus_1); }
          as_.Psllw(round_lo, int8_t{15});
          if (args.q) { as_.Psllw(round_hi, int8_t{15}); }
          as_.Psrlw(round_lo, int8_t{15});
          if (args.q) { as_.Psrlw(round_hi, int8_t{15}); }
          as_.Psraw(xn_lo, cnt);
          if (args.q) { as_.Psraw(xn_hi, cnt); }
          as_.Paddw(xn_lo, round_lo);
          if (args.q) {
            as_.Paddw(xn_hi, round_hi);
            as_.Packsswb(xn_lo, xn_hi);
          } else {
            as_.Packsswb(xn_lo, xn_lo);
            as_.Pslldq(xn_lo, int8_t{8});
            as_.Psrldq(xn_lo, int8_t{8});
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn_lo);
          return;
        }
        // SRSRA .8B / .16B byte form JIT: SRSHR byte pipeline (above) then
        // PADDB the rounded bytes into Vd (native mod-256 accumulate).
        // Scalar B SRSRA is not ARM-encoded; bail.
        if (is_byte &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t byte_shift_count =
              static_cast<uint8_t>(8 - args.immb);
          SimdRegister xn_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister round_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xn_hi = no_simd_register;
          SimdRegister round_hi = no_simd_register;
          if (args.q) {
            xn_hi = AllocTempSimdReg();
            if (xn_hi == no_simd_register) { success_ = false; return; }
            round_hi = AllocTempSimdReg();
            if (round_hi == no_simd_register) { success_ = false; return; }
          }
          as_.Pmovsxbw(xn_lo, {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Pmovsxbw(xn_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          as_.Movdqa(round_lo, xn_lo);
          if (args.q) { as_.Movdqa(round_hi, xn_hi); }
          const int8_t cnt = static_cast<int8_t>(byte_shift_count);
          const int8_t cnt_minus_1 = static_cast<int8_t>(byte_shift_count - 1);
          as_.Psrlw(round_lo, cnt_minus_1);
          if (args.q) { as_.Psrlw(round_hi, cnt_minus_1); }
          as_.Psllw(round_lo, int8_t{15});
          if (args.q) { as_.Psllw(round_hi, int8_t{15}); }
          as_.Psrlw(round_lo, int8_t{15});
          if (args.q) { as_.Psrlw(round_hi, int8_t{15}); }
          as_.Psraw(xn_lo, cnt);
          if (args.q) { as_.Psraw(xn_hi, cnt); }
          as_.Paddw(xn_lo, round_lo);
          if (args.q) {
            as_.Paddw(xn_hi, round_hi);
            as_.Packsswb(xn_lo, xn_hi);
            as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
            as_.Paddb(xd, xn_lo);
          } else {
            as_.Packsswb(xn_lo, xn_lo);
            as_.Pslldq(xn_lo, int8_t{8});
            as_.Psrldq(xn_lo, int8_t{8});
            as_.Movq(xd, {.base = Assembler::rbp, .disp = vd_off});
            as_.Paddb(xd, xn_lo);
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
          return;
        }
        // URSRA .8B / .16B byte form JIT: URSHR byte pipeline (see the URSHR
        // byte form above) then PADDB the rounded bytes into Vd (mod-256
        // accumulate).  Scalar B URSRA is not ARM-encoded; bail.
        if (is_byte &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kUrsra) {
          if (args.scalar) { success_ = false; return; }
          const uint8_t byte_shift_count =
              static_cast<uint8_t>(8 - args.immb);
          SimdRegister xn_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister round_lo = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          SimdRegister xn_hi = no_simd_register;
          SimdRegister round_hi = no_simd_register;
          if (args.q) {
            xn_hi = AllocTempSimdReg();
            if (xn_hi == no_simd_register) { success_ = false; return; }
            round_hi = AllocTempSimdReg();
            if (round_hi == no_simd_register) { success_ = false; return; }
          }
          as_.Pmovzxbw(xn_lo, {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Pmovzxbw(xn_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          as_.Movdqa(round_lo, xn_lo);
          if (args.q) { as_.Movdqa(round_hi, xn_hi); }
          const int8_t cnt = static_cast<int8_t>(byte_shift_count);
          const int8_t cnt_minus_1 = static_cast<int8_t>(byte_shift_count - 1);
          as_.Psrlw(round_lo, cnt_minus_1);
          if (args.q) { as_.Psrlw(round_hi, cnt_minus_1); }
          as_.Psllw(round_lo, int8_t{15});
          if (args.q) { as_.Psllw(round_hi, int8_t{15}); }
          as_.Psrlw(round_lo, int8_t{15});
          if (args.q) { as_.Psrlw(round_hi, int8_t{15}); }
          as_.Psrlw(xn_lo, cnt);
          if (args.q) { as_.Psrlw(xn_hi, cnt); }
          as_.Paddw(xn_lo, round_lo);
          if (args.q) {
            as_.Paddw(xn_hi, round_hi);
            as_.Packuswb(xn_lo, xn_hi);
            as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
            as_.Paddb(xd, xn_lo);
          } else {
            as_.Packuswb(xn_lo, xn_lo);
            as_.Pslldq(xn_lo, int8_t{8});
            as_.Psrldq(xn_lo, int8_t{8});
            as_.Movq(xd, {.base = Assembler::rbp, .disp = vd_off});
            as_.Paddb(xd, xn_lo);
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
          return;
        }
        if (is_byte) { success_ = false; return; }
        uint8_t esize_bits;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else /* immh & 0b0010 */ {
          esize_bits = 16;
        }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrshr ||
             args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra);
        const bool is_accumulate =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra ||
             args.opcode == Decoder::AdvSimdShiftImmOpcode::kUrsra);
        // Signed .2D bails — PSRAQ is AVX-512F-VL only.
        if (is_signed && esize_bits == 64) {
          // SRSHR .2D / scalar D GPR fallback.
          //
          // Per-lane semantics (signed rounding right shift):
          //   result = floor((Vn + 2^(shift-1)) / 2^shift)
          //
          // PSRAQ is AVX-512F-VL only.  Sibling of the SSRA .2D GPR
          // fallback above: load each Vn lane into a temp GPR, compute
          // the round bit via Shrq + Andq (bit (cnt-1) of Vn into bit
          // 0), Sarq the value by cnt, Addq the round bit, write back.
          //
          //   Movq    vn_i,    Vn[lane_i]
          //   Movq    round_i, vn_i
          //   Shrq    round_i, (cnt-1)        ; bit (cnt-1) into bit 0
          //   Andq    round_i, int32_t{1}     ; mask to bit 0
          //   Sarq    vn_i,    cnt            ; arith right shift
          //   Addq    vn_i,    round_i        ; shifted + round
          //   Movq    Vd[lane_i], vn_i
          //
          // For scalar D the upper 64 bits of Vd are zeroed after the
          // single-lane write.
          //
          // Boundary at cnt==esize (cnt==64): the identity
          //   floor((x + 2^(esize-1)) / 2^esize)
          // is always 0 for both MSB=0 (PSRA sign-fills to 0, round bit
          // is 0) and MSB=1 (PSRA sign-fills to -1, round bit is 1,
          // sum -1+1=0).  In the GPR path Sarq r64 with count==64
          // masks to 63 (wrong); special-case cnt==64 by writing 0
          // directly per lane.
          //
          // SRSRA .2D / scalar D (accumulate sibling: result += orig Vd)
          // is deferred to the next cycle — it needs the SSRA-style
          // orig-Vd preload + extra Addq per lane.  Vector .1D
          // (args.scalar=false, args.q=false) is reserved per ARM ARM
          // C7.2.342; bail to interpreter so the decoder UNDEFINED path
          // raises SIGILL.
          if (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrshr) {
            const bool is_two_lane = !args.scalar && args.q;
            if (!args.scalar && !args.q) {
              // Vector .1D is reserved.
              success_ = false; return;
            }
            const uint16_t immh_immb_local =
                static_cast<uint16_t>((immh << 3) | args.immb);
            const uint8_t shift_count_local =
                static_cast<uint8_t>(2 * 64 - immh_immb_local);  // [1, 64]
            if (shift_count_local == 64) {
              // SRSHR at shift==esize is always 0.
              as_.Movq({.base = Assembler::rbp, .disp = vd_off},
                       int32_t{0});
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                       int32_t{0});
              return;
            }
            Register vn0 = AllocTempReg();
            Register round0 = AllocTempReg();
            Register vn1 =
                is_two_lane ? AllocTempReg() : Assembler::no_register;
            Register round1 =
                is_two_lane ? AllocTempReg() : Assembler::no_register;
            if (vn0 == Assembler::no_register ||
                round0 == Assembler::no_register ||
                (is_two_lane &&
                 (vn1 == Assembler::no_register ||
                  round1 == Assembler::no_register))) {
              success_ = false; return;
            }
            const int8_t cnt = static_cast<int8_t>(shift_count_local);
            const int8_t cnt_minus_1 =
                static_cast<int8_t>(shift_count_local - 1);
            as_.Movq(vn0, {.base = Assembler::rbp, .disp = vn_off});
            if (is_two_lane) {
              as_.Movq(vn1, {.base = Assembler::rbp, .disp = vn_off + 8});
            }
            as_.Movq(round0, vn0);
            if (is_two_lane) {
              as_.Movq(round1, vn1);
            }
            as_.Shrq(round0, cnt_minus_1);
            if (is_two_lane) {
              as_.Shrq(round1, cnt_minus_1);
            }
            as_.Andq(round0, int32_t{1});
            if (is_two_lane) {
              as_.Andq(round1, int32_t{1});
            }
            as_.Sarq(vn0, cnt);
            if (is_two_lane) {
              as_.Sarq(vn1, cnt);
            }
            as_.Addq(vn0, round0);
            if (is_two_lane) {
              as_.Addq(vn1, round1);
            }
            as_.Movq({.base = Assembler::rbp, .disp = vd_off}, vn0);
            if (is_two_lane) {
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, vn1);
            } else {
              // Scalar D: zero Vd[127:64].
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                       int32_t{0});
            }
            return;
          }
          // SRSRA .2D / scalar D GPR fallback.
          //
          // Per-lane semantics (signed rounding right shift, accumulate):
          //   result = Vd<i> + floor((Vn<i> + 2^(shift-1)) / 2^shift)
          //
          // Combines the SRSHR per-lane shape (round bit via
          // Shrq + Andq, Sarq vn by cnt, Addq round) with the SSRA-style
          // orig-Vd preload + accumulate Addq.  Load BOTH Vn and orig Vd
          // up front so Vd==Vn aliasing remains safe across the writes.
          //
          //   Movq    vn_i,    Vn[lane_i]
          //   Movq    vd_i,    Vd[lane_i]    (load orig Vd before any write)
          //   Movq    round_i, vn_i
          //   Shrq    round_i, (cnt-1)       ; bit (cnt-1) into bit 0
          //   Andq    round_i, int32_t{1}    ; mask to bit 0
          //   Sarq    vn_i,    cnt           ; arith right shift
          //   Addq    vn_i,    round_i       ; SRSHR(Vn_i, cnt) in vn_i
          //   Addq    vd_i,    vn_i          ; Vd_i += SRSHR(Vn_i, cnt)
          //   Movq    Vd[lane_i], vd_i
          //
          // Boundary at cnt==esize (cnt==64): SRSHR(Vn, 64) is always 0
          // (see SRSHR rationale above), so SRSRA at cnt==64 is a no-op
          // per lane (Vd unchanged).  Vector .2D: nothing to write —
          // memory already holds the unmodified Vd.  Scalar D: need to
          // zero Vd[127:64] for D-register semantics (and Vd[63:0] keeps
          // its original value, which is already in memory).  In the GPR
          // path Sarq r64 with count==64 masks to 63 (wrong); special-
          // case cnt==64 directly.
          if (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra) {
            const bool is_two_lane = !args.scalar && args.q;
            if (!args.scalar && !args.q) {
              // Vector .1D is reserved.
              success_ = false; return;
            }
            const uint16_t immh_immb_local =
                static_cast<uint16_t>((immh << 3) | args.immb);
            const uint8_t shift_count_local =
                static_cast<uint8_t>(2 * 64 - immh_immb_local);  // [1, 64]
            if (shift_count_local == 64) {
              // SRSRA at shift==esize: no-op per lane (SRSHR yields 0).
              if (!is_two_lane) {
                // Scalar D: zero Vd[127:64]; Vd[63:0] preserved as-is.
                as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                         int32_t{0});
              }
              return;
            }
            Register vn0 = AllocTempReg();
            Register vd0 = AllocTempReg();
            Register round0 = AllocTempReg();
            Register vn1 =
                is_two_lane ? AllocTempReg() : Assembler::no_register;
            Register vd1 =
                is_two_lane ? AllocTempReg() : Assembler::no_register;
            Register round1 =
                is_two_lane ? AllocTempReg() : Assembler::no_register;
            if (vn0 == Assembler::no_register ||
                vd0 == Assembler::no_register ||
                round0 == Assembler::no_register ||
                (is_two_lane &&
                 (vn1 == Assembler::no_register ||
                  vd1 == Assembler::no_register ||
                  round1 == Assembler::no_register))) {
              success_ = false; return;
            }
            const int8_t cnt = static_cast<int8_t>(shift_count_local);
            const int8_t cnt_minus_1 =
                static_cast<int8_t>(shift_count_local - 1);
            // Load BOTH Vn and orig Vd before any writes (Vd==Vn safety).
            as_.Movq(vn0, {.base = Assembler::rbp, .disp = vn_off});
            if (is_two_lane) {
              as_.Movq(vn1, {.base = Assembler::rbp, .disp = vn_off + 8});
            }
            as_.Movq(vd0, {.base = Assembler::rbp, .disp = vd_off});
            if (is_two_lane) {
              as_.Movq(vd1, {.base = Assembler::rbp, .disp = vd_off + 8});
            }
            // Round bit per lane.
            as_.Movq(round0, vn0);
            if (is_two_lane) {
              as_.Movq(round1, vn1);
            }
            as_.Shrq(round0, cnt_minus_1);
            if (is_two_lane) {
              as_.Shrq(round1, cnt_minus_1);
            }
            as_.Andq(round0, int32_t{1});
            if (is_two_lane) {
              as_.Andq(round1, int32_t{1});
            }
            // SRSHR result in vn_i.
            as_.Sarq(vn0, cnt);
            if (is_two_lane) {
              as_.Sarq(vn1, cnt);
            }
            as_.Addq(vn0, round0);
            if (is_two_lane) {
              as_.Addq(vn1, round1);
            }
            // Accumulate: Vd_i += SRSHR(Vn_i, cnt).
            as_.Addq(vd0, vn0);
            if (is_two_lane) {
              as_.Addq(vd1, vn1);
            }
            as_.Movq({.base = Assembler::rbp, .disp = vd_off}, vd0);
            if (is_two_lane) {
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, vd1);
            } else {
              // Scalar D: zero Vd[127:64].
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                       int32_t{0});
            }
            return;
          }
          success_ = false; return;
        }
        const uint16_t immh_immb = static_cast<uint16_t>((immh << 3) | args.immb);
        const uint8_t shift_count =
            static_cast<uint8_t>(2 * esize_bits - immh_immb);  // [1, esize]
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xr = AllocTempSimdReg();
        SimdRegister xd = no_simd_register;
        if (is_accumulate) {
          xd = AllocTempSimdReg();
        }
        if (xn == no_simd_register || xr == no_simd_register ||
            (is_accumulate && xd == no_simd_register)) {
          success_ = false; return;
        }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Movdqu(xr, {.base = Assembler::rbp, .disp = vn_off});
        const int8_t cnt = static_cast<int8_t>(shift_count);
        const int8_t cnt_minus_1 = static_cast<int8_t>(shift_count - 1);
        const int8_t esize_minus_1 = static_cast<int8_t>(esize_bits - 1);
        // Main shifted value in xn.
        switch (esize_bits) {
          case 16:
            if (is_signed) as_.Psraw(xn, cnt); else as_.Psrlw(xn, cnt);
            break;
          case 32:
            if (is_signed) as_.Psrad(xn, cnt); else as_.Psrld(xn, cnt);
            break;
          case 64:
            // Signed bailed above for .2D.
            as_.Psrlq(xn, cnt);
            break;
        }
        // Round bit in xr: isolate bit (shift-1) of Vn into bit 0 per lane.
        switch (esize_bits) {
          case 16:
            as_.Psrlw(xr, cnt_minus_1);
            as_.Psllw(xr, esize_minus_1);
            as_.Psrlw(xr, esize_minus_1);
            break;
          case 32:
            as_.Psrld(xr, cnt_minus_1);
            as_.Pslld(xr, esize_minus_1);
            as_.Psrld(xr, esize_minus_1);
            break;
          case 64:
            as_.Psrlq(xr, cnt_minus_1);
            as_.Psllq(xr, esize_minus_1);
            as_.Psrlq(xr, esize_minus_1);
            break;
        }
        // Combine shifted + round_bit into xn.
        switch (esize_bits) {
          case 16: as_.Paddw(xn, xr); break;
          case 32: as_.Paddd(xn, xr); break;
          case 64: as_.Paddq(xn, xr); break;
        }
        if (is_accumulate) {
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
          switch (esize_bits) {
            case 16: as_.Paddw(xd, xn); break;
            case 32: as_.Paddd(xd, xn); break;
            case 64: as_.Paddq(xd, xn); break;
          }
        }
        SimdRegister result = is_accumulate ? xd : xn;
        if (!args.q) {
          // Zero upper 64 bits of Vd (D-register semantics).
          as_.Pslldq(result, int8_t{8});
          as_.Psrldq(result, int8_t{8});
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, result);
        return;
      }
      // SQSHL / UQSHL / SQSHLU vector shift-by-immediate
      // JIT (saturating left-shift family).
      //
      // Per-lane semantics:
      //   SQSHL  Vd<i> = signed_saturate(Vn<i> << shift)   signed in/out
      //   UQSHL  Vd<i> = unsigned_saturate(Vn<i> << shift) unsigned in/out
      //   SQSHLU Vd<i> = unsigned_saturate(max(Vn<i>,0) << shift)
      //                                                    signed in, unsigned out
      //
      // shift ∈ [0, esize-1], esize ∈ {16, 32}. Byte and .2D bail to interp.
      //
      // Saturation detection via shift-and-recover:
      //   shifted   = PSLL{W,D}(Vn, shift)
      //   recovered = PSRA{W,D} (signed) or PSRL{W,D} (unsigned) of shifted
      //   eq_mask   = PCMPEQ{W,D}(recovered, Vn): all-ones if no overflow
      //
      // Saturation value per lane:
      //   UQSHL  → UINT_MAX (all-ones)
      //   SQSHL  → INT_MAX if Vn ≥ 0 else INT_MIN.  Built as
      //                  INT_MAX XOR neg_mask, where neg_mask =
      //                  PCMPGT{W,D}(zero, Vn) and INT_MAX =
      //                  PSRL{W,D}(all_ones, 1).
      //   SQSHLU → UINT_MAX. Negative inputs are pre-zeroed (pos_xn =
      //                  ANDN(neg_mask, Vn)) so that UQSHL applied to
      //                  pos_xn yields the architectural 0-result for
      //                  negative inputs without any extra blend.
      //
      // Final blend: result = (eq_mask & shifted) | (~eq_mask & sat).
      //
      // x86 saturation matches ARM at the boundary in every case:
      //   - shift==0: PSLL(Vn,0)=Vn (Intel: count=0 no-op); PSR{A,L}(Vn,0)=Vn;
      //     eq_mask=all-ones; no saturation; result=Vn (SQSHL/UQSHL) or
      //     max(Vn,0) (SQSHLU). ✓
      //   - SQSHL  shift=esize-1, Vn=INT_MIN: PSLLW(0x8000,1)=0; PSRAW(0,1)=0;
      //     0 ≠ 0x8000 → overflow; neg_mask=all-ones; sat = INT_MAX XOR
      //     0xFFFF = INT_MIN. Matches arch (INT_MIN<<1 saturates to INT_MIN). ✓
      //   - UQSHL shift=1, Vn=0x8000: PSLLW(0x8000,1)=0; PSRLW(0,1)=0;
      //     0 ≠ 0x8000 → overflow; sat=UINT_MAX. Matches arch (0x8000<<1
      //     overflows uint16, saturates to 0xFFFF). ✓
      //
      // Bail cases:
      //   - byte (immh=0001): no PSLLB / PSRAB / PSRLB / PCMPGTB pre AVX-512BW.
      //     PCMPEQB and PCMPGTB do exist in SSE2, but the parallel shifts
      //     don't. Bails to interpreter.
      //   - .2D (immh & 0b1000):
      //       SQSHL needs PSRAQ which is AVX-512F-VL only.
      //       UQSHL / SQSHLU could be implemented (PSLLQ/PSRLQ/PCMPEQQ/
      //       PCMPGTQ all exist in SSE4.x) but we bail uniformly to keep
      //       this cycle focused. A follow-up cycle can JIT the .2D forms.
      case Decoder::AdvSimdShiftImmOpcode::kSqshl:
      case Decoder::AdvSimdShiftImmOpcode::kUqshl:
      case Decoder::AdvSimdShiftImmOpcode::kSqshlu: {
        const uint8_t immh = args.immh;
        if (immh == 0) { success_ = false; return; }
        const bool is_byte = (immh == 0b0001);
        if (is_byte) {
          // scalar byte (esize=8) saturating-shift path.
          //
          // No PSLLB/PSRAB/PSRLB in baseline SSE, so we widen Vn[7:0] to
          // a single 16-bit lane via PMOVSXBW (signed sources: SQSHL /
          // SQSHLU) or PMOVZXBW (unsigned source: UQSHL), apply PSLLW
          // (max shift is 7 and the byte source is at most |128|, so the
          // widened result is always in [-16384, +16256] — fits in i16
          // without further overflow), then clamp to the dst-byte range:
          //
          //   SQSHL  -> [INT8_MIN, INT8_MAX] = [0xFF80, 0x007F] (as i16)
          //                 Pmaxsw(INT8_MIN) + Pminsw(INT8_MAX)
          //   UQSHL  -> [0, UINT8_MAX] = [0x0000, 0x00FF]
          //                 Pminsw(UINT8_MAX)  (source already >= 0)
          //   SQSHLU -> [0, UINT8_MAX]
          //                 Pmaxsw(0) + Pminsw(UINT8_MAX)
          //
          // Per-lane mask constants are materialized via Pcmpeqw +
          // Psllw/Psrlw:
          //   0xFF80 = 0xFFFF << 7   (INT8_MIN signed-extended)
          //   0x007F = 0xFFFF >> 9   (INT8_MAX)
          //   0x00FF = 0xFFFF >> 8   (UINT8_MAX)
          //   0x0000 = Pxor self
          //
          // Vector .8B / .16B forms still bail — the vector lowering
          // needs eight or sixteen byte-saturated parallel shifts and
          // the widening trick consumes the whole xmm for one lane.
          if (!args.scalar) {
            success_ = false; return;
          }
          const uint8_t shift_count_b = static_cast<uint8_t>(
              ((static_cast<uint16_t>(immh) << 3) | args.immb) - 8);
          const int8_t cnt_b = static_cast<int8_t>(shift_count_b);
          const bool is_signed_src =
              (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshl) ||
              (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshlu);
          const bool is_signed_dst =
              (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshl);
          SimdRegister xn_b = AllocTempSimdReg();
          SimdRegister xclamp_b = AllocTempSimdReg();
          if (!success()) { return; }
          if (is_signed_src) {
            as_.Pmovsxbw(xn_b, {.base = Assembler::rbp, .disp = vn_off});
          } else {
            as_.Pmovzxbw(xn_b, {.base = Assembler::rbp, .disp = vn_off});
          }
          if (cnt_b != 0) {
            as_.Psllw(xn_b, cnt_b);
          }
          if (is_signed_dst) {
            // SQSHL byte: clamp to [INT8_MIN, INT8_MAX].
            as_.Pcmpeqw(xclamp_b, xclamp_b);
            as_.Psllw(xclamp_b, int8_t{7});
            as_.Pmaxsw(xn_b, xclamp_b);
            as_.Pcmpeqw(xclamp_b, xclamp_b);
            as_.Psrlw(xclamp_b, int8_t{9});
            as_.Pminsw(xn_b, xclamp_b);
          } else if (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshlu) {
            // SQSHLU byte: signed source, clamp negative to 0 and
            // overflow to UINT8_MAX.
            as_.Pxor(xclamp_b, xclamp_b);
            as_.Pmaxsw(xn_b, xclamp_b);
            as_.Pcmpeqw(xclamp_b, xclamp_b);
            as_.Psrlw(xclamp_b, int8_t{8});
            as_.Pminsw(xn_b, xclamp_b);
          } else {
            // UQSHL byte: zero-extended source >= 0, only upper clamp.
            as_.Pcmpeqw(xclamp_b, xclamp_b);
            as_.Psrlw(xclamp_b, int8_t{8});
            as_.Pminsw(xn_b, xclamp_b);
          }
          // Width-truncate: keep only byte 0 of lane 0; zero Vd[127:8].
          // Mirrors the byte-granular upper-zero pattern below for H/S.
          as_.Pslldq(xn_b, int8_t{15});
          as_.Psrldq(xn_b, int8_t{15});
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn_b);
          return;
        }
        const bool is_dword = (immh & 0b1000) != 0;
        // SQSHL scalar .D GPR fallback path.
        //
        // SQSHL (immediate) scalar .D (immh & 0b1000, args.scalar==true)
        // with shift_count in [0, 63].  The SSE vector .2D pipeline below
        // needs PSRAQ for the signed back-shift recovery step (AVX-512F-VL
        // only on x86), so emit a GPR-scalar fallback here using Shlq +
        // Sarq (both 64-bit GPR ops, baseline x86-64).  Vector .2D SQSHL
        // still bails to the interpreter via the per-vector filter below
        // — the GPR scalar path only handles a single 64-bit lane.
        //
        // Algorithm (mirrors the AdvSimdScalarThreeSame SQSHL D variable-
        // shift path at lite_translator.h around line 17744, specialized
        // for a compile-time shift):
        //
        //   a         = (int64_t)Vn[63:0]
        //   sat       = (a < 0) ? INT64_MIN : INT64_MAX
        //               (computed branchless as INT64_MAX ^ (a >>s 63))
        //   candidate = a << cnt
        //   back      = candidate >>s cnt
        //   result    = (back == a) ? candidate : sat
        //   Vd[63:0]  = result;  Vd[127:64] = 0.
        //
        // At cnt == 0 the shift is a no-op and no overflow is possible;
        // emit the pass-through directly to avoid the Cmpq + branch.
        if (is_dword && args.scalar &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshl) {
          const uint16_t immh_immb =
              static_cast<uint16_t>((immh << 3) | args.immb);
          const uint8_t shift_count = static_cast<uint8_t>(immh_immb - 64);
          const int8_t cnt = static_cast<int8_t>(shift_count);
          Register a = AllocTempReg();
          Register candidate = AllocTempReg();
          Register back = AllocTempReg();
          Register sat = AllocTempReg();
          if (a == Assembler::no_register ||
              candidate == Assembler::no_register ||
              back == Assembler::no_register ||
              sat == Assembler::no_register) {
            success_ = false; return;
          }
          as_.Movq(a, {.base = Assembler::rbp, .disp = vn_off});
          if (cnt == 0) {
            // No shift, no possible overflow; emit pass-through.
            as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                     int32_t{0});
            return;
          }
          // sat = sign(a) ? INT64_MIN : INT64_MAX:
          //   sat_tmp = a >>s 63   (all-zeros for a>=0, all-ones for a<0)
          //   sat     = INT64_MAX ^ sat_tmp
          //           → INT64_MAX (a>=0) or INT64_MIN (a<0).
          as_.Movq(sat, a);
          as_.Sarq(sat, int8_t{63});
          as_.Movq(back, int64_t{0x7FFFFFFFFFFFFFFFLL});
          as_.Xorq(sat, back);
          // candidate = a << cnt; back = candidate >>s cnt.
          as_.Movq(candidate, a);
          as_.Shlq(candidate, cnt);
          as_.Movq(back, candidate);
          as_.Sarq(back, cnt);
          // If back == a: no overflow → result is candidate.
          // Else: result stays at the precomputed saturation target.
          Assembler::Label* L_overflow = as_.MakeLabel();
          as_.Cmpq(back, a);
          as_.Jcc(Assembler::Condition::kNotEqual, *L_overflow);
          as_.Movq(sat, candidate);
          as_.Bind(L_overflow);
          as_.Movq({.base = Assembler::rbp, .disp = vd_off}, sat);
          as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                   int32_t{0});
          return;
        }
        // Vector .2D SQSHL (args.scalar=false, immh & 0b1000) cannot use
        // the SSE vector pipeline below — its signed back-shift recovery
        // step needs PSRAQ (AVX-512F-VL only on x86).  Emit a two-lane
        // GPR fallback that mirrors the scalar D detector twice, once per
        // .2D lane.
        //
        // Vd==Vn safety: read both Vn[63:0] and Vn[127:64] into GPRs up
        // front, before writing either Vd lane.
        //
        // The unallocated args.q=false / immh=1xxx encoding ("vector .1D"
        // — reserved per ARM ARM C7.2.243) is handled by mirroring the
        // existing vector pipeline's `if (!args.q)` upper-zero behavior:
        // produce lane 0 via the GPR detector and zero Vd[127:64].
        if (is_dword && !args.scalar &&
            args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshl) {
          const uint16_t immh_immb =
              static_cast<uint16_t>((immh << 3) | args.immb);
          const uint8_t shift_count = static_cast<uint8_t>(immh_immb - 64);
          const int8_t cnt = static_cast<int8_t>(shift_count);
          Register a_lo = AllocTempReg();
          Register a_hi = AllocTempReg();
          Register candidate = AllocTempReg();
          Register back = AllocTempReg();
          Register sat = AllocTempReg();
          if (a_lo == Assembler::no_register ||
              a_hi == Assembler::no_register ||
              candidate == Assembler::no_register ||
              back == Assembler::no_register ||
              sat == Assembler::no_register) {
            success_ = false; return;
          }
          // Read both Vn lanes before writing either Vd lane.
          as_.Movq(a_lo, {.base = Assembler::rbp, .disp = vn_off});
          if (args.q) {
            as_.Movq(a_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
          }
          if (cnt == 0) {
            // No shift, no possible overflow; emit pass-through.
            as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a_lo);
            if (args.q) {
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a_hi);
            } else {
              as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                       int32_t{0});
            }
            return;
          }
          // Lane 0: GPR-fallback signed-overflow detector.
          as_.Movq(sat, a_lo);
          as_.Sarq(sat, int8_t{63});
          as_.Movq(back, int64_t{0x7FFFFFFFFFFFFFFFLL});
          as_.Xorq(sat, back);
          as_.Movq(candidate, a_lo);
          as_.Shlq(candidate, cnt);
          as_.Movq(back, candidate);
          as_.Sarq(back, cnt);
          Assembler::Label* L_lo_overflow = as_.MakeLabel();
          as_.Cmpq(back, a_lo);
          as_.Jcc(Assembler::Condition::kNotEqual, *L_lo_overflow);
          as_.Movq(sat, candidate);
          as_.Bind(L_lo_overflow);
          as_.Movq({.base = Assembler::rbp, .disp = vd_off}, sat);
          if (!args.q) {
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8},
                     int32_t{0});
            return;
          }
          // Lane 1: reuse sat / candidate / back GPRs.
          as_.Movq(sat, a_hi);
          as_.Sarq(sat, int8_t{63});
          as_.Movq(back, int64_t{0x7FFFFFFFFFFFFFFFLL});
          as_.Xorq(sat, back);
          as_.Movq(candidate, a_hi);
          as_.Shlq(candidate, cnt);
          as_.Movq(back, candidate);
          as_.Sarq(back, cnt);
          Assembler::Label* L_hi_overflow = as_.MakeLabel();
          as_.Cmpq(back, a_hi);
          as_.Jcc(Assembler::Condition::kNotEqual, *L_hi_overflow);
          as_.Movq(sat, candidate);
          as_.Bind(L_hi_overflow);
          as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, sat);
          return;
        }
        uint8_t esize_bits;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else /* immh & 0b0010 */ {
          esize_bits = 16;
        }
        const uint16_t immh_immb = static_cast<uint16_t>((immh << 3) | args.immb);
        const uint8_t shift_count =
            static_cast<uint8_t>(immh_immb - esize_bits);  // [0, esize-1]
        const int8_t cnt = static_cast<int8_t>(shift_count);
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshl);
        const bool is_sqshlu =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshlu);
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xs = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xt = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});

        if (is_sqshlu) {
          // pos_xn = (Vn < 0) ? 0 : Vn.  Pre-zero negative lanes so the
          // UQSHL pipeline below yields the architectural 0 result.
          as_.Pxor(xt, xt);
          switch (esize_bits) {
            case 16: as_.Pcmpgtw(xt, xn); break;
            case 32: as_.Pcmpgtd(xt, xn); break;
            case 64: as_.Pcmpgtq(xt, xn); break;  // SSE4.2
          }
          // xt = neg_mask (0xFFFF / 0xFFFFFFFF / 0xFFFFFFFFFFFFFFFF
          // where xn < 0).
          as_.Pandn(xt, xn);                   // xt = ~neg_mask & xn = pos_xn
          as_.Movdqa(xn, xt);                  // xn := pos_xn for the rest
        }

        // xs = xn << shift_count.
        as_.Movdqa(xs, xn);
        switch (esize_bits) {
          case 16: as_.Psllw(xs, cnt); break;
          case 32: as_.Pslld(xs, cnt); break;
          case 64: as_.Psllq(xs, cnt); break;
        }
        // xm = recover(xs, shift_count) using arith shift (signed) or
        // logical shift (unsigned).
        as_.Movdqa(xm, xs);
        if (is_signed) {
          switch (esize_bits) {
            case 16: as_.Psraw(xm, cnt); break;
            case 32: as_.Psrad(xm, cnt); break;
            // esize 64 path filtered above (SQSHL .D bails — no PSRAQ).
          }
        } else {
          switch (esize_bits) {
            case 16: as_.Psrlw(xm, cnt); break;
            case 32: as_.Psrld(xm, cnt); break;
            case 64: as_.Psrlq(xm, cnt); break;
          }
        }
        // xm = eq_mask: per-lane all-ones if no overflow, else 0.
        switch (esize_bits) {
          case 16: as_.Pcmpeqw(xm, xn); break;
          case 32: as_.Pcmpeqd(xm, xn); break;
          case 64: as_.Pcmpeqq(xm, xn); break;  // SSE4.1
        }

        // Build saturation value in xt.
        if (is_signed) {
          // SQSHL: sat = INT_MAX XOR neg_mask(xn).
          as_.Pxor(xt, xt);
          switch (esize_bits) {
            case 16: as_.Pcmpgtw(xt, xn); break;
            case 32: as_.Pcmpgtd(xt, xn); break;
          }
          // xn is now free; reuse to hold INT_MAX = PSRL(all_ones, 1).
          as_.Pcmpeqd(xn, xn);                 // all-ones
          switch (esize_bits) {
            case 16: as_.Psrlw(xn, int8_t{1}); break;
            case 32: as_.Psrld(xn, int8_t{1}); break;
          }
          as_.Pxor(xt, xn);                    // xt = INT_MAX XOR neg_mask
        } else {
          // UQSHL / SQSHLU: sat = UINT_MAX (all-ones) per lane.
          as_.Pcmpeqd(xt, xt);
        }

        // Blend: result = (eq_mask & shifted) | (~eq_mask & sat) in xs.
        as_.Pand(xs, xm);                      // xs = eq_mask & shifted
        as_.Pandn(xm, xt);                     // xm = ~eq_mask & sat
        as_.Por(xs, xm);                       // xs = blended result

        if (!args.q) {
          // scalar B/H/S keep only `esize_bits/8` low bytes;
          // vector .8B/.4H/.2S and scalar .D keep low 8 bytes.  Byte-granular
          // Pslldq/Psrldq trick zeros everything above the kept bytes.
          int8_t shift_bytes =
              (args.scalar && esize_bits < 64)
                  ? static_cast<int8_t>(16 - (esize_bits / 8))
                  : int8_t{8};
          as_.Pslldq(xs, shift_bytes);
          as_.Psrldq(xs, shift_bytes);
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xs);
        return;
      }
      // SHRN / SHRN2 / RSHRN / RSHRN2 / UQSHRN / UQSHRN2
      // vector shift-right-narrow JIT.
      //
      //   immh=0001 → src 16-bit (.8H), dst 8-bit  (.8B / .16B for Q=1)
      //   immh=001x → src 32-bit (.4S), dst 16-bit (.4H / .8H  for Q=1)
      //   immh=01xx → src 64-bit (.2D), dst 32-bit (.2S / .4S  for Q=1)
      //   immh=1xxx → undefined (src would be 128-bit) — bail to interp.
      //
      //   narrow_rshift = src_bits - (immh:immb)  ∈ [1, src_bits/2]
      //
      // x86 lowering: PSRL{W,D,Q} src by narrow_rshift, then PSHUFB to
      // gather the low half of each src lane into the low 64 bits of the
      // dst (upper 64 bits zeroed by 0x80 selectors).  PSRL count ≥ esize
      // produces 0 per lane — matches the ARM `narrow_rshift >= src_bits`
      // branch in the interpreter.
      //
      // RSHRN adds the rounding constant (1 << (narrow_rshift - 1)) into
      // every source lane via PADDW/PADDD/PADDQ before the right shift.
      // The integer add wraps modulo lane width on x86 but the wrap is
      // benign: the subsequent right shift drops at least narrow_rshift
      // bits and PSHUFB then narrow-masks to dst_bits, so the high carry
      // bit that the interpreter's 128-bit wide-add captures is always
      // discarded.  narrow_rshift ∈ [1, src_bits/2] by the encoding
      // constraints, so (narrow_rshift - 1) ≥ 0 and the rounding
      // constant always fits in a single source lane.
      //
      // UQSHRN unsigned-saturates the post-shift value to dst_bits before
      // narrowing.  After PSRL{W,D,Q}, each src lane is in [0, 2^(src_bits
      // - narrow_rshift) - 1] (top bit is zero because the shift count is
      // ≥ 1, so the value is non-negative when interpreted as signed).
      // Clamp via:
      //   src 16-bit: PMINUW vs 0x00FF... pattern (SSE4.1)
      //   src 32-bit: PMINUD vs 0x0000FFFF... pattern (SSE4.1)
      //   src 64-bit: manual — no PMINUQ in baseline SSE.  Extract hi32
      //               of each qword via PSRLQ-by-32; compare to zero with
      //               PCMPEQD; broadcast the per-qword "hi==0" mask via
      //               PSHUFD; XOR with the xsat pattern (which has
      //               0xFFFFFFFF in the lo32-of-each-qword lanes) to
      //               invert into "saturate" polarity; POR into xn so
      //               saturating lanes' lo32 become 0xFFFFFFFF.
      // The subsequent PSHUFB gathers the saturated lo half of each lane.
      //
      // UQRSHRN combines RSHRN's rounding pre-shift add with UQSHRN's
      // post-shift unsigned saturating clamp.  Because the subsequent
      // saturate observes the WIDE (pre-shift) value, the rounding add
      // must NOT lose its carry-out — the plain PADDW/PADDD that RSHRN
      // uses wraps mod 2^src_bits and silently zeros the saturated
      // result.  Saturating-add variants:
      //   src 16-bit: PADDUSW (single instruction, saturates to 0xFFFF).
      //   src 32-bit: no PADDUSD in baseline SSE; pre-clamp via
      //               PMINUD(xn, 0xFFFFFFFF - round_lane) before PADDD.
      //               Clamped lanes land at 0xFFFFFFFF after the add,
      //               which PSRLD-then-PMINUD correctly drives to the
      //               saturated 0xFFFF (because 0xFFFFFFFF >> narrow_rshift
      //               ≥ 0xFFFF for narrow_rshift ∈ [1, 16]).
      //   src 64-bit: PADDQ wraps, and emulating PADDUSQ on baseline SSE
      //               needs PCMPGTQ + blend + a separate temp.  Bail to
      //               interpreter (`success_ = false`).
      //
      // SQSHRN signed-saturates the post-shift value to dst_bits before
      // narrowing.  The pre-shift is PSRA{W,D} (arithmetic right shift,
      // sign-preserving).  After shift, clamp each src-element-wide lane
      // to the signed dst-width range:
      //   dst 8:   [-128, 127]   i.e. [0xFF80, 0x007F] as signed 16-bit
      //            → PMAXSW+PMINSW (both SSE2).
      //   dst 16:  [-32768, 32767] i.e. [0xFFFF8000, 0x00007FFF] as
      //            signed 32-bit → PMAXSD+PMINSD (both SSE4.1).
      //   dst 32:  src=64, no PSRAQ in baseline SSE — bail.
      // PSHUFB then gathers the low byte (src 16-bit) or low halfword
      // (src 32-bit) of each lane.  After signed clamp those bottom
      // bytes/halfwords ARE the correct signed narrow result.
      //
      // SQRSHRN adds the rounding pre-shift add.  The pre-shift saturate
      // observes the WIDE value, so the rounding add must signed-
      // saturate too (the round constant is always positive
      // `1 << (rshift - 1)`):
      //   src 16-bit: PADDSW (signed saturating word add, SSE2).
      //   src 32-bit: no PADDSD-integer in baseline SSE; pre-clamp via
      //               PMINSD(xn, INT32_MAX - round_lane).  Lanes clamped
      //               there yield exactly INT32_MAX after the plain PADDD,
      //               which the post-shift PMINSD-vs-signed-max saturates
      //               correctly.  Adding a positive round constant cannot
      //               under-flow a signed value, so no lower pre-clamp.
      //   src 64-bit: bail (no PSRAQ for the shift anyway).
      //
      // SQSHRUN reads the source as signed, clamps the post-shift value
      // to the UNSIGNED dst range `[0, 2^dst_bits - 1]`, then narrows.
      // Negative results saturate to 0, results above the dst-max
      // saturate to dst-max.  Pre-shift is PSRA{W,D} (sign-preserving,
      // same as SQSHRN — the source is signed).  Post-shift clamp:
      //   src 16-bit (dst 8):  PMAXSW vs zero, then PMINSW vs 0x00FF
      //                        (after PMAXSW the lane is non-negative,
      //                        so signed-min against the positive 0xFF
      //                        cap is equivalent to unsigned-min).
      //   src 32-bit (dst 16): PMAXSD vs zero, then PMINSD vs 0xFFFF
      //                        (both SSE4.1; same non-negative argument).
      //   src 64-bit: bail (no PSRAQ).
      //
      // SQRSHRUN adds the rounding pre-shift add.  Like SQRSHRN the wide
      // add must signed-saturate (round constant is positive; saturating
      // at INT*_MAX leaves the post-clamp at dst-max which is what the
      // architectural infinite-precision result would deliver too):
      //   src 16-bit: PADDSW.
      //   src 32-bit: PMINSD-preclamp + plain PADDD (same trick as
      //               SQRSHRN — the round constant is positive).
      //   src 64-bit: bail.
      //
      // Q=0:        store narrowed-in-low | zero-upper.
      // Q=1 ("2"):  preserve Vd[63:0], OR narrowed result into Vd[127:64].
      case Decoder::AdvSimdShiftImmOpcode::kShrn:
      case Decoder::AdvSimdShiftImmOpcode::kRshrn:
      case Decoder::AdvSimdShiftImmOpcode::kUqshrn:
      case Decoder::AdvSimdShiftImmOpcode::kUqrshrn:
      case Decoder::AdvSimdShiftImmOpcode::kSqshrn:
      case Decoder::AdvSimdShiftImmOpcode::kSqrshrn:
      case Decoder::AdvSimdShiftImmOpcode::kSqshrun:
      case Decoder::AdvSimdShiftImmOpcode::kSqrshrun: {
        const uint8_t immh = args.immh;
        if (immh == 0 || (immh & 0b1000)) { success_ = false; return; }
        uint8_t src_bits;
        if (immh & 0b0100) {
          src_bits = 64;
        } else if (immh & 0b0010) {
          src_bits = 32;
        } else /* immh == 0b0001 */ {
          src_bits = 16;
        }
        const bool is_rounding =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kRshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqrshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrun);
        const bool is_saturating_unsigned =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqrshrn);
        const bool is_saturating_signed =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrn);
        const bool is_saturating_signed_to_unsigned =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshrun) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrun);
        // PSRAW/PSRAD (sign-preserving arithmetic right shift) is required
        // whenever the source is interpreted as signed.  PADDSW / PMINSD-
        // preclamp+PADDD (signed-saturating add) is required whenever the
        // wide rounding add must preserve signed saturation.
        const bool uses_signed_shift =
            is_saturating_signed || is_saturating_signed_to_unsigned;
        if (is_rounding && is_saturating_unsigned && src_bits == 64) {
          // UQRSHRN src=64 needs a saturating PADDQ that baseline SSE
          // can't express cheaply.  Fall back to the interpreter.
          success_ = false; return;
        }
        // SQSHRN / SQSHRUN / SQRSHRN scalar D-source
        // GPR fallback.
        //
        // The SIMD vector lowering needs PSRAQ for the per-lane signed
        // arithmetic right shift at src=64, which is AVX-512F-VL only.
        // For the scalar form (one lane, ARM ARM C7.2.226 / C7.2.236 /
        // C7.2.238 / C7.2.228 forms `SQSHRN Vd<s>, Vn<d>, #shift`,
        // `SQSHRUN Vd<s>, Vn<d>, #shift`, `SQRSHRN Vd<s>, Vn<d>, #shift`),
        // we can sidestep PSRAQ entirely by routing the single lane through
        // 64-bit GPR ops:
        //
        //   1. Movq a, Vn[63:0]                  (signed int64)
        //   2. (rounding only) signed-saturating add of (1<<(cnt-1)).
        //      Round const is always positive, so only positive
        //      overflow is possible — clamp to INT64_MAX on OF.
        //   3. Sarq a, cnt                       (cnt in [1, 32])
        //   4. clamp a to dst range via Cmpq + Cmovq
        //        SQSHRN / SQRSHRN : [INT32_MIN,  INT32_MAX]
        //        SQSHRUN          : [0,          UINT32_MAX]
        //   5. Pxor xzero / Movdqu Vd, xzero     (zero Vd[127:0])
        //   6. Movl Vd[31:0], a                  (write low 32 bits)
        //
        // Clamp ordering is upper-first then lower-second.  For SQSHRUN,
        // any negative Sarq result is < UINT32_MAX so the upper Cmovq.g
        // is a no-op and the lower Cmovq.l fires to deliver 0; any
        // overflow-positive result first clamps down to UINT32_MAX, and
        // UINT32_MAX is not < 0 so the lower clamp is then a no-op.  For
        // SQSHRN at cnt==32 (immh:immb=0x20) the SAR result is already in
        // [INT32_MIN, INT32_MAX] and both clamps are runtime no-ops.
        //
        // For SQRSHRN, the pre-Sarq rounding add (1 << (cnt-1)) is in
        // [1, 0x80000000].  Because round_const > 0, the add can only
        // overflow positively (a near INT64_MAX); negative-direction
        // underflow is impossible.  We use Cmovq.o against INT64_MAX as
        // the saturation target — after the post-Sarq upper clamp drives
        // INT64_MAX >> cnt (positive, at least INT32_MAX at cnt==32) to
        // exactly INT32_MAX, which is the correct signed-saturating
        // narrow result.
        //
        // FPSR.QC is not updated here, mirroring the existing SQSHL .D
        // scalar GPR fallback at `lite_translator.h:19880` — neither
        // path sets QC.  This is perf-only and matches the historical
        // behaviour of the interpreter-bailout path for these forms.
        if (args.scalar && uses_signed_shift && src_bits == 64 &&
            !(is_rounding && is_saturating_signed_to_unsigned)) {
          const uint16_t immh_immb_local =
              static_cast<uint16_t>((immh << 3) | args.immb);
          const uint8_t narrow_rshift_local =
              static_cast<uint8_t>(64 - immh_immb_local);
          const int8_t cnt = static_cast<int8_t>(narrow_rshift_local);
          const int64_t clamp_max =
              is_saturating_signed_to_unsigned
                  ? int64_t{0xFFFFFFFF}  // UINT32_MAX
                  : int64_t{INT32_MAX};
          const int64_t clamp_min =
              is_saturating_signed_to_unsigned ? int64_t{0}
                                               : int64_t{INT32_MIN};
          Register a = AllocTempReg();
          Register max_reg = AllocTempReg();
          Register min_reg = AllocTempReg();
          Register round_reg =
              is_rounding ? AllocTempReg() : Assembler::no_register;
          SimdRegister xzero = AllocTempSimdReg();
          if (a == Assembler::no_register ||
              max_reg == Assembler::no_register ||
              min_reg == Assembler::no_register ||
              (is_rounding && round_reg == Assembler::no_register) ||
              xzero == no_simd_register) {
            success_ = false; return;
          }
          // Load Vn[63:0] as signed int64.
          as_.Movq(a, {.base = Assembler::rbp, .disp = vn_off});
          if (is_rounding) {
            // Saturating-signed pre-Sarq add of the rounding constant.
            // round_const = 1 << (cnt - 1), in [1, 0x80000000].
            const int64_t round_const = int64_t{1} << (cnt - 1);
            as_.Movq(round_reg, round_const);
            // Addq sets OF iff signed overflow.  Movq imm doesn't
            // touch EFLAGS, so the materialization of INT64_MAX into
            // max_reg between Addq and Cmovq.o keeps OF intact.
            as_.Addq(a, round_reg);
            as_.Movq(max_reg, int64_t{INT64_MAX});
            as_.Cmovq(Assembler::Condition::kOverflow, a, max_reg);
          }
          // Arithmetic right shift by `cnt` (range [1, 32]).
          as_.Sarq(a, cnt);
          // Clamp a to [clamp_min, clamp_max] branchlessly.  Movq imm
          // does not affect EFLAGS, so the materialization stays clear
          // of the Cmpq->Cmovq window.  Signed Cmovq.g / Cmovq.l work
          // for both SQSHRN/SQRSHRN and SQSHRUN: clamp_max is non-
          // negative in all cases (INT32_MAX, UINT32_MAX both fit in
          // int64 positive range), and clamp_min is non-positive in
          // all cases (INT32_MIN, 0).
          as_.Movq(max_reg, clamp_max);
          as_.Cmpq(a, max_reg);
          as_.Cmovq(Assembler::Condition::kGreater, a, max_reg);
          as_.Movq(min_reg, clamp_min);
          as_.Cmpq(a, min_reg);
          as_.Cmovq(Assembler::Condition::kLess, a, min_reg);
          // Zero Vd, then overlay the saturated int32 in Vd[31:0].
          as_.Pxor(xzero, xzero);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xzero);
          as_.Movl({.base = Assembler::rbp, .disp = vd_off}, a);
          return;
        }
        if (uses_signed_shift && src_bits == 64) {
          // Remaining signed-source-D paths that still need PSRAQ:
          //   - SQSHRN  vector .2S (args.scalar=false, src=64)
          //   - SQRSHRN vector .2S (args.scalar=false, src=64)
          //   - SQSHRUN vector .2S (args.scalar=false, src=64)
          //   - SQRSHRUN scalar D-source / vector .2S
          // Baseline SSE has no PSRAQ; fall back to the interpreter.
          success_ = false; return;
        }
        const uint16_t immh_immb = static_cast<uint16_t>((immh << 3) | args.immb);
        const uint8_t narrow_rshift = static_cast<uint8_t>(src_bits - immh_immb);
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xmask = AllocTempSimdReg();
        Register r1 = AllocTempReg();
        if (xn == no_simd_register || xmask == no_simd_register ||
            r1 == no_register) {
          success_ = false; return;
        }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        // SQRSHRUN computes its rounding carry pre-shift into xcarry and adds
        // it after the arithmetic shift (see below); other forms leave it unused.
        SimdRegister xcarry = no_simd_register;
        if (is_rounding) {
          // Broadcast the per-lane rounding constant (1 << (rshift - 1))
          // across all SIMD lanes, then add.  The 64-bit pattern packs
          // the constant into every lane of the source element width.
          // Plain RSHRN uses PADDW/PADDD/PADDQ (wrap-on-overflow is
          // benign — the subsequent PSHUFB narrow-mask drops the carry).
          // UQRSHRN must preserve the carry; see the comment block above
          // for the saturating-add variants.
          const uint64_t round_lane = uint64_t{1} << (narrow_rshift - 1);
          uint64_t round_pattern;
          switch (src_bits) {
            case 16:
              round_pattern = round_lane * uint64_t{0x0001000100010001ULL};
              break;
            case 32:
              round_pattern = round_lane * uint64_t{0x0000000100000001ULL};
              break;
            case 64:
              round_pattern = round_lane;
              break;
            default: success_ = false; return;
          }
          // SQRSHRUN does not use a broadcast rounding addend (it uses the
          // carry-bit identity below), so skip building xround for it.
          SimdRegister xround = no_simd_register;
          if (!is_saturating_signed_to_unsigned) {
            xround = AllocTempSimdReg();
            if (xround == no_simd_register) { success_ = false; return; }
            as_.Movq(r1, static_cast<int64_t>(round_pattern));
            as_.Movq(xround, r1);
            as_.Pinsrq(xround, r1, int8_t{1});
          }
          if (is_saturating_unsigned) {
            switch (src_bits) {
              case 16:
                as_.Paddusw(xn, xround);
                break;
              case 32: {
                const uint32_t clamp_lane =
                    0xFFFFFFFFu - static_cast<uint32_t>(round_lane);
                const uint64_t clamp_pattern =
                    (uint64_t{clamp_lane} << 32) | uint64_t{clamp_lane};
                SimdRegister xclamp = AllocTempSimdReg();
                if (!success()) { return; }
                as_.Movq(r1, static_cast<int64_t>(clamp_pattern));
                as_.Movq(xclamp, r1);
                as_.Pinsrq(xclamp, r1, int8_t{1});
                as_.Pminud(xn, xclamp);
                as_.Paddd(xn, xround);
                break;
              }
              // src_bits == 64 already bailed above.
            }
          } else if (is_saturating_signed_to_unsigned) {
            // SQRSHRUN: the rounding-add must NOT saturate at INT_MAX before
            // the shift.  When shift == dst_bits, a source within round_const
            // of INT_MAX rounds up to a MID-RANGE unsigned output (not the
            // saturated extreme), so a signed-saturating add (PADDSW / pre-
            // clamp PADDD) drops exactly 1 LSB there.  Use the exact identity
            //   (x + (1 << (s-1))) >> s  ==  (x >> s) + bit(s-1 of x)
            // for arithmetic >>: compute the carry bit now (xn intact) and add
            // it after the arithmetic shift below.  No wide add → no premature
            // saturation.  Isolate bit (cnt-1) into bit 0 of each lane by
            // shifting it to the top (left by src_bits-cnt) then logically back.
            xcarry = AllocTempSimdReg();
            if (xcarry == no_simd_register) { success_ = false; return; }
            as_.Movdqa(xcarry, xn);
            switch (src_bits) {
              case 16:
                as_.Psllw(xcarry, static_cast<int8_t>(16 - narrow_rshift));
                as_.Psrlw(xcarry, int8_t{15});
                break;
              case 32:
                as_.Pslld(xcarry, static_cast<int8_t>(32 - narrow_rshift));
                as_.Psrld(xcarry, int8_t{31});
                break;
              // src_bits == 64 already bailed above.
            }
          } else if (uses_signed_shift) {
            // SQRSHRN (signed→signed): a signed-saturating wide add is exact —
            // the boundary value saturates to INT*_MAX, which the downstream
            // signed clamp drives to the correct dst-max value.
            switch (src_bits) {
              case 16:
                // PADDSW is signed-saturating word add (SSE2).
                as_.Paddsw(xn, xround);
                break;
              case 32: {
                // No integer PADDSD in baseline SSE; pre-clamp xn so the
                // plain PADDD cannot overflow positively.  The round
                // constant is positive (`1 << (rshift - 1)`), so adding
                // it cannot cause negative underflow — no lower
                // pre-clamp needed.  Lanes that hit the upper clamp
                // settle at exactly INT32_MAX after the add, which the
                // post-shift PMINSD-vs-signed-max drives to the saturated
                // dst value (SQRSHRN only; SQRSHRUN uses the carry-bit path).
                const uint32_t clamp_lane =
                    0x7FFFFFFFu - static_cast<uint32_t>(round_lane);
                const uint64_t clamp_pattern =
                    (uint64_t{clamp_lane} << 32) | uint64_t{clamp_lane};
                SimdRegister xclamp = AllocTempSimdReg();
                if (!success()) { return; }
                as_.Movq(r1, static_cast<int64_t>(clamp_pattern));
                as_.Movq(xclamp, r1);
                as_.Pinsrq(xclamp, r1, int8_t{1});
                as_.Pminsd(xn, xclamp);
                as_.Paddd(xn, xround);
                break;
              }
              // src_bits == 64 already bailed above.
            }
          } else {
            switch (src_bits) {
              case 16: as_.Paddw(xn, xround); break;
              case 32: as_.Paddd(xn, xround); break;
              case 64: as_.Paddq(xn, xround); break;
            }
          }
        }
        const int8_t cnt = static_cast<int8_t>(narrow_rshift);
        if (uses_signed_shift) {
          // PSRAW/PSRAD: sign-preserving arithmetic right shift.  src=64
          // already bailed (no PSRAQ in baseline SSE).
          switch (src_bits) {
            case 16: as_.Psraw(xn, cnt); break;
            case 32: as_.Psrad(xn, cnt); break;
          }
        } else {
          switch (src_bits) {
            case 16: as_.Psrlw(xn, cnt); break;
            case 32: as_.Psrld(xn, cnt); break;
            case 64: as_.Psrlq(xn, cnt); break;
          }
        }
        if (is_rounding && is_saturating_signed_to_unsigned) {
          // SQRSHRUN: add the rounding carry (0/1 per lane) computed pre-shift.
          // Post-shift values are small (|x>>cnt| ≤ INT*_MAX>>1), so a plain
          // PADDW/PADDD cannot overflow before the unsigned clamp below.
          switch (src_bits) {
            case 16: as_.Paddw(xn, xcarry); break;
            case 32: as_.Paddd(xn, xcarry); break;
          }
        }
        if (is_saturating_signed) {
          // Clamp each src-element-wide lane to the signed dst range:
          //   dst 8:  [-128, 127] as signed 16-bit = [0xFF80, 0x007F].
          //   dst 16: [-32768, 32767] as signed 32-bit
          //           = [0xFFFF8000, 0x00007FFF].
          // PSHUFB later gathers byte 0 (src=16) or halfword 0 (src=32)
          // of each lane — those low bytes/halfwords already encode the
          // correct signed narrow value once the lane is clamped.
          uint64_t sat_max_pattern;
          uint64_t sat_min_pattern;
          switch (src_bits) {
            case 16:
              sat_max_pattern = 0x007F007F007F007FULL;
              sat_min_pattern = 0xFF80FF80FF80FF80ULL;
              break;
            case 32:
              sat_max_pattern = 0x00007FFF00007FFFULL;
              sat_min_pattern = 0xFFFF8000FFFF8000ULL;
              break;
            default: success_ = false; return;
          }
          SimdRegister xsatmax = AllocTempSimdReg();
          SimdRegister xsatmin = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movq(r1, static_cast<int64_t>(sat_max_pattern));
          as_.Movq(xsatmax, r1);
          as_.Pinsrq(xsatmax, r1, int8_t{1});
          as_.Movq(r1, static_cast<int64_t>(sat_min_pattern));
          as_.Movq(xsatmin, r1);
          as_.Pinsrq(xsatmin, r1, int8_t{1});
          switch (src_bits) {
            case 16:
              as_.Pminsw(xn, xsatmax);
              as_.Pmaxsw(xn, xsatmin);
              break;
            case 32:
              as_.Pminsd(xn, xsatmax);
              as_.Pmaxsd(xn, xsatmin);
              break;
          }
        }
        if (is_saturating_signed_to_unsigned) {
          // Clamp the post-shift signed value to the unsigned dst range
          // `[0, 2^dst_bits - 1]`:
          //   PMAXS{W,D} vs zero pins negative lanes at 0.
          //   PMINS{W,D} vs the (positive) dst-max value caps positives.
          // After PMAX-with-zero every lane is non-negative, so signed-
          // min against the positive cap is equivalent to unsigned-min
          // (avoids the SSE4.1-only PMINUW; PMINUD does exist in SSE4.1
          // but the signed form is just as good here).  PSHUFB then
          // gathers byte 0 (src=16) or halfword 0 (src=32) of each lane;
          // for a non-negative ≤ dst-max value, that bottom slice IS the
          // unsigned narrow representation.
          uint64_t sat_max_pattern;
          switch (src_bits) {
            case 16:
              sat_max_pattern = 0x00FF00FF00FF00FFULL;  // dst 8: 0xFF.
              break;
            case 32:
              sat_max_pattern = 0x0000FFFF0000FFFFULL;  // dst 16: 0xFFFF.
              break;
            default: success_ = false; return;
          }
          SimdRegister xsatmax = AllocTempSimdReg();
          SimdRegister xzero = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movq(r1, static_cast<int64_t>(sat_max_pattern));
          as_.Movq(xsatmax, r1);
          as_.Pinsrq(xsatmax, r1, int8_t{1});
          as_.Pxor(xzero, xzero);
          switch (src_bits) {
            case 16:
              as_.Pmaxsw(xn, xzero);    // negatives → 0.
              as_.Pminsw(xn, xsatmax);  // positives capped at 0xFF.
              break;
            case 32:
              as_.Pmaxsd(xn, xzero);    // SSE4.1.
              as_.Pminsd(xn, xsatmax);  // SSE4.1, cap at 0xFFFF.
              break;
          }
        }
        if (is_saturating_unsigned) {
          // Clamp each src-lane value to (1 << dst_bits) - 1 before the
          // PSHUFB narrow.  Pattern packs the per-lane unsigned max into
          // every src-element-wide slot of a 64-bit word, then broadcast
          // across both qwords of an xmm via Movq+Pinsrq.  PMINUW/PMINUD
          // are SSE4.1; the 64→32 case has no PMINUQ in baseline SSE so
          // it does a manual hi32-nonzero -> lo32=all-ones rewrite.
          uint64_t sat_pattern;
          switch (src_bits) {
            case 16: sat_pattern = 0x00FF00FF00FF00FFULL; break;
            case 32: sat_pattern = 0x0000FFFF0000FFFFULL; break;
            case 64: sat_pattern = 0x00000000FFFFFFFFULL; break;
            default: success_ = false; return;
          }
          SimdRegister xsat = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movq(r1, static_cast<int64_t>(sat_pattern));
          as_.Movq(xsat, r1);
          as_.Pinsrq(xsat, r1, int8_t{1});
          switch (src_bits) {
            case 16: as_.Pminuw(xn, xsat); break;
            case 32: as_.Pminud(xn, xsat); break;
            case 64: {
              SimdRegister xhi = AllocTempSimdReg();
              SimdRegister xzero = AllocTempSimdReg();
              if (!success()) { return; }
              as_.Movdqa(xhi, xn);
              as_.Psrlq(xhi, int8_t{32});
              as_.Pxor(xzero, xzero);
              // PCMPEQD: 0xFFFFFFFF per 32-bit lane where lane==0.  For
              // xhi-after-PSRLQ-32, 32-bit lanes 1 and 3 are always
              // zero (those were the upper halves of each qword cleared
              // by the shift), so PCMPEQD lights them up unconditionally
              // — but PSHUFD-0xA0 then overwrites lanes 1,3 with the
              // lane 0,2 values, so the spurious matches don't leak.
              as_.Pcmpeqd(xhi, xzero);
              as_.Pshufd(xhi, xhi, int8_t{static_cast<int8_t>(0xA0)});
              // XOR with xsat (which is 0xFFFFFFFF in 32-bit lanes 0,2
              // and 0 in 32-bit lanes 1,3) inverts polarity in the lo32
              // slots: post-XOR, lanes 0 and 2 are 0xFFFFFFFF iff the
              // qword's hi32 was nonzero (i.e., needs saturate).  Lanes
              // 1,3 are don't-cares — PSHUFB drops them.
              as_.Pxor(xhi, xsat);
              as_.Por(xn, xhi);
              break;
            }
          }
        }
        // PSHUFB selectors: 0x80 in the upper 8 entries zeros the upper
        // 64 bits of the result; the lower 8 entries gather the bytes
        // that form the narrowed value of each source lane.
        //   src 16-bit → low byte of each .8H lane = bytes 0,2,4,...,14.
        //   src 32-bit → low half of each .4S lane = bytes 0,1,4,5,8,9,12,13.
        //   src 64-bit → low word of each .2D lane = bytes 0,1,2,3,8,9,10,11.
        int64_t mask_lo;
        const int64_t mask_hi = static_cast<int64_t>(0x8080808080808080ULL);
        switch (src_bits) {
          case 16:
            mask_lo = static_cast<int64_t>(0x0E0C0A0806040200LL);
            break;
          case 32:
            mask_lo = static_cast<int64_t>(0x0D0C090805040100LL);
            break;
          case 64:
            mask_lo = static_cast<int64_t>(0x0B0A090803020100LL);
            break;
          default: success_ = false; return;
        }
        as_.Movq(r1, mask_lo);
        as_.Movq(xmask, r1);
        as_.Movq(r1, mask_hi);
        as_.Pinsrq(xmask, r1, int8_t{1});
        as_.Pshufb(xn, xmask);
        if (args.scalar) {
          // Scalar narrow: keep only the low `dst_bytes` bytes of the
          // PSHUFB-narrowed result; zero `Vd[127:dst_bits]` per ARM
          // ARM scalar semantics.  PSHUFB has already laid out the
          // narrowed-saturated value of every input lane into the low
          // half of xn, so the low `dst_bytes` bytes already hold the
          // correct lane-0 narrow result; the higher bytes carry the
          // narrowed values of upper lanes that the ARM scalar source
          // (Vn[src_bits-1:0] only) does not address.  Byte-granular
          // Pslldq + Psrldq with shift = `16 - dst_bytes` keeps just
          // the low slice and zeros the rest.
          //   src=16 → dst=B  (dst_bytes=1, shift=15)
          //   src=32 → dst=H  (dst_bytes=2, shift=14)
          //   src=64 → dst=S  (dst_bytes=4, shift=12)
          const int8_t dst_bytes = static_cast<int8_t>(src_bits / 16);
          const int8_t pack_shift = static_cast<int8_t>(16 - dst_bytes);
          as_.Pslldq(xn, pack_shift);
          as_.Psrldq(xn, pack_shift);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        } else if (args.q) {
          // SHRN2: preserve Vd[63:0]; place narrowed lanes in Vd[127:64].
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movq(xd, {.base = Assembler::rbp, .disp = vd_off});
          as_.Pslldq(xn, int8_t{8});
          as_.Por(xd, xn);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
        } else {
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        }
        return;
      }
      // SCVTF / UCVTF scalar fixed-point conversion.
      //
      //   Per ARM ARM C7.2.301 / C7.2.342 (scalar form):
      //     immh=01xx → .S (esize=32, source is 32-bit int)
      //     immh=1xxx → .D (esize=64, source is 64-bit int)
      //     fbits = 2*esize - immh:immb, in [1, esize].
      //   Result Vd[esize-1:0] = FP(src) / 2^fbits; Vd[127:esize] = 0.
      //
      // Lowering for SCVTF .S / UCVTF .S / SCVTF .D:
      //   1. Load the integer source from Vn[esize-1:0] into a GP reg.
      //      .S: Movl (32-bit load auto-zero-extends to 64).  For SCVTF
      //          we'll then Cvtsi2ssl which reads the 32-bit subreg as
      //          signed int32.  For UCVTF we Cvtsi2ssq the 64-bit value
      //          (zero-extended above), which fits in int64 and converts
      //          unsigned correctly.
      //      .D: Movq (64-bit load).  SCVTF then Cvtsi2sdq.
      //   2. PXOR scratch xmm to zero (carries the upper-zero into the
      //      final store).
      //   3. CVTSI2{SS,SD}{L,Q} into the low lane.
      //   4. Multiply by 2^-fbits via a materialized FP constant
      //      (Movl/Movq immediate → Movd/Movq into scratch xmm → MULSS/SD).
      //      The multiply is exact: 2^-fbits is a power of 2, so it just
      //      adjusts the exponent — no precision loss.  Combined with the
      //      single-rounded CVTSI2{SS,SD}, the result equals
      //      round-to-{float,double}(int_val * 2^-fbits), matching the
      //      ARM ARM spec FixedToFP(int_val, fbits, signed, FPCR).
      //   5. MOVDQU the full 16-byte xmm to Vd; the PXOR above guarantees
      //      Vd[127:esize] = 0.
      //
      // UCVTF .D (uint64 → double, where bit 63 may be set) uses the
      // halve / round-to-odd / convert / double-back fixup mirroring
      // the existing FpFixedPoint SCVTF/UCVTF (FpDataProc1) path: branch
      // on the sign bit, take the direct CVTSI2SDQ for values < 2^63,
      // and for values ≥ 2^63 use the round-to-odd halve + double-back
      // trick that preserves single-rounding.
      case Decoder::AdvSimdShiftImmOpcode::kScvtfFixed:
      case Decoder::AdvSimdShiftImmOpcode::kUcvtfFixed: {
        const uint8_t immh = args.immh;
        // Decoder gates this; defensive guard for the .H FP16 form
        // (immh=001x) which we don't JIT here.
        if (!(immh & 0b1100)) { success_ = false; return; }
        const bool is_double = (immh & 0b1000) != 0;  // immh=1xxx → .D
        const uint8_t esize_bits = is_double ? 64 : 32;
        const uint16_t immh_immb =
            static_cast<uint16_t>((immh << 3) | args.immb);
        // fbits = 2*esize - immh:immb.  Valid range: [1, esize_bits].
        const uint8_t fbits =
            static_cast<uint8_t>(2 * esize_bits - immh_immb);
        if (fbits == 0 || fbits > esize_bits) {
          success_ = false; return;
        }
        const bool is_unsigned =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUcvtfFixed);
        // Vector forms convert EVERY lane: .2S/.4S (esize 32) or .2D
        // (esize 64, Q=1).  args.scalar forces a single lane.  An earlier
        // scalar-only lowering converted lane 0 alone and zeroed the rest,
        // which sheared Skia's vectorized glyph rasterization (the lanes
        // beyond 0 went to 0.0).
        const uint8_t esize_bytes = esize_bits / 8;
        const uint8_t vec_len = args.q ? 16 : 8;
        const uint8_t num_elements =
            args.scalar ? 1 : static_cast<uint8_t>(vec_len / esize_bytes);

        SimdRegister xmm = AllocTempSimdReg();
        SimdRegister xscale = AllocTempSimdReg();
        Register gp_int = AllocTempReg();
        Register gp_scale = AllocTempReg();
        Register low_bit = no_register;
        if (is_double && is_unsigned) {
          low_bit = AllocTempReg();
        }
        if (xmm == no_simd_register || xscale == no_simd_register ||
            gp_int == no_register || gp_scale == no_register ||
            (is_double && is_unsigned && low_bit == no_register)) {
          success_ = false; return;
        }

        // NB: do NOT pre-zero Vd here.  For the in-place forms (rd == rn)
        // a Movdqu zero of Vd would clobber the Vn source before the
        // per-lane loads below; the unused high bytes are zeroed AFTER the
        // loop instead (the loop never reads Vn beyond esize*num_elements).

        // 2^-fbits scale (exact power of two: just an exponent bias),
        // computed once and reused across lanes.
        if (is_double) {
          // FP64 2^-fbits = sign 0, exp (1023 - fbits), mantissa 0.
          const uint64_t scale_bits =
              static_cast<uint64_t>(1023u - fbits) << 52;
          as_.Movq(gp_scale, static_cast<int64_t>(scale_bits));
          as_.Movq(xscale, gp_scale);
        } else {
          // FP32 2^-fbits = sign 0, exp (127 - fbits), mantissa 0.
          const uint32_t scale_bits =
              static_cast<uint32_t>(127u - fbits) << 23;
          as_.Movl(gp_scale, static_cast<int32_t>(scale_bits));
          as_.Movd(xscale, gp_scale);
        }

        for (uint8_t lane = 0; lane < num_elements; lane++) {
          const int32_t loff = vn_off + lane * esize_bytes;
          const int32_t soff = vd_off + lane * esize_bytes;
          // Load integer source lane.  The .S Movl zero-extends
          // gp_int[63:32]=0, so Cvtsi2ssq below reads the unsigned 32-bit
          // value (UCVTF .S), while Cvtsi2ssl reads the int32 subreg
          // (SCVTF .S).
          if (is_double) {
            as_.Movq(gp_int, {.base = Assembler::rbp, .disp = loff});
          } else {
            as_.Movl(gp_int, {.base = Assembler::rbp, .disp = loff});
          }
          // Integer → FP conversion (result in xmm low lane).
          if (is_double && is_unsigned) {
            // UCVTF .D: uint64 → double.  Values < 2^63 convert directly
            // via signed CVTSI2SDQ (exact); values ≥ 2^63 go through
            // round-to-odd halve / convert / double-back to preserve
            // single-rounding when bit 63 is set.
            Assembler::Label* neg_path = as_.MakeLabel();
            Assembler::Label* done = as_.MakeLabel();
            as_.Testq(gp_int, gp_int);
            as_.Jcc(Assembler::Condition::kNegative, *neg_path);
            as_.Cvtsi2sdq(xmm, gp_int);
            as_.Jmp(*done);
            as_.Bind(neg_path);
            as_.Movq(low_bit, gp_int);
            as_.Andq(low_bit, static_cast<int32_t>(1));   // captured LSB
            as_.Shrq(gp_int, static_cast<int8_t>(1));     // logical halve
            as_.Orq(gp_int, low_bit);                     // round-to-odd
            as_.Cvtsi2sdq(xmm, gp_int);
            as_.Addsd(xmm, xmm);                          // double back
            as_.Bind(done);
          } else if (is_double) {
            // SCVTF .D: int64 → double.
            as_.Cvtsi2sdq(xmm, gp_int);
          } else if (is_unsigned) {
            // UCVTF .S: uint32 (zero-extended to uint64) → float, via
            // signed-int64 conversion which is exact for values ≤ 2^32-1.
            as_.Cvtsi2ssq(xmm, gp_int);
          } else {
            // SCVTF .S: int32 → float.
            as_.Cvtsi2ssl(xmm, gp_int);
          }
          // Scale by 2^-fbits.
          if (is_double) {
            as_.Mulsd(xmm, xscale);
          } else {
            as_.Mulss(xmm, xscale);
          }
          // Store this lane (bounce xmm → GP → memory).
          if (is_double) {
            as_.Movq(gp_int, xmm);
            as_.Movq({.base = Assembler::rbp, .disp = soff}, gp_int);
          } else {
            as_.Movd(gp_int, xmm);
            as_.Movl({.base = Assembler::rbp, .disp = soff}, gp_int);
          }
        }
        // Zero Vd[127:esize*num_elements] now that every Vn lane has been
        // read (in-place-safe: never touches a not-yet-consumed Vn lane).
        const int used_bytes = esize_bytes * num_elements;
        if (used_bytes < 16) {
          as_.Xorl(gp_int, gp_int);
          if (used_bytes == 4) {
            as_.Movl({.base = Assembler::rbp, .disp = vd_off + 4}, gp_int);
          }
          as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, gp_int);
        }
        return;
      }
      // AdvSimdScalarShiftByImm JIT — FCVTZS (FP →
      // signed fixed-point int, truncate toward zero).  Mirrors the
      // AdvSimdScalarTwoRegMisc kFcvtzs lowering at lines 11369-11427,
      // but multiplies the FP source by 2^+fbits before truncating so
      // (FP * 2^fbits) trunc to int yields the desired fixed-point
      // result.  Multiply-by-power-of-2 is exact in IEEE-754, so
      // single-rounding semantics are preserved.  .H FP16 bails on the
      // immh guard.
      case Decoder::AdvSimdShiftImmOpcode::kFcvtzsFixed: {
        const uint8_t immh = args.immh;
        if (!(immh & 0b1100)) { success_ = false; return; }
        const bool is_double = (immh & 0b1000) != 0;
        const uint8_t esize_bits = is_double ? 64 : 32;
        const uint8_t esize_bytes = esize_bits / 8;
        const uint16_t immh_immb =
            static_cast<uint16_t>((immh << 3) | args.immb);
        // fbits = 2*esize - immh:immb.  Valid range: [1, esize_bits].
        const uint8_t fbits =
            static_cast<uint8_t>(2 * esize_bits - immh_immb);
        if (fbits == 0 || fbits > esize_bits) {
          success_ = false; return;
        }
        // Vector forms truncate EVERY lane (.2S/.4S/.2D); args.scalar
        // forces a single lane.  Earlier scalar-only lowering left lanes
        // beyond 0 zeroed.
        const uint8_t vec_len = args.q ? 16 : 8;
        const uint8_t num_elements =
            args.scalar ? 1 : static_cast<uint8_t>(vec_len / esize_bytes);

        SimdRegister xmm = AllocTempSimdReg();
        SimdRegister xscale = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        Register sign_tmp = AllocTempReg();
        Register gp_scale = AllocTempReg();
        if (xmm == no_simd_register || xscale == no_simd_register ||
            tmp == no_register || sign_tmp == no_register ||
            gp_scale == no_register) {
          success_ = false; return;
        }

        // NB: do NOT pre-zero Vd (would clobber Vn for the in-place rd==rn
        // forms); the unused high bytes are zeroed after the loop.

        // 2^+fbits scale (exact power of two), computed once.  NaN
        // propagates through MULSS/MULSD (NaN*x=NaN); ±inf stays ±inf.
        if (is_double) {
          const uint64_t scale_bits =
              static_cast<uint64_t>(1023u + fbits) << 52;
          as_.Movq(gp_scale, static_cast<int64_t>(scale_bits));
          as_.Movq(xscale, gp_scale);
        } else {
          const uint32_t scale_bits =
              static_cast<uint32_t>(127u + fbits) << 23;
          as_.Movl(gp_scale, static_cast<int32_t>(scale_bits));
          as_.Movd(xscale, gp_scale);
        }

        for (uint8_t lane = 0; lane < num_elements; lane++) {
          const int32_t loff = vn_off + lane * esize_bytes;
          const int32_t soff = vd_off + lane * esize_bytes;
          // Load FP source lane and scale by 2^+fbits.
          if (is_double) {
            as_.Movsd(xmm, {.base = Assembler::rbp, .disp = loff});
            as_.Mulsd(xmm, xscale);
          } else {
            as_.Movss(xmm, {.base = Assembler::rbp, .disp = loff});
            as_.Mulss(xmm, xscale);
          }
          // Truncate to int with ARM saturation fixup.
          //   NaN              -> 0
          //   positive overflow -> INT_MAX
          //   negative overflow -> INT_MIN (matches x86 cvtt indefinite)
          //   in-range          -> trunc toward zero
          if (is_double) {
            as_.Cvttsd2siq(tmp, xmm);
          } else {
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
          // Store this lane.
          if (is_double) {
            as_.Movq({.base = Assembler::rbp, .disp = soff}, tmp);
          } else {
            as_.Movl({.base = Assembler::rbp, .disp = soff}, tmp);
          }
        }
        // In-place-safe upper-zero of Vd[127:esize*num_elements].
        const int used_bytes = esize_bytes * num_elements;
        if (used_bytes < 16) {
          as_.Xorl(tmp, tmp);
          if (used_bytes == 4) {
            as_.Movl({.base = Assembler::rbp, .disp = vd_off + 4}, tmp);
          }
          as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, tmp);
        }
        return;
      }
      // AdvSimdScalarShiftByImm JIT — FCVTZU (FP →
      // unsigned fixed-point int, truncate toward zero).  Mirrors the
      // AdvSimdScalarTwoRegMisc kFcvtzu lowering at lines 11429-11535,
      // but multiplies the FP source by 2^+fbits before truncating.
      // Multiply-by-power-of-2 is exact in IEEE-754, so saturation
      // boundaries (NaN→0, neg→0, FP≥UMAX→UMAX) carry over verbatim.
      // .H FP16 bails on the immh guard.
      case Decoder::AdvSimdShiftImmOpcode::kFcvtzuFixed: {
        const uint8_t immh = args.immh;
        if (!(immh & 0b1100)) { success_ = false; return; }
        const bool is_double = (immh & 0b1000) != 0;
        const uint8_t esize_bits = is_double ? 64 : 32;
        const uint8_t esize_bytes = esize_bits / 8;
        const uint16_t immh_immb =
            static_cast<uint16_t>((immh << 3) | args.immb);
        // fbits = 2*esize - immh:immb.  Valid range: [1, esize_bits].
        const uint8_t fbits =
            static_cast<uint8_t>(2 * esize_bits - immh_immb);
        if (fbits == 0 || fbits > esize_bits) {
          success_ = false; return;
        }
        // Vector forms truncate EVERY lane (.2S/.4S/.2D); args.scalar
        // forces a single lane.  Earlier scalar-only lowering left lanes
        // beyond 0 zeroed.
        const uint8_t vec_len = args.q ? 16 : 8;
        const uint8_t num_elements =
            args.scalar ? 1 : static_cast<uint8_t>(vec_len / esize_bytes);

        SimdRegister xmm = AllocTempSimdReg();
        SimdRegister xscale = AllocTempSimdReg();
        Register tmp = AllocTempReg();
        Register sign_tmp = AllocTempReg();
        Register gp_scale = AllocTempReg();
        SimdRegister bound_xmm = no_simd_register;
        SimdRegister bound2_xmm = no_simd_register;
        if (is_double) {
          bound_xmm = AllocTempSimdReg();
          bound2_xmm = AllocTempSimdReg();
        }
        if (xmm == no_simd_register || xscale == no_simd_register ||
            tmp == no_register || sign_tmp == no_register ||
            gp_scale == no_register ||
            (is_double && (bound_xmm == no_simd_register ||
                           bound2_xmm == no_simd_register))) {
          success_ = false; return;
        }

        // NB: do NOT pre-zero Vd (would clobber Vn for the in-place rd==rn
        // forms); the unused high bytes are zeroed after the loop.

        // 2^+fbits scale (exact power of two) and the .D classification
        // bounds (2^63, 2^64), all computed once and reused across lanes.
        // NaN propagates (NaN*x=NaN); ±inf stays ±inf; ±0 stays ±0.
        if (is_double) {
          const uint64_t scale_bits =
              static_cast<uint64_t>(1023u + fbits) << 52;
          as_.Movq(gp_scale, static_cast<int64_t>(scale_bits));
          as_.Movq(xscale, gp_scale);
          as_.Movq(tmp, int64_t{0x43E0000000000000LL});  // 2^63
          as_.Movq(bound_xmm, tmp);
          as_.Movq(tmp, int64_t{0x43F0000000000000LL});  // 2^64
          as_.Movq(bound2_xmm, tmp);
        } else {
          const uint32_t scale_bits =
              static_cast<uint32_t>(127u + fbits) << 23;
          as_.Movl(gp_scale, static_cast<int32_t>(scale_bits));
          as_.Movd(xscale, gp_scale);
        }

        for (uint8_t lane = 0; lane < num_elements; lane++) {
          const int32_t loff = vn_off + lane * esize_bytes;
          const int32_t soff = vd_off + lane * esize_bytes;
          if (is_double) {
            as_.Movsd(xmm, {.base = Assembler::rbp, .disp = loff});
            as_.Mulsd(xmm, xscale);
          } else {
            as_.Movss(xmm, {.base = Assembler::rbp, .disp = loff});
            as_.Mulss(xmm, xscale);
          }

          if (is_double) {
            // FP64 → u64: NaN→0, negative→0, [0, 2^63)→direct cvttsd2siq,
            // [2^63, 2^64)→subtract-2^63 + cvtt + Btsq bit63,
            // [2^64, +inf]→UINT64_MAX.
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
          } else {
            // FP32 → u32 via cvtt-Q (int64), classify by upper-32
            // zero-ness.
            Assembler::Label* zero_path = as_.MakeLabel();
            Assembler::Label* done = as_.MakeLabel();
            as_.Ucomiss(xmm, xmm);
            as_.Jcc(Assembler::Condition::kParityEven, *zero_path);
            // FP sign-bit check via raw bits.  -0.0 falls through;
            // cvtt(-0.0)=0.
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
            // tmp's bottom 32 bits already correct; truncate via Movl
            // into tmp (auto-zero-extends to 64).
            as_.Movl(tmp, tmp);
            as_.Jmp(*done);
            as_.Bind(sat_max);
            as_.Movl(tmp, int32_t{-1});  // 0xFFFFFFFF
            as_.Jmp(*done);
            as_.Bind(zero_path);
            as_.Xorl(tmp, tmp);
            as_.Bind(done);
          }

          // Store this lane.
          if (is_double) {
            as_.Movq({.base = Assembler::rbp, .disp = soff}, tmp);
          } else {
            as_.Movl({.base = Assembler::rbp, .disp = soff}, tmp);
          }
        }
        // In-place-safe upper-zero of Vd[127:esize*num_elements].
        const int used_bytes = esize_bytes * num_elements;
        if (used_bytes < 16) {
          as_.Xorl(tmp, tmp);
          if (used_bytes == 4) {
            as_.Movl({.base = Assembler::rbp, .disp = vd_off + 4}, tmp);
          }
          as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, tmp);
        }
        return;
      }
      default:
        Undefined();
        return;
    }
  }

