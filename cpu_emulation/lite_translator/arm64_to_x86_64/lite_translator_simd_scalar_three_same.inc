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
// Family: SIMD scalar three-same.
// Methods:
//   AdvSimdScalarThreeSame

  // FMULX / FRECPS / FRSQRTS (scalar three-same, FP32/FP64)
  // JIT.
  //
  // FMULX is identical to FMUL except the (zero * infinity) saturation case
  // returns ±2.0 (sign = sign(a) XOR sign(b)) instead of the FMUL-produced
  // NaN.  See Interpreter::FmulxScalar<> for the reference semantics.
  //
  // FMULX strategy (branchless): always compute mul = a * b first.  The
  // result is NaN iff one of {a, b} is NaN OR {a, b} is the (0, ±inf) /
  // (±inf, 0) pair.  We construct a mask that is all-ones iff the special
  // case (0*inf) fired — that is, "mul is NaN AND neither input is NaN" —
  // then blend mul with ±2.0 under that mask:
  //
  //   special_mask = cmpunord(mul, mul) AND NOT cmpunord(a, b)
  //                = "mul became NaN purely from 0*inf"
  //   two_signed   = (a XOR b) AND sign_mask  OR  bits-of(2.0)
  //   result       = (mul AND NOT special_mask) OR (two_signed AND special_mask)
  //
  // Bit-exact match to the interpreter's FmulxScalar<> for every finite,
  // ±0, ±inf, and NaN input — FMUL handles NaN propagation, and the (0,
  // ±inf) lanes get replaced with ±2.0.  Sign of ±2.0 is sign(a) XOR
  // sign(b) (XOR of the source sign bits), matching std::signbit(a) ^
  // std::signbit(b) in the interpreter.
  //
  // FRECPS  semantics (ARM ARM C7.2.151): FPRecipStepFused = FMA(-a,b,2.0).
  //   - either input NaN  -> default qNaN
  //   - (±0, ±inf) cross  -> +2.0 (unsigned — note: not ±2.0 like FMULX)
  //   - otherwise         -> single-rounded (2.0 - a*b) via VFNMADD231.
  // FRSQRTS semantics (ARM ARM C7.2.155): FPRSqrtStepFused = FMA(-a,b,3.0)/2.
  //   - either input NaN  -> default qNaN
  //   - (±0, ±inf) cross  -> +1.5
  //   - otherwise         -> (3.0 - a*b) via VFNMADD231, then /2.0 (the
  //                          divide-by-2 is exact in IEEE binary FP, so
  //                          the overall single-rounded property carries
  //                          through from the FMA).
  // FRECPS / FRSQRTS strategy: build two masks (input_unord, special_mask)
  // exactly like FMULX, then layered-select between fma_normal, the K_sat
  // saturation constant (+2.0 / +1.5), and the default qNaN.  Saturation
  // constant is sign-fixed positive here (no XOR-sign step).  NaN input must
  // be replaced with the default qNaN even though FMA naturally propagates
  // one of the input NaNs — the ARM ARM mandates the default-NaN payload.
  //
  // FABD scalar (FP32/FP64) lowers to SUBSS/SUBSD followed by an AND with
  // the non-sign-bit mask (clear bit 31 / bit 63 of the result lane).  This
  // is bit-exact for std::fabs(a - b): finite paths trivially match, and
  // NaN paths match because the subtract quietens SNaNs and propagates a
  // NaN payload with whatever sign x86 chose — the AND-mask then clears
  // the sign bit, giving the same |NaN| the interpreter's std::fabs(NaN)
  // produces.  Inexact / invalid exceptions flow through MXCSR exactly as
  // for SUB; the region-exit MXCSR→FPSR mirror picks them up.
  //
  // FCMEQ / FCMGE / FCMGT (FP32/FP64) scalar lower to CMPEQSS/CMPLESS/
  // CMPLTSS (or the SD variants) with operand reorder: ARM "a > b" is
  // x86 "b < a", so we issue Cmpltss/Cmpltsd with the operands swapped
  // and the destination is the (loaded-into-scratch) Vb register.  The
  // result is a 32-bit / 64-bit all-ones mask in lane 0 (or zero), with
  // bits above the lane zeroed by the Pxor+Movdqu+Movss/Movsd dst-write
  // pattern.  Matches the interpreter's `(a == b) ? 0xFFFF... : 0`
  // C++-operator semantics exactly for finite, +/-0, +/-inf, and NaN
  // inputs (C++ FP comparisons return false for any NaN-involved cmp;
  // x86 CMPSS predicates 0/1/2 return all-zeros for any NaN input).
  //
  // FACGE / FACGT (FP32/FP64) scalar are the same as FCMGE / FCMGT but
  // with abs() applied to both operands first via AND-with-non-sign-mask
  // — same mask construction as FABD.
  //
  // The FP16 path (is_fp16=true) bails to the interpreter via
  // success_=false — it is a JIT follow-up; the interpreter handles it
  // correctly today.
  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs& args) {
    const auto opc = args.opcode;
    const bool is_fmulx = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFmulx);
    const bool is_frecps = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFrecps);
    const bool is_frsqrts = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFrsqrts);
    const bool is_fabd = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFabd);
    const bool is_fcmeq = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFcmeq);
    const bool is_fcmge = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFcmge);
    const bool is_fcmgt = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFcmgt);
    const bool is_facge = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFacge);
    const bool is_facgt = (opc == Decoder::AdvSimdScalarThreeSameOpcode::kFacgt);
    const bool is_cmp = is_fcmeq || is_fcmge || is_fcmgt || is_facge || is_facgt;
    const bool is_sqrdmlah_scalar =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmlahScalar);
    const bool is_sqrdmlsh_scalar =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmlshScalar);
    const bool is_sqrdm_scalar = is_sqrdmlah_scalar || is_sqrdmlsh_scalar;
    const bool is_sqdmulh_scalar =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSqdmulhScalar);
    const bool is_sqrdmulh_scalar =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmulhScalar);
    const bool is_sq_d_r_mulh_scalar = is_sqdmulh_scalar || is_sqrdmulh_scalar;
    const bool is_dform_add =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kAdd);
    const bool is_dform_sub =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSub);
    const bool is_dform_cmgt =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kCmgt);
    const bool is_dform_cmhi =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kCmhi);
    const bool is_dform_cmge =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kCmge);
    const bool is_dform_cmhs =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kCmhs);
    const bool is_dform_cmtst =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kCmtst);
    const bool is_dform_cmeq =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kCmeq);
    const bool is_dform_int =
        is_dform_add || is_dform_sub || is_dform_cmgt || is_dform_cmhi ||
        is_dform_cmge || is_dform_cmhs || is_dform_cmtst || is_dform_cmeq;
    const bool is_sqadd_scalar =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSqaddScalar);
    const bool is_uqadd_scalar =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kUqaddScalar);
    const bool is_sqsub_scalar =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSqsubScalar);
    const bool is_uqsub_scalar =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kUqsubScalar);
    const bool is_satarith_scalar =
        is_sqadd_scalar || is_uqadd_scalar || is_sqsub_scalar || is_uqsub_scalar;
    // All four sizes are JIT-lowered.  B/H use the one-instruction saturating
    // SSE ops (PADDSB/PADDUSB/PSUBSB/PSUBUSB).  S uses the 32-bit recipe ported
    // single-lane from the vector SQADD/UQADD/SQSUB/UQSUB lowerings.  D uses a
    // 64-bit recipe: PADDQ/PSUBQ wrap-arith + sign-bit extraction via PSRAD 31
    // + PSHUFD 0xF5 (no PSRAQ in baseline SSE) for signed; PCMPGTQ on
    // sign-flipped operands (SSE4.2) for unsigned.
    const bool is_satarith_scalar_bhsd = is_satarith_scalar;
    // SSHL / USHL scalar (size=11 D form only — the encoding allocates other
    // sizes to the vector form, and the decoder above rejects them).  Vd[63:0]
    // = a shifted by the signed int8 in Vm[7:0]; Vd[127:64] = 0.  Implemented
    // in GPR scalar with branches because (a) PSRAQ is AVX-512F-VL only and
    // (b) the shift amount is runtime-variable, so a branched SHL/SHR/SAR
    // path is simpler than a clamped SSE-shift emulation.
    const bool is_sshl_scalar_d =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSshl);
    const bool is_ushl_scalar_d =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kUshl);
    const bool is_shl_scalar_d = is_sshl_scalar_d || is_ushl_scalar_d;
    // UQSHL scalar (all sizes — B, H, S, D).
    // Unsigned saturating variable left shift; the shift amount is the low 8
    // bits of Vm interpreted as int8_t.  Negative shifts behave exactly like
    // USHL (no saturation possible on right shifts of a non-negative value);
    // positive shifts saturate to umax_N = (1 << bits_local) - 1 when bits
    // get pushed past bit (bits_local - 1).  Implemented in GPR scalar with
    // branches, parameterized on `bits_local = 1 << (3 + args.size)`.  The
    // mask-then-back-shift overflow detector reduces to a no-op for the D
    // form (bits_local==64) where the 64-bit ShlqByCl naturally truncates.
    const bool is_uqshl_scalar_d =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kUqshlScalar);
    // SQSHL scalar (all sizes — B, H, S, D).
    // Signed saturating variable left shift; same scaffolding as UQSHL but
    // the back-shift uses SAR (arithmetic) and the saturation target depends
    // on sign(a): positive a → INT_MAX_N, negative a → INT_MIN_N.  Negative
    // shifts behave exactly like SSHL (signed arithmetic right shift; no
    // saturation possible on right shifts).  Implemented in GPR scalar with
    // branches, parameterized on `bits_local = 1 << (3 + args.size)`.  The
    // sign-extend-then-back-shift overflow detector reduces to a no-op
    // sign-extend step at N=64.
    const bool is_sqshl_scalar_d =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSqshlScalar);
    // URSHL scalar (D form only — the size=11 path).
    // Unsigned non-saturating rounded variable shift; left shifts discard
    // upper bits (same as USHL); right shifts add a round-half-up bias
    // 1 << (rshift-1) before the shift.  Implemented via the standard
    // overflow-avoiding identity
    //   (a + (1 << (rshift-1))) >>u rshift  ==
    //   (a >>u rshift) + ((a >>u (rshift-1)) & 1)
    // so that a near UINT64_MAX doesn't wrap during the bias addition.
    const bool is_urshl_scalar_d =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kUrshlScalar);
    // SRSHL scalar (D form only — the size=11 path).
    // Signed non-saturating rounded variable shift; the signed counterpart of
    // URSHL.  Left shifts truncate (same as SSHL).  Right shifts use the same
    // round-bit identity as URSHL but with SAR (signed arithmetic) on the
    // data shift; the round-bit extraction stays unsigned because we only
    // want bit value, not arithmetic interpretation.  The |sh|=64 boundary
    // collapses into the |sh|>=64 zero branch — for any int64 a, the signed
    // bias (1<<63) sums with a to a non-negative int128 in [0, 2^64-1]
    // which >>s 64 = 0, so no dedicated |sh|=64 branch is needed.
    const bool is_srshl_scalar_d =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSrshlScalar);
    // UQRSHL scalar (all sizes — unsigned saturating
    // rounded variable shift, lane width N = 8/16/32/64).  Combines UQSHL
    // scalar positive arm (shift-then-mask-then-back-shift overflow detector
    // + saturation to umax_N = (1 << N) - 1) with URSHL D negative arm
    // (overflow-safe rounding identity, dedicated |sh|=N bit-(N-1) extraction
    // branch).  Saturation and rounding apply in disjoint quadrants —
    // positive shifts saturate but never round, negative shifts round but
    // never saturate — so the two scaffoldings compose without interaction.
    // The `_d` suffix is historical; the flag detects UQRSHL scalar at any
    // size.
    const bool is_uqrshl_scalar_d =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kUqrshlScalar);
    // SQRSHL scalar (all sizes — signed saturating rounded
    // variable shift, lane width N = 8/16/32/64).  Combines SQSHL B/H/S/D
    // positive arm (sign-extended load + SignExtendFromN between Shl/Sar
    // overflow detector + LoadIntMaxN + sign-broadcast XOR + MaskToN at
    // store) with SRSHL D negative arm parameterized on bits_local
    // (overflow-safe rounding identity with SAR data shift + SHR round-bit
    // extraction; |sh|>=bits_local collapses to zero because the signed
    // bias 1<<(N-1) sums with any int_N a to a non-negative int_{>N} in
    // [0, 2^N-1] which >>s bits_local = 0, and for |sh|>bits_local the
    // rounding term dominates).  Saturation applies only on the positive
    // (left-shift) arm and rounding only on the negative (right-shift)
    // arm, so the two scaffoldings compose without interaction.  The `_d`
    // suffix is historical; the flag detects SQRSHL scalar at any size.
    const bool is_sqrshl_scalar_d =
        (opc == Decoder::AdvSimdScalarThreeSameOpcode::kSqrshlScalar);
    if (!is_fmulx && !is_frecps && !is_frsqrts && !is_fabd && !is_cmp &&
        !is_sqrdm_scalar && !is_sq_d_r_mulh_scalar && !is_dform_int &&
        !is_satarith_scalar_bhsd && !is_shl_scalar_d && !is_uqshl_scalar_d &&
        !is_sqshl_scalar_d && !is_urshl_scalar_d && !is_srshl_scalar_d &&
        !is_uqrshl_scalar_d && !is_sqrshl_scalar_d) {
      success_ = false; return;
    }
    // SQRDMLAH/SQRDMLSH scalar three-same (Armv8.1-RDM).
    // Single-lane port of the AdvSimdThreeSame vector lowering:
    //   size=01 (H): SSSE3 PMULHRSW + (INT16_MIN)² corner fix, then PADDSW /
    //                PSUBSW for the stage-2 signed-saturating accumulate.
    //   size=10 (S): SSE4.1 PMULDQ to reconstruct the signed 32×32 → 64
    //                product in qword 0, PSLLQ 1 to double, PADDQ rounding
    //                constant 2^31, PSRLQ 32 to extract the upper 32 bits
    //                into dword 0 (PSRLQ also zero-fills dwords 1..3 because
    //                qword 1 = 0x80000000 → shifted out), then PCMPEQD-based
    //                corner XOR, then the wrap-add + overflow-bit-XOR-blend
    //                signed-saturating 32-bit accumulate.
    // Vd zero-extends above the lane: Vn/Vm/Vd are loaded with Movd / Pxor +
    // Pinsrw, so all upper lanes are 0 throughout; the recipe propagates
    // zeros into the upper lanes of the result; a full 128-bit Movdqu then
    // writes Vd with Vd[127:bits_local] = 0.
    if (is_sqrdm_scalar) {
      if (args.size != 0b01 && args.size != 0b10) {
        success_ = false; return;
      }
      const bool is_sub = is_sqrdmlsh_scalar;
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      if (args.size == 0b01) {
        if (!host_platform::kHasSSSE3) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        SimdRegister xn_corner = AllocTempSimdReg();
        SimdRegister xm_corner = AllocTempSimdReg();
        SimdRegister x_min = AllocTempSimdReg();
        if (!success()) { return; }
        // Load Vn.h[0], Vm.h[0], Vd.h[0] with upper-lane zero.
        as_.Pxor(xn, xn);
        as_.Pinsrw(xn, {.base = Assembler::rbp, .disp = vn_off}, int8_t{0});
        as_.Pxor(xm, xm);
        as_.Pinsrw(xm, {.base = Assembler::rbp, .disp = vm_off}, int8_t{0});
        as_.Pxor(xd, xd);
        as_.Pinsrw(xd, {.base = Assembler::rbp, .disp = vd_off}, int8_t{0});
        // x_min = 0x8000 broadcast across 8 halfwords.  Upper lanes of xn/xm
        // are 0 ≠ 0x8000, so the corner mask collapses to 0 in lanes 1..7.
        as_.Pcmpeqw(x_min, x_min);
        as_.Psllw(x_min, int8_t{15});
        // Stage 1: SQRDMULH via PMULHRSW with corner fix at lane 0.
        as_.Movdqa(xn_corner, xn);
        as_.Movdqa(xm_corner, xm);
        as_.Pmulhrsw(xn, xm);
        as_.Pcmpeqw(xn_corner, x_min);
        as_.Pcmpeqw(xm_corner, x_min);
        as_.Pand(xn_corner, xm_corner);
        as_.Pxor(xn, xn_corner);
        // Stage 2: signed-saturating PADDSW / PSUBSW into Vd.h[0].
        if (is_sub) {
          as_.Psubsw(xd, xn);
        } else {
          as_.Paddsw(xd, xn);
        }
        // Lanes 1..7 of xd: 0 ± 0 = 0.  Lane 0 holds the saturated result.
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
        return;
      }
      // size == 0b10 (S form).
      if (!host_platform::kHasSSE4_1) { success_ = false; return; }
      SimdRegister xn = AllocTempSimdReg();
      SimdRegister xm = AllocTempSimdReg();
      SimdRegister xd = AllocTempSimdReg();
      SimdRegister x_const = AllocTempSimdReg();
      SimdRegister corner = AllocTempSimdReg();
      SimdRegister xp = AllocTempSimdReg();
      if (!success()) { return; }
      // Load Vn.s[0], Vm.s[0], Vd.s[0] with upper-lane zero.
      as_.Movd(xn, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movd(xm, {.base = Assembler::rbp, .disp = vm_off});
      as_.Movd(xd, {.base = Assembler::rbp, .disp = vd_off});
      // x_const = INT32_MIN broadcast across 4 dwords.
      as_.Pcmpeqd(x_const, x_const);
      as_.Pslld(x_const, int8_t{31});
      // Corner detection (lane 0 only — lanes 1..3 of xn/xm are 0 ≠
      // INT32_MIN, so corner mask collapses to 0 there).
      as_.Movdqa(corner, xn);
      as_.Pcmpeqd(corner, x_const);
      as_.Movdqa(xp, xm);
      as_.Pcmpeqd(xp, x_const);
      as_.Pand(corner, xp);
      // Stage 1: PMULDQ → qword 0 = sext(Vn.s[0]) * sext(Vm.s[0]); qword 1 = 0
      // (since src lanes 2/3 are 0).
      as_.Movdqa(xp, xn);
      as_.Pmuldq(xp, xm);
      // Double via PSLLQ 1.
      as_.Psllq(xp, int8_t{1});
      // Rounding constant 2^31 per qword.  Build in x_const.
      as_.Pcmpeqd(x_const, x_const);
      as_.Psllq(x_const, int8_t{63});
      as_.Psrlq(x_const, int8_t{32});
      as_.Paddq(xp, x_const);
      // Extract the upper 32 bits of qword 0 into dword 0 via PSRLQ 32.
      //   qword 0 after shift = (2*product + 0x80000000) >> 32 = high32
      //   qword 1 after shift = 0x80000000 >> 32 = 0
      // Resulting layout: dword[0] = high32, dword[1..3] = 0.
      as_.Psrlq(xp, int8_t{32});
      // Apply corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX at lane 0;
      // lanes 1..3 of corner are 0 so no perturbation there.
      as_.Pxor(xp, corner);
      // Stage 2: 32-bit signed-saturating add/sub via wrap+XOR-blend recipe.
      SimdRegister t_sum = AllocTempSimdReg();
      SimdRegister t_ovf = AllocTempSimdReg();
      SimdRegister t_sat = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqa(t_sum, xd);
      if (is_sub) {
        as_.Psubd(t_sum, xp);
      } else {
        as_.Paddd(t_sum, xp);
      }
      as_.Movdqa(t_ovf, xd);
      as_.Pxor(t_ovf, xp);
      as_.Movdqa(t_sat, xd);
      as_.Pxor(t_sat, t_sum);
      if (!is_sub) {
        as_.Pcmpeqd(xp, xp);
        as_.Pxor(t_ovf, xp);
        as_.Pand(t_ovf, t_sat);
        as_.Psrld(xp, int8_t{1});
      } else {
        as_.Pand(t_ovf, t_sat);
        as_.Pcmpeqd(xp, xp);
        as_.Psrld(xp, int8_t{1});
      }
      as_.Psrad(t_ovf, int8_t{31});
      as_.Psrad(xd, int8_t{31});
      as_.Pxor(xd, xp);
      as_.Pxor(xd, t_sum);
      as_.Pand(xd, t_ovf);
      as_.Pxor(xd, t_sum);
      // Lane-0 = saturated result.  Lanes 1..3 stay zero because xd started
      // zero in those lanes and the recipe propagates 0 through PSRAD /
      // PAND / PXOR with t_sum=0 and t_ovf=0.
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
      return;
    }
    // SQDMULH / SQRDMULH scalar (H/S only).
    //
    // Single-lane port of stage 1 of the SQRDMLAH/SQRDMLSH scalar recipe
    // above (same MOVQ/MOVD load shape, same corner-fix XOR), minus the
    // stage-2 saturating accumulate.  SQDMULH skips the rounding constant;
    // SQRDMULH includes it.
    //
    // size=01 (H): two paths.
    //   SQRDMULH .H: SSSE3 PMULHRSW + (INT16_MIN)^2 corner fix at lane 0.
    //                Lanes 1..7 of operands are 0 by construction (Pinsrw
    //                following Pxor), so PMULHRSW lanes 1..7 are 0 and the
    //                corner mask collapses to 0 there.
    //   SQDMULH  .H: PMULLW + PMULHW + PUNPCKLWD + PSRAD 15 + PACKSSDW
    //                (signed-saturating pack at 16-bit lane granularity).
    //                Packing against a zero scratch keeps word lanes 1..7
    //                clean.
    //
    // size=10 (S): single path.
    //   PMULDQ (sext32→64 of qword-low-32) gives qword 0 = sext(Vn.s[0]) *
    //   sext(Vm.s[0]) and qword 1 = 0.  PSLLQ 1 doubles.  For SQRDMULH only,
    //   add rounding constant 2^31 per qword (built via Pcmpeqd + Psllq 63
    //   + Psrlq 32 = 0x0000...0000_80000000 in both qwords).  PSRLQ 32
    //   extracts upper 32 bits of qword 0 into dword 0 and zero-fills
    //   dwords 1..3.  Corner XOR turns the (Vn=Vm=INT32_MIN) wrap-to-
    //   INT32_MIN into INT32_MAX at lane 0; corner mask is 0 elsewhere.
    if (is_sq_d_r_mulh_scalar) {
      if (args.size != 0b01 && args.size != 0b10) {
        success_ = false; return;
      }
      const bool is_rounded = is_sqrdmulh_scalar;
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      if (args.size == 0b01) {
        if (!host_platform::kHasSSSE3) { success_ = false; return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        // Load Vn.h[0] / Vm.h[0] with upper-lane zero.
        as_.Pxor(xn, xn);
        as_.Pinsrw(xn, {.base = Assembler::rbp, .disp = vn_off}, int8_t{0});
        as_.Pxor(xm, xm);
        as_.Pinsrw(xm, {.base = Assembler::rbp, .disp = vm_off}, int8_t{0});
        SimdRegister xmm_result = no_simd_register;
        if (is_rounded) {
          SimdRegister xn_corner = AllocTempSimdReg();
          SimdRegister xm_corner = AllocTempSimdReg();
          SimdRegister x_min = AllocTempSimdReg();
          if (!success()) { return; }
          // x_min = 0x8000 broadcast across 8 halfwords.
          as_.Pcmpeqw(x_min, x_min);
          as_.Psllw(x_min, int8_t{15});
          as_.Movdqa(xn_corner, xn);
          as_.Movdqa(xm_corner, xm);
          as_.Pmulhrsw(xn, xm);
          as_.Pcmpeqw(xn_corner, x_min);
          as_.Pcmpeqw(xm_corner, x_min);
          as_.Pand(xn_corner, xm_corner);
          as_.Pxor(xn, xn_corner);
          xmm_result = xn;
        } else {
          // SQDMULH .H via 32-bit reconstruction + arithmetic shift +
          // signed-saturating pack.
          SimdRegister x_low = AllocTempSimdReg();
          SimdRegister x_high = AllocTempSimdReg();
          SimdRegister x_zero = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(x_low, xn);
          as_.Movdqa(x_high, xn);
          as_.Pmullw(x_low, xm);
          as_.Pmulhw(x_high, xm);
          // Interleave (low16, high16) → 32-bit signed product in dword 0;
          // dwords 1..3 are 0 because operand lanes 1..7 are 0.
          as_.Punpcklwd(x_low, x_high);
          as_.Psrad(x_low, int8_t{15});
          // Signed-saturating pack to 16-bit.  Pack against a zero scratch
          // so word lanes 1..7 are 0 (PACKSSDW src2's 4 dwords map to dst
          // words 4..7).
          as_.Pxor(x_zero, x_zero);
          as_.Packssdw(x_low, x_zero);
          xmm_result = x_low;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_result);
        return;
      }
      // size == 0b10 (S form).
      if (!host_platform::kHasSSE4_1) { success_ = false; return; }
      SimdRegister xn = AllocTempSimdReg();
      SimdRegister xm = AllocTempSimdReg();
      SimdRegister x_const = AllocTempSimdReg();
      SimdRegister corner = AllocTempSimdReg();
      SimdRegister xp = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movd(xn, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movd(xm, {.base = Assembler::rbp, .disp = vm_off});
      // x_const = INT32_MIN broadcast across 4 dwords.
      as_.Pcmpeqd(x_const, x_const);
      as_.Pslld(x_const, int8_t{31});
      // Corner detection: lane 0 only (lanes 1..3 of xn/xm are 0 ≠ INT32_MIN).
      as_.Movdqa(corner, xn);
      as_.Pcmpeqd(corner, x_const);
      as_.Movdqa(xp, xm);
      as_.Pcmpeqd(xp, x_const);
      as_.Pand(corner, xp);
      // Stage 1: PMULDQ → qword 0 = sext(Vn.s[0]) * sext(Vm.s[0]); qword 1 = 0.
      as_.Movdqa(xp, xn);
      as_.Pmuldq(xp, xm);
      // Double via PSLLQ 1.
      as_.Psllq(xp, int8_t{1});
      if (is_rounded) {
        // Rounding constant 2^31 per qword.
        as_.Pcmpeqd(x_const, x_const);
        as_.Psllq(x_const, int8_t{63});
        as_.Psrlq(x_const, int8_t{32});
        as_.Paddq(xp, x_const);
      }
      // Extract upper 32 bits of qword 0 into dword 0; PSRLQ 32 also zero-
      // fills dwords 1..3 (qword 1's pre-shift value is at most 0x80000000
      // when is_rounded, which shifts out to 0).
      as_.Psrlq(xp, int8_t{32});
      // Apply corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX at lane 0;
      // corner is 0 at lanes 1..3 so no perturbation.
      as_.Pxor(xp, corner);
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xp);
      return;
    }
    // SQADD / UQADD / SQSUB / UQSUB scalar (B/H/S sizes).
    //
    // ARM ARM C7.2.282 / .284 / .317 / .319: the scalar single-lane forms
    // saturate at width 8/16/32/64 selected by args.size.  Here we cover
    // B (size=00), H (size=01), and S (size=10); D (size=11) bails to the
    // interpreter (no direct SSE primitive for 64-bit saturating add/sub).
    //
    // Recipe per (op, size):
    //   B/H — Pxor + Pinsrb/Pinsrw load Vn / Vm with lane 0 = source byte/
    //         halfword and lanes 1.. = 0, then apply the matching
    //         one-instruction saturating SSE op:
    //
    //       SQADD B  → PADDSB     (SSE2)
    //       SQADD H  → PADDSW     (SSE2)
    //       UQADD B  → PADDUSB    (SSE2)
    //       UQADD H  → PADDUSW    (SSE2)
    //       SQSUB B  → PSUBSB     (SSE2)
    //       SQSUB H  → PSUBSW     (SSE2)
    //       UQSUB B  → PSUBUSB    (SSE2)
    //       UQSUB H  → PSUBUSW    (SSE2)
    //
    //         Saturating-arith on (0,0) yields 0, so lanes 1.. stay 0.
    //
    //   S — Movd loads Vn.s[0] / Vm.s[0] with dwords 1..3 zero-extended,
    //       then runs the 32-bit signed/unsigned saturating recipe used by
    //       the vector lowerings at lite_translator.h:4820-4845 (SQADD via
    //       wrap-add + sign-bit XOR-blend), :4868-4879 (UQADD via PMAXUD +
    //       PCMPEQD + POR), :4896-4918 (SQSUB via wrap-sub + sign-bit
    //       XOR-blend), and :4940-4947 (UQSUB via PMINUD + PCMPEQD + PAND).
    //       Lanes 1..3 are 0 throughout: 0±0=0 (no overflow), the recipes'
    //       sign/cmp/and steps all produce 0 for zero inputs, so the
    //       result has dwords 1..3 = 0 automatically.  A full-width Movdqu
    //       then writes Vd with Vd[127:bits_local] = 0.
    //
    // Host features:
    //   * B form: SSE4.1 (Pinsrb).
    //   * H form: SSE2 baseline.
    //   * S signed (SQADD/SQSUB): SSE2 baseline (Movd / Paddd / Psubd /
    //     Pxor / Pand / Pcmpeqd / Psrad / Psrld).
    //   * S unsigned (UQADD/UQSUB): SSE4.1 (Pmaxud / Pminud).
    if (is_satarith_scalar_bhsd) {
      if (args.size == 0b00 && !host_platform::kHasSSE4_1) {
        success_ = false; return;
      }
      if (args.size == 0b10 && (is_uqadd_scalar || is_uqsub_scalar) &&
          !host_platform::kHasSSE4_1) {
        success_ = false; return;
      }
      if (args.size == 0b11 && (is_uqadd_scalar || is_uqsub_scalar) &&
          !host_platform::kHasSSE4_2) {
        success_ = false; return;
      }
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      SimdRegister xn = AllocTempSimdReg();
      SimdRegister xm = AllocTempSimdReg();
      if (!success()) { return; }
      if (args.size == 0b00) {
        as_.Pxor(xn, xn);
        as_.Pxor(xm, xm);
        as_.Pinsrb(xn, {.base = Assembler::rbp, .disp = vn_off}, int8_t{0});
        as_.Pinsrb(xm, {.base = Assembler::rbp, .disp = vm_off}, int8_t{0});
        if (is_sqadd_scalar)      as_.Paddsb(xn, xm);
        else if (is_uqadd_scalar) as_.Paddusb(xn, xm);
        else if (is_sqsub_scalar) as_.Psubsb(xn, xm);
        else                      as_.Psubusb(xn, xm);
      } else if (args.size == 0b01) {
        as_.Pxor(xn, xn);
        as_.Pxor(xm, xm);
        as_.Pinsrw(xn, {.base = Assembler::rbp, .disp = vn_off}, int8_t{0});
        as_.Pinsrw(xm, {.base = Assembler::rbp, .disp = vm_off}, int8_t{0});
        if (is_sqadd_scalar)      as_.Paddsw(xn, xm);
        else if (is_uqadd_scalar) as_.Paddusw(xn, xm);
        else if (is_sqsub_scalar) as_.Psubsw(xn, xm);
        else                      as_.Psubusw(xn, xm);
      } else if (args.size == 0b10) {  // S form (32-bit scalar)
        // Movd loads Vn.s[0] / Vm.s[0] into dword 0 and zero-extends dwords
        // 1..3 — this is the equivalent of the Pxor + Pinsr scrub used for
        // B/H, but it's a single instruction.
        as_.Movd(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Movd(xm, {.base = Assembler::rbp, .disp = vm_off});
        if (is_sqadd_scalar) {
          // 32-bit signed SQADD: wrap-add + sign-bit-XOR-blend.
          // sat = (a < 0) ? INT_MIN : INT_MAX = (a >> 31 signed) ^ 0x7FFFFFFF.
          // ovf = ~(a^b) & (a^sum) has MSB set iff signed overflow.
          // result = sum ^ ((sat ^ sum) & ovf_mask).
          SimdRegister t_sum = AllocTempSimdReg();
          SimdRegister t_ovf = AllocTempSimdReg();
          SimdRegister t_sat = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_sum, xn);
          as_.Paddd(t_sum, xm);                        // t_sum = a + b (mod 2^32).
          as_.Movdqa(t_ovf, xn);
          as_.Pxor(t_ovf, xm);                         // t_ovf = a ^ b.
          as_.Movdqa(t_sat, xn);
          as_.Pxor(t_sat, t_sum);                      // t_sat = a ^ sum.
          as_.Pcmpeqd(xm, xm);                         // xm = -1 (reuse: b done).
          as_.Pxor(t_ovf, xm);                         // t_ovf = ~(a ^ b).
          as_.Pand(t_ovf, t_sat);                      // t_ovf = ~(a^b)&(a^sum).
          as_.Psrad(t_ovf, int8_t{31});                // t_ovf = overflow mask.
          as_.Psrad(xn, int8_t{31});                   // xn = -1 if a<0, 0 else.
          as_.Psrld(xm, int8_t{1});                    // xm = 0x7FFFFFFF.
          as_.Pxor(xn, xm);                            // xn = sat.
          as_.Pxor(xn, t_sum);                         // xn = sat ^ sum.
          as_.Pand(xn, t_ovf);                         // xn = (sat^sum)&ovf.
          as_.Pxor(xn, t_sum);                         // xn = sum ^ above.
        } else if (is_uqadd_scalar) {
          // 32-bit unsigned UQADD: wrap-add + PMAXUD overflow detect.
          // Overflow iff sum < a (unsigned) ↔ max(a,sum) != sum.
          SimdRegister t_save_a = AllocTempSimdReg();
          SimdRegister t_ones = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_save_a, xn);                    // preserve a.
          as_.Paddd(xn, xm);                           // xn = sum.
          as_.Pmaxud(t_save_a, xn);                    // t_save_a = max(a,sum).
          as_.Pcmpeqd(t_save_a, xn);                   // -1 where no overflow.
          as_.Pcmpeqd(t_ones, t_ones);                 // t_ones = -1.
          as_.Pxor(t_save_a, t_ones);                  // invert: -1 where overflow.
          as_.Por(xn, t_save_a);                       // saturate to UINT32_MAX.
        } else if (is_sqsub_scalar) {
          // 32-bit signed SQSUB: wrap-sub + sign-bit XOR-blend.
          // ovf = (a^b) & (a^diff) has MSB set iff signed overflow.
          SimdRegister t_diff = AllocTempSimdReg();
          SimdRegister t_ovf = AllocTempSimdReg();
          SimdRegister t_sat = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_diff, xn);
          as_.Psubd(t_diff, xm);                       // t_diff = a - b (mod 2^32).
          as_.Movdqa(t_ovf, xn);
          as_.Pxor(t_ovf, xm);                         // t_ovf = a ^ b.
          as_.Movdqa(t_sat, xn);
          as_.Pxor(t_sat, t_diff);                     // t_sat = a ^ diff.
          as_.Pand(t_ovf, t_sat);                      // t_ovf = (a^b)&(a^diff).
          as_.Psrad(t_ovf, int8_t{31});                // overflow mask.
          as_.Psrad(xn, int8_t{31});                   // xn = -1 if a<0, 0 else.
          as_.Pcmpeqd(xm, xm);                         // xm = -1.
          as_.Psrld(xm, int8_t{1});                    // xm = 0x7FFFFFFF.
          as_.Pxor(xn, xm);                            // xn = sat.
          as_.Pxor(xn, t_diff);                        // xn = sat ^ diff.
          as_.Pand(xn, t_ovf);                         // xn = (sat^diff)&ovf.
          as_.Pxor(xn, t_diff);                        // xn = diff ^ above.
        } else {
          // 32-bit unsigned UQSUB: result = (a >= b) ? a - b : 0.
          // Mask: PMINUD(a, b) == b iff a >= b.
          SimdRegister t_mask = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_mask, xn);
          as_.Pminud(t_mask, xm);                      // t_mask = min(a,b).
          as_.Pcmpeqd(t_mask, xm);                     // -1 where a >= b.
          as_.Psubd(xn, xm);                           // xn = a - b (wrap).
          as_.Pand(xn, t_mask);                        // zero underflow lanes.
        }
      } else {  // args.size == 0b11 (D form, 64-bit scalar)
        // Movq loads Vn[63:0]/Vm[63:0] into qword 0 and zero-extends qword 1.
        // All 64-bit recipes preserve qword 1 = 0 throughout (every operation
        // between two zero qwords is zero), so a full-width Movdqu store at
        // the end zeroes Vd[127:64] naturally.
        as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
        if (is_sqadd_scalar) {
          // 64-bit signed SQADD: wrap-add + sign-bit XOR-blend.
          // ovf = ~(a^b) & (a^sum) has MSB (bit 63) set iff signed overflow.
          // sat = (a < 0) ? INT64_MIN : INT64_MAX = sign_bcast(a) ^ INT64_MAX.
          // Broadcast bit 63 across the qword via PSRAD 31 + PSHUFD 0xF5
          // (PSRAQ is AVX-512F-VL only; this two-instruction sequence is
          // the SSE2-baseline equivalent — PSRAD 31 sign-extends each dword,
          // then PSHUFD 0xF5 fans dword 1's sign across qword 0 and dword
          // 3's sign across qword 1).
          SimdRegister t_sum = AllocTempSimdReg();
          SimdRegister t_ovf = AllocTempSimdReg();
          SimdRegister t_sat = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_sum, xn);
          as_.Paddq(t_sum, xm);                        // t_sum = a + b (mod 2^64).
          as_.Movdqa(t_ovf, xn);
          as_.Pxor(t_ovf, xm);                         // t_ovf = a ^ b.
          as_.Movdqa(t_sat, xn);
          as_.Pxor(t_sat, t_sum);                      // t_sat = a ^ sum.
          as_.Pcmpeqd(xm, xm);                         // xm = -1 (reuse: b done).
          as_.Pxor(t_ovf, xm);                         // t_ovf = ~(a ^ b).
          as_.Pand(t_ovf, t_sat);                      // t_ovf = ~(a^b)&(a^sum).
          as_.Psrad(t_ovf, int8_t{31});                // sign-extend each dword.
          as_.Pshufd(t_ovf, t_ovf, static_cast<int8_t>(0xF5));
                                                       // overflow mask broadcast
                                                       // (qword-wide).
          as_.Psrad(xn, int8_t{31});                   // sign-extend each dword.
          as_.Pshufd(xn, xn, static_cast<int8_t>(0xF5));
                                                       // xn qword = -1 if a<0,
                                                       // 0 else.
          as_.Psrlq(xm, int8_t{1});                    // xm = 0x7FFFFFFFFFFFFFFF.
          as_.Pxor(xn, xm);                            // xn = sat.
          as_.Pxor(xn, t_sum);                         // xn = sat ^ sum.
          as_.Pand(xn, t_ovf);                         // xn = (sat^sum)&ovf.
          as_.Pxor(xn, t_sum);                         // xn = sum ^ above.
        } else if (is_uqadd_scalar) {
          // 64-bit unsigned UQADD: wrap-add + PCMPGTQ overflow detect.
          // Overflow iff sum < a (unsigned) ↔ a > sum (unsigned).
          // Sign-flip both operands; signed PCMPGTQ on flipped pair implements
          // unsigned > comparison.  PMAXUQ is AVX-512F-VL only; PCMPGTQ is
          // SSE4.2 baseline (and gated above).
          SimdRegister t_save_a = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_save_a, xn);                    // preserve a.
          as_.Paddq(xn, xm);                           // xn = sum.
          as_.Pcmpeqd(xm, xm);                         // xm = -1 (reuse).
          as_.Psllq(xm, int8_t{63});                   // xm = 0x80...0 sign mask.
          as_.Pxor(t_save_a, xm);                      // t_save_a = a ^ sign.
          as_.Pxor(xm, xn);                            // xm = sum ^ sign
                                                       // (sign mask consumed).
          as_.Pcmpgtq(t_save_a, xm);                   // -1 where a > sum unsigned
                                                       // (== overflow).
          as_.Por(xn, t_save_a);                       // saturate to UINT64_MAX.
        } else if (is_sqsub_scalar) {
          // 64-bit signed SQSUB: wrap-sub + sign-bit XOR-blend.
          // ovf = (a^b) & (a^diff) has bit 63 set iff signed overflow.
          SimdRegister t_diff = AllocTempSimdReg();
          SimdRegister t_ovf = AllocTempSimdReg();
          SimdRegister t_sat = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_diff, xn);
          as_.Psubq(t_diff, xm);                       // t_diff = a - b (mod 2^64).
          as_.Movdqa(t_ovf, xn);
          as_.Pxor(t_ovf, xm);                         // t_ovf = a ^ b.
          as_.Movdqa(t_sat, xn);
          as_.Pxor(t_sat, t_diff);                     // t_sat = a ^ diff.
          as_.Pand(t_ovf, t_sat);                      // t_ovf = (a^b)&(a^diff).
          as_.Psrad(t_ovf, int8_t{31});
          as_.Pshufd(t_ovf, t_ovf, static_cast<int8_t>(0xF5));
                                                       // overflow mask broadcast.
          as_.Psrad(xn, int8_t{31});
          as_.Pshufd(xn, xn, static_cast<int8_t>(0xF5));
                                                       // xn = -1 if a<0, 0 else.
          as_.Pcmpeqd(xm, xm);                         // xm = -1.
          as_.Psrlq(xm, int8_t{1});                    // xm = 0x7FFFFFFFFFFFFFFF.
          as_.Pxor(xn, xm);                            // xn = sat.
          as_.Pxor(xn, t_diff);                        // xn = sat ^ diff.
          as_.Pand(xn, t_ovf);                         // xn = (sat^diff)&ovf.
          as_.Pxor(xn, t_diff);                        // xn = diff ^ above.
        } else {
          // 64-bit unsigned UQSUB: result = (a >= b) ? a - b : 0.
          // PMINUQ is AVX-512F-VL only.  Use sign-flip + PCMPGTQ to compare
          // unsigned 64-bit values.
          SimdRegister t_diff = AllocTempSimdReg();
          SimdRegister t_mask = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_diff, xn);
          as_.Psubq(t_diff, xm);                       // t_diff = a - b (mod 2^64).
          as_.Pcmpeqd(t_mask, t_mask);                 // t_mask = -1.
          as_.Psllq(t_mask, int8_t{63});               // t_mask = 0x80...0 sign mask.
          as_.Pxor(xn, t_mask);                        // xn = a ^ sign.
          as_.Pxor(t_mask, xm);                        // t_mask = b ^ sign
                                                       // (sign mask consumed).
          as_.Pcmpgtq(t_mask, xn);                     // -1 where (b^sign) >
                                                       // (a^sign) signed = b > a
                                                       // unsigned (underflow).
          as_.Pcmpeqd(xn, xn);                         // xn = -1 (reuse).
          as_.Pxor(t_mask, xn);                        // t_mask = -1 where a >= b
                                                       // (no underflow).
          as_.Pand(t_diff, t_mask);                    // zero underflow lane.
          as_.Movdqa(xn, t_diff);                      // xn = result.
        }
      }
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
      return;
    }
    // SSHL / USHL scalar D form (non-saturating variable
    // shift).  The shift amount comes from Vm[7:0] sign-extended to int64:
    //   * shift >= 64        → result = 0  (SSHL and USHL).
    //   * 0 <= shift < 64    → result = a << shift  (left, same for both).
    //   * -63 <= shift < 0   → SSHL: a >>s |shift|.  USHL: a >>u |shift|.
    //   * shift <= -64       → SSHL: sign(a) broadcast (a >>s 63).
    //                          USHL: 0.
    // Implemented in GPR scalar with branches: Movq Vn[63:0] → a; sign-extend
    // Vm[7:0] → shift; test sign + clamped magnitude branches.  RCX is saved
    // to the stack across the shift-by-cl operations per the existing
    // DataProc2Src kLslv/kLsrv/kAsrv pattern.
    if (is_shl_scalar_d) {
      if (args.size != 0b11) { success_ = false; return; }
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      Register a = AllocTempReg();
      Register sh = AllocTempReg();
      if (a == Assembler::no_register || sh == Assembler::no_register) {
        success_ = false; return;
      }
      // Save rcx (variable shift uses cl; rcx is in the allocator pool).
      as_.Subq(Assembler::rsp, 8);
      as_.Movq({.base = Assembler::rsp}, Assembler::rcx);
      // Load a from Vn[63:0]; sign-extend Vm[7:0] into sh (int64).
      as_.Movq(a, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_off});

      Assembler::Label* L_neg = as_.MakeLabel();
      Assembler::Label* L_zero = as_.MakeLabel();
      Assembler::Label* L_done = as_.MakeLabel();

      as_.Testq(sh, sh);
      as_.Jcc(Assembler::Condition::kSign, *L_neg);

      // Positive shift in [0, 127]: shift >= 64 → 0; else SHL a, sh.
      as_.Cmpq(sh, int32_t{64});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
      as_.Movq(Assembler::rcx, sh);
      as_.ShlqByCl(a);
      as_.Jmp(*L_done);

      as_.Bind(L_neg);
      // Negative path: sh becomes |shift| in [1, 128] after Negq.  Source was
      // sign-extension of int8 (range [-128, -1]) so the negation cannot
      // overflow.
      as_.Negq(sh);
      as_.Cmpq(sh, int32_t{64});
      if (is_sshl_scalar_d) {
        Assembler::Label* L_arith_max = as_.MakeLabel();
        as_.Jcc(Assembler::Condition::kGreaterEqual, *L_arith_max);
        // 0 < |sh| < 64: a = (int64_t)a >> |sh|.
        as_.Movq(Assembler::rcx, sh);
        as_.SarqByCl(a);
        as_.Jmp(*L_done);
        as_.Bind(L_arith_max);
        // |sh| >= 64: a = (int64_t)a >> 63 (sign-broadcast across all bits).
        as_.Sarq(a, int8_t{63});
        as_.Jmp(*L_done);
      } else {
        // USHL negative: |sh| >= 64 → 0; else SHR a, |sh|.
        as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
        as_.Movq(Assembler::rcx, sh);
        as_.ShrqByCl(a);
        as_.Jmp(*L_done);
      }

      as_.Bind(L_zero);
      as_.Xorq(a, a);

      as_.Bind(L_done);

      // Restore rcx.
      as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
      as_.Addq(Assembler::rsp, 8);

      // Store result to Vd[63:0] and zero Vd[127:64].
      as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a);
      as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, int32_t{0});
      return;
    }
    // UQSHL scalar (all sizes — unsigned saturating variable
    // left shift, lane width N = 8/16/32/64).  Like USHL scalar for the
    // right-shift branches; positive shifts add an overflow check that
    // saturates to umax_N = (1 << N) - 1 when bits get pushed past bit (N-1).
    // Algorithm (`a` is the unsigned operand, `sh` the signed int8 shift):
    //   * sh >= N           → result = (a == 0) ? 0 : umax_N.
    //   * 0 <= sh < N       → candidate = (a << sh) & umax_N;
    //                         if (candidate >>u sh) != a, saturate to umax_N
    //                         (some bits were lost above bit N-1).
    //   * -(N-1) <= sh < 0  → a >>u |sh|  (right-shifting a non-negative
    //                         unsigned value cannot overflow).
    //   * sh <= -N          → 0.
    // The mask-to-N step collapses to a no-op for N==64 because ShlqByCl
    // naturally truncates to 64 bits; for N<64 we mask with Andq imm32
    // (B/H) or Shlq+Shrq by (64-N) (S, since 0xFFFFFFFF as int32 would
    // sign-extend to -1 under Andq).
    if (is_uqshl_scalar_d) {
      const int bits_local = 1 << (3 + args.size);
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      Register a = AllocTempReg();
      Register sh = AllocTempReg();
      Register back = AllocTempReg();
      if (a == Assembler::no_register || sh == Assembler::no_register ||
          back == Assembler::no_register) {
        success_ = false; return;
      }
      // Save rcx (variable shift uses cl; rcx is in the allocator pool).
      as_.Subq(Assembler::rsp, 8);
      as_.Movq({.base = Assembler::rsp}, Assembler::rcx);
      // Load a from Vn[bits_local-1:0] zero-extended to 64; sign-extend
      // Vm[7:0] into sh (int64).  The width-specific load reads exactly
      // the source lane width and zero-extends the upper bits.
      switch (args.size) {
        case 0b00:
          as_.Movzxbq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b01:
          as_.Movzxwq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b10:
          // Movl on a 64-bit dst zero-extends to 64 bits.
          as_.Movl   (a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b11:
          as_.Movq   (a, {.base = Assembler::rbp, .disp = vn_off}); break;
      }
      as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_off});

      // Mask `r` to bits_local bits in-place.  No-op for bits_local==64.
      auto MaskToN = [&](Register r) {
        switch (bits_local) {
          case 8:  as_.Andq(r, int32_t{0xFF}); break;
          case 16: as_.Andq(r, int32_t{0xFFFF}); break;
          case 32:
            // Andq with int32_t{0xFFFFFFFF} = int32_t{-1} sign-extends to
            // -1 in 64-bit, which clobbers nothing — wrong.  Use the
            // 64-bit-truncation idiom Shlq 32 / Shrq 32 instead.
            as_.Shlq(r, int8_t{32});
            as_.Shrq(r, int8_t{32});
            break;
          case 64:
            break;  // 64-bit GPR ops naturally truncate.
        }
      };
      // Load umax_N = (1 << bits_local) - 1 into `r`.
      auto LoadUmax = [&](Register r) {
        if (bits_local == 64) {
          as_.Movq(r, static_cast<int64_t>(-1));  // UINT64_MAX.
        } else {
          // For bits_local in {8, 16}, (1<<N)-1 fits in int32 positive
          // (0xFF / 0xFFFF) and Movq int64 encodes as mov reg, imm32_sext.
          // For bits_local == 32, the constant is 0xFFFFFFFF which as int32
          // is -1 (sign-extends bad); int64{0xFFFFFFFFLL} is positive and
          // Movq emits the 10-byte movabsq.
          as_.Movq(r, static_cast<int64_t>((uint64_t{1} << bits_local) - 1));
        }
      };

      Assembler::Label* L_neg = as_.MakeLabel();
      Assembler::Label* L_zero = as_.MakeLabel();
      Assembler::Label* L_pos_big = as_.MakeLabel();
      Assembler::Label* L_done = as_.MakeLabel();

      as_.Testq(sh, sh);
      as_.Jcc(Assembler::Condition::kSign, *L_neg);

      // Positive shift in [0, 127]: sh >= bits_local falls through to the
      // a-vs-umax_N picker; otherwise the mask-then-back-shift overflow check.
      as_.Cmpq(sh, int32_t{bits_local});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

      // 0 <= sh < bits_local: shift-then-mask-then-back-shift overflow check.
      // Reuse `sh` as the saved-original holder once cl is loaded — the shift
      // count lives in rcx for the duration of the SHL/SHR pair, so sh is
      // free during that window.  The mask step is essential for bits_local
      // < 64 because the 64-bit ShlqByCl can place bits above position
      // (bits_local-1) that don't belong to the result; without masking,
      // the back-shift would compare against shifted-up bits and mis-detect
      // (no-)overflow.
      as_.Movq(Assembler::rcx, sh);
      as_.Movq(sh, a);                 // sh = original a (saved for cmp).
      as_.ShlqByCl(a);                 // a = a << shift (candidate result).
      MaskToN(a);
      as_.Movq(back, a);
      as_.ShrqByCl(back);              // back = ((a << sh) & umax_N) >>u sh.
      as_.Cmpq(back, sh);              // back == original a ?
      as_.Jcc(Assembler::Condition::kEqual, *L_done);
      // Overflow: bits were lost above bit (bits_local-1); saturate to umax_N.
      LoadUmax(a);
      as_.Jmp(*L_done);

      as_.Bind(L_pos_big);
      // sh >= bits_local: if a == 0 result is 0, else umax_N.
      as_.Testq(a, a);
      as_.Jcc(Assembler::Condition::kZero, *L_zero);
      LoadUmax(a);
      as_.Jmp(*L_done);

      as_.Bind(L_neg);
      // Negative shift: |sh| in [1, 128] after Negq.
      as_.Negq(sh);
      as_.Cmpq(sh, int32_t{bits_local});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
      // 0 < |sh| < bits_local: a >>u |sh|.  `a` is zero-extended so the
      // logical right shift cannot pull garbage in from above bit
      // (bits_local-1).
      as_.Movq(Assembler::rcx, sh);
      as_.ShrqByCl(a);
      as_.Jmp(*L_done);

      as_.Bind(L_zero);
      as_.Xorq(a, a);

      as_.Bind(L_done);
      // Restore rcx.
      as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
      as_.Addq(Assembler::rsp, 8);

      // Store result to Vd: low qword holds the (zero-extended) result;
      // Vd[127:64] is explicitly zeroed.  For bits_local<64 the high bits
      // of `a` are guaranteed zero by the masking / zero-extended right-
      // shift, so a full-width Movq into Vd[63:0] naturally puts the
      // result in Vd[bits_local-1:0] and zero in Vd[63:bits_local].
      as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a);
      as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, int32_t{0});
      return;
    }
    // SQSHL scalar (all sizes — signed saturating variable
    // left shift, lane width N = 8/16/32/64).  Mirrors UQSHL scalar but:
    //   * Operand `a` is sign-extended from Vn[bits_local-1:0] to int64 at
    //     load time, so bit 63 of `a` carries the sign of the N-bit lane and
    //     SAR-by-63 produces a width-independent sign broadcast.
    //   * For 0 <= sh < bits_local, the candidate `a << sh` must fit in
    //     int_N.  The overflow detector applies SignExtendFromN to the
    //     candidate (Shl/Sar by 64-N at B/H; Movsxlq at S as the single-
    //     instruction form; no-op at D) and then SAR-by-sh back to compare
    //     against the saved `a`.  At N=64 this reduces to the existing
    //     D-form detector `(int64_t)(a << sh) >>s sh == a`.
    //   * Saturation target depends on sign(a):
    //       sign(a) = 0 → INT_MAX_N = (1 << (bits_local - 1)) - 1.
    //       sign(a) = 1 → INT_MIN_N = -(1 << (bits_local - 1)).
    //     Computed by Sarq sign-broadcast (by 63 — width-independent because
    //     `a` was sign-extended at load) then XOR with INT_MAX_N.
    //   * Negative shifts use SAR (signed arithmetic right shift) instead of
    //     SHR, mirroring SSHL.  At |sh| >= bits_local the result is the sign
    //     broadcast (0 or -1), not 0.
    //   * The store path masks `a` to bits_local before writing to Vd[63:0]
    //     so the sign-extension bits don't leak into Vd[63:bits_local]
    //     (no-op at N=64).
    if (is_sqshl_scalar_d) {
      const int bits_local = 1 << (3 + args.size);
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      Register a = AllocTempReg();
      Register sh = AllocTempReg();
      Register back = AllocTempReg();
      if (a == Assembler::no_register || sh == Assembler::no_register ||
          back == Assembler::no_register) {
        success_ = false; return;
      }
      // Save rcx (variable shift uses cl; rcx is in the allocator pool).
      as_.Subq(Assembler::rsp, 8);
      as_.Movq({.base = Assembler::rsp}, Assembler::rcx);
      // Load `a` sign-extended from Vn[bits_local-1:0] into int64; sign-
      // extend Vm[7:0] into sh.  Bit 63 of `a` thus carries the sign of
      // the N-bit lane, making SAR-by-63 a width-independent sign broadcast.
      switch (args.size) {
        case 0b00:
          as_.Movsxbq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b01:
          as_.Movsxwq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b10:
          as_.Movsxlq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b11:
          as_.Movq   (a, {.base = Assembler::rbp, .disp = vn_off}); break;
      }
      as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_off});

      // Sign-extend the low bits_local bits of `r` to int64 in-place.  This
      // re-aligns the N-bit lane to the int64 envelope so the back-shift
      // via SAR produces a value comparable to the saved (sign-extended)
      // original.  At N=64 the 64-bit ShlqByCl already produced a value
      // whose bit 63 is the lane's MSB, so this is a no-op.
      auto SignExtendFromN = [&](Register r) {
        switch (bits_local) {
          case 8:  as_.Shlq(r, int8_t{56}); as_.Sarq(r, int8_t{56}); break;
          case 16: as_.Shlq(r, int8_t{48}); as_.Sarq(r, int8_t{48}); break;
          case 32: as_.Movsxlq(r, r); break;
          case 64: break;
        }
      };
      // Zero-extend low bits_local bits of `r` (mask to N) — used at the
      // store path to drop the upper sign-extension bits before writing
      // Vd[63:0].  No-op at N=64.  Andq with int32_t{0xFFFFFFFF} would
      // sign-extend to -1, so S uses the Shlq/Shrq 32 idiom.
      auto MaskToN = [&](Register r) {
        switch (bits_local) {
          case 8:  as_.Andq(r, int32_t{0xFF}); break;
          case 16: as_.Andq(r, int32_t{0xFFFF}); break;
          case 32:
            as_.Shlq(r, int8_t{32});
            as_.Shrq(r, int8_t{32});
            break;
          case 64: break;
        }
      };
      // Load INT_MAX_N = (1 << (bits_local - 1)) - 1 into `r`.  At N=64 the
      // constant is INT64_MAX and Movq emits movabsq; at N=32 it is
      // 0x7FFFFFFF (positive int32) and Movq emits the 7-byte form; at
      // N=8/16 the constant is small positive and Movq emits the short form.
      auto LoadIntMaxN = [&](Register r) {
        as_.Movq(r,
                 static_cast<int64_t>((uint64_t{1} << (bits_local - 1)) - 1));
      };

      Assembler::Label* L_neg = as_.MakeLabel();
      Assembler::Label* L_pos_big = as_.MakeLabel();
      Assembler::Label* L_neg_big = as_.MakeLabel();
      Assembler::Label* L_zero = as_.MakeLabel();
      Assembler::Label* L_done = as_.MakeLabel();

      as_.Testq(sh, sh);
      as_.Jcc(Assembler::Condition::kSign, *L_neg);

      // Positive shift in [0, 127]: sh >= bits_local falls through to the
      // sign-aware saturation picker; otherwise the sign-extend-then-back-
      // shift overflow check.
      as_.Cmpq(sh, int32_t{bits_local});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

      // 0 <= sh < bits_local: shift, sign-extend low N bits, then back-
      // shift via SAR.  Reuse `sh` as the saved-original holder once cl is
      // loaded.  The SignExtendFromN step is essential for bits_local < 64
      // because ShlqByCl can place bits above position (bits_local-1) that
      // belong to the int64-envelope but not the int_N lane; without
      // re-sign-extending from the lane's MSB, the back-SAR would compare
      // against the wrong value.
      as_.Movq(Assembler::rcx, sh);
      as_.Movq(sh, a);                 // sh = original a (saved for cmp).
      as_.ShlqByCl(a);                 // a = a << shift (candidate result).
      SignExtendFromN(a);              // a = sign_ext_N(low N of a<<shift).
      as_.Movq(back, a);
      as_.SarqByCl(back);              // back = sign-ext lane SAR shift.
      as_.Cmpq(back, sh);              // back == saved original ?
      as_.Jcc(Assembler::Condition::kEqual, *L_done);
      // Overflow: saturate to INT_MAX_N if sign(a)==0 else INT_MIN_N.
      // `sh` still holds the saved original a (sign-extended to int64), so
      // Sarq by 63 produces the width-independent sign broadcast.
      as_.Sarq(sh, int8_t{63});        // sh = sign broadcast (0 or -1).
      LoadIntMaxN(a);                  // a = INT_MAX_N.
      as_.Xorq(a, sh);                 // pos: INT_MAX_N; neg: INT_MIN_N sign-ext.
      as_.Jmp(*L_done);

      as_.Bind(L_pos_big);
      // sh >= bits_local: if a == 0 result is 0, else saturate based on
      // sign(a).
      as_.Testq(a, a);
      as_.Jcc(Assembler::Condition::kZero, *L_zero);
      as_.Movq(back, a);
      as_.Sarq(back, int8_t{63});       // back = sign broadcast (0 or -1).
      LoadIntMaxN(a);
      as_.Xorq(a, back);
      as_.Jmp(*L_done);

      as_.Bind(L_neg);
      // Negative shift: |sh| in [1, 128] after Negq.
      as_.Negq(sh);
      as_.Cmpq(sh, int32_t{bits_local});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_neg_big);
      // 0 < |sh| < bits_local: a = (int64_t)a >>s |sh|.  `a` is sign-
      // extended at load time so SAR preserves the proper sign-extension.
      as_.Movq(Assembler::rcx, sh);
      as_.SarqByCl(a);
      as_.Jmp(*L_done);

      as_.Bind(L_neg_big);
      // |sh| >= bits_local: a = sign broadcast across all bits.
      as_.Sarq(a, int8_t{63});
      as_.Jmp(*L_done);

      as_.Bind(L_zero);
      as_.Xorq(a, a);

      as_.Bind(L_done);
      // Restore rcx.
      as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
      as_.Addq(Assembler::rsp, 8);

      // Mask `a` to bits_local before storing.  At N=64 this is a no-op;
      // for bits_local < 64 it drops the sign-extension bits so Vd[63:N]
      // ends up as 0 instead of all 1s for negative results.
      MaskToN(a);

      // Store result to Vd[63:0] and zero Vd[127:64].
      as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a);
      as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, int32_t{0});
      return;
    }
    // URSHL scalar D form (unsigned non-saturating rounded
    // variable shift, 64-bit lane).  Left shifts truncate (no saturation —
    // bits past bit 63 are discarded); right shifts add a round-half-up bias
    // before the shift.  The standard identity
    //   (a + 2^(rshift-1)) >>u rshift  ==
    //   (a >>u rshift) + ((a >>u (rshift-1)) & 1)
    // avoids the bias-addition wraparound near UINT64_MAX.
    //   * shift in [0, 63]    → a <<u shift.
    //   * shift >= 64         → 0.
    //   * shift in [-63, -1]  → (a >>u |sh|) + ((a >>u (|sh|-1)) & 1).
    //   * shift == -64        → bit 63 of a (the round bit is the only
    //                            surviving contribution; equivalent to
    //                            (a >>u 63) & 1).  Handled with Shrq by
    //                            immediate 63 since (a >>u 64) is the
    //                            undefined-shift boundary on x86.
    //   * shift <= -65        → 0 (per ARM ARM semantics; the rounding
    //                            term dominates and the quotient is 0).
    if (is_urshl_scalar_d) {
      if (args.size != 0b11) { success_ = false; return; }
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      Register a = AllocTempReg();
      Register sh = AllocTempReg();
      Register round_bit = AllocTempReg();
      if (a == Assembler::no_register || sh == Assembler::no_register ||
          round_bit == Assembler::no_register) {
        success_ = false; return;
      }
      // Save rcx (variable shift uses cl; rcx is in the allocator pool).
      as_.Subq(Assembler::rsp, 8);
      as_.Movq({.base = Assembler::rsp}, Assembler::rcx);
      // Load a from Vn[63:0]; sign-extend Vm[7:0] into sh (int64).
      as_.Movq(a, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_off});

      Assembler::Label* L_neg = as_.MakeLabel();
      Assembler::Label* L_rshift_eq_64 = as_.MakeLabel();
      Assembler::Label* L_zero = as_.MakeLabel();
      Assembler::Label* L_done = as_.MakeLabel();

      as_.Testq(sh, sh);
      as_.Jcc(Assembler::Condition::kSign, *L_neg);

      // Positive shift in [0, 127]: sh >= 64 → 0; else a <<u sh.
      as_.Cmpq(sh, int32_t{64});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
      as_.Movq(Assembler::rcx, sh);
      as_.ShlqByCl(a);
      as_.Jmp(*L_done);

      as_.Bind(L_neg);
      // Negative shift: |sh| in [1, 128] after Negq.
      as_.Negq(sh);
      as_.Cmpq(sh, int32_t{64});
      as_.Jcc(Assembler::Condition::kGreater, *L_zero);
      as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_64);
      // 1 <= |sh| <= 63: compute round bit then signed quotient.
      // round_bit = (a >>u (|sh|-1)) & 1.
      as_.Movq(round_bit, a);
      as_.Decq(sh);
      as_.Movq(Assembler::rcx, sh);
      as_.ShrqByCl(round_bit);
      as_.Andq(round_bit, int32_t{1});
      as_.Incq(sh);
      as_.Movq(Assembler::rcx, sh);
      as_.ShrqByCl(a);
      as_.Addq(a, round_bit);
      as_.Jmp(*L_done);

      as_.Bind(L_rshift_eq_64);
      // |sh| == 64: result = (a >>u 63) & 1 — i.e. just bit 63.
      as_.Shrq(a, int8_t{63});
      as_.Jmp(*L_done);

      as_.Bind(L_zero);
      as_.Xorq(a, a);

      as_.Bind(L_done);
      // Restore rcx.
      as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
      as_.Addq(Assembler::rsp, 8);

      // Store result to Vd[63:0] and zero Vd[127:64].
      as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a);
      as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, int32_t{0});
      return;
    }
    // SRSHL scalar D form (signed non-saturating rounded
    // variable shift, 64-bit lane).  Direct mirror of URSHL D with two
    // changes: the data shift on the rounding-quadrant arm uses SAR (signed)
    // instead of SHR, and the |sh|>=64 boundary collapses into a single
    // zero branch (no dedicated |sh|=64 case).
    //   * shift in [0, 63]    → a <<u shift.
    //   * shift >= 64         → 0.
    //   * shift in [-63, -1]  → (a >>s |sh|) + ((a >>u (|sh|-1)) & 1).
    //   * shift <= -64        → 0 (for any int64 a, the signed bias 1<<63
    //                            sums with a to a non-negative int128 in
    //                            [0, 2^64-1] which >>s 64 = 0; for
    //                            |sh|>64 the rounding term dominates and
    //                            quotient is 0 per ARM ARM).
    if (is_srshl_scalar_d) {
      if (args.size != 0b11) { success_ = false; return; }
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      Register a = AllocTempReg();
      Register sh = AllocTempReg();
      Register round_bit = AllocTempReg();
      if (a == Assembler::no_register || sh == Assembler::no_register ||
          round_bit == Assembler::no_register) {
        success_ = false; return;
      }
      // Save rcx (variable shift uses cl; rcx is in the allocator pool).
      as_.Subq(Assembler::rsp, 8);
      as_.Movq({.base = Assembler::rsp}, Assembler::rcx);
      // Load a from Vn[63:0]; sign-extend Vm[7:0] into sh (int64).
      as_.Movq(a, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_off});

      Assembler::Label* L_neg = as_.MakeLabel();
      Assembler::Label* L_zero = as_.MakeLabel();
      Assembler::Label* L_done = as_.MakeLabel();

      as_.Testq(sh, sh);
      as_.Jcc(Assembler::Condition::kSign, *L_neg);

      // Positive shift in [0, 127]: sh >= 64 → 0; else a <<u sh.
      as_.Cmpq(sh, int32_t{64});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
      as_.Movq(Assembler::rcx, sh);
      as_.ShlqByCl(a);
      as_.Jmp(*L_done);

      as_.Bind(L_neg);
      // Negative shift: |sh| in [1, 128] after Negq.
      as_.Negq(sh);
      as_.Cmpq(sh, int32_t{64});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
      // 1 <= |sh| <= 63: compute round bit then signed quotient.
      // round_bit = (a >>u (|sh|-1)) & 1.
      as_.Movq(round_bit, a);
      as_.Decq(sh);
      as_.Movq(Assembler::rcx, sh);
      as_.ShrqByCl(round_bit);
      as_.Andq(round_bit, int32_t{1});
      as_.Incq(sh);
      as_.Movq(Assembler::rcx, sh);
      as_.SarqByCl(a);
      as_.Addq(a, round_bit);
      as_.Jmp(*L_done);

      as_.Bind(L_zero);
      as_.Xorq(a, a);

      as_.Bind(L_done);
      // Restore rcx.
      as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
      as_.Addq(Assembler::rsp, 8);

      // Store result to Vd[63:0] and zero Vd[127:64].
      as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a);
      as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, int32_t{0});
      return;
    }
    // UQRSHL scalar (all sizes — unsigned saturating
    // rounded variable shift, lane width N = 8/16/32/64).  Combines UQSHL
    // scalar positive arm with URSHL D negative arm — saturation only on
    // the left-shift quadrant, rounding only on the right-shift quadrant,
    // so the two scaffoldings compose without interaction:
    //   * shift in [0, N-1]   → shift-then-mask-then-back-shift overflow
    //                            detector; saturate to umax_N on overflow.
    //   * shift >= N          → umax_N if a != 0 else 0.
    //   * shift in [-(N-1), -1]→ (a >>u |sh|) + ((a >>u (|sh|-1)) & 1).
    //   * shift == -N         → bit (N-1) of a (the round bit is the only
    //                            surviving contribution; the data shift
    //                            by N would land outside the int_N lane
    //                            and contributes 0).
    //   * shift <= -(N+1)     → 0 (rounding term dominates per ARM ARM).
    // `a` is zero-extended from Vn[N-1:0] at load time so bits above
    // position N-1 are 0; the MaskToN step on the positive arm clears the
    // candidate's bits above position N-1 so the back-shift detector
    // compares against the int_N envelope and not the int64 envelope.
    // For bits_local==64 the MaskToN, LoadUmax, and Shrq-by-(N-1)
    // operations collapse to no-ops / the existing D-form constants.
    if (is_uqrshl_scalar_d) {
      const int bits_local = 1 << (3 + args.size);
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      Register a = AllocTempReg();
      Register sh = AllocTempReg();
      // `scratch` doubles as the back-shift holder on the positive arm
      // (UQSHL recipe) and the round_bit holder on the negative arm
      // (URSHL recipe) — these quadrants are disjoint.
      Register scratch = AllocTempReg();
      if (a == Assembler::no_register || sh == Assembler::no_register ||
          scratch == Assembler::no_register) {
        success_ = false; return;
      }
      // Save rcx (variable shift uses cl; rcx is in the allocator pool).
      as_.Subq(Assembler::rsp, 8);
      as_.Movq({.base = Assembler::rsp}, Assembler::rcx);
      // Load a from Vn[bits_local-1:0] zero-extended to 64; sign-extend
      // Vm[7:0] into sh (int64).  The width-specific load reads exactly
      // the source lane width and zero-extends the upper bits.
      switch (args.size) {
        case 0b00:
          as_.Movzxbq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b01:
          as_.Movzxwq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b10:
          // Movl on a 64-bit dst zero-extends to 64 bits.
          as_.Movl   (a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b11:
          as_.Movq   (a, {.base = Assembler::rbp, .disp = vn_off}); break;
      }
      as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_off});

      // Mask `r` to bits_local bits in-place.  No-op for bits_local==64.
      auto MaskToN = [&](Register r) {
        switch (bits_local) {
          case 8:  as_.Andq(r, int32_t{0xFF}); break;
          case 16: as_.Andq(r, int32_t{0xFFFF}); break;
          case 32:
            // Andq with int32_t{0xFFFFFFFF} = int32_t{-1} sign-extends to
            // -1 in 64-bit, which clobbers nothing — wrong.  Use the
            // 64-bit-truncation idiom Shlq 32 / Shrq 32 instead.
            as_.Shlq(r, int8_t{32});
            as_.Shrq(r, int8_t{32});
            break;
          case 64:
            break;  // 64-bit GPR ops naturally truncate.
        }
      };
      // Load umax_N = (1 << bits_local) - 1 into `r`.
      auto LoadUmax = [&](Register r) {
        if (bits_local == 64) {
          as_.Movq(r, static_cast<int64_t>(-1));  // UINT64_MAX.
        } else {
          as_.Movq(r, static_cast<int64_t>((uint64_t{1} << bits_local) - 1));
        }
      };

      Assembler::Label* L_neg = as_.MakeLabel();
      Assembler::Label* L_pos_big = as_.MakeLabel();
      Assembler::Label* L_rshift_eq_N = as_.MakeLabel();
      Assembler::Label* L_zero = as_.MakeLabel();
      Assembler::Label* L_done = as_.MakeLabel();

      as_.Testq(sh, sh);
      as_.Jcc(Assembler::Condition::kSign, *L_neg);

      // Positive shift in [0, 127]: sh >= bits_local falls through to the
      // a-vs-umax_N picker; otherwise the back-shift overflow check.
      as_.Cmpq(sh, int32_t{bits_local});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

      // 0 <= sh < bits_local: shift-then-mask-then-back-shift overflow check.
      as_.Movq(Assembler::rcx, sh);
      as_.Movq(sh, a);                 // sh = original a (saved for cmp).
      as_.ShlqByCl(a);                 // a = a << shift (candidate result).
      MaskToN(a);
      as_.Movq(scratch, a);
      as_.ShrqByCl(scratch);           // scratch = ((a << sh) & umax_N) >>u sh.
      as_.Cmpq(scratch, sh);           // scratch == original a ?
      as_.Jcc(Assembler::Condition::kEqual, *L_done);
      // Overflow: bits were lost above bit (bits_local-1); saturate to umax_N.
      LoadUmax(a);
      as_.Jmp(*L_done);

      as_.Bind(L_pos_big);
      // sh >= bits_local: if a == 0 result is 0, else umax_N.
      as_.Testq(a, a);
      as_.Jcc(Assembler::Condition::kZero, *L_zero);
      LoadUmax(a);
      as_.Jmp(*L_done);

      as_.Bind(L_neg);
      // Negative shift: |sh| in [1, 128] after Negq.
      as_.Negq(sh);
      as_.Cmpq(sh, int32_t{bits_local});
      as_.Jcc(Assembler::Condition::kGreater, *L_zero);
      as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_N);
      // 1 <= |sh| < bits_local: overflow-safe rounding identity.
      // round_bit = (a >>u (|sh|-1)) & 1.  `a` is zero-extended so ShrqByCl
      // cannot pull garbage in from above bit (bits_local-1).
      as_.Movq(scratch, a);
      as_.Decq(sh);
      as_.Movq(Assembler::rcx, sh);
      as_.ShrqByCl(scratch);
      as_.Andq(scratch, int32_t{1});
      as_.Incq(sh);
      as_.Movq(Assembler::rcx, sh);
      as_.ShrqByCl(a);
      as_.Addq(a, scratch);
      as_.Jmp(*L_done);

      as_.Bind(L_rshift_eq_N);
      // |sh| == bits_local: result = (a >>u (bits_local-1)) — just bit
      // (bits_local-1), the round bit.  Bits above (bits_local-1) of `a`
      // are guaranteed 0 by the zero-extended load, so this single Shrq
      // extracts exactly one bit.
      as_.Shrq(a, static_cast<int8_t>(bits_local - 1));
      as_.Jmp(*L_done);

      as_.Bind(L_zero);
      as_.Xorq(a, a);

      as_.Bind(L_done);
      // Restore rcx.
      as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
      as_.Addq(Assembler::rsp, 8);

      // Store result to Vd: low qword holds the (zero-extended) result;
      // Vd[127:64] is explicitly zeroed.  For bits_local<64 the high bits
      // of `a` are guaranteed zero by the masking / zero-extended right-
      // shift, so a full-width Movq into Vd[63:0] naturally puts the
      // result in Vd[bits_local-1:0] and zero in Vd[63:bits_local].
      as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a);
      as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, int32_t{0});
      return;
    }
    // SQRSHL scalar (all sizes — signed saturating rounded
    // variable shift, lane width N = 8/16/32/64).  Combines SQSHL B/H/S/D
    // positive arm with SRSHL D negative arm (parameterized on bits_local):
    //   * shift in [0, N-1]   → sign-extend + SAR-back-shift overflow
    //                            detector with SignExtendFromN between
    //                            Shl/Sar; saturate to INT_MAX_N (positive
    //                            a) or INT_MIN_N (negative a) via sign-
    //                            broadcast XOR-with-INT_MAX_N on overflow.
    //   * shift >= N          → saturated INT_MAX_N/INT_MIN_N if a != 0
    //                            else 0.
    //   * shift in [-(N-1), -1]→ (a >>s |sh|) + ((a >>u (|sh|-1)) & 1).
    //                            For sign-extended `a` and |sh|-1 in
    //                            [0, N-2], the Shrq by (|sh|-1) places the
    //                            lane's original bit (|sh|-1) at position
    //                            0 (no sign-extension bits are read);
    //                            SAR by |sh| produces the correctly-sign-
    //                            extended int_N right-shift; the MaskToN
    //                            at store drops the high sign bits.
    //   * shift <= -N         → 0.  ARM ARM (a + 2^(|sh|-1)) >>s |sh|: at
    //                            |sh|=N the bias maps int_N a to a non-
    //                            negative int_{>N} in [0, 2^N-1] which
    //                            >>s N = 0; for |sh|>N the bias dominates
    //                            and the result remains 0.
    // For bits_local==64 the SignExtendFromN, MaskToN, and LoadIntMaxN
    // helpers collapse to no-ops / the existing D-form constants.
    if (is_sqrshl_scalar_d) {
      const int bits_local = 1 << (3 + args.size);
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      Register a = AllocTempReg();
      Register sh = AllocTempReg();
      // `scratch` doubles as the back-shift holder on the positive arm
      // (SQSHL recipe) and the round_bit holder on the negative arm
      // (SRSHL D recipe) — these quadrants are disjoint.
      Register scratch = AllocTempReg();
      if (a == Assembler::no_register || sh == Assembler::no_register ||
          scratch == Assembler::no_register) {
        success_ = false; return;
      }
      // Save rcx (variable shift uses cl; rcx is in the allocator pool).
      as_.Subq(Assembler::rsp, 8);
      as_.Movq({.base = Assembler::rsp}, Assembler::rcx);
      // Load `a` sign-extended from Vn[bits_local-1:0] into int64; sign-
      // extend Vm[7:0] into sh.  Bit 63 of `a` thus carries the sign of
      // the N-bit lane, making SAR-by-63 a width-independent sign broadcast
      // and making the negative-arm SAR-by-|sh| produce the correct int_N
      // right-shift.
      switch (args.size) {
        case 0b00:
          as_.Movsxbq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b01:
          as_.Movsxwq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b10:
          as_.Movsxlq(a, {.base = Assembler::rbp, .disp = vn_off}); break;
        case 0b11:
          as_.Movq   (a, {.base = Assembler::rbp, .disp = vn_off}); break;
      }
      as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_off});

      // Re-sign-extend low N bits of `r` to int64.  Same recipe as the
      // SQSHL B/H/S positive arm: pulls the lane's MSB at position
      // (bits_local-1) back to bit 63 so back-SAR compares against the
      // int_N envelope, not the int64 envelope.  No-op at N=64.
      auto SignExtendFromN = [&](Register r) {
        switch (bits_local) {
          case 8:  as_.Shlq(r, int8_t{56}); as_.Sarq(r, int8_t{56}); break;
          case 16: as_.Shlq(r, int8_t{48}); as_.Sarq(r, int8_t{48}); break;
          case 32: as_.Movsxlq(r, r); break;
          case 64: break;
        }
      };
      // Mask low N bits at the store path — drops the sign-extension bits
      // so Vd[63:N] ends up as 0 instead of all 1s for negative results.
      // No-op at N=64.  Andq with int32_t{0xFFFFFFFF} sign-extends to -1,
      // so S uses the Shlq/Shrq 32 idiom.
      auto MaskToN = [&](Register r) {
        switch (bits_local) {
          case 8:  as_.Andq(r, int32_t{0xFF}); break;
          case 16: as_.Andq(r, int32_t{0xFFFF}); break;
          case 32:
            as_.Shlq(r, int8_t{32});
            as_.Shrq(r, int8_t{32});
            break;
          case 64: break;
        }
      };
      // Load INT_MAX_N = (1 << (bits_local - 1)) - 1.
      auto LoadIntMaxN = [&](Register r) {
        as_.Movq(r,
                 static_cast<int64_t>((uint64_t{1} << (bits_local - 1)) - 1));
      };

      Assembler::Label* L_neg = as_.MakeLabel();
      Assembler::Label* L_pos_big = as_.MakeLabel();
      Assembler::Label* L_zero = as_.MakeLabel();
      Assembler::Label* L_done = as_.MakeLabel();

      as_.Testq(sh, sh);
      as_.Jcc(Assembler::Condition::kSign, *L_neg);

      // Positive shift in [0, 127]: sh >= bits_local falls through to the
      // sign-aware saturation picker; otherwise the sign-extend-then-back-
      // shift overflow check.
      as_.Cmpq(sh, int32_t{bits_local});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

      // 0 <= sh < bits_local: ShlqByCl, re-sign-extend low N bits, then
      // back-SAR.  Reuse `sh` as the saved-original holder once cl is
      // loaded.  Without SignExtendFromN at sub-D widths, ShlqByCl can
      // place bits above position (bits_local-1) that belong to the int64
      // envelope but not the int_N lane; the re-sign-extension makes the
      // back-SAR's comparison correct.
      as_.Movq(Assembler::rcx, sh);
      as_.Movq(sh, a);                 // sh = saved original a (sign-extended).
      as_.ShlqByCl(a);                 // a = a << shift (candidate result).
      SignExtendFromN(a);              // a = sign_ext_N(low N of a<<shift).
      as_.Movq(scratch, a);
      as_.SarqByCl(scratch);           // scratch = sign-ext lane SAR shift.
      as_.Cmpq(scratch, sh);           // scratch == saved original ?
      as_.Jcc(Assembler::Condition::kEqual, *L_done);
      // Overflow: saturate to INT_MAX_N if sign(a)==0 else INT_MIN_N.
      // `sh` still holds the saved original a (sign-extended to int64), so
      // Sarq by 63 produces the width-independent sign broadcast.
      as_.Sarq(sh, int8_t{63});        // sh = sign broadcast (0 or -1).
      LoadIntMaxN(a);                  // a = INT_MAX_N.
      as_.Xorq(a, sh);                 // pos: INT_MAX_N; neg: INT_MIN_N sign-ext.
      as_.Jmp(*L_done);

      as_.Bind(L_pos_big);
      // sh >= bits_local: if a == 0 result is 0, else saturate based on
      // sign(a).
      as_.Testq(a, a);
      as_.Jcc(Assembler::Condition::kZero, *L_zero);
      as_.Movq(scratch, a);
      as_.Sarq(scratch, int8_t{63});   // scratch = sign broadcast (0 or -1).
      LoadIntMaxN(a);
      as_.Xorq(a, scratch);
      as_.Jmp(*L_done);

      as_.Bind(L_neg);
      // Negative shift: |sh| in [1, 128] after Negq.
      as_.Negq(sh);
      as_.Cmpq(sh, int32_t{bits_local});
      as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
      // 1 <= |sh| <= bits_local-1: overflow-safe rounding identity.
      // round_bit = (a >>u (|sh|-1)) & 1; data shift uses SAR.  For sign-
      // extended `a` and |sh|-1 in [0, bits_local-2], Shrq by (|sh|-1)
      // places the lane's bit (|sh|-1) (always below the sign-extension
      // region) at position 0.  Sarq by |sh| on sign-extended `a`
      // produces the correctly-sign-extended int_N right-shift; the
      // MaskToN at the store path drops the high sign bits.
      as_.Movq(scratch, a);
      as_.Decq(sh);
      as_.Movq(Assembler::rcx, sh);
      as_.ShrqByCl(scratch);
      as_.Andq(scratch, int32_t{1});
      as_.Incq(sh);
      as_.Movq(Assembler::rcx, sh);
      as_.SarqByCl(a);
      as_.Addq(a, scratch);
      as_.Jmp(*L_done);

      as_.Bind(L_zero);
      as_.Xorq(a, a);

      as_.Bind(L_done);
      // Restore rcx.
      as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
      as_.Addq(Assembler::rsp, 8);

      // Mask `a` to bits_local before storing.  At N=64 this is a no-op;
      // for bits_local < 64 it drops the sign-extension bits so Vd[63:N]
      // ends up as 0 instead of all 1s for negative results.
      MaskToN(a);

      // Store result to Vd[63:0] and zero Vd[127:64].
      as_.Movq({.base = Assembler::rbp, .disp = vd_off}, a);
      as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, int32_t{0});
      return;
    }
    // D-form scalar integer ops (ADD / SUB / CMGT / CMHI /
    //   CMGE / CMHS / CMTST / CMEQ scalar).  The ARM ARM restricts these to
    //   size=11 (D form); Vd[63:0] = scalar op result, Vd[127:64] = 0.
    //
    // SSE port mirrors the AdvSimdThreeSame vector lowerings at size=11
    // earlier in this file (CMGT, CMHI, CMGE, CMHS, CMEQ, ADD, SUB).
    // Operands are loaded with MOVQ which puts Vn[63:0] in xmm[63:0] and
    // zeros xmm[127:64], so the lane-1 computation is always between two
    // zero operands.  We still MaskLow64 the result because PCMPEQQ and
    // the NOT-of-PCMPGTQ produce all-ones in lane 1 (0==0 / !(0>0)).
    //
    // SSE requirements:
    //   * CMEQ scalar D       → PCMPEQQ (SSE4.1)
    //   * CMTST scalar D      → PAND + PCMPEQQ (SSE4.1)
    //   * CMGT scalar D       → PCMPGTQ (SSE4.2)
    //   * CMHI scalar D       → sign-flip + PCMPGTQ (SSE4.2)
    //   * CMGE scalar D       → PCMPGTQ + NOT (SSE4.2)
    //   * CMHS scalar D       → sign-flip + PCMPGTQ + NOT (SSE4.2)
    //   * ADD / SUB scalar D  → PADDQ / PSUBQ (baseline SSE2)
    if (is_dform_int) {
      if (args.size != 0b11) { success_ = false; return; }
      const bool needs_sse4_1 = is_dform_cmeq || is_dform_cmtst;
      const bool needs_sse4_2 =
          is_dform_cmgt || is_dform_cmhi || is_dform_cmge || is_dform_cmhs;
      if (needs_sse4_2 && !host_platform::kHasSSE4_2) {
        success_ = false; return;
      }
      if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
        success_ = false; return;
      }
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);
      SimdRegister xn = AllocTempSimdReg();
      SimdRegister xm = AllocTempSimdReg();
      if (!success()) { return; }
      // Load Vn[63:0] / Vm[63:0] with upper-lane zero-extension.
      as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});

      if (is_dform_add) {
        as_.Paddq(xn, xm);
      } else if (is_dform_sub) {
        as_.Psubq(xn, xm);
      } else if (is_dform_cmgt) {
        as_.Pcmpgtq(xn, xm);
      } else if (is_dform_cmhi) {
        // Sign-flip both operands, then signed PCMPGTQ implements unsigned >.
        SimdRegister sign = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pcmpeqd(sign, sign);
        as_.Psllq(sign, int8_t{63});
        as_.Pxor(xn, sign);
        as_.Pxor(xm, sign);
        as_.Pcmpgtq(xn, xm);
      } else if (is_dform_cmge) {
        // CMGE Vn,Vm == !(Vm > Vn).  Reuse xm as the destination of
        // PCMPGTQ(xm,xn), then XOR with all-ones to invert.
        SimdRegister ones = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pcmpgtq(xm, xn);
        as_.Pcmpeqd(ones, ones);
        as_.Pxor(xm, ones);
        xn = xm;  // Result lives in xm; rewire xn so the store path is shared.
      } else if (is_dform_cmhs) {
        // Sign-flip + !(xm > xn) gives unsigned >=.
        SimdRegister sign = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pcmpeqd(sign, sign);
        as_.Psllq(sign, int8_t{63});
        as_.Pxor(xn, sign);
        as_.Pxor(xm, sign);
        as_.Pcmpgtq(xm, xn);
        // Reuse `sign` to build the all-ones invert mask.
        as_.Pcmpeqd(sign, sign);
        as_.Pxor(xm, sign);
        xn = xm;
      } else if (is_dform_cmtst) {
        // (Vn & Vm) != 0 ? all-ones : 0.  Compute AND, compare with 0
        // (PCMPEQQ against a zeroed register), invert.
        SimdRegister zero = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pand(xn, xm);
        as_.Pxor(zero, zero);
        as_.Pcmpeqq(xn, zero);  // xn = (xn & xm) == 0 ? all-ones : 0
        // Invert: all-ones for nonzero AND, 0 for zero AND.
        as_.Pcmpeqd(zero, zero);  // Reuse `zero` as the all-ones mask.
        as_.Pxor(xn, zero);
      } else if (is_dform_cmeq) {
        as_.Pcmpeqq(xn, xm);
      }
      // Zero the upper 64 bits of the destination — scalar D form writes
      // Vd[127:64] = 0, but PCMPEQQ and the inverted-PCMPGTQ paths leave
      // 0xFFFF... in lane 1.
      as_.Pslldq(xn, int8_t{8});
      as_.Psrldq(xn, int8_t{8});
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
      return;
    }
    // FP16 path is JIT-emitted for FCMxx / FACxx (mask out),
    // FMULX (real-FP out), FABD (real-FP out), FRECPS (real-FP out), and
    // FRSQRTS (real-FP out) via an F16C round-trip (each FP16 source lane is
    // lifted to FP32 in xmm lane 0; the existing FP32 core runs unchanged;
    // result narrowed back to FP16).  Mask-producing ops narrow with Packssdw
    // (against a zero scratch); real-FP ops narrow with Vcvtps2ph (which also
    // auto-zeroes the upper 64 bits of the dst xmm).
    if (args.is_fp16 && !host_platform::kHasF16C) {
      success_ = false; return;
    }
    if (args.is_fp16 && !is_cmp && !is_fmulx && !is_fabd && !is_frecps &&
        !is_frsqrts) {
      success_ = false; return;
    }
    if ((is_frecps || is_frsqrts) && !host_platform::kHasFMA) {
      success_ = false; return;
    }

    const bool is_double = (args.size != 0);  // FP: 0 -> S, 1 -> D
    int32_t src_n_off = VRegOffset(args.rn);
    int32_t src_m_off = VRegOffset(args.rm);
    int32_t dst_off = VRegOffset(args.rd);

    if (is_fabd) {
      SimdRegister xmm_a = AllocTempSimdReg();
      SimdRegister xmm_b = AllocTempSimdReg();
      SimdRegister xmm_mask = AllocTempSimdReg();
      if (!success()) { return; }
      // FP16 lift via F16C round-trip — Pxor + Pinsrw +
      // Vcvtph2ps widens each FP16 source lane to FP32 in xmm lane 0.  The
      // existing FP32 FABD core (Subss + AND with non-sign-bit mask) then
      // runs unchanged on the lifted operands; lanes 1..3 stay FP32 +0 by
      // Vcvtph2ps zero-fill, and AND with the (also-FP32) non-sign mask
      // preserves that upper-zero invariant.
      const bool use_single = args.is_fp16 || !is_double;
      if (args.is_fp16) {
        as_.Pxor(xmm_a, xmm_a);
        as_.Pinsrw(xmm_a, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_a, xmm_a);
        as_.Pxor(xmm_b, xmm_b);
        as_.Pinsrw(xmm_b, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_b, xmm_b);
        as_.Subss(xmm_a, xmm_b);
      } else if (is_double) {
        as_.Movsd(xmm_a, {.base = Assembler::rbp, .disp = src_n_off});
        as_.Movsd(xmm_b, {.base = Assembler::rbp, .disp = src_m_off});
        as_.Subsd(xmm_a, xmm_b);
      } else {
        as_.Movss(xmm_a, {.base = Assembler::rbp, .disp = src_n_off});
        as_.Movss(xmm_b, {.base = Assembler::rbp, .disp = src_m_off});
        as_.Subss(xmm_a, xmm_b);
      }
      // Build non-sign-bit mask (all-ones shifted right by 1) and AND it
      // into the lane to clear the sign bit (== std::fabs).
      as_.Pcmpeqd(xmm_mask, xmm_mask);
      if (use_single) as_.Psrld(xmm_mask, int8_t{1});
      else as_.Psrlq(xmm_mask, int8_t{1});
      as_.Pand(xmm_a, xmm_mask);
      // FP16 narrow via Vcvtps2ph (real-FP output, NOT a
      // mask — Vcvtps2ph is the right primitive here).  It rounds 4 FP32
      // lanes to 4 FP16 lanes in the dst xmm's low 64 bits and zero-fills
      // the upper 64 bits.  Lanes 1..3 are FP32 +0.0 (preserved through
      // the Subss + Pand pipeline), so narrowed word lanes 1..3 are 0x0000;
      // word lanes 4..7 are 0 (auto-zero).  Full 128-bit Movdqu then writes
      // Vd[15:0] = FP16 result with Vd[127:16] = 0.
      if (args.is_fp16) {
        as_.Vcvtps2ph(xmm_a, xmm_a, int8_t{0});
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
      } else {
        // Zero Vd above the lane, then write the scalar result at lane 0.
        as_.Pxor(xmm_mask, xmm_mask);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_mask);
        if (is_double) {
          as_.Movsd({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
        } else {
          as_.Movss({.base = Assembler::rbp, .disp = dst_off}, xmm_a);
        }
      }
      return;
    }

    if (is_fcmeq || is_fcmge || is_fcmgt || is_facge || is_facgt) {
      SimdRegister xmm_a = AllocTempSimdReg();
      SimdRegister xmm_b = AllocTempSimdReg();
      if (!success()) { return; }
      // FP16-via-F16C runs at FP32 width: the FP32 compare core is bit-exact
      // for the FP16-widened operands because Vcvtph2ps preserves NaN-ness
      // and ±0 sign, and the FP16 sign bit becomes the FP32 sign bit (the
      // FACxx fabs mask works at FP32 width).
      const bool use_single = args.is_fp16 || !is_double;
      if (args.is_fp16) {
        // Lift each FP16 source lane to FP32 in xmm lane 0.  Pxor zeroes
        // bits[127:0]; Pinsrw inserts Vn[15:0] / Vm[15:0] at xmm[15:0]
        // (other FP16 lanes stay zero); Vcvtph2ps widens 4 FP16 lanes to
        // 4 FP32 lanes — only lane 0 carries a real value, lanes 1..3 are
        // FP32 +0.0 by construction.
        as_.Pxor(xmm_a, xmm_a);
        as_.Pinsrw(xmm_a, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_a, xmm_a);
        as_.Pxor(xmm_b, xmm_b);
        as_.Pinsrw(xmm_b, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_b, xmm_b);
      } else if (is_double) {
        as_.Movsd(xmm_a, {.base = Assembler::rbp, .disp = src_n_off});
        as_.Movsd(xmm_b, {.base = Assembler::rbp, .disp = src_m_off});
      } else {
        as_.Movss(xmm_a, {.base = Assembler::rbp, .disp = src_n_off});
        as_.Movss(xmm_b, {.base = Assembler::rbp, .disp = src_m_off});
      }
      // FACGE / FACGT: clear sign bits of both operands (== std::fabs).
      if (is_facge || is_facgt) {
        SimdRegister xmm_mask = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pcmpeqd(xmm_mask, xmm_mask);
        if (use_single) as_.Psrld(xmm_mask, int8_t{1});
        else as_.Psrlq(xmm_mask, int8_t{1});
        as_.Pand(xmm_a, xmm_mask);
        as_.Pand(xmm_b, xmm_mask);
      }
      // Apply comparison.  CMPSS/CMPSD has dst := (dst cmp src):
      //   FCMEQ a, b: xmm_a := (a == b)     — predicate EQ_OQ.
      //   FCMGE a, b (a >= b == b <= a):  xmm_b := (b <= a) — LE_OS, swapped.
      //   FCMGT a, b (a > b == b < a):    xmm_b := (b <  a) — LT_OS, swapped.
      //   FACGE/FACGT — same as FCMGE/FCMGT after both operands are |·|.
      SimdRegister xmm_result = no_simd_register;
      if (is_fcmeq) {
        if (use_single) as_.Cmpeqss(xmm_a, xmm_b);
        else as_.Cmpeqsd(xmm_a, xmm_b);
        xmm_result = xmm_a;
      } else if (is_fcmge || is_facge) {
        if (use_single) as_.Cmpless(xmm_b, xmm_a);
        else as_.Cmplesd(xmm_b, xmm_a);
        xmm_result = xmm_b;
      } else {  // is_fcmgt || is_facgt
        if (use_single) as_.Cmpltss(xmm_b, xmm_a);
        else as_.Cmpltsd(xmm_b, xmm_a);
        xmm_result = xmm_b;
      }
      // Zero Vd above the lane, then write the scalar mask at lane 0.
      // Reuse whichever of xmm_a / xmm_b is not the result as the zero scratch.
      SimdRegister xmm_zero = (xmm_result == xmm_a) ? xmm_b : xmm_a;
      as_.Pxor(xmm_zero, xmm_zero);
      if (args.is_fp16) {
        // Pack the FP32 mask (lane 0 dword: 0xFFFFFFFF or 0x00000000) to a
        // 16-bit mask (lane 0 word: 0xFFFF or 0x0000) via signed-saturating
        // dword->word pack.  As signed int32: 0xFFFFFFFF == -1 saturates to
        // -1 == 0xFFFF as int16; 0x00000000 saturates to 0.  Dword lanes
        // 1..3 of xmm_result are 0 by construction (Vcvtph2ps zero-fill
        // plus AND/scalar-cmp preserving the upper-zero invariant), so
        // packing against xmm_zero (=0) leaves word lanes 1..7 == 0.  Full
        // 128-bit Movdqu then writes Vd[15:0]=mask with Vd[127:16]=0,
        // matching the AArch64 zero-extend semantic for Hd.
        as_.Packssdw(xmm_result, xmm_zero);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_result);
      } else {
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_zero);
        if (is_double) {
          as_.Movsd({.base = Assembler::rbp, .disp = dst_off}, xmm_result);
        } else {
          as_.Movss({.base = Assembler::rbp, .disp = dst_off}, xmm_result);
        }
      }
      return;
    }

    if (is_frecps || is_frsqrts) {
      // Constants: K_fma (the additive in FMA(-a,b,K)) and K_sat (the
      // saturation value when (0,inf) cross).
      const int64_t k_fma_bits_d  = is_frecps ? int64_t{0x4000000000000000LL}
                                              : int64_t{0x4008000000000000LL};
      const int32_t k_fma_bits_s  = is_frecps ? int32_t{0x40000000}
                                              : int32_t{0x40400000};
      const int64_t k_sat_bits_d  = is_frecps ? int64_t{0x4000000000000000LL}
                                              : int64_t{0x3FF8000000000000LL};
      const int32_t k_sat_bits_s  = is_frecps ? int32_t{0x40000000}
                                              : int32_t{0x3FC00000};
      const int64_t qnan_bits_d   = int64_t{0x7FF8000000000000LL};
      const int32_t qnan_bits_s   = int32_t{0x7FC00000};
      const int64_t two_bits_d    = int64_t{0x4000000000000000LL};
      const int32_t two_bits_s    = int32_t{0x40000000};

      SimdRegister xmm_n = AllocTempSimdReg();
      SimdRegister xmm_m = AllocTempSimdReg();
      SimdRegister xmm_mul = AllocTempSimdReg();
      SimdRegister xmm_iu = AllocTempSimdReg();
      SimdRegister xmm_special = AllocTempSimdReg();
      if (!success()) { return; }

      // FP16 lift via F16C round-trip — Pxor + Pinsrw +
      // Vcvtph2ps widens each FP16 source lane to FP32 in xmm lane 0.  The
      // existing FP32 FRECPS/FRSQRTS Newton-step core (Vfnmadd231ss of
      // K_fma - a*b, plus a Divss-by-2 step for FRSQRTS, with K_sat blend on
      // special_mask and qnan blend on input_unord) then runs unchanged on
      // the lifted operands.  Lanes 1..3 of xmm_n / xmm_m stay FP32 +0 by
      // Vcvtph2ps zero-fill, and all subsequent FP32-width primitives in the
      // core preserve that upper-zero invariant (Mulss / Cmpunordps / Pand /
      // Por / Vfnmadd231ss / Divss / Movd-zero-extended-into-xmm all leave
      // lanes 1..3 == 0).  Divss of (lane0=fma_result, lanes1..3=+0) by
      // (lane0=+2, lanes1..3=+0) is undefined per SSE — but Divss only writes
      // lane 0 and leaves lanes 1..3 of the dst unchanged, so the upper
      // zeroes survive.  The narrow at the store path uses Vcvtps2ph
      // (real-FP output), not Packssdw (mask output).
      const bool use_single = args.is_fp16 || !is_double;
      if (args.is_fp16) {
        as_.Pxor(xmm_n, xmm_n);
        as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_n, xmm_n);
        as_.Pxor(xmm_m, xmm_m);
        as_.Pinsrw(xmm_m, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_m, xmm_m);
      } else if (is_double) {
        as_.Movsd(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
        as_.Movsd(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      } else {
        as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
        as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      }

      // mul = a * b (only its NaN bit is observed via cmpunord below).
      as_.Movdqa(xmm_mul, xmm_n);
      if (use_single) as_.Mulss(xmm_mul, xmm_m);
      else            as_.Mulsd(xmm_mul, xmm_m);

      // input_unord = cmpunord(a, b): all-ones iff a or b is NaN.
      as_.Movdqa(xmm_iu, xmm_n);
      if (use_single) as_.Cmpunordps(xmm_iu, xmm_m);
      else            as_.Cmpunordpd(xmm_iu, xmm_m);

      // mul_unord = cmpunord(mul, mul): all-ones iff mul is NaN.  Store
      // in-place over xmm_mul (the actual product is no longer needed).
      if (use_single) as_.Cmpunordps(xmm_mul, xmm_mul);
      else            as_.Cmpunordpd(xmm_mul, xmm_mul);

      // special_mask = (NOT input_unord) AND mul_unord.  Preserve input_unord.
      as_.Movdqa(xmm_special, xmm_iu);
      as_.Pandn(xmm_special, xmm_mul);
      // xmm_mul is now free as scratch.

      // Normal-path result = K_fma - a*b via VFNMADD231.  Load K_fma into
      // xmm_mul (the destination), then VFNMADD231(dest, n, m) -> dest -= n*m.
      Register tmp_gpr = AllocTempReg();
      if (use_single) {
        as_.Movl(tmp_gpr, k_fma_bits_s);
        as_.Movd(xmm_mul, tmp_gpr);
        as_.Vfnmadd231ss(xmm_mul, xmm_n, xmm_m);
      } else {
        as_.Movq(tmp_gpr, k_fma_bits_d);
        as_.Movq(xmm_mul, tmp_gpr);
        as_.Vfnmadd231sd(xmm_mul, xmm_n, xmm_m);
      }

      // FRSQRTS: divide by 2 (exact one-exponent decrement).  Reuse xmm_n
      // as scratch — a is no longer needed past this point.
      if (is_frsqrts) {
        if (use_single) {
          as_.Movl(tmp_gpr, two_bits_s);
          as_.Movd(xmm_n, tmp_gpr);
          as_.Divss(xmm_mul, xmm_n);
        } else {
          as_.Movq(tmp_gpr, two_bits_d);
          as_.Movq(xmm_n, tmp_gpr);
          as_.Divsd(xmm_mul, xmm_n);
        }
      }
      // xmm_mul now holds fma_result.  xmm_n, xmm_m are free as scratch
      // (we reuse them as constant-load targets below).

      // First select: result_first = special_mask ? K_sat : fma_result.
      // Build K_sat in xmm_n.
      if (use_single) {
        as_.Movl(tmp_gpr, k_sat_bits_s);
        as_.Movd(xmm_n, tmp_gpr);
      } else {
        as_.Movq(tmp_gpr, k_sat_bits_d);
        as_.Movq(xmm_n, tmp_gpr);
      }
      as_.Pand(xmm_n, xmm_special);       // xmm_n = K_sat AND special
      as_.Pandn(xmm_special, xmm_mul);    // xmm_special = (NOT special) AND fma
      as_.Por(xmm_n, xmm_special);        // xmm_n = result_first

      // Second select: result_final = input_unord ? qnan : result_first.
      // Build qnan in xmm_m (b is no longer needed).
      if (use_single) {
        as_.Movl(tmp_gpr, qnan_bits_s);
        as_.Movd(xmm_m, tmp_gpr);
      } else {
        as_.Movq(tmp_gpr, qnan_bits_d);
        as_.Movq(xmm_m, tmp_gpr);
      }
      as_.Pand(xmm_m, xmm_iu);            // xmm_m = qnan AND iu
      as_.Pandn(xmm_iu, xmm_n);           // xmm_iu = (NOT iu) AND result_first
      as_.Por(xmm_m, xmm_iu);             // xmm_m = result_final

      // FP16 narrow via Vcvtps2ph (real-FP output, NOT a
      // mask).  Rounds 4 FP32 lanes to 4 FP16 word lanes in dst xmm's low
      // 64 bits and zero-fills the upper 64 bits.  Lanes 1..3 of xmm_m are
      // FP32 +0 throughout the Pand/Por blend pipeline (the upper-zero
      // invariant is preserved by all primitives above), so the narrowed
      // word lanes 1..3 are 0x0000.  Full 128-bit Movdqu writes Vd[15:0] =
      // FP16 result with Vd[127:16] = 0.
      if (args.is_fp16) {
        as_.Vcvtps2ph(xmm_m, xmm_m, int8_t{0});
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_m);
      } else {
        // Zero Vd above the result lane, then write the scalar lane 0.
        as_.Pxor(xmm_n, xmm_n);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_n);
        if (is_double) {
          as_.Movsd({.base = Assembler::rbp, .disp = dst_off}, xmm_m);
        } else {
          as_.Movss({.base = Assembler::rbp, .disp = dst_off}, xmm_m);
        }
      }
      return;
    }

    SimdRegister xmm_n = AllocTempSimdReg();
    SimdRegister xmm_m = AllocTempSimdReg();
    SimdRegister xmm_mul = AllocTempSimdReg();
    SimdRegister xmm_mul_unord = AllocTempSimdReg();
    SimdRegister xmm_input_unord = AllocTempSimdReg();
    SimdRegister xmm_two = AllocTempSimdReg();
    if (!success()) { return; }

    // when args.is_fp16, lift each FP16 source lane to
    // FP32 in xmm lane 0 (Pxor + Pinsrw + Vcvtph2ps), run the existing FP32
    // FMULX compute core on the lifted operands, then narrow the FP32 result
    // back to FP16 via Vcvtps2ph.  Vcvtph2ps puts FP32 +0.0 in lanes 1..3,
    // which means mul/input cmpunord and ±2.0 bits in lanes 1..3 stay 0
    // through the blend; Vcvtps2ph on (result, 0, 0, 0) writes word lane 0 =
    // FP16 result and word lanes 1..3 = 0, and auto-zeroes the upper 64 bits.
    // Full Movdqu then writes Vd[15:0]=result with Vd[127:16]=0.
    const bool use_single = args.is_fp16 || !is_double;
    if (args.is_fp16) {
      as_.Pxor(xmm_n, xmm_n);
      as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_n, xmm_n);
      as_.Pxor(xmm_m, xmm_m);
      as_.Pinsrw(xmm_m, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_m, xmm_m);
    } else if (is_double) {
      as_.Movsd(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movsd(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    }

    // mul = a * b
    as_.Movdqa(xmm_mul, xmm_n);
    if (use_single) as_.Mulss(xmm_mul, xmm_m);
    else            as_.Mulsd(xmm_mul, xmm_m);

    // mul_unord = cmpunord(mul, mul): all-ones iff mul is NaN.
    as_.Movdqa(xmm_mul_unord, xmm_mul);
    if (use_single) as_.Cmpunordps(xmm_mul_unord, xmm_mul_unord);
    else            as_.Cmpunordpd(xmm_mul_unord, xmm_mul_unord);

    // input_unord = cmpunord(a, b): all-ones iff either a or b is NaN.
    // CMPUNORDPS/PD sets the lane to all-ones when either operand is NaN.
    as_.Movdqa(xmm_input_unord, xmm_n);
    if (use_single) as_.Cmpunordps(xmm_input_unord, xmm_m);
    else            as_.Cmpunordpd(xmm_input_unord, xmm_m);

    // special_mask = mul_unord AND NOT input_unord.
    // Use PANDN: dst = (NOT dst) AND src.  So xmm_input_unord becomes
    // (NOT input_unord) AND mul_unord.
    as_.Pandn(xmm_input_unord, xmm_mul_unord);
    // xmm_input_unord now holds special_mask.

    // two_signed = (a XOR b) restricted to sign bit, OR'd with bits of 2.0.
    // Reuse xmm_n as the scratch for (a XOR b); we still have xmm_m intact.
    if (use_single) as_.Xorps(xmm_n, xmm_m);
    else            as_.Xorpd(xmm_n, xmm_m);
    // Build sign-bit mask in xmm_mul_unord (we no longer need mul_unord).
    as_.Pcmpeqd(xmm_mul_unord, xmm_mul_unord);
    if (use_single) as_.Pslld(xmm_mul_unord, int8_t{31});
    else            as_.Psllq(xmm_mul_unord, int8_t{63});
    as_.Pand(xmm_n, xmm_mul_unord);
    // Build ±2.0 by OR'ing in the bits of +2.0.
    Register tmp_gpr = AllocTempReg();
    if (use_single) {
      as_.Movl(tmp_gpr, int32_t{0x40000000});  // bits of 2.0 (FP32)
      as_.Movd(xmm_two, tmp_gpr);
    } else {
      as_.Movq(tmp_gpr, int64_t{0x4000000000000000LL});  // bits of 2.0 (FP64)
      as_.Movq(xmm_two, tmp_gpr);
    }
    as_.Por(xmm_n, xmm_two);
    // xmm_n now holds ±2.0 (lane 0), with sign = sign(a) XOR sign(b).

    // Blend: result = (mul AND NOT mask) OR (±2.0 AND mask).
    // Reuse xmm_m as the masked-2.0; reuse xmm_mul_unord as the masked-mul.
    as_.Movdqa(xmm_m, xmm_n);
    as_.Pand(xmm_m, xmm_input_unord);          // m = ±2.0 AND mask
    as_.Pandn(xmm_input_unord, xmm_mul);       // input_unord = NOT(mask) AND mul
    as_.Por(xmm_input_unord, xmm_m);           // result in xmm_input_unord lane 0.

    // Zero Vd, then write the scalar lane 0.  Matches AArch64 scalar
    // semantics: bits above the operand size are zero.
    // FP16 narrow path uses Vcvtps2ph (real-FP-valued
    // result, not an FP-width mask), which auto-zeroes upper 64 bits of dst
    // xmm; full Movdqu then writes Vd[15:0]=FP16 result, Vd[127:16]=0.
    if (args.is_fp16) {
      as_.Vcvtps2ph(xmm_input_unord, xmm_input_unord, int8_t{0});
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_input_unord);
    } else {
      as_.Pxor(xmm_two, xmm_two);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_two);
      if (is_double) {
        as_.Movsd({.base = Assembler::rbp, .disp = dst_off}, xmm_input_unord);
      } else {
        as_.Movss({.base = Assembler::rbp, .disp = dst_off}, xmm_input_unord);
      }
    }
  }

