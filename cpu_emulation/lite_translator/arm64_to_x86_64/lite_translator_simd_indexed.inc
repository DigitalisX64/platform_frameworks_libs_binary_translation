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
// Family: SIMD by-element (vector + scalar indexed).
// Methods:
//   AdvSimdVecXIndexedElement, AdvSimdScalarXIndexedElement

  // AdvSIMD vector by-element JIT — FMLA / FMLS / FMUL /
  // FMULX at FP32 (.2S/.4S) and FP64 (.2D).  Shape:
  //   1. Load Vn into xmm_n.
  //   2. Load Vm; broadcast lane args.index across all lanes with PSHUFD
  //      (FP32: imm = (i<<6)|(i<<4)|(i<<2)|i; FP64: imm = 0x44 for i=0,
  //      0xEE for i=1).
  //   3. For FMLA: load Vd, then VFMADD231PS/PD(d, n, m).
  //      For FMLS: load Vd, then VFNMADD231PS/PD(d, n, m).
  //      For FMUL: MULPS/MULPD(n, m), result in n.
  //      For FMULX: same MULPS/MULPD then the three-same FMULX saturation
  //      override — (±0, ±inf) lanes get ±2.0 (sign = sign(a) XOR sign(b))
  //      while NaN inputs and other normals stay as IEEE multiply.
  //   4. If Q=0, mask the upper 64 bits.
  //   5. Store back to Vd.
  //
  // FP16 (size=00) stays interpreter — needs F16C round-trip + binary64
  // FMA for bit-exact match against std::fma(double, double, double) +
  // FpSingleToHalf.
  //
  // Integer MUL/MLA/MLS by-element (size=01/10) stays interpreter — that
  // is a separate JIT family not bundled here.  Could be added later.
  //
  // Reserved .1D shape (size=11 && q=0) and hosts without FMA bail.
  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs& args) {
    using Op = Decoder::AdvSimdVecXIdxOpcode;
    // integer MUL/MLA/MLS by-element (halfword .4h/.8h
    // size=01, word .2s/.4s size=10).
    //
    // Vd = Vn op (Vm.lane[index] broadcast across destination lanes), with
    //   MUL: Vd = Vn * broadcast(Vm[index])
    //   MLA: Vd = Vd + Vn * broadcast(Vm[index])
    //   MLS: Vd = Vd - Vn * broadcast(Vm[index])
    // No saturation, no widening — the bottom esize bits of each lane-wise
    // host product match the architecturally-defined result modulo 2^esize.
    //
    // Broadcast:
    //   - halfword: index = H:L:M (3 bits, 0..7).  If index >= 4 shift Vm
    //     down by 8 bytes so the target lane sits in the low quad, then
    //     Pshuflw with imm = (i:i:i:i) (i = index & 3) replicates it across
    //     the low 4 halfword lanes, and Pshufd 0x44 mirrors low qword into
    //     high qword for all 8 lanes.
    //   - word: index = H:L (2 bits, 0..3).  Pshufd with i:i:i:i broadcasts
    //     directly across 4 word lanes.
    //
    // !Q: zero upper 64 bits (D-register semantics) via Pslldq/Psrldq 8.
    //
    // Verified encodings (aarch64-linux-gnu-as -march=armv8.2-a):
    //   mul  v0.4h, v1.4h, v2.h[0] = 0x0F428020
    //   mul  v0.8h, v1.8h, v2.h[7] = 0x4F728820
    //   mul  v0.2s, v1.2s, v2.s[1] = 0x0FA28020
    //   mul  v0.4s, v1.4s, v2.s[3] = 0x4FA28820
    //   mla  v0.4h, v1.4h, v2.h[0] = 0x2F420020
    //   mla  v0.4s, v1.4s, v2.s[2] = 0x6F820820
    //   mls  v0.8h, v1.8h, v2.h[5] = 0x6F524820
    //   mls  v0.4s, v1.4s, v2.s[1] = 0x6FA24020
    if (args.opcode == Op::kMul || args.opcode == Op::kMla ||
        args.opcode == Op::kMls) {
      if (args.size != 0b01 && args.size != 0b10) { success_ = false; return; }
      const bool is_halfword = (args.size == 0b01);

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xn = AllocTempSimdReg();
      SimdRegister xm = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});

      if (is_halfword) {
        if (args.index >= 4) {
          as_.Psrldq(xm, int8_t{8});
        }
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshuflw(xm, xm, imm);
        as_.Pshufd(xm, xm, int8_t{0x44});
      } else {
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshufd(xm, xm, imm);
      }

      SimdRegister xmm_result = no_simd_register;
      if (args.opcode == Op::kMul) {
        if (is_halfword) as_.Pmullw(xn, xm);
        else             as_.Pmulld(xn, xm);  // SSE4.1
        xmm_result = xn;
      } else {
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
        if (is_halfword) as_.Pmullw(xn, xm);
        else             as_.Pmulld(xn, xm);  // SSE4.1
        if (args.opcode == Op::kMla) {
          if (is_halfword) as_.Paddw(xd, xn);
          else             as_.Paddd(xd, xn);
        } else {
          if (is_halfword) as_.Psubw(xd, xn);
          else             as_.Psubd(xd, xn);
        }
        xmm_result = xd;
      }

      if (!args.q) {
        as_.Pslldq(xmm_result, int8_t{8});
        as_.Psrldq(xmm_result, int8_t{8});
      }
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_result);
      return;
    }

    // SQDMULH / SQRDMULH / SQRDMLAH / SQRDMLSH by-element
    // (.4h / .8h, size=01).
    //
    // Per-lane (SQDMULH/SQRDMULH):
    //   Vd[i] = SAT16((2 * Vn[i] * Vm[index]) >> 16), with SQRDMULH
    //   adding a rounding constant 1<<15 to the doubled product before the
    //   shift.
    //
    // Per-lane (SQRDMLAH/SQRDMLSH, Armv8.1-RDM):
    //   addend = SQRDMULH(Vn[i], Vm[index])
    //   Vd[i] = SignedSat16(Vd[i] ± addend)
    // — the stage-1 SQRDMULH lane reuses the PMULHRSW + corner fixup recipe;
    // stage-2 signed-saturating add/sub against [INT16_MIN, INT16_MAX] is
    // PADDSW (SQRDMLAH) / PSUBSW (SQRDMLSH), which match the architectural
    // saturation exactly at 16-bit lane granularity (SSE2).
    //
    // SQRDMULH .8h: SSSE3 PMULHRSW computes ((a*b)>>14 + 1)>>1 = SQRDMULH,
    // except for the single corner case (a=0x8000, b=0x8000) where SQRDMULH
    // saturates to 0x7FFF but PMULHRSW yields 0x8000.  Fixup: build a mask
    // where both inputs were 0x8000, then XOR the affected lanes
    // (0x8000 ^ 0xFFFF = 0x7FFF).
    //
    // SQDMULH .8h: full 32-bit signed product, arithmetic right-shift by 15,
    // then signed-saturating pack back to 16-bit lanes (PACKSSDW) handles
    // the (INT16_MIN)^2 overflow corner.  PMULLW gives the low 16 bits and
    // PMULHW gives the high 16 bits (signed); interleave via Punpcklwd /
    // Punpckhwd to reconstruct the 32-bit signed products.
    //
    // .2s/.4s (size=10) and the FP16 / FP32 / FP64 size codes fall through
    // to the existing FP-only guard below.  SQRDMLAH/SQRDMLSH at size=10
    // bail to interpreter (handled by the FP-only guard).
    if ((args.opcode == Op::kSqdmulhIdx ||
         args.opcode == Op::kSqrdmulhIdx ||
         args.opcode == Op::kSqrdmlahIdx ||
         args.opcode == Op::kSqrdmlshIdx) &&
        args.size == 0b01) {
      if (!host_platform::kHasSSSE3) { success_ = false; return; }
      const bool is_rounded =
          (args.opcode == Op::kSqrdmulhIdx ||
           args.opcode == Op::kSqrdmlahIdx ||
           args.opcode == Op::kSqrdmlshIdx);
      const bool is_accum =
          (args.opcode == Op::kSqrdmlahIdx ||
           args.opcode == Op::kSqrdmlshIdx);
      const bool is_sub = (args.opcode == Op::kSqrdmlshIdx);

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xn = AllocTempSimdReg();
      SimdRegister xm = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});

      // Broadcast Vm.h[index] across all 8 halfword lanes (same pattern as
      // the MUL/MLA/MLS .8h arm above).
      if (args.index >= 4) {
        as_.Psrldq(xm, int8_t{8});
      }
      {
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshuflw(xm, xm, imm);
        as_.Pshufd(xm, xm, int8_t{0x44});
      }

      SimdRegister xmm_result = no_simd_register;
      if (is_rounded) {
        // SQRDMULH .8h via PMULHRSW + (0x8000, 0x8000) corner fixup.
        SimdRegister xn_save = AllocTempSimdReg();
        SimdRegister xm_save = AllocTempSimdReg();
        SimdRegister x_min = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqa(xn_save, xn);
        as_.Movdqa(xm_save, xm);

        // x_min = 0x8000 broadcast across 8 halfwords (Pcmpeqw self -> all 1s
        // = 0xFFFF, then Psllw 15 -> 0x8000).
        as_.Pcmpeqw(x_min, x_min);
        as_.Psllw(x_min, int8_t{15});

        as_.Pmulhrsw(xn, xm);     // xn = SQRDMULH except (INT16_MIN)^2 corner.
        as_.Pcmpeqw(xn_save, x_min);   // lanes where Vn was 0x8000 -> 0xFFFF.
        as_.Pcmpeqw(xm_save, x_min);   // lanes where Vm was 0x8000 -> 0xFFFF.
        as_.Pand(xn_save, xm_save);    // combined corner mask.
        as_.Pxor(xn, xn_save);         // 0x8000 ^ 0xFFFF = 0x7FFF on corner.
        xmm_result = xn;
      } else {
        // SQDMULH .8h via 32-bit reconstruction + arithmetic shift + signed
        // saturating pack.
        SimdRegister x_low = AllocTempSimdReg();
        SimdRegister x_high = AllocTempSimdReg();
        SimdRegister x_lo_lanes = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqa(x_low, xn);
        as_.Movdqa(x_high, xn);
        as_.Pmullw(x_low, xm);    // x_low = (a*b) & 0xFFFF per lane.
        as_.Pmulhw(x_high, xm);   // x_high = (a*b) >> 16, signed, per lane.

        // Interleave (low16, high16) -> 32-bit signed products.
        // Punpcklwd takes lanes 0-3 of each source.
        // Punpckhwd takes lanes 4-7 of each source.
        as_.Movdqa(x_lo_lanes, x_low);
        as_.Punpcklwd(x_lo_lanes, x_high);  // 4 lanes of full 32-bit products.
        as_.Punpckhwd(x_low, x_high);       // upper 4 lanes.

        as_.Psrad(x_lo_lanes, int8_t{15});  // (a*b) >> 15 as 32-bit signed.
        as_.Psrad(x_low, int8_t{15});

        as_.Packssdw(x_lo_lanes, x_low);    // signed-saturating pack to 16-bit.
        xmm_result = x_lo_lanes;
      }

      if (is_accum) {
        // Stage 2: load Vd and signed-saturating add (SQRDMLAH) or
        // subtract (SQRDMLSH) the per-lane addend at 16-bit granularity.
        // PADDSW / PSUBSW saturate to [INT16_MIN, INT16_MAX] natively (SSE2).
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
        if (is_sub) {
          as_.Psubsw(xd, xmm_result);
        } else {
          as_.Paddsw(xd, xmm_result);
        }
        xmm_result = xd;
      }

      if (!args.q) {
        as_.Pslldq(xmm_result, int8_t{8});
        as_.Psrldq(xmm_result, int8_t{8});
      }
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_result);
      return;
    }

    // SQDMULH / SQRDMULH / SQRDMLAH / SQRDMLSH by-element
    // (.2s / .4s, size=10).
    //
    // 32-bit-lane sibling of the size=01 arm above.  Stage 1 reconstructs
    // SQDMULH .4s / SQRDMULH .4s without PSRAQ via PMULDQ at qword
    // granularity + PSHUFD-to-high-32-bits + PUNPCKLDQ-merge.  Stage 2
    // (SQRDMLAH / SQRDMLSH only) reuses the 32-bit-lane wrap-add +
    // overflow-bit-XOR-blend recipe from the SQDMLAL .4s arm below.
    //
    // Per-lane (SQDMULH/SQRDMULH .4s):
    //   product = sext_i64(Vn.s[i]) * sext_i64(Vm.s[index])
    //   doubled = 2 * product  [+ 0x80000000 for SQRDMULH]
    //   Vd.s[i] = SAT_INT32(doubled >> 32)
    // The 32x32 signed product fits in int64 (|INT32_MIN^2| = 2^62), and
    // doubling overflows int64 only at the lone corner (Vn.s[i] = INT32_MIN
    // AND Vm.s[index] = INT32_MIN), which wraps to INT64_MIN.  The upper 32
    // bits of INT64_MIN are 0x80000000 = INT32_MIN, but the architectural
    // saturated answer is INT32_MAX.  Fixup: detect the corner at 32-bit
    // lane level (PCMPEQD against INT32_MIN broadcast), then XOR the
    // affected output lanes (INT32_MIN ^ 0xFFFFFFFF = INT32_MAX).
    //
    // Per-lane (SQRDMLAH/SQRDMLSH .4s, Armv8.1-RDM):
    //   addend = SQRDMULH(Vn.s[i], Vm.s[index])
    //   Vd.s[i] = SignedSat32(Vd.s[i] ± addend)
    // Stage 2 is the same wrap-add / wrap-sub + sign-bit / XOR-blend
    // recipe the size=01 SQDMLAL/SQDMLSL arm uses for its 32-bit-lane
    // accumulator step.
    //
    // PMULDQ uses the low 32 bits of each 64-bit qword.  Vm.s[index] is
    // broadcast across all 4 dwords of xm, so the same xm serves both the
    // "lanes 0,2" PMULDQ and the "lanes 1,3" PMULDQ (after PSRLQ 32 of
    // xn) — Vm's qword-low-32 after PSRLQ 32 equals the unshifted value
    // because all 4 dwords are identical.
    //
    // Verified encodings (aarch64-linux-gnu-as -march=armv8.2-a):
    //   sqdmulh  v0.2s, v1.2s, v2.s[0]  = 0x0F82C020
    //   sqdmulh  v0.4s, v1.4s, v2.s[3]  = 0x4FA2C820
    //   sqrdmulh v0.2s, v1.2s, v2.s[1]  = 0x0FA2D020
    //   sqrdmulh v0.4s, v1.4s, v2.s[2]  = 0x4F82D820
    //   sqrdmlah v0.2s, v1.2s, v2.s[0]  = 0x2F82D020
    //   sqrdmlah v0.4s, v1.4s, v2.s[3]  = 0x6FA2D820
    //   sqrdmlsh v0.2s, v1.2s, v2.s[1]  = 0x2FA2F020
    //   sqrdmlsh v0.4s, v1.4s, v2.s[2]  = 0x6F82F820
    if ((args.opcode == Op::kSqdmulhIdx ||
         args.opcode == Op::kSqrdmulhIdx ||
         args.opcode == Op::kSqrdmlahIdx ||
         args.opcode == Op::kSqrdmlshIdx) &&
        args.size == 0b10) {
      if (!host_platform::kHasSSE4_1) { success_ = false; return; }
      const bool is_rounded =
          (args.opcode == Op::kSqrdmulhIdx ||
           args.opcode == Op::kSqrdmlahIdx ||
           args.opcode == Op::kSqrdmlshIdx);
      const bool is_accum =
          (args.opcode == Op::kSqrdmlahIdx ||
           args.opcode == Op::kSqrdmlshIdx);
      const bool is_sub = (args.opcode == Op::kSqrdmlshIdx);

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xn = AllocTempSimdReg();
      SimdRegister xm = AllocTempSimdReg();
      SimdRegister x_const = AllocTempSimdReg();
      SimdRegister corner = AllocTempSimdReg();
      SimdRegister xp_lo = AllocTempSimdReg();
      SimdRegister xp_hi = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});

      // Broadcast Vm.s[index] across all 4 dword lanes.
      {
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshufd(xm, xm, imm);
      }

      // x_const = INT32_MIN broadcast across 4 dwords (Pcmpeqd self → -1,
      // Pslld 31 → 0x80000000 per dword).
      as_.Pcmpeqd(x_const, x_const);
      as_.Pslld(x_const, int8_t{31});

      // Corner detection at 32-bit lane level: lanes where Vn.s[i] ==
      // INT32_MIN AND Vm.s[index] (broadcast) == INT32_MIN.  PCMPEQD
      // destroys dst; preserve xn and xm via Movdqa-to-temp first.
      as_.Movdqa(corner, xn);
      as_.Pcmpeqd(corner, x_const);
      as_.Movdqa(xp_lo, xm);
      as_.Pcmpeqd(xp_lo, x_const);
      as_.Pand(corner, xp_lo);
      // `corner` now holds the 4-lane int32 corner mask; xp_lo is dead and
      // about to be repurposed as the lanes-0,2 product slot.

      // Stage 1: two PMULDQs reconstruct the 4 signed 32×32 → 64 products.
      //   xp_lo qword 0 = sext_i64(Vn.s[0]) * sext_i64(Vm.s[index])
      //   xp_lo qword 1 = sext_i64(Vn.s[2]) * sext_i64(Vm.s[index])
      //   xp_hi qword 0 = sext_i64(Vn.s[1]) * sext_i64(Vm.s[index])
      //   xp_hi qword 1 = sext_i64(Vn.s[3]) * sext_i64(Vm.s[index])
      // Vm.s[index] broadcast means xm's low-32-bit-of-each-qword equals
      // the broadcast value before AND after PSRLQ 32 on the multiplicand
      // side — so xm is reused for both PMULDQs without re-shuffling.
      as_.Movdqa(xp_lo, xn);
      as_.Pmuldq(xp_lo, xm);
      as_.Movdqa(xp_hi, xn);
      as_.Psrlq(xp_hi, int8_t{32});
      as_.Pmuldq(xp_hi, xm);
      // xn and xm are dead from here.  x_const is dead (about to be reused).

      // Double each 64-bit signed product via PSLLQ 1.  Corner lanes wrap
      // from +2^62 to ±2^63 (= INT64_MIN as signed), which after the
      // upper-32-bits extraction becomes INT32_MIN — the corner mask
      // XOR-blend below restores INT32_MAX.
      as_.Psllq(xp_lo, int8_t{1});
      as_.Psllq(xp_hi, int8_t{1});

      if (is_rounded) {
        // Rounding constant 2^31 = 0x80000000 per qword.  Build in
        // x_const: Pcmpeqd self → -1 per dword; Psllq 63 → 0x8000…0000
        // per qword (only bit 63 set); Psrlq 32 → 0x0000…0000_80000000.
        as_.Pcmpeqd(x_const, x_const);
        as_.Psllq(x_const, int8_t{63});
        as_.Psrlq(x_const, int8_t{32});
        as_.Paddq(xp_lo, x_const);
        as_.Paddq(xp_hi, x_const);
      }

      // Extract the upper 32 bits of each 64-bit lane: PSHUFD with mask
      // 0b11_01_11_01 = 0xDD packs dwords [1, 3, 1, 3] → low 64 bits
      // hold [hi32_qword0, hi32_qword1].
      as_.Pshufd(xp_lo, xp_lo, static_cast<int8_t>(0xDD));
      as_.Pshufd(xp_hi, xp_hi, static_cast<int8_t>(0xDD));

      // Interleave to produce the 4 int32 result lanes in order:
      //   xp_lo low 2 lanes after PSHUFD: [prod_lane0, prod_lane2]
      //   xp_hi low 2 lanes after PSHUFD: [prod_lane1, prod_lane3]
      // PUNPCKLDQ interleaves low qwords:
      //   xp_lo := [xp_lo[0], xp_hi[0], xp_lo[1], xp_hi[1]]
      //         =  [prod_lane0, prod_lane1, prod_lane2, prod_lane3]
      as_.Punpckldq(xp_lo, xp_hi);

      // Apply the corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX, so
      // XORing with `corner` (each lane either 0 or 0xFFFFFFFF) replaces
      // INT32_MIN with INT32_MAX in corner lanes and leaves the rest
      // unchanged.
      as_.Pxor(xp_lo, corner);

      SimdRegister xmm_result = xp_lo;

      if (is_accum) {
        // Stage 2: 32-bit signed-saturating add/sub of Vd and the
        // saturated stage-1 product.  Same wrap-add + overflow-bit-XOR-
        // blend recipe as the SQDMLAL .4s arm (size=01) below.
        //   sum  = a ± b              (PADDD/PSUBD wraps mod 2^32)
        //   ovf  = ~(a^b) & (a^sum)   for ADD  (sign bit indicates overflow)
        //        |  (a^b) & (a^diff)  for SUB
        //   sat  = (a < 0) ? INT32_MIN : INT32_MAX
        //   res  = sum ^ ((sum ^ sat) & ovf_mask)
        SimdRegister xd = AllocTempSimdReg();
        SimdRegister t_sum = AllocTempSimdReg();
        SimdRegister t_ovf = AllocTempSimdReg();
        SimdRegister t_sat = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
        as_.Movdqa(t_sum, xd);
        if (is_sub) {
          as_.Psubd(t_sum, xmm_result);
        } else {
          as_.Paddd(t_sum, xmm_result);
        }
        as_.Movdqa(t_ovf, xd);
        as_.Pxor(t_ovf, xmm_result);
        as_.Movdqa(t_sat, xd);
        as_.Pxor(t_sat, t_sum);
        // xmm_result (the addend 'b') is dead after t_ovf and t_sum are
        // computed; reuse its slot for the INT32_MAX broadcast.
        if (!is_sub) {
          as_.Pcmpeqd(xmm_result, xmm_result);
          as_.Pxor(t_ovf, xmm_result);   // ~(a^b)
          as_.Pand(t_ovf, t_sat);        // ~(a^b) & (a^sum)
          as_.Psrld(xmm_result, int8_t{1});  // INT32_MAX broadcast
        } else {
          as_.Pand(t_ovf, t_sat);        // (a^b) & (a^diff)
          as_.Pcmpeqd(xmm_result, xmm_result);
          as_.Psrld(xmm_result, int8_t{1});  // INT32_MAX broadcast
        }
        as_.Psrad(t_ovf, int8_t{31});    // overflow mask (all-1s or 0)
        as_.Psrad(xd, int8_t{31});       // xd = (a<0) ? -1 : 0
        as_.Pxor(xd, xmm_result);        // xd = sat = (a<0) ? INT_MIN : INT_MAX
        as_.Pxor(xd, t_sum);             // xd = sat ^ sum
        as_.Pand(xd, t_ovf);             // ... & ovf
        as_.Pxor(xd, t_sum);             // = sum ^ ((sum^sat) & ovf)
        xmm_result = xd;
      }

      if (!args.q) {
        as_.Pslldq(xmm_result, int8_t{8});
        as_.Psrldq(xmm_result, int8_t{8});
      }
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_result);
      return;
    }

    // widening MUL/MAC by-element (size=01: .4h/.8h -> .4s).
    //
    // SMULL/UMULL/SMLAL/UMLAL/SMLSL/UMLSL by element with halfword sources
    // and word destination — 4 output lanes (.4s).  Q=0 (*MULL/*MLAL/*MLSL)
    // selects Vn.h[0..3] (low half) as multiplicands; Q=1 (*MULL2/*MLAL2/
    // *MLSL2) selects Vn.h[4..7] (high half).  Destination is always
    // 128-bit regardless of Q — widening forms do NOT zero the upper half
    // (this differs from the same-size MUL/MLA/MLS arm above).
    //
    // Strategy: widen each operand's 4 halfword lanes to 32-bit via
    // PMOVSXWD (signed: SMULL/SMLAL/SMLSL) or PMOVZXWD (unsigned: UMULL/
    // UMLAL/UMLSL), then PMULLD for the 32-bit product (low 32 of PMULLD's
    // signed result equals the correct full-width product since both 16x16
    // signed and 16x16 unsigned products fit in 32 bits).  PMOVSXWD /
    // PMOVZXWD from memory at vn_off+(q?8:0) picks the correct half
    // directly.  For Vm we broadcast Vm.h[index] across all 8 halfword
    // lanes first (shared shim with MUL/MLA/MLS), then widen the low 4
    // identical lanes.  For SMLAL/UMLAL load Vd and PADDD; for SMLSL/UMLSL
    // load Vd and PSUBD.  The size=10 (.2s/.4s -> .2d) sibling arm below
    // handles word sources via PMOVSXDQ/PMOVZXDQ + PMULDQ/PMULUDQ.
    //
    // Verified encodings (ARM ARM C7.2 / handoff-353):
    //   smull   v0.4s, v1.4h, v2.h[0] = 0x0F42A020
    //   smull2  v0.4s, v1.8h, v2.h[7] = 0x4F72A820
    //   umull   v0.4s, v1.4h, v2.h[0] = 0x2F42A020
    //   smlal   v0.4s, v1.4h, v2.h[0] = 0x0F422020
    //   umlal   v0.4s, v1.4h, v2.h[0] = 0x2F422020
    //   smlsl   v0.4s, v1.4h, v2.h[0] = 0x0F426020
    //   umlsl   v0.4s, v1.4h, v2.h[0] = 0x2F426020
    if ((args.opcode == Op::kSmullIdx || args.opcode == Op::kUmullIdx ||
         args.opcode == Op::kSmlalIdx || args.opcode == Op::kUmlalIdx ||
         args.opcode == Op::kSmlslIdx || args.opcode == Op::kUmlslIdx) &&
        args.size == 0b01) {
      const bool is_signed = (args.opcode == Op::kSmullIdx ||
                              args.opcode == Op::kSmlalIdx ||
                              args.opcode == Op::kSmlslIdx);
      const bool is_accum = (args.opcode == Op::kSmlalIdx ||
                             args.opcode == Op::kUmlalIdx ||
                             args.opcode == Op::kSmlslIdx ||
                             args.opcode == Op::kUmlslIdx);
      const bool is_sub = (args.opcode == Op::kSmlslIdx ||
                           args.opcode == Op::kUmlslIdx);

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xm = AllocTempSimdReg();
      SimdRegister xn = AllocTempSimdReg();
      if (!success()) { return; }

      // Broadcast Vm.h[index] across all 8 halfword lanes of xm.
      as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});
      if (args.index >= 4) {
        as_.Psrldq(xm, int8_t{8});
      }
      {
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshuflw(xm, xm, imm);
        as_.Pshufd(xm, xm, int8_t{0x44});
      }

      // Widen Vm's low 4 broadcast halfwords to 4 × 32-bit lanes (with the
      // correct sign-extend / zero-extend semantics for the product).
      if (is_signed) as_.Pmovsxwd(xm, xm);
      else           as_.Pmovzxwd(xm, xm);

      // Widen Vn's selected source half to 4 × 32-bit lanes.  Q=0 reads
      // bytes 0..7 of Vn; Q=1 reads bytes 8..15 — PMOVSXWD/PMOVZXWD with
      // a memory operand at vn_off+(q?8:0) picks the correct half.
      int32_t vn_src_off = vn_off + (args.q ? 8 : 0);
      if (is_signed) {
        as_.Pmovsxwd(xn, {.base = Assembler::rbp, .disp = vn_src_off});
      } else {
        as_.Pmovzxwd(xn, {.base = Assembler::rbp, .disp = vn_src_off});
      }

      // 32-bit multiply: both signed and unsigned 16x16 products fit in
      // 32 bits, so PMULLD's signed low-32 result equals the architectural
      // result for both forms.
      as_.Pmulld(xn, xm);

      if (is_accum) {
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
        if (is_sub) as_.Psubd(xd, xn);
        else        as_.Paddd(xd, xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
      } else {
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
      }
      return;
    }

    // widening MUL/MAC by-element (size=10: .2s/.4s -> .2d).
    //
    // SMULL/UMULL/SMLAL/UMLAL/SMLSL/UMLSL by element with word sources and
    // doubleword destination — 2 output lanes (.2d).  Q=0 (*MULL/*MLAL/*MLSL)
    // selects Vn.s[0..1] (low half); Q=1 (*MULL2/*MLAL2/*MLSL2) selects
    // Vn.s[2..3] (high half).  The destination is always 128-bit regardless
    // of Q.
    //
    // Strategy: PMOVSXDQ (signed) / PMOVZXDQ (unsigned) from memory at
    // vn_off+(q?8:0) loads the selected pair of source dwords into two
    // 64-bit lanes — each lane has the source value in its low 32 bits
    // (which is what PMULDQ/PMULUDQ read; the high 32 bits are ignored).
    // Broadcast Vm.s[index] across all 4 dword lanes via PSHUFD with
    // imm = index * 0x55 — this places Vm.s[index] in dword positions 0
    // and 2 of xm, which is what PMULDQ/PMULUDQ read.  Then PMULDQ
    // (signed) or PMULUDQ (unsigned) computes two 32x32 → 64 lanes in
    // a single instruction.  For SMLAL/UMLAL load Vd and PADDQ; for
    // SMLSL/UMLSL load Vd and PSUBQ.
    //
    // Verified encodings (ARM ARM C7.2):
    //   smull   v0.2d, v1.2s, v2.s[0] = 0x0F82A020
    //   smull2  v0.2d, v1.4s, v2.s[3] = 0x4FA2A820
    //   umull   v0.2d, v1.2s, v2.s[0] = 0x2F82A020
    //   smlal   v0.2d, v1.2s, v2.s[0] = 0x0F822020
    //   umlal   v0.2d, v1.2s, v2.s[0] = 0x2F822020
    //   smlsl   v0.2d, v1.2s, v2.s[0] = 0x0F826020
    //   umlsl   v0.2d, v1.2s, v2.s[0] = 0x2F826020
    if ((args.opcode == Op::kSmullIdx || args.opcode == Op::kUmullIdx ||
         args.opcode == Op::kSmlalIdx || args.opcode == Op::kUmlalIdx ||
         args.opcode == Op::kSmlslIdx || args.opcode == Op::kUmlslIdx) &&
        args.size == 0b10) {
      const bool is_signed = (args.opcode == Op::kSmullIdx ||
                              args.opcode == Op::kSmlalIdx ||
                              args.opcode == Op::kSmlslIdx);
      const bool is_accum = (args.opcode == Op::kSmlalIdx ||
                             args.opcode == Op::kUmlalIdx ||
                             args.opcode == Op::kSmlslIdx ||
                             args.opcode == Op::kUmlslIdx);
      const bool is_sub = (args.opcode == Op::kSmlslIdx ||
                           args.opcode == Op::kUmlslIdx);

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xm = AllocTempSimdReg();
      SimdRegister xn = AllocTempSimdReg();
      if (!success()) { return; }

      // Broadcast Vm.s[index] across all 4 dword lanes of xm.  PMULDQ /
      // PMULUDQ read from dword positions 0 and 2 of each operand; after
      // this PSHUFD both of those positions hold Vm.s[index].
      as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});
      {
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshufd(xm, xm, imm);
      }

      // Widen Vn's selected source half: PMOVSXDQ/PMOVZXDQ reads two
      // consecutive dwords from memory and places each in its own 64-bit
      // lane (with sign- or zero-extension into the upper 32 bits of each
      // lane that PMULDQ/PMULUDQ ignore).  Q=0 reads bytes 0..7; Q=1 reads
      // bytes 8..15.
      int32_t vn_src_off = vn_off + (args.q ? 8 : 0);
      if (is_signed) {
        as_.Pmovsxdq(xn, {.base = Assembler::rbp, .disp = vn_src_off});
      } else {
        as_.Pmovzxdq(xn, {.base = Assembler::rbp, .disp = vn_src_off});
      }

      // 32x32 → 64 multiply.  PMULDQ is signed; PMULUDQ is unsigned.
      // Both read dword positions 0 and 2 of each operand and write two
      // 64-bit product lanes.
      if (is_signed) {
        as_.Pmuldq(xn, xm);
      } else {
        as_.Pmuludq(xn, xm);
      }

      if (is_accum) {
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
        if (is_sub) as_.Psubq(xd, xn);
        else        as_.Paddq(xd, xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
      } else {
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
      }
      return;
    }

    // SQDMULL / SQDMLAL / SQDMLSL by element
    // (size=01: .4h/.8h -> .4s).
    //
    // Signed saturating doubling widening multiply (and accumulate /
    // subtract).  Mirrors the interpreter arm in interpreter.h:
    //
    //   doubled = SignedSat(2 * sign_ext(Vn[i]) * sign_ext(Vm.h[index])),
    //             saturated to [INT32_MIN, INT32_MAX]
    //   SQDMULL :  Vd[i] = doubled
    //   SQDMLAL :  Vd[i] = SignedSat(Vd[i] + doubled)
    //   SQDMLSL :  Vd[i] = SignedSat(Vd[i] - doubled)
    //
    // The 16x16 signed product fits exactly in int32 (max |INT16_MIN ×
    // INT16_MIN| = 2^30), so PMOVSXWD + PMULLD reconstructs it without
    // loss.  Doubling via PADDD self-add overflows only when the product
    // is exactly 2^30, which corresponds to (Vn[i] = INT16_MIN AND Vm =
    // INT16_MIN) — the lone architectural saturation corner.  Detect
    // that corner at 16-bit level (PCMPEQW against 0x8000 broadcast),
    // sign-extend the mask to 32-bit lanes, and XOR-blend INT32_MAX
    // over the wrapped doubled value.  The full 128-bit Vd is written
    // regardless of Q (widening forms — 4 × 32-bit lanes = 128 bits).
    //
    // SQDMLAL/SQDMLSL add a stage-2 signed saturating add/sub of
    // (Vd, addend) against [INT32_MIN, INT32_MAX], reusing the same
    // wrap-add + overflow-bit-XOR-blend recipe as the SQADD/SQSUB
    // three-same arm for 32-bit lanes.
    //
    // Verified encodings (aarch64-linux-gnu-as -march=armv8.2-a):
    //   sqdmull  v0.4s, v1.4h, v2.h[0]  = 0x0F42B020
    //   sqdmull2 v0.4s, v1.8h, v2.h[7]  = 0x4F72B820
    //   sqdmlal  v0.4s, v1.4h, v2.h[1]  = 0x0F623020
    //   sqdmlal2 v0.4s, v1.8h, v2.h[6]  = 0x4F623820
    //   sqdmlsl  v0.4s, v1.4h, v2.h[3]  = 0x0F427820
    //   sqdmlsl2 v0.4s, v1.8h, v2.h[5]  = 0x4F525820
    if ((args.opcode == Op::kSqdmullIdx ||
         args.opcode == Op::kSqdmlalIdx ||
         args.opcode == Op::kSqdmlslIdx) &&
        args.size == 0b01) {
      if (!host_platform::kHasSSE4_1) { success_ = false; return; }
      const bool is_accum = (args.opcode != Op::kSqdmullIdx);
      const bool is_sub = (args.opcode == Op::kSqdmlslIdx);

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xm = AllocTempSimdReg();
      SimdRegister xn_full = AllocTempSimdReg();
      SimdRegister x_const = AllocTempSimdReg();
      SimdRegister corner = AllocTempSimdReg();
      if (!success()) { return; }

      // Broadcast Vm.h[index] across all 8 halfword lanes of xm.
      as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});
      if (args.index >= 4) {
        as_.Psrldq(xm, int8_t{8});
      }
      {
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshuflw(xm, xm, imm);
        as_.Pshufd(xm, xm, int8_t{0x44});
      }

      // x_const = 0x8000 broadcast across 8 halfwords.
      as_.Pcmpeqw(x_const, x_const);
      as_.Psllw(x_const, int8_t{15});

      // corner = (Vn lane == 0x8000) AND (Vm broadcast == 0x8000), per
      // halfword.  x_const is dead after this block.
      as_.Movdqu(xn_full, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movdqa(corner, xn_full);
      as_.Pcmpeqw(corner, x_const);
      as_.Pcmpeqw(x_const, xm);
      as_.Pand(corner, x_const);

      // Widen the Vm broadcast's low 4 halfwords to 4 × 32-bit signed.
      as_.Pmovsxwd(xm, xm);

      // Widen Vn's Q-selected half via memory-operand PMOVSXWD.  Reuse
      // xn_full's register slot (the full-Vn copy is dead now).
      int32_t vn_src_off = vn_off + (args.q ? 8 : 0);
      as_.Pmovsxwd(xn_full, {.base = Assembler::rbp, .disp = vn_src_off});

      // 32-bit signed multiply — PMULLD's low-32 result equals the
      // architectural product (16x16 product fits in int32).
      as_.Pmulld(xn_full, xm);

      // Sign-extend the corner mask's Q-selected half to 32-bit lanes.
      if (args.q) {
        as_.Psrldq(corner, int8_t{8});
      }
      as_.Pmovsxwd(corner, corner);

      // Double via self-add — corner lanes wrap from +2^30 to -2^31.
      as_.Paddd(xn_full, xn_full);

      // x_const := INT32_MAX broadcast (Pcmpeqd self -> -1, Psrld 1 ->
      // 0x7FFFFFFF), then XOR-blend over corner lanes:
      //   xn_full = doubled ^ ((doubled ^ INT32_MAX) & corner_mask).
      as_.Pcmpeqd(x_const, x_const);
      as_.Psrld(x_const, int8_t{1});
      as_.Pxor(x_const, xn_full);
      as_.Pand(x_const, corner);
      as_.Pxor(xn_full, x_const);

      if (!is_accum) {
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn_full);
        return;
      }

      // Stage 2: 32-bit signed saturating add/sub of Vd_wide and the
      // saturated doubled product.  Same overflow-bit-XOR-blend recipe
      // as the SQADD/SQSUB 32-bit arms at lite_translator.h:4824/4899.
      //   sum  = a + b      (wrap mod 2^32)
      //   ovf  = (~(a^b) & (a^sum)) (SQADD) or ((a^b) & (a^diff)) (SQSUB)
      //   sat  = (a < 0) ? INT_MIN : INT_MAX
      //   res  = sum ^ ((sum ^ sat) & ovf_mask)
      SimdRegister xd = AllocTempSimdReg();
      SimdRegister t_sum = AllocTempSimdReg();
      SimdRegister t_ovf = AllocTempSimdReg();
      SimdRegister t_sat = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
      as_.Movdqa(t_sum, xd);
      if (is_sub) {
        as_.Psubd(t_sum, xn_full);
      } else {
        as_.Paddd(t_sum, xn_full);
      }
      as_.Movdqa(t_ovf, xd);
      as_.Pxor(t_ovf, xn_full);
      as_.Movdqa(t_sat, xd);
      as_.Pxor(t_sat, t_sum);
      // xn_full holds the addend (still live as 'b').  It becomes dead
      // after t_ovf and t_sum are computed; reuse it to build INT32_MAX.
      if (!is_sub) {
        as_.Pcmpeqd(xn_full, xn_full);
        as_.Pxor(t_ovf, xn_full);   // ~(a^b).
        as_.Pand(t_ovf, t_sat);     // ~(a^b) & (a^sum).
        as_.Psrld(xn_full, int8_t{1});  // xn_full = INT32_MAX broadcast.
      } else {
        as_.Pand(t_ovf, t_sat);     // (a^b) & (a^diff).
        as_.Pcmpeqd(xn_full, xn_full);
        as_.Psrld(xn_full, int8_t{1});  // INT32_MAX broadcast.
      }
      as_.Psrad(t_ovf, int8_t{31});     // overflow mask (all-1s or 0).
      as_.Psrad(xd, int8_t{31});        // xd = (a<0)?-1:0.
      as_.Pxor(xd, xn_full);            // xd = sat = (a<0)?INT_MIN:INT_MAX.
      as_.Pxor(xd, t_sum);              // xd = sat ^ sum (or sat ^ diff).
      as_.Pand(xd, t_ovf);              // ... & ovf.
      as_.Pxor(xd, t_sum);              // = sum ^ ((sum^sat) & ovf).
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
      return;
    }

    // SQDMULL / SQDMLAL / SQDMLSL by element
    // (size=10: .2s/.4s -> .2d).
    //
    // 32-bit-source / 64-bit-destination sibling of the size=01 arm
    // above.  Two output lanes (.2D).  Q=0 selects Vn.s[0..1] (low
    // half); Q=1 (*MULL2/*MLAL2/*MLSL2) selects Vn.s[2..3] (high half).
    // The destination is always 128-bit regardless of Q.
    //
    // Stage 1 (saturating doubling widening multiply):
    //   doubled = SignedSat(2 * sign_ext(Vn.s[i]) * sign_ext(Vm.s[index]))
    //           saturated to [INT64_MIN, INT64_MAX]
    // The 32x32 signed product fits exactly in int64 (max |INT32_MIN ×
    // INT32_MIN| = 2^62 < INT64_MAX), so PMULDQ reconstructs it
    // losslessly.  PADDQ self-add doubles; overflow happens only when
    // the product is exactly 2^62, which requires (Vn = INT32_MIN AND
    // Vm = INT32_MIN) — the lone architectural saturation corner.
    // Detect that corner at 32-bit level (PCMPEQD against 0x80000000
    // broadcast for both Vn and Vm), AND the two 32-bit masks, then
    // Q-select via Psrldq 8 and sign-extend to 64-bit lanes via
    // Pmovsxdq; XOR-blend INT64_MAX over the wrapped doubled lanes.
    //
    // Stage 2 (SQDMLAL/SQDMLSL only): 64-bit signed saturating add/sub
    // of (Vd, doubled).  No PSRAQ in baseline SSE, so the sign-bit
    // extraction uses PCMPGTQ-against-zero (SSE4.2) — guarded.
    //
    // Verified encodings (aarch64-linux-gnu-as -march=armv8.2-a):
    //   sqdmull  v0.2d, v1.2s, v2.s[0]  = 0x0F82B020
    //   sqdmull2 v0.2d, v1.4s, v2.s[3]  = 0x4FA2B820
    //   sqdmlal  v0.2d, v1.2s, v2.s[0]  = 0x0F823020
    //   sqdmlal2 v0.2d, v1.4s, v2.s[2]  = 0x4F82B820
    //   sqdmlsl  v0.2d, v1.2s, v2.s[1]  = 0x0FA27020
    //   sqdmlsl2 v0.2d, v1.4s, v2.s[2]  = 0x4F827820
    if ((args.opcode == Op::kSqdmullIdx ||
         args.opcode == Op::kSqdmlalIdx ||
         args.opcode == Op::kSqdmlslIdx) &&
        args.size == 0b10) {
      if (!host_platform::kHasSSE4_1) { success_ = false; return; }
      const bool is_accum = (args.opcode != Op::kSqdmullIdx);
      const bool is_sub = (args.opcode == Op::kSqdmlslIdx);
      // Stage 2 needs PCMPGTQ for 64-bit sign-bit extraction (no PSRAQ
      // in baseline SSE).  Bail to interpreter on SSE4.1-only hosts.
      if (is_accum && !host_platform::kHasSSE4_2) {
        success_ = false; return;
      }

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xm = AllocTempSimdReg();
      SimdRegister xn_full = AllocTempSimdReg();
      SimdRegister x_const = AllocTempSimdReg();
      SimdRegister corner = AllocTempSimdReg();
      if (!success()) { return; }

      // Broadcast Vm.s[index] across all 4 dword lanes of xm.  PMULDQ
      // reads dword positions 0 and 2; we fill all four for both the
      // multiply and the 32-bit-lane PCMPEQD corner check.
      as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});
      {
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshufd(xm, xm, imm);
      }

      // x_const = 0x80000000 broadcast across 4 dwords.
      as_.Pcmpeqd(x_const, x_const);
      as_.Pslld(x_const, int8_t{31});

      // corner = (Vn dword == 0x80000000) AND (Vm dword == 0x80000000),
      // per 32-bit lane.  Compute corner_m in `corner` from the xm
      // broadcast first, then AND-in corner_n built from the full Vn.
      // x_const is dead after the second Pcmpeqd consumes it.
      as_.Movdqa(corner, xm);
      as_.Pcmpeqd(corner, x_const);
      as_.Movdqu(xn_full, {.base = Assembler::rbp, .disp = vn_off});
      as_.Pcmpeqd(xn_full, x_const);
      as_.Pand(corner, xn_full);

      // Q-select the corner mask's two relevant dwords and sign-extend
      // them to two 64-bit qword lanes.
      if (args.q) {
        as_.Psrldq(corner, int8_t{8});
      }
      as_.Pmovsxdq(corner, corner);

      // Widen Vn's Q-selected dword pair to two 64-bit signed lanes
      // via PMOVSXDQ from memory (vn_off + (q ? 8 : 0)).  This reads
      // the same 8 bytes used for the corner check, sign-extended into
      // the upper 32 bits of each 64-bit lane that PMULDQ ignores.
      int32_t vn_src_off = vn_off + (args.q ? 8 : 0);
      as_.Pmovsxdq(xn_full, {.base = Assembler::rbp, .disp = vn_src_off});

      // 32x32 → 64 signed multiply (PMULDQ reads dword 0 and 2 of each
      // operand) and double via self-add — corner lanes wrap to
      // INT64_MIN, which the XOR-blend below replaces with INT64_MAX.
      as_.Pmuldq(xn_full, xm);
      as_.Paddq(xn_full, xn_full);

      // x_const := INT64_MAX broadcast (Pcmpeqd self → -1, Psrlq 1 →
      // 0x7FFFFFFFFFFFFFFF), then XOR-blend over corner lanes:
      //   xn_full = doubled ^ ((doubled ^ INT64_MAX) & corner_mask).
      as_.Pcmpeqd(x_const, x_const);
      as_.Psrlq(x_const, int8_t{1});
      as_.Pxor(x_const, xn_full);
      as_.Pand(x_const, corner);
      as_.Pxor(xn_full, x_const);

      if (!is_accum) {
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn_full);
        return;
      }

      // Stage 2: 64-bit signed saturating add/sub of Vd and the
      // saturated doubled product.  Same wrap-add + overflow-bit-XOR-
      // blend recipe as the 32-bit stage-2 arm above, but at qword
      // granularity using PCMPGTQ-against-zero for sign extraction
      // (no PSRAQ in baseline SSE).
      //   sum  = a ± b              (PADDQ/PSUBQ wraps mod 2^64)
      //   ovf  = ~(a^b) & (a^sum)   for SQADD (sign bit indicates overflow)
      //        |  (a^b) & (a^diff)  for SQSUB
      //   sat  = (a < 0) ? INT64_MIN : INT64_MAX
      //   res  = sum ^ ((sum ^ sat) & ovf_mask)
      SimdRegister xd = AllocTempSimdReg();
      SimdRegister t_sum = AllocTempSimdReg();
      SimdRegister t_ovf = AllocTempSimdReg();
      SimdRegister t_sat = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
      as_.Movdqa(t_sum, xd);
      if (is_sub) {
        as_.Psubq(t_sum, xn_full);
      } else {
        as_.Paddq(t_sum, xn_full);
      }
      // t_ovf = a ^ b (for SQADD) or a ^ b (also used for SQSUB).
      as_.Movdqa(t_ovf, xd);
      as_.Pxor(t_ovf, xn_full);
      if (!is_sub) {
        // ~(a^b): rebuild -1 in x_const (the INT64_MAX broadcast is
        // dead now), XOR into t_ovf.
        as_.Pcmpeqd(x_const, x_const);
        as_.Pxor(t_ovf, x_const);
      }
      // t_sat := a ^ sum (or a ^ diff).
      as_.Movdqa(t_sat, xd);
      as_.Pxor(t_sat, t_sum);
      // t_ovf := overflow indicator in sign bit of each qword.
      as_.Pand(t_ovf, t_sat);
      // ovf_mask in x_const: x_const = (0 > t_ovf) per qword (SSE4.2).
      as_.Pxor(x_const, x_const);
      as_.Pcmpgtq(x_const, t_ovf);
      // a_sign_mask in t_ovf: t_ovf = (0 > xd) per qword.
      as_.Pxor(t_ovf, t_ovf);
      as_.Pcmpgtq(t_ovf, xd);
      // INT64_MAX broadcast in xn_full (dead after t_sum).
      as_.Pcmpeqd(xn_full, xn_full);
      as_.Psrlq(xn_full, int8_t{1});
      // sat = INT64_MAX XOR a_sign_mask: (a<0)? INT64_MIN : INT64_MAX.
      as_.Movdqa(t_sat, t_ovf);
      as_.Pxor(t_sat, xn_full);
      // res = sum ^ ((sum ^ sat) & ovf_mask).
      as_.Pxor(t_sat, t_sum);
      as_.Pand(t_sat, x_const);
      as_.Pxor(t_sum, t_sat);
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, t_sum);
      return;
    }

    if (args.opcode != Op::kFmla && args.opcode != Op::kFmls &&
        args.opcode != Op::kFmul && args.opcode != Op::kFmulx) {
      success_ = false;
      return;
    }
    // FP16 vector by-element FMLA/FMLS/FMUL (size=00).
    //
    // Armv8.2-FP16.  Decoder routes size=0b00 with U=0 and opcode
    // ∈ {0001 FMLA, 0101 FMLS, 1001 FMUL} here; FMULX FP16 by-element
    // does not exist (U=1 is rejected by the decoder at size=0b00).
    // Index encodes one of 8 FP16 lanes of Vm.8H; Q selects .4H or
    // .8H destination.
    //
    // FMUL: FP32 round-trip is exact — the product of two FP16
    // numbers has at most 22 mantissa bits, which fits in FP32's 24
    // exactly, and the final Vcvtps2ph applies one round to FP16.
    //
    // FMLA / FMLS: the interpreter performs the multiply-add in
    // binary64 (std::fma((double)a, (double)b, (double)d)) and
    // narrows once to FP16, so we promote the widened FP32 lanes to
    // FP64 and use VFMADD231PD / VFNMADD231PD — the same FP64
    // round-trip shape as the existing three-same FP16 FMLA/FMLS path.
    // An FP32-only round-trip would double-round and diverge from the
    // interpreter for some inputs.
    if (args.size == 0b00) {
      if (args.opcode != Op::kFmla && args.opcode != Op::kFmls &&
          args.opcode != Op::kFmul) {
        success_ = false; return;
      }
      const bool is_fma_fp16 =
          (args.opcode == Op::kFmla || args.opcode == Op::kFmls);
      const bool is_fmls_fp16 = (args.opcode == Op::kFmls);
      if (is_fma_fp16 && !host_platform::kHasFMA) { success_ = false; return; }
      if (!host_platform::kHasF16C) { success_ = false; return; }

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      // Broadcast Vm.h[index] to xm_f32 as 4 identical FP32 lanes.
      // Strategy: load Vm.8H into xm_f32, shift the high quad down
      // if index >= 4 (Psrldq 8), then Pshuflw with imm = (i:i:i:i)
      // (i = index & 3) replicates that FP16 lane into all 4 low
      // 16-bit lanes; Vcvtph2ps then widens to 4 identical FP32
      // lanes.  Both .8H passes reuse this broadcast.
      SimdRegister xm_f32 = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xm_f32, {.base = Assembler::rbp, .disp = vm_off});
      if (args.index >= 4) {
        as_.Psrldq(xm_f32, int8_t{8});
      }
      {
        const uint8_t i = args.index & 0b11;
        const int8_t imm =
            static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshuflw(xm_f32, xm_f32, imm);
      }
      as_.Vcvtph2ps(xm_f32, xm_f32);

      if (!is_fma_fp16) {
        // FMUL .4H / .8H — Mulps + Vcvtps2ph, no FP64 promotion.
        SimdRegister xn_f32 = AllocTempSimdReg();
        SimdRegister xres = AllocTempSimdReg();
        if (!success()) { return; }
        if (!args.q) {
          as_.Movq(xn_f32, {.base = Assembler::rbp, .disp = vn_off});
          as_.Vcvtph2ps(xn_f32, xn_f32);
          as_.Mulps(xn_f32, xm_f32);
          as_.Vcvtps2ph(xres, xn_f32, int8_t{0});
          // Vcvtps2ph auto-zeroes upper 64 bits.
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xres);
        } else {
          SimdRegister xlo = AllocTempSimdReg();
          SimdRegister xn_hi = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movq(xn_f32, {.base = Assembler::rbp, .disp = vn_off});
          as_.Vcvtph2ps(xn_f32, xn_f32);
          as_.Mulps(xn_f32, xm_f32);
          as_.Vcvtps2ph(xlo, xn_f32, int8_t{0});
          as_.Movdqu(xn_hi, {.base = Assembler::rbp, .disp = vn_off});
          as_.Psrldq(xn_hi, int8_t{8});
          as_.Vcvtph2ps(xn_f32, xn_hi);
          as_.Mulps(xn_f32, xm_f32);
          as_.Vcvtps2ph(xres, xn_f32, int8_t{0});
          as_.Pslldq(xres, int8_t{8});
          as_.Por(xlo, xres);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
        }
        return;
      }

      // FMLA / FMLS — FP16 -> FP32 -> FP64 round-trip, mirrors the
      // three-same FP16 FMLA/FMLS path.  xm_f32 already holds the
      // broadcast FP32 multiplier in all 4 lanes — both halves of
      // every FP64 pass read the same xm_pd.
      SimdRegister xn_f32 = AllocTempSimdReg();
      SimdRegister xd_f32 = AllocTempSimdReg();
      SimdRegister xn_pd = AllocTempSimdReg();
      SimdRegister xm_pd = AllocTempSimdReg();
      SimdRegister xd_pd = AllocTempSimdReg();
      SimdRegister xres = AllocTempSimdReg();
      SimdRegister xlo = args.q ? AllocTempSimdReg() : no_simd_register;
      if (xn_f32 == no_simd_register || xd_f32 == no_simd_register ||
          xn_pd == no_simd_register || xm_pd == no_simd_register ||
          xd_pd == no_simd_register || xres == no_simd_register ||
          (args.q && xlo == no_simd_register)) {
        success_ = false; return;
      }

      auto emit_quad = [&]() {
        // Pass 1: low 2 FP32 lanes -> 2 FP64.
        as_.Vcvtps2pd(xn_pd, xn_f32);
        as_.Vcvtps2pd(xm_pd, xm_f32);
        as_.Vcvtps2pd(xd_pd, xd_f32);
        if (is_fmls_fp16) as_.Vfnmadd231pd(xd_pd, xn_pd, xm_pd);
        else              as_.Vfmadd231pd(xd_pd, xn_pd, xm_pd);
        as_.Vcvtpd2ps(xres, xd_pd);

        // Pass 2: shift high 2 FP32 lanes down, FMA, recombine.
        // xm_f32's high lanes equal its low lanes (broadcast).
        as_.Psrldq(xn_f32, int8_t{8});
        as_.Vcvtps2pd(xn_pd, xn_f32);
        as_.Psrldq(xd_f32, int8_t{8});
        as_.Vcvtps2pd(xd_pd, xd_f32);
        if (is_fmls_fp16) as_.Vfnmadd231pd(xd_pd, xn_pd, xm_pd);
        else              as_.Vfmadd231pd(xd_pd, xn_pd, xm_pd);
        as_.Vcvtpd2ps(xn_f32, xd_pd);

        as_.Pslldq(xn_f32, int8_t{8});
        as_.Por(xres, xn_f32);
      };

      if (!args.q) {
        // .4H — 4 FP16 lanes (low 64 bits).
        as_.Movq(xn_f32, {.base = Assembler::rbp, .disp = vn_off});
        as_.Vcvtph2ps(xn_f32, xn_f32);
        as_.Movq(xd_f32, {.base = Assembler::rbp, .disp = vd_off});
        as_.Vcvtph2ps(xd_f32, xd_f32);
        emit_quad();
        as_.Vcvtps2ph(xres, xres, int8_t{0});
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xres);
      } else {
        // .8H — low quad + high quad, recombine via Pslldq + Por.
        as_.Movq(xn_f32, {.base = Assembler::rbp, .disp = vn_off});
        as_.Vcvtph2ps(xn_f32, xn_f32);
        as_.Movq(xd_f32, {.base = Assembler::rbp, .disp = vd_off});
        as_.Vcvtph2ps(xd_f32, xd_f32);
        emit_quad();
        as_.Vcvtps2ph(xlo, xres, int8_t{0});

        as_.Movdqu(xn_f32, {.base = Assembler::rbp, .disp = vn_off});
        as_.Psrldq(xn_f32, int8_t{8});
        as_.Vcvtph2ps(xn_f32, xn_f32);
        as_.Movdqu(xd_f32, {.base = Assembler::rbp, .disp = vd_off});
        as_.Psrldq(xd_f32, int8_t{8});
        as_.Vcvtph2ps(xd_f32, xd_f32);
        emit_quad();
        as_.Vcvtps2ph(xres, xres, int8_t{0});

        as_.Pslldq(xres, int8_t{8});
        as_.Por(xlo, xres);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
      }
      return;
    }
    if (args.size != 0b10 && args.size != 0b11) {
      success_ = false;
      return;
    }
    const bool is_double = (args.size == 0b11);
    if (is_double && !args.q) { success_ = false; return; }

    const bool needs_fma =
        (args.opcode == Op::kFmla || args.opcode == Op::kFmls);
    if (needs_fma && !host_platform::kHasFMA) {
      success_ = false;
      return;
    }

    int32_t vn_off = VRegOffset(args.rn);
    int32_t vm_off = VRegOffset(args.rm);
    int32_t vd_off = VRegOffset(args.rd);

    SimdRegister xmm_n = AllocTempSimdReg();
    SimdRegister xmm_m = AllocTempSimdReg();
    if (!success()) { return; }

    as_.Movdqu(xmm_n, {.base = Assembler::rbp, .disp = vn_off});
    as_.Movdqu(xmm_m, {.base = Assembler::rbp, .disp = vm_off});

    // Broadcast lane args.index of xmm_m across all lanes.
    if (is_double) {
      // FP64: index 0 -> 0x44 (lanes 0,1,0,1 -> [lo,hi,lo,hi] copies low qword);
      //       index 1 -> 0xEE (lanes 2,3,2,3 -> copies high qword).
      const int8_t imm = (args.index == 0) ? int8_t{0x44} : int8_t{static_cast<int8_t>(0xEEu)};
      as_.Pshufd(xmm_m, xmm_m, imm);
    } else {
      // FP32: replicate the same 2-bit field 4 times.
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      as_.Pshufd(xmm_m, xmm_m, imm);
    }

    SimdRegister xmm_result = no_simd_register;
    if (args.opcode == Op::kFmul) {
      if (is_double) as_.Mulpd(xmm_n, xmm_m);
      else            as_.Mulps(xmm_n, xmm_m);
      xmm_result = xmm_n;
    } else if (args.opcode == Op::kFmulx) {
      // FMULX = FMUL except (±0 * ±inf) lanes return ±2.0 (sign =
      // sign(a) XOR sign(b)).  Direct lift of the three-same FMULX
      // saturation override (handoff #120) on top of the broadcasted Vm.
      //
      //   mul         = a * broadcast_b
      //   mul_unord   = cmpunord(mul, mul)        (-1 per lane if mul is NaN)
      //   input_unord = cmpunord(a, broadcast_b)  (-1 per lane if a or b NaN)
      //   special     = mul_unord AND NOT input_unord
      //   sign_mask   = PCMPEQD + PSLLD/Q  -> 0x80000000... per lane
      //   two_signed  = ((a XOR broadcast_b) AND sign_mask) OR bits(+2.0)
      //   result      = (mul AND NOT special) OR (two_signed AND special)
      SimdRegister xmm_mul = AllocTempSimdReg();
      SimdRegister xmm_mul_unord = AllocTempSimdReg();
      SimdRegister xmm_input_unord = AllocTempSimdReg();
      SimdRegister xmm_two = AllocTempSimdReg();
      if (!success()) { return; }

      // mul = a * broadcast_b
      as_.Movdqa(xmm_mul, xmm_n);
      if (is_double) as_.Mulpd(xmm_mul, xmm_m);
      else            as_.Mulps(xmm_mul, xmm_m);

      // mul_unord = cmpunord(mul, mul)
      as_.Movdqa(xmm_mul_unord, xmm_mul);
      if (is_double) as_.Cmpunordpd(xmm_mul_unord, xmm_mul_unord);
      else            as_.Cmpunordps(xmm_mul_unord, xmm_mul_unord);

      // input_unord = cmpunord(a, broadcast_b)
      as_.Movdqa(xmm_input_unord, xmm_n);
      if (is_double) as_.Cmpunordpd(xmm_input_unord, xmm_m);
      else            as_.Cmpunordps(xmm_input_unord, xmm_m);

      // special_mask = mul_unord AND NOT input_unord (Pandn writes
      // (NOT dst) AND src, so result lands in xmm_input_unord).
      as_.Pandn(xmm_input_unord, xmm_mul_unord);

      // two_signed: build ((a XOR broadcast_b) AND sign_mask) OR bits(+2.0).
      // Reuse xmm_n as the XOR result; reuse xmm_mul_unord as the sign mask.
      if (is_double) as_.Xorpd(xmm_n, xmm_m);
      else            as_.Xorps(xmm_n, xmm_m);
      as_.Pcmpeqd(xmm_mul_unord, xmm_mul_unord);
      if (is_double) as_.Psllq(xmm_mul_unord, int8_t{63});
      else            as_.Pslld(xmm_mul_unord, int8_t{31});
      as_.Pand(xmm_n, xmm_mul_unord);

      // Broadcast bits of +2.0 into all lanes of xmm_two.
      Register tmp_gpr = AllocTempReg();
      if (tmp_gpr == Assembler::no_register) { return; }
      if (is_double) {
        as_.Movq(tmp_gpr, int64_t{0x4000000000000000LL});
        as_.Movq(xmm_two, tmp_gpr);
        as_.Punpcklqdq(xmm_two, xmm_two);
      } else {
        as_.Movl(tmp_gpr, int32_t{0x40000000});
        as_.Movd(xmm_two, tmp_gpr);
        as_.Pshufd(xmm_two, xmm_two, int8_t{0});
      }
      as_.Por(xmm_n, xmm_two);
      // xmm_n now holds ±2.0 per lane.

      // Blend: result = (mul AND NOT special) OR (±2.0 AND special).
      // Reuse xmm_m as the masked ±2.0; reuse xmm_input_unord as the result.
      as_.Movdqa(xmm_m, xmm_n);
      as_.Pand(xmm_m, xmm_input_unord);
      as_.Pandn(xmm_input_unord, xmm_mul);
      as_.Por(xmm_input_unord, xmm_m);

      xmm_result = xmm_input_unord;
    } else {
      SimdRegister xmm_d = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xmm_d, {.base = Assembler::rbp, .disp = vd_off});
      if (args.opcode == Op::kFmla) {
        if (is_double) as_.Vfmadd231pd(xmm_d, xmm_n, xmm_m);
        else            as_.Vfmadd231ps(xmm_d, xmm_n, xmm_m);
      } else {
        // FMLS:  Vd = Vd + (-Vn)*Vm  (single fused rounding).
        if (is_double) as_.Vfnmadd231pd(xmm_d, xmm_n, xmm_m);
        else            as_.Vfnmadd231ps(xmm_d, xmm_n, xmm_m);
      }
      xmm_result = xmm_d;
    }

    if (!args.q) {
      // Zero upper 64 bits (D-register semantics).
      as_.Pslldq(xmm_result, int8_t{8});
      as_.Psrldq(xmm_result, int8_t{8});
    }
    as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_result);
  }

  // AdvSIMD scalar x indexed element — JIT lowering for FMULX scalar by
  // element only.  Same saturation shape as the three-same FMULX JIT
  // (handoff #120) and the vector by-element FMULX (handoff #124), but
  // applied to a single lane; the upper lanes of Vd are always zeroed
  // (scalar destination semantics).
  void AdvSimdScalarXIndexedElement(const Decoder::AdvSimdScalarXIdxArgs& args) {
    using Op = Decoder::AdvSimdScalarXIdxOpcode;
    // SQRDMLAH/SQRDMLSH scalar by-element (Armv8.1-RDM).
    // Same single-lane saturation recipe as the scalar three-same form,
    // but reads Vm.lane[index] instead of Vm.lane[0].
    if (args.opcode == Op::kSqrdmlahScalarIdx ||
        args.opcode == Op::kSqrdmlshScalarIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        success_ = false; return;
      }
      const bool is_sub = (args.opcode == Op::kSqrdmlshScalarIdx);
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
        const int32_t vm_lane_off = vm_off + static_cast<int32_t>(args.index) * 2;
        as_.Pxor(xn, xn);
        as_.Pinsrw(xn, {.base = Assembler::rbp, .disp = vn_off}, int8_t{0});
        as_.Pxor(xm, xm);
        as_.Pinsrw(xm, {.base = Assembler::rbp, .disp = vm_lane_off}, int8_t{0});
        as_.Pxor(xd, xd);
        as_.Pinsrw(xd, {.base = Assembler::rbp, .disp = vd_off}, int8_t{0});
        as_.Pcmpeqw(x_min, x_min);
        as_.Psllw(x_min, int8_t{15});
        as_.Movdqa(xn_corner, xn);
        as_.Movdqa(xm_corner, xm);
        as_.Pmulhrsw(xn, xm);
        as_.Pcmpeqw(xn_corner, x_min);
        as_.Pcmpeqw(xm_corner, x_min);
        as_.Pand(xn_corner, xm_corner);
        as_.Pxor(xn, xn_corner);
        if (is_sub) {
          as_.Psubsw(xd, xn);
        } else {
          as_.Paddsw(xd, xn);
        }
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
      const int32_t vm_lane_off = vm_off + static_cast<int32_t>(args.index) * 4;
      as_.Movd(xn, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movd(xm, {.base = Assembler::rbp, .disp = vm_lane_off});
      as_.Movd(xd, {.base = Assembler::rbp, .disp = vd_off});
      as_.Pcmpeqd(x_const, x_const);
      as_.Pslld(x_const, int8_t{31});
      as_.Movdqa(corner, xn);
      as_.Pcmpeqd(corner, x_const);
      as_.Movdqa(xp, xm);
      as_.Pcmpeqd(xp, x_const);
      as_.Pand(corner, xp);
      as_.Movdqa(xp, xn);
      as_.Pmuldq(xp, xm);
      as_.Psllq(xp, int8_t{1});
      as_.Pcmpeqd(x_const, x_const);
      as_.Psllq(x_const, int8_t{63});
      as_.Psrlq(x_const, int8_t{32});
      as_.Paddq(xp, x_const);
      as_.Psrlq(xp, int8_t{32});
      as_.Pxor(xp, corner);
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
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
      return;
    }
    if (args.opcode != Op::kFmulx && args.opcode != Op::kFmul &&
        args.opcode != Op::kFmla && args.opcode != Op::kFmls) {
      success_ = false;
      return;
    }
    if (args.size != 0b00 && args.size != 0b10 && args.size != 0b11) {
      success_ = false;
      return;
    }
    const bool needs_fma = (args.opcode == Op::kFmla || args.opcode == Op::kFmls);
    if (needs_fma && !host_platform::kHasFMA) {
      success_ = false;
      return;
    }
    const bool is_double = (args.size == 0b11);

    // Armv8.2-FP16 scalar by-element FMLA/FMLS/FMUL/FMULX.
    //
    // Lift each FP16 source lane to FP32 lane 0 via Pxor + Pinsrw + Vcvtph2ps
    // (same scalar-lift recipe as AdvSimdScalarThreeSame FP16).  For
    // FMLA/FMLS, additionally promote to FP64 so the multiply-add uses a
    // single binary64 rounding (matches the interpreter's FpHalfToSingle ->
    // std::fma((double)... narrow path).  Result narrows back via Vcvtps2ph
    // (real-FP output, NOT a mask), which auto-zeroes the upper 64 bits of
    // the dst xmm; a full 128-bit Movdqu then writes Vd[15:0]=FP16 result
    // with Vd[127:16]=0, matching the AArch64 scalar Hd zero-extend.
    //
    // F16C absence falls back to interpreter (success_ = false).  FMA
    // absence also falls back (FMA is already required above for FMLA/FMLS
    // in the FP32/FP64 path).
    if (args.size == 0b00) {
      if (!host_platform::kHasF16C) { success_ = false; return; }

      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xmm_n = AllocTempSimdReg();
      SimdRegister xmm_m = AllocTempSimdReg();
      if (!success()) { return; }
      // Lift Vn.h[0] and Vm.h[index] to FP32 in xmm lane 0.  Pinsrw with a
      // memory operand reads 16 bits from the offset; the upper bytes of
      // each XMM are zeroed first.
      as_.Pxor(xmm_n, xmm_n);
      as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = vn_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_n, xmm_n);
      as_.Pxor(xmm_m, xmm_m);
      as_.Pinsrw(xmm_m,
                 {.base = Assembler::rbp,
                  .disp = vm_off + static_cast<int32_t>(args.index) * 2},
                 int8_t{0});
      as_.Vcvtph2ps(xmm_m, xmm_m);

      SimdRegister xmm_result = no_simd_register;
      if (args.opcode == Op::kFmul) {
        as_.Mulss(xmm_n, xmm_m);
        xmm_result = xmm_n;
      } else if (args.opcode == Op::kFmulx) {
        // FMULX FP16 saturation: same shape as FP32 scalar FMULX three-same
        // FP16.  mul = a*b; if mul is NaN and neither input was NaN
        // (the (±0,±inf) case), return ±2.0 with sign = sign(a) XOR sign(b).
        // FP16 ±2.0 = 0x4000 / 0xC000 round-trips exactly through FP32 ±2.0
        // = 0x40000000 / 0xC0000000 via Vcvtps2ph.
        SimdRegister xmm_mul = AllocTempSimdReg();
        SimdRegister xmm_mul_unord = AllocTempSimdReg();
        SimdRegister xmm_iu = AllocTempSimdReg();
        SimdRegister xmm_two = AllocTempSimdReg();
        if (!success()) { return; }

        // mul = a * b
        as_.Movdqa(xmm_mul, xmm_n);
        as_.Mulss(xmm_mul, xmm_m);

        // mul_unord = cmpunord(mul, mul)
        as_.Movdqa(xmm_mul_unord, xmm_mul);
        as_.Cmpunordps(xmm_mul_unord, xmm_mul_unord);

        // input_unord = cmpunord(a, b)
        as_.Movdqa(xmm_iu, xmm_n);
        as_.Cmpunordps(xmm_iu, xmm_m);

        // special_mask = (NOT input_unord) AND mul_unord, lands in xmm_iu.
        as_.Pandn(xmm_iu, xmm_mul_unord);

        // two_signed = ((a XOR b) AND sign_mask) OR bits-of(+2.0).
        as_.Xorps(xmm_n, xmm_m);
        as_.Pcmpeqd(xmm_mul_unord, xmm_mul_unord);
        as_.Pslld(xmm_mul_unord, int8_t{31});
        as_.Pand(xmm_n, xmm_mul_unord);

        Register tmp_gpr = AllocTempReg();
        if (tmp_gpr == Assembler::no_register) { return; }
        as_.Movl(tmp_gpr, int32_t{0x40000000});
        as_.Movd(xmm_two, tmp_gpr);
        as_.Por(xmm_n, xmm_two);

        // Blend: result = (mul AND NOT special) OR (±2.0 AND special).
        as_.Movdqa(xmm_m, xmm_n);
        as_.Pand(xmm_m, xmm_iu);
        as_.Pandn(xmm_iu, xmm_mul);
        as_.Por(xmm_iu, xmm_m);
        xmm_result = xmm_iu;
      } else {
        // FMLA / FMLS: promote to FP64 so the multiply-add rounds once.
        SimdRegister xmm_n_pd = AllocTempSimdReg();
        SimdRegister xmm_m_pd = AllocTempSimdReg();
        SimdRegister xmm_d_pd = AllocTempSimdReg();
        SimdRegister xmm_d_f32 = AllocTempSimdReg();
        if (!success()) { return; }
        // Lift Vd.h[0] to FP32 lane 0 via the same Pinsrw + Vcvtph2ps path.
        as_.Pxor(xmm_d_f32, xmm_d_f32);
        as_.Pinsrw(xmm_d_f32, {.base = Assembler::rbp, .disp = vd_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_d_f32, xmm_d_f32);

        // Widen lane 0 of each operand to FP64 (Vcvtps2pd reads low 2 FP32
        // lanes; we only care about lane 0).
        as_.Vcvtps2pd(xmm_n_pd, xmm_n);
        as_.Vcvtps2pd(xmm_m_pd, xmm_m);
        as_.Vcvtps2pd(xmm_d_pd, xmm_d_f32);
        if (args.opcode == Op::kFmls) {
          as_.Vfnmadd231sd(xmm_d_pd, xmm_n_pd, xmm_m_pd);
        } else {
          as_.Vfmadd231sd(xmm_d_pd, xmm_n_pd, xmm_m_pd);
        }
        // Narrow FP64 -> FP32 in lane 0; xmm_n is free as the dst.
        as_.Vcvtpd2ps(xmm_n, xmm_d_pd);
        xmm_result = xmm_n;
      }

      // Narrow FP32 lane 0 to FP16 word lane 0; Vcvtps2ph zero-fills the
      // upper 64 bits of dst.  Lanes 1..3 of xmm_result were FP32 +0 by
      // Vcvtph2ps zero-fill (preserved through the scalar primitives, since
      // FMULX's PSLLD/Pcmpeqd-with-itself/Pand all leave dword lanes 1..3
      // at zero, and Mulss/Vcvtpd2ps only write lane 0).  Word lanes 1..7
      // of the narrowed xmm_result are therefore 0.
      as_.Vcvtps2ph(xmm_result, xmm_result, int8_t{0});
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_result);
      return;
    }

    // FMUL / FMLA / FMLS — simpler scalar lowering without the FMULX
    // saturation shape.  Load Vn/Vm into XMM regs, broadcast the indexed
    // lane of Vm into lane 0 (Pshufd works regardless of the upper-lane
    // garbage we'll later zero), then issue the scalar SS/SD instruction.
    if (args.opcode != Op::kFmulx) {
      int32_t vn_off = VRegOffset(args.rn);
      int32_t vm_off = VRegOffset(args.rm);
      int32_t vd_off = VRegOffset(args.rd);

      SimdRegister xmm_n = AllocTempSimdReg();
      SimdRegister xmm_m = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xmm_n, {.base = Assembler::rbp, .disp = vn_off});
      as_.Movdqu(xmm_m, {.base = Assembler::rbp, .disp = vm_off});

      // Move the indexed lane of Vm into lane 0.  Use Pshufd to broadcast
      // (the upper lanes are about to be zeroed anyway).
      if (is_double) {
        const int8_t imm = (args.index == 0) ? int8_t{0x44} : int8_t{static_cast<int8_t>(0xEEu)};
        as_.Pshufd(xmm_m, xmm_m, imm);
      } else {
        const uint8_t i = args.index & 0b11;
        const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        as_.Pshufd(xmm_m, xmm_m, imm);
      }

      SimdRegister xmm_result = no_simd_register;
      if (args.opcode == Op::kFmul) {
        // Scalar multiply: lane 0 of xmm_n = Vn.lane0 * Vm.lane[index].
        // Upper lanes of xmm_n still hold Vn — zero them below.
        if (is_double) as_.Mulsd(xmm_n, xmm_m);
        else           as_.Mulss(xmm_n, xmm_m);
        xmm_result = xmm_n;
      } else {
        // FMLA / FMLS: Vd.lane0 = Vd.lane0 ± Vn.lane0 * Vm.lane[index].
        SimdRegister xmm_d = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xmm_d, {.base = Assembler::rbp, .disp = vd_off});
        if (args.opcode == Op::kFmla) {
          if (is_double) as_.Vfmadd231sd(xmm_d, xmm_n, xmm_m);
          else           as_.Vfmadd231ss(xmm_d, xmm_n, xmm_m);
        } else {
          // FMLS: Vd = Vd + (-Vn)*Vm  (single fused rounding).
          if (is_double) as_.Vfnmadd231sd(xmm_d, xmm_n, xmm_m);
          else           as_.Vfnmadd231ss(xmm_d, xmm_n, xmm_m);
        }
        xmm_result = xmm_d;
      }

      // Scalar destination: zero the upper lanes of Vd.
      if (is_double) {
        as_.Pslldq(xmm_result, int8_t{8});
        as_.Psrldq(xmm_result, int8_t{8});
      } else {
        as_.Pslldq(xmm_result, int8_t{12});
        as_.Psrldq(xmm_result, int8_t{12});
      }
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_result);
      return;
    }

    // FMULX path follows below — broadcast shape + saturation override.

    int32_t vn_off = VRegOffset(args.rn);
    int32_t vm_off = VRegOffset(args.rm);
    int32_t vd_off = VRegOffset(args.rd);

    SimdRegister xmm_n = AllocTempSimdReg();
    SimdRegister xmm_m = AllocTempSimdReg();
    if (!success()) { return; }

    as_.Movdqu(xmm_n, {.base = Assembler::rbp, .disp = vn_off});
    as_.Movdqu(xmm_m, {.base = Assembler::rbp, .disp = vm_off});

    // Broadcast lane args.index of xmm_m across all lanes — same layout as
    // the vector by-element FMULX path.  The broadcast lets us reuse the
    // FMULX saturation shape verbatim; we only need to zero the upper
    // lanes of the result afterwards.
    if (is_double) {
      const int8_t imm = (args.index == 0) ? int8_t{0x44} : int8_t{static_cast<int8_t>(0xEEu)};
      as_.Pshufd(xmm_m, xmm_m, imm);
    } else {
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      as_.Pshufd(xmm_m, xmm_m, imm);
    }

    // FMULX saturation: mul = a*b; if mul is NaN and neither input was NaN
    // (the (±0,±inf) case), return ±2.0 with sign = sign(a)^sign(b).
    SimdRegister xmm_mul = AllocTempSimdReg();
    SimdRegister xmm_mul_unord = AllocTempSimdReg();
    SimdRegister xmm_input_unord = AllocTempSimdReg();
    SimdRegister xmm_two = AllocTempSimdReg();
    if (!success()) { return; }

    // mul = a * broadcast_b
    as_.Movdqa(xmm_mul, xmm_n);
    if (is_double) as_.Mulpd(xmm_mul, xmm_m);
    else           as_.Mulps(xmm_mul, xmm_m);

    // mul_unord = cmpunord(mul, mul)
    as_.Movdqa(xmm_mul_unord, xmm_mul);
    if (is_double) as_.Cmpunordpd(xmm_mul_unord, xmm_mul_unord);
    else           as_.Cmpunordps(xmm_mul_unord, xmm_mul_unord);

    // input_unord = cmpunord(a, broadcast_b)
    as_.Movdqa(xmm_input_unord, xmm_n);
    if (is_double) as_.Cmpunordpd(xmm_input_unord, xmm_m);
    else           as_.Cmpunordps(xmm_input_unord, xmm_m);

    // special_mask = mul_unord AND NOT input_unord.
    as_.Pandn(xmm_input_unord, xmm_mul_unord);

    // two_signed: ((a XOR broadcast_b) AND sign_mask) OR bits(+2.0).
    if (is_double) as_.Xorpd(xmm_n, xmm_m);
    else           as_.Xorps(xmm_n, xmm_m);
    as_.Pcmpeqd(xmm_mul_unord, xmm_mul_unord);
    if (is_double) as_.Psllq(xmm_mul_unord, int8_t{63});
    else           as_.Pslld(xmm_mul_unord, int8_t{31});
    as_.Pand(xmm_n, xmm_mul_unord);

    Register tmp_gpr = AllocTempReg();
    if (tmp_gpr == Assembler::no_register) { return; }
    if (is_double) {
      as_.Movq(tmp_gpr, int64_t{0x4000000000000000LL});
      as_.Movq(xmm_two, tmp_gpr);
      as_.Punpcklqdq(xmm_two, xmm_two);
    } else {
      as_.Movl(tmp_gpr, int32_t{0x40000000});
      as_.Movd(xmm_two, tmp_gpr);
      as_.Pshufd(xmm_two, xmm_two, int8_t{0});
    }
    as_.Por(xmm_n, xmm_two);

    // Blend: result = (mul AND NOT special) OR (±2.0 AND special).
    as_.Movdqa(xmm_m, xmm_n);
    as_.Pand(xmm_m, xmm_input_unord);
    as_.Pandn(xmm_input_unord, xmm_mul);
    as_.Por(xmm_input_unord, xmm_m);

    SimdRegister xmm_result = xmm_input_unord;

    // Scalar destination: keep only the low lane and zero the rest.
    // FP32: low 4 bytes -> low 32 bits of Vd; FP64: low 8 bytes -> low
    // 64 bits of Vd.
    if (is_double) {
      // PSLLDQ 8 + PSRLDQ 8 zeroes the upper 64 bits.
      as_.Pslldq(xmm_result, int8_t{8});
      as_.Psrldq(xmm_result, int8_t{8});
    } else {
      // PSLLDQ 12 + PSRLDQ 12 zeroes the upper 96 bits.
      as_.Pslldq(xmm_result, int8_t{12});
      as_.Psrldq(xmm_result, int8_t{12});
    }
    as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_result);
  }

