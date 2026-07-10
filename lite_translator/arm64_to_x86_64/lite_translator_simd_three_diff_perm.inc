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
// Family: SIMD three-different-register, extract, permute, table.
// Methods:
//   AdvSimdThreeDiff, AdvSimdExtract, AdvSimdPermute, AdvSimdTableLookup

  void AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& args) {
    // JIT lowering for the widening multiply-and-(add|sub|just-store) family:
    //   {S,U}MULL{,2}, {S,U}MLAL{,2}, {S,U}MLSL{,2}
    // at all three input sizes (8/16/32) and both Q=0 (low half of Vn/Vm) and
    // Q=1 ("2" variants — upper half of Vn/Vm). The result vector always fills
    // 128 bits (8H/4S/2D). Other ThreeDiff ops (SADDL, USUBL, SABDL, SABAL,
    // SADDW, etc., and the polynomial PMULL) still bail to the interpreter.
    //
    // Widening recipe (per size):
    //   size=00 (8b→16b): PMOVSXBW / PMOVZXBW + PMULLW       (8 lanes, fills 128)
    //   size=01 (16b→32b): PMOVSXWD / PMOVZXWD + PMULLD       (4 lanes, fills 128)
    //   size=10 (32b→64b): PMOVSXDQ / PMOVZXDQ + PMULDQ / PMULUDQ (2 lanes)
    //     The PMOVSXDQ/PMOVZXDQ widening places the two 32-bit source dwords
    //     into the low dword of each qword, which is exactly the input shape
    //     PMULDQ/PMULUDQ wants — they multiply the low dword of each qword
    //     of src1 against the low dword of each qword of src2 and produce
    //     two qword products at positions [0..7] and [8..15].
    using Op = Decoder::AdvSimdThreeDiffOpcode;

    // PMULL64 (size=11) lowers to a single PCLMULQDQ — the imm-named
    // Pclmullqlqdq for Q=0 (poly_mul64(Vn.D[0], Vm.D[0])) and Pclmulhqhqdq
    // for Q=1 (PMULL2; poly_mul64(Vn.D[1], Vm.D[1])).
    //
    // PMULL.8H (size=00) is 8 independent 8-bit polynomial products that
    // widen to full 16-bit results.  Widen both 8-byte sources to 16-bit
    // lanes (PMOVZXBW; Q=1/PMULL2 takes bytes 8..15), then run the per-bit
    // carry-less shift-and-XOR in 16-bit lanes.  Because every lane already
    // holds a value < 256 and the degree-14 product fits in 16 bits, the
    // PSLLW shifts never spill across lanes, so no per-lane masking is
    // needed (unlike the byte-truncated PMUL).  The bit selector for bit i
    // is built by PSLLW(b, 15-i) then PSRAW 15, broadcasting bit i to the
    // whole lane.  size in {01,10} is reserved for PMULL and bails.
    if (args.opcode == Op::kPmull) {
      const int32_t vn_off_pmull = VRegOffset(args.rn);
      const int32_t vm_off_pmull = VRegOffset(args.rm);
      const int32_t vd_off_pmull = VRegOffset(args.rd);
      if (args.size == 0b11) {
        SimdRegister xn_p = AllocTempSimdReg();
        SimdRegister xm_p = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn_p, {.base = Assembler::rbp, .disp = vn_off_pmull});
        as_.Movdqu(xm_p, {.base = Assembler::rbp, .disp = vm_off_pmull});
        if (args.q) {
          as_.Pclmulhqhqdq(xn_p, xm_p);
        } else {
          as_.Pclmullqlqdq(xn_p, xm_p);
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_pmull}, xn_p);
        return;
      }
      if (args.size == 0b00) {  // PMULL/PMULL2 .8H (poly8 widening)
        const int32_t extra = args.q ? 8 : 0;
        SimdRegister za = AllocTempSimdReg();
        SimdRegister zb = AllocTempSimdReg();
        SimdRegister acc = AllocTempSimdReg();
        SimdRegister shifted = AllocTempSimdReg();
        SimdRegister sel = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movq(za, {.base = Assembler::rbp, .disp = vn_off_pmull + extra});
        as_.Movq(zb, {.base = Assembler::rbp, .disp = vm_off_pmull + extra});
        as_.Pmovzxbw(za, za);  // 8 bytes -> 8x16-bit lanes
        as_.Pmovzxbw(zb, zb);
        as_.Pxor(acc, acc);
        for (int i = 0; i < 8; ++i) {
          as_.Movdqa(shifted, za);
          if (i != 0) as_.Psllw(shifted, static_cast<int8_t>(i));
          // sel = 0xFFFF per lane where bit i of zb is set.
          as_.Movdqa(sel, zb);
          if (15 - i != 0) as_.Psllw(sel, static_cast<int8_t>(15 - i));
          as_.Psraw(sel, int8_t{15});
          as_.Pand(shifted, sel);
          as_.Pxor(acc, shifted);
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_pmull}, acc);
        return;
      }
      Undefined();  // size in {01,10} reserved
      return;
    }

    // ADDHN/SUBHN/RADDHN/RSUBHN (narrowing high half).
    // Vd(narrow) = ((Vn op Vm) [+ round]) >> narrow_bits, taking the high
    // half of each wide lane.  R-variants add round = 1<<(narrow_bits-1).
    // size=00 (.8B<-.8H), 01 (.4H<-.4S) use a narrowing pack; size=10
    // (.2S<-.2D) has no 64->32 pack, so it shifts each 64-bit lane down by 32
    // and gathers the dwords with PSHUFD.  Q=1 ("2" form) writes the upper 64
    // bits and preserves Vd's low 64.
    if (args.opcode == Op::kAddhn || args.opcode == Op::kSubhn ||
        args.opcode == Op::kRaddhn || args.opcode == Op::kRsubhn) {
      if (args.size > 0b10) { success_ = false; return; }
      const bool is_sub = (args.opcode == Op::kSubhn || args.opcode == Op::kRsubhn);
      const bool is_round = (args.opcode == Op::kRaddhn || args.opcode == Op::kRsubhn);
      const int32_t vn_o = VRegOffset(args.rn);
      const int32_t vm_o = VRegOffset(args.rm);
      const int32_t vd_o = VRegOffset(args.rd);
      SimdRegister xn = AllocTempSimdReg();
      SimdRegister xm = AllocTempSimdReg();
      SimdRegister xz = AllocTempSimdReg();
      if (!success()) { return; }
      as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_o});
      as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_o});
      as_.Pxor(xz, xz);
      auto add_round = [&](int64_t pat) -> bool {
        SimdRegister xr = AllocTempSimdReg();
        Register t = AllocTempReg();
        if (xr == no_simd_register || t == no_register) return false;
        as_.Movq(t, pat);
        as_.Movq(xr, t);
        as_.Punpcklqdq(xr, xr);
        if (args.size == 0b00) as_.Paddw(xn, xr); else as_.Paddd(xn, xr);
        return true;
      };
      if (args.size == 0b10) {
        // .2S<-.2D / .4S<-.2D(2): 64-bit add/sub, then take the high 32 bits of
        // each lane (no x86 64->32 pack, so PSRLQ then gather with PSHUFD).
        if (is_sub) as_.Psubq(xn, xm); else as_.Paddq(xn, xm);
        if (is_round) {
          // Rounding adds 2^31 into each 64-bit lane before the >>32 truncate.
          SimdRegister xr = AllocTempSimdReg();
          Register t = AllocTempReg();
          if (xr == no_simd_register || t == no_register) { success_ = false; return; }
          as_.Movq(t, int64_t{0x0000000080000000LL});
          as_.Movq(xr, t);
          as_.Punpcklqdq(xr, xr);
          as_.Paddq(xn, xr);
        }
        as_.Psrlq(xn, int8_t{32});         // high 32 of each lane -> low 32
        as_.Pshufd(xn, xn, int8_t{0x08});  // {dword0, dword2} -> low 64
        if (!args.q) {
          as_.Pslldq(xn, int8_t{8});       // zero upper 64
          as_.Psrldq(xn, int8_t{8});
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_o}, xn);
        } else {
          as_.Pslldq(xn, int8_t{8});
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_o});
          as_.Pslldq(xd, int8_t{8});
          as_.Psrldq(xd, int8_t{8});
          as_.Por(xn, xd);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_o}, xn);
        }
        return;
      }
      if (args.size == 0b00) {
        if (is_sub) as_.Psubw(xn, xm); else as_.Paddw(xn, xm);
        if (is_round && !add_round(int64_t{0x0080008000800080LL})) { success_ = false; return; }
        as_.Psrlw(xn, int8_t{8});
        as_.Packuswb(xn, xz);
      } else {  // size == 01
        if (is_sub) as_.Psubd(xn, xm); else as_.Paddd(xn, xm);
        if (is_round && !add_round(int64_t{0x0000800000008000LL})) { success_ = false; return; }
        as_.Psrld(xn, int8_t{16});
        as_.Packusdw(xn, xz);
      }
      if (!args.q) {
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_o}, xn);
      } else {
        as_.Pslldq(xn, int8_t{8});
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_o});
        as_.Pslldq(xd, int8_t{8});  // mask Vd low 64
        as_.Psrldq(xd, int8_t{8});
        as_.Por(xn, xd);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_o}, xn);
      }
      return;
    }

    // Widening add/sub family — same widen-then-binop pattern as the
    // multiply family below:
    //   SADDL/UADDL/SSUBL/USUBL — widen both Vn and Vm, then add/sub
    //                              at the wide lane.
    //   SADDW/UADDW/SSUBW/USUBW — Vn is already wide (loaded as 128b),
    //                              widen only Vm, then add/sub.
    //   SABDL/UABDL              — widen both, compute abs diff at the
    //                              wider lane width.  size=00/01 uses
    //                              max(a,b)-min(a,b) (PMAXS*/PMINS* for
    //                              signed, PMAXU*/PMINU* for unsigned).
    //                              size=10 (32→64) lacks SSE 64-bit
    //                              max/min, so it instead does Psubq
    //                              followed by a Pcmpgtq-against-zero
    //                              signed-abs (mask = (0 > diff) per
    //                              qword; abs = (diff ^ mask) - mask).
    //   SABAL/UABAL              — same abs diff, then accumulate into Vd.
    // All size/Q/sign combinations are JIT-lowered.
    {
      const bool is_addl = (args.opcode == Op::kSaddl || args.opcode == Op::kUaddl);
      const bool is_subl = (args.opcode == Op::kSsubl || args.opcode == Op::kUsubl);
      const bool is_addw = (args.opcode == Op::kSaddw || args.opcode == Op::kUaddw);
      const bool is_subw = (args.opcode == Op::kSsubw || args.opcode == Op::kUsubw);
      const bool is_abdl = (args.opcode == Op::kSabdl || args.opcode == Op::kUabdl);
      const bool is_abal = (args.opcode == Op::kSabal || args.opcode == Op::kUabal);

      if (is_addl || is_subl || is_addw || is_subw || is_abdl || is_abal) {
        if (args.size > 0b10) { Undefined(); return; }
        const bool addsub_signed = (args.opcode == Op::kSaddl ||
                                    args.opcode == Op::kSsubl ||
                                    args.opcode == Op::kSaddw ||
                                    args.opcode == Op::kSsubw ||
                                    args.opcode == Op::kSabdl ||
                                    args.opcode == Op::kSabal);

        const int32_t vn_off_as = VRegOffset(args.rn);
        const int32_t vm_off_as = VRegOffset(args.rm);
        const int32_t vd_off_as = VRegOffset(args.rd);
        const int32_t narrow_disp = args.q ? 8 : 0;
        const bool n_is_wide = (is_addw || is_subw);

        SimdRegister xn_as = AllocTempSimdReg();
        SimdRegister xm_as = AllocTempSimdReg();
        if (!success()) { return; }

        // Load + widen Vn.
        if (n_is_wide) {
          // Vn is already a 128-bit wide-lane vector (SADDW family).
          as_.Movdqu(xn_as, {.base = Assembler::rbp, .disp = vn_off_as});
        } else {
          as_.Movq(xn_as, {.base = Assembler::rbp, .disp = vn_off_as + narrow_disp});
          switch (args.size) {
            case 0b00: if (addsub_signed) as_.Pmovsxbw(xn_as, xn_as); else as_.Pmovzxbw(xn_as, xn_as); break;
            case 0b01: if (addsub_signed) as_.Pmovsxwd(xn_as, xn_as); else as_.Pmovzxwd(xn_as, xn_as); break;
            case 0b10: if (addsub_signed) as_.Pmovsxdq(xn_as, xn_as); else as_.Pmovzxdq(xn_as, xn_as); break;
          }
        }
        // Load + widen Vm (always narrow).
        as_.Movq(xm_as, {.base = Assembler::rbp, .disp = vm_off_as + narrow_disp});
        switch (args.size) {
          case 0b00: if (addsub_signed) as_.Pmovsxbw(xm_as, xm_as); else as_.Pmovzxbw(xm_as, xm_as); break;
          case 0b01: if (addsub_signed) as_.Pmovsxwd(xm_as, xm_as); else as_.Pmovzxwd(xm_as, xm_as); break;
          case 0b10: if (addsub_signed) as_.Pmovsxdq(xm_as, xm_as); else as_.Pmovzxdq(xm_as, xm_as); break;
        }

        if (is_addl || is_addw) {
          switch (args.size) {
            case 0b00: as_.Paddw(xn_as, xm_as); break;
            case 0b01: as_.Paddd(xn_as, xm_as); break;
            case 0b10: as_.Paddq(xn_as, xm_as); break;
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_as}, xn_as);
          return;
        }
        if (is_subl || is_subw) {
          switch (args.size) {
            case 0b00: as_.Psubw(xn_as, xm_as); break;
            case 0b01: as_.Psubd(xn_as, xm_as); break;
            case 0b10: as_.Psubq(xn_as, xm_as); break;
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_as}, xn_as);
          return;
        }

        // ABDL / ABAL: abs(a - b)
        //
        // size=00/01 (8b/16b → 16b/32b): use max(a, b) - min(a, b) at the
        // widened lane width.  Signed/unsigned divergence picks the
        // PMAXS*/PMINS* vs PMAXU*/PMINU* flavour.
        //
        // size=10 (32b → 64b): no 64-bit lane-wise signed/unsigned max/min
        // in SSE (PMAXSQ/PMINSQ/PMAXUQ/PMINUQ are AVX-512).  Instead compute
        // diff = a - b at 64-bit lane width (Psubq after the widening done
        // above), then apply a 2-instruction signed-abs primitive built
        // from Pcmpgtq against zero: mask = (0 > diff) per qword (which is
        // -1 where diff < 0, else 0); abs(diff) = (diff ^ mask) - mask.
        // Pcmpgtq is SSE4.2.  This works for both signed and unsigned
        // inputs because the widening (Pmovsxdq vs Pmovzxdq) already
        // injected the correct extension; the subtraction at 64-bit lane
        // width then yields a signed diff whose absolute value is the same
        // for both signed and unsigned interpretations of the inputs.
        if (args.size == 0b10) {
          as_.Psubq(xn_as, xm_as);                          // diff = a - b
          SimdRegister mask = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Pxor(mask, mask);
          as_.Pcmpgtq(mask, xn_as);                         // -1 if diff<0
          as_.Pxor(xn_as, mask);
          as_.Psubq(xn_as, mask);                           // = abs(diff)
          if (is_abal) {
            SimdRegister xd_as = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqu(xd_as, {.base = Assembler::rbp, .disp = vd_off_as});
            as_.Paddq(xd_as, xn_as);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_as}, xd_as);
            return;
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_as}, xn_as);
          return;
        }

        SimdRegister xmax = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqa(xmax, xn_as);  // save original Vn
        if (addsub_signed) {
          switch (args.size) {
            case 0b00: as_.Pmaxsw(xmax, xm_as); as_.Pminsw(xn_as, xm_as); break;
            case 0b01: as_.Pmaxsd(xmax, xm_as); as_.Pminsd(xn_as, xm_as); break;  // SSE4.1
          }
        } else {
          switch (args.size) {
            case 0b00: as_.Pmaxuw(xmax, xm_as); as_.Pminuw(xn_as, xm_as); break;  // SSE4.1
            case 0b01: as_.Pmaxud(xmax, xm_as); as_.Pminud(xn_as, xm_as); break;  // SSE4.1
          }
        }
        switch (args.size) {
          case 0b00: as_.Psubw(xmax, xn_as); break;
          case 0b01: as_.Psubd(xmax, xn_as); break;
        }

        if (is_abal) {
          SimdRegister xd_as = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xd_as, {.base = Assembler::rbp, .disp = vd_off_as});
          switch (args.size) {
            case 0b00: as_.Paddw(xd_as, xmax); break;
            case 0b01: as_.Paddd(xd_as, xmax); break;
          }
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_as}, xd_as);
          return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_as}, xmax);
        return;
      }
    }

    // SQDMULL (signed doubling widening multiply,
    // saturating): product = SAT(2 * SignedWiden(Vn) * Vm).  Only the
    // .4H->.4S form (size=01) is JIT-lowered: PMOVSXWD + PMULLD give exact
    // 32-bit products, and the doubling overflows only at the INT16_MIN^2
    // case (product == 0x40000000 -> SMAX), handled with a PCMPEQD blend.
    // SQDMLAL/SQDMLSL add a second 32-bit signed *saturating* accumulate of
    // that product into Vd; x86 has no 32-bit saturating add, so it is
    // synthesised from PADDD/PSUBD + overflow detection (the carry into the
    // sign bit) and a PSRAD-built saturation constant.  The .2S->.2D form
    // (size=10) bails to the interpreter.
    if (args.opcode == Op::kSqdmull || args.opcode == Op::kSqdmlal ||
        args.opcode == Op::kSqdmlsl) {
      const bool is_acc = (args.opcode != Op::kSqdmull);
      const bool is_sub = (args.opcode == Op::kSqdmlsl);
      const int32_t vn_o = VRegOffset(args.rn);
      const int32_t vm_o = VRegOffset(args.rm);
      const int32_t vd_o = VRegOffset(args.rd);
      const int32_t extra = args.q ? 8 : 0;
      if (args.size == 0b01) {  // .4S form (manual 32-bit saturation)
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xmask = AllocTempSimdReg();
        SimdRegister xsat = AllocTempSimdReg();
        if (!success()) { return; }
        Register t = AllocTempReg();
        if (t == no_register) { success_ = false; return; }
        as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_o + extra});
        as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_o + extra});
        as_.Pmovsxwd(xn, xn);
        as_.Pmovsxwd(xm, xm);
        as_.Pmulld(xn, xm);  // exact 32-bit products
        as_.Movq(t, int64_t{0x4000000040000000LL});
        as_.Movq(xmask, t);
        as_.Punpcklqdq(xmask, xmask);
        as_.Pcmpeqd(xmask, xn);  // lanes where product == INT16_MIN^2
        as_.Paddd(xn, xn);       // double (saturating lanes are wrong here)
        as_.Movq(t, int64_t{0x7FFFFFFF7FFFFFFFLL});
        as_.Movq(xsat, t);
        as_.Punpcklqdq(xsat, xsat);
        as_.Pand(xsat, xmask);   // SMAX in saturating lanes
        as_.Pandn(xmask, xn);    // ~mask & doubled (non-saturating lanes)
        as_.Por(xmask, xsat);    // xmask = doubled saturated product P
        if (!is_acc) {
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_o}, xmask);
          return;
        }
        // result = SignedSat32(a +/- P) where a = Vd.4S accumulator, P=xmask.
        SimdRegister xd = xn;    // reuse: a (accumulator)
        SimdRegister xres = xm;  // reuse: result of the (non-saturating) op
        SimdRegister xof = xsat; // reuse: overflow sign-bit mask
        SimdRegister xtmp = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_o});
        if (is_sub) {
          as_.Movdqa(xres, xd);
          as_.Psubd(xres, xmask);     // diff = a - P
          as_.Movdqa(xof, xd);
          as_.Pxor(xof, xmask);       // a ^ P
          as_.Movdqa(xtmp, xd);
          as_.Pxor(xtmp, xres);       // a ^ diff
          as_.Pand(xof, xtmp);        // overflow when (a^P)&(a^diff) sign set
        } else {
          as_.Movdqa(xres, xd);
          as_.Paddd(xres, xmask);     // sum = a + P
          as_.Movdqa(xof, xd);
          as_.Pxor(xof, xres);        // a ^ sum
          as_.Movdqa(xtmp, xmask);
          as_.Pxor(xtmp, xres);       // P ^ sum
          as_.Pand(xof, xtmp);        // overflow when (a^sum)&(P^sum) sign set
        }
        as_.Psrad(xof, int8_t{31});   // all-ones lanes where the add overflowed
        as_.Psrad(xd, int8_t{31});    // a's sign extended (0 or -1)
        as_.Movq(t, int64_t{0x7FFFFFFF7FFFFFFFLL});
        as_.Movq(xmask, t);           // reuse xmask (P consumed) for the const
        as_.Punpcklqdq(xmask, xmask);
        as_.Pxor(xd, xmask);          // sat = (a>>31) ^ INT32_MAX (MAX or MIN)
        as_.Pand(xd, xof);            // sat in overflow lanes
        as_.Pandn(xof, xres);         // result in non-overflow lanes
        as_.Por(xd, xof);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_o}, xd);
        return;
      }
      if (args.size == 0b10) {  // .2D form (.2S->.2D, manual 64-bit saturation)
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xmask = AllocTempSimdReg();
        SimdRegister xsat = AllocTempSimdReg();
        if (!success()) { return; }
        Register t = AllocTempReg();
        if (t == no_register) { success_ = false; return; }
        as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_o + extra});
        as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_o + extra});
        as_.Pmovsxdq(xn, xn);  // 2 source words -> 2 sign-extended qwords
        as_.Pmovsxdq(xm, xm);
        as_.Pmuldq(xn, xm);    // 2 exact signed 64-bit products
        // The only doubling-overflow case is INT32_MIN^2 == 2^62 -> SMAX.
        as_.Movq(t, int64_t{0x4000000000000000LL});
        as_.Movq(xmask, t);
        as_.Punpcklqdq(xmask, xmask);
        as_.Pcmpeqq(xmask, xn);  // lanes where product == 2^62
        as_.Paddq(xn, xn);       // double
        as_.Movq(t, int64_t{0x7FFFFFFFFFFFFFFFLL});
        as_.Movq(xsat, t);
        as_.Punpcklqdq(xsat, xsat);
        as_.Pand(xsat, xmask);   // INT64_MAX in saturating lanes
        as_.Pandn(xmask, xn);    // doubled in non-saturating lanes
        as_.Por(xmask, xsat);    // xmask = doubled saturated product P
        if (!is_acc) {
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_o}, xmask);
          return;
        }
        // 64-bit signed saturating accumulate: result = SignedSat64(Vd +/- P).
        SimdRegister xd = xn;    // reuse: accumulator a
        SimdRegister xres = xm;  // reuse: a +/- P
        SimdRegister xof = xsat; // reuse: overflow raw value
        SimdRegister xtmp = AllocTempSimdReg();
        SimdRegister xzero = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pxor(xzero, xzero);
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_o});
        if (is_sub) {
          as_.Movdqa(xres, xd);
          as_.Psubq(xres, xmask);    // diff = a - P
          as_.Movdqa(xof, xd);
          as_.Pxor(xof, xmask);      // a ^ P
          as_.Movdqa(xtmp, xd);
          as_.Pxor(xtmp, xres);      // a ^ diff
          as_.Pand(xof, xtmp);       // overflow when (a^P)&(a^diff) sign set
        } else {
          as_.Movdqa(xres, xd);
          as_.Paddq(xres, xmask);    // sum = a + P
          as_.Movdqa(xof, xd);
          as_.Pxor(xof, xres);       // a ^ sum
          as_.Movdqa(xtmp, xmask);
          as_.Pxor(xtmp, xres);      // P ^ sum
          as_.Pand(xof, xtmp);       // overflow when (a^sum)&(P^sum) sign set
        }
        // overflow_mask (xtmp) = all-ones per qword where xof < 0 (sign set).
        as_.Movdqa(xtmp, xzero);
        as_.Pcmpgtq(xtmp, xof);
        // sat = sign(a) ^ INT64_MAX  (INT64_MIN if a<0, else INT64_MAX).
        as_.Pcmpgtq(xzero, xd);      // xzero = all-ones where a < 0
        as_.Movq(t, int64_t{0x7FFFFFFFFFFFFFFFLL});
        as_.Movq(xmask, t);          // reuse xmask (P consumed) for INT64_MAX
        as_.Punpcklqdq(xmask, xmask);
        as_.Pxor(xzero, xmask);      // xzero = sat value
        as_.Pand(xzero, xtmp);       // sat in overflow lanes
        as_.Pandn(xtmp, xres);       // result in non-overflow lanes
        as_.Por(xzero, xtmp);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_o}, xzero);
        return;
      }
      success_ = false;  // size=00 -> interpreter (no SQDMULL byte form)
      return;
    }

    const bool is_mull = (args.opcode == Op::kSmull || args.opcode == Op::kUmull);
    const bool is_mlal = (args.opcode == Op::kSmlal || args.opcode == Op::kUmlal);
    const bool is_mlsl = (args.opcode == Op::kSmlsl || args.opcode == Op::kUmlsl);
    if (!is_mull && !is_mlal && !is_mlsl) {
      Undefined();  // remaining ThreeDiff ops not JIT-lowered yet
      return;
    }
    const bool is_signed = (args.opcode == Op::kSmull ||
                            args.opcode == Op::kSmlal ||
                            args.opcode == Op::kSmlsl);
    if (args.size > 0b10) { Undefined(); return; }

    const int32_t vn_off = VRegOffset(args.rn);
    const int32_t vm_off = VRegOffset(args.rm);
    const int32_t vd_off = VRegOffset(args.rd);

    // Q=0 reads the low 64 bits of Vn/Vm; Q=1 reads bytes 8..15. The widening
    // turns 8 input bytes into the full 128-bit output.
    const int32_t src_disp_extra = args.q ? 8 : 0;

    SimdRegister xn = AllocTempSimdReg();
    SimdRegister xm = AllocTempSimdReg();
    if (!success()) { return; }

    as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off + src_disp_extra});
    as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off + src_disp_extra});

    // Widen + multiply. After this block xn holds the 128-bit lane-wise
    // product (eight 16-bit / four 32-bit / two 64-bit lanes).
    switch (args.size) {
      case 0b00:
        if (is_signed) { as_.Pmovsxbw(xn, xn); as_.Pmovsxbw(xm, xm); }
        else           { as_.Pmovzxbw(xn, xn); as_.Pmovzxbw(xm, xm); }
        as_.Pmullw(xn, xm);
        break;
      case 0b01:
        if (is_signed) { as_.Pmovsxwd(xn, xn); as_.Pmovsxwd(xm, xm); }
        else           { as_.Pmovzxwd(xn, xn); as_.Pmovzxwd(xm, xm); }
        as_.Pmulld(xn, xm);  // SSE4.1
        break;
      case 0b10:
        if (is_signed) {
          as_.Pmovsxdq(xn, xn);
          as_.Pmovsxdq(xm, xm);
          as_.Pmuldq(xn, xm);  // SSE4.1 — signed 32×32 → 64
        } else {
          as_.Pmovzxdq(xn, xn);
          as_.Pmovzxdq(xm, xm);
          as_.Pmuludq(xn, xm);  // unsigned 32×32 → 64
        }
        break;
    }

    if (is_mull) {
      // No accumulate — store the lane-wise product directly.
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
      return;
    }

    // MLAL (accumulate) / MLSL (subtract-accumulate): Vd = Vd ± products,
    // at the *wide* lane width.
    SimdRegister xd = AllocTempSimdReg();
    if (!success()) { return; }
    as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
    switch (args.size) {
      case 0b00:
        if (is_mlal) as_.Paddw(xd, xn);
        else         as_.Psubw(xd, xn);
        break;
      case 0b01:
        if (is_mlal) as_.Paddd(xd, xn);
        else         as_.Psubd(xd, xn);
        break;
      case 0b10:
        if (is_mlal) as_.Paddq(xd, xn);
        else         as_.Psubq(xd, xn);
        break;
    }
    as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
  }

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    // JIT for EXT Vd.<T>, Vn.<T>, Vm.<T>, #imm.
    // Concatenates Vn:Vm and extracts a vector starting at byte `index`
    // from Vn. Equivalent to:
    //   result = (Vn >> (index*8)) | (Vm << ((vlen-index)*8))
    // For q=1 this is a 16-byte window; for q=0 it's an 8-byte window.
    if (!q) {
      // 64-bit form: 8-byte window, index 0..7.  ARM ARM marks
      // imm4[3]=1 (index >= 8) as UNDEFINED for Q=0.
      if (index >= 8) { Undefined(); return; }

      const int32_t vn_off_q0 = VRegOffset(rn);
      const int32_t vm_off_q0 = VRegOffset(rm);
      const int32_t vd_off_q0 = VRegOffset(rd);

      SimdRegister xn_q0 = AllocTempSimdReg();
      SimdRegister xm_q0 = AllocTempSimdReg();
      if (!success()) { return; }
      // Movq into XMM loads 8 bytes and zero-extends the upper 64 bits,
      // matching the Q=0 view of Vn/Vm.
      as_.Movq(xn_q0, {.base = Assembler::rbp, .disp = vn_off_q0});
      as_.Movq(xm_q0, {.base = Assembler::rbp, .disp = vm_off_q0});
      // Concatenate Vn:Vm into a single 16-byte register so PSRLDQ can
      // walk a contiguous window: low 8 bytes hold Vn, high 8 bytes
      // hold Vm.
      as_.Punpcklqdq(xn_q0, xm_q0);
      // Shift right by `index` bytes; PSRLDQ zero-fills the high end.
      // The wanted 8-byte window now sits in the low 8 bytes of xn.
      if (index != 0) {
        as_.Psrldq(xn_q0, static_cast<int8_t>(index));
      }
      // Force the upper 64 bits of the result to zero: PSLLDQ-8 followed
      // by PSRLDQ-8 shuttles the wanted 8 bytes through the upper half
      // and back, clearing the previously-occupied high bytes.
      as_.Pslldq(xn_q0, int8_t{8});
      as_.Psrldq(xn_q0, int8_t{8});
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off_q0}, xn_q0);
      return;
    }
    int32_t vn_off = VRegOffset(rn);
    int32_t vm_off = VRegOffset(rm);
    int32_t vd_off = VRegOffset(rd);

    SimdRegister xn = AllocTempSimdReg();
    SimdRegister xm = AllocTempSimdReg();
    if (!success()) { return; }

    as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});

    if (index == 0) {
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
      return;
    }
    if (index == 16) {
      // ARM ARM forbids index==16 for q=1 (encoding has 4-bit index when
      // q=1, so max 15). Belt-and-braces.
      Undefined(); return;
    }

    as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});
    // xn = xn >> (index bytes) ; zero upper bytes
    as_.Psrldq(xn, static_cast<int8_t>(index));
    // xm = xm << ((16-index) bytes) ; zero lower bytes
    as_.Pslldq(xm, static_cast<int8_t>(16 - index));
    as_.Por(xn, xm);
    as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
  }

  void AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size,
                      uint8_t opcode, bool q) {
    // ZIP / UZP / TRN vector permute JIT
    //
    // opcode encoding (3 bits, from Decoder::DecodeAdvSimd at decoder.h:2808):
    //   001=UZP1, 010=TRN1, 011=ZIP1, 101=UZP2, 110=TRN2, 111=ZIP2
    //
    // ZIP1/ZIP2 (-174): interleave the LOWER/UPPER half of each source.
    // Maps directly to x86's PUNPCK family:
    //   ZIP1 .16B/.8B -> PUNPCKLBW    ZIP2 .16B -> PUNPCKHBW
    //   ZIP1 .8H /.4H -> PUNPCKLWD    ZIP2 .8H  -> PUNPCKHWD
    //   ZIP1 .4S /.2S -> PUNPCKLDQ    ZIP2 .4S  -> PUNPCKHDQ
    //   ZIP1 .2D      -> PUNPCKLQDQ   ZIP2 .2D  -> PUNPCKHQDQ
    // Q=0 ZIP2: PSRLDQ-4 prelude lands bytes 4..7 at positions 0..3
    // so PUNPCKL picks up the originally-upper-half elements; the Q=0
    // upper-zero tail masks any garbage beyond byte 7 of the result.
    //
    // UZP1/UZP2 (-175): take EVEN (UZP1) or ODD (UZP2) elements
    // from each source.  Lowering varies per element width:
    //   .16B/.8B  (PACKUSWB): mask high byte of each halfword via
    //     PSLLW(8)+PSRLW(8) for UZP1, or PSRLW(8) for UZP2, then
    //     PACKUSWB(xn, xm).  Each masked halfword is 0..255 so PACKUS
    //     saturation is a no-op.
    //   .8H /.4H  (PACKUSDW; SSE4.1): same idea with PSLLD/PSRLD by 16
    //     to mask the high halfword of each dword.  Bails to interp
    //     when host lacks SSE4.1.
    //   .4S /.2S  (SHUFPS): UZP1 imm 0x88 = [src1[0],src1[2],src2[0],
    //     src2[2]]; UZP2 imm 0xDD = [src1[1],src1[3],src2[1],src2[3]].
    //   .2D       (PUNPCKL/HQDQ): coincides with ZIP1.2D / ZIP2.2D
    //     because 2-lane uzip and zip are the same permutation.
    //
    // Q=0 UZP forms: PUNPCKLQDQ(xn, xm) first combines the lower 8 bytes
    // of vn and vm into a single 16-byte register, after which the
    // Q=1-shaped lowering (now using xn as both PACKUS / SHUFPS
    // sources) produces 16/8 bytes of result with the wanted 8 bytes
    // duplicated in the low and high halves.  The Q=0 upper-zero tail
    // discards the duplicate.
    //
    // TRN1/TRN2 (this cycle): interleave EVEN-indexed (TRN1) or
    // ODD-indexed (TRN2) elements alternately from each source:
    //   TRN1: Vd[2i]=Vn[2i],   Vd[2i+1]=Vm[2i]
    //   TRN2: Vd[2i]=Vn[2i+1], Vd[2i+1]=Vm[2i+1]
    // Lowering varies per element width:
    //   .16B/.8B  (PSHUFB + POR): precomputed mask_n picks the wanted
    //     bytes of Vn into the even positions of the result and zeros
    //     the rest (0x80 mask byte → 0 in PSHUFB); mask_m mirrors for
    //     Vm into odd positions; POR combines.  Masks are built at
    //     JIT time via the same Movq+Pinsrq idiom as the §C8 CNT
    //     lookup table (lite_translator.h:7967-7979).
    //   .8H /.4H  (PSHUFB + POR): same shape with halfword indices.
    //   .4S /.2S  (PSHUFD + PUNPCKLDQ): PSHUFD imm 0x88 collapses
    //     dwords {0,2} of each source into both halves; PUNPCKLDQ
    //     then interleaves them into [s1[0], s2[0], s1[2], s2[2]] =
    //     TRN1.4S.  TRN2.4S uses imm 0xDD to collapse {1,3}.
    //   .2D       (PUNPCKL/HQDQ): coincides with ZIP1.2D / ZIP2.2D
    //     for the same 2-lane reason as UZP.
    //   .2S Q=0   (PUNPCKLDQ): TRN1.2S = [s1[0], s2[0]] coincides with
    //     ZIP1.2S; TRN2.2S = [s1[1], s2[1]] coincides with ZIP2.2S,
    //     reached via the same PSRLDQ-4 prelude.
    //
    // No SSE4.1 dependency for TRN: PSHUFB is SSSE3 (universally
    // available on the emulator host), PSHUFD/PUNPCK are SSE2.
    const bool is_zip = (opcode == 0b011 || opcode == 0b111);
    const bool is_uzp = (opcode == 0b001 || opcode == 0b101);
    const bool is_trn = (opcode == 0b010 || opcode == 0b110);
    if (!is_zip && !is_uzp && !is_trn) {
      UNUSED(rd, rn, rm, size);
      success_ = false;
      return;
    }
    if (size > 0b11) {
      success_ = false;
      return;
    }
    // Q=0 .2D is reserved by the ARM ARM (encoding restricted).
    if (!q && size == 0b11) {
      success_ = false;
      return;
    }
    // UZP .8H/.4H lowering uses PACKUSDW which is SSE4.1.
    if (is_uzp && size == 0b01 && !host_platform::kHasSSE4_1) {
      success_ = false;
      return;
    }

    SimdRegister xn = AllocTempSimdReg();
    SimdRegister xm = AllocTempSimdReg();
    if (!success()) { return; }
    int32_t vn_off = VRegOffset(rn);
    int32_t vm_off = VRegOffset(rm);
    int32_t vd_off = VRegOffset(rd);
    as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
    as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});

    if (is_zip) {
      const bool is_zip2 = (opcode == 0b111);

      if (!q && is_zip2) {
        // Shift each source right by 4 bytes so the (originally upper-half)
        // bytes 4..7 land at positions 0..3; PUNPCKL will then pick them up.
        as_.Psrldq(xn, int8_t{4});
        as_.Psrldq(xm, int8_t{4});
      }

      if (q && is_zip2) {
        switch (size) {
          case 0b00: as_.Punpckhbw(xn, xm); break;   // ZIP2 .16B
          case 0b01: as_.Punpckhwd(xn, xm); break;   // ZIP2 .8H
          case 0b10: as_.Punpckhdq(xn, xm); break;   // ZIP2 .4S
          case 0b11: as_.Punpckhqdq(xn, xm); break;  // ZIP2 .2D
        }
      } else {
        switch (size) {
          case 0b00: as_.Punpcklbw(xn, xm); break;   // ZIP1 .16B/.8B, ZIP2 .8B (after shift)
          case 0b01: as_.Punpcklwd(xn, xm); break;   // ZIP1 .8H/.4H,  ZIP2 .4H (after shift)
          case 0b10: as_.Punpckldq(xn, xm); break;   // ZIP1 .4S/.2S,  ZIP2 .2S (after shift)
          case 0b11: as_.Punpcklqdq(xn, xm); break;  // ZIP1 .2D       (Q=0 forbidden by ARM)
        }
      }
    } else if (is_uzp) {
      const bool is_uzp2 = (opcode == 0b101);

      if (!q) {
        // Combine the lower 8 bytes of vn and vm into xn so subsequent
        // PACKUS / SHUFPS only has to consume a single source.  After
        // this, xn = [vn_lo (bytes 0..7) | vm_lo (bytes 0..7)].
        as_.Punpcklqdq(xn, xm);
      }

      switch (size) {
        case 0b00: {  // .16B (q=1) or .8B (q=0).  PACKUSWB.
          if (is_uzp2) {
            as_.Psrlw(xn, int8_t{8});
          } else {
            as_.Psllw(xn, int8_t{8});
            as_.Psrlw(xn, int8_t{8});
          }
          if (q) {
            if (is_uzp2) {
              as_.Psrlw(xm, int8_t{8});
            } else {
              as_.Psllw(xm, int8_t{8});
              as_.Psrlw(xm, int8_t{8});
            }
            as_.Packuswb(xn, xm);
          } else {
            as_.Packuswb(xn, xn);
          }
          break;
        }
        case 0b01: {  // .8H (q=1) or .4H (q=0).  PACKUSDW (SSE4.1).
          if (is_uzp2) {
            as_.Psrld(xn, int8_t{16});
          } else {
            as_.Pslld(xn, int8_t{16});
            as_.Psrld(xn, int8_t{16});
          }
          if (q) {
            if (is_uzp2) {
              as_.Psrld(xm, int8_t{16});
            } else {
              as_.Pslld(xm, int8_t{16});
              as_.Psrld(xm, int8_t{16});
            }
            as_.Packusdw(xn, xm);
          } else {
            as_.Packusdw(xn, xn);
          }
          break;
        }
        case 0b10: {  // .4S (q=1) or .2S (q=0).  SHUFPS.
          const int8_t imm =
              is_uzp2 ? static_cast<int8_t>(0xDD) : static_cast<int8_t>(0x88);
          if (q) {
            as_.Shufps(xn, xm, imm);
          } else {
            // xn already holds [vn_lo | vm_lo] from Punpcklqdq above.
            as_.Shufps(xn, xn, imm);
          }
          break;
        }
        case 0b11: {  // .2D (q=1 only; q=0 caught above).
          // UZP1.2D == ZIP1.2D, UZP2.2D == ZIP2.2D (2-lane coincidence).
          if (is_uzp2) {
            as_.Punpckhqdq(xn, xm);
          } else {
            as_.Punpcklqdq(xn, xm);
          }
          break;
        }
      }
    } else {  // is_trn
      const bool is_trn2 = (opcode == 0b110);

      switch (size) {
        case 0b00:
        case 0b01: {
          // .16B / .8B (size=00) and .8H / .4H (size=01) use PSHUFB +
          // POR with precomputed per-byte masks.  mask_n selects the
          // wanted bytes of Vn into even output lane positions (rest =
          // 0x80, which PSHUFB renders as 0); mask_m selects Vm bytes
          // into the odd positions; POR combines.  Q=0 forms reuse the
          // Q=1 masks — bytes 8..15 of the result are either zeroed by
          // PSHUFB (mask 0x80) or filled with garbage from the source's
          // unspecified upper half, both of which get cleared by the
          // shared Q=0 upper-zero tail.
          SimdRegister mask_n = AllocTempSimdReg();
          SimdRegister mask_m = AllocTempSimdReg();
          if (!success()) { return; }
          Register r1 = AllocTempReg();
          if (r1 == no_register) {
            success_ = false;
            return;
          }

          int64_t mask_n_lo, mask_n_hi, mask_m_lo, mask_m_hi;
          if (size == 0b00) {
            if (is_trn2) {
              // TRN2 .16B: bytes {1,_,3,_,5,_,7,_,  9,_,11,_,13,_,15,_}
              // for Vn (where _ = 0x80, replaced by Vm via POR).
              mask_n_lo = static_cast<int64_t>(0x8007800580038001LL);
              mask_n_hi = static_cast<int64_t>(0x800F800D800B8009LL);
              mask_m_lo = static_cast<int64_t>(0x0780058003800180LL);
              mask_m_hi = static_cast<int64_t>(0x0F800D800B800980LL);
            } else {
              // TRN1 .16B: bytes {0,_,2,_,4,_,6,_,  8,_,10,_,12,_,14,_}
              mask_n_lo = static_cast<int64_t>(0x8006800480028000LL);
              mask_n_hi = static_cast<int64_t>(0x800E800C800A8008LL);
              mask_m_lo = static_cast<int64_t>(0x0680048002800080LL);
              mask_m_hi = static_cast<int64_t>(0x0E800C800A800880LL);
            }
          } else {  // size == 0b01: halfword granularity.
            if (is_trn2) {
              // TRN2 .8H: halfwords {1,_,3,_, 5,_,7,_}  -> byte pairs
              //   {(2,3),_,_,(6,7),_,_, (10,11),_,_,(14,15),_,_}
              mask_n_lo = static_cast<int64_t>(0x8080070680800302LL);
              mask_n_hi = static_cast<int64_t>(0x80800F0E80800B0ALL);
              mask_m_lo = static_cast<int64_t>(0x0706808003028080LL);
              mask_m_hi = static_cast<int64_t>(0x0F0E80800B0A8080LL);
            } else {
              // TRN1 .8H: halfwords {0,_,2,_, 4,_,6,_}  -> byte pairs
              //   {(0,1),_,_,(4,5),_,_, (8,9),_,_,(12,13),_,_}
              mask_n_lo = static_cast<int64_t>(0x8080050480800100LL);
              mask_n_hi = static_cast<int64_t>(0x80800D0C80800908LL);
              mask_m_lo = static_cast<int64_t>(0x0504808001008080LL);
              mask_m_hi = static_cast<int64_t>(0x0D0C808009088080LL);
            }
          }

          as_.Movq(r1, mask_n_lo);
          as_.Movq(mask_n, r1);
          as_.Movq(r1, mask_n_hi);
          as_.Pinsrq(mask_n, r1, int8_t{1});

          as_.Movq(r1, mask_m_lo);
          as_.Movq(mask_m, r1);
          as_.Movq(r1, mask_m_hi);
          as_.Pinsrq(mask_m, r1, int8_t{1});

          as_.Pshufb(xn, mask_n);
          as_.Pshufb(xm, mask_m);
          as_.Por(xn, xm);
          break;
        }
        case 0b10: {  // .4S (q=1) or .2S (q=0).
          if (!q) {
            // .2S has only 2 lanes per source: TRN1.2S = [s1[0], s2[0]]
            // == ZIP1.2S; TRN2.2S = [s1[1], s2[1]] == ZIP2.2S.  Both
            // reuse the ZIP .2S lowering — PUNPCKLDQ, with a PSRLDQ-4
            // prelude for TRN2 to pull s1[1]/s2[1] down to position 0.
            if (is_trn2) {
              as_.Psrldq(xn, int8_t{4});
              as_.Psrldq(xm, int8_t{4});
            }
            as_.Punpckldq(xn, xm);
          } else {
            // .4S: PSHUFD imm 0x88 collapses each source's even-indexed
            // dwords (or 0xDD for odd) into both halves of the register,
            // then PUNPCKLDQ interleaves the low halves:
            //   xn' = [s1[0], s1[2], s1[0], s1[2]]   (after PSHUFD 0x88)
            //   xm' = [s2[0], s2[2], s2[0], s2[2]]
            //   res = PUNPCKLDQ(xn', xm') = [s1[0], s2[0], s1[2], s2[2]]
            //       = TRN1.4S  (and similarly 0xDD -> TRN2.4S).
            const int8_t imm = is_trn2 ? static_cast<int8_t>(0xDD)
                                       : static_cast<int8_t>(0x88);
            as_.Pshufd(xn, xn, imm);
            as_.Pshufd(xm, xm, imm);
            as_.Punpckldq(xn, xm);
          }
          break;
        }
        case 0b11: {  // .2D (q=1 only; q=0 caught above).
          // TRN1.2D == ZIP1.2D, TRN2.2D == ZIP2.2D (2-lane coincidence).
          if (is_trn2) {
            as_.Punpckhqdq(xn, xm);
          } else {
            as_.Punpcklqdq(xn, xm);
          }
          break;
        }
      }
    }

    if (!q) {
      // Zero upper 64 bits per D-register semantics.
      as_.Pslldq(xn, int8_t{8});
      as_.Psrldq(xn, int8_t{8});
    }
    as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
  }

  // JIT for AdvSIMD TBL/TBX (table lookup / table lookup extended).
  //   TBL Vd.<T>, {Vn.16B [, V(n+1) [, V(n+2) [, V(n+3)]]]}, Vm.<T>
  //   TBX Vd.<T>, {Vn.16B [, V(n+1) [, V(n+2) [, V(n+3)]]]}, Vm.<T>
  // T = 8B (q=0) or 16B (q=1).  len ∈ {0,1,2,3} → 1..4 table registers
  // (consecutive, wrapping (rn+r) mod 32).  op: 0=TBL, 1=TBX.
  //
  // Per output byte i (i < 8 or 16):
  //   idx = Vm.byte[i]
  //   if idx < table_regs*16: out[i] = table.byte[idx]
  //   else (TBL): out[i] = 0
  //   else (TBX): out[i] = Vd.byte[i] (preserved)
  //
  // x86 PSHUFB(dst, mask) computes per byte:
  //   if mask[i][7] = 1: dst[i] = 0
  //   else              : dst[i] = src.byte[mask[i] & 0x0F]
  //
  // Approach: for each table register r in [0, table_regs):
  //   shifted_idx_r = (Vm.byte[i] - r*16) bytewise wrap
  //   in_range_r    = 0xFF if shifted_idx_r ∈ [0,15] else 0x00
  //                  = PCMPEQB(PSUBUSB(shifted_idx_r, 15), 0)
  //   looked_up_r   = PSHUFB(V[(rn+r) % 32], shifted_idx_r)
  //   looked_up_r &= in_range_r   (zero wrong-register lookups, including
  //                                 the [16..127] band that PSHUFB doesn't
  //                                 mask)
  //   acc          |= looked_up_r
  //   in_range_all |= in_range_r
  // For TBX, blend Vd back over bytes not in any register's range:
  //   result = acc | (Vd & ~in_range_all)
  // For TBL, result = acc.
  // For q=0, zero upper 64 bits via the standard PSLLDQ/PSRLDQ pair.
  //
  // All table loads happen before any store to Vd, so in-place encodings
  // (Vd == Vn, Vd == Vm, Vd == V(n+1)..) read the original source bytes.
  void AdvSimdTableLookup(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t len,
                          uint8_t op, bool q) {
    const uint8_t table_regs = static_cast<uint8_t>(len + 1);  // 1..4
    const int32_t vm_off = VRegOffset(rm);
    const int32_t vd_off = VRegOffset(rd);

    SimdRegister xmm_idx        = AllocTempSimdReg();
    SimdRegister xmm_acc        = AllocTempSimdReg();
    SimdRegister xmm_in_range   = AllocTempSimdReg();
    SimdRegister xmm_const15    = AllocTempSimdReg();
    SimdRegister xmm_zero       = AllocTempSimdReg();
    SimdRegister xmm_tmp_idx    = AllocTempSimdReg();
    SimdRegister xmm_tmp_lookup = AllocTempSimdReg();
    SimdRegister xmm_tmp_mask   = AllocTempSimdReg();
    if (xmm_idx == no_simd_register || xmm_acc == no_simd_register ||
        xmm_in_range == no_simd_register || xmm_const15 == no_simd_register ||
        xmm_zero == no_simd_register || xmm_tmp_idx == no_simd_register ||
        xmm_tmp_lookup == no_simd_register || xmm_tmp_mask == no_simd_register) {
      success_ = false; return;
    }

    Register r_const = AllocTempReg();
    if (r_const == no_register) { success_ = false; return; }

    // Load the index vector once.  Vm is the per-byte selector.
    as_.Movdqu(xmm_idx, {.base = Assembler::rbp, .disp = vm_off});
    // acc = 0, in_range = 0, zero = 0.
    as_.Pxor(xmm_acc, xmm_acc);
    as_.Pxor(xmm_in_range, xmm_in_range);
    as_.Pxor(xmm_zero, xmm_zero);
    // const15 = byte-broadcast(0x0F).
    as_.Movq(r_const, static_cast<int64_t>(0x0F0F0F0F0F0F0F0FLL));
    as_.Movq(xmm_const15, r_const);
    as_.Punpcklqdq(xmm_const15, xmm_const15);

    for (uint8_t r = 0; r < table_regs; ++r) {
      const int32_t vn_off_r =
          VRegOffset(((rn + r) & 31));

      // shifted_idx_r = Vm - r*16 (bytewise wrap).  For r=0 this is just Vm.
      as_.Movdqa(xmm_tmp_idx, xmm_idx);
      if (r != 0) {
        const uint64_t broadcast =
            uint64_t{0x0101010101010101ULL} * static_cast<uint64_t>(r * 16);
        as_.Movq(r_const, static_cast<int64_t>(broadcast));
        as_.Movq(xmm_tmp_mask, r_const);
        as_.Punpcklqdq(xmm_tmp_mask, xmm_tmp_mask);
        as_.Psubb(xmm_tmp_idx, xmm_tmp_mask);
      }

      // in_range_r = PCMPEQB(PSUBUSB(shifted_idx_r, 15), 0).
      as_.Movdqa(xmm_tmp_mask, xmm_tmp_idx);
      as_.Psubusb(xmm_tmp_mask, xmm_const15);
      as_.Pcmpeqb(xmm_tmp_mask, xmm_zero);

      // looked_up_r = PSHUFB(V[(rn+r)%32], shifted_idx_r), masked by in_range.
      as_.Movdqu(xmm_tmp_lookup, {.base = Assembler::rbp, .disp = vn_off_r});
      as_.Pshufb(xmm_tmp_lookup, xmm_tmp_idx);
      as_.Pand(xmm_tmp_lookup, xmm_tmp_mask);

      as_.Por(xmm_acc, xmm_tmp_lookup);
      as_.Por(xmm_in_range, xmm_tmp_mask);
    }

    if (op /* TBX */) {
      // result = acc | (Vd & ~in_range_all).
      as_.Movdqu(xmm_tmp_lookup, {.base = Assembler::rbp, .disp = vd_off});
      // xmm_tmp_mask := ~in_range_all & xmm_tmp_lookup
      as_.Movdqa(xmm_tmp_mask, xmm_in_range);
      as_.Pandn(xmm_tmp_mask, xmm_tmp_lookup);
      as_.Por(xmm_acc, xmm_tmp_mask);
    }

    if (!q) {
      // Q=0: zero the upper 64 bits per D-register semantics.
      as_.Pslldq(xmm_acc, int8_t{8});
      as_.Psrldq(xmm_acc, int8_t{8});
    }
    as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_acc);
  }

