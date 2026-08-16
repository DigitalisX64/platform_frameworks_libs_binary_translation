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
// Family: SIMD three-same-register vector arithmetic.
// Methods:
//   AdvSimdThreeSame

  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& args) {
    // JIT for common SIMD three-same ops.
    //
    // Implements MUL/MLA/MLS/ADD/SUB/AND/ORR/EOR/CMEQ at the lane sizes
    // the dynamic linker's calculate_gnu_hash_neon needs (4S MUL/MLA in
    // particular), plus the Armv8.2-FP16 FADD/FSUB/FMUL/FDIV and
    // FMAX/FMIN/FMAXNM/FMINNM vector forms via F16C round-trip. The
    // round-trip is bit-exact for the binary arithmetic ops because
    // FP32's 24-bit mantissa strictly contains FP16's 11; the max/min
    // family wraps the round-trip in a NaN-handling shim that adapts
    // x86's asymmetric MAXPS/MINPS NaN semantics to ARM's (FMAX/FMIN
    // propagate NaN, FMAXNM/FMINNM suppress single NaNs). Falls back
    // to the interpreter for opcodes/sizes outside this set.
    int32_t vn_off = VRegOffset(args.rn);
    int32_t vm_off = VRegOffset(args.rm);
    int32_t vd_off = VRegOffset(args.rd);

    switch (args.opcode) {
      case Decoder::AdvSimdThreeSameOpcode::kMul: {
        // MUL .16B/.8B via PMOVZXBW + PMULLW + PACKUSWB (SSE4.1);
        // .8H/.4H via PMULLW (SSE2); .4S/.2S via PMULLD (SSE4.1).
        // .2D / scalar 64-bit is reserved by the ARM ARM and falls back to the interpreter.
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        if (args.size == 0b00) {
          // Byte multiply: x86 has no PMULLB; widen+PMULLW+PACKUSWB recipe.
          if (!EmitByteMulWiden(xn, xm)) { Undefined(); return; }
        } else if (args.size == 0b01) {
          as_.Pmullw(xn, xm);
        } else {
          as_.Pmulld(xn, xm);
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kMla: {
        // MLA .16B/.8B via PMOVZXBW+PMULLW+PACKUSWB+PADDB (SSE4.1);
        // .8H/.4H via PMULLW+PADDW (SSE2); .4S/.2S via PMULLD+PADDD (SSE4.1).
        // .2D / scalar 64-bit is reserved by the ARM ARM and falls back to the interpreter.
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        LoadVReg(xd, args.rd);
        if (args.size == 0b00) {
          // Byte MLA: byte-multiply recipe, then PADDB into Vd. Adding at
          // byte width gives the correct low-8-bit-truncated accumulate.
          if (!EmitByteMulWiden(xn, xm)) { Undefined(); return; }
          as_.Paddb(xd, xn);
        } else if (args.size == 0b01) {
          as_.Pmullw(xn, xm);
          as_.Paddw(xd, xn);
        } else {
          as_.Pmulld(xn, xm);
          as_.Paddd(xd, xn);
        }
        if (!args.q) MaskLow64(xd);
        StoreVReg(args.rd, xd);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kMls: {
        // MLS: Vd[lane] = Vd[lane] - Vn[lane] * Vm[lane].
        // Same product recipe as kMla / kMul (byte uses widen+PMULLW+
        // PACKUSWB; halfword uses PMULLW; word uses PMULLD), but PSUB
        // the product from Vd instead of PADDing it.  .2D / scalar
        // 64-bit is reserved by the ARM ARM and falls back.
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        LoadVReg(xd, args.rd);
        if (args.size == 0b00) {
          // Byte MLS: byte-multiply recipe, then PSUBB from Vd.
          if (!EmitByteMulWiden(xn, xm)) { Undefined(); return; }
          as_.Psubb(xd, xn);
        } else if (args.size == 0b01) {
          as_.Pmullw(xn, xm);
          as_.Psubw(xd, xn);
        } else {
          as_.Pmulld(xn, xm);
          as_.Psubd(xd, xn);
        }
        if (!args.q) MaskLow64(xd);
        StoreVReg(args.rd, xd);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kSabd: {
        // SABD: Vd[lane] = |signed(Vn[lane]) - signed(Vm[lane])|.
        // ARM ARM computes the difference in extended (unbounded) signed
        // precision, then truncates the absolute value to esize.  A naive
        // PSUBx + sign-mask abs only matches when the difference fits in
        // esize bits — for INT_MIN-vs-INT_MAX pairs the modular subtraction
        // wraps and the abs gives the wrong magnitude.
        //
        // Recipe: per-lane PMAXS - PMINS gives max(a,b) - min(a,b) on the
        // SIGNED interpretation, computed in modular esize arithmetic.
        // The signed max minus the signed min equals the truncated absolute
        // difference for every input, including the wrap case (INT_MAX -
        // INT_MIN = -1 mod 2^esize = 0xFF...F, which matches ARM's
        // abs(INT_MAX-INT_MIN) = 2^esize - 1 truncated).
        //
        // Byte/word need SSE4.1 PMAX/PMINSB/D; halfword uses SSE2
        // PMAXSW/PMINSW.  .2D is reserved by the ARM ARM and falls back.
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xmax = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Movdqa(xmax, xn);
        switch (args.size) {
          case 0b00:
            as_.Pmaxsb(xmax, xm);  // SSE4.1
            as_.Pminsb(xn, xm);    // SSE4.1
            as_.Psubb(xmax, xn);
            break;
          case 0b01:
            as_.Pmaxsw(xmax, xm);  // SSE2
            as_.Pminsw(xn, xm);    // SSE2
            as_.Psubw(xmax, xn);
            break;
          case 0b10:
            as_.Pmaxsd(xmax, xm);  // SSE4.1
            as_.Pminsd(xn, xm);    // SSE4.1
            as_.Psubd(xmax, xn);
            break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xmax);
        StoreVReg(args.rd, xmax);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kUabd: {
        // UABD: Vd[lane] = (Vn[lane] > Vm[lane]) ? Vn-Vm : Vm-Vn (unsigned).
        // Recipe: per-lane max(Vn,Vm) - min(Vn,Vm) via PMAXUx + PMINUx + PSUBx.
        // Byte uses PMAXUB/PMINUB (SSE2); halfword uses PMAXUW/PMINUW (SSE4.1);
        // word uses PMAXUD/PMINUD (SSE4.1).  .2D is reserved by the ARM ARM
        // (no AVX-512 PMAXUQ/PMINUQ dependency to worry about).
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xmax = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Movdqa(xmax, xn);
        switch (args.size) {
          case 0b00:
            as_.Pmaxub(xmax, xm);
            as_.Pminub(xn, xm);
            as_.Psubb(xmax, xn);
            break;
          case 0b01:
            as_.Pmaxuw(xmax, xm);
            as_.Pminuw(xn, xm);
            as_.Psubw(xmax, xn);
            break;
          case 0b10:
            as_.Pmaxud(xmax, xm);
            as_.Pminud(xn, xm);
            as_.Psubd(xmax, xn);
            break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xmax);
        StoreVReg(args.rd, xmax);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kSaba: {
        // SABA: Vd[lane] += |signed(Vn[lane]) - signed(Vm[lane])| per lane.
        // Same per-lane abs-diff recipe as kSabd (PMAXS - PMINS in modular
        // esize arithmetic gives the truncated-extended absolute difference,
        // including the INT_MIN-vs-INT_MAX wrap case where the modular
        // PSUB-then-sign-mask trick fails); then PADDx into Vd.
        // .2D / scalar 64-bit is reserved by the ARM ARM and falls back.
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xmax = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        LoadVReg(xd, args.rd);
        as_.Movdqa(xmax, xn);
        switch (args.size) {
          case 0b00:
            as_.Pmaxsb(xmax, xm);  // SSE4.1
            as_.Pminsb(xn, xm);    // SSE4.1
            as_.Psubb(xmax, xn);
            as_.Paddb(xd, xmax);
            break;
          case 0b01:
            as_.Pmaxsw(xmax, xm);  // SSE2
            as_.Pminsw(xn, xm);    // SSE2
            as_.Psubw(xmax, xn);
            as_.Paddw(xd, xmax);
            break;
          case 0b10:
            as_.Pmaxsd(xmax, xm);  // SSE4.1
            as_.Pminsd(xn, xm);    // SSE4.1
            as_.Psubd(xmax, xn);
            as_.Paddd(xd, xmax);
            break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xd);
        StoreVReg(args.rd, xd);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kUaba: {
        // UABA: Vd[lane] += |unsigned(Vn[lane]) - unsigned(Vm[lane])|.
        // Per-lane PMAXUx - PMINUx for the abs-diff, then PADDx into Vd.
        // Byte uses PMAXUB/PMINUB (SSE2); halfword uses PMAXUW/PMINUW
        // (SSE4.1); word uses PMAXUD/PMINUD (SSE4.1).  .2D is reserved.
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xmax = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        LoadVReg(xd, args.rd);
        as_.Movdqa(xmax, xn);
        switch (args.size) {
          case 0b00:
            as_.Pmaxub(xmax, xm);
            as_.Pminub(xn, xm);
            as_.Psubb(xmax, xn);
            as_.Paddb(xd, xmax);
            break;
          case 0b01:
            as_.Pmaxuw(xmax, xm);
            as_.Pminuw(xn, xm);
            as_.Psubw(xmax, xn);
            as_.Paddw(xd, xmax);
            break;
          case 0b10:
            as_.Pmaxud(xmax, xm);
            as_.Pminud(xn, xm);
            as_.Psubd(xmax, xn);
            as_.Paddd(xd, xmax);
            break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xd);
        StoreVReg(args.rd, xd);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kAdd: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        switch (args.size) {
          case 0b00: as_.Paddb(xn, xm); break;
          case 0b01: as_.Paddw(xn, xm); break;
          case 0b10: as_.Paddd(xn, xm); break;
          case 0b11: as_.Paddq(xn, xm); break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kSub: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        switch (args.size) {
          case 0b00: as_.Psubb(xn, xm); break;
          case 0b01: as_.Psubw(xn, xm); break;
          case 0b10: as_.Psubd(xn, xm); break;
          case 0b11: as_.Psubq(xn, xm); break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      // SQADD/UQADD/SQSUB/UQSUB vector for 8/16/32-bit lanes.
      // x86 SSE2 has direct saturating add/sub for byte (PADD{S,US}B /
      // PSUB{S,US}B) and halfword (PADD{S,US}W / PSUB{S,US}W) lanes, which
      // match ARM's per-lane signed/unsigned saturation semantics exactly.
      // 32-bit (.4S/.2S) lanes have no direct SSE saturating add/sub; emulate
      // via wrap-add/sub + PMAXUD/PMINUD-based overflow detect (unsigned) or
      // sign-bit / XOR-blend (signed). 64-bit lanes need AVX-512 PMAXUQ etc.
      // and bail to the interpreter.
      case Decoder::AdvSimdThreeSameOpcode::kSqadd: {
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        if (args.size == 0b00) {
          as_.Paddsb(xn, xm);
        } else if (args.size == 0b01) {
          as_.Paddsw(xn, xm);
        } else {
          // 32-bit signed saturating add: emulate via wrap-add + sign-bit
          // overflow detect (~(a^b) & (a^sum) has MSB set iff overflow).
          // sat = (a < 0) ? INT_MIN : INT_MAX = (a >> 31 signed) ^ 0x7FFFFFFF.
          // result = sum ^ ((sum ^ sat) & ovf_mask).
          SimdRegister t_sum = AllocTempSimdReg();
          SimdRegister t_ovf = AllocTempSimdReg();
          SimdRegister t_sat = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_sum, xn);
          as_.Paddd(t_sum, xm);                       // t_sum = a + b (mod 2^32).
          as_.Movdqa(t_ovf, xn);
          as_.Pxor(t_ovf, xm);                        // t_ovf = a ^ b.
          as_.Movdqa(t_sat, xn);
          as_.Pxor(t_sat, t_sum);                     // t_sat = a ^ sum.
          as_.Pcmpeqd(xm, xm);                        // xm = -1 (reuse: b not needed).
          as_.Pxor(t_ovf, xm);                        // t_ovf = ~(a ^ b).
          as_.Pand(t_ovf, t_sat);                     // t_ovf = ~(a^b) & (a^sum).
          as_.Psrad(t_ovf, int8_t{31});               // t_ovf = overflow mask.
          as_.Psrad(xn, int8_t{31});                  // xn = -1 if a<0, 0 else.
          as_.Psrld(xm, int8_t{1});                   // xm = 0x7FFFFFFF lane.
          as_.Pxor(xn, xm);                           // xn = (a<0)?INT_MIN:INT_MAX = sat.
          as_.Pxor(xn, t_sum);                        // xn = sat ^ sum.
          as_.Pand(xn, t_ovf);                        // xn = (sat ^ sum) & ovf_mask.
          as_.Pxor(xn, t_sum);                        // xn = sum ^ ((sat^sum) & ovf_mask).
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kUqadd: {
        if (args.size == 0b11) { Undefined(); return; }
        if (args.size == 0b10 && !host_platform::kHasSSE4_1) {
          Undefined(); return;                        // PMAXUD is SSE4.1.
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        if (args.size == 0b00) {
          as_.Paddusb(xn, xm);
        } else if (args.size == 0b01) {
          as_.Paddusw(xn, xm);
        } else {
          // 32-bit unsigned saturating add. sum = a + b (wrap). Overflow iff
          // sum < a (unsigned), detected via PMAXUD: max(a, sum) == sum iff no
          // overflow. Saturate overflowed lanes to UINT32_MAX.
          SimdRegister t_save_a = AllocTempSimdReg();
          SimdRegister t_ones = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_save_a, xn);                   // Preserve a before sum overwrites xn.
          as_.Paddd(xn, xm);                          // xn = sum = a + b.
          as_.Pmaxud(t_save_a, xn);                   // t_save_a = max(a, sum).
          as_.Pcmpeqd(t_save_a, xn);                  // -1 where sum == max, i.e. no overflow.
          as_.Pcmpeqd(t_ones, t_ones);                // t_ones = -1.
          as_.Pxor(t_save_a, t_ones);                 // invert: -1 where overflow.
          as_.Por(xn, t_save_a);                      // saturate overflowed lanes to UINT32_MAX.
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kSqsub: {
        if (args.size == 0b11) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        if (args.size == 0b00) {
          as_.Psubsb(xn, xm);
        } else if (args.size == 0b01) {
          as_.Psubsw(xn, xm);
        } else {
          // 32-bit signed saturating sub. Overflow iff (a^b) & (a^diff) has
          // MSB set; sat = (a < 0) ? INT_MIN : INT_MAX (same as SQADD).
          SimdRegister t_diff = AllocTempSimdReg();
          SimdRegister t_ovf = AllocTempSimdReg();
          SimdRegister t_sat = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_diff, xn);
          as_.Psubd(t_diff, xm);                      // t_diff = a - b (mod 2^32).
          as_.Movdqa(t_ovf, xn);
          as_.Pxor(t_ovf, xm);                        // t_ovf = a ^ b.
          as_.Movdqa(t_sat, xn);
          as_.Pxor(t_sat, t_diff);                    // t_sat = a ^ diff.
          as_.Pand(t_ovf, t_sat);                     // t_ovf = (a^b) & (a^diff).
          as_.Psrad(t_ovf, int8_t{31});               // overflow mask.
          as_.Psrad(xn, int8_t{31});                  // xn = -1 if a<0, 0 else.
          as_.Pcmpeqd(xm, xm);                        // xm = -1.
          as_.Psrld(xm, int8_t{1});                   // xm = 0x7FFFFFFF.
          as_.Pxor(xn, xm);                           // xn = sat = (a<0)?INT_MIN:INT_MAX.
          as_.Pxor(xn, t_diff);                       // xn = sat ^ diff.
          as_.Pand(xn, t_ovf);                        // xn = (sat ^ diff) & ovf_mask.
          as_.Pxor(xn, t_diff);                       // xn = diff ^ ((sat^diff) & ovf_mask).
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kUqsub: {
        if (args.size == 0b11) { Undefined(); return; }
        if (args.size == 0b10 && !host_platform::kHasSSE4_1) {
          Undefined(); return;                        // PMINUD is SSE4.1.
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        if (args.size == 0b00) {
          as_.Psubusb(xn, xm);
        } else if (args.size == 0b01) {
          as_.Psubusw(xn, xm);
        } else {
          // 32-bit unsigned saturating sub. result = (a >= b) ? a - b : 0.
          // Mask: PMINUD(a, b) == b iff a >= b.
          SimdRegister t_mask = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqa(t_mask, xn);
          as_.Pminud(t_mask, xm);                     // t_mask = min(a, b).
          as_.Pcmpeqd(t_mask, xm);                    // -1 where min == b, i.e. a >= b.
          as_.Psubd(xn, xm);                          // xn = a - b (wrap).
          as_.Pand(xn, t_mask);                       // zero out underflow lanes.
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      // ADDP (pairwise add) vector for all lane widths.
      // ARM ADDP concatenates Vn:Vm and adds adjacent pairs; the lower half of
      // the result comes from Vn pairs, the upper half from Vm pairs.
      //
      //   * 8H Q=1 / 4S Q=1 map directly to x86 PHADDW / PHADDD (SSSE3 — same
      //     pair-then-concat layout as ARM).
      //   * 4H Q=0 / 2S Q=0 use PHADDW/PHADDD then PSHUFD imm 0x08 to gather
      //     {xn pair-lo, xm pair-lo} into the low 64 bits before MaskLow64.
      //   * 16B Q=1 / 8B Q=0 have no PHADDB; emulate via PSRLW-8 + PADDB on
      //     each operand (even bytes hold pair sums), truncate each halfword
      //     to its low byte, then PACKUSWB to compact bytes. Q=0 additionally
      //     uses PUNPCKLDQ to interleave Vn's low dword with Vm's low dword.
      //   * 2D Q=1 has no PHADDQ; emulate via PSHUFD-0xEE (move high qword
      //     into low qword position), PADDQ, then PUNPCKLQDQ to combine.
      //     2D Q=0 (.1D) is reserved by ARM — bail.
      case Decoder::AdvSimdThreeSameOpcode::kAddp: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        switch (args.size) {
          case 0b00: {
            // Byte lanes: emulate byte-pairwise via halfword PSRLW + PADDB.
            SimdRegister tmp_n = AllocTempSimdReg();
            SimdRegister tmp_m = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqa(tmp_n, xn);
            as_.Movdqa(tmp_m, xm);
            // PSRLW(8): each halfword now has original high byte in low position,
            // zero in high position.
            as_.Psrlw(tmp_n, int8_t{8});
            as_.Psrlw(tmp_m, int8_t{8});
            // PADDB: even-indexed bytes now hold pair sums (low byte truncated).
            as_.Paddb(xn, tmp_n);
            as_.Paddb(xm, tmp_m);
            // Truncate each halfword to its low byte (clear the junk odd bytes).
            as_.Psllw(xn, int8_t{8});
            as_.Psrlw(xn, int8_t{8});
            as_.Psllw(xm, int8_t{8});
            as_.Psrlw(xm, int8_t{8});
            if (args.q) {
              // 16B: pack 8 halfwords from each into bytes [xn|xm].
              as_.Packuswb(xn, xm);
            } else {
              // 8B: only the first 4 halfwords of each are meaningful (the
              // upper 4 came from the don't-care upper halves of Vn/Vm).
              // Pack each separately into low 8 bytes, then interleave dwords:
              //   PUNPCKLDQ -> [xn_lo32, xm_lo32, xn_hi32, xm_hi32].
              // MaskLow64 below zeros the junk upper 64 bits.
              as_.Packuswb(xn, xn);
              as_.Packuswb(xm, xm);
              as_.Punpckldq(xn, xm);
            }
            break;
          }
          case 0b01: {
            // Halfword lanes: PHADDW pairs adjacent halfwords with ARM's
            // concat-then-pair layout for Q=1.
            as_.Phaddw(xn, xm);
            if (!args.q) {
              // 4H: PHADDW gave 8 halfwords [xn_p0..3, xm_p0..3]; we want
              // [xn_p0, xn_p1, xm_p0, xm_p1, junk, junk, junk, junk].
              // PSHUFD imm 0x08 selects dword 0 then dword 2.
              as_.Pshufd(xn, xn, int8_t{0x08});
            }
            break;
          }
          case 0b10: {
            // Single-word lanes: PHADDD pairs adjacent dwords.
            as_.Phaddd(xn, xm);
            if (!args.q) {
              // 2S: PHADDD gave 4 dwords [xn_p0, xn_p1, xm_p0, xm_p1]; we
              // want [xn_p0, xm_p0, junk, junk].  PSHUFD imm 0x08.
              as_.Pshufd(xn, xn, int8_t{0x08});
            }
            break;
          }
          case 0b11: {
            // .1D is reserved by ARM.
            if (!args.q) { Undefined(); return; }
            // 2D: emulate Vn[0]+Vn[1] and Vm[0]+Vm[1] separately, then combine.
            SimdRegister tmp = AllocTempSimdReg();
            if (!success()) { return; }
            // PSHUFD imm 0xEE = 0b11101110: each dst dword = src dword 2 or 3,
            // so tmp's low qword = src's high qword.
            as_.Pshufd(tmp, xn, static_cast<int8_t>(0xEE));
            as_.Paddq(xn, tmp);                 // xn[0] = Vn[0] + Vn[1]
            as_.Pshufd(tmp, xm, static_cast<int8_t>(0xEE));
            as_.Paddq(xm, tmp);                 // xm[0] = Vm[0] + Vm[1]
            // PUNPCKLQDQ: xn = [xn.low_qword, xm.low_qword].
            as_.Punpcklqdq(xn, xm);
            break;
          }
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kAnd: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Pand(xn, xm);
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kOrr: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Por(xn, xm);
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kEor: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Pxor(xn, xm);
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kCmeq: {
        // CMEQ Vd, Vn, Vm — lane-wise equality (-1 if equal, 0 otherwise).
        // 64-bit lanes need SSE4.1 (PCMPEQQ); other widths are SSE2.
        if (args.size == 0b11 && !host_platform::kHasSSE4_1) {
          Undefined(); return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        switch (args.size) {
          case 0b00: as_.Pcmpeqb(xn, xm); break;
          case 0b01: as_.Pcmpeqw(xn, xm); break;
          case 0b10: as_.Pcmpeqd(xn, xm); break;
          case 0b11: as_.Pcmpeqq(xn, xm); break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kCmtst: {
        // CMTST Vd, Vn, Vm — lane-wise (Vn & Vm) != 0 ? all-ones : 0.
        // Recipe: PAND(Vn, Vm), then PCMPEQ against zero (all-ones if the
        // lane is zero), then PXOR with all-ones to invert (so the non-zero
        // lanes become the set lanes).  .2D needs PCMPEQQ which is SSE4.1.
        if (args.size == 0b11 && !host_platform::kHasSSE4_1) {
          Undefined(); return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Pand(xn, xm);
        as_.Pxor(xm, xm);  // xm = 0.
        switch (args.size) {
          case 0b00: as_.Pcmpeqb(xn, xm); break;
          case 0b01: as_.Pcmpeqw(xn, xm); break;
          case 0b10: as_.Pcmpeqd(xn, xm); break;
          case 0b11: as_.Pcmpeqq(xn, xm); break;
          default: Undefined(); return;
        }
        // Materialize all-ones (size-independent) and invert: lanes where
        // (Vn & Vm) != 0 become all-ones, the rest stay zero.
        as_.Pcmpeqd(xm, xm);
        as_.Pxor(xn, xm);
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kShadd:
      case Decoder::AdvSimdThreeSameOpcode::kUhadd: {
        // SHADD / UHADD Vd, Vn, Vm — per-lane (a + b) >> 1 with no overflow,
        // signed (SHADD) or unsigned (UHADD).  ARM ARM C7.2 reserves .2D.
        //
        // Width recipes:
        //  - Byte (size=00): widen 8 bytes to 8 words (lo + hi halves
        //    separately), PADDW (sum fits in 9 bits, no overflow),
        //    arithmetic / logical shift right by 1, pack with signed /
        //    unsigned saturation.  Needs SSE4.1 for PMOVZXBW / PMOVSXBW
        //    (same recipe shape as kMul size=00).
        //  - Halfword (size=01) / Word (size=10): bitwise identity
        //    a + b = (a^b) + 2*(a&b), so (a + b) >> 1 = (a&b) + ((a^b) >> 1).
        //    Modular PADDx is exact here because the identity preserves
        //    the low esize bits.  Arithmetic shift (PSRAW/PSRAD) for signed,
        //    logical shift (PSRLW/PSRLD) for unsigned.  All SSE2.
        if (args.size == 0b11) { Undefined(); return; }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShadd);
        if (args.size == 0b00) {
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister xn_hi = AllocTempSimdReg();
          SimdRegister xm_hi = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);
          as_.Movdqa(xn_hi, xn);
          as_.Movdqa(xm_hi, xm);
          as_.Psrldq(xn_hi, int8_t{8});
          as_.Psrldq(xm_hi, int8_t{8});
          if (is_signed) {
            as_.Pmovsxbw(xn, xn);
            as_.Pmovsxbw(xm, xm);
            as_.Pmovsxbw(xn_hi, xn_hi);
            as_.Pmovsxbw(xm_hi, xm_hi);
          } else {
            as_.Pmovzxbw(xn, xn);
            as_.Pmovzxbw(xm, xm);
            as_.Pmovzxbw(xn_hi, xn_hi);
            as_.Pmovzxbw(xm_hi, xm_hi);
          }
          as_.Paddw(xn, xm);
          as_.Paddw(xn_hi, xm_hi);
          if (is_signed) {
            as_.Psraw(xn, int8_t{1});
            as_.Psraw(xn_hi, int8_t{1});
            as_.Packsswb(xn, xn_hi);
          } else {
            as_.Psrlw(xn, int8_t{1});
            as_.Psrlw(xn_hi, int8_t{1});
            as_.Packuswb(xn, xn_hi);
          }
          if (!args.q) MaskLow64(xn);
          StoreVReg(args.rd, xn);
          return;
        }
        // Halfword / word: bitwise (a&b) + ((a^b) >> 1).
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xand = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Movdqa(xand, xn);
        as_.Pand(xand, xm);
        as_.Pxor(xn, xm);
        if (args.size == 0b01) {
          if (is_signed) as_.Psraw(xn, int8_t{1});
          else as_.Psrlw(xn, int8_t{1});
          as_.Paddw(xn, xand);
        } else {
          // size == 0b10.
          if (is_signed) as_.Psrad(xn, int8_t{1});
          else as_.Psrld(xn, int8_t{1});
          as_.Paddd(xn, xand);
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kSrhadd:
      case Decoder::AdvSimdThreeSameOpcode::kUrhadd: {
        // SRHADD / URHADD Vd, Vn, Vm — per-lane (a + b + 1) >> 1, signed
        // (SRHADD) or unsigned (URHADD).  ARM ARM C7.2 reserves .2D.
        //
        // Width recipes:
        //  - URHADD byte (.8B/.16B): single PAVGB.  SSE2 PAVGB computes
        //    (a + b + 1) >> 1 unsigned per byte — an exact match.
        //  - URHADD halfword (.4H/.8H): single PAVGW, exact match.
        //  - SRHADD byte (.8B/.16B): widen-PADDW-add1-PSRAW-PACKSSWB.
        //    x86 has no PSRAB / PSRLB so byte signed must widen to 16-bit;
        //    +1 is materialized by PCMPEQW(xm, xm) (all -1 per word) followed
        //    by PSUBW (subtracting -1 = adding 1).  Lane values stay in
        //    [-128, 127] after the shift so PACKSSWB never saturates.
        //  - SRHADD halfword/word and URHADD word: bitwise identity
        //    (a + b + 1) >> 1 = (a | b) - ((a ^ b) >> 1), proven from
        //    a + b = (a^b) + 2*(a&b) and (a|b) = (a^b) + (a&b).  Arithmetic
        //    shift (PSRAW/PSRAD) for signed, logical shift (PSRLW/PSRLD)
        //    for unsigned.  All SSE2.
        if (args.size == 0b11) { Undefined(); return; }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrhadd);
        if (!is_signed && (args.size == 0b00 || args.size == 0b01)) {
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);
          if (args.size == 0b00) as_.Pavgb(xn, xm);
          else as_.Pavgw(xn, xm);
          if (!args.q) MaskLow64(xn);
          StoreVReg(args.rd, xn);
          return;
        }
        if (args.size == 0b00) {
          // SRHADD byte (signed): widen-PADDW-add1-PSRAW-PACKSSWB.
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister xn_hi = AllocTempSimdReg();
          SimdRegister xm_hi = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);
          as_.Movdqa(xn_hi, xn);
          as_.Movdqa(xm_hi, xm);
          as_.Psrldq(xn_hi, int8_t{8});
          as_.Psrldq(xm_hi, int8_t{8});
          as_.Pmovsxbw(xn, xn);
          as_.Pmovsxbw(xm, xm);
          as_.Pmovsxbw(xn_hi, xn_hi);
          as_.Pmovsxbw(xm_hi, xm_hi);
          as_.Paddw(xn, xm);
          as_.Paddw(xn_hi, xm_hi);
          // xm and xm_hi are dead.  Clobber xm to all-ones per word
          // (PCMPEQW(xm, xm)) so PSUBW(xn, xm) adds 1 to each lane.
          as_.Pcmpeqw(xm, xm);
          as_.Psubw(xn, xm);
          as_.Psubw(xn_hi, xm);
          as_.Psraw(xn, int8_t{1});
          as_.Psraw(xn_hi, int8_t{1});
          as_.Packsswb(xn, xn_hi);
          if (!args.q) MaskLow64(xn);
          StoreVReg(args.rd, xn);
          return;
        }
        // SRHADD halfword/word, URHADD word: bitwise (a|b) - ((a^b) >> 1).
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xxor = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Movdqa(xxor, xn);
        as_.Pxor(xxor, xm);
        as_.Por(xn, xm);
        if (args.size == 0b01) {
          if (is_signed) as_.Psraw(xxor, int8_t{1});
          else as_.Psrlw(xxor, int8_t{1});
          as_.Psubw(xn, xxor);
        } else {
          // size == 0b10.
          if (is_signed) as_.Psrad(xxor, int8_t{1});
          else as_.Psrld(xxor, int8_t{1});
          as_.Psubd(xn, xxor);
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kShsub:
      case Decoder::AdvSimdThreeSameOpcode::kUhsub: {
        // SHSUB / UHSUB Vd, Vn, Vm — per-lane (a - b) >> 1 with floor-
        // divide, signed (SHSUB) or unsigned (UHSUB).  ARM ARM C7.2
        // reserves .2D.
        //
        // Width recipes:
        //  - Byte (size=00): widen 8 bytes per half to 16-bit lanes
        //    via PMOVSXBW / PMOVZXBW, PSUBW, arithmetic / logical
        //    shift right by 1, PACKSSWB / PACKUSWB.  Unsigned needs
        //    a 0x00FF per-word mask before PACKUSWB so the modular
        //    PSRLW result of `a - b` for a < b (in [0x7F80, 0x7FFF])
        //    does not saturate up to 0xFF.
        //  - Halfword (size=01): same shape one width up.  PMOVSXWD /
        //    PMOVZXWD widen 4 halfwords per half to int32, PSUBD,
        //    PSRAD / PSRLD by 1, PACKSSDW / PACKUSDW.  Unsigned needs
        //    0x0000FFFF mask before PACKUSDW for the same reason.
        //  - Word (size=10): no SSE2 / SSE4.1 PSRAQ and no PACK 64→32
        //    with saturation, so PMOVSXDQ / PMOVZXDQ widen 2 dwords
        //    per half to int64, PSUBQ, PSRLQ by 1, then pack via
        //    PSHUFD (gather low dword of each qword) + PUNPCKLQDQ.
        //    PSRLQ is logical, but the bit-63 difference vs PSRAQ
        //    does not survive the mask-to-low-32 step in PSHUFD, so
        //    this is correct for both signed and unsigned (the
        //    widening PMOVSXDQ vs PMOVZXDQ is what distinguishes the
        //    two cases).
        if (args.size == 0b11) { Undefined(); return; }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShsub);
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xn_hi = AllocTempSimdReg();
        SimdRegister xm_hi = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Movdqa(xn_hi, xn);
        as_.Movdqa(xm_hi, xm);
        as_.Psrldq(xn_hi, int8_t{8});
        as_.Psrldq(xm_hi, int8_t{8});
        if (args.size == 0b00) {
          if (is_signed) {
            as_.Pmovsxbw(xn, xn);
            as_.Pmovsxbw(xm, xm);
            as_.Pmovsxbw(xn_hi, xn_hi);
            as_.Pmovsxbw(xm_hi, xm_hi);
            as_.Psubw(xn, xm);
            as_.Psubw(xn_hi, xm_hi);
            as_.Psraw(xn, int8_t{1});
            as_.Psraw(xn_hi, int8_t{1});
            as_.Packsswb(xn, xn_hi);
          } else {
            as_.Pmovzxbw(xn, xn);
            as_.Pmovzxbw(xm, xm);
            as_.Pmovzxbw(xn_hi, xn_hi);
            as_.Pmovzxbw(xm_hi, xm_hi);
            as_.Psubw(xn, xm);
            as_.Psubw(xn_hi, xm_hi);
            as_.Psrlw(xn, int8_t{1});
            as_.Psrlw(xn_hi, int8_t{1});
            // Clobber xm to 0x00FF per int16 so PACKUSWB only sees the
            // low byte of each lane (avoids 0x7F80..0x7FFF saturating).
            as_.Pcmpeqw(xm, xm);
            as_.Psrlw(xm, int8_t{8});
            as_.Pand(xn, xm);
            as_.Pand(xn_hi, xm);
            as_.Packuswb(xn, xn_hi);
          }
        } else if (args.size == 0b01) {
          if (is_signed) {
            as_.Pmovsxwd(xn, xn);
            as_.Pmovsxwd(xm, xm);
            as_.Pmovsxwd(xn_hi, xn_hi);
            as_.Pmovsxwd(xm_hi, xm_hi);
            as_.Psubd(xn, xm);
            as_.Psubd(xn_hi, xm_hi);
            as_.Psrad(xn, int8_t{1});
            as_.Psrad(xn_hi, int8_t{1});
            as_.Packssdw(xn, xn_hi);
          } else {
            as_.Pmovzxwd(xn, xn);
            as_.Pmovzxwd(xm, xm);
            as_.Pmovzxwd(xn_hi, xn_hi);
            as_.Pmovzxwd(xm_hi, xm_hi);
            as_.Psubd(xn, xm);
            as_.Psubd(xn_hi, xm_hi);
            as_.Psrld(xn, int8_t{1});
            as_.Psrld(xn_hi, int8_t{1});
            // Clobber xm to 0x0000FFFF per int32 for the same reason.
            as_.Pcmpeqd(xm, xm);
            as_.Psrld(xm, int8_t{16});
            as_.Pand(xn, xm);
            as_.Pand(xn_hi, xm);
            as_.Packusdw(xn, xn_hi);
          }
        } else {
          // size == 0b10 (word).
          if (is_signed) {
            as_.Pmovsxdq(xn, xn);
            as_.Pmovsxdq(xm, xm);
            as_.Pmovsxdq(xn_hi, xn_hi);
            as_.Pmovsxdq(xm_hi, xm_hi);
          } else {
            as_.Pmovzxdq(xn, xn);
            as_.Pmovzxdq(xm, xm);
            as_.Pmovzxdq(xn_hi, xn_hi);
            as_.Pmovzxdq(xm_hi, xm_hi);
          }
          as_.Psubq(xn, xm);
          as_.Psubq(xn_hi, xm_hi);
          as_.Psrlq(xn, int8_t{1});
          as_.Psrlq(xn_hi, int8_t{1});
          // PSHUFD imm 0x88 = dst[0]=src[0], dst[1]=src[2]: gathers the
          // two low-dword-of-each-int64-lane values into the low qword.
          as_.Pshufd(xn, xn, static_cast<int8_t>(0x88));
          as_.Pshufd(xn_hi, xn_hi, static_cast<int8_t>(0x88));
          as_.Punpcklqdq(xn, xn_hi);
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kBic: {
        // BIC Vd, Vn, Vm: Vd = Vn AND NOT Vm.  x86 PANDN dst, src computes
        // dst = NOT(dst) AND src, so PANDN(xm, xn) lands ~Vm & Vn in xm.
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Pandn(xm, xn);
        if (!args.q) MaskLow64(xm);
        StoreVReg(args.rd, xm);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kOrn: {
        // ORN Vd, Vn, Vm: Vd = Vn OR NOT Vm.  Materialize ones via PCMPEQD self,
        // XOR into xm to invert, then OR with xn.
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister ones = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Pcmpeqd(ones, ones);
        as_.Pxor(xm, ones);
        as_.Por(xn, xm);
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kBsl: {
        // BSL Vd, Vn, Vm: Vd = (Vd AND Vn) | (NOT Vd AND Vm).
        // Bitwise identity: ((Vn XOR Vm) AND Vd) XOR Vm equals the BSL result.
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        LoadVReg(xd, args.rd);
        as_.Pxor(xn, xm);
        as_.Pand(xn, xd);
        as_.Pxor(xn, xm);
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kBit: {
        // BIT Vd, Vn, Vm: Vd = (Vm AND Vn) | (NOT Vm AND Vd) — "insert if true".
        // Bitwise identity: ((Vd XOR Vn) AND Vm) XOR Vd.
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        LoadVReg(xd, args.rd);
        as_.Pxor(xn, xd);
        as_.Pand(xn, xm);
        as_.Pxor(xn, xd);
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kBif: {
        // BIF Vd, Vn, Vm: Vd = (Vm AND Vd) | (NOT Vm AND Vn) — "insert if false".
        // Bitwise identity: ((Vd XOR Vn) AND NOT Vm) XOR Vd.  Lower as a single
        // PANDN to fold the NOT into the AND.
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        LoadVReg(xd, args.rd);
        as_.Pxor(xn, xd);
        as_.Pandn(xm, xn);
        as_.Pxor(xd, xm);
        if (!args.q) MaskLow64(xd);
        StoreVReg(args.rd, xd);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kCmgt: {
        // CMGT Vd, Vn, Vm: lane-wise signed greater-than; result lane is
        // all-ones if Vn > Vm.  64-bit lanes need SSE4.2 (PCMPGTQ); other
        // widths are SSE2.
        if (args.size == 0b11 && !host_platform::kHasSSE4_2) {
          Undefined(); return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        switch (args.size) {
          case 0b00: as_.Pcmpgtb(xn, xm); break;
          case 0b01: as_.Pcmpgtw(xn, xm); break;
          case 0b10: as_.Pcmpgtd(xn, xm); break;
          case 0b11: as_.Pcmpgtq(xn, xm); break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kCmhi: {
        // CMHI Vd, Vn, Vm: lane-wise unsigned greater-than.  x86 has no
        // unsigned vector compare, so flip the per-lane sign bit in both
        // operands (XOR with 0x80…) and then use the signed PCMPGT*: this
        // converts the unsigned ordering into the signed ordering that
        // PCMPGT* implements.  64-bit lanes need SSE4.2.
        if (args.size == 0b11 && !host_platform::kHasSSE4_2) {
          Undefined(); return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister sign = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Pcmpeqd(sign, sign);  // 0xFFFFFFFF per dword.
        switch (args.size) {
          case 0b00: {
            // Need 0x80 in every byte.  Build via a GPR broadcast since
            // there's no PSLLB shift on x86.
            Register tmp = AllocTempReg();
            if (tmp == Assembler::no_register) { Undefined(); return; }
            as_.Movq(tmp, int64_t{static_cast<int64_t>(0x8080808080808080ULL)});
            as_.Movq(sign, tmp);
            as_.Punpcklqdq(sign, sign);
            break;
          }
          case 0b01: as_.Psllw(sign, int8_t{15}); break;
          case 0b10: as_.Pslld(sign, int8_t{31}); break;
          case 0b11: as_.Psllq(sign, int8_t{63}); break;
          default: Undefined(); return;
        }
        as_.Pxor(xn, sign);
        as_.Pxor(xm, sign);
        switch (args.size) {
          case 0b00: as_.Pcmpgtb(xn, xm); break;
          case 0b01: as_.Pcmpgtw(xn, xm); break;
          case 0b10: as_.Pcmpgtd(xn, xm); break;
          case 0b11: as_.Pcmpgtq(xn, xm); break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      // CMGE/CMHS vector via PCMPGT + invert (sign-flip for CMHS)
      case Decoder::AdvSimdThreeSameOpcode::kCmge: {
        // CMGE Vd, Vn, Vm: lane-wise signed greater-than-or-equal.
        // x86 has no PCMPGE; compute NOT(Vm > Vn) instead, since
        //   (Vn >= Vm)  ==  !(Vm > Vn)  for total signed orderings.
        // 64-bit lanes still need SSE4.2 (PCMPGTQ); other widths are SSE2.
        if (args.size == 0b11 && !host_platform::kHasSSE4_2) {
          Undefined(); return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister ones = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        // PCMPGT(xm, xn) -> xm > xn per lane.  Then XOR with all-ones to
        // invert into (xn >= xm).
        switch (args.size) {
          case 0b00: as_.Pcmpgtb(xm, xn); break;
          case 0b01: as_.Pcmpgtw(xm, xn); break;
          case 0b10: as_.Pcmpgtd(xm, xn); break;
          case 0b11: as_.Pcmpgtq(xm, xn); break;
          default: Undefined(); return;
        }
        as_.Pcmpeqd(ones, ones);  // 0xFFFFFFFF per dword (== all-ones).
        as_.Pxor(xm, ones);
        if (!args.q) MaskLow64(xm);
        StoreVReg(args.rd, xm);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kCmhs: {
        // CMHS Vd, Vn, Vm: lane-wise unsigned greater-than-or-equal.
        // Same shape as CMGE, but flip per-lane sign bits on both operands
        // first to map unsigned ordering onto the signed PCMPGT*.
        // 64-bit lanes need SSE4.2.
        if (args.size == 0b11 && !host_platform::kHasSSE4_2) {
          Undefined(); return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister sign = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        as_.Pcmpeqd(sign, sign);  // 0xFFFFFFFF per dword; specialized below.
        switch (args.size) {
          case 0b00: {
            // Need 0x80 in every byte (no PSLLB on x86).
            Register tmp = AllocTempReg();
            if (tmp == Assembler::no_register) { Undefined(); return; }
            as_.Movq(tmp, int64_t{static_cast<int64_t>(0x8080808080808080ULL)});
            as_.Movq(sign, tmp);
            as_.Punpcklqdq(sign, sign);
            break;
          }
          case 0b01: as_.Psllw(sign, int8_t{15}); break;
          case 0b10: as_.Pslld(sign, int8_t{31}); break;
          case 0b11: as_.Psllq(sign, int8_t{63}); break;
          default: Undefined(); return;
        }
        as_.Pxor(xn, sign);
        as_.Pxor(xm, sign);
        // Reuse `sign` as the all-ones invert mask after this point.
        switch (args.size) {
          case 0b00: as_.Pcmpgtb(xm, xn); break;
          case 0b01: as_.Pcmpgtw(xm, xn); break;
          case 0b10: as_.Pcmpgtd(xm, xn); break;
          case 0b11: as_.Pcmpgtq(xm, xn); break;
          default: Undefined(); return;
        }
        as_.Pcmpeqd(sign, sign);  // 0xFFFFFFFF per dword.
        as_.Pxor(xm, sign);
        if (!args.q) MaskLow64(xm);
        StoreVReg(args.rd, xm);
        return;
      }
      // SMAX/SMIN/UMAX/UMIN vector via PMAXS*/PMINS*/PMAXU*/PMINU*
      case Decoder::AdvSimdThreeSameOpcode::kSmax:
      case Decoder::AdvSimdThreeSameOpcode::kSmin:
      case Decoder::AdvSimdThreeSameOpcode::kUmax:
      case Decoder::AdvSimdThreeSameOpcode::kUmin: {
        // S{MAX,MIN} / U{MAX,MIN} Vd.<T>, Vn.<T>, Vm.<T>: lane-wise signed
        // or unsigned min/max.  x86 has direct lane-width-matched
        // PMAXS/PMINS/PMAXU/PMINU instructions for all four arithmetic
        // widths (8/16/32/64), except the .2D (64-bit) lane which has no
        // SSE-era instruction (PMAXSQ/PMINSQ/PMAXUQ/PMINUQ only land with
        // AVX-512F-VL).  Bail .2D to the interpreter; bail any size 0b00/0b01
        // /0b10 case that needs SSE4.1 when SSE4.1 isn't available.
        //
        // Per-size SSE feature gate (table from Intel SDM Vol 2):
        //   PMAXSB/PMINSB/PMAXSD/PMINSD/PMAXUW/PMINUW/PMAXUD/PMINUD -> SSE4.1.
        //   PMAXSW/PMINSW/PMAXUB/PMINUB                              -> SSE2.
        if (args.size == 0b11) {
          // No SSE/AVX-2 instruction for 64-bit lanes.  Bail.
          success_ = false;
          return;
        }
        const bool is_max =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
             args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax);
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
             args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin);
        // Need SSE4.1 for: signed-byte (PMAXSB/PMINSB), signed-dword
        // (PMAXSD/PMINSD), unsigned-word (PMAXUW/PMINUW), unsigned-dword
        // (PMAXUD/PMINUD).  SSE2 covers signed-word and unsigned-byte.
        const bool needs_sse4_1 =
            (is_signed && args.size == 0b00) ||
            (is_signed && args.size == 0b10) ||
            (!is_signed && args.size == 0b01) ||
            (!is_signed && args.size == 0b10);
        if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
          success_ = false;
          return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        // xn := op(xn, xm) lane-wise.
        switch (args.size) {
          case 0b00:  // .16B / .8B
            if (is_signed) {
              if (is_max) as_.Pmaxsb(xn, xm); else as_.Pminsb(xn, xm);
            } else {
              if (is_max) as_.Pmaxub(xn, xm); else as_.Pminub(xn, xm);
            }
            break;
          case 0b01:  // .8H / .4H
            if (is_signed) {
              if (is_max) as_.Pmaxsw(xn, xm); else as_.Pminsw(xn, xm);
            } else {
              if (is_max) as_.Pmaxuw(xn, xm); else as_.Pminuw(xn, xm);
            }
            break;
          case 0b10:  // .4S / .2S
            if (is_signed) {
              if (is_max) as_.Pmaxsd(xn, xm); else as_.Pminsd(xn, xm);
            } else {
              if (is_max) as_.Pmaxud(xn, xm); else as_.Pminud(xn, xm);
            }
            break;
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      // SMAXP/SMINP/UMAXP/UMINP vector (pairwise)
      case Decoder::AdvSimdThreeSameOpcode::kSmaxp:
      case Decoder::AdvSimdThreeSameOpcode::kSminp:
      case Decoder::AdvSimdThreeSameOpcode::kUmaxp:
      case Decoder::AdvSimdThreeSameOpcode::kUminp: {
        // S{MAX,MIN}P / U{MAX,MIN}P Vd.<T>, Vn.<T>, Vm.<T>: pairwise lane-wise
        // signed or unsigned max/min.  Vd = pair(Vn) || pair(Vm), where
        //   pair(X)[i] = op(X[2i], X[2i+1]).
        // No SSE/AVX2 horizontal-pairwise max/min instruction exists (PHADDW/D
        // is the additive analog); emulate via PSHUFB-based even/odd gather
        // (byte/halfword lanes) or PSHUFD-based even/odd lift (dword lanes),
        // then lane-wise PMAX/PMIN, then concatenate the Vn and Vm partial
        // results.
        //
        // Size 0b11 (.2D / .1D): no PMAXSQ/PMINSQ/PMAXUQ/PMINUQ before
        // AVX-512F-VL; bail to the interpreter.  size=0b11 Q=0 (.1D) is
        // ARM-reserved and the interpreter rejects it too.
        if (args.size == 0b11) {
          success_ = false;
          return;
        }
        const bool is_max =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
             args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmaxp);
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
             args.opcode == Decoder::AdvSimdThreeSameOpcode::kSminp);
        // SSE-feature gate matches the three-same SMAX/SMIN/UMAX/UMIN gate:
        //   PMAXSB/PMINSB/PMAXSD/PMINSD/PMAXUW/PMINUW/PMAXUD/PMINUD -> SSE4.1.
        //   PMAXSW/PMINSW/PMAXUB/PMINUB                              -> SSE2.
        const bool needs_sse4_1 =
            (is_signed && args.size == 0b00) ||
            (is_signed && args.size == 0b10) ||
            (!is_signed && args.size == 0b01) ||
            (!is_signed && args.size == 0b10);
        if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
          success_ = false;
          return;
        }
        // Byte/halfword lanes need PSHUFB (SSSE3).  Dword lanes use only PSHUFD
        // (SSE2).
        const bool needs_ssse3 = (args.size == 0b00 || args.size == 0b01);
        if (needs_ssse3 && !host_platform::kHasSSSE3) {
          success_ = false;
          return;
        }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        // Selector lambda: applies lane-wise PMAX/PMIN of {is_signed, is_max} ×
        // width to (dst, src) operands.
        auto pmax_pmin = [&](SimdRegister dst, SimdRegister src) {
          switch (args.size) {
            case 0b00:
              if (is_signed) {
                if (is_max) as_.Pmaxsb(dst, src); else as_.Pminsb(dst, src);
              } else {
                if (is_max) as_.Pmaxub(dst, src); else as_.Pminub(dst, src);
              }
              break;
            case 0b01:
              if (is_signed) {
                if (is_max) as_.Pmaxsw(dst, src); else as_.Pminsw(dst, src);
              } else {
                if (is_max) as_.Pmaxuw(dst, src); else as_.Pminuw(dst, src);
              }
              break;
            case 0b10:
              if (is_signed) {
                if (is_max) as_.Pmaxsd(dst, src); else as_.Pminsd(dst, src);
              } else {
                if (is_max) as_.Pmaxud(dst, src); else as_.Pminud(dst, src);
              }
              break;
          }
        };
        switch (args.size) {
          case 0b00:    // .16B (Q=1) / .8B (Q=0)
          case 0b01: {  // .8H (Q=1) / .4H (Q=0)
            // Build PSHUFB even-mask and odd-mask in xmm registers via two
            // Pinsrq's from a temp GP register.  Even-mask gathers the
            // even-indexed lanes of the source into the low 8 bytes of the
            // mask result; odd-mask gathers the odd-indexed lanes.  Mask
            // upper qword = 0x8080...80 so PSHUFB writes 0 into the upper
            // half (don't-care for our pairwise result).
            //
            // size=00 byte: even = {0,2,4,6,8,10,12,14}; odd = {1,3,...,15}.
            // size=01 word: even = {h0,h2,h4,h6}; odd = {h1,h3,h5,h7}.
            //   Byte indices for the word form:
            //     even = {0,1, 4,5, 8,9, 12,13};
            //     odd  = {2,3, 6,7, 10,11, 14,15}.
            SimdRegister even_mask = AllocTempSimdReg();
            SimdRegister odd_mask = AllocTempSimdReg();
            SimdRegister evens_n = AllocTempSimdReg();
            SimdRegister odds_n = AllocTempSimdReg();
            SimdRegister evens_m = AllocTempSimdReg();
            SimdRegister odds_m = AllocTempSimdReg();
            Register r1 = AllocTempReg();
            if (even_mask == no_simd_register || odd_mask == no_simd_register ||
                evens_n == no_simd_register || odds_n == no_simd_register ||
                evens_m == no_simd_register || odds_m == no_simd_register ||
                r1 == no_register) {
              success_ = false; return;
            }
            int64_t even_lo, even_hi, odd_lo, odd_hi;
            if (args.size == 0b00) {
              // Byte: {0,2,4,6,8,10,12,14, 0x80*8} and {1,3,5,7,9,11,13,15, 0x80*8}.
              even_lo = static_cast<int64_t>(0x0E0C0A0806040200LL);
              even_hi = static_cast<int64_t>(0x8080808080808080ULL);
              odd_lo  = static_cast<int64_t>(0x0F0D0B0907050301LL);
              odd_hi  = static_cast<int64_t>(0x8080808080808080ULL);
            } else {
              // Halfword: {0,1,4,5,8,9,12,13, 0x80*8} and {2,3,6,7,10,11,14,15, 0x80*8}.
              even_lo = static_cast<int64_t>(0x0D0C090805040100LL);
              even_hi = static_cast<int64_t>(0x8080808080808080ULL);
              odd_lo  = static_cast<int64_t>(0x0F0E0B0A07060302LL);
              odd_hi  = static_cast<int64_t>(0x8080808080808080ULL);
            }
            as_.Movq(r1, even_lo);
            as_.Movq(even_mask, r1);
            as_.Movq(r1, even_hi);
            as_.Pinsrq(even_mask, r1, int8_t{1});
            as_.Movq(r1, odd_lo);
            as_.Movq(odd_mask, r1);
            as_.Movq(r1, odd_hi);
            as_.Pinsrq(odd_mask, r1, int8_t{1});

            // pair_n into low 8 bytes of xn:
            as_.Movdqa(evens_n, xn);
            as_.Pshufb(evens_n, even_mask);
            as_.Movdqa(odds_n, xn);
            as_.Pshufb(odds_n, odd_mask);
            pmax_pmin(evens_n, odds_n);  // low 8 bytes hold pair(Vn)

            // pair_m into low 8 bytes of xm:
            as_.Movdqa(evens_m, xm);
            as_.Pshufb(evens_m, even_mask);
            as_.Movdqa(odds_m, xm);
            as_.Pshufb(odds_m, odd_mask);
            pmax_pmin(evens_m, odds_m);  // low 8 bytes hold pair(Vm)

            // Concatenate the two 8-byte partials.  For Q=1 (.16B / .8H),
            // both partials are full 8 bytes wide and PUNPCKLQDQ gives the
            // full 16-byte result.  For Q=0 (.8B / .4H), only the low 4
            // bytes of each partial are meaningful; PUNPCKLDQ packs those
            // four-byte halves end-to-end and the MaskLow64 tail below
            // zeroes the don't-care upper 64 bits.
            if (args.q) {
              as_.Punpcklqdq(evens_n, evens_m);
            } else {
              as_.Punpckldq(evens_n, evens_m);
            }
            as_.Movdqa(xn, evens_n);
            break;
          }
          case 0b10: {  // .4S (Q=1) / .2S (Q=0)
            // Dword pairwise: PSHUFD imm 0x88 picks even-indexed dwords of
            // a source into both halves of the destination ({d0,d2,d0,d2});
            // imm 0xDD picks odd-indexed dwords ({d1,d3,d1,d3}).  Lane-wise
            // PMAX/PMIN of the two halves yields pair(X) replicated in both
            // halves; PUNPCKLQDQ then concatenates the low qwords of pair_n
            // and pair_m into [pair_n[0], pair_n[1], pair_m[0], pair_m[1]]
            // — exactly the .4S pairwise result.
            //
            // For Q=0 (.2S), the upper qword of Vn / Vm is don't-care; after
            // PSHUFD 0x88 / 0xDD the dwords at positions 1,3 of the result
            // are junk, but the PMAX/PMIN at position 0 still gives the
            // correct max(n0, n1) / max(m0, m1).  After PUNPCKLQDQ the
            // result is [max(n0,n1), junk, max(m0,m1), junk]; a final
            // PSHUFD 0x08 packs positions 0 and 2 into 0 and 1, and the
            // MaskLow64 tail zeroes the don't-care upper 64 bits.
            SimdRegister evens_n = AllocTempSimdReg();
            SimdRegister odds_n = AllocTempSimdReg();
            SimdRegister evens_m = AllocTempSimdReg();
            SimdRegister odds_m = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Pshufd(evens_n, xn, static_cast<int8_t>(0x88));
            as_.Pshufd(odds_n, xn, static_cast<int8_t>(0xDD));
            pmax_pmin(evens_n, odds_n);  // pair(Vn) replicated in both halves
            as_.Pshufd(evens_m, xm, static_cast<int8_t>(0x88));
            as_.Pshufd(odds_m, xm, static_cast<int8_t>(0xDD));
            pmax_pmin(evens_m, odds_m);  // pair(Vm) replicated in both halves
            if (args.q) {
              as_.Punpcklqdq(evens_n, evens_m);
            } else {
              // Q=0: low qword of evens_n holds {max(n0,n1), junk}; low qword
              // of evens_m holds {max(m0,m1), junk}.  PUNPCKLQDQ gives
              // [max(n0,n1), junk, max(m0,m1), junk]; PSHUFD 0x08 packs
              // dwords 0 and 2 into positions 0 and 1.
              as_.Punpcklqdq(evens_n, evens_m);
              as_.Pshufd(evens_n, evens_n, static_cast<int8_t>(0x08));
            }
            as_.Movdqa(xn, evens_n);
            break;
          }
          default: Undefined(); return;
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kFaddV:
      case Decoder::AdvSimdThreeSameOpcode::kFsubV:
      case Decoder::AdvSimdThreeSameOpcode::kFmulV:
      case Decoder::AdvSimdThreeSameOpcode::kFdivV: {
        // FP16 vector FADD/FSUB/FMUL/FDIV via F16C round-trip (below).
        // FP32 (.2S/.4S) and FP64 (.2D) lower directly to SSE2 packed FP
        // arithmetic: ADDPS/PD, SUBPS/PD, MULPS/PD, DIVPS/PD.  args.size=0
        // selects FP32 (4-byte lanes); args.size=1 selects FP64 (8-byte
        // lanes; the decoder already rejects sz=1 with Q=0).
        if (!args.is_fp16) {
          const bool is_double = (args.size & 1);
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);
          if (is_double) {
            switch (args.opcode) {
              case Decoder::AdvSimdThreeSameOpcode::kFaddV: as_.Addpd(xn, xm); break;
              case Decoder::AdvSimdThreeSameOpcode::kFsubV: as_.Subpd(xn, xm); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmulV: as_.Mulpd(xn, xm); break;
              case Decoder::AdvSimdThreeSameOpcode::kFdivV: as_.Divpd(xn, xm); break;
              default: success_ = false; return;
            }
          } else {
            switch (args.opcode) {
              case Decoder::AdvSimdThreeSameOpcode::kFaddV: as_.Addps(xn, xm); break;
              case Decoder::AdvSimdThreeSameOpcode::kFsubV: as_.Subps(xn, xm); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmulV: as_.Mulps(xn, xm); break;
              case Decoder::AdvSimdThreeSameOpcode::kFdivV: as_.Divps(xn, xm); break;
              default: success_ = false; return;
            }
          }
          // .2S (q=0, FP32 only) zeroes the upper 64 bits of the destination.
          if (!args.q) MaskLow64(xn);
          StoreVReg(args.rd, xn);
          return;
        }
        if (!host_platform::kHasF16C) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        auto fp_op = [&](SimdRegister dst, SimdRegister src) {
          switch (args.opcode) {
            case Decoder::AdvSimdThreeSameOpcode::kFaddV: as_.Addps(dst, src); break;
            case Decoder::AdvSimdThreeSameOpcode::kFsubV: as_.Subps(dst, src); break;
            case Decoder::AdvSimdThreeSameOpcode::kFmulV: as_.Mulps(dst, src); break;
            case Decoder::AdvSimdThreeSameOpcode::kFdivV: as_.Divps(dst, src); break;
            default: break;  // unreachable
          }
        };
        if (!args.q) {
          // .4H: 4 FP16 lanes in low 64 bits of each operand.
          as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Vcvtph2ps(xn, xn);
          as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
          as_.Vcvtph2ps(xm, xm);
          fp_op(xn, xm);
          as_.Vcvtps2ph(xn, xn, int8_t{0});
          // Vcvtps2ph auto-zeroes upper 64 bits.
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        } else {
          // .8H: process low 4 lanes, then high 4 lanes, then recombine.
          SimdRegister xn_hi = AllocTempSimdReg();
          SimdRegister xm_hi = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xn_hi, {.base = Assembler::rbp, .disp = vn_off});
          as_.Movdqa(xn, xn_hi);
          as_.Vcvtph2ps(xn, xn);
          as_.Psrldq(xn_hi, int8_t{8});
          as_.Vcvtph2ps(xn_hi, xn_hi);
          as_.Movdqu(xm_hi, {.base = Assembler::rbp, .disp = vm_off});
          as_.Movdqa(xm, xm_hi);
          as_.Vcvtph2ps(xm, xm);
          as_.Psrldq(xm_hi, int8_t{8});
          as_.Vcvtph2ps(xm_hi, xm_hi);
          fp_op(xn, xm);
          fp_op(xn_hi, xm_hi);
          as_.Vcvtps2ph(xn, xn, int8_t{0});
          as_.Vcvtps2ph(xn_hi, xn_hi, int8_t{0});
          as_.Pslldq(xn_hi, int8_t{8});
          as_.Por(xn, xn_hi);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        }
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kFabdV: {
        // FP vector FABD = |a - b|: a packed SUB then a sign-clear (bitwise AND
        // with 0x7FFF…). ARM FABD is FPAbs(FPSub(a,b)); x86 SUBP{S,D} matches
        // ARM FPSub lane-for-lane under default rounding, and clearing the sign
        // bit yields FPAbs — including on NaN, where ARM FPAbs also clears the
        // sign bit. The sign-clear constant is built in an XMM temp with the
        // `PCMPEQD self ; PSRLD/PSRLQ 1` idiom (avoids a rodata load).
        //
        // FP32 (.2S/.4S, args.size=0b00) and FP64 (.2D, args.size=0b01) lower
        // directly; FP16 (.4H/.8H) uses an F16C round-trip: widen each operand
        // half to FP32, compute (a-b) at FP32, sign-clear the FP32 result, then
        // narrow. The F16C round-trip is bit-exact for FP16 FSUB (FP32's
        // mantissa strictly contains FP16's) and the subsequent AND is exact by
        // construction, so it composes cleanly. .1D (Q=0, size=01) is
        // ARM-reserved and rejected by the decoder before reaching here.
        if (!args.is_fp16) {
          const bool is_double = (args.size & 1);
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister mask = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);
          if (is_double) as_.Subpd(xn, xm);
          else           as_.Subps(xn, xm);
          as_.Pcmpeqd(mask, mask);
          if (is_double) as_.Psrlq(mask, int8_t{1});  // 0x7FFFFFFFFFFFFFFF/qword
          else           as_.Psrld(mask, int8_t{1});  // 0x7FFFFFFF/dword
          as_.Pand(xn, mask);
          // .2S (q=0, FP32) zeroes the destination's upper 64 bits.
          if (!args.q) MaskLow64(xn);
          StoreVReg(args.rd, xn);
          return;
        }
        if (!host_platform::kHasF16C) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister mask = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pcmpeqd(mask, mask);
        as_.Psrld(mask, int8_t{1});  // 0x7FFFFFFF per dword (FP32 sign-clear).
        if (!args.q) {
          // .4H: 4 FP16 lanes in the low 64 bits of each operand.
          as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Vcvtph2ps(xn, xn);
          as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
          as_.Vcvtph2ps(xm, xm);
          as_.Subps(xn, xm);
          as_.Pand(xn, mask);
          as_.Vcvtps2ph(xn, xn, int8_t{0});
          // Vcvtps2ph auto-zeroes the upper 64 bits.
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        } else {
          // .8H: process low 4 lanes, then high 4 lanes, then recombine.
          SimdRegister xn_hi = AllocTempSimdReg();
          SimdRegister xm_hi = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xn_hi, {.base = Assembler::rbp, .disp = vn_off});
          as_.Movdqa(xn, xn_hi);
          as_.Vcvtph2ps(xn, xn);
          as_.Psrldq(xn_hi, int8_t{8});
          as_.Vcvtph2ps(xn_hi, xn_hi);
          as_.Movdqu(xm_hi, {.base = Assembler::rbp, .disp = vm_off});
          as_.Movdqa(xm, xm_hi);
          as_.Vcvtph2ps(xm, xm);
          as_.Psrldq(xm_hi, int8_t{8});
          as_.Vcvtph2ps(xm_hi, xm_hi);
          as_.Subps(xn, xm);
          as_.Subps(xn_hi, xm_hi);
          as_.Pand(xn, mask);
          as_.Pand(xn_hi, mask);
          as_.Vcvtps2ph(xn, xn, int8_t{0});
          as_.Vcvtps2ph(xn_hi, xn_hi, int8_t{0});
          as_.Pslldq(xn_hi, int8_t{8});
          as_.Por(xn, xn_hi);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        }
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kFcmeqV:
      case Decoder::AdvSimdThreeSameOpcode::kFcmgeV:
      case Decoder::AdvSimdThreeSameOpcode::kFcmgtV:
      case Decoder::AdvSimdThreeSameOpcode::kFacgeV:
      case Decoder::AdvSimdThreeSameOpcode::kFacgtV: {
        // FP16 vector FP compares via F16C round-trip.
        //   FCMEQ: a == b
        //   FCMGE: a >= b
        //   FCMGT: a >  b
        //   FACGE: |a| >= |b|
        //   FACGT: |a| >  |b|
        // ARM result lane is all-ones (0xFFFF) on TRUE, zero on FALSE, and is
        // false for any unordered (NaN-involving) compare.
        //
        // Lowering: widen each operand half to FP32, do a 4-lane FP32 compare
        // producing 0xFFFFFFFF / 0x00000000 dwords, then narrow via PACKSSDW.
        // PACKSSDW signed-saturates each 32-bit lane to int16: 0xFFFFFFFF
        // (signed -1) saturates to 0xFFFF; 0x00000000 stays 0x0000. That
        // matches ARM's bit-mask result directly.
        //
        //   FCMEQ: Cmpeqps xn, xm                  (imm=0, EQ_OQ)
        //   FCMGE: Cmpleps xm, xn ; Movdqa xn, xm  ((xm<=xn) == (xn>=xm))
        //   FCMGT: Cmpltps xm, xn ; Movdqa xn, xm  ((xm<xn)  == (xn>xm))
        //   FACGE: pre-mask both operands with 0x7FFFFFFF (FP32 sign-clear),
        //          then FCMGE shape
        //   FACGT: pre-mask, then FCMGT shape
        //
        // The legacy SSE compare predicates Cmpeqps/Cmpltps/Cmpleps are all
        // ordered: they return 0 (false) for any NaN operand, matching ARM.
        // The sign-clear mask is built once with the `Pcmpeqd self ; Psrld 1`
        // idiom — same as the FABD lowering above — and shared across both
        // halves of a .8H lowering.
        //
        // For .8H, `Packssdw(low_mask, hi_mask)` is a one-instruction
        // recombine: low 64 bits of the destination hold the packed low_mask
        // (4 16-bit lanes for FP16 lanes 0..3), upper 64 bits hold the packed
        // hi_mask (lanes 4..7). No Pslldq+Por dance needed.
        //
        // FP16 path uses an F16C round-trip (Vcvtph2ps + 128-bit FP32 compare +
        // Packssdw narrow).  FP32 / FP64 paths run the 128-bit compare directly
        // on the loaded operands: same five-mnemonic shape (Cmpeqps/Cmpleps/
        // Cmpltps for FP32, Cmpeqpd/Cmplepd/Cmpltpd for FP64).  Mask emission
        // is bit-identical with the FP16 narrow path because PACKSSDW saturates
        // 0xFFFFFFFF -> 0xFFFF and 0x00000000 -> 0x0000 — the same bit pattern
        // FP32/FP64 packed compares write per lane (all-ones on TRUE, zero on
        // FALSE, FALSE for unordered).
        using Op = Decoder::AdvSimdThreeSameOpcode;
        const bool is_eq  = (args.opcode == Op::kFcmeqV);
        const bool is_ge  = (args.opcode == Op::kFcmgeV ||
                             args.opcode == Op::kFacgeV);
        const bool is_abs = (args.opcode == Op::kFacgeV ||
                             args.opcode == Op::kFacgtV);
        if (!args.is_fp16) {
          // FP32 (args.size = 0b00) or FP64 (args.size = 0b01).
          // .1D (Q=0, size=01) is ARM-reserved; the decoder routes it to
          // Undefined() before reaching here.  Defensive bail keeps the JIT
          // symmetric with the rest of the FP three-same lane matrix.
          if (args.size == 0b01 && !args.q) { Undefined(); return; }
          if (args.size != 0b00 && args.size != 0b01) { Undefined(); return; }
          const bool is_double = (args.size == 0b01);
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});
          if (is_abs) {
            // Sign-clear mask: 0x7FFFFFFF per dword (FP32) or
            // 0x7FFFFFFFFFFFFFFF per qword (FP64).
            SimdRegister mask = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Pcmpeqd(mask, mask);
            if (is_double) as_.Psrlq(mask, int8_t{1});
            else           as_.Psrld(mask, int8_t{1});
            as_.Pand(xn, mask);
            as_.Pand(xm, mask);
          }
          // Compare: result mask lands in xn.
          //   FCMEQ: Cmpeq*p* xn, xm                    -> xn = (xn == xm)
          //   FCMGE: Cmple*p* xm, xn ; xn = xm           -> xn = (xm <= xn)
          //   FCMGT: Cmplt*p* xm, xn ; xn = xm           -> xn = (xm <  xn)
          // SSE legacy-encoded Cmp{eq,lt,le}p{s,d} are ordered: result lane
          // is FALSE (zero) when any operand is NaN.  ARM FCMEQ/FCMGE/FCMGT
          // also return FALSE for unordered compares — identical semantics.
          if (is_eq) {
            if (is_double) as_.Cmpeqpd(xn, xm);
            else           as_.Cmpeqps(xn, xm);
          } else if (is_ge) {
            if (is_double) as_.Cmplepd(xm, xn);
            else           as_.Cmpleps(xm, xn);
            as_.Movdqa(xn, xm);
          } else {  // FCMGT / FACGT
            if (is_double) as_.Cmpltpd(xm, xn);
            else           as_.Cmpltps(xm, xn);
            as_.Movdqa(xn, xm);
          }
          if (!args.q) MaskLow64(xn);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        if (!host_platform::kHasF16C) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        SimdRegister mask = no_simd_register;
        if (is_abs) {
          mask = AllocTempSimdReg();
          if (mask == no_simd_register) { Undefined(); return; }
          as_.Pcmpeqd(mask, mask);
          as_.Psrld(mask, int8_t{1});   // 0x7FFFFFFF per dword
        }
        auto cmp = [&](SimdRegister a, SimdRegister b) {
          // Emits a = (Va_original op Vb_original). Currently a holds Vn-half
          // FP32, b holds Vm-half FP32. For GE/GT we need a swapped compare
          // to land the mask in `a`.
          if (is_abs) {
            as_.Pand(a, mask);
            as_.Pand(b, mask);
          }
          if (is_eq) {
            as_.Cmpeqps(a, b);
          } else if (is_ge) {
            as_.Cmpleps(b, a);
            as_.Movdqa(a, b);
          } else {
            as_.Cmpltps(b, a);
            as_.Movdqa(a, b);
          }
        };
        if (!args.q) {
          // .4H: 4 FP16 lanes in low 64 bits of each operand.
          as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Vcvtph2ps(xn, xn);
          as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
          as_.Vcvtph2ps(xm, xm);
          cmp(xn, xm);
          // Packssdw with a duplicate src fills both halves with the packed
          // low_mask; we want the upper 64 bits zero. Movq xn,xn zero-extends.
          as_.Packssdw(xn, xn);
          as_.Movq(xn, xn);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        } else {
          // .8H: process low 4 lanes, then high 4 lanes, then one PACKSSDW
          // to interleave (low_mask -> low 64 bits, hi_mask -> upper 64).
          SimdRegister xn_hi = AllocTempSimdReg();
          SimdRegister xm_hi = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movdqu(xn_hi, {.base = Assembler::rbp, .disp = vn_off});
          as_.Movdqa(xn, xn_hi);
          as_.Vcvtph2ps(xn, xn);
          as_.Psrldq(xn_hi, int8_t{8});
          as_.Vcvtph2ps(xn_hi, xn_hi);
          as_.Movdqu(xm_hi, {.base = Assembler::rbp, .disp = vm_off});
          as_.Movdqa(xm, xm_hi);
          as_.Vcvtph2ps(xm, xm);
          as_.Psrldq(xm_hi, int8_t{8});
          as_.Vcvtph2ps(xm_hi, xm_hi);
          cmp(xn, xm);
          cmp(xn_hi, xm_hi);
          as_.Packssdw(xn, xn_hi);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        }
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kFmaxV:
      case Decoder::AdvSimdThreeSameOpcode::kFminV:
      case Decoder::AdvSimdThreeSameOpcode::kFmaxnmV:
      case Decoder::AdvSimdThreeSameOpcode::kFminnmV: {
        // FP vector FMAX / FMIN / FMAXNM / FMINNM.
        //
        // ARM and x86 disagree on NaN handling for MAX/MIN:
        //   - ARM FMAX/FMIN  (IEEE 754-2008): if either input is NaN, result is NaN.
        //   - ARM FMAXNM/FMINNM (max/min Number): if exactly one input is NaN,
        //     return the other; if both NaN, result is NaN.
        //   - x86 MAXPS/MINPS: if either input is NaN, result = SRC2 (asymmetric).
        //
        // Lowering for FMAX (NaN-propagating):
        //   tmp = b ; MAXPS tmp, a   -> tmp = a if any NaN, else max
        //   MAXPS a, b               -> a   = b if any NaN, else max
        //   POR a, tmp               -> bitwise OR keeps all-1 exponent (NaN) if
        //                               either operand was NaN; equals max otherwise.
        // FMIN is the same shape with MINPS.
        //
        // Lowering for FMAXNM (NaN-suppressing):
        //   substitute NaN-lanes in each operand with the other operand's value,
        //   then MAXPS. After substitution:
        //     - a NaN, b non-NaN -> a' = b, b' = b -> MAXPS = b. ✓
        //     - b NaN, a non-NaN -> a' = a, b' = a -> MAXPS = a. ✓
        //     - both NaN         -> a' = b (NaN), b' = a (NaN) -> MAXPS = SRC2 = a (a NaN). ✓
        //     - neither          -> a' = a, b' = b -> MAXPS = max(a, b). ✓
        // FMINNM analogous with MINPS.
        //
        // FP16 round-trip is bit-exact for the non-NaN path because FP32's
        // mantissa strictly contains FP16's. NaN-result bit patterns may
        // differ from a canonical FP16 qNaN (0x7E00), but are still valid
        // NaNs per ARM ARM default-NaN propagation rules.
        const bool is_max = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxV ||
                             args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV);
        const bool is_nm  = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV ||
                             args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminnmV);
        if (!args.is_fp16) {
          // FP32 (.2S/.4S, args.size=0) and FP64 (.2D, args.size=1).  The
          // decoder filters sz=1 && !Q (the reserved .1D shape) to Undefined,
          // so the JIT only sees .2S, .4S, and .2D here.  Both halves of
          // each lowering match the FP16 path above structurally; differs
          // only by op width (PS vs PD) and skips the F16C round-trip.
          const bool is_double = (args.size & 1);
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);
          // ARM FMAX/FMAXNM are computed as -FMIN(-a,-b). x86 MAXPS/MINPS return
          // the SECOND source on a +-0 tie (OR-of-signs), which is correct only
          // for FMIN (most-negative). Negating inputs+output makes FMAX reuse the
          // FMIN lowering and get ARM's AND-of-signs (most-positive) for the +-0
          // tie; because the negation is applied twice, a propagated-NaN's sign
          // is preserved too. FMIN/FMINNM are already correct, so no negation.
          SimdRegister sign_mask = no_simd_register;
          if (is_max) {
            sign_mask = AllocTempSimdReg();
            if (sign_mask == no_simd_register) { success_ = false; return; }
            as_.Pcmpeqd(sign_mask, sign_mask);
            if (is_double) as_.Psllq(sign_mask, static_cast<int8_t>(63));
            else as_.Pslld(sign_mask, static_cast<int8_t>(31));
            as_.Pxor(xn, sign_mask);
            as_.Pxor(xm, sign_mask);
          }
          auto cmpunord = [&](SimdRegister dst, SimdRegister src) {
            if (is_double) as_.Cmpunordpd(dst, src); else as_.Cmpunordps(dst, src);
          };
          // Always FMIN now (FMAX reaches here with negated operands).
          auto minmax = [&](SimdRegister dst, SimdRegister src) {
            if (is_double) as_.Minpd(dst, src); else as_.Minps(dst, src);
          };
          if (!is_nm) {
            // FMAX / FMIN — NaN-propagating: maxab|maxba|OR.
            SimdRegister tmp = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqa(tmp, xm);
            minmax(tmp, xn);
            minmax(xn, xm);
            as_.Por(xn, tmp);
          } else {
            // FMAXNM / FMINNM — substitute NaN-lanes with the other operand,
            // then min/max.
            SimdRegister t_mask_a = AllocTempSimdReg();
            SimdRegister t_mask_b = AllocTempSimdReg();
            SimdRegister t_an_sub = AllocTempSimdReg();
            SimdRegister t_bn_sub = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqa(t_mask_a, xn);
            cmpunord(t_mask_a, t_mask_a);   // 1s where a is NaN
            as_.Movdqa(t_mask_b, xm);
            cmpunord(t_mask_b, t_mask_b);   // 1s where b is NaN
            as_.Movdqa(t_an_sub, t_mask_a);
            as_.Pand(t_an_sub, xm);          // mask_a & b
            as_.Movdqa(t_bn_sub, t_mask_b);
            as_.Pand(t_bn_sub, xn);          // mask_b & a
            as_.Pandn(t_mask_a, xn);         // ~mask_a & a
            as_.Pandn(t_mask_b, xm);         // ~mask_b & b
            as_.Por(t_mask_a, t_an_sub);     // a' in t_mask_a
            as_.Por(t_mask_b, t_bn_sub);     // b' in t_mask_b
            // minab|minba|OR so the +-0 tie gets OR-of-signs (most-negative,
            // ARM's FMINNM rule), not x86's "return 2nd source on tie". The
            // operands are NaN-free here (NaN lanes were substituted above).
            SimdRegister t_nm = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqa(t_nm, t_mask_b);
            minmax(t_nm, t_mask_a);          // min(b', a')
            minmax(t_mask_a, t_mask_b);      // min(a', b')
            as_.Por(t_mask_a, t_nm);
            as_.Movdqa(xn, t_mask_a);
          }
          // Negate the FMIN result back to obtain FMAX = -FMIN(-a,-b).
          if (is_max) as_.Pxor(xn, sign_mask);
          // .2S (q=0, FP32 only) zeroes upper 64 bits of the destination.
          if (!args.q) MaskLow64(xn);
          StoreVReg(args.rd, xn);
          return;
        }
        if (!host_platform::kHasF16C) { Undefined(); return; }
        // The F16C widen makes this the FP32 lowering on the widened halves,
        // so it reuses the FP32/FP64 path's +-0-tie handling above: ARM
        // FMAX/FMAXNM = -FMIN(-a,-b), so negating inputs+output makes MAX reuse
        // the FMIN lowering and get ARM's AND-of-signs (most-positive) on the
        // +-0 tie instead of x86 MAXPS's OR-of-signs. FMIN/FMINNM need no
        // negation. The per-lane FP32 sign mask is 0x80000000 (bit31).
        SimdRegister sign_mask = no_simd_register;
        if (is_max) {
          sign_mask = AllocTempSimdReg();
          if (sign_mask == no_simd_register) { success_ = false; return; }
          as_.Pcmpeqd(sign_mask, sign_mask);
          as_.Pslld(sign_mask, static_cast<int8_t>(31));
        }
        // Always FMIN now (FMAX reaches here with negated operands).
        auto minmax_op = [&](SimdRegister dst, SimdRegister src) {
          as_.Minps(dst, src);
        };
        if (!args.q) {
          // .4H: 4 FP16 lanes in low 64 bits of each operand.
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          if (!success()) { return; }
          as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Vcvtph2ps(xn, xn);
          as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
          as_.Vcvtph2ps(xm, xm);
          if (is_max) { as_.Pxor(xn, sign_mask); as_.Pxor(xm, sign_mask); }
          if (!is_nm) {
            // FMAX / FMIN — NaN-propagating: minab|minba|OR.
            SimdRegister tmp = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqa(tmp, xm);
            minmax_op(tmp, xn);
            minmax_op(xn, xm);
            as_.Por(xn, tmp);
          } else {
            // FMAXNM / FMINNM — substitute NaN lanes with the other operand,
            // then minab|minba|OR so the +-0 tie gets OR-of-signs (ARM's
            // FMINNM rule) rather than x86's "return 2nd source on tie".
            SimdRegister t_mask_a = AllocTempSimdReg();
            SimdRegister t_mask_b = AllocTempSimdReg();
            SimdRegister t_an_sub = AllocTempSimdReg();
            SimdRegister t_bn_sub = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqa(t_mask_a, xn);
            as_.Cmpunordps(t_mask_a, t_mask_a);   // 1s where a is NaN
            as_.Movdqa(t_mask_b, xm);
            as_.Cmpunordps(t_mask_b, t_mask_b);   // 1s where b is NaN
            as_.Movdqa(t_an_sub, t_mask_a);
            as_.Pand(t_an_sub, xm);                // mask_a & b
            as_.Movdqa(t_bn_sub, t_mask_b);
            as_.Pand(t_bn_sub, xn);                // mask_b & a
            as_.Pandn(t_mask_a, xn);               // ~mask_a & a
            as_.Pandn(t_mask_b, xm);               // ~mask_b & b
            as_.Por(t_mask_a, t_an_sub);           // a' in t_mask_a
            as_.Por(t_mask_b, t_bn_sub);           // b' in t_mask_b
            SimdRegister t_nm = AllocTempSimdReg();
            if (!success()) { return; }
            as_.Movdqa(t_nm, t_mask_b);
            minmax_op(t_nm, t_mask_a);             // min(b', a')
            minmax_op(t_mask_a, t_mask_b);         // min(a', b')
            as_.Por(t_mask_a, t_nm);
            as_.Movdqa(xn, t_mask_a);
          }
          if (is_max) as_.Pxor(xn, sign_mask);
          as_.Vcvtps2ph(xn, xn, int8_t{0});
          // Vcvtps2ph auto-zeroes upper 64 bits.
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        // .8H: process low 4 lanes, then high 4 lanes, then recombine.
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xn_hi = AllocTempSimdReg();
        SimdRegister xm_hi = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn_hi, {.base = Assembler::rbp, .disp = vn_off});
        as_.Movdqa(xn, xn_hi);
        as_.Vcvtph2ps(xn, xn);
        as_.Psrldq(xn_hi, int8_t{8});
        as_.Vcvtph2ps(xn_hi, xn_hi);
        as_.Movdqu(xm_hi, {.base = Assembler::rbp, .disp = vm_off});
        as_.Movdqa(xm, xm_hi);
        as_.Vcvtph2ps(xm, xm);
        as_.Psrldq(xm_hi, int8_t{8});
        as_.Vcvtph2ps(xm_hi, xm_hi);
        if (is_max) {
          as_.Pxor(xn, sign_mask);    as_.Pxor(xm, sign_mask);
          as_.Pxor(xn_hi, sign_mask); as_.Pxor(xm_hi, sign_mask);
        }
        if (!is_nm) {
          // FMAX / FMIN — NaN-propagating via minab|minba|OR; one scratch.
          SimdRegister tmp = AllocTempSimdReg();
          if (!success()) { return; }
          // Low half.
          as_.Movdqa(tmp, xm);
          minmax_op(tmp, xn);
          minmax_op(xn, xm);
          as_.Por(xn, tmp);
          // High half — reuse `tmp`.
          as_.Movdqa(tmp, xm_hi);
          minmax_op(tmp, xn_hi);
          minmax_op(xn_hi, xm_hi);
          as_.Por(xn_hi, tmp);
        } else {
          // FMAXNM / FMINNM — NaN-suppressing; five scratch temps, reused per half.
          SimdRegister t_mask_a = AllocTempSimdReg();
          SimdRegister t_mask_b = AllocTempSimdReg();
          SimdRegister t_an_sub = AllocTempSimdReg();
          SimdRegister t_bn_sub = AllocTempSimdReg();
          SimdRegister t_nm = AllocTempSimdReg();
          if (!success()) { return; }
          // Low half: process xn (a) and xm (b), result back into xn.
          as_.Movdqa(t_mask_a, xn);
          as_.Cmpunordps(t_mask_a, t_mask_a);
          as_.Movdqa(t_mask_b, xm);
          as_.Cmpunordps(t_mask_b, t_mask_b);
          as_.Movdqa(t_an_sub, t_mask_a);
          as_.Pand(t_an_sub, xm);
          as_.Movdqa(t_bn_sub, t_mask_b);
          as_.Pand(t_bn_sub, xn);
          as_.Pandn(t_mask_a, xn);
          as_.Pandn(t_mask_b, xm);
          as_.Por(t_mask_a, t_an_sub);
          as_.Por(t_mask_b, t_bn_sub);
          as_.Movdqa(t_nm, t_mask_b);
          minmax_op(t_nm, t_mask_a);
          minmax_op(t_mask_a, t_mask_b);
          as_.Por(t_mask_a, t_nm);
          as_.Movdqa(xn, t_mask_a);
          // High half: reuse the temps for xn_hi (a) and xm_hi (b).
          as_.Movdqa(t_mask_a, xn_hi);
          as_.Cmpunordps(t_mask_a, t_mask_a);
          as_.Movdqa(t_mask_b, xm_hi);
          as_.Cmpunordps(t_mask_b, t_mask_b);
          as_.Movdqa(t_an_sub, t_mask_a);
          as_.Pand(t_an_sub, xm_hi);
          as_.Movdqa(t_bn_sub, t_mask_b);
          as_.Pand(t_bn_sub, xn_hi);
          as_.Pandn(t_mask_a, xn_hi);
          as_.Pandn(t_mask_b, xm_hi);
          as_.Por(t_mask_a, t_an_sub);
          as_.Por(t_mask_b, t_bn_sub);
          as_.Movdqa(t_nm, t_mask_b);
          minmax_op(t_nm, t_mask_a);
          minmax_op(t_mask_a, t_mask_b);
          as_.Por(t_mask_a, t_nm);
          as_.Movdqa(xn_hi, t_mask_a);
        }
        if (is_max) { as_.Pxor(xn, sign_mask); as_.Pxor(xn_hi, sign_mask); }
        as_.Vcvtps2ph(xn, xn, int8_t{0});
        as_.Vcvtps2ph(xn_hi, xn_hi, int8_t{0});
        as_.Pslldq(xn_hi, int8_t{8});
        as_.Por(xn, xn_hi);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // FMULX vector three-same (FP32 .2S/.4S, FP64 .2D).
      //
      // Direct lift of the AdvSimdScalarThreeSame FMULX scalar pattern to
      // packed PS/PD:
      //   mul         = a * b
      //   mul_unord   = cmpunord(mul, mul)       (-1 per lane iff mul is NaN)
      //   input_unord = cmpunord(a, b)           (-1 per lane iff a or b is NaN)
      //   special     = mul_unord AND NOT input_unord
      //   two_signed  = ((a XOR b) AND sign_mask) OR bits-of(+2.0)
      //   result      = (mul AND NOT special) OR (two_signed AND special)
      //
      // The PCMPEQD + PSLLQ/PSLLD idiom builds the per-lane sign mask
      // (0x80000000... per FP32 lane or 0x8000000000000000... per FP64
      // lane) — no GPR temp needed for that mask.
      //
      // The +2.0 broadcast uses MOVD/MOVQ from a GPR temp into XMM lane 0,
      // then PSHUFD (FP32, broadcast lane 0 to all four 32-bit lanes) or
      // PUNPCKLQDQ (FP64, broadcast low 64 bits to both 64-bit lanes;
      // SSE2-only so available on every host we target).
      //
      // FP16 .4H/.8H via F16C round-trip — see dedicated branch below.
      case Decoder::AdvSimdThreeSameOpcode::kFmulxV: {
        // FP16 vector FMULX .4H / .8H via F16C round-trip.
        // The interpreter computes each FP16 lane as
        // FpSingleToHalf(FmulxScalar<float>(a, b)) — i.e. the whole FMULX
        // (including the ±0×±∞ → ±2.0 special-case override and sign-of-xor
        // adjustment) runs in FP32 then narrows to half.  Lift:
        //   widen FP16 -> FP32 via Vcvtph2ps (4 FP32 lanes in xmm);
        //   run the FP32 FMULX algorithm — mul = a*b; mul_unord =
        //     cmpunord(mul,mul); input_unord = cmpunord(a,b);
        //     special = mul_unord AND NOT input_unord;
        //     two_signed = ((a XOR b) AND sign_mask) OR bits-of(+2.0);
        //     result = (mul AND NOT special) OR (two_signed AND special);
        //   narrow FP32 -> FP16 via Vcvtps2ph.
        // For .8H process low 4 lanes and high 4 lanes in two passes,
        // reusing the same temp set, then recombine via Pslldq + Por
        // (matches the FP16 FRECPS / FRSQRTS vector pattern below).
        if (args.is_fp16) {
          if (!host_platform::kHasF16C) { success_ = false; return; }

          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister xmul = AllocTempSimdReg();
          SimdRegister xmuun = AllocTempSimdReg();
          SimdRegister xiu = AllocTempSimdReg();
          SimdRegister xtwo = AllocTempSimdReg();
          SimdRegister xlo = args.q ? AllocTempSimdReg() : no_simd_register;
          if (xn == no_simd_register || xm == no_simd_register ||
              xmul == no_simd_register || xmuun == no_simd_register ||
              xiu == no_simd_register || xtwo == no_simd_register ||
              (args.q && xlo == no_simd_register)) {
            success_ = false; return;
          }
          Register tmp_gpr = AllocTempReg();
          if (tmp_gpr == Assembler::no_register) {
            success_ = false; return;
          }

          // Run the FP32 FMULX algorithm on xn/xm (which already hold 4 FP32
          // lanes lifted from FP16).  Result ends up in xiu.  Destroys
          // xn/xm/xmul/xmuun/xtwo (all considered scratch within emit_pass).
          auto emit_pass = [&]() {
            // mul = a * b
            as_.Movdqa(xmul, xn);
            as_.Mulps(xmul, xm);
            // mul_unord = cmpunord(mul, mul)
            as_.Movdqa(xmuun, xmul);
            as_.Cmpunordps(xmuun, xmuun);
            // input_unord = cmpunord(a, b)
            as_.Movdqa(xiu, xn);
            as_.Cmpunordps(xiu, xm);
            // special_mask = mul_unord AND NOT input_unord.  Pandn writes
            // dst = (NOT dst) AND src, so the result lands in xiu.
            as_.Pandn(xiu, xmuun);
            // two_signed: ((a XOR b) AND sign_mask) OR bits-of(+2.0).
            // Reuse xn as the XOR scratch; xmuun as the sign mask.
            as_.Xorps(xn, xm);
            as_.Pcmpeqd(xmuun, xmuun);
            as_.Pslld(xmuun, int8_t{31});
            as_.Pand(xn, xmuun);
            // Broadcast +2.0 bits into all four FP32 lanes of xtwo.
            as_.Movl(tmp_gpr, int32_t{0x40000000});
            as_.Movd(xtwo, tmp_gpr);
            as_.Pshufd(xtwo, xtwo, int8_t{0});
            as_.Por(xn, xtwo);
            // xn now holds ±2.0 per lane (sign = sign(a) XOR sign(b)).
            // Blend: result = (mul AND NOT special) OR (±2.0 AND special).
            // Reuse xm as masked-±2.0; xiu as result.
            as_.Movdqa(xm, xn);
            as_.Pand(xm, xiu);
            as_.Pandn(xiu, xmul);
            as_.Por(xiu, xm);
            // xiu holds the FP32 result (4 lanes).
          };

          if (!args.q) {
            // .4H: 4 FP16 lanes in low 64 bits of each operand.
            as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xn, xn);
            as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
            as_.Vcvtph2ps(xm, xm);
            emit_pass();
            as_.Vcvtps2ph(xiu, xiu, int8_t{0});
            // Vcvtps2ph auto-zeroes upper 64 bits.
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xiu);
          } else {
            // .8H: pass 1 (low 4 lanes), narrow & stash in xlo; pass 2 (high
            // 4 lanes via Psrldq 8), narrow, recombine via Pslldq + Por.
            as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xn, xn);
            as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
            as_.Vcvtph2ps(xm, xm);
            emit_pass();
            as_.Vcvtps2ph(xlo, xiu, int8_t{0});
            // xlo holds 4 FP16 lanes in low 64 (lanes 0..3 of result).

            as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
            as_.Psrldq(xn, int8_t{8});
            as_.Vcvtph2ps(xn, xn);
            as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});
            as_.Psrldq(xm, int8_t{8});
            as_.Vcvtph2ps(xm, xm);
            emit_pass();
            as_.Vcvtps2ph(xiu, xiu, int8_t{0});
            // xiu holds 4 FP16 lanes in low 64 (lanes 4..7 of result).

            as_.Pslldq(xiu, int8_t{8});
            as_.Por(xlo, xiu);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          }
          return;
        }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { success_ = false; return; }

        SimdRegister xmm_n = AllocTempSimdReg();
        SimdRegister xmm_m = AllocTempSimdReg();
        SimdRegister xmm_mul = AllocTempSimdReg();
        SimdRegister xmm_mul_unord = AllocTempSimdReg();
        SimdRegister xmm_input_unord = AllocTempSimdReg();
        SimdRegister xmm_two = AllocTempSimdReg();
        if (!success()) { return; }

        LoadVReg(xmm_n, args.rn);
        LoadVReg(xmm_m, args.rm);

        // mul = a * b
        as_.Movdqa(xmm_mul, xmm_n);
        if (is_double) as_.Mulpd(xmm_mul, xmm_m);
        else            as_.Mulps(xmm_mul, xmm_m);

        // mul_unord = cmpunord(mul, mul)
        as_.Movdqa(xmm_mul_unord, xmm_mul);
        if (is_double) as_.Cmpunordpd(xmm_mul_unord, xmm_mul_unord);
        else            as_.Cmpunordps(xmm_mul_unord, xmm_mul_unord);

        // input_unord = cmpunord(a, b)
        as_.Movdqa(xmm_input_unord, xmm_n);
        if (is_double) as_.Cmpunordpd(xmm_input_unord, xmm_m);
        else            as_.Cmpunordps(xmm_input_unord, xmm_m);

        // special_mask = mul_unord AND NOT input_unord (Pandn writes its
        // destination as (NOT dst) AND src, so result lands in xmm_input_unord).
        as_.Pandn(xmm_input_unord, xmm_mul_unord);

        // two_signed: build ((a XOR b) AND sign_mask) OR bits-of(+2.0).
        // Reuse xmm_n as the XOR result; xmm_mul_unord as the sign mask.
        if (is_double) as_.Xorpd(xmm_n, xmm_m);
        else            as_.Xorps(xmm_n, xmm_m);
        as_.Pcmpeqd(xmm_mul_unord, xmm_mul_unord);
        if (is_double) as_.Psllq(xmm_mul_unord, int8_t{63});
        else            as_.Pslld(xmm_mul_unord, int8_t{31});
        as_.Pand(xmm_n, xmm_mul_unord);

        // Broadcast bits of +2.0 into all lanes of xmm_two.
        Register tmp_gpr = AllocTempReg();
        if (is_double) {
          as_.Movq(tmp_gpr, int64_t{0x4000000000000000LL});
          as_.Movq(xmm_two, tmp_gpr);
          // PUNPCKLQDQ duplicates the low 64 bits into both lanes.
          as_.Punpcklqdq(xmm_two, xmm_two);
        } else {
          as_.Movl(tmp_gpr, int32_t{0x40000000});
          as_.Movd(xmm_two, tmp_gpr);
          // PSHUFD imm=0 broadcasts lane 0 to all four 32-bit lanes.
          as_.Pshufd(xmm_two, xmm_two, int8_t{0});
        }
        as_.Por(xmm_n, xmm_two);
        // xmm_n now holds ±2.0 per lane (sign = sign(a) XOR sign(b)).

        // Blend: result = (mul AND NOT special) OR (±2.0 AND special).
        // Reuse xmm_m as the masked-±2.0; reuse xmm_input_unord as the result.
        as_.Movdqa(xmm_m, xmm_n);
        as_.Pand(xmm_m, xmm_input_unord);          // m = ±2.0 AND special
        as_.Pandn(xmm_input_unord, xmm_mul);       // input_unord = NOT(special) AND mul
        as_.Por(xmm_input_unord, xmm_m);           // result in xmm_input_unord.

        if (!args.q) MaskLow64(xmm_input_unord);
        StoreVReg(args.rd, xmm_input_unord);
        return;
      }
      // FMLA / FMLS vector three-same (FP32 .2S/.4S, FP64 .2D).
      //
      // ARM ARM defines FMLA/FMLS as fused multiply-accumulate (single
      // rounding for the whole multiply-add).  x86 FMA3 packed forms have
      // matching semantics:
      //   FMLA  Vd <- Vd + Vn*Vm  ->  VFMADD231PS / VFMADD231PD
      //   FMLS  Vd <- Vd - Vn*Vm  ->  VFNMADD231PS / VFNMADD231PD
      // Same shape as the existing scalar FMADD/FMSUB JIT in FpDataProc3.
      //
      // FP16 .4H/.8H lifts via FP16 -> FP32 -> FP64 round-trip: widen 4
      // FP16 lanes to 4 FP32 lanes (Vcvtph2ps), then promote each pair
      // of FP32 lanes to FP64 (Vcvtps2pd) and run VFMADD231PD /
      // VFNMADD231PD on 2 FP64 lanes at a time, narrow back to FP32
      // (Vcvtpd2ps), recombine the two FP32 halves, and narrow once
      // more to FP16 (Vcvtps2ph).  Matches the interpreter's
      //   r64 = std::fma((double)a, (double)b, (double)d);
      //   rh  = FpSingleToHalf((float)r64);
      // because the multiply-add is in binary64 and there is a single
      // narrow back through FP32 to half — no intermediate FP32 sum to
      // double-round.  .4H needs two FP64 passes; .8H needs four.
      //
      // Reserved .1D shape (size=01 && q=0) bails too.  Hosts without
      // FMA3 fall back to the interpreter (no MUL+ADD pair — that would
      // double-round, violating ARM's fused semantics).  Hosts without
      // F16C bail on the FP16 path.
      case Decoder::AdvSimdThreeSameOpcode::kFmlaV:
      case Decoder::AdvSimdThreeSameOpcode::kFmlsV: {
        if (args.is_fp16) {
          if (!host_platform::kHasFMA) { success_ = false; return; }
          if (!host_platform::kHasF16C) { success_ = false; return; }
          const bool is_fmls =
              (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmlsV);

          // Pre-allocate all temps once and reuse across both FP64
          // passes (lanes 0,1 and lanes 2,3 of each FP32 quad) so we
          // don't burn through the 16-reg XMM pool.  For .8H we also
          // need xlo to stash the low-4-lane FP16 result while the
          // high-4-lane quad runs.
          SimdRegister xn_f32 = AllocTempSimdReg();
          SimdRegister xm_f32 = AllocTempSimdReg();
          SimdRegister xd_f32 = AllocTempSimdReg();
          SimdRegister xn_pd = AllocTempSimdReg();
          SimdRegister xm_pd = AllocTempSimdReg();
          SimdRegister xd_pd = AllocTempSimdReg();
          SimdRegister xres = AllocTempSimdReg();
          SimdRegister xlo = args.q ? AllocTempSimdReg() : no_simd_register;
          if (xn_f32 == no_simd_register || xm_f32 == no_simd_register ||
              xd_f32 == no_simd_register || xn_pd == no_simd_register ||
              xm_pd == no_simd_register || xd_pd == no_simd_register ||
              xres == no_simd_register ||
              (args.q && xlo == no_simd_register)) {
            success_ = false; return;
          }

          // With 4 FP32 lanes already widened into xn_f32/xm_f32/xd_f32,
          // do two FP64 passes (low 2 lanes via Vcvtps2pd of the low
          // 64 bits, then high 2 lanes after Psrldq 8 brings them
          // down).  Recombines into xres as 4 FP32 lanes, ready for
          // the final Vcvtps2ph narrow.  Destroys xn_f32/xm_f32/xd_f32
          // (they're scratch within the lambda's scope of use).
          auto emit_quad = [&]() {
            // Pass 1: low 2 FP32 lanes -> 2 FP64 lanes.
            as_.Vcvtps2pd(xn_pd, xn_f32);
            as_.Vcvtps2pd(xm_pd, xm_f32);
            as_.Vcvtps2pd(xd_pd, xd_f32);
            if (is_fmls) {
              as_.Vfnmadd231pd(xd_pd, xn_pd, xm_pd);
            } else {
              as_.Vfmadd231pd(xd_pd, xn_pd, xm_pd);
            }
            as_.Vcvtpd2ps(xres, xd_pd);  // 2 FP32 lanes in low 64 of xres.

            // Pass 2: shift high 2 FP32 lanes down, promote to FP64,
            // FMA, narrow back to FP32 (low 64 of xn_f32 used as a
            // scratch since the originals are no longer needed).
            as_.Psrldq(xn_f32, int8_t{8});
            as_.Vcvtps2pd(xn_pd, xn_f32);
            as_.Psrldq(xm_f32, int8_t{8});
            as_.Vcvtps2pd(xm_pd, xm_f32);
            as_.Psrldq(xd_f32, int8_t{8});
            as_.Vcvtps2pd(xd_pd, xd_f32);
            if (is_fmls) {
              as_.Vfnmadd231pd(xd_pd, xn_pd, xm_pd);
            } else {
              as_.Vfmadd231pd(xd_pd, xn_pd, xm_pd);
            }
            as_.Vcvtpd2ps(xn_f32, xd_pd);  // 2 FP32 lanes in low 64.

            // Recombine: lanes 0,1 in xres low 64, lanes 2,3 in
            // xn_f32 low 64 -> xres lanes 0..3.
            as_.Pslldq(xn_f32, int8_t{8});
            as_.Por(xres, xn_f32);
          };

          if (!args.q) {
            // .4H: 4 FP16 lanes (low 64 bits) -> 4 FP32 -> 2 FP64
            // passes -> 4 FP32 -> 4 FP16.
            as_.Movq(xn_f32, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xn_f32, xn_f32);
            as_.Movq(xm_f32, {.base = Assembler::rbp, .disp = vm_off});
            as_.Vcvtph2ps(xm_f32, xm_f32);
            as_.Movq(xd_f32, {.base = Assembler::rbp, .disp = vd_off});
            as_.Vcvtph2ps(xd_f32, xd_f32);
            emit_quad();
            as_.Vcvtps2ph(xres, xres, int8_t{0});
            // Vcvtps2ph auto-zeroes upper 64 bits.
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xres);
          } else {
            // .8H: 8 FP16 lanes split into two quads.  Pass low 4 ->
            // narrow -> stash in xlo; then high 4 (Psrldq 8 on the
            // 128-bit source loads) -> narrow -> recombine with xlo
            // via Pslldq + Por.
            as_.Movq(xn_f32, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xn_f32, xn_f32);
            as_.Movq(xm_f32, {.base = Assembler::rbp, .disp = vm_off});
            as_.Vcvtph2ps(xm_f32, xm_f32);
            as_.Movq(xd_f32, {.base = Assembler::rbp, .disp = vd_off});
            as_.Vcvtph2ps(xd_f32, xd_f32);
            emit_quad();
            as_.Vcvtps2ph(xlo, xres, int8_t{0});
            // xlo: 4 FP16 lanes in low 64 (lanes 0..3 of result).

            as_.Movdqu(xn_f32, {.base = Assembler::rbp, .disp = vn_off});
            as_.Psrldq(xn_f32, int8_t{8});
            as_.Vcvtph2ps(xn_f32, xn_f32);
            as_.Movdqu(xm_f32, {.base = Assembler::rbp, .disp = vm_off});
            as_.Psrldq(xm_f32, int8_t{8});
            as_.Vcvtph2ps(xm_f32, xm_f32);
            as_.Movdqu(xd_f32, {.base = Assembler::rbp, .disp = vd_off});
            as_.Psrldq(xd_f32, int8_t{8});
            as_.Vcvtph2ps(xd_f32, xd_f32);
            emit_quad();
            as_.Vcvtps2ph(xres, xres, int8_t{0});
            // xres: 4 FP16 lanes in low 64 (lanes 4..7 of result).

            as_.Pslldq(xres, int8_t{8});
            as_.Por(xlo, xres);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          }
          return;
        }
        if (!host_platform::kHasFMA) { success_ = false; return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { success_ = false; return; }

        SimdRegister xmm_n = AllocTempSimdReg();
        SimdRegister xmm_m = AllocTempSimdReg();
        SimdRegister xmm_d = AllocTempSimdReg();
        if (!success()) { return; }

        LoadVReg(xmm_n, args.rn);
        LoadVReg(xmm_m, args.rm);
        LoadVReg(xmm_d, args.rd);

        const bool is_fmls =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmlsV);
        if (is_fmls) {
          // Vd = Vd + (-Vn)*Vm  -- single fused rounding.
          if (is_double) as_.Vfnmadd231pd(xmm_d, xmm_n, xmm_m);
          else            as_.Vfnmadd231ps(xmm_d, xmm_n, xmm_m);
        } else {
          if (is_double) as_.Vfmadd231pd(xmm_d, xmm_n, xmm_m);
          else            as_.Vfmadd231ps(xmm_d, xmm_n, xmm_m);
        }

        if (!args.q) MaskLow64(xmm_d);
        StoreVReg(args.rd, xmm_d);
        return;
      }
      // FRECPS / FRSQRTS vector three-same (FP32 .2S/.4S, FP64 .2D).
      //
      // Lane-parallel lift of the scalar FRECPS/FRSQRTS JIT path
      // (AdvSimdScalarThreeSame).  Same single-rounded structure:
      //
      //   FRECPS  result = 2 - a*b                    (special-case overrides)
      //   FRSQRTS result = (3 - a*b) / 2              (divide-by-2 is exact)
      //
      // Implemented as VFNMADD231PD/PS of (K_fma - a*b) into a destination
      // pre-loaded with broadcast K_fma.  The special-case override fires
      // when a*b is NaN but neither input is NaN — i.e. the (±0, ±inf)
      // cross — and replaces the lane with the saturation constant K_sat
      // (positive +2.0 for FRECPS, +1.5 for FRSQRTS; sign is *not* fixed
      // up from sign(a) XOR sign(b), unlike FMULX).  NaN inputs override
      // the lane with the default qNaN.
      //
      // FP16 .4H/.8H lowers via F16C round-trip into the FP32 algorithm
      // below; see the dedicated FP16 branch.  Hosts without FMA3 bail
      // (no MUL+SUB fallback — that would double-round, violating ARM's
      // fused semantics).  Reserved .1D shape (size=01 && q=0) bails.
      case Decoder::AdvSimdThreeSameOpcode::kFrecpsV:
      case Decoder::AdvSimdThreeSameOpcode::kFrsqrtsV: {
        // FP16 vector FRECPS / FRSQRTS .4H / .8H via
        // F16C round-trip.  The interpreter computes FP16 lanes as
        // FpSingleToHalf(FrecpsScalar<float>(a, b)) — i.e. the whole
        // Newton step is done in FP32 then narrowed to half.  Lift:
        //   widen FP16 -> FP32 via Vcvtph2ps;
        //   run the FP32 FRECPS/FRSQRTS algorithm (same constants as the
        //   FP32 path below);
        //   narrow FP32 -> FP16 via Vcvtps2ph.
        // For .8H process low 4 lanes and high 4 lanes in two passes,
        // reusing the same temp set, then recombine via PSLLDQ + POR
        // (matches the FP16 FADD/FSUB/FMUL/FDIV vector pattern earlier
        // in this function).
        if (args.is_fp16) {
          if (!host_platform::kHasFMA) { success_ = false; return; }
          if (!host_platform::kHasF16C) { success_ = false; return; }
          const bool is_frecps_fp16 =
              (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFrecpsV);

          const int32_t k_fma_bits = is_frecps_fp16 ? int32_t{0x40000000}   // +2.0
                                                    : int32_t{0x40400000};  // +3.0
          const int32_t k_sat_bits = is_frecps_fp16 ? int32_t{0x40000000}   // +2.0
                                                    : int32_t{0x3FC00000};  // +1.5
          const int32_t qnan_bits  = int32_t{0x7FC00000};
          const int32_t two_bits   = int32_t{0x40000000};

          // Allocate temps once — reuse across both passes for .8H so we
          // don't burn through the 16-reg XMM pool with redundant temps.
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister xmul = AllocTempSimdReg();
          SimdRegister xiu = AllocTempSimdReg();
          SimdRegister xsp = AllocTempSimdReg();
          SimdRegister xlo = args.q ? AllocTempSimdReg() : no_simd_register;
          if (xn == no_simd_register || xm == no_simd_register ||
              xmul == no_simd_register || xiu == no_simd_register ||
              xsp == no_simd_register ||
              (args.q && xlo == no_simd_register)) {
            success_ = false; return;
          }
          Register tmp_gpr = AllocTempReg();
          if (tmp_gpr == Assembler::no_register) {
            success_ = false; return;
          }

          // Emit the FP32 FRECPS/FRSQRTS Newton-step lowering on xn/xm
          // (which already hold 4 FP32 lanes).  Result ends up in xmul.
          auto emit_pass = [&]() {
            // mul = a * b (only its NaN bit is observed via cmpunord).
            as_.Movdqa(xmul, xn);
            as_.Mulps(xmul, xm);
            // input_unord = cmpunord(a, b)
            as_.Movdqa(xiu, xn);
            as_.Cmpunordps(xiu, xm);
            // mul_unord reused in xmul (product value no longer needed).
            as_.Cmpunordps(xmul, xmul);
            // special_mask = (NOT input_unord) AND mul_unord.
            as_.Movdqa(xsp, xiu);
            as_.Pandn(xsp, xmul);

            // fma_result = K_fma - a*b via VFNMADD231PS into broadcast K_fma.
            as_.Movl(tmp_gpr, k_fma_bits);
            as_.Movd(xmul, tmp_gpr);
            as_.Pshufd(xmul, xmul, int8_t{0});
            as_.Vfnmadd231ps(xmul, xn, xm);

            if (!is_frecps_fp16) {
              // FRSQRTS: divide by 2 (exact one-exponent decrement).  Reuse
              // xn as a broadcast-2.0 scratch; a is no longer needed.
              as_.Movl(tmp_gpr, two_bits);
              as_.Movd(xn, tmp_gpr);
              as_.Pshufd(xn, xn, int8_t{0});
              as_.Divps(xmul, xn);
            }
            // xmul holds fma_result.

            // First select: result_first = special_mask ? K_sat : fma_result.
            as_.Movl(tmp_gpr, k_sat_bits);
            as_.Movd(xn, tmp_gpr);
            as_.Pshufd(xn, xn, int8_t{0});
            as_.Pand(xn, xsp);
            as_.Pandn(xsp, xmul);
            as_.Por(xn, xsp);  // xn = result_first

            // Second select: result_final = input_unord ? qnan : result_first.
            as_.Movl(tmp_gpr, qnan_bits);
            as_.Movd(xmul, tmp_gpr);
            as_.Pshufd(xmul, xmul, int8_t{0});
            as_.Pand(xmul, xiu);
            as_.Pandn(xiu, xn);
            as_.Por(xmul, xiu);  // xmul = result_final (FP32 lanes)
          };

          if (!args.q) {
            // .4H: 4 FP16 lanes in low 64 bits.
            as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xn, xn);
            as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
            as_.Vcvtph2ps(xm, xm);
            emit_pass();
            as_.Vcvtps2ph(xmul, xmul, int8_t{0});
            // Vcvtps2ph auto-zeroes upper 64 bits.
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmul);
          } else {
            // .8H: pass 1 (low 4 lanes), narrow & save in xlo; pass 2 (high
            // 4 lanes via Psrldq 8), narrow, recombine via Pslldq + Por.
            as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xn, xn);
            as_.Movq(xm, {.base = Assembler::rbp, .disp = vm_off});
            as_.Vcvtph2ps(xm, xm);
            emit_pass();
            as_.Vcvtps2ph(xlo, xmul, int8_t{0});
            // xlo has 4 FP16 lanes in low 64 bits.

            as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
            as_.Psrldq(xn, int8_t{8});
            as_.Vcvtph2ps(xn, xn);
            as_.Movdqu(xm, {.base = Assembler::rbp, .disp = vm_off});
            as_.Psrldq(xm, int8_t{8});
            as_.Vcvtph2ps(xm, xm);
            emit_pass();
            as_.Vcvtps2ph(xmul, xmul, int8_t{0});
            // xmul has 4 FP16 lanes in low 64 bits.

            as_.Pslldq(xmul, int8_t{8});
            as_.Por(xlo, xmul);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          }
          return;
        }
        if (!host_platform::kHasFMA) { success_ = false; return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { success_ = false; return; }
        const bool is_frecps =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFrecpsV);

        // Constants (lane-broadcasted below).
        const int64_t k_fma_bits_d = is_frecps ? int64_t{0x4000000000000000LL}   // +2.0
                                                : int64_t{0x4008000000000000LL};  // +3.0
        const int32_t k_fma_bits_s = is_frecps ? int32_t{0x40000000}              // +2.0
                                                : int32_t{0x40400000};             // +3.0
        const int64_t k_sat_bits_d = is_frecps ? int64_t{0x4000000000000000LL}   // +2.0
                                                : int64_t{0x3FF8000000000000LL};  // +1.5
        const int32_t k_sat_bits_s = is_frecps ? int32_t{0x40000000}              // +2.0
                                                : int32_t{0x3FC00000};             // +1.5
        const int64_t qnan_bits_d  = int64_t{0x7FF8000000000000LL};
        const int32_t qnan_bits_s  = int32_t{0x7FC00000};
        const int64_t two_bits_d   = int64_t{0x4000000000000000LL};
        const int32_t two_bits_s   = int32_t{0x40000000};

        SimdRegister xmm_n = AllocTempSimdReg();
        SimdRegister xmm_m = AllocTempSimdReg();
        SimdRegister xmm_mul = AllocTempSimdReg();
        SimdRegister xmm_iu = AllocTempSimdReg();
        SimdRegister xmm_special = AllocTempSimdReg();
        if (!success()) { return; }

        LoadVReg(xmm_n, args.rn);
        LoadVReg(xmm_m, args.rm);

        // mul = a * b (only its NaN bit is observed via cmpunord below).
        as_.Movdqa(xmm_mul, xmm_n);
        if (is_double) as_.Mulpd(xmm_mul, xmm_m);
        else            as_.Mulps(xmm_mul, xmm_m);

        // input_unord = cmpunord(a, b): per-lane all-ones iff a or b NaN.
        as_.Movdqa(xmm_iu, xmm_n);
        if (is_double) as_.Cmpunordpd(xmm_iu, xmm_m);
        else            as_.Cmpunordps(xmm_iu, xmm_m);

        // mul_unord = cmpunord(mul, mul) reused in xmm_mul (product value
        // no longer needed past this point — only its NaN bit matters).
        if (is_double) as_.Cmpunordpd(xmm_mul, xmm_mul);
        else            as_.Cmpunordps(xmm_mul, xmm_mul);

        // special_mask = (NOT input_unord) AND mul_unord.  Pandn writes
        // (NOT dst) AND src.  Keep input_unord alive in xmm_iu.
        as_.Movdqa(xmm_special, xmm_iu);
        as_.Pandn(xmm_special, xmm_mul);
        // xmm_mul now free as scratch.

        // Normal-path result = K_fma - a*b via VFNMADD231(dst, n, m):
        // load broadcast K_fma into xmm_mul, then dst -= n*m in-place.
        Register tmp_gpr = AllocTempReg();
        if (is_double) {
          as_.Movq(tmp_gpr, k_fma_bits_d);
          as_.Movq(xmm_mul, tmp_gpr);
          as_.Punpcklqdq(xmm_mul, xmm_mul);
          as_.Vfnmadd231pd(xmm_mul, xmm_n, xmm_m);
        } else {
          as_.Movl(tmp_gpr, k_fma_bits_s);
          as_.Movd(xmm_mul, tmp_gpr);
          as_.Pshufd(xmm_mul, xmm_mul, int8_t{0});
          as_.Vfnmadd231ps(xmm_mul, xmm_n, xmm_m);
        }

        // FRSQRTS: divide by 2 (one-exponent decrement; exact under IEEE
        // binary FP — the rounding-once invariant carries through).
        // Reuse xmm_n as a broadcast-2.0 scratch; a is no longer needed.
        if (!is_frecps) {
          if (is_double) {
            as_.Movq(tmp_gpr, two_bits_d);
            as_.Movq(xmm_n, tmp_gpr);
            as_.Punpcklqdq(xmm_n, xmm_n);
            as_.Divpd(xmm_mul, xmm_n);
          } else {
            as_.Movl(tmp_gpr, two_bits_s);
            as_.Movd(xmm_n, tmp_gpr);
            as_.Pshufd(xmm_n, xmm_n, int8_t{0});
            as_.Divps(xmm_mul, xmm_n);
          }
        }
        // xmm_mul now holds fma_result.  xmm_n, xmm_m are free as scratch.

        // First select: result_first = special_mask ? K_sat : fma_result.
        // Broadcast K_sat into xmm_n.
        if (is_double) {
          as_.Movq(tmp_gpr, k_sat_bits_d);
          as_.Movq(xmm_n, tmp_gpr);
          as_.Punpcklqdq(xmm_n, xmm_n);
        } else {
          as_.Movl(tmp_gpr, k_sat_bits_s);
          as_.Movd(xmm_n, tmp_gpr);
          as_.Pshufd(xmm_n, xmm_n, int8_t{0});
        }
        as_.Pand(xmm_n, xmm_special);       // xmm_n = K_sat AND special
        as_.Pandn(xmm_special, xmm_mul);    // xmm_special = (NOT special) AND fma
        as_.Por(xmm_n, xmm_special);        // xmm_n = result_first

        // Second select: result_final = input_unord ? qnan : result_first.
        // Broadcast qNaN into xmm_m (b is no longer needed).
        if (is_double) {
          as_.Movq(tmp_gpr, qnan_bits_d);
          as_.Movq(xmm_m, tmp_gpr);
          as_.Punpcklqdq(xmm_m, xmm_m);
        } else {
          as_.Movl(tmp_gpr, qnan_bits_s);
          as_.Movd(xmm_m, tmp_gpr);
          as_.Pshufd(xmm_m, xmm_m, int8_t{0});
        }
        as_.Pand(xmm_m, xmm_iu);            // xmm_m = qnan AND iu
        as_.Pandn(xmm_iu, xmm_n);           // xmm_iu = (NOT iu) AND result_first
        as_.Por(xmm_m, xmm_iu);             // xmm_m = result_final

        if (!args.q) MaskLow64(xmm_m);
        StoreVReg(args.rd, xmm_m);
        return;
      }
      // SQDMULH / SQRDMULH three-same vector: saturating
      // doubling multiply-high, with optional rounding.  Decoder restricts
      // size to {01, 10}.
      //
      // SQDMULH per lane:  result = sat_int<esize>((SInt(a) * SInt(b) * 2) >> esize_bits).
      // SQRDMULH per lane: same, with (1 << (esize_bits - 1)) added before the shift.
      //
      // The only lane that requires the saturation step is (a, b) ==
      // (INT_MIN, INT_MIN), where 2 * INT_MIN * INT_MIN overflows int(2*esize).
      // ARM expects INT_MAX for that lane.  All other lanes are exact via
      // straight-line widen / multiply / double / extract-high.
      //
      // size=01 (.4H/.8H):
      //   SQDMULH:  PMULHW (signed high 16) + PMULLW (low 16); combine by
      //     (PMULHW << 1) | (PMULLW >> 15) — that is, the high 16 bits of the
      //     doubled product, ignoring the corner overflow.  3 PMUL + 2 shift
      //     + 1 OR + corner fixup.
      //   SQRDMULH: PMULHRSW (SSSE3) computes ((a * b * 2 + 0x8000) >> 16),
      //     exactly matching SQRDMULH except for the (INT16_MIN)^2 lane.
      //     1 PMUL + corner fixup.
      //   Corner fixup at .8H (same as kSqrdmlahVec):
      //     x_min = 0x8000 broadcast across 8 halfwords.
      //     corner = (Vn == x_min) PCMPEQW AND (Vm == x_min) PCMPEQW.
      //     result ^= corner   (0x8000 ^ 0xFFFF = 0x7FFF = INT16_MAX).
      //
      // size=10 (.2S/.4S): reuse the kSqrdmlahVec stage-1 recipe.  Two PMULDQs
      //   (SSE4.1) reconstruct the 4 signed 32×32 → 64 products.  PSLLQ 1
      //   doubles each.  For SQRDMULH, PADDQ the rounding constant
      //   0x80000000 per qword.  PSHUFD 0xDD picks the high dword of each
      //   qword into the low dword; PUNPCKLDQ interleaves the two halves.
      //   Corner fixup at .4S: x_const = INT32_MIN broadcast; corner =
      //   (Vn == x_const) AND (Vm == x_const); result ^= corner
      //   (INT32_MIN ^ 0xFFFFFFFF = INT32_MAX).
      //
      // PSLLQ-overflow correctness (size=10): for any (a, b) with a * b
      // representable as an int64 (i.e., not the (INT_MIN)^2 corner), the
      // doubled product 2 * a * b fits in int64 with at most a wrap into the
      // upper bit; PSHUFD then picks bits 32..63 which, interpreted as a
      // signed int32, equals floor((2 * a * b) / 2^32) = ARM's
      // ((2 * a * b) >> esize_bits) at size=10.  For the corner, the wrap
      // produces 0x80000000, which the corner XOR flips to 0x7FFFFFFF.
      case Decoder::AdvSimdThreeSameOpcode::kSqdmulh:
      case Decoder::AdvSimdThreeSameOpcode::kSqrdmulh: {
        const bool is_round =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrdmulh);
        if (args.size == 0b01) {
          if (is_round) {
            // SQRDMULH .4H/.8H via PMULHRSW + corner fixup (SSSE3).
            if (!host_platform::kHasSSSE3) { success_ = false; return; }
            SimdRegister xn = AllocTempSimdReg();
            SimdRegister xm = AllocTempSimdReg();
            SimdRegister xn_corner = AllocTempSimdReg();
            SimdRegister xm_corner = AllocTempSimdReg();
            SimdRegister x_min = AllocTempSimdReg();
            if (!success()) { return; }
            LoadVReg(xn, args.rn);
            LoadVReg(xm, args.rm);
            as_.Pcmpeqw(x_min, x_min);
            as_.Psllw(x_min, int8_t{15});
            as_.Movdqa(xn_corner, xn);
            as_.Movdqa(xm_corner, xm);
            as_.Pmulhrsw(xn, xm);
            as_.Pcmpeqw(xn_corner, x_min);
            as_.Pcmpeqw(xm_corner, x_min);
            as_.Pand(xn_corner, xm_corner);
            as_.Pxor(xn, xn_corner);
            if (!args.q) MaskLow64(xn);
            StoreVReg(args.rd, xn);
            return;
          }
          // SQDMULH .4H/.8H via PMULHW + PMULLW combine + corner fixup (SSE2).
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister xn_lo = AllocTempSimdReg();
          SimdRegister xn_corner = AllocTempSimdReg();
          SimdRegister xm_corner = AllocTempSimdReg();
          SimdRegister x_min = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);
          as_.Movdqa(xn_corner, xn);
          as_.Movdqa(xm_corner, xm);
          as_.Movdqa(xn_lo, xn);
          as_.Pmullw(xn_lo, xm);              // low 16 of (a * b)
          as_.Pmulhw(xn, xm);                 // high 16 of signed (a * b)
          as_.Psllw(xn, int8_t{1});           // (high << 1)
          as_.Psrlw(xn_lo, int8_t{15});       // top bit of low half
          as_.Por(xn, xn_lo);                 // high 16 of (2 * a * b), wraps on corner
          as_.Pcmpeqw(x_min, x_min);
          as_.Psllw(x_min, int8_t{15});       // INT16_MIN broadcast
          as_.Pcmpeqw(xn_corner, x_min);
          as_.Pcmpeqw(xm_corner, x_min);
          as_.Pand(xn_corner, xm_corner);
          as_.Pxor(xn, xn_corner);            // INT16_MIN ^ 0xFFFF = INT16_MAX
          if (!args.q) MaskLow64(xn);
          StoreVReg(args.rd, xn);
          return;
        }
        if (args.size == 0b10) {
          // size=10 .2S/.4S: PMULDQ widen + PSLLQ + (optional round) + corner.
          if (!host_platform::kHasSSE4_1) { success_ = false; return; }

          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister x_const = AllocTempSimdReg();
          SimdRegister corner = AllocTempSimdReg();
          SimdRegister xp_lo = AllocTempSimdReg();
          SimdRegister xp_hi = AllocTempSimdReg();
          SimdRegister xm_hi = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);

          // x_const = INT32_MIN broadcast across 4 dwords.
          as_.Pcmpeqd(x_const, x_const);
          as_.Pslld(x_const, int8_t{31});

          // Corner detection: lanes where Vn.s[i] == INT32_MIN AND
          // Vm.s[i] == INT32_MIN.
          as_.Movdqa(corner, xn);
          as_.Pcmpeqd(corner, x_const);
          as_.Movdqa(xp_lo, xm);
          as_.Pcmpeqd(xp_lo, x_const);
          as_.Pand(corner, xp_lo);
          // xp_lo is dead and about to be repurposed.

          // Two PMULDQs reconstruct the 4 signed 32×32 → 64 products.
          //   xp_lo qword 0 = sext_i64(Vn.s[0]) * sext_i64(Vm.s[0])
          //   xp_lo qword 1 = sext_i64(Vn.s[2]) * sext_i64(Vm.s[2])
          //   xp_hi qword 0 = sext_i64(Vn.s[1]) * sext_i64(Vm.s[1])
          //   xp_hi qword 1 = sext_i64(Vn.s[3]) * sext_i64(Vm.s[3])
          as_.Movdqa(xp_lo, xn);
          as_.Pmuldq(xp_lo, xm);
          as_.Movdqa(xp_hi, xn);
          as_.Psrlq(xp_hi, int8_t{32});
          as_.Movdqa(xm_hi, xm);
          as_.Psrlq(xm_hi, int8_t{32});
          as_.Pmuldq(xp_hi, xm_hi);
          // xn, xm, xm_hi dead from here.

          // Double each 64-bit signed product via PSLLQ 1.
          as_.Psllq(xp_lo, int8_t{1});
          as_.Psllq(xp_hi, int8_t{1});

          // SQRDMULH: add rounding constant 2^31 = 0x80000000 per qword.
          if (is_round) {
            as_.Pcmpeqd(x_const, x_const);
            as_.Psllq(x_const, int8_t{63});
            as_.Psrlq(x_const, int8_t{32});
            as_.Paddq(xp_lo, x_const);
            as_.Paddq(xp_hi, x_const);
          }

          // Extract the upper 32 bits of each 64-bit lane.
          as_.Pshufd(xp_lo, xp_lo, static_cast<int8_t>(0xDD));
          as_.Pshufd(xp_hi, xp_hi, static_cast<int8_t>(0xDD));
          as_.Punpckldq(xp_lo, xp_hi);

          // Apply corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX.
          as_.Pxor(xp_lo, corner);

          if (!args.q) MaskLow64(xp_lo);
          StoreVReg(args.rd, xp_lo);
          return;
        }
        // size=00 and size=11 reserved by the decoder; bail safely.
        success_ = false; return;
      }
      // PMUL polynomial multiply (vector, byte lanes).
      //
      // ARM ARM C7.2.219: PMUL .8B/.16B performs per-lane carry-less multiply
      // over GF(2)[x]; the result is the low 8 bits of the polynomial product.
      // The decoder restricts size to 00 (8-bit lanes); size in {01, 10, 11}
      // is reserved and routed to Undefined upstream.
      //
      // Lowering: per-bit unrolled SSE2 recipe (no PCLMULQDQ dependency).
      //   acc = 0
      //   bit_mask = 0x01 (byte-replicated)
      //   for i in 0..7:
      //     shift_mask = -bit_mask per byte                  // = (0xFF << i) & 0xFF
      //     shifted_a  = (a << i) AND shift_mask             // byte-clean
      //     selector   = ((b AND bit_mask) PCMPEQB bit_mask) // 0xFF if bit set
      //     acc       ^= shifted_a AND selector
      //     bit_mask  += bit_mask                             // doubles via PADDB
      //
      // Notes:
      //   * PSLLW shifts whole 16-bit lanes, so the low byte's high bits
      //     spill into the high byte after the shift; the PAND with
      //     shift_mask zeros those spilled bits in every byte.  shift_mask
      //     = -bit_mask (PSUBB(zero, bit_mask)) because for bit_mask = 1<<i,
      //     `0 - (1<<i) mod 256 = ((0xFF << i) & 0xFF)` is exactly the
      //     byte mask with bits 0..i-1 cleared and bits i..7 set.
      //   * PCMPEQB(masked_b, bit_mask) produces 0xFF per byte if bit i was
      //     set in b (since the masked value equals bit_mask only when the
      //     lone surviving bit was set), and 0 otherwise.
      //   * The bit_mask doubles cleanly via PADDB across iterations
      //     0x01 -> 0x02 -> 0x04 -> ... -> 0x80 (no overflow within 8 iters).
      case Decoder::AdvSimdThreeSameOpcode::kPmul: {
        // Decoder enforces size=00; defensive bail for safety.
        if (args.size != 0b00) { Undefined(); return; }

        SimdRegister xa = AllocTempSimdReg();
        SimdRegister xb = AllocTempSimdReg();
        SimdRegister xacc = AllocTempSimdReg();
        SimdRegister xzero = AllocTempSimdReg();
        SimdRegister xbit = AllocTempSimdReg();
        SimdRegister xshift_a = AllocTempSimdReg();
        SimdRegister xsel = AllocTempSimdReg();
        if (!success()) { return; }
        Register tmp_gpr = AllocTempReg();
        if (tmp_gpr == Assembler::no_register) {
          success_ = false; return;
        }

        LoadVReg(xa, args.rn);
        LoadVReg(xb, args.rm);

        // xacc = 0, xzero = 0.
        as_.Pxor(xacc, xacc);
        as_.Pxor(xzero, xzero);

        // xbit = byte 0x01 replicated across all 16 bytes.
        as_.Movq(tmp_gpr, int64_t{0x0101010101010101LL});
        as_.Movq(xbit, tmp_gpr);
        as_.Punpcklqdq(xbit, xbit);

        // 8 iterations, one per bit position 0..7.
        for (int i = 0; i < 8; ++i) {
          // shifted_a = (xa << i) per byte, byte-masked.
          as_.Movdqa(xshift_a, xa);
          if (i != 0) {
            as_.Psllw(xshift_a, static_cast<int8_t>(i));
            // shift_mask = -xbit per byte = byte (0xFF << i) & 0xFF.
            as_.Movdqa(xsel, xzero);
            as_.Psubb(xsel, xbit);
            as_.Pand(xshift_a, xsel);
          }
          // selector = (xb AND xbit) PCMPEQB xbit -> 0xFF per byte where bit i set.
          as_.Movdqa(xsel, xb);
          as_.Pand(xsel, xbit);
          as_.Pcmpeqb(xsel, xbit);
          // Accumulate: xacc ^= shifted_a AND selector.
          as_.Pand(xshift_a, xsel);
          as_.Pxor(xacc, xshift_a);
          // Double xbit for the next iteration (skip after the last bit).
          if (i != 7) {
            as_.Paddb(xbit, xbit);
          }
        }

        if (!args.q) MaskLow64(xacc);
        StoreVReg(args.rd, xacc);
        return;
      }
      // Armv8.1-RDM SQRDMLAH / SQRDMLSH three-same vector.
      //
      // Decoder restricts size to {01, 10}.  size=01 (.4h/.8h) uses PMULHRSW
      // (SSSE3) for stage-1 SQRDMULH with corner fixup at (INT16_MIN,
      // INT16_MIN) lanes, then PADDSW/PSUBSW for stage-2 signed-saturating
      // accumulate.  size=10 (.2s/.4s) reconstructs 4 signed 32×32 → 64
      // products via two PMULDQs (SSE4.1), doubles via PSLLQ 1, adds the
      // rounding constant 2^31 per qword, extracts the upper 32 bits via
      // PSHUFD 0xDD + PUNPCKLDQ, applies the (INT32_MIN, INT32_MIN) corner
      // fixup via PCMPEQD + XOR, then signed-saturating accumulates via the
      // wrap-add + overflow-bit-XOR-blend recipe.
      //
      // Same recipe as the by-element form at size=01 and size=10 in
      // AdvSimdVecXIndexedElement, with two adjustments for three-same:
      //   * No Vm broadcast: Vm comes from memory as-is.
      //   * size=10 needs PSRLQ 32 on BOTH xn AND xm for the "high pass"
      //     (the by-element form's xm has all 4 dwords identical from the
      //     broadcast, so it can be reused without the extra shift).
      case Decoder::AdvSimdThreeSameOpcode::kSqrdmlahVec:
      case Decoder::AdvSimdThreeSameOpcode::kSqrdmlshVec: {
        const bool is_sub =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrdmlshVec);
        if (args.size == 0b01) {
          if (!host_platform::kHasSSSE3) { success_ = false; return; }

          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister xn_corner = AllocTempSimdReg();
          SimdRegister xm_corner = AllocTempSimdReg();
          SimdRegister x_min = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);

          // x_min = 0x8000 broadcast across 8 halfwords.
          as_.Pcmpeqw(x_min, x_min);
          as_.Psllw(x_min, int8_t{15});

          // Stage 1: SQRDMULH via PMULHRSW + corner fixup.
          as_.Movdqa(xn_corner, xn);
          as_.Movdqa(xm_corner, xm);
          as_.Pmulhrsw(xn, xm);              // SQRDMULH except (INT16_MIN)^2 corner.
          as_.Pcmpeqw(xn_corner, x_min);     // Vn lanes == 0x8000.
          as_.Pcmpeqw(xm_corner, x_min);     // Vm lanes == 0x8000.
          as_.Pand(xn_corner, xm_corner);    // Combined corner mask.
          as_.Pxor(xn, xn_corner);           // 0x8000 ^ 0xFFFF = 0x7FFF on corner.

          // Stage 2: PADDSW / PSUBSW into Vd (signed-saturating).
          SimdRegister xd = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xd, args.rd);
          if (is_sub) {
            as_.Psubsw(xd, xn);
          } else {
            as_.Paddsw(xd, xn);
          }

          if (!args.q) MaskLow64(xd);
          StoreVReg(args.rd, xd);
          return;
        }
        if (args.size == 0b10) {
          if (!host_platform::kHasSSE4_1) { success_ = false; return; }

          SimdRegister xn = AllocTempSimdReg();
          SimdRegister xm = AllocTempSimdReg();
          SimdRegister x_const = AllocTempSimdReg();
          SimdRegister corner = AllocTempSimdReg();
          SimdRegister xp_lo = AllocTempSimdReg();
          SimdRegister xp_hi = AllocTempSimdReg();
          SimdRegister xm_hi = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xn, args.rn);
          LoadVReg(xm, args.rm);

          // x_const = INT32_MIN broadcast across 4 dwords.
          as_.Pcmpeqd(x_const, x_const);
          as_.Pslld(x_const, int8_t{31});

          // Corner detection: lanes where Vn.s[i] == INT32_MIN AND
          // Vm.s[i] == INT32_MIN.  PCMPEQD destroys dst, so preserve xn and
          // xm in scratch slots first.
          as_.Movdqa(corner, xn);
          as_.Pcmpeqd(corner, x_const);
          as_.Movdqa(xp_lo, xm);
          as_.Pcmpeqd(xp_lo, x_const);
          as_.Pand(corner, xp_lo);
          // xp_lo is dead and about to be repurposed.

          // Stage 1: two PMULDQs reconstruct the 4 signed 32×32 → 64 products.
          //   xp_lo qword 0 = sext_i64(Vn.s[0]) * sext_i64(Vm.s[0])
          //   xp_lo qword 1 = sext_i64(Vn.s[2]) * sext_i64(Vm.s[2])
          //   xp_hi qword 0 = sext_i64(Vn.s[1]) * sext_i64(Vm.s[1])
          //   xp_hi qword 1 = sext_i64(Vn.s[3]) * sext_i64(Vm.s[3])
          // Unlike the by-element form (Vm broadcast → all 4 dwords identical),
          // the three-same Vm has distinct dwords per lane, so the "high pass"
          // needs PSRLQ 32 on BOTH xn AND xm.
          as_.Movdqa(xp_lo, xn);
          as_.Pmuldq(xp_lo, xm);
          as_.Movdqa(xp_hi, xn);
          as_.Psrlq(xp_hi, int8_t{32});
          as_.Movdqa(xm_hi, xm);
          as_.Psrlq(xm_hi, int8_t{32});
          as_.Pmuldq(xp_hi, xm_hi);
          // xn, xm, xm_hi are dead from here; x_const is dead (reused next).

          // Double each 64-bit signed product via PSLLQ 1.
          as_.Psllq(xp_lo, int8_t{1});
          as_.Psllq(xp_hi, int8_t{1});

          // Rounding constant 2^31 = 0x80000000 per qword.
          as_.Pcmpeqd(x_const, x_const);
          as_.Psllq(x_const, int8_t{63});
          as_.Psrlq(x_const, int8_t{32});
          as_.Paddq(xp_lo, x_const);
          as_.Paddq(xp_hi, x_const);

          // Extract the upper 32 bits of each 64-bit lane.
          as_.Pshufd(xp_lo, xp_lo, static_cast<int8_t>(0xDD));
          as_.Pshufd(xp_hi, xp_hi, static_cast<int8_t>(0xDD));
          as_.Punpckldq(xp_lo, xp_hi);

          // Apply corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX.
          as_.Pxor(xp_lo, corner);

          SimdRegister xmm_result = xp_lo;

          // Stage 2: 32-bit signed-saturating add/sub of Vd and the stage-1
          // product, via the wrap-add + overflow-bit-XOR-blend recipe.
          //   sum  = a ± b              (PADDD/PSUBD wraps mod 2^32)
          //   ovf  = ~(a^b) & (a^sum)   for ADD
          //        |  (a^b) & (a^diff)  for SUB
          //   sat  = (a < 0) ? INT32_MIN : INT32_MAX
          //   res  = sum ^ ((sum ^ sat) & ovf_mask)
          SimdRegister xd = AllocTempSimdReg();
          SimdRegister t_sum = AllocTempSimdReg();
          SimdRegister t_ovf = AllocTempSimdReg();
          SimdRegister t_sat = AllocTempSimdReg();
          if (!success()) { return; }
          LoadVReg(xd, args.rd);
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
          if (!is_sub) {
            as_.Pcmpeqd(xmm_result, xmm_result);
            as_.Pxor(t_ovf, xmm_result);       // ~(a^b)
            as_.Pand(t_ovf, t_sat);            // ~(a^b) & (a^sum)
            as_.Psrld(xmm_result, int8_t{1});  // INT32_MAX broadcast
          } else {
            as_.Pand(t_ovf, t_sat);            // (a^b) & (a^diff)
            as_.Pcmpeqd(xmm_result, xmm_result);
            as_.Psrld(xmm_result, int8_t{1});  // INT32_MAX broadcast
          }
          as_.Psrad(t_ovf, int8_t{31});        // overflow mask (all-1s or 0)
          as_.Psrad(xd, int8_t{31});           // xd = (a<0) ? -1 : 0
          as_.Pxor(xd, xmm_result);            // xd = sat = INT_MIN or INT_MAX
          as_.Pxor(xd, t_sum);                 // xd = sat ^ sum
          as_.Pand(xd, t_ovf);                 // ... & ovf
          as_.Pxor(xd, t_sum);                 // = sum ^ ((sum^sat) & ovf)

          if (!args.q) MaskLow64(xd);
          StoreVReg(args.rd, xd);
          return;
        }
        // size=00 / size=11 are reserved by the decoder; bail safely.
        success_ = false; return;
      }
      // SSHL / USHL vector form: signed/unsigned variable
      // shift.  Per-lane shift count is the signed int8 in the low byte of
      // the corresponding Vm lane.  Per-lane GPR-branched recipe (the scalar
      // D recipe inlined N times, with element-width-specific zero and
      // sign-broadcast thresholds).  Quadrants per lane of width W bits:
      //   sh in [0, W-1]      -> SHL a, sh
      //   sh >= W             -> 0 (USHL); SAR a, W-1 (SSHL sign broadcast).
      //   sh in [-(W-1), -1]  -> SHR a, |sh|  (USHL); SAR a, |sh|  (SSHL).
      //   sh <= -W            -> 0  (USHL);  SAR a, W-1  (SSHL).
      // Scalar D (kSshl/kUshl scalar) handled separately via
      // AdvSimdScalarThreeSame.  This cycle adds .8B/.16B (size=00); .4H/.8H
      // (size=01), .2S/.4S (size=10) and .2D (size=11) are unchanged from
      // prior cycles.  .1D (Q=0, size=11) is ARM-reserved.  Full SSHL/USHL
      // × {.8B, .16B, .4H, .8H, .2S, .4S, .2D} matrix now JIT-lowered.
      case Decoder::AdvSimdThreeSameOpcode::kSshl:
      case Decoder::AdvSimdThreeSameOpcode::kUshl: {
        if (args.size != 0b00 && args.size != 0b01 && args.size != 0b10 &&
            args.size != 0b11) {
          success_ = false; return;
        }
        if (args.size == 0b11 && !args.q) {
          // .1D Q=0 size=11 is ARM-reserved; bail.
          success_ = false; return;
        }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSshl);
        Register a = AllocTempReg();
        Register sh = AllocTempReg();
        if (a == Assembler::no_register || sh == Assembler::no_register) {
          success_ = false; return;
        }
        // Save rcx (variable shift uses cl; rcx is in the allocator pool).
        as_.Subq(Assembler::rsp, 8);
        as_.Movq({.base = Assembler::rsp}, Assembler::rcx);

        if (args.size == 0b11) {
          // .2D: 2 lanes of 64-bit.
          for (int lane = 0; lane < 2; ++lane) {
            int32_t vn_lane = vn_off + lane * 8;
            int32_t vm_lane = vm_off + lane * 8;
            int32_t vd_lane = vd_off + lane * 8;

            as_.Movq(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: shift >= 64 -> 0; else SHL a, sh.
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShlqByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative path: sh = |shift| in [1, 128] after Negq.  Source was
            // sign-extension of int8 (range [-128, -1]) so the negation cannot
            // overflow.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{64});
            if (is_signed) {
              Assembler::Label* L_arith_max = as_.MakeLabel();
              as_.Jcc(Assembler::Condition::kGreaterEqual, *L_arith_max);
              // 0 < |sh| < 64: a = (int64_t)a >> |sh|.
              as_.Movq(Assembler::rcx, sh);
              as_.SarqByCl(a);
              as_.Jmp(*L_done);
              as_.Bind(L_arith_max);
              // |sh| >= 64: a = (int64_t)a >> 63 (sign-broadcast across bits).
              as_.Sarq(a, int8_t{63});
              as_.Jmp(*L_done);
            } else {
              // USHL negative: |sh| >= 64 -> 0; else SHR a, |sh|.
              as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
              as_.Movq(Assembler::rcx, sh);
              as_.ShrqByCl(a);
              as_.Jmp(*L_done);
            }

            as_.Bind(L_zero);
            as_.Xorq(a, a);

            as_.Bind(L_done);

            // Store lane result to Vd.
            as_.Movq({.base = Assembler::rbp, .disp = vd_lane}, a);
          }
        } else if (args.size == 0b10) {
          // .2S (Q=0) / .4S (Q=1): 2 or 4 lanes of 32-bit.  Per-lane shift
          // count is the signed int8 in the LOW byte of the corresponding
          // Vm S-lane (vm_off + lane*4).  Width-32 thresholds: sh >= 32 ->
          // zero (USHL) or arith-max (SSHL); -32 < sh < 0 -> SHR/SAR by
          // |sh|; sh <= -32 -> zero (USHL) or SAR by 31 (SSHL).
          const int num_lanes = args.q ? 4 : 2;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 4;
            int32_t vm_lane = vm_off + lane * 4;
            int32_t vd_lane = vd_off + lane * 4;

            // Movl zero-extends the 32-bit value into the 64-bit register;
            // upper-32 stays 0 across the subsequent 32-bit shifts and
            // (for L_zero) Xorl.
            as_.Movl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: shift >= 32 -> 0; else SHL a, sh.
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative path: sh = |shift| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{32});
            if (is_signed) {
              Assembler::Label* L_arith_max = as_.MakeLabel();
              as_.Jcc(Assembler::Condition::kGreaterEqual, *L_arith_max);
              // 0 < |sh| < 32: a = (int32_t)a >> |sh|.
              as_.Movq(Assembler::rcx, sh);
              as_.SarlByCl(a);
              as_.Jmp(*L_done);
              as_.Bind(L_arith_max);
              // |sh| >= 32: a = (int32_t)a >> 31 (sign-broadcast across 32 bits).
              as_.Sarl(a, int8_t{31});
              as_.Jmp(*L_done);
            } else {
              // USHL negative: |sh| >= 32 -> 0; else SHR a, |sh|.
              as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
              as_.Movq(Assembler::rcx, sh);
              as_.ShrlByCl(a);
              as_.Jmp(*L_done);
            }

            as_.Bind(L_zero);
            as_.Xorl(a, a);  // 32-bit zero (x86 zero-extends to upper 32).

            as_.Bind(L_done);

            // Movl-store the 32-bit lane.  Both arms of the lane (SHL/SAR/
            // SHR/Xorl) leave bits[63:32] of `a` clean, so a 32-bit store
            // is sufficient.
            as_.Movl({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.2S), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else if (args.size == 0b01) {
          // .4H (Q=0) / .8H (Q=1): 4 or 8 lanes of 16-bit.  Per-lane shift
          // count is the signed int8 in the LOW byte of the corresponding
          // Vm H-lane (vm_off + lane*2).  Width-16 thresholds: sh >= 16 ->
          // zero (USHL) or arith-max via Sarl 15 (SSHL); -16 < sh < 0 ->
          // SHR/SAR by |sh|; sh <= -16 -> zero (USHL) or Sarl 15 (SSHL).
          // SSHL uses Movsxwl so the 32-bit shift primitives propagate the
          // 16-bit sign correctly across the upper bits; the low 16 of the
          // result is the correct H-lane value and the Movw store discards
          // the upper bits.  USHL uses Movzxwl so the value is zero-extended.
          const int num_lanes = args.q ? 8 : 4;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 2;
            int32_t vm_lane = vm_off + lane * 2;
            int32_t vd_lane = vd_off + lane * 2;

            if (is_signed) {
              // Sign-extend 16-bit Vn lane to 32-bit.  Upper 32 of the
              // 64-bit register is zero-extended by the 32-bit dest.
              as_.Movsxwl(a, {.base = Assembler::rbp, .disp = vn_lane});
            } else {
              // Zero-extend 16-bit Vn lane to 32-bit.
              as_.Movzxwl(a, {.base = Assembler::rbp, .disp = vn_lane});
            }
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: shift >= 16 -> 0 (low 16 of
            // SHL-by-cl on a 32-bit reg is 0 once cl >= 16); else SHL a, sh.
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative path: sh = |shift| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{16});
            if (is_signed) {
              Assembler::Label* L_arith_max = as_.MakeLabel();
              as_.Jcc(Assembler::Condition::kGreaterEqual, *L_arith_max);
              // 0 < |sh| < 16: a = (int32_t)a >> |sh|.  Movsxwl-loaded
              // SignExt(low_16) gives correct sign propagation across the
              // 32-bit SAR; low 16 of result = SignExt(low_16) >>s |sh|.
              as_.Movq(Assembler::rcx, sh);
              as_.SarlByCl(a);
              as_.Jmp(*L_done);
              as_.Bind(L_arith_max);
              // |sh| >= 16: a = SignExt(low_16) >>s 15 -> sign broadcast.
              // Low 16 = 0xFFFF (negative) or 0x0000 (non-negative).
              as_.Sarl(a, int8_t{15});
              as_.Jmp(*L_done);
            } else {
              // USHL negative: |sh| >= 16 -> 0; else SHR a, |sh|.
              as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
              as_.Movq(Assembler::rcx, sh);
              as_.ShrlByCl(a);
              as_.Jmp(*L_done);
            }

            as_.Bind(L_zero);
            as_.Xorl(a, a);  // 32-bit zero (x86 zero-extends to upper 32).

            as_.Bind(L_done);

            // 16-bit lane store from the low 16 of `a`.
            as_.Movw({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.4H), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else {
          // .8B (Q=0) / .16B (Q=1): 8 or 16 lanes of 8-bit.  Per-lane shift
          // count is the signed int8 in the LOW byte of the corresponding
          // Vm B-lane (vm_off + lane).  Width-8 thresholds: sh >= 8 ->
          // zero (USHL) or arith-max via Sarl 7 (SSHL); -8 < sh < 0 ->
          // SHR/SAR by |sh|; sh <= -8 -> zero (USHL) or Sarl 7 (SSHL).
          // SSHL uses Movsxbl so the 32-bit shift primitives propagate the
          // 8-bit sign correctly across the upper bits; the low 8 of the
          // result is the correct B-lane value and the Movb store discards
          // the upper bits.  USHL uses Movzxbl so the value is zero-extended.
          const int num_lanes = args.q ? 16 : 8;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane;
            int32_t vm_lane = vm_off + lane;
            int32_t vd_lane = vd_off + lane;

            if (is_signed) {
              // Sign-extend 8-bit Vn lane to 32-bit.  Upper 32 of the
              // 64-bit register is zero-extended by the 32-bit dest.
              as_.Movsxbl(a, {.base = Assembler::rbp, .disp = vn_lane});
            } else {
              // Zero-extend 8-bit Vn lane to 32-bit.
              as_.Movzxbl(a, {.base = Assembler::rbp, .disp = vn_lane});
            }
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: shift >= 8 -> 0; else SHL a, sh.
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative path: sh = |shift| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{8});
            if (is_signed) {
              Assembler::Label* L_arith_max = as_.MakeLabel();
              as_.Jcc(Assembler::Condition::kGreaterEqual, *L_arith_max);
              // 0 < |sh| < 8: a = (int32_t)a >> |sh|.  Movsxbl-loaded
              // SignExt(low_8) gives correct sign propagation across the
              // 32-bit SAR; low 8 of result = SignExt(low_8) >>s |sh|.
              as_.Movq(Assembler::rcx, sh);
              as_.SarlByCl(a);
              as_.Jmp(*L_done);
              as_.Bind(L_arith_max);
              // |sh| >= 8: a = SignExt(low_8) >>s 7 -> sign broadcast.
              // Low 8 = 0xFF (negative) or 0x00 (non-negative).
              as_.Sarl(a, int8_t{7});
              as_.Jmp(*L_done);
            } else {
              // USHL negative: |sh| >= 8 -> 0; else SHR a, |sh|.
              as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
              as_.Movq(Assembler::rcx, sh);
              as_.ShrlByCl(a);
              as_.Jmp(*L_done);
            }

            as_.Bind(L_zero);
            as_.Xorl(a, a);  // 32-bit zero (x86 zero-extends to upper 32).

            as_.Bind(L_done);

            // 8-bit lane store from the low 8 of `a`.
            as_.Movb({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.8B), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        }

        // Restore rcx.
        as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
        as_.Addq(Assembler::rsp, 8);
        return;
      }
      // UQSHL vector form: unsigned saturating variable
      // shift.  The per-lane shift count is the signed int8 in the low byte
      // of the corresponding Vm lane.  Per-lane GPR-branched recipe (the
      // UQSHL scalar D recipe inlined N times, with element-width-specific
      // saturation thresholds).  Quadrants per lane of width W bits:
      //   sh in [0, W-1]      -> shift-then-back-shift overflow check; on
      //                          lost bits, saturate to UINT_W_MAX.
      //   sh >= W             -> a == 0 ? 0 : UINT_W_MAX.
      //   sh in [-(W-1), -1]  -> SHR a, |sh|  (no saturation; right-shifting
      //                          an unsigned value cannot overflow).
      //   sh <= -W            -> 0.
      // Scalar D form (kUqshlScalar) handled separately via
      // AdvSimdScalarThreeSame.  This case lowers .2D (size=11), .2S/.4S
      // (size=10), .4H/.8H (size=01), and .8B/.16B (size=00) — the
      // complete .B/.H/.S/.D vector matrix.  .1D (Q=0, size=11) is
      // ARM-reserved.
      case Decoder::AdvSimdThreeSameOpcode::kUqshl: {
        if (args.size == 0b11 && !args.q) {
          // .1D Q=0 size=11 is ARM-reserved; bail.
          success_ = false; return;
        }
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

        if (args.size == 0b11) {
          // .2D: 2 lanes of 64-bit.
          for (int lane = 0; lane < 2; ++lane) {
            int32_t vn_lane = vn_off + lane * 8;
            int32_t vm_lane = vm_off + lane * 8;
            int32_t vd_lane = vd_off + lane * 8;

            as_.Movq(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 64 falls to the a-vs-UINT64_MAX
            // picker; otherwise the back-shift overflow check.
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 64: shift-then-back-shift overflow check.  Reuse `sh`
            // as the saved-original holder once cl is loaded (shift count
            // already moved to rcx, so the sh register is free).
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);
            as_.ShlqByCl(a);
            as_.Movq(back, a);
            as_.ShrqByCl(back);
            as_.Cmpq(back, sh);
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            as_.Movq(a, static_cast<int64_t>(-1));  // saturate to UINT64_MAX.
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 64: if a == 0 result is 0, else UINT64_MAX.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movq(a, static_cast<int64_t>(-1));
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 0 < |sh| < 64: a >>u |sh|.
            as_.Movq(Assembler::rcx, sh);
            as_.ShrqByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorq(a, a);

            as_.Bind(L_done);

            // Store lane result to Vd.
            as_.Movq({.base = Assembler::rbp, .disp = vd_lane}, a);
          }
        } else if (args.size == 0b10) {
          // .2S (Q=0) / .4S (Q=1): 2 or 4 lanes of 32-bit.  Movl zero-
          // extends the 32-bit lane into the upper 32; the subsequent
          // 32-bit shift / Xorl chain leaves bits[63:32] of `a` clean, so
          // the 32-bit Cmpl / Movl lane store work directly.  Width-32
          // thresholds replace the width-64 ones above (sh >= 32 -> a==0
          // picker; |sh| >= 32 -> zero; saturate to UINT32_MAX instead of
          // UINT64_MAX).
          const int num_lanes = args.q ? 4 : 2;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 4;
            int32_t vm_lane = vm_off + lane * 4;
            int32_t vd_lane = vd_off + lane * 4;

            as_.Movl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 32 falls to the a-vs-UINT32_MAX
            // picker; otherwise the back-shift overflow check.
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 32: shift-then-back-shift overflow check.  Reuse `sh`
            // as the saved-original holder once cl is loaded (shift count
            // already moved to rcx, so the sh register is free).  Since `a`
            // is Movl-loaded (upper 32 zeroed), Movq(sh, a) preserves the
            // 32-bit original in low 32 with upper 32 = 0; Cmpl works.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);
            as_.ShllByCl(a);
            as_.Movq(back, a);
            as_.ShrlByCl(back);
            as_.Cmpl(back, sh);
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            as_.Movl(a, int32_t{-1});  // saturate to UINT32_MAX.
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 32: if a == 0 result is 0, else UINT32_MAX.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movl(a, int32_t{-1});
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 0 < |sh| < 32: a >>u |sh|.
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);  // 32-bit zero (x86 zero-extends to upper 32).

            as_.Bind(L_done);

            // Movl-store the 32-bit lane.  All arms leave bits[63:32] of
            // `a` clean, so a 32-bit store is sufficient.
            as_.Movl({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.2S), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else if (args.size == 0b01) {
          // .4H (Q=0) / .8H (Q=1): 4 or 8 lanes of 16-bit (size == 0b01).
          // Movzxwl zero-extends the 16-bit lane into 32 bits (upper 16 of
          // result = 0).  Width-16 thresholds: sh >= 16 -> a==0 picker or
          // UINT16_MAX saturation; |sh| >= 16 -> 0 (unsigned right shifts
          // cannot saturate).  Width-16 overflow detection in the back-shift
          // path: after ShllByCl, any non-zero bit in `a`'s bits[31:16] means
          // the value overflowed the 16-bit lane (since Movzxwl zeroed those
          // bits and ShllByCl can only push original 16-bit content into
          // them).  This is simpler than the .2S/.4S Cmpl-back-shift trick
          // because the 32-bit back-shift would not detect 16-bit overflow.
          const int num_lanes = args.q ? 8 : 4;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 2;
            int32_t vm_lane = vm_off + lane * 2;
            int32_t vd_lane = vd_off + lane * 2;

            as_.Movzxwl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 16 -> a-vs-UINT16_MAX picker;
            // else width-16 overflow check.
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 16: ShllByCl then test bits[31:16] of result.
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Movq(back, a);
            as_.Shrl(back, int8_t{16});
            as_.Testl(back, back);
            as_.Jcc(Assembler::Condition::kZero, *L_done);
            as_.Movl(a, int32_t{0xFFFF});  // saturate to UINT16_MAX (low 16).
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 16: if a == 0 result is 0, else UINT16_MAX.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movl(a, int32_t{0xFFFF});
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 0 < |sh| < 16: a >>u |sh|.  Movzxwl-loaded a has upper 16 = 0;
            // ShrlByCl preserves cleanly.
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movw-store the 16-bit lane (low 16 of `a`); upper bits ignored.
            as_.Movw({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.4H), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else {
          // .8B (Q=0) / .16B (Q=1): 8 or 16 lanes of 8-bit (size == 0b00).
          // Movzxbl zero-extends the 8-bit lane into 32 bits (upper 24 of
          // result = 0).  Width-8 thresholds: sh >= 8 -> a==0 picker or
          // UINT8_MAX saturation; |sh| >= 8 -> 0 (unsigned right shifts
          // cannot saturate).  Width-8 overflow detection in the back-shift
          // path: after ShllByCl, any non-zero bit in `a`'s bits[31:8] means
          // the value overflowed the 8-bit lane (since Movzxbl zeroed those
          // bits and ShllByCl can only push original 8-bit content into
          // them).  Mirrors the .4H/.8H Shrl-back-(W=16) test with W=8.
          const int num_lanes = args.q ? 16 : 8;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane;
            int32_t vm_lane = vm_off + lane;
            int32_t vd_lane = vd_off + lane;

            as_.Movzxbl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 8 -> a-vs-UINT8_MAX picker;
            // else width-8 overflow check.
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 8: ShllByCl then test bits[31:8] of result.
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Movq(back, a);
            as_.Shrl(back, int8_t{8});
            as_.Testl(back, back);
            as_.Jcc(Assembler::Condition::kZero, *L_done);
            as_.Movl(a, int32_t{0xFF});  // saturate to UINT8_MAX (low 8).
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 8: if a == 0 result is 0, else UINT8_MAX.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movl(a, int32_t{0xFF});
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 0 < |sh| < 8: a >>u |sh|.  Movzxbl-loaded a has upper 24 = 0;
            // ShrlByCl preserves cleanly.
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movb-store the 8-bit lane (low 8 of `a`); upper bits ignored.
            as_.Movb({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.8B), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        }

        // Restore rcx.
        as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
        as_.Addq(Assembler::rsp, 8);
        return;
      }
      // SQSHL vector form: signed saturating variable
      // shift.  The per-lane shift count is the signed int8 in the low byte
      // of the corresponding Vm lane.  Per-lane GPR-branched recipe (the
      // SQSHL scalar D recipe inlined N times, with element-width-specific
      // saturation thresholds).  Quadrants per lane of width W bits:
      //   sh in [0, W-1]      -> SHL then SAR back-shift overflow check.  On
      //                          mismatch saturate to INT_W_MAX (pos a) or
      //                          INT_W_MIN (neg a), picked via
      //                          `sign(saved_a) ^ INT_W_MAX`.
      //   sh >= W             -> a == 0 ? 0 : sign-aware saturation target.
      //   sh in [-(W-1), -1]  -> SAR a, |sh|  (signed arithmetic right shift;
      //                          no saturation possible on right shifts).
      //   sh <= -W            -> sign broadcast (SAR by W-1), not 0 —
      //                          preserves the sign bit for negative inputs
      //                          (-1 stays -1).
      // Scalar D form (kSqshlScalar) handled separately via
      // AdvSimdScalarThreeSame.  This case lowers .2D (size=11), .2S/.4S
      // (size=10), .4H/.8H (size=01), and .8B/.16B (size=00) — the
      // complete .B/.H/.S/.D vector matrix.  .1D (Q=0, size=11) is
      // ARM-reserved.
      case Decoder::AdvSimdThreeSameOpcode::kSqshl: {
        if (args.size == 0b11 && !args.q) {
          // .1D Q=0 size=11 is ARM-reserved; bail.
          success_ = false; return;
        }
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

        if (args.size == 0b11) {
          // .2D: 2 lanes of 64-bit.
          for (int lane = 0; lane < 2; ++lane) {
            int32_t vn_lane = vn_off + lane * 8;
            int32_t vm_lane = vm_off + lane * 8;
            int32_t vd_lane = vd_off + lane * 8;

            as_.Movq(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_neg_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 64 falls through to the
            // sign-aware saturation picker; otherwise back-shift overflow check.
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 64: shift-then-back-shift via SAR overflow check.
            // Reuse `sh` as the saved-original holder once cl is loaded.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);                 // sh = original a (saved for cmp).
            as_.ShlqByCl(a);                 // a = a << shift (candidate result).
            as_.Movq(back, a);
            as_.SarqByCl(back);              // back = (int64_t)(a << shift) >>s shift.
            as_.Cmpq(back, sh);              // back == original a ?
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: saturate to INT64_MAX if sign(a)==0 else INT64_MIN.
            // `sh` still holds the saved original a value.
            as_.Sarq(sh, int8_t{63});        // sh = sign broadcast (0 or -1).
            as_.Movq(a, int64_t{0x7FFFFFFFFFFFFFFFLL});  // a = INT64_MAX.
            as_.Xorq(a, sh);                 // pos: a = INT64_MAX; neg: a = INT64_MIN.
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 64: if a == 0 result is 0, else saturate based on sign(a).
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            // sign-broadcast a into back; then build saturation target.
            as_.Movq(back, a);
            as_.Sarq(back, int8_t{63});       // back = sign broadcast (0 or -1).
            as_.Movq(a, int64_t{0x7FFFFFFFFFFFFFFFLL});
            as_.Xorq(a, back);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_neg_big);
            // 0 < |sh| < 64: a = (int64_t)a >>s |sh|.
            as_.Movq(Assembler::rcx, sh);
            as_.SarqByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg_big);
            // |sh| >= 64: a = sign broadcast across all bits.
            as_.Sarq(a, int8_t{63});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorq(a, a);

            as_.Bind(L_done);

            // Store lane result to Vd.
            as_.Movq({.base = Assembler::rbp, .disp = vd_lane}, a);
          }
        } else if (args.size == 0b10) {
          // .2S (Q=0) / .4S (Q=1): 2 or 4 lanes of 32-bit.  Mirrors the .2D
          // recipe with width-32 substitutions: Movl zero-extends Vn lane
          // into upper 32; ShllByCl / SarlByCl replace 64-bit shifts; the
          // saturation pivot is INT32_MAX / INT32_MIN built via
          // `Sarl sh, 31; Movl a, INT32_MAX; Xorl a, sh` (sign-aware).
          // The 32-bit back-shift overflow check uses Cmpl(back, sh) since
          // both have upper 32 = 0 once Movl-loaded.  Negative-arm SAR
          // uses SarlByCl (signed); |sh|>=32 broadcasts the sign bit via
          // `Sarl a, 31` rather than zeroing (preserves -1 -> -1 invariant).
          const int num_lanes = args.q ? 4 : 2;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 4;
            int32_t vm_lane = vm_off + lane * 4;
            int32_t vd_lane = vd_off + lane * 4;

            as_.Movl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_neg_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 32 falls through to the
            // sign-aware saturation picker; otherwise back-shift overflow
            // check.
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 32: shift-then-back-shift via SAR overflow check.
            // Reuse `sh` as the saved-original holder once cl is loaded.
            // Movq(sh, a) preserves the 32-bit original in low 32 (upper
            // 32 = 0 since `a` was Movl-loaded); Cmpl compares the low 32.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);
            as_.ShllByCl(a);
            as_.Movq(back, a);
            as_.SarlByCl(back);
            as_.Cmpl(back, sh);
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: saturate to INT32_MAX (pos a) or INT32_MIN (neg a).
            // `sh` still holds the saved original a value (low 32; upper
            // 32 = 0).  Sarl sh, 31 broadcasts sign across low 32; Movl a,
            // INT32_MAX zero-extends 0x7FFFFFFF; Xorl(a, sh) flips to
            // INT32_MIN on negative.  Result: upper 32 of a stays 0.
            as_.Sarl(sh, int8_t{31});
            as_.Movl(a, int32_t{0x7FFFFFFF});
            as_.Xorl(a, sh);
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 32: if a == 0 result is 0, else saturate based on sign(a).
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movq(back, a);
            as_.Sarl(back, int8_t{31});
            as_.Movl(a, int32_t{0x7FFFFFFF});
            as_.Xorl(a, back);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_neg_big);
            // 0 < |sh| < 32: a = (int32_t)a >>s |sh|.  SarlByCl operates
            // on the low 32 and propagates sign across them; upper 32
            // remains 0 (Movl zero-ext + SarlByCl is 32-bit op).
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg_big);
            // |sh| >= 32: a = sign broadcast across low 32 bits.  Sarl by
            // 31 yields 0 (positive a) or 0xFFFFFFFF (negative a); upper
            // 32 of `a` stays 0.  -1 sign-broadcasts to 0xFFFFFFFF, which
            // is the correct signed -1 in the low 32.
            as_.Sarl(a, int8_t{31});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movl-store the 32-bit lane.  All arms leave bits[63:32] of
            // `a` clean, so a 32-bit store is sufficient.
            as_.Movl({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.2S), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else if (args.size == 0b01) {
          // .4H (Q=0) / .8H (Q=1): 4 or 8 lanes of 16-bit (size == 0b01).
          // Movsxwl sign-extends the 16-bit lane into 32 bits (upper 16 of
          // result = sign-extension of bit 15).  Width-16 overflow detection
          // in the back-shift path: after ShllByCl, the result fits in
          // signed 16-bit iff sign-extending the low 16 reproduces the full
          // 32-bit value, i.e. `(int32_t)(int16_t)result == result`.  Use
          // `Shll back, 16; Sarl back, 16` to materialize the sign-ext-from
          // -low-16 and compare against the full 32-bit shifted result.
          // Saturation pivot is INT16_MAX (0x7FFF) / INT16_MIN (0x8000)
          // selected via `Sarl sh, 31; Movl a, 0x7FFF; Xorl a, sh`, where
          // `sh` holds the Movsxwl-loaded saved-original (sign in bit 31).
          // Negative-arm |sh|>=16 sign-broadcasts via `Sarl a, 15` (matches
          // the SSHL .4H/.8H pattern; on a Movsxwl-loaded value, Sarl 15
          // yields 0 or 0xFFFFFFFF in the low 32, low 16 = 0 or 0xFFFF).
          const int num_lanes = args.q ? 8 : 4;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 2;
            int32_t vm_lane = vm_off + lane * 2;
            int32_t vd_lane = vd_off + lane * 2;

            as_.Movsxwl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_neg_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 16 -> sign-aware picker;
            // else signed-16-bit overflow check.
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 16: ShllByCl, then check if signed-16 fit.  Reuse
            // `sh` as the saved-original holder after cl is loaded.  Since
            // `a` is Movsxwl-loaded (sign in bit 31), Movq(sh, a) preserves
            // the sign-extended original for the saturation pivot.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);
            as_.ShllByCl(a);
            as_.Movq(back, a);
            as_.Shll(back, int8_t{16});
            as_.Sarl(back, int8_t{16});
            as_.Cmpl(back, a);
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: saturate to INT16_MAX (pos a) or INT16_MIN (neg a).
            // `sh` holds Movsxwl-loaded saved-original; Sarl 31 broadcasts
            // sign across low 32; Movl 0x7FFF zero-extends INT16_MAX;
            // Xorl flips to 0xFFFF8000 (low 16 = INT16_MIN) on negative.
            as_.Sarl(sh, int8_t{31});
            as_.Movl(a, int32_t{0x7FFF});
            as_.Xorl(a, sh);
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 16: if a == 0 result is 0, else sign-aware saturation.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movq(back, a);
            as_.Sarl(back, int8_t{31});
            as_.Movl(a, int32_t{0x7FFF});
            as_.Xorl(a, back);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_neg_big);
            // 0 < |sh| < 16: a = (int32_t)a >>s |sh|.  Movsxwl-loaded a
            // gives correct sign propagation across 32-bit SAR; low 16 of
            // result = SignExt(low_16) >>s |sh|.
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg_big);
            // |sh| >= 16: a = sign broadcast across the sign-extended
            // value.  Sarl 15 yields 0 (positive) or 0xFFFFFFFF (negative);
            // low 16 = 0x0000 or 0xFFFF respectively (-1 stays -1).
            as_.Sarl(a, int8_t{15});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movw-store the 16-bit lane (low 16 of `a`); upper bits ignored.
            as_.Movw({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.4H), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else {
          // .8B (Q=0) / .16B (Q=1): 8 or 16 lanes of 8-bit (size == 0b00).
          // Movsxbl sign-extends the 8-bit lane into 32 bits (upper 24 of
          // result = sign-extension of bit 7).  Width-8 overflow detection
          // in the back-shift path: after ShllByCl, the result fits in
          // signed 8-bit iff sign-extending the low 8 reproduces the full
          // 32-bit value, i.e. `(int32_t)(int8_t)result == result`.  Use
          // `Shll back, 24; Sarl back, 24` to materialize the sign-ext-from
          // -low-8 and compare against the full 32-bit shifted result.
          // Saturation pivot is INT8_MAX (0x7F) / INT8_MIN (0x80) selected
          // via `Sarl sh, 31; Movl a, 0x7F; Xorl a, sh`, where `sh` holds
          // the Movsxbl-loaded saved-original (sign in bit 31).  Negative
          // -arm |sh|>=8 sign-broadcasts via `Sarl a, 7` (matches the SSHL
          // .8B/.16B pattern; on a Movsxbl-loaded value, Sarl 7 yields 0
          // or 0xFFFFFFFF in the low 32, low 8 = 0 or 0xFF).
          const int num_lanes = args.q ? 16 : 8;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane;
            int32_t vm_lane = vm_off + lane;
            int32_t vd_lane = vd_off + lane;

            as_.Movsxbl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_neg_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 8 -> sign-aware picker;
            // else signed-8-bit overflow check.
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 8: ShllByCl, then check if signed-8 fit.  Reuse
            // `sh` as the saved-original holder after cl is loaded.  Since
            // `a` is Movsxbl-loaded (sign in bit 31), Movq(sh, a) preserves
            // the sign-extended original for the saturation pivot.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);
            as_.ShllByCl(a);
            as_.Movq(back, a);
            as_.Shll(back, int8_t{24});
            as_.Sarl(back, int8_t{24});
            as_.Cmpl(back, a);
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: saturate to INT8_MAX (pos a) or INT8_MIN (neg a).
            // `sh` holds Movsxbl-loaded saved-original; Sarl 31 broadcasts
            // sign across low 32; Movl 0x7F zero-extends INT8_MAX; Xorl
            // flips to 0xFFFFFF80 (low 8 = INT8_MIN) on negative.
            as_.Sarl(sh, int8_t{31});
            as_.Movl(a, int32_t{0x7F});
            as_.Xorl(a, sh);
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 8: if a == 0 result is 0, else sign-aware saturation.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movq(back, a);
            as_.Sarl(back, int8_t{31});
            as_.Movl(a, int32_t{0x7F});
            as_.Xorl(a, back);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_neg_big);
            // 0 < |sh| < 8: a = (int32_t)a >>s |sh|.  Movsxbl-loaded a
            // gives correct sign propagation across the 32-bit SAR; low 8
            // of result = SignExt(low_8) >>s |sh|.
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg_big);
            // |sh| >= 8: a = sign broadcast across the sign-extended
            // value.  Sarl 7 yields 0 (positive) or 0xFFFFFFFF (negative);
            // low 8 = 0x00 or 0xFF respectively (-1 stays -1).
            as_.Sarl(a, int8_t{7});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movb-store the 8-bit lane (low 8 of `a`); upper bits ignored.
            as_.Movb({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.8B), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        }

        // Restore rcx.
        as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
        as_.Addq(Assembler::rsp, 8);
        return;
      }
      // URSHL vector form: unsigned non-saturating rounded
      // variable shift.  Per-lane GPR-branched recipe (the URSHL scalar D
      // recipe inlined N times, with element-width-specific thresholds).
      // Left shifts truncate (no saturation); right shifts add a round-half-up
      // bias via the overflow-safe identity
      // (a >>u rshift) + ((a >>u (rshift-1)) & 1).  Quadrants per lane of
      // width W bits:
      //   sh in [0, W-1]      -> SHL a, sh
      //   sh >= W             -> 0
      //   sh in [-(W-1), -1]  -> (a >>u |sh|) + ((a >>u (|sh|-1)) & 1)
      //   sh == -W            -> bit (W-1) of a (round-bit alone;
      //                          (a >>u (W-1)) & 1)
      //   sh <= -(W+1)        -> 0 (rounding term dominates per ARM ARM).
      // Scalar D form (kUrshlScalar) handled separately via
      // AdvSimdScalarThreeSame.  This case lowers .2D (size=11), .2S/.4S
      // (size=10), .4H/.8H (size=01), and .8B/.16B (size=00) — full
      // .B/.H/.S/.D vector grid.  .1D (Q=0, size=11) is ARM-reserved.
      case Decoder::AdvSimdThreeSameOpcode::kUrshl: {
        if (args.size == 0b11 && !args.q) {
          // .1D Q=0 size=11 is ARM-reserved; bail.
          success_ = false; return;
        }
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

        if (args.size == 0b11) {
          // .2D: 2 lanes of 64-bit.
          for (int lane = 0; lane < 2; ++lane) {
            int32_t vn_lane = vn_off + lane * 8;
            int32_t vm_lane = vm_off + lane * 8;
            int32_t vd_lane = vd_off + lane * 8;

            as_.Movq(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_rshift_eq_64 = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 64 -> 0; else a <<u sh.
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
            // 1 <= |sh| <= 63: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>u |sh|) + round_bit.
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
            // |sh| == 64: result = (a >>u 63) & 1 — just bit 63 (round-bit alone).
            as_.Shrq(a, int8_t{63});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorq(a, a);

            as_.Bind(L_done);

            // Store lane result to Vd.
            as_.Movq({.base = Assembler::rbp, .disp = vd_lane}, a);
          }
        } else if (args.size == 0b10) {
          // .2S (Q=0) / .4S (Q=1): 2 or 4 lanes of 32-bit.  Movl zero-
          // extends the Vn lane into bits[63:32]; subsequent 32-bit
          // shift / Andl / Xorl chain leaves bits[63:32] of `a` clean,
          // so a 32-bit Movl lane store works directly.  Width-32
          // thresholds replace width-64 ones: positive sh >= 32 -> 0;
          // |sh| > 32 -> 0; |sh| == 32 -> bit 31 of a.  The overflow-
          // safe rounding identity (a >>u rshift) + ((a >>u (rshift-1))
          // & 1) ports cleanly; boundary case a == UINT32_MAX, |sh| ==
          // 1 yields 0x80000000 (no wraparound at 32-bit width).
          const int num_lanes = args.q ? 4 : 2;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 4;
            int32_t vm_lane = vm_off + lane * 4;
            int32_t vd_lane = vd_off + lane * 4;

            as_.Movl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_rshift_eq_32 = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 32 -> 0; else a <<u sh.
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreater, *L_zero);
            as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_32);
            // 1 <= |sh| <= 31: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>u |sh|) + round_bit.
            as_.Movq(round_bit, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(round_bit);
            as_.Andl(round_bit, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Addl(a, round_bit);
            as_.Jmp(*L_done);

            as_.Bind(L_rshift_eq_32);
            // |sh| == 32: result = (a >>u 31) & 1 — just bit 31 (round bit alone).
            as_.Shrl(a, int8_t{31});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);  // 32-bit zero (x86 zero-extends to upper 32).

            as_.Bind(L_done);

            // Movl-store the 32-bit lane.  All arms leave bits[63:32]
            // of `a` clean, so a 32-bit store is sufficient.
            as_.Movl({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.2S), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else if (args.size == 0b01) {
          // .4H (Q=0) / .8H (Q=1): 4 or 8 lanes of 16-bit.  Movzxwl zero-
          // extends the 16-bit Vn lane into bits[31:16] and the 64-bit
          // register's upper 32, so subsequent 32-bit SHL/SHR/Andl/Xorl
          // operations stay clean.  Width-16 thresholds replace width-32
          // ones: positive sh >= 16 -> 0; |sh| > 16 -> 0; |sh| == 16 ->
          // bit 15 of a (the round bit alone).  The overflow-safe
          // rounding identity (a >>u rshift) + ((a >>u (rshift-1)) & 1)
          // protects against the |a == UINT16_MAX, |sh| == 1| boundary
          // (Movzxwl a = 0x0000FFFF; round_bit shift gives 1; data shift
          // gives 0x7FFF; sum = 0x8000 — no wraparound at 16-bit width).
          const int num_lanes = args.q ? 8 : 4;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 2;
            int32_t vm_lane = vm_off + lane * 2;
            int32_t vd_lane = vd_off + lane * 2;

            as_.Movzxwl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_rshift_eq_16 = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 16 -> 0; else a <<u sh.
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreater, *L_zero);
            as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_16);
            // 1 <= |sh| <= 15: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>u |sh|) + round_bit.
            as_.Movq(round_bit, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(round_bit);
            as_.Andl(round_bit, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Addl(a, round_bit);
            as_.Jmp(*L_done);

            as_.Bind(L_rshift_eq_16);
            // |sh| == 16: result = (a >>u 15) & 1 — just bit 15 (round
            // bit alone).  Low 16 of a after Shrl 15 = 0 or 1.
            as_.Shrl(a, int8_t{15});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // 16-bit lane store from the low 16 of `a`.
            as_.Movw({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.4H), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else {
          // .8B (Q=0) / .16B (Q=1): 8 or 16 lanes of 8-bit.  Movzxbl zero-
          // extends the 8-bit Vn lane into bits[31:8] and the 64-bit
          // register's upper 32, so subsequent 32-bit SHL/SHR/Andl/Xorl
          // operations stay clean.  Width-8 thresholds replace width-16
          // ones: positive sh >= 8 -> 0; |sh| > 8 -> 0; |sh| == 8 ->
          // bit 7 of a (the round bit alone).  The overflow-safe
          // rounding identity (a >>u rshift) + ((a >>u (rshift-1)) & 1)
          // protects against the |a == UINT8_MAX, |sh| == 1| boundary
          // (Movzxbl a = 0x000000FF; round_bit shift gives 1; data shift
          // gives 0x7F; sum = 0x80 — no wraparound at 8-bit width).
          const int num_lanes = args.q ? 16 : 8;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 1;
            int32_t vm_lane = vm_off + lane * 1;
            int32_t vd_lane = vd_off + lane * 1;

            as_.Movzxbl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_rshift_eq_8 = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 8 -> 0; else a <<u sh.
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreater, *L_zero);
            as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_8);
            // 1 <= |sh| <= 7: round_bit = (a >>u (|sh|-1)) & 1;
            //                 a = (a >>u |sh|) + round_bit.
            as_.Movq(round_bit, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(round_bit);
            as_.Andl(round_bit, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Addl(a, round_bit);
            as_.Jmp(*L_done);

            as_.Bind(L_rshift_eq_8);
            // |sh| == 8: result = (a >>u 7) & 1 — just bit 7 (round
            // bit alone).  Low 8 of a after Shrl 7 = 0 or 1.
            as_.Shrl(a, int8_t{7});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // 8-bit lane store from the low 8 of `a`.
            as_.Movb({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.8B), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        }

        // Restore rcx.
        as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
        as_.Addq(Assembler::rsp, 8);
        return;
      }
      // SRSHL vector form: signed non-saturating rounded
      // variable shift.  Per-lane GPR-branched recipe.  Mirror of URSHL with
      // two changes from the unsigned recipe: the data shift on the rounding-
      // quadrant arm uses SAR (signed) instead of SHR, and the |sh|>=W
      // boundary collapses into a single zero branch (no dedicated |sh|=W
      // case — for any int_W a, the signed bias 1<<(W-1) sums with a to a non-
      // negative int_2W in [0, 2^W-1] which >>s W = 0).  The round-bit
      // extract stays unsigned (it is a bit-presence check, not arithmetic).
      // Quadrants per lane of width W bits:
      //   sh in [0, W-1]      -> SHL a, sh  (left shifts truncate — same as SSHL)
      //   sh >= W             -> 0
      //   sh in [-(W-1), -1]  -> (a >>s |sh|) + ((a >>u (|sh|-1)) & 1)
      //   sh <= -W            -> 0 (signed-bias sum >>s W = 0 per ARM ARM)
      // Scalar D form (kSrshlScalar) handled separately via
      // AdvSimdScalarThreeSame.  This case lowers .2D (size=11), .2S/.4S
      // (size=10), .4H/.8H (size=01), and .8B/.16B (size=00) — full
      // .B/.H/.S/.D vector grid.  .1D (Q=0, size=11) is ARM-reserved.
      case Decoder::AdvSimdThreeSameOpcode::kSrshl: {
        if (args.size == 0b11 && !args.q) {
          // .1D Q=0 size=11 is ARM-reserved; bail.
          success_ = false; return;
        }
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

        if (args.size == 0b11) {
          // .2D: 2 lanes of 64-bit.
          for (int lane = 0; lane < 2; ++lane) {
            int32_t vn_lane = vn_off + lane * 8;
            int32_t vm_lane = vm_off + lane * 8;
            int32_t vd_lane = vd_off + lane * 8;

            as_.Movq(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 64 -> 0; else a <<u sh.
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShlqByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.  |sh|>=64 -> 0.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 1 <= |sh| <= 63: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>s |sh|) + round_bit.
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

            // Store lane result to Vd.
            as_.Movq({.base = Assembler::rbp, .disp = vd_lane}, a);
          }
        } else if (args.size == 0b10) {
          // .2S (Q=0) / .4S (Q=1): 2 or 4 lanes of 32-bit.  Movl zero-
          // extends the Vn lane into bits[63:32]; subsequent 32-bit
          // SAR / SHR / Andl / Addl / Xorl chain leaves bits[63:32] of
          // `a` clean, so a 32-bit Movl lane store works directly.
          // Width-32 thresholds replace width-64 ones: positive sh >= 32
          // -> 0; |sh| >= 32 -> 0 (single threshold — SRSHL collapses
          // both |sh|=W and |sh|>W into the zero branch, unlike URSHL
          // which has a dedicated |sh|=W bit-31-only branch).
          const int num_lanes = args.q ? 4 : 2;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 4;
            int32_t vm_lane = vm_off + lane * 4;
            int32_t vd_lane = vd_off + lane * 4;

            as_.Movl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 32 -> 0; else a <<u sh.
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.  |sh|>=32 -> 0.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 1 <= |sh| <= 31: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>s |sh|) + round_bit.
            as_.Movq(round_bit, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(round_bit);
            as_.Andl(round_bit, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Addl(a, round_bit);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);  // 32-bit zero (x86 zero-extends to upper 32).

            as_.Bind(L_done);

            // Movl-store the 32-bit lane.  All arms leave bits[63:32]
            // of `a` clean, so a 32-bit store is sufficient.
            as_.Movl({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.2S), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else if (args.size == 0b01) {
          // .4H (Q=0) / .8H (Q=1): 4 or 8 lanes of 16-bit.  Movsxwl sign-
          // extends the 16-bit Vn lane into bits[31:16] so the 32-bit
          // SAR propagates the original sign bit correctly across the
          // upper bits; the low 16 of the result is the correct H-lane
          // value and the Movw store discards the upper bits.  Width-16
          // thresholds replace width-32 ones: positive sh >= 16 -> 0;
          // |sh| >= 16 -> 0 (single threshold — SRSHL collapses both
          // |sh|=W and |sh|>W into the zero branch, unlike URSHL which
          // has a dedicated |sh|=W bit-15-only branch).  Round-bit
          // extract remains unsigned (Shrl) because it's a bit-presence
          // check, not arithmetic.
          const int num_lanes = args.q ? 8 : 4;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 2;
            int32_t vm_lane = vm_off + lane * 2;
            int32_t vd_lane = vd_off + lane * 2;

            as_.Movsxwl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 16 -> 0; else a <<u sh.
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.  |sh|>=16 -> 0.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 1 <= |sh| <= 15: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>s |sh|) + round_bit.
            as_.Movq(round_bit, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(round_bit);
            as_.Andl(round_bit, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Addl(a, round_bit);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // 16-bit lane store from the low 16 of `a`.
            as_.Movw({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.4H), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else {
          // .8B (Q=0) / .16B (Q=1): 8 or 16 lanes of 8-bit.  Movsxbl sign-
          // extends the 8-bit Vn lane into bits[31:8] so the 32-bit
          // SAR propagates the original sign bit correctly across the
          // upper bits; the low 8 of the result is the correct B-lane
          // value and the Movb store discards the upper bits.  Width-8
          // thresholds replace width-16 ones: positive sh >= 8 -> 0;
          // |sh| >= 8 -> 0 (single threshold — SRSHL collapses both
          // |sh|=W and |sh|>W into the zero branch, unlike URSHL which
          // has a dedicated |sh|=W bit-7-only branch).  Round-bit
          // extract remains unsigned (Shrl) because it's a bit-presence
          // check, not arithmetic.
          const int num_lanes = args.q ? 16 : 8;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 1;
            int32_t vm_lane = vm_off + lane * 1;
            int32_t vd_lane = vd_off + lane * 1;

            as_.Movsxbl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 8 -> 0; else a <<u sh.
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.  |sh|>=8 -> 0.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 1 <= |sh| <= 7: round_bit = (a >>u (|sh|-1)) & 1;
            //                 a = (a >>s |sh|) + round_bit.
            as_.Movq(round_bit, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(round_bit);
            as_.Andl(round_bit, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Addl(a, round_bit);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // 8-bit lane store from the low 8 of `a`.
            as_.Movb({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.8B), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        }

        // Restore rcx.
        as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
        as_.Addq(Assembler::rsp, 8);
        return;
      }
      // UQRSHL vector form: unsigned saturating rounded
      // variable shift.  Per-lane GPR-branched recipe — combines the UQSHL
      // positive arm (shift-then-back-shift overflow detector + saturate-to-
      // UINT_W_MAX) with the URSHL negative arm (overflow-safe rounding
      // identity + dedicated |sh|=W bit-(W-1)-only branch).  Saturation
      // only on the left-shift quadrant, rounding only on the right-shift
      // quadrant, so the two scaffoldings compose without interaction.
      // `scratch` doubles as the back-shift holder (positive arm) and
      // round_bit holder (negative arm) — the disjoint-arm 3-temp sharing
      // pattern from the scalar D recipe.
      // Quadrants per lane of width W bits:
      //   sh in [0, W-1]      -> SHL then SHR back-shift overflow check; on
      //                           mismatch saturate to UINT_W_MAX
      //   sh >= W             -> a == 0 ? 0 : UINT_W_MAX
      //   sh in [-(W-1), -1]  -> (a >>u |sh|) + ((a >>u (|sh|-1)) & 1)
      //   sh == -W            -> bit (W-1) of a (round bit alone)
      //   sh <= -(W+1)        -> 0
      // Scalar D form (kUqrshlScalar) handled separately via
      // AdvSimdScalarThreeSame.  This case lowers .2D (size=11),
      // .2S/.4S (size=10), .4H/.8H (size=01), and .8B/.16B (size=00) —
      // full vector grid.  .1D (Q=0, size=11) is ARM-reserved.
      case Decoder::AdvSimdThreeSameOpcode::kUqrshl: {
        if (args.size == 0b11 && !args.q) {
          // .1D Q=0 size=11 is ARM-reserved; bail.
          success_ = false; return;
        }
        Register a = AllocTempReg();
        Register sh = AllocTempReg();
        Register scratch = AllocTempReg();
        if (a == Assembler::no_register || sh == Assembler::no_register ||
            scratch == Assembler::no_register) {
          success_ = false; return;
        }
        // Save rcx (variable shift uses cl; rcx is in the allocator pool).
        as_.Subq(Assembler::rsp, 8);
        as_.Movq({.base = Assembler::rsp}, Assembler::rcx);

        if (args.size == 0b11) {
          // .2D: 2 lanes of 64-bit.
          for (int lane = 0; lane < 2; ++lane) {
            int32_t vn_lane = vn_off + lane * 8;
            int32_t vm_lane = vm_off + lane * 8;
            int32_t vd_lane = vd_off + lane * 8;

            as_.Movq(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_rshift_eq_64 = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 64 falls through to the
            // a-vs-UINT64_MAX picker; otherwise back-shift overflow check.
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 64: shift-then-back-shift overflow check.  Reuse `sh`
            // as the saved-original holder once cl is loaded.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);                 // sh = original a (saved for cmp).
            as_.ShlqByCl(a);                 // a = a << shift (candidate result).
            as_.Movq(scratch, a);
            as_.ShrqByCl(scratch);           // scratch = (a << shift) >> shift.
            as_.Cmpq(scratch, sh);           // scratch == original a ?
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: bits were lost above the qword; saturate to UINT64_MAX.
            as_.Movq(a, static_cast<int64_t>(-1));
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 64: if a == 0 result is 0, else UINT64_MAX.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movq(a, static_cast<int64_t>(-1));
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreater, *L_zero);
            as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_64);
            // 1 <= |sh| <= 63: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>u |sh|) + round_bit.
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

            as_.Bind(L_rshift_eq_64);
            // |sh| == 64: result = (a >>u 63) & 1 — just bit 63 (round-bit alone).
            as_.Shrq(a, int8_t{63});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorq(a, a);

            as_.Bind(L_done);

            // Store lane result to Vd.
            as_.Movq({.base = Assembler::rbp, .disp = vd_lane}, a);
          }
        } else if (args.size == 0b10) {
          // .2S (Q=0) / .4S (Q=1): 2 or 4 lanes of 32-bit.  Movl zero-
          // extends the Vn lane into bits[63:32]; subsequent 32-bit shift /
          // Andl / Xorl chain leaves bits[63:32] of `a` clean, so a 32-bit
          // Movl lane store is sufficient.  Width-32 thresholds replace
          // width-64 ones: positive sh in [0,31] gets back-shift overflow
          // check; sh>=32 picks 0 or UINT32_MAX based on a==0; |sh|=32
          // extracts bit 31 (round bit alone); |sh|>32 collapses to 0.
          const int num_lanes = args.q ? 4 : 2;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 4;
            int32_t vm_lane = vm_off + lane * 4;
            int32_t vd_lane = vd_off + lane * 4;

            as_.Movl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_rshift_eq_32 = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 32 falls through to the
            // a-vs-UINT32_MAX picker; otherwise back-shift overflow check.
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 32: shift-then-back-shift overflow check.  Reuse
            // `sh` as the saved-original holder once cl is loaded.  Movl
            // zero-extends, so a 32-bit Cmpl is sufficient.
            as_.Movq(Assembler::rcx, sh);
            as_.Movl(sh, a);                 // sh = original a (saved for cmp).
            as_.ShllByCl(a);                 // a = a << shift (candidate result).
            as_.Movl(scratch, a);
            as_.ShrlByCl(scratch);           // scratch = (a << shift) >>u shift.
            as_.Cmpl(scratch, sh);           // scratch == original a ?
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: bits were lost above the dword; saturate to UINT32_MAX.
            as_.Movl(a, int32_t{-1});
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 32: if a == 0 result is 0, else UINT32_MAX.
            as_.Testl(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movl(a, int32_t{-1});
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreater, *L_zero);
            as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_32);
            // 1 <= |sh| <= 31: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>u |sh|) + round_bit.
            as_.Movl(scratch, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(scratch);
            as_.Andl(scratch, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Addl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_rshift_eq_32);
            // |sh| == 32: result = (a >>u 31) & 1 — bit 31 (round-bit alone).
            as_.Shrl(a, int8_t{31});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);  // 32-bit zero (x86 zero-extends to upper 32).

            as_.Bind(L_done);

            // Movl-store the 32-bit lane.  All arms leave bits[63:32] of
            // `a` clean, so a 32-bit store is sufficient.
            as_.Movl({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.2S), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else if (args.size == 0b01) {
          // .4H (Q=0) / .8H (Q=1): 4 or 8 lanes of 16-bit (size == 0b01).
          // Movzxwl zero-extends the 16-bit lane into 32 bits (upper 16 of
          // result = 0).  Width-16 thresholds: positive sh in [0,15] gets a
          // width-16 overflow check (Shrl-back-16 + Testl bits[31:16] of the
          // shifted value — mirrors UQSHL .4H/.8H); sh>=16 picks 0 or
          // UINT16_MAX (0xFFFF); |sh|=16 extracts bit 15 alone (round-bit
          // alone); |sh|>16 collapses to 0.  Negative arm reuses the URSHL
          // .4H/.8H overflow-safe rounding identity with width-16
          // thresholds.
          const int num_lanes = args.q ? 8 : 4;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 2;
            int32_t vm_lane = vm_off + lane * 2;
            int32_t vd_lane = vd_off + lane * 2;

            as_.Movzxwl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_rshift_eq_16 = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 16 -> a-vs-UINT16_MAX picker;
            // else width-16 back-shift overflow check.
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 16: ShllByCl then test bits[31:16] of result.
            // Since Movzxwl zeroed those bits, any non-zero bit there means
            // the value overflowed the 16-bit lane.  Same recipe as
            // UQSHL .4H/.8H positive arm.
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Movq(scratch, a);
            as_.Shrl(scratch, int8_t{16});
            as_.Testl(scratch, scratch);
            as_.Jcc(Assembler::Condition::kZero, *L_done);
            as_.Movl(a, int32_t{0xFFFF});  // saturate to UINT16_MAX (low 16).
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 16: if a == 0 result is 0, else UINT16_MAX.
            as_.Testl(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movl(a, int32_t{0xFFFF});
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreater, *L_zero);
            as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_16);
            // 1 <= |sh| <= 15: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>u |sh|) + round_bit.
            // Movzxwl-loaded a has upper 16 = 0; ShrlByCl preserves
            // cleanly across the 32-bit shift.
            as_.Movl(scratch, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(scratch);
            as_.Andl(scratch, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Addl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_rshift_eq_16);
            // |sh| == 16: result = (a >>u 15) & 1 — bit 15 (round-bit alone).
            // For Movzxwl-loaded a, this is bit 15 of the original H lane.
            as_.Shrl(a, int8_t{15});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movw-store the 16-bit lane (low 16 of `a`); upper bits ignored.
            as_.Movw({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.4H), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else {
          // .8B (Q=0) / .16B (Q=1): 8 or 16 lanes of 8-bit (size == 0b00).
          // Movzxbl zero-extends the 8-bit lane into 32 bits (upper 24 of
          // result = 0).  Width-8 thresholds: positive sh in [0,7] gets a
          // width-8 overflow check (Shrl-back-8 + Testl bits[31:8] of the
          // shifted value — mirrors UQSHL .8B/.16B); sh>=8 picks 0 or
          // UINT8_MAX (0xFF); |sh|=8 extracts bit 7 alone (round-bit
          // alone); |sh|>8 collapses to 0.  Negative arm reuses the URSHL
          // .8B/.16B overflow-safe rounding identity with width-8
          // thresholds.
          const int num_lanes = args.q ? 16 : 8;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 1;
            int32_t vm_lane = vm_off + lane * 1;
            int32_t vd_lane = vd_off + lane * 1;

            as_.Movzxbl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_rshift_eq_8 = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 8 -> a-vs-UINT8_MAX picker;
            // else width-8 back-shift overflow check.
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 8: ShllByCl then test bits[31:8] of result.  Since
            // Movzxbl zeroed those bits, any non-zero bit there means the
            // value overflowed the 8-bit lane.  Same recipe as
            // UQSHL .8B/.16B positive arm.
            as_.Movq(Assembler::rcx, sh);
            as_.ShllByCl(a);
            as_.Movq(scratch, a);
            as_.Shrl(scratch, int8_t{8});
            as_.Testl(scratch, scratch);
            as_.Jcc(Assembler::Condition::kZero, *L_done);
            as_.Movl(a, int32_t{0xFF});  // saturate to UINT8_MAX (low 8).
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 8: if a == 0 result is 0, else UINT8_MAX.
            as_.Testl(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movl(a, int32_t{0xFF});
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreater, *L_zero);
            as_.Jcc(Assembler::Condition::kEqual, *L_rshift_eq_8);
            // 1 <= |sh| <= 7: round_bit = (a >>u (|sh|-1)) & 1;
            //                 a = (a >>u |sh|) + round_bit.
            // Movzxbl-loaded a has upper 24 = 0; ShrlByCl preserves
            // cleanly across the 32-bit shift.
            as_.Movl(scratch, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(scratch);
            as_.Andl(scratch, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(a);
            as_.Addl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_rshift_eq_8);
            // |sh| == 8: result = (a >>u 7) & 1 — bit 7 (round-bit alone).
            // For Movzxbl-loaded a, this is bit 7 of the original B lane.
            as_.Shrl(a, int8_t{7});
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movb-store the 8-bit lane (low 8 of `a`); upper bits ignored.
            as_.Movb({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.8B), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        }

        // Restore rcx.
        as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
        as_.Addq(Assembler::rsp, 8);
        return;
      }
      // SQRSHL vector form: signed saturating rounded
      // variable shift.  Per-lane GPR-branched recipe — combines the SQSHL
      // positive arm (SAR-based back-shift overflow detector + sign-broadcast
      // XOR-with-INT_W_MAX saturation picker) with the SRSHL negative arm
      // (overflow-safe rounding identity with SAR data shift + SHR round-bit
      // extract).  Saturation only on the left-shift quadrant, rounding only
      // on the right-shift quadrant, so the two scaffoldings compose without
      // interaction.  `scratch` doubles as the back-shift holder on the
      // positive arm (SAR variant for sign-aware overflow detection) and
      // the round_bit holder on the negative arm — the disjoint-arm
      // 3-temp sharing pattern from the scalar D recipe.  Unlike
      // UQRSHL/URSHL, there is NO dedicated |sh|=W branch on the negative
      // arm: the signed bias 1<<(W-1) sums with int_W a to a non-negative
      // int_{2W} in [0, 2^W-1] which >>s W = 0, so |sh|>=W collapses into
      // the zero branch (matches SRSHL).
      // Quadrants per lane of width W bits:
      //   sh in [0, W-1]      -> SHL then SAR back-shift overflow check; on
      //                           mismatch saturate to INT_W_MAX (pos a)
      //                           or INT_W_MIN (neg a) via sign-broadcast XOR.
      //   sh >= W             -> a == 0 ? 0 : sign-aware INT_W_MAX/INT_W_MIN.
      //   sh in [-(W-1), -1]  -> (a >>s |sh|) + ((a >>u (|sh|-1)) & 1).
      //   sh <= -W            -> 0.
      // Scalar D form (kSqrshlScalar) handled separately via
      // AdvSimdScalarThreeSame.  This case lowers .2D (size=11),
      // .2S/.4S (size=10), .4H/.8H (size=01), and .8B/.16B (size=00) —
      // full vector grid.  .1D (Q=0, size=11) is ARM-reserved.
      case Decoder::AdvSimdThreeSameOpcode::kSqrshl: {
        if (args.size == 0b11 && !args.q) {
          // .1D Q=0 size=11 is ARM-reserved; bail.
          success_ = false; return;
        }
        Register a = AllocTempReg();
        Register sh = AllocTempReg();
        Register scratch = AllocTempReg();
        if (a == Assembler::no_register || sh == Assembler::no_register ||
            scratch == Assembler::no_register) {
          success_ = false; return;
        }
        // Save rcx (variable shift uses cl; rcx is in the allocator pool).
        as_.Subq(Assembler::rsp, 8);
        as_.Movq({.base = Assembler::rsp}, Assembler::rcx);

        if (args.size == 0b11) {
          // .2D: 2 lanes of 64-bit.
          for (int lane = 0; lane < 2; ++lane) {
            int32_t vn_lane = vn_off + lane * 8;
            int32_t vm_lane = vm_off + lane * 8;
            int32_t vd_lane = vd_off + lane * 8;

            as_.Movq(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 64 falls through to the sign-
            // aware saturation picker; otherwise the SAR-back-shift overflow
            // check.
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 64: shift-then-back-shift via SAR overflow check.
            // Reuse `sh` as the saved-original holder once cl is loaded.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);                 // sh = original a (saved for cmp).
            as_.ShlqByCl(a);                 // a = a << shift (candidate result).
            as_.Movq(scratch, a);
            as_.SarqByCl(scratch);           // scratch = (int64_t)(a << shift) >>s shift.
            as_.Cmpq(scratch, sh);           // scratch == original a ?
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: saturate to INT64_MAX if sign(a)==0 else INT64_MIN.
            // `sh` still holds the saved original a value.
            as_.Sarq(sh, int8_t{63});        // sh = sign broadcast (0 or -1).
            as_.Movq(a, int64_t{0x7FFFFFFFFFFFFFFFLL});  // a = INT64_MAX.
            as_.Xorq(a, sh);                 // pos: a = INT64_MAX; neg: a = INT64_MIN.
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 64: if a == 0 result is 0, else saturate based on sign(a).
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            // sign-broadcast a into scratch; then build saturation target.
            as_.Movq(scratch, a);
            as_.Sarq(scratch, int8_t{63});   // scratch = sign broadcast (0 or -1).
            as_.Movq(a, int64_t{0x7FFFFFFFFFFFFFFFLL});
            as_.Xorq(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.  |sh|>=64 -> 0
            // (no dedicated |sh|=64 case — the signed-bias sum >>s 64 = 0).
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{64});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 1 <= |sh| <= 63: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>s |sh|) + round_bit.
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

            // Store lane result to Vd.
            as_.Movq({.base = Assembler::rbp, .disp = vd_lane}, a);
          }
        } else if (args.size == 0b10) {
          // .2S (Q=0) / .4S (Q=1): 2 or 4 lanes of 32-bit.  Movl zero-
          // extends the Vn lane into bits[63:32]; subsequent 32-bit SHL /
          // SAR / Andl / Xorl chain leaves bits[63:32] of `a` clean, so a
          // 32-bit Movl lane store is sufficient.  Width-32 thresholds
          // replace width-64 ones: positive sh in [0,31] gets SAR-back-
          // shift overflow check with sign-broadcast XOR-INT32_MAX
          // saturation picker; sh>=32 picks 0 (if a==0) or sign-aware
          // INT32_MAX/INT32_MIN; |sh|>=32 collapses to 0 (single
          // threshold — signed-bias sum at |sh|>=W is 0 per ARM ARM,
          // matching SRSHL.2S/.4S NOT UQRSHL.2S/.4S).
          const int num_lanes = args.q ? 4 : 2;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 4;
            int32_t vm_lane = vm_off + lane * 4;
            int32_t vd_lane = vd_off + lane * 4;

            as_.Movl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 32 falls through to the
            // sign-aware saturation picker; otherwise the SAR-back-shift
            // overflow check.
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 32: shift-then-back-shift via SAR overflow check.
            // Reuse `sh` as the saved-original holder once cl is loaded.
            // Movq(sh, a) preserves the 32-bit original in low 32 (upper
            // 32 = 0 since `a` was Movl-loaded); Cmpl compares the low 32.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);
            as_.ShllByCl(a);
            as_.Movq(scratch, a);
            as_.SarlByCl(scratch);
            as_.Cmpl(scratch, sh);
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: saturate to INT32_MAX (pos a) or INT32_MIN (neg a).
            // `sh` still holds the saved original a value (low 32; upper
            // 32 = 0).  Sarl sh, 31 broadcasts sign across low 32; Movl
            // a, INT32_MAX zero-extends 0x7FFFFFFF; Xorl(a, sh) flips to
            // INT32_MIN on negative.  Result: upper 32 of a stays 0.
            as_.Sarl(sh, int8_t{31});
            as_.Movl(a, int32_t{0x7FFFFFFF});
            as_.Xorl(a, sh);
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 32: if a == 0 result is 0, else saturate based on sign(a).
            as_.Testl(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movq(scratch, a);
            as_.Sarl(scratch, int8_t{31});
            as_.Movl(a, int32_t{0x7FFFFFFF});
            as_.Xorl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.  |sh|>=32 -> 0
            // (no dedicated |sh|=32 case — the signed-bias sum >>s 32 = 0;
            // matches SRSHL.2S/.4S, NOT UQRSHL.2S/.4S).
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{32});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 1 <= |sh| <= 31: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>s |sh|) + round_bit.
            as_.Movl(scratch, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(scratch);
            as_.Andl(scratch, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Addl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);  // 32-bit zero (x86 zero-extends to upper 32).

            as_.Bind(L_done);

            // Movl-store the 32-bit lane.  All arms leave bits[63:32] of
            // `a` clean, so a 32-bit store is sufficient.
            as_.Movl({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.2S), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else if (args.size == 0b01) {
          // .4H (Q=0) / .8H (Q=1): 4 or 8 lanes of 16-bit (size == 0b01).
          // Movsxwl sign-extends the 16-bit lane into 32 bits (upper 16 of
          // result = sign-extension of bit 15).  Positive arm reuses the
          // SQSHL .4H/.8H signed-16-bit overflow check: after ShllByCl, the
          // result fits in signed 16-bit iff sign-extending the low 16
          // reproduces the full 32-bit value (`Shll back, 16; Sarl back, 16;
          // Cmpl back, a`).  Saturation pivot is INT16_MAX (0x7FFF) /
          // INT16_MIN (0x8000) selected via the sign-broadcast XOR pattern
          // (`Sarl sh, 31; Movl a, 0x7FFF; Xorl a, sh`).  Negative arm
          // reuses the SRSHL .4H/.8H overflow-safe rounding identity with
          // SAR data shift + SHR round-bit; |sh|>=16 collapses to 0
          // (single threshold — no dedicated |sh|=16 case, matching SRSHL).
          const int num_lanes = args.q ? 8 : 4;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 2;
            int32_t vm_lane = vm_off + lane * 2;
            int32_t vd_lane = vd_off + lane * 2;

            as_.Movsxwl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 16 -> sign-aware picker;
            // else signed-16-bit overflow check.
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 16: ShllByCl, then check if signed-16 fit.  Reuse
            // `sh` as the saved-original holder after cl is loaded.  Since
            // `a` is Movsxwl-loaded (sign in bit 31), Movq(sh, a) preserves
            // the sign-extended original for the saturation pivot.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);
            as_.ShllByCl(a);
            as_.Movq(scratch, a);
            as_.Shll(scratch, int8_t{16});
            as_.Sarl(scratch, int8_t{16});
            as_.Cmpl(scratch, a);
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: saturate to INT16_MAX (pos a) or INT16_MIN (neg a).
            // `sh` holds Movsxwl-loaded saved-original; Sarl 31 broadcasts
            // sign across low 32; Movl 0x7FFF zero-extends INT16_MAX;
            // Xorl flips to 0xFFFF8000 (low 16 = INT16_MIN) on negative.
            as_.Sarl(sh, int8_t{31});
            as_.Movl(a, int32_t{0x7FFF});
            as_.Xorl(a, sh);
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 16: if a == 0 result is 0, else sign-aware saturation.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movq(scratch, a);
            as_.Sarl(scratch, int8_t{31});
            as_.Movl(a, int32_t{0x7FFF});
            as_.Xorl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.  |sh|>=16 -> 0
            // (no dedicated |sh|=16 case — the signed-bias sum >>s 16 = 0;
            // matches SRSHL.4H/.8H, NOT UQRSHL.4H/.8H).
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{16});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 1 <= |sh| <= 15: round_bit = (a >>u (|sh|-1)) & 1;
            //                  a = (a >>s |sh|) + round_bit.  Data shift is
            // SarlByCl (signed); round-bit extract stays ShrlByCl (bit-
            // presence check, not arithmetic).  Movsxwl-loaded a gives
            // correct sign propagation across 32-bit SAR.
            as_.Movl(scratch, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(scratch);
            as_.Andl(scratch, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Addl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movw-store the 16-bit lane (low 16 of `a`); upper bits ignored.
            as_.Movw({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.4H), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        } else {
          // .8B (Q=0) / .16B (Q=1): 8 or 16 lanes of 8-bit (size == 0b00).
          // Movsxbl sign-extends the 8-bit lane into 32 bits (upper 24 of
          // result = sign-extension of bit 7).  Positive arm reuses the
          // SQSHL .8B/.16B signed-8-bit overflow check: after ShllByCl, the
          // result fits in signed 8-bit iff sign-extending the low 8
          // reproduces the full 32-bit value (`Shll back, 24; Sarl back, 24;
          // Cmpl back, a`).  Saturation pivot is INT8_MAX (0x7F) /
          // INT8_MIN (0x80) selected via the sign-broadcast XOR pattern
          // (`Sarl sh, 31; Movl a, 0x7F; Xorl a, sh`).  Negative arm
          // reuses the SRSHL .8B/.16B overflow-safe rounding identity with
          // SAR data shift + SHR round-bit; |sh|>=8 collapses to 0
          // (single threshold — no dedicated |sh|=8 case, matching SRSHL).
          const int num_lanes = args.q ? 16 : 8;
          for (int lane = 0; lane < num_lanes; ++lane) {
            int32_t vn_lane = vn_off + lane * 1;
            int32_t vm_lane = vm_off + lane * 1;
            int32_t vd_lane = vd_off + lane * 1;

            as_.Movsxbl(a, {.base = Assembler::rbp, .disp = vn_lane});
            as_.Movsxbq(sh, {.base = Assembler::rbp, .disp = vm_lane});

            Assembler::Label* L_neg = as_.MakeLabel();
            Assembler::Label* L_pos_big = as_.MakeLabel();
            Assembler::Label* L_zero = as_.MakeLabel();
            Assembler::Label* L_done = as_.MakeLabel();

            as_.Testq(sh, sh);
            as_.Jcc(Assembler::Condition::kSign, *L_neg);

            // Positive shift in [0, 127]: sh >= 8 -> sign-aware picker;
            // else signed-8-bit overflow check.
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_pos_big);

            // 0 <= sh < 8: ShllByCl, then check if signed-8 fit.  Reuse
            // `sh` as the saved-original holder after cl is loaded.  Since
            // `a` is Movsxbl-loaded (sign in bit 31), Movq(sh, a) preserves
            // the sign-extended original for the saturation pivot.
            as_.Movq(Assembler::rcx, sh);
            as_.Movq(sh, a);
            as_.ShllByCl(a);
            as_.Movq(scratch, a);
            as_.Shll(scratch, int8_t{24});
            as_.Sarl(scratch, int8_t{24});
            as_.Cmpl(scratch, a);
            as_.Jcc(Assembler::Condition::kEqual, *L_done);
            // Overflow: saturate to INT8_MAX (pos a) or INT8_MIN (neg a).
            // `sh` holds Movsxbl-loaded saved-original; Sarl 31 broadcasts
            // sign across low 32; Movl 0x7F zero-extends INT8_MAX;
            // Xorl flips to 0xFFFFFF80 (low 8 = INT8_MIN) on negative.
            as_.Sarl(sh, int8_t{31});
            as_.Movl(a, int32_t{0x7F});
            as_.Xorl(a, sh);
            as_.Jmp(*L_done);

            as_.Bind(L_pos_big);
            // sh >= 8: if a == 0 result is 0, else sign-aware saturation.
            as_.Testq(a, a);
            as_.Jcc(Assembler::Condition::kZero, *L_zero);
            as_.Movq(scratch, a);
            as_.Sarl(scratch, int8_t{31});
            as_.Movl(a, int32_t{0x7F});
            as_.Xorl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_neg);
            // Negative shift: |sh| in [1, 128] after Negq.  |sh|>=8 -> 0
            // (no dedicated |sh|=8 case — the signed-bias sum >>s 8 = 0;
            // matches SRSHL.8B/.16B, NOT UQRSHL.8B/.16B).
            as_.Negq(sh);
            as_.Cmpq(sh, int32_t{8});
            as_.Jcc(Assembler::Condition::kGreaterEqual, *L_zero);
            // 1 <= |sh| <= 7: round_bit = (a >>u (|sh|-1)) & 1;
            //                 a = (a >>s |sh|) + round_bit.  Data shift is
            // SarlByCl (signed); round-bit extract stays ShrlByCl (bit-
            // presence check, not arithmetic).  Movsxbl-loaded a gives
            // correct sign propagation across 32-bit SAR.
            as_.Movl(scratch, a);
            as_.Decq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.ShrlByCl(scratch);
            as_.Andl(scratch, int32_t{1});
            as_.Incq(sh);
            as_.Movq(Assembler::rcx, sh);
            as_.SarlByCl(a);
            as_.Addl(a, scratch);
            as_.Jmp(*L_done);

            as_.Bind(L_zero);
            as_.Xorl(a, a);

            as_.Bind(L_done);

            // Movb-store the 8-bit lane (low 8 of `a`); upper bits ignored.
            as_.Movb({.base = Assembler::rbp, .disp = vd_lane}, a);
          }

          // For Q=0 (.8B), zero the upper 64 bits of Vd.
          if (!args.q) {
            as_.Xorq(a, a);
            as_.Movq({.base = Assembler::rbp, .disp = vd_off + 8}, a);
          }
        }

        // Restore rcx.
        as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
        as_.Addq(Assembler::rsp, 8);
        return;
      }
      // FADDP Vd.<T>, Vn.<T>, Vm.<T> (vector) — add adjacent pairs across the
      // concatenation of Vn and Vm.
      //
      // x86 HADDPS/HADDPD produce exactly the ARM lane order for the full-width
      // forms: HADDPS xn, xm gives [n0+n1, n2+n3, m0+m1, m2+m3], which is the
      // .4S result, and HADDPD gives [n0+n1, m0+m1] for .2D. No shuffle needed.
      //
      // .2S (Q=0) reads only the low 64 bits of each source, so the wanted
      // result is [n0+n1, m0+m1] with the upper half zeroed. Those two values
      // land in lanes 0 and 2 of the HADDPS result, so one PSHUFD gathers them
      // before the .2S upper-half zeroing.
      //
      // The Armv8.2 FP16 form (.4H/.8H) keeps the same pair layout but needs an
      // F16C round-trip to reach a packed-single HADDPS; see the branch below.
      case Decoder::AdvSimdThreeSameOpcode::kFaddpV: {
        if (args.is_fp16) {
          // FP16 vector FADDP .4H / .8H via the F16C round-trip.  Widen each
          // group of four half-lanes to FP32 with VCVTPH2PS, fold the adjacent
          // pairs with HADDPS — whose lane order is already ARM's — and narrow
          // once with VCVTPS2PH imm=0 (round-to-nearest-even).
          //
          //   .4H (Q=0): HADDPS(widen(Vn.h[0..3]), widen(Vm.h[0..3])) yields
          //     [n0+n1, n2+n3, m0+m1, m2+m3], the complete result.
          //   .8H (Q=1): HADDPS(widen(Vn low), widen(Vn high)) yields result
          //     lanes 0..3 and HADDPS(widen(Vm low), widen(Vm high)) lanes
          //     4..7; narrow both and recombine with PSLLDQ + POR.
          //
          // The rounding matches a native FP16 FADDP.  The sum is rounded twice
          // — once to FP32's 24 significand bits, once to FP16's 11 — and
          // double rounding of an addition is innocuous once the intermediate
          // precision is at least 2*11 + 2 = 24 bits, which FP32 exactly meets,
          // so the pair of roundings gives the same value as a single
          // correctly-rounded FP16 add.  FP32 also brackets the whole FP16
          // exponent range, so subnormal and overflowing sums narrow to the
          // same FP16 result.  A widened FP16 NaN stays NaN through ADDPS and
          // narrows back to a NaN, matching the interpreter, which runs the
          // identical FpHalfToSingle / add / FpSingleToHalf round-trip.
          if (!host_platform::kHasF16C) { success_ = false; return; }
          SimdRegister xlo = AllocTempSimdReg();
          SimdRegister xhi = AllocTempSimdReg();
          SimdRegister xacc = args.q ? AllocTempSimdReg() : no_simd_register;
          if (!success()) { return; }
          // First HADDPS group: Vn's low four halves against the second half of
          // the pair — Vm's low four for .4H, Vn's high four for .8H.
          as_.Movq(xlo, {.base = Assembler::rbp, .disp = vn_off});
          as_.Vcvtph2ps(xlo, xlo);
          if (!args.q) {
            as_.Movq(xhi, {.base = Assembler::rbp, .disp = vm_off});
          } else {
            as_.Movdqu(xhi, {.base = Assembler::rbp, .disp = vn_off});
            as_.Psrldq(xhi, int8_t{8});
          }
          as_.Vcvtph2ps(xhi, xhi);
          as_.Haddps(xlo, xhi);
          if (!args.q) {
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            // Vcvtps2ph already zeroes the upper 64 bits; no MaskLow64 needed.
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
            return;
          }
          // .8H: stash lanes 0..3, then fold Vm's two halves into lanes 4..7.
          as_.Vcvtps2ph(xacc, xlo, int8_t{0});
          as_.Movq(xlo, {.base = Assembler::rbp, .disp = vm_off});
          as_.Vcvtph2ps(xlo, xlo);
          as_.Movdqu(xhi, {.base = Assembler::rbp, .disp = vm_off});
          as_.Psrldq(xhi, int8_t{8});
          as_.Vcvtph2ps(xhi, xhi);
          as_.Haddps(xlo, xhi);
          as_.Vcvtps2ph(xlo, xlo, int8_t{0});
          as_.Pslldq(xlo, int8_t{8});
          as_.Por(xacc, xlo);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xacc);
          return;
        }
        const bool is_double = (args.size & 1);
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (!success()) { return; }
        LoadVReg(xn, args.rn);
        LoadVReg(xm, args.rm);
        if (is_double) {
          as_.Haddpd(xn, xm);
        } else {
          as_.Haddps(xn, xm);
          if (!args.q) {
            // Gather lanes 0 and 2 into lanes 0 and 1.
            as_.Pshufd(xn, xn, static_cast<int8_t>(0x08));
          }
        }
        if (!args.q) MaskLow64(xn);
        StoreVReg(args.rd, xn);
        return;
      }

      default:
        Undefined();
        return;
    }
  }

