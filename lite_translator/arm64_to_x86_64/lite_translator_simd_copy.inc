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
// Family: SIMD copy (INS/DUP/UMOV/SMOV).
// Methods:
//   AdvSimdCopy

  void AdvSimdCopy(const Decoder::AdvSimdCopyArgs& args) {
    // Decode element size + lane index from imm5. The encoding is shared by
    // every AdvSimdCopy opcode that selects a lane (UMOV / SMOV / INS-general
    // / DUP-element). UMOV and INS-general are JIT-implemented below; the
    // other opcodes still fall through to the interpreter / Undefined() path.
    uint8_t imm5_low4 = args.imm5 & 0xf;
    uint8_t esize = 0;
    uint8_t index = 0;
    if (imm5_low4 & 0x1) {
      esize = 1; index = (args.imm5 >> 1) & 0xf;
    } else if (imm5_low4 & 0x2) {
      esize = 2; index = (args.imm5 >> 2) & 0x7;
    } else if (imm5_low4 & 0x4) {
      esize = 4; index = (args.imm5 >> 3) & 0x3;
    } else if (imm5_low4 & 0x8) {
      esize = 8; index = (args.imm5 >> 4) & 0x1;
    }

    // UMOV (unsigned move Vn.B/H/S/D[index] -> Rd). The destination width
    // is encoded by Q: Q=0 -> Wd (32-bit, zero-extended), Q=1 -> Xd. The
    // architecture only defines (esize=1,Q=0) (esize=2,Q=0) (esize=4,Q=0)
    // (esize=8,Q=1); other combinations are unallocated. Fall back to the
    // interpreter for non-canonical pairs rather than guess.
    //
    // Doubleword variant (UMOV Xd, Vn.D[i]) must use a 64-bit memory load
    // (movq, REX.W) — an earlier DUP fix flagged 32-bit-MOVD-vs-64-bit-MOVQ
    // as a load-bearing class of bug; the switch below is explicit about
    // which mov-width belongs at each esize.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kUmov && esize != 0) {
      bool canonical = (esize == 8 && args.q) || (esize != 8 && !args.q);
      if (!canonical) { success_ = false; return; }
      if (args.rd < 31) {
        Register tmp = AllocTempReg();
        if (tmp == no_register) { success_ = false; return; }
        int32_t off =
            VRegOffset(args.rn) + index * esize;
        switch (esize) {
          case 1: as_.Movzxbq(tmp, {.base = Assembler::rbp, .disp = off}); break;
          case 2: as_.Movzxwq(tmp, {.base = Assembler::rbp, .disp = off}); break;
          // Movl on a 64-bit Register dst zero-extends the 32-bit load to
          // 64 bits, which matches the ARM64 W-register write semantics.
          case 4: as_.Movl   (tmp, {.base = Assembler::rbp, .disp = off}); break;
          case 8: as_.Movq   (tmp, {.base = Assembler::rbp, .disp = off}); break;
        }
        SetReg(args.rd, tmp);
      }
      return;
    }

    // SMOV (signed move Vn.B/H/S[index] -> Rd). Like UMOV but sign-extending.
    // Canonical (esize, q) pairs per ARM ARM are:
    //   (1, 0) SMOV Wd, Vn.B[i]   sign-ext 8  -> 32, Wd upper zero
    //   (1, 1) SMOV Xd, Vn.B[i]   sign-ext 8  -> 64
    //   (2, 0) SMOV Wd, Vn.H[i]   sign-ext 16 -> 32, Wd upper zero
    //   (2, 1) SMOV Xd, Vn.H[i]   sign-ext 16 -> 64
    //   (4, 1) SMOV Xd, Vn.S[i]   sign-ext 32 -> 64
    // (4, 0) and the doubleword (8, *) variants are unallocated; fall back
    // to the interpreter rather than guess. The 32-bit Movsx*l forms
    // implicitly zero the upper 32 bits of the 64-bit Register (x86_64
    // semantics), which matches AArch64 Wd-write semantics — no separate
    // masking needed.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kSmov && esize != 0) {
      bool canonical = (esize == 1) || (esize == 2) || (esize == 4 && args.q);
      if (!canonical) { success_ = false; return; }
      if (args.rd < 31) {
        Register tmp = AllocTempReg();
        if (tmp == no_register) { success_ = false; return; }
        int32_t off =
            VRegOffset(args.rn) + index * esize;
        if (esize == 1 && !args.q) {
          as_.Movsxbl(tmp, {.base = Assembler::rbp, .disp = off});
        } else if (esize == 1 && args.q) {
          as_.Movsxbq(tmp, {.base = Assembler::rbp, .disp = off});
        } else if (esize == 2 && !args.q) {
          as_.Movsxwl(tmp, {.base = Assembler::rbp, .disp = off});
        } else if (esize == 2 && args.q) {
          as_.Movsxwq(tmp, {.base = Assembler::rbp, .disp = off});
        } else /* esize == 4 && args.q */ {
          as_.Movsxlq(tmp, {.base = Assembler::rbp, .disp = off});
        }
        SetReg(args.rd, tmp);
      }
      return;
    }

    // INS (general): insert Rn into Vd.B/H/S/D[index]. The other lanes of
    // v[rd] are unchanged. We write straight into ThreadState memory at the
    // computed byte offset; the lane width determines mov-width. Same
    // movq-vs-movd discipline as DUP-general below.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kInsGeneral && esize != 0) {
      int32_t off =
          VRegOffset(args.rd) + index * esize;
      Register src = no_register;
      if (args.rn < 31) {
        src = GetReg(args.rn);
      } else {
        // XZR: materialise a zero in a temp and use it as the source.
        src = AllocTempReg();
        if (src == no_register) { success_ = false; return; }
        as_.Xorq(src, src);
      }
      switch (esize) {
        case 1: as_.Movb({.base = Assembler::rbp, .disp = off}, src); break;
        case 2: as_.Movw({.base = Assembler::rbp, .disp = off}, src); break;
        case 4: as_.Movl({.base = Assembler::rbp, .disp = off}, src); break;
        case 8: as_.Movq({.base = Assembler::rbp, .disp = off}, src); break;
      }
      return;
    }

    // INS (element): copy Vn.B/H/S/D[src_index] -> Vd.B/H/S/D[dst_index].
    // dst_index already decoded from imm5 above as `index`; src_index lives
    // in imm4 and must be decoded against the same esize (decoder enforces
    // Q=1 so the full 128-bit Vd is in play).  All other lanes of Vd stay
    // untouched, so we route through a single GP temp: load the source lane
    // from Vn memory, store to the dest lane of Vd memory.  The two-step
    // memory copy is correct for the `rd == rn` self-INS case (e.g.
    // INS V0.S[0], V0.S[3]) because the load completes before the store
    // touches Vd.  Doubleword variant uses Movq/Movq so the upper 32 bits
    // of the element propagate intact (same movq-vs-movd discipline as
    // DUP-general).
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kInsElement && esize != 0) {
      uint8_t src_index;
      switch (esize) {
        case 1: src_index =  args.imm4       & 0xF; break;
        case 2: src_index = (args.imm4 >> 1) & 0x7; break;
        case 4: src_index = (args.imm4 >> 2) & 0x3; break;
        default /* esize == 8 */:
                src_index = (args.imm4 >> 3) & 0x1; break;
      }
      int32_t off_vn =
          VRegOffset(args.rn) + src_index * esize;
      int32_t off_vd =
          VRegOffset(args.rd) + index * esize;
      Register tmp = AllocTempReg();
      if (tmp == no_register) { success_ = false; return; }
      switch (esize) {
        case 1:
          as_.Movzxbl(tmp, {.base = Assembler::rbp, .disp = off_vn});
          as_.Movb({.base = Assembler::rbp, .disp = off_vd}, tmp);
          break;
        case 2:
          as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = off_vn});
          as_.Movw({.base = Assembler::rbp, .disp = off_vd}, tmp);
          break;
        case 4:
          as_.Movl(tmp, {.base = Assembler::rbp, .disp = off_vn});
          as_.Movl({.base = Assembler::rbp, .disp = off_vd}, tmp);
          break;
        case 8:
          as_.Movq(tmp, {.base = Assembler::rbp, .disp = off_vn});
          as_.Movq({.base = Assembler::rbp, .disp = off_vd}, tmp);
          break;
      }
      return;
    }

    // DUP (element): broadcast Vn[index] (one esize-byte element) to all
    // lanes of Vd.  Q=0 fills lower 64 bits and zeros upper 64; Q=1 fills
    // 128 bits.  Valid (esize, Q) pairs per ARM ARM:
    //   (1, 0/1) DUP Vd.8B/.16B, Vn.B[idx]    idx ∈ 0..15
    //   (2, 0/1) DUP Vd.4H/.8H,  Vn.H[idx]    idx ∈ 0..7
    //   (4, 0/1) DUP Vd.2S/.4S,  Vn.S[idx]    idx ∈ 0..3
    //   (8, 1)   DUP Vd.2D,      Vn.D[idx]    idx ∈ 0..1
    // (8, 0) is ARM-reserved.
    //
    // Lowering:
    //   esize=1 → PSHUFB xmm, mask where mask = {idx}×16 → byte broadcast.
    //   esize=2 → PSHUFB xmm, mask where mask = {idx*2, idx*2+1}×8 →
    //             halfword broadcast.
    //   esize=4 → PSHUFD xmm, xmm, imm = idx*0x55 (broadcast dword[idx]).
    //   esize=8 → PSHUFD xmm, xmm, 0x44 (idx=0) or 0xEE (idx=1).
    //
    // Q=0 forms run the same lowering and then zero the upper 64 bits via
    // PSLLDQ-8 / PSRLDQ-8 — for byte/halfword broadcast the result already
    // has the same value in both halves, so the shared upper-zero tail is
    // a no-op semantically; we keep it for uniformity with the rest of
    // the Q=0 SIMD lowerings in this file.
    //
    // PSHUFB is SSSE3 (universal on x86_64 emulator hosts), PSHUFD is
    // SSE2; no host gate needed.
    // DUP (scalar): Vd = zero-extend(Vn.element[index]) — e.g. `mov s1,
    // v0.s[1]`, the scalar lane-extract compilers emit constantly around
    // horizontal reductions. Three byte-shifts: PSRLDQ brings the element to
    // byte 0 (zero-filling from the top, but bytes above the element within
    // the low lane may still hold neighbours), then a PSLLDQ/PSRLDQ pair
    // clears everything above esize. All immediates, no masks, SSE2-only.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kDupScalar && esize != 0) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      int32_t off_vn = VRegOffset(args.rn);
      as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = off_vn});
      const int8_t elem_off = static_cast<int8_t>(index * esize);
      if (elem_off != 0) {
        as_.Psrldq(xmm, elem_off);
      }
      const int8_t clear = static_cast<int8_t>(16 - esize);
      as_.Pslldq(xmm, clear);
      as_.Psrldq(xmm, clear);
      int32_t off_vd = VRegOffset(args.rd);
      as_.Movdqu({.base = Assembler::rbp, .disp = off_vd}, xmm);
      return;
    }

    if (args.opcode == Decoder::AdvSimdCopyOpcode::kDupElement && esize != 0) {
      if (esize == 8 && !args.q) { success_ = false; return; }

      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      int32_t off_vn = VRegOffset(args.rn);
      as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = off_vn});

      if (esize == 1) {
        // mask = 0x{idx}{idx}{idx}{idx}{idx}{idx}{idx}{idx} (each byte = idx).
        SimdRegister mask = AllocTempSimdReg();
        if (!success()) { return; }
        Register r1 = AllocTempReg();
        if (r1 == no_register) { success_ = false; return; }
        uint64_t mask_qword =
            0x0101010101010101ULL * static_cast<uint64_t>(index);
        as_.Movq(r1, static_cast<int64_t>(mask_qword));
        as_.Movq(mask, r1);
        as_.Pinsrq(mask, r1, int8_t{1});
        as_.Pshufb(xmm, mask);
      } else if (esize == 2) {
        // mask qword = {b0,b1,b0,b1,b0,b1,b0,b1} where b0=idx*2, b1=idx*2+1.
        SimdRegister mask = AllocTempSimdReg();
        if (!success()) { return; }
        Register r1 = AllocTempReg();
        if (r1 == no_register) { success_ = false; return; }
        uint64_t b0 = static_cast<uint64_t>(index) * 2;
        uint64_t b1 = b0 + 1;
        uint64_t pair = (b1 << 8) | b0;
        uint64_t mask_qword =
            pair | (pair << 16) | (pair << 32) | (pair << 48);
        as_.Movq(r1, static_cast<int64_t>(mask_qword));
        as_.Movq(mask, r1);
        as_.Pinsrq(mask, r1, int8_t{1});
        as_.Pshufb(xmm, mask);
      } else if (esize == 4) {
        // PSHUFD imm = idx * 0x55: imm 0x00, 0x55, 0xAA, 0xFF broadcast
        // dword[0], [1], [2], [3] respectively to all 4 dwords.
        uint8_t imm = static_cast<uint8_t>(index * 0x55);
        as_.Pshufd(xmm, xmm, static_cast<int8_t>(imm));
      } else /* esize == 8 (Q=1 only; Q=0 rejected above) */ {
        // PSHUFD imm 0x44 = (0)(1)(0)(1) — duplicate low qword to both
        // halves.  imm 0xEE = (2)(3)(2)(3) — duplicate high qword.
        uint8_t imm = (index == 0) ? 0x44 : 0xEE;
        as_.Pshufd(xmm, xmm, static_cast<int8_t>(imm));
      }

      if (!args.q) {
        // Zero upper 64 bits per ARM D-register semantics.
        as_.Pslldq(xmm, int8_t{8});
        as_.Psrldq(xmm, int8_t{8});
      }
      int32_t off_vd = VRegOffset(args.rd);
      as_.Movdqu({.base = Assembler::rbp, .disp = off_vd}, xmm);
      return;
    }

    // implement DUP (general) for memset fast path
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kDupGeneral) {
      // DUP (general): broadcast GP register Rn to all lanes of Vd.
      //   Q=1: full 128-bit broadcast (16B / 4S / 2D).
      //   Q=0: low 64 bits = broadcast (8B / 2S), upper 64 bits zeroed per
      //        ARM D-register semantics.  The D-lane Q=0 form (1D) is
      //        ARM-reserved and bails via success_=false.
      // imm5 encodes element size: bit0=1→B, bit1=1→H, bit2=1→W, bit3=1→X.
      uint8_t esize_bits = args.imm5 & 0xf;
      if (esize_bits == 0x08 && !args.q) { success_ = false; return; }
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Register src = no_register;
      if (args.rn < 31) {
        src = GetReg(args.rn);
      } else {
        // XZR/WZR source: DUP broadcasts zero, so the whole result vector is
        // zero. Materialise a zero in a temp; the Movq/Movd + broadcast below
        // then produce an all-zero Vd. `dup v0.2d, xzr` is a common compiler
        // idiom to zero a vector register (seen in libcronet on NetEase).
        src = AllocTempReg();
        if (src == no_register) { success_ = false; return; }
        as_.Xorq(src, src);
      }
      // For 64-bit broadcast we must move the FULL 64 bits of the GP register
      // into the XMM register; using 32-bit MOVD here silently truncates the
      // upper half and the subsequent PSHUFD(0x44) then duplicates the low
      // 32-bit value into both D-lanes. That was the FB libcoldstart Yoga
      // layout x21 truncation bug: DUP V0.2D, X21 produced
      // V0 = {lo32(x21), lo32(x21)} instead of {x21, x21}, so when the
      // surrounding INS/UMOV spill cycle later reloaded X21 from V0.D[1]
      // it got the 32-bit-truncated pointer and the next post-indexed STR
      // faulted at the low-address.
      if (esize_bits == 0x08) {
        as_.Movq(xmm, src);
      } else {
        as_.Movd(xmm, src);
      }
      if (esize_bits == 0x01) {
        // Byte broadcast: PSHUFB with zero mask → each byte picks byte 0
        SimdRegister zero_mask = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pxor(zero_mask, zero_mask);
        as_.Pshufb(xmm, zero_mask);
      } else if (esize_bits == 0x02) {
        // Halfword (16-bit) broadcast: Movd above placed Rn's low 32 bits in
        // xmm[31:0]; the source halfword sits in xmm[15:0].  PSHUFLW with
        // imm=0 broadcasts halfword[0] to all 4 low halfwords (xmm[63:0]).
        // PSHUFD imm=0x44 then copies the low qword into the upper qword for
        // Q=1 (.8H); for Q=0 (.4H) the upper-zero tail below clears the
        // upper half.  Mirrors the proven UMOV-spill broadcast at line 6319.
        as_.Pshuflw(xmm, xmm, static_cast<int8_t>(0));
        if (args.q) as_.Pshufd(xmm, xmm, static_cast<int8_t>(0x44));
      } else if (esize_bits == 0x04) {
        // Word (32-bit) broadcast: PSHUFD(0) → broadcast low dword to all 4
        as_.Pshufd(xmm, xmm, static_cast<int8_t>(0));
      } else if (esize_bits == 0x08) {
        // Doubleword (64-bit) broadcast: PSHUFD(0x44) → low 64 bits to both halves
        as_.Pshufd(xmm, xmm, static_cast<int8_t>(0x44));
      } else {
        Undefined();
        return;
      }
      if (!args.q) {
        // Zero upper 64 bits per ARM D-register semantics (8B / 2S forms).
        // For byte broadcast the upper half already holds the same byte;
        // the shuttle is still needed to zero it.  For word broadcast the
        // upper half holds two copies of the dword and must be zeroed.
        as_.Pslldq(xmm, int8_t{8});
        as_.Psrldq(xmm, int8_t{8});
      }
      int32_t vreg_offset = VRegOffset(args.rd);
      as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      return;
    }
    Undefined();
  }

