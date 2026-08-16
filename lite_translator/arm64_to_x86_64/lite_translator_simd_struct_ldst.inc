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
// Family: SIMD multi/single structure load/store (LD1/ST1 ...).
// Methods:
//   AdvSimdMultiStruct, AdvSimdSingleStruct

  // JIT for LD1 / ST1 multi-register contiguous AdvSIMD load/store
  // (`ld1 {Vt.T, ...}, [Xn]{, post}` / `st1 ...`). Promoted out of the
  // interpreter because `__dl___strchr_aarch64` in the dynamic linker
  // uses `ld1 {v1.16b, v2.16b}, [x2], #32` in its hot byte-scan loop —
  // every dlopen / symbol resolve bailed to the interpreter on this
  // instruction. De-interleaving variants (LD2/LD3/LD4 / ST2/ST3/ST4)
  // still fall back to the interpreter (lane shuffling is non-trivial
  // and was not surfaced as hot by tracing).
  void AdvSimdMultiStruct(uint8_t rt, uint8_t rn, uint8_t num_regs, uint8_t size,
                          bool q, bool is_store, bool postindex, uint8_t rm,
                          bool is_interleaved) {
    UNUSED(size);  // For LD1/ST1 contiguous, the element size only matters
                   // for post-index immediate math, and that is encoded by Q
                   // (16- or 8-byte stride per register) — the transfer is
                   // bulk-vector regardless.
    const int32_t vec_bytes = q ? 16 : 8;

    // de-interleaving LD2/LD3/LD4 / interleaving ST2/ST3/ST4
    // (multiple-structure form).  Memory holds num_regs * (vec_bytes/esize)
    // elements laid out structure-major: element index e in memory belongs to
    // register (e % num_regs), lane (e / num_regs).  The universal lowering
    // moves one element per PINSR (load) / PEXTR (store) directly between the
    // computed memory operand and the destination/source lane — exact for every
    // (num_regs, element-size, Q) combination, with no interpreter bailout.
    // Q=0 zeroes the upper 64 bits of each loaded register via the initial
    // PXOR.  Shuffle-based fast paths could replace the per-lane round-trip for
    // the hot sizes, but the element-wise form keeps the whole class on the JIT.
    if (is_interleaved) {
      if (num_regs < 2 || num_regs > 4) { Undefined(); return; }
      if (size > 3) { Undefined(); return; }
      const int esize = 1 << size;
      const int num_lanes = vec_bytes / esize;

      Register ibase_orig = (rn == 31) ? GetSp() : GetReg(rn);
      if (ibase_orig == no_register) { Undefined(); return; }
      Register ibase = ApplyTbi(ibase_orig);
      if (ibase == no_register) { Undefined(); return; }

      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }

      for (uint8_t r = 0; r < num_regs; r++) {
        const uint8_t vreg = (rt + r) & 31;
        const int32_t vt_off = VRegOffset(vreg);
        if (is_store) {
          as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = vt_off});
        } else {
          as_.Pxor(xmm, xmm);
        }
        for (int l = 0; l < num_lanes; l++) {
          const int32_t mem_off = (l * num_regs + r) * esize;
          Assembler::Operand mem{.base = ibase, .disp = mem_off};
          const int8_t lane = static_cast<int8_t>(l);
          if (is_store) {
            switch (esize) {
              case 1: as_.Pextrb(mem, xmm, lane); break;
              case 2: as_.Pextrw(mem, xmm, lane); break;
              case 4: as_.Pextrd(mem, xmm, lane); break;
              default: as_.Pextrq(mem, xmm, lane); break;  // esize == 8
            }
          } else {
            switch (esize) {
              case 1: as_.Pinsrb(xmm, mem, lane); break;
              case 2: as_.Pinsrw(xmm, mem, lane); break;
              case 4: as_.Pinsrd(xmm, mem, lane); break;
              default: as_.Pinsrq(xmm, mem, lane); break;  // esize == 8
            }
          }
        }
        if (!is_store) {
          as_.Movdqu({.base = Assembler::rbp, .disp = vt_off}, xmm);
        }
      }

      if (postindex) {
        Register new_base = AllocTempReg();
        if (new_base == no_register) { Undefined(); return; }
        Register reread_base = (rn == 31) ? GetSp() : GetReg(rn);
        as_.Movq(new_base, reread_base);
        if (rm == 31) {
          as_.Addq(new_base, static_cast<int32_t>(num_regs) * vec_bytes);
        } else {
          Register rm_val = GetReg(rm);
          if (rm_val == no_register) { Undefined(); return; }
          as_.Addq(new_base, rm_val);
        }
        if (rn == 31) {
          SetSp(new_base);
        } else {
          SetReg(rn, new_base);
        }
      }
      return;
    }

    if (num_regs < 1 || num_regs > 4) { Undefined(); return; }

    Register base_orig = (rn == 31) ? GetSp() : GetReg(rn);
    if (base_orig == no_register) { Undefined(); return; }
    Register base = ApplyTbi(base_orig);
    if (base == no_register) { Undefined(); return; }

    SimdRegister xmm = AllocTempSimdReg();
    if (!success()) { return; }

    for (uint8_t r = 0; r < num_regs; r++) {
      const uint8_t vreg = (rt + r) & 31;
      const int32_t vt_off = VRegOffset(vreg);
      const int32_t mem_off = static_cast<int32_t>(r) * vec_bytes;
      Assembler::Operand mem{.base = base, .disp = mem_off};

      if (is_store) {
        if (q) {
          as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = vt_off});
          as_.Movdqu(mem, xmm);
        } else {
          // Q=0: store low 64 bits of v[vreg].
          as_.Movq(xmm, {.base = Assembler::rbp, .disp = vt_off});
          as_.Movq(mem, xmm);
        }
      } else {
        if (q) {
          as_.Movdqu(xmm, mem);
          as_.Movdqu({.base = Assembler::rbp, .disp = vt_off}, xmm);
        } else {
          // Q=0: load 8 bytes and zero-extend the upper 64 of v[vreg].
          as_.Pxor(xmm, xmm);
          as_.Movq(xmm, mem);
          as_.Movdqu({.base = Assembler::rbp, .disp = vt_off}, xmm);
        }
      }
    }

    if (postindex) {
      Register new_base = AllocTempReg();
      if (new_base == no_register) { Undefined(); return; }
      Register reread_base = (rn == 31) ? GetSp() : GetReg(rn);
      as_.Movq(new_base, reread_base);
      if (rm == 31) {
        // Immediate post-index: total bytes accessed = num_regs * vec_bytes.
        as_.Addq(new_base, static_cast<int32_t>(num_regs) * vec_bytes);
      } else {
        Register rm_val = GetReg(rm);
        if (rm_val == no_register) { Undefined(); return; }
        as_.Addq(new_base, rm_val);
      }
      if (rn == 31) {
        SetSp(new_base);
      } else {
        SetReg(rn, new_base);
      }
    }
  }

  void AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs& args) {
    // JIT for LD1R / LD1 / ST1 single-element variants
    // with num_regs == 1.  Critical for `calculate_gnu_hash_neon`'s tail
    // (`ld1r v3.4s, [x10], #4`) in the dynamic linker — without this the
    // post-loop tail bails to the interpreter on every symbol resolve.
    using Op = Decoder::AdvSimdSingleStructOp;
    const bool is_replicate = args.is_replicate;
    const bool is_single_store = (args.op == Op::kSt1 || args.op == Op::kSt2 ||
                                  args.op == Op::kSt3 || args.op == Op::kSt4);
    if (args.num_regs < 1 || args.num_regs > 4) { Undefined(); return; }

    // size: 00=B(1), 01=H(2), 10=S(4), 11=D(8).
    const uint8_t esize = static_cast<uint8_t>(1u << args.size);
    if (esize != 1 && esize != 2 && esize != 4 && esize != 8) {
      Undefined(); return;
    }


    // Compute base address (with TBI mask).
    Register base_orig = (args.rn == 31) ? GetSp() : GetReg(args.rn);
    if (base_orig == no_register) { Undefined(); return; }
    Register base = ApplyTbi(base_orig);
    if (base == no_register) { Undefined(); return; }


    if (is_single_store) {
      // ST1-ST4 lane: for each of the num_regs consecutive registers (mod 32),
      // load its element from v[reg].lane[index] (ThreadState — no fault), then
      // store to guest memory at consecutive element offsets. A fault on any
      // element exits to the interpreter at this instruction; the earlier
      // partial stores are architecturally permitted (re-execution re-stores).
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }
      for (unsigned r = 0; r < args.num_regs; ++r) {
        const uint8_t reg = static_cast<uint8_t>((args.rt + r) & 31);
        const int32_t lane_off = VRegOffset(reg) + args.index * esize;
        switch (esize) {
          case 1: as_.Movzxbl(tmp, {.base = Assembler::rbp, .disp = lane_off}); break;
          case 2: as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = lane_off}); break;
          case 4: as_.Movl(tmp, {.base = Assembler::rbp, .disp = lane_off}); break;
          case 8: as_.Movq(tmp, {.base = Assembler::rbp, .disp = lane_off}); break;
        }
        Assembler::Operand elem_mem{.base = base, .disp = static_cast<int32_t>(r * esize)};
        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        as_.SetRecoveryPoint(recovery_label);
        switch (esize) {
          case 1: as_.Movb(elem_mem, tmp); break;
          case 2: as_.Movw(elem_mem, tmp); break;
          case 4: as_.Movl(elem_mem, tmp); break;
          case 8: as_.Movq(elem_mem, tmp); break;
        }
        AssemblerBase::Label* cont = as_.MakeLabel();
        as_.Jmp(*cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(cont);
      }
    } else if (is_replicate) {
      // LD1R-LD4R: for each of the num_regs consecutive registers (mod 32),
      // load its own esize-byte element from consecutive memory offsets,
      // broadcast it to all lanes, zero the upper 64 bits if Q=0.
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }
      SimdRegister zero_mask = no_simd_register;
      if (esize == 1) {
        zero_mask = AllocTempSimdReg();
        if (zero_mask == no_simd_register) { Undefined(); return; }
      }
      for (unsigned r = 0; r < args.num_regs; ++r) {
        const uint8_t reg = static_cast<uint8_t>((args.rt + r) & 31);
        Assembler::Operand elem_mem{.base = base, .disp = static_cast<int32_t>(r * esize)};
        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        as_.SetRecoveryPoint(recovery_label);
        switch (esize) {
          case 1: as_.Movzxbl(tmp, elem_mem); break;
          case 2: as_.Movzxwl(tmp, elem_mem); break;
          case 4: as_.Movl(tmp, elem_mem); break;
          case 8: as_.Movq(tmp, elem_mem); break;
        }
        AssemblerBase::Label* cont = as_.MakeLabel();
        as_.Jmp(*cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(cont);

        // Broadcast `tmp` across XMM lanes.
        switch (esize) {
          case 1:
            as_.Movd(xmm, tmp);
            as_.Pxor(zero_mask, zero_mask);
            as_.Pshufb(xmm, zero_mask);  // broadcast byte 0 to all 16 bytes
            break;
          case 2:
            as_.Movd(xmm, tmp);
            // Broadcast 16-bit element to all 4 low words, then to upper if Q=1.
            as_.Pshuflw(xmm, xmm, static_cast<int8_t>(0));
            if (args.q) as_.Pshufd(xmm, xmm, static_cast<int8_t>(0x44));
            break;
          case 4:
            as_.Movd(xmm, tmp);
            as_.Pshufd(xmm, xmm, static_cast<int8_t>(0));
            break;
          case 8:
            as_.Movq(xmm, tmp);
            if (args.q) as_.Pshufd(xmm, xmm, static_cast<int8_t>(0x44));
            break;
        }
        if (!args.q) {
          as_.Pslldq(xmm, int8_t{8});
          as_.Psrldq(xmm, int8_t{8});
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = VRegOffset(reg)}, xmm);
      }
    } else {
      // LD1-LD4 single-lane: for each of the num_regs consecutive registers
      // (mod 32), load esize bytes from consecutive memory offsets into
      // v[reg].lane[index], preserving other lanes by writing directly to
      // ThreadState at the lane offset. A fault on any element exits to the
      // interpreter at this instruction; earlier partial lane writes are
      // architecturally permitted (re-execution re-loads them).
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }
      for (unsigned r = 0; r < args.num_regs; ++r) {
        const uint8_t reg = static_cast<uint8_t>((args.rt + r) & 31);
        Assembler::Operand elem_mem{.base = base, .disp = static_cast<int32_t>(r * esize)};
        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        as_.SetRecoveryPoint(recovery_label);
        switch (esize) {
          case 1: as_.Movzxbl(tmp, elem_mem); break;
          case 2: as_.Movzxwl(tmp, elem_mem); break;
          case 4: as_.Movl(tmp, elem_mem); break;
          case 8: as_.Movq(tmp, elem_mem); break;
        }
        AssemblerBase::Label* cont = as_.MakeLabel();
        as_.Jmp(*cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(cont);

        const int32_t lane_off = VRegOffset(reg) + args.index * esize;
        switch (esize) {
          case 1: as_.Movb({.base = Assembler::rbp, .disp = lane_off}, tmp); break;
          case 2: as_.Movw({.base = Assembler::rbp, .disp = lane_off}, tmp); break;
          case 4: as_.Movl({.base = Assembler::rbp, .disp = lane_off}, tmp); break;
          case 8: as_.Movq({.base = Assembler::rbp, .disp = lane_off}, tmp); break;
        }
      }
    }

    // Post-index update of Xn (or SP).
    if (args.postindex) {
      Register new_base = AllocTempReg();
      if (new_base == no_register) { Undefined(); return; }
      Register reread_base = (args.rn == 31) ? GetSp() : GetReg(args.rn);
      as_.Movq(new_base, reread_base);
      if (args.rm == 31) {
        // Immediate post-index: total bytes accessed == num_regs * esize.
        int32_t imm = static_cast<int32_t>(args.num_regs) * esize;
        as_.Addq(new_base, imm);
      } else {
        Register rm_val = GetReg(args.rm);
        if (rm_val == no_register) { Undefined(); return; }
        as_.Addq(new_base, rm_val);
      }
      if (args.rn == 31) {
        SetSp(new_base);
      } else {
        SetReg(args.rn, new_base);
      }
    }
  }

