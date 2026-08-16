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
// Family: integer carry, one-source, extract, cond-compare, exclusive/atomic.
// Methods:
//   AddSubWithCarry, DataProc1Src, Extr, ConditionalCompare,
//   LoadStoreExclusive

  // ADC/ADCS/SBC/SBCS (add/subtract with the guest carry flag).
  //   ADC: res = src1 + src2 + C
  //   SBC: res = src1 + ~src2 + C  (ARM subtract-with-carry)
  // The guest carry (ARM NZCV C = cpu.flags bit 8) feeds x86 CF, then one x86
  // ADC/SBB does the carry op:
  //   ADC — ARM C maps directly to x86 CF, so load CF from bit 8 and ADC.
  //   SBC — x86 SBB computes src1 - src2 - CF = src1 + ~src2 + (1 - CF), while
  //         ARM SBC wants src1 + ~src2 + C, so SBB needs CF = !C: complement
  //         the loaded CF with CMC. On output x86 SBB's CF is the borrow, and
  //         EmitStoreArmNZCV(is_sub=true) already inverts it back to ARM's
  //         "carry = no borrow" convention.
  // res is a fresh temp (never aliases the mapped rn/rm), and MOV does not
  // touch FLAGS, so the carry loaded by Btw survives up to the ADC/SBB.
  Register AddSubWithCarry(Register src1, Register src2, bool is_64bit,
                            bool is_sub, bool set_flags) {
    Register res = AllocTempReg();
    if (!success()) return no_register;
    // MOV first (no FLAGS effect), then load the guest carry into x86 CF.
    if (is_64bit) {
      as_.Movq(res, src1);
    } else {
      as_.Movl(res, src1);
    }
    const int32_t flags_offset = offsetof(ThreadState, cpu.flags);
    as_.Btw({.base = Assembler::rbp, .disp = flags_offset}, static_cast<int8_t>(8));
    if (is_sub) {
      as_.Cmc();  // SBB wants CF = !(ARM carry).
    }
    if (is_64bit) {
      if (is_sub) {
        as_.Sbbq(res, src2);
      } else {
        as_.Adcq(res, src2);
      }
    } else {
      if (is_sub) {
        as_.Sbbl(res, src2);
      } else {
        as_.Adcl(res, src2);
      }
    }
    if (set_flags) {
      EmitStoreArmNZCV(is_sub);
    }
    return res;
  }

  Register DataProc1Src(Register src, uint8_t opcode2, bool is_64bit) {
    // JIT support for REV, CLZ, RBIT
    Register res = AllocTempReg();
    // PAuth DP-1Src as identity (emit a plain move)
    // The decoder sets bit 0x40 to flag PAuth variants — Digitalis is PAC-blind,
    // so the JIT just copies src→dst (the upper-half clear of Movl handles the
    // sf=0 sign/zero-extend semantics; PAuth ops are X-form only but Movl is
    // still safe because sf=1 always reaches the Movq branch).
    if (opcode2 & 0x40) {
      if (is_64bit) {
        as_.Movq(res, src);
      } else {
        as_.Movl(res, src);
      }
      return res;
    }
    switch (opcode2) {
      case 0b000001: {  // REV16: reverse byte order within each 16-bit halfword.
        // res = ((src & 0x..00FF) << 8) | ((src & 0x..FF00) >> 8).
        Register tmp = AllocTempReg();
        if (tmp == no_register) { success_ = false; return no_register; }
        if (is_64bit) {
          Register mask = AllocTempReg();
          if (mask == no_register) { success_ = false; return no_register; }
          as_.Movq(res, src);
          as_.Movq(tmp, src);
          as_.Movq(mask, int64_t{0x00FF00FF00FF00FFLL});
          as_.Andq(res, mask);
          as_.Shlq(res, int8_t{8});
          as_.Movq(mask, static_cast<int64_t>(0xFF00FF00FF00FF00ULL));
          as_.Andq(tmp, mask);
          as_.Shrq(tmp, int8_t{8});
          as_.Orq(res, tmp);
        } else {
          as_.Movl(res, src);
          as_.Movl(tmp, src);
          as_.Andl(res, 0x00FF00FF);
          as_.Shll(res, int8_t{8});
          as_.Andl(tmp, static_cast<int32_t>(0xFF00FF00));
          as_.Shrl(tmp, int8_t{8});
          as_.Orl(res, tmp);
        }
        return res;
      }
      case 0b000010:  // REV32 (sf=1) / REV (sf=0).
        if (is_64bit) {
          // REV32 Xd: byte-reverse each 32-bit word independently. BSWAPQ
          // reverses all 8 bytes (also swapping the two words); rotating by 32
          // swaps the words back, leaving each word byte-reversed in place.
          as_.Movq(res, src);
          as_.Bswapq(res);
          as_.Rorq(res, int8_t{32});
        } else {
          // REV Wd: byte-reverse the 32-bit word (zero-extends).
          as_.Movl(res, src);
          as_.Bswapl(res);
        }
        return res;
      case 0b000011:  // REV (byte reverse) — maps to x86 BSWAP
        if (is_64bit) {
          as_.Movq(res, src);
          as_.Bswapq(res);
        } else {
          as_.Movl(res, src);
          as_.Bswapl(res);
        }
        return res;
      case 0b000100: {  // CLZ (count leading zeros) — maps to x86 LZCNT or BSR
        // Use BSR (bit scan reverse) to find highest set bit.
        // CLZ = (regsize - 1) - BSR, or regsize if input is 0.
        Assembler::Label* nonzero = as_.MakeLabel();
        Assembler::Label* done_label = as_.MakeLabel();
        if (is_64bit) {
          as_.Testq(src, src);
          as_.Jcc(Condition::kNotZero, *nonzero);
          as_.Movq(res, static_cast<int64_t>(64));
          as_.Jmp(*done_label);
          as_.Bind(nonzero);
          as_.Bsrq(res, src);
          as_.Xorq(res, static_cast<int8_t>(63));  // CLZ = 63 - BSR
        } else {
          as_.Testl(src, src);
          as_.Jcc(Condition::kNotZero, *nonzero);
          as_.Movl(res, static_cast<int32_t>(32));
          as_.Jmp(*done_label);
          as_.Bind(nonzero);
          as_.Bsrl(res, src);
          as_.Xorl(res, static_cast<int8_t>(31));  // CLZ = 31 - BSR
        }
        as_.Bind(done_label);
        return res;
      }
      case 0b000000: {  // RBIT: reverse bit order.
        // SWAR bit-reverse: swap adjacent bits, then bit-pairs, then nibbles,
        // then reverse byte order with BSWAP. Needs one scratch register.
        Register tmp = AllocTempReg();
        if (tmp == no_register) {
          success_ = false;
          return no_register;
        }
        if (is_64bit) {
          Register mask = AllocTempReg();
          if (mask == no_register) {
            success_ = false;
            return no_register;
          }
          as_.Movq(res, src);
          // res = ((res >> 1) & 0x5555...) | ((res & 0x5555...) << 1)
          as_.Movq(tmp, res);
          as_.Shrq(tmp, int8_t{1});
          as_.Movq(mask, static_cast<int64_t>(0x5555555555555555LL));
          as_.Andq(tmp, mask);
          as_.Andq(res, mask);
          as_.Shlq(res, int8_t{1});
          as_.Orq(res, tmp);
          // swap bit-pairs with 0x3333...
          as_.Movq(tmp, res);
          as_.Shrq(tmp, int8_t{2});
          as_.Movq(mask, static_cast<int64_t>(0x3333333333333333LL));
          as_.Andq(tmp, mask);
          as_.Andq(res, mask);
          as_.Shlq(res, int8_t{2});
          as_.Orq(res, tmp);
          // swap nibbles with 0x0F0F...
          as_.Movq(tmp, res);
          as_.Shrq(tmp, int8_t{4});
          as_.Movq(mask, static_cast<int64_t>(0x0F0F0F0F0F0F0F0FLL));
          as_.Andq(tmp, mask);
          as_.Andq(res, mask);
          as_.Shlq(res, int8_t{4});
          as_.Orq(res, tmp);
          // reverse byte order.
          as_.Bswapq(res);
        } else {
          as_.Movl(res, src);
          as_.Movl(tmp, res);
          as_.Shrl(tmp, int8_t{1});
          as_.Andl(tmp, static_cast<int32_t>(0x55555555));
          as_.Andl(res, static_cast<int32_t>(0x55555555));
          as_.Shll(res, int8_t{1});
          as_.Orl(res, tmp);
          as_.Movl(tmp, res);
          as_.Shrl(tmp, int8_t{2});
          as_.Andl(tmp, static_cast<int32_t>(0x33333333));
          as_.Andl(res, static_cast<int32_t>(0x33333333));
          as_.Shll(res, int8_t{2});
          as_.Orl(res, tmp);
          as_.Movl(tmp, res);
          as_.Shrl(tmp, int8_t{4});
          as_.Andl(tmp, static_cast<int32_t>(0x0F0F0F0F));
          as_.Andl(res, static_cast<int32_t>(0x0F0F0F0F));
          as_.Shll(res, int8_t{4});
          as_.Orl(res, tmp);
          as_.Bswapl(res);
        }
        return res;
      }
      default:
        Undefined();
        return no_register;
    }
  }

  // EXTR JIT
  Register Extr(Register src_n, Register src_m, uint8_t lsb, bool is_64bit) {
    // EXTR Xd, Xn, Xm, #lsb: extract from pair (Xn:Xm) >> lsb
    // When Xn == Xm, this is a rotate right (ROR).
    Register res = AllocTempReg();
    if (!success()) return no_register;

    if (lsb == 0) {
      // EXTR with lsb=0 is just a copy of Xm.
      if (is_64bit) {
        as_.Movq(res, src_m);
      } else {
        as_.Movl(res, src_m);
      }
      return res;
    }

    // Use SHRD: shifts (res:src_n) right by lsb bits into res.
    // SHRD dest, src, imm8: dest = (src:dest) >> imm8
    // ARM64 EXTR: Rd = (Xn:Xm) >> lsb = SHRD(Xm, Xn, lsb)
    if (is_64bit) {
      as_.Movq(res, src_m);
      as_.Shrdq(res, src_n, static_cast<int8_t>(lsb));
    } else {
      as_.Movl(res, src_m);
      as_.Shrdl(res, src_n, static_cast<int8_t>(lsb));
    }
    return res;
  }

  void ConditionalCompare(bool is_neg, bool is_64bit, Register rn, Register rm,
                           Decoder::Condition cond, uint8_t nzcv);  // implemented in .cc

  // Atomics JIT
  void LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& args, Register base) {
    // apply TBI mask before using base as memory operand.
    base = ApplyTbi(base);
    auto lss = static_cast<Decoder::LoadStoreSize>(args.size);
    Assembler::Operand mem{.base = base, .disp = 0};

    switch (args.op) {
      case Decoder::AtomicOp::kLdar: {
        // Load-acquire: x86 TSO provides acquire semantics for all loads.
        Register res = Load(lss, /*is_signed=*/false, /*is_64bit_target=*/true, base, 0);
        if (!success()) return;
        if (args.rt < 31) SetReg(args.rt, res);
        break;
      }

      case Decoder::AtomicOp::kStlr: {
        // Store-release: x86 TSO gives release ordering for free, but ARM STLR is
        // sequentially consistent (RCsc) — it also orders the store before any
        // later load. x86 permits StoreLoad reordering, so emit MFENCE after the
        // store to recover that ordering (matches the standard seq-cst-store
        // lowering). Without it, seq_cst guest code races flakily.
        Register data = (args.rt < 31) ? GetReg(args.rt) : AllocTempReg();
        if (!success()) return;
        if (args.rt >= 31) as_.Xorl(data, data);
        Store(lss, base, 0, data);
        as_.Mfence();
        break;
      }

      case Decoder::AtomicOp::kCas: {
        // CAS Xs, Xt, [Xn]: compare [Xn] with Xs, if equal store Xt to [Xn].
        // Old value of [Xn] written to Xs.
        // x86 LOCK CMPXCHG: compares RAX with [mem], if equal stores src to [mem].
        // Old value goes to RAX.
        Register expected = (args.rs < 31) ? GetReg(args.rs) : AllocTempReg();
        if (!success()) return;
        if (args.rs >= 31) as_.Xorl(expected, expected);

        Register desired = (args.rt < 31) ? GetReg(args.rt) : AllocTempReg();
        if (!success()) return;
        if (args.rt >= 31) as_.Xorl(desired, desired);

        // Move expected into RAX (CMPXCHG uses RAX implicitly).
        if (args.rs < 31) {
          as_.Movq(Assembler::rax, expected);
        } else {
          as_.Xorl(Assembler::rax, Assembler::rax);
        }

        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        as_.SetRecoveryPoint(recovery_label);

        switch (args.size) {
          case 0: as_.LockCmpXchgb(mem, desired); break;
          case 1: as_.LockCmpXchgw(mem, desired); break;
          case 2: as_.LockCmpXchgl(mem, desired); break;
          case 3: as_.LockCmpXchgq(mem, desired); break;
        }

        AssemblerBase::Label* cont = as_.MakeLabel();
        as_.Jmp(*cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(cont);

        // Write old value back to Rs (from RAX).
        if (args.rs < 31) {
          Register old_val = AllocTempReg();
          if (!success()) return;
          as_.Movq(old_val, Assembler::rax);
          // Sub-64-bit CAS must zero-extend the old value into Rs. On a CMPXCHG
          // *match* the destination accumulator (AL/AX/EAX) is left UNWRITTEN, so
          // RAX keeps the expected operand's stale upper bits (we loaded the full
          // 64-bit Rs into RAX). Byte/halfword are fixed by the AND masks; the
          // 32-bit (W) form must also be re-zero-extended — a 32-bit MOV clears
          // bits 63:32. Without this, CAS Ws returns a non-zero-extended old
          // value (e.g. a 32-bit std::atomic compare-exchange sees garbage upper
          // bits), which corrupts lock-free C++ structures such as React Native /
          // Fabric's shared_ptr-heavy view tree (renders blank under translation).
          if (args.size == 0) {
            as_.Andq(old_val, static_cast<int32_t>(0xFF));
          } else if (args.size == 1) {
            as_.Andq(old_val, static_cast<int32_t>(0xFFFF));
          } else if (args.size == 2) {
            as_.Movl(old_val, old_val);
          }
          SetReg(args.rs, old_val);
        }
        break;
      }

      case Decoder::AtomicOp::kSwp: {
        // SWP Xs, Xt, [Xn]: atomically swap [Xn] with Xs, old value to Xt.
        // x86 XCHG with memory has implicit LOCK prefix.
        Register new_val = AllocTempReg();
        if (!success()) return;
        if (args.rs < 31) {
          as_.Movq(new_val, GetReg(args.rs));
        } else {
          as_.Xorl(new_val, new_val);
        }

        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        as_.SetRecoveryPoint(recovery_label);

        switch (args.size) {
          case 0: as_.Xchgb(new_val, mem); break;
          case 1: as_.Xchgw(new_val, mem); break;
          case 2: as_.Xchgl(new_val, mem); break;
          case 3: as_.Xchgq(new_val, mem); break;
        }

        AssemblerBase::Label* cont = as_.MakeLabel();
        as_.Jmp(*cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(cont);

        // byte/halfword Xchg leaves upper bits of new_val
        // as the original guest Xs (copied via Movq above), not zero. ARM SWP
        // Wt zero-extends the old memory value to 64. Mask.
        if (args.size == 0) {
          as_.Andq(new_val, static_cast<int32_t>(0xFF));
        } else if (args.size == 1) {
          as_.Andq(new_val, static_cast<int32_t>(0xFFFF));
        }
        if (args.rt < 31) SetReg(args.rt, new_val);
        break;
      }

      case Decoder::AtomicOp::kLdadd: {
        // LDADD Xs, Xt, [Xn]: atomically add Xs to [Xn], old value to Xt.
        // x86 LOCK XADD: adds src to [mem], old value goes to src register.
        Register addend = AllocTempReg();
        if (!success()) return;
        if (args.rs < 31) {
          as_.Movq(addend, GetReg(args.rs));
        } else {
          as_.Xorl(addend, addend);
        }

        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        as_.SetRecoveryPoint(recovery_label);

        switch (args.size) {
          case 0: as_.LockXaddb(mem, addend); break;
          case 1: as_.LockXaddw(mem, addend); break;
          case 2: as_.LockXaddl(mem, addend); break;
          case 3: as_.LockXaddq(mem, addend); break;
        }

        AssemblerBase::Label* cont = as_.MakeLabel();
        as_.Jmp(*cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(cont);

        // byte/halfword LockXadd only updates low bits of
        // addend; upper bits stay as guest Xs. ARM LDADD Wt zero-extends the
        // old memory value to 64. Mask.
        if (args.size == 0) {
          as_.Andq(addend, static_cast<int32_t>(0xFF));
        } else if (args.size == 1) {
          as_.Andq(addend, static_cast<int32_t>(0xFFFF));
        }
        // addend now contains old value.
        if (args.rt < 31) SetReg(args.rt, addend);
        break;
      }

      case Decoder::AtomicOp::kLdxr: {
        // Load-exclusive: load value and set reservation in ThreadState.
        Register res = Load(lss, /*is_signed=*/false, /*is_64bit_target=*/true, base, 0);
        if (!success()) return;

        // Store reservation address.
        int32_t resv_addr_off = offsetof(ThreadState, cpu.reservation_address);
        as_.Movq({.base = Assembler::rbp, .disp = resv_addr_off}, base);

        // Store reservation value (always 64-bit for simplicity).
        int32_t resv_val_off = offsetof(ThreadState, cpu.reservation_value);
        as_.Movq({.base = Assembler::rbp, .disp = resv_val_off}, res);

        if (args.rt < 31) SetReg(args.rt, res);
        break;
      }

      case Decoder::AtomicOp::kStxr: {
        // Store-exclusive: compare-and-swap using reservation.
        //
        // Keep the temp footprint minimal: STXR appears mid-region in LL/SC
        // loops where most permanent register slots are already mapped, so
        // every extra simultaneous temp here risks exhausting the temp
        // reserve and failing the whole region at this instruction. RAX is
        // required by LOCK CMPXCHG anyway; use it as the scratch for both
        // the reservation-address compare and the expected value instead of
        // dedicating a temp to each.
        int32_t resv_val_off = offsetof(ThreadState, cpu.reservation_value);
        int32_t resv_addr_off = offsetof(ThreadState, cpu.reservation_address);

        // Get new value from rt.
        Register new_val = (args.rt < 31) ? GetReg(args.rt) : AllocTempReg();
        if (!success()) return;
        if (args.rt >= 31) as_.Xorl(new_val, new_val);

        // Compare the reservation address with base (via RAX), clearing the
        // reservation in between — MOV does not touch flags, so the compare
        // survives the clear.
        as_.Movq(Assembler::rax, {.base = Assembler::rbp, .disp = resv_addr_off});
        as_.Movq({.base = Assembler::rbp, .disp = resv_addr_off},
                 static_cast<int32_t>(0));
        AssemblerBase::Label* fail_label = as_.MakeLabel();
        as_.Cmpq(Assembler::rax, base);
        as_.Jcc(Condition::kNotEqual, *fail_label);

        // Expected value into RAX for LOCK CMPXCHG.
        as_.Movq(Assembler::rax, {.base = Assembler::rbp, .disp = resv_val_off});

        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        as_.SetRecoveryPoint(recovery_label);

        switch (args.size) {
          case 0: as_.LockCmpXchgb(mem, new_val); break;
          case 1: as_.LockCmpXchgw(mem, new_val); break;
          case 2: as_.LockCmpXchgl(mem, new_val); break;
          case 3: as_.LockCmpXchgq(mem, new_val); break;
        }

        AssemblerBase::Label* recovery_cont = as_.MakeLabel();
        as_.Jmp(*recovery_cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(recovery_cont);

        // CMPXCHG sets ZF on success. Rs = 0 on success, 1 on failure.
        if (args.rs < 31) {
          Register status = AllocTempReg();
          if (!success()) return;
          as_.Movq(status, static_cast<int64_t>(1));  // assume failure
          AssemblerBase::Label* done_label = as_.MakeLabel();
          as_.Jcc(Condition::kNotEqual, *done_label);  // ZF=0 means CAS failed
          as_.Xorl(status, status);  // success: status = 0
          as_.Bind(done_label);
          SetReg(args.rs, status);
        }

        AssemblerBase::Label* end_label = as_.MakeLabel();
        as_.Jmp(*end_label);

        // Reservation mismatch: store fails, Rs = 1.
        as_.Bind(fail_label);
        if (args.rs < 31) {
          Register fail_status = AllocTempReg();
          if (!success()) return;
          as_.Movq(fail_status, static_cast<int64_t>(1));
          SetReg(args.rs, fail_status);
        }

        as_.Bind(end_label);
        break;
      }

      // LSE bitwise atomics (LDCLR/LDSET/LDEOR).
      // x86 has no single-instruction equivalent; emit a CMPXCHG retry loop.
      // ARM: tmp = [Xn]; [Xn] = tmp <op> Xs; Xt = tmp (zero-extended for W form).
      case Decoder::AtomicOp::kLdclr:
      case Decoder::AtomicOp::kLdset:
      case Decoder::AtomicOp::kLdeor: {
        Register mask = (args.rs < 31) ? GetReg(args.rs) : AllocTempReg();
        if (!success()) return;
        if (args.rs >= 31) as_.Xorl(mask, mask);

        // Precompute ~Xs once for LDCLR (mask doesn't change across iterations).
        Register clr_mask = no_register;
        if (args.op == Decoder::AtomicOp::kLdclr) {
          clr_mask = AllocTempReg();
          if (!success()) return;
          as_.Movq(clr_mask, mask);
          as_.Notq(clr_mask);
        }

        // Initial fetch of [mem] into RAX (Load() emits fault recovery).
        Register init = Load(lss, /*is_signed=*/false, /*is_64bit_target=*/true,
                             base, 0);
        if (!success()) return;
        as_.Movq(Assembler::rax, init);

        Register tmp_new = AllocTempReg();
        if (!success()) return;

        AssemblerBase::Label* loop_top = as_.MakeLabel();
        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        AssemblerBase::Label* cont = as_.MakeLabel();
        as_.Bind(loop_top);
        as_.Movq(tmp_new, Assembler::rax);
        switch (args.op) {
          case Decoder::AtomicOp::kLdclr:
            as_.Andq(tmp_new, clr_mask);
            break;
          case Decoder::AtomicOp::kLdset:
            as_.Orq(tmp_new, mask);
            break;
          case Decoder::AtomicOp::kLdeor:
            as_.Xorq(tmp_new, mask);
            break;
          default:
            break;
        }

        as_.SetRecoveryPoint(recovery_label);
        switch (args.size) {
          case 0: as_.LockCmpXchgb(mem, tmp_new); break;
          case 1: as_.LockCmpXchgw(mem, tmp_new); break;
          case 2: as_.LockCmpXchgl(mem, tmp_new); break;
          case 3: as_.LockCmpXchgq(mem, tmp_new); break;
        }
        as_.Jmp(*cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(cont);
        // ZF=0 means CMPXCHG failed; retry with the updated RAX.
        as_.Jcc(Condition::kNotEqual, *loop_top);

        // Old value is in RAX. For byte/halfword sizes the failure path of
        // CMPXCHG only refreshes AL/AX; upper bits are whatever the initial
        // Movzxbl/Movzxwl in Load() set them to (zero), so they're already
        // clean. Still mask explicitly to match the existing kLdadd/kSwp
        // pattern for byte/halfword zero-extension to 64 bits.
        if (args.rt < 31) {
          Register old_val = AllocTempReg();
          if (!success()) return;
          as_.Movq(old_val, Assembler::rax);
          if (args.size == 0) {
            as_.Andq(old_val, static_cast<int32_t>(0xFF));
          } else if (args.size == 1) {
            as_.Andq(old_val, static_cast<int32_t>(0xFFFF));
          }
          SetReg(args.rt, old_val);
        }
        break;
      }

      // atomic min/max (LSE Armv8.1).
      // x86 has no single-instruction equivalent; emit a CMPXCHG retry loop
      // with a sign- or zero-extended Cmpq+Cmovq to pick max/min.  ARM
      // semantics: tmp = [Xn]; [Xn] = is_max ? max(tmp, Xs) : min(tmp, Xs);
      // Xt = tmp (zero-extended at the operation size to 64 bits for W form).
      case Decoder::AtomicOp::kLdsmax:
      case Decoder::AtomicOp::kLdsmin:
      case Decoder::AtomicOp::kLdumax:
      case Decoder::AtomicOp::kLdumin: {
        const bool is_signed = (args.op == Decoder::AtomicOp::kLdsmax ||
                                args.op == Decoder::AtomicOp::kLdsmin);
        const bool is_max = (args.op == Decoder::AtomicOp::kLdsmax ||
                             args.op == Decoder::AtomicOp::kLdumax);

        Register operand = (args.rs < 31) ? GetReg(args.rs) : AllocTempReg();
        if (!success()) return;
        if (args.rs >= 31) as_.Xorl(operand, operand);

        // Initial fetch of [mem] into RAX (Load emits fault recovery).
        Register init = Load(lss, /*is_signed=*/false,
                             /*is_64bit_target=*/true, base, 0);
        if (!success()) return;
        as_.Movq(Assembler::rax, init);

        Register tmp_new = AllocTempReg();
        if (!success()) return;
        Register cmp_old = AllocTempReg();
        if (!success()) return;
        Register cmp_op = AllocTempReg();
        if (!success()) return;

        AssemblerBase::Label* loop_top = as_.MakeLabel();
        AssemblerBase::Label* recovery_label = as_.MakeLabel();
        AssemblerBase::Label* cont = as_.MakeLabel();
        as_.Bind(loop_top);

        // Sign- or zero-extend RAX (current) and operand to 64 bits at the
        // guest operation size so the Cmpq below has the right ordering.
        switch (args.size) {
          case 0:
            if (is_signed) {
              as_.Movsxbq(cmp_old, Assembler::rax);
              as_.Movsxbq(cmp_op, operand);
            } else {
              as_.Movzxbl(cmp_old, Assembler::rax);
              as_.Movzxbl(cmp_op, operand);
            }
            break;
          case 1:
            if (is_signed) {
              as_.Movsxwq(cmp_old, Assembler::rax);
              as_.Movsxwq(cmp_op, operand);
            } else {
              as_.Movzxwl(cmp_old, Assembler::rax);
              as_.Movzxwl(cmp_op, operand);
            }
            break;
          case 2:
            if (is_signed) {
              as_.Movsxlq(cmp_old, Assembler::rax);
              as_.Movsxlq(cmp_op, operand);
            } else {
              as_.Movl(cmp_old, Assembler::rax);   // implicit zero-extend to 64
              as_.Movl(cmp_op, operand);
            }
            break;
          case 3:
            as_.Movq(cmp_old, Assembler::rax);
            as_.Movq(cmp_op, operand);
            break;
        }

        // tmp_new starts as cmp_old; Cmovq swaps in cmp_op when the
        // comparison says cmp_op is the desired max/min.
        as_.Movq(tmp_new, cmp_old);
        as_.Cmpq(tmp_new, cmp_op);
        Condition cc;
        if (is_max) {
          cc = is_signed ? Condition::kLess
                         : Condition::kBelow;
        } else {
          cc = is_signed ? Condition::kGreater
                         : Condition::kAbove;
        }
        as_.Cmovq(cc, tmp_new, cmp_op);

        as_.SetRecoveryPoint(recovery_label);
        switch (args.size) {
          case 0: as_.LockCmpXchgb(mem, tmp_new); break;
          case 1: as_.LockCmpXchgw(mem, tmp_new); break;
          case 2: as_.LockCmpXchgl(mem, tmp_new); break;
          case 3: as_.LockCmpXchgq(mem, tmp_new); break;
        }
        as_.Jmp(*cont);
        as_.Bind(recovery_label);
        ExitGeneratedCode(GetInsnAddr());
        as_.Bind(cont);
        // ZF=0 means CMPXCHG saw a stale RAX; retry with the refreshed one.
        as_.Jcc(Condition::kNotEqual, *loop_top);

        // Old value in RAX. ARM W-form atomics zero-extend the loaded value
        // to 64 bits; byte/halfword CMPXCHG only refreshes AL/AX so the
        // upper bits already match the initial zero-extending Load, but we
        // mask explicitly to match the kLdadd/kSwp/kLdset pattern.
        if (args.rt < 31) {
          Register old_val = AllocTempReg();
          if (!success()) return;
          as_.Movq(old_val, Assembler::rax);
          if (args.size == 0) {
            as_.Andq(old_val, static_cast<int32_t>(0xFF));
          } else if (args.size == 1) {
            as_.Andq(old_val, static_cast<int32_t>(0xFFFF));
          }
          SetReg(args.rt, old_val);
        }
        break;
      }

      // CASP JIT (compare-and-swap pair).
      // size=2 (32-bit pair): pack Rs:Rs+1 into a single 64-bit value and use
      //   LOCK CMPXCHGq.  Mirrors the interpreter's path (interpreter.h
      //   delegates the 32-bit pair to AtomicCASVal<uint64_t>).
      // size=3 (64-bit pair): LOCK CMPXCHG16B.  This clobbers RAX/RDX/RBX/RCX
      //   in fixed roles (RDX:RAX = expected, RCX:RBX = desired, RDX:RAX = old
      //   after).  RBX/RCX/RDX are in the allocator pool and may hold
      //   permanent guest reg mappings, so we save them on the host stack
      //   around the CMPXCHG16B and restore on both the success path and the
      //   fault-recovery path.  RAX is reserved (no guest reg maps to it) so
      //   it needs no save.
      // Other sizes (0, 1) and `acquire`/`release` variants are not
      // architecturally allowed for CASP; x86 TSO already provides the
      // ordering CASPA/CASPL/CASPAL want, so the same emit covers all four.
      case Decoder::AtomicOp::kCasp: {
        if (args.size != 2 && args.size != 3) { Undefined(); return; }

        const uint8_t rs_lo = args.rs;
        const uint8_t rs_hi = static_cast<uint8_t>(args.rs + 1);
        const uint8_t rt_lo = args.rt;
        const uint8_t rt_hi = static_cast<uint8_t>(args.rt + 1);

        if (args.size == 2) {
          // 32-bit pair via packed 64-bit LOCK CMPXCHG.
          Register expected = AllocTempReg();
          if (!success()) return;
          Register desired = AllocTempReg();
          if (!success()) return;
          Register hi_tmp = AllocTempReg();
          if (!success()) return;

          // expected = (Rs_hi & 0xFFFFFFFF) << 32 | (Rs_lo & 0xFFFFFFFF).
          // Movl with a register destination zero-extends to 64 bits.
          if (rs_lo < 31) {
            as_.Movl(expected, GetReg(rs_lo));
          } else {
            as_.Xorl(expected, expected);
          }
          if (rs_hi < 31) {
            as_.Movl(hi_tmp, GetReg(rs_hi));
            as_.Shlq(hi_tmp, int8_t{32});
            as_.Orq(expected, hi_tmp);
          }

          // desired = (Rt_hi & 0xFFFFFFFF) << 32 | (Rt_lo & 0xFFFFFFFF).
          if (rt_lo < 31) {
            as_.Movl(desired, GetReg(rt_lo));
          } else {
            as_.Xorl(desired, desired);
          }
          if (rt_hi < 31) {
            as_.Movl(hi_tmp, GetReg(rt_hi));
            as_.Shlq(hi_tmp, int8_t{32});
            as_.Orq(desired, hi_tmp);
          }

          // CMPXCHG: RAX = expected. On equal, store `desired`; on unequal,
          // RAX loaded from [mem].  RAX is reserved (no guest map), safe to
          // clobber.
          as_.Movq(Assembler::rax, expected);

          AssemblerBase::Label* recovery_label = as_.MakeLabel();
          as_.SetRecoveryPoint(recovery_label);
          as_.LockCmpXchgq(mem, desired);

          AssemblerBase::Label* cont = as_.MakeLabel();
          as_.Jmp(*cont);
          as_.Bind(recovery_label);
          ExitGeneratedCode(GetInsnAddr());
          as_.Bind(cont);

          // Unpack old: low 32 → Rs (zero-extend), high 32 → Rs+1
          // (zero-extend).  Each half is written back like ARM CAS Wt, i.e.,
          // zero-extended to 64 bits.
          if (rs_lo < 31) {
            Register old_lo = AllocTempReg();
            if (!success()) return;
            as_.Movl(old_lo, Assembler::rax);  // zero-extends to 64
            SetReg(rs_lo, old_lo);
          }
          if (rs_hi < 31) {
            Register old_hi = AllocTempReg();
            if (!success()) return;
            as_.Movq(old_hi, Assembler::rax);
            as_.Shrq(old_hi, int8_t{32});
            SetReg(rs_hi, old_hi);
          }
        } else {
          // size=3: 64-bit pair via LOCK CMPXCHG16B.
          //
          // Register roles for CMPXCHG16B [mem]:
          //   in : RDX:RAX = expected (high:low)
          //        RCX:RBX = desired  (high:low)
          //   out: RDX:RAX = old [mem] (always — equal case leaves them
          //        unchanged, which equals the old value already)
          //        ZF = 1 on success, 0 on failure
          //
          // CMPXCHG16B clobbers RBX/RCX/RDX, which are in the allocator
          // pool and may hold permanent guest mappings.  We save them on
          // the host stack and restore on both the success and the
          // fault-recovery paths.  RAX is reserved (no guest reg maps to
          // it) so it needs no save.
          //
          // Critical detail: `base` came back from ApplyTbi() (the
          // function entry), which allocates a temp.  AllocTempReg()
          // starts handing out RDX as the first temp, so `base` is
          // typically RDX itself.  We MUST move it to a stable temp
          // before clobbering RDX — otherwise the CMPXCHG16B operand's
          // memory base ends up holding exp_hi instead of the guest
          // address, producing a #GP from a bogus memory access.

          // Allocate 5 stable temps not in {RAX, RBX, RCX, RDX}: r15, r14,
          // r13, r12, r11 (post-ApplyTbi the first temp slot is already
          // taken, so AllocTempReg returns r15 first here).
          Register base_save = AllocTempReg();
          if (!success()) return;
          Register exp_lo = AllocTempReg();
          if (!success()) return;
          Register exp_hi = AllocTempReg();
          if (!success()) return;
          Register des_lo = AllocTempReg();
          if (!success()) return;
          Register des_hi = AllocTempReg();
          if (!success()) return;

          // Pin the memory base in a stable register before any clobber.
          as_.Movq(base_save, base);
          Assembler::Operand mem_save{.base = base_save, .disp = 0};

          // Stage all four source values into temps *before* clobbering
          // RBX/RCX/RDX, so GetReg() can still read guest values held in
          // RBX/RCX/RDX.
          if (rs_lo < 31) {
            as_.Movq(exp_lo, GetReg(rs_lo));
          } else {
            as_.Xorl(exp_lo, exp_lo);
          }
          if (rs_hi < 31) {
            as_.Movq(exp_hi, GetReg(rs_hi));
          } else {
            as_.Xorl(exp_hi, exp_hi);
          }
          if (rt_lo < 31) {
            as_.Movq(des_lo, GetReg(rt_lo));
          } else {
            as_.Xorl(des_lo, des_lo);
          }
          if (rt_hi < 31) {
            as_.Movq(des_hi, GetReg(rt_hi));
          } else {
            as_.Xorl(des_hi, des_hi);
          }

          // Save RBX/RCX/RDX on the host stack.  Order matters: we pop in
          // reverse on restore.  These pushes never fault (host stack is
          // always mapped), so no recovery point is needed for them.
          as_.Push(Assembler::rbx);
          as_.Push(Assembler::rcx);
          as_.Push(Assembler::rdx);

          // Load the CMPXCHG16B operand registers.  Sources are in temps
          // that are NOT in {RAX, RBX, RCX, RDX}, so no read-after-write
          // hazards here.
          as_.Movq(Assembler::rax, exp_lo);
          as_.Movq(Assembler::rdx, exp_hi);
          as_.Movq(Assembler::rbx, des_lo);
          as_.Movq(Assembler::rcx, des_hi);

          AssemblerBase::Label* recovery_label = as_.MakeLabel();
          as_.SetRecoveryPoint(recovery_label);
          as_.LockCmpXchg16b(mem_save);

          AssemblerBase::Label* cont = as_.MakeLabel();
          as_.Jmp(*cont);

          // Recovery path: the CMPXCHG16B faulted (misaligned address or
          // bad memory).  Restore RBX/RCX/RDX so StoreMappedRegs() in
          // ExitGeneratedCode sees correct permanent-mapping values, then
          // bail to the interpreter at the current guest PC.
          as_.Bind(recovery_label);
          as_.Pop(Assembler::rdx);
          as_.Pop(Assembler::rcx);
          as_.Pop(Assembler::rbx);
          ExitGeneratedCode(GetInsnAddr());

          as_.Bind(cont);

          // Success path.  Save the old values (currently in RAX:RDX) to
          // the staging temps before popping RDX restores the guest
          // mapping value.  exp_lo/exp_hi are R15/R14, not in
          // {RAX/RBX/RCX/RDX}, so these Movqs don't interfere with the
          // pending pops.
          as_.Movq(exp_lo, Assembler::rax);
          as_.Movq(exp_hi, Assembler::rdx);
          as_.Pop(Assembler::rdx);
          as_.Pop(Assembler::rcx);
          as_.Pop(Assembler::rbx);

          // Write old values back to Rs:Rs+1.  If a guest reg mapping
          // happens to live in RBX/RCX/RDX, SetReg targets that physical
          // register and overwrites the just-restored guest value.  That's
          // the correct semantics: Rs/Rs+1 ARE getting new values from the
          // CASP, even if the guest register happened to map there.
          if (rs_lo < 31) SetReg(rs_lo, exp_lo);
          if (rs_hi < 31) SetReg(rs_hi, exp_hi);
        }
        break;
      }

      // LDXP/LDAXP (load-exclusive pair). Loads Rt:Rt2 atomically and arms the
      // exclusive monitor (reservation_address = base, reservation_value = the
      // full pair). size=2 -> 32-bit pair (8 bytes), size=3 -> 64-bit pair
      // (16 bytes). Mirrors interpreter.h's kLdxp. x86 TSO gives acquire order
      // for free, so LDAXP needs no extra fence.
      case Decoder::AtomicOp::kLdxp: {
        const int32_t resv_addr_off = offsetof(ThreadState, cpu.reservation_address);
        const int32_t resv_val_off = offsetof(ThreadState, cpu.reservation_value);
        if (args.size == 2) {
          // 32-bit pair: a single aligned 64-bit load is atomic on x86.
          Register pair = AllocTempReg();
          if (!success()) return;
          AssemblerBase::Label* recovery_label = as_.MakeLabel();
          as_.SetRecoveryPoint(recovery_label);
          as_.Movq(pair, mem);
          AssemblerBase::Label* cont = as_.MakeLabel();
          as_.Jmp(*cont);
          as_.Bind(recovery_label);
          ExitGeneratedCode(GetInsnAddr());
          as_.Bind(cont);
          // Reservation: address = base, value = the 64-bit pair.
          as_.Movq({.base = Assembler::rbp, .disp = resv_addr_off}, base);
          as_.Movq({.base = Assembler::rbp, .disp = resv_val_off}, pair);
          // Rt = low 32 (zero-extended), Rt2 = high 32 (zero-extended).
          if (args.rt < 31) {
            Register lo = AllocTempReg();
            if (!success()) return;
            as_.Movl(lo, pair);  // Movl to a register zero-extends to 64.
            SetReg(args.rt, lo);
          }
          if (args.rt2 < 31) {
            Register hi = AllocTempReg();
            if (!success()) return;
            as_.Movq(hi, pair);
            as_.Shrq(hi, int8_t{32});
            SetReg(args.rt2, hi);
          }
        } else if (args.size == 3) {
          // 64-bit pair: atomic 128-bit read via LOCK CMPXCHG16B with
          // expected==desired==0 (writes 0 only when *addr==0, i.e. a no-op) --
          // the same trick the interpreter uses. It clobbers RAX/RBX/RCX/RDX;
          // RBX/RCX/RDX are in the allocator pool and may hold permanent guest
          // mappings, so save them on the host stack and pin `base` first (post-
          // ApplyTbi `base` is typically RDX itself). RAX is reserved.
          Register base_save = AllocTempReg();
          if (!success()) return;
          Register old_lo = AllocTempReg();
          if (!success()) return;
          Register old_hi = AllocTempReg();
          if (!success()) return;
          as_.Movq(base_save, base);
          Assembler::Operand mem_save{.base = base_save, .disp = 0};

          as_.Push(Assembler::rbx);
          as_.Push(Assembler::rcx);
          as_.Push(Assembler::rdx);
          // expected = desired = 0 -> RDX:RAX = RCX:RBX = 0.
          as_.Xorl(Assembler::rax, Assembler::rax);
          as_.Xorl(Assembler::rdx, Assembler::rdx);
          as_.Xorl(Assembler::rbx, Assembler::rbx);
          as_.Xorl(Assembler::rcx, Assembler::rcx);

          AssemblerBase::Label* recovery_label = as_.MakeLabel();
          as_.SetRecoveryPoint(recovery_label);
          as_.LockCmpXchg16b(mem_save);
          AssemblerBase::Label* cont = as_.MakeLabel();
          as_.Jmp(*cont);
          as_.Bind(recovery_label);
          as_.Pop(Assembler::rdx);
          as_.Pop(Assembler::rcx);
          as_.Pop(Assembler::rbx);
          ExitGeneratedCode(GetInsnAddr());
          as_.Bind(cont);
          // Old value is in RAX:RDX. Stash to stable temps before restoring RDX.
          as_.Movq(old_lo, Assembler::rax);
          as_.Movq(old_hi, Assembler::rdx);
          as_.Pop(Assembler::rdx);
          as_.Pop(Assembler::rcx);
          as_.Pop(Assembler::rbx);
          // Reservation: address = base (pinned), value = full 128-bit pair.
          as_.Movq({.base = Assembler::rbp, .disp = resv_addr_off}, base_save);
          as_.Movq({.base = Assembler::rbp, .disp = resv_val_off}, old_lo);
          as_.Movq({.base = Assembler::rbp, .disp = resv_val_off + 8}, old_hi);
          if (args.rt < 31) SetReg(args.rt, old_lo);
          if (args.rt2 < 31) SetReg(args.rt2, old_hi);
        } else {
          Undefined();
          return;
        }
        break;
      }

      // STXP/STLXP (store-exclusive pair). Stores Rt:Rt2 atomically iff the
      // exclusive monitor still holds the value LDXP read (an exact compare-and-
      // swap against reservation_value); reports success(0)/failure(1) in Rs and
      // clears the reservation. Mirrors interpreter.h's kStxp. The LOCK prefix is
      // a full barrier, so STLXP needs no extra fence.
      case Decoder::AtomicOp::kStxp: {
        const int32_t resv_addr_off = offsetof(ThreadState, cpu.reservation_address);
        const int32_t resv_val_off = offsetof(ThreadState, cpu.reservation_value);
        if (args.size == 2) {
          // 32-bit pair: new = (Rt2_lo32 << 32) | Rt_lo32; single 64-bit CAS
          // against the reserved 64-bit value.
          Register new_val = AllocTempReg();
          if (!success()) return;
          if (args.rt < 31) {
            as_.Movl(new_val, GetReg(args.rt));  // low 32, zero-extended
          } else {
            as_.Xorl(new_val, new_val);
          }
          if (args.rt2 < 31) {
            Register hi_tmp = AllocTempReg();
            if (!success()) return;
            as_.Movl(hi_tmp, GetReg(args.rt2));
            as_.Shlq(hi_tmp, int8_t{32});
            as_.Orq(new_val, hi_tmp);
          }
          Register expected_val = AllocTempReg();
          if (!success()) return;
          as_.Movq(expected_val, {.base = Assembler::rbp, .disp = resv_val_off});
          Register resv_addr = AllocTempReg();
          if (!success()) return;
          as_.Movq(resv_addr, {.base = Assembler::rbp, .disp = resv_addr_off});
          // Clear reservation.
          as_.Movq({.base = Assembler::rbp, .disp = resv_addr_off},
                   static_cast<int32_t>(0));
          AssemblerBase::Label* fail_label = as_.MakeLabel();
          as_.Cmpq(resv_addr, base);
          as_.Jcc(Condition::kNotEqual, *fail_label);

          as_.Movq(Assembler::rax, expected_val);
          AssemblerBase::Label* recovery_label = as_.MakeLabel();
          as_.SetRecoveryPoint(recovery_label);
          as_.LockCmpXchgq(mem, new_val);
          AssemblerBase::Label* recovery_cont = as_.MakeLabel();
          as_.Jmp(*recovery_cont);
          as_.Bind(recovery_label);
          ExitGeneratedCode(GetInsnAddr());
          as_.Bind(recovery_cont);
          // CMPXCHG sets ZF on success. Rs = 0 on success, 1 on failure.
          if (args.rs < 31) {
            Register status = AllocTempReg();
            if (!success()) return;
            as_.Movq(status, static_cast<int64_t>(1));
            AssemblerBase::Label* done_label = as_.MakeLabel();
            as_.Jcc(Condition::kNotEqual, *done_label);
            as_.Xorl(status, status);
            as_.Bind(done_label);
            SetReg(args.rs, status);
          }
          AssemblerBase::Label* end_label = as_.MakeLabel();
          as_.Jmp(*end_label);
          as_.Bind(fail_label);
          if (args.rs < 31) {
            Register fail_status = AllocTempReg();
            if (!success()) return;
            as_.Movq(fail_status, static_cast<int64_t>(1));
            SetReg(args.rs, fail_status);
          }
          as_.Bind(end_label);
        } else if (args.size == 3) {
          // 64-bit pair via LOCK CMPXCHG16B: expected = reservation_value
          // (RDX:RAX), desired = Rt:Rt2 (RCX:RBX). Same register discipline as
          // CASP size=3. Guard on reservation_address == base; on mismatch the
          // store fails without touching memory.
          const uint8_t rt_lo = args.rt;
          const uint8_t rt_hi = args.rt2;
          Register base_save = AllocTempReg();
          if (!success()) return;
          Register des_lo = AllocTempReg();
          if (!success()) return;
          Register des_hi = AllocTempReg();
          if (!success()) return;
          Register resv_addr = AllocTempReg();
          if (!success()) return;
          Register status = AllocTempReg();
          if (!success()) return;

          as_.Movq(base_save, base);
          Assembler::Operand mem_save{.base = base_save, .disp = 0};
          if (rt_lo < 31) {
            as_.Movq(des_lo, GetReg(rt_lo));
          } else {
            as_.Xorl(des_lo, des_lo);
          }
          if (rt_hi < 31) {
            as_.Movq(des_hi, GetReg(rt_hi));
          } else {
            as_.Xorl(des_hi, des_hi);
          }
          as_.Movq(resv_addr, {.base = Assembler::rbp, .disp = resv_addr_off});
          // Clear reservation and default status to failure (1).
          as_.Movq({.base = Assembler::rbp, .disp = resv_addr_off},
                   static_cast<int32_t>(0));
          as_.Movq(status, static_cast<int64_t>(1));

          AssemblerBase::Label* fail_label = as_.MakeLabel();
          as_.Cmpq(resv_addr, base_save);
          as_.Jcc(Condition::kNotEqual, *fail_label);

          as_.Push(Assembler::rbx);
          as_.Push(Assembler::rcx);
          as_.Push(Assembler::rdx);
          // expected = reservation_value -> RDX:RAX; desired -> RCX:RBX.
          as_.Movq(Assembler::rax, {.base = Assembler::rbp, .disp = resv_val_off});
          as_.Movq(Assembler::rdx, {.base = Assembler::rbp, .disp = resv_val_off + 8});
          as_.Movq(Assembler::rbx, des_lo);
          as_.Movq(Assembler::rcx, des_hi);

          AssemblerBase::Label* recovery_label = as_.MakeLabel();
          as_.SetRecoveryPoint(recovery_label);
          as_.LockCmpXchg16b(mem_save);
          AssemblerBase::Label* cont = as_.MakeLabel();
          as_.Jmp(*cont);
          as_.Bind(recovery_label);
          as_.Pop(Assembler::rdx);
          as_.Pop(Assembler::rcx);
          as_.Pop(Assembler::rbx);
          ExitGeneratedCode(GetInsnAddr());
          as_.Bind(cont);
          // ZF = 1 on success. Compute status (0/1) before restoring RDX so the
          // pops don't disturb the flags-derived value.
          AssemblerBase::Label* store_ok = as_.MakeLabel();
          as_.Jcc(Condition::kNotEqual, *store_ok);  // ZF=0 -> keep status=1
          as_.Xorl(status, status);                  // success -> status=0
          as_.Bind(store_ok);
          as_.Pop(Assembler::rdx);
          as_.Pop(Assembler::rcx);
          as_.Pop(Assembler::rbx);

          as_.Bind(fail_label);
          if (args.rs < 31) SetReg(args.rs, status);
        } else {
          Undefined();
          return;
        }
        break;
      }

      default:
        Undefined();
        break;
    }
  }

