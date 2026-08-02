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
// Family: integer data-processing, branches, scalar load/store, system.
// Methods:
//   AddSubImm, AddSubImmTags, LogicalImm, MoveWide, MoveWideKeep, PcRelAddr,
//   LoadLiteral, Bitfield, Branch, BranchCond(decl), BranchRegister,
//   CompareAndBranch, TestAndBranch, ApplyTbi, Load, Store, AddImm, LoadPair,
//   StorePair, ApplyOffsetExtend, LoadReg, StoreReg, Svc(decl), Brk, Mrs,
//   Msr, LogicalShiftedReg, AddSubShiftedReg, AddSubExtendedReg,
//   ConditionalSelect, DataProc2Src, DataProc3Src, Nop, DataMemoryBarrier,
//   IcIvau, Undefined, MteDataProc, MteLoadStore

  Register AddSubImm(bool is_sub, bool set_flags, bool is_64bit,
                     Register src, uint32_t imm) {
    Register res = AllocTempReg();
    if (is_64bit) {
      as_.Movq(res, src);
      // Always emit the op when setting flags (even imm==0) so EFLAGS are valid.
      if (set_flags || imm != 0) {
        if (is_sub) {
          as_.Subq(res, static_cast<int32_t>(imm));
        } else {
          as_.Addq(res, static_cast<int32_t>(imm));
        }
      }
    } else {
      as_.Movl(res, src);  // 32-bit mov zero-extends to 64 bits
      if (set_flags || imm != 0) {
        if (is_sub) {
          as_.Subl(res, static_cast<int32_t>(imm));
        } else {
          as_.Addl(res, static_cast<int32_t>(imm));
        }
      }
    }
    if (set_flags) {
      EmitStoreArmNZCV(is_sub);
    }
    return res;
  }

  // ADDG/SUBG (MTE tag arithmetic). Rare; handled by the
  // interpreter. End the region before this instruction (success_=false); the
  // returned register is discarded along with the un-installed region.
  Register AddSubImmTags(bool is_sub, Register src, uint8_t uimm6, uint8_t uimm4) {
    UNUSED(is_sub);
    UNUSED(uimm6);
    UNUSED(uimm4);
    success_ = false;
    return src;
  }

  Register LogicalImm(Decoder::LogicalImmOpcode opcode, bool is_64bit,
                      Register src, uint64_t imm) {
    Register res = AllocTempReg();
    Register imm_reg = AllocTempReg();
    as_.Movq(imm_reg, static_cast<int64_t>(imm));
    if (is_64bit) {
      as_.Movq(res, src);
      switch (opcode) {
        case Decoder::LogicalImmOpcode::kAnd:
        case Decoder::LogicalImmOpcode::kAnds:
          as_.Andq(res, imm_reg);
          break;
        case Decoder::LogicalImmOpcode::kOrr:
          as_.Orq(res, imm_reg);
          break;
        case Decoder::LogicalImmOpcode::kEor:
          as_.Xorq(res, imm_reg);
          break;
        default:
          Undefined();
          return no_register;
      }
    } else {
      as_.Movl(res, src);
      switch (opcode) {
        case Decoder::LogicalImmOpcode::kAnd:
        case Decoder::LogicalImmOpcode::kAnds:
          as_.Andl(res, imm_reg);
          break;
        case Decoder::LogicalImmOpcode::kOrr:
          as_.Orl(res, imm_reg);
          break;
        case Decoder::LogicalImmOpcode::kEor:
          as_.Xorl(res, imm_reg);
          break;
        default:
          Undefined();
          return no_register;
      }
    }
    if (opcode == Decoder::LogicalImmOpcode::kAnds) {
      // ANDS/TST: x86 AND clears CF and OF, so ARM64 C=0 and V=0 naturally.
      EmitStoreArmNZCV(/*is_sub=*/false);
    }
    return res;
  }

  Register MoveWide(Decoder::MoveWideOpcode opcode, bool is_64bit,
                    uint16_t imm16, uint8_t shift) {
    Register res = AllocTempReg();
    uint64_t value = static_cast<uint64_t>(imm16) << shift;
    if (opcode == Decoder::MoveWideOpcode::kMovn) {
      value = ~value;
    }
    if (!is_64bit) {
      value &= 0xFFFFFFFFULL;
    }
    as_.Movq(res, static_cast<int64_t>(value));
    return res;
  }

  Register MoveWideKeep(Register current, uint16_t imm16, uint8_t shift, bool is_64bit) {
    Register res = AllocTempReg();
    uint64_t mask = static_cast<uint64_t>(0xFFFF) << shift;
    uint64_t value = static_cast<uint64_t>(imm16) << shift;
    // Clear the 16-bit window, then OR in the new value.
    Register mask_reg = AllocTempReg();
    as_.Movq(res, current);
    as_.Movq(mask_reg, static_cast<int64_t>(~mask));
    as_.Andq(res, mask_reg);
    as_.Movq(mask_reg, static_cast<int64_t>(value));
    as_.Orq(res, mask_reg);
    if (!is_64bit) {
      as_.Movl(res, res);  // zero-extend 32->64
    }
    return res;
  }

  Register PcRelAddr(bool is_adrp, int64_t offset) {
    Register res = AllocTempReg();
    GuestAddr pc = GetInsnAddr();
    GuestAddr target;
    if (is_adrp) {
      target = (pc & ~static_cast<GuestAddr>(0xFFF)) + offset;
    } else {
      target = pc + offset;
    }
    as_.Movq(res, target);
    return res;
  }

  // LDR/LDRSW (literal): load from `[insn_addr + offset]`. The address
  // is constant at JIT time, so we materialize it via movabs and reuse
  // the standard Load() helper (which applies TBI, sets the recovery
  // point, and emits the size-appropriate movzx/movsx).
  Register LoadLiteral(Decoder::LoadStoreSize size, bool is_signed, int64_t offset) {
    GuestAddr target = GetInsnAddr() + offset;
    Register addr = AllocTempReg();
    as_.Movq(addr, target);
    bool is_64bit_target = (size == Decoder::LoadStoreSize::k64bit) || is_signed;
    return Load(size, is_signed, is_64bit_target, addr, 0);
  }

  Register Bitfield(Decoder::BitfieldOpcode opcode, bool is_64bit,
                    Register dst_val, Register src, uint8_t immr, uint8_t imms) {
    UNUSED(dst_val);
    unsigned reg_size = is_64bit ? 64 : 32;

    // Handle common UBFM aliases with direct x86_64 instructions.
    if (opcode == Decoder::BitfieldOpcode::kUbfm) {
      // LSR: UBFM Rd, Rn, #shift, #(regsize-1) -- logical shift right
      if (imms == reg_size - 1) {
        Register res = AllocTempReg();
        if (is_64bit) {
          as_.Movq(res, src);
          if (immr != 0) as_.Shrq(res, static_cast<int8_t>(immr));
        } else {
          as_.Movl(res, src);
          if (immr != 0) as_.Shrl(res, static_cast<int8_t>(immr));
        }
        return res;
      }
      // LSL: UBFM Rd, Rn, #(regsize-shift), #(regsize-1-shift)
      // Condition: imms+1 == immr (and imms < regsize-1 to exclude LSR-by-0).
      if (imms + 1 == immr && imms < reg_size - 1) {
        uint8_t shift = reg_size - immr;
        Register res = AllocTempReg();
        if (is_64bit) {
          as_.Movq(res, src);
          as_.Shlq(res, static_cast<int8_t>(shift));
        } else {
          as_.Movl(res, src);
          as_.Shll(res, static_cast<int8_t>(shift));
        }
        return res;
      }
      // UXTB: UBFM Wd, Wn, #0, #7 -- zero-extend byte
      if (!is_64bit && immr == 0 && imms == 7) {
        Register res = AllocTempReg();
        as_.Movzxbl(res, src);
        return res;
      }
      // UXTH: UBFM Wd, Wn, #0, #15 -- zero-extend halfword
      if (!is_64bit && immr == 0 && imms == 15) {
        Register res = AllocTempReg();
        as_.Movzxwl(res, src);
        return res;
      }
      // general UBFM (UBFX extract; UBFIZ insert).
      // imms >= immr  → UBFX: extract bits[imms:immr] of src to low bits of dst.
      // imms <  immr  → UBFIZ: insert low (imms+1) bits of src at position
      //                  (reg_size - immr); other dst bits zeroed.
      Register res = AllocTempReg();
      if (imms >= immr) {
        // UBFX-like: extract bits.
        unsigned width = imms - immr + 1;
        uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
        if (is_64bit) {
          as_.Movq(res, src);
          if (immr != 0) as_.Shrq(res, static_cast<int8_t>(immr));
          if (width < 64) {
            Register mask_reg = AllocTempReg();
            as_.Movq(mask_reg, static_cast<int64_t>(mask));
            as_.Andq(res, mask_reg);
          }
        } else {
          as_.Movl(res, src);
          if (immr != 0) as_.Shrl(res, static_cast<int8_t>(immr));
          if (width < 32) {
            as_.Andl(res, static_cast<int32_t>(mask & 0xFFFFFFFFULL));
          }
        }
      } else {
        // UBFIZ-like: extract low (imms+1) bits of src, shift left by (reg_size - immr).
        unsigned width = imms + 1;
        unsigned pos = reg_size - immr;
        uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
        if (is_64bit) {
          as_.Movq(res, src);
          if (width < 64) {
            Register mask_reg = AllocTempReg();
            as_.Movq(mask_reg, static_cast<int64_t>(mask));
            as_.Andq(res, mask_reg);
          }
          if (pos != 0) as_.Shlq(res, static_cast<int8_t>(pos));
        } else {
          as_.Movl(res, src);
          if (width < 32) {
            as_.Andl(res, static_cast<int32_t>(mask & 0xFFFFFFFFULL));
          }
          if (pos != 0) as_.Shll(res, static_cast<int8_t>(pos));
        }
      }
      return res;
    }

    // Handle common SBFM aliases with direct x86_64 instructions.
    if (opcode == Decoder::BitfieldOpcode::kSbfm) {
      // ASR: SBFM Rd, Rn, #shift, #(regsize-1) -- arithmetic shift right
      if (imms == reg_size - 1) {
        Register res = AllocTempReg();
        if (is_64bit) {
          as_.Movq(res, src);
          if (immr != 0) as_.Sarq(res, static_cast<int8_t>(immr));
        } else {
          // ASR Wd, Wn, #imm: 32-bit arithmetic shift. Sarl writes the result
          // to the low 32 bits and (per x86 semantics) zero-extends to 64.
          // ARM64 W-register writes also zero the upper 32 bits of Xd, so the
          // zero-extension we get from Sarl is exactly correct — do NOT
          // sign-extend further, that would corrupt the upper 32 with the sign
          // of bit 31 of the shifted W-value.
          as_.Movl(res, src);
          if (immr != 0) as_.Sarl(res, static_cast<int8_t>(immr));
        }
        return res;
      }
      // SXTB: SBFM Xd/Wd, Wn, #0, #7 -- sign-extend byte
      if (immr == 0 && imms == 7) {
        Register res = AllocTempReg();
        if (is_64bit) {
          as_.Movsxbq(res, src);
        } else {
          as_.Movsxbl(res, src);
        }
        return res;
      }
      // SXTH: SBFM Xd/Wd, Wn, #0, #15 -- sign-extend halfword
      if (immr == 0 && imms == 15) {
        Register res = AllocTempReg();
        if (is_64bit) {
          as_.Movsxwq(res, src);
        } else {
          as_.Movsxwl(res, src);
        }
        return res;
      }
      // SXTW: SBFM Xd, Wn, #0, #31 -- sign-extend word to 64-bit
      if (is_64bit && immr == 0 && imms == 31) {
        Register res = AllocTempReg();
        as_.Movsxlq(res, src);
        return res;
      }
      // General SBFX/SBFIZ. Shift the field's top bit to the register's top
      // bit, then arithmetic-shift it down to its destination position:
      //   A = regsize-1-imms            (field top -> bit regsize-1)
      //   B = (A + immr) mod regsize    (top -> destination)
      // For imms >= immr (SBFX) B lands the field at bit 0; for imms < immr
      // (SBFIZ) the mod wraps and lands it at bit regsize-immr. One formula,
      // both aliases, two shifts.
      //
      // This path is hot in practice: compilers emit `sbfx xN, xM, #0, #1`
      // (sign-extend bit 0, the carry/bool-to-mask idiom) all over checked
      // arithmetic and strtoull-class parsing, and without it every such
      // instruction ended its region and round-tripped through the
      // interpreter -- ~13 interpreter instructions of overhead per execution
      // in the hottest loops a media player has.
      {
        Register res = AllocTempReg();
        const uint8_t shl_amount = static_cast<uint8_t>(reg_size - 1 - imms);
        const uint8_t sar_amount =
            static_cast<uint8_t>((shl_amount + immr) & (reg_size - 1));
        if (is_64bit) {
          as_.Movq(res, src);
          if (shl_amount != 0) as_.Shlq(res, static_cast<int8_t>(shl_amount));
          if (sar_amount != 0) as_.Sarq(res, static_cast<int8_t>(sar_amount));
        } else {
          // 32-bit: operate on the low word; Shll/Sarl zero-extend the result
          // to 64 bits, which matches ARM64 W-register write semantics.
          as_.Movl(res, src);
          if (shl_amount != 0) as_.Shll(res, static_cast<int8_t>(shl_amount));
          if (sar_amount != 0) as_.Sarl(res, static_cast<int8_t>(sar_amount));
        }
        return res;
      }
    }

    // general BFM (BFI / BFXIL / BFC).
    // BFM merges bits from src into dst_val, leaving the other dst_val
    // bits unchanged. Two cases by immr/imms relationship:
    //   imms >= immr (BFXIL): take bits[imms:immr] of src and write them
    //     to bits[width-1:0] of dst, keeping the rest of dst.
    //   imms <  immr (BFI):   take bits[imms:0] of src and write them at
    //     bit position (reg_size - immr), keeping the rest of dst.
    // In both cases: result = (dst_val & ~mask) | (shifted_src & mask).
    // The semantics_player passes the live dst register as dst_val for kBfm.
    if (opcode == Decoder::BitfieldOpcode::kBfm) {
      Register res = AllocTempReg();
      Register mask_reg = AllocTempReg();
      uint64_t mask;
      if (imms >= immr) {
        // BFXIL: extract width = imms-immr+1 bits, deposit at bit 0.
        unsigned width = imms - immr + 1;
        mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
        if (is_64bit) {
          as_.Movq(res, src);
          if (immr != 0) as_.Shrq(res, static_cast<int8_t>(immr));
          as_.Movq(mask_reg, static_cast<int64_t>(mask));
          as_.Andq(res, mask_reg);
        } else {
          as_.Movl(res, src);
          if (immr != 0) as_.Shrl(res, static_cast<int8_t>(immr));
          // 32-bit AND with immediate (sign-extended to 64); mask fits in 32.
          as_.Movl(mask_reg, static_cast<int32_t>(mask & 0xFFFFFFFFULL));
          as_.Andl(res, mask_reg);
        }
      } else {
        // BFI / BFC: extract low width = imms+1 bits, deposit at pos.
        unsigned width = imms + 1;
        unsigned pos = reg_size - immr;
        uint64_t field_mask =
            (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
        mask = (pos >= 64) ? 0 : (field_mask << pos);
        if (is_64bit) {
          as_.Movq(res, src);
          as_.Movq(mask_reg, static_cast<int64_t>(field_mask));
          as_.Andq(res, mask_reg);
          if (pos != 0) as_.Shlq(res, static_cast<int8_t>(pos));
        } else {
          as_.Movl(res, src);
          as_.Movl(mask_reg, static_cast<int32_t>(field_mask & 0xFFFFFFFFULL));
          as_.Andl(res, mask_reg);
          if (pos != 0) as_.Shll(res, static_cast<int8_t>(pos));
        }
      }
      // Merge: res = (dst_val & ~mask) | res.
      // We must not mutate dst_val: copy it into a fresh temp first.
      Register keep = AllocTempReg();
      if (is_64bit) {
        as_.Movq(keep, dst_val);
        as_.Movq(mask_reg, static_cast<int64_t>(~mask));
        as_.Andq(keep, mask_reg);
        as_.Orq(res, keep);
      } else {
        as_.Movl(keep, dst_val);
        as_.Movl(mask_reg, static_cast<int32_t>((~mask) & 0xFFFFFFFFULL));
        as_.Andl(keep, mask_reg);
        as_.Orl(res, keep);
        // ARM64 W-register writes zero the upper 32 bits of the X register;
        // the 32-bit Orl above already zero-extends res to 64 on x86_64.
      }
      return res;
    }

    // For other complex bitfield operations not yet handled, fall back.
    Undefined();
    return no_register;
  }

  void Branch(int32_t offset) {
    is_region_end_reached_ = true;
    ExitRegion(GetInsnAddr() + offset);
  }

  void BranchCond(Decoder::Condition cond, int32_t offset);  // implemented in .cc

  void BranchRegister(Register target) {
    is_region_end_reached_ = true;
    ExitRegionIndirect(target);
  }

  void CompareAndBranch(bool is_nonzero, bool is_64bit, Register src, int32_t offset) {
    Assembler::Label* cont = as_.MakeLabel();
    // Test the register for zero/nonzero.
    if (is_64bit) {
      as_.Testq(src, src);
    } else {
      as_.Testl(src, src);
    }
    // Jump to continuation (fall through) on the NOT-taken case.
    if (is_nonzero) {
      // CBNZ: branch if nonzero, so skip branch (fall through) if zero.
      as_.Jcc(Condition::kEqual, *cont);
    } else {
      // CBZ: branch if zero, so skip branch (fall through) if nonzero.
      as_.Jcc(Condition::kNotEqual, *cont);
    }
    // forward branch extension with back-edge detection
    GuestAddr target = GetInsnAddr() + offset;
    if (offset <= 0) {
      // Backward branch: end region to prevent infinite loops.
      is_region_end_reached_ = true;
    }
    ExitRegion(target);
    as_.Bind(cont);
  }

  void TestAndBranch(bool is_nonzero, Register src, uint8_t bit, int32_t offset) {
    Assembler::Label* cont = as_.MakeLabel();
    // BT sets CF if the specified bit is set.
    if (bit >= 32) {
      as_.Btq(src, static_cast<int8_t>(bit));
    } else {
      as_.Btl(src, static_cast<int8_t>(bit));
    }
    // Jump to continuation (fall through) on the NOT-taken case.
    if (is_nonzero) {
      // TBNZ: branch if bit set (CF=1), so skip if CF=0.
      as_.Jcc(Condition::kNotCarry, *cont);
    } else {
      // TBZ: branch if bit clear (CF=0), so skip if CF=1.
      as_.Jcc(Condition::kCarry, *cont);
    }
    // forward branch extension with back-edge detection
    GuestAddr target = GetInsnAddr() + offset;
    if (offset <= 0) {
      // Backward branch: end region to prevent infinite loops.
      is_region_end_reached_ = true;
    }
    ExitRegion(target);
    as_.Bind(cont);
  }

  // ARM64 TBI (Top Byte Ignore): mask the top 8 bits of an
  // address register before using it in a host x86 load/store. ARM64 ignores
  // the top byte of pointers in load/store; x86 doesn't, so we must clear it
  // ourselves. Returns a temp register holding (base & 0x00FFFFFFFFFFFFFF).
  Register ApplyTbi(Register base) {
    Register tbi = AllocTempReg();
    as_.Movq(tbi, base);
    as_.Shlq(tbi, static_cast<int8_t>(8));
    as_.Shrq(tbi, static_cast<int8_t>(8));
    return tbi;
  }

  Register Load(Decoder::LoadStoreSize size, bool is_signed, bool is_64bit_target,
                Register base, int32_t offset) {
    // apply TBI mask before using base as memory operand.
    base = ApplyTbi(base);
    AssemblerBase::Label* recovery_label = as_.MakeLabel();
    as_.SetRecoveryPoint(recovery_label);

    Register res = AllocTempReg();
    Assembler::Operand mem{.base = base, .disp = offset};
    switch (size) {
      case Decoder::LoadStoreSize::k64bit:
        as_.Movq(res, mem);
        break;
      case Decoder::LoadStoreSize::k32bit:
        if (is_signed && is_64bit_target) {
          as_.Movsxlq(res, mem);
        } else {
          as_.Movl(res, mem);  // zero-extends to 64 bits
        }
        break;
      case Decoder::LoadStoreSize::k16bit:
        if (is_signed) {
          if (is_64bit_target) {
            as_.Movsxwq(res, mem);
          } else {
            as_.Movsxwl(res, mem);
          }
        } else {
          as_.Movzxwl(res, mem);
        }
        break;
      case Decoder::LoadStoreSize::k8bit:
        if (is_signed) {
          if (is_64bit_target) {
            as_.Movsxbq(res, mem);
          } else {
            as_.Movsxbl(res, mem);
          }
        } else {
          as_.Movzxbl(res, mem);
        }
        break;
    }

    // Emit recovery code for memory faults.
    AssemblerBase::Label* cont = as_.MakeLabel();
    as_.Jmp(*cont);
    as_.Bind(recovery_label);
    ExitGeneratedCode(GetInsnAddr());
    as_.Bind(cont);

    return res;
  }

  void Store(Decoder::LoadStoreSize size, Register base, int32_t offset, Register data) {
    // apply TBI mask before using base as memory operand.
    base = ApplyTbi(base);
    AssemblerBase::Label* recovery_label = as_.MakeLabel();
    as_.SetRecoveryPoint(recovery_label);

    Assembler::Operand mem{.base = base, .disp = offset};
    switch (size) {
      case Decoder::LoadStoreSize::k64bit:
        as_.Movq(mem, data);
        break;
      case Decoder::LoadStoreSize::k32bit:
        as_.Movl(mem, data);
        break;
      case Decoder::LoadStoreSize::k16bit:
        as_.Movw(mem, data);
        break;
      case Decoder::LoadStoreSize::k8bit:
        as_.Movb(mem, data);
        break;
    }

    // Emit recovery code for memory faults.
    AssemblerBase::Label* cont = as_.MakeLabel();
    as_.Jmp(*cont);
    as_.Bind(recovery_label);
    ExitGeneratedCode(GetInsnAddr());
    as_.Bind(cont);
  }

  Register AddImm(Register base, int32_t offset) {
    Register result = AllocTempReg();
    as_.Movq(result, base);
    if (offset != 0) {
      as_.Addq(result, offset);
    }
    return result;
  }

  void LoadPair(Decoder::LoadStoreSize size, Register base, int32_t offset,
                uint8_t rt1, uint8_t rt2, uint8_t scale, bool is_signed) {
    UNUSED(offset);  // Already applied by caller in semantics_player.
    // LDP Xt1, Xt2, [Xn]: when Xt1 (or Xt2) aliases Xn, the
    // ARM64 architecture loads BOTH pair elements using the *original* base
    // (post-index/writeback are decoded separately into base updates). The
    // previous implementation issued SetReg(rt1, val1) between the two Loads,
    // and since `base` references the same host register that the mapping for
    // Xt1 may write to, the second Load then read from the wrong address —
    // observed as WhatsApp's libsuperpack.so dispatcher (`ldp x0, x8, [x0];
    // ldr x3, [x8, #0x28]; br x3`) jumping into random xz-compressed bytes.
    // Fix: load both halves into temps first, then commit both via SetReg.
    // LDPSW sign-extends each 32-bit element into its 64-bit target register.
    bool is_64bit_target = (size == Decoder::LoadStoreSize::k64bit) || is_signed;
    Register val1 = Load(size, is_signed, is_64bit_target, base, 0);
    if (!success()) return;
    Register val2 = Load(size, is_signed, is_64bit_target, base,
                         static_cast<int32_t>(scale));
    if (!success()) return;
    if (rt1 != 31) SetReg(rt1, val1);
    if (rt2 != 31) SetReg(rt2, val2);
  }

  void StorePair(Decoder::LoadStoreSize size, Register base, int32_t offset,
                 Register data1, Register data2, uint8_t scale) {
    UNUSED(offset);  // Already applied by caller in semantics_player.
    Store(size, base, 0, data1);
    if (!success()) return;
    Store(size, base, static_cast<int32_t>(scale), data2);
  }

  // Apply the correct 32->64 extension to the offset
  // register before shift+add. Without this, ldrb/ldr [Xn, Wm, UXTW]
  // and SXTW forms can produce wrong addresses (silent data corruption,
  // not faults — observed as Brotli "Bad context map" in
  // libsuperpack-jni.so's decompressor on Facebook startup).
  void ApplyOffsetExtend(Register dst, Register src, uint8_t extend_type) {
    // extend_type encoding (3-bit option from the encoded instruction):
    //   010=UXTW, 011=LSL (UXTX), 110=SXTW, 111=SXTX
    // Smaller-byte forms (UXTB etc.) are not encodable for load/store.
    switch (extend_type) {
      case 0b010:  // UXTW: zero-extend low 32 bits (32-bit mov zero-extends)
        as_.Movl(dst, src);
        break;
      case 0b110:  // SXTW: sign-extend low 32 bits to 64
        as_.Movsxlq(dst, src);
        break;
      case 0b011:  // LSL / UXTX: full 64-bit value, no extension
      case 0b111:  // SXTX: full 64-bit value, sign-extend is a no-op
      default:
        as_.Movq(dst, src);
        break;
    }
  }

  Register LoadReg(Decoder::LoadStoreSize size, bool is_signed, bool is_64bit_target,
                   Register base, Register offset_reg, uint8_t extend_type,
                   uint8_t shift_amount) {
    // Compute address: base + extend(offset_reg) << shift_amount
    Register addr = AllocTempReg();
    if (!success()) return no_register;
    ApplyOffsetExtend(addr, offset_reg, extend_type);
    if (shift_amount != 0) {
      as_.Shlq(addr, static_cast<int8_t>(shift_amount));
    }
    as_.Addq(addr, base);
    return Load(size, is_signed, is_64bit_target, addr, 0);
  }

  void StoreReg(Decoder::LoadStoreSize size, Register base, Register offset_reg,
                uint8_t extend_type, uint8_t shift_amount, Register data) {
    // Compute address: base + extend(offset_reg) << shift_amount
    Register addr = AllocTempReg();
    if (!success()) return;
    ApplyOffsetExtend(addr, offset_reg, extend_type);
    if (shift_amount != 0) {
      as_.Shlq(addr, static_cast<int8_t>(shift_amount));
    }
    as_.Addq(addr, base);
    Store(size, addr, 0, data);
  }

  // Emits the syscall inline and chains to the next instruction's region
  // instead of bailing to the interpreter. Implemented in the .cc where
  // code_gen_lib's EmitSyscall is in scope. See BranchCond for the same
  // header-declaration / .cc-definition split.
  void Svc(uint16_t imm);

  // BRK must be handled by the interpreter (which raises the
  // synchronous SIGTRAP). End the region before this instruction; the dispatch
  // loop installs kInterpreted for the BRK address, exactly like SVC.
  void Brk(uint16_t imm) {
    UNUSED(imm);
    success_ = false;
  }

  Register Mrs(Decoder::SystemReg sysreg) {
    if (sysreg == Decoder::SystemReg::kTpidrEl0) {
      // TPIDR_EL0: Thread-local storage pointer, stored in ThreadState.tls.
      Register res = AllocTempReg();
      int32_t tls_offset = offsetof(ThreadState, tls);
      as_.Movq(res, {.base = Assembler::rbp, .disp = tls_offset});
      return res;
    }
    if (sysreg == Decoder::SystemReg::kNzcv) {
      // MRS Xn, NZCV: read ARM64 flags into a register.
      // Stored (x86 LAHF layout): N=bit15, Z=bit14, C=bit8, V=bit0.
      // Architectural NZCV:       N=bit31, Z=bit30, C=bit29, V=bit28.
      // Assemble each field exactly; guest code does save/restore pairs
      // (mrs ; ... ; msr NZCV) around flag-clobbering work, and an
      // approximation here silently corrupts C and V for it.
      Register res = AllocTempReg();
      Register tmp = AllocTempReg();
      if (!success()) return no_register;
      int32_t flags_offset = offsetof(ThreadState, cpu.flags);
      as_.Movzxwl(res, {.base = Assembler::rbp, .disp = flags_offset});
      // C: stored bit8 -> bit29.
      as_.Movl(tmp, res);
      as_.Andl(tmp, static_cast<int32_t>(0x100));
      as_.Shll(tmp, static_cast<int8_t>(21));
      // N/Z: stored bits 15/14 -> bits 31/30; V: stored bit0 -> bit28.
      Register vbit = AllocTempReg();
      if (!success()) return no_register;
      as_.Movl(vbit, res);
      as_.Andl(vbit, static_cast<int32_t>(0x1));
      as_.Shll(vbit, static_cast<int8_t>(28));
      as_.Andl(res, static_cast<int32_t>(0xC000));
      as_.Shll(res, static_cast<int8_t>(16));
      as_.Orl(res, tmp);
      as_.Orl(res, vbit);
      return res;
    }
    if (sysreg == Decoder::SystemReg::kFpcr) {
      // MRS Xn, FPCR: read the cached FPCR value (the interpreter's MSR FPCR
      // keeps it in sync with host MXCSR; reads never touch host state).
      // Soft-float code reads FPCR for the rounding mode on every operation
      // -- strtod and long-double emulation round-tripped the interpreter
      // here on each call before this was JIT-resolved.
      Register res = AllocTempReg();
      if (!success()) return no_register;
      int32_t fpcr_offset = offsetof(ThreadState, cpu.cached_fpcr);
      as_.Movl(res, {.base = Assembler::rbp, .disp = fpcr_offset});  // 32-bit load zero-extends
      return res;
    }
    if (sysreg == Decoder::SystemReg::kCtrEl0) {
      // CTR_EL0: Cache Type Register - constant value matching interpreter.
      Register res = AllocTempReg();
      if (!success()) return no_register;
      as_.Movq(res, 0x8444c004ULL);
      return res;
    }
    if (sysreg == Decoder::SystemReg::kDczidEl0) {
      // DCZID_EL0: Data Cache Zero ID Register.
      // DZP=1 (DC ZVA prohibited), BS=0. Matches interpreter value.
      Register res = AllocTempReg();
      if (!success()) return no_register;
      as_.Movq(res, 0x10ULL);
      return res;
    }
    if (sysreg == Decoder::SystemReg::kMidrEl1) {
      // MIDR_EL1: synthesised Cortex-A53 r0p4 layout. Same value the
      // interpreter returns; keeping it JIT-resolved avoids a region
      // exit on every CPU-detect MRS in hot code (Bionic ifunc
      // resolvers, compression-lib feature probes).
      Register res = AllocTempReg();
      if (!success()) return no_register;
      as_.Movq(res, 0x410FD034ULL);
      return res;
    }
    Undefined();
    return no_register;
  }

  void Msr(Decoder::SystemReg sysreg, Register src) {
    if (sysreg == Decoder::SystemReg::kNzcv) {
      // MSR NZCV, Xn: write to ARM64 flags. Exact inverse of the MRS
      // conversion above:
      //   arch N=bit31 -> stored bit15, Z=bit30 -> bit14,
      //   arch C=bit29 -> stored bit8,  V=bit28 -> bit0.
      Register tmp = AllocTempReg();
      Register acc = AllocTempReg();
      if (!success()) return;
      // N/Z.
      as_.Movl(acc, src);
      as_.Shrl(acc, static_cast<int8_t>(16));
      as_.Andl(acc, static_cast<int32_t>(0xC000));
      // C.
      as_.Movl(tmp, src);
      as_.Shrl(tmp, static_cast<int8_t>(21));
      as_.Andl(tmp, static_cast<int32_t>(0x100));
      as_.Orl(acc, tmp);
      // V.
      as_.Movl(tmp, src);
      as_.Shrl(tmp, static_cast<int8_t>(28));
      as_.Andl(tmp, static_cast<int32_t>(0x1));
      as_.Orl(acc, tmp);
      int32_t flags_offset = offsetof(ThreadState, cpu.flags);
      as_.Movw({.base = Assembler::rbp, .disp = flags_offset}, acc);
      return;
    }
    if (sysreg == Decoder::SystemReg::kTpidrEl0) {
      // MSR TPIDR_EL0, Xn: write thread-local storage pointer.
      int32_t tls_offset = offsetof(ThreadState, tls);
      as_.Movq({.base = Assembler::rbp, .disp = tls_offset}, src);
      return;
    }
    UNUSED(src);
    Undefined();
  }

  Register LogicalShiftedReg(Decoder::LogicalShiftedRegOpcode opcode, bool is_64bit,
                              bool invert, Register src1, Register src2,
                              Decoder::ShiftType shift_type, uint8_t shift_amount) {
    Register op2 = AllocTempReg();
    EmitShift(op2, src2, shift_type, shift_amount, is_64bit);
    if (invert) {
      if (is_64bit) {
        as_.Notq(op2);
      } else {
        as_.Notl(op2);
      }
    }
    Register res = AllocTempReg();
    if (is_64bit) {
      as_.Movq(res, src1);
      switch (opcode) {
        case Decoder::LogicalShiftedRegOpcode::kAnd:
        case Decoder::LogicalShiftedRegOpcode::kAnds:
          as_.Andq(res, op2);
          break;
        case Decoder::LogicalShiftedRegOpcode::kOrr:
          as_.Orq(res, op2);
          break;
        case Decoder::LogicalShiftedRegOpcode::kEor:
          as_.Xorq(res, op2);
          break;
        default:
          Undefined();
          return no_register;
      }
    } else {
      as_.Movl(res, src1);
      switch (opcode) {
        case Decoder::LogicalShiftedRegOpcode::kAnd:
        case Decoder::LogicalShiftedRegOpcode::kAnds:
          as_.Andl(res, op2);
          break;
        case Decoder::LogicalShiftedRegOpcode::kOrr:
          as_.Orl(res, op2);
          break;
        case Decoder::LogicalShiftedRegOpcode::kEor:
          as_.Xorl(res, op2);
          break;
        default:
          Undefined();
          return no_register;
      }
    }
    if (opcode == Decoder::LogicalShiftedRegOpcode::kAnds) {
      EmitStoreArmNZCV(/*is_sub=*/false);
    }
    return res;
  }

  Register AddSubShiftedReg(bool is_sub, bool set_flags, bool is_64bit,
                             Register src1, Register src2,
                             Decoder::ShiftType shift_type, uint8_t shift_amount) {
    Register op2 = AllocTempReg();
    EmitShift(op2, src2, shift_type, shift_amount, is_64bit);
    Register res = AllocTempReg();
    if (is_64bit) {
      as_.Movq(res, src1);
      if (is_sub) {
        as_.Subq(res, op2);
      } else {
        as_.Addq(res, op2);
      }
    } else {
      as_.Movl(res, src1);
      if (is_sub) {
        as_.Subl(res, op2);
      } else {
        as_.Addl(res, op2);
      }
    }
    if (set_flags) {
      EmitStoreArmNZCV(is_sub);
    }
    return res;
  }

  // AddSubExtendedReg JIT
  Register AddSubExtendedReg(bool is_sub, bool set_flags, bool is_64bit,
                              Register src1, Register src2,
                              uint8_t extend_type, uint8_t shift_amount) {
    if (shift_amount > 4) { Undefined(); return no_register; }

    // Apply extension to src2 into a temp register.
    Register ext = AllocTempReg();
    if (!success()) return no_register;

    // extend_type encoding: 000=UXTB, 001=UXTH, 010=UXTW, 011=UXTX,
    //                       100=SXTB, 101=SXTH, 110=SXTW, 111=SXTX
    switch (extend_type) {
      case 0b000:  // UXTB: zero-extend byte
        as_.Movzxbl(ext, src2);
        break;
      case 0b001:  // UXTH: zero-extend halfword
        as_.Movzxwl(ext, src2);
        break;
      case 0b010:  // UXTW: zero-extend word (32-bit mov zero-extends to 64)
        as_.Movl(ext, src2);
        break;
      case 0b011:  // UXTX: no extension needed for 64-bit
        as_.Movq(ext, src2);
        break;
      case 0b100:  // SXTB: sign-extend byte
        if (is_64bit) {
          as_.Movsxbq(ext, src2);
        } else {
          as_.Movsxbl(ext, src2);
        }
        break;
      case 0b101:  // SXTH: sign-extend halfword
        if (is_64bit) {
          as_.Movsxwq(ext, src2);
        } else {
          as_.Movsxwl(ext, src2);
        }
        break;
      case 0b110:  // SXTW: sign-extend word
        if (is_64bit) {
          as_.Movsxlq(ext, src2);
        } else {
          as_.Movl(ext, src2);
        }
        break;
      case 0b111:  // SXTX: sign-extend doubleword (no-op for 64-bit)
        as_.Movq(ext, src2);
        break;
      default:
        Undefined();
        return no_register;
    }

    // Apply shift.
    if (shift_amount > 0) {
      if (is_64bit) {
        as_.Shlq(ext, static_cast<int8_t>(shift_amount));
      } else {
        as_.Shll(ext, static_cast<int8_t>(shift_amount));
      }
    }

    // Add or subtract.
    Register res = AllocTempReg();
    if (!success()) return no_register;

    if (is_64bit) {
      as_.Movq(res, src1);
      if (is_sub) {
        as_.Subq(res, ext);
      } else {
        as_.Addq(res, ext);
      }
    } else {
      as_.Movl(res, src1);
      if (is_sub) {
        as_.Subl(res, ext);
      } else {
        as_.Addl(res, ext);
      }
    }

    if (set_flags) {
      EmitStoreArmNZCV(is_sub);
    }
    return res;
  }

  Register ConditionalSelect(Decoder::ConditionalSelectOpcode opcode, bool is_64bit,
                              Register src1, Register src2, Decoder::Condition cond) {
    // Prepare result = src2 (false case), then apply opcode transformation.
    Register result = AllocTempReg();
    if (is_64bit) {
      as_.Movq(result, src2);
    } else {
      as_.Movl(result, src2);
    }

    switch (opcode) {
      case Decoder::ConditionalSelectOpcode::kCsel:
        // No transformation on src2.
        break;
      case Decoder::ConditionalSelectOpcode::kCsinc:
        if (is_64bit) {
          as_.Addq(result, static_cast<int32_t>(1));
        } else {
          as_.Addl(result, static_cast<int32_t>(1));
        }
        break;
      case Decoder::ConditionalSelectOpcode::kCsinv:
        if (is_64bit) {
          as_.Notq(result);
        } else {
          as_.Notl(result);
        }
        break;
      case Decoder::ConditionalSelectOpcode::kCsneg:
        if (is_64bit) {
          as_.Negq(result);
        } else {
          as_.Negl(result);
        }
        break;
    }

    // Evaluate condition: if TRUE, overwrite result with src1. Jump over the
    // overwrite (to done) when the condition is not met.
    Assembler::Label* done = as_.MakeLabel();
    EmitJumpIfCondNotMet(cond, *done);

    // Condition is true: overwrite result with src1 (unmodified).
    if (is_64bit) {
      as_.Movq(result, src1);
    } else {
      as_.Movl(result, src1);
    }

    as_.Bind(done);
    return result;
  }

  Register DataProc2Src(Decoder::DataProc2SrcOpcode opcode, bool is_64bit,
                         Register src1, Register src2) {
    Register res = AllocTempReg();
    switch (opcode) {
      case Decoder::DataProc2SrcOpcode::kLslv:
      case Decoder::DataProc2SrcOpcode::kLsrv:
      case Decoder::DataProc2SrcOpcode::kAsrv:
      case Decoder::DataProc2SrcOpcode::kRorv: {
        // save/restore rcx (now in allocator pool)
        // All variable-shift instructions use CL for the shift amount.
        as_.Subq(Assembler::rsp, 8);
        as_.Movq({.base = Assembler::rsp}, Assembler::rcx);  // save rcx
        // If src1 is rcx, copy it to res BEFORE overwriting rcx with shift amount.
        if (src1 == Assembler::rcx) {
          if (is_64bit) { as_.Movq(res, src1); } else { as_.Movl(res, src1); }
          as_.Movq(Assembler::rcx, src2);
        } else {
          as_.Movq(Assembler::rcx, src2);
          if (is_64bit) { as_.Movq(res, src1); } else { as_.Movl(res, src1); }
        }
        if (is_64bit) {
          switch (opcode) {
            case Decoder::DataProc2SrcOpcode::kLslv: as_.ShlqByCl(res); break;
            case Decoder::DataProc2SrcOpcode::kLsrv: as_.ShrqByCl(res); break;
            case Decoder::DataProc2SrcOpcode::kAsrv: as_.SarqByCl(res); break;
            case Decoder::DataProc2SrcOpcode::kRorv: as_.RorqByCl(res); break;
            default: break;
          }
        } else {
          switch (opcode) {
            case Decoder::DataProc2SrcOpcode::kLslv: as_.ShllByCl(res); break;
            case Decoder::DataProc2SrcOpcode::kLsrv: as_.ShrlByCl(res); break;
            case Decoder::DataProc2SrcOpcode::kAsrv: as_.SarlByCl(res); break;
            case Decoder::DataProc2SrcOpcode::kRorv: as_.RorlByCl(res); break;
            default: break;
          }
        }
        // restore rcx after shift
        if (res == Assembler::rcx) {
          as_.Addq(Assembler::rsp, 8);  // discard saved rcx (result is in rcx)
        } else {
          as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
          as_.Addq(Assembler::rsp, 8);  // restore rcx
        }
        break;
      }
      case Decoder::DataProc2SrcOpcode::kUdiv: {
        // ARM64 UDIV: Rd = Rn / Rm.  If Rm == 0, Rd = 0.
        EmitDivCommon</*kSigned=*/false>(res, src1, src2, is_64bit);
        break;
      }
      case Decoder::DataProc2SrcOpcode::kSdiv: {
        // ARM64 SDIV: Rd = Rn / Rm.  If Rm == 0, Rd = 0.
        EmitDivCommon</*kSigned=*/true>(res, src1, src2, is_64bit);
        break;
      }
      // CRC32C* (Castagnoli) via the host SSE4.2 CRC32
      // instruction, which uses the same polynomial as ARM's CRC32C ops.
      // The accumulator is Wn (zero-extended), the data is Wm (b/h/w) or Xm
      // (x). The IEEE CRC32* ops use a different polynomial and stay on the
      // interpreter (they fall through to the default below). Bail if the
      // host lacks SSE4.2.
      case Decoder::DataProc2SrcOpcode::kCrc32cb:
      case Decoder::DataProc2SrcOpcode::kCrc32ch:
      case Decoder::DataProc2SrcOpcode::kCrc32cw:
      case Decoder::DataProc2SrcOpcode::kCrc32cx: {
        if (!host_platform::kHasSSE4_2) { success_ = false; return no_register; }
        as_.Movl(res, src1);  // 32-bit CRC accumulator (Wn), zero-extended
        switch (opcode) {
          case Decoder::DataProc2SrcOpcode::kCrc32cb: as_.Crc32cb(res, src2); break;
          case Decoder::DataProc2SrcOpcode::kCrc32ch: as_.Crc32ch(res, src2); break;
          case Decoder::DataProc2SrcOpcode::kCrc32cw: as_.Crc32cw(res, src2); break;
          default:                                    as_.Crc32cx(res, src2); break;
        }
        break;
      }
      // IEEE CRC32B/H/W/X — ISO-3309 poly 0x04C11DB7 (reflected 0xEDB88320).
      // The host SSE4.2 CRC32 computes the *Castagnoli* poly (used by the
      // CRC32C* block above), so it cannot serve these; lower them with a
      // carry-less multiply (PCLMULQDQ) reflected Barrett reduction. The
      // accumulator is Wn (low 32 bits); the data is Wm (b/h/w) or Xm (x).
      // Reflected fold/Barrett constants (rev33 of x^k mod P, of P, and of the
      // Barrett mu = floor(x^64/P)); these coincide with zlib's CRC32 tables:
      //   K64 = x^64 mod P = 0x163CD6124   K32 = x^32 mod P = 0x1DB710640
      //   K16 = x^16       = 0x000010000   K8  = x^8        = 0x001000000
      //   MU  = floor(x^64/P) = 0x1F7011641   POLYR = rev33(P) = 0x1DB710641
      // Bail to the interpreter if the host lacks PCLMULQDQ.
      case Decoder::DataProc2SrcOpcode::kCrc32b:
      case Decoder::DataProc2SrcOpcode::kCrc32h:
      case Decoder::DataProc2SrcOpcode::kCrc32w:
      case Decoder::DataProc2SrcOpcode::kCrc32x: {
        if (!host_platform::kHasCLMUL) { success_ = false; return no_register; }
        SimdRegister xt = AllocTempSimdReg();  // running value / low fold
        SimdRegister xk = AllocTempSimdReg();  // constant loader
        SimdRegister xh = AllocTempSimdReg();  // CRC32X high-dword fold
        Register rt = AllocTempReg();          // combined T / Barrett R (64-bit)
        Register rs = AllocTempReg();          // scratch (immediates, t1/t2)
        if (!success()) { return no_register; }
        if (opcode == Decoder::DataProc2SrcOpcode::kCrc32x) {
          // T = Xm ^ zext32(Wn); fold low and high dwords separately.
          as_.Movl(rt, src1);                  // acc (zero-extended Wn)
          as_.Xorq(rt, src2);                  // T = acc ^ Xm (64-bit)
          as_.Movd(xt, rt);                    // xt.d0 = T_lo
          as_.Movq(rs, rt);
          as_.Shrq(rs, int8_t{32});            // rs = T_hi
          as_.Movd(xh, rs);                    // xh.d0 = T_hi
          as_.Movq(rs, int64_t{0x163CD6124});  // K64
          as_.Movq(xk, rs);
          as_.Pclmullqlqdq(xt, xk);            // clmul(T_lo, K64)
          as_.Movq(rs, int64_t{0x1DB710640});  // K32
          as_.Movq(xk, rs);
          as_.Pclmullqlqdq(xh, xk);            // clmul(T_hi, K32)
          as_.Pxor(xt, xh);                    // proc (64-bit) in xt.q0
        } else {
          // W/H/B: 32-bit T = Wn ^ (Wm & mask); single fold by x^{32,16,8}.
          int64_t kc;
          as_.Movl(rt, src2);                  // Wm (zero-extended)
          switch (opcode) {
            case Decoder::DataProc2SrcOpcode::kCrc32h:
              as_.Andl(rt, int32_t{0xFFFF}); kc = 0x000010000; break;  // K16
            case Decoder::DataProc2SrcOpcode::kCrc32b:
              as_.Andl(rt, int32_t{0xFF});   kc = 0x001000000; break;  // K8
            default: /* kCrc32w */           kc = 0x1DB710640; break;  // K32
          }
          as_.Xorl(rt, src1);                  // T = (Wm & mask) ^ Wn
          as_.Movd(xt, rt);                    // xt.d0 = T
          as_.Movq(rs, kc);
          as_.Movq(xk, rs);
          as_.Pclmullqlqdq(xt, xk);            // proc = clmul(T, Kc)
        }
        // Reflected Barrett reduction of the 64-bit value in xt.q0 -> 32-bit CRC.
        // The low-32 masks are realized by the 32-bit MOVD round-trips.
        as_.Movq(rt, xt);                      // R = proc (64-bit)
        as_.Movd(xt, rt);                      // xt.d0 = R & 0xFFFFFFFF
        as_.Movq(rs, int64_t{0x1F7011641});    // MU
        as_.Movq(xk, rs);
        as_.Pclmullqlqdq(xt, xk);              // t1 = clmul(R&M32, MU)
        as_.Movd(rs, xt);                      // rs = t1 & 0xFFFFFFFF
        as_.Movd(xt, rs);                      // xt.d0 = t1 & M32
        as_.Movq(rs, int64_t{0x1DB710641});    // POLYR
        as_.Movq(xk, rs);
        as_.Pclmullqlqdq(xt, xk);              // t2 = clmul(t1&M32, POLYR)
        as_.Movq(rs, xt);                      // rs = t2 (64-bit)
        as_.Xorq(rt, rs);                      // R ^ t2
        as_.Shrq(rt, int8_t{32});              // (R ^ t2) >> 32
        as_.Movl(res, rt);                     // Wd (zero-extends into Xd)
        break;
      }
      default:
        Undefined();
        return no_register;
    }
    return res;
  }

  Register DataProc3Src(Decoder::DataProc3SrcOpcode opcode, bool is_64bit,
                         Register src1, Register src2, Register src3) {
    Register res = AllocTempReg();
    switch (opcode) {
      case Decoder::DataProc3SrcOpcode::kMadd:
        // MADD: Rd = Ra + Rn * Rm  (MUL is MADD with Ra=XZR, so src3=0)
        if (is_64bit) {
          as_.Movq(res, src1);
          as_.Imulq(res, src2);
          as_.Addq(res, src3);
        } else {
          as_.Movl(res, src1);
          as_.Imull(res, src2);
          as_.Addl(res, src3);
        }
        break;
      case Decoder::DataProc3SrcOpcode::kMsub:
        // MSUB: Rd = Ra - Rn * Rm  (MNEG is MSUB with Ra=XZR)
        if (is_64bit) {
          Register tmp = AllocTempReg();
          as_.Movq(tmp, src1);
          as_.Imulq(tmp, src2);
          as_.Movq(res, src3);
          as_.Subq(res, tmp);
        } else {
          Register tmp = AllocTempReg();
          as_.Movl(tmp, src1);
          as_.Imull(tmp, src2);
          as_.Movl(res, src3);
          as_.Subl(res, tmp);
        }
        break;
      // wider multiply JIT
      case Decoder::DataProc3SrcOpcode::kSmaddl:
      case Decoder::DataProc3SrcOpcode::kSmsubl: {
        // SMADDL/SMSUBL: Xd = Xa ± (Wn * Wm) [signed 32×32→64]
        // Sign-extend both 32-bit sources to 64-bit, then 64-bit multiply.
        Register ext1 = AllocTempReg();
        if (!success()) return no_register;
        as_.Movsxlq(ext1, src1);
        Register ext2 = AllocTempReg();
        if (!success()) return no_register;
        as_.Movsxlq(ext2, src2);
        as_.Imulq(ext1, ext2);
        if (opcode == Decoder::DataProc3SrcOpcode::kSmaddl) {
          as_.Movq(res, src3);
          as_.Addq(res, ext1);
        } else {
          as_.Movq(res, src3);
          as_.Subq(res, ext1);
        }
        break;
      }
      case Decoder::DataProc3SrcOpcode::kUmaddl:
      case Decoder::DataProc3SrcOpcode::kUmsubl: {
        // UMADDL/UMSUBL: Xd = Xa ± (Wn * Wm) [unsigned 32×32→64]
        // Zero-extend both 32-bit sources to 64-bit, then 64-bit multiply.
        Register ext1 = AllocTempReg();
        if (!success()) return no_register;
        as_.Movl(ext1, src1);  // 32-bit mov zero-extends
        Register ext2 = AllocTempReg();
        if (!success()) return no_register;
        as_.Movl(ext2, src2);
        as_.Imulq(ext1, ext2);
        if (opcode == Decoder::DataProc3SrcOpcode::kUmaddl) {
          as_.Movq(res, src3);
          as_.Addq(res, ext1);
        } else {
          as_.Movq(res, src3);
          as_.Subq(res, ext1);
        }
        break;
      }
      case Decoder::DataProc3SrcOpcode::kSmulh: {
        // save/restore rdx (now in allocator pool)
        // SMULH: Xd = (Xn * Xm) >> 64 [signed high multiply]
        // x86 IMUL r64: RAX * r64 → RDX:RAX (signed)
        as_.Subq(Assembler::rsp, 8);
        as_.Movq({.base = Assembler::rsp}, Assembler::rdx);  // save rdx (clobbered by widening IMUL)
        as_.Movq(Assembler::rax, src1);
        // src2 might be rdx — push doesn't modify it, so imul sees correct value.
        as_.Imulq(src2);  // RDX:RAX = RAX * src2
        // Result (high 64 bits) is in RDX.
        if (res != Assembler::rdx) {
          as_.Movq(res, Assembler::rdx);
          as_.Movq(Assembler::rdx, {.base = Assembler::rsp});
          as_.Addq(Assembler::rsp, 8);  // restore rdx
        } else {
          as_.Addq(Assembler::rsp, 8);  // discard saved rdx
        }
        break;
      }
      case Decoder::DataProc3SrcOpcode::kUmulh: {
        // save/restore rdx (now in allocator pool)
        // UMULH: Xd = (Xn * Xm) >> 64 [unsigned high multiply]
        // x86 MUL r64: RAX * r64 → RDX:RAX (unsigned)
        as_.Subq(Assembler::rsp, 8);
        as_.Movq({.base = Assembler::rsp}, Assembler::rdx);  // save rdx (clobbered by widening MUL)
        as_.Movq(Assembler::rax, src1);
        as_.Mulq(src2);  // RDX:RAX = RAX * src2
        if (res != Assembler::rdx) {
          as_.Movq(res, Assembler::rdx);
          as_.Movq(Assembler::rdx, {.base = Assembler::rsp});
          as_.Addq(Assembler::rsp, 8);  // restore rdx
        } else {
          as_.Addq(Assembler::rsp, 8);  // discard saved rdx
        }
        break;
      }
      default:
        Undefined();
        return no_register;
    }
    return res;
  }

  void Nop() {}

  // DMB/DSB full barrier: x86 TSO lacks StoreLoad ordering, so a full guest
  // barrier must lower to MFENCE (the store-only/load-only variants are NOPed
  // upstream in the decoder).
  void DataMemoryBarrier() { as_.Mfence(); }

  // IC IVAU, Xt — invalidate the translation cache for self-modified code. The
  // address is only known at run time, so end the region here and let the
  // interpreter perform the invalidation (see Interpreter::IcIvau).
  void IcIvau(uint8_t rt) {
    UNUSED(rt);
    success_ = false;
  }

  void Undefined() { success_ = false; }

  // MTE DP-2src (IRG/GMI/SUBP/SUBPS): bail to the interpreter. These
  // are rare in real workloads (MTE-built libraries only) and SUBPS
  // sets NZCV based on a 56-bit subtraction, which is awkward to emit
  // inline; the interpreter implementation is straightforward and the
  // cost is paid once per region containing one of these.
  void MteDataProc(const Decoder::MteDataProcArgs&) { success_ = false; }

  // MTE load/store memory tags (LDG/STG/ST2G/STZG/STZ2G): bail to the
  // interpreter. These are rare in shipping APKs (only emitted when a
  // library is built with -mmemtag-stack and the runtime opts in). STZG
  // and STZ2G also touch memory via FaultyStore which would need its own
  // host fault-recovery slot if JIT'd; deferring keeps that complexity
  // out of the JIT until profiling shows it matters.
  void MteLoadStore(const Decoder::MteLoadStoreArgs&) { success_ = false; }

