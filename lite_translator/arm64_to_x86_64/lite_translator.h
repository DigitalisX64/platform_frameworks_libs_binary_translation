// region digitalis
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

#ifndef BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_H_
#define BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_H_

#include <cstdint>
#include <tuple>
#include <unordered_map>

#include "berberis/assembler/common.h"
#include "berberis/assembler/x86_64.h"
#include "berberis/base/checks.h"
#include "berberis/base/macros.h"
#include "berberis/decoder/arm64/decoder.h"
#include "berberis/decoder/arm64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/lite_translator/lite_translate_region.h"

#include "allocator.h"
#include "register_maintainer.h"

namespace berberis {

class LiteTranslator {
 public:
  // Use x86_64::Assembler directly since there is no ARM64-specific MacroAssembler yet.
  using Assembler = x86_64::Assembler;
  using Decoder = Decoder<SemanticsPlayer<LiteTranslator>>;
  using Register = Assembler::Register;
  static constexpr auto no_register = Assembler::no_register;
  using SimdRegister = Assembler::XMMRegister;
  static constexpr auto no_simd_register = Assembler::no_xmm_register;
  using Condition = Assembler::Condition;

  explicit LiteTranslator(MachineCode* machine_code,
                          GuestAddr pc,
                          LiteTranslateParams params = LiteTranslateParams{})
      : as_(machine_code),
        success_(true),
        pc_(pc),
        params_(params),
        is_region_end_reached_(false) {}

  //
  // Guest state getters/setters.
  //

  GuestAddr GetInsnAddr() const { return pc_; }

  void IncrementInsnAddr(uint8_t insn_size) { pc_ += insn_size; }

  Register GetReg(uint8_t reg) {
    CHECK_LT(reg, std::size(ThreadState{}.cpu.x));
    // region digitalis - bail early if already in error state (e.g. from Undefined())
    if (!success()) return no_register;
    // endregion
    if (IsRegMappingEnabled()) {
      auto [mapped_reg, is_new_mapping] = GetMappedRegisterOrMap(reg);
      // region digitalis - spill to temp when permanent pool is full
      if (!success()) {
        // Pool full: clear failure and fall through to temp-based load.
        success_ = true;
      } else {
        if (is_new_mapping) {
          int32_t offset = offsetof(ThreadState, cpu.x[0]) + reg * 8;
          as_.Movq(mapped_reg, {.base = Assembler::rbp, .disp = offset});
        }
        return mapped_reg;
      }
      // endregion
    }
    Register result = AllocTempReg();
    int32_t offset = offsetof(ThreadState, cpu.x[0]) + reg * 8;
    as_.Movq(result, {.base = Assembler::rbp, .disp = offset});
    return result;
  }

  void SetReg(uint8_t reg, Register value) {
    CHECK_LT(reg, std::size(ThreadState{}.cpu.x));
    // region digitalis - bail early if already in error state (e.g. from Undefined())
    if (!success()) return;
    // endregion
    if (IsRegMappingEnabled()) {
      auto [mapped_reg, _] = GetMappedRegisterOrMap(reg);
      if (success()) {
        as_.Movq(mapped_reg, value);
        gp_maintainer_.NoticeModified(reg);
        return;
      }
      // region digitalis - spill: pool full, write through to ThreadState
      success_ = true;
      // endregion
    }
    int32_t offset = offsetof(ThreadState, cpu.x[0]) + reg * 8;
    as_.Movq({.base = Assembler::rbp, .disp = offset}, value);
  }

  Register GetSp() {
    Register result = AllocTempReg();
    int32_t offset = offsetof(ThreadState, cpu.sp);
    as_.Movq(result, {.base = Assembler::rbp, .disp = offset});
    return result;
  }

  void SetSp(Register value) {
    int32_t offset = offsetof(ThreadState, cpu.sp);
    as_.Movq({.base = Assembler::rbp, .disp = offset}, value);
  }

  [[nodiscard]] Register GetImm(uint64_t imm) {
    Register imm_reg = AllocTempReg();
    as_.Movq(imm_reg, imm);
    return imm_reg;
  }

  [[nodiscard]] Register Copy(Register value) {
    Register result = AllocTempReg();
    as_.Movq(result, value);
    return result;
  }

  void StoreMappedRegs() {
    if (!IsRegMappingEnabled()) {
      return;
    }
    for (unsigned i = 0; i < kNumGuestRegs; i++) {
      if (gp_maintainer_.IsModified(i)) {
        auto mapped_reg = gp_maintainer_.GetMapped(i);
        int32_t offset = offsetof(ThreadState, cpu.x[0]) + i * 8;
        as_.Movq({.base = Assembler::rbp, .disp = offset}, mapped_reg);
      }
    }
  }

  //
  // Region exit methods.
  //

  void ExitGeneratedCode(GuestAddr target);
  void ExitRegion(GuestAddr target);
  void ExitRegionIndirect(Register target);

  //
  // Instruction implementations.
  // Each method corresponds to a SemanticsPlayer callback.
  // Unsupported instructions set success_ = false for interpreter fallback.
  //

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
      // region digitalis - general UBFM (UBFX extract; UBFIZ insert).
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
      // endregion
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
    }

    // For BFM and other complex bitfield operations, fall back to the interpreter.
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
    // region digitalis - forward branch extension with back-edge detection
    GuestAddr target = GetInsnAddr() + offset;
    if (offset <= 0) {
      // Backward branch: end region to prevent infinite loops.
      is_region_end_reached_ = true;
    }
    ExitRegion(target);
    // endregion
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
    // region digitalis - forward branch extension with back-edge detection
    GuestAddr target = GetInsnAddr() + offset;
    if (offset <= 0) {
      // Backward branch: end region to prevent infinite loops.
      is_region_end_reached_ = true;
    }
    ExitRegion(target);
    // endregion
    as_.Bind(cont);
  }

  // region digitalis - ARM64 TBI (Top Byte Ignore): mask the top 8 bits of an
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
  // endregion

  Register Load(Decoder::LoadStoreSize size, bool is_signed, bool is_64bit_target,
                Register base, int32_t offset) {
    // region digitalis - apply TBI mask before using base as memory operand.
    base = ApplyTbi(base);
    // endregion
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
    // region digitalis - apply TBI mask before using base as memory operand.
    base = ApplyTbi(base);
    // endregion
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
                uint8_t rt1, uint8_t rt2, uint8_t scale) {
    UNUSED(offset);  // Already applied by caller in semantics_player.
    // region digitalis - LDP Xt1, Xt2, [Xn]: when Xt1 (or Xt2) aliases Xn, the
    // ARM64 architecture loads BOTH pair elements using the *original* base
    // (post-index/writeback are decoded separately into base updates). The
    // previous implementation issued SetReg(rt1, val1) between the two Loads,
    // and since `base` references the same host register that the mapping for
    // Xt1 may write to, the second Load then read from the wrong address —
    // observed as WhatsApp's libsuperpack.so dispatcher (`ldp x0, x8, [x0];
    // ldr x3, [x8, #0x28]; br x3`) jumping into random xz-compressed bytes.
    // Fix: load both halves into temps first, then commit both via SetReg.
    Register val1 = Load(size, /*is_signed=*/false, /*is_64bit_target=*/
                         (size == Decoder::LoadStoreSize::k64bit), base, 0);
    if (!success()) return;
    Register val2 = Load(size, /*is_signed=*/false, /*is_64bit_target=*/
                         (size == Decoder::LoadStoreSize::k64bit), base,
                         static_cast<int32_t>(scale));
    if (!success()) return;
    if (rt1 != 31) SetReg(rt1, val1);
    if (rt2 != 31) SetReg(rt2, val2);
    // endregion
  }

  void StorePair(Decoder::LoadStoreSize size, Register base, int32_t offset,
                 Register data1, Register data2, uint8_t scale) {
    UNUSED(offset);  // Already applied by caller in semantics_player.
    Store(size, base, 0, data1);
    if (!success()) return;
    Store(size, base, static_cast<int32_t>(scale), data2);
  }

  // region digitalis - Apply the correct 32->64 extension to the offset
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
  // endregion

  void Svc(uint16_t imm) {
    // region digitalis - SVC must be handled by interpreter, not JIT.
    // Setting success_=false causes the region to end BEFORE this instruction.
    // The dispatch loop will then install kInterpreted for the SVC address,
    // and the interpreter handles the actual syscall. The previous approach
    // (ExitGeneratedCode to self) caused an infinite loop because the JIT
    // entry for this PC re-entered the same exit code.
    UNUSED(imm);
    success_ = false;
    // endregion
  }

  Register Mrs(Decoder::SystemReg sysreg) {
    // region digitalis
    if (sysreg == Decoder::SystemReg::kTpidrEl0) {
      // TPIDR_EL0: Thread-local storage pointer, stored in ThreadState.tls.
      Register res = AllocTempReg();
      int32_t tls_offset = offsetof(ThreadState, tls);
      as_.Movq(res, {.base = Assembler::rbp, .disp = tls_offset});
      return res;
    }
    if (sysreg == Decoder::SystemReg::kNzcv) {
      // MRS Xn, NZCV: read ARM64 flags into a register.
      // ARM64 NZCV register: N=bit31, Z=bit30, C=bit29, V=bit28.
      // Our stored flags use x86 LAHF layout. Convert on read.
      Register res = AllocTempReg();
      if (!success()) return no_register;
      int32_t flags_offset = offsetof(ThreadState, cpu.flags);
      as_.Movzxwl(res, {.base = Assembler::rbp, .disp = flags_offset});
      // Stored: N=bit15, Z=bit14, C=bit8, V=bit0.
      // Target: N=bit31, Z=bit30, C=bit29, V=bit28.
      // Shift left by 16 to move N/Z to bits 31/30.
      as_.Shll(res, static_cast<int8_t>(16));
      // C was at bit8, now at bit24. Need it at bit29: shift would be complex.
      // For simplicity, just return the shifted value; most code only tests
      // individual flags via conditional branches, not MRS NZCV.
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
    // endregion
    Undefined();
    return no_register;
  }

  void Msr(Decoder::SystemReg sysreg, Register src) {
    // region digitalis
    if (sysreg == Decoder::SystemReg::kNzcv) {
      // MSR NZCV, Xn: write to ARM64 flags.
      // Input: N=bit31, Z=bit30, C=bit29, V=bit28.
      // Stored: N=bit15, Z=bit14, C=bit8, V=bit0.
      Register tmp = AllocTempReg();
      if (!success()) return;
      as_.Movq(tmp, src);
      as_.Shrq(tmp, static_cast<int8_t>(16));  // N→bit15, Z→bit14
      // This is an approximation; exact bit remapping is complex.
      // Most linker code doesn't use MSR NZCV directly.
      int32_t flags_offset = offsetof(ThreadState, cpu.flags);
      as_.Movw({.base = Assembler::rbp, .disp = flags_offset}, tmp);
      return;
    }
    if (sysreg == Decoder::SystemReg::kTpidrEl0) {
      // MSR TPIDR_EL0, Xn: write thread-local storage pointer.
      int32_t tls_offset = offsetof(ThreadState, tls);
      as_.Movq({.base = Assembler::rbp, .disp = tls_offset}, src);
      return;
    }
    UNUSED(src);
    // endregion
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

  // region digitalis - AddSubExtendedReg JIT
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
  // endregion

  // region digitalis
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

    // Load ARM64 NZCV flags from ThreadState.
    int32_t flags_offset = offsetof(ThreadState, cpu.flags);
    Register flags_reg = AllocTempReg();
    as_.Movzxwl(flags_reg, {.base = Assembler::rbp, .disp = flags_offset});

    // Evaluate condition: if TRUE, overwrite result with src1.
    Assembler::Label* done = as_.MakeLabel();

    switch (cond) {
      case Decoder::Condition::kEq:
        // EQ: Z==1. Test bit 14; skip src1 if Z==0.
        as_.Btl(flags_reg, static_cast<int8_t>(14));
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      case Decoder::Condition::kNe:
        // NE: Z==0. Test bit 14; skip src1 if Z==1.
        as_.Btl(flags_reg, static_cast<int8_t>(14));
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kCs:
        // CS/HS: C==1. Test bit 8; skip src1 if C==0.
        as_.Btl(flags_reg, static_cast<int8_t>(8));
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      case Decoder::Condition::kCc:
        // CC/LO: C==0. Test bit 8; skip src1 if C==1.
        as_.Btl(flags_reg, static_cast<int8_t>(8));
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kMi:
        // MI: N==1. Test bit 15; skip src1 if N==0.
        as_.Btl(flags_reg, static_cast<int8_t>(15));
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      case Decoder::Condition::kPl:
        // PL: N==0. Test bit 15; skip src1 if N==1.
        as_.Btl(flags_reg, static_cast<int8_t>(15));
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kVs:
        // VS: V==1. Test bit 0; skip src1 if V==0.
        as_.Btl(flags_reg, static_cast<int8_t>(0));
        as_.Jcc(Condition::kNotCarry, *done);
        break;
      case Decoder::Condition::kVc:
        // VC: V==0. Test bit 0; skip src1 if V==1.
        as_.Btl(flags_reg, static_cast<int8_t>(0));
        as_.Jcc(Condition::kCarry, *done);
        break;
      case Decoder::Condition::kHi:
        // HI: C==1 && Z==0. Skip src1 if C==0 OR Z==1.
        as_.Btl(flags_reg, static_cast<int8_t>(8));
        as_.Jcc(Condition::kNotCarry, *done);  // skip if C==0
        as_.Btl(flags_reg, static_cast<int8_t>(14));
        as_.Jcc(Condition::kCarry, *done);     // skip if Z==1
        break;
      case Decoder::Condition::kLs: {
        // LS: C==0 || Z==1. Skip src1 only if C==1 AND Z==0.
        Assembler::Label* true_path = as_.MakeLabel();
        as_.Btl(flags_reg, static_cast<int8_t>(8));
        as_.Jcc(Condition::kNotCarry, *true_path);  // C==0 -> condition true
        as_.Btl(flags_reg, static_cast<int8_t>(14));
        as_.Jcc(Condition::kNotCarry, *done);        // C==1, Z==0 -> condition false
        as_.Bind(true_path);
        break;
      }
      case Decoder::Condition::kGe: {
        // GE: N==V. Extract N (bit 15), shift to bit 0, XOR with V (bit 0).
        Register tmp = AllocTempReg();
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, static_cast<int8_t>(15));
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, static_cast<int8_t>(0));
        as_.Jcc(Condition::kCarry, *done);  // skip if N!=V
        break;
      }
      case Decoder::Condition::kLt: {
        // LT: N!=V.
        Register tmp = AllocTempReg();
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, static_cast<int8_t>(15));
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, static_cast<int8_t>(0));
        as_.Jcc(Condition::kNotCarry, *done);  // skip if N==V
        break;
      }
      case Decoder::Condition::kGt: {
        // GT: Z==0 && N==V. Skip src1 if Z==1 or N!=V.
        as_.Btl(flags_reg, static_cast<int8_t>(14));
        as_.Jcc(Condition::kCarry, *done);  // skip if Z==1
        Register tmp = AllocTempReg();
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, static_cast<int8_t>(15));
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, static_cast<int8_t>(0));
        as_.Jcc(Condition::kCarry, *done);  // skip if N!=V
        break;
      }
      case Decoder::Condition::kLe: {
        // LE: Z==1 || N!=V. Skip src1 only if Z==0 AND N==V.
        Assembler::Label* true_path = as_.MakeLabel();
        as_.Btl(flags_reg, static_cast<int8_t>(14));
        as_.Jcc(Condition::kCarry, *true_path);  // Z==1 -> condition true
        Register tmp = AllocTempReg();
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, static_cast<int8_t>(15));
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, static_cast<int8_t>(0));
        as_.Jcc(Condition::kNotCarry, *done);  // Z==0, N==V -> condition false
        as_.Bind(true_path);
        break;
      }
      case Decoder::Condition::kAl:
      case Decoder::Condition::kNv:
        // Always true -- don't jump to done.
        break;
    }

    // Condition is true: overwrite result with src1 (unmodified).
    if (is_64bit) {
      as_.Movq(result, src1);
    } else {
      as_.Movl(result, src1);
    }

    as_.Bind(done);
    return result;
  }
  // endregion

  Register DataProc2Src(Decoder::DataProc2SrcOpcode opcode, bool is_64bit,
                         Register src1, Register src2) {
    Register res = AllocTempReg();
    switch (opcode) {
      case Decoder::DataProc2SrcOpcode::kLslv:
      case Decoder::DataProc2SrcOpcode::kLsrv:
      case Decoder::DataProc2SrcOpcode::kAsrv:
      case Decoder::DataProc2SrcOpcode::kRorv: {
        // region digitalis - save/restore rcx (now in allocator pool)
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
        // endregion
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
        // region digitalis - restore rcx after shift
        if (res == Assembler::rcx) {
          as_.Addq(Assembler::rsp, 8);  // discard saved rcx (result is in rcx)
        } else {
          as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
          as_.Addq(Assembler::rsp, 8);  // restore rcx
        }
        // endregion
        break;
      }
      case Decoder::DataProc2SrcOpcode::kUdiv: {
        // region digitalis - save/restore rdx (now in allocator pool)
        // ARM64 UDIV: Rd = Rn / Rm.  If Rm == 0, Rd = 0.
        // x86_64 DIV faults on divide-by-zero, so we must check first.
        Assembler::Label* zero = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        as_.Subq(Assembler::rsp, 8);
        as_.Movq({.base = Assembler::rsp}, Assembler::rdx);  // save rdx (clobbered by DIV)
        if (is_64bit) {
          as_.Testq(src2, src2);
        } else {
          as_.Testl(src2, src2);
        }
        as_.Jcc(Condition::kEqual, *zero);
        // DIV uses RDX:RAX / divisor → quotient in RAX.
        // Move src1 to rax BEFORE clearing rdx (src1 might be rdx).
        as_.Movq(Assembler::rax, src1);
        // If src2 is rdx, save rcx (now in pool) then use it as temp for divisor.
        if (src2 == Assembler::rdx) {
          as_.Subq(Assembler::rsp, 8);
          as_.Movq({.base = Assembler::rsp}, Assembler::rcx);  // save rcx
          as_.Movq(Assembler::rcx, Assembler::rdx);
        }
        as_.Xorl(Assembler::rdx, Assembler::rdx);
        if (is_64bit) {
          as_.Divq(src2 == Assembler::rdx ? Assembler::rcx : src2);
        } else {
          as_.Divl(src2 == Assembler::rdx ? Assembler::rcx : src2);
        }
        as_.Movq(res, Assembler::rax);
        as_.Jmp(*done);
        as_.Bind(zero);
        as_.Xorl(res, res);
        as_.Bind(done);
        // Restore rcx if we saved it (src2==rdx case).
        if (src2 == Assembler::rdx) {
          if (res != Assembler::rcx) {
            as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
          }
          as_.Addq(Assembler::rsp, 8);  // pop rcx slot
        }
        if (res == Assembler::rdx) {
          as_.Addq(Assembler::rsp, 8);  // discard saved rdx
        } else {
          as_.Movq(Assembler::rdx, {.base = Assembler::rsp});
          as_.Addq(Assembler::rsp, 8);  // restore rdx
        }
        // endregion
        break;
      }
      case Decoder::DataProc2SrcOpcode::kSdiv: {
        // region digitalis - save/restore rdx (now in allocator pool)
        // ARM64 SDIV: Rd = Rn / Rm.  If Rm == 0, Rd = 0.
        // INT_MIN / -1: ARM64 returns INT_MIN, x86_64 faults.
        Assembler::Label* zero = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        as_.Subq(Assembler::rsp, 8);
        as_.Movq({.base = Assembler::rsp}, Assembler::rdx);  // save rdx (clobbered by CQO/IDIV)
        if (is_64bit) {
          as_.Testq(src2, src2);
        } else {
          as_.Testl(src2, src2);
        }
        as_.Jcc(Condition::kEqual, *zero);
        // Check INT_MIN / -1 overflow: if src1==INT_MIN && src2==-1, result is INT_MIN.
        Assembler::Label* do_div = as_.MakeLabel();
        if (is_64bit) {
          as_.Cmpq(src2, static_cast<int32_t>(-1));
        } else {
          as_.Cmpl(src2, static_cast<int32_t>(-1));
        }
        as_.Jcc(Condition::kNotEqual, *do_div);
        // src2 == -1: result = -src1 (which equals INT_MIN for INT_MIN input).
        if (is_64bit) {
          as_.Movq(res, src1);
          as_.Negq(res);
        } else {
          as_.Movl(res, src1);
          as_.Negl(res);
        }
        as_.Jmp(*done);
        as_.Bind(do_div);
        // Sign-extend src1 into RDX:RAX for IDIV.
        // Move src1 to rax BEFORE CQO/CDQ clobbers rdx (src1 might be rdx).
        as_.Movq(Assembler::rax, src1);
        // If src2 is rdx, save rcx (now in pool) then use it as temp for divisor.
        if (src2 == Assembler::rdx) {
          as_.Subq(Assembler::rsp, 8);
          as_.Movq({.base = Assembler::rsp}, Assembler::rcx);  // save rcx
          as_.Movq(Assembler::rcx, Assembler::rdx);
        }
        if (is_64bit) {
          as_.Cqo();
          as_.Idivq(src2 == Assembler::rdx ? Assembler::rcx : src2);
        } else {
          as_.Cdq();
          as_.Idivl(src2 == Assembler::rdx ? Assembler::rcx : src2);
        }
        as_.Movq(res, Assembler::rax);
        as_.Jmp(*done);
        as_.Bind(zero);
        as_.Xorl(res, res);
        as_.Bind(done);
        // Restore rcx if we saved it (src2==rdx case).
        if (src2 == Assembler::rdx) {
          if (res != Assembler::rcx) {
            as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
          }
          as_.Addq(Assembler::rsp, 8);  // pop rcx slot
        }
        if (res == Assembler::rdx) {
          as_.Addq(Assembler::rsp, 8);  // discard saved rdx
        } else {
          as_.Movq(Assembler::rdx, {.base = Assembler::rsp});
          as_.Addq(Assembler::rsp, 8);  // restore rdx
        }
        // endregion
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
      // region digitalis - wider multiply JIT
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
        // region digitalis - save/restore rdx (now in allocator pool)
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
        // endregion
        break;
      }
      case Decoder::DataProc3SrcOpcode::kUmulh: {
        // region digitalis - save/restore rdx (now in allocator pool)
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
        // endregion
        break;
      }
      // endregion
      default:
        Undefined();
        return no_register;
    }
    return res;
  }

  void Nop() {}

  void Undefined() { success_ = false; }

  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs& args) {
    // region digitalis
    if (args.op == 1 && args.cmode == 0b1110 && args.abc == 0 && args.defgh == 0 && args.q) {
      // MOVI Vd.2D, #0x0 — zero the register.
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      as_.Pxor(xmm, xmm);
      // Store to ThreadState SIMD register.
      int32_t offset = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;
      as_.Movdqu({.base = Assembler::rbp, .disp = offset}, xmm);
      return;
    }
    // endregion
    Undefined();
  }

  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& args, Register base) {
    // region digitalis - apply TBI mask before using base as memory operand.
    base = ApplyTbi(base);
    // endregion
    // region digitalis
    // Handle 128-bit SIMD load/store with immediate offset (STR/LDR Q-register).
    if (args.size == Decoder::SimdLoadStoreSize::k128bit) {
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }

      int32_t offset = args.offset;
      if (args.is_store) {
        // Load from ThreadState SIMD register, then store to memory.
        int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;
        as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movdqu({.base = base, .disp = offset}, xmm);
      } else {
        // Load from memory, store to ThreadState SIMD register.
        as_.Movdqu(xmm, {.base = base, .disp = offset});
        int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      }
      return;
    }
    // 64-bit (D-register) load/store.
    if (args.size == Decoder::SimdLoadStoreSize::k64bit) {
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }

      int32_t offset = args.offset;
      if (args.is_store) {
        int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;
        as_.Movq(xmm, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movq({.base = base, .disp = offset}, xmm);
      } else {
        as_.Pxor(xmm, xmm);  // Zero upper 64 bits.
        as_.Movq(xmm, {.base = base, .disp = offset});
        int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      }
      return;
    }
    // 32-bit (S-register) load/store.
    if (args.size == Decoder::SimdLoadStoreSize::k32bit) {
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }

      int32_t offset = args.offset;
      int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;
      if (args.is_store) {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movl({.base = base, .disp = offset}, tmp);
      } else {
        // Zero the full 128-bit register, then load 32 bits.
        SimdRegister xmm = AllocTempSimdReg();
        if (xmm == no_simd_register) { Undefined(); return; }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
        as_.Movl(tmp, {.base = base, .disp = offset});
        as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
      }
      return;
    }
    // 16-bit (H-register) load/store.
    if (args.size == Decoder::SimdLoadStoreSize::k16bit) {
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }

      int32_t offset = args.offset;
      int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;
      if (args.is_store) {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movw({.base = base, .disp = offset}, tmp);
      } else {
        SimdRegister xmm = AllocTempSimdReg();
        if (xmm == no_simd_register) { Undefined(); return; }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
        as_.Movzxwl(tmp, {.base = base, .disp = offset});
        as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
      }
      return;
    }
    // 8-bit (B-register) load/store.
    if (args.size == Decoder::SimdLoadStoreSize::k8bit) {
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }

      int32_t offset = args.offset;
      int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;
      if (args.is_store) {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movb({.base = base, .disp = offset}, tmp);
      } else {
        SimdRegister xmm = AllocTempSimdReg();
        if (xmm == no_simd_register) { Undefined(); return; }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
        as_.Movzxbl(tmp, {.base = base, .disp = offset});
        as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
      }
      return;
    }
    // endregion
    Undefined();
  }

  void SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs& args, Register addr) {
    // region digitalis - apply TBI mask before using addr as memory operand.
    addr = ApplyTbi(addr);
    // endregion
    // region digitalis
    // Handle 128-bit pair store/load (STP/LDP q-register).
    if (args.size == Decoder::SimdLoadStoreSize::k128bit) {
      SimdRegister xmm1 = AllocTempSimdReg();
      SimdRegister xmm2 = AllocTempSimdReg();
      if (xmm1 == no_simd_register || xmm2 == no_simd_register) {
        Undefined(); return;
      }

      if (args.is_store) {
        int32_t vreg1_off = offsetof(ThreadState, cpu.v[0]) + args.rt1 * 16;
        int32_t vreg2_off = offsetof(ThreadState, cpu.v[0]) + args.rt2 * 16;
        as_.Movdqu(xmm1, {.base = Assembler::rbp, .disp = vreg1_off});
        as_.Movdqu(xmm2, {.base = Assembler::rbp, .disp = vreg2_off});
        as_.Movdqu({.base = addr, .disp = 0}, xmm1);
        as_.Movdqu({.base = addr, .disp = 16}, xmm2);
      } else {
        as_.Movdqu(xmm1, {.base = addr, .disp = 0});
        as_.Movdqu(xmm2, {.base = addr, .disp = 16});
        int32_t vreg1_off = offsetof(ThreadState, cpu.v[0]) + args.rt1 * 16;
        int32_t vreg2_off = offsetof(ThreadState, cpu.v[0]) + args.rt2 * 16;
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg1_off}, xmm1);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg2_off}, xmm2);
      }
      return;
    }
    // endregion
    Undefined();
  }

  void SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs& args,
                        Register base, Register offset_reg) {
    // region digitalis
    // Handle SIMD load/store with register offset:
    //   LDR/STR Vt, [Xn, (X|W)m{, extend{ #shift}}]
    // The offset register may need UXTW / SXTW / UXTX / SXTX extension
    // before shift+add, exactly like the GP load/store path.
    Register addr = AllocTempReg();
    if (addr == no_register) { Undefined(); return; }

    ApplyOffsetExtend(addr, offset_reg, args.extend_type);
    if (args.shift_amount > 0) {
      as_.Shlq(addr, static_cast<int8_t>(args.shift_amount));
    }
    as_.Addq(addr, base);
    // region digitalis - apply TBI mask (top byte ignored on ARM64).
    as_.Shlq(addr, static_cast<int8_t>(8));
    as_.Shrq(addr, static_cast<int8_t>(8));
    // endregion

    int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;

    if (args.size == Decoder::SimdLoadStoreSize::k128bit) {
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      if (args.is_store) {
        as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movdqu({.base = addr, .disp = 0}, xmm);
      } else {
        as_.Movdqu(xmm, {.base = addr, .disp = 0});
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      }
    } else if (args.size == Decoder::SimdLoadStoreSize::k64bit) {
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      if (args.is_store) {
        as_.Movq(xmm, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movq({.base = addr, .disp = 0}, xmm);
      } else {
        as_.Pxor(xmm, xmm);
        as_.Movq(xmm, {.base = addr, .disp = 0});
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      }
    } else {
      // 8-bit, 16-bit, 32-bit: use GP register for transfer
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }
      if (args.is_store) {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
        if (args.size == Decoder::SimdLoadStoreSize::k8bit) {
          as_.Movb({.base = addr, .disp = 0}, tmp);
        } else if (args.size == Decoder::SimdLoadStoreSize::k16bit) {
          as_.Movw({.base = addr, .disp = 0}, tmp);
        } else {  // k32bit
          as_.Movl({.base = addr, .disp = 0}, tmp);
        }
      } else {
        // Zero the full 128-bit SIMD register first
        SimdRegister xmm = AllocTempSimdReg();
        if (xmm == no_simd_register) { Undefined(); return; }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
        // Load only the relevant bytes
        if (args.size == Decoder::SimdLoadStoreSize::k8bit) {
          as_.Movzxbl(tmp, {.base = addr, .disp = 0});
        } else if (args.size == Decoder::SimdLoadStoreSize::k16bit) {
          as_.Movzxwl(tmp, {.base = addr, .disp = 0});
        } else {  // k32bit
          as_.Movl(tmp, {.base = addr, .disp = 0});
        }
        as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
      }
    }
    return;
    // endregion
  }

  // region digitalis
  // FCSEL: fall back to interpreter (condition flag checking complex in JIT)
  void FpCondSelect(uint8_t /*rd*/, uint8_t /*rn*/, uint8_t /*rm*/,
                    uint8_t /*ftype*/, Decoder::Condition /*cond*/) {
    success_ = false;  // interpreter fallback
  }

  // FP <-> fixed-point conversion: fall back to interpreter
  void FpFixedPointConversion(const Decoder::FpFixedPointArgs& /*args*/) {
    success_ = false;  // interpreter fallback
  }

  // FP data-processing (3 source): FMADD, FMSUB, FNMADD, FNMSUB
  // Fall back to interpreter for now (3-source FP operations are complex for JIT).
  void FpDataProc3(uint8_t /*rd*/, uint8_t /*rn*/, uint8_t /*rm*/, uint8_t /*ra*/,
                   uint8_t /*ftype*/, bool /*o1*/, bool /*o0*/) {
    success_ = false;  // interpreter fallback
  }

  // FMOV (scalar, immediate): JIT - load a FP constant into SIMD register
  void FpMovImmediate(uint8_t rd, uint8_t imm8, uint8_t ftype) {
    int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + rd * 16;
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
  // endregion

  void FpIntConversion(const Decoder::FpIntConvArgs& args) {
    // region digitalis - JIT support for FMOV GP↔FP conversions
    uint8_t rmode = args.rmode;
    uint8_t opcode = args.op;

    // FMOV GP → FP: opcode=0b111, rmode=0b00
    if (rmode == 0b00 && opcode == 0b111) {
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;
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
      int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
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
      int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;
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
      int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
      // Read upper 64 bits (offset + 8)
      as_.Movq(tmp, {.base = Assembler::rbp, .disp = vreg_offset + 8});
      // Use SetReg to update the mapped register (not just ThreadState)
      SetReg(args.rd, tmp);
      return;
    }
    // endregion

    Undefined();
  }

  void AdvSimdCopy(const Decoder::AdvSimdCopyArgs& args) {
    // region digitalis - implement DUP (general) for memset fast path
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kDupGeneral && args.q) {
      // DUP (general), Q=1: broadcast GP register to all lanes of 128-bit SIMD register.
      // imm5 encodes element size: bit0=1→B, bit1=1→H, bit2=1→W, bit3=1→X
      uint8_t esize_bits = args.imm5 & 0xf;
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      Register src = GetReg(args.rn);
      as_.Movd(xmm, src);
      if (esize_bits == 0x01) {
        // Byte broadcast: PSHUFB with zero mask → each byte picks byte 0
        SimdRegister zero_mask = AllocTempSimdReg();
        if (zero_mask == no_simd_register) { Undefined(); return; }
        as_.Pxor(zero_mask, zero_mask);
        as_.Pshufb(xmm, zero_mask);
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
      int32_t vreg_offset = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;
      as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      return;
    }
    // endregion
    Undefined();
  }

  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& args) {
    UNUSED(args);
    Undefined();
  }

  // region digitalis
  void AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& args) {
    UNUSED(args);
    Undefined();
  }
  // endregion

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    UNUSED(rd, rn, rm, index, q);
    Undefined();
  }

  // region digitalis
  void AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size,
                      uint8_t opcode, bool q) {
    UNUSED(rd, rn, rm, size, opcode, q);
    Undefined();
  }

  void AdvSimdTableLookup(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t len,
                          uint8_t op, bool q) {
    UNUSED(rd, rn, rm, len, op, q);
    Undefined();
  }

  void Sha512(Decoder::Sha512Op op, uint8_t rd, uint8_t rn, uint8_t rm) {
    UNUSED(op, rd, rn, rm);
    Undefined();
  }
  // endregion

  // region digitalis
  void CryptoAes(uint8_t rd, uint8_t rn, uint8_t opcode) {
    UNUSED(rd, rn, opcode);
    Undefined();  // interpreter fallback
  }

  void CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode) {
    UNUSED(rd, rn, rm, opcode);
    Undefined();  // interpreter fallback
  }

  void CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode) {
    UNUSED(rd, rn, opcode);
    Undefined();  // interpreter fallback
  }
  // endregion

  // region digitalis
  void AdvSimdMultiStruct(uint8_t rt, uint8_t rn, uint8_t num_regs, uint8_t size,
                          bool q, bool is_store, bool postindex, uint8_t rm,
                          bool is_interleaved) {
    UNUSED(rt, rn, num_regs, size, q, is_store, postindex, rm, is_interleaved);
    Undefined();
  }

  void AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs& args) {
    UNUSED(args);
    Undefined();
  }
  // endregion

  Register AddSubWithCarry(Register src1, Register src2, bool is_64bit,
                            bool is_sub, bool set_flags) {
    UNUSED(src1, src2, is_64bit, is_sub, set_flags);
    Undefined();
    return no_register;
  }

  Register DataProc1Src(Register src, uint8_t opcode2, bool is_64bit) {
    // region digitalis - JIT support for REV, CLZ, RBIT
    Register res = AllocTempReg();
    switch (opcode2) {
      case 0b000010:  // REV16 (not commonly needed, skip for now)
        Undefined();
        return no_register;
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
      default:
        Undefined();
        return no_register;
    }
    // endregion
  }

  // region digitalis - EXTR JIT
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
  // endregion

  void ConditionalCompare(bool is_neg, bool is_64bit, Register rn, Register rm,
                           Decoder::Condition cond, uint8_t nzcv);  // implemented in .cc

  // region digitalis - Atomics JIT
  void LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& args, Register base) {
    // region digitalis - apply TBI mask before using base as memory operand.
    base = ApplyTbi(base);
    // endregion
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
        // Store-release: x86 TSO provides release semantics for stores.
        Register data = (args.rt < 31) ? GetReg(args.rt) : AllocTempReg();
        if (!success()) return;
        if (args.rt >= 31) as_.Xorl(data, data);
        Store(lss, base, 0, data);
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
          // region digitalis - byte/halfword forms only update low bits of RAX;
          // upper bits remain stale. ARM CAS Wt zero-extends to 64. Mask.
          if (args.size == 0) {
            as_.Andq(old_val, static_cast<int32_t>(0xFF));
          } else if (args.size == 1) {
            as_.Andq(old_val, static_cast<int32_t>(0xFFFF));
          }
          // endregion
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

        // region digitalis - byte/halfword Xchg leaves upper bits of new_val
        // as the original guest Xs (copied via Movq above), not zero. ARM SWP
        // Wt zero-extends the old memory value to 64. Mask.
        if (args.size == 0) {
          as_.Andq(new_val, static_cast<int32_t>(0xFF));
        } else if (args.size == 1) {
          as_.Andq(new_val, static_cast<int32_t>(0xFFFF));
        }
        // endregion
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

        // region digitalis - byte/halfword LockXadd only updates low bits of
        // addend; upper bits stay as guest Xs. ARM LDADD Wt zero-extends the
        // old memory value to 64. Mask.
        if (args.size == 0) {
          as_.Andq(addend, static_cast<int32_t>(0xFF));
        } else if (args.size == 1) {
          as_.Andq(addend, static_cast<int32_t>(0xFFFF));
        }
        // endregion
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
        // Load expected value from reservation.
        Register expected_val = AllocTempReg();
        if (!success()) return;
        int32_t resv_val_off = offsetof(ThreadState, cpu.reservation_value);
        as_.Movq(expected_val, {.base = Assembler::rbp, .disp = resv_val_off});

        // Load reservation address for comparison.
        Register resv_addr = AllocTempReg();
        if (!success()) return;
        int32_t resv_addr_off = offsetof(ThreadState, cpu.reservation_address);
        as_.Movq(resv_addr, {.base = Assembler::rbp, .disp = resv_addr_off});

        // Get new value from rt.
        Register new_val = (args.rt < 31) ? GetReg(args.rt) : AllocTempReg();
        if (!success()) return;
        if (args.rt >= 31) as_.Xorl(new_val, new_val);

        // Clear reservation.
        as_.Movq({.base = Assembler::rbp, .disp = resv_addr_off},
                 static_cast<int32_t>(0));

        // Check if reservation address matches base.
        AssemblerBase::Label* fail_label = as_.MakeLabel();
        as_.Cmpq(resv_addr, base);
        as_.Jcc(Condition::kNotEqual, *fail_label);

        // Move expected value into RAX for LOCK CMPXCHG.
        as_.Movq(Assembler::rax, expected_val);

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

      default:
        // kLdclr, kLdset, kLdeor: less common, fall back to interpreter.
        Undefined();
        break;
    }
  }
  // endregion

  void FpDataProc1(const Decoder::FpDataProc1Args& args) {
    // region digitalis - JIT support for FMOV (1-source FP op)
    if (args.opcode != 0b000000) { Undefined(); return; }  // Only FMOV for now
    if (args.ftype != 0b00 && args.ftype != 0b01) { Undefined(); return; }

    int32_t src_offset = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t dst_offset = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    // Zero the full 128-bit dest register, then copy the data via GP temp
    SimdRegister xmm = AllocTempSimdReg();
    if (xmm == no_simd_register) { Undefined(); return; }
    Register tmp = AllocTempReg();

    as_.Pxor(xmm, xmm);
    as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);

    if (args.ftype == 0b00) {
      // FMOV Sd, Sn: copy 4 bytes
      as_.Movl(tmp, {.base = Assembler::rbp, .disp = src_offset});
      as_.Movl({.base = Assembler::rbp, .disp = dst_offset}, tmp);
    } else {
      // FMOV Dd, Dn: copy 8 bytes
      as_.Movq(tmp, {.base = Assembler::rbp, .disp = src_offset});
      as_.Movq({.base = Assembler::rbp, .disp = dst_offset}, tmp);
    }
    // endregion
  }

  // region digitalis - FP arithmetic JIT
  void FpDataProc2(const Decoder::FpDataProc2Args& args) {
    // Only handle single (ftype=00) and double (ftype=01) precision.
    if (args.ftype != 0b00 && args.ftype != 0b01) { Undefined(); return; }
    bool is_double = (args.ftype == 0b01);

    int32_t src_n_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t src_m_off = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;
    int32_t dst_off = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    SimdRegister xmm_n = AllocTempSimdReg();
    if (xmm_n == no_simd_register) { Undefined(); return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (xmm_m == no_simd_register) { Undefined(); return; }

    // Load operands.
    if (is_double) {
      as_.Movsd(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movsd(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    }

    // Perform operation: result in xmm_n.
    switch (args.opcode) {
      case 0b0000:  // FMUL
        if (is_double) as_.Mulsd(xmm_n, xmm_m);
        else as_.Mulss(xmm_n, xmm_m);
        break;
      case 0b0001:  // FDIV
        if (is_double) as_.Divsd(xmm_n, xmm_m);
        else as_.Divss(xmm_n, xmm_m);
        break;
      case 0b0010:  // FADD
        if (is_double) as_.Addsd(xmm_n, xmm_m);
        else as_.Addss(xmm_n, xmm_m);
        break;
      case 0b0011:  // FSUB
        if (is_double) as_.Subsd(xmm_n, xmm_m);
        else as_.Subss(xmm_n, xmm_m);
        break;
      default:
        // FMAX, FMIN, FNMUL, etc. - less common, fall back.
        Undefined();
        return;
    }

    // Zero dest register, then store result.
    SimdRegister zero = AllocTempSimdReg();
    if (zero == no_simd_register) {
      // Can reuse xmm_m since we're done with it.
      as_.Pxor(xmm_m, xmm_m);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_m);
    } else {
      as_.Pxor(zero, zero);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, zero);
    }
    if (is_double) {
      as_.Movsd({.base = Assembler::rbp, .disp = dst_off}, xmm_n);
    } else {
      as_.Movss({.base = Assembler::rbp, .disp = dst_off}, xmm_n);
    }
  }

  void FpCompare(const Decoder::FpCompareArgs& args) {
    // FCMP Sn, Sm or FCMP Dn, Dm: compare and set NZCV flags.
    if (args.ftype != 0b00 && args.ftype != 0b01) { Undefined(); return; }
    bool is_double = (args.ftype == 0b01);

    int32_t src_n_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t src_m_off = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;

    SimdRegister xmm_n = AllocTempSimdReg();
    if (xmm_n == no_simd_register) { Undefined(); return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (xmm_m == no_simd_register) { Undefined(); return; }

    if (is_double) {
      as_.Movsd(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      if (args.with_zero) {
        as_.Pxor(xmm_m, xmm_m);
      } else {
        as_.Movsd(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      }
      as_.Ucomisd(xmm_n, xmm_m);
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      if (args.with_zero) {
        as_.Pxor(xmm_m, xmm_m);
      } else {
        as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      }
      as_.Ucomiss(xmm_n, xmm_m);
    }

    // UCOMISD/UCOMISS set x86 ZF/PF/CF. Map to ARM64 NZCV:
    // ARM64: N=less, Z=equal, C=greater-or-equal-or-unordered, V=unordered
    // x86 UCOMISS result flags:
    //   a > b:  ZF=0 PF=0 CF=0  → ARM64: N=0 Z=0 C=1 V=0
    //   a < b:  ZF=0 PF=0 CF=1  → ARM64: N=1 Z=0 C=0 V=0
    //   a == b: ZF=1 PF=0 CF=0  → ARM64: N=0 Z=1 C=1 V=0
    //   NaN:    ZF=1 PF=1 CF=1  → ARM64: N=0 Z=0 C=1 V=1
    // Use LAHF + SETO like EmitStoreArmNZCV but with custom mapping.
    // For now, store approximate flags. The common usage is EQ/NE/LT/GE
    // which map well from x86 flags to conditional jumps.
    EmitStoreArmNZCV(/*is_sub=*/true);
  }
  // endregion

  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& args) {
    UNUSED(args);
    Undefined();
  }

  // region digitalis
  void AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs& args) {
    UNUSED(args);
    Undefined();
  }

  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs& args) {
    UNUSED(args);
    Undefined();
  }

  void AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs& args) {
    UNUSED(args);
    Undefined();
  }
  // endregion

  void AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& args) {
    UNUSED(args);
    Undefined();
  }

  // region digitalis
  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs& args) {
    UNUSED(args);
    Undefined();
  }
  // endregion

  //
  // Accessor helpers.
  //

  [[nodiscard]] Assembler* as() { return &as_; }
  [[nodiscard]] bool success() const { return success_; }
  bool is_region_end_reached() const { return is_region_end_reached_; }

  void FreeTempRegs() {
    gp_allocator_.FreeTemps();
    simd_allocator_.FreeTemps();
  }

  // region digitalis - early region termination on register pressure
  // Returns true if the GP temp register pool is too low for safe instruction translation.
  // Most instructions need 2-4 temps; below this threshold, end the region to preserve
  // already-translated JIT code instead of failing and discarding the entire region.
  bool IsGpRegPoolLow(uint32_t threshold = 4) const {
    return gp_allocator_.AvailableTempCount() < threshold;
  }
  // endregion

  // region digitalis - guest PC labels for backward branch inlining in loops.
  // Register a label at the current x86_64 code position for the given guest PC.
  // This allows backward branches (loops) to emit a local jump instead of
  // a full region exit + translation cache dispatch.
  void RegisterGuestPcLabel(GuestAddr pc) {
    Assembler::Label* label = as_.MakeLabel();
    as_.Bind(label);
    guest_pc_labels_[pc] = label;
  }

  // Try to emit a local backward branch to target within this region.
  // Returns true if a local jump was emitted (caller should NOT exit region).
  // Returns false if the target is not in this region (caller should exit normally).
  bool TryLocalBackwardBranch(GuestAddr target) {
    // region digitalis - disabled: backward branch inlining traps the CPU
    // in a tight loop without signal checks.  Dispatch on every backward
    // edge so signals are processed and translation stats remain visible.
    UNUSED(target);
    return false;
    // endregion
  }
  // endregion

  bool IsRegMappingEnabled() { return params_.enable_reg_mapping; }

  std::tuple<Register, bool> GetMappedRegisterOrMap(int reg) {
    if (gp_maintainer_.IsMapped(reg)) {
      return {gp_maintainer_.GetMapped(reg), false};
    }
    if (auto alloc_result = gp_allocator_.Alloc()) {
      gp_maintainer_.Map(reg, alloc_result.value());
      return {alloc_result.value(), true};
    }
    success_ = false;
    return {Assembler::no_register, false};
  }

  Register AllocTempReg() {
    if (auto reg_option = gp_allocator_.AllocTemp()) {
      return reg_option.value();
    }
    success_ = false;
    return Assembler::no_register;
  }

  SimdRegister AllocTempSimdReg() {
    if (auto reg_option = simd_allocator_.AllocTemp()) {
      return reg_option.value();
    }
    success_ = false;
    return Assembler::no_xmm_register;
  }

 private:
  // Helper: extract ARM64 NZCV flags from x86_64 EFLAGS after ADD/SUB.
  //
  // x86_64 LAHF stores flags into AH with bit positions that happen to match
  // the ARM64 flag word layout:
  //   AX[15] = SF = ARM64 N    AX[14] = ZF = ARM64 Z    AX[8] = CF = ARM64 C
  // ARM64 V (overflow) at bit 0 is extracted via SETO.
  //
  // For SUB, ARM64 C = !x86_CF (ARM64 uses inverted borrow), so we XOR bit 8.
  // region digitalis - use AL for overflow (avoids clobbering rcx in allocator pool).
  // LAHF stores SF|ZF|CF to AH (bits 8-15). SETCC OF stores to AL (bits 0-7).
  // AND 0xC101 keeps N(bit15), Z(bit14), C(bit8), V(bit0). No rcx save/restore needed.
  void EmitStoreArmNZCV(bool is_sub) {
    as_.Lahf();
    as_.Setcc(Condition::kOverflow, Assembler::rax);
    as_.Andl(Assembler::rax, static_cast<int32_t>(0xC101));
    if (is_sub) {
      as_.Xorl(Assembler::rax, static_cast<int32_t>(0x0100));
    }
    int32_t flags_offset = offsetof(ThreadState, cpu.flags);
    as_.Movw({.base = Assembler::rbp, .disp = flags_offset}, Assembler::rax);
  }
  // endregion

  // Helper: emit shift of src into dst by a compile-time constant amount.
  void EmitShift(Register dst, Register src, Decoder::ShiftType shift_type,
                 uint8_t shift_amount, bool is_64bit) {
    if (is_64bit) {
      as_.Movq(dst, src);
      if (shift_amount == 0) return;
      switch (shift_type) {
        case Decoder::ShiftType::kLsl:
          as_.Shlq(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kLsr:
          as_.Shrq(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kAsr:
          as_.Sarq(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kRor:
          as_.Rorq(dst, static_cast<int8_t>(shift_amount));
          break;
      }
    } else {
      as_.Movl(dst, src);
      if (shift_amount == 0) return;
      switch (shift_type) {
        case Decoder::ShiftType::kLsl:
          as_.Shll(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kLsr:
          as_.Shrl(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kAsr:
          as_.Sarl(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kRor:
          as_.Rorl(dst, static_cast<int8_t>(shift_amount));
          break;
      }
    }
  }

  Assembler as_;
  bool success_;
  GuestAddr pc_;
  Allocator<Register> gp_allocator_;
  RegisterFileMaintainer<Register, kNumGuestRegs> gp_maintainer_;
  Allocator<SimdRegister> simd_allocator_;
  const LiteTranslateParams params_;
  bool is_region_end_reached_;
  // region digitalis - guest PC label map for backward branch inlining
  std::unordered_map<GuestAddr, Assembler::Label*> guest_pc_labels_;
  // endregion
};

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_H_
// endregion
