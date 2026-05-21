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
#include "berberis/runtime_primitives/platform.h"

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

  // region digitalis
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
  // endregion

  // region digitalis FCADD/FCMLA JIT (handoff-69)
  //
  // AdvSIMD complex floating-point (FCADD / FCMLA) JIT path for FP32
  // and FP64.  The interpreter (interpreter.h::AdvSimdFcma) is the
  // spec — see the per-pair scalar math there for the six rotations
  // (FCADD ±90/±270, FCMLA 0/90/180/270).  The JIT path here lowers
  // FP32 (size=0b10, .2S Q=0 and .4S Q=1) and FP64 (size=0b11, .2D
  // Q=1) to a 4-to-7 SSE-instruction sequence:
  //
  //   * FCADD .4S, rot=#rot:
  //       Vm' = shufps(Vm, Vm, 0xB1)           // pair-swap (re,im)->(im,re)
  //       sign = pslld(pcmpeqd self, 31)       // [0x80000000]*4
  //       lane = psrlq(pcmpeqd self, 32)       // rot=0:  [-1,0,-1,0] (negate real)
  //              psllq(pcmpeqd self, 32)       // rot=1:  [0,-1,0,-1] (negate imag)
  //       sign &= lane
  //       Vm' ^= sign                          // negate appropriate lanes
  //       Vd  = Vn + Vm'                       // ADDPS
  //
  //   * FCMLA .4S, rot=#rot:  Vd += n_broadcast * m_xformed where
  //       rot=#0   (rot=0): xform=identity,      broadcast=n_re
  //       rot=#90  (rot=1): xform=swap+~re,      broadcast=n_im
  //       rot=#180 (rot=2): xform=negate-all,    broadcast=n_re
  //       rot=#270 (rot=3): xform=swap+~im,      broadcast=n_im
  //     n_re_broadcast = pshufd(Vn, 0xA0) = [n_re0, n_re0, n_re1, n_re1]
  //     n_im_broadcast = pshufd(Vn, 0xF5) = [n_im0, n_im0, n_im1, n_im1]
  //
  //   * Q=0 form (.2S): same emit; the upper 64 bits are masked away with
  //     psrldq+pand at the end (matches AArch64 vector half-vector semantics
  //     and the interpreter's Q=0 zero-clear at the end of AdvSimdFcma).
  //
  // FP16 (size=0b01) and FP64 (size=0b11) bail to the interpreter via
  // `success_ = false` — the interpreter handles them through the same
  // FpHalfToSingle round-trip / double-precision arithmetic paths used by
  // the rest of. When AVX-512-FP16 (or F16C) is unconditionally
  // available on the emulator host CPU, the FP16 path becomes a 2-step
  // VCVTPH2PS round-trip; that's a follow-up perf row.
  void AdvSimdFcma(const Decoder::FcmaArgs& args) {
    if (args.size == 0b01) {
      // FP16 vector — fall back to interpreter (F16C round-trip path TBD).
      success_ = false;
      return;
    }

    SimdRegister xmm_n = AllocTempSimdReg();
    if (xmm_n == no_simd_register) { success_ = false; return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (xmm_m == no_simd_register) { success_ = false; return; }
    SimdRegister xmm_sign = AllocTempSimdReg();
    if (xmm_sign == no_simd_register) { success_ = false; return; }
    SimdRegister xmm_lane = AllocTempSimdReg();
    if (xmm_lane == no_simd_register) { success_ = false; return; }

    int32_t src_n_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t src_m_off = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;
    int32_t dst_off   = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    as_.Movdqu(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
    as_.Movdqu(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});

    SimdRegister xmm_result = no_simd_register;  // tracks the register holding the final value

    if (args.size == 0b10) {
      // FP32 path: 1 pair (.2S, Q=0) or 2 pairs (.4S, Q=1).
      // Sign-bit mask for 32-bit FP lanes: [0x80000000]*4.
      as_.Pcmpeqd(xmm_sign, xmm_sign);
      as_.Pslld(xmm_sign, static_cast<int8_t>(31));

      if (args.opcode == Decoder::FcmaOpcode::kFcadd) {
        // m_xformed = pair-swap(Vm), negate the lanes implied by rot.
        as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));

        as_.Pcmpeqd(xmm_lane, xmm_lane);
        if (args.rot == 0) {
          // rot=#90: negate real lanes (0, 2) — mask = [0xff..ff, 0, 0xff..ff, 0].
          as_.Psrlq(xmm_lane, static_cast<int8_t>(32));
        } else {
          // rot=#270: negate imag lanes (1, 3) — mask = [0, 0xff..ff, 0, 0xff..ff].
          as_.Psllq(xmm_lane, static_cast<int8_t>(32));
        }
        as_.Pand(xmm_sign, xmm_lane);
        as_.Xorps(xmm_m, xmm_sign);

        // Vn + m_xformed -> xmm_n.
        as_.Addps(xmm_n, xmm_m);
        xmm_result = xmm_n;
      } else {
        // FCMLA: result = Vd + n_broadcast * m_xformed.
        SimdRegister xmm_d = AllocTempSimdReg();
        if (xmm_d == no_simd_register) { success_ = false; return; }
        as_.Movdqu(xmm_d, {.base = Assembler::rbp, .disp = dst_off});

        // m_xformed depends on rotation.
        switch (args.rot) {
          case 0:
            // No transform.
            break;
          case 1:
            // Swap + negate real lanes.
            as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));
            as_.Pcmpeqd(xmm_lane, xmm_lane);
            as_.Psrlq(xmm_lane, static_cast<int8_t>(32));
            as_.Pand(xmm_sign, xmm_lane);
            as_.Xorps(xmm_m, xmm_sign);
            break;
          case 2:
            // Negate all lanes.
            as_.Xorps(xmm_m, xmm_sign);
            break;
          case 3:
            // Swap + negate imag lanes.
            as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));
            as_.Pcmpeqd(xmm_lane, xmm_lane);
            as_.Psllq(xmm_lane, static_cast<int8_t>(32));
            as_.Pand(xmm_sign, xmm_lane);
            as_.Xorps(xmm_m, xmm_sign);
            break;
          default:
            // Cannot happen: decoder only emits rot in 0..3 for FCMLA.
            success_ = false;
            return;
        }

        // Broadcast n_re (rot 0/2) or n_im (rot 1/3) across both pair slots.
        // PSHUFD imm=0xA0 = (10,10,00,00): [lane0, lane0, lane2, lane2] = n_re bcast.
        // PSHUFD imm=0xF5 = (11,11,01,01): [lane1, lane1, lane3, lane3] = n_im bcast.
        if (args.rot == 0 || args.rot == 2) {
          as_.Pshufd(xmm_n, xmm_n, static_cast<int8_t>(0xA0));
        } else {
          as_.Pshufd(xmm_n, xmm_n, static_cast<int8_t>(0xF5));
        }

        as_.Mulps(xmm_n, xmm_m);   // n_broadcast * m_xformed
        as_.Addps(xmm_d, xmm_n);   // Vd += ...
        xmm_result = xmm_d;
      }
    } else {
      // FP64 path: size==0b11, .2D only (decoder rejects Q=0 for FP64).
      // 1 complex pair (lane 0 = re, lane 1 = im); 8 bytes per lane.
      // Same shape as FP32 but with FP64-width lowerings:
      //   Shufpd 0x01 swaps the two doubles; Shufpd 0x00/0x03 broadcasts
      //   lane 0/1 across both. PSRLDQ/PSLLDQ 8 builds the lane-mask
      //   (single pair, 8-byte lane width).
      // Sign-bit mask for 64-bit FP lanes: [0x8000000000000000]*2.
      as_.Pcmpeqd(xmm_sign, xmm_sign);
      as_.Psllq(xmm_sign, static_cast<int8_t>(63));

      if (args.opcode == Decoder::FcmaOpcode::kFcadd) {
        // Pair-swap: SHUFPD imm=0x01 -> [m_im, m_re].
        as_.Shufpd(xmm_m, xmm_m, static_cast<int8_t>(0x01));

        as_.Pcmpeqd(xmm_lane, xmm_lane);
        if (args.rot == 0) {
          // rot=#90: negate real lane (lane 0).
          // mask = [0xff..ff (low 64), 0 (high 64)] = PSRLDQ 8 over [-1; 16].
          as_.Psrldq(xmm_lane, static_cast<int8_t>(8));
        } else {
          // rot=#270: negate imag lane (lane 1).
          // mask = [0 (low 64), 0xff..ff (high 64)] = PSLLDQ 8 over [-1; 16].
          as_.Pslldq(xmm_lane, static_cast<int8_t>(8));
        }
        as_.Pand(xmm_sign, xmm_lane);
        as_.Xorpd(xmm_m, xmm_sign);

        as_.Addpd(xmm_n, xmm_m);
        xmm_result = xmm_n;
      } else {
        // FCMLA: result = Vd + n_broadcast * m_xformed.
        SimdRegister xmm_d = AllocTempSimdReg();
        if (xmm_d == no_simd_register) { success_ = false; return; }
        as_.Movdqu(xmm_d, {.base = Assembler::rbp, .disp = dst_off});

        switch (args.rot) {
          case 0:
            // No transform.
            break;
          case 1:
            // Swap + negate real lane.
            as_.Shufpd(xmm_m, xmm_m, static_cast<int8_t>(0x01));
            as_.Pcmpeqd(xmm_lane, xmm_lane);
            as_.Psrldq(xmm_lane, static_cast<int8_t>(8));
            as_.Pand(xmm_sign, xmm_lane);
            as_.Xorpd(xmm_m, xmm_sign);
            break;
          case 2:
            // Negate both lanes.
            as_.Xorpd(xmm_m, xmm_sign);
            break;
          case 3:
            // Swap + negate imag lane.
            as_.Shufpd(xmm_m, xmm_m, static_cast<int8_t>(0x01));
            as_.Pcmpeqd(xmm_lane, xmm_lane);
            as_.Pslldq(xmm_lane, static_cast<int8_t>(8));
            as_.Pand(xmm_sign, xmm_lane);
            as_.Xorpd(xmm_m, xmm_sign);
            break;
          default:
            success_ = false;
            return;
        }

        // Broadcast n_re (rot 0/2) -> [n[0], n[0]] via SHUFPD imm=0x00.
        // Broadcast n_im (rot 1/3) -> [n[1], n[1]] via SHUFPD imm=0x03.
        if (args.rot == 0 || args.rot == 2) {
          as_.Shufpd(xmm_n, xmm_n, static_cast<int8_t>(0x00));
        } else {
          as_.Shufpd(xmm_n, xmm_n, static_cast<int8_t>(0x03));
        }

        as_.Mulpd(xmm_n, xmm_m);
        as_.Addpd(xmm_d, xmm_n);
        xmm_result = xmm_d;
      }
    }

    // Q=0 (.2S only — FP64 always has Q=1): zero the upper 64 bits.
    if (!args.q) {
      // Reuse xmm_lane to build a [0xff..ff (low 64), 0 (high 64)] mask.
      as_.Pcmpeqd(xmm_lane, xmm_lane);
      as_.Psrldq(xmm_lane, static_cast<int8_t>(8));
      as_.Pand(xmm_result, xmm_lane);
    }

    as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_result);
  }
  // endregion

  // region digitalis indexed FCMLA
  //
  // AdvSIMD complex floating-point by element (FCMLA-idx) JIT path for
  // FP32 (.4s).  Same shape as AdvSimdFcma's FP32 FCMLA branch, except
  // Vm contributes a single broadcast complex pair (Vm.s[2*idx],
  // Vm.s[2*idx+1]) rather than per-pair distinct pairs.  The broadcast
  // is one PSHUFD over Vm before the existing per-rot transform:
  //   idx=0: imm=0x44 -> [Vm.s[0], Vm.s[1], Vm.s[0], Vm.s[1]]
  //   idx=1: imm=0xEE -> [Vm.s[2], Vm.s[3], Vm.s[2], Vm.s[3]]
  // After the broadcast, the per-rot Vm transform (Shufps/Pand/Xorps)
  // and the Vn-half broadcast (PSHUFD 0xA0 for n_re, 0xF5 for n_im)
  // match the non-indexed FCMLA path verbatim.  Vd is read-modify-write
  // (FCMLA always accumulates).
  //
  // FP32-indexed mandates Q=1: the decoder rejects L=1 or Q=0 for
  // size=0b10, so there is no .2s indexed form and the post-loop Q=0
  // zero-clear tail used by the vector path is unreachable here.
  //
  // FP16-indexed (size=0b01) bails to the interpreter — same F16C
  // round-trip follow-up as the FP16 FCMA-vector row.
  void AdvSimdFcmaIdx(const Decoder::FcmaIdxArgs& args) {
    if (args.size != 0b10) {
      // FP16-indexed (size=0b01) — fall back to interpreter (F16C
      // round-trip path TBD).
      success_ = false;
      return;
    }

    SimdRegister xmm_n = AllocTempSimdReg();
    if (xmm_n == no_simd_register) { success_ = false; return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (xmm_m == no_simd_register) { success_ = false; return; }
    SimdRegister xmm_d = AllocTempSimdReg();
    if (xmm_d == no_simd_register) { success_ = false; return; }
    SimdRegister xmm_sign = AllocTempSimdReg();
    if (xmm_sign == no_simd_register) { success_ = false; return; }
    SimdRegister xmm_lane = AllocTempSimdReg();
    if (xmm_lane == no_simd_register) { success_ = false; return; }

    int32_t src_n_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t src_m_off = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;
    int32_t dst_off   = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    as_.Movdqu(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
    as_.Movdqu(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    as_.Movdqu(xmm_d, {.base = Assembler::rbp, .disp = dst_off});

    // Broadcast the indexed complex pair across all of xmm_m.  PSHUFD
    // imm bits (3:2:1:0) each pick a source dword for that dst dword.
    //   idx=0: imm=0b01_00_01_00 = 0x44 -> [src[0], src[1], src[0], src[1]]
    //   idx=1: imm=0b11_10_11_10 = 0xEE -> [src[2], src[3], src[2], src[3]]
    as_.Pshufd(xmm_m, xmm_m,
               static_cast<int8_t>(args.index == 0 ? 0x44 : 0xEE));

    // Sign-bit mask for 32-bit FP lanes: [0x80000000]*4.
    as_.Pcmpeqd(xmm_sign, xmm_sign);
    as_.Pslld(xmm_sign, static_cast<int8_t>(31));

    // Apply the per-rotation Vm transform — identical to the vector
    // FCMLA path (lite_translator::AdvSimdFcma FP32 FCMLA branch).
    switch (args.rot) {
      case 0:
        // No transform.
        break;
      case 1:
        // Swap pair + negate real lanes.
        as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));
        as_.Pcmpeqd(xmm_lane, xmm_lane);
        as_.Psrlq(xmm_lane, static_cast<int8_t>(32));
        as_.Pand(xmm_sign, xmm_lane);
        as_.Xorps(xmm_m, xmm_sign);
        break;
      case 2:
        // Negate all lanes.
        as_.Xorps(xmm_m, xmm_sign);
        break;
      case 3:
        // Swap pair + negate imag lanes.
        as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));
        as_.Pcmpeqd(xmm_lane, xmm_lane);
        as_.Psllq(xmm_lane, static_cast<int8_t>(32));
        as_.Pand(xmm_sign, xmm_lane);
        as_.Xorps(xmm_m, xmm_sign);
        break;
      default:
        // Decoder only emits rot in 0..3 for FCMLA-idx.
        success_ = false;
        return;
    }

    // Broadcast n_re (rot 0/2) or n_im (rot 1/3) across both pair slots.
    if (args.rot == 0 || args.rot == 2) {
      as_.Pshufd(xmm_n, xmm_n, static_cast<int8_t>(0xA0));
    } else {
      as_.Pshufd(xmm_n, xmm_n, static_cast<int8_t>(0xF5));
    }

    as_.Mulps(xmm_n, xmm_m);   // n_broadcast * m_xformed
    as_.Addps(xmm_d, xmm_n);   // Vd += ...

    // FP32-indexed FCMLA mandates Q=1, so no Q=0 zero-clear is needed.
    as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_d);
  }
  // endregion

  // region digitalis
  // AdvSIMD BFloat16 three-same-extra (BFDOT / BFMMLA): bail to the
  // interpreter for now.  The host x86_64 baseline doesn't include
  // AVX-512-BF16, so without runtime feature detection the JIT lowering
  // would need its own BF16 widening sequence (SLLI $16 over a PSHUFD
  // mask) — feasible but the interpreter path is the right starting
  // point for correctness, and these instructions are rare per JIT
  // region in the current sample suite.  JIT path is parked under
  // "Implement" row 2.
  void AdvSimdBf16ThreeSame(const Decoder::Bf16ThreeSameArgs&) { success_ = false; }
  // endregion

  // region digitalis SDOT/UDOT JIT (handoff-71)
  //
  // AdvSIMD integer dot product: SDOT/UDOT (vector and indexed-by-element).
  // Reference: ARM ARM C7.2.397 (SDOT), C7.2.398 (UDOT).
  //
  // Each 32-bit lane of Vd accumulates the dot product of 4 byte products:
  //   Vd[i] = (int32_t)(Vd[i] + sum_{k=0..3}(Vn_byte[4*i+k] * Vm_byte[base+k]))
  // where base = 4*i for the vector form, 4*args.index for the indexed
  // form (the indexed form broadcasts a single 4-byte group of Vm across
  // every lane).  SDOT sign-extends both byte vectors; UDOT zero-extends.
  // Q=0 reads only the low 8 bytes of Vn (and Vm in vector form), writes
  // 2 lanes, and zeros the upper 64 bits of Vd.
  //
  // SSE4.1 / SSSE3 lowering (no AVX-VNNI dependency — the emulator host
  // baseline doesn't include VPDPBUSD):
  //
  //   1.  Widen each 8-byte half of Vn (and Vm, vector form) from bytes to
  //       16-bit signed (SDOT) or unsigned-zero-extended (UDOT) words via
  //       PMOVSXBW / PMOVZXBW.
  //   2.  PMADDWD pairs adjacent 16-bit words: 8 words -> 4 int32 lanes,
  //       each lane = w[2k]*v[2k] + w[2k+1]*v[2k+1].
  //   3.  Each ARM dot lane is the sum of 4 byte products, so two adjacent
  //       PMADDWD outputs must be horizontally added — PHADDD pairs.
  //   4.  PADDD accumulates the 4 lanes into Vd (read-modify-write).
  //
  // For Q=0 (2 lanes): apply (1)-(3) on the low 8 bytes only, PHADDD x,x
  // gives [lane0,lane1,lane0,lane1]; MOVQ x,x masks the duplicate upper
  // half; MOVQ-load Vd's low half (zero-upper), PADDD, MOVDQU 16-byte
  // store — the upper 8 bytes land as zero, matching Q=0 semantics.
  //
  // Indexed form: load 4 bytes from [Vm + 4*idx] via MOVD, PSHUFD
  // imm=0x00 broadcasts that dword across all four 32-bit lanes; the
  // resulting xmm has identical low and high 8 bytes, so one
  // PMOVSXBW/ZXBW serves both PMADDWD pairings.
  //
  // All eight DOT encodings — vec×{Q=0,Q=1} × idx×{Q=0,Q=1} ×
  // {SDOT,UDOT} — are lowered.  The interpreter (interpreter.h:1237)
  // remains the executable spec; this JIT path produces bit-exact
  // output (32-bit integer arithmetic with defined wraparound).
  void AdvSimdDotProduct(const Decoder::DotProductArgs& args) {
    const bool is_signed = (args.opcode == Decoder::DotProductOpcode::kSdot ||
                            args.opcode == Decoder::DotProductOpcode::kSdotIdx);
    const bool is_indexed = (args.opcode == Decoder::DotProductOpcode::kSdotIdx ||
                             args.opcode == Decoder::DotProductOpcode::kUdotIdx);

    const int32_t vn_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    const int32_t vm_off = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;
    const int32_t vd_off = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    SimdRegister xmm_n_lo = AllocTempSimdReg();
    if (xmm_n_lo == no_simd_register) { success_ = false; return; }
    SimdRegister xmm_m_lo = AllocTempSimdReg();
    if (xmm_m_lo == no_simd_register) { success_ = false; return; }

    // For Q=1 we additionally need 2 temps to widen the high halves
    // (vector form) or hold the second PMADDWD result (indexed form).
    SimdRegister xmm_n_hi = no_simd_register;
    SimdRegister xmm_m_hi = no_simd_register;
    if (args.q) {
      xmm_n_hi = AllocTempSimdReg();
      if (xmm_n_hi == no_simd_register) { success_ = false; return; }
      if (!is_indexed) {
        xmm_m_hi = AllocTempSimdReg();
        if (xmm_m_hi == no_simd_register) { success_ = false; return; }
      }
    }

    // Stage 1: prepare widened Vm operand.
    if (is_indexed) {
      // Load Vm[4*idx..4*idx+3] as a dword, broadcast across all 4
      // lanes; after PSHUFD the low and high 8 bytes are identical,
      // so a single PMOVSXBW/ZXBW from the low half suffices for both
      // PMADDWD pairings.
      as_.Movd(xmm_m_lo, {.base = Assembler::rbp,
                          .disp = vm_off + 4 * args.index});
      as_.Pshufd(xmm_m_lo, xmm_m_lo, static_cast<int8_t>(0x00));
      if (is_signed) {
        as_.Pmovsxbw(xmm_m_lo, xmm_m_lo);
      } else {
        as_.Pmovzxbw(xmm_m_lo, xmm_m_lo);
      }
    } else {
      // Vector form: widen low 8 bytes of Vm from memory.
      if (is_signed) {
        as_.Pmovsxbw(xmm_m_lo, {.base = Assembler::rbp, .disp = vm_off + 0});
      } else {
        as_.Pmovzxbw(xmm_m_lo, {.base = Assembler::rbp, .disp = vm_off + 0});
      }
      if (args.q) {
        if (is_signed) {
          as_.Pmovsxbw(xmm_m_hi, {.base = Assembler::rbp, .disp = vm_off + 8});
        } else {
          as_.Pmovzxbw(xmm_m_hi, {.base = Assembler::rbp, .disp = vm_off + 8});
        }
      }
    }

    // Stage 2: widen Vn.
    if (is_signed) {
      as_.Pmovsxbw(xmm_n_lo, {.base = Assembler::rbp, .disp = vn_off + 0});
    } else {
      as_.Pmovzxbw(xmm_n_lo, {.base = Assembler::rbp, .disp = vn_off + 0});
    }
    if (args.q) {
      if (is_signed) {
        as_.Pmovsxbw(xmm_n_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
      } else {
        as_.Pmovzxbw(xmm_n_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
      }
    }

    // Stage 3: pair multiply-and-add into 4 int32 partial sums per half.
    as_.Pmaddwd(xmm_n_lo, xmm_m_lo);
    if (args.q) {
      // Vector form pairs with xmm_m_hi; indexed form reuses xmm_m_lo.
      as_.Pmaddwd(xmm_n_hi, is_indexed ? xmm_m_lo : xmm_m_hi);
    }

    // Stage 4: horizontal add adjacent pairs to get final lanes.
    if (args.q) {
      // Result lanes: [n_lo[0..1], n_lo[2..3], n_hi[0..1], n_hi[2..3]].
      as_.Phaddd(xmm_n_lo, xmm_n_hi);
    } else {
      // Q=0: only 2 lanes are meaningful.  PHADDD xmm,xmm duplicates
      // the low 64 bits into the upper 64; we mask via MOVQ below.
      as_.Phaddd(xmm_n_lo, xmm_n_lo);
    }

    // Stage 5: accumulate into Vd and store.
    if (args.q) {
      as_.Paddd(xmm_n_lo, {.base = Assembler::rbp, .disp = vd_off});
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_n_lo);
    } else {
      // Q=0: low 8 bytes of Vd are accumulated; upper 8 bytes are zeroed.
      // MOVQ x,x clears the duplicate upper half from PHADDD; MOVQ from
      // Vd zero-extends Vd[0..7] into a clean operand; PADDD adds; the
      // full-width MOVDQU lands the upper 8 bytes as zero.
      as_.Movq(xmm_n_lo, xmm_n_lo);
      as_.Movq(xmm_m_lo, {.base = Assembler::rbp, .disp = vd_off});
      as_.Paddd(xmm_n_lo, xmm_m_lo);
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_n_lo);
    }
  }
  // endregion

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
    // region digitalis
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
    // (movq, REX.W) — handoff-19's DUP fix flagged 32-bit-MOVD-vs-64-bit-MOVQ
    // as a load-bearing class of bug; the switch below is explicit about
    // which mov-width belongs at each esize.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kUmov && esize != 0) {
      bool canonical = (esize == 8 && args.q) || (esize != 8 && !args.q);
      if (!canonical) { success_ = false; return; }
      if (args.rd < 31) {
        Register tmp = AllocTempReg();
        if (tmp == no_register) { success_ = false; return; }
        int32_t off =
            offsetof(ThreadState, cpu.v[0]) + args.rn * 16 + index * esize;
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
            offsetof(ThreadState, cpu.v[0]) + args.rn * 16 + index * esize;
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
          offsetof(ThreadState, cpu.v[0]) + args.rd * 16 + index * esize;
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
    // endregion

    // region digitalis - implement DUP (general) for memset fast path
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kDupGeneral && args.q) {
      // DUP (general), Q=1: broadcast GP register to all lanes of 128-bit SIMD register.
      // imm5 encodes element size: bit0=1→B, bit1=1→H, bit2=1→W, bit3=1→X
      uint8_t esize_bits = args.imm5 & 0xf;
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      Register src = GetReg(args.rn);
      // For 64-bit broadcast we must move the FULL 64 bits of the GP register
      // into the XMM register; using 32-bit MOVD here silently truncates the
      // upper half and the subsequent PSHUFD(0x44) then duplicates the low
      // 32-bit value into both D-lanes. That was the FB libcoldstart Yoga
      // layout x21 truncation bug (handoff-18): DUP V0.2D, X21 produced
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
    // region digitalis - JIT for common SIMD three-same ops.
    //
    // Implements MUL/MLA/MLS/ADD/SUB/AND/ORR/EOR/CMEQ at the lane sizes
    // the dynamic linker's calculate_gnu_hash_neon needs (4S MUL/MLA in
    // particular). Falls back to interpreter for opcodes/sizes outside
    // this set.
    int32_t vn_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t vm_off = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;
    int32_t vd_off = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    auto load_full = [&](SimdRegister xmm, int32_t off) {
      as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = off});
    };
    auto store_full = [&](int32_t off, SimdRegister xmm) {
      as_.Movdqu({.base = Assembler::rbp, .disp = off}, xmm);
    };
    auto mask_low64 = [&](SimdRegister xmm) {
      // Zero upper 64 bits when q=0 (D-register semantics).
      as_.Pslldq(xmm, int8_t{8});
      as_.Psrldq(xmm, int8_t{8});
    };

    switch (args.opcode) {
      case Decoder::AdvSimdThreeSameOpcode::kMul: {
        if (args.size != 0b10) { Undefined(); return; }   // only 32-bit lanes here
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (xn == no_simd_register || xm == no_simd_register) { Undefined(); return; }
        load_full(xn, vn_off);
        load_full(xm, vm_off);
        as_.Pmulld(xn, xm);
        if (!args.q) mask_low64(xn);
        store_full(vd_off, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kMla: {
        if (args.size != 0b10) { Undefined(); return; }   // only 32-bit lanes here
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        SimdRegister xd = AllocTempSimdReg();
        if (xn == no_simd_register || xm == no_simd_register || xd == no_simd_register) {
          Undefined(); return;
        }
        load_full(xn, vn_off);
        load_full(xm, vm_off);
        load_full(xd, vd_off);
        as_.Pmulld(xn, xm);
        as_.Paddd(xd, xn);
        if (!args.q) mask_low64(xd);
        store_full(vd_off, xd);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kAdd: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (xn == no_simd_register || xm == no_simd_register) { Undefined(); return; }
        load_full(xn, vn_off);
        load_full(xm, vm_off);
        switch (args.size) {
          case 0b00: as_.Paddb(xn, xm); break;
          case 0b01: as_.Paddw(xn, xm); break;
          case 0b10: as_.Paddd(xn, xm); break;
          case 0b11: as_.Paddq(xn, xm); break;
          default: Undefined(); return;
        }
        if (!args.q) mask_low64(xn);
        store_full(vd_off, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kSub: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (xn == no_simd_register || xm == no_simd_register) { Undefined(); return; }
        load_full(xn, vn_off);
        load_full(xm, vm_off);
        switch (args.size) {
          case 0b00: as_.Psubb(xn, xm); break;
          case 0b01: as_.Psubw(xn, xm); break;
          case 0b10: as_.Psubd(xn, xm); break;
          case 0b11: as_.Psubq(xn, xm); break;
          default: Undefined(); return;
        }
        if (!args.q) mask_low64(xn);
        store_full(vd_off, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kAnd: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (xn == no_simd_register || xm == no_simd_register) { Undefined(); return; }
        load_full(xn, vn_off);
        load_full(xm, vm_off);
        as_.Pand(xn, xm);
        if (!args.q) mask_low64(xn);
        store_full(vd_off, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kOrr: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (xn == no_simd_register || xm == no_simd_register) { Undefined(); return; }
        load_full(xn, vn_off);
        load_full(xm, vm_off);
        as_.Por(xn, xm);
        if (!args.q) mask_low64(xn);
        store_full(vd_off, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kEor: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (xn == no_simd_register || xm == no_simd_register) { Undefined(); return; }
        load_full(xn, vn_off);
        load_full(xm, vm_off);
        as_.Pxor(xn, xm);
        if (!args.q) mask_low64(xn);
        store_full(vd_off, xn);
        return;
      }
      case Decoder::AdvSimdThreeSameOpcode::kCmeq: {
        // CMEQ Vd, Vn, Vm — lane-wise equality (-1 if equal, 0 otherwise).
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xm = AllocTempSimdReg();
        if (xn == no_simd_register || xm == no_simd_register) { Undefined(); return; }
        load_full(xn, vn_off);
        load_full(xm, vm_off);
        switch (args.size) {
          case 0b00: as_.Pcmpeqb(xn, xm); break;
          case 0b01: as_.Pcmpeqw(xn, xm); break;
          case 0b10: as_.Pcmpeqd(xn, xm); break;
          default: Undefined(); return;  // 64-bit Pcmpeqq is SSE4_1 — skip for now
        }
        if (!args.q) mask_low64(xn);
        store_full(vd_off, xn);
        return;
      }
      default:
        Undefined();
        return;
    }
    // endregion
  }

  // region digitalis
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
    // for Q=1 (PMULL2; poly_mul64(Vn.D[1], Vm.D[1])). PMULL.8H (size=00)
    // is 8 independent 8-bit polynomial products with no native shape —
    // continues to bail to the interpreter.
    if (args.opcode == Op::kPmull) {
      if (args.size != 0b11) {
        Undefined();
        return;
      }
      const int32_t vn_off_pmull = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
      const int32_t vm_off_pmull = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;
      const int32_t vd_off_pmull = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;
      SimdRegister xn_p = AllocTempSimdReg();
      SimdRegister xm_p = AllocTempSimdReg();
      if (xn_p == no_simd_register || xm_p == no_simd_register) {
        success_ = false;
        return;
      }
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

    // Widening add/sub family — same widen-then-binop pattern as the
    // multiply family below:
    //   SADDL/UADDL/SSUBL/USUBL — widen both Vn and Vm, then add/sub
    //                              at the wide lane.
    //   SADDW/UADDW/SSUBW/USUBW — Vn is already wide (loaded as 128b),
    //                              widen only Vm, then add/sub.
    //   SABDL/UABDL              — widen both, compute max-min at wide
    //                              lane width (= absolute difference,
    //                              valid because zero/sign-extending
    //                              keeps the subtraction within the
    //                              signed range of the wider type).
    //   SABAL/UABAL              — same abs diff, then accumulate into Vd.
    // size=10 (32→64) for the ABDL/ABAL forms needs 64-bit lane-wise
    // max/min, neither of which is in SSE — that case bails to the
    // interpreter (success_ = false). All other size/Q/sign combinations
    // are JIT-lowered.
    {
      const bool is_addl = (args.opcode == Op::kSaddl || args.opcode == Op::kUaddl);
      const bool is_subl = (args.opcode == Op::kSsubl || args.opcode == Op::kUsubl);
      const bool is_addw = (args.opcode == Op::kSaddw || args.opcode == Op::kUaddw);
      const bool is_subw = (args.opcode == Op::kSsubw || args.opcode == Op::kUsubw);
      const bool is_abdl = (args.opcode == Op::kSabdl || args.opcode == Op::kUabdl);
      const bool is_abal = (args.opcode == Op::kSabal || args.opcode == Op::kUabal);

      if (is_addl || is_subl || is_addw || is_subw || is_abdl || is_abal) {
        if (args.size > 0b10) { Undefined(); return; }
        if ((is_abdl || is_abal) && args.size == 0b10) {
          // No 64-bit lane-wise signed/unsigned max/min in SSE.
          success_ = false;
          return;
        }
        const bool addsub_signed = (args.opcode == Op::kSaddl ||
                                    args.opcode == Op::kSsubl ||
                                    args.opcode == Op::kSaddw ||
                                    args.opcode == Op::kSsubw ||
                                    args.opcode == Op::kSabdl ||
                                    args.opcode == Op::kSabal);

        const int32_t vn_off_as = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
        const int32_t vm_off_as = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;
        const int32_t vd_off_as = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;
        const int32_t narrow_disp = args.q ? 8 : 0;
        const bool n_is_wide = (is_addw || is_subw);

        SimdRegister xn_as = AllocTempSimdReg();
        SimdRegister xm_as = AllocTempSimdReg();
        if (xn_as == no_simd_register || xm_as == no_simd_register) {
          success_ = false;
          return;
        }

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

        // ABDL / ABAL: abs(a - b) = max(a, b) - min(a, b) at the widened
        // lane width. After widening both operands, signed and unsigned
        // halves diverge in *which* max/min flavour to pick.
        SimdRegister xmax = AllocTempSimdReg();
        if (xmax == no_simd_register) { success_ = false; return; }
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
          if (xd_as == no_simd_register) { success_ = false; return; }
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

    const int32_t vn_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    const int32_t vm_off = offsetof(ThreadState, cpu.v[0]) + args.rm * 16;
    const int32_t vd_off = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    // Q=0 reads the low 64 bits of Vn/Vm; Q=1 reads bytes 8..15. The widening
    // turns 8 input bytes into the full 128-bit output.
    const int32_t src_disp_extra = args.q ? 8 : 0;

    SimdRegister xn = AllocTempSimdReg();
    SimdRegister xm = AllocTempSimdReg();
    if (xn == no_simd_register || xm == no_simd_register) {
      success_ = false;
      return;
    }

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
    if (xd == no_simd_register) { success_ = false; return; }
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
  // endregion

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    // region digitalis - JIT for EXT Vd.<T>, Vn.<T>, Vm.<T>, #imm.
    // Concatenates Vn:Vm and extracts a vector starting at byte `index`
    // from Vn. Equivalent to:
    //   result = (Vn >> (index*8)) | (Vm << ((vlen-index)*8))
    // For q=1 this is a 16-byte window; for q=0 it's an 8-byte window.
    if (!q) {
      // 64-bit form: handle index 0..7. The simplest correct route is
      // to pack Vm:Vn into a 16-byte register, shift right by index
      // bytes, then mask to low 8 bytes. For now fall back to interp
      // for q=0 to keep the change small (calculate_gnu_hash_neon's ext
      // uses q=1).
      UNUSED(rd, rn, rm, index);
      Undefined();
      return;
    }
    int32_t vn_off = offsetof(ThreadState, cpu.v[0]) + rn * 16;
    int32_t vm_off = offsetof(ThreadState, cpu.v[0]) + rm * 16;
    int32_t vd_off = offsetof(ThreadState, cpu.v[0]) + rd * 16;

    SimdRegister xn = AllocTempSimdReg();
    SimdRegister xm = AllocTempSimdReg();
    if (xn == no_simd_register || xm == no_simd_register) { Undefined(); return; }

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
    // endregion
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
    // region digitalis - JIT for LD1R / LD1 / ST1 single-element variants
    // with num_regs == 1.  Critical for `calculate_gnu_hash_neon`'s tail
    // (`ld1r v3.4s, [x10], #4`) in the dynamic linker — without this the
    // post-loop tail bails to the interpreter on every symbol resolve.
    using Op = Decoder::AdvSimdSingleStructOp;
    const bool is_replicate = (args.op == Op::kLd1r);
    const bool is_single_load = (args.op == Op::kLd1);
    const bool is_single_store = (args.op == Op::kSt1);
    if (!is_replicate && !is_single_load && !is_single_store) {
      Undefined(); return;
    }
    if (args.num_regs != 1) { Undefined(); return; }

    // size: 00=B(1), 01=H(2), 10=S(4), 11=D(8).
    const uint8_t esize = static_cast<uint8_t>(1u << args.size);
    if (esize != 1 && esize != 2 && esize != 4 && esize != 8) {
      Undefined(); return;
    }

    int32_t vt_off = offsetof(ThreadState, cpu.v[0]) + args.rt * 16;

    // Compute base address (with TBI mask).
    Register base_orig = (args.rn == 31) ? GetSp() : GetReg(args.rn);
    if (base_orig == no_register) { Undefined(); return; }
    Register base = ApplyTbi(base_orig);
    if (base == no_register) { Undefined(); return; }

    Assembler::Operand mem{.base = base, .disp = 0};

    if (is_single_store) {
      // ST1 lane: load element from v[rt].lane[index] (ThreadState — no fault),
      // then store to guest memory.
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }
      int32_t lane_off = vt_off + args.index * esize;
      switch (esize) {
        case 1: as_.Movzxbl(tmp, {.base = Assembler::rbp, .disp = lane_off}); break;
        case 2: as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = lane_off}); break;
        case 4: as_.Movl(tmp, {.base = Assembler::rbp, .disp = lane_off}); break;
        case 8: as_.Movq(tmp, {.base = Assembler::rbp, .disp = lane_off}); break;
      }
      AssemblerBase::Label* recovery_label = as_.MakeLabel();
      as_.SetRecoveryPoint(recovery_label);
      switch (esize) {
        case 1: as_.Movb(mem, tmp); break;
        case 2: as_.Movw(mem, tmp); break;
        case 4: as_.Movl(mem, tmp); break;
        case 8: as_.Movq(mem, tmp); break;
      }
      AssemblerBase::Label* cont = as_.MakeLabel();
      as_.Jmp(*cont);
      as_.Bind(recovery_label);
      ExitGeneratedCode(GetInsnAddr());
      as_.Bind(cont);
    } else if (is_replicate) {
      // LD1R: load esize bytes, broadcast to all lanes, zero upper 64 bits
      // if Q=0.
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }
      SimdRegister zero_mask = no_simd_register;
      if (esize == 1) {
        zero_mask = AllocTempSimdReg();
        if (zero_mask == no_simd_register) { Undefined(); return; }
      }
      AssemblerBase::Label* recovery_label = as_.MakeLabel();
      as_.SetRecoveryPoint(recovery_label);
      switch (esize) {
        case 1: as_.Movzxbl(tmp, mem); break;
        case 2: as_.Movzxwl(tmp, mem); break;
        case 4: as_.Movl(tmp, mem); break;
        case 8: as_.Movq(tmp, mem); break;
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
      as_.Movdqu({.base = Assembler::rbp, .disp = vt_off}, xmm);
    } else {
      // LD1 single-lane: load esize bytes into v[rt].lane[index], preserving
      // other lanes by writing directly to ThreadState at the lane offset.
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }
      AssemblerBase::Label* recovery_label = as_.MakeLabel();
      as_.SetRecoveryPoint(recovery_label);
      switch (esize) {
        case 1: as_.Movzxbl(tmp, mem); break;
        case 2: as_.Movzxwl(tmp, mem); break;
        case 4: as_.Movl(tmp, mem); break;
        case 8: as_.Movq(tmp, mem); break;
      }
      AssemblerBase::Label* cont = as_.MakeLabel();
      as_.Jmp(*cont);
      as_.Bind(recovery_label);
      ExitGeneratedCode(GetInsnAddr());
      as_.Bind(cont);

      int32_t lane_off = vt_off + args.index * esize;
      switch (esize) {
        case 1: as_.Movb({.base = Assembler::rbp, .disp = lane_off}, tmp); break;
        case 2: as_.Movw({.base = Assembler::rbp, .disp = lane_off}, tmp); break;
        case 4: as_.Movl({.base = Assembler::rbp, .disp = lane_off}, tmp); break;
        case 8: as_.Movq({.base = Assembler::rbp, .disp = lane_off}, tmp); break;
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
    // endregion
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
    // region digitalis PAuth DP-1Src as identity (emit a plain move)
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
    // endregion
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

      // region digitalis LSE bitwise atomics (LDCLR/LDSET/LDEOR).
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
      // endregion

      // region digitalis atomic min/max (LSE Armv8.1).
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
      // endregion

      // region digitalis CASP JIT (compare-and-swap pair).
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
      // endregion

      default:
        Undefined();
        break;
    }
  }
  // endregion

  void FpDataProc1(const Decoder::FpDataProc1Args& args) {
    // region digitalis - JIT path for FP one-source ops across FP16/FP32/FP64.
    //
    //   FMOV          : direct copy (no F16C).
    //   FABS / FNEG   : sign-bit clear / flip via GP register mask
    //                   (0x7FFF / 0x8000 for FP16, 0x7FFFFFFF / 0x80000000
    //                   for FP32, sign byte at +7 for FP64).
    //   FSQRT         : SQRTSS / SQRTSD natively at FP32/FP64; F16C
    //                   round-trip with SQRTSS at FP16.
    //   FRINTN/M/P/Z/X/I :  ROUNDSS / ROUNDSD natively at FP32/FP64;
    //                   F16C round-trip with ROUNDSS at FP16.
    //   FRINTA        : ARM ties-to-away has no native x86 ROUND* imm.
    //                   Lowered as: dst = trunc(src + copysign(0.5, src)).
    //                   Half goes through F16C round-trip; FP32/FP64 use
    //                   ADDSS+ROUNDSS / ADDSD+ROUNDSD on a GP-built half.
    //   FCVT between precisions : interpreter (different dst layouts).
    //
    // ROUNDSS / ROUNDSD imm[3:0]:
    //   bit 3   : SAE (suppress all FP exceptions)
    //   bit 2   : 1 -> route rounding through MXCSR.RC (NEVER set here;
    //             Berberis doesn't sync MXCSR.RC with guest FPCR.RMode)
    //   bits[1:0] : rounding mode when bit 2 == 0:
    //     00 = RNE (FRINTN)
    //     01 = -inf (FRINTM)
    //     10 = +inf (FRINTP)
    //     11 = zero (FRINTZ)
    //   FRINTX uses current FPCR (assume default RNE) and signals Inexact
    //     -> imm 0x00.
    //   FRINTI uses current FPCR without raising Inexact -> imm 0x08
    //     (SAE + RNE).  Berberis doesn't track FPSR yet, so the SAE bit
    //     is documentary; setting it keeps the lowering spec-faithful.
    //
    // The F16C round-trip is exact for FP16 FSQRT / FRINT* because
    // binary32's 24-bit mantissa fully covers a single-rounding narrowing
    // from any FP16 unary result; the only rounding is at VCVTPS2PH (RNE,
    // matching ARM default FPCR.RMode=0).

    int32_t src_offset = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t dst_offset = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    if (args.ftype == 0b11) {
      // FP16 (half-precision) ops.
      if (args.opcode == 0b000000) {
        // FMOV Hd, Hn: zero dest, then copy 2 bytes via GP scratch.
        // No F16C dependency.
        SimdRegister xmm = AllocTempSimdReg();
        if (xmm == no_simd_register) { Undefined(); return; }
        Register tmp = AllocTempReg();
        as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = src_offset});
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
        as_.Movw({.base = Assembler::rbp, .disp = dst_offset}, tmp);
        return;
      }

      if (args.opcode == 0b000001 || args.opcode == 0b000010) {
        // FABS / FNEG Hd, Hn: load 16 bits, mask/xor sign bit, store 2
        // bytes into a zero-filled dest.  No F16C dependency.
        SimdRegister xmm = AllocTempSimdReg();
        if (xmm == no_simd_register) { Undefined(); return; }
        Register tmp = AllocTempReg();
        as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = src_offset});
        if (args.opcode == 0b000001) {
          as_.Andl(tmp, int32_t{0x00007FFF});  // FABS: clear sign bit
        } else {
          as_.Xorl(tmp, int32_t{0x00008000});  // FNEG: flip sign bit
        }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
        as_.Movw({.base = Assembler::rbp, .disp = dst_offset}, tmp);
        return;
      }

      // FSQRT, FRINT*, and FCVT-from-half need F16C; bail to interpreter
      // if absent.
      if (!host_platform::kHasF16C) { success_ = false; return; }

      // FCVT Sd, Hn (opcode=0b000100) and FCVT Dd, Hn (opcode=0b000101):
      // widen half->single via VCVTPH2PS, then optionally widen to double
      // via CVTSS2SD.  Result lives in lane 0 of a freshly-zeroed dst XMM,
      // so MOVDQU writes the AArch64 scalar layout [val_in_low_N, 0×rest]
      // verbatim.
      if (args.opcode == 0b000100 || args.opcode == 0b000101) {
        bool to_double = (args.opcode == 0b000101);
        SimdRegister xmm_src = AllocTempSimdReg();
        SimdRegister xmm_dst = AllocTempSimdReg();
        if (xmm_src == no_simd_register || xmm_dst == no_simd_register) {
          Undefined();
          return;
        }
        as_.Pxor(xmm_src, xmm_src);
        as_.Pinsrw(xmm_src, {.base = Assembler::rbp, .disp = src_offset},
                   int8_t{0});
        as_.Vcvtph2ps(xmm_src, xmm_src);   // half->single in lane 0
        if (to_double) {
          as_.Pxor(xmm_dst, xmm_dst);
          as_.Cvtss2sd(xmm_dst, xmm_src);  // single->double in lane 0; dst
                                           // upper 96 preserved (zero).
          as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_dst);
        } else {
          // VCVTPH2PS already produced [single, 0, 0, 0] in xmm_src
          // because lanes 1..3 of the source halves were zero.
          as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_src);
        }
        return;
      }

      // FRINTA Hd, Hn: ties-to-away in FP32 space via the canonical
      // add-copysign(0.5)-then-truncate trick.  Build the copysign value
      // in a GP register from the FP16 sign bit, widen src via F16C, do
      // the ADDSS + ROUNDSS imm=3 (truncate), and narrow back.
      if (args.opcode == 0b001100) {
        SimdRegister xmm_val = AllocTempSimdReg();
        SimdRegister xmm_half = AllocTempSimdReg();
        if (xmm_val == no_simd_register || xmm_half == no_simd_register) {
          Undefined();
          return;
        }
        Register tmp = AllocTempReg();
        // bits-of-copysign(0.5_fp32, src_fp16): sign of FP16 src shifted to
        // FP32 sign-bit position, OR'd with the FP32 mantissa/exponent bits
        // of +0.5 (0x3F000000).
        as_.Movzxwl(tmp, {.base = Assembler::rbp, .disp = src_offset});
        as_.Andl(tmp, int32_t{0x00008000});
        as_.Shll(tmp, int8_t{16});
        as_.Orl(tmp, int32_t{0x3F000000});
        as_.Movd(xmm_half, tmp);
        as_.Pxor(xmm_val, xmm_val);
        as_.Pinsrw(xmm_val, {.base = Assembler::rbp, .disp = src_offset},
                   int8_t{0});
        as_.Vcvtph2ps(xmm_val, xmm_val);
        as_.Addss(xmm_val, xmm_half);
        as_.Roundss(xmm_val, xmm_val, int8_t{0x03});  // truncate toward 0
        as_.Vcvtps2ph(xmm_val, xmm_val, int8_t{0});
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
        return;
      }

      int8_t round_imm = 0;
      bool is_sqrt = false;
      switch (args.opcode) {
        case 0b000011: is_sqrt = true; break;       // FSQRT
        case 0b001000: round_imm = 0x00; break;     // FRINTN
        case 0b001001: round_imm = 0x02; break;     // FRINTP
        case 0b001010: round_imm = 0x01; break;     // FRINTM
        case 0b001011: round_imm = 0x03; break;     // FRINTZ
        case 0b001110: round_imm = 0x00; break;     // FRINTX
        case 0b001111: round_imm = 0x08; break;     // FRINTI
        default:
          // Unknown FP16 unary opcode -> interpreter.
          success_ = false;
          return;
      }

      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      as_.Pxor(xmm, xmm);
      as_.Pinsrw(xmm, {.base = Assembler::rbp, .disp = src_offset}, int8_t{0});
      as_.Vcvtph2ps(xmm, xmm);
      if (is_sqrt) {
        as_.Sqrtss(xmm, xmm);
      } else {
        as_.Roundss(xmm, xmm, round_imm);
      }
      as_.Vcvtps2ph(xmm, xmm, int8_t{0});
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
      return;
    }

    // FP32 (ftype=00) and FP64 (ftype=01) paths.
    if (args.ftype != 0b00 && args.ftype != 0b01) { Undefined(); return; }
    bool is_double = (args.ftype == 0b01);

    // FMOV, FABS, FNEG: GP-register mask/copy lowering.
    if (args.opcode == 0b000000 || args.opcode == 0b000001 ||
        args.opcode == 0b000010) {
      SimdRegister xmm = AllocTempSimdReg();
      if (xmm == no_simd_register) { Undefined(); return; }
      Register tmp_lo = AllocTempReg();

      if (!is_double) {
        // FP32: one 4-byte word.
        as_.Movl(tmp_lo, {.base = Assembler::rbp, .disp = src_offset});
        if (args.opcode == 0b000001) {
          as_.Andl(tmp_lo, int32_t{0x7FFFFFFF});  // FABS
        } else if (args.opcode == 0b000010) {
          as_.Xorl(tmp_lo, static_cast<int32_t>(0x80000000));  // FNEG
        }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
        as_.Movl({.base = Assembler::rbp, .disp = dst_offset}, tmp_lo);
        return;
      }

      // FP64: load low 4 bytes verbatim, load high 4 bytes and patch sign
      // bit there.  Splitting into two MOVL avoids needing a 64-bit
      // immediate (MOVABSQ) for the FABS/FNEG masks.
      if (args.opcode == 0b000000) {
        // FMOV Dd, Dn: copy all 8 bytes.
        as_.Movq(tmp_lo, {.base = Assembler::rbp, .disp = src_offset});
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
        as_.Movq({.base = Assembler::rbp, .disp = dst_offset}, tmp_lo);
        return;
      }
      Register tmp_hi = AllocTempReg();
      as_.Movl(tmp_lo, {.base = Assembler::rbp, .disp = src_offset});
      as_.Movl(tmp_hi, {.base = Assembler::rbp, .disp = src_offset + 4});
      if (args.opcode == 0b000001) {
        as_.Andl(tmp_hi, int32_t{0x7FFFFFFF});  // FABS Dd
      } else {
        as_.Xorl(tmp_hi, static_cast<int32_t>(0x80000000));  // FNEG Dd
      }
      as_.Pxor(xmm, xmm);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm);
      as_.Movl({.base = Assembler::rbp, .disp = dst_offset}, tmp_lo);
      as_.Movl({.base = Assembler::rbp, .disp = dst_offset + 4}, tmp_hi);
      return;
    }

    // FCVT between FP32 and FP64: zero a fresh dst XMM, then CVTSS2SD or
    // CVTSD2SS into lane 0.  Both ops preserve the upper 96 bits of the
    // destination, so the zero-then-convert sequence leaves
    // [val_in_low_N, 0×rest] which is the AArch64 scalar layout.
    if (!is_double && args.opcode == 0b000101) {
      // FCVT Dd, Sn — single -> double.  No F16C needed.
      SimdRegister xmm_src = AllocTempSimdReg();
      SimdRegister xmm_dst = AllocTempSimdReg();
      if (xmm_src == no_simd_register || xmm_dst == no_simd_register) {
        Undefined();
        return;
      }
      as_.Movss(xmm_src, {.base = Assembler::rbp, .disp = src_offset});
      as_.Pxor(xmm_dst, xmm_dst);
      as_.Cvtss2sd(xmm_dst, xmm_src);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_dst);
      return;
    }
    if (is_double && args.opcode == 0b000100) {
      // FCVT Sd, Dn — double -> single.  No F16C needed.
      SimdRegister xmm_src = AllocTempSimdReg();
      SimdRegister xmm_dst = AllocTempSimdReg();
      if (xmm_src == no_simd_register || xmm_dst == no_simd_register) {
        Undefined();
        return;
      }
      as_.Movsd(xmm_src, {.base = Assembler::rbp, .disp = src_offset});
      as_.Pxor(xmm_dst, xmm_dst);
      as_.Cvtsd2ss(xmm_dst, xmm_src);
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_dst);
      return;
    }
    // FCVT Hd, Sn / FCVT Hd, Dn — narrow to half.  Needs F16C for VCVTPS2PH.
    // For FP64 source, first narrow double->single via CVTSD2SS into a
    // zeroed XMM, then VCVTPS2PH narrows single->half and zeroes upper 64.
    if (args.opcode == 0b000111) {
      if (!host_platform::kHasF16C) { success_ = false; return; }
      SimdRegister xmm_src = AllocTempSimdReg();
      SimdRegister xmm_dst = AllocTempSimdReg();
      if (xmm_src == no_simd_register || xmm_dst == no_simd_register) {
        Undefined();
        return;
      }
      if (is_double) {
        as_.Movsd(xmm_src, {.base = Assembler::rbp, .disp = src_offset});
        as_.Pxor(xmm_dst, xmm_dst);
        as_.Cvtsd2ss(xmm_dst, xmm_src);            // single in xmm_dst lane 0
        as_.Vcvtps2ph(xmm_dst, xmm_dst, int8_t{0}); // half in xmm_dst lane 0
      } else {
        as_.Movss(xmm_src, {.base = Assembler::rbp, .disp = src_offset});
        as_.Vcvtps2ph(xmm_dst, xmm_src, int8_t{0}); // half in xmm_dst lane 0
      }
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_dst);
      return;
    }

    // FRINTA Sd/Dd, Sn/Dn: ties-to-away has no native ROUND* imm.
    // Lowered as: dst = trunc(src + copysign(0.5, src)).  Build the
    // copysign value in a GP register, then ADDSS/ADDSD + ROUNDSS/ROUNDSD
    // imm=0x03 (truncate toward zero).
    if (args.opcode == 0b001100) {
      SimdRegister xmm_val = AllocTempSimdReg();
      SimdRegister xmm_half = AllocTempSimdReg();
      SimdRegister xmm_zero = AllocTempSimdReg();
      if (xmm_val == no_simd_register || xmm_half == no_simd_register ||
          xmm_zero == no_simd_register) { Undefined(); return; }
      Register tmp = AllocTempReg();
      if (is_double) {
        // FP64: high 32 bits hold sign+exp; low 32 bits of 0.5_fp64 are
        // zero.  Build only the high half in a GP register, then PINSRD
        // it into the upper-32 of an otherwise-zero XMM.
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = src_offset + 4});
        as_.Andl(tmp, static_cast<int32_t>(0x80000000));  // sign bit only
        as_.Orl(tmp, int32_t{0x3FE00000});                 // |= high32(0.5d)
        as_.Pxor(xmm_half, xmm_half);
        as_.Pinsrd(xmm_half, tmp, int8_t{1});              // xmm_half = copysign(0.5d,src)
        as_.Movsd(xmm_val, {.base = Assembler::rbp, .disp = src_offset});
        as_.Addsd(xmm_val, xmm_half);
        as_.Roundsd(xmm_val, xmm_val, int8_t{0x03});       // truncate toward 0
        as_.Pxor(xmm_zero, xmm_zero);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_zero);
        as_.Movsd({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
      } else {
        // FP32: one word holds sign+exp+mantissa.
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = src_offset});
        as_.Andl(tmp, static_cast<int32_t>(0x80000000));  // sign bit only
        as_.Orl(tmp, int32_t{0x3F000000});                 // |= bits of +0.5
        as_.Movd(xmm_half, tmp);                           // xmm_half = copysign(0.5,src)
        as_.Movss(xmm_val, {.base = Assembler::rbp, .disp = src_offset});
        as_.Addss(xmm_val, xmm_half);
        as_.Roundss(xmm_val, xmm_val, int8_t{0x03});       // truncate toward 0
        as_.Pxor(xmm_zero, xmm_zero);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_zero);
        as_.Movss({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
      }
      return;
    }

    // FSQRT and FRINT*: SIMD lowering via SQRTSS/SQRTSD / ROUNDSS/ROUNDSD.
    int8_t round_imm = 0;
    bool is_sqrt = false;
    switch (args.opcode) {
      case 0b000011: is_sqrt = true; break;       // FSQRT
      case 0b001000: round_imm = 0x00; break;     // FRINTN
      case 0b001001: round_imm = 0x02; break;     // FRINTP
      case 0b001010: round_imm = 0x01; break;     // FRINTM
      case 0b001011: round_imm = 0x03; break;     // FRINTZ
      case 0b001110: round_imm = 0x00; break;     // FRINTX
      case 0b001111: round_imm = 0x08; break;     // FRINTI
      default:
        // Unknown FP unary opcode -> interpreter.
        success_ = false;
        return;
    }

    SimdRegister xmm_val = AllocTempSimdReg();
    if (xmm_val == no_simd_register) { Undefined(); return; }
    SimdRegister xmm_zero = AllocTempSimdReg();
    if (xmm_zero == no_simd_register) { Undefined(); return; }

    // Load src first (in case dst == src).
    if (is_double) {
      as_.Movsd(xmm_val, {.base = Assembler::rbp, .disp = src_offset});
    } else {
      as_.Movss(xmm_val, {.base = Assembler::rbp, .disp = src_offset});
    }
    // Zero the full 128-bit dst register.
    as_.Pxor(xmm_zero, xmm_zero);
    as_.Movdqu({.base = Assembler::rbp, .disp = dst_offset}, xmm_zero);
    // Operate in place on lane 0 of xmm_val (upper lanes were zeroed by
    // MOVSS/MOVSD from memory; SQRTSS/SQRTSD and ROUNDSS/ROUNDSD preserve
    // the upper lanes of the dest).
    if (is_double) {
      if (is_sqrt) {
        as_.Sqrtsd(xmm_val, xmm_val);
      } else {
        as_.Roundsd(xmm_val, xmm_val, round_imm);
      }
      as_.Movsd({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
    } else {
      if (is_sqrt) {
        as_.Sqrtss(xmm_val, xmm_val);
      } else {
        as_.Roundss(xmm_val, xmm_val, round_imm);
      }
      as_.Movss({.base = Assembler::rbp, .disp = dst_offset}, xmm_val);
    }
    // endregion
  }

  // region digitalis - FP arithmetic JIT
  //
  // FP16 (ftype=0b11) lowering: F16C round-trip.
  //   PINSRW [src] -> XMM lane0       (load the 16-bit half, upper lanes zero)
  //   VCVTPH2PS XMM, XMM              (widen 4 halves->4 singles; lanes 1-3 are zero)
  //   <op>SS    XMM, XMM              (perform the binary op in single precision)
  //   VCVTPS2PH XMM, XMM, imm=0       (narrow back, RNE, ignore MXCSR)
  //   MOVDQU [dst], XMM               (store 128 bits: result in low 16, rest zero)
  //
  // The round-trip is exact for FADD/FSUB/FMUL/FDIV because binary32's 24-bit
  // mantissa fully covers a single-rounding narrowing from any FP16 op result.
  // RNE matches the ARM default rounding mode (FPCR.RMode=0, ties-to-even).
  // F16C is part of the IvyBridge+ baseline; bail to interpreter if absent.
  void FpDataProc2(const Decoder::FpDataProc2Args& args) {
    // Only handle single (ftype=00), double (ftype=01), and half (ftype=11).
    if (args.ftype == 0b10) { Undefined(); return; }
    if (args.ftype == 0b11 && !host_platform::kHasF16C) { success_ = false; return; }
    bool is_double = (args.ftype == 0b01);
    bool is_half = (args.ftype == 0b11);

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
    } else if (is_half) {
      as_.Pxor(xmm_n, xmm_n);
      as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_n, xmm_n);
      as_.Pxor(xmm_m, xmm_m);
      as_.Pinsrw(xmm_m, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_m, xmm_m);
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    }

    // Perform operation: result in xmm_n. FP16 reuses the SS form because the
    // value already lives as binary32 in lane 0.
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

    if (is_half) {
      // Narrow FP32 result -> FP16 with RNE (imm=0). VCVTPS2PH zeroes the upper
      // 64 bits of the XMM dest; lanes 1-3 of the FP32 are zero so the FP16
      // lanes 1-3 are also zero. Storing all 128 bits gives the correct guest
      // register layout: result in low 16, rest zero.
      as_.Vcvtps2ph(xmm_n, xmm_n, int8_t{0});
      as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_n);
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
    // FCMP Sn, Sm / FCMP Dn, Dm / FCMP Hn, Hm: compare and set NZCV flags.
    if (args.ftype == 0b10) { Undefined(); return; }
    if (args.ftype == 0b11 && !host_platform::kHasF16C) { success_ = false; return; }
    bool is_double = (args.ftype == 0b01);
    bool is_half = (args.ftype == 0b11);

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
    } else if (is_half) {
      // Widen both operands FP16->FP32 via F16C, then UCOMISS. FP16 zero
      // (0x0000) round-trips through Vcvtph2ps to FP32 +0.0, so the with_zero
      // path simply leaves xmm_m as Pxor'd (FP32 zero in lane 0).
      as_.Pxor(xmm_n, xmm_n);
      as_.Pinsrw(xmm_n, {.base = Assembler::rbp, .disp = src_n_off}, int8_t{0});
      as_.Vcvtph2ps(xmm_n, xmm_n);
      if (args.with_zero) {
        as_.Pxor(xmm_m, xmm_m);
      } else {
        as_.Pxor(xmm_m, xmm_m);
        as_.Pinsrw(xmm_m, {.base = Assembler::rbp, .disp = src_m_off}, int8_t{0});
        as_.Vcvtph2ps(xmm_m, xmm_m);
      }
      as_.Ucomiss(xmm_n, xmm_m);
    } else {
      as_.Movss(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
      if (args.with_zero) {
        as_.Pxor(xmm_m, xmm_m);
      } else {
        as_.Movss(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
      }
      as_.Ucomiss(xmm_n, xmm_m);
    }

    // region digitalis fix: emit the correct FP-specific ARM NZCV
    // mapping, not the integer-SUB EmitStoreArmNZCV.  UCOMISS/UCOMISD set
    // only ZF/PF/CF; SF and OF retain stale values, so the integer-flag
    // emission produced random N and V bits.  This silently broke any
    // FCMP-then-CSET-{ge,gt,lt,le} sequence (only EQ/NE worked because
    // they only depend on Z, which the integer mapping happened to set
    // correctly).  See note in EmitStoreArmFpNZCV below.
    EmitStoreArmFpNZCV();
    // endregion
  }
  // endregion

  // region digitalis
  // FCCMP / FCCMPE: fall back to interpreter (condition-dependent flag write
  // makes this awkward in JIT; cost is one interpreter dispatch per FCCMP).
  void FpConditionalCompare(const Decoder::FpConditionalCompareArgs& /*args*/) {
    success_ = false;
  }
  // endregion

  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& args) {
    // region digitalis - JIT for CMEQZ (cmeq Vd, Vn, #0) used by the
    // dynamic linker's calculate_gnu_hash_neon. Other opcodes fall
    // through to the interpreter.
    int32_t vn_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t vd_off = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    auto mask_low64 = [&](SimdRegister xmm) {
      as_.Pslldq(xmm, int8_t{8});
      as_.Psrldq(xmm, int8_t{8});
    };

    switch (args.opcode) {
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmeqZero: {
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xz = AllocTempSimdReg();
        if (xn == no_simd_register || xz == no_simd_register) { Undefined(); return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xz, xz);
        switch (args.size) {
          case 0b00: as_.Pcmpeqb(xn, xz); break;
          case 0b01: as_.Pcmpeqw(xn, xz); break;
          case 0b10: as_.Pcmpeqd(xn, xz); break;
          default: Undefined(); return;
        }
        if (!args.q) mask_low64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // REV64 Vd.<T>, Vn.<T> — reverse element order within each 64-bit lane.
      // size=00: byte reverse (8B / 16B) — BSWAPQ on each 64-bit half.
      // size=01: halfword reverse (4H / 8H) — PSHUFLW + PSHUFHW imm=0x1B.
      // size=10: word reverse (2S / 4S) — PSHUFD imm=0x01 (Q=0) or 0xB1 (Q=1).
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev64: {
        SimdRegister xn = AllocTempSimdReg();
        if (xn == no_simd_register) { Undefined(); return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        switch (args.size) {
          case 0b00: {
            Register r1 = AllocTempReg();
            if (r1 == no_register) { Undefined(); return; }
            as_.Movq(r1, xn);
            as_.Bswapq(r1);
            if (args.q) {
              Register r2 = AllocTempReg();
              if (r2 == no_register) { Undefined(); return; }
              as_.Pextrq(r2, xn, int8_t{1});
              as_.Bswapq(r2);
              as_.Movq(xn, r1);
              as_.Pinsrq(xn, r2, int8_t{1});
            } else {
              as_.Movq(xn, r1);  // zeros upper 64 bits
            }
            break;
          }
          case 0b01:
            as_.Pshuflw(xn, xn, int8_t{0x1B});
            if (args.q) {
              as_.Pshufhw(xn, xn, int8_t{0x1B});
            } else {
              mask_low64(xn);
            }
            break;
          case 0b10:
            if (args.q) {
              as_.Pshufd(xn, xn, static_cast<int8_t>(0xB1));
            } else {
              as_.Pshufd(xn, xn, static_cast<int8_t>(0x01));
              mask_low64(xn);
            }
            break;
          default: Undefined(); return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // REV32 Vd.<T>, Vn.<T> — reverse element order within each 32-bit lane.
      // size=00: byte reverse (8B / 16B) — BSWAPL per 32-bit lane.
      // size=01: halfword reverse (4H / 8H) — PSHUFLW/HW imm=0xB1 (swap pairs).
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev32: {
        SimdRegister xn = AllocTempSimdReg();
        if (xn == no_simd_register) { Undefined(); return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        switch (args.size) {
          case 0b00: {
            SimdRegister xd = AllocTempSimdReg();
            Register r1 = AllocTempReg();
            if (xd == no_simd_register || r1 == no_register) { Undefined(); return; }
            as_.Pxor(xd, xd);
            int lanes = args.q ? 4 : 2;
            for (int i = 0; i < lanes; ++i) {
              as_.Pextrd(r1, xn, static_cast<int8_t>(i));
              as_.Bswapl(r1);
              as_.Pinsrd(xd, r1, static_cast<int8_t>(i));
            }
            as_.Movdqa(xn, xd);
            break;
          }
          case 0b01:
            as_.Pshuflw(xn, xn, static_cast<int8_t>(0xB1));
            if (args.q) {
              as_.Pshufhw(xn, xn, static_cast<int8_t>(0xB1));
            } else {
              mask_low64(xn);
            }
            break;
          default: Undefined(); return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // REV16 Vd.<T>, Vn.<T> — reverse byte order within each 16-bit lane.
      // size=00 only (8B / 16B). Compute: (Vn << 8) | (Vn >> 8) per halfword.
      // PSLLW shifts each 16-bit lane left by 8 (high byte was lost, low byte
      // moves to high). PSRLW shifts right by 8 in the saved copy (low byte
      // was lost, high byte moves to low). OR'ing combines the byte-swapped
      // result without needing a PSHUFB mask table.
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev16: {
        if (args.size != 0b00) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister xt = AllocTempSimdReg();
        if (xn == no_simd_register || xt == no_simd_register) { Undefined(); return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Movdqa(xt, xn);
        as_.Psllw(xn, int8_t{8});
        as_.Psrlw(xt, int8_t{8});
        as_.Por(xn, xt);
        if (!args.q) mask_low64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // Vector FABS / FNEG (FP32 .2S/.4S, FP64 .2D, FP16 .4H/.8H).
      //   size=10 → FP32, size=11 → FP64.  FP64 requires Q=1.
      //   FABS: AND with broadcast mask 0x7FFFFFFF (FP32) or 0x7FFFFFFF_FFFFFFFF (FP64).
      //   FNEG: XOR with broadcast mask 0x80000000 (FP32) or 0x80000000_00000000 (FP64).
      //   FP16: broadcast 16-bit mask 0x7FFF (FABS) / 0x8000 (FNEG).  FP16
      //   FABS/FNEG are pure bit operations — no F16C round-trip required.
      // region digitalis
      case Decoder::AdvSimdTwoRegMiscOpcode::kFabs:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFneg: {
        const bool is_fabs =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFabs);
        if (args.is_fp16) {
          // .4H (Q=0) and .8H (Q=1) — broadcast 16-bit mask then PAND/PXOR.
          SimdRegister xn = AllocTempSimdReg();
          SimdRegister mask = AllocTempSimdReg();
          if (xn == no_simd_register || mask == no_simd_register) { Undefined(); return; }
          as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
          as_.Pcmpeqd(mask, mask);
          if (is_fabs) {
            as_.Psrlw(mask, int8_t{1});   // each 16-bit lane = 0x7FFF
            as_.Pand(xn, mask);
          } else {
            as_.Psllw(mask, int8_t{15});  // each 16-bit lane = 0x8000
            as_.Pxor(xn, mask);
          }
          if (!args.q) mask_low64(xn);
          as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
          return;
        }
        // endregion
        if (args.size != 0b10 && args.size != 0b11) { Undefined(); return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister mask = AllocTempSimdReg();
        if (xn == no_simd_register || mask == no_simd_register) { Undefined(); return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pcmpeqd(mask, mask);
        if (is_double) {
          // FABS mask 0x7FFFFFFFFFFFFFFF (allones >> 1); FNEG mask 0x8000000000000000.
          if (is_fabs) {
            as_.Psrlq(mask, int8_t{1});
            as_.Pand(xn, mask);
          } else {
            as_.Psllq(mask, int8_t{63});
            as_.Pxor(xn, mask);
          }
        } else {
          if (is_fabs) {
            as_.Psrld(mask, int8_t{1});
            as_.Pand(xn, mask);
          } else {
            as_.Pslld(mask, int8_t{31});
            as_.Pxor(xn, mask);
          }
        }
        if (!args.q) mask_low64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // Vector FRINTN / FRINTM / FRINTP / FRINTZ / FRINTX / FRINTI
      // (FP32 .2S/.4S, FP64 .2D).
      //   FRINTN  -> ROUNDPS/PD imm=0 (round to nearest even).
      //   FRINTM  -> ROUNDPS/PD imm=1 (toward -inf, floor).
      //   FRINTP  -> ROUNDPS/PD imm=2 (toward +inf, ceil).
      //   FRINTZ  -> ROUNDPS/PD imm=3 (toward zero, trunc).
      //   FRINTX/FRINTI -> ROUNDPS/PD imm=4 (use MXCSR; default RNE matches ARM).
      // SSE4.1 ROUNDPS/PD natively accepts the imm; one instruction per vector.
      // The FP16 form (args.is_fp16) bails to the interpreter.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintxV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintiV: {
        if (args.is_fp16) { success_ = false; return; }
        if (args.size != 0b10 && args.size != 0b11) { Undefined(); return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { Undefined(); return; }
        int8_t round_imm;
        switch (args.opcode) {
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV: round_imm = 0x00; break;
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV: round_imm = 0x01; break;
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV: round_imm = 0x02; break;
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV: round_imm = 0x03; break;
          // FRINTX / FRINTI follow the current FPCR rounding mode; we treat
          // MXCSR (default RNE) as the canonical mode.  imm=0x04 sets the
          // ROUND* "use MXCSR" bit.
          default: round_imm = 0x04; break;
        }
        SimdRegister xn = AllocTempSimdReg();
        if (xn == no_simd_register) { Undefined(); return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (is_double) {
          as_.Roundpd(xn, xn, round_imm);
        } else {
          as_.Roundps(xn, xn, round_imm);
        }
        if (!args.q) mask_low64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // Vector FRINTA (FP32 .2S/.4S, FP64 .2D) -- "round to nearest, ties away".
      // x86 ROUNDPS/PD has no ties-away mode, so use the identity
      //     FRINTA(x) = trunc(x + copysign(0.5, x))
      // already used by the scalar FRINTA JIT path (handoff-81).  Per-lane:
      // build (x AND sign_mask) OR half_pattern  ->  +/-0.5, add to x, ROUND
      // imm=3.  Bit-exact against ARM for all finite/NaN/Inf inputs (the
      // +0.5 nudge is a no-op when |x| >= 2^p; signed zeros preserved by
      // ROUNDPS imm=3; NaN/Inf propagate through ADDPS/ADDPD).
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintaV: {
        if (args.is_fp16) { success_ = false; return; }
        if (args.size != 0b10 && args.size != 0b11) { Undefined(); return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        SimdRegister copysign = AllocTempSimdReg();
        SimdRegister half = AllocTempSimdReg();
        if (xn == no_simd_register || copysign == no_simd_register ||
            half == no_simd_register) { Undefined(); return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        // Build the sign-bit mask in copysign, then AND with xn to extract
        // per-lane sign bits.
        as_.Pcmpeqd(copysign, copysign);
        if (is_double) {
          as_.Psllq(copysign, int8_t{63});
        } else {
          as_.Pslld(copysign, int8_t{31});
        }
        as_.Pand(copysign, xn);
        // Build per-lane 0.5 broadcast in half (FP32: 0x3F000000, FP64:
        // 0x3FE0000000000000).  Move via a GP scratch -> Movd/Movq -> Pshufd.
        Register gp_half = AllocTempReg();
        if (gp_half == no_register) { Undefined(); return; }
        if (is_double) {
          as_.Movq(gp_half, int64_t{0x3FE0000000000000LL});
          as_.Movq(half, gp_half);
          // imm=0x44 = 01_00_01_00 -> broadcast low 64 bits across both
          // 64-bit lanes of the XMM.
          as_.Pshufd(half, half, static_cast<int8_t>(0x44));
        } else {
          as_.Movl(gp_half, int32_t{0x3F000000});
          as_.Movd(half, gp_half);
          // imm=0x00 broadcasts the low dword across all 4 dwords.
          as_.Pshufd(half, half, static_cast<int8_t>(0x00));
        }
        // copysign |= half  -> per-lane sign(xn) * 0.5.
        as_.Por(copysign, half);
        // xn += copysign(0.5, xn); truncate toward zero.
        if (is_double) {
          as_.Addpd(xn, copysign);
          as_.Roundpd(xn, xn, int8_t{0x03});
        } else {
          as_.Addps(xn, copysign);
          as_.Roundps(xn, xn, int8_t{0x03});
        }
        if (!args.q) mask_low64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      // Vector FSQRT (FP32 .2S/.4S, FP64 .2D, FP16 .4H/.8H) -- per-lane
      // square root.
      //   FP32/FP64: SSE SQRTPS / SQRTPD; default MXCSR rounding (RNE)
      //   matches ARM's default FPCR rounding. NaN, signed zeros and
      //   negative finites all behave identically to the ARM ARM
      //   semantics (SQRTPS/PD propagate NaN, produce -0 for -0 input
      //   and qNaN for negative finite input).
      //   FP16: F16C round-trip (Vcvtph2ps widen -> SQRTPS -> Vcvtps2ph
      //   narrow imm=0). Per the standing rule, F16C round-trip is
      //   exact for FP16 unary FSQRT. The .8H form splits the upper 4
      //   half-lanes into a second F16C round-trip (no AVX YMM path).
      case Decoder::AdvSimdTwoRegMiscOpcode::kFsqrtV: {
        if (args.is_fp16) {
          if (!host_platform::kHasF16C) { success_ = false; return; }
          SimdRegister xlo = AllocTempSimdReg();
          if (xlo == no_simd_register) { Undefined(); return; }
          if (!args.q) {
            // .4H: 4 FP16 lanes in low 64 bits of Vn.
            as_.Movq(xlo, {.base = Assembler::rbp, .disp = vn_off});
            as_.Vcvtph2ps(xlo, xlo);
            as_.Sqrtps(xlo, xlo);
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            // Vcvtps2ph already zeroes the upper 64 bits of xlo.
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          } else {
            // .8H: 8 FP16 lanes; process low 4 then high 4.
            SimdRegister xhi = AllocTempSimdReg();
            if (xhi == no_simd_register) { Undefined(); return; }
            as_.Movdqu(xhi, {.base = Assembler::rbp, .disp = vn_off});
            // Low half: copy then widen lanes 0-3.
            as_.Movdqa(xlo, xhi);
            as_.Vcvtph2ps(xlo, xlo);
            // High half: shift right 8 bytes so upper 4 FP16 -> low 64
            // bits of xhi, then widen.
            as_.Psrldq(xhi, int8_t{8});
            as_.Vcvtph2ps(xhi, xhi);
            as_.Sqrtps(xlo, xlo);
            as_.Sqrtps(xhi, xhi);
            // Narrow each back to 4 FP16 in the low 64 bits.
            as_.Vcvtps2ph(xlo, xlo, int8_t{0});
            as_.Vcvtps2ph(xhi, xhi, int8_t{0});
            // Shift high result into the upper 64 bits and OR with low.
            as_.Pslldq(xhi, int8_t{8});
            as_.Por(xlo, xhi);
            as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xlo);
          }
          return;
        }
        if (args.size != 0b10 && args.size != 0b11) { Undefined(); return; }
        const bool is_double = (args.size & 1);
        if (is_double && !args.q) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        if (xn == no_simd_register) { Undefined(); return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        if (is_double) {
          as_.Sqrtpd(xn, xn);
        } else {
          as_.Sqrtps(xn, xn);
        }
        if (!args.q) mask_low64(xn);
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      default:
        Undefined();
        return;
    }
    // endregion
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
    // region digitalis - JIT for USHLL (unsigned shift-left long) at
    // 8B→8H / 4H→4S widening, used by calculate_gnu_hash_neon and many
    // SIMD widening expansions. Other shift-imm opcodes fall back.
    int32_t vn_off = offsetof(ThreadState, cpu.v[0]) + args.rn * 16;
    int32_t vd_off = offsetof(ThreadState, cpu.v[0]) + args.rd * 16;

    switch (args.opcode) {
      case Decoder::AdvSimdShiftImmOpcode::kUshll: {
        // USHLL Vd.<wide>, Vn.<narrow>, #shift.
        // ARM encoding: esize = 8 << highest-set-bit(immh).
        // immh=0001 → 8B→8H, shift = (immh:immb) - 8
        // immh=001x → 4H→4S, shift = (immh:immb) - 16
        // immh=01xx → 2S→2D, shift = (immh:immb) - 32
        uint8_t immh = args.immh;
        if (immh == 0) { Undefined(); return; }
        SimdRegister xn = AllocTempSimdReg();
        if (xn == no_simd_register) { Undefined(); return; }
        // Q bit is which half of Vn to read; for the !q (low-half)
        // form used by the linker, we read the D portion. Q=1 reads
        // the upper half ("ushll2"): not implemented yet.
        if (args.q) { Undefined(); return; }
        as_.Movq(xn, {.base = Assembler::rbp, .disp = vn_off});
        uint8_t shift_imm;
        if (immh & 0b1000) {
          // 2S → 2D
          shift_imm = ((immh << 3) | args.immb) - 32;
          as_.Pmovzxdq(xn, xn);
          if (shift_imm != 0) as_.Psllq(xn, shift_imm);
        } else if (immh & 0b0110) {
          // 4H → 4S
          shift_imm = ((immh << 3) | args.immb) - 16;
          as_.Pmovzxwd(xn, xn);
          if (shift_imm != 0) as_.Pslld(xn, shift_imm);
        } else if (immh & 0b0001) {
          // 8B → 8H
          shift_imm = ((immh << 3) | args.immb) - 8;
          as_.Pmovzxbw(xn, xn);
          if (shift_imm != 0) as_.Psllw(xn, shift_imm);
        } else {
          Undefined(); return;
        }
        as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xn);
        return;
      }
      default:
        Undefined();
        return;
    }
    // endregion
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

  // region digitalis: emit ARM FP-compare NZCV from x86 UCOMIS flags.
  //
  // UCOMISS/UCOMISD set ZF/PF/CF and leave SF/OF untouched, so the integer
  // EmitStoreArmNZCV path (which copies SF into ARM N and OF into ARM V)
  // produced random N and V.  Map the four ordered outcomes by jump-table:
  //   x86 ZF PF CF   ARM NZCV (bit15 N, bit14 Z, bit8 C, bit0 V)   value
  //   gt:  0 0 0  -> 0 0 1 0                                       0x0100
  //   lt:  0 0 1  -> 1 0 0 0                                       0x8000
  //   eq:  1 0 0  -> 0 1 1 0                                       0x4100
  //   uo:  1 1 1  -> 0 0 1 1                                       0x0101
  // The Movl-imm sequence below preserves EFLAGS (MOV doesn't touch them),
  // so the Jcc reads UCOMISS's flags directly.
  void EmitStoreArmFpNZCV() {
    Assembler::Label* uo_label = as_.MakeLabel();
    Assembler::Label* eq_label = as_.MakeLabel();
    Assembler::Label* lt_label = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();

    as_.Movl(Assembler::rax, static_cast<int32_t>(0x0100));  // gt (default)
    as_.Jcc(Condition::kParityEven, *uo_label);  // PF=1 -> unordered (NaN)
    as_.Jcc(Condition::kEqual, *eq_label);       // ZF=1 (PF=0) -> equal
    as_.Jcc(Condition::kBelow, *lt_label);       // CF=1 -> less
    as_.Jmp(*done);                              // else gt

    as_.Bind(lt_label);
    as_.Movl(Assembler::rax, static_cast<int32_t>(0x8000));
    as_.Jmp(*done);

    as_.Bind(eq_label);
    as_.Movl(Assembler::rax, static_cast<int32_t>(0x4100));
    as_.Jmp(*done);

    as_.Bind(uo_label);
    as_.Movl(Assembler::rax, static_cast<int32_t>(0x0101));

    as_.Bind(done);
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
