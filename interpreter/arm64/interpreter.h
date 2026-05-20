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

#include "berberis/interpreter/arm64/interpreter.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "../faulty_memory_accesses.h"
// endregion

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/decoder/arm64/decoder.h"
#include "berberis/decoder/arm64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/kernel_api/run_guest_syscall.h"
#include "berberis/runtime_primitives/interpret_helpers.h"

namespace berberis {

class Interpreter {
 public:
  using Decoder = Decoder<SemanticsPlayer<Interpreter>>;
  using Register = uint64_t;
  static constexpr Register no_register = 0;

  explicit Interpreter(ThreadState* state)
      : state_(state), branch_taken_(false), exception_raised_(false) {}

  // region digitalis
  // Reset per-instruction state for batch reuse — avoids reconstructing
  // the Interpreter object for every instruction in the batch.
  void Reset() {
    branch_taken_ = false;
    exception_raised_ = false;
  }
  // endregion

  // region digitalis
  // Memory fault handler — called when FaultyLoad/FaultyStore detects a fault.
  // Sets exception_raised_ to stop the interpreter batch. HandleFaultForRecovery
  // has already queued a SIGSEGV for guest delivery — the runtime will process it
  // in ExecuteGuest, either invoking the guest's handler or applying the default
  // action (terminate for unclaimed SIGSEGV).
  void HandleMemoryFault(uint64_t fault_addr) {
    (void)fault_addr;
    exception_raised_ = true;
  }

  bool HasException() const { return exception_raised_; }
  // endregion

  //
  // Instruction implementations.
  //

  Register AddSubImm(bool is_sub, bool set_flags, bool is_64bit,
                     Register src, uint32_t imm) {
    CHECK(!exception_raised_);
    uint64_t operand1 = is_64bit ? src : (src & 0xFFFFFFFFULL);
    uint64_t operand2 = static_cast<uint64_t>(imm);
    uint64_t result;

    if (is_sub) {
      result = operand1 - operand2;
    } else {
      result = operand1 + operand2;
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    if (set_flags) {
      UpdateFlags(operand1, operand2, result, is_sub, is_64bit);
    }

    return result;
  }

  Register LogicalImm(Decoder::LogicalImmOpcode opcode, bool is_64bit,
                      Register src, uint64_t imm) {
    CHECK(!exception_raised_);
    uint64_t operand = is_64bit ? src : (src & 0xFFFFFFFFULL);
    uint64_t result;

    switch (opcode) {
      case Decoder::LogicalImmOpcode::kAnd:
      case Decoder::LogicalImmOpcode::kAnds:
        result = operand & imm;
        break;
      case Decoder::LogicalImmOpcode::kOrr:
        result = operand | imm;
        break;
      case Decoder::LogicalImmOpcode::kEor:
        result = operand ^ imm;
        break;
      default:
        Undefined();
        return 0;
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    if (opcode == Decoder::LogicalImmOpcode::kAnds) {
      UpdateLogicalFlags(result, is_64bit);
    }

    return result;
  }

  Register MoveWide(Decoder::MoveWideOpcode opcode, bool is_64bit,
                    uint16_t imm16, uint8_t shift) {
    CHECK(!exception_raised_);
    uint64_t value = static_cast<uint64_t>(imm16) << shift;

    switch (opcode) {
      case Decoder::MoveWideOpcode::kMovz:
        break;
      case Decoder::MoveWideOpcode::kMovn:
        value = ~value;
        break;
      case Decoder::MoveWideOpcode::kMovk:
        // MOVK keeps other bits -- but we don't have the destination register value here.
        // The SemanticsPlayer should handle MOVK differently, but for the simple case
        // where dst is being initialized, we handle it as-is. For proper MOVK we need
        // the current register value. We return just the shifted value; the caller must
        // merge. Note: we handle this specially below in a dedicated method.
        break;
      default:
        Undefined();
        return 0;
    }

    if (!is_64bit) {
      value &= 0xFFFFFFFFULL;
    }

    return value;
  }

  // Special MOVK handler that merges with current register value.
  Register MoveWideKeep(Register current, uint16_t imm16, uint8_t shift, bool is_64bit) {
    CHECK(!exception_raised_);
    uint64_t mask = static_cast<uint64_t>(0xFFFF) << shift;
    uint64_t value = static_cast<uint64_t>(imm16) << shift;
    uint64_t result = (current & ~mask) | value;
    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }
    return result;
  }

  Register PcRelAddr(bool is_adrp, int64_t offset) {
    CHECK(!exception_raised_);
    uint64_t pc = state_->cpu.insn_addr;
    if (is_adrp) {
      // ADRP: page of PC + offset.
      pc &= ~0xFFFULL;  // Align PC to 4K page.
    }
    return pc + offset;
  }

  Register Bitfield(Decoder::BitfieldOpcode opcode, bool is_64bit,
                    Register dst_val, Register src, uint8_t immr, uint8_t imms) {
    CHECK(!exception_raised_);
    // region digitalis
    // Simplified implementation based on ARM ARM pseudocode for BFM/SBFM/UBFM.
    // When imms >= immr: extract bits[imms:immr] (bitfield extract / shift right)
    // When imms < immr:  insert bits[imms:0] at position (regsize-immr) (bitfield insert / shift left)
    unsigned reg_size = is_64bit ? 64 : 32;
    uint64_t src_val = is_64bit ? src : (src & 0xFFFFFFFFULL);
    uint64_t result;

    if (imms >= immr) {
      // Extraction case: extract bits[imms:immr] from source.
      // Width = imms - immr + 1
      unsigned width = imms - immr + 1;
      uint64_t extracted = (src_val >> immr);
      if (width < reg_size) {
        extracted &= ((1ULL << width) - 1);
      }

      switch (opcode) {
        case Decoder::BitfieldOpcode::kUbfm:
          // Zero-extend extracted bits. Aliases: LSR, UBFX, UXTB, UXTH.
          result = extracted;
          break;
        case Decoder::BitfieldOpcode::kSbfm: {
          // Sign-extend from bit (width-1). Aliases: ASR, SBFX, SXTB, SXTH, SXTW.
          result = extracted;
          if (width < reg_size && (extracted & (1ULL << (width - 1)))) {
            result |= ~((1ULL << width) - 1);
          }
          break;
        }
        case Decoder::BitfieldOpcode::kBfm:
          // Merge: insert extracted bits at position 0, keep other dest bits.
          if (width < reg_size) {
            uint64_t mask = (1ULL << width) - 1;
            result = (dst_val & ~mask) | (extracted & mask);
          } else {
            result = extracted;
          }
          break;
        default:
          Undefined();
          return 0;
      }
    } else {
      // Insertion case: take bits[imms:0] from source and place at position (regsize-immr).
      // Width = imms + 1
      unsigned width = imms + 1;
      unsigned pos = reg_size - immr;
      uint64_t field = src_val & ((1ULL << width) - 1);
      uint64_t placed = field << pos;
      uint64_t mask = ((1ULL << width) - 1) << pos;

      switch (opcode) {
        case Decoder::BitfieldOpcode::kUbfm:
          // Zero other bits. Aliases: LSL, UBFIZ.
          result = placed;
          break;
        case Decoder::BitfieldOpcode::kSbfm: {
          // Sign-extend from bit (pos + width - 1). Aliases: SBFIZ.
          result = placed;
          unsigned top_bit = pos + width - 1;
          if (top_bit < reg_size - 1 && (result & (1ULL << top_bit))) {
            result |= ~((1ULL << (top_bit + 1)) - 1);
          }
          break;
        }
        case Decoder::BitfieldOpcode::kBfm:
          // Merge: insert field bits, keep other dest bits. Aliases: BFI.
          result = (dst_val & ~mask) | (placed & mask);
          break;
        default:
          Undefined();
          return 0;
      }
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    return result;
    // endregion
  }

  void Branch(int32_t offset) {
    CHECK(!exception_raised_);
    state_->cpu.insn_addr += offset;
    branch_taken_ = true;
  }

  void BranchCond(Decoder::Condition cond, int32_t offset) {
    CHECK(!exception_raised_);
    if (EvaluateCondition(cond)) {
      Branch(offset);
    }
  }

  void BranchRegister(Register target) {
    CHECK(!exception_raised_);
    state_->cpu.insn_addr = target;
    branch_taken_ = true;
  }

  void CompareAndBranch(bool is_nonzero, bool is_64bit, Register src, int32_t offset) {
    CHECK(!exception_raised_);
    uint64_t val = is_64bit ? src : (src & 0xFFFFFFFFULL);
    bool take_branch = is_nonzero ? (val != 0) : (val == 0);
    if (take_branch) {
      Branch(offset);
    }
  }

  void TestAndBranch(bool is_nonzero, Register src, uint8_t bit, int32_t offset) {
    CHECK(!exception_raised_);
    bool bit_set = (src >> bit) & 1;
    bool take_branch = is_nonzero ? bit_set : !bit_set;
    if (take_branch) {
      Branch(offset);
    }
  }

  Register Load(Decoder::LoadStoreSize size, bool is_signed, bool is_64bit_target,
                Register base, int32_t offset) {
    CHECK(!exception_raised_);
    void* ptr = ToHostAddr<void>(base + offset);
    // region digitalis
    uint8_t data_bytes;
    switch (size) {
      case Decoder::LoadStoreSize::k8bit: data_bytes = 1; break;
      case Decoder::LoadStoreSize::k16bit: data_bytes = 2; break;
      case Decoder::LoadStoreSize::k32bit: data_bytes = 4; break;
      case Decoder::LoadStoreSize::k64bit: data_bytes = 8; break;
      default: Undefined(); return 0;
    }
    FaultyLoadResult fl = FaultyLoad(ptr, data_bytes);
    if (fl.is_fault) {
      HandleMemoryFault(base + offset);
      return 0;
    }
    // endregion
    uint64_t result;

    switch (size) {
      case Decoder::LoadStoreSize::k8bit: {
        uint8_t val = static_cast<uint8_t>(fl.value);
        if (is_signed) {
          if (is_64bit_target) {
            result = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int8_t>(val)));
          } else {
            result = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(
                static_cast<int8_t>(val))));
          }
        } else {
          result = val;
        }
        break;
      }
      case Decoder::LoadStoreSize::k16bit: {
        uint16_t val = static_cast<uint16_t>(fl.value);
        if (is_signed) {
          if (is_64bit_target) {
            result = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int16_t>(val)));
          } else {
            result = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(
                static_cast<int16_t>(val))));
          }
        } else {
          result = val;
        }
        break;
      }
      case Decoder::LoadStoreSize::k32bit: {
        uint32_t val = static_cast<uint32_t>(fl.value);
        if (is_signed) {
          if (is_64bit_target) {
            result = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(val)));
          } else {
            result = val;
          }
        } else {
          result = val;
        }
        break;
      }
      case Decoder::LoadStoreSize::k64bit: {
        result = fl.value;
        break;
      }
      default:
        Undefined();
        return 0;
    }

    return result;
  }

  void Store(Decoder::LoadStoreSize size, Register base, int32_t offset, Register data) {
    CHECK(!exception_raised_);
    void* ptr = ToHostAddr<void>(base + offset);
    // region digitalis
    uint8_t data_bytes;
    switch (size) {
      case Decoder::LoadStoreSize::k8bit: data_bytes = 1; break;
      case Decoder::LoadStoreSize::k16bit: data_bytes = 2; break;
      case Decoder::LoadStoreSize::k32bit: data_bytes = 4; break;
      case Decoder::LoadStoreSize::k64bit: data_bytes = 8; break;
      default: Undefined(); return;
    }
    if (FaultyStore(ptr, data_bytes, data)) {
      HandleMemoryFault(base + offset);
      return;
    }
    // endregion
  }

  Register AddImm(Register base, int32_t offset) {
    return base + offset;
  }

  void LoadPair(Decoder::LoadStoreSize size, Register base, int32_t offset,
                uint8_t rt1, uint8_t rt2, uint8_t scale) {
    CHECK(!exception_raised_);
    void* ptr1 = ToHostAddr<void>(base + offset);
    void* ptr2 = ToHostAddr<void>(base + offset + scale);
    // region digitalis
    uint8_t data_bytes = (size == Decoder::LoadStoreSize::k64bit) ? 8 : 4;
    FaultyLoadResult fl1 = FaultyLoad(ptr1, data_bytes);
    if (fl1.is_fault) { HandleMemoryFault(base + offset); return; }
    FaultyLoadResult fl2 = FaultyLoad(ptr2, data_bytes);
    if (fl2.is_fault) { HandleMemoryFault(base + offset + scale); return; }

    if (rt1 != 31) state_->cpu.x[rt1] = fl1.value;
    if (rt2 != 31) state_->cpu.x[rt2] = fl2.value;
    // endregion
  }

  void StorePair(Decoder::LoadStoreSize size, Register base, int32_t offset,
                 Register data1, Register data2, uint8_t scale) {
    CHECK(!exception_raised_);
    void* ptr1 = ToHostAddr<void>(base + offset);
    void* ptr2 = ToHostAddr<void>(base + offset + scale);
    // region digitalis
    uint8_t data_bytes = (size == Decoder::LoadStoreSize::k64bit) ? 8 : 4;
    if (FaultyStore(ptr1, data_bytes, data1)) { HandleMemoryFault(base + offset); return; }
    if (FaultyStore(ptr2, data_bytes, data2)) { HandleMemoryFault(base + offset + scale); return; }
    // endregion
  }

  // region digitalis - Apply the correct 32->64 extension to the offset
  // register before shift+add. extend_type is the raw 3-bit ARMv8
  // option field; only 010=UXTW, 011=LSL/UXTX, 110=SXTW, 111=SXTX are
  // valid for memory ops. Bug history: collapsing all four to LSL
  // silently used bits[63:32] of the X register backing a W offset,
  // corrupting addresses for SXTW or unclean upper-half UXTW. Discovered
  // chasing Brotli "Bad context map" in libsuperpack-jni.so.
  static uint64_t ApplyOffsetExtend(uint64_t reg_val, uint8_t extend_type) {
    switch (extend_type) {
      case 0b010:  // UXTW
        return reg_val & 0xFFFFFFFFULL;
      case 0b110:  // SXTW
        return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(reg_val)));
      case 0b011:  // LSL / UXTX
      case 0b111:  // SXTX
      default:
        return reg_val;
    }
  }

  Register LoadReg(Decoder::LoadStoreSize size, bool is_signed, bool is_64bit_target,
                   Register base, Register offset_reg, uint8_t extend_type,
                   uint8_t shift_amount) {
    uint64_t off = ApplyOffsetExtend(offset_reg, extend_type) << shift_amount;
    uint64_t addr = base + off;
    return Load(size, is_signed, is_64bit_target, addr, 0);
  }

  void StoreReg(Decoder::LoadStoreSize size, Register base, Register offset_reg,
                uint8_t extend_type, uint8_t shift_amount, Register data) {
    uint64_t off = ApplyOffsetExtend(offset_reg, extend_type) << shift_amount;
    uint64_t addr = base + off;
    Store(size, addr, 0, data);
  }
  // endregion

  void Svc(uint16_t /*imm*/) {
    CHECK(!exception_raised_);
    // ARM64 syscall convention: syscall number in x8, args in x0-x5, return in x0.
    RunGuestSyscall(state_);
  }

  Register Mrs(Decoder::SystemReg sysreg) {
    CHECK(!exception_raised_);
    switch (sysreg) {
      case Decoder::SystemReg::kTpidrEl0:
        return state_->tls;
      case Decoder::SystemReg::kNzcv: {
        // Convert internal flags to NZCV format: N[31] Z[30] C[29] V[28].
        uint32_t nzcv = 0;
        if (state_->cpu.flags & CPUState::kFlagNegative) nzcv |= (1u << 31);
        if (state_->cpu.flags & CPUState::kFlagZero) nzcv |= (1u << 30);
        if (state_->cpu.flags & CPUState::kFlagCarry) nzcv |= (1u << 29);
        if (state_->cpu.flags & CPUState::kFlagOverflow) nzcv |= (1u << 28);
        return nzcv;
      }
      case Decoder::SystemReg::kFpcr:
        return state_->cpu.cached_fpcr;
      case Decoder::SystemReg::kFpsr:
        return state_->cpu.emulated_fpsr;
      // region digitalis
      case Decoder::SystemReg::kCtrEl0:
        // CTR_EL0: Cache Type Register.
        // IminLine=4 (log2 of 16-byte icache line), DminLine=4 (log2 of 16-byte dcache line)
        // L1Ip=3 (PIPT), CWG=4, ERG=4
        return 0x8444c004ULL;
      case Decoder::SystemReg::kDczidEl0:
        // DCZID_EL0: Data Cache Zero ID Register.
        // DZP=1 (DC ZVA prohibited), BS=4 (log2 of 64-byte block)
        return 0x10ULL;  // DZP=1: DC ZVA not available
      case Decoder::SystemReg::kMidrEl1:
        // MIDR_EL1: Main ID Register. Cortex-A53 r0p4 layout.
        //   Implementer 0x41 ('A' = ARM Ltd)
        //   Variant 0x0, Architecture 0xF (defined by ID_AA64*_EL1)
        //   PartNum 0xD03 (Cortex-A53), Revision 0x4
        return 0x410FD034ULL;
      // endregion
      default:
        Undefined();
        return 0;
    }
  }

  void Msr(Decoder::SystemReg sysreg, Register value) {
    CHECK(!exception_raised_);
    switch (sysreg) {
      case Decoder::SystemReg::kTpidrEl0:
        state_->tls = value;
        break;
      case Decoder::SystemReg::kNzcv: {
        uint16_t flags = 0;
        if (value & (1u << 31)) flags |= CPUState::kFlagNegative;
        if (value & (1u << 30)) flags |= CPUState::kFlagZero;
        if (value & (1u << 29)) flags |= CPUState::kFlagCarry;
        if (value & (1u << 28)) flags |= CPUState::kFlagOverflow;
        state_->cpu.flags = flags;
        break;
      }
      case Decoder::SystemReg::kFpcr:
        state_->cpu.cached_fpcr = static_cast<uint32_t>(value);
        break;
      case Decoder::SystemReg::kFpsr:
        state_->cpu.emulated_fpsr = static_cast<uint32_t>(value);
        break;
      default:
        Undefined();
        break;
    }
  }

  Register LogicalShiftedReg(Decoder::LogicalShiftedRegOpcode opcode, bool is_64bit,
                             bool invert, Register src1, Register src2,
                             Decoder::ShiftType shift_type, uint8_t shift_amount) {
    CHECK(!exception_raised_);
    uint64_t operand2 = ApplyShift(src2, shift_type, shift_amount, is_64bit);
    if (invert) {
      operand2 = ~operand2;
      if (!is_64bit) operand2 &= 0xFFFFFFFFULL;
    }

    uint64_t operand1 = is_64bit ? src1 : (src1 & 0xFFFFFFFFULL);
    uint64_t result;

    switch (opcode) {
      case Decoder::LogicalShiftedRegOpcode::kAnd:
      case Decoder::LogicalShiftedRegOpcode::kAnds:
        result = operand1 & operand2;
        break;
      case Decoder::LogicalShiftedRegOpcode::kOrr:
        result = operand1 | operand2;
        break;
      case Decoder::LogicalShiftedRegOpcode::kEor:
        result = operand1 ^ operand2;
        break;
      default:
        Undefined();
        return 0;
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    if (opcode == Decoder::LogicalShiftedRegOpcode::kAnds) {
      UpdateLogicalFlags(result, is_64bit);
    }

    return result;
  }

  Register AddSubShiftedReg(bool is_sub, bool set_flags, bool is_64bit,
                            Register src1, Register src2,
                            Decoder::ShiftType shift_type, uint8_t shift_amount) {
    CHECK(!exception_raised_);
    uint64_t operand1 = is_64bit ? src1 : (src1 & 0xFFFFFFFFULL);
    uint64_t operand2 = ApplyShift(src2, shift_type, shift_amount, is_64bit);

    uint64_t result;
    if (is_sub) {
      result = operand1 - operand2;
    } else {
      result = operand1 + operand2;
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    if (set_flags) {
      UpdateFlags(operand1, operand2, result, is_sub, is_64bit);
    }

    return result;
  }

  Register AddSubExtendedReg(bool is_sub, bool set_flags, bool is_64bit,
                             Register src1, Register src2,
                             uint8_t extend_type, uint8_t shift_amount) {
    CHECK(!exception_raised_);
    uint64_t operand1 = is_64bit ? src1 : (src1 & 0xFFFFFFFFULL);
    uint64_t operand2 = ExtendReg(src2, extend_type, shift_amount);
    if (!is_64bit) operand2 &= 0xFFFFFFFFULL;

    uint64_t result;
    if (is_sub) {
      result = operand1 - operand2;
    } else {
      result = operand1 + operand2;
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    if (set_flags) {
      UpdateFlags(operand1, operand2, result, is_sub, is_64bit);
    }

    return result;
  }

  Register ConditionalSelect(Decoder::ConditionalSelectOpcode opcode, bool is_64bit,
                             Register src1, Register src2, Decoder::Condition cond) {
    CHECK(!exception_raised_);
    bool cond_true = EvaluateCondition(cond);
    uint64_t result;

    if (cond_true) {
      result = src1;
    } else {
      switch (opcode) {
        case Decoder::ConditionalSelectOpcode::kCsel:
          result = src2;
          break;
        case Decoder::ConditionalSelectOpcode::kCsinc:
          result = src2 + 1;
          break;
        case Decoder::ConditionalSelectOpcode::kCsinv:
          result = ~src2;
          break;
        case Decoder::ConditionalSelectOpcode::kCsneg:
          result = static_cast<uint64_t>(-static_cast<int64_t>(src2));
          break;
        default:
          Undefined();
          return 0;
      }
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    return result;
  }

  Register DataProc2Src(Decoder::DataProc2SrcOpcode opcode, bool is_64bit,
                        Register src1, Register src2) {
    CHECK(!exception_raised_);
    uint64_t result;

    switch (opcode) {
      case Decoder::DataProc2SrcOpcode::kUdiv: {
        if (is_64bit) {
          result = (src2 != 0) ? (src1 / src2) : 0;
        } else {
          uint32_t a = static_cast<uint32_t>(src1);
          uint32_t b = static_cast<uint32_t>(src2);
          result = (b != 0) ? (a / b) : 0;
        }
        break;
      }
      case Decoder::DataProc2SrcOpcode::kSdiv: {
        if (is_64bit) {
          int64_t a = static_cast<int64_t>(src1);
          int64_t b = static_cast<int64_t>(src2);
          if (b == 0) {
            result = 0;
          } else if (a == INT64_MIN && b == -1) {
            result = static_cast<uint64_t>(INT64_MIN);
          } else {
            result = static_cast<uint64_t>(a / b);
          }
        } else {
          int32_t a = static_cast<int32_t>(src1);
          int32_t b = static_cast<int32_t>(src2);
          if (b == 0) {
            result = 0;
          } else if (a == INT32_MIN && b == -1) {
            result = static_cast<uint64_t>(static_cast<uint32_t>(INT32_MIN));
          } else {
            result = static_cast<uint64_t>(static_cast<uint32_t>(a / b));
          }
        }
        break;
      }
      case Decoder::DataProc2SrcOpcode::kLslv:
        if (is_64bit) {
          uint8_t shift = src2 & 63;
          result = src1 << shift;
        } else {
          uint8_t shift = static_cast<uint32_t>(src2) & 31;
          result = static_cast<uint32_t>(src1) << shift;
        }
        break;
      case Decoder::DataProc2SrcOpcode::kLsrv:
        if (is_64bit) {
          uint8_t shift = src2 & 63;
          result = src1 >> shift;
        } else {
          uint8_t shift = static_cast<uint32_t>(src2) & 31;
          result = static_cast<uint32_t>(src1) >> shift;
        }
        break;
      case Decoder::DataProc2SrcOpcode::kAsrv:
        if (is_64bit) {
          uint8_t shift = src2 & 63;
          result = static_cast<uint64_t>(static_cast<int64_t>(src1) >> shift);
        } else {
          uint8_t shift = static_cast<uint32_t>(src2) & 31;
          result = static_cast<uint64_t>(static_cast<uint32_t>(
              static_cast<int32_t>(static_cast<uint32_t>(src1)) >> shift));
        }
        break;
      case Decoder::DataProc2SrcOpcode::kRorv:
        if (is_64bit) {
          uint8_t shift = src2 & 63;
          result = (src1 >> shift) | (src1 << (64 - shift));
        } else {
          uint8_t shift = static_cast<uint32_t>(src2) & 31;
          uint32_t val = static_cast<uint32_t>(src1);
          result = ((val >> shift) | (val << (32 - shift))) & 0xFFFFFFFFULL;
        }
        break;
      // region digitalis - CRC32 instructions
      case Decoder::DataProc2SrcOpcode::kCrc32b:
      case Decoder::DataProc2SrcOpcode::kCrc32h:
      case Decoder::DataProc2SrcOpcode::kCrc32w:
      case Decoder::DataProc2SrcOpcode::kCrc32x: {
        uint32_t crc = static_cast<uint32_t>(src1);
        // CRC32 uses ISO 3309 polynomial 0x04C11DB7 (bit-reversed: 0xEDB88320)
        auto crc32_byte = [](uint32_t c, uint8_t byte) -> uint32_t {
          c ^= byte;
          for (int i = 0; i < 8; i++) {
            c = (c >> 1) ^ ((c & 1) ? 0xEDB88320u : 0u);
          }
          return c;
        };
        uint8_t nbytes;
        switch (opcode) {
          case Decoder::DataProc2SrcOpcode::kCrc32b: nbytes = 1; break;
          case Decoder::DataProc2SrcOpcode::kCrc32h: nbytes = 2; break;
          case Decoder::DataProc2SrcOpcode::kCrc32w: nbytes = 4; break;
          case Decoder::DataProc2SrcOpcode::kCrc32x: nbytes = 8; break;
          default: __builtin_unreachable();
        }
        for (uint8_t i = 0; i < nbytes; i++) {
          crc = crc32_byte(crc, static_cast<uint8_t>(src2 >> (i * 8)));
        }
        result = crc;
        break;
      }
      case Decoder::DataProc2SrcOpcode::kCrc32cb:
      case Decoder::DataProc2SrcOpcode::kCrc32ch:
      case Decoder::DataProc2SrcOpcode::kCrc32cw:
      case Decoder::DataProc2SrcOpcode::kCrc32cx: {
        uint32_t crc = static_cast<uint32_t>(src1);
        // CRC32C uses Castagnoli polynomial (bit-reversed: 0x82F63B78)
        auto crc32c_byte = [](uint32_t c, uint8_t byte) -> uint32_t {
          c ^= byte;
          for (int i = 0; i < 8; i++) {
            c = (c >> 1) ^ ((c & 1) ? 0x82F63B78u : 0u);
          }
          return c;
        };
        uint8_t nbytes;
        switch (opcode) {
          case Decoder::DataProc2SrcOpcode::kCrc32cb: nbytes = 1; break;
          case Decoder::DataProc2SrcOpcode::kCrc32ch: nbytes = 2; break;
          case Decoder::DataProc2SrcOpcode::kCrc32cw: nbytes = 4; break;
          case Decoder::DataProc2SrcOpcode::kCrc32cx: nbytes = 8; break;
          default: __builtin_unreachable();
        }
        for (uint8_t i = 0; i < nbytes; i++) {
          crc = crc32c_byte(crc, static_cast<uint8_t>(src2 >> (i * 8)));
        }
        result = crc;
        break;
      }
      // endregion
      default:
        Undefined();
        return 0;
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    return result;
  }

  Register DataProc3Src(Decoder::DataProc3SrcOpcode opcode, bool is_64bit,
                        Register src1, Register src2, Register src3) {
    CHECK(!exception_raised_);
    uint64_t result;

    switch (opcode) {
      case Decoder::DataProc3SrcOpcode::kMadd:
        // MADD: Rd = Ra + Rn * Rm  (MUL is MADD with Ra=XZR)
        if (is_64bit) {
          result = src3 + (src1 * src2);
        } else {
          result = static_cast<uint32_t>(
              static_cast<uint32_t>(src3) +
              (static_cast<uint32_t>(src1) * static_cast<uint32_t>(src2)));
        }
        break;
      case Decoder::DataProc3SrcOpcode::kMsub:
        // MSUB: Rd = Ra - Rn * Rm  (MNEG is MSUB with Ra=XZR)
        if (is_64bit) {
          result = src3 - (src1 * src2);
        } else {
          result = static_cast<uint32_t>(
              static_cast<uint32_t>(src3) -
              (static_cast<uint32_t>(src1) * static_cast<uint32_t>(src2)));
        }
        break;
      case Decoder::DataProc3SrcOpcode::kSmaddl: {
        // SMADDL: Xd = Wa (sign-extended) * Wn (sign-extended) + Xa
        int64_t a = static_cast<int32_t>(static_cast<uint32_t>(src1));
        int64_t b = static_cast<int32_t>(static_cast<uint32_t>(src2));
        result = static_cast<uint64_t>(static_cast<int64_t>(src3) + a * b);
        break;
      }
      case Decoder::DataProc3SrcOpcode::kSmsubl: {
        int64_t a = static_cast<int32_t>(static_cast<uint32_t>(src1));
        int64_t b = static_cast<int32_t>(static_cast<uint32_t>(src2));
        result = static_cast<uint64_t>(static_cast<int64_t>(src3) - a * b);
        break;
      }
      case Decoder::DataProc3SrcOpcode::kSmulh: {
        // SMULH: Xd = (Xn * Xm) >> 64  (signed 128-bit multiply, return high 64 bits)
        __int128 a = static_cast<int64_t>(src1);
        __int128 b = static_cast<int64_t>(src2);
        result = static_cast<uint64_t>((a * b) >> 64);
        break;
      }
      case Decoder::DataProc3SrcOpcode::kUmaddl: {
        uint64_t a = static_cast<uint32_t>(src1);
        uint64_t b = static_cast<uint32_t>(src2);
        result = src3 + a * b;
        break;
      }
      case Decoder::DataProc3SrcOpcode::kUmsubl: {
        uint64_t a = static_cast<uint32_t>(src1);
        uint64_t b = static_cast<uint32_t>(src2);
        result = src3 - a * b;
        break;
      }
      case Decoder::DataProc3SrcOpcode::kUmulh: {
        // UMULH: Xd = (Xn * Xm) >> 64  (unsigned 128-bit multiply, return high 64 bits)
        unsigned __int128 a = src1;
        unsigned __int128 b = src2;
        result = static_cast<uint64_t>((a * b) >> 64);
        break;
      }
      default:
        Undefined();
        return 0;
    }

    if (!is_64bit && opcode != Decoder::DataProc3SrcOpcode::kSmaddl &&
        opcode != Decoder::DataProc3SrcOpcode::kSmsubl &&
        opcode != Decoder::DataProc3SrcOpcode::kSmulh &&
        opcode != Decoder::DataProc3SrcOpcode::kUmaddl &&
        opcode != Decoder::DataProc3SrcOpcode::kUmsubl &&
        opcode != Decoder::DataProc3SrcOpcode::kUmulh) {
      result &= 0xFFFFFFFFULL;
    }

    return result;
  }

  // region digitalis
  // region digitalis
  Register AddSubWithCarry(Register src1, Register src2, bool is_64bit, bool is_sub, bool set_flags) {
    CHECK(!exception_raised_);
    uint64_t op1 = is_64bit ? src1 : (src1 & 0xFFFFFFFFULL);
    uint64_t op2 = is_64bit ? src2 : (src2 & 0xFFFFFFFFULL);
    uint64_t carry = (state_->cpu.flags & CPUState::kFlagCarry) ? 1 : 0;
    uint64_t result;
    if (is_sub) {
      op2 = ~op2;
      if (!is_64bit) op2 &= 0xFFFFFFFFULL;
    }
    result = op1 + op2 + carry;
    if (!is_64bit) result &= 0xFFFFFFFFULL;
    if (set_flags) {
      UpdateFlags(op1, op2 + carry, result, is_sub, is_64bit);
    }
    return result;
  }

  Register DataProc1Src(Register src, uint8_t opcode2, bool is_64bit) {
    CHECK(!exception_raised_);
    uint64_t val = is_64bit ? src : (src & 0xFFFFFFFFULL);
    uint64_t result;
    unsigned bits = is_64bit ? 64 : 32;

    switch (opcode2) {
      case 0b000000: {
        // RBIT: reverse bits
        result = 0;
        for (unsigned i = 0; i < bits; i++) {
          if (val & (1ULL << i)) result |= (1ULL << (bits - 1 - i));
        }
        break;
      }
      case 0b000001: {
        // REV16: reverse bytes in 16-bit halfwords
        result = 0;
        for (unsigned i = 0; i < bits; i += 16) {
          uint64_t hw = (val >> i) & 0xFFFF;
          hw = ((hw & 0xFF) << 8) | ((hw >> 8) & 0xFF);
          result |= hw << i;
        }
        break;
      }
      case 0b000010:
        if (is_64bit) {
          // REV32: reverse bytes in 32-bit words (64-bit only)
          result = __builtin_bswap64(val);
          result = (result << 32) | (result >> 32);  // swap the two 32-bit halves back
          // Actually: REV32 reverses bytes within each 32-bit word
          uint32_t lo = __builtin_bswap32(static_cast<uint32_t>(val));
          uint32_t hi = __builtin_bswap32(static_cast<uint32_t>(val >> 32));
          result = (static_cast<uint64_t>(hi) << 32) | lo;
        } else {
          // REV: reverse bytes in 32-bit word
          result = __builtin_bswap32(static_cast<uint32_t>(val));
        }
        break;
      case 0b000011:
        // REV: reverse bytes in 64-bit (64-bit only)
        result = __builtin_bswap64(val);
        break;
      case 0b000100: {
        // CLZ: count leading zeros
        if (is_64bit) {
          result = val ? __builtin_clzll(val) : 64;
        } else {
          result = static_cast<uint32_t>(val) ? __builtin_clz(static_cast<uint32_t>(val)) : 32;
        }
        break;
      }
      case 0b000101: {
        // CLS: count leading sign bits (= CLZ of XOR with arithmetic shift)
        if (is_64bit) {
          int64_t sval = static_cast<int64_t>(val);
          uint64_t xored = sval ^ (sval >> 1);
          result = xored ? (__builtin_clzll(xored) - 1) : 63;
        } else {
          int32_t sval = static_cast<int32_t>(static_cast<uint32_t>(val));
          uint32_t xored = sval ^ (sval >> 1);
          result = xored ? (__builtin_clz(xored) - 1) : 31;
        }
        break;
      }
      default:
        Undefined();
        return 0;
    }

    if (!is_64bit) result &= 0xFFFFFFFFULL;
    return result;
  }
  // endregion

  // region digitalis
  //
  // AdvSIMD three different (widening): operations on narrow elements producing wide results.
  //   Q=0 uses lower half of source registers, Q=1 uses upper half.
  //   Result is always full 128-bit vector with wider elements.
  //
  void AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t dst = state_->cpu.v[args.rd];  // Needed for accumulate ops (MLAL, MLSL, ABAL)
    __uint128_t result = 0;

    // region digitalis - PMULL handles size=00 (8-bit) and size=11 (64-bit, PMULL64).
    // Dispatch it before the generic widening size table (which rejects size=11).
    if (args.opcode == Decoder::AdvSimdThreeDiffOpcode::kPmull) {
      auto poly_mul = [](uint64_t a, uint64_t b, unsigned in_bits) -> __uint128_t {
        __uint128_t res = 0;
        __uint128_t aa = a;
        for (unsigned i = 0; i < in_bits; ++i) {
          if ((b >> i) & 1u) {
            res ^= (aa << i);
          }
        }
        return res;
      };
      if (args.size == 0b00) {
        uint8_t src_n_bytes[16];
        uint8_t src_m_bytes[16];
        memcpy(src_n_bytes, &src_n, 16);
        memcpy(src_m_bytes, &src_m, 16);
        uint8_t off = args.q ? 8 : 0;
        uint16_t out_lanes[8];
        for (unsigned i = 0; i < 8; ++i) {
          out_lanes[i] =
              static_cast<uint16_t>(poly_mul(src_n_bytes[off + i], src_m_bytes[off + i], 8));
        }
        memcpy(&result, out_lanes, 16);
      } else if (args.size == 0b11) {
        uint64_t a;
        uint64_t b;
        uint8_t off = args.q ? 8 : 0;
        memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + off, 8);
        memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + off, 8);
        result = poly_mul(a, b, 64);
      } else {
        Undefined();
        return;
      }
      state_->cpu.v[args.rd] = result;
      return;
    }
    // endregion

    // Input element sizes.
    uint8_t in_esize;  // input element size in bytes
    switch (args.size) {
      case 0b00: in_esize = 1; break;  // 8-bit -> 16-bit
      case 0b01: in_esize = 2; break;  // 16-bit -> 32-bit
      case 0b10: in_esize = 4; break;  // 32-bit -> 64-bit
      default: Undefined(); return;
    }
    uint8_t out_esize = in_esize * 2;  // output element size
    uint8_t num_elements = 8 / out_esize;  // elements in output (always 128-bit result but /16 * out_esize)

    // 128-bit output, so num_elements = 16 / out_esize
    num_elements = 16 / out_esize;

    // Q=0: use lower half of source (bytes 0-7), Q=1: use upper half (bytes 8-15).
    uint8_t src_offset = args.q ? 8 : 0;

    // Helper to extract an unsigned element from source at given byte offset.
    auto get_unsigned = [&](const __uint128_t& src, uint8_t elem_idx) -> uint64_t {
      uint64_t val = 0;
      memcpy(&val, reinterpret_cast<const uint8_t*>(&src) + src_offset + elem_idx * in_esize, in_esize);
      return val;
    };

    // Helper to extract a signed element from source at given byte offset.
    auto get_signed = [&](const __uint128_t& src, uint8_t elem_idx) -> int64_t {
      uint64_t val = 0;
      memcpy(&val, reinterpret_cast<const uint8_t*>(&src) + src_offset + elem_idx * in_esize, in_esize);
      // Sign-extend
      uint8_t shift = (8 - in_esize) * 8;
      return static_cast<int64_t>(val << shift) >> shift;
    };

    // Helper to write a wide element to result.
    auto set_result = [&](uint8_t elem_idx, uint64_t val) {
      memcpy(reinterpret_cast<uint8_t*>(&result) + elem_idx * out_esize, &val, out_esize);
    };

    // Helper to get existing accumulator value.
    auto get_accum = [&](uint8_t elem_idx) -> uint64_t {
      uint64_t val = 0;
      memcpy(&val, reinterpret_cast<const uint8_t*>(&dst) + elem_idx * out_esize, out_esize);
      return val;
    };

    switch (args.opcode) {
      case Decoder::AdvSimdThreeDiffOpcode::kUaddl:
        for (uint8_t i = 0; i < num_elements; i++) {
          set_result(i, get_unsigned(src_n, i) + get_unsigned(src_m, i));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kSaddl:
        for (uint8_t i = 0; i < num_elements; i++) {
          set_result(i, static_cast<uint64_t>(get_signed(src_n, i) + get_signed(src_m, i)));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kUsubl:
        for (uint8_t i = 0; i < num_elements; i++) {
          set_result(i, get_unsigned(src_n, i) - get_unsigned(src_m, i));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kSsubl:
        for (uint8_t i = 0; i < num_elements; i++) {
          set_result(i, static_cast<uint64_t>(get_signed(src_n, i) - get_signed(src_m, i)));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kUmlal:
        for (uint8_t i = 0; i < num_elements; i++) {
          set_result(i, get_accum(i) + get_unsigned(src_n, i) * get_unsigned(src_m, i));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kSmlal:
        for (uint8_t i = 0; i < num_elements; i++) {
          int64_t prod = get_signed(src_n, i) * get_signed(src_m, i);
          set_result(i, static_cast<uint64_t>(static_cast<int64_t>(get_accum(i)) + prod));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kUmlsl:
        for (uint8_t i = 0; i < num_elements; i++) {
          set_result(i, get_accum(i) - get_unsigned(src_n, i) * get_unsigned(src_m, i));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kSmlsl:
        for (uint8_t i = 0; i < num_elements; i++) {
          int64_t prod = get_signed(src_n, i) * get_signed(src_m, i);
          set_result(i, static_cast<uint64_t>(static_cast<int64_t>(get_accum(i)) - prod));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kUmull:
        for (uint8_t i = 0; i < num_elements; i++) {
          set_result(i, get_unsigned(src_n, i) * get_unsigned(src_m, i));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kSmull:
        for (uint8_t i = 0; i < num_elements; i++) {
          set_result(i, static_cast<uint64_t>(get_signed(src_n, i) * get_signed(src_m, i)));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kUabdl:
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a = get_unsigned(src_n, i);
          uint64_t b = get_unsigned(src_m, i);
          set_result(i, a > b ? a - b : b - a);
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kSabdl:
        for (uint8_t i = 0; i < num_elements; i++) {
          int64_t a = get_signed(src_n, i);
          int64_t b = get_signed(src_m, i);
          int64_t diff = a - b;
          set_result(i, static_cast<uint64_t>(diff < 0 ? -diff : diff));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kUabal:
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a = get_unsigned(src_n, i);
          uint64_t b = get_unsigned(src_m, i);
          set_result(i, get_accum(i) + (a > b ? a - b : b - a));
        }
        break;
      case Decoder::AdvSimdThreeDiffOpcode::kSabal:
        for (uint8_t i = 0; i < num_elements; i++) {
          int64_t a = get_signed(src_n, i);
          int64_t b = get_signed(src_m, i);
          int64_t diff = a - b;
          set_result(i, get_accum(i) + static_cast<uint64_t>(diff < 0 ? -diff : diff));
        }
        break;
      // region digitalis - wide add/sub: Vn is already wide (out_esize per elem);
      // Vm is narrow (in_esize per elem, selected by Q=0 low half / Q=1 high half).
      case Decoder::AdvSimdThreeDiffOpcode::kUaddw:
      case Decoder::AdvSimdThreeDiffOpcode::kSaddw:
      case Decoder::AdvSimdThreeDiffOpcode::kUsubw:
      case Decoder::AdvSimdThreeDiffOpcode::kSsubw: {
        bool is_signed = (args.opcode == Decoder::AdvSimdThreeDiffOpcode::kSaddw ||
                          args.opcode == Decoder::AdvSimdThreeDiffOpcode::kSsubw);
        bool is_sub    = (args.opcode == Decoder::AdvSimdThreeDiffOpcode::kSsubw ||
                          args.opcode == Decoder::AdvSimdThreeDiffOpcode::kUsubw);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t wide_n = 0;
          memcpy(&wide_n, reinterpret_cast<const uint8_t*>(&src_n) + i * out_esize, out_esize);
          uint64_t r;
          if (is_signed) {
            int64_t sn = static_cast<int64_t>(wide_n << ((8 - out_esize) * 8)) >> ((8 - out_esize) * 8);
            int64_t sm = get_signed(src_m, i);
            r = static_cast<uint64_t>(is_sub ? sn - sm : sn + sm);
          } else {
            uint64_t un = wide_n;
            uint64_t um = get_unsigned(src_m, i);
            r = is_sub ? un - um : un + um;
          }
          set_result(i, r);
        }
        break;
      }
      // endregion
      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = result;
  }
  // endregion

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    CHECK(!exception_raised_);
    // EXT: extract bytes from concatenation of Vn:Vm at byte position index.
    uint8_t bytes[32];
    memcpy(bytes, &state_->cpu.v[rn], 16);
    memcpy(bytes + 16, &state_->cpu.v[rm], 16);
    __uint128_t result = 0;
    unsigned num_bytes = q ? 16 : 8;
    memcpy(&result, bytes + index, num_bytes);
    state_->cpu.v[rd] = result;
  }

  // region digitalis - TBL / TBX (vector table lookup). Reads `len+1`
  // consecutive Q registers starting at Rn to form a 16/32/48/64-byte
  // table, then for each lane i of Vm uses Vm[i] as an index into the
  // table.
  //   TBL: out-of-range indices produce 0.
  //   TBX: out-of-range indices preserve the existing Vd byte.
  // Per the ARM ARM, when len+1 source registers are used, they form a
  // single linear byte table -- Vn, V(n+1)%32, V(n+2)%32, V(n+3)%32.
  void AdvSimdTableLookup(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t len,
                          uint8_t op, bool q) {
    CHECK(!exception_raised_);
    uint8_t table_regs = len + 1;
    uint8_t table[64] = {};
    uint8_t table_bytes = table_regs * 16;
    for (uint8_t r = 0; r < table_regs; r++) {
      __uint128_t v = state_->cpu.v[(rn + r) % 32];
      memcpy(table + r * 16, &v, 16);
    }
    __uint128_t vm_val = state_->cpu.v[rm];
    uint8_t idx_bytes[16];
    memcpy(idx_bytes, &vm_val, 16);

    uint8_t out[16];
    if (op /*TBX*/) {
      __uint128_t vd_val = state_->cpu.v[rd];
      memcpy(out, &vd_val, 16);
    } else {
      memset(out, 0, sizeof(out));
    }

    uint8_t num_bytes = q ? 16 : 8;
    for (uint8_t i = 0; i < num_bytes; i++) {
      uint8_t idx = idx_bytes[i];
      if (idx < table_bytes) {
        out[i] = table[idx];
      }
      // else: TBL leaves 0 (already zeroed), TBX leaves the existing Vd byte.
    }

    // Upper bytes of result for Q=0 (8-byte) form should be zeroed.
    __uint128_t result = 0;
    memcpy(&result, out, num_bytes);
    state_->cpu.v[rd] = result;
  }
  // endregion

  // region digitalis
  // Cryptographic AES — ARMv8 crypto extension (used by libcrypto / TLS in
  // apps like WhatsApp). Spec: ARM ARM C7.2.1 (AESE/AESD/AESMC/AESIMC).
  //   opcode 00 = AESE   : Vd = ShiftRows(SubBytes(Vd XOR Vn))
  //   opcode 01 = AESD   : Vd = InvShiftRows(InvSubBytes(Vd XOR Vn))
  //   opcode 10 = AESMC  : Vd = MixColumns(Vn)
  //   opcode 11 = AESIMC : Vd = InvMixColumns(Vn)
  void CryptoAes(uint8_t rd, uint8_t rn, uint8_t opcode) {
    CHECK(!exception_raised_);
    // AES forward S-box (FIPS-197 Figure 7).
    static const uint8_t kSbox[256] = {
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    };
    // AES inverse S-box (FIPS-197 Figure 14).
    static const uint8_t kInvSbox[256] = {
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
    };
    // ShiftRows / InvShiftRows byte permutation tables (out[i] = in[table[i]]).
    static const uint8_t kShiftRows[16]    = { 0, 5, 10, 15, 4, 9, 14,  3, 8, 13,  2,  7, 12,  1,  6, 11 };
    static const uint8_t kInvShiftRows[16] = { 0, 13, 10, 7, 4, 1, 14, 11, 8,  5,  2, 15, 12,  9,  6,  3 };

    auto xtime = [](uint8_t x) -> uint8_t {
      return static_cast<uint8_t>((x << 1) ^ ((x & 0x80) ? 0x1b : 0));
    };
    auto mul_gf = [&](uint8_t x, uint8_t coeff) -> uint8_t {
      // coeff is small; standard MixColumns / InvMixColumns coefficients only.
      uint8_t x2 = xtime(x);
      uint8_t x4 = xtime(x2);
      uint8_t x8 = xtime(x4);
      switch (coeff) {
        case 1:  return x;
        case 2:  return x2;
        case 3:  return static_cast<uint8_t>(x2 ^ x);
        case 9:  return static_cast<uint8_t>(x8 ^ x);
        case 11: return static_cast<uint8_t>(x8 ^ x2 ^ x);
        case 13: return static_cast<uint8_t>(x8 ^ x4 ^ x);
        case 14: return static_cast<uint8_t>(x8 ^ x4 ^ x2);
        default: return 0;  // unreachable for AES
      }
    };

    uint8_t in[16];
    uint8_t out[16];
    if (opcode == 0b00 || opcode == 0b01) {
      // AESE / AESD: input = Vd XOR Vn (AddRoundKey).
      __uint128_t d = state_->cpu.v[rd];
      __uint128_t n = state_->cpu.v[rn];
      __uint128_t x = d ^ n;
      memcpy(in, &x, 16);
      if (opcode == 0b00) {
        // AESE: SubBytes then ShiftRows.
        uint8_t sb[16];
        for (int i = 0; i < 16; i++) sb[i] = kSbox[in[i]];
        for (int i = 0; i < 16; i++) out[i] = sb[kShiftRows[i]];
      } else {
        // AESD: InvShiftRows then InvSubBytes.
        uint8_t isr[16];
        for (int i = 0; i < 16; i++) isr[i] = in[kInvShiftRows[i]];
        for (int i = 0; i < 16; i++) out[i] = kInvSbox[isr[i]];
      }
    } else {
      // AESMC / AESIMC: input = Vn (no XOR).
      __uint128_t n = state_->cpu.v[rn];
      memcpy(in, &n, 16);
      if (opcode == 0b10) {
        // AESMC: MixColumns, matrix [2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2].
        for (int c = 0; c < 4; c++) {
          uint8_t a0 = in[c * 4 + 0];
          uint8_t a1 = in[c * 4 + 1];
          uint8_t a2 = in[c * 4 + 2];
          uint8_t a3 = in[c * 4 + 3];
          out[c * 4 + 0] = mul_gf(a0, 2) ^ mul_gf(a1, 3) ^ a2 ^ a3;
          out[c * 4 + 1] = a0 ^ mul_gf(a1, 2) ^ mul_gf(a2, 3) ^ a3;
          out[c * 4 + 2] = a0 ^ a1 ^ mul_gf(a2, 2) ^ mul_gf(a3, 3);
          out[c * 4 + 3] = mul_gf(a0, 3) ^ a1 ^ a2 ^ mul_gf(a3, 2);
        }
      } else {
        // AESIMC: InvMixColumns, matrix [14 11 13 9; 9 14 11 13; 13 9 14 11; 11 13 9 14].
        for (int c = 0; c < 4; c++) {
          uint8_t a0 = in[c * 4 + 0];
          uint8_t a1 = in[c * 4 + 1];
          uint8_t a2 = in[c * 4 + 2];
          uint8_t a3 = in[c * 4 + 3];
          out[c * 4 + 0] = mul_gf(a0, 14) ^ mul_gf(a1, 11) ^ mul_gf(a2, 13) ^ mul_gf(a3,  9);
          out[c * 4 + 1] = mul_gf(a0,  9) ^ mul_gf(a1, 14) ^ mul_gf(a2, 11) ^ mul_gf(a3, 13);
          out[c * 4 + 2] = mul_gf(a0, 13) ^ mul_gf(a1,  9) ^ mul_gf(a2, 14) ^ mul_gf(a3, 11);
          out[c * 4 + 3] = mul_gf(a0, 11) ^ mul_gf(a1, 13) ^ mul_gf(a2,  9) ^ mul_gf(a3, 14);
        }
      }
    }
    __uint128_t result;
    memcpy(&result, out, 16);
    state_->cpu.v[rd] = result;
  }
  // endregion

  // region digitalis
  // Cryptographic three-register SHA — ARMv8 crypto extension. This cycle
  // implements the SHA-1 round-mix variants (SHA1C/SHA1P/SHA1M), which differ
  // only by which choice function f(B,C,D) is used, plus SHA1SU0 (message
  // schedule helper) and the SHA-256 round-mix (SHA256H/H2) + schedule
  // helper SHA256SU1. The undefined opcode 111 still falls through to
  // Undefined().
  //
  // Spec: ARM ARM C7.2.71/72/73 (SHA1C/SHA1P/SHA1M), C7.2.75 (SHA1SU0),
  //       C7.2.77/78 (SHA256H/H2), C7.2.80 (SHA256SU1).
  //
  // SHA1C/SHA1P/SHA1M:
  //   Qd holds {A,B,C,D} in lanes 0..3; Sn = e (32-bit input);
  //   Vm.4S holds the 4 schedule words W[0..3].
  //   For j = 0..3:
  //     t = ROL(A, 5) + f(B,C,D) + e + W[j]
  //     e, D, C, B, A <- D, C, ROL(B,30), A, t
  //   Qd <- {A,B,C,D}.
  //   f for SHA1C: (B & C) | (~B & D)
  //   f for SHA1P: B ^ C ^ D
  //   f for SHA1M: (B & C) | (B & D) | (C & D)
  //
  // SHA1SU0 (message schedule update, part 1 of 2):
  //   T<127:64> = Vn<63:0>;    // lanes 2,3 of result = lanes 0,1 of Vn
  //   T<63:0>   = Vd<127:64>;  // lanes 0,1 of result = lanes 2,3 of Vd
  //   Vd = T EOR Vd EOR Vm.
  //   Equivalently per-lane:
  //     result[0] = Vd[2] XOR Vd[0] XOR Vm[0]
  //     result[1] = Vd[3] XOR Vd[1] XOR Vm[1]
  //     result[2] = Vn[0] XOR Vd[2] XOR Vm[2]
  //     result[3] = Vn[1] XOR Vd[3] XOR Vm[3]
  //   Together with SHA1SU1 this computes W[t..t+3] from W[t-16..t-1].
  //
  // SHA256H (X = Vd = {A,B,C,D}, Y = Vn = {E,F,G,H}, W = Vm = K+wt):
  //   For e = 0..3:
  //     chs = (Y0&Y1)^(~Y0&Y2)               // Ch(E,F,G)
  //     maj = (X0&X1)^(X0&X2)^(X1&X2)        // Maj(A,B,C)
  //     t   = Y3 + Sigma1(Y0) + chs + W[e]   // T1 of FIPS-180-4
  //     X3  = t + X3                         // D_new = T1 + D
  //     Y3  = t + Sigma0(X0) + maj           // H_new = T1 + T2
  //     ROL({Y,X}, 32): new X = {Y3, X0, X1, X2}, new Y = {X3, Y0, Y1, Y2}
  //   Vd <- X.
  //   SHA256H2 is the same loop but with X = Vn and Y = Vd, returning Y to Vd.
  //   Sigma0(x) = ROR(x,2) ^ ROR(x,13) ^ ROR(x,22)
  //   Sigma1(x) = ROR(x,6) ^ ROR(x,11) ^ ROR(x,25)
  //
  // SHA256SU1 (message schedule helper, part 2 of 2):
  //   d holds W[t..t+3] + σ0(W[t+1..t+4]) (from a prior SHA256SU0).
  //   n holds W[t+8..t+11], m holds W[t+12..t+15].
  //   d[0] += σ1(m[2]) + n[1];                    // → W[t+16]
  //   d[1] += σ1(m[3]) + n[2];                    // → W[t+17]
  //   d[2] += σ1(d[0]_new) + n[3];                // → W[t+18]
  //   d[3] += σ1(d[1]_new) + m[0];                // → W[t+19]
  //   where σ1(x) = ROR(x,17) ^ ROR(x,19) ^ (x >> 10). Note that lane 0 of
  //   the n operand is unused (a quirk of the ARM ARM definition; in
  //   practice OpenSSL builds n by `ext` so that lanes 1..3 align).
  void CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode) {
    CHECK(!exception_raised_);
    if (opcode > 0b110) {
      // Undefined (111).
      Undefined();
      return;
    }
    __uint128_t qd = state_->cpu.v[rd];
    __uint128_t vn = state_->cpu.v[rn];
    __uint128_t vm = state_->cpu.v[rm];
    auto unpack4 = [](__uint128_t v, uint32_t out[4]) {
      out[0] = static_cast<uint32_t>(v);
      out[1] = static_cast<uint32_t>(v >> 32);
      out[2] = static_cast<uint32_t>(v >> 64);
      out[3] = static_cast<uint32_t>(v >> 96);
    };
    auto pack4 = [](const uint32_t in[4]) -> __uint128_t {
      return static_cast<__uint128_t>(in[0]) |
             (static_cast<__uint128_t>(in[1]) << 32) |
             (static_cast<__uint128_t>(in[2]) << 64) |
             (static_cast<__uint128_t>(in[3]) << 96);
    };
    if (opcode == 0b011) {
      // SHA1SU0.
      uint32_t d0 = static_cast<uint32_t>(qd);
      uint32_t d1 = static_cast<uint32_t>(qd >> 32);
      uint32_t d2 = static_cast<uint32_t>(qd >> 64);
      uint32_t d3 = static_cast<uint32_t>(qd >> 96);
      uint32_t n0 = static_cast<uint32_t>(vn);
      uint32_t n1 = static_cast<uint32_t>(vn >> 32);
      uint32_t m0 = static_cast<uint32_t>(vm);
      uint32_t m1 = static_cast<uint32_t>(vm >> 32);
      uint32_t m2 = static_cast<uint32_t>(vm >> 64);
      uint32_t m3 = static_cast<uint32_t>(vm >> 96);
      uint32_t t0 = d2 ^ d0 ^ m0;
      uint32_t t1 = d3 ^ d1 ^ m1;
      uint32_t t2 = n0 ^ d2 ^ m2;
      uint32_t t3 = n1 ^ d3 ^ m3;
      state_->cpu.v[rd] = static_cast<__uint128_t>(t0) |
                          (static_cast<__uint128_t>(t1) << 32) |
                          (static_cast<__uint128_t>(t2) << 64) |
                          (static_cast<__uint128_t>(t3) << 96);
      return;
    }
    if (opcode == 0b100 || opcode == 0b101) {
      // SHA256H (100) / SHA256H2 (101).
      // ARM ARM: for SHA256H, X = Vd, Y = Vn, write result back to Vd as X.
      // For SHA256H2, X = Vn, Y = Vd, write result back to Vd as Y.
      auto BigSigma0 = [](uint32_t x) -> uint32_t {
        return ((x >> 2) | (x << 30)) ^ ((x >> 13) | (x << 19)) ^
               ((x >> 22) | (x << 10));
      };
      auto BigSigma1 = [](uint32_t x) -> uint32_t {
        return ((x >> 6) | (x << 26)) ^ ((x >> 11) | (x << 21)) ^
               ((x >> 25) | (x << 7));
      };
      uint32_t x[4], y[4], w[4];
      if (opcode == 0b100) {
        unpack4(qd, x);
        unpack4(vn, y);
      } else {
        unpack4(vn, x);
        unpack4(qd, y);
      }
      unpack4(vm, w);
      for (int e = 0; e < 4; e++) {
        uint32_t chs = (y[0] & y[1]) ^ (~y[0] & y[2]);
        uint32_t maj = (x[0] & x[1]) ^ (x[0] & x[2]) ^ (x[1] & x[2]);
        uint32_t t = y[3] + BigSigma1(y[0]) + chs + w[e];
        uint32_t new_x3 = t + x[3];
        uint32_t new_y3 = t + BigSigma0(x[0]) + maj;
        uint32_t x0 = x[0], x1 = x[1], x2 = x[2];
        uint32_t y0 = y[0], y1 = y[1], y2 = y[2];
        x[0] = new_y3; x[1] = x0; x[2] = x1; x[3] = x2;
        y[0] = new_x3; y[1] = y0; y[2] = y1; y[3] = y2;
      }
      state_->cpu.v[rd] = pack4(opcode == 0b100 ? x : y);
      return;
    }
    if (opcode == 0b110) {
      // SHA256SU1.
      auto LittleSigma1 = [](uint32_t x) -> uint32_t {
        return ((x >> 17) | (x << 15)) ^ ((x >> 19) | (x << 13)) ^ (x >> 10);
      };
      uint32_t d[4], n[4], m[4];
      unpack4(qd, d);
      unpack4(vn, n);
      unpack4(vm, m);
      uint32_t nd0 = d[0] + LittleSigma1(m[2]) + n[1];
      uint32_t nd1 = d[1] + LittleSigma1(m[3]) + n[2];
      uint32_t nd2 = d[2] + LittleSigma1(nd0) + n[3];
      uint32_t nd3 = d[3] + LittleSigma1(nd1) + m[0];
      uint32_t out[4] = {nd0, nd1, nd2, nd3};
      state_->cpu.v[rd] = pack4(out);
      return;
    }
    // SHA1C/SHA1P/SHA1M.
    uint32_t a = static_cast<uint32_t>(qd);
    uint32_t b = static_cast<uint32_t>(qd >> 32);
    uint32_t c = static_cast<uint32_t>(qd >> 64);
    uint32_t d = static_cast<uint32_t>(qd >> 96);
    uint32_t e = static_cast<uint32_t>(vn);  // Sn (32-bit)
    uint32_t w[4] = {
        static_cast<uint32_t>(vm),
        static_cast<uint32_t>(vm >> 32),
        static_cast<uint32_t>(vm >> 64),
        static_cast<uint32_t>(vm >> 96),
    };
    for (int j = 0; j < 4; j++) {
      uint32_t f;
      switch (opcode) {
        case 0b000: f = (b & c) | (~b & d); break;            // SHA1C
        case 0b001: f = b ^ c ^ d; break;                     // SHA1P
        case 0b010: f = (b & c) | (b & d) | (c & d); break;   // SHA1M
        default: Undefined(); return;
      }
      uint32_t rol5_a = (a << 5) | (a >> 27);
      uint32_t t = rol5_a + f + e + w[j];
      e = d;
      d = c;
      c = (b << 30) | (b >> 2);  // ROL(B, 30)
      b = a;
      a = t;
    }
    state_->cpu.v[rd] = static_cast<__uint128_t>(a) |
                        (static_cast<__uint128_t>(b) << 32) |
                        (static_cast<__uint128_t>(c) << 64) |
                        (static_cast<__uint128_t>(d) << 96);
  }

  // Cryptographic two-register SHA. Implements SHA1H, SHA1SU1, and SHA256SU0.
  // Only opcode 11 (Undefined) falls through to Undefined() now.
  //
  // Spec: ARM ARM C7.2.74 (SHA1H), C7.2.76 (SHA1SU1), C7.2.79 (SHA256SU0).
  //
  // SHA1H <Sd>, <Sn>:
  //   Sd[31:0] = ROL(Sn[31:0], 30); Sd[127:32] = 0.
  //
  // SHA1SU1 <Vd>.4S, <Vn>.4S (message schedule update, part 2 of 2):
  //   Step 1: d[i] ^= Vn[i+1] for i = 0..2   (lane-shifted XOR, lane 3 untouched)
  //   Step 2: d[i] = ROL(d[i], 1) for i = 0..2
  //   Step 3: d[3] = ROL(d[3] ^ d[0], 1)     (d[0] here is post-step-2 = W[t+16])
  //   This completes the schedule: d now holds the next four ROL1'd words.
  //
  // SHA256SU0 <Vd>.4S, <Vn>.4S (message schedule update, part 1 of 2):
  //   d[i] = Vd[i] + σ0(Vd[i+1]) for i = 0..2
  //   d[3] = Vd[3] + σ0(Vn[0])
  //   where σ0(x) = ROR(x,7) ^ ROR(x,18) ^ (x >> 3) (FIPS-180-4 lowercase σ0).
  //   Together with SHA256SU1 this computes W[t+16..t+19] from W[t..t+15].
  void CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode) {
    CHECK(!exception_raised_);
    if (opcode == 0b00) {
      // SHA1H.
      uint32_t n = static_cast<uint32_t>(state_->cpu.v[rn]);
      uint32_t result = (n << 30) | (n >> 2);  // ROL by 30 == ROR by 2.
      state_->cpu.v[rd] = static_cast<__uint128_t>(result);
      return;
    }
    if (opcode == 0b01) {
      // SHA1SU1.
      __uint128_t vd = state_->cpu.v[rd];
      __uint128_t vn = state_->cpu.v[rn];
      uint32_t d[4] = {
          static_cast<uint32_t>(vd),
          static_cast<uint32_t>(vd >> 32),
          static_cast<uint32_t>(vd >> 64),
          static_cast<uint32_t>(vd >> 96),
      };
      uint32_t n[4] = {
          static_cast<uint32_t>(vn),
          static_cast<uint32_t>(vn >> 32),
          static_cast<uint32_t>(vn >> 64),
          static_cast<uint32_t>(vn >> 96),
      };
      for (int i = 0; i < 3; i++) {
        d[i] ^= n[i + 1];
      }
      for (int i = 0; i < 3; i++) {
        d[i] = (d[i] << 1) | (d[i] >> 31);  // ROL by 1
      }
      uint32_t t3 = d[3] ^ d[0];
      d[3] = (t3 << 1) | (t3 >> 31);
      state_->cpu.v[rd] = static_cast<__uint128_t>(d[0]) |
                          (static_cast<__uint128_t>(d[1]) << 32) |
                          (static_cast<__uint128_t>(d[2]) << 64) |
                          (static_cast<__uint128_t>(d[3]) << 96);
      return;
    }
    if (opcode == 0b10) {
      // SHA256SU0.
      auto LittleSigma0 = [](uint32_t x) -> uint32_t {
        return ((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3);
      };
      __uint128_t vd = state_->cpu.v[rd];
      __uint128_t vn = state_->cpu.v[rn];
      uint32_t d[4] = {
          static_cast<uint32_t>(vd),
          static_cast<uint32_t>(vd >> 32),
          static_cast<uint32_t>(vd >> 64),
          static_cast<uint32_t>(vd >> 96),
      };
      uint32_t n0 = static_cast<uint32_t>(vn);
      uint32_t out0 = d[0] + LittleSigma0(d[1]);
      uint32_t out1 = d[1] + LittleSigma0(d[2]);
      uint32_t out2 = d[2] + LittleSigma0(d[3]);
      uint32_t out3 = d[3] + LittleSigma0(n0);
      state_->cpu.v[rd] = static_cast<__uint128_t>(out0) |
                          (static_cast<__uint128_t>(out1) << 32) |
                          (static_cast<__uint128_t>(out2) << 64) |
                          (static_cast<__uint128_t>(out3) << 96);
      return;
    }
    // Undefined (11).
    Undefined();
  }
  // endregion

  // region digitalis
  void AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size,
                      uint8_t opcode, bool q) {
    CHECK(!exception_raised_);
    uint8_t esize;
    switch (size) {
      case 0b00: esize = 1; break;
      case 0b01: esize = 2; break;
      case 0b10: esize = 4; break;
      case 0b11: esize = 8; break;
      default: Undefined(); return;
    }
    uint8_t vec_len = q ? 16 : 8;
    uint8_t num_elements = vec_len / esize;  // elements per source
    __uint128_t src_n = state_->cpu.v[rn];
    __uint128_t src_m = state_->cpu.v[rm];
    __uint128_t result = 0;
    switch (opcode) {
      case 0b001: // UZP1: even elements (0, 2, 4, ...) from Vn then Vm
      case 0b101: { // UZP2: odd elements (1, 3, 5, ...) from Vn then Vm
        uint8_t start = (opcode == 0b101) ? 1 : 0;
        uint8_t pairs = num_elements / 2;
        for (uint8_t i = 0; i < pairs; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src_n) + (i * 2 + start) * esize, esize);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &elem, esize);
        }
        for (uint8_t i = 0; i < pairs; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src_m) + (i * 2 + start) * esize, esize);
          memcpy(reinterpret_cast<uint8_t*>(&result) + (pairs + i) * esize, &elem, esize);
        }
        break;
      }
      case 0b010: // TRN1: even-indexed transpose
      case 0b110: { // TRN2: odd-indexed transpose
        uint8_t start = (opcode == 0b110) ? 1 : 0;
        for (uint8_t i = 0; i < num_elements; i += 2) {
          uint64_t elem_n = 0, elem_m = 0;
          memcpy(&elem_n, reinterpret_cast<const uint8_t*>(&src_n) + (i + start) * esize, esize);
          memcpy(&elem_m, reinterpret_cast<const uint8_t*>(&src_m) + (i + start) * esize, esize);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &elem_n, esize);
          memcpy(reinterpret_cast<uint8_t*>(&result) + (i + 1) * esize, &elem_m, esize);
        }
        break;
      }
      case 0b011: // ZIP1: interleave lower halves
      case 0b111: { // ZIP2: interleave upper halves
        uint8_t half = num_elements / 2;
        uint8_t start = (opcode == 0b111) ? half : 0;
        for (uint8_t i = 0; i < half; i++) {
          uint64_t elem_n = 0, elem_m = 0;
          memcpy(&elem_n, reinterpret_cast<const uint8_t*>(&src_n) + (start + i) * esize, esize);
          memcpy(&elem_m, reinterpret_cast<const uint8_t*>(&src_m) + (start + i) * esize, esize);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2 * esize, &elem_n, esize);
          memcpy(reinterpret_cast<uint8_t*>(&result) + (i * 2 + 1) * esize, &elem_m, esize);
        }
        break;
      }
      default:
        Undefined();
        return;
    }
    state_->cpu.v[rd] = result;
  }
  // endregion

  // region digitalis
  //
  // Multi-structure load/store. Two distinct families share this entry:
  //
  //   is_interleaved == false  ->  LD1 / ST1 with N contiguous registers.
  //                                Each Vreg gets vec_bytes of memory in
  //                                sequence. No element reordering -- what
  //                                NEON-optimised memcpy / strcmp in Bionic
  //                                libc emits.
  //
  //   is_interleaved == true   ->  LD2 / LD3 / LD4 (and their store dual).
  //                                Memory is element-interleaved across N
  //                                vectors; the load de-interleaves into
  //                                V[rt..rt+N-1], the store interleaves
  //                                from them. Used by audio/image codecs
  //                                and compression libraries (e.g.
  //                                Facebook's superpack).
  //
  // Treating the interleaved family as contiguous (the prior behaviour) was
  // silently wrong; treating the contiguous family as interleaved is also
  // silently wrong. The decoder splits the two via the is_interleaved flag.
  //
  void AdvSimdMultiStruct(uint8_t rt, uint8_t rn, uint8_t num_regs, uint8_t size,
                          bool q, bool is_store, bool postindex, uint8_t rm,
                          bool is_interleaved) {
    CHECK(!exception_raised_);
    uint8_t vec_bytes = q ? 16 : 8;
    uint64_t base_addr = (rn == 31) ? GetSp() : state_->cpu.x[rn];

    if (!is_interleaved) {
      // LD1 / ST1 multi-reg — contiguous. Bulk 8-byte transfers per vec.
      for (uint8_t r = 0; r < num_regs; r++) {
        uint8_t vreg = (rt + r) & 31;
        uint64_t addr = base_addr + r * vec_bytes;
        if (is_store) {
          __uint128_t val = state_->cpu.v[vreg];
          void* ptr = ToHostAddr<void>(addr);
          if (FaultyStore(ptr, 8, static_cast<uint64_t>(val))) {
            HandleMemoryFault(addr); return;
          }
          if (vec_bytes > 8) {
            uint64_t hi = static_cast<uint64_t>(val >> 64);
            if (FaultyStore(static_cast<uint8_t*>(ptr) + 8, 8, hi)) {
              HandleMemoryFault(addr + 8); return;
            }
          }
        } else {
          void* ptr = ToHostAddr<void>(addr);
          FaultyLoadResult lo = FaultyLoad(ptr, 8);
          if (lo.is_fault) { HandleMemoryFault(addr); return; }
          if (vec_bytes > 8) {
            FaultyLoadResult hi = FaultyLoad(static_cast<uint8_t*>(ptr) + 8, 8);
            if (hi.is_fault) { HandleMemoryFault(addr + 8); return; }
            state_->cpu.v[vreg] = static_cast<__uint128_t>(lo.value) |
                                  (static_cast<__uint128_t>(hi.value) << 64);
          } else {
            state_->cpu.v[vreg] = static_cast<__uint128_t>(lo.value);
          }
        }
      }
    } else {
      // LD2 / LD3 / LD4 / ST2 / ST3 / ST4 — element-interleaved memory.
      // Stage everything through a packed buffer (max 64 B = 4 regs * 16 B),
      // then de-interleave on load or interleave on store. Bulk 8-byte
      // FaultyLoad/Store calls keep syscall overhead low.
      uint8_t esize = 1u << size;  // 1, 2, 4, or 8 bytes per element
      if (esize > vec_bytes) { Undefined(); return; }
      uint8_t elems_per_vec = vec_bytes / esize;
      uint64_t total_bytes = static_cast<uint64_t>(num_regs) * vec_bytes;
      alignas(16) uint8_t buf[64];

      if (is_store) {
        // Pack: buf[e*n + r] = V[rt+r][e]
        for (uint8_t e = 0; e < elems_per_vec; e++) {
          for (uint8_t r = 0; r < num_regs; r++) {
            uint8_t vreg = (rt + r) & 31;
            __uint128_t val = state_->cpu.v[vreg];
            memcpy(buf + (e * num_regs + r) * esize,
                   reinterpret_cast<const uint8_t*>(&val) + e * esize, esize);
          }
        }
        void* base_ptr = ToHostAddr<void>(base_addr);
        for (uint64_t off = 0; off + 8 <= total_bytes; off += 8) {
          uint64_t chunk;
          memcpy(&chunk, buf + off, 8);
          if (FaultyStore(static_cast<uint8_t*>(base_ptr) + off, 8, chunk)) {
            HandleMemoryFault(base_addr + off); return;
          }
        }
      } else {
        void* base_ptr = ToHostAddr<void>(base_addr);
        for (uint64_t off = 0; off + 8 <= total_bytes; off += 8) {
          FaultyLoadResult res = FaultyLoad(static_cast<uint8_t*>(base_ptr) + off, 8);
          if (res.is_fault) { HandleMemoryFault(base_addr + off); return; }
          memcpy(buf + off, &res.value, 8);
        }
        // De-interleave: V[rt+r][e] = buf[e*n + r]
        for (uint8_t r = 0; r < num_regs; r++) {
          __uint128_t val = 0;  // upper lanes zero when Q=0
          for (uint8_t e = 0; e < elems_per_vec; e++) {
            memcpy(reinterpret_cast<uint8_t*>(&val) + e * esize,
                   buf + (e * num_regs + r) * esize, esize);
          }
          uint8_t vreg = (rt + r) & 31;
          state_->cpu.v[vreg] = val;
        }
      }
    }

    if (postindex) {
      int64_t post_offset;
      if (rm == 31) {
        post_offset = num_regs * vec_bytes;  // immediate post-index
      } else {
        post_offset = static_cast<int64_t>(state_->cpu.x[rm]);  // register post-index
      }
      if (rn == 31) {
        SetSp(base_addr + post_offset);
      } else {
        state_->cpu.x[rn] = base_addr + post_offset;
      }
    }
  }

  //
  // AdvSIMD load/store single structure: LD1R-LD4R (replicate) and LD/ST to one lane.
  //
  void AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs& args) {
    CHECK(!exception_raised_);

    uint64_t base_addr = (args.rn == 31) ? GetSp() : state_->cpu.x[args.rn];

    // Element size in bytes.
    uint8_t esize = 1u << args.size;
    uint8_t vec_bytes = args.q ? 16 : 8;

    if (args.is_replicate) {
      // LD1R-LD4R: Load one element per register, replicate to all lanes.
      for (uint8_t r = 0; r < args.num_regs; r++) {
        uint8_t vreg = (args.rt + r) & 31;
        uint64_t addr = base_addr + r * esize;

        FaultyLoadResult res = FaultyLoad(ToHostAddr<void>(addr), esize);
        if (res.is_fault) { HandleMemoryFault(addr); return; }

        // Replicate the loaded element across all lanes.
        __uint128_t val = 0;
        uint8_t num_lanes = vec_bytes / esize;
        for (uint8_t lane = 0; lane < num_lanes; lane++) {
          uint64_t elem = res.value;
          // Mask to element size.
          if (esize < 8) elem &= (1ULL << (esize * 8)) - 1;
          memcpy(reinterpret_cast<uint8_t*>(&val) + lane * esize, &elem, esize);
        }
        // Clear upper 64 bits if Q=0.
        if (!args.q) {
          uint64_t lo;
          memcpy(&lo, &val, 8);
          val = 0;
          memcpy(&val, &lo, 8);
        }
        state_->cpu.v[vreg] = val;
      }

      // Post-index.
      if (args.postindex) {
        int64_t offset;
        if (args.rm == 31) {
          offset = args.num_regs * esize;  // Immediate: total bytes loaded.
        } else {
          offset = static_cast<int64_t>(state_->cpu.x[args.rm]);
        }
        if (args.rn == 31) {
          SetSp(base_addr + offset);
        } else {
          state_->cpu.x[args.rn] = base_addr + offset;
        }
      }
    } else {
      // LD/ST single element to/from one lane.
      for (uint8_t r = 0; r < args.num_regs; r++) {
        uint8_t vreg = (args.rt + r) & 31;
        uint64_t addr = base_addr + r * esize;

        bool is_store = (args.op == Decoder::AdvSimdSingleStructOp::kSt1 ||
                         args.op == Decoder::AdvSimdSingleStructOp::kSt2 ||
                         args.op == Decoder::AdvSimdSingleStructOp::kSt3 ||
                         args.op == Decoder::AdvSimdSingleStructOp::kSt4);

        if (is_store) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&state_->cpu.v[vreg]) + args.index * esize, esize);
          if (FaultyStore(ToHostAddr<void>(addr), esize, elem)) {
            HandleMemoryFault(addr); return;
          }
        } else {
          FaultyLoadResult res = FaultyLoad(ToHostAddr<void>(addr), esize);
          if (res.is_fault) { HandleMemoryFault(addr); return; }
          // Write to specific lane, preserving other lanes.
          __uint128_t vec = state_->cpu.v[vreg];
          uint64_t elem = res.value;
          if (esize < 8) elem &= (1ULL << (esize * 8)) - 1;
          memcpy(reinterpret_cast<uint8_t*>(&vec) + args.index * esize, &elem, esize);
          state_->cpu.v[vreg] = vec;
        }
      }

      // Post-index.
      if (args.postindex) {
        int64_t offset;
        if (args.rm == 31) {
          offset = args.num_regs * esize;
        } else {
          offset = static_cast<int64_t>(state_->cpu.x[args.rm]);
        }
        if (args.rn == 31) {
          SetSp(base_addr + offset);
        } else {
          state_->cpu.x[args.rn] = base_addr + offset;
        }
      }
    }
  }
  // endregion

  Register Extr(Register src_n, Register src_m, uint8_t lsb, bool is_64bit) {
    CHECK(!exception_raised_);
    unsigned reg_size = is_64bit ? 64 : 32;
    uint64_t n = is_64bit ? src_n : (src_n & 0xFFFFFFFFULL);
    uint64_t m = is_64bit ? src_m : (src_m & 0xFFFFFFFFULL);
    uint64_t result;
    if (lsb == 0) {
      result = m;
    } else {
      result = (m >> lsb) | (n << (reg_size - lsb));
    }
    if (!is_64bit) result &= 0xFFFFFFFFULL;
    return result;
  }
  // endregion

  void ConditionalCompare(bool is_neg, bool is_64bit, Register rn, Register rm,
                          Decoder::Condition cond, uint8_t nzcv_imm) {
    CHECK(!exception_raised_);
    if (EvaluateCondition(cond)) {
      // Condition is true: perform the comparison and set flags.
      uint64_t operand1 = is_64bit ? rn : (rn & 0xFFFFFFFFULL);
      uint64_t operand2 = is_64bit ? rm : (rm & 0xFFFFFFFFULL);
      uint64_t result;
      if (is_neg) {
        // CCMN: add
        result = operand1 + operand2;
        if (!is_64bit) result &= 0xFFFFFFFFULL;
        UpdateFlags(operand1, operand2, result, false, is_64bit);
      } else {
        // CCMP: subtract
        result = operand1 - operand2;
        if (!is_64bit) result &= 0xFFFFFFFFULL;
        UpdateFlags(operand1, operand2, result, true, is_64bit);
      }
    } else {
      // Condition is false: set flags to the immediate value.
      uint16_t flags = 0;
      if (nzcv_imm & 0b1000) flags |= CPUState::kFlagNegative;
      if (nzcv_imm & 0b0100) flags |= CPUState::kFlagZero;
      if (nzcv_imm & 0b0010) flags |= CPUState::kFlagCarry;
      if (nzcv_imm & 0b0001) flags |= CPUState::kFlagOverflow;
      state_->cpu.flags = flags;
    }
  }

  void Nop() {}

  void Undefined() {
    UndefinedInsn(GetInsnAddr());
    exception_raised_ = true;
  }

  // region digitalis
  //
  // SIMD/FP instruction implementations.
  //

  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs& args) {
    CHECK(!exception_raised_);
    __uint128_t value = ExpandSimdModifiedImm(args.op, args.cmode, args.abc, args.defgh, args.q);
    state_->cpu.v[args.rd] = value;
  }

  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& args, Register base) {
    CHECK(!exception_raised_);
    uint64_t addr = base + args.offset;
    void* host_addr = ToHostAddr<void>(addr);

    if (args.is_store) {
      SimdStoreToMemory(host_addr, args.rt, args.size);
    } else {
      SimdLoadFromMemory(host_addr, args.rt, args.size);
    }
  }

  void SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs& args, Register base) {
    CHECK(!exception_raised_);
    uint8_t element_size;
    switch (args.size) {
      case Decoder::SimdLoadStoreSize::k32bit: element_size = 4; break;
      case Decoder::SimdLoadStoreSize::k64bit: element_size = 8; break;
      case Decoder::SimdLoadStoreSize::k128bit: element_size = 16; break;
      default: Undefined(); return;
    }

    void* addr1 = ToHostAddr<void>(base);
    void* addr2 = ToHostAddr<void>(base + element_size);

    if (args.is_store) {
      SimdStoreToMemory(addr1, args.rt1, args.size);
      SimdStoreToMemory(addr2, args.rt2, args.size);
    } else {
      SimdLoadFromMemory(addr1, args.rt1, args.size);
      SimdLoadFromMemory(addr2, args.rt2, args.size);
    }
  }

  void SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs& args,
                         Register base, Register offset_reg) {
    CHECK(!exception_raised_);
    // region digitalis - Apply the offset register extension before
    // shift+add (see ApplyOffsetExtend comment above).
    uint64_t off = ApplyOffsetExtend(offset_reg, args.extend_type) << args.shift_amount;
    uint64_t addr = base + off;
    // endregion
    void* host_addr = ToHostAddr<void>(addr);

    if (args.is_store) {
      SimdStoreToMemory(host_addr, args.rt, args.size);
    } else {
      SimdLoadFromMemory(host_addr, args.rt, args.size);
    }
  }

  // region digitalis
  // FCSEL: Floating-point conditional select
  // If condition is true, Rd = Rn; else Rd = Rm.
  void FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ftype, Decoder::Condition cond) {
    CHECK(!exception_raised_);
    bool condition_holds = EvaluateCondition(cond);
    uint8_t src = condition_holds ? rn : rm;
    state_->cpu.v[rd] = 0;
    if (ftype == 0b00) {
      // Single-precision: copy 32 bits
      uint32_t val;
      memcpy(&val, &state_->cpu.v[src], 4);
      memcpy(&state_->cpu.v[rd], &val, 4);
    } else if (ftype == 0b01) {
      // Double-precision: copy 64 bits
      uint64_t val;
      memcpy(&val, &state_->cpu.v[src], 8);
      memcpy(&state_->cpu.v[rd], &val, 8);
    } else {
      Undefined();
    }
  }

  // FP <-> fixed-point conversion: SCVTF, UCVTF, FCVTZS, FCVTZU (scalar, fixed-point)
  void FpFixedPointConversion(const Decoder::FpFixedPointArgs& args) {
    CHECK(!exception_raised_);
    using Op = Decoder::FpFixedPointOp;
    double scale = static_cast<double>(1ULL << args.fbits);

    switch (args.op) {
      case Op::kScvtf: {
        // Signed integer -> FP, divided by 2^fbits
        int64_t int_val;
        if (args.sf) {
          int_val = static_cast<int64_t>(state_->cpu.x[args.rn]);
        } else {
          int_val = static_cast<int64_t>(static_cast<int32_t>(
              static_cast<uint32_t>(state_->cpu.x[args.rn])));
        }
        state_->cpu.v[args.rd] = 0;
        if (args.ftype == 0b00) {
          float result = static_cast<float>(static_cast<double>(int_val) / scale);
          memcpy(&state_->cpu.v[args.rd], &result, 4);
        } else {
          double result = static_cast<double>(int_val) / scale;
          memcpy(&state_->cpu.v[args.rd], &result, 8);
        }
        break;
      }
      case Op::kUcvtf: {
        // Unsigned integer -> FP, divided by 2^fbits
        uint64_t uint_val;
        if (args.sf) {
          uint_val = state_->cpu.x[args.rn];
        } else {
          uint_val = static_cast<uint32_t>(state_->cpu.x[args.rn]);
        }
        state_->cpu.v[args.rd] = 0;
        if (args.ftype == 0b00) {
          float result = static_cast<float>(static_cast<double>(uint_val) / scale);
          memcpy(&state_->cpu.v[args.rd], &result, 4);
        } else {
          double result = static_cast<double>(uint_val) / scale;
          memcpy(&state_->cpu.v[args.rd], &result, 8);
        }
        break;
      }
      case Op::kFcvtzs: {
        // FP -> signed fixed-point, multiplied by 2^fbits, round toward zero
        double fp_val;
        if (args.ftype == 0b00) {
          float f;
          memcpy(&f, &state_->cpu.v[args.rn], 4);
          fp_val = static_cast<double>(f);
        } else {
          memcpy(&fp_val, &state_->cpu.v[args.rn], 8);
        }
        double scaled = fp_val * scale;
        int64_t result = static_cast<int64_t>(trunc(scaled));
        if (args.sf) {
          state_->cpu.x[args.rd] = static_cast<uint64_t>(result);
        } else {
          state_->cpu.x[args.rd] = static_cast<uint64_t>(static_cast<uint32_t>(
              static_cast<int32_t>(result)));
        }
        break;
      }
      case Op::kFcvtzu: {
        // FP -> unsigned fixed-point, multiplied by 2^fbits, round toward zero
        double fp_val;
        if (args.ftype == 0b00) {
          float f;
          memcpy(&f, &state_->cpu.v[args.rn], 4);
          fp_val = static_cast<double>(f);
        } else {
          memcpy(&fp_val, &state_->cpu.v[args.rn], 8);
        }
        double scaled = fp_val * scale;
        uint64_t result = static_cast<uint64_t>(trunc(scaled));
        if (args.sf) {
          state_->cpu.x[args.rd] = result;
        } else {
          state_->cpu.x[args.rd] = static_cast<uint32_t>(result);
        }
        break;
      }
    }
  }

  // FP data-processing (3 source): FMADD, FMSUB, FNMADD, FNMSUB
  void FpDataProc3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra,
                   uint8_t ftype, bool o1, bool o0) {
    CHECK(!exception_raised_);
    if (ftype == 0b00) {
      // Single-precision
      float fn, fm, fa;
      memcpy(&fn, &state_->cpu.v[rn], 4);
      memcpy(&fm, &state_->cpu.v[rm], 4);
      memcpy(&fa, &state_->cpu.v[ra], 4);
      float result;
      if (!o1 && !o0) {
        // FMADD: Rd = Ra + (Rn * Rm)
        result = fmaf(fn, fm, fa);
      } else if (!o1 && o0) {
        // FMSUB: Rd = Ra - (Rn * Rm) = -(Rn*Rm) + Ra = fma(-Rn, Rm, Ra)
        result = fmaf(-fn, fm, fa);
      } else if (o1 && !o0) {
        // FNMADD: Rd = -(Ra + Rn * Rm) = fma(-Rn, Rm, -Ra) = -fma(Rn, Rm, Ra)
        result = -fmaf(fn, fm, fa);
      } else {
        // FNMSUB: Rd = Rn * Rm - Ra = fma(Rn, Rm, -Ra)
        result = fmaf(fn, fm, -fa);
      }
      state_->cpu.v[rd] = 0;
      memcpy(&state_->cpu.v[rd], &result, 4);
    } else if (ftype == 0b01) {
      // Double-precision
      double fn, fm, fa;
      memcpy(&fn, &state_->cpu.v[rn], 8);
      memcpy(&fm, &state_->cpu.v[rm], 8);
      memcpy(&fa, &state_->cpu.v[ra], 8);
      double result;
      if (!o1 && !o0) {
        result = fma(fn, fm, fa);
      } else if (!o1 && o0) {
        result = fma(-fn, fm, fa);
      } else if (o1 && !o0) {
        result = -fma(fn, fm, fa);
      } else {
        result = fma(fn, fm, -fa);
      }
      state_->cpu.v[rd] = 0;
      memcpy(&state_->cpu.v[rd], &result, 8);
    } else {
      Undefined();
    }
  }

  // FMOV (scalar, immediate): load a floating-point constant into SIMD register.
  // The imm8 is expanded via VFPExpandImm to the target precision.
  void FpMovImmediate(uint8_t rd, uint8_t imm8, uint8_t ftype) {
    CHECK(!exception_raised_);
    state_->cpu.v[rd] = 0;  // zero entire 128-bit register
    if (ftype == 0b00) {
      // Single-precision: VFPExpandImm to 32-bit float
      // sign = imm8[7], exp = NOT(imm8[6]):Repeat(imm8[6],5):imm8[5:4], frac = imm8[3:0]:Zeros(19)
      uint32_t sign = (imm8 >> 7) & 1;
      uint32_t exp6 = (imm8 >> 6) & 1;
      uint32_t exp_top = exp6 ? 0b0 : 0b1;  // NOT(imm8[6])
      uint32_t exp_rep = exp6 ? 0b11111 : 0b00000;  // Repeat(imm8[6], 5)
      uint32_t exp_low = (imm8 >> 4) & 0b11;  // imm8[5:4]
      uint32_t exp = (exp_top << 7) | (exp_rep << 2) | exp_low;
      uint32_t frac = (imm8 & 0xF) << 19;
      uint32_t result = (sign << 31) | (exp << 23) | frac;
      memcpy(&state_->cpu.v[rd], &result, 4);
    } else if (ftype == 0b01) {
      // Double-precision: VFPExpandImm to 64-bit double
      // sign = imm8[7], exp = NOT(imm8[6]):Repeat(imm8[6],8):imm8[5:4], frac = imm8[3:0]:Zeros(48)
      uint64_t sign = (imm8 >> 7) & 1;
      uint64_t exp6 = (imm8 >> 6) & 1;
      uint64_t exp_top = exp6 ? 0 : 1;  // NOT(imm8[6])
      uint64_t exp_rep = exp6 ? 0xFF : 0x00;  // Repeat(imm8[6], 8)
      uint64_t exp_low = (imm8 >> 4) & 0b11;  // imm8[5:4]
      uint64_t exp = (exp_top << 10) | (exp_rep << 2) | exp_low;
      uint64_t frac = static_cast<uint64_t>(imm8 & 0xF) << 48;
      uint64_t result = (sign << 63) | (exp << 52) | frac;
      memcpy(&state_->cpu.v[rd], &result, 8);
    } else {
      // Half-precision (ftype=11) or reserved (ftype=10)
      Undefined();
    }
  }
  // endregion

  void FpIntConversion(const Decoder::FpIntConvArgs& args) {
    CHECK(!exception_raised_);
    uint8_t rmode = args.rmode;
    uint8_t opcode = args.op;

    // endregion (digitalis FMOV trace removed)

    // FMOV between GP and FP registers (rmode=00, opcode=110 or 111)
    // ARM64 encoding: opcode=111 → GP to FP (FMOV Dd, Xn)
    //                 opcode=110 → FP to GP (FMOV Xd, Dn)
    if (rmode == 0b00 && opcode == 0b111) {
      // FMOV Sd, Wn (or FMOV Dd, Xn depending on ftype/sf)
      uint64_t gp_val = (args.rn < 31) ? state_->cpu.x[args.rn] : 0;
      state_->cpu.v[args.rd] = 0;  // zero upper bits
      if (args.ftype == 0b00) {
        // FMOV Sd, Wn (32-bit)
        memcpy(&state_->cpu.v[args.rd], &gp_val, 4);
      } else if (args.ftype == 0b01) {
        // FMOV Dd, Xn (64-bit)
        memcpy(&state_->cpu.v[args.rd], &gp_val, 8);
      } else {
        Undefined();
      }
      return;
    }

    if (rmode == 0b00 && opcode == 0b110) {
      // FMOV Wn, Sd (or FMOV Xn, Dd)
      uint64_t result = 0;
      if (args.ftype == 0b00) {
        // FMOV Wn, Sd (32-bit)
        memcpy(&result, &state_->cpu.v[args.rn], 4);
        result &= 0xFFFFFFFFULL;
      } else if (args.ftype == 0b01) {
        // FMOV Xn, Dd (64-bit)
        memcpy(&result, &state_->cpu.v[args.rn], 8);
      } else {
        Undefined();
        return;
      }
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = result;
      }
      return;
    }

    // FMOV to/from top half of Q register: rmode=01, opcode=110 or 111
    // ARM64 encoding: opcode=111 → GP to FP (FMOV Vd.D[1], Xn)
    //                 opcode=110 → FP to GP (FMOV Xd, Vn.D[1])
    if (rmode == 0b01 && opcode == 0b110) {
      // FMOV Xd, Vn.D[1] — read top 64 bits
      uint64_t result = 0;
      __uint128_t v = state_->cpu.v[args.rn];
      result = static_cast<uint64_t>(v >> 64);
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = result;
      }
      return;
    }

    if (rmode == 0b01 && opcode == 0b111) {
      // FMOV Vd.D[1], Xn — write top 64 bits
      uint64_t gp_val = (args.rn < 31) ? state_->cpu.x[args.rn] : 0;
      __uint128_t v = state_->cpu.v[args.rd];
      // Keep lower 64 bits, replace upper 64 bits
      v = (v & static_cast<__uint128_t>(0xFFFFFFFFFFFFFFFFULL)) |
          (static_cast<__uint128_t>(gp_val) << 64);
      state_->cpu.v[args.rd] = v;
      return;
    }

    // SCVTF, UCVTF, FCVTZS, FCVTZU: integer <-> FP conversion
    if (rmode == 0b00 && opcode == 0b010) {
      // SCVTF: signed integer to FP
      int64_t ival = args.sf ? static_cast<int64_t>(args.rn < 31 ? state_->cpu.x[args.rn] : 0)
                             : static_cast<int64_t>(static_cast<int32_t>(
                                   static_cast<uint32_t>(args.rn < 31 ? state_->cpu.x[args.rn] : 0)));
      state_->cpu.v[args.rd] = 0;
      if (args.ftype == 0b00) {
        float f = static_cast<float>(ival);
        memcpy(&state_->cpu.v[args.rd], &f, 4);
      } else if (args.ftype == 0b01) {
        double d = static_cast<double>(ival);
        memcpy(&state_->cpu.v[args.rd], &d, 8);
      } else { Undefined(); }
      return;
    }

    if (rmode == 0b00 && opcode == 0b011) {
      // UCVTF: unsigned integer to FP
      uint64_t uval = args.sf ? (args.rn < 31 ? state_->cpu.x[args.rn] : 0)
                              : static_cast<uint64_t>(static_cast<uint32_t>(
                                    args.rn < 31 ? state_->cpu.x[args.rn] : 0));
      state_->cpu.v[args.rd] = 0;
      if (args.ftype == 0b00) {
        float f = static_cast<float>(uval);
        memcpy(&state_->cpu.v[args.rd], &f, 4);
      } else if (args.ftype == 0b01) {
        double d = static_cast<double>(uval);
        memcpy(&state_->cpu.v[args.rd], &d, 8);
      } else { Undefined(); }
      return;
    }

    // region digitalis
    // FCVTNS/FCVTNU/FCVTPS/FCVTPU/FCVTMS/FCVTMU: various rounding modes
    // rmode=00: round to nearest (ties to even)
    // rmode=01: round toward +inf
    // rmode=10: round toward -inf
    // rmode=11: round toward zero
    // opcode=000: signed, opcode=001: unsigned
    if ((rmode == 0b00 || rmode == 0b01 || rmode == 0b10) &&
        (opcode == 0b000 || opcode == 0b001)) {
      bool is_signed = (opcode == 0b000);
      uint64_t result = 0;
      double dval = 0;
      if (args.ftype == 0b00) {
        float f; memcpy(&f, &state_->cpu.v[args.rn], 4);
        dval = f;
      } else if (args.ftype == 0b01) {
        memcpy(&dval, &state_->cpu.v[args.rn], 8);
      } else { Undefined(); return; }
      // Apply rounding
      double rounded;
      if (rmode == 0b00) rounded = rint(dval);        // nearest, ties to even
      else if (rmode == 0b01) rounded = ceil(dval);    // toward +inf
      else rounded = floor(dval);                       // toward -inf
      if (is_signed) {
        if (args.sf) result = static_cast<uint64_t>(static_cast<int64_t>(rounded));
        else result = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(rounded)));
      } else {
        if (args.sf) result = static_cast<uint64_t>(rounded);
        else result = static_cast<uint64_t>(static_cast<uint32_t>(rounded));
      }
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = args.sf ? result : (result & 0xFFFFFFFFULL);
      }
      return;
    }

    // FCVTAS/FCVTAU: rmode=00, opcode=100 (signed) or 101 (unsigned)
    // Round to nearest, ties away from zero
    if (rmode == 0b00 && (opcode == 0b100 || opcode == 0b101)) {
      bool is_signed = (opcode == 0b100);
      uint64_t result = 0;
      double dval = 0;
      if (args.ftype == 0b00) {
        float f; memcpy(&f, &state_->cpu.v[args.rn], 4);
        dval = f;
      } else if (args.ftype == 0b01) {
        memcpy(&dval, &state_->cpu.v[args.rn], 8);
      } else { Undefined(); return; }
      // Round to nearest, ties away from zero
      double rounded = round(dval);
      if (is_signed) {
        if (args.sf) result = static_cast<uint64_t>(static_cast<int64_t>(rounded));
        else result = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(rounded)));
      } else {
        if (args.sf) result = static_cast<uint64_t>(rounded);
        else result = static_cast<uint64_t>(static_cast<uint32_t>(rounded));
      }
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = args.sf ? result : (result & 0xFFFFFFFFULL);
      }
      return;
    }
    // endregion

    if (rmode == 0b11 && opcode == 0b000) {
      // FCVTZS: FP to signed integer, round toward zero
      uint64_t result = 0;
      if (args.ftype == 0b00) {
        float f;
        memcpy(&f, &state_->cpu.v[args.rn], 4);
        if (args.sf) { result = static_cast<uint64_t>(static_cast<int64_t>(f)); }
        else { result = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(f))); }
      } else if (args.ftype == 0b01) {
        double d;
        memcpy(&d, &state_->cpu.v[args.rn], 8);
        if (args.sf) { result = static_cast<uint64_t>(static_cast<int64_t>(d)); }
        else { result = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(d))); }
      } else { Undefined(); return; }
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = args.sf ? result : (result & 0xFFFFFFFFULL);
      }
      return;
    }

    if (rmode == 0b11 && opcode == 0b001) {
      // FCVTZU: FP to unsigned integer, round toward zero
      uint64_t result = 0;
      if (args.ftype == 0b00) {
        float f;
        memcpy(&f, &state_->cpu.v[args.rn], 4);
        if (args.sf) { result = static_cast<uint64_t>(f); }
        else { result = static_cast<uint64_t>(static_cast<uint32_t>(f)); }
      } else if (args.ftype == 0b01) {
        double d;
        memcpy(&d, &state_->cpu.v[args.rn], 8);
        if (args.sf) { result = static_cast<uint64_t>(d); }
        else { result = static_cast<uint64_t>(static_cast<uint32_t>(d)); }
      } else { Undefined(); return; }
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = args.sf ? result : (result & 0xFFFFFFFFULL);
      }
      return;
    }

    Undefined();
  }

  void LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& args, Register base) {
    CHECK(!exception_raised_);
    void* host_addr = ToHostAddr<void>(base);
    bool need_write = (args.op != Decoder::AtomicOp::kLdxr &&
                       args.op != Decoder::AtomicOp::kLdar);
    // Atomic ops use __atomic builtins which handle their own faults via
    // the registered FaultyLoad/FaultyStore recovery code addresses.

    switch (args.op) {
      case Decoder::AtomicOp::kLdxr:
      case Decoder::AtomicOp::kLdar: {
        uint64_t val = 0;
        switch (args.size) {
          case 0: val = AtomicLoad<uint8_t>(host_addr); break;
          case 1: val = AtomicLoad<uint16_t>(host_addr); break;
          case 2: val = AtomicLoad<uint32_t>(host_addr); break;
          case 3: val = AtomicLoad<uint64_t>(host_addr); break;
        }
        if (args.op == Decoder::AtomicOp::kLdxr) {
          state_->cpu.reservation_address = base;
          memcpy(&state_->cpu.reservation_value, &val, sizeof(val));
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = val;
        break;
      }

      case Decoder::AtomicOp::kStxr: {
        uint64_t new_val = (args.rt < 31) ? state_->cpu.x[args.rt] : 0;
        uint64_t expected;
        memcpy(&expected, &state_->cpu.reservation_value, sizeof(expected));
        bool success = false;
        if (state_->cpu.reservation_address == base) {
          switch (args.size) {
            case 0: success = AtomicCAS<uint8_t>(host_addr, expected, new_val); break;
            case 1: success = AtomicCAS<uint16_t>(host_addr, expected, new_val); break;
            case 2: success = AtomicCAS<uint32_t>(host_addr, expected, new_val); break;
            case 3: success = AtomicCAS<uint64_t>(host_addr, expected, new_val); break;
          }
        }
        state_->cpu.reservation_address = 0;
        // Rs gets 0 on success, 1 on failure.
        if (args.rs < 31) state_->cpu.x[args.rs] = success ? 0 : 1;
        break;
      }

      case Decoder::AtomicOp::kStlr: {
        uint64_t val = (args.rt < 31) ? state_->cpu.x[args.rt] : 0;
        switch (args.size) {
          case 0: AtomicStore<uint8_t>(host_addr, val); break;
          case 1: AtomicStore<uint16_t>(host_addr, val); break;
          case 2: AtomicStore<uint32_t>(host_addr, val); break;
          case 3: AtomicStore<uint64_t>(host_addr, val); break;
        }
        break;
      }

      case Decoder::AtomicOp::kCas: {
        uint64_t expected = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t desired = (args.rt < 31) ? state_->cpu.x[args.rt] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicCASVal<uint8_t>(host_addr, expected, desired); break;
          case 1: old_val = AtomicCASVal<uint16_t>(host_addr, expected, desired); break;
          case 2: old_val = AtomicCASVal<uint32_t>(host_addr, expected, desired); break;
          case 3: old_val = AtomicCASVal<uint64_t>(host_addr, expected, desired); break;
        }
        // CAS writes original value back to Rs.
        if (args.rs < 31) state_->cpu.x[args.rs] = old_val;
        break;
      }

      case Decoder::AtomicOp::kSwp: {
        uint64_t new_val = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicExchange<uint8_t>(host_addr, new_val); break;
          case 1: old_val = AtomicExchange<uint16_t>(host_addr, new_val); break;
          case 2: old_val = AtomicExchange<uint32_t>(host_addr, new_val); break;
          case 3: old_val = AtomicExchange<uint64_t>(host_addr, new_val); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }

      case Decoder::AtomicOp::kLdadd: {
        uint64_t addend = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicFetchAdd<uint8_t>(host_addr, addend); break;
          case 1: old_val = AtomicFetchAdd<uint16_t>(host_addr, addend); break;
          case 2: old_val = AtomicFetchAdd<uint32_t>(host_addr, addend); break;
          case 3: old_val = AtomicFetchAdd<uint64_t>(host_addr, addend); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }

      case Decoder::AtomicOp::kLdclr: {
        uint64_t mask = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicFetchAndNot<uint8_t>(host_addr, mask); break;
          case 1: old_val = AtomicFetchAndNot<uint16_t>(host_addr, mask); break;
          case 2: old_val = AtomicFetchAndNot<uint32_t>(host_addr, mask); break;
          case 3: old_val = AtomicFetchAndNot<uint64_t>(host_addr, mask); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }

      case Decoder::AtomicOp::kLdset: {
        uint64_t bits = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicFetchOr<uint8_t>(host_addr, bits); break;
          case 1: old_val = AtomicFetchOr<uint16_t>(host_addr, bits); break;
          case 2: old_val = AtomicFetchOr<uint32_t>(host_addr, bits); break;
          case 3: old_val = AtomicFetchOr<uint64_t>(host_addr, bits); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }

      case Decoder::AtomicOp::kLdeor: {
        uint64_t bits = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicFetchXor<uint8_t>(host_addr, bits); break;
          case 1: old_val = AtomicFetchXor<uint16_t>(host_addr, bits); break;
          case 2: old_val = AtomicFetchXor<uint32_t>(host_addr, bits); break;
          case 3: old_val = AtomicFetchXor<uint64_t>(host_addr, bits); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }
    }
  }

  // region digitalis
  //
  // AdvSIMD copy: DUP (element), DUP (general), INS (general), SMOV, UMOV.
  //
  void AdvSimdCopy(const Decoder::AdvSimdCopyArgs& args) {
    CHECK(!exception_raised_);
    uint8_t imm5 = args.imm5;

    // Determine element size and index from imm5.
    // imm5[0]=1: byte (8-bit), index = imm5[4:1]
    // imm5[1:0]=10: halfword (16-bit), index = imm5[4:2]
    // imm5[2:0]=100: word (32-bit), index = imm5[4:3]
    // imm5[3:0]=1000: doubleword (64-bit), index = imm5[4]
    uint8_t esize;    // element size in bytes
    uint8_t index;    // element index

    if (imm5 & 0b00001) {
      esize = 1;
      index = (imm5 >> 1) & 0xF;
    } else if (imm5 & 0b00010) {
      esize = 2;
      index = (imm5 >> 2) & 0x7;
    } else if (imm5 & 0b00100) {
      esize = 4;
      index = (imm5 >> 3) & 0x3;
    } else if (imm5 & 0b01000) {
      esize = 8;
      index = (imm5 >> 4) & 0x1;
    } else {
      Undefined();
      return;
    }

    switch (args.opcode) {
      case Decoder::AdvSimdCopyOpcode::kDupElement: {
        // DUP (element): duplicate Vn[index] to all elements of Vd.
        // Q=0: 64-bit result (lower half), Q=1: 128-bit result.
        __uint128_t src = state_->cpu.v[args.rn];
        uint64_t element = 0;
        memcpy(&element, reinterpret_cast<const uint8_t*>(&src) + index * esize, esize);

        __uint128_t result = 0;
        uint8_t num_elements = (args.q ? 16 : 8) / esize;
        for (uint8_t i = 0; i < num_elements; i++) {
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &element, esize);
        }
        state_->cpu.v[args.rd] = result;
        break;
      }

      case Decoder::AdvSimdCopyOpcode::kDupGeneral: {
        // DUP (general): duplicate GP register Rn to all elements of Vd.
        // Element size determined by imm5 (same encoding as above).
        // Q=0: 64-bit result, Q=1: 128-bit result.
        uint64_t gp_val = (args.rn < 31) ? state_->cpu.x[args.rn] : 0;

        __uint128_t result = 0;
        uint8_t num_elements = (args.q ? 16 : 8) / esize;
        for (uint8_t i = 0; i < num_elements; i++) {
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &gp_val, esize);
        }
        state_->cpu.v[args.rd] = result;
        break;
      }

      case Decoder::AdvSimdCopyOpcode::kInsGeneral: {
        // INS (general): insert GP register Rn into Vd[index].
        // The rest of Vd is unchanged.
        uint64_t gp_val = (args.rn < 31) ? state_->cpu.x[args.rn] : 0;
        __uint128_t v = state_->cpu.v[args.rd];
        memcpy(reinterpret_cast<uint8_t*>(&v) + index * esize, &gp_val, esize);
        state_->cpu.v[args.rd] = v;
        break;
      }

      case Decoder::AdvSimdCopyOpcode::kSmov: {
        // SMOV: signed move from Vn[index] to GP register Rd.
        // Q=0: destination is Wd (32-bit), Q=1: destination is Xd (64-bit).
        // Element size must be smaller than destination.
        __uint128_t src = state_->cpu.v[args.rn];
        uint64_t element = 0;
        memcpy(&element, reinterpret_cast<const uint8_t*>(&src) + index * esize, esize);

        // Sign-extend the element to 64 bits.
        int64_t signed_val;
        switch (esize) {
          case 1:
            signed_val = static_cast<int64_t>(static_cast<int8_t>(element));
            break;
          case 2:
            signed_val = static_cast<int64_t>(static_cast<int16_t>(element));
            break;
          case 4:
            signed_val = static_cast<int64_t>(static_cast<int32_t>(element));
            break;
          default:
            Undefined();
            return;
        }

        uint64_t result = static_cast<uint64_t>(signed_val);
        if (!args.q) {
          // SMOV to Wd: truncate to 32 bits.
          result &= 0xFFFFFFFFULL;
        }
        if (args.rd < 31) {
          state_->cpu.x[args.rd] = result;
        }
        break;
      }

      case Decoder::AdvSimdCopyOpcode::kUmov: {
        // UMOV: unsigned move from Vn[index] to GP register Rd.
        // Q=0: destination is Wd (32-bit), Q=1: destination is Xd (64-bit).
        __uint128_t src = state_->cpu.v[args.rn];
        uint64_t element = 0;
        memcpy(&element, reinterpret_cast<const uint8_t*>(&src) + index * esize, esize);

        if (!args.q) {
          element &= 0xFFFFFFFFULL;
        }
        if (args.rd < 31) {
          state_->cpu.x[args.rd] = element;
        }
        break;
      }

      // region digitalis - scalar SIMD copy (DUP scalar / MOV Vd, Vn[index])
      case Decoder::AdvSimdCopyOpcode::kDupScalar: {
        // DUP (scalar): copy one esize-byte element from Vn[index] into the
        // bottom of Vd; upper bits are zeroed.
        __uint128_t src = state_->cpu.v[args.rn];
        __uint128_t result = 0;
        memcpy(reinterpret_cast<uint8_t*>(&result),
               reinterpret_cast<const uint8_t*>(&src) + index * esize,
               esize);
        state_->cpu.v[args.rd] = result;
        break;
      }
      // endregion

      case Decoder::AdvSimdCopyOpcode::kInsElement: {
        // INS (element): copy Vn[src_index] to Vd[dst_index].
        // dst_index is encoded in imm5, src_index in imm4.
        // Element size is determined by imm5 (same as other copy ops).
        __uint128_t src = state_->cpu.v[args.rn];
        uint64_t element = 0;
        // Decode source index from imm4 using same size encoding.
        uint8_t src_index;
        if (esize == 1) {
          src_index = (args.imm4 >> 0) & 0xF;
        } else if (esize == 2) {
          src_index = (args.imm4 >> 1) & 0x7;
        } else if (esize == 4) {
          src_index = (args.imm4 >> 2) & 0x3;
        } else {
          src_index = (args.imm4 >> 3) & 0x1;
        }
        memcpy(&element, reinterpret_cast<const uint8_t*>(&src) + src_index * esize, esize);
        __uint128_t v = state_->cpu.v[args.rd];
        memcpy(reinterpret_cast<uint8_t*>(&v) + index * esize, &element, esize);
        state_->cpu.v[args.rd] = v;
        break;
      }

      default:
        Undefined();
        break;
    }
  }
  // endregion

  // region digitalis
  //
  // AdvSIMD three same: element-wise vector arithmetic/logic operations.
  //
  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t dst = state_->cpu.v[args.rd];  // Needed for BSL/BIT/BIF
    __uint128_t result = 0;

    uint8_t esize;  // element size in bytes
    switch (args.size) {
      case 0b00: esize = 1; break;
      case 0b01: esize = 2; break;
      case 0b10: esize = 4; break;
      case 0b11: esize = 8; break;
      default: Undefined(); return;
    }

    uint8_t vec_len = args.q ? 16 : 8;  // total bytes in result vector
    uint8_t num_elements = vec_len / esize;

    switch (args.opcode) {
      // --- Logic operations (ignore size for element iteration, operate on whole vector) ---
      case Decoder::AdvSimdThreeSameOpcode::kAnd:
        result = src_n & src_m;
        if (!args.q) {
          // Zero upper 64 bits for 64-bit vector.
          uint64_t lo;
          memcpy(&lo, &result, 8);
          result = 0;
          memcpy(&result, &lo, 8);
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kBic:
        result = src_n & ~src_m;
        if (!args.q) {
          uint64_t lo;
          memcpy(&lo, &result, 8);
          result = 0;
          memcpy(&result, &lo, 8);
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kOrr:
        result = src_n | src_m;
        if (!args.q) {
          uint64_t lo;
          memcpy(&lo, &result, 8);
          result = 0;
          memcpy(&result, &lo, 8);
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kOrn:
        result = src_n | ~src_m;
        if (!args.q) {
          uint64_t lo;
          memcpy(&lo, &result, 8);
          result = 0;
          memcpy(&result, &lo, 8);
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kEor:
        result = src_n ^ src_m;
        if (!args.q) {
          uint64_t lo;
          memcpy(&lo, &result, 8);
          result = 0;
          memcpy(&result, &lo, 8);
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kBsl:
        // BSL: Vd = (Vd & Vn) | (~Vd & Vm) — bitwise select using Vd as mask.
        result = (dst & src_n) | (~dst & src_m);
        if (!args.q) {
          uint64_t lo;
          memcpy(&lo, &result, 8);
          result = 0;
          memcpy(&result, &lo, 8);
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kBit:
        // BIT: Vd = (Vm & Vn) | (~Vm & Vd) — insert bits where Vm is 1.
        result = (src_m & src_n) | (~src_m & dst);
        if (!args.q) {
          uint64_t lo;
          memcpy(&lo, &result, 8);
          result = 0;
          memcpy(&result, &lo, 8);
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kBif:
        // BIF: Vd = (Vm & Vd) | (~Vm & Vn) — insert bits where Vm is 0.
        result = (src_m & dst) | (~src_m & src_n);
        if (!args.q) {
          uint64_t lo;
          memcpy(&lo, &result, 8);
          result = 0;
          memcpy(&result, &lo, 8);
        }
        break;

      // --- Arithmetic: element-wise ADD/SUB ---
      case Decoder::AdvSimdThreeSameOpcode::kAdd:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t { return a + b; });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kSub:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t { return a - b; });
        break;

      // --- Compare: CMEQ, CMTST ---
      case Decoder::AdvSimdThreeSameOpcode::kCmeq:
        // CMEQ: if (Vn[i] == Vm[i]) result[i] = all-ones, else all-zeros.
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              uint64_t mask = (es >= 8) ? ~0ULL : ((1ULL << (es * 8)) - 1);
              return (a == b) ? mask : 0;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kCmtst:
        // CMTST: if (Vn[i] & Vm[i] != 0) result[i] = all-ones, else all-zeros.
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              uint64_t mask = (es >= 8) ? ~0ULL : ((1ULL << (es * 8)) - 1);
              return ((a & b) != 0) ? mask : 0;
            });
        break;

      // region digitalis
      // --- Compare: CMGT, CMHI, CMGE, CMHS ---
      case Decoder::AdvSimdThreeSameOpcode::kCmgt:
        // CMGT (signed >): if (Vn[i] > Vm[i]) signed, result = all-ones.
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              uint64_t mask = (es >= 8) ? ~0ULL : ((1ULL << (es * 8)) - 1);
              uint8_t bits = es * 8;
              int64_t sa = static_cast<int64_t>(a << (64 - bits)) >> (64 - bits);
              int64_t sb = static_cast<int64_t>(b << (64 - bits)) >> (64 - bits);
              return (sa > sb) ? mask : 0;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kCmhi:
        // CMHI (unsigned >): if (Vn[i] > Vm[i]) unsigned, result = all-ones.
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              uint64_t mask = (es >= 8) ? ~0ULL : ((1ULL << (es * 8)) - 1);
              return (a > b) ? mask : 0;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kCmge:
        // CMGE (signed >=): if (Vn[i] >= Vm[i]) signed, result = all-ones.
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              uint64_t mask = (es >= 8) ? ~0ULL : ((1ULL << (es * 8)) - 1);
              uint8_t bits = es * 8;
              int64_t sa = static_cast<int64_t>(a << (64 - bits)) >> (64 - bits);
              int64_t sb = static_cast<int64_t>(b << (64 - bits)) >> (64 - bits);
              return (sa >= sb) ? mask : 0;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kCmhs:
        // CMHS (unsigned >=): if (Vn[i] >= Vm[i]) unsigned, result = all-ones.
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              uint64_t mask = (es >= 8) ? ~0ULL : ((1ULL << (es * 8)) - 1);
              return (a >= b) ? mask : 0;
            });
        break;
      // endregion

      // --- Max/Min ---
      case Decoder::AdvSimdThreeSameOpcode::kSmax:
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [](int64_t a, int64_t b) -> int64_t { return a > b ? a : b; });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kSmin:
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [](int64_t a, int64_t b) -> int64_t { return a < b ? a : b; });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kUmax:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t {
              return a > b ? a : b;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kUmin:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t {
              return a < b ? a : b;
            });
        break;

      // --- Halving add/sub ---
      case Decoder::AdvSimdThreeSameOpcode::kShadd:
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [](int64_t a, int64_t b) -> int64_t { return (a + b) >> 1; });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kUhadd:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t {
              return (a + b) >> 1;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kSrhadd:
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [](int64_t a, int64_t b) -> int64_t { return (a + b + 1) >> 1; });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kUrhadd:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t {
              return (a + b + 1) >> 1;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kShsub:
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [](int64_t a, int64_t b) -> int64_t { return (a - b) >> 1; });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kUhsub:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t {
              return (a - b) >> 1;
            });
        break;

      // --- Saturating add/sub ---
      // SQADD / SQSUB clamp to the per-element signed range, not the lambda's
      // int64 range. Pre-fix these returned INT64_MIN/MAX which the caller's
      // mask truncated to garbage; now we capture `esize` so the lambda can
      // pick the right bounds.
      case Decoder::AdvSimdThreeSameOpcode::kSqadd: {
        uint8_t bits_local = esize * 8;
        int64_t smax = (bits_local == 64) ? INT64_MAX
                                          : ((1LL << (bits_local - 1)) - 1);
        int64_t smin = (bits_local == 64) ? INT64_MIN
                                          : -(1LL << (bits_local - 1));
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [smax, smin](int64_t a, int64_t b) -> int64_t {
              int64_t sum = a + b;
              if (b > 0 && sum < a) return smax;
              if (b < 0 && sum > a) return smin;
              if (sum > smax) return smax;
              if (sum < smin) return smin;
              return sum;
            });
        break;
      }
      case Decoder::AdvSimdThreeSameOpcode::kUqadd:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              uint64_t sum = a + b;
              uint64_t mask = (es >= 8) ? ~0ULL : ((1ULL << (es * 8)) - 1);
              return (sum > mask) ? mask : sum;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kSqsub: {
        uint8_t bits_local = esize * 8;
        int64_t smax = (bits_local == 64) ? INT64_MAX
                                          : ((1LL << (bits_local - 1)) - 1);
        int64_t smin = (bits_local == 64) ? INT64_MIN
                                          : -(1LL << (bits_local - 1));
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [smax, smin](int64_t a, int64_t b) -> int64_t {
              int64_t diff = a - b;
              if (b > 0 && diff > a) return smin;
              if (b < 0 && diff < a) return smax;
              if (diff > smax) return smax;
              if (diff < smin) return smin;
              return diff;
            });
        break;
      }
      case Decoder::AdvSimdThreeSameOpcode::kUqsub:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t {
              return (a > b) ? (a - b) : 0;
            });
        break;

      // --- Shift ---
      case Decoder::AdvSimdThreeSameOpcode::kSshl:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              // Promote to int32_t so -shift never overflows (int8_t INT8_MIN case).
              int32_t shift = static_cast<int8_t>(b & 0xFF);
              uint32_t bits = es * 8;
              if (shift >= 0) {
                return (static_cast<uint32_t>(shift) >= bits) ? 0 : (a << shift);
              } else {
                // Signed shift right: sign-extend a, then shift.
                int64_t sa = static_cast<int64_t>(a << (64 - bits)) >> (64 - bits);
                uint32_t rshift = static_cast<uint32_t>(-shift);
                return static_cast<uint64_t>(
                    (rshift >= bits) ? (sa >> (bits - 1)) : (sa >> rshift));
              }
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kUshl:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              int32_t shift = static_cast<int8_t>(b & 0xFF);
              uint32_t bits = es * 8;
              if (shift >= 0) {
                return (static_cast<uint32_t>(shift) >= bits) ? 0 : (a << shift);
              } else {
                uint32_t rshift = static_cast<uint32_t>(-shift);
                return (rshift >= bits) ? 0 : (a >> rshift);
              }
            });
        break;

      // --- Saturating shift (simplified — treat as regular shift for now) ---
      case Decoder::AdvSimdThreeSameOpcode::kSqshl:
      case Decoder::AdvSimdThreeSameOpcode::kUqshl:
      case Decoder::AdvSimdThreeSameOpcode::kSrshl:
      case Decoder::AdvSimdThreeSameOpcode::kUrshl:
      case Decoder::AdvSimdThreeSameOpcode::kSqrshl:
      case Decoder::AdvSimdThreeSameOpcode::kUqrshl:
        // Fallback: treat as SSHL/USHL for basic functionality.
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              int32_t shift = static_cast<int8_t>(b & 0xFF);
              uint32_t bits = es * 8;
              if (shift >= 0) {
                return (static_cast<uint32_t>(shift) >= bits) ? 0 : (a << shift);
              } else {
                uint32_t rshift = static_cast<uint32_t>(-shift);
                return (rshift >= bits) ? 0 : (a >> rshift);
              }
            });
        break;

      // --- ADDP (pairwise add) ---
      case Decoder::AdvSimdThreeSameOpcode::kAddp: {
        // ADDP concatenates Vn:Vm, then adds adjacent pairs.
        // First half of result from Vn pairs, second half from Vm pairs.
        uint64_t emask = ElementMask(esize);
        uint8_t half_elements = num_elements;  // total output elements
        result = 0;
        for (uint8_t i = 0; i < half_elements / 2; i++) {
          uint64_t a = 0, b = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + (2 * i) * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_n) + (2 * i + 1) * esize, esize);
          uint64_t sum = (a + b) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &sum, esize);
        }
        for (uint8_t i = 0; i < half_elements / 2; i++) {
          uint64_t a = 0, b = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_m) + (2 * i) * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + (2 * i + 1) * esize, esize);
          uint64_t sum = (a + b) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + (half_elements / 2 + i) * esize, &sum, esize);
        }
        break;
      }

      // region digitalis
      // --- Pairwise max/min (SMAXP, UMAXP, SMINP, UMINP) ---
      case Decoder::AdvSimdThreeSameOpcode::kSmaxp:
      case Decoder::AdvSimdThreeSameOpcode::kUmaxp:
      case Decoder::AdvSimdThreeSameOpcode::kSminp:
      case Decoder::AdvSimdThreeSameOpcode::kUminp: {
        bool is_unsigned = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmaxp ||
                            args.opcode == Decoder::AdvSimdThreeSameOpcode::kUminp);
        bool is_max = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
                       args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmaxp);
        uint64_t emask = ElementMask(esize);
        uint8_t half_elements = num_elements;
        result = 0;
        for (uint8_t i = 0; i < half_elements / 2; i++) {
          uint64_t a = 0, b = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + (2 * i) * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_n) + (2 * i + 1) * esize, esize);
          a &= emask;
          b &= emask;
          bool pick_a;
          if (is_unsigned) {
            pick_a = is_max ? (a >= b) : (a <= b);
          } else {
            uint8_t bits = esize * 8;
            int64_t sa = (a ^ (1ULL << (bits - 1))) - (1ULL << (bits - 1));
            int64_t sb = (b ^ (1ULL << (bits - 1))) - (1ULL << (bits - 1));
            pick_a = is_max ? (sa >= sb) : (sa <= sb);
          }
          uint64_t val = (pick_a ? a : b) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &val, esize);
        }
        for (uint8_t i = 0; i < half_elements / 2; i++) {
          uint64_t a = 0, b = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_m) + (2 * i) * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + (2 * i + 1) * esize, esize);
          a &= emask;
          b &= emask;
          bool pick_a;
          if (is_unsigned) {
            pick_a = is_max ? (a >= b) : (a <= b);
          } else {
            uint8_t bits = esize * 8;
            int64_t sa = (a ^ (1ULL << (bits - 1))) - (1ULL << (bits - 1));
            int64_t sb = (b ^ (1ULL << (bits - 1))) - (1ULL << (bits - 1));
            pick_a = is_max ? (sa >= sb) : (sa <= sb);
          }
          uint64_t val = (pick_a ? a : b) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + (half_elements / 2 + i) * esize, &val, esize);
        }
        break;
      }
      // endregion

      // --- MUL / MLA / MLS ---
      case Decoder::AdvSimdThreeSameOpcode::kMul:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t { return a * b; });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kMla: {
        // MLA: Vd[i] = Vd[i] + Vn[i] * Vm[i]
        uint64_t emask = ElementMask(esize);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a = 0, b = 0, d = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * esize, esize);
          memcpy(&d, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          uint64_t r = (d + a * b) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }
      case Decoder::AdvSimdThreeSameOpcode::kMls: {
        // MLS: Vd[i] = Vd[i] - Vn[i] * Vm[i]
        uint64_t emask = ElementMask(esize);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a = 0, b = 0, d = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * esize, esize);
          memcpy(&d, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          uint64_t r = (d - a * b) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      // --- FP three-same vector ops (Digitalis addition) ---
      // For FP cases args.size is sz alone (0 = single 32-bit, 1 = double 64-bit),
      // not the {op_high, sz} pair the raw encoding carries; the decoder
      // already split that.
      case Decoder::AdvSimdThreeSameOpcode::kFaddV:
      case Decoder::AdvSimdThreeSameOpcode::kFsubV:
      case Decoder::AdvSimdThreeSameOpcode::kFmulV:
      case Decoder::AdvSimdThreeSameOpcode::kFmlaV:
      case Decoder::AdvSimdThreeSameOpcode::kFmlsV:
      case Decoder::AdvSimdThreeSameOpcode::kFmaxV:
      case Decoder::AdvSimdThreeSameOpcode::kFminV:
      case Decoder::AdvSimdThreeSameOpcode::kFmaxnmV:
      case Decoder::AdvSimdThreeSameOpcode::kFminnmV:
      case Decoder::AdvSimdThreeSameOpcode::kFdivV:
      case Decoder::AdvSimdThreeSameOpcode::kFcmeqV:
      case Decoder::AdvSimdThreeSameOpcode::kFcmgeV:
      case Decoder::AdvSimdThreeSameOpcode::kFcmgtV:
      case Decoder::AdvSimdThreeSameOpcode::kFacgeV:
      case Decoder::AdvSimdThreeSameOpcode::kFacgtV:
      case Decoder::AdvSimdThreeSameOpcode::kFabdV: {
        bool is_double = (args.size == 0b01);
        uint8_t fp_esize = is_double ? 8 : 4;
        uint8_t fp_num = vec_len / fp_esize;
        for (uint8_t i = 0; i < fp_num; i++) {
          if (is_double) {
            double a, b, d, r;
            memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * 8, 8);
            memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * 8, 8);
            memcpy(&d, reinterpret_cast<const uint8_t*>(&dst) + i * 8, 8);
            switch (args.opcode) {
              case Decoder::AdvSimdThreeSameOpcode::kFaddV: r = a + b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFsubV: r = a - b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFmulV: r = a * b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFmlaV: r = d + a * b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFmlsV: r = d - a * b; break;
              // FMAX/FMIN: IEEE 754-2008 max/min — if either is NaN, result is NaN.
              case Decoder::AdvSimdThreeSameOpcode::kFmaxV:
                r = (std::isnan(a) || std::isnan(b)) ? std::nan("") : (a > b ? a : b);
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFminV:
                r = (std::isnan(a) || std::isnan(b)) ? std::nan("") : (a < b ? a : b);
                break;
              // FMAXNM/FMINNM: max/min number — if exactly one is NaN, return the other.
              case Decoder::AdvSimdThreeSameOpcode::kFmaxnmV:
                r = std::isnan(a) ? b : (std::isnan(b) ? a : (a > b ? a : b));
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFminnmV:
                r = std::isnan(a) ? b : (std::isnan(b) ? a : (a < b ? a : b));
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFdivV: r = a / b; break;
              // FP compare: result is all-ones (bit pattern) on TRUE, zero on FALSE.
              case Decoder::AdvSimdThreeSameOpcode::kFcmeqV: {
                uint64_t bits = (a == b) ? ~uint64_t{0} : 0;
                memcpy(&r, &bits, 8); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFcmgeV: {
                uint64_t bits = (a >= b) ? ~uint64_t{0} : 0;
                memcpy(&r, &bits, 8); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFcmgtV: {
                uint64_t bits = (a > b) ? ~uint64_t{0} : 0;
                memcpy(&r, &bits, 8); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFacgeV: {
                uint64_t bits = (std::fabs(a) >= std::fabs(b)) ? ~uint64_t{0} : 0;
                memcpy(&r, &bits, 8); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFacgtV: {
                uint64_t bits = (std::fabs(a) > std::fabs(b)) ? ~uint64_t{0} : 0;
                memcpy(&r, &bits, 8); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFabdV:
                r = std::fabs(a - b); break;
              default: Undefined(); return;
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
          } else {
            float a, b, d, r;
            memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * 4, 4);
            memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * 4, 4);
            memcpy(&d, reinterpret_cast<const uint8_t*>(&dst) + i * 4, 4);
            switch (args.opcode) {
              case Decoder::AdvSimdThreeSameOpcode::kFaddV: r = a + b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFsubV: r = a - b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFmulV: r = a * b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFmlaV: r = d + a * b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFmlsV: r = d - a * b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFmaxV:
                r = (std::isnan(a) || std::isnan(b)) ? std::nanf("") : (a > b ? a : b);
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFminV:
                r = (std::isnan(a) || std::isnan(b)) ? std::nanf("") : (a < b ? a : b);
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFmaxnmV:
                r = std::isnan(a) ? b : (std::isnan(b) ? a : (a > b ? a : b));
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFminnmV:
                r = std::isnan(a) ? b : (std::isnan(b) ? a : (a < b ? a : b));
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFdivV: r = a / b; break;
              case Decoder::AdvSimdThreeSameOpcode::kFcmeqV: {
                uint32_t bits = (a == b) ? 0xFFFFFFFFu : 0;
                memcpy(&r, &bits, 4); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFcmgeV: {
                uint32_t bits = (a >= b) ? 0xFFFFFFFFu : 0;
                memcpy(&r, &bits, 4); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFcmgtV: {
                uint32_t bits = (a > b) ? 0xFFFFFFFFu : 0;
                memcpy(&r, &bits, 4); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFacgeV: {
                uint32_t bits = (std::fabs(a) >= std::fabs(b)) ? 0xFFFFFFFFu : 0;
                memcpy(&r, &bits, 4); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFacgtV: {
                uint32_t bits = (std::fabs(a) > std::fabs(b)) ? 0xFFFFFFFFu : 0;
                memcpy(&r, &bits, 4); break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFabdV:
                r = std::fabs(a - b); break;
              default: Undefined(); return;
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
          }
        }
        // Zero upper 64 bits if Q=0 (already covered by 'result' starting at 0
        // and the loop only writing the lower lanes when fp_num < 16 / esize).
        break;
      }

      // --- Opcodes in the enum but not mapped by decoder (CMGT etc.) ---
      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = result;
  }
  // endregion

  // region digitalis
  //
  // FP data-processing (1 source): FMOV, FABS, FNEG, FSQRT, FCVT, FRINTx.
  //
  void FpDataProc1(const Decoder::FpDataProc1Args& args) {
    CHECK(!exception_raised_);
    uint8_t ftype = args.ftype;
    uint8_t opcode = args.opcode;

    if (ftype == 0b00) {
      // Single-precision.
      float src;
      memcpy(&src, &state_->cpu.v[args.rn], 4);
      float result;
      bool write_single = true;

      switch (opcode) {
        case 0b000000:  // FMOV Sd, Sn
          result = src;
          break;
        case 0b000001:  // FABS
          result = std::fabs(src);
          break;
        case 0b000010:  // FNEG
          result = -src;
          break;
        case 0b000011:  // FSQRT
          result = std::sqrt(src);
          break;
        case 0b000101: {  // FCVT Dd, Sn (single -> double)
          double d = static_cast<double>(src);
          state_->cpu.v[args.rd] = 0;
          memcpy(&state_->cpu.v[args.rd], &d, 8);
          return;
        }
        case 0b000111: {  // FCVT Hd, Sn (single -> half) - approximate
          // Store as half-precision using truncation to 16-bit float.
          // For simplicity, store the lower 16 bits of the float representation.
          uint16_t half = FpSingleToHalf(src);
          state_->cpu.v[args.rd] = 0;
          memcpy(&state_->cpu.v[args.rd], &half, 2);
          return;
        }
        case 0b001000:  // FRINTN (round to nearest, ties to even)
          result = std::nearbyint(src);
          break;
        case 0b001001:  // FRINTP (round toward +inf)
          result = std::ceil(src);
          break;
        case 0b001010:  // FRINTM (round toward -inf)
          result = std::floor(src);
          break;
        case 0b001011:  // FRINTZ (round toward zero)
          result = std::trunc(src);
          break;
        case 0b001100:  // FRINTA (round to nearest, ties away from zero)
          result = std::round(src);
          break;
        case 0b001110:  // FRINTX (round to nearest, exact, signal inexact)
          result = std::rint(src);
          break;
        case 0b001111:  // FRINTI (round using FPCR rounding mode)
          result = std::rint(src);
          break;
        default:
          Undefined();
          return;
      }

      if (write_single) {
        state_->cpu.v[args.rd] = 0;
        memcpy(&state_->cpu.v[args.rd], &result, 4);
      }
    } else if (ftype == 0b01) {
      // Double-precision.
      double src;
      memcpy(&src, &state_->cpu.v[args.rn], 8);
      double result;
      bool write_double = true;

      switch (opcode) {
        case 0b000000:  // FMOV Dd, Dn
          result = src;
          break;
        case 0b000001:  // FABS
          result = std::fabs(src);
          break;
        case 0b000010:  // FNEG
          result = -src;
          break;
        case 0b000011:  // FSQRT
          result = std::sqrt(src);
          break;
        case 0b000100: {  // FCVT Sd, Dn (double -> single)
          float f = static_cast<float>(src);
          state_->cpu.v[args.rd] = 0;
          memcpy(&state_->cpu.v[args.rd], &f, 4);
          return;
        }
        case 0b000111: {  // FCVT Hd, Dn (double -> half) - approximate
          float f = static_cast<float>(src);
          uint16_t half = FpSingleToHalf(f);
          state_->cpu.v[args.rd] = 0;
          memcpy(&state_->cpu.v[args.rd], &half, 2);
          return;
        }
        case 0b001000:  // FRINTN
          result = std::nearbyint(src);
          break;
        case 0b001001:  // FRINTP
          result = std::ceil(src);
          break;
        case 0b001010:  // FRINTM
          result = std::floor(src);
          break;
        case 0b001011:  // FRINTZ
          result = std::trunc(src);
          break;
        case 0b001100:  // FRINTA
          result = std::round(src);
          break;
        case 0b001110:  // FRINTX
          result = std::rint(src);
          break;
        case 0b001111:  // FRINTI
          result = std::rint(src);
          break;
        default:
          Undefined();
          return;
      }

      if (write_double) {
        state_->cpu.v[args.rd] = 0;
        memcpy(&state_->cpu.v[args.rd], &result, 8);
      }
    } else if (ftype == 0b11) {
      // Half-precision source — only FCVT to single/double is common.
      if (opcode == 0b000100) {
        // FCVT Sd, Hn (half -> single)
        uint16_t half;
        memcpy(&half, &state_->cpu.v[args.rn], 2);
        float f = FpHalfToSingle(half);
        state_->cpu.v[args.rd] = 0;
        memcpy(&state_->cpu.v[args.rd], &f, 4);
        return;
      }
      if (opcode == 0b000101) {
        // FCVT Dd, Hn (half -> double)
        uint16_t half;
        memcpy(&half, &state_->cpu.v[args.rn], 2);
        float f = FpHalfToSingle(half);
        double d = static_cast<double>(f);
        state_->cpu.v[args.rd] = 0;
        memcpy(&state_->cpu.v[args.rd], &d, 8);
        return;
      }
      Undefined();
    } else {
      Undefined();
    }
  }

  //
  // FP data-processing (2 source): FMUL, FDIV, FADD, FSUB, FMAX, FMIN, FNMUL.
  //
  void FpDataProc2(const Decoder::FpDataProc2Args& args) {
    CHECK(!exception_raised_);
    uint8_t ftype = args.ftype;
    uint8_t opcode = args.opcode;

    if (ftype == 0b00) {
      // Single-precision.
      float src_n, src_m;
      memcpy(&src_n, &state_->cpu.v[args.rn], 4);
      memcpy(&src_m, &state_->cpu.v[args.rm], 4);
      float result;

      switch (opcode) {
        case 0b0000: result = src_n * src_m; break;       // FMUL
        case 0b0001: result = src_n / src_m; break;       // FDIV
        case 0b0010: result = src_n + src_m; break;       // FADD
        case 0b0011: result = src_n - src_m; break;       // FSUB
        case 0b0100: result = std::fmax(src_n, src_m); break;  // FMAX
        case 0b0101: result = std::fmin(src_n, src_m); break;  // FMIN
        case 0b0110: result = std::fmax(src_n, src_m); break;  // FMAXNM (same as FMAX for non-NaN)
        case 0b0111: result = std::fmin(src_n, src_m); break;  // FMINNM (same as FMIN for non-NaN)
        case 0b1000: result = -(src_n * src_m); break;    // FNMUL
        default: Undefined(); return;
      }

      state_->cpu.v[args.rd] = 0;
      memcpy(&state_->cpu.v[args.rd], &result, 4);
    } else if (ftype == 0b01) {
      // Double-precision.
      double src_n, src_m;
      memcpy(&src_n, &state_->cpu.v[args.rn], 8);
      memcpy(&src_m, &state_->cpu.v[args.rm], 8);
      double result;

      switch (opcode) {
        case 0b0000: result = src_n * src_m; break;       // FMUL
        case 0b0001: result = src_n / src_m; break;       // FDIV
        case 0b0010: result = src_n + src_m; break;       // FADD
        case 0b0011: result = src_n - src_m; break;       // FSUB
        case 0b0100: result = std::fmax(src_n, src_m); break;  // FMAX
        case 0b0101: result = std::fmin(src_n, src_m); break;  // FMIN
        case 0b0110: result = std::fmax(src_n, src_m); break;  // FMAXNM
        case 0b0111: result = std::fmin(src_n, src_m); break;  // FMINNM
        case 0b1000: result = -(src_n * src_m); break;    // FNMUL
        default: Undefined(); return;
      }

      state_->cpu.v[args.rd] = 0;
      memcpy(&state_->cpu.v[args.rd], &result, 8);
    } else {
      Undefined();
    }
  }

  //
  // FP compare: FCMP, FCMPE — set NZCV flags.
  //
  void FpCompare(const Decoder::FpCompareArgs& args) {
    CHECK(!exception_raised_);
    uint16_t flags = 0;

    if (args.ftype == 0b00) {
      // Single-precision.
      float src_n;
      memcpy(&src_n, &state_->cpu.v[args.rn], 4);
      float src_m;
      if (args.with_zero) {
        src_m = 0.0f;
      } else {
        memcpy(&src_m, &state_->cpu.v[args.rm], 4);
      }

      if (std::isnan(src_n) || std::isnan(src_m)) {
        // Unordered: N=0, Z=0, C=1, V=1
        flags = CPUState::kFlagCarry | CPUState::kFlagOverflow;
      } else if (src_n == src_m) {
        // Equal: N=0, Z=1, C=1, V=0
        flags = CPUState::kFlagZero | CPUState::kFlagCarry;
      } else if (src_n < src_m) {
        // Less than: N=1, Z=0, C=0, V=0
        flags = CPUState::kFlagNegative;
      } else {
        // Greater than: N=0, Z=0, C=1, V=0
        flags = CPUState::kFlagCarry;
      }
    } else if (args.ftype == 0b01) {
      // Double-precision.
      double src_n;
      memcpy(&src_n, &state_->cpu.v[args.rn], 8);
      double src_m;
      if (args.with_zero) {
        src_m = 0.0;
      } else {
        memcpy(&src_m, &state_->cpu.v[args.rm], 8);
      }

      if (std::isnan(src_n) || std::isnan(src_m)) {
        flags = CPUState::kFlagCarry | CPUState::kFlagOverflow;
      } else if (src_n == src_m) {
        flags = CPUState::kFlagZero | CPUState::kFlagCarry;
      } else if (src_n < src_m) {
        flags = CPUState::kFlagNegative;
      } else {
        flags = CPUState::kFlagCarry;
      }
    } else {
      Undefined();
      return;
    }

    state_->cpu.flags = flags;
  }

  // region digitalis
  //
  // AdvSIMD scalar two-reg misc: scalar UCVTF, SCVTF, FCVTZS, FCVTZU.
  //
  void AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src = state_->cpu.v[args.rn];
    __uint128_t result = 0;

    switch (args.opcode) {
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kUcvtf: {
        if (args.size == 0) {
          // UCVTF Sd, Sn: uint32 → float32
          uint32_t ival;
          memcpy(&ival, &src, sizeof(ival));
          float fval = static_cast<float>(ival);
          memcpy(&result, &fval, sizeof(fval));
        } else {
          // UCVTF Dd, Dn: uint64 → float64
          uint64_t ival;
          memcpy(&ival, &src, sizeof(ival));
          double fval = static_cast<double>(ival);
          memcpy(&result, &fval, sizeof(fval));
        }
        break;
      }
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kScvtf: {
        if (args.size == 0) {
          // SCVTF Sd, Sn: int32 → float32
          int32_t ival;
          memcpy(&ival, &src, sizeof(ival));
          float fval = static_cast<float>(ival);
          memcpy(&result, &fval, sizeof(fval));
        } else {
          // SCVTF Dd, Dn: int64 → float64
          int64_t ival;
          memcpy(&ival, &src, sizeof(ival));
          double fval = static_cast<double>(ival);
          memcpy(&result, &fval, sizeof(fval));
        }
        break;
      }
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtzu: {
        if (args.size == 0) {
          // FCVTZU Sd, Sn: float32 → uint32, round toward zero
          float fval;
          memcpy(&fval, &src, sizeof(fval));
          uint32_t ival;
          if (std::isnan(fval) || fval < 0.0f) {
            ival = 0;
          } else if (fval >= static_cast<float>(UINT32_MAX)) {
            ival = UINT32_MAX;
          } else {
            ival = static_cast<uint32_t>(fval);
          }
          memcpy(&result, &ival, sizeof(ival));
        } else {
          // FCVTZU Dd, Dn: float64 → uint64, round toward zero
          double fval;
          memcpy(&fval, &src, sizeof(fval));
          uint64_t ival;
          if (std::isnan(fval) || fval < 0.0) {
            ival = 0;
          } else if (fval >= static_cast<double>(UINT64_MAX)) {
            ival = UINT64_MAX;
          } else {
            ival = static_cast<uint64_t>(fval);
          }
          memcpy(&result, &ival, sizeof(ival));
        }
        break;
      }
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtzs: {
        if (args.size == 0) {
          // FCVTZS Sd, Sn: float32 → int32, round toward zero
          float fval;
          memcpy(&fval, &src, sizeof(fval));
          int32_t ival;
          if (std::isnan(fval)) {
            ival = 0;
          } else if (fval >= static_cast<float>(INT32_MAX)) {
            ival = INT32_MAX;
          } else if (fval <= static_cast<float>(INT32_MIN)) {
            ival = INT32_MIN;
          } else {
            ival = static_cast<int32_t>(fval);
          }
          uint32_t uval;
          memcpy(&uval, &ival, sizeof(uval));
          memcpy(&result, &uval, sizeof(uval));
        } else {
          // FCVTZS Dd, Dn: float64 → int64, round toward zero
          double fval;
          memcpy(&fval, &src, sizeof(fval));
          int64_t ival;
          if (std::isnan(fval)) {
            ival = 0;
          } else if (fval >= static_cast<double>(INT64_MAX)) {
            ival = INT64_MAX;
          } else if (fval <= static_cast<double>(INT64_MIN)) {
            ival = INT64_MIN;
          } else {
            ival = static_cast<int64_t>(fval);
          }
          uint64_t uval;
          memcpy(&uval, &ival, sizeof(uval));
          memcpy(&result, &uval, sizeof(uval));
        }
        break;
      }
    }

    state_->cpu.v[args.rd] = result;
  }
  // endregion

  // region digitalis
  //
  // AdvSIMD scalar three same: scalar (D-form, 64-bit) integer 3-operand ops.
  // Operates on the bottom 64-bit element of each register; upper bits zero.
  //
  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs& args) {
    CHECK(!exception_raised_);

    // FP scalar ops: dispatch separately because size encodes S (0) vs D (1).
    switch (args.opcode) {
      case Decoder::AdvSimdScalarThreeSameOpcode::kFabd:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFcmgt:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFcmge:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFcmeq:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFacgt:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFacge: {
        __uint128_t src_n = state_->cpu.v[args.rn];
        __uint128_t src_m = state_->cpu.v[args.rm];
        __uint128_t result = 0;
        if (args.size == 1) {
          double a, b;
          memcpy(&a, &src_n, sizeof(a));
          memcpy(&b, &src_m, sizeof(b));
          uint64_t r64;
          switch (args.opcode) {
            case Decoder::AdvSimdScalarThreeSameOpcode::kFabd: {
              double d = std::fabs(a - b);
              memcpy(&r64, &d, sizeof(r64));
              break;
            }
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmgt:
              r64 = (a > b) ? 0xFFFFFFFFFFFFFFFFULL : 0; break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmge:
              r64 = (a >= b) ? 0xFFFFFFFFFFFFFFFFULL : 0; break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmeq:
              r64 = (a == b) ? 0xFFFFFFFFFFFFFFFFULL : 0; break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFacgt:
              r64 = (std::fabs(a) > std::fabs(b)) ? 0xFFFFFFFFFFFFFFFFULL : 0; break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFacge:
              r64 = (std::fabs(a) >= std::fabs(b)) ? 0xFFFFFFFFFFFFFFFFULL : 0; break;
            default: r64 = 0; break;
          }
          result = static_cast<__uint128_t>(r64);
        } else {
          float a, b;
          memcpy(&a, &src_n, sizeof(a));
          memcpy(&b, &src_m, sizeof(b));
          uint32_t r32;
          switch (args.opcode) {
            case Decoder::AdvSimdScalarThreeSameOpcode::kFabd: {
              float f = std::fabs(a - b);
              memcpy(&r32, &f, sizeof(r32));
              break;
            }
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmgt:
              r32 = (a > b) ? 0xFFFFFFFFu : 0; break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmge:
              r32 = (a >= b) ? 0xFFFFFFFFu : 0; break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmeq:
              r32 = (a == b) ? 0xFFFFFFFFu : 0; break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFacgt:
              r32 = (std::fabs(a) > std::fabs(b)) ? 0xFFFFFFFFu : 0; break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFacge:
              r32 = (std::fabs(a) >= std::fabs(b)) ? 0xFFFFFFFFu : 0; break;
            default: r32 = 0; break;
          }
          result = static_cast<__uint128_t>(r32);
        }
        state_->cpu.v[args.rd] = result;
        return;
      }
      default:
        break;
    }

    // D-form integer ops below.
    uint64_t a = static_cast<uint64_t>(state_->cpu.v[args.rn]);
    uint64_t b = static_cast<uint64_t>(state_->cpu.v[args.rm]);
    uint64_t r;

    switch (args.opcode) {
      case Decoder::AdvSimdScalarThreeSameOpcode::kAdd:
        r = a + b;
        break;
      case Decoder::AdvSimdScalarThreeSameOpcode::kSub:
        r = a - b;
        break;
      case Decoder::AdvSimdScalarThreeSameOpcode::kCmgt:
        r = (static_cast<int64_t>(a) > static_cast<int64_t>(b)) ? 0xFFFFFFFFFFFFFFFFULL : 0;
        break;
      case Decoder::AdvSimdScalarThreeSameOpcode::kCmhi:
        r = (a > b) ? 0xFFFFFFFFFFFFFFFFULL : 0;
        break;
      case Decoder::AdvSimdScalarThreeSameOpcode::kCmge:
        r = (static_cast<int64_t>(a) >= static_cast<int64_t>(b)) ? 0xFFFFFFFFFFFFFFFFULL : 0;
        break;
      case Decoder::AdvSimdScalarThreeSameOpcode::kCmhs:
        r = (a >= b) ? 0xFFFFFFFFFFFFFFFFULL : 0;
        break;
      case Decoder::AdvSimdScalarThreeSameOpcode::kCmtst:
        r = ((a & b) != 0) ? 0xFFFFFFFFFFFFFFFFULL : 0;
        break;
      case Decoder::AdvSimdScalarThreeSameOpcode::kCmeq:
        r = (a == b) ? 0xFFFFFFFFFFFFFFFFULL : 0;
        break;
      case Decoder::AdvSimdScalarThreeSameOpcode::kSshl: {
        // SSHL: shift left by signed amount from Rm[7:0].
        int8_t sh = static_cast<int8_t>(b & 0xFF);
        int64_t sa = static_cast<int64_t>(a);
        if (sh >= 64) { r = 0; }
        else if (sh >= 0) { r = static_cast<uint64_t>(sa << sh); }
        else if (sh <= -64) { r = static_cast<uint64_t>(sa >> 63); }  // arithmetic
        else { r = static_cast<uint64_t>(sa >> (-sh)); }
        break;
      }
      case Decoder::AdvSimdScalarThreeSameOpcode::kUshl: {
        // USHL: shift left by signed amount from Rm[7:0] (logical for negatives).
        int8_t sh = static_cast<int8_t>(b & 0xFF);
        if (sh >= 64) { r = 0; }
        else if (sh >= 0) { r = a << sh; }
        else if (sh <= -64) { r = 0; }
        else { r = a >> (-sh); }
        break;
      }
      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = static_cast<__uint128_t>(r);
  }
  // endregion

  // region digitalis
  //
  // AdvSIMD scalar pairwise.
  // ADDP scalar (D-form): Vd[0] = Vn.D[0] + Vn.D[1].
  //
  void AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src = state_->cpu.v[args.rn];
    uint64_t lo = static_cast<uint64_t>(src);
    uint64_t hi = static_cast<uint64_t>(src >> 64);

    uint64_t r;
    switch (args.opcode) {
      case Decoder::AdvSimdScalarPairwiseOpcode::kAddp:
        // D-form only.
        r = lo + hi;
        break;
      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = static_cast<__uint128_t>(r);
  }
  // endregion

  //
  // AdvSIMD two-reg misc: unary element-wise vector operations.
  //
  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src = state_->cpu.v[args.rn];
    __uint128_t result = 0;

    uint8_t esize;  // element size in bytes
    switch (args.size) {
      case 0b00: esize = 1; break;
      case 0b01: esize = 2; break;
      case 0b10: esize = 4; break;
      case 0b11: esize = 8; break;
      default: Undefined(); return;
    }

    uint8_t vec_len = args.q ? 16 : 8;
    uint8_t num_elements = vec_len / esize;

    switch (args.opcode) {
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev64: {
        // REV64: reverse bytes within each 64-bit element.
        // Element size determines the unit of reversal.
        for (uint8_t i = 0; i < vec_len; i += 8) {
          uint8_t group[8];
          memcpy(group, reinterpret_cast<const uint8_t*>(&src) + i, 8);
          // Reverse units of esize bytes within the 8-byte group.
          uint8_t reversed[8];
          uint8_t units = 8 / esize;
          for (uint8_t j = 0; j < units; j++) {
            memcpy(reversed + j * esize, group + (units - 1 - j) * esize, esize);
          }
          memcpy(reinterpret_cast<uint8_t*>(&result) + i, reversed, 8);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kRev32: {
        // REV32: reverse bytes within each 32-bit element.
        for (uint8_t i = 0; i < vec_len; i += 4) {
          uint8_t group[4];
          memcpy(group, reinterpret_cast<const uint8_t*>(&src) + i, 4);
          uint8_t reversed[4];
          uint8_t units = 4 / esize;
          for (uint8_t j = 0; j < units; j++) {
            memcpy(reversed + j * esize, group + (units - 1 - j) * esize, esize);
          }
          memcpy(reinterpret_cast<uint8_t*>(&result) + i, reversed, 4);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kRev16: {
        // REV16: reverse bytes within each 16-bit element.
        for (uint8_t i = 0; i < vec_len; i += 2) {
          uint8_t a, b;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src) + i, 1);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src) + i + 1, 1);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i, &b, 1);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i + 1, &a, 1);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kNot: {
        if (args.size == 0b00) {
          // NOT (bitwise NOT): U=1, size=00.
          result = ~src;
          if (!args.q) {
            uint64_t lo;
            memcpy(&lo, &result, 8);
            result = 0;
            memcpy(&result, &lo, 8);
          }
        } else if (args.size == 0b01) {
          // RBIT (reverse bits per byte): U=1, size=01.
          for (uint8_t i = 0; i < vec_len; i++) {
            uint8_t byte_val;
            memcpy(&byte_val, reinterpret_cast<const uint8_t*>(&src) + i, 1);
            uint8_t reversed = 0;
            for (int b = 0; b < 8; b++) {
              reversed |= ((byte_val >> b) & 1) << (7 - b);
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i, &reversed, 1);
          }
        } else {
          Undefined();
          return;
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kCnt: {
        // CNT: count set bits per byte.
        for (uint8_t i = 0; i < vec_len; i++) {
          uint8_t byte_val;
          memcpy(&byte_val, reinterpret_cast<const uint8_t*>(&src) + i, 1);
          uint8_t count = __builtin_popcount(byte_val);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i, &count, 1);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kClz: {
        // CLZ: count leading zeros per element.
        uint64_t emask = ElementMask(esize);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          uint64_t clz;
          if (elem == 0) {
            clz = esize * 8;
          } else {
            clz = __builtin_clzll(elem) - (64 - esize * 8);
          }
          clz &= emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &clz, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kAbs: {
        // ABS: absolute value per signed element.
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          uint64_t abs_val = static_cast<uint64_t>(signed_val < 0 ? -signed_val : signed_val) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &abs_val, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kNeg: {
        // NEG: negate per element.
        uint64_t emask = ElementMask(esize);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          uint64_t neg_val = (0 - elem) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &neg_val, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero: {
        // CMGT #0: compare signed > 0, result = all-ones or all-zeros.
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          uint64_t r = (signed_val > 0) ? emask : 0;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero: {
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          uint64_t r = (signed_val >= 0) ? emask : 0;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kCmeqZero: {
        uint64_t emask = ElementMask(esize);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          uint64_t r = (elem == 0) ? emask : 0;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero: {
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          uint64_t r = (signed_val <= 0) ? emask : 0;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kCmltZero: {
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          uint64_t r = (signed_val < 0) ? emask : 0;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kXtn: {
        // XTN: extract narrow — take lower half of each wider element.
        // Source element size is 2*esize, destination element size is esize.
        // Q=0: lower half of result, Q=1: upper half (XTN2).
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_count = 16 / src_esize;  // always operate on full 128-bit source
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, esize);
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &elem, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kFabs: {
        // FABS (vector): floating-point absolute value per element.
        if (args.size == 0b10) {
          // Single-precision elements (size=10 means float for this FP opcode group).
          uint8_t fp_count = args.q ? 4 : 2;
          for (uint8_t i = 0; i < fp_count; i++) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            f = std::fabs(f);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &f, 4);
          }
        } else if (args.size == 0b11 && args.q) {
          // Double-precision elements (size=11, Q=1 only).
          for (uint8_t i = 0; i < 2; i++) {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            d = std::fabs(d);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &d, 8);
          }
        } else {
          Undefined();
          return;
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kFneg: {
        // FNEG (vector): floating-point negate per element.
        if (args.size == 0b10) {
          uint8_t fp_count = args.q ? 4 : 2;
          for (uint8_t i = 0; i < fp_count; i++) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            f = -f;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &f, 4);
          }
        } else if (args.size == 0b11 && args.q) {
          for (uint8_t i = 0; i < 2; i++) {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            d = -d;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &d, 8);
          }
        } else {
          Undefined();
          return;
        }
        break;
      }

      // region digitalis
      case Decoder::AdvSimdTwoRegMiscOpcode::kAddv: {
        // ADDV: add across vector — sum all elements, produce scalar result.
        uint64_t emask = ElementMask(esize);
        uint64_t sum = 0;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          sum += elem & emask;
        }
        result = sum & emask;  // scalar result in bottom esize bytes, upper zeroed
        break;
      }
      // endregion

      // region digitalis - across-lanes max/min reductions
      // SMAXV/UMAXV/SMINV/UMINV: reduce a vector to a single scalar lane holding
      // the signed/unsigned max or min across all input lanes. The scalar result
      // is placed in the bottom esize bytes of Vd; upper bits are zeroed.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSminv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUminv: {
        if (esize > 4) { Undefined(); return; }  // no 64-bit element form
        bool is_signed =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv ||
             args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSminv);
        bool is_max =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv ||
             args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUmaxv);
        uint8_t bits = esize * 8;
        uint64_t emask = ElementMask(esize);
        // Seed accumulator from element 0.
        uint64_t acc = 0;
        memcpy(&acc, reinterpret_cast<const uint8_t*>(&src), esize);
        for (uint8_t i = 1; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          bool elem_wins;
          if (is_signed) {
            int64_t s_elem = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
            int64_t s_acc = static_cast<int64_t>(acc << (64 - bits)) >> (64 - bits);
            elem_wins = is_max ? (s_elem > s_acc) : (s_elem < s_acc);
          } else {
            elem_wins = is_max ? (elem > acc) : (elem < acc);
          }
          if (elem_wins) acc = elem;
        }
        result = acc & emask;
        break;
      }
      // endregion

      // region digitalis - SCVTF/UCVTF (vector, integer): per-lane signed or
      // unsigned int-to-FP. Element size from `size` field: sz=0 -> single
      // (.4S / .2S), sz=1 -> double (.2D). Observed `ucvtf v0.4s, v0.4s`
      // (insn 0x6e21d800) in WhatsApp's libar-bundle3.so init path.
      case Decoder::AdvSimdTwoRegMiscOpcode::kScvtfV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV: {
        // The decoder uses bit22 (sz) as the LOW bit of `size`; for FP
        // two-reg-misc the high bit of `size` is reserved. So sz = size&1.
        // Element width: sz=0 -> 32-bit (float), sz=1 -> 64-bit (double).
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        if (args.size & 0b10) { Undefined(); return; }
        uint8_t fp_count = vec_len / fp_esize;
        bool is_unsigned = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV);
        for (uint8_t i = 0; i < fp_count; i++) {
          uint64_t int_bits = 0;
          memcpy(&int_bits, reinterpret_cast<const uint8_t*>(&src) + i * fp_esize, fp_esize);
          if (fp_esize == 4) {
            float f;
            if (is_unsigned) {
              f = static_cast<float>(static_cast<uint32_t>(int_bits));
            } else {
              f = static_cast<float>(static_cast<int32_t>(int_bits));
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * fp_esize, &f, 4);
          } else {
            double d;
            if (is_unsigned) {
              d = static_cast<double>(int_bits);
            } else {
              d = static_cast<double>(static_cast<int64_t>(int_bits));
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * fp_esize, &d, 8);
          }
        }
        break;
      }
      // endregion

      // region digitalis - FCVTZS/FCVTZU (vector, FP→int): per-lane FP-to-int
      // truncating conversion. sz=0 -> 32-bit float→int32, sz=1 -> 64-bit
      // double→int64. Out-of-range values saturate per the ARM ARM spec.
      // The decoder routes opcode=11011 with bit23=1 here, so args.size's
      // high bit is always 1 -- only the low bit (sz) selects single vs
      // double, unlike SCVTF/UCVTF whose bit23=0 path keeps size's high
      // bit clear. We don't reject "size & 0b10" the way SCVTF does.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzuV: {
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        uint8_t fp_count = vec_len / fp_esize;
        bool is_unsigned =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzuV);
        for (uint8_t i = 0; i < fp_count; i++) {
          if (fp_esize == 4) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            uint32_t out;
            if (is_unsigned) {
              uint32_t v;
              if (f != f /* NaN */ || f < 0.0f) v = 0u;
              else if (f >= 4294967296.0f) v = 0xffffffffu;
              else v = static_cast<uint32_t>(f);  // truncates toward zero
              out = v;
            } else {
              int32_t v;
              if (f != f) v = 0;
              else if (f >= 2147483648.0f) v = 0x7fffffff;
              else if (f < -2147483648.0f) v = static_cast<int32_t>(0x80000000);
              else v = static_cast<int32_t>(f);
              memcpy(&out, &v, 4);
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &out, 4);
          } else {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            uint64_t out;
            if (is_unsigned) {
              uint64_t v;
              if (d != d || d < 0.0) v = 0u;
              else if (d >= 18446744073709551616.0) v = 0xffffffffffffffffULL;
              else v = static_cast<uint64_t>(d);
              out = v;
            } else {
              int64_t v;
              if (d != d) v = 0;
              else if (d >= 9223372036854775808.0) v = 0x7fffffffffffffffLL;
              else if (d < -9223372036854775808.0)
                v = static_cast<int64_t>(0x8000000000000000ULL);
              else v = static_cast<int64_t>(d);
              memcpy(&out, &v, 8);
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &out, 8);
          }
        }
        break;
      }
      // endregion

      // region digitalis - FRECPE / FRSQRTE (vector): per-lane reciprocal /
      // reciprocal-square-root estimate. The ARM spec only requires ~8 bits
      // of mantissa precision; computing 1/x and 1/sqrt(x) in full precision
      // is well within that bound, so callers that need the estimate as a
      // Newton-Raphson seed will converge identically.
      // region digitalis - FSQRT (vector): per-lane square root.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFsqrtV: {
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        uint8_t fp_count = vec_len / fp_esize;
        for (uint8_t i = 0; i < fp_count; i++) {
          if (fp_esize == 4) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            float r = __builtin_sqrtf(f);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
          } else {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            double r = __builtin_sqrt(d);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
          }
        }
        break;
      }
      // endregion

      case Decoder::AdvSimdTwoRegMiscOpcode::kFrecpeV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrsqrteV: {
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        // Same bit23=1 rationale as the FCVTZS case above.
        uint8_t fp_count = vec_len / fp_esize;
        bool is_rsqrt =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFrsqrteV);
        for (uint8_t i = 0; i < fp_count; i++) {
          if (fp_esize == 4) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            float r;
            if (is_rsqrt) {
              r = (f <= 0.0f || f != f) ? __builtin_nanf("") : 1.0f / __builtin_sqrtf(f);
            } else {
              r = (f == 0.0f) ? __builtin_inff() * (1.0f / f) : 1.0f / f;
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
          } else {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            double r;
            if (is_rsqrt) {
              r = (d <= 0.0 || d != d) ? __builtin_nan("") : 1.0 / __builtin_sqrt(d);
            } else {
              r = (d == 0.0) ? __builtin_inf() * (1.0 / d) : 1.0 / d;
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
          }
        }
        break;
      }
      // endregion

      // region digitalis - SADDLV/UADDLV: add-long across vector. Sum all
      // lanes of Vn into a single 2x-width scalar result written to bottom
      // of Vd; upper bits cleared. Observed as `uaddlv h0, v0.8b` (insn
      // 0x2e303800) in WhatsApp's libar-bundle3.so JNI_OnLoad path.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSaddlv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUaddlv: {
        if (esize >= 8) { Undefined(); return; }  // max input element 32-bit
        bool is_signed = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSaddlv);
        uint8_t bits = esize * 8;
        uint8_t out_esize = esize * 2;
        __int128_t acc = 0;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          if (is_signed) {
            int64_t s = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
            acc += s;
          } else {
            acc += elem;
          }
        }
        uint64_t r = static_cast<uint64_t>(acc) & ElementMask(out_esize);
        result = 0;
        memcpy(reinterpret_cast<uint8_t*>(&result), &r, out_esize);
        break;
      }
      // endregion

      // region digitalis - SUQADD / USQADD: per-lane saturating accumulate.
      //  SUQADD Vd, Vn: Vd[i] = sat_signed( (int)Vd[i] + (uint)Vn[i] )
      //  USQADD Vd, Vn: Vd[i] = sat_unsigned( (uint)Vd[i] + (int)Vn[i] )
      // Per-lane element widths: 1/2/4/8 bytes. Observed as `usqadd v0.8b,
      // v0.8b` (insn 0x2e303800) in WhatsApp's libar-bundle3.so JNI_OnLoad.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSuqadd:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUsqadd: {
        bool is_unsigned_sat =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUsqadd);
        uint8_t bits = esize * 8;
        // Per-element signed/unsigned ranges. For esize=8 the unsigned max is
        // 2^64-1 which overflows int64, so compute carefully via __int128.
        __uint128_t dst_vec = state_->cpu.v[args.rd];
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t src_elem = 0, dst_elem = 0;
          memcpy(&src_elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          memcpy(&dst_elem, reinterpret_cast<const uint8_t*>(&dst_vec) + i * esize, esize);
          __int128_t sum;
          if (is_unsigned_sat) {
            // Vd unsigned, Vn signed.
            __int128_t s_src =
                static_cast<__int128_t>(static_cast<int64_t>(src_elem << (64 - bits)) >> (64 - bits));
            sum = static_cast<__int128_t>(dst_elem) + s_src;
            __int128_t max_u = (bits >= 64) ? ((static_cast<__int128_t>(1) << 64) - 1)
                                            : ((static_cast<__int128_t>(1) << bits) - 1);
            if (sum < 0) sum = 0;
            else if (sum > max_u) sum = max_u;
          } else {
            // Vd signed, Vn unsigned.
            __int128_t s_dst =
                static_cast<__int128_t>(static_cast<int64_t>(dst_elem << (64 - bits)) >> (64 - bits));
            sum = s_dst + static_cast<__int128_t>(src_elem);
            __int128_t max_s = (static_cast<__int128_t>(1) << (bits - 1)) - 1;
            __int128_t min_s = -(static_cast<__int128_t>(1) << (bits - 1));
            if (sum > max_s) sum = max_s;
            else if (sum < min_s) sum = min_s;
          }
          uint64_t r = static_cast<uint64_t>(sum) & ElementMask(esize);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }
      // endregion

      // region digitalis - pairwise add long instructions
      case Decoder::AdvSimdTwoRegMiscOpcode::kSaddlp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUaddlp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSadalp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUadalp: {
        // {S,U}ADDLP: Add Long Pairwise - add pairs of adjacent elements
        // producing wider results. {S,U}ADALP: same but accumulate into dst.
        if (esize > 4) { Undefined(); return; }  // max input is 32-bit
        uint8_t out_esize = esize * 2;  // output elements are twice as wide
        uint8_t num_pairs = vec_len / (esize * 2);
        bool is_signed = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSaddlp ||
                          args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSadalp);
        bool is_accum = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSadalp ||
                         args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUadalp);
        if (is_accum) {
          result = state_->cpu.v[args.rd];
        }
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_pairs; i++) {
          uint64_t a = 0, b = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src) + i * 2 * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src) + i * 2 * esize + esize, esize);
          uint64_t sum;
          if (is_signed) {
            int64_t sa = static_cast<int64_t>(a << (64 - bits)) >> (64 - bits);
            int64_t sb = static_cast<int64_t>(b << (64 - bits)) >> (64 - bits);
            sum = static_cast<uint64_t>(sa + sb) & ElementMask(out_esize);
          } else {
            sum = (a + b) & ElementMask(out_esize);
          }
          if (is_accum) {
            uint64_t existing = 0;
            memcpy(&existing, reinterpret_cast<const uint8_t*>(&result) + i * out_esize, out_esize);
            sum = (existing + sum) & ElementMask(out_esize);
          }
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * out_esize, &sum, out_esize);
        }
        break;
      }
      // endregion

      // region digitalis - saturating extract narrow: UQXTN / SQXTN.
      // Source element size is 2*esize, destination is esize.
      // SQXTN: signed saturate source to [INT_min(esize), INT_max(esize)],
      //        write low esize bytes per element.
      // UQXTN: unsigned saturate source to [0, UINT_max(esize)] (or signed
      //        source clamped to [0, UINT_max] if negative -> 0).
      // Per ARM ARM, UQXTN reads UNSIGNED src and saturates to unsigned dest.
      // Q=0: low half of dest vector (upper zeroed),
      // Q=1: upper half (lower half preserved) — XTN2 form.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqxtn:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUqxtn: {
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_count = 16 / src_esize;
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        bool is_signed = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtn);
        uint64_t dst_emask = ElementMask(esize);
        uint64_t dst_smax = dst_emask >> 1;                // e.g. 0x7F  for esize=1
        uint64_t dst_smin_bits = (dst_emask ^ dst_smax);   // e.g. 0x80  for esize=1
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t raw = 0;
          memcpy(&raw, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          uint64_t out;
          if (is_signed) {
            // Sign-extend src to int64
            int64_t s = static_cast<int64_t>(raw << (64 - src_esize * 8)) >> (64 - src_esize * 8);
            int64_t smax = static_cast<int64_t>(dst_smax);
            int64_t smin = -smax - 1;
            if (s > smax) s = smax;
            if (s < smin) s = smin;
            out = static_cast<uint64_t>(s) & dst_emask;
          } else {
            // Unsigned saturate to dst_emask.
            out = (raw > dst_emask) ? dst_emask : raw;
          }
          (void)dst_smin_bits;  // silence unused warning when only used in is_signed branch above
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &out, esize);
        }
        break;
      }
      // endregion

      // region digitalis - floating-point convert long / narrow.
      // FCVTL: widen narrow FP source to wide FP destination.
      //   size=01 (sz=0): f32 -> f64, narrow lane count=2, wide count=2.
      //   Q=0 reads narrow elems from low half of Vn; Q=1 reads from high half.
      //   Result occupies full destination vector.
      // FCVTN: narrow wide FP source to narrow FP destination.
      //   size=01 (sz=0): f64 -> f32.
      //   Q=0 writes narrow elems into low half of Vd (upper zeroed);
      //   Q=1 writes into high half (lower preserved).
      // size=00 (half-precision) is not implemented yet.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtl: {
        if (args.size != 0b01) { Undefined(); return; }
        uint8_t src_off = args.q ? 8 : 0;
        for (uint8_t i = 0; i < 2; i++) {
          float f;
          memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + src_off + i * 4, 4);
          double d = static_cast<double>(f);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &d, 8);
        }
        break;
      }
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtn: {
        if (args.size != 0b01) { Undefined(); return; }
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_off = args.q ? 8 : 0;
        for (uint8_t i = 0; i < 2; i++) {
          double d;
          memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
          float f = static_cast<float>(d);
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_off + i * 4, &f, 4);
        }
        break;
      }
      // endregion

      // Less critical ops: leave as undefined for now.
      case Decoder::AdvSimdTwoRegMiscOpcode::kCls:
      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = result;
  }

  //
  // AdvSIMD shift by immediate: SSHR, USHR, SHL, SSRA, USRA, SLI, SRI, SHRN, SSHLL, USHLL.
  //
  // region digitalis
  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t result = state_->cpu.v[args.rd];

    if (args.size == 0b10) {
      // 32-bit float elements.
      uint8_t num_elements = args.q ? 4 : 2;
      float indexed;
      memcpy(&indexed, reinterpret_cast<const uint8_t*>(&src_m) + args.index * 4, 4);

      for (uint8_t i = 0; i < num_elements; i++) {
        float src;
        memcpy(&src, reinterpret_cast<const uint8_t*>(&src_n) + i * 4, 4);

        switch (args.opcode) {
          case Decoder::AdvSimdVecXIdxOpcode::kFmla: {
            float dst;
            memcpy(&dst, reinterpret_cast<uint8_t*>(&result) + i * 4, 4);
            float r = dst + src * indexed;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmls: {
            float dst;
            memcpy(&dst, reinterpret_cast<uint8_t*>(&result) + i * 4, 4);
            float r = dst - src * indexed;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmul: {
            float r = src * indexed;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
            break;
          }
          default:
            Undefined();
            return;
        }
      }
    } else if (args.size == 0b11) {
      // 64-bit double elements.
      uint8_t num_elements = args.q ? 2 : 1;
      double indexed;
      memcpy(&indexed, reinterpret_cast<const uint8_t*>(&src_m) + args.index * 8, 8);

      for (uint8_t i = 0; i < num_elements; i++) {
        double src;
        memcpy(&src, reinterpret_cast<const uint8_t*>(&src_n) + i * 8, 8);

        switch (args.opcode) {
          case Decoder::AdvSimdVecXIdxOpcode::kFmla: {
            double dst;
            memcpy(&dst, reinterpret_cast<uint8_t*>(&result) + i * 8, 8);
            double r = dst + src * indexed;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmls: {
            double dst;
            memcpy(&dst, reinterpret_cast<uint8_t*>(&result) + i * 8, 8);
            double r = dst - src * indexed;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmul: {
            double r = src * indexed;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
            break;
          }
          default:
            Undefined();
            return;
        }
      }
    } else {
      // Integer element sizes for MUL/MLA/MLS.
      uint8_t esize = (args.size == 0b01) ? 2 : 4;
      uint8_t num_elements = (args.q ? 16 : 8) / esize;
      uint64_t emask = ElementMask(esize);

      uint64_t indexed = 0;
      memcpy(&indexed, reinterpret_cast<const uint8_t*>(&src_m) + args.index * esize, esize);

      for (uint8_t i = 0; i < num_elements; i++) {
        uint64_t src = 0;
        memcpy(&src, reinterpret_cast<const uint8_t*>(&src_n) + i * esize, esize);

        switch (args.opcode) {
          case Decoder::AdvSimdVecXIdxOpcode::kMul: {
            uint64_t r = (src * indexed) & emask;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kMla: {
            uint64_t dst = 0;
            memcpy(&dst, reinterpret_cast<uint8_t*>(&result) + i * esize, esize);
            uint64_t r = (dst + src * indexed) & emask;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kMls: {
            uint64_t dst = 0;
            memcpy(&dst, reinterpret_cast<uint8_t*>(&result) + i * esize, esize);
            uint64_t r = (dst - src * indexed) & emask;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
            break;
          }
          default:
            Undefined();
            return;
        }
      }
    }

    state_->cpu.v[args.rd] = result;
  }
  // endregion

  void AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src = state_->cpu.v[args.rn];
    __uint128_t result = 0;

    // Determine element size from immh:
    //   immh=0001 -> 8-bit  (esize=1)
    //   immh=001x -> 16-bit (esize=2)
    //   immh=01xx -> 32-bit (esize=4)
    //   immh=1xxx -> 64-bit (esize=8)
    uint8_t esize;
    uint8_t immh = args.immh;
    if (immh & 0b1000) {
      esize = 8;
    } else if (immh & 0b0100) {
      esize = 4;
    } else if (immh & 0b0010) {
      esize = 2;
    } else {
      esize = 1;
    }

    uint8_t bits = esize * 8;
    uint8_t shift = ((immh << 3) | args.immb) - bits;  // for left shifts: shift = (immh:immb) - esize*8
    uint8_t rshift = (2 * bits) - ((immh << 3) | args.immb);  // for right shifts: shift = 2*esize*8 - (immh:immb)

    uint8_t vec_len = args.q ? 16 : 8;
    uint8_t num_elements = vec_len / esize;
    uint64_t emask = ElementMask(esize);

    switch (args.opcode) {
      case Decoder::AdvSimdShiftImmOpcode::kSshr: {
        // SSHR: signed shift right by immediate.
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          int64_t shifted = (rshift >= bits) ? (signed_val >> (bits - 1)) : (signed_val >> rshift);
          uint64_t r = static_cast<uint64_t>(shifted) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kUshr: {
        // USHR: unsigned shift right by immediate.
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          uint64_t r = (rshift >= bits) ? 0 : ((elem >> rshift) & emask);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kShl: {
        // SHL: shift left by immediate.
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          uint64_t r = (shift >= bits) ? 0 : ((elem << shift) & emask);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSsra: {
        // SSRA: signed shift right and accumulate.
        __uint128_t dst = state_->cpu.v[args.rd];
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0, dst_elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          memcpy(&dst_elem, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          int64_t shifted = (rshift >= bits) ? (signed_val >> (bits - 1)) : (signed_val >> rshift);
          uint64_t r = (dst_elem + static_cast<uint64_t>(shifted)) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kUsra: {
        // USRA: unsigned shift right and accumulate.
        __uint128_t dst = state_->cpu.v[args.rd];
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0, dst_elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          memcpy(&dst_elem, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          uint64_t shifted = (rshift >= bits) ? 0 : (elem >> rshift);
          uint64_t r = (dst_elem + shifted) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSli: {
        // SLI: shift left and insert — keep destination bits not written by shift.
        __uint128_t dst = state_->cpu.v[args.rd];
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0, dst_elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          memcpy(&dst_elem, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          uint64_t shifted = (shift >= bits) ? 0 : (elem << shift);
          // Bits [shift-1:0] come from dst, bits [bits-1:shift] come from shifted source.
          uint64_t mask = (shift >= bits) ? emask : (emask << shift) & emask;
          uint64_t r = (dst_elem & ~mask) | (shifted & mask);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSri: {
        // SRI: shift right and insert — keep destination bits not written by shift.
        __uint128_t dst = state_->cpu.v[args.rd];
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0, dst_elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          memcpy(&dst_elem, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          uint64_t shifted = (rshift >= bits) ? 0 : (elem >> rshift);
          // Bits [bits-1:bits-shift] come from dst, lower bits come from shifted source.
          uint64_t mask = (rshift >= bits) ? 0 : (emask >> rshift);
          uint64_t r = (dst_elem & ~mask) | (shifted & mask);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kShrn: {
        // SHRN: shift right narrow — narrows each element by half width.
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        uint8_t src_count = 16 / src_esize;  // always from full 128-bit source
        uint8_t narrow_rshift = src_bits - ((immh << 3) | args.immb);
        // Q=0: write lower half, Q=1: write upper half (SHRN2).
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        uint64_t narrow_mask = ElementMask(esize);
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          uint64_t shifted = (narrow_rshift >= src_bits) ? 0 : (elem >> narrow_rshift);
          uint64_t r = shifted & narrow_mask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSshll: {
        // SSHLL: signed shift left long — widen each element, then shift left.
        // Source: Q=0 lower half, Q=1 upper half. Result: full 128-bit.
        uint8_t src_esize = esize;  // source element size
        uint8_t dst_esize = esize * 2;  // destination element size
        if (dst_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        uint8_t src_count = 8 / src_esize;  // elements in 64-bit half
        uint8_t src_offset = args.q ? 8 : 0;
        uint64_t dst_mask = ElementMask(dst_esize);
        // For SSHLL, shift = (immh:immb) - source_element_bits
        uint8_t shl_amount = ((immh << 3) | args.immb) - src_bits;
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + src_offset + i * src_esize, src_esize);
          // Sign-extend.
          int64_t signed_val = static_cast<int64_t>(elem << (64 - src_bits)) >> (64 - src_bits);
          uint64_t r = (static_cast<uint64_t>(signed_val) << shl_amount) & dst_mask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * dst_esize, &r, dst_esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kUshll: {
        // USHLL: unsigned shift left long.
        uint8_t src_esize = esize;
        uint8_t dst_esize = esize * 2;
        if (dst_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        uint8_t src_count = 8 / src_esize;
        uint8_t src_offset = args.q ? 8 : 0;
        uint64_t dst_mask = ElementMask(dst_esize);
        uint8_t shl_amount = ((immh << 3) | args.immb) - src_bits;
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + src_offset + i * src_esize, src_esize);
          uint64_t r = (elem << shl_amount) & dst_mask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * dst_esize, &r, dst_esize);
        }
        break;
      }

      // region digitalis
      case Decoder::AdvSimdShiftImmOpcode::kSrshr: {
        // SRSHR: signed rounding shift right.
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          if (rshift >= bits) {
            // When shift equals element width, rounding bit is the MSB.
            int64_t r = (signed_val < 0) ? -1 : 0;
            // Rounding: add 1 if the bit shifted out at position (bits-1) is 1.
            // For shift == bits, result is 0 or -1 based on sign, then +round.
            int64_t round_bit = (signed_val >> (bits - 1)) & 1;
            int64_t shifted = r + round_bit;
            uint64_t ru = static_cast<uint64_t>(shifted) & emask;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &ru, esize);
          } else {
            __int128_t wide = static_cast<__int128_t>(signed_val) + (1LL << (rshift - 1));
            int64_t shifted = static_cast<int64_t>(wide >> rshift);
            uint64_t r = static_cast<uint64_t>(shifted) & emask;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
          }
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kUrshr: {
        // URSHR: unsigned rounding shift right.
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          if (rshift >= bits) {
            // Rounding bit is the MSB of the element.
            uint64_t r = (elem >> (bits - 1)) & 1;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
          } else {
            __uint128_t wide = static_cast<__uint128_t>(elem) + (1ULL << (rshift - 1));
            uint64_t r = static_cast<uint64_t>(wide >> rshift) & emask;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
          }
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSrsra: {
        // SRSRA: signed rounding shift right and accumulate.
        __uint128_t dst = state_->cpu.v[args.rd];
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0, dst_elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          memcpy(&dst_elem, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          int64_t shifted;
          if (rshift >= bits) {
            shifted = (signed_val < 0) ? -1 : 0;
            shifted += (signed_val >> (bits - 1)) & 1;
          } else {
            __int128_t wide = static_cast<__int128_t>(signed_val) + (1LL << (rshift - 1));
            shifted = static_cast<int64_t>(wide >> rshift);
          }
          uint64_t r = (dst_elem + static_cast<uint64_t>(shifted)) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kUrsra: {
        // URSRA: unsigned rounding shift right and accumulate.
        __uint128_t dst = state_->cpu.v[args.rd];
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0, dst_elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          memcpy(&dst_elem, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          uint64_t shifted;
          if (rshift >= bits) {
            shifted = (elem >> (bits - 1)) & 1;
          } else {
            __uint128_t wide = static_cast<__uint128_t>(elem) + (1ULL << (rshift - 1));
            shifted = static_cast<uint64_t>(wide >> rshift);
          }
          uint64_t r = (dst_elem + shifted) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSqshl: {
        // SQSHL (immediate): signed saturating shift left.
        int64_t signed_max = (bits == 64) ? INT64_MAX : ((1LL << (bits - 1)) - 1);
        int64_t signed_min = (bits == 64) ? INT64_MIN : -(1LL << (bits - 1));
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          __int128_t wide = static_cast<__int128_t>(signed_val) << shift;
          int64_t clamped;
          if (wide > signed_max) clamped = signed_max;
          else if (wide < signed_min) clamped = signed_min;
          else clamped = static_cast<int64_t>(wide);
          uint64_t r = static_cast<uint64_t>(clamped) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kUqshl: {
        // UQSHL (immediate): unsigned saturating shift left.
        uint64_t umax = (bits == 64) ? UINT64_MAX : ((1ULL << bits) - 1);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          __uint128_t wide = static_cast<__uint128_t>(elem) << shift;
          uint64_t r = (wide > umax) ? umax : static_cast<uint64_t>(wide);
          r &= emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSqshlu: {
        // SQSHLU: signed saturating shift left, unsigned result.
        uint64_t umax = (bits == 64) ? UINT64_MAX : ((1ULL << bits) - 1);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - bits)) >> (64 - bits);
          uint64_t r;
          if (signed_val < 0) {
            r = 0;
          } else {
            __uint128_t wide = static_cast<__uint128_t>(signed_val) << shift;
            r = (wide > umax) ? umax : static_cast<uint64_t>(wide);
          }
          r &= emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kRshrn: {
        // RSHRN: rounding shift right narrow.
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        uint8_t src_count = 16 / src_esize;
        uint8_t narrow_rshift = src_bits - ((immh << 3) | args.immb);
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        uint64_t narrow_mask = ElementMask(esize);
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          uint64_t shifted;
          if (narrow_rshift >= src_bits) {
            shifted = (elem >> (src_bits - 1)) & 1;
          } else if (narrow_rshift == 0) {
            shifted = elem;
          } else {
            __uint128_t wide = static_cast<__uint128_t>(elem) + (1ULL << (narrow_rshift - 1));
            shifted = static_cast<uint64_t>(wide >> narrow_rshift);
          }
          uint64_t r = shifted & narrow_mask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSqshrn: {
        // SQSHRN: signed saturating shift right narrow.
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        uint8_t src_count = 16 / src_esize;
        uint8_t narrow_rshift = src_bits - ((immh << 3) | args.immb);
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        uint8_t dst_bits = esize * 8;
        int64_t sat_max = (1LL << (dst_bits - 1)) - 1;
        int64_t sat_min = -(1LL << (dst_bits - 1));
        uint64_t narrow_mask = ElementMask(esize);
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          int64_t signed_val = static_cast<int64_t>(elem << (64 - src_bits)) >> (64 - src_bits);
          int64_t shifted = (narrow_rshift >= src_bits)
                                ? (signed_val >> (src_bits - 1))
                                : (signed_val >> narrow_rshift);
          if (shifted > sat_max) shifted = sat_max;
          else if (shifted < sat_min) shifted = sat_min;
          uint64_t r = static_cast<uint64_t>(shifted) & narrow_mask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kUqshrn: {
        // UQSHRN: unsigned saturating shift right narrow.
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        uint8_t src_count = 16 / src_esize;
        uint8_t narrow_rshift = src_bits - ((immh << 3) | args.immb);
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        uint64_t sat_max = ElementMask(esize);  // (1 << dst_bits) - 1
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          uint64_t shifted = (narrow_rshift >= src_bits) ? 0 : (elem >> narrow_rshift);
          if (shifted > sat_max) shifted = sat_max;
          uint64_t r = shifted & sat_max;
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &r, esize);
        }
        break;
      }
      // endregion

      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = result;
  }
  // endregion

  //
  // Guest state getters/setters.
  //

  Register GetReg(uint8_t reg) const {
    CHECK(reg < 31);
    return state_->cpu.x[reg];
  }

  void SetReg(uint8_t reg, Register value) {
    if (exception_raised_) {
      return;
    }
    CHECK(reg < 31);
    state_->cpu.x[reg] = value;
  }

  Register GetSp() const {
    return state_->cpu.sp;
  }

  void SetSp(Register value) {
    if (exception_raised_) {
      return;
    }
    state_->cpu.sp = value;
  }

  [[nodiscard]] uint64_t GetImm(uint64_t imm) const { return imm; }

  [[nodiscard]] Register Copy(Register value) const { return value; }

  [[nodiscard]] GuestAddr GetInsnAddr() const { return state_->cpu.insn_addr; }

  void FinalizeInsn(uint8_t insn_len) {
    if (!branch_taken_ && !exception_raised_) {
      state_->cpu.insn_addr += insn_len;
    }
  }

 private:
  // region digitalis
  // Compute element mask: all-ones for element of esize bytes.
  // Avoids UB from (1ULL << 64) when esize == 8.
  static uint64_t ElementMask(uint8_t esize) {
    return (esize >= 8) ? ~0ULL : ((1ULL << (esize * 8)) - 1);
  }

  // Helper: apply an unsigned element-wise operation across the vector.
  template <typename Op>
  void AdvSimdThreeSameElementWise(__uint128_t src_n, __uint128_t src_m,
                                   uint8_t esize, uint8_t num_elements,
                                   __uint128_t* result, Op op) {
    *result = 0;
    uint64_t mask = ElementMask(esize);
    for (uint8_t i = 0; i < num_elements; i++) {
      uint64_t a = 0, b = 0;
      memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * esize, esize);
      memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * esize, esize);
      uint64_t r = op(a, b, esize) & mask;
      memcpy(reinterpret_cast<uint8_t*>(result) + i * esize, &r, esize);
    }
  }

  // Helper: apply a signed element-wise operation across the vector.
  template <typename Op>
  void AdvSimdThreeSameElementWiseSigned(__uint128_t src_n, __uint128_t src_m,
                                         uint8_t esize, uint8_t num_elements,
                                         __uint128_t* result, Op op) {
    *result = 0;
    uint8_t bits = esize * 8;
    uint64_t mask = ElementMask(esize);
    for (uint8_t i = 0; i < num_elements; i++) {
      uint64_t a = 0, b = 0;
      memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * esize, esize);
      memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * esize, esize);
      // Sign-extend to int64_t.
      int64_t sa = static_cast<int64_t>(a << (64 - bits)) >> (64 - bits);
      int64_t sb = static_cast<int64_t>(b << (64 - bits)) >> (64 - bits);
      int64_t sr = op(sa, sb);
      uint64_t r = static_cast<uint64_t>(sr) & mask;
      memcpy(reinterpret_cast<uint8_t*>(result) + i * esize, &r, esize);
    }
  }
  // endregion

  //
  // Flag update helpers.
  //

  void UpdateFlags(uint64_t operand1, uint64_t operand2, uint64_t result,
                   bool is_sub, bool is_64bit) {
    uint16_t flags = 0;
    unsigned top_bit = is_64bit ? 63 : 31;

    // N flag: result is negative.
    if ((result >> top_bit) & 1) {
      flags |= CPUState::kFlagNegative;
    }

    // Z flag: result is zero.
    uint64_t mask = is_64bit ? ~0ULL : 0xFFFFFFFFULL;
    if ((result & mask) == 0) {
      flags |= CPUState::kFlagZero;
    }

    // C and V flags.
    if (is_sub) {
      // SUB/CMP: C is set if there is NO borrow (i.e., operand1 >= operand2 unsigned).
      if (is_64bit) {
        if (operand1 >= operand2) flags |= CPUState::kFlagCarry;
      } else {
        if (static_cast<uint32_t>(operand1) >= static_cast<uint32_t>(operand2))
          flags |= CPUState::kFlagCarry;
      }
      // V: signed overflow.
      if (is_64bit) {
        int64_t a = static_cast<int64_t>(operand1);
        int64_t b = static_cast<int64_t>(operand2);
        int64_t r = static_cast<int64_t>(result);
        if ((a >= 0 && b < 0 && r < 0) || (a < 0 && b >= 0 && r >= 0)) {
          flags |= CPUState::kFlagOverflow;
        }
      } else {
        int32_t a = static_cast<int32_t>(static_cast<uint32_t>(operand1));
        int32_t b = static_cast<int32_t>(static_cast<uint32_t>(operand2));
        int32_t r = static_cast<int32_t>(static_cast<uint32_t>(result));
        if ((a >= 0 && b < 0 && r < 0) || (a < 0 && b >= 0 && r >= 0)) {
          flags |= CPUState::kFlagOverflow;
        }
      }
    } else {
      // ADD/CMN: C is set if unsigned overflow occurred.
      if (is_64bit) {
        if (result < operand1) flags |= CPUState::kFlagCarry;
      } else {
        if (static_cast<uint32_t>(result) < static_cast<uint32_t>(operand1))
          flags |= CPUState::kFlagCarry;
      }
      // V: signed overflow.
      if (is_64bit) {
        int64_t a = static_cast<int64_t>(operand1);
        int64_t b = static_cast<int64_t>(operand2);
        int64_t r = static_cast<int64_t>(result);
        if ((a >= 0 && b >= 0 && r < 0) || (a < 0 && b < 0 && r >= 0)) {
          flags |= CPUState::kFlagOverflow;
        }
      } else {
        int32_t a = static_cast<int32_t>(static_cast<uint32_t>(operand1));
        int32_t b = static_cast<int32_t>(static_cast<uint32_t>(operand2));
        int32_t r = static_cast<int32_t>(static_cast<uint32_t>(result));
        if ((a >= 0 && b >= 0 && r < 0) || (a < 0 && b < 0 && r >= 0)) {
          flags |= CPUState::kFlagOverflow;
        }
      }
    }

    state_->cpu.flags = flags;
  }

  void UpdateLogicalFlags(uint64_t result, bool is_64bit) {
    uint16_t flags = 0;
    unsigned top_bit = is_64bit ? 63 : 31;

    // N flag.
    if ((result >> top_bit) & 1) {
      flags |= CPUState::kFlagNegative;
    }

    // Z flag.
    uint64_t mask = is_64bit ? ~0ULL : 0xFFFFFFFFULL;
    if ((result & mask) == 0) {
      flags |= CPUState::kFlagZero;
    }

    // C and V are always cleared for logical operations.
    state_->cpu.flags = flags;
  }

  bool EvaluateCondition(Decoder::Condition cond) const {
    uint16_t flags = state_->cpu.flags;
    bool n = flags & CPUState::kFlagNegative;
    bool z = flags & CPUState::kFlagZero;
    bool c = flags & CPUState::kFlagCarry;
    bool v = flags & CPUState::kFlagOverflow;

    switch (cond) {
      case Decoder::Condition::kEq:  return z;
      case Decoder::Condition::kNe:  return !z;
      case Decoder::Condition::kCs:  return c;
      case Decoder::Condition::kCc:  return !c;
      case Decoder::Condition::kMi:  return n;
      case Decoder::Condition::kPl:  return !n;
      case Decoder::Condition::kVs:  return v;
      case Decoder::Condition::kVc:  return !v;
      case Decoder::Condition::kHi:  return c && !z;
      case Decoder::Condition::kLs:  return !c || z;
      case Decoder::Condition::kGe:  return n == v;
      case Decoder::Condition::kLt:  return n != v;
      case Decoder::Condition::kGt:  return !z && (n == v);
      case Decoder::Condition::kLe:  return z || (n != v);
      case Decoder::Condition::kAl:  return true;
      case Decoder::Condition::kNv:  return true;
      default: return false;
    }
  }

  uint64_t ApplyShift(uint64_t value, Decoder::ShiftType shift_type,
                      uint8_t shift_amount, bool is_64bit) const {
    if (!is_64bit) {
      value &= 0xFFFFFFFFULL;
    }

    if (shift_amount == 0) {
      return value;
    }

    uint64_t result;
    switch (shift_type) {
      case Decoder::ShiftType::kLsl:
        result = value << shift_amount;
        break;
      case Decoder::ShiftType::kLsr:
        result = value >> shift_amount;
        break;
      case Decoder::ShiftType::kAsr:
        if (is_64bit) {
          result = static_cast<uint64_t>(static_cast<int64_t>(value) >> shift_amount);
        } else {
          result = static_cast<uint64_t>(static_cast<uint32_t>(
              static_cast<int32_t>(static_cast<uint32_t>(value)) >> shift_amount));
        }
        break;
      case Decoder::ShiftType::kRor:
        if (is_64bit) {
          result = (value >> shift_amount) | (value << (64 - shift_amount));
        } else {
          uint32_t v32 = static_cast<uint32_t>(value);
          result = ((v32 >> shift_amount) | (v32 << (32 - shift_amount))) & 0xFFFFFFFFULL;
        }
        break;
      default:
        result = value;
        break;
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    return result;
  }

  uint64_t ExtendReg(uint64_t value, uint8_t extend_type, uint8_t shift_amount) const {
    uint64_t extended;
    switch (extend_type) {
      case 0b000:  // UXTB
        extended = value & 0xFF;
        break;
      case 0b001:  // UXTH
        extended = value & 0xFFFF;
        break;
      case 0b010:  // UXTW
        extended = value & 0xFFFFFFFFULL;
        break;
      case 0b011:  // UXTX
        extended = value;
        break;
      case 0b100:  // SXTB
        extended = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int8_t>(
            static_cast<uint8_t>(value))));
        break;
      case 0b101:  // SXTH
        extended = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int16_t>(
            static_cast<uint16_t>(value))));
        break;
      case 0b110:  // SXTW
        extended = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(
            static_cast<uint32_t>(value))));
        break;
      case 0b111:  // SXTX
        extended = value;
        break;
      default:
        extended = value;
        break;
    }
    return extended << shift_amount;
  }

  // region digitalis
  //
  // Memory access safety helpers.
  // (CheckPageAccess and RaiseGuestSegv removed — replaced by FaultyLoad/FaultyStore
  // with HandleMemoryFault)

  //
  // Atomic helpers.
  //
  template <typename T>
  uint64_t AtomicLoad(void* addr) {
    return static_cast<uint64_t>(__atomic_load_n(static_cast<T*>(addr), __ATOMIC_ACQUIRE));
  }

  template <typename T>
  void AtomicStore(void* addr, uint64_t val) {
    __atomic_store_n(static_cast<T*>(addr), static_cast<T>(val), __ATOMIC_RELEASE);
  }

  template <typename T>
  bool AtomicCAS(void* addr, uint64_t expected, uint64_t desired) {
    T exp = static_cast<T>(expected);
    T des = static_cast<T>(desired);
    return __atomic_compare_exchange_n(static_cast<T*>(addr), &exp, des,
                                       false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  }

  template <typename T>
  uint64_t AtomicCASVal(void* addr, uint64_t expected, uint64_t desired) {
    T exp = static_cast<T>(expected);
    T des = static_cast<T>(desired);
    __atomic_compare_exchange_n(static_cast<T*>(addr), &exp, des,
                                false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return static_cast<uint64_t>(exp);  // Returns original value
  }

  template <typename T>
  uint64_t AtomicExchange(void* addr, uint64_t val) {
    return static_cast<uint64_t>(
        __atomic_exchange_n(static_cast<T*>(addr), static_cast<T>(val), __ATOMIC_SEQ_CST));
  }

  template <typename T>
  uint64_t AtomicFetchAdd(void* addr, uint64_t operand) {
    return static_cast<uint64_t>(
        __atomic_fetch_add(static_cast<T*>(addr), static_cast<T>(operand), __ATOMIC_SEQ_CST));
  }

  template <typename T>
  uint64_t AtomicFetchAndNot(void* addr, uint64_t mask) {
    return static_cast<uint64_t>(
        __atomic_fetch_and(static_cast<T*>(addr), static_cast<T>(~mask), __ATOMIC_SEQ_CST));
  }

  template <typename T>
  uint64_t AtomicFetchOr(void* addr, uint64_t bits) {
    return static_cast<uint64_t>(
        __atomic_fetch_or(static_cast<T*>(addr), static_cast<T>(bits), __ATOMIC_SEQ_CST));
  }

  template <typename T>
  uint64_t AtomicFetchXor(void* addr, uint64_t bits) {
    return static_cast<uint64_t>(
        __atomic_fetch_xor(static_cast<T*>(addr), static_cast<T>(bits), __ATOMIC_SEQ_CST));
  }

  //
  // SIMD helpers.
  //

  // region digitalis
  // Half-precision float conversion helpers (IEEE 754 binary16).
  static uint16_t FpSingleToHalf(float f) {
    uint32_t fbits;
    memcpy(&fbits, &f, 4);
    uint16_t sign = (fbits >> 16) & 0x8000;
    int32_t exp = ((fbits >> 23) & 0xFF) - 127;
    uint32_t frac = fbits & 0x7FFFFF;

    if (exp == 128) {
      // Inf or NaN.
      return sign | 0x7C00 | (frac ? (frac >> 13) | 1 : 0);
    }
    if (exp > 15) {
      // Overflow -> infinity.
      return sign | 0x7C00;
    }
    if (exp < -24) {
      // Underflow -> zero.
      return sign;
    }
    if (exp < -14) {
      // Denormalized half-precision.
      frac = (frac | 0x800000) >> (-exp - 14 + 13);
      return sign | static_cast<uint16_t>(frac);
    }
    return sign | static_cast<uint16_t>((exp + 15) << 10) | static_cast<uint16_t>(frac >> 13);
  }

  static float FpHalfToSingle(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    uint32_t fbits;
    if (exp == 0) {
      if (frac == 0) {
        fbits = sign;  // Zero.
      } else {
        // Denormalized: normalize.
        exp = 1;
        while (!(frac & 0x400)) {
          frac <<= 1;
          exp--;
        }
        frac &= 0x3FF;
        fbits = sign | (static_cast<uint32_t>(exp + 127 - 15) << 23) | (frac << 13);
      }
    } else if (exp == 0x1F) {
      // Inf or NaN.
      fbits = sign | 0x7F800000 | (frac << 13);
    } else {
      fbits = sign | (static_cast<uint32_t>(exp + 127 - 15) << 23) | (frac << 13);
    }

    float result;
    memcpy(&result, &fbits, 4);
    return result;
  }
  // endregion

  __uint128_t ExpandSimdModifiedImm(uint8_t op, uint8_t cmode, uint8_t abc, uint8_t defgh, bool q) {
    uint8_t imm8 = (abc << 5) | defgh;
    uint64_t imm64 = 0;

    if (op == 0) {
      uint8_t cmode_hi = cmode >> 1;
      switch (cmode_hi) {
        case 0b000:  // 32-bit, no shift
          imm64 = static_cast<uint64_t>(imm8) | (static_cast<uint64_t>(imm8) << 32);
          break;
        case 0b001:  // 32-bit, LSL #8
          imm64 = (static_cast<uint64_t>(imm8) << 8) | (static_cast<uint64_t>(imm8) << 40);
          break;
        case 0b010:  // 32-bit, LSL #16
          imm64 = (static_cast<uint64_t>(imm8) << 16) | (static_cast<uint64_t>(imm8) << 48);
          break;
        case 0b011:  // 32-bit, LSL #24
          imm64 = (static_cast<uint64_t>(imm8) << 24) | (static_cast<uint64_t>(imm8) << 56);
          break;
        case 0b100:  // 16-bit, no shift
          for (int i = 0; i < 4; i++) imm64 |= static_cast<uint64_t>(imm8) << (i * 16);
          break;
        case 0b101:  // 16-bit, LSL #8
          for (int i = 0; i < 4; i++) imm64 |= static_cast<uint64_t>(imm8) << (i * 16 + 8);
          break;
        case 0b110:
          if (!(cmode & 1)) {
            // 32-bit, ones then imm8
            uint32_t val = (imm8 << 8) | 0xFF;
            imm64 = static_cast<uint64_t>(val) | (static_cast<uint64_t>(val) << 32);
          } else {
            // 32-bit, ones then imm8 then ones
            uint32_t val = (imm8 << 16) | 0xFFFF;
            imm64 = static_cast<uint64_t>(val) | (static_cast<uint64_t>(val) << 32);
          }
          break;
        case 0b111:
          if (!(cmode & 1)) {
            // 8-bit, replicated
            for (int i = 0; i < 8; i++) imm64 |= static_cast<uint64_t>(imm8) << (i * 8);
          } else {
            // FMOV (immediate) - skip for now
            for (int i = 0; i < 8; i++) imm64 |= static_cast<uint64_t>(imm8) << (i * 8);
          }
          break;
      }
    } else {
      // op=1
      if (cmode == 0b1110) {
        // MOVI 64-bit: each bit of imm8 expands to a byte
        for (int i = 0; i < 8; i++) {
          if (imm8 & (1 << i)) {
            imm64 |= 0xFFULL << (i * 8);
          }
        }
      } else {
        // MVNI variants (op=1, cmode != 1110): NOT of the MOVI value
        // Reuse op=0 decode then invert
        __uint128_t base = ExpandSimdModifiedImm(0, cmode, abc, defgh, q);
        __uint128_t result = ~base;
        return result;
      }
    }

    __uint128_t result = static_cast<__uint128_t>(imm64);
    if (q) {
      result |= static_cast<__uint128_t>(imm64) << 64;
    }
    return result;
  }

  void SimdLoadFromMemory(void* host_addr, uint8_t rt, Decoder::SimdLoadStoreSize size) {
    // region digitalis - use FaultyLoad for sizes <= 8 bytes, two loads for 128-bit
    if (size == Decoder::SimdLoadStoreSize::k128bit) {
      FaultyLoadResult lo = FaultyLoad(host_addr, 8);
      if (lo.is_fault) { HandleMemoryFault(reinterpret_cast<uint64_t>(host_addr)); return; }
      FaultyLoadResult hi = FaultyLoad(static_cast<uint8_t*>(host_addr) + 8, 8);
      if (hi.is_fault) { HandleMemoryFault(reinterpret_cast<uint64_t>(host_addr) + 8); return; }
      state_->cpu.v[rt] = static_cast<__uint128_t>(lo.value) | (static_cast<__uint128_t>(hi.value) << 64);
      return;
    }
    uint8_t bytes = 0;
    switch (size) {
      case Decoder::SimdLoadStoreSize::k8bit: bytes = 1; break;
      case Decoder::SimdLoadStoreSize::k16bit: bytes = 2; break;
      case Decoder::SimdLoadStoreSize::k32bit: bytes = 4; break;
      case Decoder::SimdLoadStoreSize::k64bit: bytes = 8; break;
      default: break;
    }
    if (bytes > 0) {
      FaultyLoadResult fl = FaultyLoad(host_addr, bytes);
      if (fl.is_fault) { HandleMemoryFault(reinterpret_cast<uint64_t>(host_addr)); return; }
      state_->cpu.v[rt] = static_cast<__uint128_t>(fl.value);
      return;
    }
    // endregion
    state_->cpu.v[rt] = 0;  // zero-extend upper bits
    switch (size) {
      case Decoder::SimdLoadStoreSize::k8bit: {
        uint8_t val;
        memcpy(&val, host_addr, 1);
        state_->cpu.v[rt] = val;
        break;
      }
      case Decoder::SimdLoadStoreSize::k16bit: {
        uint16_t val;
        memcpy(&val, host_addr, 2);
        state_->cpu.v[rt] = val;
        break;
      }
      case Decoder::SimdLoadStoreSize::k32bit: {
        uint32_t val;
        memcpy(&val, host_addr, 4);
        state_->cpu.v[rt] = val;
        break;
      }
      case Decoder::SimdLoadStoreSize::k64bit: {
        uint64_t val;
        memcpy(&val, host_addr, 8);
        state_->cpu.v[rt] = val;
        break;
      }
      case Decoder::SimdLoadStoreSize::k128bit: {
        memcpy(&state_->cpu.v[rt], host_addr, 16);
        break;
      }
    }
  }

  void SimdStoreToMemory(void* host_addr, uint8_t rt, Decoder::SimdLoadStoreSize size) {
    // region digitalis - use FaultyStore for fault recovery
    uint64_t lo_val = static_cast<uint64_t>(state_->cpu.v[rt]);
    uint64_t hi_val = static_cast<uint64_t>(state_->cpu.v[rt] >> 64);
    uint8_t bytes = 0;
    switch (size) {
      case Decoder::SimdLoadStoreSize::k8bit: bytes = 1; break;
      case Decoder::SimdLoadStoreSize::k16bit: bytes = 2; break;
      case Decoder::SimdLoadStoreSize::k32bit: bytes = 4; break;
      case Decoder::SimdLoadStoreSize::k64bit: bytes = 8; break;
      case Decoder::SimdLoadStoreSize::k128bit: {
        if (FaultyStore(host_addr, 8, lo_val)) {
          HandleMemoryFault(reinterpret_cast<uint64_t>(host_addr)); return;
        }
        if (FaultyStore(static_cast<uint8_t*>(host_addr) + 8, 8, hi_val)) {
          HandleMemoryFault(reinterpret_cast<uint64_t>(host_addr) + 8); return;
        }
        return;
      }
    }
    if (bytes > 0) {
      if (FaultyStore(host_addr, bytes, lo_val)) {
        HandleMemoryFault(reinterpret_cast<uint64_t>(host_addr)); return;
      }
    }
    // endregion
  }
  // endregion

  ThreadState* state_;
  bool branch_taken_;
  bool exception_raised_;
  // region digitalis
  // (removed: page cache variables no longer needed with FaultyLoad/FaultyStore)
  // endregion
};

}  // namespace berberis
// endregion
