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

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>

#include "../faulty_memory_accesses.h"

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/decoder/arm64/decoder.h"
#include "berberis/decoder/arm64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/kernel_api/run_guest_syscall.h"
#include "berberis/runtime_primitives/interpret_helpers.h"
#include "berberis/runtime_primitives/runtime_library.h"

namespace berberis {

class Interpreter {
 public:
  using Decoder = Decoder<SemanticsPlayer<Interpreter>>;
  using Register = uint64_t;
  static constexpr Register no_register = 0;

  explicit Interpreter(ThreadState* state)
      : state_(state), branch_taken_(false), exception_raised_(false) {}

  // Reset per-instruction state for batch reuse — avoids reconstructing
  // the Interpreter object for every instruction in the batch.
  void Reset() {
    branch_taken_ = false;
    exception_raised_ = false;
  }

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

  // Top-byte-ignore. ARM64 ignores bits [63:56] of an address on access, and
  // Android's allocator uses them: Scudo tags heap pointers (e.g. `orr xN, xM,
  // #0x200000000000000`) and dereferences the tagged value directly. x86_64
  // has no equivalent — such an address is not even canonical — so the tag has
  // to be stripped before the host touches memory.
  //
  // Mirrors LiteTranslator::ApplyTbi (`shl 8; shr 8`), which is why the JIT
  // tiers already ran tagged pointers correctly while the interpreter faulted
  // on the first Scudo allocation of any statically-linked binary.
  //
  // Applied at address *use*, never to the register value: the guest may
  // legitimately compute with a tagged pointer (see AddSubImmTags below), and
  // masking on write-back would corrupt the tag it expects to read back.
  static GuestAddr ApplyTbi(GuestAddr addr) { return addr & 0x00FF'FFFF'FFFF'FFFFULL; }

  // ADDG/SUBG (FEAT_MTE). The address part is Xn +/- the
  // 16-byte-scaled offset; bits[59:56] are then replaced by the logical tag
  // (start tag +/- uimm4, mod 16). Digitalis does not enforce MTE, so tag
  // exclusion (GCR_EL1) is ignored — the tag nibble is updated arithmetically,
  // keeping tagged-pointer math consistent under top-byte-ignore.
  Register AddSubImmTags(bool is_sub, Register src, uint8_t uimm6, uint8_t uimm4) {
    uint64_t offset = static_cast<uint64_t>(uimm6) << 4;
    uint64_t addr = is_sub ? (src - offset) : (src + offset);
    uint8_t start_tag = static_cast<uint8_t>((src >> 56) & 0xF);
    uint8_t new_tag = static_cast<uint8_t>(
        (is_sub ? (start_tag - uimm4) : (start_tag + uimm4)) & 0xF);
    return (addr & ~(0xFULL << 56)) | (static_cast<uint64_t>(new_tag) << 56);
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

  // LDR/LDRSW (literal): load from `[insn_addr + offset]`. The Load()
  // helper handles the FaultyLoad guard, sign/zero extension, and the
  // (is_64bit_target=true for LDRSW) widening for sign-extended loads.
  Register LoadLiteral(Decoder::LoadStoreSize size, bool is_signed, int64_t offset) {
    CHECK(!exception_raised_);
    Register target = state_->cpu.insn_addr + offset;
    bool is_64bit_target = (size == Decoder::LoadStoreSize::k64bit) || is_signed;
    return Load(size, is_signed, is_64bit_target, target, 0);
  }

  Register Bitfield(Decoder::BitfieldOpcode opcode, bool is_64bit,
                    Register dst_val, Register src, uint8_t immr, uint8_t imms) {
    CHECK(!exception_raised_);
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
    void* ptr = ToHostAddr<void>(ApplyTbi(base + offset));
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
    void* ptr = ToHostAddr<void>(ApplyTbi(base + offset));
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
  }

  Register AddImm(Register base, int32_t offset) {
    return base + offset;
  }

  void LoadPair(Decoder::LoadStoreSize size, Register base, int32_t offset,
                uint8_t rt1, uint8_t rt2, uint8_t scale, bool is_signed) {
    CHECK(!exception_raised_);
    void* ptr1 = ToHostAddr<void>(ApplyTbi(base + offset));
    void* ptr2 = ToHostAddr<void>(ApplyTbi(base + offset + scale));
    uint8_t data_bytes = (size == Decoder::LoadStoreSize::k64bit) ? 8 : 4;
    FaultyLoadResult fl1 = FaultyLoad(ptr1, data_bytes);
    if (fl1.is_fault) { HandleMemoryFault(base + offset); return; }
    FaultyLoadResult fl2 = FaultyLoad(ptr2, data_bytes);
    if (fl2.is_fault) { HandleMemoryFault(base + offset + scale); return; }

    uint64_t v1 = fl1.value;
    uint64_t v2 = fl2.value;
    // LDPSW: 32-bit elements sign-extended to the 64-bit target register.
    if (is_signed) {
      v1 = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v1)));
      v2 = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v2)));
    }
    if (rt1 != 31) state_->cpu.x[rt1] = v1;
    if (rt2 != 31) state_->cpu.x[rt2] = v2;
  }

  void StorePair(Decoder::LoadStoreSize size, Register base, int32_t offset,
                 Register data1, Register data2, uint8_t scale) {
    CHECK(!exception_raised_);
    void* ptr1 = ToHostAddr<void>(ApplyTbi(base + offset));
    void* ptr2 = ToHostAddr<void>(ApplyTbi(base + offset + scale));
    uint8_t data_bytes = (size == Decoder::LoadStoreSize::k64bit) ? 8 : 4;
    if (FaultyStore(ptr1, data_bytes, data1)) { HandleMemoryFault(base + offset); return; }
    if (FaultyStore(ptr2, data_bytes, data2)) { HandleMemoryFault(base + offset + scale); return; }
  }

  // Apply the correct 32->64 extension to the offset
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

  void Svc(uint16_t /*imm*/) {
    CHECK(!exception_raised_);
    // ARM64 syscall convention: syscall number in x8, args in x0-x5, return in x0.
    RunGuestSyscall(state_);
  }

  // BRK #imm: software breakpoint. Deliver a synchronous
  // SIGTRAP at the current guest PC (the BRK), then stop the interpreter batch
  // so the host signal handler routes it to the guest's SIGTRAP action. The
  // guest PC stays at the BRK (insn_addr is not advanced), matching the
  // architectural behaviour debuggers and sanitizers rely on.
  void Brk(uint16_t /*imm*/) {
    CHECK(!exception_raised_);
    BreakpointInsn(GetInsnAddr());
    exception_raised_ = true;
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
        // FPSR exception-flag mirroring: host MXCSR cumulative exception
        // bits set by any FP op (interpreter OR JIT-emitted) reflect into
        // emulated_fpsr at MRS-read time. MXCSR bits are sticky on x86
        // (just like FPSR is on ARM), so this lazy mirror is sufficient
        // for cumulative-flag semantics without per-op JIT instrumentation.
        MirrorHostMxcsrToFpsr();
        return state_->cpu.emulated_fpsr;
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
      case Decoder::SystemReg::kRndr:
      case Decoder::SystemReg::kRndrrs: {
        // RNDR / RNDRRS (FEAT_RNG): return a 64-bit random value and report
        // success. The architecture has these reads update PSTATE.NZCV — all
        // clear (0b0000) on success, so a host that always has entropy clears
        // the flags here (clearing C signals "random available" to the common
        // `mrs Xt, RNDR; b.cc <retry>` idiom). Entropy comes from the host's
        // /dev/urandom-backed random_device, seeded once per thread.
        thread_local std::mt19937_64 rng(std::random_device{}());
        state_->cpu.flags = 0;  // NZCV = 0b0000 (success)
        return static_cast<Register>(rng());
      }
      case Decoder::SystemReg::kCntfrqEl0:
        // Counter frequency: 19.2 MHz, the de-facto Android generic-timer rate.
        // CNTVCT/CNTPCT below are scaled to match, so cntvct/cntfrq = seconds.
        return 19200000ULL;
      case Decoder::SystemReg::kCntvctEl0:
      case Decoder::SystemReg::kCntpctEl0: {
        // Monotonic virtual/physical counter backed by the host steady clock,
        // expressed in 19.2 MHz ticks. __uint128_t intermediate avoids overflow.
        uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        return static_cast<Register>(
            (static_cast<__uint128_t>(ns) * 19200000ULL) / 1000000000ULL);
      }
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
        // Mirror the ARM FPCR rounding-mode and flush-to-zero bits into the
        // host x86 MXCSR so subsequent interpreted FP ops observe the guest-
        // requested rounding mode and denormal behaviour. ARM RMode at
        // FPCR[23:22] maps to MXCSR[14:13] (RC) with a bit-swap because ARM
        // encodes RP/RM = 01/10 while x86 encodes RD/RU = 01/10 (round-toward-
        // positive on ARM is the same direction as round-up on x86, but the
        // 2-bit field encoding is the inverse). ARM FZ at FPCR[24] flushes
        // both input and output denormals, so it maps to BOTH MXCSR FTZ
        // (bit 15, output flush) and MXCSR DAZ (bit 6, input flush). Other
        // FPCR fields (DN, AHP, exception enables, FZ16) have no clean x86
        // analog; exception enables are intentionally left masked so host FP
        // never raises SIGFPE. The MXCSR->FPSR cumulative-flag mirroring
        // below builds on this.
        ProgramHostMxcsrFromFpcr(static_cast<uint32_t>(value));
        break;
      case Decoder::SystemReg::kFpsr:
        state_->cpu.emulated_fpsr = static_cast<uint32_t>(value);
        // Clear host MXCSR cumulative exception
        // bits when guest writes FPSR. Without this, future MRS-reads would
        // re-merge stale MXCSR bits that the guest believed it had cleared.
        ClearHostMxcsrExceptions();
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
      // CRC32 instructions
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
      // PACGA (Armv8.3-PAuth generic PAC compute)
      // ARM ARM C7.2.179: PACGA computes a 32-bit PAC for the value in
      // src1 keyed by src2 (or SP for Rm=31), places it in Rd[63:32], and
      // zeros Rd[31:0].  Digitalis is PAC-blind: no authentication codes
      // are ever inserted by PACIA/PACIB/PACDA/PACDB, so the corresponding
      // generic PAC here is 0, and the architectural definition zeros the
      // low 32 bits, giving Rd = 0.  PACGA is X-form only (sf=1); the
      // decoder routes 32-bit attempts to Undefined() via the opcode
      // mismatch path, so we don't need an is_64bit guard here.
      case Decoder::DataProc2SrcOpcode::kPacga:
        result = 0;
        break;
      default:
        Undefined();
        return 0;
    }

    if (!is_64bit) {
      result &= 0xFFFFFFFFULL;
    }

    return result;
  }

  //
  // MTE (Memory Tagging Extension, Armv8.5-A) data-processing 2-source:
  // IRG / GMI / SUBP / SUBPS. The translator has no MTE backing, so the
  // correct behavior is to NOT SIGILL — produce the result as if the
  // tag-related fields were absent (we already TBI-strip on every guest
  // memory access, so tag bits in addresses are inert).
  //
  // SP semantics per ARM ARM:
  //   SUBP / SUBPS:  Rd∈{Xd,XZR}, Rn∈{Xn,SP},   Rm∈{Xm,SP}
  //   IRG:           Rd∈{Xd,SP},  Rn∈{Xn,SP},   Rm∈{Xm,XZR}
  //   GMI:           Rd∈{Xd,XZR}, Rn∈{Xn,SP},   Rm∈{Xm,XZR}
  void MteDataProc(const Decoder::MteDataProcArgs& args) {
    CHECK(!exception_raised_);

    auto read_x_or_sp = [&](uint8_t r) -> uint64_t {
      return (r == 31) ? state_->cpu.sp : state_->cpu.x[r];
    };
    auto read_x_or_zr = [&](uint8_t r) -> uint64_t {
      return (r == 31) ? 0 : state_->cpu.x[r];
    };

    switch (args.opcode) {
      case Decoder::MteDataProcOpcode::kSubp:
      case Decoder::MteDataProcOpcode::kSubps: {
        // SUBP[S]: 56-bit signed difference of tag-stripped pointers,
        // sign-extended to 64 bits. Tag bits are bits[59:56] (logical
        // tag) — strip with mask 0x00FFFFFFFFFFFFFFULL, then
        // sign-extend bit[55] up to bit[63].
        uint64_t a = read_x_or_sp(args.src1) & 0x00FFFFFFFFFFFFFFULL;
        uint64_t b = read_x_or_sp(args.src2) & 0x00FFFFFFFFFFFFFFULL;
        if (a & (1ULL << 55)) a |= 0xFF00000000000000ULL;
        if (b & (1ULL << 55)) b |= 0xFF00000000000000ULL;
        uint64_t result = a - b;
        if (args.opcode == Decoder::MteDataProcOpcode::kSubps) {
          UpdateFlags(a, b, result, /*is_sub=*/true, /*is_64bit=*/true);
        }
        // Rd=31 means XZR for SUBP/SUBPS — write is discarded.
        if (args.dst != 31) state_->cpu.x[args.dst] = result;
        break;
      }
      case Decoder::MteDataProcOpcode::kIrg: {
        // IRG Xd|SP, Xn|SP{, Xm}. With no MTE backing, the
        // "random tag" is the existing tag bits of Rn — i.e., act as a
        // move. The exclusion-mask in Rm is ignored.
        uint64_t result = read_x_or_sp(args.src1);
        if (args.dst == 31) {
          state_->cpu.sp = result;
        } else {
          state_->cpu.x[args.dst] = result;
        }
        break;
      }
      case Decoder::MteDataProcOpcode::kGmi: {
        // GMI Xd, Xn|SP, Xm. The architectural semantics OR a single
        // tag bit (1<<((Rn>>56)&0xF)) into the 16-bit mask in Rm.
        // Implement it for free since the math is cheap and any future
        // MTE-aware code in the guest will observe the right bit
        // pattern.
        uint64_t mask = read_x_or_zr(args.src2);
        uint8_t tag = static_cast<uint8_t>((read_x_or_sp(args.src1) >> 56) & 0xF);
        uint64_t result = mask | (1ULL << tag);
        // GMI Rd=31 means XZR.
        if (args.dst != 31) state_->cpu.x[args.dst] = result;
        break;
      }
    }
  }

  // MTE (Armv8.5-A) load/store memory tags: LDG / STG / ST2G / STZG / STZ2G.
  //
  // Without MTE backing the tags don't exist, but the data side-effects of
  // STZG (zero 16 bytes) and STZ2G (zero 32 bytes) still execute — guest
  // code relies on the zeroing for stack-frame init even when it doesn't
  // care about tags. STG / ST2G are pure tag stores and become NOPs. LDG
  // clears the tag field of Rt (loaded tag is 0 because no MTE backing).
  //
  // Addressing rules per ARM ARM C6.2.x:
  //   op2=00 (LDG only):  addr = Xn + imm, no writeback.
  //   op2=01 (post-index): addr = Xn,       writeback Xn = Xn + imm.
  //   op2=10 (offset):     addr = Xn + imm, no writeback.
  //   op2=11 (pre-index):  addr = Xn + imm, writeback Xn = Xn + imm.
  // The effective address must be granule-aligned (16 for the 'g' forms,
  // 32 for the '2g' forms); we mask defensively before zeroing so a
  // misaligned guest address can't tear a host page.
  void MteLoadStore(const Decoder::MteLoadStoreArgs& args) {
    CHECK(!exception_raised_);

    uint64_t base = (args.rn == 31) ? state_->cpu.sp : state_->cpu.x[args.rn];
    int64_t imm = static_cast<int64_t>(args.imm);
    uint64_t access_addr = (args.op2 == 0b01) ? base : (base + imm);
    bool writeback = (args.op2 == 0b01) || (args.op2 == 0b11);
    uint64_t new_base = base + imm;

    switch (args.opcode) {
      case Decoder::MteLoadStoreOpcode::kStg:
      case Decoder::MteLoadStoreOpcode::kSt2g:
      // STGM / STZGM tag-block stores: pure tag stores,
      // NOP without MTE backing. STZGM's data-zeroing block size is defined by
      // GMID_EL1 (not emulated here), so it is treated as a tag NOP rather than
      // zeroing a guessed-size region (over-zeroing would corrupt memory).
      case Decoder::MteLoadStoreOpcode::kStgm:
      case Decoder::MteLoadStoreOpcode::kStzgm:
        // Pure tag store — NOP without MTE backing.
        break;
      case Decoder::MteLoadStoreOpcode::kLdg: {
        // LDG: load 4-bit tag from memory into Rt[59:56]. With no MTE
        // backing, the loaded tag is 0 — i.e., clear Rt[59:56].
        if (args.rt != 31) {
          state_->cpu.x[args.rt] = state_->cpu.x[args.rt] & ~0x0F00000000000000ULL;
        }
        break;
      }
      // LDGM: load tag multiple into Rt. Tags are packed as
      // 4-bit nibbles; with no MTE backing all tags read 0, so Rt = 0.
      case Decoder::MteLoadStoreOpcode::kLdgm: {
        if (args.rt != 31) {
          state_->cpu.x[args.rt] = 0;
        }
        break;
      }
      case Decoder::MteLoadStoreOpcode::kStzg: {
        // Zero the 16-byte granule containing access_addr.
        uint64_t aligned = access_addr & ~uint64_t{0x0F};
        if (FaultyStore(ToHostAddr<void>(ApplyTbi(aligned)), 8, 0)) {
          HandleMemoryFault(aligned);
          return;
        }
        if (FaultyStore(ToHostAddr<void>(ApplyTbi(aligned + 8)), 8, 0)) {
          HandleMemoryFault(aligned + 8);
          return;
        }
        break;
      }
      case Decoder::MteLoadStoreOpcode::kStz2g: {
        // Zero the 32-byte granule containing access_addr.
        uint64_t aligned = access_addr & ~uint64_t{0x1F};
        for (int i = 0; i < 32; i += 8) {
          if (FaultyStore(ToHostAddr<void>(ApplyTbi(aligned + i)), 8, 0)) {
            HandleMemoryFault(aligned + i);
            return;
          }
        }
        break;
      }
    }

    if (writeback) {
      if (args.rn == 31) {
        state_->cpu.sp = new_base;
      } else {
        state_->cpu.x[args.rn] = new_base;
      }
    }
  }

  // Advanced SIMD complex floating-point (Armv8.3-FCMA): FCADD / FCMLA.
  //
  // FCADD <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<rotation>:
  //   for each complex element pair (re=lane 2i, im=lane 2i+1):
  //     rot=#90:   (Vd_re, Vd_im) = (Vn_re - Vm_im, Vn_im + Vm_re)
  //     rot=#270:  (Vd_re, Vd_im) = (Vn_re + Vm_im, Vn_im - Vm_re)
  //
  // FCMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<rotation>:
  //   accumulate-multiply with a per-rot lane-and-sign pattern (ARM ARM
  //   C7.2.84).  Each call computes ONE half of a complex multiply:
  //     rot=#0:    (Vd_re, Vd_im) += (Vn_re * Vm_re, Vn_re * Vm_im)
  //     rot=#90:   (Vd_re, Vd_im) += (Vn_im * -Vm_im, Vn_im * Vm_re)
  //     rot=#180:  (Vd_re, Vd_im) += (Vn_re * -Vm_re, Vn_re * -Vm_im)
  //     rot=#270:  (Vd_re, Vd_im) += (Vn_im * Vm_im, Vn_im * -Vm_re)
  //   A full complex multiply C += A*B is FCMLA #0 followed by FCMLA #90.
  //
  // Element size: size == 0b10 is float (4 lanes / 4S, 2 lanes / 2S); size
  // == 0b11 is double (only valid with Q=1, i.e. 2D — 2 lanes).  The
  // decoder has already rejected size 0b00 / 0b01, plus size 0b11 with Q=0.
  void AdvSimdFcma(const Decoder::FcmaArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t dst = state_->cpu.v[args.rd];
    __uint128_t result = 0;

    uint8_t vec_len = args.q ? 16 : 8;  // bytes in result vector

    // FP16 SIMD FCMA
    if (args.size == 0b01) {
      // Half-precision: 2 bytes per lane; pairs are 4 bytes each.
      // .4H (Q=0) has 2 pairs (lanes 0..3); .8H (Q=1) has 4 pairs (0..7).
      // Promote each half to binary32 via FpHalfToSingle, apply the FCMA
      // rotation table, narrow back via FpSingleToHalf — same round-trip
      // pattern as FP16 vector three-same / two-reg-misc.
      uint8_t pairs = (vec_len / 2) / 2;
      for (uint8_t p = 0; p < pairs; p++) {
        uint8_t lane_re = 2 * p;
        uint8_t lane_im = 2 * p + 1;
        uint16_t hn_re, hn_im, hm_re, hm_im, hd_re, hd_im;
        memcpy(&hn_re, reinterpret_cast<const uint8_t*>(&src_n) + lane_re * 2, 2);
        memcpy(&hn_im, reinterpret_cast<const uint8_t*>(&src_n) + lane_im * 2, 2);
        memcpy(&hm_re, reinterpret_cast<const uint8_t*>(&src_m) + lane_re * 2, 2);
        memcpy(&hm_im, reinterpret_cast<const uint8_t*>(&src_m) + lane_im * 2, 2);
        memcpy(&hd_re, reinterpret_cast<const uint8_t*>(&dst) + lane_re * 2, 2);
        memcpy(&hd_im, reinterpret_cast<const uint8_t*>(&dst) + lane_im * 2, 2);
        float n_re = FpHalfToSingle(hn_re);
        float n_im = FpHalfToSingle(hn_im);
        float m_re = FpHalfToSingle(hm_re);
        float m_im = FpHalfToSingle(hm_im);
        float d_re = FpHalfToSingle(hd_re);
        float d_im = FpHalfToSingle(hd_im);
        float r_re, r_im;
        if (args.opcode == Decoder::FcmaOpcode::kFcadd) {
          if (args.rot == 0) {           // rot=#90
            r_re = n_re - m_im;
            r_im = n_im + m_re;
          } else {                       // rot=#270
            r_re = n_re + m_im;
            r_im = n_im - m_re;
          }
        } else {                         // FCMLA
          switch (args.rot) {
            case 0:  r_re = d_re + n_re * m_re;  r_im = d_im + n_re * m_im;  break;
            case 1:  r_re = d_re + n_im * -m_im; r_im = d_im + n_im * m_re;  break;
            case 2:  r_re = d_re + n_re * -m_re; r_im = d_im + n_re * -m_im; break;
            default: r_re = d_re + n_im * m_im;  r_im = d_im + n_im * -m_re; break;
          }
        }
        uint16_t hr_re = FpSingleToHalf(r_re);
        uint16_t hr_im = FpSingleToHalf(r_im);
        memcpy(reinterpret_cast<uint8_t*>(&result) + lane_re * 2, &hr_re, 2);
        memcpy(reinterpret_cast<uint8_t*>(&result) + lane_im * 2, &hr_im, 2);
      }
    } else if (args.size == 0b10) {
      // Single-precision: 4 bytes per lane; pairs are 8 bytes each.
      // 2S has 1 pair (lanes 0,1); 4S has 2 pairs (lanes 0..3).
      uint8_t pairs = (vec_len / 4) / 2;
      for (uint8_t p = 0; p < pairs; p++) {
        uint8_t lane_re = 2 * p;
        uint8_t lane_im = 2 * p + 1;
        float n_re, n_im, m_re, m_im, d_re, d_im;
        memcpy(&n_re, reinterpret_cast<const uint8_t*>(&src_n) + lane_re * 4, 4);
        memcpy(&n_im, reinterpret_cast<const uint8_t*>(&src_n) + lane_im * 4, 4);
        memcpy(&m_re, reinterpret_cast<const uint8_t*>(&src_m) + lane_re * 4, 4);
        memcpy(&m_im, reinterpret_cast<const uint8_t*>(&src_m) + lane_im * 4, 4);
        memcpy(&d_re, reinterpret_cast<const uint8_t*>(&dst) + lane_re * 4, 4);
        memcpy(&d_im, reinterpret_cast<const uint8_t*>(&dst) + lane_im * 4, 4);
        float r_re, r_im;
        if (args.opcode == Decoder::FcmaOpcode::kFcadd) {
          if (args.rot == 0) {           // rot=#90
            r_re = n_re - m_im;
            r_im = n_im + m_re;
          } else {                       // rot=#270
            r_re = n_re + m_im;
            r_im = n_im - m_re;
          }
        } else {                         // FCMLA
          switch (args.rot) {
            case 0:  r_re = d_re + n_re * m_re;  r_im = d_im + n_re * m_im;  break;
            case 1:  r_re = d_re + n_im * -m_im; r_im = d_im + n_im * m_re;  break;
            case 2:  r_re = d_re + n_re * -m_re; r_im = d_im + n_re * -m_im; break;
            default: r_re = d_re + n_im * m_im;  r_im = d_im + n_im * -m_re; break;
          }
        }
        memcpy(reinterpret_cast<uint8_t*>(&result) + lane_re * 4, &r_re, 4);
        memcpy(reinterpret_cast<uint8_t*>(&result) + lane_im * 4, &r_im, 4);
      }
    } else {
      // size == 0b11: double-precision; 8 bytes per lane.
      // 2D is the only valid form (Q=1, 1 pair: lanes 0,1).
      double n_re, n_im, m_re, m_im, d_re, d_im;
      memcpy(&n_re, reinterpret_cast<const uint8_t*>(&src_n) + 0,  8);
      memcpy(&n_im, reinterpret_cast<const uint8_t*>(&src_n) + 8,  8);
      memcpy(&m_re, reinterpret_cast<const uint8_t*>(&src_m) + 0,  8);
      memcpy(&m_im, reinterpret_cast<const uint8_t*>(&src_m) + 8,  8);
      memcpy(&d_re, reinterpret_cast<const uint8_t*>(&dst) + 0,    8);
      memcpy(&d_im, reinterpret_cast<const uint8_t*>(&dst) + 8,    8);
      double r_re, r_im;
      if (args.opcode == Decoder::FcmaOpcode::kFcadd) {
        if (args.rot == 0) {
          r_re = n_re - m_im;
          r_im = n_im + m_re;
        } else {
          r_re = n_re + m_im;
          r_im = n_im - m_re;
        }
      } else {
        switch (args.rot) {
          case 0:  r_re = d_re + n_re * m_re;  r_im = d_im + n_re * m_im;  break;
          case 1:  r_re = d_re + n_im * -m_im; r_im = d_im + n_im * m_re;  break;
          case 2:  r_re = d_re + n_re * -m_re; r_im = d_im + n_re * -m_im; break;
          default: r_re = d_re + n_im * m_im;  r_im = d_im + n_im * -m_re; break;
        }
      }
      memcpy(reinterpret_cast<uint8_t*>(&result) + 0, &r_re, 8);
      memcpy(reinterpret_cast<uint8_t*>(&result) + 8, &r_im, 8);
    }

    // Q=0 zeros the upper 64 bits of the destination vector.
    ClearUpperIfNotQ(&result, args.q);
    state_->cpu.v[args.rd] = result;
  }

  // indexed FCMLA
  //
  // FCMLA (by element) — Armv8.3-FCMA.  Same per-rot rotation table as
  // FCMLA (vector), but Vm is replaced by a broadcast vector where every
  // complex pair is Vm[index].
  //
  // Per ARM ARM C7.2.86:
  //   for e = 0 to output_pairs - 1:
  //     n_pair = Vn[2*e : 2*e+1]
  //     m_pair = Vm[2*index : 2*index+1]   (single broadcast pair)
  //     apply FCMLA rotation table using n_pair and m_pair,
  //     accumulate into Vd[2*e : 2*e+1].
  //
  // FP32 only (size == 0b10) for now — FP16-indexed parked alongside
  // non-indexed FP16.
  void AdvSimdFcmaIdx(const Decoder::FcmaIdxArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t dst = state_->cpu.v[args.rd];
    __uint128_t result = 0;

    uint8_t vec_len = args.q ? 16 : 8;  // bytes in result vector.

    // FP16 SIMD FCMA indexed
    if (args.size == 0b01) {
      // Half-precision: 2 bytes per lane; pairs are 4 bytes each.
      // Read the single broadcast complex pair from Vm[index].
      uint16_t hm_re, hm_im;
      memcpy(&hm_re,
             reinterpret_cast<const uint8_t*>(&src_m) + (2 * args.index) * 2,
             2);
      memcpy(&hm_im,
             reinterpret_cast<const uint8_t*>(&src_m) +
                 (2 * args.index + 1) * 2,
             2);
      float m_re = FpHalfToSingle(hm_re);
      float m_im = FpHalfToSingle(hm_im);

      uint8_t pairs = (vec_len / 2) / 2;
      for (uint8_t p = 0; p < pairs; p++) {
        uint8_t lane_re = 2 * p;
        uint8_t lane_im = 2 * p + 1;
        uint16_t hn_re, hn_im, hd_re, hd_im;
        memcpy(&hn_re, reinterpret_cast<const uint8_t*>(&src_n) + lane_re * 2, 2);
        memcpy(&hn_im, reinterpret_cast<const uint8_t*>(&src_n) + lane_im * 2, 2);
        memcpy(&hd_re, reinterpret_cast<const uint8_t*>(&dst)   + lane_re * 2, 2);
        memcpy(&hd_im, reinterpret_cast<const uint8_t*>(&dst)   + lane_im * 2, 2);
        float n_re = FpHalfToSingle(hn_re);
        float n_im = FpHalfToSingle(hn_im);
        float d_re = FpHalfToSingle(hd_re);
        float d_im = FpHalfToSingle(hd_im);
        float r_re, r_im;
        switch (args.rot) {
          case 0:  r_re = d_re + n_re *  m_re; r_im = d_im + n_re *  m_im; break;
          case 1:  r_re = d_re + n_im * -m_im; r_im = d_im + n_im *  m_re; break;
          case 2:  r_re = d_re + n_re * -m_re; r_im = d_im + n_re * -m_im; break;
          default: r_re = d_re + n_im *  m_im; r_im = d_im + n_im * -m_re; break;
        }
        uint16_t hr_re = FpSingleToHalf(r_re);
        uint16_t hr_im = FpSingleToHalf(r_im);
        memcpy(reinterpret_cast<uint8_t*>(&result) + lane_re * 2, &hr_re, 2);
        memcpy(reinterpret_cast<uint8_t*>(&result) + lane_im * 2, &hr_im, 2);
      }
    } else if (args.size == 0b10) {
      // Single-precision: 4 bytes per lane; pairs are 8 bytes each.
      // Read the single broadcast complex pair from Vm[index].
      float m_re, m_im;
      memcpy(&m_re,
             reinterpret_cast<const uint8_t*>(&src_m) + (2 * args.index) * 4,
             4);
      memcpy(&m_im,
             reinterpret_cast<const uint8_t*>(&src_m) +
                 (2 * args.index + 1) * 4,
             4);

      uint8_t pairs = (vec_len / 4) / 2;
      for (uint8_t p = 0; p < pairs; p++) {
        uint8_t lane_re = 2 * p;
        uint8_t lane_im = 2 * p + 1;
        float n_re, n_im, d_re, d_im;
        memcpy(&n_re, reinterpret_cast<const uint8_t*>(&src_n) + lane_re * 4, 4);
        memcpy(&n_im, reinterpret_cast<const uint8_t*>(&src_n) + lane_im * 4, 4);
        memcpy(&d_re, reinterpret_cast<const uint8_t*>(&dst)   + lane_re * 4, 4);
        memcpy(&d_im, reinterpret_cast<const uint8_t*>(&dst)   + lane_im * 4, 4);
        float r_re, r_im;
        switch (args.rot) {
          case 0:  r_re = d_re + n_re *  m_re; r_im = d_im + n_re *  m_im; break;
          case 1:  r_re = d_re + n_im * -m_im; r_im = d_im + n_im *  m_re; break;
          case 2:  r_re = d_re + n_re * -m_re; r_im = d_im + n_re * -m_im; break;
          default: r_re = d_re + n_im *  m_im; r_im = d_im + n_im * -m_re; break;
        }
        memcpy(reinterpret_cast<uint8_t*>(&result) + lane_re * 4, &r_re, 4);
        memcpy(reinterpret_cast<uint8_t*>(&result) + lane_im * 4, &r_im, 4);
      }
    }

    // Q=0 zeros the upper 64 bits of the destination vector.
    ClearUpperIfNotQ(&result, args.q);
    state_->cpu.v[args.rd] = result;
  }

  // hello-dotprod
  // SDOT / UDOT (Armv8.4-DotProd), vector and by-element forms.
  //
  // Both forms accumulate a 4-byte dot product into each 32-bit destination
  // lane.  Bytes are interpreted as signed for SDOT and unsigned for UDOT.
  // The 32-bit accumulators wrap on overflow (no saturation).
  //
  // Vector form: lane i pulls bytes Vn.b[4*i .. 4*i+3] and Vm.b[4*i .. 4*i+3].
  // Indexed form: lane i pulls bytes Vn.b[4*i .. 4*i+3] but Vm broadcasts one
  // 4-byte group Vm.b[4*index .. 4*index+3] across all output lanes — index
  // selects which 4-byte slice of Vm.16B to use (index ∈ [0,3] always).
  //
  // lanes = q ? 4 : 2.  Q=0 zeros the upper 64 bits of Vd.
  void AdvSimdDotProduct(const Decoder::DotProductArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t dst = state_->cpu.v[args.rd];

    uint8_t n_b[16], m_b[16];
    memcpy(n_b, &src_n, 16);
    memcpy(m_b, &src_m, 16);

    // Per-operand signedness covers SDOT/UDOT and the I8MM mixed-sign forms
    // USDOT (Vn unsigned, Vm signed) and SUDOT (Vn signed, Vm unsigned).
    using Op = Decoder::DotProductOpcode;
    const bool n_signed = (args.opcode == Op::kSdot || args.opcode == Op::kSdotIdx ||
                           args.opcode == Op::kSudotIdx);
    const bool m_signed = (args.opcode == Op::kSdot || args.opcode == Op::kSdotIdx ||
                           args.opcode == Op::kUsdot || args.opcode == Op::kUsdotIdx);
    const bool is_indexed = (args.opcode == Op::kSdotIdx || args.opcode == Op::kUdotIdx ||
                             args.opcode == Op::kUsdotIdx || args.opcode == Op::kSudotIdx);

    __uint128_t result = 0;
    uint8_t lanes = args.q ? 4 : 2;
    for (uint8_t i = 0; i < lanes; i++) {
      int32_t acc;
      memcpy(&acc, reinterpret_cast<const uint8_t*>(&dst) + i * 4, 4);
      for (uint8_t k = 0; k < 4; k++) {
        uint8_t n_byte = n_b[4 * i + k];
        uint8_t m_byte = is_indexed ? m_b[4 * args.index + k]
                                    : m_b[4 * i + k];
        int32_t n_ext = n_signed ? static_cast<int32_t>(static_cast<int8_t>(n_byte))
                                 : static_cast<int32_t>(n_byte);
        int32_t m_ext = m_signed ? static_cast<int32_t>(static_cast<int8_t>(m_byte))
                                 : static_cast<int32_t>(m_byte);
        // Cast through uint32_t to make wraparound well-defined; bit-pattern
        // of the result matches the signed-arithmetic case for both SDOT
        // and UDOT per ARM ARM C7.2.397 / C7.2.398.
        acc = static_cast<int32_t>(static_cast<uint32_t>(acc) +
                                   static_cast<uint32_t>(n_ext * m_ext));
      }
      memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &acc, 4);
    }

    // Q=0 zeros the upper 64 bits of the destination vector.
    ClearUpperIfNotQ(&result, args.q);
    state_->cpu.v[args.rd] = result;
  }

  // I8MM SMMLA/UMMLA/USMMLA (FEAT_I8MM). Vn holds a 2x8
  // int8 matrix (two rows of 8), Vm holds an 8x2 matrix stored row-major as
  // two rows of 8 (its transpose), Vd is a 2x2 int32 accumulator. The result
  // is Vd + Vn * Vm^T: lane (2*i + j) accumulates the 8-element dot product of
  // Vn row i with Vm row j. SMMLA = signed*signed, UMMLA = unsigned*unsigned,
  // USMMLA = unsigned(Vn)*signed(Vm). Always .4S (full 128-bit result).
  void AdvSimdMatMul(const Decoder::MatMulArgs& args) {
    CHECK(!exception_raised_);
    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t dst = state_->cpu.v[args.rd];
    uint8_t n_b[16], m_b[16];
    memcpy(n_b, &src_n, 16);
    memcpy(m_b, &src_m, 16);

    const bool n_signed = (args.opcode == Decoder::MatMulOpcode::kSmmla);
    const bool m_signed = (args.opcode == Decoder::MatMulOpcode::kSmmla ||
                           args.opcode == Decoder::MatMulOpcode::kUsmmla);

    __uint128_t result = 0;
    for (uint8_t i = 0; i < 2; i++) {       // row of Vn
      for (uint8_t j = 0; j < 2; j++) {     // row of Vm (column of result)
        int32_t acc;
        memcpy(&acc, reinterpret_cast<const uint8_t*>(&dst) + (2 * i + j) * 4, 4);
        for (uint8_t k = 0; k < 8; k++) {
          uint8_t nb = n_b[8 * i + k];
          uint8_t mb = m_b[8 * j + k];
          int32_t ne = n_signed ? static_cast<int32_t>(static_cast<int8_t>(nb))
                                 : static_cast<int32_t>(nb);
          int32_t me = m_signed ? static_cast<int32_t>(static_cast<int8_t>(mb))
                                 : static_cast<int32_t>(mb);
          acc = static_cast<int32_t>(static_cast<uint32_t>(acc) +
                                     static_cast<uint32_t>(ne * me));
        }
        memcpy(reinterpret_cast<uint8_t*>(&result) + (2 * i + j) * 4, &acc, 4);
      }
    }
    state_->cpu.v[args.rd] = result;
  }

  // BFloat16 helpers.  BF16 is the upper 16 bits of an IEEE-754 single-
  // precision float; widening is a pure shift, narrowing rounds to nearest
  // even with NaN quieting.

  // Widen one BF16 (low 16 bits of u16) to FP32.
  static float Bf16ToFloat(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
  }

  // Narrow one FP32 to BF16 with round-to-nearest-even.  NaN becomes a
  // quiet BF16 NaN; ±Inf and zeros pass through structurally identical.
  static uint16_t FloatToBf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    // NaN: force MSB of the BF16 mantissa to 1 so it stays quiet, and
    // make sure the BF16 keeps at least one mantissa bit set.
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
      return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    // Round to nearest, ties to even: add half-ulp plus low-bit-of-result
    // (so a tie rounds toward even).
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounded = bits + 0x7FFFu + lsb;
    return static_cast<uint16_t>(rounded >> 16);
  }

  // Narrow one FP64 to FP32 with the ARMv8 "Round to Odd" mode (used only by
  // FCVTXN / FCVTXN2). RtO is double-rounding-safe: take the
  // round-toward-zero result, then if any source bits were discarded force
  // the LSB of the result mantissa to 1. NaN propagates as a quiet FP32 NaN;
  // ±Inf passes through; ±0 passes through. The overflow case returns
  // ±FP32_MAX (round-toward-zero clamps the magnitude); the LSB of FP32_MAX
  // is already 1, so RtO leaves it alone. Implemented with manual bit
  // manipulation because there is no softfloat library in-tree and the C++
  // compiler does not expose round-to-odd.
  static uint32_t FpDoubleToFloatRtO(double d) {
    uint64_t bits;
    memcpy(&bits, &d, 8);
    uint32_t sign = static_cast<uint32_t>((bits >> 63) & 1) << 31;
    uint32_t exp_d = static_cast<uint32_t>((bits >> 52) & 0x7FF);
    uint64_t mant_d = bits & ((1ULL << 52) - 1);

    // NaN or ±Inf.
    if (exp_d == 0x7FF) {
      if (mant_d == 0) {
        return sign | 0x7F800000u;
      }
      // Quiet NaN: keep top of payload, force quiet bit.
      return sign | 0x7FC00000u | static_cast<uint32_t>(mant_d >> 29);
    }

    // ±0.
    if (exp_d == 0 && mant_d == 0) {
      return sign;
    }

    int32_t unbiased_exp;
    uint64_t mant_full;
    if (exp_d == 0) {
      // FP64 subnormal: value = mant_d * 2^-1074. Treat as having
      // unbiased exponent -1074 with no leading implicit bit. Any FP64
      // subnormal is far below FP32 range so the result will be ±smallest
      // FP32 denormal via the round-to-odd LSB fixup below.
      unbiased_exp = -1074;
      mant_full = mant_d;
    } else {
      unbiased_exp = static_cast<int32_t>(exp_d) - 1023;
      mant_full = mant_d | (1ULL << 52);  // implicit 1 + 52 fraction bits
    }

    int32_t exp_f = unbiased_exp + 127;

    // Overflow: round-toward-zero clamps to FP32_MAX = 0x7F7FFFFF, whose
    // LSB is already 1 → odd, so RtO leaves it.
    if (exp_f >= 0xFF) {
      return sign | 0x7F7FFFFFu;
    }

    uint32_t mant_f;
    bool any_discarded;
    if (exp_f >= 1) {
      // Normal FP32: top 23 bits of the 53-bit normalized mantissa.
      mant_f = static_cast<uint32_t>((mant_full >> 29) & 0x7FFFFFu);
      any_discarded = (mant_full & ((1ULL << 29) - 1)) != 0;
      mant_f |= static_cast<uint32_t>(exp_f) << 23;
    } else {
      // Subnormal FP32 (or underflow to 0): shift mant_full right by
      // (30 - exp_f) so the result lands in the FP32 denormal mantissa.
      // For exp_f = 0, shift = 30; for exp_f <= -23, the mantissa zeros
      // out and RtO forces the LSB to 1 → smallest denormal.
      int32_t shift = 30 - exp_f;
      if (shift >= 64) {
        mant_f = 0;
        any_discarded = (mant_full != 0);
      } else {
        mant_f = static_cast<uint32_t>(mant_full >> shift);
        uint64_t mask = (shift == 0) ? 0 : ((1ULL << shift) - 1);
        any_discarded = (mant_full & mask) != 0;
      }
    }

    // RtO: if any low bits were discarded, force LSB to 1. (No-op if
    // already 1.)
    if (any_discarded) {
      mant_f |= 1;
    }

    return sign | mant_f;
  }

  // Advanced SIMD BFloat16 three-same-extra (Armv8.6-BF16): BFDOT (vec),
  // BFMMLA, BFMLALB/T (vec), BFDOT (idx), BFMLALB/T (idx).
  //
  // BFDOT (vector):
  //   for each FP32 output lane i in [0..lanes):
  //     n0 = Bf16ToFloat(Vn.h[2i + 0])
  //     n1 = Bf16ToFloat(Vn.h[2i + 1])
  //     m0 = Bf16ToFloat(Vm.h[2i + 0])
  //     m1 = Bf16ToFloat(Vm.h[2i + 1])
  //     Vd.s[i] = Vd.s[i] + n0 * m0 + n1 * m1
  //   lanes = q ? 4 : 2 (Q selects 4S vs 2S).
  //   For Q=0, the upper 64 bits of Vd are zeroed.
  //
  // BFMMLA: Vd.4S viewed as a 2×2 FP32 matrix; Vn.8H / Vm.8H viewed as
  //   2×4 BF16 matrices.  Vd += Vn * Vm^T per ARM ARM C7.2.55:
  //     for i in [0..2):
  //       for j in [0..2):
  //         sum = Vd.s[i*2 + j]
  //         for k in [0..4):
  //           sum += Bf16ToFloat(Vn.h[i*4 + k]) * Bf16ToFloat(Vm.h[j*4 + k])
  //         Vd.s[i*2 + j] = sum
  //   Q is always 1 (decoder rejected Q=0); no upper-half zeroing needed.
  //
  // BFMLALB / BFMLALT (vector): per-FP32-lane single-pair widening MAC.
  //   off = (T ? 1 : 0)
  //   for i in [0..4):
  //     Vd.s[i] = Vd.s[i] + Bf16ToFloat(Vn.h[2i + off]) * Bf16ToFloat(Vm.h[2i + off])
  //   Always 4S (Q implicit 1).
  //
  // BFDOT (by element): same pair-MAC as BFDOT vector but Vm reads a
  //   single indexed BF16 pair (Vm.h[2*idx], Vm.h[2*idx + 1]) and
  //   broadcasts it across all output lanes.  lanes = q ? 4 : 2.
  //
  // BFMLAL{B,T} (by element): per-FP32-lane single-element widening MAC.
  //   off = (T ? 1 : 0)
  //   m = Bf16ToFloat(Vm.h[idx])
  //   for i in [0..4):
  //     Vd.s[i] = Vd.s[i] + Bf16ToFloat(Vn.h[2i + off]) * m
  //   Always 4S (Q implicit 1).
  void AdvSimdBf16ThreeSame(const Decoder::Bf16ThreeSameArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t dst = state_->cpu.v[args.rd];

    uint16_t n_h[8], m_h[8];
    for (uint8_t i = 0; i < 8; i++) {
      memcpy(&n_h[i], reinterpret_cast<const uint8_t*>(&src_n) + i * 2, 2);
      memcpy(&m_h[i], reinterpret_cast<const uint8_t*>(&src_m) + i * 2, 2);
    }

    __uint128_t result = 0;
    switch (args.opcode) {
      case Decoder::Bf16ThreeSameOpcode::kBfdot: {
        uint8_t lanes = args.q ? 4 : 2;
        for (uint8_t i = 0; i < lanes; i++) {
          float acc;
          memcpy(&acc, reinterpret_cast<const uint8_t*>(&dst) + i * 4, 4);
          float n0 = Bf16ToFloat(n_h[2 * i + 0]);
          float n1 = Bf16ToFloat(n_h[2 * i + 1]);
          float m0 = Bf16ToFloat(m_h[2 * i + 0]);
          float m1 = Bf16ToFloat(m_h[2 * i + 1]);
          acc = acc + n0 * m0 + n1 * m1;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &acc, 4);
        }
        break;
      }
      case Decoder::Bf16ThreeSameOpcode::kBfmmla: {
        // BFMMLA: 2x2 output matrix.
        for (uint8_t i = 0; i < 2; i++) {
          for (uint8_t j = 0; j < 2; j++) {
            float sum;
            memcpy(&sum, reinterpret_cast<const uint8_t*>(&dst) + (i * 2 + j) * 4, 4);
            for (uint8_t k = 0; k < 4; k++) {
              sum += Bf16ToFloat(n_h[i * 4 + k]) * Bf16ToFloat(m_h[j * 4 + k]);
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + (i * 2 + j) * 4, &sum, 4);
          }
        }
        break;
      }
      case Decoder::Bf16ThreeSameOpcode::kBfmlalbVec:
      case Decoder::Bf16ThreeSameOpcode::kBfmlaltVec: {
        uint8_t off = (args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlaltVec) ? 1 : 0;
        for (uint8_t i = 0; i < 4; i++) {
          float acc;
          memcpy(&acc, reinterpret_cast<const uint8_t*>(&dst) + i * 4, 4);
          float n = Bf16ToFloat(n_h[2 * i + off]);
          float m = Bf16ToFloat(m_h[2 * i + off]);
          acc = acc + n * m;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &acc, 4);
        }
        break;
      }
      case Decoder::Bf16ThreeSameOpcode::kBfdotIdx: {
        uint8_t lanes = args.q ? 4 : 2;
        float m0 = Bf16ToFloat(m_h[2 * args.index + 0]);
        float m1 = Bf16ToFloat(m_h[2 * args.index + 1]);
        for (uint8_t i = 0; i < lanes; i++) {
          float acc;
          memcpy(&acc, reinterpret_cast<const uint8_t*>(&dst) + i * 4, 4);
          float n0 = Bf16ToFloat(n_h[2 * i + 0]);
          float n1 = Bf16ToFloat(n_h[2 * i + 1]);
          acc = acc + n0 * m0 + n1 * m1;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &acc, 4);
        }
        break;
      }
      case Decoder::Bf16ThreeSameOpcode::kBfmlalbIdx:
      case Decoder::Bf16ThreeSameOpcode::kBfmlaltIdx: {
        uint8_t off = (args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlaltIdx) ? 1 : 0;
        float m = Bf16ToFloat(m_h[args.index]);
        for (uint8_t i = 0; i < 4; i++) {
          float acc;
          memcpy(&acc, reinterpret_cast<const uint8_t*>(&dst) + i * 4, 4);
          float n = Bf16ToFloat(n_h[2 * i + off]);
          acc = acc + n * m;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &acc, 4);
        }
        break;
      }
    }

    state_->cpu.v[args.rd] = result;
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
      // Compute NZCV directly here rather than via UpdateFlags(op1, op2 + carry,
      // ...): folding the carry into op2 wraps to 0 when op2 == all-ones and
      // carry == 1, which would make the carry-out check (result < op1) miss
      // the carry. That saturated-limb case is rare in ordinary code but
      // pervasive in bignum/Montgomery arithmetic (it broke RSA/ECDSA signature
      // verification, hence all TLS). op2 is already inverted above for SUB, so
      // both ADC and SBC are handled as op1 + op2 + carry.
      uint16_t flags = 0;
      unsigned top_bit = is_64bit ? 63 : 31;
      if ((result >> top_bit) & 1) flags |= CPUState::kFlagNegative;
      if (result == 0) flags |= CPUState::kFlagZero;
      // C: true carry-out of op1 + op2 + carry, computed in a wider type so the
      // sum can exceed the operand width without losing the carry bit.
      bool carry_out;
      if (is_64bit) {
        carry_out = (static_cast<__uint128_t>(op1) + op2 + carry) > ~uint64_t{0};
      } else {
        carry_out = (op1 + op2 + carry) > 0xFFFFFFFFULL;
      }
      if (carry_out) flags |= CPUState::kFlagCarry;
      // V: signed overflow — both addends agree in sign but differ from result.
      if (((op1 ^ result) & (op2 ^ result) & (uint64_t{1} << top_bit)) != 0) {
        flags |= CPUState::kFlagOverflow;
      }
      state_->cpu.flags = flags;
    }
    return result;
  }

  Register DataProc1Src(Register src, uint8_t opcode2, bool is_64bit) {
    CHECK(!exception_raised_);
    uint64_t val = is_64bit ? src : (src & 0xFFFFFFFFULL);
    uint64_t result;
    unsigned bits = is_64bit ? 64 : 32;

    // PAuth DP-1Src as identity
    // The decoder sets bit 0x40 to mark a PAuth variant (PACIA/PACIB/PACDA/
    // PACDB/AUTI*/AUTD*/PACIZ*/PACDZ*/AUTIZ*/AUTDZ*/XPACI/XPACD).  Digitalis
    // never injects PAC bits, so authenticate/strip is the identity.
    if (opcode2 & 0x40) {
      return val;
    }

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

    // PMULL handles size=00 (8-bit) and size=11 (64-bit, PMULL64).
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
      // widening multiply-accumulate / multiply-subtract.
      // Four arms parameterized by (is_signed, is_sub). Encodings differ only
      // in U (bit29: signed/unsigned) and bit13 (op: add/sub); decoder maps to
      // distinct enum values. Wraparound is the defined behaviour for both
      // signed and unsigned widening multiply-accumulate, so the unsigned
      // accumulator add/sub (well-defined mod 2^64) yields the same bit
      // pattern as the original signed-then-cast code without the UB risk on
      // signed overflow. Same fan-in shape as the SADDW/SSUBW/UADDW/USUBW arm
      // at line 1904 (four-way (is_signed, is_sub) fan-in) and the SQDMLAL +
      // SQDMLSL collapse below.
      case Decoder::AdvSimdThreeDiffOpcode::kUmlal:
      case Decoder::AdvSimdThreeDiffOpcode::kSmlal:
      case Decoder::AdvSimdThreeDiffOpcode::kUmlsl:
      case Decoder::AdvSimdThreeDiffOpcode::kSmlsl: {
        bool is_signed = (args.opcode == Decoder::AdvSimdThreeDiffOpcode::kSmlal ||
                          args.opcode == Decoder::AdvSimdThreeDiffOpcode::kSmlsl);
        bool is_sub    = (args.opcode == Decoder::AdvSimdThreeDiffOpcode::kUmlsl ||
                          args.opcode == Decoder::AdvSimdThreeDiffOpcode::kSmlsl);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t prod;
          if (is_signed) {
            // get_signed sign-extends from in_esize to int64_t. For all
            // supported in_esize (1, 2, 4), |sn * sm| <= 2^62, so the signed
            // multiplication never overflows int64_t.
            prod = static_cast<uint64_t>(get_signed(src_n, i) * get_signed(src_m, i));
          } else {
            prod = get_unsigned(src_n, i) * get_unsigned(src_m, i);
          }
          uint64_t accum = get_accum(i);
          set_result(i, is_sub ? accum - prod : accum + prod);
        }
        break;
      }
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
      // wide add/sub: Vn is already wide (out_esize per elem);
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
      // narrowing high: Vd(narrow) = ((Vn + Vm) [+ round]) >>
      // narrow_bits. ADDHN: round=0. RADDHN: round=1<<(narrow_bits-1). Q=0 writes
      // narrow lanes to lower 64 bits of Vd (upper cleared); Q=1 writes narrow
      // lanes to upper 64 bits (lower preserved). size encodes narrow elem width;
      // sources Vn/Vm are wide (2*in_esize per lane).
      case Decoder::AdvSimdThreeDiffOpcode::kAddhn:
      case Decoder::AdvSimdThreeDiffOpcode::kRaddhn: {
        uint8_t narrow_bits = in_esize * 8;
        uint64_t round = (args.opcode == Decoder::AdvSimdThreeDiffOpcode::kRaddhn)
                             ? (1ULL << (narrow_bits - 1))
                             : 0;
        uint8_t narrow_lanes[8] = {0};
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a = 0;
          uint64_t b = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * out_esize, out_esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * out_esize, out_esize);
          uint64_t sum = a + b + round;
          if (out_esize < 8) {
            // Mask to wide width before taking the high half. For out_esize==8
            // we skip the mask: shifting `(1ULL << 64) - 1` is UB, and the
            // natural uint64_t wrap already gives the wide-bit-width sum.
            sum &= (1ULL << (out_esize * 8)) - 1;
          }
          uint64_t high = sum >> narrow_bits;
          memcpy(narrow_lanes + i * in_esize, &high, in_esize);
        }
        // Q=0: `result` was initialized to 0 at function entry, so upper 64
        // bits already cleared. Q=1: copy dst so lower 64 bits are preserved
        // before we overwrite the upper half.
        if (args.q) {
          result = dst;
        }
        memcpy(reinterpret_cast<uint8_t*>(&result) + (args.q ? 8 : 0), narrow_lanes, 8);
        break;
      }
      // narrowing high subtract: Vd(narrow) =
      //   ((Vn - Vm) [+ round]) >> narrow_bits.
      // SUBHN: round=0. RSUBHN: round=1<<(narrow_bits-1). Q semantics and
      // source/dest shapes are identical to ADDHN/RADDHN. Two's-complement
      // wrap of `a - b` modulo 2^(out_esize*8) is implicit via the uint64_t
      // subtract + mask: the high-half bits we shift out are the wide-width
      // result regardless of borrow.
      case Decoder::AdvSimdThreeDiffOpcode::kSubhn:
      case Decoder::AdvSimdThreeDiffOpcode::kRsubhn: {
        uint8_t narrow_bits = in_esize * 8;
        uint64_t round = (args.opcode == Decoder::AdvSimdThreeDiffOpcode::kRsubhn)
                             ? (1ULL << (narrow_bits - 1))
                             : 0;
        uint8_t narrow_lanes[8] = {0};
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a = 0;
          uint64_t b = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * out_esize, out_esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * out_esize, out_esize);
          uint64_t diff = a - b + round;
          if (out_esize < 8) {
            diff &= (1ULL << (out_esize * 8)) - 1;
          }
          uint64_t high = diff >> narrow_bits;
          memcpy(narrow_lanes + i * in_esize, &high, in_esize);
        }
        if (args.q) {
          result = dst;
        }
        memcpy(reinterpret_cast<uint8_t*>(&result) + (args.q ? 8 : 0), narrow_lanes, 8);
        break;
      }
      // signed saturating doubling multiply long:
      //   Vd_wide[i] = SignedSat(2 * Vn_narrow[i] * Vm_narrow[i])
      // The only input pair that overflows the wide signed range is
      // (INT_MIN, INT_MIN), which yields a positive product whose double
      // exceeds INT_MAX_out. Every other lane fits safely in int64_t even
      // for the 32->64 widening: max non-INT_MIN-squared abs product is
      // |INT32_MIN * (INT32_MIN+1)| = 0x3FFFFFFF80000000, and doubling
      // gives 0x7FFFFFFF00000000, well within int64_t. The decoder
      // rejects size=00 and size=11.
      case Decoder::AdvSimdThreeDiffOpcode::kSqdmull: {
        int64_t int_min_in = -(1LL << (in_esize * 8 - 1));
        int64_t int_max_out =
            (out_esize == 8) ? INT64_MAX : ((1LL << (out_esize * 8 - 1)) - 1);
        for (uint8_t i = 0; i < num_elements; i++) {
          int64_t sn = get_signed(src_n, i);
          int64_t sm = get_signed(src_m, i);
          int64_t doubled;
          if (sn == int_min_in && sm == int_min_in) {
            doubled = int_max_out;
          } else {
            doubled = 2 * sn * sm;
          }
          set_result(i, static_cast<uint64_t>(doubled));
        }
        break;
      }
      // signed saturating doubling multiply-accumulate /
      // multiply-subtract long:
      //   SQDMLAL: Vd_wide[i] = SignedSat(Vd_wide[i] + SignedSat(2*sn*sm)).
      //   SQDMLSL: Vd_wide[i] = SignedSat(Vd_wide[i] - SignedSat(2*sn*sm)).
      // Two stages of saturation:
      //   (1) compute the doubled product with input-based saturation
      //       (same as kSqdmull; only (INT_MIN_in, INT_MIN_in) saturates,
      //       to INT_MAX_out);
      //   (2) saturate the wide signed accumulator add (SQDMLAL) or
      //       subtract (SQDMLSL) against the wide signed range
      //       [INT_MIN_out, INT_MAX_out].
      // SQDMLSL is folded into the SQDMLAL path by negating `addend`,
      // which is safe because stage-1 output range is
      // [INT_MIN_out+1, INT_MAX_out] (never reaches INT_MIN_out exactly,
      // since 2*sn*sm = INT_MIN_out would require |sn*sm| = 2^(out_bits-1)
      // which is unreachable from the narrower input range).
      // For out_esize=4 (in=16), int64_t has 32-bit headroom for both
      // operands and the sum, so a plain int64_t add + compare suffices.
      // For out_esize=8 (in=32), the sum can overflow int64_t; use
      // unsigned wrap + signed-overflow detection (same-sign operands,
      // result sign differs).
      case Decoder::AdvSimdThreeDiffOpcode::kSqdmlal:
      case Decoder::AdvSimdThreeDiffOpcode::kSqdmlsl: {
        bool is_sub = (args.opcode == Decoder::AdvSimdThreeDiffOpcode::kSqdmlsl);
        int64_t int_min_in = -(1LL << (in_esize * 8 - 1));
        int64_t int_max_out =
            (out_esize == 8) ? INT64_MAX : ((1LL << (out_esize * 8 - 1)) - 1);
        int64_t int_min_out =
            (out_esize == 8) ? INT64_MIN : -(1LL << (out_esize * 8 - 1));
        for (uint8_t i = 0; i < num_elements; i++) {
          int64_t sn = get_signed(src_n, i);
          int64_t sm = get_signed(src_m, i);
          int64_t addend;
          if (sn == int_min_in && sm == int_min_in) {
            addend = int_max_out;
          } else {
            addend = 2 * sn * sm;
          }
          if (is_sub) {
            addend = -addend;
          }
          // Sign-extend the existing accumulator lane to int64_t.
          uint64_t acc_raw = get_accum(i);
          int64_t acc;
          if (out_esize == 8) {
            acc = static_cast<int64_t>(acc_raw);
          } else {
            uint8_t shift = (8 - out_esize) * 8;
            acc = static_cast<int64_t>(acc_raw << shift) >> shift;
          }
          int64_t sum;
          if (out_esize == 8) {
            uint64_t usum = static_cast<uint64_t>(acc) + static_cast<uint64_t>(addend);
            sum = static_cast<int64_t>(usum);
            bool a_neg = acc < 0;
            bool b_neg = addend < 0;
            bool s_neg = sum < 0;
            if (a_neg == b_neg && a_neg != s_neg) {
              sum = a_neg ? int_min_out : int_max_out;
            }
          } else {
            sum = acc + addend;
            if (sum > int_max_out) {
              sum = int_max_out;
            } else if (sum < int_min_out) {
              sum = int_min_out;
            }
          }
          set_result(i, static_cast<uint64_t>(sum));
        }
        break;
      }
      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = result;
  }

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    CHECK(!exception_raised_);
    // EXT: result = bytes[index .. index+datasize) of the concatenation Vm:Vn,
    // where Vn is the LOW operand and Vm the HIGH operand, each `num_bytes` wide
    // (8 for Q=0, 16 for Q=1). Vm therefore starts at byte `num_bytes`, not 16 —
    // for Q=0 placing it at 16 leaves Vm unreachable and pulls Vn's stale upper
    // half into the result lanes. Bits[127:64] are zeroed for Q=0.
    const unsigned num_bytes = q ? 16 : 8;
    uint8_t bytes[32];
    memcpy(bytes, &state_->cpu.v[rn], num_bytes);
    memcpy(bytes + num_bytes, &state_->cpu.v[rm], num_bytes);
    __uint128_t result = 0;
    memcpy(&result, bytes + index, num_bytes);
    state_->cpu.v[rd] = result;
  }

  // TBL / TBX (vector table lookup). Reads `len+1`
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

  // SHA-512 (FEAT_SHA512). Each op operates on .2D vectors.
  // The references below match ARM ARM C7.2.85/86/87/88 pseudocode exactly,
  // double-checked against FIPS-180-4 section 4.1.3 (SHA-512 round functions).
  // SHA3 (FEAT_SHA3).
  //   EOR3 Vd = Vn ^ Vm ^ Va   (full 128-bit)
  //   BCAX Vd = Vn ^ (Vm & ~Va)
  //   RAX1 Vd[i] = Vn[i] ^ ROL(Vm[i], 1)   per 64-bit lane
  //   XAR  Vd[i] = ROR(Vn[i] ^ Vm[i], imm6) per 64-bit lane
  void Eor3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    CHECK(!exception_raised_);
    state_->cpu.v[rd] =
        state_->cpu.v[rn] ^ state_->cpu.v[rm] ^ state_->cpu.v[ra];
  }
  void Bcax(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    CHECK(!exception_raised_);
    state_->cpu.v[rd] =
        state_->cpu.v[rn] ^ (state_->cpu.v[rm] & ~state_->cpu.v[ra]);
  }
  void Rax1(uint8_t rd, uint8_t rn, uint8_t rm) {
    CHECK(!exception_raised_);
    uint64_t n[2], m[2], out[2];
    __uint128_t vn = state_->cpu.v[rn], vm = state_->cpu.v[rm];
    memcpy(n, &vn, 16);
    memcpy(m, &vm, 16);
    for (int i = 0; i < 2; i++) {
      uint64_t rol1 = (m[i] << 1) | (m[i] >> 63);
      out[i] = n[i] ^ rol1;
    }
    __uint128_t vd;
    memcpy(&vd, out, 16);
    state_->cpu.v[rd] = vd;
  }
  void Xar(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm6) {
    CHECK(!exception_raised_);
    uint64_t n[2], m[2], out[2];
    __uint128_t vn = state_->cpu.v[rn], vm = state_->cpu.v[rm];
    memcpy(n, &vn, 16);
    memcpy(m, &vm, 16);
    for (int i = 0; i < 2; i++) {
      uint64_t x = n[i] ^ m[i];
      out[i] = imm6 ? ((x >> imm6) | (x << (64u - imm6))) : x;
    }
    __uint128_t vd;
    memcpy(&vd, out, 16);
    state_->cpu.v[rd] = vd;
  }

  // SM4 (FEAT_SM4): SM4E (encryption rounds) and SM4EKEY
  // (key expansion). Word i of a vector is bits[32*i+31:32*i] (word 0 = low).
  // Each instruction performs the four SM4 rounds with the rolling update
  //   t = w[(i+1)%4] ^ w[(i+2)%4] ^ w[(i+3)%4] ^ k[i]
  //   t = S-box(each byte of t)
  //   w[i] ^= L(t)
  // where for SM4E (k = Vn round keys) L(t) = t ^ rol(t,2)^rol(t,10)^rol(t,18)
  // ^rol(t,24), and for SM4EKEY (state seeded from Vn, k = Vm) the
  // key-expansion linear map L'(t) = t ^ rol(t,13) ^ rol(t,23). Matches the
  // ARM ARM SM4E/SM4EKEY pseudocode (standard SM4 S-box / transforms).
  static uint32_t Sm4Subword(uint32_t x) {
    static const uint8_t kSm4Sbox[256] = {
        0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05,
        0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99,
        0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62,
        0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6,
        0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8,
        0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35,
        0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87,
        0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e,
        0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1,
        0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3,
        0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f,
        0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51,
        0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8,
        0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0,
        0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84,
        0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48};
    return static_cast<uint32_t>(kSm4Sbox[x & 0xff]) |
           (static_cast<uint32_t>(kSm4Sbox[(x >> 8) & 0xff]) << 8) |
           (static_cast<uint32_t>(kSm4Sbox[(x >> 16) & 0xff]) << 16) |
           (static_cast<uint32_t>(kSm4Sbox[(x >> 24) & 0xff]) << 24);
  }
  static uint32_t Sm4Rol(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
  }
  void Sm4e(uint8_t rd, uint8_t rn) {
    CHECK(!exception_raised_);
    uint32_t w[4], k[4];
    __uint128_t vd = state_->cpu.v[rd], vn = state_->cpu.v[rn];
    memcpy(w, &vd, 16);
    memcpy(k, &vn, 16);
    for (int i = 0; i < 4; i++) {
      uint32_t t = w[(i + 1) % 4] ^ w[(i + 2) % 4] ^ w[(i + 3) % 4] ^ k[i];
      t = Sm4Subword(t);
      w[i] ^= t ^ Sm4Rol(t, 2) ^ Sm4Rol(t, 10) ^ Sm4Rol(t, 18) ^ Sm4Rol(t, 24);
    }
    memcpy(&vd, w, 16);
    state_->cpu.v[rd] = vd;
  }
  void Sm4ekey(uint8_t rd, uint8_t rn, uint8_t rm) {
    CHECK(!exception_raised_);
    uint32_t w[4], k[4];
    __uint128_t vn = state_->cpu.v[rn], vm = state_->cpu.v[rm];
    memcpy(w, &vn, 16);  // state seeded from Vn
    memcpy(k, &vm, 16);  // constants/keys from Vm
    for (int i = 0; i < 4; i++) {
      uint32_t t = w[(i + 1) % 4] ^ w[(i + 2) % 4] ^ w[(i + 3) % 4] ^ k[i];
      t = Sm4Subword(t);
      w[i] ^= t ^ Sm4Rol(t, 13) ^ Sm4Rol(t, 23);
    }
    __uint128_t vd;
    memcpy(&vd, w, 16);
    state_->cpu.v[rd] = vd;
  }

  // SM3 (FEAT_SM3): SM3SS1, SM3TT1A/1B/2A/2B, SM3PARTW1/2.
  // Word i of a vector is bits[32i+31:32i] (word 0 = low). Derived from the
  // SM3 round/expansion (GB/T 32905); the full sequence reproduces the
  // published SM3("abc") digest. P0(x)=x^rol(x,9)^rol(x,17),
  // P1(x)=x^rol(x,15)^rol(x,23).
  static uint32_t Sm3P0(uint32_t x) { return x ^ Sm4Rol(x, 9) ^ Sm4Rol(x, 17); }
  static uint32_t Sm3Ror(uint32_t x, int n) { return Sm4Rol(x, 32 - n); }
  // SM3SS1: Vd[3] = rol(rol(Vn[3],12) + Vm[3] + Va[3], 7); Vd[2:0] = 0.
  void Sm3ss1(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    CHECK(!exception_raised_);
    uint32_t n[4], m[4], a[4];
    __uint128_t vn = state_->cpu.v[rn], vm = state_->cpu.v[rm], va = state_->cpu.v[ra];
    memcpy(n, &vn, 16);
    memcpy(m, &vm, 16);
    memcpy(a, &va, 16);
    uint32_t out[4] = {0, 0, 0, Sm4Rol(Sm4Rol(n[3], 12) + m[3] + a[3], 7)};
    __uint128_t vd;
    memcpy(&vd, out, 16);
    state_->cpu.v[rd] = vd;
  }
  // SM3TT: op 00=TT1A, 01=TT1B, 10=TT2A, 11=TT2B. Vd holds the working state
  // (D,C,B,A) for TT1 / (H,G,F,E) for TT2 in lanes 0..3; Vn[3] is SS2 (TT1) or
  // SS1 (TT2); Vm[imm2] is W'[j] (TT1) or W[j] (TT2).
  void Sm3tt(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm2, uint8_t op) {
    CHECK(!exception_raised_);
    uint32_t d[4], n[4], m[4];
    __uint128_t vd = state_->cpu.v[rd], vn = state_->cpu.v[rn], vm = state_->cpu.v[rm];
    memcpy(d, &vd, 16);
    memcpy(n, &vn, 16);
    memcpy(m, &vm, 16);
    const bool is_tt2 = (op & 0b10);
    const bool is_b = (op & 0b01);
    const uint32_t wj = m[imm2 & 3];
    uint32_t out[4];
    if (!is_tt2) {
      uint32_t ff = is_b ? ((d[3] & d[2]) | (d[3] & d[1]) | (d[2] & d[1]))
                         : (d[3] ^ d[2] ^ d[1]);
      uint32_t t = ff + d[0] + n[3] + wj;
      out[0] = d[1];
      out[1] = Sm3Ror(d[2], 23);
      out[2] = d[3];
      out[3] = t;
    } else {
      uint32_t gg = is_b ? ((d[3] & d[2]) | (~d[3] & d[1])) : (d[3] ^ d[2] ^ d[1]);
      uint32_t t = gg + d[0] + n[3] + wj;
      out[0] = d[1];
      out[1] = Sm3Ror(d[2], 13);
      out[2] = d[3];
      out[3] = Sm3P0(t);
    }
    memcpy(&vd, out, 16);
    state_->cpu.v[rd] = vd;
  }
  // SM3PARTW1: Vd[i] = P1(Vd[i] ^ Vn[i] ^ rol(Vm[i+1],15)); lane 3 uses the
  // just-written Vd[0] for the rotate term.
  void Sm3partw1(uint8_t rd, uint8_t rn, uint8_t rm) {
    CHECK(!exception_raised_);
    uint32_t d[4], n[4], m[4];
    __uint128_t vd = state_->cpu.v[rd], vn = state_->cpu.v[rn], vm = state_->cpu.v[rm];
    memcpy(d, &vd, 16);
    memcpy(n, &vn, 16);
    memcpy(m, &vm, 16);
    auto p1 = [](uint32_t x) { return x ^ Sm4Rol(x, 15) ^ Sm4Rol(x, 23); };
    uint32_t t;
    t = d[0] ^ n[0] ^ Sm4Rol(m[1], 15); d[0] = p1(t);
    t = d[1] ^ n[1] ^ Sm4Rol(m[2], 15); d[1] = p1(t);
    t = d[2] ^ n[2] ^ Sm4Rol(m[3], 15); d[2] = p1(t);
    t = d[3] ^ n[3] ^ Sm4Rol(d[0], 15); d[3] = p1(t);
    memcpy(&vd, d, 16);
    state_->cpu.v[rd] = vd;
  }
  // SM3PARTW2: Vd[i] ^= rol(Vn[i],7) ^ Vm[i].
  void Sm3partw2(uint8_t rd, uint8_t rn, uint8_t rm) {
    CHECK(!exception_raised_);
    uint32_t d[4], n[4], m[4];
    __uint128_t vd = state_->cpu.v[rd], vn = state_->cpu.v[rn], vm = state_->cpu.v[rm];
    memcpy(d, &vd, 16);
    memcpy(n, &vn, 16);
    memcpy(m, &vm, 16);
    for (int i = 0; i < 4; i++) d[i] ^= Sm4Rol(n[i], 7) ^ m[i];
    memcpy(&vd, d, 16);
    state_->cpu.v[rd] = vd;
  }

  void Sha512(Decoder::Sha512Op op, uint8_t rd, uint8_t rn, uint8_t rm) {
    CHECK(!exception_raised_);
    auto ror64 = [](uint64_t x, unsigned n) -> uint64_t {
      return (x >> n) | (x << (64u - n));
    };
    auto big_sigma0 = [&](uint64_t x) -> uint64_t {
      return ror64(x, 28) ^ ror64(x, 34) ^ ror64(x, 39);
    };
    auto big_sigma1 = [&](uint64_t x) -> uint64_t {
      return ror64(x, 14) ^ ror64(x, 18) ^ ror64(x, 41);
    };
    auto little_sigma0 = [&](uint64_t x) -> uint64_t {
      return ror64(x, 1) ^ ror64(x, 8) ^ (x >> 7);
    };
    auto little_sigma1 = [&](uint64_t x) -> uint64_t {
      return ror64(x, 19) ^ ror64(x, 61) ^ (x >> 6);
    };

    uint64_t d[2], n[2], m[2];
    {
      __uint128_t vd_v = state_->cpu.v[rd];
      __uint128_t vn_v = state_->cpu.v[rn];
      __uint128_t vm_v = state_->cpu.v[rm];
      memcpy(d, &vd_v, 16);
      memcpy(n, &vn_v, 16);
      memcpy(m, &vm_v, 16);
    }

    uint64_t out[2] = {d[0], d[1]};
    switch (op) {
      case Decoder::Sha512Op::kSha512h: {
        uint64_t e = m[0], f = n[0], g = n[1];
        uint64_t tmp = ((e & f) ^ (~e & g)) + big_sigma1(e) + d[1];
        out[1] = d[0] + tmp;
        out[0] = tmp;
        break;
      }
      case Decoder::Sha512Op::kSha512h2: {
        auto maj = [](uint64_t a, uint64_t b, uint64_t c) -> uint64_t {
          return (a & b) ^ (a & c) ^ (b & c);
        };
        uint64_t new_d1 =
            d[1] + big_sigma0(n[0]) + maj(n[0], m[0], m[1]);
        uint64_t new_d0 =
            d[0] + big_sigma0(new_d1) + maj(new_d1, n[0], m[0]);
        out[0] = new_d0;
        out[1] = new_d1;
        break;
      }
      case Decoder::Sha512Op::kSha512su0:
        out[0] = d[0] + little_sigma0(d[1]);
        out[1] = d[1] + little_sigma0(n[0]);
        break;
      case Decoder::Sha512Op::kSha512su1:
        out[0] = d[0] + little_sigma1(n[0]) + m[0];
        out[1] = d[1] + little_sigma1(n[1]) + m[1];
        break;
    }

    __uint128_t result = 0;
    memcpy(&result, out, 16);
    state_->cpu.v[rd] = result;
  }

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
          void* ptr = ToHostAddr<void>(ApplyTbi(addr));
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
          void* ptr = ToHostAddr<void>(ApplyTbi(addr));
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
        void* base_ptr = ToHostAddr<void>(ApplyTbi(base_addr));
        for (uint64_t off = 0; off + 8 <= total_bytes; off += 8) {
          uint64_t chunk;
          memcpy(&chunk, buf + off, 8);
          if (FaultyStore(static_cast<uint8_t*>(base_ptr) + off, 8, chunk)) {
            HandleMemoryFault(base_addr + off); return;
          }
        }
      } else {
        void* base_ptr = ToHostAddr<void>(ApplyTbi(base_addr));
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

        FaultyLoadResult res = FaultyLoad(ToHostAddr<void>(ApplyTbi(addr)), esize);
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
        ClearUpperIfNotQ(&val, args.q);
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
          if (FaultyStore(ToHostAddr<void>(ApplyTbi(addr)), esize, elem)) {
            HandleMemoryFault(addr); return;
          }
        } else {
          FaultyLoadResult res = FaultyLoad(ToHostAddr<void>(ApplyTbi(addr)), esize);
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

  // DMB/DSB full barrier: a full memory fence. x86 TSO lacks StoreLoad ordering,
  // so a seq-cst thread fence (MFENCE on x86) is required to honor a guest full
  // barrier; otherwise sequentially consistent guest code races.
  void DataMemoryBarrier() { std::atomic_thread_fence(std::memory_order_seq_cst); }

  // IC IVAU, Xt — Instruction Cache Invalidate by VA. ARM64 user-space JITs
  // (e.g. PCRE2/sljit) issue this after writing new code at a (possibly reused)
  // address. Invalidate the translation cache for that cache line so the next
  // execution re-translates the new code instead of running a stale translation.
  void IcIvau(uint8_t rt) {
    if (rt == 31) {
      return;
    }
    constexpr uint64_t kCacheLine = 64;
    GuestAddr addr = state_->cpu.x[rt];
    GuestAddr line = addr & ~(kCacheLine - 1);
    InvalidateGuestRange(line, line + kCacheLine);
  }

  void Undefined() {
    UndefinedInsn(GetInsnAddr());
    exception_raised_ = true;
  }

  //
  // SIMD/FP instruction implementations.
  //

  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs& args) {
    CHECK(!exception_raised_);
    const uint8_t cmode = args.cmode;
    // ORR/BIC (vector, immediate) are encoded as cmode<0>==1 with cmode<3:2>!=11
    // (the cmode<3:2>==11 forms are MOVI/MVNI shifting-ones, MOVI/FMOV). Unlike
    // MOVI/MVNI they are read-modify-write: the immediate is the MOVI-style
    // (op=0) expansion; ORR (op=0) sets those bits, BIC (op=1) clears them.
    const bool is_orr_bic = (cmode & 1) && ((cmode & 0b1100) != 0b1100);
    __uint128_t result;
    if (is_orr_bic) {
      const __uint128_t imm = ExpandSimdModifiedImm(0, cmode, args.abc, args.defgh, args.q);
      const __uint128_t cur = state_->cpu.v[args.rd];
      result = (args.op == 0) ? (cur | imm) : (cur & ~imm);
    } else {
      result = ExpandSimdModifiedImm(args.op, cmode, args.abc, args.defgh, args.q);
    }
    // Q==0 operates on the lower 64 bits; the upper 64 bits of Vd are zeroed.
    ClearUpperIfNotQ(&result, args.q);
    state_->cpu.v[args.rd] = result;
  }

  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& args, Register base) {
    CHECK(!exception_raised_);
    uint64_t addr = base + args.offset;
    void* host_addr = ToHostAddr<void>(ApplyTbi(addr));

    if (args.is_store) {
      SimdStoreToMemory(host_addr, args.rt, args.size);
    } else {
      SimdLoadFromMemory(host_addr, args.rt, args.size);
    }
  }

  // LDR (literal) SIMD/FP: load 32/64/128 bits from [insn_addr + offset]
  // into V[rt]. SimdLoadFromMemory zero-extends the upper bits and uses
  // FaultyLoad for fault recovery.
  void SimdLoadLiteral(const Decoder::SimdLoadLiteralArgs& args) {
    CHECK(!exception_raised_);
    uint64_t addr = state_->cpu.insn_addr + args.offset;
    void* host_addr = ToHostAddr<void>(ApplyTbi(addr));
    SimdLoadFromMemory(host_addr, args.rt, args.size);
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

    void* addr1 = ToHostAddr<void>(ApplyTbi(base));
    void* addr2 = ToHostAddr<void>(ApplyTbi(base + element_size));

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
    // Apply the offset register extension before
    // shift+add (see ApplyOffsetExtend comment above).
    uint64_t off = ApplyOffsetExtend(offset_reg, args.extend_type) << args.shift_amount;
    uint64_t addr = base + off;
    void* host_addr = ToHostAddr<void>(ApplyTbi(addr));

    if (args.is_store) {
      SimdStoreToMemory(host_addr, args.rt, args.size);
    } else {
      SimdLoadFromMemory(host_addr, args.rt, args.size);
    }
  }

  // FCSEL: Floating-point conditional select
  // If condition is true, Rd = Rn; else Rd = Rm.
  void FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ftype, Decoder::Condition cond) {
    CHECK(!exception_raised_);
    bool condition_holds = EvaluateCondition(cond);
    uint8_t src = condition_holds ? rn : rm;
    // Read the chosen source value BEFORE clearing V[rd]. FCSEL very commonly
    // has rd == rn (e.g. `fcsel d0, d0, d1, cond` for conditional negate/abs/
    // min/max), so zeroing V[rd] first would wipe the source and yield 0.
    if (ftype == 0b00) {
      // Single-precision: copy 32 bits
      uint32_t val;
      memcpy(&val, &state_->cpu.v[src], 4);
      state_->cpu.v[rd] = 0;
      memcpy(&state_->cpu.v[rd], &val, 4);
    } else if (ftype == 0b01) {
      // Double-precision: copy 64 bits
      uint64_t val;
      memcpy(&val, &state_->cpu.v[src], 8);
      state_->cpu.v[rd] = 0;
      memcpy(&state_->cpu.v[rd], &val, 8);
    } else if (ftype == 0b11) {
      // half-precision FCSEL.
      uint16_t val;
      memcpy(&val, &state_->cpu.v[src], 2);
      state_->cpu.v[rd] = 0;
      memcpy(&state_->cpu.v[rd], &val, 2);
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
        } else if (args.ftype == 0b11) {
          // int / 2^fbits in the FP32 domain, then RNE narrow to fp16. The
          // 2^-fbits scale is an exact power-of-two float (biases the exponent
          // only). Was previously mis-handled as FP64 (silent 8-byte write).
          uint32_t sb = static_cast<uint32_t>(127u - args.fbits) << 23;  // 2^-fbits
          float sf; memcpy(&sf, &sb, 4);
          uint16_t h = FpSingleToHalfRN(static_cast<float>(int_val) * sf);
          memcpy(&state_->cpu.v[args.rd], &h, 2);
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
        } else if (args.ftype == 0b11) {
          uint32_t sb = static_cast<uint32_t>(127u - args.fbits) << 23;  // 2^-fbits
          float sf; memcpy(&sf, &sb, 4);
          uint16_t h = FpSingleToHalfRN(static_cast<float>(uint_val) * sf);
          memcpy(&state_->cpu.v[args.rd], &h, 2);
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
        } else if (args.ftype == 0b11) {
          uint16_t h; memcpy(&h, &state_->cpu.v[args.rn], 2);
          fp_val = static_cast<double>(FpHalfToSingle(h));  // exact fp16 widen
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
        } else if (args.ftype == 0b11) {
          uint16_t h; memcpy(&h, &state_->cpu.v[args.rn], 2);
          fp_val = static_cast<double>(FpHalfToSingle(h));  // exact fp16 widen
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
    } else if (ftype == 0b11) {
      // Half-precision FMADD/FMSUB/FNMADD/FNMSUB.
      // Use double-precision fma() so the multiply-add is computed exactly
      // before single-rounding back to FP16.  binary64 mantissa (53 bits)
      // covers any (binary16 * binary16) + binary16 product exactly, so the
      // only rounding is the final float->half conversion.
      uint16_t bn, bm, ba;
      memcpy(&bn, &state_->cpu.v[rn], 2);
      memcpy(&bm, &state_->cpu.v[rm], 2);
      memcpy(&ba, &state_->cpu.v[ra], 2);
      double fn = static_cast<double>(FpHalfToSingle(bn));
      double fm = static_cast<double>(FpHalfToSingle(bm));
      double fa = static_cast<double>(FpHalfToSingle(ba));
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
      uint16_t result_bits = FpSingleToHalf(static_cast<float>(result));
      state_->cpu.v[rd] = 0;
      memcpy(&state_->cpu.v[rd], &result_bits, 2);
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
    } else if (ftype == 0b11) {
      // half-precision FMOV immediate.
      // VFPExpandImm to FP16:
      //   sign = imm8[7]
      //   exp  = NOT(imm8[6]):Repeat(imm8[6], 2):imm8[5:4]   (5 bits)
      //   frac = imm8[3:0]:Zeros(6)                          (10 bits)
      uint16_t sign = (imm8 >> 7) & 1;
      uint16_t exp6 = (imm8 >> 6) & 1;
      uint16_t exp_top = exp6 ? 0 : 1;
      uint16_t exp_rep = exp6 ? 0b11 : 0b00;
      uint16_t exp_low = (imm8 >> 4) & 0b11;
      uint16_t exp = (exp_top << 4) | (exp_rep << 2) | exp_low;
      uint16_t frac = static_cast<uint16_t>(imm8 & 0xF) << 6;
      uint16_t result = (sign << 15) | (exp << 10) | frac;
      memcpy(&state_->cpu.v[rd], &result, 2);
    } else {
      Undefined();
    }
  }

  void FpIntConversion(const Decoder::FpIntConvArgs& args) {
    CHECK(!exception_raised_);
    uint8_t rmode = args.rmode;
    uint8_t opcode = args.op;

    // (digitalis FMOV trace removed)

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
      } else if (args.ftype == 0b11) {
        // int -> FP32 (RNE), then narrow FP32 -> fp16 (RNE); matches the JIT's
        // CVTSI2SS + VCVTPS2PH and ARM's default rounding.
        uint16_t h = FpSingleToHalfRN(static_cast<float>(ival));
        memcpy(&state_->cpu.v[args.rd], &h, 2);
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
      } else if (args.ftype == 0b11) {
        uint16_t h = FpSingleToHalfRN(static_cast<float>(uval));
        memcpy(&state_->cpu.v[args.rd], &h, 2);
      } else { Undefined(); }
      return;
    }

    // FCVTNS/FCVTNU/FCVTPS/FCVTPU/FCVTMS/FCVTMU: various rounding modes
    // rmode=00: round to nearest (ties to even)
    // rmode=01: round toward +inf
    // rmode=10: round toward -inf
    // rmode=11: round toward zero
    // opcode=000: signed, opcode=001: unsigned
    //
    // ARM saturation rules (applied AFTER rounding):
    //   NaN              -> 0
    //   FP < min_int     -> INT_MIN (signed); 0 (unsigned, since FP<0 -> 0)
    //   FP > max_int     -> INT_MAX (signed); UINT_MAX (unsigned)
    //   in-range         -> truncated value
    // The host C++ static_cast on out-of-range FP is undefined / implementation
    // defined and on x86 typically returns INT_MIN (the "indefinite" CVTTSD2SI
    // result), which mis-saturates positive overflow.  Apply explicit ARM
    // saturation instead.
    auto sat_to_int32 = [](double r) -> int32_t {
      if (std::isnan(r)) return 0;
      if (r >= 0x1p31) return INT32_MAX;   // r >= 2^31  (= INT32_MAX + 1)
      if (r < -0x1p31) return INT32_MIN;   // r < -2^31  (= INT32_MIN exact)
      return static_cast<int32_t>(r);
    };
    auto sat_to_int64 = [](double r) -> int64_t {
      if (std::isnan(r)) return 0;
      if (r >= 0x1p63) return INT64_MAX;   // r >= 2^63
      if (r < -0x1p63) return INT64_MIN;
      return static_cast<int64_t>(r);
    };
    auto sat_to_uint32 = [](double r) -> uint32_t {
      if (std::isnan(r)) return 0;
      if (r <= 0.0) return 0;              // ARM unsigned: FP <= 0 -> 0
      if (r >= 0x1p32) return UINT32_MAX;
      return static_cast<uint32_t>(r);
    };
    auto sat_to_uint64 = [](double r) -> uint64_t {
      if (std::isnan(r)) return 0;
      if (r <= 0.0) return 0;
      if (r >= 0x1p64) return UINT64_MAX;
      return static_cast<uint64_t>(r);
    };

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
      } else if (args.ftype == 0b11) {
        uint16_t h; memcpy(&h, &state_->cpu.v[args.rn], 2);
        dval = static_cast<double>(FpHalfToSingle(h));  // fp16 widen is exact
      } else { Undefined(); return; }
      // Apply rounding
      double rounded;
      if (rmode == 0b00) rounded = rint(dval);        // nearest, ties to even
      else if (rmode == 0b01) rounded = ceil(dval);    // toward +inf
      else rounded = floor(dval);                       // toward -inf
      if (is_signed) {
        if (args.sf) result = static_cast<uint64_t>(sat_to_int64(rounded));
        else result = static_cast<uint64_t>(static_cast<uint32_t>(sat_to_int32(rounded)));
      } else {
        if (args.sf) result = sat_to_uint64(rounded);
        else result = sat_to_uint32(rounded);
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
      } else if (args.ftype == 0b11) {
        uint16_t h; memcpy(&h, &state_->cpu.v[args.rn], 2);
        dval = static_cast<double>(FpHalfToSingle(h));  // fp16 widen is exact
      } else { Undefined(); return; }
      // Round to nearest, ties away from zero
      double rounded = round(dval);
      if (is_signed) {
        if (args.sf) result = static_cast<uint64_t>(sat_to_int64(rounded));
        else result = static_cast<uint64_t>(static_cast<uint32_t>(sat_to_int32(rounded)));
      } else {
        if (args.sf) result = sat_to_uint64(rounded);
        else result = sat_to_uint32(rounded);
      }
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = args.sf ? result : (result & 0xFFFFFFFFULL);
      }
      return;
    }

    // (FJCVTZS — Armv8.3-JSCVT)
    // FJCVTZS Wd, Dn: convert double-precision FP to 32-bit signed integer
    // using ECMAScript ToInt32 semantics (round-toward-zero + modular reduction).
    // Encoding: ftype=01 (double), rmode=11, opcode=110, sf=0.
    // Semantics (ARM ARM C7.2.110): NaN/±Inf → 0; truncate toward zero, then
    // map result modulo 2^32 to signed range [-2^31, 2^31).  Sets PSTATE.Z=1
    // iff the conversion was exact (input was an integer-valued finite double
    // in [INT32_MIN, INT32_MAX]); N=C=V=0 always.
    if (rmode == 0b11 && opcode == 0b110 && args.ftype == 0b01 && !args.sf) {
      double d;
      memcpy(&d, &state_->cpu.v[args.rn], 8);

      uint32_t result;
      bool exact;
      if (std::isnan(d) || std::isinf(d)) {
        result = 0;
        exact = false;
      } else {
        double t = std::trunc(d);
        if (t >= static_cast<double>(INT32_MIN) && t <= static_cast<double>(INT32_MAX)) {
          result = static_cast<uint32_t>(static_cast<int32_t>(t));
          exact = (t == d);
        } else {
          // ECMAScript ToInt32 modular reduction.  fmod gives |t| mod 2^32 as
          // a non-negative double; cast to uint32_t to get the low 32 bits,
          // then negate for the sign-preserving signed mapping.
          double abs_t = std::fabs(t);
          double mod_v = std::fmod(abs_t, 4294967296.0);
          uint32_t u = static_cast<uint32_t>(mod_v);
          if (std::signbit(t)) {
            u = static_cast<uint32_t>(-static_cast<int64_t>(u));
          }
          result = u;
          exact = false;
        }
      }

      uint32_t flags = 0;
      if (exact) flags |= CPUState::kFlagZero;
      state_->cpu.flags = flags;

      if (args.rd < 31) {
        // Zero-extend the 32-bit result into Xd (Wd write, upper 32 bits zero).
        state_->cpu.x[args.rd] = static_cast<uint64_t>(result);
      }
      return;
    }

    if (rmode == 0b11 && opcode == 0b000) {
      // FCVTZS: FP to signed integer, round toward zero (truncate).  Use the
      // ARM-saturation helpers defined above instead of raw static_cast to
      // match real ARM hardware on NaN / out-of-range inputs.
      uint64_t result = 0;
      double rounded;
      if (args.ftype == 0b00) {
        float f; memcpy(&f, &state_->cpu.v[args.rn], 4);
        rounded = trunc(static_cast<double>(f));
      } else if (args.ftype == 0b01) {
        double d; memcpy(&d, &state_->cpu.v[args.rn], 8);
        rounded = trunc(d);
      } else if (args.ftype == 0b11) {
        uint16_t h; memcpy(&h, &state_->cpu.v[args.rn], 2);
        rounded = trunc(static_cast<double>(FpHalfToSingle(h)));
      } else { Undefined(); return; }
      if (args.sf) result = static_cast<uint64_t>(sat_to_int64(rounded));
      else result = static_cast<uint64_t>(static_cast<uint32_t>(sat_to_int32(rounded)));
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = args.sf ? result : (result & 0xFFFFFFFFULL);
      }
      return;
    }

    if (rmode == 0b11 && opcode == 0b001) {
      // FCVTZU: FP to unsigned integer, round toward zero (truncate).
      uint64_t result = 0;
      double rounded;
      if (args.ftype == 0b00) {
        float f; memcpy(&f, &state_->cpu.v[args.rn], 4);
        rounded = trunc(static_cast<double>(f));
      } else if (args.ftype == 0b01) {
        double d; memcpy(&d, &state_->cpu.v[args.rn], 8);
        rounded = trunc(d);
      } else if (args.ftype == 0b11) {
        uint16_t h; memcpy(&h, &state_->cpu.v[args.rn], 2);
        rounded = trunc(static_cast<double>(FpHalfToSingle(h)));
      } else { Undefined(); return; }
      if (args.sf) result = sat_to_uint64(rounded);
      else result = sat_to_uint32(rounded);
      if (args.rd < 31) {
        state_->cpu.x[args.rd] = args.sf ? result : (result & 0xFFFFFFFFULL);
      }
      return;
    }

    Undefined();
  }

  void LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& args, Register base) {
    CHECK(!exception_raised_);
    void* host_addr = ToHostAddr<void>(ApplyTbi(base));
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

      // CASP (compare-and-swap pair, Armv8.1 LSE).
      // CASP <Ws>,<Ws+1>,<Wt>,<Wt+1>,[<Xn>]: compare the pair at [Xn] against
      // {Rs+1, Rs}; if equal store {Rt+1, Rt}. The actual prior pair is
      // written back into {Rs+1, Rs}. ARM ARM C7.2.40.
      // args.size: 2 = 32-bit pair (8 bytes), 3 = 64-bit pair (16 bytes).
      // Rs and Rt must be even per the ARM encoding (UNPREDICTABLE otherwise);
      // the decoder doesn't reject odd registers — we just follow the same
      // permissive policy as bare ARM cores and read whatever Rs/Rs+1 are.
      case Decoder::AtomicOp::kCasp: {
        uint8_t rs_lo = args.rs;
        uint8_t rs_hi = args.rs + 1;
        uint8_t rt_lo = args.rt;
        uint8_t rt_hi = args.rt + 1;
        if (args.size == 2) {
          // 32-bit pair: pack lo/hi halves into one 64-bit CAS.
          uint64_t exp_lo = (rs_lo < 31) ? (state_->cpu.x[rs_lo] & 0xFFFFFFFF) : 0;
          uint64_t exp_hi = (rs_hi < 31) ? (state_->cpu.x[rs_hi] & 0xFFFFFFFF) : 0;
          uint64_t new_lo = (rt_lo < 31) ? (state_->cpu.x[rt_lo] & 0xFFFFFFFF) : 0;
          uint64_t new_hi = (rt_hi < 31) ? (state_->cpu.x[rt_hi] & 0xFFFFFFFF) : 0;
          uint64_t expected = (exp_hi << 32) | exp_lo;
          uint64_t desired  = (new_hi << 32) | new_lo;
          uint64_t old_pair = AtomicCASVal<uint64_t>(host_addr, expected, desired);
          // Each half written back zero-extended to the X register.
          if (rs_lo < 31) state_->cpu.x[rs_lo] = old_pair & 0xFFFFFFFF;
          if (rs_hi < 31) state_->cpu.x[rs_hi] = (old_pair >> 32) & 0xFFFFFFFF;
        } else {  // args.size == 3
          // 64-bit pair: 128-bit compare-and-swap via AtomicCASVal128
          // (inline-asm LOCK CMPXCHG16B; avoids the libatomic libcall path
          // that bare `__atomic_compare_exchange_n` on __uint128_t may take
          // when -mcx16 isn't set on the translation unit).
          uint64_t exp_lo = (rs_lo < 31) ? state_->cpu.x[rs_lo] : 0;
          uint64_t exp_hi = (rs_hi < 31) ? state_->cpu.x[rs_hi] : 0;
          uint64_t new_lo = (rt_lo < 31) ? state_->cpu.x[rt_lo] : 0;
          uint64_t new_hi = (rt_hi < 31) ? state_->cpu.x[rt_hi] : 0;
          __uint128_t expected =
              (static_cast<__uint128_t>(exp_hi) << 64) | exp_lo;
          __uint128_t desired =
              (static_cast<__uint128_t>(new_hi) << 64) | new_lo;
          __uint128_t old_pair =
              AtomicCASVal128(host_addr, expected, desired);
          if (rs_lo < 31) state_->cpu.x[rs_lo] = static_cast<uint64_t>(old_pair);
          if (rs_hi < 31) state_->cpu.x[rs_hi] = static_cast<uint64_t>(old_pair >> 64);
        }
        break;
      }

      // LDXP/LDAXP (load-exclusive pair). Loads a register pair atomically and
      // arms the exclusive monitor (modeled as an address reservation, like
      // LDXR). args.size: 2 = 32-bit pair (8 bytes), 3 = 64-bit pair (16 bytes).
      case Decoder::AtomicOp::kLdxp: {
        if (args.size == 2) {
          uint64_t pair = AtomicLoad<uint64_t>(host_addr);
          if (args.rt < 31) state_->cpu.x[args.rt] = pair & 0xFFFFFFFF;
          if (args.rt2 < 31) state_->cpu.x[args.rt2] = (pair >> 32) & 0xFFFFFFFF;
          state_->cpu.reservation_address = base;
          memcpy(&state_->cpu.reservation_value, &pair, sizeof(pair));
        } else {  // args.size == 3
          // CMPXCHG16B with expected==desired==0 is an atomic 128-bit read
          // (only writes when *addr==0, which leaves 0 in place).
          __uint128_t pair = AtomicCASVal128(host_addr, 0, 0);
          if (args.rt < 31) state_->cpu.x[args.rt] = static_cast<uint64_t>(pair);
          if (args.rt2 < 31) state_->cpu.x[args.rt2] = static_cast<uint64_t>(pair >> 64);
          state_->cpu.reservation_address = base;
          // reservation_value is __uint128_t, so retain the FULL 128-bit value
          // for STXP's exact-monitor compare-and-swap below.
          state_->cpu.reservation_value = pair;
        }
        break;
      }

      // STXP/STLXP (store-exclusive pair). Writes Rt:Rt2 atomically and reports
      // success (0) / failure (1) in Rs. Both widths use an exact monitor: a
      // single compare-and-swap against the value LDXP read, so a concurrent
      // modification since LDXP makes the CAS (and thus the STXP) fail and
      // leaves memory untouched — the LL/SC contract lock-free code depends on.
      case Decoder::AtomicOp::kStxp: {
        bool success = false;
        if (args.size == 2) {
          uint64_t new_lo = (args.rt < 31) ? (state_->cpu.x[args.rt] & 0xFFFFFFFF) : 0;
          uint64_t new_hi = (args.rt2 < 31) ? (state_->cpu.x[args.rt2] & 0xFFFFFFFF) : 0;
          uint64_t new_pair = (new_hi << 32) | new_lo;
          uint64_t expected;
          memcpy(&expected, &state_->cpu.reservation_value, sizeof(expected));
          if (state_->cpu.reservation_address == base) {
            success = AtomicCAS<uint64_t>(host_addr, expected, new_pair);
          }
        } else {  // args.size == 3
          if (state_->cpu.reservation_address == base) {
            uint64_t new_lo = (args.rt < 31) ? state_->cpu.x[args.rt] : 0;
            uint64_t new_hi = (args.rt2 < 31) ? state_->cpu.x[args.rt2] : 0;
            __uint128_t desired = (static_cast<__uint128_t>(new_hi) << 64) | new_lo;
            __uint128_t expected = state_->cpu.reservation_value;
            // CMPXCHG16B: stores desired and returns expected iff *addr ==
            // expected; otherwise returns the current value and stores nothing.
            __uint128_t prev = AtomicCASVal128(host_addr, expected, desired);
            success = (prev == expected);
          }
        }
        state_->cpu.reservation_address = 0;
        if (args.rs < 31) state_->cpu.x[args.rs] = success ? 0 : 1;
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

      // atomic min/max (LSE Armv8.1).
      // x86 has no single-instruction equivalent; AtomicFetchSMax/SMin/UMax/UMin
      // implement these via __atomic_compare_exchange retry loops.
      case Decoder::AtomicOp::kLdsmax: {
        uint64_t operand = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicFetchSMax<int8_t>(host_addr, operand); break;
          case 1: old_val = AtomicFetchSMax<int16_t>(host_addr, operand); break;
          case 2: old_val = AtomicFetchSMax<int32_t>(host_addr, operand); break;
          case 3: old_val = AtomicFetchSMax<int64_t>(host_addr, operand); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }

      case Decoder::AtomicOp::kLdsmin: {
        uint64_t operand = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicFetchSMin<int8_t>(host_addr, operand); break;
          case 1: old_val = AtomicFetchSMin<int16_t>(host_addr, operand); break;
          case 2: old_val = AtomicFetchSMin<int32_t>(host_addr, operand); break;
          case 3: old_val = AtomicFetchSMin<int64_t>(host_addr, operand); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }

      case Decoder::AtomicOp::kLdumax: {
        uint64_t operand = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicFetchUMax<uint8_t>(host_addr, operand); break;
          case 1: old_val = AtomicFetchUMax<uint16_t>(host_addr, operand); break;
          case 2: old_val = AtomicFetchUMax<uint32_t>(host_addr, operand); break;
          case 3: old_val = AtomicFetchUMax<uint64_t>(host_addr, operand); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }

      case Decoder::AtomicOp::kLdumin: {
        uint64_t operand = (args.rs < 31) ? state_->cpu.x[args.rs] : 0;
        uint64_t old_val = 0;
        switch (args.size) {
          case 0: old_val = AtomicFetchUMin<uint8_t>(host_addr, operand); break;
          case 1: old_val = AtomicFetchUMin<uint16_t>(host_addr, operand); break;
          case 2: old_val = AtomicFetchUMin<uint32_t>(host_addr, operand); break;
          case 3: old_val = AtomicFetchUMin<uint64_t>(host_addr, operand); break;
        }
        if (args.rt < 31) state_->cpu.x[args.rt] = old_val;
        break;
      }
    }
  }

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

      // scalar SIMD copy (DUP scalar / MOV Vd, Vn[index])
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
        ClearUpperIfNotQ(&result, args.q);
        break;
      case Decoder::AdvSimdThreeSameOpcode::kBic:
        result = src_n & ~src_m;
        ClearUpperIfNotQ(&result, args.q);
        break;
      case Decoder::AdvSimdThreeSameOpcode::kOrr:
        result = src_n | src_m;
        ClearUpperIfNotQ(&result, args.q);
        break;
      case Decoder::AdvSimdThreeSameOpcode::kOrn:
        result = src_n | ~src_m;
        ClearUpperIfNotQ(&result, args.q);
        break;
      case Decoder::AdvSimdThreeSameOpcode::kEor:
        result = src_n ^ src_m;
        ClearUpperIfNotQ(&result, args.q);
        break;
      case Decoder::AdvSimdThreeSameOpcode::kBsl:
        // BSL: Vd = (Vd & Vn) | (~Vd & Vm) — bitwise select using Vd as mask.
        result = (dst & src_n) | (~dst & src_m);
        ClearUpperIfNotQ(&result, args.q);
        break;
      case Decoder::AdvSimdThreeSameOpcode::kBit:
        // BIT: Vd = (Vm & Vn) | (~Vm & Vd) — insert bits where Vm is 1.
        result = (src_m & src_n) | (~src_m & dst);
        ClearUpperIfNotQ(&result, args.q);
        break;
      case Decoder::AdvSimdThreeSameOpcode::kBif:
        // BIF: Vd = (Vm & Vd) | (~Vm & Vn) — insert bits where Vm is 0.
        result = (src_m & dst) | (~src_m & src_n);
        ClearUpperIfNotQ(&result, args.q);
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

      // --- Compare: CMGT, CMHI, CMGE, CMHS ---
      case Decoder::AdvSimdThreeSameOpcode::kCmgt:
        // CMGT (signed >): if (Vn[i] > Vm[i]) signed, result = all-ones.
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              uint64_t mask = (es >= 8) ? ~0ULL : ((1ULL << (es * 8)) - 1);
              uint8_t bits = es * 8;
              int64_t sa = SignExtendElem(a, bits);
              int64_t sb = SignExtendElem(b, bits);
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
              int64_t sa = SignExtendElem(a, bits);
              int64_t sb = SignExtendElem(b, bits);
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

      // SABD/UABD: absolute-difference vector.
      // Signed: |a - b| computed via the sign of the difference; for
      // INT_MIN inputs the difference fits in int64_t after sign-extension
      // because esize is at most 4 (32-bit lanes).
      case Decoder::AdvSimdThreeSameOpcode::kSabd:
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [](int64_t a, int64_t b) -> int64_t {
              int64_t d = a - b;
              return d < 0 ? -d : d;
            });
        break;
      case Decoder::AdvSimdThreeSameOpcode::kUabd:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t {
              return a > b ? (a - b) : (b - a);
            });
        break;

      // SABA/UABA: absolute-difference-and-accumulate.
      // Vd[i] = Vd[i] + |Vn[i] - Vm[i]|.  Read-modify-write Vd, so the loop
      // is structured like kMla/kMls (loads `d` from `dst` per lane).
      // size=11 is rejected at the decoder; esize <= 4 here, so the signed
      // difference fits in int64_t after sign-extension and the absolute
      // value is computed without overflow.
      case Decoder::AdvSimdThreeSameOpcode::kSaba: {
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a = 0, b = 0, d = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * esize, esize);
          memcpy(&d, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          int64_t sa = SignExtendElem(a, bits);
          int64_t sb = SignExtendElem(b, bits);
          int64_t diff = sa - sb;
          uint64_t abs_diff = static_cast<uint64_t>(diff < 0 ? -diff : diff);
          uint64_t r = (d + abs_diff) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }
      case Decoder::AdvSimdThreeSameOpcode::kUaba: {
        uint64_t emask = ElementMask(esize);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a = 0, b = 0, d = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src_n) + i * esize, esize);
          memcpy(&b, reinterpret_cast<const uint8_t*>(&src_m) + i * esize, esize);
          memcpy(&d, reinterpret_cast<const uint8_t*>(&dst) + i * esize, esize);
          uint64_t abs_diff = a > b ? (a - b) : (b - a);
          uint64_t r = (d + abs_diff) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

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
                int64_t sa = SignExtendElem(a, bits);
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

      // --- Variable saturating / rounding shifts (SQSHL/UQSHL/SRSHL/URSHL/
      //     SQRSHL/UQRSHL). The per-lane shift count is the signed low byte of
      //     the Vm element: >=0 shifts left, <0 shifts right by its magnitude.
      //     Signed (S*) ops sign-extend the source and shift right arithmetically;
      //     rounding (R*) ops add 1<<(rshift-1) before a right shift; saturating
      //     (Q*) ops clamp to the element's signed/unsigned range. ---
      case Decoder::AdvSimdThreeSameOpcode::kSqshl:
      case Decoder::AdvSimdThreeSameOpcode::kUqshl:
      case Decoder::AdvSimdThreeSameOpcode::kSrshl:
      case Decoder::AdvSimdThreeSameOpcode::kUrshl:
      case Decoder::AdvSimdThreeSameOpcode::kSqrshl:
      case Decoder::AdvSimdThreeSameOpcode::kUqrshl: {
        const bool is_signed = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqshl ||
                                args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrshl ||
                                args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrshl);
        const bool rounding = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrshl ||
                               args.opcode == Decoder::AdvSimdThreeSameOpcode::kUrshl ||
                               args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrshl ||
                               args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqrshl);
        const bool saturating = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqshl ||
                                 args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqshl ||
                                 args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrshl ||
                                 args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqrshl);
        AdvSimdThreeSameElementWise(
            src_n, src_m, esize, num_elements, &result,
            [is_signed, rounding, saturating](uint64_t a, uint64_t b, uint8_t es) -> uint64_t {
              const uint32_t bits = es * 8;
              const int32_t shift = static_cast<int8_t>(b & 0xFF);
              // Sign- or zero-extend the element into a wide signed accumulator.
              __int128 v;
              if (is_signed) {
                v = static_cast<__int128>(SignExtendElem(a, bits));
              } else {
                v = static_cast<__int128>(a & (bits >= 64 ? ~0ULL : ((1ULL << bits) - 1)));
              }
              __int128 res;
              if (shift >= 0) {
                const uint32_t lshift = static_cast<uint32_t>(shift);
                // A shift >= element width moves every bit out of the lane: for
                // non-saturating ops the masked result is 0; for saturating ops
                // a non-zero source pins to the range extreme (a large sentinel
                // clamped below). Capping at `bits` also keeps v<<lshift inside
                // the 128-bit accumulator (v is at most `bits`<=64 wide).
                if (lshift >= bits) {
                  res = saturating ? (v > 0 ? (__int128{1} << 100)
                                            : (v < 0 ? -(__int128{1} << 100) : __int128{0}))
                                   : __int128{0};
                } else {
                  res = v << lshift;
                }
              } else {
                const uint32_t rshift = static_cast<uint32_t>(-shift);
                if (rounding && rshift >= 1 && rshift <= 127) {
                  v += (__int128{1} << (rshift - 1));
                }
                // A right shift by >= 128 (only reachable at shift == -128)
                // moves every source bit out of the lane. For non-rounding ops
                // the arithmetic result is the sign broadcast (all-ones for a
                // negative signed source, 0 otherwise). For rounding ops the
                // round constant 1<<(rshift-1) == 1<<127 dominates the small
                // (<2^63) source and lifts the pre-shift value into
                // [0, 2^rshift), so the rounded result floors to 0 for both
                // signs. The round-add above is skipped at rshift == 128 only
                // because 1<<127 overflows a signed __int128, so account for
                // its effect (result 0) directly here rather than falling into
                // the non-rounding sign-broadcast (which gave the wrong -1 for a
                // negative SRSHL/SQRSHL source).
                res = (rshift >= 128)
                          ? (rounding ? __int128{0}
                                      : (is_signed && v < 0 ? __int128{-1} : __int128{0}))
                          : (v >> rshift);  // arithmetic for signed v
              }
              if (saturating) {
                if (is_signed) {
                  const __int128 maxv = (__int128{1} << (bits - 1)) - 1;
                  const __int128 minv = -(__int128{1} << (bits - 1));
                  res = res > maxv ? maxv : (res < minv ? minv : res);
                } else {
                  const __int128 maxv =
                      (bits >= 64) ? static_cast<__int128>(~0ULL) : ((__int128{1} << bits) - 1);
                  res = res < 0 ? __int128{0} : (res > maxv ? maxv : res);
                }
              }
              const uint64_t mask = (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1);
              return static_cast<uint64_t>(res) & mask;
            });
        break;
      }

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

      // --- MUL / MLA / MLS ---
      case Decoder::AdvSimdThreeSameOpcode::kMul:
        AdvSimdThreeSameElementWise(src_n, src_m, esize, num_elements, &result,
            [](uint64_t a, uint64_t b, uint8_t /*esize*/) -> uint64_t { return a * b; });
        break;
      // PMUL polynomial multiply (byte lanes).
      // For each byte lane: multiply two 8-bit polynomials over GF(2),
      // keeping only the low 8 bits of the product. The decoder restricts
      // PMUL to size=00, so esize is always 1 byte here.
      case Decoder::AdvSimdThreeSameOpcode::kPmul: {
        auto poly_mul_low8 = [](uint8_t a, uint8_t b) -> uint8_t {
          uint16_t res = 0;
          for (unsigned i = 0; i < 8; ++i) {
            if ((b >> i) & 1u) {
              res ^= (static_cast<uint16_t>(a) << i);
            }
          }
          return static_cast<uint8_t>(res);
        };
        for (uint8_t i = 0; i < num_elements; i++) {
          uint8_t a = reinterpret_cast<const uint8_t*>(&src_n)[i];
          uint8_t b = reinterpret_cast<const uint8_t*>(&src_m)[i];
          reinterpret_cast<uint8_t*>(&result)[i] = poly_mul_low8(a, b);
        }
        break;
      }
      // SQDMULH / SQRDMULH saturating doubling multiply high.
      // For each lane: signed multiply two same-width source elements, double
      // (shift left by 1), saturate to the per-element signed range, and
      // return the high half. SQRDMULH adds a rounding constant of
      // 1 << (esize_bits - 1) before the shift; SQDMULH uses 0. The decoder
      // restricts both to size=01 (16-bit) and size=10 (32-bit). For size=S
      // (32-bit) the doubled product 2 * INT32_MIN * INT32_MIN = 2^63 overflows
      // int64_t by one, so the multiply is done in __int128 — same pattern as
      // the scalar SQDMULH/SQRDMULH path below.
      case Decoder::AdvSimdThreeSameOpcode::kSqdmulh:
      case Decoder::AdvSimdThreeSameOpcode::kSqrdmulh: {
        uint8_t bits_local = esize * 8;
        int64_t smax = (1LL << (bits_local - 1)) - 1;
        int64_t smin = -(1LL << (bits_local - 1));
        __int128 round = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrdmulh)
                          ? (static_cast<__int128>(1) << (bits_local - 1))
                          : __int128{0};
        AdvSimdThreeSameElementWiseSigned(src_n, src_m, esize, num_elements, &result,
            [smax, smin, round, bits_local](int64_t a, int64_t b) -> int64_t {
              __int128 product = (static_cast<__int128>(2) * a * b) + round;
              int64_t high = static_cast<int64_t>(product >> bits_local);
              if (high > smax) return smax;
              if (high < smin) return smin;
              return high;
            });
        break;
      }
      // Armv8.1-RDM SQRDMLAH / SQRDMLSH three-same vector.
      // Decoder restricts size to {01, 10}.  Stage 1 reuses the SQRDMULH
      // recipe per lane (doubled product + rounding constant, shifted by
      // bits_local, saturated to the signed narrow range).  Stage 2
      // signed-saturating-adds (SQRDMLAH) or subtracts (SQRDMLSH) the addend
      // into the lane's existing Vd value.  Same math as the by-element form
      // in InsnConsumer::AdvSimdVecXIndexedElement at the kSqrdmlahIdx /
      // kSqrdmlshIdx arm; only the Vm source changes from a broadcasted
      // lane to per-lane Vm[i].
      case Decoder::AdvSimdThreeSameOpcode::kSqrdmlahVec:
      case Decoder::AdvSimdThreeSameOpcode::kSqrdmlshVec: {
        uint8_t bits_local = esize * 8;
        int64_t smax = (1LL << (bits_local - 1)) - 1;
        int64_t smin = -(1LL << (bits_local - 1));
        __int128 round = static_cast<__int128>(1) << (bits_local - 1);
        bool is_sub = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrdmlshVec);
        uint8_t shift = (8 - esize) * 8;

        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t a_u = 0, b_u = 0;
          memcpy(&a_u,
                 reinterpret_cast<const uint8_t*>(&src_n) + i * esize,
                 esize);
          memcpy(&b_u,
                 reinterpret_cast<const uint8_t*>(&src_m) + i * esize,
                 esize);
          int64_t a = static_cast<int64_t>(a_u << shift) >> shift;
          int64_t b = static_cast<int64_t>(b_u << shift) >> shift;

          // Stage 1: SQRDMULH(a, b).  2*a*b fits int64_t for esize<=4.
          __int128 product = (static_cast<__int128>(2) * a * b) + round;
          int64_t addend = static_cast<int64_t>(product >> bits_local);
          if (addend > smax) addend = smax;
          if (addend < smin) addend = smin;

          // Stage 2: signed-saturating add/sub into Vd[i].
          uint64_t acc_u = 0;
          memcpy(&acc_u,
                 reinterpret_cast<const uint8_t*>(&dst) + i * esize,
                 esize);
          int64_t acc = static_cast<int64_t>(acc_u << shift) >> shift;
          int64_t out = is_sub ? acc - addend : acc + addend;
          if (out > smax) out = smax;
          if (out < smin) out = smin;
          uint64_t out_u = static_cast<uint64_t>(out);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &out_u, esize);
        }
        ClearUpperIfNotQ(&result, args.q);
        break;
      }
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

      // --- FP pairwise three-same: FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP ---
      // Reduce adjacent element pairs.  The result's low half comes from Vn's
      // pairs and the high half from Vm's pairs, i.e. result[k] reduces the
      // pair (cat[2k], cat[2k+1]) of the concatenation cat = Vn:Vm.  All FP
      // precisions (.4h/.8h via FP16 round-trip; .2s/.4s/.2d).
      case Decoder::AdvSimdThreeSameOpcode::kFaddpV:
      case Decoder::AdvSimdThreeSameOpcode::kFmaxpV:
      case Decoder::AdvSimdThreeSameOpcode::kFminpV:
      case Decoder::AdvSimdThreeSameOpcode::kFmaxnmpV:
      case Decoder::AdvSimdThreeSameOpcode::kFminnmpV: {
        auto reduce_f = [&](float a, float b) -> float {
          switch (args.opcode) {
            case Decoder::AdvSimdThreeSameOpcode::kFaddpV:   return a + b;
            case Decoder::AdvSimdThreeSameOpcode::kFmaxpV:   return FmaxScalar<float>(a, b);
            case Decoder::AdvSimdThreeSameOpcode::kFminpV:   return FminScalar<float>(a, b);
            case Decoder::AdvSimdThreeSameOpcode::kFmaxnmpV: return FmaxnmScalar<float>(a, b);
            default:                                         return FminnmScalar<float>(a, b);
          }
        };
        auto reduce_d = [&](double a, double b) -> double {
          switch (args.opcode) {
            case Decoder::AdvSimdThreeSameOpcode::kFaddpV:   return a + b;
            case Decoder::AdvSimdThreeSameOpcode::kFmaxpV:   return FmaxScalar<double>(a, b);
            case Decoder::AdvSimdThreeSameOpcode::kFminpV:   return FminScalar<double>(a, b);
            case Decoder::AdvSimdThreeSameOpcode::kFmaxnmpV: return FmaxnmScalar<double>(a, b);
            default:                                         return FminnmScalar<double>(a, b);
          }
        };
        if (args.is_fp16) {
          uint8_t lanes = args.q ? 8 : 4;
          uint16_t cat[16];
          for (uint8_t k = 0; k < lanes; k++) {
            memcpy(&cat[k], reinterpret_cast<const uint8_t*>(&src_n) + k * 2, 2);
            memcpy(&cat[lanes + k], reinterpret_cast<const uint8_t*>(&src_m) + k * 2, 2);
          }
          for (uint8_t k = 0; k < lanes; k++) {
            float a = FpHalfToSingle(cat[2 * k]);
            float b = FpHalfToSingle(cat[2 * k + 1]);
            uint16_t rh = FpSingleToHalf(reduce_f(a, b));
            memcpy(reinterpret_cast<uint8_t*>(&result) + k * 2, &rh, 2);
          }
          break;
        }
        if (args.size == 0b01) {  // double (.2d)
          uint8_t lanes = vec_len / 8;
          double cat[4];
          for (uint8_t k = 0; k < lanes; k++) {
            memcpy(&cat[k], reinterpret_cast<const uint8_t*>(&src_n) + k * 8, 8);
            memcpy(&cat[lanes + k], reinterpret_cast<const uint8_t*>(&src_m) + k * 8, 8);
          }
          for (uint8_t k = 0; k < lanes; k++) {
            double r = reduce_d(cat[2 * k], cat[2 * k + 1]);
            memcpy(reinterpret_cast<uint8_t*>(&result) + k * 8, &r, 8);
          }
        } else {  // single (.2s / .4s)
          uint8_t lanes = vec_len / 4;
          float cat[8];
          for (uint8_t k = 0; k < lanes; k++) {
            memcpy(&cat[k], reinterpret_cast<const uint8_t*>(&src_n) + k * 4, 4);
            memcpy(&cat[lanes + k], reinterpret_cast<const uint8_t*>(&src_m) + k * 4, 4);
          }
          for (uint8_t k = 0; k < lanes; k++) {
            float r = reduce_f(cat[2 * k], cat[2 * k + 1]);
            memcpy(reinterpret_cast<uint8_t*>(&result) + k * 4, &r, 4);
          }
        }
        break;
      }

      // --- FP three-same vector ops (Digitalis addition) ---
      // For FP cases args.size is sz alone (0 = single 32-bit, 1 = double 64-bit),
      // not the {op_high, sz} pair the raw encoding carries; the decoder
      // already split that. For the Armv8.2-FP16 NEON encoding the
      // decoder sets args.is_fp16=true and lanes are 2-byte half (see the
      // FP16 branch below).
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
      case Decoder::AdvSimdThreeSameOpcode::kFabdV:
      case Decoder::AdvSimdThreeSameOpcode::kFmulxV:
      case Decoder::AdvSimdThreeSameOpcode::kFrecpsV:
      case Decoder::AdvSimdThreeSameOpcode::kFrsqrtsV:
        {
        // FP16 vector lanes via float round-trip.
        // Promote each 2-byte half to binary32, do the op in binary32 (which
        // is exact for any single FP16 op because binary32's 24-bit mantissa
        // strictly contains binary16's 11), then narrow back to half.  FMA
        // uses binary64 to keep the multiply-add exact before the single
        // narrow to half. Reuses FpHalfToSingle / FpSingleToHalf from.
        if (args.is_fp16) {
          uint8_t fp_num = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_num; i++) {
            uint16_t hn, hm, hd;
            memcpy(&hn, reinterpret_cast<const uint8_t*>(&src_n) + i * 2, 2);
            memcpy(&hm, reinterpret_cast<const uint8_t*>(&src_m) + i * 2, 2);
            memcpy(&hd, reinterpret_cast<const uint8_t*>(&dst) + i * 2, 2);
            float a = FpHalfToSingle(hn);
            float b = FpHalfToSingle(hm);
            float d = FpHalfToSingle(hd);
            uint16_t rh;
            switch (args.opcode) {
              case Decoder::AdvSimdThreeSameOpcode::kFaddV:
                rh = FpSingleToHalf(a + b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFsubV:
                rh = FpSingleToHalf(a - b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmulV:
                rh = FpSingleToHalf(a * b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmulxV:
                rh = FpSingleToHalf(FmulxScalar<float>(a, b)); break;
              case Decoder::AdvSimdThreeSameOpcode::kFdivV:
                rh = FpSingleToHalf(a / b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmlaV: {
                // FMLA: Vd[i] = Vd[i] + Vn[i] * Vm[i].  Use binary64 fma() so
                // the multiply-add has no intermediate rounding before the
                // single narrow back to half.
                double r64 = std::fma(static_cast<double>(a), static_cast<double>(b),
                                      static_cast<double>(d));
                rh = FpSingleToHalf(static_cast<float>(r64));
                break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFmlsV: {
                double r64 = std::fma(static_cast<double>(-a), static_cast<double>(b),
                                      static_cast<double>(d));
                rh = FpSingleToHalf(static_cast<float>(r64));
                break;
              }
              case Decoder::AdvSimdThreeSameOpcode::kFmaxV:
                rh = FpSingleToHalf(FmaxScalar<float>(a, b)); break;
              case Decoder::AdvSimdThreeSameOpcode::kFminV:
                rh = FpSingleToHalf(FminScalar<float>(a, b)); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmaxnmV:
                rh = FpSingleToHalf(FmaxnmScalar<float>(a, b)); break;
              case Decoder::AdvSimdThreeSameOpcode::kFminnmV:
                rh = FpSingleToHalf(FminnmScalar<float>(a, b)); break;
              case Decoder::AdvSimdThreeSameOpcode::kFcmeqV:
                rh = (a == b) ? uint16_t{0xFFFF} : uint16_t{0};
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFcmgeV:
                rh = (a >= b) ? uint16_t{0xFFFF} : uint16_t{0};
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFcmgtV:
                rh = (a > b) ? uint16_t{0xFFFF} : uint16_t{0};
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFacgeV:
                rh = (std::fabs(a) >= std::fabs(b)) ? uint16_t{0xFFFF} : uint16_t{0};
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFacgtV:
                rh = (std::fabs(a) > std::fabs(b)) ? uint16_t{0xFFFF} : uint16_t{0};
                break;
              case Decoder::AdvSimdThreeSameOpcode::kFabdV:
                rh = FpSingleToHalf(std::fabs(a - b)); break;
              // FP16 vector FRECPS / FRSQRTS via FP32.
              case Decoder::AdvSimdThreeSameOpcode::kFrecpsV:
                rh = FpSingleToHalf(FrecpsScalar<float>(a, b)); break;
              case Decoder::AdvSimdThreeSameOpcode::kFrsqrtsV:
                rh = FpSingleToHalf(FrsqrtsScalar<float>(a, b)); break;
              default: Undefined(); return;
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &rh, 2);
          }
          break;
        }
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
              case Decoder::AdvSimdThreeSameOpcode::kFmulxV:
                r = FmulxScalar<double>(a, b); break;
              // FMLA/FMLS are FUSED multiply-add (single rounding) per the ARM
              // ARM; `d + a*b` would round twice. Use std::fma so coordinate
              // transforms (matrix*vector via FMLA) match hardware exactly.
              case Decoder::AdvSimdThreeSameOpcode::kFmlaV: r = std::fma(a, b, d); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmlsV: r = std::fma(-a, b, d); break;
              // FMAX/FMIN/FMAXNM/FMINNM: see FmaxScalar/FminScalar helpers for
              // NaN propagation and +0/-0 sign disambiguation.
              case Decoder::AdvSimdThreeSameOpcode::kFmaxV:
                r = FmaxScalar<double>(a, b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFminV:
                r = FminScalar<double>(a, b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmaxnmV:
                r = FmaxnmScalar<double>(a, b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFminnmV:
                r = FminnmScalar<double>(a, b); break;
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
              case Decoder::AdvSimdThreeSameOpcode::kFrecpsV:
                r = FrecpsScalar<double>(a, b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFrsqrtsV:
                r = FrsqrtsScalar<double>(a, b); break;
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
              case Decoder::AdvSimdThreeSameOpcode::kFmulxV:
                r = FmulxScalar<float>(a, b); break;
              // FUSED multiply-add (single rounding) per the ARM ARM — std::fma,
              // not `d + a*b` (which rounds twice).
              case Decoder::AdvSimdThreeSameOpcode::kFmlaV: r = std::fma(a, b, d); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmlsV: r = std::fma(-a, b, d); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmaxV:
                r = FmaxScalar<float>(a, b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFminV:
                r = FminScalar<float>(a, b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFmaxnmV:
                r = FmaxnmScalar<float>(a, b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFminnmV:
                r = FminnmScalar<float>(a, b); break;
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
              case Decoder::AdvSimdThreeSameOpcode::kFrecpsV:
                r = FrecpsScalar<float>(a, b); break;
              case Decoder::AdvSimdThreeSameOpcode::kFrsqrtsV:
                r = FrsqrtsScalar<float>(a, b); break;
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

  //
  // FP data-processing (1 source): FMOV, FABS, FNEG, FSQRT, FCVT, FRINTx.
  //
  // FRINTTS (FEAT_FRINTTS) shared rounding. Round `src` to
  // an integral value (toward zero for the Z variants, FPCR rounding mode via
  // rint for the X variants), then saturate to the signed 32- or 64-bit range.
  // Out-of-range inputs, NaN and infinities yield the most-negative value and
  // set FPSR.IOC, matching the ARM ARM FPRoundIntN definition. opcode bit0
  // selects X(1)/Z(0); bit1 selects 64-bit(1)/32-bit(0).
  double FrintTs(double src, uint8_t opcode) {
    const bool toward_zero = (opcode & 0b1u) == 0;
    const bool is64 = (opcode & 0b10u) != 0;
    double r = toward_zero ? std::trunc(src) : std::rint(src);
    const double lo = is64 ? -9223372036854775808.0 : -2147483648.0;
    const double hi = is64 ? 9223372036854775808.0 : 2147483648.0;
    if (!(r >= lo && r < hi)) {
      state_->cpu.emulated_fpsr |= 1u;  // FPSR.IOC (invalid operation)
      return lo;
    }
    return r;
  }

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
        // FRINTTS scalar (FP32).
        case 0b010000:  // FRINT32Z
        case 0b010001:  // FRINT32X
        case 0b010010:  // FRINT64Z
        case 0b010011:  // FRINT64X
          result = static_cast<float>(FrintTs(static_cast<double>(src), opcode));
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
        case 0b000110: {  // BFCVT Hd, Sn (single -> BF16)
          // The encoding uses ftype=01 even though the *source* is single-
          // precision (Sn), so explicitly reload the source as a 32-bit
          // float rather than reusing `src` (a 64-bit double from above).
          float src_single;
          memcpy(&src_single, &state_->cpu.v[args.rn], 4);
          uint16_t bf = FloatToBf16(src_single);
          state_->cpu.v[args.rd] = 0;
          memcpy(&state_->cpu.v[args.rd], &bf, 2);
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
        // FRINTTS scalar (FP64).
        case 0b010000:  // FRINT32Z
        case 0b010001:  // FRINT32X
        case 0b010010:  // FRINT64Z
        case 0b010011:  // FRINT64X
          result = FrintTs(src, opcode);
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
      // Half-precision source.
      uint16_t bits;
      memcpy(&bits, &state_->cpu.v[args.rn], 2);

      if (opcode == 0b000100) {
        // FCVT Sd, Hn (half -> single)
        float f = FpHalfToSingle(bits);
        state_->cpu.v[args.rd] = 0;
        memcpy(&state_->cpu.v[args.rd], &f, 4);
        return;
      }
      if (opcode == 0b000101) {
        // FCVT Dd, Hn (half -> double)
        float f = FpHalfToSingle(bits);
        double d = static_cast<double>(f);
        state_->cpu.v[args.rd] = 0;
        memcpy(&state_->cpu.v[args.rd], &d, 8);
        return;
      }

      // half-precision 1-source arithmetic.
      uint16_t result_bits;
      switch (opcode) {
        case 0b000000:  // FMOV Hd, Hn
          result_bits = bits;
          break;
        case 0b000001:  // FABS Hd, Hn — clear sign bit.
          result_bits = bits & 0x7FFF;
          break;
        case 0b000010:  // FNEG Hd, Hn — flip sign bit.
          result_bits = bits ^ 0x8000;
          break;
        case 0b000011: {  // FSQRT Hd, Hn
          float f = FpHalfToSingle(bits);
          result_bits = FpSingleToHalf(std::sqrt(f));
          break;
        }
        case 0b001000: {  // FRINTN
          float f = FpHalfToSingle(bits);
          result_bits = FpSingleToHalf(std::nearbyint(f));
          break;
        }
        case 0b001001: {  // FRINTP
          float f = FpHalfToSingle(bits);
          result_bits = FpSingleToHalf(std::ceil(f));
          break;
        }
        case 0b001010: {  // FRINTM
          float f = FpHalfToSingle(bits);
          result_bits = FpSingleToHalf(std::floor(f));
          break;
        }
        case 0b001011: {  // FRINTZ
          float f = FpHalfToSingle(bits);
          result_bits = FpSingleToHalf(std::trunc(f));
          break;
        }
        case 0b001100: {  // FRINTA
          float f = FpHalfToSingle(bits);
          result_bits = FpSingleToHalf(std::round(f));
          break;
        }
        case 0b001110: {  // FRINTX
          float f = FpHalfToSingle(bits);
          result_bits = FpSingleToHalf(std::rint(f));
          break;
        }
        case 0b001111: {  // FRINTI
          float f = FpHalfToSingle(bits);
          result_bits = FpSingleToHalf(std::rint(f));
          break;
        }
        default:
          Undefined();
          return;
      }
      state_->cpu.v[args.rd] = 0;
      memcpy(&state_->cpu.v[args.rd], &result_bits, 2);
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
        // ARM FMAX differs from libm fmax in NaN handling
        // (any-NaN -> default NaN, not other-operand) and in +0/-0
        // disambiguation. See FmaxScalar/FminScalar/etc.
        case 0b0100: result = FmaxScalar<float>(src_n, src_m); break;    // FMAX
        case 0b0101: result = FminScalar<float>(src_n, src_m); break;    // FMIN
        case 0b0110: result = FmaxnmScalar<float>(src_n, src_m); break;  // FMAXNM
        case 0b0111: result = FminnmScalar<float>(src_n, src_m); break;  // FMINNM
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
        // see comment above (single-precision arm).
        case 0b0100: result = FmaxScalar<double>(src_n, src_m); break;    // FMAX
        case 0b0101: result = FminScalar<double>(src_n, src_m); break;    // FMIN
        case 0b0110: result = FmaxnmScalar<double>(src_n, src_m); break;  // FMAXNM
        case 0b0111: result = FminnmScalar<double>(src_n, src_m); break;  // FMINNM
        case 0b1000: result = -(src_n * src_m); break;    // FNMUL
        default: Undefined(); return;
      }

      state_->cpu.v[args.rd] = 0;
      memcpy(&state_->cpu.v[args.rd], &result, 8);
    } else if (ftype == 0b11) {
      // Half-precision (Armv8.2-FP16).
      // Round-trip through float: load uint16 -> FpHalfToSingle -> op in
      // float -> FpSingleToHalf back.  binary32 has 24 mantissa bits vs
      // binary16's 11, so single-rounding back to half is correct.
      uint16_t bits_n, bits_m;
      memcpy(&bits_n, &state_->cpu.v[args.rn], 2);
      memcpy(&bits_m, &state_->cpu.v[args.rm], 2);
      float src_n = FpHalfToSingle(bits_n);
      float src_m = FpHalfToSingle(bits_m);
      float result;

      switch (opcode) {
        case 0b0000: result = src_n * src_m; break;       // FMUL
        case 0b0001: result = src_n / src_m; break;       // FDIV
        case 0b0010: result = src_n + src_m; break;       // FADD
        case 0b0011: result = src_n - src_m; break;       // FSUB
        // FP16 round-trip through float; helpers handle NaN + +0/-0 corner.
        case 0b0100: result = FmaxScalar<float>(src_n, src_m); break;    // FMAX
        case 0b0101: result = FminScalar<float>(src_n, src_m); break;    // FMIN
        case 0b0110: result = FmaxnmScalar<float>(src_n, src_m); break;  // FMAXNM
        case 0b0111: result = FminnmScalar<float>(src_n, src_m); break;  // FMINNM
        case 0b1000: result = -(src_n * src_m); break;    // FNMUL
        default: Undefined(); return;
      }

      uint16_t result_bits = FpSingleToHalf(result);
      state_->cpu.v[args.rd] = 0;
      memcpy(&state_->cpu.v[args.rd], &result_bits, 2);
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
    } else if (args.ftype == 0b11) {
      // Half-precision compare.
      uint16_t bits_n;
      memcpy(&bits_n, &state_->cpu.v[args.rn], 2);
      float src_n = FpHalfToSingle(bits_n);
      float src_m;
      if (args.with_zero) {
        src_m = 0.0f;
      } else {
        uint16_t bits_m;
        memcpy(&bits_m, &state_->cpu.v[args.rm], 2);
        src_m = FpHalfToSingle(bits_m);
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

  //
  // FP conditional compare: FCCMP / FCCMPE.
  // If cond is true, perform an FP compare and set NZCV;
  // otherwise copy the 4-bit immediate directly into NZCV.
  //
  void FpConditionalCompare(const Decoder::FpConditionalCompareArgs& args) {
    CHECK(!exception_raised_);

    if (!EvaluateCondition(args.cond)) {
      uint16_t flags = 0;
      if (args.nzcv & 0b1000) flags |= CPUState::kFlagNegative;
      if (args.nzcv & 0b0100) flags |= CPUState::kFlagZero;
      if (args.nzcv & 0b0010) flags |= CPUState::kFlagCarry;
      if (args.nzcv & 0b0001) flags |= CPUState::kFlagOverflow;
      state_->cpu.flags = flags;
      return;
    }

    uint16_t flags = 0;
    if (args.ftype == 0b00) {
      float src_n;
      float src_m;
      memcpy(&src_n, &state_->cpu.v[args.rn], 4);
      memcpy(&src_m, &state_->cpu.v[args.rm], 4);
      if (std::isnan(src_n) || std::isnan(src_m)) {
        flags = CPUState::kFlagCarry | CPUState::kFlagOverflow;
      } else if (src_n == src_m) {
        flags = CPUState::kFlagZero | CPUState::kFlagCarry;
      } else if (src_n < src_m) {
        flags = CPUState::kFlagNegative;
      } else {
        flags = CPUState::kFlagCarry;
      }
    } else if (args.ftype == 0b01) {
      double src_n;
      double src_m;
      memcpy(&src_n, &state_->cpu.v[args.rn], 8);
      memcpy(&src_m, &state_->cpu.v[args.rm], 8);
      if (std::isnan(src_n) || std::isnan(src_m)) {
        flags = CPUState::kFlagCarry | CPUState::kFlagOverflow;
      } else if (src_n == src_m) {
        flags = CPUState::kFlagZero | CPUState::kFlagCarry;
      } else if (src_n < src_m) {
        flags = CPUState::kFlagNegative;
      } else {
        flags = CPUState::kFlagCarry;
      }
    } else if (args.ftype == 0b11) {
      // Half-precision (Armv8.2-FP16): widen both operands to FP32 and compare.
      uint16_t bits_n;
      uint16_t bits_m;
      memcpy(&bits_n, &state_->cpu.v[args.rn], 2);
      memcpy(&bits_m, &state_->cpu.v[args.rm], 2);
      float src_n = FpHalfToSingle(bits_n);
      float src_m = FpHalfToSingle(bits_m);
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
        // Real FCVTZS/FCVTZU encoding has bit23=1 (size ∈ {10, 11}); the
        // legacy bit23=0 path is still routed here (see decoder).  Either
        // way only the sz bit (size & 1) chooses single vs double.
        if ((args.size & 1) == 0) {
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
        // See FCVTZU note above: only the sz bit (size & 1) is load-bearing.
        if ((args.size & 1) == 0) {
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
      // scalar FCVTAS / FCVTAU.
      // Single-lane collapse of vector kFcvtasV/kFcvtauV. std::round() is
      // round-to-nearest with ties-away-from-zero, which matches the ARM ARM
      // FCVTAS/FCVTAU semantics. NaN → 0 (signed) / 0 (unsigned). Out-of-range
      // saturates to INT_MAX/MIN or UINT_MAX. sz=args.size&1 chooses single
      // (0) vs double (1); the decoder pins bit23=0 so args.size ∈ {0, 1}.
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtas:
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtau: {
        bool is_unsigned =
            (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtau);
        if ((args.size & 1) == 0) {
          float fval;
          memcpy(&fval, &src, sizeof(fval));
          float r = std::isnan(fval) ? fval : std::round(fval);
          if (is_unsigned) {
            uint32_t ival;
            if (std::isnan(fval) || r < 0.0f) {
              ival = 0;
            } else if (r >= static_cast<float>(UINT32_MAX)) {
              ival = UINT32_MAX;
            } else {
              ival = static_cast<uint32_t>(r);
            }
            memcpy(&result, &ival, sizeof(ival));
          } else {
            int32_t ival;
            if (std::isnan(fval)) {
              ival = 0;
            } else if (r >= static_cast<float>(INT32_MAX)) {
              ival = INT32_MAX;
            } else if (r <= static_cast<float>(INT32_MIN)) {
              ival = INT32_MIN;
            } else {
              ival = static_cast<int32_t>(r);
            }
            uint32_t uval;
            memcpy(&uval, &ival, sizeof(uval));
            memcpy(&result, &uval, sizeof(uval));
          }
        } else {
          double fval;
          memcpy(&fval, &src, sizeof(fval));
          double r = std::isnan(fval) ? fval : std::round(fval);
          if (is_unsigned) {
            uint64_t ival;
            if (std::isnan(fval) || r < 0.0) {
              ival = 0;
            } else if (r >= static_cast<double>(UINT64_MAX)) {
              ival = UINT64_MAX;
            } else {
              ival = static_cast<uint64_t>(r);
            }
            memcpy(&result, &ival, sizeof(ival));
          } else {
            int64_t ival;
            if (std::isnan(fval)) {
              ival = 0;
            } else if (r >= static_cast<double>(INT64_MAX)) {
              ival = INT64_MAX;
            } else if (r <= static_cast<double>(INT64_MIN)) {
              ival = INT64_MIN;
            } else {
              ival = static_cast<int64_t>(r);
            }
            uint64_t uval;
            memcpy(&uval, &ival, sizeof(uval));
            memcpy(&result, &uval, sizeof(uval));
          }
        }
        break;
      }
      // scalar twins of vector SQABS / SQNEG / SQXTUN / FCVTXN.
      // Each is the single-lane collapse of the corresponding vector op
      // implemented in AdvSimdTwoRegMisc above; the result is the lane
      // value zero-extended to the full 128-bit Vd register.
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqabs:
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqneg: {
        // Lane width derived from raw 2-bit size: 0=B, 1=H, 2=S, 3=D.
        uint8_t esize = static_cast<uint8_t>(1U << args.size);
        uint8_t bits = static_cast<uint8_t>(esize * 8);
        int64_t int_min = (esize == 8) ? INT64_MIN : -(int64_t{1} << (bits - 1));
        int64_t int_max = (esize == 8) ? INT64_MAX : ((int64_t{1} << (bits - 1)) - 1);
        uint64_t emask =
            (esize == 8) ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
        uint64_t elem = 0;
        memcpy(&elem, &src, esize);
        int64_t signed_val =
            SignExtendElem(elem, bits);
        bool is_neg =
            (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqneg);
        int64_t out;
        if (signed_val == int_min) {
          out = int_max;  // saturate (single saturating input per ARM ARM)
        } else if (is_neg) {
          out = -signed_val;
        } else {
          out = (signed_val < 0) ? -signed_val : signed_val;
        }
        uint64_t out_u = static_cast<uint64_t>(out) & emask;
        memcpy(&result, &out_u, esize);
        break;
      }
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtun: {
        // size=00→8-bit dst from 16-bit src, 01→16/32, 10→32/64; size=11
        // is rejected in the decoder.
        uint8_t dst_esize = static_cast<uint8_t>(1U << args.size);
        uint8_t src_esize = static_cast<uint8_t>(dst_esize * 2);
        uint64_t raw = 0;
        memcpy(&raw, &src, src_esize);
        int64_t s = static_cast<int64_t>(raw << (64 - src_esize * 8)) >>
                    (64 - src_esize * 8);
        uint64_t dst_umax = (dst_esize == 8)
                                ? ~uint64_t{0}
                                : ((uint64_t{1} << (dst_esize * 8)) - 1);
        uint64_t out;
        if (s < 0) {
          out = 0;
        } else if (static_cast<uint64_t>(s) > dst_umax) {
          out = dst_umax;
        } else {
          out = static_cast<uint64_t>(s);
        }
        memcpy(&result, &out, dst_esize);
        break;
      }
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtn:
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kUqxtn: {
        // Single-lane saturating narrow.  size=00→8-bit dst from 16-bit src,
        // 01→16/32, 10→32/64; size=11 is rejected in the decoder.
        // SQXTN: signed source, clamp to [INT_min(dst), INT_max(dst)].
        // UQXTN: unsigned source, clamp to [0, UMAX(dst)].
        bool is_signed = (args.opcode ==
                          Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtn);
        uint8_t dst_esize = static_cast<uint8_t>(1U << args.size);
        uint8_t src_esize = static_cast<uint8_t>(dst_esize * 2);
        uint64_t raw = 0;
        memcpy(&raw, &src, src_esize);
        uint64_t dst_emask = (dst_esize == 8)
                                 ? ~uint64_t{0}
                                 : ((uint64_t{1} << (dst_esize * 8)) - 1);
        uint64_t out;
        if (is_signed) {
          int64_t s = static_cast<int64_t>(raw << (64 - src_esize * 8)) >>
                      (64 - src_esize * 8);
          int64_t smax = static_cast<int64_t>(dst_emask >> 1);
          int64_t smin = -smax - 1;
          if (s > smax) s = smax;
          if (s < smin) s = smin;
          out = static_cast<uint64_t>(s) & dst_emask;
        } else {
          out = (raw > dst_emask) ? dst_emask : raw;
        }
        memcpy(&result, &out, dst_esize);
        break;
      }
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtxn: {
        // FCVTXN Sd, Dn — single-lane FP64→FP32 round-to-odd.
        // Decoder pins args.size == 0b01 (FP64 source). Reuses the
        // FpDoubleToFloatRtO helper shipped for vector FCVTXN.
        double d;
        memcpy(&d, &src, sizeof(d));
        uint32_t f_bits = FpDoubleToFloatRtO(d);
        memcpy(&result, &f_bits, sizeof(f_bits));
        break;
      }
      // scalar FRECPE / FRSQRTE.
      // ARM spec only requires ~8 bits of mantissa precision; computing the
      // exact 1/x or 1/sqrt(x) is well within bound. Mirrors the vector
      // kFrecpeV/kFrsqrteV implementation. args.size bit 1 is pinned to 1
      // by the decoder; sz=args.size&1 selects single (0) vs double (1).
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kFrecpe:
      case Decoder::AdvSimdScalarTwoRegMiscOpcode::kFrsqrte: {
        bool is_rsqrt =
            (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFrsqrte);
        if ((args.size & 1) == 0) {
          float f;
          memcpy(&f, &src, sizeof(f));
          float r;
          if (is_rsqrt) {
            r = (f <= 0.0f || f != f) ? __builtin_nanf("")
                                      : 1.0f / __builtin_sqrtf(f);
          } else {
            r = (f == 0.0f) ? __builtin_inff() * (1.0f / f) : 1.0f / f;
          }
          memcpy(&result, &r, sizeof(r));
        } else {
          double d;
          memcpy(&d, &src, sizeof(d));
          double r;
          if (is_rsqrt) {
            r = (d <= 0.0 || d != d) ? __builtin_nan("")
                                     : 1.0 / __builtin_sqrt(d);
          } else {
            r = (d == 0.0) ? __builtin_inf() * (1.0 / d) : 1.0 / d;
          }
          memcpy(&result, &r, sizeof(r));
        }
        break;
      }
    }

    state_->cpu.v[args.rd] = result;
  }

  //
  // AdvSIMD scalar three same: scalar (D-form, 64-bit) integer 3-operand ops.
  // Operates on the bottom 64-bit element of each register; upper bits zero.
  //
  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs& args) {
    CHECK(!exception_raised_);

    // FP scalar ops: dispatch separately because size encodes S (0) vs D (1).
    switch (args.opcode) {
      case Decoder::AdvSimdScalarThreeSameOpcode::kFabd:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFmulx:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFcmgt:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFcmge:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFcmeq:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFacgt:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFacge:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFrecps:
      case Decoder::AdvSimdScalarThreeSameOpcode::kFrsqrts:
        {
        __uint128_t src_n = state_->cpu.v[args.rn];
        __uint128_t src_m = state_->cpu.v[args.rm];
        __uint128_t result = 0;
        // FP16 scalar three-same — widen/narrow round-trip
        // through FP32.  Bit-exact for the FMULX ±2.0 saturation case
        // because ±2.0 is exactly representable in FP16; bit-exact for FABD
        // because FP32 subtraction of two FP16 inputs is exact and the
        // single narrow back to FP16 applies one rounding.  Compares are
        // exact because widening to FP32 is value-preserving.
        if (args.is_fp16) {
          uint16_t hn = static_cast<uint16_t>(src_n);
          uint16_t hm = static_cast<uint16_t>(src_m);
          float a = FpHalfToSingle(hn);
          float b = FpHalfToSingle(hm);
          uint16_t r16 = 0;
          switch (args.opcode) {
            case Decoder::AdvSimdScalarThreeSameOpcode::kFmulx: {
              float f = FmulxScalar<float>(a, b);
              r16 = FpSingleToHalf(f);
              break;
            }
            case Decoder::AdvSimdScalarThreeSameOpcode::kFabd:
              r16 = FpSingleToHalf(std::fabs(a - b));
              break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmeq:
              r16 = (a == b) ? uint16_t{0xFFFF} : uint16_t{0};
              break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmge:
              r16 = (a >= b) ? uint16_t{0xFFFF} : uint16_t{0};
              break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFcmgt:
              r16 = (a > b) ? uint16_t{0xFFFF} : uint16_t{0};
              break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFacge:
              r16 = (std::fabs(a) >= std::fabs(b)) ? uint16_t{0xFFFF}
                                                   : uint16_t{0};
              break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFacgt:
              r16 = (std::fabs(a) > std::fabs(b)) ? uint16_t{0xFFFF}
                                                  : uint16_t{0};
              break;
            // FP16 scalar FRECPS / FRSQRTS — same widen/narrow
            // round-trip pattern.  Compute the refinement step in FP32 (exact
            // for any single FP16 op since FP32 strictly contains FP16's
            // mantissa precision) then narrow back to FP16 via one rounding.
            case Decoder::AdvSimdScalarThreeSameOpcode::kFrecps:
              r16 = FpSingleToHalf(FrecpsScalar<float>(a, b));
              break;
            case Decoder::AdvSimdScalarThreeSameOpcode::kFrsqrts:
              r16 = FpSingleToHalf(FrsqrtsScalar<float>(a, b));
              break;
            default:
              Undefined();
              return;
          }
          state_->cpu.v[args.rd] = static_cast<__uint128_t>(r16);
          return;
        }
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
            case Decoder::AdvSimdScalarThreeSameOpcode::kFmulx: {
              double d = FmulxScalar<double>(a, b);
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
            case Decoder::AdvSimdScalarThreeSameOpcode::kFrecps: {
              double d = FrecpsScalar<double>(a, b);
              memcpy(&r64, &d, sizeof(r64));
              break;
            }
            case Decoder::AdvSimdScalarThreeSameOpcode::kFrsqrts: {
              double d = FrsqrtsScalar<double>(a, b);
              memcpy(&r64, &d, sizeof(r64));
              break;
            }
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
            case Decoder::AdvSimdScalarThreeSameOpcode::kFmulx: {
              float f = FmulxScalar<float>(a, b);
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
            case Decoder::AdvSimdScalarThreeSameOpcode::kFrecps: {
              float f = FrecpsScalar<float>(a, b);
              memcpy(&r32, &f, sizeof(r32));
              break;
            }
            case Decoder::AdvSimdScalarThreeSameOpcode::kFrsqrts: {
              float f = FrsqrtsScalar<float>(a, b);
              memcpy(&r32, &f, sizeof(r32));
              break;
            }
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

    // scalar saturating add/sub (B/H/S/D).
    //
    // SQADD / UQADD / SQSUB / UQSUB scalar operate on a single lane of
    // width 8 / 16 / 32 / 64 bits selected by args.size, not just D-form.
    // ARM ARM C7.2.282 / .284 / .317 / .319.  The output is the saturated
    // result placed in the low `esize` bytes of Vd; upper bits are zero
    // (matches the routine's `result = 0` initialization pattern used by
    // the surrounding FP path).
    switch (args.opcode) {
      case Decoder::AdvSimdScalarThreeSameOpcode::kSqaddScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kUqaddScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kSqsubScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kUqsubScalar: {
        const uint8_t esize = uint8_t{1} << args.size;  // 1, 2, 4, 8
        const uint8_t bits_local = esize * 8;
        const uint64_t mask =
            (bits_local == 64) ? ~uint64_t{0}
                               : ((uint64_t{1} << bits_local) - 1);
        uint64_t a = static_cast<uint64_t>(state_->cpu.v[args.rn]) & mask;
        uint64_t b = static_cast<uint64_t>(state_->cpu.v[args.rm]) & mask;
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqaddScalar) ||
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqsubScalar);
        const bool is_sub =
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqsubScalar) ||
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kUqsubScalar);
        uint64_t r;
        if (is_signed) {
          // Sign-extend a and b from `bits_local` bits to 64 bits.
          const int shift = 64 - bits_local;
          int64_t sa = static_cast<int64_t>(a << shift) >> shift;
          int64_t sb = static_cast<int64_t>(b << shift) >> shift;
          const int64_t smax = (bits_local == 64)
                                   ? INT64_MAX
                                   : ((int64_t{1} << (bits_local - 1)) - 1);
          const int64_t smin = (bits_local == 64)
                                   ? INT64_MIN
                                   : -(int64_t{1} << (bits_local - 1));
          int64_t res;
          if (!is_sub) {
            // SQADD.  Mirrors the vector path's lambda (lines 4330-4345)
            // — the int64 overflow checks at bits_local=64 still hold
            // because two signed adds can only overflow toward INT64_MIN
            // or INT64_MAX, both of which the smax/smin clamps catch.
            int64_t sum = static_cast<int64_t>(
                static_cast<uint64_t>(sa) + static_cast<uint64_t>(sb));
            if (sb > 0 && sum < sa) res = smax;
            else if (sb < 0 && sum > sa) res = smin;
            else if (sum > smax) res = smax;
            else if (sum < smin) res = smin;
            else res = sum;
          } else {
            // SQSUB.  Same logic as the vector path's lambda (lines
            // 4355-4370): sa - sb saturates to the per-width signed range.
            int64_t diff = static_cast<int64_t>(
                static_cast<uint64_t>(sa) - static_cast<uint64_t>(sb));
            if (sb > 0 && diff > sa) res = smin;
            else if (sb < 0 && diff < sa) res = smax;
            else if (diff > smax) res = smax;
            else if (diff < smin) res = smin;
            else res = diff;
          }
          r = static_cast<uint64_t>(res) & mask;
        } else {
          // Unsigned: UQADD (sum > mask -> mask) / UQSUB (a < b -> 0).
          if (!is_sub) {
            uint64_t sum = a + b;
            r = (sum > mask || sum < a) ? mask : sum;
          } else {
            r = (a > b) ? (a - b) : 0;
          }
        }
        // Write low `bits_local` bits, zero the upper 128 - bits_local bits.
        state_->cpu.v[args.rd] = static_cast<__uint128_t>(r);
        return;
      }
      default:
        break;
    }

    // scalar saturating shift left (B/H/S/D).
    // Covers four new opcodes: SQSHL, UQSHL, SQRSHL, UQRSHL.  The shift
    // amount comes from the low 8 bits of Vm interpreted as int8_t —
    // positive shifts left (saturating on the per-width range), negative
    // shifts right (arithmetic for the signed forms, logical for unsigned).
    // The "R" variants round half-up by adding `1 << (rshift-1)` before
    // the right shift.  See ARM ARM C7.2.302 / .306 / .310 / .313.
    switch (args.opcode) {
      case Decoder::AdvSimdScalarThreeSameOpcode::kSqshlScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kUqshlScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kSqrshlScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kUqrshlScalar: {
        const uint8_t esize = uint8_t{1} << args.size;  // 1, 2, 4, 8
        const uint8_t bits_local = esize * 8;
        const uint64_t mask =
            (bits_local == 64) ? ~uint64_t{0}
                               : ((uint64_t{1} << bits_local) - 1);
        const uint64_t a = static_cast<uint64_t>(state_->cpu.v[args.rn]) & mask;
        // Shift amount is the low 8 bits of Vm.D[0], signed.
        const uint64_t b = static_cast<uint64_t>(state_->cpu.v[args.rm]);
        const int sh = static_cast<int8_t>(b & 0xFF);
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqshlScalar) ||
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqrshlScalar);
        const bool is_rounding =
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqrshlScalar) ||
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kUqrshlScalar);
        uint64_t r;
        if (is_signed) {
          const int shift_to_64 = 64 - bits_local;
          int64_t sa = static_cast<int64_t>(a << shift_to_64) >> shift_to_64;
          const int64_t smax = (bits_local == 64)
                                   ? INT64_MAX
                                   : ((int64_t{1} << (bits_local - 1)) - 1);
          const int64_t smin = (bits_local == 64)
                                   ? INT64_MIN
                                   : -(int64_t{1} << (bits_local - 1));
          int64_t res;
          if (sh > 0) {
            // Left shift, saturate if any bit beyond the sign bit gets set.
            if (sh >= bits_local) {
              res = (sa > 0) ? smax : ((sa < 0) ? smin : int64_t{0});
            } else {
              __int128 shifted = static_cast<__int128>(sa) << sh;
              if (shifted > smax) res = smax;
              else if (shifted < smin) res = smin;
              else res = static_cast<int64_t>(shifted);
            }
          } else if (sh == 0) {
            res = sa;
          } else {
            // Negative shift => arithmetic right shift (optionally rounding).
            const int rshift = -sh;
            if (is_rounding) {
              if (rshift > 64) {
                res = 0;
              } else {
                __int128 sum = static_cast<__int128>(sa) +
                               (static_cast<__int128>(1) << (rshift - 1));
                res = static_cast<int64_t>(sum >> rshift);
              }
            } else {
              if (rshift >= bits_local) {
                res = (sa < 0) ? int64_t{-1} : int64_t{0};
              } else {
                res = sa >> rshift;
              }
            }
          }
          // Right shifts can't grow beyond [smin, smax], but rounding can
          // bump a positive value across smax (e.g. esize=1, sa=127, rshift=
          // very-large rounding edge cases); clamp defensively.
          if (res > smax) res = smax;
          if (res < smin) res = smin;
          r = static_cast<uint64_t>(res) & mask;
        } else {
          const uint64_t ua = a;
          uint64_t res;
          if (sh > 0) {
            if (sh >= bits_local) {
              res = (ua != 0) ? mask : 0;
            } else {
              __uint128_t shifted = static_cast<__uint128_t>(ua) << sh;
              if (shifted > static_cast<__uint128_t>(mask)) res = mask;
              else res = static_cast<uint64_t>(shifted);
            }
          } else if (sh == 0) {
            res = ua;
          } else {
            const int rshift = -sh;
            if (is_rounding) {
              if (rshift > 64) {
                res = 0;
              } else {
                __uint128_t sum = static_cast<__uint128_t>(ua) +
                                  (static_cast<__uint128_t>(1) << (rshift - 1));
                res = static_cast<uint64_t>(sum >> rshift);
                if (res > mask) res = mask;
              }
            } else {
              if (rshift >= bits_local) {
                res = 0;
              } else {
                res = ua >> rshift;
              }
            }
          }
          r = res;
        }
        state_->cpu.v[args.rd] = static_cast<__uint128_t>(r);
        return;
      }
      // SRSHL / URSHL scalar (D only, non-saturating).
      // Like SSHL / USHL but with rounding when shifting right.  Per ARM
      // ARM C7.2.270 / .335, the rounding term `1 << (rshift-1)` is added
      // before the right shift; left shifts behave identically to SSHL /
      // USHL (no saturation — upper bits are discarded by the D-form mask
      // on writeback).
      case Decoder::AdvSimdScalarThreeSameOpcode::kSrshlScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kUrshlScalar: {
        const uint64_t ua = static_cast<uint64_t>(state_->cpu.v[args.rn]);
        const int sh = static_cast<int8_t>(
            static_cast<uint64_t>(state_->cpu.v[args.rm]) & 0xFF);
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSrshlScalar);
        uint64_t r;
        if (sh > 0) {
          // Left shift, no saturation — discard upper bits.
          r = (sh >= 64) ? uint64_t{0} : (ua << sh);
        } else if (sh == 0) {
          r = ua;
        } else {
          const int rshift = -sh;
          if (rshift > 64) {
            // Rounding right shift with rshift > 64 always rounds to 0
            // for any 64-bit operand (the rounding term 2^(rshift-1)
            // dominates and the quotient truncates to 0).
            r = 0;
          } else if (is_signed) {
            const int64_t sa = static_cast<int64_t>(ua);
            __int128 sum = static_cast<__int128>(sa) +
                           (static_cast<__int128>(1) << (rshift - 1));
            r = static_cast<uint64_t>(static_cast<int64_t>(sum >> rshift));
          } else {
            __uint128_t sum = static_cast<__uint128_t>(ua) +
                              (static_cast<__uint128_t>(1) << (rshift - 1));
            r = static_cast<uint64_t>(sum >> rshift);
          }
        }
        state_->cpu.v[args.rd] = static_cast<__uint128_t>(r);
        return;
      }
      // SQDMULH / SQRDMULH scalar (H/S only).
      // Per ARM ARM C7.2.301 / .305:
      //   Vd = sat_signed((2 * sext(Vn) * sext(Vm) + round) >> bits_local)
      // with bits_local = 16 (size=01, H) or 32 (size=10, S), and
      // round = 1 << (bits_local - 1) for SQRDMULH or 0 for SQDMULH.
      // The doubled product for size=S reaches 2 * INT32_MIN * INT32_MIN =
      // 2^63, which overflows int64_t by one — so the multiply is done
      // in __int128 before the right shift and the saturation clamp.
      case Decoder::AdvSimdScalarThreeSameOpcode::kSqdmulhScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmulhScalar: {
        const uint8_t esize = uint8_t{1} << args.size;  // 2 or 4
        const uint8_t bits_local = esize * 8;           // 16 or 32
        const int shift_to_64 = 64 - bits_local;
        const int64_t sa =
            static_cast<int64_t>(static_cast<uint64_t>(state_->cpu.v[args.rn])
                                 << shift_to_64) >> shift_to_64;
        const int64_t sb =
            static_cast<int64_t>(static_cast<uint64_t>(state_->cpu.v[args.rm])
                                 << shift_to_64) >> shift_to_64;
        const __int128 round =
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmulhScalar)
                ? (static_cast<__int128>(1) << (bits_local - 1))
                : __int128{0};
        const __int128 product =
            (static_cast<__int128>(2) * sa * sb) + round;
        int64_t res = static_cast<int64_t>(product >> bits_local);
        const int64_t smax = (int64_t{1} << (bits_local - 1)) - 1;
        const int64_t smin = -(int64_t{1} << (bits_local - 1));
        res = SatClampSigned(res, smin, smax);
        const uint64_t mask = (uint64_t{1} << bits_local) - 1;
        state_->cpu.v[args.rd] =
            static_cast<__uint128_t>(static_cast<uint64_t>(res) & mask);
        return;
      }
      // SQRDMLAH / SQRDMLSH scalar (H/S only, Armv8.1-RDM).
      //   Vd[0:bits_local] = sat_signed(
      //       Vd[0:bits_local]
      //       ± sat_signed((2 * sext(Vn) * sext(Vm) + round) >> bits_local))
      //   where round = 1 << (bits_local - 1); bits_local = 16 (size=01, H)
      //   or 32 (size=10, S).  SQRDMLAH adds, SQRDMLSH subtracts.  Both
      //   stages saturate to the signed narrow range, then the result is
      //   zero-extended into Vd (the top of the 128-bit vector register).
      //   Sibling of the vector form kSqrdmlahVec / kSqrdmlshVec.
      case Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmlahScalar:
      case Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmlshScalar: {
        const uint8_t esize = uint8_t{1} << args.size;  // 2 or 4
        const uint8_t bits_local = esize * 8;           // 16 or 32
        const int shift_to_64 = 64 - bits_local;
        const int64_t sa =
            static_cast<int64_t>(static_cast<uint64_t>(state_->cpu.v[args.rn])
                                 << shift_to_64) >> shift_to_64;
        const int64_t sb =
            static_cast<int64_t>(static_cast<uint64_t>(state_->cpu.v[args.rm])
                                 << shift_to_64) >> shift_to_64;
        const int64_t sd =
            static_cast<int64_t>(static_cast<uint64_t>(state_->cpu.v[args.rd])
                                 << shift_to_64) >> shift_to_64;
        const __int128 round = static_cast<__int128>(1) << (bits_local - 1);
        const __int128 product =
            (static_cast<__int128>(2) * sa * sb) + round;
        int64_t addend = static_cast<int64_t>(product >> bits_local);
        const int64_t smax = (int64_t{1} << (bits_local - 1)) - 1;
        const int64_t smin = -(int64_t{1} << (bits_local - 1));
        addend = SatClampSigned(addend, smin, smax);
        const bool is_sub =
            (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmlshScalar);
        int64_t res = is_sub ? sd - addend : sd + addend;
        res = SatClampSigned(res, smin, smax);
        const uint64_t mask = (uint64_t{1} << bits_local) - 1;
        state_->cpu.v[args.rd] =
            static_cast<__uint128_t>(static_cast<uint64_t>(res) & mask);
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

  //
  // AdvSIMD scalar pairwise.
  // ADDP scalar (D-form): Vd[0] = Vn.D[0] + Vn.D[1].
  // FP variants (FADDP / FMAXNMP / FMINNMP / FMAXP / FMINP scalar):
  //   reduce Vn.S[0] op Vn.S[1] (size=00) or Vn.D[0] op Vn.D[1] (size=01)
  //   into the bottom lane of Vd; upper bits zero.
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
      // FP scalar pairwise.  size[0] picks S (0) vs D (1)
      // for the FP32/FP64 forms; args.is_fp16 selects the Armv8.2-FP16 form.
      // FMAX/FMIN: NaN-propagating (any-NaN -> NaN).
      // FMAXNM/FMINNM: NaN-quiet (single-NaN -> other operand).
      case Decoder::AdvSimdScalarPairwiseOpcode::kFaddpScalar:
      case Decoder::AdvSimdScalarPairwiseOpcode::kFmaxnmpScalar:
      case Decoder::AdvSimdScalarPairwiseOpcode::kFminnmpScalar:
      case Decoder::AdvSimdScalarPairwiseOpcode::kFmaxpScalar:
      case Decoder::AdvSimdScalarPairwiseOpcode::kFminpScalar: {
        if (args.is_fp16) {
          // H form: read Vn.H[0] and Vn.H[1], promote each to FP32 for the
          // reduction (FP16 -> FP32 is exact and preserves NaN), then narrow
          // the result back to FP16.
          uint16_t a_h = static_cast<uint16_t>(lo);
          uint16_t b_h = static_cast<uint16_t>(lo >> 16);
          float a = FpHalfToSingle(a_h);
          float b = FpHalfToSingle(b_h);
          float f;
          switch (args.opcode) {
            case Decoder::AdvSimdScalarPairwiseOpcode::kFaddpScalar:
              f = a + b;
              break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFmaxpScalar:
              f = FmaxScalar<float>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFminpScalar:
              f = FminScalar<float>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFmaxnmpScalar:
              f = FmaxnmScalar<float>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFminnmpScalar:
              f = FminnmScalar<float>(a, b); break;
            default:
              __builtin_unreachable();
          }
          uint16_t out_h = FpSingleToHalf(f);
          r = out_h;  // upper 48 bits of the bottom 64-bit lane are zero
          break;
        }
        bool is_double = ((args.size & 1) != 0);
        if (is_double) {
          double a, b, d;
          memcpy(&a, &lo, 8);
          memcpy(&b, &hi, 8);
          switch (args.opcode) {
            case Decoder::AdvSimdScalarPairwiseOpcode::kFaddpScalar:
              d = a + b;
              break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFmaxpScalar:
              d = FmaxScalar<double>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFminpScalar:
              d = FminScalar<double>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFmaxnmpScalar:
              d = FmaxnmScalar<double>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFminnmpScalar:
              d = FminnmScalar<double>(a, b); break;
            default:
              __builtin_unreachable();
          }
          memcpy(&r, &d, 8);
        } else {
          // S form: read Vn.S[0] (bits[31:0] of lo) and Vn.S[1] (bits[63:32]).
          uint32_t a_bits = static_cast<uint32_t>(lo);
          uint32_t b_bits = static_cast<uint32_t>(lo >> 32);
          float a, b, f;
          memcpy(&a, &a_bits, 4);
          memcpy(&b, &b_bits, 4);
          switch (args.opcode) {
            case Decoder::AdvSimdScalarPairwiseOpcode::kFaddpScalar:
              f = a + b;
              break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFmaxpScalar:
              f = FmaxScalar<float>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFminpScalar:
              f = FminScalar<float>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFmaxnmpScalar:
              f = FmaxnmScalar<float>(a, b); break;
            case Decoder::AdvSimdScalarPairwiseOpcode::kFminnmpScalar:
              f = FminnmScalar<float>(a, b); break;
            default:
              __builtin_unreachable();
          }
          uint32_t f_bits;
          memcpy(&f_bits, &f, 4);
          r = f_bits;  // upper 32 bits of the bottom 64-bit lane are zero
        }
        break;
      }
      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = static_cast<__uint128_t>(r);
  }

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
          ClearUpperIfNotQ(&result, args.q);
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

      case Decoder::AdvSimdTwoRegMiscOpcode::kCls: {
        // CLS: count leading sign bits per element (number of consecutive
        // bits following the most-significant bit that equal the MSB; result
        // range 0..esize*8-1).  Standard identity:
        //   y = (x < 0) ? ~x : x
        //   cls(x) = clz_N(y) - 1
        // where clz_N treats y as an N-bit value (N = esize*8).
        if (args.size == 0b11) { Undefined(); return; }
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = SignExtendElem(elem, bits);
          uint64_t y = static_cast<uint64_t>(signed_val < 0 ? ~signed_val : signed_val) & emask;
          uint64_t cls;
          if (y == 0) {
            cls = bits - 1;
          } else {
            cls = __builtin_clzll(y) - (64 - bits) - 1;
          }
          cls &= emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &cls, esize);
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
          int64_t signed_val = SignExtendElem(elem, bits);
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

      // SQABS / SQNEG (signed saturating abs / negate).
      // For each lane the source is sign-extended to int64_t. The only
      // input that saturates is INT_MIN_for_this_size: both |INT_MIN| and
      // -INT_MIN overflow the signed range, so they clamp to INT_MAX.
      // All other inputs map to ordinary |x| (SQABS) or -x (SQNEG).
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqabs:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqneg: {
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        // Compute INT_MIN/INT_MAX for this signed element width.
        // For esize==8 use the int64 limits directly; otherwise the
        // shift fits in a signed-defined range.
        int64_t int_min = (esize == 8) ? INT64_MIN : -(1LL << (bits - 1));
        int64_t int_max = (esize == 8) ? INT64_MAX : ((1LL << (bits - 1)) - 1);
        bool is_neg = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSqneg);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = SignExtendElem(elem, bits);
          int64_t out;
          if (signed_val == int_min) {
            out = int_max;  // saturate (single saturating input per ARM ARM)
          } else if (is_neg) {
            out = -signed_val;
          } else {
            out = (signed_val < 0) ? -signed_val : signed_val;
          }
          uint64_t out_u = static_cast<uint64_t>(out) & emask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &out_u, esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmeqZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmltZero: {
        // Armv8.2-FP16 vector FCMxx zero.
        // The FP16 encoding of FCMGT/FCMEQ/FCMLT/FCMGE/FCMLE #0 routes to
        // the same enum values; is_fp16 flips the per-lane semantics from
        // integer signed compare to FP compare with FpHalfToSingle.
        if (args.is_fp16) {
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            float f = FpHalfToSingle(h);
            bool cond;
            switch (args.opcode) {
              case Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero: cond = (f >  0.0f); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero: cond = (f >= 0.0f); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kCmeqZero: cond = (f == 0.0f); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero: cond = (f <= 0.0f); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kCmltZero: cond = (f <  0.0f); break;
              default: cond = false; break;
            }
            uint16_t rh = cond ? uint16_t{0xFFFF} : uint16_t{0};
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &rh, 2);
          }
          break;
        }
        // Integer signed compare against zero.
        uint64_t emask = ElementMask(esize);
        uint8_t bits = esize * 8;
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = SignExtendElem(elem, bits);
          bool cond;
          switch (args.opcode) {
            case Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero: cond = (signed_val >  0); break;
            case Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero: cond = (signed_val >= 0); break;
            case Decoder::AdvSimdTwoRegMiscOpcode::kCmeqZero: cond = (elem == 0); break;
            case Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero: cond = (signed_val <= 0); break;
            case Decoder::AdvSimdTwoRegMiscOpcode::kCmltZero: cond = (signed_val <  0); break;
            default: cond = false; break;
          }
          uint64_t r = cond ? emask : 0;
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
        }
        break;
      }

      // FP FCMxxZero (FP32/FP64 two-reg-misc).
      // Per-lane compare against +0.0; sets all bits of the destination lane
      // when the predicate holds, else zero. args.size = bits[23:22]; bit22=0
      // selects FP32, bit22=1 selects FP64 (FP64 requires Q=1 by decoder).
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgtZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgeZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmeqZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmleZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcmltZero: {
        if (!(args.size & 0b10)) { Undefined(); return; }  // FP needs bit23=1
        if ((args.size & 1) == 0) {
          uint8_t fp_count = args.q ? 4 : 2;
          for (uint8_t i = 0; i < fp_count; i++) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            bool cond;
            switch (args.opcode) {
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgtZero: cond = (f >  0.0f); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgeZero: cond = (f >= 0.0f); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmeqZero: cond = (f == 0.0f); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmleZero: cond = (f <= 0.0f); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmltZero: cond = (f <  0.0f); break;
              default: cond = false; break;
            }
            uint32_t r = cond ? uint32_t{0xFFFFFFFF} : uint32_t{0};
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
          }
        } else {
          for (uint8_t i = 0; i < 2; i++) {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            bool cond;
            switch (args.opcode) {
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgtZero: cond = (d >  0.0); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgeZero: cond = (d >= 0.0); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmeqZero: cond = (d == 0.0); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmleZero: cond = (d <= 0.0); break;
              case Decoder::AdvSimdTwoRegMiscOpcode::kFcmltZero: cond = (d <  0.0); break;
              default: cond = false; break;
            }
            uint64_t r = cond ? uint64_t{0xFFFFFFFFFFFFFFFFULL} : uint64_t{0};
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
          }
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

      // SHLL / SHLL2 (shift left long by element size).
      // Source element width: esize (8/16/32 bits). Destination element width:
      // 2*esize (16/32/64 bits). Implicit shift amount = source element bits.
      // Each source element ends up in the upper half of the widened destination
      // lane; lower half is zero. Q=0 reads source from low 64 bits of Vn
      // (SHLL); Q=1 reads source from upper 64 bits (SHLL2). Result fills all
      // 128 bits of Vd in both cases (no preserve-lower-half semantics).
      case Decoder::AdvSimdTwoRegMiscOpcode::kShll: {
        uint8_t src_esize = esize;        // 1, 2, or 4 bytes
        uint8_t dst_esize = esize * 2;    // 2, 4, or 8 bytes
        if (dst_esize > 8) { Undefined(); return; }
        uint8_t src_off = args.q ? 8 : 0;
        uint8_t dst_count = 8 / src_esize;  // 8 lanes for B->H, 4 for H->S, 2 for S->D
        result = static_cast<__uint128_t>(0);
        for (uint8_t i = 0; i < dst_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + src_off + i * src_esize, src_esize);
          uint64_t shifted = elem << (src_esize * 8);
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * dst_esize, &shifted, dst_esize);
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kFabs: {
        // FABS (vector): floating-point absolute value per element.
        // Armv8.2-FP16 vector FABS.
        if (args.is_fp16) {
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            uint16_t rh = FpSingleToHalf(std::fabs(FpHalfToSingle(h)));
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &rh, 2);
          }
          break;
        }
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
        // Armv8.2-FP16 vector FNEG.
        if (args.is_fp16) {
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            uint16_t rh = FpSingleToHalf(-FpHalfToSingle(h));
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &rh, 2);
          }
          break;
        }
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

      // across-lanes FP reductions FMAXV / FMINV /
      // FMAXNMV / FMINNMV. Reduce all lanes of an FP vector (.4S for
      // FP32, where the decoder pins Q=1; .4H or .8H for FP16 per Q) to
      // a single scalar FP element.
      //   FMAXV/FMINV: IEEE 754-2008 max/min — any NaN in input
      //     propagates to a NaN result.
      //   FMAXNMV/FMINNMV: max-number/min-number — a NaN is skipped
      //     when the other operand is non-NaN; if both are NaN, NaN
      //     propagates.
      // The args.size field for these ops carries "o sz" (bit23=o,
      // bit22=sz), so the routine's normal size->esize mapping does
      // not apply — the lane width is determined by args.is_fp16
      // (FP16=2 bytes, otherwise FP32=4 bytes). FP16 promotes each
      // lane to FP32 via FpHalfToSingle, reduces in FP32 (same NaN
      // and signed-zero semantics), then narrows the result back via
      // FpSingleToHalf — the same round-trip used elsewhere for FP16
      // max/min (e.g. the FP16 three-same FMAX/FMIN arm).
      // Reduction order is unspecified by ARM; FP max/min is
      // associative across the non-NaN/NaN axis, so a linear sweep
      // produces the architecturally-required result.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFminv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv: {
        bool is_max = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv ||
                       args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv);
        bool is_nm = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv ||
                      args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv);
        if (args.is_fp16) {
          // .4H (Q=0) / .8H (Q=1) form: reduce 4 or 8 half lanes. Promote
          // to FP32 for the reduction (FP16 -> FP32 is exact and preserves
          // NaN); narrow the final accumulator back to FP16. The lane
          // count comes from Q directly — args.size carries "o sz" for
          // these opcodes, not an element width.
          const uint8_t fp16_lanes = args.q ? 8 : 4;
          uint16_t acc_h;
          memcpy(&acc_h, reinterpret_cast<const uint8_t*>(&src), 2);
          float acc = FpHalfToSingle(acc_h);
          for (uint8_t i = 1; i < fp16_lanes; i++) {
            uint16_t a_h;
            memcpy(&a_h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            float a = FpHalfToSingle(a_h);
            float b = acc;
            if (is_nm) {
              acc = is_max ? FmaxnmScalar<float>(a, b) : FminnmScalar<float>(a, b);
            } else {
              acc = is_max ? FmaxScalar<float>(a, b) : FminScalar<float>(a, b);
            }
          }
          uint16_t out_h = FpSingleToHalf(acc);
          result = 0;
          memcpy(reinterpret_cast<uint8_t*>(&result), &out_h, 2);
          break;
        }
        // .4S form.
        float acc;
        memcpy(&acc, reinterpret_cast<const uint8_t*>(&src), 4);
        for (uint8_t i = 1; i < 4; i++) {
          float a;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
          float b = acc;
          // FmaxScalar/etc. handle NaN propagation and +0/-0 sign disambiguation.
          if (is_nm) {
            acc = is_max ? FmaxnmScalar<float>(a, b) : FminnmScalar<float>(a, b);
          } else {
            acc = is_max ? FmaxScalar<float>(a, b) : FminScalar<float>(a, b);
          }
        }
        uint32_t out_bits;
        memcpy(&out_bits, &acc, 4);
        result = 0;
        memcpy(reinterpret_cast<uint8_t*>(&result), &out_bits, 4);
        break;
      }

      // across-lanes max/min reductions
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
            int64_t s_elem = SignExtendElem(elem, bits);
            int64_t s_acc = SignExtendElem(acc, bits);
            elem_wins = is_max ? (s_elem > s_acc) : (s_elem < s_acc);
          } else {
            elem_wins = is_max ? (elem > acc) : (elem < acc);
          }
          if (elem_wins) acc = elem;
        }
        result = acc & emask;
        break;
      }

      // SCVTF/UCVTF (vector, integer): per-lane signed or
      // unsigned int-to-FP. Element size from `size` field: sz=0 -> single
      // (.4S / .2S), sz=1 -> double (.2D). Observed `ucvtf v0.4s, v0.4s`
      // (insn 0x6e21d800) in WhatsApp's libar-bundle3.so init path.
      case Decoder::AdvSimdTwoRegMiscOpcode::kScvtfV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV: {
        // FP16 form (SCVTF/UCVTF v.4h, v.4h).
        // Per-lane sint16->half / uint16->half.
        if (args.is_fp16) {
          bool is_unsigned_fp16 = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV);
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t int_bits;
            memcpy(&int_bits, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            float f = is_unsigned_fp16
                          ? static_cast<float>(int_bits)
                          : static_cast<float>(static_cast<int16_t>(int_bits));
            uint16_t rh = FpSingleToHalf(f);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &rh, 2);
          }
          break;
        }
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

      // FCVTZS/FCVTZU (vector, FP→int): per-lane FP-to-int
      // truncating conversion. sz=0 -> 32-bit float→int32, sz=1 -> 64-bit
      // double→int64. Out-of-range values saturate per the ARM ARM spec.
      // The decoder routes opcode=11011 with bit23=1 here, so args.size's
      // high bit is always 1 -- only the low bit (sz) selects single vs
      // double, unlike SCVTF/UCVTF whose bit23=0 path keeps size's high
      // bit clear. We don't reject "size & 0b10" the way SCVTF does.
      // FCVT* vector with explicit rounding mode.
      // Mirrors the scalar FpIntConversion rounding cases. Implements
      // FCVTN[S|U] (ties-to-even), FCVTM[S|U] (toward -inf),
      // FCVTP[S|U] (toward +inf), FCVTA[S|U] (ties-away-from-zero).
      // FCVTZ[S|U] (truncate) keeps its existing dedicated case below.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmuV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtasV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV: {
        bool is_unsigned =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV ||
             args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmuV ||
             args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV ||
             args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV);
        auto apply_round = [&](double x) -> double {
          switch (args.opcode) {
            case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnsV:
            case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV:
              return rint(x);   // ties-to-even (assumes default FE_TONEAREST)
            case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmsV:
            case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmuV:
              return floor(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpsV:
            case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV:
              return ceil(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtasV:
            case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV:
              return round(x);  // ties-away-from-zero
            default:
              return 0.0;
          }
        };
        // FP16 form. Promote each half lane to
        // float, apply the same round-to-int, then narrow to int16/uint16.
        if (args.is_fp16) {
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            float f = FpHalfToSingle(h);
            double r = (f != f) ? static_cast<double>(f) : apply_round(static_cast<double>(f));
            uint16_t out;
            if (is_unsigned) {
              uint16_t v;
              if (f != f || r < 0.0) v = 0u;
              else if (r >= 65536.0) v = 0xffffu;
              else v = static_cast<uint16_t>(r);
              out = v;
            } else {
              int16_t v;
              if (f != f) v = 0;
              else if (r >= 32768.0) v = 0x7fff;
              else if (r < -32768.0) v = static_cast<int16_t>(0x8000);
              else v = static_cast<int16_t>(r);
              memcpy(&out, &v, 2);
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &out, 2);
          }
          break;
        }
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        uint8_t fp_count = vec_len / fp_esize;
        for (uint8_t i = 0; i < fp_count; i++) {
          if (fp_esize == 4) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            double r = (f != f) ? f : apply_round(static_cast<double>(f));
            uint32_t out;
            if (is_unsigned) {
              uint32_t v;
              if (f != f || r < 0.0) v = 0u;
              else if (r >= 4294967296.0) v = 0xffffffffu;
              else v = static_cast<uint32_t>(r);
              out = v;
            } else {
              int32_t v;
              if (f != f) v = 0;
              else if (r >= 2147483648.0) v = 0x7fffffff;
              else if (r < -2147483648.0) v = static_cast<int32_t>(0x80000000);
              else v = static_cast<int32_t>(r);
              memcpy(&out, &v, 4);
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &out, 4);
          } else {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            double r = (d != d) ? d : apply_round(d);
            uint64_t out;
            if (is_unsigned) {
              uint64_t v;
              if (d != d || r < 0.0) v = 0u;
              else if (r >= 18446744073709551616.0) v = 0xffffffffffffffffULL;
              else v = static_cast<uint64_t>(r);
              out = v;
            } else {
              int64_t v;
              if (d != d) v = 0;
              else if (r >= 9223372036854775808.0) v = 0x7fffffffffffffffLL;
              else if (r < -9223372036854775808.0)
                v = static_cast<int64_t>(0x8000000000000000ULL);
              else v = static_cast<int64_t>(r);
              memcpy(&out, &v, 8);
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &out, 8);
          }
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzuV: {
        bool is_unsigned =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzuV);
        // FP16 form (FCVTZS/ZU v.4h, v.4h).
        if (args.is_fp16) {
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            float f = FpHalfToSingle(h);
            uint16_t out;
            if (is_unsigned) {
              uint16_t v;
              if (f != f || f < 0.0f) v = 0u;
              else if (f >= 65536.0f) v = 0xffffu;
              else v = static_cast<uint16_t>(f);  // truncates toward zero
              out = v;
            } else {
              int16_t v;
              if (f != f) v = 0;
              else if (f >= 32768.0f) v = 0x7fff;
              else if (f < -32768.0f) v = static_cast<int16_t>(0x8000);
              else v = static_cast<int16_t>(f);
              memcpy(&out, &v, 2);
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &out, 2);
          }
          break;
        }
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        uint8_t fp_count = vec_len / fp_esize;
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

      // FRECPE / FRSQRTE (vector): per-lane reciprocal /
      // reciprocal-square-root estimate. The ARM spec only requires ~8 bits
      // of mantissa precision; computing 1/x and 1/sqrt(x) in full precision
      // is well within that bound, so callers that need the estimate as a
      // Newton-Raphson seed will converge identically.
      // FSQRT (vector): per-lane square root.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFsqrtV: {
        // Armv8.2-FP16 vector FSQRT.
        if (args.is_fp16) {
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            uint16_t rh = FpSingleToHalf(__builtin_sqrtf(FpHalfToSingle(h)));
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &rh, 2);
          }
          break;
        }
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

      // a=0/a=1 columns: FP16 vector FRINT*.
      // Per-lane round-to-integral-FP-value with mode selected by opcode.
      // FRINTN ties-to-even, FRINTA ties-away, FRINTM toward -inf,
      // FRINTP toward +inf, FRINTZ toward zero, FRINTX/FRINTI use the
      // current FPCR rounding mode (we treat both as nearbyint, which on
      // x86_64 with default FE_TONEAREST is round-half-to-even — matching
      // the ARM default).
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintaV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintxV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrintiV: {
        auto apply_round_f32 = [&](float x) -> float {
          if (x != x) return x;  // NaN propagates
          switch (args.opcode) {
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV:
              return nearbyintf(x);    // ties-to-even (default FE_TONEAREST)
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintaV:
              return roundf(x);        // ties-away-from-zero
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV:
              return floorf(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV:
              return ceilf(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV:
              return truncf(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintxV:
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintiV:
              return rintf(x);         // current FPCR rounding mode
            default:
              return 0.0f;
          }
        };
        // FP16 form (FRINT* v.4h / v.8h).
        if (args.is_fp16) {
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            uint16_t rh = FpSingleToHalf(apply_round_f32(FpHalfToSingle(h)));
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &rh, 2);
          }
          break;
        }
        // std FP32/FP64 FRINT*. `args.size`
        // here is (bit23=a, bit22=sz); only bit22 selects FP32 (0) vs FP64 (1).
        auto apply_round_f64 = [&](double x) -> double {
          if (x != x) return x;
          switch (args.opcode) {
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV:
              return nearbyint(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintaV:
              return round(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV:
              return floor(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV:
              return ceil(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV:
              return trunc(x);
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintxV:
            case Decoder::AdvSimdTwoRegMiscOpcode::kFrintiV:
              return rint(x);
            default:
              return 0.0;
          }
        };
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        uint8_t fp_count = vec_len / fp_esize;
        for (uint8_t i = 0; i < fp_count; i++) {
          if (fp_esize == 4) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            f = apply_round_f32(f);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &f, 4);
          } else {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            d = apply_round_f64(d);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &d, 8);
          }
        }
        break;
      }

      // FRINTTS vector (FRINT32Z/X, FRINT64Z/X). Per lane,
      // round to a 32/64-bit integral FP value with saturation, reusing the
      // scalar FrintTs helper. Element width comes from sz (size&1); the int
      // target width + rounding mode come from the opcode (mapped to the same
      // 6-bit selector the scalar FpDataProc1 path uses).
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrint32zV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrint32xV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrint64zV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrint64xV: {
        uint8_t scalar_op;
        switch (args.opcode) {
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrint32zV: scalar_op = 0b010000; break;
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrint32xV: scalar_op = 0b010001; break;
          case Decoder::AdvSimdTwoRegMiscOpcode::kFrint64zV: scalar_op = 0b010010; break;
          default:                                           scalar_op = 0b010011; break;  // kFrint64xV
        }
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        uint8_t fp_count = vec_len / fp_esize;
        for (uint8_t i = 0; i < fp_count; i++) {
          if (fp_esize == 4) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            float r = static_cast<float>(FrintTs(static_cast<double>(f), scalar_op));
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
          } else {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            double r = FrintTs(d, scalar_op);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
          }
        }
        break;
      }

      case Decoder::AdvSimdTwoRegMiscOpcode::kFrecpeV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFrsqrteV: {
        bool is_rsqrt =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFrsqrteV);
        // FP16 form (FRECPE/FRSQRTE v.4h, v.4h).
        if (args.is_fp16) {
          uint8_t fp_count = args.q ? 8 : 4;
          for (uint8_t i = 0; i < fp_count; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + i * 2, 2);
            float f = FpHalfToSingle(h);
            float r;
            if (is_rsqrt) {
              r = (f <= 0.0f || f != f) ? __builtin_nanf("") : 1.0f / __builtin_sqrtf(f);
            } else {
              r = (f == 0.0f) ? __builtin_inff() * (1.0f / f) : 1.0f / f;
            }
            uint16_t rh = FpSingleToHalf(r);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &rh, 2);
          }
          break;
        }
        uint8_t fp_esize = (args.size & 1) ? 8 : 4;
        // Same bit23=1 rationale as the FCVTZS case above.
        uint8_t fp_count = vec_len / fp_esize;
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

      // URECPE / URSQRTE: unsigned integer reciprocal and
      // reciprocal-square-root estimate (.2S/.4S). These are fixed-point
      // estimates defined by the ARM ARM UnsignedRecipEstimate /
      // UnsignedRSqrtEstimate pseudocode; the integer recurrences below are
      // the bit-exact equivalents (matching the architectural estimate tables).
      // An out-of-range input (top bit clear for URECPE, top two bits clear for
      // URSQRTE) yields the saturated 0xFFFFFFFF estimate.
      case Decoder::AdvSimdTwoRegMiscOpcode::kUrecpe:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUrsqrte: {
        if (esize != 4) { Undefined(); return; }  // 32-bit lanes only
        const bool is_rsqrt =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUrsqrte);
        for (uint8_t i = 0; i < num_elements; i++) {
          uint32_t a = 0;
          memcpy(&a, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
          uint32_t r;
          if (!is_rsqrt) {
            if ((a & 0x80000000u) == 0) {
              r = 0xFFFFFFFFu;
            } else {
              int input = static_cast<int>((a >> 23) & 0x1FF);  // [256,511]
              int a2 = input * 2 + 1;
              int b = (1 << 19) / a2;
              int estimate = (b + 1) / 2;  // [256,511]
              r = static_cast<uint32_t>(estimate) << 23;
            }
          } else {
            if ((a & 0xC0000000u) == 0) {
              r = 0xFFFFFFFFu;
            } else {
              int input = static_cast<int>((a >> 23) & 0x1FF);  // [128,511]
              int aa;
              if (input < 256) {
                aa = input * 2 + 1;
              } else {
                aa = (input >> 1) << 1;
                aa = (aa + 1) * 2;
              }
              int b = 512;
              while (static_cast<int64_t>(aa) * (b + 1) * (b + 1) < (1 << 28)) {
                b += 1;
              }
              int estimate = (b + 1) / 2;  // [256,511]
              r = static_cast<uint32_t>(estimate) << 23;
            }
          }
          memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
        }
        break;
      }

      // SADDLV/UADDLV: add-long across vector. Sum all
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
            int64_t s = SignExtendElem(elem, bits);
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

      // SUQADD / USQADD: per-lane saturating accumulate.
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
                static_cast<__int128_t>(SignExtendElem(src_elem, bits));
            sum = static_cast<__int128_t>(dst_elem) + s_src;
            __int128_t max_u = (bits >= 64) ? ((static_cast<__int128_t>(1) << 64) - 1)
                                            : ((static_cast<__int128_t>(1) << bits) - 1);
            if (sum < 0) sum = 0;
            else if (sum > max_u) sum = max_u;
          } else {
            // Vd signed, Vn unsigned.
            __int128_t s_dst =
                static_cast<__int128_t>(SignExtendElem(dst_elem, bits));
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

      // pairwise add long instructions
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
            int64_t sa = SignExtendElem(a, bits);
            int64_t sb = SignExtendElem(b, bits);
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
        // Q=0 (64-bit destination, vec_len==8) must zero bits[127:64]. The
        // accumulate seed copied the full prior Vd, so clear the unused upper
        // half rather than leaving stale destination bytes there.
        if (vec_len < 16) {
          memset(reinterpret_cast<uint8_t*>(&result) + vec_len, 0, 16 - vec_len);
        }
        break;
      }

      // saturating extract narrow: UQXTN / SQXTN.
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
            int64_t s = SignExtendElem(raw, src_esize * 8);
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

      // SQXTUN / SQXTUN2 (signed saturating extract unsigned narrow).
      // Read each lane as a signed value of width 2*esize, clamp to the
      // unsigned destination range [0, UMAX_dst], and write as an unsigned
      // value of width esize. Q=0 writes low 64 bits of Vd (upper zeroed);
      // Q=1 writes upper 64 bits and preserves lower (SQXTUN2 form).
      // Decoder rejects size=11 (no narrow form), so src_esize is at most 8.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun: {
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_count = 16 / src_esize;
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        uint64_t dst_umax = ElementMask(esize);
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t raw = 0;
          memcpy(&raw, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          // Sign-extend the source to int64_t.
          int64_t s = SignExtendElem(raw, src_esize * 8);
          uint64_t out;
          if (s < 0) {
            out = 0;
          } else if (static_cast<uint64_t>(s) > dst_umax) {
            out = dst_umax;
          } else {
            out = static_cast<uint64_t>(s);
          }
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &out, esize);
        }
        break;
      }

      // floating-point convert long / narrow.
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
        if (args.size == 0b01) {
          // FP32 -> FP64 widen (2 lanes). FCVTL2 (Q=1) takes the high 2 floats.
          uint8_t src_off = args.q ? 8 : 0;
          for (uint8_t i = 0; i < 2; i++) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + src_off + i * 4, 4);
            double d = static_cast<double>(f);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &d, 8);
          }
        } else if (args.size == 0b00) {
          // FP16 -> FP32 widen (4 lanes). FCVTL2 (Q=1) takes the high 4 halves.
          // Widening is exact, so the result is independent of rounding mode.
          uint8_t src_off = args.q ? 8 : 0;
          for (uint8_t i = 0; i < 4; i++) {
            uint16_t h;
            memcpy(&h, reinterpret_cast<const uint8_t*>(&src) + src_off + i * 2, 2);
            _Float16 hf;
            memcpy(&hf, &h, 2);
            float f = static_cast<float>(hf);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &f, 4);
          }
        } else {
          Undefined();
          return;
        }
        break;
      }
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtn: {
        if (args.size == 0b01) {
          // FP64 -> FP32 narrow (2 lanes). FCVTN2 (Q=1) writes the high half.
          result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
          uint8_t dst_off = args.q ? 8 : 0;
          for (uint8_t i = 0; i < 2; i++) {
            double d;
            memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            float f = static_cast<float>(d);
            memcpy(reinterpret_cast<uint8_t*>(&result) + dst_off + i * 4, &f, 4);
          }
        } else if (args.size == 0b00) {
          // FP32 -> FP16 narrow (4 lanes), round-to-nearest-even (the _Float16
          // conversion matches the JIT's VCVTPS2PH imm8=0). FCVTN2 (Q=1) writes
          // the high 64 bits and preserves the low half.
          result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
          uint8_t dst_off = args.q ? 8 : 0;
          for (uint8_t i = 0; i < 4; i++) {
            float f;
            memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            _Float16 hf = static_cast<_Float16>(f);
            uint16_t h;
            memcpy(&h, &hf, 2);
            memcpy(reinterpret_cast<uint8_t*>(&result) + dst_off + i * 2, &h, 2);
          }
        } else {
          Undefined();
          return;
        }
        break;
      }
      // BFCVTN/BFCVTN2 (vector narrow FP32->BF16).
      // BFCVTN  (Q=0): writes 4 BF16 lanes into Vd.h[0..3]; upper 64 bits zeroed.
      // BFCVTN2 (Q=1): writes 4 BF16 lanes into Vd.h[4..7]; lower 64 bits preserved.
      // Decoder pins size=10 (only valid FP32 source).
      case Decoder::AdvSimdTwoRegMiscOpcode::kBfcvtn: {
        if (args.size != 0b10) { Undefined(); return; }
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_off = args.q ? 8 : 0;
        for (uint8_t i = 0; i < 4; i++) {
          float f;
          memcpy(&f, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
          uint16_t bf = FloatToBf16(f);
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_off + i * 2, &bf, 2);
        }
        break;
      }
      // FCVTXN / FCVTXN2 (vector narrow FP64->FP32, RtO).
      // Decoder pins size=01 (FP64 source). Q=0 writes 2 narrow FP32 lanes
      // into Vd.s[0..1] (upper 64 bits zeroed); Q=1 writes Vd.s[2..3]
      // (lower 64 bits preserved). Round-to-odd is computed by
      // FpDoubleToFloatRtO from the FP64 bits — manual implementation
      // because the host compiler does not expose RtO.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtxn: {
        if (args.size != 0b01) { Undefined(); return; }
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_off = args.q ? 8 : 0;
        for (uint8_t i = 0; i < 2; i++) {
          double d;
          memcpy(&d, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
          uint32_t f_bits = FpDoubleToFloatRtO(d);
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_off + i * 4, &f_bits, 4);
        }
        break;
      }

      // Less critical ops: leave as undefined for now.
      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = result;
  }

  //
  // AdvSIMD shift by immediate: SSHR, USHR, SHL, SSRA, USRA, SLI, SRI, SHRN, SSHLL, USHLL.
  //
  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs& args) {
    CHECK(!exception_raised_);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t result = state_->cpu.v[args.rd];

    // integer MUL/MLA/MLS by-element (.4h/.8h size=01,
    // .2s/.4s size=10).  Integer word-element ops share the FP32-sized
    // size=10 encoding slot with FMLA/FMLS/FMUL/FMULX, so we cannot
    // dispatch by size alone — opcode wins.  The decoder already filters
    // size and (U, opcode) tuples; only valid integer encodings reach
    // here.
    if (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kMul ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kMla ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kMls) {
      if (args.size != 0b01 && args.size != 0b10) {
        Undefined();
        return;
      }
      uint8_t esize = (args.size == 0b01) ? 2 : 4;
      uint8_t num_elements = (args.q ? 16 : 8) / esize;
      uint64_t emask = ElementMask(esize);

      uint64_t indexed = 0;
      memcpy(&indexed,
             reinterpret_cast<const uint8_t*>(&src_m) + args.index * esize,
             esize);

      for (uint8_t i = 0; i < num_elements; i++) {
        uint64_t src = 0;
        memcpy(&src,
               reinterpret_cast<const uint8_t*>(&src_n) + i * esize,
               esize);
        uint64_t r;
        switch (args.opcode) {
          case Decoder::AdvSimdVecXIdxOpcode::kMul:
            r = (src * indexed) & emask;
            break;
          case Decoder::AdvSimdVecXIdxOpcode::kMla: {
            uint64_t dst = 0;
            memcpy(&dst,
                   reinterpret_cast<uint8_t*>(&result) + i * esize,
                   esize);
            r = (dst + src * indexed) & emask;
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kMls: {
            uint64_t dst = 0;
            memcpy(&dst,
                   reinterpret_cast<uint8_t*>(&result) + i * esize,
                   esize);
            r = (dst - src * indexed) & emask;
            break;
          }
          default:
            __builtin_unreachable();
        }
        memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &r, esize);
      }

      // For Q=0, the architecture zeroes the upper 64 bits of Vd
      // (D-register semantics) — matches the JIT lite_translator path.
      ClearUpperIfNotQ(&result, args.q);
      state_->cpu.v[args.rd] = result;
      return;
    }

    // SQDMULH / SQRDMULH by-element (vector).
    //   size=0b01 -> halfword (.4h/.8h, esize=2, 16-bit lanes).
    //   size=0b10 -> word     (.2s/.4s, esize=4, 32-bit lanes).
    // The indexed lane of Vm is broadcast across all destination lanes,
    // then the per-lane saturating-doubling-multiply-high (with optional
    // rounding for SQRDMULH) runs identically to the three-same vector
    // form at lines 4564-4590.  For Q=0 (D-register), upper 64 bits of Vd
    // are zeroed.
    if (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqdmulhIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqrdmulhIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        Undefined();
        return;
      }
      uint8_t esize = (args.size == 0b01) ? 2 : 4;
      uint8_t num_elements = (args.q ? 16 : 8) / esize;
      uint8_t bits_local = esize * 8;
      int64_t smax = (1LL << (bits_local - 1)) - 1;
      int64_t smin = -(1LL << (bits_local - 1));
      __int128 round =
          (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqrdmulhIdx)
              ? (static_cast<__int128>(1) << (bits_local - 1))
              : __int128{0};

      // Broadcast Vm.lane[index] across all 16/esize lanes of a synthetic
      // vector, then feed it through the existing per-lane signed helper.
      uint64_t indexed = 0;
      memcpy(&indexed,
             reinterpret_cast<const uint8_t*>(&src_m) + args.index * esize,
             esize);
      __uint128_t bcast = 0;
      for (uint8_t i = 0; i < (16 / esize); i++) {
        memcpy(reinterpret_cast<uint8_t*>(&bcast) + i * esize, &indexed, esize);
      }

      AdvSimdThreeSameElementWiseSigned(
          src_n, bcast, esize, num_elements, &result,
          [smax, smin, round, bits_local](int64_t a, int64_t b) -> int64_t {
            __int128 product = (static_cast<__int128>(2) * a * b) + round;
            int64_t high = static_cast<int64_t>(product >> bits_local);
            if (high > smax) return smax;
            if (high < smin) return smin;
            return high;
          });

      ClearUpperIfNotQ(&result, args.q);
      state_->cpu.v[args.rd] = result;
      return;
    }

    // SQRDMLAH / SQRDMLSH by-element (Armv8.1-RDM).
    //   size=0b01 -> halfword (.4h/.8h, esize=2).
    //   size=0b10 -> word     (.2s/.4s, esize=4).
    // Non-widening: destination lane width matches source lane width.
    // Per-lane semantics (ARM ARM):
    //   addend = SignedSat((2 * sn * sm + round) >> esize_bits)   // = SQRDMULH(sn,sm)
    //   SQRDMLAH: Vd[i] = SignedSat(Vd[i] + addend)
    //   SQRDMLSH: Vd[i] = SignedSat(Vd[i] - addend)
    // The addend lives in the narrow range [INT_MIN_out, INT_MAX_out] (SQRDMULH
    // saturates), so accumulator add/sub is done in the next wider signed type
    // (int32 for halfword, int64 for word) and re-saturated to the narrow range.
    // SQRDMLSH reuses the SQRDMLAH path with `addend = -addend`; safe because
    // `addend != INT_MIN_out` is only reachable when SQRDMULH itself rounded the
    // (INT_MIN_in, INT_MIN_in) corner to INT_MAX_out (no INT_MIN_out output).
    if (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqrdmlahIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqrdmlshIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        Undefined();
        return;
      }
      uint8_t esize = (args.size == 0b01) ? 2 : 4;
      uint8_t num_elements = (args.q ? 16 : 8) / esize;
      uint8_t bits_local = esize * 8;
      int64_t smax = (1LL << (bits_local - 1)) - 1;
      int64_t smin = -(1LL << (bits_local - 1));
      __int128 round = static_cast<__int128>(1) << (bits_local - 1);
      bool is_sub = (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqrdmlshIdx);

      uint64_t indexed_u = 0;
      memcpy(&indexed_u,
             reinterpret_cast<const uint8_t*>(&src_m) + args.index * esize,
             esize);
      uint8_t shift = (8 - esize) * 8;
      int64_t indexed_s =
          static_cast<int64_t>(indexed_u << shift) >> shift;

      for (uint8_t i = 0; i < num_elements; i++) {
        uint64_t lane_u = 0;
        memcpy(&lane_u,
               reinterpret_cast<const uint8_t*>(&src_n) + i * esize,
               esize);
        int64_t lane_s =
            static_cast<int64_t>(lane_u << shift) >> shift;

        // Stage 1: SQRDMULH(lane_s, indexed_s).  The doubled product
        // 2*lane_s*indexed_s fits int64_t for esize <= 4 (|product| <= 2^62).
        __int128 product =
            (static_cast<__int128>(2) * lane_s * indexed_s) + round;
        int64_t addend = static_cast<int64_t>(product >> bits_local);
        if (addend > smax) addend = smax;
        if (addend < smin) addend = smin;

        // Stage 2: signed-saturating add/sub of addend into Vd[i].
        uint64_t acc_u = 0;
        memcpy(&acc_u,
               reinterpret_cast<const uint8_t*>(&result) + i * esize,
               esize);
        int64_t acc = static_cast<int64_t>(acc_u << shift) >> shift;
        int64_t out = is_sub ? acc - addend : acc + addend;
        if (out > smax) out = smax;
        if (out < smin) out = smin;
        uint64_t out_u = static_cast<uint64_t>(out);
        memcpy(reinterpret_cast<uint8_t*>(&result) + i * esize, &out_u, esize);
      }

      ClearUpperIfNotQ(&result, args.q);
      state_->cpu.v[args.rd] = result;
      return;
    }

    // widening MUL/MAC by element.
    //   size=01 (.4h/.8h sources -> .4s dst, esize 2->4, 4 output lanes).
    //   size=10 (.2s/.4s sources -> .2d dst, esize 4->8, 2 output lanes).
    // Q=0 (SMULL/SMLAL/SMLSL form):  uses Vn low half  (bytes 0..7).
    // Q=1 (SMULL2/SMLAL2/SMLSL2):    uses Vn high half (bytes 8..15).
    // Destination is always 128-bit regardless of Q.  Vm.lane[index] is
    // broadcast as the per-lane multiplier.  Wraparound matches the
    // three-diff vector form's accumulate semantics.
    if (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSmullIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kUmullIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSmlalIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kUmlalIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSmlslIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kUmlslIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        Undefined();
        return;
      }
      uint8_t in_esize = (args.size == 0b01) ? 2 : 4;
      uint8_t out_esize = in_esize * 2;
      uint8_t num_elements = 16 / out_esize;
      uint8_t src_offset = args.q ? 8 : 0;

      bool is_signed = (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSmullIdx ||
                        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSmlalIdx ||
                        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSmlslIdx);
      bool is_accum  = (args.opcode != Decoder::AdvSimdVecXIdxOpcode::kSmullIdx &&
                        args.opcode != Decoder::AdvSimdVecXIdxOpcode::kUmullIdx);
      bool is_sub    = (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSmlslIdx ||
                        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kUmlslIdx);

      uint64_t indexed_u = 0;
      memcpy(&indexed_u,
             reinterpret_cast<const uint8_t*>(&src_m) + args.index * in_esize,
             in_esize);
      uint8_t shift = (8 - in_esize) * 8;
      int64_t indexed_s =
          static_cast<int64_t>(indexed_u << shift) >> shift;

      __uint128_t new_result = is_accum ? result : __uint128_t{0};
      for (uint8_t i = 0; i < num_elements; i++) {
        uint64_t lane_u = 0;
        memcpy(&lane_u,
               reinterpret_cast<const uint8_t*>(&src_n) + src_offset + i * in_esize,
               in_esize);
        uint64_t prod;
        if (is_signed) {
          int64_t lane_s =
              static_cast<int64_t>(lane_u << shift) >> shift;
          // For in_esize ∈ {2, 4}, |lane_s * indexed_s| ≤ 2^62, so the
          // signed multiply never overflows int64_t.
          prod = static_cast<uint64_t>(lane_s * indexed_s);
        } else {
          prod = lane_u * indexed_u;
        }
        if (is_accum) {
          uint64_t accum = 0;
          memcpy(&accum,
                 reinterpret_cast<const uint8_t*>(&new_result) + i * out_esize,
                 out_esize);
          uint64_t new_val = is_sub ? accum - prod : accum + prod;
          memcpy(reinterpret_cast<uint8_t*>(&new_result) + i * out_esize,
                 &new_val, out_esize);
        } else {
          memcpy(reinterpret_cast<uint8_t*>(&new_result) + i * out_esize,
                 &prod, out_esize);
        }
      }

      // The widening forms write a full 128-bit destination regardless of Q
      // (matches the three-diff vector path's lack of upper-half clearing).
      state_->cpu.v[args.rd] = new_result;
      return;
    }

    // SQDMULL / SQDMLAL / SQDMLSL by element.
    // Saturating doubling widening multiply (and accumulate / subtract),
    // signed only.  Mirrors the SQDMULL/SQDMLAL/SQDMLSL three-different
    // vector arms at lines ~2074-2163, with Vm.lane[index] broadcast as
    // the per-lane multiplier.  Two-stage saturation: stage 1 saturates
    // the doubled product against [INT_MIN_out, INT_MAX_out] (only the
    // (INT_MIN_in, INT_MIN_in) pair saturates), stage 2 saturates the
    // wide accumulator add/subtract.  Destination is always 128-bit.
    if (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqdmullIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqdmlalIdx ||
        args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqdmlslIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        Undefined();
        return;
      }
      bool is_accum = (args.opcode != Decoder::AdvSimdVecXIdxOpcode::kSqdmullIdx);
      bool is_sub   = (args.opcode == Decoder::AdvSimdVecXIdxOpcode::kSqdmlslIdx);

      uint8_t in_esize = (args.size == 0b01) ? 2 : 4;
      uint8_t out_esize = in_esize * 2;
      uint8_t num_elements = 16 / out_esize;
      uint8_t src_offset = args.q ? 8 : 0;

      int64_t int_min_in = -(1LL << (in_esize * 8 - 1));
      int64_t int_max_out =
          (out_esize == 8) ? INT64_MAX : ((1LL << (out_esize * 8 - 1)) - 1);
      int64_t int_min_out =
          (out_esize == 8) ? INT64_MIN : -(1LL << (out_esize * 8 - 1));
      uint8_t shift = (8 - in_esize) * 8;

      uint64_t indexed_u = 0;
      memcpy(&indexed_u,
             reinterpret_cast<const uint8_t*>(&src_m) + args.index * in_esize,
             in_esize);
      int64_t indexed_s = static_cast<int64_t>(indexed_u << shift) >> shift;

      __uint128_t new_result = is_accum ? result : __uint128_t{0};
      for (uint8_t i = 0; i < num_elements; i++) {
        uint64_t lane_u = 0;
        memcpy(&lane_u,
               reinterpret_cast<const uint8_t*>(&src_n) + src_offset + i * in_esize,
               in_esize);
        int64_t sn = static_cast<int64_t>(lane_u << shift) >> shift;

        int64_t doubled;
        if (sn == int_min_in && indexed_s == int_min_in) {
          doubled = int_max_out;
        } else {
          doubled = 2 * sn * indexed_s;
        }

        if (is_accum) {
          int64_t addend = is_sub ? -doubled : doubled;
          uint64_t acc_raw = 0;
          memcpy(&acc_raw,
                 reinterpret_cast<const uint8_t*>(&new_result) + i * out_esize,
                 out_esize);
          int64_t acc;
          if (out_esize == 8) {
            acc = static_cast<int64_t>(acc_raw);
          } else {
            uint8_t s2 = (8 - out_esize) * 8;
            acc = static_cast<int64_t>(acc_raw << s2) >> s2;
          }
          int64_t sum;
          if (out_esize == 8) {
            uint64_t usum =
                static_cast<uint64_t>(acc) + static_cast<uint64_t>(addend);
            sum = static_cast<int64_t>(usum);
            bool a_neg = acc < 0;
            bool b_neg = addend < 0;
            bool s_neg = sum < 0;
            if (a_neg == b_neg && a_neg != s_neg) {
              sum = a_neg ? int_min_out : int_max_out;
            }
          } else {
            sum = acc + addend;
            if (sum > int_max_out) {
              sum = int_max_out;
            } else if (sum < int_min_out) {
              sum = int_min_out;
            }
          }
          memcpy(reinterpret_cast<uint8_t*>(&new_result) + i * out_esize,
                 &sum, out_esize);
        } else {
          memcpy(reinterpret_cast<uint8_t*>(&new_result) + i * out_esize,
                 &doubled, out_esize);
        }
      }
      state_->cpu.v[args.rd] = new_result;
      return;
    }

    // FP16 vector indexed FMLA/FMLS/FMUL
    if (args.size == 0b00) {
      // Half-precision: 2 bytes per lane.  Q=0 (.4h) → 4 output lanes,
      // Q=1 (.8h) → 8 output lanes.  Index selects one lane (0..7) from
      // Vm.8H to broadcast as the multiplier.  Promote via FpHalfToSingle,
      // op in binary32 (exact for any single FP16 multiply / FMA goes
      // through binary64 std::fma per reasoning), narrow via
      // FpSingleToHalf — same round-trip pattern as three-same.
      uint8_t num_elements = args.q ? 8 : 4;
      uint16_t hm_indexed;
      memcpy(&hm_indexed, reinterpret_cast<const uint8_t*>(&src_m) + args.index * 2, 2);
      float m_indexed = FpHalfToSingle(hm_indexed);
      for (uint8_t i = 0; i < num_elements; i++) {
        uint16_t hn, hd, hr;
        memcpy(&hn, reinterpret_cast<const uint8_t*>(&src_n) + i * 2, 2);
        memcpy(&hd, reinterpret_cast<const uint8_t*>(&result) + i * 2, 2);
        float n = FpHalfToSingle(hn);
        float d = FpHalfToSingle(hd);
        float r;
        switch (args.opcode) {
          case Decoder::AdvSimdVecXIdxOpcode::kFmla: {
            double r_d = std::fma(static_cast<double>(n),
                                  static_cast<double>(m_indexed),
                                  static_cast<double>(d));
            r = static_cast<float>(r_d);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmls: {
            double r_d = std::fma(static_cast<double>(-n),
                                  static_cast<double>(m_indexed),
                                  static_cast<double>(d));
            r = static_cast<float>(r_d);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmul:
            r = n * m_indexed;
            break;
          case Decoder::AdvSimdVecXIdxOpcode::kFmulx:
            r = FmulxScalar<float>(n, m_indexed);
            break;
          default:
            Undefined();
            return;
        }
        hr = FpSingleToHalf(r);
        memcpy(reinterpret_cast<uint8_t*>(&result) + i * 2, &hr, 2);
      }
      // For Q=0, clear the upper half of the destination register.
      ClearUpperIfNotQ(&result, args.q);
    } else if (args.size == 0b10) {
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
            float r = std::fma(src, indexed, dst);  // FUSED (single rounding)
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmls: {
            float dst;
            memcpy(&dst, reinterpret_cast<uint8_t*>(&result) + i * 4, 4);
            float r = std::fma(-src, indexed, dst);  // FUSED
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmul: {
            float r = src * indexed;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &r, 4);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmulx: {
            float r = FmulxScalar<float>(src, indexed);
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
            double r = std::fma(src, indexed, dst);  // FUSED (single rounding)
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmls: {
            double dst;
            memcpy(&dst, reinterpret_cast<uint8_t*>(&result) + i * 8, 8);
            double r = std::fma(-src, indexed, dst);  // FUSED
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmul: {
            double r = src * indexed;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
            break;
          }
          case Decoder::AdvSimdVecXIdxOpcode::kFmulx: {
            double r = FmulxScalar<double>(src, indexed);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &r, 8);
            break;
          }
          default:
            Undefined();
            return;
        }
      }
    } else {
      // size=01 with non-integer opcodes: decoder already filters this,
      // but be defensive — integer kMul/kMla/kMls were handled at the
      // top of the function; anything else with size=01 is reserved.
      Undefined();
      return;
    }

    // For Q=0, the architecture zeroes the upper 64 bits of Vd (D-register
    // semantics) for every FP by-element form — the FP32/FP64 branches above
    // fill only the low lanes, so apply the zeroing here (the FP16 branch
    // already zeroed, making this idempotent for it). Matches the lite and
    // heavy JIT paths (SetVRegFull(rd, res, q)).
    ClearUpperIfNotQ(&result, args.q);
    state_->cpu.v[args.rd] = result;
  }

  // AdvSIMD scalar x indexed element (sibling of vector form above).
  // Reads one lane from Vn (lane 0), one lane from Vm (args.index), computes
  // FmulxScalar, writes the result to Vd lane 0 and zeros the upper lanes.
  void AdvSimdScalarXIndexedElement(const Decoder::AdvSimdScalarXIdxArgs& args) {
    CHECK(!exception_raised_);

    using Op = typename Decoder::AdvSimdScalarXIdxOpcode;

    // SQRDMLAH / SQRDMLSH scalar by-element (Armv8.1-RDM).
    //   Vd = sat_signed(
    //       Vd ± sat_signed((2 * sext(Vn) * sext(Vm[index]) + round) >>
    //                       bits_local))
    //   bits_local = 16 (size=01, H) or 32 (size=10, S).  Reads exactly one
    //   esize-wide lane from Vn and the indexed lane of Vm; saturates to
    //   the signed narrow range at both stages; zero-extends the result
    //   into the 128-bit Vd.
    if (args.opcode == Op::kSqrdmlahScalarIdx ||
        args.opcode == Op::kSqrdmlshScalarIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        Undefined();
        return;
      }
      const uint8_t esize_rdm = uint8_t{1} << args.size;  // 2 or 4
      const uint8_t bits_local = esize_rdm * 8;           // 16 or 32
      const int shift_to_64 = 64 - bits_local;
      __uint128_t src_n_rdm = state_->cpu.v[args.rn];
      __uint128_t src_m_rdm = state_->cpu.v[args.rm];
      __uint128_t src_d_rdm = state_->cpu.v[args.rd];
      uint64_t n_u = 0, m_u = 0, d_u = 0;
      memcpy(&n_u, reinterpret_cast<const uint8_t*>(&src_n_rdm), esize_rdm);
      memcpy(&m_u,
             reinterpret_cast<const uint8_t*>(&src_m_rdm) + args.index * esize_rdm,
             esize_rdm);
      memcpy(&d_u, reinterpret_cast<const uint8_t*>(&src_d_rdm), esize_rdm);
      const int64_t sa = static_cast<int64_t>(n_u << shift_to_64) >> shift_to_64;
      const int64_t sb = static_cast<int64_t>(m_u << shift_to_64) >> shift_to_64;
      const int64_t sd = static_cast<int64_t>(d_u << shift_to_64) >> shift_to_64;
      const __int128 round = static_cast<__int128>(1) << (bits_local - 1);
      const __int128 product = (static_cast<__int128>(2) * sa * sb) + round;
      int64_t addend = static_cast<int64_t>(product >> bits_local);
      const int64_t smax = (int64_t{1} << (bits_local - 1)) - 1;
      const int64_t smin = -(int64_t{1} << (bits_local - 1));
      addend = SatClampSigned(addend, smin, smax);
      const bool is_sub = (args.opcode == Op::kSqrdmlshScalarIdx);
      int64_t res = is_sub ? sd - addend : sd + addend;
      res = SatClampSigned(res, smin, smax);
      const uint64_t mask = (uint64_t{1} << bits_local) - 1;
      state_->cpu.v[args.rd] =
          static_cast<__uint128_t>(static_cast<uint64_t>(res) & mask);
      return;
    }

    // SQDMULH / SQRDMULH scalar by-element (H/S). Signed saturating doubling
    // multiply returning the high half; SQRDMULH rounds. Same single-lane
    // recipe as the scalar three-same form but reads Vm.lane[index].
    if (args.opcode == Op::kSqdmulhScalarIdx ||
        args.opcode == Op::kSqrdmulhScalarIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        Undefined();
        return;
      }
      const uint8_t esize_q = uint8_t{1} << args.size;  // 2 or 4
      const uint8_t bits_local = esize_q * 8;           // 16 or 32
      const int shift_to_64 = 64 - bits_local;
      __uint128_t src_n_q = state_->cpu.v[args.rn];
      __uint128_t src_m_q = state_->cpu.v[args.rm];
      uint64_t n_u = 0, m_u = 0;
      memcpy(&n_u, reinterpret_cast<const uint8_t*>(&src_n_q), esize_q);
      memcpy(&m_u,
             reinterpret_cast<const uint8_t*>(&src_m_q) + args.index * esize_q,
             esize_q);
      const int64_t sa = static_cast<int64_t>(n_u << shift_to_64) >> shift_to_64;
      const int64_t sb = static_cast<int64_t>(m_u << shift_to_64) >> shift_to_64;
      const __int128 round = (args.opcode == Op::kSqrdmulhScalarIdx)
                                 ? (static_cast<__int128>(1) << (bits_local - 1))
                                 : __int128{0};
      const __int128 product = (static_cast<__int128>(2) * sa * sb) + round;
      int64_t res = static_cast<int64_t>(product >> bits_local);
      const int64_t smax = (int64_t{1} << (bits_local - 1)) - 1;
      const int64_t smin = -(int64_t{1} << (bits_local - 1));
      res = SatClampSigned(res, smin, smax);
      const uint64_t mask = (uint64_t{1} << bits_local) - 1;
      state_->cpu.v[args.rd] =
          static_cast<__uint128_t>(static_cast<uint64_t>(res) & mask);
      return;
    }

    // SQDMULL scalar by-element (H->S or S->D). Signed saturating doubling
    // multiply long: the doubled product widened to 2x the element width; only
    // (INT_MIN_in, INT_MIN_in) saturates (to INT_MAX_out). Mirrors the
    // three-different SQDMULL, single lane, reading Vm.lane[index].
    if (args.opcode == Op::kSqdmullScalarIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        Undefined();
        return;
      }
      const uint8_t in_esize = uint8_t{1} << args.size;  // 2 (H) or 4 (S)
      const uint8_t out_esize = static_cast<uint8_t>(in_esize * 2);  // 4 or 8
      const int in_shift = 64 - in_esize * 8;
      __uint128_t src_n_l = state_->cpu.v[args.rn];
      __uint128_t src_m_l = state_->cpu.v[args.rm];
      uint64_t n_u = 0, m_u = 0;
      memcpy(&n_u, reinterpret_cast<const uint8_t*>(&src_n_l), in_esize);
      memcpy(&m_u,
             reinterpret_cast<const uint8_t*>(&src_m_l) + args.index * in_esize,
             in_esize);
      const int64_t sn = static_cast<int64_t>(n_u << in_shift) >> in_shift;
      const int64_t sm = static_cast<int64_t>(m_u << in_shift) >> in_shift;
      const int64_t int_min_in = -(int64_t{1} << (in_esize * 8 - 1));
      const int64_t int_max_out =
          (out_esize == 8) ? INT64_MAX : ((int64_t{1} << (out_esize * 8 - 1)) - 1);
      const int64_t doubled =
          (sn == int_min_in && sm == int_min_in) ? int_max_out : (2 * sn * sm);
      const uint64_t out_mask =
          (out_esize == 8) ? ~uint64_t{0} : ((uint64_t{1} << (out_esize * 8)) - 1);
      state_->cpu.v[args.rd] =
          static_cast<__uint128_t>(static_cast<uint64_t>(doubled) & out_mask);
      return;
    }

    if (args.size != 0b00 && args.size != 0b10 && args.size != 0b11) {
      Undefined();
      return;
    }
    const bool is_double = (args.size == 0b11);
    const bool is_fp16 = (args.size == 0b00);
    const uint8_t esize = is_double ? 8 : (is_fp16 ? 2 : 4);
    const bool needs_dst = (args.opcode == Op::kFmla || args.opcode == Op::kFmls);

    __uint128_t src_n = state_->cpu.v[args.rn];
    __uint128_t src_m = state_->cpu.v[args.rm];
    __uint128_t src_d = needs_dst ? state_->cpu.v[args.rd] : __uint128_t{0};
    __uint128_t result = 0;

    // FP16 scalar by-element FMLA/FMLS/FMUL/FMULX
    // (Armv8.2-FP16). Lift each FP16 lane to FP32 via FpHalfToSingle, run
    // the scalar op in FP32 (single rounding for FMUL/FMULX; binary64 for
    // FMLA/FMLS to mirror the FP32 FMLA path's single-round binary64
    // accumulate), narrow result back via FpSingleToHalf. Mirrors the
    // existing AdvSimdScalarThreeSame FP16 interpreter shape.
    if (is_fp16) {
      uint16_t hn, hm, hd = 0;
      memcpy(&hn, reinterpret_cast<const uint8_t*>(&src_n), 2);
      memcpy(&hm,
             reinterpret_cast<const uint8_t*>(&src_m) + args.index * 2,
             2);
      if (needs_dst) {
        memcpy(&hd, reinterpret_cast<const uint8_t*>(&src_d), 2);
      }
      float a = FpHalfToSingle(hn);
      float b = FpHalfToSingle(hm);
      float d = FpHalfToSingle(hd);
      uint16_t rh;
      switch (args.opcode) {
        case Op::kFmulx:
          rh = FpSingleToHalf(FmulxScalar<float>(a, b));
          break;
        case Op::kFmul:
          rh = FpSingleToHalf(a * b);
          break;
        case Op::kFmla: {
          double r64 = std::fma(static_cast<double>(a),
                                static_cast<double>(b),
                                static_cast<double>(d));
          rh = FpSingleToHalf(static_cast<float>(r64));
          break;
        }
        case Op::kFmls: {
          double r64 = std::fma(static_cast<double>(-a),
                                static_cast<double>(b),
                                static_cast<double>(d));
          rh = FpSingleToHalf(static_cast<float>(r64));
          break;
        }
        default:
          Undefined();
          return;
      }
      memcpy(reinterpret_cast<uint8_t*>(&result), &rh, 2);
      state_->cpu.v[args.rd] = result;
      return;
    }

    if (is_double) {
      double n_lane, m_lane, d_lane = 0.0;
      memcpy(&n_lane, reinterpret_cast<const uint8_t*>(&src_n), esize);
      memcpy(&m_lane, reinterpret_cast<const uint8_t*>(&src_m) + args.index * esize, esize);
      if (needs_dst) {
        memcpy(&d_lane, reinterpret_cast<const uint8_t*>(&src_d), esize);
      }
      double r;
      switch (args.opcode) {
        case Op::kFmulx:
          r = FmulxScalar<double>(n_lane, m_lane);
          break;
        case Op::kFmul:
          r = n_lane * m_lane;
          break;
        case Op::kFmla:
          r = d_lane + n_lane * m_lane;
          break;
        case Op::kFmls:
          r = d_lane - n_lane * m_lane;
          break;
        default:
          Undefined();
          return;
      }
      memcpy(reinterpret_cast<uint8_t*>(&result), &r, esize);
    } else {
      float n_lane, m_lane, d_lane = 0.0f;
      memcpy(&n_lane, reinterpret_cast<const uint8_t*>(&src_n), esize);
      memcpy(&m_lane, reinterpret_cast<const uint8_t*>(&src_m) + args.index * esize, esize);
      if (needs_dst) {
        memcpy(&d_lane, reinterpret_cast<const uint8_t*>(&src_d), esize);
      }
      float r;
      switch (args.opcode) {
        case Op::kFmulx:
          r = FmulxScalar<float>(n_lane, m_lane);
          break;
        case Op::kFmul:
          r = n_lane * m_lane;
          break;
        case Op::kFmla:
          r = d_lane + n_lane * m_lane;
          break;
        case Op::kFmls:
          r = d_lane - n_lane * m_lane;
          break;
        default:
          Undefined();
          return;
      }
      memcpy(reinterpret_cast<uint8_t*>(&result), &r, esize);
    }

    state_->cpu.v[args.rd] = result;
  }

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
    // Scalar shift-by-immediate (DecodeAdvSimdScalarShiftByImm) sets
    // args.scalar=true to force single-lane semantics regardless of
    // esize.  For B/H/S esize this overrides the vector lane count
    // (16/8/4/2/1) → 1, so only the lowest element is read/written;
    // the zero-initialised result keeps Vd[127:esize] = 0 per ARM ARM.
    uint8_t num_elements = args.scalar ? 1 : (vec_len / esize);
    uint64_t emask = ElementMask(esize);

    switch (args.opcode) {
      case Decoder::AdvSimdShiftImmOpcode::kSshr: {
        // SSHR: signed shift right by immediate.
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = SignExtendElem(elem, bits);
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
          int64_t signed_val = SignExtendElem(elem, bits);
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
        // Scalar narrow-shift dispatch (DecodeAdvSimdScalarShiftByImm)
        // sets args.scalar=true (and q=false) to mean: read exactly one
        // source element of src_esize bits and write Vd[esize-1:0],
        // leaving Vd[127:esize] = 0. The dst_offset / result-init
        // expressions already produce dst_offset=0 and result=0 when
        // q=false, so only src_count needs the per-scalar override.
        uint8_t src_count = args.scalar ? 1 : (16 / src_esize);
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
          int64_t signed_val = SignExtendElem(elem, src_bits);
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

      case Decoder::AdvSimdShiftImmOpcode::kSrshr: {
        // SRSHR: signed rounding shift right.
        for (uint8_t i = 0; i < num_elements; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * esize, esize);
          int64_t signed_val = SignExtendElem(elem, bits);
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
          int64_t signed_val = SignExtendElem(elem, bits);
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
          int64_t signed_val = SignExtendElem(elem, bits);
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
          int64_t signed_val = SignExtendElem(elem, bits);
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
        // see kShrn above for the scalar override rationale.
        uint8_t src_count = args.scalar ? 1 : (16 / src_esize);
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
        // see kShrn above for the scalar override rationale.
        uint8_t src_count = args.scalar ? 1 : (16 / src_esize);
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
          int64_t signed_val = SignExtendElem(elem, src_bits);
          int64_t shifted = (narrow_rshift >= src_bits)
                                ? (signed_val >> (src_bits - 1))
                                : (signed_val >> narrow_rshift);
          shifted = SatClampSigned(shifted, sat_min, sat_max);
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
        // see kShrn above for the scalar override rationale.
        uint8_t src_count = args.scalar ? 1 : (16 / src_esize);
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

      // AdvSIMD narrow-shift family (saturating, rounding,
      // and signed-saturating-unsigned variants). The non-rounding non-unsigned
      // variants live above (kShrn / kRshrn / kSqshrn / kUqshrn); these are
      // the four that were previously missing from the decoder and thus had
      // no interpreter case either. Each computes
      //   shifted_i = (signed_or_unsigned)src_i >> rshift     (with rounding for
      //                                                        the "R" forms)
      //   sat       = clamp(shifted_i, dst_min, dst_max)
      //   write     = sat narrowed to esize
      // and writes lower-half (Q=0) / upper-half (Q=1) of Vd.
      case Decoder::AdvSimdShiftImmOpcode::kSqshrun: {
        // SQSHRUN: signed saturating shift right unsigned narrow.
        // Source is signed at 2*esize bits, result is unsigned at esize bits,
        // negative shifted values saturate to 0.
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        // see kShrn above for the scalar override rationale.
        uint8_t src_count = args.scalar ? 1 : (16 / src_esize);
        uint8_t narrow_rshift = src_bits - ((immh << 3) | args.immb);
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        uint64_t sat_max = ElementMask(esize);
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          int64_t signed_val = SignExtendElem(elem, src_bits);
          int64_t shifted = (narrow_rshift >= src_bits)
                                ? (signed_val >> (src_bits - 1))
                                : (signed_val >> narrow_rshift);
          uint64_t clamped = SatClampUnsigned(shifted, sat_max);
          uint64_t r = clamped & sat_max;
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSqrshrun: {
        // SQRSHRUN: signed saturating rounding shift right unsigned narrow.
        // narrow_rshift is always in [1, esize] for valid encodings, so the
        // simple signed-rounding path below is safe (no overflow in int128).
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        // see kShrn above for the scalar override rationale.
        uint8_t src_count = args.scalar ? 1 : (16 / src_esize);
        uint8_t narrow_rshift = src_bits - ((immh << 3) | args.immb);
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        uint64_t sat_max = ElementMask(esize);
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          int64_t signed_val = SignExtendElem(elem, src_bits);
          __int128_t wide = static_cast<__int128_t>(signed_val) + (__int128_t{1} << (narrow_rshift - 1));
          int64_t shifted = static_cast<int64_t>(wide >> narrow_rshift);
          uint64_t clamped = SatClampUnsigned(shifted, sat_max);
          uint64_t r = clamped & sat_max;
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kSqrshrn: {
        // SQRSHRN: signed saturating rounding shift right narrow.
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        // see kShrn above for the scalar override rationale.
        uint8_t src_count = args.scalar ? 1 : (16 / src_esize);
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
          int64_t signed_val = SignExtendElem(elem, src_bits);
          __int128_t wide = static_cast<__int128_t>(signed_val) + (__int128_t{1} << (narrow_rshift - 1));
          int64_t shifted = static_cast<int64_t>(wide >> narrow_rshift);
          shifted = SatClampSigned(shifted, sat_min, sat_max);
          uint64_t r = static_cast<uint64_t>(shifted) & narrow_mask;
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &r, esize);
        }
        break;
      }

      case Decoder::AdvSimdShiftImmOpcode::kUqrshrn: {
        // UQRSHRN: unsigned saturating rounding shift right narrow.
        uint8_t src_esize = esize * 2;
        if (src_esize > 8) { Undefined(); return; }
        uint8_t src_bits = src_esize * 8;
        // see kShrn above for the scalar override rationale.
        uint8_t src_count = args.scalar ? 1 : (16 / src_esize);
        uint8_t narrow_rshift = src_bits - ((immh << 3) | args.immb);
        result = args.q ? state_->cpu.v[args.rd] : static_cast<__uint128_t>(0);
        uint8_t dst_offset = args.q ? 8 : 0;
        uint64_t sat_max = ElementMask(esize);
        for (uint8_t i = 0; i < src_count; i++) {
          uint64_t elem = 0;
          memcpy(&elem, reinterpret_cast<const uint8_t*>(&src) + i * src_esize, src_esize);
          __uint128_t wide = static_cast<__uint128_t>(elem) + (__uint128_t{1} << (narrow_rshift - 1));
          uint64_t shifted = static_cast<uint64_t>(wide >> narrow_rshift);
          if (shifted > sat_max) shifted = sat_max;
          uint64_t r = shifted & sat_max;
          memcpy(reinterpret_cast<uint8_t*>(&result) + dst_offset + i * esize, &r, esize);
        }
        break;
      }

      // fixed-point conversion (SCVTF / UCVTF / FCVTZS /
      // FCVTZU at S/D, both scalar and vector forms). esize is 4 (S) or 8
      // (D); FP16 (esize=2) is rejected by the decoder. rshift is the
      // encoded fbits: esize=4 -> rshift ∈ [1..32]; esize=8 -> rshift ∈
      // [1..64]. num_elements honors args.scalar (=1) vs Q (.2s/.4s/.2d).
      // Result is left-zeroed at function entry, satisfying ARM ARM's
      // "Vd[127:esize*num_elements] = 0" requirement.
      case Decoder::AdvSimdShiftImmOpcode::kScvtfFixed: {
        if (esize != 4 && esize != 8) { Undefined(); return; }
        // 2^rshift. ldexp avoids the UB of `1 << 64` for the .2D max-fbits
        // corner (rshift==64), where a host shift count == width silently
        // collapsed the scale to 1.0.
        double scale = std::ldexp(1.0, rshift);
        for (uint8_t i = 0; i < num_elements; i++) {
          if (esize == 4) {
            int32_t ival;
            memcpy(&ival, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            float fval = static_cast<float>(static_cast<double>(ival) / scale);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &fval, 4);
          } else {
            int64_t ival;
            memcpy(&ival, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            double fval = static_cast<double>(ival) / scale;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &fval, 8);
          }
        }
        break;
      }
      case Decoder::AdvSimdShiftImmOpcode::kUcvtfFixed: {
        if (esize != 4 && esize != 8) { Undefined(); return; }
        // 2^rshift. ldexp avoids the UB of `1 << 64` for the .2D max-fbits
        // corner (rshift==64), where a host shift count == width silently
        // collapsed the scale to 1.0.
        double scale = std::ldexp(1.0, rshift);
        for (uint8_t i = 0; i < num_elements; i++) {
          if (esize == 4) {
            uint32_t ival;
            memcpy(&ival, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            float fval = static_cast<float>(static_cast<double>(ival) / scale);
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &fval, 4);
          } else {
            uint64_t ival;
            memcpy(&ival, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            double fval = static_cast<double>(ival) / scale;
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &fval, 8);
          }
        }
        break;
      }
      case Decoder::AdvSimdShiftImmOpcode::kFcvtzsFixed: {
        if (esize != 4 && esize != 8) { Undefined(); return; }
        // 2^rshift. ldexp avoids the UB of `1 << 64` for the .2D max-fbits
        // corner (rshift==64), where a host shift count == width silently
        // collapsed the scale to 1.0.
        double scale = std::ldexp(1.0, rshift);
        for (uint8_t i = 0; i < num_elements; i++) {
          if (esize == 4) {
            float fval;
            memcpy(&fval, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            double scaled = static_cast<double>(fval) * scale;
            int32_t ival;
            if (std::isnan(scaled)) {
              ival = 0;
            } else if (scaled >= static_cast<double>(INT32_MAX)) {
              ival = INT32_MAX;
            } else if (scaled <= static_cast<double>(INT32_MIN)) {
              ival = INT32_MIN;
            } else {
              ival = static_cast<int32_t>(trunc(scaled));
            }
            uint32_t uval;
            memcpy(&uval, &ival, sizeof(uval));
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &uval, 4);
          } else {
            double fval;
            memcpy(&fval, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            double scaled = fval * scale;
            int64_t ival;
            if (std::isnan(scaled)) {
              ival = 0;
            } else if (scaled >= static_cast<double>(INT64_MAX)) {
              ival = INT64_MAX;
            } else if (scaled <= static_cast<double>(INT64_MIN)) {
              ival = INT64_MIN;
            } else {
              ival = static_cast<int64_t>(trunc(scaled));
            }
            uint64_t uval;
            memcpy(&uval, &ival, sizeof(uval));
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &uval, 8);
          }
        }
        break;
      }
      case Decoder::AdvSimdShiftImmOpcode::kFcvtzuFixed: {
        if (esize != 4 && esize != 8) { Undefined(); return; }
        // 2^rshift. ldexp avoids the UB of `1 << 64` for the .2D max-fbits
        // corner (rshift==64), where a host shift count == width silently
        // collapsed the scale to 1.0.
        double scale = std::ldexp(1.0, rshift);
        for (uint8_t i = 0; i < num_elements; i++) {
          if (esize == 4) {
            float fval;
            memcpy(&fval, reinterpret_cast<const uint8_t*>(&src) + i * 4, 4);
            double scaled = static_cast<double>(fval) * scale;
            uint32_t ival;
            if (std::isnan(scaled) || scaled < 0.0) {
              ival = 0;
            } else if (scaled >= static_cast<double>(UINT32_MAX)) {
              ival = UINT32_MAX;
            } else {
              ival = static_cast<uint32_t>(trunc(scaled));
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 4, &ival, 4);
          } else {
            double fval;
            memcpy(&fval, reinterpret_cast<const uint8_t*>(&src) + i * 8, 8);
            double scaled = fval * scale;
            uint64_t ival;
            if (std::isnan(scaled) || scaled < 0.0) {
              ival = 0;
            } else if (scaled >= static_cast<double>(UINT64_MAX)) {
              ival = UINT64_MAX;
            } else {
              ival = static_cast<uint64_t>(trunc(scaled));
            }
            memcpy(reinterpret_cast<uint8_t*>(&result) + i * 8, &ival, 8);
          }
        }
        break;
      }

      default:
        Undefined();
        return;
    }

    state_->cpu.v[args.rd] = result;
  }

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
  // Program host x86 MXCSR rounding mode + FTZ/DAZ from an ARM FPCR value.
  // FPSR exception-flag infrastructure: writing FPCR via MSR must
  // program MXCSR rounding mode + DAZ/FTZ. Called from the kFpcr MSR case.
  //
  //   ARM FPCR[23:22] RMode: 00=RNE, 01=RP(+inf), 10=RM(-inf), 11=RZ
  //   x86 MXCSR[14:13] RC:   00=RNE, 01=RD(-inf), 10=RU(+inf), 11=RZ
  //
  // The middle two encodings are swapped, so a 4-entry lookup table is the
  // clearest mapping. FZ -> FTZ+DAZ because ARM FZ flushes both inputs and
  // outputs, while x86 splits the two into separate bits. Exception masks
  // (MXCSR bits 7-12) are forced to 1 so host FP never raises SIGFPE; we
  // emulate exception flag reporting via emulated_fpsr instead. Other FPCR
  // fields (DN, AHP, IDE/IXE/.../IOE, FZ16) have no clean x86 analog.
  static void ProgramHostMxcsrFromFpcr(uint32_t fpcr) {
#if defined(__x86_64__)
    uint32_t mxcsr;
    asm volatile("stmxcsr %0" : "=m"(mxcsr));
    // Clear RC (bits 13-14), FTZ (bit 15), DAZ (bit 6).
    mxcsr &= ~((0b11u << 13) | (1u << 15) | (1u << 6));
    // Force all exception masks set (bits 7-12) so host FP doesn't trap.
    mxcsr |= (0b111111u << 7);
    static constexpr uint8_t kArmRmodeToX86Rc[4] = {
        0b00,  // ARM RNE  -> x86 RNE
        0b10,  // ARM RP   -> x86 RU
        0b01,  // ARM RM   -> x86 RD
        0b11,  // ARM RZ   -> x86 RZ
    };
    uint32_t arm_rmode = (fpcr >> 22) & 0b11;
    mxcsr |= static_cast<uint32_t>(kArmRmodeToX86Rc[arm_rmode]) << 13;
    if (fpcr & (1u << 24)) {  // FPCR.FZ
      mxcsr |= (1u << 15) | (1u << 6);
    }
    asm volatile("ldmxcsr %0" : : "m"(mxcsr));
#else
    (void)fpcr;
#endif
  }

  // Clear the host x86 MXCSR cumulative exception flags (bits 0-5: IE/DE/ZE/
  // OE/UE/PE). Called from the kFpsr MSR write case so that a
  // guest-side FPSR clear also resets the host sticky bits — otherwise the
  // next MRS read would re-mirror stale exceptions the guest believed
  // cleared.
  static void ClearHostMxcsrExceptions() {
#if defined(__x86_64__)
    uint32_t mxcsr;
    asm volatile("stmxcsr %0" : "=m"(mxcsr));
    mxcsr &= ~0b111111u;
    asm volatile("ldmxcsr %0" : : "m"(mxcsr));
#endif
  }

  // Read host x86 MXCSR cumulative exception flags, map them to ARM FPSR
  // bit positions, and OR (cumulatively) into emulated_fpsr.
  // Called from the kFpsr MRS read case (lazy mirror). MXCSR sticky bits
  // accumulate across all host FP ops — both interpreter and JIT-emitted —
  // so this single readback at MRS time captures the full cumulative
  // exception state without per-op JIT instrumentation.
  //
  //   MXCSR[0] IE (invalid)   -> FPSR[0] IOC
  //   MXCSR[1] DE (denormal)  -> FPSR[7] IDC
  //   MXCSR[2] ZE (div-zero)  -> FPSR[1] DZC
  //   MXCSR[3] OE (overflow)  -> FPSR[2] OFC
  //   MXCSR[4] UE (underflow) -> FPSR[3] UFC
  //   MXCSR[5] PE (inexact)   -> FPSR[4] IXC
  //
  // Bit positions per ARM ARM C5.2.8 (FPSR) and Intel SDM 11.6.6 (MXCSR).
  // The flags are sticky on both architectures, so OR-into matches the
  // cumulative ARM semantics naturally.
  void MirrorHostMxcsrToFpsr() {
#if defined(__x86_64__)
    uint32_t mxcsr;
    asm volatile("stmxcsr %0" : "=m"(mxcsr));
    uint32_t exc = mxcsr & 0b111111u;
    uint32_t fpsr_add =
        (((exc >> 0) & 1u) << 0) |   // IE -> IOC
        (((exc >> 1) & 1u) << 7) |   // DE -> IDC
        (((exc >> 2) & 1u) << 1) |   // ZE -> DZC
        (((exc >> 3) & 1u) << 2) |   // OE -> OFC
        (((exc >> 4) & 1u) << 3) |   // UE -> UFC
        (((exc >> 5) & 1u) << 4);    // PE -> IXC
    state_->cpu.emulated_fpsr |= fpsr_add;
#endif
  }

  // Compute element mask: all-ones for element of esize bytes.
  // Avoids UB from (1ULL << 64) when esize == 8.
  static uint64_t ElementMask(uint8_t esize) {
    return (esize >= 8) ? ~0ULL : ((1ULL << (esize * 8)) - 1);
  }

  // Zero the upper 64 bits of a vector value when Q=0 (D-register semantics).
  static void ClearUpperIfNotQ(__uint128_t* value, bool q) {
    if (!q) {
      *value = static_cast<uint64_t>(*value);
    }
  }

  // Sign-extend the low `bits` bits of x to int64_t.
  static int64_t SignExtendElem(uint64_t x, unsigned bits) {
    return static_cast<int64_t>(x << (64 - bits)) >> (64 - bits);
  }

  // Clamp a signed value to the inclusive [min, max] range.
  static int64_t SatClampSigned(int64_t value, int64_t min, int64_t max) {
    if (value > max) return max;
    if (value < min) return min;
    return value;
  }

  // Clamp a signed value to the unsigned [0, max] range (negatives saturate to 0).
  static uint64_t SatClampUnsigned(int64_t value, uint64_t max) {
    if (value < 0) return 0;
    if (static_cast<uint64_t>(value) > max) return max;
    return static_cast<uint64_t>(value);
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
      int64_t sa = SignExtendElem(a, bits);
      int64_t sb = SignExtendElem(b, bits);
      int64_t sr = op(sa, sb);
      uint64_t r = static_cast<uint64_t>(sr) & mask;
      memcpy(reinterpret_cast<uint8_t*>(result) + i * esize, &r, esize);
    }
  }

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
    // ARM STLR is sequentially consistent (RCsc), not merely release: it orders
    // the store before any subsequent load. x86 release (a plain store) does not
    // give StoreLoad ordering, so use SEQ_CST (MFENCE / locked xchg on x86).
    __atomic_store_n(static_cast<T*>(addr), static_cast<T>(val), __ATOMIC_SEQ_CST);
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

  // 128-bit compare-and-swap for CASP (64-bit pair).
  // Emit LOCK CMPXCHG16B via inline asm so we don't take a libatomic dependency
  // (the bare `__atomic_compare_exchange_n` on `__uint128_t` may lower to a
  // libcall without -mcx16).  The Digitalis host is x86_64; the upstream ARM64
  // build does not link this file (interpreter.h is host-only).
  __uint128_t AtomicCASVal128(void* addr, __uint128_t expected, __uint128_t desired) {
    uint64_t exp_lo = static_cast<uint64_t>(expected);
    uint64_t exp_hi = static_cast<uint64_t>(expected >> 64);
    uint64_t new_lo = static_cast<uint64_t>(desired);
    uint64_t new_hi = static_cast<uint64_t>(desired >> 64);
    // GCC/clang inline asm: LOCK CMPXCHG16B [addr]
    //   In : RAX=exp_lo, RDX=exp_hi, RBX=new_lo, RCX=new_hi, mem=*addr
    //   Out: RAX=old_lo, RDX=old_hi (always — independent of success).
    asm volatile(
        "lock cmpxchg16b %[mem]\n"
        : "+a"(exp_lo), "+d"(exp_hi),
          [mem] "+m"(*static_cast<__uint128_t*>(addr))
        : "b"(new_lo), "c"(new_hi)
        : "cc", "memory");
    return (static_cast<__uint128_t>(exp_hi) << 64) | exp_lo;
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

  // atomic min/max (LSE Armv8.1).
  // No __atomic_fetch_max/min builtin exists; emulate via a CAS retry loop.
  // T is the signed/unsigned host type at the guest operation size; the
  // returned uint64_t is the prior memory value, zero-extended.
  template <typename T>
  uint64_t AtomicFetchSMax(void* addr, uint64_t operand) {
    T* p = static_cast<T*>(addr);
    T op_v = static_cast<T>(operand);
    T cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    T desired;
    do {
      desired = (cur > op_v) ? cur : op_v;
    } while (!__atomic_compare_exchange_n(p, &cur, desired, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
    return static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(cur));
  }

  template <typename T>
  uint64_t AtomicFetchSMin(void* addr, uint64_t operand) {
    T* p = static_cast<T*>(addr);
    T op_v = static_cast<T>(operand);
    T cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    T desired;
    do {
      desired = (cur < op_v) ? cur : op_v;
    } while (!__atomic_compare_exchange_n(p, &cur, desired, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
    return static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(cur));
  }

  template <typename T>
  uint64_t AtomicFetchUMax(void* addr, uint64_t operand) {
    T* p = static_cast<T*>(addr);
    T op_v = static_cast<T>(operand);
    T cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    T desired;
    do {
      desired = (cur > op_v) ? cur : op_v;
    } while (!__atomic_compare_exchange_n(p, &cur, desired, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
    return static_cast<uint64_t>(cur);
  }

  template <typename T>
  uint64_t AtomicFetchUMin(void* addr, uint64_t operand) {
    T* p = static_cast<T*>(addr);
    T op_v = static_cast<T>(operand);
    T cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    T desired;
    do {
      desired = (cur < op_v) ? cur : op_v;
    } while (!__atomic_compare_exchange_n(p, &cur, desired, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
    return static_cast<uint64_t>(cur);
  }

  //
  // SIMD helpers.
  //

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

  // FP32 -> FP16 round-to-nearest-even. Distinct from FpSingleToHalf above,
  // which truncates (round-toward-zero). This matches x86 VCVTPS2PH imm=0 and
  // ARM's default FPCR rounding, so int->fp16 conversions (SCVTF/UCVTF, ftype=11)
  // agree with the JIT lowering. Verified bit-identical to hardware F16C over the
  // entire 2^32 FP32 space. The rounding carry is allowed to propagate from the
  // mantissa into the exponent naturally (an overflow to exponent 0x1F yields the
  // +/-Inf pattern), which is why no explicit post-round overflow fix is needed.
  static uint16_t FpSingleToHalfRN(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e = (x >> 23) & 0xFF;
    uint32_t m = x & 0x7FFFFFu;
    if (e == 255) {  // Inf / NaN
      return static_cast<uint16_t>(m ? (sign | 0x7E00u | (m >> 13)) : (sign | 0x7C00u));
    }
    int32_t big_e = static_cast<int32_t>(e) - 127 + 15;  // rebiased fp16 exponent
    if (big_e >= 0x1F) {
      return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> Inf
    }
    if (big_e <= 0) {  // subnormal fp16 or zero
      if (big_e < -10) {
        return static_cast<uint16_t>(sign);  // too small -> +/-0
      }
      m |= 0x800000u;
      int shift = 14 - big_e;
      uint32_t half = 1u << (shift - 1);
      uint32_t frac = m >> shift;
      uint32_t rem = m & ((1u << shift) - 1);
      if (rem > half || (rem == half && (frac & 1))) {
        frac++;  // carry into the exponent bit -> smallest normal, which is correct
      }
      return static_cast<uint16_t>(sign | frac);
    }
    uint32_t half = 1u << 12;
    uint32_t frac = m >> 13;
    uint32_t rem = m & 0x1FFFu;
    uint16_t out = static_cast<uint16_t>((static_cast<uint32_t>(big_e) << 10) | frac);
    if (rem > half || (rem == half && (frac & 1))) {
      out++;  // carry propagates into the exponent (0x1F,0 == Inf) automatically
    }
    return static_cast<uint16_t>(sign | out);
  }

  // FMULX scalar semantics, parameterized by FP type.  Same as a * b except
  // the (zero * infinity) saturation case is replaced by ±2.0 (sign = sign of
  // a XOR sign of b), per ARM ARM C7.2.149 FMULX.  Standard FP NaN
  // propagation otherwise.
  template <typename FpType>
  static FpType FmulxScalar(FpType a, FpType b) {
    if ((a == FpType{0} && std::isinf(b)) ||
        (std::isinf(a) && b == FpType{0})) {
      bool neg = std::signbit(a) ^ std::signbit(b);
      return neg ? FpType{-2} : FpType{2};
    }
    return a * b;
  }

  // FRECPS scalar semantics: (2.0 - a*b), used as a Newton-Raphson refinement
  // step after FRECPE.  Per ARM ARM C7.2.151 FRECPS:
  //   - if either operand is NaN, the default NaN is returned
  //   - if exactly one operand is ±0 and the other is ±inf, result is +2.0
  //     (unsigned — unlike FMULX which preserves XOR sign)
  //   - otherwise: FPRecipStepFused = FMA(-a, b, 2.0)  (single rounding)
  template <typename FpType>
  static FpType FrecpsScalar(FpType a, FpType b) {
    if (std::isnan(a) || std::isnan(b)) {
      return std::numeric_limits<FpType>::quiet_NaN();
    }
    if ((a == FpType{0} && std::isinf(b)) ||
        (std::isinf(a) && b == FpType{0})) {
      return FpType{2};
    }
    return std::fma(-a, b, FpType{2});
  }

  // FRSQRTS scalar semantics: (3.0 - a*b)/2, used as a Newton-Raphson
  // refinement step after FRSQRTE.  Per ARM ARM C7.2.155 FRSQRTS:
  //   - if either operand is NaN, the default NaN is returned
  //   - if exactly one operand is ±0 and the other is ±inf, result is +1.5
  //   - otherwise: FPRSqrtStepFused = FMA(-a, b, 3.0) / 2  (FMA single rounding,
  //     then divide-by-two is exact since it's just an exponent decrement for
  //     non-denormals; for denormals the divide-by-two is also exact in IEEE
  //     binary FP)
  template <typename FpType>
  static FpType FrsqrtsScalar(FpType a, FpType b) {
    if (std::isnan(a) || std::isnan(b)) {
      return std::numeric_limits<FpType>::quiet_NaN();
    }
    if ((a == FpType{0} && std::isinf(b)) ||
        (std::isinf(a) && b == FpType{0})) {
      return FpType{1.5};
    }
    return std::fma(-a, b, FpType{3}) / FpType{2};
  }

  // FMAX / FMIN / FMAXNM / FMINNM scalar semantics, parameterized by FP type.
  // Per ARM ARM C7.2.130 FMAX, C7.2.134 FMIN, C7.2.131 FMAXNM, C7.2.135 FMINNM.
  //
  // The +0/-0 corner is the reason these helpers exist: C/C++ `a > b ? a : b`
  // returns whichever operand happens to come second when both are zero,
  // because `+0.0 == -0.0` is true and `+0.0 > -0.0` is false. ARM specifies a
  // definite sign for the zero result — FMAX(+0,-0) = +0, FMIN(+0,-0) = -0,
  // independent of operand order — so we disambiguate explicitly via signbit.
  //
  // Difference between FMAX/FMIN and FMAXNM/FMINNM is only NaN handling:
  //   - FMAX/FMIN: any NaN input -> default NaN output.
  //   - FMAXNM/FMINNM: single NaN input -> the other (non-NaN) operand;
  //                    both-NaN input -> default NaN.
  // The +0/-0 disambiguation is the same in both families.
  template <typename FpType>
  static FpType FmaxScalar(FpType a, FpType b) {
    if (std::isnan(a) || std::isnan(b)) {
      return std::numeric_limits<FpType>::quiet_NaN();
    }
    if (a == FpType{0} && b == FpType{0}) {
      return std::signbit(a) ? b : a;
    }
    return a > b ? a : b;
  }

  template <typename FpType>
  static FpType FminScalar(FpType a, FpType b) {
    if (std::isnan(a) || std::isnan(b)) {
      return std::numeric_limits<FpType>::quiet_NaN();
    }
    if (a == FpType{0} && b == FpType{0}) {
      return std::signbit(a) ? a : b;
    }
    return a < b ? a : b;
  }

  template <typename FpType>
  static FpType FmaxnmScalar(FpType a, FpType b) {
    if (std::isnan(a)) {
      return std::isnan(b) ? std::numeric_limits<FpType>::quiet_NaN() : b;
    }
    if (std::isnan(b)) {
      return a;
    }
    if (a == FpType{0} && b == FpType{0}) {
      return std::signbit(a) ? b : a;
    }
    return a > b ? a : b;
  }

  template <typename FpType>
  static FpType FminnmScalar(FpType a, FpType b) {
    if (std::isnan(a)) {
      return std::isnan(b) ? std::numeric_limits<FpType>::quiet_NaN() : b;
    }
    if (std::isnan(b)) {
      return a;
    }
    if (a == FpType{0} && b == FpType{0}) {
      return std::signbit(a) ? a : b;
    }
    return a < b ? a : b;
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

  // VFPExpandImm for 32-bit single precision (ARM ARM J1: VFPExpandImm).
  // imm8 = a:b:c:d:e:f:g:h. float = a : NOT(b) : b*5 : c:d : e:f:g:h:Zeros(19).
  static uint32_t VFPExpandImm32(uint8_t imm8) {
    uint32_t sign = (imm8 >> 7) & 1;
    uint32_t b = (imm8 >> 6) & 1;
    uint32_t exp = ((1 - b) << 7) | ((b ? 0x1Fu : 0u) << 2) | ((imm8 >> 4) & 0x3u);
    uint32_t mantissa = static_cast<uint32_t>(imm8 & 0xFu) << 19;
    return (sign << 31) | (exp << 23) | mantissa;
  }

  // VFPExpandImm for 64-bit double precision.
  // double = a : NOT(b) : b*8 : c:d : e:f:g:h:Zeros(48).
  static uint64_t VFPExpandImm64(uint8_t imm8) {
    uint64_t sign = (imm8 >> 7) & 1;
    uint64_t b = (imm8 >> 6) & 1;
    uint64_t exp = ((1 - b) << 10) | ((b ? 0xFFull : 0ull) << 2) | ((imm8 >> 4) & 0x3ull);
    uint64_t mantissa = static_cast<uint64_t>(imm8 & 0xFull) << 48;
    return (sign << 63) | (exp << 52) | mantissa;
  }

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
            // 8-bit, replicated (MOVI, cmode=0b1110)
            for (int i = 0; i < 8; i++) imm64 |= static_cast<uint64_t>(imm8) << (i * 8);
          } else {
            // FMOV (vector, immediate) single-precision (cmode=0b1111, op=0).
            // Expand imm8 to a 32-bit float and replicate across 32-bit lanes.
            uint32_t f = VFPExpandImm32(imm8);
            imm64 = static_cast<uint64_t>(f) | (static_cast<uint64_t>(f) << 32);
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
      } else if (cmode == 0b1111) {
        // FMOV (vector, immediate) double-precision (op=1, cmode=0b1111).
        // Expand imm8 to a 64-bit double; the element is 64-bit (.2d).
        imm64 = VFPExpandImm64(imm8);
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
    // use FaultyLoad for sizes <= 8 bytes, two loads for 128-bit
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
    }
  }

  void SimdStoreToMemory(void* host_addr, uint8_t rt, Decoder::SimdLoadStoreSize size) {
    // use FaultyStore for fault recovery
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
  }

  ThreadState* state_;
  bool branch_taken_;
  bool exception_raised_;
  // (removed: page cache variables no longer needed with FaultyLoad/FaultyStore)
};

}  // namespace berberis
