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

#include "frontend.h"

#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "berberis/assembler/x86_64.h"
#include "berberis/backend/common/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

namespace {

// Process-global heavy-tier bail histogram. Relaxed atomics: best-effort
// diagnostic counters, not a synchronization point. Only mutated under
// Tracing::IsOn(), so production builds never touch it.
struct HeavyBailStats {
  std::atomic<uint64_t> total{0};
  std::atomic<uint64_t> by_reason[static_cast<size_t>(BailReason::kCount)] = {};
};
HeavyBailStats g_heavy_bail_stats;

// Dump the histogram to the trace every N bails; 0 (default) disables periodic
// dumps. Mirrors GetGearUpMinInsns' ConfigStr pattern.
size_t GetHeavyBailDumpEvery() {
  static const size_t value = []() -> size_t {
    static ConfigStr config("BERBERIS_HEAVY_BAIL_DUMP_EVERY", "berberis.heavy_bail_dump_every");
    const char* str = config.get();
    if (str) {
      char* end = nullptr;
      unsigned long parsed = strtoul(str, &end, 10);
      if (end != str && (*end == '\0' || *end == '\n')) {
        return static_cast<size_t>(parsed);
      }
    }
    return 0;  // disabled
  }();
  return value;
}

void DumpHeavyBailStats() {
  TRACE("heavy-bail histogram: total=%" PRIu64,
        g_heavy_bail_stats.total.load(std::memory_order_relaxed));
  for (size_t i = 0; i < static_cast<size_t>(BailReason::kCount); ++i) {
    uint64_t n = g_heavy_bail_stats.by_reason[i].load(std::memory_order_relaxed);
    if (n != 0) {
      // NOTE: FormatBuffer (TRACE's formatter) does not support the '-' width
      // flag; keep the format to plain specifiers.
      TRACE("heavy-bail   %s=%" PRIu64, BailReasonName(static_cast<BailReason>(i)), n);
    }
  }
}

}  // namespace

using Register = HeavyOptimizerFrontend::Register;

int32_t HeavyOptimizerFrontend::GetThreadStateRegOffset(uint8_t reg) {
  return static_cast<int32_t>(offsetof(ThreadState, cpu.x[0]) + reg * sizeof(uint64_t));
}

int32_t HeavyOptimizerFrontend::GetThreadStateSpOffset() {
  return static_cast<int32_t>(offsetof(ThreadState, cpu.sp));
}

void HeavyOptimizerFrontend::GenJump(GuestAddr target) {
  auto map_it = branch_targets_.find(target);
  if (map_it == branch_targets_.end()) {
    // Remember that this address was taken to help region formation. If we
    // translate it later the data will be overwritten with the actual location.
    branch_targets_[target] = MachineInsnPosition{};
  }

  // Checking pending signals only on back jumps guarantees no infinite loops
  // without pending-signal checks.
  auto kind = target <= GetInsnAddr() ? PseudoJump::Kind::kJumpWithPendingSignalsCheck
                                      : PseudoJump::Kind::kJumpWithoutPendingSignalsCheck;

  builder_.Gen<PseudoJump>(target, kind);
}

void HeavyOptimizerFrontend::ExitGeneratedCode(GuestAddr target) {
  builder_.Gen<PseudoJump>(target, PseudoJump::Kind::kExitGeneratedCode);
}

void HeavyOptimizerFrontend::ExitRegionIndirect(Register target) {
  builder_.Gen<PseudoIndirectJump>(target);
}

// After a faulting host memory access, split off a recovery basic block that
// exits the region so the guest signal handler runs. This mechanism is
// guest-agnostic and is copied verbatim from heavy_optimizer/riscv64.
void HeavyOptimizerFrontend::GenRecoveryBlockForLastInsn() {
  auto* ir = builder_.ir();
  auto* current_bb = builder_.bb();
  auto* continue_bb = ir->NewBasicBlock();
  auto* recovery_bb = ir->NewBasicBlock();
  ir->AddEdge(current_bb, continue_bb);
  ir->AddEdge(current_bb, recovery_bb);

  builder_.SetRecoveryPointAtLastInsn(recovery_bb);

  // Note, even though there are two bb successors, we only explicitly branch to
  // the continue_bb, since jump to the recovery_bb is set up by the signal
  // handler.
  builder_.Gen<PseudoBranch>(continue_bb);

  builder_.StartBasicBlock(recovery_bb);
  ExitGeneratedCode(GetInsnAddr());

  builder_.StartBasicBlock(continue_bb);
}

//
// Branches.
//

// B (unconditional). SemanticsPlayer has already written X30 for the BL form.
void HeavyOptimizerFrontend::Branch(int32_t offset) {
  if (!success()) {
    return;
  }
  is_uncond_branch_ = true;
  GenJump(GetInsnAddr() + offset);
}

// BR / RET / BLR (indirect). SemanticsPlayer has already written X30 for BLR.
// Mirrors lite_translator.h::BranchRegister, which does NOT mask the top byte
// (no TBI): it simply exits indirect to `target`.
void HeavyOptimizerFrontend::BranchRegister(Register target) {
  if (!success()) {
    return;
  }
  is_uncond_branch_ = true;
  ExitRegionIndirect(target);
}

// Materialize a 0/1 predicate for ARM64 condition `cond` from the NZCV bits in
// ThreadState.cpu.flags. Bit positions and boolean algebra mirror
// lite_translator.h::EmitJumpIfCondNotMet exactly:
//   N = bit 15, Z = bit 14, C = bit 8, V = bit 0.
// The predicate is 1 iff the condition is satisfied. We extract each needed bit
// to its low position with a shift+and and combine with and/or/xor; "not" is
// xor with 1. kAl/kNv are unconditional and are handled by the caller before
// reaching here.
Register HeavyOptimizerFrontend::EmitArmCondPredicate(Decoder::Condition cond) {
  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));
  // Load the 16-bit NZCV word from ThreadState.cpu.flags. We use the 16-bit
  // MovwRegOp form (not MovzxwlRegOp) on purpose: RemoveLoopGuestContextAccesses
  // only recognizes MovwRegMemBaseDisp as a guest-context read of a 16-bit
  // field, so for an in-region loop the flags read must use this opcode to stay
  // consistent with the MovwOpReg write EmitMaterializeNZCV emits (otherwise the
  // optimizer caches the flag write in a register and the read keeps loading a
  // stale memory value, wedging the loop). MovwRegOp leaves the upper bits
  // untouched, so mask to the low 16 to get a clean zero-extended value.
  Register flags =
      std::get<0>(Gen<x86_64::MovwRegOp>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}));
  flags = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(flags, int32_t{0xFFFF}));

  // bit_to_low(pos): (flags >> pos) & 1, as a fresh 0/1 register.
  auto bit_to_low = [&](int8_t pos) -> Register {
    Register r = std::get<0>(Gen<x86_64::MovlRegReg>(flags));
    if (pos != 0) {
      r = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(r, pos));
    }
    return std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(r, int32_t{1}));
  };

  switch (cond) {
    case Decoder::Condition::kEq:  // Z==1
      return bit_to_low(kFlagZeroBit);
    case Decoder::Condition::kNe:  // Z==0
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(kFlagZeroBit), int32_t{1}));
    case Decoder::Condition::kCs:  // C==1
      return bit_to_low(kFlagCarryBit);
    case Decoder::Condition::kCc:  // C==0
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(kFlagCarryBit), int32_t{1}));
    case Decoder::Condition::kMi:  // N==1
      return bit_to_low(kFlagNegativeBit);
    case Decoder::Condition::kPl:  // N==0
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(kFlagNegativeBit), int32_t{1}));
    case Decoder::Condition::kVs:  // V==1
      return bit_to_low(kFlagOverflowBit);
    case Decoder::Condition::kVc:  // V==0
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(kFlagOverflowBit), int32_t{1}));
    case Decoder::Condition::kHi: {  // C==1 && Z==0
      Register c = bit_to_low(kFlagCarryBit);
      Register not_z = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(kFlagZeroBit), int32_t{1}));
      return std::get<0>(Gen<x86_64::AndlRegReg, kNoSSA>(c, not_z));
    }
    case Decoder::Condition::kLs: {  // C==0 || Z==1
      Register not_c = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(kFlagCarryBit), int32_t{1}));
      Register z = bit_to_low(kFlagZeroBit);
      return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(not_c, z));
    }
    case Decoder::Condition::kGe: {  // N==V  -> !(N^V)
      Register n = bit_to_low(kFlagNegativeBit);
      Register v = bit_to_low(kFlagOverflowBit);
      Register n_xor_v = std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(n, v));
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(n_xor_v, int32_t{1}));
    }
    case Decoder::Condition::kLt: {  // N!=V  -> N^V
      Register n = bit_to_low(kFlagNegativeBit);
      Register v = bit_to_low(kFlagOverflowBit);
      return std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(n, v));
    }
    case Decoder::Condition::kGt: {  // Z==0 && N==V
      Register not_z = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(kFlagZeroBit), int32_t{1}));
      Register n = bit_to_low(kFlagNegativeBit);
      Register v = bit_to_low(kFlagOverflowBit);
      Register n_xor_v = std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(n, v));
      Register n_eq_v = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(n_xor_v, int32_t{1}));
      return std::get<0>(Gen<x86_64::AndlRegReg, kNoSSA>(not_z, n_eq_v));
    }
    case Decoder::Condition::kLe: {  // Z==1 || N!=V
      Register z = bit_to_low(kFlagZeroBit);
      Register n = bit_to_low(kFlagNegativeBit);
      Register v = bit_to_low(kFlagOverflowBit);
      Register n_xor_v = std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(n, v));
      return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(z, n_xor_v));
    }
    case Decoder::Condition::kAl:
    case Decoder::Condition::kNv:
      // Unconditional: handled by the caller; never reached.
      CHECK(false);
      return flags;
  }
  CHECK(false);
  return flags;
}

// Branch to then_bb when `cond` is met, else_bb otherwise. The predicate is a
// 0/1 value; TestlRegReg sets ZF=1 when it is 0 (not taken) and ZF=0 when it is
// 1 (taken), so the then_bb is selected on kNotZero.
void HeavyOptimizerFrontend::EmitCondBranch(Decoder::Condition cond,
                                            MachineBasicBlock* then_bb,
                                            MachineBasicBlock* else_bb) {
  Register pred = EmitArmCondPredicate(cond);
  builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kNotZero,
                                 then_bb,
                                 else_bb,
                                 std::get<0>(Gen<x86_64::TestlRegReg>(pred, pred)));
}

// B.cond (conditional). AL/NV are unconditional. Otherwise split into a taken
// then_bb (GenJump to the target) and a fall-through else_bb in which
// translation continues.
void HeavyOptimizerFrontend::BranchCond(Decoder::Condition cond, int32_t offset) {
  if (!success()) {
    return;
  }
  GuestAddr target = GetInsnAddr() + offset;

  // AL/NV always branch: lower as an unconditional B.
  if (cond == Decoder::Condition::kAl || cond == Decoder::Condition::kNv) {
    is_uncond_branch_ = true;
    GenJump(target);
    return;
  }

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* else_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, else_bb);

  EmitCondBranch(cond, then_bb, else_bb);

  builder_.StartBasicBlock(then_bb);
  GenJump(target);

  // Continue translating the not-taken path. A backward target is handled as an
  // in-region back-edge by GenJump+ResolveJumps (with the pending-signal
  // check), so do NOT set is_uncond_branch_/region-end here.
  builder_.StartBasicBlock(else_bb);
}

// CBZ (is_nonzero=false) / CBNZ (is_nonzero=true). Test the source for zero and
// branch like B.cond, mirroring lite_translator.h::CompareAndBranch.
void HeavyOptimizerFrontend::CompareAndBranch(bool is_nonzero,
                                              bool is_64bit,
                                              Register src,
                                              int32_t offset) {
  if (!success()) {
    return;
  }
  GuestAddr target = GetInsnAddr() + offset;

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* else_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, else_bb);

  // TEST sets ZF=1 when src is zero. CBNZ branches when nonzero (ZF==0 ->
  // kNotZero); CBZ branches when zero (ZF==1 -> kZero).
  Register flags = is_64bit ? std::get<0>(Gen<x86_64::TestqRegReg>(src, src))
                            : std::get<0>(Gen<x86_64::TestlRegReg>(src, src));
  builder_.Gen<PseudoCondBranch>(
      is_nonzero ? x86_64::Assembler::Condition::kNotZero : x86_64::Assembler::Condition::kZero,
      then_bb,
      else_bb,
      flags);

  builder_.StartBasicBlock(then_bb);
  GenJump(target);

  builder_.StartBasicBlock(else_bb);
}

// TBZ (is_nonzero=false) / TBNZ (is_nonzero=true). BT of bit `bit` of src sets
// CF; branch like B.cond, mirroring lite_translator.h::TestAndBranch. Bt is a
// 64-bit-register op, so it covers bits 0..63 directly.
void HeavyOptimizerFrontend::TestAndBranch(bool is_nonzero,
                                           Register src,
                                           uint8_t bit,
                                           int32_t offset) {
  if (!success()) {
    return;
  }
  GuestAddr target = GetInsnAddr() + offset;

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* else_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, else_bb);

  // BTQ src, bit -> CF = bit of src. TBNZ branches when the bit is set
  // (CF==1 -> kCarry); TBZ branches when clear (CF==0 -> kNotCarry).
  Register flags = std::get<0>(Gen<x86_64::BtqRegImm>(src, static_cast<int8_t>(bit)));
  builder_.Gen<PseudoCondBranch>(
      is_nonzero ? x86_64::Assembler::Condition::kCarry : x86_64::Assembler::Condition::kNotCarry,
      then_bb,
      else_bb,
      flags);

  builder_.StartBasicBlock(then_bb);
  GenJump(target);

  builder_.StartBasicBlock(else_bb);
}

// CSEL/CSINC/CSINV/CSNEG. Materialize the false case (transform(src2)) into a
// result vreg, then conditionally overwrite it with src1 when `cond` holds.
// Mirrors lite_translator.h::ConditionalSelect. AL/NV always select src1.
Register HeavyOptimizerFrontend::ConditionalSelect(Decoder::ConditionalSelectOpcode opcode,
                                                   bool is_64bit,
                                                   Register src1,
                                                   Register src2,
                                                   Decoder::Condition cond) {
  if (!success()) {
    return AllocTempReg();
  }

  // The false case = transform(src2), and the true case = src1, each produced
  // into a fresh 64-bit-wide value (a 32-bit op clears the upper half, matching
  // ARM64 W-write semantics). They are funneled into a single `result` vreg via
  // PseudoCopy so that, after the conditional overwrite, `result` holds the
  // selected operand on every control-flow edge.
  auto width_adjust = [&](Register r) -> Register {
    if (is_64bit) {
      return Copy(r);
    }
    return std::get<0>(Gen<x86_64::MovlRegReg>(r));
  };

  Register false_val;
  switch (opcode) {
    case Decoder::ConditionalSelectOpcode::kCsel:
      false_val = width_adjust(src2);
      break;
    case Decoder::ConditionalSelectOpcode::kCsinc:
      false_val = width_adjust(src2);
      if (is_64bit) {
        false_val = std::get<0>(Gen<x86_64::AddqRegImm, kNoSSA>(false_val, int32_t{1}));
      } else {
        false_val = std::get<0>(Gen<x86_64::AddlRegImm, kNoSSA>(false_val, int32_t{1}));
      }
      break;
    case Decoder::ConditionalSelectOpcode::kCsinv:
      // ~src2. The heavy IR has only a 64-bit NOT; the 32-bit case re-clears the
      // upper half with a 32-bit mov afterwards.
      false_val = Copy(src2);
      false_val = std::get<0>(Gen<x86_64::NotqReg, kNoSSA>(false_val));
      if (!is_64bit) {
        false_val = std::get<0>(Gen<x86_64::MovlRegReg>(false_val));
      }
      break;
    case Decoder::ConditionalSelectOpcode::kCsneg: {
      // -src2 = 0 - src2 (the heavy IR has no Neg op). The l-suffix subtract
      // zero-extends the 32-bit result.
      Register zero = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
      if (is_64bit) {
        false_val = std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(zero, src2));
      } else {
        false_val = std::get<0>(Gen<x86_64::SublRegReg, kNoSSA>(zero, src2));
      }
      break;
    }
  }

  Register result = AllocTempReg();
  builder_.Gen<PseudoCopy>(result, false_val, 8);

  // AL/NV: always select src1 (unconditional). No branch needed.
  if (cond == Decoder::Condition::kAl || cond == Decoder::Condition::kNv) {
    builder_.Gen<PseudoCopy>(result, width_adjust(src1), 8);
    return result;
  }

  // Conditionally overwrite result with src1 when the condition holds. then_bb
  // does the overwrite; both paths fall into merge_bb. result is the same vreg
  // written on both edges, so its value after merge_bb is the selected operand.
  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, merge_bb);

  EmitCondBranch(cond, then_bb, merge_bb);

  builder_.StartBasicBlock(then_bb);
  builder_.Gen<PseudoCopy>(result, width_adjust(src1), 8);
  ir->AddEdge(then_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
  return result;
}

// UDIV Xd/Wd, Xn/Wn, Xm/Wm. ARM division never traps: if the divisor is 0 the
// result is 0. x86 DIV #DE-faults on a zero divisor, so guard it with a branch.
// Mirrors lite_translator.h::DataProc2Src kUdiv. The dividend goes into RAX and
// the high half (zero, unsigned) into RDX; the DivRegRegReg pseudo-op binds
// those fixed registers via the Gen<> SSA wrapper, so the divisor stays a free
// vreg. A 32-bit DIV writes EAX, which zero-extends to the X register.
Register HeavyOptimizerFrontend::EmitUDiv(bool is_64bit, Register src1, Register src2) {
  if (!success()) {
    return AllocTempReg();
  }

  Register result = AllocTempReg();
  // Divide-by-zero path writes 0; the divide path overwrites result with the
  // quotient. Both edges define `result`, so it holds the right value at merge.
  Register zero = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  builder_.Gen<PseudoCopy>(result, zero, 8);

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* div_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, div_bb);
  ir->AddEdge(cur_bb, merge_bb);

  // TEST sets ZF=1 when the divisor is zero -> skip the divide (result stays 0).
  Register flags = is_64bit ? std::get<0>(Gen<x86_64::TestqRegReg>(src2, src2))
                            : std::get<0>(Gen<x86_64::TestlRegReg>(src2, src2));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kZero, merge_bb, div_bb, flags);

  builder_.StartBasicBlock(div_bb);
  // High half of the dividend is 0 for unsigned division.
  Register hi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  Register quotient;
  if (is_64bit) {
    quotient = std::get<0>(Gen<x86_64::DivqRegRegReg>(src1, hi, src2));
  } else {
    quotient = std::get<0>(Gen<x86_64::DivlRegRegReg>(src1, hi, src2));
  }
  builder_.Gen<PseudoCopy>(result, quotient, 8);
  ir->AddEdge(div_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
  return result;
}

// SDIV Xd/Wd, Xn/Wn, Xm/Wm. ARM division never traps: Rm==0 -> 0, and the
// INT_MIN/-1 overflow case -> INT_MIN (x86 IDIV #DE-faults on both). Mirrors
// lite_translator.h::DataProc2Src kSdiv. Three blocks: zero divisor (result
// stays 0), divisor==-1 (result = -Rn, which is INT_MIN for INT_MIN input and
// correct for every other Rn), and the real IDIV (RDX = sign-extension of RAX).
Register HeavyOptimizerFrontend::EmitSDiv(bool is_64bit, Register src1, Register src2) {
  if (!success()) {
    return AllocTempReg();
  }

  Register result = AllocTempReg();
  Register zero = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  builder_.Gen<PseudoCopy>(result, zero, 8);

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* nonzero_bb = ir->NewBasicBlock();
  MachineBasicBlock* neg_one_bb = ir->NewBasicBlock();
  MachineBasicBlock* div_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();

  // Divisor == 0 -> result stays 0.
  ir->AddEdge(cur_bb, merge_bb);
  ir->AddEdge(cur_bb, nonzero_bb);
  Register zflags = is_64bit ? std::get<0>(Gen<x86_64::TestqRegReg>(src2, src2))
                             : std::get<0>(Gen<x86_64::TestlRegReg>(src2, src2));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kZero, merge_bb, nonzero_bb, zflags);

  // Divisor == -1 -> result = -Rn (= INT_MIN when Rn==INT_MIN, no IDIV).
  builder_.StartBasicBlock(nonzero_bb);
  ir->AddEdge(nonzero_bb, neg_one_bb);
  ir->AddEdge(nonzero_bb, div_bb);
  Register cflags = is_64bit
                        ? std::get<0>(Gen<x86_64::CmpqRegImm>(src2, int32_t{-1}))
                        : std::get<0>(Gen<x86_64::CmplRegImm>(src2, int32_t{-1}));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kEqual, neg_one_bb, div_bb, cflags);

  // result = 0 - Rn (the heavy IR has no NEG op; a 32-bit sub zero-extends).
  builder_.StartBasicBlock(neg_one_bb);
  Register negbase = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  Register neg;
  if (is_64bit) {
    neg = std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(negbase, src1));
  } else {
    neg = std::get<0>(Gen<x86_64::SublRegReg, kNoSSA>(negbase, src1));
  }
  builder_.Gen<PseudoCopy>(result, neg, 8);
  ir->AddEdge(neg_one_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  // Real division: RDX = sign-extension of the dividend (CQO/CDQ equivalent).
  builder_.StartBasicBlock(div_bb);
  Register quotient;
  if (is_64bit) {
    Register hi = std::get<0>(Gen<x86_64::SarqRegImm>(Copy(src1), int8_t{63}));
    quotient = std::get<0>(Gen<x86_64::IdivqRegRegReg>(src1, hi, src2));
  } else {
    Register hi = std::get<0>(Gen<x86_64::SarlRegImm>(
        std::get<0>(Gen<x86_64::MovlRegReg>(src1)), int8_t{31}));
    quotient = std::get<0>(Gen<x86_64::IdivlRegRegReg>(src1, hi, src2));
  }
  builder_.Gen<PseudoCopy>(result, quotient, 8);
  ir->AddEdge(div_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
  return result;
}

// CCMP/CCMN. If `cond` holds, set NZCV from a real CMP (is_neg=false) / CMN
// (is_neg=true); otherwise set NZCV from the 4-bit nzcv immediate. Mirrors
// lite_translator.cc::ConditionalCompare with then/else/merge basic blocks.
void HeavyOptimizerFrontend::ConditionalCompare(bool is_neg,
                                                bool is_64bit,
                                                Register rn,
                                                Register rm,
                                                Decoder::Condition cond,
                                                uint8_t nzcv) {
  if (!success()) {
    return;
  }

  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));

  // Pack the immediate-path NZCV: ARM bit3=N,bit2=Z,bit1=C,bit0=V map to
  // cpu.flags N@15, Z@14, C@8, V@0 (the same layout EmitMaterializeNZCV writes).
  auto emit_immediate_path = [&]() {
    uint16_t flags_val = 0;
    if (nzcv & 0x8) {
      flags_val |= CPUState::kFlagNegative;
    }
    if (nzcv & 0x4) {
      flags_val |= CPUState::kFlagZero;
    }
    if (nzcv & 0x2) {
      flags_val |= CPUState::kFlagCarry;
    }
    if (nzcv & 0x1) {
      flags_val |= CPUState::kFlagOverflow;
    }
    Register imm_reg = GetImm(flags_val);
    builder_.Gen<x86_64::MovwOpReg>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}, imm_reg);
  };

  // The compare path: CMP is non-destructive (CmpqRegReg only defs FLAGS); CMN
  // has no non-destructive x86 add, so add rn+rm into a scratch (never into rn,
  // which is the live guest register under register mapping) and take its flags.
  auto emit_compare_path = [&]() {
    Register flags;
    if (is_64bit) {
      if (is_neg) {
        Register tmp = Copy(rn);
        flags = std::get<1>(Gen<x86_64::AddqRegReg, kNoSSA>(tmp, rm));
      } else {
        flags = std::get<0>(Gen<x86_64::CmpqRegReg>(rn, rm));
      }
    } else {
      if (is_neg) {
        Register tmp = std::get<0>(Gen<x86_64::MovlRegReg>(rn));
        flags = std::get<1>(Gen<x86_64::AddlRegReg, kNoSSA>(tmp, rm));
      } else {
        flags = std::get<0>(Gen<x86_64::CmplRegReg>(rn, rm));
      }
    }
    EmitMaterializeNZCV(flags, /*is_sub=*/!is_neg);
  };

  // AL/NV: always the compare path (no branch).
  if (cond == Decoder::Condition::kAl || cond == Decoder::Condition::kNv) {
    emit_compare_path();
    return;
  }

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* cmp_bb = ir->NewBasicBlock();   // condition met -> real compare
  MachineBasicBlock* imm_bb = ir->NewBasicBlock();   // condition not met -> nzcv imm
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, cmp_bb);
  ir->AddEdge(cur_bb, imm_bb);

  EmitCondBranch(cond, cmp_bb, imm_bb);

  builder_.StartBasicBlock(cmp_bb);
  emit_compare_path();
  ir->AddEdge(cmp_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(imm_bb);
  emit_immediate_path();
  ir->AddEdge(imm_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
}

// Map the x86 EFLAGS a UCOMIS{S,D} left in `flags_vreg` to ARM64 FP NZCV and
// store the packed 16-bit word to ThreadState.cpu.flags. Bit-exact with
// lite_translator.h::EmitStoreArmFpNZCV, which branches on the x86 FLAGS
// directly; here the flags are first read into a GP register (PseudoReadFlags:
// LAHF + SETO) so the branch tree can test individual bits without keeping the
// single host FLAGS live across basic-block boundaries. In that GP word:
// CF@8, PF@10, ZF@14 (OF@0, unused here). Priority PF > ZF > CF matches lite:
//   PF set   -> unordered -> NZCV C,V   (0x0101)
//   ZF set   -> equal     -> NZCV Z,C   (0x4100)
//   CF set   -> less      -> NZCV N     (0x8000)
//   else     -> greater   -> NZCV C     (0x0100)
void HeavyOptimizerFrontend::EmitStoreArmFpNZCV(Register flags_vreg) {
  if (!success()) {
    return;
  }
  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));

  Register raw = AllocTempReg();
  builder_.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, raw, flags_vreg);

  auto* ir = builder_.ir();
  MachineBasicBlock* uo_bb = ir->NewBasicBlock();       // unordered (PF)
  MachineBasicBlock* chk_eq_bb = ir->NewBasicBlock();   // test ZF
  MachineBasicBlock* eq_bb = ir->NewBasicBlock();       // equal (ZF)
  MachineBasicBlock* chk_lt_bb = ir->NewBasicBlock();   // test CF
  MachineBasicBlock* lt_bb = ir->NewBasicBlock();       // less (CF)
  MachineBasicBlock* gt_bb = ir->NewBasicBlock();       // greater (default)
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();

  // Fill a leaf block: write the packed NZCV immediate and jump to merge.
  auto store_leaf = [&](uint16_t nzcv_word, MachineBasicBlock* bb) {
    builder_.StartBasicBlock(bb);
    Register imm = GetImm(nzcv_word);
    builder_.Gen<x86_64::MovwOpReg>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}, imm);
    ir->AddEdge(bb, merge_bb);
    builder_.Gen<PseudoBranch>(merge_bb);
  };

  // PF (bit 10) set -> unordered, else fall to the ZF test.
  auto* cur_bb = builder_.bb();
  ir->AddEdge(cur_bb, uo_bb);
  ir->AddEdge(cur_bb, chk_eq_bb);
  Register pf = std::get<0>(Gen<x86_64::TestlRegImm>(raw, int32_t{1 << 10}));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNotZero, uo_bb, chk_eq_bb, pf);

  // ZF (bit 14) set -> equal, else fall to the CF test.
  builder_.StartBasicBlock(chk_eq_bb);
  ir->AddEdge(chk_eq_bb, eq_bb);
  ir->AddEdge(chk_eq_bb, chk_lt_bb);
  Register zf = std::get<0>(Gen<x86_64::TestlRegImm>(raw, int32_t{1 << 14}));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNotZero, eq_bb, chk_lt_bb, zf);

  // CF (bit 8) set -> less, else greater.
  builder_.StartBasicBlock(chk_lt_bb);
  ir->AddEdge(chk_lt_bb, lt_bb);
  ir->AddEdge(chk_lt_bb, gt_bb);
  Register cf = std::get<0>(Gen<x86_64::TestlRegImm>(raw, int32_t{1 << 8}));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNotZero, lt_bb, gt_bb, cf);

  store_leaf(kFlagsFpUnordered, uo_bb);
  store_leaf(kFlagsFpEqual, eq_bb);
  store_leaf(kFlagsFpLess, lt_bb);
  store_leaf(kFlagsFpGreater, gt_bb);

  builder_.StartBasicBlock(merge_bb);
}

// FCMP/FCMPE Sn/Dn, Sm/Dm (or #0.0). Mirrors lite_translator.h::FpCompare:
// load lane 0 of the operands, UCOMIS{S,D}, then EmitStoreArmFpNZCV. S/D only;
// FP16 (ftype 0b11) and the reserved ftype 0b10 bail to the lite tier.
void HeavyOptimizerFrontend::FpCompare(const Decoder::FpCompareArgs& args) {
  if (!success()) {
    return;
  }
  if (args.ftype != 0b00 && args.ftype != 0b01) {
    Undefined();
    return;
  }
  const bool is_double = (args.ftype == 0b01);

  FpRegister xmm_n = GetVRegScalar(args.rn, is_double);
  FpRegister xmm_m = args.with_zero ? AllocZeroedSimdReg() : GetVRegScalar(args.rm, is_double);

  Register flags = is_double
                       ? std::get<0>(Gen<x86_64::UcomisdXRegXReg>(xmm_n.machine_reg(),
                                                                  xmm_m.machine_reg()))
                       : std::get<0>(Gen<x86_64::UcomissXRegXReg>(xmm_n.machine_reg(),
                                                                  xmm_m.machine_reg()));
  EmitStoreArmFpNZCV(flags);
}

// FCCMP/FCCMPE: if `cond` holds do the FCMP compare + NZCV mapping, else write
// the 4-bit nzcv immediate straight to cpu.flags. Same then/else/merge shape as
// ConditionalCompare; the compare path's EmitStoreArmFpNZCV creates its own
// sub-tree, so the edge into merge is taken from whatever block it leaves the
// builder in. S/D only; FP16 / reserved ftype bail.
void HeavyOptimizerFrontend::FpConditionalCompare(
    const Decoder::FpConditionalCompareArgs& args) {
  if (!success()) {
    return;
  }
  if (args.ftype != 0b00 && args.ftype != 0b01) {
    Undefined();
    return;
  }
  const bool is_double = (args.ftype == 0b01);
  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));

  // Pack the false-path NZCV immediate: bit3=N,bit2=Z,bit1=C,bit0=V map to
  // cpu.flags N@15, Z@14, C@8, V@0 (same layout EmitMaterializeNZCV writes).
  auto emit_immediate_path = [&]() {
    uint16_t flags_val = 0;
    if (args.nzcv & 0x8) {
      flags_val |= CPUState::kFlagNegative;
    }
    if (args.nzcv & 0x4) {
      flags_val |= CPUState::kFlagZero;
    }
    if (args.nzcv & 0x2) {
      flags_val |= CPUState::kFlagCarry;
    }
    if (args.nzcv & 0x1) {
      flags_val |= CPUState::kFlagOverflow;
    }
    Register imm = GetImm(flags_val);
    builder_.Gen<x86_64::MovwOpReg>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}, imm);
  };

  auto emit_compare_path = [&]() {
    FpRegister xmm_n = GetVRegScalar(args.rn, is_double);
    FpRegister xmm_m = GetVRegScalar(args.rm, is_double);
    Register flags = is_double
                         ? std::get<0>(Gen<x86_64::UcomisdXRegXReg>(xmm_n.machine_reg(),
                                                                    xmm_m.machine_reg()))
                         : std::get<0>(Gen<x86_64::UcomissXRegXReg>(xmm_n.machine_reg(),
                                                                    xmm_m.machine_reg()));
    EmitStoreArmFpNZCV(flags);
  };

  // AL/NV: always the compare path (no predicate branch). The compare path's
  // NZCV sub-tree leaves the builder at its merge; translation continues there.
  if (args.cond == Decoder::Condition::kAl || args.cond == Decoder::Condition::kNv) {
    emit_compare_path();
    return;
  }

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* cmp_bb = ir->NewBasicBlock();
  MachineBasicBlock* imm_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, cmp_bb);
  ir->AddEdge(cur_bb, imm_bb);

  // Condition is read from the pre-compare cpu.flags (EmitCondBranch loads them
  // in cur_bb, before the compare path overwrites them).
  EmitCondBranch(args.cond, cmp_bb, imm_bb);

  builder_.StartBasicBlock(cmp_bb);
  emit_compare_path();
  // emit_compare_path built an NZCV sub-tree; connect its tail to merge.
  MachineBasicBlock* cmp_tail = builder_.bb();
  ir->AddEdge(cmp_tail, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(imm_bb);
  emit_immediate_path();
  ir->AddEdge(imm_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
}

// SCVTF (op 010) / UCVTF (op 011), unscaled (rmode == 00): convert a general
// register integer to scalar FP via x86 CVTSI2SS/SD, which reads a SIGNED
// source. Mirrors lite_translator.h::FpIntConversion's SCVTF/UCVTF path.
//   * SCVTF: the L-form sign-extends the 32-bit source (sf=0), the Q-form takes
//     the 64-bit source (sf=1) — both match ARM's signed convert directly.
//   * UCVTF sf=0 (32-bit unsigned): zero-extend with a 32-bit MOV (auto-clears
//     the upper 32) then Q-convert (value <= UINT32_MAX < INT64_MAX, exact).
//   * UCVTF sf=1 (64-bit unsigned): values < 2^63 Q-convert directly; values
//     >= 2^63 branch to the round-to-odd halve/convert/double fix-up so the
//     round-to-nearest-even result matches static_cast<float|double>(uint64_t)
//     bit-for-bit. Two paths write a shared merge XMM.
// The scalar result is committed with SetVRegScalar (upper V[] bytes zeroed).
void HeavyOptimizerFrontend::EmitScvtfUcvtf(const Decoder::FpIntConvArgs& args,
                                            bool is_double,
                                            uint8_t fbits,
                                            bool src_from_simd) {
  if (!success()) {
    return;
  }
  const bool is_unsigned = (args.op == 0b011);

  // Fixed-point SCVTF/UCVTF (fbits != 0): scale the FP result by 2^-fbits before
  // storing to V[rd]. Exact power-of-2 multiply; fbits == 0 is a no-op (plain
  // integer form). Applied on every result path via this closure.
  auto finish = [&](FpRegister xmm) {
    if (fbits != 0) {
      FpRegister xscale = AllocTempSimdReg();
      if (is_double) {
        const uint64_t scale_bits = static_cast<uint64_t>(1023u - fbits) << 52;
        builder_.Gen<x86_64::MovqXRegReg>(xscale.machine_reg(), GetImm(scale_bits));
        builder_.Gen<x86_64::MulsdXRegXReg>(xmm.machine_reg(), xscale.machine_reg());
      } else {
        const uint32_t scale_bits = static_cast<uint32_t>(127u - fbits) << 23;
        builder_.Gen<x86_64::MovdXRegReg>(xscale.machine_reg(), GetImm(uint64_t{scale_bits}));
        builder_.Gen<x86_64::MulssXRegXReg>(xmm.machine_reg(), xscale.machine_reg());
      }
    }
    SetVRegScalar(args.rd, xmm, is_double);
  };

  // rn == 31 is WZR/XZR -> 0 for the GP-source form; static_cast<FP>(0) == +0.0
  // (scaled 0 is still 0). For the scalar-SIMD source form V31 is a real
  // register, so skip this special case.
  if (!src_from_simd && args.rn == 31) {
    finish(AllocZeroedSimdReg());
    return;
  }

  // Source integer: X[rn] (GP form) or the low 64-bit lane of V[rn] (scalar-SIMD
  // `.d` form; MOVQ pulls the int64 into a GP so the same CVTSI2SD ladder runs).
  Register src;
  if (src_from_simd) {
    FpRegister xn = GetVRegScalar(args.rn, is_double);
    src = std::get<0>(Gen<x86_64::MovqRegXReg>(xn.machine_reg()));
  } else {
    src = GetReg(args.rn);
  }

  // SCVTF (any sf): straight-line signed convert.
  if (!is_unsigned) {
    FpRegister xmm = AllocTempSimdReg();
    if (args.sf) {
      if (is_double) {
        builder_.Gen<x86_64::Cvtsi2sdqXRegReg>(xmm.machine_reg(), src);
      } else {
        builder_.Gen<x86_64::Cvtsi2ssqXRegReg>(xmm.machine_reg(), src);
      }
    } else {
      if (is_double) {
        builder_.Gen<x86_64::Cvtsi2sdlXRegReg>(xmm.machine_reg(), src);
      } else {
        builder_.Gen<x86_64::Cvtsi2sslXRegReg>(xmm.machine_reg(), src);
      }
    }
    finish(xmm);
    return;
  }

  // UCVTF sf=0: zero-extend the 32-bit source, convert as signed 64-bit.
  if (!args.sf) {
    Register zx = std::get<0>(Gen<x86_64::MovlRegReg>(src));
    FpRegister xmm = AllocTempSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::Cvtsi2sdqXRegReg>(xmm.machine_reg(), zx);
    } else {
      builder_.Gen<x86_64::Cvtsi2ssqXRegReg>(xmm.machine_reg(), zx);
    }
    finish(xmm);
    return;
  }

  // UCVTF sf=1: 64-bit unsigned. Branch on the sign bit (bit 63). Both paths
  // write the shared merge XMM `result`, defined on every edge into merge_bb.
  FpRegister result = AllocTempSimdReg();

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* direct_bb = ir->NewBasicBlock();
  MachineBasicBlock* fixup_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, direct_bb);
  ir->AddEdge(cur_bb, fixup_bb);

  // TEST sets SF = bit 63; take the fix-up path when the source is >= 2^63.
  Register flags = std::get<0>(Gen<x86_64::TestqRegReg>(src, src));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNegative, fixup_bb, direct_bb, flags);

  // Direct: value < 2^63, signed Q-convert is exact.
  builder_.StartBasicBlock(direct_bb);
  {
    FpRegister xd = AllocTempSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::Cvtsi2sdqXRegReg>(xd.machine_reg(), src);
    } else {
      builder_.Gen<x86_64::Cvtsi2ssqXRegReg>(xd.machine_reg(), src);
    }
    builder_.Gen<x86_64::MovdqaXRegXReg>(result.machine_reg(), xd.machine_reg());
  }
  ir->AddEdge(direct_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  // Fix-up: value >= 2^63. odd = (src >> 1) | (src & 1); convert; double.
  builder_.StartBasicBlock(fixup_bb);
  {
    Register low_bit = std::get<0>(Gen<x86_64::AndqRegImm>(Copy(src), int32_t{1}));
    Register halved = std::get<0>(Gen<x86_64::ShrqRegImm>(Copy(src), int8_t{1}));
    Register odd = std::get<0>(Gen<x86_64::OrqRegReg>(halved, low_bit));
    FpRegister xf = AllocTempSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::Cvtsi2sdqXRegReg>(xf.machine_reg(), odd);
      builder_.Gen<x86_64::AddsdXRegXReg>(xf.machine_reg(), xf.machine_reg());
    } else {
      builder_.Gen<x86_64::Cvtsi2ssqXRegReg>(xf.machine_reg(), odd);
      builder_.Gen<x86_64::AddssXRegXReg>(xf.machine_reg(), xf.machine_reg());
    }
    builder_.Gen<x86_64::MovdqaXRegXReg>(result.machine_reg(), xf.machine_reg());
  }
  ir->AddEdge(fixup_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
  finish(result);
}

// FCVTZS (op 000) / FCVTZU (op 001), truncating (rmode == 11): scalar FP -> GP
// integer via x86 CVTT{SS,SD}2SI, which returns the destination type's INT_MIN
// ("indefinite") for NaN / Inf / overflow. ARM's by-sign saturation is rebuilt
// with a BB-split fix-up ladder, bit-for-bit with lite_translator.h's FCVTZS /
// FCVTZU paths. A single `result` GP vreg is merged across the fix-up branches
// via PseudoCopy (mirrors ConditionalSelect); FLAGS never cross a basic block —
// each UCOMI/TEST is consumed by the PseudoCondBranch in its own block.
void HeavyOptimizerFrontend::EmitFcvtz(const Decoder::FpIntConvArgs& args,
                                       bool is_double,
                                       int8_t round_imm,
                                       bool ties_away,
                                       uint8_t fbits,
                                       bool dest_to_simd) {
  if (!success()) {
    return;
  }
  // Commit the integer result: X[rd] (GP form, rd==31 discards) or the low
  // 64-bit lane of V[rd] with Vd[127:64] zeroed (scalar-SIMD `.d` form).
  auto commit = [&](Register result) {
    if (dest_to_simd) {
      SetVRegScalarFromGp(args.rd, result, is_double);
    } else if (args.rd != 31) {
      SetReg(args.rd, result);
    }
  };
  // Unsigned forms: FCVTZU/FCVTNU/FCVTPU/FCVTMU (op 001) and FCVTAU (op 101).
  const bool is_unsigned = (args.op == 0b001 || args.op == 0b101);
  auto* ir = builder_.ir();

  FpRegister xmm = GetVRegScalar(args.rn, is_double);

  // Fixed-point FCVTZS/FCVTZU (fbits != 0): pre-multiply the FP source by
  // 2^+fbits (exact power-of-2 scale) in a private temp, leaving the guest v[]
  // slot untouched. The truncating cvtt + sign/NaN saturation ladder below then
  // operates on the scaled value, matching lite. Fixed-point never combines with
  // the rounding/ties-away paths (round_imm < 0, ties_away false), so this runs
  // first and independently.
  if (fbits != 0) {
    FpRegister scaled = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(scaled.machine_reg(), xmm.machine_reg());
    FpRegister xscale = AllocTempSimdReg();
    if (is_double) {
      const uint64_t scale_bits = static_cast<uint64_t>(1023u + fbits) << 52;
      builder_.Gen<x86_64::MovqXRegReg>(xscale.machine_reg(), GetImm(scale_bits));
      builder_.Gen<x86_64::MulsdXRegXReg>(scaled.machine_reg(), xscale.machine_reg());
    } else {
      const uint32_t scale_bits = static_cast<uint32_t>(127u + fbits) << 23;
      builder_.Gen<x86_64::MovdXRegReg>(xscale.machine_reg(), GetImm(uint64_t{scale_bits}));
      builder_.Gen<x86_64::MulssXRegXReg>(scaled.machine_reg(), xscale.machine_reg());
    }
    xmm = scaled;
  }

  // FCVTAS/FCVTAU (ties-away): add copysign(0.5, x), gated to 0 when |x| >= 2^23
  // (already an integer, where a 0.5 addend would round the wrong way), then let
  // the truncating cvtt + saturation ladder below finish. FP32 only (the caller
  // bails D). NaN/±Inf pass through: |NaN|,|Inf| exceed 2^23, so their addend is
  // gated to 0 and the downstream NaN/overflow branches still fire. Mirrors the
  // vector kFcvtasV/kFcvtauV path and the lite FRINTA trick. Round into a private
  // temp so the guest v[] slot is untouched.
  if (ties_away) {
    FpRegister sign = AllocZeroedSimdReg();
    FpRegister absx = AllocZeroedSimdReg();
    FpRegister half = AllocTempSimdReg();
    FpRegister thresh = AllocTempSimdReg();
    // sign = x & 0x80000000, addend = sign | 0.5f = copysign(0.5, x).
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
    builder_.Gen<x86_64::PslldXRegImm>(sign.machine_reg(), int8_t{31});  // 0x80000000
    builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), xmm.machine_reg());
    builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), GetImm(uint64_t{0x3F000000}));  // 0.5f
    builder_.Gen<x86_64::PorXRegXReg>(sign.machine_reg(), half.machine_reg());
    // |x| = x & 0x7FFFFFFF; gate the addend to 0 where |x| >= 2^23.
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(absx.machine_reg(), absx.machine_reg());
    builder_.Gen<x86_64::PsrldXRegImm>(absx.machine_reg(), int8_t{1});  // 0x7FFFFFFF
    builder_.Gen<x86_64::PandXRegXReg>(absx.machine_reg(), xmm.machine_reg());
    builder_.Gen<x86_64::MovdXRegReg>(thresh.machine_reg(), GetImm(uint64_t{0x4B000000}));  // 2^23
    builder_.Gen<x86_64::PcmpgtdXRegXReg>(thresh.machine_reg(), absx.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), thresh.machine_reg());
    FpRegister rounded = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(rounded.machine_reg(), xmm.machine_reg());
    builder_.Gen<x86_64::AddpsXRegXReg>(rounded.machine_reg(), sign.machine_reg());
    xmm = rounded;
  }

  // Rounding FP->int conversions (FCVTNS/NU/PS/PU/MS/MU) prepend an x86 ROUND
  // that makes the finite input an integer-valued FP (NaN/±Inf/sign-of-zero
  // pass through unchanged); the truncating cvtt + saturation ladder below then
  // produces the correctly-rounded ARM result. FCVTZS/FCVTZU pass round_imm < 0
  // (no rounding — cvtt already truncates). Round in a private temp so the guest
  // v[] slot is untouched. FP32 uses ROUNDPS (only lane 0 is consumed by cvtt);
  // FP64 uses scalar ROUNDSD.
  if (round_imm >= 0) {
    FpRegister rounded = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(rounded.machine_reg(), xmm.machine_reg());
    if (is_double) {
      builder_.Gen<x86_64::RoundsdXRegXRegImm>(
          rounded.machine_reg(), rounded.machine_reg(), round_imm);
    } else {
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(
          rounded.machine_reg(), rounded.machine_reg(), round_imm);
    }
    xmm = rounded;
  }

  // xmm self-compare: PF=1 iff NaN.
  auto ucomi_self = [&]() -> Register {
    return is_double
               ? std::get<0>(Gen<x86_64::UcomisdXRegXReg>(xmm.machine_reg(), xmm.machine_reg()))
               : std::get<0>(Gen<x86_64::UcomissXRegXReg>(xmm.machine_reg(), xmm.machine_reg()));
  };
  // xmm raw bits -> GP, then TEST self: SF = FP sign bit (bit 31 / bit 63).
  auto fp_sign_flags = [&]() -> Register {
    if (is_double) {
      Register raw = std::get<0>(Gen<x86_64::MovqRegXReg>(xmm.machine_reg()));
      return std::get<0>(Gen<x86_64::TestqRegReg>(raw, raw));
    }
    Register raw = std::get<0>(Gen<x86_64::MovdRegXReg>(xmm.machine_reg()));
    return std::get<0>(Gen<x86_64::TestlRegReg>(raw, raw));
  };
  // Truncating convert to a signed 64-bit GP (Q-form).
  auto cvtt_q = [&](FpRegister x) -> Register {
    return is_double ? std::get<0>(Gen<x86_64::Cvttsd2siqRegXReg>(x.machine_reg()))
                     : std::get<0>(Gen<x86_64::Cvttss2siqRegXReg>(x.machine_reg()));
  };
  // Materialize an FP constant (given its raw bits) into a fresh XMM.
  auto fp_const = [&](uint64_t bits) -> FpRegister {
    FpRegister c = AllocTempSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::MovqXRegReg>(c.machine_reg(), GetImm(bits));
    } else {
      builder_.Gen<x86_64::MovdXRegReg>(c.machine_reg(), GetImm(bits & 0xFFFFFFFF));
    }
    return c;
  };

  Register result = AllocTempReg();

  if (!is_unsigned) {
    // FCVTZS: NaN -> 0; positive overflow -> INT_MAX; negative overflow ->
    // INT_MIN (already the cvtt indefinite value). Default keeps the cvtt tmp.
    Register tmp;
    if (args.sf) {
      tmp = cvtt_q(xmm);
    } else {
      tmp = is_double ? std::get<0>(Gen<x86_64::Cvttsd2silRegXReg>(xmm.machine_reg()))
                      : std::get<0>(Gen<x86_64::Cvttss2silRegXReg>(xmm.machine_reg()));
    }
    builder_.Gen<PseudoCopy>(result, tmp, 8);  // default: keep tmp

    auto* cur_bb = builder_.bb();
    MachineBasicBlock* nan_bb = ir->NewBasicBlock();
    MachineBasicBlock* notnan_bb = ir->NewBasicBlock();
    MachineBasicBlock* pos_bb = ir->NewBasicBlock();
    MachineBasicBlock* ovf_bb = ir->NewBasicBlock();
    MachineBasicBlock* done_bb = ir->NewBasicBlock();

    ir->AddEdge(cur_bb, nan_bb);
    ir->AddEdge(cur_bb, notnan_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kParityEven, nan_bb, notnan_bb,
                                   ucomi_self());

    builder_.StartBasicBlock(nan_bb);
    builder_.Gen<PseudoCopy>(result, GetImm(0), 8);
    ir->AddEdge(nan_bb, done_bb);
    builder_.Gen<PseudoBranch>(done_bb);

    // Non-NaN: FP < 0 keeps tmp (in-range neg or INT_MIN indefinite); FP >= 0
    // falls to the positive-overflow test.
    builder_.StartBasicBlock(notnan_bb);
    ir->AddEdge(notnan_bb, done_bb);
    ir->AddEdge(notnan_bb, pos_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kNegative, done_bb, pos_bb,
                                   fp_sign_flags());

    // FP >= 0: tmp >= 0 keeps it (in-range positive); tmp < 0 == INT_MIN means
    // positive overflow -> INT_MAX.
    builder_.StartBasicBlock(pos_bb);
    Register tmp_flags = args.sf ? std::get<0>(Gen<x86_64::TestqRegReg>(tmp, tmp))
                                 : std::get<0>(Gen<x86_64::TestlRegReg>(tmp, tmp));
    ir->AddEdge(pos_bb, done_bb);
    ir->AddEdge(pos_bb, ovf_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kPositiveOrZero, done_bb, ovf_bb,
                                   tmp_flags);

    builder_.StartBasicBlock(ovf_bb);
    const uint64_t int_max = args.sf ? static_cast<uint64_t>(INT64_MAX) : uint64_t{0x7FFFFFFF};
    builder_.Gen<PseudoCopy>(result, GetImm(int_max), 8);
    ir->AddEdge(ovf_bb, done_bb);
    builder_.Gen<PseudoBranch>(done_bb);

    builder_.StartBasicBlock(done_bb);
    commit(result);
    return;
  }

  // FCVTZU: NaN / (FP < 0, incl -0) -> 0; FP > UINT*_MAX -> UINT*_MAX.
  MachineBasicBlock* zero_bb = ir->NewBasicBlock();
  MachineBasicBlock* notnan_bb = ir->NewBasicBlock();
  MachineBasicBlock* done_bb = ir->NewBasicBlock();

  auto* cur_bb = builder_.bb();
  ir->AddEdge(cur_bb, zero_bb);
  ir->AddEdge(cur_bb, notnan_bb);
  builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kParityEven, zero_bb, notnan_bb,
                                 ucomi_self());

  if (!args.sf) {
    // sf=0 (uint32): Q-form cvtt always fits int64. Upper-32 zero -> in-range
    // (low 32 are the answer); upper-32 non-zero -> saturate to UINT32_MAX.
    MachineBasicBlock* pos_bb = ir->NewBasicBlock();
    MachineBasicBlock* ovf_bb = ir->NewBasicBlock();

    builder_.StartBasicBlock(notnan_bb);
    ir->AddEdge(notnan_bb, zero_bb);
    ir->AddEdge(notnan_bb, pos_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kNegative, zero_bb, pos_bb,
                                   fp_sign_flags());

    builder_.StartBasicBlock(pos_bb);
    Register tmp = cvtt_q(xmm);
    builder_.Gen<PseudoCopy>(result, tmp, 8);  // default in-range
    Register hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(tmp), int8_t{32}));
    Register hi_flags = std::get<0>(Gen<x86_64::TestqRegReg>(hi, hi));
    ir->AddEdge(pos_bb, done_bb);
    ir->AddEdge(pos_bb, ovf_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kZero, done_bb, ovf_bb, hi_flags);

    builder_.StartBasicBlock(ovf_bb);
    builder_.Gen<PseudoCopy>(result, GetImm(uint64_t{0xFFFFFFFF}), 8);
    ir->AddEdge(ovf_bb, done_bb);
    builder_.Gen<PseudoBranch>(done_bb);
  } else {
    // sf=1 (uint64): cvtt-Q covers [0, 2^63) directly. FP >= 2^63 uses the
    // offset trick (subtract 2^63, cvtt, set bit 63); FP >= 2^64 saturates.
    const uint64_t bits_2p63 = is_double ? 0x43E0000000000000ULL : 0x5F000000ULL;
    const uint64_t bits_2p64 = is_double ? 0x43F0000000000000ULL : 0x5F800000ULL;

    MachineBasicBlock* cmp_bb = ir->NewBasicBlock();
    MachineBasicBlock* direct_bb = ir->NewBasicBlock();
    MachineBasicBlock* ge63_bb = ir->NewBasicBlock();
    MachineBasicBlock* satmax_bb = ir->NewBasicBlock();
    MachineBasicBlock* inrange_bb = ir->NewBasicBlock();

    builder_.StartBasicBlock(notnan_bb);
    ir->AddEdge(notnan_bb, zero_bb);
    ir->AddEdge(notnan_bb, cmp_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kNegative, zero_bb, cmp_bb,
                                   fp_sign_flags());

    // FP >= 0: FP < 2^63 -> direct cvtt-Q; else check the upper bound.
    builder_.StartBasicBlock(cmp_bb);
    FpRegister bound63 = fp_const(bits_2p63);
    Register lt_flags = is_double
                            ? std::get<0>(Gen<x86_64::UcomisdXRegXReg>(xmm.machine_reg(),
                                                                       bound63.machine_reg()))
                            : std::get<0>(Gen<x86_64::UcomissXRegXReg>(xmm.machine_reg(),
                                                                       bound63.machine_reg()));
    ir->AddEdge(cmp_bb, direct_bb);
    ir->AddEdge(cmp_bb, ge63_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kBelow, direct_bb, ge63_bb,
                                   lt_flags);

    builder_.StartBasicBlock(direct_bb);
    builder_.Gen<PseudoCopy>(result, cvtt_q(xmm), 8);
    ir->AddEdge(direct_bb, done_bb);
    builder_.Gen<PseudoBranch>(done_bb);

    // FP >= 2^63: FP >= 2^64 (or +Inf) saturates; else the offset trick.
    builder_.StartBasicBlock(ge63_bb);
    FpRegister bound64 = fp_const(bits_2p64);
    Register ge_flags = is_double
                            ? std::get<0>(Gen<x86_64::UcomisdXRegXReg>(xmm.machine_reg(),
                                                                       bound64.machine_reg()))
                            : std::get<0>(Gen<x86_64::UcomissXRegXReg>(xmm.machine_reg(),
                                                                       bound64.machine_reg()));
    ir->AddEdge(ge63_bb, satmax_bb);
    ir->AddEdge(ge63_bb, inrange_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kAboveEqual, satmax_bb, inrange_bb,
                                   ge_flags);

    builder_.StartBasicBlock(satmax_bb);
    builder_.Gen<PseudoCopy>(result, GetImm(~uint64_t{0}), 8);  // UINT64_MAX
    ir->AddEdge(satmax_bb, done_bb);
    builder_.Gen<PseudoBranch>(done_bb);

    // FP in [2^63, 2^64): subtract 2^63 (exact), cvtt to int64, set bit 63.
    // Subss/Subsd is use_def; copy the source into a temp first.
    builder_.StartBasicBlock(inrange_bb);
    FpRegister bound63b = fp_const(bits_2p63);
    FpRegister xsub = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(xsub.machine_reg(), xmm.machine_reg());
    if (is_double) {
      builder_.Gen<x86_64::SubsdXRegXReg>(xsub.machine_reg(), bound63b.machine_reg());
    } else {
      builder_.Gen<x86_64::SubssXRegXReg>(xsub.machine_reg(), bound63b.machine_reg());
    }
    Register tmp = cvtt_q(xsub);
    tmp = std::get<0>(Gen<x86_64::BtsqRegImm, kNoSSA>(tmp, int8_t{63}));
    builder_.Gen<PseudoCopy>(result, tmp, 8);
    ir->AddEdge(inrange_bb, done_bb);
    builder_.Gen<PseudoBranch>(done_bb);
  }

  builder_.StartBasicBlock(zero_bb);
  builder_.Gen<PseudoCopy>(result, GetImm(0), 8);
  ir->AddEdge(zero_bb, done_bb);
  builder_.Gen<PseudoBranch>(done_bb);

  builder_.StartBasicBlock(done_bb);
  commit(result);
}

// LDXR/STXR/LDAXR/STLXR (exclusive) and LDAR/STLR (acquire/release). Mirrors
// lite_translator.h::LoadStoreExclusive byte-for-byte:
//   * base is TBI-masked first (the top-byte-ignore tag is not part of the host
//     address), then used for the reservation address, the load/store, and the
//     CMPXCHG memory operand, so the heavy and lite/interp monitors agree on the
//     same guest under tagged pointers.
//   * LDAR/STLR: x86-TSO gives acquire/release for plain loads/stores, so they
//     are a plain sized Load()/Store() (both emit fault recovery internally).
//   * LDXR/LDAXR: Load() the value, then record cpu.reservation_address = base
//     and cpu.reservation_value = value (64-bit slot, matching the lite tier and
//     interpreter; only LDXP/STXP use the full 128-bit width).
//   * STXR/STLXR: if cpu.reservation_address still equals base, do a sized
//     LOCK CMPXCHG of the saved reservation_value (expected, in the accumulator)
//     against [base] with Rt as the new value; clear cpu.reservation_address;
//     Rs gets 0 on CMPXCHG success (ZF=1), 1 on CMPXCHG failure or address
//     mismatch. XZR (reg 31) reads as 0 / discards writes.
// This deliberately reuses the lite tier's reservation_address + reservation_value
// + plain CMPXCHG monitor model (NOT the riscv64 heavy tier's
// MemoryRegionReservation owner-tracking model), so the heavy, lite, and
// interpreter tiers share one monitor scheme for the same guest.
void HeavyOptimizerFrontend::LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& args,
                                                Register base) {
  if (!success()) {
    return;
  }

  // Apply the TBI mask before using base as a memory operand or reservation key.
  base = ApplyTbi(base);
  auto lss = static_cast<Decoder::LoadStoreSize>(args.size);
  const int32_t resv_addr_off = static_cast<int32_t>(offsetof(ThreadState, cpu.reservation_address));
  const int32_t resv_val_off = static_cast<int32_t>(offsetof(ThreadState, cpu.reservation_value));

  switch (args.op) {
    case Decoder::AtomicOp::kLdar: {
      // Load-acquire: x86-TSO provides acquire ordering for all loads.
      Register res = Load(lss, /*is_signed=*/false, /*is_64bit_target=*/true, base, 0);
      if (!success()) {
        return;
      }
      if (args.rt != 31) {
        SetReg(args.rt, res);
      }
      return;
    }

    case Decoder::AtomicOp::kStlr: {
      // Store-release: x86-TSO gives release ordering for free, but ARM STLR is
      // sequentially consistent (RCsc) and also orders the store before later
      // loads. x86 permits StoreLoad reordering, so emit MFENCE after the store.
      Register data = (args.rt != 31) ? GetReg(args.rt) : GetImm(0);
      if (!success()) {
        return;
      }
      Store(lss, base, 0, data);
      builder_.Gen<x86_64::Mfence>();
      return;
    }

    case Decoder::AtomicOp::kLdxr: {
      // Load-exclusive: load the value, then record the reservation.
      Register res = Load(lss, /*is_signed=*/false, /*is_64bit_target=*/true, base, 0);
      if (!success()) {
        return;
      }
      // reservation_address = base.
      builder_.Gen<x86_64::MovqOpReg>(
          {.base = x86_64::kMachineRegRBP, .disp = resv_addr_off}, base);
      // reservation_value = res (low 64 bits of the 128-bit slot, as the lite
      // tier does; the single-register forms only ever use 64 bits).
      builder_.Gen<x86_64::MovqOpReg>(
          {.base = x86_64::kMachineRegRBP, .disp = resv_val_off}, res);
      if (args.rt != 31) {
        SetReg(args.rt, res);
      }
      return;
    }

    case Decoder::AtomicOp::kStxr: {
      // Store-exclusive: compare-and-swap against the reservation. Structured to
      // mirror the riscv64 heavy frontend's MemoryRegionReservationExchange: the
      // status vreg (result) is written only in the two terminal predecessors of
      // the merge block (XOR -> 0 on success, MOVQ #1 on failure), never in the
      // entry block, so its live range is the simple two-def/one-use shape the
      // lifetime analyzer expects.
      Register new_val = (args.rt != 31) ? GetReg(args.rt) : GetImm(0);
      Register resv_addr =
          std::get<0>(Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = resv_addr_off}));
      if (!success()) {
        return;
      }

      // Clear the reservation (STXR always clears, success or not).
      builder_.GenPutImm(resv_addr_off, 0);

      Register status = AllocTempReg();
      auto* ir = builder_.ir();
      auto* cur_bb = builder_.bb();
      MachineBasicBlock* match_bb = ir->NewBasicBlock();   // reservation addr matches base
      MachineBasicBlock* fail_bb = ir->NewBasicBlock();    // addr mismatch or CMPXCHG fail
      MachineBasicBlock* swap_ok_bb = ir->NewBasicBlock();  // CMPXCHG succeeded
      MachineBasicBlock* merge_bb = ir->NewBasicBlock();
      ir->AddEdge(cur_bb, match_bb);
      ir->AddEdge(cur_bb, fail_bb);

      // if (reservation_address != base) goto fail_bb (status = 1).
      builder_.Gen<PseudoCondBranch>(
          x86_64::Assembler::Condition::kNotEqual,
          fail_bb,
          match_bb,
          std::get<0>(Gen<x86_64::CmpqRegReg>(resv_addr, base)));

      // --- match path: sized LOCK CMPXCHG(expected, [base], new_val). ---
      builder_.StartBasicBlock(match_bb);
      // Load the expected value (saved reservation) into the CMPXCHG accumulator.
      Register expected =
          std::get<0>(Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = resv_val_off}));
      Register host_flags;
      switch (args.size) {
        case 0:
          std::tie(expected, host_flags) =
              Gen<x86_64::LockCmpXchgbRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 1:
          std::tie(expected, host_flags) =
              Gen<x86_64::LockCmpXchgwRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 2:
          std::tie(expected, host_flags) =
              Gen<x86_64::LockCmpXchglRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 3:
          std::tie(expected, host_flags) =
              Gen<x86_64::LockCmpXchgqRegOpReg>(expected, {.base = base}, new_val);
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      // No fault-recovery split here (unlike Load()/Store()): the CMPXCHG runs
      // only when reservation_address == base, i.e. a prior LDXR already faulted
      // in this page, so a fault is pathological. Splitting the block would also
      // strand the CMPXCHG's FLAGS output across the recovery edge. This mirrors
      // the riscv64 heavy frontend's MemoryRegionReservationExchange, which also
      // omits recovery around the swap CMPXCHG.

      // ZF=0 (kNotZero) means CMPXCHG failed -> fail_bb (status = 1); ZF=1 ->
      // swap_ok_bb (status = 0).
      ir->AddEdge(builder_.bb(), fail_bb);
      ir->AddEdge(builder_.bb(), swap_ok_bb);
      builder_.Gen<PseudoCondBranch>(
          x86_64::Assembler::Condition::kNotZero, fail_bb, swap_ok_bb, host_flags);

      // --- success: status = 0 (XOR self, with a PseudoDef to seat the value). ---
      builder_.StartBasicBlock(swap_ok_bb);
      builder_.Gen<PseudoDefReg>(status);
      builder_.Gen<x86_64::XorqRegReg>(status, status, GetFlagsRegister());
      ir->AddEdge(swap_ok_bb, merge_bb);
      builder_.Gen<PseudoBranch>(merge_bb);

      // --- failure (addr mismatch or CMPXCHG miss): status = 1. ---
      builder_.StartBasicBlock(fail_bb);
      builder_.Gen<x86_64::MovqRegImm>(status, int64_t{1});
      ir->AddEdge(fail_bb, merge_bb);
      builder_.Gen<PseudoBranch>(merge_bb);

      builder_.StartBasicBlock(merge_bb);
      if (args.rs != 31) {
        SetReg(args.rs, status);
      }
      return;
    }

    case Decoder::AtomicOp::kCas: {
      // CAS Xs, Xt, [Xn]: compare [Xn] with Xs; on a match store Xt to [Xn];
      // the old value of [Xn] is written back to Xs. Mirrors lite_translator.h's
      // kCas: x86 LOCK CMPXCHG compares RAX with [mem], stores the source on a
      // match, and leaves the old memory value in RAX. GenRecoveryBlockForLastInsn()
      // delivers a guest fault on a bad pointer. Plain CAS never branches, so the
      // CMPXCHG's FLAGS output is discarded (std::get<0> keeps only the RAX
      // result); nothing FLAGS-class is live across the recovery edge — unlike
      // STXR, where the FLAGS feed a branch and recovery is therefore omitted.
      Register expected = (args.rs != 31) ? GetReg(args.rs) : GetImm(0);
      Register desired = (args.rt != 31) ? GetReg(args.rt) : GetImm(0);
      switch (args.size) {
        case 0:
          expected =
              std::get<0>(Gen<x86_64::LockCmpXchgbRegOpReg>(expected, {.base = base}, desired));
          break;
        case 1:
          expected =
              std::get<0>(Gen<x86_64::LockCmpXchgwRegOpReg>(expected, {.base = base}, desired));
          break;
        case 2:
          expected =
              std::get<0>(Gen<x86_64::LockCmpXchglRegOpReg>(expected, {.base = base}, desired));
          break;
        case 3:
          expected =
              std::get<0>(Gen<x86_64::LockCmpXchgqRegOpReg>(expected, {.base = base}, desired));
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      GenRecoveryBlockForLastInsn();
      // The old value is now in RAX (expected). ARM CAS Ws zero-extends the old
      // value to 64 bits; on a CMPXCHG *match* the accumulator keeps the full
      // 64-bit operand's upper bits, so re-zero-extend the sub-64 forms (mirrors
      // lite's byte/halfword AND masks and 32-bit MOVL).
      if (args.rs != 31) {
        if (args.size == 0) {
          expected =
              std::get<0>(Gen<x86_64::AndqRegImm>(expected, static_cast<int32_t>(0xFF)));
        } else if (args.size == 1) {
          expected =
              std::get<0>(Gen<x86_64::AndqRegImm>(expected, static_cast<int32_t>(0xFFFF)));
        } else if (args.size == 2) {
          expected = std::get<0>(Gen<x86_64::MovlRegReg>(expected));
        }
        SetReg(args.rs, expected);
      }
      return;
    }

    case Decoder::AtomicOp::kSwp: {
      // SWP Xs, Xt, [Xn]: atomically swap [Xn] with Xs; the old value goes to
      // Xt. Mirrors lite_translator.h's kSwp: x86 XCHG with a memory operand is
      // implicitly LOCKed. XCHG has no FLAGS operand, so the recovery split is
      // exactly as safe as a plain Store's.
      Register new_val = (args.rs != 31) ? GetReg(args.rs) : GetImm(0);
      switch (args.size) {
        case 0:
          new_val = std::get<0>(Gen<x86_64::XchgbRegOp>(new_val, {.base = base}));
          break;
        case 1:
          new_val = std::get<0>(Gen<x86_64::XchgwRegOp>(new_val, {.base = base}));
          break;
        case 2:
          new_val = std::get<0>(Gen<x86_64::XchglRegOp>(new_val, {.base = base}));
          break;
        case 3:
          new_val = std::get<0>(Gen<x86_64::XchgqRegOp>(new_val, {.base = base}));
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      GenRecoveryBlockForLastInsn();
      // byte/halfword XCHG leaves the upper bits of new_val as the original guest
      // Xs; ARM SWP Wt zero-extends the old memory value to 64. The 32-bit XCHG
      // already zero-extends (a 32-bit reg write clears bits 63:32), matching
      // lite, which only masks the byte/halfword forms.
      if (args.size == 0) {
        new_val = std::get<0>(Gen<x86_64::AndqRegImm>(new_val, static_cast<int32_t>(0xFF)));
      } else if (args.size == 1) {
        new_val = std::get<0>(Gen<x86_64::AndqRegImm>(new_val, static_cast<int32_t>(0xFFFF)));
      }
      if (args.rt != 31) {
        SetReg(args.rt, new_val);
      }
      return;
    }

    case Decoder::AtomicOp::kLdadd: {
      // LDADD Xs, Xt, [Xn]: atomically add Xs to [Xn]; the old value goes to Xt.
      // Mirrors lite_translator.h's kLdadd: x86 LOCK XADD adds the source to
      // [mem] and leaves the old memory value in the source register. ARM LDADD
      // does not set NZCV, so the XADD's FLAGS output is discarded (std::get<0>
      // keeps only the value); nothing FLAGS-class is live across the recovery
      // edge.
      Register addend = (args.rs != 31) ? GetReg(args.rs) : GetImm(0);
      // LOCK XADD models FLAGS as use_def (XADD writes all flags from the sum),
      // so the frontend Gen adapter takes an explicit FLAGS operand; the value
      // is ignored — Gen overrides it with GetFlagsRegister() — but must be
      // passed to satisfy the operand count.
      switch (args.size) {
        case 0:
          addend =
              std::get<0>(Gen<x86_64::LockXaddbOpReg>({.base = base}, addend, GetFlagsRegister()));
          break;
        case 1:
          addend =
              std::get<0>(Gen<x86_64::LockXaddwOpReg>({.base = base}, addend, GetFlagsRegister()));
          break;
        case 2:
          addend =
              std::get<0>(Gen<x86_64::LockXaddlOpReg>({.base = base}, addend, GetFlagsRegister()));
          break;
        case 3:
          addend =
              std::get<0>(Gen<x86_64::LockXaddqOpReg>({.base = base}, addend, GetFlagsRegister()));
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      GenRecoveryBlockForLastInsn();
      // byte/halfword XADD only updates the low bits of addend; ARM LDADD Wt
      // zero-extends the old memory value to 64. The 32-bit XADD already
      // zero-extends, so mirror lite and mask only the byte/halfword forms.
      if (args.size == 0) {
        addend = std::get<0>(Gen<x86_64::AndqRegImm>(addend, static_cast<int32_t>(0xFF)));
      } else if (args.size == 1) {
        addend = std::get<0>(Gen<x86_64::AndqRegImm>(addend, static_cast<int32_t>(0xFFFF)));
      }
      if (args.rt != 31) {
        SetReg(args.rt, addend);
      }
      return;
    }

    default:
      // Remaining LSE atomics (LDCLR/LDSET/LDEOR, LDSMAX/MIN, LDUMAX/MIN),
      // CASP, and the LDXP/STXP pair forms are not yet mirrored into the heavy
      // tier; bail to the lite translator (correct, just slower).
      UndefinedReturningVoid();
      return;
  }
}

void HeavyOptimizerFrontend::DataMemoryBarrier() {
  // Full DMB/DSB -> MFENCE: recover the StoreLoad ordering x86 TSO omits.
  builder_.Gen<x86_64::Mfence>();
}

void HeavyOptimizerFrontend::RecordBail(BailReason reason) {
  const GuestAddr pc = GetInsnAddr();
  // The frontend doesn't retain the 32-bit word; read it back from guest memory
  // so the trace line carries the encoding for offline per-opcode bucketing.
  const uint32_t insn = *ToHostAddr<const uint32_t>(pc);

  g_heavy_bail_stats.by_reason[static_cast<size_t>(reason)].fetch_add(1,
                                                                      std::memory_order_relaxed);
  const uint64_t total = g_heavy_bail_stats.total.fetch_add(1, std::memory_order_relaxed) + 1;

  TRACE("heavy-bail pc=0x%lx insn=0x%08x reason=%s", pc, insn, BailReasonName(reason));

  const size_t dump_every = GetHeavyBailDumpEvery();
  if (dump_every != 0 && (total % dump_every) == 0) {
    DumpHeavyBailStats();
  }
}

void HeavyOptimizerFrontend::Undefined(BailReason reason) {
  // Idempotent: a single guest instruction can trigger several listener calls
  // (e.g. a pre/post-index access calls AddImm then Load/Store). If more than one
  // bails, only the first may append the region-exit terminator — a second exit
  // in the same basic block would make it fail CheckMachineIR.
  if (!success_) {
    return;
  }
  success_ = false;

  // Bail-reason instrumentation, gated on the trace fd: a production build with
  // no berberis.tracing property pays only this branch.
  if (Tracing::IsOn()) {
    RecordBail(reason);
  }

  ExitGeneratedCode(GetInsnAddr());
  // We don't require the region to end here as control flow may jump around the
  // undefined instruction, so handle it as an unconditional branch.
  is_uncond_branch_ = true;
}

bool HeavyOptimizerFrontend::IsRegionEndReached() const {
  if (!is_uncond_branch_) {
    return false;
  }

  auto map_it = branch_targets_.find(GetInsnAddr());
  // If this instruction following an unconditional branch isn't reachable by
  // some other branch - it's a region end.
  return map_it == branch_targets_.end();
}

void HeavyOptimizerFrontend::ResolveJumps() {
  if (!config::kLinkJumpsWithinRegion) {
    return;
  }
  auto ir = builder_.ir();

  MachineBasicBlockList bb_list_copy(ir->bb_list());
  for (auto bb : bb_list_copy) {
    if (bb->is_recovery()) {
      // Recovery blocks must exit the region, do not try to resolve into a local branch.
      continue;
    }

    const MachineInsn* last_insn = bb->insn_list().back();
    if (last_insn->opcode() != kMachineOpPseudoJump) {
      continue;
    }

    auto* jump = static_cast<const PseudoJump*>(last_insn);
    if (jump->kind() == PseudoJump::Kind::kSyscall ||
        jump->kind() == PseudoJump::Kind::kExitGeneratedCode) {
      // Syscall or generated-code exit must always exit the region.
      continue;
    }

    GuestAddr target = jump->target();
    auto map_it = branch_targets_.find(target);
    // All PseudoJump insns must add their targets to branch_targets.
    CHECK(map_it != branch_targets_.end());

    MachineInsnPosition pos = map_it->second;
    MachineBasicBlock* target_containing_bb = pos.first;
    if (!target_containing_bb) {
      // Branch target is not in the current region.
      continue;
    }

    CHECK(pos.second.has_value());
    auto target_insn_it = pos.second.value();
    MachineBasicBlock* target_bb;
    if (target_insn_it == target_containing_bb->insn_list().begin()) {
      // We don't need to split if target_insn_it is at the beginning of target_containing_bb.
      target_bb = target_containing_bb;
    } else {
      // target_bb is split from target_containing_bb.
      target_bb = ir->SplitBasicBlock(target_containing_bb, target_insn_it);
      UpdateBranchTargetsAfterSplit(target, target_containing_bb, target_bb);

      // Make sure target_bb is also considered for jump resolution. Otherwise we
      // may leave code referenced by it unlinked from the rest of the IR.
      bb_list_copy.push_back(target_bb);

      // If bb is equal to target_containing_bb, then the branch instruction at
      // the end of bb is moved to the new target_bb, so we replace the
      // instruction at the end of target_bb instead of bb.
      if (bb == target_containing_bb) {
        bb = target_bb;
      }
    }

    ReplaceJumpWithBranch(bb, target_bb);
  }
}

void HeavyOptimizerFrontend::ReplaceJumpWithBranch(MachineBasicBlock* bb,
                                                   MachineBasicBlock* target_bb) {
  auto ir = builder_.ir();
  const auto* last_insn = bb->insn_list().back();
  CHECK_EQ(last_insn->opcode(), kMachineOpPseudoJump);
  auto* jump = static_cast<const PseudoJump*>(last_insn);
  GuestAddr target = static_cast<const PseudoJump*>(jump)->target();
  // Do not invalidate this iterator as it may be a target for another jump.
  // Instead overwrite the instruction.
  auto jump_it = std::prev(bb->insn_list().end());

  if (jump->kind() == PseudoJump::Kind::kJumpWithoutPendingSignalsCheck) {
    // Simple branch for forward jump.
    *jump_it = ir->NewInsn<PseudoBranch>(target_bb);
    ir->AddEdge(bb, target_bb);
  } else {
    CHECK(jump->kind() == PseudoJump::Kind::kJumpWithPendingSignalsCheck);
    // This is a backward branch resolved into an in-region target, i.e. an
    // in-region loop back-edge. Record that the region captured a hot loop.
    has_in_region_backedge_ = true;
    // See EmitCheckSignalsAndMaybeReturn.
    auto* exit_bb = ir->NewBasicBlock();
    // Note that we intentionally don't mark exit_bb as recovery and therefore
    // don't request its reordering away from hot code spots. target_bb is a
    // back branch and is unlikely to be a fall-through jump for the current bb.
    // At the same time exit_bb can be a fall-through jump and benchmarks benefit
    // from it.
    const size_t offset = offsetof(ThreadState, pending_signals_status);
    auto* cmpb = ir->NewInsn<x86_64::CmpbOpImm>({.base = x86_64::kMachineRegRBP, .disp = offset},
                                                kPendingSignalsPresent,
                                                GetFlagsRegister());
    *jump_it = cmpb;
    auto* cond_branch = ir->NewInsn<PseudoCondBranch>(
        x86_64::Assembler::Condition::kEqual, exit_bb, target_bb, GetFlagsRegister());
    bb->insn_list().push_back(cond_branch);

    builder_.StartBasicBlock(exit_bb);
    ExitGeneratedCode(target);

    ir->AddEdge(bb, exit_bb);
    ir->AddEdge(bb, target_bb);
  }
}

void HeavyOptimizerFrontend::UpdateBranchTargetsAfterSplit(GuestAddr addr,
                                                           const MachineBasicBlock* old_bb,
                                                           MachineBasicBlock* new_bb) {
  auto map_it = branch_targets_.find(addr);
  CHECK(map_it != branch_targets_.end());
  while (map_it != branch_targets_.end() && map_it->second.first == old_bb) {
    map_it->second.first = new_bb;
    map_it++;
  }
}

//
// Methods that are not part of the SemanticsListener implementation.
//
void HeavyOptimizerFrontend::StartInsn() {
  if (is_uncond_branch_) {
    auto* ir = builder_.ir();
    builder_.StartBasicBlock(ir->NewBasicBlock());
  }

  is_uncond_branch_ = false;
  // The iterators in branch_targets are the last iterators before generating an
  // insn. We advance iterators by one step in Finalize(), as we'll use it to
  // iterate the sub-list of instructions starting from the first one for the
  // given guest address.
  //
  // If a basic block is empty before generating insn, an empty optional typed
  // value is returned. We will resolve it to the first insn of the basic block
  // in Finalize().
  branch_targets_[GetInsnAddr()] = builder_.GetMachineInsnPosition();
}

void HeavyOptimizerFrontend::Finalize(GuestAddr stop_pc) {
  // Make sure the last basic block isn't empty before fixing iterators in
  // branch_targets.
  if (builder_.bb()->insn_list().empty() ||
      !builder_.ir()->IsControlTransfer(builder_.bb()->insn_list().back())) {
    GenJump(stop_pc);
  }

  // This loop advances the iterators in branch_targets by one. Because in
  // StartInsn(), we saved the iterator to the last insn before we generate the
  // first insn for each guest address. If an insn is saved as an empty optional,
  // then the basic block is empty before we generate the first insn for the
  // guest address. So we resolve it to the first insn in the basic block.
  for (auto& [unused_address, insn_pos] : branch_targets_) {
    auto& [bb, insn_it] = insn_pos;
    if (!bb) {
      // Branch target is not in the current region.
      continue;
    }

    if (insn_it.has_value()) {
      insn_it.value()++;
    } else {
      // Make sure bb isn't still empty.
      CHECK(!bb->insn_list().empty());
      insn_it = bb->insn_list().begin();
    }
  }

  ResolveJumps();
}

}  // namespace berberis
