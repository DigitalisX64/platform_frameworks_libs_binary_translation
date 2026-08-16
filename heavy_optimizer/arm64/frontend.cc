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

// URECPE / URSQRTE unsigned integer reciprocal / reciprocal-sqrt estimate tables.
// The result is a pure function of the 9-bit field (lane>>23)&0x1FF (the saturation
// condition — top bit clear for URECPE, top two bits clear for URSQRTE — is encoded
// in that field's high bits), so a 512-entry table per op lets the JIT do a per-lane
// lookup that bit-matches the interpreter's UnsignedRecipEstimate/RSqrtEstimate. This
// mirrors the identical table the lite translator builds; kept self-contained here so
// the heavy backend takes no cross-file dependency.
const uint32_t* BuildHeavyUnsignedEstimateTable(bool is_rsqrt) {
  uint32_t* t = new uint32_t[512];
  for (int idx = 0; idx < 512; ++idx) {
    uint32_t r;
    if (!is_rsqrt) {
      if (idx < 256) {  // top bit (bit31 of input) clear -> saturate
        r = 0xFFFFFFFFu;
      } else {
        int a2 = idx * 2 + 1;
        int b = (1 << 19) / a2;
        int estimate = (b + 1) / 2;
        r = static_cast<uint32_t>(estimate) << 23;
      }
    } else {
      if (idx < 128) {  // top two bits clear -> saturate
        r = 0xFFFFFFFFu;
      } else {
        int aa;
        if (idx < 256) {
          aa = idx * 2 + 1;
        } else {
          aa = (idx >> 1) << 1;
          aa = (aa + 1) * 2;
        }
        int b = 512;
        while (static_cast<int64_t>(aa) * (b + 1) * (b + 1) < (1 << 28)) {
          b += 1;
        }
        int estimate = (b + 1) / 2;
        r = static_cast<uint32_t>(estimate) << 23;
      }
    }
    t[idx] = r;
  }
  return t;
}
const uint32_t* HeavyUnsignedEstimateTable(bool is_rsqrt) {
  static const uint32_t* const kRecpe = BuildHeavyUnsignedEstimateTable(false);
  static const uint32_t* const kRsqrte = BuildHeavyUnsignedEstimateTable(true);
  return is_rsqrt ? kRsqrte : kRecpe;
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
  if (args.ftype == 0b10) {
    Undefined();
    return;
  }
  if (args.ftype == 0b11) {
    // FP16: widen both operands to FP32, then UCOMISS (mirrors lite FpCompare).
    // FCMP Hn,#0.0 compares against +0.0 (a zeroed XMM widens to +0.0f). Bails
    // to lite/interp without host F16C.
    if (!host_platform::kHasF16C) {
      Undefined();
      return;
    }
    FpRegister n = EmitWidenHalfToF32(args.rn);
    FpRegister m = args.with_zero ? AllocZeroedSimdReg() : EmitWidenHalfToF32(args.rm);
    Register flags = std::get<0>(Gen<x86_64::UcomissXRegXReg>(n.machine_reg(), m.machine_reg()));
    EmitStoreArmFpNZCV(flags);
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
  if (args.ftype == 0b10 || (args.ftype == 0b11 && !host_platform::kHasF16C)) {
    Undefined();
    return;
  }
  const bool is_double = (args.ftype == 0b01);
  const bool is_half = (args.ftype == 0b11);
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
    FpRegister xmm_n = is_half ? EmitWidenHalfToF32(args.rn) : GetVRegScalar(args.rn, is_double);
    FpRegister xmm_m = is_half ? EmitWidenHalfToF32(args.rm) : GetVRegScalar(args.rm, is_double);
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

  // FCVTAS/FCVTAU (ties-away): add copysign(0.5, x), gated to 0 when |x| is
  // already an integer (>= 2^23 for FP32, >= 2^52 for FP64, where a 0.5 addend
  // would round the wrong way), then let the truncating cvtt + saturation ladder
  // below finish. NaN/±Inf pass through: their |bits| exceed the threshold, so
  // the addend is gated to 0 and the downstream NaN/overflow branches still fire.
  // Mirrors the FP32/FP64 FRINTA branchless recipe in FpDataProc1 and lite's
  // FCVTAS/AU path. Round into a private temp so the guest v[] slot is untouched.
  // The magnitude compare is a signed PCMPGT on |bits(x)| (top bit cleared, so
  // signed == unsigned).
  if (ties_away) {
    FpRegister sign = AllocZeroedSimdReg();
    FpRegister absx = AllocZeroedSimdReg();
    FpRegister half = AllocTempSimdReg();
    FpRegister thresh = AllocTempSimdReg();
    FpRegister rounded = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(rounded.machine_reg(), xmm.machine_reg());
    if (is_double) {
      // addend = copysign(0.5d, x) = (x & sign_bit) | bits(0.5d).
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
      builder_.Gen<x86_64::PsllqXRegImm>(sign.machine_reg(), int8_t{63});  // 0x8000...0
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), xmm.machine_reg());
      builder_.Gen<x86_64::MovqXRegReg>(half.machine_reg(),
                                        GetImm(uint64_t{0x3FE0000000000000}));  // 0.5d
      builder_.Gen<x86_64::PorXRegXReg>(sign.machine_reg(), half.machine_reg());
      // gate = (|x| < 2^52) ? all-ones : 0, via thresh > |x|.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(absx.machine_reg(), absx.machine_reg());
      builder_.Gen<x86_64::PsrlqXRegImm>(absx.machine_reg(), int8_t{1});  // 0x7FFF...F
      builder_.Gen<x86_64::PandXRegXReg>(absx.machine_reg(), xmm.machine_reg());
      builder_.Gen<x86_64::MovqXRegReg>(thresh.machine_reg(),
                                        GetImm(uint64_t{0x4330000000000000}));  // 2^52
      builder_.Gen<x86_64::PcmpgtqXRegXReg>(thresh.machine_reg(), absx.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), thresh.machine_reg());
      builder_.Gen<x86_64::AddpdXRegXReg>(rounded.machine_reg(), sign.machine_reg());
    } else {
      // addend = copysign(0.5f, x) = (x & sign_bit) | bits(0.5f).
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
      builder_.Gen<x86_64::AddpsXRegXReg>(rounded.machine_reg(), sign.machine_reg());
    }
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

    case Decoder::AtomicOp::kLdclr:
    case Decoder::AtomicOp::kLdset:
    case Decoder::AtomicOp::kLdeor: {
      // LSE bitwise fetch-and-{clear,set,xor}: old=[Xn]; [Xn] = old OP Xs;
      // Xt = old. x86 has no single fetch-and-bitwise, so emit a LOCK CMPXCHG
      // retry loop. Mirrors lite_translator's kLdclr/kLdset/kLdeor. The loop
      // re-reads [mem] fresh each iteration, so no value is carried across the
      // back-edge (only the loop-invariant mask enters the loop). All four
      // widths: the relaxed load zero-extends the sized value (byte/half via
      // Movzxbl/Movzxwl, 32-bit via Movl, 64-bit via Movq), and the sized LOCK
      // CMPXCHG (b/w/l/q) compares the accumulator's low bits against [mem].
      // The bitwise op runs on the full 64-bit register; only the low `size`
      // bytes reach memory (CMPXCHGB/W store AL/AX), so the upper bits are
      // don't-care and the exit reads the zero-extended `old` for Xt.
      if (args.size > 3) {
        UndefinedReturningVoid();
        return;
      }
      Register mask = (args.rs != 31) ? GetReg(args.rs) : GetImm(0);
      // LDCLR clears the bits in Xs, i.e. AND with ~Xs. Precompute ~Xs once
      // (loop-invariant). For byte/half only the low 8/16 bits of ~Xs matter.
      Register applied = mask;
      if (args.op == Decoder::AtomicOp::kLdclr) {
        applied = std::get<0>(Gen<x86_64::NotqReg, kNoSSA>(Copy(mask)));
      }

      auto* ir = builder_.ir();
      MachineBasicBlock* loop_bb = ir->NewBasicBlock();
      MachineBasicBlock* exit_bb = ir->NewBasicBlock();
      ir->AddEdge(builder_.bb(), loop_bb);
      builder_.Gen<PseudoBranch>(loop_bb);

      builder_.StartBasicBlock(loop_bb);
      // Relaxed, zero-extended load of the current memory value (the CMPXCHG
      // provides atomicity/ordering; a mismatch just retries).
      Register old;
      switch (args.size) {
        case 0:
          old = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = base}));
          break;
        case 1:
          old = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = base}));
          break;
        case 2:
          old = std::get<0>(Gen<x86_64::MovlRegOp>({.base = base}));
          break;
        default:  // size == 3
          old = std::get<0>(Gen<x86_64::MovqRegOp>({.base = base}));
          break;
      }
      Register new_val = Copy(old);
      switch (args.op) {
        case Decoder::AtomicOp::kLdclr:
          new_val = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(new_val, applied));
          break;
        case Decoder::AtomicOp::kLdset:
          new_val = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(new_val, applied));
          break;
        default:  // kLdeor
          new_val = std::get<0>(Gen<x86_64::XorqRegReg, kNoSSA>(new_val, applied));
          break;
      }
      // CMPXCHG(expected=old, [mem], new_val). On a miss RAX is reloaded from
      // [mem] and ZF=0 -> loop; on a match ZF=1 -> exit. Pass a copy of `old` so
      // the original loaded value survives to the exit block for Xt. The sized
      // CMPXCHG compares only the low 8/16/32/64 bits of the accumulator.
      Register expected = Copy(old);
      Register cmpxchg_flags;
      Register unused_rax;
      switch (args.size) {
        case 0:
          std::tie(unused_rax, cmpxchg_flags) =
              Gen<x86_64::LockCmpXchgbRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 1:
          std::tie(unused_rax, cmpxchg_flags) =
              Gen<x86_64::LockCmpXchgwRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 2:
          std::tie(unused_rax, cmpxchg_flags) =
              Gen<x86_64::LockCmpXchglRegOpReg>(expected, {.base = base}, new_val);
          break;
        default:  // size == 3
          std::tie(unused_rax, cmpxchg_flags) =
              Gen<x86_64::LockCmpXchgqRegOpReg>(expected, {.base = base}, new_val);
          break;
      }
      UNUSED_ARGS(unused_rax);
      ir->AddEdge(loop_bb, loop_bb);
      ir->AddEdge(loop_bb, exit_bb);
      builder_.Gen<PseudoCondBranch>(
          x86_64::Assembler::Condition::kNotZero, loop_bb, exit_bb, cmpxchg_flags);

      builder_.StartBasicBlock(exit_bb);
      if (args.rt != 31) {
        // `old` was loaded zero-extended (Movzxbl/Movzxwl/Movl clear the upper
        // bits; Movq is full-width), matching ARM Wt/Xt zero-extension.
        SetReg(args.rt, old);
      }
      return;
    }

    case Decoder::AtomicOp::kLdsmax:
    case Decoder::AtomicOp::kLdsmin:
    case Decoder::AtomicOp::kLdumax:
    case Decoder::AtomicOp::kLdumin: {
      // LSE atomic min/max: old=[Xn]; [Xn] = {max,min}(old, Xs); Xt = old. Same
      // CMPXCHG retry loop as LDCLR/LDSET/LDEOR, with a select diamond inside the
      // loop to compute the new value. All four widths.
      //
      // Sign-extension discipline for the compare: both operands are extended to
      // 64 bits at the guest size before Cmpq, so the ordering condition is
      // correct for the guest width.
      //   * signed (LDSMAX/LDSMIN): sign-extend from the guest size — byte
      //     Movsxbq, half Movsxwq, 32-bit Movsxlq; 64-bit is a no-op.
      //   * unsigned (LDUMAX/LDUMIN): zero-extend from the guest size — byte
      //     Movzxbq, half Movzxwq, 32-bit Movl; 64-bit is a no-op.
      // `old` was loaded via a zero-extending sized load, but for the SIGNED
      // compare we must re-extend from the byte/half so its sign bit is honored;
      // for the UNSIGNED compare re-zero-extending is idempotent. `operand`
      // (guest Xs) may hold non-zero upper bits, so it is always re-extended too.
      // Only the low `size` bytes of the picked value reach memory (sized
      // CMPXCHG), and the exit reads the zero-extended `old` for Xt.
      if (args.size > 3) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_signed = (args.op == Decoder::AtomicOp::kLdsmax ||
                              args.op == Decoder::AtomicOp::kLdsmin);
      const bool is_max = (args.op == Decoder::AtomicOp::kLdsmax ||
                           args.op == Decoder::AtomicOp::kLdumax);
      Register operand = (args.rs != 31) ? GetReg(args.rs) : GetImm(0);

      auto* ir = builder_.ir();
      MachineBasicBlock* loop_bb = ir->NewBasicBlock();
      MachineBasicBlock* pick_old_bb = ir->NewBasicBlock();
      MachineBasicBlock* pick_op_bb = ir->NewBasicBlock();
      MachineBasicBlock* cmpxchg_bb = ir->NewBasicBlock();
      MachineBasicBlock* exit_bb = ir->NewBasicBlock();
      ir->AddEdge(builder_.bb(), loop_bb);
      builder_.Gen<PseudoBranch>(loop_bb);

      builder_.StartBasicBlock(loop_bb);
      Register old;
      switch (args.size) {
        case 0:
          old = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = base}));
          break;
        case 1:
          old = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = base}));
          break;
        case 2:
          old = std::get<0>(Gen<x86_64::MovlRegOp>({.base = base}));
          break;
        default:  // size == 3
          old = std::get<0>(Gen<x86_64::MovqRegOp>({.base = base}));
          break;
      }
      // Extend both operands to a full 64-bit signed/unsigned value for Cmpq.
      Register ext_old = old;
      Register ext_op = operand;
      switch (args.size) {
        case 0:
          if (is_signed) {
            ext_old = std::get<0>(Gen<x86_64::MovsxbqRegReg>(old));
            ext_op = std::get<0>(Gen<x86_64::MovsxbqRegReg>(operand));
          } else {
            ext_old = std::get<0>(Gen<x86_64::MovzxbqRegReg>(old));
            ext_op = std::get<0>(Gen<x86_64::MovzxbqRegReg>(operand));
          }
          break;
        case 1:
          if (is_signed) {
            ext_old = std::get<0>(Gen<x86_64::MovsxwqRegReg>(old));
            ext_op = std::get<0>(Gen<x86_64::MovsxwqRegReg>(operand));
          } else {
            ext_old = std::get<0>(Gen<x86_64::MovzxwqRegReg>(old));
            ext_op = std::get<0>(Gen<x86_64::MovzxwqRegReg>(operand));
          }
          break;
        case 2:
          if (is_signed) {
            ext_old = std::get<0>(Gen<x86_64::MovsxlqRegReg>(old));
            ext_op = std::get<0>(Gen<x86_64::MovsxlqRegReg>(operand));
          } else {
            ext_old = std::get<0>(Gen<x86_64::MovlRegReg>(old));
            ext_op = std::get<0>(Gen<x86_64::MovlRegReg>(operand));
          }
          break;
        default:  // size == 3: already full width.
          break;
      }
      Register cmp_flags = std::get<0>(Gen<x86_64::CmpqRegReg>(ext_old, ext_op));
      // Keep old when old {>=,<=} op (max/min). Signed uses G/L, unsigned A/B.
      x86_64::Assembler::Condition keep_old =
          is_max ? (is_signed ? x86_64::Assembler::Condition::kGreaterEqual
                              : x86_64::Assembler::Condition::kAboveEqual)
                 : (is_signed ? x86_64::Assembler::Condition::kLessEqual
                              : x86_64::Assembler::Condition::kBelowEqual);
      Register new_val = AllocTempReg();
      ir->AddEdge(loop_bb, pick_old_bb);
      ir->AddEdge(loop_bb, pick_op_bb);
      builder_.Gen<PseudoCondBranch>(keep_old, pick_old_bb, pick_op_bb, cmp_flags);

      builder_.StartBasicBlock(pick_old_bb);
      builder_.Gen<PseudoCopy>(new_val, old, 8);
      ir->AddEdge(pick_old_bb, cmpxchg_bb);
      builder_.Gen<PseudoBranch>(cmpxchg_bb);

      builder_.StartBasicBlock(pick_op_bb);
      builder_.Gen<PseudoCopy>(new_val, operand, 8);
      ir->AddEdge(pick_op_bb, cmpxchg_bb);
      builder_.Gen<PseudoBranch>(cmpxchg_bb);

      builder_.StartBasicBlock(cmpxchg_bb);
      Register expected = Copy(old);
      Register cmpxchg_flags;
      Register unused_rax;
      switch (args.size) {
        case 0:
          std::tie(unused_rax, cmpxchg_flags) =
              Gen<x86_64::LockCmpXchgbRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 1:
          std::tie(unused_rax, cmpxchg_flags) =
              Gen<x86_64::LockCmpXchgwRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 2:
          std::tie(unused_rax, cmpxchg_flags) =
              Gen<x86_64::LockCmpXchglRegOpReg>(expected, {.base = base}, new_val);
          break;
        default:  // size == 3
          std::tie(unused_rax, cmpxchg_flags) =
              Gen<x86_64::LockCmpXchgqRegOpReg>(expected, {.base = base}, new_val);
          break;
      }
      UNUSED_ARGS(unused_rax);
      ir->AddEdge(cmpxchg_bb, loop_bb);
      ir->AddEdge(cmpxchg_bb, exit_bb);
      builder_.Gen<PseudoCondBranch>(
          x86_64::Assembler::Condition::kNotZero, loop_bb, exit_bb, cmpxchg_flags);

      builder_.StartBasicBlock(exit_bb);
      if (args.rt != 31) {
        SetReg(args.rt, old);
      }
      return;
    }

    case Decoder::AtomicOp::kCasp: {
      // CASP: compare-and-swap pair. Rs:Rs+1 = expected, Rt:Rt+1 = new; the old
      // pair is written back to Rs:Rs+1. size=2 packs each 32-bit half into a
      // 64-bit value and uses LOCK CMPXCHGq. size=3 (128-bit DWCAS) uses LOCK
      // CMPXCHG16B (RDX:RAX = expected, RCX:RBX = desired, RDX:RAX = old after).
      // Mirrors lite for both widths.
      if (args.size != 2 && args.size != 3) {
        UndefinedReturningVoid();
        return;
      }
      const uint8_t rs_lo = args.rs;
      const uint8_t rs_hi = static_cast<uint8_t>(args.rs + 1);
      const uint8_t rt_lo = args.rt;
      const uint8_t rt_hi = static_cast<uint8_t>(args.rt + 1);

      if (args.size == 3) {
        // 128-bit DWCAS. Like the single CAS, CASP always writes the old pair
        // back to Rs:Rs+1 and does not branch on the result, so the CMPXCHG16B
        // FLAGS output is discarded and GenRecoveryBlockForLastInsn() can wrap
        // the memory access (nothing FLAGS-class is live across the recovery
        // edge — unlike STXP, where FLAGS feed the status branch). Each half is
        // read/written as a full 64-bit guest register; XZR (31) reads 0 and
        // discards its write-back half.
        Register exp_lo = (rs_lo != 31) ? GetReg(rs_lo) : GetImm(0);
        Register exp_hi = (rs_hi != 31) ? GetReg(rs_hi) : GetImm(0);
        Register des_lo = (rt_lo != 31) ? GetReg(rt_lo) : GetImm(0);
        Register des_hi = (rt_hi != 31) ? GetReg(rt_hi) : GetImm(0);
        auto res = Gen<x86_64::LockCmpXchg16bRegRegRegRegOp>(
            exp_lo, exp_hi, des_lo, des_hi, {.base = base});
        GenRecoveryBlockForLastInsn();
        Register old_lo = res[0];  // RAX
        Register old_hi = res[1];  // RDX
        if (rs_lo != 31) {
          SetReg(rs_lo, old_lo);
        }
        if (rs_hi != 31) {
          SetReg(rs_hi, old_hi);
        }
        return;
      }

      // Pack (hi << 32) | zext(lo) for both expected and desired.
      auto pack = [&](uint8_t lo, uint8_t hi) -> Register {
        Register v = (lo != 31)
            ? std::get<0>(Gen<x86_64::MovlRegReg>(GetReg(lo)))
            : std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
        if (hi != 31) {
          Register h = std::get<0>(Gen<x86_64::MovlRegReg>(GetReg(hi)));
          h = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(h, int8_t{32}));
          v = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(v, h));
        }
        return v;
      };
      Register expected = pack(rs_lo, rs_hi);
      Register desired = pack(rt_lo, rt_hi);
      Register old =
          std::get<0>(Gen<x86_64::LockCmpXchgqRegOpReg>(expected, {.base = base}, desired));
      GenRecoveryBlockForLastInsn();
      // Unpack the old pair back to Rs:Rs+1 (each half zero-extended).
      if (rs_lo != 31) {
        SetReg(rs_lo, std::get<0>(Gen<x86_64::MovlRegReg>(old)));
      }
      if (rs_hi != 31) {
        SetReg(rs_hi, std::get<0>(Gen<x86_64::ShrqRegImm>(Copy(old), int8_t{32})));
      }
      return;
    }

    case Decoder::AtomicOp::kLdxp: {
      // LDXP/LDAXP (load-exclusive pair). Loads Rt:Rt2 atomically and arms the
      // monitor (reservation_address = base, reservation_value = the full pair).
      // Mirrors lite_translator_int_carry_excl.inc's kLdxp. x86 TSO gives
      // acquire order for free, so LDAXP needs no extra fence.
      if (args.size == 2) {
        // 32-bit pair: a single aligned 64-bit load is atomic on x86.
        Register pair = std::get<0>(Gen<x86_64::MovqRegOp>({.base = base}));
        GenRecoveryBlockForLastInsn();
        // reservation_address = base; reservation_value = the 64-bit pair.
        builder_.Gen<x86_64::MovqOpReg>(
            {.base = x86_64::kMachineRegRBP, .disp = resv_addr_off}, base);
        builder_.Gen<x86_64::MovqOpReg>(
            {.base = x86_64::kMachineRegRBP, .disp = resv_val_off}, pair);
        if (args.rt != 31) {
          // Rt = low 32, zero-extended (Movl clears the upper 32).
          SetReg(args.rt, std::get<0>(Gen<x86_64::MovlRegReg>(pair)));
        }
        if (args.rt2 != 31) {
          // Rt2 = high 32, zero-extended.
          SetReg(args.rt2, std::get<0>(Gen<x86_64::ShrqRegImm>(Copy(pair), int8_t{32})));
        }
        return;
      }
      if (args.size == 3) {
        // 64-bit pair: atomic 128-bit read via LOCK CMPXCHG16B with
        // expected==desired==0 (writes 0 only when *addr==0, i.e. a no-op) — the
        // same trick the interpreter and lite tier use. FLAGS discarded, so
        // GenRecoveryBlockForLastInsn() safely wraps the access.
        Register zero_a = GetImm(0);
        Register zero_b = GetImm(0);
        Register zero_c = GetImm(0);
        Register zero_d = GetImm(0);
        auto res = Gen<x86_64::LockCmpXchg16bRegRegRegRegOp>(
            zero_a, zero_b, zero_c, zero_d, {.base = base});
        GenRecoveryBlockForLastInsn();
        Register old_lo = res[0];  // RAX
        Register old_hi = res[1];  // RDX
        // reservation_address = base; reservation_value = full 128-bit pair.
        builder_.Gen<x86_64::MovqOpReg>(
            {.base = x86_64::kMachineRegRBP, .disp = resv_addr_off}, base);
        builder_.Gen<x86_64::MovqOpReg>(
            {.base = x86_64::kMachineRegRBP, .disp = resv_val_off}, old_lo);
        builder_.Gen<x86_64::MovqOpReg>(
            {.base = x86_64::kMachineRegRBP, .disp = resv_val_off + 8}, old_hi);
        if (args.rt != 31) {
          SetReg(args.rt, old_lo);
        }
        if (args.rt2 != 31) {
          SetReg(args.rt2, old_hi);
        }
        return;
      }
      UndefinedReturningVoid();
      return;
    }

    case Decoder::AtomicOp::kStxp: {
      // STXP/STLXP (store-exclusive pair). Stores Rt:Rt2 atomically iff the
      // monitor still holds the value LDXP read (an exact compare-and-swap
      // against reservation_value); reports success(0)/failure(1) in Rs and
      // clears the reservation. Mirrors lite's kStxp and the single-register
      // STXR structure: the status vreg is written only in the terminal
      // predecessors of the merge block (never in the entry block), so its live
      // range keeps the two-def/one-use shape the lifetime analyzer expects, and
      // — like STXR — the CMPXCHG's FLAGS feed the status branch, so NO recovery
      // split wraps the CMPXCHG (a split would strand the FLAGS across the edge;
      // and a faulting STXP is pathological because a prior LDXP already
      // validated the address when it armed the reservation).
      if (args.size == 2) {
        // 32-bit pair: new = (Rt2_lo32 << 32) | Rt_lo32; single 64-bit CAS
        // against the reserved 64-bit value.
        Register new_val =
            (args.rt != 31) ? std::get<0>(Gen<x86_64::MovlRegReg>(GetReg(args.rt))) : GetImm(0);
        if (args.rt2 != 31) {
          Register hi = std::get<0>(Gen<x86_64::MovlRegReg>(GetReg(args.rt2)));
          hi = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(hi, int8_t{32}));
          new_val = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(new_val, hi));
        }
        Register resv_addr = std::get<0>(
            Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = resv_addr_off}));
        builder_.GenPutImm(resv_addr_off, 0);  // STXP always clears the reservation.

        Register status = AllocTempReg();
        auto* ir = builder_.ir();
        auto* cur_bb = builder_.bb();
        MachineBasicBlock* match_bb = ir->NewBasicBlock();
        MachineBasicBlock* fail_bb = ir->NewBasicBlock();
        MachineBasicBlock* swap_ok_bb = ir->NewBasicBlock();
        MachineBasicBlock* merge_bb = ir->NewBasicBlock();
        ir->AddEdge(cur_bb, match_bb);
        ir->AddEdge(cur_bb, fail_bb);
        builder_.Gen<PseudoCondBranch>(
            x86_64::Assembler::Condition::kNotEqual, fail_bb, match_bb,
            std::get<0>(Gen<x86_64::CmpqRegReg>(resv_addr, base)));

        builder_.StartBasicBlock(match_bb);
        Register expected = std::get<0>(
            Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = resv_val_off}));
        auto [unused_rax, host_flags] =
            Gen<x86_64::LockCmpXchgqRegOpReg>(expected, {.base = base}, new_val);
        UNUSED_ARGS(unused_rax);
        ir->AddEdge(builder_.bb(), fail_bb);
        ir->AddEdge(builder_.bb(), swap_ok_bb);
        builder_.Gen<PseudoCondBranch>(
            x86_64::Assembler::Condition::kNotZero, fail_bb, swap_ok_bb, host_flags);

        builder_.StartBasicBlock(swap_ok_bb);
        builder_.Gen<PseudoDefReg>(status);
        builder_.Gen<x86_64::XorqRegReg>(status, status, GetFlagsRegister());
        ir->AddEdge(swap_ok_bb, merge_bb);
        builder_.Gen<PseudoBranch>(merge_bb);

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
      if (args.size == 3) {
        // 64-bit pair via LOCK CMPXCHG16B: expected = reservation_value
        // (RDX:RAX), desired = Rt:Rt2 (RCX:RBX). Guard on reservation_address ==
        // base; on mismatch the store fails without touching memory.
        Register des_lo = (args.rt != 31) ? GetReg(args.rt) : GetImm(0);
        Register des_hi = (args.rt2 != 31) ? GetReg(args.rt2) : GetImm(0);
        Register resv_addr = std::get<0>(
            Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = resv_addr_off}));
        builder_.GenPutImm(resv_addr_off, 0);  // STXP always clears the reservation.

        Register status = AllocTempReg();
        auto* ir = builder_.ir();
        auto* cur_bb = builder_.bb();
        MachineBasicBlock* match_bb = ir->NewBasicBlock();
        MachineBasicBlock* fail_bb = ir->NewBasicBlock();
        MachineBasicBlock* swap_ok_bb = ir->NewBasicBlock();
        MachineBasicBlock* merge_bb = ir->NewBasicBlock();
        ir->AddEdge(cur_bb, match_bb);
        ir->AddEdge(cur_bb, fail_bb);
        builder_.Gen<PseudoCondBranch>(
            x86_64::Assembler::Condition::kNotEqual, fail_bb, match_bb,
            std::get<0>(Gen<x86_64::CmpqRegReg>(resv_addr, base)));

        builder_.StartBasicBlock(match_bb);
        // expected = reservation_value (low @ resv_val_off, high @ +8).
        Register exp_lo = std::get<0>(
            Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = resv_val_off}));
        Register exp_hi = std::get<0>(Gen<x86_64::MovqRegOp>(
            {.base = x86_64::kMachineRegRBP, .disp = resv_val_off + 8}));
        auto res = Gen<x86_64::LockCmpXchg16bRegRegRegRegOp>(
            exp_lo, exp_hi, des_lo, des_hi, {.base = base});
        Register host_flags = res[2];  // FLAGS (== GetFlagsRegister())
        // No recovery split here (like STXR): FLAGS feed the status branch.
        ir->AddEdge(builder_.bb(), fail_bb);
        ir->AddEdge(builder_.bb(), swap_ok_bb);
        builder_.Gen<PseudoCondBranch>(
            x86_64::Assembler::Condition::kNotZero, fail_bb, swap_ok_bb, host_flags);

        builder_.StartBasicBlock(swap_ok_bb);
        builder_.Gen<PseudoDefReg>(status);
        builder_.Gen<x86_64::XorqRegReg>(status, status, GetFlagsRegister());
        ir->AddEdge(swap_ok_bb, merge_bb);
        builder_.Gen<PseudoBranch>(merge_bb);

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
      UndefinedReturningVoid();
      return;
    }

    default:
      // Any remaining atomic op form not mirrored into the heavy tier bails to
      // the lite translator (correct, just slower).
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

// ---------------------------------------------------------------------------
// Decoder-callback handlers and large emit-helpers (moved out of frontend.h
// to match the upstream riscv64 h/cc split; definitions kept in header
// declaration order).
// ---------------------------------------------------------------------------

// ADD/SUB (immediate), including the flag-setting SUBS/ADDS/CMP/CMN forms.
// When set_flags is true the op always emits (even imm==0) so host EFLAGS are
// valid, then EmitMaterializeNZCV packs NZCV into cpu.flags exactly as
// lite_translator.h::EmitStoreArmNZCV. CMP/CMN to XZR (rd==31) discard the
// result in the SemanticsPlayer (SetRegOrIgnore). Mirrors
// lite_translator.h::AddSubImm: a 32-bit op uses the l-suffix insns (which
// zero-extend the upper 32 bits, matching ARM64 W-register write semantics).
Register HeavyOptimizerFrontend::AddSubImm(bool is_sub, bool set_flags, bool is_64bit, Register src, uint32_t imm) {
  // A prior callback for this guest instruction may have already bailed (e.g.
  // post-index Load() bails before this AddImm-equivalent runs); emit nothing.
  if (!success()) {
    return AllocTempReg();
  }
  // The ARM imm12 fits in int32 and x86 add/sub-immediate take int32.
  if (is_64bit) {
    Register res = Copy(src);
    // When setting flags, always emit the op (even imm==0) so EFLAGS are valid.
    if (set_flags || imm != 0) {
      if (is_sub) {
        auto [r, flags] = Gen<x86_64::SubqRegImm, kNoSSA>(res, static_cast<int32_t>(imm));
        res = r;
        if (set_flags) {
          EmitMaterializeNZCV(flags, /*is_sub=*/true);
        }
      } else {
        auto [r, flags] = Gen<x86_64::AddqRegImm, kNoSSA>(res, static_cast<int32_t>(imm));
        res = r;
        if (set_flags) {
          EmitMaterializeNZCV(flags, /*is_sub=*/false);
        }
      }
    }
    return res;
  }
  // 32-bit: a 32-bit mov zero-extends src to 64, then the 32-bit op keeps the
  // upper 32 bits clear (ARM64 W-write semantics).
  Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
  if (set_flags || imm != 0) {
    if (is_sub) {
      auto [r, flags] = Gen<x86_64::SublRegImm, kNoSSA>(res, static_cast<int32_t>(imm));
      res = r;
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/true);
      }
    } else {
      auto [r, flags] = Gen<x86_64::AddlRegImm, kNoSSA>(res, static_cast<int32_t>(imm));
      res = r;
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/false);
      }
    }
  }
  return res;
}

Register HeavyOptimizerFrontend::AddSubImmTags(bool is_sub, Register src, uint8_t uimm6, uint8_t uimm4) {
  UndefinedReturningReg();
  UNUSED_ARGS(is_sub, src, uimm6, uimm4);
  return AllocTempReg();
}

// AND/ORR/EOR (immediate, decoded 64-bit bitmask), including ANDS/TST. x86 AND
// clears CF and OF, so ANDS materializes ARM64 NZCV with C=0 and V=0 (is_sub
// false, no borrow XOR). x86 logical-immediate forms only take a 32-bit
// immediate, but the bitmask
// immediate needs the full 64 bits, so materialize it into a register and use
// the reg-reg forms (mirrors lite_translator.h::LogicalImm).
Register HeavyOptimizerFrontend::LogicalImm(Decoder::LogicalImmOpcode opcode, bool is_64bit, Register src, uint64_t imm) {
  if (!success()) {
    return AllocTempReg();
  }
  Register imm_reg = GetImm(imm);
  if (is_64bit) {
    Register res = Copy(src);
    switch (opcode) {
      case Decoder::LogicalImmOpcode::kAnd:
        return std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, imm_reg));
      case Decoder::LogicalImmOpcode::kAnds: {
        // ANDS/TST: x86 AND clears CF and OF, so ARM64 C=0 and V=0 naturally.
        auto [r, flags] = Gen<x86_64::AndqRegReg, kNoSSA>(res, imm_reg);
        EmitMaterializeNZCV(flags, /*is_sub=*/false);
        return r;
      }
      case Decoder::LogicalImmOpcode::kOrr:
        return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(res, imm_reg));
      case Decoder::LogicalImmOpcode::kEor:
        return std::get<0>(Gen<x86_64::XorqRegReg, kNoSSA>(res, imm_reg));
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }
  // 32-bit op: the l-suffix form zero-extends the result to 64 bits.
  Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
  switch (opcode) {
    case Decoder::LogicalImmOpcode::kAnd:
      return std::get<0>(Gen<x86_64::AndlRegReg, kNoSSA>(res, imm_reg));
    case Decoder::LogicalImmOpcode::kAnds: {
      auto [r, flags] = Gen<x86_64::AndlRegReg, kNoSSA>(res, imm_reg);
      EmitMaterializeNZCV(flags, /*is_sub=*/false);
      return r;
    }
    case Decoder::LogicalImmOpcode::kOrr:
      return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(res, imm_reg));
    case Decoder::LogicalImmOpcode::kEor:
      return std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(res, imm_reg));
    default:
      UndefinedReturningReg();
      return AllocTempReg();
  }
}

// MOVZ / MOVN: result is compile-time known.
Register HeavyOptimizerFrontend::MoveWide(Decoder::MoveWideOpcode opcode, bool is_64bit, uint16_t imm16, uint8_t shift) {
  uint64_t value = static_cast<uint64_t>(imm16) << shift;
  if (opcode == Decoder::MoveWideOpcode::kMovn) {
    value = ~value;
  }
  if (!is_64bit) {
    value &= 0xFFFFFFFFULL;
  }
  return std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(value)));
}

// MOVK: keep other bits of `current`, overwrite the 16-bit window at `shift`.
// x86_64 AND/OR-immediate forms only accept a 32-bit sign-extended immediate,
// but ~mask / value can need a full 64-bit immediate (shift up to 48), so
// materialize each constant into a register and use the reg-reg forms,
// mirroring the lite translator.
Register HeavyOptimizerFrontend::MoveWideKeep(Register current, uint16_t imm16, uint8_t shift, bool is_64bit) {
  uint64_t mask = static_cast<uint64_t>(0xFFFF) << shift;
  uint64_t value = static_cast<uint64_t>(imm16) << shift;
  // res = current & ~mask
  Register not_mask = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(~mask)));
  Register res = std::get<0>(Gen<x86_64::AndqRegReg>(current, not_mask));
  // res = res | value
  Register value_reg = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(value)));
  res = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(res, value_reg));
  if (!is_64bit) {
    // Zero-extend 32->64 (a 32-bit mov clears the upper 32 bits).
    res = std::get<0>(Gen<x86_64::MovlRegReg, kNoSSA>(res));
  }
  return res;
}

// ADR / ADRP: the target is a pure function of the (translation-time-constant)
// guest PC and the decoded offset, so materialize it as an immediate. ADRP
// page-aligns the PC first. Mirrors lite_translator.h::PcRelAddr.
Register HeavyOptimizerFrontend::PcRelAddr(bool is_adrp, int64_t offset) {
  if (!success()) {
    return AllocTempReg();
  }
  GuestAddr pc = GetInsnAddr();
  GuestAddr target = is_adrp ? ((pc & ~static_cast<GuestAddr>(0xFFF)) + offset) : (pc + offset);
  return GetImm(static_cast<uint64_t>(target));
}

// LDR/LDRSW (literal): load from [insn_addr + offset]. The address is constant
// at translation time, so materialize it with GetImm and reuse Load() (which
// applies TBI, emits the size-appropriate movzx/movsx, and sets the recovery
// point). Mirrors lite_translator.h::LoadLiteral.
Register HeavyOptimizerFrontend::LoadLiteral(Decoder::LoadStoreSize size, bool is_signed, int64_t offset) {
  if (!success()) {
    return AllocTempReg();
  }
  GuestAddr target = GetInsnAddr() + offset;
  Register addr = GetImm(target);
  bool is_64bit_target = (size == Decoder::LoadStoreSize::k64bit) || is_signed;
  return Load(size, is_signed, is_64bit_target, addr, 0);
}

// SBFM/UBFM/BFM (bitfield move). Mirrors lite_translator.h::Bitfield. The
// 32-bit paths use l-suffix shifts/and (which zero-extend the upper 32 bits,
// matching ARM64 W-register write semantics).
Register HeavyOptimizerFrontend::Bitfield(Decoder::BitfieldOpcode opcode,
                                          bool is_64bit,
                                          Register dst_val,
                                          Register src,
                                          uint8_t immr,
                                          uint8_t imms) {
  if (!success()) {
    return AllocTempReg();
  }
  unsigned reg_size = is_64bit ? 64 : 32;

  if (opcode == Decoder::BitfieldOpcode::kUbfm) {
    // LSR: UBFM Rd, Rn, #shift, #(regsize-1).
    if (imms == reg_size - 1) {
      if (is_64bit) {
        Register res = Copy(src);
        if (immr != 0) {
          res = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
        }
        return res;
      }
      Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      if (immr != 0) {
        res = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
      }
      return res;
    }
    // LSL: UBFM Rd, Rn, #(regsize-shift), #(regsize-1-shift).
    if (imms + 1 == immr && imms < reg_size - 1) {
      uint8_t shift = reg_size - immr;
      if (is_64bit) {
        Register res = Copy(src);
        return std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(shift)));
      }
      Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      return std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(shift)));
    }
    // UXTB: UBFM Wd, Wn, #0, #7.
    if (!is_64bit && immr == 0 && imms == 7) {
      return std::get<0>(Gen<x86_64::MovzxblRegReg>(src));
    }
    // UXTH: UBFM Wd, Wn, #0, #15.
    if (!is_64bit && immr == 0 && imms == 15) {
      return std::get<0>(Gen<x86_64::MovzxwlRegReg>(src));
    }
    // General UBFM (UBFX extract; UBFIZ insert).
    if (imms >= immr) {
      // UBFX-like: extract bits[imms:immr] of src to low bits of dst.
      unsigned width = imms - immr + 1;
      uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
      if (is_64bit) {
        Register res = Copy(src);
        if (immr != 0) {
          res = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
        }
        if (width < 64) {
          Register mask_reg = GetImm(mask);
          res = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, mask_reg));
        }
        return res;
      }
      Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      if (immr != 0) {
        res = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
      }
      if (width < 32) {
        res = std::get<0>(
            Gen<x86_64::AndlRegImm, kNoSSA>(res, static_cast<int32_t>(mask & 0xFFFFFFFFULL)));
      }
      return res;
    }
    // UBFIZ-like: extract low (imms+1) bits of src, shift left by (reg_size-immr).
    unsigned width = imms + 1;
    unsigned pos = reg_size - immr;
    uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    if (is_64bit) {
      Register res = Copy(src);
      if (width < 64) {
        Register mask_reg = GetImm(mask);
        res = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, mask_reg));
      }
      if (pos != 0) {
        res = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(pos)));
      }
      return res;
    }
    Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
    if (width < 32) {
      res = std::get<0>(
          Gen<x86_64::AndlRegImm, kNoSSA>(res, static_cast<int32_t>(mask & 0xFFFFFFFFULL)));
    }
    if (pos != 0) {
      res = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(pos)));
    }
    return res;
  }

  if (opcode == Decoder::BitfieldOpcode::kSbfm) {
    // ASR: SBFM Rd, Rn, #shift, #(regsize-1).
    if (imms == reg_size - 1) {
      if (is_64bit) {
        Register res = Copy(src);
        if (immr != 0) {
          res = std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
        }
        return res;
      }
      // 32-bit ASR: Sarl writes the low 32 and zero-extends to 64, matching
      // ARM64 W-write semantics. Do NOT sign-extend further.
      Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      if (immr != 0) {
        res = std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
      }
      return res;
    }
    // SXTB: SBFM Xd/Wd, Wn, #0, #7.
    if (immr == 0 && imms == 7) {
      if (is_64bit) {
        return std::get<0>(Gen<x86_64::MovsxbqRegReg>(src));
      }
      return std::get<0>(Gen<x86_64::MovsxblRegReg>(src));
    }
    // SXTH: SBFM Xd/Wd, Wn, #0, #15.
    if (immr == 0 && imms == 15) {
      if (is_64bit) {
        return std::get<0>(Gen<x86_64::MovsxwqRegReg>(src));
      }
      return std::get<0>(Gen<x86_64::MovsxwlRegReg>(src));
    }
    // SXTW: SBFM Xd, Wn, #0, #31.
    if (is_64bit && immr == 0 && imms == 31) {
      return std::get<0>(Gen<x86_64::MovsxlqRegReg>(src));
    }
    // General SBFX (imms >= immr): sign-extend the field src[imms:immr] down to
    // bit 0. Two shifts: LSL by (reg_size-1-imms) lands bit imms at the MSB,
    // then ASR by (reg_size-1-imms+immr) shifts the field back to bit 0 while
    // sign-extending from the field's top bit. For 32-bit the Sar writes the
    // low 32 and zero-extends to 64, matching ARM64 W-write semantics. The
    // imms < immr case (SBFIZ) is handled separately below.
    if (imms >= immr) {
      const uint8_t left = static_cast<uint8_t>(reg_size - 1 - imms);
      const uint8_t right = static_cast<uint8_t>(left + immr);
      if (is_64bit) {
        Register res = Copy(src);
        if (left != 0) {
          res = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(left)));
        }
        if (right != 0) {
          res = std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(res, static_cast<int8_t>(right)));
        }
        return res;
      }
      Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      if (left != 0) {
        res = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(left)));
      }
      if (right != 0) {
        res = std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(res, static_cast<int8_t>(right)));
      }
      return res;
    }
    // SBFIZ (imms < immr): sign-extend the low (imms+1) bits of src, then shift
    // left by lsb = reg_size - immr. Two shifts mirror the general-SBFX idiom:
    // LSL by (reg_size-1-imms) lands the field's top bit at the MSB, then ASR
    // by (immr-1-imms) sign-extends and lands bit 0 at position lsb. Since
    // imms < immr, right = immr-1-imms >= 0. For 32-bit the Sar writes the low
    // 32 and zero-extends to 64 (ARM64 W-write semantics).
    const uint8_t sbfiz_left = static_cast<uint8_t>(reg_size - 1 - imms);
    const uint8_t sbfiz_right = static_cast<uint8_t>(immr - 1 - imms);
    if (is_64bit) {
      Register res = Copy(src);
      if (sbfiz_left != 0) {
        res = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(sbfiz_left)));
      }
      if (sbfiz_right != 0) {
        res = std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(res, static_cast<int8_t>(sbfiz_right)));
      }
      return res;
    }
    Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
    if (sbfiz_left != 0) {
      res = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(sbfiz_left)));
    }
    if (sbfiz_right != 0) {
      res = std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(res, static_cast<int8_t>(sbfiz_right)));
    }
    return res;
  }

  // General BFM (BFI / BFXIL / BFC).
  //   result = (dst_val & ~mask) | (shifted_src & mask)
  if (opcode == Decoder::BitfieldOpcode::kBfm) {
    uint64_t mask;
    Register res;
    if (imms >= immr) {
      // BFXIL: extract width = imms-immr+1 bits, deposit at bit 0.
      unsigned width = imms - immr + 1;
      mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
      if (is_64bit) {
        res = Copy(src);
        if (immr != 0) {
          res = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
        }
        Register mask_reg = GetImm(mask);
        res = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, mask_reg));
      } else {
        res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        if (immr != 0) {
          res = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
        }
        res = std::get<0>(
            Gen<x86_64::AndlRegImm, kNoSSA>(res, static_cast<int32_t>(mask & 0xFFFFFFFFULL)));
      }
    } else {
      // BFI / BFC: extract low width = imms+1 bits, deposit at pos.
      unsigned width = imms + 1;
      unsigned pos = reg_size - immr;
      uint64_t field_mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
      mask = (pos >= 64) ? 0 : (field_mask << pos);
      if (is_64bit) {
        res = Copy(src);
        Register fmask_reg = GetImm(field_mask);
        res = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, fmask_reg));
        if (pos != 0) {
          res = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(pos)));
        }
      } else {
        res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        res = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(
            res, static_cast<int32_t>(field_mask & 0xFFFFFFFFULL)));
        if (pos != 0) {
          res = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(pos)));
        }
      }
    }
    // Merge: res = res | (dst_val & ~mask). Copy dst_val into a fresh temp.
    if (is_64bit) {
      Register keep = Copy(dst_val);
      Register notmask_reg = GetImm(~mask);
      keep = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(keep, notmask_reg));
      return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(res, keep));
    }
    Register keep = std::get<0>(Gen<x86_64::MovlRegReg>(dst_val));
    keep = std::get<0>(
        Gen<x86_64::AndlRegImm, kNoSSA>(keep, static_cast<int32_t>((~mask) & 0xFFFFFFFFULL)));
    return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(res, keep));
  }

  UndefinedReturningReg();
  return AllocTempReg();
}

// Integer load with a base+imm offset. Mirrors lite_translator.h::Load: ApplyTbi
// masks the top byte of the address (ARM64 TBI), then a size/sign-appropriate
// host load reads from {masked, offset}. A <32-bit unsigned load zero-extends to
// 32 (a 32-bit dest reg clears the upper 32); a signed load sign-extends to 32 or
// 64 per is_64bit_target; a 32-bit unsigned load zero-extends to 64 (Movl), and
// LDRSW (signed 32->64) uses Movsxlq. GenRecoveryBlockForLastInsn() exits the
// region if the host access faults, so the guest signal is delivered.
Register HeavyOptimizerFrontend::Load(Decoder::LoadStoreSize size,
                                      bool is_signed,
                                      bool is_64bit_target,
                                      Register base,
                                      int32_t offset) {
  if (!success()) {
    return AllocTempReg();
  }
  Register masked = ApplyTbi(base);
  Register res;
  switch (size) {
    case Decoder::LoadStoreSize::k64bit:
      res = std::get<0>(Gen<x86_64::MovqRegOp>({.base = masked, .disp = offset}));
      break;
    case Decoder::LoadStoreSize::k32bit:
      if (is_signed && is_64bit_target) {
        // LDRSW: 32 -> sign-extend to 64.
        res = std::get<0>(Gen<x86_64::MovsxlqRegOp>({.base = masked, .disp = offset}));
      } else {
        // 32-bit unsigned: zero-extends to 64.
        res = std::get<0>(Gen<x86_64::MovlRegOp>({.base = masked, .disp = offset}));
      }
      break;
    case Decoder::LoadStoreSize::k16bit:
      if (is_signed) {
        if (is_64bit_target) {
          res = std::get<0>(Gen<x86_64::MovsxwqRegOp>({.base = masked, .disp = offset}));
        } else {
          res = std::get<0>(Gen<x86_64::MovsxwlRegOp>({.base = masked, .disp = offset}));
        }
      } else {
        res = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = masked, .disp = offset}));
      }
      break;
    case Decoder::LoadStoreSize::k8bit:
      if (is_signed) {
        if (is_64bit_target) {
          res = std::get<0>(Gen<x86_64::MovsxbqRegOp>({.base = masked, .disp = offset}));
        } else {
          res = std::get<0>(Gen<x86_64::MovsxblRegOp>({.base = masked, .disp = offset}));
        }
      } else {
        res = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = masked, .disp = offset}));
      }
      break;
    default:
      UndefinedReturningReg();
      return AllocTempReg();
  }
  GenRecoveryBlockForLastInsn();
  return res;
}

// Integer store of the low `size` bytes of `data` to {ApplyTbi(base), offset}.
// Mirrors lite_translator.h::Store. GenRecoveryBlockForLastInsn() exits the
// region on a host fault so the guest signal handler runs.
void HeavyOptimizerFrontend::Store(Decoder::LoadStoreSize size, Register base, int32_t offset, Register data) {
  if (!success()) {
    return;
  }
  Register masked = ApplyTbi(base);
  switch (size) {
    case Decoder::LoadStoreSize::k64bit:
      Gen<x86_64::MovqOpReg>({.base = masked, .disp = offset}, data);
      break;
    case Decoder::LoadStoreSize::k32bit:
      Gen<x86_64::MovlOpReg>({.base = masked, .disp = offset}, data);
      break;
    case Decoder::LoadStoreSize::k16bit:
      Gen<x86_64::MovwOpReg>({.base = masked, .disp = offset}, data);
      break;
    case Decoder::LoadStoreSize::k8bit:
      Gen<x86_64::MovbOpReg>({.base = masked, .disp = offset}, data);
      break;
    default:
      UndefinedReturningVoid();
      return;
  }
  GenRecoveryBlockForLastInsn();
}

// Plain 64-bit add of an immediate (used for address computation). Always
// non-flag-setting. Mirrors lite_translator.h::AddImm.
Register HeavyOptimizerFrontend::AddImm(Register base, int32_t offset) {
  // A prior callback (e.g. a post-index Load) may have already bailed; emit
  // nothing so we don't append IR after the region-exit terminator.
  if (!success()) {
    return AllocTempReg();
  }
  Register res = Copy(base);
  if (offset != 0) {
    res = std::get<0>(Gen<x86_64::AddqRegImm, kNoSSA>(res, offset));
  }
  return res;
}

// LDP/LDPSW: load two adjacent sized elements at {base, 0} and {base, scale}.
// The SemanticsPlayer passes rt1/rt2 as register NUMBERS and applies any
// pre/post-index base writeback itself; this method commits the two loaded
// values to the destination registers. Mirrors lite_translator.h::LoadPair:
// both halves are loaded into temps FIRST, then committed via SetReg, so when
// a destination aliases the base register (e.g. `ldp x0, x8, [x0]`) the second
// load still reads from the original base. LDPSW (is_signed) sign-extends each
// 32-bit element into its 64-bit target.
void HeavyOptimizerFrontend::LoadPair(Decoder::LoadStoreSize size,
                                      Register base,
                                      int32_t offset,
                                      uint8_t rt1,
                                      uint8_t rt2,
                                      uint8_t scale,
                                      bool is_signed) {
  if (!success()) {
    return;
  }
  UNUSED_ARGS(offset);  // Already applied by the SemanticsPlayer.
  bool is_64bit_target = (size == Decoder::LoadStoreSize::k64bit) || is_signed;
  Register val1 = Load(size, is_signed, is_64bit_target, base, 0);
  if (!success()) {
    return;
  }
  Register val2 = Load(size, is_signed, is_64bit_target, base, static_cast<int32_t>(scale));
  if (!success()) {
    return;
  }
  if (rt1 != 31) {
    SetReg(rt1, val1);
  }
  if (rt2 != 31) {
    SetReg(rt2, val2);
  }
}

// STP: store two adjacent sized elements at {base, 0} and {base, scale}.
// The SemanticsPlayer applies any pre/post-index base writeback itself.
// Mirrors lite_translator.h::StorePair.
void HeavyOptimizerFrontend::StorePair(Decoder::LoadStoreSize size,
                                       Register base,
                                       int32_t offset,
                                       Register data1,
                                       Register data2,
                                       uint8_t scale) {
  if (!success()) {
    return;
  }
  UNUSED_ARGS(offset);  // Already applied by the SemanticsPlayer.
  Store(size, base, 0, data1);
  if (!success()) {
    return;
  }
  Store(size, base, static_cast<int32_t>(scale), data2);
}

// LDR (register offset): address = base + extend(offset_reg) << shift_amount,
// then the same TBI + sized/sign-appropriate access + recovery as Load().
// Mirrors lite_translator.h::LoadReg (+ ApplyOffsetExtend). The decoder only
// emits the word-or-larger options (010=UXTW, 011=LSL/UXTX, 110=SXTW,
// 111=SXTX); any other option was already rejected as UNDEFINED.
Register HeavyOptimizerFrontend::LoadReg(Decoder::LoadStoreSize size,
                                         bool is_signed,
                                         bool is_64bit_target,
                                         Register base,
                                         Register offset_reg,
                                         uint8_t extend_type,
                                         uint8_t shift_amount) {
  if (!success()) {
    return AllocTempReg();
  }
  Register addr = EmitRegOffsetAddr(base, offset_reg, extend_type, shift_amount);
  return Load(size, is_signed, is_64bit_target, addr, 0);
}

// STR (register offset): mirrors lite_translator.h::StoreReg (+
// ApplyOffsetExtend).
void HeavyOptimizerFrontend::StoreReg(Decoder::LoadStoreSize size,
                                      Register base,
                                      Register offset_reg,
                                      uint8_t extend_type,
                                      uint8_t shift_amount,
                                      Register data) {
  if (!success()) {
    return;
  }
  Register addr = EmitRegOffsetAddr(base, offset_reg, extend_type, shift_amount);
  Store(size, addr, 0, data);
}

void HeavyOptimizerFrontend::Svc(uint16_t imm) {
  UndefinedReturningVoid();
  UNUSED_ARGS(imm);
}

void HeavyOptimizerFrontend::Brk(uint16_t imm) {
  UndefinedReturningVoid();
  UNUSED_ARGS(imm);
}

// MRS Xt, TPIDR_EL0: read the guest thread-local-storage pointer from
// ThreadState.tls. This is the read the bionic stack-canary prologue and all
// TLS accesses issue, so it appears in nearly every real function; without it
// the heavy tier bails on almost all real-app code. Other system registers
// (NZCV, the CPU-detect MIDR/ID regs, FPSR) bail to the lite tier, which
// handles them. Mirrors lite_translator.h::Mrs for the TPIDR_EL0 case.
Register HeavyOptimizerFrontend::Mrs(Decoder::SystemReg sysreg) {
  if (sysreg == Decoder::SystemReg::kTpidrEl0) {
    Register res = AllocTempReg();
    if (success()) {
      builder_.GenGet(res, static_cast<int32_t>(offsetof(ThreadState, tls)));
    }
    return res;
  }
  UndefinedReturningReg();
  UNUSED_ARGS(sysreg);
  return AllocTempReg();
}

void HeavyOptimizerFrontend::Msr(Decoder::SystemReg sysreg, Register src) {
  UndefinedReturningVoid();
  UNUSED_ARGS(sysreg, src);
}

void HeavyOptimizerFrontend::IcIvau(uint8_t rt) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rt);
}

// AND/ORR/EOR/BIC/ORN/EON (shifted register), including ANDS/BICS/TST. `invert`
// selects the BIC/ORN/EON variants (src2 is bitwise-inverted before the op).
// x86 AND clears CF and OF, so ANDS materializes NZCV with C=0 and V=0.
// Mirrors lite_translator.h::LogicalShiftedReg.
Register HeavyOptimizerFrontend::LogicalShiftedReg(Decoder::LogicalShiftedRegOpcode opcode,
                                                   bool is_64bit,
                                                   bool invert,
                                                   Register src1,
                                                   Register src2,
                                                   Decoder::ShiftType shift_type,
                                                   uint8_t shift_amount) {
  if (!success()) {
    return AllocTempReg();
  }
  Register op2 = EmitShiftImm(src2, shift_type, shift_amount, is_64bit);
  if (invert) {
    // NotqReg is 64-bit only; for the 32-bit case the upper half is re-cleared
    // by the subsequent 32-bit op, so a 64-bit NOT is safe.
    op2 = std::get<0>(Gen<x86_64::NotqReg, kNoSSA>(op2));
  }
  if (is_64bit) {
    Register res = Copy(src1);
    switch (opcode) {
      case Decoder::LogicalShiftedRegOpcode::kAnd:
        return std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, op2));
      case Decoder::LogicalShiftedRegOpcode::kAnds: {
        // ANDS/BICS/TST: x86 AND clears CF and OF, so ARM64 C=0 and V=0.
        auto [r, flags] = Gen<x86_64::AndqRegReg, kNoSSA>(res, op2);
        EmitMaterializeNZCV(flags, /*is_sub=*/false);
        return r;
      }
      case Decoder::LogicalShiftedRegOpcode::kOrr:
        return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(res, op2));
      case Decoder::LogicalShiftedRegOpcode::kEor:
        return std::get<0>(Gen<x86_64::XorqRegReg, kNoSSA>(res, op2));
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }
  Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
  switch (opcode) {
    case Decoder::LogicalShiftedRegOpcode::kAnd:
      return std::get<0>(Gen<x86_64::AndlRegReg, kNoSSA>(res, op2));
    case Decoder::LogicalShiftedRegOpcode::kAnds: {
      auto [r, flags] = Gen<x86_64::AndlRegReg, kNoSSA>(res, op2);
      EmitMaterializeNZCV(flags, /*is_sub=*/false);
      return r;
    }
    case Decoder::LogicalShiftedRegOpcode::kOrr:
      return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(res, op2));
    case Decoder::LogicalShiftedRegOpcode::kEor:
      return std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(res, op2));
    default:
      UndefinedReturningReg();
      return AllocTempReg();
  }
}

// ADD/SUB (shifted register), including the flag-setting ADDS/SUBS/CMP/CMN
// forms (NZCV materialized via EmitMaterializeNZCV). ROR is not a valid shift
// for add/sub and bails. Mirrors lite_translator.h::AddSubShiftedReg.
Register HeavyOptimizerFrontend::AddSubShiftedReg(bool is_sub,
                                                  bool set_flags,
                                                  bool is_64bit,
                                                  Register src1,
                                                  Register src2,
                                                  Decoder::ShiftType shift_type,
                                                  uint8_t shift_amount) {
  if (!success()) {
    return AllocTempReg();
  }
  // Validate first; emit nothing on bail. ROR is not a valid shift for ADD/SUB.
  if (shift_type == Decoder::ShiftType::kRor) {
    UndefinedReturningReg();
    return AllocTempReg();
  }
  Register op2 = EmitShiftImm(src2, shift_type, shift_amount, is_64bit);
  if (is_64bit) {
    Register res = Copy(src1);
    if (is_sub) {
      auto [r, flags] = Gen<x86_64::SubqRegReg, kNoSSA>(res, op2);
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/true);
      }
      return r;
    }
    auto [r, flags] = Gen<x86_64::AddqRegReg, kNoSSA>(res, op2);
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/false);
    }
    return r;
  }
  Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
  if (is_sub) {
    auto [r, flags] = Gen<x86_64::SublRegReg, kNoSSA>(res, op2);
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/true);
    }
    return r;
  }
  auto [r, flags] = Gen<x86_64::AddlRegReg, kNoSSA>(res, op2);
  if (set_flags) {
    EmitMaterializeNZCV(flags, /*is_sub=*/false);
  }
  return r;
}

// ADD/SUB (extended register), including the flag-setting ADDS/SUBS/CMP/CMN
// forms (NZCV materialized via EmitMaterializeNZCV). Mirrors
// lite_translator.h::AddSubExtendedReg.
Register HeavyOptimizerFrontend::AddSubExtendedReg(bool is_sub,
                                                   bool set_flags,
                                                   bool is_64bit,
                                                   Register src1,
                                                   Register src2,
                                                   uint8_t extend_type,
                                                   uint8_t shift_amount) {
  if (!success()) {
    return AllocTempReg();
  }
  // Validate first; emit nothing on bail.
  if (shift_amount > 4 || extend_type > 0b111) {
    UndefinedReturningReg();
    return AllocTempReg();
  }

  // Apply extension to src2.
  // extend_type: 000=UXTB 001=UXTH 010=UXTW 011=UXTX 100=SXTB 101=SXTH
  //              110=SXTW 111=SXTX.
  Register ext;
  switch (extend_type) {
    case 0b000:  // UXTB
      ext = std::get<0>(Gen<x86_64::MovzxblRegReg>(src2));
      break;
    case 0b001:  // UXTH
      ext = std::get<0>(Gen<x86_64::MovzxwlRegReg>(src2));
      break;
    case 0b010:  // UXTW: 32-bit mov zero-extends to 64.
      ext = std::get<0>(Gen<x86_64::MovlRegReg>(src2));
      break;
    case 0b011:  // UXTX: no extension.
      ext = Copy(src2);
      break;
    case 0b100:  // SXTB
      if (is_64bit) {
        ext = std::get<0>(Gen<x86_64::MovsxbqRegReg>(src2));
      } else {
        ext = std::get<0>(Gen<x86_64::MovsxblRegReg>(src2));
      }
      break;
    case 0b101:  // SXTH
      if (is_64bit) {
        ext = std::get<0>(Gen<x86_64::MovsxwqRegReg>(src2));
      } else {
        ext = std::get<0>(Gen<x86_64::MovsxwlRegReg>(src2));
      }
      break;
    case 0b110:  // SXTW
      if (is_64bit) {
        ext = std::get<0>(Gen<x86_64::MovsxlqRegReg>(src2));
      } else {
        ext = std::get<0>(Gen<x86_64::MovlRegReg>(src2));
      }
      break;
    default:  // 0b111 SXTX: no extension.
      ext = Copy(src2);
      break;
  }

  // Apply shift.
  if (shift_amount > 0) {
    if (is_64bit) {
      ext = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(ext, static_cast<int8_t>(shift_amount)));
    } else {
      ext = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(ext, static_cast<int8_t>(shift_amount)));
    }
  }

  if (is_64bit) {
    Register res = Copy(src1);
    if (is_sub) {
      auto [r, flags] = Gen<x86_64::SubqRegReg, kNoSSA>(res, ext);
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/true);
      }
      return r;
    }
    auto [r, flags] = Gen<x86_64::AddqRegReg, kNoSSA>(res, ext);
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/false);
    }
    return r;
  }
  Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
  if (is_sub) {
    auto [r, flags] = Gen<x86_64::SublRegReg, kNoSSA>(res, ext);
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/true);
    }
    return r;
  }
  auto [r, flags] = Gen<x86_64::AddlRegReg, kNoSSA>(res, ext);
  if (set_flags) {
    EmitMaterializeNZCV(flags, /*is_sub=*/false);
  }
  return r;
}

// LSLV/LSRV/ASRV/RORV (variable shifts) and UDIV/SDIV. CRC32/PACGA still bail.
// Division routes to the .cc EmitUDiv/EmitSDiv helpers because the ARM
// divide-by-zero (Rd=0) and SDIV INT_MIN/-1 (Rd=INT_MIN) guards need basic
// blocks around the fixed-RDX:RAX x86 DIV/IDIV pseudo-op. The variable shifts
// use the x86 shift-by-CL forms; the backend register allocator binds the
// count operand to RCX automatically.
Register HeavyOptimizerFrontend::DataProc2Src(Decoder::DataProc2SrcOpcode opcode,
                                              bool is_64bit,
                                              Register src1,
                                              Register src2) {
  if (!success()) {
    return AllocTempReg();
  }
  switch (opcode) {
    case Decoder::DataProc2SrcOpcode::kLslv:
      if (is_64bit) {
        return std::get<0>(Gen<x86_64::ShlqRegReg>(src1, src2));
      }
      return std::get<0>(Gen<x86_64::ShllRegReg>(src1, src2));
    case Decoder::DataProc2SrcOpcode::kLsrv:
      if (is_64bit) {
        return std::get<0>(Gen<x86_64::ShrqRegReg>(src1, src2));
      }
      return std::get<0>(Gen<x86_64::ShrlRegReg>(src1, src2));
    case Decoder::DataProc2SrcOpcode::kAsrv:
      if (is_64bit) {
        return std::get<0>(Gen<x86_64::SarqRegReg>(src1, src2));
      }
      return std::get<0>(Gen<x86_64::SarlRegReg>(src1, src2));
    case Decoder::DataProc2SrcOpcode::kRorv:
      if (is_64bit) {
        return std::get<0>(Gen<x86_64::RorqRegReg>(src1, src2));
      }
      return std::get<0>(Gen<x86_64::RorlRegReg>(src1, src2));
    case Decoder::DataProc2SrcOpcode::kUdiv:
      return EmitUDiv(is_64bit, src1, src2);
    case Decoder::DataProc2SrcOpcode::kSdiv:
      return EmitSDiv(is_64bit, src1, src2);
    // CRC32C* (Castagnoli) via the host SSE4.2 CRC32 instruction, which uses
    // the same polynomial as ARM's CRC32C ops. The accumulator is Wn
    // (zero-extended) and the data is Wm (b/h/w) or Xm (x). Mirrors
    // lite_translator_integer_ctrl.inc. The IEEE CRC32* ops use a different
    // polynomial and stay on the interpreter (they fall through to the default
    // bail below). Bail to lite if the host lacks SSE4.2.
    case Decoder::DataProc2SrcOpcode::kCrc32cb:
    case Decoder::DataProc2SrcOpcode::kCrc32ch:
    case Decoder::DataProc2SrcOpcode::kCrc32cw:
    case Decoder::DataProc2SrcOpcode::kCrc32cx: {
      if (!host_platform::kHasSSE4_2) {
        UndefinedReturningReg();
        return AllocTempReg();
      }
      // Wn accumulator, zero-extended into a fresh 32-bit vreg (upper bits
      // cleared so the CRC32 result also zero-extends into Xd).
      Register acc = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
      switch (opcode) {
        case Decoder::DataProc2SrcOpcode::kCrc32cb:
          return std::get<0>(Gen<x86_64::Crc32cbRegReg, kNoSSA>(acc, src2));
        case Decoder::DataProc2SrcOpcode::kCrc32ch:
          return std::get<0>(Gen<x86_64::Crc32chRegReg, kNoSSA>(acc, src2));
        case Decoder::DataProc2SrcOpcode::kCrc32cw:
          return std::get<0>(Gen<x86_64::Crc32cwRegReg, kNoSSA>(acc, src2));
        default:
          return std::get<0>(Gen<x86_64::Crc32cxRegReg, kNoSSA>(acc, src2));
      }
    }
    // IEEE CRC32B/H/W/X — poly 0x04C11DB7 (reflected 0xEDB88320). The host SSE4.2
    // CRC32 computes the *Castagnoli* poly (CRC32C* above) and cannot serve
    // these; lower them with a carry-less multiply (PCLMULQDQ) reflected Barrett
    // reduction. Accumulator is Wn (low 32); data is Wm (b/h/w) or Xm (x). Line-
    // by-line mirror of lite_translator_integer_ctrl.inc. Reflected fold/Barrett
    // constants (rev33 of x^k mod P, of P, of mu=floor(x^64/P); == zlib's):
    //   K64=0x163CD6124 K32=0x1DB710640 K16=0x10000 K8=0x1000000
    //   MU=0x1F7011641  POLYR=0x1DB710641
    // Bail to lite if the host lacks PCLMULQDQ.
    case Decoder::DataProc2SrcOpcode::kCrc32b:
    case Decoder::DataProc2SrcOpcode::kCrc32h:
    case Decoder::DataProc2SrcOpcode::kCrc32w:
    case Decoder::DataProc2SrcOpcode::kCrc32x: {
      if (!host_platform::kHasCLMUL) {
        UndefinedReturningReg();
        return AllocTempReg();
      }
      FpRegister xt = AllocTempSimdReg();  // fold accumulator (proc)
      FpRegister xk = AllocTempSimdReg();  // constant loader
      Register acc = std::get<0>(Gen<x86_64::MovlRegReg>(src1));  // zext Wn
      if (opcode == Decoder::DataProc2SrcOpcode::kCrc32x) {
        Register t = std::get<0>(Gen<x86_64::XorqRegReg>(acc, src2));  // T=acc^Xm
        Register tlo = std::get<0>(Gen<x86_64::MovlRegReg>(t));
        Register thi = std::get<0>(Gen<x86_64::ShrqRegImm>(t, int8_t{32}));
        FpRegister xh = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdXRegReg>(xt.machine_reg(), tlo);
        builder_.Gen<x86_64::MovdXRegReg>(xh.machine_reg(), thi);
        builder_.Gen<x86_64::MovqXRegReg>(xk.machine_reg(), GetImm(uint64_t{0x163CD6124}));
        builder_.Gen<x86_64::PclmulqdqXRegXRegImm>(xt.machine_reg(), xk.machine_reg(), int8_t{0x00});
        builder_.Gen<x86_64::MovqXRegReg>(xk.machine_reg(), GetImm(uint64_t{0x1DB710640}));
        builder_.Gen<x86_64::PclmulqdqXRegXRegImm>(xh.machine_reg(), xk.machine_reg(), int8_t{0x00});
        builder_.Gen<x86_64::PxorXRegXReg>(xt.machine_reg(), xh.machine_reg());
      } else {
        Register masked = std::get<0>(Gen<x86_64::MovlRegReg>(src2));  // zext Wm
        uint64_t kc;
        switch (opcode) {
          case Decoder::DataProc2SrcOpcode::kCrc32h:
            masked = std::get<0>(Gen<x86_64::AndqRegImm>(masked, int32_t{0xFFFF}));
            kc = 0x10000; break;
          case Decoder::DataProc2SrcOpcode::kCrc32b:
            masked = std::get<0>(Gen<x86_64::AndqRegImm>(masked, int32_t{0xFF}));
            kc = 0x1000000; break;
          default: /* kCrc32w */
            kc = 0x1DB710640; break;
        }
        Register t = std::get<0>(Gen<x86_64::XorqRegReg>(acc, masked));  // T
        builder_.Gen<x86_64::MovdXRegReg>(xt.machine_reg(), t);
        builder_.Gen<x86_64::MovqXRegReg>(xk.machine_reg(), GetImm(kc));
        builder_.Gen<x86_64::PclmulqdqXRegXRegImm>(xt.machine_reg(), xk.machine_reg(), int8_t{0x00});
      }
      // Reflected Barrett reduction of the 64-bit fold value (xt.q0) -> 32-bit.
      // 32-bit MOVD round-trips realize the R&M32 / t1&M32 masks. A fresh xb
      // holds the Barrett running value so xt is only read, never rewritten.
      FpRegister xb = AllocTempSimdReg();
      Register r = std::get<0>(Gen<x86_64::MovqRegXReg>(xt.machine_reg()));  // R=proc
      Register rlo = std::get<0>(Gen<x86_64::MovlRegReg>(r));                // R&M32
      builder_.Gen<x86_64::MovdXRegReg>(xb.machine_reg(), rlo);
      builder_.Gen<x86_64::MovqXRegReg>(xk.machine_reg(), GetImm(uint64_t{0x1F7011641}));  // MU
      builder_.Gen<x86_64::PclmulqdqXRegXRegImm>(xb.machine_reg(), xk.machine_reg(), int8_t{0x00});
      Register t1 = std::get<0>(Gen<x86_64::MovdRegXReg>(xb.machine_reg()));  // t1&M32
      builder_.Gen<x86_64::MovdXRegReg>(xb.machine_reg(), t1);
      builder_.Gen<x86_64::MovqXRegReg>(xk.machine_reg(), GetImm(uint64_t{0x1DB710641}));  // POLYR
      builder_.Gen<x86_64::PclmulqdqXRegXRegImm>(xb.machine_reg(), xk.machine_reg(), int8_t{0x00});
      Register t2 = std::get<0>(Gen<x86_64::MovqRegXReg>(xb.machine_reg()));  // t2 (64-bit)
      Register xored = std::get<0>(Gen<x86_64::XorqRegReg>(r, t2));
      Register crc = std::get<0>(Gen<x86_64::ShrqRegImm>(xored, int8_t{32}));
      return std::get<0>(Gen<x86_64::MovlRegReg>(crc));  // Wd (zero-extends into Xd)
    }
    default:
      UndefinedReturningReg();
      return AllocTempReg();
  }
}

// MADD/MSUB and the signed/unsigned widening multiply-accumulates
// (SMADDL/SMSUBL/UMADDL/UMSUBL), plus SMULH/UMULH (high 64 bits of a 64x64
// product via the widening x86 IMUL/MUL into RDX:RAX). Mirrors
// lite_translator.h::DataProc3Src.
//   MADD:  Rd = Ra + Rn * Rm     MSUB:  Rd = Ra - Rn * Rm
//   SMADDL: Xd = Xa + sext(Wn)*sext(Wm)   (UMADDL uses zext; *SUBL subtracts)
Register HeavyOptimizerFrontend::DataProc3Src(Decoder::DataProc3SrcOpcode opcode,
                                              bool is_64bit,
                                              Register src1,
                                              Register src2,
                                              Register src3) {
  if (!success()) {
    return AllocTempReg();
  }
  switch (opcode) {
    case Decoder::DataProc3SrcOpcode::kMadd: {
      if (is_64bit) {
        Register prod = std::get<0>(Gen<x86_64::ImulqRegReg>(src1, src2));
        return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(prod, src3));
      }
      Register prod = std::get<0>(Gen<x86_64::ImullRegReg>(src1, src2));
      return std::get<0>(Gen<x86_64::AddlRegReg, kNoSSA>(prod, src3));
    }
    case Decoder::DataProc3SrcOpcode::kMsub: {
      if (is_64bit) {
        Register prod = std::get<0>(Gen<x86_64::ImulqRegReg>(src1, src2));
        Register res = Copy(src3);
        return std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(res, prod));
      }
      Register prod = std::get<0>(Gen<x86_64::ImullRegReg>(src1, src2));
      Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src3));
      return std::get<0>(Gen<x86_64::SublRegReg, kNoSSA>(res, prod));
    }
    case Decoder::DataProc3SrcOpcode::kSmaddl:
    case Decoder::DataProc3SrcOpcode::kSmsubl: {
      // Sign-extend both 32-bit sources to 64-bit, then 64-bit multiply.
      Register ext1 = std::get<0>(Gen<x86_64::MovsxlqRegReg>(src1));
      Register ext2 = std::get<0>(Gen<x86_64::MovsxlqRegReg>(src2));
      Register prod = std::get<0>(Gen<x86_64::ImulqRegReg, kNoSSA>(ext1, ext2));
      Register res = Copy(src3);
      if (opcode == Decoder::DataProc3SrcOpcode::kSmaddl) {
        return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(res, prod));
      }
      return std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(res, prod));
    }
    case Decoder::DataProc3SrcOpcode::kUmaddl:
    case Decoder::DataProc3SrcOpcode::kUmsubl: {
      // Zero-extend both 32-bit sources to 64-bit (Movl), then 64-bit multiply.
      Register ext1 = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
      Register ext2 = std::get<0>(Gen<x86_64::MovlRegReg>(src2));
      Register prod = std::get<0>(Gen<x86_64::ImulqRegReg, kNoSSA>(ext1, ext2));
      Register res = Copy(src3);
      if (opcode == Decoder::DataProc3SrcOpcode::kUmaddl) {
        return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(res, prod));
      }
      return std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(res, prod));
    }
    case Decoder::DataProc3SrcOpcode::kUmulh:
      // UMULH: Xd = high 64 bits of (Xn * Xm), unsigned. MulqRegRegReg returns
      // [low(RAX), high(RDX), flags]; the high half is the result. X-form only.
      return std::get<1>(Gen<x86_64::MulqRegRegReg>(src1, src2));
    case Decoder::DataProc3SrcOpcode::kSmulh:
      // SMULH: Xd = high 64 bits of (Xn * Xm), signed (ImulqRegRegReg). X-form only.
      return std::get<1>(Gen<x86_64::ImulqRegRegReg>(src1, src2));
    default:
      UndefinedReturningReg();
      return AllocTempReg();
  }
}

// ADC/ADCS/SBC/SBCS (add/subtract with the guest carry flag). Mirrors
// lite_translator.h::AddSubWithCarry: the guest carry (ARM NZCV C =
// cpu.flags bit 8) is loaded into x86 CF, then one x86 ADC/SBB does the
// carry op.
//   ADC — ARM C maps directly to x86 CF: BT bit 8 sets CF, then ADC.
//   SBC — x86 SBB computes src1 - src2 - CF = src1 + ~src2 + (1 - CF), while
//         ARM SBC wants src1 + ~src2 + C, so SBB needs CF = !C (CMC after the
//         BT). On output x86 SBB's CF is the borrow; EmitMaterializeNZCV
//         (is_sub=true) inverts it back to ARM's "carry = no borrow".
// res is a fresh temp; the whole MOV/BT/CMC/ADC(SBB) chain stays in one basic
// block so the host FLAGS never cross a BB boundary. MOV does not touch
// FLAGS, so the carry loaded by BT survives to the ADC/SBB.
Register HeavyOptimizerFrontend::AddSubWithCarry(Register src1, Register src2, bool is_64bit, bool is_sub, bool set_flags) {
  if (!success()) {
    return AllocTempReg();
  }
  // res = src1 (32-bit MOV for sf=0 so the later 32-bit ADC/SBB zero-extends).
  Register res = is_64bit ? Copy(src1) : std::get<0>(Gen<x86_64::MovlRegReg>(src1));
  // Load the guest carry into x86 CF. MovwRegOp matches the 16-bit flags-read
  // opcode RemoveLoopGuestContextAccesses recognizes; only bit 8 is tested, so
  // the untouched upper bits of the loaded word do not matter.
  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));
  Register armflags =
      std::get<0>(Gen<x86_64::MovwRegOp>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}));
  Gen<x86_64::BtqRegImm>(armflags, static_cast<int8_t>(8));  // CF = bit 8 = ARM carry.
  if (is_sub) {
    Gen<x86_64::Cmc>(GetFlagsRegister());  // SBB wants CF = !(ARM carry).
  }
  if (is_64bit) {
    if (is_sub) {
      auto [r, flags] = Gen<x86_64::SbbqRegReg, kNoSSA>(res, src2, GetFlagsRegister());
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/true);
      }
      return r;
    }
    auto [r, flags] = Gen<x86_64::AdcqRegReg, kNoSSA>(res, src2, GetFlagsRegister());
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/false);
    }
    return r;
  }
  if (is_sub) {
    auto [r, flags] = Gen<x86_64::SbblRegReg, kNoSSA>(res, src2, GetFlagsRegister());
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/true);
    }
    return r;
  }
  auto [r, flags] = Gen<x86_64::AdclRegReg, kNoSSA>(res, src2, GetFlagsRegister());
  if (set_flags) {
    EmitMaterializeNZCV(flags, /*is_sub=*/false);
  }
  return r;
}

// RBIT/REV16/REV32/REV/CLZ/CLS. REV16/REV32/REV are byte-reversed with SWAR
// shift/mask/or sequences (no x86 BSWAP MachineIR op). CLZ maps to LZCNT and
// CLS to LZCNT(x ^ (x>>1))-1 (both gated on host LZCNT). RBIT has no x86
// mapping and bails to the lite translator/interpreter. PAuth DP-1Src variants
// (opcode2 bit 0x40) are treated as identity because Digitalis is PAC-blind.
// Mirrors lite_translator.h::DataProc1Src.
Register HeavyOptimizerFrontend::DataProc1Src(Register src, uint8_t opcode2, bool is_64bit) {
  if (!success()) {
    return AllocTempReg();
  }
  // PAuth DP-1Src: identity copy (the upper-half clear of a 32-bit mov handles
  // the sf=0 zero-extend; PAuth ops are X-form, so the Movq branch is taken).
  if (opcode2 & 0x40) {
    if (is_64bit) {
      return Copy(src);
    }
    return std::get<0>(Gen<x86_64::MovlRegReg>(src));
  }
  switch (opcode2) {
    case 0b000001: {  // REV16: reverse byte order within each 16-bit halfword.
      // res = ((src & lo_mask) << 8) | ((src & hi_mask) >> 8).
      if (is_64bit) {
        Register lo = Copy(src);
        Register lo_mask = GetImm(0x00FF00FF00FF00FFULL);
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, lo_mask));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{8}));
        Register hi = Copy(src);
        Register hi_mask = GetImm(0xFF00FF00FF00FF00ULL);
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, hi_mask));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{8}));
        return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
      }
      Register lo = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(lo, 0x00FF00FF));
      lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{8}));
      Register hi = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, static_cast<int32_t>(0xFF00FF00)));
      hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(hi, int8_t{8}));
      return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
    }
    case 0b000010:  // REV32 (sf=1) / REV (sf=0): byte-reverse each 32-bit word.
      // No x86 BSWAP MachineIR op, so byte-swap with the SWAR shift/mask/or
      // sequence: swap bytes within each halfword, then swap halves within each
      // 32-bit word. For the X-form this byte-reverses each 32-bit word in place
      // (no cross-word swap); for the W-form it byte-reverses the 32-bit value
      // and zero-extends.
      if (is_64bit) {
        Register x = Copy(src);
        // Swap bytes within each 16-bit halfword.
        Register lo = Copy(x);
        Register m1 = GetImm(0x00FF00FF00FF00FFULL);
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, m1));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{8}));
        Register hi = Copy(x);
        Register m2 = GetImm(0xFF00FF00FF00FF00ULL);
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, m2));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{8}));
        x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        // Swap 16-bit halves within each 32-bit word.
        lo = Copy(x);
        Register m3 = GetImm(0x0000FFFF0000FFFFULL);
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, m3));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{16}));
        hi = Copy(x);
        Register m4 = GetImm(0xFFFF0000FFFF0000ULL);
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, m4));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{16}));
        return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
      } else {
        // REV Wd: byte-reverse the 32-bit word (zero-extends to 64).
        Register lo = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(lo, 0x00FF00FF));
        lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{8}));
        Register hi = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, static_cast<int32_t>(0xFF00FF00)));
        hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(hi, int8_t{8}));
        Register x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
        lo = Copy(x);
        lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{16}));
        hi = Copy(x);
        hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(hi, int8_t{16}));
        return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
      }
    case 0b000011:  // REV (64-bit full byte reverse).
      // SWAR byte-swap: halfword swap, then halfword-pair swap within words,
      // then 32-bit word swap. (No BSWAP MachineIR op.)
      if (is_64bit) {
        Register x = Copy(src);
        // Swap bytes within each 16-bit halfword.
        Register lo = Copy(x);
        Register m1 = GetImm(0x00FF00FF00FF00FFULL);
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, m1));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{8}));
        Register hi = Copy(x);
        Register m2 = GetImm(0xFF00FF00FF00FF00ULL);
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, m2));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{8}));
        x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        // Swap 16-bit halves within each 32-bit word.
        lo = Copy(x);
        Register m3 = GetImm(0x0000FFFF0000FFFFULL);
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, m3));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{16}));
        hi = Copy(x);
        Register m4 = GetImm(0xFFFF0000FFFF0000ULL);
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, m4));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{16}));
        x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        // Swap the two 32-bit words.
        lo = Copy(x);
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{32}));
        hi = Copy(x);
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{32}));
        return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
      }
      // REV is X-form only for opcode2=000011; sf=0 is not encoded. Bail safely.
      UndefinedReturningReg();
      return AllocTempReg();
    case 0b000100:  // CLZ (count leading zeros) maps directly to x86 LZCNT.
      // Without LZCNT the encoding decodes as BSR (wrong result for CLZ), so
      // bail to the lite translator (which uses a BSR-with-zero-check sequence).
      if (!host_platform::kHasLZCNT) {
        UndefinedReturningReg();
        return AllocTempReg();
      }
      if (is_64bit) {
        return std::get<0>(Gen<x86_64::LzcntqRegReg>(src));
      }
      return std::get<0>(Gen<x86_64::LzcntlRegReg>(src));
    case 0b000101:  // CLS (count leading sign bits).
      // CLS = LZCNT(x ^ (x >>arith 1)) - 1. LZCNT(0) == reg_size, so the
      // all-same-bits case yields reg_size-1 with no branch. Needs LZCNT for
      // the zero-input result; without it, bail to the lite/interp path.
      if (!host_platform::kHasLZCNT) {
        UndefinedReturningReg();
        return AllocTempReg();
      }
      if (is_64bit) {
        Register sar = Copy(src);
        sar = std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(sar, int8_t{1}));
        Register xored = Copy(src);
        xored = std::get<0>(Gen<x86_64::XorqRegReg, kNoSSA>(xored, sar));
        Register lz = std::get<0>(Gen<x86_64::LzcntqRegReg>(xored));
        return std::get<0>(Gen<x86_64::SubqRegImm, kNoSSA>(lz, int32_t{1}));
      } else {
        // 32-bit: Movl src into a clean 32-bit value first, Sarl/Lzcntl operate
        // over 32 bits and zero-extend the result to 64 (ARM64 W-write).
        Register clean = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        Register sar = Copy(clean);
        sar = std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(sar, int8_t{1}));
        Register xored = Copy(clean);
        xored = std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(xored, sar));
        Register lz = std::get<0>(Gen<x86_64::LzcntlRegReg>(xored));
        return std::get<0>(Gen<x86_64::SublRegImm, kNoSSA>(lz, int32_t{1}));
      }
    case 0b000000:  // RBIT: reverse bit order.
      // SWAR bit-reverse: swap adjacent bits, then bit-pairs, then nibbles
      // (each as ((x & m) << s) | ((x >> s) & m)), then reverse byte order
      // with the same shift/mask/or sequence as REV (no x86 BSWAP MachineIR op).
      if (is_64bit) {
        Register x = Copy(src);
        // Swap adjacent bits (mask 0x5555..., shift 1).
        Register lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x5555555555555555ULL)));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{1}));
        Register hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{1}));
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x5555555555555555ULL)));
        x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        // Swap bit-pairs (mask 0x3333..., shift 2).
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x3333333333333333ULL)));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{2}));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{2}));
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x3333333333333333ULL)));
        x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        // Swap nibbles (mask 0x0F0F..., shift 4).
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x0F0F0F0F0F0F0F0FULL)));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{4}));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{4}));
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x0F0F0F0F0F0F0F0FULL)));
        x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        // Reverse byte order: halfword-byte swap, halfword-pair swap, word swap.
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x00FF00FF00FF00FFULL)));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{8}));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{8}));
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x00FF00FF00FF00FFULL)));
        x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x0000FFFF0000FFFFULL)));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{16}));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{16}));
        hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x0000FFFF0000FFFFULL)));
        x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(Copy(x), int8_t{32}));
        hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{32}));
        return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
      } else {
        // RBIT Wd: reverse the low 32 bits, zero-extend.
        Register x = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        Register lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(x), 0x55555555));
        lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{1}));
        Register hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{1}));
        hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, 0x55555555));
        x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
        lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(x), 0x33333333));
        lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{2}));
        hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{2}));
        hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, 0x33333333));
        x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
        lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(x), 0x0F0F0F0F));
        lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{4}));
        hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{4}));
        hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, 0x0F0F0F0F));
        x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
        // Byte-reverse the 32-bit word: halfword-byte swap then halfword swap.
        lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(x), 0x00FF00FF));
        lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{8}));
        hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{8}));
        hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, 0x00FF00FF));
        x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
        lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(Copy(x), int8_t{16}));
        hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{16}));
        return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
      }
    default:
      // Any remaining unhandled DP-1Src opcode: fall back to the interpreter.
      UndefinedReturningReg();
      return AllocTempReg();
  }
}

// EXTR Rd, Rn, Rm, #lsb: Rd = (Rn:Rm) >> lsb. lsb==0 is a copy of Rm. Both the
// 32-bit and 64-bit non-zero cases map to x86 SHRD (the ROR Rn==Rm alias falls
// out for free). Mirrors lite_translator.h::Extr.
Register HeavyOptimizerFrontend::Extr(Register src_n, Register src_m, uint8_t lsb, bool is_64bit) {
  if (!success()) {
    return AllocTempReg();
  }
  if (lsb == 0) {
    if (is_64bit) {
      return Copy(src_m);
    }
    return std::get<0>(Gen<x86_64::MovlRegReg>(src_m));
  }
  // SHRD dest, src, imm: dest = (src:dest) >> imm.
  // ARM EXTR Rd = (Rn:Rm) >> lsb = SHRD(Rm, Rn, lsb): dest=Rm, src=Rn.
  if (is_64bit) {
    Register res = Copy(src_m);
    return std::get<0>(Gen<x86_64::ShrdqRegRegImm, kNoSSA>(res, src_n, static_cast<int8_t>(lsb)));
  }
  Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src_m));
  return std::get<0>(Gen<x86_64::ShrdlRegRegImm, kNoSSA>(res, src_n, static_cast<int8_t>(lsb)));
}

void HeavyOptimizerFrontend::MteDataProc(const Decoder::MteDataProcArgs& args) {
  UndefinedReturningVoid();
  UNUSED_ARGS(args);
}

void HeavyOptimizerFrontend::MteLoadStore(const Decoder::MteLoadStoreArgs& args) {
  UndefinedReturningVoid();
  UNUSED_ARGS(args);
}

// FCSEL Sd|Dd, Sn|Dn, Sm|Dm, cond:
//   V[rd] = ZeroExtend(ConditionHolds(cond) ? V[rn] : V[rm]).
// Lowered branchlessly (no basic-block manipulation, unlike the lite tier's
// Jcc form) so the whole scalar select stays inside one machine BB: build a
// full-width 0/-1 mask from the ARM condition predicate and blend the two
// scalars with PAND/PANDN/POR. Mirrors the interpreter's FpCondSelect
// (read the chosen source, zero-extend into V[rd]) semantically.
//
//   pred    = EmitArmCondPredicate(cond)   // 0/1, 1 iff cond holds
//   mask_gp = 0 - pred                      // cond ? 0xFFFF..FFFF : 0
//   mask    = MOVQ(mask_gp)                 // low 64 bits carry the mask
//   res     = Vn ; res &= mask ; mask = ~mask & Vm ; res |= mask  // cond?Vn:Vm
//   SetVRegScalar(rd, res)                  // zero-extends the low lane
//
// The blend runs in a PRIVATE temp `res` (a copy of Vn), NOT in the Vn/Vm GET
// results: those are cached guest-context loads the local-guest-context
// optimizer forwards to later reads of the same V register, so mutating them in
// place would hand a subsequent GET the blended value (the original heavy-tier
// regression — see the body). Only the low scalar lane needs a correct mask
// (SetVRegScalar re-zeroes the upper bytes), so a low-64 mask covers both S and
// D. ftype 0b11 (FP16) and reserved 0b10 bail to the lite tier, whose
// intrinsics cover them (matches FpDataProc3's FP16 bail).
void HeavyOptimizerFrontend::FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ftype, Decoder::Condition cond) {
  if (!success()) {
    return;
  }
  if (ftype != 0b00 && ftype != 0b01) {
    UndefinedReturningVoid();
    return;
  }
  const bool is_double = (ftype == 0b01);

  FpRegister vn = GetVRegScalar(rn, is_double);  // condition-TRUE operand
  FpRegister vm = GetVRegScalar(rm, is_double);  // condition-FALSE operand

  // AL/NV are unconditional on FCSEL and always select the TRUE operand.
  if (cond == Decoder::Condition::kAl || cond == Decoder::Condition::kNv) {
    SetVRegScalar(rd, vn, is_double);
    return;
  }

  Register pred = EmitArmCondPredicate(cond);  // 0/1
  // mask_gp = 0 - pred (no Neg op in the heavy IR; mirrors CSNEG's negate).
  Register zero = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  Register mask_gp = std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(zero, pred));
  FpRegister mask = AllocTempSimdReg();
  builder_.Gen<x86_64::MovqXRegReg>(mask.machine_reg(), mask_gp);

  // Blend into a private temp: res = (Vn & mask) | (Vm & ~mask), PANDN(d,s) =
  // ~d & s. `vn`/`vm` are the guest-context GET results (a MOVSD of V[rn]/V[rm]
  // the local-guest-context optimizer caches and forwards to later reads of the
  // same V register). They MUST NOT be mutated in place: doing so hands a later
  // GET of V[rn]/V[rm] the blended value instead of the original. That in-place
  // mutation was this lowering's original heavy-tier regression -- it destroyed
  // the cached V[rn] load and, before the optimizer learned to invalidate a
  // redefined forwarded vreg, deterministically miscompiled real regions (a
  // later `FADD V[rn]` saw the select result). Copy Vn into `res` and mutate
  // only `res` and the private `mask`, leaving the GET results pristine --
  // mirrors the FRINT/FCVT "round into a private temp so the guest v[] slot is
  // untouched" idiom, and is correct regardless of the forwarding optimizer.
  FpRegister res = AllocTempSimdReg();
  builder_.Gen<x86_64::MovdqaXRegXReg>(res.machine_reg(), vn.machine_reg());
  builder_.Gen<x86_64::PandXRegXReg>(res.machine_reg(), mask.machine_reg());
  builder_.Gen<x86_64::PandnXRegXReg>(mask.machine_reg(), vm.machine_reg());
  builder_.Gen<x86_64::PorXRegXReg>(res.machine_reg(), mask.machine_reg());

  SetVRegScalar(rd, res, is_double);
}

// FCVTZS/FCVTZU/SCVTF/UCVTF (fixed-point): GP integer <-> scalar FP with a
// 2^±fbits scale. Bridges into the shared unscaled EmitScvtfUcvtf / EmitFcvtz
// helpers with the fbits scale inserted (SCVTF/UCVTF scale the FP result by
// 2^-fbits; FCVTZS/FCVTZU pre-multiply the FP source by 2^+fbits). The scale is
// an exact power-of-2 multiply, so the helpers' single-rounding CVTSI2 and
// truncating-cvtt + saturation/NaN ladder carry through verbatim. Mirrors
// lite_translator.h::FpFixedPointConversion. FP16 (ftype 0b1x) bails to lite.
void HeavyOptimizerFrontend::FpFixedPointConversion(const Decoder::FpFixedPointArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::FpFixedPointOp;
  if (args.ftype != 0b00 && args.ftype != 0b01) {
    UndefinedReturningVoid(BailReason::kUnsupportedFpType);
    return;
  }
  const uint8_t fbits = args.fbits;
  if (fbits == 0 || fbits > (args.sf ? 64u : 32u)) {
    UndefinedReturningVoid();
    return;
  }
  const bool is_double = (args.ftype == 0b01);
  // Only op/sf/ftype/rd/rn are read by the helpers; `op` uses the
  // FpIntConversion encoding (SCVTF=010, UCVTF=011, FCVTZS=000, FCVTZU=001).
  Decoder::FpIntConvArgs iargs{};
  iargs.rd = args.rd;
  iargs.rn = args.rn;
  iargs.sf = args.sf;
  iargs.ftype = args.ftype;
  switch (args.op) {
    case Op::kScvtf:
      iargs.op = 0b010;
      EmitScvtfUcvtf(iargs, is_double, fbits);
      return;
    case Op::kUcvtf:
      iargs.op = 0b011;
      EmitScvtfUcvtf(iargs, is_double, fbits);
      return;
    case Op::kFcvtzs:
      iargs.op = 0b000;
      EmitFcvtz(iargs, is_double, /*round_imm=*/-1, /*ties_away=*/false, fbits);
      return;
    case Op::kFcvtzu:
      iargs.op = 0b001;
      EmitFcvtz(iargs, is_double, /*round_imm=*/-1, /*ties_away=*/false, fbits);
      return;
  }
}

// FMADD/FMSUB/FNMADD/FNMSUB (FP data-processing, 3 source) at S/D. Lowered to
// the x86 FMA3 231-form ops, which — like ARM's fused multiply-add — round the
// whole a+n*m once. A plain MUL+ADD would double-round and is wrong, so a host
// without FMA3 bails to the lite tier (which uses libc fma()/fmaf()). Mirrors
// lite_translator.h::FpDataProc3's S/D paths. The FMA231 op is use_def on its
// dest (the accumulator = Ra), so copy Ra into a fresh temp first to avoid
// clobbering the guest V[ra] mapping:
//   FMADD  (o1=0,o0=0) Ra + Rn*Rm     -> Vfmadd231  (acc + Rn*Rm)
//   FMSUB  (o1=0,o0=1) Ra - Rn*Rm     -> Vfnmadd231 (acc + -(Rn*Rm))
//   FNMADD (o1=1,o0=0) -(Ra + Rn*Rm)  -> Vfnmsub231 (-(Rn*Rm) - acc)
//   FNMSUB (o1=1,o0=1) Rn*Rm - Ra     -> Vfmsub231  (Rn*Rm - acc)
// FP16 (ftype=0b11, needs F16C widen/narrow ops not in the backend gen inputs)
// and the reserved ftype=0b10 bail to the lite tier.
void HeavyOptimizerFrontend::FpDataProc3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra, uint8_t ftype, bool o1, bool o0) {
  if (!success()) {
    return;
  }
  if (ftype != 0b00 && ftype != 0b01 && ftype != 0b11) {
    UndefinedReturningVoid(BailReason::kUnsupportedFpType);
    return;
  }
  if (!host_platform::kHasFMA) {
    UndefinedReturningVoid();
    return;
  }
  if (ftype == 0b11 && !host_platform::kHasF16C) {
    UndefinedReturningVoid(BailReason::kUnsupportedFpType);
    return;
  }
  const bool is_double = (ftype == 0b01);
  const bool is_half = (ftype == 0b11);
  // FP16 FMA (FMADD/FMSUB/FNMADD/FNMSUB Hd) is correctly-rounded by fusing in
  // FP64 and narrowing once: FP64's 53 significand bits exceed 2*11+1, so
  // fp16->FP64 fuse then RNE-narrow to fp16 avoids the double-rounding a naive
  // FP32 round-trip would suffer. This mirrors the lite tier's is_half recipe and
  // the interpreter's ftype==0b11 fma() branch. Widen each operand fp16->FP32
  // (F16C) -> FP64; lane 1 carries FP64 +0.0 through the SD FMA so the FP64->FP32
  // narrow yields +0.0 upper lanes for the clean fp16 store.
  FpRegister xmm_n, xmm_m, xmm_a;
  if (is_half) {
    xmm_n = EmitWidenHalfToF32(rn);
    builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xmm_n.machine_reg(), xmm_n.machine_reg());
    xmm_m = EmitWidenHalfToF32(rm);
    builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xmm_m.machine_reg(), xmm_m.machine_reg());
    xmm_a = EmitWidenHalfToF32(ra);
    builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xmm_a.machine_reg(), xmm_a.machine_reg());
  } else {
    xmm_n = GetVRegScalar(rn, is_double);
    xmm_m = GetVRegScalar(rm, is_double);
    xmm_a = GetVRegScalar(ra, is_double);
  }
  // FP16 fuses in the SD (FP64) form, exactly as the lite tier does.
  const bool use_double_fma = is_double || is_half;
  FpRegister acc = AllocTempSimdReg();
  builder_.Gen<x86_64::MovdqaXRegXReg>(acc.machine_reg(), xmm_a.machine_reg());
  if (!o1 && !o0) {  // FMADD
    if (use_double_fma) {
      builder_.Gen<x86_64::Vfmadd231sdXRegXRegXReg>(
          acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
    } else {
      builder_.Gen<x86_64::Vfmadd231ssXRegXRegXReg>(
          acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
    }
  } else if (!o1 && o0) {  // FMSUB
    if (use_double_fma) {
      builder_.Gen<x86_64::Vfnmadd231sdXRegXRegXReg>(
          acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
    } else {
      builder_.Gen<x86_64::Vfnmadd231ssXRegXRegXReg>(
          acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
    }
  } else if (o1 && !o0) {  // FNMADD
    if (use_double_fma) {
      builder_.Gen<x86_64::Vfnmsub231sdXRegXRegXReg>(
          acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
    } else {
      builder_.Gen<x86_64::Vfnmsub231ssXRegXRegXReg>(
          acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
    }
  } else {  // FNMSUB
    if (use_double_fma) {
      builder_.Gen<x86_64::Vfmsub231sdXRegXRegXReg>(
          acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
    } else {
      builder_.Gen<x86_64::Vfmsub231ssXRegXRegXReg>(
          acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
    }
  }
  if (is_half) {
    // Narrow FP64 -> FP32 (lane 1 = +0.0 preserved by the SD FMA) -> fp16, store.
    builder_.Gen<x86_64::Cvtpd2psXRegXReg>(acc.machine_reg(), acc.machine_reg());
    EmitNarrowF32ToHalfAndStore(rd, acc);
    return;
  }
  SetVRegScalar(rd, acc, is_double);
}

// FMOV (scalar, immediate): the FP constant is fully known at translation
// time (VFPExpandImm of imm8). Zero the 16-byte V[d] slot, then store the
// 32/64-bit constant into lane 0. Mirrors lite_translator.h::FpMovImmediate;
// bail FP16 (ftype=0b11) and the reserved ftype=0b10.
void HeavyOptimizerFrontend::FpMovImmediate(uint8_t rd, uint8_t imm8, uint8_t ftype) {
  if (!success()) {
    return;
  }
  if (ftype == 0b10) {
    UndefinedReturningVoid(BailReason::kUnsupportedFpType);
    return;
  }
  if (ftype == 0b11) {
    // FMOV Hd,#imm: VFPExpandImm16(imm8) is an exact translation-time constant.
    // No F16C. (The lite tier currently bails this to the interpreter; heavy is a
    // correct superset — the interpreter also implements it, so all tiers agree.)
    uint16_t imm16 = VFPExpandImm16(imm8);
    Register tmp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(imm16)));
    SetVRegScalarFromGp(rd, tmp, /*is_double=*/false);  // MOVD [imm16,0,0,0] -> zeroed V[rd]
    return;
  }
  if (ftype == 0b00) {
    uint32_t imm32 = VFPExpandImm32(imm8);
    Register tmp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(imm32)));
    SetVRegScalarFromGp(rd, tmp, /*is_double=*/false);
  } else {
    uint64_t imm64 = VFPExpandImm64(imm8);
    Register tmp = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(imm64)));
    SetVRegScalarFromGp(rd, tmp, /*is_double=*/true);
  }
}

// FMOV between a general register and a scalar FP register, single (S/W) or
// double (D/X), via x86 MOVD/MOVQ; plus SCVTF/UCVTF (integer -> FP) via the
// x86 CVTSI2SS/SD ops (EmitScvtfUcvtf) and FCVTZS/FCVTZU (FP -> int, truncate
// toward zero) via CVTT{SS,SD}2SI + the ARM saturation/NaN fix-up (EmitFcvtz).
// The FMOV top-half (V.D[1]) forms are handled here; the rounding FP -> int
// conversions (FCVTNS/PS/MS) and ties-away FCVTAS/AU are routed through EmitFcvtz
// (both S and D). FP16 (ftype == 0b11) bails to the lite tier, whose intrinsics
// cover it.
// Guest V[] access stays in the XMM domain (GetVRegScalar / SetVRegScalar*),
// and the GP<->XMM crossing is an explicit register move, not a forwarded
// guest-context GET. Mirrors lite_translator.h::FpIntConversion (FMOV +
// SCVTF/UCVTF subset).
void HeavyOptimizerFrontend::FpIntConversion(const Decoder::FpIntConvArgs& args) {
  if (!success()) {
    return;
  }
  // FMOV top-half (V.D[1]) forms: rmode=01, op=110/111. These decode with
  // ftype=0b10 (the 128-bit "Q" size selector, sf=1) so they must be handled
  // BEFORE the ftype gate below. Each is a 64-bit lane move between a GP and the
  // HIGH half of V, preserving the LOW half — the scalar-SIMD lane-move idiom
  // (GenGetSimd<16> full load, PEXTRQ/PINSRQ lane 1, GenSetSimd<16> full store),
  // matching the INS-general / UMOV.D handlers and lite_translator_fp_scalar.inc.
  if (args.rmode == 0b01 && (args.op == 0b110 || args.op == 0b111)) {
    if (args.op == 0b111) {
      // FMOV Vd.D[1], Xn: write V[rd].D[1] = Xn, leaving V[rd].D[0] intact.
      const int32_t vd_off = GetVRegOffset(args.rd);
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      Register src = (args.rn < 31) ? GetReg(args.rn)
                                    : std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));  // XZR
      builder_.Gen<x86_64::PinsrqXRegRegImm>(xd.machine_reg(), src, int8_t{1});
      builder_.GenSetSimd<16>(vd_off, xd.machine_reg());
    } else {
      // FMOV Xd, Vn.D[1]: read V[rn].D[1] into Xd (rd==31 discards).
      if (args.rd != 31) {
        const int32_t vn_off = GetVRegOffset(args.rn);
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        Register gp = std::get<0>(Gen<x86_64::PextrqRegXRegImm>(xn.machine_reg(), int8_t{1}));
        SetReg(args.rd, gp);
      }
    }
    return;
  }
  if (args.ftype != 0b00 && args.ftype != 0b01) {
    UndefinedReturningVoid(BailReason::kUnsupportedFpType);
    return;
  }
  const bool is_double = (args.ftype == 0b01);
  if (args.rmode == 0b00 && (args.op == 0b110 || args.op == 0b111)) {
    if (args.op == 0b111) {
      // FMOV Sd, Wn / Dd, Xn: general register -> scalar FP (upper lanes zeroed).
      if (args.rn == 31) {
        SetVRegScalar(args.rd, AllocZeroedSimdReg(), is_double);  // WZR/XZR -> 0
      } else {
        SetVRegScalarFromGp(args.rd, GetReg(args.rn), is_double);
      }
    } else {
      // FMOV Wd, Sn / Xd, Dn: scalar FP -> general register.
      if (args.rd == 31) {
        return;  // WZR/XZR destination: discard.
      }
      FpRegister xmm = GetVRegScalar(args.rn, is_double);
      Register gp = is_double
                        ? std::get<0>(Gen<x86_64::MovqRegXReg>(xmm.machine_reg()))
                        : std::get<0>(Gen<x86_64::MovdRegXReg>(xmm.machine_reg()));
      SetReg(args.rd, gp);
    }
    return;
  }
  // SCVTF (op 010) / UCVTF (op 011): integer -> FP, unscaled (rmode == 00).
  if (args.rmode == 0b00 && (args.op == 0b010 || args.op == 0b011)) {
    EmitScvtfUcvtf(args, is_double);
    return;
  }
  // FCVTZS (op 000) / FCVTZU (op 001): FP -> int, truncate (rmode == 11).
  if (args.rmode == 0b11 && (args.op == 0b000 || args.op == 0b001)) {
    EmitFcvtz(args, is_double);
    return;
  }
  // Rounding FP -> int conversions FCVTNS/NU (rmode 00, round-to-nearest
  // ties-even), FCVTPS/PU (rmode 01, toward +inf) and FCVTMS/MU (rmode 10,
  // toward -inf). ARM ties-away FCVTAS/AU (op 100/101) has no x86 round mode
  // and still bails. Route through EmitFcvtz with the matching x86 ROUND imm8
  // so the shared saturation/NaN ladder handles out-of-range/NaN.
  if ((args.op == 0b000 || args.op == 0b001) &&
      (args.rmode == 0b00 || args.rmode == 0b01 || args.rmode == 0b10)) {
    const int8_t round_imm = (args.rmode == 0b00) ? int8_t{0x08}    // RNE + suppress-inexact
                             : (args.rmode == 0b01) ? int8_t{0x0A}  // toward +inf
                                                    : int8_t{0x09};  // toward -inf
    EmitFcvtz(args, is_double, round_imm);
    return;
  }
  // FCVTAS (op 100) / FCVTAU (op 101): FP->int, round-to-nearest ties-away.
  // x86 has no ties-away round mode; EmitFcvtz's ties_away path adds
  // copysign(0.5, x) (gated at |x| >= 2^23 FP32 / 2^52 FP64) before the
  // truncating saturation ladder. Both FP32 (S) and FP64 (D) are covered.
  if ((args.op == 0b100 || args.op == 0b101) && args.rmode == 0b00) {
    EmitFcvtz(args, is_double, /*round_imm=*/-1, /*ties_away=*/true);
    return;
  }
  // Everything else (rmode==01 V.D[1] FMOV) bails to lite.
  UndefinedReturningVoid();
}

// FMOV(reg) / FABS / FNEG for FP32 (ftype=00) and FP64 (ftype=01). These are
// pure bit operations: copy (FMOV), sign-bit clear (FABS), sign-bit flip
// (FNEG). Everything stays in the XMM domain — read the scalar into an XMM,
// apply the sign mask there (PAND for FABS, XORPD for FNEG against a mask XMM),
// and write back. Keeping reads/writes of guest v[] in the XMM domain is
// required: RemoveLocalGuestContextAccesses forwards a prior 16-byte MOVDQA
// store of an XMM vreg straight into a same-offset GET, so a GP-domain read of
// the same v[] slot would funnel an XMM vreg into a GP operand and fail
// register-class intersection.
//
// FSQRT, FRINT*, FCVT (precision change) are lowered below (SQRTSS/SQRTSD,
// ROUNDSS/ROUNDSD, CVTSS2SD/CVTSD2SS). BFCVT, FCVT-to-half (F16C), FP16, and
// ftype=10 still bail: their SSE ops are not allowlisted for the ARM64 backend.
void HeavyOptimizerFrontend::FpDataProc1(const Decoder::FpDataProc1Args& args) {
  if (!success()) {
    return;
  }
  if (args.ftype == 0b10) {
    UndefinedReturningVoid(BailReason::kUnsupportedFpType);
    return;
  }
  if (args.ftype == 0b11) {
    // FP16 one-source ops. FMOV/FABS/FNEG are pure 16-bit bit ops (no F16C);
    // FCVT-from-half, FSQRT, FRINT*, FRINTA use the F16C round-trip. Bit-exact
    // with lite_translator_fp_dataproc.inc's FpDataProc1 half path.
    if (args.opcode == 0b000000 || args.opcode == 0b000001 || args.opcode == 0b000010) {
      // FMOV Hd,Hn / FABS (clear bit15) / FNEG (flip bit15). No F16C.
      FpRegister raw = GetVRegScalar(args.rn, /*is_double=*/false);
      FpRegister mask = AllocTempSimdReg();
      uint32_t keep = (args.opcode == 0b000001) ? 0x00007FFFu : 0x0000FFFFu;  // FABS clears sign
      builder_.Gen<x86_64::MovdXRegReg>(mask.machine_reg(), GetImm(uint64_t{keep}));
      builder_.Gen<x86_64::PandXRegXReg>(raw.machine_reg(), mask.machine_reg());
      if (args.opcode == 0b000010) {  // FNEG: flip sign bit 15
        FpRegister sb = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdXRegReg>(sb.machine_reg(), GetImm(uint64_t{0x00008000u}));
        builder_.Gen<x86_64::PxorXRegXReg>(raw.machine_reg(), sb.machine_reg());
      }
      SetVRegScalar(args.rd, raw, /*is_double=*/false);
      return;
    }
    if (!host_platform::kHasF16C) {
      UndefinedReturningVoid();
      return;
    }
    if (args.opcode == 0b000100 || args.opcode == 0b000101) {
      // FCVT Sd,Hn (widen half->single) / FCVT Dd,Hn (->double).
      FpRegister w = EmitWidenHalfToF32(args.rn);  // [f32(half),+0,+0,+0]
      if (args.opcode == 0b000101) {
        FpRegister d = AllocZeroedSimdReg();
        builder_.Gen<x86_64::Cvtss2sdXRegXReg>(d.machine_reg(), w.machine_reg());
        SetVRegScalar(args.rd, d, /*is_double=*/true);
      } else {
        SetVRegScalar(args.rd, w, /*is_double=*/false);
      }
      return;
    }
    if (args.opcode == 0b001100) {
      // FRINTA Hd,Hn (ties-away). Every FP16 magnitude is < 2^23, so — unlike the
      // FP32/FP64 path — no magnitude gate is needed: add copysign(0.5f,x), then
      // truncate. NaN/Inf pass through (add-to-NaN=NaN, trunc(Inf)=Inf). Mirrors
      // the lite half FRINTA (no gate).
      FpRegister w = EmitWidenHalfToF32(args.rn);   // [f32,+0,+0,+0]
      FpRegister signbit = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdXRegReg>(signbit.machine_reg(), GetImm(uint64_t{0x80000000u}));
      FpRegister addend = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(addend.machine_reg(), w.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(addend.machine_reg(), signbit.machine_reg());  // sign of x
      FpRegister halfc = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdXRegReg>(halfc.machine_reg(), GetImm(uint64_t{0x3F000000u}));  // 0.5f
      builder_.Gen<x86_64::PorXRegXReg>(addend.machine_reg(), halfc.machine_reg());  // copysign(0.5,x)
      builder_.Gen<x86_64::AddssXRegXReg>(w.machine_reg(), addend.machine_reg());
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(w.machine_reg(), w.machine_reg(), int8_t{0x03});
      EmitNarrowF32ToHalfAndStore(args.rd, w);
      return;
    }
    // FSQRT / FRINT{N,P,M,Z,X,I} Hd,Hn.
    int8_t round_imm = 0;
    bool is_sqrt = false;
    switch (args.opcode) {
      case 0b000011: is_sqrt = true; break;         // FSQRT
      case 0b001000: round_imm = int8_t{0x00}; break;  // FRINTN
      case 0b001001: round_imm = int8_t{0x02}; break;  // FRINTP
      case 0b001010: round_imm = int8_t{0x01}; break;  // FRINTM
      case 0b001011: round_imm = int8_t{0x03}; break;  // FRINTZ
      case 0b001110: round_imm = int8_t{0x00}; break;  // FRINTX
      case 0b001111: round_imm = int8_t{0x08}; break;  // FRINTI
      default:
        UndefinedReturningVoid();
        return;
    }
    FpRegister w = EmitWidenHalfToF32(args.rn);  // [f32,+0,+0,+0]
    if (is_sqrt) {
      builder_.Gen<x86_64::SqrtssXRegXReg>(w.machine_reg(), w.machine_reg());  // scalar; upper +0.0
    } else {
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(w.machine_reg(), w.machine_reg(), round_imm);
    }
    EmitNarrowF32ToHalfAndStore(args.rd, w);
    return;
  }
  const bool is_double = (args.ftype == 0b01);

  // FCVT between FP32 and FP64: a single CVTSS2SD (Dd,Sn) / CVTSD2SS (Sd,Dn)
  // into lane 0 of a zeroed dst, then a scalar (upper-zeroing) store. The
  // result precision is the OPPOSITE of the source ftype, so the write uses
  // the complemented is_double. Mirrors lite_translator.h::FpDataProc1's
  // CVTSS2SD/CVTSD2SS arms. FCVT-to-half (0b000111) needs F16C and BFCVT
  // (0b000110) needs the NaN-fixup narrow — both still bail to the lite tier.
  // FSQRT (0b000011) is lowered just below via SQRTSS/SQRTSD.
  if (!is_double && args.opcode == 0b000101) {  // FCVT Dd, Sn (single -> double)
    FpRegister src = GetVRegScalar(args.rn, /*is_double=*/false);
    FpRegister dst = AllocZeroedSimdReg();
    builder_.Gen<x86_64::Cvtss2sdXRegXReg>(dst.machine_reg(), src.machine_reg());
    SetVRegScalar(args.rd, dst, /*is_double=*/true);
    return;
  }
  if (is_double && args.opcode == 0b000100) {  // FCVT Sd, Dn (double -> single)
    FpRegister src = GetVRegScalar(args.rn, /*is_double=*/true);
    FpRegister dst = AllocZeroedSimdReg();
    builder_.Gen<x86_64::Cvtsd2ssXRegXReg>(dst.machine_reg(), src.machine_reg());
    SetVRegScalar(args.rd, dst, /*is_double=*/false);
    return;
  }

  // FCVT Hd, Sn / FCVT Hd, Dn (opcode 0b000111): narrow to half. Needs F16C. For
  // a double source, narrow double->single first (into a zeroed reg so lanes 1..3
  // stay +0.0), then VCVTPS2PH single->half. Mirrors lite_translator_fp_dataproc.
  if (args.opcode == 0b000111) {
    if (!host_platform::kHasF16C) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister src = GetVRegScalar(args.rn, is_double);
    FpRegister s32 = AllocZeroedSimdReg();  // lanes 1..3 = +0.0
    if (is_double) {
      // Narrow double->single into a fresh reg, then MOVSS lane 0 into the
      // zeroed s32. Funneling the double path through the same MOVSS-into-zeroed
      // step as the single path keeps the narrow's input identical for both (a
      // Cvtsd2ss result read directly by the in-place VCVTPS2PH does not
      // materialize under the local optimizer).
      FpRegister single = AllocZeroedSimdReg();
      builder_.Gen<x86_64::Cvtsd2ssXRegXReg>(single.machine_reg(), src.machine_reg());
      builder_.Gen<x86_64::MovssXRegXReg>(s32.machine_reg(), single.machine_reg());
    } else {
      builder_.Gen<x86_64::MovssXRegXReg>(s32.machine_reg(), src.machine_reg());  // lane 0 only
    }
    EmitNarrowF32ToHalfAndStore(args.rd, s32);
    return;
  }

  // BFCVT Hd, Sn (opcode 0b000110, ftype=01): narrow FP32 -> BF16 (bfloat16) with
  // round-to-nearest-even and NaN quieting. The source is read as FP32 (Sn)
  // despite ftype=01 — the discriminant is a pure opcode-namespace selector, not
  // a "source is double" hint. Pure integer bit-manip on the raw FP32 bits, so no
  // AVX-512-BF16 dependency. Bit-exact with interpreter.h::FloatToBf16 and
  // lite_translator_fp_dataproc.inc's BFCVT:
  //   if (exp==0xFF && mant!=0): out = (bits >> 16) | 0x0040   (quiet NaN)
  //   else:                      out = (bits + 0x7FFF + ((bits>>16)&1)) >> 16
  // The FP32 bits are read via the XMM domain (MOVD out of GetVRegScalar) — a GP
  // read of the v[] slot would break the local optimizer's MOVDQA forwarding.
  if (args.opcode == 0b000110) {
    FpRegister src = GetVRegScalar(args.rn, /*is_double=*/false);
    Register bits = std::get<0>(Gen<x86_64::MovdRegXReg>(src.machine_reg()));

    auto* ir = builder_.ir();
    Register out = AllocTempReg();

    MachineBasicBlock* checkmant_bb = ir->NewBasicBlock();
    MachineBasicBlock* rtne_bb = ir->NewBasicBlock();
    MachineBasicBlock* nan_bb = ir->NewBasicBlock();
    MachineBasicBlock* done_bb = ir->NewBasicBlock();

    // exp == 0xFF ? Compute (bits & 0x7F800000) ^ 0x7F800000; ZF set iff all-ones.
    Register exp = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(bits),
                                                              int32_t{0x7F800000}));
    exp = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(exp, int32_t{0x7F800000}));
    Register exp_flags = std::get<0>(Gen<x86_64::TestlRegReg>(exp, exp));
    auto* cur_bb = builder_.bb();
    ir->AddEdge(cur_bb, checkmant_bb);
    ir->AddEdge(cur_bb, rtne_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kZero, checkmant_bb,
                                   rtne_bb, exp_flags);

    // exp all-ones: mantissa != 0 -> NaN, else (±Inf) -> RTNE.
    builder_.StartBasicBlock(checkmant_bb);
    Register mant = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(bits),
                                                               int32_t{0x007FFFFF}));
    Register mant_flags = std::get<0>(Gen<x86_64::TestlRegReg>(mant, mant));
    ir->AddEdge(checkmant_bb, nan_bb);
    ir->AddEdge(checkmant_bb, rtne_bb);
    builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kNotZero, nan_bb,
                                   rtne_bb, mant_flags);

    // RTNE: out = (bits + 0x7FFF + ((bits>>16)&1)) >> 16.
    builder_.StartBasicBlock(rtne_bb);
    Register lsb = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(bits), int8_t{16}));
    lsb = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(lsb, int32_t{1}));
    Register r = std::get<0>(Gen<x86_64::AddlRegImm, kNoSSA>(Copy(bits), int32_t{0x7FFF}));
    r = std::get<0>(Gen<x86_64::AddlRegReg, kNoSSA>(r, lsb));
    r = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(r, int8_t{16}));
    builder_.Gen<PseudoCopy>(out, r, 8);
    ir->AddEdge(rtne_bb, done_bb);
    builder_.Gen<PseudoBranch>(done_bb);

    // NaN: out = (bits >> 16) | 0x0040 (force BF16 mantissa MSB -> quiet).
    builder_.StartBasicBlock(nan_bb);
    Register n = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(bits), int8_t{16}));
    n = std::get<0>(Gen<x86_64::OrlRegImm, kNoSSA>(n, int32_t{0x0040}));
    builder_.Gen<PseudoCopy>(out, n, 8);
    ir->AddEdge(nan_bb, done_bb);
    builder_.Gen<PseudoBranch>(done_bb);

    // Commit: BF16 in bits[15:0] of V[rd] (out's high 16 are already 0), rest
    // zeroed. MOVD zero-extends into lane 0; SetVRegScalar zeroes the upper bytes.
    builder_.StartBasicBlock(done_bb);
    FpRegister res = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdXRegReg>(res.machine_reg(), out);
    SetVRegScalar(args.rd, res, /*is_double=*/false);
    return;
  }

  // FSQRT Sd/Dd, Sn/Dn (opcode 0b000011): a single SQRTSS (FP32) / SQRTSD
  // (FP64). x86 SQRTSS/SQRTSD compute the exact IEEE-754 square root with
  // round-to-nearest-even, matching ARM FSQRT under the default FPCR rounding
  // mode, and propagate NaN/-Inf per the Intel SDM identically to ARM. Mirrors
  // lite_translator.h::FpDataProc1's Sqrtss/Sqrtsd arms. Only lane 0 is read
  // and SetVRegScalar commits only lane 0, so the scalar op is exact.
  if (args.opcode == 0b000011) {
    FpRegister val = GetVRegScalar(args.rn, is_double);
    if (is_double) {
      builder_.Gen<x86_64::SqrtsdXRegXReg>(val.machine_reg(), val.machine_reg());
    } else {
      builder_.Gen<x86_64::SqrtssXRegXReg>(val.machine_reg(), val.machine_reg());
    }
    SetVRegScalar(args.rd, val, is_double);
    return;
  }

  // FRINT{N,M,P,Z,X,I} (round float to integral float): a single ROUNDSD (D)
  // / ROUNDPS-on-lane-0 (S — the backend has no scalar ROUNDSS, but SetVReg
  // Scalar commits only lane 0 so the packed round is safe) with the matching
  // rounding-mode immediate. FRINT keeps its result in the FP register, so —
  // unlike the FCVT* integer converts — there is no saturation fix-up; one
  // ROUND handles every finite/NaN/±Inf/signed-zero value. FRINTA (ties-to-
  // away, 0b001100) has no native ROUND* imm; it is handled just below via the
  // copysign(0.5)+truncate sequence with a magnitude gate. Mirrors
  // lite_translator.h::FpDataProc1's FP32/FP64 FRINT switch (FRINTN->0x00,
  // FRINTP->0x02, FRINTM->0x01, FRINTZ->0x03, FRINTX->0x00, FRINTI->0x08).
  {
    int8_t round_imm = 0;
    bool is_frint = true;
    switch (args.opcode) {
      case 0b001000: round_imm = int8_t{0x00}; break;  // FRINTN (nearest-even)
      case 0b001001: round_imm = int8_t{0x02}; break;  // FRINTP (toward +inf)
      case 0b001010: round_imm = int8_t{0x01}; break;  // FRINTM (toward -inf)
      case 0b001011: round_imm = int8_t{0x03}; break;  // FRINTZ (toward zero)
      case 0b001110: round_imm = int8_t{0x00}; break;  // FRINTX (FPCR; RNE)
      case 0b001111: round_imm = int8_t{0x08}; break;  // FRINTI (SAE+RNE)
      default: is_frint = false; break;
    }
    if (is_frint) {
      FpRegister val = GetVRegScalar(args.rn, is_double);
      if (is_double) {
        builder_.Gen<x86_64::RoundsdXRegXRegImm>(
            val.machine_reg(), val.machine_reg(), round_imm);
      } else {
        builder_.Gen<x86_64::RoundpsXRegXRegImm>(
            val.machine_reg(), val.machine_reg(), round_imm);
      }
      SetVRegScalar(args.rd, val, is_double);
      return;
    }
  }

  // FRINTA Sd/Dd, Sn/Dn (round to nearest, ties away from zero, opcode
  // 0b001100): x86 has no ROUND* immediate for ties-away, so mirror
  // lite_translator.h::FpDataProc1's copysign(0.5)+truncate trick, using the
  // branchless SIMD magnitude gate from the FCVTAS/FCVTAU ties-away path in
  // frontend.cc. Add copysign(0.5, x) then truncate toward zero (ROUND imm
  // 0x03), gating the addend to 0 where |x| >= 2^23 (FP32) / 2^52 (FP64): at
  // that magnitude x is already an integer and a 0.5 addend would round the
  // wrong way. NaN/±Inf bits also exceed the threshold, so the addend is gated
  // off and ROUND-toward-zero preserves them per the Intel SDM. The magnitude
  // compare is a signed PCMPGT on |bits(x)| (top bit cleared, so signed ==
  // unsigned). sign/absx use AllocZeroedSimdReg because they are self-PCMPEQ
  // targets (a fresh AllocTempSimdReg vreg would trip the lifetime use-before-
  // def CHECK).
  if (args.opcode == 0b001100) {
    FpRegister val = GetVRegScalar(args.rn, is_double);
    FpRegister sign = AllocZeroedSimdReg();
    FpRegister absx = AllocZeroedSimdReg();
    FpRegister half = AllocTempSimdReg();
    FpRegister thresh = AllocTempSimdReg();
    if (is_double) {
      // addend = copysign(0.5d, x) = (x & sign_bit) | bits(0.5d).
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
      builder_.Gen<x86_64::PsllqXRegImm>(sign.machine_reg(), int8_t{63});  // 0x8000...0
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), val.machine_reg());
      builder_.Gen<x86_64::MovqXRegReg>(half.machine_reg(),
                                        GetImm(uint64_t{0x3FE0000000000000}));  // 0.5d
      builder_.Gen<x86_64::PorXRegXReg>(sign.machine_reg(), half.machine_reg());
      // gate = (|x| < 2^52) ? all-ones : 0, via thresh > |x|.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(absx.machine_reg(), absx.machine_reg());
      builder_.Gen<x86_64::PsrlqXRegImm>(absx.machine_reg(), int8_t{1});  // 0x7FFF...F
      builder_.Gen<x86_64::PandXRegXReg>(absx.machine_reg(), val.machine_reg());
      builder_.Gen<x86_64::MovqXRegReg>(thresh.machine_reg(),
                                        GetImm(uint64_t{0x4330000000000000}));  // 2^52
      builder_.Gen<x86_64::PcmpgtqXRegXReg>(thresh.machine_reg(), absx.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), thresh.machine_reg());
      builder_.Gen<x86_64::AddpdXRegXReg>(val.machine_reg(), sign.machine_reg());
      builder_.Gen<x86_64::RoundsdXRegXRegImm>(
          val.machine_reg(), val.machine_reg(), int8_t{0x03});
    } else {
      // addend = copysign(0.5f, x) = (x & sign_bit) | bits(0.5f).
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(sign.machine_reg(), int8_t{31});  // 0x80000000
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), val.machine_reg());
      builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(),
                                        GetImm(uint64_t{0x3F000000}));  // 0.5f
      builder_.Gen<x86_64::PorXRegXReg>(sign.machine_reg(), half.machine_reg());
      // gate = (|x| < 2^23) ? all-ones : 0, via thresh > |x|.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(absx.machine_reg(), absx.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(absx.machine_reg(), int8_t{1});  // 0x7FFFFFFF
      builder_.Gen<x86_64::PandXRegXReg>(absx.machine_reg(), val.machine_reg());
      builder_.Gen<x86_64::MovdXRegReg>(thresh.machine_reg(),
                                        GetImm(uint64_t{0x4B000000}));  // 2^23
      builder_.Gen<x86_64::PcmpgtdXRegXReg>(thresh.machine_reg(), absx.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), thresh.machine_reg());
      builder_.Gen<x86_64::AddpsXRegXReg>(val.machine_reg(), sign.machine_reg());
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(
          val.machine_reg(), val.machine_reg(), int8_t{0x03});
    }
    SetVRegScalar(args.rd, val, is_double);
    return;
  }

  // FMOV / FABS / FNEG only. Anything else bails.
  if (args.opcode != 0b000000 && args.opcode != 0b000001 && args.opcode != 0b000010) {
    UndefinedReturningVoid();
    return;
  }

  FpRegister val = GetVRegScalar(args.rn, is_double);

  if (args.opcode != 0b000000) {
    // Build the sign mask in a GP register (a fresh immediate, never a
    // forwarded guest-context value, so the GP->XMM move is conflict-free),
    // move it into an XMM, then PAND (FABS: clear sign) / XORPD (FNEG: flip
    // sign). FP32 masks live in the low 32 bits; lanes above 0 are irrelevant
    // because SetVRegScalar only commits lane 0.
    FpRegister mask = AllocTempSimdReg();
    if (is_double) {
      uint64_t m = (args.opcode == 0b000001) ? 0x7FFFFFFFFFFFFFFFULL : 0x8000000000000000ULL;
      Register gm = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(m)));
      builder_.Gen<x86_64::MovqXRegReg>(mask.machine_reg(), gm);
    } else {
      uint32_t m = (args.opcode == 0b000001) ? 0x7FFFFFFFu : 0x80000000u;
      Register gm = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(m)));
      builder_.Gen<x86_64::MovdXRegReg>(mask.machine_reg(), gm);
    }
    if (args.opcode == 0b000001) {  // FABS
      builder_.Gen<x86_64::PandXRegXReg>(val.machine_reg(), mask.machine_reg());
    } else {  // FNEG
      builder_.Gen<x86_64::XorpdXRegXReg>(val.machine_reg(), mask.machine_reg());
    }
  }
  SetVRegScalar(args.rd, val, is_double);
}

// FADD/FSUB/FMUL/FDIV for FP32 (ftype=00) and FP64 (ftype=01), lowered through
// the guest-agnostic intrinsic layer (InlineIntrinsicForHeavyOptimizer ->
// FXxxHostRounding -> SSE ADDSS/ADDSD/... per machine_ir_intrinsic_binding.json).
// Host default rounding is round-to-nearest, matching ARM FPCR.RMode=0.
//
// FPSR note: heavy-optimizer FP regions do NOT yet accumulate FPSR exception
// bits into cpu.emulated_fpsr (the FeGetExceptions/FeSetExceptions intrinsics
// are riscv64-only macro defs and convert to the RISC-V exception bit layout,
// not ARM's). FP *results* are correct without this; MRS-of-FPSR bails to the
// lite translator/interpreter, which maintains the flags. This is a follow-up.
//
// FMUL/FDIV/FADD/FSUB, FMAX/FMIN/FMAXNM/FMINNM (opcode 0b0100..0b0111), and
// FNMUL (0b1000) are wired here. FP16 (ftype=0b11) and the reserved ftype=0b10
// bail: their intrinsics/SSE ops are not available in this tier.
void HeavyOptimizerFrontend::FpDataProc2(const Decoder::FpDataProc2Args& args) {
  if (!success()) {
    return;
  }
  if (args.ftype == 0b10) {
    UndefinedReturningVoid(BailReason::kUnsupportedFpType);
    return;
  }
  if (args.ftype == 0b11) {
    // FP16 via F16C round-trip. Widen both operands (half in lane 0, lanes 1..3
    // = +0.0), compute in FP32 with SCALAR ops (upper lanes stay +0.0), narrow
    // back. Exact for +,-,*,/ (FP32 mantissa covers a single narrowing) and for
    // sqrt; FMA (FpDataProc3) still bails (double rounding). Bit-exact with
    // lite_translator_fp_dataproc.inc's FpDataProc2 half path. Needs host F16C.
    if (!host_platform::kHasF16C) {
      UndefinedReturningVoid();
      return;
    }
    if (args.opcode > 0b1000) {
      UndefinedReturningVoid();  // reserved
      return;
    }
    FpRegister n = EmitWidenHalfToF32(args.rn);
    FpRegister m = EmitWidenHalfToF32(args.rm);
    FpRegister result = n;
    switch (args.opcode) {
      case 0b0000:    // FMUL
      case 0b1000:    // FNMUL: -(n*m) — sign flipped below
        builder_.Gen<x86_64::MulssXRegXReg>(n.machine_reg(), m.machine_reg());
        break;
      case 0b0001:    // FDIV
        builder_.Gen<x86_64::DivssXRegXReg>(n.machine_reg(), m.machine_reg());
        break;
      case 0b0010:    // FADD
        builder_.Gen<x86_64::AddssXRegXReg>(n.machine_reg(), m.machine_reg());
        break;
      case 0b0011:    // FSUB
        builder_.Gen<x86_64::SubssXRegXReg>(n.machine_reg(), m.machine_reg());
        break;
      case 0b0100:      // FMAX (NaN-propagating: MAXPS|MAXPS|POR)
      case 0b0101: {    // FMIN
        FpRegister tmp = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), m.machine_reg());
        if (args.opcode == 0b0100) {
          builder_.Gen<x86_64::MaxpsXRegXReg>(tmp.machine_reg(), n.machine_reg());
          builder_.Gen<x86_64::MaxpsXRegXReg>(n.machine_reg(), m.machine_reg());
        } else {
          builder_.Gen<x86_64::MinpsXRegXReg>(tmp.machine_reg(), n.machine_reg());
          builder_.Gen<x86_64::MinpsXRegXReg>(n.machine_reg(), m.machine_reg());
        }
        builder_.Gen<x86_64::PorXRegXReg>(n.machine_reg(), tmp.machine_reg());
        break;
      }
      case 0b0110:      // FMAXNM (NaN-suppressing: substitute NaN lane, then MAX/MIN)
      case 0b0111: {    // FMINNM
        FpRegister mask_a = AllocTempSimdReg();
        FpRegister mask_b = AllocTempSimdReg();
        FpRegister an_sub = AllocTempSimdReg();
        FpRegister bn_sub = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(mask_a.machine_reg(), n.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_a.machine_reg(), mask_a.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(mask_b.machine_reg(), m.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_b.machine_reg(), mask_b.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(an_sub.machine_reg(), mask_a.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(an_sub.machine_reg(), m.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(bn_sub.machine_reg(), mask_b.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(bn_sub.machine_reg(), n.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(mask_a.machine_reg(), n.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(mask_b.machine_reg(), m.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), an_sub.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(mask_b.machine_reg(), bn_sub.machine_reg());
        if (args.opcode == 0b0110) {
          builder_.Gen<x86_64::MaxpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
        } else {
          builder_.Gen<x86_64::MinpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
        }
        result = mask_a;
        break;
      }
    }
    if (args.opcode == 0b1000) {
      // FNMUL: flip lane-0 FP32 sign; lanes 1..3 (=+0.0) XOR 0 stay +0.0.
      FpRegister sign = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdXRegReg>(sign.machine_reg(), GetImm(uint64_t{0x80000000u}));
      builder_.Gen<x86_64::PxorXRegXReg>(result.machine_reg(), sign.machine_reg());
    }
    EmitNarrowF32ToHalfAndStore(args.rd, result);
    return;
  }
  if (args.opcode > 0b1000) {
    // Opcodes above FNMUL (0b1000) are reserved.
    UndefinedReturningVoid();
    return;
  }
  const bool is_double = (args.ftype == 0b01);
  FpRegister src1 = GetVRegScalar(args.rn, is_double);
  FpRegister src2 = GetVRegScalar(args.rm, is_double);

  // FMAX/FMIN/FMAXNM/FMINNM (0b0100..0b0111). x86 MAXP{S,D}/MINP{S,D} have
  // ARM-incompatible NaN and signed-zero semantics, so mirror the lite tier's
  // explicit sequences. Packed ops run on the scalar-loaded registers; only
  // lane 0 is written back by SetVRegScalar, so upper-lane garbage is
  // irrelevant. NaN-propagating FMAX/FMIN (ARM: any NaN in -> NaN out) use the
  // symmetric MAX|MAX|POR idiom (the POR keeps a NaN exponent if either input
  // was NaN, and the two-sided MAX makes +-0 order-independent). NaN-
  // suppressing FMAXNM/FMINNM (ARM: exactly one NaN -> the number) substitute
  // each NaN lane with the other operand — via a CMPUNORDP{S,D} self-compare
  // mask — before the MAX/MIN. Mirrors lite_translator.h's scalar path.
  if (args.opcode >= 0b0100 && args.opcode <= 0b0111) {
    const bool is_max = (args.opcode == 0b0100 || args.opcode == 0b0110);
    const bool is_nm = (args.opcode == 0b0110 || args.opcode == 0b0111);
    FpRegister n = src1;
    FpRegister m = src2;
    if (!is_nm) {
      // NaN-propagating: tmp = m; MAXP tmp,n; MAXP n,m; POR n,tmp.
      FpRegister tmp = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), m.machine_reg());
      if (is_max) {
        if (is_double) {
          builder_.Gen<x86_64::MaxpdXRegXReg>(tmp.machine_reg(), n.machine_reg());
          builder_.Gen<x86_64::MaxpdXRegXReg>(n.machine_reg(), m.machine_reg());
        } else {
          builder_.Gen<x86_64::MaxpsXRegXReg>(tmp.machine_reg(), n.machine_reg());
          builder_.Gen<x86_64::MaxpsXRegXReg>(n.machine_reg(), m.machine_reg());
        }
      } else {
        if (is_double) {
          builder_.Gen<x86_64::MinpdXRegXReg>(tmp.machine_reg(), n.machine_reg());
          builder_.Gen<x86_64::MinpdXRegXReg>(n.machine_reg(), m.machine_reg());
        } else {
          builder_.Gen<x86_64::MinpsXRegXReg>(tmp.machine_reg(), n.machine_reg());
          builder_.Gen<x86_64::MinpsXRegXReg>(n.machine_reg(), m.machine_reg());
        }
      }
      builder_.Gen<x86_64::PorXRegXReg>(n.machine_reg(), tmp.machine_reg());
      SetVRegScalar(args.rd, n, is_double);
    } else {
      // NaN-suppressing: mask_a=isnan(n), mask_b=isnan(m). Substitute the NaN
      // lanes with the other operand (an_sub = isnan(n) ? m; bn_sub =
      // isnan(m) ? n), giving sub_n = select(isnan(n), m, n) and sub_m =
      // select(isnan(m), n, m); then MAX/MIN the substituted pair.
      FpRegister mask_a = AllocTempSimdReg();
      FpRegister mask_b = AllocTempSimdReg();
      FpRegister an_sub = AllocTempSimdReg();
      FpRegister bn_sub = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(mask_a.machine_reg(), n.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(mask_b.machine_reg(), m.machine_reg());
      if (is_double) {
        builder_.Gen<x86_64::CmpunordpdXRegXReg>(mask_a.machine_reg(), mask_a.machine_reg());
        builder_.Gen<x86_64::CmpunordpdXRegXReg>(mask_b.machine_reg(), mask_b.machine_reg());
      } else {
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_a.machine_reg(), mask_a.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_b.machine_reg(), mask_b.machine_reg());
      }
      builder_.Gen<x86_64::MovdqaXRegXReg>(an_sub.machine_reg(), mask_a.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(an_sub.machine_reg(), m.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(bn_sub.machine_reg(), mask_b.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(bn_sub.machine_reg(), n.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(mask_a.machine_reg(), n.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(mask_b.machine_reg(), m.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), an_sub.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(mask_b.machine_reg(), bn_sub.machine_reg());
      if (is_max) {
        if (is_double) {
          builder_.Gen<x86_64::MaxpdXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
        } else {
          builder_.Gen<x86_64::MaxpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
        }
      } else {
        if (is_double) {
          builder_.Gen<x86_64::MinpdXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
        } else {
          builder_.Gen<x86_64::MinpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
        }
      }
      SetVRegScalar(args.rd, mask_a, is_double);
    }
    return;
  }

  FpRegister result = AllocTempSimdReg();
  if (is_double) {
    switch (args.opcode) {
      case 0b0000:  // FMUL
      case 0b1000:  // FNMUL: -(n * m) — negate the product below.
        EmitFpBinop<&intrinsics::FMul<Float64>>(result, src1, src2);
        break;
      case 0b0001:  // FDIV
        EmitFpBinop<&intrinsics::FDiv<Float64>>(result, src1, src2);
        break;
      case 0b0010:  // FADD
        EmitFpBinop<&intrinsics::FAdd<Float64>>(result, src1, src2);
        break;
      case 0b0011:  // FSUB
        EmitFpBinop<&intrinsics::FSub<Float64>>(result, src1, src2);
        break;
    }
  } else {
    switch (args.opcode) {
      case 0b0000:  // FMUL
      case 0b1000:  // FNMUL: -(n * m) — negate the product below.
        EmitFpBinop<&intrinsics::FMul<Float32>>(result, src1, src2);
        break;
      case 0b0001:  // FDIV
        EmitFpBinop<&intrinsics::FDiv<Float32>>(result, src1, src2);
        break;
      case 0b0010:  // FADD
        EmitFpBinop<&intrinsics::FAdd<Float32>>(result, src1, src2);
        break;
      case 0b0011:  // FSUB
        EmitFpBinop<&intrinsics::FSub<Float32>>(result, src1, src2);
        break;
    }
  }
  if (args.opcode == 0b1000) {
    // FNMUL: flip the sign bit of the product. Build the sign mask in a fresh
    // GP register (never a forwarded guest value, so the GP->XMM move is
    // conflict-free), move it into an XMM, then XORPD. FP32 masks live in the
    // low 32 bits; upper lanes are irrelevant because SetVRegScalar commits
    // only lane 0. Mirrors lite_translator.h's FpDataProc2 FNMUL path.
    FpRegister sign = AllocTempSimdReg();
    if (is_double) {
      Register gs = std::get<0>(Gen<x86_64::MovqRegImm>(
          static_cast<int64_t>(0x8000000000000000ULL)));
      builder_.Gen<x86_64::MovqXRegReg>(sign.machine_reg(), gs);
    } else {
      Register gs = std::get<0>(Gen<x86_64::MovlRegImm>(
          static_cast<int32_t>(0x80000000u)));
      builder_.Gen<x86_64::MovdXRegReg>(sign.machine_reg(), gs);
    }
    builder_.Gen<x86_64::XorpdXRegXReg>(result.machine_reg(), sign.machine_reg());
  }
  SetVRegScalar(args.rd, result, is_double);
}

// FCADD / FCMLA (complex vector arithmetic), FP32 (.2s/.4s) only. FP16 (size=01)
// and FP64 (size=11) fall back to the lite tier, which handles them. Mirrors
// lite_translator_simd_fp_misc.inc's FP32 AdvSimdFcma: build a per-lane sign mask,
// transform Vm (real/imag swap + selective negation per the rotation), then for
// FCADD add to Vn and for FCMLA add Vd + broadcast(Vn)*Vm_xformed.
void HeavyOptimizerFrontend::AdvSimdFcma(const Decoder::FcmaArgs& args) {
  if (!success()) {
    return;
  }
  if (args.size != 0b10 && args.size != 0b11) {
    UndefinedReturningVoid();
    return;
  }
  const int32_t vn_off = GetVRegOffset(args.rn);
  const int32_t vm_off = GetVRegOffset(args.rm);
  const int32_t vd_off = GetVRegOffset(args.rd);
  const bool is_fcmla = (args.opcode == Decoder::FcmaOpcode::kFcmla);

  if (args.size == 0b11) {
    // FP64 (.2D, Q=1 only — decoder rejects Q=0). One complex pair (lane0=re,
    // lane1=im), 8 bytes/lane. Same rotation algebra as the FP32 path with FP64
    // ops. Mirrors lite_translator_simd_fp_misc.inc's FP64 AdvSimdFcma: SHUFPD 0x01
    // swaps the two doubles; SHUFPD 0x00/0x03 broadcasts lane 0/1; PSRLDQ/PSLLDQ 8
    // builds the single-pair 8-byte lane mask.
    FpRegister xn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);

    // Sign-bit mask for 64-bit FP lanes: [0x8000000000000000]*2. A self-Pcmpeq must
    // start from a zeroed reg to avoid a use-before-def lifetime check.
    FpRegister sign = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
    builder_.Gen<x86_64::PsllqXRegImm>(sign.machine_reg(), int8_t{63});

    // Negate exactly the real (lane0) or imag (lane1) 64-bit slot, then XOR into xm.
    auto negate_lane = [&](bool negate_real) {
      FpRegister lane = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(lane.machine_reg(), lane.machine_reg());
      if (negate_real) {
        builder_.Gen<x86_64::PsrldqXRegImm>(lane.machine_reg(), int8_t{8});  // lane0
      } else {
        builder_.Gen<x86_64::PslldqXRegImm>(lane.machine_reg(), int8_t{8});  // lane1
      }
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), lane.machine_reg());
      builder_.Gen<x86_64::XorpdXRegXReg>(xm.machine_reg(), sign.machine_reg());
    };

    FpRegister result;
    if (!is_fcmla) {
      // FCADD: pair-swap Vm (SHUFPD 0x01 -> [im, re]); rot==0 (#90) negates real,
      // rot==1 (#270) negates imag; then Vn + Vm_xformed.
      builder_.Gen<x86_64::ShufpdXRegXRegImm>(
          xm.machine_reg(), xm.machine_reg(), int8_t{0x01});
      negate_lane(/*negate_real=*/args.rot == 0);
      builder_.Gen<x86_64::AddpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
      result = xn;
    } else {
      // FCMLA: result = Vd + broadcast(Vn) * Vm_xformed.
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      switch (args.rot) {
        case 0:
          break;
        case 1:  // swap + negate real
          builder_.Gen<x86_64::ShufpdXRegXRegImm>(
              xm.machine_reg(), xm.machine_reg(), int8_t{0x01});
          negate_lane(/*negate_real=*/true);
          break;
        case 2:  // negate both lanes
          builder_.Gen<x86_64::XorpdXRegXReg>(xm.machine_reg(), sign.machine_reg());
          break;
        default:  // rot == 3: swap + negate imag
          builder_.Gen<x86_64::ShufpdXRegXRegImm>(
              xm.machine_reg(), xm.machine_reg(), int8_t{0x01});
          negate_lane(/*negate_real=*/false);
          break;
      }
      // Broadcast n_re (rot 0/2, SHUFPD 0x00) or n_im (rot 1/3, SHUFPD 0x03).
      const int8_t bcast =
          (args.rot == 0 || args.rot == 2) ? int8_t{0x00} : int8_t{0x03};
      builder_.Gen<x86_64::ShufpdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), bcast);
      builder_.Gen<x86_64::MulpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::AddpdXRegXReg>(xd.machine_reg(), xn.machine_reg());
      result = xd;
    }
    // FP64 is always Q=1; SetVRegFull(q=true) stores the full 128 bits.
    SetVRegFull(args.rd, result, args.q);
    return;
  }

  FpRegister xn = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
  FpRegister xm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);

  // Sign-bit mask [0x80000000]*4. A self-Pcmpeq must start from a zeroed reg to
  // avoid a use-before-def lifetime check on a fresh temp.
  FpRegister sign = AllocZeroedSimdReg();
  builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
  builder_.Gen<x86_64::PslldXRegImm>(sign.machine_reg(), int8_t{31});

  // Negate exactly the lanes selected by a [-1] mask shifted into either the
  // real (lanes 0,2) or imag (lanes 1,3) 32-bit slots, then XOR into Vm.
  auto negate_lanes = [&](bool negate_real) {
    FpRegister lane = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(lane.machine_reg(), lane.machine_reg());
    if (negate_real) {
      builder_.Gen<x86_64::PsrlqXRegImm>(lane.machine_reg(), int8_t{32});  // lanes 0,2
    } else {
      builder_.Gen<x86_64::PsllqXRegImm>(lane.machine_reg(), int8_t{32});  // lanes 1,3
    }
    builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), lane.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign.machine_reg());
  };

  FpRegister result;
  if (!is_fcmla) {
    // FCADD: swap real/imag of Vm; rot==0 (#90) negates real, rot==1 (#270)
    // negates imag; then Vn + Vm_xformed.
    builder_.Gen<x86_64::ShufpsXRegXRegImm>(
        xm.machine_reg(), xm.machine_reg(), static_cast<int8_t>(0xB1));
    negate_lanes(/*negate_real=*/args.rot == 0);
    builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
    result = xn;
  } else {
    // FCMLA: result = Vd + broadcast(Vn) * Vm_xformed.
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
    switch (args.rot) {
      case 0:
        break;
      case 1:  // swap + negate real
        builder_.Gen<x86_64::ShufpsXRegXRegImm>(
            xm.machine_reg(), xm.machine_reg(), static_cast<int8_t>(0xB1));
        negate_lanes(/*negate_real=*/true);
        break;
      case 2:  // negate both lanes
        builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign.machine_reg());
        break;
      default:  // rot == 3: swap + negate imag
        builder_.Gen<x86_64::ShufpsXRegXRegImm>(
            xm.machine_reg(), xm.machine_reg(), static_cast<int8_t>(0xB1));
        negate_lanes(/*negate_real=*/false);
        break;
    }
    // Broadcast n_re (rot 0/2, PSHUFD 0xA0) or n_im (rot 1/3, PSHUFD 0xF5).
    const int8_t bcast = (args.rot == 0 || args.rot == 2) ? static_cast<int8_t>(0xA0)
                                                          : static_cast<int8_t>(0xF5);
    FpRegister n_bcast =
        FpRegister{std::get<0>(Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), bcast))};
    builder_.Gen<x86_64::MulpsXRegXReg>(n_bcast.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::AddpsXRegXReg>(xd.machine_reg(), n_bcast.machine_reg());
    result = xd;
  }

  // Q=0 (.2s): zero the upper 64 bits of Vd (D-form).
  if (!args.q) {
    FpRegister lane = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(lane.machine_reg(), lane.machine_reg());
    builder_.Gen<x86_64::PsrldqXRegImm>(lane.machine_reg(), int8_t{8});
    builder_.Gen<x86_64::PandXRegXReg>(result.machine_reg(), lane.machine_reg());
  }
  builder_.GenSetSimd<16>(vd_off, result.machine_reg());
}

// FCMLA by-element (Armv8.3-FCMA), FP32 (.4s, Q=1) only. Mirrors the FP32 branch
// of lite_translator_simd_fp_misc.inc's AdvSimdFcmaIdx and the vector
// AdvSimdFcma FCMLA path above, adding a one-PSHUFD broadcast of the indexed
// complex pair (Vm.s[2*index], Vm.s[2*index+1]) before the per-rotation Vm
// transform. FP16 (size==0b01, needs an F16C round-trip) and the reserved
// size values bail to the lite tier, which handles them — matching AdvSimdFcma.
void HeavyOptimizerFrontend::AdvSimdFcmaIdx(const Decoder::FcmaIdxArgs& args) {
  if (!success()) {
    return;
  }
  if (args.size != 0b10) {
    UndefinedReturningVoid();
    return;
  }
  const int32_t vn_off = GetVRegOffset(args.rn);
  const int32_t vm_off = GetVRegOffset(args.rm);
  const int32_t vd_off = GetVRegOffset(args.rd);

  FpRegister xn = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
  FpRegister xm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
  FpRegister xd = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);

  // Broadcast the indexed complex pair across both pair slots:
  //   index==0 -> imm 0x44 = [s0,s1,s0,s1];  index==1 -> imm 0xEE = [s2,s3,s2,s3].
  xm = FpRegister{std::get<0>(Gen<x86_64::PshufdXRegXRegImm>(
      xm.machine_reg(), static_cast<int8_t>(args.index == 0 ? 0x44 : 0xEE)))};

  // Sign-bit mask [0x80000000]*4 (self-Pcmpeq off a zeroed reg to satisfy the
  // use-before-def lifetime check on a fresh temp).
  FpRegister sign = AllocZeroedSimdReg();
  builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
  builder_.Gen<x86_64::PslldXRegImm>(sign.machine_reg(), int8_t{31});

  // Negate the real (lanes 0,2) or imag (lanes 1,3) 32-bit slots of Vm.
  auto negate_lanes = [&](bool negate_real) {
    FpRegister lane = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(lane.machine_reg(), lane.machine_reg());
    if (negate_real) {
      builder_.Gen<x86_64::PsrlqXRegImm>(lane.machine_reg(), int8_t{32});  // lanes 0,2
    } else {
      builder_.Gen<x86_64::PsllqXRegImm>(lane.machine_reg(), int8_t{32});  // lanes 1,3
    }
    builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), lane.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign.machine_reg());
  };

  switch (args.rot) {
    case 0:
      break;
    case 1:  // swap pair + negate real
      builder_.Gen<x86_64::ShufpsXRegXRegImm>(
          xm.machine_reg(), xm.machine_reg(), static_cast<int8_t>(0xB1));
      negate_lanes(/*negate_real=*/true);
      break;
    case 2:  // negate both lanes
      builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign.machine_reg());
      break;
    default:  // rot == 3: swap pair + negate imag
      builder_.Gen<x86_64::ShufpsXRegXRegImm>(
          xm.machine_reg(), xm.machine_reg(), static_cast<int8_t>(0xB1));
      negate_lanes(/*negate_real=*/false);
      break;
  }

  // Broadcast n_re (rot 0/2, PSHUFD 0xA0) or n_im (rot 1/3, PSHUFD 0xF5).
  const int8_t bcast = (args.rot == 0 || args.rot == 2) ? static_cast<int8_t>(0xA0)
                                                        : static_cast<int8_t>(0xF5);
  FpRegister n_bcast =
      FpRegister{std::get<0>(Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), bcast))};
  builder_.Gen<x86_64::MulpsXRegXReg>(n_bcast.machine_reg(), xm.machine_reg());
  builder_.Gen<x86_64::AddpsXRegXReg>(xd.machine_reg(), n_bcast.machine_reg());
  // FP32-indexed FCMLA mandates Q=1 (decoder rejects Q=0), so no D-form clear.
  builder_.GenSetSimd<16>(vd_off, xd.machine_reg());
}

// AdvSIMD BFloat16 three-same-extra (Armv8.6-BF16): BFDOT / BFMMLA /
// BFMLAL{B,T}, vector and by-element. Mirrors
// lite_translator_simd_fp_misc.inc's AdvSimdBf16ThreeSame. Bf16ToFloat(x) is
// just (uint32_t(x) << 16) reinterpreted as FP32 (bit-exact for normals,
// subnormals, zeros, inf, NaN), so every widen is PMOVZXWD+PSLLD $16 (BFDOT/
// BFMMLA) or an in-lane PSLLD/PSRLD (BFMLAL). The host baseline lacks AVX-512-
// BF16, so products go through MULPS and horizontal folds through HADDPS.
void HeavyOptimizerFrontend::AdvSimdBf16ThreeSame(const Decoder::Bf16ThreeSameArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::Bf16ThreeSameOpcode;
  const bool is_bfdot = (args.opcode == Op::kBfdot || args.opcode == Op::kBfdotIdx);
  const bool is_bfmmla = (args.opcode == Op::kBfmmla);
  const bool is_bfmlal =
      (args.opcode == Op::kBfmlalbVec || args.opcode == Op::kBfmlaltVec ||
       args.opcode == Op::kBfmlalbIdx || args.opcode == Op::kBfmlaltIdx);
  if (!is_bfdot && !is_bfmmla && !is_bfmlal) {
    UndefinedReturningVoid();
    return;
  }

  const int32_t vn_off = GetVRegOffset(args.rn);
  const int32_t vm_off = GetVRegOffset(args.rm);
  const int32_t vd_off = GetVRegOffset(args.rd);

  // Widen the low 4 BF16 words of `src` to 4 FP32 lanes (PMOVZXWD then PSLLD $16
  // lifts each 16-bit BF16 into the high half of its dword = Bf16ToFloat).
  auto widen_bf = [&](FpRegister src) -> FpRegister {
    FpRegister r = FpRegister{std::get<0>(Gen<x86_64::PmovzxwdXRegXReg>(src.machine_reg()))};
    builder_.Gen<x86_64::PslldXRegImm>(r.machine_reg(), int8_t{16});
    return r;
  };
  // High 8 bytes of `src` moved into the low 8 (PSHUFD 0x0E).
  auto high_half = [&](FpRegister src) -> FpRegister {
    return FpRegister{std::get<0>(
        Gen<x86_64::PshufdXRegXRegImm>(src.machine_reg(), static_cast<int8_t>(0x0E)))};
  };

  if (is_bfmmla) {
    // 2x2 FP32 output; Vn/Vm are 2x4 BF16 row matrices. Output lane (2i+j) =
    // Vd += dot(Vn_row_i, Vm_row_j) over 4 BF16. Q always 1.
    FpRegister vn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vn.machine_reg(), vn_off);
    FpRegister vm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vm.machine_reg(), vm_off);
    FpRegister n0 = widen_bf(vn);              // Vn row0 (h0..3)
    FpRegister n1 = widen_bf(high_half(vn));   // Vn row1 (h4..7)
    FpRegister m0 = widen_bf(vm);              // Vm row0
    FpRegister m1 = widen_bf(high_half(vm));   // Vm row1

    // Row 0: dots (0,0) and (0,1). Copy n0 so it survives the second MULPS.
    FpRegister r0 = FpRegister{std::get<0>(Gen<x86_64::MovdqaXRegXReg>(n0.machine_reg()))};
    builder_.Gen<x86_64::MulpsXRegXReg>(r0.machine_reg(), m0.machine_reg());
    builder_.Gen<x86_64::MulpsXRegXReg>(n0.machine_reg(), m1.machine_reg());
    builder_.Gen<x86_64::HaddpsXRegXReg>(r0.machine_reg(), n0.machine_reg());
    builder_.Gen<x86_64::HaddpsXRegXReg>(r0.machine_reg(), r0.machine_reg());
    // r0[63:0] = (dot00, dot01).

    // Row 1: dots (1,0) and (1,1).
    FpRegister r1 = FpRegister{std::get<0>(Gen<x86_64::MovdqaXRegXReg>(n1.machine_reg()))};
    builder_.Gen<x86_64::MulpsXRegXReg>(r1.machine_reg(), m0.machine_reg());
    builder_.Gen<x86_64::MulpsXRegXReg>(n1.machine_reg(), m1.machine_reg());
    builder_.Gen<x86_64::HaddpsXRegXReg>(r1.machine_reg(), n1.machine_reg());
    builder_.Gen<x86_64::HaddpsXRegXReg>(r1.machine_reg(), r1.machine_reg());
    // r1[63:0] = (dot10, dot11).

    // Pack: MOVLHPS r0, r1 -> [dot00, dot01, dot10, dot11].
    builder_.Gen<x86_64::MovlhpsXRegXReg>(r0.machine_reg(), r1.machine_reg());
    FpRegister vd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vd.machine_reg(), vd_off);
    builder_.Gen<x86_64::AddpsXRegXReg>(r0.machine_reg(), vd.machine_reg());
    builder_.GenSetSimd<16>(vd_off, r0.machine_reg());
    return;
  }

  if (is_bfmlal) {
    // Per-FP32-lane widening MAC. off = T?1:0 selects odd/even BF16 of each
    // Vn dword lane; Vm is the matching vector lane (B/T) or a single indexed
    // BF16 broadcast. Q always 1 (dest .4s).
    const bool is_t = (args.opcode == Op::kBfmlaltVec || args.opcode == Op::kBfmlaltIdx);
    const bool is_idx = (args.opcode == Op::kBfmlalbIdx || args.opcode == Op::kBfmlaltIdx);

    // Vn widen: keep the wanted BF16 in the high 16 of each dword, clear low 16.
    //   B (off=0): PSLLD $16 lifts the low BF16 to the high half.
    //   T (off=1): PSRLD $16 then PSLLD $16 = AND 0xFFFF0000 (keep high BF16).
    FpRegister xn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    if (is_t) {
      builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{16});
    }
    builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), int8_t{16});

    FpRegister xm;
    if (is_idx) {
      // m = Bf16ToFloat(Vm.h[index]) broadcast to all 4 lanes. Broadcast the
      // dword holding h[index] (index/2), then align by parity: even -> PSLLD
      // (low BF16 to high), odd -> PSRLD+PSLLD (keep high BF16).
      FpRegister vm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(vm.machine_reg(), vm_off);
      const uint8_t d = args.index >> 1;
      xm = FpRegister{std::get<0>(Gen<x86_64::PshufdXRegXRegImm>(
          vm.machine_reg(), static_cast<int8_t>(d * 0x55)))};
      if (args.index & 1) {
        builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{16});
      }
      builder_.Gen<x86_64::PslldXRegImm>(xm.machine_reg(), int8_t{16});
    } else {
      xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      if (is_t) {
        builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{16});
      }
      builder_.Gen<x86_64::PslldXRegImm>(xm.machine_reg(), int8_t{16});
    }

    builder_.Gen<x86_64::MulpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
    FpRegister vd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vd.machine_reg(), vd_off);
    builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), vd.machine_reg());
    builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
    return;
  }

  // BFDOT: each FP32 lane += Bf16(Vn.h[2i])*Bf16(Vm.h[2i]) + Bf16(Vn.h[2i+1])*
  // Bf16(Vm.h[2i+1]). Widen both, MULPS, HADDPS folds adjacent pairs.
  const bool is_idx = (args.opcode == Op::kBfdotIdx);
  FpRegister vn = AllocTempSimdReg();
  builder_.GenGetSimd<16>(vn.machine_reg(), vn_off);
  FpRegister vm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(vm.machine_reg(), vm_off);

  // Widened Vm low operand. Indexed broadcasts the dword pair at `index` first.
  FpRegister m_lo;
  if (is_idx) {
    FpRegister m_bcast = FpRegister{std::get<0>(Gen<x86_64::PshufdXRegXRegImm>(
        vm.machine_reg(), static_cast<int8_t>(args.index * 0x55)))};
    m_lo = widen_bf(m_bcast);
  } else {
    m_lo = widen_bf(vm);
  }
  FpRegister n_lo = widen_bf(vn);
  builder_.Gen<x86_64::MulpsXRegXReg>(n_lo.machine_reg(), m_lo.machine_reg());

  if (args.q) {
    FpRegister n_hi = widen_bf(high_half(vn));
    FpRegister m_hi = is_idx ? m_lo : widen_bf(high_half(vm));
    builder_.Gen<x86_64::MulpsXRegXReg>(n_hi.machine_reg(), m_hi.machine_reg());
    // [p0+p1, p2+p3, p4+p5, p6+p7] = the 4 output lanes.
    builder_.Gen<x86_64::HaddpsXRegXReg>(n_lo.machine_reg(), n_hi.machine_reg());
    FpRegister vd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vd.machine_reg(), vd_off);
    builder_.Gen<x86_64::AddpsXRegXReg>(n_lo.machine_reg(), vd.machine_reg());
    builder_.GenSetSimd<16>(vd_off, n_lo.machine_reg());
  } else {
    // .2s: 2 lanes. HADDPS vs a zeroed reg -> [lane0, lane1, 0, 0]; accumulate
    // Vd, then PUNPCKLQDQ with zero keeps the low 64 and clears the D-form upper.
    FpRegister zero = AllocZeroedSimdReg();
    builder_.Gen<x86_64::HaddpsXRegXReg>(n_lo.machine_reg(), zero.machine_reg());
    FpRegister vd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vd.machine_reg(), vd_off);
    builder_.Gen<x86_64::AddpsXRegXReg>(n_lo.machine_reg(), vd.machine_reg());
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(n_lo.machine_reg(), zero.machine_reg());
    builder_.GenSetSimd<16>(vd_off, n_lo.machine_reg());
  }
}

// I8MM 8-bit matrix multiply-accumulate SMMLA/UMMLA/USMMLA (Q=1 only). Vd is a
// 2x2 int32 matrix, Vn/Vm hold 2 rows of 8 int8. Output lane (2i+j) +=
// dot(Vn row i, Vm row j) over 8 bytes. Mirrors lite's AdvSimdMatMul: widen each
// 8-byte row to 8x int16 (sign per operand — an unsigned byte < 32768 keeps
// PMADDWD's signed 16x16 multiply correct), PMADDWD each (row,col) pairing, then
// three PHADDDs fold the four partial vectors into [d00, d01, d10, d11].
void HeavyOptimizerFrontend::AdvSimdMatMul(const Decoder::MatMulArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::MatMulOpcode;
  const bool n_signed = (args.opcode == Op::kSmmla);
  const bool m_signed = (args.opcode == Op::kSmmla || args.opcode == Op::kUsmmla);
  const int32_t vn_off = GetVRegOffset(args.rn);
  const int32_t vm_off = GetVRegOffset(args.rm);
  const int32_t vd_off = GetVRegOffset(args.rd);

  FpRegister vn = AllocTempSimdReg();
  builder_.GenGetSimd<16>(vn.machine_reg(), vn_off);
  FpRegister vm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(vm.machine_reg(), vm_off);

  // Widen the low 8 bytes of `src` to 8x int16.
  auto widen = [&](FpRegister src, bool is_signed) -> FpRegister {
    if (is_signed) {
      return FpRegister{std::get<0>(Gen<x86_64::PmovsxbwXRegXReg>(src.machine_reg()))};
    }
    return FpRegister{std::get<0>(Gen<x86_64::PmovzxbwXRegXReg>(src.machine_reg()))};
  };
  auto high_half = [&](FpRegister src) -> FpRegister {
    return FpRegister{std::get<0>(
        Gen<x86_64::PshufdXRegXRegImm>(src.machine_reg(), static_cast<int8_t>(0x0E)))};
  };

  FpRegister n0 = widen(vn, n_signed);              // Vn row 0
  FpRegister n1 = widen(high_half(vn), n_signed);   // Vn row 1
  FpRegister m0 = widen(vm, m_signed);              // Vm row 0
  FpRegister m1 = widen(high_half(vm), m_signed);   // Vm row 1

  // p00, p01, p10, p11 (each PMADDWD gives 4 partial int32; PHADDD folds later).
  FpRegister a = FpRegister{std::get<0>(Gen<x86_64::MovdqaXRegXReg>(n0.machine_reg()))};
  builder_.Gen<x86_64::PmaddwdXRegXReg>(a.machine_reg(), m0.machine_reg());   // p00
  FpRegister b = FpRegister{std::get<0>(Gen<x86_64::MovdqaXRegXReg>(n0.machine_reg()))};
  builder_.Gen<x86_64::PmaddwdXRegXReg>(b.machine_reg(), m1.machine_reg());   // p01
  FpRegister c = FpRegister{std::get<0>(Gen<x86_64::MovdqaXRegXReg>(n1.machine_reg()))};
  builder_.Gen<x86_64::PmaddwdXRegXReg>(c.machine_reg(), m0.machine_reg());   // p10
  builder_.Gen<x86_64::PmaddwdXRegXReg>(n1.machine_reg(), m1.machine_reg());  // p11 (in place)

  builder_.Gen<x86_64::PhadddXRegXReg>(a.machine_reg(), b.machine_reg());   // pair-sums p00|p01
  builder_.Gen<x86_64::PhadddXRegXReg>(c.machine_reg(), n1.machine_reg());  // pair-sums p10|p11
  builder_.Gen<x86_64::PhadddXRegXReg>(a.machine_reg(), c.machine_reg());   // [d00,d01,d10,d11]

  FpRegister vd = AllocTempSimdReg();
  builder_.GenGetSimd<16>(vd.machine_reg(), vd_off);
  builder_.Gen<x86_64::PadddXRegXReg>(a.machine_reg(), vd.machine_reg());
  builder_.GenSetSimd<16>(vd_off, a.machine_reg());
}

// SDOT/UDOT (and the I8MM mixed-sign USDOT/SUDOT) 8-bit dot product. Each output
// int32 lane is the sum of 4 signed/unsigned byte products accumulated into Vd.
// Mirrors lite_translator_simd_fp_misc.inc's AdvSimdDotProduct: widen bytes to
// int16 (PMOVSXBW/PMOVZXBW; an unsigned byte stays < 32768 so PMADDWD's signed
// 16x16 multiply gives correct mixed-sign products), PMADDWD to pair-multiply-add,
// PHADDD to fold adjacent pairs into the 4 (or 2) lanes, then accumulate into Vd.
void HeavyOptimizerFrontend::AdvSimdDotProduct(const Decoder::DotProductArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::DotProductOpcode;
  const bool n_signed = (args.opcode == Op::kSdot || args.opcode == Op::kSdotIdx ||
                         args.opcode == Op::kSudotIdx);
  const bool m_signed = (args.opcode == Op::kSdot || args.opcode == Op::kSdotIdx ||
                         args.opcode == Op::kUsdot || args.opcode == Op::kUsdotIdx);
  const bool is_indexed = (args.opcode == Op::kSdotIdx || args.opcode == Op::kUdotIdx ||
                           args.opcode == Op::kUsdotIdx || args.opcode == Op::kSudotIdx);
  const int32_t vn_off = GetVRegOffset(args.rn);
  const int32_t vm_off = GetVRegOffset(args.rm);
  const int32_t vd_off = GetVRegOffset(args.rd);

  FpRegister vn = AllocTempSimdReg();
  builder_.GenGetSimd<16>(vn.machine_reg(), vn_off);
  FpRegister vm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(vm.machine_reg(), vm_off);

  // Widen the low 8 bytes of `src` to 8x int16.
  auto widen = [&](FpRegister src, bool is_signed) -> FpRegister {
    if (is_signed) {
      return FpRegister{std::get<0>(Gen<x86_64::PmovsxbwXRegXReg>(src.machine_reg()))};
    }
    return FpRegister{std::get<0>(Gen<x86_64::PmovzxbwXRegXReg>(src.machine_reg()))};
  };
  // Move the high 8 bytes of `src` into the low 8 bytes of a fresh reg (PSHUFD
  // 0x0E puts the high 64 bits into the low 64).
  auto high_half = [&](FpRegister src) -> FpRegister {
    return FpRegister{std::get<0>(
        Gen<x86_64::PshufdXRegXRegImm>(src.machine_reg(), static_cast<int8_t>(0x0E)))};
  };

  // Widened Vm low operand. The indexed form broadcasts dword `index` (PSHUFD
  // with imm = index replicated into all four 2-bit selectors) before widening.
  FpRegister m_lo;
  if (is_indexed) {
    FpRegister m_bcast = FpRegister{std::get<0>(Gen<x86_64::PshufdXRegXRegImm>(
        vm.machine_reg(), static_cast<int8_t>(args.index * 0x55)))};
    m_lo = widen(m_bcast, m_signed);
  } else {
    m_lo = widen(vm, m_signed);
  }
  FpRegister n_lo = widen(vn, n_signed);
  builder_.Gen<x86_64::PmaddwdXRegXReg>(n_lo.machine_reg(), m_lo.machine_reg());

  if (args.q) {
    FpRegister n_hi = widen(high_half(vn), n_signed);
    FpRegister m_hi = is_indexed ? m_lo : widen(high_half(vm), m_signed);
    builder_.Gen<x86_64::PmaddwdXRegXReg>(n_hi.machine_reg(), m_hi.machine_reg());
    // Fold adjacent int32 pairs: lanes become [n_lo01, n_lo23, n_hi01, n_hi23].
    builder_.Gen<x86_64::PhadddXRegXReg>(n_lo.machine_reg(), n_hi.machine_reg());
    FpRegister vd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vd.machine_reg(), vd_off);
    builder_.Gen<x86_64::PadddXRegXReg>(n_lo.machine_reg(), vd.machine_reg());
    builder_.GenSetSimd<16>(vd_off, n_lo.machine_reg());
  } else {
    // .2s: only 2 lanes. PHADDD n_lo,n_lo folds the low half into [p01, p23, ..].
    builder_.Gen<x86_64::PhadddXRegXReg>(n_lo.machine_reg(), n_lo.machine_reg());
    FpRegister vd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vd.machine_reg(), vd_off);
    builder_.Gen<x86_64::PadddXRegXReg>(n_lo.machine_reg(), vd.machine_reg());
    // D-form zeroes the upper 64 bits of Vd; PUNPCKLQDQ with a zeroed reg keeps
    // the low 64 (the 2 result lanes) and clears the high 64.
    FpRegister zero = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(n_lo.machine_reg(), zero.machine_reg());
    builder_.GenSetSimd<16>(vd_off, n_lo.machine_reg());
  }
}

// AdvSIMD modified immediate: MOVI / MVNI / vector FMOV (replace forms) and
// ORR / BIC (vector, immediate) (read-modify-write forms). The 128-bit value
// is a pure function of (op, cmode, abc, defgh, q) via AdvSIMDExpandImm, so it
// is computed at translation time (ExpandSimdModifiedImmJit, shared with the
// lite translator) and materialized into V[rd].
//
// Materialization uses only allowlisted XMM ops: a PXOR-zeroed XMM, MOVQ to
// load the low 64 bits from a GP reg, and (when the upper half is non-zero and
// Q==1) PINSRQ to insert the high 64 bits. For the D-form (Q==0) only the low
// 64 bits are inserted and the upper half stays zero (PXOR), matching ARM64
// 64-bit-vector write semantics. The commit is a single 16-byte MOVDQA store at
// the v[rd] displacement (GenSetSimd<16>). Mirrors lite_translator.h::Simd-
// ModifiedImm. Bails (FP16 vector FMOV / reserved cmodes) never reach here:
// the decoder rejects them upstream, and the lite tier handles whatever does.
void HeavyOptimizerFrontend::SimdModifiedImm(const Decoder::SimdModifiedImmArgs& args) {
  if (!success()) {
    return;
  }
  const uint8_t cmode = args.cmode;
  // ORR/BIC (vector, immediate): cmode<0>==1 and cmode<3:2>!=11. These read-
  // modify-write V[rd] with the MOVI-style (op=0) expanded immediate: ORR
  // (op=0) sets bits, BIC (op=1) clears them.
  const bool is_orr_bic = (cmode & 1) && ((cmode & 0b1100) != 0b1100);
  __uint128_t value =
      is_orr_bic
          ? ExpandSimdModifiedImmJit(0, cmode, args.abc, args.defgh, args.q)
          : ExpandSimdModifiedImmJit(args.op, cmode, args.abc, args.defgh, args.q);
  const uint64_t lo = static_cast<uint64_t>(value);
  // Q==0 operates on the low 64 bits and zeroes the upper 64 of V[rd].
  const uint64_t hi = args.q ? static_cast<uint64_t>(value >> 64) : 0;
  const int32_t off = GetVRegOffset(args.rd);

  // Build the 128-bit immediate constant into a PXOR-zeroed XMM. For Q==0 the
  // high half is left zero by construction.
  FpRegister ximm = AllocZeroedSimdReg();
  if (lo != 0) {
    Register glo = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(lo)));
    builder_.Gen<x86_64::MovqXRegReg>(ximm.machine_reg(), glo);  // zero-extends upper 64
  }
  if (hi != 0) {
    Register ghi = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(hi)));
    builder_.Gen<x86_64::PinsrqXRegRegImm>(ximm.machine_reg(), ghi, int8_t{1});
  }

  if (!is_orr_bic) {
    // MOVI / MVNI / FMOV: replace V[rd] with the constant.
    builder_.GenSetSimd<16>(off, ximm.machine_reg());
    return;
  }

  // ORR/BIC: load current V[rd] (D-form must zero-extend the low 64 so the
  // upper half ends up zeroed), then OR (set) / AND-NOT (clear) the immediate.
  FpRegister xd = AllocTempSimdReg();
  if (args.q) {
    builder_.GenGetSimd<16>(xd.machine_reg(), off);
  } else {
    // MOVSD reg<-mem zero-extends the upper 64 bits of the XMM. The MOVSD load
    // is one of the SIMD opcodes RemoveLocalGuestContextAccesses recognizes as
    // a guest-context GET, so a prior 16-byte store forwards correctly.
    xd = FpRegister{std::get<0>(Gen<x86_64::MovsdXRegOp>(
        {.base = x86_64::kMachineRegRBP, .disp = off}))};
  }
  if (args.op == 0) {
    builder_.Gen<x86_64::PorXRegXReg>(xd.machine_reg(), ximm.machine_reg());
    builder_.GenSetSimd<16>(off, xd.machine_reg());
  } else {
    // BIC: xd = ~imm & Vd. PANDN(dst, src) computes dst = ~dst & src, so with
    // dst=ximm, src=xd we get ~imm & Vd; the result lands in ximm.
    builder_.Gen<x86_64::PandnXRegXReg>(ximm.machine_reg(), xd.machine_reg());
    builder_.GenSetSimd<16>(off, ximm.machine_reg());
  }
}

void HeavyOptimizerFrontend::SimdLoadLiteral(const Decoder::SimdLoadLiteralArgs& args) {
  UndefinedReturningVoid();
  UNUSED_ARGS(args);
}

// LDR/STR (SIMD&FP, immediate): 128/64/32-bit (Q/D/S). A load zero-extends the
// rest of the 128-bit V[rt] (MOVSD/MOVSS already clear the unused lanes; the
// full 16-byte slot is then committed). The memory access uses the unaligned
// MOVDQU/MOVSD/MOVSS forms (guest memory is not 16-byte aligned) and a recovery
// block so a host fault is delivered to the guest signal handler. The V[rt]
// slot is 16-byte aligned, so its access uses GenGetSimd/GenSetSimd (MOVDQA).
// 8/16-bit (B/H) bail to the lite tier. Mirrors lite_translator.h::SimdLoadStoreImm.
void HeavyOptimizerFrontend::SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& args, Register base) {
  if (!success()) {
    return;
  }
  if (args.size != Decoder::SimdLoadStoreSize::k32bit &&
      args.size != Decoder::SimdLoadStoreSize::k64bit &&
      args.size != Decoder::SimdLoadStoreSize::k128bit) {
    UndefinedReturningVoid(BailReason::kUnsupportedSize);
    return;
  }
  Register masked = ApplyTbi(base);
  const int32_t off = static_cast<int32_t>(args.offset);
  const int32_t vreg_off = GetVRegOffset(args.rt);
  FpRegister xmm = AllocTempSimdReg();
  if (args.is_store) {
    builder_.GenGetSimd<16>(xmm.machine_reg(), vreg_off);
    switch (args.size) {
      case Decoder::SimdLoadStoreSize::k128bit:
        builder_.Gen<x86_64::MovdquOpXReg>({.base = masked, .disp = off}, xmm.machine_reg());
        break;
      case Decoder::SimdLoadStoreSize::k64bit:
        builder_.Gen<x86_64::MovsdOpXReg>({.base = masked, .disp = off}, xmm.machine_reg());
        break;
      default:  // k32bit
        builder_.Gen<x86_64::MovssOpXReg>({.base = masked, .disp = off}, xmm.machine_reg());
        break;
    }
    GenRecoveryBlockForLastInsn();
  } else {
    switch (args.size) {
      case Decoder::SimdLoadStoreSize::k128bit:
        xmm = FpRegister{std::get<0>(Gen<x86_64::MovdquXRegOp>({.base = masked, .disp = off}))};
        break;
      case Decoder::SimdLoadStoreSize::k64bit:
        // MOVSD reg<-mem zero-extends the upper 64 bits of the XMM.
        xmm = FpRegister{std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = masked, .disp = off}))};
        break;
      default:  // k32bit; MOVSS reg<-mem zero-extends the upper 96 bits.
        xmm = FpRegister{std::get<0>(Gen<x86_64::MovssXRegOp>({.base = masked, .disp = off}))};
        break;
    }
    GenRecoveryBlockForLastInsn();
    builder_.GenSetSimd<16>(vreg_off, xmm.machine_reg());
  }
}

// LDP/STP (SIMD&FP): 128-bit Q-pair, 64-bit D-pair, and 32-bit S-pair. The two
// elements sit at [addr] and [addr + element_size] (16/8/4). Each memory access
// is recovery-wrapped. A load zero-extends the unused lanes (MOVDQU is full
// width; MOVSD zeroes bits 127:64; MOVSS zeroes bits 127:32) and the full
// 16-byte v[] slot is committed so the guest register's upper bits read as
// zero. The 64/32-bit D/S pair is the ubiquitous callee-saved FP save/restore
// (ldp/stp d8-d15) in function prologues/epilogues. Mirrors
// lite_translator.h::SimdLoadStorePair.
void HeavyOptimizerFrontend::SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs& args, Register addr) {
  if (!success()) {
    return;
  }
  int32_t element_size;
  switch (args.size) {
    case Decoder::SimdLoadStoreSize::k128bit: element_size = 16; break;
    case Decoder::SimdLoadStoreSize::k64bit: element_size = 8; break;
    case Decoder::SimdLoadStoreSize::k32bit: element_size = 4; break;
    default:
      UndefinedReturningVoid();
      return;
  }
  Register masked = ApplyTbi(addr);
  const int32_t v1_off = GetVRegOffset(args.rt1);
  const int32_t v2_off = GetVRegOffset(args.rt2);
  if (args.is_store) {
    FpRegister xmm1 = AllocTempSimdReg();
    FpRegister xmm2 = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xmm1.machine_reg(), v1_off);
    builder_.GenGetSimd<16>(xmm2.machine_reg(), v2_off);
    switch (args.size) {
      case Decoder::SimdLoadStoreSize::k128bit:
        builder_.Gen<x86_64::MovdquOpXReg>({.base = masked, .disp = 0}, xmm1.machine_reg());
        GenRecoveryBlockForLastInsn();
        builder_.Gen<x86_64::MovdquOpXReg>({.base = masked, .disp = element_size},
                                           xmm2.machine_reg());
        GenRecoveryBlockForLastInsn();
        break;
      case Decoder::SimdLoadStoreSize::k64bit:
        builder_.Gen<x86_64::MovsdOpXReg>({.base = masked, .disp = 0}, xmm1.machine_reg());
        GenRecoveryBlockForLastInsn();
        builder_.Gen<x86_64::MovsdOpXReg>({.base = masked, .disp = element_size},
                                          xmm2.machine_reg());
        GenRecoveryBlockForLastInsn();
        break;
      default:  // k32bit
        builder_.Gen<x86_64::MovssOpXReg>({.base = masked, .disp = 0}, xmm1.machine_reg());
        GenRecoveryBlockForLastInsn();
        builder_.Gen<x86_64::MovssOpXReg>({.base = masked, .disp = element_size},
                                          xmm2.machine_reg());
        GenRecoveryBlockForLastInsn();
        break;
    }
  } else {
    FpRegister xmm1;
    FpRegister xmm2;
    switch (args.size) {
      case Decoder::SimdLoadStoreSize::k128bit:
        xmm1 = FpRegister{std::get<0>(Gen<x86_64::MovdquXRegOp>({.base = masked, .disp = 0}))};
        GenRecoveryBlockForLastInsn();
        xmm2 = FpRegister{
            std::get<0>(Gen<x86_64::MovdquXRegOp>({.base = masked, .disp = element_size}))};
        GenRecoveryBlockForLastInsn();
        break;
      case Decoder::SimdLoadStoreSize::k64bit:
        // MOVSD reg<-mem zero-extends the upper 64 bits of the XMM.
        xmm1 = FpRegister{std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = masked, .disp = 0}))};
        GenRecoveryBlockForLastInsn();
        xmm2 = FpRegister{
            std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = masked, .disp = element_size}))};
        GenRecoveryBlockForLastInsn();
        break;
      default:  // k32bit; MOVSS reg<-mem zero-extends the upper 96 bits.
        xmm1 = FpRegister{std::get<0>(Gen<x86_64::MovssXRegOp>({.base = masked, .disp = 0}))};
        GenRecoveryBlockForLastInsn();
        xmm2 = FpRegister{
            std::get<0>(Gen<x86_64::MovssXRegOp>({.base = masked, .disp = element_size}))};
        GenRecoveryBlockForLastInsn();
        break;
    }
    builder_.GenSetSimd<16>(v1_off, xmm1.machine_reg());
    builder_.GenSetSimd<16>(v2_off, xmm2.machine_reg());
  }
}

// LDR/STR (SIMD&FP, register offset): 128/64/32-bit (Q/D/S). Compute the
// register-offset address (offset extend + shift + base add) with the same
// EmitRegOffsetAddr helper the GP register-offset load/store uses, apply TBI,
// then reuse SimdLoadStoreImm's mem<->xmm<->v[] path at disp=0. A load's
// MOVSD/MOVSS zero-extends the unused lanes and the full 16-byte v[] slot is
// committed; the memory access uses the unaligned MOVDQU/MOVSD/MOVSS forms and
// a recovery block. 8/16-bit (B/H) bail to the lite tier (which covers all
// sizes), matching SimdLoadStoreImm. Mirrors lite_translator.h::SimdLoadStoreReg.
void HeavyOptimizerFrontend::SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs& args, Register base, Register offset) {
  if (!success()) {
    return;
  }
  if (args.size != Decoder::SimdLoadStoreSize::k32bit &&
      args.size != Decoder::SimdLoadStoreSize::k64bit &&
      args.size != Decoder::SimdLoadStoreSize::k128bit) {
    UndefinedReturningVoid(BailReason::kUnsupportedSize);
    return;
  }
  Register addr = EmitRegOffsetAddr(base, offset, args.extend_type, args.shift_amount);
  Register masked = ApplyTbi(addr);
  const int32_t vreg_off = GetVRegOffset(args.rt);
  FpRegister xmm = AllocTempSimdReg();
  if (args.is_store) {
    builder_.GenGetSimd<16>(xmm.machine_reg(), vreg_off);
    switch (args.size) {
      case Decoder::SimdLoadStoreSize::k128bit:
        builder_.Gen<x86_64::MovdquOpXReg>({.base = masked, .disp = 0}, xmm.machine_reg());
        break;
      case Decoder::SimdLoadStoreSize::k64bit:
        builder_.Gen<x86_64::MovsdOpXReg>({.base = masked, .disp = 0}, xmm.machine_reg());
        break;
      default:  // k32bit
        builder_.Gen<x86_64::MovssOpXReg>({.base = masked, .disp = 0}, xmm.machine_reg());
        break;
    }
    GenRecoveryBlockForLastInsn();
  } else {
    switch (args.size) {
      case Decoder::SimdLoadStoreSize::k128bit:
        xmm = FpRegister{std::get<0>(Gen<x86_64::MovdquXRegOp>({.base = masked, .disp = 0}))};
        break;
      case Decoder::SimdLoadStoreSize::k64bit:
        // MOVSD reg<-mem zero-extends the upper 64 bits of the XMM.
        xmm = FpRegister{std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = masked, .disp = 0}))};
        break;
      default:  // k32bit; MOVSS reg<-mem zero-extends the upper 96 bits.
        xmm = FpRegister{std::get<0>(Gen<x86_64::MovssXRegOp>({.base = masked, .disp = 0}))};
        break;
    }
    GenRecoveryBlockForLastInsn();
    builder_.GenSetSimd<16>(vreg_off, xmm.machine_reg());
  }
}

// AdvSIMD copy. DUP (general), INS (general), UMOV, SMOV, and INS (element)
// are implemented in the optimizing tier; DUP (element) still bails to the
// lite translator (its PSHUFB byte-broadcast mask constant / PSHUFD lane
// select don't map cleanly onto the optimizer allowlist). The lane-select
// moves (UMOV/SMOV/INS-element) route through PEXTR/PINSR in the XMM domain
// for MOVDQA store/load-forwarding consistency. Mirrors
// lite_translator.h::AdvSimdCopy.
//
// DUP (general) — broadcast a GP register Rn across all lanes of Vd —
// lowers as follows:
// imm5 encodes the element size: bit0=B(1), bit1=H(2), bit2=S(4), bit3=D(8).
//   B: MOVD Rn->xmm, then PSHUFB with a zeroed mask broadcasts byte 0 to all 16.
//   H: MOVD Rn->xmm (low halfword in lane 0), then PINSRW into all 8 lanes.
//   S: MOVD Rn->xmm, then PINSRD into all 4 lanes.
//   D: MOVQ Rn->xmm (FULL 64 bits — a 32-bit MOVD here would truncate a
//      pointer), then PUNPCKLQDQ self duplicates the low qword to both halves.
// Q==0 forms zero the upper 64 bits of Vd (D-register semantics): for B/H/S
// they are built into a PXOR-zeroed XMM whose upper half is only filled for the
// Q==1 broadcast, and the 64-bit (1D) Q==0 form is ARM-reserved and bails.
void HeavyOptimizerFrontend::AdvSimdCopy(const Decoder::AdvSimdCopyArgs& args) {
  if (!success()) {
    return;
  }
  // INS (general): insert a GP register into one lane of Vd, preserving the
  // others. Read v[rd] as an XMM (GenGetSimd<16>), PINSR the GP value at the
  // lane, and store the full 128 bits back (GenSetSimd<16>) — staying in the
  // XMM domain so the MOVDQA store/load forwarding stays consistent. The lane
  // width and index come from imm5. Mirrors lite_translator.h's INS-general.
  if (args.opcode == Decoder::AdvSimdCopyOpcode::kInsGeneral) {
    const uint8_t imm5 = args.imm5;
    uint8_t esize;
    int8_t lane;
    if (imm5 & 0b00001) {
      esize = 1;
      lane = static_cast<int8_t>((imm5 >> 1) & 0xf);
    } else if (imm5 & 0b00010) {
      esize = 2;
      lane = static_cast<int8_t>((imm5 >> 2) & 0x7);
    } else if (imm5 & 0b00100) {
      esize = 4;
      lane = static_cast<int8_t>((imm5 >> 3) & 0x3);
    } else if (imm5 & 0b01000) {
      esize = 8;
      lane = static_cast<int8_t>((imm5 >> 4) & 0x1);
    } else {
      UndefinedReturningVoid();  // reserved imm5
      return;
    }
    const int32_t off = GetVRegOffset(args.rd);
    FpRegister xmm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xmm.machine_reg(), off);
    Register src = (args.rn < 31) ? GetReg(args.rn)
                                  : std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
    switch (esize) {
      case 1:
        builder_.Gen<x86_64::PinsrbXRegRegImm>(xmm.machine_reg(), src, lane);
        break;
      case 2:
        builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), src, lane);
        break;
      case 4:
        builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), src, lane);
        break;
      default:
        builder_.Gen<x86_64::PinsrqXRegRegImm>(xmm.machine_reg(), src, lane);
        break;
    }
    builder_.GenSetSimd<16>(off, xmm.machine_reg());
    return;
  }

  // UMOV / SMOV / INS (element): lane-select moves. Decode element size and
  // the (destination) lane index from imm5 — the same encoding kInsGeneral
  // uses. All three stay in the XMM domain (GenGetSimd<16> full loads,
  // PEXTR/PINSR, GenSetSimd<16> full stores), NOT a narrow memory access to
  // a v[] sub-lane, so the MOVDQA store/load forwarding the optimizer relies
  // on stays consistent (a partial-width access to a 16-byte v[] slot could
  // be reordered around a wide store/load of the same slot). The lite tier
  // is memory-direct instead — safe there because the single-pass lite
  // translator never reorders.
  if (args.opcode == Decoder::AdvSimdCopyOpcode::kUmov ||
      args.opcode == Decoder::AdvSimdCopyOpcode::kSmov ||
      args.opcode == Decoder::AdvSimdCopyOpcode::kInsElement) {
    const uint8_t imm5_low4 = args.imm5 & 0xf;
    uint8_t esize;
    uint8_t index;
    if (imm5_low4 & 0x1) {
      esize = 1;
      index = (args.imm5 >> 1) & 0xf;
    } else if (imm5_low4 & 0x2) {
      esize = 2;
      index = (args.imm5 >> 2) & 0x7;
    } else if (imm5_low4 & 0x4) {
      esize = 4;
      index = (args.imm5 >> 3) & 0x3;
    } else if (imm5_low4 & 0x8) {
      esize = 8;
      index = (args.imm5 >> 4) & 0x1;
    } else {
      UndefinedReturningVoid();  // reserved imm5
      return;
    }
    const int32_t vn_off =
        GetVRegOffset(args.rn);

    // UMOV Vn.<T>[index] -> Rd (unsigned). Canonical (esize, Q) pairs: B/H/S
    // with Q=0 (Wd), D with Q=1 (Xd). PEXTR zero-extends the extracted
    // element into the GP register (upper bits cleared) — exactly UMOV's
    // unsigned semantics; the 32-bit PEXTR forms clear the upper 32 bits,
    // matching a Wd write. rd==31 (WZR/XZR) discards the result.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kUmov) {
      const bool canonical = (esize == 8 && args.q) || (esize != 8 && !args.q);
      if (!canonical) {
        UndefinedReturningVoid();
        return;
      }
      if (args.rd < 31) {
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        Register res;
        switch (esize) {
          case 1:
            res = std::get<0>(
                Gen<x86_64::PextrbRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
            break;
          case 2:
            res = std::get<0>(
                Gen<x86_64::PextrwRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
            break;
          case 4:
            res = std::get<0>(
                Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
            break;
          default:
            res = std::get<0>(
                Gen<x86_64::PextrqRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
            break;
        }
        SetReg(args.rd, res);
      }
      return;
    }

    // SMOV Vn.<T>[index] -> Rd (signed). Canonical: (1,*), (2,*), (4,Q=1).
    // Extract the raw element (zero-extended by PEXTR) then re-sign-extend
    // the low esize bits to the destination width; the 32-bit Movsx*l forms
    // clear the upper 32 (Wd), the 64-bit Movsx*q forms fill it (Xd). No
    // 32-bit sign-extend-from-memory LIR op exists, so this two-step is why
    // SMOV routes through PEXTR + a RegReg sign-extend.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kSmov) {
      const bool canonical = (esize == 1) || (esize == 2) || (esize == 4 && args.q);
      if (!canonical) {
        UndefinedReturningVoid();
        return;
      }
      if (args.rd < 31) {
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        Register raw;
        if (esize == 4) {
          raw = std::get<0>(
              Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
        } else if (esize == 2) {
          raw = std::get<0>(
              Gen<x86_64::PextrwRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
        } else {
          raw = std::get<0>(
              Gen<x86_64::PextrbRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
        }
        Register res;
        if (esize == 1 && !args.q) {
          res = std::get<0>(Gen<x86_64::MovsxblRegReg>(raw));
        } else if (esize == 1) {
          res = std::get<0>(Gen<x86_64::MovsxbqRegReg>(raw));
        } else if (esize == 2 && !args.q) {
          res = std::get<0>(Gen<x86_64::MovsxwlRegReg>(raw));
        } else if (esize == 2) {
          res = std::get<0>(Gen<x86_64::MovsxwqRegReg>(raw));
        } else /* esize == 4 && q */ {
          res = std::get<0>(Gen<x86_64::MovsxlqRegReg>(raw));
        }
        SetReg(args.rd, res);
      }
      return;
    }

    // INS (element): Vn.<T>[src_index] -> Vd.<T>[index], other Vd lanes kept.
    // src_index comes from imm4, decoded against the same esize. Load both Vd
    // and Vn as full XMMs, PEXTR the source lane into a GP temp, PINSR it into
    // Vd at the dest lane, store Vd back. Loading both before storing is
    // correct for the rd==rn self-INS case (e.g. INS V0.S[0], V0.S[3]). The
    // decoder enforces Q=1, so the full 128-bit Vd is in play.
    uint8_t src_index;
    switch (esize) {
      case 1:
        src_index = args.imm4 & 0xF;
        break;
      case 2:
        src_index = (args.imm4 >> 1) & 0x7;
        break;
      case 4:
        src_index = (args.imm4 >> 2) & 0x3;
        break;
      default /* esize == 8 */:
        src_index = (args.imm4 >> 3) & 0x1;
        break;
    }
    const int32_t vd_off =
        GetVRegOffset(args.rd);
    FpRegister xd = AllocTempSimdReg();
    FpRegister xn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    switch (esize) {
      case 1: {
        Register t = std::get<0>(
            Gen<x86_64::PextrbRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(src_index)));
        builder_.Gen<x86_64::PinsrbXRegRegImm>(
            xd.machine_reg(), t, static_cast<int8_t>(index));
        break;
      }
      case 2: {
        Register t = std::get<0>(
            Gen<x86_64::PextrwRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(src_index)));
        builder_.Gen<x86_64::PinsrwXRegRegImm>(
            xd.machine_reg(), t, static_cast<int8_t>(index));
        break;
      }
      case 4: {
        Register t = std::get<0>(
            Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(src_index)));
        builder_.Gen<x86_64::PinsrdXRegRegImm>(
            xd.machine_reg(), t, static_cast<int8_t>(index));
        break;
      }
      default: {
        Register t = std::get<0>(
            Gen<x86_64::PextrqRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(src_index)));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(
            xd.machine_reg(), t, static_cast<int8_t>(index));
        break;
      }
    }
    builder_.GenSetSimd<16>(vd_off, xd.machine_reg());
    return;
  }

  // DUP (element): broadcast Vn.<T>[index] (one esize-byte element) to every
  // lane of Vd. Q=0 fills the low 64 and zeros the upper 64 (D-register
  // semantics); Q=1 fills all 128. imm5 encodes (esize, index) exactly as
  // UMOV/SMOV/INS-element above. Lowering mirrors lite_translator.h's
  // DUP-element:
  //   esize=1 → PSHUFB with a materialized {idx}×16 byte-index mask.
  //   esize=2 → PSHUFB with a {idx*2, idx*2+1}×8 halfword-index mask.
  //   esize=4 → PSHUFD imm = idx*0x55 (broadcast dword[idx]).
  //   esize=8 → PSHUFD imm 0x44 (idx=0) / 0xEE (idx=1); (8,Q=0)=1D reserved.
  // Both halves of the 128-bit index mask are identical, so it is built
  // qword-at-a-time via MOVQ r->xmm + PUNPCKLQDQ self-broadcast (the same
  // mask-materialization recipe used elsewhere in this file). SetVRegFull with
  // q=false zeros the upper 64 for the D-form. Stays entirely in the XMM
  // domain so MOVDQA store/load forwarding remains consistent.
  if (args.opcode == Decoder::AdvSimdCopyOpcode::kDupElement) {
    const uint8_t imm5_low4 = args.imm5 & 0xf;
    uint8_t esize;
    uint8_t index;
    if (imm5_low4 & 0x1) {
      esize = 1;
      index = (args.imm5 >> 1) & 0xf;
    } else if (imm5_low4 & 0x2) {
      esize = 2;
      index = (args.imm5 >> 2) & 0x7;
    } else if (imm5_low4 & 0x4) {
      esize = 4;
      index = (args.imm5 >> 3) & 0x3;
    } else if (imm5_low4 & 0x8) {
      esize = 8;
      index = (args.imm5 >> 4) & 0x1;
    } else {
      UndefinedReturningVoid();  // reserved imm5
      return;
    }
    // DUP Vd.1D (esize=8, Q=0) is ARM-reserved.
    if (esize == 8 && !args.q) {
      UndefinedReturningVoid();
      return;
    }
    const int32_t vn_off =
        GetVRegOffset(args.rn);
    FpRegister xmm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xmm.machine_reg(), vn_off);
    if (esize == 1 || esize == 2) {
      uint64_t mask_qword;
      if (esize == 1) {
        mask_qword = 0x0101010101010101ULL * static_cast<uint64_t>(index);
      } else {
        const uint64_t b0 = static_cast<uint64_t>(index) * 2;
        const uint64_t pair = ((b0 + 1) << 8) | b0;
        mask_qword = pair | (pair << 16) | (pair << 32) | (pair << 48);
      }
      Register r =
          std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(mask_qword)));
      FpRegister mask = AllocTempSimdReg();
      builder_.Gen<x86_64::MovqXRegReg>(mask.machine_reg(), r);
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(mask.machine_reg(), mask.machine_reg());
      builder_.Gen<x86_64::PshufbXRegXReg>(xmm.machine_reg(), mask.machine_reg());
    } else if (esize == 4) {
      const int8_t imm = static_cast<int8_t>(static_cast<uint8_t>(index * 0x55));
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xmm.machine_reg(), xmm.machine_reg(), imm);
    } else {  // esize == 8, Q=1
      const int8_t imm =
          (index == 0) ? static_cast<int8_t>(0x44) : static_cast<int8_t>(0xEE);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xmm.machine_reg(), xmm.machine_reg(), imm);
    }
    SetVRegFull(args.rd, xmm, args.q);
    return;
  }

  // DUP (scalar): Vd = zero-extend(Vn.element[index]) — e.g. `mov s1,
  // v0.s[1]`, the scalar lane-extract compilers emit around horizontal
  // reductions. Mirrors the lite recipe: PSRLDQ brings the element to byte 0
  // (zero-filling from the top), then a PSLLDQ/PSRLDQ pair clears everything
  // above esize. Stays in the XMM domain end to end.
  if (args.opcode == Decoder::AdvSimdCopyOpcode::kDupScalar) {
    const uint8_t imm5 = args.imm5;
    uint8_t esize;
    uint8_t index;
    if (imm5 & 0b00001) {
      esize = 1;
      index = (imm5 >> 1) & 0xf;
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
      UndefinedReturningVoid();  // reserved imm5
      return;
    }
    FpRegister xmm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xmm.machine_reg(), GetVRegOffset(args.rn));
    const int8_t elem_off = static_cast<int8_t>(index * esize);
    if (elem_off != 0) {
      builder_.Gen<x86_64::PsrldqXRegImm>(xmm.machine_reg(), elem_off);
    }
    const int8_t clear = static_cast<int8_t>(16 - esize);
    builder_.Gen<x86_64::PslldqXRegImm>(xmm.machine_reg(), clear);
    builder_.Gen<x86_64::PsrldqXRegImm>(xmm.machine_reg(), clear);
    builder_.GenSetSimd<16>(GetVRegOffset(args.rd), xmm.machine_reg());
    return;
  }

  if (args.opcode != Decoder::AdvSimdCopyOpcode::kDupGeneral) {
    // Any remaining AdvSimdCopy opcode is not lowered by the optimizing tier;
    // the lite translator handles it.
    UndefinedReturningVoid();
    return;
  }

  const uint8_t esize_bits = args.imm5 & 0xf;
  // 1D (Q==0, esize=D) is ARM-reserved.
  if (esize_bits == 0x08 && !args.q) {
    UndefinedReturningVoid();
    return;
  }

  // Source GP value (XZR/WZR -> 0: a common compiler idiom to zero a vector).
  Register src;
  if (args.rn < 31) {
    src = GetReg(args.rn);
  } else {
    src = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  }

  const int32_t off = GetVRegOffset(args.rd);

  if (esize_bits == 0x08) {  // D (Q==1 only): 2D broadcast.
    FpRegister xmm = AllocTempSimdReg();
    builder_.Gen<x86_64::MovqXRegReg>(xmm.machine_reg(), src);          // low qword = src, upper = 0
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmm.machine_reg(), xmm.machine_reg());  // dup low qword
    builder_.GenSetSimd<16>(off, xmm.machine_reg());
    return;
  }

  if (esize_bits == 0x01) {  // B: byte broadcast.
    FpRegister xmm = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdXRegReg>(xmm.machine_reg(), src);  // byte 0 in lane 0
    // PSHUFB with an all-zero mask selects byte 0 for every output byte.
    FpRegister zero_mask = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PshufbXRegXReg>(xmm.machine_reg(), zero_mask.machine_reg());
    // PSHUFB filled all 16 bytes; for Q==0 we still must zero the upper half.
    SetVRegFull(args.rd, xmm, args.q);
    return;
  }

  if (esize_bits == 0x02) {  // H: halfword broadcast via PINSRW into every lane.
    // Build into a zeroed XMM so the Q==0 upper half stays 0 (only lanes 0..3
    // are written for the D-form; the SetVRegFull merge then drops 4..7 too).
    FpRegister xmm = AllocZeroedSimdReg();
    const int8_t n = args.q ? int8_t{8} : int8_t{4};
    for (int8_t lane = 0; lane < n; ++lane) {
      builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), src, lane);
    }
    SetVRegFull(args.rd, xmm, args.q);
    return;
  }

  // esize_bits == 0x04: S: word broadcast via PINSRD into every lane.
  FpRegister xmm = AllocZeroedSimdReg();
  const int8_t n = args.q ? int8_t{4} : int8_t{2};
  for (int8_t lane = 0; lane < n; ++lane) {
    builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), src, lane);
  }
  SetVRegFull(args.rd, xmm, args.q);
}

// AdvSIMD three-same INTEGER ops that lower to a single packed SSE2/SSE4.1
// instruction: ADD, SUB, AND, ORR, EOR, MUL, and CMEQ (register). Each loads
// the full 128-bit Vn and Vm with GenGetSimd<16> (MOVDQA), runs the packed op
// on the host XMM, and writes the result back with GenSetSimd<16> (MOVDQA).
// The 16-byte aligned MOVDQA access at the v[reg] displacement is the form
// RemoveLocalGuestContextAccesses recognizes for store-to-load forwarding, so
// reads of a just-written V register inside the region see the fresh value.
//
// For the D-form (Q=0) the upper 64 bits of Vd are zeroed: the result's low
// 64 bits are merged (MOVSD reg-reg) into a PXOR-zeroed XMM, exactly the
// single-store discipline SetVRegScalar uses, so the committed MOVDQA holds a
// clean zero-extended 64-bit value.
//
// Element size comes from args.size (00=byte, 01=half, 10=word, 11=double).
// The available packed ops constrain which sizes are handled:
//   ADD: Paddb (8), Paddw (16), Paddd (32), Paddq (64) — all sizes handled.
//   SUB: Psubb (8), Psubw (16), Psubd (32), Psubq (64) — all sizes handled.
//   MUL: Pmullw (16), Pmulld (32). 8-bit and 64-bit have no packed op; bail.
//   AND/ORR/EOR: Pand/Por/Pxor are element-size-independent (one op covers
//     all). ORR with rn==rm is the AdvSIMD MOV (vector) alias and lowers the
//     same way.
//   BSL/BIT/BIF: bitwise-select — element-size-independent Pxor/Pand/Pandn
//     sequences that read Vd; handled in a self-contained pre-switch block.
//   CMEQ: Pcmpeqb (8), Pcmpeqw (16), Pcmpeqd (32). 64-bit (Pcmpeqq) bails.
//   CMGT (signed): Pcmpgtb (8), Pcmpgtw (16), Pcmpgtd (32). 64-bit bails.
//   CMGE (signed >=): NOT(Pcmpgt(Vm, Vn)); CMHI/CMHS (unsigned >, >=):
//     sign-bias both operands then the signed Pcmpgt (+ invert for CMHS).
//     All three are 8/16/32-bit; the 64-bit form needs PCMPGTQ and bails.
//     Handled in a self-contained pre-switch block (result register varies).
// Everything else (saturating, shifts, polynomial, FP, pairwise, widening,
// CMTST, etc.) bails to the lite translator/interpreter.
void HeavyOptimizerFrontend::AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& args) {
  if (!success()) {
    return;
  }
  const int32_t vn_off = GetVRegOffset(args.rn);
  const int32_t vm_off = GetVRegOffset(args.rm);

  // PMUL polynomial multiply (vector, byte lanes). ARM ARM C7.2.219: per-lane
  // carry-less (GF(2)[x]) multiply keeping the low 8 bits. The decoder pins
  // size=00 (.8B/.16B); other sizes are reserved. Mirror of
  // lite_translator.h::AdvSimdThreeSame kPmul's per-bit unrolled SSE2 recipe
  // (no PCLMULQDQ dependency):
  //   acc = 0; bit = 0x01 (byte-replicated)
  //   for i in 0..7:
  //     shifted_a = (a << i) per byte, byte-masked with (0xFF<<i)&0xFF
  //     selector  = ((b & bit) PCMPEQB bit)   -> 0xFF per byte where bit i set
  //     acc      ^= shifted_a & selector
  //     bit      += bit   (PADDB doubles 0x01->0x02->...->0x80, no overflow)
  // PSLLW shifts whole 16-bit lanes, so the low byte's high bits spill into the
  // adjacent byte; the PAND with shift_mask = -bit re-zeros those spilled bits
  // in every byte. Q=0 (.8B) zeroes Vd[127:64] via SetVRegFull's D-form merge.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kPmul) {
    if (args.size != 0b00) {
      UndefinedReturningVoid();  // decoder pins size=00; defensive bail.
      return;
    }
    FpRegister xa = AllocTempSimdReg();
    FpRegister xb = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xa.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xb.machine_reg(), vm_off);
    // acc/zero pre-zeroed (AllocZeroedSimdReg avoids a use-before-def
    // self-PXOR on a fresh temp — see lifetime.h reg_class_ CHECK).
    FpRegister xacc = AllocZeroedSimdReg();
    FpRegister xzero = AllocZeroedSimdReg();
    // bit = 0x01 replicated across all 16 bytes (MOVQ zero-extends, then
    // PUNPCKLQDQ broadcasts the low 64 into the high 64).
    FpRegister xbit = AllocTempSimdReg();
    Register gp = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0101010101010101LL}));
    builder_.Gen<x86_64::MovqXRegReg>(xbit.machine_reg(), gp);
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xbit.machine_reg(), xbit.machine_reg());
    FpRegister xshift = AllocTempSimdReg();
    FpRegister xsel = AllocTempSimdReg();
    for (int i = 0; i < 8; ++i) {
      // shifted_a = (a << i) per byte, byte-masked.
      builder_.Gen<x86_64::MovdqaXRegXReg>(xshift.machine_reg(), xa.machine_reg());
      if (i != 0) {
        builder_.Gen<x86_64::PsllwXRegImm>(xshift.machine_reg(), static_cast<int8_t>(i));
        // shift_mask = -bit per byte = (0xFF<<i)&0xFF.
        builder_.Gen<x86_64::MovdqaXRegXReg>(xsel.machine_reg(), xzero.machine_reg());
        builder_.Gen<x86_64::PsubbXRegXReg>(xsel.machine_reg(), xbit.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xshift.machine_reg(), xsel.machine_reg());
      }
      // selector = (b & bit) PCMPEQB bit -> 0xFF per byte where bit i set.
      builder_.Gen<x86_64::MovdqaXRegXReg>(xsel.machine_reg(), xb.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xsel.machine_reg(), xbit.machine_reg());
      builder_.Gen<x86_64::PcmpeqbXRegXReg>(xsel.machine_reg(), xbit.machine_reg());
      // acc ^= shifted_a & selector.
      builder_.Gen<x86_64::PandXRegXReg>(xshift.machine_reg(), xsel.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xacc.machine_reg(), xshift.machine_reg());
      // Double bit for the next iteration (skip after the last bit).
      if (i != 7) {
        builder_.Gen<x86_64::PaddbXRegXReg>(xbit.machine_reg(), xbit.machine_reg());
      }
    }
    SetVRegFull(args.rd, xacc, args.q);
    return;
  }

  // CMGE/CMHI/CMHS (signed >= / unsigned > / unsigned >=) — handled here as
  // self-contained sequences because their result does not always land in vn
  // (the accumulator the shared switch below assumes) and the unsigned forms
  // need a sign-bias step. x86 has no unsigned vector compare, so CMHI/CMHS
  // flip the per-lane sign bit of both operands (XOR with the width's sign
  // mask) to map the unsigned ordering onto the signed PCMPGT*. CMGE =
  // NOT(Vm > Vn); CMHS = NOT(biased Vm > biased Vn). The .2D (size=11) form
  // uses PCMPGTQ and is gated on host SSE4.2.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmge ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhi ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhs) {
    if (args.size == 0b11 && !host_platform::kHasSSE4_2) {
      UndefinedReturningVoid();  // .2D needs PCMPGTQ.
      return;
    }
    const bool is_unsigned =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhi) ||
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhs);
    // CMHI computes (Vn > Vm) with no invert; CMGE/CMHS compute (Vm > Vn)
    // then invert into (Vn >= Vm) / (Vn >=u Vm) via XOR with all-ones.
    const bool invert =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmge) ||
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhs);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    if (is_unsigned) {
      // Build the per-lane sign-bit mask and XOR it into both operands.
      FpRegister sign = AllocZeroedSimdReg();
      switch (args.size) {
        case 0b00: {
          // 0x80 in every byte (no PSLLB on x86): broadcast from a GPR.
          Register t = std::get<0>(
              Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0x8080808080808080ULL)));
          builder_.Gen<x86_64::MovqXRegReg>(sign.machine_reg(), t);
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(sign.machine_reg(), sign.machine_reg());
          break;
        }
        case 0b01:
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
          builder_.Gen<x86_64::PsllwXRegImm>(sign.machine_reg(), int8_t{15});
          break;
        case 0b10:
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
          builder_.Gen<x86_64::PslldXRegImm>(sign.machine_reg(), int8_t{31});
          break;
        default:  // 0b11
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
          builder_.Gen<x86_64::PsllqXRegImm>(sign.machine_reg(), int8_t{63});
          break;
      }
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), sign.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign.machine_reg());
    }
    // CMHI (no invert): PCMPGT(xn, xm) -> result in xn.
    // CMGE/CMHS (invert): PCMPGT(xm, xn) -> result in xm, then invert.
    FpRegister res = invert ? xm : xn;
    FpRegister a = invert ? xm : xn;
    FpRegister b = invert ? xn : xm;
    switch (args.size) {
      case 0b00:
        builder_.Gen<x86_64::PcmpgtbXRegXReg>(a.machine_reg(), b.machine_reg());
        break;
      case 0b01:
        builder_.Gen<x86_64::PcmpgtwXRegXReg>(a.machine_reg(), b.machine_reg());
        break;
      case 0b10:
        builder_.Gen<x86_64::PcmpgtdXRegXReg>(a.machine_reg(), b.machine_reg());
        break;
      default:  // 0b11
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(a.machine_reg(), b.machine_reg());
        break;
    }
    if (invert) {
      FpRegister ones = AllocOnesSimdReg();
      builder_.Gen<x86_64::PxorXRegXReg>(res.machine_reg(), ones.machine_reg());
    }
    SetVRegFull(args.rd, res, args.q);
    return;
  }

  // FP vector FMAX/FMIN/FMAXNM/FMINNM (.2S/.4S/.2D). x86 MAXP{S,D}/MINP{S,D}
  // disagree with ARM on both NaN and signed-zero results, so mirror
  // lite_translator.h's vector sequence exactly. Everything is computed as
  // FMIN (FMAX = -FMIN(-a,-b)): x86 MINP{S,D} returns the SECOND source on a
  // +-0 tie, which gives ARM's OR-of-signs (most-negative) for FMIN and, after
  // the double negation, AND-of-signs (most-positive) for FMAX. NaN-propagating
  // FMAX/FMIN (any NaN in -> NaN out) use the symmetric MIN|MIN|POR idiom;
  // NaN-suppressing FMAXNM/FMINNM (exactly one NaN -> the number) substitute
  // each NaN lane with the other operand via a CMPUNORDP{S,D} self-compare mask
  // before the MIN. FP16 (is_fp16, needs an F16C round-trip not in the backend
  // gen inputs) bails to lite; the decoder already filters the reserved
  // sz=1&&!Q (.1D) shape, so only .2S/.4S/.2D reach here.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminnmV) {
    if (args.is_fp16) {
      // FP16 .4H/.8H via F16C round-trip. The widen makes this the FP32
      // lowering on the widened halves, so it reuses the FP32/FP64 path's
      // +-0-tie handling below: ARM FMAX/FMAXNM = -FMIN(-a,-b), so negating
      // inputs+output makes MAX reuse the FMIN lowering and get ARM's
      // AND-of-signs (most-positive) on the +-0 tie instead of x86 MAXPS's
      // OR-of-signs. NaN-suppressing FMAXNM/FMINNM substitute each NaN lane
      // with the other operand, then minab|minba|OR (matches the FP32 nm path
      // so the +-0 tie is order-independent). FMIN/FMINNM need no negation.
      if (!host_platform::kHasF16C) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_max = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxV ||
                           args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV);
      const bool is_nm = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV ||
                          args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminnmV);
      FpRegister lo_n = no_fp_register, hi_n = no_fp_register;
      FpRegister lo_m = no_fp_register, hi_m = no_fp_register;
      EmitWidenHalfVec(vn_off, args.q, &lo_n, &hi_n);
      EmitWidenHalfVec(vm_off, args.q, &lo_m, &hi_m);
      // FMAX negate-trick: per-lane FP32 sign mask (all-ones << 31), XORed into
      // the widened inputs now and back out of the result after the MIN.
      FpRegister sign_mask = no_fp_register;
      if (is_max) {
        sign_mask = AllocOnesSimdReg();
        builder_.Gen<x86_64::PslldXRegImm>(sign_mask.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PxorXRegXReg>(lo_n.machine_reg(), sign_mask.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(lo_m.machine_reg(), sign_mask.machine_reg());
        if (args.q) {
          builder_.Gen<x86_64::PxorXRegXReg>(hi_n.machine_reg(), sign_mask.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(hi_m.machine_reg(), sign_mask.machine_reg());
        }
      }
      // Always MIN now (FMAX reaches here with negated operands).
      auto minmax = [&](FpRegister d, FpRegister s) {
        builder_.Gen<x86_64::MinpsXRegXReg>(d.machine_reg(), s.machine_reg());
      };
      // Returns the reg holding the (negated-for-max) min of the pair.
      auto do_half = [&](FpRegister a, FpRegister b) -> FpRegister {
        if (!is_nm) {
          // NaN-propagating: minab|minba|OR.
          FpRegister tmp = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), b.machine_reg());
          minmax(tmp, a);      // min(b, a)
          minmax(a, b);        // min(a, b)
          builder_.Gen<x86_64::PorXRegXReg>(a.machine_reg(), tmp.machine_reg());
          return a;
        }
        // NaN-suppressing: a' = select(isnan(a), b, a); b' = select(isnan(b), a, b),
        // then minab|minba|OR so the +-0 tie is order-independent (OR-of-signs).
        FpRegister ma = AllocTempSimdReg(), mb = AllocTempSimdReg();
        FpRegister as = AllocTempSimdReg(), bs = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(ma.machine_reg(), a.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(ma.machine_reg(), ma.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(mb.machine_reg(), b.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(mb.machine_reg(), mb.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(as.machine_reg(), ma.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(as.machine_reg(), b.machine_reg());   // ma & b
        builder_.Gen<x86_64::MovdqaXRegXReg>(bs.machine_reg(), mb.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(bs.machine_reg(), a.machine_reg());   // mb & a
        builder_.Gen<x86_64::PandnXRegXReg>(ma.machine_reg(), a.machine_reg());  // ~ma & a
        builder_.Gen<x86_64::PandnXRegXReg>(mb.machine_reg(), b.machine_reg());  // ~mb & b
        builder_.Gen<x86_64::PorXRegXReg>(ma.machine_reg(), as.machine_reg());   // a'
        builder_.Gen<x86_64::PorXRegXReg>(mb.machine_reg(), bs.machine_reg());   // b'
        FpRegister t_nm = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_nm.machine_reg(), mb.machine_reg());
        minmax(t_nm, ma);   // min(b', a')
        minmax(ma, mb);     // min(a', b')
        builder_.Gen<x86_64::PorXRegXReg>(ma.machine_reg(), t_nm.machine_reg());
        return ma;
      };
      FpRegister rlo = do_half(lo_n, lo_m);
      FpRegister rhi = args.q ? do_half(hi_n, hi_m) : no_fp_register;
      if (is_max) {
        builder_.Gen<x86_64::PxorXRegXReg>(rlo.machine_reg(), sign_mask.machine_reg());
        if (args.q) {
          builder_.Gen<x86_64::PxorXRegXReg>(rhi.machine_reg(), sign_mask.machine_reg());
        }
      }
      EmitNarrowHalfVecAndStore(args.rd, rlo, rhi, args.q);
      return;
    }
    const bool is_max = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxV ||
                         args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV);
    const bool is_nm = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV ||
                        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminnmV);
    const bool is_double = (args.size & 1);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    // FMAX: negate both inputs so the shared FMIN lowering computes it; the
    // per-lane sign mask (all-ones << 31/63) is XORed in now and, after the
    // MIN, XORed back out of the result.
    FpRegister sign_mask = no_fp_register;
    if (is_max) {
      // AllocZeroedSimdReg establishes a def before the all-ones self-compare
      // (a bare AllocTempSimdReg would trip the lifetime use-before-def CHECK).
      sign_mask = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign_mask.machine_reg(), sign_mask.machine_reg());
      if (is_double) {
        builder_.Gen<x86_64::PsllqXRegImm>(sign_mask.machine_reg(), int8_t{63});
      } else {
        builder_.Gen<x86_64::PslldXRegImm>(sign_mask.machine_reg(), int8_t{31});
      }
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), sign_mask.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign_mask.machine_reg());
    }
    auto gen_min = [&](FpRegister dst, FpRegister src) {
      if (is_double) {
        builder_.Gen<x86_64::MinpdXRegXReg>(dst.machine_reg(), src.machine_reg());
      } else {
        builder_.Gen<x86_64::MinpsXRegXReg>(dst.machine_reg(), src.machine_reg());
      }
    };
    auto gen_cmpunord = [&](FpRegister dst, FpRegister src) {
      if (is_double) {
        builder_.Gen<x86_64::CmpunordpdXRegXReg>(dst.machine_reg(), src.machine_reg());
      } else {
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(dst.machine_reg(), src.machine_reg());
      }
    };
    FpRegister result;
    if (!is_nm) {
      // NaN-propagating: tmp = xm; MIN(tmp, xn); MIN(xn, xm); POR(xn, tmp).
      FpRegister tmp = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), xm.machine_reg());
      gen_min(tmp, xn);
      gen_min(xn, xm);
      builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), tmp.machine_reg());
      result = xn;
    } else {
      // NaN-suppressing: substitute each NaN lane with the other operand
      // (a' = select(isnan(a), b, a); b' = select(isnan(b), a, b)), then MIN.
      FpRegister mask_a = AllocTempSimdReg();
      FpRegister mask_b = AllocTempSimdReg();
      FpRegister an_sub = AllocTempSimdReg();
      FpRegister bn_sub = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(mask_a.machine_reg(), xn.machine_reg());
      gen_cmpunord(mask_a, mask_a);  // 1s where a is NaN
      builder_.Gen<x86_64::MovdqaXRegXReg>(mask_b.machine_reg(), xm.machine_reg());
      gen_cmpunord(mask_b, mask_b);  // 1s where b is NaN
      builder_.Gen<x86_64::MovdqaXRegXReg>(an_sub.machine_reg(), mask_a.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(an_sub.machine_reg(), xm.machine_reg());  // mask_a & b
      builder_.Gen<x86_64::MovdqaXRegXReg>(bn_sub.machine_reg(), mask_b.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(bn_sub.machine_reg(), xn.machine_reg());  // mask_b & a
      builder_.Gen<x86_64::PandnXRegXReg>(mask_a.machine_reg(), xn.machine_reg());  // ~mask_a & a
      builder_.Gen<x86_64::PandnXRegXReg>(mask_b.machine_reg(), xm.machine_reg());  // ~mask_b & b
      builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), an_sub.machine_reg());  // a'
      builder_.Gen<x86_64::PorXRegXReg>(mask_b.machine_reg(), bn_sub.machine_reg());  // b'
      // minab|minba|OR so the +-0 tie gets OR-of-signs (ARM's FMINNM rule);
      // operands are NaN-free here (NaN lanes were substituted above).
      FpRegister t_nm = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_nm.machine_reg(), mask_b.machine_reg());
      gen_min(t_nm, mask_a);      // min(b', a')
      gen_min(mask_a, mask_b);    // min(a', b')
      builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), t_nm.machine_reg());
      result = mask_a;
    }
    // Negate the FMIN result back to obtain FMAX = -FMIN(-a,-b).
    if (is_max) {
      builder_.Gen<x86_64::PxorXRegXReg>(result.machine_reg(), sign_mask.machine_reg());
    }
    // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, result, args.q);
    return;
  }

  // FP vector FADD/FSUB/FMUL/FDIV (.2S/.4S FP32, .2D FP64) lower directly to
  // SSE2 packed FP arithmetic (ADDP{S,D}/SUBP{S,D}/MULP{S,D}/DIVP{S,D}), whose
  // default-rounding IEEE-754 results match ARM's lane-for-lane. Mirrors
  // lite_translator.h's non-FP16 path exactly. FP16 (is_fp16) needs an F16C
  // round-trip absent from the backend Gen inputs and bails to lite; the
  // decoder already rejects the reserved sz=1&&!Q (.1D) shape, so only
  // .2S/.4S/.2D reach here.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFaddV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFsubV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmulV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFdivV) {
    if (args.is_fp16) {
      // FP16 .4H/.8H via F16C round-trip. Widen each operand's halves to FP32,
      // compute packed FP32, narrow once (RNE). Exact for +,-,*,/ (FP32 mantissa
      // strictly contains FP16's). Mirrors lite's FADD/FSUB/FMUL/FDIV FP16 path.
      if (!host_platform::kHasF16C) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister lo_n = no_fp_register, hi_n = no_fp_register;
      FpRegister lo_m = no_fp_register, hi_m = no_fp_register;
      EmitWidenHalfVec(vn_off, args.q, &lo_n, &hi_n);
      EmitWidenHalfVec(vm_off, args.q, &lo_m, &hi_m);
      auto fp_op = [&](FpRegister d, FpRegister s) {
        switch (args.opcode) {
          case Decoder::AdvSimdThreeSameOpcode::kFaddV:
            builder_.Gen<x86_64::AddpsXRegXReg>(d.machine_reg(), s.machine_reg()); break;
          case Decoder::AdvSimdThreeSameOpcode::kFsubV:
            builder_.Gen<x86_64::SubpsXRegXReg>(d.machine_reg(), s.machine_reg()); break;
          case Decoder::AdvSimdThreeSameOpcode::kFmulV:
            builder_.Gen<x86_64::MulpsXRegXReg>(d.machine_reg(), s.machine_reg()); break;
          default:  // kFdivV
            builder_.Gen<x86_64::DivpsXRegXReg>(d.machine_reg(), s.machine_reg()); break;
        }
      };
      fp_op(lo_n, lo_m);
      if (args.q) fp_op(hi_n, hi_m);
      EmitNarrowHalfVecAndStore(args.rd, lo_n, hi_n, args.q);
      return;
    }
    const bool is_double = (args.size & 1);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    if (is_double) {
      switch (args.opcode) {
        case Decoder::AdvSimdThreeSameOpcode::kFaddV:
          builder_.Gen<x86_64::AddpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        case Decoder::AdvSimdThreeSameOpcode::kFsubV:
          builder_.Gen<x86_64::SubpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        case Decoder::AdvSimdThreeSameOpcode::kFmulV:
          builder_.Gen<x86_64::MulpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        default:  // kFdivV
          builder_.Gen<x86_64::DivpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
      }
    } else {
      switch (args.opcode) {
        case Decoder::AdvSimdThreeSameOpcode::kFaddV:
          builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        case Decoder::AdvSimdThreeSameOpcode::kFsubV:
          builder_.Gen<x86_64::SubpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        case Decoder::AdvSimdThreeSameOpcode::kFmulV:
          builder_.Gen<x86_64::MulpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        default:  // kFdivV
          builder_.Gen<x86_64::DivpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
      }
    }
    // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // FP16 vector FMULX (.4H/.8H) via the F16C round-trip. FP32/FP64 vector FMULX
  // is NOT lowered in this tier (it bails to lite->interp) — only the FP16 half
  // is added, guarded on is_fp16. Widen each operand's halves to FP32, apply the
  // packed-single FMULX saturation blend (+-0*+-inf -> +-2.0), narrow once. A
  // single FP16 multiply single-rounds through FP32 and +-2.0 is representable.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmulxV) {
    if (!args.is_fp16 || !host_platform::kHasF16C) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister lo_n = no_fp_register, hi_n = no_fp_register;
    FpRegister lo_m = no_fp_register, hi_m = no_fp_register;
    EmitWidenHalfVec(vn_off, args.q, &lo_n, &hi_n);
    EmitWidenHalfVec(vm_off, args.q, &lo_m, &hi_m);
    FpRegister rlo = EmitFmulxF32Packed(lo_n, lo_m);
    FpRegister rhi = args.q ? EmitFmulxF32Packed(hi_n, hi_m) : no_fp_register;
    EmitNarrowHalfVecAndStore(args.rd, rlo, rhi, args.q);
    return;
  }

  // FP16 vector pairwise FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP (.4H/.8H) via the F16C
  // round-trip. Result lane k reduces the pair (cat[2k], cat[2k+1]) of the
  // concatenation cat = Vn:Vm — Vd's low half comes from Vn's adjacent pairs, its
  // high half from Vm's. FP32/FP64 vector pairwise is not lowered here (bails to
  // lite->interp); only the FP16 half is added, guarded on is_fp16. FADDP folds
  // with HADDPS; min/max forms gather even/odd cat elements with SHUFPS (0x88/
  // 0xDD, same as UZP1/UZP2) and apply the +-0-correct NaN-aware min/max.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFaddpV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxpV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminpV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmpV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminnmpV) {
    if (!args.is_fp16 || !host_platform::kHasF16C) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_add = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFaddpV);
    const bool is_max = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxpV ||
                         args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmpV);
    const bool is_nm = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmpV ||
                        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminnmpV);
    FpRegister lo_n = no_fp_register, hi_n = no_fp_register;
    FpRegister lo_m = no_fp_register, hi_m = no_fp_register;
    EmitWidenHalfVec(vn_off, args.q, &lo_n, &hi_n);
    EmitWidenHalfVec(vm_off, args.q, &lo_m, &hi_m);
    // Reduce one output group of 4 lanes formed by cat = X(lower 4):Y(upper 4).
    auto reduce_group = [&](FpRegister x, FpRegister y) -> FpRegister {
      if (is_add) {
        // HADDPS(r, y) = [x0+x1, x2+x3, y0+y1, y2+y3].
        FpRegister r = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(r.machine_reg(), x.machine_reg());
        builder_.Gen<x86_64::HaddpsXRegXReg>(r.machine_reg(), y.machine_reg());
        return r;
      }
      // evens = [x0,x2,y0,y2] (SHUFPS 0x88); odds = [x1,x3,y1,y3] (SHUFPS 0xDD).
      FpRegister evens = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(evens.machine_reg(), x.machine_reg());
      builder_.Gen<x86_64::ShufpsXRegXRegImm>(evens.machine_reg(), y.machine_reg(),
                                              static_cast<int8_t>(0x88));
      FpRegister odds = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(odds.machine_reg(), x.machine_reg());
      builder_.Gen<x86_64::ShufpsXRegXRegImm>(odds.machine_reg(), y.machine_reg(),
                                              static_cast<int8_t>(0xDD));
      return EmitFpPairwiseMinMaxF32Packed(evens, odds, is_max, is_nm);
    };
    FpRegister rlo, rhi;
    if (!args.q) {
      rlo = reduce_group(lo_n, lo_m);
      rhi = no_fp_register;
    } else {
      rlo = reduce_group(lo_n, hi_n);
      rhi = reduce_group(lo_m, hi_m);
    }
    EmitNarrowHalfVecAndStore(args.rd, rlo, rhi, args.q);
    return;
  }

  // FP vector FMLA/FMLS (.2S/.4S FP32, .2D FP64) — fused multiply-accumulate,
  // single rounding. ARM ARM: FMLA Vd = Vd + Vn*Vm, FMLS Vd = Vd - Vn*Vm, both
  // fused (one rounding). x86 FMA3 packed VFMADD231P{S,D}/VFNMADD231P{S,D}
  // reproduce the single-rounded result lane-for-lane (231 form: dst =
  // src1*src2 (+/-) dst, RMW dst). Mirrors lite_translator.h's non-FP16 path:
  // FP16 (is_fp16) needs an F16C round-trip absent from the heavy Gen inputs and
  // bails to lite; the decoder rejects the reserved sz=1&&!Q (.1D) shape, so only
  // .2S/.4S/.2D reach here. Requires host FMA — bail if absent.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmlaV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmlsV) {
    if (!host_platform::kHasFMA) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_fmls =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmlsV);
    if (args.is_fp16) {
      // FP16 .4H/.8H FMLA/FMLS: correctly-rounded by fusing each pair of widened
      // lanes in FP64 (two 2-lane passes per 4-lane quad) then RNE-narrowing to
      // fp16 — FP64's 53 significand bits exceed 2*11+1, avoiding the double-
      // rounding a plain FP32 round-trip could suffer. Mirrors the lite tier's
      // is_fp16 recipe and the interpreter's FP64 fma(). Bails without F16C.
      if (!host_platform::kHasF16C) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister lo_n = no_fp_register, hi_n = no_fp_register;
      FpRegister lo_m = no_fp_register, hi_m = no_fp_register;
      FpRegister lo_d = no_fp_register, hi_d = no_fp_register;
      EmitWidenHalfVec(vn_off, args.q, &lo_n, &hi_n);
      EmitWidenHalfVec(vm_off, args.q, &lo_m, &hi_m);
      EmitWidenHalfVec(GetVRegOffset(args.rd), args.q, &lo_d, &hi_d);
      // Fuse one 4-FP32-lane quad in FP64 across two 2-lane passes; returns the 4
      // FP32 results. Destroys xn/xm/xd (PSRLDQ walks them to the high pair).
      auto fma_quad = [&](FpRegister xn, FpRegister xm, FpRegister xd) -> FpRegister {
        FpRegister xn_pd = AllocTempSimdReg();
        FpRegister xm_pd = AllocTempSimdReg();
        FpRegister xd_pd = AllocTempSimdReg();
        FpRegister xres = AllocTempSimdReg();
        builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xn_pd.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xm_pd.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xd_pd.machine_reg(), xd.machine_reg());
        if (is_fmls) {
          builder_.Gen<x86_64::Vfnmadd231pdXRegXRegXReg>(
              xd_pd.machine_reg(), xn_pd.machine_reg(), xm_pd.machine_reg());
        } else {
          builder_.Gen<x86_64::Vfmadd231pdXRegXRegXReg>(
              xd_pd.machine_reg(), xn_pd.machine_reg(), xm_pd.machine_reg());
        }
        builder_.Gen<x86_64::Cvtpd2psXRegXReg>(xres.machine_reg(), xd_pd.machine_reg());
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xn_pd.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xm_pd.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PsrldqXRegImm>(xd.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::Cvtps2pdXRegXReg>(xd_pd.machine_reg(), xd.machine_reg());
        if (is_fmls) {
          builder_.Gen<x86_64::Vfnmadd231pdXRegXRegXReg>(
              xd_pd.machine_reg(), xn_pd.machine_reg(), xm_pd.machine_reg());
        } else {
          builder_.Gen<x86_64::Vfmadd231pdXRegXRegXReg>(
              xd_pd.machine_reg(), xn_pd.machine_reg(), xm_pd.machine_reg());
        }
        builder_.Gen<x86_64::Cvtpd2psXRegXReg>(xn.machine_reg(), xd_pd.machine_reg());
        builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PorXRegXReg>(xres.machine_reg(), xn.machine_reg());
        return xres;
      };
      FpRegister rlo = fma_quad(lo_n, lo_m, lo_d);
      FpRegister rhi = args.q ? fma_quad(hi_n, hi_m, hi_d) : no_fp_register;
      EmitNarrowHalfVecAndStore(args.rd, rlo, rhi, args.q);
      return;
    }
    const bool is_double = (args.size & 1);
    const int32_t vd_off =
        GetVRegOffset(args.rd);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
    if (is_fmls) {
      // Vd = Vd + (-Vn)*Vm -- single fused rounding.
      if (is_double) {
        builder_.Gen<x86_64::Vfnmadd231pdXRegXRegXReg>(
            xd.machine_reg(), xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::Vfnmadd231psXRegXRegXReg>(
            xd.machine_reg(), xn.machine_reg(), xm.machine_reg());
      }
    } else {
      if (is_double) {
        builder_.Gen<x86_64::Vfmadd231pdXRegXRegXReg>(
            xd.machine_reg(), xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::Vfmadd231psXRegXRegXReg>(
            xd.machine_reg(), xn.machine_reg(), xm.machine_reg());
      }
    }
    // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, xd, args.q);
    return;
  }

  // FP vector FCMEQ/FCMGE/FCMGT (.2S/.4S FP32, .2D FP64) produce a per-lane
  // all-ones/zero mask. Mirrors lite_translator.h's non-FP16 path: the SSE
  // legacy-encoded CMP{EQ,LT,LE}P{S,D} predicates are ordered, returning FALSE
  // (zero) for any NaN operand, exactly matching ARM's unordered-is-false rule.
  //   FCMEQ: CMPEQP* xn, xm                 -> xn = (xn == xm)
  //   FCMGE: CMPLEP* xm, xn   [result = xm] -> xm = (xm <= xn) == (xn >= xm)
  //   FCMGT: CMPLTP* xm, xn   [result = xm] -> xm = (xm <  xn) == (xn >  xm)
  // FP16 (is_fp16, needs an F16C round-trip absent from the backend Gen inputs)
  // bails to lite; the decoder already rejects the reserved sz=1&&!Q (.1D)
  // shape, so only .2S/.4S/.2D reach here. FACGE/FACGT (abs-compare) first clear
  // the sign bit of both operands (magnitude compare), then run the FCMGE/FCMGT
  // shape. SSE legacy Cmp{eq,le,lt}p{s,d} are ordered — FALSE for any NaN lane —
  // matching ARM (sign-clearing a NaN keeps it NaN, so the unordered lane still
  // returns 0).
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFcmeqV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFcmgeV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFcmgtV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFacgeV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFacgtV) {
    if (args.is_fp16) {
      // FP16 .4H/.8H compares via F16C round-trip: widen halves to FP32, run the
      // ordered 128-bit FP32 compare (0xFFFFFFFF/0 dwords), then PACKSSDW narrow the
      // MASK (signed-saturates -1 -> 0xFFFF and 0 -> 0x0000, exactly ARM's per-lane
      // all-ones/zero bit pattern; unordered -> 0 matches ARM). FACGE/FACGT sign-clear
      // both operands first (magnitude compare). Mirrors lite's FP16 compare path.
      if (!host_platform::kHasF16C) {
        UndefinedReturningVoid();
        return;
      }
      using Op = Decoder::AdvSimdThreeSameOpcode;
      const bool is_eq = (args.opcode == Op::kFcmeqV);
      const bool is_ge = (args.opcode == Op::kFcmgeV || args.opcode == Op::kFacgeV);
      const bool is_abs = (args.opcode == Op::kFacgeV || args.opcode == Op::kFacgtV);
      FpRegister lo_n = no_fp_register, hi_n = no_fp_register;
      FpRegister lo_m = no_fp_register, hi_m = no_fp_register;
      EmitWidenHalfVec(vn_off, args.q, &lo_n, &hi_n);
      EmitWidenHalfVec(vm_off, args.q, &lo_m, &hi_m);
      FpRegister mask = no_fp_register;
      if (is_abs) {
        mask = AllocOnesSimdReg();
        builder_.Gen<x86_64::PsrldXRegImm>(mask.machine_reg(), int8_t{1});  // 0x7FFFFFFF/dword
      }
      // Returns the reg holding the result mask for the pair (a=Vn-half, b=Vm-half).
      auto cmp = [&](FpRegister a, FpRegister b) -> FpRegister {
        if (is_abs) {
          builder_.Gen<x86_64::PandXRegXReg>(a.machine_reg(), mask.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(b.machine_reg(), mask.machine_reg());
        }
        if (is_eq) {
          builder_.Gen<x86_64::CmpeqpsXRegXReg>(a.machine_reg(), b.machine_reg());
          return a;
        } else if (is_ge) {  // (b <= a) == (a >= b); mask lands in b
          builder_.Gen<x86_64::CmplepsXRegXReg>(b.machine_reg(), a.machine_reg());
          return b;
        } else {  // FCMGT/FACGT: (b < a) == (a > b); mask lands in b
          builder_.Gen<x86_64::CmpltpsXRegXReg>(b.machine_reg(), a.machine_reg());
          return b;
        }
      };
      FpRegister rlo = cmp(lo_n, lo_m);
      if (args.q) {
        FpRegister rhi = cmp(hi_n, hi_m);
        builder_.Gen<x86_64::PackssdwXRegXReg>(rlo.machine_reg(), rhi.machine_reg());
        SetVRegFull(args.rd, rlo, true);
      } else {
        builder_.Gen<x86_64::PackssdwXRegXReg>(rlo.machine_reg(), rlo.machine_reg());
        SetVRegFull(args.rd, rlo, false);
      }
      return;
    }
    using Op = Decoder::AdvSimdThreeSameOpcode;
    const bool is_double = (args.size & 1);
    const bool is_eq = (args.opcode == Op::kFcmeqV);
    const bool is_ge = (args.opcode == Op::kFcmgeV || args.opcode == Op::kFacgeV);
    const bool is_abs = (args.opcode == Op::kFacgeV || args.opcode == Op::kFacgtV);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    if (is_abs) {
      // Sign-clear mask: 0x7FFFFFFF per dword (FP32) / 0x7FFF..F per qword (FP64),
      // built with the ones>>1 idiom (see FABD). AllocOnesSimdReg is lifetime-safe.
      FpRegister mask = AllocOnesSimdReg();
      if (is_double) {
        builder_.Gen<x86_64::PsrlqXRegImm>(mask.machine_reg(), int8_t{1});
      } else {
        builder_.Gen<x86_64::PsrldXRegImm>(mask.machine_reg(), int8_t{1});
      }
      builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), mask.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xm.machine_reg(), mask.machine_reg());
    }
    FpRegister result;
    if (is_eq) {
      if (is_double) {
        builder_.Gen<x86_64::CmpeqpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::CmpeqpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      result = xn;
    } else if (is_ge) {
      // (xm <= xn) == (xn >= xm); mask lands in xm.
      if (is_double) {
        builder_.Gen<x86_64::CmplepdXRegXReg>(xm.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::CmplepsXRegXReg>(xm.machine_reg(), xn.machine_reg());
      }
      result = xm;
    } else {  // FCMGT / FACGT
      // (xm < xn) == (xn > xm); mask lands in xm.
      if (is_double) {
        builder_.Gen<x86_64::CmpltpdXRegXReg>(xm.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::CmpltpsXRegXReg>(xm.machine_reg(), xn.machine_reg());
      }
      result = xm;
    }
    // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, result, args.q);
    return;
  }

  // FP vector FABD = |a - b| (.2S/.4S FP32, .2D FP64): a packed SUB then a
  // sign-clear (bitwise AND with 0x7FFF…). ARM FABD is FPAbs(FPSub(a,b)); x86
  // SUBP{S,D} matches ARM FPSub lane-for-lane under default rounding, and
  // clearing the sign bit yields FPAbs — including on NaN, where ARM FPAbs
  // also clears the sign bit. Mirrors lite_translator.h's non-FP16 path. The
  // sign-clear mask (0x7FFFFFFF/dword FP32, 0x7FFFFFFFFFFFFFFF/qword FP64) is
  // built with the PCMPEQD-self ; PSRLD/PSRLQ 1 idiom. FP16 (needs an F16C
  // round-trip absent from the backend Gen inputs) bails to lite; the decoder
  // already rejects the reserved sz=1&&!Q (.1D) shape.
  // SSHL / USHL (vector, 32- and 64-bit lanes). Per-lane variable shift: the
  // amount is the signed low BYTE of each Vm lane, left when non-negative,
  // plain (non-rounding) right when negative -- arithmetic for SSHL, logical
  // for USHL. The bail histogram over real apps put USHL.2D at the top of the
  // actionable list (Chromium's V8/Blink emit it heavily), so unlike the
  // rounding variants below this block covers the 64-bit lanes too.
  //
  // 32-bit lanes use VPSLLVD/VPSRLVD/VPSRAVD directly; their >= 32 count
  // behaviour (zero for the logical forms, sign-fill for VPSRAVD, which
  // saturates the count to 31) matches ARM exactly. 64-bit lanes use
  // VPSLLVQ/VPSRLVQ, with two SSE tricks where AVX2 has no instruction:
  //   * sign-extending the low byte of a qword lane has no PSRAQ-by-imm, so
  //     sext8(b) = (b & 0xFF) ^ 0x80 - 0x80 per lane;
  //   * SSHL's arithmetic right shift has no VPSRAVQ below AVX-512, so
  //     sar(v, s) = ((v ^ m) >>logical s) ^ m with m = (v < 0 ? ~0 : 0) from
  //     PCMPGTQ -- for s >= 64 the logical shift gives 0 and the XOR restores
  //     the sign fill, exactly ARM's semantics.
  // 8/16-bit lanes have no variable-shift instruction below AVX-512 and bail.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSshl ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUshl) {
    const bool is_64bit = (args.size == 0b11);
    if ((args.size != 0b10 && args.size != 0b11) || !host_platform::kHasAVX2) {
      UndefinedReturningVoid();
      return;
    }
    if (is_64bit && !args.q) {
      UndefinedReturningVoid();  // .1D is ARM-reserved.
      return;
    }
    const bool is_signed = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSshl);

    FpRegister xa = AllocTempSimdReg();
    FpRegister xsh = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xa.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xsh.machine_reg(), vm_off);

    if (!is_64bit) {
      // shift = sext8(Vm.lane[7:0]) per dword.
      builder_.Gen<x86_64::PslldXRegImm>(xsh.machine_reg(), int8_t{24});
      builder_.Gen<x86_64::PsradXRegImm>(xsh.machine_reg(), int8_t{24});

      FpRegister xisneg = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpgtdXRegXReg>(xisneg.machine_reg(), xsh.machine_reg());

      FpRegister xleft = AllocTempSimdReg();
      builder_.Gen<x86_64::VpsllvdXRegXRegXReg>(
          xleft.machine_reg(), xa.machine_reg(), xsh.machine_reg());

      FpRegister xs = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PsubdXRegXReg>(xs.machine_reg(), xsh.machine_reg());
      FpRegister xright = AllocTempSimdReg();
      if (is_signed) {
        builder_.Gen<x86_64::VpsravdXRegXRegXReg>(
            xright.machine_reg(), xa.machine_reg(), xs.machine_reg());
      } else {
        builder_.Gen<x86_64::VpsrlvdXRegXRegXReg>(
            xright.machine_reg(), xa.machine_reg(), xs.machine_reg());
      }

      FpRegister xres = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xisneg.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xright.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(xisneg.machine_reg(), xleft.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xres.machine_reg(), xisneg.machine_reg());
      SetVRegFull(args.rd, xres, args.q);
      return;
    }

    // 64-bit lanes. shift = sext8(lane & 0xFF) = ((lane & 0xFF) ^ 0x80) - 0x80.
    Register gp80 = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x80}));
    FpRegister x80 = AllocTempSimdReg();
    builder_.Gen<x86_64::MovqXRegReg>(x80.machine_reg(), gp80);
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(x80.machine_reg(), x80.machine_reg());
    Register gpff = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0xFF}));
    FpRegister xff = AllocTempSimdReg();
    builder_.Gen<x86_64::MovqXRegReg>(xff.machine_reg(), gpff);
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xff.machine_reg(), xff.machine_reg());

    builder_.Gen<x86_64::PandXRegXReg>(xsh.machine_reg(), xff.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(xsh.machine_reg(), x80.machine_reg());
    builder_.Gen<x86_64::PsubqXRegXReg>(xsh.machine_reg(), x80.machine_reg());

    FpRegister xisneg = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PcmpgtqXRegXReg>(xisneg.machine_reg(), xsh.machine_reg());

    FpRegister xleft = AllocTempSimdReg();
    builder_.Gen<x86_64::VpsllvqXRegXRegXReg>(
        xleft.machine_reg(), xa.machine_reg(), xsh.machine_reg());

    FpRegister xs = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PsubqXRegXReg>(xs.machine_reg(), xsh.machine_reg());

    FpRegister xright = AllocTempSimdReg();
    if (is_signed) {
      // sar via sign-smear XOR: m = (a < 0) ? ~0 : 0.
      FpRegister xm = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpgtqXRegXReg>(xm.machine_reg(), xa.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xright.machine_reg(), xa.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xright.machine_reg(), xm.machine_reg());
      FpRegister xshifted = AllocTempSimdReg();
      builder_.Gen<x86_64::VpsrlvqXRegXRegXReg>(
          xshifted.machine_reg(), xright.machine_reg(), xs.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xshifted.machine_reg(), xm.machine_reg());
      xright = xshifted;
    } else {
      builder_.Gen<x86_64::VpsrlvqXRegXRegXReg>(
          xright.machine_reg(), xa.machine_reg(), xs.machine_reg());
    }

    FpRegister xres = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xisneg.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xright.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(xisneg.machine_reg(), xleft.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(xres.machine_reg(), xisneg.machine_reg());
    SetVRegFull(args.rd, xres, /*q=*/true);
    return;
  }

  // SRSHL / SQRSHL (vector, 32-bit lanes only). Per-lane variable signed shift:
  // the amount is the low BYTE of each Vm lane read as a signed value, left when
  // non-negative and rounding-right when negative. SQRSHL additionally saturates
  // the left arm.
  //
  // The lite tier lowers these by looping over lanes in GPRs with real branches,
  // which is correct but slow and cannot be expressed in MachineIR without
  // exploding the CFG. AVX2's per-lane variable shifts make a branchless
  // lowering possible instead, so this is gated on AVX2 and on size=10; the
  // other element widths have no variable-shift instruction below AVX512 and
  // still fall back to lite. .2S/.4S is where this pays: of the SRSHL and SQRSHL
  // encodings harvested from real libraries, 116/128 and 90/128 respectively are
  // 32-bit lanes.
  //
  // Two x86 behaviours make the out-of-range cases fall out for free:
  //   * VPSLLVD zeroes a lane when the (unsigned) count is >= 32, which is what
  //     ARM wants for a left shift that pushes every bit out;
  //   * VPSRAVD saturates the count to 31, so the rounding identity below
  //     collapses to sign_fill + sign_bit == 0 for every |shift| >= 32, matching
  //     ARM's "shift right by at least the element width is zero".
  //
  // The rounding uses the overflow-free identity
  //     (a + (1 << (s-1))) >> s  ==  (a >> s) + ((a >> (s-1)) & 1)
  // because computing the bias directly overflows for a == INT32_MAX, s == 1.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrshl ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrshl) {
    if (args.size != 0b10 || !host_platform::kHasAVX2) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_saturating = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrshl);

    FpRegister xa = AllocTempSimdReg();
    FpRegister xsh = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xa.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xsh.machine_reg(), vm_off);

    // shift = sign-extend(Vm.lane[7:0]) — the ARM shift amount is the low byte
    // of the lane, not the whole lane.
    builder_.Gen<x86_64::PslldXRegImm>(xsh.machine_reg(), int8_t{24});
    builder_.Gen<x86_64::PsradXRegImm>(xsh.machine_reg(), int8_t{24});

    // is_neg = shift < 0, i.e. 0 > shift.
    FpRegister xisneg = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PcmpgtdXRegXReg>(xisneg.machine_reg(), xsh.machine_reg());

    // Left arm: a << shift. Lanes where shift < 0 get a huge unsigned count and
    // are zeroed by VPSLLVD; they are discarded by the select below.
    FpRegister xleft = AllocTempSimdReg();
    builder_.Gen<x86_64::VpsllvdXRegXRegXReg>(
        xleft.machine_reg(), xa.machine_reg(), xsh.machine_reg());

    if (is_saturating) {
      // Overflow iff shifting back does not reproduce the input. This also
      // covers shift >= 32: VPSLLVD zeroed the lane, so the back-shift differs
      // from any non-zero input, and a zero input correctly stays zero.
      FpRegister xback = AllocTempSimdReg();
      builder_.Gen<x86_64::VpsravdXRegXRegXReg>(
          xback.machine_reg(), xleft.machine_reg(), xsh.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xback.machine_reg(), xa.machine_reg());
      // xback is now "no overflow" (all-ones where the round trip matched).

      // saturated = (a >> 31) ^ INT32_MAX: 0 ^ MAX == INT32_MAX for a >= 0,
      // -1 ^ MAX == INT32_MIN for a < 0.
      FpRegister xsat = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(xsat.machine_reg(), xa.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(xsat.machine_reg(), int8_t{31});
      FpRegister xmax = AllocOnesSimdReg();
      builder_.Gen<x86_64::PsrldXRegImm>(xmax.machine_reg(), int8_t{1});  // 0x7FFFFFFF
      builder_.Gen<x86_64::PxorXRegXReg>(xsat.machine_reg(), xmax.machine_reg());

      // left = no_overflow ? left : saturated.
      FpRegister xkeep = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(xkeep.machine_reg(), xback.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xkeep.machine_reg(), xleft.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(xback.machine_reg(), xsat.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xkeep.machine_reg(), xback.machine_reg());
      xleft = xkeep;
    }

    // Right arm: s = -shift, then (a >> s) + ((a >> (s-1)) & 1).
    FpRegister xs = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PsubdXRegXReg>(xs.machine_reg(), xsh.machine_reg());
    FpRegister xsm1 = AllocOnesSimdReg();  // all-ones == -1 per lane
    builder_.Gen<x86_64::PadddXRegXReg>(xsm1.machine_reg(), xs.machine_reg());

    FpRegister xhi = AllocTempSimdReg();
    builder_.Gen<x86_64::VpsravdXRegXRegXReg>(
        xhi.machine_reg(), xa.machine_reg(), xs.machine_reg());
    FpRegister xrb = AllocTempSimdReg();
    builder_.Gen<x86_64::VpsravdXRegXRegXReg>(
        xrb.machine_reg(), xa.machine_reg(), xsm1.machine_reg());
    FpRegister xone = AllocOnesSimdReg();
    builder_.Gen<x86_64::PsrldXRegImm>(xone.machine_reg(), int8_t{31});  // 0x00000001
    builder_.Gen<x86_64::PandXRegXReg>(xrb.machine_reg(), xone.machine_reg());
    builder_.Gen<x86_64::PadddXRegXReg>(xhi.machine_reg(), xrb.machine_reg());

    // result = is_neg ? right : left.
    FpRegister xres = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xisneg.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xhi.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(xisneg.machine_reg(), xleft.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(xres.machine_reg(), xisneg.machine_reg());

    // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, xres, args.q);
    return;
  }

  // FP vector FRECPS/FRSQRTS (.2S/.4S FP32, .2D FP64) — the Newton-Raphson step
  // instructions. ARM ARM: FRECPS Vd = 2.0 - Vn*Vm, FRSQRTS Vd = (3.0 - Vn*Vm)/2,
  // both fused (one rounding), with two architectural special cases:
  //   * if either input is NaN the lane is the default qNaN;
  //   * if the product would be 0*inf (i.e. the multiply itself is the only
  //     source of a NaN), the lane saturates to 2.0 (FRECPS) or 1.5 (FRSQRTS)
  //     rather than propagating that NaN.
  // Detecting the second case is what the two cmpunord masks are for: the
  // product is unordered while the inputs are not. Branchless mask ladder,
  // op-for-op mirror of the lite tier's recipe, so the two tiers round
  // identically. FP16 stays with lite (F16C round-trip, owned by that slice);
  // the decoder rejects the reserved sz=1&&!Q (.1D) shape.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFrecpsV ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kFrsqrtsV) {
    if (args.is_fp16 || !host_platform::kHasFMA) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_double = (args.size & 1);
    if (is_double && !args.q) {
      UndefinedReturningVoid();  // .1D is ARM-reserved.
      return;
    }
    const bool is_frecps = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFrecpsV);

    const int64_t k_fma_bits_d = is_frecps ? int64_t{0x4000000000000000LL}    // +2.0
                                           : int64_t{0x4008000000000000LL};   // +3.0
    const int32_t k_fma_bits_s = is_frecps ? int32_t{0x40000000}              // +2.0
                                           : int32_t{0x40400000};             // +3.0
    const int64_t k_sat_bits_d = is_frecps ? int64_t{0x4000000000000000LL}    // +2.0
                                           : int64_t{0x3FF8000000000000LL};   // +1.5
    const int32_t k_sat_bits_s = is_frecps ? int32_t{0x40000000}              // +2.0
                                           : int32_t{0x3FC00000};             // +1.5
    constexpr int64_t kQnanBitsD = int64_t{0x7FF8000000000000LL};
    constexpr int32_t kQnanBitsS = int32_t{0x7FC00000};
    constexpr int64_t kTwoBitsD = int64_t{0x4000000000000000LL};
    constexpr int32_t kTwoBitsS = int32_t{0x40000000};

    // Broadcasts a lane-width constant across the whole register.
    auto broadcast = [&](int64_t bits_d, int32_t bits_s) -> FpRegister {
      FpRegister x = AllocTempSimdReg();
      if (is_double) {
        builder_.Gen<x86_64::MovqXRegReg>(x.machine_reg(),
                                          GetImm(static_cast<uint64_t>(bits_d)));
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(x.machine_reg(), x.machine_reg());
      } else {
        builder_.Gen<x86_64::MovdXRegReg>(
            x.machine_reg(), GetImm(static_cast<uint64_t>(static_cast<uint32_t>(bits_s))));
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            x.machine_reg(), x.machine_reg(), int8_t{0});
      }
      return x;
    };

    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);

    // input_unord = cmpunord(a, b): all-ones per lane iff either input is NaN.
    FpRegister xiu = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(xiu.machine_reg(), xn.machine_reg());
    if (is_double) {
      builder_.Gen<x86_64::CmpunordpdXRegXReg>(xiu.machine_reg(), xm.machine_reg());
    } else {
      builder_.Gen<x86_64::CmpunordpsXRegXReg>(xiu.machine_reg(), xm.machine_reg());
    }

    // mul_unord = cmpunord(a*b, a*b). Only the NaN-ness of the product is
    // observed, never its value, so the product register is reused in place.
    FpRegister xmul = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(xmul.machine_reg(), xn.machine_reg());
    if (is_double) {
      builder_.Gen<x86_64::MulpdXRegXReg>(xmul.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::CmpunordpdXRegXReg>(xmul.machine_reg(), xmul.machine_reg());
    } else {
      builder_.Gen<x86_64::MulpsXRegXReg>(xmul.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::CmpunordpsXRegXReg>(xmul.machine_reg(), xmul.machine_reg());
    }

    // special = (NOT input_unord) AND mul_unord — the 0*inf lanes. PANDN writes
    // (NOT dst) AND src, so input_unord stays live in xiu for the second select.
    FpRegister xspecial = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(xspecial.machine_reg(), xiu.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(xspecial.machine_reg(), xmul.machine_reg());

    // Normal path: K_fma - a*b in one rounding via VFNMADD231 (dst -= n*m).
    FpRegister xres = broadcast(k_fma_bits_d, k_fma_bits_s);
    if (is_double) {
      builder_.Gen<x86_64::Vfnmadd231pdXRegXRegXReg>(
          xres.machine_reg(), xn.machine_reg(), xm.machine_reg());
    } else {
      builder_.Gen<x86_64::Vfnmadd231psXRegXRegXReg>(
          xres.machine_reg(), xn.machine_reg(), xm.machine_reg());
    }

    // FRSQRTS halves the result. Dividing by exactly 2.0 only decrements the
    // exponent, so the single-rounding invariant carries through.
    if (!is_frecps) {
      FpRegister xtwo = broadcast(kTwoBitsD, kTwoBitsS);
      if (is_double) {
        builder_.Gen<x86_64::DivpdXRegXReg>(xres.machine_reg(), xtwo.machine_reg());
      } else {
        builder_.Gen<x86_64::DivpsXRegXReg>(xres.machine_reg(), xtwo.machine_reg());
      }
    }

    // select 1: special ? K_sat : fma_result.
    FpRegister xsat = broadcast(k_sat_bits_d, k_sat_bits_s);
    builder_.Gen<x86_64::PandXRegXReg>(xsat.machine_reg(), xspecial.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(xspecial.machine_reg(), xres.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(xsat.machine_reg(), xspecial.machine_reg());

    // select 2: input_unord ? qNaN : previous.
    FpRegister xqnan = broadcast(kQnanBitsD, kQnanBitsS);
    builder_.Gen<x86_64::PandXRegXReg>(xqnan.machine_reg(), xiu.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(xiu.machine_reg(), xsat.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(xqnan.machine_reg(), xiu.machine_reg());

    // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, xqnan, args.q);
    return;
  }

  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFabdV) {
    if (args.is_fp16) {
      // FP16 |a-b| via F16C round-trip: widen, packed FP32 SUB, sign-clear (0x7FFFFFFF
      // /dword), narrow. FSUB is exact for a single FP16 narrow; the AND is exact by
      // construction (matches ARM FPAbs, incl. NaN). Mirrors lite's FP16 FABD path.
      if (!host_platform::kHasF16C) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister lo_n = no_fp_register, hi_n = no_fp_register;
      FpRegister lo_m = no_fp_register, hi_m = no_fp_register;
      EmitWidenHalfVec(vn_off, args.q, &lo_n, &hi_n);
      EmitWidenHalfVec(vm_off, args.q, &lo_m, &hi_m);
      FpRegister mask = AllocOnesSimdReg();
      builder_.Gen<x86_64::PsrldXRegImm>(mask.machine_reg(), int8_t{1});  // 0x7FFFFFFF/dword
      builder_.Gen<x86_64::SubpsXRegXReg>(lo_n.machine_reg(), lo_m.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(lo_n.machine_reg(), mask.machine_reg());
      if (args.q) {
        builder_.Gen<x86_64::SubpsXRegXReg>(hi_n.machine_reg(), hi_m.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(hi_n.machine_reg(), mask.machine_reg());
      }
      EmitNarrowHalfVecAndStore(args.rd, lo_n, hi_n, args.q);
      return;
    }
    const bool is_double = (args.size & 1);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    if (is_double) {
      builder_.Gen<x86_64::SubpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
    } else {
      builder_.Gen<x86_64::SubpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
    }
    FpRegister mask = AllocOnesSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::PsrlqXRegImm>(mask.machine_reg(), int8_t{1});
    } else {
      builder_.Gen<x86_64::PsrldXRegImm>(mask.machine_reg(), int8_t{1});
    }
    builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), mask.machine_reg());
    // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // Integer vector byte-lane MUL/MLA/MLS (.16B/.8B, size=00). x86 has no
  // packed 8-bit multiply (no PMULLB), so mirror lite_translator.h's
  // EmitByteMulWiden recipe: widen each 64-bit half of Vn/Vm from bytes to
  // 16-bit words (PMOVZXBW), PMULLW, mask each product to its low byte
  // (PAND 0x00FF x8 so PACKUSWB does not saturate), and PACKUSWB back to
  // bytes. MLA/MLS then PADDB/PSUBB the truncated byte product into Vd at
  // byte width (native mod-256 wrap). The 16/32-bit forms fall through to the
  // MLA/MLS block and the shared PMULLW/PMULLD switch below; the reserved .2D
  // (size=11) form has no packed multiply and bails there. Byte SSHR/USHR
  // heavy above widens the same way with no SSE4.1 gate, so match that.
  if ((args.opcode == Decoder::AdvSimdThreeSameOpcode::kMul ||
       args.opcode == Decoder::AdvSimdThreeSameOpcode::kMla ||
       args.opcode == Decoder::AdvSimdThreeSameOpcode::kMls) &&
      args.size == 0b00) {
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    FpRegister xn_hi = AllocTempSimdReg();
    FpRegister xm_hi = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(xn_hi.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(xm_hi.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PsrldqXRegImm>(xn_hi.machine_reg(), int8_t{8});
    builder_.Gen<x86_64::PsrldqXRegImm>(xm_hi.machine_reg(), int8_t{8});
    builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
    builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
    builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PmullwXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
    // Reuse xm (already defined) as the 0x00FF x8 mask so PACKUSWB does not
    // saturate on the high byte of each 16-bit product lane.
    builder_.Gen<x86_64::PcmpeqbXRegXReg>(xm.machine_reg(), xm.machine_reg());  // -1
    builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});            // 0x00FF x8
    builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(xn_hi.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kMul) {
      // Q=0 (.8B) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xn, args.q);
      return;
    }
    // MLA/MLS accumulate the truncated byte product into Vd at byte width.
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), GetVRegOffset(args.rd));
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kMls) {
      builder_.Gen<x86_64::PsubbXRegXReg>(xd.machine_reg(), xn.machine_reg());
    } else {
      builder_.Gen<x86_64::PaddbXRegXReg>(xd.machine_reg(), xn.machine_reg());
    }
    SetVRegFull(args.rd, xd, args.q);
    return;
  }

  // Integer vector MLA/MLS (multiply-accumulate / multiply-subtract):
  //   MLA: Vd[lane] += Vn[lane] * Vm[lane]
  //   MLS: Vd[lane] -= Vn[lane] * Vm[lane]
  // at .8H/.4H (size=01, PMULLW) and .4S/.2S (size=10, PMULLD). Unlike ADD/SUB
  // above, these READ Vd as the accumulator, so they are handled here rather
  // than in the shared vn-only switch: compute the low-half product Vn*Vm, then
  // PADD (MLA) / PSUB (MLS) it into Vd. Mirrors lite_translator.h's non-byte
  // path exactly (the low 16/32 bits of the product are what ARM keeps). The
  // .16B/.8B (size=00) byte form needs the widen+PMULLW+PACKUSWB recipe and the
  // reserved .2D (size=11) form has no packed 64-bit multiply — both bail to the
  // lite tier, exactly like the heavy kMul path.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kMla ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kMls) {
    if (args.size != 0b01 && args.size != 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_mls = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kMls);
    const int32_t vd_off =
        GetVRegOffset(args.rd);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
    // Low-half product Vn*Vm into xn.
    if (args.size == 0b01) {
      builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
    } else {
      builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
    }
    // Accumulate into / subtract from Vd at the element width.
    if (is_mls) {
      if (args.size == 0b01) {
        builder_.Gen<x86_64::PsubwXRegXReg>(xd.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::PsubdXRegXReg>(xd.machine_reg(), xn.machine_reg());
      }
    } else {
      if (args.size == 0b01) {
        builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
      }
    }
    // Q=0 (.4H/.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, xd, args.q);
    return;
  }

  // Integer vector bitwise-select BSL / BIT / BIF (opcode 0b00011, U=1; the
  // size field selects which of the three). These are element-size-independent
  // bit ops that all READ Vd, so they are handled here (like MLA/MLS) rather
  // than in the shared vn-only switch. Sequences mirror lite_translator.h
  // exactly (Pxor/Pand/Pandn are all already allowlisted; no new LIR ops):
  //   BSL  Vd = (Vd & Vn) | (~Vd & Vm) == ((Vn ^ Vm) & Vd) ^ Vm
  //   BIT  Vd = (Vm & Vn) | (~Vm & Vd) == ((Vn ^ Vd) & Vm) ^ Vd  ("if true")
  //   BIF  Vd = (Vm & Vd) | (~Vm & Vn) == Vd ^ (~Vm & (Vn ^ Vd))  ("if false")
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kBsl ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kBit ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kBif) {
    const int32_t vd_off =
        GetVRegOffset(args.rd);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
    FpRegister res = xn;  // BSL/BIT accumulate into xn; BIF stores xd.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kBsl) {
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());  // Vn^Vm
      builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xd.machine_reg());  // &Vd
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());  // ^Vm
    } else if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kBit) {
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xd.machine_reg());  // Vn^Vd
      builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());  // &Vm
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xd.machine_reg());  // ^Vd
    } else {  // kBif: fold the NOT into PANDN (~xm & xn).
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xd.machine_reg());   // Vn^Vd
      builder_.Gen<x86_64::PandnXRegXReg>(xm.machine_reg(), xn.machine_reg());  // ~Vm & (Vn^Vd)
      builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), xm.machine_reg());   // Vd ^ ...
      res = xd;
    }
    // Q=0 (.8B) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, res, args.q);
    return;
  }

  // Integer vector saturating add/sub SQADD/UQADD/SQSUB/UQSUB (opcode 00001 for
  // add, 00101 for sub; U selects signed/unsigned). The 8-bit (size=00) and
  // 16-bit (size=01) forms have direct SSE2 saturating packed ops; the 32-bit
  // (size=10) forms have no native saturating dword op, so they are emulated
  // exactly as lite_translator.h does (wrap-add/sub + overflow detect +
  // saturate). These read only Vn/Vm (never Vd) but need multiple temps and a
  // per-op emulation sequence, so they are handled here rather than in the
  // shared vn-only switch. The reserved .2D (size=11, 64-bit) form has no
  // packed 64-bit saturating path in either tier and bails to lite (which in
  // turn routes it to the interpreter) — mirroring lite's `size==0b11` bail.
  // The result always lands in xn; SetVRegFull's Q=0 merge zeroes Vd[127:64].
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqadd ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqadd ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqsub ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqsub) {
    if (args.size == 0b11) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqadd) {
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PaddsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else if (args.size == 0b01) {
        builder_.Gen<x86_64::PaddswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        // 32-bit signed saturating add. sum = a + b (wrap); overflow iff
        // ~(a^b) & (a^sum) has its MSB set; sat = (a<0)?INT_MIN:INT_MAX;
        // result = sum ^ ((sum ^ sat) & ovf_mask).
        FpRegister t_sum = AllocTempSimdReg();
        FpRegister t_ovf = AllocTempSimdReg();
        FpRegister t_sat = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_sum.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PadddXRegXReg>(t_sum.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_sum.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());  // -1
        builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());   // ~(a^b)
        builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{31});            // a<0?-1:0
        builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{1});             // 0x7FFFFFFF
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());      // sat
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_sum.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_ovf.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_sum.machine_reg());
      }
    } else if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqadd) {
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PaddusbXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else if (args.size == 0b01) {
        builder_.Gen<x86_64::PadduswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        // 32-bit unsigned saturating add. sum = a + b (wrap); overflow iff
        // sum < a (unsigned), detected via PMAXUD: max(a,sum)==sum iff no
        // overflow; saturate overflowed lanes to UINT32_MAX.
        FpRegister t_save_a = AllocTempSimdReg();
        FpRegister t_ones = AllocZeroedSimdReg();  // def before the self-compare
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_save_a.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xm.machine_reg());       // sum
        builder_.Gen<x86_64::PmaxudXRegXReg>(t_save_a.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(t_save_a.machine_reg(), xn.machine_reg());  // -1 if no ovf
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(t_ones.machine_reg(), t_ones.machine_reg());  // -1
        builder_.Gen<x86_64::PxorXRegXReg>(t_save_a.machine_reg(), t_ones.machine_reg());   // -1 if ovf
        builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), t_save_a.machine_reg());        // saturate
      }
    } else if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqsub) {
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PsubsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else if (args.size == 0b01) {
        builder_.Gen<x86_64::PsubswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        // 32-bit signed saturating sub. diff = a - b (wrap); overflow iff
        // (a^b) & (a^diff) has its MSB set; sat = (a<0)?INT_MIN:INT_MAX;
        // result = diff ^ ((diff ^ sat) & ovf_mask).
        FpRegister t_diff = AllocTempSimdReg();
        FpRegister t_ovf = AllocTempSimdReg();
        FpRegister t_sat = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_diff.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsubdXRegXReg>(t_diff.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());   // a^b
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_diff.machine_reg());  // a^diff
        builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{31});            // a<0?-1:0
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());   // -1
        builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{1});             // 0x7FFFFFFF
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());      // sat
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_ovf.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
      }
    } else {  // kUqsub
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PsubusbXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else if (args.size == 0b01) {
        builder_.Gen<x86_64::PsubuswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        // 32-bit unsigned saturating sub. result = (a>=b) ? a-b : 0. Mask:
        // PMINUD(a,b)==b iff a>=b; AND the wrap-diff with it to zero underflow.
        FpRegister t_mask = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PminudXRegXReg>(t_mask.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(t_mask.machine_reg(), xm.machine_reg());  // -1 if a>=b
        builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());        // a-b
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_mask.machine_reg());     // zero underflow
      }
    }
    // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // Vector saturating-doubling multiply-high SQDMULH / rounding SQRDMULH
  // (opcode 10110; U selects round). Per lane: (2*Vn*Vm) >> esize, with the
  // sole INT_MIN*INT_MIN corner (== -2*minval^2 which would overflow) saturated
  // to INT_MAX. SQRDMULH adds a rounding term of 2^(esize-1) before the shift.
  // Only the 16-bit (size=01) and 32-bit (size=10) lane forms are defined; the
  // decoder reserves size 00/11, so those bail. This is a byte-for-byte mirror
  // of lite_translator.h's kSqdmulh/kSqrdmulh lowering — reads only Vn/Vm, so
  // it lives here (like SQADD) rather than the shared vn-only switch. Result
  // lands in xn (size=01) / xp_lo (size=10); SetVRegFull's Q=0 merge zeroes
  // Vd[127:64].
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqdmulh ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrdmulh) {
    const bool is_round =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrdmulh);
    if (args.size == 0b01) {
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      if (is_round) {
        // SQRDMULH .4H/.8H via PMULHRSW + corner fixup (SSSE3). PMULHRSW
        // computes ((a*b >> 14) + 1) >> 1 = round((2*a*b)/2^16) already, so
        // only the INT16_MIN*INT16_MIN corner needs the ^0xFFFF flip.
        if (!host_platform::kHasSSSE3) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn_corner = AllocTempSimdReg();
        FpRegister xm_corner = AllocTempSimdReg();
        FpRegister x_min = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqw idiom
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn_corner.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(xm_corner.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(x_min.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});  // INT16_MIN
        builder_.Gen<x86_64::PmulhrswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_corner.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_corner.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn_corner.machine_reg(), xm_corner.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_corner.machine_reg());
        SetVRegFull(args.rd, xn, args.q);
        return;
      }
      // SQDMULH .4H/.8H via PMULHW + PMULLW combine + corner fixup (SSE2).
      // high16(2*a*b) = (high16(a*b) << 1) | (top bit of low16(a*b)).
      FpRegister xn_lo = AllocTempSimdReg();
      FpRegister xn_corner = AllocTempSimdReg();
      FpRegister xm_corner = AllocTempSimdReg();
      FpRegister x_min = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqw idiom
      builder_.Gen<x86_64::MovdqaXRegXReg>(xn_corner.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xm_corner.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xn_lo.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PmullwXRegXReg>(xn_lo.machine_reg(), xm.machine_reg());  // low16(a*b)
      builder_.Gen<x86_64::PmulhwXRegXReg>(xn.machine_reg(), xm.machine_reg());     // high16(a*b)
      builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{1});              // high<<1
      builder_.Gen<x86_64::PsrlwXRegImm>(xn_lo.machine_reg(), int8_t{15});          // low top bit
      builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xn_lo.machine_reg());     // high16(2*a*b)
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(x_min.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});          // INT16_MIN
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_corner.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_corner.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xn_corner.machine_reg(), xm_corner.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_corner.machine_reg());  // INT16_MIN^0xFFFF=MAX
      SetVRegFull(args.rd, xn, args.q);
      return;
    }
    if (args.size == 0b10) {
      // size=10 .2S/.4S: PMULDQ widen (SSE4.1) + PSLLQ double + optional round
      // + corner fixup; PSHUFD 0xDD lifts each product's upper 32 bits.
      if (!host_platform::kHasSSE4_1) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      FpRegister x_const = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqd idiom
      FpRegister corner = AllocTempSimdReg();
      FpRegister xp_lo = AllocTempSimdReg();
      FpRegister xp_hi = AllocTempSimdReg();
      FpRegister xm_hi = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      // x_const = INT32_MIN broadcast across 4 dwords.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(x_const.machine_reg(), int8_t{31});
      // Corner: lanes where Vn.s[i] == INT32_MIN AND Vm.s[i] == INT32_MIN.
      builder_.Gen<x86_64::MovdqaXRegXReg>(corner.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(corner.machine_reg(), x_const.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xp_lo.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xp_lo.machine_reg(), x_const.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(corner.machine_reg(), xp_lo.machine_reg());
      // Two PMULDQs reconstruct the 4 signed 32x32 -> 64 products (even lanes
      // in xp_lo, odd lanes in xp_hi).
      builder_.Gen<x86_64::MovdqaXRegXReg>(xp_lo.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PmuldqXRegXReg>(xp_lo.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xp_hi.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsrlqXRegImm>(xp_hi.machine_reg(), int8_t{32});
      builder_.Gen<x86_64::MovdqaXRegXReg>(xm_hi.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PsrlqXRegImm>(xm_hi.machine_reg(), int8_t{32});
      builder_.Gen<x86_64::PmuldqXRegXReg>(xp_hi.machine_reg(), xm_hi.machine_reg());
      // Double each 64-bit signed product.
      builder_.Gen<x86_64::PsllqXRegImm>(xp_lo.machine_reg(), int8_t{1});
      builder_.Gen<x86_64::PsllqXRegImm>(xp_hi.machine_reg(), int8_t{1});
      if (is_round) {
        // SQRDMULH: add rounding constant 2^31 = 0x80000000 per qword.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
        builder_.Gen<x86_64::PsllqXRegImm>(x_const.machine_reg(), int8_t{63});
        builder_.Gen<x86_64::PsrlqXRegImm>(x_const.machine_reg(), int8_t{32});
        builder_.Gen<x86_64::PaddqXRegXReg>(xp_lo.machine_reg(), x_const.machine_reg());
        builder_.Gen<x86_64::PaddqXRegXReg>(xp_hi.machine_reg(), x_const.machine_reg());
      }
      // Extract the upper 32 bits of each 64-bit lane and interleave.
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xp_lo.machine_reg(), xp_lo.machine_reg(), static_cast<int8_t>(0xDD));
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xp_hi.machine_reg(), xp_hi.machine_reg(), static_cast<int8_t>(0xDD));
      builder_.Gen<x86_64::PunpckldqXRegXReg>(xp_lo.machine_reg(), xp_hi.machine_reg());
      // Apply corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX.
      builder_.Gen<x86_64::PxorXRegXReg>(xp_lo.machine_reg(), corner.machine_reg());
      SetVRegFull(args.rd, xp_lo, args.q);
      return;
    }
    // size=00 and size=11 reserved by the decoder; bail to lite/interp.
    UndefinedReturningVoid();
    return;
  }

  // Integer vector S{MAX,MIN}/U{MAX,MIN} (opcode 01100 max / 01101 min; U
  // selects signed/unsigned): lane-wise signed or unsigned min/max. x86 has
  // direct lane-width-matched PMAXS/PMINS/PMAXU/PMINU for 8/16/32-bit lanes;
  // the .2D (size=11) form has no SSE-era 64-bit min/max (PMAXSQ/… need
  // AVX-512F-VL) and bails to lite. Mirrors lite_translator.h's SMAX/… block
  // exactly, including the per-size SSE4.1 gate (PMAXSB/PMINSB/PMAXSD/PMINSD/
  // PMAXUW/PMINUW/PMAXUD/PMINUD need SSE4.1; PMAXSW/PMINSW/PMAXUB/PMINUB are
  // SSE2). The result lands in xn; SetVRegFull's Q=0 merge zeroes Vd[127:64].
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmin) {
    if (args.size == 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_max = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
                         args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax);
    const bool is_signed = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
                            args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin);
    const bool needs_sse4_1 = (is_signed && args.size == 0b00) ||
                              (is_signed && args.size == 0b10) ||
                              (!is_signed && args.size == 0b01) ||
                              (!is_signed && args.size == 0b10);
    if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    switch (args.size) {
      case 0b00:  // .16B / .8B
        if (is_signed) {
          if (is_max) builder_.Gen<x86_64::PmaxsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          if (is_max) builder_.Gen<x86_64::PmaxubXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminubXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
      case 0b01:  // .8H / .4H
        if (is_signed) {
          if (is_max) builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          if (is_max) builder_.Gen<x86_64::PmaxuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
      default:  // 0b10: .4S / .2S
        if (is_signed) {
          if (is_max) builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          if (is_max) builder_.Gen<x86_64::PmaxudXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
    }
    // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // SABD/UABD (absolute difference) and SABA/UABA (absolute difference then
  // accumulate into Vd): per-lane |Vn - Vm|. Mirrors lite_translator.h's
  // kSabd/kUabd/kSaba/kUaba block exactly. ARM computes the difference in
  // extended precision then truncates the absolute value; a naive PSUB +
  // sign-mask abs mismatches on the INT_MIN-vs-INT_MAX wrap. The correct
  // modular-arithmetic recipe is per-lane max(a,b) - min(a,b) on the
  // signed (SABD/SABA) or unsigned (UABD/UABA) interpretation, so it reuses
  // the same PMAXS/PMINS/PMAXU/PMINU packed ops as the min/max block above,
  // then PSUB (and PADD into Vd for the *ABA accumulate). The .2D (size=11)
  // form is reserved by the ARM ARM and bails. Per-size SSE4.1 gate matches
  // the min/max block (PMAXSB/PMINSB/PMAXSD/PMINSD/PMAXUW/PMINUW/PMAXUD/
  // PMINUD need SSE4.1; PMAXSW/PMINSW/PMAXUB/PMINUB are SSE2).
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSabd ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUabd ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kSaba ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUaba) {
    if (args.size == 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_signed = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSabd ||
                            args.opcode == Decoder::AdvSimdThreeSameOpcode::kSaba);
    const bool is_accum = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSaba ||
                           args.opcode == Decoder::AdvSimdThreeSameOpcode::kUaba);
    const bool needs_sse4_1 = (is_signed && args.size == 0b00) ||
                              (is_signed && args.size == 0b10) ||
                              (!is_signed && args.size == 0b01) ||
                              (!is_signed && args.size == 0b10);
    if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    FpRegister xmax = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    // xmax = max(Vn, Vm); xn = min(Vn, Vm); xmax -= xn == |Vn - Vm|.
    builder_.Gen<x86_64::MovdqaXRegXReg>(xmax.machine_reg(), xn.machine_reg());
    switch (args.size) {
      case 0b00:  // .16B / .8B
        if (is_signed) {
          builder_.Gen<x86_64::PmaxsbXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmaxubXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminubXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        builder_.Gen<x86_64::PsubbXRegXReg>(xmax.machine_reg(), xn.machine_reg());
        break;
      case 0b01:  // .8H / .4H
        if (is_signed) {
          builder_.Gen<x86_64::PmaxswXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmaxuwXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        builder_.Gen<x86_64::PsubwXRegXReg>(xmax.machine_reg(), xn.machine_reg());
        break;
      default:  // 0b10: .4S / .2S
        if (is_signed) {
          builder_.Gen<x86_64::PmaxsdXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmaxudXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        builder_.Gen<x86_64::PsubdXRegXReg>(xmax.machine_reg(), xn.machine_reg());
        break;
    }
    if (is_accum) {
      // Accumulate the abs-diff (xmax) into Vd at the element width.
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PaddbXRegXReg>(xd.machine_reg(), xmax.machine_reg());
          break;
        case 0b01:
          builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xmax.machine_reg());
          break;
        default:  // 0b10
          builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xmax.machine_reg());
          break;
      }
      // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xd, args.q);
    } else {
      // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xmax, args.q);
    }
    return;
  }

  // SHADD/UHADD (halving add: (a+b)>>1), SRHADD/URHADD (rounding halving add:
  // (a+b+1)>>1), and SHSUB/UHSUB (halving sub: (a-b)>>1), signed or unsigned.
  // Mirrors lite_translator.h's kShadd/kUhadd/kSrhadd/kUrhadd/kShsub/kUhsub
  // blocks. ARM ARM C7.2 reserves the .2D (size=11) form -> bail.
  //   * Halfword/word ADD forms use the bitwise identities (proven from
  //     a+b = (a^b) + 2*(a&b) and a|b = (a^b) + (a&b)):
  //       (a+b)>>1   = (a&b) + ((a^b)>>1)   [SHADD/UHADD]
  //       (a+b+1)>>1 = (a|b) - ((a^b)>>1)   [SRHADD/URHADD]
  //     Arithmetic shift (PSRAW/PSRAD) for signed, logical (PSRLW/PSRLD) for
  //     unsigned. All SSE2, no widening.
  //   * Byte ADD forms and every SUB form widen each 64-bit half to the next
  //     lane width (PMOVSXBW/PMOVZXBW etc., SSE4.1), do PADDW/PSUBW, shift
  //     right by 1, and PACK back with signed/unsigned saturation. The +1 of
  //     the rounding-add byte form is materialized by PCMPEQW ones + PSUBW
  //     (PAVGB is not an allowlisted heavy LIR op). SUB unsigned masks the low
  //     sub-lane before PACK so a modular a<b result does not saturate up.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShadd ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUhadd ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrhadd ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUrhadd ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kShsub ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUhsub) {
    if (args.size == 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_signed =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShadd) ||
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrhadd) ||
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShsub);
    const bool is_round =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrhadd) ||
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kUrhadd);
    const bool is_sub =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShsub) ||
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kUhsub);
    // Byte-widened path needs SSE4.1 (PMOVSX/PMOVZX + PACKUSDW at size=01).
    const bool needs_widen = is_sub || (args.size == 0b00);
    if (needs_widen && !host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }

    // Halfword/word ADD forms: bitwise identity, no widening.
    if (!is_sub && args.size != 0b00) {
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      FpRegister xtmp = AllocTempSimdReg();  // (a&b) for floor, (a^b) is in xn
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
      if (is_round) {
        // (a|b) - ((a^b)>>1): xtmp = a|b, xn = (a^b)>>1, xtmp -= xn.
        builder_.Gen<x86_64::PorXRegXReg>(xtmp.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());
        if (args.size == 0b01) {
          if (is_signed) builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), int8_t{1});
          else builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PsubwXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
        } else {  // size == 0b10
          if (is_signed) builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{1});
          else builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PsubdXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
        }
      } else {
        // (a&b) + ((a^b)>>1): xtmp = a&b, xn = (a^b)>>1, xtmp += xn.
        builder_.Gen<x86_64::PandXRegXReg>(xtmp.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());
        if (args.size == 0b01) {
          if (is_signed) builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), int8_t{1});
          else builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PaddwXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
        } else {  // size == 0b10
          if (is_signed) builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{1});
          else builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PadddXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
        }
      }
      SetVRegFull(args.rd, xtmp, args.q);
      return;
    }

    // Byte ADD forms and all SUB forms: widen each 64-bit half, op, shift, pack.
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    FpRegister xn_hi = AllocTempSimdReg();
    FpRegister xm_hi = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    builder_.Gen<x86_64::MovdqaXRegXReg>(xn_hi.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(xm_hi.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PsrldqXRegImm>(xn_hi.machine_reg(), int8_t{8});
    builder_.Gen<x86_64::PsrldqXRegImm>(xm_hi.machine_reg(), int8_t{8});

    if (args.size == 0b00) {
      // 8->16 widen.
      if (is_signed) {
        builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
        builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
      } else {
        builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
        builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
      }
      if (is_sub) {
        builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PsubwXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
      } else {
        builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PaddwXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
        if (is_round) {
          // +1 per word: xm/xm_hi are dead; clobber to all-ones and PSUBW.
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PsubwXRegXReg>(xn_hi.machine_reg(), xm.machine_reg());
        }
      }
      if (is_signed) {
        builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), int8_t{1});
        builder_.Gen<x86_64::PsrawXRegImm>(xn_hi.machine_reg(), int8_t{1});
        builder_.Gen<x86_64::PacksswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
      } else {
        builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{1});
        builder_.Gen<x86_64::PsrlwXRegImm>(xn_hi.machine_reg(), int8_t{1});
        if (is_sub) {
          // Mask low byte per word so PACKUSWB does not saturate the modular
          // a<b result (in [0x7F80, 0x7FFF]) up to 0xFF.
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(xn_hi.machine_reg(), xm.machine_reg());
        }
        builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
      }
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // SUB size=01 (halfword): 16->32 widen, PSUBD, shift, PACKSSDW/PACKUSDW.
    if (args.size == 0b01) {
      if (is_signed) {
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
        builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PsubdXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{1});
        builder_.Gen<x86_64::PsradXRegImm>(xn_hi.machine_reg(), int8_t{1});
        builder_.Gen<x86_64::PackssdwXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
      } else {
        builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
        builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
        builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PsubdXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
        builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{1});
        builder_.Gen<x86_64::PsrldXRegImm>(xn_hi.machine_reg(), int8_t{1});
        // Mask low 16 bits per dword before PACKUSDW.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{16});
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn_hi.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
      }
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // SUB size=10 (word): 32->64 widen, PSUBQ, PSRLQ 1, gather low dwords via
    // PSHUFD 0x88 + PUNPCKLQDQ. PSRLQ is logical, but the bit-63 difference vs
    // an arithmetic shift is discarded by the low-dword gather, so PMOVSXDQ vs
    // PMOVZXDQ alone distinguishes signed/unsigned.
    if (is_signed) {
      builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
      builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
    } else {
      builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
      builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
    }
    builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PsubqXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
    builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), int8_t{1});
    builder_.Gen<x86_64::PsrlqXRegImm>(xn_hi.machine_reg(), int8_t{1});
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                            static_cast<int8_t>(0x88));
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xn_hi.machine_reg(), xn_hi.machine_reg(),
                                            static_cast<int8_t>(0x88));
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // ADDP (pairwise add) vector — mirror of lite_translator.h's kAddp. Vd =
  // pair(Vn) || pair(Vm) where pair(X)[i] = X[2i] + X[2i+1]. 8H/4S map to
  // PHADDW/PHADDD directly (SSSE3 — ARM's concat-then-pair layout); 4H/2S use
  // the same op then PSHUFD 0x08 to gather the two low pair-lanes into the low
  // 64 bits; byte lanes emulate via PSRLW-8 + PADDB (even bytes hold the pair
  // sums) then truncate each halfword's low byte and PACKUSWB; .2D (size=11
  // Q=1) has no PHADDQ, so PSHUFD 0xEE + PADDQ + PUNPCKLQDQ. .1D (size=11 Q=0)
  // is ARM-reserved. SetVRegFull's Q=0 merge zeroes Vd[127:64].
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kAddp) {
    // Halfword/word forms need PHADDW/PHADDD (SSSE3); byte and .2D are SSE2.
    if ((args.size == 0b01 || args.size == 0b10) && !host_platform::kHasSSSE3) {
      UndefinedReturningVoid();
      return;
    }
    if (args.size == 0b11 && !args.q) {  // .1D reserved
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    switch (args.size) {
      case 0b00: {
        FpRegister tmp_n = AllocTempSimdReg();
        FpRegister tmp_m = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(tmp_n.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(tmp_m.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PsrlwXRegImm>(tmp_n.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrlwXRegImm>(tmp_m.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PaddbXRegXReg>(xn.machine_reg(), tmp_n.machine_reg());
        builder_.Gen<x86_64::PaddbXRegXReg>(xm.machine_reg(), tmp_m.machine_reg());
        builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsllwXRegImm>(xm.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});
        if (args.q) {
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PackuswbXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PunpckldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
      }
      case 0b01:
        builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        if (!args.q) {
          builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                  int8_t{0x08});
        }
        break;
      case 0b10:
        builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xm.machine_reg());
        if (!args.q) {
          builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                  int8_t{0x08});
        }
        break;
      case 0b11: {  // .2D (Q=1); .1D already bailed above.
        FpRegister tmp = AllocTempSimdReg();
        builder_.Gen<x86_64::PshufdXRegXRegImm>(tmp.machine_reg(), xn.machine_reg(),
                                                static_cast<int8_t>(0xEE));
        builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), tmp.machine_reg());
        builder_.Gen<x86_64::PshufdXRegXRegImm>(tmp.machine_reg(), xm.machine_reg(),
                                                static_cast<int8_t>(0xEE));
        builder_.Gen<x86_64::PaddqXRegXReg>(xm.machine_reg(), tmp.machine_reg());
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      }
    }
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // SMAXP/SMINP/UMAXP/UMINP (pairwise signed/unsigned max/min) — mirror of
  // lite_translator.h. Vd = pair(Vn) || pair(Vm), pair(X)[i] = op(X[2i],
  // X[2i+1]). No horizontal-pairwise min/max exists in SSE; gather even/odd
  // lanes (PSHUFB for byte/halfword, PSHUFD 0x88/0xDD for dword), lane-wise
  // PMAX/PMIN, then concatenate the Vn and Vm partials. size=11 (.2D) needs
  // 64-bit packed min/max (AVX-512) — bail to lite/interpreter.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kSminp ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmaxp ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUminp) {
    if (args.size == 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_max =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
         args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmaxp);
    const bool is_signed =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
         args.opcode == Decoder::AdvSimdThreeSameOpcode::kSminp);
    // Feature gate mirrors lite: PMAXSB/PMINSB/PMAXSD/PMINSD/PMAXUW/PMINUW/
    // PMAXUD/PMINUD -> SSE4.1; PMAXSW/PMINSW/PMAXUB/PMINUB -> SSE2. Byte and
    // halfword even/odd gathers need PSHUFB (SSSE3).
    const bool needs_sse4_1 =
        (is_signed && args.size == 0b00) ||
        (is_signed && args.size == 0b10) ||
        (!is_signed && args.size == 0b01) ||
        (!is_signed && args.size == 0b10);
    if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }
    const bool needs_ssse3 = (args.size == 0b00 || args.size == 0b01);
    if (needs_ssse3 && !host_platform::kHasSSSE3) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    auto pmax_pmin = [&](FpRegister dst, FpRegister src) {
      switch (args.size) {
        case 0b00:
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxsbXRegXReg>(dst.machine_reg(), src.machine_reg());
            else builder_.Gen<x86_64::PminsbXRegXReg>(dst.machine_reg(), src.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxubXRegXReg>(dst.machine_reg(), src.machine_reg());
            else builder_.Gen<x86_64::PminubXRegXReg>(dst.machine_reg(), src.machine_reg());
          }
          break;
        case 0b01:
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxswXRegXReg>(dst.machine_reg(), src.machine_reg());
            else builder_.Gen<x86_64::PminswXRegXReg>(dst.machine_reg(), src.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxuwXRegXReg>(dst.machine_reg(), src.machine_reg());
            else builder_.Gen<x86_64::PminuwXRegXReg>(dst.machine_reg(), src.machine_reg());
          }
          break;
        case 0b10:
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxsdXRegXReg>(dst.machine_reg(), src.machine_reg());
            else builder_.Gen<x86_64::PminsdXRegXReg>(dst.machine_reg(), src.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxudXRegXReg>(dst.machine_reg(), src.machine_reg());
            else builder_.Gen<x86_64::PminudXRegXReg>(dst.machine_reg(), src.machine_reg());
          }
          break;
      }
    };
    if (args.size == 0b00 || args.size == 0b01) {
      // Byte/halfword: build PSHUFB even/odd gather masks from immediates
      // (upper qword 0x80 => PSHUFB writes zero there), gather each operand's
      // even and odd lanes into the low 8 bytes, PMAX/PMIN them to form the
      // 8-byte partial pair(Vn) / pair(Vm), then concatenate.
      FpRegister even_mask = AllocTempSimdReg();
      FpRegister odd_mask = AllocTempSimdReg();
      FpRegister evens_n = AllocTempSimdReg();
      FpRegister odds_n = AllocTempSimdReg();
      FpRegister evens_m = AllocTempSimdReg();
      FpRegister odds_m = AllocTempSimdReg();
      int64_t even_lo, even_hi, odd_lo, odd_hi;
      if (args.size == 0b00) {
        even_lo = static_cast<int64_t>(0x0E0C0A0806040200LL);
        even_hi = static_cast<int64_t>(0x8080808080808080ULL);
        odd_lo = static_cast<int64_t>(0x0F0D0B0907050301LL);
        odd_hi = static_cast<int64_t>(0x8080808080808080ULL);
      } else {
        even_lo = static_cast<int64_t>(0x0D0C090805040100LL);
        even_hi = static_cast<int64_t>(0x8080808080808080ULL);
        odd_lo = static_cast<int64_t>(0x0F0E0B0A07060302LL);
        odd_hi = static_cast<int64_t>(0x8080808080808080ULL);
      }
      Register elo = std::get<0>(Gen<x86_64::MovqRegImm>(even_lo));
      builder_.Gen<x86_64::MovqXRegReg>(even_mask.machine_reg(), elo);
      Register ehi = std::get<0>(Gen<x86_64::MovqRegImm>(even_hi));
      builder_.Gen<x86_64::PinsrqXRegRegImm>(even_mask.machine_reg(), ehi, int8_t{1});
      Register olo = std::get<0>(Gen<x86_64::MovqRegImm>(odd_lo));
      builder_.Gen<x86_64::MovqXRegReg>(odd_mask.machine_reg(), olo);
      Register ohi = std::get<0>(Gen<x86_64::MovqRegImm>(odd_hi));
      builder_.Gen<x86_64::PinsrqXRegRegImm>(odd_mask.machine_reg(), ohi, int8_t{1});

      builder_.Gen<x86_64::MovdqaXRegXReg>(evens_n.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PshufbXRegXReg>(evens_n.machine_reg(), even_mask.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(odds_n.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PshufbXRegXReg>(odds_n.machine_reg(), odd_mask.machine_reg());
      pmax_pmin(evens_n, odds_n);  // low 8 bytes hold pair(Vn)

      builder_.Gen<x86_64::MovdqaXRegXReg>(evens_m.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PshufbXRegXReg>(evens_m.machine_reg(), even_mask.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(odds_m.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PshufbXRegXReg>(odds_m.machine_reg(), odd_mask.machine_reg());
      pmax_pmin(evens_m, odds_m);  // low 8 bytes hold pair(Vm)

      if (args.q) {
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(evens_n.machine_reg(), evens_m.machine_reg());
      } else {
        builder_.Gen<x86_64::PunpckldqXRegXReg>(evens_n.machine_reg(), evens_m.machine_reg());
      }
      SetVRegFull(args.rd, evens_n, args.q);
      return;
    }
    // size == 0b10 (.4S / .2S): dword pairwise via PSHUFD even/odd lift.
    // 0x88 => {d0,d2,d0,d2}; 0xDD => {d1,d3,d1,d3}; PMAX/PMIN yields pair(X)
    // replicated in both halves. Q=1 concatenates the low qwords; Q=0 packs
    // dwords 0 and 2 into positions 0,1 with a final PSHUFD 0x08.
    FpRegister evens_n = AllocTempSimdReg();
    FpRegister odds_n = AllocTempSimdReg();
    FpRegister evens_m = AllocTempSimdReg();
    FpRegister odds_m = AllocTempSimdReg();
    builder_.Gen<x86_64::PshufdXRegXRegImm>(evens_n.machine_reg(), xn.machine_reg(),
                                            static_cast<int8_t>(0x88));
    builder_.Gen<x86_64::PshufdXRegXRegImm>(odds_n.machine_reg(), xn.machine_reg(),
                                            static_cast<int8_t>(0xDD));
    pmax_pmin(evens_n, odds_n);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(evens_m.machine_reg(), xm.machine_reg(),
                                            static_cast<int8_t>(0x88));
    builder_.Gen<x86_64::PshufdXRegXRegImm>(odds_m.machine_reg(), xm.machine_reg(),
                                            static_cast<int8_t>(0xDD));
    pmax_pmin(evens_m, odds_m);
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(evens_n.machine_reg(), evens_m.machine_reg());
    if (!args.q) {
      builder_.Gen<x86_64::PshufdXRegXRegImm>(evens_n.machine_reg(), evens_n.machine_reg(),
                                              int8_t{0x08});
    }
    SetVRegFull(args.rd, evens_n, args.q);
    return;
  }

  // SMAX/SMIN/UMAX/UMIN (plain, non-pairwise lane-wise signed/unsigned
  // min/max) — mirror of lite_translator.h. x86 has direct lane-width-matched
  // PMAXS/PMINS/PMAXU/PMINU for 8/16/32-bit widths; the .2D (64-bit) lane has
  // no SSE-era op (PMAXSQ/PMINSQ/PMAXUQ/PMINUQ are AVX-512F-VL only) — bail.
  // Per-size SSE feature gate (Intel SDM Vol 2):
  //   PMAXSB/PMINSB/PMAXSD/PMINSD/PMAXUW/PMINUW/PMAXUD/PMINUD -> SSE4.1;
  //   PMAXSW/PMINSW/PMAXUB/PMINUB                              -> SSE2.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax ||
      args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmin) {
    if (args.size == 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_max =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
         args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax);
    const bool is_signed =
        (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
         args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin);
    const bool needs_sse4_1 =
        (is_signed && args.size == 0b00) ||
        (is_signed && args.size == 0b10) ||
        (!is_signed && args.size == 0b01) ||
        (!is_signed && args.size == 0b10);
    if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    switch (args.size) {
      case 0b00:  // .16B / .8B
        if (is_signed) {
          if (is_max) builder_.Gen<x86_64::PmaxsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          if (is_max) builder_.Gen<x86_64::PmaxubXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminubXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
      case 0b01:  // .8H / .4H
        if (is_signed) {
          if (is_max) builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          if (is_max) builder_.Gen<x86_64::PmaxuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
      case 0b10:  // .4S / .2S
        if (is_signed) {
          if (is_max) builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          if (is_max) builder_.Gen<x86_64::PmaxudXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
    }
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // BIC / ORN (vector bitwise AND-NOT / OR-NOT) — mirror of lite_translator.h.
  // Element-size-independent bit ops that read only Vn/Vm, but need the PANDN /
  // ones-materialize sequences rather than a single in-place op, so they live
  // here rather than the shared vn-only switch below.
  //   BIC  Vd = Vn AND NOT Vm.  x86 PANDN(dst, src) = ~dst & src, so
  //        PANDN(xm, xn) lands ~Vm & Vn in xm.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kBic) {
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    builder_.Gen<x86_64::PandnXRegXReg>(xm.machine_reg(), xn.machine_reg());
    SetVRegFull(args.rd, xm, args.q);
    return;
  }
  //   ORN  Vd = Vn OR NOT Vm.  Materialize all-ones via self-PCMPEQD on a
  //        pre-zeroed vreg (AllocZeroedSimdReg establishes a def before the
  //        self-compare), XOR into xm to get ~Vm, then OR with xn.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kOrn) {
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    FpRegister ones = AllocZeroedSimdReg();  // def before the self-compare
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(ones.machine_reg(), ones.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), ones.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xm.machine_reg());
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // CMTST (test bits) — mirror of lite_translator.h. Vd = (Vn & Vm) != 0 ?
  // all-ones : 0 per lane. PAND, compare the AND result against zero (yielding
  // all-ones where the lane IS zero), then invert so the non-zero lanes are
  // set. Byte/halfword/word use PCMPEQ{B,W,D} (SSE2); the .2D (size=11) form
  // uses PCMPEQQ (SSE4.1) — bail cleanly if absent, matching lite.
  if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmtst) {
    if (args.size == 0b11 && !host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());  // Vn & Vm
    FpRegister z = AllocZeroedSimdReg();  // zero, proper def for the self-compare below
    switch (args.size) {
      case 0b00: builder_.Gen<x86_64::PcmpeqbXRegXReg>(xn.machine_reg(), z.machine_reg()); break;
      case 0b01: builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn.machine_reg(), z.machine_reg()); break;
      case 0b10: builder_.Gen<x86_64::PcmpeqdXRegXReg>(xn.machine_reg(), z.machine_reg()); break;
      default:   builder_.Gen<x86_64::PcmpeqqXRegXReg>(xn.machine_reg(), z.machine_reg()); break;
    }
    // z is now dead as the zero comparand; clobber it to all-ones (it already
    // has a def, so the self-PCMPEQD is legal) and XOR to invert the mask.
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(z.machine_reg(), z.machine_reg());  // z = -1
    builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), z.machine_reg());
    SetVRegFull(args.rd, xn, args.q);
    return;
  }

  // Validate the (opcode, size) pair up front and emit nothing on bail. After
  // this switch every reachable case has a single allowlisted packed op.
  switch (args.opcode) {
    case Decoder::AdvSimdThreeSameOpcode::kAdd:
    case Decoder::AdvSimdThreeSameOpcode::kSub:
      // ADD -> Padd{b,w,d,q}; SUB -> Psub{b,w,d,q}. All four element sizes
      // (8/16/32/64) have a direct SSE2 packed op, so every size is handled
      // (mirrors lite_translator.h). The reserved .1D shape (size=11, Q=0) is
      // UNALLOCATED; like lite we don't special-case it — SetVRegFull's Q=0
      // merge just zero-extends the low 64 bits.
      break;
    case Decoder::AdvSimdThreeSameOpcode::kMul:
      if (args.size != 0b01 && args.size != 0b10) {
        UndefinedReturningVoid();
        return;
      }
      break;
    case Decoder::AdvSimdThreeSameOpcode::kAnd:
    case Decoder::AdvSimdThreeSameOpcode::kOrr:
    case Decoder::AdvSimdThreeSameOpcode::kEor:
      // Bitwise: element size is irrelevant; all forms are handled.
      break;
    case Decoder::AdvSimdThreeSameOpcode::kCmeq:
    case Decoder::AdvSimdThreeSameOpcode::kCmgt:
      // CMEQ -> PCMPEQ{B,W,D,Q}; CMGT (signed) -> PCMPGT{B,W,D,Q}. The 64-bit
      // (2D) form needs PCMPEQQ (SSE4.1) / PCMPGTQ (SSE4.2) -- gate on the
      // host feature; the scalar D compare block below already relies on both.
      if (args.size == 0b11 &&
          !((args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmeq)
                ? host_platform::kHasSSE4_1
                : host_platform::kHasSSE4_2)) {
        UndefinedReturningVoid();
        return;
      }
      break;
    default:
      UndefinedReturningVoid();
      return;
  }

  FpRegister vn = AllocTempSimdReg();
  FpRegister vm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(vn.machine_reg(), vn_off);
  builder_.GenGetSimd<16>(vm.machine_reg(), vm_off);

  // Run the packed op in place on vn (vn := vn OP vm).
  switch (args.opcode) {
    case Decoder::AdvSimdThreeSameOpcode::kAdd:
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PaddbXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else if (args.size == 0b01) {
        builder_.Gen<x86_64::PaddwXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else if (args.size == 0b10) {
        builder_.Gen<x86_64::PadddXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else {
        builder_.Gen<x86_64::PaddqXRegXReg>(vn.machine_reg(), vm.machine_reg());
      }
      break;
    case Decoder::AdvSimdThreeSameOpcode::kSub:
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PsubbXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else if (args.size == 0b01) {
        builder_.Gen<x86_64::PsubwXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else if (args.size == 0b10) {
        builder_.Gen<x86_64::PsubdXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else {
        builder_.Gen<x86_64::PsubqXRegXReg>(vn.machine_reg(), vm.machine_reg());
      }
      break;
    case Decoder::AdvSimdThreeSameOpcode::kMul:
      if (args.size == 0b01) {
        builder_.Gen<x86_64::PmullwXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmulldXRegXReg>(vn.machine_reg(), vm.machine_reg());
      }
      break;
    case Decoder::AdvSimdThreeSameOpcode::kAnd:
      builder_.Gen<x86_64::PandXRegXReg>(vn.machine_reg(), vm.machine_reg());
      break;
    case Decoder::AdvSimdThreeSameOpcode::kOrr:
      builder_.Gen<x86_64::PorXRegXReg>(vn.machine_reg(), vm.machine_reg());
      break;
    case Decoder::AdvSimdThreeSameOpcode::kEor:
      builder_.Gen<x86_64::PxorXRegXReg>(vn.machine_reg(), vm.machine_reg());
      break;
    case Decoder::AdvSimdThreeSameOpcode::kCmeq:
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PcmpeqbXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else if (args.size == 0b01) {
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else if (args.size == 0b10) {
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else {
        builder_.Gen<x86_64::PcmpeqqXRegXReg>(vn.machine_reg(), vm.machine_reg());
      }
      break;
    case Decoder::AdvSimdThreeSameOpcode::kCmgt:
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PcmpgtbXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else if (args.size == 0b01) {
        builder_.Gen<x86_64::PcmpgtwXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else if (args.size == 0b10) {
        builder_.Gen<x86_64::PcmpgtdXRegXReg>(vn.machine_reg(), vm.machine_reg());
      } else {
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(vn.machine_reg(), vm.machine_reg());
      }
      break;
    default:
      // Unreachable: the validation switch above already bailed.
      UndefinedReturningVoid();
      return;
  }

  SetVRegFull(args.rd, vn, args.q);
}

// Heavy-tier mirror of the register-domain-pure subset of
// lite_translator.h::AdvSimdThreeDiff:
//   {S,U}MULL{,2}, {S,U}MLAL{,2}, {S,U}MLSL{,2}   (widening multiply-accumulate)
//   {S,U}ADDL{,2}, {S,U}SUBL{,2}                  (widening add/sub, both narrow)
//   {S,U}ADDW{,2}, {S,U}SUBW{,2}                  (widening add/sub, Vn wide)
// at input sizes 8/16/32 (size 00/01/10) and both Q halves. The widening
// recipe matches lite exactly: widen the narrow sources (PMOVSX/PMOVZX per
// sign), then for the multiply subset lane-wise multiply
// (PMULLW / PMULLD / PMULDQ|PMULUDQ) — MLAL/MLSL load Vd and PADD/PSUB the
// product — and for the add/sub subset PADD/PSUB the widened operands
// directly at the wide lane width. The result always fills 128 bits
// (8H/4S/2D), so SetVRegFull q=true regardless of Q. Q=1 ("2") forms take
// the upper 64 of the narrow sources — bring bytes 8..15 down with PSRLDQ
// before widening (the W-forms' Vn is already a 128-bit wide vector and is
// loaded full, never shifted).
//
// Also mirrors the narrowing-high subset:
//   ADDHN/SUBHN/RADDHN/RSUBHN — add/sub the two wide-lane sources, then take
//   the HIGH half of each lane as the narrow result (rounding variants add a
//   half-ulp bias first). Committed via SetVRegNarrow (Q=0 zero-extend / Q2
//   merge into Vd.high).
//
// Also mirrors the saturating-doubling widening subset:
//   SQDMULL/SQDMLAL/SQDMLSL — signed saturating doubling multiply long, with
//   the accumulate/subtract forms saturating again on the add. size 01
//   (.4H->.4S, manual 32-bit saturation) and size 10 (.2S->.2D, manual 64-bit
//   saturation); size 00/11 are decoder-rejected. The remaining ThreeDiff
//   opcodes (ABDL/ABAL, PMULL) still bail to the lite tier — emit NOTHING
//   before a bail.
void HeavyOptimizerFrontend::AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::AdvSimdThreeDiffOpcode;

  // PMULL/PMULL2 — polynomial multiply long over GF(2). Line-by-line mirror
  // of lite_translator.h::AdvSimdThreeDiff's kPmull block.
  //   size=11 (PMULL64): a single PCLMULQDQ (x86 CLMUL is an exact GF(2)
  //     carry-less multiply, semantically identical to ARM PMULL64). Q=0
  //     multiplies the low qwords (imm 0x00 = Pclmullqlqdq); Q=1/PMULL2
  //     multiplies the high qwords (imm 0x11 = Pclmulhqhqdq).
  //   size=00 (PMULL .8H, poly8 widening): 8 independent 8-bit polynomial
  //     products widened to 16-bit. Widen the low 8 bytes (Q=1/PMULL2 takes
  //     bytes 8..15) to 16-bit lanes (PMOVZXBW), then run the per-bit carry-
  //     less shift-and-XOR loop in 16-bit lanes. Each lane holds a value <256
  //     and the degree-14 product fits in 16 bits, so the PSLLW shifts never
  //     spill across lanes (no per-lane masking needed). The bit-i selector is
  //     PSLLW(b, 15-i) then PSRAW 15, broadcasting bit i to the whole lane.
  //   size in {01,10} is reserved for PMULL and bails.
  // The "long" result always fills 128 bits, so SetVRegFull q=true regardless
  // of the Q ("2") variant.
  if (args.opcode == Op::kPmull) {
    const int32_t vn_o =
        GetVRegOffset(args.rn);
    const int32_t vm_o =
        GetVRegOffset(args.rm);
    if (args.size == 0b11) {
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_o);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_o);
      const int8_t imm = args.q ? int8_t{0x11} : int8_t{0x00};
      builder_.Gen<x86_64::PclmulqdqXRegXRegImm>(
          xn.machine_reg(), xm.machine_reg(), imm);
      SetVRegFull(args.rd, xn, /*q=*/true);
      return;
    }
    if (args.size == 0b00) {  // PMULL/PMULL2 .8H (poly8 widening)
      const int32_t extra = args.q ? 8 : 0;
      FpRegister za = AllocTempSimdReg();
      FpRegister zb = AllocTempSimdReg();
      FpRegister acc = AllocZeroedSimdReg();  // PXOR acc,acc
      FpRegister shifted = AllocTempSimdReg();
      FpRegister sel = AllocTempSimdReg();
      // MOVSD zero-extends bits [127:64]; PMOVZXBW then widens 8 bytes -> 8
      // 16-bit lanes (Q=1 loads bytes 8..15 via the +8 offset).
      builder_.GenGetSimd<8>(za.machine_reg(), vn_o + extra);
      builder_.GenGetSimd<8>(zb.machine_reg(), vm_o + extra);
      builder_.Gen<x86_64::PmovzxbwXRegXReg>(za.machine_reg(), za.machine_reg());
      builder_.Gen<x86_64::PmovzxbwXRegXReg>(zb.machine_reg(), zb.machine_reg());
      for (int i = 0; i < 8; ++i) {
        builder_.Gen<x86_64::MovdqaXRegXReg>(shifted.machine_reg(), za.machine_reg());
        if (i != 0) {
          builder_.Gen<x86_64::PsllwXRegImm>(shifted.machine_reg(),
                                             static_cast<int8_t>(i));
        }
        // sel = 0xFFFF per lane where bit i of zb is set.
        builder_.Gen<x86_64::MovdqaXRegXReg>(sel.machine_reg(), zb.machine_reg());
        if (15 - i != 0) {
          builder_.Gen<x86_64::PsllwXRegImm>(sel.machine_reg(),
                                             static_cast<int8_t>(15 - i));
        }
        builder_.Gen<x86_64::PsrawXRegImm>(sel.machine_reg(), int8_t{15});
        builder_.Gen<x86_64::PandXRegXReg>(shifted.machine_reg(), sel.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(acc.machine_reg(), shifted.machine_reg());
      }
      SetVRegFull(args.rd, acc, /*q=*/true);
      return;
    }
    UndefinedReturningVoid();  // size in {01,10} reserved
    return;
  }

  // Widening add/sub subset: {S,U}ADDL/{S,U}SUBL (both sources narrow) and
  // {S,U}ADDW/{S,U}SUBW (Vn already a wide-lane 128-bit vector). Handle first;
  // fall through to the multiply-accumulate subset otherwise.
  const bool is_addl = (args.opcode == Op::kSaddl || args.opcode == Op::kUaddl);
  const bool is_subl = (args.opcode == Op::kSsubl || args.opcode == Op::kUsubl);
  const bool is_addw = (args.opcode == Op::kSaddw || args.opcode == Op::kUaddw);
  const bool is_subw = (args.opcode == Op::kSsubw || args.opcode == Op::kUsubw);
  if (is_addl || is_subl || is_addw || is_subw) {
    if (args.size > 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool addsub_signed = (args.opcode == Op::kSaddl ||
                                args.opcode == Op::kSsubl ||
                                args.opcode == Op::kSaddw ||
                                args.opcode == Op::kSsubw);
    const bool is_add = (is_addl || is_addw);
    const bool n_is_wide = (is_addw || is_subw);
    const int32_t vn_off_as =
        GetVRegOffset(args.rn);
    const int32_t vm_off_as =
        GetVRegOffset(args.rm);

    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off_as);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off_as);
    // Q=1 selects the upper 64 of the *narrow* sources; the W-forms' Vn is
    // already wide, so only shift it for the L-forms.
    if (args.q) {
      if (!n_is_wide) {
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
      }
      builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
    }
    // Widen Vn (skip for W-forms — Vn is already a wide-lane vector).
    if (!n_is_wide) {
      switch (args.size) {
        case 0b00:
          if (addsub_signed) builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          else builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          break;
        case 0b01:
          if (addsub_signed) builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          else builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          break;
        case 0b10:
          if (addsub_signed) builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          else builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          break;
      }
    }
    // Widen Vm (always narrow).
    switch (args.size) {
      case 0b00:
        if (addsub_signed) builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
        else builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
        break;
      case 0b01:
        if (addsub_signed) builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        else builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        break;
      case 0b10:
        if (addsub_signed) builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        else builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        break;
    }
    // xn = Vn +/- Vm at the wide lane width.
    switch (args.size) {
      case 0b00:
        if (is_add) builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        else builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      case 0b01:
        if (is_add) builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xm.machine_reg());
        else builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      case 0b10:
        if (is_add) builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        else builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
    }
    SetVRegFull(args.rd, xn, /*q=*/true);
    return;
  }

  // SABDL/UABDL (absolute-difference-long) and SABAL/UABAL (abs-diff-long
  // accumulate). Widen both narrow sources (Q selects the low/high 64 of
  // Vn/Vm), then abs(a - b) at the widened lane width:
  //   size=00/01: max(a,b) - min(a,b) via PMAXS*/PMINS* (signed) or
  //               PMAXU*/PMINU* (unsigned), then PSUB.
  //   size=10   : no 64-bit SSE lane-wise max/min, so diff = a - b (PSUBQ
  //               after the sign/zero-widening), then a Pcmpgtq-against-zero
  //               signed-abs: mask = (0 > diff); abs = (diff ^ mask) - mask.
  // ABAL then accumulates the abs-diff into Vd. The result always fills 128
  // bits (8H/4S/2D). Mirrors lite_translator.h::AdvSimdThreeDiff's
  // SABDL/UABDL/SABAL/UABAL block size-by-size.
  if (args.opcode == Op::kSabdl || args.opcode == Op::kUabdl ||
      args.opcode == Op::kSabal || args.opcode == Op::kUabal) {
    if (args.size > 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_signed =
        (args.opcode == Op::kSabdl || args.opcode == Op::kSabal);
    const bool is_abal =
        (args.opcode == Op::kSabal || args.opcode == Op::kUabal);
    const int32_t vn_off_ab =
        GetVRegOffset(args.rn);
    const int32_t vm_off_ab =
        GetVRegOffset(args.rm);
    const int32_t vd_off_ab =
        GetVRegOffset(args.rd);

    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off_ab);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off_ab);
    // Q=1 ("2" form) selects the upper 64 of the narrow sources.
    if (args.q) {
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
      builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
    }
    // Widen both narrow sources to the wide lane width.
    switch (args.size) {
      case 0b00:
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
        }
        break;
      case 0b01:
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        }
        break;
      case 0b10:
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        }
        break;
    }

    if (args.size == 0b10) {
      // 32->64: diff = a - b at 64-bit lane width, then signed-abs via
      // Pcmpgtq against zero. Correct for both signs because the widening
      // already injected the correct sign/zero extension.
      builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      FpRegister mask = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpgtqXRegXReg>(mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), mask.machine_reg());
      builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), mask.machine_reg());
      if (is_abal) {
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off_ab);
        builder_.Gen<x86_64::PaddqXRegXReg>(xd.machine_reg(), xn.machine_reg());
        SetVRegFull(args.rd, xd, /*q=*/true);
      } else {
        SetVRegFull(args.rd, xn, /*q=*/true);
      }
      return;
    }

    // 8->16 / 16->32: abs diff = max(a,b) - min(a,b). Copy Vn into xmax
    // (max clobbers its dst), leaving xn to hold the min.
    FpRegister xmax = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(xmax.machine_reg(), xn.machine_reg());
    if (args.size == 0b00) {
      if (is_signed) {
        builder_.Gen<x86_64::PmaxswXRegXReg>(xmax.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmaxuwXRegXReg>(xmax.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      builder_.Gen<x86_64::PsubwXRegXReg>(xmax.machine_reg(), xn.machine_reg());
    } else {  // size == 0b01
      if (is_signed) {
        builder_.Gen<x86_64::PmaxsdXRegXReg>(xmax.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmaxudXRegXReg>(xmax.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      builder_.Gen<x86_64::PsubdXRegXReg>(xmax.machine_reg(), xn.machine_reg());
    }
    if (is_abal) {
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off_ab);
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xmax.machine_reg());
      } else {
        builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xmax.machine_reg());
      }
      SetVRegFull(args.rd, xd, /*q=*/true);
    } else {
      SetVRegFull(args.rd, xmax, /*q=*/true);
    }
    return;
  }

  // ADDHN/SUBHN/RADDHN/RSUBHN — add/sub the two wide-lane sources, then take
  // the HIGH half of each lane as the narrow result. size 00/01/10 selects
  // source lane 16/32/64 -> narrow dst 8/16/32. Rounding variants add a
  // half-ulp bias (1 << (esize-1) of the SOURCE lane, i.e. bit just below the
  // >>esize truncation point) before the shift. size=00: PADDW/PSUBW, >>8,
  // PACKUSWB gathers the 8 high-bytes to the low 64. size=01: PADDD/PSUBD,
  // >>16, PACKUSDW. size=10: PADDQ/PSUBQ, >>32, PSHUFD 0b00001000 gathers
  // dwords {0,2} to the low 64 (no 64->32 pack). The narrow result is
  // committed via SetVRegNarrow (Q=0 zero-extend / Q2 merge into Vd.high).
  // Mirrors lite_translator.h::AdvSimdThreeDiff's ADDHN/SUBHN block.
  if (args.opcode == Op::kAddhn || args.opcode == Op::kSubhn ||
      args.opcode == Op::kRaddhn || args.opcode == Op::kRsubhn) {
    if (args.size > 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_sub = (args.opcode == Op::kSubhn || args.opcode == Op::kRsubhn);
    const bool is_round = (args.opcode == Op::kRaddhn || args.opcode == Op::kRsubhn);
    const int32_t vn_o =
        GetVRegOffset(args.rn);
    const int32_t vm_o =
        GetVRegOffset(args.rm);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_o);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_o);

    // Materialize a broadcast constant (`pattern` in both qwords) for the
    // rounding bias.
    auto broadcast = [&](uint64_t pattern) -> FpRegister {
      FpRegister x = AllocTempSimdReg();
      Register gr =
          std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pattern)));
      builder_.Gen<x86_64::MovqXRegReg>(x.machine_reg(), gr);
      builder_.Gen<x86_64::PinsrqXRegRegImm>(x.machine_reg(), gr, int8_t{1});
      return x;
    };

    if (args.size == 0b10) {
      if (is_sub) {
        builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      if (is_round) {
        FpRegister xr = broadcast(uint64_t{0x0000000080000000ULL});
        builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xr.machine_reg());
      }
      builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), int8_t{32});
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                              int8_t{0b00001000});
      SetVRegNarrow(args.rd, xn, args.q);
      return;
    }
    if (args.size == 0b00) {
      if (is_sub) {
        builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      if (is_round) {
        FpRegister xr = broadcast(uint64_t{0x0080008000800080ULL});
        builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xr.machine_reg());
      }
      builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{8});
      FpRegister xz = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
    } else {  // size == 0b01
      if (is_sub) {
        builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      if (is_round) {
        FpRegister xr = broadcast(uint64_t{0x0000800000008000ULL});
        builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xr.machine_reg());
      }
      builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{16});
      FpRegister xz = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
    }
    SetVRegNarrow(args.rd, xn, args.q);
    return;
  }

  // SQDMULL/SQDMLAL/SQDMLSL — signed saturating doubling multiply long (+
  // saturating accumulate/subtract). Line-by-line mirror of
  // lite_translator.h::AdvSimdThreeDiff's SQDMULL block. Only size=01
  // (.4H->.4S) and size=10 (.2S->.2D) are defined (size 00/11 are
  // decoder-rejected). The exact lane products are doubled with manual
  // saturation of the single INT_MIN^2 overflow lane (product == 2^30 / 2^62
  // -> SMAX), then the accumulate forms apply a second signed-saturating
  // add/sub via the (a^P)&(a^res) sign-bit overflow-detection idiom. The
  // "long" result always fills 128 bits, so SetVRegFull q=true regardless of
  // the Q ("2") bit, which only selects the upper half of the narrow sources.
  if (args.opcode == Op::kSqdmull || args.opcode == Op::kSqdmlal ||
      args.opcode == Op::kSqdmlsl) {
    if (args.size != 0b01 && args.size != 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_acc = (args.opcode != Op::kSqdmull);
    const bool is_sub = (args.opcode == Op::kSqdmlsl);
    const int32_t vn_o =
        GetVRegOffset(args.rn);
    const int32_t vm_o =
        GetVRegOffset(args.rm);
    const int32_t vd_o =
        GetVRegOffset(args.rd);

    // Materialize a broadcast constant (`pattern` in both qwords).
    auto broadcast = [&](uint64_t pattern) -> FpRegister {
      FpRegister x = AllocTempSimdReg();
      Register gr =
          std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pattern)));
      builder_.Gen<x86_64::MovqXRegReg>(x.machine_reg(), gr);
      builder_.Gen<x86_64::PinsrqXRegRegImm>(x.machine_reg(), gr, int8_t{1});
      return x;
    };

    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_o);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_o);
    // Q=1 ("2") forms take the upper 64 bits of the narrow sources.
    if (args.q) {
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
      builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
    }

    // Compute the doubled, saturated product P into xP (128 bits: 4S or 2D).
    FpRegister xP = broadcast(args.size == 0b01 ? uint64_t{0x4000000040000000ULL}
                                                : uint64_t{0x4000000000000000ULL});
    if (args.size == 0b01) {
      builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
      // xP starts as INT16_MIN^2 (0x40000000 per lane); mark == lanes.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xP.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xn.machine_reg());  // double
      FpRegister xsat = broadcast(uint64_t{0x7FFFFFFF7FFFFFFFULL});
      builder_.Gen<x86_64::PandXRegXReg>(xsat.machine_reg(), xP.machine_reg());   // SMAX in sat lanes
      builder_.Gen<x86_64::PandnXRegXReg>(xP.machine_reg(), xn.machine_reg());    // doubled in non-sat
      builder_.Gen<x86_64::PorXRegXReg>(xP.machine_reg(), xsat.machine_reg());    // xP = P
    } else {  // size == 0b10
      builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PmuldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      // xP starts as INT32_MIN^2 (2^62 per qword); mark == lanes.
      builder_.Gen<x86_64::PcmpeqqXRegXReg>(xP.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xn.machine_reg());  // double
      FpRegister xsat = broadcast(uint64_t{0x7FFFFFFFFFFFFFFFULL});
      builder_.Gen<x86_64::PandXRegXReg>(xsat.machine_reg(), xP.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(xP.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xP.machine_reg(), xsat.machine_reg());
    }

    if (!is_acc) {
      SetVRegFull(args.rd, xP, /*q=*/true);
      return;
    }

    // Signed saturating accumulate: result = SignedSat(Vd +/- P). a = Vd.
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_o);
    FpRegister xres = AllocTempSimdReg();
    FpRegister xof = AllocTempSimdReg();
    FpRegister xtmp = AllocTempSimdReg();
    if (args.size == 0b01) {
      if (is_sub) {
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), xP.machine_reg());   // diff = a - P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xP.machine_reg());     // a ^ P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // a ^ diff
        builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
      } else {
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PadddXRegXReg>(xres.machine_reg(), xP.machine_reg());   // sum = a + P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xres.machine_reg());   // a ^ sum
        builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xP.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // P ^ sum
        builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
      }
      builder_.Gen<x86_64::PsradXRegImm>(xof.machine_reg(), int8_t{31});  // overflow lanes
      builder_.Gen<x86_64::PsradXRegImm>(xd.machine_reg(), int8_t{31});   // a's sign (0 / -1)
      FpRegister xmaxc = broadcast(uint64_t{0x7FFFFFFF7FFFFFFFULL});
      builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), xmaxc.machine_reg());   // sat = (a>>31)^INT32_MAX
      builder_.Gen<x86_64::PandXRegXReg>(xd.machine_reg(), xof.machine_reg());     // sat in overflow lanes
      builder_.Gen<x86_64::PandnXRegXReg>(xof.machine_reg(), xres.machine_reg());  // result in non-overflow
      builder_.Gen<x86_64::PorXRegXReg>(xd.machine_reg(), xof.machine_reg());
      SetVRegFull(args.rd, xd, /*q=*/true);
      return;
    }
    // size == 0b10 accumulate (64-bit signed saturating).
    FpRegister xzero = AllocZeroedSimdReg();
    if (is_sub) {
      builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PsubqXRegXReg>(xres.machine_reg(), xP.machine_reg());   // diff = a - P
      builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xP.machine_reg());     // a ^ P
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // a ^ diff
      builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
    } else {
      builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xP.machine_reg());   // sum = a + P
      builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xres.machine_reg());   // a ^ sum
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xP.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // P ^ sum
      builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
    }
    // overflow_mask (xtmp) = all-ones per qword where xof < 0 (sign set).
    builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xzero.machine_reg());
    builder_.Gen<x86_64::PcmpgtqXRegXReg>(xtmp.machine_reg(), xof.machine_reg());
    // sat = sign(a) ^ INT64_MAX  (INT64_MIN if a<0, else INT64_MAX).
    builder_.Gen<x86_64::PcmpgtqXRegXReg>(xzero.machine_reg(), xd.machine_reg());  // all-ones where a<0
    FpRegister xmaxc = broadcast(uint64_t{0x7FFFFFFFFFFFFFFFULL});
    builder_.Gen<x86_64::PxorXRegXReg>(xzero.machine_reg(), xmaxc.machine_reg());  // xzero = sat value
    builder_.Gen<x86_64::PandXRegXReg>(xzero.machine_reg(), xtmp.machine_reg());   // sat in overflow lanes
    builder_.Gen<x86_64::PandnXRegXReg>(xtmp.machine_reg(), xres.machine_reg());   // result in non-overflow
    builder_.Gen<x86_64::PorXRegXReg>(xzero.machine_reg(), xtmp.machine_reg());
    SetVRegFull(args.rd, xzero, /*q=*/true);
    return;
  }

  const bool is_mull = (args.opcode == Op::kSmull || args.opcode == Op::kUmull);
  const bool is_mlal = (args.opcode == Op::kSmlal || args.opcode == Op::kUmlal);
  const bool is_mlsl = (args.opcode == Op::kSmlsl || args.opcode == Op::kUmlsl);
  if ((!is_mull && !is_mlal && !is_mlsl) || args.size > 0b10) {
    UndefinedReturningVoid();
    return;
  }
  const bool is_signed = (args.opcode == Op::kSmull ||
                          args.opcode == Op::kSmlal ||
                          args.opcode == Op::kSmlsl);
  const int32_t vn_off =
      GetVRegOffset(args.rn);
  const int32_t vm_off =
      GetVRegOffset(args.rm);

  FpRegister xn = AllocTempSimdReg();
  FpRegister xm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
  builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
  if (args.q) {
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
    builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
  }
  switch (args.size) {
    case 0b00:
      if (is_signed) {
        builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
      }
      builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
      break;
    case 0b01:
      if (is_signed) {
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
      }
      builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
      break;
    case 0b10:
      if (is_signed) {
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmuldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmuludqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      break;
  }

  if (is_mull) {
    SetVRegFull(args.rd, xn, /*q=*/true);
    return;
  }

  const int32_t vd_off =
      GetVRegOffset(args.rd);
  FpRegister xd = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
  switch (args.size) {
    case 0b00:
      if (is_mlal) {
        builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::PsubwXRegXReg>(xd.machine_reg(), xn.machine_reg());
      }
      break;
    case 0b01:
      if (is_mlal) {
        builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::PsubdXRegXReg>(xd.machine_reg(), xn.machine_reg());
      }
      break;
    case 0b10:
      if (is_mlal) {
        builder_.Gen<x86_64::PaddqXRegXReg>(xd.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::PsubqXRegXReg>(xd.machine_reg(), xn.machine_reg());
      }
      break;
  }
  SetVRegFull(args.rd, xd, /*q=*/true);
}

// Heavy-tier mirror of the lite AdvSimdSingleStruct lowering
// (lite_translator.h). Covers the same subset the lite tier does: LD1R
// (replicate one element to every lane), single-element LD1 (load one lane),
// and single-element ST1 (store one lane), all with num_regs == 1 — critical
// for the dynamic linker's calculate_gnu_hash_neon tail (`ld1r v3.4s, [x10],
// #4`), which otherwise bails heavy->lite on every symbol resolve. Every
// element move routes mem<->lane through a GP temp (the heavy tier has no
// memory-operand PINSR/PEXTR, only the register forms), each guest-memory
// access carries a recovery block, and the v[] slot is always accessed
// full-width (GenGetSimd<16>/GenSetSimd<16>) so the optimizer's 16-byte
// slot store/load forwarding is never split by a narrow sub-lane access
// (see the UMOV/INS-element note above). LD1R starts from a zeroed XMM and
// only fills the active lanes, so Q=0 upper-half zeroing is automatic.
void HeavyOptimizerFrontend::AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::AdvSimdSingleStructOp;
  const bool is_replicate = args.is_replicate;
  const bool is_single_store = (args.op == Op::kSt1 || args.op == Op::kSt2 ||
                                args.op == Op::kSt3 || args.op == Op::kSt4);
  if (args.num_regs < 1 || args.num_regs > 4 || args.size > 3) {
    UndefinedReturningVoid();
    return;
  }
  const int esize = 1 << args.size;  // 1/2/4/8 bytes (B/H/S/D).
  const int32_t vec_bytes = args.q ? 16 : 8;

  Register base_orig = (args.rn == 31) ? GetSp() : GetReg(args.rn);
  Register base = ApplyTbi(base_orig);

  if (is_single_store) {
    // ST1-ST4 lane: for each of the num_regs consecutive registers (mod 32),
    // full-width v[reg] load, PEXTR lane->gp, MOV* gp->mem at consecutive
    // element offsets (+recovery). A fault on any element re-executes the
    // whole instruction from the interpreter; earlier partial stores are
    // architecturally permitted.
    const int8_t lane = static_cast<int8_t>(args.index);
    for (unsigned r = 0; r < args.num_regs; ++r) {
      const uint8_t reg = static_cast<uint8_t>((args.rt + r) & 31);
      const int32_t disp = static_cast<int32_t>(r) * esize;
      FpRegister xmm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xmm.machine_reg(), GetVRegOffset(reg));
      Register elem;
      switch (esize) {
        case 1:
          elem = std::get<0>(Gen<x86_64::PextrbRegXRegImm>(xmm.machine_reg(), lane));
          Gen<x86_64::MovbOpReg>({.base = base, .disp = disp}, elem);
          break;
        case 2:
          elem = std::get<0>(Gen<x86_64::PextrwRegXRegImm>(xmm.machine_reg(), lane));
          Gen<x86_64::MovwOpReg>({.base = base, .disp = disp}, elem);
          break;
        case 4:
          elem = std::get<0>(Gen<x86_64::PextrdRegXRegImm>(xmm.machine_reg(), lane));
          Gen<x86_64::MovlOpReg>({.base = base, .disp = disp}, elem);
          break;
        default:  // esize == 8
          elem = std::get<0>(Gen<x86_64::PextrqRegXRegImm>(xmm.machine_reg(), lane));
          Gen<x86_64::MovqOpReg>({.base = base, .disp = disp}, elem);
          break;
      }
      GenRecoveryBlockForLastInsn();
    }
  } else if (is_replicate) {
    // LD1R-LD4R: each of the num_regs consecutive registers gets its OWN
    // element, loaded from consecutive memory offsets (+recovery), broadcast
    // via PINSR into every lane. Starting from a zeroed XMM leaves the upper
    // 64 bits zero for Q=0.
    const int num_lanes = vec_bytes / esize;
    for (unsigned r = 0; r < args.num_regs; ++r) {
      const uint8_t reg = static_cast<uint8_t>((args.rt + r) & 31);
      const int32_t disp = static_cast<int32_t>(r) * esize;
      FpRegister xmm = AllocZeroedSimdReg();
      Register elem;
      switch (esize) {
        case 1:
          elem = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = base, .disp = disp}));
          break;
        case 2:
          elem = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = base, .disp = disp}));
          break;
        case 4:
          elem = std::get<0>(Gen<x86_64::MovlRegOp>({.base = base, .disp = disp}));
          break;
        default:  // esize == 8
          elem = std::get<0>(Gen<x86_64::MovqRegOp>({.base = base, .disp = disp}));
          break;
      }
      GenRecoveryBlockForLastInsn();
      for (int l = 0; l < num_lanes; l++) {
        const int8_t lane = static_cast<int8_t>(l);
        switch (esize) {
          case 1:
            builder_.Gen<x86_64::PinsrbXRegRegImm>(xmm.machine_reg(), elem, lane);
            break;
          case 2:
            builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), elem, lane);
            break;
          case 4:
            builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), elem, lane);
            break;
          default:  // esize == 8
            builder_.Gen<x86_64::PinsrqXRegRegImm>(xmm.machine_reg(), elem, lane);
            break;
        }
      }
      builder_.GenSetSimd<16>(GetVRegOffset(reg), xmm.machine_reg());
    }
  } else {
    // LD1-LD4 single lane: for each of the num_regs consecutive registers,
    // load full-width v[reg], PINSR the element loaded from its consecutive
    // memory offset into lane[index] (preserving the other lanes), store back
    // full-width.
    const int8_t lane = static_cast<int8_t>(args.index);
    for (unsigned r = 0; r < args.num_regs; ++r) {
      const uint8_t reg = static_cast<uint8_t>((args.rt + r) & 31);
      const int32_t disp = static_cast<int32_t>(r) * esize;
      FpRegister xmm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xmm.machine_reg(), GetVRegOffset(reg));
      Register elem;
      switch (esize) {
        case 1:
          elem = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = base, .disp = disp}));
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::PinsrbXRegRegImm>(xmm.machine_reg(), elem, lane);
          break;
        case 2:
          elem = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = base, .disp = disp}));
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), elem, lane);
          break;
        case 4:
          elem = std::get<0>(Gen<x86_64::MovlRegOp>({.base = base, .disp = disp}));
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), elem, lane);
          break;
        default:  // esize == 8
          elem = std::get<0>(Gen<x86_64::MovqRegOp>({.base = base, .disp = disp}));
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::PinsrqXRegRegImm>(xmm.machine_reg(), elem, lane);
          break;
      }
      builder_.GenSetSimd<16>(GetVRegOffset(reg), xmm.machine_reg());
    }
  }

  if (args.postindex) {
    // Writeback preserves the original (un-TBI-masked) top byte, so re-read
    // the base register rather than reusing the masked access address.
    Register reread_base = (args.rn == 31) ? GetSp() : GetReg(args.rn);
    Register new_base = Copy(reread_base);
    if (args.rm == 31) {
      // Immediate post-index: total bytes accessed = num_regs * esize.
      new_base = std::get<0>(Gen<x86_64::AddqRegImm, kNoSSA>(
          new_base, static_cast<int32_t>(args.num_regs) * esize));
    } else {
      Register rm_val = GetReg(args.rm);
      new_base = std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(new_base, rm_val));
    }
    if (args.rn == 31) {
      SetSp(new_base);
    } else {
      SetReg(args.rn, new_base);
    }
  }
}

// Heavy-tier mirror of the register-domain-pure AdvSIMD two-reg-misc opcodes
// the lite translator already lowers: REV16, CNT, NOT/RBIT, NEG, ABS. All are
// packed SSE sequences with no host FLAGS and no memory operand beyond the
// guest v[] load/store, so they map straight onto the MachineIR builder the
// same way AdvSimdThreeSame does. Every other opcode bails to the lite tier
// (UndefinedReturningVoid) — emit NOTHING before a bail. Q=0 upper-half zeroing
// is handled by SetVRegFull. The lowerings match lite_translator.h exactly.
void HeavyOptimizerFrontend::AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& args) {
  if (!success()) {
    return;
  }
  const int32_t vn_off = GetVRegOffset(args.rn);

  switch (args.opcode) {
    // BFCVTN/BFCVTN2: narrow 4 FP32 lanes to 4 BF16 lanes. BF16 is the top 16
    // bits of an FP32, so the conversion is a round-to-nearest-even of the
    // discarded low half plus one architectural exception: a NaN input must
    // produce a quiet BF16 NaN rather than whatever the rounded truncation
    // happens to give, because rounding can carry into the exponent and turn a
    // NaN into an infinity.
    //
    //   rounded  = (n + ((n >> 16) & 1) + 0x7FFF) >> 16     (RTNE)
    //   nan_val  = (n >> 16) | 0x0040                       (quiet bit set)
    //   nan_mask = (exp == 0xFF) & (mantissa != 0)
    //   out      = nan_mask ? nan_val : rounded
    //
    // Branchless throughout, op-for-op mirror of the lite tier's recipe so both
    // tiers round identically. Q selects which half of Vd is written: BFCVTN
    // (Q=0) writes the low 64 bits and zeroes the rest, BFCVTN2 (Q=1) writes the
    // high 64 bits and preserves the low half of Vd.
    case Decoder::AdvSimdTwoRegMiscOpcode::kBfcvtn: {
      if (args.size != 0b10) {
        UndefinedReturningVoid();
        return;
      }
      // Constants are materialized with AllocOnesSimdReg/AllocZeroedSimdReg
      // rather than the assembler idiom of self-PCMPEQD/self-PXOR on a fresh
      // temp: in MachineIR that reads the register before anything defines it
      // and trips the lifetime analyzer's use-before-def check.
      FpRegister xn = AllocTempSimdReg();
      FpRegister xnan_val = AllocTempSimdReg();
      FpRegister xtmp = AllocTempSimdReg();
      FpRegister xexp = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);

      // nan_val = (n >> 16) | 0x0040.
      FpRegister xone = AllocOnesSimdReg();
      builder_.Gen<x86_64::PsrldXRegImm>(xone.machine_reg(), int8_t{31});  // 0x00000001
      builder_.Gen<x86_64::MovdqaXRegXReg>(xnan_val.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(xnan_val.machine_reg(), int8_t{16});
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xone.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(xtmp.machine_reg(), int8_t{6});  // 0x00000040
      builder_.Gen<x86_64::PorXRegXReg>(xnan_val.machine_reg(), xtmp.machine_reg());

      // bias = ((n >> 16) & 1) + 0x7FFF.
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(xtmp.machine_reg(), int8_t{16});
      builder_.Gen<x86_64::PandXRegXReg>(xtmp.machine_reg(), xone.machine_reg());
      FpRegister x7fff = AllocOnesSimdReg();
      builder_.Gen<x86_64::PsrldXRegImm>(x7fff.machine_reg(), int8_t{17});  // 0x00007FFF
      builder_.Gen<x86_64::PadddXRegXReg>(xtmp.machine_reg(), x7fff.machine_reg());

      // rounded = (n + bias) >> 16, kept in the low 16 bits of each dword. The
      // original lanes are needed afterwards for the NaN test, so keep a copy.
      FpRegister xorig = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(xorig.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xtmp.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{16});

      // nan_mask = (n & 0x7F800000) == 0x7F800000  AND  (n & 0x007FFFFF) != 0.
      FpRegister xexpmask = AllocOnesSimdReg();
      builder_.Gen<x86_64::PsrldXRegImm>(xexpmask.machine_reg(), int8_t{24});  // 0x000000FF
      builder_.Gen<x86_64::PslldXRegImm>(xexpmask.machine_reg(), int8_t{23});  // 0x7F800000
      builder_.Gen<x86_64::MovdqaXRegXReg>(xexp.machine_reg(), xorig.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xexp.machine_reg(), xexpmask.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xexp.machine_reg(), xexpmask.machine_reg());

      FpRegister xmantmask = AllocOnesSimdReg();
      builder_.Gen<x86_64::PsrldXRegImm>(xmantmask.machine_reg(), int8_t{9});  // 0x007FFFFF
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xorig.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xtmp.machine_reg(), xmantmask.machine_reg());
      FpRegister xzero = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xtmp.machine_reg(), xzero.machine_reg());
      FpRegister xallones = AllocOnesSimdReg();
      builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xallones.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xexp.machine_reg(), xtmp.machine_reg());

      // out = (nan_mask & nan_val) | (~nan_mask & rounded).
      builder_.Gen<x86_64::PandXRegXReg>(xnan_val.machine_reg(), xexp.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xexp.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xnan_val.machine_reg(), xtmp.machine_reg());

      // 4 dwords -> 4 words. PACKUSDW saturates unsigned, and every dword here
      // holds a 16-bit value, so no lane is clamped; the result is duplicated
      // into both 64-bit halves.
      builder_.Gen<x86_64::PackusdwXRegXReg>(xnan_val.machine_reg(), xnan_val.machine_reg());

      if (!args.q) {
        // BFCVTN: low 64 bits of Vd, upper 64 zeroed.
        builder_.Gen<x86_64::PslldqXRegImm>(xnan_val.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrldqXRegImm>(xnan_val.machine_reg(), int8_t{8});
        SetVRegFull(args.rd, xnan_val, /*q=*/true);
      } else {
        // BFCVTN2: high 64 bits of Vd, low half preserved.
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), GetVRegOffset(args.rd));
        builder_.Gen<x86_64::PslldqXRegImm>(xd.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrldqXRegImm>(xd.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PslldqXRegImm>(xnan_val.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PorXRegXReg>(xnan_val.machine_reg(), xd.machine_reg());
        SetVRegFull(args.rd, xnan_val, /*q=*/true);
      }
      return;
    }

    // REV16 V.<T>, V.<T> (size=00 only): reverse byte order within each 16-bit
    // lane via (Vn << 8) | (Vn >> 8) per halfword. PSLLW/PSRLW shift the 16-bit
    // lanes; OR recombines the swapped bytes without a PSHUFB mask table.
    case Decoder::AdvSimdTwoRegMiscOpcode::kRev16: {
      if (args.size != 0b00) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xt = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.Gen<x86_64::MovdqaXRegXReg>(xt.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{8});
      builder_.Gen<x86_64::PsrlwXRegImm>(xt.machine_reg(), int8_t{8});
      builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xt.machine_reg());
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // REV64 V.<T>, V.<T> (U=0): reverse element order within each 64-bit
    // doubleword. size=00 byte-reverse via PSHUFB + a materialized per-lane
    // byte-index mask; size=01 halfword-reverse via PSHUFLW then PSHUFHW
    // (imm=0x1B reverses the four words in each 64-bit half); size=10
    // word-reverse via PSHUFD (imm=0xB1 swaps the two 32-bit words in each
    // 64-bit half). The Q=0 forms shuffle both halves and let SetVRegFull
    // discard the upper 64. size=11 is reserved and bails.
    case Decoder::AdvSimdTwoRegMiscOpcode::kRev64: {
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (args.size) {
        case 0b00: {
          FpRegister xmask = AllocTempSimdReg();
          Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0001020304050607LL}));
          builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), mlo);
          Register mhi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x08090A0B0C0D0E0FLL}));
          builder_.Gen<x86_64::PinsrqXRegRegImm>(xmask.machine_reg(), mhi, int8_t{1});
          builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), xmask.machine_reg());
          break;
        }
        case 0b01:
          builder_.Gen<x86_64::PshuflwXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                   int8_t{0x1B});
          builder_.Gen<x86_64::PshufhwXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                   int8_t{0x1B});
          break;
        case 0b10:
          builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                  static_cast<int8_t>(0xB1));
          break;
        default:  // 0b11 reserved
          UndefinedReturningVoid();
          return;
      }
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // REV32 V.<T>, V.<T> (U=1): reverse element order within each 32-bit
    // word. size=00 byte-reverse via PSHUFB + mask; size=01 halfword-swap
    // via PSHUFLW then PSHUFHW (imm=0xB1 swaps the two words in each 32-bit
    // lane). size>=10 is reserved for REV32 and bails.
    case Decoder::AdvSimdTwoRegMiscOpcode::kRev32: {
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (args.size) {
        case 0b00: {
          FpRegister xmask = AllocTempSimdReg();
          Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0405060700010203LL}));
          builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), mlo);
          Register mhi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0C0D0E0F08090A0BLL}));
          builder_.Gen<x86_64::PinsrqXRegRegImm>(xmask.machine_reg(), mhi, int8_t{1});
          builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), xmask.machine_reg());
          break;
        }
        case 0b01:
          builder_.Gen<x86_64::PshuflwXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                   static_cast<int8_t>(0xB1));
          builder_.Gen<x86_64::PshufhwXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                   static_cast<int8_t>(0xB1));
          break;
        default:  // size >= 0b10 reserved
          UndefinedReturningVoid();
          return;
      }
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // CNT V.16B / V.8B (size=00): per-byte population count via the
    // Mula-Wojcik nibble-LUT (two PSHUFB lookups on the low/high nibbles,
    // summed with PADDB). Table = popcount-per-nibble; mask = 0x0F broadcast.
    case Decoder::AdvSimdTwoRegMiscOpcode::kCnt: {
      if (args.size != 0b00) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xtable_lo = AllocTempSimdReg();
      FpRegister xtable_hi = AllocTempSimdReg();
      FpRegister xmask = AllocTempSimdReg();
      FpRegister xn = AllocTempSimdReg();
      FpRegister xt_hi = AllocTempSimdReg();
      // Build the popcount nibble table {0,1,1,2,...,4} into xtable_lo.
      Register tlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0302020102010100LL}));
      builder_.Gen<x86_64::MovqXRegReg>(xtable_lo.machine_reg(), tlo);
      Register thi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0403030203020201LL}));
      builder_.Gen<x86_64::PinsrqXRegRegImm>(xtable_lo.machine_reg(), thi, int8_t{1});
      // PSHUFB is destructive; keep a second copy of the table.
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtable_hi.machine_reg(), xtable_lo.machine_reg());
      // Build the 0x0F low-nibble mask, broadcast to all 16 bytes.
      Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0F0F0F0F0F0F0F0FLL}));
      builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), mlo);
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmask.machine_reg(), xmask.machine_reg());
      // Split Vn into low and high nibbles.
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.Gen<x86_64::MovdqaXRegXReg>(xt_hi.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsrlwXRegImm>(xt_hi.machine_reg(), int8_t{4});
      builder_.Gen<x86_64::PandXRegXReg>(xt_hi.machine_reg(), xmask.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xmask.machine_reg());
      // Vd = PSHUFB(table, low_nibbles) + PSHUFB(table, high_nibbles).
      builder_.Gen<x86_64::PshufbXRegXReg>(xtable_lo.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PshufbXRegXReg>(xtable_hi.machine_reg(), xt_hi.machine_reg());
      builder_.Gen<x86_64::PaddbXRegXReg>(xtable_lo.machine_reg(), xtable_hi.machine_reg());
      SetVRegFull(args.rd, xtable_lo, args.q);
      return;
    }

    // NOT V.16B / V.8B (size=00): per-lane bitwise complement Vd = ~Vn.
    // RBIT V.16B / V.8B (size=01): per-byte bit reversal via two PSHUFB
    // nibble-LUTs (reverse_bits(b) = (reverse4(L)<<4) | reverse4(H)), the two
    // results occupying disjoint nibble positions and combined with POR.
    case Decoder::AdvSimdTwoRegMiscOpcode::kNot: {
      if (args.size == 0b00) {
        FpRegister xn = AllocTempSimdReg();
        FpRegister allones = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(allones.machine_reg(), allones.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), allones.machine_reg());
        SetVRegFull(args.rd, xn, args.q);
        return;
      }
      if (args.size == 0b01) {
        FpRegister xlow_table = AllocTempSimdReg();
        FpRegister xhigh_table = AllocTempSimdReg();
        FpRegister xmask = AllocTempSimdReg();
        FpRegister xn = AllocTempSimdReg();
        FpRegister xn_hi = AllocTempSimdReg();
        // low_table[i]  = reverse4(i)       (result in low nibble)
        Register lt_lo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0E060A020C040800LL}));
        builder_.Gen<x86_64::MovqXRegReg>(xlow_table.machine_reg(), lt_lo);
        Register lt_hi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0F070B030D050901LL}));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(xlow_table.machine_reg(), lt_hi, int8_t{1});
        // high_table[i] = reverse4(i) << 4  (result in high nibble)
        Register ht_lo =
            std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0xE060A020C0408000ULL)));
        builder_.Gen<x86_64::MovqXRegReg>(xhigh_table.machine_reg(), ht_lo);
        Register ht_hi =
            std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0xF070B030D0509010ULL)));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(xhigh_table.machine_reg(), ht_hi, int8_t{1});
        // Build the 0x0F broadcast mask.
        Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0F0F0F0F0F0F0F0FLL}));
        builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), mlo);
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmask.machine_reg(), xmask.machine_reg());
        // t_lo = low nibbles, t_hi = high nibbles.
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn_hi.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsrlwXRegImm>(xn_hi.machine_reg(), int8_t{4});
        builder_.Gen<x86_64::PandXRegXReg>(xn_hi.machine_reg(), xmask.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xmask.machine_reg());
        // Vd = PSHUFB(high_table, t_lo) | PSHUFB(low_table, t_hi).
        builder_.Gen<x86_64::PshufbXRegXReg>(xhigh_table.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PshufbXRegXReg>(xlow_table.machine_reg(), xn_hi.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(xhigh_table.machine_reg(), xlow_table.machine_reg());
        SetVRegFull(args.rd, xhigh_table, args.q);
        return;
      }
      UndefinedReturningVoid();
      return;
    }

    // NEG V.<T>, V.<T>: per-lane integer negation Vd = 0 - Vn (PSUBB/W/D/Q).
    // size=11 with Q=0 (.1D) is reserved and bails.
    case Decoder::AdvSimdTwoRegMiscOpcode::kNeg: {
      if (args.size == 0b11 && !args.q) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xz = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PsubbXRegXReg>(xz.machine_reg(), xn.machine_reg());
          break;
        case 0b01:
          builder_.Gen<x86_64::PsubwXRegXReg>(xz.machine_reg(), xn.machine_reg());
          break;
        case 0b10:
          builder_.Gen<x86_64::PsubdXRegXReg>(xz.machine_reg(), xn.machine_reg());
          break;
        default:  // 0b11 (.2D, Q=1)
          builder_.Gen<x86_64::PsubqXRegXReg>(xz.machine_reg(), xn.machine_reg());
          break;
      }
      SetVRegFull(args.rd, xz, args.q);
      return;
    }

    // ABS V.<T>, V.<T>: per-lane integer absolute value via
    // (Vn ^ sign_mask) - sign_mask, sign_mask = PCMPGT(0, Vn) = -1 if Vn<0.
    // size=11 (.2D) needs PCMPGTQ (not allowlisted) and bails.
    case Decoder::AdvSimdTwoRegMiscOpcode::kAbs: {
      // .2D (size=0b11) uses PCMPGTQ for the sign mask; gate on SSE4.2.
      if (args.size == 0b11 && !host_platform::kHasSSE4_2) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister mask = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PcmpgtbXRegXReg>(mask.machine_reg(), xn.machine_reg());
          break;
        case 0b01:
          builder_.Gen<x86_64::PcmpgtwXRegXReg>(mask.machine_reg(), xn.machine_reg());
          break;
        case 0b10:
          builder_.Gen<x86_64::PcmpgtdXRegXReg>(mask.machine_reg(), xn.machine_reg());
          break;
        default:  // 0b11 (.2D)
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(mask.machine_reg(), xn.machine_reg());
          break;
      }
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), mask.machine_reg());
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PsubbXRegXReg>(xn.machine_reg(), mask.machine_reg());
          break;
        case 0b01:
          builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), mask.machine_reg());
          break;
        case 0b10:
          builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), mask.machine_reg());
          break;
        default:  // 0b11 (.2D)
          builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), mask.machine_reg());
          break;
      }
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // CMEQ Vd.<T>, Vn.<T>, #0 — per-lane integer compare-equal against zero.
    // PCMPEQ{B,W,D,Q} against a zeroed register.
    case Decoder::AdvSimdTwoRegMiscOpcode::kCmeqZero: {
      if (args.is_fp16) {
        // FP16 FCMEQ Vd,Vn,#0.0 — per-16-bit-lane FP compare vs +0.0. (Without this,
        // the integer PCMPEQB path below miscompiles FP16: e.g. -0.0h=0x8000 must give
        // 0xFFFF but a byte compare gives 0x00FF.) Widen -> Cmpeqps vs +0.0 -> PACKSSDW.
        if (!host_platform::kHasF16C) { UndefinedReturningVoid(); return; }
        FpRegister lo = no_fp_register, hi = no_fp_register;
        EmitWidenHalfVec(vn_off, args.q, &lo, &hi);
        FpRegister z = AllocZeroedSimdReg();  // +0.0 per lane; Cmpeqps leaves src z intact
        builder_.Gen<x86_64::CmpeqpsXRegXReg>(lo.machine_reg(), z.machine_reg());
        if (args.q) {
          builder_.Gen<x86_64::CmpeqpsXRegXReg>(hi.machine_reg(), z.machine_reg());
          builder_.Gen<x86_64::PackssdwXRegXReg>(lo.machine_reg(), hi.machine_reg());
          SetVRegFull(args.rd, lo, true);
        } else {
          builder_.Gen<x86_64::PackssdwXRegXReg>(lo.machine_reg(), lo.machine_reg());
          SetVRegFull(args.rd, lo, false);
        }
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xz = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PcmpeqbXRegXReg>(xn.machine_reg(), xz.machine_reg());
          break;
        case 0b01:
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn.machine_reg(), xz.machine_reg());
          break;
        case 0b10:
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(xn.machine_reg(), xz.machine_reg());
          break;
        default:  // 0b11 (.2D): PCMPEQQ (SSE4.1).
          builder_.Gen<x86_64::PcmpeqqXRegXReg>(xn.machine_reg(), xz.machine_reg());
          break;
      }
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // CMGT/CMGE/CMLE/CMLT Vd.<T>, Vn.<T>, #0 — per-lane signed integer
    // compare against zero. CMGT/CMLE compute (Vn > 0) via PCMPGT(Vn, 0);
    // CMGE/CMLT compute (0 > Vn) via PCMPGT(0, Vn). CMGE = NOT(0 > Vn) and
    // CMLE = NOT(Vn > 0), inverted with XOR against all-ones. size=11 (.2D)
    // needs PCMPGTQ (not allowlisted) and bails; FP16 lanes bail too.
    case Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero:
    case Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero:
    case Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero:
    case Decoder::AdvSimdTwoRegMiscOpcode::kCmltZero: {
      if (args.is_fp16) {
        // FP16 FCMGT/FCMGE/FCMLT/FCMLE Vd,Vn,#0.0 — per-16-bit-lane FP compare vs +0.0.
        //   FCMGT#0: Vn>0  <=> 0<Vn   -> Cmpltps(z, Vn)   [mask in z]
        //   FCMGE#0: Vn>=0 <=> 0<=Vn  -> Cmpleps(z, Vn)   [mask in z]
        //   FCMLT#0: Vn<0             -> Cmpltps(Vn, z)   [mask in Vn]
        //   FCMLE#0: Vn<=0            -> Cmpleps(Vn, z)   [mask in Vn]
        // Ordered SSE predicates return 0 for NaN (ARM unordered-is-false). PACKSSDW
        // narrows the FP32 mask to 16-bit lanes. Heavy is a correct superset here (lite
        // bails these to the interpreter).
        if (!host_platform::kHasF16C) { UndefinedReturningVoid(); return; }
        using Op = Decoder::AdvSimdTwoRegMiscOpcode;
        FpRegister lo = no_fp_register, hi = no_fp_register;
        EmitWidenHalfVec(vn_off, args.q, &lo, &hi);
        // Uses a fresh zero per half (the gt/ge forms clobber the zero reg).
        auto cmp = [&](FpRegister v) -> FpRegister {
          FpRegister z = AllocZeroedSimdReg();
          switch (args.opcode) {
            case Op::kCmgtZero:
              builder_.Gen<x86_64::CmpltpsXRegXReg>(z.machine_reg(), v.machine_reg()); return z;
            case Op::kCmgeZero:
              builder_.Gen<x86_64::CmplepsXRegXReg>(z.machine_reg(), v.machine_reg()); return z;
            case Op::kCmltZero:
              builder_.Gen<x86_64::CmpltpsXRegXReg>(v.machine_reg(), z.machine_reg()); return v;
            default:  // kCmleZero
              builder_.Gen<x86_64::CmplepsXRegXReg>(v.machine_reg(), z.machine_reg()); return v;
          }
        };
        FpRegister rlo = cmp(lo);
        if (args.q) {
          FpRegister rhi = cmp(hi);
          builder_.Gen<x86_64::PackssdwXRegXReg>(rlo.machine_reg(), rhi.machine_reg());
          SetVRegFull(args.rd, rlo, true);
        } else {
          builder_.Gen<x86_64::PackssdwXRegXReg>(rlo.machine_reg(), rlo.machine_reg());
          SetVRegFull(args.rd, rlo, false);
        }
        return;
      }
      // .2D (size=0b11) uses PCMPGTQ; gate on SSE4.2.
      if (args.size == 0b11 && !host_platform::kHasSSE4_2) {
        UndefinedReturningVoid();
        return;
      }
      const bool n_gt_z =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero) ||
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero);
      const bool invert =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero) ||
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xz = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // For (Vn > 0) the result lands in xn; for (0 > Vn) it lands in xz.
      FpRegister res = n_gt_z ? xn : xz;
      FpRegister a = n_gt_z ? xn : xz;
      FpRegister b = n_gt_z ? xz : xn;
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PcmpgtbXRegXReg>(a.machine_reg(), b.machine_reg());
          break;
        case 0b01:
          builder_.Gen<x86_64::PcmpgtwXRegXReg>(a.machine_reg(), b.machine_reg());
          break;
        case 0b10:
          builder_.Gen<x86_64::PcmpgtdXRegXReg>(a.machine_reg(), b.machine_reg());
          break;
        default:  // 0b11 (.2D)
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(a.machine_reg(), b.machine_reg());
          break;
      }
      if (invert) {
        FpRegister ones = AllocOnesSimdReg();
        builder_.Gen<x86_64::PxorXRegXReg>(res.machine_reg(), ones.machine_reg());
      }
      SetVRegFull(args.rd, res, args.q);
      return;
    }

    // FCMEQ/FCMGT/FCMGE/FCMLT/FCMLE Vd.<T>, Vn.<T>, #0.0 — per-lane FP compare
    // against zero (.2S/.4S FP32, .2D FP64). Mirrors lite_translator.h's
    // non-FP16 path: SSE legacy CMP{EQ,LT,LE}P{S,D} are ordered — they write
    // FALSE (zero) on any NaN operand, matching ARM's unordered-is-false rule.
    // Result lane is all-ones on TRUE, zero on FALSE.
    //   FCMEQ: Vn == 0 -> CMPEQP* xn, xz            (result in xn)
    //   FCMLT: Vn <  0 -> CMPLTP* xn, xz            (result in xn)
    //   FCMLE: Vn <= 0 -> CMPLEP* xn, xz            (result in xn)
    //   FCMGT: Vn >  0 <=> 0 < Vn -> CMPLTP* xz, xn (result in xz)
    //   FCMGE: Vn >= 0 <=> 0 <= Vn -> CMPLEP* xz, xn (result in xz)
    // size encodes bit23:bit22; FP forms always have bit23=1 -> size&0b10.
    // .2D (size=0b11) needs Q=1 (.1D reserved). FP16 lives in a different
    // encoding slot and bails to lite (interpreter handles it).
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcmeqZero:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgtZero:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcmgeZero:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcmleZero:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcmltZero: {
      if (args.is_fp16 || (args.size & 0b10) == 0) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_double = (args.size == 0b11);
      if (is_double && !args.q) {  // .1D reserved
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xz = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      using Op = Decoder::AdvSimdTwoRegMiscOpcode;
      FpRegister res = xn;
      switch (args.opcode) {
        case Op::kFcmeqZero:
          if (is_double)
            builder_.Gen<x86_64::CmpeqpdXRegXReg>(xn.machine_reg(), xz.machine_reg());
          else
            builder_.Gen<x86_64::CmpeqpsXRegXReg>(xn.machine_reg(), xz.machine_reg());
          break;
        case Op::kFcmltZero:
          if (is_double)
            builder_.Gen<x86_64::CmpltpdXRegXReg>(xn.machine_reg(), xz.machine_reg());
          else
            builder_.Gen<x86_64::CmpltpsXRegXReg>(xn.machine_reg(), xz.machine_reg());
          break;
        case Op::kFcmleZero:
          if (is_double)
            builder_.Gen<x86_64::CmplepdXRegXReg>(xn.machine_reg(), xz.machine_reg());
          else
            builder_.Gen<x86_64::CmplepsXRegXReg>(xn.machine_reg(), xz.machine_reg());
          break;
        case Op::kFcmgtZero:
          // Vn > 0 <=> 0 < Vn; mask lands in xz.
          if (is_double)
            builder_.Gen<x86_64::CmpltpdXRegXReg>(xz.machine_reg(), xn.machine_reg());
          else
            builder_.Gen<x86_64::CmpltpsXRegXReg>(xz.machine_reg(), xn.machine_reg());
          res = xz;
          break;
        case Op::kFcmgeZero:
          // Vn >= 0 <=> 0 <= Vn; mask lands in xz.
          if (is_double)
            builder_.Gen<x86_64::CmplepdXRegXReg>(xz.machine_reg(), xn.machine_reg());
          else
            builder_.Gen<x86_64::CmplepsXRegXReg>(xz.machine_reg(), xn.machine_reg());
          res = xz;
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      SetVRegFull(args.rd, res, args.q);
      return;
    }

    // XTN/XTN2 Vd.<Tb>, Vn.<Ta> — truncating narrow: keep the low half of each
    // element. size=00 (8H->8B) / size=01 (4S->4H): mask off the high half of
    // every lane, then PACKUSWB/PACKUSDW into the low 64 (the mask guarantees
    // all values sit in the unsigned pack's non-saturating range, so the pack
    // is a pure truncation). size=10 (.2D->.2S): PSHUFD gathers dwords {0,2}
    // into the low 64. Q=0 zero-extends the upper 64; Q=1 (XTN2) merges into
    // Vd's high 64. size=11 is reserved and bails. Mirrors lite kXtn.
    case Decoder::AdvSimdTwoRegMiscOpcode::kXtn: {
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (args.size == 0b00 || args.size == 0b01) {
        FpRegister xz = AllocZeroedSimdReg();
        FpRegister xm = AllocTempSimdReg();
        Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(
            args.size == 0b00 ? int64_t{0x00FF00FF00FF00FFLL}
                              : int64_t{0x0000FFFF0000FFFFLL}));
        builder_.Gen<x86_64::MovqXRegReg>(xm.machine_reg(), mlo);
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
        } else {
          builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
        }
      } else if (args.size == 0b10) {
        builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                int8_t{0b00001000});
      } else {  // size=11 reserved
        UndefinedReturningVoid();
        return;
      }
      SetVRegNarrow(args.rd, xn, args.q);
      return;
    }

    // SHLL/SHLL2 Vd.<Ta>, Vn.<Tb>, #<esize> — shift-left-long: zero-extend each
    // narrow lane, then shift left by the source element width (8/16/32). Q=0
    // widens Vn's low 8 bytes; Q=1 (SHLL2) the high 8 bytes (PSRLDQ brings them
    // low first). The widened result always fills all 128 bits, so it is stored
    // full regardless of Q. size=11 is unallocated and bails. Mirrors lite kShll.
    case Decoder::AdvSimdTwoRegMiscOpcode::kShll: {
      if (args.size > 0b10) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (args.q) {
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
      }
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{8});
          break;
        case 0b01:
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), int8_t{16});
          break;
        default:  // 0b10
          builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsllqXRegImm>(xn.machine_reg(), int8_t{32});
          break;
      }
      SetVRegFull(args.rd, xn, /*q=*/true);
      return;
    }

    // SQXTN/UQXTN/SQXTUN Vd.<Tb>, Vn.<Ta> — saturating extract narrow. Same
    // dst-width and Q layout as XTN; the pack flavour differs by saturation:
    //   SQXTN  signed->signed    : PACKSSWB / PACKSSDW
    //   SQXTUN signed->unsigned  : PACKUSWB / PACKUSDW
    //   UQXTN  unsigned->unsigned : PMINUW/PMINUD clamp to the unsigned dst max
    //                               (keeps values in the positive signed range
    //                               so the following PACKUS is exact), then
    //                               PACKUSWB / PACKUSDW.
    // size=00 (8H->8B) and size=01 (4S->4H) use the x86 narrowing packs.
    // size=10 (.2D->.2S) has no 64->32 x86 pack, so each 64-bit lane is
    // clamped into the destination range with PCMPGTQ/PCMPEQQ (SSE4.2/4.1)
    // masked blends and the two low dwords gathered with PSHUFD — a bit-exact
    // mirror of lite kSqxtn/kUqxtn/kSqxtun's size=10 path. size=11 is reserved.
    case Decoder::AdvSimdTwoRegMiscOpcode::kSqxtn:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUqxtn:
    case Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun: {
      const auto opc = args.opcode;
      if (args.size == 0b10) {
        FpRegister x = AllocTempSimdReg();
        builder_.GenGetSimd<16>(x.machine_reg(), vn_off);
        // Broadcast a 64-bit constant into both lanes.
        auto set_const = [&](FpRegister r, int64_t v) {
          Register gp = std::get<0>(Gen<x86_64::MovqRegImm>(v));
          builder_.Gen<x86_64::MovqXRegReg>(r.machine_reg(), gp);
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(r.machine_reg(), r.machine_reg());
        };
        // x = (x & ~mask) | (val & mask), via x ^= (x ^ val) & mask.
        auto blend = [&](FpRegister val, FpRegister mask) {
          FpRegister t = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), val.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(t.machine_reg(), mask.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(x.machine_reg(), t.machine_reg());
        };
        if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtn) {
          FpRegister c = AllocTempSimdReg();
          FpRegister m = AllocTempSimdReg();
          set_const(c, int64_t{0x000000007FFFFFFFLL});  // INT32_MAX
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());  // x > MAX
          blend(c, m);
          set_const(c, static_cast<int64_t>(0xFFFFFFFF80000000ULL));  // INT32_MIN
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), c.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), x.machine_reg());  // x < MIN
          blend(c, m);
        } else if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun) {
          FpRegister c = AllocTempSimdReg();
          FpRegister m = AllocTempSimdReg();
          set_const(c, int64_t{0x00000000FFFFFFFFLL});  // UINT32_MAX
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());  // x > UMAX
          blend(c, m);
          FpRegister z = AllocZeroedSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), z.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), x.machine_reg());  // x < 0
          builder_.Gen<x86_64::PandnXRegXReg>(m.machine_reg(), x.machine_reg());     // neg -> 0
          builder_.Gen<x86_64::MovdqaXRegXReg>(x.machine_reg(), m.machine_reg());
        } else {  // kUqxtn: unsigned uint64 -> clamp to UINT32_MAX
          FpRegister c = AllocTempSimdReg();
          FpRegister m = AllocTempSimdReg();
          set_const(c, int64_t{0x00000000FFFFFFFFLL});  // UINT32_MAX
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PsrlqXRegImm>(m.machine_reg(), int8_t{32});  // high 32 bits
          FpRegister z = AllocZeroedSimdReg();
          builder_.Gen<x86_64::PcmpeqqXRegXReg>(m.machine_reg(), z.machine_reg());  // high32==0
          FpRegister t = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), c.machine_reg());
          builder_.Gen<x86_64::PandnXRegXReg>(m.machine_reg(), t.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(x.machine_reg(), m.machine_reg());
        }
        builder_.Gen<x86_64::PshufdXRegXRegImm>(x.machine_reg(), x.machine_reg(),
                                                int8_t{0b00001000});
        SetVRegNarrow(args.rd, x, args.q);
        return;
      }
      if (args.size != 0b00 && args.size != 0b01) {  // size=11 reserved
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xz = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kUqxtn) {
        FpRegister xm = AllocTempSimdReg();
        Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(
            args.size == 0b00 ? int64_t{0x00FF00FF00FF00FFLL}
                              : int64_t{0x0000FFFF0000FFFFLL}));
        builder_.Gen<x86_64::MovqXRegReg>(xm.machine_reg(), mlo);
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
        } else {
          builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
        }
      } else if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun) {
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
        } else {
          builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
        }
      } else {  // kSqxtn
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PacksswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
        } else {
          builder_.Gen<x86_64::PackssdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
        }
      }
      SetVRegNarrow(args.rd, xn, args.q);
      return;
    }

    // SUQADD / USQADD Vd.<T>, Vn.<T> — per-lane saturating accumulate of
    // mixed signedness:
    //   SUQADD Vd, Vn: Vd[i] = SignedSat( int(Vd[i]) + uint(Vn[i]) )
    //   USQADD Vd, Vn: Vd[i] = UnsignedSat( uint(Vd[i]) + int(Vn[i]) )
    // x86 has no mixed-sign saturating add, but an N-bit signed plus an N-bit
    // unsigned always fits in N+1 bits, so widen each operand with its own
    // signedness, add in the wider lane, and narrow with the pack matching the
    // destination signedness (PACKUS* for USQADD, PACKSS* for SUQADD — PACKUS
    // clamps negatives to 0 and over-range to max = UnsignedSat exactly).
    // Mirrors the branchless lite kSuqadd/kUsqadd byte/halfword path. size=10
    // (.2S/.4S) widens the 32-bit lanes to 64-bit (PMOVSXDQ/PMOVZXDQ), adds
    // in PADDQ, and clamps each 64-bit sum to the int32/uint32 range with
    // PCMPGTQ blends (the same masked-blend idiom as the kSqxtn size=10 arm),
    // then gathers the low dwords (PSHUFD/PUNPCKLQDQ). size=11 (.1D/.2D) needs
    // 65-bit saturation and bails to lite.
    case Decoder::AdvSimdTwoRegMiscOpcode::kSuqadd:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUsqadd: {
      const bool usqadd = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUsqadd);
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      if (args.size == 0b00 || args.size == 0b01) {
        FpRegister xd = AllocTempSimdReg();
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        // Widen the low 8 bytes of `r` in place with the given signedness.
        auto widen_lo = [&](FpRegister r, bool is_signed) {
          if (args.size == 0b00) {
            if (is_signed) {
              builder_.Gen<x86_64::PmovsxbwXRegXReg>(r.machine_reg(), r.machine_reg());
            } else {
              builder_.Gen<x86_64::PmovzxbwXRegXReg>(r.machine_reg(), r.machine_reg());
            }
          } else {
            if (is_signed) {
              builder_.Gen<x86_64::PmovsxwdXRegXReg>(r.machine_reg(), r.machine_reg());
            } else {
              builder_.Gen<x86_64::PmovzxwdXRegXReg>(r.machine_reg(), r.machine_reg());
            }
          }
        };
        auto add_wide = [&](FpRegister a, FpRegister b) {
          if (args.size == 0b00) {
            builder_.Gen<x86_64::PaddwXRegXReg>(a.machine_reg(), b.machine_reg());
          } else {
            builder_.Gen<x86_64::PadddXRegXReg>(a.machine_reg(), b.machine_reg());
          }
        };
        // Vd carries the destination flavour (SUQADD signed / USQADD unsigned);
        // Vn carries the opposite signedness.
        FpRegister dlo = AllocTempSimdReg();
        FpRegister nlo = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(dlo.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(nlo.machine_reg(), xn.machine_reg());
        widen_lo(dlo, /*is_signed=*/!usqadd);
        widen_lo(nlo, /*is_signed=*/usqadd);
        add_wide(dlo, nlo);  // dlo = widened lane sums (low half)
        FpRegister hi = args.q ? AllocTempSimdReg() : AllocZeroedSimdReg();
        if (args.q) {
          builder_.Gen<x86_64::PsrldqXRegImm>(xd.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
          FpRegister nhi = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(hi.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(nhi.machine_reg(), xn.machine_reg());
          widen_lo(hi, /*is_signed=*/!usqadd);
          widen_lo(nhi, /*is_signed=*/usqadd);
          add_wide(hi, nhi);  // hi = widened lane sums (high half)
        }
        // Narrow with the destination-signedness saturating pack.
        if (args.size == 0b00) {
          if (usqadd) {
            builder_.Gen<x86_64::PackuswbXRegXReg>(dlo.machine_reg(), hi.machine_reg());
          } else {
            builder_.Gen<x86_64::PacksswbXRegXReg>(dlo.machine_reg(), hi.machine_reg());
          }
        } else {
          if (usqadd) {
            builder_.Gen<x86_64::PackusdwXRegXReg>(dlo.machine_reg(), hi.machine_reg());
          } else {
            builder_.Gen<x86_64::PackssdwXRegXReg>(dlo.machine_reg(), hi.machine_reg());
          }
        }
        SetVRegFull(args.rd, dlo, args.q);
        return;
      }
      if (args.size == 0b10) {
        FpRegister xd = AllocTempSimdReg();
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        const int64_t hi_bound = usqadd ? int64_t{0x00000000FFFFFFFFLL}   // UINT32_MAX
                                        : int64_t{0x000000007FFFFFFFLL};  // INT32_MAX
        const int64_t lo_bound = usqadd
                                     ? int64_t{0}
                                     : static_cast<int64_t>(0xFFFFFFFF80000000ULL);  // INT32_MIN
        auto set_const = [&](FpRegister r, int64_t v) {
          Register gp = std::get<0>(Gen<x86_64::MovqRegImm>(v));
          builder_.Gen<x86_64::MovqXRegReg>(r.machine_reg(), gp);
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(r.machine_reg(), r.machine_reg());
        };
        // dst = (mask ? val : dst) via dst ^= (dst ^ val) & mask.
        auto blend = [&](FpRegister dst, FpRegister val, FpRegister mask) {
          FpRegister t = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), dst.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), val.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(t.machine_reg(), mask.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(dst.machine_reg(), t.machine_reg());
        };
        // Widen the low 2 dwords of (d,n) to 64-bit lanes, add, clamp to the
        // destination range, and gather the two clamped low dwords into the
        // low 64 bits (PSHUFD 0b00001000). Consumes d and n; returns d.
        auto process_half = [&](FpRegister d, FpRegister n) -> FpRegister {
          if (usqadd) {
            builder_.Gen<x86_64::PmovzxdqXRegXReg>(d.machine_reg(), d.machine_reg());
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(n.machine_reg(), n.machine_reg());
          } else {
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(d.machine_reg(), d.machine_reg());
            builder_.Gen<x86_64::PmovzxdqXRegXReg>(n.machine_reg(), n.machine_reg());
          }
          builder_.Gen<x86_64::PaddqXRegXReg>(d.machine_reg(), n.machine_reg());
          FpRegister c = AllocTempSimdReg();
          FpRegister m = AllocTempSimdReg();
          // clamp above: where d > hi_bound, take hi_bound.
          set_const(c, hi_bound);
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), d.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());
          blend(d, c, m);
          // clamp below: where lo_bound > d, take lo_bound.
          set_const(c, lo_bound);
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), c.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), d.machine_reg());
          blend(d, c, m);
          builder_.Gen<x86_64::PshufdXRegXRegImm>(d.machine_reg(), d.machine_reg(),
                                                  int8_t{0b00001000});
          return d;
        };
        if (args.q) {
          FpRegister xdh = AllocTempSimdReg();
          FpRegister xnh = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(xdh.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(xnh.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsrldqXRegImm>(xdh.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsrldqXRegImm>(xnh.machine_reg(), int8_t{8});
          FpRegister lo = process_half(xd, xn);
          FpRegister hi = process_half(xdh, xnh);
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(lo.machine_reg(), hi.machine_reg());
          SetVRegFull(args.rd, lo, /*q=*/true);
        } else {
          FpRegister lo = process_half(xd, xn);
          SetVRegFull(args.rd, lo, /*q=*/false);
        }
        return;
      }
      UndefinedReturningVoid();  // size=11 (.1D/.2D): 65-bit saturation, bail to lite
      return;
    }

    // CLZ / CLS V.<T>, V.<T> — per-lane count-leading-zeros / count-leading-
    // sign-bits. size=00 .8B/.16B (N=8), size=01 .4H/.8H (N=16), size=10
    // .2S/.4S (N=32); size=11 reserved. Mirrors lite kClz/kCls but uses LZCNT
    // (branchless: LZCNT_32(0)==32) instead of the lite tier's BSR + zero
    // branch. Each lane is scalarized: PEXTR{b,w,d} zero-extends the element
    // into a GP reg, LZCNTL counts leading zeros over 32 bits, and SUBL folds
    // the width correction — CLZ_N = LZCNT_32(x) - (32-N). CLS first maps
    // negative lanes to ~x (vector PCMPGT sign mask + PXOR, exactly the lite
    // preprocess) so CLS = CLZ_N(y) - 1 = LZCNT_32(y) - (33-N); the y==0 lane
    // (all-same-bits input) then yields N-1 with no branch. Bails to lite if
    // the host lacks LZCNT.
    case Decoder::AdvSimdTwoRegMiscOpcode::kClz:
    case Decoder::AdvSimdTwoRegMiscOpcode::kCls: {
      if (args.size == 0b11 || !host_platform::kHasLZCNT) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_cls = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCls);
      const int lane_bits = 8 << args.size;             // 8, 16, 32
      const int bytes_per_lane = 1 << args.size;        // 1, 2, 4
      const int lanes_per_vec = (args.q ? 16 : 8) / bytes_per_lane;
      // CLZ_N = LZCNT_32 - (32-lane_bits); CLS subtracts one more.
      const int32_t correction =
          static_cast<int32_t>(32 - lane_bits) + (is_cls ? 1 : 0);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xd = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (is_cls) {
        // y = (x < 0) ? ~x : x per lane: xsign = (0 > xn) all-ones mask, then
        // xn ^= xsign collapses negative lanes to ~x, positive lanes unchanged.
        FpRegister xsign = AllocZeroedSimdReg();
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PcmpgtbXRegXReg>(xsign.machine_reg(), xn.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PcmpgtwXRegXReg>(xsign.machine_reg(), xn.machine_reg());
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PcmpgtdXRegXReg>(xsign.machine_reg(), xn.machine_reg());
            break;
        }
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xsign.machine_reg());
      }
      for (int i = 0; i < lanes_per_vec; ++i) {
        Register lane;
        switch (args.size) {
          case 0b00:
            lane = std::get<0>(
                Gen<x86_64::PextrbRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(i)));
            break;
          case 0b01:
            lane = std::get<0>(
                Gen<x86_64::PextrwRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(i)));
            break;
          default:  // 0b10
            lane = std::get<0>(
                Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(i)));
            break;
        }
        Register cnt = std::get<0>(Gen<x86_64::LzcntlRegReg>(lane));
        if (correction != 0) {
          cnt = std::get<0>(Gen<x86_64::SublRegImm, kNoSSA>(cnt, correction));
        }
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PinsrbXRegRegImm>(xd.machine_reg(), cnt,
                                                   static_cast<int8_t>(i));
            break;
          case 0b01:
            builder_.Gen<x86_64::PinsrwXRegRegImm>(xd.machine_reg(), cnt,
                                                   static_cast<int8_t>(i));
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PinsrdXRegRegImm>(xd.machine_reg(), cnt,
                                                   static_cast<int8_t>(i));
            break;
        }
      }
      SetVRegFull(args.rd, xd, args.q);
      return;
    }

    // ADDV Vd, Vn.<T> — sum all source lanes; the single esize-wide result
    // is written to Vd's low lane with every other byte of Vd zeroed. Mirrors
    // the validated lite lowering: PSADBW (byte), cascading PHADDW (halfword)
    // and PHADDD (word) reductions, then a PSLLDQ/PSRLDQ shuttle keeps only
    // the low result lane. size=10 Q=0 (.2S) is reserved; size=11 bails.
    case Decoder::AdvSimdTwoRegMiscOpcode::kAddv: {
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (args.size) {
        case 0b00: {
          FpRegister xz = AllocZeroedSimdReg();
          // xn := [sum(bytes 0..7), 0..., sum(bytes 8..15), 0...] (16-bit
          // sums in qword-lane positions).
          builder_.Gen<x86_64::PsadbwXRegXReg>(xn.machine_reg(), xz.machine_reg());
          if (args.q) {
            // .16B: fold the high-qword sum into the low qword.
            FpRegister xt = AllocTempSimdReg();
            builder_.Gen<x86_64::PshufdXRegXRegImm>(xt.machine_reg(),
                                                    xn.machine_reg(),
                                                    static_cast<int8_t>(0xEE));
            builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xt.machine_reg());
          }
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{15});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{15});
          break;
        }
        case 0b01: {
          // .4H: 2 cascading PHADDW; .8H: 3.
          builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          if (args.q) {
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{14});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{14});
          break;
        }
        case 0b10: {
          if (!args.q) {  // .2S reserved for ADDV.
            UndefinedReturningVoid();
            return;
          }
          // .4S: 2 cascading PHADDD.
          builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
          break;
        }
        default:
          UndefinedReturningVoid();
          return;
      }
      builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
      return;
    }

    // SADDLV / UADDLV Vd, Vn.<T> — across-lanes long sum. Each source element
    // is widened to 2*esize before summing; the single 2*esize-wide result is
    // written to Vd's low lane with all other bytes zeroed. Mirrors the
    // validated lite lowering. UADDLV at .8B/.16B reuses the PSADBW byte-sum;
    // SADDLV and all halfword/word inputs widen first via PMOVSX/PMOVZX, then
    // reduce with PHADDW/PHADDD (or PADDQ folds for the .4S->D case).
    case Decoder::AdvSimdTwoRegMiscOpcode::kSaddlv:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUaddlv: {
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSaddlv);
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (args.size) {
        case 0b00: {
          // Bytes -> 16-bit sum.
          if (!is_signed) {
            FpRegister xz = AllocZeroedSimdReg();
            builder_.Gen<x86_64::PsadbwXRegXReg>(xn.machine_reg(), xz.machine_reg());
            if (args.q) {
              FpRegister xt = AllocTempSimdReg();
              builder_.Gen<x86_64::PshufdXRegXRegImm>(xt.machine_reg(),
                                                      xn.machine_reg(),
                                                      static_cast<int8_t>(0xEE));
              builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xt.machine_reg());
            }
          } else if (args.q) {
            // SADDLV .16B: widen low/high 8 bytes separately, lane-add, then
            // 3 PHADDW collapses.
            FpRegister xt = AllocTempSimdReg();
            builder_.Gen<x86_64::PmovsxbwXRegXReg>(xt.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xt.machine_reg());
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          } else {
            // SADDLV .8B: widen low 8 bytes; 3 PHADDW collapses.
            builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{14});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{14});
          break;
        }
        case 0b01: {
          // Halfwords -> 32-bit sum.
          if (args.q) {
            FpRegister xt = AllocTempSimdReg();
            if (is_signed) {
              builder_.Gen<x86_64::PmovsxwdXRegXReg>(xt.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            } else {
              builder_.Gen<x86_64::PmovzxwdXRegXReg>(xt.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            }
            builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xt.machine_reg());
            builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
          } else {
            if (is_signed) {
              builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            } else {
              builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            }
            builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
          break;
        }
        case 0b10: {
          if (!args.q) {  // .2S reserved.
            UndefinedReturningVoid();
            return;
          }
          // .4S -> 64-bit sum: widen 4 dwords -> 4 qwords across two regs,
          // lane-add, fold hi-qword to lo-qword.
          FpRegister xt = AllocTempSimdReg();
          if (is_signed) {
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(xt.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          } else {
            builder_.Gen<x86_64::PmovzxdqXRegXReg>(xt.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xt.machine_reg());
          builder_.Gen<x86_64::PshufdXRegXRegImm>(xt.machine_reg(), xn.machine_reg(),
                                                  static_cast<int8_t>(0xEE));
          builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xt.machine_reg());
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
          break;
        }
        default:
          UndefinedReturningVoid();
          return;
      }
      builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
      return;
    }

    // SMAXV / SMINV / UMAXV / UMINV Vd, Vn.<T> — across-lanes integer
    // max/min reduce. Scan all source lanes; write the single scalar
    // max/min to Vd's low lane with all other bytes zeroed. The result
    // width equals esize (unlike SADDLV/UADDLV which widens to 2*esize).
    // Mirrors the validated lite lowering: for Q=0 the low qword is
    // replicated across both halves (PSHUFD 0x44) so the don't-care upper
    // half can't poison the reduction (max(x,x)=x is idempotent), then a
    // cascading PMAX/PMIN against a shuffle-of-xn halves the surviving lane
    // count each step; a final PSLLDQ/PSRLDQ shuttle keeps only the low lane.
    case Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv:
    case Decoder::AdvSimdTwoRegMiscOpcode::kSminv:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUmaxv:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUminv: {
      const bool is_max =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv) ||
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUmaxv);
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv) ||
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSminv);
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xt = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (!args.q) {
        // Replicate low qword to high qword: { dw0, dw1, dw0, dw1 } via
        // _MM_SHUFFLE(1,0,1,0) = 0x44. Neutralizes the don't-care upper half.
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            xn.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x44));
      }
      auto EmitPmaxPmin = [&](unsigned width_bits) {
        switch (width_bits) {
          case 8:
            if (is_signed) {
              if (is_max)
                builder_.Gen<x86_64::PmaxsbXRegXReg>(xn.machine_reg(), xt.machine_reg());
              else
                builder_.Gen<x86_64::PminsbXRegXReg>(xn.machine_reg(), xt.machine_reg());
            } else {
              if (is_max)
                builder_.Gen<x86_64::PmaxubXRegXReg>(xn.machine_reg(), xt.machine_reg());
              else
                builder_.Gen<x86_64::PminubXRegXReg>(xn.machine_reg(), xt.machine_reg());
            }
            break;
          case 16:
            if (is_signed) {
              if (is_max)
                builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xt.machine_reg());
              else
                builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xt.machine_reg());
            } else {
              if (is_max)
                builder_.Gen<x86_64::PmaxuwXRegXReg>(xn.machine_reg(), xt.machine_reg());
              else
                builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xt.machine_reg());
            }
            break;
          case 32:
            if (is_signed) {
              if (is_max)
                builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xt.machine_reg());
              else
                builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xt.machine_reg());
            } else {
              if (is_max)
                builder_.Gen<x86_64::PmaxudXRegXReg>(xn.machine_reg(), xt.machine_reg());
              else
                builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xt.machine_reg());
            }
            break;
        }
      };
      switch (args.size) {
        case 0b00: {
          // Bytes: 16 -> 8 -> 4 -> 2 -> 1 surviving lanes per step.
          builder_.Gen<x86_64::PshufdXRegXRegImm>(
              xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x4E));
          EmitPmaxPmin(8);
          builder_.Gen<x86_64::PshufdXRegXRegImm>(
              xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xB1));
          EmitPmaxPmin(8);
          builder_.Gen<x86_64::MovdqaXRegXReg>(xt.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsrldqXRegImm>(xt.machine_reg(), int8_t{2});
          EmitPmaxPmin(8);
          builder_.Gen<x86_64::MovdqaXRegXReg>(xt.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsrldqXRegImm>(xt.machine_reg(), int8_t{1});
          EmitPmaxPmin(8);
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{15});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{15});
          break;
        }
        case 0b01: {
          // Halfwords: 8 -> 4 -> 2 -> 1 surviving lanes per step.
          builder_.Gen<x86_64::PshufdXRegXRegImm>(
              xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x4E));
          EmitPmaxPmin(16);
          builder_.Gen<x86_64::PshufdXRegXRegImm>(
              xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xB1));
          EmitPmaxPmin(16);
          builder_.Gen<x86_64::MovdqaXRegXReg>(xt.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsrldqXRegImm>(xt.machine_reg(), int8_t{2});
          EmitPmaxPmin(16);
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{14});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{14});
          break;
        }
        case 0b10: {
          if (!args.q) {  // .2S reserved.
            UndefinedReturningVoid();
            return;
          }
          // Dwords: 4 -> 2 -> 1 surviving lanes per step.
          builder_.Gen<x86_64::PshufdXRegXRegImm>(
              xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x4E));
          EmitPmaxPmin(32);
          builder_.Gen<x86_64::PshufdXRegXRegImm>(
              xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xB1));
          EmitPmaxPmin(32);
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
          break;
        }
        default:
          UndefinedReturningVoid();
          return;
      }
      builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
      return;
    }

    // FMAXV / FMINV / FMAXNMV / FMINNMV Sd, Vn.4S — floating-point across-lanes
    // reduction. Reduce the 4 FP32 lanes to a single scalar written to Sd's low
    // lane with the upper 96 bits zeroed. Two pairwise steps (PSHUFD 0x4E then
    // 0xB1) each fold lanes together via the same NaN-handling min/max idioms the
    // vector three-same FMAX/FMIN path uses (lines ~1828):
    //   FMAXV/FMINV       — NaN-propagating: tmp=b; MAX/MIN tmp,a; MAX/MIN a,b;
    //                        POR a,tmp. (x86 MAX/MINPS returns src on NaN, so the
    //                        two-sided op + POR keeps a NaN exponent if either
    //                        input was NaN and makes +-0 order-independent.)
    //   FMAXNMV/FMINNMV   — NaN-suppressing: substitute each NaN lane with the
    //                        other operand via a CMPUNORDPS self-compare mask,
    //                        then MAX/MIN. Mirrors the interpreter FmaxScalar /
    //                        FmaxnmScalar reduction. The Armv8.2 FP16 (.4H/.8H)
    //                        form is handled separately just below, through the
    //                        F16C round-trip.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFminv:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv: {
      // Armv8.2-FP16 across-lanes FMAXV/FMINV/FMAXNMV/FMINNMV Hd, Vn.4H (Q=0) /
      // Vn.8H (Q=1) via the F16C round-trip. Widen the halves to FP32, reduce
      // there, narrow the single surviving lane once. The round-trip is exact for
      // this family: FP16->FP32 is exact and order-preserving, min/max returns one
      // of its inputs unmodified, so the value narrowed at the end is a value that
      // arrived as an FP16 and round-trips bit-for-bit. NaN and the +-0 tie use the
      // same policy idioms as the FP16 vector pairwise path, in FP32 space, via
      // EmitFpPairwiseMinMaxF32Packed (NaN-propagating for FMAXV/FMINV, suppressing
      // for the NM forms, and sign = sign1 AND sign2 for max / OR for min when both
      // inputs are zero). Without F16C the region bails to lite->interp.
      if (args.is_fp16) {
        if (!host_platform::kHasF16C) {
          UndefinedReturningVoid();
          return;
        }
        const bool h_is_max =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv);
        const bool h_is_nm =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv);
        FpRegister lo = no_fp_register, hi = no_fp_register;
        EmitWidenHalfVec(vn_off, args.q, &lo, &hi);
        // .8H: lanes 4..7 fold elementwise into lanes 0..3 first (8 -> 4 FP32).
        // .4H: only the low group exists.
        FpRegister r =
            args.q ? EmitFpPairwiseMinMaxF32Packed(lo, hi, h_is_max, h_is_nm) : lo;
        // Two shuffle-and-combine steps leave the full reduction in lane 0:
        // PSHUFD 0x4E swaps the 64-bit halves (folds lanes 2,3 into 0,1), 0xB1
        // swaps adjacent dwords (folds lane 1 into lane 0).
        FpRegister t0 = AllocTempSimdReg();
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            t0.machine_reg(), r.machine_reg(), static_cast<int8_t>(0x4E));
        r = EmitFpPairwiseMinMaxF32Packed(r, t0, h_is_max, h_is_nm);
        FpRegister t1 = AllocTempSimdReg();
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            t1.machine_reg(), r.machine_reg(), static_cast<int8_t>(0xB1));
        r = EmitFpPairwiseMinMaxF32Packed(r, t1, h_is_max, h_is_nm);
        // Keep lane 0 and zero the other three FP32 lanes before narrowing, so the
        // scalar store writes Hd with bits[127:16] clear: VCVTPS2PH turns the three
        // +0.0 lanes into 0x0000 halves and SetVRegScalar zeroes everything above
        // the low 32 bits.
        builder_.Gen<x86_64::PslldqXRegImm>(r.machine_reg(), int8_t{12});
        builder_.Gen<x86_64::PsrldqXRegImm>(r.machine_reg(), int8_t{12});
        EmitNarrowF32ToHalfAndStore(args.rd, r);
        return;
      }
      const bool is_max =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv) ||
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv);
      const bool is_nm =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv) ||
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv);
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xt = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // Fold b (=xt) into a (=xn) with FP32 packed NaN-aware min/max.
      auto EmitPairFp = [&](FpRegister a, FpRegister b) {
        if (!is_nm) {
          FpRegister tmp = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), b.machine_reg());
          if (is_max) {
            builder_.Gen<x86_64::MaxpsXRegXReg>(tmp.machine_reg(), a.machine_reg());
            builder_.Gen<x86_64::MaxpsXRegXReg>(a.machine_reg(), b.machine_reg());
          } else {
            builder_.Gen<x86_64::MinpsXRegXReg>(tmp.machine_reg(), a.machine_reg());
            builder_.Gen<x86_64::MinpsXRegXReg>(a.machine_reg(), b.machine_reg());
          }
          builder_.Gen<x86_64::PorXRegXReg>(a.machine_reg(), tmp.machine_reg());
        } else {
          FpRegister mask_a = AllocTempSimdReg();
          FpRegister mask_b = AllocTempSimdReg();
          FpRegister an_sub = AllocTempSimdReg();
          FpRegister bn_sub = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(mask_a.machine_reg(), a.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(mask_b.machine_reg(), b.machine_reg());
          builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_a.machine_reg(), mask_a.machine_reg());
          builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_b.machine_reg(), mask_b.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(an_sub.machine_reg(), mask_a.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(an_sub.machine_reg(), b.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(bn_sub.machine_reg(), mask_b.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(bn_sub.machine_reg(), a.machine_reg());
          builder_.Gen<x86_64::PandnXRegXReg>(mask_a.machine_reg(), a.machine_reg());
          builder_.Gen<x86_64::PandnXRegXReg>(mask_b.machine_reg(), b.machine_reg());
          builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), an_sub.machine_reg());
          builder_.Gen<x86_64::PorXRegXReg>(mask_b.machine_reg(), bn_sub.machine_reg());
          if (is_max) {
            builder_.Gen<x86_64::MaxpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
          } else {
            builder_.Gen<x86_64::MinpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
          }
          builder_.Gen<x86_64::MovdqaXRegXReg>(a.machine_reg(), mask_a.machine_reg());
        }
      };
      // Step 1: fold lanes {2,3} into {0,1}. PSHUFD 0x4E swaps the 64-bit halves.
      builder_.Gen<x86_64::PshufdXRegXRegImm>(
          xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x4E));
      EmitPairFp(xn, xt);
      // Step 2: fold lane 1 into lane 0. PSHUFD 0xB1 swaps adjacent dwords.
      builder_.Gen<x86_64::PshufdXRegXRegImm>(
          xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xB1));
      EmitPairFp(xn, xt);
      // Keep only the low 32-bit result lane; zero the upper 96 bits.
      builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
      builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
      return;
    }

    // FCVTZS V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — floating-point convert to
    // signed integer, round toward zero. Branchless mirror of the validated
    // lite lowering (lite_translator.h::AdvSimdTwoRegMisc kFcvtzsV, FP32
    // path): CVTTPS2DQ does the truncating conversion, then a packed
    // saturation fix-up folds in the ARM out-of-range semantics that x86
    // CVTTPS2DQ gets wrong — NaN lanes must become 0 (x86 yields
    // 0x80000000), and positive-overflow lanes must become INT32_MAX (x86
    // yields the 0x80000000 "integer indefinite"). The FP64 .2D form (branchy
    // per-lane in lite) and the Armv8.2 FP16 form bail to lite→interp.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzsV: {
      if (args.is_fp16 || args.size != 0b10) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister x_dst = AllocTempSimdReg();
      FpRegister x_mask = AllocTempSimdReg();
      FpRegister x_eqmin = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // 1. Primary truncating conversion.
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
      // 2. NaN lanes -> 0.
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
      // 3. INT32_MIN (0x80000000) per lane.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
      // 4. result == INT32_MIN ?
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
      // 5. src non-negative ? PSRAD 31 -> 0 if non-neg, all-1s if neg.
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
      // 6. Flip INT_MIN -> INT_MAX in positive-overflow lanes.
      builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
      SetVRegFull(args.rd, x_dst, args.q);
      return;
    }

    // FCVTZU V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — floating-point convert to
    // unsigned integer, round toward zero. Branchless mirror of the lite
    // lowering (kFcvtzuV, FP32 path). x86 has no packed FP->u32, so the
    // classic "subtract 2^31" offset trick brings [2^31, 2^32) into signed
    // CVTTPS2DQ range, restores bit 31, and saturates too-big/Inf lanes to
    // UINT32_MAX; MAXPS(src, 0) collapses NaN/negative to 0. FP64 .2D / FP16
    // bail to lite.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzuV: {
      if (args.is_fp16 || args.size != 0b10) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister x_dst = AllocTempSimdReg();       // src -> result
      FpRegister x_pow31 = AllocTempSimdReg();
      FpRegister x_needs_off = AllocTempSimdReg();
      // Zeroed scratch (AllocZeroedSimdReg gives it a defining write, avoiding
      // a PXOR-self read of an undefined register). Reused after step 1.
      FpRegister x_scratch = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(x_dst.machine_reg(), vn_off);
      // 1. Clamp neg/NaN to 0 via MAXPS with the zeroed scratch.
      builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      // 2. Broadcast 2^31 = 0x4F000000 to all four FP32 lanes.
      Register gp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
      builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(
          x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
      // 3. needs_offset = (2^31 <= src_clamped).
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
      // 4. offset_amount = 2^31 where needs_offset.
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
      // 5. src_for_cvt = src_clamped - offset_amount.
      builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      // 6. too_big = (2^31 <= src_for_cvt).
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
      // 7. CVTTPS2DQ.
      builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
      // 8. Build 0x80000000 per lane; AND needs_offset; OR into result.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
      // 9. Saturate too-big lanes to 0xFFFFFFFF.
      builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      SetVRegFull(args.rd, x_dst, args.q);
      return;
    }

    // FCVTNS / FCVTPS / FCVTMS V Vd.<T>, Vn.<T> (FP32 .2S/.4S) —
    // floating-point convert to signed integer with an explicit rounding
    // mode (round-to-nearest-ties-even / toward +inf / toward -inf).
    // Branchless mirror of the validated lite lowering
    // (lite_translator.h::AdvSimdTwoRegMisc kFcvtnsV/kFcvtpsV/kFcvtmsV, FP32
    // path): ROUNDPS with the matching imm makes each finite lane an exact
    // integer-valued FP (NaN/±Inf/sign-of-zero pass through unchanged), then
    // the same CVTTPS2DQ + packed saturation fix-up used by FCVTZS V folds in
    // the ARM out-of-range semantics x86 gets wrong. `(args.size & 1) == 1`
    // is the FP64 .2D form (branchy per-lane in lite) and FP16 both bail to
    // lite→interp — mirroring the lite FP32-only fast path.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnsV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpsV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmsV: {
      if (args.is_fp16 || (args.size & 1) == 1) {
        UndefinedReturningVoid();
        return;
      }
      int8_t round_imm;
      if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnsV) {
        round_imm = int8_t{0x08};   // RNE + suppress-inexact
      } else if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpsV) {
        round_imm = int8_t{0x0A};   // toward +inf
      } else {
        round_imm = int8_t{0x09};   // toward -inf
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister x_dst = AllocTempSimdReg();
      FpRegister x_mask = AllocTempSimdReg();
      FpRegister x_eqmin = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), round_imm);
      // FCVTZS V saturation fix-up (Roundps preserves NaN/±Inf/sign-of-zero).
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
      SetVRegFull(args.rd, x_dst, args.q);
      return;
    }

    // FCVTNU / FCVTPU / FCVTMU V Vd.<T>, Vn.<T> (FP32 .2S/.4S) —
    // floating-point convert to unsigned integer with an explicit rounding
    // mode. Branchless mirror of the lite lowering
    // (kFcvtnuV/kFcvtpuV/kFcvtmuV, FP32 path): ROUNDPS then the FCVTZU V
    // offset-by-2^31 trick (MAXPS(src,0) collapses NaN/negative to 0; the
    // [2^31,2^32) range is brought into signed CVTTPS2DQ range and bit 31 is
    // restored; too-big/Inf lanes saturate to UINT32_MAX). FP64 .2D / FP16
    // bail to lite.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmuV: {
      if (args.is_fp16 || (args.size & 1) == 1) {
        UndefinedReturningVoid();
        return;
      }
      int8_t round_imm;
      if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV) {
        round_imm = int8_t{0x08};   // RNE + suppress-inexact
      } else if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV) {
        round_imm = int8_t{0x0A};   // toward +inf
      } else {
        round_imm = int8_t{0x09};   // toward -inf
      }
      FpRegister x_dst = AllocTempSimdReg();
      FpRegister x_pow31 = AllocTempSimdReg();
      FpRegister x_needs_off = AllocTempSimdReg();
      FpRegister x_scratch = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(x_dst.machine_reg(), vn_off);
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(x_dst.machine_reg(), x_dst.machine_reg(), round_imm);
      // 1. Clamp neg/NaN to 0 via MAXPS with the zeroed scratch.
      builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      // 2. Broadcast 2^31 = 0x4F000000 to all four FP32 lanes.
      Register gp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
      builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(
          x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
      // 3. needs_offset = (2^31 <= src_clamped).
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
      // 4. offset_amount = 2^31 where needs_offset.
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
      // 5. src_for_cvt = src_clamped - offset_amount.
      builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      // 6. too_big = (2^31 <= src_for_cvt).
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
      // 7. CVTTPS2DQ.
      builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
      // 8. Build 0x80000000 per lane; AND needs_offset; OR into result.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
      // 9. Saturate too-big lanes to 0xFFFFFFFF.
      builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      SetVRegFull(args.rd, x_dst, args.q);
      return;
    }

    // FCVTAS / FCVTAU V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — floating-point
    // convert to integer, round-to-nearest ties-away-from-zero. x86 ROUNDPS
    // has no ties-away mode, so mirror the lite FRINTA trick
    // (kFcvtasV/kFcvtauV, FP32 path): per lane addend = copysign(0.5, x),
    // gated to 0 when |x| >= 2^23 (already integer), ADDPS, ROUNDPS imm=3
    // (trunc). Then the FCVTZS V (signed) / FCVTZU V (unsigned) saturation
    // fix-up. FP64 .2D / FP16 bail to lite.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtasV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV: {
      if (args.is_fp16 || (args.size & 1) == 1) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_unsigned =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV);
      // abs_bits and copysign are first written by a self-PCMPEQD (all-ones
      // idiom), so they need a defining write first (AllocZeroedSimdReg) to
      // satisfy lifetime analysis; half is first written by MovdXRegReg.
      FpRegister xn = AllocTempSimdReg();
      FpRegister copysign = AllocZeroedSimdReg();
      FpRegister half = AllocTempSimdReg();
      FpRegister abs_bits = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // FRINTA dance (FP32 form).
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(abs_bits.machine_reg(), abs_bits.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(abs_bits.machine_reg(), int8_t{1});   // 0x7FFFFFFF
      builder_.Gen<x86_64::PandXRegXReg>(abs_bits.machine_reg(), xn.machine_reg());  // |bits|
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(copysign.machine_reg(), copysign.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(copysign.machine_reg(), int8_t{31});  // 0x80000000
      builder_.Gen<x86_64::PandXRegXReg>(copysign.machine_reg(), xn.machine_reg());  // sign bit
      Register gp_half = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x3F000000)));  // 0.5
      builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), gp_half);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(half.machine_reg(), half.machine_reg(), int8_t{0x00});
      builder_.Gen<x86_64::PorXRegXReg>(copysign.machine_reg(), half.machine_reg());  // sign|0.5
      Register gp_pow = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4B000000)));  // 2^23
      builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), gp_pow);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(half.machine_reg(), half.machine_reg(), int8_t{0x00});
      builder_.Gen<x86_64::PcmpgtdXRegXReg>(half.machine_reg(), abs_bits.machine_reg());  // 1s where |x|<2^23
      builder_.Gen<x86_64::PandXRegXReg>(copysign.machine_reg(), half.machine_reg());  // zero addend if int
      builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), copysign.machine_reg());
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), int8_t{0x03});  // trunc
      if (!is_unsigned) {
        // FCVTZS V .2S/.4S saturation fix-up.
        FpRegister x_dst = AllocTempSimdReg();
        FpRegister x_mask = AllocTempSimdReg();
        FpRegister x_eqmin = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
        SetVRegFull(args.rd, x_dst, args.q);
      } else {
        // FCVTZU V .2S/.4S saturation fix-up (offset-by-2^31 trick).
        FpRegister x_dst = AllocTempSimdReg();
        FpRegister x_pow31 = AllocTempSimdReg();
        FpRegister x_needs_off = AllocTempSimdReg();
        FpRegister x_scratch = AllocZeroedSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        Register gp2 = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
        builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp2);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
        builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        SetVRegFull(args.rd, x_dst, args.q);
      }
      return;
    }

    // SCVTF / UCVTF V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — integer to
    // floating-point convert (the inverse direction of FCVTZS/FCVTZU V).
    // Branchless mirror of the validated lite lowering
    // (lite_translator.h::AdvSimdTwoRegMisc kScvtfV/kUcvtfV, FP32 path):
    //   SCVTF: a single CVTDQ2PS — x86 signed int32 -> FP32 matches ARM.
    //   UCVTF: CVTDQ2PS treats the input as signed, so lanes with bit31 set
    //          come out negative; recover the unsigned value by adding 2^32
    //          (FP32 bits 0x4F800000) per lane wherever bit31 was set
    //          (PSRAD 31 mask), which is exact for the [2^31, 2^32) range.
    // The FP64 .2D form (branchy per-lane in lite) and FP16 bail to
    // lite→interp, mirroring the lite FP32-only fast path.
    case Decoder::AdvSimdTwoRegMiscOpcode::kScvtfV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV: {
      if (args.is_fp16 || (args.size & 1) == 1) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_unsigned =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (!is_unsigned) {
        // SCVTF V: signed int32 -> FP32 is native.
        builder_.Gen<x86_64::Cvtdq2psXRegXReg>(xn.machine_reg(), xn.machine_reg());
        SetVRegFull(args.rd, xn, args.q);
        return;
      }
      // UCVTF V: CVTDQ2PS + per-lane 2^32 addend for MSB-set lanes.
      FpRegister msb = AllocTempSimdReg();
      FpRegister addend = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(msb.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(msb.machine_reg(), int8_t{31});  // 0 or all-1s
      builder_.Gen<x86_64::Cvtdq2psXRegXReg>(xn.machine_reg(), xn.machine_reg());  // signed convert
      Register gp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F800000)));  // 2^32
      builder_.Gen<x86_64::MovdXRegReg>(addend.machine_reg(), gp);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(addend.machine_reg(), addend.machine_reg(), int8_t{0x00});
      builder_.Gen<x86_64::PandXRegXReg>(addend.machine_reg(), msb.machine_reg());  // 2^32 where MSB set
      builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), addend.machine_reg());
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // FRINTN / FRINTM / FRINTP / FRINTZ / FRINTX / FRINTI V Vd.<T>, Vn.<T>
    // (FP32 .2S/.4S) — round floating-point to integral floating-point with a
    // fixed rounding mode. The result stays in FP, so unlike the FCVT*
    // converts there is no int saturation fix-up: a single SSE4.1 ROUNDPS
    // with the matching imm handles every finite/NaN/Inf/signed-zero lane.
    // Branchless mirror of the validated lite lowering
    // (lite_translator.h::AdvSimdTwoRegMisc kFrintnV.. path):
    //   FRINTN -> imm=0x00 (nearest-even), FRINTM -> 0x01 (toward -inf),
    //   FRINTP -> 0x02 (toward +inf), FRINTZ -> 0x03 (toward zero),
    //   FRINTX/FRINTI -> 0x04 (use MXCSR; default RNE matches ARM's FPCR).
    // FP64 .2D needs ROUNDPD (not allowlisted in the heavy backend) and FP16
    // needs the F16C round-trip, so both bail to lite→interp, mirroring the
    // lite FP32-only fast path used by the sibling FCVT* V converts.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrintxV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrintiV: {
      int8_t round_imm;
      switch (args.opcode) {
        case Decoder::AdvSimdTwoRegMiscOpcode::kFrintnV: round_imm = int8_t{0x00}; break;
        case Decoder::AdvSimdTwoRegMiscOpcode::kFrintmV: round_imm = int8_t{0x01}; break;
        case Decoder::AdvSimdTwoRegMiscOpcode::kFrintpV: round_imm = int8_t{0x02}; break;
        case Decoder::AdvSimdTwoRegMiscOpcode::kFrintzV: round_imm = int8_t{0x03}; break;
        // FRINTX / FRINTI follow the current FPCR rounding mode; treat MXCSR
        // (default RNE) as the canonical mode via the ROUND* "use MXCSR" bit.
        default: round_imm = int8_t{0x04}; break;
      }
      if (args.is_fp16) {
        // FP16 .4H/.8H FRINT* via F16C round-trip + ROUNDPS imm. Exact for FP16: an
        // exact-FP16 input widened to FP32 is unchanged; ROUNDPS to integral gives an
        // integer that is representable back in FP16 (values >=2048 are already integral,
        // integers <2048 are representable), so the narrow is lossless. Mirrors lite.
        if (!host_platform::kHasF16C) { UndefinedReturningVoid(); return; }
        FpRegister lo = no_fp_register, hi = no_fp_register;
        EmitWidenHalfVec(vn_off, args.q, &lo, &hi);
        builder_.Gen<x86_64::RoundpsXRegXRegImm>(lo.machine_reg(), lo.machine_reg(), round_imm);
        if (args.q)
          builder_.Gen<x86_64::RoundpsXRegXRegImm>(hi.machine_reg(), hi.machine_reg(), round_imm);
        EmitNarrowHalfVecAndStore(args.rd, lo, hi, args.q);
        return;
      }
      if ((args.size & 1) == 1) {  // .2D needs ROUNDPD (not allowlisted) -> bail
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), round_imm);
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // FRINTA V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — round float to integral float,
    // ties away from zero. x86 has no ROUNDPS immediate for ties-away, so
    // mirror the validated lite vector lowering
    // (lite_translator.h::AdvSimdTwoRegMisc kFrintaV FP32 path): per lane add
    // copysign(0.5, x) then ROUNDPS toward zero (imm 0x03), gating the addend
    // to 0 where |x| >= 2^23 (already an integer, where a 0.5 addend would
    // round the wrong way; NaN/±Inf bits also exceed the threshold, so their
    // addend is gated off and ROUND-toward-zero preserves them per the Intel
    // SDM). The magnitude compare is a signed PCMPGTD on |bits(x)| (top bit
    // cleared, so signed == unsigned). Unlike the scalar FRINTA the 0.5 and
    // threshold constants must be broadcast across all 4 lanes (PSHUFD imm
    // 0x00). sign/absx use AllocZeroedSimdReg (self-PCMPEQ targets). FP64 .2D
    // needs PCMPGTQ + ROUNDPD (ROUNDPD not allowlisted in the heavy backend)
    // and FP16 needs the F16C round-trip, so both bail to lite→interp,
    // matching the sibling FRINT V and FCVTA V handlers.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrintaV: {
      if (args.is_fp16) {
        // FP16 .4H/.8H FRINTA (ties away) via F16C round-trip: per lane add
        // copysign(0.5f, x) then ROUNDPS toward zero (imm 0x03). Unlike the FP32 path,
        // no |x|>=2^23 magnitude gate is needed: every FP16 magnitude is exactly
        // representable in FP32 and either <2^11 (0.5 nudge exact) or already integral
        // (nudge is a no-op, 0.5 < ulp). NaN/Inf survive add+trunc. Mirrors lite's FRINTA.
        if (!host_platform::kHasF16C) { UndefinedReturningVoid(); return; }
        FpRegister lo = no_fp_register, hi = no_fp_register;
        EmitWidenHalfVec(vn_off, args.q, &lo, &hi);
        FpRegister smask = AllocOnesSimdReg();
        builder_.Gen<x86_64::PslldXRegImm>(smask.machine_reg(), int8_t{31});  // 0x80000000/dword
        FpRegister half = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), GetImm(uint64_t{0x3F000000}));  // 0.5f
        builder_.Gen<x86_64::PshufdXRegXRegImm>(half.machine_reg(), half.machine_reg(), int8_t{0x00});
        auto rinta = [&](FpRegister x) {
          FpRegister t = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(t.machine_reg(), smask.machine_reg());  // sign(x)
          builder_.Gen<x86_64::PorXRegXReg>(t.machine_reg(), half.machine_reg());    // copysign(0.5,x)
          builder_.Gen<x86_64::AddpsXRegXReg>(x.machine_reg(), t.machine_reg());
          builder_.Gen<x86_64::RoundpsXRegXRegImm>(x.machine_reg(), x.machine_reg(), int8_t{0x03});
        };
        rinta(lo);
        if (args.q) rinta(hi);
        EmitNarrowHalfVecAndStore(args.rd, lo, hi, args.q);
        return;
      }
      if ((args.size & 1) == 1) {  // .2D needs PCMPGTQ+ROUNDPD -> bail
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister sign = AllocZeroedSimdReg();
      FpRegister absx = AllocZeroedSimdReg();
      FpRegister half = AllocTempSimdReg();
      FpRegister thresh = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // addend = copysign(0.5f, x) = (x & 0x80000000) | 0.5f, per lane.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(sign.machine_reg(), int8_t{31});  // 0x80000000
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), GetImm(uint64_t{0x3F000000}));  // 0.5f
      builder_.Gen<x86_64::PshufdXRegXRegImm>(half.machine_reg(), half.machine_reg(), int8_t{0x00});
      builder_.Gen<x86_64::PorXRegXReg>(sign.machine_reg(), half.machine_reg());
      // gate = (|x| < 2^23) ? all-ones : 0, via thresh > |x| (signed PCMPGTD).
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(absx.machine_reg(), absx.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(absx.machine_reg(), int8_t{1});  // 0x7FFFFFFF
      builder_.Gen<x86_64::PandXRegXReg>(absx.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::MovdXRegReg>(thresh.machine_reg(), GetImm(uint64_t{0x4B000000}));  // 2^23
      builder_.Gen<x86_64::PshufdXRegXRegImm>(thresh.machine_reg(), thresh.machine_reg(), int8_t{0x00});
      builder_.Gen<x86_64::PcmpgtdXRegXReg>(thresh.machine_reg(), absx.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), thresh.machine_reg());
      builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), sign.machine_reg());
      builder_.Gen<x86_64::RoundpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), int8_t{0x03});
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // SADDLP / UADDLP / SADALP / UADALP Vd.<Ta>, Vn.<Tb> — pairwise long
    // add / add-accumulate. Each adjacent pair of esize-wide source lanes is
    // widened (sign/zero) to 2*esize and summed; SADALP/UADALP accumulate the
    // pairwise sums into the existing Vd. Mirrors the validated lite lowering
    // (lite_translator.h::AdvSimdTwoRegMisc kSaddlp path) line-by-line:
    //   byte->half   signed:   PSLLW/PSRAW extract+sign-extend the low byte,
    //                          PSRAW the high byte, PADDW.
    //   byte->half   unsigned: 0x00FF mask (PCMPEQW+PSRLW 8) low byte, PSRLW 8
    //                          high byte, PADDW.
    //   half->word   signed:   PMADDWD against a per-half 0x0001 (exact signed
    //                          pair sum -> 32 bits per dword).
    //   half->word   unsigned: 0x0000FFFF mask (PCMPEQD+PSRLD 16) low half,
    //                          PSRLD 16 high half, PADDD.
    //   word->dword  signed:   PMOVSXDQ low/high dword pairs, PUNPCKL/HQDQ to
    //                          re-pair, PADDQ.
    //   word->dword  unsigned: 0xFFFFFFFF-per-qword mask (PCMPEQD+PSRLQ 32)
    //                          low dword, PSRLQ 32 high dword, PADDQ.
    // Accumulate forms PADD into Vd; Q=0 zeroes the upper 64 bits (mask_low64).
    case Decoder::AdvSimdTwoRegMiscOpcode::kSaddlp:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUaddlp:
    case Decoder::AdvSimdTwoRegMiscOpcode::kSadalp:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUadalp: {
      if (args.size == 0b11) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSaddlp) ||
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSadalp);
      const bool is_accum =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSadalp) ||
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUadalp);
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xres = AllocTempSimdReg();
      // xtmp seeds the all-ones masks via a self-compare, so it needs a def
      // (AllocZeroedSimdReg) before the PCMPEQ; the Movdqa branches overwrite
      // it harmlessly.
      FpRegister xtmp = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (args.size) {
        case 0b00: {  // byte -> half
          if (is_signed) {
            builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsllwXRegImm>(xres.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PsrawXRegImm>(xres.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrawXRegImm>(xtmp.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PaddwXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
          } else {
            builder_.Gen<x86_64::PcmpeqwXRegXReg>(xtmp.machine_reg(), xtmp.machine_reg());
            builder_.Gen<x86_64::PsrlwXRegImm>(xtmp.machine_reg(), int8_t{8});  // 0x00FF
            builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrlwXRegImm>(xtmp.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PaddwXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
          }
          break;
        }
        case 0b01: {  // half -> word
          if (is_signed) {
            builder_.Gen<x86_64::PcmpeqwXRegXReg>(xtmp.machine_reg(), xtmp.machine_reg());
            builder_.Gen<x86_64::PsrlwXRegImm>(xtmp.machine_reg(), int8_t{15});  // 0x0001
            builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PmaddwdXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
          } else {
            builder_.Gen<x86_64::PcmpeqdXRegXReg>(xtmp.machine_reg(), xtmp.machine_reg());
            builder_.Gen<x86_64::PsrldXRegImm>(xtmp.machine_reg(), int8_t{16});  // 0x0000FFFF
            builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrldXRegImm>(xtmp.machine_reg(), int8_t{16});
            builder_.Gen<x86_64::PadddXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
          }
          break;
        }
        case 0b10: {  // word -> dword
          if (is_signed) {
            FpRegister xhi = AllocTempSimdReg();
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(xres.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::MovdqaXRegXReg>(xhi.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrldqXRegImm>(xhi.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(xhi.machine_reg(), xhi.machine_reg());
            builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xres.machine_reg());
            builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xtmp.machine_reg(), xhi.machine_reg());
            builder_.Gen<x86_64::PunpckhqdqXRegXReg>(xres.machine_reg(), xhi.machine_reg());
            builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
          } else {
            builder_.Gen<x86_64::PcmpeqdXRegXReg>(xtmp.machine_reg(), xtmp.machine_reg());
            builder_.Gen<x86_64::PsrlqXRegImm>(xtmp.machine_reg(), int8_t{32});  // lo dword mask
            builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrlqXRegImm>(xtmp.machine_reg(), int8_t{32});
            builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
          }
          break;
        }
        default:
          UndefinedReturningVoid();
          return;
      }
      if (is_accum) {
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PaddwXRegXReg>(xres.machine_reg(), xd.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PadddXRegXReg>(xres.machine_reg(), xd.machine_reg());
            break;
          default:
            builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xd.machine_reg());
            break;
        }
      }
      if (!args.q) {
        // mask_low64: zero the upper 64 bits (D-register semantics).
        builder_.Gen<x86_64::PslldqXRegImm>(xres.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrldqXRegImm>(xres.machine_reg(), int8_t{8});
      }
      builder_.GenSetSimd<16>(vd_off, xres.machine_reg());
      return;
    }

    // Vector FABS / FNEG (FP32 .2S/.4S size=10, FP64 .2D size=11). Pure bit
    // ops: FABS clears the sign bit (PAND 0x7FFF..), FNEG flips it (PXOR
    // 0x8000..). Broadcast mask via the ones>>1 (FABS) / ones<<{31,63} (FNEG)
    // idiom. FP16 and the reserved .1D bail to lite. Mirrors lite's kFabs/kFneg.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFabs:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFneg: {
      const bool is_fabs =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFabs);
      if (args.is_fp16) {
        // FP16 .4H/.8H FABS/FNEG: pure per-16-bit-lane bit op, no F16C. FABS clears
        // bit15 (AND 0x7FFF), FNEG flips it (XOR 0x8000). Broadcast the 16-bit mask via
        // the ones>>1 (FABS) / ones<<15 (FNEG) idiom. Mirrors lite's FP16 FABS/FNEG.
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        FpRegister mask = AllocOnesSimdReg();
        if (is_fabs) {
          builder_.Gen<x86_64::PsrlwXRegImm>(mask.machine_reg(), int8_t{1});   // 0x7FFF/lane
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), mask.machine_reg());
        } else {
          builder_.Gen<x86_64::PsllwXRegImm>(mask.machine_reg(), int8_t{15});  // 0x8000/lane
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), mask.machine_reg());
        }
        SetVRegFull(args.rd, xn, args.q);
        return;
      }
      if (args.size != 0b10 && args.size != 0b11) { UndefinedReturningVoid(); return; }
      const bool is_double = (args.size & 1);
      if (is_double && !args.q) { UndefinedReturningVoid(); return; }
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      FpRegister mask = AllocOnesSimdReg();
      if (is_fabs) {
        if (is_double) {
          builder_.Gen<x86_64::PsrlqXRegImm>(mask.machine_reg(), int8_t{1});
        } else {
          builder_.Gen<x86_64::PsrldXRegImm>(mask.machine_reg(), int8_t{1});
        }
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), mask.machine_reg());
      } else {
        if (is_double) {
          builder_.Gen<x86_64::PsllqXRegImm>(mask.machine_reg(), int8_t{63});
        } else {
          builder_.Gen<x86_64::PslldXRegImm>(mask.machine_reg(), int8_t{31});
        }
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), mask.machine_reg());
      }
      // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // Vector FSQRT (FP32 .2S/.4S size=10, FP64 .2D size=11) via SQRTPS/SQRTPD:
    // one packed op, exact IEEE round-to-nearest-even, matching the scalar
    // FSQRT rationale (SDM == ARM under default FPCR, incl. NaN/-Inf). FP16 and
    // the reserved .1D bail. Mirrors lite's kFsqrtV non-FP16 path.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFsqrtV: {
      if (args.is_fp16) {
        // FP16 .4H/.8H FSQRT via F16C round-trip (SQRTPS is exact for one FP16 narrow).
        // Mirrors lite's FP16 FSQRT path.
        if (!host_platform::kHasF16C) { UndefinedReturningVoid(); return; }
        FpRegister lo = no_fp_register, hi = no_fp_register;
        EmitWidenHalfVec(vn_off, args.q, &lo, &hi);
        builder_.Gen<x86_64::SqrtpsXRegXReg>(lo.machine_reg(), lo.machine_reg());
        if (args.q) builder_.Gen<x86_64::SqrtpsXRegXReg>(hi.machine_reg(), hi.machine_reg());
        EmitNarrowHalfVecAndStore(args.rd, lo, hi, args.q);
        return;
      }
      if (args.size != 0b10 && args.size != 0b11) { UndefinedReturningVoid(); return; }
      const bool is_double = (args.size & 1);
      if (is_double && !args.q) { UndefinedReturningVoid(); return; }
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      FpRegister res = AllocTempSimdReg();
      if (is_double) {
        builder_.Gen<x86_64::SqrtpdXRegXReg>(res.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::SqrtpsXRegXReg>(res.machine_reg(), xn.machine_reg());
      }
      // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, res, args.q);
      return;
    }

    // FCVTL / FCVTL2 Vd.2D, Vn.2S/4S (size=01): widen 2 FP32 -> 2 FP64 via
    // CVTPS2PD. FCVTL (Q=0) takes the low 2 S lanes; FCVTL2 (Q=1) the high 2
    // (PSRLDQ 8 brings them down first). The .2D result fills all 16 bytes, so
    // it is stored full-width regardless of Q. size=00 (FP16->FP32) needs F16C
    // (absent) and bails. Mirrors lite's kFcvtl.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtl: {
      if (args.size != 0b01) { UndefinedReturningVoid(); return; }
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (args.q) {
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
      }
      FpRegister res = AllocTempSimdReg();
      builder_.Gen<x86_64::Cvtps2pdXRegXReg>(res.machine_reg(), xn.machine_reg());
      builder_.GenSetSimd<16>(GetVRegOffset(args.rd), res.machine_reg());
      return;
    }

    // FCVTN / FCVTN2 Vd.2S/4S, Vn.2D (size=01): narrow 2 FP64 -> 2 FP32 via
    // CVTPD2PS (result in low 64, upper zeroed). FCVTN (Q=0) writes Vd's low 64
    // and zeroes the upper (full-width store: CVTPD2PS already zeroed it).
    // FCVTN2 (Q=1) merges into Vd's high 64, preserving the low 64. size=00
    // (FP32->FP16) needs F16C and bails. Mirrors lite's kFcvtn.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtn: {
      if (args.size != 0b01) { UndefinedReturningVoid(); return; }
      const int32_t vd_off = GetVRegOffset(args.rd);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      FpRegister res = AllocTempSimdReg();
      builder_.Gen<x86_64::Cvtpd2psXRegXReg>(res.machine_reg(), xn.machine_reg());
      if (!args.q) {
        builder_.GenSetSimd<16>(vd_off, res.machine_reg());
      } else {
        // FCVTN2: shift the 2 floats into the high 64, keep Vd's low 64.
        builder_.Gen<x86_64::PslldqXRegImm>(res.machine_reg(), int8_t{8});
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        // Keep Vd's low 64, zero its high 64 (PSLLDQ 8 then PSRLDQ 8).
        builder_.Gen<x86_64::PslldqXRegImm>(xd.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrldqXRegImm>(xd.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PorXRegXReg>(res.machine_reg(), xd.machine_reg());
        builder_.GenSetSimd<16>(vd_off, res.machine_reg());
      }
      return;
    }

    // FCVTXN / FCVTXN2 Vd.2S/4S, Vn.2D (size=01): narrow 2 FP64 -> 2 FP32 with the
    // ARMv8 "round to odd" mode. Round-to-odd = round-toward-zero, then force the
    // result mantissa LSB to 1 whenever the conversion discarded any bits (this is the
    // double-rounding-safe rounding FCVTXN mandates). Computed with pure SSE, no MXCSR
    // (a mode flip is unsafe in a reordering backend): n = CVTPD2PS(x) (RNE), back =
    // CVTPS2PD(n) (exact). A lane is FINITE-inexact where (x==x) AND (back!=x). For
    // such a lane, step n one ULP toward zero iff RNE overshot (|back|>|x|) -- that
    // recovers the round-toward-zero neighbor -- then OR the LSB. NaN passes through
    // CVTPD2PS's qNaN untouched (bit-exact to the interpreter's
    // sign|0x7FC00000|(mant>>29)); +/-Inf, +/-0 and exact lanes are not inexact so are
    // untouched; a finite overflow makes RNE round to +/-Inf, which the overshoot step
    // decrements to +/-FP32_MAX -- exactly ARM's clamped RtO result. The two FP64 lane
    // masks are collapsed to n's two low 32-bit lanes with PSHUFD 0x08. Verified
    // bit-exact vs interpreter FpDoubleToFloatRtO over 20M fuzz cases. size!=01 is not
    // an FCVTXN encoding and bails. Q=0 stores Vd.low (upper zeroed); Q=1 (FCVTXN2)
    // stores Vd.high preserving the low 64 -- both via SetVRegNarrow.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtxn: {
      if (args.size != 0b01) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);  // x = 2 doubles

      // n = CVTPD2PS(x) (RNE);  back = CVTPS2PD(n) (exact widen).
      FpRegister n = AllocTempSimdReg();
      builder_.Gen<x86_64::Cvtpd2psXRegXReg>(n.machine_reg(), xn.machine_reg());
      FpRegister back = AllocTempSimdReg();
      builder_.Gen<x86_64::Cvtps2pdXRegXReg>(back.machine_reg(), n.machine_reg());

      // force64 = (NOT (back==x)) AND (x==x)  -> finite-AND-inexact, per 64-bit lane.
      FpRegister force = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(force.machine_reg(), back.machine_reg());
      builder_.Gen<x86_64::CmpeqpdXRegXReg>(force.machine_reg(), xn.machine_reg());  // back==x
      FpRegister ord = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(ord.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::CmpeqpdXRegXReg>(ord.machine_reg(), xn.machine_reg());  // x==x
      builder_.Gen<x86_64::PandnXRegXReg>(force.machine_reg(), ord.machine_reg());  // ~force & ord

      // over64 = |x| < |back|  (RNE rounded away from zero), per 64-bit lane.
      FpRegister absm = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(absm.machine_reg(), absm.machine_reg());
      builder_.Gen<x86_64::PsrlqXRegImm>(absm.machine_reg(), int8_t{1});  // 0x7FFF..F/qword
      FpRegister over = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(over.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(over.machine_reg(), absm.machine_reg());  // |x|
      FpRegister aback = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(aback.machine_reg(), back.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(aback.machine_reg(), absm.machine_reg());  // |back|
      builder_.Gen<x86_64::CmpltpdXRegXReg>(over.machine_reg(), aback.machine_reg());  // |x|<|back|

      // Collapse both 64-bit lane masks to 32-bit dword masks aligned with n's two low
      // floats: PSHUFD 0x08 moves src dwords {0,2} into dwords {0,1}.
      FpRegister force32 = FpRegister{
          std::get<0>(Gen<x86_64::PshufdXRegXRegImm>(force.machine_reg(), int8_t{0x08}))};
      FpRegister over32 = FpRegister{
          std::get<0>(Gen<x86_64::PshufdXRegXRegImm>(over.machine_reg(), int8_t{0x08}))};

      // Per-dword constant masks from an all-ones reg (self-Pcmpeq off a zeroed reg to
      // satisfy the use-before-def lifetime check).
      FpRegister ones = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(ones.machine_reg(), ones.machine_reg());
      FpRegister signm = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(signm.machine_reg(), ones.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(signm.machine_reg(), int8_t{31});  // 0x80000000/dword
      FpRegister magm = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(magm.machine_reg(), ones.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(magm.machine_reg(), int8_t{1});  // 0x7FFFFFFF/dword
      FpRegister onem = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(onem.machine_reg(), ones.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(onem.machine_reg(), int8_t{31});  // 0x00000001/dword

      // sign = n & 0x80000000 ; mag = n & 0x7FFFFFFF.
      FpRegister sign = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(sign.machine_reg(), n.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(sign.machine_reg(), signm.machine_reg());
      FpRegister mag = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(mag.machine_reg(), n.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(mag.machine_reg(), magm.machine_reg());

      // dec = (over & force) & 1 -> step to the round-toward-zero neighbor.
      FpRegister dec = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(dec.machine_reg(), over32.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(dec.machine_reg(), force32.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(dec.machine_reg(), onem.machine_reg());
      builder_.Gen<x86_64::PsubdXRegXReg>(mag.machine_reg(), dec.machine_reg());

      // orb = force & 1 -> round-to-odd (force LSB on inexact finite lanes).
      FpRegister orb = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(orb.machine_reg(), force32.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(orb.machine_reg(), onem.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(mag.machine_reg(), orb.machine_reg());

      // result = sign | mag (reconstructs n exactly for untouched lanes).
      builder_.Gen<x86_64::PorXRegXReg>(mag.machine_reg(), sign.machine_reg());

      SetVRegNarrow(args.rd, mag, args.q);
      return;
    }

    // FRECPE / FRSQRTE (vector, FP32 .2S/.4S size=10, FP64 .2D size=11). ARM
    // only requires ~8 bits of estimate precision; the interpreter and lite both
    // compute the FULL-precision reciprocal (1/x) / reciprocal-sqrt (1/sqrt(x)),
    // so this mirror is bit-exact vs the interpreter. FRECPE endpoints fall out
    // of SSE divide (±0->±inf, ±inf->±0, NaN->NaN). FRSQRTE needs the
    // (src<=0 || isNaN)->default-qNaN blend. FP16 and the reserved .1D bail.
    // Mirrors lite's kFrecpeV/kFrsqrteV.
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrecpeV:
    case Decoder::AdvSimdTwoRegMiscOpcode::kFrsqrteV: {
      if (args.is_fp16) { UndefinedReturningVoid(); return; }
      if (args.size != 0b10 && args.size != 0b11) { UndefinedReturningVoid(); return; }
      const bool is_double = (args.size & 1);
      if (is_double && !args.q) { UndefinedReturningVoid(); return; }
      const bool is_rsqrt =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFrsqrteV);
      FpRegister src = AllocTempSimdReg();
      builder_.GenGetSimd<16>(src.machine_reg(), vn_off);
      // recip = broadcast(1.0).
      FpRegister recip = AllocTempSimdReg();
      if (is_double) {
        builder_.Gen<x86_64::MovqXRegReg>(
            recip.machine_reg(), GetImm(uint64_t{0x3FF0000000000000}));
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            recip.machine_reg(), recip.machine_reg(), int8_t{0x44});
      } else {
        builder_.Gen<x86_64::MovdXRegReg>(
            recip.machine_reg(), GetImm(uint64_t{0x3F800000}));
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            recip.machine_reg(), recip.machine_reg(), int8_t{0x00});
      }
      if (!is_rsqrt) {
        // FRECPE: recip = 1.0 / src.
        if (is_double) {
          builder_.Gen<x86_64::DivpdXRegXReg>(recip.machine_reg(), src.machine_reg());
        } else {
          builder_.Gen<x86_64::DivpsXRegXReg>(recip.machine_reg(), src.machine_reg());
        }
        SetVRegFull(args.rd, recip, args.q);
        return;
      }
      // FRSQRTE: recip = 1.0 / sqrt(src), then blend qNaN where src<=0 || NaN.
      FpRegister sq = AllocTempSimdReg();
      if (is_double) {
        builder_.Gen<x86_64::SqrtpdXRegXReg>(sq.machine_reg(), src.machine_reg());
        builder_.Gen<x86_64::DivpdXRegXReg>(recip.machine_reg(), sq.machine_reg());
      } else {
        builder_.Gen<x86_64::SqrtpsXRegXReg>(sq.machine_reg(), src.machine_reg());
        builder_.Gen<x86_64::DivpsXRegXReg>(recip.machine_reg(), sq.machine_reg());
      }
      // mask = (src <= 0) | isNaN(src).
      FpRegister zero = AllocZeroedSimdReg();
      FpRegister mask = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(mask.machine_reg(), src.machine_reg());
      FpRegister nan_mask = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(nan_mask.machine_reg(), src.machine_reg());
      if (is_double) {
        builder_.Gen<x86_64::CmplepdXRegXReg>(mask.machine_reg(), zero.machine_reg());
        builder_.Gen<x86_64::CmpunordpdXRegXReg>(
            nan_mask.machine_reg(), nan_mask.machine_reg());
      } else {
        builder_.Gen<x86_64::CmplepsXRegXReg>(mask.machine_reg(), zero.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(
            nan_mask.machine_reg(), nan_mask.machine_reg());
      }
      builder_.Gen<x86_64::PorXRegXReg>(mask.machine_reg(), nan_mask.machine_reg());
      // qnan = broadcast(default qNaN).
      FpRegister qnan = AllocTempSimdReg();
      if (is_double) {
        builder_.Gen<x86_64::MovqXRegReg>(
            qnan.machine_reg(), GetImm(uint64_t{0x7FF8000000000000}));
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            qnan.machine_reg(), qnan.machine_reg(), int8_t{0x44});
      } else {
        builder_.Gen<x86_64::MovdXRegReg>(
            qnan.machine_reg(), GetImm(uint64_t{0x7FC00000}));
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            qnan.machine_reg(), qnan.machine_reg(), int8_t{0x00});
      }
      // result = (mask & qnan) | (~mask & recip).
      builder_.Gen<x86_64::PandXRegXReg>(qnan.machine_reg(), mask.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(mask.machine_reg(), recip.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(qnan.machine_reg(), mask.machine_reg());
      SetVRegFull(args.rd, qnan, args.q);
      return;
    }

    // SQABS / SQNEG Vd.<T>, Vn.<T> — per-lane saturating signed abs / negate,
    // size=00 .8B/.16B, 01 .4H/.8H, 10 .2S/.4S. INT_MIN_w is the only saturating
    // input (|INT_MIN| and -INT_MIN both overflow) and maps to INT_MAX_w.
    //   SQNEG: res = 0 - xn.
    //   SQABS: sign = PCMPGT(0, xn); res = (xn ^ sign) - sign.
    // Saturation as a branchless XOR: lanes where src==INT_MIN still hold
    // 0x80..0; PCMPEQ(src, INT_MIN) selects them and XOR flips 0x80..0->0x7F..F.
    // size=11 (.2D) needs PCMPGTQ/PCMPEQQ and bails to lite (mirrors lite and the
    // kAbs/kNeg boundary). Mirrors lite's kSqabs/kSqneg.
    case Decoder::AdvSimdTwoRegMiscOpcode::kSqabs:
    case Decoder::AdvSimdTwoRegMiscOpcode::kSqneg: {
      if (args.size == 0b11) { UndefinedReturningVoid(); return; }
      const bool is_neg =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSqneg);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      FpRegister xres = AllocZeroedSimdReg();
      if (is_neg) {
        // SQNEG: res = 0 - xn.
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PsubbXRegXReg>(xres.machine_reg(), xn.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PsubwXRegXReg>(xres.machine_reg(), xn.machine_reg());
            break;
          default:
            builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), xn.machine_reg());
            break;
        }
      } else {
        // SQABS: sign = PCMPGT(0, xn); res = (xn ^ sign) - sign.
        FpRegister sign = AllocZeroedSimdReg();
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PcmpgtbXRegXReg>(sign.machine_reg(), xn.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PcmpgtwXRegXReg>(sign.machine_reg(), xn.machine_reg());
            break;
          default:
            builder_.Gen<x86_64::PcmpgtdXRegXReg>(sign.machine_reg(), xn.machine_reg());
            break;
        }
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xres.machine_reg(), sign.machine_reg());
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PsubbXRegXReg>(xres.machine_reg(), sign.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PsubwXRegXReg>(xres.machine_reg(), sign.machine_reg());
            break;
          default:
            builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), sign.machine_reg());
            break;
        }
      }
      // Broadcast INT_MIN_w (repeats every 32 bits at byte/half/word granularity).
      int32_t int_min_bcast;
      switch (args.size) {
        case 0b00: int_min_bcast = static_cast<int32_t>(0x80808080u); break;
        case 0b01: int_min_bcast = static_cast<int32_t>(0x80008000u); break;
        default:   int_min_bcast = static_cast<int32_t>(0x80000000u); break;
      }
      FpRegister xmin = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdXRegReg>(
          xmin.machine_reg(),
          GetImm(static_cast<uint64_t>(static_cast<uint32_t>(int_min_bcast))));
      builder_.Gen<x86_64::PshufdXRegXRegImm>(
          xmin.machine_reg(), xmin.machine_reg(), int8_t{0x00});
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PcmpeqbXRegXReg>(xmin.machine_reg(), xn.machine_reg());
          break;
        case 0b01:
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(xmin.machine_reg(), xn.machine_reg());
          break;
        default:
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(xmin.machine_reg(), xn.machine_reg());
          break;
      }
      builder_.Gen<x86_64::PxorXRegXReg>(xres.machine_reg(), xmin.machine_reg());
      // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xres, args.q);
      return;
    }

    // URECPE / URSQRTE Vd.<T>, Vn.<T> (.2S/.4S, 32-bit lanes only) — per-lane
    // unsigned integer reciprocal / reciprocal-sqrt estimate. The result is a pure
    // function of the 9-bit field (lane>>23)&0x1FF (saturation encoded in its high
    // bits), so extract that index per lane and load the precomputed estimate from a
    // 512-entry table (bit-exact vs the interpreter). Q=0 zeroes the upper 64 bits.
    // Mirrors lite_translator_simd_two_reg_misc.inc's kUrecpe/kUrsqrte.
    case Decoder::AdvSimdTwoRegMiscOpcode::kUrecpe:
    case Decoder::AdvSimdTwoRegMiscOpcode::kUrsqrte: {
      if (args.size != 0b10) {  // 32-bit lanes only
        UndefinedReturningVoid();
        return;
      }
      const bool is_rsqrt =
          (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUrsqrte);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      FpRegister xres = AllocZeroedSimdReg();
      Register tbl = std::get<0>(Gen<x86_64::MovqRegImm>(
          reinterpret_cast<int64_t>(HeavyUnsignedEstimateTable(is_rsqrt))));
      const int lanes = args.q ? 4 : 2;
      for (int i = 0; i < lanes; ++i) {
        Register idx = std::get<0>(
            Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(i)));
        idx = std::get<0>(Gen<x86_64::ShrlRegImm>(idx, int8_t{23}));
        idx = std::get<0>(Gen<x86_64::AndlRegImm>(idx, int32_t{0x1FF}));
        Register val = std::get<0>(Gen<x86_64::MovlRegOp>(x86_64::MemoryOperand{
            .base = tbl, .index = idx, .scale = x86_64::Assembler::kTimesFour}));
        builder_.Gen<x86_64::PinsrdXRegRegImm>(
            xres.machine_reg(), val, static_cast<int8_t>(i));
      }
      // Q=0 (.2S) writes low 64, upper zeroed (xres started zeroed anyway).
      SetVRegFull(args.rd, xres, args.q);
      return;
    }

    default:
      UndefinedReturningVoid();
      return;
  }
}

// AdvSIMD scalar two-register misc.  The single-lane FP<->integer converts
// FCVTZS / FCVTZU (float->int, round toward zero) and SCVTF / UCVTF
// (int->float), all S form / FP32, are lowered here; every other scalar
// two-reg-misc opcode bails to lite via UndefinedReturningVoid.
//
// Lowering is a single-lane application of the corresponding vector .2S/.4S
// recipes in AdvSimdTwoRegMisc (kFcvtzsV / kFcvtzuV / kScvtfV / kUcvtfV).  The
// source is loaded full-width and scrubbed to lane 0 (PSLLDQ+PSRLDQ keep the
// low 32 bits and zero everything above), so lanes 1..3 become 0 and convert
// to 0.  This is required for correctness: SetVRegFull q=false writes the low
// 64 bits via MOVSD, so lane 1 (Vd[63:32]) survives — scrubbing forces it to
// 0 as the scalar S result requires (Vd[31:0]=result, Vd[127:32]=0).  For the
// int->float SCVTF/UCVTF forms the scrubbed lanes are integer 0, converting to
// +0.0f = 0x00000000, which is exactly the required zero fill.
//   * FCVTZS/FCVTZU/SCVTF/UCVTF with sz(bit0 of size)=0 (S/FP32): vector recipe.
//   * sz=1 (D / FP64) is branchy per-lane in lite -> bail.
//   * every other scalar two-reg-misc opcode -> bail.
void HeavyOptimizerFrontend::AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs& args) {
  if (!success()) {
    return;
  }
  // Scalar saturating extract-narrow SQXTN/UQXTN/SQXTUN Vd.<Tb>, Vn.<Ta> —
  // single-lane collapse of the vector kSqxtn/kUqxtn/kSqxtun recipe (see
  // AdvSimdTwoRegMisc). size=00 (H->B), 01 (S->H), 10 (D->S); size=11 is
  // decoder-rejected. Scrub the source to lane 0 (keep the low 2<<size source
  // bytes, zero the rest) so the packed vector recipe processes only the
  // scalar element while lanes 1.. stay 0 (saturate(0)=0), then commit with
  // SetVRegNarrow q=false (MOVSD low-64, upper 64 zeroed) for the scalar
  // Vd[127:esize]=0 result. Mirrors interpreter kSqxtn/kUqxtn/kSqxtun.
  if (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtn ||
      args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kUqxtn ||
      args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtun) {
    const auto opc = args.opcode;
    const int32_t vn_off_narrow =
        GetVRegOffset(args.rn);
    FpRegister x = AllocTempSimdReg();
    builder_.GenGetSimd<16>(x.machine_reg(), vn_off_narrow);
    // Scrub to lane 0: keep the low (2<<size) source bytes, zero everything
    // above (PSLLDQ N + PSRLDQ N, N = 16 - src_bytes).
    const int8_t scrub = static_cast<int8_t>(16 - (2 << args.size));  // 14,12,8
    builder_.Gen<x86_64::PslldqXRegImm>(x.machine_reg(), scrub);
    builder_.Gen<x86_64::PsrldqXRegImm>(x.machine_reg(), scrub);
    if (args.size == 0b10) {
      // 64->32: no 64->32 x86 pack, so clamp each 64-bit lane into the
      // destination range with PCMPGTQ/PCMPEQQ masked blends and gather the
      // two low dwords with PSHUFD — bit-exact mirror of the vector size=10
      // path. The scrubbed lane 1 is 0 -> clamp(0)=0 -> gathered dword=0.
      auto set_const = [&](FpRegister r, int64_t v) {
        Register gp = std::get<0>(Gen<x86_64::MovqRegImm>(v));
        builder_.Gen<x86_64::MovqXRegReg>(r.machine_reg(), gp);
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(r.machine_reg(), r.machine_reg());
      };
      // x = (x & ~mask) | (val & mask), via x ^= (x ^ val) & mask.
      auto blend = [&](FpRegister val, FpRegister mask) {
        FpRegister t = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), val.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(t.machine_reg(), mask.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(x.machine_reg(), t.machine_reg());
      };
      if (opc == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtn) {
        FpRegister c = AllocTempSimdReg();
        FpRegister m = AllocTempSimdReg();
        set_const(c, int64_t{0x000000007FFFFFFFLL});  // INT32_MAX
        builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());  // x > MAX
        blend(c, m);
        set_const(c, static_cast<int64_t>(0xFFFFFFFF80000000ULL));  // INT32_MIN
        builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), c.machine_reg());
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), x.machine_reg());  // x < MIN
        blend(c, m);
      } else if (opc == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtun) {
        FpRegister c = AllocTempSimdReg();
        FpRegister m = AllocTempSimdReg();
        set_const(c, int64_t{0x00000000FFFFFFFFLL});  // UINT32_MAX
        builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());  // x > UMAX
        blend(c, m);
        FpRegister z = AllocZeroedSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), z.machine_reg());
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), x.machine_reg());  // x < 0
        builder_.Gen<x86_64::PandnXRegXReg>(m.machine_reg(), x.machine_reg());     // neg -> 0
        builder_.Gen<x86_64::MovdqaXRegXReg>(x.machine_reg(), m.machine_reg());
      } else {  // kUqxtn: unsigned uint64 -> clamp to UINT32_MAX
        FpRegister c = AllocTempSimdReg();
        FpRegister m = AllocTempSimdReg();
        set_const(c, int64_t{0x00000000FFFFFFFFLL});  // UINT32_MAX
        builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
        builder_.Gen<x86_64::PsrlqXRegImm>(m.machine_reg(), int8_t{32});  // high 32 bits
        FpRegister z = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PcmpeqqXRegXReg>(m.machine_reg(), z.machine_reg());  // high32==0
        FpRegister t = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), c.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(m.machine_reg(), t.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(x.machine_reg(), m.machine_reg());
      }
      builder_.Gen<x86_64::PshufdXRegXRegImm>(x.machine_reg(), x.machine_reg(),
                                              int8_t{0b00001000});
      SetVRegNarrow(args.rd, x, /*q=*/false);
      return;
    }
    // 8<-16 / 16<-32: x86 narrowing packs against a zeroed high source.
    FpRegister xz = AllocZeroedSimdReg();
    if (opc == Decoder::AdvSimdScalarTwoRegMiscOpcode::kUqxtn) {
      FpRegister xm = AllocTempSimdReg();
      Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(
          args.size == 0b00 ? int64_t{0x00FF00FF00FF00FFLL}
                            : int64_t{0x0000FFFF0000FFFFLL}));
      builder_.Gen<x86_64::MovqXRegReg>(xm.machine_reg(), mlo);
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PminuwXRegXReg>(x.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PackuswbXRegXReg>(x.machine_reg(), xz.machine_reg());
      } else {
        builder_.Gen<x86_64::PminudXRegXReg>(x.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PackusdwXRegXReg>(x.machine_reg(), xz.machine_reg());
      }
    } else if (opc == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtun) {
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PackuswbXRegXReg>(x.machine_reg(), xz.machine_reg());
      } else {
        builder_.Gen<x86_64::PackusdwXRegXReg>(x.machine_reg(), xz.machine_reg());
      }
    } else {  // kSqxtn
      if (args.size == 0b00) {
        builder_.Gen<x86_64::PacksswbXRegXReg>(x.machine_reg(), xz.machine_reg());
      } else {
        builder_.Gen<x86_64::PackssdwXRegXReg>(x.machine_reg(), xz.machine_reg());
      }
    }
    SetVRegNarrow(args.rd, x, /*q=*/false);
    return;
  }
  // Scalar SQABS/SQNEG Vd, Vn — single-lane saturating signed absolute
  // value / negate. Mirrors the vector kSqabs/kSqneg SSE recipe
  // (lite_translator.h) on a lane-0-scrubbed source: scrub Vn so lanes 1..
  // become the integer 0 (SQABS(0)=SQNEG(0)=0, never the saturating input),
  // apply the packed abs/neg + INT_MIN->INT_MAX saturation, then commit with
  // SetVRegFull q=false (MOVSD low-64, upper 64 zeroed). size=00/01/10
  // (B/H/S); size=11 (D) needs PCMPGTQ/PCMPEQQ 64-bit lanes and bails to lite
  // — the same boundary as the vector .2D form. Matches interpreter
  // AdvSimdScalarTwoRegMisc kSqabs/kSqneg.
  if (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqabs ||
      args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqneg) {
    if (args.size == 0b11) {
      UndefinedReturningVoid();  // D-width: bail to lite (64-bit-lane saturation).
      return;
    }
    const bool is_neg =
        (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqneg);
    const int32_t vn_off_sq =
        GetVRegOffset(args.rn);
    FpRegister xn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off_sq);
    // Scrub to lane 0: keep the low (1<<size) bytes, zero everything above,
    // so lanes 1.. hold the integer 0.
    const int8_t scrub = static_cast<int8_t>(16 - (1 << args.size));  // 15,14,12
    builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), scrub);
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), scrub);
    FpRegister xres = AllocTempSimdReg();
    // xtmp is defined below by MovD (INT_MIN broadcast) — no self-op on a
    // fresh temp; the branches use AllocZeroedSimdReg for their zero source
    // to avoid the lifetime-analysis use-before-def on a self-PXOR.
    FpRegister xtmp = AllocTempSimdReg();
    if (is_neg) {
      // SQNEG: res = 0 - xn (packed).  xres starts at zero (copied from a
      // zeroed reg so the def is a MOVDQA, not a use-before-def self-PXOR).
      builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(),
                                           AllocZeroedSimdReg().machine_reg());
      switch (args.size) {
        case 0b00: builder_.Gen<x86_64::PsubbXRegXReg>(xres.machine_reg(), xn.machine_reg()); break;
        case 0b01: builder_.Gen<x86_64::PsubwXRegXReg>(xres.machine_reg(), xn.machine_reg()); break;
        default:   builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), xn.machine_reg()); break;
      }
    } else {
      // SQABS: sign = PCMPGT(0, xn); res = (xn ^ sign) - sign = |xn|.
      FpRegister xsign = AllocZeroedSimdReg();
      switch (args.size) {
        case 0b00: builder_.Gen<x86_64::PcmpgtbXRegXReg>(xsign.machine_reg(), xn.machine_reg()); break;
        case 0b01: builder_.Gen<x86_64::PcmpgtwXRegXReg>(xsign.machine_reg(), xn.machine_reg()); break;
        default:   builder_.Gen<x86_64::PcmpgtdXRegXReg>(xsign.machine_reg(), xn.machine_reg()); break;
      }
      builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xres.machine_reg(), xsign.machine_reg());
      switch (args.size) {
        case 0b00: builder_.Gen<x86_64::PsubbXRegXReg>(xres.machine_reg(), xsign.machine_reg()); break;
        case 0b01: builder_.Gen<x86_64::PsubwXRegXReg>(xres.machine_reg(), xsign.machine_reg()); break;
        default:   builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), xsign.machine_reg()); break;
      }
    }
    // Saturation: INT_MIN lanes still hold INT_MIN (0x80..0) after abs/neg;
    // XOR with an all-1s mask on those lanes turns INT_MIN -> INT_MAX. A
    // 32-bit broadcast pattern covers all three widths (the INT_MIN bit
    // repeats every 32 bits at byte/half/word granularity). xtmp is dead
    // here (sign mask consumed for SQABS; never set for SQNEG).
    int32_t int_min_bcast;
    switch (args.size) {
      case 0b00: int_min_bcast = static_cast<int32_t>(0x80808080u); break;
      case 0b01: int_min_bcast = static_cast<int32_t>(0x80008000u); break;
      default:   int_min_bcast = static_cast<int32_t>(0x80000000u); break;
    }
    Register gp_min = std::get<0>(Gen<x86_64::MovlRegImm>(int_min_bcast));
    builder_.Gen<x86_64::MovdXRegReg>(xtmp.machine_reg(), gp_min);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xtmp.machine_reg(), xtmp.machine_reg(), int8_t{0x00});
    switch (args.size) {
      case 0b00: builder_.Gen<x86_64::PcmpeqbXRegXReg>(xtmp.machine_reg(), xn.machine_reg()); break;
      case 0b01: builder_.Gen<x86_64::PcmpeqwXRegXReg>(xtmp.machine_reg(), xn.machine_reg()); break;
      default:   builder_.Gen<x86_64::PcmpeqdXRegXReg>(xtmp.machine_reg(), xn.machine_reg()); break;
    }
    builder_.Gen<x86_64::PxorXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
    SetVRegFull(args.rd, xres, /*q=*/false);
    return;
  }
  // FCVTAS / FCVTAU scalar (S): float -> integer, round-to-nearest ties-away.
  // x86 ROUNDPS has no ties-away mode, so mirror the vector kFcvtasV/kFcvtauV
  // FRINTA trick (see AdvSimdTwoRegMisc) on a lane-0-scrubbed source: per-lane
  // addend = copysign(0.5, x) gated to 0 when |x| >= 2^23 (already integer),
  // ADDPS, ROUNDPS imm=3 (trunc), then the FCVTZS (signed) / FCVTZU (unsigned)
  // saturation/NaN fix-up. FP32 (size=00) only; FP16 (size=01) and D bail to
  // lite (size&1 gate below). The scrubbed lanes 1.. are +0.0 -> 0, so the
  // scalar Vd[127:32]=0 result is preserved by SetVRegFull q=false.
  if (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtas ||
      args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtau) {
    if ((args.size & 1) != 0) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_unsigned =
        (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtau);
    const int32_t vn_off_a =
        GetVRegOffset(args.rn);
    FpRegister xn = AllocTempSimdReg();
    FpRegister copysign = AllocZeroedSimdReg();
    FpRegister half = AllocTempSimdReg();
    FpRegister abs_bits = AllocZeroedSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off_a);
    // Scrub to lane 0: keep the low 32 bits (the scalar float), zero bytes [15:4].
    builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
    // FRINTA dance (FP32 form).
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(abs_bits.machine_reg(), abs_bits.machine_reg());
    builder_.Gen<x86_64::PsrldXRegImm>(abs_bits.machine_reg(), int8_t{1});   // 0x7FFFFFFF
    builder_.Gen<x86_64::PandXRegXReg>(abs_bits.machine_reg(), xn.machine_reg());  // |bits|
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(copysign.machine_reg(), copysign.machine_reg());
    builder_.Gen<x86_64::PslldXRegImm>(copysign.machine_reg(), int8_t{31});  // 0x80000000
    builder_.Gen<x86_64::PandXRegXReg>(copysign.machine_reg(), xn.machine_reg());  // sign bit
    Register gp_half = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x3F000000)));  // 0.5
    builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), gp_half);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(half.machine_reg(), half.machine_reg(), int8_t{0x00});
    builder_.Gen<x86_64::PorXRegXReg>(copysign.machine_reg(), half.machine_reg());  // sign|0.5
    Register gp_pow = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4B000000)));  // 2^23
    builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), gp_pow);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(half.machine_reg(), half.machine_reg(), int8_t{0x00});
    builder_.Gen<x86_64::PcmpgtdXRegXReg>(half.machine_reg(), abs_bits.machine_reg());  // 1s where |x|<2^23
    builder_.Gen<x86_64::PandXRegXReg>(copysign.machine_reg(), half.machine_reg());  // zero addend if int
    builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), copysign.machine_reg());
    builder_.Gen<x86_64::RoundpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), int8_t{0x03});  // trunc
    if (!is_unsigned) {
      // FCVTZS .2S/.4S saturation fix-up.
      FpRegister x_dst = AllocTempSimdReg();
      FpRegister x_mask = AllocTempSimdReg();
      FpRegister x_eqmin = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
      SetVRegFull(args.rd, x_dst, /*q=*/false);
    } else {
      // FCVTZU .2S/.4S saturation fix-up (offset-by-2^31 trick).
      FpRegister x_dst = AllocTempSimdReg();
      FpRegister x_pow31 = AllocTempSimdReg();
      FpRegister x_needs_off = AllocTempSimdReg();
      FpRegister x_scratch = AllocZeroedSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      Register gp2 = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
      builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp2);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(
          x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
      builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
      SetVRegFull(args.rd, x_dst, /*q=*/false);
    }
    return;
  }
  const bool is_fcvtzs =
      (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtzs);
  const bool is_fcvtzu =
      (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtzu);
  const bool is_scvtf =
      (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kScvtf);
  const bool is_ucvtf =
      (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kUcvtf);
  // FP64 (`.d`, size&1 == 1) scalar converts: x86 has no packed FP64<->int64
  // op (CVTTPD2QQ / CVTUQQ2PD are AVX-512), so the `.d` forms can't use the
  // packed FP32 lane recipe below. Route them through the GP-register scalar
  // helpers instead — the same EmitScvtfUcvtf / EmitFcvtz used by the
  // FpIntConversion GP path — reading/writing the SIMD lane via the
  // src_from_simd / dest_to_simd flags. Mirrors lite_translator.h::
  // AdvSimdScalarTwoRegMisc's `.d` lowering (Cvtsi2sdq / Cvttsd2siq + the
  // ARM saturation/NaN ladder, sf=1 int64).
  if ((args.size & 1) != 0 && (is_scvtf || is_ucvtf || is_fcvtzs || is_fcvtzu)) {
    Decoder::FpIntConvArgs fargs{};
    fargs.rd = args.rd;
    fargs.rn = args.rn;
    fargs.sf = true;  // `.d` integer operand is 64-bit
    if (is_scvtf || is_ucvtf) {
      fargs.op = is_ucvtf ? uint8_t{0b011} : uint8_t{0b010};
      EmitScvtfUcvtf(fargs, /*is_double=*/true, /*fbits=*/0, /*src_from_simd=*/true);
    } else {
      fargs.op = is_fcvtzu ? uint8_t{0b001} : uint8_t{0b000};
      EmitFcvtz(fargs, /*is_double=*/true, /*round_imm=*/-1, /*ties_away=*/false,
                /*fbits=*/0, /*dest_to_simd=*/true);
    }
    return;
  }
  if ((!is_fcvtzs && !is_fcvtzu && !is_scvtf && !is_ucvtf) ||
      (args.size & 1) != 0) {
    UndefinedReturningVoid();
    return;
  }
  const int32_t vn_off =
      GetVRegOffset(args.rn);
  // SCVTF / UCVTF scalar (S): int32 -> FP32.  Mirror the kScvtfV/kUcvtfV FP32
  // recipe on a lane-0-scrubbed source.
  if (is_scvtf || is_ucvtf) {
    FpRegister xn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    // Scrub to lane 0: keep the low 32 bits (the scalar int), zero bytes [15:4].
    builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
    if (is_scvtf) {
      // SCVTF: signed int32 -> FP32 is native.
      builder_.Gen<x86_64::Cvtdq2psXRegXReg>(xn.machine_reg(), xn.machine_reg());
      SetVRegFull(args.rd, xn, /*q=*/false);
      return;
    }
    // UCVTF: CVTDQ2PS + per-lane 2^32 addend for MSB-set lanes.
    FpRegister msb = AllocTempSimdReg();
    FpRegister addend = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(msb.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PsradXRegImm>(msb.machine_reg(), int8_t{31});  // 0 or all-1s
    builder_.Gen<x86_64::Cvtdq2psXRegXReg>(xn.machine_reg(), xn.machine_reg());  // signed convert
    Register gp =
        std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F800000)));  // 2^32
    builder_.Gen<x86_64::MovdXRegReg>(addend.machine_reg(), gp);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(
        addend.machine_reg(), addend.machine_reg(), int8_t{0x00});
    builder_.Gen<x86_64::PandXRegXReg>(addend.machine_reg(), msb.machine_reg());
    builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), addend.machine_reg());
    SetVRegFull(args.rd, xn, /*q=*/false);
    return;
  }
  const bool is_unsigned = is_fcvtzu;
  if (!is_unsigned) {
    // FCVTZS scalar (S): mirror the kFcvtzsV FP32 recipe on a lane-0-scrubbed
    // source (CVTTPS2DQ + NaN->0 + positive-overflow->INT32_MAX fix-up).
    FpRegister xn = AllocTempSimdReg();
    FpRegister x_dst = AllocTempSimdReg();
    FpRegister x_mask = AllocTempSimdReg();
    FpRegister x_eqmin = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    // Scrub to lane 0: keep the low 32 bits, zero bytes [15:4].
    builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
    // 1. Primary truncating conversion.
    builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
    // 2. NaN lanes -> 0.
    builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
    // 3. INT32_MIN (0x80000000) per lane.
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
    builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
    // 4. result == INT32_MIN ?
    builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
    // 5. src non-negative ? PSRAD 31 -> 0 if non-neg, all-1s if neg.
    builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
    builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
    // 6. Flip INT_MIN -> INT_MAX in positive-overflow lanes.
    builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
    SetVRegFull(args.rd, x_dst, /*q=*/false);
    return;
  }
  // FCVTZU scalar (S): mirror the kFcvtzuV FP32 recipe on a lane-0-scrubbed
  // source (subtract-2^31 offset trick + saturate too-big/Inf to UINT32_MAX).
  FpRegister x_dst = AllocTempSimdReg();
  FpRegister x_pow31 = AllocTempSimdReg();
  FpRegister x_needs_off = AllocTempSimdReg();
  FpRegister x_scratch = AllocZeroedSimdReg();
  builder_.GenGetSimd<16>(x_dst.machine_reg(), vn_off);
  // Scrub to lane 0: keep the low 32 bits, zero bytes [15:4].
  builder_.Gen<x86_64::PslldqXRegImm>(x_dst.machine_reg(), int8_t{12});
  builder_.Gen<x86_64::PsrldqXRegImm>(x_dst.machine_reg(), int8_t{12});
  // 1. Clamp neg/NaN to 0 via MAXPS with the zeroed scratch.
  builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
  // 2. Broadcast 2^31 = 0x4F000000 to all four FP32 lanes.
  Register gp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
  builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp);
  builder_.Gen<x86_64::PshufdXRegXRegImm>(
      x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
  // 3. needs_offset = (2^31 <= src_clamped).
  builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
  builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
  // 4. offset_amount = 2^31 where needs_offset.
  builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
  builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
  // 5. src_for_cvt = src_clamped - offset_amount.
  builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
  // 6. too_big = (2^31 <= src_for_cvt).
  builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
  builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
  // 7. CVTTPS2DQ.
  builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
  // 8. Build 0x80000000 per lane; AND needs_offset; OR into result.
  builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
  builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
  builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
  builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
  // 9. Saturate too-big lanes to 0xFFFFFFFF.
  builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
  SetVRegFull(args.rd, x_dst, /*q=*/false);
}

// AdvSIMD scalar three-same.  Only the saturating-doubling-multiply-high
// scalar forms (SQDMULH / SQRDMULH, H and S lanes) are lowered here; every
// other scalar three-same opcode bails to lite via UndefinedReturningVoid.
//
// Lowering is a single-lane application of the vector .4H/.8H and .2S/.4S
// SQDMULH/SQRDMULH recipes in AdvSimdThreeSame (opcode 0b10110).  The
// operands are loaded full-width and scrubbed to lane 0 (PSLLDQ+PSRLDQ keep
// the low 16 / 32 bits and zero everything above), so the packed recipe
// processes only the scalar lane while lanes 1.. stay 0 (a corner-free 0
// result).  SetVRegFull q=false then writes the scalar result with
// Vd[127:lane] = 0.
//   * size=01 (H): SQDMULH via PMULHW+PMULLW combine; SQRDMULH via PMULHRSW
//     (SSSE3); both with the INT16_MIN*INT16_MIN corner fixup.
//   * size=10 (S): two PMULDQ + PSLLQ double + optional 2^31 round + PSHUFD
//     0xDD lift + INT32_MIN*INT32_MIN corner fixup (SSE4.1).
//   * size=00/11 are unallocated for this opcode -> bail.
void HeavyOptimizerFrontend::AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs& args) {
  if (!success()) {
    return;
  }
  using ScOp = Decoder::AdvSimdScalarThreeSameOpcode;
  // Scalar D-form integer ADD/SUB and compares (CMEQ/CMGT/CMGE/CMHI/CMHS/
  // CMTST). ARM restricts these to size=11 (D). Mirror of lite's D-form scalar
  // integer block: run the .2D packed op on lane 0 (the upper lane is computed
  // from Vn/Vm bits above 64 but discarded by the scalar upper-zero store) and
  // commit with SetVRegScalar(rd, x, /*is_double=*/true) — writes Vd[63:0],
  // zeroes Vd[127:64].
  //   ADD -> PADDQ, SUB -> PSUBQ                                     (SSE2)
  //   CMEQ -> PCMPEQQ ; CMTST -> PAND + PCMPEQQ-vs-0 + invert         (SSE4.1)
  //   CMGT -> PCMPGTQ ; CMGE -> !(PCMPGTQ(m,n)) ;
  //   CMHI/CMHS -> sign-bias(1<<63) + PCMPGTQ (+ invert for CMHS)     (SSE4.2)
  if (args.opcode == ScOp::kAdd || args.opcode == ScOp::kSub ||
      args.opcode == ScOp::kCmeq || args.opcode == ScOp::kCmtst ||
      args.opcode == ScOp::kCmgt || args.opcode == ScOp::kCmge ||
      args.opcode == ScOp::kCmhi || args.opcode == ScOp::kCmhs) {
    if (args.size != 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const bool needs_sse4_1 =
        (args.opcode == ScOp::kCmeq || args.opcode == ScOp::kCmtst);
    const bool needs_sse4_2 =
        (args.opcode == ScOp::kCmgt || args.opcode == ScOp::kCmge ||
         args.opcode == ScOp::kCmhi || args.opcode == ScOp::kCmhs);
    if ((needs_sse4_1 && !host_platform::kHasSSE4_1) ||
        (needs_sse4_2 && !host_platform::kHasSSE4_2)) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), GetVRegOffset(args.rn));
    builder_.GenGetSimd<16>(xm.machine_reg(), GetVRegOffset(args.rm));
    FpRegister res = xn;
    switch (args.opcode) {
      case ScOp::kAdd:
        builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      case ScOp::kSub:
        builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      case ScOp::kCmeq:
        builder_.Gen<x86_64::PcmpeqqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      case ScOp::kCmgt:
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      case ScOp::kCmge: {
        // CMGE Vn,Vm == !(Vm > Vn): PCMPGTQ(xm,xn) then XOR all-ones.
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(xm.machine_reg(), xn.machine_reg());
        FpRegister ones = AllocOnesSimdReg();
        builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), ones.machine_reg());
        res = xm;
        break;
      }
      case ScOp::kCmhi: {
        // Unsigned >: XOR the sign bit (1<<63) of both operands, then signed
        // PCMPGTQ(xn,xm) implements the unsigned compare.
        FpRegister sign = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
        builder_.Gen<x86_64::PsllqXRegImm>(sign.machine_reg(), int8_t{63});
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), sign.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign.machine_reg());
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      }
      case ScOp::kCmhs: {
        // Unsigned >=: sign-bias, then !(xm > xn).
        FpRegister sign = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
        builder_.Gen<x86_64::PsllqXRegImm>(sign.machine_reg(), int8_t{63});
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), sign.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign.machine_reg());
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(xm.machine_reg(), xn.machine_reg());
        FpRegister ones = AllocOnesSimdReg();
        builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), ones.machine_reg());
        res = xm;
        break;
      }
      case ScOp::kCmtst: {
        // (Vn & Vm) != 0 ? all-ones : 0. PAND, compare AND-vs-0, invert.
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());
        FpRegister z = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PcmpeqqXRegXReg>(xn.machine_reg(), z.machine_reg());
        // z already has a def, so self-PCMPEQD (build all-ones) is legal.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(z.machine_reg(), z.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), z.machine_reg());
        break;
      }
      default:
        UndefinedReturningVoid();
        return;
    }
    SetVRegScalar(args.rd, res, /*is_double=*/true);
    return;
  }

  // Scalar saturating add/sub SQADD/UQADD/SQSUB/UQSUB, widths B/H/S/D. Mirror of
  // the lite tier's D-form-and-BHS scalar saturating block: load Vn/Vm full,
  // scrub to lane 0 (keep the low (1<<size) bytes, zero the rest via PSLLDQ N +
  // PSRLDQ N) so lanes 1.. compute sat(0,0)=0, apply the per-width recipe, and
  // commit with SetVRegFull q=false (Vd[esize:]=0). QC is left unmodeled, exactly
  // as in the lite tier and the existing heavy SQADD-class vector lowerings.
  if (args.opcode == ScOp::kSqaddScalar || args.opcode == ScOp::kUqaddScalar ||
      args.opcode == ScOp::kSqsubScalar || args.opcode == ScOp::kUqsubScalar) {
    const bool is_add =
        (args.opcode == ScOp::kSqaddScalar || args.opcode == ScOp::kUqaddScalar);
    const bool is_unsigned =
        (args.opcode == ScOp::kUqaddScalar || args.opcode == ScOp::kUqsubScalar);
    // Host-feature gates: S unsigned needs PMAXUD/PMINUD (SSE4.1); D unsigned
    // needs PCMPGTQ (SSE4.2). B/H and S/D signed are SSE2-only.
    if (args.size == 0b10 && is_unsigned && !host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }
    if (args.size == 0b11 && is_unsigned && !host_platform::kHasSSE4_2) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), GetVRegOffset(args.rn));
    builder_.GenGetSimd<16>(xm.machine_reg(), GetVRegOffset(args.rm));
    // Scrub both operands to lane 0 (keep low (1<<size) bytes).
    const int8_t scrub = static_cast<int8_t>(16 - (1 << args.size));  // 15,14,12,8
    builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), scrub);
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), scrub);
    builder_.Gen<x86_64::PslldqXRegImm>(xm.machine_reg(), scrub);
    builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), scrub);
    if (args.size == 0b00) {  // B
      if (is_add && !is_unsigned)       builder_.Gen<x86_64::PaddsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
      else if (is_add && is_unsigned)   builder_.Gen<x86_64::PaddusbXRegXReg>(xn.machine_reg(), xm.machine_reg());
      else if (!is_add && !is_unsigned) builder_.Gen<x86_64::PsubsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
      else                              builder_.Gen<x86_64::PsubusbXRegXReg>(xn.machine_reg(), xm.machine_reg());
      SetVRegFull(args.rd, xn, /*q=*/false);
      return;
    }
    if (args.size == 0b01) {  // H
      if (is_add && !is_unsigned)       builder_.Gen<x86_64::PaddswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      else if (is_add && is_unsigned)   builder_.Gen<x86_64::PadduswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      else if (!is_add && !is_unsigned) builder_.Gen<x86_64::PsubswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      else                              builder_.Gen<x86_64::PsubuswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      SetVRegFull(args.rd, xn, /*q=*/false);
      return;
    }
    if (args.size == 0b10) {  // S (32-bit lane)
      if (is_add && !is_unsigned) {
        // SQADD 32: wrap-add + sign-bit XOR-blend saturation.
        FpRegister t_sum = AllocTempSimdReg();
        FpRegister t_ovf = AllocTempSimdReg();
        FpRegister t_sat = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_sum.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PadddXRegXReg>(t_sum.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_sum.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());  // -1
        builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());  // ~(a^b)
        builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});        // ovf mask
        builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{31});           // sign(a)
        builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{1});            // 0x7FFFFFFF
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());     // sat
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_sum.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_ovf.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_sum.machine_reg());
      } else if (is_add && is_unsigned) {
        // UQADD 32: wrap-add + PMAXUD overflow detect.
        FpRegister t_a = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_a.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xm.machine_reg());     // sum
        builder_.Gen<x86_64::PmaxudXRegXReg>(t_a.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(t_a.machine_reg(), xn.machine_reg());  // -1 no ovf
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());   // -1
        builder_.Gen<x86_64::PxorXRegXReg>(t_a.machine_reg(), xm.machine_reg());     // -1 where ovf
        builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), t_a.machine_reg());
      } else if (!is_add && !is_unsigned) {
        // SQSUB 32: wrap-sub + sign-bit XOR-blend.
        FpRegister t_diff = AllocTempSimdReg();
        FpRegister t_ovf = AllocTempSimdReg();
        FpRegister t_sat = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_diff.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsubdXRegXReg>(t_diff.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());   // a^b
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_diff.machine_reg()); // a^diff
        builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{31});            // sign(a)
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{1});             // 0x7FFFFFFF
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());      // sat
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_ovf.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
      } else {
        // UQSUB 32: (a>=b) ? a-b : 0, via PMINUD.
        FpRegister t_mask = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PminudXRegXReg>(t_mask.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(t_mask.machine_reg(), xm.machine_reg());  // -1 where a>=b
        builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_mask.machine_reg());
      }
      SetVRegFull(args.rd, xn, /*q=*/false);
      return;
    }
    // args.size == 0b11 (D, 64-bit lane). Per-qword recipes; qword 1 = 0 after
    // scrub, SetVRegFull q=false re-zeroes Vd[127:64].
    if (is_add && !is_unsigned) {
      // SQADD 64: wrap-add + sign-bit XOR-blend; PSRAD 31 + PSHUFD 0xF5 fans
      // bit63 across the qword (no PSRAQ in baseline SSE).
      FpRegister t_sum = AllocTempSimdReg();
      FpRegister t_ovf = AllocTempSimdReg();
      FpRegister t_sat = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_sum.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PaddqXRegXReg>(t_sum.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_sum.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());  // -1
      builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());  // ~(a^b)
      builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PshufdXRegXRegImm>(t_ovf.machine_reg(), t_ovf.machine_reg(), static_cast<int8_t>(0xF5));
      builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xF5));
      builder_.Gen<x86_64::PsrlqXRegImm>(xm.machine_reg(), int8_t{1});            // 0x7FFF..FF
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());     // sat
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_sum.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_ovf.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_sum.machine_reg());
    } else if (is_add && is_unsigned) {
      // UQADD 64: wrap-add + PCMPGTQ(sign-flipped) overflow detect (SSE4.2).
      FpRegister t_a = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_a.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xm.machine_reg());    // sum
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PsllqXRegImm>(xm.machine_reg(), int8_t{63});           // sign mask
      builder_.Gen<x86_64::PxorXRegXReg>(t_a.machine_reg(), xm.machine_reg());    // a ^ sign
      builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), xn.machine_reg());     // sum ^ sign
      builder_.Gen<x86_64::PcmpgtqXRegXReg>(t_a.machine_reg(), xm.machine_reg()); // -1 where a>sum (ovf)
      builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), t_a.machine_reg());
    } else if (!is_add && !is_unsigned) {
      // SQSUB 64: wrap-sub + sign-bit XOR-blend.
      FpRegister t_diff = AllocTempSimdReg();
      FpRegister t_ovf = AllocTempSimdReg();
      FpRegister t_sat = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_diff.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsubqXRegXReg>(t_diff.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());   // a^b
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_diff.machine_reg()); // a^diff
      builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PshufdXRegXRegImm>(t_ovf.machine_reg(), t_ovf.machine_reg(), static_cast<int8_t>(0xF5));
      builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xF5));
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PsrlqXRegImm>(xm.machine_reg(), int8_t{1});             // 0x7FFF..FF
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());      // sat
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_ovf.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
    } else {
      // UQSUB 64: (a>=b) ? a-b : 0, via sign-flipped PCMPGTQ (SSE4.2).
      FpRegister t_diff = AllocTempSimdReg();
      FpRegister t_mask = AllocOnesSimdReg();  // -1 (AllocOnes avoids fresh self-PCMPEQ trap)
      builder_.Gen<x86_64::MovdqaXRegXReg>(t_diff.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsubqXRegXReg>(t_diff.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PsllqXRegImm>(t_mask.machine_reg(), int8_t{63});        // sign mask
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_mask.machine_reg());  // a ^ sign
      builder_.Gen<x86_64::PxorXRegXReg>(t_mask.machine_reg(), xm.machine_reg());  // b ^ sign
      builder_.Gen<x86_64::PcmpgtqXRegXReg>(t_mask.machine_reg(), xn.machine_reg()); // -1 where b>a (underflow)
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xn.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(t_mask.machine_reg(), xn.machine_reg());  // -1 where a>=b
      builder_.Gen<x86_64::PandXRegXReg>(t_diff.machine_reg(), t_mask.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
    }
    SetVRegFull(args.rd, xn, /*q=*/false);
    return;
  }

  // SSHL/USHL/SRSHL/URSHL scalar, D-form only (size=11; other sizes are the
  // vector encoding, decoder-rejected here). The lite tier lowers these with GPR
  // branches (>=64 -> 0, negative count -> opposite-direction shift); the heavy
  // tier is branchless single-BB, so this re-derives the same ARM semantics with
  // arithmetic sign/keep masks (SAR #63 -> 0/-1) and AND/OR/NOT blends. No CMOV.
  if (args.opcode == ScOp::kSshl || args.opcode == ScOp::kUshl ||
      args.opcode == ScOp::kSrshlScalar || args.opcode == ScOp::kUrshlScalar) {
    if (args.size != 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_signed =
        (args.opcode == ScOp::kSshl || args.opcode == ScOp::kSrshlScalar);
    const bool is_rounding =
        (args.opcode == ScOp::kSrshlScalar || args.opcode == ScOp::kUrshlScalar);
    // Load a = Vn[63:0] and the sign-extended int8 shift count.
    FpRegister xn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), GetVRegOffset(args.rn));
    Register a = std::get<0>(Gen<x86_64::MovqRegXReg>(xn.machine_reg()));
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xm.machine_reg(), GetVRegOffset(args.rm));
    Register mraw = std::get<0>(Gen<x86_64::MovqRegXReg>(xm.machine_reg()));
    Register shx = std::get<0>(Gen<x86_64::ShlqRegImm>(mraw, int8_t{56}));
    Register sh = std::get<0>(Gen<x86_64::SarqRegImm>(shx, int8_t{56}));  // sign-ext int8
    // neg = (sh<0) ? -1 : 0 ; absh = |sh| ; keeplt64 = (absh<64) ? -1 : 0.
    Register neg = std::get<0>(Gen<x86_64::SarqRegImm>(sh, int8_t{63}));
    Register t = std::get<0>(Gen<x86_64::XorqRegReg>(sh, neg));
    Register absh = std::get<0>(Gen<x86_64::SubqRegReg>(t, neg));
    Register absm64 = std::get<0>(Gen<x86_64::SubqRegImm>(absh, int32_t{64}));
    Register keeplt64 = std::get<0>(Gen<x86_64::SarqRegImm>(absm64, int8_t{63}));
    // LEFT arm: a << (absh&63), zeroed when absh>=64.
    Register left = std::get<0>(Gen<x86_64::ShlqRegReg>(a, absh));
    left = std::get<0>(Gen<x86_64::AndqRegReg>(left, keeplt64));
    // RIGHT arm.
    Register right;
    if (!is_rounding) {
      // SSHL/USHL: plain shift; masking differs by signedness for n>=64.
      Register rmain = is_signed
          ? std::get<0>(Gen<x86_64::SarqRegReg>(a, absh))
          : std::get<0>(Gen<x86_64::ShrqRegReg>(a, absh));
      if (is_signed) {
        // n>=64 -> sign-broadcast; blend override with keeplt64.
        Register signb = std::get<0>(Gen<x86_64::SarqRegImm>(a, int8_t{63}));
        Register keep = std::get<0>(Gen<x86_64::AndqRegReg>(rmain, keeplt64));
        Register nk = std::get<0>(Gen<x86_64::NotqReg>(keeplt64));
        Register ov = std::get<0>(Gen<x86_64::AndqRegReg>(signb, nk));
        right = std::get<0>(Gen<x86_64::OrqRegReg>(keep, ov));
      } else {
        // n>=64 -> 0.
        right = std::get<0>(Gen<x86_64::AndqRegReg>(rmain, keeplt64));
      }
    } else {
      // URSHL/SRSHL: rounded right shift = (a >> n) + bit(n-1).
      Register rmain0 = is_signed
          ? std::get<0>(Gen<x86_64::SarqRegReg>(a, absh))
          : std::get<0>(Gen<x86_64::ShrqRegReg>(a, absh));
      Register rmain = std::get<0>(Gen<x86_64::AndqRegReg>(rmain0, keeplt64));  // 0 when absh>=64
      Register nm1 = std::get<0>(Gen<x86_64::SubqRegImm>(absh, int32_t{1}));    // n-1
      Register r0 = std::get<0>(Gen<x86_64::ShrqRegReg>(a, nm1));
      Register rbit = std::get<0>(Gen<x86_64::AndqRegImm>(r0, int32_t{1}));
      Register round;
      if (is_signed) {
        // SRSHL collapses n>=64 to 0 (round dropped too): keeplt64.
        round = std::get<0>(Gen<x86_64::AndqRegReg>(rbit, keeplt64));
      } else {
        // URSHL keeps bit63 at n==64: keep_round = (absh-1<64) ? -1 : 0.
        Register nm1m64 = std::get<0>(Gen<x86_64::SubqRegImm>(nm1, int32_t{64}));
        Register keepr = std::get<0>(Gen<x86_64::SarqRegImm>(nm1m64, int8_t{63}));
        round = std::get<0>(Gen<x86_64::AndqRegReg>(rbit, keepr));
      }
      right = std::get<0>(Gen<x86_64::AddqRegReg>(rmain, round));
    }
    // Direction blend: res = (right & neg) | (left & ~neg).
    Register rmask = std::get<0>(Gen<x86_64::AndqRegReg>(right, neg));
    Register nneg = std::get<0>(Gen<x86_64::NotqReg>(neg));
    Register lmask = std::get<0>(Gen<x86_64::AndqRegReg>(left, nneg));
    Register res = std::get<0>(Gen<x86_64::OrqRegReg>(rmask, lmask));
    SetVRegScalarFromGp(args.rd, res, /*is_double=*/true);
    return;
  }

  // Scalar FP three-same FABD/FMULX/FRECPS/FRSQRTS (S/D). Branchless SSE + FMA
  // mask ladders, op-for-op mirror of the lite tier's scalar FP recipes. FP16
  // (is_fp16) bails to lite (owned by the FP16 slice; matches AdvSimdScalarPairwise).
  if (args.opcode == ScOp::kFabd || args.opcode == ScOp::kFmulx ||
      args.opcode == ScOp::kFrecps || args.opcode == ScOp::kFrsqrts) {
    if (args.is_fp16) {
      // FP16 scalar: only FMULX is covered here via the F16C round-trip (widen
      // Vn.h[0]/Vm.h[0] to FP32 lane 0 with upper lanes +0.0, apply the packed
      // FMULX blend, narrow once). FABD/FRECPS/FRSQRTS FP16 still bail to lite.
      if (args.opcode != ScOp::kFmulx || !host_platform::kHasF16C) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister n = EmitWidenHalfToF32(args.rn);
      FpRegister m = EmitWidenHalfToF32(args.rm);
      FpRegister r = EmitFmulxF32Packed(n, m);
      EmitNarrowF32ToHalfAndStore(args.rd, r);
      return;
    }
    if ((args.opcode == ScOp::kFrecps || args.opcode == ScOp::kFrsqrts) &&
        !host_platform::kHasFMA) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_double = (args.size != 0);
    // Load Vn/Vm full into private temps; only lane 0 matters.
    FpRegister n = AllocTempSimdReg();
    FpRegister m = AllocTempSimdReg();
    builder_.GenGetSimd<16>(n.machine_reg(), GetVRegOffset(args.rn));
    builder_.GenGetSimd<16>(m.machine_reg(), GetVRegOffset(args.rm));

    if (args.opcode == ScOp::kFabd) {
      // |a - b|: SUBSS/SUBSD then AND with the non-sign mask.
      if (is_double) builder_.Gen<x86_64::SubsdXRegXReg>(n.machine_reg(), m.machine_reg());
      else           builder_.Gen<x86_64::SubssXRegXReg>(n.machine_reg(), m.machine_reg());
      FpRegister mask = AllocOnesSimdReg();  // all-ones
      if (is_double) builder_.Gen<x86_64::PsrlqXRegImm>(mask.machine_reg(), int8_t{1});
      else           builder_.Gen<x86_64::PsrldXRegImm>(mask.machine_reg(), int8_t{1});
      builder_.Gen<x86_64::PandXRegXReg>(n.machine_reg(), mask.machine_reg());
      SetVRegScalar(args.rd, n, is_double);
      return;
    }

    if (args.opcode == ScOp::kFmulx) {
      // FMUL, with (0*inf) lanes replaced by +-2.0 (sign = sign(a)^sign(b)).
      FpRegister mul = AllocTempSimdReg();
      FpRegister mul_unord = AllocTempSimdReg();
      FpRegister input_unord = AllocTempSimdReg();
      FpRegister two = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(mul.machine_reg(), n.machine_reg());
      if (is_double) builder_.Gen<x86_64::MulsdXRegXReg>(mul.machine_reg(), m.machine_reg());
      else           builder_.Gen<x86_64::MulssXRegXReg>(mul.machine_reg(), m.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(mul_unord.machine_reg(), mul.machine_reg());
      if (is_double) builder_.Gen<x86_64::CmpunordpdXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
      else           builder_.Gen<x86_64::CmpunordpsXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(input_unord.machine_reg(), n.machine_reg());
      if (is_double) builder_.Gen<x86_64::CmpunordpdXRegXReg>(input_unord.machine_reg(), m.machine_reg());
      else           builder_.Gen<x86_64::CmpunordpsXRegXReg>(input_unord.machine_reg(), m.machine_reg());
      // special = (NOT input_unord) AND mul_unord  (PANDN dst = ~dst & src).
      builder_.Gen<x86_64::PandnXRegXReg>(input_unord.machine_reg(), mul_unord.machine_reg());
      // two_signed = (a ^ b) & signmask | bits(+-2.0).  (n is free after mul.)
      builder_.Gen<x86_64::PxorXRegXReg>(n.machine_reg(), m.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());  // -1
      if (is_double) builder_.Gen<x86_64::PsllqXRegImm>(mul_unord.machine_reg(), int8_t{63});
      else           builder_.Gen<x86_64::PslldXRegImm>(mul_unord.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PandXRegXReg>(n.machine_reg(), mul_unord.machine_reg());
      if (is_double) builder_.Gen<x86_64::MovqXRegReg>(two.machine_reg(), GetImm(uint64_t{0x4000000000000000ULL}));
      else           builder_.Gen<x86_64::MovdXRegReg>(two.machine_reg(), GetImm(uint64_t{0x40000000ULL}));
      builder_.Gen<x86_64::PorXRegXReg>(n.machine_reg(), two.machine_reg());
      // result = (mul & ~special) | (two & special).
      builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), n.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(m.machine_reg(), input_unord.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(input_unord.machine_reg(), mul.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(input_unord.machine_reg(), m.machine_reg());
      SetVRegScalar(args.rd, input_unord, is_double);
      return;
    }

    // FRECPS / FRSQRTS Newton step.
    const bool is_frecps = (args.opcode == ScOp::kFrecps);
    const uint64_t k_fma_d = is_frecps ? 0x4000000000000000ULL : 0x4008000000000000ULL;
    const uint64_t k_fma_s = is_frecps ? 0x40000000ULL : 0x40400000ULL;
    const uint64_t k_sat_d = is_frecps ? 0x4000000000000000ULL : 0x3FF8000000000000ULL;
    const uint64_t k_sat_s = is_frecps ? 0x40000000ULL : 0x3FC00000ULL;
    const uint64_t qnan_d = 0x7FF8000000000000ULL;
    const uint64_t qnan_s = 0x7FC00000ULL;
    const uint64_t two_d = 0x4000000000000000ULL;
    const uint64_t two_s = 0x40000000ULL;
    FpRegister mul = AllocTempSimdReg();
    FpRegister iu = AllocTempSimdReg();
    FpRegister special = AllocTempSimdReg();
    // input_unord = cmpunord(a,b).
    builder_.Gen<x86_64::MovdqaXRegXReg>(iu.machine_reg(), n.machine_reg());
    if (is_double) builder_.Gen<x86_64::CmpunordpdXRegXReg>(iu.machine_reg(), m.machine_reg());
    else           builder_.Gen<x86_64::CmpunordpsXRegXReg>(iu.machine_reg(), m.machine_reg());
    // mul_unord = cmpunord(a*b, a*b).
    builder_.Gen<x86_64::MovdqaXRegXReg>(mul.machine_reg(), n.machine_reg());
    if (is_double) builder_.Gen<x86_64::MulsdXRegXReg>(mul.machine_reg(), m.machine_reg());
    else           builder_.Gen<x86_64::MulssXRegXReg>(mul.machine_reg(), m.machine_reg());
    if (is_double) builder_.Gen<x86_64::CmpunordpdXRegXReg>(mul.machine_reg(), mul.machine_reg());
    else           builder_.Gen<x86_64::CmpunordpsXRegXReg>(mul.machine_reg(), mul.machine_reg());
    // special = (NOT iu) AND mul_unord (preserve iu).
    builder_.Gen<x86_64::MovdqaXRegXReg>(special.machine_reg(), iu.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(special.machine_reg(), mul.machine_reg());
    // fma = K_fma - a*b (into mul, reused).
    if (is_double) {
      builder_.Gen<x86_64::MovqXRegReg>(mul.machine_reg(), GetImm(k_fma_d));
      builder_.Gen<x86_64::Vfnmadd231sdXRegXRegXReg>(mul.machine_reg(), n.machine_reg(), m.machine_reg());
    } else {
      builder_.Gen<x86_64::MovdXRegReg>(mul.machine_reg(), GetImm(k_fma_s));
      builder_.Gen<x86_64::Vfnmadd231ssXRegXRegXReg>(mul.machine_reg(), n.machine_reg(), m.machine_reg());
    }
    if (!is_frecps) {
      // FRSQRTS: divide by 2 (exact). Reuse n as the +2.0 divisor.
      if (is_double) {
        builder_.Gen<x86_64::MovqXRegReg>(n.machine_reg(), GetImm(two_d));
        builder_.Gen<x86_64::DivsdXRegXReg>(mul.machine_reg(), n.machine_reg());
      } else {
        builder_.Gen<x86_64::MovdXRegReg>(n.machine_reg(), GetImm(two_s));
        builder_.Gen<x86_64::DivssXRegXReg>(mul.machine_reg(), n.machine_reg());
      }
    }
    // result_first = special ? K_sat : fma  (build K_sat in n).
    if (is_double) builder_.Gen<x86_64::MovqXRegReg>(n.machine_reg(), GetImm(k_sat_d));
    else           builder_.Gen<x86_64::MovdXRegReg>(n.machine_reg(), GetImm(k_sat_s));
    builder_.Gen<x86_64::PandXRegXReg>(n.machine_reg(), special.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(special.machine_reg(), mul.machine_reg());  // ~special & fma
    builder_.Gen<x86_64::PorXRegXReg>(n.machine_reg(), special.machine_reg());       // result_first
    // result_final = iu ? qnan : result_first  (build qnan in m).
    if (is_double) builder_.Gen<x86_64::MovqXRegReg>(m.machine_reg(), GetImm(qnan_d));
    else           builder_.Gen<x86_64::MovdXRegReg>(m.machine_reg(), GetImm(qnan_s));
    builder_.Gen<x86_64::PandXRegXReg>(m.machine_reg(), iu.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(iu.machine_reg(), n.machine_reg());          // ~iu & result_first
    builder_.Gen<x86_64::PorXRegXReg>(m.machine_reg(), iu.machine_reg());            // result_final
    SetVRegScalar(args.rd, m, is_double);
    return;
  }

  // SQRDMLAH/SQRDMLSH scalar (Armv8.1-RDM), H/S only. Lite lowers these
  // (SSSE3 H / SSE4.1 S), so heavy mirrors. Stage 1 = SQRDMULH(Vn,Vm) with the
  // (INT_MIN)^2 corner fix; stage 2 = signed-saturating add (MLAH) / sub (MLSH)
  // of the stage-1 result into Vd. Sources/Vd loaded full and scrubbed to lane 0.
  if (args.opcode == ScOp::kSqrdmlahScalar || args.opcode == ScOp::kSqrdmlshScalar) {
    if (args.size != 0b01 && args.size != 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_sub = (args.opcode == ScOp::kSqrdmlshScalar);
    if (args.size == 0b01) {  // H: PMULHRSW stage 1 (SSSE3).
      if (!host_platform::kHasSSSE3) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), GetVRegOffset(args.rn));
      builder_.GenGetSimd<16>(xm.machine_reg(), GetVRegOffset(args.rm));
      builder_.GenGetSimd<16>(xd.machine_reg(), GetVRegOffset(args.rd));
      // Scrub to lane 0 (keep low 2 bytes).
      for (FpRegister r : {xn, xm, xd}) {
        builder_.Gen<x86_64::PslldqXRegImm>(r.machine_reg(), int8_t{14});
        builder_.Gen<x86_64::PsrldqXRegImm>(r.machine_reg(), int8_t{14});
      }
      FpRegister x_min = AllocOnesSimdReg();
      builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});  // 0x8000 bcast
      FpRegister xn_c = AllocTempSimdReg();
      FpRegister xm_c = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(xn_c.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xm_c.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PmulhrswXRegXReg>(xn.machine_reg(), xm.machine_reg());  // stage 1
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_c.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_c.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xn_c.machine_reg(), xm_c.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_c.machine_reg());    // corner fix
      if (is_sub) builder_.Gen<x86_64::PsubswXRegXReg>(xd.machine_reg(), xn.machine_reg());
      else        builder_.Gen<x86_64::PaddswXRegXReg>(xd.machine_reg(), xn.machine_reg());
      SetVRegFull(args.rd, xd, /*q=*/false);
      return;
    }
    // S: PMULDQ stage 1 (SSE4.1) + 32-bit signed-saturating accumulate.
    if (!host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), GetVRegOffset(args.rn));
    builder_.GenGetSimd<16>(xm.machine_reg(), GetVRegOffset(args.rm));
    builder_.GenGetSimd<16>(xd.machine_reg(), GetVRegOffset(args.rd));
    for (FpRegister r : {xn, xm, xd}) {
      builder_.Gen<x86_64::PslldqXRegImm>(r.machine_reg(), int8_t{12});  // keep low 4 bytes
      builder_.Gen<x86_64::PsrldqXRegImm>(r.machine_reg(), int8_t{12});
    }
    FpRegister x_const = AllocOnesSimdReg();
    builder_.Gen<x86_64::PslldXRegImm>(x_const.machine_reg(), int8_t{31});  // INT32_MIN bcast
    FpRegister corner = AllocTempSimdReg();
    FpRegister xp = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(corner.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(corner.machine_reg(), x_const.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(xp.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(xp.machine_reg(), x_const.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(corner.machine_reg(), xp.machine_reg());
    // Stage 1: 2 * sext(Vn.s0) * sext(Vm.s0), rounded (+2^31), high32 -> dword0.
    builder_.Gen<x86_64::MovdqaXRegXReg>(xp.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PmuldqXRegXReg>(xp.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PsllqXRegImm>(xp.machine_reg(), int8_t{1});  // double
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
    builder_.Gen<x86_64::PsllqXRegImm>(x_const.machine_reg(), int8_t{63});
    builder_.Gen<x86_64::PsrlqXRegImm>(x_const.machine_reg(), int8_t{32});  // 2^31 per qword
    builder_.Gen<x86_64::PaddqXRegXReg>(xp.machine_reg(), x_const.machine_reg());  // rounding
    builder_.Gen<x86_64::PsrlqXRegImm>(xp.machine_reg(), int8_t{32});  // high32 -> dword0
    builder_.Gen<x86_64::PxorXRegXReg>(xp.machine_reg(), corner.machine_reg());  // corner -> INT32_MAX
    // Stage 2: 32-bit signed-saturating accumulate xd (+/-)= xp (wrap + XOR-blend).
    FpRegister t_sum = AllocTempSimdReg();
    FpRegister t_ovf = AllocTempSimdReg();
    FpRegister t_sat = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(t_sum.machine_reg(), xd.machine_reg());
    if (is_sub) builder_.Gen<x86_64::PsubdXRegXReg>(t_sum.machine_reg(), xp.machine_reg());
    else        builder_.Gen<x86_64::PadddXRegXReg>(t_sum.machine_reg(), xp.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xd.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xp.machine_reg());       // d ^ p
    builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xd.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_sum.machine_reg());    // d ^ sum
    if (!is_sub) {
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xp.machine_reg(), xp.machine_reg());     // -1
      builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xp.machine_reg());     // ~(d^p)
      builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(xp.machine_reg(), int8_t{1});               // 0x7FFFFFFF
    } else {
      builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xp.machine_reg(), xp.machine_reg());
      builder_.Gen<x86_64::PsrldXRegImm>(xp.machine_reg(), int8_t{1});               // 0x7FFFFFFF
    }
    builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});             // ovf mask
    builder_.Gen<x86_64::PsradXRegImm>(xd.machine_reg(), int8_t{31});                // sign(d)
    builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), xp.machine_reg());          // sat
    builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), t_sum.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(xd.machine_reg(), t_ovf.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), t_sum.machine_reg());
    SetVRegFull(args.rd, xd, /*q=*/false);
    return;
  }

  if (args.opcode != Decoder::AdvSimdScalarThreeSameOpcode::kSqdmulhScalar &&
      args.opcode != Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmulhScalar) {
    UndefinedReturningVoid();
    return;
  }
  if (args.size != 0b01 && args.size != 0b10) {
    UndefinedReturningVoid();
    return;
  }
  const bool is_round =
      (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmulhScalar);
  const int32_t vn_off =
      GetVRegOffset(args.rn);
  const int32_t vm_off =
      GetVRegOffset(args.rm);
  if (args.size == 0b01) {
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    // Scrub to lane 0: keep the low 16 bits, zero bytes [15:2].
    builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{14});
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{14});
    builder_.Gen<x86_64::PslldqXRegImm>(xm.machine_reg(), int8_t{14});
    builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{14});
    if (is_round) {
      if (!host_platform::kHasSSSE3) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn_corner = AllocTempSimdReg();
      FpRegister xm_corner = AllocTempSimdReg();
      FpRegister x_min = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqw idiom
      builder_.Gen<x86_64::MovdqaXRegXReg>(xn_corner.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xm_corner.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(x_min.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});  // INT16_MIN
      builder_.Gen<x86_64::PmulhrswXRegXReg>(xn.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_corner.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_corner.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xn_corner.machine_reg(), xm_corner.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_corner.machine_reg());
      SetVRegFull(args.rd, xn, /*q=*/false);
      return;
    }
    // SQDMULH .H via PMULHW + PMULLW combine + corner fixup (SSE2).
    FpRegister xn_lo = AllocTempSimdReg();
    FpRegister xn_corner = AllocTempSimdReg();
    FpRegister xm_corner = AllocTempSimdReg();
    FpRegister x_min = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqw idiom
    builder_.Gen<x86_64::MovdqaXRegXReg>(xn_corner.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(xm_corner.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(xn_lo.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PmullwXRegXReg>(xn_lo.machine_reg(), xm.machine_reg());  // low16(a*b)
    builder_.Gen<x86_64::PmulhwXRegXReg>(xn.machine_reg(), xm.machine_reg());     // high16(a*b)
    builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{1});              // high<<1
    builder_.Gen<x86_64::PsrlwXRegImm>(xn_lo.machine_reg(), int8_t{15});          // low top bit
    builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xn_lo.machine_reg());     // high16(2*a*b)
    builder_.Gen<x86_64::PcmpeqwXRegXReg>(x_min.machine_reg(), x_min.machine_reg());
    builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});          // INT16_MIN
    builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_corner.machine_reg(), x_min.machine_reg());
    builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_corner.machine_reg(), x_min.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(xn_corner.machine_reg(), xm_corner.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_corner.machine_reg());
    SetVRegFull(args.rd, xn, /*q=*/false);
    return;
  }
  // size == 0b10 (S form): PMULDQ widen (SSE4.1) + PSLLQ double + optional
  // round + corner fixup; PSHUFD 0xDD lifts each product's upper 32 bits.
  if (!host_platform::kHasSSE4_1) {
    UndefinedReturningVoid();
    return;
  }
  FpRegister xn = AllocTempSimdReg();
  FpRegister xm = AllocTempSimdReg();
  FpRegister x_const = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqd idiom
  FpRegister corner = AllocTempSimdReg();
  FpRegister xp_lo = AllocTempSimdReg();
  FpRegister xp_hi = AllocTempSimdReg();
  FpRegister xm_hi = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
  builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
  // Scrub to lane 0: keep the low 32 bits, zero bytes [15:4].
  builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
  builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
  builder_.Gen<x86_64::PslldqXRegImm>(xm.machine_reg(), int8_t{12});
  builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{12});
  // x_const = INT32_MIN broadcast across 4 dwords.
  builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
  builder_.Gen<x86_64::PslldXRegImm>(x_const.machine_reg(), int8_t{31});
  // Corner: lane 0 where Vn.s[0] == INT32_MIN AND Vm.s[0] == INT32_MIN.
  builder_.Gen<x86_64::MovdqaXRegXReg>(corner.machine_reg(), xn.machine_reg());
  builder_.Gen<x86_64::PcmpeqdXRegXReg>(corner.machine_reg(), x_const.machine_reg());
  builder_.Gen<x86_64::MovdqaXRegXReg>(xp_lo.machine_reg(), xm.machine_reg());
  builder_.Gen<x86_64::PcmpeqdXRegXReg>(xp_lo.machine_reg(), x_const.machine_reg());
  builder_.Gen<x86_64::PandXRegXReg>(corner.machine_reg(), xp_lo.machine_reg());
  // Two PMULDQs reconstruct the signed 32x32 -> 64 products (even lanes in
  // xp_lo, odd lanes in xp_hi).
  builder_.Gen<x86_64::MovdqaXRegXReg>(xp_lo.machine_reg(), xn.machine_reg());
  builder_.Gen<x86_64::PmuldqXRegXReg>(xp_lo.machine_reg(), xm.machine_reg());
  builder_.Gen<x86_64::MovdqaXRegXReg>(xp_hi.machine_reg(), xn.machine_reg());
  builder_.Gen<x86_64::PsrlqXRegImm>(xp_hi.machine_reg(), int8_t{32});
  builder_.Gen<x86_64::MovdqaXRegXReg>(xm_hi.machine_reg(), xm.machine_reg());
  builder_.Gen<x86_64::PsrlqXRegImm>(xm_hi.machine_reg(), int8_t{32});
  builder_.Gen<x86_64::PmuldqXRegXReg>(xp_hi.machine_reg(), xm_hi.machine_reg());
  // Double each 64-bit signed product.
  builder_.Gen<x86_64::PsllqXRegImm>(xp_lo.machine_reg(), int8_t{1});
  builder_.Gen<x86_64::PsllqXRegImm>(xp_hi.machine_reg(), int8_t{1});
  if (is_round) {
    // SQRDMULH: add rounding constant 2^31 = 0x80000000 per qword.
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
    builder_.Gen<x86_64::PsllqXRegImm>(x_const.machine_reg(), int8_t{63});
    builder_.Gen<x86_64::PsrlqXRegImm>(x_const.machine_reg(), int8_t{32});
    builder_.Gen<x86_64::PaddqXRegXReg>(xp_lo.machine_reg(), x_const.machine_reg());
    builder_.Gen<x86_64::PaddqXRegXReg>(xp_hi.machine_reg(), x_const.machine_reg());
  }
  // Extract the upper 32 bits of each 64-bit lane and interleave.
  builder_.Gen<x86_64::PshufdXRegXRegImm>(xp_lo.machine_reg(), xp_lo.machine_reg(), static_cast<int8_t>(0xDD));
  builder_.Gen<x86_64::PshufdXRegXRegImm>(xp_hi.machine_reg(), xp_hi.machine_reg(), static_cast<int8_t>(0xDD));
  builder_.Gen<x86_64::PunpckldqXRegXReg>(xp_lo.machine_reg(), xp_hi.machine_reg());
  // Apply corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX.
  builder_.Gen<x86_64::PxorXRegXReg>(xp_lo.machine_reg(), corner.machine_reg());
  SetVRegFull(args.rd, xp_lo, /*q=*/false);
}

HeavyOptimizerFrontend::FpRegister HeavyOptimizerFrontend::EmitFmulxF32Packed(
    FpRegister a, FpRegister b) {
  // mul = a * b.
  FpRegister mul = AllocTempSimdReg();
  builder_.Gen<x86_64::MovdqaXRegXReg>(mul.machine_reg(), a.machine_reg());
  builder_.Gen<x86_64::MulpsXRegXReg>(mul.machine_reg(), b.machine_reg());
  // mul_unord = cmpunord(mul, mul) : -1/lane where mul is NaN.
  FpRegister mul_unord = AllocTempSimdReg();
  builder_.Gen<x86_64::MovdqaXRegXReg>(mul_unord.machine_reg(), mul.machine_reg());
  builder_.Gen<x86_64::CmpunordpsXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
  // special = (NOT input_unord) AND mul_unord : lanes that are exactly +-0*+-inf
  // (product NaN but neither input NaN). Pandn writes (NOT dst) AND src.
  FpRegister special = AllocTempSimdReg();
  builder_.Gen<x86_64::MovdqaXRegXReg>(special.machine_reg(), a.machine_reg());
  builder_.Gen<x86_64::CmpunordpsXRegXReg>(special.machine_reg(), b.machine_reg());
  builder_.Gen<x86_64::PandnXRegXReg>(special.machine_reg(), mul_unord.machine_reg());
  // two_signed = ((a XOR b) AND signmask) OR bits(+2.0). Reuse a as (a XOR b),
  // mul_unord as the per-lane sign mask (all-ones << 31).
  builder_.Gen<x86_64::PxorXRegXReg>(a.machine_reg(), b.machine_reg());
  builder_.Gen<x86_64::PcmpeqdXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
  builder_.Gen<x86_64::PslldXRegImm>(mul_unord.machine_reg(), int8_t{31});
  builder_.Gen<x86_64::PandXRegXReg>(a.machine_reg(), mul_unord.machine_reg());
  // +2.0 broadcast to all 4 dwords (0x40000000 per dword).
  FpRegister two = AllocTempSimdReg();
  Register gr = GetImm(uint64_t{0x4000000040000000ULL});
  builder_.Gen<x86_64::MovqXRegReg>(two.machine_reg(), gr);
  builder_.Gen<x86_64::PunpcklqdqXRegXReg>(two.machine_reg(), two.machine_reg());
  builder_.Gen<x86_64::PorXRegXReg>(a.machine_reg(), two.machine_reg());  // a = +-2.0 per lane
  // result = (mul AND NOT special) OR (+-2.0 AND special).
  builder_.Gen<x86_64::PandXRegXReg>(a.machine_reg(), special.machine_reg());
  builder_.Gen<x86_64::PandnXRegXReg>(special.machine_reg(), mul.machine_reg());
  builder_.Gen<x86_64::PorXRegXReg>(special.machine_reg(), a.machine_reg());
  return special;
}

HeavyOptimizerFrontend::FpRegister HeavyOptimizerFrontend::EmitFpPairwiseMinMaxF32Packed(
    FpRegister xa, FpRegister xb, bool is_max, bool is_nm) {
  auto gen_min = [&](FpRegister d, FpRegister s) {
    builder_.Gen<x86_64::MinpsXRegXReg>(d.machine_reg(), s.machine_reg());
  };
  auto gen_max = [&](FpRegister d, FpRegister s) {
    builder_.Gen<x86_64::MaxpsXRegXReg>(d.machine_reg(), s.machine_reg());
  };
  auto gen_cmpeq0 = [&](FpRegister d, FpRegister z) {
    builder_.Gen<x86_64::CmpeqpsXRegXReg>(d.machine_reg(), z.machine_reg());
  };
  auto gen_cmpunord = [&](FpRegister d, FpRegister s) {
    builder_.Gen<x86_64::CmpunordpsXRegXReg>(d.machine_reg(), s.machine_reg());
  };
  // Corrective value for the +-0 tie: AND(a,b) for max, OR(a,b) for min.
  FpRegister corr = AllocTempSimdReg();
  builder_.Gen<x86_64::MovdqaXRegXReg>(corr.machine_reg(), xa.machine_reg());
  if (is_max) builder_.Gen<x86_64::PandXRegXReg>(corr.machine_reg(), xb.machine_reg());
  else        builder_.Gen<x86_64::PorXRegXReg>(corr.machine_reg(), xb.machine_reg());
  // "both zero" mask (IEEE -0 == +0, so this catches both sign forms).
  FpRegister zero = AllocZeroedSimdReg();
  FpRegister eqa = AllocTempSimdReg();
  FpRegister eqb = AllocTempSimdReg();
  builder_.Gen<x86_64::MovdqaXRegXReg>(eqa.machine_reg(), xa.machine_reg());
  gen_cmpeq0(eqa, zero);
  builder_.Gen<x86_64::MovdqaXRegXReg>(eqb.machine_reg(), xb.machine_reg());
  gen_cmpeq0(eqb, zero);
  builder_.Gen<x86_64::PandXRegXReg>(eqa.machine_reg(), eqb.machine_reg());
  if (!is_nm) {
    // NaN-propagating: tmp = b; (Max|Min)(tmp,a); (Max|Min)(a,b); POR(a,tmp).
    FpRegister tmp = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), xb.machine_reg());
    if (is_max) { gen_max(tmp, xa); gen_max(xa, xb); }
    else        { gen_min(tmp, xa); gen_min(xa, xb); }
    builder_.Gen<x86_64::PorXRegXReg>(xa.machine_reg(), tmp.machine_reg());
  } else {
    // NaN-suppressing: substitute each NaN lane with the other operand, then Max|Min.
    FpRegister ma = AllocTempSimdReg();
    FpRegister mb = AllocTempSimdReg();
    FpRegister sa = AllocTempSimdReg();
    FpRegister sb = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(ma.machine_reg(), xa.machine_reg());
    gen_cmpunord(ma, ma);
    builder_.Gen<x86_64::MovdqaXRegXReg>(mb.machine_reg(), xb.machine_reg());
    gen_cmpunord(mb, mb);
    builder_.Gen<x86_64::MovdqaXRegXReg>(sa.machine_reg(), ma.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(sa.machine_reg(), xb.machine_reg());   // ma & b
    builder_.Gen<x86_64::MovdqaXRegXReg>(sb.machine_reg(), mb.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(sb.machine_reg(), xa.machine_reg());   // mb & a
    builder_.Gen<x86_64::PandnXRegXReg>(ma.machine_reg(), xa.machine_reg());  // ~ma & a
    builder_.Gen<x86_64::PandnXRegXReg>(mb.machine_reg(), xb.machine_reg());  // ~mb & b
    builder_.Gen<x86_64::PorXRegXReg>(ma.machine_reg(), sa.machine_reg());    // a'
    builder_.Gen<x86_64::PorXRegXReg>(mb.machine_reg(), sb.machine_reg());    // b'
    if (is_max) gen_max(ma, mb); else gen_min(ma, mb);
    builder_.Gen<x86_64::MovdqaXRegXReg>(xa.machine_reg(), ma.machine_reg());
  }
  // +-0 blend: corr = (eqa & corr) | (~eqa & xa).
  builder_.Gen<x86_64::PandXRegXReg>(corr.machine_reg(), eqa.machine_reg());
  builder_.Gen<x86_64::PandnXRegXReg>(eqa.machine_reg(), xa.machine_reg());
  builder_.Gen<x86_64::PorXRegXReg>(corr.machine_reg(), eqa.machine_reg());
  return corr;
}

void HeavyOptimizerFrontend::AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::AdvSimdScalarPairwiseOpcode;
  // FP16 scalar pairwise FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP (H): reduce Vn.h[0]
  // and Vn.h[1] via the F16C round-trip. Widen both halves to FP32 lane 0 (upper
  // lanes +0.0), fold, narrow once. (kAddp is the integer D-form and never sets
  // is_fp16.) Bit-identical to the interpreter's per-precision reduce.
  if (args.is_fp16) {
    if (!host_platform::kHasF16C) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_faddp = (args.opcode == Op::kFaddpScalar);
    const bool is_max = (args.opcode == Op::kFmaxpScalar || args.opcode == Op::kFmaxnmpScalar);
    const bool is_min = (args.opcode == Op::kFminpScalar || args.opcode == Op::kFminnmpScalar);
    const bool is_nm = (args.opcode == Op::kFmaxnmpScalar || args.opcode == Op::kFminnmpScalar);
    if (!is_faddp && !is_max && !is_min) {
      UndefinedReturningVoid();
      return;
    }
    // Widen Vn.h[0] -> xa lane 0, Vn.h[1] -> xb lane 0 (upper lanes +0.0).
    FpRegister lo = no_fp_register, hi = no_fp_register;
    EmitWidenHalfVec(GetVRegOffset(args.rn), /*q=*/false, &lo, &hi);  // lo = [h0,h1,h2,h3]
    FpRegister xa = AllocZeroedSimdReg();
    builder_.Gen<x86_64::MovssXRegXReg>(xa.machine_reg(), lo.machine_reg());  // [h0,+0,+0,+0]
    FpRegister t = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), lo.machine_reg());
    builder_.Gen<x86_64::PshufdXRegXRegImm>(t.machine_reg(), t.machine_reg(), int8_t{0x01});
    FpRegister xb = AllocZeroedSimdReg();
    builder_.Gen<x86_64::MovssXRegXReg>(xb.machine_reg(), t.machine_reg());  // [h1,+0,+0,+0]
    FpRegister result;
    if (is_faddp) {
      builder_.Gen<x86_64::AddpsXRegXReg>(xa.machine_reg(), xb.machine_reg());
      result = xa;
    } else {
      result = EmitFpPairwiseMinMaxF32Packed(xa, xb, is_max, is_nm);
    }
    EmitNarrowF32ToHalfAndStore(args.rd, result);
    return;
  }
  const bool is_addp = (args.opcode == Op::kAddp);
  const bool is_faddp = (args.opcode == Op::kFaddpScalar);
  const bool is_fmax_family =
      (args.opcode == Op::kFmaxpScalar || args.opcode == Op::kFmaxnmpScalar);
  const bool is_fmin_family =
      (args.opcode == Op::kFminpScalar || args.opcode == Op::kFminnmpScalar);
  const bool is_nm =
      (args.opcode == Op::kFmaxnmpScalar || args.opcode == Op::kFminnmpScalar);
  if (!is_addp && !is_faddp && !is_fmax_family && !is_fmin_family) {
    UndefinedReturningVoid();
    return;
  }
  // ADDP is D-form (64-bit); the FP forms pick S (size bit0=0) / D (bit0=1).
  const bool is_double = is_addp || ((args.size & 1) != 0);
  FpRegister xa = AllocTempSimdReg();
  FpRegister xb = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xa.machine_reg(), GetVRegOffset(args.rn));
  // xb = Vn lane 1 brought into lane 0. D/ADDP: qword1 via PSHUFD 0xEE; S:
  // dword1 via PSHUFD 0x01. Only lane 0 of xa/xb is consumed; upper-lane
  // garbage is discarded by SetVRegScalar.
  builder_.Gen<x86_64::PshufdXRegXRegImm>(
      xb.machine_reg(), xa.machine_reg(),
      is_double ? static_cast<int8_t>(0xEEu) : int8_t{0x01});

  FpRegister result = xa;
  if (is_addp) {
    builder_.Gen<x86_64::PaddqXRegXReg>(xa.machine_reg(), xb.machine_reg());
  } else if (is_faddp) {
    if (is_double) {
      builder_.Gen<x86_64::AddpdXRegXReg>(xa.machine_reg(), xb.machine_reg());
    } else {
      builder_.Gen<x86_64::AddpsXRegXReg>(xa.machine_reg(), xb.machine_reg());
    }
  } else {
    // FMAXP/FMINP/FMAXNMP/FMINNMP scalar. Mirror of lite's ±0-correct,
    // NaN-handling min/max idiom (packed ops on lane 0; upper garbage
    // discarded). The ±0 tie is fixed by an explicit blend (AND for max, OR
    // for min) over lanes where both inputs are zero, computed BEFORE the
    // min/max clobbers xa.
    auto gen_min = [&](FpRegister d, FpRegister s) {
      if (is_double) builder_.Gen<x86_64::MinpdXRegXReg>(d.machine_reg(), s.machine_reg());
      else builder_.Gen<x86_64::MinpsXRegXReg>(d.machine_reg(), s.machine_reg());
    };
    auto gen_max = [&](FpRegister d, FpRegister s) {
      if (is_double) builder_.Gen<x86_64::MaxpdXRegXReg>(d.machine_reg(), s.machine_reg());
      else builder_.Gen<x86_64::MaxpsXRegXReg>(d.machine_reg(), s.machine_reg());
    };
    auto gen_cmpeq0 = [&](FpRegister d, FpRegister zero) {
      if (is_double) builder_.Gen<x86_64::CmpeqpdXRegXReg>(d.machine_reg(), zero.machine_reg());
      else builder_.Gen<x86_64::CmpeqpsXRegXReg>(d.machine_reg(), zero.machine_reg());
    };
    auto gen_cmpunord = [&](FpRegister d, FpRegister s) {
      if (is_double) builder_.Gen<x86_64::CmpunordpdXRegXReg>(d.machine_reg(), s.machine_reg());
      else builder_.Gen<x86_64::CmpunordpsXRegXReg>(d.machine_reg(), s.machine_reg());
    };
    // Corrective value for the ±0 tie: AND(a,b) for max, OR(a,b) for min.
    FpRegister corr = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(corr.machine_reg(), xa.machine_reg());
    if (is_fmax_family) builder_.Gen<x86_64::PandXRegXReg>(corr.machine_reg(), xb.machine_reg());
    else builder_.Gen<x86_64::PorXRegXReg>(corr.machine_reg(), xb.machine_reg());
    // "both zero" mask (IEEE -0 == +0, so this catches both sign forms).
    FpRegister zero = AllocZeroedSimdReg();
    FpRegister eqa = AllocTempSimdReg();
    FpRegister eqb = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(eqa.machine_reg(), xa.machine_reg());
    gen_cmpeq0(eqa, zero);
    builder_.Gen<x86_64::MovdqaXRegXReg>(eqb.machine_reg(), xb.machine_reg());
    gen_cmpeq0(eqb, zero);
    builder_.Gen<x86_64::PandXRegXReg>(eqa.machine_reg(), eqb.machine_reg());
    if (!is_nm) {
      // FMAXP/FMINP: NaN-propagating idiom. tmp = b; (Max|Min)(tmp,a);
      // (Max|Min)(a,b); POR(a,tmp) keeps NaN if either input was NaN.
      FpRegister tmp = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), xb.machine_reg());
      if (is_fmax_family) { gen_max(tmp, xa); gen_max(xa, xb); }
      else { gen_min(tmp, xa); gen_min(xa, xb); }
      builder_.Gen<x86_64::PorXRegXReg>(xa.machine_reg(), tmp.machine_reg());
    } else {
      // FMAXNMP/FMINNMP: substitute each NaN lane with the other operand,
      // then Max|Min. Both-NaN yields NaN (min/max returns src2, still NaN).
      FpRegister ma = AllocTempSimdReg();
      FpRegister mb = AllocTempSimdReg();
      FpRegister sa = AllocTempSimdReg();
      FpRegister sb = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(ma.machine_reg(), xa.machine_reg());
      gen_cmpunord(ma, ma);  // 1s where a is NaN
      builder_.Gen<x86_64::MovdqaXRegXReg>(mb.machine_reg(), xb.machine_reg());
      gen_cmpunord(mb, mb);  // 1s where b is NaN
      builder_.Gen<x86_64::MovdqaXRegXReg>(sa.machine_reg(), ma.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(sa.machine_reg(), xb.machine_reg());   // ma & b
      builder_.Gen<x86_64::MovdqaXRegXReg>(sb.machine_reg(), mb.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(sb.machine_reg(), xa.machine_reg());   // mb & a
      builder_.Gen<x86_64::PandnXRegXReg>(ma.machine_reg(), xa.machine_reg());  // ~ma & a
      builder_.Gen<x86_64::PandnXRegXReg>(mb.machine_reg(), xb.machine_reg());  // ~mb & b
      builder_.Gen<x86_64::PorXRegXReg>(ma.machine_reg(), sa.machine_reg());    // a'
      builder_.Gen<x86_64::PorXRegXReg>(mb.machine_reg(), sb.machine_reg());    // b'
      if (is_fmax_family) gen_max(ma, mb);
      else gen_min(ma, mb);
      builder_.Gen<x86_64::MovdqaXRegXReg>(xa.machine_reg(), ma.machine_reg());
    }
    // ±0 blend: xa = (eqa & corr) | (~eqa & xa).
    builder_.Gen<x86_64::PandXRegXReg>(corr.machine_reg(), eqa.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(eqa.machine_reg(), xa.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(corr.machine_reg(), eqa.machine_reg());
    result = corr;
  }
  SetVRegScalar(args.rd, result, is_double);
}

// AdvSIMD shift-by-immediate (vector and scalar).  Mirrors the high-value
// subset of lite_translator.h::AdvSimdShiftByImm into the optimizing tier so
// NEON widening/shift loops (calculate_gnu_hash_neon and every SIMD widening
// expansion) stop bailing the heavy region wholesale to lite.  Covered here:
//   * USHLL / SSHLL (incl. UXTL/SXTL == #0) — 8B->8H / 4H->4S / 2S->2D
//     widening, both Q halves (long2 reads Vn[127:64]).
//   * SHL / USHR / SSHR at H/S/D lanes (esize 16/32/64).
//   * SSRA / USRA (shift-accumulate) at H/S/D lanes (USRA all three; SSRA
//     esize 16/32 only — SSRA .2D needs PSRAQ and bails).
//   * SLI / SRI (shift-and-insert) at H/S/D lanes (all three; PSLL/PSRL mask
//     round trip + shifted Vn + POR).
//   * SRSHR / URSHR / SRSRA / URSRA (rounding right shift + accumulate) at
//     H/S/D lanes (URSHR/URSRA all three; SRSHR/SRSRA esize 16/32 only —
//     signed .2D needs PSRAQ and bails). Round bit via a PSRL/PSLL/PSRL
//     bit-bracket, PADD, then (accumulate) PADD into Vd.
//   * SQSHL / UQSHL / SQSHLU (saturating shift left) at vector H/S lanes
//     (all three; UQSHL/SQSHLU also .2D via PSLLQ/PSRLQ/PCMPEQQ; SQSHL .2D
//     needs PSRAQ recovery and bails). Shift-left, recover via the inverse
//     shift, PCMPEQ-vs-source for a per-lane no-overflow mask, blend the
//     shifted value with the per-lane saturation limit.
// The byte-lane (.8B/.16B) SHL/USHR/SSHR/SSRA/USRA/SLI/SRI and rounding
// SRSHR/URSHR/SRSRA/URSRA forms are all lowered here (via the PMOVSX/PMOVZX-
// widen + word-shift + PACK recipe, and PSLLW/PSRLW + per-byte mask for
// SLI/SRI).  Everything else — byte-lane SATURATING shifts (SQSHL/UQSHL/
// SQSHLU), the signed .2D forms (SSHR/SSRA/SRSHR/SRSRA/SQSHL .2D, which need
// PSRAQ), scalar saturating B/H/S/D, and narrow / fixed-point conversions —
// calls Undefined() which sets success_=false and bails the region to lite
// (the lite tier already lowers those correctly; a heavy bail is
// correct-but-slow, acceptable for the rarer shift variants).
void HeavyOptimizerFrontend::AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& args) {
  if (!success()) {
    return;
  }
  const uint8_t immh = args.immh;
  if (immh == 0) {
    UndefinedReturningVoid();
    return;
  }
  const int32_t vn_off =
      GetVRegOffset(args.rn);
  const uint16_t immh_immb = static_cast<uint16_t>((immh << 3) | args.immb);

  switch (args.opcode) {
    case Decoder::AdvSimdShiftImmOpcode::kUshll:
    case Decoder::AdvSimdShiftImmOpcode::kSshll: {
      // USHLL/SSHLL Vd.<wide>, Vn.<narrow>, #shift.  esize = 8 <<
      // highest-set-bit(immh); Q=1 ("long2") widens the upper 64 of Vn.
      // Left-shift is bit-identical signed/unsigned, so PSLL{W,D,Q} is
      // shared after the signed/unsigned widen.  Result is always a full
      // 128-bit register (SetVRegFull q=true).
      if (immh & 0b1000) {  // RESERVED for shift-left-long.
        UndefinedReturningVoid();
        return;
      }
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSshll);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (args.q) {
        // Bring Vn[127:64] into the low 64 so PMOVSX/PMOVZX widens it.
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
      }
      uint8_t shift_imm;
      if (immh & 0b0100) {  // 2S -> 2D
        shift_imm = static_cast<uint8_t>(immh_immb - 32);
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
        }
        if (shift_imm != 0) {
          builder_.Gen<x86_64::PsllqXRegImm>(xn.machine_reg(), static_cast<int8_t>(shift_imm));
        }
      } else if (immh & 0b0010) {  // 4H -> 4S
        shift_imm = static_cast<uint8_t>(immh_immb - 16);
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
        }
        if (shift_imm != 0) {
          builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), static_cast<int8_t>(shift_imm));
        }
      } else {  // immh & 0b0001: 8B -> 8H
        shift_imm = static_cast<uint8_t>(immh_immb - 8);
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
        }
        if (shift_imm != 0) {
          builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), static_cast<int8_t>(shift_imm));
        }
      }
      SetVRegFull(args.rd, xn, /*q=*/true);
      return;
    }
    case Decoder::AdvSimdShiftImmOpcode::kShl:
    case Decoder::AdvSimdShiftImmOpcode::kUshr:
    case Decoder::AdvSimdShiftImmOpcode::kSshr: {
      // esize from immh: bit3->64, bit2->32, bit1->16, bit0->8 (byte, no
      // x86 packed shift). SHL byte JITs here via PSLLW + per-byte AND
      // mask; SSHR/USHR byte still bail to lite (PMOVSX/PACKSS widen).
      if (immh == 0b0001) {  // byte lane
        if (args.opcode == Decoder::AdvSimdShiftImmOpcode::kShl) {
          // Byte SHL by n (n = immh:immb - 8, range 0..7). x86 has no
          // packed byte shift, so shift the 16-bit words left by n
          // (PSLLW), then AND each byte with (0xFF << n) & 0xFF to drop
          // the bits that spilled across the byte boundary from the
          // neighbouring low byte. Mirrors the lite byte-SHL lowering.
          const uint8_t n = static_cast<uint8_t>(immh_immb - 8);
          FpRegister xn = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
          builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), static_cast<int8_t>(n));
          const uint8_t mbyte = static_cast<uint8_t>((0xFFu << n) & 0xFFu);
          const uint64_t mword = 0x0101010101010101ULL * mbyte;
          FpRegister mask = AllocTempSimdReg();
          Register gm = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(mword)));
          builder_.Gen<x86_64::MovqXRegReg>(mask.machine_reg(), gm);
          // Broadcast the low 64-bit mask to the high 64 bits.
          builder_.Gen<x86_64::PshufdXRegXRegImm>(mask.machine_reg(), mask.machine_reg(), int8_t{0x44});
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), mask.machine_reg());
          SetVRegFull(args.rd, xn, args.q);
          return;
        }
        // SSHR/USHR byte lane: SSE has no packed byte shift. Widen each
        // 64-bit half of Vn to 16-bit words (PMOVSXBW sign-extend for
        // SSHR, PMOVZXBW zero-extend for USHR), shift the words
        // (PSRAW/PSRLW), then narrow back to bytes (PACKSSWB/PACKUSWB).
        // byte_cnt = 2*8 - immh:immb in [1, 8]; at cnt==8 PSRAW on a
        // sign-extended byte yields byte-wide sign-fill and PSRLW yields
        // 0, matching ARM SSHR/USHR at shift==esize. Mirrors the lite
        // byte-form lowering (lite_translator.h::AdvSimdShiftByImm). The
        // PACK of both halves places low-lane bytes in Vd[63:0] and
        // high-lane bytes in Vd[127:64] for .16B; for .8B (q=0)
        // SetVRegFull zeroes Vd[127:64]. Scalar byte SSHR/USHR is not
        // ARM-encoded (scalar forms exist only at D) -> bail to lite.
        if (args.scalar) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_signed_byte =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSshr);
        const int8_t byte_cnt = static_cast<int8_t>(16 - immh_immb);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xn_hi = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn_hi.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsrldqXRegImm>(xn_hi.machine_reg(), int8_t{8});
        if (is_signed_byte) {
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), byte_cnt);
          builder_.Gen<x86_64::PsrawXRegImm>(xn_hi.machine_reg(), byte_cnt);
          builder_.Gen<x86_64::PacksswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), byte_cnt);
          builder_.Gen<x86_64::PsrlwXRegImm>(xn_hi.machine_reg(), byte_cnt);
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        }
        SetVRegFull(args.rd, xn, args.q);
        return;
      }
      uint8_t esize_bits;
      if (immh & 0b1000) {
        esize_bits = 64;
      } else if (immh & 0b0100) {
        esize_bits = 32;
      } else {  // immh & 0b0010
        esize_bits = 16;
      }
      const bool is_left =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kShl);
      // SHL count = immh:immb - esize (0..esize-1); USHR/SSHR count =
      // 2*esize - immh:immb (1..esize).  PSLL/PSRL/PSRA saturate to
      // 0/sign-fill at count>=esize, matching ARM at the boundary.
      const uint8_t shift_count =
          is_left ? static_cast<uint8_t>(immh_immb - esize_bits)
                  : static_cast<uint8_t>(2 * esize_bits - immh_immb);
      const bool is_arith =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSshr);
      // SSHR .2D / scalar-D has no packed 64-bit arithmetic shift below
      // AVX-512 (PSRAQ). Emulate with the sign mask: sign = (0 > Vn) via
      // PCMPGTQ (SSE4.2), result = PSRLQ(Vn, n) | PSLLQ(sign, 64-n); at
      // n==64 the result is the sign mask itself (ARM's spec'd sign-fill).
      if (is_arith && esize_bits == 64) {
        if (!host_platform::kHasSSE4_2) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xv = AllocTempSimdReg();
        FpRegister sign = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xv.machine_reg(), vn_off);
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(sign.machine_reg(), xv.machine_reg());
        if (shift_count == 64) {
          SetVRegFull(args.rd, sign, args.q);
          return;
        }
        builder_.Gen<x86_64::PsrlqXRegImm>(xv.machine_reg(),
                                           static_cast<int8_t>(shift_count));
        builder_.Gen<x86_64::PsllqXRegImm>(sign.machine_reg(),
                                           static_cast<int8_t>(64 - shift_count));
        builder_.Gen<x86_64::PorXRegXReg>(xv.machine_reg(), sign.machine_reg());
        SetVRegFull(args.rd, xv, args.q);
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      const int8_t cnt = static_cast<int8_t>(shift_count);
      if (is_left) {
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), cnt);
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsllqXRegImm>(xn.machine_reg(), cnt);
        }
      } else if (is_arith) {  // SSHR, esize 16 or 32
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), cnt);
        }
      } else {  // USHR
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
        }
      }
      // Q=0 (incl. scalar) forms zero Vd[127:64] via SetVRegFull.
      SetVRegFull(args.rd, xn, args.q);
      return;
    }
    case Decoder::AdvSimdShiftImmOpcode::kSsra:
    case Decoder::AdvSimdShiftImmOpcode::kUsra: {
      // Shift-accumulate: Vd<i> += (Vn<i> >>{arith,logical} shift) per lane.
      // Mirrors lite_translator.h::AdvSimdShiftByImm SSRA/USRA (H/S/D lanes):
      // shift the source, then PADD the packed result into Vd's current value.
      // Byte lane needs PMOVSX/PMOVZX widening + PACK (no x86 packed byte
      // shift); SSRA .2D needs PSRAQ (AVX-512F-VL). Both bail to lite, which
      // ships a widening / GPR fallback (correct-but-slow).
      if (immh == 0b0001) {  // byte lane
        // Byte SSRA/USRA (.16B/.8B): no packed byte shift on x86. Widen each
        // 64-bit half of Vn to 16-bit words (PMOVSXBW for the arith SSRA,
        // PMOVZXBW for the logical USRA), shift the words (PSRAW/PSRLW), narrow
        // back to bytes (PACKSSWB/PACKUSWB — the arith-shifted bytes stay in
        // int8 range so no saturation), then PADDB the shifted bytes into Vd
        // (native mod-256 wrap). Mirror of lite byte SSRA/USRA. cnt = 16 -
        // immh:immb in [1, 8]; at cnt==8 PSRAW sign-fills / PSRLW zeroes the
        // widened byte, matching ARM at shift==esize. Scalar B is not
        // ARM-encoded; bail defensively.
        if (args.scalar) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_arith_byte =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSsra);
        const int8_t byte_cnt = static_cast<int8_t>(16 - immh_immb);  // [1, 8]
        FpRegister xn = AllocTempSimdReg();
        FpRegister xn_hi = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn_hi.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsrldqXRegImm>(xn_hi.machine_reg(), int8_t{8});
        if (is_arith_byte) {
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), byte_cnt);
          builder_.Gen<x86_64::PsrawXRegImm>(xn_hi.machine_reg(), byte_cnt);
          builder_.Gen<x86_64::PacksswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), byte_cnt);
          builder_.Gen<x86_64::PsrlwXRegImm>(xn_hi.machine_reg(), byte_cnt);
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        }
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), GetVRegOffset(args.rd));
        builder_.Gen<x86_64::PaddbXRegXReg>(xd.machine_reg(), xn.machine_reg());
        // Q=0 (.8B) zeroes Vd[127:64] via SetVRegFull's D-form merge.
        SetVRegFull(args.rd, xd, args.q);
        return;
      }
      uint8_t esize_bits;
      if (immh & 0b1000) {
        esize_bits = 64;
      } else if (immh & 0b0100) {
        esize_bits = 32;
      } else {  // immh & 0b0010
        esize_bits = 16;
      }
      const bool is_arith =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSsra);
      if (is_arith && esize_bits == 64) {  // SSRA .2D / scalar-D: PSRAQ only.
        UndefinedReturningVoid();
        return;
      }
      // SSRA/USRA count = 2*esize - immh:immb, range [1, esize]. At the
      // esize boundary PSRL saturates to 0 (USRA: add 0 -> Vd unchanged) and
      // PSRA sign-fills (SSRA: add 0/-1 per lane) — both match ARM.
      const uint8_t shift_count =
          static_cast<uint8_t>(2 * esize_bits - immh_immb);
      const int8_t cnt = static_cast<int8_t>(shift_count);
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // Load Vd's current value up front; rd==rn is safe because xn/xd are
      // independent temps and the writeback (SetVRegFull) happens last.
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      if (is_arith) {  // SSRA, esize 16 or 32.
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
          builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {  // 32
          builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), cnt);
          builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
      } else {  // USRA, esize 16/32/64.
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
          builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
          builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {  // 64
          builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
          builder_.Gen<x86_64::PaddqXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
      }
      // Q=0 (.4H/.2S, incl. scalar) zero Vd[127:64] via SetVRegFull's D-form
      // merge, discarding the accumulate's garbage in the upper lanes.
      SetVRegFull(args.rd, xd, args.q);
      return;
    }
    case Decoder::AdvSimdShiftImmOpcode::kSrshr:
    case Decoder::AdvSimdShiftImmOpcode::kUrshr:
    case Decoder::AdvSimdShiftImmOpcode::kSrsra:
    case Decoder::AdvSimdShiftImmOpcode::kUrsra: {
      // Rounding right shift (+ accumulate). Per lane:
      //   *RSHR Vd<i> = floor((Vn<i> + 2^(shift-1)) / 2^shift)
      //   *RSRA Vd<i> = Vd<i> + floor((Vn<i> + 2^(shift-1)) / 2^shift)
      // Mirrors lite_translator.h::AdvSimdShiftByImm rounding family (H/S/D
      // packed path): shift Vn right (arith for signed, logical for
      // unsigned) into xn, isolate bit (shift-1) of Vn as the +round term in
      // xr via a PSRL/PSLL/PSRL bit-bracket, PADD them, then (accumulate)
      // PADD into Vd. Byte lane (needs PMOVZXBW widening) and signed .2D
      // SRSHR/SRSRA (need PSRAQ, AVX-512F-VL only) bail to lite, which ships
      // a widening / GPR fallback (correct-but-slow).
      if (immh == 0b0001) {  // byte lane
        // Rounding right shift (+ accumulate) on bytes. No packed byte shift on
        // x86: widen each 64-bit half of Vn to 16-bit words (PMOVSXBW for the
        // signed SRSHR/SRSRA, PMOVZXBW for the unsigned URSHR/URSRA), compute
        // *RSHR(x,cnt) = (x >>{a,l} cnt) + ((x >> (cnt-1)) & 1) at word width,
        // and PACK{SS,US}WB back to bytes. For the accumulate forms (SRSRA/
        // URSRA) PADDB the rounded bytes into Vd (mod-256). cnt = 16 -
        // immh:immb in [1, 8]; at cnt==8 the shift sign-fills (signed) / zeroes
        // (unsigned) and the round bit is the MSB, matching ARM at
        // shift==esize. The signed arith-shifted byte stays in i8 range so
        // PACKSSWB never saturates. Mirror of the lite byte rounding forms.
        // Scalar B is not ARM-encoded (scalar forms exist only at D) -> bail.
        if (args.scalar) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_signed_byte =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrshr ||
             args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra);
        const bool is_accum_byte =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra ||
             args.opcode == Decoder::AdvSimdShiftImmOpcode::kUrsra);
        const int8_t cnt = static_cast<int8_t>(16 - immh_immb);        // [1, 8]
        const int8_t cnt_minus_1 = static_cast<int8_t>(cnt - 1);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xn_hi = AllocTempSimdReg();
        FpRegister round = AllocTempSimdReg();
        FpRegister round_hi = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn_hi.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsrldqXRegImm>(xn_hi.machine_reg(), int8_t{8});
        if (is_signed_byte) {
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
        }
        builder_.Gen<x86_64::MovdqaXRegXReg>(round.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(round_hi.machine_reg(), xn_hi.machine_reg());
        // Isolate bit (cnt-1) of each byte into bit 0 of each 16-bit word.
        builder_.Gen<x86_64::PsrlwXRegImm>(round.machine_reg(), cnt_minus_1);
        builder_.Gen<x86_64::PsrlwXRegImm>(round_hi.machine_reg(), cnt_minus_1);
        builder_.Gen<x86_64::PsllwXRegImm>(round.machine_reg(), int8_t{15});
        builder_.Gen<x86_64::PsllwXRegImm>(round_hi.machine_reg(), int8_t{15});
        builder_.Gen<x86_64::PsrlwXRegImm>(round.machine_reg(), int8_t{15});
        builder_.Gen<x86_64::PsrlwXRegImm>(round_hi.machine_reg(), int8_t{15});
        // Main shifted value: arith for signed, logical for unsigned.
        if (is_signed_byte) {
          builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
          builder_.Gen<x86_64::PsrawXRegImm>(xn_hi.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
          builder_.Gen<x86_64::PsrlwXRegImm>(xn_hi.machine_reg(), cnt);
        }
        builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), round.machine_reg());
        builder_.Gen<x86_64::PaddwXRegXReg>(xn_hi.machine_reg(), round_hi.machine_reg());
        if (is_signed_byte) {
          builder_.Gen<x86_64::PacksswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        } else {
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        }
        FpRegister result = xn;
        if (is_accum_byte) {
          FpRegister xd = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xd.machine_reg(), GetVRegOffset(args.rd));
          builder_.Gen<x86_64::PaddbXRegXReg>(xd.machine_reg(), xn.machine_reg());
          result = xd;
        }
        // Q=0 (.8B) zeroes Vd[127:64] via SetVRegFull's D-form merge.
        SetVRegFull(args.rd, result, args.q);
        return;
      }
      uint8_t esize_bits;
      if (immh & 0b1000) {
        esize_bits = 64;
      } else if (immh & 0b0100) {
        esize_bits = 32;
      } else {  // immh & 0b0010
        esize_bits = 16;
      }
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrshr ||
           args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra);
      const bool is_accumulate =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra ||
           args.opcode == Decoder::AdvSimdShiftImmOpcode::kUrsra);
      // Signed .2D needs PSRAQ (AVX-512F-VL) -> bail; lite's GPR fallback
      // ships each 64-bit lane through SARQ + round.
      if (is_signed && esize_bits == 64) {
        UndefinedReturningVoid();
        return;
      }
      // count = 2*esize - immh:immb, range [1, esize]. At count==esize PSRL
      // saturates to 0 and the round bit isolates the MSB, giving the
      // ARM-correct result (floor((x + 2^(esize-1)) / 2^esize) = MSB).
      const uint8_t shift_count =
          static_cast<uint8_t>(2 * esize_bits - immh_immb);
      const int8_t cnt = static_cast<int8_t>(shift_count);
      const int8_t cnt_minus_1 = static_cast<int8_t>(shift_count - 1);
      const int8_t esize_minus_1 = static_cast<int8_t>(esize_bits - 1);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xr = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xr.machine_reg(), vn_off);
      // Main shifted value in xn (arith for signed H/S, logical otherwise).
      if (esize_bits == 16) {
        if (is_signed) {
          builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
        }
      } else if (esize_bits == 32) {
        if (is_signed) {
          builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
        }
      } else {  // 64 (unsigned only; signed .2D bailed above)
        builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
      }
      // Round bit in xr: isolate bit (shift-1) of Vn into bit 0 per lane via
      // PSRL(cnt-1) then a PSLL(esize-1)/PSRL(esize-1) bit-0 bracket.
      if (esize_bits == 16) {
        builder_.Gen<x86_64::PsrlwXRegImm>(xr.machine_reg(), cnt_minus_1);
        builder_.Gen<x86_64::PsllwXRegImm>(xr.machine_reg(), esize_minus_1);
        builder_.Gen<x86_64::PsrlwXRegImm>(xr.machine_reg(), esize_minus_1);
      } else if (esize_bits == 32) {
        builder_.Gen<x86_64::PsrldXRegImm>(xr.machine_reg(), cnt_minus_1);
        builder_.Gen<x86_64::PslldXRegImm>(xr.machine_reg(), esize_minus_1);
        builder_.Gen<x86_64::PsrldXRegImm>(xr.machine_reg(), esize_minus_1);
      } else {  // 64
        builder_.Gen<x86_64::PsrlqXRegImm>(xr.machine_reg(), cnt_minus_1);
        builder_.Gen<x86_64::PsllqXRegImm>(xr.machine_reg(), esize_minus_1);
        builder_.Gen<x86_64::PsrlqXRegImm>(xr.machine_reg(), esize_minus_1);
      }
      // shifted + round bit.
      if (esize_bits == 16) {
        builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xr.machine_reg());
      } else if (esize_bits == 32) {
        builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xr.machine_reg());
      } else {
        builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xr.machine_reg());
      }
      FpRegister result = xn;
      if (is_accumulate) {
        FpRegister xd = AllocTempSimdReg();
        const int32_t vd_off =
            GetVRegOffset(args.rd);
        // Load orig Vd before the writeback; rd==rn safe (independent temps).
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PaddqXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
        result = xd;
      }
      // Q=0 (.4H/.2S, incl. scalar D) zero Vd[127:64] via SetVRegFull.
      SetVRegFull(args.rd, result, args.q);
      return;
    }
    case Decoder::AdvSimdShiftImmOpcode::kSli:
    case Decoder::AdvSimdShiftImmOpcode::kSri: {
      // Shift-and-insert. SLI: Vd<i> = (Vn<i> << shift) with Vd's low `shift`
      // bits preserved. SRI: Vd<i> = USHR(Vn<i>, shift) with Vd's high
      // `shift` bits preserved. Mirrors lite_translator.h::AdvSimdShiftByImm
      // SLI/SRI (H/S/D lanes): clear Vd's inserted bits with a PSLL+PSRL (SLI)
      // or PSRL+PSLL (SRI) round trip, shift Vn into place, POR the two.
      if (immh == 0b0001) {  // byte lane
        // No packed byte shift on x86.  Word-shift Vn (PSLLW for SLI / PSRLW
        // for SRI), keep the shifted bits of each byte with a broadcast mask
        // (PAND), and merge the preserved Vd field via PANDN + POR.  Mirror of
        // the lite byte SLI/SRI lowering.  Scalar B is not ARM-encoded; bail.
        if (args.scalar) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_sli_byte =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSli);
        // SLI n = immh:immb - 8 in [0,7]; SRI cnt = 16 - immh:immb in [1,8].
        const uint8_t sh = is_sli_byte
                               ? static_cast<uint8_t>(immh_immb - 8)
                               : static_cast<uint8_t>(16 - immh_immb);
        // Bits taken from the SHIFTED source per byte:
        //   SLI keeps bits [sh,7]     -> (0xFF << sh) & 0xFF
        //   SRI keeps bits [0,7-sh]   -> 0xFF >> sh   (0 at sh==8: Vd unchanged)
        const uint8_t mbyte =
            is_sli_byte ? static_cast<uint8_t>((0xFFu << sh) & 0xFFu)
                        : static_cast<uint8_t>(0xFFu >> sh);
        const uint64_t mword = 0x0101010101010101ULL * mbyte;
        FpRegister xn = AllocTempSimdReg();
        FpRegister xd = AllocTempSimdReg();
        FpRegister mask = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.GenGetSimd<16>(xd.machine_reg(), GetVRegOffset(args.rd));
        if (is_sli_byte) {
          if (sh != 0) {
            builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), static_cast<int8_t>(sh));
          }
        } else {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), static_cast<int8_t>(sh));
        }
        Register gm = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(mword)));
        builder_.Gen<x86_64::MovqXRegReg>(mask.machine_reg(), gm);
        // Broadcast the low 64-bit mask to the high 64 bits.
        builder_.Gen<x86_64::PshufdXRegXRegImm>(mask.machine_reg(), mask.machine_reg(), int8_t{0x44});
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), mask.machine_reg());   // shifted bits
        builder_.Gen<x86_64::PandnXRegXReg>(mask.machine_reg(), xd.machine_reg());  // ~mask & Vd
        builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), mask.machine_reg());
        // Q=0 (.8B) zeroes Vd[127:64] via SetVRegFull's D-form merge.
        SetVRegFull(args.rd, xn, args.q);
        return;
      }
      uint8_t esize_bits;
      if (immh & 0b1000) {
        esize_bits = 64;
      } else if (immh & 0b0100) {
        esize_bits = 32;
      } else {  // immh & 0b0010
        esize_bits = 16;
      }
      const bool is_sli = (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSli);
      // SLI count = immh:immb - esize, range [0, esize-1].
      // SRI count = 2*esize - immh:immb, range [1, esize].
      const uint8_t shift_count =
          is_sli ? static_cast<uint8_t>(immh_immb - esize_bits)
                 : static_cast<uint8_t>(2 * esize_bits - immh_immb);
      const int8_t cnt = static_cast<int8_t>(shift_count);
      // inv = esize - shift is the width of Vd's preserved field. At the
      // boundaries the PSLL/PSRL round trip collapses to the ARM-correct
      // extreme: SLI count 0 -> inv==esize clears all of Vd (POR yields Vn);
      // SRI count==esize -> inv==0 preserves all of Vd and USHR gives 0.
      const int8_t inv = static_cast<int8_t>(esize_bits - shift_count);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // Load Vd up front; rd==rn is safe because xn/xd are independent temps
      // and the writeback (SetVRegFull) happens last.
      const int32_t vd_off =
          GetVRegOffset(args.rd);
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      if (is_sli) {
        // Vd = (Vd keep low `shift` bits) | (Vn << shift).
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsllwXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsrlwXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), cnt);
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PslldXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsrldXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), cnt);
        } else {  // 64
          builder_.Gen<x86_64::PsllqXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsrlqXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsllqXRegImm>(xn.machine_reg(), cnt);
        }
      } else {
        // SRI: Vd = (Vd keep high `shift` bits) | USHR(Vn, shift).
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsllwXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PsrldXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PslldXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
        } else {  // 64
          builder_.Gen<x86_64::PsrlqXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsllqXRegImm>(xd.machine_reg(), inv);
          builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
        }
      }
      builder_.Gen<x86_64::PorXRegXReg>(xd.machine_reg(), xn.machine_reg());
      // Q=0 (.4H/.2S, incl. scalar D) zero Vd[127:64] via SetVRegFull.
      SetVRegFull(args.rd, xd, args.q);
      return;
    }
    case Decoder::AdvSimdShiftImmOpcode::kSqshl:
    case Decoder::AdvSimdShiftImmOpcode::kUqshl:
    case Decoder::AdvSimdShiftImmOpcode::kSqshlu: {
      // Saturating shift left by immediate. Per lane, shift Vn left by
      // `shift`; if the result overflows the element's saturating range,
      // replace it with the saturation limit. Mirrors
      // lite_translator.h::AdvSimdShiftByImm SQSHL/UQSHL/SQSHLU vector packed
      // pipeline (H/S lanes for all three; UQSHL/SQSHLU also .2D via
      // PSLLQ/PSRLQ/PCMPEQQ; SQSHL .2D needs PSRAQ recovery and bails).
      // Mechanism: shift-left into xs, recover with the inverse shift into xm,
      // PCMPEQ against the source to build a per-lane "no-overflow" mask,
      // blend the shifted value with the per-lane saturation target. SQSHLU
      // pre-zeros negative source lanes so the unsigned clamp yields the
      // architectural 0.
      //
      // Scalar B/H/S/D forms (args.scalar — saturating shift accepts any
      // non-zero immh) and byte-lane vector forms bail to lite, which has
      // full scalar + byte coverage (correct-but-slow).
      if (args.scalar || immh == 0b0001) {
        UndefinedReturningVoid();
        return;
      }
      uint8_t esize_bits;
      if (immh & 0b1000) {
        esize_bits = 64;
      } else if (immh & 0b0100) {
        esize_bits = 32;
      } else {  // immh & 0b0010
        esize_bits = 16;
      }
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshl);
      const bool is_sqshlu =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshlu);
      // SQSHL .2D needs PSRAQ (AVX-512F-VL) for the signed back-shift
      // recovery -> bail; lite ships each 64-bit lane through a GPR fallback.
      if (is_signed && esize_bits == 64) {
        UndefinedReturningVoid();
        return;
      }
      // SQSHL/UQSHL/SQSHLU count = immh:immb - esize, range [0, esize-1].
      const uint8_t shift_count =
          static_cast<uint8_t>(immh_immb - esize_bits);
      const int8_t cnt = static_cast<int8_t>(shift_count);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xs = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      // xt starts zeroed so its self-referencing PCMPEQ/PXOR idioms below
      // (all-ones / re-zero) read a defined register — the heavy tier's SSA
      // lifetime analysis rejects a use-before-def, unlike the lite tier's
      // physical registers.
      FpRegister xt = AllocZeroedSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);

      if (is_sqshlu) {
        // pos_xn = (Vn < 0) ? 0 : Vn. Pre-zero negative source lanes so the
        // unsigned clamp below yields the architectural 0. xt is already zero
        // (AllocZeroedSimdReg), so PCMPGT(xt, xn) yields the neg-lane mask.
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PcmpgtwXRegXReg>(xt.machine_reg(), xn.machine_reg());
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PcmpgtdXRegXReg>(xt.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(xt.machine_reg(), xn.machine_reg());
        }
        builder_.Gen<x86_64::PandnXRegXReg>(xt.machine_reg(), xn.machine_reg());  // ~neg & xn
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn.machine_reg(), xt.machine_reg());
      }

      // xs = xn << cnt.
      builder_.Gen<x86_64::MovdqaXRegXReg>(xs.machine_reg(), xn.machine_reg());
      if (esize_bits == 16) {
        builder_.Gen<x86_64::PsllwXRegImm>(xs.machine_reg(), cnt);
      } else if (esize_bits == 32) {
        builder_.Gen<x86_64::PslldXRegImm>(xs.machine_reg(), cnt);
      } else {
        builder_.Gen<x86_64::PsllqXRegImm>(xs.machine_reg(), cnt);
      }
      // xm = recover(xs): arithmetic (signed) / logical (unsigned) shift-right
      // by the same count. If it reproduces the source, the shift did not
      // overflow that lane.
      builder_.Gen<x86_64::MovdqaXRegXReg>(xm.machine_reg(), xs.machine_reg());
      if (is_signed) {  // SQSHL, esize 16/32 (esize 64 bailed above).
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrawXRegImm>(xm.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsradXRegImm>(xm.machine_reg(), cnt);
        }
      } else {  // UQSHL / SQSHLU, esize 16/32/64.
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), cnt);
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsrlqXRegImm>(xm.machine_reg(), cnt);
        }
      }
      // xm = eq_mask: per-lane all-ones where recover(shift) == source.
      if (esize_bits == 16) {
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm.machine_reg(), xn.machine_reg());
      } else if (esize_bits == 32) {
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::PcmpeqqXRegXReg>(xm.machine_reg(), xn.machine_reg());
      }

      // Saturation target in xt.
      if (is_signed) {
        // SQSHL: sat = INT_MAX XOR neg_mask(xn). (esize 16/32 only.)
        builder_.Gen<x86_64::PxorXRegXReg>(xt.machine_reg(), xt.machine_reg());
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PcmpgtwXRegXReg>(xt.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PcmpgtdXRegXReg>(xt.machine_reg(), xn.machine_reg());
        }
        // xn is now free (its value was consumed by the eq_mask + neg_mask);
        // reuse it to hold INT_MAX = PSRL(all-ones, 1).
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xn.machine_reg(), xn.machine_reg());
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{1});
        } else {
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{1});
        }
        builder_.Gen<x86_64::PxorXRegXReg>(xt.machine_reg(), xn.machine_reg());
      } else {
        // UQSHL / SQSHLU: sat = UINT_MAX (all-ones) per lane.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xt.machine_reg(), xt.machine_reg());
      }

      // Blend: result = (eq_mask & shifted) | (~eq_mask & sat).
      builder_.Gen<x86_64::PandXRegXReg>(xs.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(xm.machine_reg(), xt.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xs.machine_reg(), xm.machine_reg());
      // Q=0 (.4H/.2S) zero Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xs, args.q);
      return;
    }
    case Decoder::AdvSimdShiftImmOpcode::kShrn:
    case Decoder::AdvSimdShiftImmOpcode::kRshrn: {
      // SHRN/RSHRN Vd.<narrow>, Vn.<wide>, #shift — shift-right narrow.
      // Mirrors lite_translator.h::AdvSimdShiftByImm's non-saturating narrow
      // path: (RSHRN only) broadcast+add the per-lane rounding constant ->
      // logical right shift (PSRL{W,D,Q}) -> PSHUFB gather of each wide
      // lane's low half into the packed low 64. src_bits = 8<<highest-set-
      // bit(immh); dst = src/2. immh bit3 set is RESERVED (bail). SHRN/RSHRN
      // have no scalar form, so args.scalar is always false here. The
      // saturating narrows (SQSHRN/UQSHRN/SQRSHRN/UQRSHRN/SQSHRUN/SQRSHRUN)
      // still bail to lite via the default case below.
      if (immh & 0b1000) {  // RESERVED for narrowing shifts.
        UndefinedReturningVoid();
        return;
      }
      uint8_t src_bits;
      if (immh & 0b0100) {
        src_bits = 64;
      } else if (immh & 0b0010) {
        src_bits = 32;
      } else {  // immh == 0b0001
        src_bits = 16;
      }
      const uint8_t narrow_rshift = static_cast<uint8_t>(src_bits - immh_immb);
      const bool is_rounding =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kRshrn);
      const int8_t cnt = static_cast<int8_t>(narrow_rshift);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      if (is_rounding) {
        // Broadcast the per-lane rounding constant (1 << (rshift-1)) across
        // every src-width lane, then add. Wrap-on-overflow is benign — the
        // shift range [1, dst_bits] keeps the add's carry inside the low
        // half PSHUFB gathers; any wrap at bit src_bits is dropped anyway.
        const uint64_t round_lane = uint64_t{1} << (narrow_rshift - 1);
        uint64_t round_pattern;
        if (src_bits == 16) {
          round_pattern = round_lane * uint64_t{0x0001000100010001ULL};
        } else if (src_bits == 32) {
          round_pattern = round_lane * uint64_t{0x0000000100000001ULL};
        } else {
          round_pattern = round_lane;
        }
        FpRegister xround = AllocTempSimdReg();
        Register gr = std::get<0>(
            Gen<x86_64::MovqRegImm>(static_cast<int64_t>(round_pattern)));
        builder_.Gen<x86_64::MovqXRegReg>(xround.machine_reg(), gr);
        builder_.Gen<x86_64::PinsrqXRegRegImm>(xround.machine_reg(), gr, int8_t{1});
        if (src_bits == 16) {
          builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xround.machine_reg());
        } else if (src_bits == 32) {
          builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xround.machine_reg());
        } else {
          builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xround.machine_reg());
        }
      }
      // Logical right shift by cnt (in [1, dst_bits] <= src_bits/2), so the
      // high bits the narrow discards never reach the kept low half — SHRN's
      // untyped shift is bit-identical to a logical shift here.
      if (src_bits == 16) {
        builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
      } else if (src_bits == 32) {
        builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
      } else {
        builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
      }
      // PSHUFB narrow: the low-8 selector entries gather each wide lane's low
      // half into the packed low 64; the high-8 entries (0x80) zero the upper
      // 64.  src 16 -> low byte of each .8H lane; src 32 -> low half of each
      // .4S lane; src 64 -> low word of each .2D lane.
      int64_t mask_lo;
      if (src_bits == 16) {
        mask_lo = static_cast<int64_t>(0x0E0C0A0806040200LL);
      } else if (src_bits == 32) {
        mask_lo = static_cast<int64_t>(0x0D0C090805040100LL);
      } else {
        mask_lo = static_cast<int64_t>(0x0B0A090803020100LL);
      }
      FpRegister xmask = AllocTempSimdReg();
      Register gm = std::get<0>(Gen<x86_64::MovqRegImm>(mask_lo));
      builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), gm);
      Register gmhi = std::get<0>(
          Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0x8080808080808080ULL)));
      builder_.Gen<x86_64::PinsrqXRegRegImm>(xmask.machine_reg(), gmhi, int8_t{1});
      builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), xmask.machine_reg());
      if (!args.q) {
        // SHRN: narrowed lanes sit in the low 64; SetVRegFull zeroes
        // Vd[127:64] via its D-form merge.
        SetVRegFull(args.rd, xn, /*q=*/false);
      } else {
        // SHRN2: place the narrowed lanes in Vd[127:64], preserving Vd[63:0].
        const int32_t vd_off =
            GetVRegOffset(args.rd);
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        FpRegister xdlow = AllocZeroedSimdReg();
        builder_.Gen<x86_64::MovsdXRegXReg>(xdlow.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PorXRegXReg>(xdlow.machine_reg(), xn.machine_reg());
        builder_.GenSetSimd<16>(vd_off, xdlow.machine_reg());
      }
      return;
    }
    case Decoder::AdvSimdShiftImmOpcode::kSqshrn:
    case Decoder::AdvSimdShiftImmOpcode::kUqshrn:
    case Decoder::AdvSimdShiftImmOpcode::kSqshrun:
    case Decoder::AdvSimdShiftImmOpcode::kSqrshrn:
    case Decoder::AdvSimdShiftImmOpcode::kUqrshrn:
    case Decoder::AdvSimdShiftImmOpcode::kSqrshrun: {
      // (Rounding and non-rounding) saturating shift-right narrow (vector,
      // 16/32-bit source).  Mirrors lite_translator.h::AdvSimdShiftByImm's
      // saturating narrow path: (rounding only) add the per-lane rounding
      // bias -> shift each wide lane right (arithmetic for a signed source,
      // logical for unsigned) -> clamp to the destination range ->
      // PSHUFB-gather the low half of each lane into the packed low 64.
      //   SQSHRN/SQRSHRN   (signed->signed):     PSRA{W,D} + PMINS/PMAXS clamp.
      //   UQSHRN/UQRSHRN   (unsigned->unsigned): PSRL{W,D} + PMINU clamp.
      //   SQSHRUN/SQRSHRUN (signed->unsigned):   PSRA{W,D} + PMAXS-vs-0 + PMINS.
      // Rounding bias:
      //   SQRSHRN: signed-saturating pre-add of (1<<(shift-1)) — PADDSW for
      //     src16, PMINSD-preclamp+PADDD for src32 (no integer PADDSD).
      //   UQRSHRN: unsigned-saturating pre-add — PADDUSW for src16,
      //     PMINUD-preclamp+PADDD for src32.
      //   SQRSHRUN: the carry-bit identity (x+(1<<(s-1)))>>s == (x>>s) +
      //     bit(s-1 of x) for arithmetic >> — compute the carry pre-shift
      //     (PSLL then PSRL to bit 0), add it after the shift; avoids a
      //     premature signed saturation that would drop 1 LSB at the boundary.
      // src=64 needs PSRAQ (AVX-512F-VL) for the signed shift and a manual
      // hi32-nonzero rewrite for the unsigned clamp; UQRSHRN src=64 needs a
      // saturating PADDQ — all of which the lite tier also bails, so bail
      // here.  The scalar forms fall to lite via the bail below.
      if (immh & 0b1000) {  // RESERVED for narrowing shifts.
        UndefinedReturningVoid();
        return;
      }
      if (args.scalar) {  // Scalar saturating narrow -> lite handles it.
        UndefinedReturningVoid();
        return;
      }
      uint8_t src_bits;
      if (immh & 0b0100) {
        src_bits = 64;
      } else if (immh & 0b0010) {
        src_bits = 32;
      } else {  // immh == 0b0001
        src_bits = 16;
      }
      if (src_bits == 64) {  // needs PSRAQ / manual clamp -> bail to lite.
        UndefinedReturningVoid();
        return;
      }
      const bool is_rounding =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrn) ||
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqrshrn) ||
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrun);
      const bool is_saturating_signed =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshrn) ||
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrn);
      const bool is_saturating_unsigned =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqshrn) ||
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqrshrn);
      const bool is_signed_to_unsigned =
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshrun) ||
          (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrun);
      // Signed source (SQSHRN/SQSHRUN + rounding) uses an arithmetic right
      // shift; unsigned source (UQSHRN/UQRSHRN) uses a logical right shift.
      const bool uses_signed_shift =
          is_saturating_signed || is_signed_to_unsigned;
      const uint8_t narrow_rshift = static_cast<uint8_t>(src_bits - immh_immb);
      const int8_t cnt = static_cast<int8_t>(narrow_rshift);

      // Materialize a broadcast constant (`pattern` in both qwords) into a
      // fresh SIMD reg — the rounding bias and the clamp bounds need this.
      auto broadcast = [&](uint64_t pattern) -> FpRegister {
        FpRegister x = AllocTempSimdReg();
        Register gr =
            std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pattern)));
        builder_.Gen<x86_64::MovqXRegReg>(x.machine_reg(), gr);
        builder_.Gen<x86_64::PinsrqXRegRegImm>(x.machine_reg(), gr, int8_t{1});
        return x;
      };

      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);

      // xcarry holds the SQRSHRUN pre-shift rounding carry (0/1 per lane);
      // computed while xn is intact, added back after the arithmetic shift.
      FpRegister xcarry;
      if (is_rounding) {
        const uint64_t round_lane = uint64_t{1} << (narrow_rshift - 1);
        const uint64_t round_pattern =
            (src_bits == 16) ? round_lane * uint64_t{0x0001000100010001ULL}
                             : round_lane * uint64_t{0x0000000100000001ULL};
        if (is_saturating_unsigned) {
          FpRegister xround = broadcast(round_pattern);
          if (src_bits == 16) {
            // PADDUSW: unsigned-saturating word add (SSE2).
            builder_.Gen<x86_64::PadduswXRegXReg>(xn.machine_reg(),
                                                  xround.machine_reg());
          } else {
            // No unsigned-saturating PADDUSD in baseline SSE; pre-clamp xn to
            // (0xFFFFFFFF - round_lane) so the plain PADDD cannot overflow
            // past UINT32_MAX. Lanes hitting the pre-clamp settle at exactly
            // UINT32_MAX, which the post-shift PMINUD then narrows correctly.
            const uint32_t clamp_lane =
                0xFFFFFFFFu - static_cast<uint32_t>(round_lane);
            const uint64_t clamp_pattern =
                (uint64_t{clamp_lane} << 32) | uint64_t{clamp_lane};
            FpRegister xclamp = broadcast(clamp_pattern);
            builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(),
                                                 xclamp.machine_reg());
            builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(),
                                                xround.machine_reg());
          }
        } else if (is_signed_to_unsigned) {
          // SQRSHRUN carry-bit identity: isolate bit (cnt-1) of each lane into
          // bit 0 (shift it to the top, then logically back). No wide add ->
          // no premature saturation.
          xcarry = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(xcarry.machine_reg(),
                                               xn.machine_reg());
          if (src_bits == 16) {
            builder_.Gen<x86_64::PsllwXRegImm>(
                xcarry.machine_reg(), static_cast<int8_t>(16 - narrow_rshift));
            builder_.Gen<x86_64::PsrlwXRegImm>(xcarry.machine_reg(), int8_t{15});
          } else {
            builder_.Gen<x86_64::PslldXRegImm>(
                xcarry.machine_reg(), static_cast<int8_t>(32 - narrow_rshift));
            builder_.Gen<x86_64::PsrldXRegImm>(xcarry.machine_reg(), int8_t{31});
          }
        } else {  // SQRSHRN (signed->signed).
          FpRegister xround = broadcast(round_pattern);
          if (src_bits == 16) {
            // PADDSW: signed-saturating word add (SSE2).
            builder_.Gen<x86_64::PaddswXRegXReg>(xn.machine_reg(),
                                                 xround.machine_reg());
          } else {
            // No integer PADDSD in baseline SSE; pre-clamp xn to
            // (0x7FFFFFFF - round_lane) so the plain PADDD cannot overflow
            // positively. The round constant is positive, so no negative
            // underflow is possible. Lanes hitting the pre-clamp settle at
            // exactly INT32_MAX after the add, which the post-shift
            // PMINSD-vs-signed-max then drives to the saturated dst value.
            const uint32_t clamp_lane =
                0x7FFFFFFFu - static_cast<uint32_t>(round_lane);
            const uint64_t clamp_pattern =
                (uint64_t{clamp_lane} << 32) | uint64_t{clamp_lane};
            FpRegister xclamp = broadcast(clamp_pattern);
            builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(),
                                                 xclamp.machine_reg());
            builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(),
                                                xround.machine_reg());
          }
        }
      }

      if (uses_signed_shift) {
        if (src_bits == 16) {
          builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), cnt);
        }
      } else {
        if (src_bits == 16) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
        }
      }

      if (is_rounding && is_signed_to_unsigned) {
        // SQRSHRUN: add the pre-shift rounding carry. Post-shift magnitudes
        // are small (|x>>cnt| <= INT*_MAX>>1), so a plain PADD cannot overflow
        // before the unsigned clamp below.
        if (src_bits == 16) {
          builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(),
                                              xcarry.machine_reg());
        } else {
          builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(),
                                              xcarry.machine_reg());
        }
      }

      if (is_saturating_signed) {
        // Clamp each src-lane to the signed dst range:
        //   dst 8 : [-128, 127]   as signed 16 = [0xFF80, 0x007F].
        //   dst 16: [-32768,32767] as signed 32 = [0xFFFF8000, 0x00007FFF].
        uint64_t sat_max_pattern, sat_min_pattern;
        if (src_bits == 16) {
          sat_max_pattern = 0x007F007F007F007FULL;
          sat_min_pattern = 0xFF80FF80FF80FF80ULL;
        } else {
          sat_max_pattern = 0x00007FFF00007FFFULL;
          sat_min_pattern = 0xFFFF8000FFFF8000ULL;
        }
        FpRegister xsatmax = broadcast(sat_max_pattern);
        FpRegister xsatmin = broadcast(sat_min_pattern);
        if (src_bits == 16) {
          builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xsatmax.machine_reg());
          builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xsatmin.machine_reg());
        } else {
          builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xsatmax.machine_reg());
          builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xsatmin.machine_reg());
        }
      } else if (is_signed_to_unsigned) {
        // SQSHRUN: clamp the post-shift signed value to the unsigned dst
        // range [0, 2^dst_bits - 1].  PMAXS-vs-zero pins negatives at 0;
        // after that every lane is non-negative, so a signed PMINS against
        // the positive dst-max is equivalent to unsigned-min.
        const uint64_t sat_max_pattern =
            (src_bits == 16) ? 0x00FF00FF00FF00FFULL   // dst 8: 0xFF
                             : 0x0000FFFF0000FFFFULL;  // dst 16: 0xFFFF
        FpRegister xsatmax = broadcast(sat_max_pattern);
        FpRegister xzero = AllocZeroedSimdReg();
        if (src_bits == 16) {
          builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xzero.machine_reg());
          builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xsatmax.machine_reg());
        } else {
          builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xzero.machine_reg());
          builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xsatmax.machine_reg());
        }
      } else {  // UQSHRN: unsigned-min clamp to (1 << dst_bits) - 1.
        const uint64_t sat_pattern =
            (src_bits == 16) ? 0x00FF00FF00FF00FFULL
                             : 0x0000FFFF0000FFFFULL;
        FpRegister xsat = broadcast(sat_pattern);
        if (src_bits == 16) {
          builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xsat.machine_reg());
        } else {
          builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xsat.machine_reg());
        }
      }

      // PSHUFB narrow gather: the low-8 selector entries gather each wide
      // lane's low half into the packed low 64; the high-8 entries (0x80)
      // zero the upper 64.  Same layout as the SHRN/RSHRN path above.
      const int64_t mask_lo =
          (src_bits == 16) ? static_cast<int64_t>(0x0E0C0A0806040200LL)
                           : static_cast<int64_t>(0x0D0C090805040100LL);
      FpRegister xmask = AllocTempSimdReg();
      Register gm = std::get<0>(Gen<x86_64::MovqRegImm>(mask_lo));
      builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), gm);
      Register gmhi = std::get<0>(
          Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0x8080808080808080ULL)));
      builder_.Gen<x86_64::PinsrqXRegRegImm>(xmask.machine_reg(), gmhi, int8_t{1});
      builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), xmask.machine_reg());
      if (!args.q) {
        // Q=0: narrowed lanes in the low 64; SetVRegFull zeroes Vd[127:64].
        SetVRegFull(args.rd, xn, /*q=*/false);
      } else {
        // Q=1 ("...2" form): place narrowed lanes in Vd[127:64], preserve
        // Vd[63:0].
        const int32_t vd_off =
            GetVRegOffset(args.rd);
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        FpRegister xdlow = AllocZeroedSimdReg();
        builder_.Gen<x86_64::MovsdXRegXReg>(xdlow.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PorXRegXReg>(xdlow.machine_reg(), xn.machine_reg());
        builder_.GenSetSimd<16>(vd_off, xdlow.machine_reg());
      }
      return;
    }
    default:
      UndefinedReturningVoid();
      return;
  }
}

// Heavy-tier mirror of lite_translator.h::AdvSimdVecXIndexedElement's
// integer MUL/MLA/MLS by-element block (halfword .4h/.8h size=01, word
// .2s/.4s size=10):
//   MUL: Vd = Vn * broadcast(Vm.lane[index])
//   MLA: Vd = Vd + Vn * broadcast(Vm.lane[index])
//   MLS: Vd = Vd - Vn * broadcast(Vm.lane[index])
// No saturation, no widening — the bottom esize bits of each host product
// match the architectural result modulo 2^esize. Broadcast is Pshuflw+Pshufd
// (halfword; Psrldq 8 first if index>=4) or Pshufd (word). Q=0 upper-64
// discard via SetVRegFull(rd, res, q). All other by-element opcodes still
// bail to the lite tier — emit NOTHING before a bail.
void HeavyOptimizerFrontend::AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::AdvSimdVecXIdxOpcode;

  // SQDMULL/SQDMLAL/SQDMLSL by element (size=01: .4h/.8h -> .4s; size=10:
  // .2s/.4s -> .2d). Signed saturating doubling widening multiply, and the
  // accumulate/subtract siblings. Heavy-tier mirror of lite_translator.h's
  // saturating-doubling by-element arms, structured exactly like the
  // AdvSimdThreeDiff SQDMULL heavy arm above; the only differences are
  //   (a) Vm is the broadcast of Vm.lane[index], not an element-wise source;
  //   (b) Q ("2") selects the upper half of Vn ONLY (Vm is index-selected,
  //       so it is not "2"-shifted).
  //   doubled = SignedSat(2 * sext(Vn.sel[i]) * sext(Vm.lane[index]))
  //   SQDMULL : Vd[i]  = doubled
  //   SQDMLAL : Vd[i]  = SignedSat(Vd[i] + doubled)
  //   SQDMLSL : Vd[i]  = SignedSat(Vd[i] - doubled)
  // The 16x16 (32x32) product fits exactly in int32 (int64), so
  // PMULLD/PMOVSXWD (PMULDQ/PMOVSXDQ) reconstruct it losslessly. The lone
  // saturation corner is Vn.lane == Vm.lane == INT_MIN, where the product is
  // exactly 2^30 (2^62); detect it by PCMPEQ against that product and blend
  // SMAX over the corner lanes. Widening forms fill 128 bits regardless of Q
  // -> SetVRegFull q=true. The accumulate stage reuses the signed-saturating
  // add/sub (a^P)&(a^res) sign-bit overflow idiom from the three-diff arm.
  //
  // Verified encodings (aarch64-linux-gnu-as -march=armv8.2-a):
  //   sqdmull  v0.4s, v1.4h, v2.h[0] = 0x0F42B020
  //   sqdmull2 v0.4s, v1.8h, v2.h[7] = 0x4F72B820
  //   sqdmlal  v0.4s, v1.4h, v2.h[1] = 0x0F523020
  //   sqdmlsl2 v0.4s, v1.8h, v2.h[5] = 0x4F527820
  //   sqdmull  v0.2d, v1.2s, v2.s[0] = 0x0F82B020
  //   sqdmlal2 v0.2d, v1.4s, v2.s[3] = 0x4FA23820
  if (args.opcode == Op::kSqdmullIdx || args.opcode == Op::kSqdmlalIdx ||
      args.opcode == Op::kSqdmlslIdx) {
    if (args.size != 0b01 && args.size != 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_acc = (args.opcode != Op::kSqdmullIdx);
    const bool is_sub = (args.opcode == Op::kSqdmlslIdx);
    const int32_t vn_o =
        GetVRegOffset(args.rn);
    const int32_t vm_o =
        GetVRegOffset(args.rm);
    const int32_t vd_o =
        GetVRegOffset(args.rd);

    // Materialize a broadcast constant (`pattern` in both qwords).
    auto broadcast = [&](uint64_t pattern) -> FpRegister {
      FpRegister x = AllocTempSimdReg();
      Register gr =
          std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pattern)));
      builder_.Gen<x86_64::MovqXRegReg>(x.machine_reg(), gr);
      builder_.Gen<x86_64::PinsrqXRegRegImm>(x.machine_reg(), gr, int8_t{1});
      return x;
    };

    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_o);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_o);
    // Q=1 ("2") selects Vn's upper 64 bits; Vm stays index-selected.
    if (args.q) {
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
    }
    // Broadcast Vm.lane[index] across every source lane.
    if (args.size == 0b01) {
      if (args.index >= 4) {
        builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
      }
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      builder_.Gen<x86_64::PshuflwXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), int8_t{0x44});
    } else {  // size == 0b10
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
    }

    // Compute the doubled, saturated product P into xP.
    FpRegister xP = broadcast(args.size == 0b01 ? uint64_t{0x4000000040000000ULL}
                                                : uint64_t{0x4000000000000000ULL});
    if (args.size == 0b01) {
      builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(xP.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xn.machine_reg());  // double
      FpRegister xsat = broadcast(uint64_t{0x7FFFFFFF7FFFFFFFULL});
      builder_.Gen<x86_64::PandXRegXReg>(xsat.machine_reg(), xP.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(xP.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xP.machine_reg(), xsat.machine_reg());
    } else {  // size == 0b10
      builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
      // xm is already index-broadcast (all dwords equal); PMULDQ reads dword
      // positions 0 and 2, so no widen of xm is needed.
      builder_.Gen<x86_64::PmuldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PcmpeqqXRegXReg>(xP.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xn.machine_reg());  // double
      FpRegister xsat = broadcast(uint64_t{0x7FFFFFFFFFFFFFFFULL});
      builder_.Gen<x86_64::PandXRegXReg>(xsat.machine_reg(), xP.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(xP.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xP.machine_reg(), xsat.machine_reg());
    }

    if (!is_acc) {
      SetVRegFull(args.rd, xP, /*q=*/true);
      return;
    }

    // Signed saturating accumulate: result = SignedSat(Vd +/- P). a = Vd.
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_o);
    FpRegister xres = AllocTempSimdReg();
    FpRegister xof = AllocTempSimdReg();
    FpRegister xtmp = AllocTempSimdReg();
    if (args.size == 0b01) {
      if (is_sub) {
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), xP.machine_reg());   // diff = a - P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xP.machine_reg());     // a ^ P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // a ^ diff
        builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
      } else {
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PadddXRegXReg>(xres.machine_reg(), xP.machine_reg());   // sum = a + P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xres.machine_reg());   // a ^ sum
        builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xP.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // P ^ sum
        builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
      }
      builder_.Gen<x86_64::PsradXRegImm>(xof.machine_reg(), int8_t{31});  // overflow lanes
      builder_.Gen<x86_64::PsradXRegImm>(xd.machine_reg(), int8_t{31});   // a's sign (0 / -1)
      FpRegister xmaxc = broadcast(uint64_t{0x7FFFFFFF7FFFFFFFULL});
      builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), xmaxc.machine_reg());   // sat = (a>>31)^INT32_MAX
      builder_.Gen<x86_64::PandXRegXReg>(xd.machine_reg(), xof.machine_reg());     // sat in overflow lanes
      builder_.Gen<x86_64::PandnXRegXReg>(xof.machine_reg(), xres.machine_reg());  // result in non-overflow
      builder_.Gen<x86_64::PorXRegXReg>(xd.machine_reg(), xof.machine_reg());
      SetVRegFull(args.rd, xd, /*q=*/true);
      return;
    }
    // size == 0b10 accumulate (64-bit signed saturating).
    FpRegister xzero = AllocZeroedSimdReg();
    if (is_sub) {
      builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PsubqXRegXReg>(xres.machine_reg(), xP.machine_reg());   // diff = a - P
      builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xP.machine_reg());     // a ^ P
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // a ^ diff
      builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
    } else {
      builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xP.machine_reg());   // sum = a + P
      builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xres.machine_reg());   // a ^ sum
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xP.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // P ^ sum
      builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
    }
    // overflow_mask (xtmp) = all-ones per qword where xof < 0 (sign set).
    builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xzero.machine_reg());
    builder_.Gen<x86_64::PcmpgtqXRegXReg>(xtmp.machine_reg(), xof.machine_reg());
    // sat = sign(a) ^ INT64_MAX  (INT64_MIN if a<0, else INT64_MAX).
    builder_.Gen<x86_64::PcmpgtqXRegXReg>(xzero.machine_reg(), xd.machine_reg());  // all-ones where a<0
    FpRegister xmaxc = broadcast(uint64_t{0x7FFFFFFFFFFFFFFFULL});
    builder_.Gen<x86_64::PxorXRegXReg>(xzero.machine_reg(), xmaxc.machine_reg());  // xzero = sat value
    builder_.Gen<x86_64::PandXRegXReg>(xzero.machine_reg(), xtmp.machine_reg());   // sat in overflow lanes
    builder_.Gen<x86_64::PandnXRegXReg>(xtmp.machine_reg(), xres.machine_reg());   // result in non-overflow
    builder_.Gen<x86_64::PorXRegXReg>(xzero.machine_reg(), xtmp.machine_reg());
    SetVRegFull(args.rd, xzero, /*q=*/true);
    return;
  }

  // Widening MUL/MAC by element: SMULL/UMULL/SMLAL/UMLAL/SMLSL/UMLSL.
  // Heavy-tier mirror of lite_translator.h's widening by-element arms
  // (size=01: .4h/.8h -> .4s; size=10: .2s/.4s -> .2d). The destination is
  // always 128-bit regardless of Q — widening forms do NOT zero the upper
  // half, so SetVRegFull(rd, res, /*q=*/true) stores the full register.
  //   *MULL : Vd  = widen(Vn.selected) * widen(broadcast(Vm.lane[index]))
  //   *MLAL : Vd += widen(Vn.selected) * widen(broadcast(Vm.lane[index]))
  //   *MLSL : Vd -= widen(Vn.selected) * widen(broadcast(Vm.lane[index]))
  // Q=0 selects the low source half (Vn bytes 0..7); Q=1 selects the high
  // half (Vn bytes 8..15) — brought into the low quad by PSRLDQ 8 before the
  // register-form PMOVSX/PMOVZX widen (the lite tier uses a memory-operand
  // PMOVSX at vn_off+(q?8:0); heavy has no such op, so shift-then-widen).
  // Both signed 16x16 and unsigned 16x16 products fit in 32 bits, so PMULLD's
  // signed low-32 result is correct for both forms at size=01; size=10 uses
  // PMULDQ (signed) / PMULUDQ (unsigned) for the 32x32 -> 64 products.
  //
  // Verified encodings (ARM ARM C7.2):
  //   smull  v0.4s, v1.4h, v2.h[0] = 0x0F42A020
  //   umull  v0.4s, v1.4h, v2.h[0] = 0x2F42A020
  //   smlal  v0.4s, v1.4h, v2.h[0] = 0x0F422020
  //   smlsl  v0.4s, v1.4h, v2.h[0] = 0x0F426020
  //   smull  v0.2d, v1.2s, v2.s[0] = 0x0F82A020
  //   umlal  v0.2d, v1.2s, v2.s[0] = 0x2F822020
  if (args.opcode == Op::kSmullIdx || args.opcode == Op::kUmullIdx ||
      args.opcode == Op::kSmlalIdx || args.opcode == Op::kUmlalIdx ||
      args.opcode == Op::kSmlslIdx || args.opcode == Op::kUmlslIdx) {
    if (args.size != 0b01 && args.size != 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool w_signed = (args.opcode == Op::kSmullIdx ||
                           args.opcode == Op::kSmlalIdx ||
                           args.opcode == Op::kSmlslIdx);
    const bool w_accum = (args.opcode == Op::kSmlalIdx ||
                          args.opcode == Op::kUmlalIdx ||
                          args.opcode == Op::kSmlslIdx ||
                          args.opcode == Op::kUmlslIdx);
    const bool w_sub = (args.opcode == Op::kSmlslIdx ||
                        args.opcode == Op::kUmlslIdx);

    const int32_t w_vn_off = GetVRegOffset(args.rn);
    const int32_t w_vm_off = GetVRegOffset(args.rm);
    const int32_t w_vd_off = GetVRegOffset(args.rd);

    FpRegister wm = AllocTempSimdReg();
    FpRegister wn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(wm.machine_reg(), w_vm_off);
    builder_.GenGetSimd<16>(wn.machine_reg(), w_vn_off);
    // Bring Vn's selected source half (Q=1 -> bytes 8..15) into the low quad.
    if (args.q) {
      builder_.Gen<x86_64::PsrldqXRegImm>(wn.machine_reg(), int8_t{8});
    }

    if (args.size == 0b01) {
      // Broadcast Vm.h[index] across all 8 halfword lanes.
      if (args.index >= 4) {
        builder_.Gen<x86_64::PsrldqXRegImm>(wm.machine_reg(), int8_t{8});
      }
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      builder_.Gen<x86_64::PshuflwXRegXRegImm>(wm.machine_reg(), wm.machine_reg(), imm);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(wm.machine_reg(), wm.machine_reg(), int8_t{0x44});
      // Widen low 4 halfwords of each source to 4 x 32-bit lanes.
      if (w_signed) {
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(wm.machine_reg(), wm.machine_reg());
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(wn.machine_reg(), wn.machine_reg());
      } else {
        builder_.Gen<x86_64::PmovzxwdXRegXReg>(wm.machine_reg(), wm.machine_reg());
        builder_.Gen<x86_64::PmovzxwdXRegXReg>(wn.machine_reg(), wn.machine_reg());
      }
      builder_.Gen<x86_64::PmulldXRegXReg>(wn.machine_reg(), wm.machine_reg());
      FpRegister w_result = wn;
      if (w_accum) {
        FpRegister wd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(wd.machine_reg(), w_vd_off);
        if (w_sub) {
          builder_.Gen<x86_64::PsubdXRegXReg>(wd.machine_reg(), wn.machine_reg());
        } else {
          builder_.Gen<x86_64::PadddXRegXReg>(wd.machine_reg(), wn.machine_reg());
        }
        w_result = wd;
      }
      SetVRegFull(args.rd, w_result, /*q=*/true);
      return;
    }

    // size=10: word sources -> .2d. Broadcast Vm.s[index] across all 4 dword
    // lanes; PMULDQ/PMULUDQ read dword positions 0 and 2, so the broadcast
    // suffices (no widen of Vm). Widen Vn's low 2 dwords to 2 qword lanes.
    {
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(wm.machine_reg(), wm.machine_reg(), imm);
    }
    if (w_signed) {
      builder_.Gen<x86_64::PmovsxdqXRegXReg>(wn.machine_reg(), wn.machine_reg());
      builder_.Gen<x86_64::PmuldqXRegXReg>(wn.machine_reg(), wm.machine_reg());
    } else {
      builder_.Gen<x86_64::PmovzxdqXRegXReg>(wn.machine_reg(), wn.machine_reg());
      builder_.Gen<x86_64::PmuludqXRegXReg>(wn.machine_reg(), wm.machine_reg());
    }
    FpRegister w_result = wn;
    if (w_accum) {
      FpRegister wd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(wd.machine_reg(), w_vd_off);
      if (w_sub) {
        builder_.Gen<x86_64::PsubqXRegXReg>(wd.machine_reg(), wn.machine_reg());
      } else {
        builder_.Gen<x86_64::PaddqXRegXReg>(wd.machine_reg(), wn.machine_reg());
      }
      w_result = wd;
    }
    SetVRegFull(args.rd, w_result, /*q=*/true);
    return;
  }

  // FP by-element FMUL/FMLA/FMLS (.2S/.4S FP32 size=10, .2D FP64 size=11).
  // Heavy-tier mirror of lite_translator.h's vector FP by-element FP32/FP64
  // arm (the FMUL/FMLA/FMLS half of it): broadcast Vm.lane[index] across all
  // lanes with PSHUFD, then packed multiply (FMUL) or packed fused
  // multiply-accumulate (FMLA/FMLS, single rounding via VFMADD231P/VFNMADD231P
  // -- the same 231 RMW form and one-rounding guarantee as the three-same
  // FMLA/FMLS heavy arm above).
  //   FMUL:  Vd = Vn * broadcast(Vm.lane[index])
  //   FMULX: FMUL except (+/-0 * +/-inf) lanes return +/-2.0 (sign =
  //          sign(a) XOR sign(b)) instead of NaN -- the saturation blend
  //          lifted from lite_translator.h below.
  //   FMLA: Vd = Vd + Vn * broadcast(Vm.lane[index])   (fused, one rounding)
  //   FMLS: Vd = Vd - Vn * broadcast(Vm.lane[index])   (fused, one rounding)
  // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge. FP16
  // (size=00, an F16C round-trip absent from the heavy Gen inputs) still
  // bails to lite -- emit NOTHING before the bail. FMLA/FMLS require host
  // FMA; bail if absent (FMULX does not).
  //
  // Verified encodings (aarch64-linux-gnu-as -march=armv8.2-a):
  //   fmul  v0.2s, v1.2s, v2.s[0] = 0x0F829020
  //   fmul  v0.4s, v1.4s, v2.s[3] = 0x4FA29820
  //   fmla  v0.4s, v1.4s, v2.s[1] = 0x4FA21020
  //   fmls  v0.2d, v1.2d, v2.d[1] = 0x4FC25820
  //   fmla  v5.2d, v5.2d, v3.d[0] = 0x4FC310A5 (rd==rn)
  //   fmulx v0.2s, v1.2s, v2.s[0] = 0x2F829020 (U=1)
  //   fmulx v0.2d, v1.2d, v2.d[1] = 0x6FC29820
  if (args.opcode == Op::kFmul || args.opcode == Op::kFmulx ||
      args.opcode == Op::kFmla || args.opcode == Op::kFmls) {
    if (args.size != 0b10 && args.size != 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const bool fp_is_double = (args.size == 0b11);
    const bool needs_fma = (args.opcode == Op::kFmla || args.opcode == Op::kFmls);
    if (needs_fma && !host_platform::kHasFMA) {
      UndefinedReturningVoid();
      return;
    }
    const int32_t fp_vn_off =
        GetVRegOffset(args.rn);
    const int32_t fp_vm_off =
        GetVRegOffset(args.rm);
    const int32_t fp_vd_off =
        GetVRegOffset(args.rd);

    FpRegister fxn = AllocTempSimdReg();
    FpRegister fxm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(fxn.machine_reg(), fp_vn_off);
    builder_.GenGetSimd<16>(fxm.machine_reg(), fp_vm_off);

    // Broadcast Vm.lane[index] across all lanes.
    if (fp_is_double) {
      // FP64: index 0 -> 0x44 (copies low qword), index 1 -> 0xEE (high qword).
      const int8_t imm =
          (args.index == 0) ? int8_t{0x44} : int8_t{static_cast<int8_t>(0xEEu)};
      builder_.Gen<x86_64::PshufdXRegXRegImm>(fxm.machine_reg(), fxm.machine_reg(), imm);
    } else {
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(fxm.machine_reg(), fxm.machine_reg(), imm);
    }

    FpRegister fp_result = fxn;
    if (args.opcode == Op::kFmul) {
      if (fp_is_double) {
        builder_.Gen<x86_64::MulpdXRegXReg>(fxn.machine_reg(), fxm.machine_reg());
      } else {
        builder_.Gen<x86_64::MulpsXRegXReg>(fxn.machine_reg(), fxm.machine_reg());
      }
    } else if (args.opcode == Op::kFmulx) {
      // FMULX = FMUL except (+/-0 * +/-inf) lanes return +/-2.0 (sign =
      // sign(a) XOR sign(broadcast_b)).  Line-by-line lift of
      // lite_translator.h's by-element FMULX saturation blend on top of the
      // already-broadcast Vm lane (fxm):
      //   mul         = a * broadcast_b
      //   mul_unord   = cmpunord(mul, mul)        (-1/lane if mul is NaN)
      //   input_unord = cmpunord(a, broadcast_b)  (-1/lane if a or b NaN)
      //   special     = mul_unord AND NOT input_unord  (exactly +-0*+-inf)
      //   two_signed  = ((a XOR broadcast_b) AND sign_mask) OR bits(+2.0)
      //   result      = (mul AND NOT special) OR (two_signed AND special)
      FpRegister mul = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(mul.machine_reg(), fxn.machine_reg());
      if (fp_is_double) {
        builder_.Gen<x86_64::MulpdXRegXReg>(mul.machine_reg(), fxm.machine_reg());
      } else {
        builder_.Gen<x86_64::MulpsXRegXReg>(mul.machine_reg(), fxm.machine_reg());
      }
      // mul_unord = cmpunord(mul, mul).
      FpRegister mul_unord = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(mul_unord.machine_reg(), mul.machine_reg());
      if (fp_is_double) {
        builder_.Gen<x86_64::CmpunordpdXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
      } else {
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
      }
      // special = input_unord = cmpunord(a, broadcast_b), then
      // special = mul_unord AND NOT input_unord (Pandn writes (NOT dst) AND src).
      FpRegister special = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(special.machine_reg(), fxn.machine_reg());
      if (fp_is_double) {
        builder_.Gen<x86_64::CmpunordpdXRegXReg>(special.machine_reg(), fxm.machine_reg());
      } else {
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(special.machine_reg(), fxm.machine_reg());
      }
      builder_.Gen<x86_64::PandnXRegXReg>(special.machine_reg(), mul_unord.machine_reg());
      // two_signed: reuse fxn as (a XOR broadcast_b); reuse mul_unord as the
      // per-lane sign mask (all-ones << 31/63).  Bitwise XOR == XORPS/XORPD.
      builder_.Gen<x86_64::PxorXRegXReg>(fxn.machine_reg(), fxm.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
      if (fp_is_double) {
        builder_.Gen<x86_64::PsllqXRegImm>(mul_unord.machine_reg(), int8_t{63});
      } else {
        builder_.Gen<x86_64::PslldXRegImm>(mul_unord.machine_reg(), int8_t{31});
      }
      builder_.Gen<x86_64::PandXRegXReg>(fxn.machine_reg(), mul_unord.machine_reg());
      // Broadcast bits of +2.0 into all lanes (FP64: 0x4000000000000000 per
      // qword; FP32: 0x40000000 per dword = 0x4000000040000000 per qword).
      FpRegister two = AllocTempSimdReg();
      {
        const uint64_t pat = fp_is_double ? uint64_t{0x4000000000000000ULL}
                                          : uint64_t{0x4000000040000000ULL};
        Register gr =
            std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pat)));
        builder_.Gen<x86_64::MovqXRegReg>(two.machine_reg(), gr);
        builder_.Gen<x86_64::PinsrqXRegRegImm>(two.machine_reg(), gr, int8_t{1});
      }
      builder_.Gen<x86_64::PorXRegXReg>(fxn.machine_reg(), two.machine_reg());
      // fxn now holds +/-2.0 per lane.  Blend:
      //   result = (mul AND NOT special) OR (+/-2.0 AND special).
      builder_.Gen<x86_64::PandXRegXReg>(fxn.machine_reg(), special.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(special.machine_reg(), mul.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(special.machine_reg(), fxn.machine_reg());
      fp_result = special;
    } else {
      FpRegister fxd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(fxd.machine_reg(), fp_vd_off);
      if (args.opcode == Op::kFmla) {
        if (fp_is_double) {
          builder_.Gen<x86_64::Vfmadd231pdXRegXRegXReg>(
              fxd.machine_reg(), fxn.machine_reg(), fxm.machine_reg());
        } else {
          builder_.Gen<x86_64::Vfmadd231psXRegXRegXReg>(
              fxd.machine_reg(), fxn.machine_reg(), fxm.machine_reg());
        }
      } else {
        // FMLS: Vd = Vd + (-Vn)*Vm -- single fused rounding.
        if (fp_is_double) {
          builder_.Gen<x86_64::Vfnmadd231pdXRegXRegXReg>(
              fxd.machine_reg(), fxn.machine_reg(), fxm.machine_reg());
        } else {
          builder_.Gen<x86_64::Vfnmadd231psXRegXRegXReg>(
              fxd.machine_reg(), fxn.machine_reg(), fxm.machine_reg());
        }
      }
      fp_result = fxd;
    }
    SetVRegFull(args.rd, fp_result, args.q);
    return;
  }

  if (args.opcode != Op::kMul && args.opcode != Op::kMla &&
      args.opcode != Op::kMls) {
    UndefinedReturningVoid();
    return;
  }
  if (args.size != 0b01 && args.size != 0b10) {
    UndefinedReturningVoid();
    return;
  }
  const bool is_halfword = (args.size == 0b01);

  const int32_t vn_off = GetVRegOffset(args.rn);
  const int32_t vm_off = GetVRegOffset(args.rm);
  const int32_t vd_off = GetVRegOffset(args.rd);

  FpRegister xn = AllocTempSimdReg();
  FpRegister xm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
  builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);

  // Broadcast Vm.lane[index] across every destination lane.
  if (is_halfword) {
    if (args.index >= 4) {
      builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
    }
    const uint8_t i = args.index & 0b11;
    const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
    builder_.Gen<x86_64::PshuflwXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), int8_t{0x44});
  } else {
    const uint8_t i = args.index & 0b11;
    const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
  }

  FpRegister result = xn;
  if (args.opcode == Op::kMul) {
    if (is_halfword) {
      builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
    } else {
      builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
    }
  } else {
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
    if (is_halfword) {
      builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
    } else {
      builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
    }
    if (args.opcode == Op::kMla) {
      if (is_halfword) {
        builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
      }
    } else {
      if (is_halfword) {
        builder_.Gen<x86_64::PsubwXRegXReg>(xd.machine_reg(), xn.machine_reg());
      } else {
        builder_.Gen<x86_64::PsubdXRegXReg>(xd.machine_reg(), xn.machine_reg());
      }
    }
    result = xd;
  }

  SetVRegFull(args.rd, result, args.q);
}

void HeavyOptimizerFrontend::AdvSimdScalarXIndexedElement(const Decoder::AdvSimdScalarXIdxArgs& args) {
  if (!success()) {
    return;
  }
  using Op = Decoder::AdvSimdScalarXIdxOpcode;
  // Scalar by-element FP FMUL/FMULX/FMLA/FMLS (size=10 FP32, size=11 FP64).
  // Heavy-tier mirror of the vector FP by-element arm restricted to lane 0:
  // broadcast Vm.lane[index], multiply / fused multiply-accumulate on lane 0,
  // and commit via SetVRegScalar (zeroing Vd above the result lane). FP16
  // (size=00, F16C round-trip) and the saturating scalar-by-element forms
  // (SQDMULH/SQRDMULH/SQDMULL/SQRDMLAH/SQRDMLSH) bail to lite — emit NOTHING
  // before a bail. FMLA/FMLS require host FMA (FMULX does not).
  if (args.opcode != Op::kFmul && args.opcode != Op::kFmulx &&
      args.opcode != Op::kFmla && args.opcode != Op::kFmls) {
    UndefinedReturningVoid();
    return;
  }
  if (args.size != 0b10 && args.size != 0b11) {
    UndefinedReturningVoid();
    return;
  }
  const bool is_double = (args.size == 0b11);
  const bool needs_fma = (args.opcode == Op::kFmla || args.opcode == Op::kFmls);
  if (needs_fma && !host_platform::kHasFMA) {
    UndefinedReturningVoid();
    return;
  }
  FpRegister xn = AllocTempSimdReg();
  FpRegister xm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xn.machine_reg(), GetVRegOffset(args.rn));
  builder_.GenGetSimd<16>(xm.machine_reg(), GetVRegOffset(args.rm));
  // Broadcast Vm.lane[index] across all lanes.
  if (is_double) {
    const int8_t imm =
        (args.index == 0) ? int8_t{0x44} : int8_t{static_cast<int8_t>(0xEEu)};
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
  } else {
    const uint8_t i = args.index & 0b11;
    const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
  }
  FpRegister result = xn;
  if (args.opcode == Op::kFmul) {
    if (is_double) builder_.Gen<x86_64::MulpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
    else builder_.Gen<x86_64::MulpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
  } else if (args.opcode == Op::kFmulx) {
    // FMULX = FMUL except (+/-0 * +/-inf) lanes return +/-2.0 (sign =
    // sign(a) XOR sign(b)). Line-by-line lift of the vector by-element FMULX
    // saturation blend on top of the already-broadcast Vm lane (xm).
    FpRegister mul = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(mul.machine_reg(), xn.machine_reg());
    if (is_double) builder_.Gen<x86_64::MulpdXRegXReg>(mul.machine_reg(), xm.machine_reg());
    else builder_.Gen<x86_64::MulpsXRegXReg>(mul.machine_reg(), xm.machine_reg());
    FpRegister mul_unord = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(mul_unord.machine_reg(), mul.machine_reg());
    if (is_double) builder_.Gen<x86_64::CmpunordpdXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
    else builder_.Gen<x86_64::CmpunordpsXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
    FpRegister special = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(special.machine_reg(), xn.machine_reg());
    if (is_double) builder_.Gen<x86_64::CmpunordpdXRegXReg>(special.machine_reg(), xm.machine_reg());
    else builder_.Gen<x86_64::CmpunordpsXRegXReg>(special.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(special.machine_reg(), mul_unord.machine_reg());
    // two_signed: reuse xn as (a XOR b); reuse mul_unord as the sign mask.
    builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(mul_unord.machine_reg(), mul_unord.machine_reg());
    if (is_double) builder_.Gen<x86_64::PsllqXRegImm>(mul_unord.machine_reg(), int8_t{63});
    else builder_.Gen<x86_64::PslldXRegImm>(mul_unord.machine_reg(), int8_t{31});
    builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), mul_unord.machine_reg());
    FpRegister two = AllocTempSimdReg();
    {
      const uint64_t pat = is_double ? uint64_t{0x4000000000000000ULL}
                                     : uint64_t{0x4000000040000000ULL};
      Register gr = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pat)));
      builder_.Gen<x86_64::MovqXRegReg>(two.machine_reg(), gr);
      builder_.Gen<x86_64::PinsrqXRegRegImm>(two.machine_reg(), gr, int8_t{1});
    }
    builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), two.machine_reg());
    // Blend: result = (mul AND NOT special) OR (+/-2.0 AND special).
    builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), special.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(special.machine_reg(), mul.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(special.machine_reg(), xn.machine_reg());
    result = special;
  } else {
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), GetVRegOffset(args.rd));
    if (args.opcode == Op::kFmla) {
      if (is_double) builder_.Gen<x86_64::Vfmadd231pdXRegXRegXReg>(xd.machine_reg(), xn.machine_reg(), xm.machine_reg());
      else builder_.Gen<x86_64::Vfmadd231psXRegXRegXReg>(xd.machine_reg(), xn.machine_reg(), xm.machine_reg());
    } else {  // FMLS: Vd = Vd + (-Vn)*Vm, single fused rounding.
      if (is_double) builder_.Gen<x86_64::Vfnmadd231pdXRegXRegXReg>(xd.machine_reg(), xn.machine_reg(), xm.machine_reg());
      else builder_.Gen<x86_64::Vfnmadd231psXRegXRegXReg>(xd.machine_reg(), xn.machine_reg(), xm.machine_reg());
    }
    result = xd;
  }
  SetVRegScalar(args.rd, result, is_double);
}

// EXT Vd.<T>, Vn.<T>, Vm.<T>, #index — extract a vector from the byte-wise
// concatenation Vm:Vn (Vn is the low half) starting at byte `index`. Mirrors
// lite_translator.h::AdvSimdExtract: for the 8-byte form (Q=0) PUNPCKLQDQ
// packs Vn.low64 into bytes 0..7 and Vm.low64 into bytes 8..15 so a single
// PSRLDQ walks a contiguous 16-byte window; for the 16-byte form (Q=1) the
// window is (Vn >> index) | (Vm << (16-index)) via PSRLDQ/PSLLDQ + POR. The
// index-out-of-range cases (imm4[3]=1 for Q=0, index>=16 for Q=1) are
// UNDEFINED and bail, matching the lite tier.
void HeavyOptimizerFrontend::AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
  if (!success()) {
    return;
  }
  const int32_t vn_off = GetVRegOffset(rn);
  const int32_t vm_off = GetVRegOffset(rm);
  if (!q) {
    // 64-bit form: 8-byte window, index 0..7 (imm4[3] set is UNDEFINED).
    if (index >= 8) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    // Concatenate Vn.low64 (bytes 0..7) with Vm.low64 (bytes 8..15); the
    // upper 64 of Vn/Vm never participate in the Q=0 window.
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
    if (index != 0) {
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), static_cast<int8_t>(index));
    }
    // The wanted 8-byte window sits in the low 64 bits; SetVRegFull(q=false)
    // zero-extends the upper 64.
    SetVRegFull(rd, xn, /*q=*/false);
    return;
  }
  // 128-bit form: 16-byte window, index 0..15.
  FpRegister xn = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
  if (index == 0) {
    SetVRegFull(rd, xn, /*q=*/true);
    return;
  }
  if (index >= 16) {
    UndefinedReturningVoid();
    return;
  }
  FpRegister xm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
  // result = (Vn >> index bytes) | (Vm << (16-index) bytes).
  builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), static_cast<int8_t>(index));
  builder_.Gen<x86_64::PslldqXRegImm>(xm.machine_reg(), static_cast<int8_t>(16 - index));
  builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xm.machine_reg());
  SetVRegFull(rd, xn, /*q=*/true);
}

// ZIP1/ZIP2, UZP1/UZP2, TRN1/TRN2 vector permute — heavy-tier mirror of
// lite_translator.h::AdvSimdPermute. Register-domain-pure, straight-line SSE:
//   ZIP  -> PUNPCKL/H{bw,wd,dq,qdq} (interleave lower/upper halves)
//   UZP  -> PACKUSWB/PACKUSDW (byte/halfword) or SHUFPS (word); .2D == ZIP.2D
//   TRN  -> PSHUFB + POR with materialized per-byte masks (byte/halfword) or
//           PSHUFD + PUNPCKLDQ (word); .2D == ZIP.2D
// opcode (3 bits, from Decoder::DecodeAdvSimd): 001=UZP1 010=TRN1 011=ZIP1
// 101=UZP2 110=TRN2 111=ZIP2 (same encoding the lite tier consumes).
// SetVRegFull(q=false) zeroes Vd[127:64], so no manual upper-zero tail is
// needed (unlike the lite path). Q=0 .2D is reserved and bails; unknown
// opcodes bail. The emulator host always has SSE4.1 (matching the sibling
// USHLL/widening handlers, which use PACKUSDW/PMOVZX unguarded), so the
// UZP .8H PACKUSDW form is not host-gated here.
void HeavyOptimizerFrontend::AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size, uint8_t opcode, bool q) {
  if (!success()) {
    return;
  }
  const bool is_zip = (opcode == 0b011 || opcode == 0b111);
  const bool is_uzp = (opcode == 0b001 || opcode == 0b101);
  const bool is_trn = (opcode == 0b010 || opcode == 0b110);
  if (!is_zip && !is_uzp && !is_trn) {
    UndefinedReturningVoid();
    return;
  }
  if (size > 0b11) {
    UndefinedReturningVoid();
    return;
  }
  // Q=0 .2D is reserved by the ARM ARM (encoding restricted).
  if (!q && size == 0b11) {
    UndefinedReturningVoid();
    return;
  }
  const int32_t vn_off = GetVRegOffset(rn);
  const int32_t vm_off = GetVRegOffset(rm);
  FpRegister xn = AllocTempSimdReg();
  FpRegister xm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
  builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);

  if (is_zip) {
    const bool is_zip2 = (opcode == 0b111);
    if (!q && is_zip2) {
      // Shift each source right by 4 bytes so the (originally upper-half)
      // bytes 4..7 land at positions 0..3; PUNPCKL then picks them up.
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{4});
      builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{4});
    }
    if (q && is_zip2) {
      switch (size) {
        case 0b00: builder_.Gen<x86_64::PunpckhbwXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
        case 0b01: builder_.Gen<x86_64::PunpckhwdXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
        case 0b10: builder_.Gen<x86_64::PunpckhdqXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
        case 0b11: builder_.Gen<x86_64::PunpckhqdqXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
      }
    } else {
      switch (size) {
        case 0b00: builder_.Gen<x86_64::PunpcklbwXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
        case 0b01: builder_.Gen<x86_64::PunpcklwdXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
        case 0b10: builder_.Gen<x86_64::PunpckldqXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
        case 0b11: builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
      }
    }
  } else if (is_uzp) {
    const bool is_uzp2 = (opcode == 0b101);
    if (!q) {
      // Combine the lower 8 bytes of Vn and Vm into xn = [vn_lo | vm_lo] so
      // subsequent PACKUS/SHUFPS consumes a single source; the Q=0 tail then
      // discards the duplicated high half.
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
    }
    switch (size) {
      case 0b00: {  // .16B (q=1) or .8B (q=0). PACKUSWB.
        if (is_uzp2) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{8});
        } else {
          builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{8});
        }
        if (q) {
          if (is_uzp2) {
            builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});
          } else {
            builder_.Gen<x86_64::PsllwXRegImm>(xm.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});
          }
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn.machine_reg());
        }
        break;
      }
      case 0b01: {  // .8H (q=1) or .4H (q=0). PACKUSDW (SSE4.1).
        if (is_uzp2) {
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{16});
        } else {
          builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), int8_t{16});
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{16});
        }
        if (q) {
          if (is_uzp2) {
            builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{16});
          } else {
            builder_.Gen<x86_64::PslldXRegImm>(xm.machine_reg(), int8_t{16});
            builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{16});
          }
          builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xn.machine_reg());
        }
        break;
      }
      case 0b10: {  // .4S (q=1) or .2S (q=0). SHUFPS.
        const int8_t imm =
            is_uzp2 ? static_cast<int8_t>(0xDD) : static_cast<int8_t>(0x88);
        if (q) {
          builder_.Gen<x86_64::ShufpsXRegXRegImm>(xn.machine_reg(), xm.machine_reg(), imm);
        } else {
          // xn already holds [vn_lo | vm_lo] from PUNPCKLQDQ above.
          builder_.Gen<x86_64::ShufpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), imm);
        }
        break;
      }
      case 0b11: {  // .2D (q=1 only; q=0 caught above).
        // UZP1.2D == ZIP1.2D, UZP2.2D == ZIP2.2D (2-lane coincidence).
        if (is_uzp2) {
          builder_.Gen<x86_64::PunpckhqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
      }
    }
  } else {  // is_trn
    const bool is_trn2 = (opcode == 0b110);
    switch (size) {
      case 0b00:
      case 0b01: {
        // .16B/.8B and .8H/.4H use PSHUFB + POR with precomputed per-byte
        // masks (0x80 mask byte renders as 0 in PSHUFB). mask_n picks the
        // wanted Vn bytes into even output positions; mask_m picks Vm bytes
        // into odd positions; POR combines. Q=0 forms reuse the Q=1 masks —
        // the shared Q=0 upper-zero (SetVRegFull) discards bytes 8..15.
        int64_t mask_n_lo, mask_n_hi, mask_m_lo, mask_m_hi;
        if (size == 0b00) {
          if (is_trn2) {
            mask_n_lo = static_cast<int64_t>(0x8007800580038001LL);
            mask_n_hi = static_cast<int64_t>(0x800F800D800B8009LL);
            mask_m_lo = static_cast<int64_t>(0x0780058003800180LL);
            mask_m_hi = static_cast<int64_t>(0x0F800D800B800980LL);
          } else {
            mask_n_lo = static_cast<int64_t>(0x8006800480028000LL);
            mask_n_hi = static_cast<int64_t>(0x800E800C800A8008LL);
            mask_m_lo = static_cast<int64_t>(0x0680048002800080LL);
            mask_m_hi = static_cast<int64_t>(0x0E800C800A800880LL);
          }
        } else {  // size == 0b01: halfword granularity.
          if (is_trn2) {
            mask_n_lo = static_cast<int64_t>(0x8080070680800302LL);
            mask_n_hi = static_cast<int64_t>(0x80800F0E80800B0ALL);
            mask_m_lo = static_cast<int64_t>(0x0706808003028080LL);
            mask_m_hi = static_cast<int64_t>(0x0F0E80800B0A8080LL);
          } else {
            mask_n_lo = static_cast<int64_t>(0x8080050480800100LL);
            mask_n_hi = static_cast<int64_t>(0x80800D0C80800908LL);
            mask_m_lo = static_cast<int64_t>(0x0504808001008080LL);
            mask_m_hi = static_cast<int64_t>(0x0D0C808009088080LL);
          }
        }
        FpRegister mask_n = AllocTempSimdReg();
        FpRegister mask_m = AllocTempSimdReg();
        builder_.Gen<x86_64::MovqXRegReg>(mask_n.machine_reg(),
                                          GetImm(static_cast<uint64_t>(mask_n_lo)));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(mask_n.machine_reg(),
                                               GetImm(static_cast<uint64_t>(mask_n_hi)), int8_t{1});
        builder_.Gen<x86_64::MovqXRegReg>(mask_m.machine_reg(),
                                          GetImm(static_cast<uint64_t>(mask_m_lo)));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(mask_m.machine_reg(),
                                               GetImm(static_cast<uint64_t>(mask_m_hi)), int8_t{1});
        builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), mask_n.machine_reg());
        builder_.Gen<x86_64::PshufbXRegXReg>(xm.machine_reg(), mask_m.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      }
      case 0b10: {  // .4S (q=1) or .2S (q=0).
        if (!q) {
          // TRN1.2S == ZIP1.2S = [s1[0], s2[0]]; TRN2.2S == ZIP2.2S =
          // [s1[1], s2[1]] (PSRLDQ-4 prelude pulls lane 1 down to 0).
          if (is_trn2) {
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{4});
            builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{4});
          }
          builder_.Gen<x86_64::PunpckldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          // PSHUFD imm 0x88 (even) / 0xDD (odd) collapses each source's
          // wanted dwords into both halves; PUNPCKLDQ interleaves the lows:
          //   res = [s1[0], s2[0], s1[2], s2[2]] = TRN1.4S (0x88), etc.
          const int8_t imm = is_trn2 ? static_cast<int8_t>(0xDD)
                                     : static_cast<int8_t>(0x88);
          builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), imm);
          builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
          builder_.Gen<x86_64::PunpckldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
      }
      case 0b11: {  // .2D (q=1 only; q=0 caught above).
        // TRN1.2D == ZIP1.2D, TRN2.2D == ZIP2.2D (2-lane coincidence).
        if (is_trn2) {
          builder_.Gen<x86_64::PunpckhqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
      }
    }
  }

  SetVRegFull(rd, xn, q);
}

void HeavyOptimizerFrontend::AdvSimdTableLookup(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t len, uint8_t op, bool q) {
  if (!success()) {
    return;
  }
  // TBL/TBX: per-byte gather across 1..4 consecutive table registers, mirroring
  // the lite lowering (register-domain-pure, no BB-splits). For each table reg r,
  // shift the index vector down by r*16 (bytewise) so PSHUFB picks lane (idx-r*16)
  // when it is in [0,15]; an in-range mask built via PSUBUSB/PCMPEQB gates the
  // result and forms the running "any table hit" mask. TBL zeroes misses; TBX
  // blends the original Vd back into the miss lanes.
  const uint8_t table_regs = static_cast<uint8_t>(len + 1);  // 1..4
  const int32_t vm_off = GetVRegOffset(rm);
  const int32_t vd_off = GetVRegOffset(rd);

  FpRegister xmm_idx        = AllocTempSimdReg();
  FpRegister xmm_acc        = AllocTempSimdReg();
  FpRegister xmm_in_range   = AllocTempSimdReg();
  FpRegister xmm_const15    = AllocTempSimdReg();
  FpRegister xmm_zero       = AllocTempSimdReg();
  FpRegister xmm_tmp_idx    = AllocTempSimdReg();
  FpRegister xmm_tmp_mask   = AllocTempSimdReg();
  FpRegister xmm_tmp_lookup = AllocTempSimdReg();

  // Load the index vector once. Vm is the per-byte selector.
  builder_.GenGetSimd<16>(xmm_idx.machine_reg(), vm_off);
  // acc = 0, in_range = 0, zero = 0. MOVQ from a zero GP reg zero-extends to a
  // full 128-bit zero and is a proper full-def (a self-PXOR would read an
  // undefined vreg and trip the lifetime analysis in this tier).
  builder_.Gen<x86_64::MovqXRegReg>(xmm_acc.machine_reg(), GetImm(uint64_t{0}));
  builder_.Gen<x86_64::MovqXRegReg>(xmm_in_range.machine_reg(), GetImm(uint64_t{0}));
  builder_.Gen<x86_64::MovqXRegReg>(xmm_zero.machine_reg(), GetImm(uint64_t{0}));
  // const15 = byte-broadcast(0x0F).
  builder_.Gen<x86_64::MovqXRegReg>(xmm_const15.machine_reg(),
                                    GetImm(uint64_t{0x0F0F0F0F0F0F0F0FULL}));
  builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmm_const15.machine_reg(), xmm_const15.machine_reg());

  for (uint8_t r = 0; r < table_regs; ++r) {
    const int32_t vn_off_r =
        GetVRegOffset(((rn + r) & 31));

    // shifted_idx_r = Vm - r*16 (bytewise wrap). For r=0 this is just Vm.
    builder_.Gen<x86_64::MovdqaXRegXReg>(xmm_tmp_idx.machine_reg(), xmm_idx.machine_reg());
    if (r != 0) {
      const uint64_t broadcast =
          uint64_t{0x0101010101010101ULL} * static_cast<uint64_t>(r * 16);
      builder_.Gen<x86_64::MovqXRegReg>(xmm_tmp_mask.machine_reg(), GetImm(broadcast));
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmm_tmp_mask.machine_reg(),
                                               xmm_tmp_mask.machine_reg());
      builder_.Gen<x86_64::PsubbXRegXReg>(xmm_tmp_idx.machine_reg(), xmm_tmp_mask.machine_reg());
    }

    // in_range_r = PCMPEQB(PSUBUSB(shifted_idx_r, 15), 0).
    builder_.Gen<x86_64::MovdqaXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_tmp_idx.machine_reg());
    builder_.Gen<x86_64::PsubusbXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_const15.machine_reg());
    builder_.Gen<x86_64::PcmpeqbXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_zero.machine_reg());

    // looked_up_r = PSHUFB(V[(rn+r)%32], shifted_idx_r), masked by in_range.
    builder_.GenGetSimd<16>(xmm_tmp_lookup.machine_reg(), vn_off_r);
    builder_.Gen<x86_64::PshufbXRegXReg>(xmm_tmp_lookup.machine_reg(), xmm_tmp_idx.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(xmm_tmp_lookup.machine_reg(), xmm_tmp_mask.machine_reg());

    builder_.Gen<x86_64::PorXRegXReg>(xmm_acc.machine_reg(), xmm_tmp_lookup.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(xmm_in_range.machine_reg(), xmm_tmp_mask.machine_reg());
  }

  if (op /* TBX */) {
    // result = acc | (Vd & ~in_range_all).
    builder_.GenGetSimd<16>(xmm_tmp_lookup.machine_reg(), vd_off);
    // xmm_tmp_mask := ~in_range_all & Vd
    builder_.Gen<x86_64::MovdqaXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_in_range.machine_reg());
    builder_.Gen<x86_64::PandnXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_tmp_lookup.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(xmm_acc.machine_reg(), xmm_tmp_mask.machine_reg());
  }

  // SetVRegFull zeroes Vd[127:64] when q=false (D-register semantics).
  SetVRegFull(rd, xmm_acc, q);
}

// AdvSIMD load/store multiple structures. Wave 1 covers the NON-interleaved
// contiguous form (LD1/ST1 with 1..4 registers): a plain bulk transfer of
// num_regs consecutive 16-byte (Q=1) or 8-byte (Q=0) vectors between guest
// memory [Xn{, offset}] and V[rt .. rt+num_regs-1], with optional post-index
// writeback (immediate = num_regs*vec_bytes, or register Xm). Each memory
// access is the unaligned MOVDQU/MOVSD form wrapped in a recovery block so a
// host fault reaches the guest signal handler; the 16-byte-aligned V[] slots
// use GenGetSimd/GenSetSimd (MOVDQA). The Q=0 load uses MOVSD reg<-mem, which
// zero-extends the upper 64 bits (D-register semantics). The interleaving
// LD2/LD3/LD4 / ST2/ST3/ST4 de-interleave forms still bail to the lite tier
// (their element-wise PINSR/PEXTR-from-memory has no heavy LIR op yet).
// Mirrors lite_translator.h::AdvSimdMultiStruct (both the interleaved
// LDn/STn de-/interleave path and the non-interleaved contiguous LD1/ST1).
void HeavyOptimizerFrontend::AdvSimdMultiStruct(uint8_t rt,
                                                uint8_t rn,
                                                uint8_t num_regs,
                                                uint8_t size,
                                                bool q,
                                                bool is_store,
                                                bool postindex,
                                                uint8_t rm,
                                                bool is_interleaved) {
  if (!success()) {
    return;
  }
  const int32_t vec_bytes = q ? 16 : 8;

  // De-interleaving LD2/LD3/LD4 / interleaving ST2/ST3/ST4 (multiple-structure
  // form). Mirrors the lite translator's element-wise lowering: memory holds
  // num_regs * (vec_bytes/esize) elements laid out structure-major — element e
  // in memory belongs to register (e % num_regs), lane (e / num_regs). Because
  // the heavy tier has NO memory-operand PINSR/PEXTR (only the register forms),
  // each element is routed through a GP temp:
  //   load : MOV{zx,}* mem->gp (+recovery) then PINSR gp->lane; the destination
  //          register starts from a zeroed XMM so Q=0 zeroes the upper 64 bits.
  //   store: PEXTR lane->gp then MOV* gp->mem (+recovery).
  // The v[] register accesses stay full-width (GenGetSimd<16>/GenSetSimd<16>) so
  // the optimizer's 16-byte store/load forwarding on a v[] slot is never split
  // by a narrow sub-lane access (see the UMOV/INS-element note above); only the
  // guest-memory side is narrow. Exact for every (num_regs, esize, Q) combo.
  if (is_interleaved) {
    if (num_regs < 2 || num_regs > 4) {
      UndefinedReturningVoid();
      return;
    }
    if (size > 3) {
      UndefinedReturningVoid();
      return;
    }
    const int esize = 1 << size;
    const int num_lanes = vec_bytes / esize;

    Register ibase_orig = (rn == 31) ? GetSp() : GetReg(rn);
    Register ibase = ApplyTbi(ibase_orig);

    for (uint8_t r = 0; r < num_regs; r++) {
      const uint8_t vreg = (rt + r) & 31;
      const int32_t vt_off =
          GetVRegOffset(vreg);
      if (is_store) {
        FpRegister xmm = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xmm.machine_reg(), vt_off);
        for (int l = 0; l < num_lanes; l++) {
          const int32_t mem_off = static_cast<int32_t>((l * num_regs + r) * esize);
          const int8_t lane = static_cast<int8_t>(l);
          Register elem;
          switch (esize) {
            case 1:
              elem = std::get<0>(Gen<x86_64::PextrbRegXRegImm>(xmm.machine_reg(), lane));
              Gen<x86_64::MovbOpReg>({.base = ibase, .disp = mem_off}, elem);
              break;
            case 2:
              elem = std::get<0>(Gen<x86_64::PextrwRegXRegImm>(xmm.machine_reg(), lane));
              Gen<x86_64::MovwOpReg>({.base = ibase, .disp = mem_off}, elem);
              break;
            case 4:
              elem = std::get<0>(Gen<x86_64::PextrdRegXRegImm>(xmm.machine_reg(), lane));
              Gen<x86_64::MovlOpReg>({.base = ibase, .disp = mem_off}, elem);
              break;
            default:  // esize == 8
              elem = std::get<0>(Gen<x86_64::PextrqRegXRegImm>(xmm.machine_reg(), lane));
              Gen<x86_64::MovqOpReg>({.base = ibase, .disp = mem_off}, elem);
              break;
          }
          GenRecoveryBlockForLastInsn();
        }
      } else {
        FpRegister xmm = AllocZeroedSimdReg();
        for (int l = 0; l < num_lanes; l++) {
          const int32_t mem_off = static_cast<int32_t>((l * num_regs + r) * esize);
          const int8_t lane = static_cast<int8_t>(l);
          Register elem;
          switch (esize) {
            case 1:
              elem = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = ibase, .disp = mem_off}));
              GenRecoveryBlockForLastInsn();
              builder_.Gen<x86_64::PinsrbXRegRegImm>(xmm.machine_reg(), elem, lane);
              break;
            case 2:
              elem = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = ibase, .disp = mem_off}));
              GenRecoveryBlockForLastInsn();
              builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), elem, lane);
              break;
            case 4:
              elem = std::get<0>(Gen<x86_64::MovlRegOp>({.base = ibase, .disp = mem_off}));
              GenRecoveryBlockForLastInsn();
              builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), elem, lane);
              break;
            default:  // esize == 8
              elem = std::get<0>(Gen<x86_64::MovqRegOp>({.base = ibase, .disp = mem_off}));
              GenRecoveryBlockForLastInsn();
              builder_.Gen<x86_64::PinsrqXRegRegImm>(xmm.machine_reg(), elem, lane);
              break;
          }
        }
        builder_.GenSetSimd<16>(vt_off, xmm.machine_reg());
      }
    }

    if (postindex) {
      // Writeback preserves the original (un-TBI-masked) top byte, so re-read
      // the base register rather than reusing the masked access address.
      Register reread_base = (rn == 31) ? GetSp() : GetReg(rn);
      Register new_base = Copy(reread_base);
      if (rm == 31) {
        new_base = std::get<0>(Gen<x86_64::AddqRegImm, kNoSSA>(
            new_base, static_cast<int32_t>(num_regs) * vec_bytes));
      } else {
        Register rm_val = GetReg(rm);
        new_base = std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(new_base, rm_val));
      }
      if (rn == 31) {
        SetSp(new_base);
      } else {
        SetReg(rn, new_base);
      }
    }
    return;
  }

  if (num_regs < 1 || num_regs > 4) {
    UndefinedReturningVoid();
    return;
  }

  Register base_orig = (rn == 31) ? GetSp() : GetReg(rn);
  Register base = ApplyTbi(base_orig);

  for (uint8_t r = 0; r < num_regs; r++) {
    const uint8_t vreg = (rt + r) & 31;
    const int32_t vt_off = GetVRegOffset(vreg);
    const int32_t mem_off = static_cast<int32_t>(r) * vec_bytes;
    if (is_store) {
      FpRegister xmm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xmm.machine_reg(), vt_off);
      if (q) {
        builder_.Gen<x86_64::MovdquOpXReg>({.base = base, .disp = mem_off}, xmm.machine_reg());
      } else {
        // Q=0: store the low 64 bits of V[vreg].
        builder_.Gen<x86_64::MovsdOpXReg>({.base = base, .disp = mem_off}, xmm.machine_reg());
      }
      GenRecoveryBlockForLastInsn();
    } else {
      FpRegister xmm =
          q ? FpRegister{std::get<0>(
                  Gen<x86_64::MovdquXRegOp>({.base = base, .disp = mem_off}))}
            // Q=0: MOVSD reg<-mem loads 8 bytes and zero-extends the upper 64.
            : FpRegister{std::get<0>(
                  Gen<x86_64::MovsdXRegOp>({.base = base, .disp = mem_off}))};
      GenRecoveryBlockForLastInsn();
      builder_.GenSetSimd<16>(vt_off, xmm.machine_reg());
    }
  }

  if (postindex) {
    // Writeback preserves the original (un-TBI-masked) top byte, so re-read the
    // base register rather than reusing the masked address used for the access.
    Register reread_base = (rn == 31) ? GetSp() : GetReg(rn);
    Register new_base = Copy(reread_base);
    if (rm == 31) {
      // Immediate post-index: total bytes accessed = num_regs * vec_bytes.
      new_base = std::get<0>(
          Gen<x86_64::AddqRegImm, kNoSSA>(new_base, static_cast<int32_t>(num_regs) * vec_bytes));
    } else {
      Register rm_val = GetReg(rm);
      new_base = std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(new_base, rm_val));
    }
    if (rn == 31) {
      SetSp(new_base);
    } else {
      SetReg(rn, new_base);
    }
  }
}

void HeavyOptimizerFrontend::Sha512(Decoder::Sha512Op op, uint8_t rd, uint8_t rn, uint8_t rm) {
  UndefinedReturningVoid();
  UNUSED_ARGS(op, rd, rn, rm);
}

void HeavyOptimizerFrontend::Eor3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm, ra);
}

void HeavyOptimizerFrontend::Bcax(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm, ra);
}

void HeavyOptimizerFrontend::Rax1(uint8_t rd, uint8_t rn, uint8_t rm) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm);
}

void HeavyOptimizerFrontend::Xar(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm6) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm, imm6);
}

void HeavyOptimizerFrontend::Sm4e(uint8_t rd, uint8_t rn) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn);
}

void HeavyOptimizerFrontend::Sm4ekey(uint8_t rd, uint8_t rn, uint8_t rm) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm);
}

void HeavyOptimizerFrontend::Sm3ss1(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm, ra);
}

void HeavyOptimizerFrontend::Sm3tt(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm2, uint8_t op) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm, imm2, op);
}

void HeavyOptimizerFrontend::Sm3partw1(uint8_t rd, uint8_t rn, uint8_t rm) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm);
}

void HeavyOptimizerFrontend::Sm3partw2(uint8_t rd, uint8_t rn, uint8_t rm) {
  UndefinedReturningVoid();
  UNUSED_ARGS(rd, rn, rm);
}

// AESE/AESD/AESMC/AESIMC via host AES-NI (mirrors lite_translator_crypto.inc):
//   AESE  Vd = AESENCLAST(Vd^Vn, 0)   AESD   Vd = AESDECLAST(Vd^Vn, 0)
//   AESMC Vd = AESENC(AESDECLAST(Vn, 0), 0)   (standalone MixColumns identity)
//   AESIMC Vd = AESIMC(Vn)
// opcode: 00=AESE, 01=AESD, 10=AESMC, 11=AESIMC. Bail to lite if no host AES-NI.
void HeavyOptimizerFrontend::CryptoAes(uint8_t rd, uint8_t rn, uint8_t opcode) {
  if (!success()) {
    return;
  }
  if (!host_platform::kHasAES) {
    UndefinedReturningVoid();
    return;
  }
  const int32_t vd_off = GetVRegOffset(rd);
  const int32_t vn_off = GetVRegOffset(rn);
  FpRegister zero = AllocZeroedSimdReg();  // zero round key / round constant
  FpRegister result;
  switch (opcode) {
    case 0b00:      // AESE
    case 0b01: {    // AESD
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), xn.machine_reg());
      if (opcode == 0b00) {
        builder_.Gen<x86_64::AesenclastXRegXReg>(xd.machine_reg(), zero.machine_reg());
      } else {
        builder_.Gen<x86_64::AesdeclastXRegXReg>(xd.machine_reg(), zero.machine_reg());
      }
      result = xd;
      break;
    }
    case 0b10: {    // AESMC = AESENC(AESDECLAST(Vn, 0), 0)
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vn_off);
      builder_.Gen<x86_64::AesdeclastXRegXReg>(xd.machine_reg(), zero.machine_reg());
      builder_.Gen<x86_64::AesencXRegXReg>(xd.machine_reg(), zero.machine_reg());
      result = xd;
      break;
    }
    default: {      // 0b11 AESIMC = AESIMC(Vn)
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      result = FpRegister{std::get<0>(Gen<x86_64::AesimcXRegXReg>(xn.machine_reg()))};
      break;
    }
  }
  builder_.GenSetSimd<16>(vd_off, result.machine_reg());
}

// SHA-256 helpers shared by CryptoSha3Reg (SHA256H/H2/SU1) and CryptoSha2Reg
// (SHA256SU0). The math mirrors the interpreter's CryptoSha3Reg/CryptoSha2Reg
// exactly (it is the differential oracle); the only cleverness is expressing
// the FIPS-180-4 choose/majority without a NOT (absent from the backend LIR):
//   ch(e,f,g)  = g ^ (e & (f ^ g))
//   maj(a,b,c) = (a & b) ^ (c & (a ^ b))
// and ROL(x,n) as ROR(x, 32-n) (only RORL is in the LIR).
//
// Lane words are moved between the V-register XMMs and GP virtual registers
// with PEXTRD/PINSRD, so every V-register access stays 128-bit wide -- a
// sub-lane 32-bit memory write would risk the width-blind guest-context cache
// forwarding it to a later 128-bit read (the bcrypt-corruption class).
//
// These are common in real apps (TLS/QUIC record auth runs SHA-256 in a
// per-packet loop); bailing fragments that loop into per-instruction
// interpreter round-trips, so lowering it keeps the region -- and the second
// gear -- intact.
struct HeavyOptimizerFrontend::Sha256Ops {
  HeavyOptimizerFrontend* self;

  // SSA Gen<> auto-copies the use_def first operand, so each returns a fresh
  // register and leaves the inputs intact.
  Register Rotr(Register v, int8_t n) {
    return std::get<0>(self->Gen<x86_64::RorlRegImm>(v, n));
  }
  Register ShrImm(Register v, int8_t n) {
    return std::get<0>(self->Gen<x86_64::ShrlRegImm>(v, n));
  }
  Register Xor(Register a, Register b) {
    return std::get<0>(self->Gen<x86_64::XorlRegReg>(a, b));
  }
  Register And(Register a, Register b) {
    return std::get<0>(self->Gen<x86_64::AndlRegReg>(a, b));
  }
  Register Add(Register a, Register b) {
    return std::get<0>(self->Gen<x86_64::AddlRegReg>(a, b));
  }
  Register BigSigma0(Register x) { return Xor(Xor(Rotr(x, 2), Rotr(x, 13)), Rotr(x, 22)); }
  Register BigSigma1(Register x) { return Xor(Xor(Rotr(x, 6), Rotr(x, 11)), Rotr(x, 25)); }
  Register LittleSigma0(Register x) { return Xor(Xor(Rotr(x, 7), Rotr(x, 18)), ShrImm(x, 3)); }
  Register LittleSigma1(Register x) { return Xor(Xor(Rotr(x, 17), Rotr(x, 19)), ShrImm(x, 10)); }
  Register Ch(Register e, Register f, Register g) { return Xor(g, And(e, Xor(f, g))); }
  Register Maj(Register a, Register b, Register c) { return Xor(And(a, b), And(c, Xor(a, b))); }

  Register Lane(FpRegister x, int8_t lane) {
    return std::get<0>(self->Gen<x86_64::PextrdRegXRegImm>(x.machine_reg(), lane));
  }
  FpRegister Pack(Register w0, Register w1, Register w2, Register w3) {
    FpRegister x = self->AllocTempSimdReg();
    self->builder_.Gen<x86_64::MovdXRegReg>(x.machine_reg(), w0);
    self->builder_.Gen<x86_64::PinsrdXRegRegImm>(x.machine_reg(), w1, int8_t{1});
    self->builder_.Gen<x86_64::PinsrdXRegRegImm>(x.machine_reg(), w2, int8_t{2});
    self->builder_.Gen<x86_64::PinsrdXRegRegImm>(x.machine_reg(), w3, int8_t{3});
    return x;
  }
};

void HeavyOptimizerFrontend::CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode) {
  if (!success()) {
    return;
  }
  if (opcode > 0b110) {
    UndefinedReturningVoid();
    return;
  }
  Sha256Ops op{this};
  FpRegister qd = AllocTempSimdReg();
  FpRegister vn = AllocTempSimdReg();
  FpRegister vm = AllocTempSimdReg();
  builder_.GenGetSimd<16>(qd.machine_reg(), GetVRegOffset(rd));
  builder_.GenGetSimd<16>(vn.machine_reg(), GetVRegOffset(rn));
  builder_.GenGetSimd<16>(vm.machine_reg(), GetVRegOffset(rm));

  if (opcode <= 0b010) {
    // SHA1C (000) / SHA1P (001) / SHA1M (010): four SHA-1 rounds.
    // {a,b,c,d} = Vd lanes, e = Sn (Vn lane 0), w[j] = Vm lanes. Per round:
    //   f = Ch/Parity/Maj of (b,c,d);  t = ROL(a,5) + f + e + w[j];
    //   e=d; d=c; c=ROL(b,30); b=a; a=t.   ROL(x,n) == ROR(x,32-n).
    Register a = op.Lane(qd, 0), b = op.Lane(qd, 1), c = op.Lane(qd, 2), d = op.Lane(qd, 3);
    Register e = op.Lane(vn, 0);
    Register w[4] = {op.Lane(vm, 0), op.Lane(vm, 1), op.Lane(vm, 2), op.Lane(vm, 3)};
    for (int j = 0; j < 4; j++) {
      Register f;
      if (opcode == 0b000) {
        f = op.Ch(b, c, d);
      } else if (opcode == 0b001) {
        f = op.Xor(op.Xor(b, c), d);
      } else {
        f = op.Maj(b, c, d);
      }
      Register t = op.Add(op.Add(op.Add(op.Rotr(a, 27), f), e), w[j]);
      e = d;
      d = c;
      c = op.Rotr(b, 2);
      b = a;
      a = t;
    }
    SetVRegFull(rd, op.Pack(a, b, c, d), /*q=*/true);
    return;
  }
  if (opcode == 0b011) {
    // SHA1SU0: message-schedule update, part 1 —
    //   t0 = d2^d0^m0; t1 = d3^d1^m1; t2 = n0^d2^m2; t3 = n1^d3^m3.
    Register d0 = op.Lane(qd, 0), d1 = op.Lane(qd, 1), d2 = op.Lane(qd, 2), d3 = op.Lane(qd, 3);
    Register n0 = op.Lane(vn, 0), n1 = op.Lane(vn, 1);
    Register m0 = op.Lane(vm, 0), m1 = op.Lane(vm, 1), m2 = op.Lane(vm, 2), m3 = op.Lane(vm, 3);
    Register t0 = op.Xor(op.Xor(d2, d0), m0);
    Register t1 = op.Xor(op.Xor(d3, d1), m1);
    Register t2 = op.Xor(op.Xor(n0, d2), m2);
    Register t3 = op.Xor(op.Xor(n1, d3), m3);
    SetVRegFull(rd, op.Pack(t0, t1, t2, t3), /*q=*/true);
    return;
  }

  if (opcode == 0b110) {
    // SHA256SU1: nd0 = d0 + σ1(m2) + n1; nd1 = d1 + σ1(m3) + n2;
    //            nd2 = d2 + σ1(nd0) + n3; nd3 = d3 + σ1(nd1) + m0.
    Register d0 = op.Lane(qd, 0), d1 = op.Lane(qd, 1), d2 = op.Lane(qd, 2), d3 = op.Lane(qd, 3);
    Register n1 = op.Lane(vn, 1), n2 = op.Lane(vn, 2), n3 = op.Lane(vn, 3);
    Register m0 = op.Lane(vm, 0), m2 = op.Lane(vm, 2), m3 = op.Lane(vm, 3);
    Register nd0 = op.Add(op.Add(d0, op.LittleSigma1(m2)), n1);
    Register nd1 = op.Add(op.Add(d1, op.LittleSigma1(m3)), n2);
    Register nd2 = op.Add(op.Add(d2, op.LittleSigma1(nd0)), n3);
    Register nd3 = op.Add(op.Add(d3, op.LittleSigma1(nd1)), m0);
    SetVRegFull(rd, op.Pack(nd0, nd1, nd2, nd3), /*q=*/true);
    return;
  }

  // SHA256H (100): X = Vd, Y = Vn; SHA256H2 (101): X = Vn, Y = Vd.
  FpRegister xreg = (opcode == 0b100) ? qd : vn;
  FpRegister yreg = (opcode == 0b100) ? vn : qd;
  Register x0 = op.Lane(xreg, 0), x1 = op.Lane(xreg, 1), x2 = op.Lane(xreg, 2),
           x3 = op.Lane(xreg, 3);
  Register y0 = op.Lane(yreg, 0), y1 = op.Lane(yreg, 1), y2 = op.Lane(yreg, 2),
           y3 = op.Lane(yreg, 3);
  Register w[4] = {op.Lane(vm, 0), op.Lane(vm, 1), op.Lane(vm, 2), op.Lane(vm, 3)};
  for (int e = 0; e < 4; e++) {
    Register chs = op.Ch(y0, y1, y2);
    Register maj = op.Maj(x0, x1, x2);
    Register t = op.Add(op.Add(op.Add(y3, op.BigSigma1(y0)), chs), w[e]);
    Register new_x3 = op.Add(t, x3);
    Register new_y3 = op.Add(op.Add(t, op.BigSigma0(x0)), maj);
    // Rotate: x = {new_y3, x0, x1, x2}, y = {new_x3, y0, y1, y2}.
    x3 = x2; x2 = x1; x1 = x0; x0 = new_y3;
    y3 = y2; y2 = y1; y1 = y0; y0 = new_x3;
  }
  FpRegister result = (opcode == 0b100) ? op.Pack(x0, x1, x2, x3) : op.Pack(y0, y1, y2, y3);
  SetVRegFull(rd, result, /*q=*/true);
}

void HeavyOptimizerFrontend::CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode) {
  if (!success()) {
    return;
  }
  if (opcode > 0b10) {
    UndefinedReturningVoid();
    return;
  }
  Sha256Ops op{this};
  FpRegister qd = AllocTempSimdReg();
  FpRegister vn = AllocTempSimdReg();
  builder_.GenGetSimd<16>(qd.machine_reg(), GetVRegOffset(rd));
  builder_.GenGetSimd<16>(vn.machine_reg(), GetVRegOffset(rn));
  if (opcode == 0b00) {
    // SHA1H: Sd = ROL(Sn, 30), upper 96 bits zeroed (MOVD zero-extends).
    Register r = op.Rotr(op.Lane(vn, 0), 2);
    FpRegister res = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdXRegReg>(res.machine_reg(), r);
    SetVRegFull(rd, res, /*q=*/true);
    return;
  }
  if (opcode == 0b01) {
    // SHA1SU1: message-schedule update, part 2 —
    //   t_i   = ROL(d_i ^ n_{i+1}, 1) for i = 0..2;
    //   t3    = ROL(d3 ^ t0, 1)        (t0 is the already-rotated word).
    Register d0 = op.Lane(qd, 0), d1 = op.Lane(qd, 1), d2 = op.Lane(qd, 2), d3 = op.Lane(qd, 3);
    Register n1 = op.Lane(vn, 1), n2 = op.Lane(vn, 2), n3 = op.Lane(vn, 3);
    Register t0 = op.Rotr(op.Xor(d0, n1), 31);
    Register t1 = op.Rotr(op.Xor(d1, n2), 31);
    Register t2 = op.Rotr(op.Xor(d2, n3), 31);
    Register t3 = op.Rotr(op.Xor(d3, t0), 31);
    SetVRegFull(rd, op.Pack(t0, t1, t2, t3), /*q=*/true);
    return;
  }
  // SHA256SU0: out_i = d_i + σ0(d_{i+1}), with d_4 taken from Vn[0].
  Register d0 = op.Lane(qd, 0), d1 = op.Lane(qd, 1), d2 = op.Lane(qd, 2), d3 = op.Lane(qd, 3);
  Register n0 = op.Lane(vn, 0);
  Register o0 = op.Add(d0, op.LittleSigma0(d1));
  Register o1 = op.Add(d1, op.LittleSigma0(d2));
  Register o2 = op.Add(d2, op.LittleSigma0(d3));
  Register o3 = op.Add(d3, op.LittleSigma0(n0));
  SetVRegFull(rd, op.Pack(o0, o1, o2, o3), /*q=*/true);
}

// Compute the address for a register-offset load/store:
//   base + extend(offset_reg) << shift_amount
// The extend applies the correct 32->64 widening (UXTW zero-extends, SXTW
// sign-extends, LSL/UXTX/SXTX use the full 64-bit value), mirroring
// lite_translator.h::ApplyOffsetExtend. Returns a fresh register; the base is
// not modified (so an aliased base/dest stays correct).
Register HeavyOptimizerFrontend::EmitRegOffsetAddr(Register base,
                                                   Register offset_reg,
                                                   uint8_t extend_type,
                                                   uint8_t shift_amount) {
  Register addr;
  switch (extend_type) {
    case 0b010:  // UXTW: zero-extend the low 32 bits (a 32-bit mov zero-extends).
      addr = std::get<0>(Gen<x86_64::MovlRegReg>(offset_reg));
      break;
    case 0b110:  // SXTW: sign-extend the low 32 bits to 64.
      addr = std::get<0>(Gen<x86_64::MovsxlqRegReg>(offset_reg));
      break;
    case 0b011:  // LSL / UXTX: full 64-bit value, no extension.
    case 0b111:  // SXTX: full 64-bit value, sign-extend is a no-op here.
    default:
      addr = Copy(offset_reg);
      break;
  }
  if (shift_amount != 0) {
    addr = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(addr, static_cast<int8_t>(shift_amount)));
  }
  return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(addr, base));
}

// Emit `src` shifted by a constant amount (LSL/LSR/ASR/ROR) into a fresh
// register, mirroring lite_translator.h::EmitShift. A 32-bit shift uses the
// l-suffix forms (which zero-extend the result, matching ARM64 W-write
// semantics). When shift_amount == 0 this is just a width-correct copy.
Register HeavyOptimizerFrontend::EmitShiftImm(Register src,
                                              Decoder::ShiftType shift_type,
                                              uint8_t shift_amount,
                                              bool is_64bit) {
  if (is_64bit) {
    Register dst = Copy(src);
    if (shift_amount == 0) {
      return dst;
    }
    int8_t amt = static_cast<int8_t>(shift_amount);
    switch (shift_type) {
      case Decoder::ShiftType::kLsl:
        return std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(dst, amt));
      case Decoder::ShiftType::kLsr:
        return std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(dst, amt));
      case Decoder::ShiftType::kAsr:
        return std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(dst, amt));
      case Decoder::ShiftType::kRor:
        return std::get<0>(Gen<x86_64::RorqRegImm, kNoSSA>(dst, amt));
    }
    return dst;
  }
  Register dst = std::get<0>(Gen<x86_64::MovlRegReg>(src));
  if (shift_amount == 0) {
    return dst;
  }
  int8_t amt = static_cast<int8_t>(shift_amount);
  switch (shift_type) {
    case Decoder::ShiftType::kLsl:
      return std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(dst, amt));
    case Decoder::ShiftType::kLsr:
      return std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(dst, amt));
    case Decoder::ShiftType::kAsr:
      return std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(dst, amt));
    case Decoder::ShiftType::kRor:
      return std::get<0>(Gen<x86_64::RorlRegImm, kNoSSA>(dst, amt));
  }
  return dst;
}

// Write the scalar `value` (in lane 0) to guest V[reg] and ZERO the upper
// bytes, matching ARM scalar-FP write semantics (a later vector read of the
// same register must see a clean zero-extended value).
//
// The whole 16-byte slot is written with ONE aligned MOVDQA store of a
// fully-formed XMM (lane 0 = the scalar, lanes above zero). A single store —
// rather than a zero-store followed by a partial MOVSS/MOVSD — is required
// because RemoveLocalGuestContextAccesses keys dead-store elimination on the
// store displacement alone (not its width): a later MOVSD to the same v[reg]
// offset would otherwise erase a preceding 16-byte zero-store and leave the
// upper 8 bytes stale. We merge `value`'s lane 0 into a zeroed XMM (MOVSS/
// MOVSD reg-reg preserve the dest's upper lanes), then store it once.
void HeavyOptimizerFrontend::SetVRegScalar(uint8_t reg, FpRegister value, bool is_double) {
  if (!success()) {
    return;
  }
  const int32_t off = GetVRegOffset(reg);
  FpRegister merged = AllocZeroedSimdReg();
  if (is_double) {
    builder_.Gen<x86_64::MovsdXRegXReg>(merged.machine_reg(), value.machine_reg());
  } else {
    builder_.Gen<x86_64::MovssXRegXReg>(merged.machine_reg(), value.machine_reg());
  }
  builder_.Gen<x86_64::MovdqaOpXReg>({.base = x86_64::kMachineRegRBP, .disp = off},
                                     merged.machine_reg());
}

// Write a scalar value held in a GP register (low 4 bytes for S, low 8 for D)
// to guest V[reg], zeroing the upper bytes. Same single-MOVDQA-store rationale
// as SetVRegScalar: move the GP value into lane 0 of a zeroed XMM, store once.
void HeavyOptimizerFrontend::SetVRegScalarFromGp(uint8_t reg, Register value, bool is_double) {
  if (!success()) {
    return;
  }
  const int32_t off = GetVRegOffset(reg);
  FpRegister merged = AllocZeroedSimdReg();
  if (is_double) {
    // MOVQ xmm, r64 zero-extends into the XMM (upper 64 cleared).
    builder_.Gen<x86_64::MovqXRegReg>(merged.machine_reg(), value);
  } else {
    // MOVD xmm, r32 zero-extends into the XMM (upper 96 cleared).
    builder_.Gen<x86_64::MovdXRegReg>(merged.machine_reg(), value);
  }
  builder_.Gen<x86_64::MovdqaOpXReg>({.base = x86_64::kMachineRegRBP, .disp = off},
                                     merged.machine_reg());
}

// Write a full or D-form vector result to guest V[reg]. For the Q-form
// (q == true) all 128 bits of `value` are committed. For the D-form
// (q == false) the upper 64 bits MUST be zeroed (ARM64 writes of a 64-bit
// vector clear the rest of the register), so the low 64 bits of `value` are
// merged (MOVSD reg-reg, which copies the low 8 bytes and preserves the
// destination's upper lanes) into a PXOR-zeroed XMM before the store. Both
// paths use ONE 16-byte MOVDQA store at the v[reg] displacement, matching the
// single-store discipline of SetVRegScalar so RemoveLocalGuestContextAccesses
// forwards and dead-store-eliminates correctly (it keys on the store
// displacement, not its width).
void HeavyOptimizerFrontend::SetVRegFull(uint8_t reg, FpRegister value, bool q) {
  if (!success()) {
    return;
  }
  const int32_t off = GetVRegOffset(reg);
  if (q) {
    builder_.GenSetSimd<16>(off, value.machine_reg());
    return;
  }
  FpRegister merged = AllocZeroedSimdReg();
  builder_.Gen<x86_64::MovsdXRegXReg>(merged.machine_reg(), value.machine_reg());
  builder_.GenSetSimd<16>(off, merged.machine_reg());
}

// Commit a narrowing-op result: the narrowed lanes occupy the low 64 bits of
// `narrowed`. Q=0 stores them to Vd.low with the upper 64 zeroed; Q=1 (the
// "2" form) shifts them into Vd.high (PSLLDQ by 8) while preserving Vd's
// existing low 64 (masked via MOVSD into a zeroed reg, then POR). Mirrors the
// Q=0/Q2 store discipline of lite_translator.h's XTN/SQXTN family.
void HeavyOptimizerFrontend::SetVRegNarrow(uint8_t rd, FpRegister narrowed, bool q) {
  if (!success()) {
    return;
  }
  if (!q) {
    SetVRegFull(rd, narrowed, /*q=*/false);
    return;
  }
  const int32_t vd_off = GetVRegOffset(rd);
  builder_.Gen<x86_64::PslldqXRegImm>(narrowed.machine_reg(), int8_t{8});
  FpRegister xd = AllocTempSimdReg();
  FpRegister xd_low = AllocZeroedSimdReg();
  builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
  builder_.Gen<x86_64::MovsdXRegXReg>(xd_low.machine_reg(), xd.machine_reg());
  builder_.Gen<x86_64::PorXRegXReg>(narrowed.machine_reg(), xd_low.machine_reg());
  SetVRegFull(rd, narrowed, /*q=*/true);
}

// AdvSIMDExpandImm (ARM ARM): the 128-bit MOVI/MVNI/FMOV(vector) immediate as
// a pure function of (op, cmode, abc, defgh, q). Identical to
// lite_translator.h::ExpandSimdModifiedImmJit; uses the VFPExpandImm helpers
// above for the FMOV (cmode=1111) forms.
__uint128_t HeavyOptimizerFrontend::ExpandSimdModifiedImmJit(uint8_t op, uint8_t cmode, uint8_t abc,
                                                             uint8_t defgh, bool q) {
  uint8_t imm8 = (abc << 5) | defgh;
  uint64_t imm64 = 0;
  if (op == 0) {
    switch (cmode >> 1) {
      case 0b000: imm64 = uint64_t{imm8} | (uint64_t{imm8} << 32); break;
      case 0b001: imm64 = (uint64_t{imm8} << 8) | (uint64_t{imm8} << 40); break;
      case 0b010: imm64 = (uint64_t{imm8} << 16) | (uint64_t{imm8} << 48); break;
      case 0b011: imm64 = (uint64_t{imm8} << 24) | (uint64_t{imm8} << 56); break;
      case 0b100: for (int i = 0; i < 4; i++) imm64 |= uint64_t{imm8} << (i * 16); break;
      case 0b101: for (int i = 0; i < 4; i++) imm64 |= uint64_t{imm8} << (i * 16 + 8); break;
      case 0b110:
        if (!(cmode & 1)) {
          uint32_t v = (uint32_t{imm8} << 8) | 0xFF;
          imm64 = uint64_t{v} | (uint64_t{v} << 32);
        } else {
          uint32_t v = (uint32_t{imm8} << 16) | 0xFFFF;
          imm64 = uint64_t{v} | (uint64_t{v} << 32);
        }
        break;
      case 0b111:
        if (!(cmode & 1)) {
          for (int i = 0; i < 8; i++) imm64 |= uint64_t{imm8} << (i * 8);
        } else {
          uint32_t f = VFPExpandImm32(imm8);
          imm64 = uint64_t{f} | (uint64_t{f} << 32);
        }
        break;
    }
  } else {
    if (cmode == 0b1110) {
      for (int i = 0; i < 8; i++)
        if (imm8 & (1 << i)) imm64 |= 0xFFULL << (i * 8);
    } else if (cmode == 0b1111) {
      imm64 = VFPExpandImm64(imm8);
    } else {
      return ~ExpandSimdModifiedImmJit(0, cmode, abc, defgh, q);
    }
  }
  __uint128_t result = static_cast<__uint128_t>(imm64);
  if (q) result |= static_cast<__uint128_t>(imm64) << 64;
  return result;
}

}  // namespace berberis
