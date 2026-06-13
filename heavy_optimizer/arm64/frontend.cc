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

#include <cstddef>
#include <cstdint>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

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

void HeavyOptimizerFrontend::Undefined() {
  // Idempotent: a single guest instruction can trigger several listener calls
  // (e.g. a pre/post-index access calls AddImm then Load/Store). If more than one
  // bails, only the first may append the region-exit terminator — a second exit
  // in the same basic block would make it fail CheckMachineIR.
  if (!success_) {
    return;
  }
  success_ = false;
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
