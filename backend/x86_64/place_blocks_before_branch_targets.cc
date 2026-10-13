#include "berberis/backend/x86_64/place_blocks_before_branch_targets.h"

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_analysis.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

namespace {

// Loop nesting deeper than this weighs the same.
constexpr uint32_t kMaxWeightedDepth = 8;

const MachineBasicBlock* UncondBranchTarget(const MachineBasicBlock* bb) {
  if (bb->insn_list().empty()) {
    return nullptr;
  }
  const auto* last = bb->insn_list().back();
  if (last->opcode() != kMachineOpPseudoBranch) {
    return nullptr;
  }
  return static_cast<const PseudoBranch*>(last)->then_bb();
}

// Whether `from` falls through to `to` when `to` is laid out right after it:
// one of its branch arms goes there.
bool BranchesTo(const MachineBasicBlock* from, const MachineBasicBlock* to) {
  if (from == nullptr || to == nullptr || from->insn_list().empty()) {
    return false;
  }
  const auto* last = from->insn_list().back();
  if (last->opcode() == kMachineOpPseudoBranch) {
    return static_cast<const PseudoBranch*>(last)->then_bb() == to;
  }
  if (last->opcode() == kMachineOpPseudoCondBranch) {
    const auto* branch = static_cast<const PseudoCondBranch*>(last);
    return branch->then_bb() == to || branch->else_bb() == to;
  }
  return false;
}

}  // namespace

void PlaceBlocksBeforeBranchTargets(MachineIR* machine_ir) {
  auto& bb_list = machine_ir->bb_list();
  if (bb_list.size() < 3) {
    return;
  }

  ArenaVector<uint32_t> depth(machine_ir->NumBasicBlocks(), 0, machine_ir->arena());
  for (const auto* loop : FindLoops(machine_ir)) {
    for (const auto* bb : *loop) {
      depth[bb->id()]++;
    }
  }
  auto weight = [&depth](const MachineBasicBlock* bb) -> uint64_t {
    if (bb == nullptr) {
      return 0;
    }
    return uint64_t{1} << (3 * std::min(depth[bb->id()], kMaxWeightedDepth));
  };

  const MachineBasicBlock* entry = bb_list.front();
  MachineBasicBlockList candidates(bb_list);
  for (auto* bb : candidates) {
    const MachineBasicBlock* target = UncondBranchTarget(bb);
    if (bb == entry || bb->is_recovery() || target == nullptr || target == bb ||
        target == entry || target->is_recovery()) {
      continue;
    }
    auto bb_it = std::find(bb_list.begin(), bb_list.end(), bb);
    auto target_it = std::find(bb_list.begin(), bb_list.end(), target);
    if (bb_it == bb_list.end() || target_it == bb_list.end()) {
      continue;
    }
    const MachineBasicBlock* target_prev = *std::prev(target_it);
    if (target_prev == bb) {
      continue;
    }
    // bb_it is not the first block, so it has a layout predecessor.
    const MachineBasicBlock* bb_prev = *std::prev(bb_it);
    auto bb_next_it = std::next(bb_it);
    const MachineBasicBlock* bb_next = bb_next_it == bb_list.end() ? nullptr : *bb_next_it;

    // Before: bb_prev, bb, bb_next ... target_prev, target.
    // After:  bb_prev, bb_next ... target_prev, bb, target.
    uint64_t gain = weight(bb);
    uint64_t loss = 0;
    if (BranchesTo(target_prev, bb)) {
      gain += weight(target_prev);
    }
    if (BranchesTo(target_prev, target)) {
      loss += weight(target_prev);
    }
    if (BranchesTo(bb_prev, bb)) {
      loss += weight(bb_prev);
    }
    if (BranchesTo(bb_prev, bb_next)) {
      gain += weight(bb_prev);
    }
    if (gain <= loss) {
      continue;
    }
    bb_list.splice(target_it, bb_list, bb_it);
  }
}

}  // namespace berberis::x86_64
