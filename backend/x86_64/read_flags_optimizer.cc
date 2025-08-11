/*
 * Copyright (C) 2025 The Android Open Source Project
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

#include "berberis/backend/x86_64/read_flags_optimizer.h"

#include <iterator>
#include <optional>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/algorithm.h"
#include "berberis/base/arena_set.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

// Reads range of instructions to see if any of the registers in regs is used.
// Will also insert new registers into regs if we encounter PSEUDO_COPY.
// Returns true iff we reach the end without encountering any uses of regs.
bool CheckRegsUnusedWithinInsnRange(MachineInsnList::iterator insn_it,
                                    MachineInsnList::iterator end,
                                    MachineRegVector& regs) {
  for (; insn_it != end; ++insn_it) {
    for (auto i = 0; i < (*insn_it)->NumRegOperands(); i++) {
      if (Contains(regs, (*insn_it)->RegAt(i))) {
        if (AsMachineInsnX86_64(*insn_it)->opcode() != kMachineOpPseudoCopy || i != 1) {
          return false;
        }
        regs.push_back((*insn_it)->RegAt(0));
      }
    }
  }
  return true;
}

// Checks if a successor node meets requirements for read flags optimization
// Requirements:
// * must be exit node or not use registers
// * only one in_edge - guarantees register comes from the readflags node
// * any registers from regs can only be live_in to the post loop nodes
// * nothing from regs used in node
// * Postloop node connected to this node must meet same post loop node as
//   original node with readflags instruction
//
// Returns true iff this node doesn't stop us from using the optimization.
bool CheckSuccessorNode(Loop* loop, MachineBasicBlock* bb, MachineRegVector& regs) {
  // If the node doesn't actually use any of regs we can just skip it.
  if (!RegsLiveInBasicBlock(bb, regs)) {
    return true;
  }

  // To simplify things we only allow one in_edge.
  if (bb->in_edges().size() != 1) {
    return false;
  }

  MachineEdge* postloop_edge;
  MachineEdge* loop_edge;
  // Nodes have at most 2 out_edges so if this is a successor node there can be
  // at most one postloop edge.
  for (auto edge : bb->out_edges()) {
    if (Contains(*loop, edge->dst())) {
      loop_edge = edge;
    } else {
      // There should only be one exit edge.
      CHECK_EQ(postloop_edge, nullptr);
      postloop_edge = edge;
    }
  }
  // Check if exit node.
  if (postloop_edge == nullptr) {
    return false;
  }
  CHECK(loop_edge);

  // Check regs not used in node. Note this can add additional elements into regs.
  if (!CheckRegsUnusedWithinInsnRange(bb->insn_list().begin(), bb->insn_list().end(), regs)) {
    return false;
  }
  // Check if regs found in live_in of other loop nodes.
  // Must be done after CheckRegsUnusedWithinInsnRange in case we added new registers to regs.
  if (RegsLiveInBasicBlock(loop_edge->dst(), regs)) {
    return false;
  }
  // Check post loop nodes.
  return CheckPostLoopNode(postloop_edge->dst(), regs);
}

// Checks if this post loop node meets requirements for the read flags
// optimization.
// Requirements:
// * the node must have only one in_edge - this guarantees the register is coming
// from the readflags
// * nothing in regs should be in live_out
// * does not redefine registers in regs - this simplifies the logic of figuring out when to
// insert instructions (see b/417321580 for more context)
bool CheckPostLoopNode(MachineBasicBlock* bb, const MachineRegVector& regs) {
  // If the node doesn't actually use any of regs we can just skip it.
  if (!RegsLiveInBasicBlock(bb, regs)) {
    return true;
  }

  // Check that there's only one in_edge.
  if (bb->in_edges().size() != 1) {
    return false;
  }
  // Check that it's not live_out.
  for (auto r : bb->live_out()) {
    if (Contains(regs, r)) {
      return false;
    }
  }

  for (auto insn : bb->insn_list()) {
    for (int i = 0; i < insn->NumRegOperands(); i++) {
      if (Contains(regs, insn->RegAt(i)) && insn->RegKindAt(i).IsDef()) {
        return false;
      }
    }
  }

  return true;
}

// Checks if anything in regs is in bb->live_in().
bool RegsLiveInBasicBlock(MachineBasicBlock* bb, const MachineRegVector& regs) {
  for (auto r : bb->live_in()) {
    if (Contains(regs, r)) {
      return true;
    }
  }
  return false;
}

bool SupportedInsn(MachineOpcode opcode) {
  switch (opcode) {
    case kMachineOpAddqRegReg:
    case kMachineOpPseudoReadFlags:
    case kMachineOpCmplRegImm:
    case kMachineOpCmplRegReg:
    case kMachineOpCmpqRegImm:
    case kMachineOpCmpqRegReg:
    case kMachineOpSublRegImm:
    case kMachineOpSublRegReg:
    case kMachineOpSubqRegImm:
    case kMachineOpSubqRegReg:
      return true;
    default:
      return false;
  }
}

// Finds all read flags we can optimize away and removes them.
void RemoveEligibleReadFlagsInLoopTree(MachineIR* machine_ir, LoopTreeNode* loop_tree_node) {
  if (loop_tree_node->NumInnerloops() > 0) {
    // Remove from inner loops first.
    for (size_t i = 0; i < loop_tree_node->NumInnerloops(); i++) {
      RemoveEligibleReadFlagsInLoopTree(machine_ir, loop_tree_node->GetInnerloopNode(i));
    }
  }
  auto loop = loop_tree_node->loop();
  if (loop == nullptr) {
    return;
  }
  // TODO(b/417284998): We could skip the nodes which were already scanned in inner loops.
  for (auto* bb : *loop) {
    for (auto insn_it = bb->insn_list().begin(); insn_it != bb->insn_list().end(); insn_it++) {
      if (AsMachineInsnX86_64(*insn_it)->opcode() == kMachineOpPseudoReadFlags) {
        auto flag_set_opt = IsEligibleReadFlag(machine_ir, loop, bb, insn_it);
        if (flag_set_opt.has_value()) {
          RemoveReadFlags(machine_ir, ReadFlagsOptContext{bb, insn_it, flag_set_opt.value()});
        }
      }
    }
  }
}

// Finds the instruction which sets a flag register.
// insn_it should point to one past the element we first want to check
// (typically it should point to the readflags instruction).
std::optional<FlagSettingInsn> FindFlagSettingInsn(MachineInsnList::iterator insn_it,
                                                   MachineInsnList::iterator begin,
                                                   MachineReg reg) {
  bool cmc = false;
  while (insn_it != begin) {
    insn_it--;
    for (int i = 0; i < (*insn_it)->NumRegOperands(); i++) {
      if ((*insn_it)->RegAt(i) == reg && (*insn_it)->RegKindAt(i).IsDef()) {
        if ((*insn_it)->opcode() == kMachineOpCmc) {
          // CMC just inverts the carry flag so we still need to find what sets the original EFLAGS.
          cmc = true;
          // We need to go to the previous instruction, but since we know that CMC has just one
          // operand simple "continue" here immediately exits the loop over operands.
          continue;
        }
        return FlagSettingInsn{insn_it, cmc};
      }
    }
  }
  return std::nullopt;
}

void InsertFlagGenInstructions(MachineIR* machine_ir,
                               ReadFlagsOptContext& context,
                               MachineInsnList::iterator insn_it,
                               const ArenaMap<MachineReg, MachineReg>& reg_map,
                               MachineReg reg) {
  auto flag_reg_used = NeedsToSaveFlags(context.bb, insn_it);
  MachineReg flag_copy;
  if (flag_reg_used.has_value()) {
    flag_copy = machine_ir->AllocVReg();
    context.bb->insn_list().insert(
        insn_it,
        machine_ir->NewInsn<PseudoReadFlags>(
            PseudoReadFlags::kWithOverflow, flag_copy, flag_reg_used.value()));
  }
  MachineReg flag_reg;
  // First add instruction that sets flags register.
  auto insn = machine_ir->CloneInstruction(*context.flag_set_insn.insn);
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegKindAt(i).IsInput()) {
      CHECK(reg_map.contains(insn->RegAt(i)));
      MachineReg input_reg;
      if (insn->RegKindAt(i).IsDef()) {
        // If it gets overwritten by the instruction, we need to make a new copy.
        input_reg = machine_ir->AllocVReg();
        context.bb->insn_list().insert(
            insn_it, machine_ir->NewInsn<PseudoCopy>(input_reg, reg_map.at(insn->RegAt(i)), 8));
      } else {
        // if it's not def we can just reuse the copy from before.
        input_reg = reg_map.at(insn->RegAt(i));
      }
      insn->SetRegAt(i, input_reg);
    } else {
      // Allocate new registers for non-input as original ones are
      // probably not in scope.
      insn->SetRegAt(i, machine_ir->AllocVReg());
      if (insn->RegKindAt(i).RegClass()->IsSubsetOf(&kFLAGS)) {
        // Save the flag register to set PSEUDO_READFLAGS.
        flag_reg = insn->RegAt(i);
      }
    }
  }
  CHECK(!flag_reg.IsInvalidReg());
  context.bb->insn_list().insert(insn_it, insn);
  if (context.flag_set_insn.cmc) {
    context.bb->insn_list().insert(insn_it, machine_ir->NewInsn<Cmc>(flag_reg));
  }

  // Now add readflags instruction.
  insn = machine_ir->CloneInstruction(*context.readflags_insn);
  insn->SetRegAt(0, reg);
  insn->SetRegAt(1, flag_reg);
  context.bb->insn_list().insert(insn_it, insn);

  if (flag_reg_used.has_value()) {
    context.bb->insn_list().insert(
        insn_it, machine_ir->NewInsn<PseudoWriteFlags>(flag_copy, flag_reg_used.value()));
  }
}

// Given an iterator that points to a READFLAGS instruction, checks if the
// instruction can be optimized away.
//
// In the case we can't optimize it, we return std::nullopt. If we can optimize
// it, we return an optional containing a pointer to the MachineInsn which set
// the flag register which we would be reading.
//
// For now we only consider the common special case.
// READFLAG is eligible to be removed if
// * in loop
// * register must not be used elsewhere in the loop
// * lifetime of register should be limited to an exit node,
//   post loop node, neighboring exit nodes, and post loop nodes of those
//   neighbors
//   * We can guarantee this via live_in and live_out properties
//   * For post loop, neighbor exit nodes, and post loop nodes of
//     those neighbors, only one in_edges
//   * Register must not be live_in besides in the aforementioned nodes.
//
// As example of the allowed configuration
//   (LOOP NODE) -> (READFLAG NODE) -> (POST LOOP NODE)
//         ^             |
//         |             v
//   (LOOP NODE) <-  (EXIT NODE) ---> (NEIGHBOR'S POST LOOP NODE)
std::optional<FlagSettingInsn> IsEligibleReadFlag(MachineIR* machine_ir,
                                                  Loop* loop,
                                                  MachineBasicBlock* bb,
                                                  MachineInsnList::iterator insn_it) {
  CHECK_EQ(AsMachineInsnX86_64(*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  auto flag_register = (*insn_it)->RegAt(1);
  // We use a set here because the original register will be pseudocopy'd when
  // used as live_out. So long as these new registers adhere to the same
  // constraints this is fine.
  MachineRegVector regs({(*insn_it)->RegAt(0)}, machine_ir->arena());
  insn_it++;
  if (!CheckRegsUnusedWithinInsnRange(insn_it, bb->insn_list().end(), regs)) {
    return std::nullopt;
  }

  bool is_exit_node = false;
  // Reached end of basic block, check neighbors.
  for (auto edge : bb->out_edges()) {
    if (Contains(*loop, edge->dst())) {
      // Check if it's a neighbor exit node.
      if (!CheckSuccessorNode(loop, edge->dst(), regs)) {
        return std::nullopt;
      }
    } else {
      is_exit_node = true;
      // Check if it satisifes post loop node requirements.
      if (!CheckPostLoopNode(edge->dst(), regs)) {
        return std::nullopt;
      }
    }
  }
  if (!is_exit_node) {
    return std::nullopt;
  }

  // Make sure we know how to copy this instruction.
  auto flag_setter = FindFlagSettingInsn(insn_it, bb->insn_list().begin(), flag_register);
  if (flag_setter.has_value() && SupportedInsn((*flag_setter.value().insn)->opcode())) {
    return flag_setter.value();
  }
  return std::nullopt;
}

// Check if we need to save the flag register because a later instruction uses it. If so, returns
// the flag MachineReg that's used.
std::optional<MachineReg> NeedsToSaveFlags(MachineBasicBlock* bb,
                                           MachineInsnList::iterator insn_it) {
  for (; insn_it != bb->insn_list().end(); ++insn_it) {
    for (int i = 0; i < (*insn_it)->NumRegOperands(); i++) {
      auto reg_kind = (*insn_it)->RegKindAt(i);
      if (!reg_kind.RegClass()->IsSubsetOf(&kFLAGS)) {
        continue;
      }
      if (reg_kind.IsInput()) {
        return (*insn_it)->RegAt(i);
      }
      // Instruction clobbers it so we don't need to worry about rest of instructions.
      return std::nullopt;
    }
  }
  // Host flags should never be live_out across basic blocks.
  // It would be better to do a CHECK but currently there's no way to know
  // whether a virtual register is a flag or not.
  return std::nullopt;
}

void OptimizeReadFlags(MachineIR* machine_ir) {
  auto loop_tree = BuildLoopTree(machine_ir);
  RemoveEligibleReadFlagsInLoopTree(machine_ir, loop_tree.root());
}

// Removes all elements of regs_to_remove from remove_from_regs. Returns true if anything was
// removed.
//
// Note this ideally only be used for small vectors as it's O(n^2).
bool RemoveRegs(MachineRegVector& remove_from_regs, const MachineRegVector& regs_to_remove) {
  auto orig_size = remove_from_regs.size();
  for (auto rit = remove_from_regs.rbegin(); rit != remove_from_regs.rend();
       /* Incremented in loop */) {
    if (Contains(regs_to_remove, *rit)) {
      // erase only takes forward iterator so we create one from rit.
      rit = EraseFromReverseIterator(remove_from_regs, rit);
    } else {
      rit++;
    }
  }
  return orig_size != remove_from_regs.size();
}

// Removes the READFLAGS instruction, finds the instruction which generated the
// flags, and creates copies of the registers.
void RemoveReadFlags(MachineIR* machine_ir, ReadFlagsOptContext context) {
  auto insn_it = context.readflags_insn;
  auto flags_reg = (*insn_it)->RegAt(0);
  // Delete READFLAGS instruction
  context.bb->insn_list().erase(insn_it);

  insn_it = context.flag_set_insn.insn;

  berberis::MachineInsn* insn = *insn_it;

  // Create copies of input registers.
  ArenaMap<MachineReg, MachineReg> reg_map(machine_ir->arena());
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegKindAt(i).IsInput()) {
      MachineReg copy = machine_ir->AllocVReg();
      reg_map[insn->RegAt(i)] = copy;
      context.bb->insn_list().insert(insn_it,
                                     machine_ir->NewInsn<PseudoCopy>(copy, insn->RegAt(i), 8));
    }
  }

  ArenaVector<MachineReg> reg_vec({flags_reg}, machine_ir->arena());
  ReplaceFlagRegisters(
      machine_ir, context, std::next(context.flag_set_insn.insn), reg_vec, reg_map, insn);
}

// Propagates the copied input registers, and regenerates the EFLAGs register if
// we find an instruction that uses it. Updates live_in/live_out of blocks to
// include copied input registers.
//
// Params:
// * context - ReadFlagsOptContext generated from where the readflags
// instruction was found.
// * insn_it - iterator for MachineInsn in block for where we should begin
// reading instructions. Should be begin() except when called from
// RemoveReadFlags
// * flags_regs - set of flags register and its PSEUDOCOPY's
// * reg_map - the mapping from the original input registers to their copies
// * insn - Original instruction which created the EFLAGS register.
void ReplaceFlagRegisters(MachineIR* machine_ir,
                          ReadFlagsOptContext context,
                          MachineInsnList::iterator insn_it,
                          MachineRegVector flags_regs,
                          const ArenaMap<MachineReg, MachineReg>& reg_map,
                          berberis::MachineInsn* insn) {
  ArenaSet<MachineReg> used_flags{machine_ir->arena()};
  while (insn_it != context.bb->insn_list().end()) {
    if (AsMachineInsnX86_64(*insn_it)->opcode() == kMachineOpPseudoCopy &&
        Contains(flags_regs, (*insn_it)->RegAt(1))) {
      // If flags register was copied we add the copy to flags_regs and delete instruction.
      flags_regs.push_back((*insn_it)->RegAt(0));
      insn_it = context.bb->insn_list().erase(insn_it);
      continue;
    }
    // Check if we use the register.
    used_flags.clear();
    for (int i = 0; i < (*insn_it)->NumRegOperands(); i++) {
      if (Contains(flags_regs, (*insn_it)->RegAt(i))) {
        used_flags.insert((*insn_it)->RegAt(i));
      }
    }
    // Insert instructions for any flags we used.
    for (auto reg : used_flags) {
      InsertFlagGenInstructions(machine_ir, context, insn_it, reg_map, reg);
    }
    insn_it++;
  }

  // Add copied registers to live_in if needed.
  for (auto reg : context.bb->live_in()) {
    if (Contains(flags_regs, reg)) {
      for (auto mapping : reg_map) {
        context.bb->live_in().push_back(mapping.second);
      }
      break;
    }
  }

  // Remove flags_regs from live_in and live_out.
  RemoveRegs(context.bb->live_in(), flags_regs);
  auto was_live_out = RemoveRegs(context.bb->live_out(), flags_regs);
  // Update live_out with our copied input registers if flags_regs was in
  // live_out.
  if (was_live_out) {
    for (auto mapping : reg_map) {
      context.bb->live_out().push_back(mapping.second);
    }
  }

  // Recurse on neighbors where flags registers are live_in.
  for (auto* out_edge : context.bb->out_edges()) {
    for (auto live_in_reg : out_edge->dst()->live_in()) {
      if (Contains(flags_regs, live_in_reg)) {
        ReplaceFlagRegisters(
            machine_ir,
            ReadFlagsOptContext{out_edge->dst(), context.readflags_insn, context.flag_set_insn},
            out_edge->dst()->insn_list().begin(),
            flags_regs,
            reg_map,
            insn);
        break;
      }
    }
  }
}

}  // namespace berberis::x86_64
