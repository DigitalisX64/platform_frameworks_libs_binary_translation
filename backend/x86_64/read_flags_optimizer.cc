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

#include <optional>

#include "berberis/backend/common/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/algorithm.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

// Reads range of instructions to see if any of the registers in regs is used.
// Will also insert new registers into regs if we encounter PSEUDO_COPY.
// Returns true iff we reach the end without encountering any uses of regs.
bool CheckRegsUnusedWithinInsnRange(MachineInsnList::iterator insn_it,
                                    MachineInsnList::iterator end,
                                    ArenaVector<MachineReg>& regs) {
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
bool CheckSuccessorNode(Loop* loop, MachineBasicBlock* bb, ArenaVector<MachineReg>& regs) {
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
bool CheckPostLoopNode(MachineBasicBlock* bb, const ArenaVector<MachineReg>& regs) {
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
  return true;
}

// Checks if anything in regs is in bb->live_in().
bool RegsLiveInBasicBlock(MachineBasicBlock* bb, const ArenaVector<MachineReg>& regs) {
  for (auto r : bb->live_in()) {
    if (Contains(regs, r)) {
      return true;
    }
  }
  return false;
}

template <typename T>
MachineInsn* CopyInstruction(MachineIR* machine_ir, MachineInsn* insn) {
  return machine_ir->NewInsn<T>(*static_cast<T*>(insn));
}

std::optional<InsnGenerator> GetInsnGen(MachineOpcode opcode) {
  switch (opcode) {
    case kMachineOpAddqRegReg:
      return CopyInstruction<AddqRegReg>;
    case kMachineOpPseudoReadFlags:
      return CopyInstruction<PseudoReadFlags>;
    case kMachineOpCmplRegImm:
      return CopyInstruction<CmplRegImm>;
    case kMachineOpCmplRegReg:
      return CopyInstruction<CmpqRegReg>;
    case kMachineOpCmpqRegImm:
      return CopyInstruction<CmpqRegImm>;
    case kMachineOpCmpqRegReg:
      return CopyInstruction<CmpqRegReg>;
    case kMachineOpSublRegImm:
      return CopyInstruction<SublRegImm>;
    case kMachineOpSublRegReg:
      return CopyInstruction<SublRegReg>;
    case kMachineOpSubqRegImm:
      return CopyInstruction<SubqRegImm>;
    case kMachineOpSubqRegReg:
      return CopyInstruction<SubqRegReg>;
    default:
      return std::nullopt;
  }
}

// Finds the instruction which sets a flag register.
// insn_it should point to one past the element we first want to check
// (typically it should point to the readflags instruction).
std::optional<MachineInsnList::iterator> FindFlagSettingInsn(MachineInsnList::iterator insn_it,
                                                             MachineInsnList::iterator begin,
                                                             MachineReg reg) {
  while (insn_it != begin) {
    insn_it--;
    for (int i = 0; i < (*insn_it)->NumRegOperands(); i++) {
      if ((*insn_it)->RegAt(i) == reg && (*insn_it)->RegKindAt(i).IsDef()) {
        return insn_it;
      }
    }
  }
  return std::nullopt;
}

}  // namespace berberis::x86_64
