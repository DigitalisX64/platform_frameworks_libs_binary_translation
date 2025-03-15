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

#include <algorithm>

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

// Checks if this post loop node meets requirements for the read flags
// optimization.
// Requirements:
// * the node must have only one in_edge - this guarantees the register is coming
// from the readflags
// * nothing in regs should be in live_out
bool CheckPostLoopNode(MachineBasicBlock* bb, const ArenaVector<MachineReg>& regs) {
  // If the node doesn't actually use any of regs we can just skip it.
  bool is_live_in = false;
  for (auto r : bb->live_in()) {
    if (Contains(regs, r)) {
      is_live_in = true;
      break;
    }
  }
  if (!is_live_in) {
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

}  // namespace berberis::x86_64
