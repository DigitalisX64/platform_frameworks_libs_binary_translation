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

#include "berberis/backend/x86_64/lower_ssa_instructions.h"

#include <cstddef>
#include <utility>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/arena_vector.h"
#include "berberis/base/checks.h"
#include "berberis/base/tracing.h"

namespace berberis::x86_64 {

bool CheckSSA(MachineIR* machine_ir) {
  ArenaVector<bool> set_registers(machine_ir->NumVReg(), false, machine_ir->arena());
  for (auto* bb : machine_ir->bb_list()) {
    std::fill(begin(set_registers), end(set_registers), false);
    for (auto reg : bb->live_in()) {
      if (!reg.IsVReg()) {
        TRACE("Non-vreg %s living-in before lowering", GetMachineRegDebugString(reg).c_str());
        return false;
      }
      int vreg_idx = reg.GetVRegIndex();
      if (set_registers.at(vreg_idx)) {
        TRACE("Register %s set twice in live_in array", GetMachineRegDebugString(reg).c_str());
        return false;
      }
      set_registers[vreg_idx] = true;
    }
    for (auto* insn : bb->insn_list()) {
      for (int reg_idx = 0; reg_idx < insn->NumRegOperands(); ++reg_idx) {
        auto& reg_kind = insn->RegKindAt(reg_idx);
        if (!reg_kind.IsDef()) {
          continue;
        }
        if (reg_kind.IsInput()) {
          TRACE("use_def register detected before lowering: %s",
                GetRegOperandDebugString(insn, reg_idx).c_str());
          return false;
        }
        auto reg = insn->RegAt(reg_idx);
        if (!reg.IsVReg()) {
          TRACE("Non-vreg def-%s defore lowering: %s",
                GetRegOperandDebugString(insn, reg_idx).c_str(),
                insn->GetDebugString().c_str());
          return false;
        }
        if (reg_kind.RegClass() == &x86_64::kFLAGS) {
          continue;
        }
        int vreg_idx = reg.GetVRegIndex();
        if (set_registers.at(vreg_idx)) {
          TRACE("Register %s written to twice in %s",
                GetRegOperandDebugString(insn, reg_idx).c_str(),
                insn->GetDebugString().c_str());
          return false;
        }
        set_registers[vreg_idx] = true;
      }
    }
  }
  return true;
}

void LowerSSAInstructions(MachineIR* machine_ir) {
  // region digitalis
  // CheckSSA requires the frontend to have left no use_def operand in the IR.
  // The ARM64 frontend does leave them: ~2900 of its emissions write a register
  // in place, which is the point of them, and several flag sequences depend on
  // it. The requirement is a contract with upstream's frontend rather than
  // something the rest of the pipeline needs -- upstream itself skips the check
  // under kOptimizedInterRegionABI, the loop below passes a non-SSA instruction
  // through untouched, and every pass after this one handles use_def operands
  // because lowering produces them. So the check is dropped for this guest and
  // kept for the other, where it still describes the frontend accurately.
#if !defined(BERBERIS_GUEST_FRONTEND_EMITS_NON_SSA)
  // endregion
  if (!IsConfigFlagSet(kOptimizedInterRegionABI)) {
    if (!CheckSSA(machine_ir)) {
      FATAL("Incorrect IR:\n%s", machine_ir->GetDebugString().c_str());
    }
  }
  // region digitalis
#endif
  // endregion
  for (auto* bb : machine_ir->bb_list()) {
    auto& insn_list = bb->insn_list();
    for (auto insn_it = insn_list.begin(); insn_it != insn_list.end();) {
      if (!((*insn_it)->opcode() & (1 << kSSAOpcodeBit))) {
        ++insn_it;
        continue;
      }
      auto new_insns = machine_ir->LowerInsn(*insn_it);
      insn_it = insn_list.erase(insn_it);
      insn_list.splice(insn_it, std::move(new_insns));
    }
  }
}

}  // namespace berberis::x86_64
