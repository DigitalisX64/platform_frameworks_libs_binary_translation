/*
 * Copyright (C) 2021 The Android Open Source Project
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

#include "berberis/backend/x86_64/code_gen.h"

#include "berberis/assembler/machine_code.h"
#include "berberis/backend/code_emitter.h"
#include "berberis/backend/common/machine_ir_opt.h"
#include "berberis/backend/common/reg_alloc.h"
#include "berberis/backend/x86_64/insn_folding.h"
#include "berberis/backend/x86_64/local_guest_context_optimizer.h"
#include "berberis/backend/x86_64/loop_guest_context_optimizer.h"
#include "berberis/backend/x86_64/lower_ssa_instructions.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/backend/x86_64/machine_ir_opt.h"
#include "berberis/backend/x86_64/merge_basic_blocks.h"
#include "berberis/backend/x86_64/read_flags_optimizer.h"
// region digitalis
#include "berberis/backend/x86_64/place_blocks_before_branch_targets.h"
#include "berberis/backend/x86_64/remove_nop_reg_moves.h"
#include "berberis/backend/x86_64/use_lea_for_copied_adds.h"
#include "berberis/backend/x86_64/use_vex_three_operand_forms.h"
// endregion
#include "berberis/backend/x86_64/rename_copy_uses.h"
#include "berberis/backend/x86_64/rename_vregs.h"
#include "berberis/base/checks.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/tracing.h"
// region digitalis
#include "berberis/runtime_primitives/platform.h"
// endregion

namespace berberis::x86_64 {

void GenCode(MachineIR* machine_ir, MachineCode* machine_code, const GenCodeParams& params) {
  LowerSSAInstructions(machine_ir);
  // region digitalis
  // MachineIR validation costs a full IR walk per translation; production
  // images opt out via ro.berberis.flags=disable-ir-check, while host tests
  // (no flags configured) always validate.
  if (!IsConfigFlagSet(kDisableIrCheck)) {
    // endregion
    CHECK_EQ(CheckMachineIR(*machine_ir), kMachineIRCheckSuccess);
    // region digitalis
  }
  // endregion
  if (IsConfigFlagSet(kVerboseTranslation) || IsConfigFlagSet(kPrintIRs)) {
    TRACE("MachineIR before optimizations {\n");
    TRACE("%s", machine_ir->GetDebugString().c_str());
    TRACE("}\n\n");
  }

  if (!IsConfigFlagSet(kDisableHeavyOptimizations)) {
    // Must be called before any IR passes which introduce bb live_in's and live_out's
    MergeBasicBlocks(machine_ir);
  }

  // Required unconditionally by RenameVRegs.
  RemoveCriticalEdges(machine_ir);

  if (!IsConfigFlagSet(kDisableHeavyOptimizations)) {
    MoveColdBlocksToEnd(machine_ir);
    RemoveLoopGuestContextAccesses(machine_ir);
    // region digitalis
    // Values the local pass cannot carry past the end of a recovery split.
    ForwardGuestContextGetsAcrossRecoverySplits(machine_ir);
    // endregion digitalis
  }

  // Required unconditionally for the correct work of AllocRegs.
  RenameVRegs(machine_ir);

  if (!IsConfigFlagSet(kDisableHeavyOptimizations)) {
    RemoveLocalGuestContextAccesses(machine_ir);
    RemoveRedundantPut(machine_ir);
    FoldInsns(machine_ir);
    // Call this after all phases that create copy instructions.
    // region digitalis
    // Skipped for this guest. RenameCopyUses maps a copy's destination to its
    // source and requires that no mapped register is itself mapped --
    // ComputeDuplicateLiveOuts CHECKs it and then feeds the pair to
    // SaveDuplicates, so a chain would not merely fail validation, it would
    // record the wrong duplicate. The ARM64 frontend produces such chains
    // because its IR mixes forms: 443 emissions go through
    // MachineIRBuilder::GenMachineInsn, which inserts copies of its own, while
    // ~2900 write their register in place, and the frontend copies guest values
    // itself before mutating them. Upstream's frontend emits one uniform SSA
    // form and never builds a chain.
    //
    //   rename_copy_uses.cc:168: CHECK failed:
    //       rename_copy_uses_map->Get(mapped).IsInvalidReg()
    //
    // reached from HeavyOptimizeRegion on a hot VLC H.264 decode loop.
    //
    // This costs an optimization, not correctness: without the pass the copies
    // it would have renamed away simply stay, so the code is bigger and a little
    // slower. Making the frontend emit one form would let the pass run again and
    // is the real fix; it is a codegen change for the benchmark gate to drive.
#if !defined(BERBERIS_GUEST_FRONTEND_EMITS_NON_SSA)
    // endregion
    RenameCopyUses(machine_ir);
    // region digitalis
#endif
    // endregion
    RemoveDeadCode(machine_ir);
    OptimizeReadFlags(machine_ir);
    // region digitalis
    if (host_platform::kHasAVX) {
      UseVexThreeOperandForms(machine_ir);
    }
    UseLeaForCopiedAdds(machine_ir);
    // endregion
  }

  AllocRegs(machine_ir);

  if (!IsConfigFlagSet(kDisableHeavyOptimizations)) {
    RemoveNopCopyAndMovq(machine_ir);
        // region digitalis
    RemoveNopRegMoves(machine_ir);
    // endregion
    x86_64::RemoveForwarderBlocks(machine_ir);
    // region digitalis
    PlaceBlocksBeforeBranchTargets(machine_ir);
    // endregion
  }

  // region digitalis
  if (!IsConfigFlagSet(kDisableIrCheck)) {
    // endregion
    CHECK_EQ(CheckMachineIR(*machine_ir), kMachineIRCheckSuccess);
    // region digitalis
  }
  // endregion

  if (IsConfigFlagSet(kVerboseTranslation) || IsConfigFlagSet(kPrintIRs)) {
    TRACE("MachineIR before emit {\n");
    TRACE("%s", machine_ir->GetDebugString().c_str());
    TRACE("}\n\n");
  }

  if (IsConfigFlagSet(kPrintIRsAsDot)) {
    TRACE("MachineIR before emit as Dot {\n");
    TRACE("%s", machine_ir->GetDebugStringAsDot().c_str());
    TRACE("}\n\n");
  }

  if (!params.skip_emit) {
    CodeEmitter emitter(
        machine_code, machine_ir->FrameSize(), machine_ir->NumBasicBlocks(), machine_ir->arena());
    bool region_contains_calls = machine_ir->Emit(&emitter);
    if (region_contains_calls) {
      machine_ir->set_contains_calls();
    }
    emitter.Finalize();
  }
}

}  // namespace berberis::x86_64
