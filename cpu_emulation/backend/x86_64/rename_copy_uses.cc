/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include "berberis/backend/x86_64/rename_copy_uses.h"

#include <list>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/algorithm.h"
#include "berberis/base/checks.h"

namespace berberis::x86_64 {

MachineReg RenameCopyUsesMap::Get(MachineReg reg) const {
  MachineReg renamed = RenameDataForReg(reg).renamed;
  if (renamed == kInvalidMachineReg) {
    return kInvalidMachineReg;
  }
  if (RenameDataForReg(renamed).last_def_time > RenameDataForReg(reg).renaming_time) {
    return kInvalidMachineReg;
  }
  return renamed;
}

void RenameCopyUsesMap::RenameUseIfMapped(berberis::MachineInsn* insn, int i) {
  if (insn->RegKindAt(i).RegClass()->NumRegs() == 1) {
    return;
  }
  MachineReg reg = insn->RegAt(i);
  if (!reg.IsVReg()) {
    return;
  }
  // Renaming is only possible for USE without DEF.
  MachineReg mapped = Get(reg);
  if (mapped != kInvalidMachineReg) {
    insn->SetRegAt(i, mapped);
  }
}

void RenameCopyUsesMap::ProcessDef(berberis::MachineInsn* insn, int i) {
  MachineReg reg = insn->RegAt(i);

  if (!reg.IsVReg()) {
    return;
  }

  RenameDataForReg(reg) = {kInvalidMachineReg, 0, time_};
}

void RenameCopyUsesMap::ProcessCopy(berberis::MachineInsn* copy) {
  auto dst = copy->RegAt(0);
  auto src = copy->RegAt(1);
  if (!dst.IsVReg() || !src.IsVReg()) {
    return;
  }

  if (Contains(bb_->live_out(), dst) && !Contains(bb_->live_out(), src)) {
    // If dst is a live-out then renaming it to src won't help eliminate the copy. So instead we
    // rename src to dst.
    // When dst and src are both live-out, we choose to rename dst to src. This means that when
    // either dst or src is copied to another live-out register, say dst2, both dst and dst2 will
    // be renamed to src by the global renamer.
    RenameDataForReg(src).renamed = dst;
    RenameDataForReg(src).renaming_time = time_;
  } else {
    RenameDataForReg(dst) = {src, time_, time_};
  }
}

void RenameCopyUsesMap::StartBasicBlock(MachineBasicBlock* bb) {
  bb_ = bb;
  for (auto& data : map_) {
    data = {kInvalidMachineReg, 0, 0};
  }
}

void DuplicateLiveOutsMap::SaveDuplicates(int bb_id, MachineReg reg1, MachineReg reg2) {
  CHECK(reg1.IsVReg() && reg2.IsVReg());
  CHECK_NE(reg1.GetVRegIndex(), reg2.GetVRegIndex());
  duplicate_live_outs_map_.at(bb_id).at(reg1.GetVRegIndex()) = reg2;
  bb_contains_duplicates_map_.at(bb_id) = true;
}

void DuplicateLiveOutsMap::ProcessDef(const berberis::MachineBasicBlock* bb,
                                      const berberis::MachineInsn* insn,
                                      int i) {
  MachineReg reg = insn->RegAt(i);
  if (!reg.IsVReg()) {
    return;
  }
  registers_defined_map_.at(bb->id()).at(reg.GetVRegIndex()) = true;
}

void DuplicateLiveOutsMap::RenameAndRemoveDuplicateLiveIns(
    berberis::MachineIR* machine_ir,
    berberis::MachineBasicBlock* bb,
    const berberis::MachineBasicBlock* prev_bb) {
  ArenaVector<bool> bb_renamed_live_in_registers_map(
      machine_ir->NumVReg(), false, machine_ir->arena());
  ArenaVector<bool> live_ins_map(machine_ir->NumVReg(), false, machine_ir->arena());
  for (auto live_in : bb->live_in()) {
    live_ins_map.at(live_in.GetVRegIndex()) = true;
  }
  bool live_in_register_renaming_occurred = false;
  for (auto* insn : bb->insn_list()) {
    for (int i = 0; i < insn->NumRegOperands(); ++i) {
      auto reg = insn->RegAt(i);
      if (Contains(bb->live_out(), reg)) {
        continue;
      }
      auto mapped_reg = GetDuplicate(prev_bb, reg);
      if (mapped_reg.IsInvalidReg()) {
        continue;
      }
      if (BasicBlockDefinesRegister(bb, reg) || BasicBlockDefinesRegister(bb, mapped_reg)) {
        continue;
      }
      CHECK(live_ins_map.at(reg.GetVRegIndex()));
      // 'mapped_reg' may not live-in into this block and only be used in another successor of
      // 'prev_bb'. In this case, we can introduce it as a new live_in register to replace reg.
      if (!live_ins_map.at(mapped_reg.GetVRegIndex())) {
        bb->live_in().push_back(mapped_reg);
        live_ins_map.at(mapped_reg.GetVRegIndex()) = true;
      }
      live_in_register_renaming_occurred = true;
      bb_renamed_live_in_registers_map.at(reg.GetVRegIndex()) = true;
      insn->SetRegAt(i, mapped_reg);
    }
  }
  if (!live_in_register_renaming_occurred) {
    return;
  }
  auto& live_in = bb->live_in();
  std::erase_if(live_in, [&](MachineReg live_in_reg) {
    return bb_renamed_live_in_registers_map.at(live_in_reg.GetVRegIndex());
  });
}

void ComputeDuplicateLiveOuts(MachineIR* machine_ir,
                              MachineBasicBlock* bb,
                              const RenameCopyUsesMap* rename_copy_uses_map,
                              DuplicateLiveOutsMap* duplicate_live_outs_map) {
  ArenaVector<bool> bb_live_outs_set(machine_ir->NumVReg(), false, machine_ir->arena());
  for (auto reg : bb->live_out()) {
    CHECK(reg.IsVReg());
    bb_live_outs_set.at(reg.GetVRegIndex()) = true;
  }
  for (auto reg : bb->live_out()) {
    MachineReg mapped = rename_copy_uses_map->Get(reg);
    if (mapped.IsInvalidReg()) {
      continue;
    }
    if (bb_live_outs_set.at(mapped.GetVRegIndex())) {
      // We should never see a reg mapped to a mapped reg
      CHECK(rename_copy_uses_map->Get(mapped).IsInvalidReg());
      // Note that bb->live_out() can contain multiple instances of the same vreg, and so
      // SaveDuplicates may be called multiple times for the same pair, but SaveDuplicates is
      // implemented such that this is harmless.
      duplicate_live_outs_map->SaveDuplicates(bb->id(), reg, mapped);
    }
  }
}

void RemoveRedundantLiveOuts(MachineBasicBlock* bb) {
  auto& live_out = bb->live_out();
  std::erase_if(live_out, [&](MachineReg live_out_reg) {
    for (auto* edge : bb->out_edges()) {
      if (Contains(edge->dst()->live_in(), live_out_reg)) {
        return false;
      }
    }
    return true;
  });
}

void RenameCopyUsesInBasicBlock(MachineBasicBlock* bb,
                                RenameCopyUsesMap* rename_copy_uses_map,
                                DuplicateLiveOutsMap* duplicate_live_outs_map) {
  rename_copy_uses_map->StartBasicBlock(bb);
  for (auto* insn : bb->insn_list()) {
    // region digitalis
    // An instruction reads its uses before it writes its defs, so rename every use
    // first: otherwise a def that comes earlier in the operand list (`pmovsxbw x, x`
    // after `x = copy y`) drops the mapping its own use still needs, and the copy
    // stays. Upstream still interleaves both in one pass.
    for (int i = 0; i < insn->NumRegOperands(); ++i) {
      // Note that Def-Use operands cannot be renamed, so we handle them as Defs.
      if (!insn->RegKindAt(i).IsDef()) {
        rename_copy_uses_map->RenameUseIfMapped(insn, i);
      }
    }  // for use operand in insn
    for (int i = 0; i < insn->NumRegOperands(); ++i) {
      if (insn->RegKindAt(i).IsDef()) {
        rename_copy_uses_map->ProcessDef(insn, i);
        duplicate_live_outs_map->ProcessDef(bb, insn, i);
      }
    }  // for def operand in insn
    // endregion

    // Note that we intentionally rename copy's use before attempting to create a mapping, so that
    // the existing mappings are applied and propagated further.
    if (insn->is_copy()) {
      rename_copy_uses_map->ProcessCopy(insn);
    }
    rename_copy_uses_map->Tick();
  }  // For insn in bb
}

void RenameCopyUses(MachineIR* machine_ir) {
  RenameCopyUsesMap rename_copy_uses_map(machine_ir);
  DuplicateLiveOutsMap duplicate_live_outs_map(machine_ir);

  for (auto* bb : machine_ir->bb_list()) {
    RenameCopyUsesInBasicBlock(bb, &rename_copy_uses_map, &duplicate_live_outs_map);
    ComputeDuplicateLiveOuts(machine_ir, bb, &rename_copy_uses_map, &duplicate_live_outs_map);
  }

  for (auto* bb : machine_ir->bb_list()) {
    // This implementation only applies the optimization to basic blocks which have exactly one
    // predecessor. This is because extending this optimization to handle basic blocks with
    // multiple predecessors increases translation overhead, and we have never seen a case of a
    // basic block with multiple predecessors which can be optimized this way.
    // See b/448293427#comment16 for more details.
    if (bb->in_edges().size() != 1 || bb->live_in().size() < 1) {
      continue;
    }
    auto prev_bb = bb->in_edges().at(0)->src();
    if (!duplicate_live_outs_map.BasicBlockContainsDuplicateLiveOuts(prev_bb)) {
      continue;
    }
    duplicate_live_outs_map.RenameAndRemoveDuplicateLiveIns(machine_ir, bb, prev_bb);
  }

  for (auto* bb : machine_ir->bb_list()) {
    if (!duplicate_live_outs_map.BasicBlockContainsDuplicateLiveOuts(bb)) {
      continue;
    }
    RemoveRedundantLiveOuts(bb);
  }
}

}  // namespace berberis::x86_64
