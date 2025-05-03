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

#ifndef BERBERIS_BACKEND_X86_64_READ_FLAGS_OPTIMIZER_H_
#define BERBERIS_BACKEND_X86_64_READ_FLAGS_OPTIMIZER_H_

#include "berberis/backend/common/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_analysis.h"
#include "berberis/base/arena_map.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

using InsnGenerator = MachineInsn* (*)(MachineIR*, MachineInsn*);

struct ReadFlagsOptContext {
  MachineBasicBlock* bb;
  // Original readflag instruction.
  MachineInsnList::iterator readflags_insn;
  // Original instruction that set flag register.
  MachineInsnList::iterator flag_set_insn;
};

bool CheckRegsUnusedWithinInsnRange(MachineInsnList::iterator insn_it,
                                    MachineInsnList::iterator end,
                                    MachineRegVector& regs);
bool CheckPostLoopNode(MachineBasicBlock* block, const MachineRegVector& regs);
bool CheckSuccessorNode(Loop* loop, MachineBasicBlock* block, MachineRegVector& regs);
std::optional<InsnGenerator> GetInsnGen(MachineOpcode opcode);
std::optional<MachineInsnList::iterator> FindFlagSettingInsn(MachineInsnList::iterator insn_it,
                                                             MachineInsnList::iterator begin,
                                                             MachineReg reg);
void InsertFlagGenInstructions(MachineIR* machine_ir,
                               ReadFlagsOptContext& context,
                               MachineInsnList::iterator insn_it,
                               const ArenaMap<MachineReg, MachineReg>& reg_map,
                               MachineReg reg);
std::optional<MachineInsnList::iterator> IsEligibleReadFlag(MachineIR* machine_ir,
                                                            Loop* loop,
                                                            MachineBasicBlock* bb,
                                                            MachineInsnList::iterator insn_it);
std::optional<MachineReg> NeedsToSaveFlags(MachineBasicBlock* bb,
                                           MachineInsnList::iterator insn_it);
void OptimizeReadFlags(MachineIR* machine_ir);
bool RegsLiveInBasicBlock(MachineBasicBlock* bb, const MachineRegVector& regs);
void RemoveEligibleReadFlagsInLoopTree(MachineIR* machine_ir, LoopTreeNode* loop_tree_node);
void RemoveReadFlags(MachineIR* machine_ir, ReadFlagsOptContext context);
bool RemoveRegs(MachineRegVector& remove_from_regs, const MachineRegVector& regs_to_remove);
// Note flags_regs must not be a reference because we update it with new flag
// registers based on our current basic block, but they are only applicable to
// the current and future basic blocks.
void ReplaceFlagRegisters(MachineIR* machine_ir,
                          ReadFlagsOptContext context,
                          MachineInsnList::iterator insn_it,
                          MachineRegVector flags_regs,
                          const ArenaMap<MachineReg, MachineReg>& reg_map,
                          MachineInsn* insn);

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_READ_FLAGS_OPTIMIZER_H_
