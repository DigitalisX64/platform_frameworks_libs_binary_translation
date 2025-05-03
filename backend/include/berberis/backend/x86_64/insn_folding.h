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

#ifndef BERBERIS_BACKEND_X86_64_INSN_FOLDING_H_
#define BERBERIS_BACKEND_X86_64_INSN_FOLDING_H_

#include <tuple>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

// The DefMap class stores a map between registers and their latest definitions and positions.
class DefMap {
 public:
  DefMap(size_t size, Arena* arena)
      : def_map_(size, {std::nullopt, 0}, arena), flags_reg_(kInvalidMachineReg), index_(0) {}
  [[nodiscard]] std::pair<std::optional<MachineInsnList::iterator>, int> Get(MachineReg reg) const {
    if (!reg.IsVReg()) {
      return {std::nullopt, 0};
    }
    auto [def_insn, def_insn_index] = def_map_.at(reg.GetVRegIndex());
    if (!def_insn) {
      return {std::nullopt, 0};
    }
    return {def_insn, def_insn_index};
  }
  [[nodiscard]] std::pair<std::optional<MachineInsnList::iterator>, int> Get(MachineReg reg,
                                                                             int use_index) const {
    if (!reg.IsVReg()) {
      return {std::nullopt, 0};
    }
    auto [def_insn, def_insn_index] = def_map_.at(reg.GetVRegIndex());
    if (!def_insn || def_insn_index >= use_index) {
      return {std::nullopt, 0};
    }
    return {def_insn, def_insn_index};
  }
  void ProcessInsn(MachineInsnList::iterator insn_it);
  void Initialize();

 private:
  void Set(MachineReg reg, MachineInsnList::iterator insn_it) {
    if (reg.IsVReg()) {
      def_map_.at(reg.GetVRegIndex()) = std::pair(insn_it, index_);
    }
  }
  void MapDefRegs(MachineInsnList::iterator insn_it);
  ArenaVector<std::pair<std::optional<MachineInsnList::iterator>, int>> def_map_;
  MachineReg flags_reg_;
  int index_;
};

class InsnFolding {
 public:
  explicit InsnFolding(DefMap& def_map, MachineIR* machine_ir)
      : def_map_(def_map), machine_ir_(machine_ir) {}

  std::tuple<bool, MachineInsn*> TryFoldInsn(const MachineInsnList::iterator insn,
                                             const MachineBasicBlock* bb);

 private:
  DefMap& def_map_;
  MachineIR* machine_ir_;
  bool IsRegImm(MachineReg reg, uint64_t* imm) const;
  bool IsWritingSameFlagsValue(MachineInsnList::iterator insn_it) const;
  std::tuple<std::optional<MachineInsnList::iterator>, int> FindNonPseudoCopyDef(
      MachineReg src_reg) const;
  template <bool is_input_64bit>
  std::tuple<bool, MachineInsn*> TryFoldImmediateInput(MachineInsnList::iterator insn_it);
  std::tuple<bool, MachineInsn*> TryFoldRedundantMovl(MachineInsnList::iterator insn_it);
  std::tuple<bool, MachineInsn*> TryFoldCountLeadingZeroes(MachineInsnList::iterator insn_it,
                                                           const MachineBasicBlock* bb);
  MachineInsn* NewImmInsnFromRegInsn(const MachineInsn* insn, int32_t imm);
};

void FoldInsns(MachineIR* machine_ir);

void FoldWriteFlags(MachineIR* machine_ir);

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_INSN_FOLDING_H_
