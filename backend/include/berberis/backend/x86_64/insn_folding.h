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
#include "berberis/guest_state/guest_state.h"

namespace berberis::x86_64 {

enum class FoldingType { kImpossible, kReplaceInsn, kInsertInsn, kRemoveInsn };

// The DefMap class stores a map between registers and their latest definitions and positions.
// It also contains the index of the last insn which accessed memory.
class DefMap {
 public:
  DefMap(size_t size, Arena* arena)
      : def_map_(size, {std::nullopt, 0, 0}, arena),
        flags_reg_(kInvalidMachineReg),
        index_(0),
        last_context_write_insn_(0) {}
  [[nodiscard]] std::tuple<std::optional<MachineInsnList::iterator>, int, int> Get(
      MachineReg reg) const {
    if (!reg.IsVReg()) {
      return {std::nullopt, 0, 0};
    }
    auto [def_insn, def_insn_index, reg_pos] = def_map_.at(reg.GetVRegIndex());
    if (!def_insn) {
      return {std::nullopt, 0, 0};
    }
    return {def_insn, def_insn_index, reg_pos};
  }
  [[nodiscard]] std::tuple<std::optional<MachineInsnList::iterator>, int, int> Get(
      MachineReg reg,
      int use_index) const {
    if (!reg.IsVReg()) {
      return {std::nullopt, 0, 0};
    }
    auto [def_insn, def_insn_index, reg_pos] = def_map_.at(reg.GetVRegIndex());
    if (!def_insn || def_insn_index >= use_index) {
      return {std::nullopt, 0, 0};
    }
    return {def_insn, def_insn_index, reg_pos};
  }
  [[nodiscard]] bool IsContextReadActive(int context_read_pos) {
    if (last_context_write_insn_ == 0) {
      return true;
    }
    CHECK_NE(last_context_write_insn_, context_read_pos);
    return last_context_write_insn_ < context_read_pos;
  }
  void ProcessInsn(MachineInsnList::iterator insn_it);
  void Initialize();
  std::tuple<std::optional<MachineInsnList::iterator>, int, int> FindNonPseudoCopyDef(
      MachineReg src_reg) const;

 private:
  void Set(MachineReg reg, MachineInsnList::iterator insn_it, int reg_pos) {
    if (reg.IsVReg()) {
      def_map_.at(reg.GetVRegIndex()) = std::tuple(insn_it, index_, reg_pos);
    }
  }
  void MapDefRegs(MachineInsnList::iterator insn_it);

  // Each tuple in the def_map_ vector contains:
  // - The iterator to the instruction that defines the register
  // - The index of the instruction that defines the register
  // - The position of the register in the instruction that defines it
  ArenaVector<std::tuple<std::optional<MachineInsnList::iterator>, int, int>> def_map_;
  MachineReg flags_reg_;
  int index_;
  int last_context_write_insn_;
};

// Tracks the usage of values read from the guest CPU state within a basic block.
// For each guest CPU state offset, it counts how many times a value loaded from that offset is
// used by subsequent instructions.
class ContextAccessInfo {
 public:
  ContextAccessInfo(size_t size, Arena* arena)
      : reg_to_offset_map_(size, std::nullopt, arena),
        context_read_usage_map_(sizeof(CPUState), {0}, arena) {}

  [[nodiscard]] uint32_t GetContextReadUsageCount(uint32_t disp) const {
    return context_read_usage_map_.at(disp);
  }

  void ProcessInsn(const berberis::MachineInsn* insn);
  void Initialize(const MachineInsnList& insn_list);

 private:
  [[nodiscard]] std::optional<uint32_t> GetOffset(MachineReg reg) const {
    if (!reg.IsVReg()) {
      return {std::nullopt};
    }
    return reg_to_offset_map_.at(reg.GetVRegIndex());
  }

  void MapRegToOffset(MachineReg reg, uint32_t offset) {
    if (!reg.IsVReg()) {
      return;
    }
    reg_to_offset_map_.at(reg.GetVRegIndex()) = offset;
  }

  void UnmapReg(MachineReg reg) {
    if (!reg.IsVReg()) {
      return;
    }
    reg_to_offset_map_.at(reg.GetVRegIndex()) = std::nullopt;
  }

  void IncrementContextReadUsageCount(uint32_t disp) { context_read_usage_map_.at(disp)++; }

  void HandleRegisterUse(const berberis::MachineInsn* insn, MachineReg reg);

  void HandleRegisterDef(const berberis::MachineInsn* insn, MachineReg reg);

  // reg_to_offset_map_[i] contains the offset of the context read stored in register i, or
  // nullopt if the register is unwritten or contains a value that isn't the result of a context
  // read.
  ArenaVector<std::optional<uint32_t>> reg_to_offset_map_;

  // context_read_usage_map_[i] stores the number of times the value loaded from
  // kCPUStatePointer + (8 * i) was used in an instruction.
  ArenaVector<uint32_t> context_read_usage_map_;
};

class InsnFolding {
 public:
  explicit InsnFolding(DefMap& def_map,
                       ContextAccessInfo& context_access_info,
                       MachineIR* machine_ir)
      : def_map_(def_map), context_access_info_(context_access_info), machine_ir_(machine_ir) {}

  std::tuple<FoldingType, berberis::MachineInsn*> TryFoldInsn(const MachineInsnList::iterator insn,
                                                              const MachineBasicBlock* bb);

  std::tuple<FoldingType, berberis::MachineInsn*> TryFoldContextReadForTesting(
      const MachineInsnList::iterator insn) {
    return TryFoldContextRead(insn);
  }

 private:
  DefMap& def_map_;
  ContextAccessInfo& context_access_info_;
  MachineIR* machine_ir_;
  std::optional<uint64_t> GetImmValueIfPossible(MachineReg reg) const;
  bool IsWritingSameFlagsValue(MachineInsnList::iterator insn_it) const;
  template <bool kIsInput64Bit>
  std::tuple<FoldingType, berberis::MachineInsn*> TryFoldImmediateInput(
      MachineInsnList::iterator insn_it);
  std::tuple<FoldingType, berberis::MachineInsn*> TryFoldTwoImmediates(
      MachineInsnList::iterator insn_it);
  std::tuple<FoldingType, berberis::MachineInsn*> TryFoldRedundantMovl(
      MachineInsnList::iterator insn_it);
  template <bool kBMI, bool kIsInput64Bit>
  std::tuple<FoldingType, berberis::MachineInsn*> TryFoldCountLeadingZeros(
      MachineInsnList::iterator insn_it,
      const MachineBasicBlock* bb);
  std::tuple<FoldingType, berberis::MachineInsn*> TryFoldContextRead(
      const MachineInsnList::iterator insn);
  berberis::MachineInsn* NewImmInsnFromRegInsn(const berberis::MachineInsn* insn, int32_t imm);
  berberis::MachineInsn* NewInsnFromTwoImmediatesOperation(const berberis::MachineInsn* insn,
                                                           uint64_t imm1,
                                                           uint64_t imm2);
  berberis::MachineInsn* NewArithmeticInsnWithFoldedContextRead(
      const berberis::MachineInsn* insn,
      const berberis::MachineInsn* read_context_insn);
};

MachineInsnList::iterator ExecuteInsnFold(MachineInsnList& insn_list,
                                          MachineInsnList::iterator folded_insn_it,
                                          berberis::MachineInsn* new_insn,
                                          FoldingType folding_type);

void FoldInsns(MachineIR* machine_ir);

void FoldWriteFlags(MachineIR* machine_ir);

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_INSN_FOLDING_H_
