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

#include "berberis/backend/x86_64/local_guest_context_optimizer.h"

#include <optional>

#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

namespace {

using OffsetCounterMap = ArenaVector<std::pair<size_t, int>>;

class LocalGuestContextOptimizer {
 public:
  explicit LocalGuestContextOptimizer(x86_64::MachineIR* machine_ir)
      : machine_ir_(machine_ir),
        mem_reg_map_(sizeof(CPUState), std::nullopt, machine_ir->arena()) {}

  void RemoveLocalGuestContextAccesses(const OptimizeLocalParams& params);

 private:
  struct MappedRegUsage {
    MachineReg reg;
    std::optional<MachineInsnList::iterator> last_store;
  };

  void ReplaceGetAndUpdateMap(const MachineInsnList::iterator insn_it);
  void ReplacePutAndUpdateMap(MachineInsnList& insn_list, const MachineInsnList::iterator insn_it);

  MachineIR* machine_ir_;
  ArenaVector<std::optional<MappedRegUsage>> mem_reg_map_;
};

ArenaVector<int> CountGuestRegAccesses(const MachineIR* ir, MachineBasicBlock* bb) {
  ArenaVector<int> guest_access_count(sizeof(CPUState), 0, ir->arena());
  for (auto* base_insn : bb->insn_list()) {
    auto insn = AsMachineInsnX86_64(base_insn);
    if (insn->IsCPUStateGet() || insn->IsCPUStatePut()) {
      guest_access_count.at(insn->disp())++;
    }
  }
  return guest_access_count;
}

OffsetCounterMap GetSortedOffsetCounters(MachineIR* ir, MachineBasicBlock* bb) {
  auto guest_access_count = CountGuestRegAccesses(ir, bb);

  OffsetCounterMap offset_counter_map(ir->arena());
  for (size_t offset = 0; offset < sizeof(CPUState); offset++) {
    int cnt = guest_access_count.at(offset);
    if (cnt > 0) {
      offset_counter_map.push_back({offset, cnt});
    }
  }

  std::sort(offset_counter_map.begin(), offset_counter_map.end(), [](auto pair1, auto pair2) {
    return std::get<1>(pair1) > std::get<1>(pair2);
  });

  return offset_counter_map;
}

void LocalGuestContextOptimizer::RemoveLocalGuestContextAccesses(
    const OptimizeLocalParams& params) {
  for (auto* bb : machine_ir_->bb_list()) {
    std::fill(mem_reg_map_.begin(), mem_reg_map_.end(), std::nullopt);

    auto sorted_offsets = GetSortedOffsetCounters(machine_ir_, bb);
    ArenaVector<bool> optimized_offsets(sizeof(CPUState), false, machine_ir_->arena());

    size_t general_reg_count = 0;
    size_t simd_reg_count = 0;
    for (auto [offset, unused_counter] : sorted_offsets) {
      // TODO(b/232598137): Account for f and v register classes.
      // Simd regs.
      if (IsSimdOffset(offset)) {
        if (simd_reg_count++ < params.simd_reg_limit) {
          optimized_offsets[offset] = true;
        }
        continue;
      }
      // General regs and flags.
      if (general_reg_count++ < params.general_reg_limit) {
        optimized_offsets[offset] = true;
      }
    }

    for (auto insn_it = bb->insn_list().begin(); insn_it != bb->insn_list().end(); insn_it++) {
      auto* insn = AsMachineInsnX86_64(*insn_it);

      // Skip insn if it accesses regs with low priority
      if (insn->IsCPUStateGet() || insn->IsCPUStatePut()) {
        if (!optimized_offsets.at(insn->disp())) {
          continue;
        }
      }

      if (insn->IsCPUStateGet()) {
        ReplaceGetAndUpdateMap(insn_it);
      } else if (insn->IsCPUStatePut()) {
        ReplacePutAndUpdateMap(bb->insn_list(), insn_it);
      }
    }
  }
}

void LocalGuestContextOptimizer::ReplaceGetAndUpdateMap(const MachineInsnList::iterator insn_it) {
  auto* insn = AsMachineInsnX86_64(*insn_it);
  auto dst = insn->RegAt(0);
  auto disp = insn->disp();

  // We only need to keep this load instruction if this is the first access to
  // the guest context at disp.
  if (!mem_reg_map_[disp].has_value()) {
    mem_reg_map_[disp] = {dst, {}};
    return;
  }

  auto copy_size = insn->opcode() == kMachineOpMovdqaXRegMemBaseDisp ? 16 : 8;
  *insn_it = machine_ir_->NewInsn<PseudoCopy>(dst, mem_reg_map_[disp].value().reg, copy_size);
}

void LocalGuestContextOptimizer::ReplacePutAndUpdateMap(MachineInsnList& insn_list,
                                                        const MachineInsnList::iterator insn_it) {
  auto* insn = AsMachineInsnX86_64(*insn_it);
  auto src = insn->RegAt(1);
  auto disp = insn->disp();

  if (mem_reg_map_[disp].has_value() && mem_reg_map_[disp].value().last_store.has_value()) {
    // Remove the last store instruction.
    auto last_store_it = mem_reg_map_[disp].value().last_store.value();
    insn_list.erase(last_store_it);
  }

  mem_reg_map_[disp] = {src, {insn_it}};
}

}  // namespace

void RemoveLocalGuestContextAccesses(x86_64::MachineIR* machine_ir,
                                     const OptimizeLocalParams& params) {
  LocalGuestContextOptimizer optimizer(machine_ir);
  optimizer.RemoveLocalGuestContextAccesses(params);
}

}  // namespace berberis::x86_64
