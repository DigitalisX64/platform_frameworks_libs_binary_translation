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

#ifndef BERBERIS_BACKEND_X86_64_LOCAL_GUEST_CONTEXT_OPTIMIZER_H_
#define BERBERIS_BACKEND_X86_64_LOCAL_GUEST_CONTEXT_OPTIMIZER_H_

#include <cstddef>
#include <optional>
#include <variant>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/reg_lifetime.h"
#include "berberis/base/arena_vector.h"
#include "berberis/base/config_globals.h"

namespace berberis::x86_64 {

struct OptimizeLocalParams {
  size_t general_reg_limit = 13;
  size_t simd_reg_limit = 13;
};

struct PredecessorReg {
  MachineReg reg;
};

using MappedValue = std::variant<MachineReg, uint64_t, PredecessorReg>;
// region digitalis
// Bytes of the CPU-state slot a guest-context access actually touches, and
// whether it goes through an SIMD register. mem_reg_map is keyed by offset
// alone, so a mapping is only interchangeable with an access of the same
// register class that touches no more bytes than the mapping recorded.
struct ContextAccess {
  size_t size;
  bool is_simd;
};
// endregion
struct MappedRegUsage {
  MappedValue value;
  std::optional<MachineInsnList::iterator> last_store;
  // region digitalis
  ContextAccess access;
  // endregion
};

using MemRegUsageMap = ArenaVector<std::optional<MappedRegUsage>>;

class LocalGuestContextOptimizer {
 public:
  explicit LocalGuestContextOptimizer(x86_64::MachineIR* machine_ir,
                                      const OptimizeLocalParams& params)
      : machine_ir_(machine_ir),
        context_map_(
            ContextMapping{MemRegUsageMap(sizeof(CPUState), std::nullopt, machine_ir->arena()),
                           RegLifetimeCounter(machine_ir)}),
        params_(params),
        global_mappings_(machine_ir->arena()),
        offsets_used_reservation_(machine_ir->arena()) {
    params_.general_reg_limit =
        machine_ir_->abi() == MachineIR::ABI::kOptimizedEnabled
            ? (params_.general_reg_limit >= 6 ? params_.general_reg_limit - 6 : 0UL)
            : params_.general_reg_limit;
    offsets_used_reservation_.reserve(sizeof(CPUState));
    global_mappings_.resize(machine_ir->NumBasicBlocks(),
                            ArenaVector<DispRegMapping>(machine_ir->arena()));
  }

  // Removes entries from mem_reg_map with last use <= pos.
  void UnmapOlderThan(size_t pos, RegType reg_type_to_unmap);
  const RegLifetimeCounter& GetLifetimeCounterForTesting() const {
    return context_map_.reg_counter;
  }
  const MemRegUsageMap& GetMemRegUsageMapForTesting() const { return context_map_.mem_reg_map; }
  void RemoveLocalGuestContextAccesses();

 private:
  struct ContextMapping {
    MemRegUsageMap mem_reg_map;
    RegLifetimeCounter reg_counter;
  };
  struct DispRegMapping {
    uint32_t disp;
    std::variant<MachineReg, uint64_t> value;
  };

  bool EligibleForGlobalOpt(const MachineBasicBlockList& nonloop_nodes, MachineBasicBlock* bb);
  void InitMemRegMapFromPreds(MachineBasicBlock* bb);
  void PopulateOffsetsUsed(MachineBasicBlock* bb, ArenaVector<uint32_t>& offsets);
  std::optional<MachineReg> ReplaceGetAndUpdateMap(MachineBasicBlock* bb,
                                                   const MachineInsnList::iterator insn_it);
  void ReplacePutAndUpdateMap(MachineBasicBlock* bb, const MachineInsnList::iterator insn_it);
  void PrepareGlobalMappingsForSuccessors(MachineBasicBlock* bb,
                                          const MachineBasicBlockList& nonloop_nodes);
  // Updates register lifetimes and mem_reg_map when a register's lifetime is extended.
  void UpdateRegLastUse(MachineReg reg, size_t new_pos, uint32_t offset);

  MachineIR* machine_ir_;
  ContextMapping context_map_;
  OptimizeLocalParams params_;
  // These are the mappings available for global context optimization. They
  // map from bb.id -> [DispRegMapping]
  ArenaVector<ArenaVector<DispRegMapping>> global_mappings_;
  // Kept as member variable as optimization to prevent reallocations.
  ArenaVector<uint32_t> offsets_used_reservation_;
};

void RemoveLocalGuestContextAccesses(x86_64::MachineIR* machine_ir,
                                     const OptimizeLocalParams& params = OptimizeLocalParams());

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_LOCAL_GUEST_CONTEXT_OPTIMIZER_H_
