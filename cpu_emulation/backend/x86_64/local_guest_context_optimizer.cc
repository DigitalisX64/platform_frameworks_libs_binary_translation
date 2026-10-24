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

#include <bitset>
#include <cstddef>
#include <optional>
#include <variant>

#include "berberis/backend/x86_64/machine_ir_analysis.h"
#include "berberis/backend/x86_64/machine_ir_opt.h"
#include "berberis/base/algorithm.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

// region digitalis
ContextAccess GetContextAccess(const berberis::MachineInsn* insn) {
  switch (insn->opcode()) {
    case kMachineOpMovwRegMemBaseDisp:
    case kMachineOpMovwMemBaseDispReg:
      return {2, false};
    case kMachineOpMovqRegMemBaseDisp:
    case kMachineOpMovqMemBaseDispReg:
    case kMachineOpMovqMemBaseDispImm:
      return {8, false};
    case kMachineOpMovsdXRegMemBaseDisp:
    case kMachineOpMovsdMemBaseDispXReg:
      return {8, true};
    case kMachineOpMovdqaXRegMemBaseDisp:
    case kMachineOpMovdqaMemBaseDispXReg:
      return {16, true};
    default:
      // Not a recognized context access; IsCPUStateGet/Put gate every caller.
      return {16, false};
  }
}
// endregion


using OffsetCounterMap = ArenaVector<std::pair<size_t, int>>;

void LocalGuestContextOptimizer::UnmapOlderThan(size_t pos, RegType reg_type_to_unmap) {
  MemRegUsageMap& mem_reg_map = context_map_.mem_reg_map;
  RegLifetimeCounter& reg_counter = context_map_.reg_counter;
  for (auto& mapping : mem_reg_map) {
    if (!mapping.has_value()) {
      continue;
    }
    auto reg_usage = mapping.value();
    // Ignore immediates.
    if (std::holds_alternative<uint64_t>(reg_usage.value)) {
      continue;
    } else if (std::holds_alternative<PredecessorReg>(reg_usage.value)) {
      mapping = std::nullopt;
      continue;
    }

    MachineReg mapped_reg = std::get<MachineReg>(reg_usage.value);
    const auto& mapped_lifetime = reg_counter.GetLifetimeAt(mapped_reg);
    if (mapped_lifetime.reg_type != reg_type_to_unmap) {
      continue;
    }

    if (mapped_lifetime.end_pos <= pos) {
      mapping = std::nullopt;
    }
  }
}

void LocalGuestContextOptimizer::RemoveLocalGuestContextAccesses() {
  MachineBasicBlockList nonloop_nodes(machine_ir_->arena());
  // We need reverse post order for global guest context optimization.
  ReorderBasicBlocksInReversePostOrder(machine_ir_);
  nonloop_nodes = FindNonloopNodes(machine_ir_);

  for (auto* bb : machine_ir_->bb_list()) {
    MemRegUsageMap& mem_reg_map = context_map_.mem_reg_map;
    std::fill(mem_reg_map.begin(), mem_reg_map.end(), std::nullopt);
    InitMemRegMapFromPreds(bb);
    RegLifetimeCounter& reg_counter = context_map_.reg_counter;
    reg_counter.Count(bb);

    size_t pos = 0;
    for (auto insn_it = bb->insn_list().begin(); insn_it != bb->insn_list().end();
         insn_it++, pos++) {
      // If the register pressure at the current instruction is too big, then cancel
      // active mappings with lifetimes which end before current instruction.
      if (reg_counter.RegCountAt(pos, RegType::kGeneral) >= params_.general_reg_limit) {
        UnmapOlderThan(pos, RegType::kGeneral);
      }
      if (reg_counter.RegCountAt(pos, RegType::kXmm) >= params_.simd_reg_limit) {
        UnmapOlderThan(pos, RegType::kXmm);
      }

      if (machine_ir_->IsCPUStateGet(*insn_it) || machine_ir_->IsCPUStatePut(*insn_it)) {
        auto* insn = AsMachineInsnX86_64(*insn_it);

        if (machine_ir_->IsCPUStatePut(insn)) {
          // Replacing PUT doesn't prolong the lifetime of its argument. It will
          // only be prolonged if we optimize next GET.
          ReplacePutAndUpdateMap(bb, insn_it);
        } else if (machine_ir_->IsCPUStateGet(insn)) {
          std::optional<MachineReg> src_reg_opt = ReplaceGetAndUpdateMap(bb, insn_it);
          if (!src_reg_opt.has_value()) {
            continue;
          }

          // If GET replacement is successful, it means there was an allowed
          // active mapping, the lifetime for which is now prolonged, which
          // we need to reflect in reg_lifetimes_counter_.
          auto src_reg = src_reg_opt.value();
          const auto& lifetime = reg_counter.GetLifetimeAt(src_reg);
          if (lifetime.reg_type == RegType::kUnknown) {
            continue;
          }
          UpdateRegLastUse(src_reg, pos + 1, AsMachineInsnX86_64(insn)->disp());
        }
      }
    }

    PrepareGlobalMappingsForSuccessors(bb, nonloop_nodes);
  }
}

bool LocalGuestContextOptimizer::EligibleForGlobalOpt(const MachineBasicBlockList& nonloop_nodes,
                                                      MachineBasicBlock* bb) {
  if (!Contains(nonloop_nodes, bb)) {
    return false;
  }
  for (auto in_edge : bb->in_edges()) {
    MachineBasicBlock* pred = in_edge->src();
    if (!Contains(nonloop_nodes, pred)) {
      return false;
    }
  }
  return true;
}

void LocalGuestContextOptimizer::InitMemRegMapFromPreds(MachineBasicBlock* bb) {
  MemRegUsageMap& mem_reg_map = context_map_.mem_reg_map;
  if (bb->in_edges().size() == 1) {
    auto* pred_bb = bb->in_edges().front()->src();
    for (const auto& m : global_mappings_.at(pred_bb->id())) {
      if (std::holds_alternative<uint64_t>(m.value)) {
        mem_reg_map.at(m.disp) =
            MappedRegUsage{.value = std::get<uint64_t>(m.value), .last_store = std::nullopt};
      } else if (std::holds_alternative<MachineReg>(m.value)) {
        // We convert it to a predecessor reg here so we know that we need to
        // add it to live_in/out later if we use it.
        mem_reg_map.at(m.disp) = MappedRegUsage{
            .value = PredecessorReg{std::get<MachineReg>(m.value)}, .last_store = std::nullopt};
      }
      // else ignore PredecessorRegs.
    }
  }
  // TODO(b/449996703): handle multiple in_edges.
}

void LocalGuestContextOptimizer::PopulateOffsetsUsed(MachineBasicBlock* bb,
                                                     ArenaVector<uint32_t>& offsets) {
  // Keep track of registers we PUT to and those we have already seen as they should be ignored.
  std::bitset<sizeof(CPUState)> ignored;
  for (auto* insn : bb->insn_list()) {
    if (machine_ir_->IsCPUStatePut(insn)) {
      ignored.set(AsMachineInsnX86_64(insn)->disp());
    } else if (machine_ir_->IsCPUStateGet(insn)) {
      auto disp = AsMachineInsnX86_64(insn)->disp();
      CHECK_LT(disp, ignored.size());
      if (!ignored[disp]) {
        offsets.push_back(disp);
        ignored.set(disp);
      }
    }
  }
}

// Optimizes a GET instruction if possible by replacing with COPY, while
// also setting up the mapping for future optimizations. On successful
// optimization, returns the source register we made a copy from.
std::optional<MachineReg> LocalGuestContextOptimizer::ReplaceGetAndUpdateMap(
    MachineBasicBlock* bb,
    const MachineInsnList::iterator insn_it) {
  auto* insn = AsMachineInsnX86_64(*insn_it);
  auto dst = insn->RegAt(0);
  auto disp = insn->disp();
  auto& mem_reg_map = context_map_.mem_reg_map;

  // We only need to keep this load instruction if this is the first access to
  // the guest context at disp.
  // region digitalis
  // A narrower mapping cannot serve a wider read -- a 64-bit MOVSD get zeroes the
  // upper half of its XMM, so it does not hold the bits a later 128-bit read of the
  // same slot needs. Forwarding it silently zeroed a vector register's upper half
  // (seen as a wrong bcrypt key expansion with no crash), so re-map instead.
  auto access = GetContextAccess(*insn_it);
  if (!mem_reg_map[disp].has_value() || mem_reg_map[disp].value().access.size < access.size ||
      mem_reg_map[disp].value().access.is_simd != access.is_simd) {
    mem_reg_map[disp] = {dst, {}, access};
  // endregion
    return std::nullopt;
  }

  // If it's a predecessor reg, replace with MachineReg and update live_in/outs.
  if (std::holds_alternative<PredecessorReg>(mem_reg_map[disp].value().value)) {
    // TODO(b/449996703): handle two in_edges.
    CHECK(bb->in_edges().size() == 1);
    MachineReg reg = std::get<PredecessorReg>(mem_reg_map[disp].value().value).reg;
    if (!Contains(bb->live_in(), reg)) {
      bb->live_in().push_back(reg);
      context_map_.reg_counter.AddLiveInLifetime(
          reg, IsSimdOffset(disp) ? RegType::kXmm : RegType::kGeneral);
    }
    MachineBasicBlock* pred_bb = bb->in_edges().front()->src();
    if (!Contains(pred_bb->live_out(), reg)) {
      pred_bb->live_out().push_back(reg);
    }

    mem_reg_map[disp].value().value = reg;
  }

  if (std::holds_alternative<MachineReg>(mem_reg_map[disp].value().value)) {
    MachineReg ret = std::get<MachineReg>(mem_reg_map[disp].value().value);
    *insn_it = machine_ir_->NewInsn<Copy>(dst, ret, insn->RegKindAt(0).RegClass());
    return ret;
  } else {
    CHECK(insn->opcode() != kMachineOpMovdqaXRegMemBaseDisp &&
          insn->opcode() != kMachineOpMovsdXRegMemBaseDisp);
    *insn_it =
        machine_ir_->NewInsn<MovqRegImm>(dst, std::get<uint64_t>(mem_reg_map[disp].value().value));
  }
  return std::nullopt;
}

void LocalGuestContextOptimizer::ReplacePutAndUpdateMap(MachineBasicBlock* bb,
                                                        const MachineInsnList::iterator insn_it) {
  auto* insn = AsMachineInsnX86_64(*insn_it);
  auto disp = insn->disp();
  auto& mem_reg_map = context_map_.mem_reg_map;

  // region digitalis
  // A narrower store does not fully overwrite a wider one, so only drop the
  // earlier store when this one covers every byte of it.
  auto store_access = GetContextAccess(*insn_it);
  if (mem_reg_map[disp].has_value() && mem_reg_map[disp].value().last_store.has_value() &&
      mem_reg_map[disp].value().access.size <= store_access.size) {
  // endregion
    // Remove the last store instruction.
    auto last_store_it = mem_reg_map[disp].value().last_store.value();
    bb->insn_list().erase(last_store_it);
  }

  MappedValue new_value;
  if (insn->opcode() == kMachineOpMovqMemBaseDispImm) {
    new_value = insn->imm();
  } else {
    new_value = insn->RegAt(1);
  }
  // region digitalis
  mem_reg_map[disp] = {new_value, {insn_it}, GetContextAccess(*insn_it)};
  // endregion
}

void LocalGuestContextOptimizer::PrepareGlobalMappingsForSuccessors(
    MachineBasicBlock* bb,
    const MachineBasicBlockList& nonloop_nodes) {
  auto& offsets_used = offsets_used_reservation_;
  auto& mem_reg_map = context_map_.mem_reg_map;
  auto& reg_counter = context_map_.reg_counter;
  // Get offsets used in successors.
  offsets_used.clear();
  for (auto* edge : bb->out_edges()) {
    // TODO(449996703): we should probably coalesce when there are two successors.
    if (EligibleForGlobalOpt(nonloop_nodes, edge->dst())) {
      PopulateOffsetsUsed(edge->dst(), offsets_used);
    }
  }

  for (uint32_t offset : offsets_used) {
    if (!mem_reg_map.at(offset).has_value()) {
      continue;
    }
    ArenaVector<DispRegMapping>& global_mappings = global_mappings_.at(bb->id());
    MappedValue mapped_val = mem_reg_map.at(offset).value().value;
    // For constants we can always reuse them.
    if (std::holds_alternative<uint64_t>(mapped_val)) {
      global_mappings.push_back(
          DispRegMapping{.disp = offset, .value = std::get<uint64_t>(mapped_val)});
    } else if (std::holds_alternative<MachineReg>(mapped_val)) {
      global_mappings.push_back(
          DispRegMapping{.disp = offset, .value = std::get<MachineReg>(mapped_val)});
      MachineReg reg = std::get<MachineReg>(mapped_val);
      // Note we do not push them to live_out until they're actually used.
      // Although we try our best to emulate the register lifetimes, it's not
      // exact so if we pushed it to live_out it's possible we would
      // unnecessarily cause register overuse as the registers aren't
      // guaranteed to be used in successors here.
      // Update register lifetimes and unmap if necessary.
      UpdateRegLastUse(reg, reg_counter.GetCounts().size(), offset);
    }
  }
}

void LocalGuestContextOptimizer::UpdateRegLastUse(MachineReg reg, size_t new_pos, uint32_t offset) {
  const size_t kLimit = IsSimdOffset(offset) ? params_.simd_reg_limit : params_.general_reg_limit;
  std::optional<size_t> pos_over_limit =
      context_map_.reg_counter.UpdateLastUse(reg, new_pos, kLimit);
  if (pos_over_limit.has_value()) {
    UnmapOlderThan(pos_over_limit.value(),
                   IsSimdOffset(offset) ? RegType::kXmm : RegType::kGeneral);
  }
}

void RemoveLocalGuestContextAccesses(x86_64::MachineIR* machine_ir,
                                     const OptimizeLocalParams& params) {
  LocalGuestContextOptimizer optimizer(machine_ir, params);
  optimizer.RemoveLocalGuestContextAccesses();
}

}  // namespace berberis::x86_64
