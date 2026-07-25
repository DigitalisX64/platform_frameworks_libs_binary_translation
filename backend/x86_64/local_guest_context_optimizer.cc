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
#include <variant>

#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

namespace {

using OffsetCounterMap = ArenaVector<std::pair<size_t, int>>;

// region digitalis
// Bytes of the CPU-state slot a guest-context access actually touches, and
// whether it goes through an SIMD register. The map below is keyed by offset
// alone, so a mapping is only interchangeable with an access of the same
// register class that touches no more bytes than the mapping recorded.
struct ContextAccess {
  size_t size;
  bool is_simd;
};

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
      return {0, false};
  }
}
// endregion digitalis

class LocalGuestContextOptimizer {
 public:
  explicit LocalGuestContextOptimizer(x86_64::MachineIR* machine_ir)
      : machine_ir_(machine_ir),
        mem_reg_map_(sizeof(CPUState), std::nullopt, machine_ir->arena()) {}

  void RemoveLocalGuestContextAccesses(const OptimizeLocalParams& params);

 private:
  using MappedValue = std::variant<MachineReg, uint64_t>;
  struct MappedRegUsage {
    MappedValue value;
    std::optional<MachineInsnList::iterator> last_store;
    // region digitalis
    ContextAccess access;
    // endregion digitalis
  };

  void ReplaceGetAndUpdateMap(const MachineInsnList::iterator insn_it);
  void ReplacePutAndUpdateMap(MachineInsnList& insn_list, const MachineInsnList::iterator insn_it);

  MachineIR* machine_ir_;
  ArenaVector<std::optional<MappedRegUsage>> mem_reg_map_;
};

ArenaVector<int> CountGuestRegAccesses(const MachineIR* ir, MachineBasicBlock* bb) {
  ArenaVector<int> guest_access_count(sizeof(CPUState), 0, ir->arena());
  for (auto* base_insn : bb->insn_list()) {
    if (ir->IsCPUStateGet(base_insn) || ir->IsCPUStatePut(base_insn)) {
      auto insn = AsMachineInsnX86_64(base_insn);
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
    // region digitalis
    // The subset of offsets that can ever hold a live mapping (only optimized
    // offsets are ever recorded in mem_reg_map_). Iterating just these keeps the
    // per-instruction def-invalidation below cheap (bounded by the reg limits)
    // instead of scanning all of CPUState.
    ArenaVector<size_t> optimized_offset_list(machine_ir_->arena());
    // endregion digitalis
    for (auto [offset, unused_counter] : sorted_offsets) {
      // TODO(b/232598137): Account for f and v register classes.
      // Simd regs.
      if (IsSimdOffset(offset)) {
        if (simd_reg_count++ < params.simd_reg_limit) {
          optimized_offsets[offset] = true;
          // region digitalis
          optimized_offset_list.push_back(offset);
          // endregion digitalis
        }
        continue;
      }
      // General regs and flags.
      if (general_reg_count++ < params.general_reg_limit) {
        optimized_offsets[offset] = true;
        // region digitalis
        optimized_offset_list.push_back(offset);
        // endregion digitalis
      }
    }

    for (auto insn_it = bb->insn_list().begin(); insn_it != bb->insn_list().end(); insn_it++) {
      auto* insn = AsMachineInsnX86_64(*insn_it);
      // region digitalis
      // Invalidate any cached offset whose register is redefined by this insn.
      // A destructive 2-address op (e.g. `psubd xn, xm`, kUseDef on xn) mutates
      // xn in place; if a prior Get cached xn as holding some guest offset, that
      // cache is now stale and must not be forwarded to a later Get of the same
      // offset. Runs BEFORE this insn's own Get records its dst, so a Get does
      // not self-invalidate the mapping it is about to establish. This preserves
      // riscv64 behaviour (its frontend never redefines a context-load register
      // while it is mapped, so the invalidation never fires there).
      for (int i = 0; i < insn->NumRegOperands(); i++) {
        if (!insn->RegKindAt(i).IsDef()) {
          continue;
        }
        MachineReg def_reg = insn->RegAt(i);
        for (size_t off : optimized_offset_list) {
          auto& entry = mem_reg_map_[off];
          if (entry.has_value() && std::holds_alternative<MachineReg>(entry.value().value) &&
              std::get<MachineReg>(entry.value().value) == def_reg) {
            entry = std::nullopt;
          }
        }
      }
      // endregion digitalis
      // Skip insn if it accesses regs with low priority
      if (machine_ir_->IsCPUStateGet(*insn_it) || machine_ir_->IsCPUStatePut(*insn_it)) {
        if (!optimized_offsets.at(insn->disp())) {
          continue;
        }

        if (machine_ir_->IsCPUStateGet(insn)) {
          ReplaceGetAndUpdateMap(insn_it);
        } else if (machine_ir_->IsCPUStatePut(insn)) {
          ReplacePutAndUpdateMap(bb->insn_list(), insn_it);
        }
      }
    }
  }
}

void LocalGuestContextOptimizer::ReplaceGetAndUpdateMap(const MachineInsnList::iterator insn_it) {
  auto* insn = AsMachineInsnX86_64(*insn_it);
  auto dst = insn->RegAt(0);
  auto disp = insn->disp();

  // region digitalis
  // The map is keyed by offset alone, so a mapping recorded by a narrower
  // access does not hold the bits a wider one needs: a 64-bit MOVSD of a guest
  // vector register zeroes the upper half of its XMM, and forwarding that to a
  // later 128-bit MOVDQA of the same register would read those zeroes instead
  // of the register's real upper half. Keep the load whenever the mapping is
  // narrower, or belongs to the other register class, and let it re-establish
  // the mapping at its own width.
  auto access = GetContextAccess(insn);
  // endregion digitalis

  // We only need to keep this load instruction if this is the first access to
  // the guest context at disp.
  // region digitalis
  if (!mem_reg_map_[disp].has_value() || mem_reg_map_[disp].value().access.size < access.size ||
      mem_reg_map_[disp].value().access.is_simd != access.is_simd) {
    mem_reg_map_[disp] = {dst, {}, access};
    return;
  }
  // endregion digitalis

  auto copy_size = insn->opcode() == kMachineOpMovdqaXRegMemBaseDisp ? 16 : 8;
  if (std::holds_alternative<MachineReg>(mem_reg_map_[disp].value().value)) {
    *insn_it = machine_ir_->NewInsn<PseudoCopy>(
        dst, std::get<MachineReg>(mem_reg_map_[disp].value().value), copy_size);
  } else {
    CHECK(insn->opcode() != kMachineOpMovdqaXRegMemBaseDisp &&
          insn->opcode() != kMachineOpMovsdXRegMemBaseDisp);
    *insn_it =
        machine_ir_->NewInsn<MovqRegImm>(dst, std::get<uint64_t>(mem_reg_map_[disp].value().value));
  }
}

void LocalGuestContextOptimizer::ReplacePutAndUpdateMap(MachineInsnList& insn_list,
                                                        const MachineInsnList::iterator insn_it) {
  auto* insn = AsMachineInsnX86_64(*insn_it);
  auto disp = insn->disp();

  // region digitalis
  auto access = GetContextAccess(insn);
  // A narrower store does not fully overwrite a wider one -- a 64-bit MOVSD
  // leaves the upper half of a guest vector register slot holding what the
  // earlier 128-bit MOVDQA wrote -- so only drop the earlier store when this
  // one covers every byte of it.
  if (mem_reg_map_[disp].has_value() && mem_reg_map_[disp].value().last_store.has_value() &&
      mem_reg_map_[disp].value().access.size <= access.size) {
    // endregion digitalis
    // Remove the last store instruction.
    auto last_store_it = mem_reg_map_[disp].value().last_store.value();
    insn_list.erase(last_store_it);
  }

  MappedValue new_value;
  if (insn->opcode() == kMachineOpMovqMemBaseDispImm) {
    new_value = insn->imm();
  } else {
    new_value = insn->RegAt(1);
  }
  // region digitalis
  mem_reg_map_[disp] = {new_value, {insn_it}, access};
  // endregion digitalis
}

}  // namespace

void RemoveLocalGuestContextAccesses(x86_64::MachineIR* machine_ir,
                                     const OptimizeLocalParams& params) {
  LocalGuestContextOptimizer optimizer(machine_ir);
  optimizer.RemoveLocalGuestContextAccesses(params);
}

}  // namespace berberis::x86_64
