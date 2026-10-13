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

#include <algorithm>
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

// region digitalis
namespace {

// The continuation of a block that a faulting guest memory access ended: the
// target of its PseudoBranch, when the block's only other successor is the
// recovery block and the target has no other predecessor.
MachineBasicBlock* GetRecoveryContinuation(const MachineBasicBlock* bb) {
  if (bb->is_recovery() || bb->insn_list().empty() ||
      bb->insn_list().back()->opcode() != kMachineOpPseudoBranch || bb->out_edges().size() != 2) {
    return nullptr;
  }
  const auto* target = static_cast<const PseudoBranch*>(bb->insn_list().back())->then_bb();
  MachineBasicBlock* next = nullptr;
  bool has_recovery = false;
  for (auto* edge : bb->out_edges()) {
    if (edge->dst() == target) {
      next = edge->dst();
    } else {
      has_recovery |= edge->dst()->is_recovery();
    }
  }
  if (!next || !has_recovery || next->is_recovery() || next->in_edges().size() != 1) {
    return nullptr;
  }
  return next;
}

// Forwards guest-context reads within one chain of recovery-split blocks. The
// chain is a straight line (each continuation has a single predecessor), so a
// value recorded earlier in it reaches every later instruction unless an
// instruction in between redefines its register or writes the slot.
class RecoverySplitGetForwarder {
 public:
  RecoverySplitGetForwarder(MachineIR* machine_ir, const OptimizeLocalParams& params)
      : machine_ir_(machine_ir),
        params_(params),
        map_(sizeof(CPUState), std::nullopt, machine_ir->arena()),
        optimized_offsets_(sizeof(CPUState), false, machine_ir->arena()),
        optimized_offset_list_(machine_ir->arena()) {}

  void ForwardChain(const ArenaVector<MachineBasicBlock*>& chain);

 private:
  struct MappedValue {
    MachineReg reg;
    ContextAccess access;
    // Instruction count along the chain when the value was recorded.
    size_t position;
    const MachineBasicBlock* bb;
    // Recorded by this pass's own copy rather than by a get or put the local
    // pass also sees.
    bool forwarded;
    // The register holds exactly `access.size` bytes of the slot with the
    // rest zeroed, as a load from memory leaves it.
    bool zero_extended;
  };

  void SelectOffsets(const ArenaVector<MachineBasicBlock*>& chain);
  void InvalidateOverlapping(int64_t begin, int64_t end);
  void InvalidateReg(MachineReg reg);
  void ForwardInsn(const MachineBasicBlock* bb, MachineInsnList::iterator insn_it);

  MachineIR* machine_ir_;
  OptimizeLocalParams params_;
  ArenaVector<std::optional<MappedValue>> map_;
  ArenaVector<bool> optimized_offsets_;
  ArenaVector<size_t> optimized_offset_list_;
  size_t position_ = 0;
};

// Like the local pass, map only the most accessed slots of each class so the
// forwarded values do not crowd out the allocator. Flag-sized (2-byte) slots
// are left to the local pass: the flag fusions read cpu.flags across their
// own entry blocks.
void RecoverySplitGetForwarder::SelectOffsets(const ArenaVector<MachineBasicBlock*>& chain) {
  for (size_t offset : optimized_offset_list_) {
    optimized_offsets_[offset] = false;
    map_[offset] = std::nullopt;
  }
  optimized_offset_list_.clear();

  ArenaVector<int> counts(sizeof(CPUState), 0, machine_ir_->arena());
  for (auto* bb : chain) {
    for (auto* insn : bb->insn_list()) {
      if ((machine_ir_->IsCPUStateGet(insn) || machine_ir_->IsCPUStatePut(insn)) &&
          GetContextAccess(insn).size != 2) {
        counts.at(AsMachineInsnX86_64(insn)->disp())++;
      }
    }
  }
  OffsetCounterMap sorted(machine_ir_->arena());
  for (size_t offset = 0; offset < sizeof(CPUState); offset++) {
    if (counts[offset] > 0) {
      sorted.push_back({offset, counts[offset]});
    }
  }
  std::stable_sort(
      sorted.begin(), sorted.end(), [](auto a, auto b) { return a.second > b.second; });
  size_t general_reg_count = 0;
  size_t simd_reg_count = 0;
  for (auto [offset, unused_count] : sorted) {
    size_t& count = IsSimdOffset(offset) ? simd_reg_count : general_reg_count;
    size_t limit = IsSimdOffset(offset) ? params_.simd_reg_limit : params_.general_reg_limit;
    if (count++ < limit) {
      optimized_offsets_[offset] = true;
      optimized_offset_list_.push_back(offset);
    }
  }
}

void RecoverySplitGetForwarder::InvalidateOverlapping(int64_t begin, int64_t end) {
  for (size_t offset : optimized_offset_list_) {
    auto& entry = map_[offset];
    if (entry.has_value() && static_cast<int64_t>(offset) < end &&
        begin < static_cast<int64_t>(offset + entry.value().access.size)) {
      entry = std::nullopt;
    }
  }
}

void RecoverySplitGetForwarder::InvalidateReg(MachineReg reg) {
  for (size_t offset : optimized_offset_list_) {
    auto& entry = map_[offset];
    if (entry.has_value() && entry.value().reg == reg) {
      entry = std::nullopt;
    }
  }
}

void RecoverySplitGetForwarder::ForwardInsn(const MachineBasicBlock* bb,
                                            MachineInsnList::iterator insn_it) {
  auto* insn = AsMachineInsnX86_64(*insn_it);
  // No guest-context pass models a call as reading or writing guest state;
  // forget everything rather than carry a value across one.
  if (insn->opcode() == kMachineOpCallImm || insn->opcode() == kMachineOpCallImmArg) {
    for (size_t offset : optimized_offset_list_) {
      map_[offset] = std::nullopt;
    }
    return;
  }
  // A redefined register no longer holds the slot it was recorded for (this
  // runs before a get records its own destination).
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegKindAt(i).IsDef()) {
      InvalidateReg(insn->RegAt(i));
    }
  }

  if (machine_ir_->IsCPUStateGet(insn)) {
    size_t disp = insn->disp();
    auto access = GetContextAccess(insn);
    if (!optimized_offsets_[disp] || access.size == 2) {
      return;
    }
    MachineReg dst = insn->RegAt(0);
    auto& entry = map_[disp];
    if (!entry.has_value() || entry.value().access.is_simd != access.is_simd ||
        entry.value().access.size < access.size) {
      if (dst.IsVReg()) {
        entry = MappedValue{dst, access, position_, bb, false, true};
      }
      return;
    }
    // Within one block the local pass forwards the value later, with a plain
    // copy the allocator can coalesce.
    if (entry.value().bb == bb && !entry.value().forwarded) {
      return;
    }
    // A value carried further keeps its register live across the whole
    // chain: in long straight-line regions the allocator then spills it
    // (unbounded forwarding added hundreds of stack accesses to some hot
    // regions). Full vectors are not carried at all: XMM registers run out
    // first in the vector kernels the loop pass already keeps in registers.
    // Reload instead, and carry the reloaded value on.
    constexpr size_t kMaxForwardDistance = 32;
    constexpr size_t kMaxForwardedSimdSize = 8;
    if (access.is_simd && access.size > kMaxForwardedSimdSize) {
      if (dst.IsVReg()) {
        entry = MappedValue{dst, access, position_, bb, false, true};
      }
      return;
    }
    if (position_ - entry.value().position > kMaxForwardDistance) {
      if (dst.IsVReg()) {
        entry = MappedValue{dst, access, position_, bb, false, true};
      }
      return;
    }
    MachineReg src = entry.value().reg;
    if (access.is_simd && access.size == 8 &&
        !(entry.value().access.size == 8 && entry.value().zero_extended)) {
      // MOVSD from memory zeroes the upper half; the recorded register may
      // hold a full vector (or a scalar put's stale upper half).
      *insn_it = machine_ir_->NewInsn<MovqXRegXReg>(dst, src);
    } else {
      *insn_it = machine_ir_->NewInsn<PseudoCopy>(dst, src, access.size);
    }
    // The copy now holds the slot; carrying it on keeps the source's live
    // range short.
    if (access.size == entry.value().access.size && dst.IsVReg()) {
      entry = MappedValue{dst, access, position_, bb, true, true};
    }
    return;
  }

  if (machine_ir_->IsCPUStatePut(insn)) {
    size_t disp = insn->disp();
    auto access = GetContextAccess(insn);
    InvalidateOverlapping(disp, disp + access.size);
    if (optimized_offsets_[disp] && access.size != 2 &&
        insn->opcode() != kMachineOpMovqMemBaseDispImm && insn->RegAt(1).IsVReg()) {
      map_[disp] = MappedValue{insn->RegAt(1), access, position_, bb, false, false};
    }
    return;
  }

  // Any other access through the state pointer may write guest state in a
  // form the map does not model: forget every slot it could touch.
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegAt(i) == kCPUStatePointer) {
      constexpr int64_t kMaxAccessSize = 32;
      int64_t disp = static_cast<int32_t>(insn->disp());
      InvalidateOverlapping(disp, disp + kMaxAccessSize);
      int64_t disp2 = static_cast<int32_t>(insn->disp2());
      if (disp2 != 0) {
        InvalidateOverlapping(disp2, disp2 + kMaxAccessSize);
      }
      break;
    }
  }
}

void RecoverySplitGetForwarder::ForwardChain(const ArenaVector<MachineBasicBlock*>& chain) {
  SelectOffsets(chain);
  for (auto* bb : chain) {
    for (auto insn_it = bb->insn_list().begin(); insn_it != bb->insn_list().end(); insn_it++) {
      ForwardInsn(bb, insn_it);
      position_++;
    }
  }
}

}  // namespace

void ForwardGuestContextGetsAcrossRecoverySplits(x86_64::MachineIR* machine_ir,
                                                 const OptimizeLocalParams& params) {
  RecoverySplitGetForwarder forwarder(machine_ir, params);
  ArenaVector<MachineBasicBlock*> chain(machine_ir->arena());
  for (auto* bb : machine_ir->bb_list()) {
    if (bb->in_edges().size() == 1 && GetRecoveryContinuation(bb->in_edges()[0]->src()) == bb) {
      continue;  // Not a chain head: forwarded with its predecessor.
    }
    chain.clear();
    for (auto* link = bb; link != nullptr; link = GetRecoveryContinuation(link)) {
      chain.push_back(link);
    }
    if (chain.size() > 1) {
      forwarder.ForwardChain(chain);
    }
  }
}
// endregion digitalis

}  // namespace berberis::x86_64
