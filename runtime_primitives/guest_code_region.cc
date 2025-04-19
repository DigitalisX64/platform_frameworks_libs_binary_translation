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

#include "berberis/runtime_primitives/guest_code_region.h"

#include <cstddef>
#include <string>

#include "berberis/base/arena_alloc.h"
#include "berberis/base/arena_list.h"
#include "berberis/base/arena_map.h"
#include "berberis/base/arena_set.h"
#include "berberis/base/arena_vector.h"
#include "berberis/base/stringprintf.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

GuestCodeBasicBlock* GuestCodeRegion::NewBasicBlock(GuestAddr guest_addr,
                                                    size_t size,
                                                    const ArenaVector<GuestAddr>& out_edges) {
  CHECK(!code_region_finalized_);
  auto [it, inserted] = basic_blocks_.try_emplace(guest_addr, arena_, guest_addr, size, out_edges);
  CHECK(inserted);
  branch_targets_.insert(out_edges.begin(), out_edges.end());
  return &it->second;
}

void GuestCodeRegion::ResolveEdges() {
  CHECK(!code_region_finalized_);
  ValidateRegionBeforeFinalize();
  SplitBasicBlocks();
  // SplitBasicBlocks can end up splitting a block
  // into referenced and unreferenced ones, so it
  // needs to happen before RemoveUnreferencedBlocks.
  RemoveUnreachableBlocks();
  ResolveInEdges();
  code_region_finalized_ = true;
}

ArenaSet<GuestAddr> GuestCodeRegion::CollectReachableBranchTargets() const {
  ArenaSet<GuestAddr> reachable_targets(arena_);
  ArenaList<GuestAddr> worklist_targets(arena_);

  const GuestAddr kRegionStartAddr = basic_blocks_.begin()->first;
  const GuestAddr kRegionEndAddr = basic_blocks_.rbegin()->second.end_addr();
  const size_t kRegionSize = kRegionEndAddr - kRegionStartAddr;

  ArenaVector<bool> visited_target_offsets(kRegionSize, false, arena_);

  worklist_targets.push_back(kRegionStartAddr);

  // Collect reachable branch targets.
  while (!worklist_targets.empty()) {
    GuestAddr branch_addr = worklist_targets.front();
    worklist_targets.pop_front();

    auto it = basic_blocks_.find(branch_addr);
    if (it == basic_blocks_.end()) {
      continue;
    }

    CHECK_GE(branch_addr, kRegionStartAddr);
    CHECK_LT(branch_addr, kRegionEndAddr);
    if (visited_target_offsets.at(branch_addr - kRegionStartAddr)) {
      continue;
    }

    const auto& basic_block = it->second;
    worklist_targets.insert(
        worklist_targets.end(), basic_block.out_edges().begin(), basic_block.out_edges().end());
    reachable_targets.insert(basic_block.out_edges().begin(), basic_block.out_edges().end());
    visited_target_offsets[branch_addr - kRegionStartAddr] = true;
  };

  return reachable_targets;
}

void GuestCodeRegion::RemoveUnreachableBlocks() {
  CHECK(!basic_blocks_.empty());

  auto branch_targets = CollectReachableBranchTargets();

  // Remove unreachable basic_blocks.
  auto bb_it = basic_blocks_.begin();
  // Always keep the first basic block.
  ++bb_it;

  while (bb_it != basic_blocks_.end()) {
    if (branch_targets.contains(bb_it->first)) {
      ++bb_it;
    } else {
      bb_it = basic_blocks_.erase(bb_it);
    }
  }

  // Update branch_targets.
  branch_targets_ = std::move(branch_targets);
}

void GuestCodeRegion::SplitBasicBlocks() {
  for (auto branch_target : branch_targets_) {
    auto it = basic_blocks_.upper_bound(branch_target);
    if (it == basic_blocks_.begin()) {
      continue;
    }

    --it;
    auto& [guest_addr, code_block] = *it;
    if (branch_target <= guest_addr || branch_target >= code_block.end_addr()) {
      // Nothing to split.
      continue;
    }

    size_t updated_size = branch_target - code_block.start_addr();
    size_t new_code_block_size = code_block.size() - updated_size;

    NewBasicBlock(branch_target, new_code_block_size, code_block.out_edges());

    code_block.SetSize(updated_size);
    code_block.SetOutEdges(ArenaVector<GuestAddr>({branch_target}, arena_));
  }
}

void GuestCodeRegion::ResolveInEdges() {
  for (auto& [source_addr, basic_block] : basic_blocks_) {
    for (auto target_addr : basic_block.out_edges()) {
      auto it = basic_blocks_.find(target_addr);
      if (it != basic_blocks_.end()) {
        it->second.AddInEdge(source_addr);
      }
    }
  }
}

void GuestCodeRegion::ValidateRegionBeforeFinalize() const {
  GuestAddr last_seen_end_addr = kNullGuestAddr;
  for (const auto& [start_addr, basic_block] : basic_blocks_) {
    CHECK_GE(start_addr, last_seen_end_addr);
    last_seen_end_addr = basic_block.end_addr();
    CHECK(basic_block.in_edges().empty());
  }
}

std::string GuestCodeRegion::GetDebugString() const {
  std::string out;
  out += StringPrintf("BasicBlocks: { size=%zd, elements=[", basic_blocks_.size());
  for (const auto& basic_block : basic_blocks_) {
    out += " ";
    out += basic_block.second.GetDebugString();
  }

  out += "]}\n";
  out += "branch_targets={";
  for (auto addr : branch_targets_) {
    out += StringPrintf(" %zx", addr);
  }

  out += "}";

  return out;
}

std::string GuestCodeBasicBlock::GetDebugString() const {
  std::string out = "(";
  out += StringPrintf("start=%zx, size=0x%zx(%zd), ", start_addr_, size_, size_);
  out += "in_edges={";
  for (auto addr : in_edges_) {
    out += StringPrintf(" %zx", addr);
  }
  out += " }, out_edges={";

  for (auto addr : out_edges_) {
    out += StringPrintf(" %zx", addr);
  }

  out += " })";
  return out;
}

}  // namespace berberis
