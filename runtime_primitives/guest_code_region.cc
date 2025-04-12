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

#include "berberis/base/arena_alloc.h"
#include "berberis/base/arena_map.h"
#include "berberis/base/arena_set.h"
#include "berberis/base/arena_vector.h"
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
  ResolveInEdges();
  code_region_finalized_ = true;
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

}  // namespace berberis
