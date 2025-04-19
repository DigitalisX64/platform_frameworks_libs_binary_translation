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

#ifndef BERBERIS_RUNTIME_PRIMITIVES_GUEST_CODE_REGION_H_
#define BERBERIS_RUNTIME_PRIMITIVES_GUEST_CODE_REGION_H_

#include <cstddef>

#include "berberis/base/arena_alloc.h"
#include "berberis/base/arena_map.h"
#include "berberis/base/arena_set.h"
#include "berberis/base/arena_vector.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

class GuestCodeBasicBlock {
 public:
  explicit GuestCodeBasicBlock(Arena* arena,
                               GuestAddr start_addr,
                               size_t size,
                               ArenaVector<GuestAddr> out_edges)
      : start_addr_{start_addr}, size_{size}, in_edges_{arena}, out_edges_{std::move(out_edges)} {}

  void SetOutEdges(ArenaVector<GuestAddr> out_edges) { out_edges_ = std::move(out_edges); }

  void AddInEdge(GuestAddr source_addr) { in_edges_.push_back(source_addr); }

  void SetSize(size_t size) { size_ = size; }

  [[nodiscard]] GuestAddr start_addr() const { return start_addr_; }
  [[nodiscard]] GuestAddr end_addr() const { return start_addr_ + size_; }
  [[nodiscard]] size_t size() const { return size_; }
  [[nodiscard]] const ArenaVector<GuestAddr>& out_edges() const { return out_edges_; }
  [[nodiscard]] const ArenaVector<GuestAddr>& in_edges() const { return in_edges_; }

  [[nodiscard]] std::string GetDebugString() const;

 private:
  const GuestAddr start_addr_;
  size_t size_;
  ArenaVector<GuestAddr> in_edges_;
  ArenaVector<GuestAddr> out_edges_;
};

class GuestCodeRegion {
 public:
  explicit GuestCodeRegion(Arena* arena)
      : arena_{arena}, basic_blocks_{arena}, branch_targets_{arena} {}

  /* may_discard */ GuestCodeBasicBlock* NewBasicBlock(GuestAddr guest_addr,
                                                       size_t size,
                                                       const ArenaVector<GuestAddr>& out_edges);

  // This method must be called only once.
  void ResolveEdges();

  [[nodiscard]] const ArenaMap<GuestAddr, GuestCodeBasicBlock>& basic_blocks() const {
    return basic_blocks_;
  }

  [[nodiscard]] const ArenaSet<GuestAddr>& branch_targets() const { return branch_targets_; }

  [[nodiscard]] std::string GetDebugString() const;

 private:
  // Collects targets reachable from the first basic_block.
  // The first block is included in the result iff it's reachable from
  // some other block (through a back edge). The resulted set contains
  // external reachable branch_targets as well as internal ones.
  ArenaSet<GuestAddr> CollectReachableBranchTargets() const;
  void SplitBasicBlocks();
  void RemoveUnreachableBlocks();
  void ResolveInEdges();
  void ValidateRegionBeforeFinalize() const;

  Arena* arena_;
  ArenaMap<GuestAddr, GuestCodeBasicBlock> basic_blocks_;
  ArenaSet<GuestAddr> branch_targets_;
  bool code_region_finalized_{false};
};

}  // namespace berberis

#endif  // BERBERIS_RUNTIME_PRIMITIVES_GUEST_CODE_REGION_H_
