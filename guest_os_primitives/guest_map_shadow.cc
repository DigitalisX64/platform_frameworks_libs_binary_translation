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

#include "berberis/guest_os_primitives/guest_map_shadow.h"

#include <sys/mman.h>
#include <climits>  // CHAR_BIT
#include <cmath>
#include <cstddef>
#include <mutex>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/base/config.h"  // kGuestPageSize
#include "berberis/base/forever_alloc.h"
#include "berberis/base/large_mmap.h"
#include "berberis/base/mmap.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/runtime_library.h"  // InvalidateGuestRange

namespace berberis {

namespace {

#if defined(BERBERIS_GUEST_LP64)
// On LP64 the address space is limited to 48 bits
constexpr size_t kGuestAddressSizeLog2 = 48;
constexpr size_t kMaxGuestAddress{0xffff'ffff'ffff};
#else
constexpr size_t kGuestAddressSizeLog2 = sizeof(GuestAddr) * CHAR_BIT;
constexpr size_t kMaxGuestAddress{0xffff'ffff};
#endif

const size_t kGuestPageSize = config::kGuestPageSize;
// One bit per each page.
const size_t kGuestPageSizeLog2 = std::log2(kGuestPageSize);
const size_t kShadowSize = 1UL << (kGuestAddressSizeLog2 - kGuestPageSizeLog2 - 3);

inline GuestAddr AlignDownGuestPageSize(GuestAddr addr) {
  return AlignDown(addr, kGuestPageSize);
}

inline GuestAddr AlignUpGuestPageSize(GuestAddr addr) {
  return AlignUp(addr, kGuestPageSize);
}

bool DoIntervalsIntersect(const void* start,
                          const void* end,
                          const void* other_start,
                          const void* other_end) {
  bool not_intersect = (other_end <= start) || (other_start >= end);
  return !not_intersect;
}

}  // namespace

GuestMapShadow* GuestMapShadow::GetInstance() {
  static auto* g_map_shadow = NewForever<GuestMapShadow>();
  return g_map_shadow;
}

bool GuestMapShadow::IsExecAddr(GuestAddr addr) const {
  if (addr > kMaxGuestAddress) {
    // Addresses outside the supported range are always non-executable.
    // In practice we may end up here when parsing kernel addresses
    // from /proc/self/maps.
    return false;
  }
  uintptr_t page = addr >> kGuestPageSizeLog2;
  return shadow_[page >> 3] & (1 << (page & 7));
}

// Returns true if value changed.
bool GuestMapShadow::SetExecAddr(GuestAddr addr, int set) {
  if (addr > kMaxGuestAddress) {
    // See IsExecAddr for explanation.
    return false;
  }
  uintptr_t page = addr >> kGuestPageSizeLog2;
  uint8_t mask = 1 << (page & 7);
  int old = shadow_[page >> 3] & mask;
  if (set) {
    shadow_[page >> 3] |= mask;
    return old == 0;
  } else {
    shadow_[page >> 3] &= ~mask;
    return old != 0;
  }
}

void GuestMapShadow::CopyExecutable(GuestAddr from,
                                    size_t from_size,
                                    GuestAddr to,
                                    size_t to_size) {
  CHECK_EQ(from, AlignDownGuestPageSize(from));
  CHECK_EQ(to, AlignDownGuestPageSize(to));
  // Regions must not overlap.
  CHECK((from + from_size) <= to || (to + to_size) <= from);

  if (IsExecutable(from, from_size)) {
    SetExecutable(to, to_size);
  } else {
    // Note, we also get here if old region is partially
    // executable, to be on the safe side.
    ClearExecutable(to, to_size);
  }
}

GuestMapShadow::GuestMapShadow() : protected_maps_(&arena_) {
  shadow_ = static_cast<uint8_t*>(LargeMmapImplOrDie(
      {.size = kShadowSize, .flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE}));
}

GuestMapShadow::~GuestMapShadow() {
  MunmapOrDie(shadow_, kShadowSize);
}

std::tuple<bool, size_t> GuestMapShadow::GetExecutableRegionSize(GuestAddr start,
                                                                 size_t scan_size) const {
  GuestAddr pc = AlignDownGuestPageSize(start);
  GuestAddr scan_end_pc = AlignUpGuestPageSize(start + scan_size);

  bool is_exec = IsExecAddr(pc);
  for (pc += kGuestPageSize; pc < scan_end_pc; pc += kGuestPageSize) {
    if (is_exec != IsExecAddr(pc)) {
      break;
    }
  }
  return {is_exec, pc - start};
}

BitValue GuestMapShadow::GetExecutable(GuestAddr start, size_t scan_size) const {
  auto [is_exec, region_size] = GetExecutableRegionSize(start, scan_size);
  if (region_size < scan_size) {
    return kBitMixed;
  }
  return is_exec ? kBitSet : kBitUnset;
}

bool GuestMapShadow::IsExecutable(GuestAddr start, size_t size) const {
  return GetExecutable(start, size) == kBitSet;
}

void GuestMapShadow::SetExecutable(GuestAddr start, size_t size) {
  TRACE("SetExecutable: %zx..%zx", start, start + size);
  GuestAddr end = AlignUpGuestPageSize(start + size);
  GuestAddr pc = AlignDownGuestPageSize(start);
  while (pc < end) {
    SetExecAddr(pc, 1);
    pc += kGuestPageSize;
  }
}

void GuestMapShadow::ClearExecutable(GuestAddr start, size_t size) {
  TRACE("ClearExecutable: %zx..%zx", start, start + size);
  GuestAddr end = AlignUpGuestPageSize(start + size);
  GuestAddr pc = AlignDownGuestPageSize(start);
  bool changed = false;
  while (pc < end) {
    changed |= SetExecAddr(pc, 0);
    pc += kGuestPageSize;
  }
  if (changed) {
    InvalidateGuestRange(start, end);
  }
}

void GuestMapShadow::RemapExecutable(GuestAddr old_start,
                                     size_t old_size,
                                     GuestAddr new_start,
                                     size_t new_size) {
  TRACE("RemapExecutable: from %zx..%zx to %zx..%zx",
        old_start,
        old_start + old_size,
        new_start,
        new_start + new_size);

  CHECK_EQ(old_start, AlignDownGuestPageSize(old_start));
  CHECK_EQ(new_start, AlignDownGuestPageSize(new_start));
  GuestAddr old_end_page = AlignUpGuestPageSize(old_start + old_size);
  GuestAddr new_end_page = AlignUpGuestPageSize(new_start + new_size);

  // Special processing if only size is changed and regions overlap.
  if (old_start == new_start) {
    if (new_end_page <= old_end_page) {
      ClearExecutable(new_end_page, old_end_page - new_end_page);
    } else {
      CopyExecutable(old_start, old_size, old_end_page, new_end_page - old_end_page);
    }
    return;
  }

  // Otherwise, regions must not overlap.
  CHECK((old_start + old_size) <= new_start || (new_start + new_size) <= old_start);

  CopyExecutable(old_start, old_size < new_size ? old_size : new_size, new_start, new_size);
  ClearExecutable(old_start, old_size);
}

void GuestMapShadow::AddProtectedMapping(const void* start, const void* end) {
  std::lock_guard<std::mutex> lock(mutex_);
  protected_maps_.emplace_back(std::make_pair(start, end));
}

bool GuestMapShadow::IntersectsWithProtectedMapping(const void* start, const void* end) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto pair : protected_maps_) {
    if (DoIntervalsIntersect(pair.first, pair.second, start, end)) {
      return true;
    }
  }
  return false;
}

}  // namespace berberis
