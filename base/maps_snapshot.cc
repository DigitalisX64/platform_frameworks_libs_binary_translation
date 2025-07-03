/*
 * Copyright (C) 2024 The Android Open Source Project
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

#include "berberis/base/maps_snapshot.h"

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>

#include "berberis/base/arena_string.h"
#include "berberis/base/forever_alloc.h"
#include "berberis/base/tracing.h"

namespace berberis {

MapsSnapshot* MapsSnapshot::GetInstance() {
  static auto* g_maps_snapshot = PrivateNewForever<MapsSnapshot>::Alloc();
  return g_maps_snapshot;
}

void MapsSnapshot::Update() {
  std::scoped_lock lock(mutex_);

  FILE* maps_file = fopen("/proc/self/maps", "r");
  if (!maps_file) {
    TRACE("Error opening /proc/self/maps");
    return;
  }

  maps_.clear();

  char line[512], pathname[256];
  uintptr_t start, end;
  while (fgets(line, sizeof(line), maps_file)) {
    // Maximum string size 255 so that we have space for the terminating '\0'.
    int match_count = sscanf(
        line, "%" SCNxPTR "-%" SCNxPTR " %*s %*lx %*x:%*x %*lu%*[ ]%255s", &start, &end, pathname);
    if (match_count == 2 || match_count == 3) {
      // If there is no pathname we still memorize the record, so that we can differentiate this
      // case from missing mapping, e.g. when the snapshot is not up to date.
      const char* recorded_pathname = (match_count == 3) ? pathname : "";
      // Addresses go in the increasing order in /proc/self/maps, so we hint to add new records
      // to the end of the map.
      maps_.emplace_hint(
          maps_.end(), start, Record{start, end, ArenaString{recorded_pathname, &arena_}});
    }
  }

  fclose(maps_file);
}

std::optional<const MapsSnapshot::Record> MapsSnapshot::GetMappedObjectRecord(uintptr_t addr) {
  std::scoped_lock lock(mutex_);
  auto next_it = maps_.upper_bound(addr);
  if (next_it == maps_.begin()) {
    return std::nullopt;
  }
  auto& rec = std::prev(next_it)->second;
  if (addr >= rec.start && addr < rec.end) {
    // Make sure we return a copy since the storage for ArenaString may be invalidated as soon as we
    // release the lock (e.g. if another thread calls Update()).
    return rec;
  }
  return std::nullopt;
}

}  // namespace berberis
