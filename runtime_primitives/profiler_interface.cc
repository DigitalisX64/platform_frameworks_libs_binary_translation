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

#include "berberis/runtime_primitives/profiler_interface.h"

#include <fcntl.h>   // open
#include <unistd.h>  // write

#include <array>
#include <cstring>  // str*

#include "berberis/base/config_globals.h"
#include "berberis/base/format_buffer.h"
#include "berberis/base/gettid.h"
#include "berberis/base/maps_snapshot.h"
#include "berberis/base/scoped_errno.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

namespace {

int ProfilerOpenLogFile() {
  auto env = GetProfilingConfig();
  if (!env) {
    TRACE("Profiling: None");
    return -1;
  }

  auto app = GetAppPackageName();

  if (strcmp(env, "1") == 0) {
    // Special case - profile everything.
  } else if (app) {
    // Running an app - must match package name.
    if (strcmp(app, env) != 0) {
      TRACE("Profiling: Skipping: app %s doesn't match filter %s", app, env);
      return -1;
    }
  } else if (auto exe = GetMainExecutableRealPath()) {
    // Running a standalone program - must somehow match main executable path.
    if (!strstr(exe, env)) {
      TRACE("Profiling: Skipping: executable %s doesn't match filter %s", exe, env);
      return -1;
    }
  } else {
    // Running a unit test, or some other non-app, non-executable case.
    return -1;
  }

  ScopedErrno scoped_errno;

  char buf[160];

  int pid = GetpidSyscall();
  if (app) {
    FormatBuffer(buf, sizeof(buf), "/data/data/%s/perf-%u.map", app, pid);
  } else {
    FormatBuffer(buf, sizeof(buf), "/data/local/tmp/perf-%u.map", pid);
  }

  int fd = open(buf, O_WRONLY | O_CREAT | O_CLOEXEC, S_IWUSR);
  if (fd == -1) {
    TRACE("Profiling Error: Failed to open map file %s", buf);
  } else {
    TRACE("Probfiling to %s", buf);
  }
  return fd;
}

constexpr size_t kMaxMappedNameLen = 16;
// Name c-string + terminating null + underscore.
using MappedNameBuffer = std::array<char, kMaxMappedNameLen + 2>;

// Malloc-free implementation.
MappedNameBuffer ConstructMappedNameBuffer(GuestAddr guest_addr) {
  MappedNameBuffer buf;
  auto* maps_snapshot = MapsSnapshot::GetInstance();

  auto mapped_name = maps_snapshot->FindMappedObjectName(guest_addr);
  if (!mapped_name.has_value()) {
    // If no mapping is found renew the snapshot and try again.
    maps_snapshot->Update();
    auto updated_mapped_name = maps_snapshot->FindMappedObjectName(guest_addr);
    if (!updated_mapped_name.has_value()) {
      TRACE("Guest addr %p not found in /proc/self/maps", ToHostAddr<void>(guest_addr));
      buf[0] = '\0';
      return buf;
    }
    mapped_name.emplace(std::move(updated_mapped_name.value()));
  }

  // We can use more clever logic here and try to extract the basename, but the parent directory
  // name may also be interesting (e.g. <guest_arch>/libc.so) so we just take the last
  // kMaxMappedNameLen symbols for simplicity until it's proven we need something more advanced.
  // An added benefit of this approach is that symbols look well aligned in the profile.
  auto& result = mapped_name.value();
  size_t terminator_pos;
  if (result.length() > kMaxMappedNameLen) {
    // In this case it should be safe to call strcpy, but we still use strncpy to be extra careful.
    strncpy(buf.data(), result.c_str() + result.length() - kMaxMappedNameLen, kMaxMappedNameLen);
    terminator_pos = kMaxMappedNameLen;
  } else {
    strncpy(buf.data(), result.c_str(), kMaxMappedNameLen);
    terminator_pos = result.length();
  }
  buf[terminator_pos] = '_';
  buf[terminator_pos + 1] = '\0';

  return buf;
}

}  // namespace

void ProfilerLogGeneratedCode(const void* start,
                              size_t size,
                              GuestAddr guest_start,
                              size_t guest_size,
                              const char* jit_suffix) {
  static int fd = ProfilerOpenLogFile();
  if (fd == -1) {
    return;
  }

  MappedNameBuffer mapped_name_buf = ConstructMappedNameBuffer(guest_start);

  char guest_range_buf[64];

  if (IsConfigFlagSet(kMergeProfilesForSameModeRegions)) {
    guest_range_buf[0] = '\0';
  } else {
    FormatBuffer(guest_range_buf, sizeof(guest_range_buf), "_0x%lx+%zu", guest_start, guest_size);
  }

  char buf[128];
  // start size symbol-name
  size_t n = FormatBuffer(buf,
                          sizeof(buf),
                          "%p 0x%zx %s%s%s\n",
                          start,
                          size,
                          mapped_name_buf.data(),
                          jit_suffix,
                          guest_range_buf);
  UNUSED(write(fd, buf, n));
}

}  // namespace berberis
