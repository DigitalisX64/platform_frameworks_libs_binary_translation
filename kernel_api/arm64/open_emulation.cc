// region digitalis
/*
 * Copyright (C) 2026 utzcoz
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

#include "berberis/kernel_api/open_emulation.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "berberis/base/tracing.h"

namespace berberis {

#if !defined(__x86_64__)
#error Currently open flags conversion is only supported on x86_64
#endif

#if (O_LARGEFILE == 0)
#undef O_LARGEFILE
#define O_LARGEFILE 00100000
#endif

#if !defined(__O_SYNC)
#if defined(__BIONIC__)
#error __O_SYNC undefined in bionic
#endif
#define __O_SYNC 04000000
#endif

#ifndef O_SEARCH
#define O_SEARCH 0
#endif

static_assert((O_ACCMODE & ~O_SEARCH) == 00000003);
static_assert(O_CREAT == 00000100);
static_assert(O_EXCL == 00000200);
static_assert(O_NOCTTY == 00000400);
static_assert(O_TRUNC == 00001000);
static_assert(O_APPEND == 00002000);
static_assert(O_NONBLOCK == 00004000);
static_assert(O_DSYNC == 00010000);
static_assert(FASYNC == 00020000);
static_assert(O_NOATIME == 01000000);
static_assert(O_DIRECTORY == 0200000);
static_assert(O_NOFOLLOW == 00400000);
static_assert(O_CLOEXEC == 02000000);
static_assert(O_DIRECT == 040000);
static_assert(__O_SYNC == 04000000);
static_assert(O_SYNC == (O_DSYNC | __O_SYNC));
static_assert(O_PATH == 010000000);
static_assert(O_LARGEFILE == 00100000);

namespace {

const int kCompatibleOpenFlags =
    O_ACCMODE | O_CREAT | O_EXCL | O_NOCTTY | O_TRUNC | O_APPEND | O_NONBLOCK | O_DSYNC | FASYNC |
    O_NOATIME | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_DIRECT | __O_SYNC | O_PATH | O_LARGEFILE;

}  // namespace

const char* kGuestCpuinfoPath = "/system/etc/cpuinfo.arm64.txt";

int ToHostOpenFlags(int guest_flags) {
  int unknown_guest_flags = guest_flags & ~kCompatibleOpenFlags;
  if (unknown_guest_flags) {
    TRACE("Unrecognized guest open flags: original=0x%x unsupported=0x%x. Passing to host as is.",
          guest_flags,
          unknown_guest_flags);
  }
  return guest_flags;
}

int ToGuestOpenFlags(int host_flags) {
  int unknown_host_flags = host_flags & ~kCompatibleOpenFlags;
  if (unknown_host_flags) {
    TRACE("Unrecognized host open flags: original=0x%x unsupported=0x%x. Passing to guest as is.",
          host_flags,
          unknown_host_flags);
  }
  return host_flags;
}

}  // namespace berberis
// endregion
