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

#include <cstdio>
#include <string>

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

// ARM64 (arch/arm64/include/uapi/asm/fcntl.h) overrides a handful of the
// asm-generic flag bits to the legacy ARM/Alpha layout, swapping O_DIRECTORY
// with O_DIRECT and O_NOFOLLOW with O_LARGEFILE relative to the asm-generic
// (x86_64) values. Translating these is mandatory: passing a guest
// O_DIRECTORY (0o40000) to the host kernel verbatim looks like O_DIRECT,
// which the host rejects with EINVAL when applied to a directory --
// silently breaking opendir / QDirListing / any directory-iteration code.
constexpr int kGuestODirectory = 040000;    // 0x4000  (host O_DIRECT bit position)
constexpr int kGuestONofollow  = 0100000;   // 0x8000  (host O_LARGEFILE bit position)
constexpr int kGuestODirect    = 0200000;   // 0x10000 (host O_DIRECTORY bit position)
constexpr int kGuestOLargefile = 0400000;   // 0x20000 (host O_NOFOLLOW bit position)

static_assert(O_DIRECTORY == 0200000);
static_assert(O_NOFOLLOW  == 0400000);
static_assert(O_DIRECT    == 040000);
static_assert(O_LARGEFILE == 0100000);

constexpr int kArchSpecificMask =
    kGuestODirectory | kGuestONofollow | kGuestODirect | kGuestOLargefile;

// Flags whose values match between guest (arm64) and host (x86_64). The
// arch-specific bits are excluded; they are handled by the explicit swap below.
const int kCompatibleOpenFlags =
    O_ACCMODE | O_CREAT | O_EXCL | O_NOCTTY | O_TRUNC | O_APPEND | O_NONBLOCK | O_DSYNC | FASYNC |
    O_NOATIME | O_CLOEXEC | __O_SYNC | O_PATH;

// The swap pairs (O_DIRECTORY <-> O_DIRECT, O_NOFOLLOW <-> O_LARGEFILE) occupy
// exactly the same four bit positions on both arches; swapping is self-inverse,
// so the same helper translates in either direction.
int SwapArchSpecificBits(int flags) {
  int out = flags & ~kArchSpecificMask;
  if (flags & kGuestODirectory) out |= O_DIRECTORY;   // 0x4000  -> 0x10000
  if (flags & kGuestODirect)    out |= O_DIRECT;      // 0x10000 -> 0x4000
  if (flags & kGuestONofollow)  out |= O_NOFOLLOW;    // 0x8000  -> 0x20000
  if (flags & kGuestOLargefile) out |= O_LARGEFILE;   // 0x20000 -> 0x8000
  return out;
}

}  // namespace

const char* kGuestCpuinfoPath = "/system/etc/cpuinfo.arm64.txt";

std::string FormatGuestCpuinfo(int num_cpus) {
  if (num_cpus < 1) {
    num_cpus = 1;
  }
  // One block per online CPU, in the ARM64 /proc/cpuinfo field layout the
  // guest cpuinfo library parses. The feature list is the ARM64 ISA surface
  // Digitalis translates (matches the HWCAPs the guest sees: LSE 'atomics',
  // FP16 'fphp'/'asimdhp', RDM 'asimdrdm', LRCPC 'lrcpc', DC CVAP 'dcpop',
  // dotprod 'asimddp', AES/PMULL/SHA1/SHA2/CRC32). The implementer/part/variant
  // describe a generic ARMv8 core (0x41 = 'ARM', 0xd05 = Cortex-A55); only the
  // CPU count is taken from the real device, so a guest sees the emulator's
  // actual online core count rather than a hard-coded value.
  std::string out;
  out.reserve(static_cast<size_t>(num_cpus) * 256);
  for (int i = 0; i < num_cpus; ++i) {
    char block[512];
    int n = snprintf(block, sizeof(block),
                     "processor\t: %d\n"
                     "BogoMIPS\t: 38.40\n"
                     "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 "
                     "atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp\n"
                     "CPU implementer\t: 0x41\n"
                     "CPU architecture: 8\n"
                     "CPU variant\t: 0x1\n"
                     "CPU part\t: 0xd05\n"
                     "CPU revision\t: 0\n"
                     "\n",
                     i);
    if (n > 0) {
      out.append(block, static_cast<size_t>(n));
    }
  }
  return out;
}

int ToHostOpenFlags(int guest_flags) {
  int unknown_guest_flags = guest_flags & ~(kCompatibleOpenFlags | kArchSpecificMask);
  if (unknown_guest_flags) {
    TRACE("Unrecognized guest open flags: original=0x%x unsupported=0x%x. Passing to host as is.",
          guest_flags,
          unknown_guest_flags);
  }
  return SwapArchSpecificBits(guest_flags);
}

int ToGuestOpenFlags(int host_flags) {
  int unknown_host_flags = host_flags & ~(kCompatibleOpenFlags | kArchSpecificMask);
  if (unknown_host_flags) {
    TRACE("Unrecognized host open flags: original=0x%x unsupported=0x%x. Passing to guest as is.",
          host_flags,
          unknown_host_flags);
  }
  return SwapArchSpecificBits(host_flags);
}

}  // namespace berberis
