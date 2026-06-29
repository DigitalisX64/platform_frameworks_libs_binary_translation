/*
 * Copyright (C) 2014 The Android Open Source Project
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

#ifndef BERBERIS_KERNEL_API_OPEN_EMULATION_H_
#define BERBERIS_KERNEL_API_OPEN_EMULATION_H_

// Documentation says that to get access to the constants used below one
// must include these three files.  In reality it looks as if all constants
// are defined by <fcntl.h>.  Including sys/stat.h conflicts with the asm/stat.h
// needed in sys_stat_emulation.h when compliling against musl, so leave it out.
#include <fcntl.h>
// #include <sys/stat.h>
#include <sys/types.h>

// region digitalis
#include <string>
// endregion

namespace berberis {

int ToHostOpenFlags(int guest_flags);
int ToGuestOpenFlags(int host_flags);

int OpenatForGuest(int dirfd, const char* pathname, int flags, mode_t mode);
int OpenForGuest(const char* pathname, int flags, mode_t mode);

bool IsFileDescriptorEmulatedProcSelfMaps(int fd);
void CloseEmulatedProcSelfMapsFileDescriptor(int fd);

extern const char* kGuestCpuinfoPath;

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
// Builds a synthetic guest /proc/cpuinfo for `num_cpus` online CPUs, in the
// ARM64 field layout the guest's cpuinfo library expects. Defined in the
// arch-specific arm64/open_emulation.cc; consumed by OpenatForGuest, which
// serves the result from a memfd instead of opening the static
// kGuestCpuinfoPath file, so the reported core count tracks the real device.
std::string FormatGuestCpuinfo(int num_cpus);
#endif
// endregion digitalis

}  // namespace berberis

#endif  // BERBERIS_KERNEL_API_OPEN_EMULATION_H_
