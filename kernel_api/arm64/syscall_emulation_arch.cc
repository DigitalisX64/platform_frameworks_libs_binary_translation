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

#include <sys/stat.h>

#include <cstddef>
#include <tuple>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/kernel_api/fcntl_emulation.h"
#include "berberis/kernel_api/sys_ptrace_emulation.h"

#include "arm64/guest_types.h"

namespace berberis {

std::tuple<bool, int> GuestFcntlArch(int, int, long) {
  return {false, -1};
}

std::tuple<bool, int> PtraceForGuestArch(int, pid_t, void*, void*) {
  return {false, -1};
}

void ConvertHostStatToGuestArch(const struct stat& host_stat, GuestAddr guest_addr) {
  auto* guest_stat = ToHostAddr<Guest_stat>(guest_addr);
  guest_stat->st_dev = host_stat.st_dev;
  guest_stat->st_ino = host_stat.st_ino;
  guest_stat->st_mode = host_stat.st_mode;
  guest_stat->st_nlink = host_stat.st_nlink;
  guest_stat->st_uid = host_stat.st_uid;
  guest_stat->st_gid = host_stat.st_gid;
  guest_stat->st_rdev = host_stat.st_rdev;
  guest_stat->st_size = host_stat.st_size;
  guest_stat->st_blksize = host_stat.st_blksize;
  guest_stat->st_blocks = host_stat.st_blocks;
  guest_stat->st_atim = host_stat.st_atim;
  guest_stat->st_mtim = host_stat.st_mtim;
  guest_stat->st_ctim = host_stat.st_ctim;
}

}  // namespace berberis
