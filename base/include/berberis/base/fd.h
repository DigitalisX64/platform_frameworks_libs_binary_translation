/*
 * Copyright (C) 2021 The Android Open Source Project
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

// File operations without libc. Most important is not touching thread-local errno.

#ifndef BERBERIS_BASE_FD_H_
#define BERBERIS_BASE_FD_H_

#include <linux/unistd.h>
#include <sys/mman.h>
#include <unistd.h>

// region digitalis
#include <sys/stat.h>

#include <cerrno>
#include <cstdint>

#if defined(__ANDROID__)
#include <android/fdsan.h>
#endif
// endregion

#include "berberis/base/bit_util.h"
#include "berberis/base/logging.h"
#include "berberis/base/raw_syscall.h"

namespace berberis {

inline int CreateMemfdOrDie(const char* name) {
  // Use MFD_CLOEXEC to avoid leaking the file descriptor to child processes.
  int fd = static_cast<int>(RawSyscall(__NR_memfd_create, bit_cast<long>(name), MFD_CLOEXEC));
  CHECK(fd >= 0);
  return fd;
}

inline void FtruncateOrDie(int fd, off64_t size) {
  // Call libc instead of syscall because we want 64 version and do not want to
  // do ifdefs for 32/64/glibc/bionic in order to get the correct one.
  CHECK_EQ(ftruncate64(fd, size), 0);
}

inline void WriteFullyOrDie(int fd, const void* data, size_t size) {
  auto* curr = bit_cast<const uint8_t*>(data);
  auto* end = curr + size;
  while (curr < end) {
    auto written = RawSyscall(__NR_write, fd, bit_cast<long>(curr), end - curr);
    // It is not clear if write syscall can return 0 when writing more than 0 bytes.
    if (written >= 0) {
      curr += written;
    } else {
      CHECK(written == -EINTR);
    }
  }
}

inline void CloseUnsafe(int fd) {
  RawSyscall(__NR_close, fd);
}

// region digitalis
// Translator-internal fds live in the same fd table as the guest app's fds, so
// guest fd hygiene can destroy them: a forked child sweeping "its" descriptors
// before exec (Chromium's CloseSuperfluousFds, posix_spawn file actions) closes
// every fd it doesn't recognize. The guest close/close_range emulation in
// kernel_api spares fds that carry a live host fdsan owner tag, but fds Berberis
// creates for itself (the TableOfTables memfds, exec-region memfds) were
// untagged, so a sweep raw-closed them and the next translator use of the
// cached fd died on EBADF (seen as a MmapImplOrDie abort in
// TableOfTables::AllocateIfNecessary when a Chromium fork child reset its
// signal handlers). Tag such fds as host-owned so the existing guards skip
// them, and close them with their tag so no stale tag is left on the slot.
// fdsan is bionic-only; on other hosts these degrade to a raw close.
#if defined(__ANDROID__)
inline uint64_t HostOwnedFdTag() {
  // Any stable non-zero cookie works; debuggerd shows it as a generic native
  // owner in the "open files" list. bionic's enum has no translator slot.
  return android_fdsan_create_owner_tag(ANDROID_FDSAN_OWNER_TYPE_GENERIC_00,
                                        reinterpret_cast<uint64_t>(&CloseUnsafe));
}
#endif

inline void TagHostOwnedFdUnsafe(int fd) {
#if defined(__ANDROID__)
  // Scrub any stale tag left on this slot by an earlier raw close, then claim
  // the fd. exchange has no error path and does not touch errno.
  android_fdsan_exchange_owner_tag(fd, android_fdsan_get_owner_tag(fd), HostOwnedFdTag());
#else
  static_cast<void>(fd);
#endif
}

inline void CloseHostOwnedFdUnsafe(int fd) {
#if defined(__ANDROID__)
  // Close with the fd's current tag: always passes the fdsan check and clears
  // the table entry (same pattern as ScopedFd). Preserve errno per this file's
  // contract.
  int saved_errno = errno;
  android_fdsan_close_with_tag(fd, android_fdsan_get_owner_tag(fd));
  errno = saved_errno;
#else
  RawSyscall(__NR_close, fd);
#endif
}

// Identity of the open file description behind an fd. Two descriptors name the
// same file iff their (st_dev, st_ino) match. Used to detect a translator-owned
// fd being silently replaced: the fdsan host-owner tag stops a guest
// close/close_range sweep from raw-closing our fds, but a guest dup2/dup3 onto
// one of our fd numbers is legal POSIX (the emulation only TRACEs and proceeds),
// so the kernel swaps our file description for the guest's target. A cached fd
// can re-fstat its identity and heal if it no longer matches.
struct FdIdentity {
  uint64_t dev = 0;
  uint64_t ino = 0;
  bool valid = false;  // false when the fd cannot be fstat'd (closed/invalid).
};

inline FdIdentity GetFdIdentityUnsafe(int fd) {
  // Preserve errno per this file's contract.
  int saved_errno = errno;
  FdIdentity id;
  struct stat st;
  if (fstat(fd, &st) == 0) {
    id.dev = static_cast<uint64_t>(st.st_dev);
    id.ino = static_cast<uint64_t>(st.st_ino);
    id.valid = true;
  }
  errno = saved_errno;
  return id;
}

inline bool FdIdentityMatches(const FdIdentity& a, const FdIdentity& b) {
  return a.valid && b.valid && a.dev == b.dev && a.ino == b.ino;
}
// endregion

}  // namespace berberis

#endif  // BERBERIS_BASE_FD_H_
