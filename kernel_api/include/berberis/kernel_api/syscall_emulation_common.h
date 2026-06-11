/*
 * Copyright (C) 2019 The Android Open Source Project
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

#ifndef BERBERIS_KERNEL_API_SYSCALL_EMULATION_COMMON_H_
#define BERBERIS_KERNEL_API_SYSCALL_EMULATION_COMMON_H_

#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64) && defined(__ANDROID__)
#include <fcntl.h>
#include <sys/resource.h>

#include <android/fdsan.h>

#ifndef CLOSE_RANGE_CLOEXEC
#define CLOSE_RANGE_CLOEXEC (1U << 2)
#endif
#endif
// endregion

#include "berberis/base/bit_util.h"
#include "berberis/base/macros.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/kernel_api/exec_emulation.h"
#include "berberis/kernel_api/fcntl_emulation.h"
#include "berberis/kernel_api/open_emulation.h"
#include "berberis/kernel_api/sys_prctl_emulation.h"
#include "berberis/kernel_api/sys_ptrace_emulation.h"
#include "berberis/kernel_api/unistd_emulation.h"

namespace berberis {

void ConvertHostStatToGuestArch(const struct stat& host_stat, GuestAddr guest_stat);

inline long RunGuestSyscall___NR_clone3(long arg_1, long arg_2) {
  UNUSED(arg_1, arg_2);
  TRACE("unimplemented syscall __NR_clone3");
  errno = ENOSYS;
  return -1;
}

inline long RunGuestSyscall___NR_close(long arg_1) {
  // TODO(b/346604197): Enable on arm64 once guest_os_primitives is ported.
#ifdef __aarch64__
  UNUSED(arg_1);
  TRACE("unimplemented syscall __NR_close");
  errno = ENOSYS;
  return -1;
#else
  CloseEmulatedProcSelfMapsFileDescriptor(arg_1);
  // region digitalis
  // A guest close() arrives here as a raw syscall, but the host libc keeps an
  // fdsan owner-tag table for this process that raw closes corrupt or trip
  // over, so the host tag state must be reconciled by hand. Three cases:
  //  - No host owner tag on the fd: plain raw close. An already-closed fd
  //    stays a silent EBADF exactly like hardware. (Routing through
  //    android_fdsan_close_with_tag here aborts with "double-close of file
  //    descriptor N detected" when the fd is already closed — that abort took
  //    down apps whose bundled SDKs double-close via raw syscalls.)
  //  - Tagged but already closed: the tag is a stale leftover from an earlier
  //    raw close. Scrub it with android_fdsan_exchange_owner_tag (which has no
  //    error path) so a later host open() landing on this fd number isn't
  //    poisoned (e.g. TinyLoader::OpenFile during CloneGuestThread aborted on
  //    such a stale Parcel tag), then raw-close.
  //  - Tagged and still open: a live host object (Fence, Parcel, unique_fd)
  //    owns this fd. Closing it would yank the fd out from under the host
  //    owner and make its eventual close abort ("expected to be owned by ...,
  //    actually unowned" in Fence::~Fence on the BLASTBufferQueue release
  //    path). Report success without closing and leave the close to the real
  //    owner.
  // Only needed for arm64 guest on Android (riscv64 guest doesn't hit this
  // path yet).
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64) && defined(__ANDROID__)
  uint64_t tag = android_fdsan_get_owner_tag(arg_1);
  if (tag != 0) {
    if (fcntl(static_cast<int>(arg_1), F_GETFD) >= 0) {
      TRACE("guest close(%ld) of live host-owned fd (tag 0x%llx): ignored",
            arg_1,
            static_cast<unsigned long long>(tag));
      return 0;
    }
    android_fdsan_exchange_owner_tag(arg_1, tag, 0);
  }
  return syscall(__NR_close, arg_1);
#else
  return syscall(__NR_close, arg_1);
#endif
  // endregion
#endif
}

// region digitalis
inline long RunGuestSyscall___NR_close_range(long arg_1, long arg_2, long arg_3) {
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64) && defined(__ANDROID__)
  // CLOSE_RANGE_CLOEXEC only marks fds close-on-exec — nothing is destroyed,
  // so the raw syscall is safe (CLOSE_RANGE_UNSHARE just unshares the table).
  if ((arg_3 & CLOSE_RANGE_CLOEXEC) != 0) {
    return syscall(__NR_close_range, arg_1, arg_2, arg_3);
  }
  // A raw close_range() would destroy every fd in range at the kernel level:
  // fds owned by live host objects (Fence, Parcel, unique_fd, DIR*) get yanked
  // out from under their owners — whose later fdsan-checked closes then abort —
  // and host fdsan owner tags are left stale on the freed slots. Emulate it
  // per-fd with the same tag rules as RunGuestSyscall___NR_close. Iterate the
  // numeric range directly rather than enumerating /proc/self/fd, since
  // opendir() would itself allocate a tagged DIR* fd in the middle of fd
  // teardown. The guest passes UINT_MAX for "to the top"; clamp to the fd
  // table's soft limit.
  unsigned long lo = static_cast<unsigned long>(arg_1);
  unsigned long hi = static_cast<unsigned long>(arg_2);
  struct rlimit rl;
  unsigned long max_fd = 4096;
  if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
    max_fd = static_cast<unsigned long>(rl.rlim_cur);
  }
  if (hi > max_fd) {
    hi = max_fd;
  }
  for (unsigned long fd = lo; fd <= hi; ++fd) {
    uint64_t tag = android_fdsan_get_owner_tag(static_cast<int>(fd));
    if (tag != 0) {
      if (fcntl(static_cast<int>(fd), F_GETFD) >= 0) {
        TRACE("guest close_range [%lu, %lu]: skipping live host-owned fd %lu", lo, hi, fd);
        continue;
      }
      android_fdsan_exchange_owner_tag(static_cast<int>(fd), tag, 0);
    }
    syscall(__NR_close, fd);
  }
  return 0;
#else
  return syscall(__NR_close_range, arg_1, arg_2, arg_3);
#endif
}

inline long RunGuestSyscall___NR_dup3(long arg_1, long arg_2, long arg_3) {
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64) && defined(__ANDROID__)
  // dup3 implicitly closes newfd inside the kernel, bypassing host fdsan. If
  // newfd's slot still carries a stale owner tag from an earlier raw close,
  // the guest's new fd would inherit it and poison later host-side closes —
  // scrub it first. A newfd that a live host owner has tagged is yanked by
  // dup3 exactly as on hardware; keep its tag so the owner's eventual
  // tag-matched close stays silent, but trace it since the owner now holds
  // the guest's duplicate.
  uint64_t tag = android_fdsan_get_owner_tag(arg_2);
  if (tag != 0) {
    if (fcntl(static_cast<int>(arg_2), F_GETFD) < 0) {
      android_fdsan_exchange_owner_tag(arg_2, tag, 0);
    } else {
      TRACE("guest dup3 onto live host-owned fd %ld (tag 0x%llx)",
            arg_2,
            static_cast<unsigned long long>(tag));
    }
  }
#endif
  return syscall(__NR_dup3, arg_1, arg_2, arg_3);
}
// endregion

inline long RunGuestSyscall___NR_execve(long arg_1, long arg_2, long arg_3) {
  return static_cast<long>(ExecveForGuest(bit_cast<const char*>(arg_1),     // filename
                                          bit_cast<char* const*>(arg_2),    // argv
                                          bit_cast<char* const*>(arg_3)));  // envp
}

inline long RunGuestSyscall___NR_faccessat(long arg_1, long arg_2, long arg_3) {
  // TODO(b/128614662): translate!
  TRACE("unimplemented syscall __NR_faccessat, running host syscall as is");
  return syscall(__NR_faccessat, arg_1, arg_2, arg_3);
}

inline long RunGuestSyscall___NR_fstat(long arg_1, long arg_2) {
  // TODO(b/346604197): Enable on arm64 once guest_os_primitives is ported.
#ifdef __aarch64__
  UNUSED(arg_1, arg_2);
  TRACE("unimplemented syscall __NR_fstat");
  errno = ENOSYS;
  return -1;
#else
  // We are including this structure from library headers (sys/stat.h) and assume
  // that it matches kernel's layout.
  // TODO(b/232598137): Add a check for this. It seems like this is an issue for 32-bit
  // guest syscall, since compiled with bionic this declares `struct stat64` while
  // the syscall will expect `struct stat`
  struct stat host_stat;
  long result;
  if (IsFileDescriptorEmulatedProcSelfMaps(arg_1)) {
    TRACE("Emulating fstat for /proc/self/maps");
#if defined(__LP64__)
    result = syscall(__NR_newfstatat, AT_FDCWD, "/proc/self/maps", &host_stat, 0);
#else
    result = syscall(__NR_fstatat64, AT_FDCWD, "/proc/self/maps", &host_stat, 0);
#endif
  } else {
    result = syscall(__NR_fstat, arg_1, &host_stat);
  }
  if (result != -1) {
    ConvertHostStatToGuestArch(host_stat, bit_cast<GuestAddr>(arg_2));
  }
  return result;
#endif
}

inline long RunGuestSyscall___NR_fstatfs(long arg_1, long arg_2) {
  // TODO(b/346604197): Enable on arm64 once guest_os_primitives is ported.
#ifdef __aarch64__
  UNUSED(arg_1, arg_2);
  TRACE("unimplemented syscall __NR_fstatfs");
  errno = ENOSYS;
  return -1;
#else
  if (IsFileDescriptorEmulatedProcSelfMaps(arg_1)) {
    TRACE("Emulating fstatfs for /proc/self/maps");
    // arg_2 (struct statfs*) has kernel expected layout, which is different from
    // what libc may expect. E.g. this happens for 32-bit bionic where the library call
    // expects struct statfs64. Thus ensure we invoke syscall, not library call.
    return syscall(__NR_statfs, "/proc/self/maps", arg_2);
  }
  return syscall(__NR_fstatfs, arg_1, arg_2);
#endif
}

inline long RunGuestSyscall___NR_fcntl(long arg_1, long arg_2, long arg_3) {
  return GuestFcntl(arg_1, arg_2, arg_3);
}

inline long RunGuestSyscall___NR_openat(long arg_1, long arg_2, long arg_3, long arg_4) {
  // TODO(b/346604197): Enable on arm64 once guest_os_primitives is ported.
#ifdef __aarch64__
  UNUSED(arg_1, arg_2, arg_3, arg_4);
  TRACE("unimplemented syscall __NR_openat");
  errno = ENOSYS;
  return -1;
#else
  return static_cast<long>(OpenatForGuest(static_cast<int>(arg_1),       // dirfd
                                          bit_cast<const char*>(arg_2),  // path
                                          static_cast<int>(arg_3),       // flags
                                          static_cast<mode_t>(arg_4)));  // mode
#endif
}

inline long RunGuestSyscall___NR_prctl(long arg_1, long arg_2, long arg_3, long arg_4, long arg_5) {
  // TODO(b/346604197): Enable on arm64 once guest_os_primitives is ported.
#ifdef __aarch64__
  UNUSED(arg_1, arg_2, arg_3, arg_4, arg_5);
  TRACE("unimplemented syscall __NR_prctl");
  errno = ENOSYS;
  return -1;
#else
  return PrctlForGuest(arg_1, arg_2, arg_3, arg_4, arg_5);
#endif
}

inline long RunGuestSyscall___NR_ptrace(long arg_1, long arg_2, long arg_3, long arg_4) {
  return static_cast<long>(PtraceForGuest(static_cast<int>(arg_1),    // request
                                          static_cast<pid_t>(arg_2),  // pid
                                          bit_cast<void*>(arg_3),     // addr
                                          bit_cast<void*>(arg_4)));   // data
}

inline long RunGuestSyscall___NR_readlinkat(long arg_1, long arg_2, long arg_3, long arg_4) {
  return static_cast<long>(ReadLinkAtForGuest(static_cast<int>(arg_1),       // dirfd
                                              bit_cast<const char*>(arg_2),  // path
                                              bit_cast<char*>(arg_3),        // buf
                                              bit_cast<size_t>(arg_4)));     // buf_size
}

inline long RunGuestSyscall___NR_rt_sigreturn(long) {
  TRACE("unsupported syscall __NR_rt_sigaction");
  errno = ENOSYS;
  return -1;
}

inline long RunGuestSyscall___NR_statx(long arg_1, long arg_2, long arg_3, long arg_4, long arg_5) {
#if defined(__NR_statx)
  // TODO(b/128614662): add struct statx layout checkers.
  return syscall(__NR_statx, arg_1, arg_2, arg_3, arg_4, arg_5);
#else
  UNUSED(arg_1, arg_2, arg_3, arg_4, arg_5);
  errno = ENOSYS;
  return -1;
#endif
}

long RunUnknownGuestSyscall(long guest_nr,
                            long arg_1,
                            long arg_2,
                            long arg_3,
                            long arg_4,
                            long arg_5,
                            long arg_6) {
  UNUSED(arg_1, arg_2, arg_3, arg_4, arg_5, arg_6);
  TRACE("unknown syscall %ld", guest_nr);
  errno = ENOSYS;
  return -1;
}

}  // namespace berberis

#endif  // BERBERIS_KERNEL_API_SYSCALL_EMULATION_COMMON_H_
