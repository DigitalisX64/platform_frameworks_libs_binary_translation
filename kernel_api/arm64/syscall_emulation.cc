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

#include <fcntl.h>  // AT_FDCWD, AT_SYMLINK_NOFOLLOW
#include <linux/futex.h>
#include <linux/sched.h>
#include <linux/unistd.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>

#include <cerrno>

#include "berberis/base/macros.h"
#include "berberis/base/scoped_errno.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/instrument/syscall.h"
#include "berberis/kernel_api/main_executable_real_path_emulation.h"
#include "berberis/kernel_api/runtime_bridge.h"
#include "berberis/kernel_api/syscall_emulation_common.h"

#include "berberis/guest_os_primitives/scoped_pending_signals.h"
#include "berberis/runtime_primitives/runtime_library.h"
// region digitalis
#include <android/log.h>
// endregion

#include "epoll_emulation.h"
#include "guest_types.h"

namespace berberis {

namespace {

int FstatatForGuest(int dirfd, const char* path, struct stat* buf, int flags) {
  const char* real_path = nullptr;
  if ((flags & AT_SYMLINK_NOFOLLOW) == 0) {
    real_path = TryReadLinkToMainExecutableRealPath(path);
  }
  return syscall(__NR_newfstatat, dirfd, real_path ? real_path : path, buf, flags);
}

long RunGuestSyscall___NR_execveat(long arg_1, long arg_2, long arg_3, long arg_4, long arg_5) {
  UNUSED(arg_1, arg_2, arg_3, arg_4, arg_5);
  TRACE("unimplemented syscall __NR_execveat");
  errno = ENOSYS;
  return -1;
}

// sys_fadvise64 has a different entry-point symbol name between arm64 and x86_64.
#ifdef __x86_64__
long RunGuestSyscall___NR_fadvise64(long arg_1, long arg_2, long arg_3, long arg_4) {
  // on 64-bit architectures, sys_fadvise64 and sys_fadvise64_64 are equal.
  return syscall(__NR_fadvise64, arg_1, arg_2, arg_3, arg_4);
}
#endif

long RunGuestSyscall___NR_ioctl(long arg_1, long arg_2, long arg_3) {
  // TODO(b/128614662): translate!
  TRACE("unimplemented ioctl 0x%lx, running host syscall as is", arg_2);
  return syscall(__NR_ioctl, arg_1, arg_2, arg_3);
}

long RunGuestSyscall___NR_newfstatat(long arg_1, long arg_2, long arg_3, long arg_4) {
  struct stat host_stat;
  int result = FstatatForGuest(static_cast<int>(arg_1),       // dirfd
                               bit_cast<const char*>(arg_2),  // path
                               &host_stat,
                               static_cast<int>(arg_4));  // flags
  if (result != -1) {
    ConvertHostStatToGuestArch(host_stat, bit_cast<GuestAddr>(arg_3));
  }
  return result;
}

// RunGuestSyscallImpl.
// ARM64 uses the same generic Linux syscall table as RISC-V.
// The syscall numbers are identical, so we reuse the same translation table.
#if defined(__x86_64__)
#include "gen_syscall_emulation_arm64_to_x86_64-inl.h"
#else
#error "Unsupported host arch"
#endif

}  // namespace

void RunGuestSyscall(ThreadState* state) {
  // ATTENTION: run guest signal handlers instantly!
  // If signal arrives while in a syscall, syscall should immediately return with EINTR.
  // In this case pending signals are OK, as guest handlers will run on return from syscall.
  // BUT, if signal action has SA_RESTART, certain syscalls will restart instead of returning.
  // In this case, pending signals will never run...
  ScopedPendingSignalsDisabler scoped_pending_signals_disabler(state->thread);
  ScopedErrno scoped_errno;

  // ARM64 Linux takes arguments in x0-x5 and syscall number in x8.
  long guest_nr = state->cpu.x[8];

  if (kInstrumentSyscalls) {
    OnSyscall(state, guest_nr);
  }

  // region digitalis - futex BSS workaround
  // Bionic's pthread_mutex uses 16-bit atomics for the state field (offset 0-1),
  // leaving the adjacent __pad field (offset 2-3) untouched. When __futex_wait_ex
  // passes the 16-bit state as the expected 32-bit value, it assumes __pad is zero.
  // However, if the mutex is in a .bss section whose partial page wasn't zeroed by
  // the guest linker's memset (e.g., file data left behind), the __pad bytes contain
  // garbage, causing the kernel's 32-bit comparison to fail (EAGAIN) and the thread
  // to spin instead of sleeping.
  //
  // Fix: for FUTEX_WAIT/FUTEX_WAIT_BITSET, if the lower 16 bits of the expected
  // value match the actual 32-bit word but the upper 16 bits differ (expected has
  // upper=0, actual has upper=garbage), substitute the actual value so the kernel
  // comparison succeeds and the thread properly sleeps.
  long futex_arg3 = state->cpu.x[2];
  if (guest_nr == 98) {  // __NR_futex
    long uaddr = state->cpu.x[0];
    int futex_op = static_cast<int>(state->cpu.x[1]) & FUTEX_CMD_MASK;
    if ((futex_op == FUTEX_WAIT || futex_op == FUTEX_WAIT_BITSET) && uaddr != 0) {
      uint32_t actual = *reinterpret_cast<volatile uint32_t*>(uaddr);
      uint32_t expected = static_cast<uint32_t>(futex_arg3);
      if (actual != expected &&
          (actual & 0xFFFF) == (expected & 0xFFFF) &&
          (expected >> 16) == 0 && (actual >> 16) != 0) {
        static uint64_t fixup_count = 0;
        if (++fixup_count <= 5) {
          __android_log_print(ANDROID_LOG_DEBUG, "berberis",
              "futex-fixup#%lu: uaddr=0x%llx actual=0x%x expected=0x%x -> using 0x%x",
              (unsigned long)fixup_count, (unsigned long long)uaddr, actual, expected, actual);
        }
        futex_arg3 = static_cast<long>(static_cast<int32_t>(actual));
      }
    }
  }
  // endregion

  long result = RunGuestSyscallImpl(guest_nr,
                                    state->cpu.x[0],
                                    state->cpu.x[1],
                                    (guest_nr == 98) ? futex_arg3 : state->cpu.x[2],
                                    state->cpu.x[3],
                                    state->cpu.x[4],
                                    state->cpu.x[5]);
  if (result == -1) {
    state->cpu.x[0] = -errno;
  } else {
    state->cpu.x[0] = result;
  }

  if (kInstrumentSyscalls) {
    OnSyscallReturn(state, guest_nr);
  }
}

}  // namespace berberis
// endregion
