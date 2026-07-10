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
#include <linux/seccomp.h>
#include <linux/unistd.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <time.h>

#include <cerrno>
#include <cstring>

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

#include "epoll_emulation.h"
#include "guest_types.h"

namespace berberis {

namespace {

// AArch64 uses the generic syscall ABI (asm-generic/unistd.h); these numbers are
// the arm64 guest's, NOT the host's. The host runs as x86_64, so its __NR_*
// differ — e.g. guest futex is 98 while host x86_64 __NR_futex is 202 — and must
// never be used to match guest_nr here. Named so the hot-syscall special-cases
// below read by name (mirrors kGuestO* in open_emulation.cc).
constexpr long kGuestNrFutex = 98;           // asm-generic __NR_futex
constexpr long kGuestNrClockGettime = 113;   // asm-generic __NR_clock_gettime
constexpr long kGuestNrUname = 160;          // asm-generic __NR_uname
constexpr long kGuestNrGettimeofday = 169;   // asm-generic __NR_gettimeofday
constexpr long kGuestNrSeccomp = 277;        // asm-generic __NR_seccomp

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

// ioctl translation
//
// Both arm64 and x86_64 use the asm-generic ioctl encoding
// (<asm-generic/ioctl.h>) -- bionic's kernel-headers include
// <asm-generic/ioctl.h> from asm-arm64/asm/ioctl.h and asm-x86/asm/ioctl.h
// alike. The 32-bit cmd layout is:
//
//   bits [31:30] dir | [29:16] size | [15:8] type | [7:0] nr
//
// Both architectures are LP64 (`long` = 8 bytes, pointer = 8 bytes, alignof
// identical), so for ioctls whose payload struct contains only fixed-width
// integers, __u64, pointers, or `long`, the encoded size -- and therefore the
// full cmd -- is bitwise-identical between arm64 and x86_64. This covers every
// Android ioctl family currently exercised by the sample suite and by the
// prebuilt third-party APKs we test against: ASHMEM, binder, fbdev, evdev,
// tty, DRM, V4L2, ION, gralloc, sndcontrol, uhid, etc.
//
// The hook below decodes the cmd into (dir, size, type, nr) so a future
// per-family conversion can be inserted if a struct ever turns up whose
// layout actually differs (none today). For now every cmd is passed through
// unmodified after a structured TRACE.
namespace {

constexpr struct {
  uint8_t type;
  const char* family;
} kKnownIoctlFamilies[] = {
    {0x77, "ashmem"},     // <linux/ashmem.h>           __ASHMEMIOC = 0x77
    {'b', "binder"},      // <linux/android/binder.h>
    {'F', "fbdev"},       // <linux/fb.h>
    {'E', "evdev"},       // <linux/input.h>
    {'T', "tty"},         // <asm-generic/ioctls.h>
    {'d', "drm"},         // <drm/drm.h>
    {'V', "v4l2"},        // <linux/videodev2.h>
    {'I', "ion-iio"},     // <linux/ion.h>, <linux/iio/...>
    {0xF7, "uhid"},       // <linux/uhid.h>
    {'M', "sndcontrol"},  // <sound/asound.h>
    {'U', "sndtimer"},    // <sound/asound.h>           SNDRV_TIMER_*
    {'A', "sndpcm"},      // <sound/asound.h>           SNDRV_PCM_*
    {0x12, "blkdev"},     // <linux/fs.h>               BLK*
};

const char* IoctlFamilyName(uint8_t type) {
  for (const auto& entry : kKnownIoctlFamilies) {
    if (entry.type == type) return entry.family;
  }
  return "unknown";
}

}  // namespace

long RunGuestSyscall___NR_ioctl(long arg_1, long arg_2, long arg_3) {
  const unsigned long cmd = static_cast<unsigned long>(arg_2);
  const unsigned int dir = (cmd >> 30) & 0x3u;        // _IOC_DIR
  const unsigned int size = (cmd >> 16) & 0x3FFFu;    // _IOC_SIZE (14 bits)
  const unsigned int type = (cmd >> 8) & 0xFFu;       // _IOC_TYPE
  const unsigned int nr = cmd & 0xFFu;                // _IOC_NR
  TRACE(
      "ioctl fd=%ld cmd=0x%lx dir=%u type=0x%02x(%s) nr=0x%02x size=%u"
      " -- arm64/x86_64 LP64 cmd identical, passing through",
      arg_1, cmd, dir, type, IoctlFamilyName(static_cast<uint8_t>(type)), nr,
      size);
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

  // vDSO fast path for the hottest time syscalls.
  // RunGuestSyscallImpl forwards via glibc syscall(), which always traps into
  // the kernel; the libc clock_gettime()/gettimeofday() wrappers instead read
  // the host vDSO and avoid kernel entry (measured ~365ns -> ~70ns per call, a
  // 5x win on a clock_gettime-bound loop). We call the wrapper into a host-local
  // buffer (so the vDSO never dereferences a guest pointer) and then copy the
  // result into guest memory via ToHostAddr, exactly as ConvertHostStatToGuestArch
  // does for newfstatat (flat-mapped guest addresses; arm64/x86_64 timespec and
  // timeval layouts are identical under LP64). A null/absent guest pointer falls
  // through to the normal kernel path, which returns EFAULT rather than crashing.
  if (guest_nr == kGuestNrClockGettime && state->cpu.x[1] != 0) {
    struct timespec ts;
    int r = clock_gettime(static_cast<clockid_t>(state->cpu.x[0]), &ts);
    if (r == 0) {
      *ToHostAddr<struct timespec>(state->cpu.x[1]) = ts;
      state->cpu.x[0] = 0;
    } else {
      state->cpu.x[0] = -errno;
    }
    if (kInstrumentSyscalls) {
      OnSyscallReturn(state, guest_nr);
    }
    return;
  }
  if (guest_nr == kGuestNrGettimeofday && state->cpu.x[0] != 0) {
    struct timeval tv;
    struct timezone tz;
    int r = gettimeofday(&tv, &tz);
    if (r == 0) {
      *ToHostAddr<struct timeval>(state->cpu.x[0]) = tv;
      if (state->cpu.x[1] != 0) {
        *ToHostAddr<struct timezone>(state->cpu.x[1]) = tz;
      }
      state->cpu.x[0] = 0;
    } else {
      state->cpu.x[0] = -errno;
    }
    if (kInstrumentSyscalls) {
      OnSyscallReturn(state, guest_nr);
    }
    return;
  }

  // uname: present an arm64 machine to the guest.
  // The host kernel reports machine="x86_64" (and the host kernel release), which
  // is incorrect for a guest that believes it is running on arm64 - cpuinfo is
  // already redirected to an arm64 description, so uname must agree. Anti-emulator
  // / anti-translation native code (e.g. obfuscated device-id SDKs) treats a
  // "x86_64" machine string from an arm64 process as a translation signal. The
  // other utsname fields describe the shared kernel accurately and pass through.
  // utsname is char[]-only and laid out identically for arm64/x86_64 under LP64.
  if (guest_nr == kGuestNrUname && state->cpu.x[0] != 0) {
    struct utsname uts;
    int r = uname(&uts);
    if (r == 0) {
      strncpy(uts.machine, "aarch64", sizeof(uts.machine) - 1);
      uts.machine[sizeof(uts.machine) - 1] = '\0';
      *ToHostAddr<struct utsname>(state->cpu.x[0]) = uts;
      state->cpu.x[0] = 0;
    } else {
      state->cpu.x[0] = -errno;
    }
    if (kInstrumentSyscalls) {
      OnSyscallReturn(state, guest_nr);
    }
    return;
  }

  // seccomp filter neutering
  // Guests that harden themselves with their own seccomp policy (Chromium and
  // Gecko install one in every renderer/GPU process) call seccomp() with a
  // seccomp-bpf program built for the GUEST ABI: its architecture gate compares
  // seccomp_data.arch against AUDIT_ARCH_AARCH64 and its per-syscall checks use
  // AArch64 syscall numbers. Under translation the process actually executes as
  // x86_64 and Berberis issues HOST x86_64 syscalls, so the filter never matches
  // the running architecture -- the first post-install syscall trips
  // SECCOMP_RET_TRAP/KILL and the kernel raises SIGSYS, killing the process. That
  // is exactly what kills every Chromium renderer at launch (Chromium reports
  // termination status 6 / LAUNCH_FAILED), leaving the browser UI process alive
  // but unable to render any page. A guest filter also cannot meaningfully
  // confine the host under translation, because it would govern the translator's
  // own host syscalls (the b/110423578 caveat the prctl path already notes). So
  // do NOT install a guest seccomp filter: report success to the guest while
  // leaving the process unfiltered. The host zygote's native x86_64 seccomp
  // policy still confines the app, so the OS sandbox is unchanged; only the
  // guest's redundant, non-functional extra layer is dropped.
  //
  // SECCOMP_SET_MODE_STRICT (which permits only read/write/exit/sigreturn) would
  // likewise instantly SIGSYS the translator, so it is neutered too. The query
  // operations (SECCOMP_GET_ACTION_AVAIL / SECCOMP_GET_NOTIF_SIZES) install
  // nothing and fall through to the host.
  if (guest_nr == kGuestNrSeccomp) {
    unsigned int operation = static_cast<unsigned int>(state->cpu.x[0]);
    if (operation == SECCOMP_SET_MODE_FILTER || operation == SECCOMP_SET_MODE_STRICT) {
      TRACE(
          "ignoring guest seccomp(operation=%u) under translation: a guest BPF "
          "filter targets the AArch64 ABI and would SIGSYS the x86_64 host",
          operation);
      state->cpu.x[0] = 0;
      if (kInstrumentSyscalls) {
        OnSyscallReturn(state, guest_nr);
      }
      return;
    }
  }

  // futex BSS workaround
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
  if (guest_nr == kGuestNrFutex) {
    long uaddr = state->cpu.x[0];
    int futex_op = static_cast<int>(state->cpu.x[1]) & FUTEX_CMD_MASK;
    if ((futex_op == FUTEX_WAIT || futex_op == FUTEX_WAIT_BITSET) && uaddr != 0) {
      uint32_t actual = *reinterpret_cast<volatile uint32_t*>(uaddr);
      uint32_t expected = static_cast<uint32_t>(futex_arg3);
      if (actual != expected &&
          (actual & 0xFFFF) == (expected & 0xFFFF) &&
          (expected >> 16) == 0 && (actual >> 16) != 0) {
        futex_arg3 = static_cast<long>(static_cast<int32_t>(actual));
      }
    }
  }

  long result = RunGuestSyscallImpl(guest_nr,
                                    state->cpu.x[0],
                                    state->cpu.x[1],
                                    (guest_nr == kGuestNrFutex) ? futex_arg3 : state->cpu.x[2],
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
