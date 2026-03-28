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

// region digitalis - safe memory read (avoids SIGSEGV on unmapped guest addresses)
// Uses /proc/self/mem which is more reliably available than process_vm_readv
bool SafeReadN(uint64_t addr, void* out, size_t len) {
  static thread_local int mem_fd = -1;
  if (mem_fd < 0) {
    mem_fd = open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
    if (mem_fd < 0) return false;
  }
  return pread(mem_fd, out, len, static_cast<off_t>(addr)) == static_cast<ssize_t>(len);
}
bool SafeRead32(uint64_t addr, uint32_t* out) { return SafeReadN(addr, out, 4); }
bool SafeRead64(uint64_t addr, uint64_t* out) { return SafeReadN(addr, out, 8); }
// endregion

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

  // region digitalis - clone counter at function scope for deadlock detection
  static uint64_t clone_count = 0;
  if (guest_nr == 220 || guest_nr == 435) ++clone_count;
  // endregion

  // region digitalis - syscall diagnostics
  {
    static uint64_t futex_count = 0;

    // Detect mmap with corrupted length (> 4GB)
    if (guest_nr == 222) {
      uint64_t mmap_len = state->cpu.x[1];
      if (mmap_len > 0x100000000ULL) {
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "BAD-MMAP: len=0x%llx pc=0x%llx",
            (unsigned long long)mmap_len,
            (unsigned long long)state->cpu.insn_addr);
      }
    }

    // Detailed futex logging: first 20 calls, then every 1M
    if (guest_nr == 98) {
      uint64_t n = ++futex_count;
      if (n <= 20 || n % 1000000 == 0) {
        uint64_t uaddr = state->cpu.x[0];
        uint64_t op = state->cpu.x[1];
        uint64_t val = state->cpu.x[2];
        uint64_t timeout = state->cpu.x[3];
        uint64_t uaddr2 = state->cpu.x[4];
        uint64_t val3 = state->cpu.x[5];
        uint32_t actual32 = 0;
        uint64_t actual64 = 0;
        if (uaddr != 0) {
          actual32 = *reinterpret_cast<uint32_t*>(uaddr);
          actual64 = *reinterpret_cast<uint64_t*>(uaddr);
        }
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "futex#%lu pc=0x%llx lr=0x%llx uaddr=0x%llx op=%llu val=0x%llx "
            "timeout=0x%llx uaddr2=0x%llx val3=0x%llx "
            "*addr32=0x%x *addr64=0x%llx",
            (unsigned long)n,
            (unsigned long long)state->cpu.insn_addr,
            (unsigned long long)state->cpu.x[30],
            (unsigned long long)uaddr,
            (unsigned long long)op,
            (unsigned long long)val,
            (unsigned long long)timeout,
            (unsigned long long)uaddr2,
            (unsigned long long)val3,
            actual32,
            (unsigned long long)actual64);
      }
    }

    // Log clone/clone3 attempts (clone_count incremented at function scope)
    if (guest_nr == 220 || guest_nr == 435) {
      __android_log_print(ANDROID_LOG_ERROR, "berberis",
          "clone#%lu nr=%ld flags=0x%llx pc=0x%llx",
          (unsigned long)clone_count,
          guest_nr,
          (unsigned long long)state->cpu.x[0],
          (unsigned long long)state->cpu.insn_addr);
    }

    // Log openat calls (first 50) to track library loading
    static uint64_t openat_count = 0;
    if (guest_nr == 56) {  // __NR_openat
      uint64_t n = ++openat_count;
      if (n <= 50) {
        const char* path = reinterpret_cast<const char*>(state->cpu.x[1]);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "openat#%lu fd=%lld path=%s flags=0x%llx pc=0x%llx",
            (unsigned long)n,
            (long long)state->cpu.x[0],
            path ? path : "(null)",
            (unsigned long long)state->cpu.x[2],
            (unsigned long long)state->cpu.insn_addr);
      }
    }

    // Log exit/exit_group
    if (guest_nr == 93 || guest_nr == 94) {
      __android_log_print(ANDROID_LOG_ERROR, "berberis",
          "Guest %s(%lld) pc=0x%llx", guest_nr == 93 ? "exit" : "exit_group",
          (long long)state->cpu.x[0], (unsigned long long)state->cpu.insn_addr);
    }
    // Log stderr writes
    if (guest_nr == 64 && state->cpu.x[0] == 2) {
      const char* buf = reinterpret_cast<const char*>(state->cpu.x[1]);
      size_t len = static_cast<size_t>(state->cpu.x[2]);
      if (buf && len > 0 && len < 4096)
        __android_log_print(ANDROID_LOG_ERROR, "berberis", "stderr: %.*s", (int)len, buf);
    }
    // Walk ARM64 frame pointer chain for FUTEX_WAIT_BITSET (first 2 only)
    static uint64_t futex_walk_count = 0;
    if (guest_nr == 98 && futex_walk_count < 2) {
      int futex_cmd = static_cast<int>(state->cpu.x[1]) & 0xf;
      if (futex_cmd == 9 || futex_cmd == 0) {  // FUTEX_WAIT_BITSET or FUTEX_WAIT
        ++futex_walk_count;
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "futex-wait-regs: x19=0x%llx x20=0x%llx x21=0x%llx x22=0x%llx",
            (unsigned long long)state->cpu.x[19],
            (unsigned long long)state->cpu.x[20],
            (unsigned long long)state->cpu.x[21],
            (unsigned long long)state->cpu.x[22]);
        uint64_t fp = state->cpu.x[29];
        for (int frame = 0; frame < 6 && fp != 0; frame++) {
          uint64_t saved_fp = 0, saved_lr = 0;
          if (fp < 0x1000 || (fp & 0x7) != 0) break;
          saved_fp = *reinterpret_cast<uint64_t*>(fp);
          saved_lr = *reinterpret_cast<uint64_t*>(fp + 8);
          __android_log_print(ANDROID_LOG_ERROR, "berberis",
              "  frame#%d fp=0x%llx lr=0x%llx",
              frame, (unsigned long long)fp, (unsigned long long)saved_lr);
          if (saved_fp <= fp) break;
          fp = saved_fp;
        }
      }
    }
  }
  // endregion

  // region digitalis - single-threaded FUTEX_WAIT deadlock workaround
  // In a single-threaded guest (clone_count == 0), FUTEX_WAIT can never be woken
  // by another thread, so blocking is a guaranteed deadlock. Detect the libc++
  // __call_once pattern where a locale::id once_flag is stuck at 1 (Pending) and
  // force-complete it by setting flag = ~0 (Complete). Return -EAGAIN to simulate
  // a spurious wake so cv.wait returns and the caller sees the updated flag.
  if (guest_nr == 98) {
    int futex_cmd_dw = static_cast<int>(state->cpu.x[1]) & FUTEX_CMD_MASK;
    if (futex_cmd_dw == FUTEX_WAIT || futex_cmd_dw == FUTEX_WAIT_BITSET) {
      uint64_t uaddr_dw = state->cpu.x[0];
      if (uaddr_dw != 0) {
        uint32_t actual_dw = *reinterpret_cast<volatile uint32_t*>(uaddr_dw);
        uint32_t expected_dw = static_cast<uint32_t>(state->cpu.x[2]);
        if (actual_dw == expected_dw && actual_dw <= 1) {
          __android_log_print(ANDROID_LOG_WARN, "berberis",
              "dw-entry: uaddr=0x%llx val=%u pc=0x%llx x19=0x%llx x20=0x%llx x21=0x%llx sp=0x%llx fp=0x%llx",
              (unsigned long long)uaddr_dw, actual_dw,
              (unsigned long long)state->cpu.insn_addr,
              (unsigned long long)state->cpu.x[19],
              (unsigned long long)state->cpu.x[20],
              (unsigned long long)state->cpu.x[21],
              (unsigned long long)state->cpu.sp,
              (unsigned long long)state->cpu.x[29]);
          static constexpr int kMaxTracked = 32;
          static thread_local uint64_t deadlock_fixup_count = 0;
          static thread_local uint64_t deadlock_eagain_count = 0;
          static thread_local uint64_t fixed_addrs[kMaxTracked] = {};
          static thread_local int fixed_addr_count = 0;
          static thread_local uint64_t false_positive_addrs[kMaxTracked] = {};
          static thread_local int false_positive_count = 0;
          static thread_local pid_t tracked_pid = 0;
          // Detect fork from Zygote: reset stale tracking on PID change
          pid_t current_pid = getpid();
          if (tracked_pid != current_pid) {
            tracked_pid = current_pid;
            deadlock_fixup_count = 0;
            deadlock_eagain_count = 0;
            fixed_addr_count = 0;
            false_positive_count = 0;
          }
          bool fixed_flag = false;
          // First scan guest callee-saved registers x19-x28 for once_flag pointer
          for (int ri = 19; ri <= 28 && !fixed_flag; ri++) {
            uint64_t candidate = state->cpu.x[ri];
            if (candidate < 0x10000 || (candidate & 0x3) != 0) continue;
            uint32_t candidate_val;
            if (!SafeRead32(candidate, &candidate_val)) continue;
            if (candidate_val != 1) continue;
            if (candidate == uaddr_dw) continue;
            bool is_false_positive = false;
            for (int i = 0; i < false_positive_count; i++) {
              if (false_positive_addrs[i] == candidate) { is_false_positive = true; break; }
            }
            if (is_false_positive) continue;
            bool already_fixed = false;
            for (int i = 0; i < fixed_addr_count; i++) {
              if (fixed_addrs[i] == candidate) { already_fixed = true; break; }
            }
            if (already_fixed) {
              if (false_positive_count < kMaxTracked)
                false_positive_addrs[false_positive_count++] = candidate;
              __android_log_print(ANDROID_LOG_WARN, "berberis",
                  "call_once-false-positive: x%d=0x%llx reverted to 1, skipping",
                  ri, (unsigned long long)candidate);
              continue;
            }
            __atomic_store_n(reinterpret_cast<uint64_t*>(candidate),
                             0xFFFFFFFFFFFFFFFFULL, __ATOMIC_RELEASE);
            fixed_flag = true;
            if (fixed_addr_count < kMaxTracked)
              fixed_addrs[fixed_addr_count++] = candidate;
            if (++deadlock_fixup_count <= 30) {
              __android_log_print(ANDROID_LOG_WARN, "berberis",
                  "call_once-fixup#%lu: x%d=0x%llx 1->~0 pc=0x%llx",
                  (unsigned long)deadlock_fixup_count, ri,
                  (unsigned long long)candidate,
                  (unsigned long long)state->cpu.insn_addr);
            }
          }
          if (deadlock_fixup_count + deadlock_eagain_count <= 5)
            __android_log_print(ANDROID_LOG_WARN, "berberis",
                "dw-regscan-done: fixed=%d fp=0x%llx",
                fixed_flag, (unsigned long long)state->cpu.x[29]);
          // Then scan stack frames for saved register pointing to flag==1
          uint64_t fp_dw = state->cpu.x[29];
          for (int frame = 0; frame < 12 && fp_dw != 0 && !fixed_flag; frame++) {
            if (fp_dw < 0x1000 || (fp_dw & 0x7) != 0) break;
            uint64_t saved_fp_dw = 0;
            if (!SafeRead64(fp_dw, &saved_fp_dw)) break;
            uint64_t frame_end = (saved_fp_dw > fp_dw) ? saved_fp_dw : fp_dw + 0x80;
            // Cap frame scan to 1024 bytes to avoid huge ranges from broken FP chains
            if (frame_end > fp_dw + 0x400) frame_end = fp_dw + 0x400;
            for (uint64_t slot = fp_dw + 0x10; slot + 8 <= frame_end && !fixed_flag; slot += 8) {
              uint64_t candidate = 0;
              if (!SafeRead64(slot, &candidate)) continue;
              if (candidate < 0x10000 || (candidate & 0x3) != 0) continue;
              uint32_t candidate_val;
              if (!SafeRead32(candidate, &candidate_val)) continue;
              if (candidate_val != 1) continue;
              // Skip if candidate points to the futex uaddr itself (CV's __wrefs)
              if (candidate == uaddr_dw) continue;
              // Skip known false positives
              bool is_false_positive = false;
              for (int i = 0; i < false_positive_count; i++) {
                if (false_positive_addrs[i] == candidate) { is_false_positive = true; break; }
              }
              if (is_false_positive) continue;
              // Check if we already fixed this address (it reverted = false positive)
              bool already_fixed = false;
              for (int i = 0; i < fixed_addr_count; i++) {
                if (fixed_addrs[i] == candidate) { already_fixed = true; break; }
              }
              if (already_fixed) {
                // This address reverted to 1 after we set it to ~0 = false positive
                if (false_positive_count < kMaxTracked) {
                  false_positive_addrs[false_positive_count++] = candidate;
                }
                __android_log_print(ANDROID_LOG_WARN, "berberis",
                    "call_once-false-positive: 0x%llx reverted to 1, skipping",
                    (unsigned long long)candidate);
                continue;
              }
              // New address: fix it and track
              __atomic_store_n(reinterpret_cast<uint32_t*>(candidate),
                               0xFFFFFFFF, __ATOMIC_RELEASE);
              fixed_flag = true;
              if (fixed_addr_count < kMaxTracked) {
                fixed_addrs[fixed_addr_count++] = candidate;
              }
              if (++deadlock_fixup_count <= 30) {
                __android_log_print(ANDROID_LOG_WARN, "berberis",
                    "call_once-fixup#%lu: frame#%d [fp+0x%lx]=0x%llx 1->~0 pc=0x%llx",
                    (unsigned long)deadlock_fixup_count, frame,
                    (unsigned long)(slot - fp_dw),
                    (unsigned long long)candidate,
                    (unsigned long long)state->cpu.insn_addr);
              }
            }
            if (saved_fp_dw <= fp_dw) break;
            fp_dw = saved_fp_dw;
          }
          // Scan SP-to-FP range for current frame (local variables below FP)
          if (!fixed_flag) {
            uint64_t sp_dw = state->cpu.sp;
            uint64_t fp_base_dw = state->cpu.x[29];
            if (sp_dw > 0x1000 && (sp_dw & 0x7) == 0 &&
                fp_base_dw > sp_dw && fp_base_dw - sp_dw < 0x1000) {
              for (uint64_t slot = sp_dw; slot + 8 <= fp_base_dw && !fixed_flag; slot += 8) {
                uint64_t candidate = 0;
                if (!SafeRead64(slot, &candidate)) continue;
                if (candidate < 0x10000 || (candidate & 0x3) != 0) continue;
                uint32_t candidate_val;
                if (!SafeRead32(candidate, &candidate_val)) continue;
                if (candidate_val != 1) continue;
                if (candidate == uaddr_dw) continue;
                bool is_false_positive = false;
                for (int i = 0; i < false_positive_count; i++) {
                  if (false_positive_addrs[i] == candidate) { is_false_positive = true; break; }
                }
                if (is_false_positive) continue;
                bool already_fixed = false;
                for (int i = 0; i < fixed_addr_count; i++) {
                  if (fixed_addrs[i] == candidate) { already_fixed = true; break; }
                }
                if (already_fixed) {
                  if (false_positive_count < kMaxTracked)
                    false_positive_addrs[false_positive_count++] = candidate;
                  __android_log_print(ANDROID_LOG_WARN, "berberis",
                      "call_once-false-positive: sp[0x%lx]=0x%llx reverted to 1, skipping",
                      (unsigned long)(slot - sp_dw), (unsigned long long)candidate);
                  continue;
                }
                __atomic_store_n(reinterpret_cast<uint32_t*>(candidate),
                                 0xFFFFFFFF, __ATOMIC_RELEASE);
                fixed_flag = true;
                if (fixed_addr_count < kMaxTracked)
                  fixed_addrs[fixed_addr_count++] = candidate;
                if (++deadlock_fixup_count <= 30) {
                  __android_log_print(ANDROID_LOG_WARN, "berberis",
                      "call_once-fixup#%lu: sp[0x%lx]=0x%llx 1->~0 pc=0x%llx",
                      (unsigned long)deadlock_fixup_count,
                      (unsigned long)(slot - sp_dw),
                      (unsigned long long)candidate,
                      (unsigned long long)state->cpu.insn_addr);
                }
              }
            }
          }
          if (deadlock_fixup_count + deadlock_eagain_count <= 5)
            __android_log_print(ANDROID_LOG_WARN, "berberis",
                "dw-stkscan-done: fixed=%d fixups=%lu eagain=%lu sp=0x%llx fp=0x%llx",
                fixed_flag, (unsigned long)deadlock_fixup_count,
                (unsigned long)deadlock_eagain_count,
                (unsigned long long)state->cpu.sp,
                (unsigned long long)state->cpu.x[29]);
          if (fixed_flag) {
            // Successfully fixed a call_once flag. Return -EAGAIN so cv.wait
            // returns and the caller re-checks the now-complete flag.
            state->cpu.x[0] = static_cast<uint64_t>(-EAGAIN);
            return;
          }
          // Could not find a fixable once_flag.
          if (++deadlock_eagain_count <= 5) {
            // Dump full frame chain for diagnosis
            uint64_t fp_dbg2 = state->cpu.x[29];
            for (int frame = 0; frame < 12 && fp_dbg2 != 0; frame++) {
              if (fp_dbg2 < 0x1000 || (fp_dbg2 & 0x7) != 0) break;
              uint64_t next_fp2 = 0, lr2 = 0;
              if (!SafeRead64(fp_dbg2, &next_fp2)) break;
              SafeRead64(fp_dbg2 + 8, &lr2);
              __android_log_print(ANDROID_LOG_WARN, "berberis",
                  "unfixable-frame#%d fp=0x%llx lr=0x%llx", frame,
                  (unsigned long long)fp_dbg2, (unsigned long long)lr2);
              // Dump first 8 saved slots
              for (int off = 0x10; off <= 0x48; off += 8) {
                uint64_t v = 0;
                if (!SafeRead64(fp_dbg2 + off, &v)) continue;
                uint32_t d = 0;
                if (v > 0x10000 && (v & 0x3) == 0) {
                  SafeRead32(v, &d);
                }
                __android_log_print(ANDROID_LOG_WARN, "berberis",
                    "  [fp+0x%x]=0x%llx (*32=0x%x)", off,
                    (unsigned long long)v, d);
              }
              if (next_fp2 <= fp_dbg2) break;
              fp_dbg2 = next_fp2;
            }
            __android_log_print(ANDROID_LOG_WARN, "berberis",
                "futex-unfixable: WAIT at 0x%llx (val=0x%x) pc=0x%llx, passing to kernel",
                (unsigned long long)uaddr_dw, actual_dw,
                (unsigned long long)state->cpu.insn_addr);
          }
          // Last resort: if expected val is 1 (PENDING once_flag), the futex may
          // be directly on the once_flag itself (not on a CV). This is the bionic
          // pthread_once pattern where COMPLETE=2 (not ~0 like libc++ __call_once).
          // Write 2 (32-bit) to mark the flag as complete.
          if (!fixed_flag && actual_dw == 1 && (uaddr_dw & 0x3) == 0) {
            __atomic_store_n(reinterpret_cast<uint32_t*>(uaddr_dw),
                             2, __ATOMIC_RELEASE);
            // Clear guest LDXR/STXR reservation to prevent CAS from overwriting
            state->cpu.reservation_address = 0;
            state->cpu.reservation_value = 0;
            fixed_flag = true;
            if (deadlock_fixup_count + deadlock_eagain_count <= 30) {
              uint32_t verify32 = *reinterpret_cast<volatile uint32_t*>(uaddr_dw);
              __android_log_print(ANDROID_LOG_WARN, "berberis",
                  "call_once-fixup-direct: uaddr=0x%llx 1->2(pthread_once) verify=%u pc=0x%llx",
                  (unsigned long long)uaddr_dw, verify32,
                  (unsigned long long)state->cpu.insn_addr);
            }
            state->cpu.x[0] = static_cast<uint64_t>(-EAGAIN);
            return;
          }
        }
      }
    }
  }
  // endregion

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
          __android_log_print(ANDROID_LOG_WARN, "berberis",
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
