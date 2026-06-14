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

#include <atomic>
#include <csignal>
#include <cstring>
#include <memory>
#include <mutex>
// region digitalis
#include <android/log.h>
// endregion

#if defined(__BIONIC__)
#include <platform/bionic/reserved_signals.h>
#endif

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/forever_alloc.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_signal.h"
#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_os_primitives/syscall_numbers.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_opaque.h"
// region digitalis
#include "berberis/guest_state/guest_state.h"
// endregion
#include "berberis/runtime_primitives/crash_reporter.h"
#include "berberis/runtime_primitives/recovery_code.h"

#include "guest_signal_action.h"
#include "guest_thread_manager_impl.h"  // AttachCurrentThread, DetachCurrentThread
#include "scoped_signal_blocker.h"

// Glibc didn't define this macro for i386 and x86_64 at the moment of adding
// its use below. This condition still stands though.
#ifndef SI_FROMKERNEL
#define SI_FROMKERNEL(siptr) ((siptr)->si_code > 0)
#endif

namespace berberis {

namespace {

// Execution cannot proceed until the next pending signals check for _kernel_ sent
// synchronious signals: the faulty instruction will be executed again, leading
// to the infinite recursion. So crash immediately to simplify debugging.
//
// Note that a _user_ sent signal which is typically synchronious, such as SIGSEGV,
// can continue until pending signals check.
bool IsPendingSignalWithoutRecoveryCodeFatal(siginfo_t* info) {
  switch (info->si_signo) {
    case SIGSEGV:
    case SIGBUS:
    case SIGILL:
    case SIGFPE:
      return SI_FROMKERNEL(info);
    default:
      return false;
  }
}

// Technically guest threads may work with different signal action tables, so it's possible to
// optimize by using different mutexes. But it's rather an exotic corner case, so we keep it simple.
std::mutex* GetSignalActionsGuardMutex() {
  static auto* g_mutex = NewForever<std::mutex>();
  return g_mutex;
}

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
// For ARM64, return a nullable pointer so unclaimed signals get the kernel
// default action (terminate) instead of CHECK-failing.
const Guest_sigaction* FindSignalHandler(const GuestSignalActionsTable& signal_actions,
                                         int signal) {
  CHECK_GT(signal, 0);
  CHECK_LE(signal, Guest__KERNEL__NSIG);
  std::lock_guard<std::mutex> lock(*GetSignalActionsGuardMutex());
  return signal_actions.at(signal - 1).TryGetGuestAction();
}
#else
// endregion
const Guest_sigaction* FindSignalHandler(const GuestSignalActionsTable& signal_actions,
                                         int signal) {
  CHECK_GT(signal, 0);
  CHECK_LE(signal, Guest__KERNEL__NSIG);
  std::lock_guard<std::mutex> lock(*GetSignalActionsGuardMutex());
  return &signal_actions.at(signal - 1).GetClaimedGuestAction();
}
// region digitalis
#endif
// endregion

uintptr_t GetHostRegIP(const ucontext_t* ucontext) {
#if defined(__i386__)
  return ucontext->uc_mcontext.gregs[REG_EIP];
#elif defined(__x86_64__)
  return ucontext->uc_mcontext.gregs[REG_RIP];
#elif defined(__riscv)
  return ucontext->uc_mcontext.__gregs[REG_PC];
#elif defined(__aarch64__)
  return ucontext->uc_mcontext.pc;
#else
#error "Unknown host arch"
#endif
}

void SetHostRegIP(ucontext* ucontext, uintptr_t addr) {
#if defined(__i386__)
  ucontext->uc_mcontext.gregs[REG_EIP] = addr;
#elif defined(__x86_64__)
  ucontext->uc_mcontext.gregs[REG_RIP] = addr;
#elif defined(__riscv)
  ucontext->uc_mcontext.__gregs[REG_PC] = addr;
#elif defined(__aarch64__)
  ucontext->uc_mcontext.pc = addr;
#else
#error "Unknown host arch"
#endif
}

// region digitalis - nested-signal recursion depth tracking
// Thread-local count of how many HandleHostSignal frames are active on this
// thread right now. depth > 1 means a nested signal arrived while we were
// already handling one (e.g. breakpad's signal handler itself faulted).
// Used purely for diagnostics — does not change recovery behaviour.
static thread_local int g_handle_host_signal_depth = 0;
// endregion

// Can be interrupted by another HandleHostSignal!
void HandleHostSignal(int sig, siginfo_t* info, void* context) {
  ucontext_t* ucontext = bit_cast<ucontext_t*>(context);
  // region digitalis
  ++g_handle_host_signal_depth;
  int depth = g_handle_host_signal_depth;
  // endregion
  TRACE("Handle host signal %s (%d) at pc=%p si_addr=%p depth=%d",
        strsignal(sig),
        sig,
        bit_cast<void*>(GetHostRegIP(ucontext)),
        info->si_addr,
        depth);

  bool attached;
  GuestThread* thread = AttachCurrentThread(false, &attached);

  // If pending signals are enabled, just add this signal to currently pending.
  // If pending signals are disabled, run handlers for currently pending signals
  // and for this signal now. While running the handlers, enable nested signals
  // to be pending.
  bool prev_pending_signals_enabled = thread->TestAndEnablePendingSignals();
  thread->SetSignalFromHost(*info);
  if (!prev_pending_signals_enabled) {
    CHECK_EQ(GetResidence(*thread->state()), kOutsideGeneratedCode);
    thread->ProcessAndDisablePendingSignals();
    if (attached) {
      DetachCurrentThread();
    }
  } else {
    // We can't make signals pendings as we need to detach the thread!
    CHECK(!attached);

    // Run recovery code to restore precise context and exit generated code.
    uintptr_t addr = GetHostRegIP(ucontext);
    uintptr_t recovery_addr = FindRecoveryCode(addr, thread->state());

    if (recovery_addr) {
      if (!IsConfigFlagSet(kAccurateSigsegv)) {
        // We often get asynchronious signals at instructions with recovery code.
        // This is okay when the recovery is accurate, but highly fragile with inaccurate recovery.
        if (!IsPendingSignalWithoutRecoveryCodeFatal(info)) {
          TRACE("Skipping imprecise context recovery for non-fatal signal");
          TRACE("Guest signal handler suspended, continue");
          // region digitalis
          --g_handle_host_signal_depth;
          // endregion
          return;
        }
        TRACE(
            "Imprecise context at recovery, only guest pc is in sync."
            " Other registers may be stale.");
      }
      SetHostRegIP(ucontext, recovery_addr);
      TRACE("guest signal handler suspended, run recovery for host pc %p at host pc %p",
            bit_cast<void*>(addr),
            bit_cast<void*>(recovery_addr));
      // region digitalis - NULL-page / low-address fault forensics.
      // For any SIGSEGV/SIGBUS where si_addr lies below 4 GB, dump the
      // guest CPU. On Android ARM64 real heap/stack/lib mappings are
      // always >= 0x10000000000, so anything < 0x100000000 is a
      // truncated or bogus pointer. Dump caller PC (x30), arg regs
      // (x0..x7), and callee-saved + iterator regs (x16..x29) — the
      // latter group commonly holds loop pointers (x21 was the buggy
      // value in FB libcoldstart Yoga layout). The
      // "Imprecise context" warning still applies for JIT execution —
      // for interpreter execution state is precise.
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
      if (thread &&
          (sig == SIGSEGV || sig == SIGBUS) &&
          reinterpret_cast<uintptr_t>(info->si_addr) < 0x100000000ULL) {
        auto& cpu = thread->state()->cpu;
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "LowAddr fault: sig=%d si_addr=%p si_code=%d guest_pc=0x%lx "
            "x30(lr)=0x%lx x0=0x%lx x1=0x%lx",
            sig, info->si_addr, info->si_code,
            (unsigned long)cpu.insn_addr,
            (unsigned long)cpu.x[30],
            (unsigned long)cpu.x[0],
            (unsigned long)cpu.x[1]);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  x2=0x%lx x3=0x%lx x4=0x%lx x5=0x%lx x6=0x%lx x7=0x%lx "
            "sp=0x%lx x29(fp)=0x%lx",
            (unsigned long)cpu.x[2], (unsigned long)cpu.x[3],
            (unsigned long)cpu.x[4], (unsigned long)cpu.x[5],
            (unsigned long)cpu.x[6], (unsigned long)cpu.x[7],
            (unsigned long)cpu.sp, (unsigned long)cpu.x[29]);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  x16=0x%lx x17=0x%lx x18=0x%lx x19=0x%lx x20=0x%lx x21=0x%lx "
            "x22=0x%lx x23=0x%lx",
            (unsigned long)cpu.x[16], (unsigned long)cpu.x[17],
            (unsigned long)cpu.x[18], (unsigned long)cpu.x[19],
            (unsigned long)cpu.x[20], (unsigned long)cpu.x[21],
            (unsigned long)cpu.x[22], (unsigned long)cpu.x[23]);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  x24=0x%lx x25=0x%lx x26=0x%lx x27=0x%lx x28=0x%lx",
            (unsigned long)cpu.x[24], (unsigned long)cpu.x[25],
            (unsigned long)cpu.x[26], (unsigned long)cpu.x[27],
            (unsigned long)cpu.x[28]);
      }
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64

      // if this is a NESTED host signal (depth > 1) the
      // first frame is still mid-ProcessGuestSignal (guest handler in flight).
      // Recording the guest CPU here gives forensic data the existing
      // "delivering signal" trace doesn't capture for the inner fault.
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
      if (depth > 1 && thread) {
        auto& cpu = thread->state()->cpu;
        TRACE(
            "NESTED host signal: depth=%d sig=%d host_pc=0x%lx host_recovery=0x%lx "
            "fault_addr=%p si_code=%d guest_insn_addr=0x%lx guest_sp=0x%lx "
            "guest_x30=0x%lx guest_x29=0x%lx",
            depth,
            sig,
            (unsigned long)addr,
            (unsigned long)recovery_addr,
            info->si_addr,
            info->si_code,
            (unsigned long)cpu.insn_addr,
            (unsigned long)cpu.sp,
            (unsigned long)cpu.x[30],
            (unsigned long)cpu.x[29]);
      }
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
      // endregion
    } else {
      // Failed to find recovery code.
      // Translated code should be arranged to continue till
      // the next pending signals check unless it's fatal.
      if (IsPendingSignalWithoutRecoveryCodeFatal(info)) {
        // region digitalis - log fatal no-recovery host signals to logcat.
        // Promoted from TRACE-only (visible only when berberis.tracing is set)
        // to __android_log_print so the diagnostic is captured unconditionally.
        // The FB Katana / breakpad-style second-SEGV path enters this branch
        // when the inner host fault has no Berberis recovery entry. Without
        // the guest CPU dump here, the second-fault's host PC + guest state
        // remain unknown — exactly the gap earlier forensics flagged.
        // Fires for both depth==1 (first fatal) and depth>1 (nested).
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
        if (thread) {
          auto& cpu = thread->state()->cpu;
          __android_log_print(ANDROID_LOG_ERROR, "berberis",
              "FATAL host signal NO-RECOVERY: depth=%d sig=%d host_pc=0x%lx "
              "fault_addr=%p si_code=%d guest_insn_addr=0x%lx",
              depth, sig, (unsigned long)addr,
              info->si_addr, info->si_code,
              (unsigned long)cpu.insn_addr);
          __android_log_print(ANDROID_LOG_ERROR, "berberis",
              "  x0=0x%lx x1=0x%lx x2=0x%lx x3=0x%lx x4=0x%lx x5=0x%lx "
              "x6=0x%lx x7=0x%lx",
              (unsigned long)cpu.x[0], (unsigned long)cpu.x[1],
              (unsigned long)cpu.x[2], (unsigned long)cpu.x[3],
              (unsigned long)cpu.x[4], (unsigned long)cpu.x[5],
              (unsigned long)cpu.x[6], (unsigned long)cpu.x[7]);
          __android_log_print(ANDROID_LOG_ERROR, "berberis",
              "  x16=0x%lx x17=0x%lx x18=0x%lx x19=0x%lx x20=0x%lx x21=0x%lx "
              "x22=0x%lx x23=0x%lx",
              (unsigned long)cpu.x[16], (unsigned long)cpu.x[17],
              (unsigned long)cpu.x[18], (unsigned long)cpu.x[19],
              (unsigned long)cpu.x[20], (unsigned long)cpu.x[21],
              (unsigned long)cpu.x[22], (unsigned long)cpu.x[23]);
          __android_log_print(ANDROID_LOG_ERROR, "berberis",
              "  x24=0x%lx x25=0x%lx x26=0x%lx x27=0x%lx x28=0x%lx "
              "x29(fp)=0x%lx x30(lr)=0x%lx sp=0x%lx",
              (unsigned long)cpu.x[24], (unsigned long)cpu.x[25],
              (unsigned long)cpu.x[26], (unsigned long)cpu.x[27],
              (unsigned long)cpu.x[28], (unsigned long)cpu.x[29],
              (unsigned long)cpu.x[30], (unsigned long)cpu.sp);
        } else {
          __android_log_print(ANDROID_LOG_ERROR, "berberis",
              "FATAL host signal NO-RECOVERY (no guest thread): depth=%d "
              "sig=%d host_pc=0x%lx fault_addr=%p si_code=%d",
              depth, sig, (unsigned long)addr,
              info->si_addr, info->si_code);
        }
#else
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "FATAL host signal NO-RECOVERY: depth=%d sig=%d host_pc=0x%lx "
            "fault_addr=%p si_code=%d",
            depth, sig, (unsigned long)addr,
            info->si_addr, info->si_code);
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
        // endregion
        HandleFatalSignal(sig, info, context);
        // If the raised signal is blocked we may need to return from the handler to unblock it.
        TRACE("Detected return from HandleFatalSignal, continue");
        // region digitalis
        --g_handle_host_signal_depth;
        // endregion
        return;
      }
      TRACE("guest signal handler suspended, continue");
    }
  }
  // region digitalis
  --g_handle_host_signal_depth;
  // endregion
}

bool IsReservedSignal(int signal) {
  switch (signal) {
    // Disallow guest action for SIGABRT to simplify debugging (b/32167022).
    case SIGABRT:
#if defined(__BIONIC__)
    // Disallow overwriting the host profiler handler from guest code. Otherwise
    // guest __libc_init_profiling_handlers() would install its own handler, which
    // is not yet supported for guest code (at least need a proxy for
    // heapprofd_client.so) and fundamentally cannot be supported for host code.
    // TODO(b/167966989): Instead intercept __libc_init_profiling_handlers.
    case BIONIC_SIGNAL_PROFILER:
#endif
      return true;
  }
  return false;
}

}  // namespace

void GuestThread::SetDefaultSignalActionsTable() {
  static auto* g_signal_actions = NewForever<GuestSignalActionsTable>();
  // We need to initialize shared_ptr, but we don't want to attempt to delete the default
  // signal actions when guest thread terminates. Hence we specify a void deleter.
  signal_actions_ = std::shared_ptr<GuestSignalActionsTable>(g_signal_actions, [](auto) {});
}

void GuestThread::CloneSignalActionsTableFrom(GuestSignalActionsTable* from_table) {
  // Need lock to make sure from_table isn't changed concurrently.
  std::lock_guard<std::mutex> lock(*GetSignalActionsGuardMutex());
  signal_actions_ = std::make_shared<GuestSignalActionsTable>(*from_table);
}

// Can be interrupted by another SetSignal!
void GuestThread::SetSignalFromHost(const siginfo_t& host_info) {
  siginfo_t* guest_info = pending_signals_.AllocSignal();

  // Convert host siginfo to guest.
  *guest_info = host_info;
  switch (host_info.si_signo) {
    case SIGILL:
    // region digitalis - BRK raises a synchronous SIGTRAP; like SIGILL/SIGFPE
    // its faulting address is the current guest instruction (the BRK), which a
    // guest breakpoint/sanitizer handler reads to recover the BRK immediate.
    case SIGTRAP:
    // endregion
    case SIGFPE: {
      guest_info->si_addr = ToHostAddr<void>(GetInsnAddr(GetCPUState(*state_)));
      break;
    }
    case SIGSYS: {
      guest_info->si_syscall = ToGuestSyscallNumber(host_info.si_syscall);
      break;
    }
  }

  // This is never interrupted by code that clears queue or status,
  // so the order in which to set them is not important.
  pending_signals_.EnqueueSignal(guest_info);
  // Check that pending signals are not disabled and mark them as present.
  uint8_t old_status = GetPendingSignalsStatusAtomic(*state_).exchange(kPendingSignalsPresent,
                                                                       std::memory_order_relaxed);
  CHECK_NE(kPendingSignalsDisabled, old_status);
}

bool GuestThread::SigAltStack(const stack_t* ss, stack_t* old_ss, int* error) {
  // The following code is not reentrant!
  ScopedSignalBlocker signal_blocker;

  if (old_ss) {
    if (sig_alt_stack_) {
      old_ss->ss_sp = sig_alt_stack_;
      old_ss->ss_size = sig_alt_stack_size_;
      old_ss->ss_flags = IsOnSigAltStack() ? SS_ONSTACK : 0;
    } else {
      old_ss->ss_sp = nullptr;
      old_ss->ss_size = 0;
      old_ss->ss_flags = SS_DISABLE;
    }
  }
  if (ss) {
    if (sig_alt_stack_ && IsOnSigAltStack()) {
      *error = EPERM;
      return false;
    }
    if (ss->ss_flags == SS_DISABLE) {
      sig_alt_stack_ = nullptr;
      sig_alt_stack_size_ = 0;
      return true;
    }
    if (ss->ss_flags != 0) {
      *error = EINVAL;
      return false;
    }
    if (ss->ss_size < GetGuest_MINSIGSTKSZ()) {
      *error = ENOMEM;
      return false;
    }
    sig_alt_stack_ = ss->ss_sp;
    sig_alt_stack_size_ = ss->ss_size;
  }
  return true;
}

void GuestThread::SwitchToSigAltStack() {
  if (sig_alt_stack_ && !IsOnSigAltStack()) {
    // TODO(b/289563835): Try removing `- 16` while ensuring app compatibility.
    // Reliable context on why we use `- 16` here seems to be lost.
    SetStackRegister(GetCPUState(*state_), ToGuestAddr(sig_alt_stack_) + sig_alt_stack_size_ - 16);
  }
}

bool GuestThread::IsOnSigAltStack() const {
  CHECK_NE(sig_alt_stack_, nullptr);
  const char* ss_start = static_cast<const char*>(sig_alt_stack_);
  const char* ss_curr = ToHostAddr<const char>(GetStackRegister(GetCPUState(*state_)));
  return ss_curr >= ss_start && ss_curr < ss_start + sig_alt_stack_size_;
}

void GuestThread::ProcessPendingSignals() {
  for (;;) {
    // Process pending signals while present.
    uint8_t status = GetPendingSignalsStatusAtomic(*state_).load(std::memory_order_acquire);
    CHECK_NE(kPendingSignalsDisabled, status);
    if (status == kPendingSignalsEnabled) {
      return;
    }
    ProcessPendingSignalsImpl();
  }
}

bool GuestThread::ProcessAndDisablePendingSignals() {
  for (;;) {
    // If pending signals are not present, cas should disable them.
    // Otherwise, process pending signals and try again.
    uint8_t old_status = kPendingSignalsEnabled;
    if (GetPendingSignalsStatusAtomic(*state_).compare_exchange_weak(
            old_status, kPendingSignalsDisabled, std::memory_order_acq_rel)) {
      return true;
    }
    if (old_status == kPendingSignalsDisabled) {
      return false;
    }
    ProcessPendingSignalsImpl();
  }
}

bool GuestThread::TestAndEnablePendingSignals() {
  // If pending signals are disabled, cas should mark them enabled.
  // Otherwise, pending signals are already enabled.
  uint8_t old_status = kPendingSignalsDisabled;
  return !GetPendingSignalsStatusAtomic(*state_).compare_exchange_strong(
      old_status, kPendingSignalsEnabled, std::memory_order_acq_rel);
}

// Return if another iteration is needed.
// ATTENTION: Can be interrupted by SetSignal!
void GuestThread::ProcessPendingSignalsImpl() {
  // Clear pending signals status and queue.
  // ATTENTION: It is important to change status before the queue!
  // Otherwise if interrupted by SetSignal, we might end up with
  // no pending signals status but with non-empty queue!
  GetPendingSignalsStatusAtomic(*state_).store(kPendingSignalsEnabled, std::memory_order_relaxed);

  siginfo_t* signal_info;
  while ((signal_info = pending_signals_.DequeueSignalUnsafe())) {
    const Guest_sigaction* sa = FindSignalHandler(*signal_actions_.get(), signal_info->si_signo);
    // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
    if (sa) {
      ProcessGuestSignal(this, sa, signal_info);
      pending_signals_.FreeSignal(signal_info);
    } else {
      int signo = signal_info->si_signo;
      void* fault_addr = signal_info->si_addr;
      pending_signals_.FreeSignal(signal_info);

      {
        auto& cpu = GetCPUState(*state_);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "Guest signal %d (fault_addr=%p, pc=%p) has no handler — "
            "applying default action (terminate)",
            signo, fault_addr, ToHostAddr<void>(GetInsnAddr(cpu)));
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  x0=%p x1=%p x2=%p x3=%p",
            (void*)cpu.x[0], (void*)cpu.x[1], (void*)cpu.x[2], (void*)cpu.x[3]);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  x8=%p x9=%p x10=%p x17=%p x28=%p x29=%p x30=%p",
            (void*)cpu.x[8], (void*)cpu.x[9], (void*)cpu.x[10],
            (void*)cpu.x[17], (void*)cpu.x[28], (void*)cpu.x[29], (void*)cpu.x[30]);
        uint64_t sp_val = cpu.sp;
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  SP=%p", (void*)sp_val);
      }

      ::signal(signo, SIG_DFL);
      raise(signo);
    }
#else
    // endregion
    ProcessGuestSignal(this, sa, signal_info);
    pending_signals_.FreeSignal(signal_info);
    // region digitalis
#endif
    // endregion
  }
}

bool SetGuestSignalHandler(int signal,
                           const Guest_sigaction* act,
                           Guest_sigaction* old_act,
                           int* error) {
#if defined(__riscv)
  TRACE("ATTENTION: SetGuestSignalHandler is unimplemented - skipping it without raising an error");
  return true;
#endif
  if (signal < 1 || signal > Guest__KERNEL__NSIG) {
    *error = EINVAL;
    return false;
  }

  if (act && IsReservedSignal(signal)) {
    TRACE("sigaction for reserved signal %d not set", signal);
    act = nullptr;
  }

  std::lock_guard<std::mutex> lock(*GetSignalActionsGuardMutex());
  GuestSignalAction& action = GetCurrentGuestThread()->GetSignalActionsTable()->at(signal - 1);
  return action.Change(signal, act, HandleHostSignal, old_act, error);
}

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
namespace {

// Fault handler for FaultyLoad/FaultyStore recovery.
// After redirecting to recovery code, queues the signal for guest delivery.
// If the guest has a handler, it runs. If not, ProcessPendingSignalsImpl
// applies the kernel default action (terminate for SIGSEGV/SIGBUS).
void HandleFaultForRecovery(int sig, siginfo_t* info, void* context) {
  ucontext_t* ucontext = bit_cast<ucontext_t*>(context);
  uintptr_t fault_pc = GetHostRegIP(ucontext);

  GuestThread* thread = GetCurrentGuestThread();
  if (thread) {
    uintptr_t recovery = FindRecoveryCode(fault_pc, thread->state());
    if (recovery) {
      SetHostRegIP(ucontext, recovery);

      // Queue the signal for guest delivery. On real hardware, a SIGSEGV
      // during guest execution would be delivered to the guest's signal handler.
      // If no handler is registered, the default action (terminate) applies.
      // This ensures matching behavior under translation.
      // Log for any fault in the first page (NULL + small offset). FB's
      // libcoldstart.so faults at NULL+8 (ldp w22, w23, [x0, #8]); capturing
      // x30 here points at the caller that passed x0=NULL.
      if (reinterpret_cast<uintptr_t>(info->si_addr) < 0x1000) {
        auto& cpu = thread->state()->cpu;
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "Guest NULL-page deref: sig=%d si_addr=%p guest_pc=0x%llx x30(lr)=0x%llx",
            sig,
            info->si_addr,
            (unsigned long long)cpu.insn_addr,
            (unsigned long long)cpu.x[30]);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  x0=0x%llx x1=0x%llx x2=0x%llx x3=0x%llx",
            (unsigned long long)cpu.x[0], (unsigned long long)cpu.x[1],
            (unsigned long long)cpu.x[2], (unsigned long long)cpu.x[3]);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  x4=0x%llx x5=0x%llx x6=0x%llx x7=0x%llx",
            (unsigned long long)cpu.x[4], (unsigned long long)cpu.x[5],
            (unsigned long long)cpu.x[6], (unsigned long long)cpu.x[7]);
        __android_log_print(ANDROID_LOG_ERROR, "berberis",
            "  sp=0x%llx x29(fp)=0x%llx",
            (unsigned long long)cpu.sp, (unsigned long long)cpu.x[29]);
      }
      thread->TestAndEnablePendingSignals();
      thread->SetSignalFromHost(*info);
      return;
    }
  }

  // No recovery found. Re-raise with default handler to terminate.
  ::signal(sig, SIG_DFL);
  raise(sig);
}

}  // namespace

void ClaimHostFaultSignals() {
  // Register a minimal fault handler for SIGSEGV/SIGBUS so that
  // FaultyLoad/FaultyStore recovery works before guest code sets up handlers.
  for (int sig : {SIGSEGV, SIGBUS}) {
    struct sigaction sa = {};
    sa.sa_sigaction = HandleFaultForRecovery;
    sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_NODEFER;
    sigfillset(&sa.sa_mask);
    sigaction(sig, &sa, nullptr);
  }

  // Neutralize a handler-less SIGALRM deadman watchdog.
  //
  // Anti-tamper SDKs (e.g. com.kuaishou.weapon / libweapon982.so) arm an
  // ITIMER_REAL / alarm() / timer_create(SIGALRM) "deadman" timer and expect
  // their integrity check to clear it before it fires; they deliberately leave
  // SIGALRM with no handler so a missed deadline default-terminates the process
  // (a check for debuggers / slow execution). Guest setitimer/timer_create
  // forward straight to the host kernel, which raises a REAL host SIGALRM.
  // Berberis installs no host handler for SIGALRM, so the host default action
  // (Term) kills the process before any guest code runs ("exited due to signal
  // 14 (Alarm clock)"). Under translation the integrity check runs slower than
  // on bare metal, loses the race, and the app dies a few seconds after the SDK
  // loads; on a fast host it passes.
  //
  // Set the baseline host disposition for SIGALRM to SIG_IGN so a handler-less
  // deadman fire is discarded harmlessly. This only changes the UNHANDLED
  // default from Term to Ignore: the moment the guest installs its own SIGALRM
  // action, SetGuestSignalHandler -> GuestSignalAction::Change -> DoSigaction
  // calls host sigaction(SIGALRM, ...) and replaces this SIG_IGN with the
  // claimed guest wrapper (SIGALRM is not an IsReservedSignal), so a guest that
  // legitimately uses alarm()/SIGALRM with a handler is unaffected. We only
  // override when the disposition is still the untouched SIG_DFL. Covers every
  // SIGALRM source (setitimer/alarm/timer_create) uniformly because it acts on
  // the terminal disposition, not the arming syscall. arm64-guest only (this
  // whole function is arm64-only).
  {
    struct sigaction old_sa = {};
    if (sigaction(SIGALRM, nullptr, &old_sa) == 0 && (old_sa.sa_flags & SA_SIGINFO) == 0 &&
        old_sa.sa_handler == SIG_DFL) {
      struct sigaction sa = {};
      sa.sa_handler = SIG_IGN;
      sigemptyset(&sa.sa_mask);
      sigaction(SIGALRM, &sa, nullptr);
      TRACE("Berberis: host SIGALRM default set to SIG_IGN (deadman-watchdog guard)");
    }
  }
}
#endif
// endregion

}  // namespace berberis
