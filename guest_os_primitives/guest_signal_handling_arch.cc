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

// Generic implementation that relies on guest arch-specific headers. This file must be compiled
// separately for each guest architecture.

#include "berberis/guest_os_primitives/guest_signal.h"

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
#include <sys/syscall.h>
#include <unistd.h>

#include <csignal>

#include "berberis/base/gettid.h"
#endif
// endregion
#include "berberis/base/host_signal.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_abi/guest_call.h"
#include "berberis/guest_os_primitives/guest_signal_arch.h"
#include "berberis/guest_os_primitives/guest_thread.h"

#include "guest_context_arch.h"
#include "scoped_signal_blocker.h"

namespace berberis {

void ProcessGuestSignal(GuestThread* thread, const Guest_sigaction* sa, Guest_siginfo_t* info) {
  // ATTENTION: action mask is ADDED to currently blocked signals!
  // Should be no-op if invoked from HandleHostSignal, as it must run under guest action mask!
  HostSigset block_mask;
  ConvertToBigSigset(sa->sa_mask, &block_mask);
  if ((sa->sa_flags & SA_NODEFER) == 0u) {
    HostSigaddset(&block_mask, info->si_signo);
  }
  ScopedSignalBlocker signal_blocker(&block_mask);

  // Save state to ucontext.
  ThreadState* state = thread->state();
  GuestContext ctx;
  ctx.Save(&state->cpu);

  // region digitalis - capture pre-altstack SP for diagnostics (arm64-guest
  // only; consumed by the forensics TRACE below).
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  uint64_t pre_altstack_sp = GetStackRegister(GetCPUState(*state));
#endif
  // endregion

  // Switch to alternate stack.
  if (sa->sa_flags & SA_ONSTACK) {
    thread->SwitchToSigAltStack();
  }

  // region digitalis - signal-delivery forensics for FB-breakpad and similar
  // (arm64-guest only).
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  TRACE(
      "berberis: delivering signal %d to handler=%p guest_pc=0x%lx "
      "fault_addr=%p si_code=%d pre_sp=0x%lx post_sp=0x%lx sa_flags=0x%lx",
      info->si_signo,
      ToHostAddr<void>(sa->guest_sa_sigaction),
      (unsigned long)GetInsnAddr(GetCPUState(*state)),
      info->si_addr,
      info->si_code,
      (unsigned long)pre_altstack_sp,
      (unsigned long)GetStackRegister(GetCPUState(*state)),
      (unsigned long)sa->sa_flags);
#endif
  // endregion
  TRACE("delivering signal %d at %p", info->si_signo, ToHostAddr<void>(sa->guest_sa_sigaction));
  // We get here only if guest set a custom signal action, default actions are handled by host.
  CHECK_NE(sa->guest_sa_sigaction, Guest_SIG_DFL);
  CHECK_NE(sa->guest_sa_sigaction, Guest_SIG_IGN);
  CHECK_NE(sa->guest_sa_sigaction, Guest_SIG_ERR);
  // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  // An ARM64 signal handler entry PC must be 4-byte aligned; a misaligned
  // handler cannot execute — real hardware raises a PC-alignment fault and, for
  // a synchronous fatal signal masked in its own handler, terminates the
  // process. Some hardened/anti-tamper libraries deliberately register random,
  // misaligned handlers as tamper traps that are never meant to fire; delivering
  // to one here makes the interpreter fetch 4-byte-misaligned garbage, raise
  // SIGILL, deliver that to another misaligned handler, and livelock forever
  // (the app "hangs at a black screen", pinning a CPU). Match hardware: take the
  // signal's default action (terminate) instead of jumping into garbage.
  if ((static_cast<uint64_t>(sa->guest_sa_sigaction) & 3) != 0) {
    TRACE("berberis: guest signal %d handler=%p is misaligned (invalid ARM64 PC); "
          "taking default action (terminate) instead of livelocking",
          info->si_signo,
          ToHostAddr<void>(sa->guest_sa_sigaction));
    // Reset to the host default disposition and re-raise so the host kernel
    // performs the default action for this signal (terminate + tombstone).
    struct sigaction dfl {};
    dfl.sa_handler = SIG_DFL;
    sigaction(info->si_signo, &dfl, nullptr);
    syscall(__NR_tgkill, GetpidSyscall(), GettidSyscall(), info->si_signo);
    // The re-raised signal fires once the ScopedSignalBlocker at the top of this
    // function unblocks it on return; nothing below should run.
    return;
  }
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
  // endregion
  // Run guest signal handler. Assume this is
  //   void (*sa_sigaction)(int, siginfo_t*, void*);
  // If this is actually
  //   void (*sa_handler)(int);
  // then extra args will be just ignored.
  GuestCall guest_call;
  guest_call.AddArgInt32(info->si_signo);
  guest_call.AddArgGuestAddr(ToGuestAddr(info));
  guest_call.AddArgGuestAddr(ToGuestAddr(ctx.ptr()));
  guest_call.RunVoid(sa->guest_sa_sigaction);
  TRACE("signal %d delivered", info->si_signo);

  // Restore state from ucontext, it may be updated by the handler.
  ctx.Restore(&state->cpu);
}

}  // namespace berberis
