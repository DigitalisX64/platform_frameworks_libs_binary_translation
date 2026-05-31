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

// Follow-up to an earlier SIGUSR1 stress test: lock down the dual-ABI
// signal-handler dispatch path through GuestSignalAction::Change() for the
// arm64 guest. The hello-jni stress test observed that sa_handler-style
// guest actions (sa_flags=0) appeared not to deliver signals, while
// sa_sigaction-style (sa_flags=SA_SIGINFO) worked. The tests here exercise
// both ABIs side-by-side and verify:
//
//   (1) Change() preserves the guest sa_flags exactly as set (including the
//       no-SA_SIGINFO case), so a later sigaction(NULL, &old) round-trip
//       reads back what the guest wrote.
//   (2) The host kernel ends up with a sigaction installed that uses
//       SA_SIGINFO + the Berberis claimed handler in BOTH cases — that's
//       the design of ConvertGuestSigactionToHost: a guest function pointer
//       (i.e., one that UnwrapHostFunction can't resolve to host code)
//       always routes through the claim path, regardless of guest sa_flags.
//
// If (1) ever drifts, sigaction(NULL, &old) leaks SA_SIGINFO into the guest
// view and breaks bionic's tests for old_act.sa_flags == 0. If (2) ever
// drifts, the host kernel would dispatch the guest handler directly (bypassing
// Berberis), which would crash because guest code isn't host-callable.

#include "gtest/gtest.h"

#include <signal.h>

#include "berberis/guest_os_primitives/guest_signal.h"
#include "berberis/guest_state/guest_addr.h"

#include "guest_signal_action.h"

namespace berberis {

namespace {

// A function in guest space — its host address is intentionally NOT registered
// with WrapHostFunction, so UnwrapHostFunction returns null for ToGuestAddr(&x).
// That forces ConvertGuestSigactionToHost into the "claim" branch — the one
// the hello-jni stress test specifically wanted to verify.
int g_fake_guest_func_anchor;

GuestAddr FakeGuestHandlerAddr() {
  return ToGuestAddr(&g_fake_guest_func_anchor);
}

// Stand-in for HandleHostSignal as the claimed host sigaction. Never actually
// called — the host kernel needs a real function pointer to install, but the
// tests assert against ITS address, not its execution.
void ClaimedHostSaSigaction(int, siginfo_t*, void*) {}

// RAII scope that restores the host sigaction for `sig` on destruction, so
// each test leaves the process signal table as it found it.
class ScopedHostSigaction {
 public:
  explicit ScopedHostSigaction(int sig) : sig_(sig) {
    sigaction(sig_, nullptr, &saved_);
  }
  ~ScopedHostSigaction() { sigaction(sig_, &saved_, nullptr); }

 private:
  int sig_;
  struct sigaction saved_{};
};

}  // namespace

TEST(GuestSignalActionDualAbi, SaHandlerStylePreservesGuestSaFlagsZero) {
  ScopedHostSigaction restore(SIGUSR1);

  GuestSignalAction action;

  // Guest sigaction in the sa_handler-style ABI: handler pointer in
  // guest_sa_sigaction (the union field), sa_flags == 0 (no SA_SIGINFO,
  // no SA_RESTART, no SA_NODEFER). This is the case the hello-jni stress
  // test originally tripped over.
  Guest_sigaction new_sa{};
  new_sa.guest_sa_sigaction = FakeGuestHandlerAddr();
  new_sa.sa_flags = 0;

  Guest_sigaction old_sa{};
  int error = 0;
  EXPECT_TRUE(action.Change(SIGUSR1, &new_sa, ClaimedHostSaSigaction, &old_sa, &error));
  EXPECT_EQ(0, error);

  // The claimed guest action must round-trip the original sa_flags. If
  // Berberis were to ALWAYS leak SA_SIGINFO into the saved guest action
  // (because ConvertGuestSigactionToHost ORs it into host_sa->sa_flags),
  // a later sigaction(NULL, &out) would surface SA_SIGINFO to guest code
  // that never asked for it — and tests like bionic's
  //   sigaction(sig, &act, &old); EXPECT_EQ(0, old.sa_flags)
  // would silently break.
  const Guest_sigaction& claimed = action.GetClaimedGuestAction();
  EXPECT_EQ(FakeGuestHandlerAddr(), claimed.guest_sa_sigaction);
  EXPECT_EQ(0u, claimed.sa_flags & SA_SIGINFO)
      << "sa_handler-style guest action must not carry SA_SIGINFO in the "
         "saved guest view, even though the installed host action does";
  EXPECT_EQ(0u, claimed.sa_flags);
}

TEST(GuestSignalActionDualAbi, SaSigactionStylePreservesGuestSaFlags) {
  ScopedHostSigaction restore(SIGUSR2);

  GuestSignalAction action;

  // Guest sigaction in the sa_sigaction-style ABI: same union field, but
  // sa_flags has SA_SIGINFO set. Tests that the working case (the one
  // hello-jni shipped after pivoting) preserves the guest's SA_SIGINFO bit
  // intact in the saved view.
  Guest_sigaction new_sa{};
  new_sa.guest_sa_sigaction = FakeGuestHandlerAddr();
  new_sa.sa_flags = SA_SIGINFO;

  Guest_sigaction old_sa{};
  int error = 0;
  EXPECT_TRUE(action.Change(SIGUSR2, &new_sa, ClaimedHostSaSigaction, &old_sa, &error));
  EXPECT_EQ(0, error);

  const Guest_sigaction& claimed = action.GetClaimedGuestAction();
  EXPECT_EQ(FakeGuestHandlerAddr(), claimed.guest_sa_sigaction);
  EXPECT_NE(0u, claimed.sa_flags & SA_SIGINFO);
  EXPECT_EQ(static_cast<unsigned long>(SA_SIGINFO), claimed.sa_flags);
}

TEST(GuestSignalActionDualAbi, OldSaRoundTripsBothAbis) {
  ScopedHostSigaction restore(SIGUSR1);

  GuestSignalAction action;

  // First install: sa_handler-style.
  Guest_sigaction new_a{};
  new_a.guest_sa_sigaction = FakeGuestHandlerAddr();
  new_a.sa_flags = 0;
  int error = 0;
  ASSERT_TRUE(action.Change(SIGUSR1, &new_a, ClaimedHostSaSigaction, nullptr, &error));

  // Replace with sa_sigaction-style and capture the previous action into old.
  Guest_sigaction new_b{};
  new_b.guest_sa_sigaction = FakeGuestHandlerAddr();
  new_b.sa_flags = SA_SIGINFO;
  Guest_sigaction old{};
  ASSERT_TRUE(action.Change(SIGUSR1, &new_b, ClaimedHostSaSigaction, &old, &error));

  // old must reflect the previous (sa_handler-style) install exactly.
  EXPECT_EQ(FakeGuestHandlerAddr(), old.guest_sa_sigaction);
  EXPECT_EQ(0u, old.sa_flags & SA_SIGINFO)
      << "Old-sa read-back of a sa_handler-style action must not have "
         "SA_SIGINFO set, or guest sigaction probes will see a flag they "
         "never set";
  EXPECT_EQ(0u, old.sa_flags);
}

TEST(GuestSignalActionDualAbi, HostKernelGetsSigInfoClaimRegardlessOfGuestFlags) {
  // Verifies the design promise: when the guest installs a sa_handler-style
  // action whose handler is in guest space (UnwrapHostFunction returns null),
  // the *host* sigaction ends up with SA_SIGINFO and the Berberis claimed
  // dispatcher — so the kernel runs Berberis on signal delivery, not the
  // guest's raw address (which would be a non-callable host pointer).
  ScopedHostSigaction restore(SIGUSR1);

  GuestSignalAction action;

  Guest_sigaction new_sa{};
  new_sa.guest_sa_sigaction = FakeGuestHandlerAddr();
  new_sa.sa_flags = 0;  // no SA_SIGINFO on the guest side
  int error = 0;
  ASSERT_TRUE(action.Change(SIGUSR1, &new_sa, ClaimedHostSaSigaction, nullptr, &error));

  // Read back the live host sigaction the kernel saw.
  struct sigaction host_out{};
  ASSERT_EQ(0, sigaction(SIGUSR1, nullptr, &host_out));

  EXPECT_NE(0, host_out.sa_flags & SA_SIGINFO)
      << "Host sigaction must carry SA_SIGINFO even when the guest didn't "
         "set it — otherwise the kernel would invoke the guest handler "
         "address directly via sa_handler, which is not host-callable";
  EXPECT_EQ(reinterpret_cast<void (*)(int, siginfo_t*, void*)>(ClaimedHostSaSigaction),
            host_out.sa_sigaction);
}

TEST(GuestSignalActionDualAbi, ResettingSaHandlerStyleToDfl) {
  // Tests the unclaim path: install sa_handler-style, then SIG_DFL.
  ScopedHostSigaction restore(SIGUSR1);

  GuestSignalAction action;

  Guest_sigaction install{};
  install.guest_sa_sigaction = FakeGuestHandlerAddr();
  install.sa_flags = 0;
  int error = 0;
  ASSERT_TRUE(action.Change(SIGUSR1, &install, ClaimedHostSaSigaction, nullptr, &error));

  Guest_sigaction set_dfl{};
  set_dfl.guest_sa_sigaction = Guest_SIG_DFL;
  set_dfl.sa_flags = 0;
  Guest_sigaction old{};
  ASSERT_TRUE(action.Change(SIGUSR1, &set_dfl, ClaimedHostSaSigaction, &old, &error));

  // old still reflects the sa_handler-style action installed first.
  EXPECT_EQ(FakeGuestHandlerAddr(), old.guest_sa_sigaction);
  EXPECT_EQ(0u, old.sa_flags);
}

}  // namespace berberis
