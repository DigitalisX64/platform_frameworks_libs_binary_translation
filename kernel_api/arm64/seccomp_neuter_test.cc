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

#include "gtest/gtest.h"

#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>

#include <cstdint>
#include <thread>

#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/kernel_api/run_guest_syscall.h"

namespace berberis {

namespace {

constexpr long kArm64NrSeccomp = 277;  // arm64 __NR_seccomp (asm-generic table)

// A guest that installs a seccomp-bpf filter via seccomp(SECCOMP_SET_MODE_FILTER)
// builds that filter for the GUEST ABI: its architecture gate compares
// seccomp_data.arch against AUDIT_ARCH_AARCH64 and its per-syscall checks use
// AArch64 syscall numbers. Berberis, however, executes HOST x86_64 syscalls, so
// such a filter never matches the running architecture and the first post-install
// syscall traps SECCOMP_RET_TRAP/KILL -> SIGSYS, killing the process (this is what
// kills every Chromium renderer at launch). A guest filter also cannot meaningfully
// sandbox the host under translation, since it would govern the translator's own
// host syscalls. RunGuestSyscall must therefore NOT install a guest seccomp filter:
// it reports success to the guest but leaves the process unfiltered (the host
// zygote's native seccomp policy still confines it).
//
// This runs on a dedicated thread so that, were the neutering to regress and a
// filter actually be installed, it stays confined to that thread (seccomp without
// TSYNC is per-thread) and is torn down on join -- it cannot leak into the rest of
// the test binary.
TEST(SeccompNeuterTest, SetModeFilterReportsSuccessButInstallsNothing) {
  struct Result {
    bool setup_ok = false;
    long guest_ret = -1;
    int seccomp_mode_pre = -1;
    int seccomp_mode_post = -1;
  } r;

  std::thread worker([&r] {
    // A real seccomp(SET_MODE_FILTER) needs NO_NEW_PRIVS (or CAP_SYS_ADMIN). Set
    // it so that, absent our neutering, the host kernel WOULD install the filter --
    // making "no filter installed afterwards" a meaningful post-condition rather
    // than an artifact of insufficient privilege.
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
      return;
    }
    r.seccomp_mode_pre = prctl(PR_GET_SECCOMP);  // expect 0: no filter yet

    // An allow-all program: if neutering regressed and the kernel installed this,
    // the thread would keep running (RET_ALLOW), but PR_GET_SECCOMP would report
    // SECCOMP_MODE_FILTER (2). That lets the test observe a regression without
    // SIGSYS-killing the test binary.
    struct sock_filter insns[] = {
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog prog = {
        .len = static_cast<unsigned short>(1),
        .filter = insns,
    };

    ThreadState state{};
    GuestThread* guest_thread = GuestThread::CreateForTest(&state);
    if (guest_thread == nullptr) {
      return;
    }
    state.thread = guest_thread;

    // ARM64 syscall ABI: x0..x5 = args, x8 = syscall number.
    state.cpu.x[0] = static_cast<uint64_t>(SECCOMP_SET_MODE_FILTER);  // operation
    state.cpu.x[1] = 0;                                               // flags
    state.cpu.x[2] = reinterpret_cast<uint64_t>(&prog);              // args (filter)
    state.cpu.x[3] = 0;
    state.cpu.x[4] = 0;
    state.cpu.x[5] = 0;
    state.cpu.x[8] = kArm64NrSeccomp;

    RunGuestSyscall(&state);

    r.guest_ret = static_cast<long>(state.cpu.x[0]);
    r.seccomp_mode_post = prctl(PR_GET_SECCOMP);
    r.setup_ok = true;

    GuestThread::Destroy(guest_thread);
  });
  worker.join();

  ASSERT_TRUE(r.setup_ok) << "worker setup failed (NO_NEW_PRIVS or GuestThread)";
  EXPECT_EQ(r.seccomp_mode_pre, 0) << "thread should start unfiltered";
  // The guest is told the sandbox installed successfully...
  EXPECT_EQ(r.guest_ret, 0) << "guest seccomp() should return success";
  // ...but no filter is actually applied to the (host) process/thread.
  EXPECT_EQ(r.seccomp_mode_post, 0)
      << "no seccomp filter should be installed under translation (got mode "
      << r.seccomp_mode_post << ")";
}

}  // namespace

}  // namespace berberis
