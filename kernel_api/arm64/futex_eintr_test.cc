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

#include <errno.h>
#include <linux/futex.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/kernel_api/run_guest_syscall.h"

namespace berberis {

namespace {

// FUTEX_WAIT delivered with a signal-interrupting handler (no SA_RESTART) must
// return -EINTR back through RunGuestSyscall, encoded in x[0] as the ARM64
// syscall ABI requires: a negative errno value as a sign-extended 64-bit
// integer (i.e. (uint64_t)(int64_t)-EINTR).
//
// Why this matters: the §K4 plan box asks for confirmation that bionic's
// pthread_mutex / pthread_cond, which call __futex_wait_ex and check for
// -EINTR explicitly, are correctly fed the EINTR encoding through the
// translator's syscall path. The host kernel returns -EINTR; libc's syscall(3)
// wrapper converts that to -1 + errno=EINTR; syscall_emulation.cc re-encodes
// it as x[0] = -errno. This test exercises that round-trip end-to-end.

constexpr int kFutexOp = FUTEX_WAIT;
constexpr uint32_t kFutexWord = 0xa5a5a5a5u;
constexpr long kArm64NrFutex = 98;  // arm64 __NR_futex (asm-generic table)

// Set by the SIGUSR1 handler; the handler itself is intentionally a no-op so
// the host kernel's interrupted blocking-syscall path returns -EINTR.
std::atomic<int> g_sigusr1_count{0};

void NoopSigusr1Handler(int /*sig*/) {
  g_sigusr1_count.fetch_add(1, std::memory_order_relaxed);
}

struct KickerArgs {
  pthread_t target;
};

void* KickerThread(void* raw) {
  auto* args = static_cast<KickerArgs*>(raw);
  // Sleep long enough for the main thread to be parked in futex_wait.
  // 50 ms is well above the futex setup overhead and avoids races with
  // a missed wakeup if the signal lands before the main thread blocks.
  const timespec delay{0, 50 * 1000 * 1000};
  nanosleep(&delay, nullptr);
  pthread_kill(args->target, SIGUSR1);
  return nullptr;
}

class FutexEintrTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Install a no-op SIGUSR1 handler WITHOUT SA_RESTART, so the kernel's
    // blocking futex syscall returns -EINTR rather than restarting.
    g_sigusr1_count.store(0, std::memory_order_relaxed);
    struct sigaction sa{};
    sa.sa_handler = NoopSigusr1Handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // explicitly NOT SA_RESTART
    ASSERT_EQ(sigaction(SIGUSR1, &sa, &prev_action_), 0)
        << "install SIGUSR1 handler: " << strerror(errno);
  }

  void TearDown() override {
    sigaction(SIGUSR1, &prev_action_, nullptr);
  }

 private:
  struct sigaction prev_action_{};
};

TEST_F(FutexEintrTest, FutexWaitReturnsEintrOnSignal) {
  // Map a writable page to host the futex word.
  void* page = mmap(nullptr,
                    4096,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0);
  ASSERT_NE(page, MAP_FAILED) << "mmap futex page: " << strerror(errno);
  auto* futex_word = reinterpret_cast<volatile uint32_t*>(page);
  *futex_word = kFutexWord;

  // Set up a guest ThreadState that requests a FUTEX_WAIT on the page word.
  // ARM64 syscall ABI: x0..x5 = args, x8 = syscall number.
  ThreadState state{};
  std::unique_ptr<GuestThread, decltype(&GuestThread::Destroy)> guest_thread(
      GuestThread::CreateForTest(&state), GuestThread::Destroy);
  ASSERT_NE(guest_thread.get(), nullptr);
  state.thread = guest_thread.get();

  state.cpu.x[0] = reinterpret_cast<uint64_t>(futex_word);  // uaddr
  state.cpu.x[1] = static_cast<uint64_t>(kFutexOp);         // op = FUTEX_WAIT
  state.cpu.x[2] = static_cast<uint64_t>(kFutexWord);       // expected value
  state.cpu.x[3] = 0;                                       // timeout = NULL
  state.cpu.x[4] = 0;
  state.cpu.x[5] = 0;
  state.cpu.x[8] = kArm64NrFutex;

  // Spawn the kicker thread which will deliver SIGUSR1 to us once we're
  // parked in the kernel's futex_wait.
  KickerArgs args{pthread_self()};
  pthread_t kicker{};
  ASSERT_EQ(pthread_create(&kicker, nullptr, KickerThread, &args), 0);

  // Blocks until the kicker thread's SIGUSR1 interrupts the syscall.
  RunGuestSyscall(&state);

  ASSERT_EQ(pthread_join(kicker, nullptr), 0);
  ASSERT_EQ(munmap(page, 4096), 0);

  // The host kernel returns -EINTR; syscall_emulation.cc encodes this as
  // x[0] = -errno = -EINTR (sign-extended into 64 bits).
  const uint64_t expected_x0 = static_cast<uint64_t>(static_cast<int64_t>(-EINTR));
  EXPECT_EQ(state.cpu.x[0], expected_x0)
      << "expected x[0] == -EINTR (" << expected_x0 << "), got " << state.cpu.x[0];

  // Sanity: the kicker actually fired (otherwise we'd be hung, not failing).
  EXPECT_GE(g_sigusr1_count.load(std::memory_order_relaxed), 1);
}

}  // namespace

}  // namespace berberis
