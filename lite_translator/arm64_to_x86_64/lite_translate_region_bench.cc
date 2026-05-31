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

// Digitalis translator-throughput microbenchmark.
//
// Times tight ARM64 guest loops end-to-end through the real dispatch path
// (`ExecuteGuest`), so it measures JIT-generated code quality AND region/
// indirect-branch dispatch cost — the things the B3/B4 performance work
// optimizes. Each kernel is a self-contained guest loop (counter in x0); the
// loop falls through to a stop PC when the counter reaches zero.
//
// This is NOT a correctness test: it lives in its own `DigitalisBench` suite so
// the `--gtest_filter='Arm64*'` correctness gate never runs it. Invoke it
// explicitly:
//
//   berberis_arm64_host_tests --gtest_filter='DigitalisBench.*'
//
// Each kernel prints one line:
//   BENCH <name>: <ns> ns / <guest_insns> insns = <Mips> Mips (median of N)
// where Mips = millions of guest instructions retired per second. Compare the
// Mips figure before/after a change to gate a perf commit on a measured delta.

#include "gtest/gtest.h"

#include <time.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime/berberis.h"
#include "berberis/runtime/execute_guest.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {
namespace {

// Assembled from /tmp/bench_kernels.S (see comments for the source mnemonics);
// verified with llvm-objdump. Each array is loop-entry at index 0; `stop_off`
// is the byte offset of the instruction the loop falls through to on exit.

// integer ALU: 16 int ops + subs/b.ne per iter.
constexpr uint32_t kInt[] = {
    0x8b020021, 0xca040063, 0xcb030042, 0xaa010084, 0x8b0600a5, 0xca0100c6,
    0xcb0300e7, 0x8a040108, 0x8b050021, 0xca060063, 0xcb070042, 0xaa080084,
    0x8b0200a5, 0xca0400c6, 0xd37ff8e7, 0x8b010108, 0xf1000400, 0x54fffde1,
};  // 18 words; stop = 18*4. ops/iter executed = 18.

// branch-dense: 3 cmp+cond-branch + adds, subs/b.ne.
constexpr uint32_t kBranch[] = {
    0xeb02003f, 0x5400004b, 0x91000463, 0xeb04007f, 0x5400004a, 0xd10004a5,
    0xeb0100bf, 0x54000040, 0x8b0300c6, 0x91000421, 0xf1000400, 0x54fffea1,
};  // 12 words; stop = 12*4. ~ up to 12 ops/iter (some branches skip).

// call/return: 4×(BL leaf; ...) leaf={add x9,#1; ret}. Exercises indirect
// dispatch (RET) heavily. Layout: [bl×4, subs, b.ne, <main-ret=stop>, leaf-add,
// leaf-ret]. BL at index0 targets index7 (leaf), +28 bytes.
constexpr uint32_t kCall[] = {
    0x94000007, 0x94000006, 0x94000005, 0x94000004, 0xf1000400, 0x54ffff61,
    0xd65f03c0, 0x91000529, 0xd65f03c0,
};  // stop = 6*4 (the main ret landmark). guest insns/iter = 4 bl + 4*(add+ret)+subs+bne = 14.

// NEON arithmetic: 8 vector ops + subs/b.ne.
constexpr uint32_t kNeon[] = {
    0x4ea28421, 0x4ea49c63, 0x6ea18442, 0x4e26d4a5, 0x6e27dcc6, 0x4ea18508,
    0x6e231c84, 0x4e25d4e7, 0xf1000400, 0x54fffee1,
};  // 10 words; stop = 10*4. ops/iter = 10.

// scalar FP: 8 double ops + subs/b.ne.
constexpr uint32_t kFp[] = {
    0x1e622821, 0x1e640863, 0x1e613842, 0x1e6328a5, 0x1e650884, 0x1e6128c6,
    0x1e6418e7, 0x1e662842, 0xf1000400, 0x54fffee1,
};  // 10 words; stop = 10*4. ops/iter = 10.

// memcpy: 4 ldr/str pairs + subs/b.ne. x1=src, x2=dst.
constexpr uint32_t kMem[] = {
    0xf9400023, 0xf9000043, 0xf9400424, 0xf9000444, 0xf9400825, 0xf9000845,
    0xf9400c26, 0xf9000c46, 0xf1000400, 0x54fffee1,
};  // 10 words; stop = 10*4. ops/iter = 10.

struct Kernel {
  const char* name;
  const uint32_t* code;
  size_t n_words;
  uint32_t stop_off;     // byte offset of the fall-through (stop) instruction
  uint64_t iters;        // loop trip count
  uint64_t insns_iter;   // guest instructions retired per iteration
};

double TimeOnce(const Kernel& k) {
  // Copy the kernel into a fresh page-aligned buffer so each run translates
  // anew is NOT what we want — keep the same buffer across runs so the cache is
  // warm. Caller passes a stable buffer via static storage below.
  static std::vector<uint32_t> buf;
  buf.assign(k.code, k.code + k.n_words);
  GuestAddr base = ToGuestAddr(buf.data());
  GuestMapShadow::GetInstance()->SetExecutable(base, k.n_words * 4);
  GuestThread* thread = GetCurrentGuestThread();
  auto& cpu = thread->state()->cpu;
  GuestAddr stop = base + k.stop_off;
  auto* cache = TranslationCache::GetInstance();

  // Warm the translation cache once (untimed).
  cpu.insn_addr = base;
  cpu.x[0] = 1;
  cache->SetStop(stop);
  ExecuteGuest(thread->state());

  // Timed run.
  cpu.insn_addr = base;
  cpu.x[0] = k.iters;
  // memcpy kernel needs valid src/dst pointers.
  static uint64_t srcbuf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  static uint64_t dstbuf[8] = {0};
  cpu.x[1] = ToGuestAddr(srcbuf);
  cpu.x[2] = ToGuestAddr(dstbuf);
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  ExecuteGuest(thread->state());
  clock_gettime(CLOCK_MONOTONIC, &t1);
  cache->TestingClearStop(stop);
  GuestMapShadow::GetInstance()->ClearExecutable(base, k.n_words * 4);

  return (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
}

void RunKernel(Kernel k) {
  InitBerberis();
  constexpr int kRuns = 5;
  std::vector<double> ns;
  for (int i = 0; i < kRuns; ++i) ns.push_back(TimeOnce(k));
  std::sort(ns.begin(), ns.end());
  double median = ns[kRuns / 2];
  uint64_t insns = k.iters * k.insns_iter;
  double mips = insns / (median / 1e9) / 1e6;
  std::printf("BENCH %-8s: %10.0f ns / %9lu insns = %8.1f Mips (median of %d)\n",
              k.name, median, static_cast<unsigned long>(insns), mips, kRuns);
  std::fflush(stdout);
  // Always passes — this is a measurement, not an assertion.
  SUCCEED();
}

TEST(DigitalisBench, Int) {
  RunKernel({"int", kInt, std::size(kInt), std::size(kInt) * 4u, 3'000'000, 18});
}
TEST(DigitalisBench, Branch) {
  RunKernel({"branch", kBranch, std::size(kBranch), std::size(kBranch) * 4u, 3'000'000, 12});
}
TEST(DigitalisBench, Call) {
  RunKernel({"call", kCall, std::size(kCall), 6u * 4u, 3'000'000, 14});
}
TEST(DigitalisBench, Neon) {
  RunKernel({"neon", kNeon, std::size(kNeon), std::size(kNeon) * 4u, 3'000'000, 10});
}
TEST(DigitalisBench, Fp) {
  RunKernel({"fp", kFp, std::size(kFp), std::size(kFp) * 4u, 3'000'000, 10});
}
TEST(DigitalisBench, Mem) {
  RunKernel({"mem", kMem, std::size(kMem), std::size(kMem) * 4u, 3'000'000, 10});
}

}  // namespace
}  // namespace berberis
