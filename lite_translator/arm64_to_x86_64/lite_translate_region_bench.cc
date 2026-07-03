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

// syscall-heavy: clock_gettime(CLOCK_MONOTONIC) in a tight loop. Preamble
// (idx0) copies the harness's x0 iteration count into x9; the loop (idx1..5)
// issues svc #0 with x8=113, x0=1, x1=timespec buffer. Measures the syscall
// emulation path; the vDSO fast path should raise this number.
constexpr uint32_t kSys[] = {
    0xaa0003e9,  // mov  x9, x0   (iter count)
    0xd2800e28,  // mov  x8, #113 (__NR_clock_gettime)
    0xd2800020,  // mov  x0, #1   (CLOCK_MONOTONIC)
    0xd4000001,  // svc  #0
    0xf1000529,  // subs x9, x9, #1
    0x54ffff81,  // b.ne idx1
};  // stop = 6*4; guest insns/iter (loop body) = 5.

struct Kernel {
  const char* name;
  const uint32_t* code;
  size_t n_words;
  uint32_t stop_off;     // byte offset of the fall-through (stop) instruction
  uint64_t iters;        // loop trip count
  uint64_t insns_iter;   // guest instructions retired per iteration
};

double TimeOnce(const Kernel& k) {
  // Keep the kernel in a stable static buffer so the 5 timed runs share a warm
  // translation cache. Two hazards come with reusing one buffer across kernels,
  // and both are handled below.
  //
  // (1) UDF guard tail. The lite translator forms a region up to
  // `pc + GetExecutableRegionSize(pc)`, and GuestMapShadow is PAGE-granular, so
  // formation does NOT stop at k.n_words — it keeps decoding whatever bytes
  // follow the kernel in the page. Because the buffer is reused, those trailing
  // bytes are the PREVIOUS (often larger) kernel's leftover loop code, and a
  // conditional loop-back branch (b.ne) does not end a region — its fall-through
  // is translated inline. Formation would then splice two kernels into one hybrid
  // region that never reaches the stop PC and spins at 100% CPU. That is exactly
  // what wedged `DigitalisBench.Branch`/`.Fp`/`.Mem` when they ran after `.Int`.
  // Zero-filling the words at and past k.stop_off makes them UDF (0x00000000):
  // formation fails to decode there, so the region is clamped to
  // [base, base+stop_off) and base+stop_off stays a dispatch boundary where
  // SetStop halts execution cleanly. (The first kernel gets this for free from
  // zeroed heap past its fresh buffer; the guard makes it deterministic for every
  // kernel regardless of run order.)
  static constexpr size_t kUdfGuardWords = 4;
  static std::vector<uint32_t> buf;
  buf.assign(k.n_words + kUdfGuardWords, 0u);  // zero => UDF guard tail
  std::copy(k.code, k.code + k.n_words, buf.begin());
  size_t exec_size = (k.n_words + kUdfGuardWords) * 4;
  GuestAddr base = ToGuestAddr(buf.data());
  GuestMapShadow::GetInstance()->SetExecutable(base, exec_size);
  GuestThread* thread = GetCurrentGuestThread();
  auto& cpu = thread->state()->cpu;
  GuestAddr stop = base + k.stop_off;
  auto* cache = TranslationCache::GetInstance();

  // (2) Stale-cache invalidation. The translation cache is keyed by guest address
  // and is NOT invalidated when we overwrite the reused buffer with new bytes, so
  // without this a later kernel would execute an earlier kernel's stale cached
  // region (measuring the wrong code) — a self-modifying-code contract violation.
  // Invalidate the range we are about to translate, exactly as real SMC / IC IVAU
  // handling does.
  cache->InvalidateGuestRange(base, base + exec_size);

  // Valid pointer args for the mem (x1/x2) and syscall (x1=timespec) kernels;
  // set before the warmup so even the warmup pass has a valid buffer.
  static uint64_t srcbuf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  static uint64_t dstbuf[8] = {0};
  cpu.x[1] = ToGuestAddr(srcbuf);
  cpu.x[2] = ToGuestAddr(dstbuf);

  // Warm the translation cache once (untimed).
  cpu.insn_addr = base;
  cpu.x[0] = 1;
  cache->SetStop(stop);
  ExecuteGuest(thread->state());

  // Timed run.
  cpu.insn_addr = base;
  cpu.x[0] = k.iters;
  cpu.x[1] = ToGuestAddr(srcbuf);
  cpu.x[2] = ToGuestAddr(dstbuf);
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  ExecuteGuest(thread->state());
  clock_gettime(CLOCK_MONOTONIC, &t1);
  cache->TestingClearStop(stop);
  GuestMapShadow::GetInstance()->ClearExecutable(base, exec_size);

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

// Correctness guard for the fused compare-branch (B3.2): a taken conditional
// branch whose target reads NZCV again before re-setting it. If the fusion
// skips materialising cpu.flags on the taken path, the second B.EQ reads stale
// flags and lands on the x0=13 path instead of x0=42.
TEST(Arm64FusedFlagLiveness, TakenBranchTargetReadsFlags) {
  static const uint32_t code[] = {
      0xd2800000,  // mov  x0, #0
      0xf100001f,  // cmp  x0, #0        (Z=1; fused with the next b.eq)
      0x54000040,  // b.eq +8  -> idx4   (taken: Z==1)
      0xd28000e0,  // mov  x0, #7        (skipped)
      0x54000040,  // b.eq +8  -> idx6   (reads NZCV again; Z must still be 1)
      0xd28001a0,  // mov  x0, #13       (reached only on stale flags = bug)
      0xd2800540,  // mov  x0, #42       (correct path)
  };
  InitBerberis();
  GuestAddr base = ToGuestAddr(code);
  GuestMapShadow::GetInstance()->SetExecutable(base, sizeof(code));
  GuestThread* thread = GetCurrentGuestThread();
  auto& cpu = thread->state()->cpu;
  cpu.insn_addr = base;
  cpu.x[0] = 123;
  GuestAddr stop = base + sizeof(code);
  auto* cache = TranslationCache::GetInstance();
  cache->SetStop(stop);
  ExecuteGuest(thread->state());
  cache->TestingClearStop(stop);
  GuestMapShadow::GetInstance()->ClearExecutable(base, sizeof(code));
  EXPECT_EQ(cpu.x[0], 42u);  // 13 (or 7) would mean a fused-flag staleness bug
}

// Sharper guard: a TAKEN b.ne (Z=0) whose target reads NZCV, with a prior
// compare having left Z=1. If the taken path doesn't materialise cpu.flags, the
// target's b.eq reads the stale Z=1 and lands on x0=99 instead of x0=42.
TEST(Arm64FusedFlagLiveness, TakenNeTargetReadsStaleFlags) {
  static const uint32_t code[] = {
      0xd28000a0,  // mov  x0, #5
      0xf100141f,  // cmp  x0, #5      (Z=1; fused with next b.ne)
      0x54000021,  // b.ne +4 -> idx3  (Z=1 -> not taken; materialises Z=1)
      0xf100181f,  // cmp  x0, #6      (Z=0; fused with next b.ne)
      0x54000041,  // b.ne +8 -> idx6  (Z=0 -> TAKEN)
      0xd28001a0,  // mov  x0, #13     (skipped)
      0x54000060,  // b.eq +12 -> idx9 (reads NZCV: correct Z=0 -> not taken)
      0xd2800540,  // mov  x0, #42     (correct)
      0x14000002,  // b    +8 -> idx10
      0xd2800c60,  // mov  x0, #99     (bug: stale Z=1 -> b.eq taken)
      0xd503201f,  // nop
  };
  InitBerberis();
  GuestAddr base = ToGuestAddr(code);
  GuestMapShadow::GetInstance()->SetExecutable(base, sizeof(code));
  GuestThread* thread = GetCurrentGuestThread();
  auto& cpu = thread->state()->cpu;
  cpu.insn_addr = base;
  GuestAddr stop = base + sizeof(code);
  auto* cache = TranslationCache::GetInstance();
  cache->SetStop(stop);
  ExecuteGuest(thread->state());
  cache->TestingClearStop(stop);
  GuestMapShadow::GetInstance()->ClearExecutable(base, sizeof(code));
  EXPECT_EQ(cpu.x[0], 42u);  // 99 would mean stale flags on the taken path
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
TEST(DigitalisBench, Syscall) {
  // Fewer iters: each carries a clock_gettime. insns/iter counts the 5 loop-body
  // guest instructions; the figure is syscall-bound, so a vDSO fast path shows.
  RunKernel({"syscall", kSys, std::size(kSys), 6u * 4u, 1'000'000, 5});
}

}  // namespace
}  // namespace berberis
