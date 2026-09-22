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

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>

#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_guest_exec_region.h"
#include "berberis/test_utils/testing_run_generated_code.h"
#include "berberis/test_utils/translation_test.h"

namespace berberis {

// runtime/arm64/translator_x86_64.cc defines these in namespace berberis with
// external linkage ("Exported for testing only") but has no companion header
// (unlike runtime/riscv64/translator_x86_64.h). Forward-declare them here with
// the exact signatures so the linker resolves them against libberberis_runtime_arm64.
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> TryLiteTranslateAndInstallRegion(
    GuestAddr pc,
    LiteTranslateParams params = LiteTranslateParams());
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> HeavyOptimizeAndInstallRegion(
    GuestAddr pc);

namespace {

class Arm64RuntimeTranslator : public TranslationTest {};

// A region the lite translator fully handles installs a kLiteTranslated entry.
TEST_F(Arm64RuntimeTranslator, LiteTranslateSupportedRegion) {
  static const uint32_t code[] = {
      0x8b020023,  // add x3, x1, x2
      0x14000001,  // b .+4  (unconditional branch ends the region)
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      TryLiteTranslateAndInstallRegion(ToGuestAddr(code));

  EXPECT_TRUE(success);
  EXPECT_NE(host_code_piece.code, kNullHostCodeAddr);
  EXPECT_GT(host_code_piece.size, 0U);
  EXPECT_EQ(guest_size, 8U);
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kLiteTranslated);
}

// A region whose very first instruction the lite translator cannot handle (BRK
// is delivered by the interpreter, which raises the synchronous SIGTRAP) fails
// to translate at all — nothing to install. Mirrors the riscv64 `ecall` case.
TEST_F(Arm64RuntimeTranslator, LiteTranslateUnsupportedRegion) {
  static const uint32_t code[] = {
      0xd4200000,  // brk #0
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      TryLiteTranslateAndInstallRegion(ToGuestAddr(code));

  EXPECT_FALSE(success);
}

// A supported prefix followed by an unsupported instruction installs the prefix
// only: the region is truncated at the bail point (here, after the single ADD).
TEST_F(Arm64RuntimeTranslator, LiteTranslatePartiallySupportedRegion) {
  static const uint32_t code[] = {
      0x8b020023,  // add x3, x1, x2
      0xd4200000,  // brk #0  (interpreter-only, bails the lite translator)
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      TryLiteTranslateAndInstallRegion(ToGuestAddr(code));

  EXPECT_TRUE(success);
  EXPECT_NE(host_code_piece.code, kNullHostCodeAddr);
  EXPECT_GT(host_code_piece.size, 0U);
  EXPECT_EQ(guest_size, 4U);
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kLiteTranslated);
}

// The heavy optimizer bails on BRK exactly like the lite tier; with no prefix
// translated and no in-region back-edge, the install wrapper reports failure so
// the runtime re-lite-translates (and ultimately interprets) the region.
TEST_F(Arm64RuntimeTranslator, HeavyOptimizeUnsupportedRegion) {
  static const uint32_t code[] = {
      0xd4200000,  // brk #0
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      HeavyOptimizeAndInstallRegion(ToGuestAddr(code));

  EXPECT_FALSE(success);
  EXPECT_EQ(guest_size, 0U);
}

// A region the heavy optimizer fully translates but that is smaller than the
// gear-up threshold (GetGearUpMinInsns, default 20) is declined: gearing up a
// tiny region cannot recoup the optimizing tier's codegen overhead, so the
// wrapper returns failure and the runtime keeps the lite region. This exercises
// runtime/arm64/translator_x86_64.cc's gear-up policy, which riscv64 lacks.
TEST_F(Arm64RuntimeTranslator, HeavyOptimizeDeclinesSmallSupportedRegion) {
  static const uint32_t code[] = {
      0xd2800220,  // movz x0, #0x11
      0xd2800441,  // movz x1, #0x22
      0xd65f03c0,  // ret  (indirect branch: definitively ends the region)
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      HeavyOptimizeAndInstallRegion(ToGuestAddr(code));

  // Fully heavy-translatable, but 3 < 20 instructions -> gear-up declined.
  EXPECT_FALSE(success);
}

// A region above the gear-up threshold that the heavy optimizer fully translates
// installs a kHeavyOptimized entry.
TEST_F(Arm64RuntimeTranslator, HeavyOptimizeLargeSupportedRegion) {
  static const uint32_t code[] = {
      0x91000400, 0x91000400, 0x91000400, 0x91000400, 0x91000400,  // add x0, x0, #1
      0x91000400, 0x91000400, 0x91000400, 0x91000400, 0x91000400,  // (x24)
      0x91000400, 0x91000400, 0x91000400, 0x91000400, 0x91000400,
      0x91000400, 0x91000400, 0x91000400, 0x91000400, 0x91000400,
      0x91000400, 0x91000400, 0x91000400, 0x91000400,
      0xd65f03c0,  // ret  (indirect branch: definitively ends the region)
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      HeavyOptimizeAndInstallRegion(ToGuestAddr(code));

  EXPECT_TRUE(success);
  EXPECT_NE(host_code_piece.code, kNullHostCodeAddr);
  EXPECT_GT(host_code_piece.size, 0U);
  EXPECT_EQ(guest_size, sizeof(code));
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kHeavyOptimized);
}

// Runs the installed region at `code` until it dispatches to `stop_pc`. The
// caller supplies a ThreadState with a guest thread attached: RunGuestSyscall
// disables pending signals on it around the host syscall.
void RunInstalledRegionWithSyscall(ThreadState* state,
                                   HostCodePiece piece,
                                   GuestAddr code,
                                   GuestAddr stop_pc) {
  // The translation cache is process-global; clear this window so SetStop
  // installs cleanly even if an earlier test left an entry at stop_pc.
  TranslationCache::GetInstance()->InvalidateGuestRange(code, stop_pc + 4);
  state->cpu.insn_addr = code;
  TestingRunGeneratedCode(state, AsHostCode(piece.code), stop_pc);
}

// A hot region that ends in SVC gears up. The heavy optimizer lowers SVC as a
// syscall region exit (store insn_addr, call RunGuestSyscall, direct-dispatch to
// pc+4 -- the sequence the lite tier ends its region with) rather than bailing,
// which sent every such region back to the lite tier. Running the installed code
// checks what the syscall saw: the length and the syscall number are computed
// inside the region, so write(2) only writes the whole message if X2 and X8
// reached ThreadState before the call.
TEST_F(Arm64RuntimeTranslator, HeavyOptimizeRegionEndingInSvc) {
  // clang-format off
  static const uint32_t code[] = {
      0x91000442, 0x91000442, 0x91000442, 0x91000442, 0x91000442,  // add x2, x2, #1
      0x91000442, 0x91000442, 0x91000442, 0x91000442, 0x91000442,  // (x10)
      0x91000463, 0x91000463, 0x91000463, 0x91000463, 0x91000463,  // add x3, x3, #1
      0x91000463, 0x91000463, 0x91000463, 0x91000463, 0x91000463,  // (x10)
      0xd2800808,  // movz x8, #64  (__NR_write)
      0xd4000001,  // svc #0
  };
  // clang-format on
  const char message[] = "digitalis";  // 10 bytes with the NUL, one per add
  static_assert(sizeof(message) == 10);
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      HeavyOptimizeAndInstallRegion(ToGuestAddr(code));

  ASSERT_TRUE(success);
  EXPECT_EQ(guest_size, sizeof(code));
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kHeavyOptimized);

  ThreadState state{};
  std::unique_ptr<GuestThread, decltype(&GuestThread::Destroy)> guest_thread(
      GuestThread::CreateForTest(&state), GuestThread::Destroy);
  ASSERT_NE(guest_thread, nullptr);
  state.thread = guest_thread.get();
  // Non-blocking, so a region that never reaches the syscall fails the read
  // below instead of hanging the test.
  int pipefd[2];
  ASSERT_EQ(0, pipe2(pipefd, O_NONBLOCK));
  state.cpu.x[0] = static_cast<uint64_t>(pipefd[1]);
  state.cpu.x[1] = reinterpret_cast<uint64_t>(message);
  // A stale X8 in memory would make this getppid instead of write.
  state.cpu.x[8] = 173;

  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  RunInstalledRegionWithSyscall(&state, host_code_piece, ToGuestAddr(code), stop_pc);

  EXPECT_EQ(state.cpu.insn_addr, stop_pc);     // resumed right after the SVC
  EXPECT_EQ(state.cpu.x[0], sizeof(message));  // write(2) result
  EXPECT_EQ(state.cpu.x[2], sizeof(message));
  EXPECT_EQ(state.cpu.x[3], 10u);
  EXPECT_EQ(state.cpu.x[8], 64u);
  char buf[sizeof(message)] = {};
  EXPECT_EQ(read(pipefd[0], buf, sizeof(buf)), static_cast<ssize_t>(sizeof(message)));
  EXPECT_EQ(memcmp(buf, message, sizeof(message)), 0);
  close(pipefd[0]);
  close(pipefd[1]);
}

// The same exit taken out of an in-region loop. RemoveLoopGuestContextAccesses
// keeps the loop's guest registers in host registers and stores them back on the
// loop-exit edge; that store has to land before the syscall exit, or write(2)
// sees the pre-loop X2 of 0 and writes nothing.
TEST_F(Arm64RuntimeTranslator, HeavyOptimizeLoopExitingIntoSvc) {
  // clang-format off
  static const uint32_t code[] = {
      0xd2800149,  // movz x9, #10
      // loop:
      0x91000442,  // add x2, x2, #1
      0x91000463, 0x91000463, 0x91000463, 0x91000463, 0x91000463,  // add x3, x3, #1
      0x91000463, 0x91000463, 0x91000463, 0x91000463, 0x91000463,  // (x15)
      0x91000463, 0x91000463, 0x91000463, 0x91000463, 0x91000463,
      0xf1000529,  // subs x9, x9, #1
      0x54fffde1,  // b.ne loop  (-68: back to the add x2)
      0xd2800808,  // movz x8, #64  (__NR_write)
      0xd4000001,  // svc #0
  };
  // clang-format on
  const char message[] = "digitalis";
  static_assert(sizeof(message) == 10);
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      HeavyOptimizeAndInstallRegion(ToGuestAddr(code));

  ASSERT_TRUE(success);
  EXPECT_EQ(guest_size, sizeof(code));
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kHeavyOptimized);

  ThreadState state{};
  std::unique_ptr<GuestThread, decltype(&GuestThread::Destroy)> guest_thread(
      GuestThread::CreateForTest(&state), GuestThread::Destroy);
  ASSERT_NE(guest_thread, nullptr);
  state.thread = guest_thread.get();
  int pipefd[2];
  ASSERT_EQ(0, pipe2(pipefd, O_NONBLOCK));
  state.cpu.x[0] = static_cast<uint64_t>(pipefd[1]);
  state.cpu.x[1] = reinterpret_cast<uint64_t>(message);

  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  RunInstalledRegionWithSyscall(&state, host_code_piece, ToGuestAddr(code), stop_pc);

  EXPECT_EQ(state.cpu.insn_addr, stop_pc);
  EXPECT_EQ(state.cpu.x[0], sizeof(message));
  EXPECT_EQ(state.cpu.x[2], 10u);
  EXPECT_EQ(state.cpu.x[3], 150u);
  EXPECT_EQ(state.cpu.x[9], 0u);
  char buf[sizeof(message)] = {};
  EXPECT_EQ(read(pipefd[0], buf, sizeof(buf)), static_cast<ssize_t>(sizeof(message)));
  EXPECT_EQ(memcmp(buf, message, sizeof(message)), 0);
  close(pipefd[0]);
  close(pipefd[1]);
}

}  // namespace

}  // namespace berberis
