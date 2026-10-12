// In-region self-loops (LiteTranslateParams::allow_in_region_loops): a branch
// back to the region's start loops inside the region, every iteration checks
// pending signals, and every exit and fault stub sees the registers and FP
// state any iteration may have changed.

#include "gtest/gtest.h"

#include <signal.h>
#include <ucontext.h>
#include <xmmintrin.h>

#include <cstdint>
#include <cstring>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

constexpr uint32_t kAddX1X1X2 = 0x8b020021;    // add x1, x1, x2
constexpr uint32_t kAddX1X1Imm1 = 0x91000421;  // add x1, x1, #1
constexpr uint32_t kSubsX0X0Imm1 = 0xf1000400;  // subs x0, x0, #1
constexpr uint32_t kSubX4X4Imm1 = 0xd1000484;  // sub x4, x4, #1
constexpr uint32_t kCbzX4Fwd16 = 0xb4000084;   // cbz x4, .+16
constexpr uint32_t kLdrX2X0 = 0xf9400002;      // ldr x2, [x0]
constexpr uint32_t kMovX0X3 = 0xaa0303e0;      // mov x0, x3
constexpr uint32_t kFdivS0S0S1 = 0x1e211800;   // fdiv s0, s0, s1
constexpr uint32_t kBack2Cond = 0x54ffffc1;    // b.ne .-8
constexpr uint32_t kBack3 = 0x17fffffd;        // b .-12

// add xN, xN, #1
constexpr uint32_t AddImm1(uint32_t n) {
  return 0x91000400 | (n << 5) | n;
}

// b.ne .-(insns * 4)
constexpr uint32_t BneBack(uint32_t insns) {
  return 0x54000001 | (((0x80000 - insns) & 0x7ffff) << 5);
}
static_assert(BneBack(2) == kBack2Cond);

constexpr uint64_t kFaultingAddr = 0x10;  // Never mapped: the first page.
constexpr uint64_t kFpsrDzc = 1u << 1;
constexpr uint32_t kMxcsrFlags = 0x3f;

uintptr_t g_recovery_rip;

// Does what the runtime's guest signal handler does for a fault inside
// generated code: continue at the access's recovery code.
void RecoverySegvHandler(int, siginfo_t*, void* raw_context) {
  auto* context = static_cast<ucontext_t*>(raw_context);
  g_recovery_rip = GetDefaultCodePoolInstance()->FindRecoveryCode(context->uc_mcontext.gregs[REG_RIP]);
  if (g_recovery_rip == 0) {
    signal(SIGSEGV, SIG_DFL);
    return;
  }
  context->uc_mcontext.gregs[REG_RIP] = g_recovery_rip;
}

class Arm64LiteLoopTest : public ::testing::Test {
 protected:
  void SetUp() override {
    struct sigaction sa = {};
    sa.sa_sigaction = RecoverySegvHandler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    ASSERT_EQ(sigaction(SIGSEGV, &sa, &old_sa_), 0);
  }

  void TearDown() override { sigaction(SIGSEGV, &old_sa_, nullptr); }

  template <size_t N>
  void TranslateInto(const uint32_t (&code)[N], MachineCode* machine_code, bool allow_loops) {
    GuestAddr start = ToGuestAddr(code);
    auto [success, stop_pc] = TryLiteTranslateRegion(start,
                                                     machine_code,
                                                     LiteTranslateParams{
                                                         .end_pc = start + N * 4,
                                                         .allow_dispatch = false,
                                                         .allow_in_region_loops = allow_loops,
                                                     });
    EXPECT_TRUE(success);
    EXPECT_EQ(stop_pc, start + N * 4);
  }

  template <size_t N>
  HostCode Translate(const uint32_t (&code)[N], bool allow_loops = true) {
    MachineCode machine_code;
    TranslateInto(code, &machine_code, allow_loops);
    return AsHostCode(GetDefaultCodePoolInstance()->Add(&machine_code));
  }

  // One entry into the region: with allow_dispatch off, any region exit
  // returns here.
  void RunOnce(HostCode host_code, GuestAddr expected_exit) {
    g_recovery_rip = 0;
    TestingRunGeneratedCode(&state_, host_code, expected_exit);
    EXPECT_EQ(state_.cpu.insn_addr, expected_exit);
  }

  ThreadState state_{};
  struct sigaction old_sa_ = {};
};

TEST_F(Arm64LiteLoopTest, CountedLoopRunsInRegion) {
  static const uint32_t code[] = {kAddX1X1X2, kSubsX0X0Imm1, kBack2Cond};
  HostCode host_code = Translate(code);
  state_.cpu.x[0] = 5;
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 3;
  // All five iterations run in one entry; the region exits past its end.
  RunOnce(host_code, ToGuestAddr(code) + sizeof(code));
  EXPECT_EQ(state_.cpu.x[0], 0u);
  EXPECT_EQ(state_.cpu.x[1], 15u);
  EXPECT_EQ(state_.cpu.x[2], 3u);
}

TEST_F(Arm64LiteLoopTest, WithoutParamBackEdgeExitsRegion) {
  static const uint32_t code[] = {kAddX1X1X2, kSubsX0X0Imm1, kBack2Cond};
  HostCode host_code = Translate(code, /*allow_loops=*/false);
  state_.cpu.x[0] = 5;
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 3;
  RunOnce(host_code, ToGuestAddr(code));
  EXPECT_EQ(state_.cpu.x[0], 4u);
  EXPECT_EQ(state_.cpu.x[1], 3u);
}

TEST_F(Arm64LiteLoopTest, PendingSignalExitsAtBackEdge) {
  static const uint32_t code[] = {kAddX1X1X2, kSubsX0X0Imm1, kBack2Cond};
  HostCode host_code = Translate(code);
  state_.cpu.x[0] = 5;
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 3;
  state_.pending_signals_status.store(kPendingSignalsPresent);
  // One iteration, then the back-edge sees the signal and exits to the start.
  RunOnce(host_code, ToGuestAddr(code));
  EXPECT_EQ(state_.cpu.x[0], 4u);
  EXPECT_EQ(state_.cpu.x[1], 3u);
}

TEST_F(Arm64LiteLoopTest, SideExitStoresRegisterModifiedLaterInBody) {
  // x1 is written after the side exit, so in the first iteration the exit
  // has nothing to store; in later ones it must store x1.
  static const uint32_t code[] = {kCbzX4Fwd16, kAddX1X1Imm1, kSubX4X4Imm1, kBack3};
  HostCode host_code = Translate(code);
  state_.cpu.x[1] = 10;
  state_.cpu.x[4] = 3;
  RunOnce(host_code, ToGuestAddr(code) + sizeof(code));
  EXPECT_EQ(state_.cpu.x[1], 13u);
  EXPECT_EQ(state_.cpu.x[4], 0u);
}

TEST_F(Arm64LiteLoopTest, FaultInLaterIterationStoresModifiedRegisters) {
  // Iteration 1 loads from a valid cell and points x0 at an unmapped page;
  // iteration 2's load faults before it modifies anything.
  static const uint32_t code[] = {kLdrX2X0, kAddX1X1Imm1, kMovX0X3, kBack3};
  HostCode host_code = Translate(code);
  alignas(8) static uint64_t cell = 0x1122'3344'5566'7788ULL;
  state_.cpu.x[0] = ToGuestAddr(&cell);
  state_.cpu.x[1] = 41;
  state_.cpu.x[3] = kFaultingAddr;
  RunOnce(host_code, ToGuestAddr(code));
  ASSERT_NE(g_recovery_rip, 0u);
  EXPECT_EQ(state_.cpu.x[0], kFaultingAddr);
  EXPECT_EQ(state_.cpu.x[1], 42u);
  EXPECT_EQ(state_.cpu.x[2], cell);
}

TEST_F(Arm64LiteLoopTest, FpExceptionOfEarlierIterationReachesFpsrAtSideExit) {
  // Iteration 1 divides by zero; iteration 2 leaves at the side exit, which
  // comes before the division in the body.
  static const uint32_t code[] = {kCbzX4Fwd16, kFdivS0S0S1, kSubX4X4Imm1, kBack3};
  HostCode host_code = Translate(code);
  float one = 1.0f;
  uint32_t one_bits;
  memcpy(&one_bits, &one, sizeof(one_bits));
  state_.cpu.v[0] = one_bits;
  state_.cpu.v[1] = 0;
  state_.cpu.x[4] = 1;
  state_.cpu.emulated_fpsr = 0;
  _mm_setcsr(_mm_getcsr() & ~kMxcsrFlags);
  RunOnce(host_code, ToGuestAddr(code) + sizeof(code));
  _mm_setcsr(_mm_getcsr() & ~kMxcsrFlags);
  EXPECT_EQ(state_.cpu.x[4], 0u);
  EXPECT_NE(state_.cpu.emulated_fpsr & kFpsrDzc, 0u);
}

TEST_F(Arm64LiteLoopTest, LoopUnderRegisterPressureSpillsCorrectly) {
  // Ten guest registers: more than the mapping pool holds, so some go
  // through ThreadState on every access.
  static const uint32_t code[] = {AddImm1(1),
                                  AddImm1(2),
                                  AddImm1(3),
                                  AddImm1(4),
                                  AddImm1(5),
                                  AddImm1(6),
                                  AddImm1(7),
                                  AddImm1(8),
                                  AddImm1(9),
                                  kSubsX0X0Imm1,
                                  BneBack(10)};
  HostCode host_code = Translate(code);
  state_.cpu.x[0] = 4;
  for (unsigned i = 1; i <= 9; i++) {
    state_.cpu.x[i] = i * 100;
  }
  RunOnce(host_code, ToGuestAddr(code) + sizeof(code));
  EXPECT_EQ(state_.cpu.x[0], 0u);
  for (unsigned i = 1; i <= 9; i++) {
    EXPECT_EQ(state_.cpu.x[i], i * 100 + 4) << "x" << i;
  }
}

TEST_F(Arm64LiteLoopTest, SelfProfilingRegionIsUnchanged) {
  static const uint32_t code[] = {kAddX1X1X2, kSubsX0X0Imm1, kBack2Cond};
  GuestAddr start = ToGuestAddr(code);
  uint32_t counter = 0;
  auto translate = [&](MachineCode* machine_code, bool allow_loops) {
    auto [success, stop_pc] = TryLiteTranslateRegion(start,
                                                     machine_code,
                                                     LiteTranslateParams{
                                                         .end_pc = start + sizeof(code),
                                                         .allow_dispatch = false,
                                                         .enable_self_profiling = true,
                                                         .allow_in_region_loops = allow_loops,
                                                         .counter_location = &counter,
                                                     });
    EXPECT_TRUE(success);
    EXPECT_EQ(stop_pc, start + sizeof(code));
  };
  MachineCode plain;
  MachineCode with_param;
  translate(&plain, false);
  translate(&with_param, true);
  ASSERT_EQ(plain.install_size(), with_param.install_size());
  EXPECT_EQ(memcmp(plain.AddrAs<uint8_t>(0), with_param.AddrAs<uint8_t>(0), plain.install_size()),
            0);
}

}  // namespace

}  // namespace berberis
