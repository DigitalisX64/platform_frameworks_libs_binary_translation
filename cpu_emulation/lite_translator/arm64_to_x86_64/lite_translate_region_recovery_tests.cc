// Fault recovery of lite-translated memory accesses: a faulting load or store
// must exit to its own guest instruction with every guest register modified
// before it flushed to ThreadState, and the recovery stubs must sit out of line
// so the fault-free path never jumps over them.

#include "gtest/gtest.h"

#include <csignal>
#include <cstdint>
#include <ucontext.h>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

constexpr uint32_t kAddX1X1Imm1 = 0x91000421;  // add x1, x1, #1
constexpr uint32_t kAddX5X5Imm1 = 0x910004a5;  // add x5, x5, #1
constexpr uint32_t kLdrX2X0 = 0xf9400002;      // ldr x2, [x0]
constexpr uint32_t kLdrX4X3 = 0xf9400064;      // ldr x4, [x3]
constexpr uint32_t kStrX2X3 = 0xf9000062;      // str x2, [x3]

// Never mapped: the first page.
constexpr uint64_t kFaultingAddr = 0x10;

uintptr_t g_fault_rip;
uintptr_t g_recovery_rip;

// Does what the runtime's guest signal handler does for a fault inside
// generated code: continue at the access's recovery code.
void RecoverySegvHandler(int, siginfo_t*, void* raw_context) {
  auto* context = static_cast<ucontext_t*>(raw_context);
  g_fault_rip = context->uc_mcontext.gregs[REG_RIP];
  g_recovery_rip = GetDefaultCodePoolInstance()->FindRecoveryCode(g_fault_rip);
  if (g_recovery_rip == 0) {
    signal(SIGSEGV, SIG_DFL);
    return;
  }
  context->uc_mcontext.gregs[REG_RIP] = g_recovery_rip;
}

class Arm64LiteRecoveryTest : public ::testing::Test {
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
  HostCode Translate(const uint32_t (&code)[N]) {
    GuestAddr start = ToGuestAddr(code);
    MachineCode machine_code;
    auto [success, stop_pc] = TryLiteTranslateRegion(
        start, &machine_code, LiteTranslateParams{.end_pc = start + N * 4, .allow_dispatch = false});
    EXPECT_TRUE(success);
    EXPECT_EQ(stop_pc, start + N * 4);
    return AsHostCode(GetDefaultCodePoolInstance()->Add(&machine_code));
  }

  void RunUntil(HostCode host_code, GuestAddr stop_pc) {
    g_fault_rip = 0;
    g_recovery_rip = 0;
    TestingRunGeneratedCode(&state_, host_code, stop_pc);
    EXPECT_EQ(state_.cpu.insn_addr, stop_pc);
  }

  ThreadState state_{};
  struct sigaction old_sa_ = {};
};

TEST_F(Arm64LiteRecoveryTest, FaultingLoadsExitWithModifiedRegistersFlushed) {
  static const uint32_t code[] = {kAddX1X1Imm1, kLdrX2X0, kAddX5X5Imm1, kLdrX4X3};
  HostCode host_code = Translate(code);
  alignas(8) static uint64_t cell = 0x1234'5678'9abc'def0ULL;

  // The first load faults: only x1 was modified before it.
  state_.cpu.x[0] = kFaultingAddr;
  state_.cpu.x[1] = 41;
  state_.cpu.x[2] = 2;
  state_.cpu.x[3] = ToGuestAddr(&cell);
  state_.cpu.x[5] = 5;
  RunUntil(host_code, ToGuestAddr(&code[1]));
  uintptr_t first_fault_rip = g_fault_rip;
  uintptr_t first_recovery_rip = g_recovery_rip;
  ASSERT_NE(first_recovery_rip, 0u);
  EXPECT_EQ(state_.cpu.x[1], 42u);
  EXPECT_EQ(state_.cpu.x[2], 2u);
  EXPECT_EQ(state_.cpu.x[5], 5u);

  // The second load faults: x1, x2 and x5 were modified before it.
  state_.cpu.x[0] = ToGuestAddr(&cell);
  state_.cpu.x[1] = 41;
  state_.cpu.x[3] = kFaultingAddr;
  state_.cpu.x[4] = 4;
  RunUntil(host_code, ToGuestAddr(&code[3]));
  uintptr_t second_fault_rip = g_fault_rip;
  ASSERT_NE(g_recovery_rip, 0u);
  EXPECT_EQ(state_.cpu.x[1], 42u);
  EXPECT_EQ(state_.cpu.x[2], cell);
  EXPECT_EQ(state_.cpu.x[4], 4u);
  EXPECT_EQ(state_.cpu.x[5], 6u);

  // Out of line: the first load's stub comes after the second load, so the
  // fault-free path runs from one access to the next without a jump.
  EXPECT_LT(first_fault_rip, second_fault_rip);
  EXPECT_GT(first_recovery_rip, second_fault_rip);
}

TEST_F(Arm64LiteRecoveryTest, FaultingStoreExitsWithModifiedRegistersFlushed) {
  static const uint32_t code[] = {kAddX1X1Imm1, kStrX2X3};
  HostCode host_code = Translate(code);

  state_.cpu.x[1] = 41;
  state_.cpu.x[2] = 2;
  state_.cpu.x[3] = kFaultingAddr;
  RunUntil(host_code, ToGuestAddr(&code[1]));
  ASSERT_NE(g_recovery_rip, 0u);
  EXPECT_GT(g_recovery_rip, g_fault_rip);
  EXPECT_EQ(state_.cpu.x[1], 42u);
  EXPECT_EQ(state_.cpu.x[2], 2u);
}

}  // namespace

}  // namespace berberis
