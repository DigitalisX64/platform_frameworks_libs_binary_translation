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

#include <cstdint>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/heavy_optimizer/arm64/heavy_optimize_region.h"
#include "berberis/test_utils/scoped_exec_region.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

// ARM64 instruction encoding helpers (same forms as the lite-translator tests).
// MOVZ Xd, #imm16 (shift 0).
constexpr uint32_t MovzX(uint8_t rd, uint16_t imm16) {
  return 0xD2800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVZ Xd, #imm16, LSL #(hw*16).
constexpr uint32_t MovzHwX(uint8_t rd, uint16_t imm16, uint8_t hw) {
  return 0xD2800000 | (static_cast<uint32_t>(hw) << 21) |
         (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVN Xd, #imm16 (shift 0).
constexpr uint32_t MovnX(uint8_t rd, uint16_t imm16) {
  return 0x92800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVK Xd, #imm16, LSL #(hw*16).
constexpr uint32_t MovkHwX(uint8_t rd, uint16_t imm16, uint8_t hw) {
  return 0xF2800000 | (static_cast<uint32_t>(hw) << 21) |
         (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVZ Wd, #imm16 (32-bit, shift 0).
constexpr uint32_t MovzW(uint8_t rd, uint16_t imm16) {
  return 0x52800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// Heavy-optimize and execute one instruction, mirroring the riscv64 exec-test
// harness. Returns false if the optimizing frontend bailed (didn't translate the
// instruction) so a test can assert it actually went through the JIT.
bool RunOneInstruction(ThreadState* state, GuestAddr stop_pc) {
  MachineCode machine_code;
  auto [new_addr, success, number_of_instructions] =
      HeavyOptimizeRegion(state->cpu.insn_addr,
                          &machine_code,
                          HeavyOptimizeParams{
                              .max_number_of_instructions = 1,
                          });
  if (!success || number_of_instructions != 1) {
    return false;
  }

  ScopedExecRegion exec(&machine_code);
  TestingRunGeneratedCode(state, exec.get(), stop_pc);
  return true;
}

class Arm64HeavyOptimizerFrontendTest : public ::testing::Test {
 protected:
  ThreadState state_{};
};

TEST_F(Arm64HeavyOptimizerFrontendTest, MovzX0Basic) {
  static const uint32_t code[] = {MovzX(0, 0x1234)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.insn_addr, stop_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1234});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovzWithShift) {
  // MOVZ X1, #0xABCD, LSL #16 -> 0xABCD0000.
  static const uint32_t code[] = {MovzHwX(1, 0xABCD, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0xABCD0000});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovnBasic) {
  // MOVN X2, #0 -> ~0 = 0xFFFFFFFFFFFFFFFF.
  static const uint32_t code[] = {MovnX(2, 0)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0xFFFFFFFFFFFFFFFF});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovzW32BitZeroExtends) {
  // MOVZ W3, #0x5678 -> upper 32 bits cleared.
  static const uint32_t code[] = {MovzW(3, 0x5678)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[3], uint64_t{0x5678});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovkKeepsOtherBits) {
  // X4 = 0x1111, then MOVK X4, #0x2222, LSL #16 -> 0x22221111.
  static const uint32_t setup[] = {MovzX(4, 0x1111)};
  state_.cpu.insn_addr = ToGuestAddr(setup);
  GuestAddr setup_stop = ToGuestAddr(setup) + sizeof(setup);
  ASSERT_TRUE(RunOneInstruction(&state_, setup_stop));
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0x1111});

  static const uint32_t movk[] = {MovkHwX(4, 0x2222, 1)};
  state_.cpu.insn_addr = ToGuestAddr(movk);
  GuestAddr movk_stop = ToGuestAddr(movk) + sizeof(movk);
  ASSERT_TRUE(RunOneInstruction(&state_, movk_stop));
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0x22221111});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovkHighWindow) {
  // X5 = 0, then MOVK X5, #0xBEEF, LSL #48 -> 0xBEEF000000000000.
  static const uint32_t setup[] = {MovzX(5, 0)};
  state_.cpu.insn_addr = ToGuestAddr(setup);
  GuestAddr setup_stop = ToGuestAddr(setup) + sizeof(setup);
  ASSERT_TRUE(RunOneInstruction(&state_, setup_stop));

  static const uint32_t movk[] = {MovkHwX(5, 0xBEEF, 3)};
  state_.cpu.insn_addr = ToGuestAddr(movk);
  GuestAddr movk_stop = ToGuestAddr(movk) + sizeof(movk);
  ASSERT_TRUE(RunOneInstruction(&state_, movk_stop));
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0xBEEF000000000000});
}

// Repro: multi-instruction regions (what the runtime actually produces) must
// pass GenCode's CheckMachineIR. The 1-insn tests above never exercised this.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiMoveWideRegion) {
  static const uint32_t code[] = {MovzX(0, 0x11), MovzX(1, 0x22)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_TRUE(ok);
  EXPECT_EQ(n, 2u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MoveWideThenBailRegion) {
  static const uint32_t code[] = {MovzX(0, 0x11), MovzX(1, 0x22), 0x91000400 /*ADD bail*/};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // 2 MoveWide translate, the ADD bails -> partial region.
  EXPECT_EQ(n, 2u);
}

// A single guest instruction whose decode fires several listener callbacks that
// all bail (a pre-index load calls AddImm then Load) must still produce valid IR
// (one region exit, no trailing insns) — i.e. GenCode must not abort. This is the
// structure that broke on-device before Undefined() was made idempotent.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiCallbackBailIsValidIR) {
  // LDR X1, [X0, #8]!  (pre-index): decodes to AddImm + Load, both bail.
  static const uint32_t code[] = {0xF8408401};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  // No abort here is the assertion (GenCode runs CheckMachineIR internally).
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_FALSE(ok);
  EXPECT_EQ(n, 0u);
}

// MoveWide followed by a multi-callback bail (the common on-device prefix shape).
TEST_F(Arm64HeavyOptimizerFrontendTest, MoveWideThenMultiCallbackBail) {
  static const uint32_t code[] = {MovzX(0, 0x11), 0xF8408401 /*LDR pre-index, bails*/};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 1u);  // the MOVZ translated; the load bailed without corrupting IR
}

// A non-MoveWide instruction must bail out of the optimizing frontend (the
// runtime then falls back to the lite translator / interpreter).
TEST_F(Arm64HeavyOptimizerFrontendTest, NonMoveWideBails) {
  // ADD X0, X0, #1 (immediate) is not yet translated by the optimizing frontend.
  static const uint32_t code[] = {0x91000400};  // ADD X0, X0, #1
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
}

}  // namespace

}  // namespace berberis
