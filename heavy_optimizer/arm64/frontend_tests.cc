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

// --- Non-flag integer ALU encoders (same forms as the lite-translator tests). ---
// ADD/SUB Xd, Xn, #imm12.
constexpr uint32_t AddImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x91000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
constexpr uint32_t SubImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0xD1000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// SUBS Xd, Xn, #imm12 (flag-setting subtract).
constexpr uint32_t SubsImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0xF1000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// ADDS Xd, Xn, #imm12 (flag-setting add).
constexpr uint32_t AddsImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0xB1000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// CMP Xn, #imm12 == SUBS XZR, Xn, #imm12.
constexpr uint32_t CmpImmX(uint8_t rn, uint16_t imm12) {
  return SubsImmX(31, rn, imm12);
}
// SUBS/ADDS Wd, Wn, #imm12 (32-bit, flag-setting).
constexpr uint32_t SubsImmW(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x71000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
constexpr uint32_t AddsImmW(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x31000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// ADD/SUB Wd, Wn, #imm12 (32-bit).
constexpr uint32_t AddImmW(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x11000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// ADD/SUB/AND/ORR/EOR Xd, Xn, Xm (shifted register, no shift).
constexpr uint32_t AddRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x8B000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t SubRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xCB000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t AndRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x8A000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t OrrRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xAA000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t EorRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xCA000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// BIC Xd, Xn, Xm (AND with inverted Xm).
constexpr uint32_t BicRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x8A200000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// ORN Xd, Xn, Xm (ORR with inverted Xm).
constexpr uint32_t OrnRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xAA200000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SUB Xd, Xn, Xm, LSL #shift.
constexpr uint32_t SubRegLsl(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t shift) {
  return 0xCB000000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(shift) << 10) | (rn << 5) | rd;
}
// AND Xd, Xn, Xm, LSR #shift.
constexpr uint32_t AndRegLsr(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t shift) {
  return 0x8A400000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(shift) << 10) | (rn << 5) | rd;
}
// ANDS Xd, Xn, Xm (flag-setting AND, shifted register).
constexpr uint32_t AndsRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xEA000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SUBS/ADDS Xd, Xn, Xm (flag-setting, shifted register, no shift).
constexpr uint32_t SubsRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xEB000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t AddsRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xAB000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SUBS/ADDS Wd, Wn, Wm (32-bit, flag-setting, shifted register, no shift).
constexpr uint32_t SubsRegW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x6B000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t AddsRegW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x2B000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SUBS Xd, Xn, Xm, LSL #shift.
constexpr uint32_t SubsRegLsl(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t shift) {
  return 0xEB000000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(shift) << 10) | (rn << 5) | rd;
}
// ADDS Xd, Xn, Xm, UXTB #shift (extended register, flag-setting).
constexpr uint32_t AddsExtX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t option, uint8_t shift) {
  return 0xAB200000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(option) << 13) | (static_cast<uint32_t>(shift) << 10) |
         (rn << 5) | rd;
}
// ANDS Xd, Xn, #bitmask (logical immediate, N=1 64-bit, flag-setting).
constexpr uint32_t AndsImmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xF2400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// ADD Xd, Xn, Xm, UXTB #shift (extended register).
constexpr uint32_t AddExtX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t option, uint8_t shift) {
  return 0x8B200000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(option) << 13) | (static_cast<uint32_t>(shift) << 10) |
         (rn << 5) | rd;
}
// AND/ORR/EOR Xd, Xn, #bitmask (logical immediate, N=1 64-bit).
constexpr uint32_t AndImmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0x92400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
constexpr uint32_t OrrImmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xB2400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
constexpr uint32_t EorImmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xD2400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// MADD/MSUB Xd, Xn, Xm, Xa.
constexpr uint32_t MaddX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x9B000000 | (static_cast<uint32_t>(rm) << 16) | (ra << 10) | (rn << 5) | rd;
}
constexpr uint32_t MsubX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x9B008000 | (static_cast<uint32_t>(rm) << 16) | (ra << 10) | (rn << 5) | rd;
}
// SMADDL/UMADDL Xd, Wn, Wm, Xa.
constexpr uint32_t SmaddlX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x9B200000 | (static_cast<uint32_t>(rm) << 16) | (ra << 10) | (rn << 5) | rd;
}
constexpr uint32_t UmaddlX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x9BA00000 | (static_cast<uint32_t>(rm) << 16) | (ra << 10) | (rn << 5) | rd;
}
// LSLV/LSRV/ASRV/RORV Xd, Xn, Xm (variable shift).
constexpr uint32_t LslvX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC02000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t LsrvX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC02400 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t AsrvX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC02800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t RorvX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC02C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// UDIV Xd, Xn, Xm (must bail).
constexpr uint32_t UdivX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC00800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SMULH Xd, Xn, Xm (must bail).
constexpr uint32_t SmulhX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9B407C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SBFM/BFM/UBFM Xd, Xn, #immr, #imms (64-bit, N=1).
constexpr uint32_t SbfmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0x93400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
constexpr uint32_t BfmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xB3400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
constexpr uint32_t UbfmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xD3400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// UBFM Wd, Wn, #immr, #imms (32-bit, N=0).
constexpr uint32_t UbfmW(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0x53000000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// EXTR Xd, Xn, Xm, #lsb (64-bit) and Wd (32-bit).
constexpr uint32_t ExtrX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t lsb) {
  return 0x93C00000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(lsb) << 10) | (rn << 5) | rd;
}
constexpr uint32_t ExtrW(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t lsb) {
  return 0x13800000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(lsb) << 10) | (rn << 5) | rd;
}
// CLZ/REV16 Xd, Xn (DP-1Src).
constexpr uint32_t ClzX(uint8_t rd, uint8_t rn) {
  return 0xDAC01000 | (rn << 5) | rd;
}
constexpr uint32_t Rev16X(uint8_t rd, uint8_t rn) {
  return 0xDAC00400 | (rn << 5) | rd;
}
// REV Xd, Xn (DP-1Src) — must bail (no x86 BSWAP MachineIR op).
constexpr uint32_t RevX(uint8_t rd, uint8_t rn) {
  return 0xDAC00C00 | (rn << 5) | rd;
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
  // SMULH (DataProc3Src high-multiply) still bails after the two MoveWides.
  static const uint32_t code[] = {MovzX(0, 0x11), MovzX(1, 0x22), SmulhX(2, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // 2 MoveWide translate, the SMULH bails -> partial region.
  EXPECT_EQ(n, 2u);
}

// A single guest instruction whose decode fires several listener callbacks where
// a later one bails (a post-index load calls Load then AddImm) must still produce
// valid IR (one region exit, no trailing insns) — i.e. GenCode must not abort.
// The Load bails first, so AddImm must emit nothing once success_ is false.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiCallbackBailIsValidIR) {
  // LDR X1, [X0], #8  (post-index): decodes to Load (bails) then AddImm.
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
  static const uint32_t code[] = {MovzX(0, 0x11), 0xF8408401 /*LDR post-index, bails*/};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 1u);  // the MOVZ translated; the load bailed without corrupting IR
}

//
// Non-flag integer ALU value checks (full pipeline via RunOneInstruction).
//

TEST_F(Arm64HeavyOptimizerFrontendTest, AddImm64) {
  static const uint32_t code[] = {AddImmX(0, 1, 0x123)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1123});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SubImm64) {
  static const uint32_t code[] = {SubImmX(0, 1, 0x10)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFF0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AddImm32ZeroExtends) {
  // ADD W0, W1, #1: result is zero-extended into X0.
  static const uint32_t code[] = {AddImmW(0, 1, 1)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;  // W1 = 0xFFFFFFFF
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});  // 0xFFFFFFFF + 1 = 0, upper cleared
}

//
// Flag-setting integer ALU: assert BOTH the result register AND cpu.flags.
// NZCV packing matches the lite translator: N=bit15, Z=bit14, C=bit8, V=bit0.
//

// SUBS producing Z=1, C=1 (equal operands -> no borrow).
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsImmEqualSetsZC) {
  // SUBS X0, X1, #5 with X1 == 5.
  static const uint32_t code[] = {SubsImmX(0, 1, 5)};
  state_.cpu.x[1] = 5;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);  // no borrow
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// SUBS producing N=1, C=0 (borrow: smaller minus larger).
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsImmBorrowSetsNC) {
  // SUBS X0, X1, #10 with X1 == 5 -> -5, borrow.
  static const uint32_t code[] = {SubsImmX(0, 1, 10)};
  state_.cpu.x[1] = 5;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFFBULL});  // -5
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);  // ARM borrow -> C=0
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// SUBS signed overflow: INT64_MIN - 1 overflows -> V=1.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsImmSignedOverflow) {
  // SUBS X0, X1, #1 with X1 == INT64_MIN -> wraps to INT64_MAX, V=1.
  static const uint32_t code[] = {SubsImmX(0, 1, 1)};
  state_.cpu.x[1] = 0x8000000000000000ULL;  // INT64_MIN
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x7FFFFFFFFFFFFFFFULL});  // INT64_MAX
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagOverflow);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);  // no unsigned borrow
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
}

// ADDS producing C=1 (unsigned carry-out).
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsImmCarryOut) {
  // ADDS X0, X1, #1 with X1 == UINT64_MAX -> 0, carry out.
  static const uint32_t code[] = {AddsImmX(0, 1, 1)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// ADDS producing V=1 (signed overflow): INT64_MAX + 1.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsImmSignedOverflow) {
  // ADDS X0, X1, #1 with X1 == INT64_MAX -> INT64_MIN, V=1, N=1.
  static const uint32_t code[] = {AddsImmX(0, 1, 1)};
  state_.cpu.x[1] = 0x7FFFFFFFFFFFFFFFULL;  // INT64_MAX
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x8000000000000000ULL});  // INT64_MIN
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagOverflow);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
}

// 32-bit SUBS (W form): borrow case with zero-extended result.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsImmW32) {
  // SUBS W0, W1, #1 with W1 == 0 -> 0xFFFFFFFF, N=1, C=0 (borrow).
  static const uint32_t code[] = {SubsImmW(0, 1, 1)};
  state_.cpu.x[1] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFULL});  // W-result zero-extended
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
}

// 32-bit ADDS (W form): carry-out at the 32-bit boundary, Z=1.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsImmW32CarryOut) {
  // ADDS W0, W1, #1 with W1 == 0xFFFFFFFF -> 0, C=1, Z=1.
  static const uint32_t code[] = {AddsImmW(0, 1, 1)};
  state_.cpu.x[1] = 0xFFFFFFFFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// CMP (SUBS to XZR): flags set, result discarded, x[0] untouched.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmpImmDiscardsResult) {
  // X0 holds a sentinel; CMP X1, #5 (== SUBS XZR, X1, #5) with X1 == 5.
  static const uint32_t code[] = {CmpImmX(1, 5)};
  state_.cpu.x[0] = 0xDEADBEEFCAFEF00DULL;  // sentinel: must not be clobbered
  state_.cpu.x[1] = 5;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xDEADBEEFCAFEF00DULL});  // untouched
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
}

// SUBS shifted register: X0 = X1 - (X2 << 4), flags set.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsRegShifted) {
  // SUBS X0, X1, X2, LSL #4 with X1 == 0x100, X2 == 0x10 -> 0x100 - 0x100 = 0.
  static const uint32_t code[] = {SubsRegLsl(0, 1, 2, 4)};
  state_.cpu.x[1] = 0x100;
  state_.cpu.x[2] = 0x10;  // << 4 = 0x100
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
}

// SUBS register (no shift): borrow case.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsRegBorrow) {
  // SUBS X0, X1, X2 with X1 < X2 -> borrow (C=0), N=1.
  static const uint32_t code[] = {SubsRegX(0, 1, 2)};
  state_.cpu.x[1] = 3;
  state_.cpu.x[2] = 10;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFF9ULL});  // -7
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
}

// ADDS register (no shift): carry-out and overflow.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsRegCarryAndOverflow) {
  // ADDS X0, X1, X2 with X1 == X2 == INT64_MIN -> 0, C=1 (carry out), V=1.
  static const uint32_t code[] = {AddsRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x8000000000000000ULL;
  state_.cpu.x[2] = 0x8000000000000000ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagOverflow);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
}

// 32-bit SUBS register: equal operands -> Z=1, C=1, zero-extended result.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsRegW32Equal) {
  static const uint32_t code[] = {SubsRegW(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFF00001234ULL;  // W1 = 0x1234
  state_.cpu.x[2] = 0x1234;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});  // upper 32 cleared
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
}

// 32-bit ADDS register: carry-out at 32-bit boundary.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsRegW32CarryOut) {
  static const uint32_t code[] = {AddsRegW(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFFULL;
  state_.cpu.x[2] = 1;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
}

// ADDS extended register: X0 = X1 + ((X2 & 0xFF) << 2), flags set.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsExtendedUxtb) {
  // ADDS X0, X1, X2, UXTB #2 with low byte of X2 == 0x80.
  static const uint32_t code[] = {AddsExtX(0, 1, 2, /*UXTB=*/0b000, 2)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.x[2] = 0xFFFFFF80ULL;  // low byte 0x80
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1000 + (0x80 << 2)});
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// ANDS producing Z=1 (result zero), and C==0 && V==0 always for AND.
TEST_F(Arm64HeavyOptimizerFrontendTest, AndsRegZeroSetsZ) {
  // ANDS X0, X1, X2 with disjoint bits -> 0, Z=1, C=0, V=0.
  static const uint32_t code[] = {AndsRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x00FF;
  state_.cpu.x[2] = 0xFF00;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);     // AND clears C
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);  // AND clears V
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
}

// ANDS producing N=1 (MSB set), and C==0 && V==0.
TEST_F(Arm64HeavyOptimizerFrontendTest, AndsRegNegativeSetsN) {
  // ANDS X0, X1, X2 with top bit kept -> N=1, C=0, V=0.
  static const uint32_t code[] = {AndsRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x8000000000000001ULL;
  state_.cpu.x[2] = 0x8000000000000000ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x8000000000000000ULL});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// ANDS logical immediate: Z=1, and C==0 && V==0.
TEST_F(Arm64HeavyOptimizerFrontendTest, AndsImmZeroSetsZ) {
  // ANDS X0, X1, #0xFF with X1 having no low byte -> 0, Z=1.
  static const uint32_t code[] = {AndsImmX(0, 1, 0, 7)};
  state_.cpu.x[1] = 0xAB00;  // low byte zero
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// TST (ANDS to XZR): flags set, result discarded.
TEST_F(Arm64HeavyOptimizerFrontendTest, TstImmDiscardsResult) {
  // TST X1, #0xFF == ANDS XZR, X1, #0xFF. X0 sentinel must be preserved.
  static const uint32_t code[] = {AndsImmX(31, 1, 0, 7)};
  state_.cpu.x[0] = 0x1234567890ABCDEFULL;  // sentinel
  state_.cpu.x[1] = 0xAB00;                 // low byte zero -> Z=1
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1234567890ABCDEFULL});  // untouched
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
}

// ADC/SBC (add/subtract with carry) bails: needs the carry-in from cpu.flags.
TEST_F(Arm64HeavyOptimizerFrontendTest, AdcBails) {
  // ADC X0, X1, X2 -> 0x9A000000 base | (rm<<16) | (rn<<5) | rd.
  static const uint32_t code[] = {0x9A020020u};  // ADC X0, X1, X2
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AddReg64) {
  static const uint32_t code[] = {AddRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x100;
  state_.cpu.x[2] = 0x023;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x123});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SubReg64) {
  static const uint32_t code[] = {SubRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x500;
  state_.cpu.x[2] = 0x123;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x500 - 0x123});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AndReg64) {
  static const uint32_t code[] = {AndRegX(0, 1, 2)};
  state_.cpu.x[1] = 0xFF0F;
  state_.cpu.x[2] = 0x0FF0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFF0F & 0x0FF0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SubRegShifted) {
  // SUB X0, X1, X2, LSL #4 -> 0x1000 - (0x10 << 4) = 0x1000 - 0x100 = 0xF00.
  static const uint32_t code[] = {SubRegLsl(0, 1, 2, 4)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.x[2] = 0x10;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xF00});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AndRegLsrShifted) {
  // AND X0, X1, X2, LSR #4.
  static const uint32_t code[] = {AndRegLsr(0, 1, 2, 4)};
  state_.cpu.x[1] = 0x0F0F;
  state_.cpu.x[2] = 0xFF00;  // >>4 = 0x0FF0
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0F0F & 0x0FF0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, OrrReg64) {
  static const uint32_t code[] = {OrrRegX(0, 1, 2)};
  state_.cpu.x[1] = 0xF0F0;
  state_.cpu.x[2] = 0x0F0F;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFF});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, EorReg64) {
  static const uint32_t code[] = {EorRegX(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFF;
  state_.cpu.x[2] = 0x0F0F;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xF0F0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BicReg64) {
  // BIC X0, X1, X2 -> X1 & ~X2.
  static const uint32_t code[] = {BicRegX(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFF;
  state_.cpu.x[2] = 0x0F0F;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xF0F0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, OrnReg64) {
  // ORN X0, X1, X2 -> X1 | ~X2.
  static const uint32_t code[] = {OrnRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x00FF;
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFF00ULL;  // ~X2 = 0xFF
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFF});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AddExtendedUxtb) {
  // ADD X0, X1, X2, UXTB #2 -> X1 + ((X2 & 0xFF) << 2).
  static const uint32_t code[] = {AddExtX(0, 1, 2, /*UXTB=*/0b000, 2)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.x[2] = 0xFFFFFF80ULL;  // low byte 0x80
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1000 + (0x80 << 2)});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AddExtendedSxtb) {
  // ADD X0, X1, X2, SXTB -> X1 + sext(X2[7:0]).
  static const uint32_t code[] = {AddExtX(0, 1, 2, /*SXTB=*/0b100, 0)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.x[2] = 0x80;  // sign-extends to -128
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1000 - 128});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AndImm64) {
  // AND X0, X1, #0xFF (N=1, immr=0, imms=7).
  static const uint32_t code[] = {AndImmX(0, 1, 0, 7)};
  state_.cpu.x[1] = 0x12345678;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x78});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, OrrImm64) {
  // ORR X0, X1, #0xF (N=1, immr=0, imms=3).
  static const uint32_t code[] = {OrrImmX(0, 1, 0, 3)};
  state_.cpu.x[1] = 0x1230;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x123F});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, EorImm64) {
  // EOR X0, X1, #0xFF (N=1, immr=0, imms=7).
  static const uint32_t code[] = {EorImmX(0, 1, 0, 7)};
  state_.cpu.x[1] = 0x12FF;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1200});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Madd64) {
  // MADD X0, X1, X2, X3 -> X3 + X1*X2.
  static const uint32_t code[] = {MaddX(0, 1, 2, 3)};
  state_.cpu.x[1] = 6;
  state_.cpu.x[2] = 7;
  state_.cpu.x[3] = 5;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{5 + 6 * 7});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Msub64) {
  // MSUB X0, X1, X2, X3 -> X3 - X1*X2.
  static const uint32_t code[] = {MsubX(0, 1, 2, 3)};
  state_.cpu.x[1] = 6;
  state_.cpu.x[2] = 7;
  state_.cpu.x[3] = 100;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{100 - 6 * 7});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Smaddl) {
  // SMADDL X0, W1, W2, X3 -> X3 + sext(W1)*sext(W2).
  static const uint32_t code[] = {SmaddlX(0, 1, 2, 3)};
  state_.cpu.x[1] = 0xFFFFFFFFULL;  // W1 = -1
  state_.cpu.x[2] = 5;
  state_.cpu.x[3] = 100;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{100 - 5});  // 100 + (-1)*5
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Umaddl) {
  // UMADDL X0, W1, W2, X3 -> X3 + zext(W1)*zext(W2).
  static const uint32_t code[] = {UmaddlX(0, 1, 2, 3)};
  state_.cpu.x[1] = 0xFFFFFFFFULL;  // W1 = 4294967295
  state_.cpu.x[2] = 2;
  state_.cpu.x[3] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFULL * 2});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SmulhBails) {
  static const uint32_t code[] = {SmulhX(0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Lslv64) {
  static const uint32_t code[] = {LslvX(0, 1, 2)};
  state_.cpu.x[1] = 0x1;
  state_.cpu.x[2] = 8;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x100});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Lsrv64) {
  static const uint32_t code[] = {LsrvX(0, 1, 2)};
  state_.cpu.x[1] = 0x100;
  state_.cpu.x[2] = 4;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x10});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Asrv64) {
  static const uint32_t code[] = {AsrvX(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFF00ULL;  // -256
  state_.cpu.x[2] = 4;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFF0ULL});  // -16
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Rorv64) {
  static const uint32_t code[] = {RorvX(0, 1, 2)};
  state_.cpu.x[1] = 0x0000000000000001ULL;
  state_.cpu.x[2] = 4;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1000000000000000ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UdivBails) {
  static const uint32_t code[] = {UdivX(0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UbfmLsr64) {
  // LSR X0, X1, #8  == UBFM X0, X1, #8, #63.
  static const uint32_t code[] = {UbfmX(0, 1, 8, 63)};
  state_.cpu.x[1] = 0x1234567800000000ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0012345678000000ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UbfmLsl64) {
  // LSL X0, X1, #4 == UBFM X0, X1, #(64-4), #(63-4) = #60, #59.
  static const uint32_t code[] = {UbfmX(0, 1, 60, 59)};
  state_.cpu.x[1] = 0x12;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x120});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UbfmUxtbW) {
  // UXTB W0, W1 == UBFM W0, W1, #0, #7.
  static const uint32_t code[] = {UbfmW(0, 1, 0, 7)};
  state_.cpu.x[1] = 0x1234ABCD;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xCD});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UbfmExtract64) {
  // UBFX X0, X1, #8, #8 (extract 8 bits starting at bit 8) == UBFM X0,X1,#8,#15.
  static const uint32_t code[] = {UbfmX(0, 1, 8, 15)};
  state_.cpu.x[1] = 0x0000000000ABCDEFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xCD});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfmAsr64) {
  // ASR X0, X1, #4 == SBFM X0, X1, #4, #63.
  static const uint32_t code[] = {SbfmX(0, 1, 4, 63)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFF00ULL;  // -256
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFF0ULL});  // -16
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfmSxtb64) {
  // SXTB X0, W1 == SBFM X0, X1, #0, #7.
  static const uint32_t code[] = {SbfmX(0, 1, 0, 7)};
  state_.cpu.x[1] = 0x80;  // sign bit set -> -128
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFF80ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfmSxtw64) {
  // SXTW X0, W1 == SBFM X0, X1, #0, #31.
  static const uint32_t code[] = {SbfmX(0, 1, 0, 31)};
  state_.cpu.x[1] = 0x80000000ULL;  // sign bit set -> negative
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFF80000000ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BfmBfi64) {
  // BFI X0, X1, #8, #8: insert low 8 bits of X1 at bit 8, keep other X0 bits.
  // == BFM X0, X1, #(64-8)=#56, #(8-1)=#7.
  static const uint32_t code[] = {BfmX(0, 1, 56, 7)};
  state_.cpu.x[0] = 0xFFFFFFFFFFFFFFFFULL;
  state_.cpu.x[1] = 0xAB;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFABFFULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BfmBfxil64) {
  // BFXIL X0, X1, #8, #8: take bits[15:8] of X1 into bits[7:0] of X0, keep rest.
  // == BFM X0, X1, #immr=8, #imms=15.
  static const uint32_t code[] = {BfmX(0, 1, 8, 15)};
  state_.cpu.x[0] = 0xFFFFFFFFFFFFFF00ULL;
  state_.cpu.x[1] = 0xAB00;  // bits[15:8] = 0xAB
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFABULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, ExtrCopyLsb0) {
  // EXTR X0, X1, X2, #0 -> copy of X2.
  static const uint32_t code[] = {ExtrX(0, 1, 2, 0)};
  state_.cpu.x[1] = 0x1111111111111111ULL;
  state_.cpu.x[2] = 0x2222222222222222ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x2222222222222222ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Extr32) {
  // EXTR W0, W1, W2, #4: W0 = (W1:W2) >> 4 (low 32).
  static const uint32_t code[] = {ExtrW(0, 1, 2, 4)};
  state_.cpu.x[1] = 0x0000000F;  // W1 low nibble -> top of result
  state_.cpu.x[2] = 0xABCDEF00;  // W2
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  // (0x0000000F:0xABCDEF00) >> 4 = 0xFABCDEF0.
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFABCDEF0ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Extr64Bails) {
  // EXTR X0, X1, X2, #4 (64-bit non-zero lsb) bails (no 64-bit SHRD MachineIR op).
  static const uint32_t code[] = {ExtrX(0, 1, 2, 4)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Clz64) {
  static const uint32_t code[] = {ClzX(0, 1)};
  state_.cpu.x[1] = 0x0000000000000100ULL;  // bit 8 set -> 55 leading zeros
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  if (RunOneInstruction(&state_, stop_pc)) {
    EXPECT_EQ(state_.cpu.x[0], uint64_t{55});
  } else {
    // Host without LZCNT: CLZ bails to lite, which is also correct.
    GTEST_SKIP() << "host lacks LZCNT; CLZ bailed";
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Rev16_64) {
  // REV16 reverses bytes within each 16-bit halfword.
  static const uint32_t code[] = {Rev16X(0, 1)};
  state_.cpu.x[1] = 0x1122334455667788ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x2211443366558877ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, RevBails) {
  // REV needs x86 BSWAP, which has no MachineIR op here -> bails.
  static const uint32_t code[] = {RevX(0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
}

//
// Multi-instruction region tests: these exercise GenCode's CheckMachineIR on
// real multi-insn IR (the 1-insn tests above do not).
//

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluRegionNoBail) {
  // MOVZ X0,#5; ADD X1,X0,#3; SUB X2,X1,#1; ORR X3,X2,X0.
  static const uint32_t code[] = {
      MovzX(0, 5), AddImmX(1, 0, 3), SubImmX(2, 1, 1), OrrRegX(3, 2, 0)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_TRUE(ok);
  EXPECT_EQ(n, 4u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluThenBailRegion) {
  // MOVZ; ADD; SUB; then a bailing UDIV ends the region after 3 translated.
  static const uint32_t code[] = {
      MovzX(0, 0x10), AddImmX(1, 0, 4), SubImmX(2, 1, 2), UdivX(3, 2, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 3u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluMixedShiftRegion) {
  // A mix of shifted-register, extended-register, logical-immediate and a
  // multiply, then a bailing SMULH. Exercises CheckMachineIR over the whole run.
  static const uint32_t code[] = {MovzX(0, 0x7),
                                  MovzX(1, 0x3),
                                  AddRegX(2, 0, 1),
                                  SubRegLsl(3, 2, 1, 2),
                                  AndImmX(4, 3, 0, 7),
                                  MaddX(5, 0, 1, 4),
                                  SmulhX(6, 5, 0)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // 6 translate, the SMULH bails.
  EXPECT_EQ(n, 6u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluRegionExecutes) {
  // Full region execution end-to-end: MOVZ X0,#5; ADD X1,X0,#3; SUB X2,X1,#1.
  // Expected: X0=5, X1=8, X2=7.
  static const uint32_t code[] = {MovzX(0, 5), AddImmX(1, 0, 3), SubImmX(2, 1, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  MachineCode mc;
  auto [stop, ok, n] =
      HeavyOptimizeRegion(ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = end_pc});
  ASSERT_TRUE(ok);
  ASSERT_EQ(n, 3u);
  ScopedExecRegion exec(&mc);
  TestingRunGeneratedCode(&state_, exec.get(), end_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{5});
  EXPECT_EQ(state_.cpu.x[1], uint64_t{8});
  EXPECT_EQ(state_.cpu.x[2], uint64_t{7});
}

// A flag-setting SUBS inside a multi-insn region must produce valid IR (the
// EmitMaterializeNZCV sequence — PseudoReadFlags + AND + store — must survive
// GenCode's CheckMachineIR). A following ADC bails, ending the region.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiFlagSetterThenBailRegion) {
  // MOVZ X0,#10; SUBS X1,X0,#3; ADC X2,X1,X0 (bails).
  static const uint32_t code[] = {MovzX(0, 10), SubsImmX(1, 0, 3), 0x9A000022u /*ADC X2,X1,X0*/};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // MOVZ + SUBS translate, the ADC bails -> partial region of 2.
  EXPECT_EQ(n, 2u);
}

// Full region execution end-to-end with a flag-setter: MOVZ X0,#7; SUBS X1,X0,#7
// must leave X1==0 and the Z and C flags set in cpu.flags after the region runs.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiFlagSetterRegionExecutes) {
  static const uint32_t code[] = {MovzX(0, 7), SubsImmX(1, 0, 7)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  MachineCode mc;
  auto [stop, ok, n] =
      HeavyOptimizeRegion(ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = end_pc});
  ASSERT_TRUE(ok);
  ASSERT_EQ(n, 2u);
  ScopedExecRegion exec(&mc);
  TestingRunGeneratedCode(&state_, exec.get(), end_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{7});
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
}

// A non-MoveWide instruction must bail out of the optimizing frontend (the
// runtime then falls back to the lite translator / interpreter).
TEST_F(Arm64HeavyOptimizerFrontendTest, NonMoveWideBails) {
  // CRC32 (DataProc2Src) is not translated by the optimizing frontend.
  static const uint32_t code[] = {0x1AC04020};  // CRC32B W0, W1, W0
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
}

}  // namespace

}  // namespace berberis
