// region digitalis
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
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

// Stub for ARM64 intrinsics init — not needed in JIT tests.
namespace intrinsics {
void InitState() {}
}  // namespace intrinsics

namespace {

// ARM64 instruction encoding helpers.
// MOVZ Xd, #imm16
constexpr uint32_t MovzX(uint8_t rd, uint16_t imm16) {
  return 0xD2800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// ADD Xd, Xn, #imm12
constexpr uint32_t AddImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x91000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}

// SUB Xd, Xn, #imm12
constexpr uint32_t SubImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0xD1000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}

// SUBS Xd, Xn, #imm12 (flag-setting subtract)
constexpr uint32_t SubsImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0xF1000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}

// CMP Xn, #imm12 (SUBS XZR, Xn, #imm12)
constexpr uint32_t CmpImmX(uint8_t rn, uint16_t imm12) {
  return SubsImmX(31, rn, imm12);
}

// ADD Xd, Xn, Xm (shifted register, no shift)
constexpr uint32_t AddRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x8B000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}

// B.cond offset (offset in bytes, must be multiple of 4)
constexpr uint32_t Bcond(uint8_t cond, int32_t offset) {
  uint32_t imm19 = static_cast<uint32_t>(offset / 4) & 0x7FFFF;
  return 0x54000000 | (imm19 << 5) | cond;
}

constexpr uint8_t kCondEQ = 0;
constexpr uint8_t kCondNE = 1;
constexpr uint8_t kCondCS = 0x2;
constexpr uint8_t kCondCC = 0x3;
constexpr uint8_t kCondHI = 0x8;
constexpr uint8_t kCondLS = 0x9;
constexpr uint8_t kCondGE = 0xA;
constexpr uint8_t kCondLT = 0xB;
constexpr uint8_t kCondGT = 0xC;
constexpr uint8_t kCondLE = 0xD;
constexpr uint8_t kCondAL = 0xE;

// CBZ Xt, offset
constexpr uint32_t CbzX(uint8_t rt, int32_t offset) {
  uint32_t imm19 = static_cast<uint32_t>(offset / 4) & 0x7FFFF;
  return 0xB4000000 | (imm19 << 5) | rt;
}

// CBNZ Xt, offset
constexpr uint32_t CbnzX(uint8_t rt, int32_t offset) {
  uint32_t imm19 = static_cast<uint32_t>(offset / 4) & 0x7FFFF;
  return 0xB5000000 | (imm19 << 5) | rt;
}

constexpr uint32_t kNop = 0xD503201F;

class Arm64LiteTranslateRegionTest : public ::testing::Test {
 public:
  template <typename T>
  void Reset(const T code) {
    state_.cpu.insn_addr = ToGuestAddr(code);
  }

  template <typename T>
  bool Run(T& code, GuestAddr expected_stop_addr) {
    Reset(code);
    GuestAddr code_end = ToGuestAddr(bit_cast<char*>(&code[0]) + sizeof(code));
    MachineCode machine_code;
    auto [success, stop_pc] = TryLiteTranslateRegion(state_.cpu.insn_addr,
                                                     &machine_code,
                                                     LiteTranslateParams{
                                                         .end_pc = code_end,
                                                         .allow_dispatch = false,
                                                     });

    if (!success || (stop_pc > code_end)) {
      return false;
    }

    // Install via CodePool so Jmp32 relocations to entry points are reachable.
    HostCodeAddr host_code = GetDefaultCodePoolInstance()->Add(&machine_code);
    TestingRunGeneratedCode(&state_, AsHostCode(host_code), expected_stop_addr);

    EXPECT_EQ(state_.cpu.insn_addr, expected_stop_addr);
    return true;
  }

 protected:
  ThreadState state_{};
};

TEST_F(Arm64LiteTranslateRegionTest, AddRegister) {
  static const uint32_t code[] = {
      MovzX(0, 10),       // MOVZ X0, #10
      MovzX(1, 20),       // MOVZ X1, #20
      AddRegX(2, 0, 1),   // ADD X2, X0, X1
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], 30ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, AddImmediate) {
  static const uint32_t code[] = {
      MovzX(0, 100),      // MOVZ X0, #100
      AddImmX(0, 0, 5),   // ADD X0, X0, #5
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 105ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, SubImmediate) {
  static const uint32_t code[] = {
      MovzX(0, 100),      // MOVZ X0, #100
      SubImmX(0, 0, 3),   // SUB X0, X0, #3
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 97ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, SubsSetsFlagsZero) {
  // SUBS X2, X0, X1 where X0==X1 should set Z flag.
  static const uint32_t code[] = {
      MovzX(0, 42),               // MOVZ X0, #42
      MovzX(1, 42),               // MOVZ X1, #42
      // SUBS X2, X0, X1 (shifted reg): sf=1, op=1, S=1, shift=00, imm6=0
      0xEB010002,                  // SUBS X2, X0, X1
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], 0ULL);
  // Flags: N=0, Z=1, C=1 (no borrow), V=0
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

TEST_F(Arm64LiteTranslateRegionTest, SubsSetsFlagsNegative) {
  // SUBS X2, X0, X1 where X0 < X1 should set N flag and clear C.
  static const uint32_t code[] = {
      MovzX(0, 5),                // MOVZ X0, #5
      MovzX(1, 10),               // MOVZ X1, #10
      0xEB010002,                  // SUBS X2, X0, X1
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  // 5 - 10 wraps around in unsigned
  EXPECT_EQ(state_.cpu.x[2], static_cast<uint64_t>(-5LL));
  // Flags: N=1, Z=0, C=0 (borrow occurred), V=0
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

TEST_F(Arm64LiteTranslateRegionTest, CmpBranchEqTaken) {
  // CMP X0, #5 with X0=5 → Z=1 → B.EQ taken.
  // JIT region: MOVZ, CMP, B.EQ. Branch exits the region.
  static const uint32_t code[] = {
      MovzX(0, 5),        // MOVZ X0, #5
      CmpImmX(0, 5),      // CMP X0, #5
      Bcond(kCondEQ, 8),  // B.EQ +8 (target = code+8+8 = code+16)
      kNop,               // padding (not translated)
      kNop,               // branch target (not translated)
  };
  // B.EQ is at code+8, target = code+16. JIT exits to code+16.
  GuestAddr branch_target = ToGuestAddr(code) + 16;
  EXPECT_TRUE(Run(code, branch_target));
  EXPECT_EQ(state_.cpu.x[0], 5ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CmpBranchEqNotTaken) {
  // CMP X0, #5 with X0=10 → Z=0 → B.EQ not taken → fall through.
  static const uint32_t code[] = {
      MovzX(0, 10),       // MOVZ X0, #10
      CmpImmX(0, 5),      // CMP X0, #5
      Bcond(kCondEQ, 8),  // B.EQ +8
  };
  // B.EQ not taken → exit to code+12 (past B.EQ).
  GuestAddr fall_through = ToGuestAddr(code) + sizeof(code);
  EXPECT_TRUE(Run(code, fall_through));
  EXPECT_EQ(state_.cpu.x[0], 10ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CmpBranchNeTaken) {
  // CMP X0, #5 with X0=10 → Z=0 → B.NE taken.
  static const uint32_t code[] = {
      MovzX(0, 10),       // MOVZ X0, #10
      CmpImmX(0, 5),      // CMP X0, #5
      Bcond(kCondNE, 8),  // B.NE +8 (target = code+16)
      kNop,
      kNop,
  };
  GuestAddr branch_target = ToGuestAddr(code) + 16;
  EXPECT_TRUE(Run(code, branch_target));
}

TEST_F(Arm64LiteTranslateRegionTest, CmpBranchLtTaken) {
  // CMP X0, #10 with X0=5 → N=1, V=0, N!=V → LT taken.
  static const uint32_t code[] = {
      MovzX(0, 5),         // MOVZ X0, #5
      CmpImmX(0, 10),      // CMP X0, #10
      Bcond(kCondLT, 8),   // B.LT +8
      kNop,
      kNop,
  };
  GuestAddr branch_target = ToGuestAddr(code) + 16;
  EXPECT_TRUE(Run(code, branch_target));
}

TEST_F(Arm64LiteTranslateRegionTest, CmpBranchGeTaken) {
  // CMP X0, #5 with X0=10 → N=0, V=0, N==V → GE taken.
  static const uint32_t code[] = {
      MovzX(0, 10),        // MOVZ X0, #10
      CmpImmX(0, 5),       // CMP X0, #5
      Bcond(kCondGE, 8),   // B.GE +8
      kNop,
      kNop,
  };
  GuestAddr branch_target = ToGuestAddr(code) + 16;
  EXPECT_TRUE(Run(code, branch_target));
}

TEST_F(Arm64LiteTranslateRegionTest, CbzTaken) {
  // CBZ X0, +8 with X0=0 → taken.
  static const uint32_t code[] = {
      MovzX(0, 0),        // MOVZ X0, #0
      CbzX(0, 8),         // CBZ X0, +8
      kNop,
      kNop,
  };
  GuestAddr branch_target = ToGuestAddr(code) + 12;
  EXPECT_TRUE(Run(code, branch_target));
}

TEST_F(Arm64LiteTranslateRegionTest, CbzNotTaken) {
  // CBZ X0, +8 with X0=1 → not taken.
  static const uint32_t code[] = {
      MovzX(0, 1),        // MOVZ X0, #1
      CbzX(0, 8),         // CBZ X0, +8
  };
  GuestAddr fall_through = ToGuestAddr(code) + sizeof(code);
  EXPECT_TRUE(Run(code, fall_through));
}

TEST_F(Arm64LiteTranslateRegionTest, CbnzTaken) {
  // CBNZ X0, +8 with X0=42 → taken.
  static const uint32_t code[] = {
      MovzX(0, 42),       // MOVZ X0, #42
      CbnzX(0, 8),        // CBNZ X0, +8
      kNop,
      kNop,
  };
  GuestAddr branch_target = ToGuestAddr(code) + 12;
  EXPECT_TRUE(Run(code, branch_target));
}

TEST_F(Arm64LiteTranslateRegionTest, GracefulFailure) {
  // region digitalis - SVC now ends region cleanly instead of failing.
  // SVC exits to interpreter, preserving JIT work before the SVC.
  static const uint32_t code[] = {
      MovzX(0, 1),    // MOVZ X0, #1 (translatable)
      0xD4000001,     // SVC #0 (ends region, interpreter handles syscall)
  };
  MachineCode machine_code;
  auto [success, stop_pc] = TryLiteTranslateRegion(ToGuestAddr(code),
                                                   &machine_code,
                                                   LiteTranslateParams{
                                                       .end_pc = ToGuestAddr(code) + 8,
                                                       .allow_dispatch = false,
                                                   });
  EXPECT_TRUE(success);
  // Region includes the SVC instruction (which exits to interpreter).
  // stop_pc is past the SVC since the instruction was processed.
  EXPECT_EQ(stop_pc, ToGuestAddr(code) + 8);
  // endregion
}

TEST_F(Arm64LiteTranslateRegionTest, Nop) {
  static const uint32_t code[] = {
      kNop,
      kNop,
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
}

// region digitalis
// CSEL Xd, Xn, Xm, cond: sf=1, op=0, S=0, 11010100, Rm, cond, 0, op2=0, Rn, Rd
constexpr uint32_t CselX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return 0x9A800000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) | (rn << 5) | rd;
}

TEST_F(Arm64LiteTranslateRegionTest, CselEqTrue) {
  // CMP X0, #5 with X0=5 -> Z=1 -> CSEL EQ selects src1 (X1=100).
  static const uint32_t code[] = {
      MovzX(0, 5),          // MOVZ X0, #5
      MovzX(1, 100),        // MOVZ X1, #100  (src1 = true case)
      MovzX(2, 200),        // MOVZ X2, #200  (src2 = false case)
      CmpImmX(0, 5),        // CMP X0, #5  -> sets Z=1
      CselX(3, 1, 2, kCondEQ),  // CSEL X3, X1, X2, EQ
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 100ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselEqFalse) {
  // CMP X0, #5 with X0=10 -> Z=0 -> CSEL EQ selects src2 (X2=200).
  static const uint32_t code[] = {
      MovzX(0, 10),         // MOVZ X0, #10
      MovzX(1, 100),        // MOVZ X1, #100  (src1 = true case)
      MovzX(2, 200),        // MOVZ X2, #200  (src2 = false case)
      CmpImmX(0, 5),        // CMP X0, #5  -> sets Z=0
      CselX(3, 1, 2, kCondEQ),  // CSEL X3, X1, X2, EQ
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselLtTrue) {
  // CMP X0, #10 with X0=5 -> N=1, V=0, N!=V -> LT true -> selects src1.
  static const uint32_t code[] = {
      MovzX(0, 5),                  // MOVZ X0, #5
      MovzX(1, 100),                // MOVZ X1, #100  (src1 = true case)
      MovzX(2, 200),                // MOVZ X2, #200  (src2 = false case)
      CmpImmX(0, 10),               // CMP X0, #10  -> N=1, V=0
      CselX(3, 1, 2, kCondLT),      // CSEL X3, X1, X2, LT
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 100ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselLtFalse) {
  // CMP X0, #5 with X0=10 -> N=0, V=0, N==V -> LT false -> selects src2.
  static const uint32_t code[] = {
      MovzX(0, 10),                 // MOVZ X0, #10
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 5),                // CMP X0, #5  -> N=0, V=0
      CselX(3, 1, 2, kCondLT),      // CSEL X3, X1, X2, LT
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselGeTrue) {
  // CMP X0, #5 with X0=10 -> N=0, V=0, N==V -> GE true -> selects src1.
  static const uint32_t code[] = {
      MovzX(0, 10),                 // MOVZ X0, #10
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 5),                // CMP X0, #5  -> N=0, V=0
      CselX(3, 1, 2, kCondGE),      // CSEL X3, X1, X2, GE
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 100ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselGeFalse) {
  // CMP X0, #10 with X0=5 -> N=1, V=0, N!=V -> GE false -> selects src2.
  static const uint32_t code[] = {
      MovzX(0, 5),                  // MOVZ X0, #5
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 10),               // CMP X0, #10  -> N=1, V=0
      CselX(3, 1, 2, kCondGE),      // CSEL X3, X1, X2, GE
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselHiTrue) {
  // CMP X0, #5 with X0=10 -> C=1, Z=0 -> HI true -> selects src1.
  static const uint32_t code[] = {
      MovzX(0, 10),                 // MOVZ X0, #10
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 5),                // CMP X0, #5  -> C=1, Z=0
      CselX(3, 1, 2, kCondHI),      // CSEL X3, X1, X2, HI
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 100ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselHiFalseEqual) {
  // CMP X0, #5 with X0=5 -> C=1, Z=1 -> HI false (Z==1) -> selects src2.
  static const uint32_t code[] = {
      MovzX(0, 5),                  // MOVZ X0, #5
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 5),                // CMP X0, #5  -> C=1, Z=1
      CselX(3, 1, 2, kCondHI),      // CSEL X3, X1, X2, HI
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselHiFalseLower) {
  // CMP X0, #10 with X0=5 -> C=0 -> HI false -> selects src2.
  static const uint32_t code[] = {
      MovzX(0, 5),                  // MOVZ X0, #5
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 10),               // CMP X0, #10  -> C=0
      CselX(3, 1, 2, kCondHI),      // CSEL X3, X1, X2, HI
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselLsTrue) {
  // CMP X0, #10 with X0=5 -> C=0 -> LS true (C==0) -> selects src1.
  static const uint32_t code[] = {
      MovzX(0, 5),                  // MOVZ X0, #5
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 10),               // CMP X0, #10  -> C=0
      CselX(3, 1, 2, kCondLS),      // CSEL X3, X1, X2, LS
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 100ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselLsFalse) {
  // CMP X0, #5 with X0=10 -> C=1, Z=0 -> LS false -> selects src2.
  static const uint32_t code[] = {
      MovzX(0, 10),                 // MOVZ X0, #10
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 5),                // CMP X0, #5  -> C=1, Z=0
      CselX(3, 1, 2, kCondLS),      // CSEL X3, X1, X2, LS
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselGtTrue) {
  // CMP X0, #5 with X0=10 -> Z=0, N=0, V=0, N==V -> GT true -> selects src1.
  static const uint32_t code[] = {
      MovzX(0, 10),                 // MOVZ X0, #10
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 5),                // CMP X0, #5  -> Z=0, N=0, V=0
      CselX(3, 1, 2, kCondGT),      // CSEL X3, X1, X2, GT
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 100ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselGtFalseEqual) {
  // CMP X0, #5 with X0=5 -> Z=1 -> GT false -> selects src2.
  static const uint32_t code[] = {
      MovzX(0, 5),                  // MOVZ X0, #5
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 5),                // CMP X0, #5  -> Z=1
      CselX(3, 1, 2, kCondGT),      // CSEL X3, X1, X2, GT
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselLeTrue) {
  // CMP X0, #10 with X0=5 -> N=1, V=0, N!=V -> LE true -> selects src1.
  static const uint32_t code[] = {
      MovzX(0, 5),                  // MOVZ X0, #5
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 10),               // CMP X0, #10  -> N=1, V=0
      CselX(3, 1, 2, kCondLE),      // CSEL X3, X1, X2, LE
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 100ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselLeFalse) {
  // CMP X0, #5 with X0=10 -> Z=0, N=0, V=0, N==V -> LE false -> selects src2.
  static const uint32_t code[] = {
      MovzX(0, 10),                 // MOVZ X0, #10
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CmpImmX(0, 5),                // CMP X0, #5  -> Z=0, N=0, V=0
      CselX(3, 1, 2, kCondLE),      // CSEL X3, X1, X2, LE
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, CselAlways) {
  // CSEL AL always selects src1 regardless of flags.
  static const uint32_t code[] = {
      MovzX(1, 100),                // MOVZ X1, #100
      MovzX(2, 200),                // MOVZ X2, #200
      CselX(3, 1, 2, kCondAL),      // CSEL X3, X1, X2, AL
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 100ULL);
}
// endregion

}  // namespace

}  // namespace berberis
// endregion
