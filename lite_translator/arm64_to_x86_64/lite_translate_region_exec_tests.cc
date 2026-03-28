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

TEST_F(Arm64LiteTranslateRegionTest, SvcEndsRegion) {
  // region digitalis - SVC sets success_=false so the interpreter handles it.
  // The region translates MOVZ successfully, then fails at SVC.
  // The dispatch loop installs kInterpreted for the SVC address.
  static const uint32_t code[] = {
      MovzX(0, 1),    // MOVZ X0, #1 (translatable)
      0xD4000001,     // SVC #0 (interpreter handles syscall)
  };
  MachineCode machine_code;
  auto [success, stop_pc] = TryLiteTranslateRegion(ToGuestAddr(code),
                                                   &machine_code,
                                                   LiteTranslateParams{
                                                       .end_pc = ToGuestAddr(code) + 8,
                                                       .allow_dispatch = false,
                                                   });
  // SVC causes translation failure at its PC — interpreter will handle it.
  EXPECT_FALSE(success);
  EXPECT_EQ(stop_pc, ToGuestAddr(code) + 4);  // PC of the SVC instruction
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

// STLR Wt, [Xn]: size=10, o2=1, L=0, o1=0, Rs=11111, o0=1, Rt2=11111
// Encoding: 10 001000 1 0 0 11111 1 11111 Rn Rt
constexpr uint32_t StlrW(uint8_t rt, uint8_t rn) {
  return 0x889FFC00 | (static_cast<uint32_t>(rn) << 5) | rt;
}

// LDR Wt, [Xn] (unsigned offset 0): size=10, V=0, opc=01
// Encoding: 1011 1001 01 imm12=0 Rn Rt
constexpr uint32_t LdrWUnsigned(uint8_t rt, uint8_t rn) {
  return 0xB9400000 | (static_cast<uint32_t>(rn) << 5) | rt;
}

TEST_F(Arm64LiteTranslateRegionTest, StlrWritesToMemory) {
  // STLR W1, [X0] should write the value in W1 to the address in X0.
  // Then LDR W2, [X0] should read it back.
  static uint32_t target_mem = 0;
  static const uint32_t code[] = {
      MovzX(1, 42),                // MOVZ X1, #42 (value to store)
      StlrW(1, 0),                 // STLR W1, [X0] (store-release)
      LdrWUnsigned(2, 0),          // LDR W2, [X0] (load it back)
  };
  target_mem = 0;
  state_.cpu.x[0] = ToGuestAddr(&target_mem);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target_mem, 42u);
  EXPECT_EQ(state_.cpu.x[2], 42ULL);
}

// DUP V0.16B, Wn: broadcast byte from GP register to all 16 lanes of V0.
// Encoding: 0 Q=1 0 01110 000 imm5=00001 0 imm4=0001 1 Rn Rd
// = 0100 1110 0000 0001 0000 0111 00 Rn Rd
constexpr uint32_t DupV16B(uint8_t rd, uint8_t rn) {
  return 0x4E010C00 | (static_cast<uint32_t>(rn) << 5) | rd;
}

// STP Q<rt1>, Q<rt2>, [Xn], #32 (post-index, 128-bit pair store)
// opc=10, V=1, type=001(post-index), L=0(store), imm7=2(32/16), Rt2, Rn, Rt
// Encoding: 10 101 1 001 0 0000010 Rt2 Rn Rt
constexpr uint32_t StpQPostIndex(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm_div16) {
  uint32_t imm7 = static_cast<uint32_t>(imm_div16) & 0x7F;
  return 0xAC800000 | (imm7 << 15) | (static_cast<uint32_t>(rt2) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rt1;
}

TEST_F(Arm64LiteTranslateRegionTest, MemsetPattern) {
  // Test the ARM64 memset pattern: DUP V0.16B, W0 + STP Q0, Q0, [X1], #32
  // This writes 32 bytes of the byte value in W0 to the address in X1.
  alignas(16) static uint8_t buffer[64];
  memset(buffer, 0xCC, sizeof(buffer));

  static const uint32_t code[] = {
      DupV16B(0, 0),                 // DUP V0.16B, W0 (broadcast byte)
      StpQPostIndex(0, 0, 1, 2),     // STP Q0, Q0, [X1], #32
  };
  state_.cpu.x[0] = 0xAB;            // byte value to fill
  state_.cpu.x[1] = ToGuestAddr(buffer);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // First 32 bytes should be 0xAB
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(buffer[i], 0xAB) << "byte " << i;
  }
  // Remaining bytes should be untouched
  for (int i = 32; i < 64; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
  // X1 should be advanced by 32
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(buffer) + 32);
}

TEST_F(Arm64LiteTranslateRegionTest, ForwardBranchExtension) {
  // Forward B.EQ should NOT end the region — code after it should also translate.
  // X0=10, CMP X0,#5 → Z=0 → B.EQ not taken → fall through to ADD.
  static const uint32_t code[] = {
      MovzX(0, 10),         // MOVZ X0, #10
      CmpImmX(0, 5),        // CMP X0, #5 → Z=0
      Bcond(kCondEQ, 8),    // B.EQ +8 (forward, not taken)
      AddImmX(0, 0, 1),     // ADD X0, X0, #1 → X0=11 (should be in same region)
      AddImmX(0, 0, 2),     // ADD X0, X0, #2 → X0=13 (branch target, also translated)
  };
  // With forward branch extension, the entire block should be one region.
  // Without it, the region would end at B.EQ and X0 would be 10 (ADD not reached).
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 13ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, ForwardBranchTaken) {
  // Forward B.EQ taken: X0=5, CMP X0,#5 → Z=1 → B.EQ taken → skips ADD X0,#1.
  static const uint32_t code[] = {
      MovzX(0, 5),          // MOVZ X0, #5
      CmpImmX(0, 5),        // CMP X0, #5 → Z=1
      Bcond(kCondEQ, 8),    // B.EQ +8 (forward, taken → skips next insn)
      AddImmX(0, 0, 1),     // ADD X0, X0, #1 (skipped)
      AddImmX(0, 0, 2),     // ADD X0, X0, #2 → X0=7 (branch target)
  };
  // B.EQ is taken, so it exits the region to code+16 (the second ADD).
  // The dispatch will handle code+16 as a new region.
  GuestAddr branch_target = ToGuestAddr(code) + 16;
  EXPECT_TRUE(Run(code, branch_target));
  EXPECT_EQ(state_.cpu.x[0], 5ULL);  // Branch exits before any ADD executes
}
// --- Memset instruction variant tests ---

// STUR Q<rt>, [Xn, #imm9]: Store 128-bit SIMD with unscaled immediate.
// Encoding: 00 111 1 00 10 imm9 00 Rn Rt
constexpr uint32_t SturQ(uint8_t rt, uint8_t rn, int16_t imm9) {
  uint32_t uimm9 = static_cast<uint32_t>(imm9) & 0x1FF;
  return 0x3C800000 | (uimm9 << 12) | (static_cast<uint32_t>(rn) << 5) | rt;
}

// STR Q<rt>, [Xn, Xm]: Store 128-bit SIMD with register offset (no shift).
// Encoding: 00 111 1 00 10 1 Rm 011 0 00 Rn Rt
constexpr uint32_t StrQReg(uint8_t rt, uint8_t rn, uint8_t rm) {
  return 0x3CA06800 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

// STR Q<rt>, [Xn, #uimm12]: Store 128-bit SIMD with unsigned immediate (scaled by 16).
constexpr uint32_t StrQUnsigned(uint8_t rt, uint8_t rn, uint16_t imm12_div16) {
  return 0x3D800000 | (static_cast<uint32_t>(imm12_div16) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

// STUR S<rt>, [Xn, #imm9]: Store 32-bit SIMD with unscaled immediate.
// Encoding: 10 111 1 00 00 imm9 00 Rn Rt
constexpr uint32_t SturS(uint8_t rt, uint8_t rn, int16_t imm9) {
  uint32_t uimm9 = static_cast<uint32_t>(imm9) & 0x1FF;
  return 0xBC000000 | (uimm9 << 12) | (static_cast<uint32_t>(rn) << 5) | rt;
}

// STR S<rt>, [Xn, Xm, LSL #2]: Store 32-bit SIMD with register offset and shift.
// Encoding: 10 111 1 00 00 1 Rm 011 S 10 Rn Rt
// S=1 means shift by 2 (log2 of 4 bytes).
constexpr uint32_t StrSRegLsl2(uint8_t rt, uint8_t rn, uint8_t rm) {
  return 0xBC207800 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

// STR S<rt>, [Xn, #uimm]: Store 32-bit SIMD unsigned offset.
constexpr uint32_t StrSUnsigned(uint8_t rt, uint8_t rn, uint16_t imm12_div4) {
  return 0xBD000000 | (static_cast<uint32_t>(imm12_div4) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

// STP Q<rt1>, Q<rt2>, [Xn, #imm]: Store pair 128-bit, signed offset.
// opc=10, V=1, type=10(signed-offset), L=0, imm7, Rt2, Rn, Rt1
constexpr uint32_t StpQSigned(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm_div16) {
  uint32_t imm7 = static_cast<uint32_t>(imm_div16) & 0x7F;
  return 0xAD000000 | (imm7 << 15) | (static_cast<uint32_t>(rt2) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rt1;
}

// AND Xd, Xn, Xm, LSR #amount (logical shifted register)
constexpr uint32_t AndRegLsr(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t shift) {
  return 0x8A400000 | (static_cast<uint32_t>(shift) << 10) |
         (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5) | rd;
}

// SUB Xd, Xn, Xm, LSL #amount (shifted register)
constexpr uint32_t SubRegLsl(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t shift) {
  return 0xCB000000 | (static_cast<uint32_t>(shift) << 10) |
         (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5) | rd;
}

TEST_F(Arm64LiteTranslateRegionTest, SturQ_NegativeOffset) {
  // STUR Q0, [X1, #-16]: store 16 bytes of V0 at address X1-16.
  alignas(16) static uint8_t buffer[64];
  memset(buffer, 0xCC, sizeof(buffer));

  static const uint32_t code[] = {
      DupV16B(0, 0),           // DUP V0.16B, W0 (fill with 0xAB)
      SturQ(0, 1, -16),        // STUR Q0, [X1, #-16]
  };
  state_.cpu.x[0] = 0xAB;
  state_.cpu.x[1] = ToGuestAddr(buffer + 32);  // X1 points to buffer+32
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // Bytes 0-15: untouched
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
  // Bytes 16-31: should be 0xAB (stored at X1-16 = buffer+16)
  for (int i = 16; i < 32; i++) {
    EXPECT_EQ(buffer[i], 0xAB) << "byte " << i;
  }
  // Bytes 32-63: untouched
  for (int i = 32; i < 64; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, StrQ_RegisterOffset) {
  // STR Q0, [X0, X3]: store 16 bytes of V0 at address X0+X3.
  alignas(16) static uint8_t buffer[64];
  memset(buffer, 0xCC, sizeof(buffer));

  static const uint32_t code[] = {
      DupV16B(0, 0),           // DUP V0.16B, W0 (fill with 0xBB)
      StrQReg(0, 1, 2),        // STR Q0, [X1, X2]
  };
  state_.cpu.x[0] = 0xBB;
  state_.cpu.x[1] = ToGuestAddr(buffer);
  state_.cpu.x[2] = 16;  // offset
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // Bytes 0-15: untouched
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
  // Bytes 16-31: should be 0xBB
  for (int i = 16; i < 32; i++) {
    EXPECT_EQ(buffer[i], 0xBB) << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SturS_NegativeOffset) {
  // STUR S0, [X1, #-4]: store low 4 bytes of V0 at address X1-4.
  alignas(16) static uint8_t buffer[32];
  memset(buffer, 0xCC, sizeof(buffer));

  static const uint32_t code[] = {
      DupV16B(0, 0),           // DUP V0.16B, W0 (fill with 0xDD)
      SturS(0, 1, -4),         // STUR S0, [X1, #-4]
  };
  state_.cpu.x[0] = 0xDD;
  state_.cpu.x[1] = ToGuestAddr(buffer + 16);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // Bytes 12-15: should be 0xDD (stored at X1-4 = buffer+12)
  for (int i = 12; i < 16; i++) {
    EXPECT_EQ(buffer[i], 0xDD) << "byte " << i;
  }
  // Others untouched
  for (int i = 0; i < 12; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, StrS_RegisterOffsetShift) {
  // STR S0, [X1, X2, LSL #2]: store low 4 bytes of V0 at X1 + (X2 << 2).
  alignas(16) static uint8_t buffer[32];
  memset(buffer, 0xCC, sizeof(buffer));

  static const uint32_t code[] = {
      DupV16B(0, 0),             // DUP V0.16B, W0 (fill with 0xEE)
      StrSRegLsl2(0, 1, 2),      // STR S0, [X1, X2, LSL #2]
  };
  state_.cpu.x[0] = 0xEE;
  state_.cpu.x[1] = ToGuestAddr(buffer);
  state_.cpu.x[2] = 3;  // offset = 3 << 2 = 12
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // Bytes 12-15: should be 0xEE
  for (int i = 12; i < 16; i++) {
    EXPECT_EQ(buffer[i], 0xEE) << "byte " << i;
  }
  // Others untouched
  for (int i = 0; i < 12; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, StpQ_SignedOffset_Negative) {
  // STP Q0, Q0, [X1, #-32]: store 32 bytes at X1-32.
  alignas(16) static uint8_t buffer[64];
  memset(buffer, 0xCC, sizeof(buffer));

  static const uint32_t code[] = {
      DupV16B(0, 0),              // DUP V0.16B, W0 (fill with 0x55)
      StpQSigned(0, 0, 1, -2),    // STP Q0, Q0, [X1, #-32] (imm/16 = -2)
  };
  state_.cpu.x[0] = 0x55;
  state_.cpu.x[1] = ToGuestAddr(buffer + 48);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // Bytes 16-47: should be 0x55 (stored at X1-32 = buffer+16)
  for (int i = 16; i < 48; i++) {
    EXPECT_EQ(buffer[i], 0x55) << "byte " << i;
  }
  // Others untouched
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
  for (int i = 48; i < 64; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, AndShiftedReg_Lsr) {
  // AND X3, X3, X2, LSR #1: x3 = x3 & (x2 >> 1)
  static const uint32_t code[] = {
      MovzX(2, 32),              // X2 = 32
      MovzX(3, 16),              // X3 = 16
      AndRegLsr(3, 3, 2, 1),     // AND X3, X3, X2, LSR #1
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  // X2 >> 1 = 16, X3 & 16 = 16 & 16 = 16
  EXPECT_EQ(state_.cpu.x[3], 16ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, SubShiftedReg_Lsl) {
  // SUB X5, X4, X3, LSL #2: x5 = x4 - (x3 << 2)
  static const uint32_t code[] = {
      MovzX(4, 100),             // X4 = 100
      MovzX(3, 5),               // X3 = 5
      SubRegLsl(5, 4, 3, 2),     // SUB X5, X4, X3, LSL #2
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  // X3 << 2 = 20, X4 - 20 = 80
  EXPECT_EQ(state_.cpu.x[5], 80ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, Memset_16_63_Path) {
  // Test the full 16-63 byte memset path from __memset_aarch64.
  // For count=32, val=0:
  //   dup v0.16b, w1      (v0 = all zeros)
  //   mov x3, #16         (x3 = 16)
  //   and x3, x3, x2, lsr #1  (x3 = 16 & (32>>1) = 16 & 16 = 16)
  //   sub x5, x4, x3      (x5 = end - 16)
  //   str q0, [x0]         (store 16 bytes at start)
  //   str q0, [x0, x3]     (store 16 bytes at start+16)
  //   stur q0, [x5, #-16]  (store 16 bytes at (end-16)-16 = start)
  //   stur q0, [x4, #-16]  (store 16 bytes at end-16 = start+16)
  alignas(16) static uint8_t buffer[64];
  memset(buffer, 0xFF, sizeof(buffer));

  static const uint32_t code[] = {
      DupV16B(0, 1),                 // DUP V0.16B, W1 (w1=0 → v0 all zeros)
      MovzX(3, 16),                  // MOV X3, #16
      AndRegLsr(3, 3, 2, 1),         // AND X3, X3, X2, LSR #1
      // SUB X5, X4, X3 (no shift)
      0xCB030085,                    // SUB X5, X4, X3
      StrQUnsigned(0, 0, 0),         // STR Q0, [X0, #0]
      StrQReg(0, 0, 3),             // STR Q0, [X0, X3]
      SturQ(0, 5, -16),             // STUR Q0, [X5, #-16]
      SturQ(0, 4, -16),             // STUR Q0, [X4, #-16]
  };

  // Setup: x0=buffer, x1=0 (val), x2=32 (count), x4=buffer+32 (dstend)
  state_.cpu.x[0] = ToGuestAddr(buffer);
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 32;
  state_.cpu.x[4] = ToGuestAddr(buffer + 32);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // First 32 bytes should be zeroed
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(buffer[i], 0x00) << "byte " << i;
  }
  // Rest untouched
  for (int i = 32; i < 64; i++) {
    EXPECT_EQ(buffer[i], 0xFF) << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Memset_4_15_Path) {
  // Test the 4-15 byte memset path from __memset_aarch64.
  // For count=8, val=0:
  //   dup v0.16b, w1      (v0 = all zeros)
  //   lsr x3, x2, #3      (x3 = 8 >> 3 = 1)
  //   sub x5, x4, x3, lsl #2  (x5 = end - (1<<2) = end - 4)
  //   str s0, [x0]         (store 4 bytes at start)
  //   str s0, [x0, x3, lsl #2]  (store 4 bytes at start + 4)
  //   stur s0, [x5, #-4]   (store 4 bytes at (end-4) - 4 = start)
  //   stur s0, [x4, #-4]   (store 4 bytes at end - 4 = start + 4)
  alignas(16) static uint8_t buffer[32];
  memset(buffer, 0xFF, sizeof(buffer));

  // LSR X3, X2, #3 is UBFM X3, X2, #3, #63 = 0xd343fc43
  // But with rd=3, rn=2: matches the encoding.
  // UBFM: sf=1, opc=10, N=1, immr=3, imms=63
  // 1 10 100110 1 000011 111111 00010 00011
  // = 1101 0011 0100 0011 1111 1100 0100 0011 = 0xd343fc43
  constexpr uint32_t LsrImm3_X3_X2 = 0xd343fc43;

  static const uint32_t code[] = {
      DupV16B(0, 1),               // DUP V0.16B, W1 (w1=0 → v0 all zeros)
      LsrImm3_X3_X2,               // LSR X3, X2, #3
      SubRegLsl(5, 4, 3, 2),        // SUB X5, X4, X3, LSL #2
      StrSUnsigned(0, 0, 0),        // STR S0, [X0, #0]
      StrSRegLsl2(0, 0, 3),         // STR S0, [X0, X3, LSL #2]
      SturS(0, 5, -4),             // STUR S0, [X5, #-4]
      SturS(0, 4, -4),             // STUR S0, [X4, #-4]
  };

  // Setup: x0=buffer, x1=0 (val), x2=8 (count), x4=buffer+8 (dstend)
  state_.cpu.x[0] = ToGuestAddr(buffer);
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 8;
  state_.cpu.x[4] = ToGuestAddr(buffer + 8);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // First 8 bytes should be zeroed
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(buffer[i], 0x00) << "byte " << i;
  }
  // Rest untouched
  for (int i = 8; i < 32; i++) {
    EXPECT_EQ(buffer[i], 0xFF) << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, StpQ_SignedOffset_Positive) {
  // STP Q0, Q0, [X0, #32]: store 32 bytes at X0+32.
  alignas(16) static uint8_t buffer[96];
  memset(buffer, 0xCC, sizeof(buffer));

  static const uint32_t code[] = {
      DupV16B(0, 0),              // DUP V0.16B, W0 (fill with 0x77)
      StpQSigned(0, 0, 1, 2),    // STP Q0, Q0, [X1, #32] (imm/16 = 2)
  };
  state_.cpu.x[0] = 0x77;
  state_.cpu.x[1] = ToGuestAddr(buffer);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  // Bytes 0-31: untouched
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
  // Bytes 32-63: should be 0x77
  for (int i = 32; i < 64; i++) {
    EXPECT_EQ(buffer[i], 0x77) << "byte " << i;
  }
  // Bytes 64-95: untouched
  for (int i = 64; i < 96; i++) {
    EXPECT_EQ(buffer[i], 0xCC) << "byte " << i;
  }
}
// endregion

}  // namespace

}  // namespace berberis
// endregion
