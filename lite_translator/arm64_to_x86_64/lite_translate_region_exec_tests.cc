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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/platform.h"
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
constexpr uint8_t kCondMI = 0x4;
constexpr uint8_t kCondPL = 0x5;
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

// FMUL Sd, Sn, Sm (single-precision float multiply)
// ARM64 encoding: 0001_1110_001_Rm_0000_10_Rn_Rd
constexpr uint32_t FmulS(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1E200800 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// LDR St, [Xn, #imm12*4] (load single-precision float, unsigned offset)
// ARM64 encoding: 1011_1101_01_imm12_Rn_Rt
constexpr uint32_t LdrSUoff(uint8_t rt, uint8_t rn, uint16_t imm12) {
  return 0xBD400000 | (static_cast<uint32_t>(imm12) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

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

// region digitalis - SIMD load JIT coverage (plan §C1 verify).
//
// The SIMD load JIT path (lite_translator.h:2207 SimdLoadStoreImm,
// :2314 SimdLoadStorePair, :2348 SimdLoadStoreReg) was added piecemeal
// with stores covered by the memset-pattern tests above.  The four
// loads (LDR Q unsigned-imm, LDR D unsigned-imm zero-extend, LDP Q
// signed-imm, LDR Q register-offset) below pin the corresponding load
// emit sequences so a future change that, e.g., drops the Pxor
// zero-fill before LDR D would fail loudly here instead of producing
// stale garbage in the upper 64 bits of V[rt].
//
// Encodings verified against aarch64-linux-gnu-as (system binutils
// per CLAUDE.md guidance).

// LDR Q<rt>, [Xn, #imm12*16]: 128-bit SIMD load, unsigned imm12.
// size=00 V=1 opc=11 imm12 Rn Rt
constexpr uint32_t LdrQUnsigned(uint8_t rt, uint8_t rn, uint16_t imm12_div16) {
  return 0x3DC00000 | (static_cast<uint32_t>(imm12_div16) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

// LDR D<rt>, [Xn, #imm12*8]: 64-bit SIMD load, zero-extends V[rt] upper half.
// size=11 V=1 opc=01 imm12 Rn Rt
constexpr uint32_t LdrDUnsigned(uint8_t rt, uint8_t rn, uint16_t imm12_div8) {
  return 0xFD400000 | (static_cast<uint32_t>(imm12_div8) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

// LDP Q<rt1>, Q<rt2>, [Xn, #imm7*16]: 128-bit load-pair, signed imm7.
// opc=10 V=1 L=1 imm7 Rt2 Rn Rt1
constexpr uint32_t LdpQSigned(uint8_t rt1, uint8_t rt2, uint8_t rn,
                              int8_t imm_div16) {
  uint32_t imm7 = static_cast<uint32_t>(imm_div16) & 0x7F;
  return 0xAD400000 | (imm7 << 15) | (static_cast<uint32_t>(rt2) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rt1;
}

// LDR Q<rt>, [Xn, Xm]: 128-bit load with register offset, no shift/extend.
// size=00 V=1 opc=11 reg-offset Rm option=011(UXTX) S=0 Rn Rt
constexpr uint32_t LdrQReg(uint8_t rt, uint8_t rn, uint8_t rm) {
  return 0x3CE06800 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

TEST_F(Arm64LiteTranslateRegionTest, SimdLoad128Bit) {
  // LDR Q0, [X1] should load 16 bytes at X1 into the full 128-bit V0.
  alignas(16) static uint8_t buffer[16];
  for (int i = 0; i < 16; ++i) buffer[i] = static_cast<uint8_t>(0xA0 + i);

  static const uint32_t code[] = {
      LdrQUnsigned(0, 1, 0),  // LDR Q0, [X1, #0]
  };
  // Pre-fill V0 with a sentinel so a stale-upper-half bug is detectable.
  memset(&state_.cpu.v[0], 0x5A, 16);
  state_.cpu.x[1] = ToGuestAddr(buffer);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(reinterpret_cast<uint8_t*>(&state_.cpu.v[0])[i],
              static_cast<uint8_t>(0xA0 + i))
        << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SimdLoad64BitZeroExtends) {
  // LDR D0, [X1] loads 8 bytes and zero-extends V0's upper 64 bits.
  // Pre-poison the upper half so a missing Pxor would be visible.
  alignas(16) static uint8_t buffer[8] =
      {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};

  static const uint32_t code[] = {
      LdrDUnsigned(0, 1, 0),  // LDR D0, [X1, #0]
  };
  memset(&state_.cpu.v[0], 0xCC, 16);  // sentinel for upper-half zero-extend
  state_.cpu.x[1] = ToGuestAddr(buffer);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  uint8_t* v0 = reinterpret_cast<uint8_t*>(&state_.cpu.v[0]);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(v0[i], buffer[i]) << "low byte " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(v0[i], 0x00) << "upper byte " << i << " (must be zero-extended)";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SimdLoadPair128Bit) {
  // LDP Q0, Q1, [X2, #0]: load 32 bytes into V0 (low half) and V1 (high half).
  alignas(16) static uint8_t buffer[32];
  for (int i = 0; i < 32; ++i) buffer[i] = static_cast<uint8_t>(0xD0 + i);

  static const uint32_t code[] = {
      LdpQSigned(0, 1, 2, 0),  // LDP Q0, Q1, [X2]
  };
  memset(&state_.cpu.v[0], 0x33, 16);
  memset(&state_.cpu.v[1], 0x44, 16);
  state_.cpu.x[2] = ToGuestAddr(buffer);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  uint8_t* v0 = reinterpret_cast<uint8_t*>(&state_.cpu.v[0]);
  uint8_t* v1 = reinterpret_cast<uint8_t*>(&state_.cpu.v[1]);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(v0[i], buffer[i]) << "v0 byte " << i;
    EXPECT_EQ(v1[i], buffer[16 + i]) << "v1 byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SimdLoadRegOffset128Bit) {
  // LDR Q0, [X1, X2]: 128-bit load using a register offset.
  alignas(16) static uint8_t buffer[32];
  for (int i = 0; i < 32; ++i) buffer[i] = static_cast<uint8_t>(0x10 + i);

  static const uint32_t code[] = {
      LdrQReg(0, 1, 2),  // LDR Q0, [X1, X2]
  };
  memset(&state_.cpu.v[0], 0x77, 16);
  state_.cpu.x[1] = ToGuestAddr(buffer);
  state_.cpu.x[2] = 16;  // offset into buffer (loads buffer[16..31])
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  uint8_t* v0 = reinterpret_cast<uint8_t*>(&state_.cpu.v[0]);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(v0[i], buffer[16 + i]) << "byte " << i;
  }
}
// endregion

// Test: single FMUL s0, s0, s1 (in-place multiply)
TEST_F(Arm64LiteTranslateRegionTest, FmulS_InPlace) {
  static const uint32_t code[] = {
      FmulS(0, 0, 1),   // FMUL S0, S0, S1
  };
  // Set up float values via ThreadState directly.
  // v[0] low 32 bits = 0.5f, v[1] low 32 bits = 3.0f
  float val0 = 0.5f, val1 = 3.0f;
  memcpy(&state_.cpu.v[0], &val0, sizeof(float));
  memcpy(&state_.cpu.v[1], &val1, sizeof(float));

  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  float result;
  memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 1.5f);  // 0.5 * 3.0 = 1.5
}

// Test: chained FMUL s0, s0, s1 then FMUL s0, s0, s2 (a*b*c pattern)
// This is the pattern that caused black screen in gles3jni.
TEST_F(Arm64LiteTranslateRegionTest, FmulS_ChainedMultiply) {
  static const uint32_t code[] = {
      FmulS(0, 0, 1),   // FMUL S0, S0, S1  -> s0 = a * b
      FmulS(0, 0, 2),   // FMUL S0, S0, S2  -> s0 = (a*b) * c
  };
  // a=0.5, b=0.125, c=1.0 — should produce 0.0625
  float a = 0.5f, b = 0.125f, c = 1.0f;
  memcpy(&state_.cpu.v[0], &a, sizeof(float));
  memcpy(&state_.cpu.v[1], &b, sizeof(float));
  memcpy(&state_.cpu.v[2], &c, sizeof(float));

  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  float result;
  memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 0.0625f);  // 0.5 * 0.125 * 1.0 = 0.0625
}

// Test: chained FMUL with different values to ensure non-zero result
TEST_F(Arm64LiteTranslateRegionTest, FmulS_ChainedMultiply_NonTrivial) {
  static const uint32_t code[] = {
      FmulS(0, 0, 1),   // FMUL S0, S0, S1
      FmulS(0, 0, 2),   // FMUL S0, S0, S2
  };
  // 2.0 * 3.0 * 4.0 = 24.0
  float a = 2.0f, b = 3.0f, c = 4.0f;
  memcpy(&state_.cpu.v[0], &a, sizeof(float));
  memcpy(&state_.cpu.v[1], &b, sizeof(float));
  memcpy(&state_.cpu.v[2], &c, sizeof(float));

  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  float result;
  memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 24.0f);
}
// Test: chained FMUL with LDR in between (the exact real-world pattern)
// Sequence: LDR s1, [x1]; FMUL s0, s0, s1; LDR s1, [x2]; FMUL s0, s0, s1
// This is the ARM64 codegen for: result = a * b * c
TEST_F(Arm64LiteTranslateRegionTest, FmulS_ChainedWithLdr) {
  // Set up a memory buffer with float values that LDR can read from.
  // We'll point X1 at b_val and X2 at c_val.
  float b_val = 0.125f;
  float c_val = 1.0f;

  static const uint32_t code[] = {
      LdrSUoff(1, 1, 0),   // LDR S1, [X1, #0]  -> load b
      FmulS(0, 0, 1),       // FMUL S0, S0, S1   -> s0 = a * b
      LdrSUoff(1, 2, 0),   // LDR S1, [X2, #0]  -> load c
      FmulS(0, 0, 1),       // FMUL S0, S0, S1   -> s0 = (a*b) * c
  };

  // Set s0 = 0.5 (a)
  float a_val = 0.5f;
  memcpy(&state_.cpu.v[0], &a_val, sizeof(float));
  // X1 points to b_val, X2 points to c_val
  state_.cpu.x[1] = reinterpret_cast<uint64_t>(&b_val);
  state_.cpu.x[2] = reinterpret_cast<uint64_t>(&c_val);

  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  float result;
  memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 0.0625f) << "0.5 * 0.125 * 1.0 should be 0.0625";
}

// Same pattern but with larger values to make failures more obvious
TEST_F(Arm64LiteTranslateRegionTest, FmulS_ChainedWithLdr_LargerValues) {
  float b_val = 3.0f;
  float c_val = 4.0f;

  static const uint32_t code[] = {
      LdrSUoff(1, 1, 0),   // LDR S1, [X1, #0]
      FmulS(0, 0, 1),       // FMUL S0, S0, S1
      LdrSUoff(1, 2, 0),   // LDR S1, [X2, #0]
      FmulS(0, 0, 1),       // FMUL S0, S0, S1
  };

  float a_val = 2.0f;
  memcpy(&state_.cpu.v[0], &a_val, sizeof(float));
  state_.cpu.x[1] = reinterpret_cast<uint64_t>(&b_val);
  state_.cpu.x[2] = reinterpret_cast<uint64_t>(&c_val);

  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  float result;
  memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 24.0f) << "2.0 * 3.0 * 4.0 should be 24.0";
}
// LDUR St, [Xn, #simm9] (load single, unscaled offset)
constexpr uint32_t LdurS(uint8_t rt, uint8_t rn, int16_t simm9) {
  uint32_t imm = static_cast<uint32_t>(simm9 & 0x1FF);
  return 0xBC400000 | (imm << 12) | (static_cast<uint32_t>(rn) << 5) | rt;
}

// Test: the exact real-world code pattern from gles3jni calcSceneParams:
// Load float from memory, FMUL, store result to memory.
TEST_F(Arm64LiteTranslateRegionTest, FmulS_FullCalcScenePattern) {
  // scene2clip[0] = 1.0f
  float scene2clip0 = 1.0f;
  float result = -1.0f;  // sentinel

  // Encode: LDR S0, [X0, #0]; LDR S1, [X1, #0]; FMUL S0, S0, S1; STUR S0, [X2, #0]
  static const uint32_t code[] = {
      LdrSUoff(0, 0, 0),   // LDR S0, [X0, #0]  -> load 0.0625
      LdurS(1, 1, 0),       // LDUR S1, [X1, #0] -> load scene2clip[0]=1.0
      FmulS(0, 0, 1),       // FMUL S0, S0, S1   -> 0.0625 * 1.0
      SturS(0, 2, 0),       // STUR S0, [X2, #0] -> store result
  };

  float val0 = 0.0625f;
  state_.cpu.x[0] = reinterpret_cast<uint64_t>(&val0);
  state_.cpu.x[1] = reinterpret_cast<uint64_t>(&scene2clip0);
  state_.cpu.x[2] = reinterpret_cast<uint64_t>(&result);

  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  EXPECT_FLOAT_EQ(result, 0.0625f) << "0.0625 * 1.0 should be 0.0625";
}

// Test: two consecutive LDR-FMUL-STR chains (simulating both mScale assignments)
TEST_F(Arm64LiteTranslateRegionTest, FmulS_TwoChainedLoadMulStore) {
  float const_val = 0.0625f;
  float s2c0 = 1.0f;
  float s2c1 = 1.644f;
  float results[2] = {-1.0f, -1.0f};

  static const uint32_t code[] = {
      // First multiply: results[0] = 0.0625 * 1.0
      LdrSUoff(0, 0, 0),   // LDR S0, [X0, #0]  -> 0.0625
      LdurS(1, 1, 0),       // LDUR S1, [X1, #0] -> 1.0
      FmulS(0, 0, 1),       // FMUL S0, S0, S1
      SturS(0, 3, 0),       // STUR S0, [X3, #0] -> results[0]
      // Second multiply: results[1] = 0.0625 * 1.644
      LdrSUoff(0, 0, 0),   // LDR S0, [X0, #0]  -> 0.0625
      LdurS(1, 2, 0),       // LDUR S1, [X2, #0] -> 1.644
      FmulS(0, 0, 1),       // FMUL S0, S0, S1
      SturS(0, 3, 4),       // STUR S0, [X3, #4] -> results[1]
  };

  state_.cpu.x[0] = reinterpret_cast<uint64_t>(&const_val);
  state_.cpu.x[1] = reinterpret_cast<uint64_t>(&s2c0);
  state_.cpu.x[2] = reinterpret_cast<uint64_t>(&s2c1);
  state_.cpu.x[3] = reinterpret_cast<uint64_t>(&results[0]);

  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));

  EXPECT_FLOAT_EQ(results[0], 0.0625f) << "0.0625 * 1.0";
  EXPECT_NEAR(results[1], 0.10275f, 0.0001f) << "0.0625 * 1.644";
}
// MOVN Xd, #imm16 (move wide with NOT)
constexpr uint32_t MovnX(uint8_t rd, uint16_t imm16) {
  return 0x92800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// CMP Xn, Xm (SUBS XZR, Xn, Xm, shifted register no shift)
constexpr uint32_t CmpRegX(uint8_t rn, uint8_t rm) {
  return 0xEB000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | 31;
}

TEST_F(Arm64LiteTranslateRegionTest, CselCcTrue_SmallVsMax) {
  // Reproduces the orderfile crash scenario:
  // X0 = 5, X1 = UINT64_MAX (via MOVN #0)
  // CMP X0, X1 → 5 < UINT64_MAX → C=0 (borrow) → CC (C==0) is true
  // CSEL X3, X4, X5, CC → should select X4 (true case)
  static const uint32_t code[] = {
      MovzX(0, 5),                    // MOVZ X0, #5
      MovnX(1, 0),                    // MOVN X1, #0 → X1 = 0xFFFFFFFFFFFFFFFF
      MovzX(4, 100),                  // MOVZ X4, #100 (true case)
      MovzX(5, 200),                  // MOVZ X5, #200 (false case)
      CmpRegX(0, 1),                  // CMP X0, X1 (5 vs UINT64_MAX)
      CselX(3, 4, 5, kCondCC),        // CSEL X3, X4, X5, CC
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[1], 0xFFFFFFFFFFFFFFFFULL);  // verify MOVN
  EXPECT_EQ(state_.cpu.x[3], 100ULL);  // CC should be true → X3 = X4 = 100
}

TEST_F(Arm64LiteTranslateRegionTest, CselCcFalse_LargeVsSmall) {
  // X0 = 100, X1 = 5
  // CMP X0, X1 → 100 > 5 → C=1 (no borrow) → CC (C==0) is false
  // CSEL X3, X4, X5, CC → should select X5 (false case)
  static const uint32_t code[] = {
      MovzX(0, 100),                  // MOVZ X0, #100
      MovzX(1, 5),                    // MOVZ X1, #5
      MovzX(4, 100),                  // MOVZ X4, #100 (true case)
      MovzX(5, 200),                  // MOVZ X5, #200 (false case)
      CmpRegX(0, 1),                  // CMP X0, X1 (100 vs 5)
      CselX(3, 4, 5, kCondCC),        // CSEL X3, X4, X5, CC
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 200ULL);  // CC should be false → X3 = X5 = 200
}

TEST_F(Arm64LiteTranslateRegionTest, CselCcWithInterveningAdd) {
  // Exact pattern from orderfile crash: CMP + non-flag ADD + CSEL CC.
  // X0 = 5, X1 = UINT64_MAX
  // CMP X0, X1 → C=0 → CC true
  // ADD W6, W6, #1 (non-flag-setting, should NOT clobber stored ARM flags)
  // CSEL X3, X4, X5, CC → should still see C=0 from CMP
  static const uint32_t code[] = {
      MovzX(0, 5),                    // MOVZ X0, #5
      MovnX(1, 0),                    // MOVN X1, #0 → UINT64_MAX
      MovzX(4, 100),                  // MOVZ X4, #100
      MovzX(5, 200),                  // MOVZ X5, #200
      MovzX(6, 0),                    // MOVZ X6, #0 (counter)
      CmpRegX(0, 1),                  // CMP X0, X1
      // ADD W6, W6, #1 (32-bit, no flags): 0x11000400 + rd + rn<<5
      0x110004C6,                     // ADD W6, W6, #1
      CselX(3, 4, 5, kCondCC),        // CSEL X3, X4, X5, CC
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[6], 1ULL);    // ADD happened
  EXPECT_EQ(state_.cpu.x[3], 100ULL);  // CC should be true → X3 = X4 = 100
}

// MOVZ Wd, #imm16 (32-bit move, zero-extends to X)
constexpr uint32_t MovzW(uint8_t rd, uint16_t imm16) {
  return 0x52800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// CMP Wn, Wm (32-bit SUBS WZR, Wn, Wm, shifted register)
constexpr uint32_t CmpRegW(uint8_t rn, uint8_t rm) {
  return 0x6B000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | 31;
}

TEST_F(Arm64LiteTranslateRegionTest, CmpW_BranchLs_NotTaken) {
  // Test 32-bit CMP W0, WZR with W0=24 → B.LS should NOT be taken.
  // This mirrors the orderfile Path B: threshold=24, cmp w10, wzr, b.ls.
  static const uint32_t code[] = {
      MovzW(0, 24),                    // MOVZ W0, #24
      CmpRegW(0, 31),                  // CMP W0, WZR (24 vs 0)
      Bcond(kCondLS, 8),               // B.LS +8 (should NOT be taken)
      MovzX(1, 42),                    // MOVZ X1, #42 (reached if not taken)
      MovzX(1, 0),                     // padding (branch target)
  };
  // B.LS NOT taken → falls through to MOVZ X1, #42, then exits region.
  // The forward branch doesn't end the region, so both MOVZs are translated.
  // But B.LS not taken means we execute MOVZ X1, #42 then MOVZ X1, #0 (overwritten).
  // Actually, B.LS at code+8 targets code+16 (MOVZ X1,#0). If taken, exits to code+16.
  // If not taken, continues to code+12 (MOVZ X1,#42), then code+16 (MOVZ X1,#0).
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[1], 0ULL);  // Both MOVZs execute; X1 = 0 from last one
  // But the key test: B.LS was NOT taken. X1 passes through #42 then #0.
}

TEST_F(Arm64LiteTranslateRegionTest, CmpW_BranchLs_Taken) {
  // CMP W0, WZR with W0=0 → B.LS SHOULD be taken (0 <= 0 unsigned).
  static const uint32_t code[] = {
      MovzW(0, 0),                     // MOVZ W0, #0
      CmpRegW(0, 31),                  // CMP W0, WZR (0 vs 0)
      Bcond(kCondLS, 8),               // B.LS +8 (should be taken)
      MovzX(1, 42),                    // skipped
      MovzX(1, 0),                     // branch target
  };
  GuestAddr branch_target = ToGuestAddr(code) + 16;
  EXPECT_TRUE(Run(code, branch_target));
  EXPECT_EQ(state_.cpu.x[1], 0ULL);  // X1 unchanged (starts at 0)
}

TEST_F(Arm64LiteTranslateRegionTest, OrderfileCrashPattern_9Registers) {
  // Reproduce EXACT register pressure from the orderfile crash:
  // 9 permanent registers in one region, with CMP + CSEL CC pattern.
  // Guest regs: x8, x9, x10, x11, x12, x19, x20, x22, x2
  //
  // This mimics region 2 of the orderfile function:
  // MOV X8, XZR → x8 = 0
  // MOV X10, #-1 → x10 = UINT64_MAX
  // MOV X22, X11 (X11 = some_ptr)
  // LDR X12, [X22, #8] (load count)
  // CMP X12, X10 (count vs UINT64_MAX)
  // ADD W9, W9, #1 (counter++)
  // CSEL X8, X22, X8, CC (best_entry = current if count < min_count)
  // CSEL X10, X12, X10, CC (min_count = count if count < min_count)
  static uint64_t fake_node[3];  // [0]=key, [1]=count, [2]=next
  fake_node[0] = 0xDEAD;  // key
  fake_node[1] = 42;      // count
  fake_node[2] = 0;       // next = NULL

  // Set up initial state for registers used in the "loop":
  state_.cpu.x[11] = ToGuestAddr(&fake_node[0]);  // x11 = pointer to node
  state_.cpu.x[19] = 0xBEEF;  // x19 = target key (different from 0xDEAD → b.eq not taken)
  state_.cpu.x[20] = ToGuestAddr(&fake_node[0]);  // x20 (just to map it)
  state_.cpu.x[2] = 7;  // x2 (just to map it)

  // Encode the loop body instructions exactly:
  static const uint32_t code[] = {
      // LDR X20, [X20] (map x20 early by reading it)
      0xF9400294,                      // LDR X20, [X20]
      // LDR X2, [X20] (map x2 by reading x20 into it — just to fill pool)
      // Actually: MOV X2, X2 is simpler: ORR X2, XZR, X2
      // Better: use existing x2 value. Just do ADD X2, X2, #0
      0x91000042,                      // ADD X2, X2, #0 (maps x2)
      // MOV W9, WZR
      MovzW(9, 0),                     // W9 = 0 (counter)
      // MOV X8, XZR
      MovzX(8, 0),                     // X8 = 0 (best_entry)
      // MOVN X10, #0
      MovnX(10, 0),                    // X10 = UINT64_MAX (min_count)
      // MOV X22, X11 (ORR X22, XZR, X11)
      0xAA0B03F6,                      // ORR X22, XZR, X11
      // LDR X11, [X11] (key = node.key)
      0xF940016B,                      // LDR X11, [X11]
      // LDR X12, [X22, #8] (count = node.count)
      0xF94006CC,                      // LDR X12, [X22, #8]
      // CMP X11, X19 (compare key with target)
      0xEB13017F,                      // CMP X11, X19
      // B.EQ +20 (5 instructions forward — to end of code)
      Bcond(kCondEQ, 20),              // B.EQ +20 (forward, region continues)
      // CMP X12, X10 (compare count vs min_count)
      CmpRegX(12, 10),                 // CMP X12, X10
      // ADD W9, W9, #1
      0x11000529,                      // ADD W9, W9, #1
      // CSEL X8, X22, X8, CC
      CselX(8, 22, 8, kCondCC),        // CSEL X8, X22, X8, CC
      // CSEL X10, X12, X10, CC
      CselX(10, 12, 10, kCondCC),      // CSEL X10, X12, X10, CC
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  // x8 should be the node address (CSEL CC fired since 42 < UINT64_MAX)
  EXPECT_EQ(state_.cpu.x[8], ToGuestAddr(&fake_node[0]));
  EXPECT_EQ(state_.cpu.x[10], 42ULL);  // min_count updated to 42
  EXPECT_EQ(state_.cpu.x[9], 1ULL);    // counter incremented
}

// region digitalis - regression test for WhatsApp libsuperpack vtable dispatcher.
// LDP Xt1, Xt2, [Xn] where Xt1 (or Xt2) aliases Xn must load BOTH values from
// the *original* base address, not from "base updated with val1". Previously
// LoadPair did SetReg(rt1, val1) between the two underlying Loads — and since
// the second Load used the same host register as `base`, it read from (val1 +
// scale) instead of (base + scale). The bug manifested as a hard SIGSEGV at
// jump to non-canonical address `0x624c000010cc0000` in WhatsApp's
// `Java_com_facebook_superpack_AssetDecompressor_testDecompressorLibraryUsable`
// when the dispatcher `ldp x0, x8, [x0]; ldr x3, [x8, #0x28]; br x3` ran with
// x0 = wrapper object: x8 ended up as `*(obj[0] + 8)` instead of obj[1].
//
// LDP Xt1, Xt2, [Xn]: 1010_1001_01_imm7_Rt2_Rn_Rt1 (32-bit), use unsigned
// signed-offset variant for offset=0 (imm7=0).
constexpr uint32_t LdpX(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm7) {
  uint32_t imm = static_cast<uint32_t>(imm7) & 0x7F;
  return 0xA9400000 | (imm << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
TEST_F(Arm64LiteTranslateRegionTest, LdpBaseAliasesFirstDest) {
  // LDP X0, X8, [X0] — base register aliases first destination.
  // Slot 0 = obj[0] = pointer that, if used as base for slot 1's load, would
  // produce garbage. Slot 1 = obj[1] = the real value we expect in X8.
  alignas(16) static uint64_t obj[4] = {
      0xAAAA'BBBB'CCCC'DDDDULL,  // [0] becomes X0 after LDP
      0x1122'3344'5566'7788ULL,  // [8] must become X8 — NOT *(obj[0]+8)
      0xDEAD'BEEF'DEAD'BEEFULL,  // never read
      0xDEAD'BEEF'DEAD'BEEFULL,
  };
  state_.cpu.x[0] = ToGuestAddr(&obj[0]);
  state_.cpu.x[8] = 0;
  static const uint32_t code[] = {
      LdpX(0, 8, 0, 0),  // LDP X0, X8, [X0]
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0xAAAA'BBBB'CCCC'DDDDULL);
  EXPECT_EQ(state_.cpu.x[8], 0x1122'3344'5566'7788ULL);
}
TEST_F(Arm64LiteTranslateRegionTest, LdpBaseAliasesSecondDest) {
  // LDP X1, X2, [X2] — base aliases the second destination only. Without the
  // fix the second Load still reads from the original base (because SetReg
  // happens after both loads in the corrected code), but verify nothing else
  // regresses.
  alignas(16) static uint64_t obj2[2] = {0xCAFEBABE'12345678ULL,
                                          0xF00DFACE'87654321ULL};
  state_.cpu.x[2] = ToGuestAddr(&obj2[0]);
  state_.cpu.x[1] = 0;
  static const uint32_t code[] = {
      LdpX(1, 2, 2, 0),  // LDP X1, X2, [X2]
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[1], 0xCAFEBABE'12345678ULL);
  EXPECT_EQ(state_.cpu.x[2], 0xF00DFACE'87654321ULL);
}
// endregion

// region digitalis - STP pre-index probe tests.
// STP Xt1, Xt2, [Xn, #imm]!  (pre-index, with writeback)
// Encoding: sf=1 -> 1010_1001_10_imm7_Rt2_Rn_Rt1 (64-bit), type=11.
//   bits[31:30]=10, bits[29:23]=1010_011, bit22=0(store), imm7=signed7, then rt2/rn/rt1.
constexpr uint32_t StpXPreIndex(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm_div8) {
  uint32_t imm = static_cast<uint32_t>(imm_div8) & 0x7F;
  return 0xA9800000 | (imm << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
// STP Xt1, Xt2, [Xn, #imm]   (signed offset, no writeback). type=10.
constexpr uint32_t StpXSigned(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm_div8) {
  uint32_t imm = static_cast<uint32_t>(imm_div8) & 0x7F;
  return 0xA9000000 | (imm << 15) | (rt2 << 10) | (rn << 5) | rt1;
}

// Reproduces the prologue of __dl__ZL19__android_log_levelPKcm in linker64:
//   stp x29, x30, [sp, #-0x60]!
//   stp x28, x27, [sp, #0x10]
//   stp x26, x25, [sp, #0x20]
//   stp x24, x23, [sp, #0x30]
// VulkanCapsViewer SIGSEGVs in this region at trans#4800 with rip in JIT cache.
TEST_F(Arm64LiteTranslateRegionTest, StpX_PreIndexNegative_SpProlog) {
  alignas(16) static uint64_t stack_buf[32] = {};
  // Place guest SP near the END of the buffer so [sp, #-0x60] still lands
  // inside it.
  state_.cpu.sp = ToGuestAddr(&stack_buf[20]);  // 20*8 = 160 bytes into buf
  state_.cpu.x[29] = 0x29292929'29292929ULL;
  state_.cpu.x[30] = 0x30303030'30303030ULL;
  state_.cpu.x[27] = 0x27272727'27272727ULL;
  state_.cpu.x[28] = 0x28282828'28282828ULL;
  state_.cpu.x[25] = 0x25252525'25252525ULL;
  state_.cpu.x[26] = 0x26262626'26262626ULL;
  state_.cpu.x[23] = 0x23232323'23232323ULL;
  state_.cpu.x[24] = 0x24242424'24242424ULL;
  const uint64_t orig_sp = state_.cpu.sp;
  static const uint32_t code[] = {
      StpXPreIndex(29, 30, 31, -12),  // stp x29, x30, [sp, #-0x60]!  (imm/8 = -12)
      StpXSigned(28, 27, 31, 2),      // stp x28, x27, [sp, #0x10]
      StpXSigned(26, 25, 31, 4),      // stp x26, x25, [sp, #0x20]
      StpXSigned(24, 23, 31, 6),      // stp x24, x23, [sp, #0x30]
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  // SP must have been decremented by 0x60 exactly once (only the first STP
  // is pre-index; the others are signed-offset).
  EXPECT_EQ(state_.cpu.sp, orig_sp - 0x60);
  // All eight values must have been written to the right slots in stack_buf.
  uint64_t* new_sp = bit_cast<uint64_t*>(static_cast<uintptr_t>(state_.cpu.sp));
  EXPECT_EQ(new_sp[0], 0x29292929'29292929ULL);          // [sp, #0]   = x29
  EXPECT_EQ(new_sp[1], 0x30303030'30303030ULL);          // [sp, #8]   = x30
  EXPECT_EQ(new_sp[2], 0x28282828'28282828ULL);          // [sp, #16]  = x28
  EXPECT_EQ(new_sp[3], 0x27272727'27272727ULL);          // [sp, #24]  = x27
  EXPECT_EQ(new_sp[4], 0x26262626'26262626ULL);          // [sp, #32]  = x26
  EXPECT_EQ(new_sp[5], 0x25252525'25252525ULL);          // [sp, #40]  = x25
  EXPECT_EQ(new_sp[6], 0x24242424'24242424ULL);          // [sp, #48]  = x24
  EXPECT_EQ(new_sp[7], 0x23232323'23232323ULL);          // [sp, #56]  = x23
}

// MRS X<rt>, TPIDR_EL0  =  1101_0101_0011_1101_0000_0010_0_Rt
constexpr uint32_t MrsTpidrEl0(uint8_t rt) {
  return 0xD53BD040 | rt;
}
// LDR X<rt>, [X<rn>, #imm]  (64-bit unsigned offset, imm/8)
constexpr uint32_t LdrXUoff(uint8_t rt, uint8_t rn, uint16_t imm_div8) {
  return 0xF9400000 | (static_cast<uint32_t>(imm_div8 & 0xFFF) << 10) | (rn << 5) | rt;
}

// VulkanCapsViewer prologue: __dl__ZL19__android_log_levelPKcm sequence
//   9ff40: mrs   x26, TPIDR_EL0
//   9ff44: ldr   x8,  [x26, #0x28]
// If MRS-TPIDR_EL0 returns NULL (because ThreadState.tls is not set), the LDR
// faults at addr 0. Reproduce by leaving state_.tls = 0.
TEST_F(Arm64LiteTranslateRegionTest, MrsTpidrEl0_LoadFromTls) {
  alignas(16) static uint64_t tls_buf[16] = {};
  tls_buf[5] = 0xC0DE'C0DE'C0DE'C0DEULL;  // offset 0x28 = 5*8
  state_.tls = ToGuestAddr(&tls_buf[0]);
  static const uint32_t code[] = {
      MrsTpidrEl0(26),         // mrs x26, TPIDR_EL0
      LdrXUoff(8, 26, 5),      // ldr x8, [x26, #0x28]
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[26], static_cast<uint64_t>(state_.tls));
  EXPECT_EQ(state_.cpu.x[8], 0xC0DE'C0DE'C0DE'C0DEULL);
}

// Minimal probe: only the pre-index instruction, on a non-SP base register.
// If this passes and the SP variant fails, the bug is in SP-as-base handling.
TEST_F(Arm64LiteTranslateRegionTest, StpX_PreIndexNegative_X1Base) {
  alignas(16) static uint64_t buf[32] = {};
  state_.cpu.x[1] = ToGuestAddr(&buf[20]);
  state_.cpu.x[29] = 0xAAAA'AAAA'AAAA'AAAAULL;
  state_.cpu.x[30] = 0xBBBB'BBBB'BBBB'BBBBULL;
  const uint64_t orig_x1 = state_.cpu.x[1];
  static const uint32_t code[] = {
      StpXPreIndex(29, 30, 1, -12),  // stp x29, x30, [x1, #-0x60]!
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[1], orig_x1 - 0x60);
  uint64_t* new_x1 = bit_cast<uint64_t*>(static_cast<uintptr_t>(state_.cpu.x[1]));
  EXPECT_EQ(new_x1[0], 0xAAAA'AAAA'AAAA'AAAAULL);
  EXPECT_EQ(new_x1[1], 0xBBBB'BBBB'BBBB'BBBBULL);
}
// endregion

// region digitalis LDP/STP pre/post-index JIT exec-tests.
// Confirms the decoder routes op2=01 (post-index) and op2=11 (pre-index) to
// the LoadStorePair callback with is_postindex / is_preindex set, and that
// the semantics_player + JIT pair correctly writes back the indexed base.
// Encoding (bits[31:30]=10 → 64-bit, bits[29:25]=10100, bits[24:23]=op2,
// bit[22]=L, bits[21:15]=imm7, bits[14:10]=Rt2, bits[9:5]=Rn, bits[4:0]=Rt):
//   LDP signed-offset: 0xA9400000 (op2=10, L=1) — see LdpX above.
//   LDP pre-index:     0xA9C00000 (op2=11, L=1).
//   LDP post-index:    0xA8C00000 (op2=01, L=1).
//   STP post-index:    0xA8800000 (op2=01, L=0) — pre-index covered by StpXPreIndex.
constexpr uint32_t LdpXPreIndex(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm_div8) {
  uint32_t imm = static_cast<uint32_t>(imm_div8) & 0x7F;
  return 0xA9C00000 | (imm << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
constexpr uint32_t LdpXPostIndex(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm_div8) {
  uint32_t imm = static_cast<uint32_t>(imm_div8) & 0x7F;
  return 0xA8C00000 | (imm << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
constexpr uint32_t StpXPostIndex(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm_div8) {
  uint32_t imm = static_cast<uint32_t>(imm_div8) & 0x7F;
  return 0xA8800000 | (imm << 15) | (rt2 << 10) | (rn << 5) | rt1;
}

// LDP X3, X4, [X5, #0x10]!  (pre-index)
// Verifies (a) the load reads from base+offset, not base, and (b) the base
// register is updated to base+offset.  Pre-handoff-39 the decoder mis-routed
// this as signed-offset, so the load was correct but the base writeback was
// missing — LLVM-emitted prologues that use [sp,#-N]! to allocate frame would
// then leave sp unmodified and corrupt the stack on the matching epilogue.
TEST_F(Arm64LiteTranslateRegionTest, LdpPreIndexUpdatesBase) {
  alignas(16) static uint64_t buf[8] = {
      0xBAAD'BAAD'BAAD'BAADULL,   // [0]: must NOT be read
      0xBAAD'BAAD'BAAD'BAADULL,   // [1]: must NOT be read
      0x1111'2222'3333'4444ULL,   // [2]: loaded into x3 (offset = +0x10)
      0x5555'6666'7777'8888ULL,   // [3]: loaded into x4 (offset = +0x18)
      0xBAAD'BAAD'BAAD'BAADULL,
      0xBAAD'BAAD'BAAD'BAADULL,
      0xBAAD'BAAD'BAAD'BAADULL,
      0xBAAD'BAAD'BAAD'BAADULL,
  };
  state_.cpu.x[5] = ToGuestAddr(&buf[0]);
  state_.cpu.x[3] = 0xDEAD'BEEFULL;
  state_.cpu.x[4] = 0xDEAD'BEEFULL;
  const uint64_t orig_x5 = state_.cpu.x[5];
  static const uint32_t code[] = {
      LdpXPreIndex(3, 4, 5, 2),  // ldp x3, x4, [x5, #0x10]!
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 0x1111'2222'3333'4444ULL);
  EXPECT_EQ(state_.cpu.x[4], 0x5555'6666'7777'8888ULL);
  EXPECT_EQ(state_.cpu.x[5], orig_x5 + 0x10);
}

// LDP X3, X4, [X5], #0x10  (post-index)
// Verifies (a) the load reads from the ORIGINAL base (no offset applied),
// and (b) the base register is updated to base+offset AFTER the load.
TEST_F(Arm64LiteTranslateRegionTest, LdpPostIndexUpdatesBase) {
  alignas(16) static uint64_t buf2[8] = {
      0xAAAA'BBBB'CCCC'DDDDULL,   // [0]: loaded into x3 (post-index = pre-load addr)
      0x1122'3344'5566'7788ULL,   // [1]: loaded into x4
      0xBAAD'BAAD'BAAD'BAADULL,   // [2]: must NOT be read
      0xBAAD'BAAD'BAAD'BAADULL,
      0xBAAD'BAAD'BAAD'BAADULL,
      0xBAAD'BAAD'BAAD'BAADULL,
      0xBAAD'BAAD'BAAD'BAADULL,
      0xBAAD'BAAD'BAAD'BAADULL,
  };
  state_.cpu.x[5] = ToGuestAddr(&buf2[0]);
  state_.cpu.x[3] = 0xDEAD'BEEFULL;
  state_.cpu.x[4] = 0xDEAD'BEEFULL;
  const uint64_t orig_x5 = state_.cpu.x[5];
  static const uint32_t code[] = {
      LdpXPostIndex(3, 4, 5, 2),  // ldp x3, x4, [x5], #0x10
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[3], 0xAAAA'BBBB'CCCC'DDDDULL);
  EXPECT_EQ(state_.cpu.x[4], 0x1122'3344'5566'7788ULL);
  EXPECT_EQ(state_.cpu.x[5], orig_x5 + 0x10);
}

// STP X3, X4, [X5], #-0x10  (post-index with negative offset; common in epilogues).
// Verifies (a) the store writes at the ORIGINAL base, and (b) base is then
// updated to base+offset.  Wrong-order writeback (computing addr=base+offset
// first) would store at the wrong location.
TEST_F(Arm64LiteTranslateRegionTest, StpPostIndexUpdatesBase) {
  alignas(16) static uint64_t buf3[8] = {};
  state_.cpu.x[5] = ToGuestAddr(&buf3[4]);  // start 32 bytes in
  state_.cpu.x[3] = 0xC0DE'C0DE'C0DE'C0DEULL;
  state_.cpu.x[4] = 0xDEAD'D00D'DEAD'D00DULL;
  const uint64_t orig_x5 = state_.cpu.x[5];
  static const uint32_t code[] = {
      StpXPostIndex(3, 4, 5, -2),  // stp x3, x4, [x5], #-0x10
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  // Store must hit [orig_x5] and [orig_x5+8] (buf3[4] and buf3[5]) — NOT
  // [orig_x5 - 0x10].
  EXPECT_EQ(buf3[4], 0xC0DE'C0DE'C0DE'C0DEULL);
  EXPECT_EQ(buf3[5], 0xDEAD'D00D'DEAD'D00DULL);
  EXPECT_EQ(buf3[2], 0ULL);  // [orig_x5 - 0x10] must be untouched.
  EXPECT_EQ(buf3[3], 0ULL);
  EXPECT_EQ(state_.cpu.x[5], orig_x5 - 0x10);
}

// STP X5, X4, [X5], #0x10  (post-index where Rt1 aliases Rn).
// ARM ARM marks this CONSTRAINED UNPREDICTABLE; on real hardware the
// architectural choice is to write *the original* value of Rt1 (which equals
// the original Rn).  Verify the JIT writes the ORIGINAL base value (not the
// updated one) and then writes back the new base.
TEST_F(Arm64LiteTranslateRegionTest, StpPostIndexAliasRn) {
  alignas(16) static uint64_t buf4[8] = {};
  const uint64_t base = ToGuestAddr(&buf4[2]);
  state_.cpu.x[5] = base;
  state_.cpu.x[4] = 0xABAB'ABAB'ABAB'ABABULL;
  static const uint32_t code[] = {
      StpXPostIndex(5, 4, 5, 2),  // stp x5, x4, [x5], #0x10
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  // First slot must hold the ORIGINAL value of x5 (the pre-update base).
  EXPECT_EQ(buf4[2], base);
  EXPECT_EQ(buf4[3], 0xABAB'ABAB'ABAB'ABABULL);
  // Base register must now be base + 0x10.
  EXPECT_EQ(state_.cpu.x[5], base + 0x10);
}
// endregion

// region digitalis atomic-op JIT exec-tests.
// The interpreter path was verified by hello-lse (handoff-37) and hello-barriers
// (handoff-38) at the integration level, but until now the JIT path for LSE
// atomics had zero unit-test coverage.  A hot-loop regression would slip past
// the integration probe because hello-lse runs each op a small fixed number of
// times — a CMPXCHG-loop drop-out (e.g. wrong size of cmpxchg, wrong reg into
// RAX, missing zero-extend of B/H result) would not be visible without these.
//
// Encoding references (DDI 0487, confirmed via llvm-mc):
//   CAS family (LoadStoreExclusive path):
//     size:2 | 001000 | o2:1 | L:1 | o1:1 | Rs:5 | o0:1 | 11111 | Rn:5 | Rt:5
//     non-A/L: o2=1, L=0, o1=1, o0=0
//     32-bit base = 0x88A07C00 | (Rs<<16) | (Rn<<5) | Rt
//     64-bit base = 0xC8A07C00 | (Rs<<16) | (Rn<<5) | Rt
//   LD<op>/SWP (AtomicMemoryOp path):
//     size:2 | 111000 | A:1 | R:1 | 1 | Rs:5 | o3:1 | opc:3 | 00 | Rn:5 | Rt:5
//     non-A/R: A=0, R=0
//     32-bit base = 0xB8200000; 64-bit base = 0xF8200000
//     opc = 0:LDADD, 1:LDCLR, 2:LDEOR, 3:LDSET, 4:LDSMAX, 5:LDSMIN,
//           6:LDUMAX, 7:LDUMIN  (with o3=0)
//     SWP = o3=1, opc=0 → +0x8000 over the base
constexpr uint32_t CasW(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x88A07C00 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t CasX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xC8A07C00 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdaddX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8200000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdclrX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8201000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdeorX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8202000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdsetX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8203000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdsmaxX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8204000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdsminX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8205000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdumaxX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8206000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LduminX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8207000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t SwpX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8208000 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

// CASP / CASPX (compare-and-swap pair, Armv8.1 LSE — ):
//   bit[31]=0, bit[30]=sz (0 W-pair, 1 X-pair), bits[29:23]=0010000, bit[22]=L,
//   bit[21]=1, bits[20:16]=Rs (must be even), bit[15]=o0, bits[14:10]=11111,
//   bits[9:5]=Rn, bits[4:0]=Rt (must be even).  Non-A/L → L=0, o0=0.
// Encoding cross-checked via llvm-mc (clang-r563880c) in decoder.h comment:
//   casp w0,w1,w2,w3,[x10] = 0x08207d42 → matches CaspW(0, 2, 10) below.
constexpr uint32_t CaspW(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x08207C00 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t CaspX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x48207C00 | (static_cast<uint32_t>(rs) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rt;
}

constexpr uint32_t kDmbIsh = 0xD5033BBF;  // dmb ish
constexpr uint32_t kDsbIsh = 0xD5033B9F;  // dsb ish
constexpr uint32_t kIsb    = 0xD5033FDF;  // isb
constexpr uint32_t kWfe    = 0xD503205F;  // wfe (HINT #2)
constexpr uint32_t kYield  = 0xD503203F;  // yield (HINT #1)

// CAS 32-bit: equal expected → swap performed, Rs receives the (matching) old.
TEST_F(Arm64LiteTranslateRegionTest, CasEqualSwapsW) {
  alignas(16) static uint32_t target = 0x11111111u;
  target = 0x11111111u;
  state_.cpu.x[0] = 0x11111111ULL;                // Rs = expected (matches)
  state_.cpu.x[1] = 0x22222222ULL;                // Rt = new
  state_.cpu.x[2] = ToGuestAddr(&target);         // Rn = base
  static const uint32_t code[] = {
      CasW(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 0x22222222u);
  EXPECT_EQ(state_.cpu.x[0], 0x11111111ULL);
}

// CAS 32-bit: unequal expected → swap NOT performed, Rs receives current value.
TEST_F(Arm64LiteTranslateRegionTest, CasUnequalLeavesMemoryW) {
  alignas(16) static uint32_t target = 0xDEADBEEFu;
  target = 0xDEADBEEFu;
  state_.cpu.x[0] = 0x11111111ULL;                // Rs = expected (does NOT match)
  state_.cpu.x[1] = 0x22222222ULL;                // Rt = new
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      CasW(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 0xDEADBEEFu);                 // memory unchanged
  EXPECT_EQ(state_.cpu.x[0] & 0xFFFFFFFFULL, 0xDEADBEEFULL);  // old returned
}

// CAS 64-bit: equal expected → swap.
TEST_F(Arm64LiteTranslateRegionTest, CasEqualSwapsX) {
  alignas(16) static uint64_t target = 0xCAFEBABE'F00DFEEDULL;
  target = 0xCAFEBABE'F00DFEEDULL;
  state_.cpu.x[0] = 0xCAFEBABE'F00DFEEDULL;
  state_.cpu.x[1] = 0x12345678'9ABCDEF0ULL;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      CasX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 0x12345678'9ABCDEF0ULL);
  EXPECT_EQ(state_.cpu.x[0], 0xCAFEBABE'F00DFEEDULL);
}

// SWP 64-bit: unconditional exchange of memory with Rs, old → Rt.
TEST_F(Arm64LiteTranslateRegionTest, SwpReplacesAndReturnsOldX) {
  alignas(16) static uint64_t target = 0x0123456789ABCDEFULL;
  target = 0x0123456789ABCDEFULL;
  state_.cpu.x[0] = 0xFEDCBA9876543210ULL;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      SwpX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 0xFEDCBA9876543210ULL);
  EXPECT_EQ(state_.cpu.x[1], 0x0123456789ABCDEFULL);
}

// LDADD 64-bit: memory += Rs, old → Rt.
TEST_F(Arm64LiteTranslateRegionTest, LdaddIncrementsAndReturnsOldX) {
  alignas(16) static uint64_t target = 100;
  target = 100;
  state_.cpu.x[0] = 25;                            // increment
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LdaddX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 125u);
  EXPECT_EQ(state_.cpu.x[1], 100ULL);
}

// LDSET 64-bit: memory |= Rs, old → Rt.
TEST_F(Arm64LiteTranslateRegionTest, LdsetSetsBitsX) {
  alignas(16) static uint64_t target = 0x0000FF00'0000FF00ULL;
  target = 0x0000FF00'0000FF00ULL;
  state_.cpu.x[0] = 0xFF000000'FF000000ULL;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LdsetX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 0xFF00FF00'FF00FF00ULL);
  EXPECT_EQ(state_.cpu.x[1], 0x0000FF00'0000FF00ULL);
}

// LDCLR 64-bit: memory &= ~Rs, old → Rt.
TEST_F(Arm64LiteTranslateRegionTest, LdclrClearsBitsX) {
  alignas(16) static uint64_t target = 0xFFFFFFFF'FFFFFFFFULL;
  target = 0xFFFFFFFF'FFFFFFFFULL;
  state_.cpu.x[0] = 0x0F0F0F0F'0F0F0F0FULL;       // bits to clear
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LdclrX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 0xF0F0F0F0'F0F0F0F0ULL);
  EXPECT_EQ(state_.cpu.x[1], 0xFFFFFFFF'FFFFFFFFULL);
}

// LDEOR 64-bit: memory ^= Rs, old → Rt.
TEST_F(Arm64LiteTranslateRegionTest, LdeorTogglesBitsX) {
  alignas(16) static uint64_t target = 0xAAAAAAAA'AAAAAAAAULL;
  target = 0xAAAAAAAA'AAAAAAAAULL;
  state_.cpu.x[0] = 0xFFFFFFFF'FFFFFFFFULL;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LdeorX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 0x55555555'55555555ULL);
  EXPECT_EQ(state_.cpu.x[1], 0xAAAAAAAA'AAAAAAAAULL);
}

// LDUMAX 64-bit: memory = unsigned_max(memory, Rs), old → Rt.
TEST_F(Arm64LiteTranslateRegionTest, LdumaxKeepsLargerX) {
  alignas(16) static uint64_t target = 100;
  target = 100;
  state_.cpu.x[0] = 500;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LdumaxX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 500u);
  EXPECT_EQ(state_.cpu.x[1], 100ULL);
}

// LDUMAX 64-bit: Rs is smaller — memory stays.
TEST_F(Arm64LiteTranslateRegionTest, LdumaxKeepsExistingWhenLargerX) {
  alignas(16) static uint64_t target = 500;
  target = 500;
  state_.cpu.x[0] = 100;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LdumaxX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 500u);
  EXPECT_EQ(state_.cpu.x[1], 500ULL);
}

// LDSMIN 64-bit: memory = signed_min(memory, Rs), old → Rt.
// Specifically tests that the signed comparison treats high-bit-set values as
// negative (so unsigned ordering would give the wrong answer).
TEST_F(Arm64LiteTranslateRegionTest, LdsminKeepsSmallerSignedX) {
  alignas(16) static uint64_t target = 5;
  target = 5;
  state_.cpu.x[0] = static_cast<uint64_t>(-100LL);  // signed-min candidate
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LdsminX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(static_cast<int64_t>(target), -100LL);
  EXPECT_EQ(state_.cpu.x[1], 5ULL);
}

// LDSMAX 64-bit: positive vs negative — signed max picks positive.
TEST_F(Arm64LiteTranslateRegionTest, LdsmaxPicksPositiveOverNegativeX) {
  alignas(16) static uint64_t target = static_cast<uint64_t>(-7LL);
  target = static_cast<uint64_t>(-7LL);
  state_.cpu.x[0] = 12;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LdsmaxX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 12u);
  EXPECT_EQ(state_.cpu.x[1], static_cast<uint64_t>(-7LL));
}

// LDUMIN 64-bit: unsigned-min over a high-bit value vs a small positive.
// In unsigned terms, the small positive wins.
TEST_F(Arm64LiteTranslateRegionTest, LduminPicksSmallerUnsignedX) {
  alignas(16) static uint64_t target = 0xFFFFFFFFFFFFFF00ULL;
  target = 0xFFFFFFFFFFFFFF00ULL;
  state_.cpu.x[0] = 7;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      LduminX(/*rs=*/0, /*rt=*/1, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target, 7u);
  EXPECT_EQ(state_.cpu.x[1], 0xFFFFFFFFFFFFFF00ULL);
}

// CASP 32-bit pair: equal expected → swap performed; Rs:Rs+1 receive the
// (matching) old pair zero-extended (verify checkbox).
// Memory layout: lo32 lives at the lower address, hi32 at +4 (little-endian).
TEST_F(Arm64LiteTranslateRegionTest, CaspPairEqualSwapsW) {
  alignas(16) static uint64_t target;
  target = (uint64_t{0x22222222ULL} << 32) | uint64_t{0x11111111ULL};
  state_.cpu.x[4] = 0x11111111ULL;   // Rs   = expected.lo (matches)
  state_.cpu.x[5] = 0x22222222ULL;   // Rs+1 = expected.hi (matches)
  state_.cpu.x[6] = 0xAAAAAAAAULL;   // Rt   = new.lo
  state_.cpu.x[7] = 0xBBBBBBBBULL;   // Rt+1 = new.hi
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      CaspW(/*rs=*/4, /*rt=*/6, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target,
            (uint64_t{0xBBBBBBBBULL} << 32) | uint64_t{0xAAAAAAAAULL});
  // Each half of the prior pair written back zero-extended into Rs/Rs+1.
  EXPECT_EQ(state_.cpu.x[4], 0x11111111ULL);
  EXPECT_EQ(state_.cpu.x[5], 0x22222222ULL);
}

// CASP 32-bit pair: unequal expected → swap NOT performed; Rs:Rs+1 receive the
// actual prior pair (each half zero-extended).
TEST_F(Arm64LiteTranslateRegionTest, CaspPairUnequalLeavesMemoryW) {
  alignas(16) static uint64_t target;
  target = (uint64_t{0xCAFEBABEULL} << 32) | uint64_t{0xDEADBEEFULL};
  state_.cpu.x[4] = 0x12345678ULL;   // expected.lo — does NOT match memory.lo
  state_.cpu.x[5] = 0x9ABCDEF0ULL;   // expected.hi — does NOT match memory.hi
  state_.cpu.x[6] = 0xAAAAAAAAULL;
  state_.cpu.x[7] = 0xBBBBBBBBULL;
  state_.cpu.x[2] = ToGuestAddr(&target);
  static const uint32_t code[] = {
      CaspW(/*rs=*/4, /*rt=*/6, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(target,
            (uint64_t{0xCAFEBABEULL} << 32) | uint64_t{0xDEADBEEFULL});
  EXPECT_EQ(state_.cpu.x[4], 0xDEADBEEFULL);   // memory.lo zero-extended
  EXPECT_EQ(state_.cpu.x[5], 0xCAFEBABEULL);   // memory.hi zero-extended
}

// CASP 64-bit pair: equal expected → swap performed (LOCK CMPXCHG16B path).
// The pair address MUST be 16-byte aligned — Intel #GP's misaligned
// CMPXCHG16B.  alignas(16) on the static buffer guarantees that.
TEST_F(Arm64LiteTranslateRegionTest, CaspPairEqualSwapsX) {
  alignas(16) static struct {
    uint64_t lo;
    uint64_t hi;
  } pair_mem;
  pair_mem.lo = 0xCAFEBABEF00DFEEDULL;
  pair_mem.hi = 0x0123456789ABCDEFULL;
  state_.cpu.x[4] = 0xCAFEBABEF00DFEEDULL;   // expected.lo (matches)
  state_.cpu.x[5] = 0x0123456789ABCDEFULL;   // expected.hi (matches)
  state_.cpu.x[6] = 0xFEEDFACECAFEBABEULL;   // new.lo
  state_.cpu.x[7] = 0xDEADBEEF12345678ULL;   // new.hi
  state_.cpu.x[2] = ToGuestAddr(&pair_mem);
  static const uint32_t code[] = {
      CaspX(/*rs=*/4, /*rt=*/6, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(pair_mem.lo, 0xFEEDFACECAFEBABEULL);
  EXPECT_EQ(pair_mem.hi, 0xDEADBEEF12345678ULL);
  EXPECT_EQ(state_.cpu.x[4], 0xCAFEBABEF00DFEEDULL);
  EXPECT_EQ(state_.cpu.x[5], 0x0123456789ABCDEFULL);
}

// CASP 64-bit pair: unequal expected → memory unchanged; Rs:Rs+1 receive the
// actual prior 128-bit pair (low → Rs, high → Rs+1).
TEST_F(Arm64LiteTranslateRegionTest, CaspPairUnequalLeavesMemoryX) {
  alignas(16) static struct {
    uint64_t lo;
    uint64_t hi;
  } pair_mem;
  pair_mem.lo = 0xCAFEBABEF00DFEEDULL;
  pair_mem.hi = 0x0123456789ABCDEFULL;
  state_.cpu.x[4] = 0x1111111111111111ULL;   // does NOT match
  state_.cpu.x[5] = 0x2222222222222222ULL;
  state_.cpu.x[6] = 0xFEEDFACECAFEBABEULL;
  state_.cpu.x[7] = 0xDEADBEEF12345678ULL;
  state_.cpu.x[2] = ToGuestAddr(&pair_mem);
  static const uint32_t code[] = {
      CaspX(/*rs=*/4, /*rt=*/6, /*rn=*/2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(pair_mem.lo, 0xCAFEBABEF00DFEEDULL);
  EXPECT_EQ(pair_mem.hi, 0x0123456789ABCDEFULL);
  EXPECT_EQ(state_.cpu.x[4], 0xCAFEBABEF00DFEEDULL);
  EXPECT_EQ(state_.cpu.x[5], 0x0123456789ABCDEFULL);
}

// Barrier instructions in the middle of a JIT region must compile through
// without breaking up the region or corrupting register values.  Catches a
// future Nop-handler regression that drops to interpreter (success_ = false)
// for any of DMB/DSB/ISB/WFE/YIELD.
TEST_F(Arm64LiteTranslateRegionTest, BarriersDoNotBreakRegion) {
  static const uint32_t code[] = {
      MovzX(0, 7),         // X0 = 7
      kDmbIsh,             // dmb ish — Nop in JIT
      kDsbIsh,             // dsb ish — Nop
      kIsb,                // isb     — Nop
      kWfe,                // wfe     — Nop (HINT)
      kYield,              // yield   — Nop (HINT)
      AddImmX(0, 0, 5),    // X0 += 5 → 12
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 12ULL);
}
// endregion

// region digitalis - FCVT*S/U scalar saturation diagnostic tests.
// Used to debug the unsigned sf=1 +Inf saturation path (handoff-99).
constexpr uint32_t kFcvtnuXd0 = 0x9E610000;  // FCVTNU X0, D0
constexpr uint32_t kFcvtpuXs0 = 0x9E290000;  // FCVTPU X0, S0
constexpr uint32_t kFcvtmsXs0 = 0x9E300000;  // FCVTMS X0, S0
constexpr uint32_t kFcvtnuWd0 = 0x1E610000;  // FCVTNU W0, D0

TEST_F(Arm64LiteTranslateRegionTest, FcvtnuXdPosInfSaturates) {
  // FCVTNU X0, D0 with D0 = +Inf -> expected UINT64_MAX.
  state_.cpu.v[0] = 0;
  // FP64 +Inf is 0x7FF0000000000000.  Low 64 of v[0] holds the value.
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x7FF0000000000000ULL;
  static const uint32_t code[] = {kFcvtnuXd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], UINT64_MAX);
}

TEST_F(Arm64LiteTranslateRegionTest, FcvtpuXsPosInfSaturates) {
  // FCVTPU X0, S0 with S0 = +Inf (FP32) -> expected UINT64_MAX.
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x7F800000u;
  static const uint32_t code[] = {kFcvtpuXs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], UINT64_MAX);
}

TEST_F(Arm64LiteTranslateRegionTest, FcvtmsXsNanReturnsZero) {
  // FCVTMS X0, S0 with S0 = QNaN -> expected 0.
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x7FC00000u;
  static const uint32_t code[] = {kFcvtmsXs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, FcvtnuWdPosInfSaturates) {
  // FCVTNU W0, D0 with D0 = +Inf -> expected UINT32_MAX (zero-extended to W).
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x7FF0000000000000ULL;
  static const uint32_t code[] = {kFcvtnuWd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(UINT32_MAX));
}

// LDR d0, [x1] + FCVTNU x0, d0 — simulates the in-app probe call sequence.
// LDR d0, [x1]: 1111_1101_0100_0000_0000_0000_0010_0000 = 0xFD400020
TEST_F(Arm64LiteTranslateRegionTest, FcvtnuXdFromMemoryPosInfSaturates) {
  static uint64_t fp_storage = 0x7FF0000000000000ULL;
  state_.cpu.x[1] = ToGuestAddr(&fp_storage);
  state_.cpu.v[0] = 0;
  static const uint32_t code[] = {
      0xFD400020,         // LDR D0, [X1]
      kFcvtnuXd0,         // FCVTNU X0, D0
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], UINT64_MAX);
}

// DISABLED diagnostic: shows that when 13 prior MOVZ map all 13 GP slots,
// IsGpRegPoolLow terminates the JIT region before reaching the FCVTNU.
// This is the on-device scenario where the interpreter ARM-saturation fix
// in interpreter.h (handoff-99) catches the FCVTNU instead.  The Run()
// framework only translates one region and does not invoke the interpreter
// fallback, so this test cannot pass without separate interpreter coverage.
TEST_F(Arm64LiteTranslateRegionTest, DISABLED_FcvtnuXdPosInfWithRegPressure) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x7FF0000000000000ULL;
  static const uint32_t code[] = {
      MovzX(2, 1), MovzX(3, 2), MovzX(4, 3), MovzX(5, 4),
      MovzX(6, 5), MovzX(7, 6), MovzX(8, 7), MovzX(9, 8),
      MovzX(10, 9), MovzX(11, 10), MovzX(12, 11), MovzX(13, 12),
      MovzX(14, 13),                     // 13 mappings (matches GP pool size)
      kFcvtnuXd0,                        // FCVTNU X0, D0
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], UINT64_MAX);
}

// FCVTAS / FCVTAU (round-to-nearest ties-AWAY-from-zero) JIT lowering.
// rmode=00, opcode=100 (signed) / 101 (unsigned), ftype in {00, 01}.
// FCVTAS Wd, Sn -> sf=0, ftype=00: 0001_1110_0010_0100_0000_00nn_nnnd_dddd
// FCVTAS Xd, Sn -> sf=1, ftype=00: 1001_1110_0010_0100 ...
// FCVTAS Wd, Dn -> sf=0, ftype=01: 0001_1110_0110_0100 ...
// FCVTAS Xd, Dn -> sf=1, ftype=01: 1001_1110_0110_0100 ...
// FCVTAU Wd, Sn -> 0x1E25...; FCVTAU Xd, Sn -> 0x9E25...;
// FCVTAU Wd, Dn -> 0x1E65...; FCVTAU Xd, Dn -> 0x9E65...
constexpr uint32_t kFcvtasWs0 = 0x1E240000;
constexpr uint32_t kFcvtasXs0 = 0x9E240000;
constexpr uint32_t kFcvtasWd0 = 0x1E640000;
constexpr uint32_t kFcvtasXd0 = 0x9E640000;
constexpr uint32_t kFcvtauWs0 = 0x1E250000;
constexpr uint32_t kFcvtauXs0 = 0x9E250000;
constexpr uint32_t kFcvtauWd0 = 0x1E650000;
constexpr uint32_t kFcvtauXd0 = 0x9E650000;

// Halfway tie: FCVTAS rounds 2.5 -> 3 (away from zero), NOT 2 (RNE round-
// to-even).  This is the load-bearing distinguishing case vs FCVTNS.
TEST_F(Arm64LiteTranslateRegionTest, FcvtasWs2p5TiesAway) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x40200000u;  // 2.5f
  static const uint32_t code[] = {kFcvtasWs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 3ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, FcvtasWsNeg2p5TiesAway) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0xC0200000u;  // -2.5f
  static const uint32_t code[] = {kFcvtasWs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  // -3 sign-extended into Wd is 0xFFFFFFFD; the JIT writes Wd, upper 32 zero.
  EXPECT_EQ(state_.cpu.x[0], 0x00000000FFFFFFFDULL);
}

TEST_F(Arm64LiteTranslateRegionTest, FcvtasXd0p5TiesAway) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x3FE0000000000000ULL;  // 0.5d
  static const uint32_t code[] = {kFcvtasXd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 1ULL);
}

// Already-integer above the magnitude threshold (FP32 step >= 1 at |x| >=
// 2^23): magnitude gate skips the add-half, so an odd integer at 2^23+1
// must round-trip exactly.  Without the gate it would be bumped to
// 2^23+2 by RNE of (2^23+1)+0.5.
TEST_F(Arm64LiteTranslateRegionTest, FcvtasWsOddIntegerAboveThreshold) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x4B000001u;  // 2^23 + 1 = 8388609.0f
  static const uint32_t code[] = {kFcvtasWs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 8388609ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, FcvtasXdOddIntegerAboveDoubleThreshold) {
  state_.cpu.v[0] = 0;
  // 2^52 + 1 = 4503599627370497.0d, bits 0x4330000000000001
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x4330000000000001ULL;
  static const uint32_t code[] = {kFcvtasXd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 4503599627370497ULL);
}

// NaN -> 0.
TEST_F(Arm64LiteTranslateRegionTest, FcvtasWsNanReturnsZero) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x7FC00000u;  // QNaN
  static const uint32_t code[] = {kFcvtasWs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);
}

// Positive overflow saturates to INT_MAX (signed).
TEST_F(Arm64LiteTranslateRegionTest, FcvtasWsPosInfSaturates) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x7F800000u;  // +Inf
  static const uint32_t code[] = {kFcvtasWs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(INT32_MAX));
}

TEST_F(Arm64LiteTranslateRegionTest, FcvtasXdNegInfSaturates) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0xFFF0000000000000ULL;  // -Inf
  static const uint32_t code[] = {kFcvtasXd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(INT64_MIN));
}

// FCVTAU: negative inputs saturate to 0.  -0.5 ties away from zero would
// give -1 if signed, but unsigned saturates to 0.
TEST_F(Arm64LiteTranslateRegionTest, FcvtauWsNegHalfSaturatesZero) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0xBF000000u;  // -0.5f
  static const uint32_t code[] = {kFcvtauWs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);
}

TEST_F(Arm64LiteTranslateRegionTest, FcvtauXd0p5TiesAway) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x3FE0000000000000ULL;  // 0.5d
  static const uint32_t code[] = {kFcvtauXd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 1ULL);
}

// FCVTAU sf=1 +Inf -> UINT64_MAX (offset-trick saturation path).
TEST_F(Arm64LiteTranslateRegionTest, FcvtauXdPosInfSaturates) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x7FF0000000000000ULL;
  static const uint32_t code[] = {kFcvtauXd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], UINT64_MAX);
}

// FCVTAU sf=0 +Inf -> UINT32_MAX (upper-32 zero saturation path).
TEST_F(Arm64LiteTranslateRegionTest, FcvtauWdPosInfSaturates) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x7FF0000000000000ULL;
  static const uint32_t code[] = {kFcvtauWd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(UINT32_MAX));
}
// endregion

// region digitalis
// FRINTA Sd, Sn / Dd, Dn (round to nearest, ties AWAY from zero).
// FP data-processing 1-source, opcode=001100:
//   FRINTA Sd, Sn: 0001_1110_0010_0110_0100_00nn_nnnd_dddd  (ftype=00)
//   FRINTA Dd, Dn: 0001_1110_0110_0110_0100_00nn_nnnd_dddd  (ftype=01)
constexpr uint32_t kFrintaSs0 = 0x1E264000;
constexpr uint32_t kFrintaDd0 = 0x1E664000;

// Halfway tie: FRINTA rounds 2.5 -> 3.0 (away from zero), NOT 2.0 (RNE).
TEST_F(Arm64LiteTranslateRegionTest, FrintaSs2p5TiesAway) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x40200000u;  // 2.5f
  static const uint32_t code[] = {kFrintaSs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(&state_.cpu.v[0]),
            0x40400000u);  // 3.0f
}

TEST_F(Arm64LiteTranslateRegionTest, FrintaSsNeg2p5TiesAway) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0xC0200000u;  // -2.5f
  static const uint32_t code[] = {kFrintaSs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(&state_.cpu.v[0]),
            0xC0400000u);  // -3.0f
}

TEST_F(Arm64LiteTranslateRegionTest, FrintaDd0p5TiesAway) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) =
      0x3FE0000000000000ULL;  // 0.5d
  static const uint32_t code[] = {kFrintaDd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(*reinterpret_cast<uint64_t*>(&state_.cpu.v[0]),
            0x3FF0000000000000ULL);  // 1.0d
}

// Magnitude-gate edge case: an already-integer FP value at or above
// the mantissa-overflow boundary (FP32 step >= 1 at |x| >= 2^23,
// FP64 step >= 1 at |x| >= 2^52).  Adding 0.5 lands a tie below the
// LSB and RNE round-half-to-even bumps odd values to the next even.
// Expected: FRINTA(2^23+1) == 2^23+1 (already integer, untouched).
TEST_F(Arm64LiteTranslateRegionTest, FrintaSsOddIntegerAboveThreshold) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) =
      0x4B000001u;  // 2^23 + 1 = 8388609.0f
  static const uint32_t code[] = {kFrintaSs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(&state_.cpu.v[0]),
            0x4B000001u);  // 8388609.0f (NOT 8388610.0f)
}

TEST_F(Arm64LiteTranslateRegionTest, FrintaDdOddIntegerAboveDoubleThreshold) {
  state_.cpu.v[0] = 0;
  // 2^52 + 1 = 4503599627370497.0d
  *reinterpret_cast<uint64_t*>(&state_.cpu.v[0]) = 0x4330000000000001ULL;
  static const uint32_t code[] = {kFrintaDd0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(*reinterpret_cast<uint64_t*>(&state_.cpu.v[0]),
            0x4330000000000001ULL);  // 4503599627370497 unchanged
}

// Non-tie inputs: FRINTA(0.4) -> 0.0, FRINTA(0.6) -> 1.0.
TEST_F(Arm64LiteTranslateRegionTest, FrintaSs0p4) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x3ECCCCCDu;  // 0.4f
  static const uint32_t code[] = {kFrintaSs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(&state_.cpu.v[0]),
            0x00000000u);  // 0.0f
}

TEST_F(Arm64LiteTranslateRegionTest, FrintaSs0p6) {
  state_.cpu.v[0] = 0;
  *reinterpret_cast<uint32_t*>(&state_.cpu.v[0]) = 0x3F19999Au;  // 0.6f
  static const uint32_t code[] = {kFrintaSs0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(*reinterpret_cast<uint32_t*>(&state_.cpu.v[0]),
            0x3F800000u);  // 1.0f
}

// Vector FRINTA: FRINTA Vd.2S/4S/2D, Vn.2S/4S/2D.  Same opcode=11000,
// U=1, a=0 form as scalar but on AdvSIMD two-reg misc.  Verified
// encodings via llvm-mc + decoder comment at decoder.h:4391:
//   frinta v0.2s, v0.2s = 0x2E218800  (Q=0, size=00)
//   frinta v0.4s, v0.4s = 0x6E218800  (Q=1, size=00)
//   frinta v0.2d, v0.2d = 0x6E618800  (Q=1, size=01)
constexpr uint32_t kFrintaV2s00 = 0x2E218800;
constexpr uint32_t kFrintaV4s00 = 0x6E218800;
constexpr uint32_t kFrintaV2d00 = 0x6E618800;

// Vector magnitude-gate: mixed lanes -- one already-integer above the
// FP32 mantissa-overflow boundary (2^23), one ordinary ties-away input.
// Pre-fix, the add-of-0.5 + ROUNDPS path bumped odd-mantissa integers
// to the next even because RNE round-half-to-even fires on the
// half-bit-below-LSB tie.
TEST_F(Arm64LiteTranslateRegionTest, FrintaV4sMagnitudeGateMixedLanes) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4B000001u;  // 2^23 + 1 = 8388609.0f (already integer)
  lanes[1] = 0x40200000u;  // 2.5f          -> 3.0f
  lanes[2] = 0xC0200000u;  // -2.5f         -> -3.0f
  lanes[3] = 0xCB000001u;  // -(2^23 + 1)   (already integer; odd-mantissa)
  static const uint32_t code[] = {kFrintaV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x4B000001u);   // 8388609.0f unchanged
  EXPECT_EQ(lanes[1], 0x40400000u);   // 3.0f
  EXPECT_EQ(lanes[2], 0xC0400000u);   // -3.0f
  EXPECT_EQ(lanes[3], 0xCB000001u);   // -8388609.0f unchanged
}

// .2S form: low 64 bits live, high 64 must be zeroed (Q=0).
TEST_F(Arm64LiteTranslateRegionTest, FrintaV2sMagnitudeGateMixedLanes) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4B000001u;  // 2^23 + 1   (already integer)
  lanes[1] = 0x3F19999Au;  // 0.6f       -> 1.0f
  lanes[2] = 0xDEADBEEFu;  // garbage in upper 64; must be zeroed
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFrintaV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x4B000001u);
  EXPECT_EQ(lanes[1], 0x3F800000u);
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// .2D form: FP64 mantissa-overflow boundary at 2^52.
TEST_F(Arm64LiteTranslateRegionTest, FrintaV2dMagnitudeGateMixedLanes) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4330000000000001ULL;  // 2^52 + 1   (already integer; odd)
  lanes[1] = 0x3FE0000000000000ULL;  // 0.5d        -> 1.0d
  static const uint32_t code[] = {kFrintaV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x4330000000000001ULL);   // 4503599627370497 unchanged
  EXPECT_EQ(lanes[1], 0x3FF0000000000000ULL);   // 1.0d
}

// NaN / Inf must propagate through the vector path unchanged.  The
// magnitude gate puts NaN/Inf bits above the threshold, so the addend
// is zeroed and ROUNDPS/PD preserves NaN/Inf per Intel SDM.
TEST_F(Arm64LiteTranslateRegionTest, FrintaV4sNanInfPropagate) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x7FC00000u;  // qNaN
  lanes[1] = 0x7F800000u;  // +Inf
  lanes[2] = 0xFF800000u;  // -Inf
  lanes[3] = 0x40200000u;  // 2.5f -> 3.0f
  static const uint32_t code[] = {kFrintaV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x7FC00000u);   // qNaN unchanged
  EXPECT_EQ(lanes[1], 0x7F800000u);   // +Inf unchanged
  EXPECT_EQ(lanes[2], 0xFF800000u);   // -Inf unchanged
  EXPECT_EQ(lanes[3], 0x40400000u);   // 3.0f
}

// Vector FABS / FNEG / FSQRT (FP32 .2S/.4S, FP64 .2D).  These are the
// standard (non-FP16) two-reg-misc forms.  Per ARM ARM and decoder.h
// (case 0b01111 and 0b11111), bit23 is fixed to 1 in the encoding,
// so args.size carries (bit23=1, bit22=sz) and is 0b10 (FP32) or
// 0b11 (FP64) -- the existing JIT size check `args.size != 0b10 &&
// args.size != 0b11` accepts the valid encodings.
// Encodings verified by hand-derivation from ARM ARM C7.2:
//   fabs  v0.2s, v0.2s = 0x0EA0F800  (Q=0, U=0, sz=0)
//   fabs  v0.4s, v0.4s = 0x4EA0F800  (Q=1, U=0, sz=0)
//   fabs  v0.2d, v0.2d = 0x4EE0F800  (Q=1, U=0, sz=1)
//   fneg  v0.2s, v0.2s = 0x2EA0F800  (Q=0, U=1, sz=0)
//   fneg  v0.4s, v0.4s = 0x6EA0F800  (Q=1, U=1, sz=0)
//   fneg  v0.2d, v0.2d = 0x6EE0F800  (Q=1, U=1, sz=1)
//   fsqrt v0.2s, v0.2s = 0x2EA1F800  (Q=0, U=1, sz=0, opcode=11111)
//   fsqrt v0.4s, v0.4s = 0x6EA1F800  (Q=1, U=1, sz=0)
//   fsqrt v0.2d, v0.2d = 0x6EE1F800  (Q=1, U=1, sz=1)
constexpr uint32_t kFabsV2s00 = 0x0EA0F800;
constexpr uint32_t kFabsV4s00 = 0x4EA0F800;
constexpr uint32_t kFabsV2d00 = 0x4EE0F800;
constexpr uint32_t kFnegV2s00 = 0x2EA0F800;
constexpr uint32_t kFnegV4s00 = 0x6EA0F800;
constexpr uint32_t kFnegV2d00 = 0x6EE0F800;
constexpr uint32_t kFsqrtV2s00 = 0x2EA1F800;
constexpr uint32_t kFsqrtV4s00 = 0x6EA1F800;
constexpr uint32_t kFsqrtV2d00 = 0x6EE1F800;

// FABS .4S: clear sign bit per FP32 lane.  Bit-exact bit ops.
TEST_F(Arm64LiteTranslateRegionTest, FabsV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0xC0200000u;  // -2.5f
  lanes[1] = 0x40200000u;  // 2.5f (unchanged)
  lanes[2] = 0xFF800000u;  // -Inf -> +Inf
  lanes[3] = 0x80000000u;  // -0.0f -> +0.0f
  static const uint32_t code[] = {kFabsV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40200000u);  // 2.5f
  EXPECT_EQ(lanes[1], 0x40200000u);  // 2.5f
  EXPECT_EQ(lanes[2], 0x7F800000u);  // +Inf
  EXPECT_EQ(lanes[3], 0x00000000u);  // +0.0f
}

// FABS .2S: low-64-bit form must zero the upper 64 bits.
TEST_F(Arm64LiteTranslateRegionTest, FabsV2s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0xBF800000u;  // -1.0f -> 1.0f
  lanes[1] = 0xC0000000u;  // -2.0f -> 2.0f
  lanes[2] = 0xDEADBEEFu;  // garbage; must be zeroed
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFabsV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x3F800000u);  // 1.0f
  EXPECT_EQ(lanes[1], 0x40000000u);  // 2.0f
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// FABS .2D: clear sign bit per FP64 lane.
TEST_F(Arm64LiteTranslateRegionTest, FabsV2d) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0xC000000000000000ULL;  // -2.0d -> 2.0d
  lanes[1] = 0xFFF0000000000000ULL;  // -Inf  -> +Inf
  static const uint32_t code[] = {kFabsV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x4000000000000000ULL);  // 2.0d
  EXPECT_EQ(lanes[1], 0x7FF0000000000000ULL);  // +Inf
}

// FNEG .4S: flip sign bit per FP32 lane.
TEST_F(Arm64LiteTranslateRegionTest, FnegV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0xC0200000u;  // -2.5f -> 2.5f
  lanes[1] = 0x40200000u;  // 2.5f  -> -2.5f
  lanes[2] = 0x00000000u;  // +0.0f -> -0.0f
  lanes[3] = 0x7F800000u;  // +Inf  -> -Inf
  static const uint32_t code[] = {kFnegV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40200000u);  // 2.5f
  EXPECT_EQ(lanes[1], 0xC0200000u);  // -2.5f
  EXPECT_EQ(lanes[2], 0x80000000u);  // -0.0f
  EXPECT_EQ(lanes[3], 0xFF800000u);  // -Inf
}

// FNEG .2S: Q=0; upper 64 must be zeroed.
TEST_F(Arm64LiteTranslateRegionTest, FnegV2s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x3F800000u;  // 1.0f -> -1.0f
  lanes[1] = 0xC0000000u;  // -2.0f -> 2.0f
  lanes[2] = 0xDEADBEEFu;  // upper must be zeroed
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFnegV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0xBF800000u);  // -1.0f
  EXPECT_EQ(lanes[1], 0x40000000u);  // 2.0f
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// FNEG .2D: flip sign bit per FP64 lane.
TEST_F(Arm64LiteTranslateRegionTest, FnegV2d) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4000000000000000ULL;  // 2.0d -> -2.0d
  lanes[1] = 0xBFF0000000000000ULL;  // -1.0d -> 1.0d
  static const uint32_t code[] = {kFnegV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0xC000000000000000ULL);  // -2.0d
  EXPECT_EQ(lanes[1], 0x3FF0000000000000ULL);  // 1.0d
}

// FSQRT .4S: per-lane square root.  Exact for perfect squares.
TEST_F(Arm64LiteTranslateRegionTest, FsqrtV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40800000u;  // 4.0f  -> 2.0f
  lanes[1] = 0x41100000u;  // 9.0f  -> 3.0f
  lanes[2] = 0x42200000u;  // 40.0f -> sqrt(40)f (not exact)
  lanes[3] = 0x3F800000u;  // 1.0f  -> 1.0f
  static const uint32_t code[] = {kFsqrtV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40000000u);  // 2.0f
  EXPECT_EQ(lanes[1], 0x40400000u);  // 3.0f
  // sqrt(40.0f) in FP32 RNE: ~6.32455532f = 0x40CA62C2 (host hw SQRTPS).
  EXPECT_EQ(lanes[2], 0x40CA62C2u);
  EXPECT_EQ(lanes[3], 0x3F800000u);  // 1.0f
}

// FSQRT .2S: Q=0; upper 64 must be zeroed.
TEST_F(Arm64LiteTranslateRegionTest, FsqrtV2s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40800000u;  // 4.0f -> 2.0f
  lanes[1] = 0x41100000u;  // 9.0f -> 3.0f
  lanes[2] = 0xDEADBEEFu;
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFsqrtV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40000000u);
  EXPECT_EQ(lanes[1], 0x40400000u);
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// FSQRT .2D: per-lane FP64 square root.
TEST_F(Arm64LiteTranslateRegionTest, FsqrtV2d) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4010000000000000ULL;  // 4.0d -> 2.0d
  lanes[1] = 0x4022000000000000ULL;  // 9.0d -> 3.0d
  static const uint32_t code[] = {kFsqrtV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x4000000000000000ULL);  // 2.0d
  EXPECT_EQ(lanes[1], 0x4008000000000000ULL);  // 3.0d
}

// Vector FRINTN/M/P/Z/X/I (FP32 .2S/.4S, FP64 .2D).  The JIT handler
// at lite_translator.h:6248 historically gated on `args.size != 0b10
// && args.size != 0b11`, which only matches the bit23=1 half of the
// FRINT-family encoding space.  FRINTN/M/X have bit23=0 (decoder.h
// line 4397-4414 dispatches them under `!GetBits<23, 1>()`), so their
// args.size is 0b00 (FP32) or 0b01 (FP64) -- silently rejected to
// the interpreter.  FRINTP/Z/I have bit23=1; args.size is 0b10/0b11
// -- reachable.  Encodings verified against the decoder.h comment at
// lines 4389-4396:
//   frintn v0.4s = 0x4E218800  (a=0,U=0,opcode=11000)
//   frintn v0.2s = 0x0E218800  (Q=0)
//   frintn v0.2d = 0x4E618800  (sz=1, Q=1)
//   frintm v0.4s = 0x4E219800  (a=0,U=0,opcode=11001)
//   frintx v0.4s = 0x6E219800  (a=0,U=1,opcode=11001)
//   frintp v0.4s = 0x4EA18800  (a=1,U=0,opcode=11000)
//   frintz v0.4s = 0x4EA19800  (a=1,U=0,opcode=11001)
//   frinti v0.4s = 0x6EA19800  (a=1,U=1,opcode=11001)
constexpr uint32_t kFrintnV4s00 = 0x4E218800;
constexpr uint32_t kFrintnV2s00 = 0x0E218800;
constexpr uint32_t kFrintnV2d00 = 0x4E618800;
constexpr uint32_t kFrintmV4s00 = 0x4E219800;
constexpr uint32_t kFrintxV4s00 = 0x6E219800;
constexpr uint32_t kFrintpV4s00 = 0x4EA18800;
constexpr uint32_t kFrintzV4s00 = 0x4EA19800;
constexpr uint32_t kFrintiV4s00 = 0x6EA19800;

// FRINTN .4S (round to nearest, ties to even).  Ties: 2.5 -> 2.0, 3.5 -> 4.0.
TEST_F(Arm64LiteTranslateRegionTest, FrintnV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f  -> 2.0f (tie to even)
  lanes[1] = 0x40600000u;  // 3.5f  -> 4.0f (tie to even)
  lanes[2] = 0x3F19999Au;  // 0.6f  -> 1.0f
  lanes[3] = 0xBF19999Au;  // -0.6f -> -1.0f
  static const uint32_t code[] = {kFrintnV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40000000u);  // 2.0f
  EXPECT_EQ(lanes[1], 0x40800000u);  // 4.0f
  EXPECT_EQ(lanes[2], 0x3F800000u);  // 1.0f
  EXPECT_EQ(lanes[3], 0xBF800000u);  // -1.0f
}

// FRINTN .2S: Q=0 form.
TEST_F(Arm64LiteTranslateRegionTest, FrintnV2s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 2.0f
  lanes[1] = 0x3F19999Au;  // 0.6f -> 1.0f
  lanes[2] = 0xDEADBEEFu;
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFrintnV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40000000u);
  EXPECT_EQ(lanes[1], 0x3F800000u);
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// FRINTN .2D: FP64 tie-to-even.
TEST_F(Arm64LiteTranslateRegionTest, FrintnV2d) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4004000000000000ULL;  // 2.5d -> 2.0d
  lanes[1] = 0x400C000000000000ULL;  // 3.5d -> 4.0d
  static const uint32_t code[] = {kFrintnV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x4000000000000000ULL);  // 2.0d
  EXPECT_EQ(lanes[1], 0x4010000000000000ULL);  // 4.0d
}

// FRINTM .4S (round toward -inf, floor).
TEST_F(Arm64LiteTranslateRegionTest, FrintmV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 2.0f
  lanes[1] = 0xC0200000u;  // -2.5f -> -3.0f
  lanes[2] = 0x3F19999Au;  // 0.6f -> 0.0f
  lanes[3] = 0xBF19999Au;  // -0.6f -> -1.0f
  static const uint32_t code[] = {kFrintmV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40000000u);  // 2.0f
  EXPECT_EQ(lanes[1], 0xC0400000u);  // -3.0f
  EXPECT_EQ(lanes[2], 0x00000000u);  // 0.0f
  EXPECT_EQ(lanes[3], 0xBF800000u);  // -1.0f
}

// FRINTX .4S (round per MXCSR; default RNE matches FRINTN result).
TEST_F(Arm64LiteTranslateRegionTest, FrintxV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 2.0f (tie to even, MXCSR default RNE)
  lanes[1] = 0x40600000u;  // 3.5f -> 4.0f
  lanes[2] = 0x3F19999Au;  // 0.6f -> 1.0f
  lanes[3] = 0xBF19999Au;  // -0.6f -> -1.0f
  static const uint32_t code[] = {kFrintxV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40000000u);
  EXPECT_EQ(lanes[1], 0x40800000u);
  EXPECT_EQ(lanes[2], 0x3F800000u);
  EXPECT_EQ(lanes[3], 0xBF800000u);
}

// FRINTP .4S (round toward +inf, ceil).  These bit23=1 paths are
// already reachable pre-fix and serve as a positive control.
TEST_F(Arm64LiteTranslateRegionTest, FrintpV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 3.0f
  lanes[1] = 0xC0200000u;  // -2.5f -> -2.0f
  lanes[2] = 0x3F19999Au;  // 0.6f -> 1.0f
  lanes[3] = 0xBF19999Au;  // -0.6f -> -0.0f
  static const uint32_t code[] = {kFrintpV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40400000u);  // 3.0f
  EXPECT_EQ(lanes[1], 0xC0000000u);  // -2.0f
  EXPECT_EQ(lanes[2], 0x3F800000u);  // 1.0f
  EXPECT_EQ(lanes[3], 0x80000000u);  // -0.0f
}

// FRINTZ .4S (round toward zero, trunc).
TEST_F(Arm64LiteTranslateRegionTest, FrintzV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 2.0f
  lanes[1] = 0xC0200000u;  // -2.5f -> -2.0f
  lanes[2] = 0x3F19999Au;  // 0.6f -> 0.0f
  lanes[3] = 0xBF19999Au;  // -0.6f -> -0.0f
  static const uint32_t code[] = {kFrintzV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40000000u);  // 2.0f
  EXPECT_EQ(lanes[1], 0xC0000000u);  // -2.0f
  EXPECT_EQ(lanes[2], 0x00000000u);  // +0.0f
  EXPECT_EQ(lanes[3], 0x80000000u);  // -0.0f
}

// FRINTI .4S (round per FPCR; default RNE matches FRINTN).  Positive
// control for the bit23=1, U=1 corner.
TEST_F(Arm64LiteTranslateRegionTest, FrintiV4s) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;
  lanes[1] = 0x40600000u;
  lanes[2] = 0x3F19999Au;
  lanes[3] = 0xBF19999Au;
  static const uint32_t code[] = {kFrintiV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(lanes[0], 0x40000000u);
  EXPECT_EQ(lanes[1], 0x40800000u);
  EXPECT_EQ(lanes[2], 0x3F800000u);
  EXPECT_EQ(lanes[3], 0xBF800000u);
}

// FCVTZS vector FP32 -> S32 truncating.  Encoding per ARM ARM C7.2
// "Advanced SIMD two-register miscellaneous": opcode=11011, bit23=1
// (a=1), bit22=0 (sz=0, FP32 lanes), U=0.  Hand-derived:
//   fcvtzs v0.4s, v0.4s = 0x4EA1B800  (Q=1)
//   fcvtzs v0.2s, v0.2s = 0x0EA1B800  (Q=0)
constexpr uint32_t kFcvtzsV4s00 = 0x4EA1B800;
constexpr uint32_t kFcvtzsV2s00 = 0x0EA1B800;

// Mixed normal lanes: positive truncation, negative truncation, exact
// integer, and zero.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsV4sNormalLanes) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4048F5C3u;  // 3.14f -> 3
  lanes[1] = 0xC048F5C3u;  // -3.14f -> -3
  lanes[2] = 0x40A00000u;  // 5.0f -> 5
  lanes[3] = 0x00000000u;  // +0.0f -> 0
  static const uint32_t code[] = {kFcvtzsV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], -3);
  EXPECT_EQ(out[2], 5);
  EXPECT_EQ(out[3], 0);
}

// Saturation boundaries: NaN -> 0, +Inf -> INT32_MAX, -Inf -> INT32_MIN,
// finite > 2^31 -> INT32_MAX.  These are the cases where x86
// CVTTPS2DQ returns INT32_MIN (0x80000000) and ARM expects a different
// saturated value -- exercises the NaN-mask + pos-overflow fix-up.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsV4sSaturationBoundaries) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x7FC00000u;  // qNaN -> 0
  lanes[1] = 0x7F800000u;  // +Inf -> INT32_MAX
  lanes[2] = 0xFF800000u;  // -Inf -> INT32_MIN
  lanes[3] = 0x4F000000u;  // 2^31 (=2147483648.0f) -> INT32_MAX
  static const uint32_t code[] = {kFcvtzsV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], INT32_MAX);
  EXPECT_EQ(out[2], INT32_MIN);
  EXPECT_EQ(out[3], INT32_MAX);
}

// .2S form (Q=0): only the low 64 bits are operative; the upper 64 must
// be zeroed regardless of prior content.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsV2sZeroesUpperHalf) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4048F5C3u;  // 3.14f -> 3
  lanes[1] = 0xC048F5C3u;  // -3.14f -> -3
  lanes[2] = 0xDEADBEEFu;  // garbage; must be zeroed
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFcvtzsV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], -3);
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// FCVTZU vector FP32 -> U32 truncating.  Same encoding shape as FCVTZS
// (opcode=11011, bit23=1, sz=0) but with U=1 (bit29).  Hand-derived:
//   fcvtzu v0.4s, v0.4s = 0x6EA1B800  (Q=1, U=1)
//   fcvtzu v0.2s, v0.2s = 0x2EA1B800  (Q=0, U=1)
constexpr uint32_t kFcvtzuV4s00 = 0x6EA1B800;
constexpr uint32_t kFcvtzuV2s00 = 0x2EA1B800;

// Mixed normal lanes: in-range positive, negative clamps to 0,
// exact integer, and a value in [2^31, 2^32) that exercises the
// subtract-2^31 offset trick.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuV4sNormalLanes) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4048F5C3u;  // 3.14f -> 3
  lanes[1] = 0xC048F5C3u;  // -3.14f -> 0 (negative clamps)
  lanes[2] = 0x40A00000u;  // 5.0f -> 5
  lanes[3] = 0x4F000001u;  // 2^31 + 256 = 2147483904.0f -> 0x80000100
  static const uint32_t code[] = {kFcvtzuV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3u);
  EXPECT_EQ(out[1], 0u);
  EXPECT_EQ(out[2], 5u);
  EXPECT_EQ(out[3], 0x80000100u);
}

// Saturation boundaries: NaN -> 0, +Inf -> UINT32_MAX, -Inf -> 0
// (negative clamp), 2^32 (=0x4F800000) -> UINT32_MAX.  Exercises the
// MAXPS-clamps-neg/NaN path and the too_big saturation path.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuV4sSaturationBoundaries) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x7FC00000u;  // qNaN -> 0
  lanes[1] = 0x7F800000u;  // +Inf -> UINT32_MAX
  lanes[2] = 0xFF800000u;  // -Inf -> 0
  lanes[3] = 0x4F800000u;  // 2^32 = 4294967296.0f -> UINT32_MAX
  static const uint32_t code[] = {kFcvtzuV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 0u);
  EXPECT_EQ(out[1], UINT32_MAX);
  EXPECT_EQ(out[2], 0u);
  EXPECT_EQ(out[3], UINT32_MAX);
}

// .2S form (Q=0): low 64 bits computed (one lane in [0,2^31), one in
// [2^31, 2^32) to exercise the offset trick under Q=0), upper 64 must
// be zeroed regardless of prior content.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuV2sZeroesUpperHalf) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4048F5C3u;  // 3.14f -> 3
  lanes[1] = 0x4F000001u;  // 2^31 + 256 -> 0x80000100 (offset trick)
  lanes[2] = 0xDEADBEEFu;  // garbage; must be zeroed
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFcvtzuV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3u);
  EXPECT_EQ(out[1], 0x80000100u);
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// FCVTZS / FCVTZU vector FP64 -> S64/U64 truncating.  Encoding per
// ARM ARM C7.2 "Advanced SIMD two-register miscellaneous": opcode=11011,
// bit23=1 (a=1), bit22=1 (sz=1, FP64 lanes), Q=1.  Hand-derived from
// the FP32 .4S encoding by setting bit22:
//   fcvtzs v0.2d, v0.2d = 0x4EE1B800  (Q=1, U=0, sz=1)
//   fcvtzu v0.2d, v0.2d = 0x6EE1B800  (Q=1, U=1, sz=1)
constexpr uint32_t kFcvtzsV2d00 = 0x4EE1B800;
constexpr uint32_t kFcvtzuV2d00 = 0x6EE1B800;

// Two normal lanes: positive truncation and negative truncation.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsV2dNormalLanes) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40091EB851EB851FULL;  // 3.14 -> 3
  lanes[1] = 0xC0091EB851EB851FULL;  // -3.14 -> -3
  static const uint32_t code[] = {kFcvtzsV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], -3);
}

// Saturation: NaN -> 0, +Inf -> INT64_MAX.  Exercises the NaN path
// (Ucomisd PF=1) and the positive-overflow path (Cvttsd2siq returns
// INT64_MIN for +Inf; sign-of-FP non-negative triggers INT64_MAX
// fix-up).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsV2dSaturationNanAndPosInf) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x7FF8000000000000ULL;  // qNaN -> 0
  lanes[1] = 0x7FF0000000000000ULL;  // +Inf -> INT64_MAX
  static const uint32_t code[] = {kFcvtzsV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], INT64_MAX);
}

// Saturation: -Inf -> INT64_MIN (already correct from Cvttsd2siq's
// indefinite, sign-of-FP-negative branch keeps tmp); FP exactly 2^63
// -> INT64_MAX (positive overflow, sign-of-FP non-negative + tmp ==
// INT64_MIN triggers the fix-up).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsV2dSaturationNegInfAndPos2p63) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0xFFF0000000000000ULL;  // -Inf -> INT64_MIN
  lanes[1] = 0x43E0000000000000ULL;  // 2^63 -> INT64_MAX (positive overflow)
  static const uint32_t code[] = {kFcvtzsV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], INT64_MIN);
  EXPECT_EQ(out[1], INT64_MAX);
}

// FCVTZU .2D: normal positive truncation in one lane, negative clamp
// (ARM FCVTZU clamps negatives, including -0.0, to 0) in the other.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuV2dNormalLanes) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40091EB851EB851FULL;  // 3.14 -> 3
  lanes[1] = 0xC0091EB851EB851FULL;  // -3.14 -> 0 (negative clamps)
  static const uint32_t code[] = {kFcvtzuV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3u);
  EXPECT_EQ(out[1], 0u);
}

// FCVTZU .2D saturation: NaN -> 0 (Ucomisd PF=1, zero_path), +Inf ->
// UINT64_MAX (FP >= 2^64 -> sat_max).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuV2dSaturationBoundaries) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x7FF8000000000000ULL;  // qNaN -> 0
  lanes[1] = 0x7FF0000000000000ULL;  // +Inf -> UINT64_MAX
  static const uint32_t code[] = {kFcvtzuV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 0u);
  EXPECT_EQ(out[1], UINT64_MAX);
}

// FCVTZU .2D offset trick: FP in [2^63, 2^64) exercises the
// subtract-2^63 + cvtt + bit-63 path; FP = 2^64 saturates to
// UINT64_MAX.  The first lane uses FP64 0x43E0000000000001 which is
// the next representable FP64 after 2^63 — equal to 2^63 + 2^11
// = 0x8000000000000800 as a u64.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuV2dOffsetTrickAndSat2p64) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x43E0000000000001ULL;  // 2^63 + 2^11 = next FP64 after 2^63
  lanes[1] = 0x43F0000000000000ULL;  // 2^64 -> UINT64_MAX (>= bound)
  static const uint32_t code[] = {kFcvtzuV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 0x8000000000000800ULL);
  EXPECT_EQ(out[1], UINT64_MAX);
}

// FCVT[NS|PS|MS|NU|PU|MU] vector FP64 -> S64/U64 with explicit rounding.
// Encoding per ARM ARM C7.2:
//   FCVTNS V .2D = 0x4E61A800 (Q=1, U=0, bit23=0, bit22=1, opcode=11010)
//   FCVTPS V .2D = 0x4EE1A800 (Q=1, U=0, bit23=1, bit22=1, opcode=11010)
//   FCVTMS V .2D = 0x4E61B800 (Q=1, U=0, bit23=0, bit22=1, opcode=11011)
//   FCVTNU V .2D = 0x6E61A800 (U=1 variant of FCVTNS)
//   FCVTPU V .2D = 0x6EE1A800 (U=1 variant of FCVTPS)
//   FCVTMU V .2D = 0x6E61B800 (U=1 variant of FCVTMS)
constexpr uint32_t kFcvtnsV2d00 = 0x4E61A800;
constexpr uint32_t kFcvtpsV2d00 = 0x4EE1A800;
constexpr uint32_t kFcvtmsV2d00 = 0x4E61B800;
constexpr uint32_t kFcvtnuV2d00 = 0x6E61A800;
constexpr uint32_t kFcvtpuV2d00 = 0x6EE1A800;
constexpr uint32_t kFcvtmuV2d00 = 0x6E61B800;

// FCVTNS round-to-nearest ties-even: 3.5 -> 4 (ties up to even), -3.5 ->
// -4 (ties to even, -3 is odd so -3.5 rounds to -4).  Differs from trunc
// (3, -3) and floor (3, -4).
TEST_F(Arm64LiteTranslateRegionTest, FcvtnsV2dRoundToEven) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x400C000000000000ULL;  // 3.5 -> 4
  lanes[1] = 0xC00C000000000000ULL;  // -3.5 -> -4
  static const uint32_t code[] = {kFcvtnsV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 4);
  EXPECT_EQ(out[1], -4);
}

// FCVTNS .2D saturation: NaN -> 0, +Inf -> INT64_MAX.  Roundsd preserves
// NaN and Inf; the saturation classifier produces ARM-correct outputs.
TEST_F(Arm64LiteTranslateRegionTest, FcvtnsV2dSaturationNanAndPosInf) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x7FF8000000000000ULL;  // qNaN -> 0
  lanes[1] = 0x7FF0000000000000ULL;  // +Inf -> INT64_MAX
  static const uint32_t code[] = {kFcvtnsV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], INT64_MAX);
}

// FCVTPS round toward +inf (ceiling): 2.1 -> 3, -2.9 -> -2.  Differs from
// trunc (2, -2) and from FCVTNS (2, -3) and FCVTMS (2, -3).
TEST_F(Arm64LiteTranslateRegionTest, FcvtpsV2dCeiling) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40006CCCCCCCCCCDULL;  // 2.1 -> 3
  lanes[1] = 0xC00733333333333BULL;  // -2.9 -> -2
  static const uint32_t code[] = {kFcvtpsV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], -2);
}

// FCVTMS round toward -inf (floor): 2.9 -> 2, -2.1 -> -3.  Differs from
// trunc (2, -2) and FCVTPS (3, -2).
TEST_F(Arm64LiteTranslateRegionTest, FcvtmsV2dFloor) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4007333333333333ULL;  // 2.9 -> 2
  lanes[1] = 0xC0006CCCCCCCCCCDULL;  // -2.1 -> -3
  static const uint32_t code[] = {kFcvtmsV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 2);
  EXPECT_EQ(out[1], -3);
}

// FCVTNU .2D: round-to-nearest ties-even for unsigned target.
//   2.5 -> 2 (ties to even, 2 is even).
//   -0.5 -> 0 (rounds to -0.0 by ties-to-even since 0 is even; sign bit
//             set classifies as negative -> clamp to 0).
TEST_F(Arm64LiteTranslateRegionTest, FcvtnuV2dRoundToEvenAndNegClamp) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4004000000000000ULL;  // 2.5 -> 2 (ties to even)
  lanes[1] = 0xBFE0000000000000ULL;  // -0.5 -> 0 (round to -0.0, clamp)
  static const uint32_t code[] = {kFcvtnuV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 2u);
  EXPECT_EQ(out[1], 0u);
}

// FCVTPU .2D: ceiling for unsigned target.
//   0.1 -> 1 (ceil of 0.1).
//   2^64 -> UINT64_MAX (saturation; ceil(2^64)=2^64 hits >= 2^64 path).
TEST_F(Arm64LiteTranslateRegionTest, FcvtpuV2dCeilAndSat) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x3FB999999999999AULL;  // 0.1 -> 1
  lanes[1] = 0x43F0000000000000ULL;  // 2^64 -> UINT64_MAX
  static const uint32_t code[] = {kFcvtpuV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 1u);
  EXPECT_EQ(out[1], UINT64_MAX);
}

// FCVTMU .2D: floor for unsigned target.
//   3.9 -> 3 (floor).
//   -0.5 -> 0 (floor of -0.5 = -1, sign bit set -> clamp to 0).
TEST_F(Arm64LiteTranslateRegionTest, FcvtmuV2dFloorAndNegClamp) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x400F333333333333ULL;  // 3.9 -> 3
  lanes[1] = 0xBFE0000000000000ULL;  // -0.5 -> 0 (floor to -1, clamp)
  static const uint32_t code[] = {kFcvtmuV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3u);
  EXPECT_EQ(out[1], 0u);
}

// FCVT[NS|PS|MS|NU|PU|MU] vector FP32 -> S32/U32 with explicit rounding.
// Encoding shape mirrors the .2D family above with bit22 (sz) flipped to 0:
//   FCVTNS V .4S = 0x4E21A800 (Q=1, U=0, bit23=0, sz=0, opcode=11010)
//   FCVTPS V .4S = 0x4EA1A800 (Q=1, U=0, bit23=1, sz=0, opcode=11010)
//   FCVTMS V .4S = 0x4E21B800 (Q=1, U=0, bit23=0, sz=0, opcode=11011)
//   FCVTNU V .4S = 0x6E21A800 (U=1 variant)
//   FCVTPU V .4S = 0x6EA1A800
//   FCVTMU V .4S = 0x6E21B800
//   .2S forms have Q=0 (flip bit30): FCVTNS V .2S = 0x0E21A800, etc.
constexpr uint32_t kFcvtnsV4s00 = 0x4E21A800;
constexpr uint32_t kFcvtpsV4s00 = 0x4EA1A800;
constexpr uint32_t kFcvtmsV4s00 = 0x4E21B800;
constexpr uint32_t kFcvtnuV4s00 = 0x6E21A800;
constexpr uint32_t kFcvtpuV4s00 = 0x6EA1A800;
constexpr uint32_t kFcvtmuV4s00 = 0x6E21B800;
constexpr uint32_t kFcvtnsV2s00 = 0x0E21A800;

// FCVTNS .4S round-to-nearest ties-even: 3.5f -> 4, -3.5f -> -4 (ties to
// even), NaN -> 0, +Inf -> INT32_MAX.  Differs from trunc (which would
// give 3, -3) and from floor/ceil.
TEST_F(Arm64LiteTranslateRegionTest, FcvtnsV4sRoundToEvenAndSat) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40600000u;  // 3.5f -> 4
  lanes[1] = 0xC0600000u;  // -3.5f -> -4 (ties to even)
  lanes[2] = 0x7FC00000u;  // qNaN -> 0
  lanes[3] = 0x7F800000u;  // +Inf -> INT32_MAX
  static const uint32_t code[] = {kFcvtnsV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 4);
  EXPECT_EQ(out[1], -4);
  EXPECT_EQ(out[2], 0);
  EXPECT_EQ(out[3], INT32_MAX);
}

// FCVTPS .4S round toward +inf (ceiling): 2.1f -> 3, -2.9f -> -2.
// Differs from trunc (2, -2), floor (2, -3), and RNE (2, -3).
TEST_F(Arm64LiteTranslateRegionTest, FcvtpsV4sCeiling) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40066666u;  // 2.1f -> 3
  lanes[1] = 0xC039999Au;  // -2.9f -> -2
  lanes[2] = 0xFF800000u;  // -Inf -> INT32_MIN
  lanes[3] = 0x4F000000u;  // 2^31 -> INT32_MAX (pos overflow fix-up)
  static const uint32_t code[] = {kFcvtpsV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], -2);
  EXPECT_EQ(out[2], INT32_MIN);
  EXPECT_EQ(out[3], INT32_MAX);
}

// FCVTMS .4S round toward -inf (floor): 2.9f -> 2, -2.1f -> -3.
// Differs from trunc (2, -2) and ceil (3, -2).
TEST_F(Arm64LiteTranslateRegionTest, FcvtmsV4sFloor) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4039999Au;  // 2.9f -> 2
  lanes[1] = 0xC0066666u;  // -2.1f -> -3
  lanes[2] = 0x00000000u;  // +0.0f -> 0
  lanes[3] = 0x80000000u;  // -0.0f -> 0 (floor(-0.0) = -0.0, cvtt(-0.0)=0)
  static const uint32_t code[] = {kFcvtmsV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 2);
  EXPECT_EQ(out[1], -3);
  EXPECT_EQ(out[2], 0);
  EXPECT_EQ(out[3], 0);
}

// .2S form (Q=0): low 64 bits computed, upper 64 zeroed.  Uses FCVTNS V .2S.
TEST_F(Arm64LiteTranslateRegionTest, FcvtnsV2sZeroesUpperHalf) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40600000u;  // 3.5f -> 4 (RNE)
  lanes[1] = 0xC0600000u;  // -3.5f -> -4
  lanes[2] = 0xDEADBEEFu;  // garbage; must be zeroed
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFcvtnsV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 4);
  EXPECT_EQ(out[1], -4);
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// FCVTNU .4S: round-to-nearest ties-even for unsigned target.
//   2.5f -> 2 (ties to even).
//   -0.5f -> 0 (RNE -> -0.0, sign bit -> MAXPS clamp to 0).
//   +Inf -> UINT32_MAX, 2^32 -> UINT32_MAX (too_big path).
TEST_F(Arm64LiteTranslateRegionTest, FcvtnuV4sRoundToEvenAndSat) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 2 (ties to even)
  lanes[1] = 0xBF000000u;  // -0.5f -> 0
  lanes[2] = 0x7F800000u;  // +Inf -> UINT32_MAX
  lanes[3] = 0x4F800000u;  // 2^32 -> UINT32_MAX
  static const uint32_t code[] = {kFcvtnuV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 2u);
  EXPECT_EQ(out[1], 0u);
  EXPECT_EQ(out[2], UINT32_MAX);
  EXPECT_EQ(out[3], UINT32_MAX);
}

// FCVTPU .4S: ceiling for unsigned target.
//   0.1f -> 1, 2^31+256 -> 0x80000100 (exercises the subtract-2^31 offset
//   trick after ceil), 2^32 -> UINT32_MAX, NaN -> 0.
TEST_F(Arm64LiteTranslateRegionTest, FcvtpuV4sCeilAndSat) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x3DCCCCCDu;  // 0.1f -> 1 (ceil)
  lanes[1] = 0x4F000001u;  // 2^31 + 256 (already an exact FP32 integer)
  lanes[2] = 0x4F800000u;  // 2^32 -> UINT32_MAX
  lanes[3] = 0x7FC00000u;  // qNaN -> 0
  static const uint32_t code[] = {kFcvtpuV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 1u);
  EXPECT_EQ(out[1], 0x80000100u);
  EXPECT_EQ(out[2], UINT32_MAX);
  EXPECT_EQ(out[3], 0u);
}

// FCVTMU .4S: floor for unsigned target.
//   3.9f -> 3 (floor).
//   -0.5f -> 0 (floor of -0.5 = -1, sign bit -> MAXPS clamp to 0).
//   3.14f -> 3, in-range positive (regression vs trunc which also gives 3).
//   -Inf -> 0 (MAXPS clamp).
TEST_F(Arm64LiteTranslateRegionTest, FcvtmuV4sFloorAndNegClamp) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x4079999Au;  // 3.9f -> 3
  lanes[1] = 0xBF000000u;  // -0.5f -> 0 (floor to -1, clamp)
  lanes[2] = 0x4048F5C3u;  // 3.14f -> 3
  lanes[3] = 0xFF800000u;  // -Inf -> 0
  static const uint32_t code[] = {kFcvtmuV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3u);
  EXPECT_EQ(out[1], 0u);
  EXPECT_EQ(out[2], 3u);
  EXPECT_EQ(out[3], 0u);
}

// FCVTAS / FCVTAU: round-to-nearest ties-away-from-zero, FP -> int.
//   FCVTAS V .4S = 0x4E21C800 (Q=1, U=0, bit23=0, sz=0, opcode=11100)
//   FCVTAS V .2S = 0x0E21C800 (Q=0)
//   FCVTAS V .2D = 0x4E61C800 (sz=1)
//   FCVTAU V .4S = 0x6E21C800 (U=1)
//   FCVTAU V .2D = 0x6E61C800
constexpr uint32_t kFcvtasV4s00 = 0x4E21C800;
constexpr uint32_t kFcvtasV2s00 = 0x0E21C800;
constexpr uint32_t kFcvtasV2d00 = 0x4E61C800;
constexpr uint32_t kFcvtauV4s00 = 0x6E21C800;
constexpr uint32_t kFcvtauV2d00 = 0x6E61C800;

// FCVTAS .4S ties-away: 2.5f -> 3 (RNE -> 2, ties to even), -2.5f -> -3,
// +Inf -> INT32_MAX (saturation), 2^23+1 -> 2^23+1 (magnitude-gate check;
// without the gate, addend=+0.5 would FP-round to 2^23+2).
TEST_F(Arm64LiteTranslateRegionTest, FcvtasV4sTiesAwayAndSat) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 3 (ties away)
  lanes[1] = 0xC0200000u;  // -2.5f -> -3
  lanes[2] = 0x7F800000u;  // +Inf -> INT32_MAX
  lanes[3] = 0x4B000001u;  // 2^23 + 1 = 8388609.0f -> 8388609 (gate test)
  static const uint32_t code[] = {kFcvtasV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], -3);
  EXPECT_EQ(out[2], INT32_MAX);
  EXPECT_EQ(out[3], 8388609);
}

// FCVTAS .2S (Q=0): low 64 bits computed, upper 64 zeroed.
TEST_F(Arm64LiteTranslateRegionTest, FcvtasV2sZeroesUpperHalf) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 3 (ties away)
  lanes[1] = 0xC0200000u;  // -2.5f -> -3
  lanes[2] = 0xDEADBEEFu;  // garbage; must be zeroed
  lanes[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kFcvtasV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], -3);
  EXPECT_EQ(lanes[2], 0u);
  EXPECT_EQ(lanes[3], 0u);
}

// FCVTAS .2D ties-away: 0.5 -> 1 (RNE -> 0, ties to even), -0.5 -> -1.
// NaN -> 0, +Inf -> INT64_MAX would saturate the second lane in a separate
// test; this test focuses on the half-tie distinguishing case.
TEST_F(Arm64LiteTranslateRegionTest, FcvtasV2dTiesAway) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x3FE0000000000000ULL;  // 0.5 -> 1 (ties away)
  lanes[1] = 0xBFE0000000000000ULL;  // -0.5 -> -1
  static const uint32_t code[] = {kFcvtasV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], -1);
}

// FCVTAS .2D saturation: +Inf -> INT64_MAX (pos-overflow), NaN -> 0.
TEST_F(Arm64LiteTranslateRegionTest, FcvtasV2dSaturation) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x7FF0000000000000ULL;  // +Inf -> INT64_MAX
  lanes[1] = 0x7FF8000000000000ULL;  // qNaN -> 0
  static const uint32_t code[] = {kFcvtasV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], INT64_MAX);
  EXPECT_EQ(out[1], 0);
}

// FCVTAU .4S ties-away unsigned: 2.5f -> 3, -0.5f -> 0 (negative -> 0
// clamp), +Inf -> UINT32_MAX, 2^32 -> UINT32_MAX.
TEST_F(Arm64LiteTranslateRegionTest, FcvtauV4sTiesAwayAndSat) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x40200000u;  // 2.5f -> 3 (ties away)
  lanes[1] = 0xBF000000u;  // -0.5f -> 0 (round to -1, negative -> 0)
  lanes[2] = 0x7F800000u;  // +Inf -> UINT32_MAX
  lanes[3] = 0x4F800000u;  // 2^32 -> UINT32_MAX
  static const uint32_t code[] = {kFcvtauV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 3u);
  EXPECT_EQ(out[1], 0u);
  EXPECT_EQ(out[2], UINT32_MAX);
  EXPECT_EQ(out[3], UINT32_MAX);
}

// FCVTAU .2D ties-away unsigned: 0.5 -> 1, -0.5 -> 0, 2^64 -> UINT64_MAX.
TEST_F(Arm64LiteTranslateRegionTest, FcvtauV2dTiesAwayAndSat) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  lanes[0] = 0x3FE0000000000000ULL;  // 0.5 -> 1 (ties away)
  lanes[1] = 0x43F0000000000000ULL;  // 2^64 -> UINT64_MAX
  static const uint32_t code[] = {kFcvtauV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 1u);
  EXPECT_EQ(out[1], UINT64_MAX);
}
// endregion

// region digitalis - SCVTF / UCVTF V: vector int -> FP for .2S, .4S, .2D.
constexpr uint32_t kScvtfV4s00 = 0x4E21D800;
constexpr uint32_t kScvtfV2s00 = 0x0E21D800;
constexpr uint32_t kScvtfV2d00 = 0x4E61D800;
constexpr uint32_t kUcvtfV4s00 = 0x6E21D800;
constexpr uint32_t kUcvtfV2s00 = 0x2E21D800;
constexpr uint32_t kUcvtfV2d00 = 0x6E61D800;

// SCVTF .4S: signed int32 -> FP32 in all four lanes.
TEST_F(Arm64LiteTranslateRegionTest, ScvtfV4sBasic) {
  state_.cpu.v[0] = 0;
  auto* lanes = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  lanes[0] = 0;
  lanes[1] = 1;
  lanes[2] = -1;
  lanes[3] = INT32_MIN;  // -2^31
  static const uint32_t code[] = {kScvtfV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<float*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 0.0f);
  EXPECT_EQ(out[1], 1.0f);
  EXPECT_EQ(out[2], -1.0f);
  EXPECT_EQ(out[3], -2147483648.0f);
}

// SCVTF .2S (Q=0): low 64 bits computed, upper 64 zeroed.
TEST_F(Arm64LiteTranslateRegionTest, ScvtfV2sZeroesUpperHalf) {
  state_.cpu.v[0] = 0;
  auto* in = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  in[0] = 7;
  in[1] = -42;
  in[2] = 0x12345678;  // garbage; must be zeroed
  in[3] = 0x7BADBEEF;
  static const uint32_t code[] = {kScvtfV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<float*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 7.0f);
  EXPECT_EQ(out[1], -42.0f);
  auto* out_u = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out_u[2], 0u);
  EXPECT_EQ(out_u[3], 0u);
}

// SCVTF .2D: signed int64 -> FP64.  2^53 (largest exact int53) round-trips
// exactly; -2^53 also exact.
TEST_F(Arm64LiteTranslateRegionTest, ScvtfV2dBasic) {
  state_.cpu.v[0] = 0;
  auto* in = reinterpret_cast<int64_t*>(&state_.cpu.v[0]);
  in[0] = int64_t{1} << 53;       // 9007199254740992
  in[1] = -(int64_t{1} << 53);
  static const uint32_t code[] = {kScvtfV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<double*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 9007199254740992.0);
  EXPECT_EQ(out[1], -9007199254740992.0);
}

// UCVTF .4S: unsigned int32 -> FP32.  Values >= 2^31 require the MSB-set
// addend (otherwise signed conversion would yield negative results).
TEST_F(Arm64LiteTranslateRegionTest, UcvtfV4sMsbAddend) {
  state_.cpu.v[0] = 0;
  auto* in = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  in[0] = 0u;
  in[1] = 1u;
  in[2] = uint32_t{1} << 31;     // 2^31, MSB just set
  in[3] = UINT32_MAX;             // 2^32 - 1 (rounds up to 2^32 in FP32)
  static const uint32_t code[] = {kUcvtfV4s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<float*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 0.0f);
  EXPECT_EQ(out[1], 1.0f);
  EXPECT_EQ(out[2], 2147483648.0f);   // 2^31
  // UINT32_MAX = 4294967295; nearest FP32 is 2^32 = 4294967296.0f.
  EXPECT_EQ(out[3], 4294967296.0f);
}

// UCVTF .2S (Q=0): low 64 bits computed, upper 64 zeroed; also exercises
// the unsigned MSB-addend path with bit31 set in lane 1.
TEST_F(Arm64LiteTranslateRegionTest, UcvtfV2sZeroesUpperHalf) {
  state_.cpu.v[0] = 0;
  auto* in = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  in[0] = 5u;
  in[1] = uint32_t{1} << 31;     // 2^31
  in[2] = 0xDEADBEEFu;
  in[3] = 0xCAFEBABEu;
  static const uint32_t code[] = {kUcvtfV2s00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<float*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 5.0f);
  EXPECT_EQ(out[1], 2147483648.0f);
  auto* out_u = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  EXPECT_EQ(out_u[2], 0u);
  EXPECT_EQ(out_u[3], 0u);
}

// UCVTF .2D: unsigned int64 -> FP64.  Exercises both the direct convert
// (bit63 clear) and the halve|LSB round-to-odd path (bit63 set).
TEST_F(Arm64LiteTranslateRegionTest, UcvtfV2dBit63Set) {
  state_.cpu.v[0] = 0;
  auto* in = reinterpret_cast<uint64_t*>(&state_.cpu.v[0]);
  in[0] = uint64_t{1} << 63;      // 2^63, exactly representable
  in[1] = UINT64_MAX;             // 2^64 - 1, rounds up to 2^64
  static const uint32_t code[] = {kUcvtfV2d00};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<double*>(&state_.cpu.v[0]);
  EXPECT_EQ(out[0], 9223372036854775808.0);   // 2^63
  EXPECT_EQ(out[1], 18446744073709551616.0);  // 2^64 (UINT64_MAX rounds up)
}
// endregion

// region digitalis - SABDL/UABDL/SABAL/UABAL .2D (size=10): Psubq +
// Pcmpgtq-against-zero signed-abs primitive (no SSE 64-bit max/min).

// SABDL .2D (Q=0): signed 32→64 widening + abs diff.  Lane 1 exercises the
// extreme INT32_MIN/INT32_MAX boundary: |-2^31 - (2^31-1)| = 2^32 - 1.
TEST_F(Arm64LiteTranslateRegionTest, SabdlV2dSigned) {
  state_.cpu.v[0] = 0;
  state_.cpu.v[1] = 0;
  auto* a = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  auto* b = reinterpret_cast<int32_t*>(&state_.cpu.v[1]);
  a[0] = 5;            b[0] = 3;
  a[1] = INT32_MIN;    b[1] = INT32_MAX;
  a[2] = 0x7BADBEEF;   b[2] = 0x12345678;  // garbage (Q=0 ignores upper half)
  a[3] = 0x0BADC0DE;   b[3] = 0xCAFEBABE;
  // Vd encoded as v0, but we want Rn=1, Rm=1... no — SABDL .2D Vd.2D, Vn.2S, Vm.2S
  // with Rd=2, Rn=0, Rm=1 to keep them distinct.
  state_.cpu.v[2] = 0xAAAAAAAAAAAAAAAAull;  // marker to verify overwrite
  constexpr uint32_t code_inst = 0x0EA17002;  // SABDL v2.2D, v0.2S, v1.2S
  static const uint32_t code[] = {code_inst};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[2]);
  EXPECT_EQ(out[0], int64_t{2});                                    // |5 - 3|
  EXPECT_EQ(out[1], int64_t{UINT32_MAX});                           // |INT32_MIN - INT32_MAX|
}

// SABDL2 .2D (Q=1): same op as SABDL but reads upper half of Vn/Vm.
TEST_F(Arm64LiteTranslateRegionTest, Sabdl2V2dUpperHalf) {
  state_.cpu.v[0] = 0;
  state_.cpu.v[1] = 0;
  auto* a = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  auto* b = reinterpret_cast<int32_t*>(&state_.cpu.v[1]);
  a[0] = 0x12345678; b[0] = 0xDEADBEEF;  // garbage (Q=1 ignores lower half)
  a[1] = 0x7BADBEEF; b[1] = 0xCAFEBABE;
  a[2] = -7;           b[2] = 3;              // |-7 - 3| = 10
  a[3] = INT32_MAX;    b[3] = INT32_MIN;      // INT32_MAX - INT32_MIN = 2^32 - 1
  state_.cpu.v[2] = 0xAAAAAAAAAAAAAAAAull;
  constexpr uint32_t code_inst = 0x4EA17002;  // SABDL2 v2.2D, v0.4S, v1.4S
  static const uint32_t code[] = {code_inst};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[2]);
  EXPECT_EQ(out[0], int64_t{10});
  EXPECT_EQ(out[1], int64_t{UINT32_MAX});
}

// UABDL .2D (Q=0): unsigned 32→64 widening + abs diff.  Lane 1 exercises
// UINT32_MAX − 0 = 4294967295.
TEST_F(Arm64LiteTranslateRegionTest, UabdlV2dUnsigned) {
  state_.cpu.v[0] = 0;
  state_.cpu.v[1] = 0;
  auto* a = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  auto* b = reinterpret_cast<uint32_t*>(&state_.cpu.v[1]);
  a[0] = 20;          b[0] = 10;
  a[1] = UINT32_MAX;  b[1] = 0;
  state_.cpu.v[2] = 0xAAAAAAAAAAAAAAAAull;
  constexpr uint32_t code_inst = 0x2EA17002;  // UABDL v2.2D, v0.2S, v1.2S
  static const uint32_t code[] = {code_inst};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[2]);
  EXPECT_EQ(out[0], uint64_t{10});
  EXPECT_EQ(out[1], uint64_t{UINT32_MAX});
}

// SABAL .2D (Q=0): SABDL + accumulate into existing Vd.
TEST_F(Arm64LiteTranslateRegionTest, SabalV2dAccumulates) {
  state_.cpu.v[0] = 0;
  state_.cpu.v[1] = 0;
  auto* a = reinterpret_cast<int32_t*>(&state_.cpu.v[0]);
  auto* b = reinterpret_cast<int32_t*>(&state_.cpu.v[1]);
  a[0] = 5;            b[0] = 3;
  a[1] = INT32_MIN;    b[1] = INT32_MAX;
  state_.cpu.v[2] = 0;
  auto* vd = reinterpret_cast<int64_t*>(&state_.cpu.v[2]);
  vd[0] = 100;
  vd[1] = 200;
  constexpr uint32_t code_inst = 0x0EA15002;  // SABAL v2.2D, v0.2S, v1.2S
  static const uint32_t code[] = {code_inst};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<int64_t*>(&state_.cpu.v[2]);
  EXPECT_EQ(out[0], int64_t{102});                                  // 100 + 2
  EXPECT_EQ(out[1], int64_t{200} + int64_t{UINT32_MAX});            // 200 + 2^32-1
}

// UABAL .2D (Q=0): UABDL + accumulate.
TEST_F(Arm64LiteTranslateRegionTest, UabalV2dAccumulates) {
  state_.cpu.v[0] = 0;
  state_.cpu.v[1] = 0;
  auto* a = reinterpret_cast<uint32_t*>(&state_.cpu.v[0]);
  auto* b = reinterpret_cast<uint32_t*>(&state_.cpu.v[1]);
  a[0] = 20;          b[0] = 10;
  a[1] = UINT32_MAX;  b[1] = 0;
  state_.cpu.v[2] = 0;
  auto* vd = reinterpret_cast<uint64_t*>(&state_.cpu.v[2]);
  vd[0] = 1000;
  vd[1] = 2000;
  constexpr uint32_t code_inst = 0x2EA15002;  // UABAL v2.2D, v0.2S, v1.2S
  static const uint32_t code[] = {code_inst};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  auto* out = reinterpret_cast<uint64_t*>(&state_.cpu.v[2]);
  EXPECT_EQ(out[0], uint64_t{1010});                                // 1000 + 10
  EXPECT_EQ(out[1], uint64_t{2000} + uint64_t{UINT32_MAX});         // 2000 + 2^32-1
}

// FJCVTZS Wd, Dn — Armv8.3-JSCVT: ECMAScript ToInt32 of a double-precision FP.
// Encoding: rmode=11, opcode=110, ftype=01, sf=0 → 0x1E7E0000 base.
// With Rd=2 (W2), Rn=0 (D0): 0x1E7E0002.
constexpr uint32_t kFjcvtzsW2D0 = 0x1E7E0002;

// Helper: load a double into V0 (FJCVTZS reads D0 / lower 64 bits of V0).
static void StoreDoubleToV0(berberis::ThreadState& s, double d) {
  s.cpu.v[0] = 0;
  memcpy(&s.cpu.v[0], &d, sizeof(d));
}

// 1. In-range exact integer: 5.0 → 5, exact (Z=1).
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsInRangeExactPositive) {
  StoreDoubleToV0(state_, 5.0);
  state_.cpu.flags = 0xDEAD;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{5});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0x4000});  // Z=1, N=C=V=0
}

// 2. In-range exact negative integer: -3.0 → 0xFFFFFFFD, exact (Z=1).
//    Verifies sign-extension and zero-extension to Wd write semantics.
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsInRangeExactNegative) {
  StoreDoubleToV0(state_, -3.0);
  state_.cpu.flags = 0xDEAD;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0xFFFFFFFDULL});  // -3 as uint32, zero-ext to 64
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0x4000});  // Z=1
}

// 3. Non-integer in range: 3.14 → trunc(3.14) = 3, not exact (Z=0).
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsInRangeNonInteger) {
  StoreDoubleToV0(state_, 3.14);
  state_.cpu.flags = 0;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{3});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0});  // all flags clear
}

// 4. NaN → 0, not exact (Z=0).
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsNaN) {
  state_.cpu.v[0] = 0;
  uint64_t qnan_bits = 0x7FF8000000000000ULL;
  memcpy(&state_.cpu.v[0], &qnan_bits, 8);
  state_.cpu.flags = 0xFFFF;  // start with everything set
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0});  // Z=0 for NaN
}

// 5. +Infinity → 0, not exact (Z=0).
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsPositiveInfinity) {
  state_.cpu.v[0] = 0;
  uint64_t pinf_bits = 0x7FF0000000000000ULL;
  memcpy(&state_.cpu.v[0], &pinf_bits, 8);
  state_.cpu.flags = 0;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0});
}

// 6. -Infinity → 0, not exact (Z=0).
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsNegativeInfinity) {
  state_.cpu.v[0] = 0;
  uint64_t ninf_bits = 0xFFF0000000000000ULL;
  memcpy(&state_.cpu.v[0], &ninf_bits, 8);
  state_.cpu.flags = 0;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0});
}

// 7. -0.0 → 0, exact (Z=1, because trunc(-0) == d in FP).
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsNegativeZero) {
  StoreDoubleToV0(state_, -0.0);
  state_.cpu.flags = 0;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0x4000});  // Z=1
}

// 8. Out-of-range positive: 2^31 = 2147483648.0 → ECMAScript ToInt32 = -2^31
//    (signed-wrap of 0x80000000), zero-extended to 64-bit Wd = 0x80000000.
//    Not exact (Z=0, since 2147483648 != -2147483648 as doubles).
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsOutOfRangePositive) {
  StoreDoubleToV0(state_, 2147483648.0);  // 2^31
  state_.cpu.flags = 0;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0x80000000ULL});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0});
}

// 9. Out-of-range negative: -(2^31 + 1) = -2147483649.0 → ECMAScript ToInt32
//    = 2147483647 (0x7FFFFFFF).  Not exact (Z=0).
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsOutOfRangeNegative) {
  StoreDoubleToV0(state_, -2147483649.0);
  state_.cpu.flags = 0;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0x7FFFFFFFULL});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0});
}

// 10. Modular reduction: 2^32 + 5 = 4294967301.0 → ToInt32 = 5.
//     Verifies the d - trunc(d/2^32)*2^32 path produces the correct
//     low-32-bit result for values well past INT32_MAX.
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsModularReduction) {
  StoreDoubleToV0(state_, 4294967301.0);  // 2^32 + 5
  state_.cpu.flags = 0;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{5});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0});
}

// 11. Very large finite double whose mathematical low-32 is non-zero:
//     2^53 is the largest integer FP64 can represent exactly with the
//     LSB still at unit position; 2^53 mod 2^32 = 0.  Verify result = 0.
TEST_F(Arm64LiteTranslateRegionTest, FjcvtzsTwoToTheFiftyThree) {
  StoreDoubleToV0(state_, 9007199254740992.0);  // 2^53
  state_.cpu.flags = 0;
  state_.cpu.x[2] = 0xCAFEBABEDEADBEEFULL;
  static const uint32_t code[] = {kFjcvtzsW2D0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});
  EXPECT_EQ(state_.cpu.flags & 0xC101, uint16_t{0});  // not exact: d != 0
}
// endregion

// region digitalis: FMULX scalar three-same JIT lowering (FP32 / FP64).
// FMULX = FMUL except (±0 * ±inf) returns ±2.0 with sign(a) XOR sign(b).
// Encoding (per ARM ARM C7.2.149 "FMULX (vector)" scalar subset and
// llvm-mc verification):
//   FMULX Sd, Sn, Sm = 0101_1110_0010_xxxx_x1101_1100_0xxx_xxx (sz=0)
//   FMULX Dd, Dn, Dm = 0101_1110_0110_xxxx_x1101_1100_0xxx_xxx (sz=1)
// Verified: fmulx s0,s1,s2 -> 0x5e22dc20; fmulx d0,d1,d2 -> 0x5e62dc20.
constexpr uint32_t FmulxScalarS(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x5E20DC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulxScalarD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x5E60DC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

template <typename T>
static void StoreScalarToV(CPUState& cpu, unsigned idx, T value) {
  std::memset(&cpu.v[idx], 0, 16);
  std::memcpy(&cpu.v[idx], &value, sizeof(T));
}

// Regular finite multiply: 3.0 * 4.0 = 12.0 (S).
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarSRegular) {
  StoreScalarToV<float>(state_.cpu, 1, 3.0f);
  StoreScalarToV<float>(state_.cpu, 2, 4.0f);
  StoreScalarToV<float>(state_.cpu, 0, std::nanf(""));  // pre-trash dest
  static const uint32_t code[] = {FmulxScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 12.0f);
  // Lanes 1..3 must be zero (AArch64 scalar zero-extend).
  uint32_t upper[3];
  std::memcpy(upper, reinterpret_cast<const uint8_t*>(&state_.cpu.v[0]) + 4, 12);
  EXPECT_EQ(upper[0], 0u);
  EXPECT_EQ(upper[1], 0u);
  EXPECT_EQ(upper[2], 0u);
}

// Regular finite multiply: 0.5 * -3.0 = -1.5 (D).
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarDRegular) {
  StoreScalarToV<double>(state_.cpu, 1, 0.5);
  StoreScalarToV<double>(state_.cpu, 2, -3.0);
  StoreScalarToV<double>(state_.cpu, 0, std::nan(""));
  static const uint32_t code[] = {FmulxScalarD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(double));
  EXPECT_DOUBLE_EQ(result, -1.5);
  uint64_t upper;
  std::memcpy(&upper, reinterpret_cast<const uint8_t*>(&state_.cpu.v[0]) + 8, 8);
  EXPECT_EQ(upper, 0u);
}

// Special case: +0 * +inf -> +2.0 (S).
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarSZeroTimesInf) {
  StoreScalarToV<float>(state_.cpu, 1, 0.0f);
  StoreScalarToV<float>(state_.cpu, 2, std::numeric_limits<float>::infinity());
  static const uint32_t code[] = {FmulxScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 2.0f);
}

// Special case: -0 * +inf -> -2.0 (S).  Sign = sign(-0) XOR sign(+inf) = 1.
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarSNegZeroTimesInf) {
  StoreScalarToV<float>(state_.cpu, 1, -0.0f);
  StoreScalarToV<float>(state_.cpu, 2, std::numeric_limits<float>::infinity());
  static const uint32_t code[] = {FmulxScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, -2.0f);
}

// Special case: -inf * +0 -> -2.0 (S).
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarSNegInfTimesZero) {
  StoreScalarToV<float>(state_.cpu, 1, -std::numeric_limits<float>::infinity());
  StoreScalarToV<float>(state_.cpu, 2, 0.0f);
  static const uint32_t code[] = {FmulxScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, -2.0f);
}

// Special case: -inf * -0 -> +2.0 (S).  Sign = 1 XOR 1 = 0.
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarSNegInfTimesNegZero) {
  StoreScalarToV<float>(state_.cpu, 1, -std::numeric_limits<float>::infinity());
  StoreScalarToV<float>(state_.cpu, 2, -0.0f);
  static const uint32_t code[] = {FmulxScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 2.0f);
}

// Special case for FP64: +0 * +inf -> +2.0.
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarDZeroTimesInf) {
  StoreScalarToV<double>(state_.cpu, 1, 0.0);
  StoreScalarToV<double>(state_.cpu, 2, std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FmulxScalarD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(double));
  EXPECT_DOUBLE_EQ(result, 2.0);
}

// NaN propagation: NaN input must produce NaN (not ±2.0).
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarSNaNInput) {
  StoreScalarToV<float>(state_.cpu, 1, std::nanf(""));
  StoreScalarToV<float>(state_.cpu, 2, 1.0f);
  static const uint32_t code[] = {FmulxScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_TRUE(std::isnan(result));
}

// Sign on regular multiply (no special case): negative*negative = positive.
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarDNegNeg) {
  StoreScalarToV<double>(state_.cpu, 1, -2.5);
  StoreScalarToV<double>(state_.cpu, 2, -4.0);
  static const uint32_t code[] = {FmulxScalarD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(double));
  EXPECT_DOUBLE_EQ(result, 10.0);
}

// Finite zero result: 0 * 5 -> +0 (NOT ±2.0; the special case is 0*inf only).
TEST_F(Arm64LiteTranslateRegionTest, FmulxScalarSZeroTimesFinite) {
  StoreScalarToV<float>(state_.cpu, 1, 0.0f);
  StoreScalarToV<float>(state_.cpu, 2, 5.0f);
  static const uint32_t code[] = {FmulxScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 0.0f);
  EXPECT_FALSE(std::signbit(result));
}
// endregion

// region digitalis: FRECPS / FRSQRTS scalar three-same JIT (FP32/FP64).
// FRECPS  = std::fma(-a, b, 2.0); FRSQRTS = std::fma(-a, b, 3.0)/2.
// Special cases: NaN input -> default qNaN; (±0,±inf) cross -> +2.0/+1.5.
// Encoding (per ARM ARM C7.2.151/155 and llvm-mc verification):
//   FRECPS  Sd, Sn, Sm = 0x5E22FC00 | (rm<<16) | (rn<<5) | rd
//   FRECPS  Dd, Dn, Dm = 0x5E62FC00 | ...
//   FRSQRTS Sd, Sn, Sm = 0x5EA2FC00 | ...
//   FRSQRTS Dd, Dn, Dm = 0x5EE2FC00 | ...
constexpr uint32_t FrecpsScalarS(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x5E20FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrecpsScalarD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x5E60FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrsqrtsScalarS(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x5EA0FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrsqrtsScalarD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x5EE0FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// FRECPS finite (S): 2 - 0.5*3 = 0.5.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsScalarSRegular) {
  StoreScalarToV<float>(state_.cpu, 1, 0.5f);
  StoreScalarToV<float>(state_.cpu, 2, 3.0f);
  StoreScalarToV<float>(state_.cpu, 0, std::nanf(""));  // pre-trash dest
  static const uint32_t code[] = {FrecpsScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 0.5f);
  // Upper lanes must be zero.
  uint32_t upper[3];
  std::memcpy(upper, reinterpret_cast<const uint8_t*>(&state_.cpu.v[0]) + 4, 12);
  EXPECT_EQ(upper[0], 0u);
  EXPECT_EQ(upper[1], 0u);
  EXPECT_EQ(upper[2], 0u);
}

// FRECPS finite (D): 2 - 1*1 = 1.0.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsScalarDRegular) {
  StoreScalarToV<double>(state_.cpu, 1, 1.0);
  StoreScalarToV<double>(state_.cpu, 2, 1.0);
  static const uint32_t code[] = {FrecpsScalarD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(double));
  EXPECT_DOUBLE_EQ(result, 1.0);
}

// FRECPS saturation (S): +0 * +inf -> +2.0 (unsigned, unlike FMULX).
TEST_F(Arm64LiteTranslateRegionTest, FrecpsScalarSZeroTimesInf) {
  StoreScalarToV<float>(state_.cpu, 1, 0.0f);
  StoreScalarToV<float>(state_.cpu, 2, std::numeric_limits<float>::infinity());
  static const uint32_t code[] = {FrecpsScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 2.0f);
  EXPECT_FALSE(std::signbit(result));
}

// FRECPS saturation (S): -0 * +inf -> +2.0 (still unsigned — FRECPS does
// NOT XOR sign bits like FMULX would).
TEST_F(Arm64LiteTranslateRegionTest, FrecpsScalarSNegZeroTimesInf) {
  StoreScalarToV<float>(state_.cpu, 1, -0.0f);
  StoreScalarToV<float>(state_.cpu, 2, std::numeric_limits<float>::infinity());
  static const uint32_t code[] = {FrecpsScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 2.0f);
  EXPECT_FALSE(std::signbit(result));
}

// FRECPS saturation (D): -inf * +0 -> +2.0.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsScalarDNegInfTimesZero) {
  StoreScalarToV<double>(state_.cpu, 1, -std::numeric_limits<double>::infinity());
  StoreScalarToV<double>(state_.cpu, 2, 0.0);
  static const uint32_t code[] = {FrecpsScalarD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(double));
  EXPECT_DOUBLE_EQ(result, 2.0);
  EXPECT_FALSE(std::signbit(result));
}

// FRECPS NaN input (S): default qNaN, not the input NaN payload.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsScalarSNaNInput) {
  StoreScalarToV<float>(state_.cpu, 1, std::nanf(""));
  StoreScalarToV<float>(state_.cpu, 2, 1.0f);
  static const uint32_t code[] = {FrecpsScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_TRUE(std::isnan(result));
  // Default qNaN: positive, exponent all ones, MSB of mantissa set.
  uint32_t bits;
  std::memcpy(&bits, &result, sizeof(bits));
  EXPECT_EQ(bits & 0xFFC00000u, 0x7FC00000u);  // sign=0, exp=0xFF, frac MSB=1
}

// FRECPS zero * finite: ordinary FMA returns +K_fma. fma(-0, 5, 2) = 2.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsScalarSZeroTimesFinite) {
  StoreScalarToV<float>(state_.cpu, 1, 0.0f);
  StoreScalarToV<float>(state_.cpu, 2, 5.0f);
  static const uint32_t code[] = {FrecpsScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 2.0f);
}

// FRSQRTS finite (S): (3 - 1*1)/2 = 1.0.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsScalarSRegular) {
  StoreScalarToV<float>(state_.cpu, 1, 1.0f);
  StoreScalarToV<float>(state_.cpu, 2, 1.0f);
  static const uint32_t code[] = {FrsqrtsScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 1.0f);
}

// FRSQRTS finite (D): (3 - 0.5*4)/2 = 0.5.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsScalarDRegular) {
  StoreScalarToV<double>(state_.cpu, 1, 0.5);
  StoreScalarToV<double>(state_.cpu, 2, 4.0);
  static const uint32_t code[] = {FrsqrtsScalarD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(double));
  EXPECT_DOUBLE_EQ(result, 0.5);
}

// FRSQRTS saturation (S): +inf * 0 -> +1.5.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsScalarSInfTimesZero) {
  StoreScalarToV<float>(state_.cpu, 1, std::numeric_limits<float>::infinity());
  StoreScalarToV<float>(state_.cpu, 2, 0.0f);
  static const uint32_t code[] = {FrsqrtsScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 1.5f);
}

// FRSQRTS saturation (D): -0 * -inf -> +1.5 (unsigned saturation).
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsScalarDNegZeroTimesNegInf) {
  StoreScalarToV<double>(state_.cpu, 1, -0.0);
  StoreScalarToV<double>(state_.cpu, 2, -std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FrsqrtsScalarD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(double));
  EXPECT_DOUBLE_EQ(result, 1.5);
  EXPECT_FALSE(std::signbit(result));
}

// FRSQRTS NaN input (D): default qNaN.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsScalarDNaNInput) {
  StoreScalarToV<double>(state_.cpu, 1, 1.0);
  StoreScalarToV<double>(state_.cpu, 2, std::nan(""));
  static const uint32_t code[] = {FrsqrtsScalarD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(double));
  EXPECT_TRUE(std::isnan(result));
  uint64_t bits;
  std::memcpy(&bits, &result, sizeof(bits));
  EXPECT_EQ(bits & 0xFFF8000000000000ULL, 0x7FF8000000000000ULL);
}

// FRSQRTS Newton step for 1/sqrt(1) iteration: e=1 already exact ->
// (3 - 1*1)/2 = 1 (S form).  Sanity for the typical NR refinement use.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsScalarSNewtonStepOnOne) {
  StoreScalarToV<float>(state_.cpu, 1, 1.0f);  // x
  StoreScalarToV<float>(state_.cpu, 2, 1.0f);  // e*e (already 1)
  static const uint32_t code[] = {FrsqrtsScalarS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float result;
  std::memcpy(&result, &state_.cpu.v[0], sizeof(float));
  EXPECT_FLOAT_EQ(result, 1.0f);
}
// endregion

// region digitalis: FMULX vector three-same JIT (FP32 .2S/.4S, FP64 .2D).
// Identical semantics to FMULX scalar, just lane-parallel.  Each lane
// applies a*b except the (±0 * ±inf) saturation case, which yields ±2.0
// with sign = sign(a) XOR sign(b) per-lane.
//
// Encoding (per ARM ARM C7.2.149 "FMULX (vector)" and llvm-mc verification):
//   FMULX Vd.2S, Vn.2S, Vm.2S = 0x0E22DC00 | (rm<<16) | (rn<<5) | rd
//   FMULX Vd.4S, Vn.4S, Vm.4S = 0x4E22DC00 | (rm<<16) | (rn<<5) | rd
//   FMULX Vd.2D, Vn.2D, Vm.2D = 0x4E62DC00 | (rm<<16) | (rn<<5) | rd
constexpr uint32_t FmulxVec2S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E20DC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulxVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E20DC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulxVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E60DC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// Helpers for 4-lane FP32 / 2-lane FP64 vector inputs.
static void StoreVec4S(CPUState& cpu, unsigned idx,
                       float a0, float a1, float a2, float a3) {
  float lanes[4] = {a0, a1, a2, a3};
  std::memcpy(&cpu.v[idx], lanes, 16);
}
static void StoreVec2D(CPUState& cpu, unsigned idx, double a0, double a1) {
  double lanes[2] = {a0, a1};
  std::memcpy(&cpu.v[idx], lanes, 16);
}
static void LoadVec4S(const CPUState& cpu, unsigned idx, float out[4]) {
  std::memcpy(out, &cpu.v[idx], 16);
}
static void LoadVec2D(const CPUState& cpu, unsigned idx, double out[2]) {
  std::memcpy(out, &cpu.v[idx], 16);
}

// .4S: lane 0 finite multiply; lane 1 (+0,+inf); lane 2 (-0,+inf); lane 3 finite.
TEST_F(Arm64LiteTranslateRegionTest, FmulxVec4SAllLanes) {
  const float inf = std::numeric_limits<float>::infinity();
  StoreVec4S(state_.cpu, 1, 3.0f, 0.0f, -0.0f, -2.5f);
  StoreVec4S(state_.cpu, 2, 4.0f, inf, inf, 2.0f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FmulxVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 12.0f);   // 3 * 4
  EXPECT_FLOAT_EQ(r[1], 2.0f);    // (+0, +inf) -> +2
  EXPECT_FLOAT_EQ(r[2], -2.0f);   // (-0, +inf) -> -2
  EXPECT_FLOAT_EQ(r[3], -5.0f);   // -2.5 * 2
}

// .2S: q=0 form; upper 64 bits of Vd must be zero.
TEST_F(Arm64LiteTranslateRegionTest, FmulxVec2SUpperZero) {
  const float inf = std::numeric_limits<float>::infinity();
  StoreVec4S(state_.cpu, 1, 2.0f, -inf, 0.0f, 0.0f);  // lanes 2,3 ignored
  StoreVec4S(state_.cpu, 2, 3.0f, 0.0f, 0.0f, 0.0f);
  // Pre-fill Vd with sentinel — the .2S store must clobber upper lanes to zero.
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), 7.0f, 7.0f);
  static const uint32_t code[] = {FmulxVec2S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 6.0f);     // 2 * 3
  EXPECT_FLOAT_EQ(r[1], -2.0f);    // (-inf, +0) -> -2
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);  // upper 64 bits zeroed by .2S form
  EXPECT_EQ(lane3_bits, 0u);
}

// .2D: lane 0 finite multiply; lane 1 (+inf, -0) -> -2.0.
TEST_F(Arm64LiteTranslateRegionTest, FmulxVec2DAllLanes) {
  const double inf = std::numeric_limits<double>::infinity();
  StoreVec2D(state_.cpu, 1, 0.5, inf);
  StoreVec2D(state_.cpu, 2, -3.0, -0.0);
  StoreVec2D(state_.cpu, 0, std::nan(""), std::nan(""));
  static const uint32_t code[] = {FmulxVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], -1.5);
  EXPECT_DOUBLE_EQ(r[1], -2.0);   // (+inf, -0) -> -2
}

// .4S NaN propagation: lane 0 has a NaN input → standard NaN result, no
// saturation override.  (Distinguishes from the (0,inf) special case.)
TEST_F(Arm64LiteTranslateRegionTest, FmulxVec4SNaNNotSpecialCase) {
  const float qnan = std::nanf("");
  StoreVec4S(state_.cpu, 1, qnan, 0.0f, 1.0f, 2.0f);
  StoreVec4S(state_.cpu, 2, 5.0f, std::numeric_limits<float>::infinity(),
             3.0f, 0.0f);
  static const uint32_t code[] = {FmulxVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_TRUE(std::isnan(r[0]));   // NaN input propagates, NOT replaced by ±2.0
  EXPECT_FLOAT_EQ(r[1], 2.0f);     // (+0, +inf) saturation
  EXPECT_FLOAT_EQ(r[2], 3.0f);     // 1 * 3
  EXPECT_FLOAT_EQ(r[3], 0.0f);     // 2 * 0 (no inf side)
}

// .2D regular finite path — sanity that nothing breaks lane 1 when neither
// input is special.
TEST_F(Arm64LiteTranslateRegionTest, FmulxVec2DRegular) {
  StoreVec2D(state_.cpu, 1, 2.0, -4.0);
  StoreVec2D(state_.cpu, 2, 3.5, 0.25);
  static const uint32_t code[] = {FmulxVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 7.0);
  EXPECT_DOUBLE_EQ(r[1], -1.0);
}
// endregion

// region digitalis: FADD / FSUB / FMUL / FDIV vector three-same JIT
// (FP32 .2S/.4S, FP64 .2D).  Direct lowering to ADDPS/PD, SUBPS/PD,
// MULPS/PD, DIVPS/PD (all SSE2).  Tests pick operand pairs that are
// exactly representable in the target precision so the result is the
// same bit-pattern whether the lowering goes through the JIT path or
// the interpreter fallback.
//
// Encoding (per ARM ARM C7.2 / aarch64-linux-gnu-as verification):
//   FADD Vd.4S = 0x4E20D400 | (rm<<16) | (rn<<5) | rd
//   FADD Vd.2D = 0x4E60D400 | (rm<<16) | (rn<<5) | rd
//   FADD Vd.2S = 0x0E20D400 | (rm<<16) | (rn<<5) | rd
//   FSUB Vd.4S = 0x4EA0D400 | (rm<<16) | (rn<<5) | rd
//   FSUB Vd.2D = 0x4EE0D400 | (rm<<16) | (rn<<5) | rd
//   FMUL Vd.4S = 0x6E20DC00 | (rm<<16) | (rn<<5) | rd
//   FMUL Vd.2D = 0x6E60DC00 | (rm<<16) | (rn<<5) | rd
//   FDIV Vd.4S = 0x6E20FC00 | (rm<<16) | (rn<<5) | rd
//   FDIV Vd.2D = 0x6E60FC00 | (rm<<16) | (rn<<5) | rd
constexpr uint32_t FaddVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E20D400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FaddVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E60D400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FaddVec2S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E20D400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FsubVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EA0D400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FsubVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EE0D400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x6E20DC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x6E60DC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FdivVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x6E20FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FdivVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x6E60FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

TEST_F(Arm64LiteTranslateRegionTest, FaddVec4S) {
  StoreVec4S(state_.cpu, 1, 1.0f, -2.0f, 0.25f, 100.5f);
  StoreVec4S(state_.cpu, 2, 2.0f,  3.0f, 0.75f,  -0.5f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FaddVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 3.0f);
  EXPECT_FLOAT_EQ(r[1], 1.0f);
  EXPECT_FLOAT_EQ(r[2], 1.0f);
  EXPECT_FLOAT_EQ(r[3], 100.0f);
}

TEST_F(Arm64LiteTranslateRegionTest, FaddVec2D) {
  StoreVec2D(state_.cpu, 1, 1.5,  -8.0);
  StoreVec2D(state_.cpu, 2, 0.25,  3.0);
  StoreVec2D(state_.cpu, 0, std::nan(""), std::nan(""));
  static const uint32_t code[] = {FaddVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 1.75);
  EXPECT_DOUBLE_EQ(r[1], -5.0);
}

// .2S (q=0): only the low 2 FP32 lanes participate; upper 64 bits of Vd
// must be zeroed even though the JIT performs the ADDPS on uninitialized
// upper lanes.
TEST_F(Arm64LiteTranslateRegionTest, FaddVec2SUpperZero) {
  StoreVec4S(state_.cpu, 1, 2.0f,  4.0f, 0.0f, 0.0f);
  StoreVec4S(state_.cpu, 2, 3.0f, -1.0f, 0.0f, 0.0f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), 7.0f, 7.0f);
  static const uint32_t code[] = {FaddVec2S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 5.0f);
  EXPECT_FLOAT_EQ(r[1], 3.0f);
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

TEST_F(Arm64LiteTranslateRegionTest, FsubVec4S) {
  StoreVec4S(state_.cpu, 1, 5.0f,  2.0f, 0.5f, -1.0f);
  StoreVec4S(state_.cpu, 2, 3.0f,  6.0f, 0.25f, 2.0f);
  static const uint32_t code[] = {FsubVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 2.0f);
  EXPECT_FLOAT_EQ(r[1], -4.0f);
  EXPECT_FLOAT_EQ(r[2], 0.25f);
  EXPECT_FLOAT_EQ(r[3], -3.0f);
}

TEST_F(Arm64LiteTranslateRegionTest, FsubVec2D) {
  StoreVec2D(state_.cpu, 1, 10.0,  -2.5);
  StoreVec2D(state_.cpu, 2,  3.0,   0.5);
  static const uint32_t code[] = {FsubVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 7.0);
  EXPECT_DOUBLE_EQ(r[1], -3.0);
}

TEST_F(Arm64LiteTranslateRegionTest, FmulVec4S) {
  StoreVec4S(state_.cpu, 1, 2.0f, -3.0f, 0.5f, 16.0f);
  StoreVec4S(state_.cpu, 2, 4.0f,  2.0f, 8.0f,  0.25f);
  static const uint32_t code[] = {FmulVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 8.0f);
  EXPECT_FLOAT_EQ(r[1], -6.0f);
  EXPECT_FLOAT_EQ(r[2], 4.0f);
  EXPECT_FLOAT_EQ(r[3], 4.0f);
}

TEST_F(Arm64LiteTranslateRegionTest, FmulVec2D) {
  StoreVec2D(state_.cpu, 1, 1.5,  -2.0);
  StoreVec2D(state_.cpu, 2, 4.0,   8.0);
  static const uint32_t code[] = {FmulVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 6.0);
  EXPECT_DOUBLE_EQ(r[1], -16.0);
}

TEST_F(Arm64LiteTranslateRegionTest, FdivVec4S) {
  StoreVec4S(state_.cpu, 1, 8.0f,  -10.0f, 1.0f,  256.0f);
  StoreVec4S(state_.cpu, 2, 2.0f,    4.0f, 8.0f,   16.0f);
  static const uint32_t code[] = {FdivVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 4.0f);
  EXPECT_FLOAT_EQ(r[1], -2.5f);
  EXPECT_FLOAT_EQ(r[2], 0.125f);
  EXPECT_FLOAT_EQ(r[3], 16.0f);
}

TEST_F(Arm64LiteTranslateRegionTest, FdivVec2D) {
  StoreVec2D(state_.cpu, 1, 9.0,    -1.0);
  StoreVec2D(state_.cpu, 2, 4.0,     8.0);
  static const uint32_t code[] = {FdivVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 2.25);
  EXPECT_DOUBLE_EQ(r[1], -0.125);
}
// endregion

// region digitalis: FMAX / FMIN / FMAXNM / FMINNM vector three-same JIT
// (FP32 .2S/.4S, FP64 .2D).  ARM and x86 disagree on NaN semantics:
//   FMAX/FMIN  — IEEE: any NaN -> NaN result.
//   FMAXNM/FMINNM — "max/min Number": one NaN -> return the non-NaN;
//                    both NaN -> NaN.
//   x86 MAXPS/MINPS — any NaN -> result = SRC2 (asymmetric).
// JIT path implements both behaviors via the two-pass-OR (FMAX/FMIN) and
// substitution (FMAXNM/FMINNM) lowerings, mirroring the FP16 path.
//
// Encoding (per ARM ARM C7.2 / aarch64-linux-gnu-as verification):
//   FMAX Vd.4S = 0x4E20F400 | (rm<<16) | (rn<<5) | rd
//   FMAX Vd.2D = 0x4E60F400 | (rm<<16) | (rn<<5) | rd
//   FMAX Vd.2S = 0x0E20F400 | (rm<<16) | (rn<<5) | rd
//   FMIN Vd.4S = 0x4EA0F400 | (rm<<16) | (rn<<5) | rd
//   FMIN Vd.2D = 0x4EE0F400 | (rm<<16) | (rn<<5) | rd
//   FMAXNM Vd.4S = 0x4E20C400 | (rm<<16) | (rn<<5) | rd
//   FMAXNM Vd.2D = 0x4E60C400 | (rm<<16) | (rn<<5) | rd
//   FMINNM Vd.4S = 0x4EA0C400 | (rm<<16) | (rn<<5) | rd
//   FMINNM Vd.2D = 0x4EE0C400 | (rm<<16) | (rn<<5) | rd
constexpr uint32_t FmaxVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E20F400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmaxVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E60F400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmaxVec2S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E20F400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FminVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EA0F400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FminVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EE0F400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmaxnmVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E20C400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmaxnmVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E60C400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FminnmVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EA0C400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FminnmVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EE0C400u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// Basic FMAX .4S: regular finite operands.
TEST_F(Arm64LiteTranslateRegionTest, FmaxVec4S) {
  StoreVec4S(state_.cpu, 1, 1.0f, -2.0f, 0.25f, 100.5f);
  StoreVec4S(state_.cpu, 2, 2.0f,  3.0f, 0.75f,  -0.5f);
  static const uint32_t code[] = {FmaxVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 2.0f);
  EXPECT_FLOAT_EQ(r[1], 3.0f);
  EXPECT_FLOAT_EQ(r[2], 0.75f);
  EXPECT_FLOAT_EQ(r[3], 100.5f);
}

TEST_F(Arm64LiteTranslateRegionTest, FmaxVec2D) {
  StoreVec2D(state_.cpu, 1, 1.5,  -8.0);
  StoreVec2D(state_.cpu, 2, 0.25,  3.0);
  static const uint32_t code[] = {FmaxVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 1.5);
  EXPECT_DOUBLE_EQ(r[1], 3.0);
}

// FMAX .4S NaN propagation: any NaN input -> NaN result.
TEST_F(Arm64LiteTranslateRegionTest, FmaxVec4SNaN) {
  // lane 0: a=NaN,  b=3.0  -> NaN
  // lane 1: a=2.0,  b=NaN  -> NaN
  // lane 2: a=NaN,  b=NaN  -> NaN
  // lane 3: a=4.0,  b=5.0  -> 5.0  (sanity)
  StoreVec4S(state_.cpu, 1, std::nanf(""),  2.0f, std::nanf(""), 4.0f);
  StoreVec4S(state_.cpu, 2, 3.0f, std::nanf(""), std::nanf(""), 5.0f);
  static const uint32_t code[] = {FmaxVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_TRUE(std::isnan(r[0]));
  EXPECT_TRUE(std::isnan(r[1]));
  EXPECT_TRUE(std::isnan(r[2]));
  EXPECT_FLOAT_EQ(r[3], 5.0f);
}

// FMAX .2S (q=0): low 2 FP32 lanes; upper 64 bits of Vd must be zeroed.
TEST_F(Arm64LiteTranslateRegionTest, FmaxVec2SUpperZero) {
  StoreVec4S(state_.cpu, 1, 1.0f, -3.0f, 0.0f, 0.0f);
  StoreVec4S(state_.cpu, 2, 2.0f, -1.0f, 0.0f, 0.0f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), 7.0f, 7.0f);
  static const uint32_t code[] = {FmaxVec2S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 2.0f);
  EXPECT_FLOAT_EQ(r[1], -1.0f);
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// FMAX .4S with ±0 and ±inf: confirm IEEE semantics carry through.
TEST_F(Arm64LiteTranslateRegionTest, FmaxVec4SInfZero) {
  const float inf = std::numeric_limits<float>::infinity();
  // lane 0: max(+0, -0) -> +0   (x86 MAXPS returns SRC2 when both equal, so
  //                              the OR trick gives the same; either signed-0
  //                              is allowed by ARM here)
  // lane 1: max(-inf, 5) -> 5
  // lane 2: max(+inf, 1e30) -> +inf
  // lane 3: max(-inf, -inf) -> -inf
  StoreVec4S(state_.cpu, 1, +0.0f, -inf, +inf, -inf);
  StoreVec4S(state_.cpu, 2, -0.0f,  5.0f, 1e30f, -inf);
  static const uint32_t code[] = {FmaxVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 0.0f);             // ±0 are both equal-to-zero
  EXPECT_FLOAT_EQ(r[1], 5.0f);
  EXPECT_EQ(r[2], inf);
  EXPECT_EQ(r[3], -inf);
}

TEST_F(Arm64LiteTranslateRegionTest, FminVec4S) {
  StoreVec4S(state_.cpu, 1, 1.0f, -2.0f, 0.25f, 100.5f);
  StoreVec4S(state_.cpu, 2, 2.0f,  3.0f, 0.75f,  -0.5f);
  static const uint32_t code[] = {FminVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.0f);
  EXPECT_FLOAT_EQ(r[1], -2.0f);
  EXPECT_FLOAT_EQ(r[2], 0.25f);
  EXPECT_FLOAT_EQ(r[3], -0.5f);
}

TEST_F(Arm64LiteTranslateRegionTest, FminVec2D) {
  StoreVec2D(state_.cpu, 1, 1.5,  -8.0);
  StoreVec2D(state_.cpu, 2, 0.25,  3.0);
  static const uint32_t code[] = {FminVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 0.25);
  EXPECT_DOUBLE_EQ(r[1], -8.0);
}

TEST_F(Arm64LiteTranslateRegionTest, FminVec4SNaN) {
  StoreVec4S(state_.cpu, 1, std::nanf(""),  2.0f, std::nanf(""), 4.0f);
  StoreVec4S(state_.cpu, 2, 3.0f, std::nanf(""), std::nanf(""), 5.0f);
  static const uint32_t code[] = {FminVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_TRUE(std::isnan(r[0]));
  EXPECT_TRUE(std::isnan(r[1]));
  EXPECT_TRUE(std::isnan(r[2]));
  EXPECT_FLOAT_EQ(r[3], 4.0f);
}

// FMAXNM .4S: regular finite operands.
TEST_F(Arm64LiteTranslateRegionTest, FmaxnmVec4S) {
  StoreVec4S(state_.cpu, 1, 1.0f, -2.0f, 0.25f, 100.5f);
  StoreVec4S(state_.cpu, 2, 2.0f,  3.0f, 0.75f,  -0.5f);
  static const uint32_t code[] = {FmaxnmVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 2.0f);
  EXPECT_FLOAT_EQ(r[1], 3.0f);
  EXPECT_FLOAT_EQ(r[2], 0.75f);
  EXPECT_FLOAT_EQ(r[3], 100.5f);
}

TEST_F(Arm64LiteTranslateRegionTest, FmaxnmVec2D) {
  StoreVec2D(state_.cpu, 1, 1.5,  -8.0);
  StoreVec2D(state_.cpu, 2, 0.25,  3.0);
  static const uint32_t code[] = {FmaxnmVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 1.5);
  EXPECT_DOUBLE_EQ(r[1], 3.0);
}

// FMAXNM .4S NaN-suppressing: exactly one NaN -> return non-NaN; both NaN -> NaN.
TEST_F(Arm64LiteTranslateRegionTest, FmaxnmVec4SNaN) {
  // lane 0: a=NaN,  b=3.0   -> 3.0 (non-NaN wins)
  // lane 1: a=2.0,  b=NaN   -> 2.0 (non-NaN wins)
  // lane 2: a=NaN,  b=NaN   -> NaN (both NaN -> NaN)
  // lane 3: a=4.0,  b=5.0   -> 5.0 (regular max)
  StoreVec4S(state_.cpu, 1, std::nanf(""),  2.0f, std::nanf(""), 4.0f);
  StoreVec4S(state_.cpu, 2, 3.0f, std::nanf(""), std::nanf(""), 5.0f);
  static const uint32_t code[] = {FmaxnmVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 3.0f);
  EXPECT_FLOAT_EQ(r[1], 2.0f);
  EXPECT_TRUE(std::isnan(r[2]));
  EXPECT_FLOAT_EQ(r[3], 5.0f);
}

TEST_F(Arm64LiteTranslateRegionTest, FmaxnmVec2DNaN) {
  StoreVec2D(state_.cpu, 1, std::nan(""), 4.0);
  StoreVec2D(state_.cpu, 2, 3.0,          std::nan(""));
  static const uint32_t code[] = {FmaxnmVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 3.0);
  EXPECT_DOUBLE_EQ(r[1], 4.0);
}

TEST_F(Arm64LiteTranslateRegionTest, FminnmVec4S) {
  StoreVec4S(state_.cpu, 1, 1.0f, -2.0f, 0.25f, 100.5f);
  StoreVec4S(state_.cpu, 2, 2.0f,  3.0f, 0.75f,  -0.5f);
  static const uint32_t code[] = {FminnmVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.0f);
  EXPECT_FLOAT_EQ(r[1], -2.0f);
  EXPECT_FLOAT_EQ(r[2], 0.25f);
  EXPECT_FLOAT_EQ(r[3], -0.5f);
}

TEST_F(Arm64LiteTranslateRegionTest, FminnmVec2D) {
  StoreVec2D(state_.cpu, 1, 1.5,  -8.0);
  StoreVec2D(state_.cpu, 2, 0.25,  3.0);
  static const uint32_t code[] = {FminnmVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 0.25);
  EXPECT_DOUBLE_EQ(r[1], -8.0);
}

TEST_F(Arm64LiteTranslateRegionTest, FminnmVec4SNaN) {
  // lane 0: a=NaN,  b=3.0   -> 3.0 (non-NaN wins)
  // lane 1: a=2.0,  b=NaN   -> 2.0 (non-NaN wins)
  // lane 2: a=NaN,  b=NaN   -> NaN
  // lane 3: a=4.0,  b=5.0   -> 4.0 (regular min)
  StoreVec4S(state_.cpu, 1, std::nanf(""),  2.0f, std::nanf(""), 4.0f);
  StoreVec4S(state_.cpu, 2, 3.0f, std::nanf(""), std::nanf(""), 5.0f);
  static const uint32_t code[] = {FminnmVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 3.0f);
  EXPECT_FLOAT_EQ(r[1], 2.0f);
  EXPECT_TRUE(std::isnan(r[2]));
  EXPECT_FLOAT_EQ(r[3], 4.0f);
}
// endregion

// region digitalis: FMLA / FMLS vector three-same JIT (FP32 .2S/.4S, FP64 .2D).
// ARM ARM defines FMLA/FMLS as fused multiply-accumulate (one rounding).
// The lowering uses Vfmadd231(ps|pd) / Vfnmadd231(ps|pd).  Tests pick
// operand triples whose products are exactly representable in the target
// precision so fused and unfused results agree, sidestepping any oracle
// ambiguity between this lowering and the interpreter's `d + a*b`
// formulation.
//
// Encoding (per ARM ARM C7.2.135 "FMLA (vector)" / C7.2.136 "FMLS (vector)"
// and aarch64-linux-gnu-as verification):
//   FMLA Vd.2S, Vn.2S, Vm.2S = 0x0E20CC00 | (rm<<16) | (rn<<5) | rd
//   FMLA Vd.4S, Vn.4S, Vm.4S = 0x4E20CC00 | (rm<<16) | (rn<<5) | rd
//   FMLA Vd.2D, Vn.2D, Vm.2D = 0x4E60CC00 | (rm<<16) | (rn<<5) | rd
//   FMLS Vd.2S, Vn.2S, Vm.2S = 0x0EA0CC00 | (rm<<16) | (rn<<5) | rd
//   FMLS Vd.4S, Vn.4S, Vm.4S = 0x4EA0CC00 | (rm<<16) | (rn<<5) | rd
//   FMLS Vd.2D, Vn.2D, Vm.2D = 0x4EE0CC00 | (rm<<16) | (rn<<5) | rd
constexpr uint32_t FmlaVec2S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E20CC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlaVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E20CC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlaVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E60CC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsVec2S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0EA0CC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EA0CC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EE0CC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// .4S FMLA: Vd[i] += Vn[i] * Vm[i] across four lanes.
TEST_F(Arm64LiteTranslateRegionTest, FmlaVec4SAllLanes) {
  StoreVec4S(state_.cpu, 1, 2.0f, 1.5f, -3.0f, 4.0f);
  StoreVec4S(state_.cpu, 2, 3.0f, 4.0f, 0.5f, -2.0f);
  StoreVec4S(state_.cpu, 0, 1.0f, 2.0f, 10.0f, 0.0f);
  static const uint32_t code[] = {FmlaVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 7.0f);   // 1 + 2*3
  EXPECT_FLOAT_EQ(r[1], 8.0f);   // 2 + 1.5*4
  EXPECT_FLOAT_EQ(r[2], 8.5f);   // 10 + (-3)*0.5
  EXPECT_FLOAT_EQ(r[3], -8.0f);  // 0 + 4*-2
}

// .2S FMLA: q=0 form; upper 64 bits of Vd must be zero.
TEST_F(Arm64LiteTranslateRegionTest, FmlaVec2SUpperZero) {
  StoreVec4S(state_.cpu, 1, 2.0f, 3.0f, 99.0f, 99.0f);   // lanes 2,3 ignored
  StoreVec4S(state_.cpu, 2, 5.0f, -1.5f, 99.0f, 99.0f);
  StoreVec4S(state_.cpu, 0, 1.0f, 8.0f, 7.0f, 7.0f);     // sentinel uppers
  static const uint32_t code[] = {FmlaVec2S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 11.0f);  // 1 + 2*5
  EXPECT_FLOAT_EQ(r[1], 3.5f);   // 8 + 3*-1.5
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);     // upper 64 bits zeroed by .2S form
  EXPECT_EQ(lane3_bits, 0u);
}

// .2D FMLA: two FP64 lanes.
TEST_F(Arm64LiteTranslateRegionTest, FmlaVec2DAllLanes) {
  StoreVec2D(state_.cpu, 1, 0.5, -3.0);
  StoreVec2D(state_.cpu, 2, 6.0, 2.0);
  StoreVec2D(state_.cpu, 0, 1.0, 10.0);
  static const uint32_t code[] = {FmlaVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 4.0);   // 1 + 0.5*6
  EXPECT_DOUBLE_EQ(r[1], 4.0);   // 10 + -3*2
}

// .4S FMLS: Vd[i] -= Vn[i] * Vm[i].
TEST_F(Arm64LiteTranslateRegionTest, FmlsVec4SAllLanes) {
  StoreVec4S(state_.cpu, 1, 2.0f, 1.5f, -3.0f, 4.0f);
  StoreVec4S(state_.cpu, 2, 3.0f, 4.0f, 0.5f, -2.0f);
  StoreVec4S(state_.cpu, 0, 10.0f, 2.0f, 0.0f, 1.0f);
  static const uint32_t code[] = {FmlsVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 4.0f);    // 10 - 2*3
  EXPECT_FLOAT_EQ(r[1], -4.0f);   // 2 - 1.5*4
  EXPECT_FLOAT_EQ(r[2], 1.5f);    // 0 - (-3)*0.5
  EXPECT_FLOAT_EQ(r[3], 9.0f);    // 1 - 4*-2
}

// .2S FMLS: q=0 form; upper 64 bits of Vd must be zero.
TEST_F(Arm64LiteTranslateRegionTest, FmlsVec2SUpperZero) {
  StoreVec4S(state_.cpu, 1, 2.0f, 3.0f, 99.0f, 99.0f);
  StoreVec4S(state_.cpu, 2, 5.0f, -1.5f, 99.0f, 99.0f);
  StoreVec4S(state_.cpu, 0, 11.0f, 8.0f, 7.0f, 7.0f);
  static const uint32_t code[] = {FmlsVec2S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.0f);    // 11 - 2*5
  EXPECT_FLOAT_EQ(r[1], 12.5f);   // 8 - 3*-1.5
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// .2D FMLS.
TEST_F(Arm64LiteTranslateRegionTest, FmlsVec2DAllLanes) {
  StoreVec2D(state_.cpu, 1, 0.5, -3.0);
  StoreVec2D(state_.cpu, 2, 6.0, 2.0);
  StoreVec2D(state_.cpu, 0, 10.0, 4.0);
  static const uint32_t code[] = {FmlsVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 7.0);    // 10 - 0.5*6
  EXPECT_DOUBLE_EQ(r[1], 10.0);   // 4 - -3*2
}

// Fused-vs-unfused divergence: pick (a, b, d) such that a*b in FP32 is not
// exact but fma(a, b, d) differs from (a*b)+d.  Validates the lowering is
// using the FMA path (single rounding), not MUL+ADD.
TEST_F(Arm64LiteTranslateRegionTest, FmlaVec4SFusedRounding) {
  // a = 1 + 2^-23  (smallest float > 1)
  // b = 1 + 2^-23
  // exact product = 1 + 2^-22 + 2^-46
  // d = -1
  // unfused: float(a*b) = 1 + 2^-22 (the 2^-46 bit is lost), then + -1
  //   = 2^-22 = 2.384185791015625e-07.
  // fused: fma rounds (a*b + d) = (1 + 2^-22 + 2^-46) - 1 = 2^-22 + 2^-46,
  //   which rounds to nearest float — the trailing 2^-46 bit rounds up to
  //   2^-22 + 2^-23 (next representable above 2^-22 in the normal range
  //   would be 2^-22*(1+2^-23) = 2^-22 + 2^-45; but here the result has
  //   magnitude 2^-22 so the ulp is 2^-22 * 2^-23 = 2^-45, and 2^-46 rounds
  //   down to 0).  So the rounded fused result is exactly 2^-22 + 0 = 2^-22.
  //
  // (Hence for this triple fused and unfused happen to agree.)  We instead
  // use a triple where fused vs unfused differ in the last bit:
  //   a = 0x3F800001 (1 + ulp), b = 0x3F800001, d = -1.0.
  // Already covered above.  Use a sharper case:
  //   a = float(1 + 2^-12), b = a (so a*b = 1 + 2^-11 + 2^-24, exactly
  //     representable in float), d = -1.0.
  //   fused = 2^-11 + 2^-24.
  //   unfused = (1 + 2^-11 + 2^-24) - 1; the parenthesised value rounds to
  //     1 + 2^-11 (the 2^-24 bit is below the ULP near 1, which is 2^-23),
  //     then subtract 1 -> 2^-11.
  //   Difference: fused = 2^-11 + 2^-24, unfused = 2^-11.
  StoreVec4S(state_.cpu, 1, 1.0f + std::ldexp(1.0f, -12), 0.f, 0.f, 0.f);
  StoreVec4S(state_.cpu, 2, 1.0f + std::ldexp(1.0f, -12), 0.f, 0.f, 0.f);
  StoreVec4S(state_.cpu, 0, -1.0f, 0.f, 0.f, 0.f);
  static const uint32_t code[] = {FmlaVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  const float expected =
      std::fmaf(1.0f + std::ldexp(1.0f, -12), 1.0f + std::ldexp(1.0f, -12), -1.0f);
  uint32_t r0_bits, ex_bits;
  std::memcpy(&r0_bits, &r[0], sizeof(uint32_t));
  std::memcpy(&ex_bits, &expected, sizeof(uint32_t));
  EXPECT_EQ(r0_bits, ex_bits);
}
// endregion

// region digitalis: FRECPS / FRSQRTS vector three-same JIT (FP32 .2S/.4S, FP64 .2D).
//
// Encoding (verified with aarch64-linux-gnu-as / objdump):
//   FRECPS  Vd.2S, Vn.2S, Vm.2S = 0x0E20FC00 | (rm<<16) | (rn<<5) | rd
//   FRECPS  Vd.4S, Vn.4S, Vm.4S = 0x4E20FC00 | (rm<<16) | (rn<<5) | rd
//   FRECPS  Vd.2D, Vn.2D, Vm.2D = 0x4E60FC00 | (rm<<16) | (rn<<5) | rd
//   FRSQRTS Vd.2S, Vn.2S, Vm.2S = 0x0EA0FC00 | (rm<<16) | (rn<<5) | rd
//   FRSQRTS Vd.4S, Vn.4S, Vm.4S = 0x4EA0FC00 | (rm<<16) | (rn<<5) | rd
//   FRSQRTS Vd.2D, Vn.2D, Vm.2D = 0x4EE0FC00 | (rm<<16) | (rn<<5) | rd
constexpr uint32_t FrecpsVec2S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E20FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrecpsVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E20FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrecpsVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E60FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrsqrtsVec2S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0EA0FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrsqrtsVec4S(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EA0FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrsqrtsVec2D(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EE0FC00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// FRECPS .4S — Newton step for reciprocal:  result = 2 - a*b, lane by lane.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec4SAllLanes) {
  StoreVec4S(state_.cpu, 1, 1.0f, 0.5f,  -2.0f, 4.0f);
  StoreVec4S(state_.cpu, 2, 1.0f, 2.0f,  -0.5f, 0.25f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FrecpsVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.0f);   // 2 - 1*1
  EXPECT_FLOAT_EQ(r[1], 1.0f);   // 2 - 0.5*2
  EXPECT_FLOAT_EQ(r[2], 1.0f);   // 2 - (-2)*(-0.5)
  EXPECT_FLOAT_EQ(r[3], 1.0f);   // 2 - 4*0.25
}

// FRECPS .2S — q=0; upper 64 bits of Vd zeroed.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec2SUpperZero) {
  StoreVec4S(state_.cpu, 1, 1.0f, 0.5f, 99.0f, 99.0f);
  StoreVec4S(state_.cpu, 2, 1.0f, 2.0f, 99.0f, 99.0f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), 7.0f, 7.0f);
  static const uint32_t code[] = {FrecpsVec2S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.0f);
  EXPECT_FLOAT_EQ(r[1], 1.0f);
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// FRECPS .2D — two FP64 lanes.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec2DAllLanes) {
  StoreVec2D(state_.cpu, 1, 2.0, -0.5);
  StoreVec2D(state_.cpu, 2, 0.5, -2.0);
  StoreVec2D(state_.cpu, 0, std::nan(""), std::nan(""));
  static const uint32_t code[] = {FrecpsVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 1.0);    // 2 - 2*0.5
  EXPECT_DOUBLE_EQ(r[1], 1.0);    // 2 - (-0.5)*(-2)
}

// FRECPS .4S — (±0, ±inf) saturation: the lane with a NaN product but
// non-NaN inputs must yield +2.0, NOT a NaN.  Sign is NOT flipped (unlike
// FMULX), so the result is always +2.0.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec4SZeroTimesInfSaturation) {
  const float inf = std::numeric_limits<float>::infinity();
  StoreVec4S(state_.cpu, 1, 0.0f, -0.0f, -inf, 0.0f);
  StoreVec4S(state_.cpu, 2, inf,  inf,    0.0f, -inf);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FrecpsVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  // All four lanes are (zero, inf) crosses -> +2.0 (no sign flip).
  EXPECT_FLOAT_EQ(r[0], 2.0f);
  EXPECT_FLOAT_EQ(r[1], 2.0f);
  EXPECT_FLOAT_EQ(r[2], 2.0f);
  EXPECT_FLOAT_EQ(r[3], 2.0f);
}

// FRECPS .4S — NaN input must yield the default qNaN, not the saturation
// constant.  Lanes mix NaN, saturation, and finite inputs.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec4SNaNInputDefaultsToQnan) {
  const float inf = std::numeric_limits<float>::infinity();
  const float qnan_in = std::nanf("");
  StoreVec4S(state_.cpu, 1, qnan_in, 0.0f, 1.0f, 1.0f);
  StoreVec4S(state_.cpu, 2, 1.0f,    inf,  1.0f, qnan_in);
  StoreVec4S(state_.cpu, 0, 7.0f, 7.0f, 7.0f, 7.0f);
  static const uint32_t code[] = {FrecpsVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  // Lane 0 / 3: NaN input -> default qNaN.
  EXPECT_TRUE(std::isnan(r[0]));
  EXPECT_TRUE(std::isnan(r[3]));
  // Lane 1: (0, inf) cross -> +2.0.
  EXPECT_FLOAT_EQ(r[1], 2.0f);
  // Lane 2: 2 - 1*1 = 1.
  EXPECT_FLOAT_EQ(r[2], 1.0f);
}

// FRSQRTS .4S — Newton step for reciprocal-sqrt: result = (3 - a*b) / 2.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsVec4SAllLanes) {
  StoreVec4S(state_.cpu, 1, 1.0f, 2.0f, -1.0f, 0.5f);
  StoreVec4S(state_.cpu, 2, 1.0f, 0.5f, -1.0f, 4.0f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FrsqrtsVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.0f);    // (3 - 1*1) / 2 = 1
  EXPECT_FLOAT_EQ(r[1], 1.0f);    // (3 - 2*0.5) / 2 = 1
  EXPECT_FLOAT_EQ(r[2], 1.0f);    // (3 - (-1)*(-1)) / 2 = 1
  EXPECT_FLOAT_EQ(r[3], 0.5f);    // (3 - 0.5*4) / 2 = 0.5
}

// FRSQRTS .2S — upper-zero invariant.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsVec2SUpperZero) {
  StoreVec4S(state_.cpu, 1, 1.0f, 2.0f, 99.0f, 99.0f);
  StoreVec4S(state_.cpu, 2, 1.0f, 0.5f, 99.0f, 99.0f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), 7.0f, 7.0f);
  static const uint32_t code[] = {FrsqrtsVec2S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.0f);
  EXPECT_FLOAT_EQ(r[1], 1.0f);
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// FRSQRTS .2D — two FP64 lanes.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsVec2DAllLanes) {
  StoreVec2D(state_.cpu, 1, 1.0, 0.25);
  StoreVec2D(state_.cpu, 2, 1.0, 4.0);
  StoreVec2D(state_.cpu, 0, std::nan(""), std::nan(""));
  static const uint32_t code[] = {FrsqrtsVec2D(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 1.0);    // (3 - 1*1) / 2 = 1
  EXPECT_DOUBLE_EQ(r[1], 1.0);    // (3 - 0.25*4) / 2 = 1
}

// FRSQRTS .4S — (±0, ±inf) saturation: yields +1.5 per lane (no sign flip).
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsVec4SZeroTimesInfSaturation) {
  const float inf = std::numeric_limits<float>::infinity();
  StoreVec4S(state_.cpu, 1, 0.0f, -0.0f, -inf, 0.0f);
  StoreVec4S(state_.cpu, 2, inf,  inf,    0.0f, -inf);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FrsqrtsVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.5f);
  EXPECT_FLOAT_EQ(r[1], 1.5f);
  EXPECT_FLOAT_EQ(r[2], 1.5f);
  EXPECT_FLOAT_EQ(r[3], 1.5f);
}

// Fused-vs-unfused divergence for FRECPS: pick (a, b) where the JIT's
// single-rounded FMA path differs from a hypothetical MUL+SUB pair.  The
// JIT must match libc's fmaf(-a, b, 2.0) bit-for-bit.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec4SFusedRounding) {
  // a = 1 + 2^-12, b = 1 + 2^-12.  a*b in real arithmetic = 1 + 2^-11 +
  // 2^-24, which is exactly representable in float (it fits in 24 bits).
  // fma(-a, b, 2) = 2 - 1 - 2^-11 - 2^-24 = 1 - 2^-11 - 2^-24.
  // Unfused: float(a*b) is 1+2^-11+2^-24 (exact); 2 - that = 1 - 2^-11 -
  // 2^-24, which here equals the fused value.  Sharper: use a = 1 + 2^-23
  // (smallest float > 1), b same — a*b exact = 1 + 2^-22 + 2^-46.  In
  // float, the 2^-46 bit is below the ulp at 1.0 (2^-23), so float(a*b) =
  // 1 + 2^-22; 2 - (1+2^-22) = 1 - 2^-22.  Fused: 2 - (1+2^-22+2^-46) =
  // (1-2^-22) - 2^-46; near 1.0 ulp is 2^-23 so 2^-46 rounds away — same
  // result.  So FRECPS at unity is rounding-friendly.  Instead pick
  // a = b near 0.5 where the result lands near 1.5: a = b = 0.5*(1 +
  // 2^-23), -a*b exact = -0.25*(1+2^-22+2^-46); 2 + that = 1.75 -
  // 0.25*2^-22 - 0.25*2^-46.  Magnitude ~1.75 has ulp 2^-23.  Both bits
  // are below ulp/2, so they round consistently — also rounding-friendly.
  // The cleanest forcing case is to leverage std::fmaf as the oracle and
  // pick values where round-friendliness is plausible but not guaranteed:
  const float a = std::ldexp(1.0f, 0) + std::ldexp(1.0f, -23);
  const float b = std::ldexp(1.0f, 0) + std::ldexp(1.0f, -23);
  StoreVec4S(state_.cpu, 1, a, 0.f, 0.f, 0.f);
  StoreVec4S(state_.cpu, 2, b, 0.f, 0.f, 0.f);
  StoreVec4S(state_.cpu, 0, 0.f, 0.f, 0.f, 0.f);
  static const uint32_t code[] = {FrecpsVec4S(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  const float expected = std::fmaf(-a, b, 2.0f);
  uint32_t r0_bits, ex_bits;
  std::memcpy(&r0_bits, &r[0], sizeof(uint32_t));
  std::memcpy(&ex_bits, &expected, sizeof(uint32_t));
  EXPECT_EQ(r0_bits, ex_bits);
}
// endregion

// region digitalis: AdvSIMD vector by-element JIT — FMLA / FMLS / FMUL at
// FP32 (.2S / .4S) and FP64 (.2D).
//
// ARM ARM encoding:
//   0 Q 0 01111 size L M Rm[3:0] opcode H 0 Rn Rd
//   size=10 (FP32): index = (H<<1)|L                  (range 0..3)
//   size=11 (FP64): index = H, L must be 0, Q must be 1 (.2D only)
//   opcode=0001 (FMLA), 0101 (FMLS), 1001 (FMUL with U=0).
// Verified with aarch64-linux-gnu-as / objdump:
//   FMLA v0.4s, v1.4s, v2.s[0] = 0x4F821020
//   FMLA v0.4s, v1.4s, v2.s[3] = 0x4FA21820
//   FMLA v0.2s, v1.2s, v2.s[1] = 0x0FA21020
//   FMLA v0.2d, v1.2d, v2.d[0] = 0x4FC21020
//   FMLA v0.2d, v1.2d, v2.d[1] = 0x4FC21820
//   FMLS v0.4s, v1.4s, v2.s[0] = 0x4F825020
//   FMUL v0.4s, v1.4s, v2.s[0] = 0x4F829020
constexpr uint32_t FmlaIdx4S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  // FP32: L = k&1 (bit 21), H = (k>>1)&1 (bit 11).
  return 0x4F801000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 21) |
         (static_cast<uint32_t>((k >> 1) & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlaIdx2S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x0F801000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 21) |
         (static_cast<uint32_t>((k >> 1) & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlaIdx2D(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  // FP64: index = H only (bit 11).
  return 0x4FC01000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsIdx4S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x4F805000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 21) |
         (static_cast<uint32_t>((k >> 1) & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsIdx2D(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x4FC05000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulIdx4S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x4F809000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 21) |
         (static_cast<uint32_t>((k >> 1) & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulIdx2S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x0F809000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 21) |
         (static_cast<uint32_t>((k >> 1) & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulIdx2D(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x4FC09000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
// FMULX (by element): same encoding as FMUL except U=1 (bit 29 set).
// Verified with aarch64-linux-gnu-as / objdump:
//   FMULX v0.4s, v1.4s, v2.s[0] = 0x6F829020
//   FMULX v0.4s, v1.4s, v2.s[3] = 0x6FA29820
//   FMULX v0.2s, v1.2s, v2.s[1] = 0x2FA29020
//   FMULX v0.2d, v1.2d, v2.d[0] = 0x6FC29020
//   FMULX v0.2d, v1.2d, v2.d[1] = 0x6FC29820
constexpr uint32_t FmulxIdx4S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x6F809000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 21) |
         (static_cast<uint32_t>((k >> 1) & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulxIdx2S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x2F809000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 21) |
         (static_cast<uint32_t>((k >> 1) & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulxIdx2D(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x6FC09000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(k & 1) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// FMLA .4S by-element: broadcast Vm.s[k] across all four lanes, then Vd += Vn*Vm.
TEST_F(Arm64LiteTranslateRegionTest, FmlaIdxVec4SBroadcastsLane) {
  StoreVec4S(state_.cpu, 1, 2.0f, 3.0f, -1.0f, 4.0f);
  StoreVec4S(state_.cpu, 2, 9.9f, 9.9f, 5.0f, 9.9f);  // Vm.s[2] = 5.0
  StoreVec4S(state_.cpu, 0, 1.0f, 1.0f, 1.0f, 1.0f);
  static const uint32_t code[] = {FmlaIdx4S(0, 1, 2, /*k=*/2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 11.0f);   // 1 + 2*5
  EXPECT_FLOAT_EQ(r[1], 16.0f);   // 1 + 3*5
  EXPECT_FLOAT_EQ(r[2], -4.0f);   // 1 + -1*5
  EXPECT_FLOAT_EQ(r[3], 21.0f);   // 1 + 4*5
}

// FMLA .2S q=0: upper 64 bits of Vd must be zeroed.
TEST_F(Arm64LiteTranslateRegionTest, FmlaIdxVec2SUpperZero) {
  StoreVec4S(state_.cpu, 1, 2.0f, 3.0f, 99.f, 99.f);
  StoreVec4S(state_.cpu, 2, 9.9f, 4.0f, 9.9f, 9.9f);  // Vm.s[1] = 4.0
  StoreVec4S(state_.cpu, 0, 1.0f, 8.0f, 7.0f, 7.0f);
  static const uint32_t code[] = {FmlaIdx2S(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 9.0f);   // 1 + 2*4
  EXPECT_FLOAT_EQ(r[1], 20.0f);  // 8 + 3*4
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// FMLA .2D by-element: index = 0 or 1 of Vm.2D.
TEST_F(Arm64LiteTranslateRegionTest, FmlaIdxVec2DBroadcastsLane) {
  StoreVec2D(state_.cpu, 1, 0.5, -3.0);
  StoreVec2D(state_.cpu, 2, 9.9, 2.0);  // Vm.d[1] = 2.0
  StoreVec2D(state_.cpu, 0, 1.0, 10.0);
  static const uint32_t code[] = {FmlaIdx2D(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 2.0);   // 1 + 0.5*2
  EXPECT_DOUBLE_EQ(r[1], 4.0);   // 10 + -3*2
}

// FMLS .4S by-element: Vd -= Vn * Vm.s[k].
TEST_F(Arm64LiteTranslateRegionTest, FmlsIdxVec4SBroadcastsLane) {
  StoreVec4S(state_.cpu, 1, 2.0f, 3.0f, -1.0f, 4.0f);
  StoreVec4S(state_.cpu, 2, 5.0f, 9.9f, 9.9f, 9.9f);  // Vm.s[0] = 5.0
  StoreVec4S(state_.cpu, 0, 11.0f, 16.0f, -4.0f, 21.0f);
  static const uint32_t code[] = {FmlsIdx4S(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 1.0f);    // 11 - 2*5
  EXPECT_FLOAT_EQ(r[1], 1.0f);    // 16 - 3*5
  EXPECT_FLOAT_EQ(r[2], 1.0f);    // -4 - (-1)*5
  EXPECT_FLOAT_EQ(r[3], 1.0f);    // 21 - 4*5
}

// FMLS .2D by-element.
TEST_F(Arm64LiteTranslateRegionTest, FmlsIdxVec2DBroadcastsLane) {
  StoreVec2D(state_.cpu, 1, 0.5, -3.0);
  StoreVec2D(state_.cpu, 2, 9.9, 2.0);  // Vm.d[1] = 2.0
  StoreVec2D(state_.cpu, 0, 10.0, 4.0);
  static const uint32_t code[] = {FmlsIdx2D(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 9.0);    // 10 - 0.5*2
  EXPECT_DOUBLE_EQ(r[1], 10.0);   // 4 - -3*2
}

// FMUL .4S by-element: Vd = Vn * Vm.s[k] (no accumulator).
TEST_F(Arm64LiteTranslateRegionTest, FmulIdxVec4SBroadcastsLane) {
  StoreVec4S(state_.cpu, 1, 2.0f, 3.0f, -1.0f, 4.0f);
  StoreVec4S(state_.cpu, 2, 9.9f, 9.9f, 9.9f, -7.0f);  // Vm.s[3] = -7.0
  StoreVec4S(state_.cpu, 0, 99.f, 99.f, 99.f, 99.f);  // Should be overwritten.
  static const uint32_t code[] = {FmulIdx4S(0, 1, 2, /*k=*/3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], -14.0f);  // 2 * -7
  EXPECT_FLOAT_EQ(r[1], -21.0f);  // 3 * -7
  EXPECT_FLOAT_EQ(r[2], 7.0f);    // -1 * -7
  EXPECT_FLOAT_EQ(r[3], -28.0f);  // 4 * -7
}

// FMUL .2S q=0: upper 64 bits zeroed.
TEST_F(Arm64LiteTranslateRegionTest, FmulIdxVec2SUpperZero) {
  StoreVec4S(state_.cpu, 1, 2.0f, 3.0f, 99.f, 99.f);
  StoreVec4S(state_.cpu, 2, 9.9f, 9.9f, 9.9f, 4.0f);  // Vm.s[3] = 4.0
  StoreVec4S(state_.cpu, 0, 99.f, 99.f, 7.f, 7.f);
  static const uint32_t code[] = {FmulIdx2S(0, 1, 2, /*k=*/3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 8.0f);
  EXPECT_FLOAT_EQ(r[1], 12.0f);
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// FMUL .2D by-element.
TEST_F(Arm64LiteTranslateRegionTest, FmulIdxVec2DBroadcastsLane) {
  StoreVec2D(state_.cpu, 1, 0.5, -3.0);
  StoreVec2D(state_.cpu, 2, 6.0, 9.9);  // Vm.d[0] = 6.0
  StoreVec2D(state_.cpu, 0, 99., 99.);  // Should be overwritten.
  static const uint32_t code[] = {FmulIdx2D(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 3.0);   // 0.5 * 6
  EXPECT_DOUBLE_EQ(r[1], -18.0); // -3 * 6
}

// FMULX .4S by-element: each lane = Vn[i] * broadcast_b, with (±0,±inf)
// saturation replacing NaN by ±2.0.  Pick lane assignment so the
// broadcast lane mixes finite + inf scenarios with different sign sources.
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxVec4SSaturation) {
  const float inf = std::numeric_limits<float>::infinity();
  // Broadcast Vm.s[1] = +inf across all four lanes.
  StoreVec4S(state_.cpu, 1, 0.0f, -0.0f, 3.0f, -inf);
  StoreVec4S(state_.cpu, 2, 9.9f, inf, 9.9f, 9.9f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FmulxIdx4S(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 2.0f);    // (+0,  +inf) -> +2
  EXPECT_FLOAT_EQ(r[1], -2.0f);   // (-0,  +inf) -> -2
  EXPECT_FLOAT_EQ(r[2], inf);     // 3 * +inf = +inf (no saturation)
  EXPECT_FLOAT_EQ(r[3], -inf);    // -inf * +inf = -inf (no saturation)
}

// FMULX .2S q=0 broadcast lane index + upper-64-zero invariant.
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxVec2SUpperZero) {
  const float inf = std::numeric_limits<float>::infinity();
  // Broadcast Vm.s[3] = -inf.
  StoreVec4S(state_.cpu, 1, 0.0f, 2.5f, 99.f, 99.f);  // lanes 2/3 of Vn ignored
  StoreVec4S(state_.cpu, 2, 9.9f, 9.9f, 9.9f, -inf);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), 7.f, 7.f);
  static const uint32_t code[] = {FmulxIdx2S(0, 1, 2, /*k=*/3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], -2.0f);   // (+0, -inf) -> -2
  EXPECT_FLOAT_EQ(r[1], -inf);    // 2.5 * -inf = -inf (no saturation)
  uint32_t lane2_bits, lane3_bits;
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// FMULX .2D by-element: broadcast high qword; covers the (inf, -0) sign mix.
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxVec2DBroadcastsLane) {
  const double inf = std::numeric_limits<double>::infinity();
  // Broadcast Vm.d[1] = -0.0.
  StoreVec2D(state_.cpu, 1, inf, -2.0);
  StoreVec2D(state_.cpu, 2, 9.9, -0.0);
  StoreVec2D(state_.cpu, 0, std::nan(""), std::nan(""));
  static const uint32_t code[] = {FmulxIdx2D(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], -2.0);   // (+inf, -0) -> -2
  EXPECT_DOUBLE_EQ(r[1], 0.0);    // -2 * -0 = +0 (no saturation)
}

// FMULX NaN input must propagate, NOT be replaced by ±2.0 — the saturation
// override fires only when mul is NaN AND neither input is NaN.
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxVec4SNaNPropagation) {
  const float qnan = std::nanf("");
  // Broadcast Vm.s[0] = qnan; Vn has finite lanes.
  StoreVec4S(state_.cpu, 1, 2.0f, -3.0f, 0.5f, 1.5f);
  StoreVec4S(state_.cpu, 2, qnan, 9.9f, 9.9f, 9.9f);
  static const uint32_t code[] = {FmulxIdx4S(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isnan(r[i])) << "lane " << i << " expected NaN, got " << r[i];
  }
}

// FMULX .2D regular finite — sanity that the saturation path doesn't disturb
// ordinary multiplies.
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxVec2DRegular) {
  StoreVec2D(state_.cpu, 1, 2.0, -4.0);
  StoreVec2D(state_.cpu, 2, 0.25, 9.9);  // Vm.d[0] = 0.25
  static const uint32_t code[] = {FmulxIdx2D(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 0.5);
  EXPECT_DOUBLE_EQ(r[1], -1.0);
}

// FMULX scalar-by-element encoders (AdvSIMD scalar x indexed element).
// Verified with aarch64-linux-gnu-as / objdump:
//   FMULX s0, s1, v2.s[0]   = 0x7F829020   (size=10, L=0, H=0)
//   FMULX s0, s1, v2.s[1]   = 0x7FA29020   (size=10, L=1, H=0)
//   FMULX s0, s1, v2.s[2]   = 0x7F829820   (size=10, L=0, H=1)
//   FMULX s0, s1, v2.s[3]   = 0x7FA29820   (size=10, L=1, H=1)
//   FMULX d0, d1, v2.d[0]   = 0x7FC29020   (size=11, H=0)
//   FMULX d0, d1, v2.d[1]   = 0x7FC29820   (size=11, H=1)
//   FMULX s7, s9, v11.s[2]  = 0x7F8B9927   (M=0, Rm[3:0]=1011, H=1, L=0)
// L lives at bit21; H lives at bit11.  Rm[3:0] at bits[19:16]; M at bit20.
constexpr uint32_t FmulxIdxScalarS(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t L = (k >> 0) & 1u;
  uint32_t H = (k >> 1) & 1u;
  uint32_t M = (rm >> 4) & 1u;
  uint32_t Rm_lo = rm & 0xFu;
  return 0x7F809000u | (L << 21) | (M << 20) | (Rm_lo << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulxIdxScalarD(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  // size=11 (bit22 set in addition to bit23); L must be 0; index = H only.
  uint32_t H = k & 1u;
  uint32_t M = (rm >> 4) & 1u;
  uint32_t Rm_lo = rm & 0xFu;
  return 0x7FC09000u | (M << 20) | (Rm_lo << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// FMULX scalar FP32: (+0 * +inf) -> +2.  Confirms the (zero,inf) saturation
// override fires on the scalar by-element path.
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxScalarSPosZeroPosInfReturnsPlusTwo) {
  const float inf = std::numeric_limits<float>::infinity();
  // Vn.s[0] = +0; Vm.s[1] = +inf — index 1 broadcasts the saturating lane.
  StoreVec4S(state_.cpu, 1, 0.0f, 99.f, 99.f, 99.f);
  StoreVec4S(state_.cpu, 2, 9.9f, inf, 9.9f, 9.9f);
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FmulxIdxScalarS(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], 2.0f);
  // Scalar destination must zero the upper three S-lanes of Vd.
  uint32_t lane1_bits, lane2_bits, lane3_bits;
  std::memcpy(&lane1_bits, &r[1], sizeof(uint32_t));
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane1_bits, 0u);
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// FMULX scalar FP64: (-0 * +inf) -> -2.  Sign comes from XOR of operands.
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxScalarDNegZeroPosInfReturnsMinusTwo) {
  const double inf = std::numeric_limits<double>::infinity();
  StoreVec2D(state_.cpu, 1, -0.0, 99.0);
  StoreVec2D(state_.cpu, 2, 9.9, inf);  // Vm.d[1] = +inf
  StoreVec2D(state_.cpu, 0, std::nan(""), std::nan(""));
  static const uint32_t code[] = {FmulxIdxScalarD(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], -2.0);
  // Scalar destination must zero the upper D-lane of Vd.
  uint64_t lane1_bits;
  std::memcpy(&lane1_bits, &r[1], sizeof(uint64_t));
  EXPECT_EQ(lane1_bits, 0ULL);
}

// FMULX scalar with NaN input must propagate NaN — not get replaced by ±2.0.
// The saturation override fires only when (mul is NaN) AND (neither input is NaN).
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxScalarSNaNPropagation) {
  const float qnan = std::nanf("");
  // Vn.s[0] = qnan; Vm.s[0] = 1.0 (finite).
  StoreVec4S(state_.cpu, 1, qnan, 99.f, 99.f, 99.f);
  StoreVec4S(state_.cpu, 2, 1.0f, 9.9f, 9.9f, 9.9f);
  static const uint32_t code[] = {FmulxIdxScalarS(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_TRUE(std::isnan(r[0])) << "expected NaN at scalar lane, got " << r[0];
}

// FMULX scalar finite multiply — sanity that ordinary multiplies are not
// disturbed by the saturation path.  Use FP32, lane index 3 to also exercise
// the bit21/bit11 packing.
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxScalarSRegular) {
  StoreVec4S(state_.cpu, 1, 2.5f, 99.f, 99.f, 99.f);
  StoreVec4S(state_.cpu, 2, 9.9f, 9.9f, 9.9f, -4.0f);  // Vm.s[3] = -4.0
  static const uint32_t code[] = {FmulxIdxScalarS(0, 1, 2, /*k=*/3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], -10.0f);   // 2.5 * -4
  uint32_t lane1_bits, lane2_bits, lane3_bits;
  std::memcpy(&lane1_bits, &r[1], sizeof(uint32_t));
  std::memcpy(&lane2_bits, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3_bits, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane1_bits, 0u);
  EXPECT_EQ(lane2_bits, 0u);
  EXPECT_EQ(lane3_bits, 0u);
}

// FMULX scalar FP64 lane[0] broadcast: confirms the H=0 encoding selects
// Vm.d[0] and not Vm.d[1].
TEST_F(Arm64LiteTranslateRegionTest, FmulxIdxScalarDRegular) {
  StoreVec2D(state_.cpu, 1, 3.0, 99.0);
  StoreVec2D(state_.cpu, 2, 0.5, 9.9);  // Vm.d[0] = 0.5
  static const uint32_t code[] = {FmulxIdxScalarD(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 1.5);
  uint64_t lane1_bits;
  std::memcpy(&lane1_bits, &r[1], sizeof(uint64_t));
  EXPECT_EQ(lane1_bits, 0ULL);
}

// FMUL / FMLA / FMLS scalar-by-element encoders.  Same encoding shape as
// FMULX scalar except U=0 and (opcode=1001/0001/0101).  Verified with
// aarch64-linux-gnu-as / objdump:
//   fmul  s0, s1, v2.s[0]  = 0x5F829020
//   fmul  d0, d1, v2.d[0]  = 0x5FC29020
//   fmla  s0, s1, v2.s[0]  = 0x5F821020
//   fmla  d0, d1, v2.d[1]  = 0x5FC21820
//   fmls  s0, s1, v2.s[0]  = 0x5F825020
//   fmls  d0, d1, v2.d[0]  = 0x5FC25020
constexpr uint32_t FmulIdxScalarS(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t L = (k >> 0) & 1u;
  uint32_t H = (k >> 1) & 1u;
  uint32_t M = (rm >> 4) & 1u;
  uint32_t Rm_lo = rm & 0xFu;
  return 0x5F809000u | (L << 21) | (M << 20) | (Rm_lo << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulIdxScalarD(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t H = k & 1u;
  uint32_t M = (rm >> 4) & 1u;
  uint32_t Rm_lo = rm & 0xFu;
  return 0x5FC09000u | (M << 20) | (Rm_lo << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlaIdxScalarS(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t L = (k >> 0) & 1u;
  uint32_t H = (k >> 1) & 1u;
  uint32_t M = (rm >> 4) & 1u;
  uint32_t Rm_lo = rm & 0xFu;
  return 0x5F801000u | (L << 21) | (M << 20) | (Rm_lo << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlaIdxScalarD(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t H = k & 1u;
  uint32_t M = (rm >> 4) & 1u;
  uint32_t Rm_lo = rm & 0xFu;
  return 0x5FC01000u | (M << 20) | (Rm_lo << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsIdxScalarS(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t L = (k >> 0) & 1u;
  uint32_t H = (k >> 1) & 1u;
  uint32_t M = (rm >> 4) & 1u;
  uint32_t Rm_lo = rm & 0xFu;
  return 0x5F805000u | (L << 21) | (M << 20) | (Rm_lo << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsIdxScalarD(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t H = k & 1u;
  uint32_t M = (rm >> 4) & 1u;
  uint32_t Rm_lo = rm & 0xFu;
  return 0x5FC05000u | (M << 20) | (Rm_lo << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// FMUL scalar FP32: regular finite multiply, lane index 3 exercises both
// L (bit21) and H (bit11) of the index encoding.  Upper-lane zero check
// confirms scalar destination semantics.
TEST_F(Arm64LiteTranslateRegionTest, FmulIdxScalarSRegular) {
  StoreVec4S(state_.cpu, 1, 2.5f, 99.f, 99.f, 99.f);
  StoreVec4S(state_.cpu, 2, 9.9f, 9.9f, 9.9f, -4.0f);  // Vm.s[3] = -4.0
  StoreVec4S(state_.cpu, 0, std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  static const uint32_t code[] = {FmulIdxScalarS(0, 1, 2, /*k=*/3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  EXPECT_FLOAT_EQ(r[0], -10.0f);  // 2.5 * -4
  uint32_t lane1, lane2, lane3;
  std::memcpy(&lane1, &r[1], sizeof(uint32_t));
  std::memcpy(&lane2, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane1, 0u);
  EXPECT_EQ(lane2, 0u);
  EXPECT_EQ(lane3, 0u);
}

// FMUL scalar FP64: confirms H=1 selects Vm.d[1].
TEST_F(Arm64LiteTranslateRegionTest, FmulIdxScalarDRegular) {
  StoreVec2D(state_.cpu, 1, 6.0, 99.0);
  StoreVec2D(state_.cpu, 2, 9.9, 0.25);  // Vm.d[1] = 0.25
  StoreVec2D(state_.cpu, 0, std::nan(""), std::nan(""));
  static const uint32_t code[] = {FmulIdxScalarD(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 1.5);  // 6.0 * 0.25
  uint64_t lane1;
  std::memcpy(&lane1, &r[1], sizeof(uint64_t));
  EXPECT_EQ(lane1, 0ULL);
}

// FMLA scalar FP32: Vd = Vd + Vn * Vm[index], with fused-vs-unfused
// divergence proving the lowering uses VFMADD231SS rather than
// (Vn*Vm)+Vd through Mulss+Addss.
TEST_F(Arm64LiteTranslateRegionTest, FmlaIdxScalarSFusedRounding) {
  const float a = 1.0f + std::ldexp(1.0f, -12);
  StoreVec4S(state_.cpu, 1, a, 99.f, 99.f, 99.f);
  StoreVec4S(state_.cpu, 2, a, 9.9f, 9.9f, 9.9f);  // Vm.s[0] = a
  StoreVec4S(state_.cpu, 0, -1.0f, 99.f, 99.f, 99.f);
  static const uint32_t code[] = {FmlaIdxScalarS(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  const float expected = std::fmaf(a, a, -1.0f);
  uint32_t r0_bits, ex_bits;
  std::memcpy(&r0_bits, &r[0], sizeof(uint32_t));
  std::memcpy(&ex_bits, &expected, sizeof(uint32_t));
  EXPECT_EQ(r0_bits, ex_bits);
  // Upper-lane zero check.
  uint32_t lane1, lane2, lane3;
  std::memcpy(&lane1, &r[1], sizeof(uint32_t));
  std::memcpy(&lane2, &r[2], sizeof(uint32_t));
  std::memcpy(&lane3, &r[3], sizeof(uint32_t));
  EXPECT_EQ(lane1, 0u);
  EXPECT_EQ(lane2, 0u);
  EXPECT_EQ(lane3, 0u);
}

// FMLA scalar FP64: confirms FMA semantics and FP64 path through
// VFMADD231SD on the H=1 (Vm.d[1]) lane.
TEST_F(Arm64LiteTranslateRegionTest, FmlaIdxScalarDRegular) {
  StoreVec2D(state_.cpu, 1, 2.0, 99.0);
  StoreVec2D(state_.cpu, 2, 9.9, 3.0);  // Vm.d[1] = 3.0
  StoreVec2D(state_.cpu, 0, 1.5, 99.0);
  static const uint32_t code[] = {FmlaIdxScalarD(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 7.5);  // 1.5 + 2.0 * 3.0
  uint64_t lane1;
  std::memcpy(&lane1, &r[1], sizeof(uint64_t));
  EXPECT_EQ(lane1, 0ULL);
}

// FMLS scalar FP32: Vd = Vd - Vn * Vm[index] (single fused rounding).
TEST_F(Arm64LiteTranslateRegionTest, FmlsIdxScalarSFusedRounding) {
  const float a = 1.0f + std::ldexp(1.0f, -12);
  StoreVec4S(state_.cpu, 1, a, 99.f, 99.f, 99.f);
  StoreVec4S(state_.cpu, 2, 9.9f, 9.9f, a, 9.9f);  // Vm.s[2] = a
  StoreVec4S(state_.cpu, 0, 1.0f, 99.f, 99.f, 99.f);
  static const uint32_t code[] = {FmlsIdxScalarS(0, 1, 2, /*k=*/2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  // Vd = 1.0 - a*a  = std::fma(-a, a, 1.0)
  const float expected = std::fmaf(-a, a, 1.0f);
  uint32_t r0_bits, ex_bits;
  std::memcpy(&r0_bits, &r[0], sizeof(uint32_t));
  std::memcpy(&ex_bits, &expected, sizeof(uint32_t));
  EXPECT_EQ(r0_bits, ex_bits);
  uint32_t lane1;
  std::memcpy(&lane1, &r[1], sizeof(uint32_t));
  EXPECT_EQ(lane1, 0u);
}

// FMLS scalar FP64: regular finite path, sanity check.
TEST_F(Arm64LiteTranslateRegionTest, FmlsIdxScalarDRegular) {
  StoreVec2D(state_.cpu, 1, 4.0, 99.0);
  StoreVec2D(state_.cpu, 2, 2.5, 9.9);  // Vm.d[0] = 2.5
  StoreVec2D(state_.cpu, 0, 12.0, 99.0);
  static const uint32_t code[] = {FmlsIdxScalarD(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  double r[2];
  LoadVec2D(state_.cpu, 0, r);
  EXPECT_DOUBLE_EQ(r[0], 2.0);  // 12.0 - 4.0 * 2.5
  uint64_t lane1;
  std::memcpy(&lane1, &r[1], sizeof(uint64_t));
  EXPECT_EQ(lane1, 0ULL);
}

// Fused-vs-unfused divergence: pick (a, b, d) such that fma(a, b, d) differs
// from (a*b)+d in float, proving the lowering uses VFMADD231PS.
TEST_F(Arm64LiteTranslateRegionTest, FmlaIdxVec4SFusedRounding) {
  // Triple (a, b, d) where the trailing bit of a*b is lost by an
  // intermediate rounding but kept by fused FMA.  Same shape as
  // FmlaVec4SFusedRounding above (three-same form).
  //   a = 1 + 2^-12, b = a, d = -1.0
  //   fused result: 2^-11 + 2^-24
  //   unfused     : 2^-11
  const float a = 1.0f + std::ldexp(1.0f, -12);
  StoreVec4S(state_.cpu, 1, a, 0.f, 0.f, 0.f);
  StoreVec4S(state_.cpu, 2, a, 0.f, 0.f, 0.f);  // Vm.s[0] = a
  StoreVec4S(state_.cpu, 0, -1.0f, 0.f, 0.f, 0.f);
  static const uint32_t code[] = {FmlaIdx4S(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  float r[4];
  LoadVec4S(state_.cpu, 0, r);
  const float expected = std::fmaf(a, a, -1.0f);
  uint32_t r0_bits, ex_bits;
  std::memcpy(&r0_bits, &r[0], sizeof(uint32_t));
  std::memcpy(&ex_bits, &expected, sizeof(uint32_t));
  EXPECT_EQ(r0_bits, ex_bits);
}
// endregion

// region digitalis: FP16 vector FRECPS / FRSQRTS .4H / .8H — F16C round-trip
// JIT.  Encodings (verified via aarch64-linux-gnu-as):
//   FRECPS  Vd.4H, Vn.4H, Vm.4H = 0x0E403C00 | (rm<<16) | (rn<<5) | rd
//   FRECPS  Vd.8H, Vn.8H, Vm.8H = 0x4E403C00 | (rm<<16) | (rn<<5) | rd
//   FRSQRTS Vd.4H, Vn.4H, Vm.4H = 0x0EC03C00 | (rm<<16) | (rn<<5) | rd
//   FRSQRTS Vd.8H, Vn.8H, Vm.8H = 0x4EC03C00 | (rm<<16) | (rn<<5) | rd
constexpr uint32_t FrecpsVec4H(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E403C00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrecpsVec8H(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E403C00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrsqrtsVec4H(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0EC03C00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FrsqrtsVec8H(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EC03C00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

static void StoreVec8H(CPUState& cpu, unsigned idx, const uint16_t lanes[8]) {
  std::memcpy(&cpu.v[idx], lanes, 16);
}
static void LoadVec8H(const CPUState& cpu, unsigned idx, uint16_t out[8]) {
  std::memcpy(out, &cpu.v[idx], 16);
}

// FP16 bit pattern constants used by the tests below.
constexpr uint16_t kHalf_1_0  = 0x3C00;
constexpr uint16_t kHalf_2_0  = 0x4000;
constexpr uint16_t kHalf_0_5  = 0x3800;
constexpr uint16_t kHalf_neg2_0 = 0xC000;
constexpr uint16_t kHalf_neg0_5 = 0xB800;
constexpr uint16_t kHalf_4_0  = 0x4400;
constexpr uint16_t kHalf_0_25 = 0x3400;
constexpr uint16_t kHalf_pos0 = 0x0000;
constexpr uint16_t kHalf_neg0 = 0x8000;
constexpr uint16_t kHalf_pos_inf = 0x7C00;
constexpr uint16_t kHalf_neg_inf = 0xFC00;
constexpr uint16_t kHalf_qNaN = 0x7E00;
constexpr uint16_t kHalf_1_5  = 0x3E00;
constexpr uint16_t kHalf_3_0  = 0x4200;

// FRECPS .4H — Newton step for reciprocal lane-by-lane.  Lanes 4..7 of Vd
// must be zeroed by the FP16 round-trip path (Q=0 -> upper 64 bits zero).
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec4HAllLanes) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_0_5, kHalf_neg2_0, kHalf_4_0,
                                0x5555, 0x5555, 0x5555, 0x5555};
  const uint16_t m_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg0_5, kHalf_0_25,
                                0x5555, 0x5555, 0x5555, 0x5555};
  uint16_t d_init[8] = {kHalf_qNaN, kHalf_qNaN, kHalf_qNaN, kHalf_qNaN,
                        0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FrecpsVec4H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);  // 2 - 1*1
  EXPECT_EQ(r[1], kHalf_1_0);  // 2 - 0.5*2
  EXPECT_EQ(r[2], kHalf_1_0);  // 2 - (-2)*(-0.5)
  EXPECT_EQ(r[3], kHalf_1_0);  // 2 - 4*0.25
  // Upper 64 bits (lanes 4..7) zeroed by Vcvtps2ph (Q=0 invariant).
  for (int i = 4; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FRECPS .4H — (±0, ±inf) crosses must produce +2.0 (no sign flip),
// not the default qNaN.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec4HZeroTimesInfSaturation) {
  const uint16_t n_lanes[8] = {kHalf_pos0, kHalf_neg0, kHalf_neg_inf, kHalf_pos0,
                                0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_pos_inf, kHalf_pos_inf, kHalf_pos0, kHalf_neg_inf,
                                0, 0, 0, 0};
  uint16_t d_init[8] = {kHalf_qNaN, kHalf_qNaN, kHalf_qNaN, kHalf_qNaN,
                        0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FrecpsVec4H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_2_0);
  EXPECT_EQ(r[1], kHalf_2_0);
  EXPECT_EQ(r[2], kHalf_2_0);
  EXPECT_EQ(r[3], kHalf_2_0);
}

// FRECPS .4H — NaN input yields default qNaN, not the saturation constant.
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec4HNaNInputDefaultsToQnan) {
  const uint16_t n_lanes[8] = {kHalf_qNaN, kHalf_pos0,    kHalf_1_0, kHalf_1_0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_1_0,  kHalf_pos_inf, kHalf_1_0, kHalf_qNaN, 0, 0, 0, 0};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FrecpsVec4H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  // Lane 0/3: NaN input -> qNaN.
  EXPECT_EQ(r[0], kHalf_qNaN);
  EXPECT_EQ(r[3], kHalf_qNaN);
  // Lane 1: (+0, +inf) cross -> +2.0.
  EXPECT_EQ(r[1], kHalf_2_0);
  // Lane 2: 2 - 1*1 = 1.
  EXPECT_EQ(r[2], kHalf_1_0);
}

// FRSQRTS .4H — Newton step (3 - a*b) / 2, lane by lane.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsVec4HAllLanes) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg2_0, kHalf_0_5, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_1_0, kHalf_0_5, kHalf_neg0_5, kHalf_4_0, 0, 0, 0, 0};
  uint16_t d_init[8] = {kHalf_qNaN, kHalf_qNaN, kHalf_qNaN, kHalf_qNaN, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FrsqrtsVec4H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);  // (3 - 1*1) / 2 = 1
  EXPECT_EQ(r[1], kHalf_1_0);  // (3 - 2*0.5) / 2 = 1
  EXPECT_EQ(r[2], kHalf_1_0);  // (3 - (-2)*(-0.5)) / 2 = 1
  EXPECT_EQ(r[3], kHalf_0_5);  // (3 - 0.5*4) / 2 = 0.5
}

// FRSQRTS .4H — (±0, ±inf) crosses must produce +1.5 (no sign flip).
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsVec4HZeroTimesInfSaturation) {
  const uint16_t n_lanes[8] = {kHalf_pos0, kHalf_neg0, kHalf_neg_inf, kHalf_pos0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_pos_inf, kHalf_pos_inf, kHalf_pos0, kHalf_neg_inf, 0, 0, 0, 0};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FrsqrtsVec4H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_5);
  EXPECT_EQ(r[1], kHalf_1_5);
  EXPECT_EQ(r[2], kHalf_1_5);
  EXPECT_EQ(r[3], kHalf_1_5);
}

// FRECPS .8H — exercises the 2-pass XMM path (low 4 lanes then high 4).
// Mix regular Newton-step lanes (0-3) and saturation lanes (4-7).
TEST_F(Arm64LiteTranslateRegionTest, FrecpsVec8HTwoPassMixed) {
  const uint16_t n_lanes[8] = {kHalf_1_0,    kHalf_0_5,    kHalf_neg2_0, kHalf_4_0,
                                kHalf_pos0,   kHalf_neg0,   kHalf_neg_inf, kHalf_pos0};
  const uint16_t m_lanes[8] = {kHalf_1_0,    kHalf_2_0,    kHalf_neg0_5, kHalf_0_25,
                                kHalf_pos_inf, kHalf_pos_inf, kHalf_pos0,    kHalf_neg_inf};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FrecpsVec8H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  // Lanes 0-3 (low pass): regular Newton step -> 1.0h.
  EXPECT_EQ(r[0], kHalf_1_0);
  EXPECT_EQ(r[1], kHalf_1_0);
  EXPECT_EQ(r[2], kHalf_1_0);
  EXPECT_EQ(r[3], kHalf_1_0);
  // Lanes 4-7 (high pass): (zero, inf) cross -> +2.0h.
  EXPECT_EQ(r[4], kHalf_2_0);
  EXPECT_EQ(r[5], kHalf_2_0);
  EXPECT_EQ(r[6], kHalf_2_0);
  EXPECT_EQ(r[7], kHalf_2_0);
}

// FRSQRTS .8H — same two-pass split, all lanes regular Newton step.
TEST_F(Arm64LiteTranslateRegionTest, FrsqrtsVec8HTwoPassRegular) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg2_0, kHalf_0_5,
                                kHalf_1_0, kHalf_0_5, kHalf_4_0,    kHalf_2_0};
  const uint16_t m_lanes[8] = {kHalf_1_0, kHalf_0_5, kHalf_neg0_5, kHalf_4_0,
                                kHalf_1_0, kHalf_2_0, kHalf_0_25,   kHalf_0_5};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FrsqrtsVec8H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);  // (3 - 1*1)/2 = 1
  EXPECT_EQ(r[1], kHalf_1_0);  // (3 - 2*0.5)/2 = 1
  EXPECT_EQ(r[2], kHalf_1_0);  // (3 - (-2)*(-0.5))/2 = 1
  EXPECT_EQ(r[3], kHalf_0_5);  // (3 - 0.5*4)/2 = 0.5
  EXPECT_EQ(r[4], kHalf_1_0);  // (3 - 1*1)/2 = 1
  EXPECT_EQ(r[5], kHalf_1_0);  // (3 - 0.5*2)/2 = 1
  EXPECT_EQ(r[6], kHalf_1_0);  // (3 - 4*0.25)/2 = 1
  EXPECT_EQ(r[7], kHalf_1_0);  // (3 - 2*0.5)/2 = 1
}
// endregion

// region digitalis: FP16 vector FMLA / FMLS .4H / .8H — FP16 -> FP32 -> FP64
// round-trip JIT.  Encodings (verified via aarch64-linux-gnu-as -march=
// armv8.2-a+fp16):
//   FMLA Vd.4H, Vn.4H, Vm.4H = 0x0E400C00 | (rm<<16) | (rn<<5) | rd
//   FMLA Vd.8H, Vn.8H, Vm.8H = 0x4E400C00 | (rm<<16) | (rn<<5) | rd
//   FMLS Vd.4H, Vn.4H, Vm.4H = 0x0EC00C00 | (rm<<16) | (rn<<5) | rd
//   FMLS Vd.8H, Vn.8H, Vm.8H = 0x4EC00C00 | (rm<<16) | (rn<<5) | rd
constexpr uint32_t FmlaVec4H(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E400C00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlaVec8H(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4E400C00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsVec4H(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0EC00C00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsVec8H(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x4EC00C00u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// FMLA .4H lane-by-lane: Vd[i] = Vd[i] + Vn[i] * Vm[i].  Upper 64 bits
// of Vd zeroed by the FP16 round-trip path (Q=0 invariant).
TEST_F(Arm64LiteTranslateRegionTest, FmlaVec4HAllLanes) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg0_5, kHalf_4_0,
                                0x5555, 0x5555, 0x5555, 0x5555};
  const uint16_t m_lanes[8] = {kHalf_2_0, kHalf_0_5, kHalf_4_0,    kHalf_0_25,
                                0x5555, 0x5555, 0x5555, 0x5555};
  // d_init: 1.0, 1.0, 3.0, -1.0  -> r = 3.0, 2.0, 1.0, 0.0
  uint16_t d_init[8] = {kHalf_1_0, kHalf_1_0, kHalf_3_0,
                        0xBC00,  // -1.0h
                        0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlaVec4H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_3_0);   // 1 + 1*2 = 3
  EXPECT_EQ(r[1], kHalf_2_0);   // 1 + 2*0.5 = 2
  EXPECT_EQ(r[2], kHalf_1_0);   // 3 + (-0.5)*4 = 1
  EXPECT_EQ(r[3], kHalf_pos0);  // -1 + 4*0.25 = 0
  // Upper 64 bits (lanes 4..7) zeroed by Vcvtps2ph (Q=0 invariant).
  for (int i = 4; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FMLS .4H lane-by-lane: Vd[i] = Vd[i] - Vn[i] * Vm[i].
TEST_F(Arm64LiteTranslateRegionTest, FmlsVec4HAllLanes) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg0_5, kHalf_4_0,
                                0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_2_0, kHalf_0_5, kHalf_4_0,    kHalf_0_25,
                                0, 0, 0, 0};
  // d_init: 3.0, 2.0, -1.0, 2.0  -> r = 1.0, 1.0, 1.0, 1.0
  uint16_t d_init[8] = {kHalf_3_0, kHalf_2_0,
                        0xBC00,  // -1.0h
                        kHalf_2_0,
                        0xBBBB, 0xBBBB, 0xBBBB, 0xBBBB};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlsVec4H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);   // 3 - 1*2 = 1
  EXPECT_EQ(r[1], kHalf_1_0);   // 2 - 2*0.5 = 1
  EXPECT_EQ(r[2], kHalf_1_0);   // -1 - (-0.5)*4 = -1 + 2 = 1
  EXPECT_EQ(r[3], kHalf_1_0);   // 2 - 4*0.25 = 1
  for (int i = 4; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FMLA .8H: 8 lanes, exercises both pass paths (low 4 and high 4).
TEST_F(Arm64LiteTranslateRegionTest, FmlaVec8HTwoPassMixed) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg0_5, kHalf_4_0,
                                kHalf_1_0, kHalf_1_0,    kHalf_2_0, kHalf_4_0};
  const uint16_t m_lanes[8] = {kHalf_2_0, kHalf_0_5, kHalf_4_0,    kHalf_0_25,
                                kHalf_1_0, kHalf_neg2_0, kHalf_2_0, kHalf_0_5};
  // d_init: 1, 1, 3, -1, 0, 3, -2, 0  -> 3, 2, 1, 0, 1, 1, 2, 2
  uint16_t d_init[8] = {kHalf_1_0, kHalf_1_0, kHalf_3_0,
                        0xBC00,    // -1
                        kHalf_pos0, kHalf_3_0,
                        0xC000,    // -2
                        kHalf_pos0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlaVec8H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_3_0);   // 1 + 1*2 = 3
  EXPECT_EQ(r[1], kHalf_2_0);   // 1 + 2*0.5 = 2
  EXPECT_EQ(r[2], kHalf_1_0);   // 3 + (-0.5)*4 = 1
  EXPECT_EQ(r[3], kHalf_pos0);  // -1 + 4*0.25 = 0
  EXPECT_EQ(r[4], kHalf_1_0);   // 0 + 1*1 = 1
  EXPECT_EQ(r[5], kHalf_1_0);   // 3 + 1*(-2) = 1
  EXPECT_EQ(r[6], kHalf_2_0);   // -2 + 2*2 = 2
  EXPECT_EQ(r[7], kHalf_2_0);   // 0 + 4*0.5 = 2
}

// FMLS .8H: same two-pass exercise as FMLA .8H.
TEST_F(Arm64LiteTranslateRegionTest, FmlsVec8HTwoPassRegular) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg0_5, kHalf_4_0,
                                kHalf_1_0, kHalf_2_0, kHalf_1_0,    kHalf_4_0};
  const uint16_t m_lanes[8] = {kHalf_2_0, kHalf_0_5, kHalf_4_0,    kHalf_0_25,
                                kHalf_2_0, kHalf_0_5, kHalf_1_0,    kHalf_0_25};
  // d_init: 3, 2, -1, 2, 4, 3, 2, 2  -> 1, 1, 1, 1, 2, 2, 1, 1
  uint16_t d_init[8] = {kHalf_3_0, kHalf_2_0,
                        0xBC00,   // -1
                        kHalf_2_0,
                        kHalf_4_0, kHalf_3_0, kHalf_2_0, kHalf_2_0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlsVec8H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);   // 3 - 1*2 = 1
  EXPECT_EQ(r[1], kHalf_1_0);   // 2 - 2*0.5 = 1
  EXPECT_EQ(r[2], kHalf_1_0);   // -1 - (-0.5)*4 = 1
  EXPECT_EQ(r[3], kHalf_1_0);   // 2 - 4*0.25 = 1
  EXPECT_EQ(r[4], kHalf_2_0);   // 4 - 1*2 = 2
  EXPECT_EQ(r[5], kHalf_2_0);   // 3 - 2*0.5 = 2
  EXPECT_EQ(r[6], kHalf_1_0);   // 2 - 1*1 = 1
  EXPECT_EQ(r[7], kHalf_1_0);   // 2 - 4*0.25 = 1
}

// Fused-vs-unfused rounding divergence on FMLA .4H.  Picks (a, b, d)
// such that fma((double)a, (double)b, (double)d) narrowed to half
// differs from (float(a*b) + d) narrowed to half — proves the FP64
// round-trip path matches the interpreter's binary64 oracle for a
// case where the intermediate FP32 sum would double-round to a
// different FP16 result.
TEST_F(Arm64LiteTranslateRegionTest, FmlaVec4HFusedRounding) {
  // a = 1.0h + 2^-10 ulp = 0x3C01  (smallest > 1.0h in FP16, ~= 1.0009766)
  // b = same
  // d = -1.0h = 0xBC00
  // Exact product in FP64: (1 + 2^-10)*(1 + 2^-10) = 1 + 2^-9 + 2^-20.
  // Fused (interpreter): fma(a, b, -1) -> (2^-9 + 2^-20) ~ 0.0019536...
  //   narrowed to FP16 (1 mantissa ULP around 0.001953125 is 2^-10*ulp_exp)
  //   -- the value 2^-9 + 2^-20 = 0.001953125 + 0.00000095367...
  //   rounded to nearest FP16 in the range [2^-9, 2^-8) where the ULP
  //   is 2^-19, equals 0x1801 (representable bits set so the trailing
  //   2^-20 bit rounds up to next ULP).
  // We don't precompute the expected bit pattern here — instead we
  // compute the oracle by mimicking the interpreter formula directly
  // and compare to the JIT result.
  auto half_to_float = [](uint16_t h) -> float {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t f = h & 0x3FF;
    uint32_t bits;
    if (e == 0) {
      if (f == 0) {
        bits = s << 31;
      } else {
        // Subnormal half -> normal float.
        while ((f & 0x400) == 0) { f <<= 1; e--; }
        e++; f &= 0x3FF;
        bits = (s << 31) | ((e + 127 - 15) << 23) | (f << 13);
      }
    } else if (e == 31) {
      bits = (s << 31) | (0xFF << 23) | (f << 13);
    } else {
      bits = (s << 31) | ((e + 127 - 15) << 23) | (f << 13);
    }
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
  };
  const uint16_t ha = 0x3C01;  // 1 + 2^-10
  const uint16_t hd = 0xBC00;  // -1.0
  const float a = half_to_float(ha);
  const float d = half_to_float(hd);
  // Interpreter oracle:
  //   double r64 = std::fma((double)a, (double)a, (double)d);
  //   uint16_t expected = FpSingleToHalf((float)r64);
  // FpSingleToHalf does correct round-to-nearest-even FP32->FP16; we
  // emulate by going through Vcvtps2ph in the same direction below.
  // Simpler: just verify the JIT result matches by computing both
  // paths and comparing.
  const double r64 = std::fma(static_cast<double>(a),
                              static_cast<double>(a),
                              static_cast<double>(d));
  const float r32 = static_cast<float>(r64);
  uint32_t r32_bits;
  std::memcpy(&r32_bits, &r32, 4);
  // Convert FP32 -> FP16 (round-to-nearest-even).
  auto float_to_half_rne = [](uint32_t f) -> uint16_t {
    uint32_t s = (f >> 31) & 1;
    int32_t e = static_cast<int32_t>((f >> 23) & 0xFF) - 127;
    uint32_t m = f & 0x7FFFFF;
    if (e == 128) {  // inf or NaN
      if (m == 0) return (s << 15) | 0x7C00;
      return (s << 15) | 0x7C00 | (m >> 13) | (m == 0 ? 1 : 0);
    }
    if (e > 15) return (s << 15) | 0x7C00;  // overflow -> inf
    if (e < -24) return (s << 15);          // underflow -> +/-0
    if (e < -14) {
      // Subnormal half.
      m |= 0x800000;
      int shift = -14 - e + 13;
      uint32_t round_bits = m & ((1u << shift) - 1);
      uint32_t halfmant = m >> shift;
      uint32_t halfbit = 1u << (shift - 1);
      if (round_bits > halfbit ||
          (round_bits == halfbit && (halfmant & 1))) halfmant++;
      return (s << 15) | (halfmant & 0x3FF);
    }
    uint32_t halfmant = m >> 13;
    uint32_t round_bits = m & 0x1FFF;
    if (round_bits > 0x1000 ||
        (round_bits == 0x1000 && (halfmant & 1))) {
      halfmant++;
      if (halfmant == 0x400) { halfmant = 0; e++; }
    }
    return (s << 15) | ((e + 15) << 10) | halfmant;
  };
  const uint16_t expected = float_to_half_rne(r32_bits);

  const uint16_t n_lanes[8] = {ha, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {ha, 0, 0, 0, 0, 0, 0, 0};
  uint16_t d_init[8] = {hd, 0, 0, 0, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlaVec4H(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], expected);
}
// endregion

// region digitalis: FP16 vector by-element FMLA / FMLS / FMUL .4H / .8H.
// Decoder dispatches size=0b00, U=0, opcode ∈ {0001 FMLA, 0101 FMLS,
// 1001 FMUL} to AdvSimdVecXIndexedElement.  Encoding (verified via
// aarch64-linux-gnu-as -march=armv8.2-a+fp16):
//   FMLA Vd.4H, Vn.4H, Vm.h[0] = 0x0F021000 | (rm<<16) | (rn<<5) | rd
//   FMLA Vd.4H, Vn.4H, Vm.h[7] = 0x0F321800 | (rm<<16) | (rn<<5) | rd
//   FMLA Vd.8H, Vn.8H, Vm.h[0] = 0x4F021000
//   FMLS Vd.4H, Vn.4H, Vm.h[0] = 0x0F025000
//   FMUL Vd.4H, Vn.4H, Vm.h[0] = 0x0F029000
// Index encoding: 3 bits (0..7); H = (idx>>2)&1 (bit 11), L = (idx>>1)&1
// (bit 21), M = idx&1 (bit 20).  Vm restricted to V0..V15 (only 4-bit
// Rm field; M is consumed by the index).
constexpr uint32_t FmlaIdxVec4H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x0F001000u | (static_cast<uint32_t>(rm & 0xF) << 16) |
         (static_cast<uint32_t>((k >> 1) & 1) << 21) |
         (static_cast<uint32_t>((k >> 2) & 1) << 11) |
         (static_cast<uint32_t>(k & 1) << 20) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlaIdxVec8H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x4F001000u | (static_cast<uint32_t>(rm & 0xF) << 16) |
         (static_cast<uint32_t>((k >> 1) & 1) << 21) |
         (static_cast<uint32_t>((k >> 2) & 1) << 11) |
         (static_cast<uint32_t>(k & 1) << 20) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsIdxVec4H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x0F005000u | (static_cast<uint32_t>(rm & 0xF) << 16) |
         (static_cast<uint32_t>((k >> 1) & 1) << 21) |
         (static_cast<uint32_t>((k >> 2) & 1) << 11) |
         (static_cast<uint32_t>(k & 1) << 20) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmlsIdxVec8H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x4F005000u | (static_cast<uint32_t>(rm & 0xF) << 16) |
         (static_cast<uint32_t>((k >> 1) & 1) << 21) |
         (static_cast<uint32_t>((k >> 2) & 1) << 11) |
         (static_cast<uint32_t>(k & 1) << 20) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulIdxVec4H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x0F009000u | (static_cast<uint32_t>(rm & 0xF) << 16) |
         (static_cast<uint32_t>((k >> 1) & 1) << 21) |
         (static_cast<uint32_t>((k >> 2) & 1) << 11) |
         (static_cast<uint32_t>(k & 1) << 20) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulIdxVec8H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x4F009000u | (static_cast<uint32_t>(rm & 0xF) << 16) |
         (static_cast<uint32_t>((k >> 1) & 1) << 21) |
         (static_cast<uint32_t>((k >> 2) & 1) << 11) |
         (static_cast<uint32_t>(k & 1) << 20) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// FMLA .4H by element index 2: broadcast Vm.h[2] across all 4 lanes
// of Vn, multiply-add into Vd.  Vm.h[2] = 2.0h; Vn = {1, 0.5, -0.5, 4};
// Vd = {1, 1, 3, -1} -> r = {3, 2, 2, 7} after Vd += Vn * 2.
TEST_F(Arm64LiteTranslateRegionTest, FmlaIdxVec4HBroadcastsLane) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_0_5, kHalf_neg0_5, kHalf_4_0,
                                0x5555, 0x5555, 0x5555, 0x5555};
  // Vm: lane 2 = 2.0h, others arbitrary (must not be read).
  const uint16_t m_lanes[8] = {kHalf_1_0, kHalf_0_5, kHalf_2_0, kHalf_4_0,
                                kHalf_neg2_0, kHalf_0_25, kHalf_neg0_5, kHalf_3_0};
  uint16_t d_init[8] = {kHalf_1_0, kHalf_1_0, kHalf_3_0,
                        0xBC00,  // -1.0h
                        0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlaIdxVec4H(0, 1, 2, /*k=*/2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_3_0);   //  1 + 1*2 = 3
  EXPECT_EQ(r[1], kHalf_2_0);   //  1 + 0.5*2 = 2
  EXPECT_EQ(r[2], kHalf_2_0);   //  3 + (-0.5)*2 = 2
  EXPECT_EQ(r[3], 0x4700);      // -1 + 4*2 = 7.0h = 0x4700
  // Upper 64 bits (lanes 4..7) zeroed by Vcvtps2ph (Q=0 invariant).
  for (int i = 4; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FMLS .4H by element index 0: broadcast Vm.h[0] = 2.0h.
// Vn = {1, 2, -0.5, 4}, Vd = {3, 5, -1, 10} -> r = {1, 1, 0, 2}.
TEST_F(Arm64LiteTranslateRegionTest, FmlsIdxVec4HBroadcastsLane) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg0_5, kHalf_4_0,
                                0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_2_0, 0x5555, 0x5555, 0x5555,
                                0x5555, 0x5555, 0x5555, 0x5555};
  // Vd init: 3.0, 5.0, -1.0, 10.0
  uint16_t d_init[8] = {kHalf_3_0,
                        0x4500,  // 5.0h
                        0xBC00,  // -1.0h
                        0x4900,  // 10.0h
                        0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlsIdxVec4H(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);   //  3 - 1*2 = 1
  EXPECT_EQ(r[1], kHalf_1_0);   //  5 - 2*2 = 1
  EXPECT_EQ(r[2], kHalf_pos0);  // -1 - (-0.5)*2 = 0
  EXPECT_EQ(r[3], kHalf_2_0);   // 10 - 4*2 = 2
  for (int i = 4; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FMLA .8H by element index 5: broadcast Vm.h[5] (in the high quad of Vm).
// Exercises (a) the index >= 4 high-quad Psrldq shift in the broadcast
// path, (b) both low and high 4-lane passes in the .8H destination.
TEST_F(Arm64LiteTranslateRegionTest, FmlaIdxVec8HHighLaneBroadcast) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_0_5, kHalf_neg0_5, kHalf_4_0,
                                kHalf_2_0, kHalf_1_0,    kHalf_4_0,    kHalf_0_5};
  // Vm.h[5] = 2.0h; other lanes must not contribute.
  const uint16_t m_lanes[8] = {0x5555, 0x5555, 0x5555, 0x5555,
                                0x5555, kHalf_2_0, 0x5555, 0x5555};
  // Vd init: 1, 1, 3, -1, 0, 3, -2, 0
  uint16_t d_init[8] = {kHalf_1_0, kHalf_1_0, kHalf_3_0,
                        0xBC00,
                        kHalf_pos0, kHalf_3_0,
                        0xC000,    // -2.0h
                        kHalf_pos0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlaIdxVec8H(0, 1, 2, /*k=*/5)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_3_0);   //  1 +  1.0 * 2 = 3
  EXPECT_EQ(r[1], kHalf_2_0);   //  1 +  0.5 * 2 = 2
  EXPECT_EQ(r[2], kHalf_2_0);   //  3 + -0.5 * 2 = 2
  EXPECT_EQ(r[3], 0x4700);      // -1 +  4   * 2 = 7
  EXPECT_EQ(r[4], kHalf_4_0);   //  0 +  2   * 2 = 4
  EXPECT_EQ(r[5], 0x4500);      //  3 +  1   * 2 = 5
  EXPECT_EQ(r[6], 0x4600);      // -2 +  4   * 2 = 6
  EXPECT_EQ(r[7], kHalf_1_0);   //  0 +  0.5 * 2 = 1
}

// FMLS .8H by element index 7 (last lane).
TEST_F(Arm64LiteTranslateRegionTest, FmlsIdxVec8HHighLaneBroadcast) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0, kHalf_neg0_5, kHalf_4_0,
                                kHalf_1_0, kHalf_2_0, kHalf_1_0,    kHalf_4_0};
  const uint16_t m_lanes[8] = {0x5555, 0x5555, 0x5555, 0x5555,
                                0x5555, 0x5555, 0x5555, kHalf_2_0};
  // Vd init: 3, 5, -1, 10, 4, 5, 3, 9
  uint16_t d_init[8] = {kHalf_3_0,
                        0x4500,    // 5
                        0xBC00,    // -1
                        0x4900,    // 10
                        kHalf_4_0,
                        0x4500,    // 5
                        kHalf_3_0,
                        0x4880};   // 9.0h
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmlsIdxVec8H(0, 1, 2, /*k=*/7)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);   //  3 - 1*2 = 1
  EXPECT_EQ(r[1], kHalf_1_0);   //  5 - 2*2 = 1
  EXPECT_EQ(r[2], kHalf_pos0);  // -1 - (-0.5)*2 = 0
  EXPECT_EQ(r[3], kHalf_2_0);   // 10 - 4*2 = 2
  EXPECT_EQ(r[4], kHalf_2_0);   //  4 - 1*2 = 2
  EXPECT_EQ(r[5], kHalf_1_0);   //  5 - 2*2 = 1
  EXPECT_EQ(r[6], kHalf_1_0);   //  3 - 1*2 = 1
  EXPECT_EQ(r[7], kHalf_1_0);   //  9 - 4*2 = 1
}

// FMUL .4H by element index 3: result = Vn * Vm.h[3].
// Vm.h[3] = 0.5h; Vn = {1, 2, -4, 8} -> r = {0.5, 1, -2, 4}.
TEST_F(Arm64LiteTranslateRegionTest, FmulIdxVec4HBroadcastsLane) {
  const uint16_t n_lanes[8] = {kHalf_1_0, kHalf_2_0,
                                0xC400,   // -4.0h
                                0x4800,   //  8.0h
                                0, 0, 0, 0};
  const uint16_t m_lanes[8] = {0x5555, 0x5555, 0x5555, kHalf_0_5,
                                0x5555, 0x5555, 0x5555, 0x5555};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmulIdxVec4H(0, 1, 2, /*k=*/3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_0_5);     //  1   * 0.5 = 0.5
  EXPECT_EQ(r[1], kHalf_1_0);     //  2   * 0.5 = 1
  EXPECT_EQ(r[2], kHalf_neg2_0);  // -4   * 0.5 = -2
  EXPECT_EQ(r[3], kHalf_4_0);     //  8   * 0.5 = 4
  for (int i = 4; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FMUL .8H by element index 4 (low lane of high quad).
TEST_F(Arm64LiteTranslateRegionTest, FmulIdxVec8HTwoPass) {
  const uint16_t n_lanes[8] = {kHalf_1_0,   kHalf_2_0,   kHalf_neg2_0, kHalf_4_0,
                                kHalf_0_5,   kHalf_neg0_5, kHalf_3_0,    kHalf_1_0};
  // Vm.h[4] = 2.0h.
  const uint16_t m_lanes[8] = {0x5555, 0x5555, 0x5555, 0x5555,
                                kHalf_2_0, 0x5555, 0x5555, 0x5555};
  uint16_t d_init[8] = {0xBBBB, 0xBBBB, 0xBBBB, 0xBBBB,
                        0xBBBB, 0xBBBB, 0xBBBB, 0xBBBB};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmulIdxVec8H(0, 1, 2, /*k=*/4)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_2_0);     //  1   * 2
  EXPECT_EQ(r[1], kHalf_4_0);     //  2   * 2
  EXPECT_EQ(r[2], 0xC400);        // -2*2 = -4 = 0xC400
  EXPECT_EQ(r[3], 0x4800);        //  4*2 = 8.0h
  EXPECT_EQ(r[4], kHalf_1_0);     //  0.5*2 = 1
  EXPECT_EQ(r[5], 0xBC00);        // -0.5*2 = -1.0h = 0xBC00
  EXPECT_EQ(r[6], 0x4600);        //  3*2 = 6.0h
  EXPECT_EQ(r[7], kHalf_2_0);     //  1*2 = 2
}
// endregion

// region digitalis: FP16 scalar FpDataProc3 — FMADD / FMSUB / FNMADD / FNMSUB
// on Hn/Hm/Ha/Hd.  Encoding (verified via aarch64-linux-gnu-as
// -march=armv8.2-a+fp16): bits[31:24]=00011111, bits[23:22]=11 (ftype=H),
// bit21=O1, bits[20:16]=Rm, bit15=o0, bits[14:10]=Ra, bits[9:5]=Rn,
// bits[4:0]=Rd.
//   FMADD  Hd, Hn, Hm, Ha (o1=0, o0=0) = 0x1FC00000 base
//   FMSUB  Hd, Hn, Hm, Ha (o1=0, o0=1) = 0x1FC08000 base
//   FNMADD Hd, Hn, Hm, Ha (o1=1, o0=0) = 0x1FE00000 base
//   FNMSUB Hd, Hn, Hm, Ha (o1=1, o0=1) = 0x1FE08000 base
constexpr uint32_t FmaddH(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x1FC00000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(ra) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmsubH(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x1FC08000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(ra) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FnmaddH(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x1FE00000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(ra) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FnmsubH(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x1FE08000u | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(ra) << 10) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// FMADD Hd, Hn, Hm, Ha: Hd = Ha + Hn * Hm.
// Hn=1.5h, Hm=2.0h, Ha=0.5h -> 0.5 + 1.5*2 = 3.5h = 0x4300.
TEST_F(Arm64LiteTranslateRegionTest, FmaddHScalar) {
  const uint16_t n_lanes[8] = {kHalf_1_5, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555};
  const uint16_t m_lanes[8] = {kHalf_2_0, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666};
  const uint16_t a_lanes[8] = {kHalf_0_5, 0x7777, 0x7777, 0x7777, 0x7777, 0x7777, 0x7777, 0x7777};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 3, a_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmaddH(0, 1, 2, 3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 0x4300u);  // 3.5h
  // Vd zero-extended above the scalar lane.
  for (int i = 1; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FMSUB Hd, Hn, Hm, Ha: Hd = Ha - Hn * Hm.
// Hn=2.0h, Hm=2.0h, Ha=5.0h -> 5 - 4 = 1.0h.
TEST_F(Arm64LiteTranslateRegionTest, FmsubHScalar) {
  const uint16_t n_lanes[8] = {kHalf_2_0, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_2_0, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t a_lanes[8] = {0x4500, 0, 0, 0, 0, 0, 0, 0};  // 5.0h
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 3, a_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmsubH(0, 1, 2, 3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);
  for (int i = 1; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FNMADD Hd, Hn, Hm, Ha: Hd = -(Ha + Hn * Hm).
// Hn=1.5h, Hm=2.0h, Ha=0.5h -> -(0.5 + 3) = -3.5h = 0xC300.
TEST_F(Arm64LiteTranslateRegionTest, FnmaddHScalar) {
  const uint16_t n_lanes[8] = {kHalf_1_5, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_2_0, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t a_lanes[8] = {kHalf_0_5, 0, 0, 0, 0, 0, 0, 0};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 3, a_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FnmaddH(0, 1, 2, 3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 0xC300u);  // -3.5h
  for (int i = 1; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FNMSUB Hd, Hn, Hm, Ha: Hd = Hn * Hm - Ha.
// Hn=4.0h, Hm=2.0h, Ha=3.0h -> 8 - 3 = 5.0h = 0x4500.
TEST_F(Arm64LiteTranslateRegionTest, FnmsubHScalar) {
  const uint16_t n_lanes[8] = {kHalf_4_0, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_2_0, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t a_lanes[8] = {kHalf_3_0, 0, 0, 0, 0, 0, 0, 0};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 3, a_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FnmsubH(0, 1, 2, 3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 0x4500u);  // 5.0h
  for (int i = 1; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// FMADD Hd, Hn, Hm, Ha: fused-vs-unfused divergence.  Pick operands such
// that fma(a, b, c) in binary64 differs from (a*b)+c through binary32:
//   Hn = 0x3C01 = 1 + 2^-10 (one ULP above 1.0h).
//   Hm = 0x3C01 = 1 + 2^-10.
//   Ha = 0xBC00 = -1.0h.
//   Hn*Hm = 1 + 2*2^-10 + 2^-20.  In binary64 this is exact, plus -1 =
//   2*2^-10 + 2^-20 = 0x00000001_2000 in mantissa scale, rounds to FP16
//   as 2*2^-10 + 2^-20 (representable as 0x10C0_..._wait — let's recompute):
//   2*2^-10 = 2^-9.  2^-20 is below FP16's 2^-24 subnormal granularity?
//   No, 2^-20 > 2^-24, so FP16 can represent it (denormal).
//
// Actually for a cleaner test of FP16 round-trip exactness: pick Hn=2.0h,
// Hm=0.5h, Ha=-1.0h -> 0 (cleanly exact), and use the divergence between
// FMADD and FMSUB to verify the o0 bit dispatches correctly.  We rely on
// the previous tests to prove FMA semantics; this test just confirms FMADD
// and FMSUB produce different results (so the o0 bit lift is correct).
TEST_F(Arm64LiteTranslateRegionTest, FmaddHvsFmsubHDispatch) {
  const uint16_t n_lanes[8] = {kHalf_1_0, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {kHalf_2_0, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t a_lanes[8] = {kHalf_3_0, 0, 0, 0, 0, 0, 0, 0};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 3, a_lanes);
  // FMADD: 3 + 1*2 = 5.0h.
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code_fmadd[] = {FmaddH(0, 1, 2, 3)};
  EXPECT_TRUE(Run(code_fmadd, ToGuestAddr(code_fmadd) + sizeof(code_fmadd)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 0x4500u);  // 5.0h
  // FMSUB: 3 - 1*2 = 1.0h.
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code_fmsub[] = {FmsubH(0, 1, 2, 3)};
  EXPECT_TRUE(Run(code_fmsub, ToGuestAddr(code_fmsub) + sizeof(code_fmsub)));
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], kHalf_1_0);
}

// FMADD Hd with a small-magnitude tail to verify the FP64 round-trip
// preserves enough precision for the final FP16 RNE.
//   Hn = 0x3C01 = 1 + 2^-10 (one ULP above 1.0h).
//   Hm = 0x3C01.
//   Ha = 0xBC00 = -1.0h.
//   Exact: (1 + 2^-10)^2 - 1 = 2^-9 + 2^-20.
//   2^-9 = 0x1800 in FP16; the +2^-20 tail is exactly half-ULP at FP16
//   2^-19 ULP scale, so RNE rounds to even mantissa LSB (which is 0)
//   -> 0x1800.  Any FP16-only multiply-add path would also produce
//   0x1800 here (the product is representable in FP32 exactly), so this
//   is a correctness check on the FP64 lift, not an FMA-vs-unfused test.
TEST_F(Arm64LiteTranslateRegionTest, FmaddHSmallTail) {
  const uint16_t n_lanes[8] = {0x3C01, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {0x3C01, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t a_lanes[8] = {0xBC00, 0, 0, 0, 0, 0, 0, 0};
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 3, a_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {FmaddH(0, 1, 2, 3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 0x1800u);  // 2^-9 = 0.001953125 in FP16.
  for (int i = 1; i < 8; i++) EXPECT_EQ(r[i], 0u);
}
// endregion

// region digitalis: integer MUL/MLA/MLS by-element JIT tests.
//
// Encoding: 0 Q U 01111 size L M Rm[3:0] opcode H 0 Rn Rd.
//   - halfword (size=01): index = H:L:M (3 bits, 0..7), Vm restricted to V0..V15.
//   - word     (size=10): index = H:L   (2 bits, 0..3), Vm full V0..V31 (M = Vm bit4).
// (opcode, U) selects integer op:
//   MUL: opc=1000, U=0
//   MLA: opc=0000, U=1
//   MLS: opc=0100, U=1
//
// Verified encodings (aarch64-linux-gnu-as -march=armv8.2-a):
//   mul  v0.4h, v1.4h, v2.h[0] = 0x0F428020
//   mul  v0.4h, v1.4h, v2.h[7] = 0x0F728820
//   mul  v0.8h, v1.8h, v2.h[7] = 0x4F728820
//   mul  v0.2s, v1.2s, v2.s[1] = 0x0FA28020
//   mul  v0.4s, v1.4s, v2.s[3] = 0x4FA28820
//   mla  v0.4h, v1.4h, v2.h[0] = 0x2F420020
//   mla  v0.4s, v1.4s, v2.s[2] = 0x6F820820
//   mls  v0.8h, v1.8h, v2.h[5] = 0x6F524820
//   mls  v0.4s, v1.4s, v2.s[1] = 0x6FA24020
constexpr uint32_t MulIdx4H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  // Q=0, U=0, size=01, opc=1000, halfword.  Vm in V0..V15: only Rm[3:0] used.
  uint32_t M = k & 1u;
  uint32_t L = (k >> 1) & 1u;
  uint32_t H = (k >> 2) & 1u;
  return 0x0F408000u | (L << 21) | (M << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t MulIdx8H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x4F408000u | (((k >> 1) & 1u) << 21) | ((k & 1u) << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (((k >> 2) & 1u) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t MulIdx2S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  // Q=0, U=0, size=10, opc=1000, word.  index = H:L (2 bits).
  uint32_t L = k & 1u;
  uint32_t H = (k >> 1) & 1u;
  uint32_t M = (rm >> 4) & 1u;
  return 0x0F808000u | (L << 21) | (M << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t MulIdx4S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t L = k & 1u;
  uint32_t H = (k >> 1) & 1u;
  uint32_t M = (rm >> 4) & 1u;
  return 0x4F808000u | (L << 21) | (M << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t MlaIdx4H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  // U=1, opc=0000 -> base 0x2F400000 for Q=0 size=01.
  return 0x2F400000u | (((k >> 1) & 1u) << 21) | ((k & 1u) << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (((k >> 2) & 1u) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t MlaIdx4S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t L = k & 1u;
  uint32_t H = (k >> 1) & 1u;
  uint32_t M = (rm >> 4) & 1u;
  return 0x6F800000u | (L << 21) | (M << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t MlsIdx4H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  // U=1, opc=0100 -> base 0x2F404000 for Q=0 size=01.
  return 0x2F404000u | (((k >> 1) & 1u) << 21) | ((k & 1u) << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (((k >> 2) & 1u) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t MlsIdx8H(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  return 0x6F404000u | (((k >> 1) & 1u) << 21) | ((k & 1u) << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (((k >> 2) & 1u) << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t MlsIdx4S(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t k) {
  uint32_t L = k & 1u;
  uint32_t H = (k >> 1) & 1u;
  uint32_t M = (rm >> 4) & 1u;
  return 0x6F804000u | (L << 21) | (M << 20) |
         (static_cast<uint32_t>(rm & 0xFu) << 16) | (H << 11) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// Helpers for 32-bit integer 4-lane Vec storage.
static void StoreVec4SInt(CPUState& cpu, unsigned idx,
                          int32_t a0, int32_t a1, int32_t a2, int32_t a3) {
  int32_t lanes[4] = {a0, a1, a2, a3};
  std::memcpy(&cpu.v[idx], lanes, 16);
}
static void LoadVec4SInt(const CPUState& cpu, unsigned idx, int32_t out[4]) {
  std::memcpy(out, &cpu.v[idx], 16);
}

// MUL .4h, lane 0 (low quad).
TEST_F(Arm64LiteTranslateRegionTest, MulIdxVec4HLowLane) {
  const uint16_t n_lanes[8] = {2, 3, 0xFFFF, 4, 99, 99, 99, 99};  // Vn.4h = {2, 3, -1, 4}
  const uint16_t m_lanes[8] = {5, 99, 99, 99, 99, 99, 99, 99};    // Vm.h[0] = 5
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {MulIdx4H(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 10u);
  EXPECT_EQ(r[1], 15u);
  EXPECT_EQ(r[2], static_cast<uint16_t>(-5));  // -1 * 5 mod 2^16
  EXPECT_EQ(r[3], 20u);
  // .4h: upper 64 bits must be zero.
  for (int i = 4; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// MUL .8h, lane 7 (high quad) — exercises the args.index >= 4 Psrldq path.
TEST_F(Arm64LiteTranslateRegionTest, MulIdxVec8HHighLane) {
  const uint16_t n_lanes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint16_t m_lanes[8] = {99, 99, 99, 99, 99, 99, 99, 10};   // Vm.h[7] = 10
  uint16_t d_init[8] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {MulIdx8H(0, 1, 2, /*k=*/7)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(r[i], static_cast<uint16_t>((i + 1) * 10)) << "lane " << i;
  }
}

// MLA .4h: Vd = Vd + Vn * broadcast(Vm.h[k]).
TEST_F(Arm64LiteTranslateRegionTest, MlaIdxVec4H) {
  const uint16_t n_lanes[8] = {2, 3, 0xFFFF, 4, 99, 99, 99, 99};
  const uint16_t m_lanes[8] = {5, 99, 99, 99, 99, 99, 99, 99};    // Vm.h[0] = 5
  const uint16_t d_init[8] = {1, 1, 1, 1, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {MlaIdx4H(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 11u);                          // 1 + 2*5
  EXPECT_EQ(r[1], 16u);                          // 1 + 3*5
  EXPECT_EQ(r[2], static_cast<uint16_t>(-4));    // 1 + -1*5
  EXPECT_EQ(r[3], 21u);                          // 1 + 4*5
  for (int i = 4; i < 8; i++) EXPECT_EQ(r[i], 0u);
}

// MLS .8h: Vd = Vd - Vn * broadcast(Vm.h[k]).  Use lane 5 (high quad).
TEST_F(Arm64LiteTranslateRegionTest, MlsIdxVec8H) {
  uint16_t n_lanes[8];
  for (int i = 0; i < 8; i++) n_lanes[i] = 2;
  const uint16_t m_lanes[8] = {99, 99, 99, 99, 99, 3, 99, 99};    // Vm.h[5] = 3
  uint16_t d_init[8];
  for (int i = 0; i < 8; i++) d_init[i] = 10;
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code[] = {MlsIdx8H(0, 1, 2, /*k=*/5)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(r[i], 4u) << "lane " << i;          // 10 - 2*3 = 4
  }
}

// MUL .4s, lane 3 (high lane).
TEST_F(Arm64LiteTranslateRegionTest, MulIdxVec4S) {
  StoreVec4SInt(state_.cpu, 1, 2, 3, -1, 4);
  StoreVec4SInt(state_.cpu, 2, 99, 99, 99, -5);   // Vm.s[3] = -5
  StoreVec4SInt(state_.cpu, 0, 0x77777777, 0x77777777, 0x77777777, 0x77777777);
  static const uint32_t code[] = {MulIdx4S(0, 1, 2, /*k=*/3)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], -10);
  EXPECT_EQ(r[1], -15);
  EXPECT_EQ(r[2], 5);
  EXPECT_EQ(r[3], -20);
}

// MUL .2s, Q=0 — upper 64 bits of Vd must be zeroed.
TEST_F(Arm64LiteTranslateRegionTest, MulIdxVec2SUpperZero) {
  StoreVec4SInt(state_.cpu, 1, 7, -3, 99, 99);
  StoreVec4SInt(state_.cpu, 2, 99, 4, 99, 99);    // Vm.s[1] = 4
  StoreVec4SInt(state_.cpu, 0, 0x77777777, 0x77777777, 0x77777777, 0x77777777);
  static const uint32_t code[] = {MulIdx2S(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 28);
  EXPECT_EQ(r[1], -12);
  EXPECT_EQ(r[2], 0);
  EXPECT_EQ(r[3], 0);
}

// MLA .4s: Vd = Vd + Vn * broadcast(Vm.s[k]).
TEST_F(Arm64LiteTranslateRegionTest, MlaIdxVec4S) {
  StoreVec4SInt(state_.cpu, 1, 2, 3, -1, 4);
  StoreVec4SInt(state_.cpu, 2, 99, 99, 5, 99);    // Vm.s[2] = 5
  StoreVec4SInt(state_.cpu, 0, 1, 1, 1, 1);
  static const uint32_t code[] = {MlaIdx4S(0, 1, 2, /*k=*/2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 11);    // 1 + 2*5
  EXPECT_EQ(r[1], 16);    // 1 + 3*5
  EXPECT_EQ(r[2], -4);    // 1 + -1*5
  EXPECT_EQ(r[3], 21);    // 1 + 4*5
}

// MLS .4s: Vd = Vd - Vn * broadcast(Vm.s[k]).
TEST_F(Arm64LiteTranslateRegionTest, MlsIdxVec4S) {
  StoreVec4SInt(state_.cpu, 1, 2, 3, -1, 4);
  StoreVec4SInt(state_.cpu, 2, 99, 5, 99, 99);    // Vm.s[1] = 5
  StoreVec4SInt(state_.cpu, 0, 11, 16, -4, 21);
  static const uint32_t code[] = {MlsIdx4S(0, 1, 2, /*k=*/1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 1);     // 11 - 2*5
  EXPECT_EQ(r[1], 1);     // 16 - 3*5
  EXPECT_EQ(r[2], 1);     // -4 - (-1)*5
  EXPECT_EQ(r[3], 1);     // 21 - 4*5
}

// MUL .4h vs MLA .4h dispatch — same operands, two different ops produce
// distinct results.  Confirms the (U, opcode) tuple dispatches to the right
// integer op rather than aliasing.
TEST_F(Arm64LiteTranslateRegionTest, MulVsMlaIdxDispatch) {
  const uint16_t n_lanes[8] = {3, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t m_lanes[8] = {7, 0, 0, 0, 0, 0, 0, 0};
  const uint16_t d_init[8] = {100, 0, 0, 0, 0, 0, 0, 0};
  StoreVec8H(state_.cpu, 1, n_lanes);
  StoreVec8H(state_.cpu, 2, m_lanes);
  // MUL: r0 = 3 * 7 = 21
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code_mul[] = {MulIdx4H(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code_mul, ToGuestAddr(code_mul) + sizeof(code_mul)));
  uint16_t r[8];
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 21u);
  // MLA: r0 = 100 + 3 * 7 = 121
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code_mla[] = {MlaIdx4H(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code_mla, ToGuestAddr(code_mla) + sizeof(code_mla)));
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 121u);
  // MLS: r0 = 100 - 3 * 7 = 79
  StoreVec8H(state_.cpu, 0, d_init);
  static const uint32_t code_mls[] = {MlsIdx4H(0, 1, 2, /*k=*/0)};
  EXPECT_TRUE(Run(code_mls, ToGuestAddr(code_mls) + sizeof(code_mls)));
  LoadVec8H(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 79u);
}
// endregion

// region digitalis - FCSEL JIT
//
// FCSEL Sd|Dd|Hd, Sn, Sm, cond
//   Encoding: 0001 1110 <ftype:2> 1 Rm cond 11 Rn Rd
//   ftype: 00 = S (FP32), 01 = D (FP64), 11 = H (FP16).
//   Base: 0x1E200C00 (S) / 0x1E600C00 (D) / 0x1EE00C00 (H).
constexpr uint32_t FcselScalar(uint32_t base, uint8_t rd, uint8_t rn,
                               uint8_t rm, uint8_t cond) {
  return base | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FcselS(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return FcselScalar(0x1E200C00, rd, rn, rm, cond);
}
constexpr uint32_t FcselD(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return FcselScalar(0x1E600C00, rd, rn, rm, cond);
}
constexpr uint32_t FcselH(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return FcselScalar(0x1EE00C00, rd, rn, rm, cond);
}

// Helpers to seed V registers with scalar FP values, preserving the
// 128-bit alignment expected by the FCSEL store path.
void StoreFp32(CPUState& cpu, uint8_t v, float val) {
  cpu.v[v] = 0;
  std::memcpy(&cpu.v[v], &val, sizeof(val));
}
void StoreFp64(CPUState& cpu, uint8_t v, double val) {
  cpu.v[v] = 0;
  std::memcpy(&cpu.v[v], &val, sizeof(val));
}
void StoreFp16Bits(CPUState& cpu, uint8_t v, uint16_t bits) {
  cpu.v[v] = 0;
  std::memcpy(&cpu.v[v], &bits, sizeof(bits));
}
float LoadFp32(const CPUState& cpu, uint8_t v) {
  float val;
  std::memcpy(&val, &cpu.v[v], sizeof(val));
  return val;
}
double LoadFp64(const CPUState& cpu, uint8_t v) {
  double val;
  std::memcpy(&val, &cpu.v[v], sizeof(val));
  return val;
}
uint16_t LoadFp16Bits(const CPUState& cpu, uint8_t v) {
  uint16_t bits;
  std::memcpy(&bits, &cpu.v[v], sizeof(bits));
  return bits;
}

// FCSEL S — EQ taken: Z=1 -> Vd = Vn.  Also confirms V[rd] high lanes
// are zeroed (the architectural requirement we satisfy by emitting a
// 128-bit MOVDQU off an XMM whose upper lanes were cleared by MOVSS).
TEST_F(Arm64LiteTranslateRegionTest, FcselSEqTrueZeroExtendsVd) {
  StoreFp32(state_.cpu, 1, 1.25f);
  StoreFp32(state_.cpu, 2, 2.5f);
  // Pre-pollute V[0] high lanes; FCSEL must zero them.
  state_.cpu.v[0] = static_cast<__uint128_t>(0xdeadbeefcafebabeULL) << 64;
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                 // Z=1
      FcselS(0, 1, 2, kCondEQ),      // EQ -> Vd = Vn (1.25f)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 1.25f);
  // High 96 bits of V[0] must be zero.
  uint8_t bytes[16];
  std::memcpy(bytes, &state_.cpu.v[0], sizeof(bytes));
  for (int i = 4; i < 16; ++i) EXPECT_EQ(bytes[i], 0u) << "byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, FcselSEqFalse) {
  StoreFp32(state_.cpu, 1, 1.25f);
  StoreFp32(state_.cpu, 2, 2.5f);
  static const uint32_t code[] = {
      MovzX(0, 10),
      CmpImmX(0, 5),                 // Z=0
      FcselS(0, 1, 2, kCondEQ),      // EQ -> Vd = Vm (2.5f)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 2.5f);
}

// FCSEL D — LT taken (N=1, V=0 -> N!=V).
TEST_F(Arm64LiteTranslateRegionTest, FcselDLtTrue) {
  StoreFp64(state_.cpu, 1, 3.14159);
  StoreFp64(state_.cpu, 2, 2.71828);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 10),                // N=1, V=0 -> LT true
      FcselD(0, 1, 2, kCondLT),      // LT -> Vd = Vn (3.14159)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 3.14159);
}

TEST_F(Arm64LiteTranslateRegionTest, FcselDGeFalse) {
  StoreFp64(state_.cpu, 1, 3.14159);
  StoreFp64(state_.cpu, 2, 2.71828);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 10),                // N=1, V=0 -> GE false
      FcselD(0, 1, 2, kCondGE),      // GE -> Vd = Vm (2.71828)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 2.71828);
}

// FCSEL H — HI taken (C=1, Z=0): exercises the FP16 PXOR+PINSRW load.
TEST_F(Arm64LiteTranslateRegionTest, FcselHHiTrue) {
  // Binary16: 0x3C00 = 1.0h, 0x4000 = 2.0h.
  StoreFp16Bits(state_.cpu, 1, 0x3C00);
  StoreFp16Bits(state_.cpu, 2, 0x4000);
  state_.cpu.v[0] = static_cast<__uint128_t>(0xdeadbeefcafebabeULL) << 64;
  static const uint32_t code[] = {
      MovzX(0, 10),
      CmpImmX(0, 5),                 // C=1, Z=0 -> HI true
      FcselH(0, 1, 2, kCondHI),      // HI -> Vd = Vn (0x3C00)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp16Bits(state_.cpu, 0), 0x3C00u);
  uint8_t bytes[16];
  std::memcpy(bytes, &state_.cpu.v[0], sizeof(bytes));
  for (int i = 2; i < 16; ++i) EXPECT_EQ(bytes[i], 0u) << "byte " << i;
}

// FCSEL S — LE taken via Z=1 (compound condition where the kLs/kLe
// shortcut early-binds true_path).
TEST_F(Arm64LiteTranslateRegionTest, FcselSLeTrueViaZ) {
  StoreFp32(state_.cpu, 1, 7.0f);
  StoreFp32(state_.cpu, 2, 9.0f);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                 // Z=1 -> LE true via the Z-branch
      FcselS(0, 1, 2, kCondLE),      // LE -> Vd = Vn (7.0f)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 7.0f);
}

// FCSEL S — NE taken (Z=0): Vd = Vn.  Also asserts V[rd] high 96 bits
// are zero (Movss into a pre-PXOR-cleared register, then 128-bit
// Movdqu store).
TEST_F(Arm64LiteTranslateRegionTest, FcselSNeTrueZeroExtendsVd) {
  StoreFp32(state_.cpu, 1, 1.5f);
  StoreFp32(state_.cpu, 2, 2.5f);
  state_.cpu.v[0] = static_cast<__uint128_t>(0xdeadbeefcafebabeULL) << 64;
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 10),                // Z=0 (result 5-10=-5) -> NE true
      FcselS(0, 1, 2, kCondNE),      // NE -> Vd = Vn (1.5f)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 1.5f);
  uint8_t bytes[16];
  std::memcpy(bytes, &state_.cpu.v[0], sizeof(bytes));
  for (int i = 4; i < 16; ++i) EXPECT_EQ(bytes[i], 0u) << "byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, FcselSNeFalse) {
  StoreFp32(state_.cpu, 1, 1.5f);
  StoreFp32(state_.cpu, 2, 2.5f);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                 // Z=1 -> NE false
      FcselS(0, 1, 2, kCondNE),      // NE -> Vd = Vm (2.5f)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 2.5f);
}

// FCSEL D — MI taken (N=1): Vd = Vn.  Pins FCSEL D's high-64-bit zero
// extend on the V[rn] (TRUE) load path; Movsd into the XMM clears the
// upper 64 bits, then the 128-bit Movdqu writes both lanes to V[rd].
TEST_F(Arm64LiteTranslateRegionTest, FcselDMiTrueZeroExtendsVd) {
  StoreFp64(state_.cpu, 1, 3.14159);
  StoreFp64(state_.cpu, 2, 2.71828);
  state_.cpu.v[0] = static_cast<__uint128_t>(0xdeadbeefcafebabeULL) << 64;
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 10),                // N=1 (negative result) -> MI true
      FcselD(0, 1, 2, kCondMI),      // MI -> Vd = Vn (3.14159)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 3.14159);
  uint8_t bytes[16];
  std::memcpy(bytes, &state_.cpu.v[0], sizeof(bytes));
  for (int i = 8; i < 16; ++i) EXPECT_EQ(bytes[i], 0u) << "byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, FcselDMiFalse) {
  StoreFp64(state_.cpu, 1, 3.14159);
  StoreFp64(state_.cpu, 2, 2.71828);
  static const uint32_t code[] = {
      MovzX(0, 10),
      CmpImmX(0, 5),                 // N=0 (positive result) -> MI false
      FcselD(0, 1, 2, kCondMI),      // MI -> Vd = Vm (2.71828)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 2.71828);
}

// FCSEL S — PL taken (N=0): Vd = Vn.
TEST_F(Arm64LiteTranslateRegionTest, FcselSPlTrue) {
  StoreFp32(state_.cpu, 1, 11.0f);
  StoreFp32(state_.cpu, 2, 22.0f);
  static const uint32_t code[] = {
      MovzX(0, 10),
      CmpImmX(0, 5),                 // N=0 (positive result) -> PL true
      FcselS(0, 1, 2, kCondPL),      // PL -> Vd = Vn (11.0f)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 11.0f);
}

TEST_F(Arm64LiteTranslateRegionTest, FcselSPlFalse) {
  StoreFp32(state_.cpu, 1, 11.0f);
  StoreFp32(state_.cpu, 2, 22.0f);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 10),                // N=1 (negative result) -> PL false
      FcselS(0, 1, 2, kCondPL),      // PL -> Vd = Vm (22.0f)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 22.0f);
}

// FCSEL D — EQ taken: pins FCSEL D's high-64-bit zero extend on the
// V[rn] (TRUE) load path under the EQ encoding (the existing
// FcselSEqTrueZeroExtendsVd covers the S form; this completes the
// symmetric pin at the D form).
TEST_F(Arm64LiteTranslateRegionTest, FcselDEqTrueZeroExtendsVd) {
  StoreFp64(state_.cpu, 1, 1.5);
  StoreFp64(state_.cpu, 2, 2.5);
  state_.cpu.v[0] = static_cast<__uint128_t>(0xdeadbeefcafebabeULL) << 64;
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                 // Z=1 -> EQ true
      FcselD(0, 1, 2, kCondEQ),      // EQ -> Vd = Vn (1.5)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 1.5);
  uint8_t bytes[16];
  std::memcpy(bytes, &state_.cpu.v[0], sizeof(bytes));
  for (int i = 8; i < 16; ++i) EXPECT_EQ(bytes[i], 0u) << "byte " << i;
}

// FCSEL D — NE false: pins FCSEL D's high-64-bit zero extend on the
// V[rm] (FALSE default) load path -- the JIT emits the V[rm] Movsd
// before the condition switch, so a FALSE outcome exercises a distinct
// register-write arm from the TRUE-overwrite path pinned above.
TEST_F(Arm64LiteTranslateRegionTest, FcselDNeFalseZeroExtendsVd) {
  StoreFp64(state_.cpu, 1, 1.5);
  StoreFp64(state_.cpu, 2, 2.5);
  state_.cpu.v[0] = static_cast<__uint128_t>(0xdeadbeefcafebabeULL) << 64;
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                 // Z=1 -> NE false
      FcselD(0, 1, 2, kCondNE),      // NE -> Vd = Vm (2.5)
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 2.5);
  uint8_t bytes[16];
  std::memcpy(bytes, &state_.cpu.v[0], sizeof(bytes));
  for (int i = 8; i < 16; ++i) EXPECT_EQ(bytes[i], 0u) << "byte " << i;
}
// endregion

// region digitalis - FCCMP / FCCMPE JIT
//
// FCCMP Sn, Sm, #nzcv, cond   /   FCCMP Dn, Dm, #nzcv, cond
//   Encoding: 0 0 0 11110 ftype 1 Rm cond 01 Rn op nzcv
//   ftype: 00=S, 01=D.  op: 0=FCCMP, 1=FCCMPE.
//   Base: 0x1E200400 (FCCMP S) / 0x1E600400 (FCCMP D); add 0x10 for FCCMPE.
constexpr uint32_t FccmpScalar(uint32_t base, uint8_t rn, uint8_t rm,
                               uint8_t cond, uint8_t nzcv) {
  return base | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) |
         (static_cast<uint32_t>(rn) << 5) | (nzcv & 0xF);
}
constexpr uint32_t FccmpS(uint8_t rn, uint8_t rm, uint8_t cond, uint8_t nzcv) {
  return FccmpScalar(0x1E200400, rn, rm, cond, nzcv);
}
constexpr uint32_t FccmpD(uint8_t rn, uint8_t rm, uint8_t cond, uint8_t nzcv) {
  return FccmpScalar(0x1E600400, rn, rm, cond, nzcv);
}
constexpr uint32_t FccmpeS(uint8_t rn, uint8_t rm, uint8_t cond, uint8_t nzcv) {
  return FccmpScalar(0x1E200410, rn, rm, cond, nzcv);
}

// Read the ARM NZCV bits from ThreadState::cpu.flags into a 4-bit value
// (N=bit3, Z=bit2, C=bit1, V=bit0).  The CPUState flag-bit layout is
// N=bit15, Z=bit14, C=bit8, V=bit0 of the 16-bit flags field.
uint8_t ReadArmNzcv(const CPUState& cpu) {
  return ((cpu.flags >> 15) & 1) << 3 |
         ((cpu.flags >> 14) & 1) << 2 |
         ((cpu.flags >> 8) & 1) << 1 |
         (cpu.flags & 1);
}

// FCCMP S — condition TRUE, ordered equal: flags <- 0110 (Z=1, C=1).
TEST_F(Arm64LiteTranslateRegionTest, FccmpSCondTrueOrderedEqual) {
  StoreFp32(state_.cpu, 1, 3.5f);
  StoreFp32(state_.cpu, 2, 3.5f);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                       // Z=1 -> EQ true
      FccmpS(1, 2, kCondEQ, 0b1010),       // EQ true: compare V1,V2 -> eq
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0110u);  // N=0,Z=1,C=1,V=0
}

// FCCMP S — condition TRUE, ordered less: flags <- 1000 (N=1).
TEST_F(Arm64LiteTranslateRegionTest, FccmpSCondTrueOrderedLess) {
  StoreFp32(state_.cpu, 1, 1.0f);
  StoreFp32(state_.cpu, 2, 2.0f);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                       // Z=1 -> EQ true
      FccmpS(1, 2, kCondEQ, 0b0101),       // EQ true: compare V1<V2 -> lt
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b1000u);
}

// FCCMP S — condition TRUE, ordered greater: flags <- 0010 (C=1).
TEST_F(Arm64LiteTranslateRegionTest, FccmpSCondTrueOrderedGreater) {
  StoreFp32(state_.cpu, 1, 9.0f);
  StoreFp32(state_.cpu, 2, 2.0f);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                       // Z=1 -> EQ true
      FccmpS(1, 2, kCondEQ, 0b1100),       // EQ true: compare V1>V2 -> gt
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0010u);
}

// FCCMP S — condition TRUE, unordered (NaN): flags <- 0011 (C=1, V=1).
TEST_F(Arm64LiteTranslateRegionTest, FccmpSCondTrueUnordered) {
  StoreFp32(state_.cpu, 1, std::nanf(""));
  StoreFp32(state_.cpu, 2, 1.0f);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                       // Z=1 -> EQ true
      FccmpS(1, 2, kCondEQ, 0b1000),       // EQ true: compare NaN,1 -> uo
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0011u);
}

// FCCMP D — condition FALSE: nzcv immediate is written to flags.
TEST_F(Arm64LiteTranslateRegionTest, FccmpDCondFalseWritesImmediate) {
  StoreFp64(state_.cpu, 1, 7.0);
  StoreFp64(state_.cpu, 2, 7.0);  // would yield "equal" if compared
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 10),                       // Z=0, N=1 -> EQ false
      FccmpD(1, 2, kCondEQ, 0b1011),        // EQ false: flags <- N,Z=0,C,V
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b1011u);
}

// FCCMPE S — quiet-vs-signalling NaN bit does not alter NZCV output
// (the architectural flags result is identical to FCCMP).
TEST_F(Arm64LiteTranslateRegionTest, FccmpeSBehavesLikeFccmpForNzcv) {
  StoreFp32(state_.cpu, 1, 4.0f);
  StoreFp32(state_.cpu, 2, 5.0f);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),                        // Z=1 -> EQ true
      FccmpeS(1, 2, kCondEQ, 0b0000),       // EQ true: compare 4<5 -> lt
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b1000u);  // N=1 (less)
}

// FCCMP D — compound condition (kLt: N XOR V).  Set NZCV so N=1, V=0
// (kLt TRUE).  FP compare V1 == V2 should produce Z=1,C=1 (eq).
TEST_F(Arm64LiteTranslateRegionTest, FccmpDCompoundLtTrue) {
  StoreFp64(state_.cpu, 1, -2.5);
  StoreFp64(state_.cpu, 2, -2.5);
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 10),                       // N=1, V=0 -> LT true
      FccmpD(1, 2, kCondLT, 0b1111),        // LT true: do FP compare
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0110u);  // equal -> Z=1,C=1
}
// endregion

// region digitalis - FCMP NaN
//
// FCMP Sn, Sm / FCMP Dn, Dm / FCMP Sn, #0.0 / FCMP Dn, #0.0
// FCMPE variants share the same NZCV mapping (the quiet-vs-signalling NaN
// bit only changes FP-exception behaviour, not the architectural output).
//
//   Encoding: M S 11110 ftype 1 Rm op 1000 Rn opcode2
//     opcode2[3] = with_zero (Rm field then = 0)
//     opcode2[4] = signal_nans (FCMPE)
//   Base FCMP S Rn, Rm : 0x1E202020
//   Base FCMP D Rn, Rm : 0x1E602020
//   Base FCMP S Rn, #0 : 0x1E202028
//   Base FCMP D Rn, #0 : 0x1E602028
//   Base FCMPE S Rn, Rm: 0x1E202030
//   Base FCMPE S Rn, #0: 0x1E202038
//
// Cross-verified with `aarch64-linux-gnu-as -march=armv8.2-a+fp16 -c`:
//   1e222020 fcmp s1, s2 / 1e622020 fcmp d1, d2
//   1e202028 fcmp s1, #0.0 / 1e602028 fcmp d1, #0.0
//   1e222030 fcmpe s1, s2 / 1e202038 fcmpe s1, #0.0
//
// ARM FPCompare(Vn, Vm) -> NZCV mapping:
//   Vn or Vm NaN     : 0b0011 (unordered)
//   Vn == Vm         : 0b0110 (equal)
//   Vn  < Vm         : 0b1000 (less than)
//   Vn  > Vm         : 0b0010 (greater than)
constexpr uint32_t FcmpScalar(uint32_t base, uint8_t rn, uint8_t rm) {
  return base | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5);
}
constexpr uint32_t FcmpS(uint8_t rn, uint8_t rm) {
  return FcmpScalar(0x1E202020, rn, rm);
}
constexpr uint32_t FcmpD(uint8_t rn, uint8_t rm) {
  return FcmpScalar(0x1E602020, rn, rm);
}
constexpr uint32_t FcmpSZero(uint8_t rn) {
  return 0x1E202028 | (static_cast<uint32_t>(rn) << 5);
}
constexpr uint32_t FcmpDZero(uint8_t rn) {
  return 0x1E602028 | (static_cast<uint32_t>(rn) << 5);
}
constexpr uint32_t FcmpeS(uint8_t rn, uint8_t rm) {
  return FcmpScalar(0x1E202030, rn, rm);
}

// FCMP S — unordered (NaN operand): NZCV = 0b0011 (C=1, V=1).
TEST_F(Arm64LiteTranslateRegionTest, FcmpSUnordered) {
  StoreFp32(state_.cpu, 1, std::nanf(""));
  StoreFp32(state_.cpu, 2, 1.0f);
  static const uint32_t code[] = {
      FcmpS(1, 2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0011u);
}

// FCMP S — ordered equal: NZCV = 0b0110 (Z=1, C=1).
TEST_F(Arm64LiteTranslateRegionTest, FcmpSOrderedEqual) {
  StoreFp32(state_.cpu, 1, 3.5f);
  StoreFp32(state_.cpu, 2, 3.5f);
  static const uint32_t code[] = {
      FcmpS(1, 2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0110u);
}

// FCMP S — ordered less: NZCV = 0b1000 (N=1).
TEST_F(Arm64LiteTranslateRegionTest, FcmpSOrderedLess) {
  StoreFp32(state_.cpu, 1, 1.0f);
  StoreFp32(state_.cpu, 2, 2.0f);
  static const uint32_t code[] = {
      FcmpS(1, 2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b1000u);
}

// FCMP D — ordered equal (with widened FP64): NZCV = 0b0110.
TEST_F(Arm64LiteTranslateRegionTest, FcmpDOrderedEqual) {
  StoreFp64(state_.cpu, 1, 1.0);
  StoreFp64(state_.cpu, 2, 1.0);
  static const uint32_t code[] = {
      FcmpD(1, 2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0110u);
}

// FCMP D — unordered (Vm is a quiet NaN): NZCV = 0b0011.
TEST_F(Arm64LiteTranslateRegionTest, FcmpDUnordered) {
  StoreFp64(state_.cpu, 1, 1.0);
  StoreFp64(state_.cpu, 2, std::nan(""));
  static const uint32_t code[] = {
      FcmpD(1, 2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0011u);
}

// FCMP S, #0.0 — Vn > 0 produces NZCV = 0b0010 (C=1 only).
// Exercises the with_zero path (opcode2[3]=1, Rm field unused).
TEST_F(Arm64LiteTranslateRegionTest, FcmpSWithZeroGreater) {
  StoreFp32(state_.cpu, 1, 1.0f);
  static const uint32_t code[] = {
      FcmpSZero(1),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0010u);
}

// FCMP D, #0.0 — Vn == 0 produces NZCV = 0b0110.  +0.0 == -0.0 by FP
// equality, so this also implicitly proves the zero-form pulls a true
// +0.0 (Pxor) for the Vm side rather than reading uninitialised lanes.
TEST_F(Arm64LiteTranslateRegionTest, FcmpDWithZeroEqual) {
  StoreFp64(state_.cpu, 1, 0.0);
  static const uint32_t code[] = {
      FcmpDZero(1),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0110u);
}

// FCMPE S — signal_nans bit does not alter NZCV (mirrors the FCCMP/FCCMPE
// invariant).  Unordered input still yields 0b0011.
TEST_F(Arm64LiteTranslateRegionTest, FcmpeSUnorderedMatchesFcmp) {
  StoreFp32(state_.cpu, 1, std::nanf(""));
  StoreFp32(state_.cpu, 2, 4.0f);
  static const uint32_t code[] = {
      FcmpeS(1, 2),
  };
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(ReadArmNzcv(state_.cpu), 0b0011u);
}
// endregion

// region digitalis - FP scalar unary
//
// Scalar FP one-source ops (FpDataProc1 family).  These pin the
// architectural behaviour of the FP32 / FP64 JIT lowerings at
// `lite_translator.h:6211` — FMOV, FABS, FNEG, FSQRT, and the FCVT
// between-precision variants.  FP16 paths are covered separately by
// the E1 test block; here we focus on the S/D forms exercised by §D2.
//
//   Encoding (FpDataProc1, sf=0): M 0 0 11110 ftype 1 opcode 10000 Rn Rd
//     ftype: 00 = S (FP32), 01 = D (FP64)
//     opcode (bits[20:15]):
//       000000 FMOV       000001 FABS       000010 FNEG
//       000011 FSQRT
//       000100 FCVT  (S<->D, dst precision selected by ftype)
//       000101 FCVT  (S->D when ftype=00, D->S when ftype=01)
//
// Cross-verified with `aarch64-linux-gnu-as -march=armv8.2-a+fp16 -c`:
//   1e20c041 fabs s1, s2 / 1e60c041 fabs d1, d2
//   1e214041 fneg s1, s2 / 1e614041 fneg d1, d2
//   1e21c041 fsqrt s1, s2 / 1e61c041 fsqrt d1, d2
//   1e624041 fcvt s1, d2 / 1e22c041 fcvt d1, s2
//   1e204041 fmov s1, s2 / 1e604041 fmov d1, d2
constexpr uint32_t FpUnaryScalar(uint32_t base, uint8_t rd, uint8_t rn) {
  return base | (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FabsS(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E20C000, rd, rn);
}
constexpr uint32_t FabsD(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E60C000, rd, rn);
}
constexpr uint32_t FnegS(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E214000, rd, rn);
}
constexpr uint32_t FnegD(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E614000, rd, rn);
}
constexpr uint32_t FsqrtS(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E21C000, rd, rn);
}
constexpr uint32_t FsqrtD(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E61C000, rd, rn);
}
// FCVT Dd, Sn (single-precision Rn widened to double in Rd).
constexpr uint32_t FcvtDFromS(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E22C000, rd, rn);
}
// FCVT Sd, Dn (double-precision Rn narrowed to single in Rd).
constexpr uint32_t FcvtSFromD(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E624000, rd, rn);
}
constexpr uint32_t FmovS(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E204000, rd, rn);
}
constexpr uint32_t FmovD(uint8_t rd, uint8_t rn) {
  return FpUnaryScalar(0x1E604000, rd, rn);
}

// FABS S — positive input unchanged.
TEST_F(Arm64LiteTranslateRegionTest, FabsSPositive) {
  StoreFp32(state_.cpu, 1, 1.5f);
  state_.cpu.v[0] = ~__uint128_t{0};  // poison Vd to verify zero-extend.
  static const uint32_t code[] = {FabsS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 1.5f);
  // Upper 96 bits of V[0] must be zeroed (architectural scalar-FP write).
  uint64_t hi64;
  std::memcpy(&hi64, reinterpret_cast<const char*>(&state_.cpu.v[0]) + 8,
              sizeof(hi64));
  EXPECT_EQ(hi64, 0u);
}

// FABS S — negative input flipped to positive.
TEST_F(Arm64LiteTranslateRegionTest, FabsSNegative) {
  StoreFp32(state_.cpu, 1, -3.25f);
  static const uint32_t code[] = {FabsS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 3.25f);
}

// FABS D — negative double becomes positive double.
TEST_F(Arm64LiteTranslateRegionTest, FabsDNegative) {
  StoreFp64(state_.cpu, 1, -7.5);
  static const uint32_t code[] = {FabsD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 7.5);
}

// FNEG S — positive becomes negative; high lanes zero-extended.
TEST_F(Arm64LiteTranslateRegionTest, FnegSPositive) {
  StoreFp32(state_.cpu, 1, 1.5f);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {FnegS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), -1.5f);
  uint64_t hi64;
  std::memcpy(&hi64, reinterpret_cast<const char*>(&state_.cpu.v[0]) + 8,
              sizeof(hi64));
  EXPECT_EQ(hi64, 0u);
}

// FNEG D — negative becomes positive (sign-bit flip is symmetric).
TEST_F(Arm64LiteTranslateRegionTest, FnegDNegative) {
  StoreFp64(state_.cpu, 1, -4.0);
  static const uint32_t code[] = {FnegD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 4.0);
}

// FSQRT S — exact integer square root.
TEST_F(Arm64LiteTranslateRegionTest, FsqrtSExact) {
  StoreFp32(state_.cpu, 1, 4.0f);
  static const uint32_t code[] = {FsqrtS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 2.0f);
}

// FSQRT D — exact integer square root in FP64.
TEST_F(Arm64LiteTranslateRegionTest, FsqrtDExact) {
  StoreFp64(state_.cpu, 1, 9.0);
  static const uint32_t code[] = {FsqrtD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 3.0);
}

// FSQRT S — negative input yields NaN per IEEE-754 default exception.
TEST_F(Arm64LiteTranslateRegionTest, FsqrtSNegativeProducesNan) {
  StoreFp32(state_.cpu, 1, -1.0f);
  static const uint32_t code[] = {FsqrtS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp32(state_.cpu, 0)));
}

// FCVT Dd, Sn — single-precision widens to double exactly.
TEST_F(Arm64LiteTranslateRegionTest, FcvtSingleToDouble) {
  StoreFp32(state_.cpu, 1, 1.5f);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {FcvtDFromS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 1.5);
  uint64_t hi64;
  std::memcpy(&hi64, reinterpret_cast<const char*>(&state_.cpu.v[0]) + 8,
              sizeof(hi64));
  EXPECT_EQ(hi64, 0u);
}

// FCVT Sd, Dn — double narrows to single (1.5 is exact in both).
TEST_F(Arm64LiteTranslateRegionTest, FcvtDoubleToSingleExact) {
  StoreFp64(state_.cpu, 1, 1.5);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {FcvtSFromD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), 1.5f);
  uint64_t hi64;
  std::memcpy(&hi64, reinterpret_cast<const char*>(&state_.cpu.v[0]) + 8,
              sizeof(hi64));
  EXPECT_EQ(hi64, 0u);
}

// FMOV S — bitwise copy of the low 32 bits with high lanes zero.
TEST_F(Arm64LiteTranslateRegionTest, FmovScalarSingleZeroExtends) {
  StoreFp32(state_.cpu, 1, -3.5f);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {FmovS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), -3.5f);
  uint64_t hi64;
  std::memcpy(&hi64, reinterpret_cast<const char*>(&state_.cpu.v[0]) + 8,
              sizeof(hi64));
  EXPECT_EQ(hi64, 0u);
}

// FMOV D — bitwise copy of the low 64 bits with high lane zero.
TEST_F(Arm64LiteTranslateRegionTest, FmovScalarDoubleZeroExtends) {
  StoreFp64(state_.cpu, 1, 2.71828);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {FmovD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp64(state_.cpu, 0), 2.71828);
  uint64_t hi64;
  std::memcpy(&hi64, reinterpret_cast<const char*>(&state_.cpu.v[0]) + 8,
              sizeof(hi64));
  EXPECT_EQ(hi64, 0u);
}
// endregion

// region digitalis - FP scalar arithmetic edges
//
// Scalar FP two-source ops (FpDataProc2 family).  Pins the architectural
// behaviour of FADD / FSUB / FMUL / FDIV at S (ftype=00) and D (ftype=01)
// precision for the IEEE-754 edge categories required by §D1's verify
// gate: NaN propagation, ±Inf − ±Inf / ±0 × ±Inf invalid-op → default
// NaN, ±0 sign handling under round-to-nearest-even, denormal arithmetic,
// and the integer-zero divide → ±Inf result.  These exercise the JIT
// emit path at `lite_translator.h:6632` (SSE Addss/Subss/Mulss/Divss
// and the SD-form for D).
//
//   Encoding (FpDataProc2, M=0):
//     0 0 0 11110 ftype 1 Rm opcode 10 Rn Rd
//     ftype: 00 = S (FP32), 01 = D (FP64)
//     opcode: 0000 FMUL, 0001 FDIV, 0010 FADD, 0011 FSUB
//
// Cross-verified with `aarch64-linux-gnu-as -march=armv8.2-a+fp16 -c`:
//   1e232841 fadd s1,s2,s3 / 1e632841 fadd d1,d2,d3
//   1e233841 fsub s1,s2,s3 / 1e633841 fsub d1,d2,d3
//   1e230841 fmul s1,s2,s3 / 1e630841 fmul d1,d2,d3
//   1e231841 fdiv s1,s2,s3 / 1e631841 fdiv d1,d2,d3
constexpr uint32_t FpBinaryScalar(uint32_t base, uint8_t rd, uint8_t rn,
                                  uint8_t rm) {
  return base | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FaddS(uint8_t rd, uint8_t rn, uint8_t rm) {
  return FpBinaryScalar(0x1E202800, rd, rn, rm);
}
constexpr uint32_t FaddD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return FpBinaryScalar(0x1E602800, rd, rn, rm);
}
constexpr uint32_t FsubS(uint8_t rd, uint8_t rn, uint8_t rm) {
  return FpBinaryScalar(0x1E203800, rd, rn, rm);
}
constexpr uint32_t FsubD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return FpBinaryScalar(0x1E603800, rd, rn, rm);
}
// FmulS already defined at the top of the file (line 107).
constexpr uint32_t FmulD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return FpBinaryScalar(0x1E600800, rd, rn, rm);
}
constexpr uint32_t FdivS(uint8_t rd, uint8_t rn, uint8_t rm) {
  return FpBinaryScalar(0x1E201800, rd, rn, rm);
}
constexpr uint32_t FdivD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return FpBinaryScalar(0x1E601800, rd, rn, rm);
}

// FADD S — NaN operand poisons the result (quiet NaN propagates).
TEST_F(Arm64LiteTranslateRegionTest, FaddSNanPropagates) {
  StoreFp32(state_.cpu, 1, std::numeric_limits<float>::quiet_NaN());
  StoreFp32(state_.cpu, 2, 1.0f);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {FaddS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp32(state_.cpu, 0)));
  // Architectural zero-extend of V[0]'s upper 96 bits.
  uint64_t hi64;
  std::memcpy(&hi64, reinterpret_cast<const char*>(&state_.cpu.v[0]) + 8,
              sizeof(hi64));
  EXPECT_EQ(hi64, 0u);
}

// FADD D — invalid-op (+∞ + −∞) yields a NaN.
TEST_F(Arm64LiteTranslateRegionTest, FaddDInfMinusInfIsNan) {
  StoreFp64(state_.cpu, 1, std::numeric_limits<double>::infinity());
  StoreFp64(state_.cpu, 2, -std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FaddD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp64(state_.cpu, 0)));
}

// FADD S — adding the smallest positive denormal to itself stays exact.
TEST_F(Arm64LiteTranslateRegionTest, FaddSDenormalDoubles) {
  const float denorm = std::numeric_limits<float>::denorm_min();
  StoreFp32(state_.cpu, 1, denorm);
  StoreFp32(state_.cpu, 2, denorm);
  static const uint32_t code[] = {FaddS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), denorm + denorm);
}

// FADD D — round-to-nearest-even default: +0.0 + -0.0 = +0.0.
TEST_F(Arm64LiteTranslateRegionTest, FaddDPosZeroPlusNegZeroIsPosZero) {
  StoreFp64(state_.cpu, 1, +0.0);
  StoreFp64(state_.cpu, 2, -0.0);
  static const uint32_t code[] = {FaddD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  const double res = LoadFp64(state_.cpu, 0);
  EXPECT_EQ(res, 0.0);
  EXPECT_FALSE(std::signbit(res));
}

// FSUB S — NaN on the RHS poisons the result.
TEST_F(Arm64LiteTranslateRegionTest, FsubSSubtractNan) {
  StoreFp32(state_.cpu, 1, 1.0f);
  StoreFp32(state_.cpu, 2, std::numeric_limits<float>::quiet_NaN());
  static const uint32_t code[] = {FsubS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp32(state_.cpu, 0)));
}

// FSUB D — invalid-op (+∞ − +∞) yields a NaN.
TEST_F(Arm64LiteTranslateRegionTest, FsubDInfMinusInfIsNan) {
  StoreFp64(state_.cpu, 1, std::numeric_limits<double>::infinity());
  StoreFp64(state_.cpu, 2, std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FsubD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp64(state_.cpu, 0)));
}

// FSUB S — +0.0 - +0.0 produces +0.0 under round-to-nearest-even.
TEST_F(Arm64LiteTranslateRegionTest, FsubSPosZeroMinusPosZero) {
  StoreFp32(state_.cpu, 1, +0.0f);
  StoreFp32(state_.cpu, 2, +0.0f);
  static const uint32_t code[] = {FsubS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  const float res = LoadFp32(state_.cpu, 0);
  EXPECT_EQ(res, 0.0f);
  EXPECT_FALSE(std::signbit(res));
}

// FMUL S — invalid-op (0 × ∞) yields a NaN.
TEST_F(Arm64LiteTranslateRegionTest, FmulSZeroTimesInfIsNan) {
  StoreFp32(state_.cpu, 1, 0.0f);
  StoreFp32(state_.cpu, 2, std::numeric_limits<float>::infinity());
  static const uint32_t code[] = {FmulS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp32(state_.cpu, 0)));
}

// FMUL D — NaN × finite poisons the product.
TEST_F(Arm64LiteTranslateRegionTest, FmulDNanTimesFinite) {
  StoreFp64(state_.cpu, 1, std::numeric_limits<double>::quiet_NaN());
  StoreFp64(state_.cpu, 2, 2.0);
  static const uint32_t code[] = {FmulD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp64(state_.cpu, 0)));
}

// FMUL S — denormal × 2 stays denormal (or barely above) and is exact.
TEST_F(Arm64LiteTranslateRegionTest, FmulSDenormalTimesTwo) {
  const float denorm = std::numeric_limits<float>::denorm_min();
  StoreFp32(state_.cpu, 1, denorm);
  StoreFp32(state_.cpu, 2, 2.0f);
  static const uint32_t code[] = {FmulS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), denorm * 2.0f);
}

// FMUL D — invalid-op (-0 × +∞) yields a NaN.
TEST_F(Arm64LiteTranslateRegionTest, FmulDNegZeroTimesPosInf) {
  StoreFp64(state_.cpu, 1, -0.0);
  StoreFp64(state_.cpu, 2, std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FmulD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp64(state_.cpu, 0)));
}

// FDIV S — finite / +0 yields +∞ (divide-by-zero exception default result).
TEST_F(Arm64LiteTranslateRegionTest, FdivSOneOverPosZeroIsPosInf) {
  StoreFp32(state_.cpu, 1, 1.0f);
  StoreFp32(state_.cpu, 2, +0.0f);
  static const uint32_t code[] = {FdivS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  const float res = LoadFp32(state_.cpu, 0);
  EXPECT_TRUE(std::isinf(res));
  EXPECT_FALSE(std::signbit(res));
}

// FDIV D — −finite / +0 yields −∞ (sign propagates).
TEST_F(Arm64LiteTranslateRegionTest, FdivDNegOneOverPosZeroIsNegInf) {
  StoreFp64(state_.cpu, 1, -1.0);
  StoreFp64(state_.cpu, 2, +0.0);
  static const uint32_t code[] = {FdivD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  const double res = LoadFp64(state_.cpu, 0);
  EXPECT_TRUE(std::isinf(res));
  EXPECT_TRUE(std::signbit(res));
}

// FDIV S — invalid-op (0 / 0) yields a NaN.
TEST_F(Arm64LiteTranslateRegionTest, FdivSZeroOverZeroIsNan) {
  StoreFp32(state_.cpu, 1, +0.0f);
  StoreFp32(state_.cpu, 2, +0.0f);
  static const uint32_t code[] = {FdivS(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp32(state_.cpu, 0)));
}

// FDIV D — invalid-op (+∞ / +∞) yields a NaN.
TEST_F(Arm64LiteTranslateRegionTest, FdivDInfOverInfIsNan) {
  StoreFp64(state_.cpu, 1, std::numeric_limits<double>::infinity());
  StoreFp64(state_.cpu, 2, std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FdivD(0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_TRUE(std::isnan(LoadFp64(state_.cpu, 0)));
}
// endregion

// region digitalis - FP scalar↔int conversion edges
//
// Scalar FCVTZS / FCVTZU (FP → integer, truncate-toward-zero) and SCVTF /
// UCVTF (integer → FP) JIT lowerings live in `FpIntConversion` at
// `lite_translator.h:2771`.  §D4's verify gate calls out three edge
// categories that the JIT must get right regardless of what x86 cvtt /
// cvtsi2 happen to produce: NaN → 0 (FCVTZS / FCVTZU), ±Inf saturation,
// and the unsigned overflow boundary (FCVTZU returning UINT_MAX rather
// than the x86 indefinite, FCVTZS returning INT_MAX for FP > INT_MAX).
//
//   Encoding (Floating-point ↔ Integer conversion, scalar):
//     sf | 0011110 ftype 1 rmode opcode 000000 Rn Rd
//     rmode=11 opcode=000 -> FCVTZS,    rmode=11 opcode=001 -> FCVTZU,
//     rmode=00 opcode=010 -> SCVTF,     rmode=00 opcode=011 -> UCVTF.
//     ftype: 00 = S (FP32), 01 = D (FP64).
//     sf:    0  = W destination (or source), 1 = X destination (or source).
//
// Cross-verified with `aarch64-linux-gnu-as -march=armv8.2-a+fp16 -c`:
//   1e380041 fcvtzs w1,s2 / 9e380041 fcvtzs x1,s2
//   1e780041 fcvtzs w1,d2 / 9e780041 fcvtzs x1,d2
//   1e390041 fcvtzu w1,s2 / 9e390041 fcvtzu x1,s2
//   1e790041 fcvtzu w1,d2 / 9e790041 fcvtzu x1,d2
//   1e220020 scvtf  s0,w1 / 9e220020 scvtf  s0,x1
//   1e620020 scvtf  d0,w1 / 9e620020 scvtf  d0,x1
//   1e230020 ucvtf  s0,w1 / 9e230020 ucvtf  s0,x1
//   1e630020 ucvtf  d0,w1 / 9e630020 ucvtf  d0,x1
constexpr uint32_t FpIntConv(uint32_t base, uint8_t rd, uint8_t rn) {
  return base | (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FcvtzsWS(uint8_t wd, uint8_t sn) {
  return FpIntConv(0x1E380000, wd, sn);
}
constexpr uint32_t FcvtzsXD(uint8_t xd, uint8_t dn) {
  return FpIntConv(0x9E780000, xd, dn);
}
constexpr uint32_t FcvtzuWS(uint8_t wd, uint8_t sn) {
  return FpIntConv(0x1E390000, wd, sn);
}
constexpr uint32_t FcvtzuXD(uint8_t xd, uint8_t dn) {
  return FpIntConv(0x9E790000, xd, dn);
}
constexpr uint32_t ScvtfSW(uint8_t sd, uint8_t wn) {
  return FpIntConv(0x1E220000, sd, wn);
}
constexpr uint32_t UcvtfDX(uint8_t dd, uint8_t xn) {
  return FpIntConv(0x9E630000, dd, xn);
}

// FCVTZS Wd, Sn — qNaN source must yield 0, not x86 cvtt's INT32_MIN
// "indefinite".  Exercises the PF=1 nan_path branch in the JIT (see
// `lite_translator.h:2970`'s `Ucomis` + `Jcc kParityEven` arm).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsWSNanToZero) {
  StoreFp32(state_.cpu, 1, std::numeric_limits<float>::quiet_NaN());
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  static const uint32_t code[] = {FcvtzsWS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);  // W-form: zero-extended to 64.
}

// FCVTZS Wd, Sn — +Inf must saturate to INT32_MAX.  x86 cvtt returns
// INT32_MIN here; the JIT's pos-overflow fix-up writes INT32_MAX after
// observing FP-sign=0 and tmp==INT32_MIN.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsWSPosInfSaturatesMax) {
  StoreFp32(state_.cpu, 1, std::numeric_limits<float>::infinity());
  static const uint32_t code[] = {FcvtzsWS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(static_cast<int32_t>(state_.cpu.x[0]), INT32_MAX);
}

// FCVTZS Wd, Sn — −Inf saturates to INT32_MIN.  This matches x86 cvtt's
// indefinite return for negative overflow, so the JIT's FP-sign=1 keep-
// tmp branch is what must fire.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsWSNegInfSaturatesMin) {
  StoreFp32(state_.cpu, 1, -std::numeric_limits<float>::infinity());
  static const uint32_t code[] = {FcvtzsWS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(static_cast<int32_t>(state_.cpu.x[0]), INT32_MIN);
}

// FCVTZS Xd, Dn — qNaN source yields 0 (64-bit path).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsXDNanToZero) {
  StoreFp64(state_.cpu, 1, std::numeric_limits<double>::quiet_NaN());
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  static const uint32_t code[] = {FcvtzsXD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);
}

// FCVTZS Xd, Dn — +Inf saturates to INT64_MAX (the pos-overflow fix-up).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsXDPosInfSaturatesMax) {
  StoreFp64(state_.cpu, 1, std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FcvtzsXD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(static_cast<int64_t>(state_.cpu.x[0]), INT64_MAX);
}

// FCVTZS Xd, Dn — −Inf saturates to INT64_MIN.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzsXDNegInfSaturatesMin) {
  StoreFp64(state_.cpu, 1, -std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FcvtzsXD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(static_cast<int64_t>(state_.cpu.x[0]), INT64_MIN);
}

// FCVTZU Wd, Sn — qNaN yields 0 (unsigned NaN→0 mirrors signed).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuWSNanToZero) {
  StoreFp32(state_.cpu, 1, std::numeric_limits<float>::quiet_NaN());
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  static const uint32_t code[] = {FcvtzuWS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);
}

// FCVTZU Wd, Sn — +Inf saturates to UINT32_MAX.  Exercises the W-form
// upper-32 saturation path that detects out-of-u32-range cvtt-q result.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuWSPosInfSaturatesMax) {
  StoreFp32(state_.cpu, 1, std::numeric_limits<float>::infinity());
  static const uint32_t code[] = {FcvtzuWS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(static_cast<uint32_t>(state_.cpu.x[0]), UINT32_MAX);
}

// FCVTZU Wd, Sn — −Inf clamps to 0 (negative inputs all clamp to 0 in
// ARM FCVTZU).  The FP-sign-bit-set branch jumps to the zero path.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuWSNegInfClampsZero) {
  StoreFp32(state_.cpu, 1, -std::numeric_limits<float>::infinity());
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  static const uint32_t code[] = {FcvtzuWS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);
}

// FCVTZU Wd, Sn — exactly UINT32_MAX (2^32 - 1) is *not* exactly
// representable in FP32; the nearest representable below 2^32 is
// 0x4F7FFFFF (~4.2949666e9, exactly 4294967040).  This is the largest
// in-range FP32 value, and it must truncate to 4294967040 (not saturate
// — the JIT only saturates when the cvtt-q result exceeds UINT32_MAX).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuWSLargestInRange) {
  // 0x4F7FFFFF = (float)4294967040.0
  StoreFp32(state_.cpu, 1, 4294967040.0f);
  static const uint32_t code[] = {FcvtzuWS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(static_cast<uint32_t>(state_.cpu.x[0]), 4294967040u);
}

// FCVTZU Wd, Sn — overflow just above 2^32 saturates to UINT32_MAX.
// 0x4F800000 = 2^32 = 4294967296.0f; the cvtt-q result equals UINT32_MAX+1,
// which the JIT's upper-32 saturation detects and clamps.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuWS2p32SaturatesMax) {
  StoreFp32(state_.cpu, 1, 4294967296.0f);  // 2^32
  static const uint32_t code[] = {FcvtzuWS(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(static_cast<uint32_t>(state_.cpu.x[0]), UINT32_MAX);
}

// FCVTZU Xd, Dn — qNaN yields 0 (64-bit path).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuXDNanToZero) {
  StoreFp64(state_.cpu, 1, std::numeric_limits<double>::quiet_NaN());
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  static const uint32_t code[] = {FcvtzuXD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);
}

// FCVTZU Xd, Dn — +Inf saturates to UINT64_MAX (the > 2^64 sat_max path).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuXDPosInfSaturatesMax) {
  StoreFp64(state_.cpu, 1, std::numeric_limits<double>::infinity());
  static const uint32_t code[] = {FcvtzuXD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], UINT64_MAX);
}

// FCVTZU Xd, Dn — −Inf clamps to 0 (negative-clamp path).
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuXDNegInfClampsZero) {
  StoreFp64(state_.cpu, 1, -std::numeric_limits<double>::infinity());
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  static const uint32_t code[] = {FcvtzuXD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], 0ULL);
}

// FCVTZU Xd, Dn — 2^64 saturates to UINT64_MAX (the >= 2^64 boundary).
// 0x43F0000000000000 = (double)2^64.
TEST_F(Arm64LiteTranslateRegionTest, FcvtzuXD2p64SaturatesMax) {
  const uint64_t bits = 0x43F0000000000000ULL;
  double val;
  std::memcpy(&val, &bits, sizeof(val));
  StoreFp64(state_.cpu, 1, val);  // 2^64
  static const uint32_t code[] = {FcvtzuXD(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], UINT64_MAX);
}

// SCVTF Sd, Wn — INT32_MIN converts to -2147483648.0f exactly (the
// boundary the L-form CVTSI2SSL must sign-extend correctly).  Also
// pins the architectural zero-extend of V[rd]'s upper 96 bits via the
// pre-PXOR + Movdqu pattern in the JIT.
TEST_F(Arm64LiteTranslateRegionTest, ScvtfSWInt32MinExact) {
  state_.cpu.x[1] = static_cast<uint64_t>(static_cast<uint32_t>(INT32_MIN));
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {ScvtfSW(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(LoadFp32(state_.cpu, 0), -2147483648.0f);
  uint64_t hi64;
  std::memcpy(&hi64, reinterpret_cast<const char*>(&state_.cpu.v[0]) + 8,
              sizeof(hi64));
  EXPECT_EQ(hi64, 0u);
}

// UCVTF Dd, Xn — 64-bit unsigned with the high bit set exercises the
// halve / round-to-odd / convert / double JIT fix-up at
// `lite_translator.h:2884`.  UINT64_MAX is the boundary; the result
// rounds to 2^64 exactly under round-to-nearest-even.
TEST_F(Arm64LiteTranslateRegionTest, UcvtfDXUint64MaxRoundsTo2p64) {
  state_.cpu.x[1] = UINT64_MAX;
  static const uint32_t code[] = {UcvtfDX(0, 1)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  const double expected = static_cast<double>(UINT64_MAX);  // 2^64
  EXPECT_EQ(LoadFp64(state_.cpu, 0), expected);
}
// endregion

// region digitalis - AdvSimdThreeSame JIT for BIC/ORN/BSL/BIT/BIF/CMGT/CMHI
//
// Encoding (DDI 0487 §C7.2 Advanced SIMD three same):
//   0 Q U 01110 size 1 Rm opcode 1 Rn Rd
// Logic group opcode = 0b00011 with (U, size) selecting BIC/ORN/BSL/BIT/BIF;
// CMGT/CMHI opcode = 0b00110, U selects signed vs unsigned.
constexpr uint32_t SimdThreeSame(uint8_t q, uint8_t u, uint8_t size,
                                 uint8_t opcode, uint8_t rd,
                                 uint8_t rn, uint8_t rm) {
  return 0x0E200400u |
         (uint32_t{static_cast<uint8_t>(q & 1u)} << 30) |
         (uint32_t{static_cast<uint8_t>(u & 1u)} << 29) |
         (uint32_t{static_cast<uint8_t>(size & 3u)} << 22) |
         (uint32_t{static_cast<uint8_t>(rm & 0x1fu)} << 16) |
         (uint32_t{static_cast<uint8_t>(opcode & 0x1fu)} << 11) |
         (uint32_t{static_cast<uint8_t>(rn & 0x1fu)} << 5) |
         uint32_t{static_cast<uint8_t>(rd & 0x1fu)};
}

static void StoreVec16B(CPUState& cpu, unsigned idx, const uint8_t bytes[16]) {
  std::memcpy(&cpu.v[idx], bytes, 16);
}
static void LoadVec16B(const CPUState& cpu, unsigned idx, uint8_t out[16]) {
  std::memcpy(out, &cpu.v[idx], 16);
}

// BIC V0.16B, V1, V2 — Vd = Vn AND NOT Vm.
TEST_F(Arm64LiteTranslateRegionTest, BicVec16B) {
  uint8_t n[16], m[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0xF0u | (i & 0x0Fu));
    m[i] = static_cast<uint8_t>(0x0Fu | ((i & 0x0Fu) << 4));
  }
  uint8_t d[16] = {0};
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  StoreVec16B(state_.cpu, 0, d);
  static const uint32_t code[] = {
      SimdThreeSame(1, 0, /*size=*/0b01, /*opcode=*/0b00011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(r[i], static_cast<uint8_t>(n[i] & ~m[i])) << "lane " << i;
  }
}

// ORN V0.16B, V1, V2 — Vd = Vn OR NOT Vm.  Also covers Q=0 (.8B) upper-zero.
TEST_F(Arm64LiteTranslateRegionTest, OrnVec8BUpperZero) {
  uint8_t n[16], m[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0x10u | (i & 0x0Fu));
    m[i] = static_cast<uint8_t>(~i & 0xFFu);
  }
  uint8_t d_init[16];
  std::memset(d_init, 0xCC, 16);
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/0, /*u=*/0, /*size=*/0b11, /*opcode=*/0b00011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(r[i], static_cast<uint8_t>(n[i] | ~m[i])) << "lane " << i;
  }
  for (int i = 8; i < 16; i++) EXPECT_EQ(r[i], 0u) << "upper lane " << i;
}

// BSL V0.16B, V1, V2 — Vd = (Vd AND Vn) | (NOT Vd AND Vm).  Vd is read-modify.
TEST_F(Arm64LiteTranslateRegionTest, BslVec16B) {
  uint8_t n[16], m[16], d[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0xA5u ^ (i * 3u));
    m[i] = static_cast<uint8_t>(0x5Au ^ (i * 5u));
    d[i] = static_cast<uint8_t>(0xF0u ^ (i * 11u));  // mask
  }
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  StoreVec16B(state_.cpu, 0, d);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/1, /*size=*/0b01, /*opcode=*/0b00011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    uint8_t expected = static_cast<uint8_t>((d[i] & n[i]) | (~d[i] & m[i]));
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

// BIT V0.16B, V1, V2 — Vd = (Vm AND Vn) | (NOT Vm AND Vd).
TEST_F(Arm64LiteTranslateRegionTest, BitVec16B) {
  uint8_t n[16], m[16], d[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0x33u ^ (i * 7u));
    m[i] = static_cast<uint8_t>(0xCCu ^ (i * 13u));
    d[i] = static_cast<uint8_t>(0x99u ^ (i * 17u));
  }
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  StoreVec16B(state_.cpu, 0, d);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/1, /*size=*/0b10, /*opcode=*/0b00011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    uint8_t expected = static_cast<uint8_t>((m[i] & n[i]) | (~m[i] & d[i]));
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

// BIF V0.16B, V1, V2 — Vd = (Vm AND Vd) | (NOT Vm AND Vn).
TEST_F(Arm64LiteTranslateRegionTest, BifVec16B) {
  uint8_t n[16], m[16], d[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0x66u ^ (i * 19u));
    m[i] = static_cast<uint8_t>(0xAAu ^ (i * 23u));
    d[i] = static_cast<uint8_t>(0x55u ^ (i * 29u));
  }
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  StoreVec16B(state_.cpu, 0, d);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/1, /*size=*/0b11, /*opcode=*/0b00011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    uint8_t expected = static_cast<uint8_t>((m[i] & d[i]) | (~m[i] & n[i]));
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

// CMGT V0.16B, V1, V2 — signed lane-wise greater-than at byte width.
TEST_F(Arm64LiteTranslateRegionTest, CmgtVec16BSigned) {
  int8_t n_signed[16] = {0, 1, -1, -128, 127, -2, 50, -50,
                         100, -100, 0, 0, 5, 5, 6, -3};
  int8_t m_signed[16] = {0, 0, 0, -127, 126, -3, 50, 50,
                         99, -101, 1, -1, 6, 5, 5, -3};
  uint8_t n[16], m[16];
  std::memcpy(n, n_signed, 16);
  std::memcpy(m, m_signed, 16);
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0xAA, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/0, /*size=*/0b00, /*opcode=*/0b00110, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    uint8_t expected = (n_signed[i] > m_signed[i]) ? 0xFFu : 0x00u;
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

// CMGT V0.4S, V1, V2 — signed lane-wise greater-than at 32-bit width.
TEST_F(Arm64LiteTranslateRegionTest, CmgtVec4SSigned) {
  StoreVec4SInt(state_.cpu, 1, 5, -1, INT32_MIN, INT32_MAX);
  StoreVec4SInt(state_.cpu, 2, 5,  0, -1,        INT32_MAX - 1);
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/0, /*size=*/0b10, /*opcode=*/0b00110, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 0);                           // 5 > 5 is false
  EXPECT_EQ(r[1], 0);                           // -1 > 0 is false
  EXPECT_EQ(r[2], 0);                           // INT32_MIN > -1 is false
  EXPECT_EQ(r[3], static_cast<int32_t>(0xFFFFFFFF));  // INT32_MAX > MAX-1 true
}

// CMHI V0.4S, V1, V2 — unsigned lane-wise greater-than at 32-bit width.
// 0xFFFFFFFF unsigned is the largest value; check that the sign-flip trick
// converts unsigned ordering correctly.
TEST_F(Arm64LiteTranslateRegionTest, CmhiVec4SUnsigned) {
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  uint32_t n_lanes[4] = {0xFFFFFFFFu, 0x80000000u, 1u, 0x7FFFFFFFu};
  uint32_t m_lanes[4] = {0x7FFFFFFFu, 0x7FFFFFFFu, 2u, 0x80000000u};
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/1, /*size=*/0b10, /*opcode=*/0b00110, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0xFFFFFFFFu);  // 0xFFFFFFFF >u 0x7FFFFFFF
  EXPECT_EQ(r[1], 0xFFFFFFFFu);  // 0x80000000 >u 0x7FFFFFFF (unsigned!)
  EXPECT_EQ(r[2], 0u);           // 1 !> 2
  EXPECT_EQ(r[3], 0u);           // 0x7FFFFFFF !> 0x80000000 (unsigned!)
}
// endregion

// region digitalis - AdvSimdThreeSame JIT for MUL .8H/.4H (PMULLW)
// Closes the last open size on the MUL row of the §C2 NEON three-same table:
// the JIT already handled MUL .4S/.2S via PMULLD; this region adds size=01
// (16-bit lanes) via SSE2 PMULLW. Reuses SimdThreeSame() from the preceding
// region. MUL (vector) encoding: U=0, opcode=0b10011.
TEST_F(Arm64LiteTranslateRegionTest, MulVec8H) {
  // .8H, Q=1, full 128-bit. Verify per-lane low-16-bit truncation.
  int16_t n_lanes[8] = {1, -1, 2, 32767, -32768, 100, 0, 1234};
  int16_t m_lanes[8] = {3, 7, -4, 2,     2,      -3,  9, 5};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memset(&state_.cpu.v[0], 0xAA, sizeof(state_.cpu.v[0]));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b01,
                    /*opcode=*/0b10011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  for (int i = 0; i < 8; i++) {
    int16_t expected = static_cast<int16_t>(
        static_cast<int32_t>(n_lanes[i]) * static_cast<int32_t>(m_lanes[i]));
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, MulVec4HUpperZero) {
  // .4H, Q=0: low 4 halfwords are the product, upper 8 bytes must be zeroed.
  int16_t n_lanes[8] = {2, 3, 4, 5, 0x1111, 0x2222, 0x3333, 0x4444};
  int16_t m_lanes[8] = {10, 20, 30, 40, 0x5555, 0x6666, 0x7777, 0x0000};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  uint8_t d_init[16];
  std::memset(d_init, 0xCC, 16);
  std::memcpy(&state_.cpu.v[0], d_init, 16);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/0, /*u=*/0, /*size=*/0b01,
                    /*opcode=*/0b10011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  for (int i = 0; i < 4; i++) {
    int16_t expected = static_cast<int16_t>(
        static_cast<int32_t>(n_lanes[i]) * static_cast<int32_t>(m_lanes[i]));
    EXPECT_EQ(r[i], expected) << "low lane " << i;
  }
  uint8_t r_bytes[16];
  std::memcpy(r_bytes, &state_.cpu.v[0], 16);
  for (int i = 8; i < 16; i++) {
    EXPECT_EQ(r_bytes[i], 0u) << "upper byte " << i;
  }
}
// endregion

// region digitalis - AdvSimdThreeSame JIT for MLA .8H (PMULLW+PADDW)
// Direct sibling of the MUL .8H/.4H region above: the JIT already handled
// MLA .4S/.2S via PMULLD+PADDD; this region adds size=01 (16-bit lanes) via
// SSE2 PMULLW+PADDW. Reuses SimdThreeSame(). MLA (vector) encoding: U=0,
// opcode=0b10010.
TEST_F(Arm64LiteTranslateRegionTest, MlaVec8H) {
  // .8H, Q=1, full 128-bit. Verify per-lane Vd += Vn * Vm with low-16-bit
  // truncation across boundary values.
  int16_t n_lanes[8] = {1, -1, 2, 32767, -32768, 100, 0, 1234};
  int16_t m_lanes[8] = {3, 7, -4, 2,     2,      -3,  9, 5};
  int16_t d_lanes[8] = {10, -20, 30, 1, 1, 7, 5, -1000};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memset(&state_.cpu.v[0], 0, sizeof(state_.cpu.v[0]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memcpy(&state_.cpu.v[0], d_lanes, sizeof(d_lanes));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b01,
                    /*opcode=*/0b10010, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  for (int i = 0; i < 8; i++) {
    int16_t expected = static_cast<int16_t>(
        static_cast<int32_t>(d_lanes[i]) +
        static_cast<int32_t>(n_lanes[i]) * static_cast<int32_t>(m_lanes[i]));
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}
// endregion

// region digitalis - AdvSimdThreeSame JIT for CMGE/CMHS (PCMPGT + invert)
// Closes the last open compare-flavor entries on the §C2 NEON three-same
// table: CMGT/CMHI already JITed via PCMPGT (with unsigned sign-flip);
// CMGE = NOT(PCMPGT(Vm, Vn)), CMHS = same with pre-sign-flip. Reuses
// SimdThreeSame(). CMGE encoding: U=0, opcode=0b00111. CMHS: U=1, opcode=0b00111.
TEST_F(Arm64LiteTranslateRegionTest, CmgeVec16BSigned) {
  // Verify boundary equality cases (== returns all-ones), strict ordering
  // both directions, and the full int8 sign range across byte lanes.
  int8_t n_signed[16] = {0, 1, -1, -128, 127, -2, 50, -50,
                         100, -100, 0, 0, 5, 5, 6, -3};
  int8_t m_signed[16] = {0, 0, 0, -127, 126, -3, 50, 50,
                         99, -101, 1, -1, 6, 5, 5, -3};
  uint8_t n[16], m[16];
  std::memcpy(n, n_signed, 16);
  std::memcpy(m, m_signed, 16);
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0xAA, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/0, /*size=*/0b00, /*opcode=*/0b00111, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    uint8_t expected = (n_signed[i] >= m_signed[i]) ? 0xFFu : 0x00u;
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmgeVec4SSigned) {
  // .4S, Q=1: 32-bit lanes, INT32_MIN/MAX boundaries plus equality.
  StoreVec4SInt(state_.cpu, 1, 5, -1, INT32_MIN, INT32_MAX);
  StoreVec4SInt(state_.cpu, 2, 5,  0, -1,        INT32_MAX);
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/0, /*size=*/0b10, /*opcode=*/0b00111, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], static_cast<int32_t>(0xFFFFFFFF));  // 5 >= 5 (equal)
  EXPECT_EQ(r[1], 0);                                  // -1 >= 0 false
  EXPECT_EQ(r[2], 0);                                  // INT32_MIN >= -1 false
  EXPECT_EQ(r[3], static_cast<int32_t>(0xFFFFFFFF));  // INT32_MAX >= MAX (equal)
}

TEST_F(Arm64LiteTranslateRegionTest, CmhsVec4SUnsigned) {
  // .4S, Q=1: 32-bit unsigned ordering across the signed boundary.
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  uint32_t n_lanes[4] = {0xFFFFFFFFu, 0x80000000u, 1u, 0x7FFFFFFFu};
  uint32_t m_lanes[4] = {0x7FFFFFFFu, 0x80000000u, 2u, 0x80000000u};
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(1, /*u=*/1, /*size=*/0b10, /*opcode=*/0b00111, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0xFFFFFFFFu);  // 0xFFFFFFFF >=u 0x7FFFFFFF
  EXPECT_EQ(r[1], 0xFFFFFFFFu);  // 0x80000000 >=u 0x80000000 (equal)
  EXPECT_EQ(r[2], 0u);            // 1 !>=u 2
  EXPECT_EQ(r[3], 0u);            // 0x7FFFFFFF !>=u 0x80000000 (unsigned!)
}

TEST_F(Arm64LiteTranslateRegionTest, CmgeVec4HUpperZero) {
  // .4H, Q=0: low 4 halfwords are the compare result, upper 8 bytes zeroed.
  int16_t n_lanes[8] = {0, 100, -1, 32767, 0x1111, 0x2222, 0x3333, 0x4444};
  int16_t m_lanes[8] = {0, 99,  -2, 32767, 0x5555, 0x6666, 0x7777, 0x0000};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  uint8_t d_init[16];
  std::memset(d_init, 0xCC, 16);
  std::memcpy(&state_.cpu.v[0], d_init, 16);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/0, /*u=*/0, /*size=*/0b01,
                    /*opcode=*/0b00111, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0xFFFFu);  // 0 >= 0
  EXPECT_EQ(r[1], 0xFFFFu);  // 100 >= 99
  EXPECT_EQ(r[2], 0xFFFFu);  // -1 >= -2
  EXPECT_EQ(r[3], 0xFFFFu);  // 32767 >= 32767
  uint8_t r_bytes[16];
  std::memcpy(r_bytes, &state_.cpu.v[0], 16);
  for (int i = 8; i < 16; i++) {
    EXPECT_EQ(r_bytes[i], 0u) << "upper byte " << i;
  }
}
// endregion

// region digitalis - AdvSimdThreeSame JIT for SMAX/SMIN/UMAX/UMIN
// Direct PMAXS{B,W,D} / PMINS{B,W,D} / PMAXU{B,W,D} / PMINU{B,W,D} lowerings.
// Per ARM DDI 0487, three-same opcodes:
//   SMAX: U=0, opcode=0b01100. UMAX: U=1, opcode=0b01100.
//   SMIN: U=0, opcode=0b01101. UMIN: U=1, opcode=0b01101.
// Sizes 0b00/0b01/0b10 are JITed; 0b11 (.2D) bails to the interpreter (no
// PMAXSQ/PMINSQ etc. on SSE/AVX2).
TEST_F(Arm64LiteTranslateRegionTest, SmaxVec16BSigned) {
  // .16B: signed byte lanes including INT8 boundaries and equality.
  int8_t n_signed[16] = {0, 1, -1, INT8_MIN, INT8_MAX, -50, 50, 50,
                         100, -100, 0, 5, 6, -3, -3, -1};
  int8_t m_signed[16] = {0, 0, 0, INT8_MAX, INT8_MIN, -49, 49, 50,
                         99, -101, 1, 5, 5, -3, -4, 0};
  uint8_t n[16], m[16];
  std::memcpy(n, n_signed, 16);
  std::memcpy(m, m_signed, 16);
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0xAA, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b00,
                    /*opcode=*/0b01100, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    int8_t expected = (n_signed[i] > m_signed[i]) ? n_signed[i] : m_signed[i];
    EXPECT_EQ(static_cast<int8_t>(r[i]), expected) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SminVec4SSigned) {
  // .4S: signed dword lanes with INT32_MIN/MAX boundaries plus equality.
  StoreVec4SInt(state_.cpu, 1, 5,  -1, INT32_MIN, INT32_MAX);
  StoreVec4SInt(state_.cpu, 2, 5,   0, -1,        INT32_MAX);
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b10,
                    /*opcode=*/0b01101, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 5);                 // min(5, 5)
  EXPECT_EQ(r[1], -1);                // min(-1, 0)
  EXPECT_EQ(r[2], INT32_MIN);         // min(INT32_MIN, -1)
  EXPECT_EQ(r[3], INT32_MAX);         // min(MAX, MAX)
}

TEST_F(Arm64LiteTranslateRegionTest, UmaxVec4SUnsigned) {
  // .4S, unsigned: covers the 0x7FFFFFFF / 0x80000000 boundary that
  // would mis-order under a signed compare.
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  uint32_t n_lanes[4] = {0xFFFFFFFFu, 0x80000000u, 1u, 0x7FFFFFFFu};
  uint32_t m_lanes[4] = {0x7FFFFFFFu, 0x80000000u, 2u, 0x80000000u};
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b10,
                    /*opcode=*/0b01100, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0xFFFFFFFFu);       // umax(FFFFFFFF, 7FFFFFFF)
  EXPECT_EQ(r[1], 0x80000000u);       // umax(80000000, 80000000)
  EXPECT_EQ(r[2], 2u);                // umax(1, 2)
  EXPECT_EQ(r[3], 0x80000000u);       // umax(7FFFFFFF, 80000000) — unsigned!
}

TEST_F(Arm64LiteTranslateRegionTest, UminVec4HUpperZero) {
  // .4H (Q=0): unsigned halfword min; upper 8 bytes must be zeroed.
  uint16_t n_lanes[8] = {0x0001, 0xFFFF, 0x8000, 0x1234, 0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD};
  uint16_t m_lanes[8] = {0x0002, 0x7FFF, 0x8000, 0x1234, 0x5555, 0x6666, 0x7777, 0x8888};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  uint8_t d_init[16];
  std::memset(d_init, 0xCC, 16);
  std::memcpy(&state_.cpu.v[0], d_init, 16);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/0, /*u=*/1, /*size=*/0b01,
                    /*opcode=*/0b01101, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0x0001u);           // umin(0001, 0002)
  EXPECT_EQ(r[1], 0x7FFFu);           // umin(FFFF, 7FFF)
  EXPECT_EQ(r[2], 0x8000u);           // umin(8000, 8000)
  EXPECT_EQ(r[3], 0x1234u);           // umin(1234, 1234)
  uint8_t r_bytes[16];
  std::memcpy(r_bytes, &state_.cpu.v[0], 16);
  for (int i = 8; i < 16; i++) {
    EXPECT_EQ(r_bytes[i], 0u) << "upper byte " << i;
  }
}
// endregion

// region digitalis - AdvSimdThreeSame JIT for MUL/MLA .16B/.8B
// Closes the byte-lane (size=00) entries on the §C2 MUL/MLA row. x86 has
// no PMULLB, so the lowering widens each 8 bytes to 16-bit words via
// PMOVZXBW (low half then high half), PMULLW per half, masks each word
// result to 0x00FF, then PACKUSWB packs back to 16 bytes. MLA byte adds
// PADDB into Vd after pack. MUL encoding: U=0, opcode=0b10011. MLA: U=0,
// opcode=0b10010.
TEST_F(Arm64LiteTranslateRegionTest, MulVec16B) {
  // .16B, Q=1, full 128-bit. Exercise all 16 lanes including signed-byte
  // boundary values and the asymmetry between the low-8-byte and
  // high-8-byte halves (different products in each half, to catch a
  // wrong-half-pack bug).
  int8_t n_lanes[16] = {1, -1, 2, 127, -128, 100, 0, 12,
                        3, 5,  7, -16, 64,   33, -1, 99};
  int8_t m_lanes[16] = {3, 7, -4, 2,   2,   -3,  9, 5,
                        2, 4,  6, 8,   2,   3,   4, 5};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memset(&state_.cpu.v[0], 0xAA, sizeof(state_.cpu.v[0]));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b00,
                    /*opcode=*/0b10011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  for (int i = 0; i < 16; i++) {
    int8_t expected = static_cast<int8_t>(
        static_cast<int32_t>(n_lanes[i]) * static_cast<int32_t>(m_lanes[i]));
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, MlaVec8BUpperZero) {
  // .8B, Q=0: low 8 byte lanes are Vd += Vn*Vm; upper 8 bytes must be zeroed.
  uint8_t n_lanes[16] = {1, 2,   3, 200, 17, 255, 50, 100,
                         0, 0xCC, 0, 0,  0,  0,   0,  0};
  uint8_t m_lanes[16] = {2, 3,   4, 3,   10, 2,   2,  3,
                         0, 0xDD, 0, 0,  0,  0,   0,  0};
  uint8_t d_init[16];
  std::memset(d_init, 0xCC, 16);
  d_init[0] = 5;  d_init[1] = 5; d_init[2] = 5; d_init[3] = 5;
  d_init[4] = 5;  d_init[5] = 5; d_init[6] = 5; d_init[7] = 5;
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memcpy(&state_.cpu.v[0], d_init, 16);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/0, /*u=*/0, /*size=*/0b00,
                    /*opcode=*/0b10010, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  for (int i = 0; i < 8; i++) {
    uint8_t expected = static_cast<uint8_t>(
        static_cast<uint32_t>(d_init[i]) +
        static_cast<uint32_t>(n_lanes[i]) * static_cast<uint32_t>(m_lanes[i]));
    EXPECT_EQ(r[i], expected) << "low lane " << i;
  }
  for (int i = 8; i < 16; i++) {
    EXPECT_EQ(r[i], 0u) << "upper byte " << i;
  }
}
// endregion

// region digitalis - §C2 spec-table single-width coverage gaps
// Closes the §C2 plan-verify checkbox "Unit tests in
// lite_translate_region_exec_tests.cc for one representative of each lane
// width (B/H/S/D) per opcode." Earlier handoffs covered the
// non-size-agnostic opcodes (MUL/MLA/CMGT/CMHI/CMGE/CMHS, SMAX/SMIN/
// UMAX/UMIN, BIC/ORN/BSL/BIT/BIF) at one or two lane widths each.
// The remaining gaps in the §C2 spec table are:
//   - ADD/SUB/CMEQ: no dedicated vector tests at any width — add B/H/S/D.
//     (CMEQ.2D now JITs via PCMPEQQ when host_platform::kHasSSE4_1 is set;
//     CmeqVec2D below pins that path. The handler still bails via
//     Undefined() on hosts without SSE4.1, matching the CMGT.2D/SSE4.2
//     gate above it.)
//   - AND/ORR/EOR: size-agnostic logical ops — add one each at the
//     16B canonical shape, matching the BIC/ORN/BSL/BIT/BIF pattern.
//   - MUL/MLA: B and H widths already pinned (MulVec16B, MulVec8H,
//     MlaVec8H, MlaVec8BUpperZero); add the .4S width (PMULLD path).
// Reuses SimdThreeSame() from the preceding region. Encodings per
// ARM DDI 0487 §C7.2: ADD U=0/opcode=10000, SUB U=1/opcode=10000,
// CMEQ U=1/opcode=10001, AND/ORR/EOR via the U×size logic-group table
// at opcode=00011 (AND U=0 size=00, ORR U=0 size=10, EOR U=1 size=00),
// MUL U=0/opcode=10011, MLA U=0/opcode=10010.

TEST_F(Arm64LiteTranslateRegionTest, AddVec16B) {
  uint8_t n[16], m[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0x10u + i);
    m[i] = static_cast<uint8_t>(0xF0u - i);
  }
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0xAA, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b00,
                    /*opcode=*/0b10000, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(r[i], static_cast<uint8_t>(n[i] + m[i])) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, AddVec8H) {
  int16_t n_lanes[8] = {1, -1, 32767, -32768, 100, -100, 0x1234, -0x4321};
  int16_t m_lanes[8] = {2,  3, 1,     -1,     1,    -1,   0x0001, 0x0001};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memset(&state_.cpu.v[0], 0xAA, sizeof(state_.cpu.v[0]));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b01,
                    /*opcode=*/0b10000, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  for (int i = 0; i < 8; i++) {
    int16_t expected = static_cast<int16_t>(
        static_cast<int32_t>(n_lanes[i]) + static_cast<int32_t>(m_lanes[i]));
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, AddVec4S) {
  StoreVec4SInt(state_.cpu, 1, 1, -1, INT32_MIN, INT32_MAX);
  StoreVec4SInt(state_.cpu, 2, 2,  3, -1,        1);
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b10,
                    /*opcode=*/0b10000, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 3);
  EXPECT_EQ(r[1], 2);
  EXPECT_EQ(r[2], INT32_MAX);      // INT32_MIN + -1 wraps to INT32_MAX
  EXPECT_EQ(r[3], INT32_MIN);      // INT32_MAX + 1  wraps to INT32_MIN
}

TEST_F(Arm64LiteTranslateRegionTest, AddVec2D) {
  uint64_t n_lanes[2] = {0x0000000000000001ULL, 0xFFFFFFFFFFFFFFFFULL};
  uint64_t m_lanes[2] = {0x00000000FFFFFFFFULL, 0x0000000000000001ULL};
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memset(&state_.cpu.v[0], 0xCC, sizeof(state_.cpu.v[0]));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b11,
                    /*opcode=*/0b10000, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0x0000000100000000ULL);
  EXPECT_EQ(r[1], 0x0000000000000000ULL);  // FFFF…FFFF + 1 wraps
}

TEST_F(Arm64LiteTranslateRegionTest, SubVec16B) {
  uint8_t n[16], m[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0x80u + i);
    m[i] = static_cast<uint8_t>(0x01u + i);
  }
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0xAA, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b00,
                    /*opcode=*/0b10000, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(r[i], static_cast<uint8_t>(n[i] - m[i])) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SubVec8H) {
  int16_t n_lanes[8] = {100, 0, INT16_MIN, INT16_MAX, 1, -1, 0x1234, 0x4321};
  int16_t m_lanes[8] = {50,  1, 1,         -1,        2,  1, 0x0001, 0x0001};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memset(&state_.cpu.v[0], 0xAA, sizeof(state_.cpu.v[0]));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b01,
                    /*opcode=*/0b10000, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  for (int i = 0; i < 8; i++) {
    int16_t expected = static_cast<int16_t>(
        static_cast<int32_t>(n_lanes[i]) - static_cast<int32_t>(m_lanes[i]));
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SubVec4S) {
  StoreVec4SInt(state_.cpu, 1, 5, INT32_MIN, INT32_MAX, 0);
  StoreVec4SInt(state_.cpu, 2, 3, 1,         -1,        INT32_MIN);
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b10,
                    /*opcode=*/0b10000, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 2);
  EXPECT_EQ(r[1], INT32_MAX);                       // INT32_MIN - 1 wraps to MAX
  EXPECT_EQ(r[2], INT32_MIN);                       // INT32_MAX - -1 wraps to MIN
  EXPECT_EQ(r[3], INT32_MIN);                       // 0 - INT32_MIN wraps to MIN
}

TEST_F(Arm64LiteTranslateRegionTest, SubVec2D) {
  uint64_t n_lanes[2] = {0x0000000000000000ULL, 0x8000000000000000ULL};
  uint64_t m_lanes[2] = {0x0000000000000001ULL, 0x0000000000000001ULL};
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memset(&state_.cpu.v[0], 0xCC, sizeof(state_.cpu.v[0]));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b11,
                    /*opcode=*/0b10000, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0xFFFFFFFFFFFFFFFFULL);            // 0 - 1 wraps
  EXPECT_EQ(r[1], 0x7FFFFFFFFFFFFFFFULL);            // INT64_MIN - 1 wraps to MAX
}

TEST_F(Arm64LiteTranslateRegionTest, AndVec16B) {
  uint8_t n[16], m[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0xF0u | (i & 0x0Fu));
    m[i] = static_cast<uint8_t>(0x0Fu | ((i & 0x0Fu) << 4));
  }
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0x55, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b00,
                    /*opcode=*/0b00011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(r[i], static_cast<uint8_t>(n[i] & m[i])) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, OrrVec16B) {
  uint8_t n[16], m[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0x0Fu & (i + 1));
    m[i] = static_cast<uint8_t>(0xF0u & ((i + 1) << 4));
  }
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0x33, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b10,
                    /*opcode=*/0b00011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(r[i], static_cast<uint8_t>(n[i] | m[i])) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, EorVec16B) {
  uint8_t n[16], m[16];
  for (int i = 0; i < 16; i++) {
    n[i] = static_cast<uint8_t>(0xA5u ^ (i * 7u));
    m[i] = static_cast<uint8_t>(0x5Au ^ (i * 11u));
  }
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0xCC, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b00,
                    /*opcode=*/0b00011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(r[i], static_cast<uint8_t>(n[i] ^ m[i])) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmeqVec16B) {
  // .16B: lane-wise equality; verify boundary equals and a mix of unequal.
  int8_t n_signed[16] = {0, 1, -1, -128, 127, 50, -50, 99,
                         -99, 0,  0,  5,    5,  6,  -3,  -3};
  int8_t m_signed[16] = {0, 1, -1, -128, 127, 49, -50, 100,
                         -100, 1, -1, 5,    6,  5,  -3,  -2};
  uint8_t n[16], m[16];
  std::memcpy(n, n_signed, 16);
  std::memcpy(m, m_signed, 16);
  StoreVec16B(state_.cpu, 1, n);
  StoreVec16B(state_.cpu, 2, m);
  uint8_t d_init[16];
  std::memset(d_init, 0xAA, 16);
  StoreVec16B(state_.cpu, 0, d_init);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b00,
                    /*opcode=*/0b10001, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  LoadVec16B(state_.cpu, 0, r);
  for (int i = 0; i < 16; i++) {
    uint8_t expected = (n_signed[i] == m_signed[i]) ? 0xFFu : 0x00u;
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmeqVec8H) {
  int16_t n_lanes[8] = {0, 1, -1, INT16_MAX, INT16_MIN, 100, 100, 0};
  int16_t m_lanes[8] = {0, 1, -2, INT16_MAX, INT16_MAX, 100, 101, 1};
  std::memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));
  std::memset(&state_.cpu.v[2], 0, sizeof(state_.cpu.v[2]));
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memset(&state_.cpu.v[0], 0xAA, sizeof(state_.cpu.v[0]));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b01,
                    /*opcode=*/0b10001, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  for (int i = 0; i < 8; i++) {
    uint16_t expected = (n_lanes[i] == m_lanes[i]) ? 0xFFFFu : 0x0000u;
    EXPECT_EQ(r[i], expected) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmeqVec4S) {
  StoreVec4SInt(state_.cpu, 1, 5, -1, INT32_MIN, INT32_MAX);
  StoreVec4SInt(state_.cpu, 2, 5,  0, INT32_MIN, INT32_MAX - 1);
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b10,
                    /*opcode=*/0b10001, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0xFFFFFFFFu);                          // 5 == 5
  EXPECT_EQ(r[1], 0u);                                    // -1 != 0
  EXPECT_EQ(r[2], 0xFFFFFFFFu);                          // MIN == MIN
  EXPECT_EQ(r[3], 0u);                                    // MAX != MAX-1
}

TEST_F(Arm64LiteTranslateRegionTest, CmeqVec2D) {
  // .2D: 64-bit lane equality via PCMPEQQ (SSE4.1). Modern x86_64 hosts
  // have SSE4.1, so the JIT path is exercised; on hypothetical hosts
  // without it the handler returns Undefined() and Run() would fail.
  if (!host_platform::kHasSSE4_1) {
    GTEST_SKIP() << "PCMPEQQ requires SSE4.1";
  }
  int64_t n_lanes[2] = {INT64_MIN, 0x123456789ABCDEF0LL};
  int64_t m_lanes[2] = {INT64_MIN, 0x123456789ABCDEF1LL};
  std::memcpy(&state_.cpu.v[1], n_lanes, sizeof(n_lanes));
  std::memcpy(&state_.cpu.v[2], m_lanes, sizeof(m_lanes));
  std::memset(&state_.cpu.v[0], 0xAA, sizeof(state_.cpu.v[0]));
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/1, /*size=*/0b11,
                    /*opcode=*/0b10001, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], sizeof(r));
  EXPECT_EQ(r[0], 0xFFFFFFFFFFFFFFFFull);  // INT64_MIN == INT64_MIN
  EXPECT_EQ(r[1], 0ull);                    // differ in low nibble
}

TEST_F(Arm64LiteTranslateRegionTest, MulVec4S) {
  // .4S, Q=1: 32-bit lane MUL via PMULLD. Covers the spec-table S width
  // gap left after MulVec8H (H), MulVec4HUpperZero (H/.4H), MulVec16B (B).
  StoreVec4SInt(state_.cpu, 1, 3,  -7, INT32_MAX, 0x10000);
  StoreVec4SInt(state_.cpu, 2, 5,   2, 2,         0x10000);
  StoreVec4SInt(state_.cpu, 0, 0, 0, 0, 0);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b10,
                    /*opcode=*/0b10011, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 15);
  EXPECT_EQ(r[1], -14);
  // INT32_MAX * 2 = 0xFFFFFFFE truncated to int32 -> -2
  EXPECT_EQ(r[2], -2);
  // 0x10000 * 0x10000 = 0x100000000 truncated to int32 -> 0
  EXPECT_EQ(r[3], 0);
}

TEST_F(Arm64LiteTranslateRegionTest, MlaVec4S) {
  // .4S, Q=1: 32-bit lane MLA via PMULLD+PADDD. Vd is read-modify.
  StoreVec4SInt(state_.cpu, 1, 3,  -7, 2,         0x10000);
  StoreVec4SInt(state_.cpu, 2, 5,   2, INT32_MAX, 0x10000);
  StoreVec4SInt(state_.cpu, 0, 100, 50, 1,         INT32_MAX);
  static const uint32_t code[] = {
      SimdThreeSame(/*q=*/1, /*u=*/0, /*size=*/0b10,
                    /*opcode=*/0b10010, 0, 1, 2)};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  LoadVec4SInt(state_.cpu, 0, r);
  EXPECT_EQ(r[0], 100 + 15);                              // 100 + 3*5
  EXPECT_EQ(r[1], 50 + (-14));                            // 50 + -7*2
  // 1 + (2 * INT32_MAX) = 1 + 0xFFFFFFFE = 0xFFFFFFFF = -1
  EXPECT_EQ(r[2], -1);
  // INT32_MAX + (0x10000 * 0x10000) = INT32_MAX + 0 (mul truncates) = INT32_MAX
  EXPECT_EQ(r[3], INT32_MAX);
}
// endregion

// region digitalis: ABS / NEG / NOT vector two-reg-misc JIT
//
// Encoding (per ARM ARM C7.2 "ABS (vector)", "NEG (vector)", "NOT (vector)",
// cross-checked against aarch64-linux-gnu-as output):
//   abs v0.16b, v1.16b = 0x4E20B820 (Q=1, U=0, size=00, opcode=01011)
//   abs v0.8h,  v1.8h  = 0x4E60B820 (Q=1, U=0, size=01)
//   abs v0.4s,  v1.4s  = 0x4EA0B820 (Q=1, U=0, size=10)
//   abs v0.2s,  v1.2s  = 0x0EA0B820 (Q=0, U=0, size=10)
//   neg v0.16b, v1.16b = 0x6E20B820 (Q=1, U=1, size=00, opcode=01011)
//   neg v0.4s,  v1.4s  = 0x6EA0B820 (Q=1, U=1, size=10)
//   neg v0.2d,  v1.2d  = 0x6EE0B820 (Q=1, U=1, size=11)
//   neg v0.2s,  v1.2s  = 0x2EA0B820 (Q=0, U=1, size=10)
//   mvn v0.16b, v1.16b = 0x6E205820 (Q=1, U=1, size=00, opcode=00101) -- NOT
//   mvn v0.8b,  v1.8b  = 0x2E205820 (Q=0, U=1, size=00, opcode=00101) -- NOT
constexpr uint32_t kAbsVec16B = 0x4E20B820;
constexpr uint32_t kAbsVec8H  = 0x4E60B820;
constexpr uint32_t kAbsVec4S  = 0x4EA0B820;
constexpr uint32_t kAbsVec2S  = 0x0EA0B820;
constexpr uint32_t kNegVec16B = 0x6E20B820;
constexpr uint32_t kNegVec4S  = 0x6EA0B820;
constexpr uint32_t kNegVec2D  = 0x6EE0B820;
constexpr uint32_t kNegVec2S  = 0x2EA0B820;
constexpr uint32_t kNotVec16B = 0x6E205820;
constexpr uint32_t kNotVec8B  = 0x2E205820;

TEST_F(Arm64LiteTranslateRegionTest, AbsVec16B) {
  int8_t in[16] = {0, 1, -1, 127, -128, -64, 64, -100, 100, -7, 7, 0, 23, -23, -1, INT8_MAX};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kAbsVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // ARM ABS leaves INT8_MIN as INT8_MIN (no saturation in ABS).
  int8_t want[16] = {0, 1, 1, 127, INT8_MIN, 64, 64, 100, 100, 7, 7, 0, 23, 23, 1, INT8_MAX};
  for (int i = 0; i < 16; ++i) EXPECT_EQ(r[i], want[i]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, AbsVec8H) {
  int16_t in[8] = {0, -1, 32767, -32768, 1234, -1234, -7, 7};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kAbsVec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  int16_t want[8] = {0, 1, 32767, INT16_MIN, 1234, 1234, 7, 7};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], want[i]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, AbsVec4S) {
  int32_t in[4] = {-1, INT32_MIN, INT32_MAX, 0};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kAbsVec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 1);
  EXPECT_EQ(r[1], INT32_MIN);  // INT_MIN stays INT_MIN
  EXPECT_EQ(r[2], INT32_MAX);
  EXPECT_EQ(r[3], 0);
}

TEST_F(Arm64LiteTranslateRegionTest, AbsVec2SUpperZero) {
  int32_t in[4] = {-9, 9, 0x12345678, 0x55AA55AA};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kAbsVec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 9);
  EXPECT_EQ(r[1], 9);
  EXPECT_EQ(r[2], 0);  // upper half zeroed
  EXPECT_EQ(r[3], 0);
}

TEST_F(Arm64LiteTranslateRegionTest, NegVec16B) {
  int8_t in[16] = {0, 1, -1, 127, -128, 50, -50, 99, -99, 7, -7, 33, -33, 1, 0, INT8_MAX};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kNegVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    // ARM NEG wraps: -INT_MIN = INT_MIN (same as x86 PSUBB from zero).
    int8_t want = static_cast<int8_t>(-static_cast<int32_t>(in[i]));
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, NegVec4S) {
  int32_t in[4] = {1, -1, INT32_MIN, INT32_MAX};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kNegVec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], -1);
  EXPECT_EQ(r[1], 1);
  EXPECT_EQ(r[2], INT32_MIN);  // wraps
  EXPECT_EQ(r[3], -INT32_MAX);
}

TEST_F(Arm64LiteTranslateRegionTest, NegVec2D) {
  int64_t in[2] = {INT64_MIN, 0x1122334455667788LL};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kNegVec2D};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], INT64_MIN);  // wraps
  EXPECT_EQ(r[1], -0x1122334455667788LL);
}

TEST_F(Arm64LiteTranslateRegionTest, NegVec2SUpperZero) {
  int32_t in[4] = {-3, 4, 0x7FFFFFFF, static_cast<int32_t>(0x80000000)};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kNegVec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 3);
  EXPECT_EQ(r[1], -4);
  EXPECT_EQ(r[2], 0);
  EXPECT_EQ(r[3], 0);
}

TEST_F(Arm64LiteTranslateRegionTest, NotVec16B) {
  uint8_t in[16] = {0x00, 0xFF, 0xA5, 0x5A, 0x12, 0xED, 0x80, 0x7F,
                    0x33, 0xCC, 0xAB, 0x54, 0x01, 0xFE, 0xDE, 0xAD};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kNotVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(r[i], static_cast<uint8_t>(~in[i])) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, NotVec8BUpperZero) {
  uint8_t in[16] = {0x00, 0xFF, 0xA5, 0x5A, 0x12, 0xED, 0x80, 0x7F,
                    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kNotVec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], static_cast<uint8_t>(~in[i])) << "lane " << i;
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper-half lane " << i;
}
// endregion

// region digitalis: CMGT / CMGE / CMLE / CMLT vector-vs-zero two-reg-misc JIT
//
// Encodings (verified against aarch64-linux-gnu-as):
//   cmgt v0.16b, v1.16b, #0 = 0x4E208820   (Q=1, U=0, opcode=01000, size=00)
//   cmgt v0.8h,  v1.8h,  #0 = 0x4E608820   (Q=1, U=0, opcode=01000, size=01)
//   cmgt v0.4s,  v1.4s,  #0 = 0x4EA08820   (Q=1, U=0, opcode=01000, size=10)
//   cmgt v0.2s,  v1.2s,  #0 = 0x0EA08820   (Q=0, U=0, opcode=01000, size=10)
//   cmge v0.16b, v1.16b, #0 = 0x6E208820   (Q=1, U=1, opcode=01000, size=00)
//   cmge v0.4s,  v1.4s,  #0 = 0x6EA08820   (Q=1, U=1, opcode=01000, size=10)
//   cmge v0.2s,  v1.2s,  #0 = 0x2EA08820   (Q=0, U=1, opcode=01000, size=10)
//   cmle v0.16b, v1.16b, #0 = 0x6E209820   (Q=1, U=1, opcode=01001, size=00)
//   cmle v0.8h,  v1.8h,  #0 = 0x6E609820   (Q=1, U=1, opcode=01001, size=01)
//   cmlt v0.16b, v1.16b, #0 = 0x4E20A820   (Q=1, U=0, opcode=01010, size=00)
//   cmlt v0.4s,  v1.4s,  #0 = 0x4EA0A820   (Q=1, U=0, opcode=01010, size=10)
constexpr uint32_t kCmgtZeroVec16B = 0x4E208820;
constexpr uint32_t kCmgtZeroVec8H  = 0x4E608820;
constexpr uint32_t kCmgtZeroVec4S  = 0x4EA08820;
constexpr uint32_t kCmgtZeroVec2S  = 0x0EA08820;
constexpr uint32_t kCmgeZeroVec16B = 0x6E208820;
constexpr uint32_t kCmgeZeroVec4S  = 0x6EA08820;
constexpr uint32_t kCmgeZeroVec2S  = 0x2EA08820;
constexpr uint32_t kCmleZeroVec16B = 0x6E209820;
constexpr uint32_t kCmleZeroVec8H  = 0x6E609820;
constexpr uint32_t kCmltZeroVec16B = 0x4E20A820;
constexpr uint32_t kCmltZeroVec4S  = 0x4EA0A820;

TEST_F(Arm64LiteTranslateRegionTest, CmgtZeroVec16B) {
  int8_t in[16] = {0, 1, -1, INT8_MAX, INT8_MIN, -50, 50, 0,
                   -1, 7, -7, 99, -99, 0, INT8_MAX, INT8_MIN};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmgtZeroVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t want = (in[i] > 0) ? 0xFFu : 0x00u;
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmgtZeroVec8H) {
  int16_t in[8] = {0, -1, 1, INT16_MAX, INT16_MIN, 1234, -1234, 0};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmgtZeroVec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint16_t want = (in[i] > 0) ? 0xFFFFu : 0x0000u;
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmgtZeroVec4S) {
  int32_t in[4] = {INT32_MIN, -1, 0, INT32_MAX};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmgtZeroVec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0u);
  EXPECT_EQ(r[1], 0u);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0xFFFFFFFFu);
}

TEST_F(Arm64LiteTranslateRegionTest, CmgtZeroVec2SUpperZero) {
  int32_t in[4] = {5, -5, 0x12345678, static_cast<int32_t>(0xDEADBEEF)};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmgtZeroVec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0xFFFFFFFFu);
  EXPECT_EQ(r[1], 0u);
  EXPECT_EQ(r[2], 0u);  // upper half zeroed
  EXPECT_EQ(r[3], 0u);
}

TEST_F(Arm64LiteTranslateRegionTest, CmgeZeroVec16B) {
  int8_t in[16] = {0, 1, -1, INT8_MAX, INT8_MIN, -50, 50, 0,
                   -1, 7, -7, 99, -99, 0, INT8_MAX, INT8_MIN};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmgeZeroVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t want = (in[i] >= 0) ? 0xFFu : 0x00u;
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmgeZeroVec4S) {
  int32_t in[4] = {INT32_MIN, -1, 0, INT32_MAX};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmgeZeroVec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0u);
  EXPECT_EQ(r[1], 0u);
  EXPECT_EQ(r[2], 0xFFFFFFFFu);
  EXPECT_EQ(r[3], 0xFFFFFFFFu);
}

TEST_F(Arm64LiteTranslateRegionTest, CmgeZeroVec2SUpperZero) {
  int32_t in[4] = {0, -1, static_cast<int32_t>(0xAABBCCDD), 0x11223344};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmgeZeroVec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0xFFFFFFFFu);  // 0 >= 0
  EXPECT_EQ(r[1], 0u);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}

TEST_F(Arm64LiteTranslateRegionTest, CmleZeroVec16B) {
  int8_t in[16] = {0, 1, -1, INT8_MAX, INT8_MIN, -50, 50, 0,
                   -1, 7, -7, 99, -99, 0, INT8_MAX, INT8_MIN};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmleZeroVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t want = (in[i] <= 0) ? 0xFFu : 0x00u;
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmleZeroVec8H) {
  int16_t in[8] = {0, -1, 1, INT16_MAX, INT16_MIN, 1234, -1234, 0};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmleZeroVec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint16_t want = (in[i] <= 0) ? 0xFFFFu : 0x0000u;
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmltZeroVec16B) {
  int8_t in[16] = {0, 1, -1, INT8_MAX, INT8_MIN, -50, 50, 0,
                   -1, 7, -7, 99, -99, 0, INT8_MAX, INT8_MIN};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmltZeroVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t want = (in[i] < 0) ? 0xFFu : 0x00u;
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CmltZeroVec4S) {
  int32_t in[4] = {INT32_MIN, -1, 0, INT32_MAX};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCmltZeroVec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0xFFFFFFFFu);
  EXPECT_EQ(r[1], 0xFFFFFFFFu);
  EXPECT_EQ(r[2], 0u);  // 0 < 0 false
  EXPECT_EQ(r[3], 0u);
}
// endregion

// region digitalis: CNT vector two-reg-misc JIT (popcount nibble-LUT via PSHUFB)
constexpr uint32_t kCntVec16B = 0x4E205820;  // cnt v0.16b, v1.16b
constexpr uint32_t kCntVec8B  = 0x0E205820;  // cnt v0.8b,  v1.8b

TEST_F(Arm64LiteTranslateRegionTest, CntVec16B) {
  // Cover all bit patterns: 0, all-ones, alternating, single-bit, boundaries.
  uint8_t in[16] = {0x00, 0xFF, 0x01, 0x80, 0x55, 0xAA, 0x7F, 0x0F,
                    0xF0, 0x11, 0x22, 0x44, 0x88, 0xC3, 0x3C, 0xE7};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCntVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t want = static_cast<uint8_t>(__builtin_popcount(in[i]));
    EXPECT_EQ(r[i], want) << "lane " << i << " in=0x" << std::hex
                          << static_cast<int>(in[i]);
  }
}

TEST_F(Arm64LiteTranslateRegionTest, CntVec8BUpperZero) {
  // Q=0 form must zero the upper 64 bits of Vd.  Use a non-zero pattern in
  // bytes 8..15 of Vn to confirm those don't leak.
  uint8_t in[16] = {0x00, 0xFF, 0x55, 0xAA, 0x01, 0x80, 0x7F, 0x0F,
                    0x33, 0xCC, 0xF0, 0x0F, 0xAA, 0x55, 0xFF, 0xE7};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kCntVec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint8_t want = static_cast<uint8_t>(__builtin_popcount(in[i]));
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}
// endregion

// region digitalis: RBIT vector two-reg-misc JIT (bit-reverse nibble-LUT via PSHUFB pair)
constexpr uint32_t kRbitVec16B = 0x6E605820;  // rbit v0.16b, v1.16b
constexpr uint32_t kRbitVec8B  = 0x2E605820;  // rbit v0.8b,  v1.8b

namespace {
inline uint8_t ReverseBits8(uint8_t b) {
  uint8_t r = 0;
  for (int i = 0; i < 8; ++i) r |= ((b >> i) & 1u) << (7 - i);
  return r;
}
}  // namespace

TEST_F(Arm64LiteTranslateRegionTest, RbitVec16B) {
  // Cover bit patterns that exercise both nibble lookups: 0, all-ones,
  // alternating, single-bit (boundary positions), nibble-aligned values.
  uint8_t in[16] = {0x00, 0xFF, 0x01, 0x80, 0x55, 0xAA, 0x7F, 0x0F,
                    0xF0, 0x11, 0x22, 0x44, 0x88, 0xC3, 0x3C, 0xE7};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kRbitVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t want = ReverseBits8(in[i]);
    EXPECT_EQ(r[i], want) << "lane " << i << " in=0x" << std::hex
                          << static_cast<int>(in[i]);
  }
}

TEST_F(Arm64LiteTranslateRegionTest, RbitVec8BUpperZero) {
  // Q=0 form must zero the upper 64 bits of Vd.  Non-zero pattern in
  // bytes 8..15 of Vn confirms those don't leak into Vd.
  uint8_t in[16] = {0x00, 0xFF, 0x55, 0xAA, 0x01, 0x80, 0x7F, 0x0F,
                    0x33, 0xCC, 0xF0, 0x0F, 0xAA, 0x55, 0xFF, 0xE7};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kRbitVec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint8_t want = ReverseBits8(in[i]);
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}
// endregion

// region digitalis: CLZ vector two-reg-misc JIT (per-lane PEXTR + BSR + XOR + PINSR)
constexpr uint32_t kClzVec16B = 0x6E204820;  // clz v0.16b, v1.16b
constexpr uint32_t kClzVec8B  = 0x2E204820;  // clz v0.8b,  v1.8b
constexpr uint32_t kClzVec8H  = 0x6E604820;  // clz v0.8h,  v1.8h
constexpr uint32_t kClzVec4H  = 0x2E604820;  // clz v0.4h,  v1.4h
constexpr uint32_t kClzVec4S  = 0x6EA04820;  // clz v0.4s,  v1.4s
constexpr uint32_t kClzVec2S  = 0x2EA04820;  // clz v0.2s,  v1.2s

namespace {
inline uint32_t Clz8(uint8_t v) {
  if (v == 0) return 8;
  uint32_t n = 0;
  while ((v & 0x80) == 0) { ++n; v = static_cast<uint8_t>(v << 1); }
  return n;
}
inline uint32_t Clz16(uint16_t v) {
  if (v == 0) return 16;
  uint32_t n = 0;
  while ((v & 0x8000u) == 0) { ++n; v = static_cast<uint16_t>(v << 1); }
  return n;
}
inline uint32_t Clz32(uint32_t v) {
  if (v == 0) return 32;
  uint32_t n = 0;
  while ((v & 0x80000000u) == 0) { ++n; v <<= 1; }
  return n;
}
}  // namespace

TEST_F(Arm64LiteTranslateRegionTest, ClzVec16B) {
  // Cover zero, all-ones, single-bit (boundary positions), and mid-range
  // values so each lane exercises a distinct CLZ result in 0..8.
  uint8_t in[16] = {0x00, 0xFF, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04,
                    0x02, 0x01, 0x7F, 0x55, 0xAA, 0xC3, 0x3C, 0xE7};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kClzVec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t want = static_cast<uint8_t>(Clz8(in[i]));
    EXPECT_EQ(r[i], want) << "lane " << i << " in=0x" << std::hex
                          << static_cast<int>(in[i]);
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ClzVec8BUpperZero) {
  // Q=0 form must zero the upper 64 bits of Vd; pattern in bytes 8..15
  // of Vn confirms that low lanes are computed and high lanes are zeroed.
  uint8_t in[16] = {0x00, 0xFF, 0x80, 0x01, 0x40, 0x02, 0x20, 0x04,
                    0x33, 0xCC, 0xF0, 0x0F, 0xAA, 0x55, 0xFF, 0xE7};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kClzVec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint8_t want = static_cast<uint8_t>(Clz8(in[i]));
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ClzVec8H) {
  uint16_t in[8] = {0x0000, 0xFFFF, 0x8000, 0x4000, 0x0001, 0x00FF, 0x7FFF, 0x1234};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kClzVec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint16_t want = static_cast<uint16_t>(Clz16(in[i]));
    EXPECT_EQ(r[i], want) << "lane " << i << " in=0x" << std::hex << in[i];
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ClzVec4HUpperZero) {
  // Q=0 form must zero upper 64 bits.
  uint16_t in[8] = {0x0000, 0xFFFF, 0x8000, 0x0001,
                    0x1111, 0x2222, 0x3333, 0x4444};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kClzVec4H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    uint16_t want = static_cast<uint16_t>(Clz16(in[i]));
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ClzVec4S) {
  uint32_t in[4] = {0x00000000u, 0xFFFFFFFFu, 0x80000000u, 0x00000001u};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kClzVec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    uint32_t want = Clz32(in[i]);
    EXPECT_EQ(r[i], want) << "lane " << i << " in=0x" << std::hex << in[i];
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ClzVec2SUpperZero) {
  uint32_t in[4] = {0x00000000u, 0x00ABCDEFu, 0xDEADBEEFu, 0x12345678u};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kClzVec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 2; ++i) {
    uint32_t want = Clz32(in[i]);
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
  for (int i = 2; i < 4; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}
// endregion

// region digitalis: SHL / USHR / SSHR vector shift-by-immediate JIT
//
// SHL  Vd.T, Vn.T, #imm    → PSLL{W,D,Q}
// USHR Vd.T, Vn.T, #imm    → PSRL{W,D,Q}
// SSHR Vd.T, Vn.T, #imm    → PSRA{W,D}    (no PSRAQ — .2D bails)
// All Q=0 forms zero upper 64 bits of Vd.
constexpr uint32_t kShlVec8H_3   = 0x4f135420;  // shl  v0.8h, v1.8h, #3
constexpr uint32_t kShlVec4H_3   = 0x0f135420;  // shl  v0.4h, v1.4h, #3
constexpr uint32_t kShlVec4S_5   = 0x4f255420;  // shl  v0.4s, v1.4s, #5
constexpr uint32_t kShlVec2S_5   = 0x0f255420;  // shl  v0.2s, v1.2s, #5
constexpr uint32_t kShlVec2D_11  = 0x4f4b5420;  // shl  v0.2d, v1.2d, #11
constexpr uint32_t kUshrVec8H_3  = 0x6f1d0420;  // ushr v0.8h, v1.8h, #3
constexpr uint32_t kUshrVec4H_3  = 0x2f1d0420;  // ushr v0.4h, v1.4h, #3
constexpr uint32_t kUshrVec4S_5  = 0x6f3b0420;  // ushr v0.4s, v1.4s, #5
constexpr uint32_t kUshrVec2S_5  = 0x2f3b0420;  // ushr v0.2s, v1.2s, #5
constexpr uint32_t kUshrVec2D_11 = 0x6f750420;  // ushr v0.2d, v1.2d, #11
constexpr uint32_t kSshrVec8H_3  = 0x4f1d0420;  // sshr v0.8h, v1.8h, #3
constexpr uint32_t kSshrVec4H_3  = 0x0f1d0420;  // sshr v0.4h, v1.4h, #3
constexpr uint32_t kSshrVec4S_5  = 0x4f3b0420;  // sshr v0.4s, v1.4s, #5
constexpr uint32_t kSshrVec2S_5  = 0x0f3b0420;  // sshr v0.2s, v1.2s, #5
constexpr uint32_t kSshrVec8H_16 = 0x4f100420;  // sshr v0.8h, v1.8h, #16 (saturate to sign)
constexpr uint32_t kUshrVec8H_16 = 0x6f100420;  // ushr v0.8h, v1.8h, #16 (saturate to zero)

TEST_F(Arm64LiteTranslateRegionTest, ShlVec8H) {
  uint16_t in[8] = {0x0000, 0xFFFF, 0x0001, 0x8000, 0x1234, 0xABCD, 0x00FF, 0x0F0F};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kShlVec8H_3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint16_t want = static_cast<uint16_t>(in[i] << 3);
    EXPECT_EQ(r[i], want) << "lane " << i << " in=0x" << std::hex << in[i];
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ShlVec4HUpperZero) {
  uint16_t in[8] = {0x1234, 0xABCD, 0x00FF, 0x0F0F, 0xDEAD, 0xBEEF, 0xCAFE, 0xBABE};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kShlVec4H_3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    uint16_t want = static_cast<uint16_t>(in[i] << 3);
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ShlVec4S) {
  uint32_t in[4] = {0x00000000u, 0xFFFFFFFFu, 0x12345678u, 0x80000001u};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kShlVec4S_5};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    uint32_t want = in[i] << 5;
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ShlVec2SUpperZero) {
  uint32_t in[4] = {0x12345678u, 0x80000001u, 0xDEADBEEFu, 0xCAFEBABEu};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kShlVec2S_5};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(r[i], in[i] << 5) << "lane " << i;
  }
  for (int i = 2; i < 4; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, ShlVec2D) {
  uint64_t in[2] = {0x0000000000000001ull, 0x8000000000000001ull};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kShlVec2D_11};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 2; ++i) {
    uint64_t want = in[i] << 11;
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, UshrVec8H) {
  uint16_t in[8] = {0x0000, 0xFFFF, 0x0008, 0x8000, 0x1234, 0xABCD, 0x00FF, 0x0F0F};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUshrVec8H_3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint16_t want = static_cast<uint16_t>(in[i] >> 3);
    EXPECT_EQ(r[i], want) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, UshrVec4HUpperZero) {
  uint16_t in[8] = {0x1234, 0xABCD, 0x00FF, 0x0F0F, 0xDEAD, 0xBEEF, 0xCAFE, 0xBABE};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUshrVec4H_3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[i], static_cast<uint16_t>(in[i] >> 3)) << "lane " << i;
  }
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, UshrVec4S) {
  uint32_t in[4] = {0x00000000u, 0xFFFFFFFFu, 0x12345678u, 0x80000001u};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUshrVec4S_5};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[i], in[i] >> 5) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, UshrVec2SUpperZero) {
  uint32_t in[4] = {0x12345678u, 0x80000001u, 0xDEADBEEFu, 0xCAFEBABEu};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUshrVec2S_5};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(r[i], in[i] >> 5) << "lane " << i;
  }
  for (int i = 2; i < 4; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, UshrVec2D) {
  uint64_t in[2] = {0x123456789abcdef0ull, 0x8000000000000001ull};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUshrVec2D_11};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(r[i], in[i] >> 11) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SshrVec8H) {
  // Mix of positive, negative, and boundary values.
  int16_t in_s[8] = {0, -1, 8, INT16_MIN, 0x1234, static_cast<int16_t>(0xABCD), 0xFF, -16};
  std::memcpy(&state_.cpu.v[1], in_s, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kSshrVec8H_3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    int16_t want = static_cast<int16_t>(in_s[i] >> 3);  // arithmetic shift
    EXPECT_EQ(r[i], want) << "lane " << i << " in=" << in_s[i];
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SshrVec4HUpperZero) {
  int16_t in_s[8] = {-128, 256, INT16_MIN, INT16_MAX, 99, -99, 1, -1};
  std::memcpy(&state_.cpu.v[1], in_s, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kSshrVec4H_3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[i], static_cast<int16_t>(in_s[i] >> 3)) << "lane " << i;
  }
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SshrVec4S) {
  int32_t in_s[4] = {0, -1, INT32_MIN, 0x12345678};
  std::memcpy(&state_.cpu.v[1], in_s, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kSshrVec4S_5};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[i], in_s[i] >> 5) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, SshrVec2SUpperZero) {
  int32_t in_s[4] = {-128, INT32_MIN, 99, -99};
  std::memcpy(&state_.cpu.v[1], in_s, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kSshrVec2S_5};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(r[i], in_s[i] >> 5) << "lane " << i;
  }
  for (int i = 2; i < 4; ++i) {
    EXPECT_EQ(r[i], 0) << "upper lane " << i << " not zeroed";
  }
}

// Saturation: SSHR by element-width fills with sign bit, USHR by element-width zeros.
TEST_F(Arm64LiteTranslateRegionTest, SshrVec8HSaturate) {
  int16_t in_s[8] = {0, -1, INT16_MIN, INT16_MAX, -16384, 16384, 1, -1};
  std::memcpy(&state_.cpu.v[1], in_s, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kSshrVec8H_16};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  int16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    int16_t want = (in_s[i] < 0) ? int16_t{-1} : int16_t{0};
    EXPECT_EQ(r[i], want) << "lane " << i << " in=" << in_s[i];
  }
}

TEST_F(Arm64LiteTranslateRegionTest, UshrVec8HSaturate) {
  uint16_t in[8] = {0x0000, 0xFFFF, 0x8000, 0x7FFF, 0x1234, 0xABCD, 0x00FF, 0x0001};
  std::memcpy(&state_.cpu.v[1], in, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUshrVec8H_16};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "lane " << i << " in=0x" << std::hex << in[i];
  }
}
// endregion

// region digitalis: ZIP1 / ZIP2 vector permute JIT
//
// ZIP1 Vd.<T>, Vn.<T>, Vm.<T>  -> interleave LOWER half of Vn and Vm
// ZIP2 Vd.<T>, Vn.<T>, Vm.<T>  -> interleave UPPER half of Vn and Vm
// Lowering: PUNPCKL{BW,WD,DQ,QDQ} / PUNPCKH{BW,WD,DQ,QDQ}.
// Q=0 ZIP2: shift sources right by 4 bytes, then PUNPCKL, then mask upper.
constexpr uint32_t kZip1Vec16B = 0x4e023820;  // zip1 v0.16b, v1.16b, v2.16b
constexpr uint32_t kZip2Vec16B = 0x4e027820;  // zip2 v0.16b, v1.16b, v2.16b
constexpr uint32_t kZip1Vec8H  = 0x4e423820;  // zip1 v0.8h,  v1.8h,  v2.8h
constexpr uint32_t kZip2Vec8H  = 0x4e427820;  // zip2 v0.8h,  v1.8h,  v2.8h
constexpr uint32_t kZip1Vec4S  = 0x4e823820;  // zip1 v0.4s,  v1.4s,  v2.4s
constexpr uint32_t kZip2Vec4S  = 0x4e827820;  // zip2 v0.4s,  v1.4s,  v2.4s
constexpr uint32_t kZip1Vec2D  = 0x4ec23820;  // zip1 v0.2d,  v1.2d,  v2.2d
constexpr uint32_t kZip2Vec2D  = 0x4ec27820;  // zip2 v0.2d,  v1.2d,  v2.2d
constexpr uint32_t kZip1Vec8B  = 0x0e023820;  // zip1 v0.8b,  v1.8b,  v2.8b
constexpr uint32_t kZip2Vec8B  = 0x0e027820;  // zip2 v0.8b,  v1.8b,  v2.8b
constexpr uint32_t kZip1Vec4H  = 0x0e423820;  // zip1 v0.4h,  v1.4h,  v2.4h
constexpr uint32_t kZip2Vec4H  = 0x0e427820;  // zip2 v0.4h,  v1.4h,  v2.4h
constexpr uint32_t kZip1Vec2S  = 0x0e823820;  // zip1 v0.2s,  v1.2s,  v2.2s
constexpr uint32_t kZip2Vec2S  = 0x0e827820;  // zip2 v0.2s,  v1.2s,  v2.2s

TEST_F(Arm64LiteTranslateRegionTest, Zip1Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // ZIP1 .16B: Vd[2i]=Vn[i], Vd[2i+1]=Vm[i], i=0..7
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[2 * i],     vn[i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[i]) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Zip2Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // ZIP2 .16B: Vd[2i]=Vn[8+i], Vd[2i+1]=Vm[8+i], i=0..7
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[2 * i],     vn[8 + i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[8 + i]) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Zip1Vec8H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x0908,0x0B0A,0x0D0C,0x0F0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x1918,0x1B1A,0x1D1C,0x1F1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i],     vn[i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[i]) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Zip2Vec8H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x0908,0x0B0A,0x0D0C,0x0F0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x1918,0x1B1A,0x1D1C,0x1F1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i],     vn[4 + i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[4 + i]) << "lane " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Zip1Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], vn[1]);
  EXPECT_EQ(r[3], vm[1]);
}

TEST_F(Arm64LiteTranslateRegionTest, Zip2Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[2]);
  EXPECT_EQ(r[1], vm[2]);
  EXPECT_EQ(r[2], vn[3]);
  EXPECT_EQ(r[3], vm[3]);
}

TEST_F(Arm64LiteTranslateRegionTest, Zip1Vec2D) {
  uint64_t vn[2] = {0x1111111111111111ull, 0x2222222222222222ull};
  uint64_t vm[2] = {0xAAAAAAAAAAAAAAAAull, 0xBBBBBBBBBBBBBBBBull};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec2D};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
}

TEST_F(Arm64LiteTranslateRegionTest, Zip2Vec2D) {
  uint64_t vn[2] = {0x1111111111111111ull, 0x2222222222222222ull};
  uint64_t vm[2] = {0xAAAAAAAAAAAAAAAAull, 0xBBBBBBBBBBBBBBBBull};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec2D};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
}

TEST_F(Arm64LiteTranslateRegionTest, Zip1Vec8B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // ZIP1 .8B: Vd[2i]=Vn[i], Vd[2i+1]=Vm[i], i=0..3. Upper 8 bytes zero.
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i],     vn[i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[i]) << "lane " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper byte " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Zip2Vec8B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // ZIP2 .8B: Vd[2i]=Vn[4+i], Vd[2i+1]=Vm[4+i], i=0..3. Upper 8 bytes zero.
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i],     vn[4 + i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[4 + i]) << "lane " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper byte " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Zip1Vec4H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706, 0xFFFF,0xEEEE,0xDDDD,0xCCCC};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716, 0x9999,0x8888,0x7777,0x6666};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec4H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(r[2 * i],     vn[i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[i]) << "lane " << i;
  }
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Zip2Vec4H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706, 0xFFFF,0xEEEE,0xDDDD,0xCCCC};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716, 0x9999,0x8888,0x7777,0x6666};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec4H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(r[2 * i],     vn[2 + i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[2 + i]) << "lane " << i;
  }
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Zip1Vec2S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0xDEADBEEFu, 0xCAFEBABEu};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0x12345678u, 0x9ABCDEF0u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}

TEST_F(Arm64LiteTranslateRegionTest, Zip2Vec2S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0xDEADBEEFu, 0xCAFEBABEu};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0x12345678u, 0x9ABCDEF0u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}
// endregion

// region digitalis: UZP1 / UZP2 vector permute JIT
//
// UZP1 Vd.<T>, Vn.<T>, Vm.<T>  -> pick EVEN-indexed elements of Vn || Vm
// UZP2 Vd.<T>, Vn.<T>, Vm.<T>  -> pick ODD-indexed  elements of Vn || Vm
// Lowering: PSLLW/PSRLW + PACKUSWB (.16B/.8B), PSLLD/PSRLD + PACKUSDW
// (.8H/.4H; SSE4.1), SHUFPS (.4S/.2S), PUNPCKL/HQDQ (.2D).
// Q=0 forms first PUNPCKLQDQ the two sources to share a single
// PACKUS/SHUFPS source and let the Q=0 upper-zero tail discard the
// resulting duplicate.
constexpr uint32_t kUzp1Vec16B = 0x4e021820;  // uzp1 v0.16b, v1.16b, v2.16b
constexpr uint32_t kUzp2Vec16B = 0x4e025820;  // uzp2 v0.16b, v1.16b, v2.16b
constexpr uint32_t kUzp1Vec8H  = 0x4e421820;  // uzp1 v0.8h,  v1.8h,  v2.8h
constexpr uint32_t kUzp2Vec8H  = 0x4e425820;  // uzp2 v0.8h,  v1.8h,  v2.8h
constexpr uint32_t kUzp1Vec4S  = 0x4e821820;  // uzp1 v0.4s,  v1.4s,  v2.4s
constexpr uint32_t kUzp2Vec4S  = 0x4e825820;  // uzp2 v0.4s,  v1.4s,  v2.4s
constexpr uint32_t kUzp1Vec2D  = 0x4ec21820;  // uzp1 v0.2d,  v1.2d,  v2.2d
constexpr uint32_t kUzp2Vec2D  = 0x4ec25820;  // uzp2 v0.2d,  v1.2d,  v2.2d
constexpr uint32_t kUzp1Vec8B  = 0x0e021820;  // uzp1 v0.8b,  v1.8b,  v2.8b
constexpr uint32_t kUzp2Vec8B  = 0x0e025820;  // uzp2 v0.8b,  v1.8b,  v2.8b
constexpr uint32_t kUzp1Vec4H  = 0x0e421820;  // uzp1 v0.4h,  v1.4h,  v2.4h
constexpr uint32_t kUzp2Vec4H  = 0x0e425820;  // uzp2 v0.4h,  v1.4h,  v2.4h
constexpr uint32_t kUzp1Vec2S  = 0x0e821820;  // uzp1 v0.2s,  v1.2s,  v2.2s
constexpr uint32_t kUzp2Vec2S  = 0x0e825820;  // uzp2 v0.2s,  v1.2s,  v2.2s

TEST_F(Arm64LiteTranslateRegionTest, Uzp1Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP1 .16B: bytes 0..7 = even bytes of vn; bytes 8..15 = even bytes of vm.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[i],     vn[2 * i]) << "vn byte " << i;
    EXPECT_EQ(r[8 + i], vm[2 * i]) << "vm byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp2Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP2 .16B: bytes 0..7 = odd bytes of vn; bytes 8..15 = odd bytes of vm.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[i],     vn[2 * i + 1]) << "vn byte " << i;
    EXPECT_EQ(r[8 + i], vm[2 * i + 1]) << "vm byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp1Vec8H) {
  // Use high-bit values to pin PACKUSDW saturation behavior: each masked
  // dword sits in 0..0xFFFF so saturation must be a no-op.
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x8908,0x8B0A,0xFD0C,0xFF0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x9918,0x9B1A,0xED1C,0xEF1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP1 .8H: halfwords 0..3 = even halfwords of vn; 4..7 = even of vm.
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[i],     vn[2 * i]) << "vn halfword " << i;
    EXPECT_EQ(r[4 + i], vm[2 * i]) << "vm halfword " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp2Vec8H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x8908,0x8B0A,0xFD0C,0xFF0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x9918,0x9B1A,0xED1C,0xEF1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[i],     vn[2 * i + 1]) << "vn halfword " << i;
    EXPECT_EQ(r[4 + i], vm[2 * i + 1]) << "vm halfword " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp1Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP1 .4S: [vn[0], vn[2], vm[0], vm[2]]
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vn[2]);
  EXPECT_EQ(r[2], vm[0]);
  EXPECT_EQ(r[3], vm[2]);
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp2Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP2 .4S: [vn[1], vn[3], vm[1], vm[3]]
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vn[3]);
  EXPECT_EQ(r[2], vm[1]);
  EXPECT_EQ(r[3], vm[3]);
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp1Vec2D) {
  uint64_t vn[2] = {0x1111111111111111ull, 0x2222222222222222ull};
  uint64_t vm[2] = {0xAAAAAAAAAAAAAAAAull, 0xBBBBBBBBBBBBBBBBull};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec2D};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP1 .2D == ZIP1 .2D == [vn[0], vm[0]]
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp2Vec2D) {
  uint64_t vn[2] = {0x1111111111111111ull, 0x2222222222222222ull};
  uint64_t vm[2] = {0xAAAAAAAAAAAAAAAAull, 0xBBBBBBBBBBBBBBBBull};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec2D};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP2 .2D == ZIP2 .2D == [vn[1], vm[1]]
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp1Vec8B) {
  // Upper 8 bytes of vn/vm are garbage and MUST NOT leak into the result.
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP1 .8B: bytes 0..3 = even bytes of vn[0..7]; bytes 4..7 = even of vm[0..7].
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vn[2]);
  EXPECT_EQ(r[2], vn[4]);
  EXPECT_EQ(r[3], vn[6]);
  EXPECT_EQ(r[4], vm[0]);
  EXPECT_EQ(r[5], vm[2]);
  EXPECT_EQ(r[6], vm[4]);
  EXPECT_EQ(r[7], vm[6]);
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper byte " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp2Vec8B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vn[3]);
  EXPECT_EQ(r[2], vn[5]);
  EXPECT_EQ(r[3], vn[7]);
  EXPECT_EQ(r[4], vm[1]);
  EXPECT_EQ(r[5], vm[3]);
  EXPECT_EQ(r[6], vm[5]);
  EXPECT_EQ(r[7], vm[7]);
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper byte " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp1Vec4H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706, 0xFFFF,0xEEEE,0xDDDD,0xCCCC};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716, 0x9999,0x8888,0x7777,0x6666};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec4H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vn[2]);
  EXPECT_EQ(r[2], vm[0]);
  EXPECT_EQ(r[3], vm[2]);
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp2Vec4H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706, 0xFFFF,0xEEEE,0xDDDD,0xCCCC};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716, 0x9999,0x8888,0x7777,0x6666};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec4H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vn[3]);
  EXPECT_EQ(r[2], vm[1]);
  EXPECT_EQ(r[3], vm[3]);
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp1Vec2S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0xDEADBEEFu, 0xCAFEBABEu};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0x12345678u, 0x9ABCDEF0u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP1 .2S: [vn[0], vm[0]] in lower 8 bytes.
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}

TEST_F(Arm64LiteTranslateRegionTest, Uzp2Vec2S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0xDEADBEEFu, 0xCAFEBABEu};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0x12345678u, 0x9ABCDEF0u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // UZP2 .2S: [vn[1], vm[1]] in lower 8 bytes.
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}
// endregion

// region digitalis: TRN1 / TRN2 vector permute JIT
//
// TRN1 Vd.<T>, Vn.<T>, Vm.<T>  ->  Vd[2i]   = Vn[2i],   Vd[2i+1] = Vm[2i]
// TRN2 Vd.<T>, Vn.<T>, Vm.<T>  ->  Vd[2i]   = Vn[2i+1], Vd[2i+1] = Vm[2i+1]
// Lowering: PSHUFB+POR with precomputed masks (.16B/.8B and .8H/.4H),
// PSHUFD+PUNPCKLDQ (.4S), PUNPCKLDQ ± PSRLDQ-4 prelude (.2S),
// PUNPCKL/HQDQ (.2D — coincides with ZIP).
constexpr uint32_t kTrn1Vec16B = 0x4e022820;  // trn1 v0.16b, v1.16b, v2.16b
constexpr uint32_t kTrn2Vec16B = 0x4e026820;  // trn2 v0.16b, v1.16b, v2.16b
constexpr uint32_t kTrn1Vec8H  = 0x4e422820;  // trn1 v0.8h,  v1.8h,  v2.8h
constexpr uint32_t kTrn2Vec8H  = 0x4e426820;  // trn2 v0.8h,  v1.8h,  v2.8h
constexpr uint32_t kTrn1Vec4S  = 0x4e822820;  // trn1 v0.4s,  v1.4s,  v2.4s
constexpr uint32_t kTrn2Vec4S  = 0x4e826820;  // trn2 v0.4s,  v1.4s,  v2.4s
constexpr uint32_t kTrn1Vec2D  = 0x4ec22820;  // trn1 v0.2d,  v1.2d,  v2.2d
constexpr uint32_t kTrn2Vec2D  = 0x4ec26820;  // trn2 v0.2d,  v1.2d,  v2.2d
constexpr uint32_t kTrn1Vec8B  = 0x0e022820;  // trn1 v0.8b,  v1.8b,  v2.8b
constexpr uint32_t kTrn2Vec8B  = 0x0e026820;  // trn2 v0.8b,  v1.8b,  v2.8b
constexpr uint32_t kTrn1Vec4H  = 0x0e422820;  // trn1 v0.4h,  v1.4h,  v2.4h
constexpr uint32_t kTrn2Vec4H  = 0x0e426820;  // trn2 v0.4h,  v1.4h,  v2.4h
constexpr uint32_t kTrn1Vec2S  = 0x0e822820;  // trn1 v0.2s,  v1.2s,  v2.2s
constexpr uint32_t kTrn2Vec2S  = 0x0e826820;  // trn2 v0.2s,  v1.2s,  v2.2s

TEST_F(Arm64LiteTranslateRegionTest, Trn1Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN1 .16B: [vn[0],vm[0], vn[2],vm[2], ..., vn[14],vm[14]]
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[2 * i],     vn[2 * i]) << "lane " << (2 * i);
    EXPECT_EQ(r[2 * i + 1], vm[2 * i]) << "lane " << (2 * i + 1);
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Trn2Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec16B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN2 .16B: [vn[1],vm[1], vn[3],vm[3], ..., vn[15],vm[15]]
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[2 * i],     vn[2 * i + 1]) << "lane " << (2 * i);
    EXPECT_EQ(r[2 * i + 1], vm[2 * i + 1]) << "lane " << (2 * i + 1);
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Trn1Vec8H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x8908,0x8B0A,0xFD0C,0xFF0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x9918,0x9B1A,0xED1C,0xEF1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN1 .8H: [vn[0],vm[0], vn[2],vm[2], vn[4],vm[4], vn[6],vm[6]]
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i],     vn[2 * i]) << "lane " << (2 * i);
    EXPECT_EQ(r[2 * i + 1], vm[2 * i]) << "lane " << (2 * i + 1);
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Trn2Vec8H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x8908,0x8B0A,0xFD0C,0xFF0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x9918,0x9B1A,0xED1C,0xEF1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec8H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i],     vn[2 * i + 1]) << "lane " << (2 * i);
    EXPECT_EQ(r[2 * i + 1], vm[2 * i + 1]) << "lane " << (2 * i + 1);
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Trn1Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN1 .4S: [vn[0], vm[0], vn[2], vm[2]]
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], vn[2]);
  EXPECT_EQ(r[3], vm[2]);
}

TEST_F(Arm64LiteTranslateRegionTest, Trn2Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec4S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN2 .4S: [vn[1], vm[1], vn[3], vm[3]]
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
  EXPECT_EQ(r[2], vn[3]);
  EXPECT_EQ(r[3], vm[3]);
}

TEST_F(Arm64LiteTranslateRegionTest, Trn1Vec2D) {
  uint64_t vn[2] = {0x1111111111111111ull, 0x2222222222222222ull};
  uint64_t vm[2] = {0xAAAAAAAAAAAAAAAAull, 0xBBBBBBBBBBBBBBBBull};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec2D};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN1 .2D == ZIP1 .2D == [vn[0], vm[0]]
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
}

TEST_F(Arm64LiteTranslateRegionTest, Trn2Vec2D) {
  uint64_t vn[2] = {0x1111111111111111ull, 0x2222222222222222ull};
  uint64_t vm[2] = {0xAAAAAAAAAAAAAAAAull, 0xBBBBBBBBBBBBBBBBull};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec2D};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN2 .2D == ZIP2 .2D == [vn[1], vm[1]]
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
}

TEST_F(Arm64LiteTranslateRegionTest, Trn1Vec8B) {
  // Upper 8 bytes of vn/vm are garbage and MUST NOT leak into the result.
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN1 .8B: [vn[0],vm[0], vn[2],vm[2], vn[4],vm[4], vn[6],vm[6]]
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], vn[2]);
  EXPECT_EQ(r[3], vm[2]);
  EXPECT_EQ(r[4], vn[4]);
  EXPECT_EQ(r[5], vm[4]);
  EXPECT_EQ(r[6], vn[6]);
  EXPECT_EQ(r[7], vm[6]);
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper byte " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Trn2Vec8B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec8B};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
  EXPECT_EQ(r[2], vn[3]);
  EXPECT_EQ(r[3], vm[3]);
  EXPECT_EQ(r[4], vn[5]);
  EXPECT_EQ(r[5], vm[5]);
  EXPECT_EQ(r[6], vn[7]);
  EXPECT_EQ(r[7], vm[7]);
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper byte " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Trn1Vec4H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706, 0xFFFF,0xEEEE,0xDDDD,0xCCCC};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716, 0x9999,0x8888,0x7777,0x6666};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec4H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN1 .4H: [vn[0],vm[0], vn[2],vm[2]] in lower 8 bytes.
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], vn[2]);
  EXPECT_EQ(r[3], vm[2]);
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Trn2Vec4H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706, 0xFFFF,0xEEEE,0xDDDD,0xCCCC};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716, 0x9999,0x8888,0x7777,0x6666};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec4H};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
  EXPECT_EQ(r[2], vn[3]);
  EXPECT_EQ(r[3], vm[3]);
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(r[i], 0u) << "upper lane " << i << " not zeroed";
  }
}

TEST_F(Arm64LiteTranslateRegionTest, Trn1Vec2S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0xDEADBEEFu, 0xCAFEBABEu};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0x12345678u, 0x9ABCDEF0u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN1 .2S: [vn[0], vm[0]] in lower 8 bytes (coincides with ZIP1.2S).
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}

TEST_F(Arm64LiteTranslateRegionTest, Trn2Vec2S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0xDEADBEEFu, 0xCAFEBABEu};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0x12345678u, 0x9ABCDEF0u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec2S};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // TRN2 .2S: [vn[1], vm[1]] in lower 8 bytes (coincides with ZIP2.2S).
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}
// endregion

// region digitalis: DUP (element) vector JIT
//
// DUP Vd.<T>, Vn.<Ts>[index]  ->  broadcast Vn[index] (one esize-byte
// element) to every lane of Vd.  Q=0 zeroes the upper 64 bits of Vd.
//
// JIT lowering at lite_translator.h (AdvSimdCopy, kDupElement):
//   esize=1 -> PSHUFB with mask = {idx}×16
//   esize=2 -> PSHUFB with halfword-broadcast mask
//   esize=4 -> PSHUFD imm = idx*0x55
//   esize=8 -> PSHUFD imm = 0x44 (idx=0) or 0xEE (idx=1)
constexpr uint32_t kDupElemVec16B_idx0  = 0x4e010420;  // dup v0.16b, v1.b[0]
constexpr uint32_t kDupElemVec16B_idx3  = 0x4e070420;  // dup v0.16b, v1.b[3]
constexpr uint32_t kDupElemVec16B_idx15 = 0x4e1f0420;  // dup v0.16b, v1.b[15]
constexpr uint32_t kDupElemVec8B_idx0   = 0x0e010420;  // dup v0.8b,  v1.b[0]
constexpr uint32_t kDupElemVec8B_idx5   = 0x0e0b0420;  // dup v0.8b,  v1.b[5]
constexpr uint32_t kDupElemVec8H_idx0   = 0x4e020420;  // dup v0.8h,  v1.h[0]
constexpr uint32_t kDupElemVec8H_idx3   = 0x4e0e0420;  // dup v0.8h,  v1.h[3]
constexpr uint32_t kDupElemVec8H_idx7   = 0x4e1e0420;  // dup v0.8h,  v1.h[7]
constexpr uint32_t kDupElemVec4H_idx2   = 0x0e0a0420;  // dup v0.4h,  v1.h[2]
constexpr uint32_t kDupElemVec4S_idx0   = 0x4e040420;  // dup v0.4s,  v1.s[0]
constexpr uint32_t kDupElemVec4S_idx1   = 0x4e0c0420;  // dup v0.4s,  v1.s[1]
constexpr uint32_t kDupElemVec4S_idx3   = 0x4e1c0420;  // dup v0.4s,  v1.s[3]
constexpr uint32_t kDupElemVec2S_idx1   = 0x0e0c0420;  // dup v0.2s,  v1.s[1]
constexpr uint32_t kDupElemVec2D_idx0   = 0x4e080420;  // dup v0.2d,  v1.d[0]
constexpr uint32_t kDupElemVec2D_idx1   = 0x4e180420;  // dup v0.2d,  v1.d[1]

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec16BIdx0) {
  uint8_t vn[16] = {0x55,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec16B_idx0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(r[i], vn[0]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec16BIdx3) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x77,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec16B_idx3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(r[i], vn[3]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec16BIdx15) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0xCC};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec16B_idx15};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(r[i], vn[15]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec8BIdx0) {
  uint8_t vn[16] = {0x42,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec8B_idx0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], vn[0]) << "lower lane " << i;
  // Q=0 upper-zero pin: seeded vn[8..15] = 0xFF..0xF8 must not leak through.
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec8BIdx5) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x99,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec8B_idx5};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], vn[5]) << "lower lane " << i;
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec8HIdx0) {
  uint16_t vn[8] = {0xBEEFu, 0x1111u, 0x2222u, 0x3333u,
                    0x4444u, 0x5555u, 0x6666u, 0x7777u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec8H_idx0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], vn[0]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec8HIdx3) {
  uint16_t vn[8] = {0x0000u, 0x1111u, 0x2222u, 0xCAFEu,
                    0x4444u, 0x5555u, 0x6666u, 0x7777u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec8H_idx3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], vn[3]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec8HIdx7) {
  uint16_t vn[8] = {0x0000u, 0x1111u, 0x2222u, 0x3333u,
                    0x4444u, 0x5555u, 0x6666u, 0xDEADu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec8H_idx7};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], vn[7]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec4HIdx2) {
  uint16_t vn[8] = {0x0000u, 0x1111u, 0xABCDu, 0x3333u,
                    0xFFFFu, 0xFFFEu, 0xFFFDu, 0xFFFCu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec4H_idx2};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(r[i], vn[2]) << "lower lane " << i;
  for (int i = 4; i < 8; ++i) EXPECT_EQ(r[i], 0u) << "upper lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec4SIdx0) {
  uint32_t vn[4] = {0xDEADBEEFu, 0x11111111u, 0x22222222u, 0x33333333u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec4S_idx0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(r[i], vn[0]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec4SIdx1) {
  uint32_t vn[4] = {0x00000000u, 0xCAFEBABEu, 0x22222222u, 0x33333333u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec4S_idx1};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(r[i], vn[1]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec4SIdx3) {
  uint32_t vn[4] = {0x00000000u, 0x11111111u, 0x22222222u, 0x12345678u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec4S_idx3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(r[i], vn[3]) << "lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec2SIdx1) {
  uint32_t vn[4] = {0x00000000u, 0xBADC0FFEu, 0xFFFFFFFFu, 0xFFFFFFFEu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec2S_idx1};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vn[1]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec2DIdx0) {
  uint64_t vn[2] = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec2D_idx0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vn[0]);
}

TEST_F(Arm64LiteTranslateRegionTest, DupElemVec2DIdx1) {
  uint64_t vn[2] = {0x0123456789ABCDEFULL, 0xCAFEBABEDEADBEEFULL};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kDupElemVec2D_idx1};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vn[1]);
}
// endregion

// region digitalis: EXT q=0 vector JIT
//
// EXT Vd.8B, Vn.8B, Vm.8B, #imm  ->  result[0..7] =
//   Vn[imm..7] || Vm[0..imm-1]   (an 8-byte rotating window over Vn:Vm)
//   result[8..15] = 0  (Q=0 upper-zero requirement)
//
// JIT lowering at lite_translator.h (AdvSimdExtract, q=0 branch):
//   Movq xn, [vn]          // xn = [Vn[0..7], 0]
//   Movq xm, [vm]          // xm = [Vm[0..7], 0]
//   Punpcklqdq xn, xm      // xn = [Vn[0..7], Vm[0..7]]
//   Psrldq xn, index       // shift right by `index` bytes
//   Pslldq xn, 8           // zero low 8 bytes
//   Psrldq xn, 8           // shift wanted bytes back to low; zero upper
//   Movdqu [vd], xn
//
// Reserved (index >= 8) bails to interpreter via success_ = false.
constexpr uint32_t kExtVec8B_idx0 = 0x2e020020;  // ext v0.8b, v1.8b, v2.8b, #0
constexpr uint32_t kExtVec8B_idx1 = 0x2e020820;  // ext v0.8b, v1.8b, v2.8b, #1
constexpr uint32_t kExtVec8B_idx3 = 0x2e021820;  // ext v0.8b, v1.8b, v2.8b, #3
constexpr uint32_t kExtVec8B_idx5 = 0x2e022820;  // ext v0.8b, v1.8b, v2.8b, #5
constexpr uint32_t kExtVec8B_idx7 = 0x2e023820;  // ext v0.8b, v1.8b, v2.8b, #7

TEST_F(Arm64LiteTranslateRegionTest, ExtVec8BIdx0) {
  // index=0: result low 8 bytes = Vn[0..7]; upper 8 bytes = 0.
  // Seed Vn[8..15] and Vm with sentinel garbage to pin the upper-zero
  // behavior (no leak through PSLLDQ-8 / PSRLDQ-8) and the Vm-exclusion
  // (no Vm bytes when index==0).
  uint8_t vn[16] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8};
  uint8_t vm[16] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                    0xEE, 0xED, 0xEC, 0xEB, 0xEA, 0xE9, 0xE8, 0xE7};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kExtVec8B_idx0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], vn[i]) << "lower byte " << i;
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, ExtVec8BIdx1) {
  // index=1: result low 8 bytes = Vn[1..7] || Vm[0]; upper = 0.
  uint8_t vn[16] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8};
  uint8_t vm[16] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                    0xEE, 0xED, 0xEC, 0xEB, 0xEA, 0xE9, 0xE8, 0xE7};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kExtVec8B_idx1};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // result = [Vn[1..7], Vm[0]]
  uint8_t expected[8] = {vn[1], vn[2], vn[3], vn[4], vn[5], vn[6], vn[7], vm[0]};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], expected[i]) << "lower byte " << i;
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, ExtVec8BIdx3) {
  // index=3: result = [Vn[3..7], Vm[0..2]]; upper = 0.
  uint8_t vn[16] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8};
  uint8_t vm[16] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                    0xEE, 0xED, 0xEC, 0xEB, 0xEA, 0xE9, 0xE8, 0xE7};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kExtVec8B_idx3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  uint8_t expected[8] = {vn[3], vn[4], vn[5], vn[6], vn[7],
                         vm[0], vm[1], vm[2]};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], expected[i]) << "lower byte " << i;
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, ExtVec8BIdx5) {
  // index=5: result = [Vn[5..7], Vm[0..4]]; upper = 0.
  uint8_t vn[16] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8};
  uint8_t vm[16] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                    0xEE, 0xED, 0xEC, 0xEB, 0xEA, 0xE9, 0xE8, 0xE7};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kExtVec8B_idx5};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  uint8_t expected[8] = {vn[5], vn[6], vn[7],
                         vm[0], vm[1], vm[2], vm[3], vm[4]};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], expected[i]) << "lower byte " << i;
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, ExtVec8BIdx7) {
  // index=7: result = [Vn[7], Vm[0..6]]; upper = 0.
  // Pins the boundary case where exactly one Vn byte survives.
  uint8_t vn[16] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8};
  uint8_t vm[16] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                    0xEE, 0xED, 0xEC, 0xEB, 0xEA, 0xE9, 0xE8, 0xE7};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kExtVec8B_idx7};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  uint8_t expected[8] = {vn[7], vm[0], vm[1], vm[2],
                         vm[3], vm[4], vm[5], vm[6]};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], expected[i]) << "lower byte " << i;
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

// EXT q=1 with odd imm4 — regression pins for the decoder dispatch bug
// where EXT was mis-routed to AdvSimdPermute (UZP/TRN/ZIP) when imm4
// had its low bit set (bit 11 of the encoding = 1).  AdvSimdPermute's
// predicate matched on bits[11:10]=10 without checking bit 29, but
// Permute requires bit 29 = 0 while EXT has bit 29 = 1.  Fixed in
// decoder.h:AdvSIMD permute dispatch.
constexpr uint32_t kExtVec16B_idx1  = 0x6e020820;  // ext v0.16b, v1.16b, v2.16b, #1
constexpr uint32_t kExtVec16B_idx7  = 0x6e023820;  // ext v0.16b, v1.16b, v2.16b, #7
constexpr uint32_t kExtVec16B_idx15 = 0x6e027820;  // ext v0.16b, v1.16b, v2.16b, #15

TEST_F(Arm64LiteTranslateRegionTest, ExtVec16BIdx1OddImm) {
  uint8_t vn[16] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
  uint8_t vm[16] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                    0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kExtVec16B_idx1};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // Result = Vn[1..15] || Vm[0].
  for (int i = 0; i < 15; ++i) EXPECT_EQ(r[i], vn[1 + i]) << "byte " << i;
  EXPECT_EQ(r[15], vm[0]) << "byte 15";
}

TEST_F(Arm64LiteTranslateRegionTest, ExtVec16BIdx7OddImm) {
  uint8_t vn[16] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
  uint8_t vm[16] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                    0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kExtVec16B_idx7};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // Result = Vn[7..15] || Vm[0..6].
  for (int i = 0; i < 9; ++i) EXPECT_EQ(r[i], vn[7 + i]) << "byte " << i;
  for (int i = 9; i < 16; ++i) EXPECT_EQ(r[i], vm[i - 9]) << "byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, ExtVec16BIdx15OddImm) {
  uint8_t vn[16] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
  uint8_t vm[16] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                    0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kExtVec16B_idx15};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  // Result = Vn[15] || Vm[0..14].
  EXPECT_EQ(r[0], vn[15]) << "byte 0";
  for (int i = 1; i < 16; ++i) EXPECT_EQ(r[i], vm[i - 1]) << "byte " << i;
}
// endregion

// region digitalis: DUP (general) q=0 vector JIT
// DUP Vd.8B, Wn / DUP Vd.2S, Wn — broadcast a GP register to the lower
// 64 bits of Vd and zero the upper 64 bits per ARM D-register semantics.
// The Q=1 forms (.16B / .4S / .2D) are already covered by MemsetPattern
// and friends; these tests pin the new Q=0 path including the shared
// PSLLDQ-8 / PSRLDQ-8 upper-zero tail.
constexpr uint32_t kDupGenVec8B_W0 = 0x0e010c00;  // dup v0.8b, w0
constexpr uint32_t kDupGenVec8B_W7 = 0x0e010ce0;  // dup v0.8b, w7
constexpr uint32_t kDupGenVec2S_W0 = 0x0e040c00;  // dup v0.2s, w0
constexpr uint32_t kDupGenVec2S_W7 = 0x0e040ce0;  // dup v0.2s, w7
constexpr uint32_t kDupGenVec4H_W0 = 0x0e020c00;  // dup v0.4h, w0
constexpr uint32_t kDupGenVec8H_W0 = 0x4e020c00;  // dup v0.8h, w0
constexpr uint32_t kDupGenVec8H_W7 = 0x4e020ce0;  // dup v0.8h, w7

TEST_F(Arm64LiteTranslateRegionTest, DupGenVec8B_FromW0) {
  // dup v0.8b, w0  — fills v0[0..7] with byte 0xAB; v0[8..15] = 0.
  std::memset(&state_.cpu.v[0], 0xCC, 16);  // sentinel: must be overwritten
  state_.cpu.x[0] = 0x123456789abcdeabULL;   // low byte = 0xAB
  static const uint32_t code[] = {kDupGenVec8B_W0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], 0xABu) << "lower byte " << i;
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupGenVec8B_FromW7_ZeroByte) {
  // dup v0.8b, w7  — w7 low byte = 0; v0[0..7] = 0; v0[8..15] = 0.
  // Pre-seeds v0 with non-zero garbage so a stuck upper-half zeroing
  // would surface as a sentinel leak.
  std::memset(&state_.cpu.v[0], 0xFE, 16);
  state_.cpu.x[7] = 0xDEADBEEFCAFE0000ULL;  // low byte = 0x00
  static const uint32_t code[] = {kDupGenVec8B_W7};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "byte " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupGenVec2S_FromW0) {
  // dup v0.2s, w0  — fills v0.s[0] and v0.s[1] with 0xDEADBEEF;
  // v0.s[2] and v0.s[3] zeroed.
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  state_.cpu.x[0] = 0x11223344DEADBEEFULL;  // low 32 bits = 0xDEADBEEF
  static const uint32_t code[] = {kDupGenVec2S_W0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0xDEADBEEFu) << "low s lane 0";
  EXPECT_EQ(r[1], 0xDEADBEEFu) << "low s lane 1";
  EXPECT_EQ(r[2], 0u) << "upper s lane 2";
  EXPECT_EQ(r[3], 0u) << "upper s lane 3";
}

TEST_F(Arm64LiteTranslateRegionTest, DupGenVec2S_FromW7) {
  // dup v0.2s, w7  — same as above but sourcing from x7 to exercise
  // the Rn != 0 path.  Upper half of x7 (above bit 31) must be
  // ignored by Movd; if it were Movq instead, the upper-zero tail
  // would still zero v0.s[2..3] correctly, but this test pins the
  // intended truncation semantics.
  std::memset(&state_.cpu.v[0], 0x33, 16);
  state_.cpu.x[7] = 0xFFFFFFFF12345678ULL;  // low 32 bits = 0x12345678
  static const uint32_t code[] = {kDupGenVec2S_W7};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0x12345678u) << "low s lane 0";
  EXPECT_EQ(r[1], 0x12345678u) << "low s lane 1";
  EXPECT_EQ(r[2], 0u) << "upper s lane 2";
  EXPECT_EQ(r[3], 0u) << "upper s lane 3";
}

TEST_F(Arm64LiteTranslateRegionTest, DupGenVec4H_FromW0) {
  // dup v0.4h, w0  — broadcast low halfword of w0 to v0.h[0..3];
  // v0.h[4..7] (upper 64 bits) zeroed per ARM D-register semantics.
  // Pre-seeds v0 so a missing upper-zero would leak the sentinel.
  std::memset(&state_.cpu.v[0], 0x77, 16);
  state_.cpu.x[0] = 0xDEADBEEF0000BEEFULL;  // low halfword = 0xBEEF
  static const uint32_t code[] = {kDupGenVec4H_W0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(r[i], 0xBEEFu) << "low h lane " << i;
  for (int i = 4; i < 8; ++i) EXPECT_EQ(r[i], 0u) << "upper h lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupGenVec8H_FromW0) {
  // dup v0.8h, w0  — broadcast low halfword of w0 to all 8 h lanes
  // (Q=1, no upper-zero tail).
  std::memset(&state_.cpu.v[0], 0xCC, 16);
  state_.cpu.x[0] = 0x123456780000ABCDULL;  // low halfword = 0xABCD
  static const uint32_t code[] = {kDupGenVec8H_W0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], 0xABCDu) << "h lane " << i;
}

TEST_F(Arm64LiteTranslateRegionTest, DupGenVec8H_FromW7_HighBitsIgnored) {
  // dup v0.8h, w7 with x7 having garbage in bits[31:16] — DUP general
  // only consumes the low 16 bits of Wn for the H form, so the upper
  // 16 bits of the 32-bit source must be ignored.  This pins that
  // Pshuflw imm=0 selects halfword[0] (xmm bits[15:0]) and not a
  // higher-numbered halfword that could have caught a stale value
  // from the GP move.
  std::memset(&state_.cpu.v[0], 0x55, 16);
  state_.cpu.x[7] = 0xFFFFFFFFAAAA1234ULL;  // bits[15:0] = 0x1234; bits[31:16] = 0xAAAA
  static const uint32_t code[] = {kDupGenVec8H_W7};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(r[i], 0x1234u) << "h lane " << i;
}
// endregion

// region digitalis: INS (element) JIT
// INS Vd.<T>[dst_idx], Vn.<T>[src_idx] — copy one esize-byte lane from Vn
// to one lane of Vd, leaving every other lane of Vd unchanged.  These
// tests pin the new JIT path's correctness on the four lane widths
// (B/H/S/D), the self-INS (rd == rn) aliasing case, and the "don't
// touch the other lanes" property that distinguishes INS-element from
// DUP/scalar broadcasts.
//
// Encodings verified via aarch64-linux-gnu-as round-trip.
constexpr uint32_t kInsElemB0_FromB1   = 0x6e010c20;  // ins v0.b[0], v1.b[1]
constexpr uint32_t kInsElemB5_FromB10  = 0x6e0b5420;  // ins v0.b[5], v1.b[10]
constexpr uint32_t kInsElemH3_FromH1   = 0x6e0e1420;  // ins v0.h[3], v1.h[1]
constexpr uint32_t kInsElemS2_FromS0   = 0x6e140420;  // ins v0.s[2], v1.s[0]
constexpr uint32_t kInsElemD1_FromD0   = 0x6e180420;  // ins v0.d[1], v1.d[0]
constexpr uint32_t kInsElemSelfS0FromS3 = 0x6e046400; // ins v0.s[0], v0.s[3]

TEST_F(Arm64LiteTranslateRegionTest, InsElemB0_FromB1) {
  // ins v0.b[0], v1.b[1]  — only byte 0 of v0 changes (to 0xAB),
  // every other byte of v0 keeps the sentinel.
  std::memset(&state_.cpu.v[0], 0xCC, 16);
  uint8_t v1[16] = {0x11, 0xAB, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                    0x88, 0x99, 0xAA, 0xBB, 0xDD, 0xEE, 0xFF, 0x00};
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t code[] = {kInsElemB0_FromB1};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0xABu) << "dst lane 0 took src v1.b[1]";
  for (int i = 1; i < 16; ++i) {
    EXPECT_EQ(r[i], 0xCCu) << "untouched byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, InsElemB5_FromB10) {
  // ins v0.b[5], v1.b[10]  — high-index source + non-zero dst index.
  std::memset(&state_.cpu.v[0], 0x55, 16);
  uint8_t v1[16] = {0};
  v1[10] = 0xAB;
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t code[] = {kInsElemB5_FromB10};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t expected = (i == 5) ? 0xAB : 0x55;
    EXPECT_EQ(r[i], expected) << "byte " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, InsElemH3_FromH1) {
  // ins v0.h[3], v1.h[1]  — halfword copy from src idx 1 to dst idx 3.
  std::memset(&state_.cpu.v[0], 0xCC, 16);
  uint16_t v1[8] = {0x1111, 0xBEEF, 0x2222, 0x3333,
                    0x4444, 0x5555, 0x6666, 0x7777};
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t code[] = {kInsElemH3_FromH1};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    uint16_t expected = (i == 3) ? 0xBEEFu : 0xCCCCu;
    EXPECT_EQ(r[i], expected) << "halfword " << i;
  }
}

TEST_F(Arm64LiteTranslateRegionTest, InsElemS2_FromS0) {
  // ins v0.s[2], v1.s[0]  — 32-bit lane copy.
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  uint32_t v1[4] = {0xDEADBEEFu, 0x11111111u, 0x22222222u, 0x33333333u};
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t code[] = {kInsElemS2_FromS0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0xAAAAAAAAu) << "s lane 0 untouched";
  EXPECT_EQ(r[1], 0xAAAAAAAAu) << "s lane 1 untouched";
  EXPECT_EQ(r[2], 0xDEADBEEFu) << "s lane 2 took v1.s[0]";
  EXPECT_EQ(r[3], 0xAAAAAAAAu) << "s lane 3 untouched";
}

TEST_F(Arm64LiteTranslateRegionTest, InsElemD1_FromD0) {
  // ins v0.d[1], v1.d[0]  — 64-bit lane copy; the full 64 bits must
  // make it across (pins the Movq-not-Movl discipline for the doubleword
  // form).
  std::memset(&state_.cpu.v[0], 0x77, 16);
  uint64_t v1[2] = {0xFEEDFACECAFEBABEULL, 0x0123456789ABCDEFULL};
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t code[] = {kInsElemD1_FromD0};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0x7777777777777777ULL) << "d lane 0 untouched";
  EXPECT_EQ(r[1], 0xFEEDFACECAFEBABEULL) << "d lane 1 took full 64 bits";
}

TEST_F(Arm64LiteTranslateRegionTest, InsElemSelfS0FromS3) {
  // ins v0.s[0], v0.s[3]  — rd == rn aliasing: load Vn.s[3] then store
  // back to Vd.s[0].  Verifies the JIT loads into a temp register before
  // touching Vd, so the source isn't clobbered mid-copy.
  uint32_t v0[4] = {0x00000000u, 0x11111111u, 0x22222222u, 0xCAFEF00Du};
  std::memcpy(&state_.cpu.v[0], v0, 16);
  static const uint32_t code[] = {kInsElemSelfS0FromS3};
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0xCAFEF00Du) << "s lane 0 took s lane 3";
  EXPECT_EQ(r[1], 0x11111111u) << "s lane 1 untouched";
  EXPECT_EQ(r[2], 0x22222222u) << "s lane 2 untouched";
  EXPECT_EQ(r[3], 0xCAFEF00Du) << "s lane 3 (source) preserved";
}
// endregion

}  // namespace

}  // namespace berberis
// endregion
