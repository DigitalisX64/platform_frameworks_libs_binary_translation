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

}  // namespace

}  // namespace berberis
// endregion
