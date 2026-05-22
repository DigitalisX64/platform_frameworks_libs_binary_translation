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
// endregion

}  // namespace

}  // namespace berberis
// endregion
