/*
 * Copyright (C) 2024 The Android Open Source Project
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
#include <initializer_list>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/runtime_primitives/memory_region_reservation.h"

namespace berberis {

namespace {

class Riscv64ToArm64InterpreterTest : public ::testing::Test {
 public:
  template <uint8_t kInsnSize = 4>
  bool RunOneInstruction(ThreadState* state, GuestAddr stop_pc) {
    InterpretInsn(state);
    return state->cpu.insn_addr == stop_pc;
  }

  template <uint8_t kInsnSize = 4>
  void RunInstruction(const uint32_t& insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    EXPECT_TRUE(RunOneInstruction<kInsnSize>(&state_, state_.cpu.insn_addr + kInsnSize));
  }

  void TestOp(uint32_t insn_bytes,
              std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      SetXReg<2>(state_.cpu, arg1);
      SetXReg<3>(state_.cpu, arg2);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  void TestOpImm(uint32_t insn_bytes,
                 std::initializer_list<std::tuple<uint64_t, uint16_t, uint64_t>> args) {
    for (auto [arg1, imm, expected_result] : args) {
      CHECK_LE(imm, 63);
      uint32_t insn_bytes_with_immediate = insn_bytes | imm << 20;
      SetXReg<2>(state_.cpu, arg1);
      RunInstruction(insn_bytes_with_immediate);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  void TestAuipc(uint32_t insn_bytes, uint64_t expected_offset) {
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_offset + ToGuestAddr(&insn_bytes));
  }

  void TestLui(uint32_t insn_bytes, uint64_t expected_result) {
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  void TestBranch(uint32_t insn_bytes,
                  std::initializer_list<std::tuple<uint64_t, uint64_t, int8_t>> args) {
    auto code_start = ToGuestAddr(&insn_bytes);
    for (auto [arg1, arg2, expected_offset] : args) {
      state_.cpu.insn_addr = code_start;
      SetXReg<1>(state_.cpu, arg1);
      SetXReg<2>(state_.cpu, arg2);
      InterpretInsn(&state_);
      EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    }
  }

  void TestJumpAndLink(uint32_t insn_bytes, int8_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    InterpretInsn(&state_);
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + 4);
  }

  void TestLoad(uint32_t insn_bytes, uint64_t expected_result) {
    // Offset is always 8.
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  // kLinkRegisterOffsetIfUsed is size of instruction or 0 if instruction does not link register.
  template <uint8_t kLinkRegisterOffsetIfUsed>
  void TestJumpAndLinkRegister(uint32_t insn_bytes, uint64_t base_disp, int64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<1>(state_.cpu, 0);
    SetXReg<2>(state_.cpu, code_start + base_disp);
    InterpretInsn(&state_);
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    if constexpr (kLinkRegisterOffsetIfUsed == 0) {
      EXPECT_EQ(GetXReg<1>(state_.cpu), 0UL);
    } else {
      EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + kLinkRegisterOffsetIfUsed);
    }
  }

  void TestStore(uint32_t insn_bytes, uint64_t expected_result) {
    // Offset is always 8.
    SetXReg<1>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - 8));
    SetXReg<2>(state_.cpu, kDataToStore);
    store_area_ = 0;
    RunInstruction(insn_bytes);
    EXPECT_EQ(store_area_, expected_result);
  }

  void TestAtomicLoad(uint32_t insn_bytes,
                      const uint64_t* const data_to_load,
                      uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(data_to_load));
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetXReg<2>(state_.cpu), expected_result);
    EXPECT_EQ(state_.cpu.reservation_address, ToGuestAddr(data_to_load));
    // We always reserve the full 64-bit range of the reservation address.
    EXPECT_EQ(state_.cpu.reservation_value, *data_to_load);
  }

  template <typename T>
  void TestAtomicStore(uint32_t insn_bytes, T expected_result) {
    store_area_ = ~uint64_t{0};
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdead'beef);
    state_.cpu.reservation_address = ToGuestAddr(&store_area_);
    state_.cpu.reservation_value = store_area_;
    MemoryRegionReservation::SetOwner(ToGuestAddr(&store_area_), &state_.cpu);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(static_cast<T>(store_area_), expected_result);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 0u);
  }

  void TestAtomicStoreNoLoadFailure(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdead'beef);
    store_area_ = 0;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(store_area_, 0u);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 1u);
  }

  void TestAtomicStoreDifferentLoadFailure(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdead'beef);
    state_.cpu.reservation_address = ToGuestAddr(&kDataToStore);
    state_.cpu.reservation_value = 0;
    MemoryRegionReservation::SetOwner(ToGuestAddr(&kDataToStore), &state_.cpu);
    store_area_ = 0;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(store_area_, 0u);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 1u);
  }

  void TestAmo(uint32_t insn_bytes,
               uint64_t arg1,
               uint64_t arg2,
               uint64_t expected_result,
               uint64_t expected_memory) {
    // Copy arg1 into store_area_
    store_area_ = arg1;
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_)));
    SetXReg<3>(state_.cpu, arg2);
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    EXPECT_EQ(store_area_, expected_memory);
  }

  void TestAmo(uint32_t insn_bytes32, uint32_t insn_bytes64, uint64_t expected_memory) {
    TestAmo(insn_bytes32,
            0xffff'eeee'dddd'ccccULL,
            0xaaaa'bbbb'cccc'ddddULL,
            0xffff'ffff'dddd'ccccULL,
            0xffff'eeee'0000'0000 | uint32_t(expected_memory));
    TestAmo(insn_bytes64,
            0xffff'eeee'dddd'ccccULL,
            0xaaaa'bbbb'cccc'ddddULL,
            0xffff'eeee'dddd'ccccULL,
            expected_memory);
  }

 protected:
  static constexpr uint64_t kDataToLoad{0xffff'eeee'dddd'ccccULL};
  static constexpr uint64_t kDataToStore = kDataToLoad;
  uint64_t store_area_;
  ThreadState state_;
};

TEST_F(Riscv64ToArm64InterpreterTest, OpInstructions) {
  // Add
  TestOp(0x0031'00b3, {{19, 23, 42}});
  // Sub
  TestOp(0x4031'00b3, {{42, 23, 19}});
  // And
  TestOp(0x0031'70b3, {{0b0101, 0b0011, 0b0001}});
  // Or
  TestOp(0x0031'60b3, {{0b0101, 0b0011, 0b0111}});
  // Xor
  TestOp(0x0031'40b3, {{0b0101, 0b0011, 0b0110}});
  // Sll
  TestOp(0x0031'10b3, {{0b1010, 3, 0b1010'000}});
  // Srl
  TestOp(0x0031'50b3, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Sra
  TestOp(0x4031'50b3, {{0xf000'0000'0000'0000ULL, 12, 0xffff'0000'0000'0000ULL}});
  // Slt
  TestOp(0x0031'20b3,
         {
             {19, 23, 1},
             {23, 19, 0},
             {~0ULL, 0, 1},
         });
  // Sltu
  TestOp(0x0031'30b3,
         {
             {19, 23, 1},
             {23, 19, 0},
             {~0ULL, 0, 0},
         });
  // Div
  TestOp(0x231'40b3, {{0x9999'9999'9999'9999, 0x3333, 0xfffd'fffd'fffd'fffe}});
  TestOp(0x231'40b3, {{42, 2, 21}});
  TestOp(0x231'40b3, {{42, 0, -1}});
  TestOp(0x231'40b3, {{-2147483648, -1, 2147483648}});
  TestOp(0x231'40b3, {{0x8000'0000'0000'0000, -1, 0x8000'0000'0000'0000}});
  // Divu
  TestOp(0x231'50b3, {{0x9999'9999'9999'9999, 0x3333, 0x0003'0003'0003'0003}});
  TestOp(0x231'50b3, {{42, 2, 21}});
  TestOp(0x231'50b3, {{42, 0, 0xffff'ffff'ffff'ffffULL}});
  // Rem
  TestOp(0x231'60b3, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'ffff'ffff}});
  TestOp(0x231'60b3, {{0x9999'9999'9999'9999, 0, 0x9999'9999'9999'9999}});
  // Remu
  TestOp(0x231'70b3, {{0x9999'9999'9999'9999, 0x3333, 0}});
  TestOp(0x231'70b3, {{0x9999'9999'9999'9999, 0, 0x9999'9999'9999'9999}});
  // Andn
  TestOp(0x4031'70b3, {{0b0101, 0b0011, 0b0100}});
  // Orn
  TestOp(0x4031'60b3, {{0b0101, 0b0011, 0xffff'ffff'ffff'fffd}});
  // Xnor
  TestOp(0x4031'40b3, {{0b0101, 0b0011, 0xffff'ffff'ffff'fff9}});
  // Max
  TestOp(0x0a31'60b3, {{bit_cast<uint64_t>(int64_t{-5}), 4, 4}});
  TestOp(0x0a31'60b3,
         {{bit_cast<uint64_t>(int64_t{-5}),
           bit_cast<uint64_t>(int64_t{-10}),
           bit_cast<uint64_t>(int64_t{-5})}});
  // Maxu
  TestOp(0x0a31'70b3, {{50, 1, 50}});
  // Min
  TestOp(0x0a31'40b3, {{bit_cast<uint64_t>(int64_t{-5}), 4, bit_cast<uint64_t>(int64_t{-5})}});
  TestOp(0x0a31'40b3,
         {{bit_cast<uint64_t>(int64_t{-5}),
           bit_cast<uint64_t>(int64_t{-10}),
           bit_cast<uint64_t>(int64_t{-10})}});
  // Minu
  TestOp(0x0a31'50b3, {{50, 1, 1}});
  // Ror
  TestOp(0x6031'50b3, {{0xf000'0000'0000'000fULL, 4, 0xff00'0000'0000'0000ULL}});
  TestOp(0x6031'50b3, {{0xf000'0000'0000'000fULL, 8, 0x0ff0'0000'0000'0000ULL}});
  // // Rol
  TestOp(0x6031'10b3, {{0xff00'0000'0000'0000ULL, 4, 0xf000'0000'0000'000fULL}});
  TestOp(0x6031'10b3, {{0x000f'ff00'0000'000fULL, 8, 0x0fff'0000'0000'0f00ULL}});
  // Sh1add
  TestOp(0x2031'20b3,
         {{0x0008'0000'0000'0001, 0x1001'0001'0000'0000ULL, 0x1011'0001'0000'0002ULL}});
  // Sh2add
  TestOp(0x2031'40b3,
         {{0x0008'0000'0000'0001, 0x0001'0001'0000'0000ULL, 0x0021'0001'0000'0004ULL}});
  // Sh3add
  TestOp(0x2031'60b3,
         {{0x0008'0000'0000'0001, 0x1001'0011'0000'0000ULL, 0x1041'0011'0000'0008ULL}});
  // Bclr
  TestOp(0x4831'10b3, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0000ULL}});
  TestOp(0x4831'10b3, {{0b1000'0001'0000'0001ULL, 8, 0b1000'0000'0000'0001ULL}});
  // Bext
  TestOp(0x4831'50b3, {{0b1000'0001'0000'0001ULL, 0, 0b0000'0000'0000'0001ULL}});
  TestOp(0x4831'50b3, {{0b1000'0001'0000'0001ULL, 8, 0b0000'0000'0000'0001ULL}});
  TestOp(0x4831'50b3, {{0b1000'0001'0000'0001ULL, 7, 0b0000'0000'0000'0000ULL}});
  // Binv
  TestOp(0x6831'10b3, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0000ULL}});
  TestOp(0x6831'10b3, {{0b1000'0001'0000'0001ULL, 1, 0b1000'0001'0000'0011ULL}});
  // Bset
  TestOp(0x2831'10b3, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0001ULL}});
  TestOp(0x2831'10b3, {{0b1000'0001'0000'0001ULL, 1, 0b1000'0001'0000'0011ULL}});
}

TEST_F(Riscv64ToArm64InterpreterTest, OpImmInstructions) {
  // Addi
  TestOpImm(0x0001'0093, {{19, 23, 42}});
  // Slti
  TestOpImm(0x0001'2093,
            {
                {19, 23, 1},
                {23, 19, 0},
                {~0ULL, 0, 1},
            });
  // Sltiu
  TestOpImm(0x0001'3093,
            {
                {19, 23, 1},
                {23, 19, 0},
                {~0ULL, 0, 0},
            });
  // Xori
  TestOpImm(0x0001'4093, {{0b0101, 0b0011, 0b0110}});
  // Ori
  TestOpImm(0x0001'6093, {{0b0101, 0b0011, 0b0111}});
  // Andi
  TestOpImm(0x0001'7093, {{0b0101, 0b0011, 0b0001}});
  // Slli
  TestOpImm(0x0001'1093, {{0b1010, 3, 0b1010'000}});
  // Srli
  TestOpImm(0x0001'5093, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Srai
  TestOpImm(0x4001'5093, {{0xf000'0000'0000'0000ULL, 12, 0xffff'0000'0000'0000ULL}});
  // Rori
  TestOpImm(0x6001'5093, {{0xf000'0000'0000'000fULL, 4, 0xff00'0000'0000'0000ULL}});
  // Rev8
  TestOpImm(0x6b81'5093, {{0x0000'0000'0000'000fULL, 0, 0x0f00'0000'0000'0000ULL}});
  TestOpImm(0x6b81'5093, {{0xf000'0000'0000'0000ULL, 0, 0x0000'0000'0000'00f0ULL}});
  TestOpImm(0x6b81'5093, {{0x00f0'0000'0000'0000ULL, 0, 0x0000'0000'0000'f000ULL}});
  TestOpImm(0x6b81'5093, {{0x0000'000f'0000'0000ULL, 0, 0x0000'0000'0f00'0000ULL}});

  // Sext.b
  TestOpImm(0x6041'1093, {{0b1111'1110, 0, 0xffff'ffff'ffff'fffe}});  // -2
  // Sext.h
  TestOpImm(0x6051'1093, {{0b1111'1110, 0, 0xfe}});
  TestOpImm(0x6051'1093, {{0b1111'1111'1111'1110, 0, 0xffff'ffff'ffff'fffe}});
  // Bclri
  TestOpImm(0x4801'1093, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0000ULL}});
  TestOpImm(0x4801'1093, {{0b1000'0001'0000'0001ULL, 8, 0b1000'0000'0000'0001ULL}});
  // Bexti
  TestOpImm(0x4801'5093, {{0b1000'0001'0000'0001ULL, 0, 0b0000'0000'0000'0001ULL}});
  TestOpImm(0x4801'5093, {{0b1000'0001'0000'0001ULL, 8, 0b0000'0000'0000'0001ULL}});
  TestOpImm(0x4801'5093, {{0b1000'0001'0000'0001ULL, 7, 0b0000'0000'0000'0000ULL}});
  // Binvi
  TestOpImm(0x6801'1093, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0000ULL}});
  TestOpImm(0x6801'1093, {{0b1000'0001'0000'0001ULL, 1, 0b1000'0001'0000'0011ULL}});
  // Bseti
  TestOpImm(0x2801'1093, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0001ULL}});
  TestOpImm(0x2801'1093, {{0b1000'0001'0000'0001ULL, 1, 0b1000'0001'0000'0011ULL}});
}

TEST_F(Riscv64ToArm64InterpreterTest, UpperImmInstructions) {
  // Auipc
  TestAuipc(0xfedc'b097, 0xffff'ffff'fedc'b000);
  // Lui
  TestLui(0xfedc'b0b7, 0xffff'ffff'fedc'b000);
}

TEST_F(Riscv64ToArm64InterpreterTest, TestBranchInstructions) {
  // Beq
  TestBranch(0x0020'8463,
             {
                 {42, 42, 8},
                 {41, 42, 4},
                 {42, 41, 4},
             });
  // Bne
  TestBranch(0x0020'9463,
             {
                 {42, 42, 4},
                 {41, 42, 8},
                 {42, 41, 8},
             });
  // Bltu
  TestBranch(0x0020'e463,
             {
                 {41, 42, 8},
                 {42, 42, 4},
                 {42, 41, 4},
                 {0xf000'0000'0000'0000ULL, 42, 4},
                 {42, 0xf000'0000'0000'0000ULL, 8},
             });
  // Bgeu
  TestBranch(0x0020'f463,
             {
                 {42, 41, 8},
                 {42, 42, 8},
                 {41, 42, 4},
                 {0xf000'0000'0000'0000ULL, 42, 8},
                 {42, 0xf000'0000'0000'0000ULL, 4},
             });
  // Blt
  TestBranch(0x0020'c463,
             {
                 {41, 42, 8},
                 {42, 42, 4},
                 {42, 41, 4},
                 {0xf000'0000'0000'0000ULL, 42, 8},
                 {42, 0xf000'0000'0000'0000ULL, 4},
             });
  // Bge
  TestBranch(0x0020'd463,
             {
                 {42, 41, 8},
                 {42, 42, 8},
                 {41, 42, 4},
                 {0xf000'0000'0000'0000ULL, 42, 4},
                 {42, 0xf000'0000'0000'0000ULL, 8},
             });
  // Beq with negative offset.
  TestBranch(0xfe20'8ee3,
             {
                 {42, 42, -4},
             });
}

TEST_F(Riscv64ToArm64InterpreterTest, JumpAndLinkInstructions) {
  // Jal
  TestJumpAndLink(0x0080'00ef, 8);
  // Jal with negative offset.
  TestJumpAndLink(0xffdf'f0ef, -4);
}

TEST_F(Riscv64ToArm64InterpreterTest, JumpAndLinkRegisterInstructions) {
  // Jalr offset=4.
  TestJumpAndLinkRegister<4>(0x0041'00e7, 38, 42);
  // Jalr offset=-4.
  TestJumpAndLinkRegister<4>(0xffc1'00e7, 42, 38);
  // Jalr offset=5 - must properly align the target to even.
  TestJumpAndLinkRegister<4>(0x0051'00e7, 38, 42);
  // Jr offset=4.
  TestJumpAndLinkRegister<0>(0x0041'0067, 38, 42);
  // Jr offset=-4.
  TestJumpAndLinkRegister<0>(0xffc1'0067, 42, 38);
  // Jr offset=5 - must properly align the target to even.
  TestJumpAndLinkRegister<0>(0x0051'0067, 38, 42);
}

TEST_F(Riscv64ToArm64InterpreterTest, LoadInstructions) {
  // Offset is always 8.
  // Lbu
  TestLoad(0x0081'4083, kDataToLoad & 0xffULL);
  // Lhu
  TestLoad(0x0081'5083, kDataToLoad & 0xffffULL);
  // Lwu
  TestLoad(0x0081'6083, kDataToLoad & 0xffff'ffffULL);
  // Ldu
  TestLoad(0x0081'3083, kDataToLoad);
  // Lb
  TestLoad(0x0081'0083, int64_t{int8_t(kDataToLoad)});
  // Lh
  TestLoad(0x0081'1083, int64_t{int16_t(kDataToLoad)});
  // Lw
  TestLoad(0x0081'2083, int64_t{int32_t(kDataToLoad)});
}

TEST_F(Riscv64ToArm64InterpreterTest, StoreInstructions) {
  // Offset is always 8.
  // Sb
  TestStore(0x0020'8423, kDataToStore & 0xffULL);
  // Sh
  TestStore(0x0020'9423, kDataToStore & 0xffffULL);
  // Sw
  TestStore(0x0020'a423, kDataToStore & 0xffff'ffffULL);
  // Sd
  TestStore(0x0020'b423, kDataToStore);
}

TEST_F(Riscv64ToArm64InterpreterTest, AtomicLoadInstructions) {
  // Validate sign-extension of returned value.
  const uint64_t kNegative32BitValue = 0x0000'0000'8000'0000ULL;
  const uint64_t kSignExtendedNegative = 0xffff'ffff'8000'0000ULL;
  const uint64_t kPositive32BitValue = 0xffff'ffff'0000'0000ULL;
  const uint64_t kSignExtendedPositive = 0ULL;
  static_assert(static_cast<int32_t>(kSignExtendedPositive) >= 0);
  static_assert(static_cast<int32_t>(kSignExtendedNegative) < 0);

  // Lrw - sign extends from 32 to 64.
  TestAtomicLoad(0x1000'a12f, &kPositive32BitValue, kSignExtendedPositive);
  TestAtomicLoad(0x1000'a12f, &kNegative32BitValue, kSignExtendedNegative);

  // Lrd
  TestAtomicLoad(0x1000'b12f, &kDataToLoad, kDataToLoad);
}

TEST_F(Riscv64ToArm64InterpreterTest, AtomicStoreInstructions) {
  // Scw
  TestAtomicStore(0x1820'a1af, static_cast<uint32_t>(kDataToStore));

  // Scd
  TestAtomicStore(0x1820'b1af, kDataToStore);
}

TEST_F(Riscv64ToArm64InterpreterTest, AtomicStoreInstructionNoLoadFailure) {
  // Scw
  TestAtomicStoreNoLoadFailure(0x1820'a1af);

  // Scd
  TestAtomicStoreNoLoadFailure(0x1820'b1af);
}

TEST_F(Riscv64ToArm64InterpreterTest, AtomicStoreInstructionDifferentLoadFailure) {
  // Scw
  TestAtomicStoreDifferentLoadFailure(0x1820'a1af);

  // Scd
  TestAtomicStoreDifferentLoadFailure(0x1820'b1af);
}

TEST_F(Riscv64ToArm64InterpreterTest, AmoInstructions) {
  // Verifying that all aq and rl combinations work for Amoswap, but only test relaxed one for most
  // other instructions for brevity.

  // AmoswaoW/AmoswaoD
  TestAmo(0x0831'20af, 0x0831'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWAq/AmoswapDAq
  TestAmo(0x0c31'20af, 0x0c31'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWRl/AmoswapDRl
  TestAmo(0x0a31'20af, 0x0a31'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWAqrl/AmoswapDAqrl
  TestAmo(0x0e31'20af, 0x0e31'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoaddW/AmoaddD
  TestAmo(0x0031'20af, 0x0031'30af, 0xaaaa'aaaa'aaaa'aaa9);

  // AmoxorW/AmoxorD
  TestAmo(0x2031'20af, 0x2031'30af, 0x5555'5555'1111'1111);

  // AmoandW/AmoandD
  TestAmo(0x6031'20af, 0x6031'30af, 0xaaaa'aaaa'cccc'cccc);

  // AmoorW/AmoorD
  TestAmo(0x4031'20af, 0x4031'30af, 0xffff'ffff'dddd'dddd);

  // AmominW/AmominD
  TestAmo(0x8031'20af, 0x8031'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmomaxW/AmomaxD
  TestAmo(0xa031'20af, 0xa031'30af, 0xffff'eeee'dddd'ccccULL);

  // AmominuW/AmominuD
  TestAmo(0xc031'20af, 0xc031'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmomaxuW/AmomaxuD
  TestAmo(0xe031'20af, 0xe031'30af, 0xffff'eeee'dddd'ccccULL);
}

// Corresponding to interpreter_test.cc

TEST_F(Riscv64ToArm64InterpreterTest, FenceInstructions) {
  // Fence
  RunInstruction(0x0ff0'000f);
  // FenceTso
  RunInstruction(0x8330'000f);

  // FenceI explicitly not supported.
}

}  // namespace

}  // namespace berberis
