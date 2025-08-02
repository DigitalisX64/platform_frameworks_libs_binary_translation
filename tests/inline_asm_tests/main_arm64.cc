/*
 * Copyright (C) 2019 The Android Open Source Project
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
#include <limits>

#include "utility.h"

namespace {

TEST(Arm64InsnTest, UnsignedBitfieldMoveNoShift) {
  uint64_t arg = 0x3952'2473'7190'7021ULL;
  uint64_t res;

  asm("ubfm %0, %1, #0, #63" : "=r"(res) : "r"(arg));

  ASSERT_EQ(res, 0x3952'2473'7190'7021ULL);
}

TEST(Arm64InsnTest, BitfieldLeftInsertion) {
  uint64_t arg = 0x3895'2286'8478'abcdULL;
  uint64_t res = 0x1101'0446'8232'5271ULL;

  asm("bfm %0, %1, #40, #15" : "=r"(res) : "r"(arg), "0"(res));

  ASSERT_EQ(res, 0x1101'04ab'cd32'5271ULL);
}

TEST(Arm64InsnTest, BitfieldRightInsertion) {
  uint64_t arg = 0x3276'5618'0937'7344ULL;
  uint64_t res = 0x1668'0396'2657'9787ULL;

  asm("bfm %0, %1, #4, #39" : "=r"(res) : "r"(arg), "0"(res));

  ASSERT_EQ(res, 0x1668'0391'8093'7734ULL);
}

TEST(Arm64InsnTest, MoveImmToFp32) {
  // The tests below verify that fmov works with various immediates.
  // Specifically, the instruction has an 8-bit immediate field consisting of
  // the following four subfields:
  //
  // - sign (one bit)
  // - upper exponent (one bit)
  // - lower exponent (two bits)
  // - mantisa (four bits)
  //
  // For example, we decompose imm8 = 0b01001111 into:
  //
  // - sign = 0 (positive)
  // - upper exponent = 1
  // - lower exponent = 00
  // - mantisa = 1111
  //
  // This immediate corresponds to 32-bit floating point value:
  //
  // 0 011111 00 1111 0000000000000000000
  // | |      |  |    |
  // | |      |  |    +- 19 zeros
  // | |      |  +------ mantisa
  // | |      +--------- lower exponent
  // | +---------------- upper exponent (custom extended to 6 bits)
  // +------------------ sign
  //
  // Thus we have:
  //
  //   1.11110000... * 2^(124-127) = 0.2421875
  //
  // where 1.11110000... is in binary.
  //
  // See VFPExpandImm in the ARM Architecture Manual for details.
  //
  // We enumerate all possible 8-bit immediate encodings of the form:
  //
  //   {0,1}{0,1}{00,11}{0000,1111}
  //
  // to verify that the decoder correctly splits the immediate into the
  // subfields and reconstructs the intended floating-point value.

  // imm8 = 0b00000000
  __uint128_t res1 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #2.0e+00")();
  ASSERT_EQ(res1, MakeUInt128(0x4000'0000U, 0U));

  // imm8 = 0b00001111
  __uint128_t res2 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #3.8750e+00")();
  ASSERT_EQ(res2, MakeUInt128(0x4078'0000U, 0U));

  // imm8 = 0b00110000
  __uint128_t res3 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #1.60e+01")();
  ASSERT_EQ(res3, MakeUInt128(0x4180'0000U, 0U));

  // imm8 = 0b00111111
  __uint128_t res4 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #3.10e+01")();
  ASSERT_EQ(res4, MakeUInt128(0x41f8'0000U, 0U));

  // imm8 = 0b01000000
  __uint128_t res5 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #1.250e-01")();
  ASSERT_EQ(res5, MakeUInt128(0x3e00'0000U, 0U));

  // imm8 = 0b01001111
  __uint128_t res6 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #2.4218750e-01")();
  ASSERT_EQ(res6, MakeUInt128(0x3e78'0000U, 0U));

  // imm8 = 0b01110000
  __uint128_t res7 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #1.0e+00")();
  ASSERT_EQ(res7, MakeUInt128(0x3f80'0000U, 0U));

  // imm8 = 0b01111111
  __uint128_t res8 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #1.93750e+00")();
  ASSERT_EQ(res8, MakeUInt128(0x3ff8'0000U, 0U));

  // imm8 = 0b10000000
  __uint128_t res9 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #-2.0e+00")();
  ASSERT_EQ(res9, MakeUInt128(0xc000'0000U, 0U));

  // imm8 = 0b10001111
  __uint128_t res10 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #-3.8750e+00")();
  ASSERT_EQ(res10, MakeUInt128(0xc078'0000U, 0U));

  // imm8 = 0b10110000
  __uint128_t res11 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #-1.60e+01")();
  ASSERT_EQ(res11, MakeUInt128(0xc180'0000U, 0U));

  // imm8 = 0b10111111
  __uint128_t res12 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #-3.10e+01")();
  ASSERT_EQ(res12, MakeUInt128(0xc1f8'0000U, 0U));

  // imm8 = 0b11000000
  __uint128_t res13 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #-1.250e-01")();
  ASSERT_EQ(res13, MakeUInt128(0xbe00'0000U, 0U));

  // imm8 = 0b11001111
  __uint128_t res14 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #-2.4218750e-01")();
  ASSERT_EQ(res14, MakeUInt128(0xbe78'0000U, 0U));

  // imm8 = 0b11110000
  __uint128_t res15 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #-1.0e+00")();
  ASSERT_EQ(res15, MakeUInt128(0xbf80'0000U, 0U));

  // imm8 = 0b11111111
  __uint128_t res16 = ASM_INSN_WRAP_FUNC_W_RES("fmov s0, #-1.93750e+00")();
  ASSERT_EQ(res16, MakeUInt128(0xbff8'0000U, 0U));
}

TEST(Arm64InsnTest, MoveImmToFp64) {
  // The tests below verify that fmov works with various immediates.
  // Specifically, the instruction has an 8-bit immediate field consisting of
  // the following four subfields:
  //
  // - sign (one bit)
  // - upper exponent (one bit)
  // - lower exponent (two bits)
  // - mantisa (four bits)
  //
  // For example, we decompose imm8 = 0b01001111 into:
  //
  // - sign = 0 (positive)
  // - upper exponent = 1
  // - lower exponent = 00
  // - mantisa = 1111
  //
  // This immediate corresponds to 64-bit floating point value:
  //
  // 0 011111111 00 1111 000000000000000000000000000000000000000000000000
  // | |         |  |    |
  // | |         |  |    +- 48 zeros
  // | |         |  +------ mantisa
  // | |         +--------- lower exponent
  // | +------------------- upper exponent (custom extended to 9 bits)
  // +--------------------- sign
  //
  // Thus we have:
  //
  //   1.11110000... * 2^(1020-1023) = 0.2421875
  //
  // where 1.11110000... is in binary.
  //
  // See VFPExpandImm in the ARM Architecture Manual for details.
  //
  // We enumerate all possible 8-bit immediate encodings of the form:
  //
  //   {0,1}{0,1}{00,11}{0000,1111}
  //
  // to verify that the decoder correctly splits the immediate into the
  // subfields and reconstructs the intended floating-point value.

  // imm8 = 0b00000000
  __uint128_t res1 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #2.0e+00")();
  ASSERT_EQ(res1, MakeUInt128(0x4000'0000'0000'0000ULL, 0U));

  // imm8 = 0b00001111
  __uint128_t res2 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #3.8750e+00")();
  ASSERT_EQ(res2, MakeUInt128(0x400f'0000'0000'0000ULL, 0U));

  // imm8 = 0b00110000
  __uint128_t res3 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #1.60e+01")();
  ASSERT_EQ(res3, MakeUInt128(0x4030'0000'0000'0000ULL, 0U));

  // imm8 = 0b00111111
  __uint128_t res4 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #3.10e+01")();
  ASSERT_EQ(res4, MakeUInt128(0x403f'0000'0000'0000ULL, 0U));

  // imm8 = 0b01000000
  __uint128_t res5 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #1.250e-01")();
  ASSERT_EQ(res5, MakeUInt128(0x3fc0'0000'0000'0000ULL, 0U));

  // imm8 = 0b01001111
  __uint128_t res6 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #2.4218750e-01")();
  ASSERT_EQ(res6, MakeUInt128(0x3fcf'0000'0000'0000ULL, 0U));

  // imm8 = 0b01110000
  __uint128_t res7 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #1.0e+00")();
  ASSERT_EQ(res7, MakeUInt128(0x3ff0'0000'0000'0000ULL, 0U));

  // imm8 = 0b01111111
  __uint128_t res8 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #1.93750e+00")();
  ASSERT_EQ(res8, MakeUInt128(0x3fff'0000'0000'0000ULL, 0U));

  // imm8 = 0b10000000
  __uint128_t res9 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #-2.0e+00")();
  ASSERT_EQ(res9, MakeUInt128(0xc000'0000'0000'0000ULL, 0U));

  // imm8 = 0b10001111
  __uint128_t res10 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #-3.8750e+00")();
  ASSERT_EQ(res10, MakeUInt128(0xc00f'0000'0000'0000ULL, 0U));

  // imm8 = 0b10110000
  __uint128_t res11 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #-1.60e+01")();
  ASSERT_EQ(res11, MakeUInt128(0xc030'0000'0000'0000ULL, 0U));

  // imm8 = 0b10111111
  __uint128_t res12 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #-3.10e+01")();
  ASSERT_EQ(res12, MakeUInt128(0xc03f'0000'0000'0000ULL, 0U));

  // imm8 = 0b11000000
  __uint128_t res13 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #-1.250e-01")();
  ASSERT_EQ(res13, MakeUInt128(0xbfc0'0000'0000'0000ULL, 0U));

  // imm8 = 0b11001111
  __uint128_t res14 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #-2.4218750e-01")();
  ASSERT_EQ(res14, MakeUInt128(0xbfcf'0000'0000'0000ULL, 0U));

  // imm8 = 0b11110000
  __uint128_t res15 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #-1.0e+00")();
  ASSERT_EQ(res15, MakeUInt128(0xbff0'0000'0000'0000ULL, 0U));

  // imm8 = 0b11111111
  __uint128_t res16 = ASM_INSN_WRAP_FUNC_W_RES("fmov %d0, #-1.93750e+00")();
  ASSERT_EQ(res16, MakeUInt128(0xbfff'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, MoveImmToF32x4) {
  // The tests below verify that fmov works with various immediates.
  // Specifically, the instruction has an 8-bit immediate field consisting of
  // the following four subfields:
  //
  // - sign (one bit)
  // - upper exponent (one bit)
  // - lower exponent (two bits)
  // - mantisa (four bits)
  //
  // We enumerate all possible 8-bit immediate encodings of the form:
  //
  //   {0,1}{0,1}{00,11}{0000,1111}
  //
  // to verify that the decoder correctly splits the immediate into the
  // subfields and reconstructs the intended floating-point value.

  // imm8 = 0b00000000
  __uint128_t res1 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #2.0e+00")();
  ASSERT_EQ(res1, MakeUInt128(0x4000'0000'4000'0000ULL, 0x4000'0000'4000'0000ULL));

  // imm8 = 0b00001111
  __uint128_t res2 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #3.8750e+00")();
  ASSERT_EQ(res2, MakeUInt128(0x4078'0000'4078'0000ULL, 0x4078'0000'4078'0000ULL));

  // imm8 = 0b00110000
  __uint128_t res3 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #1.60e+01")();
  ASSERT_EQ(res3, MakeUInt128(0x4180'0000'4180'0000ULL, 0x4180'0000'4180'0000ULL));

  // imm8 = 0b00111111
  __uint128_t res4 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #3.10e+01")();
  ASSERT_EQ(res4, MakeUInt128(0x41f8'0000'41f8'0000ULL, 0x41f8'0000'41f8'0000ULL));

  // imm8 = 0b01000000
  __uint128_t res5 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #1.250e-01")();
  ASSERT_EQ(res5, MakeUInt128(0x3e00'0000'3e00'0000ULL, 0x3e00'0000'3e00'0000ULL));

  // imm8 = 0b01001111
  __uint128_t res6 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #2.4218750e-01")();
  ASSERT_EQ(res6, MakeUInt128(0x3e78'0000'3e78'0000ULL, 0x3e78'0000'3e78'0000ULL));

  // imm8 = 0b01110000
  __uint128_t res7 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #1.0e+00")();
  ASSERT_EQ(res7, MakeUInt128(0x3f80'0000'3f80'0000ULL, 0x3f80'0000'3f80'0000ULL));

  // imm8 = 0b01111111
  __uint128_t res8 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #1.93750e+00")();
  ASSERT_EQ(res8, MakeUInt128(0x3ff8'0000'3ff8'0000ULL, 0x3ff8'0000'3ff8'0000ULL));

  // imm8 = 0b10000000
  __uint128_t res9 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #-2.0e+00")();
  ASSERT_EQ(res9, MakeUInt128(0xc000'0000'c000'0000ULL, 0xc000'0000'c000'0000ULL));

  // imm8 = 0b10001111
  __uint128_t res10 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #-3.8750e+00")();
  ASSERT_EQ(res10, MakeUInt128(0xc078'0000'c078'0000ULL, 0xc078'0000'c078'0000ULL));

  // imm8 = 0b10110000
  __uint128_t res11 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #-1.60e+01")();
  ASSERT_EQ(res11, MakeUInt128(0xc180'0000'c180'0000ULL, 0xc180'0000'c180'0000ULL));

  // imm8 = 0b10111111
  __uint128_t res12 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #-3.10e+01")();
  ASSERT_EQ(res12, MakeUInt128(0xc1f8'0000'c1f8'0000ULL, 0xc1f8'0000'c1f8'0000ULL));

  // imm8 = 0b11000000
  __uint128_t res13 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #-1.250e-01")();
  ASSERT_EQ(res13, MakeUInt128(0xbe00'0000'be00'0000ULL, 0xbe00'0000'be00'0000ULL));

  // imm8 = 0b11001111
  __uint128_t res14 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #-2.4218750e-01")();
  ASSERT_EQ(res14, MakeUInt128(0xbe78'0000'be78'0000ULL, 0xbe78'0000'be78'0000ULL));

  // imm8 = 0b11110000
  __uint128_t res15 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #-1.0e+00")();
  ASSERT_EQ(res15, MakeUInt128(0xbf80'0000'bf80'0000ULL, 0xbf80'0000'bf80'0000ULL));

  // imm8 = 0b11111111
  __uint128_t res16 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.4s, #-1.93750e+00")();
  ASSERT_EQ(res16, MakeUInt128(0xbff8'0000'bff8'0000ULL, 0xbff8'0000'bff8'0000ULL));
}

TEST(Arm64InsnTest, MoveImmToF64x2) {
  // The tests below verify that fmov works with various immediates.
  // Specifically, the instruction has an 8-bit immediate field consisting of
  // the following four subfields:
  //
  // - sign (one bit)
  // - upper exponent (one bit)
  // - lower exponent (two bits)
  // - mantisa (four bits)
  //
  // We enumerate all possible 8-bit immediate encodings of the form:
  //
  //   {0,1}{0,1}{00,11}{0000,1111}
  //
  // to verify that the decoder correctly splits the immediate into the
  // subfields and reconstructs the intended floating-point value.

  // imm8 = 0b00000000
  __uint128_t res1 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #2.0e+00")();
  ASSERT_EQ(res1, MakeUInt128(0x4000'0000'0000'0000ULL, 0x4000'0000'0000'0000ULL));

  // imm8 = 0b00001111
  __uint128_t res2 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #3.8750e+00")();
  ASSERT_EQ(res2, MakeUInt128(0x400f'0000'0000'0000ULL, 0x400f'0000'0000'0000ULL));

  // imm8 = 0b00110000
  __uint128_t res3 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #1.60e+01")();
  ASSERT_EQ(res3, MakeUInt128(0x4030'0000'0000'0000ULL, 0x4030'0000'0000'0000ULL));

  // imm8 = 0b00111111
  __uint128_t res4 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #3.10e+01")();
  ASSERT_EQ(res4, MakeUInt128(0x403f'0000'0000'0000ULL, 0x403f'0000'0000'0000ULL));

  // imm8 = 0b01000000
  __uint128_t res5 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #1.250e-01")();
  ASSERT_EQ(res5, MakeUInt128(0x3fc0'0000'0000'0000ULL, 0x3fc0'0000'0000'0000ULL));

  // imm8 = 0b01001111
  __uint128_t res6 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #2.4218750e-01")();
  ASSERT_EQ(res6, MakeUInt128(0x3fcf'0000'0000'0000ULL, 0x3fcf'0000'0000'0000ULL));

  // imm8 = 0b01110000
  __uint128_t res7 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #1.0e+00")();
  ASSERT_EQ(res7, MakeUInt128(0x3ff0'0000'0000'0000ULL, 0x3ff0'0000'0000'0000ULL));

  // imm8 = 0b01111111
  __uint128_t res8 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #1.93750e+00")();
  ASSERT_EQ(res8, MakeUInt128(0x3fff'0000'0000'0000ULL, 0x3fff'0000'0000'0000ULL));

  // imm8 = 0b10000000
  __uint128_t res9 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #-2.0e+00")();
  ASSERT_EQ(res9, MakeUInt128(0xc000'0000'0000'0000ULL, 0xc000'0000'0000'0000ULL));

  // imm8 = 0b10001111
  __uint128_t res10 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #-3.8750e+00")();
  ASSERT_EQ(res10, MakeUInt128(0xc00f'0000'0000'0000ULL, 0xc00f'0000'0000'0000ULL));

  // imm8 = 0b10110000
  __uint128_t res11 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #-1.60e+01")();
  ASSERT_EQ(res11, MakeUInt128(0xc030'0000'0000'0000ULL, 0xc030'0000'0000'0000ULL));

  // imm8 = 0b10111111
  __uint128_t res12 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #-3.10e+01")();
  ASSERT_EQ(res12, MakeUInt128(0xc03f'0000'0000'0000ULL, 0xc03f'0000'0000'0000ULL));

  // imm8 = 0b11000000
  __uint128_t res13 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #-1.250e-01")();
  ASSERT_EQ(res13, MakeUInt128(0xbfc0'0000'0000'0000ULL, 0xbfc0'0000'0000'0000ULL));

  // imm8 = 0b11001111
  __uint128_t res14 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #-2.4218750e-01")();
  ASSERT_EQ(res14, MakeUInt128(0xbfcf'0000'0000'0000ULL, 0xbfcf'0000'0000'0000ULL));

  // imm8 = 0b11110000
  __uint128_t res15 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #-1.0e+00")();
  ASSERT_EQ(res15, MakeUInt128(0xbff0'0000'0000'0000ULL, 0xbff0'0000'0000'0000ULL));

  // imm8 = 0b11111111
  __uint128_t res16 = ASM_INSN_WRAP_FUNC_W_RES("fmov %0.2d, #-1.93750e+00")();
  ASSERT_EQ(res16, MakeUInt128(0xbfff'0000'0000'0000ULL, 0xbfff'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MoveFpRegToReg) {
  __uint128_t arg = MakeUInt128(0x1111'aaaa'2222'bbbbULL, 0x3333'cccc'4444'ddddULL);
  uint64_t res = 0xffff'eeee'dddd'ccccULL;

  // Move from high double.
  asm("fmov %0, %1.d[1]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0x3333'cccc'4444'ddddULL);

  // Move from low double.
  asm("fmov %0, %d1" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0x1111'aaaa'2222'bbbbULL);

  // Move from single.
  asm("fmov %w0, %s1" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0x2222'bbbbULL);
}

TEST(Arm64InsnTest, MoveRegToFpReg) {
  uint64_t arg = 0xffff'eeee'dddd'ccccULL;
  __uint128_t res = MakeUInt128(0x1111'aaaa'2222'bbbbULL, 0x3333'cccc'4444'ddddULL);

  // Move to high double.
  asm("fmov %0.d[1], %1" : "=w"(res) : "r"(arg), "0"(res));
  ASSERT_EQ(res, MakeUInt128(0x1111'aaaa'2222'bbbbULL, 0xffff'eeee'dddd'ccccULL));

  // Move to low double.
  asm("fmov %d0, %1" : "=w"(res) : "r"(arg));
  ASSERT_EQ(res, MakeUInt128(0xffff'eeee'dddd'ccccULL, 0x0));

  // Move to single.
  asm("fmov %s0, %w1" : "=w"(res) : "r"(arg));
  ASSERT_EQ(res, MakeUInt128(0xdddd'ccccULL, 0x0));
}

TEST(Arm64InsnTest, MoveFpRegToFpReg) {
  __uint128_t res;

  __uint128_t fp64_arg =
      MakeUInt128(0x402e'9eb8'51eb'851fULL, 0xdead'beef'aabb'ccddULL);  // 15.31 in double
  asm("fmov %d0, %d1" : "=w"(res) : "w"(fp64_arg));
  ASSERT_EQ(res, MakeUInt128(0x402e'9eb8'51eb'851fULL, 0ULL));

  __uint128_t fp32_arg =
      MakeUInt128(0xaabb'ccdd'40e5'1eb8ULL, 0x0011'2233'4455'6677ULL);  // 7.16 in float
  asm("fmov %s0, %s1" : "=w"(res) : "w"(fp32_arg));
  ASSERT_EQ(res, MakeUInt128(0x40e5'1eb8ULL, 0ULL));
}

TEST(Arm64InsnTest, InsertRegPartIntoSimd128) {
  uint64_t arg = 0xffff'eeee'dddd'ccccULL;
  __uint128_t res = MakeUInt128(0x1111'aaaa'2222'bbbbULL, 0x3333'cccc'4444'ddddULL);

  // Byte.
  asm("mov %0.b[3], %w1" : "=w"(res) : "r"(arg), "0"(res));
  ASSERT_EQ(res, MakeUInt128(0x1111'aaaa'cc22'bbbbULL, 0x3333'cccc'4444'ddddULL));

  // Double word.
  asm("mov %0.d[1], %1" : "=w"(res) : "r"(arg), "0"(res));
  ASSERT_EQ(res, MakeUInt128(0x1111'aaaa'cc22'bbbbULL, 0xffff'eeee'dddd'ccccULL));
}

TEST(Arm64InsnTest, DuplicateRegIntoSimd128) {
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("dup %0.16b, %w1")(0xabU);
  ASSERT_EQ(res, MakeUInt128(0xabab'abab'abab'ababULL, 0xabab'abab'abab'ababULL));
}

TEST(Arm64InsnTest, MoveSimd128ElemToRegSigned) {
  uint64_t res = 0;
  __uint128_t arg = MakeUInt128(0x9796'9594'9392'9190ULL, 0x9f'9e9d'9c9b'9a99ULL);

  // Single word.
  asm("smov %0, %1.s[0]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xffff'ffff'9392'9190ULL);

  asm("smov %0, %1.s[2]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xffff'ffff'9c9b'9a99ULL);

  // Half word.
  asm("smov %w0, %1.h[0]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0x0000'0000'ffff'9190ULL);

  asm("smov %w0, %1.h[2]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0x0000'0000'ffff'9594ULL);

  // Byte.
  asm("smov %w0, %1.b[0]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0x0000'0000'ffff'ff90ULL);

  asm("smov %w0, %1.b[2]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0x0000'0000'ffff'ff92ULL);
}

TEST(Arm64InsnTest, MoveSimd128ElemToRegUnsigned) {
  uint64_t res = 0;
  __uint128_t arg = MakeUInt128(0xaaaa'bbbb'cccc'eeeeULL, 0xffff'0000'1111'2222ULL);

  // Double word.
  asm("umov %0, %1.d[0]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xaaaa'bbbb'cccc'eeeeULL);

  asm("umov %0, %1.d[1]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xffff'0000'1111'2222ULL);

  // Single word.
  asm("umov %w0, %1.s[0]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xcccc'eeeeULL);

  asm("umov %w0, %1.s[2]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0x1111'2222ULL);

  // Half word.
  asm("umov %w0, %1.h[0]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xeeeeULL);

  asm("umov %w0, %1.h[2]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xbbbbULL);

  // Byte.
  asm("umov %w0, %1.b[0]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xeeULL);

  asm("umov %w0, %1.b[2]" : "=r"(res) : "w"(arg));
  ASSERT_EQ(res, 0xccULL);
}

TEST(Arm64InsnTest, SignedMultiplyAddLongElemI16x4) {
  __uint128_t arg1 = MakeUInt128(0x9463'2295'6398'9898ULL, 0x9358'2116'7456'2701ULL);
  __uint128_t arg2 = MakeUInt128(0x0218'3564'6220'1349ULL, 0x6715'1881'9097'3038ULL);
  __uint128_t arg3 = MakeUInt128(0x1198'0049'7340'7239ULL, 0x6103'6854'0664'3193ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlal %0.4s, %1.4h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x37c4'a349'4b9d'b539ULL, 0x37c3'dab4'13a5'8e33ULL));
}

TEST(Arm64InsnTest, SignedMultiplyAddLongElemI16x4Upper) {
  __uint128_t arg1 = MakeUInt128(0x9478'2218'1852'8624ULL, 0x0851'4006'6604'4332ULL);
  __uint128_t arg2 = MakeUInt128(0x5888'5698'6705'4315ULL, 0x4706'9657'4745'8550ULL);
  __uint128_t arg3 = MakeUInt128(0x3323'2334'2107'3015ULL, 0x4594'0516'5537'9068ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlal2 %0.4s, %1.8h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x5c30'bd48'3c11'9e0fULL, 0x48ec'c5ab'6efb'3a86ULL));
}

TEST(Arm64InsnTest, SignedMultiplyAddLongElemI16x4Upper2) {
  __uint128_t arg1 = MakeUInt128(0x9968'2628'2472'7064ULL, 0x1336'2221'7892'3903ULL);
  __uint128_t arg2 = MakeUInt128(0x1760'8542'8943'7339ULL, 0x3561'8891'6512'5042ULL);
  __uint128_t arg3 = MakeUInt128(0x4404'0089'5271'9837ULL, 0x8738'6480'5847'2689ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlal2 %0.4s, %1.8h, %2.h[7]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x5d27'e9db'5e54'd15aULL, 0x8b39'd9f6'5f64'ea0aULL));
}

TEST(Arm64InsnTest, SignedMultiplySubtractLongElemI16x4) {
  __uint128_t arg1 = MakeUInt128(0x9143'4478'8636'0410ULL, 0x3182'3507'3650'2778ULL);
  __uint128_t arg2 = MakeUInt128(0x5908'9757'8272'7313ULL, 0x0504'8893'9890'0992ULL);
  __uint128_t arg3 = MakeUInt128(0x3913'5033'7325'0855ULL, 0x9826'5586'7089'2426ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlsl %0.4s, %1.4h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0xfd58'2027'7523'1935ULL, 0x61d6'9fb0'921d'b6b6ULL));
}

TEST(Arm64InsnTest, SignedMultiplySubtractLongElemI16x4Upper) {
  __uint128_t arg1 = MakeUInt128(0x9320'1991'9968'8285ULL, 0x1718'3953'6691'3452ULL);
  __uint128_t arg2 = MakeUInt128(0x2244'4708'0459'2396ULL, 0x6028'1715'6551'5656ULL);
  __uint128_t arg3 = MakeUInt128(0x6611'1359'8231'1225ULL, 0x0628'9058'5491'4509ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlsl2 %0.4s, %1.8h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x6453'26f0'814d'99a3ULL, 0x05c4'2900'5398'0b2eULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyAddLongElemI16x4) {
  __uint128_t arg1 = MakeUInt128(0x9027'6018'3484'0306ULL, 0x8113'8185'5105'9797ULL);
  __uint128_t arg2 = MakeUInt128(0x0566'4007'5094'2608ULL, 0x7885'7357'9603'7324ULL);
  __uint128_t arg3 = MakeUInt128(0x5141'4678'6703'6880ULL, 0x9880'6097'1642'5849ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlal %0.4s, %1.4h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x61c8'e2c8'67f7'07f8ULL, 0xc5df'e723'3481'6629ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyAddLongElemI16x4Upper) {
  __uint128_t arg1 = MakeUInt128(0x9454'2368'2886'0613ULL, 0x4084'1486'3776'7009ULL);
  __uint128_t arg2 = MakeUInt128(0x6120'7151'2491'4043ULL, 0x0272'5386'0764'8236ULL);
  __uint128_t arg3 = MakeUInt128(0x3414'3346'2351'8975ULL, 0x7664'5216'4137'6796ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlal2 %0.4s, %1.8h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x3c00'351c'3352'428eULL, 0x7f9b'6cda'4425'df7cULL));
}

TEST(Arm64InsnTest, UnsignedMultiplySubtractLongElemI16x4) {
  __uint128_t arg1 = MakeUInt128(0x9128'0092'8252'5619ULL, 0x0205'2630'1639'1147ULL);
  __uint128_t arg2 = MakeUInt128(0x7247'3314'8573'9107ULL, 0x7758'7442'5387'6117ULL);
  __uint128_t arg3 = MakeUInt128(0x4657'8671'1694'1477ULL, 0x6421'4411'1126'3583ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlsl %0.4s, %1.4h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x0268'619b'e9b2'6a3cULL, 0x1876'4719'10da'19edULL));
}

TEST(Arm64InsnTest, UnsignedMultiplySubtractLongElemI16x4Upper) {
  __uint128_t arg1 = MakeUInt128(0x9420'7571'3627'5167ULL, 0x4573'1891'8945'6283ULL);
  __uint128_t arg2 = MakeUInt128(0x5257'0441'3354'3758ULL, 0x5753'4269'8699'4725ULL);
  __uint128_t arg3 = MakeUInt128(0x4703'1656'6139'9199ULL, 0x9682'6282'4727'0641ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlsl2 %0.4s, %1.8h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x2b7d'4cb2'4d79'259dULL, 0x8895'afc6'423a'13adULL));
}

TEST(Arm64InsnTest, SignedMultiplySubtractLongElemI32x2) {
  __uint128_t arg1 = MakeUInt128(0x9143'4478'8636'0410ULL, 0x3182'3507'3650'2778ULL);
  __uint128_t arg2 = MakeUInt128(0x5908'9757'8272'7313ULL, 0x0504'8893'9890'0992ULL);
  __uint128_t arg3 = MakeUInt128(0x3913'5033'7325'0855ULL, 0x9826'5586'7089'2426ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlsl %0.2d, %1.2s, %2.s[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x636e'9f19'49e4'36e5ULL, 0xbea9'aa15'898a'175eULL));
}

TEST(Arm64InsnTest, SignedMultiplySubtractLongElemI32x2Upper) {
  __uint128_t arg1 = MakeUInt128(0x9320'1991'9968'8285ULL, 0x1718'3953'6691'3452ULL);
  __uint128_t arg2 = MakeUInt128(0x2244'4708'0459'2396ULL, 0x6028'1715'6551'5656ULL);
  __uint128_t arg3 = MakeUInt128(0x6611'1359'8231'1225ULL, 0x0628'9058'5491'4509ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlsl2 %0.2d, %1.4s, %2.s[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x5856'6f5f'3e5c'b195ULL, 0x311'2fe3'a3dd'7571ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplySubtractLongElemI32x2) {
  __uint128_t arg1 = MakeUInt128(0x9128'0092'8252'5619ULL, 0x0205'2630'1639'1147ULL);
  __uint128_t arg2 = MakeUInt128(0x7247'3314'8573'9107ULL, 0x7758'7442'5387'6117ULL);
  __uint128_t arg3 = MakeUInt128(0x4657'8671'1694'1477ULL, 0x6421'4411'1126'3583ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlsl %0.2d, %1.2s, %2.s[0]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x267eae5395490c8ULL, 0x1875f49155257f85ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplySubtractLongElemI32x2Upper) {
  __uint128_t arg1 = MakeUInt128(0x9420'7571'3627'5167ULL, 0x4573'1891'8945'6283ULL);
  __uint128_t arg2 = MakeUInt128(0x5257'0441'3354'3758ULL, 0x5753'4269'8699'4725ULL);
  __uint128_t arg3 = MakeUInt128(0x4703'1656'6139'9199ULL, 0x9682'6282'4727'0641ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlsl2 %0.2d, %1.4s, %2.s[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x1adc'33ec'008c'8256ULL, 0x802b'e95f'0d44'8570ULL));
}

TEST(Arm64InsnTest, AsmConvertI32F32) {
  constexpr auto AsmConvertI32F32 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("scvtf %s0, %w1");
  ASSERT_EQ(AsmConvertI32F32(21), MakeUInt128(0x41a8'0000U, 0U));
}

TEST(Arm64InsnTest, AsmConvertU32F32) {
  constexpr auto AsmConvertU32F32 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("ucvtf %s0, %w1");

  ASSERT_EQ(AsmConvertU32F32(29), MakeUInt128(0x41e8'0000U, 0U));

  // Test that the topmost bit isn't treated as the sign.
  ASSERT_EQ(AsmConvertU32F32(1U << 31), MakeUInt128(0x4f00'0000U, 0U));
}

TEST(Arm64InsnTest, AsmConvertU32F32FromSimdReg) {
  constexpr auto AsmUcvtf = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %s0, %s1");

  ASSERT_EQ(AsmUcvtf(28), MakeUInt128(0x41e0'0000U, 0U));

  // Test that the topmost bit isn't treated as the sign.
  ASSERT_EQ(AsmUcvtf(1U << 31), MakeUInt128(0x4f00'0000U, 0U));
}

TEST(Arm64InsnTest, AsmConvertI32F64) {
  constexpr auto AsmConvertI32F64 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("scvtf %d0, %w1");
  ASSERT_EQ(AsmConvertI32F64(21), MakeUInt128(0x4035'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertU32F64) {
  constexpr auto AsmConvertU32F64 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("ucvtf %d0, %w1");

  ASSERT_EQ(AsmConvertU32F64(18), MakeUInt128(0x4032'0000'0000'0000ULL, 0U));

  // Test that the topmost bit isn't treated as the sign.
  ASSERT_EQ(AsmConvertU32F64(1U << 31), MakeUInt128(0x41e0'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertI64F32) {
  constexpr auto AsmConvertI64F32 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("scvtf %s0, %x1");
  ASSERT_EQ(AsmConvertI64F32(11), MakeUInt128(0x4130'0000U, 0U));
}

TEST(Arm64InsnTest, AsmConvertU64F32) {
  constexpr auto AsmConvertU64F32 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("ucvtf %s0, %x1");

  ASSERT_EQ(AsmConvertU64F32(3), MakeUInt128(0x4040'0000U, 0U));

  // Test that the topmost bit isn't treated as the sign.
  ASSERT_EQ(AsmConvertU64F32(1ULL << 63), MakeUInt128(0x5f00'0000U, 0U));
}

TEST(Arm64InsnTest, AsmConvertI64F64) {
  constexpr auto AsmConvertI64F64 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("scvtf %d0, %x1");
  ASSERT_EQ(AsmConvertI64F64(137), MakeUInt128(0x4061'2000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertI32F32FromSimdReg) {
  constexpr auto AsmConvertI32F32 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("scvtf %s0, %s1");
  ASSERT_EQ(AsmConvertI32F32(1109), MakeUInt128(0x448a'a000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertI64F64FromSimdReg) {
  constexpr auto AsmConvertI64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("scvtf %d0, %d1");
  ASSERT_EQ(AsmConvertI64F64(123), MakeUInt128(0x405e'c000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertI32x4F32x4) {
  constexpr auto AsmConvertI32F32 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("scvtf %0.4s, %1.4s");
  __uint128_t arg = MakeUInt128(0x0000'0035'0000'0014ULL, 0x0000'0054'0000'0009ULL);
  ASSERT_EQ(AsmConvertI32F32(arg), MakeUInt128(0x4254'0000'41a0'0000ULL, 0x42a8'0000'4110'0000ULL));
}

TEST(Arm64InsnTest, AsmConvertI64x2F64x2) {
  constexpr auto AsmConvertI64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("scvtf %0.2d, %1.2d");
  __uint128_t arg = MakeUInt128(static_cast<int64_t>(-9), 17U);
  ASSERT_EQ(AsmConvertI64F64(arg), MakeUInt128(0xc022'0000'0000'0000ULL, 0x4031'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AsmConvertU32x4F32x4) {
  constexpr auto AsmConvertU32F32 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %0.4s, %1.4s");
  __uint128_t arg = MakeUInt128(0x8000'0000'0000'0019ULL, 0x0000'0058'0000'0010ULL);
  ASSERT_EQ(AsmConvertU32F32(arg), MakeUInt128(0x4f00'0000'41c8'0000ULL, 0x42b0'0000'4180'0000ULL));
}

TEST(Arm64InsnTest, AsmConvertU64x2F64x2) {
  constexpr auto AsmConvertU64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %0.2d, %1.2d");
  __uint128_t arg = MakeUInt128(1ULL << 63, 29U);
  ASSERT_EQ(AsmConvertU64F64(arg), MakeUInt128(0x43e0'0000'0000'0000ULL, 0x403d'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AsmConvertU64F64) {
  constexpr auto AsmConvertU64F64 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("ucvtf %d0, %x1");

  ASSERT_EQ(AsmConvertU64F64(49), MakeUInt128(0x4048'8000'0000'0000ULL, 0U));

  // Test that the topmost bit isn't treated as the sign.
  ASSERT_EQ(AsmConvertU64F64(1ULL << 63), MakeUInt128(0x43e0'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertU64F64FromSimdReg) {
  constexpr auto AsmUcvtf = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %d0, %d1");

  ASSERT_EQ(AsmUcvtf(47), MakeUInt128(0x4047'8000'0000'0000ULL, 0U));

  // Test that the topmost bit isn't treated as the sign.
  ASSERT_EQ(AsmUcvtf(1ULL << 63), MakeUInt128(0x43e0'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertLiterals) {
  // Verify that the compiler encodes the floating-point literals used in the
  // conversion tests below exactly as expected.
  ASSERT_EQ(bit_cast<uint32_t>(-7.50f), 0xc0f0'0000U);
  ASSERT_EQ(bit_cast<uint32_t>(-6.75f), 0xc0d8'0000U);
  ASSERT_EQ(bit_cast<uint32_t>(-6.50f), 0xc0d0'0000U);
  ASSERT_EQ(bit_cast<uint32_t>(-6.25f), 0xc0c8'0000U);
  ASSERT_EQ(bit_cast<uint32_t>(6.25f), 0x40c8'0000U);
  ASSERT_EQ(bit_cast<uint32_t>(6.50f), 0x40d0'0000U);
  ASSERT_EQ(bit_cast<uint32_t>(6.75f), 0x40d8'0000U);
  ASSERT_EQ(bit_cast<uint32_t>(7.50f), 0x40f0'0000U);

  ASSERT_EQ(bit_cast<uint64_t>(-7.50), 0xc01e'0000'0000'0000ULL);
  ASSERT_EQ(bit_cast<uint64_t>(-6.75), 0xc01b'0000'0000'0000ULL);
  ASSERT_EQ(bit_cast<uint64_t>(-6.50), 0xc01a'0000'0000'0000ULL);
  ASSERT_EQ(bit_cast<uint64_t>(-6.25), 0xc019'0000'0000'0000ULL);
  ASSERT_EQ(bit_cast<uint64_t>(6.25), 0x4019'0000'0000'0000ULL);
  ASSERT_EQ(bit_cast<uint64_t>(6.50), 0x401a'0000'0000'0000ULL);
  ASSERT_EQ(bit_cast<uint64_t>(6.75), 0x401b'0000'0000'0000ULL);
  ASSERT_EQ(bit_cast<uint64_t>(7.50), 0x401e'0000'0000'0000ULL);
}

template <typename IntType, typename FuncType>
void TestConvertF32ToInt(FuncType AsmFunc, std::initializer_list<int> expected) {
  // Note that bit_cast isn't a constexpr.
  static const uint32_t kConvertF32ToIntInputs[] = {
      bit_cast<uint32_t>(-7.50f),
      bit_cast<uint32_t>(-6.75f),
      bit_cast<uint32_t>(-6.50f),
      bit_cast<uint32_t>(-6.25f),
      bit_cast<uint32_t>(6.25f),
      bit_cast<uint32_t>(6.50f),
      bit_cast<uint32_t>(6.75f),
      bit_cast<uint32_t>(7.50f),
  };

  const size_t kConvertF32ToIntInputsSize = sizeof(kConvertF32ToIntInputs) / sizeof(uint32_t);
  ASSERT_EQ(kConvertF32ToIntInputsSize, expected.size());

  auto expected_it = expected.begin();
  for (size_t input_it = 0; input_it < kConvertF32ToIntInputsSize; input_it++) {
    ASSERT_EQ(AsmFunc(kConvertF32ToIntInputs[input_it]), static_cast<IntType>(*expected_it++));
  }
}

TEST(Arm64InsnTest, AsmConvertF32I32TieAway) {
  constexpr auto AsmFcvtas = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtas %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtas, {-8, -7, -7, -6, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U32TieAway) {
  constexpr auto AsmFcvtau = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtau %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtau, {0U, 0U, 0U, 0U, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I32NegInf) {
  constexpr auto AsmFcvtms = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtms %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtms, {-8, -7, -7, -7, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32U32NegInf) {
  constexpr auto AsmFcvtmu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtmu %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtmu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32I32TieEven) {
  constexpr auto AsmFcvtns = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtns %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtns, {-8, -7, -6, -6, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U32TieEven) {
  constexpr auto AsmFcvtnu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtnu %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtnu, {0U, 0U, 0U, 0U, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I32PosInf) {
  constexpr auto AsmFcvtps = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtps %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtps, {-7, -6, -6, -6, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U32PosInf) {
  constexpr auto AsmFcvtpu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtpu %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtpu, {0U, 0U, 0U, 0U, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I32Truncate) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtzs, {-7, -6, -6, -6, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32U32Truncate) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzu %w0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtzu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32I64TieAway) {
  constexpr auto AsmFcvtas = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtas %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtas, {-8, -7, -7, -6, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U64TieAway) {
  constexpr auto AsmFcvtau = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtau %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtau, {0U, 0U, 0U, 0U, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I64NegInf) {
  constexpr auto AsmFcvtms = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtms %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtms, {-8, -7, -7, -7, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32U64NegInf) {
  constexpr auto AsmFcvtmu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtmu %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtmu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32I64TieEven) {
  constexpr auto AsmFcvtns = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtns %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtns, {-8, -7, -6, -6, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U64TieEven) {
  constexpr auto AsmFcvtnu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtnu %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtnu, {0U, 0U, 0U, 0U, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I64PosInf) {
  constexpr auto AsmFcvtps = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtps %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtps, {-7, -6, -6, -6, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U64PosInf) {
  constexpr auto AsmFcvtpu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtpu %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtpu, {0U, 0U, 0U, 0U, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I64Truncate) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtzs, {-7, -6, -6, -6, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32U64Truncate) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzu %x0, %s1");
  TestConvertF32ToInt<uint64_t>(AsmFcvtzu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

template <typename IntType, typename FuncType>
void TestConvertF64ToInt(FuncType AsmFunc, std::initializer_list<int> expected) {
  // Note that bit_cast isn't a constexpr.
  static const uint64_t kConvertF64ToIntInputs[] = {
      bit_cast<uint64_t>(-7.50),
      bit_cast<uint64_t>(-6.75),
      bit_cast<uint64_t>(-6.50),
      bit_cast<uint64_t>(-6.25),
      bit_cast<uint64_t>(6.25),
      bit_cast<uint64_t>(6.50),
      bit_cast<uint64_t>(6.75),
      bit_cast<uint64_t>(7.50),
  };

  const size_t kConvertF64ToIntInputsSize = sizeof(kConvertF64ToIntInputs) / sizeof(uint64_t);
  ASSERT_EQ(kConvertF64ToIntInputsSize, expected.size());

  auto expected_it = expected.begin();
  for (size_t input_it = 0; input_it < kConvertF64ToIntInputsSize; input_it++) {
    ASSERT_EQ(AsmFunc(kConvertF64ToIntInputs[input_it]), static_cast<IntType>(*expected_it++));
  }
}

TEST(Arm64InsnTest, AsmConvertF64I32TieAway) {
  constexpr auto AsmFcvtas = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtas %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtas, {-8, -7, -7, -6, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U32TieAway) {
  constexpr auto AsmFcvtau = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtau %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtau, {0U, 0U, 0U, 0U, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I32NegInf) {
  constexpr auto AsmFcvtms = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtms %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtms, {-8, -7, -7, -7, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64U32NegInf) {
  constexpr auto AsmFcvtmu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtmu %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtmu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64I32TieEven) {
  constexpr auto AsmFcvtns = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtns %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtns, {-8, -7, -6, -6, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U32TieEven) {
  constexpr auto AsmFcvtnu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtnu %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtnu, {0U, 0U, 0U, 0U, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I32PosInf) {
  constexpr auto AsmFcvtps = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtps %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtps, {-7, -6, -6, -6, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U32PosInf) {
  constexpr auto AsmFcvtpu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtpu %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtpu, {0U, 0U, 0U, 0U, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I32Truncate) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtzs, {-7, -6, -6, -6, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64U32Truncate) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzu %w0, %d1");
  TestConvertF64ToInt<uint32_t>(AsmFcvtzu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64I64TieAway) {
  constexpr auto AsmFcvtas = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtas %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtas, {-8, -7, -7, -6, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U64TieAway) {
  constexpr auto AsmFcvtau = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtau %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtau, {0U, 0U, 0U, 0U, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I64NegInf) {
  constexpr auto AsmFcvtms = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtms %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtms, {-8, -7, -7, -7, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64U64NegInf) {
  constexpr auto AsmFcvtmu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtmu %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtmu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64I64TieEven) {
  constexpr auto AsmFcvtns = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtns %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtns, {-8, -7, -6, -6, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U64TieEven) {
  constexpr auto AsmFcvtnu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtnu %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtnu, {0U, 0U, 0U, 0U, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I64PosInf) {
  constexpr auto AsmFcvtps = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtps %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtps, {-7, -6, -6, -6, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U64PosInf) {
  constexpr auto AsmFcvtpu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtpu %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtpu, {0U, 0U, 0U, 0U, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I64Truncate) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtzs, {-7, -6, -6, -6, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64U64Truncate) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzu %x0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtzu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32I32ScalarTieAway) {
  constexpr auto AsmFcvtas = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtas %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtas, {-8, -7, -7, -6, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U32ScalarTieAway) {
  constexpr auto AsmFcvtau = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtau %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtau, {0U, 0U, 0U, 0U, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I32ScalarNegInf) {
  constexpr auto AsmFcvtms = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtms %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtms, {-8, -7, -7, -7, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32U32ScalarNegInf) {
  constexpr auto AsmFcvtmu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtmu %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtmu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32I32ScalarTieEven) {
  constexpr auto AsmFcvtns = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtns %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtns, {-8, -7, -6, -6, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U32ScalarTieEven) {
  constexpr auto AsmFcvtnu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtnu %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtnu, {0U, 0U, 0U, 0U, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I32ScalarPosInf) {
  constexpr auto AsmFcvtps = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtps %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtps, {-7, -6, -6, -6, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32U32ScalarPosInf) {
  constexpr auto AsmFcvtpu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtpu %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtpu, {0U, 0U, 0U, 0U, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF32I32ScalarTruncate) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzs %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtzs, {-7, -6, -6, -6, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32U32ScalarTruncate) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzu %s0, %s1");
  TestConvertF32ToInt<uint32_t>(AsmFcvtzu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64I64ScalarTieAway) {
  constexpr auto AsmFcvtas = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtas %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtas, {-8, -7, -7, -6, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U64ScalarTieAway) {
  constexpr auto AsmFcvtau = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtau %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtau, {0U, 0U, 0U, 0U, 6U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I64ScalarNegInf) {
  constexpr auto AsmFcvtms = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtms %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtms, {-8, -7, -7, -7, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64U64ScalarNegInf) {
  constexpr auto AsmFcvtmu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtmu %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtmu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64I64ScalarTieEven) {
  constexpr auto AsmFcvtns = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtns %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtns, {-8, -7, -6, -6, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U64ScalarTieEven) {
  constexpr auto AsmFcvtnu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtnu %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtnu, {0U, 0U, 0U, 0U, 6U, 6U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I64ScalarPosInf) {
  constexpr auto AsmFcvtps = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtps %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtps, {-7, -6, -6, -6, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64U64ScalarPosInf) {
  constexpr auto AsmFcvtpu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtpu %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtpu, {0U, 0U, 0U, 0U, 7U, 7U, 7U, 8U});
}

TEST(Arm64InsnTest, AsmConvertF64I64ScalarTruncate) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzs %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtzs, {-7, -6, -6, -6, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF64U64ScalarTruncate) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzu %d0, %d1");
  TestConvertF64ToInt<uint64_t>(AsmFcvtzu, {0U, 0U, 0U, 0U, 6U, 6U, 6U, 7U});
}

TEST(Arm64InsnTest, AsmConvertF32I32x4TieAway) {
  constexpr auto AsmFcvtas = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtas %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtas(arg1), MakeUInt128(0xffff'fff9'ffff'fff8ULL, 0xffff'fffa'ffff'fff9ULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtas(arg2), MakeUInt128(0x0000'0007'0000'0006ULL, 0x0000'0008'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF32U32x4TieAway) {
  constexpr auto AsmFcvtau = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtau %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtau(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtau(arg2), MakeUInt128(0x0000'0007'0000'0006ULL, 0x0000'0008'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF32I32x4NegInf) {
  constexpr auto AsmFcvtms = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtms %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtms(arg1), MakeUInt128(0xffff'fff9'ffff'fff8ULL, 0xffff'fff9'ffff'fff9ULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtms(arg2), MakeUInt128(0x0000'0006'0000'0006ULL, 0x0000'0007'0000'0006ULL));
}

TEST(Arm64InsnTest, AsmConvertF32U32x4NegInf) {
  constexpr auto AsmFcvtmu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtmu %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtmu(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtmu(arg2), MakeUInt128(0x0000'0006'0000'0006ULL, 0x0000'0007'0000'0006ULL));
}

TEST(Arm64InsnTest, AsmConvertF32I32x4TieEven) {
  constexpr auto AsmFcvtns = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtns %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtns(arg1), MakeUInt128(0xffff'fff9'ffff'fff8ULL, 0xffff'fffa'ffff'fffaULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtns(arg2), MakeUInt128(0x0000'0006'0000'0006ULL, 0x0000'0008'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF32U32x4TieEven) {
  constexpr auto AsmFcvtnu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtnu %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtnu(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtnu(arg2), MakeUInt128(0x0000'0006'0000'0006ULL, 0x0000'0008'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF32I32x4PosInf) {
  constexpr auto AsmFcvtps = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtps %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtps(arg1), MakeUInt128(0xffff'fffa'ffff'fff9ULL, 0xffff'fffa'ffff'fffaULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtps(arg2), MakeUInt128(0x0000'0007'0000'0007ULL, 0x0000'0008'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF32U32x4PosInf) {
  constexpr auto AsmFcvtpu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtpu %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtpu(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtpu(arg2), MakeUInt128(0x0000'0007'0000'0007ULL, 0x0000'0008'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF32I32x4Truncate) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzs %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtzs(arg1), MakeUInt128(0xffff'fffa'ffff'fff9ULL, 0xffff'fffa'ffff'fffaULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtzs(arg2), MakeUInt128(0x0000'0006'0000'0006ULL, 0x0000'0007'0000'0006ULL));
}

TEST(Arm64InsnTest, AsmConvertF32U32x4Truncate) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzu %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtzu(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtzu(arg2), MakeUInt128(0x0000'0006'0000'0006ULL, 0x0000'0007'0000'0006ULL));
}

TEST(Arm64InsnTest, AsmConvertF64I64x4TieAway) {
  constexpr auto AsmFcvtas = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtas %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtas(arg1), MakeUInt128(0xffff'ffff'ffff'fff8ULL, 0xffff'ffff'ffff'fff9ULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtas(arg2), MakeUInt128(0xffff'ffff'ffff'fff9ULL, 0xffff'ffff'ffff'fffaULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtas(arg3), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0007ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtas(arg4), MakeUInt128(0x0000'0000'0000'0007ULL, 0x0000'0000'0000'0008ULL));
}

TEST(Arm64InsnTest, AsmConvertF64U64x4TieAway) {
  constexpr auto AsmFcvtau = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtau %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtau(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtau(arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtau(arg3), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0007ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtau(arg4), MakeUInt128(0x0000'0000'0000'0007ULL, 0x0000'0000'0000'0008ULL));
}

TEST(Arm64InsnTest, AsmConvertF64I64x4NegInf) {
  constexpr auto AsmFcvtms = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtms %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtms(arg1), MakeUInt128(0xffff'ffff'ffff'fff8ULL, 0xffff'ffff'ffff'fff9ULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtms(arg2), MakeUInt128(0xffff'ffff'ffff'fff9ULL, 0xffff'ffff'ffff'fff9ULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtms(arg3), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0006ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtms(arg4), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF64U64x4NegInf) {
  constexpr auto AsmFcvtmu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtmu %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtmu(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtmu(arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtmu(arg3), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0006ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtmu(arg4), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF64I64x4TieEven) {
  constexpr auto AsmFcvtns = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtns %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtns(arg1), MakeUInt128(0xffff'ffff'ffff'fff8ULL, 0xffff'ffff'ffff'fff9ULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtns(arg2), MakeUInt128(0xffff'ffff'ffff'fffaULL, 0xffff'ffff'ffff'fffaULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtns(arg3), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0006ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtns(arg4), MakeUInt128(0x0000'0000'0000'0007ULL, 0x0000'0000'0000'0008ULL));
}

TEST(Arm64InsnTest, AsmConvertF64U64x4TieEven) {
  constexpr auto AsmFcvtnu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtnu %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtnu(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtnu(arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtnu(arg3), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0006ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtnu(arg4), MakeUInt128(0x0000'0000'0000'0007ULL, 0x0000'0000'0000'0008ULL));
}

TEST(Arm64InsnTest, AsmConvertF64I64x4PosInf) {
  constexpr auto AsmFcvtps = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtps %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtps(arg1), MakeUInt128(0xffff'ffff'ffff'fff9ULL, 0xffff'ffff'ffff'fffaULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtps(arg2), MakeUInt128(0xffff'ffff'ffff'fffaULL, 0xffff'ffff'ffff'fffaULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtps(arg3), MakeUInt128(0x0000'0000'0000'0007ULL, 0x0000'0000'0000'0007ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtps(arg4), MakeUInt128(0x0000'0000'0000'0007ULL, 0x0000'0000'0000'0008ULL));
}

TEST(Arm64InsnTest, AsmConvertF64U64x4PosInf) {
  constexpr auto AsmFcvtpu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtpu %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtpu(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtpu(arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtpu(arg3), MakeUInt128(0x0000'0000'0000'0007ULL, 0x0000'0000'0000'0007ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtpu(arg4), MakeUInt128(0x0000'0000'0000'0007ULL, 0x0000'0000'0000'0008ULL));
}

TEST(Arm64InsnTest, AsmConvertF64I64x4Truncate) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzs %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtzs(arg1), MakeUInt128(0xffff'ffff'ffff'fff9ULL, 0xffff'ffff'ffff'fffaULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtzs(arg2), MakeUInt128(0xffff'ffff'ffff'fffaULL, 0xffff'ffff'ffff'fffaULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtzs(arg3), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0006ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtzs(arg4), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertF64U64x4Truncate) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzu %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtzu(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtzu(arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtzu(arg3), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0006ULL));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtzu(arg4), MakeUInt128(0x0000'0000'0000'0006ULL, 0x0000'0000'0000'0007ULL));
}

TEST(Arm64InsnTest, AsmConvertX32F32Scalar) {
  constexpr auto AsmConvertX32F32 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("scvtf %s0, %w1, #7");

  ASSERT_EQ(AsmConvertX32F32(0x610), MakeUInt128(0x4142'0000ULL, 0U));

  ASSERT_EQ(AsmConvertX32F32(1U << 31), MakeUInt128(0xcb80'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertX32F64Scalar) {
  constexpr auto AsmConvertX32F64 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("scvtf %d0, %w1, #8");

  ASSERT_EQ(AsmConvertX32F64(0x487), MakeUInt128(0x4012'1c00'0000'0000ULL, 0U));

  ASSERT_EQ(AsmConvertX32F64(1 << 31), MakeUInt128(0xc160'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertX32F32) {
  constexpr auto AsmConvertX32F32 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("scvtf %s0, %s1, #7");

  ASSERT_EQ(AsmConvertX32F32(0x123), MakeUInt128(0x4011'8000ULL, 0U));

  ASSERT_EQ(AsmConvertX32F32(1U << 31), MakeUInt128(0xcb80'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertX32x4F32x4) {
  constexpr auto AsmConvertX32F32 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("scvtf %0.4s, %1.4s, #11");
  __uint128_t arg = MakeUInt128(0x8000'0000'ffff'9852ULL, 0x0000'1102'0000'1254ULL);
  ASSERT_EQ(AsmConvertX32F32(arg), MakeUInt128(0xc980'0000'c14f'5c00ULL, 0x4008'1000'4012'a000ULL));
}

TEST(Arm64InsnTest, AsmConvertUX32F32Scalar) {
  constexpr auto AsmConvertUX32F32 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("ucvtf %s0, %w1, #7");

  ASSERT_EQ(AsmConvertUX32F32(0x857), MakeUInt128(0x4185'7000ULL, 0U));

  ASSERT_EQ(AsmConvertUX32F32(1U << 31), MakeUInt128(0x4b80'0000ULL, 0U));

  // Test the default rounding behavior (FPRounding_TIEEVEN).
  ASSERT_EQ(AsmConvertUX32F32(0x8000'0080), MakeUInt128(0x4b80'0000ULL, 0U));
  ASSERT_EQ(AsmConvertUX32F32(0x8000'00c0), MakeUInt128(0x4b80'0001ULL, 0U));
  ASSERT_EQ(AsmConvertUX32F32(0x8000'0140), MakeUInt128(0x4b80'0001ULL, 0U));
  ASSERT_EQ(AsmConvertUX32F32(0x8000'0180), MakeUInt128(0x4b80'0002ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertUX32F64Scalar) {
  constexpr auto AsmConvertUX32F64 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("ucvtf %d0, %w1, #8");

  ASSERT_EQ(AsmConvertUX32F64(0x361), MakeUInt128(0x400b'0800'0000'0000ULL, 0U));

  ASSERT_EQ(AsmConvertUX32F64(1U << 31), MakeUInt128(0x4160'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertUX32F32) {
  constexpr auto AsmConvertUX32F32 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %s0, %s1, #7");

  ASSERT_EQ(AsmConvertUX32F32(0x456), MakeUInt128(0x410a'c000ULL, 0U));

  ASSERT_EQ(AsmConvertUX32F32(1U << 31), MakeUInt128(0x4b80'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertUX32x4F32x4) {
  constexpr auto AsmConvertUX32F32 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %0.4s, %1.4s, #11");
  __uint128_t arg = MakeUInt128(0x8000'0000'0000'8023ULL, 0x0000'2018'0000'1956ULL);
  ASSERT_EQ(AsmConvertUX32F32(arg),
            MakeUInt128(0x4980'0000'4180'2300ULL, 0x4080'6000'404a'b000ULL));
}

TEST(Arm64InsnTest, AsmConvertX64F32Scalar) {
  constexpr auto AsmConvertX64F32 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("scvtf %s0, %x1, #10");

  ASSERT_EQ(AsmConvertX64F32(0x2234), MakeUInt128(0x4108'd000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertX64F64Scalar) {
  constexpr auto AsmConvertX64F64 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("scvtf %d0, %x1, #10");

  ASSERT_EQ(AsmConvertX64F64(0x1324), MakeUInt128(0x4013'2400'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertUX64F32Scalar) {
  constexpr auto AsmConvertUX64F32 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("ucvtf %s0, %x1, #10");

  ASSERT_EQ(AsmConvertUX64F32(0x5763), MakeUInt128(0x41ae'c600ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertUX64F64Scalar) {
  constexpr auto AsmConvertUX64F64 = ASM_INSN_WRAP_FUNC_W_RES_R_ARG("ucvtf %d0, %x1, #10");

  ASSERT_EQ(AsmConvertUX64F64(0x2217), MakeUInt128(0x4021'0b80'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertX64F64) {
  constexpr auto AsmConvertX64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("scvtf %d0, %d1, #12");

  ASSERT_EQ(AsmConvertX64F64(0x723), MakeUInt128(0x3fdc'8c00'0000'0000ULL, 0U));

  ASSERT_EQ(AsmConvertX64F64(1ULL << 63), MakeUInt128(0xc320'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertUX64F64) {
  constexpr auto AsmConvertUX64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %d0, %d1, #12");

  ASSERT_EQ(AsmConvertUX64F64(0x416), MakeUInt128(0x3fd0'5800'0000'0000ULL, 0U));

  ASSERT_EQ(AsmConvertUX64F64(1ULL << 63), MakeUInt128(0x4320'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertUX64F64With64BitFraction) {
  constexpr auto AsmConvertUX64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %d0, %d1, #64");

  ASSERT_EQ(AsmConvertUX64F64(1ULL << 63), MakeUInt128(0x3fe0'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertX64x2F64x2) {
  constexpr auto AsmConvertX64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("scvtf %0.2d, %1.2d, #12");
  __uint128_t arg = MakeUInt128(1ULL << 63, 0x8086U);
  ASSERT_EQ(AsmConvertX64F64(arg), MakeUInt128(0xc320'0000'0000'0000ULL, 0x4020'10c0'0000'0000ULL));
}

TEST(Arm64InsnTest, AsmConvertUX64x2F64x2) {
  constexpr auto AsmConvertUX64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %0.2d, %1.2d, #12");
  __uint128_t arg = MakeUInt128(1ULL << 63, 0x6809U);
  ASSERT_EQ(AsmConvertUX64F64(arg),
            MakeUInt128(0x4320'0000'0000'0000ULL, 0x401a'0240'0000'0000ULL));
}

TEST(Arm64InsnTest, AsmConvertUX64x2F64x2With64BitFraction) {
  constexpr auto AsmConvertUX64F64 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ucvtf %0.2d, %1.2d, #64");
  __uint128_t arg = MakeUInt128(0x7874'211c'b7aa'f597ULL, 0x2c0f'5504'd25e'f673ULL);
  ASSERT_EQ(AsmConvertUX64F64(arg),
            MakeUInt128(0x3fde'1d08'472d'eabdULL, 0x3fc6'07aa'8269'2f7bULL));
}

TEST(Arm64InsnTest, AsmConvertF32X32Scalar) {
  constexpr auto AsmConvertF32X32 = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %w0, %s1, #16");
  uint32_t arg1 = 0x4091'eb85U;  // 4.56 in float
  ASSERT_EQ(AsmConvertF32X32(arg1), MakeUInt128(0x0004'8f5cU, 0U));

  uint32_t arg2 = 0xc0d8'0000U;  // -6.75 in float
  ASSERT_EQ(AsmConvertF32X32(arg2), MakeUInt128(0xfff9'4000U, 0U));

  ASSERT_EQ(AsmConvertF32X32(kDefaultNaN32AsInteger), MakeUInt128(bit_cast<uint32_t>(0.0f), 0U));
}

TEST(Arm64InsnTest, AsmConvertF32UX32Scalar) {
  constexpr auto AsmConvertF32UX32 = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %w0, %s1, #16");
  uint32_t arg1 = 0x4122'3d71U;  // 10.14 in float
  ASSERT_EQ(AsmConvertF32UX32(arg1), MakeUInt128(0x000a'23d7U, 0U));

  uint32_t arg2 = 0xc154'0000U;  // -13.25 in float
  ASSERT_EQ(AsmConvertF32UX32(arg2), MakeUInt128(0xfff2'c000U, 0U));

  ASSERT_EQ(AsmConvertF32UX32(kDefaultNaN32AsInteger), MakeUInt128(bit_cast<uint32_t>(0.0f), 0U));
}

TEST(Arm64InsnTest, AsmConvertF32UX32With31FractionalBits) {
  constexpr auto AsmConvertF32UX32 = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %w0, %s1, #31");
  uint32_t arg1 = bit_cast<uint32_t>(0.25f);
  ASSERT_EQ(AsmConvertF32UX32(arg1), MakeUInt128(0x2000'0000U, 0U));
}

TEST(Arm64InsnTest, AsmConvertF64X32Scalar) {
  constexpr auto AsmConvertF64X32 = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %w0, %d1, #16");
  uint64_t arg1 = 0x401e'8f5c'28f5'c28fULL;  // 7.46 in double
  ASSERT_EQ(AsmConvertF64X32(arg1), MakeUInt128(0x0007'a3d7U, 0U));

  uint64_t arg2 = 0xc040'2000'0000'0000ULL;  // -32.44 in double
  ASSERT_EQ(AsmConvertF64X32(arg2), MakeUInt128(0xffdf'c000U, 0U));
}

TEST(Arm64InsnTest, AsmConvertF32X64Scalar) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %x0, %s1, #16");
  uint64_t arg1 = bit_cast<uint32_t>(7.50f);
  ASSERT_EQ(AsmFcvtzs(arg1), MakeUInt128(0x0000'0000'0007'8000ULL, 0ULL));

  uint64_t arg2 = bit_cast<uint32_t>(-6.50f);
  ASSERT_EQ(AsmFcvtzs(arg2), MakeUInt128(0xffff'ffff'fff9'8000ULL, 0ULL));
}

TEST(Arm64InsnTest, AsmConvertF32UX64With63FractionalBits) {
  constexpr auto AsmConvertF32UX64 = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %x0, %s1, #63");
  uint32_t arg1 = bit_cast<uint32_t>(0.25f);
  ASSERT_EQ(AsmConvertF32UX64(arg1), MakeUInt128(0x2000'0000'0000'0000ULL, 0U));
}

TEST(Arm64InsnTest, AsmConvertF64X64Scalar) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzs %x0, %d1, #16");
  uint64_t arg1 = bit_cast<uint64_t>(7.50);
  ASSERT_EQ(AsmFcvtzs(arg1), MakeUInt128(0x0000'0000'0007'8000ULL, 0ULL));

  uint64_t arg2 = bit_cast<uint64_t>(-6.50);
  ASSERT_EQ(AsmFcvtzs(arg2), MakeUInt128(0xffff'ffff'fff9'8000ULL, 0ULL));
}

TEST(Arm64InsnTest, AsmConvertF32X32x4) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzs %0.4s, %1.4s, #2");
  __uint128_t res = AsmFcvtzs(MakeF32x4(-5.5f, -0.0f, 0.0f, 6.5f));
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'ffff'ffeaULL, 0x0000'001a'0000'0000ULL));
}

TEST(Arm64InsnTest, AsmConvertF64UX32Scalar) {
  constexpr auto AsmConvertF64UX32 = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzu %w0, %d1, #16");
  uint64_t arg1 = 0x4020'947a'e147'ae14ULL;  // 8.29 in double
  ASSERT_EQ(AsmConvertF64UX32(arg1), MakeUInt128(0x0008'4a3dU, 0U));

  uint64_t arg2 = 0xc023'6666'6666'6666ULL;  // -9.70 in double
  ASSERT_EQ(AsmConvertF64UX32(arg2), MakeUInt128(0U, 0U));
}

TEST(Arm64InsnTest, AsmConvertF32UX64Scalar) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzu %x0, %s1, #16");
  uint64_t arg1 = bit_cast<uint32_t>(7.50f);
  ASSERT_EQ(AsmFcvtzu(arg1), MakeUInt128(0x0000'0000'0007'8000ULL, 0ULL));
  uint64_t arg2 = bit_cast<uint32_t>(-6.50f);
  ASSERT_EQ(AsmFcvtzu(arg2), 0ULL);
}

TEST(Arm64InsnTest, AsmConvertF64UX64Scalar) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzu %x0, %d1, #16");
  uint64_t arg1 = bit_cast<uint64_t>(7.50);
  ASSERT_EQ(AsmFcvtzu(arg1), MakeUInt128(0x0000'0000'0007'8000ULL, 0ULL));

  uint64_t arg2 = bit_cast<uint64_t>(-6.50);
  ASSERT_EQ(AsmFcvtzu(arg2), MakeUInt128(0ULL, 0ULL));
}

TEST(Arm64InsnTest, AsmConvertF64UX64ScalarWith64BitFraction) {
  constexpr auto AsmFcvtzu = ASM_INSN_WRAP_FUNC_R_RES_W_ARG("fcvtzu %x0, %d1, #64");
  uint64_t arg = bit_cast<uint64_t>(0.625);
  ASSERT_EQ(AsmFcvtzu(arg), MakeUInt128(0xa000'0000'0000'0000ULL, 0ULL));
}

TEST(Arm64InsnTest, AsmConvertF32UX32x4) {
  constexpr auto AsmFcvtzs = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtzu %0.4s, %1.4s, #2");
  __uint128_t res = AsmFcvtzs(MakeF32x4(-5.5f, -0.0f, 0.0f, 6.5f));
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'001a'0000'0000ULL));
}

TEST(Arm64InsnTest, Fp32ConditionalSelect) {
  uint64_t int_arg1 = 3;
  uint64_t int_arg2 = 7;
  uint64_t fp_arg1 = 0xfedc'ba98'7654'3210ULL;
  uint64_t fp_arg2 = 0x0123'4567'89ab'cdefULL;
  __uint128_t res;

  asm("cmp %x1,%x2\n\t"
      "fcsel %s0, %s3, %s4, eq"
      : "=w"(res)
      : "r"(int_arg1), "r"(int_arg2), "w"(fp_arg1), "w"(fp_arg2));
  ASSERT_EQ(res, MakeUInt128(0x89ab'cdefULL, 0U));

  asm("cmp %x1,%x2\n\t"
      "fcsel %s0, %s3, %s4, ne"
      : "=w"(res)
      : "r"(int_arg1), "r"(int_arg2), "w"(fp_arg1), "w"(fp_arg2));
  ASSERT_EQ(res, MakeUInt128(0x7654'3210ULL, 0U));
}

TEST(Arm64InsnTest, Fp64ConditionalSelect) {
  uint64_t int_arg1 = 8;
  uint64_t int_arg2 = 3;
  uint64_t fp_arg1 = 0xfedc'ba98'7654'3210ULL;
  uint64_t fp_arg2 = 0x0123'4567'89ab'cdefULL;
  __uint128_t res;

  asm("cmp %x1,%x2\n\t"
      "fcsel %d0, %d3, %d4, eq"
      : "=w"(res)
      : "r"(int_arg1), "r"(int_arg2), "w"(fp_arg1), "w"(fp_arg2));
  ASSERT_EQ(res, MakeUInt128(0x0123'4567'89ab'cdefULL, 0U));

  asm("cmp %x1,%x2\n\t"
      "fcsel %d0, %d3, %d4, ne"
      : "=w"(res)
      : "r"(int_arg1), "r"(int_arg2), "w"(fp_arg1), "w"(fp_arg2));
  ASSERT_EQ(res, MakeUInt128(0xfedc'ba98'7654'3210ULL, 0U));
}

TEST(Arm64InsnTest, RoundUpFp32) {
  // The lower 32-bit represents 2.7182817 in float.
  uint64_t fp_arg = 0xdead'beef'402d'f854ULL;
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintp %s0, %s1")(fp_arg);
  ASSERT_EQ(res, MakeUInt128(0x4040'0000ULL, 0U));  // 3.0 in float
}

TEST(Arm64InsnTest, RoundUpFp64) {
  // 2.7182817 in double.
  uint64_t fp_arg = 0x4005'bf0a'8b14'5769ULL;
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintp %d0, %d1")(fp_arg);
  ASSERT_EQ(res, MakeUInt128(0x4008'0000'0000'0000ULL, 0U));  // 3.0 in double
}

TEST(Arm64InsnTest, RoundToIntNearestTiesAwayFp64) {
  constexpr auto AsmFrinta = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frinta %d0, %d1");

  // -7.50 -> -8.00 (ties away from zero as opposted to even)
  ASSERT_EQ(AsmFrinta(0xc01e'0000'0000'0000ULL), MakeUInt128(0xc020'0000'0000'0000ULL, 0U));

  // -6.75 -> -7.00
  ASSERT_EQ(AsmFrinta(0xc01b'0000'0000'0000ULL), MakeUInt128(0xc01c'0000'0000'0000ULL, 0U));

  // -6.50 -> -7.00 (ties away from zero as opposted to even)
  ASSERT_EQ(AsmFrinta(0xc01a'0000'0000'0000ULL), MakeUInt128(0xc01c'0000'0000'0000ULL, 0U));

  // -6.25 -> -6.00
  ASSERT_EQ(AsmFrinta(0xc019'0000'0000'0000ULL), MakeUInt128(0xc018'0000'0000'0000ULL, 0U));

  // 6.25 -> 6.00
  ASSERT_EQ(AsmFrinta(0x4019'0000'0000'0000ULL), MakeUInt128(0x4018'0000'0000'0000ULL, 0U));

  // 6.50 -> 7.00 (ties away from zero as opposted to even)
  ASSERT_EQ(AsmFrinta(0x401a'0000'0000'0000ULL), MakeUInt128(0x401c'0000'0000'0000ULL, 0U));

  // 6.75 -> 7.00
  ASSERT_EQ(AsmFrinta(0x401b'0000'0000'0000ULL), MakeUInt128(0x401c'0000'0000'0000ULL, 0U));

  // 7.50 -> 8.00 (ties away from zero as opposted to even)
  ASSERT_EQ(AsmFrinta(0x401e'0000'0000'0000ULL), MakeUInt128(0x4020'0000'0000'0000ULL, 0U));

  // -0.49999999999999994 -> -0.0 (should not "tie away" since -0.4999... != -0.5)
  ASSERT_EQ(AsmFrinta(0xbfdf'ffff'ffff'ffff), MakeUInt128(0x8000'0000'0000'0000U, 0U));

  // A number too large to have fractional precision, should not change upon rounding with tie-away
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(0.5 / std::numeric_limits<double>::epsilon())),
            MakeUInt128(bit_cast<uint64_t>(0.5 / std::numeric_limits<double>::epsilon()), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(-0.5 / std::numeric_limits<double>::epsilon())),
            MakeUInt128(bit_cast<uint64_t>(-0.5 / std::numeric_limits<double>::epsilon()), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(0.75 / std::numeric_limits<double>::epsilon())),
            MakeUInt128(bit_cast<uint64_t>(0.75 / std::numeric_limits<double>::epsilon()), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(-0.75 / std::numeric_limits<double>::epsilon())),
            MakeUInt128(bit_cast<uint64_t>(-0.75 / std::numeric_limits<double>::epsilon()), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(1.0 / std::numeric_limits<double>::epsilon())),
            MakeUInt128(bit_cast<uint64_t>(1.0 / std::numeric_limits<double>::epsilon()), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(-1.0 / std::numeric_limits<double>::epsilon())),
            MakeUInt128(bit_cast<uint64_t>(-1.0 / std::numeric_limits<double>::epsilon()), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(2.0 / std::numeric_limits<double>::epsilon())),
            MakeUInt128(bit_cast<uint64_t>(2.0 / std::numeric_limits<double>::epsilon()), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(-2.0 / std::numeric_limits<double>::epsilon())),
            MakeUInt128(bit_cast<uint64_t>(-2.0 / std::numeric_limits<double>::epsilon()), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(1.0e100)), MakeUInt128(bit_cast<uint64_t>(1.0e100), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint64_t>(-1.0e100)), MakeUInt128(bit_cast<uint64_t>(-1.0e100), 0U));
}

TEST(Arm64InsnTest, RoundToIntNearestTiesAwayFp32) {
  constexpr auto AsmFrinta = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frinta %s0, %s1");

  // -7.50 -> -8.00 (ties away from zero as opposted to even)
  ASSERT_EQ(AsmFrinta(0xc0f0'0000U), MakeUInt128(0xc100'0000U, 0U));

  // -6.75 -> -7.00
  ASSERT_EQ(AsmFrinta(0xc0d8'0000U), MakeUInt128(0xc0e0'0000U, 0U));

  // -6.50 -> -7.00 (ties away from zero as opposted to even)
  ASSERT_EQ(AsmFrinta(0xc0d0'0000U), MakeUInt128(0xc0e0'0000U, 0U));

  // -6.25 -> -6.00
  ASSERT_EQ(AsmFrinta(0xc0c8'0000U), MakeUInt128(0xc0c0'0000U, 0U));

  // 6.25 -> 6.00
  ASSERT_EQ(AsmFrinta(0x40c8'0000U), MakeUInt128(0x40c0'0000U, 0U));

  // 6.50 -> 7.00 (ties away from zero as opposted to even)
  ASSERT_EQ(AsmFrinta(0x40d0'0000U), MakeUInt128(0x40e0'0000U, 0U));

  // 6.75 -> 7.00
  ASSERT_EQ(AsmFrinta(0x40d8'0000U), MakeUInt128(0x40e0'0000U, 0U));

  // 7.50 -> 8.00 (ties away from zero as opposted to even)
  ASSERT_EQ(AsmFrinta(0x40f0'0000U), MakeUInt128(0x4100'0000U, 0U));

  // -0.49999997019767761 -> -0.0 (should not "tie away" since -0.4999... != -0.5)
  ASSERT_EQ(AsmFrinta(0xbeff'ffff), MakeUInt128(0x8000'0000U, 0U));

  // A number too large to have fractional precision, should not change upon rounding with tie-away
  ASSERT_EQ(
      AsmFrinta(bit_cast<uint32_t>(float{0.5 / std::numeric_limits<float>::epsilon()})),
      MakeUInt128(bit_cast<uint32_t>(float{0.5 / std::numeric_limits<float>::epsilon()}), 0U));
  ASSERT_EQ(
      AsmFrinta(bit_cast<uint32_t>(float{-0.5 / std::numeric_limits<float>::epsilon()})),
      MakeUInt128(bit_cast<uint32_t>(float{-0.5 / std::numeric_limits<float>::epsilon()}), 0U));
  ASSERT_EQ(
      AsmFrinta(bit_cast<uint32_t>(float{0.75 / std::numeric_limits<float>::epsilon()})),
      MakeUInt128(bit_cast<uint32_t>(float{0.75 / std::numeric_limits<float>::epsilon()}), 0U));
  ASSERT_EQ(
      AsmFrinta(bit_cast<uint32_t>(float{-0.75 / std::numeric_limits<float>::epsilon()})),
      MakeUInt128(bit_cast<uint32_t>(float{-0.75 / std::numeric_limits<float>::epsilon()}), 0U));
  ASSERT_EQ(
      AsmFrinta(bit_cast<uint32_t>(float{1.0 / std::numeric_limits<float>::epsilon()})),
      MakeUInt128(bit_cast<uint32_t>(float{1.0 / std::numeric_limits<float>::epsilon()}), 0U));
  ASSERT_EQ(
      AsmFrinta(bit_cast<uint32_t>(float{-1.0 / std::numeric_limits<float>::epsilon()})),
      MakeUInt128(bit_cast<uint32_t>(float{-1.0 / std::numeric_limits<float>::epsilon()}), 0U));
  ASSERT_EQ(
      AsmFrinta(bit_cast<uint32_t>(float{2.0 / std::numeric_limits<float>::epsilon()})),
      MakeUInt128(bit_cast<uint32_t>(float{2.0 / std::numeric_limits<float>::epsilon()}), 0U));
  ASSERT_EQ(
      AsmFrinta(bit_cast<uint32_t>(float{-2.0 / std::numeric_limits<float>::epsilon()})),
      MakeUInt128(bit_cast<uint32_t>(float{-2.0 / std::numeric_limits<float>::epsilon()}), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint32_t>(1.0e38f)), MakeUInt128(bit_cast<uint32_t>(1.0e38f), 0U));
  ASSERT_EQ(AsmFrinta(bit_cast<uint32_t>(-1.0e38f)), MakeUInt128(bit_cast<uint32_t>(-1.0e38f), 0U));
}

TEST(Arm64InsnTest, RoundToIntDownwardFp64) {
  constexpr auto AsmFrintm = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintm %d0, %d1");

  // 7.7 -> 7.00
  ASSERT_EQ(AsmFrintm(0x401e'cccc'cccc'cccdULL), MakeUInt128(0x401c'0000'0000'0000, 0U));

  // 7.1 -> 7.00
  ASSERT_EQ(AsmFrintm(0x401c'6666'6666'6666ULL), MakeUInt128(0x401c'0000'0000'0000, 0U));

  // -7.10 -> -8.00
  ASSERT_EQ(AsmFrintm(0xc01c'6666'6666'6666ULL), MakeUInt128(0xc020'0000'0000'0000, 0U));

  // -7.90 -> -8.00
  ASSERT_EQ(AsmFrintm(0xc01f'9999'9999'999aULL), MakeUInt128(0xc020'0000'0000'0000, 0U));

  // 0 -> 0
  ASSERT_EQ(AsmFrintm(0x0000'0000'0000'0000ULL), MakeUInt128(0x0000'0000'0000'0000, 0U));

  // -0 -> -0
  ASSERT_EQ(AsmFrintm(0x8000'0000'0000'0000ULL), MakeUInt128(0x8000'0000'0000'0000, 0U));
}

TEST(Arm64InsnTest, RoundToIntDownwardFp32) {
  constexpr auto AsmFrintm = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintm %s0, %s1");

  // 7.7 -> 7.00
  ASSERT_EQ(AsmFrintm(0x40f6'6666), 0x40e0'0000);

  // 7.1 -> 7.00
  ASSERT_EQ(AsmFrintm(0x40e3'3333), 0x40e0'0000);

  // -7.10 -> -8.00
  ASSERT_EQ(AsmFrintm(0xc0e3'3333), 0xc100'0000);

  // -7.90 -> -8.00
  ASSERT_EQ(AsmFrintm(0xc0fc'cccd), 0xc100'0000);

  // 0 -> 0
  ASSERT_EQ(AsmFrintm(0x0000'0000), 0x0000'0000);

  // -0 -> -0
  ASSERT_EQ(AsmFrintm(0x8000'0000), 0x8000'0000);
}

TEST(Arm64InsnTest, RoundToIntNearestFp64) {
  constexpr auto AsmFrintn = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintn %d0, %d1");

  // 7.5 -> 8.00 (ties to even)
  ASSERT_EQ(AsmFrintn(0x401e'0000'0000'0000ULL), MakeUInt128(0x4020'0000'0000'0000, 0U));

  // 8.5 -> 8.00 (ties to even)
  ASSERT_EQ(AsmFrintn(0x4021'0000'0000'0000), MakeUInt128(0x4020'0000'0000'0000, 0U));

  // 7.10 -> 7.00
  ASSERT_EQ(AsmFrintn(0x401c'6666'6666'6666), MakeUInt128(0x401c'0000'0000'0000, 0U));

  // 7.90 -> 8.00
  ASSERT_EQ(AsmFrintn(0x401f'9999'9999'999a), MakeUInt128(0x4020'0000'0000'0000, 0U));

  // -7.5 -> -8.00 (ties to even)
  ASSERT_EQ(AsmFrintn(0xc01e'0000'0000'0000), MakeUInt128(0xc020'0000'0000'0000, 0U));

  // // -8.5 -> -8.00 (ties to even)
  ASSERT_EQ(AsmFrintn(0xc021'0000'0000'0000), MakeUInt128(0xc020'0000'0000'0000, 0U));

  // -7.10 -> -7.00
  ASSERT_EQ(AsmFrintn(0xc01c'6666'6666'6666), MakeUInt128(0xc01c'0000'0000'0000, 0U));

  // -7.90 -> -8.00
  ASSERT_EQ(AsmFrintn(0xc01f'9999'9999'999a), MakeUInt128(0xc020'0000'0000'0000, 0U));

  // 0 -> 0
  ASSERT_EQ(AsmFrintn(0x0000'0000'0000'0000ULL), MakeUInt128(0x0000'0000'0000'0000, 0U));

  // -0 -> -0
  ASSERT_EQ(AsmFrintn(0x8000'0000'0000'0000ULL), MakeUInt128(0x8000'0000'0000'0000, 0U));
}

TEST(Arm64InsnTest, RoundToIntToNearestFp32) {
  constexpr auto AsmFrintn = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintn %s0, %s1");

  // 7.5 -> 8.00 (ties to even)
  ASSERT_EQ(AsmFrintn(0x40f0'0000), 0x4100'0000);

  // 8.5 -> 8.00 (ties to even)
  ASSERT_EQ(AsmFrintn(0x4108'0000), 0x4100'0000);

  // 7.10 -> 7.00
  ASSERT_EQ(AsmFrintn(0x40e3'3333), 0x40e0'0000);

  // 7.90 -> 8.00
  ASSERT_EQ(AsmFrintn(0x40fc'cccd), 0x4100'0000);

  // -7.5 -> -8.00 (ties to even)
  ASSERT_EQ(AsmFrintn(0xc0f0'0000), 0xc100'0000);

  // -8.5 -> -8.00 (ties to even)
  ASSERT_EQ(AsmFrintn(0xc108'0000), 0xc100'0000);

  // -7.10 -> -7.00
  ASSERT_EQ(AsmFrintn(0xc0e3'3333), 0xc0e0'0000);

  // -7.90 -> -8.00
  ASSERT_EQ(AsmFrintn(0xc0fc'cccd), 0xc100'0000);

  // 0 -> 0
  ASSERT_EQ(AsmFrintn(0x0000'0000), 0x0000'0000);

  // -0 -> -0
  ASSERT_EQ(AsmFrintn(0x8000'0000), 0x8000'0000);
}

TEST(Arm64InsnTest, RoundToIntTowardZeroFp64) {
  constexpr auto AsmFrintz = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintz %d0, %d1");

  // 7.7 -> 7.00
  ASSERT_EQ(AsmFrintz(0x401e'cccc'cccc'cccdULL), MakeUInt128(0x401c'0000'0000'0000, 0U));

  // 7.1 -> 7.00
  ASSERT_EQ(AsmFrintz(0x401c'6666'6666'6666ULL), MakeUInt128(0x401c'0000'0000'0000, 0U));

  // -7.10 -> -7.00
  ASSERT_EQ(AsmFrintz(0xc01c'6666'6666'6666ULL), MakeUInt128(0xc01c'0000'0000'0000, 0U));

  // -7.90 -> -7.00
  ASSERT_EQ(AsmFrintz(0xc01f'9999'9999'999aULL), MakeUInt128(0xc01c'0000'0000'0000, 0U));

  // 0 -> 0
  ASSERT_EQ(AsmFrintz(0x0000'0000'0000'0000ULL), MakeUInt128(0x0000'0000'0000'0000, 0U));

  // -0 -> -0
  ASSERT_EQ(AsmFrintz(0x8000'0000'0000'0000ULL), MakeUInt128(0x8000'0000'0000'0000, 0U));
}

TEST(Arm64InsnTest, RoundToIntTowardZeroFp32) {
  constexpr auto AsmFrintz = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintz %s0, %s1");

  // 7.7 -> 7.00
  ASSERT_EQ(AsmFrintz(0x40f6'6666), 0x40e0'0000);

  // 7.1 -> 7.00
  ASSERT_EQ(AsmFrintz(0x40e3'3333), 0x40e0'0000);

  // -7.10 -> -7.00
  ASSERT_EQ(AsmFrintz(0xc0e3'3333), 0xc0e0'0000);

  // -7.90 -> -7.00
  ASSERT_EQ(AsmFrintz(0xc0fc'cccd), 0xc0e0'0000);

  // 0 -> 0
  ASSERT_EQ(AsmFrintz(0x0000'0000), 0x0000'0000);

  // -0 -> -0
  ASSERT_EQ(AsmFrintz(0x8000'0000), 0x8000'0000);
}

TEST(Arm64InsnTest, AsmConvertF32x4TieAway) {
  constexpr auto AsmFcvta = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frinta %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvta(arg1), MakeF32x4(-8.00f, -7.00f, -7.00f, -6.00f));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvta(arg2), MakeF32x4(6.00f, 7.00f, 7.00f, 8.00f));
}

TEST(Arm64InsnTest, AsmConvertF32x4NegInf) {
  constexpr auto AsmFcvtm = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintm %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtm(arg1), MakeF32x4(-8.00f, -7.00f, -7.00f, -7.00f));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtm(arg2), MakeF32x4(6.00f, 6.00f, 6.00f, 7.00f));
}

TEST(Arm64InsnTest, AsmConvertF32x4TieEven) {
  constexpr auto AsmFcvtn = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintn %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtn(arg1), MakeF32x4(-8.00f, -7.00f, -6.00f, -6.00f));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtn(arg2), MakeF32x4(6.00f, 6.00f, 7.00f, 8.00f));
}

TEST(Arm64InsnTest, AsmConvertF32x4PosInf) {
  constexpr auto AsmFcvtp = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintp %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtp(arg1), MakeF32x4(-7.00f, -6.00f, -6.00f, -6.00f));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtp(arg2), MakeF32x4(7.00f, 7.00f, 7.00f, 8.00f));
}

TEST(Arm64InsnTest, AsmConvertF32x4Truncate) {
  constexpr auto AsmFcvtz = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintz %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFcvtz(arg1), MakeF32x4(-7.00f, -6.00f, -6.00f, -6.00f));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFcvtz(arg2), MakeF32x4(6.00f, 6.00f, 6.00f, 7.00f));
}

TEST(Arm64InsnTest, AsmConvertF64x4TieAway) {
  constexpr auto AsmFcvta = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frinta %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvta(arg1), MakeF64x2(-8.00, -7.00));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvta(arg2), MakeF64x2(-7.00, -6.00));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvta(arg3), MakeF64x2(6.00, 7.00));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvta(arg4), MakeF64x2(7.00, 8.00));
}

TEST(Arm64InsnTest, AsmConvertF64x4NegInf) {
  constexpr auto AsmFcvtm = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintm %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtm(arg1), MakeF64x2(-8.00, -7.00));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtm(arg2), MakeF64x2(-7.00, -7.00));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtm(arg3), MakeF64x2(6.00, 6.00));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtm(arg4), MakeF64x2(6.00, 7.00));
}

TEST(Arm64InsnTest, AsmConvertF64x4TieEven) {
  constexpr auto AsmFcvtn = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintn %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtn(arg1), MakeF64x2(-8.00, -7.00));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtn(arg2), MakeF64x2(-6.00, -6.00));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtn(arg3), MakeF64x2(6.00, 6.00));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtn(arg4), MakeF64x2(7.00, 8.00));
}

TEST(Arm64InsnTest, AsmConvertF64x4PosInf) {
  constexpr auto AsmFcvtp = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintp %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtp(arg1), MakeF64x2(-7.00, -6.00));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtp(arg2), MakeF64x2(-6.00, -6.00));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtp(arg3), MakeF64x2(7.00, 7.00));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtp(arg4), MakeF64x2(7.00, 8.00));
}

TEST(Arm64InsnTest, AsmConvertF64x4Truncate) {
  constexpr auto AsmFcvtz = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frintz %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFcvtz(arg1), MakeF64x2(-7.00, -6.00));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFcvtz(arg2), MakeF64x2(-6.00, -6.00));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFcvtz(arg3), MakeF64x2(6.00, 6.00));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFcvtz(arg4), MakeF64x2(6.00, 7.00));
}

TEST(Arm64InsnTest, AsmRoundCurrentModeF32) {
  constexpr auto AsmFrinti = ASM_INSN_WRAP_FUNC_W_RES_WC_ARG("frinti %s0, %s1");
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-7.50f), kFpcrRModeTieEven), bit_cast<uint32_t>(-8.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.75f), kFpcrRModeTieEven), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.50f), kFpcrRModeTieEven), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.25f), kFpcrRModeTieEven), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.25f), kFpcrRModeTieEven), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.50f), kFpcrRModeTieEven), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.75f), kFpcrRModeTieEven), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(7.50f), kFpcrRModeTieEven), bit_cast<uint32_t>(8.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-7.50f), kFpcrRModeNegInf), bit_cast<uint32_t>(-8.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.75f), kFpcrRModeNegInf), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.50f), kFpcrRModeNegInf), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.25f), kFpcrRModeNegInf), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.25f), kFpcrRModeNegInf), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.50f), kFpcrRModeNegInf), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.75f), kFpcrRModeNegInf), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(7.50f), kFpcrRModeNegInf), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-7.50f), kFpcrRModePosInf), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.75f), kFpcrRModePosInf), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.50f), kFpcrRModePosInf), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.25f), kFpcrRModePosInf), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.25f), kFpcrRModePosInf), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.50f), kFpcrRModePosInf), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.75f), kFpcrRModePosInf), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(7.50f), kFpcrRModePosInf), bit_cast<uint32_t>(8.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-7.50f), kFpcrRModeZero), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.75f), kFpcrRModeZero), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.50f), kFpcrRModeZero), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(-6.25f), kFpcrRModeZero), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.25f), kFpcrRModeZero), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.50f), kFpcrRModeZero), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(6.75f), kFpcrRModeZero), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrinti(bit_cast<uint32_t>(7.50f), kFpcrRModeZero), bit_cast<uint32_t>(7.00f));
}

TEST(Arm64InsnTest, AsmRoundCurrentModeF64) {
  constexpr auto AsmFrinti = ASM_INSN_WRAP_FUNC_W_RES_WC_ARG("frinti %d0, %d1");
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-7.50), kFpcrRModeTieEven), bit_cast<uint64_t>(-8.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.75), kFpcrRModeTieEven), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.50), kFpcrRModeTieEven), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.25), kFpcrRModeTieEven), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.25), kFpcrRModeTieEven), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.50), kFpcrRModeTieEven), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.75), kFpcrRModeTieEven), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(7.50), kFpcrRModeTieEven), bit_cast<uint64_t>(8.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-7.50), kFpcrRModeNegInf), bit_cast<uint64_t>(-8.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.75), kFpcrRModeNegInf), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.50), kFpcrRModeNegInf), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.25), kFpcrRModeNegInf), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.25), kFpcrRModeNegInf), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.50), kFpcrRModeNegInf), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.75), kFpcrRModeNegInf), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(7.50), kFpcrRModeNegInf), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-7.50), kFpcrRModePosInf), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.75), kFpcrRModePosInf), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.50), kFpcrRModePosInf), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.25), kFpcrRModePosInf), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.25), kFpcrRModePosInf), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.50), kFpcrRModePosInf), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.75), kFpcrRModePosInf), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(7.50), kFpcrRModePosInf), bit_cast<uint64_t>(8.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-7.50), kFpcrRModeZero), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.75), kFpcrRModeZero), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.50), kFpcrRModeZero), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(-6.25), kFpcrRModeZero), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.25), kFpcrRModeZero), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.50), kFpcrRModeZero), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(6.75), kFpcrRModeZero), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrinti(bit_cast<uint64_t>(7.50), kFpcrRModeZero), bit_cast<uint64_t>(7.00));
}

TEST(Arm64InsnTest, AsmRoundCurrentModeF32x4) {
  constexpr auto AsmFrinti = ASM_INSN_WRAP_FUNC_W_RES_WC_ARG("frinti %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFrinti(arg1, kFpcrRModeTieEven), MakeF32x4(-8.00f, -7.00f, -6.00f, -6.00f));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFrinti(arg2, kFpcrRModeTieEven), MakeF32x4(6.00f, 6.00f, 7.00f, 8.00f));
  __uint128_t arg3 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFrinti(arg3, kFpcrRModeNegInf), MakeF32x4(-8.00f, -7.00f, -7.00f, -7.00f));
  __uint128_t arg4 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFrinti(arg4, kFpcrRModeNegInf), MakeF32x4(6.00f, 6.00f, 6.00f, 7.00f));
  __uint128_t arg5 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFrinti(arg5, kFpcrRModePosInf), MakeF32x4(-7.00f, -6.00f, -6.00f, -6.00f));
  __uint128_t arg6 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFrinti(arg6, kFpcrRModePosInf), MakeF32x4(7.00f, 7.00f, 7.00f, 8.00f));
  __uint128_t arg7 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFrinti(arg7, kFpcrRModeZero), MakeF32x4(-7.00f, -6.00f, -6.00f, -6.00f));
  __uint128_t arg8 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFrinti(arg8, kFpcrRModeZero), MakeF32x4(6.00f, 6.00f, 6.00f, 7.00f));
}

TEST(Arm64InsnTest, AsmRoundCurrentModeF64x2) {
  constexpr auto AsmFrinti = ASM_INSN_WRAP_FUNC_W_RES_WC_ARG("frinti %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFrinti(arg1, kFpcrRModeTieEven), MakeF64x2(-8.00, -7.00));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFrinti(arg2, kFpcrRModeTieEven), MakeF64x2(-6.00, -6.00));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFrinti(arg3, kFpcrRModeTieEven), MakeF64x2(6.00, 6.00));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFrinti(arg4, kFpcrRModeTieEven), MakeF64x2(7.00, 8.00));
  __uint128_t arg5 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFrinti(arg5, kFpcrRModeNegInf), MakeF64x2(-8.00, -7.00));
  __uint128_t arg6 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFrinti(arg6, kFpcrRModeNegInf), MakeF64x2(-7.00, -7.00));
  __uint128_t arg7 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFrinti(arg7, kFpcrRModeNegInf), MakeF64x2(6.00, 6.00));
  __uint128_t arg8 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFrinti(arg8, kFpcrRModeNegInf), MakeF64x2(6.00, 7.00));
  __uint128_t arg9 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFrinti(arg9, kFpcrRModePosInf), MakeF64x2(-7.00, -6.00));
  __uint128_t arg10 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFrinti(arg10, kFpcrRModePosInf), MakeF64x2(-6.00, -6.00));
  __uint128_t arg11 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFrinti(arg11, kFpcrRModePosInf), MakeF64x2(7.00, 7.00));
  __uint128_t arg12 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFrinti(arg12, kFpcrRModePosInf), MakeF64x2(7.00, 8.00));
  __uint128_t arg13 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFrinti(arg13, kFpcrRModeZero), MakeF64x2(-7.00, -6.00));
  __uint128_t arg14 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFrinti(arg14, kFpcrRModeZero), MakeF64x2(-6.00, -6.00));
  __uint128_t arg15 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFrinti(arg15, kFpcrRModeZero), MakeF64x2(6.00, 6.00));
  __uint128_t arg16 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFrinti(arg16, kFpcrRModeZero), MakeF64x2(6.00, 7.00));
}

TEST(Arm64InsnTest, AsmRoundExactF32) {
  constexpr auto AsmFrintx = ASM_INSN_WRAP_FUNC_W_RES_WC_ARG("frintx %s0, %s1");
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-7.50f), kFpcrRModeTieEven), bit_cast<uint32_t>(-8.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.75f), kFpcrRModeTieEven), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.50f), kFpcrRModeTieEven), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.25f), kFpcrRModeTieEven), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.25f), kFpcrRModeTieEven), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.50f), kFpcrRModeTieEven), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.75f), kFpcrRModeTieEven), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(7.50f), kFpcrRModeTieEven), bit_cast<uint32_t>(8.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-7.50f), kFpcrRModeNegInf), bit_cast<uint32_t>(-8.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.75f), kFpcrRModeNegInf), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.50f), kFpcrRModeNegInf), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.25f), kFpcrRModeNegInf), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.25f), kFpcrRModeNegInf), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.50f), kFpcrRModeNegInf), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.75f), kFpcrRModeNegInf), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(7.50f), kFpcrRModeNegInf), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-7.50f), kFpcrRModePosInf), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.75f), kFpcrRModePosInf), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.50f), kFpcrRModePosInf), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.25f), kFpcrRModePosInf), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.25f), kFpcrRModePosInf), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.50f), kFpcrRModePosInf), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.75f), kFpcrRModePosInf), bit_cast<uint32_t>(7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(7.50f), kFpcrRModePosInf), bit_cast<uint32_t>(8.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-7.50f), kFpcrRModeZero), bit_cast<uint32_t>(-7.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.75f), kFpcrRModeZero), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.50f), kFpcrRModeZero), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(-6.25f), kFpcrRModeZero), bit_cast<uint32_t>(-6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.25f), kFpcrRModeZero), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.50f), kFpcrRModeZero), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(6.75f), kFpcrRModeZero), bit_cast<uint32_t>(6.00f));
  ASSERT_EQ(AsmFrintx(bit_cast<uint32_t>(7.50f), kFpcrRModeZero), bit_cast<uint32_t>(7.00f));
}

TEST(Arm64InsnTest, AsmRoundExactF64) {
  constexpr auto AsmFrintx = ASM_INSN_WRAP_FUNC_W_RES_WC_ARG("frintx %d0, %d1");
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-7.50), kFpcrRModeTieEven), bit_cast<uint64_t>(-8.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.75), kFpcrRModeTieEven), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.50), kFpcrRModeTieEven), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.25), kFpcrRModeTieEven), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.25), kFpcrRModeTieEven), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.50), kFpcrRModeTieEven), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.75), kFpcrRModeTieEven), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(7.50), kFpcrRModeTieEven), bit_cast<uint64_t>(8.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-7.50), kFpcrRModeNegInf), bit_cast<uint64_t>(-8.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.75), kFpcrRModeNegInf), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.50), kFpcrRModeNegInf), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.25), kFpcrRModeNegInf), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.25), kFpcrRModeNegInf), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.50), kFpcrRModeNegInf), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.75), kFpcrRModeNegInf), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(7.50), kFpcrRModeNegInf), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-7.50), kFpcrRModePosInf), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.75), kFpcrRModePosInf), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.50), kFpcrRModePosInf), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.25), kFpcrRModePosInf), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.25), kFpcrRModePosInf), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.50), kFpcrRModePosInf), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.75), kFpcrRModePosInf), bit_cast<uint64_t>(7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(7.50), kFpcrRModePosInf), bit_cast<uint64_t>(8.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-7.50), kFpcrRModeZero), bit_cast<uint64_t>(-7.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.75), kFpcrRModeZero), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.50), kFpcrRModeZero), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(-6.25), kFpcrRModeZero), bit_cast<uint64_t>(-6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.25), kFpcrRModeZero), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.50), kFpcrRModeZero), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(6.75), kFpcrRModeZero), bit_cast<uint64_t>(6.00));
  ASSERT_EQ(AsmFrintx(bit_cast<uint64_t>(7.50), kFpcrRModeZero), bit_cast<uint64_t>(7.00));
}

TEST(Arm64InsnTest, AsmRoundExactF32x4) {
  constexpr auto AsmFrintx = ASM_INSN_WRAP_FUNC_W_RES_WC_ARG("frintx %0.4s, %1.4s");
  __uint128_t arg1 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFrintx(arg1, kFpcrRModeTieEven), MakeF32x4(-8.00f, -7.00f, -6.00f, -6.00f));
  __uint128_t arg2 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFrintx(arg2, kFpcrRModeTieEven), MakeF32x4(6.00f, 6.00f, 7.00f, 8.00f));
  __uint128_t arg3 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFrintx(arg3, kFpcrRModeNegInf), MakeF32x4(-8.00f, -7.00f, -7.00f, -7.00f));
  __uint128_t arg4 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFrintx(arg4, kFpcrRModeNegInf), MakeF32x4(6.00f, 6.00f, 6.00f, 7.00f));
  __uint128_t arg5 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFrintx(arg5, kFpcrRModePosInf), MakeF32x4(-7.00f, -6.00f, -6.00f, -6.00f));
  __uint128_t arg6 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFrintx(arg6, kFpcrRModePosInf), MakeF32x4(7.00f, 7.00f, 7.00f, 8.00f));
  __uint128_t arg7 = MakeF32x4(-7.50f, -6.75f, -6.50f, -6.25f);
  ASSERT_EQ(AsmFrintx(arg7, kFpcrRModeZero), MakeF32x4(-7.00f, -6.00f, -6.00f, -6.00f));
  __uint128_t arg8 = MakeF32x4(6.25f, 6.50f, 6.75f, 7.50f);
  ASSERT_EQ(AsmFrintx(arg8, kFpcrRModeZero), MakeF32x4(6.00f, 6.00f, 6.00f, 7.00f));
}

TEST(Arm64InsnTest, AsmRoundExactF64x2) {
  constexpr auto AsmFrintx = ASM_INSN_WRAP_FUNC_W_RES_WC_ARG("frintx %0.2d, %1.2d");
  __uint128_t arg1 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFrintx(arg1, kFpcrRModeTieEven), MakeF64x2(-8.00, -7.00));
  __uint128_t arg2 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFrintx(arg2, kFpcrRModeTieEven), MakeF64x2(-6.00, -6.00));
  __uint128_t arg3 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFrintx(arg3, kFpcrRModeTieEven), MakeF64x2(6.00, 6.00));
  __uint128_t arg4 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFrintx(arg4, kFpcrRModeTieEven), MakeF64x2(7.00, 8.00));
  __uint128_t arg5 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFrintx(arg5, kFpcrRModeNegInf), MakeF64x2(-8.00, -7.00));
  __uint128_t arg6 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFrintx(arg6, kFpcrRModeNegInf), MakeF64x2(-7.00, -7.00));
  __uint128_t arg7 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFrintx(arg7, kFpcrRModeNegInf), MakeF64x2(6.00, 6.00));
  __uint128_t arg8 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFrintx(arg8, kFpcrRModeNegInf), MakeF64x2(6.00, 7.00));
  __uint128_t arg9 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFrintx(arg9, kFpcrRModePosInf), MakeF64x2(-7.00, -6.00));
  __uint128_t arg10 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFrintx(arg10, kFpcrRModePosInf), MakeF64x2(-6.00, -6.00));
  __uint128_t arg11 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFrintx(arg11, kFpcrRModePosInf), MakeF64x2(7.00, 7.00));
  __uint128_t arg12 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFrintx(arg12, kFpcrRModePosInf), MakeF64x2(7.00, 8.00));
  __uint128_t arg13 = MakeF64x2(-7.50, -6.75);
  ASSERT_EQ(AsmFrintx(arg13, kFpcrRModeZero), MakeF64x2(-7.00, -6.00));
  __uint128_t arg14 = MakeF64x2(-6.50, -6.25);
  ASSERT_EQ(AsmFrintx(arg14, kFpcrRModeZero), MakeF64x2(-6.00, -6.00));
  __uint128_t arg15 = MakeF64x2(6.25, 6.50);
  ASSERT_EQ(AsmFrintx(arg15, kFpcrRModeZero), MakeF64x2(6.00, 6.00));
  __uint128_t arg16 = MakeF64x2(6.75, 7.50);
  ASSERT_EQ(AsmFrintx(arg16, kFpcrRModeZero), MakeF64x2(6.00, 7.00));
}

uint64_t Fp32Compare(uint64_t arg1, uint64_t arg2) {
  uint64_t res;
  asm("fcmp %s1, %s2\n\t"
      "mrs %x0, nzcv"
      : "=r"(res)
      : "w"(arg1), "w"(arg2));
  return res;
}

uint64_t Fp64Compare(uint64_t arg1, uint64_t arg2) {
  uint64_t res;
  asm("fcmp %d1, %d2\n\t"
      "mrs %x0, nzcv"
      : "=r"(res)
      : "w"(arg1), "w"(arg2));
  return res;
}

constexpr uint64_t MakeNZCV(uint64_t nzcv) {
  return nzcv << 28;
}

TEST(Arm64InsnTest, Fp32Compare) {
  // NaN and 1.83
  ASSERT_EQ(Fp32Compare(0x7fc0'0000ULL, 0x3fea'3d71ULL), MakeNZCV(0b0011));

  // 6.31 == 6.31
  ASSERT_EQ(Fp32Compare(0x40c9'eb85ULL, 0x40c9'eb85ULL), MakeNZCV(0b0110));

  // 1.23 < 2.34
  ASSERT_EQ(Fp32Compare(0x3f9d'70a4ULL, 0x4015'c28fULL), MakeNZCV(0b1000));

  // 5.25 > 2.94
  ASSERT_EQ(Fp32Compare(0x40a8'0000ULL, 0x403c'28f6ULL), MakeNZCV(0b0010));
}

TEST(Arm64InsnTest, Fp32CompareZero) {
  constexpr auto Fp32CompareZero = ASM_INSN_WRAP_FUNC_R_RES_W_ARG(
      "fcmp %s1, #0.0\n\t"
      "mrs %x0, nzcv");

  // NaN and 0.00
  ASSERT_EQ(Fp32CompareZero(0x7fa0'0000ULL), MakeNZCV(0b0011));

  // 0.00 == 0.00
  ASSERT_EQ(Fp32CompareZero(0x0000'0000ULL), MakeNZCV(0b0110));

  // -2.67 < 0.00
  ASSERT_EQ(Fp32CompareZero(0xc02a'e148ULL), MakeNZCV(0b1000));

  // 1.56 > 0.00
  ASSERT_EQ(Fp32CompareZero(0x3fc7'ae14ULL), MakeNZCV(0b0010));
}

TEST(Arm64InsnTest, Fp64Compare) {
  // NaN and 1.19
  ASSERT_EQ(Fp64Compare(0x7ff8'0000'0000'0000ULL, 0x3ff3'0a3d'70a3'd70aULL), MakeNZCV(0b0011));

  // 8.42 == 8.42
  ASSERT_EQ(Fp64Compare(0x4020'd70a'3d70'a3d7ULL, 0x4020'd70a'3d70'a3d7ULL), MakeNZCV(0b0110));

  // 0.50 < 1.00
  ASSERT_EQ(Fp64Compare(0x3fe0'0000'0000'0000ULL, 0x3ff0'0000'0000'0000ULL), MakeNZCV(0b1000));

  // 7.38 > 1.54
  ASSERT_EQ(Fp64Compare(0x401d'851e'b851'eb85ULL, 0x3ff8'a3d7'0a3d'70a4ULL), MakeNZCV(0b0010));
}

TEST(Arm64InsnTest, Fp64CompareZero) {
  constexpr auto Fp64CompareZero = ASM_INSN_WRAP_FUNC_R_RES_W_ARG(
      "fcmp %d1, #0.0\n\t"
      "mrs %x0, nzcv");

  // NaN and 0.00
  ASSERT_EQ(Fp64CompareZero(0x7ff4'0000'0000'0000ULL), MakeNZCV(0b0011));

  // 0.00 == 0.00
  ASSERT_EQ(Fp64CompareZero(0x0000'0000'0000'0000ULL), MakeNZCV(0b0110));

  // -7.23 < 0.00
  ASSERT_EQ(Fp64CompareZero(0xc01c'eb85'1eb8'51ecULL), MakeNZCV(0b1000));

  // 5.39 > 0.00
  ASSERT_EQ(Fp64CompareZero(0x4015'8f5c'28f5'c28fULL), MakeNZCV(0b0010));
}

uint64_t Fp32CompareIfEqualOrSetAllFlags(float arg1, float arg2, uint64_t nzcv) {
  asm("msr nzcv, %x0\n\t"
      "fccmp %s2, %s3, #15, eq\n\t"
      "mrs %x0, nzcv\n\t"
      : "=r"(nzcv)
      : "0"(nzcv), "w"(arg1), "w"(arg2));
  return nzcv;
}

TEST(Arm64InsnTest, Fp32ConditionalCompare) {
  // Comparison is performed.
  constexpr uint64_t kEqual = MakeNZCV(0b0100);
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  ASSERT_EQ(Fp32CompareIfEqualOrSetAllFlags(1.0f, 1.0f, kEqual), MakeNZCV(0b0110));
  ASSERT_EQ(Fp32CompareIfEqualOrSetAllFlags(1.0f, 2.0f, kEqual), MakeNZCV(0b1000));
  ASSERT_EQ(Fp32CompareIfEqualOrSetAllFlags(2.0f, 1.0f, kEqual), MakeNZCV(0b0010));
  ASSERT_EQ(Fp32CompareIfEqualOrSetAllFlags(kNan, 1.0f, kEqual), MakeNZCV(0b0011));
  // Comparison is not performed; alt-nzcv is returned.
  constexpr uint64_t kNotEqual = MakeNZCV(0b0000);
  ASSERT_EQ(Fp32CompareIfEqualOrSetAllFlags(1.0f, 1.0f, kNotEqual), MakeNZCV(0b1111));
  ASSERT_EQ(Fp32CompareIfEqualOrSetAllFlags(1.0f, 2.0f, kNotEqual), MakeNZCV(0b1111));
  ASSERT_EQ(Fp32CompareIfEqualOrSetAllFlags(2.0f, 1.0f, kNotEqual), MakeNZCV(0b1111));
  ASSERT_EQ(Fp32CompareIfEqualOrSetAllFlags(kNan, 1.0f, kNotEqual), MakeNZCV(0b1111));
}

uint64_t Fp64CompareIfEqualOrSetAllFlags(double arg1, double arg2, uint64_t nzcv) {
  asm("msr nzcv, %x0\n\t"
      "fccmp %d2, %d3, #15, eq\n\t"
      "mrs %x0, nzcv\n\t"
      : "=r"(nzcv)
      : "0"(nzcv), "w"(arg1), "w"(arg2));
  return nzcv;
}

TEST(Arm64InsnTest, Fp64ConditionalCompare) {
  // Comparison is performed.
  constexpr uint64_t kEqual = MakeNZCV(0b0100);
  constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
  ASSERT_EQ(Fp64CompareIfEqualOrSetAllFlags(1.0, 1.0, kEqual), MakeNZCV(0b0110));
  ASSERT_EQ(Fp64CompareIfEqualOrSetAllFlags(1.0, 2.0, kEqual), MakeNZCV(0b1000));
  ASSERT_EQ(Fp64CompareIfEqualOrSetAllFlags(2.0, 1.0, kEqual), MakeNZCV(0b0010));
  ASSERT_EQ(Fp64CompareIfEqualOrSetAllFlags(kNan, 1.0, kEqual), MakeNZCV(0b0011));
  // Comparison is not performed; alt-nzcv is returned.
  constexpr uint64_t kNotEqual = MakeNZCV(0b0000);
  ASSERT_EQ(Fp64CompareIfEqualOrSetAllFlags(1.0, 1.0, kNotEqual), MakeNZCV(0b1111));
  ASSERT_EQ(Fp64CompareIfEqualOrSetAllFlags(1.0, 2.0, kNotEqual), MakeNZCV(0b1111));
  ASSERT_EQ(Fp64CompareIfEqualOrSetAllFlags(2.0, 1.0, kNotEqual), MakeNZCV(0b1111));
  ASSERT_EQ(Fp64CompareIfEqualOrSetAllFlags(kNan, 1.0f, kNotEqual), MakeNZCV(0b1111));
}

TEST(Arm64InsnTest, ConvertFp32ToFp64) {
  uint64_t arg = 0x40cd'70a4ULL;  // 6.42 in float
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvt %d0, %s1")(arg);
  ASSERT_EQ(res, MakeUInt128(0x4019'ae14'8000'0000ULL, 0U));
}

TEST(Arm64InsnTest, ConvertFp64ToFp32) {
  uint64_t arg = 0x401a'0a3d'70a3'd70aULL;  // 6.51 in double
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvt %s0, %d1")(arg);
  ASSERT_EQ(res, MakeUInt128(0x40d0'51ecULL, 0U));
}

TEST(Arm64InsnTest, ConvertFp32ToFp16) {
  constexpr auto AsmFcvt = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvt %h0, %s1");
  EXPECT_EQ(AsmFcvt(bit_cast<uint32_t>(2.5f)), MakeUInt128(0x4100U, 0U));
  EXPECT_EQ(AsmFcvt(bit_cast<uint32_t>(4.5f)), MakeUInt128(0x4480U, 0U));
  EXPECT_EQ(AsmFcvt(bit_cast<uint32_t>(8.5f)), MakeUInt128(0x4840U, 0U));
  EXPECT_EQ(AsmFcvt(bit_cast<uint32_t>(16.5f)), MakeUInt128(0x4c20U, 0U));
}

TEST(Arm64InsnTest, ConvertFp16ToFp32) {
  uint64_t arg = 0x4100U;
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvt %s0, %h1")(arg);
  ASSERT_EQ(res, bit_cast<uint32_t>(2.5f));
}

TEST(Arm64InsnTest, ConvertFp64ToFp16) {
  uint64_t arg = bit_cast<uint64_t>(2.5);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvt %h0, %d1")(arg);
  ASSERT_EQ(res, MakeUInt128(0x4100U, 0U));
}

TEST(Arm64InsnTest, ConvertFp16ToFp64) {
  uint64_t arg = 0x4100U;
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvt %d0, %h1")(arg);
  ASSERT_EQ(res, bit_cast<uint64_t>(2.5));
}

TEST(Arm64InsnTest, ConvertToNarrowF64F32x2) {
  constexpr auto AsmFcvtn = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtn %0.2s, %1.2d");
  ASSERT_EQ(AsmFcvtn(MakeF64x2(2.0, 3.0)), MakeF32x4(2.0f, 3.0f, 0.0f, 0.0f));
  // Overflow or inf arguments result in inf.
  __uint128_t res = AsmFcvtn(
      MakeF64x2(std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity()));
  ASSERT_EQ(res,
            MakeF32x4(std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::infinity(),
                      0.0f,
                      0.0f));
  res = AsmFcvtn(
      MakeF64x2(std::numeric_limits<double>::lowest(), -std::numeric_limits<double>::infinity()));
  ASSERT_EQ(res,
            MakeF32x4(-std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(),
                      0.0f,
                      0.0f));
}

TEST(Arm64InsnTest, ConvertToNarrowF64F32x2Upper) {
  constexpr auto AsmFcvtn = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("fcvtn2 %0.4s, %1.2d");
  __uint128_t arg1 = MakeF64x2(2.0, 3.0);
  __uint128_t arg2 = MakeF32x4(4.0f, 5.0f, 6.0f, 7.0f);
  ASSERT_EQ(AsmFcvtn(arg1, arg2), MakeF32x4(4.0f, 5.0f, 2.0f, 3.0f));
}

TEST(Arm64InsnTest, ConvertToNarrowRoundToOddF64F32) {
  constexpr auto AsmFcvtxn = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtxn %s0, %d1");
  ASSERT_EQ(AsmFcvtxn(bit_cast<uint64_t>(2.0)), bit_cast<uint32_t>(2.0f));
  // Overflow is saturated.
  ASSERT_EQ(AsmFcvtxn(bit_cast<uint64_t>(std::numeric_limits<double>::max())),
            bit_cast<uint32_t>(std::numeric_limits<float>::max()));
  ASSERT_EQ(AsmFcvtxn(bit_cast<uint64_t>(std::numeric_limits<double>::lowest())),
            bit_cast<uint32_t>(std::numeric_limits<float>::lowest()));
  // inf is converted to inf.
  ASSERT_EQ(AsmFcvtxn(bit_cast<uint64_t>(std::numeric_limits<double>::infinity())),
            bit_cast<uint32_t>(std::numeric_limits<float>::infinity()));
  // -inf is converted to -inf.
  ASSERT_EQ(AsmFcvtxn(bit_cast<uint64_t>(-std::numeric_limits<double>::infinity())),
            bit_cast<uint32_t>(-std::numeric_limits<float>::infinity()));
}

TEST(Arm64InsnTest, ConvertToNarrowRoundToOddF64F32x2) {
  constexpr auto AsmFcvtxn = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtxn %0.2s, %1.2d");
  __uint128_t res = AsmFcvtxn(MakeF64x2(2.0, 3.0));
  ASSERT_EQ(res, MakeF32x4(2.0f, 3.0f, 0.0f, 0.0f));
}

TEST(Arm64InsnTest, ConvertToNarrowRoundToOddF64F32x2Upper) {
  constexpr auto AsmFcvtxn = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("fcvtxn2 %0.4s, %1.2d");
  __uint128_t arg1 = MakeF64x2(2.0, 3.0);
  __uint128_t arg2 = MakeF32x4(4.0f, 5.0f, 6.0f, 7.0f);
  ASSERT_EQ(AsmFcvtxn(arg1, arg2), MakeF32x4(4.0f, 5.0f, 2.0f, 3.0f));
}

TEST(Arm64InsnTest, ConvertToWiderF32F64x2Lower) {
  constexpr auto AsmFcvtl = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtl %0.2d, %1.2s");
  __uint128_t arg = MakeF32x4(2.0f, 3.0f, 4.0f, 5.0f);
  ASSERT_EQ(AsmFcvtl(arg), MakeF64x2(2.0, 3.0));
}

TEST(Arm64InsnTest, ConvertToWiderF32F64x2Upper) {
  constexpr auto AsmFcvtl2 = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtl2 %0.2d, %1.4s");
  __uint128_t arg = MakeF32x4(2.0f, 3.0f, 4.0f, 5.0f);
  ASSERT_EQ(AsmFcvtl2(arg), MakeF64x2(4.0, 5.0));
}

TEST(Arm64InsnTest, ConvertToWiderF16F32x4Lower) {
  constexpr auto AsmFcvtl = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtl %0.4s, %1.4h");
  // 4xF16 in the lower half.
  __uint128_t arg = MakeUInt128(0x4c20'4840'4480'4100ULL, 0);
  ASSERT_EQ(AsmFcvtl(arg), MakeF32x4(2.5f, 4.5f, 8.5f, 16.5f));
}

TEST(Arm64InsnTest, ConvertToWiderF16F32x4Upper) {
  constexpr auto AsmFcvtl = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtl2 %0.4s, %1.8h");
  // 4xF16 in the upper half.
  __uint128_t arg = MakeUInt128(0, 0x4c20'4840'4480'4100ULL);
  ASSERT_EQ(AsmFcvtl(arg), MakeF32x4(2.5f, 4.5f, 8.5f, 16.5f));
}

TEST(Arm64InsnTest, ConvertToNarrowF32F16x4Lower) {
  constexpr auto AsmFcvtn = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcvtn %0.4h, %1.4s");
  __uint128_t arg = MakeF32x4(2.5f, 4.5f, 8.5f, 16.5f);
  // 4xF16 in the lower half.
  ASSERT_EQ(AsmFcvtn(arg), MakeUInt128(0x4c20'4840'4480'4100ULL, 0));
}

TEST(Arm64InsnTest, ConvertToNarrowF32F16x4Upper) {
  constexpr auto AsmFcvtn = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("fcvtn2 %0.8h, %1.4s");
  __uint128_t arg1 = MakeF32x4(2.5f, 4.5f, 8.5f, 16.5f);
  __uint128_t arg2 = MakeF32x4(3.0f, 5.0f, 7.0f, 11.0f);
  // 4xF16 in the upper half, lower half preserved.
  ASSERT_EQ(AsmFcvtn(arg1, arg2), MakeUInt128(uint64_t(arg2), 0x4c20'4840'4480'4100ULL));
}

TEST(Arm64InsnTest, AbsF32) {
  uint32_t arg = 0xc127'3333U;  // -10.45 in float
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fabs %s0, %s1")(arg);
  ASSERT_EQ(res, MakeUInt128(0x4127'3333ULL, 0U));  // 10.45 in float
}

TEST(Arm64InsnTest, AbsF64) {
  uint64_t arg = 0xc03d'e8f5'c28f'5c29ULL;  // -29.91 in double
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fabs %d0, %d1")(arg);
  ASSERT_EQ(res, MakeUInt128(0x403d'e8f5'c28f'5c29ULL, 0U));  // 29.91 in double
}

TEST(Arm64InsnTest, AbsF32x4) {
  constexpr auto AsmFabs = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fabs %0.4s, %1.4s");
  __uint128_t arg = MakeF32x4(-0.0f, 0.0f, 3.0f, -7.0f);
  ASSERT_EQ(AsmFabs(arg), MakeF32x4(0.0f, 0.0f, 3.0f, 7.0f));
}

TEST(Arm64InsnTest, AbsF64x2) {
  constexpr auto AsmFabs = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fabs %0.2d, %1.2d");
  __uint128_t arg = MakeF64x2(-0.0, 3.0);
  ASSERT_EQ(AsmFabs(arg), MakeF64x2(0.0, 3.0));
}

TEST(Arm64InsnTest, AbdF32) {
  uint32_t arg1 = 0x4181'851fU;  // 16.19 in float
  uint32_t arg2 = 0x4121'1eb8U;  // 10.06 in float
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fabd %s0, %s1, %s2")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x40c3'd70cULL, 0U));  // 6.12 in float
}

TEST(Arm64InsnTest, AbdF64) {
  constexpr auto AsmFabd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fabd %d0, %d1, %d2");
  uint64_t arg1 = 0x4038'28f5'c28f'5c29U;  // 24.16 in double
  uint64_t arg2 = 0x4027'd70a'3d70'a3d7U;  // 11.92 in double
  __uint128_t res = AsmFabd(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x4028'7ae1'47ae'147bULL, 0U));  // 12.24 in double
}

TEST(Arm64InsnTest, AbdF32x4) {
  constexpr auto AsmFabd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fabd %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(1.0f, 5.0f, -3.0f, -2.0f);
  __uint128_t arg2 = MakeF32x4(-1.0f, 2.0f, -5.0f, 3.0f);
  __uint128_t res = AsmFabd(arg1, arg2);
  ASSERT_EQ(res, MakeF32x4(2.0f, 3.0f, 2.0f, 5.0f));
}

TEST(Arm64InsnTest, AbdF64x2) {
  constexpr auto AsmFabd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fabd %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(5.0, -2.0);
  __uint128_t arg2 = MakeF64x2(4.0, 3.0);
  __uint128_t res = AsmFabd(arg1, arg2);
  ASSERT_EQ(res, MakeF64x2(1.0, 5.0));
}

TEST(Arm64InsnTest, NegF32) {
  uint32_t arg = 0x40ee'b852U;  // 7.46 in float
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fneg %s0, %s1")(arg);
  ASSERT_EQ(res, MakeUInt128(0xc0ee'b852ULL, 0U));  // -7.46 in float
}

TEST(Arm64InsnTest, NegF64) {
  uint64_t arg = 0x4054'b28f'5c28'f5c3ULL;  // 82.79 in double
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fneg %d0, %d1")(arg);
  ASSERT_EQ(res, MakeUInt128(0xc054'b28f'5c28'f5c3ULL, 0U));  // -82.79 in double
}

TEST(Arm64InsnTest, NegF32x4) {
  constexpr auto AsmFneg = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fneg %0.4s, %1.4s");
  __uint128_t arg = MakeF32x4(-0.0f, 0.0f, 1.0f, -3.0f);
  ASSERT_EQ(AsmFneg(arg), MakeF32x4(0.0f, -0.0f, -1.0f, 3.0f));
}

TEST(Arm64InsnTest, NegF64x2) {
  constexpr auto AsmFneg = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fneg %0.2d, %1.2d");
  __uint128_t arg = MakeF64x2(0.0, 3.0);
  ASSERT_EQ(AsmFneg(arg), MakeF64x2(-0.0, -3.0));
}

TEST(Arm64InsnTest, SqrtF32) {
  uint32_t arg = 0x41f3'cac1U;  // 30.474 in float
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fsqrt %s0, %s1")(arg);
  ASSERT_EQ(res, MakeUInt128(0x40b0'a683ULL, 0U));  // 5.5203261 in float
}

TEST(Arm64InsnTest, SqrtF64) {
  uint64_t arg = 0x403d'4666'6666'6666ULL;  // 29.275 in double
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fsqrt %d0, %d1")(arg);
  ASSERT_EQ(res, MakeUInt128(0x4015'a47e'3392'efb8ULL, 0U));  // 5.41... in double
}

TEST(Arm64InsnTest, SqrtF32x4) {
  constexpr auto AsmSqrt = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fsqrt %0.4s, %1.4s");
  __uint128_t arg = MakeF32x4(0.0f, 1.0f, 4.0f, 9.0f);
  ASSERT_EQ(AsmSqrt(arg), MakeF32x4(0.0f, 1.0f, 2.0f, 3.0f));
}

TEST(Arm64InsnTest, RecipEstimateF32) {
  constexpr auto AsmFrecpe = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frecpe %s0, %s1");
  ASSERT_EQ(AsmFrecpe(bit_cast<uint32_t>(0.25f)), bit_cast<uint32_t>(3.9921875f));
  ASSERT_EQ(AsmFrecpe(bit_cast<uint32_t>(0.50f)), bit_cast<uint32_t>(1.99609375f));
  ASSERT_EQ(AsmFrecpe(bit_cast<uint32_t>(2.00f)), bit_cast<uint32_t>(0.4990234375f));
  ASSERT_EQ(AsmFrecpe(bit_cast<uint32_t>(4.00f)), bit_cast<uint32_t>(0.24951171875f));
}

TEST(Arm64InsnTest, RecipEstimateF32x4) {
  constexpr auto AsmFrecpe = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frecpe %0.4s, %1.4s");
  __uint128_t res = AsmFrecpe(MakeF32x4(0.25f, 0.50f, 2.00f, 4.00f));
  ASSERT_EQ(res, MakeF32x4(3.9921875f, 1.99609375f, 0.4990234375f, 0.24951171875f));
}

TEST(Arm64InsnTest, RecipStepF32) {
  constexpr auto AsmFrecps = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("frecps %s0, %s1, %s2");
  __uint128_t res1 = AsmFrecps(bit_cast<uint32_t>(1.50f), bit_cast<uint32_t>(0.50f));
  ASSERT_EQ(res1, bit_cast<uint32_t>(1.25f));
  __uint128_t res2 = AsmFrecps(bit_cast<uint32_t>(2.00f), bit_cast<uint32_t>(0.50f));
  ASSERT_EQ(res2, bit_cast<uint32_t>(1.00f));
  __uint128_t res3 = AsmFrecps(bit_cast<uint32_t>(3.00f), bit_cast<uint32_t>(0.25f));
  ASSERT_EQ(res3, bit_cast<uint32_t>(1.25f));
  __uint128_t res4 = AsmFrecps(bit_cast<uint32_t>(3.00f), bit_cast<uint32_t>(0.50f));
  ASSERT_EQ(res4, bit_cast<uint32_t>(0.50f));
}

TEST(Arm64InsnTest, RecipStepF64) {
  constexpr auto AsmFrecps = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("frecps %d0, %d1, %d2");
  __uint128_t res1 = AsmFrecps(bit_cast<uint64_t>(1.50), bit_cast<uint64_t>(0.50));
  ASSERT_EQ(res1, bit_cast<uint64_t>(1.25));
  __uint128_t res2 = AsmFrecps(bit_cast<uint64_t>(2.00), bit_cast<uint64_t>(0.50));
  ASSERT_EQ(res2, bit_cast<uint64_t>(1.00));
  __uint128_t res3 = AsmFrecps(bit_cast<uint64_t>(3.00), bit_cast<uint64_t>(0.25));
  ASSERT_EQ(res3, bit_cast<uint64_t>(1.25));
  __uint128_t res4 = AsmFrecps(bit_cast<uint64_t>(3.00), bit_cast<uint64_t>(0.50));
  ASSERT_EQ(res4, bit_cast<uint64_t>(0.50));
}

TEST(Arm64InsnTest, RecipStepF32x4) {
  constexpr auto AsmFrecps = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("frecps %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(1.50f, 2.00f, 3.00f, 3.00f);
  __uint128_t arg2 = MakeF32x4(0.50f, 0.50f, 0.25f, 0.50f);
  __uint128_t res = AsmFrecps(arg1, arg2);
  ASSERT_EQ(res, MakeF32x4(1.25f, 1.00f, 1.25f, 0.50f));
}

TEST(Arm64InsnTest, RecipStepF64x2) {
  constexpr auto AsmFrecps = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("frecps %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(1.50, 2.00);
  __uint128_t arg2 = MakeF64x2(0.50, 0.50);
  ASSERT_EQ(AsmFrecps(arg1, arg2), MakeF64x2(1.25, 1.00));
  __uint128_t arg3 = MakeF64x2(3.00, 3.00);
  __uint128_t arg4 = MakeF64x2(0.25, 0.50);
  ASSERT_EQ(AsmFrecps(arg3, arg4), MakeF64x2(1.25, 0.50));
}

TEST(Arm64InsnTest, RecipSqrtEstimateF32) {
  constexpr auto AsmFrsqrte = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frsqrte %s0, %s1");
  ASSERT_EQ(AsmFrsqrte(bit_cast<uint32_t>(2.0f)), bit_cast<uint32_t>(0.705078125f));
  ASSERT_EQ(AsmFrsqrte(bit_cast<uint32_t>(3.0f)), bit_cast<uint32_t>(0.576171875f));
  ASSERT_EQ(AsmFrsqrte(bit_cast<uint32_t>(4.0f)), bit_cast<uint32_t>(0.4990234375f));
  ASSERT_EQ(AsmFrsqrte(bit_cast<uint32_t>(5.0f)), bit_cast<uint32_t>(0.4462890625f));
}

TEST(Arm64InsnTest, RecipSqrtEstimateF32x2) {
  constexpr auto AsmFrsqrte = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frsqrte %0.2s, %1.2s");
  __uint128_t arg = MakeF32x4(2.0f, 3.0f, 0, 0);
  __uint128_t res = AsmFrsqrte(arg);
  ASSERT_EQ(res, MakeF32x4(0.705078125f, 0.576171875f, 0, 0));
}

TEST(Arm64InsnTest, RecipSqrtEstimateF32x4) {
  constexpr auto AsmFrsqrte = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frsqrte %0.4s, %1.4s");
  __uint128_t arg = MakeF32x4(2.0f, 3.0f, 4.0f, 5.0f);
  __uint128_t res = AsmFrsqrte(arg);
  ASSERT_EQ(res, MakeF32x4(0.705078125f, 0.576171875f, 0.4990234375f, 0.4462890625f));
}

TEST(Arm64InsnTest, RecipSqrtEstimateF64) {
  constexpr auto AsmFrsqrte = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frsqrte %d0, %d1");
  ASSERT_EQ(AsmFrsqrte(bit_cast<uint64_t>(2.0)), bit_cast<uint64_t>(0.705078125));
  ASSERT_EQ(AsmFrsqrte(bit_cast<uint64_t>(3.0)), bit_cast<uint64_t>(0.576171875));
  ASSERT_EQ(AsmFrsqrte(bit_cast<uint64_t>(4.0)), bit_cast<uint64_t>(0.4990234375));
  ASSERT_EQ(AsmFrsqrte(bit_cast<uint64_t>(5.0)), bit_cast<uint64_t>(0.4462890625));
}

TEST(Arm64InsnTest, RecipSqrtEstimateF64x2) {
  constexpr auto AsmFrsqrte = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("frsqrte %0.2d, %1.2d");
  __uint128_t arg = MakeF64x2(2.0, 3.0);
  __uint128_t res = AsmFrsqrte(arg);
  ASSERT_EQ(res, MakeUInt128(bit_cast<uint64_t>(0.705078125), bit_cast<uint64_t>(0.576171875)));
}

TEST(Arm64InsnTest, RecipSqrtStepF32) {
  constexpr auto AsmFrsqrts = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("frsqrts %s0, %s1, %s2");
  __uint128_t res1 = AsmFrsqrts(bit_cast<uint32_t>(1.50f), bit_cast<uint32_t>(0.50f));
  ASSERT_EQ(res1, bit_cast<uint32_t>(1.125f));
  __uint128_t res2 = AsmFrsqrts(bit_cast<uint32_t>(2.00f), bit_cast<uint32_t>(0.50f));
  ASSERT_EQ(res2, bit_cast<uint32_t>(1.000f));
  __uint128_t res3 = AsmFrsqrts(bit_cast<uint32_t>(3.00f), bit_cast<uint32_t>(0.25f));
  ASSERT_EQ(res3, bit_cast<uint32_t>(1.125f));
  __uint128_t res4 = AsmFrsqrts(bit_cast<uint32_t>(3.00f), bit_cast<uint32_t>(0.50f));
  ASSERT_EQ(res4, bit_cast<uint32_t>(0.750f));
}

TEST(Arm64InsnTest, RecipSqrtStepF64) {
  constexpr auto AsmFrsqrts = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("frsqrts %d0, %d1, %d2");
  __uint128_t res1 = AsmFrsqrts(bit_cast<uint64_t>(1.50), bit_cast<uint64_t>(0.50));
  ASSERT_EQ(res1, bit_cast<uint64_t>(1.125));
  __uint128_t res2 = AsmFrsqrts(bit_cast<uint64_t>(2.00), bit_cast<uint64_t>(0.50));
  ASSERT_EQ(res2, bit_cast<uint64_t>(1.000));
  __uint128_t res3 = AsmFrsqrts(bit_cast<uint64_t>(3.00), bit_cast<uint64_t>(0.25));
  ASSERT_EQ(res3, bit_cast<uint64_t>(1.125));
  __uint128_t res4 = AsmFrsqrts(bit_cast<uint64_t>(3.00), bit_cast<uint64_t>(0.50));
  ASSERT_EQ(res4, bit_cast<uint64_t>(0.750));
}

TEST(Arm64InsnTest, RecipSqrtStepF32x4) {
  constexpr auto AsmFrsqrts = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("frsqrts %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(1.50f, 2.00f, 3.00f, 3.00f);
  __uint128_t arg2 = MakeF32x4(0.50f, 0.50f, 0.25f, 0.50f);
  __uint128_t res = AsmFrsqrts(arg1, arg2);
  ASSERT_EQ(res, MakeF32x4(1.125f, 1.000f, 1.125f, 0.750f));
}

TEST(Arm64InsnTest, RecipSqrtStepF64x2) {
  constexpr auto AsmFrsqrts = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("frsqrts %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(1.50, 2.00);
  __uint128_t arg2 = MakeF64x2(0.50, 0.50);
  ASSERT_EQ(AsmFrsqrts(arg1, arg2), MakeF64x2(1.125, 1.000));
  __uint128_t arg3 = MakeF64x2(3.00, 3.00);
  __uint128_t arg4 = MakeF64x2(0.25, 0.50);
  ASSERT_EQ(AsmFrsqrts(arg3, arg4), MakeF64x2(1.125, 0.750));
}

TEST(Arm64InsnTest, AddFp32) {
  uint64_t fp_arg1 = 0x40d5'c28fULL;  // 6.68 in float
  uint64_t fp_arg2 = 0x409f'5c29ULL;  // 4.98 in float
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fadd %s0, %s1, %s2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd, MakeUInt128(0x413a'8f5cULL, 0U));  // 11.66 in float
}

TEST(Arm64InsnTest, AddFp64) {
  uint64_t fp_arg1 = 0x4020'9999'9999'999aULL;  // 8.30 in double
  uint64_t fp_arg2 = 0x4010'ae14'7ae1'47aeULL;  // 4.17 in double
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fadd %d0, %d1, %d2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd, MakeUInt128(0x4028'f0a3'd70a'3d71ULL, 0U));  // 12.47 in double
}

TEST(Arm64InsnTest, AddF32x4) {
  constexpr auto AsmFadd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fadd %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFadd(arg1, arg2), MakeF32x4(3.0f, 3.0f, -1.0f, 5.0f));
}

TEST(Arm64InsnTest, AddF64x2) {
  constexpr auto AsmFadd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fadd %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(3.0, 5.0);
  __uint128_t arg2 = MakeF64x2(-4.0, 2.0);
  ASSERT_EQ(AsmFadd(arg1, arg2), MakeF64x2(-1.0, 7.0));
}

TEST(Arm64InsnTest, AddPairwiseF32x2) {
  constexpr auto AsmFaddp = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("faddp %s0, %1.2s");
  __uint128_t arg1 = MakeF32x4(1.0f, 2.0f, 4.0f, 8.0f);
  ASSERT_EQ(AsmFaddp(arg1), bit_cast<uint32_t>(3.0f));
}

TEST(Arm64InsnTest, AddPairwiseF32x4) {
  constexpr auto AsmFaddp = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("faddp %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFaddp(arg1, arg2), MakeF32x4(-1.0f, 7.0f, 7.0f, -3.0f));
}

TEST(Arm64InsnTest, SubFp32) {
  uint64_t fp_arg1 = 0x411f'5c29ULL;  // 9.96 in float
  uint64_t fp_arg2 = 0x4048'51ecULL;  // 3.13 in float
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fsub %s0, %s1, %s2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd, MakeUInt128(0x40da'8f5cULL, 0U));  // 6.83 in float
}

TEST(Arm64InsnTest, SubFp64) {
  uint64_t fp_arg1 = 0x401e'e147'ae14'7ae1ULL;  // 7.72 in double
  uint64_t fp_arg2 = 0x4015'6666'6666'6666ULL;  // 5.35 in double
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fsub %d0, %d1, %d2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd, MakeUInt128(0x4002'f5c2'8f5c'28f6ULL, 0U));  // 2.37 in double
}

TEST(Arm64InsnTest, SubF32x4) {
  constexpr auto AsmFsub = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fsub %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFsub(arg1, arg2), MakeF32x4(-9.0f, 1.0f, 15.0f, -5.0f));
}

TEST(Arm64InsnTest, SubF64x2) {
  constexpr auto AsmFsub = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fsub %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(3.0, 5.0);
  __uint128_t arg2 = MakeF64x2(-4.0, 2.0);
  ASSERT_EQ(AsmFsub(arg1, arg2), MakeF64x2(7.0, 3.0));
}

TEST(Arm64InsnTest, MaxFp32) {
  constexpr auto AsmFmax = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmax %s0, %s1, %s2");
  uint32_t fp_arg_two = bit_cast<uint32_t>(2.0f);
  uint32_t fp_arg_three = bit_cast<uint32_t>(3.0f);

  ASSERT_EQ(AsmFmax(fp_arg_two, fp_arg_three), MakeU32x4(fp_arg_three, 0, 0, 0));
  ASSERT_EQ(AsmFmax(kDefaultNaN32AsInteger, fp_arg_three), kDefaultNaN32AsInteger);
  ASSERT_EQ(AsmFmax(fp_arg_three, kDefaultNaN32AsInteger), kDefaultNaN32AsInteger);
}

TEST(Arm64InsnTest, MaxFp64) {
  constexpr auto AsmFmax = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmax %d0, %d1, %d2");
  uint64_t fp_arg_two = bit_cast<uint64_t>(2.0);
  uint64_t fp_arg_three = bit_cast<uint64_t>(3.0);

  ASSERT_EQ(AsmFmax(fp_arg_two, fp_arg_three), MakeUInt128(fp_arg_three, 0U));
  ASSERT_EQ(AsmFmax(kDefaultNaN64AsInteger, fp_arg_three), kDefaultNaN64AsInteger);
  ASSERT_EQ(AsmFmax(fp_arg_three, kDefaultNaN64AsInteger), kDefaultNaN64AsInteger);
}

TEST(Arm64InsnTest, MaxF32x4) {
  constexpr auto AsmFmax = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmax %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-0.0f, 2.0f, 3.0f, -4.0f);
  __uint128_t arg2 = MakeF32x4(0.0f, 1.0f, -3.0f, -3.0f);
  ASSERT_EQ(AsmFmax(arg1, arg2), MakeF32x4(0.0f, 2.0f, 3.0f, -3.0f));

  __uint128_t arg3 = MakeF32x4(-0.0f, bit_cast<float>(kDefaultNaN32AsInteger), 3.0f, -4.0f);
  __uint128_t arg4 = MakeF32x4(0.0f, 1.0f, -3.0f, bit_cast<float>(kDefaultNaN32AsInteger));
  ASSERT_EQ(AsmFmax(arg3, arg4),
            MakeF32x4(0.0f,
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      3.0f,
                      bit_cast<float>(kDefaultNaN32AsInteger)));
}

TEST(Arm64InsnTest, MaxF64x2) {
  constexpr auto AsmFmax = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmax %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(-0.0, 3.0);
  __uint128_t arg2 = MakeF64x2(0.0, -3.0);
  ASSERT_EQ(AsmFmax(arg1, arg2), MakeF64x2(0.0, 3.0));

  __uint128_t arg3 = MakeF64x2(bit_cast<double>(kDefaultNaN64AsInteger), 3.0);
  __uint128_t arg4 = MakeF64x2(1.0, bit_cast<double>(kDefaultNaN64AsInteger));
  ASSERT_EQ(AsmFmax(arg3, arg4),
            MakeF64x2(bit_cast<double>(kDefaultNaN64AsInteger),
                      bit_cast<double>(kDefaultNaN64AsInteger)));
}

TEST(Arm64InsnTest, MaxNumberFp32) {
  constexpr auto AsmFmaxnm = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmaxnm %s0, %s1, %s2");
  uint32_t fp_arg_two = bit_cast<uint32_t>(2.0f);
  uint32_t fp_arg_three = bit_cast<uint32_t>(3.0f);
  uint64_t fp_arg_minus_two = bit_cast<uint64_t>(-2.0);

  ASSERT_EQ(AsmFmaxnm(fp_arg_two, fp_arg_three), MakeU32x4(fp_arg_three, 0, 0, 0));

  ASSERT_EQ(AsmFmaxnm(fp_arg_two, kQuietNaN32AsInteger), MakeU32x4(fp_arg_two, 0, 0, 0));
  ASSERT_EQ(AsmFmaxnm(fp_arg_minus_two, kQuietNaN32AsInteger),
            MakeU32x4(fp_arg_minus_two, 0, 0, 0));
  ASSERT_EQ(AsmFmaxnm(kQuietNaN32AsInteger, fp_arg_two), MakeU32x4(fp_arg_two, 0, 0, 0));
  ASSERT_EQ(AsmFmaxnm(kQuietNaN32AsInteger, fp_arg_minus_two),
            MakeU32x4(fp_arg_minus_two, 0, 0, 0));
}

TEST(Arm64InsnTest, MaxNumberFp64) {
  constexpr auto AsmFmaxnm = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmaxnm %d0, %d1, %d2");
  uint64_t fp_arg_two = bit_cast<uint64_t>(2.0);
  uint64_t fp_arg_three = bit_cast<uint64_t>(3.0);
  uint64_t fp_arg_minus_two = bit_cast<uint64_t>(-2.0);

  ASSERT_EQ(AsmFmaxnm(fp_arg_two, fp_arg_three), MakeUInt128(fp_arg_three, 0U));

  ASSERT_EQ(AsmFmaxnm(fp_arg_two, kQuietNaN64AsInteger), MakeUInt128(fp_arg_two, 0U));
  ASSERT_EQ(AsmFmaxnm(fp_arg_minus_two, kQuietNaN64AsInteger), MakeUInt128(fp_arg_minus_two, 0));
  ASSERT_EQ(AsmFmaxnm(kQuietNaN64AsInteger, fp_arg_two), MakeUInt128(fp_arg_two, 0));
  ASSERT_EQ(AsmFmaxnm(kQuietNaN64AsInteger, fp_arg_minus_two), MakeUInt128(fp_arg_minus_two, 0));
}

TEST(Arm64InsnTest, MinNumberFp32) {
  constexpr auto AsmFminnm = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fminnm %s0, %s1, %s2");
  uint32_t fp_arg_two = bit_cast<uint32_t>(2.0f);
  uint32_t fp_arg_three = bit_cast<uint32_t>(3.0f);
  uint32_t fp_arg_minus_two = bit_cast<uint32_t>(-2.0f);

  ASSERT_EQ(AsmFminnm(fp_arg_two, fp_arg_three), MakeU32x4(fp_arg_two, 0, 0, 0));

  ASSERT_EQ(AsmFminnm(fp_arg_two, kQuietNaN32AsInteger), MakeU32x4(fp_arg_two, 0, 0, 0));
  ASSERT_EQ(AsmFminnm(fp_arg_minus_two, kQuietNaN32AsInteger),
            MakeU32x4(fp_arg_minus_two, 0, 0, 0));
  ASSERT_EQ(AsmFminnm(kQuietNaN32AsInteger, fp_arg_two), MakeU32x4(fp_arg_two, 0, 0, 0));
  ASSERT_EQ(AsmFminnm(kQuietNaN32AsInteger, fp_arg_minus_two),
            MakeU32x4(fp_arg_minus_two, 0, 0, 0));
}

TEST(Arm64InsnTest, MinNumberFp64) {
  constexpr auto AsmFminnm = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fminnm %d0, %d1, %d2");
  uint64_t fp_arg_two = bit_cast<uint64_t>(2.0);
  uint64_t fp_arg_three = bit_cast<uint64_t>(3.0);
  uint64_t fp_arg_minus_two = bit_cast<uint64_t>(-2.0);

  ASSERT_EQ(AsmFminnm(fp_arg_two, fp_arg_three), MakeUInt128(fp_arg_two, 0U));

  ASSERT_EQ(AsmFminnm(fp_arg_two, kQuietNaN64AsInteger), MakeUInt128(fp_arg_two, 0U));
  ASSERT_EQ(AsmFminnm(fp_arg_minus_two, kQuietNaN64AsInteger), MakeUInt128(fp_arg_minus_two, 0));
  ASSERT_EQ(AsmFminnm(kQuietNaN64AsInteger, fp_arg_two), MakeUInt128(fp_arg_two, 0));
  ASSERT_EQ(AsmFminnm(kQuietNaN64AsInteger, fp_arg_minus_two), MakeUInt128(fp_arg_minus_two, 0));
}

TEST(Arm64InsnTest, MaxNumberF32x4) {
  constexpr auto AsmFmaxnm = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmaxnm %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-1.0f, 2.0f, 3.0f, -4.0f);
  __uint128_t arg2 = MakeF32x4(2.0f, 1.0f, -3.0f, -3.0f);
  ASSERT_EQ(AsmFmaxnm(arg1, arg2), MakeF32x4(2.0f, 2.0f, 3.0f, -3.0f));

  __uint128_t arg3 = MakeU32x4(bit_cast<uint32_t>(1.0f),
                               bit_cast<uint32_t>(-1.0f),
                               kNegativeQuietNaN32AsInteger,
                               kQuietNaN32AsInteger);
  __uint128_t arg4 = MakeU32x4(kNegativeQuietNaN32AsInteger,
                               kQuietNaN32AsInteger,
                               bit_cast<uint32_t>(1.0f),
                               bit_cast<uint32_t>(-1.0f));
  ASSERT_EQ(AsmFmaxnm(arg3, arg4), MakeF32x4(1.0f, -1.0f, 1.0f, -1.0f));

  __uint128_t arg5 = MakeU32x4(bit_cast<uint32_t>(1.0f),
                               bit_cast<uint32_t>(-1.0f),
                               kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger);
  __uint128_t arg6 = MakeU32x4(kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger,
                               bit_cast<uint32_t>(1.0f),
                               bit_cast<uint32_t>(-1.0f));
  ASSERT_EQ(AsmFmaxnm(arg5, arg6),
            MakeF32x4(bit_cast<float>(kDefaultNaN32AsInteger),
                      -1.0f,
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      -1.0f));

  __uint128_t arg7 = MakeU32x4(kSignalingNaN32AsInteger_1,
                               kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger,
                               kQuietNaN32AsInteger);
  __uint128_t arg8 = MakeU32x4(kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger,
                               kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger);
  ASSERT_EQ(AsmFmaxnm(arg7, arg8),
            MakeF32x4(bit_cast<float>(kDefaultNaN32AsInteger),
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      bit_cast<float>(kDefaultNaN32AsInteger)));

  __uint128_t arg9 = MakeF32x4(-0.0f, -0.0f, 0.0f, 0.0f);
  __uint128_t arg10 = MakeF32x4(-0.0f, 0.0f, -0.0f, 0.0f);
  ASSERT_EQ(AsmFmaxnm(arg9, arg10), MakeF32x4(-0.0f, 0.0f, 0.0f, 0.0f));
}

TEST(Arm64InsnTest, MaxNumberF64x2) {
  constexpr auto AsmFmaxnm = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmaxnm %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(-1.0, -4.0);
  __uint128_t arg2 = MakeF64x2(2.0, -3.0);
  ASSERT_EQ(AsmFmaxnm(arg1, arg2), MakeF64x2(2.0, -3.0));

  __uint128_t arg3 = MakeUInt128(bit_cast<uint64_t>(1.0), kQuietNaN64AsInteger);
  __uint128_t arg4 = MakeUInt128(kQuietNaN64AsInteger, bit_cast<uint64_t>(-1.0));
  ASSERT_EQ(AsmFmaxnm(arg3, arg4), MakeF64x2(1.0, -1.0));
}

TEST(Arm64InsnTest, MinNumberF32x4) {
  constexpr auto AsmFminnm = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fminnm %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-1.0f, 2.0f, 3.0f, -4.0f);
  __uint128_t arg2 = MakeF32x4(2.0f, 1.0f, -3.0f, -3.0f);
  ASSERT_EQ(AsmFminnm(arg1, arg2), MakeF32x4(-1.0f, 1.0f, -3.0f, -4.0f));

  __uint128_t arg3 = MakeU32x4(bit_cast<uint32_t>(1.0f),
                               bit_cast<uint32_t>(-1.0f),
                               kNegativeQuietNaN32AsInteger,
                               kQuietNaN32AsInteger);
  __uint128_t arg4 = MakeU32x4(kNegativeQuietNaN32AsInteger,
                               kQuietNaN32AsInteger,
                               bit_cast<uint32_t>(1.0f),
                               bit_cast<uint32_t>(-1.0f));
  ASSERT_EQ(AsmFminnm(arg3, arg4), MakeF32x4(1.0f, -1.0f, 1.0f, -1.0f));

  __uint128_t arg5 = MakeU32x4(bit_cast<uint32_t>(1.0f),
                               bit_cast<uint32_t>(-1.0f),
                               kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger);
  __uint128_t arg6 = MakeU32x4(kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger,
                               bit_cast<uint32_t>(1.0f),
                               bit_cast<uint32_t>(-1.0f));
  ASSERT_EQ(AsmFminnm(arg5, arg6),
            MakeF32x4(bit_cast<float>(kDefaultNaN32AsInteger),
                      -1.0f,
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      -1.0f));

  __uint128_t arg7 = MakeU32x4(kSignalingNaN32AsInteger_1,
                               kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger,
                               kQuietNaN32AsInteger);
  __uint128_t arg8 = MakeU32x4(kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger,
                               kSignalingNaN32AsInteger_1,
                               kQuietNaN32AsInteger);
  ASSERT_EQ(AsmFminnm(arg7, arg8),
            MakeF32x4(bit_cast<float>(kDefaultNaN32AsInteger),
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      bit_cast<float>(kDefaultNaN32AsInteger)));

  __uint128_t arg9 = MakeF32x4(-0.0f, -0.0f, 0.0f, 0.0f);
  __uint128_t arg10 = MakeF32x4(-0.0f, 0.0f, -0.0f, 0.0f);
  ASSERT_EQ(AsmFminnm(arg9, arg10), MakeF32x4(-0.0f, -0.0f, -0.0f, 0.0f));
}

TEST(Arm64InsnTest, MinNumberF64x2) {
  constexpr auto AsmFminnm = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fminnm %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(0.0, 3.0);
  __uint128_t arg2 = MakeF64x2(-0.0, -3.0);
  ASSERT_EQ(AsmFminnm(arg1, arg2), MakeF64x2(-0.0, -3.0));

  __uint128_t arg3 = MakeUInt128(bit_cast<uint64_t>(1.0), kQuietNaN64AsInteger);
  __uint128_t arg4 = MakeUInt128(kQuietNaN64AsInteger, bit_cast<uint64_t>(-1.0));
  __uint128_t res = AsmFminnm(arg3, arg4);
  ASSERT_EQ(res, MakeF64x2(1.0, -1.0));
}

TEST(Arm64InsnTest, MinFp32) {
  constexpr auto AsmFmin = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmin %s0, %s1, %s2");
  uint32_t fp_arg_two = bit_cast<uint32_t>(2.0f);
  uint32_t fp_arg_three = bit_cast<uint32_t>(3.0f);

  ASSERT_EQ(AsmFmin(fp_arg_two, fp_arg_three), MakeU32x4(fp_arg_two, 0, 0, 0));
  ASSERT_EQ(AsmFmin(kDefaultNaN32AsInteger, fp_arg_three), kDefaultNaN32AsInteger);
  ASSERT_EQ(AsmFmin(fp_arg_three, kDefaultNaN32AsInteger), kDefaultNaN32AsInteger);
}

TEST(Arm64InsnTest, MinFp64) {
  constexpr auto AsmFmin = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmin %d0, %d1, %d2");
  uint64_t fp_arg_two = bit_cast<uint64_t>(2.0);
  uint64_t fp_arg_three = bit_cast<uint64_t>(3.0);

  ASSERT_EQ(AsmFmin(fp_arg_two, fp_arg_three), MakeUInt128(fp_arg_two, 0U));
  ASSERT_EQ(AsmFmin(kDefaultNaN64AsInteger, fp_arg_three), kDefaultNaN64AsInteger);
  ASSERT_EQ(AsmFmin(fp_arg_three, kDefaultNaN64AsInteger), kDefaultNaN64AsInteger);
}

TEST(Arm64InsnTest, MinF32x4) {
  constexpr auto AsmFmin = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmin %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(0.0f, 2.0f, 3.0f, -4.0f);
  __uint128_t arg2 = MakeF32x4(-0.0f, 1.0f, -3.0f, -3.0f);
  ASSERT_EQ(AsmFmin(arg1, arg2), MakeF32x4(-0.0f, 1.0f, -3.0f, -4.0f));

  __uint128_t arg3 = MakeF32x4(-0.0f, bit_cast<float>(kDefaultNaN32AsInteger), 3.0f, -4.0f);
  __uint128_t arg4 = MakeF32x4(0.0f, 1.0f, -3.0f, bit_cast<float>(kDefaultNaN32AsInteger));
  ASSERT_EQ(AsmFmin(arg3, arg4),
            MakeF32x4(-0.0f,
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      -3.0f,
                      bit_cast<float>(kDefaultNaN32AsInteger)));
}

TEST(Arm64InsnTest, MinF64x2) {
  constexpr auto AsmFmin = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmin %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(0.0, 3.0);
  __uint128_t arg2 = MakeF64x2(-0.0, -3.0);
  ASSERT_EQ(AsmFmin(arg1, arg2), MakeF64x2(-0.0, -3.0));

  __uint128_t arg3 = MakeF64x2(bit_cast<double>(kDefaultNaN64AsInteger), 3.0);
  __uint128_t arg4 = MakeF64x2(1.0, bit_cast<double>(kDefaultNaN64AsInteger));
  ASSERT_EQ(AsmFmin(arg3, arg4),
            MakeF64x2(bit_cast<double>(kDefaultNaN64AsInteger),
                      bit_cast<double>(kDefaultNaN64AsInteger)));
}

TEST(Arm64InsnTest, MaxPairwiseF32Scalar) {
  constexpr auto AsmFmaxp = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fmaxp %s0, %1.2s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  ASSERT_EQ(AsmFmaxp(arg1), bit_cast<uint32_t>(2.0f));

  __uint128_t arg2 = MakeF32x4(bit_cast<float>(kDefaultNaN32AsInteger), 2.0f, 7.0f, -0.0f);
  ASSERT_EQ(AsmFmaxp(arg2), kDefaultNaN32AsInteger);
}

TEST(Arm64InsnTest, MaxPairwiseF32x4) {
  constexpr auto AsmFmaxp = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmaxp %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFmaxp(arg1, arg2), MakeF32x4(2.0f, 7.0f, 6.0f, 5.0f));

  __uint128_t arg3 = MakeF32x4(
      bit_cast<float>(kDefaultNaN32AsInteger), 2.0f, 7.0f, bit_cast<float>(kDefaultNaN32AsInteger));
  __uint128_t arg4 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFmaxp(arg3, arg4),
            MakeF32x4(bit_cast<float>(kDefaultNaN32AsInteger),
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      6.0f,
                      5.0f));
}

TEST(Arm64InsnTest, MinPairwiseF32Scalar) {
  constexpr auto AsmFminp = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fminp %s0, %1.2s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  ASSERT_EQ(AsmFminp(arg1), bit_cast<uint32_t>(-3.0f));

  __uint128_t arg2 = MakeF32x4(bit_cast<float>(kDefaultNaN32AsInteger), 2.0f, 7.0f, -0.0f);
  ASSERT_EQ(AsmFminp(arg2), kDefaultNaN32AsInteger);
}

TEST(Arm64InsnTest, MinPairwiseF32x4) {
  constexpr auto AsmFminp = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fminp %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFminp(arg1, arg2), MakeF32x4(-3.0f, -0.0f, 1.0f, -8.0f));

  __uint128_t arg3 = MakeF32x4(
      bit_cast<float>(kDefaultNaN32AsInteger), 2.0f, 7.0f, bit_cast<float>(kDefaultNaN32AsInteger));
  __uint128_t arg4 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFminp(arg3, arg4),
            MakeF32x4(bit_cast<float>(kDefaultNaN32AsInteger),
                      bit_cast<float>(kDefaultNaN32AsInteger),
                      1.0f,
                      -8.0f));
}

TEST(Arm64InsnTest, MaxPairwiseNumberF32Scalar) {
  constexpr auto AsmFmaxnmp = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fmaxnmp %s0, %1.2s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  ASSERT_EQ(AsmFmaxnmp(arg1), bit_cast<uint32_t>(2.0f));

  __uint128_t arg2 = MakeF32x4(bit_cast<float>(kQuietNaN32AsInteger), 2.0f, 7.0f, -0.0f);
  ASSERT_EQ(AsmFmaxnmp(arg2), bit_cast<uint32_t>(2.0f));
}

TEST(Arm64InsnTest, MaxPairwiseNumberF32x4) {
  constexpr auto AsmFmaxnmp = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmaxnmp %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFmaxnmp(arg1, arg2), MakeF32x4(2.0f, 7.0f, 6.0f, 5.0f));

  __uint128_t arg3 = MakeF32x4(
      bit_cast<float>(kQuietNaN32AsInteger), 2.0f, 7.0f, bit_cast<float>(kQuietNaN32AsInteger));
  __uint128_t arg4 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFmaxnmp(arg3, arg4), MakeF32x4(2.0f, 7.0f, 6.0f, 5.0f));
}

TEST(Arm64InsnTest, MinPairwiseNumberF32Scalar) {
  constexpr auto AsmFminnmp = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fminnmp %s0, %1.2s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  ASSERT_EQ(AsmFminnmp(arg1), bit_cast<uint32_t>(-3.0f));

  __uint128_t arg2 = MakeF32x4(bit_cast<float>(kQuietNaN32AsInteger), 2.0f, 7.0f, -0.0f);
  ASSERT_EQ(AsmFminnmp(arg2), bit_cast<uint32_t>(2.0f));
}

TEST(Arm64InsnTest, MinPairwiseNumberF32x4) {
  constexpr auto AsmFminnmp = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fminnmp %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFminnmp(arg1, arg2), MakeF32x4(-3.0f, -0.0f, 1.0f, -8.0f));

  __uint128_t arg3 = MakeF32x4(
      bit_cast<float>(kQuietNaN32AsInteger), 2.0f, 7.0f, bit_cast<float>(kQuietNaN32AsInteger));
  __uint128_t arg4 = MakeF32x4(6.0f, 1.0f, -8.0f, 5.0f);
  ASSERT_EQ(AsmFminnmp(arg3, arg4), MakeF32x4(2.0f, 7.0f, 1.0f, -8.0f));
}

TEST(Arm64InsnTest, MaxAcrossF32x4) {
  constexpr auto AsmFmaxv = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fmaxv %s0, %1.4s");
  __uint128_t arg1 = MakeF32x4(0.0f, 2.0f, 3.0f, -4.0f);
  ASSERT_EQ(AsmFmaxv(arg1), bit_cast<uint32_t>(3.0f));

  __uint128_t arg2 = MakeF32x4(0.0f, 2.0f, bit_cast<float>(kDefaultNaN32AsInteger), -4.0f);
  ASSERT_EQ(AsmFmaxv(arg2), kDefaultNaN32AsInteger);
}

TEST(Arm64InsnTest, MinAcrossF32x4) {
  constexpr auto AsmFminv = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fminv %s0, %1.4s");
  __uint128_t arg1 = MakeF32x4(0.0f, 2.0f, 3.0f, -4.0f);
  ASSERT_EQ(AsmFminv(arg1), bit_cast<uint32_t>(-4.0f));

  __uint128_t arg2 = MakeF32x4(0.0f, 2.0f, bit_cast<float>(kDefaultNaN32AsInteger), -4.0f);
  ASSERT_EQ(AsmFminv(arg2), kDefaultNaN32AsInteger);
}

TEST(Arm64InsnTest, MaxNumberAcrossF32x4) {
  constexpr auto AsmFmaxnmv = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fmaxnmv %s0, %1.4s");
  __uint128_t arg1 = MakeF32x4(0.0f, 2.0f, 3.0f, -4.0f);
  ASSERT_EQ(AsmFmaxnmv(arg1), bit_cast<uint32_t>(3.0f));

  __uint128_t arg2 = MakeF32x4(0.0f, bit_cast<float>(kQuietNaN32AsInteger), 3.0f, -4.0f);
  ASSERT_EQ(AsmFmaxnmv(arg2), bit_cast<uint32_t>(3.0f));
}

TEST(Arm64InsnTest, MinNumberAcrossF32x4) {
  constexpr auto AsmFminnmv = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fminnmv %s0, %1.4s");
  __uint128_t arg1 = MakeF32x4(0.0f, 2.0f, 3.0f, -4.0f);
  ASSERT_EQ(AsmFminnmv(arg1), bit_cast<uint32_t>(-4.0f));

  __uint128_t arg2 = MakeF32x4(0.0f, bit_cast<float>(kQuietNaN32AsInteger), 3.0f, -4.0f);
  ASSERT_EQ(AsmFminnmv(arg2), bit_cast<uint32_t>(-4.0f));
}

TEST(Arm64InsnTest, MulFp32) {
  uint64_t fp_arg1 = 0x40a1'999aULL;  // 5.05 in float
  uint64_t fp_arg2 = 0x40da'e148ULL;  // 6.84 in float
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmul %s0, %s1, %s2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd, MakeUInt128(0x420a'2b03ULL, 0U));  // 34.5420 in float
}

TEST(Arm64InsnTest, MulFp64) {
  uint64_t fp_arg1 = 0x4022'6b85'1eb8'51ecULL;  // 9.21 in double
  uint64_t fp_arg2 = 0x4020'c7ae'147a'e148ULL;  // 8.39 in double
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmul %d0, %d1, %d2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd, MakeUInt128(0x4053'5166'cf41'f214ULL, 0U));  // 77.2719 in double
}

TEST(Arm64InsnTest, MulF32x4) {
  constexpr auto AsmFmul = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmul %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(1.0f, -2.0f, 3.0f, -4.0f);
  __uint128_t arg2 = MakeF32x4(-3.0f, -1.0f, 4.0f, 1.0f);
  ASSERT_EQ(AsmFmul(arg1, arg2), MakeF32x4(-3.0f, 2.0f, 12.0f, -4.0f));
}

TEST(Arm64InsnTest, MulF64x2) {
  constexpr auto AsmFmul = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmul %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(-4.0, 2.0);
  __uint128_t arg2 = MakeF64x2(2.0, 3.0);
  ASSERT_EQ(AsmFmul(arg1, arg2), MakeF64x2(-8.0, 6.0));
}

TEST(Arm64InsnTest, MulF32x4ByScalar) {
  __uint128_t arg1 = MakeF32x4(2.0f, 3.0f, 4.0f, 5.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 7.0f, 8.0f, 9.0f);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmul %0.4s, %1.4s, %2.s[3]")(arg1, arg2);
  ASSERT_EQ(res, MakeF32x4(18.0f, 27.0f, 36.0f, 45.0f));
}

TEST(Arm64InsnTest, MulF64x2ByScalar) {
  __uint128_t arg1 = MakeF64x2(2.0, 3.0);
  __uint128_t arg2 = MakeF64x2(5.0, 4.0);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmul %0.2d, %1.2d, %2.d[1]")(arg1, arg2);
  ASSERT_EQ(res, MakeF64x2(8.0, 12.0));
}

TEST(Arm64InsnTest, MulF32IndexedElem) {
  constexpr auto AsmFmul = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmul %s0, %s1, %2.s[2]");
  __uint128_t arg1 = MakeF32x4(2.0f, 3.0f, 5.0f, 7.0f);
  __uint128_t arg2 = MakeF32x4(11.0f, 13.0f, 17.0f, 19.0f);
  ASSERT_EQ(AsmFmul(arg1, arg2), bit_cast<uint32_t>(34.0f));
}

TEST(Arm64InsnTest, MulF64IndexedElem) {
  constexpr auto AsmFmul = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmul %d0, %d1, %2.d[1]");
  __uint128_t arg1 = MakeF64x2(2.0, 3.0);
  __uint128_t arg2 = MakeF64x2(5.0, 4.0);
  ASSERT_EQ(AsmFmul(arg1, arg2), bit_cast<uint64_t>(8.0));
}

TEST(Arm64InsnTest, MulExtendedF32) {
  constexpr auto AsmFmulx = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmulx %s0, %s1, %s2");
  __uint128_t arg1 = MakeF32x4(2.0f, 3.0f, 5.0f, 7.0f);
  __uint128_t arg2 = MakeF32x4(11.0f, 13.0f, 17.0f, 19.0f);
  ASSERT_EQ(AsmFmulx(arg1, arg2), bit_cast<uint32_t>(22.0f));
}

TEST(Arm64InsnTest, MulExtendedF32x4) {
  constexpr auto AsmFmulx = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmulx %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(2.0f, 3.0f, 5.0f, 7.0f);
  __uint128_t arg2 = MakeF32x4(11.0f, 13.0f, 17.0f, 19.0f);
  ASSERT_EQ(AsmFmulx(arg1, arg2), MakeF32x4(22.0f, 39.0f, 85.0f, 133.0f));
}

TEST(Arm64InsnTest, MulExtendedF32IndexedElem) {
  constexpr auto AsmFmulx = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmulx %s0, %s1, %2.s[2]");
  __uint128_t arg1 = MakeF32x4(2.0f, 3.0f, 5.0f, 7.0f);
  __uint128_t arg2 = MakeF32x4(11.0f, 13.0f, 17.0f, 19.0f);
  ASSERT_EQ(AsmFmulx(arg1, arg2), bit_cast<uint32_t>(34.0f));
}

TEST(Arm64InsnTest, MulExtendedF64IndexedElem) {
  constexpr auto AsmFmulx = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmulx %d0, %d1, %2.d[1]");
  __uint128_t arg1 = MakeF64x2(2.0, 3.0);
  __uint128_t arg2 = MakeF64x2(5.0, 4.0);
  ASSERT_EQ(AsmFmulx(arg1, arg2), bit_cast<uint64_t>(8.0));
}

TEST(Arm64InsnTest, MulExtendedF32x4IndexedElem) {
  constexpr auto AsmFmulx = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fmulx %0.4s, %1.4s, %2.s[2]");
  __uint128_t arg1 = MakeF32x4(2.0f, 3.0f, 5.0f, 7.0f);
  __uint128_t arg2 = MakeF32x4(11.0f, 13.0f, 17.0f, 19.0f);
  ASSERT_EQ(AsmFmulx(arg1, arg2), MakeF32x4(34.0f, 51.0f, 85.0f, 119.0f));
}

TEST(Arm64InsnTest, MulNegFp32) {
  uint64_t fp_arg1 = bit_cast<uint32_t>(2.0f);
  uint64_t fp_arg2 = bit_cast<uint32_t>(3.0f);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fnmul %s0, %s1, %s2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd, MakeUInt128(bit_cast<uint32_t>(-6.0f), 0U));
}

TEST(Arm64InsnTest, MulNegFp64) {
  uint64_t fp_arg1 = bit_cast<uint64_t>(2.0);
  uint64_t fp_arg2 = bit_cast<uint64_t>(3.0);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fnmul %d0, %d1, %d2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd, MakeUInt128(bit_cast<uint64_t>(-6.0), 0U));
}

TEST(Arm64InsnTest, DivFp32) {
  constexpr auto AsmFdiv = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fdiv %s0, %s1, %s2");

  uint32_t arg1 = 0x40c2'3d71U;                                     // 6.07 in float
  uint32_t arg2 = 0x401a'3d71U;                                     // 2.41 in float
  ASSERT_EQ(AsmFdiv(arg1, arg2), MakeUInt128(0x4021'31edULL, 0U));  // 2.5186722 in float

  // Make sure that FDIV can produce a denormal result under the default FPCR,
  // where the FZ bit (flush-to-zero) is off.
  uint32_t arg3 = 0xa876'eff9U;  // exponent (without offset) = -47
  uint32_t arg4 = 0xe7d8'6b60U;  // exponent (without offset) = 80
  ASSERT_EQ(AsmFdiv(arg3, arg4), MakeUInt128(0x0049'065cULL, 0U));  // denormal
}

TEST(Arm64InsnTest, DivFp64) {
  uint64_t fp_arg1 = 0x401e'5c28'f5c2'8f5cULL;  // 7.59 in double
  uint64_t fp_arg2 = 0x3ff2'8f5c'28f5'c28fULL;  // 1.16 in double
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fdiv %d0, %d1, %d2")(fp_arg1, fp_arg2);
  ASSERT_EQ(rd,
            MakeUInt128(0x401a'2c23'4f72'c235ULL, 0U));  // 6.5431034482758620995923593 in double
}

TEST(Arm64InsnTest, DivFp32_FlagsWhenDivByZero) {
  uint64_t fpsr;
  volatile float dividend = 123.0f;
  volatile float divisor = 0.0f;
  float res;
  asm volatile(
      "msr fpsr, xzr\n\t"
      "fdiv %s1, %s2, %s3\n\t"
      "mrs %0, fpsr"
      : "=r"(fpsr), "=w"(res)
      : "w"(dividend), "w"(divisor));
  ASSERT_TRUE((fpsr & kFpsrDzcBit) == (kFpsrDzcBit));

  // Previous bug caused IOC to be set upon scalar div by zero.
  ASSERT_TRUE((fpsr & kFpsrIocBit) == 0);
}

TEST(Arm64InsnTest, DivFp64_FlagsWhenDivByZero) {
  uint64_t fpsr;
  double res;
  asm volatile(
      "msr fpsr, xzr\n\t"
      "fdiv %d1, %d2, %d3\n\t"
      "mrs %0, fpsr"
      : "=r"(fpsr), "=w"(res)
      : "w"(123.0), "w"(0.0));
  ASSERT_TRUE((fpsr & kFpsrDzcBit) == (kFpsrDzcBit));

  // Previous bug caused IOC to be set upon scalar div by zero.
  ASSERT_TRUE((fpsr & kFpsrIocBit) == 0);
}

TEST(Arm64InsnTest, DivFp32x4) {
  constexpr auto AsmFdiv = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fdiv %0.4s, %1.4s, %2.4s");

  // 16.39, 80.286, 41.16, 98.01
  __uint128_t arg1 = MakeUInt128(0x4183'1eb8'42a0'926fULL, 0x4224'a3d7'42c4'051fULL);
  // 13.3, 45.45, 7.89, -2.63
  __uint128_t arg2 = MakeUInt128(0x4154'cccd'4235'cccdULL, 0x40fc'7ae1'c028'51ecULL);
  __uint128_t res1 = AsmFdiv(arg1, arg2);
  // 1.2323308, 1.7664686, 5.21673, -37.26616
  ASSERT_EQ(res1, MakeUInt128(0x3f9d'bd04'3fe2'1ba5ULL, 0x40a6'ef74'c215'108cULL));

  // Verify that fdiv produces a denormal result under the default FPCR.
  __uint128_t arg3 = MakeF32x4(1.0f, 1.0f, 1.0f, -0x1.eddff2p-47f);
  __uint128_t arg4 = MakeF32x4(1.0f, 1.0f, 1.0f, -0x1.b0d6c0p80f);
  __uint128_t res2 = AsmFdiv(arg3, arg4);
  __uint128_t expected2 = MakeF32x4(1.0f, 1.0f, 1.0f, 0x0.920cb8p-126f);
  ASSERT_EQ(res2, expected2);
}

TEST(Arm64InsnTest, DivFp64x2) {
  // 6.23, 65.02
  __uint128_t arg1 = MakeUInt128(0x4018'eb85'1eb8'51ecULL, 0x4050'4147'ae14'7ae1ULL);
  // -7.54, 11.92
  __uint128_t arg2 = MakeUInt128(0xc01e'28f5'c28f'5c29ULL, 0x4027'd70a'3d70'a3d7ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fdiv %0.2d, %1.2d, %2.2d")(arg1, arg2);
  // -0.82625994695, 5.45469798658
  ASSERT_EQ(res, MakeUInt128(0xbfea'70b8'b344'9564ULL, 0x4015'd19c'5957'9fc9ULL));
}

TEST(Arm64InsnTest, MulAddFp32) {
  constexpr auto AsmFmadd = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fmadd %s0, %s1, %s2, %s3");

  __uint128_t res1 =
      AsmFmadd(bit_cast<uint32_t>(2.0f), bit_cast<uint32_t>(3.0f), bit_cast<uint32_t>(5.0f));
  ASSERT_EQ(res1, MakeF32x4(11.0f, 0, 0, 0));

  __uint128_t res2 =
      AsmFmadd(bit_cast<uint32_t>(2.5f), bit_cast<uint32_t>(2.0f), bit_cast<uint32_t>(-5.0f));
  ASSERT_EQ(res2, MakeF32x4(0, 0, 0, 0));

  // These tests verify that fmadd does not lose precision while doing the mult + add.
  __uint128_t res3 = AsmFmadd(bit_cast<uint32_t>(0x1.fffffep22f),
                              bit_cast<uint32_t>(0x1.000002p0f),
                              bit_cast<uint32_t>(-0x1.p23f));
  ASSERT_EQ(res3, MakeF32x4(0x1.fffffcp-2f, 0, 0, 0));

  __uint128_t res4 = AsmFmadd(bit_cast<uint32_t>(0x1.fffffep22f),
                              bit_cast<uint32_t>(0x1.000002p0f),
                              bit_cast<uint32_t>(-0x1.fffffep22f));
  ASSERT_EQ(res4, MakeF32x4(0x1.fffffep-1f, 0, 0, 0));

  __uint128_t res5 = AsmFmadd(bit_cast<uint32_t>(0x1.p23f),
                              bit_cast<uint32_t>(0x1.fffffep-1f),
                              bit_cast<uint32_t>(-0x1.000002p23f));
  ASSERT_EQ(res5, MakeF32x4(-0x1.80p0f, 0, 0, 0));
}

TEST(Arm64InsnTest, MulAddFp64) {
  uint64_t arg1 = 0x4032'3d70'a3d7'0a3dULL;  // 18.24
  uint64_t arg2 = 0x4050'4147'ae14'7ae1ULL;  // 65.02
  uint64_t arg3 = 0x4027'd70a'3d70'a3d7ULL;  // 11.92
  __uint128_t res1 = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fmadd %d0, %d1, %d2, %d3")(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x4092'b78a'0902'de00ULL, 0U));  // 1197.8848
  __uint128_t res2 =
      ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fnmadd %d0, %d1, %d2, %d3")(arg1, arg2, arg3);
  ASSERT_EQ(res2, MakeUInt128(0xc092'b78a'0902'de00ULL, 0U));  // -1197.8848
}

TEST(Arm64InsnTest, MulAddFp64Precision) {
  uint64_t arg1 = bit_cast<uint64_t>(0x1.0p1023);
  uint64_t arg2 = bit_cast<uint64_t>(0x1.0p-1);
  uint64_t arg3 = bit_cast<uint64_t>(0x1.fffffffffffffp1022);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fmadd %d0, %d1, %d2, %d3")(arg1, arg2, arg3);
  ASSERT_EQ(res, bit_cast<uint64_t>(0x1.7ffffffffffff8p1023));
}

TEST(Arm64InsnTest, NegMulAddFp32) {
  constexpr auto AsmFnmadd = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fnmadd %s0, %s1, %s2, %s3");

  __uint128_t res1 =
      AsmFnmadd(bit_cast<uint32_t>(2.0f), bit_cast<uint32_t>(3.0f), bit_cast<uint32_t>(5.0f));
  ASSERT_EQ(res1, MakeF32x4(-11.0f, 0, 0, 0));

  // No -0 (proper negation)
  __uint128_t res2 =
      AsmFnmadd(bit_cast<uint32_t>(2.5f), bit_cast<uint32_t>(2.0f), bit_cast<uint32_t>(-5.0f));
  ASSERT_EQ(res2, MakeF32x4(0.0f, 0, 0, 0));

  // These tests verify that fmadd does not lose precision while doing the mult + add.
  __uint128_t res3 = AsmFnmadd(bit_cast<uint32_t>(0x1.fffffep22f),
                               bit_cast<uint32_t>(0x1.000002p0f),
                               bit_cast<uint32_t>(-0x1.p23f));
  ASSERT_EQ(res3, MakeF32x4(-0x1.fffffcp-2f, 0, 0, 0));

  __uint128_t res4 = AsmFnmadd(bit_cast<uint32_t>(0x1.fffffep22f),
                               bit_cast<uint32_t>(0x1.000002p0f),
                               bit_cast<uint32_t>(-0x1.fffffep22f));
  ASSERT_EQ(res4, MakeF32x4(-0x1.fffffep-1f, 0, 0, 0));

  __uint128_t res5 = AsmFnmadd(bit_cast<uint32_t>(0x1.p23f),
                               bit_cast<uint32_t>(0x1.fffffep-1f),
                               bit_cast<uint32_t>(-0x1.000002p23f));
  ASSERT_EQ(res5, MakeF32x4(0x1.80p0f, 0, 0, 0));
}

TEST(Arm64InsnTest, NegMulAddFp64) {
  constexpr auto AsmFnmadd = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fnmadd %d0, %d1, %d2, %d3");

  __uint128_t res1 =
      AsmFnmadd(bit_cast<uint64_t>(2.0), bit_cast<uint64_t>(3.0), bit_cast<uint64_t>(5.0));
  ASSERT_EQ(res1, MakeF64x2(-11.0, 0));

  // Proper negation (no -0 in this case)
  __uint128_t res2 =
      AsmFnmadd(bit_cast<uint64_t>(2.5), bit_cast<uint64_t>(2.0), bit_cast<uint64_t>(-5.0));
  ASSERT_EQ(res2, MakeF64x2(0.0, 0));
}

TEST(Arm64InsnTest, NegMulSubFp64) {
  constexpr auto AsmFnmsub = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fnmsub %d0, %d1, %d2, %d3");

  __uint128_t res1 =
      AsmFnmsub(bit_cast<uint64_t>(-2.0), bit_cast<uint64_t>(3.0), bit_cast<uint64_t>(5.0));
  ASSERT_EQ(res1, MakeF64x2(-11.0, 0));

  uint64_t arg1 = 0x4035'7ae1'47ae'147bULL;  // 21.48
  uint64_t arg2 = 0x404c'e3d7'0a3d'70a4ULL;  // 57.78
  uint64_t arg3 = 0x405e'2999'9999'999aULL;  // 120.65
  __uint128_t res2 = AsmFnmsub(arg1, arg2, arg3);
  ASSERT_EQ(res2, MakeUInt128(0x4091'81db'8bac'710dULL, 0U));  // 1120.4644

  // Assert no -0 in this case
  __uint128_t res3 =
      AsmFnmsub(bit_cast<uint64_t>(2.5), bit_cast<uint64_t>(2.0), bit_cast<uint64_t>(5.0));
  ASSERT_EQ(res3, MakeF64x2(0.0, 0));
}

TEST(Arm64InsnTest, NegMulSubFp64Precision) {
  constexpr auto AsmFnmsub = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fnmsub %d0, %d1, %d2, %d3");

  __uint128_t res = AsmFnmsub(bit_cast<uint64_t>(0x1.0p1023),
                              bit_cast<uint64_t>(0x1.0p-1),
                              bit_cast<uint64_t>(-0x1.fffffffffffffp1022));
  ASSERT_EQ(res, bit_cast<uint64_t>(0x1.7ffffffffffff8p1023));
}

TEST(Arm64InsnTest, MulAddF32x4) {
  constexpr auto AsmFmla = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmla %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(1.0f, 2.0f, 4.0f, 3.0f);
  __uint128_t arg2 = MakeF32x4(3.0f, 1.0f, 2.0f, 4.0f);
  __uint128_t arg3 = MakeF32x4(2.0f, 3.0f, 1.0f, 2.0f);
  ASSERT_EQ(AsmFmla(arg1, arg2, arg3), MakeF32x4(5.0f, 5.0f, 9.0f, 14.0f));
}

TEST(Arm64InsnTest, MulAddF32IndexedElem) {
  constexpr auto AsmFmla = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmla %s0, %s1, %2.s[2]");
  __uint128_t arg1 = MakeF32x4(1.0f, 2.0f, 4.0f, 3.0f);
  __uint128_t arg2 = MakeF32x4(3.0f, 1.0f, 2.0f, 4.0f);
  __uint128_t arg3 = MakeF32x4(2.0f, 3.0f, 1.0f, 2.0f);
  // 2 + (1 * 2)
  ASSERT_EQ(AsmFmla(arg1, arg2, arg3), bit_cast<uint32_t>(4.0f));
}

TEST(Arm64InsnTest, MulAddF64IndexedElem) {
  constexpr auto AsmFmla = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmla %d0, %d1, %2.d[1]");
  __uint128_t arg1 = MakeF64x2(2.0, 3.0);
  __uint128_t arg2 = MakeF64x2(4.0, 5.0);
  __uint128_t arg3 = MakeF64x2(6.0, 7.0);
  // 6 + (2 * 5)
  ASSERT_EQ(AsmFmla(arg1, arg2, arg3), bit_cast<uint64_t>(16.0));
}

TEST(Arm64InsnTest, MulAddF64x2) {
  constexpr auto AsmFmla = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmla %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(1.0f, 2.0f);
  __uint128_t arg2 = MakeF64x2(3.0f, 1.0f);
  __uint128_t arg3 = MakeF64x2(2.0f, 3.0f);
  ASSERT_EQ(AsmFmla(arg1, arg2, arg3), MakeF64x2(5.0f, 5.0f));
}

TEST(Arm64InsnTest, MulAddF32x4IndexedElem) {
  constexpr auto AsmFmla = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmla %0.4s, %1.4s, %2.s[2]");
  __uint128_t arg1 = MakeF32x4(1.0f, 2.0f, 4.0f, 3.0f);
  __uint128_t arg2 = MakeF32x4(3.0f, 1.0f, 2.0f, 4.0f);
  __uint128_t arg3 = MakeF32x4(2.0f, 3.0f, 1.0f, 2.0f);
  ASSERT_EQ(AsmFmla(arg1, arg2, arg3), MakeF32x4(4.0f, 7.0f, 9.0f, 8.0f));
}

TEST(Arm64InsnTest, MulSubFp32) {
  uint32_t arg1 = bit_cast<uint32_t>(2.0f);
  uint32_t arg2 = bit_cast<uint32_t>(5.0f);
  uint32_t arg3 = bit_cast<uint32_t>(3.0f);
  __uint128_t res1 = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fmsub %s0, %s1, %s2, %s3")(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(bit_cast<uint32_t>(-7.0f), 0U));
  __uint128_t res2 =
      ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fnmsub %s0, %s1, %s2, %s3")(arg1, arg2, arg3);
  ASSERT_EQ(res2, MakeUInt128(bit_cast<uint32_t>(7.0f), 0U));
}

TEST(Arm64InsnTest, MulSubFp64) {
  constexpr auto AsmFmsub = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fmsub %d0, %d1, %d2, %d3");

  uint64_t arg1 = 0x4035'7ae1'47ae'147bULL;  // 21.48
  uint64_t arg2 = 0x404c'e3d7'0a3d'70a4ULL;  // 57.78
  uint64_t arg3 = 0x405e'2999'9999'999aULL;  // 120.65
  __uint128_t res1 = AsmFmsub(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0xc091'81db'8bac'710dULL, 0U));  // -1120.4644

  // Basic case
  __uint128_t res3 =
      AsmFmsub(bit_cast<uint64_t>(2.0), bit_cast<uint64_t>(3.0), bit_cast<uint64_t>(-5.0));
  ASSERT_EQ(res3, MakeF64x2(-11.0, 0));

  // No -0 in this case (proper negation order)
  __uint128_t res4 =
      AsmFmsub(bit_cast<uint64_t>(2.5), bit_cast<uint64_t>(2.0), bit_cast<uint64_t>(5.0));
  ASSERT_EQ(res4, MakeF64x2(0.0, 0));
}

TEST(Arm64InsnTest, MulSubFp64Precision) {
  constexpr auto AsmFmsub = ASM_INSN_WRAP_FUNC_W_RES_WWW_ARG("fmsub %d0, %d1, %d2, %d3");
  __uint128_t res5 = AsmFmsub(bit_cast<uint64_t>(-0x1.0p1023),
                              bit_cast<uint64_t>(0x1.0p-1),
                              bit_cast<uint64_t>(0x1.fffffffffffffp1022));
  ASSERT_EQ(res5, bit_cast<uint64_t>(0x1.7ffffffffffff8p1023));
}

TEST(Arm64InsnTest, MulSubF32x4) {
  constexpr auto AsmFmls = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmls %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(1.0f, 2.0f, 4.0f, 3.0f);
  __uint128_t arg2 = MakeF32x4(3.0f, 1.0f, 2.0f, 4.0f);
  __uint128_t arg3 = MakeF32x4(2.0f, 3.0f, 1.0f, 2.0f);
  ASSERT_EQ(AsmFmls(arg1, arg2, arg3), MakeF32x4(-1.0f, 1.0f, -7.0f, -10.0f));
}

TEST(Arm64InsnTest, MulSubF32IndexedElem) {
  constexpr auto AsmFmls = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmls %s0, %s1, %2.s[2]");
  __uint128_t arg1 = MakeF32x4(2.0f, 1.0f, 4.0f, 3.0f);
  __uint128_t arg2 = MakeF32x4(4.0f, 3.0f, 2.0f, 1.0f);
  __uint128_t arg3 = MakeF32x4(8.0f, 3.0f, 1.0f, 2.0f);
  // 8 - (2 * 2)
  ASSERT_EQ(AsmFmls(arg1, arg2, arg3), bit_cast<uint32_t>(4.0f));
}

TEST(Arm64InsnTest, MulSubF32x4IndexedElem) {
  constexpr auto AsmFmls = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmls %0.4s, %1.4s, %2.s[2]");
  __uint128_t arg1 = MakeF32x4(1.0f, 2.0f, 4.0f, 3.0f);
  __uint128_t arg2 = MakeF32x4(3.0f, 1.0f, 2.0f, 4.0f);
  __uint128_t arg3 = MakeF32x4(2.0f, 3.0f, 1.0f, 2.0f);
  ASSERT_EQ(AsmFmls(arg1, arg2, arg3), MakeF32x4(0.0f, -1.0f, -7.0f, -4.0f));
}

TEST(Arm64InsnTest, MulSubF64x2) {
  constexpr auto AsmFmls = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmls %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(1.0f, 2.0f);
  __uint128_t arg2 = MakeF64x2(3.0f, 1.0f);
  __uint128_t arg3 = MakeF64x2(2.0f, 3.0f);
  ASSERT_EQ(AsmFmls(arg1, arg2, arg3), MakeF64x2(-1.0f, 1.0f));
}

TEST(Arm64InsnTest, MulSubF64IndexedElem) {
  constexpr auto AsmFmls = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("fmls %d0, %d1, %2.d[1]");
  __uint128_t arg1 = MakeF64x2(2.0, 5.0);
  __uint128_t arg2 = MakeF64x2(4.0, 1.0);
  __uint128_t arg3 = MakeF64x2(6.0, 7.0f);
  // 6 - (2 * 1)
  ASSERT_EQ(AsmFmls(arg1, arg2, arg3), bit_cast<uint64_t>(4.0));
}

TEST(Arm64InsnTest, CompareEqualF32) {
  constexpr auto AsmFcmeq = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmeq %s0, %s1, %s2");
  uint32_t two = bit_cast<uint32_t>(2.0f);
  uint32_t six = bit_cast<uint32_t>(6.0f);
  ASSERT_EQ(AsmFcmeq(two, six), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmeq(two, two), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmeq(kDefaultNaN32AsInteger, two), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmeq(two, kDefaultNaN32AsInteger), 0x0000'0000ULL);
}

TEST(Arm64InsnTest, CompareEqualF32x4) {
  constexpr auto AsmFcmeq = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmeq %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 2.0f, -8.0f, 5.0f);
  __uint128_t res = AsmFcmeq(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterEqualF32) {
  constexpr auto AsmFcmge = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmge %s0, %s1, %s2");
  uint32_t two = bit_cast<uint32_t>(2.0f);
  uint32_t six = bit_cast<uint32_t>(6.0f);
  ASSERT_EQ(AsmFcmge(two, six), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmge(two, two), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmge(six, two), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmge(kDefaultNaN32AsInteger, two), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmge(two, kDefaultNaN32AsInteger), 0x0000'0000ULL);
}

TEST(Arm64InsnTest, CompareGreaterEqualF32x4) {
  constexpr auto AsmFcmge = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmge %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 2.0f, -8.0f, 5.0f);
  __uint128_t res = AsmFcmge(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'0000'0000ULL, 0x0000'0000'ffff'ffffULL));
}

TEST(Arm64InsnTest, CompareGreaterF32) {
  constexpr auto AsmFcmgt = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmgt %s0, %s1, %s2");
  uint32_t two = bit_cast<uint32_t>(2.0f);
  uint32_t six = bit_cast<uint32_t>(6.0f);
  ASSERT_EQ(AsmFcmgt(two, six), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmgt(two, two), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmgt(six, two), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmgt(kDefaultNaN32AsInteger, two), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmgt(two, kDefaultNaN32AsInteger), 0x0000'0000ULL);
}

TEST(Arm64InsnTest, CompareGreaterF32x4) {
  constexpr auto AsmFcmgt = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmgt %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 2.0f, 7.0f, -0.0f);
  __uint128_t arg2 = MakeF32x4(6.0f, 2.0f, -8.0f, 5.0f);
  __uint128_t res = AsmFcmgt(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'ffff'ffffULL));
}

TEST(Arm64InsnTest, CompareEqualZeroF32) {
  constexpr auto AsmFcmeq = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmeq %s0, %s1, #0");
  ASSERT_EQ(AsmFcmeq(bit_cast<uint32_t>(0.0f)), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmeq(bit_cast<uint32_t>(4.0f)), 0x0000'0000ULL);
}

TEST(Arm64InsnTest, CompareEqualZeroF32x4) {
  constexpr auto AsmFcmeq = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmeq %0.4s, %1.4s, #0");
  __uint128_t arg = MakeF32x4(-3.0f, 0.0f, 7.0f, 1.0f);
  __uint128_t res = AsmFcmeq(arg);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterThanZeroF32) {
  constexpr auto AsmFcmgt = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmgt %s0, %s1, #0");
  ASSERT_EQ(AsmFcmgt(bit_cast<uint32_t>(-1.0f)), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmgt(bit_cast<uint32_t>(0.0f)), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmgt(bit_cast<uint32_t>(1.0f)), 0xffff'ffffULL);
}

TEST(Arm64InsnTest, CompareGreaterThanZeroF32x4) {
  constexpr auto AsmFcmgt = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmgt %0.4s, %1.4s, #0");
  __uint128_t arg = MakeF32x4(-3.0f, 0.0f, 7.0f, 1.0f);
  __uint128_t res = AsmFcmgt(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0xffff'ffff'ffff'ffffULL));
}

TEST(Arm64InsnTest, CompareGreaterThanOrEqualZeroF32) {
  constexpr auto AsmFcmge = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmge %s0, %s1, #0");
  ASSERT_EQ(AsmFcmge(bit_cast<uint32_t>(-1.0f)), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmge(bit_cast<uint32_t>(0.0f)), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmge(bit_cast<uint32_t>(1.0f)), 0xffff'ffffULL);
}

TEST(Arm64InsnTest, CompareGreaterThanOrEqualZeroF32x4) {
  constexpr auto AsmFcmge = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmge %0.4s, %1.4s, #0");
  __uint128_t arg = MakeF32x4(-3.0f, 0.0f, 7.0f, 1.0f);
  __uint128_t res = AsmFcmge(arg);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'0000'0000ULL, 0xffff'ffff'ffff'ffffULL));
}

TEST(Arm64InsnTest, CompareLessThanZeroF32) {
  constexpr auto AsmFcmlt = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmlt %s0, %s1, #0");
  ASSERT_EQ(AsmFcmlt(bit_cast<uint32_t>(-1.0f)), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmlt(bit_cast<uint32_t>(0.0f)), 0x0000'0000ULL);
  ASSERT_EQ(AsmFcmlt(bit_cast<uint32_t>(1.0f)), 0x0000'0000ULL);
}

TEST(Arm64InsnTest, CompareLessThanZeroF32x4) {
  constexpr auto AsmFcmlt = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmlt %0.4s, %1.4s, #0");
  __uint128_t arg = MakeF32x4(-3.0f, 0.0f, 7.0f, 1.0f);
  __uint128_t res = AsmFcmlt(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareLessThanOrEqualZeroF32) {
  constexpr auto AsmFcmle = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmle %s0, %s1, #0");
  ASSERT_EQ(AsmFcmle(bit_cast<uint32_t>(-1.0f)), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmle(bit_cast<uint32_t>(0.0f)), 0xffff'ffffULL);
  ASSERT_EQ(AsmFcmle(bit_cast<uint32_t>(1.0f)), 0x0000'0000ULL);
}

TEST(Arm64InsnTest, CompareLessThanOrEqualZeroF32x4) {
  constexpr auto AsmFcmle = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("fcmle %0.4s, %1.4s, #0");
  __uint128_t arg = MakeF32x4(-3.0f, 0.0f, 7.0f, 1.0f);
  __uint128_t res = AsmFcmle(arg);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AbsoluteCompareGreaterThanF32) {
  constexpr auto AsmFacgt = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("facgt %s0, %s1, %s2");
  ASSERT_EQ(AsmFacgt(bit_cast<uint32_t>(-3.0f), bit_cast<uint32_t>(1.0f)), 0xffff'ffffULL);
  ASSERT_EQ(AsmFacgt(bit_cast<uint32_t>(1.0f), bit_cast<uint32_t>(-1.0f)), 0x0000'0000ULL);
  ASSERT_EQ(AsmFacgt(bit_cast<uint32_t>(3.0f), bit_cast<uint32_t>(-7.0f)), 0x0000'0000ULL);
}

TEST(Arm64InsnTest, AbsoluteCompareGreaterThanOrEqualF32) {
  constexpr auto AsmFacge = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("facge %s0, %s1, %s2");
  ASSERT_EQ(AsmFacge(bit_cast<uint32_t>(-3.0f), bit_cast<uint32_t>(1.0f)), 0xffff'ffffULL);
  ASSERT_EQ(AsmFacge(bit_cast<uint32_t>(1.0f), bit_cast<uint32_t>(-1.0f)), 0xffff'ffffULL);
  ASSERT_EQ(AsmFacge(bit_cast<uint32_t>(3.0f), bit_cast<uint32_t>(-7.0f)), 0x0000'0000ULL);
}

TEST(Arm64InsnTest, AbsoluteCompareGreaterThanF32x4) {
  constexpr auto AsmFacgt = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("facgt %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 1.0f, 3.0f, 4.0f);
  __uint128_t arg2 = MakeF32x4(1.0f, -1.0f, -7.0f, 2.0f);
  ASSERT_EQ(AsmFacgt(arg1, arg2), MakeUInt128(0x0000'0000'ffff'ffffULL, 0xffff'ffff'0000'0000ULL));
}

TEST(Arm64InsnTest, AbsoluteCompareGreaterThanEqualF32x4) {
  constexpr auto AsmFacge = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("facge %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeF32x4(-3.0f, 1.0f, 3.0f, 4.0f);
  __uint128_t arg2 = MakeF32x4(1.0f, -1.0f, -7.0f, 2.0f);
  ASSERT_EQ(AsmFacge(arg1, arg2), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0xffff'ffff'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareEqualF64) {
  constexpr auto AsmFcmeq = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmeq %d0, %d1, %d2");
  uint64_t two = bit_cast<uint64_t>(2.0);
  uint64_t six = bit_cast<uint64_t>(6.0);
  ASSERT_EQ(AsmFcmeq(two, six), 0x0000'0000'0000'0000ULL);
  ASSERT_EQ(AsmFcmeq(two, two), 0xffff'ffff'ffff'ffffULL);
  ASSERT_EQ(AsmFcmeq(kDefaultNaN64AsInteger, two), 0x0000'0000'0000'0000ULL);
  ASSERT_EQ(AsmFcmeq(two, kDefaultNaN64AsInteger), 0x0000'0000'0000'0000ULL);
}

TEST(Arm64InsnTest, CompareEqualF64x2) {
  constexpr auto AsmFcmeq = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmeq %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(-3.0, 2.0);
  __uint128_t arg2 = MakeF64x2(6.0, 2.0);
  __uint128_t res = AsmFcmeq(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0xffff'ffff'ffff'ffffULL));
  arg1 = MakeF64x2(7.0, -0.0);
  arg2 = MakeF64x2(-8.0, 5.0);
  res = AsmFcmeq(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterEqualF64) {
  constexpr auto AsmFcmge = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmge %d0, %d1, %d2");
  uint64_t two = bit_cast<uint64_t>(2.0);
  uint64_t six = bit_cast<uint64_t>(6.0);
  ASSERT_EQ(AsmFcmge(two, six), 0x0000'0000'0000'0000ULL);
  ASSERT_EQ(AsmFcmge(two, two), 0xffff'ffff'ffff'ffffULL);
  ASSERT_EQ(AsmFcmge(six, two), 0xffff'ffff'ffff'ffffULL);
  ASSERT_EQ(AsmFcmge(kDefaultNaN64AsInteger, two), 0x0000'0000'0000'0000ULL);
  ASSERT_EQ(AsmFcmge(two, kDefaultNaN64AsInteger), 0x0000'0000'0000'0000ULL);
}

TEST(Arm64InsnTest, CompareGreaterEqualF64x2) {
  constexpr auto AsmFcmge = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmge %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(-3.0, 2.0);
  __uint128_t arg2 = MakeF64x2(6.0, 2.0);
  __uint128_t res = AsmFcmge(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0xffff'ffff'ffff'ffffULL));
  arg1 = MakeF64x2(7.0, -0.0);
  arg2 = MakeF64x2(-8.0, 5.0);
  res = AsmFcmge(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterF64) {
  constexpr auto AsmFcmgt = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmgt %d0, %d1, %d2");
  uint64_t two = bit_cast<uint64_t>(2.0);
  uint64_t six = bit_cast<uint64_t>(6.0);
  ASSERT_EQ(AsmFcmgt(two, six), 0x0000'0000'0000'0000ULL);
  ASSERT_EQ(AsmFcmgt(two, two), 0x0000'0000'0000'0000ULL);
  ASSERT_EQ(AsmFcmgt(six, two), 0xffff'ffff'ffff'ffffULL);
  ASSERT_EQ(AsmFcmgt(kDefaultNaN64AsInteger, two), 0x0000'0000'0000'0000ULL);
  ASSERT_EQ(AsmFcmgt(two, kDefaultNaN64AsInteger), 0x0000'0000'0000'0000ULL);
}

TEST(Arm64InsnTest, CompareGreaterF64x2) {
  constexpr auto AsmFcmgt = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("fcmgt %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeF64x2(-3.0, 2.0);
  __uint128_t arg2 = MakeF64x2(6.0, 2.0);
  __uint128_t res = AsmFcmgt(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  arg1 = MakeF64x2(7.0, -0.0);
  arg2 = MakeF64x2(-8.0, 5.0);
  res = AsmFcmgt(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AndInt8x16) {
  __uint128_t op1 = MakeUInt128(0x7781'8577'8053'2171ULL, 0x2268'0661'3001'9278ULL);
  __uint128_t op2 = MakeUInt128(0x0498'8627'2327'9178ULL, 0x6085'7843'8382'7967ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("and %0.16b, %1.16b, %2.16b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0480'8427'0003'0170ULL, 0x2000'0041'0000'1060ULL));
}

TEST(Arm64InsnTest, AndInt8x8) {
  __uint128_t op1 = MakeUInt128(0x7781'8577'8053'2171ULL, 0x2268'0661'3001'9278ULL);
  __uint128_t op2 = MakeUInt128(0x0498'8627'2327'9178ULL, 0x6085'7843'8382'7967ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("and %0.8b, %1.8b, %2.8b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0480'8427'0003'0170ULL, 0));
}

TEST(Arm64InsnTest, OrInt8x16) {
  __uint128_t op1 = MakeUInt128(0x00ff'aa55'0011'2244ULL, 0x1248'1248'1248'1248ULL);
  __uint128_t op2 = MakeUInt128(0x4422'1100'ffaa'5500ULL, 0x1122'4488'1122'4488ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("orr %0.16b, %1.16b, %2.16b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x44ff'bb55'ffbb'7744ULL, 0x136a'56c8'136a'56c8ULL));
}

TEST(Arm64InsnTest, OrInt8x8) {
  __uint128_t op1 = MakeUInt128(0x00ff'aa55'0011'2244ULL, 0x1248'1248'1248'1248ULL);
  __uint128_t op2 = MakeUInt128(0x4422'1100'ffaa'5500ULL, 0x1122'4488'1122'4488ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("orr %0.8b, %1.8b, %2.8b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x44ff'bb55'ffbb'7744ULL, 0));
}

TEST(Arm64InsnTest, XorInt8x16) {
  __uint128_t op1 = MakeUInt128(0x1050'7922'7968'9258ULL, 0x9235'4201'9956'1121ULL);
  __uint128_t op2 = MakeUInt128(0x8239'8645'6596'1163ULL, 0x5488'6230'5774'5649ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("eor %0.16b, %1.16b, %2.16b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x9269'ff67'1cfe'833bULL, 0xc6bd'2031'ce22'4768ULL));
}

TEST(Arm64InsnTest, XorInt8x8) {
  __uint128_t op1 = MakeUInt128(0x1050'7922'7968'9258ULL, 0x9235'4201'9956'1121ULL);
  __uint128_t op2 = MakeUInt128(0x8239'8645'6596'1163ULL, 0x5488'6230'5774'5649ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("eor %0.8b, %1.8b, %2.8b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x9269'ff67'1cfe'833bULL, 0));
}

TEST(Arm64InsnTest, AndNotInt8x16) {
  __uint128_t op1 = MakeUInt128(0x0313'7838'7528'8658ULL, 0x7533'2083'8142'0617ULL);
  __uint128_t op2 = MakeUInt128(0x2327'9178'6085'7843ULL, 0x8382'7967'9766'8145ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("bic %0.16b, %1.16b, %2.16b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0010'6800'1528'8618ULL, 0x7431'0080'0000'0612ULL));
}

TEST(Arm64InsnTest, AndNotInt8x8) {
  __uint128_t op1 = MakeUInt128(0x4861'0454'3266'4821ULL, 0x2590'3600'1133'0530ULL);
  __uint128_t op2 = MakeUInt128(0x5420'1995'6112'1290ULL, 0x8572'4245'4150'6959ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("bic %0.8b, %1.8b, %2.8b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0841'0440'1264'4821ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AndNotInt16x4Imm) {
  __uint128_t res = MakeUInt128(0x9690'3149'5019'1085ULL, 0x7598'4423'9198'6291ULL);

  asm("bic %0.4h, #0x3" : "=w"(res) : "0"(res));

  ASSERT_EQ(res, MakeUInt128(0x9690'3148'5018'1084ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AndNotInt16x4ImmShiftedBy8) {
  __uint128_t res = MakeUInt128(0x8354'0567'0403'8674ULL, 0x3513'6222'2477'1589ULL);

  asm("bic %0.4h, #0xa8, lsl #8" : "=w"(res) : "0"(res));

  ASSERT_EQ(res, MakeUInt128(0x0354'0567'0403'0674ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AndNotInt32x2ImmShiftedBy8) {
  __uint128_t res = MakeUInt128(0x1842'6312'9860'8099ULL, 0x8886'8741'3260'4721ULL);

  asm("bic %0.2s, #0xd3, lsl #8" : "=w"(res) : "0"(res));

  ASSERT_EQ(res, MakeUInt128(0x1842'2012'9860'0099ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AndNotInt32x2ImmShiftedBy16) {
  __uint128_t res = MakeUInt128(0x2947'8672'4229'2465ULL, 0x4366'8009'8067'6928ULL);

  asm("bic %0.2s, #0x22, lsl #16" : "=w"(res) : "0"(res));

  ASSERT_EQ(res, MakeUInt128(0x2945'8672'4209'2465ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AndNotInt32x2ImmShiftedBy24) {
  __uint128_t res = MakeUInt128(0x0706'9779'4223'6250ULL, 0x8221'6889'5738'3798ULL);

  asm("bic %0.2s, #0x83, lsl #24" : "=w"(res) : "0"(res));

  ASSERT_EQ(res, MakeUInt128(0x0406'9779'4023'6250ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, OrInt16x4Imm) {
  __uint128_t res = MakeUInt128(0x0841'2848'8626'9456ULL, 0x0424'1965'2850'2221ULL);

  asm("orr %0.4h, #0x5" : "=w"(res) : "0"(res));

  ASSERT_EQ(res, MakeUInt128(0x0845'284d'8627'9457ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, OrNotInt8x16) {
  __uint128_t op1 = MakeUInt128(0x5428'5844'4795'2658ULL, 0x6782'1051'1413'5473ULL);
  __uint128_t op2 = MakeUInt128(0x3558'7640'2474'9647ULL, 0x3263'9141'9927'2604ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("orn %0.16b, %1.16b, %2.16b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0xdeaf'd9ff'df9f'6ff8ULL, 0xef9e'7eff'76db'ddfbULL));
}

TEST(Arm64InsnTest, OrNotInt8x8) {
  __uint128_t op1 = MakeUInt128(0x3279'1786'0857'8438ULL, 0x3827'9679'7668'1454ULL);
  __uint128_t op2 = MakeUInt128(0x6838'6894'2774'1559ULL, 0x9185'5925'2459'5395ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("orn %0.8b, %1.8b, %2.8b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0xb7ff'97ef'd8df'eebeULL, 0x0000'0000'0000'0000ULL));
}

__uint128_t BitwiseSelect(__uint128_t mask, __uint128_t src1, __uint128_t src2) {
  return (src1 & mask) | (src2 & ~mask);
}

TEST(Arm64InsnTest, BitwiseSelectInt8x16) {
  __uint128_t op1 = MakeUInt128(0x1234'5678'9abc'def0ULL, 0xfedc'ba98'7654'3210ULL);
  __uint128_t op2 = MakeUInt128(0x0101'0101'0101'0101ULL, 0x0202'0202'0202'0202ULL);
  __uint128_t op3 = MakeUInt128(0x8080'8080'8080'8080ULL, 0x0000'0000'0000'0000ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("bsl %0.16b, %1.16b, %2.16b")(op1, op2, op3);
  ASSERT_EQ(res, BitwiseSelect(op3, op1, op2));
}

TEST(Arm64InsnTest, BitwiseSelectInt8x8) {
  __uint128_t op1 = MakeUInt128(0x2000'5681'2714'5263ULL, 0x5608'2778'5771'3427ULL);
  __uint128_t op2 = MakeUInt128(0x0792'2796'8925'8923ULL, 0x5420'1995'6112'1290ULL);
  __uint128_t op3 = MakeUInt128(0x8372'9780'4995'1059ULL, 0x7317'3281'6096'3185ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("bsl %0.8b, %1.8b, %2.8b")(op1, op2, op3);
  ASSERT_EQ(res, static_cast<uint64_t>(BitwiseSelect(op3, op1, op2)));
}

TEST(Arm64InsnTest, BitwiseInsertIfTrueInt8x16) {
  __uint128_t op1 = MakeUInt128(0x1234'5678'9abc'def0ULL, 0xfedc'ba98'7654'3210ULL);
  __uint128_t op2 = MakeUInt128(0x0101'0101'0101'0101ULL, 0x0202'0202'0202'0202ULL);
  __uint128_t op3 = MakeUInt128(0x8080'8080'8080'8080ULL, 0x0000'0000'0000'0000ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("bit %0.16b, %1.16b, %2.16b")(op1, op2, op3);
  ASSERT_EQ(res, BitwiseSelect(op2, op1, op3));
}

TEST(Arm64InsnTest, BitwiseInsertIfTrueInt8x8) {
  __uint128_t op1 = MakeUInt128(0x3678'9259'0360'0113ULL, 0x3053'0548'8204'6652ULL);
  __uint128_t op2 = MakeUInt128(0x9326'1179'3105'1185ULL, 0x4807'4462'3799'6274ULL);
  __uint128_t op3 = MakeUInt128(0x6430'8602'1394'9463ULL, 0x9522'4737'1907'0217ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("bit %0.8b, %1.8b, %2.8b")(op1, op2, op3);
  ASSERT_EQ(res, static_cast<uint64_t>(BitwiseSelect(op2, op1, op3)));
}

TEST(Arm64InsnTest, BitwiseInsertIfFalseInt8x16) {
  __uint128_t op1 = MakeUInt128(0x1234'5678'9abc'def0ULL, 0xfedc'ba98'7654'3210ULL);
  __uint128_t op2 = MakeUInt128(0x0101'0101'0101'0101ULL, 0x0202'0202'0202'0202ULL);
  __uint128_t op3 = MakeUInt128(0x8080'8080'8080'8080ULL, 0x0000'0000'0000'0000ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("bif %0.16b, %1.16b, %2.16b")(op1, op2, op3);
  ASSERT_EQ(res, BitwiseSelect(op2, op3, op1));
}

TEST(Arm64InsnTest, BitwiseInsertIfFalseInt8x8) {
  __uint128_t op1 = MakeUInt128(0x7067'9821'4808'6513ULL, 0x2823'0664'7093'8446ULL);
  __uint128_t op2 = MakeUInt128(0x5964'4622'9489'5493ULL, 0x0381'9644'2881'0975ULL);
  __uint128_t op3 = MakeUInt128(0x0348'6104'5432'6648ULL, 0x2133'9360'7260'2491ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("bif %0.8b, %1.8b, %2.8b")(op1, op2, op3);
  ASSERT_EQ(res, static_cast<uint64_t>(BitwiseSelect(op2, op3, op1)));
}

TEST(Arm64InsnTest, ArithmeticShiftRightInt64x1) {
  __uint128_t arg = MakeUInt128(0x9486'0150'4665'2681ULL, 0x4398'7705'1615'3170ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("sshr %d0, %d1, #39")(arg);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'ff29'0c02ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ArithmeticShiftRightBy64Int64x1) {
  __uint128_t arg = MakeUInt128(0x9176'0426'0176'3387ULL, 0x0454'9901'7614'3641ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("sshr %d0, %d1, #64")(arg);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ArithmeticShiftRightInt64x2) {
  __uint128_t arg = MakeUInt128(0x7501'1164'9832'7856ULL, 0x3531'6145'1684'5769ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("sshr %0.2d, %1.2d, #35")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0ea0'222cULL, 0x0000'0000'06a6'2c28ULL));
}

TEST(Arm64InsnTest, ArithmeticShiftRightAccumulateInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x9667'1796'4346'8760ULL, 0x0770'4799'9537'8833ULL);
  __uint128_t arg2 = MakeUInt128(0x2557'1769'0819'6030ULL, 0x9201'8240'1884'2705ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("ssra %d0, %d1, #40")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x2557'1769'07af'c747ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ArithmeticShiftRightBy64AccumulateInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x9223'3436'5779'1601ULL, 0x2809'3179'4017'1859ULL);
  __uint128_t arg2 = MakeUInt128(0x3498'0252'4990'6698ULL, 0x4233'0173'5035'8044ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("ssra %d0, %d1, #64")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x3498'0252'4990'6697ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ArithmeticShiftRightAccumulateInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9276'4579'3106'5792ULL, 0x2955'2498'8727'5846ULL);
  __uint128_t arg2 = MakeUInt128(0x0101'6552'5637'5678ULL, 0x5667'2279'6619'8857ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("ssra %0.8h, %1.8h, #12")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x00fa'6556'563a'567dULL, 0x5669'227b'6611'885cULL));
}

TEST(Arm64InsnTest, ArithmeticRoundingShiftRightAccumulateInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9894'6715'4357'8468ULL, 0x7886'1444'5812'3145ULL);
  __uint128_t arg2 = MakeUInt128(0x1412'1478'0573'4551ULL, 0x0500'8019'0869'9603ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("srsra %0.8h, %1.8h, #12")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x140c'147e'0577'4549ULL, 0x0508'801a'086f'9606ULL));
}

TEST(Arm64InsnTest, LogicalShiftRightInt64x1) {
  __uint128_t arg = MakeUInt128(0x9859'7719'2180'5158ULL, 0x5321'4739'2653'2515ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ushr %d0, %d1, #33")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'4c2c'bb8cULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, LogicalShiftRightBy64Int64x1) {
  __uint128_t arg = MakeUInt128(0x9474'6961'3436'0928ULL, 0x6148'4941'7850'1718ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ushr %d0, %d1, #64")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, LogicalShiftRightInt64x2) {
  __uint128_t op = MakeUInt128(0x3962'6579'7877'1855ULL, 0x6084'5529'6541'2665ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ushr %0.2d, %1.2d, #33")(op);
  ASSERT_EQ(rd, MakeUInt128(0x0000'0000'1cb1'32bcULL, 0x0000'0000'3042'2a94ULL));
}

TEST(Arm64InsnTest, LogicalShiftRightAccumulateInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x9004'1124'5379'0153ULL, 0x3296'6156'9705'2237ULL);
  __uint128_t arg2 = MakeUInt128(0x0499'9395'3221'5362ULL, 0x2748'4766'0361'3677ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("usra %d0, %d1, #40")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0499'9395'32b1'5773ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, LogicalShiftRightBy64AccumulateInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x9886'5925'7866'2856ULL, 0x1249'6655'2353'3829ULL);
  __uint128_t arg2 = MakeUInt128(0x3559'1525'3478'4459ULL, 0x8183'1341'1290'0199ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("usra %d0, %d1, #64")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x3559'1525'3478'4459ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, LogicalShiftRightAccumulateInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9984'3452'2516'1050ULL, 0x7027'0562'3526'6012ULL);
  __uint128_t arg2 = MakeUInt128(0x4628'6540'3603'6745ULL, 0x3286'5105'7065'8748ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("usra %0.8h, %1.8h, #12")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x4631'6543'3605'6746ULL, 0x328d'5105'7068'874eULL));
}

TEST(Arm64InsnTest, LogicalRoundingShiftRightAccumulateInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9843'4522'5161'0507ULL, 0x0270'5623'5266'0127ULL);
  __uint128_t arg2 = MakeUInt128(0x6286'5403'6036'7453ULL, 0x2865'1057'0658'7488ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("srsra %0.8h, %1.8h, #12")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x6280'5407'603b'7453ULL, 0x2865'105c'065d'7488ULL));
}

TEST(Arm64InsnTest, SignedRoundingShiftRightInt64x1) {
  __uint128_t arg = MakeUInt128(0x9323'6857'8558'5581ULL, 0x9555'6042'1562'5088ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("srshr %d0, %d1, #40")(arg);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'ff93'2368ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SignedRoundingShiftRightInt64x2) {
  __uint128_t arg = MakeUInt128(0x8714'8783'9890'8107ULL, 0x4295'3094'1060'5969ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("srshr %0.2d, %1.2d, #36")(arg);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'f871'4878ULL, 0x0000'0000'0429'5309ULL));
}

TEST(Arm64InsnTest, SignedRoundingShiftRightAccumulateInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x9946'0165'2057'7405ULL, 0x2942'3053'6017'8031ULL);
  __uint128_t arg2 = MakeUInt128(0x3960'1880'1378'2542ULL, 0x1927'0947'6733'7191ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("srsra %d0, %d1, #33")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x3960'187f'e01b'25f5ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, UnsignedRoundingShiftRightInt64x1) {
  __uint128_t arg = MakeUInt128(0x9713'5522'0844'5285ULL, 0x2640'0812'5202'7665ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("urshr %d0, %d1, #33")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'4b89'aa91ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, UnsignedRoundingShiftRightInt64x2) {
  __uint128_t arg = MakeUInt128(0x6653'3985'7388'8786ULL, 0x6147'6294'4341'4010ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("urshr %0.2d, %1.2d, #34")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'1994'ce61ULL, 0x0000'0000'1851'd8a5ULL));
}

TEST(Arm64InsnTest, UnsignedRoundingShiftRightAccumulateInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x9616'1432'0400'6381ULL, 0x3224'6584'1111'1577ULL);
  __uint128_t arg2 = MakeUInt128(0x7184'7281'4751'9983ULL, 0x5050'4781'2977'1859ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("ursra %d0, %d1, #33")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7184'7281'925c'a39cULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ShiftLeftInt64x1) {
  __uint128_t arg = MakeUInt128(0x3903'5946'6469'1623ULL, 0x5396'8092'0139'4578ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("shl %d0, %d1, #35")(arg);
  ASSERT_EQ(res, MakeUInt128(0x2348'b118'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ShiftLeftInt64x2) {
  __uint128_t arg = MakeUInt128(0x0750'1116'4983'2785ULL, 0x6353'1614'5168'4576ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("shl %0.2d, %1.2d, #37")(arg);
  ASSERT_EQ(res, MakeUInt128(0x3064'f0a0'0000'0000ULL, 0x2d08'aec0'0000'0000ULL));
}

TEST(Arm64InsnTest, ShiftLeftInt8x8) {
  __uint128_t arg = MakeUInt128(0x0402'9560'4734'6131ULL, 0x1382'6387'8897'5517ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("shl %0.8b, %1.8b, #6")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0080'4000'c000'4040ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ShiftRightInsertInt8x8) {
  __uint128_t arg1 = MakeUInt128(0x9112'2326'1879'4059ULL, 0x9415'5406'3270'1319ULL);
  __uint128_t arg2 = MakeUInt128(0x1537'6751'1583'0432ULL, 0x0849'8720'9202'8092ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("sri %0.8b, %1.8b, #4")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1931'6252'1187'0435ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ShiftRightInsertInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x9112'2326'1879'4059ULL, 0x9415'5406'3270'1319ULL);
  __uint128_t arg2 = MakeUInt128(0x1537'6751'1583'0432ULL, 0x0849'8720'9202'8092ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("sri %d0, %d1, #20")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1537'6911'2232'6187ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ShiftRightInsertInt64x2) {
  __uint128_t arg1 = MakeUInt128(0x7332'3356'0348'4653ULL, 0x1873'0293'0266'5964ULL);
  __uint128_t arg2 = MakeUInt128(0x5013'7183'7542'8897ULL, 0x5579'7144'9924'6540ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("sri %0.2d, %1.2d, #21")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x5013'7399'919a'b01aULL, 0x5579'70c3'9814'9813ULL));
}

TEST(Arm64InsnTest, ShiftLeftInsertInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x3763'5269'6934'4354ULL, 0x4004'7306'7198'8689ULL);
  __uint128_t arg2 = MakeUInt128(0x6369'4985'6730'2175ULL, 0x2313'2529'2653'7589ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("sli %d0, %d1, #23")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x34b4'9a21'aa30'2175ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ShiftLeftInsertInt64x2) {
  __uint128_t arg1 = MakeUInt128(0x3270'2069'0287'2323ULL, 0x3005'3862'1634'7988ULL);
  __uint128_t arg2 = MakeUInt128(0x5094'6954'7200'4795ULL, 0x2311'2015'0432'9322ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("sli %0.2d, %1.2d, #21")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0d20'50e4'6460'4795ULL, 0x0c42'c68f'3112'9322ULL));
}

TEST(Arm64InsnTest, ShiftLeftLongInt8x8) {
  __uint128_t arg = MakeUInt128(0x2650'6976'2020'1995ULL, 0x5484'1265'0005'3944ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("shll %0.8h, %1.8b, #8")(arg);
  ASSERT_EQ(res, MakeUInt128(0x2000'2000'1900'9500ULL, 0x2600'5000'6900'7600ULL));
}

TEST(Arm64InsnTest, UnsignedShiftLeftLongInt8x8) {
  __uint128_t arg = MakeUInt128(0x2650'6976'2020'1995ULL, 0x5484'1265'0005'3944ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ushll %0.8h, %1.8b, #4")(arg);
  ASSERT_EQ(res, MakeUInt128(0x200'0200'0190'0950ULL, 0x260'0500'0690'0760ULL));
}

TEST(Arm64InsnTest, ShiftLeftLongInt8x8Upper) {
  __uint128_t arg = MakeUInt128(0x9050'4292'2597'8771ULL, 0x0667'8738'4000'0616ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("shll2 %0.8h, %1.16b, #8")(arg);
  ASSERT_EQ(res, MakeUInt128(0x4000'0000'0600'1600ULL, 0x0600'6700'8700'3800ULL));
}

TEST(Arm64InsnTest, SignedShiftLeftLongInt32x2) {
  __uint128_t arg = MakeUInt128(0x9075'4079'2342'4023ULL, 0x0092'5900'7017'3196ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("sshll %0.2d, %1.2s, #9")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0046'8480'4600ULL, 0xffff'ff20'ea80'f200ULL));
}

TEST(Arm64InsnTest, SignedShiftLeftLongInt32x2Upper) {
  __uint128_t arg = MakeUInt128(0x9382'4322'2718'8515ULL, 0x9740'5470'2148'2897ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("sshll2 %0.2d, %1.4s, #9")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0042'9051'2e00ULL, 0xffff'ff2e'80a8'e000ULL));
}

TEST(Arm64InsnTest, SignedShiftLeftLongInt32x2By0) {
  __uint128_t arg = MakeUInt128(0x9008'7776'9776'3127ULL, 0x9572'2672'6555'6259ULL);
  // SXTL is an alias for SSHLL for the shift count being zero.
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("sxtl %0.2d, %1.2s")(arg);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'9776'3127ULL, 0xffff'ffff'9008'7776ULL));
}

TEST(Arm64InsnTest, ShiftLeftLongInt32x2) {
  __uint128_t arg = MakeUInt128(0x9094'3346'7685'1422ULL, 0x1447'7379'3937'5170ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ushll %0.2d, %1.2s, #9")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'00ed'0a28'4400ULL, 0x0000'0121'2866'8c00ULL));
}

TEST(Arm64InsnTest, ShiftLeftLongInt32x2Upper) {
  __uint128_t arg = MakeUInt128(0x7096'8340'8005'3559ULL, 0x8491'7541'7381'8839ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ushll2 %0.2d, %1.4s, #17")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'e703'1072'0000ULL, 0x0001'0922'ea82'0000ULL));
}

TEST(Arm64InsnTest, ShiftLeftLongInt32x2By0) {
  __uint128_t arg = MakeUInt128(0x9945'6815'0652'6530ULL, 0x5371'8294'1270'3369ULL);
  // UXTL is an alias for USHLL for the shift count being zero.
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("uxtl %0.2d, %1.2s")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0652'6530ULL, 0x0000'0000'9945'6815ULL));
}

TEST(Arm64InsnTest, ShiftRightNarrowI16x8) {
  __uint128_t arg = MakeUInt128(0x9378'5417'8610'9696ULL, 0x9202'5388'6503'4577ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("shrn %0.8b, %1.8h, #2")(arg);
  ASSERT_EQ(res, MakeUInt128(0x80e2'405d'de05'84a5ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ShiftRightNarrowI16x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x9779'9400'1260'1642ULL, 0x2760'9260'8234'9304ULL);
  __uint128_t arg2 = MakeUInt128(0x3879'1582'9984'8645ULL, 0x9271'7340'5922'5620ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("shrn2 %0.16b, %1.8h, #2")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x3879'1582'9984'8645ULL, 0xd898'8dc1'de00'9890ULL));
}

TEST(Arm64InsnTest, RoundingShiftRightNarrowI16x8) {
  __uint128_t arg = MakeUInt128(0x9303'7746'8809'9929ULL, 0x6877'5824'4104'7878ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("rshrn %0.8b, %1.8h, #2")(arg);
  ASSERT_EQ(res, MakeUInt128(0x1e09'411e'c1d2'024aULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, RoundingShiftRightNarrowI16x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x9314'5076'0716'7064ULL, 0x3556'8274'3774'3965ULL);
  __uint128_t arg2 = MakeUInt128(0x2103'0986'0409'2717ULL, 0x0909'5128'0863'0902ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("rshrn2 %0.16b, %1.8h, #2")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x2103'0986'0409'2717ULL, 0x569d'dd59'c51e'c619ULL));
}

TEST(Arm64InsnTest, AddInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x0080'0000'0000'0003ULL, 0xdead'beef'0123'4567ULL);
  __uint128_t arg2 = MakeUInt128(0x0080'0000'0000'0005ULL, 0x0123'dead'beef'4567ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("add %d0, %d1, %d2")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0100'0000'0000'0008ULL, 0x0ULL));
}

TEST(Arm64InsnTest, AddInt32x4) {
  // The "add" below adds two vectors, each with four 32-bit elements.  We set the sign
  // bit for each element to verify that the carry does not affect any lane.
  __uint128_t op1 = MakeUInt128(0x8000'0003'8000'0001ULL, 0x8000'0007'8000'0005ULL);
  __uint128_t op2 = MakeUInt128(0x8000'0004'8000'0002ULL, 0x8000'0008'8000'0006ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("add %0.4s, %1.4s, %2.4s")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0000'0007'0000'0003ULL, 0x0000'000f'0000'000bULL));
}

TEST(Arm64InsnTest, AddInt32x2) {
  __uint128_t op1 = MakeUInt128(0x8000'0003'8000'0001ULL, 0x8000'0007'8000'0005ULL);
  __uint128_t op2 = MakeUInt128(0x8000'0004'8000'0002ULL, 0x8000'0008'8000'0006ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("add %0.2s, %1.2s, %2.2s")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0000'0007'0000'0003ULL, 0));
}

TEST(Arm64InsnTest, AddInt64x2) {
  __uint128_t op1 = MakeUInt128(0x8000'0003'8000'0001ULL, 0x8000'0007'8000'0005ULL);
  __uint128_t op2 = MakeUInt128(0x8000'0004'8000'0002ULL, 0x8000'0008'8000'0006ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("add %0.2d, %1.2d, %2.2d")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0000'0008'0000'0003ULL, 0x0000'0010'0000'000bULL));
}

TEST(Arm64InsnTest, SubInt64x1) {
  __uint128_t arg1 = MakeUInt128(0x0000'0000'0000'0002ULL, 0x0011'2233'4455'6677ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0003ULL, 0x0123'4567'89ab'cdefULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sub %d0, %d1, %d2")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0ULL));
}

TEST(Arm64InsnTest, SubInt64x2) {
  constexpr auto AsmSub = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sub %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeUInt128(0x6873'1159'5628'6388ULL, 0x2353'7875'9375'1957ULL);
  __uint128_t arg2 = MakeUInt128(0x7818'5778'0532'1712ULL, 0x2680'6613'0019'2787ULL);
  __uint128_t res = AsmSub(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xf05a'b9e1'50f6'4c76ULL, 0xfcd3'1262'935b'f1d0ULL));
}

TEST(Arm64InsnTest, SubInt32x4) {
  __uint128_t op1 = MakeUInt128(0x0000'000a'0000'0005ULL, 0x0000'000c'0000'0c45ULL);
  __uint128_t op2 = MakeUInt128(0x0000'0005'0000'0003ULL, 0x0000'0002'0000'0c45ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sub %0.4s, %1.4s, %2.4s")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0000'0005'0000'0002ULL, 0x0'0000'000a'0000'0000ULL));
}

TEST(Arm64InsnTest, SubInt32x2) {
  __uint128_t op1 = MakeUInt128(0x0000'0000'0000'0005ULL, 0x0000'0000'0000'0c45ULL);
  __uint128_t op2 = MakeUInt128(0x0000'0000'0000'0003ULL, 0x0000'0000'0000'0c45ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sub %0.2s, %1.2s, %2.2s")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0000'0000'0000'0002ULL, 0x0'0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SubInt16x4) {
  __uint128_t arg1 = MakeUInt128(0x8888'7777'6666'5555ULL, 0);
  __uint128_t arg2 = MakeUInt128(0x1111'2222'3333'4444ULL, 0);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sub %0.4h, %1.4h, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7777'5555'3333'1111ULL, 0));
}

TEST(Arm64InsnTest, MultiplyI8x8) {
  __uint128_t arg1 = MakeUInt128(0x5261'3655'4978'1893ULL, 0x1297'8482'1682'9989ULL);
  __uint128_t arg2 = MakeUInt128(0x4542'8584'4479'5265ULL, 0x8678'2105'1141'3547ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("mul %0.8b, %1.8b, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1a02'0ed4'64b8'b0ffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MultiplyAndAccumulateI8x8) {
  __uint128_t arg1 = MakeUInt128(0x5848'4063'5342'2072ULL, 0x2258'2848'8648'1584ULL);
  __uint128_t arg2 = MakeUInt128(0x7823'9864'5659'6116ULL, 0x3548'8623'0577'4564ULL);
  __uint128_t arg3 = MakeUInt128(0x8797'1089'3145'6691ULL, 0x3686'7228'7489'4056ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("mla %0.8b, %1.8b, %2.8b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0xc76f'1035'1337'865dULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MultiplyAndAccumulateI8x8IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x4143'3345'4776'2416ULL, 0x8625'1898'3569'4855ULL);
  __uint128_t arg2 = MakeUInt128(0x5346'4620'8046'6842ULL, 0x5906'9491'2933'1367ULL);
  __uint128_t arg3 = MakeUInt128(0x0355'8764'0247'4964ULL, 0x7326'3914'1992'7260ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("mla %0.4h, %1.4h, %2.h[0]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x0e9b'c72e'5eb3'8710ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MultiplyAndAccumulateI8x8IndexedElemPosition2) {
  __uint128_t arg1 = MakeUInt128(0x1431'4298'0919'0659ULL, 0x2509'3722'1696'4615ULL);
  __uint128_t arg2 = MakeUInt128(0x2686'8386'8942'7741ULL, 0x5599'1855'9252'4595ULL);
  __uint128_t arg3 = MakeUInt128(0x6099'1246'0805'1243ULL, 0x8843'9045'1244'1365ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("mla %0.2s, %1.2s, %2.s[2]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x6ce7'ccbe'dccd'c110ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MultiplyAndSubtractI8x8IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x8297'4555'7067'4983ULL, 0x8505'4945'8858'6926ULL);
  __uint128_t arg2 = MakeUInt128(0x6549'9119'8818'3479ULL, 0x7753'5663'6980'7426ULL);
  __uint128_t arg3 = MakeUInt128(0x4524'9192'1732'1721ULL, 0x4772'3501'4144'1973ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("mls %0.4h, %1.4h, %2.h[1]")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0xcefc'e99a'd58a'9ad9ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MultiplyAndSubtractI8x8) {
  __uint128_t arg1 = MakeUInt128(0x0635'3422'0722'2582ULL, 0x8488'6481'5845'6028ULL);
  __uint128_t arg2 = MakeUInt128(0x9864'5659'6116'3548ULL, 0x8623'0577'4564'9803ULL);
  __uint128_t arg3 = MakeUInt128(0x1089'3145'6691'3686ULL, 0x7228'7489'4056'0101ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("mls %0.8b, %1.8b, %2.8b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x80d5'b973'bfa5'8df6ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MultiplyI32x4IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x143'3345'4776'2416ULL, 0x8625'1898'3569'4855ULL);
  __uint128_t arg2 = MakeUInt128(0x627'2327'9178'6085ULL, 0x7843'8382'7967'9766ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("mul %0.4s, %1.4s, %2.s[1]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xcec2'3e83'0d48'815aULL, 0xd12b'8728'8ae0'a3f3ULL));
}

TEST(Arm64InsnTest, PolynomialMultiplyU8x8) {
  __uint128_t arg1 = MakeUInt128(0x1862'0564'7693'1257ULL, 0x0586'3566'2018'5581ULL);
  __uint128_t arg2 = MakeUInt128(0x1668'0396'2657'9787ULL, 0x7185'5608'4552'9654ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("pmul %0.8b, %1.8b, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xd0d0'0f18'f409'5e25ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, PolynomialMultiplyLongU8x8) {
  __uint128_t arg1 = MakeUInt128(0x1327'6561'8093'7734ULL, 0x4403'0707'4692'1120ULL);
  __uint128_t arg2 = MakeUInt128(0x9838'9522'8684'7831ULL, 0x2355'2658'2131'4495ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("pmull %0.8h, %1.8b, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x4300'4bcc'17e8'05f4ULL, 0x0828'07a8'3521'0ce2ULL));
}

TEST(Arm64InsnTest, PolynomialMultiplyLongU8x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x4439'6582'5337'5438ULL, 0x8569'0941'1303'1509ULL);
  __uint128_t arg2 = MakeUInt128(0x1865'6196'7337'8623ULL, 0x6256'1252'1632'0862ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("pmull2 %0.8h, %1.16b, %2.16b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x015a'0056'00a8'0372ULL, 0x30ea'1da6'0082'14d2ULL));
}

TEST(Arm64InsnTest, PolynomialMultiplyLongU64x2) {
  __uint128_t arg1 = MakeUInt128(0x1000'1000'1000'1000ULL, 0xffff'eeee'ffff'eeeeULL);
  __uint128_t arg2 = MakeUInt128(0x1'0001ULL, 0xffff'eeee'ffff'eeeeULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("pmull %0.1q, %1.1d, %2.1d")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1000ULL, 0x1000ULL));
}

TEST(Arm64InsnTest, PolynomialMultiplyLongU64x2Upper) {
  __uint128_t arg1 = MakeUInt128(0xffff'eeee'ffff'eeeeULL, 0x1000'1000'1000'1000ULL);
  __uint128_t arg2 = MakeUInt128(0xffff'eeee'ffff'eeeeULL, 0x1'0001ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("pmull2 %0.1q, %1.2d, %2.2d")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1000ULL, 0x1000ULL));
}

TEST(Arm64InsnTest, PairwiseAddInt8x16) {
  __uint128_t op1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t op2 = MakeUInt128(0x0706'0504'0302'0100ULL, 0x0f0e'0d0c'0b0a'0908ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("addp %0.16b, %1.16b, %2.16b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0xeda9'6521'dd99'5511ULL, 0x1d19'1511'0d09'0501ULL));
}

TEST(Arm64InsnTest, PairwiseAddInt8x8) {
  __uint128_t op1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t op2 = MakeUInt128(0x0706'0504'0302'0100ULL, 0x0f0e'0d0c'0b0a'0908ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("addp %0.8b, %1.8b, %2.8b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0d09'0501'dd99'5511ULL, 0));
}

TEST(Arm64InsnTest, PairwiseAddInt64x2) {
  __uint128_t op1 = MakeUInt128(1ULL, 2ULL);
  __uint128_t op2 = MakeUInt128(3ULL, 4ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("addp %0.2d, %1.2d, %2.2d")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(3ULL, 7ULL));
}

TEST(Arm64InsnTest, CompareEqualInt8x16) {
  __uint128_t op1 = MakeUInt128(0x9375'1957'7818'5778ULL, 0x0532'1712'2680'6613ULL);
  __uint128_t op2 = MakeUInt128(0x9371'5957'7881'5787ULL, 0x0352'1721'2606'8613ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmeq %0.16b, %1.16b, %2.16b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0xff00'00ff'ff00'ff00ULL, 0x0000'ff00'ff00'00ffULL));
}

TEST(Arm64InsnTest, CompareEqualInt8x8) {
  __uint128_t op1 = MakeUInt128(0x9375'1957'7818'5778ULL, 0x0532'1712'2680'6613ULL);
  __uint128_t op2 = MakeUInt128(0x9371'5957'7881'5787ULL, 0x0352'1721'2606'8613ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmeq %0.8b, %1.8b, %2.8b")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0xff00'00ff'ff00'ff00ULL, 0));
}

TEST(Arm64InsnTest, CompareEqualInt16x4) {
  __uint128_t op1 = MakeUInt128(0x4444'3333'2222'1111ULL, 0);
  __uint128_t op2 = MakeUInt128(0x8888'3333'0000'1111ULL, 0);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmeq %0.4h, %1.4h, %2.4h")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x0000'ffff'0000'ffffULL, 0));
}

TEST(Arm64InsnTest, CompareEqualInt64x1) {
  constexpr auto AsmCmeq = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmeq %d0, %d1, %d2");
  __uint128_t arg1 = MakeUInt128(0x8297'4555'7067'4983ULL, 0x8505'4945'8858'6926ULL);
  __uint128_t arg2 = MakeUInt128(0x0665'4991'1988'1834ULL, 0x7977'5356'6369'8074ULL);
  __uint128_t arg3 = MakeUInt128(0x8297'4555'7067'4983ULL, 0x1452'4919'2173'2172ULL);
  ASSERT_EQ(AsmCmeq(arg1, arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmeq(arg1, arg3), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareEqualZeroInt64x1) {
  constexpr auto AsmCmeq = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmeq %d0, %d1, #0");
  __uint128_t arg1 = MakeUInt128(0x6517'1667'7667'2793ULL, 0x0354'8515'4204'0238ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0000ULL, 0x1746'0892'3283'9170ULL);
  ASSERT_EQ(AsmCmeq(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmeq(arg2), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareEqualZeroInt8x16) {
  __uint128_t op = MakeUInt128(0x0000'5555'0033'2200ULL, 0x0000'0000'7700'1100ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmeq %0.16b, %1.16b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0xffff'0000'ff00'00ffULL, 0xffff'ffff'00ff'00ffULL));
}

TEST(Arm64InsnTest, CompareEqualZeroInt8x8) {
  __uint128_t op = MakeUInt128(0x0011'2233'0000'aaaaULL, 0xdead'beef'0000'cafeULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmeq %0.8b, %1.8b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0xff00'0000'ffff'0000ULL, 0));
}

TEST(Arm64InsnTest, CompareGreaterInt64x1) {
  constexpr auto AsmCmgt = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmgt %d0, %d1, %d2");
  __uint128_t arg1 = MakeUInt128(0x1976'6685'5923'3565ULL, 0x4639'1383'6318'5745ULL);
  __uint128_t arg2 = MakeUInt128(0x3474'9407'8488'4423ULL, 0x7721'7515'4334'2603ULL);
  __uint128_t arg3 = MakeUInt128(0x1976'6685'5923'3565ULL, 0x8183'1963'7637'0761ULL);
  __uint128_t arg4 = MakeUInt128(0x9243'5301'3677'6310ULL, 0x8491'3516'1564'2269ULL);
  ASSERT_EQ(AsmCmgt(arg1, arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmgt(arg1, arg3), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmgt(arg1, arg4), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterZeroInt64x1) {
  constexpr auto AsmCmgt = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmgt %d0, %d1, #0");
  __uint128_t arg1 = MakeUInt128(0x6517'1667'7667'2793ULL, 0x0354'8515'4204'0238ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0000ULL, 0x6174'5997'0567'4507ULL);
  __uint128_t arg3 = MakeUInt128(0x9592'0576'6827'8967ULL, 0x7644'5318'4040'4185ULL);
  ASSERT_EQ(AsmCmgt(arg1), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmgt(arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmgt(arg3), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterThanZeroInt8x16) {
  __uint128_t op = MakeUInt128(0x807f'ff00'017e'fe02ULL, 0xff7f'8000'0102'fe02ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmgt %0.16b, %1.16b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0x00ff'0000'ffff'00ffULL, 0x00ff'0000'ffff'00ffULL));
}

TEST(Arm64InsnTest, CompareGreaterThanZeroInt8x8) {
  __uint128_t op = MakeUInt128(0x00ff'7f80'017e'fe00ULL, 0x0000'cafe'dead'beefULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmgt %0.8b, %1.8b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0x0000'ff00'ffff'0000ULL, 0));
}

TEST(Arm64InsnTest, CompareGreaterThanInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9789'3890'0185'2956ULL, 0x9196'7804'5544'8285ULL);
  __uint128_t arg2 = MakeUInt128(0x7269'3890'8179'5897ULL, 0x5469'3992'6421'8285);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmgt %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'ffff'0000ULL, 0x0000'ffff'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterThanInt32x4) {
  __uint128_t arg1 = MakeUInt128(0x0000'0000'ffff'ffffULL, 0xffff'ffff'0000'0000ULL);
  __uint128_t arg2 = MakeUInt128(0xffff'ffff'0000'0000ULL, 0x0000'0000'ffff'ffffULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmgt %0.4s, %1.4s, %2.4s")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'0000'0000ULL, 0x0000'0000'ffff'ffffULL));
}

TEST(Arm64InsnTest, CompareLessZeroInt64x1) {
  constexpr auto AsmCmlt = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmlt %d0, %d1, #0");
  __uint128_t arg1 = MakeUInt128(0x4784'2645'6763'3881ULL, 0x8807'5656'1216'8960ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0000ULL, 0x8955'9999'1120'9916ULL);
  __uint128_t arg3 = MakeUInt128(0x9364'6101'7568'5060ULL, 0x1671'4535'4315'8148ULL);
  ASSERT_EQ(AsmCmlt(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmlt(arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmlt(arg3), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareLessThanZeroInt8x16) {
  __uint128_t op = MakeUInt128(0xff00'017f'fe02'0180ULL, 0x0001'027e'7ffe'ff80ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmlt %0.16b, %1.16b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0xff00'0000'ff00'00ffULL, 0x0000'0000'00ff'ffffULL));
}

TEST(Arm64InsnTest, CompareLessThanZeroInt8x8) {
  __uint128_t op = MakeUInt128(0x0002'017e'7fff'8000ULL, 0x0011'0022'0000'ffffULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmlt %0.8b, %1.8b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0x0000'0000'00ff'ff00ULL, 0));
}

TEST(Arm64InsnTest, CompareGreaterThanEqualInt64x1) {
  constexpr auto AsmCmge = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmge %d0, %d1, %d2");
  __uint128_t arg1 = MakeUInt128(0x1009'3913'6913'8107ULL, 0x2581'3781'3578'9400ULL);
  __uint128_t arg2 = MakeUInt128(0x5890'9395'6881'4856ULL, 0x0263'2243'9372'6562ULL);
  __uint128_t arg3 = MakeUInt128(0x1009'3913'6913'8107ULL, 0x5511'9958'1831'9637ULL);
  __uint128_t arg4 = MakeUInt128(0x9427'1410'0939'1369ULL, 0x1381'0725'8137'8135ULL);
  ASSERT_EQ(AsmCmge(arg1, arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmge(arg1, arg3), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmge(arg1, arg4), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterThanEqualZeroInt64x1) {
  constexpr auto AsmCmge = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmge %d0, %d1, #0");
  __uint128_t arg1 = MakeUInt128(0x5562'1167'1546'8484ULL, 0x7780'3944'7569'7980ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0000ULL, 0x3548'4875'6252'9875ULL);
  __uint128_t arg3 = MakeUInt128(0x9212'3661'6890'2596ULL, 0x2730'4306'7931'6531ULL);
  ASSERT_EQ(AsmCmge(arg1), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmge(arg2), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmge(arg3), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareGreaterThanEqualZeroInt8x16) {
  __uint128_t op = MakeUInt128(0x00ff'0102'7ffe'8002ULL, 0x80ff'fe7f'7e02'0100ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmge %0.16b, %1.16b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0xff00'ffff'ff00'00ffULL, 0x0000'00ff'ffff'ffffULL));
}

TEST(Arm64InsnTest, CompareGreaterThanEqualZeroInt8x8) {
  __uint128_t op = MakeUInt128(0x0001'027f'80fe'ff00ULL, 0x0011'2233'4455'6677ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmge %0.8b, %1.8b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0xffff'ffff'0000'00ffULL, 0));
}

TEST(Arm64InsnTest, CompareGreaterEqualInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x4391'9628'3887'0543ULL, 0x6777'4322'4276'8091ULL);
  __uint128_t arg2 = MakeUInt128(0x4391'8385'4831'8875ULL, 0x0142'4322'0899'5068ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmge %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'0000'ffffULL, 0xffff'ffff'ffff'0000ULL));
}

TEST(Arm64InsnTest, CompareLessThanEqualZeroInt64x1) {
  constexpr auto AsmCmle = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmle %d0, %d1, #0");
  __uint128_t arg1 = MakeUInt128(0x3643'2964'0633'5728ULL, 0x1070'7887'5816'4043ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0000ULL, 0x5865'7202'2763'7840ULL);
  __uint128_t arg3 = MakeUInt128(0x8694'3468'2859'0066ULL, 0x6408'0631'4077'7577ULL);
  ASSERT_EQ(AsmCmle(arg1), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmle(arg2), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmle(arg3), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareLessThanEqualZeroInt8x16) {
  __uint128_t op = MakeUInt128(0x80ff'fe7f'7e02'0100ULL, 0x00ff'0102'7ffe'8002ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmle %0.16b, %1.16b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0xffff'ff00'0000'00ffULL, 0xffff'0000'00ff'ff00ULL));
}

TEST(Arm64InsnTest, CompareHigherInt64x1) {
  constexpr auto AsmCmhi = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmhi %d0, %d1, %d2");
  __uint128_t arg1 = MakeUInt128(0x1009'3913'6913'8107ULL, 0x2581'3781'3578'9400ULL);
  __uint128_t arg2 = MakeUInt128(0x0759'1672'9700'7850ULL, 0x5807'1718'6381'0549ULL);
  __uint128_t arg3 = MakeUInt128(0x1009'3913'6913'8107ULL, 0x6026'3224'3937'2656ULL);
  __uint128_t arg4 = MakeUInt128(0x9087'8395'2324'5323ULL, 0x7896'0298'4166'9225ULL);
  ASSERT_EQ(AsmCmhi(arg1, arg2), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmhi(arg1, arg3), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmhi(arg1, arg4), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareHigherInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x6517'1667'7667'2793ULL, 0x0354'8515'4204'0238ULL);
  __uint128_t arg2 = MakeUInt128(0x2057'1667'7896'7764ULL, 0x4531'8404'4204'5540ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmhi %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'0000'0000'0000ULL, 0x0000'ffff'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareHigherInt32x4) {
  __uint128_t arg1 = MakeUInt128(0x0000'0000'ffff'ffffULL, 0xffff'ffff'0000'0000ULL);
  __uint128_t arg2 = MakeUInt128(0xffff'ffff'0000'0000ULL, 0x0000'0000'ffff'ffffULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmhi %0.4s, %1.4s, %2.4s")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'ffff'ffffULL, 0xffff'ffff'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareHigherSameInt64x1) {
  constexpr auto AsmCmhs = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmhs %d0, %d1, %d2");
  __uint128_t arg1 = MakeUInt128(0x3529'5661'3978'8848ULL, 0x6050'9786'0859'5701ULL);
  __uint128_t arg2 = MakeUInt128(0x1769'8458'7581'0446ULL, 0x6283'9988'0600'6162ULL);
  __uint128_t arg3 = MakeUInt128(0x3529'5661'3978'8848ULL, 0x9001'8529'5691'9678ULL);
  __uint128_t arg4 = MakeUInt128(0x9628'3887'0543'6777ULL, 0x4322'4276'8091'3236ULL);
  ASSERT_EQ(AsmCmhs(arg1, arg2), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmhs(arg1, arg3), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmhs(arg1, arg4), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CompareHigherSameInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x4599'7056'7450'7183ULL, 0x3206'5034'5566'4403ULL);
  __uint128_t arg2 = MakeUInt128(0x4264'7056'3388'1880ULL, 0x3206'6121'6896'0504ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmhs %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0xffff'0000'0000'ffffULL));
}

TEST(Arm64InsnTest, CompareLessThanEqualZeroInt8x8) {
  __uint128_t op = MakeUInt128(0x00ff'fe80'7f02'0100ULL, 0x00aa'bbcc'ddee'ff00ULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cmle %0.8b, %1.8b, #0")(op);
  ASSERT_EQ(rd, MakeUInt128(0xffff'ffff'0000'00ffULL, 0));
}

TEST(Arm64InsnTest, TestInt64x1) {
  constexpr auto AsmCmtst = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmtst %d0, %d1, %d2");
  __uint128_t arg1 = MakeUInt128(0xaaaa'aaaa'5555'5555ULL, 0x7698'3854'8318'8750ULL);
  __uint128_t arg2 = MakeUInt128(0x5555'5555'aaaa'aaaaULL, 0x1429'3890'8995'0685ULL);
  __uint128_t arg3 = MakeUInt128(0xaa00'aa00'5500'5500ULL, 0x4530'7651'1680'3337ULL);
  ASSERT_EQ(AsmCmtst(arg1, arg2), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmCmtst(arg1, arg3), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, TestInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x5999'9112'0991'6464ULL, 0x6441'1918'5682'7700ULL);
  __uint128_t arg2 = MakeUInt128(0x6101'7568'5060'1671ULL, 0x4535'4315'8148'0105ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("cmtst %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffff'0000'ffffULL, 0xffff'ffff'0000'ffffULL));
}

TEST(Arm64InsnTest, ExtractVectorFromPair) {
  __uint128_t op1 = MakeUInt128(0x0011'2233'4455'6677ULL, 0x8899'aabb'ccdd'eeffULL);
  __uint128_t op2 = MakeUInt128(0x0001'0203'0405'0607ULL, 0x0809'0a0b'0c0d'0e0fULL);
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ext %0.16b, %1.16b, %2.16b, #8")(op1, op2);
  ASSERT_EQ(rd, MakeUInt128(0x8899'aabb'ccdd'eeffULL, 0x0001'0203'0405'0607ULL));
}

TEST(Arm64InsnTest, ExtractVectorFromPairHalfWidth) {
  __uint128_t op1 = MakeUInt128(0x8138'2686'8386'8942ULL, 0x7741'5599'1855'9252ULL);
  __uint128_t op2 = MakeUInt128(0x3622'2626'0991'2460ULL, 0x8051'2438'8439'0451ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ext %0.8b, %1.8b, %2.8b, #3")(op1, op2);
  ASSERT_EQ(res, MakeUInt128(0x9124'6081'3826'8683ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ExtractVectorFromPairHalfWidthPosition1) {
  __uint128_t op1 = MakeUInt128(0x9471'3296'2107'3404ULL, 0x3751'8957'3596'1458ULL);
  __uint128_t op2 = MakeUInt128(0x9048'0109'4121'4722ULL, 0x1317'9476'4777'2622ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ext %0.8b, %1.8b, %2.8b, #1")(op1, op2);
  ASSERT_EQ(res, MakeUInt128(0x2294'7132'9621'0734ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, Load1OneI8x8) {
  static constexpr uint64_t arg = 0x8867'9158'9690'4956ULL;
  __uint128_t res;
  asm("ld1 {%0.8b}, [%1]" : "=w"(res) : "r"(&arg) : "memory");
  ASSERT_EQ(res, arg);
}

TEST(Arm64InsnTest, Load1ThreeI8x8) {
  static constexpr uint64_t arg[3] = {
      0x3415'3545'8428'3376ULL, 0x4378'1119'8855'6318ULL, 0x7777'9253'7201'1667ULL};
  __uint128_t res[3];
  asm("ld1 {v0.8b-v2.8b}, [%3]\n\t"
      "mov %0.16b, v0.16b\n\t"
      "mov %1.16b, v1.16b\n\t"
      "mov %2.16b, v2.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(arg)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], static_cast<__uint128_t>(arg[0]));
  ASSERT_EQ(res[1], static_cast<__uint128_t>(arg[1]));
  ASSERT_EQ(res[2], static_cast<__uint128_t>(arg[2]));
}

TEST(Arm64InsnTest, Load1FourI8x8) {
  static constexpr uint64_t arg[4] = {
      0x9523'6884'8309'9930ULL,
      0x2757'4199'1646'3841ULL,
      0x4270'7798'8708'8742ULL,
      0x2927'7053'8912'2717ULL,
  };
  __uint128_t res[4];
  asm("ld1 {v0.8b-v3.8b}, [%4]\n\t"
      "mov %0.16b, v0.16b\n\t"
      "mov %1.16b, v1.16b\n\t"
      "mov %2.16b, v2.16b\n\t"
      "mov %3.16b, v3.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(arg)
      : "v0", "v1", "v2", "v3", "memory");
  ASSERT_EQ(res[0], static_cast<__uint128_t>(arg[0]));
  ASSERT_EQ(res[1], static_cast<__uint128_t>(arg[1]));
  ASSERT_EQ(res[2], static_cast<__uint128_t>(arg[2]));
  ASSERT_EQ(res[3], static_cast<__uint128_t>(arg[3]));
}

TEST(Arm64InsnTest, Store1OneI8x16) {
  static constexpr __uint128_t arg =
      MakeUInt128(0x7642'2915'8342'5006ULL, 0x7361'2453'8491'6067ULL);
  __uint128_t res;
  asm("st1 {%0.16b}, [%1]" : : "w"(arg), "r"(&res) : "memory");
  ASSERT_EQ(res, arg);
}

TEST(Arm64InsnTest, Store1ThreeI8x8) {
  static constexpr uint64_t arg[3] = {
      0x3086'4361'1138'9069ULL, 0x4202'7908'8143'1194ULL, 0x4879'9417'1540'4210ULL};
  uint64_t res[3];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "st1 {v0.8b-v2.8b}, [%3]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "r"(res)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], arg[0]);
  ASSERT_EQ(res[1], arg[1]);
  ASSERT_EQ(res[2], arg[2]);
}

TEST(Arm64InsnTest, Store1FourI8x8) {
  static constexpr uint64_t arg[4] = {0x8954'7504'4833'9314ULL,
                                      0x6896'3076'3396'6572ULL,
                                      0x2672'7043'3932'1674ULL,
                                      0x5421'8245'5706'2524ULL};
  uint64_t res[4];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "mov v3.16b, %3.16b\n\t"
      "st1 {v0.8b-v3.8b}, [%4]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "w"(arg[3]), "r"(res)
      : "v0", "v1", "v2", "v3", "memory");
  ASSERT_EQ(res[0], arg[0]);
  ASSERT_EQ(res[1], arg[1]);
  ASSERT_EQ(res[2], arg[2]);
  ASSERT_EQ(res[3], arg[3]);
}

TEST(Arm64InsnTest, Load1TwoPostIndex) {
  __uint128_t op0 = MakeUInt128(0x5499'1198'8183'4797ULL, 0x0507'9227'9689'2589ULL);
  __uint128_t op1 = MakeUInt128(0x0511'8548'0744'6237ULL, 0x6691'3686'7228'7489ULL);
  __uint128_t array[] = {
      op0,
      op1,
  };
  __uint128_t* addr = &array[0];
  __uint128_t res0 = 0;
  __uint128_t res1 = 0;

  // The "memory" below ensures that the array contents are up to date.  Without it, the
  // compiler might decide to initialize the array after the asm statement.
  //
  // We hardcode SIMD registers v0 and v1 below because there is no other way to express
  // consecutive registers, which in turn requires the mov instructions to retrieve the
  // loaded values into res0 and res1.
  asm("ld1 {v0.16b, v1.16b}, [%2], #32\n\t"
      "mov %0.16b, v0.16b\n\t"
      "mov %1.16b, v1.16b"
      : "=w"(res0), "=w"(res1), "+r"(addr)
      :
      : "v0", "v1", "memory");

  ASSERT_EQ(res0, op0);
  ASSERT_EQ(res1, op1);
  ASSERT_EQ(addr, &array[2]);
}

TEST(Arm64InsnTest, Load1OnePostIndexReg) {
  static constexpr __uint128_t arg =
      MakeUInt128(0x4884'7610'0556'4018ULL, 0x2423'9219'2695'0620ULL);
  __uint128_t res_val;
  uint64_t res_addr;
  asm("ld1 {%0.16b}, [%1], %2"
      : "=w"(res_val), "=r"(res_addr)
      : "r"(static_cast<uint64_t>(32U)), "1"(&arg)
      : "memory");
  ASSERT_EQ(res_val, arg);
  ASSERT_EQ(res_addr, reinterpret_cast<uint64_t>(&arg) + 32);
}

TEST(Arm64InsnTest, LoadSingleInt8) {
  static constexpr __uint128_t reg_before =
      MakeUInt128(0x0011'2233'4455'6677ULL, 0x8899'aabb'ccdd'eeffULL);
  static constexpr __uint128_t mem_src =
      MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t reg_after;
  asm("ld1 {%0.b}[3], [%1]" : "=w"(reg_after) : "r"(&mem_src), "0"(reg_before) : "memory");
  ASSERT_EQ(reg_after, MakeUInt128(0x0011'2233'0855'6677ULL, 0x8899'aabb'ccdd'eeffULL));
}

TEST(Arm64InsnTest, LoadSingleInt16) {
  static constexpr __uint128_t reg_before =
      MakeUInt128(0x0000'1111'2222'3333ULL, 0x4444'5555'6666'7777ULL);
  static constexpr __uint128_t mem_src =
      MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t reg_after;
  asm("ld1 {%0.h}[2], [%1]" : "=w"(reg_after) : "r"(&mem_src), "0"(reg_before) : "memory");
  ASSERT_EQ(reg_after, MakeUInt128(0x0000'0708'2222'3333ULL, 0x4444'5555'6666'7777ULL));
}

TEST(Arm64InsnTest, LoadSingleInt32) {
  static constexpr __uint128_t reg_before =
      MakeUInt128(0x0000'0000'1111'1111ULL, 0x2222'2222'3333'3333ULL);
  static constexpr __uint128_t mem_src =
      MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t reg_after;
  asm("ld1 {%0.s}[1], [%1]" : "=w"(reg_after) : "r"(&mem_src), "0"(reg_before) : "memory");
  ASSERT_EQ(reg_after, MakeUInt128(0x0506'0708'1111'1111ULL, 0x2222'2222'3333'3333ULL));
}

TEST(Arm64InsnTest, LoadSingleInt64) {
  static constexpr __uint128_t reg_before =
      MakeUInt128(0x0000'0000'0000'0000ULL, 0x1111'1111'1111'1111ULL);
  static constexpr __uint128_t mem_src =
      MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t reg_after;
  asm("ld1 {%0.d}[1], [%1]" : "=w"(reg_after) : "r"(&mem_src), "0"(reg_before) : "memory");
  ASSERT_EQ(reg_after, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0102'0304'0506'0708ULL));
}

TEST(Arm64InsnTest, StoreSingleInt8) {
  static constexpr __uint128_t arg =
      MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t mem_dest = MakeUInt128(0x0011'2233'4455'6677ULL, 0x8899'aabb'ccdd'eeffULL);
  asm("st1 {%1.b}[3], [%0]" : : "r"(&mem_dest), "w"(arg) : "memory");
  ASSERT_EQ(mem_dest, MakeUInt128(0x0011'2233'4455'6605ULL, 0x8899'aabb'ccdd'eeffULL));
}

TEST(Arm64InsnTest, StoreSingleInt16) {
  static constexpr __uint128_t arg =
      MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t mem_dest = MakeUInt128(0x0000'1111'2222'3333ULL, 0x4444'5555'6666'7777ULL);
  asm("st1 {%1.h}[5], [%0]" : : "r"(&mem_dest), "w"(arg) : "memory");
  ASSERT_EQ(mem_dest, MakeUInt128(0x0000'1111'2222'0d0eULL, 0x4444'5555'6666'7777ULL));
}

TEST(Arm64InsnTest, StoreSingleInt32) {
  static constexpr __uint128_t arg =
      MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t mem_dest = MakeUInt128(0x0000'0000'1111'1111ULL, 0x2222'2222'3333'3333ULL);
  asm("st1 {%1.s}[2], [%0]" : : "r"(&mem_dest), "w"(arg) : "memory");
  ASSERT_EQ(mem_dest, MakeUInt128(0x0000'0000'0d0e'0f10ULL, 0x2222'2222'3333'3333ULL));
}

TEST(Arm64InsnTest, StoreSingleInt64) {
  static constexpr __uint128_t arg =
      MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t mem_dest = MakeUInt128(0x0000'0000'0000'0000ULL, 0x1111'1111'1111'1111ULL);
  asm("st1 {%1.d}[1], [%0]" : : "r"(&mem_dest), "w"(arg) : "memory");
  ASSERT_EQ(mem_dest, MakeUInt128(0x090a'0b0c'0d0e'0f10ULL, 0x1111'1111'1111'1111ULL));
}

TEST(Arm64InsnTest, LoadSinglePostIndexImmInt8) {
  static constexpr __uint128_t arg1 =
      MakeUInt128(0x5494'1675'9460'5487ULL, 0x1172'3594'6429'1058ULL);
  static constexpr __uint128_t arg2 =
      MakeUInt128(0x5090'9950'2149'5879ULL, 0x3112'1961'3590'8315ULL);
  __uint128_t res;
  uint8_t* addr;
  asm("ld1 {%0.b}[3], [%1], #1" : "=w"(res), "=r"(addr) : "0"(arg1), "1"(&arg2) : "memory");
  ASSERT_EQ(res, MakeUInt128(0x5494'1675'7960'5487ULL, 0x1172'3594'6429'1058ULL));
  ASSERT_EQ(addr, reinterpret_cast<const uint8_t*>(&arg2) + 1);
}

TEST(Arm64InsnTest, LoadSinglePostIndexRegInt16) {
  static constexpr __uint128_t arg1 =
      MakeUInt128(0x0080'5878'2410'7493ULL, 0x5751'4889'9789'1173ULL);
  static constexpr __uint128_t arg2 =
      MakeUInt128(0x9746'1293'2035'1081ULL, 0x4327'0325'1409'0304ULL);
  __uint128_t res;
  uint8_t* addr;
  asm("ld1 {%0.h}[7], [%1], %2"
      : "=w"(res), "=r"(addr)
      : "r"(static_cast<uint64_t>(17U)), "0"(arg1), "1"(&arg2)
      : "memory");
  ASSERT_EQ(res, MakeUInt128(0x0080'5878'2410'7493ULL, 0x1081'4889'9789'1173ULL));
  ASSERT_EQ(addr, reinterpret_cast<const uint8_t*>(&arg2) + 17);
}

TEST(Arm64InsnTest, StoreSimdPostIndex) {
  __uint128_t old_val = MakeUInt128(0x4939'9651'4314'2980ULL, 0x9190'6592'5093'7221ULL);
  __uint128_t new_val = MakeUInt128(0x5985'2613'6554'9781ULL, 0x8931'2978'4821'6829ULL);
  __uint128_t* addr = &old_val;

  // Verify that the interpreter accepts "str q0, [x0], #8" where the register numbers are
  // the same, when the data register is one of the SIMD registers.
  asm("mov x0, %0\n\t"
      "mov v0.2D, %1.2D\n\t"
      "str q0, [x0], #8\n\t"
      "mov %0, x0"
      : "+r"(addr)
      : "w"(new_val)
      : "v0", "x0", "memory");

  ASSERT_EQ(old_val, MakeUInt128(0x5985'2613'6554'9781ULL, 0x8931'2978'4821'6829ULL));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(addr), reinterpret_cast<uintptr_t>(&old_val) + 8);
}

TEST(Arm64InsnTest, StoreZeroPostIndex1) {
  uint64_t res;
  asm("str xzr, [sp, #-16]!\n\t"
      "ldr %0, [sp, #0]\n\t"
      "add sp, sp, #16"
      : "=r"(res));
  ASSERT_EQ(res, 0);
}

TEST(Arm64InsnTest, StoreZeroPostIndex2) {
  __uint128_t arg1 = MakeUInt128(0x9415'5732'9382'0485ULL, 0x4212'3508'1739'1254ULL);
  __uint128_t arg2 = MakeUInt128(0x9749'8193'0871'4396ULL, 0x6151'3294'2045'9193ULL);
  __uint128_t res1;
  __uint128_t res2;
  asm("mov v30.16b, %2.16b\n\t"
      "mov v31.16b, %3.16b\n\t"
      "stp q30, q31, [sp, #-32]!\n\t"
      "ldr %q0, [sp, #0]\n\t"
      "ldr %q1, [sp, #16]\n\t"
      "add sp, sp, #32"
      : "=w"(res1), "=w"(res2)
      : "w"(arg1), "w"(arg2)
      : "v30", "v31");

  ASSERT_EQ(res1, arg1);
  ASSERT_EQ(res2, arg2);
}

TEST(Arm64InsnTest, Load2MultipleInt8x8) {
  static constexpr uint8_t mem[] = {0x02,
                                    0x16,
                                    0x91,
                                    0x83,
                                    0x37,
                                    0x23,
                                    0x68,
                                    0x03,
                                    0x99,
                                    0x02,
                                    0x79,
                                    0x31,
                                    0x60,
                                    0x64,
                                    0x20,
                                    0x43};
  __uint128_t res[2];
  asm("ld2 {v0.8b, v1.8b}, [%2]\n\t"
      "mov %0.16b, v0.16b\n\t"
      "mov %1.16b, v1.16b"
      : "=w"(res[0]), "=w"(res[1])
      : "r"(mem)
      : "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x2060'7999'6837'9102ULL, 0U));
  ASSERT_EQ(res[1], MakeUInt128(0x4364'3102'0323'8316ULL, 0U));
}

TEST(Arm64InsnTest, Load3MultipleInt8x8) {
  static constexpr uint8_t mem[3 * 8] = {0x32, 0x87, 0x67, 0x03, 0x80, 0x92, 0x52, 0x16,
                                         0x79, 0x07, 0x57, 0x12, 0x04, 0x06, 0x12, 0x37,
                                         0x59, 0x63, 0x27, 0x68, 0x56, 0x74, 0x84, 0x50};
  __uint128_t res[3];
  asm("ld3 {v7.8b-v9.8b}, [%3]\n\t"
      "mov %0.16b, v7.16b\n\t"
      "mov %1.16b, v8.16b\n\t"
      "mov %2.16b, v9.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(mem)
      : "v7", "v8", "v9", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x7427'3704'0752'0332ULL, 0U));
  ASSERT_EQ(res[1], MakeUInt128(0x8468'5906'5716'8087ULL, 0U));
  ASSERT_EQ(res[2], MakeUInt128(0x5056'6312'1279'9267ULL, 0U));
}

TEST(Arm64InsnTest, Store3MultipleInt8x8) {
  static constexpr uint64_t arg[3] = {
      0x7427'3704'0752'0332ULL, 0x8468'5906'5716'8087ULL, 0x5056'6312'1279'9267ULL};
  uint64_t res[3];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "st3 {v0.8b-v2.8b}, [%3]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "r"(res)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], 0x1652'9280'0367'8732ULL);
  ASSERT_EQ(res[1], 0x3712'0604'1257'0779ULL);
  ASSERT_EQ(res[2], 0x5084'7456'6827'6359ULL);
}

TEST(Arm64InsnTest, Load3MultipleInt8x16) {
  static constexpr uint8_t mem[3 * 16] = {
      0x69, 0x20, 0x35, 0x65, 0x63, 0x38, 0x44, 0x96, 0x25, 0x32, 0x83, 0x38,
      0x52, 0x27, 0x99, 0x24, 0x59, 0x60, 0x97, 0x86, 0x59, 0x47, 0x23, 0x88,
      0x91, 0x29, 0x63, 0x62, 0x59, 0x54, 0x32, 0x73, 0x45, 0x44, 0x37, 0x16,
      0x33, 0x55, 0x77, 0x43, 0x29, 0x49, 0x99, 0x28, 0x81, 0x05, 0x57, 0x17};
  __uint128_t res[3];
  asm("ld3 {v7.16b-v9.16b}, [%3]\n\t"
      "mov %0.16b, v7.16b\n\t"
      "mov %1.16b, v8.16b\n\t"
      "mov %2.16b, v9.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(mem)
      : "v7", "v8", "v9", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x4797'2452'3244'6569ULL, 0x599'4333'4432'6291ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x2386'5927'8396'6320ULL, 0x5728'2955'3773'5929ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x8859'6099'3825'3835ULL, 0x1781'4977'1645'5463ULL));
}

TEST(Arm64InsnTest, Store3MultipleInt8x16) {
  static constexpr __uint128_t arg[3] = {
      MakeUInt128(0x4797'2452'3244'6569ULL, 0x599'4333'4432'6291ULL),
      MakeUInt128(0x2386'5927'8396'6320ULL, 0x5728'2955'3773'5929ULL),
      MakeUInt128(0x8859'6099'3825'3835ULL, 0x1781'4977'1645'5463ULL)};
  __uint128_t res[3];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "st3 {v0.16b-v2.16b}, [%3]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "r"(res)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x2499'2752'3883'3225ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'8697'6059ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x4377'5533'1637'4445ULL, 0x1757'0581'2899'4929ULL));
}

TEST(Arm64InsnTest, Load3MultipleInt16x4) {
  static constexpr uint16_t mem[3 * 4] = {0x2069,
                                          0x6535,
                                          0x3863,
                                          0x9644,
                                          0x3225,
                                          0x3883,
                                          0x2752,
                                          0x2499,
                                          0x6059,
                                          0x8697,
                                          0x4759,
                                          0x8823};
  __uint128_t res[3];
  asm("ld3 {v30.4h-v0.4h}, [%3]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(mem)
      : "v30", "v31", "v0", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x8697'2752'9644'2069ULL, 0));
  ASSERT_EQ(res[1], MakeUInt128(0x4759'2499'3225'6535ULL, 0));
  ASSERT_EQ(res[2], MakeUInt128(0x8823'6059'3883'3863ULL, 0));
}

TEST(Arm64InsnTest, Store3MultipleInt16x4) {
  static constexpr uint64_t arg[3] = {
      0x8697'2752'9644'2069ULL, 0x4759'2499'3225'6535ULL, 0x8823'6059'3883'3863ULL};
  uint64_t res[3];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "st3 {v0.4h-v2.4h}, [%3]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "r"(res)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], 0x9644'3863'6535'2069ULL);
  ASSERT_EQ(res[1], 0x2499'2752'3883'3225ULL);
  ASSERT_EQ(res[2], 0x8823'4759'8697'6059ULL);
}

TEST(Arm64InsnTest, Load3MultipleInt16x8) {
  static constexpr uint16_t mem[3 * 8] = {0x2069, 0x6535, 0x3863, 0x9644, 0x3225, 0x3883,
                                          0x2752, 0x2499, 0x6059, 0x8697, 0x4759, 0x8823,
                                          0x2991, 0x6263, 0x5459, 0x7332, 0x4445, 0x1637,
                                          0x5533, 0x4377, 0x4929, 0x2899, 0x0581, 0x1757};
  __uint128_t res[3];
  asm("ld3 {v30.8h-v0.8h}, [%3]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(mem)
      : "v30", "v31", "v0", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x8697'2752'9644'2069ULL, 0x2899'5533'7332'2991ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x4759'2499'3225'6535ULL, 0x581'4377'4445'6263ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x8823'6059'3883'3863ULL, 0x1757'4929'1637'5459ULL));
}

TEST(Arm64InsnTest, Store3MultipleInt16x8) {
  static constexpr __uint128_t arg[3] = {
      MakeUInt128(0x8697'2752'9644'2069ULL, 0x2899'5533'7332'2991ULL),
      MakeUInt128(0x4759'2499'3225'6535ULL, 0x581'4377'4445'6263ULL),
      MakeUInt128(0x8823'6059'3883'3863ULL, 0x1757'4929'1637'5459ULL)};
  __uint128_t res[3];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "st3 {v0.8h-v2.8h}, [%3]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "r"(res)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x2499'2752'3883'3225ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'8697'6059ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x4377'5533'1637'4445ULL, 0x1757'0581'2899'4929ULL));
}

TEST(Arm64InsnTest, Load3MultipleInt32x2) {
  static constexpr uint32_t mem[3 * 2] = {
      0x6535'2069, 0x9644'3863, 0x3883'3225, 0x2499'2752, 0x8697'6059, 0x8823'4759};
  __uint128_t res[3];
  asm("ld3 {v30.2s-v0.2s}, [%3]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(mem)
      : "v30", "v31", "v0", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x2499'2752'6535'2069ULL, 0));
  ASSERT_EQ(res[1], MakeUInt128(0x8697'6059'9644'3863ULL, 0));
  ASSERT_EQ(res[2], MakeUInt128(0x8823'4759'3883'3225ULL, 0));
}

TEST(Arm64InsnTest, Store3MultipleInt32x2) {
  static constexpr uint64_t arg[3] = {
      0x2499'2752'6535'2069ULL, 0x8697'6059'9644'3863ULL, 0x8823'4759'3883'3225ULL};
  uint64_t res[3];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "st3 {v0.2s-v2.2s}, [%3]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "r"(res)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], 0x9644'3863'6535'2069ULL);
  ASSERT_EQ(res[1], 0x2499'2752'3883'3225ULL);
  ASSERT_EQ(res[2], 0x8823'4759'8697'6059ULL);
}

TEST(Arm64InsnTest, Load3MultipleInt32x4) {
  static constexpr uint32_t mem[3 * 4] = {0x6535'2069,
                                          0x9644'3863,
                                          0x3883'3225,
                                          0x2499'2752,
                                          0x8697'6059,
                                          0x8823'4759,
                                          0x6263'2991,
                                          0x7332'5459,
                                          0x1637'4445,
                                          0x4377'5533,
                                          0x2899'4929,
                                          0x1757'0581};
  __uint128_t res[3];
  asm("ld3 {v30.4s-v0.4s}, [%3]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(mem)
      : "v30", "v31", "v0", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x2499'2752'6535'2069ULL, 0x4377'5533'6263'2991ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8697'6059'9644'3863ULL, 0x2899'4929'7332'5459ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x8823'4759'3883'3225ULL, 0x1757'0581'1637'4445ULL));
}

TEST(Arm64InsnTest, Store3MultipleInt32x4) {
  static constexpr __uint128_t arg[3] = {
      MakeUInt128(0x2499'2752'6535'2069ULL, 0x4377'5533'6263'2991ULL),
      MakeUInt128(0x8697'6059'9644'3863ULL, 0x2899'4929'7332'5459ULL),
      MakeUInt128(0x8823'4759'3883'3225ULL, 0x1757'0581'1637'4445ULL)};
  __uint128_t res[3];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "st3 {v0.4s-v2.4s}, [%3]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "r"(res)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x2499'2752'3883'3225ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'8697'6059ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x4377'5533'1637'4445ULL, 0x1757'0581'2899'4929ULL));
}

TEST(Arm64InsnTest, Load3MultipleInt64x2) {
  static constexpr uint64_t mem[3 * 2] = {0x9644'3863'6535'2069,
                                          0x2499'2752'3883'3225,
                                          0x8823'4759'8697'6059,
                                          0x7332'5459'6263'2991,
                                          0x4377'5533'1637'4445,
                                          0x1757'0581'2899'4929};
  __uint128_t res[3];
  asm("ld3 {v30.2d-v0.2d}, [%3]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(mem)
      : "v30", "v31", "v0", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x2499'2752'3883'3225ULL, 0x4377'5533'1637'4445ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x8823'4759'8697'6059ULL, 0x1757'0581'2899'4929ULL));
}

TEST(Arm64InsnTest, Store3MultipleInt64x2) {
  static constexpr __uint128_t arg[3] = {
      MakeUInt128(0x9644'3863'6535'2069ULL, 0x7332'5459'6263'2991ULL),
      MakeUInt128(0x2499'2752'3883'3225ULL, 0x4377'5533'1637'4445ULL),
      MakeUInt128(0x8823'4759'8697'6059ULL, 0x1757'0581'2899'4929ULL)};
  __uint128_t res[3];
  asm("mov v0.16b, %0.16b\n\t"
      "mov v1.16b, %1.16b\n\t"
      "mov v2.16b, %2.16b\n\t"
      "st3 {v0.2d-v2.2d}, [%3]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "r"(res)
      : "v0", "v1", "v2", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x2499'2752'3883'3225ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'8697'6059ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x4377'5533'1637'4445ULL, 0x1757'0581'2899'4929ULL));
}

TEST(Arm64InsnTest, Load4MultipleInt8x8) {
  static constexpr uint8_t mem[4 * 8] = {0x69, 0x20, 0x35, 0x65, 0x63, 0x38, 0x44, 0x96,
                                         0x25, 0x32, 0x83, 0x38, 0x52, 0x27, 0x99, 0x24,
                                         0x59, 0x60, 0x97, 0x86, 0x59, 0x47, 0x23, 0x88,
                                         0x91, 0x29, 0x63, 0x62, 0x59, 0x54, 0x32, 0x73};
  __uint128_t res[4];
  asm("ld4 {v7.8b-v10.8b}, [%4]\n\t"
      "mov %0.16b, v7.16b\n\t"
      "mov %1.16b, v8.16b\n\t"
      "mov %2.16b, v9.16b\n\t"
      "mov %3.16b, v10.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(mem)
      : "v7", "v8", "v9", "v10", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x5991'5959'5225'6369ULL, 0));
  ASSERT_EQ(res[1], MakeUInt128(0x5429'4760'2732'3820ULL, 0));
  ASSERT_EQ(res[2], MakeUInt128(0x3263'2397'9983'4435ULL, 0));
  ASSERT_EQ(res[3], MakeUInt128(0x7362'8886'2438'9665ULL, 0));
}

TEST(Arm64InsnTest, Store4MultipleInt8x8) {
  static constexpr uint64_t arg[4] = {0x5991'5959'5225'6369ULL,
                                      0x5429'4760'2732'3820ULL,
                                      0x3263'2397'9983'4435ULL,
                                      0x7362'8886'2438'9665ULL};
  uint64_t res[4];
  asm("mov v7.16b, %0.16b\n\t"
      "mov v8.16b, %1.16b\n\t"
      "mov v9.16b, %2.16b\n\t"
      "mov v10.16b, %3.16b\n\t"
      "st4 {v7.8b-v10.8b}, [%4]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "w"(arg[3]), "r"(res)
      : "v7", "v8", "v9", "v10", "memory");
  ASSERT_EQ(res[0], 0x9644'3863'6535'2069ULL);
  ASSERT_EQ(res[1], 0x2499'2752'3883'3225ULL);
  ASSERT_EQ(res[2], 0x8823'4759'8697'6059ULL);
  ASSERT_EQ(res[3], 0x7332'5459'6263'2991ULL);
}

TEST(Arm64InsnTest, Load4MultipleInt8x16) {
  static constexpr uint8_t mem[4 * 16] = {
      0x69, 0x20, 0x35, 0x65, 0x63, 0x38, 0x44, 0x96, 0x25, 0x32, 0x83, 0x38, 0x52,
      0x27, 0x99, 0x24, 0x59, 0x60, 0x97, 0x86, 0x59, 0x47, 0x23, 0x88, 0x91, 0x29,
      0x63, 0x62, 0x59, 0x54, 0x32, 0x73, 0x45, 0x44, 0x37, 0x16, 0x33, 0x55, 0x77,
      0x43, 0x29, 0x49, 0x99, 0x28, 0x81, 0x05, 0x57, 0x17, 0x81, 0x98, 0x78, 0x50,
      0x68, 0x14, 0x62, 0x52, 0x32, 0x13, 0x47, 0x52, 0x37, 0x38, 0x11, 0x65};
  __uint128_t res[4];
  asm("ld4 {v7.16b-v10.16b}, [%4]\n\t"
      "mov %0.16b, v7.16b\n\t"
      "mov %1.16b, v8.16b\n\t"
      "mov %2.16b, v9.16b\n\t"
      "mov %3.16b, v10.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(mem)
      : "v7", "v8", "v9", "v10", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x5991'5959'5225'6369ULL, 0x3732'6881'8129'3345ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x5429'4760'2732'3820ULL, 0x3813'1498'0549'5544ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x3263'2397'9983'4435ULL, 0x1147'6278'5799'7737ULL));
  ASSERT_EQ(res[3], MakeUInt128(0x7362'8886'2438'9665ULL, 0x6552'5250'1728'4316ULL));
}

TEST(Arm64InsnTest, Store4MultipleInt8x16) {
  static constexpr __uint128_t arg[4] = {
      MakeUInt128(0x5991'5959'5225'6369ULL, 0x3732'6881'8129'3345ULL),
      MakeUInt128(0x5429'4760'2732'3820ULL, 0x3813'1498'0549'5544ULL),
      MakeUInt128(0x3263'2397'9983'4435ULL, 0x1147'6278'5799'7737ULL),
      MakeUInt128(0x7362'8886'2438'9665ULL, 0x6552'5250'1728'4316ULL)};
  __uint128_t res[4];
  asm("mov v7.16b, %0.16b\n\t"
      "mov v8.16b, %1.16b\n\t"
      "mov v9.16b, %2.16b\n\t"
      "mov v10.16b, %3.16b\n\t"
      "st4 {v7.16b-v10.16b}, [%4]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "w"(arg[3]), "r"(res)
      : "v7", "v8", "v9", "v10", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x2499'2752'3883'3225ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'8697'6059ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x4377'5533'1637'4445ULL, 0x1757'0581'2899'4929ULL));
  ASSERT_EQ(res[3], MakeUInt128(0x5262'1468'5078'9881ULL, 0x6511'3837'5247'1332ULL));
}

TEST(Arm64InsnTest, Load4MultipleInt16x4) {
  static constexpr uint16_t mem[4 * 4] = {0x2069,
                                          0x6535,
                                          0x3863,
                                          0x9644,
                                          0x3225,
                                          0x3883,
                                          0x2752,
                                          0x2499,
                                          0x6059,
                                          0x8697,
                                          0x4759,
                                          0x8823,
                                          0x2991,
                                          0x6263,
                                          0x5459,
                                          0x7332};
  __uint128_t res[4];
  asm("ld4 {v30.4h-v1.4h}, [%4]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b\n\t"
      "mov %3.16b, v1.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(mem)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x2991'6059'3225'2069ULL, 0));
  ASSERT_EQ(res[1], MakeUInt128(0x6263'8697'3883'6535ULL, 0));
  ASSERT_EQ(res[2], MakeUInt128(0x5459'4759'2752'3863ULL, 0));
  ASSERT_EQ(res[3], MakeUInt128(0x7332'8823'2499'9644ULL, 0));
}

TEST(Arm64InsnTest, Store4MultipleInt16x4) {
  static constexpr uint64_t arg[4] = {0x2991'6059'3225'2069ULL,
                                      0x6263'8697'3883'6535ULL,
                                      0x5459'4759'2752'3863ULL,
                                      0x7332'8823'2499'9644ULL};
  uint64_t res[4];
  asm("mov v30.16b, %0.16b\n\t"
      "mov v31.16b, %1.16b\n\t"
      "mov v0.16b, %2.16b\n\t"
      "mov v1.16b, %3.16b\n\t"
      "st4 {v30.4h-v1.4h}, [%4]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "w"(arg[3]), "r"(res)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], 0x9644'3863'6535'2069ULL);
  ASSERT_EQ(res[1], 0x2499'2752'3883'3225ULL);
  ASSERT_EQ(res[2], 0x8823'4759'8697'6059ULL);
  ASSERT_EQ(res[3], 0x7332'5459'6263'2991ULL);
}

TEST(Arm64InsnTest, Load4MultipleInt16x8) {
  static constexpr uint16_t mem[4 * 8] = {
      0x2069, 0x6535, 0x3863, 0x9644, 0x3225, 0x3883, 0x2752, 0x2499, 0x6059, 0x8697, 0x4759,
      0x8823, 0x2991, 0x6263, 0x5459, 0x7332, 0x4445, 0x1637, 0x5533, 0x4377, 0x4929, 0x2899,
      0x0581, 0x1757, 0x9881, 0x5078, 0x1468, 0x5262, 0x1332, 0x5247, 0x3837, 0x6511};
  __uint128_t res[4];
  asm("ld4 {v30.8h-v1.8h}, [%4]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b\n\t"
      "mov %3.16b, v1.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(mem)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x2991'6059'3225'2069ULL, 0x1332'9881'4929'4445ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x6263'8697'3883'6535ULL, 0x5247'5078'2899'1637ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x5459'4759'2752'3863ULL, 0x3837'1468'0581'5533ULL));
  ASSERT_EQ(res[3], MakeUInt128(0x7332'8823'2499'9644ULL, 0x6511'5262'1757'4377ULL));
}

TEST(Arm64InsnTest, Store4MultipleInt16x8) {
  static constexpr __uint128_t arg[4] = {
      MakeUInt128(0x2991'6059'3225'2069ULL, 0x1332'9881'4929'4445ULL),
      MakeUInt128(0x6263'8697'3883'6535ULL, 0x5247'5078'2899'1637ULL),
      MakeUInt128(0x5459'4759'2752'3863ULL, 0x3837'1468'0581'5533ULL),
      MakeUInt128(0x7332'8823'2499'9644ULL, 0x6511'5262'1757'4377ULL)};
  __uint128_t res[4];
  asm("mov v30.16b, %0.16b\n\t"
      "mov v31.16b, %1.16b\n\t"
      "mov v0.16b, %2.16b\n\t"
      "mov v1.16b, %3.16b\n\t"
      "st4 {v30.8h-v1.8h}, [%4]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "w"(arg[3]), "r"(res)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x2499'2752'3883'3225ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'8697'6059ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x4377'5533'1637'4445ULL, 0x1757'0581'2899'4929ULL));
  ASSERT_EQ(res[3], MakeUInt128(0x5262'1468'5078'9881ULL, 0x6511'3837'5247'1332ULL));
}

TEST(Arm64InsnTest, Load4MultipleInt32x2) {
  static constexpr uint32_t mem[4 * 2] = {0x6535'2069,
                                          0x9644'3863,
                                          0x3883'3225,
                                          0x2499'2752,
                                          0x8697'6059,
                                          0x8823'4759,
                                          0x6263'2991,
                                          0x7332'5459};
  __uint128_t res[4];
  asm("ld4 {v30.2s-v1.2s}, [%4]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b\n\t"
      "mov %3.16b, v1.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(mem)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x8697'6059'6535'2069ULL, 0));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'9644'3863ULL, 0));
  ASSERT_EQ(res[2], MakeUInt128(0x6263'2991'3883'3225ULL, 0));
  ASSERT_EQ(res[3], MakeUInt128(0x7332'5459'2499'2752ULL, 0));
}

TEST(Arm64InsnTest, Store4MultipleInt32x2) {
  static constexpr uint64_t arg[4] = {0x8697'6059'6535'2069ULL,
                                      0x8823'4759'9644'3863ULL,
                                      0x6263'2991'3883'3225ULL,
                                      0x7332'5459'2499'2752ULL};
  uint64_t res[4];
  asm("mov v30.16b, %0.16b\n\t"
      "mov v31.16b, %1.16b\n\t"
      "mov v0.16b, %2.16b\n\t"
      "mov v1.16b, %3.16b\n\t"
      "st4 {v30.2s-v1.2s}, [%4]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "w"(arg[3]), "r"(res)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], 0x9644'3863'6535'2069ULL);
  ASSERT_EQ(res[1], 0x2499'2752'3883'3225ULL);
  ASSERT_EQ(res[2], 0x8823'4759'8697'6059ULL);
  ASSERT_EQ(res[3], 0x7332'5459'6263'2991ULL);
}

TEST(Arm64InsnTest, Load4MultipleInt32x4) {
  static constexpr uint32_t mem[4 * 4] = {0x6535'2069,
                                          0x9644'3863,
                                          0x3883'3225,
                                          0x2499'2752,
                                          0x8697'6059,
                                          0x8823'4759,
                                          0x6263'2991,
                                          0x7332'5459,
                                          0x1637'4445,
                                          0x4377'5533,
                                          0x2899'4929,
                                          0x1757'0581,
                                          0x5078'9881,
                                          0x5262'1468,
                                          0x5247'1332,
                                          0x6511'3837};
  __uint128_t res[4];
  asm("ld4 {v30.4s-v1.4s}, [%4]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b\n\t"
      "mov %3.16b, v1.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(mem)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x8697'6059'6535'2069ULL, 0x5078'9881'1637'4445ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'9644'3863ULL, 0x5262'1468'4377'5533ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x6263'2991'3883'3225ULL, 0x5247'1332'2899'4929ULL));
  ASSERT_EQ(res[3], MakeUInt128(0x7332'5459'2499'2752ULL, 0x6511'3837'1757'0581ULL));
}

TEST(Arm64InsnTest, Store4MultipleInt32x4) {
  static constexpr __uint128_t arg[4] = {
      MakeUInt128(0x8697'6059'6535'2069ULL, 0x5078'9881'1637'4445ULL),
      MakeUInt128(0x8823'4759'9644'3863ULL, 0x5262'1468'4377'5533ULL),
      MakeUInt128(0x6263'2991'3883'3225ULL, 0x5247'1332'2899'4929ULL),
      MakeUInt128(0x7332'5459'2499'2752ULL, 0x6511'3837'1757'0581ULL)};
  __uint128_t res[4];
  asm("mov v30.16b, %0.16b\n\t"
      "mov v31.16b, %1.16b\n\t"
      "mov v0.16b, %2.16b\n\t"
      "mov v1.16b, %3.16b\n\t"
      "st4 {v30.4s-v1.4s}, [%4]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "w"(arg[3]), "r"(res)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x2499'2752'3883'3225ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'8697'6059ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x4377'5533'1637'4445ULL, 0x1757'0581'2899'4929ULL));
  ASSERT_EQ(res[3], MakeUInt128(0x5262'1468'5078'9881ULL, 0x6511'3837'5247'1332ULL));
}

TEST(Arm64InsnTest, Load4MultipleInt64x2) {
  static constexpr uint64_t mem[4 * 2] = {0x9644'3863'6535'2069,
                                          0x2499'2752'3883'3225,
                                          0x8823'4759'8697'6059,
                                          0x7332'5459'6263'2991,
                                          0x4377'5533'1637'4445,
                                          0x1757'0581'2899'4929,
                                          0x5262'1468'5078'9881,
                                          0x6511'3837'5247'1332};
  __uint128_t res[4];
  asm("ld4 {v30.2d-v1.2d}, [%4]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b\n\t"
      "mov %3.16b, v1.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(mem)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x4377'5533'1637'4445ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x2499'2752'3883'3225ULL, 0x1757'0581'2899'4929ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x8823'4759'8697'6059ULL, 0x5262'1468'5078'9881ULL));
  ASSERT_EQ(res[3], MakeUInt128(0x7332'5459'6263'2991ULL, 0x6511'3837'5247'1332ULL));
}

TEST(Arm64InsnTest, Store4MultipleInt64x2) {
  static constexpr __uint128_t arg[4] = {
      MakeUInt128(0x9644'3863'6535'2069ULL, 0x4377'5533'1637'4445ULL),
      MakeUInt128(0x2499'2752'3883'3225ULL, 0x1757'0581'2899'4929ULL),
      MakeUInt128(0x8823'4759'8697'6059ULL, 0x5262'1468'5078'9881ULL),
      MakeUInt128(0x7332'5459'6263'2991ULL, 0x6511'3837'5247'1332ULL)};
  __uint128_t res[4];
  asm("mov v30.16b, %0.16b\n\t"
      "mov v31.16b, %1.16b\n\t"
      "mov v0.16b, %2.16b\n\t"
      "mov v1.16b, %3.16b\n\t"
      "st4 {v30.2d-v1.2d}, [%4]"
      :
      : "w"(arg[0]), "w"(arg[1]), "w"(arg[2]), "w"(arg[3]), "r"(res)
      : "v30", "v31", "v0", "v1", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x9644'3863'6535'2069ULL, 0x2499'2752'3883'3225ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8823'4759'8697'6059ULL, 0x7332'5459'6263'2991ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x4377'5533'1637'4445ULL, 0x1757'0581'2899'4929ULL));
  ASSERT_EQ(res[3], MakeUInt128(0x5262'1468'5078'9881ULL, 0x6511'3837'5247'1332ULL));
}

TEST(Arm64InsnTest, Load1ReplicateInt8x8) {
  static constexpr uint8_t mem = 0x81U;
  __uint128_t res;
  asm("ld1r {%0.8b}, [%1]" : "=w"(res) : "r"(&mem) : "memory");
  ASSERT_EQ(res, MakeUInt128(0x8181'8181'8181'8181ULL, 0U));
}

TEST(Arm64InsnTest, Load2ReplicateInt16x8) {
  static constexpr uint16_t mem[] = {0x7904, 0x8715};
  __uint128_t res[2];
  asm("ld2r {v6.8h, v7.8h}, [%2]\n\t"
      "mov %0.16b, v6.16b\n\t"
      "mov %1.16b, v7.16b"
      : "=w"(res[0]), "=w"(res[1])
      : "r"(mem)
      : "v6", "v7", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x7904'7904'7904'7904ULL, 0x7904'7904'7904'7904ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x8715'8715'8715'8715ULL, 0x8715'8715'8715'8715ULL));
}

TEST(Arm64InsnTest, Load3ReplicateInt32x4) {
  static constexpr uint32_t mem[] = {0x7871'3710U, 0x6051'0637U, 0x9555'8588U};
  __uint128_t res[3];
  asm("ld3r {v30.4s-v0.4s}, [%3]\n\t"
      "mov %0.16b, v30.16b\n\t"
      "mov %1.16b, v31.16b\n\t"
      "mov %2.16b, v0.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2])
      : "r"(mem)
      : "v30", "v31", "v0", "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x7871'3710'7871'3710ULL, 0x7871'3710'7871'3710ULL));
  ASSERT_EQ(res[1], MakeUInt128(0x6051'0637'6051'0637ULL, 0x6051'0637'6051'0637ULL));
  ASSERT_EQ(res[2], MakeUInt128(0x9555'8588'9555'8588ULL, 0x9555'8588'9555'8588ULL));
}

TEST(Arm64InsnTest, Load4ReplicateInt64x2) {
  static constexpr uint64_t mem[] = {0x8150'7814'6852'6213ULL,
                                     0x3252'4738'3765'1192ULL,
                                     0x9901'5610'9189'7779ULL,
                                     0x2200'8705'7933'9646ULL};
  __uint128_t res[4];
  asm("ld4r {v29.2d-v0.2d}, [%4]\n\t"
      "mov %0.16b, v29.16b\n\t"
      "mov %1.16b, v30.16b\n\t"
      "mov %2.16b, v31.16b\n\t"
      "mov %3.16b, v0.16b"
      : "=w"(res[0]), "=w"(res[1]), "=w"(res[2]), "=w"(res[3])
      : "r"(mem)
      : "v29", "v30", "v31", "v0", "memory");
  ASSERT_EQ(res[0], MakeUInt128(mem[0], mem[0]));
  ASSERT_EQ(res[1], MakeUInt128(mem[1], mem[1]));
  ASSERT_EQ(res[2], MakeUInt128(mem[2], mem[2]));
  ASSERT_EQ(res[3], MakeUInt128(mem[3], mem[3]));
}

TEST(Arm64InsnTest, LoadPairNonTemporarlInt64) {
  static constexpr uint64_t mem[] = {0x3843'6017'3747'4215ULL, 0x2476'0851'5209'9016ULL};
  __uint128_t res[2];
  asm("ldnp %d0, %d1, [%2]" : "=w"(res[0]), "=w"(res[1]) : "r"(mem) : "memory");
  ASSERT_EQ(res[0], MakeUInt128(0x3843'6017'3747'4215ULL, 0U));
  ASSERT_EQ(res[1], MakeUInt128(0x2476'0851'5209'9016ULL, 0U));
}

TEST(Arm64InsnTest, MoviVector2S) {
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES("movi %0.2s, #0xe4")();
  ASSERT_EQ(rd, MakeUInt128(0x0000'00e4'0000'00e4ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MoviVector2D) {
  __uint128_t rd = ASM_INSN_WRAP_FUNC_W_RES("movi %0.2d, #0xff")();
  ASSERT_EQ(rd, MakeUInt128(0x0000'0000'0000'00ffULL, 0x0000'0000'0000'00ffULL));
}

TEST(Arm64InsnTest, MoviVector8B) {
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES("movi %0.8b, #0xda")();
  ASSERT_EQ(res, MakeUInt128(0xdada'dada'dada'dadaULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MoviVector4HShiftBy8) {
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES("movi %0.4h, #0xd1, lsl #8")();
  ASSERT_EQ(res, MakeUInt128(0xd100'd100'd100'd100ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MoviVector2SShiftBy16) {
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES("movi %0.2s, #0x37, msl #16")();
  ASSERT_EQ(res, MakeUInt128(0x0037'ffff'0037'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MvniVector4H) {
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES("mvni %0.4h, #0xbc")();
  ASSERT_EQ(res, MakeUInt128(0xff43'ff43'ff43'ff43ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MvniVector2SShiftBy8) {
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES("mvni %0.2s, #0x24, lsl #8")();
  ASSERT_EQ(res, MakeUInt128(0xffff'dbff'ffff'dbffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, MvniVector2SShiftBy16) {
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES("mvni %0.2s, #0x25, msl #16")();
  ASSERT_EQ(res, MakeUInt128(0xffda'0000'ffda'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, LoadSimdRegPlusReg) {
  __uint128_t array[] = {
      MakeUInt128(0x6517'9806'9411'3528ULL, 0x0131'4701'3047'8164ULL),
      MakeUInt128(0x8672'4229'2465'4366ULL, 0x8009'8067'6928'2382ULL),
  };
  uint64_t offset = 16;
  __uint128_t rd;

  asm("ldr %q0, [%1, %2]" : "=w"(rd) : "r"(array), "r"(offset) : "memory");

  ASSERT_EQ(rd, MakeUInt128(0x8672'4229'2465'4366ULL, 0x8009'8067'6928'2382ULL));
}

TEST(Arm64InsnTest, ExtractNarrowI16x8ToI8x8) {
  __uint128_t arg = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x0011'2233'4455'6677ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("xtn %0.8b, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x1133'5577'2367'abefULL, 0x0ULL));
}

TEST(Arm64InsnTest, ExtractNarrowI32x4ToI16x4) {
  __uint128_t arg = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x0011'2233'4455'6677ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("xtn %0.4h, %1.4s")(arg);
  ASSERT_EQ(res, MakeUInt128(0x2233'6677'4567'cdefULL, 0x0ULL));
}

TEST(Arm64InsnTest, ExtractNarrowI64x2ToI32x2) {
  __uint128_t arg = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x0011'2233'4455'6677ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("xtn %0.2s, %1.2d")(arg);
  ASSERT_EQ(res, MakeUInt128(0x4455'6677'89ab'cdefULL, 0x0ULL));
}

TEST(Arm64InsnTest, ExtractNarrow2Int16x8ToInt8x16) {
  __uint128_t arg1 = MakeUInt128(0x1844'3965'8253'3754ULL, 0x3885'6909'4113'0315ULL);
  __uint128_t arg2 = MakeUInt128(0x6121'8656'1967'3378ULL, 0x6236'2561'2521'6320ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("xtn2 %0.16b, %1.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x6121'8656'1967'3378ULL, 0x8509'1315'4465'5354ULL));
}

TEST(Arm64InsnTest, LoadLiteralSimd) {
  // We call an external assembly function to perform LDR literal because we
  // need to place the literal in .rodata.  The literal placed in .text would
  // trigger a segfault.
  ASSERT_EQ(get_fp64_literal(), 0x0123'4567'89ab'cdefULL);
}

TEST(Arm64InsnTest, AbsInt64x1) {
  __uint128_t arg = MakeUInt128(0xffff'ffff'ffff'fffdULL, 0xdead'beef'0123'4567ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("abs %d0, %d1")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0003ULL, 0x0ULL));
}

TEST(Arm64InsnTest, AbsInt8x8) {
  __uint128_t arg = MakeUInt128(0x0001'027e'7f80'81ffULL, 0x0123'4567'89ab'cdefULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("abs %0.8b, %1.8b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0001'027e'7f80'7f01ULL, 0x0ULL));
}

TEST(Arm64InsnTest, UseV31) {
  __uint128_t res;

  asm("movi v31.2d, #0xffffffffffffffff\n\t"
      "mov %0.16b, v31.16b"
      : "=w"(res)
      :
      : "v31");

  ASSERT_EQ(res, MakeUInt128(~0ULL, ~0ULL));
}

TEST(Arm64InsnTest, AddHighNarrowInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x2296'6171'1963'7792ULL, 0x1337'5751'1495'9501ULL);
  __uint128_t arg2 = MakeUInt128(0x0941'2147'2213'1794ULL, 0x7647'7726'2241'4254ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("addhn %0.8b, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x89ce'36d7'2b82'3b8fULL, 0x0ULL));
}

TEST(Arm64InsnTest, AddHighNarrowUpperInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x6561'8093'7734'4403ULL, 0x0707'4692'1120'1913ULL);
  __uint128_t arg2 = MakeUInt128(0x6095'7527'0695'7220ULL, 0x9175'6711'6722'9109ULL);
  __uint128_t arg3 = MakeUInt128(0x5797'8771'8556'0845ULL, 0x5296'5412'6654'0853ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("addhn2 %0.16b, %1.8h, %2.8h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x5797'8771'8556'0845ULL, 0x98ad'78aa'c5f5'7db6ULL));
}

TEST(Arm64InsnTest, SubHighNarrowInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x4978'1893'1297'8482ULL, 0x1682'9989'4872'2658ULL);
  __uint128_t arg2 = MakeUInt128(0x1210'8357'9151'3698ULL, 0x8209'1444'2100'6751ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("subhn %0.8b, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x9485'27bf'3795'814dULL, 0x0ULL));
}

TEST(Arm64InsnTest, SubHighNarrowUpperInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x5324'9441'6680'3962ULL, 0x6579'7877'1855'6084ULL);
  __uint128_t arg2 = MakeUInt128(0x1066'5879'6998'1635ULL, 0x7473'6384'0525'7145ULL);
  __uint128_t arg3 = MakeUInt128(0x3142'9809'1906'5925ULL, 0x0937'2216'9646'1515ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("subhn2 %0.16b, %1.8h, %2.8h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x3142'9809'1906'5925ULL, 0xf114'13ef'423b'fc23ULL));
}

TEST(Arm64InsnTest, RoundingAddHighNarrowInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x8039'6265'7978'7718ULL, 0x5560'8455'2965'4126ULL);
  __uint128_t arg2 = MakeUInt128(0x3440'1712'7494'7042ULL, 0x0562'2305'3899'4561ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("raddhn %0.8b, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x5ba7'6287'b479'eee7ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, RoundingSubHighNarrowInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x3063'4328'5878'5698ULL, 0x3052'3580'8933'0657ULL);
  __uint128_t arg2 = MakeUInt128(0x0216'4715'5097'9259ULL, 0x2309'9079'6547'3761ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("rsubhn %0.8b, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0da5'24cf'2efc'08c4ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, ScalarPairwiseAddInt8x2) {
  __uint128_t arg = MakeUInt128(0x6257'5916'3330'3910ULL, 0x7225'3837'4218'2140ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("addp %d0, %1.2d")(arg);
  ASSERT_EQ(res, MakeUInt128(0xd47c'914d'7548'5a50ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, AddAcrossInt8x8) {
  __uint128_t arg = MakeUInt128(0x0681'2160'2876'4962ULL, 0x8674'4604'7746'4915ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("addv %b0, %1.8b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x51ULL, 0x0ULL));
}

TEST(Arm64InsnTest, AddAcrossInt8x16) {
  __uint128_t arg = MakeUInt128(0x0102'0304'0506'0708ULL, 0x090a'0b0c'0d0e'0f10ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("addv %b0, %1.16b")(arg);
  // Sum of 1 to 16 is (16 * 17) / 2 = 136 (0x88)
  ASSERT_EQ(res, MakeUInt128(0x88ULL, 0x0ULL));
}

TEST(Arm64InsnTest, AddAcrossInt16x4) {
  __uint128_t arg = MakeUInt128(0x0001'0002'0003'0004ULL, 0x0ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("addv %h0, %1.4h")(arg);
  // Sum of 1 to 4 is 10 (0xa)
  ASSERT_EQ(res, MakeUInt128(0x000aULL, 0x0ULL));
}

TEST(Arm64InsnTest, AddAcrossInt16x8) {
  __uint128_t arg = MakeUInt128(0x0001'0002'0003'0004ULL, 0x0005'0006'0007'0008ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("addv %h0, %1.8h")(arg);
  // Sum of 1 to 8 is 36 (0x24)
  ASSERT_EQ(res, MakeUInt128(0x0024ULL, 0x0ULL));
}

TEST(Arm64InsnTest, AddAcrossInt32x4) {
  __uint128_t arg = MakeUInt128(0x0000'0001'0000'0002ULL, 0x0000'0003'0000'0004ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("addv %s0, %1.4s")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'000aULL, 0x0ULL));
}

TEST(Arm64InsnTest, SignedAddLongAcrossInt16x8) {
  __uint128_t arg = MakeUInt128(0x9699'5573'7727'3756ULL, 0x6761'5527'1139'2258ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("saddlv %s0, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0001'8aa2ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, UnsignedAddLongAcrossInt16x8) {
  __uint128_t arg = MakeUInt128(0x7986'3965'2296'1312ULL, 0x8017'8267'9717'2898ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("uaddlv %s0, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0002'aac0ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SignedMaximumAcrossInt16x8) {
  __uint128_t arg = MakeUInt128(0x8482'0659'6737'9473ULL, 0x1680'8641'5645'6505ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("smaxv %h0, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'6737ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SignedMinimumAcrossInt16x8) {
  __uint128_t arg = MakeUInt128(0x6772'5304'3182'5197ULL, 0x5791'6792'9699'6504ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("sminv %h0, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'9699ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, UnsignedMaximumAcrossInt16x8) {
  __uint128_t arg = MakeUInt128(0x6500'3780'7046'6126ULL, 0x4706'0214'5750'5793ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("umaxv %h0, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'7046ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, UnsignedMinimumAcrossInt16x8) {
  __uint128_t arg = MakeUInt128(0x5223'5723'9739'5128ULL, 0x8181'6405'9785'9142ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("uminv %h0, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'5128ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CountLeadingZerosI8x8) {
  __uint128_t arg = MakeUInt128(0x1452'6356'0827'7857ULL, 0x7134'2757'7896'0917ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("clz %0.8b, %1.8b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0301'0101'0402'0101ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, CountLeadingSignBitsI8x8) {
  __uint128_t arg = MakeUInt128(0x8925'8923'5420'1995ULL, 0x6112'1290'2196'0864ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cls %0.8b, %1.8b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0001'0001'0001'0200ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, Cnt) {
  __uint128_t arg = MakeUInt128(0x9835'4848'7562'5298ULL, 0x7524'2387'3077'5595ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("cnt %0.16b, %1.16b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0304'0202'0503'0303ULL, 0x0502'0304'0206'0404ULL));
}

TEST(Arm64InsnTest, SimdScalarMove) {
  __uint128_t arg = MakeUInt128(0x1433'3454'7762'4168ULL, 0x6251'8983'5694'8556ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("mov %b0, %1.b[5]")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0034ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SimdVectorElemDuplicate) {
  __uint128_t arg = MakeUInt128(0x3021'6471'5509'7925ULL, 0x9230'9907'9654'7376ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("dup %0.8b, %1.b[5]")(arg);
  ASSERT_EQ(res, MakeUInt128(0x6464'6464'6464'6464ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SimdVectorElemDuplicateInt16AtIndex7) {
  __uint128_t arg = MakeUInt128(0x2582'2620'5224'8940ULL, 0x7726'7194'7826'8482ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("dup %0.4h, %1.h[7]")(arg);
  ASSERT_EQ(res, MakeUInt128(0x7726'7726'7726'7726ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SimdVectorElemInsert) {
  __uint128_t arg1 = MakeUInt128(0x7120'8443'3573'2654ULL, 0x8938'2391'1932'5974ULL);
  __uint128_t arg2 = MakeUInt128(0x7656'1809'3773'4440ULL, 0x3070'7469'2112'0191ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("mov %0.s[2], %1.s[1]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7656'1809'3773'4440ULL, 0x3070'7469'7120'8443ULL));
}

TEST(Arm64InsnTest, NegateInt64x1) {
  constexpr auto AsmNeg = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("neg %d0, %d1");
  __uint128_t arg1 = MakeUInt128(0x8389'5228'6847'8312ULL, 0x3552'6582'1314'4957ULL);
  ASSERT_EQ(AsmNeg(arg1), MakeUInt128(0x7c76'add7'97b8'7ceeULL, 0x0000'0000'0000'0000ULL));

  __uint128_t arg2 = MakeUInt128(1ULL << 63, 0U);
  ASSERT_EQ(AsmNeg(arg2), MakeUInt128(1ULL << 63, 0U));
}

TEST(Arm64InsnTest, NegateInt16x8) {
  __uint128_t arg = MakeUInt128(0x4411'0104'4682'3252ULL, 0x7162'0105'2652'2721ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("neg %0.8h, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0xbbef'fefc'b97e'cdaeULL, 0x8e9e'fefb'd9ae'd8dfULL));
}

TEST(Arm64InsnTest, NotI8x8) {
  __uint128_t arg = MakeUInt128(0x6205'6476'9312'5705ULL, 0x8635'6620'1855'8100ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("not %0.8b, %1.8b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x9dfa'9b89'6ced'a8faULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, RbitInt8x8) {
  __uint128_t arg = MakeUInt128(0x4713'2962'1073'4043ULL, 0x7518'9573'5961'4589ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("rbit %0.8b, %1.8b")(arg);
  ASSERT_EQ(res, MakeUInt128(0xe2c8'9446'08ce'02c2ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, Rev16Int8x16) {
  __uint128_t arg = MakeUInt128(0x9904'8010'9412'1472ULL, 0x2131'7947'6477'7262ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("rev16 %0.16b, %1.16b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0499'1080'1294'7214ULL, 0x3121'4779'7764'6272ULL));
}

TEST(Arm64InsnTest, Rev32Int16x8) {
  __uint128_t arg = MakeUInt128(0x8662'2371'7215'9160ULL, 0x7716'6925'4748'7389ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("rev32 %0.8h, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x2371'8662'9160'7215ULL, 0x6925'7716'7389'4748ULL));
}

TEST(Arm64InsnTest, Rev64Int32x4) {
  __uint128_t arg = MakeUInt128(0x5306'7360'9657'1209ULL, 0x1807'6383'2716'6416ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("rev64 %0.4s, %1.4s")(arg);
  ASSERT_EQ(res, MakeUInt128(0x9657'1209'5306'7360ULL, 0x2716'6416'1807'6383ULL));
}

TEST(Arm64InsnTest, TblInt8x8) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x0104'0115'0912'0605ULL, 0x0315'0809'0709'1312ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("tbl %0.8b, {%1.16b}, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1144'1100'9900'6655ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, TblInt8x16) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x0905'0608'0801'0408ULL, 0x0506'0002'0603'0202ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("tbl %0.16b, {%1.16b}, %2.16b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x9955'6688'8811'4488ULL, 0x5566'0022'6633'2222ULL));
}

TEST(Arm64InsnTest, Tbl2Int8x16) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x7665'5443'3221'10ffULL, 0xfeed'dccb'baa9'9887ULL);
  __uint128_t arg3 = MakeUInt128(0x0224'0528'0002'0910ULL, 0x1807'2803'1900'2203ULL);
  __uint128_t res;

  // Hardcode v30 and v0 so that the TBL instruction gets consecutive registers.
  asm("mov v31.16b, %1.16b\n\t"
      "mov v0.16b, %2.16b\n\t"
      "tbl %0.16b, {v31.16b, v0.16b}, %3.16b"
      : "=w"(res)
      : "w"(arg1), "w"(arg2), "w"(arg3)
      : "v31", "v0");

  ASSERT_EQ(res, MakeUInt128(0x2200'5500'0022'99ffULL, 0x8777'0033'9800'0033ULL));
}

TEST(Arm64InsnTest, Tbl3Int8x16) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x7665'5443'3221'10ffULL, 0xfeed'dccb'baa9'9887ULL);
  __uint128_t arg3 = MakeUInt128(0x7060'5040'3020'1000ULL, 0xf0e0'd0c0'b0a0'9080ULL);
  __uint128_t arg4 = MakeUInt128(0x0718'2640'3929'1035ULL, 0x3526'1900'4021'1304ULL);
  __uint128_t res;

  // Hardcode v0, v1, and v2 so that the TBL instruction gets consecutive registers.
  asm("mov v30.16b, %1.16b\n\t"
      "mov v31.16b, %2.16b\n\t"
      "mov v0.16b, %3.16b\n\t"
      "tbl %0.16b, {v30.16b-v0.16b}, %4.16b"
      : "=w"(res)
      : "w"(arg1), "w"(arg2), "w"(arg3), "w"(arg4)
      : "v0", "v1", "v2");

  ASSERT_EQ(res, MakeUInt128(0x7787'6000'0090'ff00ULL, 0x0060'9800'0010'3244ULL));
}

TEST(Arm64InsnTest, Tbl4Int8x16) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x7665'5443'3221'10ffULL, 0xfeed'dccb'baa9'9887ULL);
  __uint128_t arg3 = MakeUInt128(0x7060'5040'3020'1000ULL, 0xf0e0'd0c0'b0a0'9080ULL);
  __uint128_t arg4 = MakeUInt128(0x7f6f'5f4f'3f2f'1fffULL, 0xffef'dfcf'bfaf'9f8fULL);
  __uint128_t arg5 = MakeUInt128(0x0718'2640'3929'1035ULL, 0x3526'1900'4021'1304ULL);
  __uint128_t res;

  // Hardcode v30, v31, v0, and v1 so that the TBX instruction gets consecutive registers.
  asm("mov v30.16b, %1.16b\n\t"
      "mov v31.16b, %2.16b\n\t"
      "mov v0.16b, %3.16b\n\t"
      "mov v1.16b, %4.16b\n\t"
      "tbl %0.16b, {v30.16b-v1.16b}, %5.16b"
      : "=w"(res)
      : "w"(arg1), "w"(arg2), "w"(arg3), "w"(arg4), "w"(arg5)
      : "v30", "v31", "v0", "v1");

  ASSERT_EQ(res, MakeUInt128(0x7787'6000'9f90'ff5fULL, 0x5f60'9800'0010'3244ULL));
}

TEST(Arm64InsnTest, TbxInt8x16) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x0915'0618'0801'0408ULL, 0x0516'0002'0603'1202ULL);
  __uint128_t arg3 = MakeUInt128(0x6668'5592'3356'5463ULL, 0x9138'3631'8574'5698ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("tbx %0.16b, {%1.16b}, %2.16b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x9968'6692'8811'4488ULL, 0x5538'0022'6633'5622ULL));
}

TEST(Arm64InsnTest, Tbx2Int8x16) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x7665'5443'3221'10ffULL, 0xfeed'dccb'baa9'9887ULL);
  __uint128_t arg3 = MakeUInt128(0x0224'0528'0002'0910ULL, 0x1807'2803'1900'2203ULL);
  __uint128_t res = MakeUInt128(0x7494'0784'8844'2377ULL, 0x2175'1543'3426'0306ULL);

  // Hardcode v0 and v1 so that the TBX instruction gets consecutive registers.
  asm("mov v0.16b, %1.16b\n\t"
      "mov v1.16b, %2.16b\n\t"
      "tbx %0.16b, {v0.16b, v1.16b}, %3.16b"
      : "=w"(res)
      : "w"(arg1), "w"(arg2), "w"(arg3), "0"(res)
      : "v0", "v1");

  ASSERT_EQ(res, MakeUInt128(0x2294'5584'0022'99ffULL, 0x8777'1533'9800'0333ULL));
}

TEST(Arm64InsnTest, Tbx3Int8x16) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x7665'5443'3221'10ffULL, 0xfeed'dccb'baa9'9887ULL);
  __uint128_t arg3 = MakeUInt128(0x7060'5040'3020'1000ULL, 0xf0e0'd0c0'b0a0'9080ULL);
  __uint128_t arg4 = MakeUInt128(0x0718'2640'3929'1035ULL, 0x3526'1900'4021'1304ULL);
  __uint128_t res = MakeUInt128(0x0136'7763'1084'9135ULL, 0x1615'6422'6984'7507ULL);

  // Hardcode v0, v1, and v2 so that the TBX instruction gets consecutive registers.
  asm("mov v0.16b, %1.16b\n\t"
      "mov v1.16b, %2.16b\n\t"
      "mov v2.16b, %3.16b\n\t"
      "tbx %0.16b, {v0.16b, v1.16b, v2.16b}, %4.16b"
      : "=w"(res)
      : "w"(arg1), "w"(arg2), "w"(arg3), "w"(arg4), "0"(res)
      : "v0", "v1", "v2");

  ASSERT_EQ(res, MakeUInt128(0x7787'6063'1090'ff35ULL, 0x1660'9800'6910'3244ULL));
}

TEST(Arm64InsnTest, Tbx4Int8x16) {
  __uint128_t arg1 = MakeUInt128(0x7766'5544'3322'1100ULL, 0xffee'ddcc'bbaa'9988ULL);
  __uint128_t arg2 = MakeUInt128(0x7665'5443'3221'10ffULL, 0xfeed'dccb'baa9'9887ULL);
  __uint128_t arg3 = MakeUInt128(0x7060'5040'3020'1000ULL, 0xf0e0'd0c0'b0a0'9080ULL);
  __uint128_t arg4 = MakeUInt128(0x7f6f'5f4f'3f2f'1fffULL, 0xffef'dfcf'bfaf'9f8fULL);
  __uint128_t arg5 = MakeUInt128(0x0718'2640'3929'1035ULL, 0x3526'1900'4021'1304ULL);
  __uint128_t res = MakeUInt128(0x5818'3196'3763'7076ULL, 0x1799'1919'2035'7958ULL);

  // Hardcode v0, v1, v2, and v3 so that the TBX instruction gets consecutive registers.
  asm("mov v0.16b, %1.16b\n\t"
      "mov v1.16b, %2.16b\n\t"
      "mov v2.16b, %3.16b\n\t"
      "mov v3.16b, %4.16b\n\t"
      "tbx %0.16b, {v0.16b-v3.16b}, %5.16b"
      : "=w"(res)
      : "w"(arg1), "w"(arg2), "w"(arg3), "w"(arg4), "w"(arg5), "0"(res)
      : "v0", "v1", "v2", "v3");

  ASSERT_EQ(res, MakeUInt128(0x7787'6096'9f90'ff5fULL, 0x5f60'9800'2010'3244ULL));
}

TEST(Arm64InsnTest, Trn1Int8x8) {
  __uint128_t arg1 = MakeUInt128(0x2075'9167'2970'0785ULL, 0x0580'7171'8638'1054ULL);
  __uint128_t arg2 = MakeUInt128(0x2786'0990'5569'0013ULL, 0x4137'1823'6837'0991ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("trn1 %0.8b, %1.8b, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x8675'9067'6970'1385ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, Trn2Int16x8) {
  __uint128_t arg1 = MakeUInt128(0x6685'5923'3565'4639ULL, 0x1383'6318'5745'6981ULL);
  __uint128_t arg2 = MakeUInt128(0x7494'0784'8844'2377ULL, 0x2175'1543'3426'0306ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("trn2 %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7494'6685'8844'3565ULL, 0x2175'1383'3426'5745ULL));
}

TEST(Arm64InsnTest, Uzp1Int8x8) {
  __uint128_t arg1 = MakeUInt128(0x4954'8931'3939'4489ULL, 0x9216'1255'2559'7701ULL);
  __uint128_t arg2 = MakeUInt128(0x2783'4679'2610'1995ULL, 0x5852'2471'7220'1777ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uzp1 %0.8b, %1.8b, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x8379'1095'5431'3989ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, Uzp2Int16x8) {
  __uint128_t arg1 = MakeUInt128(0x6745'6423'9058'5850ULL, 0x2167'1903'1395'2629ULL);
  __uint128_t arg2 = MakeUInt128(0x3620'1294'7691'8749ULL, 0x7519'1011'4723'1528ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uzp2 %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x2167'1395'6745'9058ULL, 0x7519'4723'3620'7691ULL));
}

TEST(Arm64InsnTest, Zip2Int64x2) {
  __uint128_t arg1 = MakeUInt128(0x1494'2714'1009'3913ULL, 0x6913'8107'2581'3781ULL);
  __uint128_t arg2 = MakeUInt128(0x3578'9400'5599'5001ULL, 0x8354'2511'8417'2136ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uzp2 %0.2d, %1.2d, %2.2d")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x6913'8107'2581'3781ULL, 0x8354'2511'8417'2136ULL));
}

TEST(Arm64InsnTest, Zip1Int8x8) {
  __uint128_t arg1 = MakeUInt128(0x7499'2356'3025'4947ULL, 0x8024'9011'4195'2123ULL);
  __uint128_t arg2 = MakeUInt128(0x3331'2394'8049'4707ULL, 0x9119'1532'6734'3028ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("zip1 %0.8b, %1.8b, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x8030'4925'4749'0747ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, Zip1Int64x2) {
  __uint128_t arg1 = MakeUInt128(0x9243'5301'3677'6310ULL, 0x8491'3516'1564'2269ULL);
  __uint128_t arg2 = MakeUInt128(0x0551'1995'8183'1963ULL, 0x7637'0761'7991'9192ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("zip1 %0.2d, %1.2d, %2.2d")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x9243'5301'3677'6310ULL, 0x0551'1995'8183'1963ULL));
}

TEST(Arm64InsnTest, Zip2Int16x8) {
  __uint128_t arg1 = MakeUInt128(0x5831'8327'1314'2517ULL, 0x0296'9234'8896'2766ULL);
  __uint128_t arg2 = MakeUInt128(0x2934'5958'8970'6953ULL, 0x6534'9406'0340'2166ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("zip2 %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0340'8896'2166'2766ULL, 0x6534'0296'9406'9234ULL));
}

TEST(Arm64InsnTest, SignedMaxInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9901'5734'6610'2371ULL, 0x2235'4789'1129'2547ULL);
  __uint128_t arg2 = MakeUInt128(0x4922'1576'5045'0812ULL, 0x0677'1735'7120'2718ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smax %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x4922'5734'6610'2371ULL, 0x2235'4789'7120'2718ULL));
}

TEST(Arm64InsnTest, SignedMinInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x7820'3856'5390'9910ULL, 0x4775'9414'1321'5432ULL);
  __uint128_t arg2 = MakeUInt128(0x0084'5312'1406'5935ULL, 0x8090'4127'1135'9200ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smin %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0084'3856'1406'9910ULL, 0x8090'9414'1135'9200ULL));
}

TEST(Arm64InsnTest, SignedMaxPairwiseInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x6998'4698'8477'0232ULL, 0x3823'8400'5565'5517ULL);
  __uint128_t arg2 = MakeUInt128(0x3272'8676'0072'4817ULL, 0x2987'6375'6981'6335ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smaxp %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x3823'5565'6998'0232ULL, 0x6375'6981'3272'4817ULL));
}

TEST(Arm64InsnTest, SignedMinPairwiseInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x8865'7015'6850'1691ULL, 0x8647'4885'4167'9154ULL);
  __uint128_t arg2 = MakeUInt128(0x1821'5535'5973'2353ULL, 0x0686'0430'1067'5760ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sminp %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x8647'9154'8865'1691ULL, 0x0430'1067'1821'2353ULL));
}

TEST(Arm64InsnTest, UnsignedMaxInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x7639'9759'7461'9383ULL, 0x5845'7491'5988'0976ULL);
  __uint128_t arg2 = MakeUInt128(0x5928'4936'9594'1434ULL, 0x0814'6852'9815'0539ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umax %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7639'9759'9594'9383ULL, 0x5845'7491'9815'0976ULL));
}

TEST(Arm64InsnTest, UnsignedMinInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x2888'7737'1766'3748ULL, 0x6027'6606'3496'0353ULL);
  __uint128_t arg2 = MakeUInt128(0x6983'3495'1510'1986ULL, 0x4269'8878'4717'1939ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umin %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x2888'3495'1510'1986ULL, 0x4269'6606'3496'0353ULL));
}

TEST(Arm64InsnTest, UnsignedMaxPairwiseInt8x16) {
  __uint128_t arg1 = MakeUInt128(0x2211'2211'1122'1122ULL, 0x6655'6655'5566'5566ULL);
  __uint128_t arg2 = MakeUInt128(0x4433'4433'3344'3344ULL, 0x8877'8877'7788'7788ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umaxp %0.16b, %1.16b, %2.16b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x6666'6666'2222'2222ULL, 0x8888'8888'4444'4444ULL));
}

TEST(Arm64InsnTest, UnsignedMinPairwiseInt8x16) {
  __uint128_t arg1 = MakeUInt128(0x2211'2211'1122'1122ULL, 0x6655'6655'5566'5566ULL);
  __uint128_t arg2 = MakeUInt128(0x4433'4433'3344'3344ULL, 0x8877'8877'7788'7788ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uminp %0.16b, %1.16b, %2.16b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x5555'5555'1111'1111ULL, 0x7777'7777'3333'3333ULL));
}

TEST(Arm64InsnTest, UnsignedMaxPairwiseInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x1318'5835'8406'6747ULL, 0x2370'2971'4978'5084ULL);
  __uint128_t arg2 = MakeUInt128(0x4570'2494'1398'3163ULL, 0x4332'3789'7595'5680ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umaxp %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x2971'5084'5835'8406ULL, 0x4332'7595'4570'3163ULL));
}

TEST(Arm64InsnTest, UnsignedMinPairwiseInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9538'1217'9131'9145ULL, 0x1350'0993'8463'1177ULL);
  __uint128_t arg2 = MakeUInt128(0x7769'0554'8102'8850ULL, 0x2080'8580'0878'1157ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uminp %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0993'1177'1217'9131ULL, 0x2080'0878'0554'8102ULL));
}

TEST(Arm64InsnTest, SignedHalvingAddInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x1021'9447'1971'3869ULL, 0x2560'8416'2451'1239ULL);
  __uint128_t arg2 = MakeUInt128(0x8062'0113'1845'4124ULL, 0x4782'0501'1079'8760ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("shadd %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xc841'caad'18db'3cc6ULL, 0x3671'c48b'1a65'ccccULL));
}

TEST(Arm64InsnTest, SignedHalvingSubInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9041'2108'7303'2402ULL, 0x0106'8534'1947'2304ULL);
  __uint128_t arg2 = MakeUInt128(0x7666'6721'7498'6986ULL, 0x8547'0767'8120'5124ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("shsub %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x8ced'dcf3'ff35'dd3eULL, 0x3ddf'bee6'4c13'e8f0ULL));
}

TEST(Arm64InsnTest, SignedRoundingHalvingAddInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x5871'4878'3989'0810ULL, 0x7429'5309'4106'0596ULL);
  __uint128_t arg2 = MakeUInt128(0x9443'1584'7753'9700ULL, 0x9439'8839'4914'4323ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("srhadd %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xf65a'2efe'586e'cf88ULL, 0x0431'eda1'450d'245dULL));
}

TEST(Arm64InsnTest, SignedAbsoluteDifferenceInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x1349'6075'0111'6498ULL, 0x3278'5635'3161'4516ULL);
  __uint128_t arg2 = MakeUInt128(0x8457'6956'8710'9002ULL, 0x9997'6984'1263'2665ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sabd %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x8ef2'08e1'7a01'd496ULL, 0x98e1'134f'1efe'1eb1ULL));
}

TEST(Arm64InsnTest, SignedAbsoluteDifferenceLongInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x7419'8509'7334'6267ULL, 0x9332'1072'6868'7076ULL);
  __uint128_t arg2 = MakeUInt128(0x8062'6399'1936'1965ULL, 0x0440'9954'2167'6278ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sabdl %0.4s, %1.4h, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'59fe'0000'4902ULL, 0x0000'f3b7'0000'de90ULL));
}

TEST(Arm64InsnTest, SignedAbsoluteDifferenceLongUpperInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x4980'5596'1033'0799ULL, 0x4145'3477'8457'4699ULL);
  __uint128_t arg2 = MakeUInt128(0x9921'2859'9999'3996ULL, 0x1228'1615'2193'1488ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sabdl2 %0.4s, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'9d3c'0000'3211ULL, 0x0000'2f1d'0000'1e62ULL));
}

TEST(Arm64InsnTest, SignedAbsoluteDifferenceAccumulateInt16x8) {
  // The lowest element tests the overflow.
  __uint128_t arg1 = MakeUInt128(0x8967'0031'9258'7fffULL, 0x9410'5105'3358'4384ULL);
  __uint128_t arg2 = MakeUInt128(0x6560'2339'1796'8000ULL, 0x6784'4763'7084'7497ULL);
  __uint128_t arg3 = MakeUInt128(0x8333'6555'7900'5555ULL, 0x1914'7319'8862'7135ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("saba %0.8h, %1.8h, %2.8h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x5f2c'885d'fe3e'5554ULL, 0xec88'7cbb'c58e'a248ULL));
}

TEST(Arm64InsnTest, SignedAbsoluteDifferenceAccumulateInt32x4) {
  // The lowest element tests the overflow.
  __uint128_t arg1 = MakeUInt128(0x8967'0031'7fff'ffffULL, 0x9410'5105'3358'4384ULL);
  __uint128_t arg2 = MakeUInt128(0x6560'2339'8000'0000ULL, 0x6784'4763'7084'7497ULL);
  __uint128_t arg3 = MakeUInt128(0x8333'6555'aaaa'5555ULL, 0x1914'7319'8862'7135ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("saba %0.4s, %1.4s, %2.4s")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x5f2c'885d'aaaa'5554ULL, 0xec88'6977'c58e'a248ULL));
}

TEST(Arm64InsnTest, SignedAbsoluteDifferenceAccumulateLongInt16x4) {
  __uint128_t arg1 = MakeUInt128(0x078'4641'6745'2167ULL, 0x719'0483'1096'7671ULL);
  __uint128_t arg2 = MakeUInt128(0x344'3494'8192'6268ULL, 0x110'7399'4825'0607ULL);
  __uint128_t arg3 = MakeUInt128(0x949'5073'5031'6901ULL, 0x731'8521'1955'2635ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("sabal %0.4s, %1.4h, %2.4h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x094a'3626'5031'aa02ULL, 0x0731'87ed'1955'37e2ULL));
}

TEST(Arm64InsnTest, SignedAbsoluteDifferenceLongInt32x2) {
  __uint128_t arg1 = MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'8000'0000ULL, 0x0000'0000'0000'0000ULL);
  __uint128_t arg3 = MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("sabal %0.2d, %1.2s, %2.2s")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SignedAbsoluteDifferenceAccumulateLongUpperInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x690'9434'7048'2932ULL, 0x414'0411'1465'4092ULL);
  __uint128_t arg2 = MakeUInt128(0x988'3444'3515'9133ULL, 0x010'7739'4411'1840ULL);
  __uint128_t arg3 = MakeUInt128(0x410'7684'9810'6634ULL, 0x241'0482'3935'8274ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("sabal2 %0.4s, %1.8h, %2.8h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x0410'a630'9810'8e86ULL, 0x0241'0886'3935'f59cULL));
}

TEST(Arm64InsnTest, UnsignedHalvingAddInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x4775'3798'5379'9732ULL, 0x2344'5612'2785'8432ULL);
  __uint128_t arg2 = MakeUInt128(0x9684'6647'5133'3657ULL, 0x3692'3872'0146'4723ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uhadd %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x6efc'4eef'5256'66c4ULL, 0x2ceb'4742'1465'65aaULL));
}

TEST(Arm64InsnTest, UnsignedHalvingSubInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x9926'8843'4959'2876ULL, 0x1240'0755'8756'9464ULL);
  __uint128_t arg2 = MakeUInt128(0x1370'5625'1400'1179ULL, 0x7133'1662'0715'3715ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uhsub %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x42db'190f'1aac'0b7eULL, 0xd086'f879'4020'2ea7ULL));
}

TEST(Arm64InsnTest, UnsignedRoundingHalvingAddInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x5066'5339'8573'8887ULL, 0x8661'4762'9443'4140ULL);
  __uint128_t arg2 = MakeUInt128(0x1049'8889'9316'0051ULL, 0x2076'7810'3588'6116ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("urhadd %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x3058'6de1'8c45'446cULL, 0x536c'5fb9'64e6'512bULL));
}

TEST(Arm64InsnTest, UnsignedAbsoluteDifferenceInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x8574'6646'0772'2834ULL, 0x1540'3114'4152'9418ULL);
  __uint128_t arg2 = MakeUInt128(0x8047'8254'3876'1770ULL, 0x7904'3000'1566'9867ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uabd %0.8h, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x052d'1c0e'3104'10c4ULL, 0x63c4'0114'2bec'044fULL));
}

TEST(Arm64InsnTest, UnsignedAbsoluteDifferenceLongInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x1614'5855'0583'9727ULL, 0x4209'8090'9781'7293ULL);
  __uint128_t arg2 = MakeUInt128(0x2393'0106'7663'8682ULL, 0x4040'1113'0402'4700ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uabdl %0.4s, %1.4h, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'70e0'0000'10a5ULL, 0x0000'0d7f'0000'574fULL));
}

TEST(Arm64InsnTest, UnsignedAbsoluteDifferenceLongUpperInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x0347'9995'8886'7695ULL, 0x0161'2497'2282'0403ULL);
  __uint128_t arg2 = MakeUInt128(0x0399'5463'2788'3069ULL, 0x5976'2493'6151'0102ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uabdl2 %0.4s, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'3ecf'0000'0301ULL, 0x0000'5815'0000'0004ULL));
}

TEST(Arm64InsnTest, UnsignedAbsoluteDifferenceAccumulateInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x0857'4664'6077'2283ULL, 0x4154'0311'4415'2941ULL);
  __uint128_t arg2 = MakeUInt128(0x8804'7825'4387'6177ULL, 0x0790'4300'0156'6986ULL);
  __uint128_t arg3 = MakeUInt128(0x7767'9576'0909'9669ULL, 0x3607'5594'9651'5273ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("uaba %0.8h, %1.8h, %2.8h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0xf714'c737'25f9'd55dULL, 0x6fcb'9583'd910'92b8ULL));
}

TEST(Arm64InsnTest, UnsignedAbsoluteDifferenceAccumulateLongInt16x4) {
  __uint128_t arg1 = MakeUInt128(0x8343'4170'4415'7348ULL, 0x2481'8333'0164'0566ULL);
  __uint128_t arg2 = MakeUInt128(0x9596'6886'6769'5634ULL, 0x9141'6328'4264'1497ULL);
  __uint128_t arg3 = MakeUInt128(0x4533'3499'9948'0002ULL, 0x6699'8758'8815'9350ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("uabal %0.4s, %1.4h, %2.4h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x4533'57ed'9948'1d16ULL, 0x6699'99ab'8815'ba66ULL));
}

TEST(Arm64InsnTest, UnsignedAbsoluteDifferenceAccumulateLongUpperInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x998'6855'4170'3188ULL, 0x778'8675'9290'2607ULL);
  __uint128_t arg2 = MakeUInt128(0x043'2126'6617'9192ULL, 0x352'0938'2278'7888ULL);
  __uint128_t arg3 = MakeUInt128(0x988'6335'9911'6081ULL, 0x235'3555'7046'4634ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("uabal2 %0.4s, %1.8h, %2.8h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x0988'd34d'9911'b302ULL, 0x0235'397b'7046'c371ULL));
}

TEST(Arm64InsnTest, SignedAddLongPairwiseInt8x8) {
  __uint128_t arg = MakeUInt128(0x6164'4110'9625'6633ULL, 0x7305'4092'1951'9675ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("saddlp %0.4h, %1.8b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x00c5'0051'ffbb'0099ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SignedAddLongPairwiseInt8x16) {
  __uint128_t arg = MakeUInt128(0x6164'4110'9625'6633ULL, 0x7305'4092'1951'9675ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("saddlp %0.8h, %1.16b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x00c5'0051'ffbb'0099ULL, 0x0078'ffd2'006a'000bULL));
}

TEST(Arm64InsnTest, SignedAddLongPairwiseInt16x4) {
  __uint128_t arg = MakeUInt128(0x6164'4110'9625'6633ULL, 0x7305'4092'1951'9675ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("saddlp %0.2s, %1.4h")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0000'a274'ffff'fc58ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SignedAddLongPairwiseInt16x8) {
  __uint128_t arg = MakeUInt128(0x6164'4110'9625'6633ULL, 0x7305'4092'1951'9675ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("saddlp %0.4s, %1.8h")(arg);
  ASSERT_EQ(res, MakeUInt128(0xa274'ffff'fc58ULL, 0xb397'ffff'afc6ULL));
}

TEST(Arm64InsnTest, SignedAddAccumulateLongPairwiseInt8x16) {
  __uint128_t arg1 = MakeUInt128(0x1991'6463'8414'2707ULL, 0x7988'7088'7422'9277ULL);
  __uint128_t arg2 = MakeUInt128(0x7217'8260'3050'0994ULL, 0x5108'2478'3572'9056ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("sadalp %0.8h, %1.16b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x71c1'8327'2fe8'09c2ULL, 0x5109'2470'3608'905fULL));
}

TEST(Arm64InsnTest, SignedAddAccumulateLongPairwiseInt16x8) {
  __uint128_t arg1 = MakeUInt128(0x1991'6463'8414'2707ULL, 0x7988'7088'7422'9277ULL);
  __uint128_t arg2 = MakeUInt128(0x7217'8260'3050'0994ULL, 0x5108'2478'3572'9056ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("sadalp %0.4s, %1.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7218'0054'304f'b4afULL, 0x5109'0e88'3572'96efULL));
}

TEST(Arm64InsnTest, UnsignedAddLongPairwiseInt8x16) {
  __uint128_t arg = MakeUInt128(0x1483'2873'4808'9574ULL, 0x7777'5278'3442'2109ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("uaddlp %0.8h, %1.16b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x0097'009b'0050'0109ULL, 0x00ee'00ca'0076'002aULL));
}

TEST(Arm64InsnTest, UnsignedAddAccumulateLongPairwiseInt8x16) {
  __uint128_t arg1 = MakeUInt128(0x9348'1546'9163'1162ULL, 0x4928'8735'7471'8824ULL);
  __uint128_t arg2 = MakeUInt128(0x5207'6657'3882'5139ULL, 0x6391'6357'6723'1510ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W0_ARG("uadalp %0.8h, %1.16b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x52e2'66b2'3976'51acULL, 0x6402'6413'6808'15bcULL));
}

TEST(Arm64InsnTest, SignedAddLong) {
  __uint128_t arg1 = MakeUInt128(0x3478'0745'8506'7606ULL, 0x3048'2294'0965'3041ULL);
  __uint128_t arg2 = MakeUInt128(0x1183'0667'1081'8930ULL, 0x3110'8871'7281'6751ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("saddl %0.4s, %1.4h, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'9587'ffff'ff36ULL, 0x0000'45fb'0000'0dacULL));
}

TEST(Arm64InsnTest, SignedAddLongUpper) {
  __uint128_t arg1 = MakeUInt128(0x3160'6831'5867'9946ULL, 0x0165'2057'7405'2942ULL);
  __uint128_t arg2 = MakeUInt128(0x3053'6017'8031'3357ULL, 0x2632'6705'4790'3384ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("saddl2 %0.4s, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'bb95'0000'5cc6ULL, 0x0000'2797'0000'875cULL));
}

TEST(Arm64InsnTest, SignedSubLong) {
  __uint128_t arg1 = MakeUInt128(0x8566'7462'6087'9482ULL, 0x0186'4748'7672'7272ULL);
  __uint128_t arg2 = MakeUInt128(0x2206'2676'4653'3809ULL, 0x9801'9668'8368'0994ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ssubl %0.4s, %1.4h, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'1a34'ffff'5c79ULL, 0xffff'6360'0000'4decULL));
}

TEST(Arm64InsnTest, SignedSubLongUpper) {
  __uint128_t arg1 = MakeUInt128(0x3011'3317'5330'5329ULL, 0x8020'1668'8817'4813ULL);
  __uint128_t arg2 = MakeUInt128(0x4298'8681'5855'7781ULL, 0x0343'2317'5306'4784ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ssubl2 %0.4s, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xffff'3511'0000'008fULL, 0xffff'7cdd'ffff'f351ULL));
}

TEST(Arm64InsnTest, UnsignedAddLong) {
  __uint128_t arg1 = MakeUInt128(0x3126'0595'0577'7727ULL, 0x5424'7124'1648'3128ULL);
  __uint128_t arg2 = MakeUInt128(0x3298'2072'3617'5057ULL, 0x4673'8701'2820'9575ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uaddl %0.4s, %1.4h, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'3b8e'0000'c77eULL, 0x0000'63be'0000'2607ULL));
}

TEST(Arm64InsnTest, UnsignedAddLongUpper) {
  __uint128_t arg1 = MakeUInt128(0x3384'6984'9977'8726ULL, 0x7065'5519'1854'4686ULL);
  __uint128_t arg2 = MakeUInt128(0x9846'9478'4957'3462ULL, 0x2606'2942'1962'4557ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uaddl2 %0.4s, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'31b6'0000'8bddULL, 0x0000'966b'0000'7e5bULL));
}

TEST(Arm64InsnTest, UnsignedSubLong) {
  __uint128_t arg1 = MakeUInt128(0x4378'1119'8855'6318ULL, 0x7777'9253'7201'1667ULL);
  __uint128_t arg2 = MakeUInt128(0x1853'9541'8359'8443ULL, 0x8305'2037'6281'9440ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("usubl %0.4s, %1.4h, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'04fc'ffff'ded5ULL, 0x0000'2b25'ffff'7bd8ULL));
}

TEST(Arm64InsnTest, UnsignedSubLongUpper) {
  __uint128_t arg1 = MakeUInt128(0x5228'7174'4026'6638ULL, 0x9148'8171'7308'6436ULL);
  __uint128_t arg2 = MakeUInt128(0x1113'8906'9420'2790ULL, 0x8814'3119'4487'9941ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("usubl2 %0.4s, %1.8h, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0000'2e81'ffff'caf5ULL, 0x0000'0934'0000'5058ULL));
}

TEST(Arm64InsnTest, SignedAddWide8x8) {
  __uint128_t arg1 = MakeUInt128(0x7844'5981'8313'4112ULL, 0x9001'9992'0598'1352ULL);
  __uint128_t arg2 = MakeUInt128(0x2051'1733'6585'6407ULL, 0x8264'8494'2764'4113ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("saddw %0.8h, %1.8h, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x78a9'5906'8377'4119ULL, 0x9021'99e3'05af'1385ULL));
}

TEST(Arm64InsnTest, SignedAddWide16x4) {
  __uint128_t arg1 = MakeUInt128(0x7844'5981'8313'4112ULL, 0x9001'9992'0598'1352ULL);
  __uint128_t arg2 = MakeUInt128(0x2051'1733'6585'6407ULL, 0x8264'8494'2764'4113ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("saddw %0.4s, %1.4s, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7844'bf06'8313'a519ULL, 0x9001'b9e3'0598'2a85ULL));
}

TEST(Arm64InsnTest, SignedAddWide32x2) {
  __uint128_t arg1 = MakeUInt128(0x7844'5981'8313'4112ULL, 0x9001'9992'0598'1352ULL);
  __uint128_t arg2 = MakeUInt128(0x2051'1733'6585'6407ULL, 0x8264'8494'2764'4113ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("saddw %0.2d, %1.2d, %2.2s")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7844'5981'e898'a519ULL, 0x9001'9992'25e9'2a85ULL));
}

TEST(Arm64InsnTest, SignedAddWideUpper) {
  __uint128_t arg1 = MakeUInt128(0x3407'0922'3343'6577ULL, 0x9160'1280'9317'9401ULL);
  __uint128_t arg2 = MakeUInt128(0x7185'9859'9933'8492ULL, 0x3549'5640'0570'9955ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("saddw2 %0.4s, %1.4s, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x3407'0e92'3342'feccULL, 0x9160'47c9'9317'ea41ULL));
}

TEST(Arm64InsnTest, SignedSubWide) {
  __uint128_t arg1 = MakeUInt128(0x2302'8470'0731'2065ULL, 0x8032'6264'1711'6165ULL);
  __uint128_t arg2 = MakeUInt128(0x9576'1327'2351'5666ULL, 0x6253'6672'7189'9853ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ssubw %0.4s, %1.4s, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x2302'611f'0730'c9ffULL, 0x8032'ccee'1711'4e3eULL));
}

TEST(Arm64InsnTest, SignedSubWideUpper) {
  __uint128_t arg1 = MakeUInt128(0x4510'8247'8357'2905ULL, 0x6919'8855'5467'8860ULL);
  __uint128_t arg2 = MakeUInt128(0x7946'2805'3712'2704ULL, 0x2466'5431'9214'5281ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ssubw2 %0.4s, %1.4s, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x4510'f033'8356'd684ULL, 0x6919'63ef'5467'342fULL));
}

TEST(Arm64InsnTest, UnsignedAddWide8x8) {
  __uint128_t arg1 = MakeUInt128(0x5870'7859'5129'8344ULL, 0x1729'5351'9537'8855ULL);
  __uint128_t arg2 = MakeUInt128(0x3457'3742'6085'9029ULL, 0x0817'6515'5780'3905ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uaddw %0.8h, %1.8h, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x58d0'78de'51b9'836dULL, 0x175d'53a8'956e'8897ULL));
}

TEST(Arm64InsnTest, UnsignedAddWide16x4) {
  __uint128_t arg1 = MakeUInt128(0x5870'7859'5129'8344ULL, 0x1729'5351'9537'8855ULL);
  __uint128_t arg2 = MakeUInt128(0x3457'3742'6085'9029ULL, 0x0817'6515'5780'3905ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uaddw %0.4s, %1.4s, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x5870'd8de'512a'136dULL, 0x1729'87a8'9537'bf97ULL));
}

TEST(Arm64InsnTest, UnsignedAddWide32x2) {
  __uint128_t arg1 = MakeUInt128(0x5870'7859'5129'8344ULL, 0x1729'5351'9537'8855ULL);
  __uint128_t arg2 = MakeUInt128(0x3457'3742'6085'9029ULL, 0x0817'6515'5780'3905ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uaddw %0.2d, %1.2d, %2.2s")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x5870'7859'b1af'136dULL, 0x1729'5351'c98e'bf97ULL));
}

TEST(Arm64InsnTest, UnsignedAddWideUpper) {
  __uint128_t arg1 = MakeUInt128(0x7516'4932'7095'0493ULL, 0x4639'3824'3222'7188ULL);
  __uint128_t arg2 = MakeUInt128(0x5159'7405'4702'1482ULL, 0x8971'1177'7923'7612ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("uaddw2 %0.4s, %1.4s, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x7516'c255'7095'7aa5ULL, 0x4639'c195'3222'82ffULL));
}

TEST(Arm64InsnTest, UnsignedSubWide) {
  __uint128_t arg1 = MakeUInt128(0x0625'2479'7219'9786ULL, 0x6854'2798'9779'9233ULL);
  __uint128_t arg2 = MakeUInt128(0x9579'0575'8189'0622ULL, 0x5254'7358'2205'2364ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("usubw %0.4s, %1.4s, %2.4h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0624'a2f0'7219'9164ULL, 0x6853'921f'9779'8cbeULL));
}

TEST(Arm64InsnTest, UnsignedSubWideUpper) {
  __uint128_t arg1 = MakeUInt128(0x8242'3921'9269'5062ULL, 0x0831'8381'4546'9839ULL);
  __uint128_t arg2 = MakeUInt128(0x2366'4613'6398'9101ULL, 0x2102'1770'9597'6704ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("usubw2 %0.4s, %1.4s, %2.8h")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x8241'a38a'9268'e95eULL, 0x0831'627f'4546'80c9ULL));
}

TEST(Arm64InsnTest, SignedMultiplyLongInt8x8) {
  __uint128_t arg1 = MakeUInt128(0x9191'7915'5224'1718ULL, 0x9585'3616'8059'4741ULL);
  __uint128_t arg2 = MakeUInt128(0x2341'9339'8420'2187ULL, 0x4564'9256'4434'6239ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smull %0.8h, %1.8b, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xd848'0480'02f7'f4a8ULL, 0xf0d3'e3d1'cc7b'04adULL));
}

TEST(Arm64InsnTest, SignedMultiplyLongInt8x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x9314'0529'7634'7574ULL, 0x8119'3567'0911'0137ULL);
  __uint128_t arg2 = MakeUInt128(0x7517'2100'8031'5590ULL, 0x2485'3090'6692'0376ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smull2 %0.8h, %1.16b, %2.16b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0396'f8b2'0003'195aULL, 0xee24'f3fd'09f0'd2f0ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt8x8) {
  __uint128_t arg1 = MakeUInt128(0x9149'0556'2842'5039ULL, 0x1275'7710'2840'2799ULL);
  __uint128_t arg2 = MakeUInt128(0x8066'3658'2548'8926ULL, 0x4880'2545'6610'1729ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull %0.8h, %1.8b, %2.8b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x05c8'1290'2ad0'0876ULL, 0x4880'1d16'010e'1d90ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt8x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x9709'6834'0800'5355ULL, 0x9849'1754'1738'1883ULL);
  __uint128_t arg2 = MakeUInt128(0x9994'4697'4867'6265ULL, 0x5165'8276'5848'3588ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull2 %0.8h, %1.16b, %2.16b")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x07e8'0fc0'04f8'4598ULL, 0x3018'1ccd'0bae'26b8ULL));
}

TEST(Arm64InsnTest, SignedMultiplyLongInt8x8IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x9293'4595'8897'0695ULL, 0x3653'4940'6034'0216ULL);
  __uint128_t arg2 = MakeUInt128(0x6544'3755'8900'4563ULL, 0x2882'2505'4525'5640ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smull %0.4s, %1.4h, %2.h[2]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xe630'cb23'016c'3279ULL, 0xe859'3fcf'0f0a'1d79ULL));
}

TEST(Arm64InsnTest, SignedMultiplyLongInt8x8IndexedElemUpper) {
  __uint128_t arg1 = MakeUInt128(0x9279'0682'1207'3883ULL, 0x7781'4233'5628'2360ULL);
  __uint128_t arg2 = MakeUInt128(0x8963'2080'6822'2468ULL, 0x0122'4826'1177'1858ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smull2 %0.4s, %1.8h, %2.h[2]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0af0'1400'047d'b000ULL, 0x0f2b'e080'0867'7980ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt8x8IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x9086'9960'3302'7634ULL, 0x7870'8108'1754'5011ULL);
  __uint128_t arg2 = MakeUInt128(0x9307'1412'2339'0866ULL, 0x3938'3395'2942'5786ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull %0.4s, %1.4h, %2.h[2]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x03ff'be24'0944'5fa8ULL, 0x0b54'a16c'0c06'48c0ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt8x8IndexedElem2) {
  __uint128_t arg1 = MakeUInt128(0x9132'7104'9547'8599ULL, 0x1801'9696'7835'3214ULL);
  __uint128_t arg2 = MakeUInt128(0x6444'1189'2606'3152ULL, 0x6618'1674'4319'3550ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull %0.4s, %1.4h, %2.h[4]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1f16'5930'1bd2'6cd0ULL, 0x1e3c'b9a0'1789'2540ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt8x8IndexedElemUpper) {
  __uint128_t arg1 = MakeUInt128(0x9815'7936'7897'6697ULL, 0x4220'5750'5968'3440ULL);
  __uint128_t arg2 = MakeUInt128(0x8697'3502'0141'0206ULL, 0x7235'8502'0072'4522ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull2 %0.4s, %1.8h, %2.h[2]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1283'3ad0'0ad1'a880ULL, 0x0db1'2440'1214'3ea0ULL));
}

TEST(Arm64InsnTest, SignedMultiplyLongInt16x4IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x9293'4595'8897'0695ULL, 0x3653'4940'6034'0216ULL);
  __uint128_t arg2 = MakeUInt128(0x6544'3755'8900'4563ULL, 0x2882'2505'4525'5640ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smull %0.4s, %1.4h, %2.h[0]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0xdfa2'8565'01c8'b49fULL, 0xe257'4dd9'12dc'119fULL));
}

TEST(Arm64InsnTest, SignedMultiplyLongInt16x4IndexedElemUpper) {
  __uint128_t arg1 = MakeUInt128(0x9279'0682'1207'3883ULL, 0x7781'4233'5628'2360ULL);
  __uint128_t arg2 = MakeUInt128(0x8963'2080'6822'2468ULL, 0x0122'4826'1177'1858ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smull2 %0.4s, %1.8h, %2.h[0]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0c40'a040'0507'df00ULL, 0x10fe'b068'096a'10b8ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt16x4IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x9086'9960'3302'7634ULL, 0x7870'8108'1754'5011ULL);
  __uint128_t arg2 = MakeUInt128(0x9307'1412'2339'0866ULL, 0x3938'3395'2942'5786ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull %0.4s, %1.4h, %2.h[0]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x01ac'62cc'03e0'b8b8ULL, 0x04bd'c564'0508'1c40ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt16x4IndexedElemUpper) {
  __uint128_t arg1 = MakeUInt128(0x9815'7936'7897'6697ULL, 0x4220'5750'5968'3440ULL);
  __uint128_t arg2 = MakeUInt128(0x8697'3502'0141'0206ULL, 0x7235'8502'0072'4522ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull2 %0.4s, %1.8h, %2.h[2]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x1283'3ad0'0ad1'a880ULL, 0x0db1'2440'1214'3ea0ULL));
}

TEST(Arm64InsnTest, SignedMultiplyLongInt32x2IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x8293'4595'4897'0695ULL, 0x3653'4940'6034'0216ULL);
  __uint128_t arg2 = MakeUInt128(0x6544'3755'0900'4563ULL, 0x2882'2505'4525'5640ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smull %0.2d, %1.2s, %2.s[0]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x028d'62e8'042d'b49fULL, 0xfb97'0b73'6db5'119fULL));
}

TEST(Arm64InsnTest, SignedMultiplyLongInt32x2IndexedElemUpper) {
  __uint128_t arg1 = MakeUInt128(0x9279'0682'1207'3883ULL, 0x7781'4233'5628'2360ULL);
  __uint128_t arg2 = MakeUInt128(0x8963'2080'6822'2468ULL, 0x0122'4826'1177'1858ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("smull2 %0.2d, %1.4s, %2.s[0]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x230b'cbf4'5807'df00ULL, 0x309c'730e'3c98'10b8ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt32x2IndexedElem) {
  __uint128_t arg1 = MakeUInt128(0x9086'9960'3302'7634ULL, 0x7870'8108'1754'5011ULL);
  __uint128_t arg2 = MakeUInt128(0x9307'1412'2339'0866ULL, 0x3938'3395'2942'5786ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull %0.2d, %1.2s, %2.s[0]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0704'b361'd440'b8b8ULL, 0x13e2'99ae'10cc'1c40ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyLongInt32x2IndexedElemUpper) {
  __uint128_t arg1 = MakeUInt128(0x9815'7936'7897'6697ULL, 0x4220'5750'5968'3440ULL);
  __uint128_t arg2 = MakeUInt128(0x8697'3502'0141'0206ULL, 0x7235'8502'0072'4522ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("umull2 %0.2d, %1.4s, %2.s[0]")(arg1, arg2);
  ASSERT_EQ(res, MakeUInt128(0x0070'1c5e'6d19'b980ULL, 0x0052'eb13'48c0'abe0ULL));
}

TEST(Arm64InsnTest, SignedMultiplyAddLongInt8x8) {
  __uint128_t arg1 = MakeUInt128(0x9779'9400'1260'1642ULL, 0x2760'9260'8234'9304ULL);
  __uint128_t arg2 = MakeUInt128(0x1180'6438'2913'8347ULL, 0x3546'7972'5399'2623ULL);
  __uint128_t arg3 = MakeUInt128(0x3879'1582'9984'8645ULL, 0x9271'7340'5922'5620ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlal %0.8h, %1.8b, %2.8b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x3b5b'1ca2'8ec6'9893ULL, 0x8b78'36c0'2ef2'5620ULL));
}

TEST(Arm64InsnTest, SignedMultiplyAddLongInt8x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x5514'4350'2182'8702ULL, 0x6685'6106'6500'3531ULL);
  __uint128_t arg2 = MakeUInt128(0x0502'1631'8206'0176ULL, 0x0921'7984'6849'3686ULL);
  __uint128_t arg3 = MakeUInt128(0x3161'2937'2795'1873ULL, 0x0789'7263'7353'7171ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlal2 %0.8h, %1.16b, %2.16b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x5a69'2937'32c3'0119ULL, 0x0b1f'6288'a12c'6e89ULL));
}

TEST(Arm64InsnTest, SignedMultiplySubtractLongInt8x8) {
  __uint128_t arg1 = MakeUInt128(0x9662'5393'3953'8092ULL, 0x2195'5919'1818'8552ULL);
  __uint128_t arg2 = MakeUInt128(0x6780'6214'9923'1727ULL, 0x6316'3218'3398'9693ULL);
  __uint128_t arg3 = MakeUInt128(0x8075'6168'5591'1752ULL, 0x9984'5013'2067'1293ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlsl %0.8h, %1.8b, %2.8b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x9764'560f'6111'2814ULL, 0xc42a'8113'00a1'1b17ULL));
}

TEST(Arm64InsnTest, SignedMultiplySubtractLongInt8x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x9826'9030'8911'1856ULL, 0x8798'6929'4705'1352ULL);
  __uint128_t arg2 = MakeUInt128(0x4816'0917'4324'3015ULL, 0x3836'8470'7292'8989ULL);
  __uint128_t arg3 = MakeUInt128(0x8284'6022'2373'0145ULL, 0x2655'6798'9862'7767ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlsl2 %0.8h, %1.16b, %2.16b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x62e6'6248'2c48'2763ULL, 0x40cd'7d88'cb3e'6577ULL));
}

TEST(Arm64InsnTest, SignedMultiplyAddLongInt16x4) {
  __uint128_t arg1 = MakeUInt128(0x9779'9400'1260'1642ULL, 0x2760'9260'8234'9304ULL);
  __uint128_t arg2 = MakeUInt128(0x1180'6438'2913'8347ULL, 0x3546'7972'5399'2623ULL);
  __uint128_t arg3 = MakeUInt128(0x3879'1582'9984'8645ULL, 0x9271'7340'5922'5620ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("smlal %0.4s, %1.4h, %2.4h")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x3b6b'd2a2'8eac'7893ULL, 0x8b4c'38c0'2eda'b620ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyAddLongInt8x8) {
  __uint128_t arg1 = MakeUInt128(0x9696'9202'5388'6503ULL, 0x4577'1831'7668'6885ULL);
  __uint128_t arg2 = MakeUInt128(0x9236'8148'8475'2764ULL, 0x9846'8821'9497'3972ULL);
  __uint128_t arg3 = MakeUInt128(0x9707'7371'8718'8400ULL, 0x4143'2312'7636'5048ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlal %0.8h, %1.8b, %2.8b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0xc1d3'b199'967b'852cULL, 0x96cf'42b6'bfc8'50d8ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplyAddLongInt8x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x9055'6376'9525'2326ULL, 0x5361'4424'7802'3082ULL);
  __uint128_t arg2 = MakeUInt128(0x6811'8310'3773'5887ULL, 0x0892'4061'3031'3364ULL);
  __uint128_t arg3 = MakeUInt128(0x7737'1011'6282'1461ULL, 0x4661'6794'0409'0518ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlal2 %0.8h, %1.16b, %2.16b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x8db7'1073'6c12'4729ULL, 0x48f9'9ee6'1509'12bcULL));
}

TEST(Arm64InsnTest, UnsignedMultiplySubtractLongInt8x8) {
  __uint128_t arg1 = MakeUInt128(0x4577'7724'5752'0386ULL, 0x5437'5428'2825'6714ULL);
  __uint128_t arg2 = MakeUInt128(0x1288'5834'5444'3513ULL, 0x2562'0544'6424'1011ULL);
  __uint128_t arg3 = MakeUInt128(0x0379'5546'4190'5811ULL, 0x6862'3059'6447'6958ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlsl %0.8h, %1.8b, %2.8b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0xe6ed'3f7e'40f1'4e1fULL, 0x6388'f121'3b5f'6208ULL));
}

TEST(Arm64InsnTest, UnsignedMultiplySubtractLongInt8x8Upper) {
  __uint128_t arg1 = MakeUInt128(0x4739'3765'6433'6319ULL, 0x7978'6803'6718'7307ULL);
  __uint128_t arg2 = MakeUInt128(0x9693'9242'3632'1448ULL, 0x4503'5477'6315'6702ULL);
  __uint128_t arg3 = MakeUInt128(0x5539'0065'4231'1792ULL, 0x0153'4649'7792'9066ULL);
  __uint128_t res =
      ASM_INSN_WRAP_FUNC_W_RES_WW0_ARG("umlsl2 %0.8h, %1.16b, %2.16b")(arg1, arg2, arg3);
  ASSERT_EQ(res, MakeUInt128(0x2d64'fe6d'13ec'1784ULL, 0xe0b6'44e1'5572'8f01ULL));
}

TEST(Arm64InsnTest, SignedShiftLeftInt64x1) {
  constexpr auto AsmSshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sshl %d0, %d1, %d2");
  __uint128_t arg = MakeUInt128(0x9007'4972'9736'3549ULL, 0x6453'3288'8698'4406ULL);
  ASSERT_EQ(AsmSshl(arg, -65), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSshl(arg, -64), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSshl(arg, -63), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSshl(arg, -1), MakeUInt128(0xc803'a4b9'4b9b'1aa4ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSshl(arg, 0), MakeUInt128(0x9007'4972'9736'3549ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSshl(arg, 1), MakeUInt128(0x200e'92e5'2e6c'6a92ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSshl(arg, 63), MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSshl(arg, 64), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSshl(arg, 65), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, SignedRoundingShiftLeftInt64x1) {
  constexpr auto AsmSrshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("srshl %d0, %d1, %d2");
  __uint128_t arg = MakeUInt128(0x9276'4579'3106'5792ULL, 0x2955'2498'8727'5846ULL);
  ASSERT_EQ(AsmSrshl(arg, -65), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg, -64), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg, -63), MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg, -1), MakeUInt128(0xc93b'22bc'9883'2bc9ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg, 0), MakeUInt128(0x9276'4579'3106'5792ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg, 1), MakeUInt128(0x24ec'8af2'620c'af24ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg, 63), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg, 64), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg, 65), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, UnsignedShiftLeftInt64x1) {
  constexpr auto AsmUshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ushl %d0, %d1, %d2");
  __uint128_t arg = MakeUInt128(0x9138'2966'8246'8185ULL, 0x7103'1887'9065'2870ULL);
  ASSERT_EQ(AsmUshl(arg, -65), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg, -64), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg, -63), MakeUInt128(0x0000'0000'0000'0001ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg, -1), MakeUInt128(0x489c'14b3'4123'40c2ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg, 0), MakeUInt128(0x9138'2966'8246'8185ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg, 1), MakeUInt128(0x2270'52cd'048d'030aULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg, 63), MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg, 64), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg, 65), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, UnsignedRoundingShiftLeftInt64x1) {
  constexpr auto AsmUrshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("urshl %d0, %d1, %d2");
  __uint128_t arg = MakeUInt128(0x9023'4529'2440'7736ULL, 0x5949'5630'5100'7421ULL);
  ASSERT_EQ(AsmUrshl(arg, -65), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUrshl(arg, -64), MakeUInt128(0x0000'0000'0000'0001ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUrshl(arg, -63), MakeUInt128(0x0000'0000'0000'0001ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUrshl(arg, -1), MakeUInt128(0x4811'a294'9220'3b9bULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUrshl(arg, 0), MakeUInt128(0x9023'4529'2440'7736ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUrshl(arg, 1), MakeUInt128(0x2046'8a52'4880'ee6cULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUrshl(arg, 63), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUrshl(arg, 64), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_EQ(AsmUrshl(arg, 65), MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
}

TEST(Arm64InsnTest, UnsignedRoundingShiftLeftInt64x2) {
  constexpr auto AsmSrshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("urshl %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x8765'4321'8765'432fULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0004ULL, 0xffff'ffff'ffff'fffcULL);
  ASSERT_EQ(AsmSrshl(arg1, arg2), MakeUInt128(0x2345'6781'2345'6780ULL, 0x0876'5432'1876'5433ULL));
}

TEST(Arm64InsnTest, UnsignedShiftLeftInt64x2) {
  constexpr auto AsmUshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ushl %0.2d, %1.2d, %2.2d");
  __uint128_t arg1 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x8765'4321'8765'432fULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0004ULL, 0xffff'ffff'ffff'fffcULL);
  ASSERT_EQ(AsmUshl(arg1, arg2), MakeUInt128(0x2345'6781'2345'6780ULL, 0x0876'5432'1876'5432ULL));
}

TEST(Arm64InsnTest, SignedShiftLeftInt16x8) {
  constexpr auto AsmSshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sshl %0.8h, %1.8h, %2.8h");
  __uint128_t arg1 = MakeUInt128(0x9999'9999'9999'9999ULL, 0x9999'9999'9999'9999ULL);
  __uint128_t arg2 = MakeUInt128(0x0010'000f'0002'0001ULL, 0xffff'fff1'fff0'ffefULL);
  ASSERT_EQ(AsmSshl(arg1, arg2), MakeUInt128(0x0000'8000'6664'3332ULL, 0xcccc'ffff'ffff'ffffULL));
  ASSERT_EQ(AsmSshl(arg1, 0), MakeUInt128(0x9999'9999'9999'9999ULL, 0x9999'9999'9999'9999ULL));
}

TEST(Arm64InsnTest, SignedRoundingShiftLeftInt16x8) {
  constexpr auto AsmSrshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("srshl %0.8h, %1.8h, %2.8h");
  __uint128_t arg1 = MakeUInt128(0x9999'9999'9999'9999ULL, 0x9999'9999'9999'9999ULL);
  __uint128_t arg2 = MakeUInt128(0x0010'000f'0002'0001ULL, 0xffff'fff1'fff0'ffefULL);
  ASSERT_EQ(AsmSrshl(arg1, arg2), MakeUInt128(0x0000'8000'6664'3332ULL, 0xcccd'ffff'0000'0000ULL));
  ASSERT_EQ(AsmSrshl(arg1, 0), MakeUInt128(0x9999'9999'9999'9999ULL, 0x9999'9999'9999'9999ULL));
}

TEST(Arm64InsnTest, UnsignedShiftLeftInt16x8) {
  constexpr auto AsmUshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ushl %0.8h, %1.8h, %2.8h");
  __uint128_t arg1 = MakeUInt128(0x9999'9999'9999'9999ULL, 0x9999'9999'9999'9999ULL);
  __uint128_t arg2 = MakeUInt128(0x0010'000f'0002'0001ULL, 0xffff'fff1'fff0'ffefULL);
  ASSERT_EQ(AsmUshl(arg1, arg2), MakeUInt128(0x0000'8000'6664'3332ULL, 0x4ccc'0001'0000'0000ULL));
  ASSERT_EQ(AsmUshl(arg1, 0), MakeUInt128(0x9999'9999'9999'9999ULL, 0x9999'9999'9999'9999ULL));
}

TEST(Arm64InsnTest, UnsignedRoundingShiftLeftInt16x8) {
  constexpr auto AsmUrshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("urshl %0.8h, %1.8h, %2.8h");
  __uint128_t arg1 = MakeUInt128(0x9999'9999'9999'9999ULL, 0x9999'9999'9999'9999ULL);
  __uint128_t arg2 = MakeUInt128(0x0010'000f'0002'0001ULL, 0xffff'fff1'fff0'ffefULL);
  ASSERT_EQ(AsmUrshl(arg1, arg2), MakeUInt128(0x0000'8000'6664'3332ULL, 0x4ccd'0001'0001'0000ULL));
  ASSERT_EQ(AsmUrshl(arg1, 0), MakeUInt128(0x9999'9999'9999'9999ULL, 0x9999'9999'9999'9999ULL));
}

TEST(Arm64InsnTest, SignedRoundingShiftLeftInt32x4) {
  constexpr auto AsmSrshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("srshl %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x8765'4321'8765'4321ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'ffff'fffcULL, 0xffff'fffc'ffff'ffe9ULL);
  ASSERT_EQ(AsmSrshl(arg1, arg2), MakeUInt128(0x2345'6780'0123'4568ULL, 0xf876'5432'ffff'ff0fULL));
}

TEST(Arm64InsnTest, SignedShiftLeftInt32x4) {
  constexpr auto AsmSshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("sshl %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x8765'4321'8765'4321ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'ffff'fffcULL, 0xffff'fffc'ffff'ffe9ULL);
  ASSERT_EQ(AsmSshl(arg1, arg2), MakeUInt128(0x2345'6780'0123'4567ULL, 0xf876'5432'ffff'ff0eULL));
}

TEST(Arm64InsnTest, UnsignedRoundingShiftLeftInt32x4) {
  constexpr auto AsmSrshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("urshl %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x8765'4321'8765'4321ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'ffff'fffcULL, 0xffff'fffc'ffff'ffe9ULL);
  ASSERT_EQ(AsmSrshl(arg1, arg2), MakeUInt128(0x2345'6780'0123'4568ULL, 0x0876'5432'0000'010fULL));
}

TEST(Arm64InsnTest, UnsignedShiftLeftInt32x4) {
  constexpr auto AsmUshl = ASM_INSN_WRAP_FUNC_W_RES_WW_ARG("ushl %0.4s, %1.4s, %2.4s");
  __uint128_t arg1 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x8765'4321'8765'4321ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'ffff'fffcULL, 0xffff'fffc'ffff'ffe9ULL);
  ASSERT_EQ(AsmUshl(arg1, arg2), MakeUInt128(0x2345'6780'0123'4567ULL, 0x0876'5432'0000'010eULL));
}

TEST(Arm64InsnTest, UnsignedReciprocalSquareRootEstimateInt32x4) {
  __uint128_t arg = MakeUInt128(0x9641'1228'2140'7533ULL, 0x0265'5100'4241'0489ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("ursqrte %0.4s, %1.4s")(arg);
  ASSERT_EQ(res, MakeUInt128(0xa700'0000'ffff'ffffULL, 0xffff'ffff'fb80'0000ULL));
}

TEST(Arm64InsnTest, UnsignedReciprocalEstimateInt32x4) {
  __uint128_t arg = MakeUInt128(0x9714'8648'9946'8611ULL, 0x2476'0542'8673'4367ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("urecpe %0.4s, %1.4s")(arg);
  ASSERT_EQ(res, MakeUInt128(0xd880'0000'd600'0000ULL, 0xffff'ffff'f400'0000ULL));
}

bool IsQcBitSet(uint32_t fpsr) {
  return (fpsr & kFpsrQcBit) != 0;
}

TEST(Arm64InsnTest, SignedSaturatingAddInt64x1) {
  constexpr auto AsmSqadd = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqadd %d0, %d2, %d3");

  __uint128_t arg1 = MakeUInt128(0x4342'5277'5311'9724ULL, 0x7430'8730'4361'9511ULL);
  __uint128_t arg2 = MakeUInt128(0x3961'1908'0030'2558ULL, 0x7838'7644'2060'8504ULL);
  auto [res1, fpsr1] = AsmSqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x7ca3'6b7f'5341'bc7cULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x2557'1853'0891'9284ULL, 0x4038'0507'1030'0647ULL);
  __uint128_t arg4 = MakeUInt128(0x7684'7863'2431'9100ULL, 0x0223'9297'8525'5372ULL);
  auto [res2, fpsr2] = AsmSqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingAddInt32x4) {
  constexpr auto AsmSqadd = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqadd %0.4s, %2.4s, %3.4s");

  __uint128_t arg1 = MakeUInt128(0x9883'5544'4560'2495ULL, 0x5666'8436'6029'2219ULL);
  __uint128_t arg2 = MakeUInt128(0x5124'8309'1060'5377ULL, 0x2019'8021'8310'1032ULL);
  auto [res1, fpsr1] = AsmSqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0xe9a7'd84d'55c0'780cULL, 0x7680'0457'e339'324bULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x9713'3088'4461'7410ULL, 0x7959'1625'1171'4864ULL);
  __uint128_t arg4 = MakeUInt128(0x8744'6861'1247'6054ULL, 0x2867'3436'7090'4667ULL);
  auto [res2, fpsr2] = AsmSqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x8000'0000'56a8'd464ULL, 0x7fff'ffff'7fff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingAddInt8x1) {
  constexpr auto AsmUqadd = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqadd %b0, %b2, %b3");

  __uint128_t arg1 = MakeUInt128(0x6017'1742'2996'0273ULL, 0x5310'2768'7194'4944ULL);
  __uint128_t arg2 = MakeUInt128(0x4917'9397'8514'4631ULL, 0x5973'1443'5351'8504ULL);
  auto [res1, fpsr1] = AsmUqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'00a4ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x3306'2636'9562'6490ULL, 0x9108'2762'7115'9038ULL);
  __uint128_t arg4 = MakeUInt128(0x5699'5051'2465'2999ULL, 0x6062'8554'4383'8330ULL);
  auto [res2, fpsr2] = AsmUqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'00ffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingAddInt64x1) {
  constexpr auto AsmUqadd = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqadd %d0, %d2, %d3");

  __uint128_t arg1 = MakeUInt128(0x0606'8851'3723'4627ULL, 0x0799'7327'2331'3469ULL);
  __uint128_t arg2 = MakeUInt128(0x3971'4562'8554'2615ULL, 0x4676'5063'2465'6766ULL);
  auto [res1, fpsr1] = AsmUqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x3f77'cdb3'bc77'6c3cULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x9534'9570'1860'0154ULL, 0x1262'3962'2864'1389ULL);
  __uint128_t arg4 = MakeUInt128(0x7796'7333'2907'0567ULL, 0x3769'6215'6498'1845ULL);
  auto [res2, fpsr2] = AsmUqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingAddInt32x4) {
  constexpr auto AsmUqadd = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqadd %0.4s, %2.4s, %3.4s");

  __uint128_t arg1 = MakeUInt128(0x9737'4257'0073'5921ULL, 0x0031'5415'0893'6793ULL);
  __uint128_t arg2 = MakeUInt128(0x0081'6998'0536'5202ULL, 0x7600'7277'4967'4584ULL);
  auto [res1, fpsr1] = AsmUqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x97b8'abef'05a9'ab23ULL, 0x7631'c68c'51fa'ad17ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x9727'8564'7198'3963ULL, 0x0878'1543'2211'6691ULL);
  __uint128_t arg4 = MakeUInt128(0x8654'5222'6812'6887ULL, 0x2684'4596'8442'4161ULL);
  auto [res2, fpsr2] = AsmUqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0xffff'ffff'd9aa'a1eaULL, 0x2efc'5ad9'a653'a7f2ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingSubtractInt32x1) {
  constexpr auto AsmSqsub = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqsub %s0, %s2, %s3");

  __uint128_t arg1 = MakeUInt128(0x3178'5348'7076'0322ULL, 0x1982'9705'7975'1191ULL);
  __uint128_t arg2 = MakeUInt128(0x4405'1099'4235'8830ULL, 0x3454'6353'4923'4982ULL);
  auto [res1, fpsr1] = AsmSqsub(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x2e40'7af2ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x1423'6964'8308'6410ULL, 0x2592'8874'5799'9322ULL);
  __uint128_t arg4 = MakeUInt128(0x3749'5519'1221'9519ULL, 0x0342'4452'3075'3513ULL);
  auto [res2, fpsr2] = AsmSqsub(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x8000'0000ULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  __uint128_t arg5 = MakeUInt128(0x3083'5088'7958'4152ULL, 0x1489'9127'6106'5137ULL);
  __uint128_t arg6 = MakeUInt128(0x4153'9435'8072'1139ULL, 0x0328'5749'1876'9094ULL);
  auto [res3, fpsr3] = AsmSqsub(arg5, arg6);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingSubtractInt64x1) {
  constexpr auto AsmSqsub = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqsub %d0, %d2, %d3");

  __uint128_t arg1 = MakeUInt128(0x4416'1252'2319'6943ULL, 0x4712'0641'7375'4912ULL);
  __uint128_t arg2 = MakeUInt128(0x1635'7008'5736'9439ULL, 0x7305'9797'0971'9726ULL);
  auto [res1, fpsr1] = AsmSqsub(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x2de0'a249'cbe2'd50aULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x7862'7664'9024'2516ULL, 0x1990'2774'7109'0335ULL);
  __uint128_t arg4 = MakeUInt128(0x9333'0930'4948'3805ULL, 0x9785'6628'8447'8744ULL);
  auto [res2, fpsr2] = AsmSqsub(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingSubtractInt32x4) {
  constexpr auto AsmSqsub = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqsub %0.4s, %2.4s, %3.4s");

  __uint128_t arg1 = MakeUInt128(0x4485'6809'7756'9630ULL, 0x3129'5887'1916'1129ULL);
  __uint128_t arg2 = MakeUInt128(0x2946'8188'4936'3386ULL, 0x4739'2747'6012'2696ULL);
  auto [res1, fpsr1] = AsmSqsub(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x1b3e'e681'2e20'62aaULL, 0xe9f0'3140'b903'ea93ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x9304'1271'0072'7784ULL, 0x9301'5550'3889'5360ULL);
  __uint128_t arg4 = MakeUInt128(0x3382'6192'9343'7970ULL, 0x8187'4320'9499'1415ULL);
  auto [res2, fpsr2] = AsmSqsub(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x8000'0000'6d2e'fe14ULL, 0x117a'1230'7fff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingSubtractInt32x1) {
  constexpr auto AsmUqsub = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqsub %s0, %s2, %s3");

  __uint128_t arg1 = MakeUInt128(0x2548'1560'9137'2812ULL, 0x8406'3330'3937'3562ULL);
  __uint128_t arg2 = MakeUInt128(0x4200'1604'5664'5574ULL, 0x1458'8166'0521'6660ULL);
  auto [res1, fpsr1] = AsmUqsub(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x3ad2'd29eULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x1259'9602'8183'9309ULL, 0x5487'0905'9073'8613ULL);
  __uint128_t arg4 = MakeUInt128(0x5191'4591'8195'1029ULL, 0x7327'8755'7104'9729ULL);
  auto [res2, fpsr2] = AsmUqsub(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0U, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingSubtractInt64x1) {
  constexpr auto AsmUqsub = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqsub %d0, %d2, %d3");

  __uint128_t arg1 = MakeUInt128(0x9691'0775'4257'6474ULL, 0x8832'5341'4121'3280ULL);
  __uint128_t arg2 = MakeUInt128(0x0626'7170'9400'9098ULL, 0x2235'2965'7957'9978ULL);
  auto [res1, fpsr1] = AsmUqsub(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x906a'9604'ae56'd3dcULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x7752'9291'0692'5043ULL, 0x2614'4695'0109'8610ULL);
  __uint128_t arg4 = MakeUInt128(0x8889'9914'6585'5188ULL, 0x1873'5825'2816'4302ULL);
  auto [res2, fpsr2] = AsmUqsub(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingSubtractInt32x4) {
  constexpr auto AsmUqsub = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqsub %0.4s, %2.4s, %3.4s");

  __uint128_t arg1 = MakeUInt128(0x6884'9625'7866'5885ULL, 0x9991'7986'7520'5545ULL);
  __uint128_t arg2 = MakeUInt128(0x5809'9004'5564'6117ULL, 0x8755'2493'7012'4553ULL);
  auto [res1, fpsr1] = AsmUqsub(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x107b'0621'2301'f76eULL, 0x123c'54f3'050e'0ff2ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x5032'6783'4058'6301ULL, 0x9301'9324'2996'3972ULL);
  __uint128_t arg4 = MakeUInt128(0x0444'5179'2881'2285ULL, 0x4478'2119'5353'0898ULL);
  auto [res2, fpsr2] = AsmUqsub(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x4bee'160a'17d7'407cULL, 0x4e89'720b'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingAbsoluteInt8x1) {
  constexpr auto AsmSqabs = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqabs %b0, %b2");

  __uint128_t arg1 = MakeUInt128(0x8918'0168'5572'7981ULL, 0x5642'1858'1911'9749ULL);
  auto [res1, fpsr1] = AsmSqabs(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'007fULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0080ULL, 0x6464'6072'8757'4305ULL);
  auto [res2, fpsr2] = AsmSqabs(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'007fULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingAbsoluteInt64x1) {
  constexpr auto AsmSqabs = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqabs %d0, %d2");

  __uint128_t arg1 = MakeUInt128(0x9717'3172'8131'5179ULL, 0x3290'4431'1218'1587ULL);
  auto [res1, fpsr1] = AsmSqabs(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x68e8'ce8d'7ece'ae87ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x1001'2376'8721'9447ULL);
  auto [res2, fpsr2] = AsmSqabs(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingAbsoluteInt32x4) {
  constexpr auto AsmSqabs = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqabs %0.4s, %2.4s");

  __uint128_t arg1 = MakeUInt128(0x9133'8205'7849'2800ULL, 0x6982'5519'5740'2018ULL);
  auto [res1, fpsr1] = AsmSqabs(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x6ecc'7dfb'7849'2800ULL, 0x6982'5519'5740'2018ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x1810'5641'2972'5083ULL, 0x6070'3568'8000'0000ULL);
  auto [res2, fpsr2] = AsmSqabs(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x1810'5641'2972'5083ULL, 0x6070'3568'7fff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingNegateInt32x1) {
  constexpr auto AsmSqneg = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqneg %s0, %s2");

  __uint128_t arg1 = MakeUInt128(0x6461'5826'9456'3802ULL, 0x3950'2837'1216'8644ULL);
  auto [res1, fpsr1] = AsmSqneg(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'6ba9'c7feULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x6561'7852'8000'0000ULL, 0x1277'1282'6918'6886ULL);
  auto [res2, fpsr2] = AsmSqneg(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingNegateInt64x1) {
  constexpr auto AsmSqneg = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqneg %d0, %d2");

  __uint128_t arg1 = MakeUInt128(0x9703'6007'9569'8276ULL, 0x2639'2344'1071'4658ULL);
  auto [res1, fpsr1] = AsmSqneg(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x68fc'9ff8'6a96'7d8aULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x4052'2953'6937'4997ULL);
  auto [res2, fpsr2] = AsmSqneg(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingNegateInt32x4) {
  constexpr auto AsmSqneg = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqneg %0.4s, %2.4s");

  __uint128_t arg1 = MakeUInt128(0x9172'3202'0282'2291ULL, 0x4886'9593'9972'9974ULL);
  auto [res1, fpsr1] = AsmSqneg(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x6e8d'cdfe'fd7d'dd6fULL, 0xb779'6a6d'668d'668cULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x2974'7115'5371'8589ULL, 0x2423'8493'8000'0000ULL);
  auto [res2, fpsr2] = AsmSqneg(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xd68b'8eeb'ac8e'7a77ULL, 0xdbdc'7b6d'7fff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftLeftImmInt32x1) {
  constexpr auto AsmSqshl = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshl %s0, %s2, #20");

  __uint128_t arg1 = MakeUInt128(0x9724'6116'0000'0181ULL, 0x0003'5098'9286'4120ULL);
  auto [res1, fpsr1] = AsmSqshl(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'1810'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x4195'1635'5110'8763ULL, 0x2042'6761'2979'8265ULL);
  auto [res2, fpsr2] = AsmSqshl(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftLeftImmInt64x1) {
  constexpr auto AsmSqshl = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshl %d0, %d2, #28");

  __uint128_t arg1 = MakeUInt128(0x0000'0007'7400'0539ULL, 0x2622'7603'2365'9751ULL);
  auto [res1, fpsr1] = AsmSqshl(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x7740'0053'9000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x9938'7149'9544'9137ULL, 0x3020'5184'3669'0767ULL);
  auto [res2, fpsr2] = AsmSqshl(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftLeftImmInt32x4) {
  constexpr auto AsmSqshl = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshl %0.4s, %2.4s, #12");

  __uint128_t arg1 = MakeUInt128(0x0007'2568'0004'2011ULL, 0x0000'3135'0003'3555ULL);
  auto [res1, fpsr1] = AsmSqshl(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x7256'8000'4201'1000ULL, 0x0313'5000'3355'5000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0944'0319'0007'2034ULL, 0x8651'0105'6104'9872ULL);
  auto [res2, fpsr2] = AsmSqshl(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'7203'4000ULL, 0x8000'0000'7fff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftLeftByRegisterImmInt32x1) {
  constexpr auto AsmSqshl = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqshl %s0, %s2, %s3");

  __uint128_t res;
  uint32_t fpsr;
  __uint128_t arg1 = MakeUInt128(0x7480'7718'1155'5330ULL, 0x9098'8702'5505'2076ULL);

  std::tie(res, fpsr) = AsmSqshl(arg1, -33);
  ASSERT_EQ(res, MakeUInt128(0U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqshl(arg1, -32);
  ASSERT_EQ(res, MakeUInt128(0U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqshl(arg1, -31);
  ASSERT_EQ(res, MakeUInt128(0U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqshl(arg1, -1);
  ASSERT_EQ(res, MakeUInt128(0x08aa'a998ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqshl(arg1, 0);
  ASSERT_EQ(res, MakeUInt128(0x1155'5330ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqshl(arg1, 1);
  ASSERT_EQ(res, MakeUInt128(0x22aa'a660ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqshl(arg1, 31);
  ASSERT_EQ(res, MakeUInt128(0x7fff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqshl(arg1, 32);
  ASSERT_EQ(res, MakeUInt128(0x7fff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqshl(arg1, 33);
  ASSERT_EQ(res, MakeUInt128(0x7fff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr));
}

TEST(Arm64InsnTest, UnsignedSaturatingShiftLeftImmInt64x1) {
  constexpr auto AsmUqshl = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("uqshl %d0, %d2, #28");

  __uint128_t arg1 = MakeUInt128(0x0000'0009'6157'3564ULL, 0x8883'4431'8528'0853ULL);
  auto [res1, fpsr1] = AsmUqshl(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x9615'7356'4000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x9759'2773'4433'6553ULL, 0x8418'8340'3035'1782ULL);
  auto [res2, fpsr2] = AsmUqshl(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingShiftLeftImmInt32x4) {
  constexpr auto AsmUqshl = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("uqshl %0.4s, %2.4s, #12");

  __uint128_t arg1 = MakeUInt128(0x0000'3263'0009'6218ULL, 0x0004'5659'0006'6853ULL);
  auto [res1, fpsr1] = AsmUqshl(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0326'3000'9621'8000ULL, 0x4565'9000'6685'3000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0009'9113'1401'0804ULL, 0x0009'7323'3544'9090ULL);
  auto [res2, fpsr2] = AsmUqshl(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x9911'3000'ffff'ffffULL, 0x9732'3000'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingShiftLeftByRegisterImmInt32x1) {
  constexpr auto AsmUqshl = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqshl %s0, %s2, %s3");

  __uint128_t res;
  uint32_t fpsr;
  __uint128_t arg1 = MakeUInt128(0x9714'9785'0741'4585ULL, 0x3085'7813'3915'6270ULL);

  std::tie(res, fpsr) = AsmUqshl(arg1, -33);
  ASSERT_EQ(res, MakeUInt128(0U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqshl(arg1, -32);
  ASSERT_EQ(res, MakeUInt128(0U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqshl(arg1, -31);
  ASSERT_EQ(res, MakeUInt128(0U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqshl(arg1, -1);
  ASSERT_EQ(res, MakeUInt128(0x03a0'a2c2ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqshl(arg1, 0);
  ASSERT_EQ(res, MakeUInt128(0x0741'4585ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqshl(arg1, 1);
  ASSERT_EQ(res, MakeUInt128(0x0e82'8b0aULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqshl(arg1, 31);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqshl(arg1, 32);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqshl(arg1, 33);
  ASSERT_EQ(res, MakeUInt128(0xffff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr));
}

TEST(Arm64InsnTest, SignedSaturatingShiftLeftByRegisterImmInt16x8) {
  constexpr auto AsmSqshl = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqshl %0.8h, %2.8h, %3.8h");

  __uint128_t arg1 = 0U;
  __uint128_t arg2 = MakeUInt128(0xffdf'ffe0'ffe1'ffffULL, 0x0001'001f'0020'0021ULL);
  auto [res1, fpsr1] = AsmSqshl(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x3333'3333'3333'3333ULL, 0x3333'3333'3333'3333ULL);
  auto [res2, fpsr2] = AsmSqshl(arg3, arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'1999ULL, 0x6666'7fff'7fff'7fffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingShiftLeftByRegisterImmInt16x8) {
  constexpr auto AsmUqshl = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqshl %0.8h, %2.8h, %3.8h");

  __uint128_t arg1 = 0U;
  __uint128_t arg2 = MakeUInt128(0xffdf'ffe0'ffe1'ffffULL, 0x0001'001f'0020'0021ULL);
  auto [res1, fpsr1] = AsmUqshl(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x7777'7777'7777'7777ULL, 0x7777'7777'7777'7777ULL);
  auto [res2, fpsr2] = AsmUqshl(arg3, arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'3bbbULL, 0xeeee'ffff'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingExtractNarrowInt64x2ToInt32x2) {
  constexpr auto AsmSqxtn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqxtn %0.2s, %2.2d");

  __uint128_t arg1 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqxtn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x8000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0000'0000'0123'4567ULL, 0x0000'0000'7ecd'ba98LL);
  auto [res2, fpsr2] = AsmSqxtn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x7ecd'ba98'0123'4567ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingExtractNarrowInt64x1ToInt32x1) {
  constexpr auto AsmSqxtn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqxtn %s0, %d2");

  __uint128_t arg1 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x0ULL);
  auto [res1, fpsr1] = AsmSqxtn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0000'0000'1234'5678ULL, 0x0ULL);
  auto [res2, fpsr2] = AsmSqxtn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'1234'5678ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingExtractNarrowInt64x2ToInt32x2) {
  constexpr auto AsmUqstn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("uqxtn %0.2s, %2.2d");

  __uint128_t arg1 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmUqstn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0000'0000'0123'4567ULL, 0x0000'0000'fecd'ba98LL);
  auto [res2, fpsr2] = AsmUqstn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xfecd'ba98'0123'4567ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingExtractNarrowInt64x1ToInt32x1) {
  constexpr auto AsmUqxtn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("uqxtn %s0, %d2");

  __uint128_t arg1 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x0ULL);
  auto [res1, fpsr1] = AsmUqxtn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0000'0000'8765'4321ULL, 0x0ULL);
  auto [res2, fpsr2] = AsmUqxtn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'8765'4321ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingExtractNarrow2Int64x2ToInt32x2) {
  constexpr auto AsmSqxtn2 = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("sqxtn2 %0.4s, %2.2d");

  __uint128_t arg1 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x6121'8656'1967'3378ULL, 0x6236'2561'2521'6320ULL);
  auto [res1, fpsr1] = AsmSqxtn2(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x6121'8656'1967'3378ULL, 0x8000'0000'7fff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x0000'0000'0123'4567ULL, 0x0000'0000'7ecd'ba98LL);
  __uint128_t arg4 = MakeUInt128(0x6121'8656'1967'3378ULL, 0x6236'2561'2521'6320ULL);
  auto [res2, fpsr2] = AsmSqxtn2(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x6121'8656'1967'3378ULL, 0x7ecd'ba98'0123'4567ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingExtractNarrow2Int64x2ToInt32x4) {
  constexpr auto AsmUqxtn2 = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("uqxtn2 %0.4s, %2.2d");

  __uint128_t arg1 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x6121'8656'1967'3378ULL, 0x6236'2561'2521'6320ULL);
  auto [res1, fpsr1] = AsmUqxtn2(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x6121'8656'1967'3378ULL, 0xffff'ffff'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x0000'0000'0123'4567ULL, 0x0000'0000'fecd'ba98LL);
  __uint128_t arg4 = MakeUInt128(0x6121'8656'1967'3378ULL, 0x6236'2561'2521'6320ULL);
  auto [res2, fpsr2] = AsmUqxtn2(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x6121'8656'1967'3378ULL, 0xfecd'ba98'0123'4567ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingExtractUnsignedNarrowInt64x2ToInt32x2) {
  constexpr auto AsmSqxtun = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqxtun %0.2s, %2.2d");

  __uint128_t arg1 = MakeUInt128(0x0000'0000'4433'2211ULL, 0x0000'0001'aabb'ccddULL);
  auto [res1, fpsr1] = AsmSqxtun(arg1);
  ASSERT_EQ(res1, MakeUInt128(0xffff'ffff'4433'2211ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0000'0000'0123'4567ULL, 0x0000'0000'fecd'ba98LL);
  auto [res2, fpsr2] = AsmSqxtun(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xfecd'ba98'0123'4567ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingExtractUnsignedNarrowInt64x1ToInt32x1) {
  constexpr auto AsmSqxtun = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqxtun %s0, %d2");

  __uint128_t arg1 = MakeUInt128(0x0000'0001'ff33'2211ULL, 0x0ULL);
  auto [res1, fpsr1] = AsmSqxtun(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0000'0000'ff33'2211ULL, 0x0ULL);
  auto [res2, fpsr2] = AsmSqxtun(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'ff33'2211ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingExtractUnsignedNarrow2Int64x2ToInt32x4) {
  constexpr auto AsmSqxtun2 = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("sqxtun2 %0.4s, %2.2d");

  __uint128_t arg1 = MakeUInt128(0x0000'0000'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqxtun2(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0123'4567'89ab'cdefULL, 0x0000'0000'89ab'cdefULL));
  ASSERT_TRUE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x0000'0000'0123'4567ULL, 0x0000'0000'fecd'ba98LL);
  __uint128_t arg4 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res2, fpsr2] = AsmSqxtun2(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfecd'ba98'0123'4567ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingAccumulateOfUnsignedValueInt32x1) {
  constexpr auto AsmSuqadd = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("suqadd %s0, %s2");

  __uint128_t arg1 = MakeUInt128(0x9392'0231'1563'8719ULL, 0x5080'5024'6797'2579ULL);
  __uint128_t arg2 = MakeUInt128(0x2497'6057'6262'5913ULL, 0x3285'5972'6371'2112ULL);
  auto [res1, fpsr1] = AsmSuqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'77c5'e02cULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x9099'7917'7668'7477ULL, 0x4481'8828'7063'2315ULL);
  __uint128_t arg4 = MakeUInt128(0x5158'6503'2898'1642ULL, 0x2828'8232'7468'6610ULL);
  auto [res2, fpsr2] = AsmSuqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingAccumulateOfUnsignedValueInt32x4) {
  constexpr auto AsmSuqadd = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("suqadd %0.4s, %2.4s");

  __uint128_t arg1 = MakeUInt128(0x2590'1810'0035'0989ULL, 0x2864'1204'1951'6355ULL);
  __uint128_t arg2 = MakeUInt128(0x1108'7632'0426'7612ULL, 0x9798'2652'9425'8829ULL);
  auto [res1, fpsr1] = AsmSuqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x3698'8e42'045b'7f9bULL, 0xbffc'3856'ad76'eb7eULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x9082'8889'3493'8376ULL, 0x4393'9925'6900'6040ULL);
  __uint128_t arg4 = MakeUInt128(0x6731'1422'0933'1219ULL, 0x5936'2029'8297'2351ULL);
  auto [res2, fpsr2] = AsmSuqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'3dc6'958fULL, 0x7fff'ffff'eb97'8391ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingAccumulateOfSignedValueInt32x1) {
  constexpr auto AsmUsqadd = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("usqadd %s0, %s2");

  __uint128_t arg1 = MakeUInt128(0x9052'5232'4234'8615ULL, 0x3152'0976'9384'6104ULL);
  __uint128_t arg2 = MakeUInt128(0x2582'8497'1496'3475ULL, 0x3418'3756'2003'0149ULL);
  auto [res1, fpsr1] = AsmUsqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'56ca'ba8aULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x9887'1253'8780'1719ULL, 0x6071'8164'0781'2484ULL);
  __uint128_t arg4 = MakeUInt128(0x7847'2579'1240'7824ULL, 0x5443'6168'2345'2395ULL);
  auto [res2, fpsr2] = AsmUsqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  __uint128_t arg5 = MakeUInt128(0x9708'5839'7076'1645ULL, 0x8229'6303'2442'4328ULL);
  __uint128_t arg6 = MakeUInt128(0x2377'3745'9517'0285ULL, 0x6069'8067'8895'2176ULL);
  auto [res3, fpsr3] = AsmUsqadd(arg5, arg6);
  ASSERT_EQ(res3, MakeUInt128(0x0000'0000'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, UnsignedSaturatingAccumulateOfSignedValueInt32x4) {
  constexpr auto AsmUsqadd = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("usqadd %0.4s, %2.4s");

  __uint128_t arg1 = MakeUInt128(0x4129'1370'7498'2305ULL, 0x7592'9091'6629'3919ULL);
  __uint128_t arg2 = MakeUInt128(0x5014'7211'5758'6067ULL, 0x2700'9254'7718'0257ULL);
  auto [res1, fpsr1] = AsmUsqadd(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x913d'8581'cbf0'836cULL, 0x9c93'22e5'dd41'3b70ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x7816'4228'2882'3274ULL, 0x6866'1065'9273'2197ULL);
  __uint128_t arg4 = MakeUInt128(0x9071'6238'4642'1534ULL, 0x8985'2476'2167'8905ULL);
  auto [res2, fpsr2] = AsmUsqadd(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0xffff'ffff'6ec4'47a8ULL, 0xf1eb'34db'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingShiftLeftInt32x1) {
  constexpr auto AsmSqrshl = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrshl %s0, %s2, %s3");

  __uint128_t res;
  uint32_t fpsr;

  __uint128_t arg = MakeUInt128(0x9736'7054'3558'0445ULL, 0x8657'2022'7637'8404ULL);
  std::tie(res, fpsr) = AsmSqrshl(arg, -33);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqrshl(arg, -32);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqrshl(arg, -31);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqrshl(arg, -1);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'1aac'0223ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqrshl(arg, 0);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'3558'0445ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqrshl(arg, 1);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'6ab0'088aULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqrshl(arg, 31);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqrshl(arg, 32);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmSqrshl(arg, 33);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingShiftLeftInt16x8) {
  constexpr auto AsmSqrshl = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrshl %0.8h, %2.8h, %3.8h");

  __uint128_t arg1 = MakeUInt128(0x0000'0000'0000'0099ULL, 0x9999'0999'9999'9999ULL);
  __uint128_t arg2 = MakeUInt128(0x0011'0010'000f'0001ULL, 0xffff'fff1'fff0'ffefULL);
  auto [res1, fpsr1] = AsmSqrshl(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0132ULL, 0xcccd'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x0099'0099'0099'0099ULL, 0x0099'0099'0099'0099ULL);
  auto [res2, fpsr2] = AsmSqrshl(arg3, arg2);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'7fff'7fff'0132ULL, 0x004d'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingRoundingShiftLeftInt32x1) {
  constexpr auto AsmUqrshl = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqrshl %s0, %s2, %s3");

  __uint128_t res;
  uint32_t fpsr;

  __uint128_t arg = MakeUInt128(0x9984'1248'4826'2367ULL, 0x3771'4672'2606'1633ULL);
  std::tie(res, fpsr) = AsmUqrshl(arg, -33);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqrshl(arg, -32);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqrshl(arg, -31);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'0000'0001ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqrshl(arg, -1);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'2413'11b4ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqrshl(arg, 0);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'4826'2367ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqrshl(arg, 1);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'904c'46ceULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqrshl(arg, 31);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqrshl(arg, 32);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr));

  std::tie(res, fpsr) = AsmUqrshl(arg, 33);
  ASSERT_EQ(res, MakeUInt128(0x0000'0000'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr));
}

TEST(Arm64InsnTest, UnsignedSaturatingRoundingShiftLeftInt16x8) {
  constexpr auto AsmUqrshl = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("uqrshl %0.8h, %2.8h, %3.8h");

  __uint128_t arg1 = MakeUInt128(0x0000'0000'0000'0099ULL, 0x9999'0999'9999'9999ULL);
  __uint128_t arg2 = MakeUInt128(0x0011'0010'000f'0001ULL, 0xffff'fff1'fff0'ffefULL);
  auto [res1, fpsr1] = AsmUqrshl(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0132ULL, 0x4ccd'0000'0001'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x0099'0099'0099'0099ULL, 0x0099'0099'0099'0099ULL);
  auto [res2, fpsr2] = AsmUqrshl(arg3, arg2);
  ASSERT_EQ(res2, MakeUInt128(0xffff'ffff'ffff'0132ULL, 0x004d'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftRightNarrowInt16x1) {
  constexpr auto AsmSqshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshrn %b0, %h2, #4");

  __uint128_t arg1 = MakeUInt128(0x8887'8661'4762'f943ULL, 0x4140'1049'8889'9316ULL);
  auto [res1, fpsr1] = AsmSqshrn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x94U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0051'2076'7810'3588ULL, 0x6116'6020'2961'1936ULL);
  auto [res2, fpsr2] = AsmSqshrn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x7fU, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftRightNarrowInt16x8) {
  constexpr auto AsmSqshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshrn %0.8b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0625'0516'0434'0253ULL, 0x0299'0286'0267'0568ULL);
  auto [res1, fpsr1] = AsmSqshrn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x2928'2656'6251'4325ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x2405'8060'0564'2114ULL, 0x9386'4368'6422'4724ULL);
  auto [res2, fpsr2] = AsmSqshrn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x807f'7f7f'7f80'567fULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftRightNarrowInt16x8Upper) {
  constexpr auto AsmSqshrn2 = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("sqshrn2 %0.16b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0367'0347'0410'0536ULL, 0x0175'0648'0300'0078ULL);
  __uint128_t arg2 = MakeUInt128(0x3494'8192'6268'1110ULL, 0x7399'4825'0607'3949ULL);
  auto [res1, fpsr1] = AsmSqshrn2(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x3494'8192'6268'1110ULL, 0x1764'3007'3634'4153ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x4641'0745'0167'3719ULL, 0x0483'1096'7671'1344ULL);
  auto [res2, fpsr2] = AsmSqshrn2(arg3, arg2);
  ASSERT_EQ(res2, MakeUInt128(0x3494'8192'6268'1110ULL, 0x487f'7f7f'7f74'167fULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingShiftRightNarrowInt16x1) {
  constexpr auto AsmUqshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("uqshrn %b0, %h2, #4");

  __uint128_t arg1 = MakeUInt128(0x6797'1728'9822'0360ULL, 0x7028'8069'0877'6866ULL);
  auto [res1, fpsr1] = AsmUqshrn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x36U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x0593'2527'4637'8405ULL, 0x3976'9184'8082'0410ULL);
  auto [res2, fpsr2] = AsmUqshrn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xffU, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingShiftRightNarrowInt16x8) {
  constexpr auto AsmUqshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("uqshrn %0.8b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0867'0679'0760'0099ULL, 0x0693'0075'0949'0515ULL);
  auto [res1, fpsr1] = AsmUqshrn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x6907'9451'8667'7609ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x2736'0498'1189'0413ULL, 0x0433'1166'2774'7123ULL);
  auto [res2, fpsr2] = AsmUqshrn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x43ff'ffff'ff49'ff41ULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnignedSaturatingShiftRightNarrowInt16x8Upper) {
  constexpr auto AsmUqshrn2 = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("uqshrn2 %0.16b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0441'0184'0741'0768ULL, 0x0981'0663'0724'0048ULL);
  __uint128_t arg2 = MakeUInt128(0x2393'5827'4019'4493ULL, 0x5665'1610'8846'3125ULL);
  auto [res1, fpsr1] = AsmUqshrn2(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x2393'5827'4019'4493ULL, 0x9866'7204'4418'7476ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x0785'2977'0973'4684ULL, 0x3030'6146'2418'0358ULL);
  auto [res2, fpsr2] = AsmUqshrn2(arg3, arg2);
  ASSERT_EQ(res2, MakeUInt128(0x2393'5827'4019'4493ULL, 0xffff'ff35'78ff'97ffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingShiftRightNarrowInt16x1) {
  constexpr auto AsmSqrshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqrshrn %b0, %h2, #4");

  __uint128_t arg1 = MakeUInt128(0x9610'3307'9941'0534ULL, 0x7784'5746'9999'2128ULL);
  auto [res1, fpsr1] = AsmSqrshrn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0053ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x5999'9939'9612'2816ULL, 0x1521'9314'8887'6938ULL);
  auto [res2, fpsr2] = AsmSqrshrn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'007fULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  __uint128_t arg3 = MakeUInt128(0x8022'2810'8300'9986ULL, 0x0165'4941'6542'6169ULL);
  auto [res3, fpsr3] = AsmSqrshrn(arg3);
  ASSERT_EQ(res3, MakeUInt128(0x0000'0000'0000'0080ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingShiftRightNarrowInt16x8) {
  constexpr auto AsmSqrshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqrshrn %0.8b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0666'0704'0170'0260ULL, 0x0520'0592'0493'0759ULL);
  auto [res1, fpsr1] = AsmSqrshrn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x5259'4976'6670'1726ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x4143'4081'4685'2981ULL, 0x5053'9471'7890'0451ULL);
  auto [res2, fpsr2] = AsmSqrshrn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x7f80'7f45'7f7f'7f7fULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingShiftRightNarrowInt16x8Upper) {
  constexpr auto AsmSqrshrn2 = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("sqrshrn2 %0.16b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0784'0171'0396'0497ULL, 0x0707'0725'0174'0336ULL);
  __uint128_t arg2 = MakeUInt128(0x5662'7259'2844'0620ULL, 0x4302'1411'3719'9227ULL);
  auto [res1, fpsr1] = AsmSqrshrn2(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x5662'7259'2844'0620ULL, 0x7072'1733'7817'3949ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x2066'8865'1275'6882ULL, 0x6614'9730'7886'5701ULL);
  __uint128_t arg4 = MakeUInt128(0x5685'0169'1864'7488ULL, 0x5416'7915'4596'5072ULL);
  auto [res2, fpsr2] = AsmSqrshrn2(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x5685'0169'1864'7488ULL, 0x7f80'7f7f'7f80'7f7fULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingRoundingShiftRightNarrowInt16x1) {
  constexpr auto AsmUqrshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("uqrshrn %b0, %h2, #4");

  __uint128_t arg1 = MakeUInt128(0x9614'2365'8595'0920ULL, 0x9083'0733'2335'6034ULL);
  auto [res1, fpsr1] = AsmUqrshrn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0092ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x8465'3187'3029'9026ULL, 0x6596'4501'3718'3754ULL);
  auto [res2, fpsr2] = AsmUqrshrn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'00ffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingRoundingShiftRightNarrowInt16x8) {
  constexpr auto AsmUqrshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("uqrshrn %0.8b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0301'0676'0386'0240ULL, 0x0011'0304'0247'0073ULL);
  auto [res1, fpsr1] = AsmUqrshrn(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x0130'2407'3067'3824ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x5085'0828'7246'2713ULL, 0x4946'3685'0181'5469ULL);
  auto [res2, fpsr2] = AsmUqrshrn(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xffff'18ff'ff83'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, UnsignedSaturatingRoundingShiftRightNarrowInt16x8Upper) {
  constexpr auto AsmUqrshrn = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("uqrshrn2 %0.16b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0388'0990'0573'0661ULL, 0x0237'0223'0478'0112ULL);
  __uint128_t arg2 = MakeUInt128(0x0392'2691'1027'7722ULL, 0x6102'5441'4922'1576ULL);
  auto [res1, fpsr1] = AsmUqrshrn(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0392'2691'1027'7722ULL, 0x2322'4811'3999'5766ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x9254'0696'1760'0504ULL, 0x7974'9280'6072'1268ULL);
  __uint128_t arg4 = MakeUInt128(0x8414'6957'2639'7884ULL, 0x2560'0845'3121'4065ULL);
  auto [res2, fpsr2] = AsmUqrshrn(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x8414'6957'2639'7884ULL, 0xffff'ffff'ff69'ff50ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftRightUnsignedNarrowInt16x1) {
  constexpr auto AsmSqshrun = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshrun %b0, %h2, #4");

  __uint128_t arg1 = MakeUInt128(0x9143'6114'3992'0063ULL, 0x8005'0832'1409'8760ULL);
  auto [res1, fpsr1] = AsmSqshrun(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x06U, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x3815'1745'7125'9975ULL, 0x4953'5802'3998'3146ULL);
  auto [res2, fpsr2] = AsmSqshrun(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x00U, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  __uint128_t arg3 = MakeUInt128(0x4599'3093'2485'1025ULL, 0x1682'9446'7260'6661ULL);
  auto [res3, fpsr3] = AsmSqshrun(arg3);
  ASSERT_EQ(res3, MakeUInt128(0xffU, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingShiftRightUnsignedNarrowInt16x8) {
  constexpr auto AsmSqshrun = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshrun %0.8b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0911'0664'0834'0874ULL, 0x0800'0741'0725'0670ULL);
  auto [res1, fpsr1] = AsmSqshrun(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x8074'7267'9166'8387ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x4792'2583'1912'9415ULL, 0x7390'8091'4383'1384ULL);
  auto [res2, fpsr2] = AsmSqshrun(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xff00'ffff'ffff'ff00ULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftRightUnsignedNarrowInt16x8Upper) {
  constexpr auto AsmSqshrun2 = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("sqshrun2 %0.16b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0625'0821'0174'0415ULL, 0x0233'0749'0396'0353ULL);
  __uint128_t arg2 = MakeUInt128(0x0136'1786'5367'3760ULL, 0x6421'6677'8137'7399ULL);
  auto [res1, fpsr1] = AsmSqshrun2(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0136'1786'5367'3760ULL, 0x2374'3935'6282'1741ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x4295'8105'4565'1083ULL, 0x1046'2972'8293'7584ULL);
  __uint128_t arg4 = MakeUInt128(0x1611'6253'2562'5165ULL, 0x7249'8078'4920'9989ULL);
  auto [res2, fpsr2] = AsmSqshrun2(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x1611'6253'2562'5165ULL, 0xffff'00ff'ff00'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingShiftRightUnsignedNarrowInt16x1) {
  constexpr auto AsmSqrshrun = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqrshrun %b0, %h2, #4");

  __uint128_t arg1 = MakeUInt128(0x5760'1869'4649'0886ULL, 0x8154'5285'6213'4698ULL);
  auto [res1, fpsr1] = AsmSqrshrun(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x88ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x8355'4445'6024'9556ULL, 0x6684'3660'2922'1951ULL);
  auto [res2, fpsr2] = AsmSqrshrun(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x00ULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  __uint128_t arg3 = MakeUInt128(0x2483'0910'6053'7720ULL, 0x1980'2183'1010'3270ULL);
  auto [res3, fpsr3] = AsmSqrshrun(arg3);
  ASSERT_EQ(res3, MakeUInt128(0xffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingShiftRightUnsignedNarrowInt16x8) {
  constexpr auto AsmSqrshrun = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqrshrun %0.8b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0150'0690'0149'0702ULL, 0x0673'0338'0834'0550ULL);
  auto [res1, fpsr1] = AsmSqrshrun(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x6734'8355'1569'1570ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x8363'6601'7848'7710ULL, 0x6080'9804'2692'4713ULL);
  auto [res2, fpsr2] = AsmSqrshrun(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xff00'ffff'00ff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingShiftRightUnsignedNarrowInt16x8Upper) {
  constexpr auto AsmSqrshrun2 = ASM_INSN_WRAP_FUNC_WQ_RES_W0_ARG("sqrshrun2 %0.16b, %2.8h, #4");

  __uint128_t arg1 = MakeUInt128(0x0733'0495'0208'0757ULL, 0x0651'0187'0599'0498ULL);
  __uint128_t arg2 = MakeUInt128(0x5693'7956'2387'5551ULL, 0x6175'7543'8091'7805ULL);
  auto [res1, fpsr1] = AsmSqrshrun2(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x5693'7956'2387'5551ULL, 0x6518'5a4a'7349'2175ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x1444'6712'9861'5527ULL, 0x5982'0145'1410'2756ULL);
  __uint128_t arg4 = MakeUInt128(0x0068'9297'5024'6304ULL, 0x0173'5148'9194'5763ULL);
  auto [res2, fpsr2] = AsmSqrshrun2(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0068'9297'5024'6304ULL, 0xff14'ffff'ffff'00ffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingShiftLeftUnsignedImmInt32x1) {
  constexpr auto AsmSqshlu = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshlu %s0, %s2, #4");

  __uint128_t arg1 = MakeUInt128(0x9704'0330'0186'2556ULL, 0x1473'3211'7771'1744ULL);
  auto [res1, fpsr1] = AsmSqshlu(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x1862'5560ULL, 0U));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x3095'7601'9694'6490ULL, 0x8868'1545'2856'2134ULL);
  auto [res2, fpsr2] = AsmSqshlu(arg2);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000ULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  __uint128_t arg3 = MakeUInt128(0x1335'0281'6088'4035ULL, 0x1781'4525'4196'4320ULL);
  auto [res3, fpsr3] = AsmSqshlu(arg3);
  ASSERT_EQ(res3, MakeUInt128(0xffff'ffffULL, 0U));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingShiftLeftUnsignedImmInt32x4) {
  constexpr auto AsmSqshlu = ASM_INSN_WRAP_FUNC_WQ_RES_W_ARG("sqshlu %0.4s, %2.4s, #4");

  __uint128_t arg1 = MakeUInt128(0x0865'1745'0787'7133ULL, 0x0813'8752'0598'0941ULL);
  auto [res1, fpsr1] = AsmSqshlu(arg1);
  ASSERT_EQ(res1, MakeUInt128(0x8651'7450'7877'1330ULL, 0x8138'7520'5980'9410ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg2 = MakeUInt128(0x2174'2273'0035'2296ULL, 0x0080'8917'9705'0682ULL);
  auto [res2, fpsr2] = AsmSqshlu(arg2);
  ASSERT_EQ(res2, MakeUInt128(0xffff'ffff'0352'2960ULL, 0x0808'9170'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLong32x2) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull %0.2d, %2.2s, %3.2s");

  __uint128_t arg1 = MakeUInt128(0x0000'0002'0000'0004ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0003'0000'0002ULL, 0xfee'd000'4000'0002ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0010ULL, 0x0000'0000'0000'000cULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'0000'0000'0004ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg4 = MakeUInt128(0x8000'0000'0000'0002ULL, 0xfee'd000'4000'0002ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'0010ULL, 0x7fff'ffff'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLong16x4) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull %0.4s, %2.4h, %3.4h");

  __uint128_t arg1 = MakeUInt128(0x0004'0002'00f0'0004ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg2 = MakeUInt128(0x0008'0003'0080'0002ULL, 0xabcd'0123'ffff'4567ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'f000'0000'0010ULL, 0x0000'0040'0000'000cULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'0002'00f0'0004ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg4 = MakeUInt128(0x8000'0003'0080'0002ULL, 0xabcd'0123'ffff'4567ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'f000'0000'0010ULL, 0x7fff'ffff'0000'000cULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLongUpper32x2) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull2 %0.2d, %2.4s, %3.4s");

  __uint128_t arg1 = MakeUInt128(0x0000'0002'0000'0004ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0003'0000'0002ULL, 0xfee'd000'4000'0002ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0008'0000'0040ULL, 0xffdd'c4ed'7f98'e000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'0000'0000'0004ULL, 0x8000'0000'0000'0010ULL);
  __uint128_t arg4 = MakeUInt128(0x8000'0000'0000'0002ULL, 0x8000'0000'0000'0002ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'0040ULL, 0x7fff'ffff'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLongUpper16x4) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull2 %0.4s, %2.8h, %3.8h");

  __uint128_t arg1 = MakeUInt128(0x0004'0002'00f0'0004ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg2 = MakeUInt128(0x0008'0003'0080'0002ULL, 0xabcd'0123'ffff'4567ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0226'ff6a'e4b6ULL, 0x00b4'e592'fffd'8eceULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'0000'0000'0004ULL, 0x8000'0000'0000'0010ULL);
  __uint128_t arg4 = MakeUInt128(0x8000'0000'0000'0002ULL, 0x8000'0000'0000'0002ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'0040ULL, 0x7fff'ffff'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLong64x2IndexedElem) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull %0.2d, %2.2s, %3.s[1]");

  __uint128_t arg1 = MakeUInt128(0x0022'0022'1122'3344ULL, 0x1122'3344'0011'0011LL);
  __uint128_t arg2 = MakeUInt128(0x0000'0002'0000'0000ULL, 0x000'0000'0000'0000ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'4488'cd10ULL, 0x0000'0000'0088'0088ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x0022'0022'8000'0000ULL, 0x1122'3344'0011'0011LL);
  __uint128_t arg4 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x000'0000'0000'0000ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0xffdd'ffde'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLong32x4IndexedElem) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull %0.4s, %2.4h, %3.h[4]");

  __uint128_t arg1 = MakeUInt128(0x0022'0022'1122'3344ULL, 0x1122'3344'0011'0011LL);
  __uint128_t arg2 = MakeUInt128(0x000f'000f'000f'000fULL, 0x000f'000f'000f'0002ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'4488'0000'cd10ULL, 0x0000'0088'0000'0088ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x0022'0022'8000'0000ULL, 0x1122'3344'0011'8000ULL);
  __uint128_t arg4 = MakeUInt128(0x1111'1111'2222'2222ULL, 0x1122'3344'1122'8000ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'0000'0000ULL, 0xffde'0000'ffde'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLongUpper64x2IndexedElem) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull2 %0.2d, %2.4s, %3.s[3]");

  __uint128_t arg1 = MakeUInt128(0x0022'0022'1122'3344ULL, 0x1122'3344'0011'0011ULL);
  __uint128_t arg2 = MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0000'0002'ffff'ffffULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0044'0044ULL, 0x0000'0000'4488'cd10ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'0000'ffff'ffffULL, 0x1122'3344'8000'0000ULL);
  __uint128_t arg4 = MakeUInt128(0x1122'3344'1122'3344ULL, 0x8000'0000'ffff'ffffULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0xeedd'ccbc'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLongUpper32x4IndexedElem) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull2 %0.4s, %2.8h, %3.h[7]");

  __uint128_t arg1 = MakeUInt128(0x0022'0022'1122'3344ULL, 0x1122'3344'0011'0011ULL);
  __uint128_t arg2 = MakeUInt128(0xffff'ffff'ffff'ffffULL, 0x0002'ffff'ffff'ffffULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0044'0000'0044ULL, 0x0000'4488'0000'cd10ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'0000'ffff'ffffULL, 0x1122'3344'8000'ffffULL);
  __uint128_t arg4 = MakeUInt128(0x1122'3344'1122'3344ULL, 0x8000'ffff'ffff'ffffULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'0001'0000ULL, 0xeede'0000'ccbc'0000ULL));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLong64x1) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull %d0, %s2, %s3");
  __uint128_t arg1 = MakeUInt128(0x0000'0008'1111'2222ULL, 0x0000'0007'0000'0006ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0005'1000'0000ULL, 0x0000'0003'0000'0002ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0222'2444'4000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0xaabb'ccdd'8000'0000ULL, 0x1122'3344'0011'0011ULL);
  __uint128_t arg4 = MakeUInt128(0xff11'ff11'8000'0000ULL, 0xffff'ffff'1122'3344ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLong32x1) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull %s0, %h2, %h3");
  __uint128_t arg1 = MakeUInt128(0x1111'1118'1111'2222ULL, 0xf000'0007'0008'0006ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0005'1000'4444ULL, 0xf000'0003'0008'0002ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'1234'3210ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0xaabb'ccdd'0000'8000ULL, 0x1122'3344'0011'0011ULL);
  __uint128_t arg4 = MakeUInt128(0xff11'ff11'0000'8000ULL, 0xffff'ffff'1122'3344ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLong32x1IndexedElem) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull %s0, %h2, %3.h[7]");
  __uint128_t arg1 = MakeUInt128(0x0000'0008'1111'2222ULL, 0x0000'0007'0000'0006ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0005'1000'0000ULL, 0x1111'0003'0000'0002ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'048d'0c84ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0xaabb'ccdd'aabb'8000ULL, 0x1122'3344'0011'0011ULL);
  __uint128_t arg4 = MakeUInt128(0xff11'ff11'ff00'0ff0ULL, 0x8000'aabb'1122'3344ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyLong64x1IndexedElem) {
  constexpr auto AsmSqdmull = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmull %d0, %s2, %3.s[3]");
  __uint128_t arg1 = MakeUInt128(0x0000'0008'1111'2222ULL, 0x0000'0007'0000'0006ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0005'1000'0000ULL, 0x0000'0003'0000'0002ULL);
  auto [res1, fpsr1] = AsmSqdmull(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'6666'ccccULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0xaabb'ccdd'8000'0000ULL, 0x1122'3344'0011'0011ULL);
  __uint128_t arg4 = MakeUInt128(0xff11'ff11'ff00'0ff0ULL, 0x8000'0000'1122'3344ULL);
  auto [res2, fpsr2] = AsmSqdmull(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLong32x2) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal %0.2d, %2.2s, %3.2s");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0100'0101'1101'1100ULL, 0x0400'0400'8c00'8c00ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x8000'0000'0000'0004ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0x8000'0000'0000'0002ULL, 0xfeed'0004'0000'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x0000'0800'0000'0900ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0800'0000'0910ULL, 0x7fff'ffff'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'8801'3800ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLong16x4) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal %0.4s, %2.4h, %3.4h");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x8000'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0010'0011'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0100'0100'0101'1100ULL, 0x03f0'0400'0402'4600ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x8000'1111'1111'1111ULL, 0x1234'1234'1234'1234ULL);
  __uint128_t arg5 = MakeUInt128(0x8000'1111'1111'1111ULL, 0x1234'1234'1234'1234ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0369'cba9'0369'cba9ULL, 0x7fff'ffff'0369'cba9ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0001'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x7fff'ffff'1234'5678ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffff'1235'6678ULL, 0x0000'0a00'0001'3800ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLongUpper32x2) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal2 %0.2d, %2.4s, %3.4s");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x020d'4492'6c1c'e9e0ULL, 0x050d'4792'6f1c'ece0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x1234'5678'0000'0004ULL, 0x8000'0000'0110'0010ULL);
  __uint128_t arg5 = MakeUInt128(0x1234'5678'0000'0002ULL, 0x8000'0000'0110'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x0000'0800'0000'0900ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0002'4a00'6600'0d00ULL, 0x7fff'ffff'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x7fff'ffff'ffff'ffffULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x1341'9a0a'7d51'3f58ULL, 0x7fff'ffff'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLongUpper16x4) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal2 %0.4s, %2.8h, %3.8h");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x8000'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0010'0011'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x020d'03f8'1c24'e9e0ULL, 0x050d'06f8'1f24'ece0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x1111'1111'1111'1111ULL, 0x8000'1234'1234'1234ULL);
  __uint128_t arg5 = MakeUInt128(0x1111'1111'1111'1111ULL, 0x8000'1234'1234'1234ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x03b9'fa87'03b9'fa87ULL, 0x7fff'ffff'03b9'fa87ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0001'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x7fff'ffff'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x1341'5970'2d59'3f58ULL, 0x7fff'ffff'1b25'98e0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLong64x1) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal %d0, %s2, %s3");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'1122'3344ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'2000'0000ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x1234'5678'0000'00ffULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x167c'e349'0000'00ffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x1122'3344'8000'0000ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0xaabb'ccdd'8000'0000ULL, 0xfeed'0004'0000'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x1122'3344'1111'1111ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1122'3344'0011'1111ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0xaabb'ccdd'0022'2222ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLong32x1) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal %s0, %h2, %h3");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0101'1100ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x1122'3344'1122'8000ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0xaabb'ccdd'aabb'8000ULL, 0xfeed'0004'0000'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x1122'3344'1111'1111ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1122'3344'1122'0123ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0xaabb'ccdd'aabb'0044ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0xaabb'ccdd'7fff'ffffULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLong64x2IndexedElem) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal %0.2d, %2.2s, %3.s[1]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0100'0101'1101'1100ULL, 0x0400'0400'8c00'8c00ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x8000'0000'0000'0004ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0x8000'0000'0000'0002ULL, 0xfeed'0004'0000'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x0000'0800'0000'0900ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0000'07fc'0000'0900ULL, 0x7fff'ffff'ffff'ffffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'8801'3800ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLong32x4IndexedElem) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal %0.4s, %2.4h, %3.h[7]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0102'0304'0506'0708ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x1122'3344'8877'6655ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x012e'b10b'89bb'ca1fULL, 0xfedf'0524'765b'0d28ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x8000'0123'4567'89a4ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'fedc'ba12'3456ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0xbbbc'4567'777f'4567ULL, 0x7fff'ffff'0000'4567ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x8888'1111'2222'3333ULL, 0x0123'4567'ffff'eeeeULL);
  __uint128_t arg9 = MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffff'004d'4bffULL, 0x0026'b000'0027'5600ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLongUpper64x2IndexedElem) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal2 %0.2d, %2.4s, %3.s[3]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x020d'4492'6c1c'e9e0ULL, 0x050d'4792'6f1c'ece0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x1122'3344'8000'0000ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'0000'1122'3344ULL);
  __uint128_t arg6 = MakeUInt128(0x0101'0101'0202'0202ULL, 0x0303'0303'0404'0404ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0xf1e0'cfbf'0404'0404ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x1122'3344'4433'2211ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x010d'4d92'6b1d'98e0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLongUpper32x4IndexedElem) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal2 %0.4s, %2.8h, %3.h[7]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0102'0304'0506'0708ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x1122'3344'8877'6655ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0230'485f'8a1d'9e4fULL, 0xffe9'bd90'76c6'0270ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x0011'2233'4455'6677ULL, 0xfeed'feed'feed'8000ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'fedc'ba12'3456ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0236'4567'7fff'ffffULL, 0x0236'4567'0236'4567ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x8888'1111'2222'3333ULL, 0x0123'4567'ffff'eeeeULL);
  __uint128_t arg9 = MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffff'0071'd05fULL, 0x010d'0cf8'0072'8060ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLong64x1IndexedElem) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal %d0, %s2, %3.s[3]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0102'0304'0506'0708ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x1122'3344'8877'6655ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x012e'b3d4'd07f'c65fULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x0011'2233'8000'0000ULL, 0xfeed'feed'feed'8000ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'0000'ba12'3456ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x8888'1111'2222'3333ULL, 0x0123'4567'ffff'eeeeULL);
  __uint128_t arg9 = MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x7fff'ffff'ffff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyAddLong32x1IndexedElem) {
  constexpr auto AsmSqdmlal = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlal %s0, %h2, %3.h[7]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0102'0304'0506'0708ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x1122'3344'8877'6655ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqdmlal(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'89bb'ca1fULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x0011'2233'4455'8000ULL, 0xfeed'feed'feed'1234ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'fedc'ba12'3456ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlal(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the addition.
  __uint128_t arg7 = MakeUInt128(0xaabb'ccdd'eeff'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x8888'1111'2222'3333ULL, 0x0123'aabb'ccdd'eeffULL);
  __uint128_t arg9 = MakeUInt128(0xaabb'ccdd'7fff'ffffULL, 0x0011'2233'4455'6677ULL);
  auto [res3, fpsr3] = AsmSqdmlal(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x0000'0000'7fff'ffffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLong32x2) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl %0.2d, %2.2s, %3.2s");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0000'0000'8000'0001ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0001'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0000'1000'0000'0001ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0000'1003'ffff'fff9ULL, 0x0400'0400'0400'0400ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x8000'0000'0000'0004ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0x8000'0000'0000'0002ULL, 0xfeed'0004'0000'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x0000'0000'0000'0900ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'08f0ULL, 0x8000'0a00'0000'b001ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'09ff'7800'2800ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLong16x4) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl %0.4s, %2.4h, %3.4h");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x8000'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0010'0011'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0100'0100'00fe'f100ULL, 0x0410'0400'03fd'c200ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x8000'1111'1111'1111ULL, 0x1234'1234'1234'1234ULL);
  __uint128_t arg5 = MakeUInt128(0x8000'1111'1111'1111ULL, 0x1234'1234'1234'1234ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0xfedc'bf25'fedc'bf25ULL, 0x8123'4568'fedc'bf25ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0001'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x8000'0000'1234'5678ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x8000'0000'1233'4678ULL, 0x0000'0a00'0000'2800ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLongUpper32x2) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl2 %0.2d, %2.4s, %3.4s");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0xfff2'bd6d'95e3'1820ULL, 0x02f2'c06d'98e3'1b20ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x1234'5678'0000'0004ULL, 0x8000'0000'0110'0010ULL);
  __uint128_t arg5 = MakeUInt128(0x1234'5678'0000'0002ULL, 0x8000'0000'0110'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x0000'0800'0000'0900ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0xfffd'c5ff'9a00'0500ULL, 0x8000'0a00'0000'b001ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x8000'0000'0000'0000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x1127'12e5'a717'6d98ULL, 0x8000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLongUpper16x4) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl2 %0.4s, %2.8h, %3.8h");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x8000'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0010'0011'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0xfff2'fe08'e5db'1820ULL, 0x02f3'0108'e8db'1b20ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x1111'1111'1111'1111ULL, 0x8000'1234'1234'1234ULL);
  __uint128_t arg5 = MakeUInt128(0x1111'1111'1111'1111ULL, 0x8000'1234'1234'1234ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0xfe8c'9047'fe8c'9047ULL, 0x8123'4568'fe8c'9047ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0001'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x1234'5678'1234'5678ULL, 0x8000'0000'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x1127'5380'f70f'6d98ULL, 0x8000'0000'e4db'c720ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLong64x1) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl %d0, %s2, %s3");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'1122'3344ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'2000'0000ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x1234'5678'0000'00ffULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0deb'c9a7'0000'00ffULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x1122'3344'8000'0000ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0xaabb'ccdd'8000'0000ULL, 0xfeed'0004'0000'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x1122'3344'1111'1111ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x9122'3344'1111'1112ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1122'3344'0011'1111ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0xaabb'ccdd'0022'2222ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLong32x1) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl %s0, %h2, %h3");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0000'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'00fe'f100ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x1122'3344'1122'8000ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0xaabb'ccdd'aabb'8000ULL, 0xfeed'0004'0000'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x1122'3344'1111'1111ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'9111'1112ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1122'3344'1122'0123ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0xaabb'ccdd'aabb'0044ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0xaabb'ccdd'8000'0000ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x0000'0000'8000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLong64x2IndexedElem) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl %0.2d, %2.2s, %3.s[1]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0100'00fe'f0fe'f100ULL, 0x0400'03ff'7bff'7c00ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x8000'0000'0000'0004ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0x8000'0000'0000'0002ULL, 0xfeed'0004'0000'0020ULL);
  __uint128_t arg6 = MakeUInt128(0x0000'0800'0000'0900ULL, 0x0000'0a00'0000'b000ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0804'0000'0900ULL, 0x8000'0a00'0000'b001ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'09ff'7800'2800ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLong32x4IndexedElem) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl %0.4s, %2.4h, %3.h[7]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0102'0304'0506'0708ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x1122'3344'8877'6655ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0117'd9c3'899b'd1bfULL, 0xfeda'700c'764d'56f8ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x8000'0123'4567'89a4ULL, 0xfeed'0003'0000'0010ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'fedc'ba12'3456ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x468a'4567'8ac7'4567ULL, 0x8123'4568'0246'4567ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x8888'1111'2222'3333ULL, 0x0123'4567'ffff'eeeeULL);
  __uint128_t arg9 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x8000'0000'ffb2'b400ULL, 0xffd9'6400'ffda'0a00ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLongUpper64x2IndexedElem) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl2 %0.2d, %2.4s, %3.s[3]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x0000'0004'0000'0004ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0100'0100'0100'0100ULL, 0x0400'0400'0400'0400ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0xfff2'bd6d'95e3'1820ULL, 0x02f2'c06d'98e3'1b20ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x1122'3344'8000'0000ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'0000'1122'3344ULL);
  __uint128_t arg6 = MakeUInt128(0x0101'0101'0202'0202ULL, 0x0303'0303'0404'0404ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x8101'0101'0202'0203ULL, 0x1425'3647'0404'0404ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x1122'3344'4433'2211ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg9 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x8000'0000'0000'0000ULL, 0xfef2'c66d'94e3'c720ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLongUpper32x4IndexedElem) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl2 %0.4s, %2.8h, %3.h[7]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0102'0304'0506'0708ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x1122'3344'8877'6655ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0016'426f'8939'fd8fULL, 0xfdcf'b7a0'75e2'61b0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x0011'2233'4455'6677ULL, 0xfeed'feed'feed'8000ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'fedc'ba12'3456ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0010'4567'8123'4568ULL, 0x0010'4567'0010'4567ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x8888'1111'2222'3333ULL, 0x0123'4567'ffff'eeeeULL);
  __uint128_t arg9 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x8000'0000'ff8e'2fa0ULL, 0xfef3'0708'ff8e'dfa0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLong64x1IndexedElem) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl %d0, %s2, %3.s[3]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0102'0304'0506'0708ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x1122'3344'8877'6655ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0117'd6fa'42d7'd57fULL, 0x0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x0011'2233'8000'0000ULL, 0xfeed'feed'feed'8000ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'0000'ba12'3456ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x8123'4567'0123'4568ULL, 0x0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0x1100'1100'2200'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x8888'1111'2222'3333ULL, 0x0123'4567'ffff'eeeeULL);
  __uint128_t arg9 = MakeUInt128(0x8000'0000'0000'0000ULL, 0x0000'0a00'0000'b000ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x8000'0000'0000'0000ULL, 0x0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplySubtractLong32x1IndexedElem) {
  constexpr auto AsmSqdmlsl = ASM_INSN_WRAP_FUNC_WQ_RES_WW0_ARG("sqdmlsl %s0, %h2, %3.h[7]");

  // No saturation.
  __uint128_t arg1 = MakeUInt128(0x0102'0304'0506'0708ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg2 = MakeUInt128(0x1122'3344'8877'6655ULL, 0x0123'4567'0123'4567ULL);
  __uint128_t arg3 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
  auto [res1, fpsr1] = AsmSqdmlsl(arg1, arg2, arg3);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'899b'd1bfULL, 0x0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  // Saturates in the multiplication.
  __uint128_t arg4 = MakeUInt128(0x0011'2233'4455'8000ULL, 0xfeed'feed'feed'1234ULL);
  __uint128_t arg5 = MakeUInt128(0x0123'4567'89ab'cdefULL, 0x8000'fedc'ba12'3456ULL);
  __uint128_t arg6 = MakeUInt128(0x0123'4567'0123'4567ULL, 0x0123'4567'0123'4567ULL);
  auto [res2, fpsr2] = AsmSqdmlsl(arg4, arg5, arg6);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'8123'4568ULL, 0x0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));

  // Saturates in the subtraction.
  __uint128_t arg7 = MakeUInt128(0xaabb'ccdd'eeff'2200ULL, 0x7654'3210'7654'3210ULL);
  __uint128_t arg8 = MakeUInt128(0x8888'1111'2222'3333ULL, 0x0123'aabb'ccdd'eeffULL);
  __uint128_t arg9 = MakeUInt128(0xaabb'ccdd'8000'0000ULL, 0x0011'2233'4455'6677ULL);
  auto [res3, fpsr3] = AsmSqdmlsl(arg7, arg8, arg9);
  ASSERT_EQ(res3, MakeUInt128(0x0000'0000'8000'0000ULL, 0x0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr3));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf32x4) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %0.4s, %2.4s, %3.4s");

  __uint128_t arg1 = MakeU32x4(0x2000'0001UL, 0x0000'0004UL, 0x7eed'0003UL, 0x0000'0010UL);
  __uint128_t arg2 = MakeU32x4(0x0000'0008UL, 0x0000'0002UL, 0x7eed'0004UL, 0x0000'0002UL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x2UL, 0x0UL, 0x7ddc'4ed9UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0x0000'0004UL, 0xfeed'0003UL, 0x0000'0010UL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0x0000'0002UL, 0xfeed'0004UL, 0x0000'0002UL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0x0UL, 0x0002'4ed2UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf32x2) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %0.2s, %2.2s, %3.2s");

  __uint128_t arg1 = MakeU32x4(0x5555'5555UL, 0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL);
  __uint128_t arg2 = MakeU32x4(0x0000'0004UL, 0x0000'0002UL, 0xdead'c0deUL, 0xdead'c0deUL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x3, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0x0000'0002UL, 0xdead'c0deUL, 0xdead'c0deUL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf16x8) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %0.8h, %2.8h, %3.8h");

  __uint128_t arg1 = MakeUInt128(0x2000'0001'7fff'1111ULL, 0x7eed'0003'0000'0010ULL);
  __uint128_t arg2 = MakeUInt128(0x0008'0008'4000'0000ULL, 0x7eed'0004'0000'0002ULL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0002'0000'4000'0000ULL, 0x7ddc'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'7000'4001'0000ULL, 0xfeed'0003'ffff'0010ULL);
  __uint128_t arg4 = MakeUInt128(0x8000'0001'0004'0000ULL, 0xfeed'0004'ffff'0002ULL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'0001'0002'0000ULL, 0x0002'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf16x4) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %0.4h, %2.4h, %3.4h");

  __uint128_t arg1 = MakeUInt128(0x5555'0001'7fff'1111ULL, 0xdead'c0de'dead'c0deULL);
  __uint128_t arg2 = MakeUInt128(0x0004'0008'4000'0000ULL, 0xdead'c0de'dead'c0deULL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0003'0000'4000'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'7000'4001'0000ULL, 0xdead'c0de'dead'c0deULL);
  __uint128_t arg4 = MakeUInt128(0x8000'0001'0004'0000ULL, 0xdead'c0de'dead'c0deULL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'0001'0002'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf32x4IndexedElem) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %0.4s, %2.4s, %3.s[0]");

  __uint128_t arg1 = MakeU32x4(0x2000'0001UL, 0x0000'0004UL, 0x7eed'0003, 0x0000'0010UL);
  __uint128_t arg2 = MakeU32x4(0x0000'0008UL, 0xfeed'feedUL, 0xfeed'feed, 0xfeed'feedUL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  // Without rounding, result should be 7 instead of 8.
  ASSERT_EQ(res1, MakeU32x4(0x2UL, 0x0UL, 0x8UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0x0000'0004UL, 0xfeed'0003UL, 0x0000'0010UL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0xffff'fffcUL, 0x0112'fffdUL, 0xffff'fff0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf32x2IndexedElem) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %0.2s, %2.2s, %3.s[0]");

  __uint128_t arg1 = MakeU32x4(0x5555'5555UL, 0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL);
  __uint128_t arg2 = MakeU32x4(0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL, 0xdead'c0deUL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x3UL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0xdead'c0deUL, 0xdead'c0deUL, 0xdead'c0deUL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0xffff'fffcUL, 0x0UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf16x8IndexedElem) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %0.8h, %2.8h, %3.h[7]");

  __uint128_t arg1 = MakeUInt128(0x7fff'8000'4567'0000ULL, 0xfe00'7800'2000'4001ULL);
  __uint128_t arg2 = MakeUInt128(0xfeed'feed'feed'feedULL, 0x0008'feed'feed'feedULL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0008'fff8'0004'0000ULL, 0x0000'0008'0002'0004ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x7fff'8000'4567'0000ULL, 0xfe00'7800'2000'4001ULL);
  __uint128_t arg4 = MakeUInt128(0xfeed'feed'feed'feedULL, 0x8000'feed'feed'feedULL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x8001'7fff'ba99'0000ULL, 0x0200'8800'e000'bfffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf16x4IndexedElem) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %0.4h, %2.4h, %3.h[7]");

  __uint128_t arg1 = MakeUInt128(0x7fff'8000'5555'0000ULL, 0xdead'c0de'dead'c0deULL);
  __uint128_t arg2 = MakeUInt128(0xdead'c0de'dead'c0deULL, 0x0004'c0de'dead'c0deULL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0004'fffc'0003'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x7fff'8000'4567'0000ULL, 0xdead'c0de'dead'c0deULL);
  __uint128_t arg4 = MakeUInt128(0xdead'c0de'dead'c0deULL, 0x8000'c0de'dead'c0deULL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x8001'7fff'ba99'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf32x1) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %s0, %s2, %s3");

  __uint128_t arg1 = MakeU32x4(0x5567'89abUL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  __uint128_t arg2 = MakeU32x4(0x0000'0004UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  // Without roundings, result should be 2 instead of 3.
  ASSERT_EQ(res1, MakeU32x4(0x3UL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf16x1) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %h0, %h2, %h3");

  __uint128_t arg1 = MakeUInt128(0xfeed'feed'feed'5567ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg2 = MakeUInt128(0xfeed'feed'feed'0004ULL, 0xfeed'feed'feed'feedULL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0003ULL, 0x0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0xfeed'feed'feed'8000ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg4 = MakeUInt128(0xfeed'feed'feed'8000ULL, 0xfeed'feed'feed'feedULL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'7fffULL, 0x0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf32x1IndexedElem) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %s0, %s2, %3.s[2]");

  __uint128_t arg1 = MakeU32x4(0x5567'89abUL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  __uint128_t arg2 = MakeU32x4(0xfeed'feedUL, 0xfeed'feedUL, 0x0000'0004UL, 0xfeed'feedUL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  // Without rounding, result should be 2 instead of 3.
  ASSERT_EQ(res1, MakeU32x4(0x3UL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  __uint128_t arg4 = MakeU32x4(0xfeed'feedUL, 0xfeed'feedUL, 0x8000'0000UL, 0xfeed'feedUL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingRoundingDoublingMultiplyHighHalf16x1IndexedElem) {
  constexpr auto AsmSqrdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqrdmulh %h0, %h2, %3.h[7]");

  __uint128_t arg1 = MakeUInt128(0xfeed'feed'feed'5567ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg2 = MakeUInt128(0xfeed'feed'feed'feedULL, 0x0004'feed'feed'feedULL);
  auto [res1, fpsr1] = AsmSqrdmulh(arg1, arg2);
  // Without rounding, result should be 2 instead of 3.
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0003ULL, 0x0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0xfeed'feed'feed'8000ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg4 = MakeUInt128(0xfeed'feed'feed'feedULL, 0x8000'feed'feed'feedULL);
  auto [res2, fpsr2] = AsmSqrdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'7fffULL, 0x0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf32x4) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %0.4s, %2.4s, %3.4s");

  __uint128_t arg1 = MakeU32x4(0x2000'0001UL, 0x0000'0004UL, 0x7eed'0003UL, 0x0000'0010UL);
  __uint128_t arg2 = MakeU32x4(0x0000'0008UL, 0x0000'0002UL, 0x7eed'0004UL, 0x0000'0002UL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x2UL, 0x0UL, 0x7ddc'4ed8UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0x0000'0004UL, 0xfeed'0003UL, 0x0000'0010UL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0x0000'0002UL, 0xfeed'0004UL, 0x0000'0002UL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0x0UL, 0x0002'4ed1UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf32x2) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %0.2s, %2.2s, %3.2s");

  __uint128_t arg1 = MakeU32x4(0x5555'5555UL, 0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL);
  __uint128_t arg2 = MakeU32x4(0x0000'0004UL, 0x0000'0002UL, 0xdead'c0deUL, 0xdead'c0deUL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x2UL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0x0000'0002UL, 0xdead'c0deUL, 0xdead'c0deUL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffff, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf16x8) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %0.8h, %2.8h, %3.8h");

  __uint128_t arg1 = MakeUInt128(0x2000'0001'7fff'1111ULL, 0x7eed'0003'0000'0010ULL);
  __uint128_t arg2 = MakeUInt128(0x0008'0008'4000'0000ULL, 0x7eed'0004'0000'0002ULL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0002'0000'3fff'0000ULL, 0x7ddc'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'7000'4001'0000ULL, 0xfeed'0003'ffff'0010ULL);
  __uint128_t arg4 = MakeUInt128(0x8000'0001'0004'0000ULL, 0xfeed'0004'ffff'0002ULL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'0000'0002'0000ULL, 0x0002'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf16x4) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %0.4h, %2.4h, %3.4h");

  __uint128_t arg1 = MakeUInt128(0x5555'0001'7fff'1111ULL, 0xdead'c0de'dead'c0deULL);
  __uint128_t arg2 = MakeUInt128(0x0004'0008'4000'0000ULL, 0xdead'c0de'dead'c0deULL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0002'0000'3fff'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x8000'7000'4001'0000ULL, 0xdead'c0de'dead'c0deULL);
  __uint128_t arg4 = MakeUInt128(0x8000'0001'0004'0000ULL, 0xdead'c0de'dead'c0deULL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x7fff'0000'0002'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf32x4IndexedElem) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %0.4s, %2.4s, %3.s[0]");

  __uint128_t arg1 = MakeU32x4(0x2000'0001UL, 0x0000'0004UL, 0x7eed'0003UL, 0x0000'0010UL);
  __uint128_t arg2 = MakeU32x4(0x0000'0008UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x2UL, 0x0UL, 0x7UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0x0000'0004UL, 0xfeed'0003UL, 0x0000'0010UL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0xffff'fffcUL, 0x0112'fffdUL, 0xffff'fff0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf32x2IndexedElem) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %0.2s, %2.2s, %3.s[0]");

  __uint128_t arg1 = MakeU32x4(0x5555'5555UL, 0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL);
  __uint128_t arg2 = MakeU32x4(0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL, 0xdead'c0deUL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x2UL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0x0000'0004UL, 0xdead'c0deUL, 0xdead'c0deUL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0xdead'c0deUL, 0xdead'c0deUL, 0xdead'c0deUL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0xffff'fffcUL, 0x0UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf16x8IndexedElem) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %0.8h, %2.8h, %3.h[7]");

  __uint128_t arg1 = MakeUInt128(0x7fff'8000'4567'0000ULL, 0xfe00'7800'2000'4001ULL);
  __uint128_t arg2 = MakeUInt128(0xfeed'feed'feed'feedULL, 0x0008'feed'feed'feedULL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0007'fff8'0004'0000ULL, 0xffff'0007'0002'0004ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x7fff'8000'4567'0000ULL, 0xfe00'7800'2000'4001ULL);
  __uint128_t arg4 = MakeUInt128(0xfeed'feed'feed'feedULL, 0x8000'feed'feed'feedULL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x8001'7fff'ba99'0000ULL, 0x0200'8800'e000'bfffULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf16x4IndexedElem) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %0.4h, %2.4h, %3.h[7]");

  __uint128_t arg1 = MakeUInt128(0x7fff'8000'5555'0000ULL, 0xdead'c0de'dead'c0deULL);
  __uint128_t arg2 = MakeUInt128(0xdead'c0de'dead'c0deULL, 0x0004'c0de'dead'c0deULL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0003'fffc'0002'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0x7fff'8000'4567'0000ULL, 0xdead'c0de'dead'c0deULL);
  __uint128_t arg4 = MakeUInt128(0xdead'c0de'dead'c0deULL, 0x8000'c0de'dead'c0deULL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x8001'7fff'ba99'0000ULL, 0x0000'0000'0000'0000ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf32x1) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %s0, %s2, %s3");

  __uint128_t arg1 = MakeU32x4(0x5567'89abUL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  __uint128_t arg2 = MakeU32x4(0x0000'0004UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x2UL, 0x0UL, 0x0UL, 0x0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  __uint128_t arg4 = MakeU32x4(0x8000'0000UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf16x1) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %h0, %h2, %h3");

  __uint128_t arg1 = MakeUInt128(0xfeed'feed'feed'5567ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg2 = MakeUInt128(0xfeed'feed'feed'0004ULL, 0xfeed'feed'feed'feedULL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0002ULL, 0x0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0xfeed'feed'feed'8000ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg4 = MakeUInt128(0xfeed'feed'feed'8000ULL, 0xfeed'feed'feed'feedULL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'7fffULL, 0x0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf32x1IndexedElem) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %s0, %s2, %3.s[2]");

  __uint128_t arg1 = MakeU32x4(0x5567'89abUL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  __uint128_t arg2 = MakeU32x4(0xfeed'feedUL, 0xfeed'feedUL, 0x0000'0004UL, 0xfeed'feedUL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeU32x4(0x2UL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeU32x4(0x8000'0000UL, 0xfeed'feedUL, 0xfeed'feedUL, 0xfeed'feedUL);
  __uint128_t arg4 = MakeU32x4(0xfeed'feedUL, 0xfeed'feedUL, 0x8000'0000UL, 0xfeed'feedUL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeU32x4(0x7fff'ffffUL, 0x0UL, 0x0UL, 0x0UL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

TEST(Arm64InsnTest, SignedSaturatingDoublingMultiplyHighHalf16x1IndexedElem) {
  constexpr auto AsmSqdmulh = ASM_INSN_WRAP_FUNC_WQ_RES_WW_ARG("sqdmulh %h0, %h2, %3.h[7]");

  __uint128_t arg1 = MakeUInt128(0xfeed'feed'feed'5567ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg2 = MakeUInt128(0xfeed'feed'feed'feedULL, 0x0004'feed'feed'feedULL);
  auto [res1, fpsr1] = AsmSqdmulh(arg1, arg2);
  ASSERT_EQ(res1, MakeUInt128(0x0000'0000'0000'0002ULL, 0x0ULL));
  ASSERT_FALSE(IsQcBitSet(fpsr1));

  __uint128_t arg3 = MakeUInt128(0xfeed'feed'feed'8000ULL, 0xfeed'feed'feed'feedULL);
  __uint128_t arg4 = MakeUInt128(0xfeed'feed'feed'feedULL, 0x8000'feed'feed'feedULL);
  auto [res2, fpsr2] = AsmSqdmulh(arg3, arg4);
  ASSERT_EQ(res2, MakeUInt128(0x0000'0000'0000'7fffULL, 0x0ULL));
  ASSERT_TRUE(IsQcBitSet(fpsr2));
}

class FpcrBitSupport : public testing::TestWithParam<uint64_t> {};

TEST_P(FpcrBitSupport, SupportsBit) {
  uint64_t fpcr1;
  asm("msr fpcr, %x1\n\t"
      "mrs %x0, fpcr"
      : "=r"(fpcr1)
      : "r"(static_cast<uint64_t>(GetParam())));
  ASSERT_EQ(fpcr1, GetParam()) << "Should be able to set then get FPCR bit: " << GetParam();
};

// Note: The exception enablement flags (such as IOE) are not checked, because when tested on actual
// ARM64 device we find that the tests fail either because they cannot be written or are RAZ (read
// as zero).
INSTANTIATE_TEST_SUITE_P(Arm64InsnTest,
                         FpcrBitSupport,
                         testing::Values(kFpcrRModeTieEven,
                                         kFpcrRModeZero,
                                         kFpcrRModeNegInf,
                                         kFpcrRModePosInf,
                                         kFpcrFzBit,
                                         kFpcrDnBit,
                                         0));

class FpsrBitSupport : public testing::TestWithParam<uint64_t> {};

TEST_P(FpsrBitSupport, SupportsBit) {
  uint64_t fpsr1;
  asm("msr fpsr, %1\n\t"
      "mrs %0, fpsr"
      : "=r"(fpsr1)
      : "r"(static_cast<uint64_t>(GetParam())));
  ASSERT_EQ(fpsr1, GetParam()) << "Should be able to set then get FPSR bit";
};

INSTANTIATE_TEST_SUITE_P(Arm64InsnTest,
                         FpsrBitSupport,
                         testing::Values(kFpsrIocBit,
                                         kFpsrDzcBit,
                                         kFpsrOfcBit,
                                         kFpsrUfcBit,
                                         kFpsrIxcBit,
                                         kFpsrIdcBit,
                                         kFpsrQcBit));

TEST(Arm64InsnTest, UnsignedDivide64) {
  auto udiv64 = [](uint64_t num, uint64_t den) {
    uint64_t result;
    asm("udiv %0, %1, %2" : "=r"(result) : "r"(num), "r"(den));
    return result;
  };
  ASSERT_EQ(udiv64(0x8'0000'0000ULL, 2ULL), 0x4'0000'0000ULL) << "Division should be 64-bit.";
  ASSERT_EQ(udiv64(123ULL, 0ULL), 0ULL) << "Div by 0 should result in 0.";
}

TEST(Arm64InsnTest, SignedDivide64) {
  auto div64 = [](int64_t num, int64_t den) {
    int64_t result;
    asm("sdiv %0, %1, %2" : "=r"(result) : "r"(num), "r"(den));
    return result;
  };
  ASSERT_EQ(div64(67802402LL, -1LL), -67802402LL)
      << "Division by -1 should flip sign if dividend is not numeric_limits::min.";
  ASSERT_EQ(div64(-531675317891LL, -1LL), 531675317891LL)
      << "Division by -1 should flip sign if dividend is not numeric_limits::min.";
  ASSERT_EQ(div64(std::numeric_limits<int64_t>::min(), -1LL), std::numeric_limits<int64_t>::min())
      << "Div of numeric_limits::min by -1 should result in numeric_limits::min.";
}

TEST(Arm64InsnTest, AesEncode) {
  __uint128_t arg = MakeUInt128(0x1111'2222'3333'4444ULL, 0x5555'6666'7777'8888ULL);
  __uint128_t key = MakeUInt128(0xaaaa'bbbb'cccc'ddddULL, 0xeeee'ffff'0000'9999ULL);
  __uint128_t res;
  asm("aese %0.16b, %2.16b" : "=w"(res) : "0"(arg), "w"(key));
  ASSERT_EQ(res, MakeUInt128(0x16ea'82ee'eaf5'eeeeULL, 0xf5ea'eeee'ea16'ee82ULL));
}

TEST(Arm64InsnTest, AesMixColumns) {
  __uint128_t arg = MakeUInt128(0x1111'2222'3333'4444ULL, 0x5555'6666'7777'8888ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("aesmc %0.16b, %1.16b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x7711'4422'dd33'aa44ULL, 0x3355'0066'9277'6d88ULL));
}

TEST(Arm64InsnTest, AesDecode) {
  // Check that it's opposite to AesEncode with extra XORs.
  __uint128_t arg = MakeUInt128(0x16ea'82ee'eaf5'eeeeULL, 0xf5ea'eeee'ea16'ee82ULL);
  __uint128_t key = MakeUInt128(0xaaaa'bbbb'cccc'ddddULL, 0xeeee'ffff'0000'9999ULL);
  arg ^= key;
  __uint128_t res;
  asm("aesd %0.16b, %2.16b" : "=w"(res) : "0"(arg), "w"(key));
  ASSERT_EQ(res ^ key, MakeUInt128(0x1111'2222'3333'4444ULL, 0x5555'6666'7777'8888ULL));
}

TEST(Arm64InsnTest, AesInverseMixColumns) {
  __uint128_t arg = MakeUInt128(0x7711'4422'dd33'aa44ULL, 0x3355'0066'9277'6d88ULL);
  __uint128_t res = ASM_INSN_WRAP_FUNC_W_RES_W_ARG("aesimc %0.16b, %1.16b")(arg);
  ASSERT_EQ(res, MakeUInt128(0x1111'2222'3333'4444ULL, 0x5555'6666'7777'8888ULL));
}

}  // namespace
