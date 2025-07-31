/*
 * Copyright (C) 2020 The Android Open Source Project
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

#include <cstring>

#include "berberis/base/bit_util.h"
#include "berberis/guest_abi/guest_arguments.h"

namespace berberis {

namespace {

TEST(NdkTest, GuestArgumentsAndResultTest) {
  union {
    GuestArgumentBuffer buffer;
    uint64_t padding[AlignUp(offsetof(GuestArgumentBuffer, stack_argv), sizeof(uint64_t)) /
                         sizeof(uint64_t) +
                     4];
  } arguments = {{.argv = {1, 2, 3, 4, 5, 6, 7, 8},
                  .simd_argv = {0x3ff0'0000'0000'0000,
                                0x3fe0'0000'0000'0000,
                                0x3fd5'5555'5555'5555,
                                0x3fd0'0000'0000'0000,
                                0x3fc9'9999'9999'999a,
                                0x3fc5'5555'5555'5555,
                                0x3fc2'4924'9249'2492,
                                0x3fc0'0000'0000'0000},
                  .stack_argv = {0x3fbc'71c7'1c71'c71c}}};
  arguments.buffer.stack_argv[1] = 0x3fb9'9999'9999'999a;
  arguments.buffer.stack_argv[2] = 0x3fb7'45d1'745d'1746;
  arguments.buffer.stack_argv[3] = 0x3fb5'5555'5555'5555;

  GuestArgumentsAndResult<double(int,
                                 double,
                                 int,
                                 double,
                                 int,
                                 double,
                                 int,
                                 double,
                                 int,
                                 double,
                                 int,
                                 double,
                                 int,
                                 double,
                                 int,
                                 double,
                                 int,
                                 double,
                                 int,
                                 double)>
      f1_args(&arguments.buffer);
  EXPECT_EQ(1, f1_args.GuestArgument<0>());
  EXPECT_EQ(1.0, f1_args.GuestArgument<1>());
  EXPECT_EQ(2, f1_args.GuestArgument<2>());
  EXPECT_EQ(1.0 / 2.0, f1_args.GuestArgument<3>());
  EXPECT_EQ(3, f1_args.GuestArgument<4>());
  EXPECT_EQ(1.0 / 3.0, f1_args.GuestArgument<5>());
  EXPECT_EQ(4, f1_args.GuestArgument<6>());
  EXPECT_EQ(1.0 / 4.0, f1_args.GuestArgument<7>());
  EXPECT_EQ(5, f1_args.GuestArgument<8>());
  EXPECT_EQ(1.0 / 5.0, f1_args.GuestArgument<9>());
  EXPECT_EQ(6, f1_args.GuestArgument<10>());
  EXPECT_EQ(1.0 / 6.0, f1_args.GuestArgument<11>());
  EXPECT_EQ(7, f1_args.GuestArgument<12>());
  EXPECT_EQ(1.0 / 7.0, f1_args.GuestArgument<13>());
  EXPECT_EQ(8, f1_args.GuestArgument<14>());
  EXPECT_EQ(1.0 / 8.0, f1_args.GuestArgument<15>());
  EXPECT_EQ(0x1c71'c71c, f1_args.GuestArgument<16>());
  EXPECT_EQ(1.0 / 10.0, f1_args.GuestArgument<17>());
  EXPECT_EQ(0x745d'1746, f1_args.GuestArgument<18>());
  EXPECT_EQ(1.0 / 12.0, f1_args.GuestArgument<19>());
  EXPECT_EQ(1.0, f1_args.GuestResult());

  GuestArgumentsAndResult<int(double,
                              int,
                              double,
                              int,
                              double,
                              int,
                              double,
                              int,
                              double,
                              int,
                              double,
                              int,
                              double,
                              int,
                              double,
                              int,
                              double,
                              int,
                              double,
                              int)>
      f2_args(&arguments.buffer);
  EXPECT_EQ(1.0, f2_args.GuestArgument<0>());
  EXPECT_EQ(1, f2_args.GuestArgument<1>());
  EXPECT_EQ(1.0 / 2.0, f2_args.GuestArgument<2>());
  EXPECT_EQ(2, f2_args.GuestArgument<3>());
  EXPECT_EQ(1.0 / 3.0, f2_args.GuestArgument<4>());
  EXPECT_EQ(3, f2_args.GuestArgument<5>());
  EXPECT_EQ(1.0 / 4.0, f2_args.GuestArgument<6>());
  EXPECT_EQ(4, f2_args.GuestArgument<7>());
  EXPECT_EQ(1.0 / 5.0, f2_args.GuestArgument<8>());
  EXPECT_EQ(5, f2_args.GuestArgument<9>());
  EXPECT_EQ(1.0 / 6.0, f2_args.GuestArgument<10>());
  EXPECT_EQ(6, f2_args.GuestArgument<11>());
  EXPECT_EQ(1.0 / 7.0, f2_args.GuestArgument<12>());
  EXPECT_EQ(7, f2_args.GuestArgument<13>());
  EXPECT_EQ(1.0 / 8.0, f2_args.GuestArgument<14>());
  EXPECT_EQ(8, f2_args.GuestArgument<15>());
  EXPECT_EQ(1.0 / 9.0, f2_args.GuestArgument<16>());
  EXPECT_EQ(-0x6666'6666, f2_args.GuestArgument<17>());
  EXPECT_EQ(1.0 / 11.0, f2_args.GuestArgument<18>());
  EXPECT_EQ(0x5555'5555, f2_args.GuestArgument<19>());
  EXPECT_EQ(1, f2_args.GuestResult());
}

}  // namespace

}  // namespace berberis
