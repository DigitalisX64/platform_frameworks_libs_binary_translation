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
    uint32_t
        padding[AlignUp(offsetof(GuestArgumentBuffer, argv), sizeof(uint32_t)) / sizeof(uint32_t) +
                8];
  } arguments = {{.argv = {0x5555'5555}}};
  arguments.buffer.argv[1] = 0x3fd5'5555;
  arguments.buffer.argv[2] = 0x9999'999a;
  arguments.buffer.argv[3] = 0x3fc9'9999;
  arguments.buffer.argv[4] = 0x9249'2492;
  arguments.buffer.argv[5] = 0x3fc2'4924;
  arguments.buffer.argv[6] = 0x745d'1746;
  arguments.buffer.argv[7] = 0x3fb7'45d1;

  GuestArgumentsAndResult<double(int, double, int, double)> f1_args(&arguments.buffer);
  EXPECT_EQ(0x5555'5555, f1_args.GuestArgument<0>());
  EXPECT_EQ(1.0 / 5.0, f1_args.GuestArgument<1>());
  EXPECT_EQ(-0x6db6'db6e, f1_args.GuestArgument<2>());
  EXPECT_EQ(1 / 11.0, f1_args.GuestArgument<3>());
  EXPECT_EQ(1.0 / 3.0, f1_args.GuestResult());

  GuestArgumentsAndResult<int(double, int, double, int)> f2_args(&arguments.buffer);
  EXPECT_EQ(1.0 / 3.0, f2_args.GuestArgument<0>());
  EXPECT_EQ(-0x6666'6666, f2_args.GuestArgument<1>());
  EXPECT_EQ(1.0 / 7.0, f2_args.GuestArgument<2>());
  EXPECT_EQ(0x745d'1746, f2_args.GuestArgument<3>());
  EXPECT_EQ(0x5555'5555, f2_args.GuestResult());
}

TEST(NdkTest, GuestArgumentsAndResultTestAapcsVfp) {
  union {
    GuestArgumentBuffer buffer;
    uint32_t
        padding[AlignUp(offsetof(GuestArgumentBuffer, argv), sizeof(uint32_t)) / sizeof(uint32_t) +
                8];
  } arguments = {{.argv = {0x5555'5555}}};
  arguments.buffer.argv[1] = 0x3fd5'5555;
  arguments.buffer.argv[2] = 0x9999'999a;
  arguments.buffer.argv[3] = 0x3fc9'9999;
  arguments.buffer.argv[4] = 0x9249'2492;
  arguments.buffer.argv[5] = 0x3fc2'4924;
  arguments.buffer.argv[6] = 0x745d'1746;
  arguments.buffer.argv[7] = 0x3fb7'45d1;

  GuestArgumentsAndResult<double(int, double, int, double), GuestAbi::kAapcsVfp> f1_args(
      &arguments.buffer);
  EXPECT_EQ(0x5555'5555, f1_args.GuestArgument<0>());
  EXPECT_DEATH(f1_args.GuestArgument<1>(),
               "berberis/guest_abi/guest_arguments_arch.h:[0-9]*: CHECK failed: false");
  EXPECT_EQ(0x3fd5'5555, f1_args.GuestArgument<2>());
  EXPECT_DEATH(f1_args.GuestArgument<3>(),
               "berberis/guest_abi/guest_arguments_arch.h:[0-9]*: CHECK failed: false");
  EXPECT_DEATH(f1_args.GuestResult(), "");

  GuestArgumentsAndResult<int(double, int, double, int), GuestAbi::kAapcsVfp> f2_args(
      &arguments.buffer);
  EXPECT_DEATH(f2_args.GuestArgument<0>(),
               "berberis/guest_abi/guest_arguments_arch.h:[0-9]*: CHECK failed: false");
  EXPECT_EQ(0x5555'5555, f2_args.GuestArgument<1>());
  EXPECT_DEATH(f2_args.GuestArgument<2>(),
               "berberis/guest_abi/guest_arguments_arch.h:[0-9]*: CHECK failed: false");
  EXPECT_EQ(0x3fd5'5555, f2_args.GuestArgument<3>());
  EXPECT_EQ(0x5555'5555, f2_args.GuestResult());
}

}  // namespace

}  // namespace berberis
