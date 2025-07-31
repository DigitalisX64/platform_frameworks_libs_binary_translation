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

#include <cstring>

#include "berberis/base/bit_util.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

namespace {

TEST(Params, PtrIntArgs) {
  ThreadState state{};

  static int x;

  state.cpu.x[0] = ToGuestAddr(&x);
  state.cpu.x[1] = static_cast<uint64_t>(1234u);
  state.cpu.x[2] = static_cast<uint64_t>(-7);

  auto [param1, param2, param3] = GuestParamsValues<void(int*, unsigned int, int)>(&state);
  auto [param1f, param2f, param3f] = GuestParamsValues<void (*)(int*, unsigned int, int)>(&state);
  auto [param1v, param2v, param3v] = GuestParamsValues<void(int*, unsigned int, int, ...)>(&state);
  auto [param1fv, param2fv, param3fv] =
      GuestParamsValues<void (*)(int*, unsigned int, int, ...)>(&state);

  EXPECT_EQ(&x, param1);
  EXPECT_EQ(1234u, param2);
  EXPECT_EQ(-7, param3);

  EXPECT_EQ(&x, param1f);
  EXPECT_EQ(1234u, param2f);
  EXPECT_EQ(-7, param3f);

  EXPECT_EQ(&x, param1v);
  EXPECT_EQ(1234u, param2v);
  EXPECT_EQ(-7, param3v);

  EXPECT_EQ(&x, param1fv);
  EXPECT_EQ(1234u, param2fv);
  EXPECT_EQ(-7, param3fv);
}

TEST(Params, IntRes) {
  ThreadState state{};

  auto&& [ret] = GuestReturnReference<int()>(&state);
  auto&& [retf] = GuestReturnReference<int (*)()>(&state);
  auto&& [retv] = GuestReturnReference<int(...)>(&state);
  auto&& [retfv] = GuestReturnReference<int (*)(...)>(&state);

  ret = 123;
  EXPECT_EQ(123u, state.cpu.x[0]);

  retf = 234;
  EXPECT_EQ(234u, state.cpu.x[0]);

  retv = 345;
  EXPECT_EQ(345u, state.cpu.x[0]);

  retfv = 456;
  EXPECT_EQ(456u, state.cpu.x[0]);
}

TEST(Params, SignedCharRes) {
  ThreadState state{};

  state.cpu.x[0] = 0;

  auto&& [ret] = GuestReturnReference<signed char()>(&state);
  auto&& [retf] = GuestReturnReference<signed char (*)()>(&state);
  auto&& [retv] = GuestReturnReference<signed char(...)>(&state);
  auto&& [retfv] = GuestReturnReference<signed char (*)(...)>(&state);

  ret = -1;
  EXPECT_EQ(0xffU, state.cpu.x[0]);

  retf = -2;
  EXPECT_EQ(0xfeU, state.cpu.x[0]);

  retv = -3;
  EXPECT_EQ(0xfdU, state.cpu.x[0]);

  retfv = -4;
  EXPECT_EQ(0xfcU, state.cpu.x[0]);
}

TEST(Params, PtrRes) {
  ThreadState state{};

  state.cpu.x[0] = static_cast<uint64_t>(42);

  auto&& [ret] = GuestReturnReference<void*()>(&state);

  ret = nullptr;

  EXPECT_EQ(0u, state.cpu.x[0]);
}

TEST(Params, SignedCharArg) {
  ThreadState state{};

  state.cpu.x[0] = 0xf0f0'f0f0'f0f0'f0f0ULL;

  auto [arg] = GuestParamsValues<void(signed char)>(&state);
  auto [argf] = GuestParamsValues<void (*)(signed char)>(&state);
  auto [argv] = GuestParamsValues<void(signed char, ...)>(&state);
  auto [argfv] = GuestParamsValues<void (*)(signed char, ...)>(&state);

  EXPECT_EQ(-16, arg);

  EXPECT_EQ(-16, argf);

  EXPECT_EQ(-16, argv);

  EXPECT_EQ(-16, argfv);
}

TEST(Params, IntFloatIntDoubleArgs) {
  ThreadState state{};

  state.cpu.x[0] = static_cast<uint64_t>(1234u);
  state.cpu.x[1] = static_cast<uint64_t>(-7);
  float f = 2.71f;
  memcpy(state.cpu.v + 0, &f, sizeof(f));
  double d = 3.14;
  memcpy(state.cpu.v + 1, &d, sizeof(d));

  auto [param1, param2, param3, param4] =
      GuestParamsValues<void(unsigned int, float, int, double)>(&state);
  auto [param1f, param2f, param3f, param4f] =
      GuestParamsValues<void (*)(unsigned int, float, int, double)>(&state);
  auto [param1v, param2v, param3v, param4v] =
      GuestParamsValues<void(unsigned int, float, int, double, ...)>(&state);
  auto [param1fv, param2fv, param3fv, param4fv] =
      GuestParamsValues<void (*)(unsigned int, float, int, double, ...)>(&state);

  EXPECT_EQ(1234u, param1);
  EXPECT_FLOAT_EQ(2.71f, param2);
  EXPECT_EQ(-7, param3);
  EXPECT_DOUBLE_EQ(3.14, param4);

  EXPECT_EQ(1234u, param1f);
  EXPECT_FLOAT_EQ(2.71f, param2f);
  EXPECT_EQ(-7, param3f);
  EXPECT_DOUBLE_EQ(3.14, param4f);

  EXPECT_EQ(1234u, param1v);
  EXPECT_FLOAT_EQ(2.71f, param2v);
  EXPECT_EQ(-7, param3v);
  EXPECT_DOUBLE_EQ(3.14, param4v);

  EXPECT_EQ(1234u, param1fv);
  EXPECT_FLOAT_EQ(2.71f, param2fv);
  EXPECT_EQ(-7, param3fv);
  EXPECT_DOUBLE_EQ(3.14, param4fv);
}

TEST(Params, DoubleRes) {
  ThreadState state{};
  double d;

  auto&& [ret] = GuestReturnReference<double()>(&state);
  auto&& [retf] = GuestReturnReference<double (*)()>(&state);
  auto&& [retv] = GuestReturnReference<double(...)>(&state);
  auto&& [retfv] = GuestReturnReference<double (*)(...)>(&state);

  ret = 3.14;
  memcpy(&d, state.cpu.v + 0, sizeof(d));
  EXPECT_DOUBLE_EQ(3.14, d);

  retf = 3.15;
  memcpy(&d, state.cpu.v + 0, sizeof(d));
  EXPECT_DOUBLE_EQ(3.15, d);

  retv = 3.15;
  memcpy(&d, state.cpu.v + 0, sizeof(d));
  EXPECT_DOUBLE_EQ(3.15, d);

  retfv = 3.16;
  memcpy(&d, state.cpu.v + 0, sizeof(d));
  EXPECT_DOUBLE_EQ(3.16, d);
}

TEST(Params, StackArgs) {
  static_assert(sizeof(double) == sizeof(int64_t));
  int64_t stack[8];

  ThreadState state{};
  state.cpu.sp = ToGuestAddr(stack);

  state.cpu.x[0] = 0;
  state.cpu.x[1] = 1;
  state.cpu.x[2] = 2;
  state.cpu.x[3] = 3;
  state.cpu.x[4] = 4;
  state.cpu.x[5] = 5;
  state.cpu.x[6] = 6;
  state.cpu.x[7] = 7;
  stack[0] = 8;
  stack[1] = 9;

  double d;
  d = 0.0;
  memcpy(state.cpu.v + 0, &d, sizeof(d));
  d = 1.1;
  memcpy(state.cpu.v + 1, &d, sizeof(d));
  d = 2.2;
  memcpy(state.cpu.v + 2, &d, sizeof(d));
  d = 3.3;
  memcpy(state.cpu.v + 3, &d, sizeof(d));
  d = 4.4;
  memcpy(state.cpu.v + 4, &d, sizeof(d));
  d = 5.5;
  memcpy(state.cpu.v + 5, &d, sizeof(d));
  d = 6.6;
  memcpy(state.cpu.v + 6, &d, sizeof(d));
  d = 7.7;
  memcpy(state.cpu.v + 7, &d, sizeof(d));
  d = 8.8;
  memcpy(stack + 2, &d, sizeof(d));
  d = 9.9;
  memcpy(stack + 3, &d, sizeof(d));

  auto [param1,
        param2,
        param3,
        param4,
        param5,
        param6,
        param7,
        param8,
        param9,
        param10,
        param11,
        param12,
        param13,
        param14,
        param15,
        param16,
        param17,
        param18,
        param19,
        param20] = GuestParamsValues<void(int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          double,
                                          double,
                                          double,
                                          double,
                                          double,
                                          double,
                                          double,
                                          double,
                                          double,
                                          double)>(&state);
  auto [param1f,
        param2f,
        param3f,
        param4f,
        param5f,
        param6f,
        param7f,
        param8f,
        param9f,
        param10f,
        param11f,
        param12f,
        param13f,
        param14f,
        param15f,
        param16f,
        param17f,
        param18f,
        param19f,
        param20f] = GuestParamsValues<void (*)(int,
                                               int,
                                               int,
                                               int,
                                               int,
                                               int,
                                               int,
                                               int,
                                               int,
                                               int,
                                               double,
                                               double,
                                               double,
                                               double,
                                               double,
                                               double,
                                               double,
                                               double,
                                               double,
                                               double)>(&state);
  auto [param1v,
        param2v,
        param3v,
        param4v,
        param5v,
        param6v,
        param7v,
        param8v,
        param9v,
        param10v,
        param11v,
        param12v,
        param13v,
        param14v,
        param15v,
        param16v,
        param17v,
        param18v,
        param19v,
        param20v] = GuestParamsValues<void(int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           double,
                                           double,
                                           double,
                                           double,
                                           double,
                                           double,
                                           double,
                                           double,
                                           double,
                                           double,
                                           ...)>(&state);
  auto [param1fv,
        param2fv,
        param3fv,
        param4fv,
        param5fv,
        param6fv,
        param7fv,
        param8fv,
        param9fv,
        param10fv,
        param11fv,
        param12fv,
        param13fv,
        param14fv,
        param15fv,
        param16fv,
        param17fv,
        param18fv,
        param19fv,
        param20fv] = GuestParamsValues<void (*)(int,
                                                int,
                                                int,
                                                int,
                                                int,
                                                int,
                                                int,
                                                int,
                                                int,
                                                int,
                                                double,
                                                double,
                                                double,
                                                double,
                                                double,
                                                double,
                                                double,
                                                double,
                                                double,
                                                double,
                                                ...)>(&state);

  EXPECT_EQ(0, param1);
  EXPECT_EQ(1, param2);
  EXPECT_EQ(2, param3);
  EXPECT_EQ(3, param4);
  EXPECT_EQ(4, param5);
  EXPECT_EQ(5, param6);
  EXPECT_EQ(6, param7);
  EXPECT_EQ(7, param8);
  EXPECT_EQ(8, param9);
  EXPECT_EQ(9, param10);

  EXPECT_DOUBLE_EQ(0.0, param11);
  EXPECT_DOUBLE_EQ(1.1, param12);
  EXPECT_DOUBLE_EQ(2.2, param13);
  EXPECT_DOUBLE_EQ(3.3, param14);
  EXPECT_DOUBLE_EQ(4.4, param15);
  EXPECT_DOUBLE_EQ(5.5, param16);
  EXPECT_DOUBLE_EQ(6.6, param17);
  EXPECT_DOUBLE_EQ(7.7, param18);
  EXPECT_DOUBLE_EQ(8.8, param19);
  EXPECT_DOUBLE_EQ(9.9, param20);

  EXPECT_EQ(0, param1f);
  EXPECT_EQ(1, param2f);
  EXPECT_EQ(2, param3f);
  EXPECT_EQ(3, param4f);
  EXPECT_EQ(4, param5f);
  EXPECT_EQ(5, param6f);
  EXPECT_EQ(6, param7f);
  EXPECT_EQ(7, param8f);
  EXPECT_EQ(8, param9f);
  EXPECT_EQ(9, param10f);

  EXPECT_DOUBLE_EQ(0.0, param11f);
  EXPECT_DOUBLE_EQ(1.1, param12f);
  EXPECT_DOUBLE_EQ(2.2, param13f);
  EXPECT_DOUBLE_EQ(3.3, param14f);
  EXPECT_DOUBLE_EQ(4.4, param15f);
  EXPECT_DOUBLE_EQ(5.5, param16f);
  EXPECT_DOUBLE_EQ(6.6, param17f);
  EXPECT_DOUBLE_EQ(7.7, param18f);
  EXPECT_DOUBLE_EQ(8.8, param19f);
  EXPECT_DOUBLE_EQ(9.9, param20f);

  EXPECT_EQ(0, param1v);
  EXPECT_EQ(1, param2v);
  EXPECT_EQ(2, param3v);
  EXPECT_EQ(3, param4v);
  EXPECT_EQ(4, param5v);
  EXPECT_EQ(5, param6v);
  EXPECT_EQ(6, param7v);
  EXPECT_EQ(7, param8v);
  EXPECT_EQ(8, param9v);
  EXPECT_EQ(9, param10v);

  EXPECT_DOUBLE_EQ(0.0, param11v);
  EXPECT_DOUBLE_EQ(1.1, param12v);
  EXPECT_DOUBLE_EQ(2.2, param13v);
  EXPECT_DOUBLE_EQ(3.3, param14v);
  EXPECT_DOUBLE_EQ(4.4, param15v);
  EXPECT_DOUBLE_EQ(5.5, param16v);
  EXPECT_DOUBLE_EQ(6.6, param17v);
  EXPECT_DOUBLE_EQ(7.7, param18v);
  EXPECT_DOUBLE_EQ(8.8, param19v);
  EXPECT_DOUBLE_EQ(9.9, param20v);

  EXPECT_EQ(0, param1fv);
  EXPECT_EQ(1, param2fv);
  EXPECT_EQ(2, param3fv);
  EXPECT_EQ(3, param4fv);
  EXPECT_EQ(4, param5fv);
  EXPECT_EQ(5, param6fv);
  EXPECT_EQ(6, param7fv);
  EXPECT_EQ(7, param8fv);
  EXPECT_EQ(8, param9fv);
  EXPECT_EQ(9, param10fv);

  EXPECT_DOUBLE_EQ(0.0, param11fv);
  EXPECT_DOUBLE_EQ(1.1, param12fv);
  EXPECT_DOUBLE_EQ(2.2, param13fv);
  EXPECT_DOUBLE_EQ(3.3, param14fv);
  EXPECT_DOUBLE_EQ(4.4, param15fv);
  EXPECT_DOUBLE_EQ(5.5, param16fv);
  EXPECT_DOUBLE_EQ(6.6, param17fv);
  EXPECT_DOUBLE_EQ(7.7, param18fv);
  EXPECT_DOUBLE_EQ(8.8, param19fv);
  EXPECT_DOUBLE_EQ(9.9, param20fv);
}

TEST(Params, LongArgHugeStructResult) {
  ThreadState state{};

  struct Result {
    uint64_t values[10];
  } result{};

  state.cpu.x[0] = 0xdead'0000'beef;
  state.cpu.x[8] = bit_cast<uint64_t>(&result);

  auto [arg] = GuestParamsValues<Result(uint64_t)>(&state);

  EXPECT_EQ(0xdead'0000'beefUL, arg);

  auto&& [ret] = GuestReturnReference<Result(uint64_t)>(&state);

  ret = Result{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_EQ(1U, result.values[0]);
  EXPECT_EQ(2U, result.values[1]);
  EXPECT_EQ(3U, result.values[2]);
  EXPECT_EQ(4U, result.values[3]);
  EXPECT_EQ(5U, result.values[4]);
  EXPECT_EQ(6U, result.values[5]);
  EXPECT_EQ(7U, result.values[6]);
  EXPECT_EQ(8U, result.values[7]);
  EXPECT_EQ(9U, result.values[8]);
  EXPECT_EQ(10U, result.values[9]);
}

TEST(GuestVAListParams, PtrIntArgs) {
  ThreadState state{};

  static int x;

  state.cpu.x[0] = ToGuestAddr(&x);
  state.cpu.x[1] = static_cast<uint64_t>(1234u);
  state.cpu.x[2] = static_cast<uint64_t>(-7);

  GuestVAListParams params = GuestParamsValues<void(...)>(&state);

  EXPECT_EQ(&x, params.GetPointerParam<int>());
  EXPECT_EQ(1234u, params.GetParam<unsigned int>());
  EXPECT_EQ(-7, params.GetParam<int>());
}

TEST(GuestVAListParams, IntFloatIntDoubleArgs) {
  ThreadState state{};

  state.cpu.x[0] = static_cast<uint64_t>(1234u);
  state.cpu.x[1] = static_cast<uint64_t>(-7);
  float f = 2.71f;
  memcpy(state.cpu.v + 0, &f, sizeof(f));
  double d = 3.14;
  memcpy(state.cpu.v + 1, &d, sizeof(d));

  GuestVAListParams params = GuestParamsValues<void(...)>(&state);

  EXPECT_EQ(1234u, params.GetParam<unsigned int>());
  EXPECT_FLOAT_EQ(2.71f, params.GetParam<float>());
  EXPECT_EQ(-7, params.GetParam<int>());
  EXPECT_DOUBLE_EQ(3.14, params.GetParam<double>());
}

TEST(GuestVAListParams, StackArgs) {
  static_assert(sizeof(double) == sizeof(int64_t));
  int64_t stack[8];

  ThreadState state{};
  state.cpu.sp = ToGuestAddr(stack);

  state.cpu.x[0] = 0;
  state.cpu.x[1] = 1;
  state.cpu.x[2] = 2;
  state.cpu.x[3] = 3;
  state.cpu.x[4] = 4;
  state.cpu.x[5] = 5;
  state.cpu.x[6] = 6;
  state.cpu.x[7] = 7;
  stack[0] = 8;
  stack[1] = 9;

  double d;
  d = 0.0;
  memcpy(state.cpu.v + 0, &d, sizeof(d));
  d = 1.1;
  memcpy(state.cpu.v + 1, &d, sizeof(d));
  d = 2.2;
  memcpy(state.cpu.v + 2, &d, sizeof(d));
  d = 3.3;
  memcpy(state.cpu.v + 3, &d, sizeof(d));
  d = 4.4;
  memcpy(state.cpu.v + 4, &d, sizeof(d));
  d = 5.5;
  memcpy(state.cpu.v + 5, &d, sizeof(d));
  d = 6.6;
  memcpy(state.cpu.v + 6, &d, sizeof(d));
  d = 7.7;
  memcpy(state.cpu.v + 7, &d, sizeof(d));
  d = 8.8;
  memcpy(stack + 2, &d, sizeof(d));
  d = 9.9;
  memcpy(stack + 3, &d, sizeof(d));

  GuestVAListParams params = GuestParamsValues<void(...)>(&state);

  EXPECT_EQ(0, params.GetParam<int>());
  EXPECT_EQ(1, params.GetParam<int>());
  EXPECT_EQ(2, params.GetParam<int>());
  EXPECT_EQ(3, params.GetParam<int>());
  EXPECT_EQ(4, params.GetParam<int>());
  EXPECT_EQ(5, params.GetParam<int>());
  EXPECT_EQ(6, params.GetParam<int>());
  EXPECT_EQ(7, params.GetParam<int>());
  EXPECT_EQ(8, params.GetParam<int>());
  EXPECT_EQ(9, params.GetParam<int>());

  EXPECT_DOUBLE_EQ(0.0, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(1.1, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(2.2, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(3.3, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(4.4, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(5.5, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(6.6, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(7.7, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(8.8, params.GetParam<double>());
  EXPECT_DOUBLE_EQ(9.9, params.GetParam<double>());
}

}  // namespace

}  // namespace berberis
