/*
 * Copyright (C) 2018 The Android Open Source Project
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

#include "berberis/base/bit_util.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

namespace {

void SetVfpFloat(ThreadState* state, int index, float v) {
  reinterpret_cast<float*>(state->cpu.d)[index] = v;
}

void SetVfpDouble(ThreadState* state, int index, double v) {
  reinterpret_cast<double*>(state->cpu.d)[index] = v;
}

TEST(Params, IntRes) {
  ThreadState state{};

  auto&& [ret] = GuestReturnReference<int()>(&state);
  auto&& [retf] = GuestReturnReference<int (*)()>(&state);
  auto&& [retv] = GuestReturnReference<int(...)>(&state);
  auto&& [retfv] = GuestReturnReference<int (*)(...)>(&state);

  ret = 123;
  EXPECT_EQ(123u, state.cpu.r[0]);

  retf = 234;
  EXPECT_EQ(234u, state.cpu.r[0]);

  retv = 345;
  EXPECT_EQ(345u, state.cpu.r[0]);

  retfv = 456;
  EXPECT_EQ(456u, state.cpu.r[0]);
}

TEST(Params, SignedCharRes) {
  ThreadState state{};

  state.cpu.r[0] = 0;

  auto&& [ret] = GuestReturnReference<signed char()>(&state);
  auto&& [retf] = GuestReturnReference<signed char (*)()>(&state);
  auto&& [retv] = GuestReturnReference<signed char(...)>(&state);
  auto&& [retfv] = GuestReturnReference<signed char (*)(...)>(&state);

  ret = -1;
  EXPECT_EQ(0xffff'ffffU, state.cpu.r[0]);

  retf = -2;
  EXPECT_EQ(0xffff'fffeU, state.cpu.r[0]);

  retv = -3;
  EXPECT_EQ(0xffff'fffdU, state.cpu.r[0]);

  retfv = -4;
  EXPECT_EQ(0xffff'fffcU, state.cpu.r[0]);
}

TEST(Params, PtrFloatFloatArgs) {
  ThreadState state{};

  static int x;

  state.cpu.r[0] = bit_cast<uint32_t>(&x);
  state.cpu.r[1] = bit_cast<uint32_t>(1.0f);
  state.cpu.r[2] = bit_cast<uint32_t>(-.75f);

  auto [arg1, arg2, arg3] = GuestParamsValues<void(int*, float, float)>(&state);
  auto [arg1f, arg2f, arg3f] = GuestParamsValues<void (*)(int*, float, float)>(&state);
  auto [arg1v, arg2v, arg3v] = GuestParamsValues<void(int*, float, float, ...)>(&state);
  auto [arg1fv, arg2fv, arg3fv] = GuestParamsValues<void (*)(int*, float, float, ...)>(&state);

  EXPECT_EQ(&x, arg1);
  EXPECT_FLOAT_EQ(1.0f, arg2);
  EXPECT_FLOAT_EQ(-.75f, arg3);

  EXPECT_EQ(&x, arg1f);
  EXPECT_FLOAT_EQ(1.0f, arg2f);
  EXPECT_FLOAT_EQ(-.75f, arg3f);

  EXPECT_EQ(&x, arg1v);
  EXPECT_FLOAT_EQ(1.0f, arg2v);
  EXPECT_FLOAT_EQ(-.75f, arg3v);

  EXPECT_EQ(&x, arg1fv);
  EXPECT_FLOAT_EQ(1.0f, arg2fv);
  EXPECT_FLOAT_EQ(-.75f, arg3fv);
}

TEST(Params, PtrFloatFloatArgsVfp) {
  ThreadState state{};

  static int x;

  state.cpu.r[0] = bit_cast<uint32_t>(&x);
  state.cpu.r[1] = bit_cast<uint32_t>(42.0f);
  state.cpu.r[2] = 0xa3d7'0a3d;    // -0.57 - bottom half
  state.cpu.r[3] = 0xbfe2'3d70;    // -0.57 - top half
  SetVfpFloat(&state, 0, 1.0f);    // s0
  SetVfpDouble(&state, 1, -.75f);  // d1

  auto [arg1, arg2, arg3] =
      GuestParamsValues<void(int*, float, double), GuestAbi::kAapcsVfp>(&state);
  auto [arg1f, arg2f, arg3f] =
      GuestParamsValues<void (*)(int*, float, double), GuestAbi::kAapcsVfp>(&state);
  auto [arg1v, arg2v, arg3v] =
      GuestParamsValues<void(int*, float, double, ...), GuestAbi::kAapcsVfp>(&state);
  auto [arg1fv, arg2fv, arg3fv] =
      GuestParamsValues<void (*)(int*, float, double, ...), GuestAbi::kAapcsVfp>(&state);

  EXPECT_EQ(&x, arg1);
  EXPECT_FLOAT_EQ(1.0f, arg2);
  EXPECT_DOUBLE_EQ(-.75, arg3);

  EXPECT_EQ(&x, arg1f);
  EXPECT_FLOAT_EQ(1.0f, arg2f);
  EXPECT_DOUBLE_EQ(-.75, arg3f);

  // “Note: There are no VFP CPRCs in a variadic procedure” ⇦ from AAPCS
  EXPECT_EQ(&x, arg1v);
  EXPECT_FLOAT_EQ(42.0f, arg2v);
  EXPECT_DOUBLE_EQ(-.57, arg3v);

  EXPECT_EQ(&x, arg1fv);
  EXPECT_FLOAT_EQ(42.0f, arg2fv);
  EXPECT_DOUBLE_EQ(-.57, arg3fv);
}

TEST(Params, PtrIntPtrLongLongArgs) {
  ThreadState state{};

  alignas(8) uint64_t stack[4];
  state.cpu.r[13] = bit_cast<uint32_t>(&stack[0]);

  static int x;
  constexpr uint64_t kTestValue64 = 0xffff'0000'ffff'0000ULL;

  state.cpu.r[0] = bit_cast<uint32_t>(&x);
  state.cpu.r[1] = bit_cast<uint32_t>(123);
  state.cpu.r[2] = bit_cast<uint32_t>(&x);
  stack[0] = kTestValue64;

  auto [arg1, arg2, arg3, arg4] = GuestParamsValues<void(int*, int, int*, uint64_t)>(&state);
  auto [arg1f, arg2f, arg3f, arg4f] =
      GuestParamsValues<void (*)(int*, int, int*, uint64_t)>(&state);
  auto [arg1v, arg2v, arg3v, arg4v] =
      GuestParamsValues<void(int*, int, int*, uint64_t, ...)>(&state);
  auto [arg1fv, arg2fv, arg3fv, arg4fv] =
      GuestParamsValues<void (*)(int*, int, int*, uint64_t, ...)>(&state);

  EXPECT_EQ(&x, arg1);
  EXPECT_EQ(123, arg2);
  EXPECT_EQ(&x, arg3);
  EXPECT_EQ(kTestValue64, arg4);

  EXPECT_EQ(&x, arg1f);
  EXPECT_EQ(123, arg2f);
  EXPECT_EQ(&x, arg3f);
  EXPECT_EQ(kTestValue64, arg4f);

  EXPECT_EQ(&x, arg1v);
  EXPECT_EQ(123, arg2v);
  EXPECT_EQ(&x, arg3v);
  EXPECT_EQ(kTestValue64, arg4v);

  EXPECT_EQ(&x, arg1fv);
  EXPECT_EQ(123, arg2fv);
  EXPECT_EQ(&x, arg3fv);
  EXPECT_EQ(kTestValue64, arg4fv);
}

TEST(Params, LongLongArgHugeStructResult) {
  ThreadState state{};

  struct Result {
    uint64_t values[10];
  } result{};

  state.cpu.r[0] = bit_cast<uint32_t>(&result);
  state.cpu.r[2] = 0xbeef;
  state.cpu.r[3] = 0xdead;

  auto [arg] = GuestParamsValues<Result(uint64_t)>(&state);

  EXPECT_EQ(0xdead'0000'beefULL, arg);

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

TEST(GuestVAListParams, PtrFloatFloatArgs) {
  ThreadState state{};

  static int x;

  state.cpu.r[0] = bit_cast<uint32_t>(&x);
  state.cpu.r[1] = bit_cast<uint32_t>(1.0f);
  state.cpu.r[2] = bit_cast<uint32_t>(-.75f);

  GuestVAListParams params = GuestParamsValues<void(...)>(&state);

  EXPECT_EQ(&x, params.GetPointerParam<int>());
  EXPECT_FLOAT_EQ(1.0f, params.GetParam<float>());
  EXPECT_FLOAT_EQ(-.75f, params.GetParam<float>());
}

TEST(GuestVAListParams, PtrIntPtrLongLongArgs) {
  ThreadState state{};

  alignas(8) uint64_t stack[4];
  state.cpu.r[13] = bit_cast<uint32_t>(&stack[0]);

  static int x;
  constexpr uint64_t kTestValue64 = 0xffff'0000'ffff'0000ULL;

  state.cpu.r[0] = bit_cast<uint32_t>(&x);
  state.cpu.r[1] = bit_cast<uint32_t>(123);
  state.cpu.r[2] = bit_cast<uint32_t>(&x);
  stack[0] = kTestValue64;

  GuestVAListParams params = GuestParamsValues<void(...)>(&state);

  EXPECT_EQ(&x, params.GetPointerParam<int>());
  EXPECT_EQ(123, params.GetParam<int>());
  EXPECT_EQ(&x, params.GetPointerParam<int>());
  EXPECT_EQ(kTestValue64, params.GetParam<uint64_t>());
}

}  // namespace

}  // namespace berberis
