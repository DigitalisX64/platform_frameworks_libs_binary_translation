/*
 * Copyright (C) 2025 The Android Open Source Project
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

#include "cstddef"

#include "berberis/base/bit_util.h"
#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/host_function_wrapper_impl.h"

namespace berberis {

namespace {

constexpr size_t kLargeNumber{0x0007'ffff'fff0};
constexpr size_t kMask32Bit{0xffff'ffff};

static_assert((kLargeNumber & kMask32Bit) != kLargeNumber);

TEST(TrampolineFuncGenerator64, OutputIsTruncated) {
  struct Callee {
    static size_t foo() { return kLargeNumber; }
  };

  TrampolineFunc func = GetTrampolineFunc<int(void)>();

  ProcessState state{};

  func(bit_cast<void*>(&Callee::foo), &state);

  EXPECT_EQ(kLargeNumber & kMask32Bit, state.cpu.x[0]);
}

TEST(TrampolineFuncGenerator64, InputIsTruncated) {
  struct Callee {
    static void foo(size_t x) { EXPECT_EQ(kLargeNumber & kMask32Bit, x); }
  };

  TrampolineFunc func = GetTrampolineFunc<void(int)>();

  ProcessState state{};

  state.cpu.x[0] = kLargeNumber;

  EXPECT_NO_FATAL_FAILURE(func(bit_cast<void*>(&Callee::foo), &state));
}

}  // namespace

}  // namespace berberis
