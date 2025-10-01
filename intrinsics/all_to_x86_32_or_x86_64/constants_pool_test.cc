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

#include "berberis/intrinsics/all_to_x86_32_or_x86_64/constants_pool.h"

namespace berberis {

namespace {

TEST(ConstantsPool, kBsrToClz32_offset) {
  EXPECT_NE(constants_offsets::kBsrToClz<int32_t>, 0);
  EXPECT_NE(constants_pool::Const<uint32_t{63}>::kValue, 0);
  EXPECT_NE(constants_pool::kConst<uint32_t{63}>, 0);
  EXPECT_NE(constants_pool::kBsrToClz<int32_t>, 0);
}

TEST(ConstantsPool, kWidthInBits32_offset) {
  EXPECT_NE(constants_offsets::kWidthInBits<int32_t>, 0);
  EXPECT_NE(constants_pool::Const<uint32_t{32}>::kValue, 0);
  EXPECT_NE(constants_pool::kConst<uint32_t{32}>, 0);
  EXPECT_NE(constants_pool::kWidthInBits<int32_t>, 0);
}

}  // namespace

}  // namespace berberis
