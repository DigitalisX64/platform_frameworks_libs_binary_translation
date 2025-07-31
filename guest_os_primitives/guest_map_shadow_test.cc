/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include <memory>

#include "berberis/base/large_mmap.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

namespace {

class GuestMapShadowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ::testing::Test::SetUp();
    InitLargeMmap();
  }

  template <bool kExpectedExec, size_t kExpectedSize>
  void ExpectExecRegionSize(GuestAddr start, size_t test_size) {
    auto [is_exec, size] = shadow_.GetExecutableRegionSize(start, test_size);
    EXPECT_EQ(is_exec, kExpectedExec);
    EXPECT_EQ(size, kExpectedSize);
  }

  GuestMapShadow shadow_;
};

constexpr GuestAddr kGuestAddr{0x7f01'8000};
constexpr size_t kGuestRegionSize{0x0002'0000};

TEST_F(GuestMapShadowTest, Basic) {
  ASSERT_EQ(kBitUnset, shadow_.GetExecutable(kGuestAddr, kGuestRegionSize));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, 1));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, kGuestRegionSize));

  shadow_.SetExecutable(kGuestAddr, kGuestRegionSize / 2);

  ASSERT_EQ(kBitMixed, shadow_.GetExecutable(kGuestAddr, kGuestRegionSize));
  ASSERT_EQ(kBitSet, shadow_.GetExecutable(kGuestAddr, kGuestRegionSize / 2));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, 1));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, kGuestRegionSize / 2));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, kGuestRegionSize));

  shadow_.SetExecutable(kGuestAddr, kGuestRegionSize);
  ASSERT_EQ(kBitSet, shadow_.GetExecutable(kGuestAddr, kGuestRegionSize));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, 1));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, kGuestRegionSize / 2));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, kGuestRegionSize));

  shadow_.ClearExecutable(kGuestAddr, kGuestRegionSize * 2);
  ASSERT_EQ(kBitUnset, shadow_.GetExecutable(kGuestAddr, kGuestRegionSize));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, 1));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, kGuestRegionSize / 2));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, kGuestRegionSize));
}

TEST_F(GuestMapShadowTest, Remap) {
  constexpr GuestAddr kRemapAddr = 0x0010'7000;
  constexpr size_t kRemapRegionSize1 = kGuestRegionSize / 2;
  constexpr size_t kRemapRegionSize2 = kGuestRegionSize * 2;

  shadow_.SetExecutable(kGuestAddr, kGuestRegionSize);
  ASSERT_EQ(kBitSet, shadow_.GetExecutable(kGuestAddr, kGuestRegionSize));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, 1));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, kGuestRegionSize / 2));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, kGuestRegionSize));

  shadow_.RemapExecutable(kGuestAddr, kGuestRegionSize, kRemapAddr, kRemapRegionSize1);
  ASSERT_EQ(kBitUnset, shadow_.GetExecutable(kGuestAddr, kGuestRegionSize));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, 1));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, kGuestRegionSize / 2));
  ASSERT_TRUE(!shadow_.IsExecutable(kGuestAddr, kGuestRegionSize));

  ASSERT_EQ(kBitSet, shadow_.GetExecutable(kRemapAddr, kRemapRegionSize1));
  ASSERT_TRUE(shadow_.IsExecutable(kRemapAddr, 1));
  ASSERT_TRUE(shadow_.IsExecutable(kRemapAddr, kRemapRegionSize1));

  shadow_.RemapExecutable(kRemapAddr, kRemapRegionSize1, kGuestAddr, kRemapRegionSize2);
  ASSERT_EQ(kBitUnset, shadow_.GetExecutable(kRemapAddr, kRemapRegionSize1));
  ASSERT_TRUE(!shadow_.IsExecutable(kRemapAddr, 1));
  ASSERT_TRUE(!shadow_.IsExecutable(kRemapAddr, kRemapRegionSize1));

  ASSERT_EQ(kBitSet, shadow_.GetExecutable(kGuestAddr, kRemapRegionSize2));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, 1));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, kRemapRegionSize2 / 2));
  ASSERT_TRUE(shadow_.IsExecutable(kGuestAddr, kRemapRegionSize2));
}

TEST_F(GuestMapShadowTest, ProtectedMappings) {
  const char* kStart = ToHostAddr<char>(0x0010'7000);
  const char* kEnd = kStart + kGuestRegionSize;
  const size_t kHalf = kGuestRegionSize / 2;

  shadow_.AddProtectedMapping(kStart, kEnd);

  EXPECT_TRUE(shadow_.IntersectsWithProtectedMapping(kStart, kEnd));

  // Intersecting mappings are also protected.
  EXPECT_TRUE(shadow_.IntersectsWithProtectedMapping(kStart - kHalf, kEnd - kHalf));
  EXPECT_TRUE(shadow_.IntersectsWithProtectedMapping(kStart + kHalf, kEnd + kHalf));

  // Adjacent mappings are not protected.
  EXPECT_FALSE(shadow_.IntersectsWithProtectedMapping(kStart - kGuestRegionSize, kStart));
  EXPECT_FALSE(shadow_.IntersectsWithProtectedMapping(kEnd, kEnd + kGuestRegionSize));

  // Add and test another mapping.

  const char* kAnotherStart = kStart + kGuestRegionSize;
  const char* kAnotherEnd = kAnotherStart + kGuestRegionSize;
  shadow_.AddProtectedMapping(kAnotherStart, kAnotherEnd);

  EXPECT_TRUE(shadow_.IntersectsWithProtectedMapping(kAnotherStart, kAnotherEnd));

  // Intersecting mappings, including those that span across
  // multiple protected mappings, are also protected.
  EXPECT_TRUE(shadow_.IntersectsWithProtectedMapping(kAnotherStart - kHalf, kAnotherEnd - kHalf));
  EXPECT_TRUE(shadow_.IntersectsWithProtectedMapping(kAnotherStart + kHalf, kAnotherEnd + kHalf));
  EXPECT_TRUE(shadow_.IntersectsWithProtectedMapping(kStart - kHalf, kAnotherEnd + kHalf));

  // Adjacent mappings, including between the protected mappings, are not protected.
  EXPECT_FALSE(shadow_.IntersectsWithProtectedMapping(kEnd, kAnotherStart));
  EXPECT_FALSE(shadow_.IntersectsWithProtectedMapping(kAnotherEnd, kAnotherEnd + kGuestRegionSize));
}

#if defined(BERBERIS_GUEST_LP64)

TEST_F(GuestMapShadowTest, 64BitAddress) {
  // We only really allow up to 48 bit addresses.
  constexpr uint64_t k64BitAddr{0x0000'7fff'dddd'ccccULL};

  ASSERT_EQ(kBitUnset, shadow_.GetExecutable(k64BitAddr, kGuestRegionSize));

  shadow_.SetExecutable(k64BitAddr, kGuestRegionSize);

  ASSERT_EQ(kBitSet, shadow_.GetExecutable(k64BitAddr, kGuestRegionSize));
  // The address with 4 upper bits truncated doesn't map to
  // the same entry as the full address (b/369950324).
  constexpr uint64_t kTruncated64BitAddr{k64BitAddr & ~(uint64_t{0xf} << 44)};
  ASSERT_EQ(kBitUnset, shadow_.GetExecutable(kTruncated64BitAddr, kGuestRegionSize));
}

#endif

TEST_F(GuestMapShadowTest, GetExecutableRegionSize) {
  shadow_.SetExecutable(kGuestAddr, kGuestRegionSize);

  ExpectExecRegionSize<false, kGuestRegionSize>(kGuestAddr - kGuestRegionSize, kGuestRegionSize);
  ExpectExecRegionSize<true, kGuestRegionSize>(kGuestAddr, kGuestRegionSize);
  ExpectExecRegionSize<false, kGuestRegionSize>(kGuestAddr + kGuestRegionSize, kGuestRegionSize);

  // Cases where region size is shorter than the tested size.
  ExpectExecRegionSize<false, kGuestRegionSize / 2>(kGuestAddr - kGuestRegionSize / 2,
                                                    kGuestRegionSize);
  ExpectExecRegionSize<true, kGuestRegionSize / 2>(kGuestAddr + kGuestRegionSize / 2,
                                                   kGuestRegionSize);
  ExpectExecRegionSize<true, kGuestRegionSize>(kGuestAddr, kGuestRegionSize * 2);
}

}  // namespace

}  // namespace berberis
