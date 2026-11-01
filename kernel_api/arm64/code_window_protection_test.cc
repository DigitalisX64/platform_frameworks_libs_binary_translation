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

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <thread>

#include "berberis/base/mmap.h"
#include "berberis/kernel_api/sys_mman_emulation.h"

namespace berberis {
namespace {

class CodeWindowProtection : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(GetExecutableMmapFlags(), kMmapBerberisCode32Bit);
    code_ = MmapImpl({.size = kPageSize, .berberis_flags = kMmapBerberisCode32Bit});
    ASSERT_NE(code_, MAP_FAILED);
    *static_cast<uint64_t*>(code_) = kValue;
  }
  void TearDown() override {
    if (code_ != MAP_FAILED) {
      MunmapOrDie(code_, kPageSize);
    }
  }
  void CheckValue() { EXPECT_EQ(*static_cast<uint64_t*>(code_), kValue); }
  static constexpr uint64_t kValue = 0x123456789abcdef0;
  void* code_ = MAP_FAILED;
};

TEST_F(CodeWindowProtection, RejectsFixedReplacementAndUnmapBeforeChangingData) {
  for (int fixed : {MAP_FIXED, kMmapFixedNoReplace}) {
    EXPECT_EQ(
        MmapForGuest(
            code_, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | fixed, -1, 0),
        MAP_FAILED);
    EXPECT_EQ(errno, fixed == kMmapFixedNoReplace ? EEXIST : EACCES);
    CheckValue();
  }
  EXPECT_EQ(MunmapForGuest(code_, kPageSize), -1);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(MprotectForGuest(code_, kPageSize, PROT_NONE), -1);
  EXPECT_EQ(errno, EACCES);
  CheckValue();
}

TEST_F(CodeWindowProtection, GuestAllocatorRetriesAfterNoReplaceCollision) {
  void* available = reinterpret_cast<void*>(0x903fe000);
  void* guest = MAP_FAILED;
  for (void* candidate : {code_, available}) {
    guest = MmapForGuest(candidate,
                         kPageSize,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | kMmapFixedNoReplace,
                         -1,
                         0);
    if (guest != MAP_FAILED || errno != EEXIST) {
      break;
    }
  }
  ASSERT_NE(guest, MAP_FAILED);
  EXPECT_EQ(guest, available);
  *static_cast<uint64_t*>(guest) = 99;
  EXPECT_EQ(*static_cast<uint64_t*>(guest), 99u);
  CheckValue();
  EXPECT_EQ(MunmapForGuest(guest, kPageSize), 0);
}

TEST_F(CodeWindowProtection, RejectsRemapSourcesAndDestinations) {
  void* guest =
      MmapForGuest(nullptr, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(guest, MAP_FAILED);
  *static_cast<uint64_t*>(guest) = 42;
  EXPECT_EQ(MremapForGuest(guest, kPageSize, kPageSize, MREMAP_MAYMOVE | MREMAP_FIXED, code_),
            MAP_FAILED);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(MremapForGuest(code_, kPageSize, kPageSize, MREMAP_MAYMOVE | MREMAP_FIXED, guest),
            MAP_FAILED);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(MremapForGuest(code_, 0, kPageSize, MREMAP_MAYMOVE, nullptr), MAP_FAILED);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(*static_cast<uint64_t*>(guest), 42u);
  CheckValue();
  EXPECT_EQ(MunmapForGuest(guest, kPageSize), 0);
}

TEST_F(CodeWindowProtection, ProtectsUnusedAndReleasedReservations) {
  void* unused =
      MmapImpl({.size = kPageSize, .prot = PROT_NONE, .berberis_flags = kMmapBerberisCode32Bit});
  ASSERT_NE(unused, MAP_FAILED);
  EXPECT_EQ(MmapForGuest(unused,
                         kPageSize,
                         PROT_READ | PROT_WRITE,
                         MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS,
                         -1,
                         0),
            MAP_FAILED);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(MunmapForGuest(unused, kPageSize), -1);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(MprotectForGuest(unused, kPageSize, PROT_READ | PROT_WRITE), -1);
  EXPECT_EQ(errno, EACCES);
  MunmapOrDie(unused, kPageSize);
  EXPECT_TRUE(IntersectsCodeWindow(unused, kPageSize));
  EXPECT_EQ(
      mmap(unused, kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | kMmapFixedNoReplace, -1, 0),
      MAP_FAILED);
  EXPECT_EQ(errno, EEXIST);
  void* hinted =
      MmapForGuest(unused, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(hinted, MAP_FAILED);
  EXPECT_FALSE(IntersectsCodeWindow(hinted, kPageSize));
  EXPECT_EQ(MunmapForGuest(hinted, kPageSize), 0);
}

TEST_F(CodeWindowProtection, ChecksPageRoundingAndOverflow) {
  void* before_owned_range = reinterpret_cast<void*>(0x903ff000);
  EXPECT_FALSE(IntersectsCodeWindow(before_owned_range, kPageSize));
  EXPECT_TRUE(IntersectsCodeWindow(before_owned_range, kPageSize + 1));
  EXPECT_EQ(MunmapForGuest(before_owned_range, kPageSize + 1), -1);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(MunmapForGuest(code_, std::numeric_limits<size_t>::max()), -1);
  EXPECT_EQ(errno, EACCES);
  EXPECT_FALSE(IntersectsCodeWindow(code_, 0));
  CheckValue();
}

TEST_F(CodeWindowProtection, AllowsExistingMappingsAndOrdinaryGuestOperations) {
  // The startup collision belongs to the guest, not to the JIT reservation.
  void* existing = reinterpret_cast<void*>(0x90200000);
  EXPECT_FALSE(IntersectsCodeWindow(existing, kPageSize));
  EXPECT_EQ(MprotectForGuest(existing, kPageSize, PROT_READ | PROT_WRITE), 0);
  void* guest =
      MmapForGuest(nullptr, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(guest, MAP_FAILED);
  *static_cast<uint64_t*>(guest) = 99;
  EXPECT_EQ(MprotectForGuest(guest, kPageSize, PROT_READ), 0);
  EXPECT_EQ(MprotectForGuest(guest, kPageSize, PROT_READ | PROT_WRITE), 0);
  void* grown = MremapForGuest(guest, kPageSize, 2 * kPageSize, MREMAP_MAYMOVE, nullptr);
  ASSERT_NE(grown, MAP_FAILED);
  EXPECT_EQ(*static_cast<uint64_t*>(grown), 99u);
  EXPECT_EQ(MunmapForGuest(grown, 2 * kPageSize), 0);
  CheckValue();
}

TEST_F(CodeWindowProtection, AllowsMovingGrowthPastTheReservationBoundary) {
  void* guest = MmapForGuest(reinterpret_cast<void*>(0x903ff000),
                             kPageSize,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | kMmapFixedNoReplace,
                             -1,
                             0);
  ASSERT_EQ(guest, reinterpret_cast<void*>(0x903ff000));
  *static_cast<uint64_t*>(guest) = 99;
  // Compare the retry error with the same request sent directly to the kernel.
  EXPECT_EQ(mremap(guest, kPageSize, 2 * kPageSize, 0), MAP_FAILED);
  EXPECT_EQ(errno, ENOMEM);
  void* grown = MremapForGuest(guest, kPageSize, 2 * kPageSize, 0, nullptr);
  EXPECT_EQ(grown, MAP_FAILED);
  EXPECT_EQ(errno, ENOMEM);
  const bool can_retry = grown == MAP_FAILED && errno == ENOMEM;
  EXPECT_EQ(*static_cast<uint64_t*>(guest), 99u);
  if (can_retry) {
    grown = MremapForGuest(guest, kPageSize, 2 * kPageSize, MREMAP_MAYMOVE, nullptr);
  }
  ASSERT_NE(grown, MAP_FAILED);
  EXPECT_FALSE(IntersectsCodeWindow(grown, 2 * kPageSize));
  EXPECT_EQ(*static_cast<uint64_t*>(grown), 99u);
  EXPECT_EQ(MunmapForGuest(grown, 2 * kPageSize), 0);
  CheckValue();
}

TEST_F(CodeWindowProtection, ConcurrentGuestRequestsCannotRemoveCodeOwnership) {
  std::atomic<bool> failed{false};
  std::array<std::thread, 4> threads;
  for (auto& thread : threads) {
    thread = std::thread([&] {
      for (int i = 0; i < 100; ++i) {
        if (MunmapForGuest(code_, kPageSize) != -1 || errno != EACCES ||
            MmapForGuest(code_,
                         kPageSize,
                         PROT_READ | PROT_WRITE,
                         MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS,
                         -1,
                         0) != MAP_FAILED ||
            errno != EACCES) {
          failed.store(true);
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_FALSE(failed.load());
  CheckValue();
}

}  // namespace
}  // namespace berberis
