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

#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#include "scudo_header_probe.h"

namespace berberis {

namespace {

// The production guard masks with 0xfff, i.e. it assumes a 4 KiB page. The host
// this test runs on is x86_64 (4 KiB pages), matching the emulator's guest ABI.
constexpr size_t kPageSize = 4096;

// Lays out a PROT_NONE guard page immediately before a readable/writable page —
// exactly how GWP-ASan places an underflow-guarded allocation: the user pointer
// sits at the start of `page`, and the 16 bytes at [ptr-16] fall inside the
// unmapped guard page. Reading there faults; InspectProbeHeader must not.
class ScudoHeaderProbeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    base_ = static_cast<uint8_t*>(mmap(nullptr, 2 * kPageSize,
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(base_, MAP_FAILED);
    guard_ = base_;
    page_ = base_ + kPageSize;
    ASSERT_EQ(mprotect(guard_, kPageSize, PROT_NONE), 0);
  }

  void TearDown() override {
    if (base_ != MAP_FAILED) {
      munmap(base_, 2 * kPageSize);
    }
  }

  uint8_t* base_ = static_cast<uint8_t*>(MAP_FAILED);
  uint8_t* guard_ = nullptr;
  uint8_t* page_ = nullptr;
};

// The regression this whole guard exists for: a pointer whose [ptr-16] lands in
// a PROT_NONE guard page must NOT be peeked. Without the guard, each of these
// InspectProbeHeader calls faults with SEGV_ACCERR ("Buffer Underflow") and the
// test process crashes — which is precisely the intermittent app crash the fix
// eliminated.
TEST_F(ScudoHeaderProbeTest, DoesNotPeekAcrossGuardPage) {
  // ptr exactly at the page start: [ptr-16] is 16 bytes into the guard page.
  EXPECT_FALSE(InspectProbeHeader(page_).peeked);
  // ptr up to 15 bytes past the start: [ptr-16] still reaches into the guard.
  EXPECT_FALSE(InspectProbeHeader(page_ + 1).peeked);
  EXPECT_FALSE(InspectProbeHeader(page_ + 8).peeked);
  EXPECT_FALSE(InspectProbeHeader(page_ + 15).peeked);
}

// 16 bytes past the page start is the exact cutoff: [ptr-16] == page start,
// fully inside the mapped page, so the peek is safe and must happen.
TEST_F(ScudoHeaderProbeTest, PeeksAtSixteenByteBoundary) {
  memset(page_, 0, 32);
  ProbeHeader h = InspectProbeHeader(page_ + 16);
  EXPECT_TRUE(h.peeked);
}

// An all-zero 16-byte header window is a static "shared-null" — never returned
// from malloc — and must be reported as non-heap so the wrapper reroutes it.
TEST_F(ScudoHeaderProbeTest, DetectsAllZeroHeaderAsNonHeap) {
  uint8_t* ptr = page_ + 64;
  memset(ptr - 16, 0, 16);
  ProbeHeader h = InspectProbeHeader(ptr);
  EXPECT_TRUE(h.peeked);
  EXPECT_TRUE(h.non_heap);
  EXPECT_EQ(h.hdr8, 0u);
  EXPECT_EQ(h.hdr16, 0u);
}

// A real Scudo chunk has a non-zero packed header at [ptr-8]; any non-zero byte
// in the 16-byte window means the pointer IS heap-managed — forward untouched.
TEST_F(ScudoHeaderProbeTest, DetectsNonZeroHeaderAsHeap) {
  uint8_t* ptr = page_ + 128;
  memset(ptr - 16, 0, 16);
  const uint64_t kPackedHeader = 0xDEADBEEFCAFEF00DULL;
  memcpy(ptr - 8, &kPackedHeader, sizeof(kPackedHeader));
  ProbeHeader h = InspectProbeHeader(ptr);
  EXPECT_TRUE(h.peeked);
  EXPECT_FALSE(h.non_heap);
  EXPECT_EQ(h.hdr8, kPackedHeader);
}

// Both qwords must be zero for non-heap: a zero packed header ([ptr-8]) with
// non-zero origin metadata ([ptr-16]) is still a real chunk.
TEST_F(ScudoHeaderProbeTest, NonZeroOriginOnlyIsHeap) {
  uint8_t* ptr = page_ + 192;
  memset(ptr - 16, 0, 16);
  const uint64_t kOrigin = 0x0000000000000042ULL;
  memcpy(ptr - 16, &kOrigin, sizeof(kOrigin));
  ProbeHeader h = InspectProbeHeader(ptr);
  EXPECT_TRUE(h.peeked);
  EXPECT_FALSE(h.non_heap);
  EXPECT_EQ(h.hdr16, kOrigin);
  EXPECT_EQ(h.hdr8, 0u);
}

// A null pointer is never peeked (realloc(nullptr) == malloc; free(nullptr) is a
// no-op) — the wrapper forwards it to the real allocator.
TEST_F(ScudoHeaderProbeTest, NullPointerIsNotPeeked) {
  EXPECT_FALSE(InspectProbeHeader(nullptr).peeked);
}

}  // namespace

}  // namespace berberis
