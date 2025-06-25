/*
 * Copyright (C) 2024 The Android Open Source Project
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
#include "berberis/base/mmap.h"

#include <sys/mman.h>

#include <cstdint>

namespace berberis {

namespace {

#if defined(__LP64__)
TEST(MmapTest, kMmapImpl_MmapBerberis32Bit) {
  constexpr size_t k8Mb = 0x1 << 23;
  for (size_t i = 0; i < 100; i++) {
    void* result = MmapImpl({.size = k8Mb,
                             .prot = PROT_READ | PROT_WRITE,
                             .flags = MAP_PRIVATE | MAP_ANONYMOUS,
                             .berberis_flags = kMmapBerberis32Bit});
    ASSERT_NE(result, MAP_FAILED);
    *bit_cast<uint64_t*>(result) = 42;
    ASSERT_EQ(*bit_cast<uint64_t*>(result), 42UL);
  }
}

TEST(MmapTest, MmapImpl_kMmapBerberis32Bit_FailsFor4G) {
  constexpr size_t k4Gb = 0x1L << 32;
  void* result = MmapImpl({.size = k4Gb,
                           .prot = PROT_READ | PROT_WRITE,
                           .flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                           .berberis_flags = kMmapBerberis32Bit});
  ASSERT_EQ(result, MAP_FAILED);
}

TEST(MmapTest, MmapImpl_kMmapBerberis32Bit_FailsFor0) {
  void* result = MmapImpl({.size = 0,
                           .prot = PROT_READ | PROT_WRITE,
                           .flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                           .berberis_flags = kMmapBerberis32Bit});
  ASSERT_EQ(result, MAP_FAILED);
}
#endif

}  // namespace

}  // namespace berberis
