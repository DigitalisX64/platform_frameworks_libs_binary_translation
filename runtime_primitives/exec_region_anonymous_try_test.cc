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

// A failed JIT exec-region allocation must degrade to interpretation, never
// kill the guest app.
//
// ExecRegionAnonymousFactory::Create maps a 4 MiB PROT_EXEC memfd through
// MmapImplOrDie, so ANY allocation failure aborted the process. Under memory
// pressure a media app died mid-playback with
//   mmap(addr=0x0, size=4194304, prot=0x5, flags=0x1, fd=952, offset=0x0)
//   failed: Out of memory
// raised from MmapImplOrDie under TryLiteTranslateAndInstallRegion. The
// interpreter can always run any region, so this is a recoverable condition:
// TryCreate reports failure, CodePool::Add returns kNullHostCodeAddr, and the
// translator installs an interpreted entry.

#include "gtest/gtest.h"

#include <sys/resource.h>

#include <cstddef>

#include "berberis/runtime_primitives/exec_region_anonymous.h"

namespace berberis {

namespace {

TEST(ExecRegionAnonymousTryCreate, SucceedsForNormalSize) {
  ExecRegion region = ExecRegionAnonymousFactory::TryCreate(
      ExecRegionAnonymousFactory::kExecRegionSize);
  ASSERT_NE(region.begin(), nullptr) << "a normal-size request should succeed";
  EXPECT_GE(region.size(), size_t{ExecRegionAnonymousFactory::kExecRegionSize});
  region.Free();
}

// An absurd size cannot be mapped. Create() would abort here; TryCreate must
// return an empty region so the caller can fall back to the interpreter.
TEST(ExecRegionAnonymousTryCreate, ReturnsEmptyInsteadOfAbortingOnFailure) {
  // Far more than any host can map, but still a valid size_t: this fails in
  // ftruncate/mmap rather than by exhausting real memory, so the test is safe
  // to run anywhere.
  constexpr size_t kImpossible = size_t{1} << (sizeof(size_t) * 8 - 4);
  ExecRegion region = ExecRegionAnonymousFactory::TryCreate(kImpossible);
  EXPECT_EQ(region.begin(), nullptr) << "an unsatisfiable request must report failure";
}

}  // namespace

}  // namespace berberis
