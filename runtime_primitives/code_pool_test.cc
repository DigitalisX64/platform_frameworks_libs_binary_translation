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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

// region digitalis
#include <sys/wait.h>
#include <unistd.h>
// endregion

#include <string_view>

#include "berberis/base/bit_util.h"
// region digitalis
#include "berberis/base/mmap.h"
// endregion
#include "berberis/runtime_primitives/code_pool.h"

namespace berberis {

class MockExecRegionFactory {
 public:
  static void SetImpl(MockExecRegionFactory* impl) { impl_ = impl; }

  static const uint32_t kExecRegionSize;

  // Gmock is not able to mock static methods so we call *Impl counterpart
  // and mock it instead.
  static ExecRegion Create(size_t size) { return impl_->CreateImpl(size); }

  MOCK_METHOD(ExecRegion, CreateImpl, (size_t));

 private:
  static MockExecRegionFactory* impl_;
};

const uint32_t MockExecRegionFactory::kExecRegionSize = sysconf(_SC_PAGESIZE);
MockExecRegionFactory* MockExecRegionFactory::impl_ = nullptr;

namespace {

uint8_t* AllocWritableRegion() {
  return bit_cast<uint8_t*>(MmapImplOrDie({
      .size = MockExecRegionFactory::kExecRegionSize,
      .prot = PROT_READ | PROT_WRITE,
      .flags = MAP_PRIVATE | MAP_ANONYMOUS,
  }));
}

uint8_t* AllocExecutableRegion() {
  return bit_cast<uint8_t*>(MmapImplOrDie({
      .size = MockExecRegionFactory::kExecRegionSize,
      .prot = PROT_NONE,
      .flags = MAP_PRIVATE | MAP_ANONYMOUS,
      .berberis_flags = kMmapBerberis32Bit,
  }));
}

TEST(CodePool, Smoke) {
  MockExecRegionFactory exec_region_factory_mock;
  MockExecRegionFactory::SetImpl(&exec_region_factory_mock);
  auto* first_exec_region_memory_write = AllocWritableRegion();
  auto* first_exec_region_memory_exec = AllocExecutableRegion();
  auto* second_exec_region_memory_write = AllocWritableRegion();
  auto* second_exec_region_memory_exec = AllocExecutableRegion();

  EXPECT_CALL(exec_region_factory_mock, CreateImpl(MockExecRegionFactory::kExecRegionSize))
      .WillOnce([&](size_t) {
        return ExecRegion{first_exec_region_memory_exec,
                          first_exec_region_memory_write,
                          MockExecRegionFactory::kExecRegionSize};
      })
      .WillOnce([&](size_t) {
        return ExecRegion{second_exec_region_memory_exec,
                          second_exec_region_memory_write,
                          MockExecRegionFactory::kExecRegionSize};
      });

  CodePool<MockExecRegionFactory> code_pool;
  {
    MachineCode machine_code;
    constexpr std::string_view kCode = "test1";
    machine_code.AddSequence(kCode.data(), kCode.size());
    auto host_code = code_pool.Add(&machine_code);
    ASSERT_EQ(host_code, AsHostCodeAddr(first_exec_region_memory_exec));
    EXPECT_EQ(std::string_view{bit_cast<const char*>(first_exec_region_memory_write)}, kCode);
  }

  code_pool.ResetExecRegion();

  {
    MachineCode machine_code;
    constexpr std::string_view kCode = "test2";
    machine_code.AddSequence(kCode.data(), kCode.size());
    auto host_code = code_pool.Add(&machine_code);
    ASSERT_EQ(host_code, AsHostCodeAddr(second_exec_region_memory_exec));
    EXPECT_EQ(std::string_view{bit_cast<const char*>(second_exec_region_memory_write)}, kCode);
  }
}

// region digitalis
// Counts Create() calls (no gmock, so it survives fork's copy-on-write static
// state) and hands out a fresh distinct exec/write region pair each time.
class CountingExecRegionFactory {
 public:
  static const uint32_t kExecRegionSize;
  static int create_count;

  static ExecRegion Create(size_t /*size*/) {
    ++create_count;
    auto* exec = bit_cast<uint8_t*>(MmapImplOrDie({
        .size = kExecRegionSize,
        .prot = PROT_NONE,
        .flags = MAP_PRIVATE | MAP_ANONYMOUS,
        .berberis_flags = kMmapBerberis32Bit,
    }));
    auto* write = bit_cast<uint8_t*>(MmapImplOrDie({
        .size = kExecRegionSize,
        .prot = PROT_READ | PROT_WRITE,
        .flags = MAP_PRIVATE | MAP_ANONYMOUS,
    }));
    return ExecRegion{exec, write, kExecRegionSize};
  }
};

const uint32_t CountingExecRegionFactory::kExecRegionSize = sysconf(_SC_PAGESIZE);
int CountingExecRegionFactory::create_count = 0;

// The executable region is a MAP_SHARED memfd that survives fork, so a child
// process must NOT keep translating into the parent's region (siblings would
// corrupt each other's code -- the Chromium-renderer crash this guards against).
// CodePool detects the fork lazily: the first Add() after the pid changes must
// call ResetExecRegion() to obtain a fresh private region.
TEST(CodePool, ForkResetsExecRegion) {
  CountingExecRegionFactory::create_count = 0;
  CodePool<CountingExecRegionFactory> code_pool;  // Create() #1 at construction.
  ASSERT_EQ(CountingExecRegionFactory::create_count, 1);

  // Parent translates once; the region has space, so no reset.
  MachineCode parent_code;
  constexpr std::string_view kParent = "aaaa";
  parent_code.AddSequence(kParent.data(), kParent.size());
  (void)code_pool.Add(&parent_code);
  ASSERT_EQ(CountingExecRegionFactory::create_count, 1);

  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);

  pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    // Child: getpid() now differs from the pool's owner pid, so the first Add()
    // must reset the exec region (Create() #2).
    close(pipefd[0]);
    MachineCode child_code;
    constexpr std::string_view kChild = "bbbb";
    child_code.AddSequence(kChild.data(), kChild.size());
    (void)code_pool.Add(&child_code);
    int count = CountingExecRegionFactory::create_count;
    ssize_t written = write(pipefd[1], &count, sizeof(count));
    _exit((written == sizeof(count) && count == 2) ? 0 : 1);
  }

  close(pipefd[1]);
  int child_count = -1;
  ASSERT_EQ(read(pipefd[0], &child_count, sizeof(child_count)),
            static_cast<ssize_t>(sizeof(child_count)));
  close(pipefd[0]);
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);

  // With the fix the child's first Add() resets -> create_count == 2.
  // Without it the child would reuse the parent's region -> create_count == 1.
  EXPECT_EQ(child_count, 2);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
// endregion

TEST(DataPool, Smoke) {
  DataPool data_pool;
  static uint32_t kConst1 = 0x1234'5678;
  static uint32_t kConst2 = 0x8765'4321;
  uint32_t kVar = kConst2;
  uint32_t* ptr = data_pool.Add(kVar);
  EXPECT_EQ(kConst2, *ptr);
  *ptr = kConst1;
  EXPECT_EQ(kConst1, *ptr);
}

}  // namespace

}  // namespace berberis
