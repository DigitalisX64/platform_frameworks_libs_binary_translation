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

// region digitalis
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
// endregion
#include <cstdint>
// region digitalis
#include <cstdio>
#include <limits>
#include <thread>

#if defined(__BIONIC__)
extern "C" void malloc_disable();
extern "C" void malloc_enable();
#endif
// endregion

namespace berberis {

namespace {

// region digitalis
TEST(MmapCodeFlagsTest, RejectsFixedMappingsAndZeroSizeOnAllHosts) {
  for (int flags : {MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                    MAP_PRIVATE | MAP_ANONYMOUS | kMmapFixedNoReplace}) {
    EXPECT_EQ(
        MmapImpl({.size = kPageSize, .flags = flags, .berberis_flags = kMmapBerberisCode32Bit}),
        MAP_FAILED);
    EXPECT_EQ(errno, EINVAL);
  }
  EXPECT_EQ(MmapImpl({.size = 0, .berberis_flags = kMmapBerberisCode32Bit}), MAP_FAILED);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_EQ(
      MmapImpl({.size = kPageSize, .berberis_flags = kMmapBerberisCode32Bit | kMmapBerberis32Bit}),
      MAP_FAILED);
  EXPECT_EQ(errno, EINVAL);
}

#if defined(__x86_64__)
class MmapCode32Test : public testing::Test {
 protected:
  static constexpr uintptr_t kBegin = uintptr_t{1} << 31;
  static constexpr uintptr_t kEnd = uintptr_t{1} << 32;

  static void* Allocate(size_t size, void* hint = nullptr) {
    return MmapImpl({.addr = hint, .size = size, .berberis_flags = kMmapBerberisCode32Bit});
  }

  void CheckRange(void* mapping, size_t size) {
    auto address = bit_cast<uintptr_t>(mapping);
    EXPECT_GE(address, kBegin);
    EXPECT_LE(address + AlignUpPageSize(size), kEnd - kPageSize);
  }
};

TEST_F(MmapCode32Test, MapsCodeAbove2GiB) {
  constexpr size_t kSize = 4 * 1024 * 1024;
  void* mapping = Allocate(kSize);
  ASSERT_NE(mapping, MAP_FAILED);
  CheckRange(mapping, kSize);
  *static_cast<uint64_t*>(mapping) = 42;
  EXPECT_EQ(*static_cast<uint64_t*>(mapping), 42u);
  MunmapOrDie(mapping, kSize);
}

TEST_F(MmapCode32Test, OccupiedHintIsNotReplaced) {
  void* occupied = Allocate(kPageSize);
  ASSERT_NE(occupied, MAP_FAILED);
  *static_cast<uint64_t*>(occupied) = 0x123456789abcdef0;
  void* mapping = Allocate(kPageSize, occupied);
  ASSERT_NE(mapping, MAP_FAILED);
  CheckRange(mapping, kPageSize);
  EXPECT_NE(mapping, occupied);
  EXPECT_EQ(*static_cast<uint64_t*>(occupied), 0x123456789abcdef0u);
  MunmapOrDie(mapping, kPageSize);
  MunmapOrDie(occupied, kPageSize);
}

TEST_F(MmapCode32Test, RejectsInvalidSizesAndNativeLowAddressFlag) {
  EXPECT_EQ(Allocate(kEnd - kBegin), MAP_FAILED);
  EXPECT_EQ(errno, ENOMEM);
  EXPECT_EQ(Allocate(std::numeric_limits<size_t>::max()), MAP_FAILED);
  EXPECT_EQ(errno, ENOMEM);
  EXPECT_EQ(MmapImpl({.size = kPageSize,
                      .flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT,
                      .berberis_flags = kMmapBerberisCode32Bit}),
            MAP_FAILED);
  EXPECT_EQ(errno, EINVAL);
}

TEST_F(MmapCode32Test, ExhaustionIsIsolatedAndRetriesNeedNoSyscalls) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  auto run = [] {
    constexpr size_t kSize = 4 * 1024 * 1024;
    size_t count = 0;
    while (count <= 512) {
      void* mapping =
          MmapImpl({.size = kSize, .prot = PROT_NONE, .berberis_flags = kMmapBerberisCode32Bit});
      if (mapping == MAP_FAILED) {
        break;
      }
      uintptr_t start = bit_cast<uintptr_t>(mapping);
      if (start < kBegin || start + kSize > kEnd - kPageSize) {
        _exit(1);
      }
      ++count;
    }
    if (count == 0 || count > 512 || errno != ENOMEM) {
      _exit(2);
    }
    // If an exhausted retry performs any mapping or procfs open, its error
    // changes to EPERM. The code-window allocator must retain ENOMEM instead.
    sock_filter filters[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_mmap, 3, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_openat, 2, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_open, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
    };
    sock_fprog program = {static_cast<unsigned short>(std::size(filters)), filters};
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) ||
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program)) {
      _exit(3);
    }
    for (size_t i = 0; i < 100; ++i) {
      if (Allocate(kSize) != MAP_FAILED || errno != ENOMEM) {
        _exit(4);
      }
    }
    _exit(0);
  };
  ASSERT_EXIT(run(), ::testing::ExitedWithCode(0), "");
  void* mapping = Allocate(kPageSize);
  ASSERT_NE(mapping, MAP_FAILED);
  MunmapOrDie(mapping, kPageSize);
}

TEST_F(MmapCode32Test, ConcurrentAllocationsAreDistinct) {
  constexpr size_t kCount = 16;
  std::array<void*, kCount> mappings{};
  std::array<std::thread, kCount> threads;
  for (size_t i = 0; i < kCount; ++i) {
    threads[i] = std::thread([&, i] { mappings[i] = Allocate(kPageSize); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (size_t i = 0; i < kCount; ++i) {
    ASSERT_NE(mappings[i], MAP_FAILED);
    CheckRange(mappings[i], kPageSize);
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(mappings[i], mappings[j]);
    }
    MunmapOrDie(mappings[i], kPageSize);
  }
}

#if defined(__BIONIC__)
int RawForkWithMallocDisabled(bool scan_maps, void* original) {
  int channel[2];
  if (pipe(channel) != 0) {
    return -1;
  }
  std::atomic<bool> start{false};
  std::atomic<bool> locked{false};
  std::thread holder([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    malloc_disable();
    locked.store(true, std::memory_order_release);
    char value;
    while (read(channel[0], &value, 1) < 0 && errno == EINTR) {
    }
    malloc_enable();
  });
  start.store(true, std::memory_order_release);
  while (!locked.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  // Bypass libc's atfork handlers while another thread holds real malloc locks.
  pid_t child = syscall(SYS_fork);
  if (child == 0) {
    // Runtime initialization may ignore SIGALRM; make the child deadline fatal.
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGALRM);
    if (sigaction(SIGALRM, &action, nullptr) != 0 ||
        sigprocmask(SIG_UNBLOCK, &signals, nullptr) != 0) {
      _exit(5);
    }
    alarm(scan_maps ? 1 : 5);
    if (scan_maps) {
      FILE* maps = fopen("/proc/self/maps", "r");
      if (!maps) {
        _exit(2);
      }
      char line[512];
      if (!fgets(line, sizeof(line), maps)) {
        _exit(3);
      }
      fclose(maps);
      _exit(4);
    }
    void* mapping = MmapImpl({.size = kPageSize, .berberis_flags = kMmapBerberisCode32Bit});
    if (mapping == MAP_FAILED || mapping == original || *static_cast<uint64_t*>(original) != 42) {
      _exit(1);
    }
    *static_cast<uint64_t*>(mapping) = 99;
    _exit(0);
  }
  int status = 0;
  pid_t waited;
  do {
    waited = child > 0 ? waitpid(child, &status, 0) : -1;
  } while (waited < 0 && errno == EINTR);
  while (write(channel[1], "x", 1) < 0 && errno == EINTR) {
  }
  holder.join();
  close(channel[0]);
  close(channel[1]);
  return child > 0 && waited == child ? status : -1;
}
#endif

TEST_F(MmapCode32Test, RawForkAllocationAvoidsTheMallocLocksUsedByMapsScanning) {
#if defined(__BIONIC__)
  void* original = Allocate(kPageSize);
  ASSERT_NE(original, MAP_FAILED);
  *static_cast<uint64_t*>(original) = 42;
  int scan_status = RawForkWithMallocDisabled(true, original);
  int allocation_status = RawForkWithMallocDisabled(false, original);
  EXPECT_EQ(*static_cast<uint64_t*>(original), 42u);
  MunmapOrDie(original, kPageSize);
  // The maps-scanning control must block while the malloc locks are held.
  ASSERT_NE(scan_status, -1);
  ASSERT_TRUE(WIFSIGNALED(scan_status));
  EXPECT_EQ(WTERMSIG(scan_status), SIGALRM);
  ASSERT_NE(allocation_status, -1);
  ASSERT_TRUE(WIFEXITED(allocation_status));
  EXPECT_EQ(WEXITSTATUS(allocation_status), 0);
#else
  GTEST_SKIP() << "Bionic malloc_disable is required to hold the allocator locks";
#endif
}
#endif
// endregion

#if defined(__LP64__)
TEST(MmapTest, kMmapImpl_MmapBerberis32Bit) {
  constexpr size_t k8Mb = 0x1 << 23;
  for (size_t i = 0; i < 100; i++) {
    void* result = MmapImpl({.size = k8Mb,
                             .prot = PROT_READ | PROT_WRITE,
                             .flags = MAP_PRIVATE | MAP_ANONYMOUS,
                             .berberis_flags = kMmapBerberis32Bit});
    ASSERT_NE(result, MAP_FAILED) << "iteration " << i;
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
