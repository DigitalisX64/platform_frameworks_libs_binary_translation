/*
 * Copyright (C) 2016 The Android Open Source Project
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

#include "berberis/base/mmap.h"

#include <sys/mman.h>

// region digitalis
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <limits>
// endregion
#include <atomic>
// region digitalis
#include <cerrno>
// endregion
#include <cstdint>
#include <cstdlib>
// region digitalis
#include <cstring>
// endregion
#include <random>  // for old versions of GLIBC only (see below)

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"

namespace berberis {

#if defined(__LP64__) && !defined(__x86_64__)
namespace {

// arc4random was introduced in GLIBC 2.36
#if defined(__GLIBC__) && ((__GLIBC__ < 2) || ((__GLIBC__ == 2) && (__GLIBC_MINOR__ < 36)))
uint32_t arc4random_uniform(uint32_t upper_bound) {
  // Fall back to implementation-defined stl random
  static std::random_device random_device("/dev/urandom");
  static std::mt19937 generator(random_device());
  std::uniform_int_distribution<uint32_t> distrib(0, upper_bound);
  return distrib(generator);
}
#endif

void* TryMmap32Bit(MmapImplArgs args) {
  // Outside of x86_64 mapping in the lower 32bit address space
  // is achieved by trying to map at the random 32bit address with
  // hint and then verifying that the resulted map indeed falls in
  // lower 32bit address space. Note that if another mapping already
  // exists "the kernel picks a new address that may or may not
  // depend on the hint." which makes it more difficult.

  constexpr uintptr_t kMinAddress = 0x1'0000;

  // This is always positive hence no sign-extend.
  constexpr uintptr_t kMaxAddress = std::numeric_limits<int32_t>::max();

  // This number is somewhat arbitrary. We want it to be big enough so that it
  // doesn't fail prematurely when 2G space has lower availability, but not too
  // big so it doesn't take forever.
  constexpr size_t kMaxMapAttempts = 512;

  static std::atomic<uintptr_t> saved_hint = 0;
  uintptr_t hint = saved_hint.load();

  uintptr_t arc4_random_upper_bound = kMaxAddress - kMinAddress;

  if (args.size == 0) {
    return MAP_FAILED;
  }

  if (__builtin_usubl_overflow(arc4_random_upper_bound, args.size, &arc4_random_upper_bound)) {
    return MAP_FAILED;
  }
  CHECK_LE(arc4_random_upper_bound, kMaxAddress - kMinAddress);

  if (hint == 0 || hint > (arc4_random_upper_bound + kMinAddress)) {
    hint = arc4random_uniform(static_cast<uint32_t>(arc4_random_upper_bound)) + kMinAddress;
  }

  for (size_t i = 0; i < kMaxMapAttempts; i++) {
    // PROT_NONE, MAP_NORESERVE to make it faster since this may take several attempts.
    // We'll do another mmap() with proper flags on top of this one below.
    void* addr = mmap(bit_cast<void*>(hint),
                      args.size,
                      PROT_NONE,
                      MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE,
                      0,
                      0);
    if (addr == MAP_FAILED) {
      return MAP_FAILED;
    }

    uintptr_t start = bit_cast<uintptr_t>(addr);
    uintptr_t end = start + args.size;

    if (end <= kMaxAddress) {
      saved_hint.store(AlignUpPageSize(end));  // next hint
      return mmap(addr, args.size, args.prot, MAP_FIXED | args.flags, args.fd, args.offset);
    }

    hint = arc4random_uniform(static_cast<uint32_t>(arc4_random_upper_bound)) + kMinAddress;
  }

  saved_hint.store(0);
  return MAP_FAILED;
}

}  // namespace

#endif  // defined(__LP64__) && !defined(__x86_64__)

// region digitalis
#if defined(__x86_64__)
namespace {

// Reservations are established before translated guest threads can fork. The
// ownership ranges remain immutable; only lock-free allocation cursors change.
class CodeWindow {
 public:
  static constexpr uintptr_t kBegin = uintptr_t{1} << 31;
  static constexpr uintptr_t kEnd = uintptr_t{1} << 32;
  static constexpr size_t kChunkSize = 64 * 1024 * 1024;
  // Subdivide occupied chunks down to the normal anonymous code-region size.
  static constexpr size_t kMinChunkSize = 4 * 1024 * 1024;
  static constexpr size_t kMaxRanges = (kEnd - kBegin) / kMinChunkSize;

  CodeWindow() {
    uint64_t random;
    size_t received = 0;
    while (received < sizeof(random)) {
      ssize_t n = syscall(
          SYS_getrandom, reinterpret_cast<char*>(&random) + received, sizeof(random) - received, 0);
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        error_ = n < 0 ? errno : EIO;
        return;
      }
      received += n;
    }

    for (uintptr_t begin = kBegin; begin < kEnd - kPageSize; begin += kChunkSize) {
      Reserve(begin, std::min(begin + kChunkSize, kEnd - kPageSize));
    }
    if (range_count_ == 0) {
      return;
    }

    // Randomize the first page without discarding the preceding capacity.
    size_t first = (random >> 32) % range_count_;
    const Range& range = ranges_[first];
    size_t headroom = std::min(range.end - range.begin, uintptr_t{kMinChunkSize});
    size_t pages = (range.end - range.begin - headroom) / kPageSize + 1;
    uintptr_t pivot = range.begin + (static_cast<uint32_t>(random) % pages) * kPageSize;
    AddAllocationRange(pivot, range.end);
    for (size_t i = 1; i < range_count_; ++i) {
      const Range& next = ranges_[(first + i) % range_count_];
      AddAllocationRange(next.begin, next.end);
    }
    if (pivot != range.begin) {
      AddAllocationRange(range.begin, pivot);
    }
  }

  void* Allocate(size_t size) {
    for (size_t i = 0; i < count_; ++i) {
      uintptr_t next = cursors_[i].load(std::memory_order_relaxed);
      while (size <= ends_[i] - next) {
        if (cursors_[i].compare_exchange_weak(next, next + size, std::memory_order_relaxed)) {
          return bit_cast<void*>(next);
        }
      }
    }
    errno = error_;
    return MAP_FAILED;
  }

  bool Intersects(uintptr_t begin, size_t size) const {
    if (size == 0 || begin >= kEnd) {
      return false;
    }
    uintptr_t end = RangeEnd(begin, size);
    if (end <= kBegin) {
      return false;
    }
    for (size_t i = 0; i < range_count_; ++i) {
      if (begin < ranges_[i].end && ranges_[i].begin < end) {
        return true;
      }
    }
    return false;
  }

  bool Contains(uintptr_t begin, size_t size) const {
    if (size == 0 || begin < kBegin || begin >= kEnd || !IsAlignedPageSize(begin)) {
      return false;
    }
    uintptr_t end = RangeEnd(begin, size);
    for (size_t i = 0; i < range_count_; ++i) {
      if (ranges_[i].begin <= begin && end <= ranges_[i].end) {
        return true;
      }
    }
    return false;
  }

 private:
  struct Range {
    uintptr_t begin;
    uintptr_t end;
  };

  static uintptr_t RangeEnd(uintptr_t begin, size_t size) {
    size_t rounded;
    uintptr_t end;
    if (AlignUpPageSizeOverflow(size, &rounded) || __builtin_add_overflow(begin, rounded, &end)) {
      return std::numeric_limits<uintptr_t>::max();
    }
    return end;
  }

  void Reserve(uintptr_t begin, uintptr_t end) {
    void* reservation = mmap(bit_cast<void*>(begin),
                             end - begin,
                             PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | kMmapFixedNoReplace,
                             -1,
                             0);
    if (reservation == bit_cast<void*>(begin)) {
      if (range_count_ != 0 && ranges_[range_count_ - 1].end == begin) {
        ranges_[range_count_ - 1].end = end;
      } else {
        ranges_[range_count_++] = {begin, end};
      }
      return;
    }
    if (reservation == MAP_FAILED && errno != EEXIST) {
      return;
    }
    if (reservation != MAP_FAILED) {
      // Older kernels can ignore MAP_FIXED_NOREPLACE; never claim a misplaced map.
      MunmapOrDie(reservation, end - begin);
    }
    if (end - begin <= kMinChunkSize) {
      return;
    }
    uintptr_t middle =
        std::max(begin + kMinChunkSize, AlignDown(begin + (end - begin) / 2, kMinChunkSize));
    Reserve(begin, middle);
    Reserve(middle, end);
  }

  void AddAllocationRange(uintptr_t begin, uintptr_t end) {
    cursors_[count_].store(begin, std::memory_order_relaxed);
    ends_[count_++] = end;
  }

  static_assert(std::atomic<uintptr_t>::is_always_lock_free);
  std::array<Range, kMaxRanges> ranges_{};
  size_t range_count_ = 0;
  std::array<std::atomic<uintptr_t>, kMaxRanges + 1> cursors_{};
  std::array<uintptr_t, kMaxRanges + 1> ends_{};
  size_t count_ = 0;
  int error_ = ENOMEM;
};

CodeWindow& GetCodeWindow() {
  static CodeWindow window;
  return window;
}

std::atomic<bool> g_code_window_enabled{false};

void* TryMmapCode32Bit(MmapImplArgs args) {
  size_t size;
  if (AlignUpPageSizeOverflow(args.size, &size) ||
      size > CodeWindow::kEnd - CodeWindow::kBegin - kPageSize) {
    errno = ENOMEM;
    return MAP_FAILED;
  }
  void* address = GetCodeWindow().Allocate(size);
  if (address == MAP_FAILED) {
    return MAP_FAILED;
  }
  if (args.prot == PROT_NONE && args.flags == (MAP_PRIVATE | MAP_ANONYMOUS)) {
    return address;
  }
  return mmap(address, size, args.prot, args.flags | MAP_FIXED, args.fd, args.offset);
}

}  // namespace
#endif  // defined(__x86_64__)

void EnableCodeWindow() {
#if defined(__x86_64__)
  // Complete the local-static initialization guard before guest execution and
  // raw-clone children can reach code allocation.
  (void)GetCodeWindow();
  g_code_window_enabled.store(true, std::memory_order_release);
#endif
}

bool IntersectsCodeWindow(const void* addr, size_t size) {
#if defined(__x86_64__)
  if (g_code_window_enabled.load(std::memory_order_acquire)) {
    return GetCodeWindow().Intersects(bit_cast<uintptr_t>(addr), size);
  }
#else
  UNUSED(addr);
  UNUSED(size);
#endif
  return false;
}

int GetExecutableMmapFlags() {
#if defined(__x86_64__)
  if (g_code_window_enabled.load(std::memory_order_acquire)) {
    return kMmapBerberisCode32Bit;
  }
#endif
  return kMmapBerberis32Bit;
}
// endregion

void* MmapImpl(MmapImplArgs args) {
  // region digitalis
  if ((args.berberis_flags & kMmapBerberisCode32Bit) != 0) {
    int invalid_flags = MAP_FIXED | kMmapFixedNoReplace;
#if defined(__x86_64__)
    invalid_flags |= MAP_32BIT;
#endif
    if (args.size == 0 || (args.flags & invalid_flags) != 0 ||
        (args.berberis_flags & kMmapBerberis32Bit) != 0) {
      errno = EINVAL;
      return MAP_FAILED;
    }
#if defined(__x86_64__)
    return TryMmapCode32Bit(args);
#else
    args.berberis_flags |= kMmapBerberis32Bit;
#endif
  }
  // endregion
  if ((args.berberis_flags & kMmapBerberis32Bit) != 0) {
    // This doesn't make sense for MAP_FIXED
    CHECK_EQ(args.flags & MAP_FIXED, 0);
#if defined(__x86_64__)
    args.flags |= MAP_32BIT;
#elif defined(__LP64__)
    return TryMmap32Bit(args);
#endif
  }
  return mmap(args.addr, args.size, args.prot, args.flags, args.fd, args.offset);
}

void* MmapImplOrDie(MmapImplArgs args) {
  void* ptr = MmapImpl(args);
  // region digitalis - the bare CHECK printed only "0xff..ff != 0xff..ff" on
  // failure, hiding both errno and the arguments; name them so a field abort
  // (EBADF from a swept fd, ENOMEM from a resource ceiling, ...) is
  // self-diagnosing.
  // CHECK_NE(ptr, MAP_FAILED);
  if (ptr == MAP_FAILED) {
    FATAL("mmap(addr=%p, size=%zu, prot=0x%x, flags=0x%x, fd=%d, offset=0x%llx) failed: %s",
          args.addr,
          args.size,
          args.prot,
          args.flags,
          args.fd,
          static_cast<unsigned long long>(args.offset),
          strerror(errno));
  }
  // endregion
  return ptr;
}

void MunmapOrDie(void* ptr, size_t size) {
  // region digitalis
#if defined(__x86_64__)
  if (g_code_window_enabled.load(std::memory_order_acquire) &&
      GetCodeWindow().Contains(bit_cast<uintptr_t>(ptr), size)) {
    // Retain address ownership when executable pages are released, so ordinary
    // guest allocations cannot acquire a hole inside the protected window.
    void* reserved =
        mmap(ptr, size, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    CHECK_EQ(reserved, ptr);
    return;
  }
#endif
  // endregion
  int res = munmap(ptr, size);
  CHECK_EQ(res, 0);
}

void MprotectOrDie(void* ptr, size_t size, int prot) {
  int res = mprotect(ptr, size, prot);
  CHECK_EQ(res, 0);
}

}  // namespace berberis
