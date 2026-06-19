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

// Diagnostic + safety wrapper for host free.  VkCaps (Vulkan caps viewer)
// aborts inside scudo_deallocate with "chunk header is zero" during
// Qt6Widgets's recursive widget destruction in setupUi.  The aborting
// pointer is consistent with Qt's classic "static shared-null instance"
// pattern: a .bss / .rodata const QXxxData that doubles as a default
// QXxxData value and gets put on the same free() path as real heap
// allocations.  When Scudo sees free(&shared_null), the 16 bytes
// preceding the pointer are all-zero (no chunk header), and Scudo's
// integrity check aborts.
//
// Approach: intercept free via --wrap=free, peek the 16-byte chunk-header
// window at [ptr-16..ptr-1].  If those bytes are all zero, the pointer is
// not Scudo-managed; short-circuit to a no-op (and log loudly) instead of
// calling the real free that would abort.  Otherwise forward unchanged.
//
// This matches the established "Qt shared null" workaround used in other
// non-native runtimes: silently drop frees on .bss/.rodata pointers and
// let the static-storage lifetime take care of the actual "deallocation".
//
// Second purpose: a bounded free-quarantine. Berberis shares one host (Scudo)
// heap between the guest and the translator's own allocations. A guest with a
// benign use-after-free (free a chunk, then briefly read it again — harmless on
// real hardware where the chunk is not recycled in that window) breaks under
// translation: when the freed chunk's Scudo region empties, Scudo releases the
// pages and the lite/heavy translator's bump Arena (MmapPool, plain mmap)
// immediately re-grabs that address and zero-initialises an IR node over the
// still-referenced bytes, so the guest then reads zeros. MapLibre's getAPIBaseUrl
// hits this: it frees its base-URL std::string at one call site and converts it
// via std::wstring_convert at the next, throwing "wstring_convert: from_bytes
// error". Holding the most recent guest frees in a small ring (deferring the real
// free) keeps the chunk and its region live across that window, bringing free
// timing closer to hardware so benign guest UAFs stay benign. See QuarantineSwap.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
#include <atomic>
#include <sys/syscall.h>
#include <unistd.h>

#include <android/log.h>

#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_state/guest_state.h"
#endif

extern "C" void __real_free(void* ptr);

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
namespace {

std::atomic<uint64_t> g_free_count{0};
std::atomic<uint64_t> g_free_skipped{0};

// Bounded free-quarantine.
//
// Guest code occasionally has a *benign* use-after-free: it frees a heap object
// and then briefly reads it again, relying on the (real-hardware) allocator not
// recycling the chunk in that tiny window. Under Berberis the guest heap (host
// Scudo) is shared with the translator's own allocations: when the guest frees a
// string and its Scudo region empties, Scudo releases the pages and the lite/heavy
// translator's bump Arena (MmapPool, plain mmap) immediately re-grabs that address
// and zero-initialises an IR node over the still-referenced bytes. The guest then
// reads zeros and faults (e.g. MapLibre's getAPIBaseUrl frees its base-URL
// std::string at one call site and converts it via wstring_convert at the next,
// throwing "wstring_convert: from_bytes error").
//
// Hold the most recent guest frees in a fixed ring so the chunk (and thus its
// Scudo region) stays live across that window, then release the evicted oldest.
// Bounded memory; each pointer is freed exactly once (just deferred). This brings
// Berberis's free timing closer to real hardware so benign guest UAFs stay benign.
constexpr size_t kQuarantineSize = 1024;
std::atomic<void*> g_quarantine[kQuarantineSize];
std::atomic<uint64_t> g_quarantine_idx{0};

// Returns the evicted (oldest) pointer to actually free, or nullptr.
void* QuarantineSwap(void* ptr) {
  uint64_t i = g_quarantine_idx.fetch_add(1, std::memory_order_relaxed) % kQuarantineSize;
  return g_quarantine[i].exchange(ptr, std::memory_order_acq_rel);
}

void LogFreeCall(const char* tag, uint64_t n, void* ptr,
                 uint64_t hdr16, uint64_t hdr8) {
  uint64_t lr = 0;
  uint64_t fp = 0;
  berberis::GuestThread* gt = berberis::GetCurrentGuestThread();
  if (gt != nullptr && gt->state() != nullptr) {
    lr = gt->state()->cpu.x[30];
    fp = gt->state()->cpu.x[29];
  }
  pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  __android_log_print(
      ANDROID_LOG_ERROR, "berberis",
      "free_probe %s #%llu ptr=%p lr=0x%llx fp=0x%llx tid=%d "
      "hdr[-16]=0x%016llx hdr[-8]=0x%016llx",
      tag,
      static_cast<unsigned long long>(n),
      ptr,
      static_cast<unsigned long long>(lr),
      static_cast<unsigned long long>(fp),
      tid,
      static_cast<unsigned long long>(hdr16),
      static_cast<unsigned long long>(hdr8));
}

}  // namespace
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64

extern "C" void __wrap_free(void* ptr) {
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  if (ptr == nullptr) {
    __real_free(ptr);
    return;
  }

  uint64_t n = g_free_count.fetch_add(1, std::memory_order_relaxed) + 1;

  // GWP-ASan safety: GWP-ASan (the kernel's sampling allocator, ~1/1000 mallocs)
  // places an allocation flush against a guard page for underflow detection, so
  // the user pointer is page-aligned and the 16 bytes at [ptr-16] live in an
  // unmapped guard page. Peeking the header there faults (SEGV_ACCERR, "Buffer
  // Underflow"), an intermittent crash for any heavy-allocation app. A real Scudo
  // chunk's header is never within 16 bytes of a page start, so when ptr is that
  // close to a page boundary, skip the peek and free normally — it is a real
  // (GWP-ASan-guarded) heap pointer, never a static shared-null.
  if ((reinterpret_cast<uintptr_t>(ptr) & 0xfffUL) < 16) {
    void* evicted = QuarantineSwap(ptr);
    if (evicted != nullptr) {
      __real_free(evicted);
    }
    return;
  }

  // Peek the 16-byte window preceding ptr.  A real Scudo chunk has a
  // non-zero packed header at [-8..-1] (Scudo Standalone) plus origin
  // metadata at [-16..-9].  An all-zero window means the caller passed a
  // pointer that was never returned from malloc (e.g., a Qt static shared
  // null living in .bss).
  const uint8_t* p = static_cast<const uint8_t*>(ptr);
  uint64_t hdr8 = 0;
  uint64_t hdr16 = 0;
  memcpy(&hdr8, p - 8, sizeof(hdr8));
  memcpy(&hdr16, p - 16, sizeof(hdr16));

  bool non_heap = (hdr8 == 0 && hdr16 == 0);
  if (non_heap) {
    // Load-bearing: the band-aid is actively dropping a free of a pointer
    // with no Scudo chunk header (a static .bss/.rodata "shared null" on the
    // deallocation path).  Surface that loudly — it is a rare anomaly, not a
    // normal free.  Ordinary heap frees are forwarded silently below; do NOT
    // log them (a per-free log floods every translated app's logcat).
    LogFreeCall("SKIP-non-heap", n, ptr, hdr16, hdr8);
    g_free_skipped.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  // Defer the real free through the quarantine ring so a just-freed chunk (and
  // its Scudo region) stays live long enough that the translator's Arena cannot
  // re-grab and clobber it under a benign guest use-after-free. Release whatever
  // pointer the ring evicts.
  void* evicted = QuarantineSwap(ptr);
  if (evicted != nullptr) {
    __real_free(evicted);
  }
  return;
#endif
  __real_free(ptr);
}
