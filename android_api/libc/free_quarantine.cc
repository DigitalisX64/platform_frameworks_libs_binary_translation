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

// Bounded free-quarantine for guest frees, via --wrap=free on the host proxy
// libc.
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
// throwing "wstring_convert: from_bytes error"). A zeroed-out chunk header on a
// still-referenced pointer also turns a later benign touch of that chunk into a
// loud Scudo "chunk header is zero" abort — the shape a Qt widget-teardown abort
// in a prebuilt Vulkan tool exhibited before this quarantine existed.
//
// Hold the most recent guest frees in a fixed ring so the chunk (and thus its
// Scudo region) stays live across that window, then release the evicted oldest.
// Bounded memory; each pointer is freed exactly once (just deferred). This brings
// Berberis's free timing closer to real hardware so benign guest UAFs stay benign.

#include <stddef.h>

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
#include <atomic>
#include <cstdint>
#endif

extern "C" void __real_free(void* ptr);

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
namespace {

constexpr size_t kQuarantineSize = 1024;
std::atomic<void*> g_quarantine[kQuarantineSize];
std::atomic<uint64_t> g_quarantine_idx{0};

// Returns the evicted (oldest) pointer to actually free, or nullptr.
void* QuarantineSwap(void* ptr) {
  uint64_t i = g_quarantine_idx.fetch_add(1, std::memory_order_relaxed) % kQuarantineSize;
  return g_quarantine[i].exchange(ptr, std::memory_order_acq_rel);
}

}  // namespace
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64

extern "C" void __wrap_free(void* ptr) {
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  if (ptr == nullptr) {
    __real_free(ptr);
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
