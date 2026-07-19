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

// Diagnostic + safety wrapper for host realloc. VkCaps (Vulkan caps viewer)
// aborts inside scudo_realloc / scudo_free with "chunk header is zero",
// which means a guest caller is passing a pointer whose Scudo chunk header
// has been zeroed (Qt's static "shared-null" QArrayData pattern: a .bss
// const that doubles as a default-constructed value and gets put on the
// same realloc/free path as real heap allocations).
//
// In bionic's Scudo (Standalone, AndroidNormalConfig), the packed header is
// 8 bytes immediately preceding the user pointer; the 8 bytes before that
// hold tagged origin metadata.  A real Scudo chunk has a non-zero packed
// header at [-8..-1] (or non-zero origin at [-16..-9]); an all-zero
// 16-byte window means the pointer was never returned from malloc.
//
// Approach: peek the 16-byte window before ptr.  If both qwords are zero,
// treat the input as a non-heap pointer (e.g., Qt shared null) — do a
// fresh zero-initialised allocation of the requested size and return it
// without touching the original storage.  The shared-null instance lives
// in static storage and needs no deallocation; the caller's bookkeeping
// will pick up the new (heap-managed) pointer like any normal realloc.

#include <malloc.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "scudo_header_probe.h"

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
#include <atomic>
#include <sys/syscall.h>
#include <unistd.h>

#include <android/log.h>

#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_state/guest_state.h"
#endif

extern "C" void* __real_realloc(void* ptr, size_t size);

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
namespace {

std::atomic<uint64_t> g_realloc_count{0};

// Takes the already-read header qwords rather than re-peeking [ptr-16]: the
// only caller has them in hand, and re-reading would reintroduce the very
// GWP-ASan guard-page fault the page check in __wrap_realloc exists to avoid
// (mirrors free_probe.cc's LogFreeCall, which is likewise passed the values).
void LogReallocCall(uint64_t n, void* ptr, size_t size, uint64_t hdr16, uint64_t hdr8) {
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
      "realloc_probe #%llu ptr=%p size=%zu lr=0x%llx fp=0x%llx tid=%d "
      "hdr[-16]=0x%016llx hdr[-8]=0x%016llx",
      static_cast<unsigned long long>(n),
      ptr, size,
      static_cast<unsigned long long>(lr),
      static_cast<unsigned long long>(fp),
      tid,
      static_cast<unsigned long long>(hdr16),
      static_cast<unsigned long long>(hdr8));
}

}  // namespace
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64

extern "C" void* __wrap_realloc(void* ptr, size_t size) {
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  uint64_t n = g_realloc_count.fetch_add(1, std::memory_order_relaxed) + 1;
  // Inspect the Scudo chunk-header window before ptr, skipping the peek when ptr
  // is null or within 16 bytes of a page start (a GWP-ASan-guarded heap pointer
  // whose [ptr-16] would fault on the guard page). Shared with free_probe.cc so
  // the guard cannot drift; see scudo_header_probe.h.
  berberis::ProbeHeader h = berberis::InspectProbeHeader(ptr);
  if (h.peeked && h.non_heap) {
    // Load-bearing: surface the all-zero-header symptom that triggers the
    // Scudo abort path (the band-aid actively rerouting a non-heap realloc).
    // Do NOT log ordinary reallocs — a per-call log floods every translated
    // app's logcat at startup.
    LogReallocCall(n, ptr, size, h.hdr16, h.hdr8);
    // Qt shared-null / non-heap pointer.  Returning calloc(size, 1) gives
    // the caller a fresh writable buffer pre-zeroed to the same byte
    // pattern the shared null had, matching the COW semantics Qt expects.
    // size == 0 collapses to malloc(0) / free(ptr) — for non-heap ptr the
    // "free" half is a no-op, so just return NULL.
    if (size == 0) return nullptr;
    return calloc(1, size);
  }
#endif
  return __real_realloc(ptr, size);
}
