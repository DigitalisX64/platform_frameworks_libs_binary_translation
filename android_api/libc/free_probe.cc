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
#endif
  __real_free(ptr);
}
