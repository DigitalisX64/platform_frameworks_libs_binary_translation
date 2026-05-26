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

// Diagnostic wrapper for host realloc. VkCaps (Vulkan caps viewer) aborts
// inside scudo_realloc / scudo_free with "chunk header is zero", which means
// a guest caller is passing a pointer whose Scudo chunk header has been
// zeroed (uninitialized chunk, double-free, or pointer not actually returned
// from malloc).  To pin the guest caller, intercept realloc via --wrap=realloc
// and log a window of bytes around the chunk header plus the guest LR/FP/TID
// just before forwarding to the host realloc that may trip the Scudo check.
//
// In bionic's Scudo (Standalone, AndroidNormalConfig), the packed header is
// 8 bytes immediately preceding the user pointer.  The 8 bytes before that
// hold the chunk's tagged origin metadata.  We log both windows so that a
// zero-or-non-zero comparison across multiple calls disambiguates which
// region Scudo's "chunk header" check is actually reading.

#include <malloc.h>
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

extern "C" void* __real_realloc(void* ptr, size_t size);

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
namespace {

std::atomic<uint64_t> g_realloc_count{0};

void LogReallocCall(uint64_t n, void* ptr, size_t size) {
  uint64_t lr = 0;
  uint64_t fp = 0;
  berberis::GuestThread* gt = berberis::GetCurrentGuestThread();
  if (gt != nullptr && gt->state() != nullptr) {
    lr = gt->state()->cpu.x[30];
    fp = gt->state()->cpu.x[29];
  }
  pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  uint64_t hdr8 = 0;
  uint64_t hdr16 = 0;
  if (ptr != nullptr) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    memcpy(&hdr8, p - 8, sizeof(hdr8));
    memcpy(&hdr16, p - 16, sizeof(hdr16));
  }
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
  // First 4 calls log unconditionally so the diagnostic is visible at startup
  // even in healthy runs.  Beyond that, only log if the 16-byte chunk-header
  // window [-16..-1] is all-zero — that's the Scudo "chunk header is zero"
  // condition.  Keeps log volume bounded in production while loudly catching
  // the abort symptom.
  uint64_t n = g_realloc_count.fetch_add(1, std::memory_order_relaxed) + 1;
  bool log = (n <= 4);
  if (!log && ptr != nullptr) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t hdr8 = 0;
    uint64_t hdr16 = 0;
    memcpy(&hdr8, p - 8, sizeof(hdr8));
    memcpy(&hdr16, p - 16, sizeof(hdr16));
    log = (hdr8 == 0 && hdr16 == 0);
  }
  if (log) {
    LogReallocCall(n, ptr, size);
  }
#endif
  return __real_realloc(ptr, size);
}
