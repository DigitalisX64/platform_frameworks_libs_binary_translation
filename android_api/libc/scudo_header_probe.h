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

#ifndef BERBERIS_ANDROID_API_LIBC_SCUDO_HEADER_PROBE_H_
#define BERBERIS_ANDROID_API_LIBC_SCUDO_HEADER_PROBE_H_

#include <stdint.h>
#include <string.h>

#include <cstddef>

namespace berberis {

// Result of inspecting the 16-byte Scudo chunk-header window preceding a pointer
// handed to the free/realloc proxy wrappers (free_probe.cc / realloc_probe.cc).
struct ProbeHeader {
  // Whether the header window was actually read. False when ptr is null or
  // within 16 bytes of a page start (see InspectProbeHeader) — in that case the
  // pointer must be forwarded to the real allocator untouched.
  bool peeked;
  // True only when peeked and both header qwords are zero: the pointer was never
  // returned from malloc (a static "shared-null" living in .bss/.rodata).
  bool non_heap;
  uint64_t hdr8;   // [ptr-8],  0 when !peeked
  uint64_t hdr16;  // [ptr-16], 0 when !peeked
};

// Inspect the Scudo chunk-header window before `ptr` without ever faulting.
//
// The free/realloc wrappers detect Qt's static "shared-null" pattern (a pointer
// that was never returned from malloc, whose 16 preceding bytes are all zero) by
// peeking [ptr-16..ptr-1]. That peek is unsafe for one class of real heap
// pointers: GWP-ASan — the platform's sampling allocator (~1/1000 mallocs, in
// ~1/128 processes) — places a guarded allocation flush against a PROT_NONE
// guard page for underflow detection, so its user pointer sits within 16 bytes
// of a page start and [ptr-16] lands in the unmapped guard page. Reading there
// faults (SEGV_ACCERR, "Buffer Underflow") — an intermittent crash for any
// heavy-allocation app, and, because GWP-ASan samples randomly, for any
// translated app at all.
//
// A real Scudo chunk header is never within 16 bytes of a page boundary, so a
// page-adjacent pointer is always a real (GWP-ASan-guarded) heap pointer, never
// a static shared-null: report peeked=false and let the caller forward it. This
// is the single source of truth for that guard, shared by both wrappers so they
// cannot drift (regression-tested in scudo_header_probe_test.cc against a real
// PROT_NONE guard page).
inline ProbeHeader InspectProbeHeader(const void* ptr) {
  ProbeHeader r{/*peeked=*/false, /*non_heap=*/false, /*hdr8=*/0, /*hdr16=*/0};
  if (ptr == nullptr) {
    return r;
  }
  if ((reinterpret_cast<uintptr_t>(ptr) & 0xfffUL) < 16) {
    return r;
  }
  const uint8_t* p = static_cast<const uint8_t*>(ptr);
  memcpy(&r.hdr8, p - 8, sizeof(r.hdr8));
  memcpy(&r.hdr16, p - 16, sizeof(r.hdr16));
  r.peeked = true;
  r.non_heap = (r.hdr8 == 0 && r.hdr16 == 0);
  return r;
}

}  // namespace berberis

#endif  // BERBERIS_ANDROID_API_LIBC_SCUDO_HEADER_PROBE_H_
