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

#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"

#include "berberis/base/checks.h"

namespace berberis {

// Invalidate regions overlapping with the range. Could be pretty slow.
void InvalidateGuestRange(GuestAddr start, GuestAddr end) {
  TranslationCache* cache = TranslationCache::GetInstance();
  // Only flush the guest code cache if an already-translated region was
  // actually invalidated. FlushGuestCodeCache forces EVERY guest thread to the
  // dispatcher (its own comment: "really, really, REALLY bad for performance"),
  // so calling it for a range with nothing translated — e.g. every 64-byte line
  // of a large IC IVAU flush over freshly-decrypted, not-yet-translated code
  // (mihoyo anti-tamper) — repeatedly stalls all threads into an ANR. A range
  // with no executable entry cannot leave any thread running stale code, so
  // skipping the flush there is correct.
  if (cache->InvalidateGuestRange(start, end)) {
    // TODO(b/28081995): Specify region to avoid flushing too much.
    FlushGuestCodeCache();
  }
}

}  // namespace berberis
