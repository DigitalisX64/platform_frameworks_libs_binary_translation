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

#include "berberis/guest_os_primitives/guest_signal.h"

#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_signal_arch.h"

namespace berberis {

size_t GetGuest_MINSIGSTKSZ() {
  // See bionic/libc/kernel/uapi/asm-arm64/asm/signal.h
  // ARM64 MINSIGSTKSZ is 5120 (accounts for NEON/SVE state).
  return 5120;
}

void CheckSigactionRestorer(const Guest_sigaction* guest_sa) {
  // ARM64 sigaction has sa_restorer field.
  // Check that the restorer is the kernel-provided one.
  TRACE("Checking arm64 sa_restorer in guest sigaction");
  if (guest_sa->sa_restorer != 0) {
    TRACE("Guest sigaction has non-zero sa_restorer: 0x%lx",
          static_cast<unsigned long>(guest_sa->sa_restorer));
  }
}

void ResetSigactionRestorer(Guest_sigaction* guest_sa) {
  guest_sa->sa_restorer = 0;
}

}  // namespace berberis
