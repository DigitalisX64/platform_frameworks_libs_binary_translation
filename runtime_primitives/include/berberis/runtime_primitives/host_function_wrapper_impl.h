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

#ifndef BERBERIS_RUNTIME_PRIMITIVES_HOST_FUNCTION_WRAPPER_IMPL_H_
#define BERBERIS_RUNTIME_PRIMITIVES_HOST_FUNCTION_WRAPPER_IMPL_H_

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/checks.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

// When guest branches to 'pc', call the host function 'func' via trampoline 'trampoline_func'.
// Trampoline is a helper function invoked as 'trampoline_func(func, process_state)'.
// It extracts guest parameters, applies necessary conversions and calls 'func', then converts
// the return value and writes it back to the guest state.
// 'name' is used for debugging.
using TrampolineFunc = void (*)(HostCode, ThreadState*);

struct NamedTrampolineFunc {
  const char* name;
  TrampolineFunc trampoline;
};

void MakeTrampolineCallable(GuestAddr pc,
                            bool is_host_func,
                            TrampolineFunc trampoline_func,
                            HostCode func,
                            const char* name);

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
// Digitalis named-trampoline override seam: replaces an upstream trampoline
// registered under `name` (typically a proxied vtable method) with a Digitalis
// implementation, recording the upstream one as the delegable "original". See
// berberis/digitalis_extra_proxy/named_trampoline_override.h for the registry.
// Weak: the definition lives in libberberis_digitalis_extra_proxy_arm64
// (whole-linked into libberberis_arm64.so, so proxy libraries resolve it
// dynamically); binaries that inline this header without linking that library
// (e.g. host-test slices) resolve the symbol to null and skip the lookup.
__attribute__((weak)) TrampolineFunc FindDigitalisNamedTrampolineOverride(const char* name,
                                                                          TrampolineFunc original);
#endif
// endregion

inline void WrapHostFunctionImpl(HostCode func, TrampolineFunc trampoline_func, const char* name) {
  // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  if (&FindDigitalisNamedTrampolineOverride != nullptr) {
    if (TrampolineFunc replacement = FindDigitalisNamedTrampolineOverride(name, trampoline_func)) {
      trampoline_func = replacement;
    }
  }
#endif
  // endregion
  MakeTrampolineCallable(ToGuestAddr(func), true, trampoline_func, func, name);
}

void* UnwrapHostFunction(GuestAddr pc);

}  // namespace berberis

#endif  // BERBERIS_RUNTIME_PRIMITIVES_HOST_FUNCTION_WRAPPER_IMPL_H_
