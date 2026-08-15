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

#ifndef BERBERIS_DIGITALIS_EXTRA_PROXY_NAMED_TRAMPOLINE_OVERRIDE_H_
#define BERBERIS_DIGITALIS_EXTRA_PROXY_NAMED_TRAMPOLINE_OVERRIDE_H_

#include <cstddef>

#include "berberis/runtime_primitives/host_function_wrapper_impl.h"

namespace berberis {

// Named-trampoline overrides: a Digitalis seam for replacing an upstream
// custom trampoline that is registered through WrapHostFunctionImpl under a
// unique name (typically a proxied vtable method such as
// "SLAndroidBufferQueueItf::RegisterCallback"). This complements the
// symbol-level extras registry in proxy_library_builder.cc, which can only
// intercept entries of a proxy's primary KnownTrampoline table; vtable-method
// trampolines never appear in those tables — they are wrapped lazily at
// runtime (e.g. when a guest obtains an interface via GetInterface), so the
// wrap-time name is the only stable handle for interception.
//
// Flow: an override file compiled into the owning proxy library registers
// {name, replacement} pairs from a constructor. When upstream code later
// calls WrapHostFunctionImpl with a matching name, the (arm64-only) hook in
// that inline function consults FindDigitalisNamedTrampolineOverride, which
// records the upstream trampoline as the "original" for that name and returns
// the replacement to be installed instead. A replacement that only extends
// upstream behaviour can fetch the original via
// GetDigitalisNamedTrampolineOriginal and delegate to it.
struct NamedTrampolineOverride {
  // Exact name upstream passes to WrapHostFunctionImpl, e.g.
  // "XAObject::GetInterface". Compared with strcmp.
  const char* name;
  // Installed in place of the upstream trampoline for that name.
  TrampolineFunc override_trampoline;
};

// Register overrides. Call from a library constructor; registrations are
// appended under dlopen's single-threaded constructor ordering and must all
// happen before the first guest call into the owning proxy library. The
// `overrides` array must have static storage duration (the registry keeps the
// pointer, not a copy).
void RegisterDigitalisNamedTrampolineOverrides(const NamedTrampolineOverride* overrides,
                                               size_t count);

// The upstream trampoline displaced by the override for `name`, recorded when
// the override was installed. Null until upstream first tries to wrap that
// name — which, for a name reached through the override itself (e.g. a
// GetInterface replacement), cannot happen before the original is recorded.
TrampolineFunc GetDigitalisNamedTrampolineOriginal(const char* name);

// Consulted by the WrapHostFunctionImpl hook (weak-declared there): returns
// the replacement trampoline for `name` after recording `original`, or null
// if no override is registered. Exposed here for host-side unit tests.
TrampolineFunc FindDigitalisNamedTrampolineOverride(const char* name, TrampolineFunc original);

}  // namespace berberis

#endif  // BERBERIS_DIGITALIS_EXTRA_PROXY_NAMED_TRAMPOLINE_OVERRIDE_H_
