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

// Digitalis-side contract stub for ANativeWindow_setPerformInterceptor, which
// the upstream proxy leaves as DoBadTrampoline. It is a private/system debug
// hook whose interceptor callback takes a per-op va_list that cannot be
// forwarded without per-op interpretation; no known app calls it. The stub is a
// crash-free no-op (install no interceptor) instead of aborting. See
// DoStub_ANativeWindow_setPerformInterceptor in digitalis_extra_stubs.h and
// digitalis/docs/proxy-coverage-gaps.md.

#if defined(__x86_64__)

#include "berberis/proxy_loader/proxy_library_builder.h"

#include "digitalis_extra_stubs.h"
#include "register_extra_trampolines.h"

namespace berberis {
namespace {

const KnownTrampoline kDigitalisExtraLibnativewindowTrampolines[] = {
    {"ANativeWindow_setPerformInterceptor", DoStub_ANativeWindow_setPerformInterceptor, nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libnativewindow.so",
                                     kDigitalisExtraLibnativewindowTrampolines)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
