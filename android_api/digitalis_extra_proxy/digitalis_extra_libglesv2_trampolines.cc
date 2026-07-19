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

// Digitalis-side contract stub for glGetVkProcAddrNV, which the upstream proxy
// leaves as DoBadTrampoline in both libGLESv2 and libGLESv3. It returns a
// function pointer for the GL_NV_draw_vulkan_image interop; GFXStream does not
// implement that extension, so the host entrypoint returns NULL and there is
// nothing to wrap. Returning NULL (the documented "unavailable" contract value)
// instead of aborting closes the gap without a GFXStream change. Registered for
// both libGLESv2.so and libGLESv3.so.

#if defined(__x86_64__)

#include "berberis/proxy_loader/proxy_library_builder.h"

#include "digitalis_extra_stubs.h"
#include "register_extra_trampolines.h"

namespace berberis {
namespace {

const KnownTrampoline kDigitalisExtraLibGLESv2Trampolines[] = {
    {"glGetVkProcAddrNV", DoStub_glGetVkProcAddrNV, nullptr},
};
const KnownTrampoline kDigitalisExtraLibGLESv3Trampolines[] = {
    {"glGetVkProcAddrNV", DoStub_glGetVkProcAddrNV, nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libGLESv2.so", kDigitalisExtraLibGLESv2Trampolines)
REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libGLESv3.so", kDigitalisExtraLibGLESv3Trampolines)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
