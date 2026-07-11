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

#ifndef BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_DIGITALIS_EXTRA_STUBS_H_
#define BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_DIGITALIS_EXTRA_STUBS_H_

// Contract stubs for the three reachable NDK-stable proxy symbols the upstream
// generator leaves as DoBadTrampoline (which aborts). Each returns the API's
// own documented "unavailable" value instead of aborting, so a real app that
// resolves and calls the symbol degrades gracefully rather than crashing.
//
// The bodies live here as `inline` functions so both the per-library trampoline
// translation units (which register them in their KnownTrampoline tables) and
// the host unit test can exercise them with no device-library link. They are
// arm64-guest ABI code (guest CPUState / AAPCS64 return conventions), compiled
// only in the arm64-translation flavor (berberis_arm64_defaults, which defines
// NATIVE_BRIDGE_GUEST_ARCH_ARM64 and __x86_64__ on the host).

#include <cstdint>

#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

// AIBinder_toPlatformBinder(AIBinder*) returns android::sp<IBinder> BY VALUE.
// sp<> is non-trivially-copyable (user copy-ctor/dtor), so AAPCS64 returns it
// via the indirect-result-location register x8: the caller allocates the
// sp<IBinder> buffer and passes its address in x8, the callee constructs the sp
// there, and (per AAPCS64) returns that address in x0. An empty sp<IBinder> is
// a single null pointer. A guest cannot use a host sp<IBinder> anyway, so "no
// platform binder" (null) is the only honest, crash-free contract value.
inline void DoStub_AIBinder_toPlatformBinder(HostCode /*callee*/, ProcessState* state) {
  CPUState& cpu = GetCPUState(*state);
  GuestAddr sret = cpu.x[8];
  if (sret != 0) {
    // sizeof(sp<IBinder>) == sizeof(void*); a zeroed buffer is an empty sp.
    *reinterpret_cast<uint64_t*>(ToHostAddr<void>(sret)) = 0;
  }
  cpu.x[0] = sret;  // AAPCS64: return the indirect-result address in x0.
}

// glGetVkProcAddrNV(const GLchar*) returns a function pointer for the
// GL_NV_draw_vulkan_image interop. GFXStream does not implement that extension,
// so the host entrypoint returns NULL and there is nothing to wrap — the honest
// contract value is NULL ("entrypoint unavailable"), which well-behaved callers
// already handle. Exported by both libGLESv2.so and libGLESv3.so.
inline void DoStub_glGetVkProcAddrNV(HostCode /*callee*/, ProcessState* state) {
  auto&& [ret] = GuestReturnReference<void*(const void*)>(state);
  ret = nullptr;
}

// ANativeWindow_setPerformInterceptor(ANativeWindow*, interceptor, void*) -> void
// is a private/system debug hook whose interceptor callback takes a per-op
// va_list that cannot be forwarded without per-op interpretation. No known app
// calls it. The crash-free contract value is a no-op: no interceptor is
// installed and the window behaves as if none were set. (A full per-op va_list
// dispatcher is a future follow-up if a real app ever needs live interception.)
inline void DoStub_ANativeWindow_setPerformInterceptor(HostCode /*callee*/,
                                                       ProcessState* /*state*/) {
  // Intentional no-op: install no interceptor; return void.
}

}  // namespace berberis

#endif  // BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_DIGITALIS_EXTRA_STUBS_H_
