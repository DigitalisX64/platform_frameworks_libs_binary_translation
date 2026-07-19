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

// Digitalis-side extra trampolines for libbinder_ndk symbols the upstream proxy
// leaves as DoBadTrampoline. libbinder_ndk has three arm64 DoBadTrampoline
// entries; two are covered here, one is genuinely uncoverable:
//
//   AServiceManager_NotificationRegistration_delete(handle) -> void
//       Opaque host-owned handle; verbatim pointer pass-through under LP64.
//
//   AServiceManager_registerForServiceNotifications(
//       const char* instance, AServiceManager_onRegister onRegister,
//       void* cookie) -> AServiceManager_NotificationRegistration*
//       onRegister is a fixed-signature guest callback the host invokes
//       (possibly on a host-spawned binder thread). We wrap it with
//       WrapGuestFunction, which routes host->guest through RunGuestCall;
//       RunGuestCall -> GetCurrentGuestThread -> AttachCurrentThread
//       auto-attaches a guest thread (with TLS) to any unattached host thread,
//       so delivery on a binder thread is safe. The AIBinder* the host passes
//       to the callback is an opaque host pointer the guest only hands back to
//       (proxied) AIBinder_* functions, so verbatim pass-through is correct —
//       the same convention every other AIBinder* parameter in the table uses.
//
//   _Z25AIBinder_toPlatformBinderP8AIBinder - contract-stubbed: returns a C++
//       android::sp<IBinder> by value (no NDK-stable C signature); internal
//       NDK<->platform-binder interop. A guest cannot use a host sp<IBinder>, so
//       the stub returns an empty (null) sp via the AAPCS64 sret register (x8)
//       instead of aborting. See DoStub_AIBinder_toPlatformBinder in
//       digitalis_extra_stubs.h.
//
// Host functions are reached via the dlsym'd `callee` (not by name) so this
// static lib adds no libbinder_ndk link dependency to libberberis_arm64.so.

#if defined(__x86_64__)

#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_abi/guest_function_wrapper.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/proxy_loader/proxy_library_builder.h"
#include "berberis/runtime_primitives/host_code.h"

#include "digitalis_extra_stubs.h"
#include "register_extra_trampolines.h"

namespace berberis {
namespace {

// AServiceManager_onRegister: void(const char* instance, AIBinder* registered,
// void* cookie). AIBinder* is an opaque handle -> void*.
using OnRegister = void (*)(const char*, void*, void*);
// AServiceManager_registerForServiceNotifications signature, with the opaque
// AServiceManager_NotificationRegistration* return modeled as void*.
using RegisterSig = void*(const char*, OnRegister, void*);

void DoCustomTrampoline_AServiceManager_registerForServiceNotifications(HostCode callee,
                                                                        ProcessState* state) {
  auto [instance, guest_callback, cookie] = GuestParamsValues<RegisterSig>(state);
  OnRegister host_callback =
      ToGuestAddr(guest_callback) == 0
          ? nullptr
          : WrapGuestFunction(guest_callback, "AServiceManager_onRegister");
  auto&& [ret] = GuestReturnReference<RegisterSig>(state);
  ret = reinterpret_cast<RegisterSig*>(const_cast<void*>(callee))(instance, host_callback, cookie);
}

const KnownTrampoline kDigitalisExtraLibbinderNdkTrampolines[] = {
    {"AServiceManager_NotificationRegistration_delete",
     GetTrampolineFunc<auto(void*)->void>(), nullptr},
    {"AServiceManager_registerForServiceNotifications",
     DoCustomTrampoline_AServiceManager_registerForServiceNotifications, nullptr},
    {"_Z25AIBinder_toPlatformBinderP8AIBinder", DoStub_AIBinder_toPlatformBinder, nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libbinder_ndk.so", kDigitalisExtraLibbinderNdkTrampolines)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
