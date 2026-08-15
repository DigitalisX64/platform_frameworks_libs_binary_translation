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

// Working trampolines for the nine OpenSL ES vtable methods the upstream
// libOpenSLES proxy registers as LOG_ALWAYS_FATAL("not implemented: ...")
// stubs. All nine take a guest callback function pointer plus an opaque
// context, so all nine need the same treatment the upstream *working* custom
// trampolines apply (SLPlayItf::RegisterCallback and friends): pull the guest
// callback out of guest state, build a host-callable thunk for it with
// WrapGuestFunction, and hand that thunk to the real host method.
//
// Each callback here has a fixed, NDK-stable signature declared in
// <SLES/OpenSLES.h> / <SLES/OpenSLES_Android.h>, which is exactly what
// WrapGuestFunction requires; the thunk routes host->guest through
// RunGuestCall, whose GetCurrentGuestThread -> AttachCurrentThread attaches a
// guest thread to whatever host audio/callback thread the OpenSL ES
// implementation delivers on. A null guest callback (how apps unregister)
// stays null: WrapGuestFunctionImpl returns nullptr for guest address 0.
//
// This file is compiled into libberberis_proxy_libOpenSLES alongside the
// upstream sles_trampolines.cc, which it must not modify. Interception happens
// by name through the named-trampoline override registry: the constructor
// below registers {name, replacement} pairs, and when upstream later calls
// WrapHostFunctionImpl for a matching name (lazily, when a guest obtains the
// interface via SLObjectItf::GetInterface), the replacement is installed in
// place of the fatal stub.

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)

#include <utility>

#include "SLES/OpenSLES.h"
#include "SLES/OpenSLES_Android.h"

#include "berberis/digitalis_extra_proxy/named_trampoline_override.h"
#include "berberis/guest_abi/guest_function_wrapper.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

namespace {

void DoDigitalisTrampoline_SLAndroidBufferQueueItf_RegisterCallback(HostCode callee,
                                                                    ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLAndroidBufferQueueItf_>().RegisterCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  auto [buffer_queue, guest_callback, context] = GuestParamsValues<PFN_callee>(state);
  slAndroidBufferQueueCallback host_callback =
      WrapGuestFunction(guest_callback, "SLAndroidBufferQueueItf_RegisterCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(buffer_queue, host_callback, context);
}

void DoDigitalisTrampoline_SLAudioIODeviceCapabilitiesItf_RegisterAvailableAudioInputsChangedCallback(
    HostCode callee,
    ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLAudioIODeviceCapabilitiesItf_>()
                                  .RegisterAvailableAudioInputsChangedCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  auto [self, guest_callback, context] = GuestParamsValues<PFN_callee>(state);
  slAvailableAudioInputsChangedCallback host_callback = WrapGuestFunction(
      guest_callback,
      "SLAudioIODeviceCapabilitiesItf_RegisterAvailableAudioInputsChangedCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(self, host_callback, context);
}

void DoDigitalisTrampoline_SLAudioIODeviceCapabilitiesItf_RegisterAvailableAudioOutputsChangedCallback(
    HostCode callee,
    ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLAudioIODeviceCapabilitiesItf_>()
                                  .RegisterAvailableAudioOutputsChangedCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  auto [self, guest_callback, context] = GuestParamsValues<PFN_callee>(state);
  slAvailableAudioOutputsChangedCallback host_callback = WrapGuestFunction(
      guest_callback,
      "SLAudioIODeviceCapabilitiesItf_RegisterAvailableAudioOutputsChangedCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(self, host_callback, context);
}

void DoDigitalisTrampoline_SLAudioIODeviceCapabilitiesItf_RegisterDefaultDeviceIDMapChangedCallback(
    HostCode callee,
    ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLAudioIODeviceCapabilitiesItf_>()
                                  .RegisterDefaultDeviceIDMapChangedCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  auto [self, guest_callback, context] = GuestParamsValues<PFN_callee>(state);
  slDefaultDeviceIDMapChangedCallback host_callback = WrapGuestFunction(
      guest_callback,
      "SLAudioIODeviceCapabilitiesItf_RegisterDefaultDeviceIDMapChangedCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(self, host_callback, context);
}

void DoDigitalisTrampoline_SLDynamicInterfaceManagementItf_RegisterCallback(HostCode callee,
                                                                            ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLDynamicInterfaceManagementItf_>().RegisterCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  auto [self, guest_callback, context] = GuestParamsValues<PFN_callee>(state);
  slDynamicInterfaceManagementCallback host_callback = WrapGuestFunction(
      guest_callback, "SLDynamicInterfaceManagementItf_RegisterCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(self, host_callback, context);
}

void DoDigitalisTrampoline_SLMIDIMessageItf_RegisterMetaEventCallback(HostCode callee,
                                                                      ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLMIDIMessageItf_>().RegisterMetaEventCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  auto [self, guest_callback, context] = GuestParamsValues<PFN_callee>(state);
  slMetaEventCallback host_callback =
      WrapGuestFunction(guest_callback, "SLMIDIMessageItf_RegisterMetaEventCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(self, host_callback, context);
}

void DoDigitalisTrampoline_SLMIDIMessageItf_RegisterMIDIMessageCallback(HostCode callee,
                                                                        ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLMIDIMessageItf_>().RegisterMIDIMessageCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  auto [self, guest_callback, context] = GuestParamsValues<PFN_callee>(state);
  slMIDIMessageCallback host_callback =
      WrapGuestFunction(guest_callback, "SLMIDIMessageItf_RegisterMIDIMessageCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(self, host_callback, context);
}

void DoDigitalisTrampoline_SLOutputMixItf_RegisterDeviceChangeCallback(HostCode callee,
                                                                       ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLOutputMixItf_>().RegisterDeviceChangeCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  auto [self, guest_callback, context] = GuestParamsValues<PFN_callee>(state);
  // The spec names the output-mix device-change callback type
  // slMixDeviceChangeCallback (slDeviceChangeCallback does not exist).
  slMixDeviceChangeCallback host_callback =
      WrapGuestFunction(guest_callback, "SLOutputMixItf_RegisterDeviceChangeCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(self, host_callback, context);
}

void DoDigitalisTrampoline_SLVisualizationItf_RegisterVisualizationCallback(HostCode callee,
                                                                            ProcessState* state) {
  using PFN_callee = decltype(std::declval<SLVisualizationItf_>().RegisterVisualizationCallback);
  PFN_callee callee_function = AsFuncPtr(callee);
  // Unlike the other eight, this method takes a fourth parameter: the rate at
  // which the callback is invoked.
  auto [self, guest_callback, context, rate] = GuestParamsValues<PFN_callee>(state);
  slVisualizationCallback host_callback = WrapGuestFunction(
      guest_callback, "SLVisualizationItf_RegisterVisualizationCallback-callback");
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(self, host_callback, context, rate);
}

const NamedTrampolineOverride kDigitalisSlesCallbackOverrides[] = {
    {"SLAndroidBufferQueueItf::RegisterCallback",
     DoDigitalisTrampoline_SLAndroidBufferQueueItf_RegisterCallback},
    {"SLAudioIODeviceCapabilitiesItf::RegisterAvailableAudioInputsChangedCallback",
     DoDigitalisTrampoline_SLAudioIODeviceCapabilitiesItf_RegisterAvailableAudioInputsChangedCallback},
    {"SLAudioIODeviceCapabilitiesItf::RegisterAvailableAudioOutputsChangedCallback",
     DoDigitalisTrampoline_SLAudioIODeviceCapabilitiesItf_RegisterAvailableAudioOutputsChangedCallback},
    {"SLAudioIODeviceCapabilitiesItf::RegisterDefaultDeviceIDMapChangedCallback",
     DoDigitalisTrampoline_SLAudioIODeviceCapabilitiesItf_RegisterDefaultDeviceIDMapChangedCallback},
    {"SLDynamicInterfaceManagementItf::RegisterCallback",
     DoDigitalisTrampoline_SLDynamicInterfaceManagementItf_RegisterCallback},
    {"SLMIDIMessageItf::RegisterMetaEventCallback",
     DoDigitalisTrampoline_SLMIDIMessageItf_RegisterMetaEventCallback},
    {"SLMIDIMessageItf::RegisterMIDIMessageCallback",
     DoDigitalisTrampoline_SLMIDIMessageItf_RegisterMIDIMessageCallback},
    {"SLOutputMixItf::RegisterDeviceChangeCallback",
     DoDigitalisTrampoline_SLOutputMixItf_RegisterDeviceChangeCallback},
    {"SLVisualizationItf::RegisterVisualizationCallback",
     DoDigitalisTrampoline_SLVisualizationItf_RegisterVisualizationCallback},
};

// Priority 101 matches the other Digitalis proxy-extras registrations: after
// the early proxy-library prep constructors (priorities 0-100), before the
// default-priority upstream ones. Interface method wrapping happens later
// still, at guest GetInterface time, so every override is in place by then.
__attribute__((constructor(101))) void RegisterDigitalisSlesCallbackOverrides() {
  RegisterDigitalisNamedTrampolineOverrides(
      kDigitalisSlesCallbackOverrides,
      sizeof(kDigitalisSlesCallbackOverrides) / sizeof(kDigitalisSlesCallbackOverrides[0]));
}

}  // namespace

}  // namespace berberis

#endif  // defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
