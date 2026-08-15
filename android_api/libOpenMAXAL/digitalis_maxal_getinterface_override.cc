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

// The upstream "XAObject::GetInterface" trampoline calls the host GetInterface
// and then dispatches on the interface ID to wrap that interface's methods.
// For most IIDs that dispatch is a LOG_ALWAYS_FATAL: upstream marshals only
// XA_IID_ENGINE, XA_IID_PLAY, XA_IID_STREAMINFORMATION, XA_IID_VOLUME,
// XA_IID_VIDEODECODERCAPABILITIES and XA_IID_ANDROIDBUFFERQUEUESOURCE.
//
// The platform OpenMAX AL implementation, however, can genuinely hand out
// more, all on the MediaPlayer object class: XA_IID_SEEK and
// XA_IID_PREFETCHSTATUS (explicit, exposed when requested at
// CreateMediaPlayer) and XA_IID_OBJECT (implicit, pre-realize). An app asking
// for any of those aborts the process today.
// XA_IID_DYNAMICINTERFACEMANAGEMENT sits in the media player's class table as
// implicit, but the implementation ships no init hook for its MPH, so today's
// platform never actually exposes it and GetInterface reports
// feature-unsupported before reaching the dispatch; it is marshalled here
// anyway so a platform that does expose it works instead of aborting.
//
// This file installs a Digitalis replacement for the named trampoline that
// marshals those interfaces and delegates every other IID to the upstream
// trampoline, so upstream behaviour is unchanged wherever it already works.
//
// Note on XA_IID_OBJECT: it needs no method registration. The implementation
// serves every object's XAObjectItf out of one static vtable, so the methods
// were already wrapped when the engine object was wrapped at xaCreateEngine
// time. Upstream relies on exactly that sharing already -- XAEngine's
// CreateMediaPlayer is auto-wrapped and never registers the returned player
// object's methods, yet calls on that object work. So the branch only has to
// avoid the abort. It must NOT re-wrap "XAObject::GetInterface": the override
// registry records the trampoline passed at wrap time as the delegable
// original, so wrapping that name from here would record this function as its
// own original and recurse forever.

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)

#include <utility>

#include "OMXAL/OpenMAXAL.h"

#include "berberis/digitalis_extra_proxy/named_trampoline_override.h"
#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

namespace {

#define REGISTER_TRAMPOLINE(itf_name, func_name) \
  WrapHostFunction((*itf)->func_name, #itf_name "::" #func_name)

#define REGISTER_CUSTOM_TRAMPOLINE(itf_name, func_name)             \
  WrapHostFunctionImpl(reinterpret_cast<void*>((*itf)->func_name),  \
                       DoCustomTrampoline_##itf_name##_##func_name, \
                       #itf_name "::" #func_name)

// Interfaces are just structures listing function pointers, thus are layout-compatible.
typedef XADynamicInterfaceManagementItf Guest_XADynamicInterfaceManagementItf;
typedef XASeekItf Guest_XASeekItf;
typedef XAPrefetchStatusItf Guest_XAPrefetchStatusItf;

// XAresult (*RegisterCallback) (XADynamicInterfaceManagementItf self,
//                               xaDynamicInterfaceManagementCallback callback,
//                               void * pContext);
void DoCustomTrampoline_XADynamicInterfaceManagement_RegisterCallback(HostCode /*callee*/,
                                                                      ProcessState* state) {
  using PFN_callee = decltype(std::declval<XADynamicInterfaceManagementItf_>().RegisterCallback);
  auto [self, guest_callback, callback_context] = GuestParamsValues<PFN_callee>(state);

  // typedef void (XAAPIENTRY * xaDynamicInterfaceManagementCallback) (
  //    XADynamicInterfaceManagementItf caller,
  //    void * pContext,
  //    XAuint32 event,
  //    XAresult result,
  //    const XAInterfaceID iid);
  // The iid the callback receives is a pointer into the implementation's own
  // IID table, identical on both sides -- the same assumption the IID
  // comparisons in GetInterface make -- so it is forwarded unmodified.
  auto host_callback =
      WrapGuestFunction(guest_callback, "XADynamicInterfaceManagement_RegisterCallback-callback");

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (*self)->RegisterCallback(self, host_callback, callback_context);
}

void RegisterXADynamicInterfaceManagementItfMethods(Guest_XADynamicInterfaceManagementItf itf) {
  REGISTER_TRAMPOLINE(XADynamicInterfaceManagement, AddInterface);
  REGISTER_TRAMPOLINE(XADynamicInterfaceManagement, RemoveInterface);
  REGISTER_TRAMPOLINE(XADynamicInterfaceManagement, ResumeInterface);
  REGISTER_CUSTOM_TRAMPOLINE(XADynamicInterfaceManagement, RegisterCallback);
}

void RegisterXASeekItfMethods(Guest_XASeekItf itf) {
  REGISTER_TRAMPOLINE(XASeek, SetPosition);
  REGISTER_TRAMPOLINE(XASeek, SetLoop);
  REGISTER_TRAMPOLINE(XASeek, GetLoop);
}

// XAresult (*RegisterCallback) (XAPrefetchStatusItf self,
//                               xaPrefetchCallback callback,
//                               void * pContext);
void DoCustomTrampoline_XAPrefetchStatus_RegisterCallback(HostCode /*callee*/,
                                                          ProcessState* state) {
  using PFN_callee = decltype(std::declval<XAPrefetchStatusItf_>().RegisterCallback);
  auto [self, guest_callback, callback_context] = GuestParamsValues<PFN_callee>(state);

  // typedef void (XAAPIENTRY * xaPrefetchCallback) (
  //    XAPrefetchStatusItf caller,
  //    void * pContext,
  //    XAuint32 event);
  auto host_callback =
      WrapGuestFunction(guest_callback, "XAPrefetchStatus_RegisterCallback-callback");

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (*self)->RegisterCallback(self, host_callback, callback_context);
}

void RegisterXAPrefetchStatusItfMethods(Guest_XAPrefetchStatusItf itf) {
  REGISTER_TRAMPOLINE(XAPrefetchStatus, GetPrefetchStatus);
  REGISTER_TRAMPOLINE(XAPrefetchStatus, GetFillLevel);
  REGISTER_CUSTOM_TRAMPOLINE(XAPrefetchStatus, RegisterCallback);
  REGISTER_TRAMPOLINE(XAPrefetchStatus, SetCallbackEventsMask);
  REGISTER_TRAMPOLINE(XAPrefetchStatus, GetCallbackEventsMask);
  REGISTER_TRAMPOLINE(XAPrefetchStatus, SetFillUpdatePeriod);
  REGISTER_TRAMPOLINE(XAPrefetchStatus, GetFillUpdatePeriod);
}

constexpr const char* kXAObjectGetInterfaceName = "XAObject::GetInterface";

//  XAresult (*GetInterface) (XAObjectItf self, const XAInterfaceID iid, void * pInterface);
void DoDigitalisTrampoline_XAObject_GetInterface(HostCode callee, ProcessState* state) {
  using PFN_callee = decltype(std::declval<XAObjectItf_>().GetInterface);
  // Reading parameters does not consume them: these are lightweight adapters
  // over the guest state, so the upstream trampoline can still parse them.
  auto [self, iid, interface] = GuestParamsValues<PFN_callee>(state);

  // Note, that iid is not an integer, but a pointer to a structure, and both
  // sides see the implementation's own IID objects -- so, like upstream, IIDs
  // are compared by pointer.
  if (iid != XA_IID_DYNAMICINTERFACEMANAGEMENT && iid != XA_IID_SEEK &&
      iid != XA_IID_PREFETCHSTATUS && iid != XA_IID_OBJECT) {
    TrampolineFunc original = GetDigitalisNamedTrampolineOriginal(kXAObjectGetInterfaceName);
    if (original == nullptr) {
      // Unreachable: the original is recorded when this override is installed
      // in place of it, which necessarily precedes any guest call reaching
      // here. Fail soft rather than jump through a null pointer.
      auto&& [ret] = GuestReturnReference<PFN_callee>(state);
      ret = XA_RESULT_FEATURE_UNSUPPORTED;
      return;
    }
    original(callee, state);
    return;
  }

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (*self)->GetInterface(self, iid, interface);

  if (ret != XA_RESULT_SUCCESS) {
    return;
  }

  // Re-obtaining an interface re-registers its methods, which is a no-op: the
  // translation cache is first-wins per host address.
  if (iid == XA_IID_DYNAMICINTERFACEMANAGEMENT) {
    RegisterXADynamicInterfaceManagementItfMethods(
        *static_cast<Guest_XADynamicInterfaceManagementItf*>(interface));
  } else if (iid == XA_IID_SEEK) {
    RegisterXASeekItfMethods(*static_cast<Guest_XASeekItf*>(interface));
  } else if (iid == XA_IID_PREFETCHSTATUS) {
    RegisterXAPrefetchStatusItfMethods(*static_cast<Guest_XAPrefetchStatusItf*>(interface));
  }
  // XA_IID_OBJECT needs no registration -- see the note at the top of the file.
}

const NamedTrampolineOverride kDigitalisMaxalNamedTrampolineOverrides[] = {
    {kXAObjectGetInterfaceName, DoDigitalisTrampoline_XAObject_GetInterface},
};

// Constructor priority 101 matches the other Digitalis proxy registrations: it
// runs after the early proxy-library prep (priorities 0-100) and before the
// default-priority upstream table constructors. Any time before the first
// guest call into this proxy library would do -- the interfaces here are
// wrapped lazily, long after load.
__attribute__((constructor(101))) void RegisterDigitalisMaxalNamedTrampolineOverrides() {
  RegisterDigitalisNamedTrampolineOverrides(
      kDigitalisMaxalNamedTrampolineOverrides,
      sizeof(kDigitalisMaxalNamedTrampolineOverrides) /
          sizeof(kDigitalisMaxalNamedTrampolineOverrides[0]));
}

}  // namespace

}  // namespace berberis

#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
