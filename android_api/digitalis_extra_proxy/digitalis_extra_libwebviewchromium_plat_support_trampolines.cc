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

// Digitalis-side extra trampolines for libwebviewchromium_plat_support, the
// WebView hardware-accelerated-drawing support library. The upstream proxy
// leaves every symbol as DoBadTrampoline, so a guest hitting any of them aborts
// with `Bad '<sym>' call` (Douyin's libclay.so calls RegisterDrawFunctor
// during WebView init -> abort). This covers the whole C-ABI surface of the
// library (17 of its 18 exported symbols) so those calls forward to the host
// library instead of aborting. The host functions are reached via the dlsym'd
// `callee` (not by name), so this static lib adds no link dependency on
// libwebviewchromium_plat_support to libberberis_arm64.so.
//
// Two marshalling kinds are used:
//   * The three `android::Register{DrawFunctor,DrawGLFunctor,GraphicsUtils}
//     (JNIEnv*)` JNI-registration entry points run jniRegisterNativeMethods on
//     the *host* VM, so they need a valid HOST JNIEnv. Translating the guest
//     JNIEnv via ToHostJNIEnv does NOT work here: these are called from
//     guest-spawned worker threads (Douyin's "lynx-card-servi" thread via
//     libclay.so) that were never attached to the host VM, so the guest JNIEnv
//     has no host mapping and the host function would deref a null env and
//     SIGSEGV at its first `*env` (confirmed by disassembly). Instead the custom
//     trampoline fetches a host env for the current thread from the host VM
//     (GetEnv, attaching the thread if detached and detaching afterwards), and
//     forwards. Each returns the jniRegisterNativeMethods result (0 == ok); once
//     registered the draw path runs host-side and never re-enters a guest
//     DoBadTrampoline. If no host VM/env is available the trampoline returns a
//     failure code rather than crashing, so the caller can fall back.
//   * `android::GraphicBufferImpl` (the gralloc-backed buffer the draw functor
//     maps) and `android::RaiseFileNumberLimit()` have flat C-ABI signatures
//     (int / long / void* / void** / the AwMapMode enum — all LP64-identical
//     between the arm64 guest and x86_64 host), so they use the auto-generated
//     GetTrampolineFunc marshaller with void* for every pointer and int for the
//     enum. The buffer ids and CPU mappings these return live in the shared
//     guest/host address space, so they are valid for the guest to use directly.
//
// JNI_OnLoad is intentionally NOT covered here: it is the library's
// JavaVM*-taking entry point with dedicated native-bridge handling, and the
// observed path is the app calling the individual Register* symbols directly.

#if defined(__x86_64__)

#include <jni.h>

#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/jni/jni_trampolines.h"
#include "berberis/proxy_loader/proxy_library_builder.h"
#include "berberis/runtime_primitives/host_code.h"

#include "register_extra_trampolines.h"

namespace berberis {
namespace {

// jint Register{DrawFunctor,DrawGLFunctor,GraphicsUtils}(JNIEnv*): each calls
// jniRegisterNativeMethods and returns its result (0 on success; a negative
// value is treated as failure by the caller). The guest JNIEnv* is ignored; a
// host env is fetched from the host VM (see the trampoline below).
using Sig_RegisterWebView = jint(JNIEnv*);

// HostCode is `const void*`; a function pointer cannot carry the const, so drop
// it before reinterpreting. `callee` is the host libwebviewchromium_plat_support
// function the proxy dlsym'd at registration.
template <typename Sig>
Sig* HostFn(HostCode callee) {
  return reinterpret_cast<Sig*>(const_cast<void*>(callee));
}

// All three registration symbols share the same jint(JNIEnv*) signature, so one
// trampoline serves them; the KnownTrampoline table binds the correct per-symbol
// `callee`. The guest JNIEnv* argument is consumed but not used (see file header):
// these run jniRegisterNativeMethods on the host VM, so they need a host env for
// the current thread, obtained from the host VM rather than translated.
void DoCustomTrampoline_RegisterWebView(HostCode callee, ProcessState* state) {
  auto [guest_env] = GuestParamsValues<Sig_RegisterWebView>(state);
  (void)guest_env;  // Unused: a host env for this thread is obtained from the host VM below.
  auto&& [ret] = GuestReturnReference<Sig_RegisterWebView>(state);

  JavaVM* host_vm = GetHostJavaVM();
  JNIEnv* host_env = nullptr;
  bool attached = false;
  if (host_vm != nullptr) {
    if (host_vm->GetEnv(reinterpret_cast<void**>(&host_env), JNI_VERSION_1_6) != JNI_OK ||
        host_env == nullptr) {
      // Guest-spawned worker thread not attached to the host VM: attach it so the
      // host RegisterNatives can run, and detach afterwards (an attached thread
      // that exits without detaching is fatal under ART).
      host_env = nullptr;
      attached = host_vm->AttachCurrentThread(&host_env, nullptr) == JNI_OK && host_env != nullptr;
    }
  }
  if (host_env == nullptr) {
    ret = -1;  // No host env available; signal failure so the caller can fall back.
    return;
  }
  ret = HostFn<Sig_RegisterWebView>(callee)(host_env);
  if (attached) {
    host_vm->DetachCurrentThread();
  }
}

const KnownTrampoline kDigitalisExtraLibwebviewchromiumPlatSupportTrampolines[] = {
    // JNI registration entry points (translate JNIEnv*).
    {"_ZN7android19RegisterDrawFunctorEP7_JNIEnv", DoCustomTrampoline_RegisterWebView, nullptr},
    {"_ZN7android21RegisterDrawGLFunctorEP7_JNIEnv", DoCustomTrampoline_RegisterWebView, nullptr},
    {"_ZN7android21RegisterGraphicsUtilsEP7_JNIEnv", DoCustomTrampoline_RegisterWebView, nullptr},

    // android::RaiseFileNumberLimit().
    {"_ZN7android20RaiseFileNumberLimitEv", GetTrampolineFunc<auto()->void>(), nullptr},

    // GraphicBufferImpl static methods.
    //   static long Create(int w, int h);
    {"_ZN7android17GraphicBufferImpl6CreateEii", GetTrampolineFunc<auto(int, int)->long>(), nullptr},
    //   static void Release(long buffer_id);
    {"_ZN7android17GraphicBufferImpl7ReleaseEl", GetTrampolineFunc<auto(long)->void>(), nullptr},
    //   static int MapStatic(long buffer_id, AwMapMode mode, void** vaddr);
    {"_ZN7android17GraphicBufferImpl9MapStaticEl9AwMapModePPv",
     GetTrampolineFunc<auto(long, int, void*)->int>(), nullptr},
    //   static int UnmapStatic(long buffer_id);
    {"_ZN7android17GraphicBufferImpl11UnmapStaticEl", GetTrampolineFunc<auto(long)->int>(), nullptr},
    //   static void* GetNativeBufferStatic(long buffer_id);
    {"_ZN7android17GraphicBufferImpl21GetNativeBufferStaticEl",
     GetTrampolineFunc<auto(long)->void*>(), nullptr},
    //   static int GetStrideStatic(long buffer_id);
    {"_ZN7android17GraphicBufferImpl15GetStrideStaticEl", GetTrampolineFunc<auto(long)->int>(),
     nullptr},

    // GraphicBufferImpl instance methods (the leading void* is `this`).
    //   int Map(AwMapMode mode, void** vaddr);
    {"_ZN7android17GraphicBufferImpl3MapE9AwMapModePPv",
     GetTrampolineFunc<auto(void*, int, void*)->int>(), nullptr},
    //   int Unmap();
    {"_ZN7android17GraphicBufferImpl5UnmapEv", GetTrampolineFunc<auto(void*)->int>(), nullptr},
    //   void* GetNativeBuffer() const;
    {"_ZNK7android17GraphicBufferImpl15GetNativeBufferEv",
     GetTrampolineFunc<auto(void*)->void*>(), nullptr},
    //   int GetStride() const;
    {"_ZNK7android17GraphicBufferImpl9GetStrideEv", GetTrampolineFunc<auto(void*)->int>(), nullptr},
    //   int InitCheck() const;
    {"_ZNK7android17GraphicBufferImpl9InitCheckEv", GetTrampolineFunc<auto(void*)->int>(), nullptr},
    //   GraphicBufferImpl(uint32_t w, uint32_t h);  (C2: base-object constructor)
    {"_ZN7android17GraphicBufferImplC2Ejj",
     GetTrampolineFunc<auto(void*, unsigned, unsigned)->void>(), nullptr},
    //   ~GraphicBufferImpl();  (D2: base-object destructor)
    {"_ZN7android17GraphicBufferImplD2Ev", GetTrampolineFunc<auto(void*)->void>(), nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libwebviewchromium_plat_support.so",
                                     kDigitalisExtraLibwebviewchromiumPlatSupportTrampolines)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
