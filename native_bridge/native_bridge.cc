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

#include "native_bridge.h"

#include <dlfcn.h>
#include <libgen.h>
#include <stdio.h>
#include <sys/system_properties.h>

#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>

#include "procinfo/process_map.h"

// region digitalis
#include <android/log.h>
#define DIGITALIS_LOG(...) __android_log_print(ANDROID_LOG_DEBUG, "berberis", __VA_ARGS__)

#include <cstring>

// arm64-guest-only prebuilt-app support (in-APK guest-lib extract fallback +
// Qt plugin-path injection) lives in a dedicated arm64-only translation unit
// (arm64/digitalis_app_support.cc) so its libziparchive dependency and code
// never enter the riscv64 native bridge. fdsan demotion below is likewise
// arm64-only.
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
#include <android/fdsan.h>

#include "arm64/digitalis_app_support.h"
#endif
// endregion

#include "berberis/base/algorithm.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/file.h"
#include "berberis/base/strings.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_abi/guest_call.h"
#include "berberis/guest_loader/guest_loader.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/jni/jni_trampolines.h"
#include "berberis/native_activity/native_activity_wrapper.h"
#include "berberis/native_bridge/native_bridge.h"
#include "berberis/runtime/berberis.h"
#include "berberis/runtime_primitives/known_guest_function_wrapper.h"

#define LOG_NB TRACE

extern "C" {

// Extended android loader functions for namespace management

bool android_init_anonymous_namespace(const char* shared_libs_sonames,
                                      const char* library_search_path);

struct android_namespace_t* android_create_namespace(const char* name,
                                                     const char* ld_library_path,
                                                     const char* default_library_path,
                                                     uint64_t type,
                                                     const char* permitted_when_isolated_path,
                                                     struct android_namespace_t* parent);

bool android_link_namespaces(struct android_namespace_t* from,
                             struct android_namespace_t* to,
                             const char* shared_libs_sonames);

struct android_namespace_t* android_get_exported_namespace(const char* name);

}  // extern "C"

namespace android {

// We maintain host namespace here to provide ability to open host
// libraries via native bridge. See http://b/308371292 for details.
struct native_bridge_namespace_t {
  android_namespace_t* guest_namespace;
  android_namespace_t* host_namespace;
};

}  // namespace android

namespace {

// See android/system/core/libnativebridge/native_bridge.cc
// Even thought berberis does not support early version of NB interface
// (deprecated methods do not work anymore) v2 support is needed to have NB call
// getSignalHandler function.
const constexpr uint32_t kNativeBridgeCallbackMinVersion = 2;
const constexpr uint32_t kNativeBridgeCallbackVersion = 8;
const constexpr uint32_t kNativeBridgeCallbackMaxVersion = kNativeBridgeCallbackVersion;

const android::NativeBridgeRuntimeCallbacks* g_runtime_callbacks = nullptr;

using native_bridge_namespace_t = android::native_bridge_namespace_t;
using GuestAddr = berberis::GuestAddr;

// Treble uses "sphal" name for the vendor namespace.
constexpr const char* kVendorNamespaceName = "sphal";

class NdktNativeBridge {
 public:
  NdktNativeBridge();
  ~NdktNativeBridge();

  bool Initialize(std::string* error_msg);
  void* LoadLibrary(const char* libpath, int flags, const native_bridge_namespace_t* ns = nullptr);
  void* LoadGuestLibrary(const char* libpath,
                         int flags,
                         const native_bridge_namespace_t* ns = nullptr);
  GuestAddr DlSym(void* handle, const char* name);
  const char* DlError();
  bool InitAnonymousNamespace(const char* public_ns_sonames, const char* anon_ns_library_path);
  native_bridge_namespace_t* CreateNamespace(const char* name,
                                             const char* ld_library_path,
                                             const char* default_library_path,
                                             uint64_t type,
                                             const char* permitted_when_isolated_path,
                                             native_bridge_namespace_t* parent_ns);
  native_bridge_namespace_t* GetExportedNamespace(const char* name);
  bool LinkNamespaces(native_bridge_namespace_t* from,
                      native_bridge_namespace_t* to,
                      const char* shared_libs_sonames);

  bool IsHostHandle(void* handle) const;

 private:
  bool FinalizeInit();
  native_bridge_namespace_t* CreateNativeBridgeNamespace(android_namespace_t* host_namespace,
                                                         android_namespace_t* guest_namespace);
  void AddHostLibrary(void* handle);

  berberis::GuestLoader* guest_loader_;

  mutable std::mutex host_libraries_lock_;
  std::set<void*> host_libraries_;

  std::mutex namespaces_lock_;
  std::deque<native_bridge_namespace_t> namespaces_;
  std::map<std::string, native_bridge_namespace_t> exported_namespaces_;
};

NdktNativeBridge::NdktNativeBridge() : guest_loader_(nullptr) {}

NdktNativeBridge::~NdktNativeBridge() {}

bool NdktNativeBridge::Initialize(std::string* error_msg) {
  guest_loader_ = berberis::GuestLoader::StartAppProcessInNewThread(error_msg);
  berberis::RegisterKnownGuestFunctionWrapper("ANativeActivity_onCreate",
                                              berberis::WrapGuestNativeActivityOnCreate);
  berberis::InitializeJNI();
  return guest_loader_ != nullptr;
}

void* NdktNativeBridge::LoadLibrary(const char* libpath,
                                    int flags,
                                    const native_bridge_namespace_t* ns) {
  // We don't have a callback after all java initialization is finished. So we call the finalizing
  // routine from here, just before we load any app's native code.
  static bool init_finalized = FinalizeInit();
  UNUSED(init_finalized);

  // region digitalis
  // In-APK path fast path: when libpath is "<apk>!/<entry>" (Qt 6 apps that
  // ship native libs inside base.apk), the guest linker's own load of that
  // path is known to either fail fast (sister Qt 6 libs) or hang for >30 s
  // (observed with libQt6Widgets in VulkanCapsViewer 4.11 — first-launch
  // extraction inside the guest linker). Either way, the eventual successful
  // load is via our host-side ExtractInApkLibToCache + cache-path reload.
  // Skip the broken guest-linker `!/` attempt entirely and extract first.
  // This also turns the previously-hanging Widgets case into a one-time
  // host-side ZIP extract + a normal cache-path dlopen.
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  if (libpath != nullptr && strstr(libpath, "!/") != nullptr) {
    std::string extracted = berberis::ExtractInApkLibToCache(libpath);
    if (!extracted.empty()) {
      void* cached = LoadGuestLibrary(extracted.c_str(), flags, ns);
      if (cached != nullptr) {
        return cached;
      }
      const char* cache_err = guest_loader_->DlError();
      DIGITALIS_LOG("LoadGuestLibrary for extracted %s failed: %s",
                    extracted.c_str(), cache_err ? cache_err : "(no error)");
    }
  }
#endif  // defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  // endregion

  void* handle = LoadGuestLibrary(libpath, flags, ns);
  if (handle != nullptr) {
    return handle;
  }

  // region digitalis
  {
    const char* guest_err = guest_loader_->DlError();
    DIGITALIS_LOG("LoadGuestLibrary FAILED for %s: %s", libpath, guest_err ? guest_err : "(no error)");
  }
  // endregion

  // http://b/206676167: Do not fallback to host for libRS.so
  if (berberis::Basename(libpath) == "libRS.so") {
    return handle;
  }

  // Try falling back to host loader.
  android_dlextinfo extinfo_holder;
  android_dlextinfo* extinfo = nullptr;

  if (ns != nullptr) {
    extinfo_holder.flags = ANDROID_DLEXT_USE_NAMESPACE;
    extinfo_holder.library_namespace = ns->host_namespace;
    extinfo = &extinfo_holder;
  }

  handle = android_dlopen_ext(libpath, flags, extinfo);
  if (handle != nullptr) {
    TRACE_AND_ALOGI("'%s' library was loaded for the host platform.", libpath);
    AddHostLibrary(handle);
  }

  return handle;
}

void* NdktNativeBridge::LoadGuestLibrary(const char* libpath,
                                         int flags,
                                         const native_bridge_namespace_t* ns) {
  android_dlextinfo extinfo_holder;
  android_dlextinfo* extinfo = nullptr;

  if (ns != nullptr) {
    extinfo_holder.flags = ANDROID_DLEXT_USE_NAMESPACE;
    extinfo_holder.library_namespace = ns->guest_namespace;
    extinfo = &extinfo_holder;
  }

  return guest_loader_->DlOpenExt(libpath, flags, extinfo);
}

void NdktNativeBridge::AddHostLibrary(void* handle) {
  const std::lock_guard<std::mutex> guard(host_libraries_lock_);
  host_libraries_.insert(handle);
}

bool NdktNativeBridge::IsHostHandle(void* handle) const {
  const std::lock_guard<std::mutex> guard(host_libraries_lock_);
  return berberis::Contains(host_libraries_, handle);
}

GuestAddr NdktNativeBridge::DlSym(void* handle, const char* name) {
  CHECK(!IsHostHandle(handle));
  return guest_loader_->DlSym(handle, name);
}

const char* NdktNativeBridge::DlError() {
  // There is no good way of knowing where the error happened, - prioritize the guest loader.
  const char* error = guest_loader_->DlError();
  if (error != nullptr) {
    return error;
  }

  return dlerror();
}

native_bridge_namespace_t* NdktNativeBridge::CreateNamespace(
    const char* name,
    const char* ld_library_path,
    const char* default_library_path,
    uint64_t type,
    const char* permitted_when_isolated_path,
    native_bridge_namespace_t* parent_ns) {
  // Android SDK libraries do not have a good mechanism for using JNI libraries.
  // The only way to make it work is to put them to system search path and make
  // the library public (visible from apps). See http://b/308371292 for details.
  //
  // Since `ClassLoader.findLibrary` is looking for the library in 'java.library.path'
  // in addition to paths used here it is able to find a JNI library located in system
  // library path. If then such a library appears to be a public library, the android
  // loader will be able to load it from the system linker namespace.
  //
  // It could also happen so that the app puts different architecture libraries
  // in the same folder (say x86_64 libraries to arm64 folder), in which case
  // they will work if the architecture happens to match with host one. This is
  // why we preserve guest search path for the host namespace.
  auto* host_namespace = android_create_namespace(name,
                                                  ld_library_path,
                                                  default_library_path,
                                                  type,
                                                  permitted_when_isolated_path,
                                                  parent_ns->host_namespace);

  // region digitalis
  std::string guest_default_path;
  std::string guest_permitted;
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  // Do NOT add /system/lib64/arm64 to an app's (isolated) namespace search or
  // permitted paths. Real Android keeps app classloader namespaces isolated from
  // the system namespace: the app can only load its own libs plus the NDK public
  // libraries reached through the namespace *link* (shared instances that resolve
  // their own symbols in the system namespace). Adding the system path let the app
  // namespace load its own copies of non-public system libraries (libutils,
  // libc++, libandroid_runtime, ...). Because those copies then live in the app's
  // symbol scope, an app library's *strong* replaceable symbol — most importantly
  // Unity's GLOBAL `operator new` in libunity.so — interposes for them. A system
  // library calling `operator new` was thus routed into the app's not-yet-
  // initialized allocator and crashed (Honkai: Star Rail SIGILL on libunity's
  // encrypted lazy-init path). Proxy/public libraries the app legitimately needs
  // (libandroid.so, libvulkan.so, libEGL.so, ...) still resolve via the framework's
  // linkNamespaces() link to the guest "default" namespace, which has the ARM64
  // system search path, so this only removes the incorrect direct-load fallback.
  guest_default_path = (default_library_path != nullptr) ? default_library_path : "";
  guest_permitted = (permitted_when_isolated_path != nullptr) ? permitted_when_isolated_path : "";
#else
  // Non-arm64 guests keep the original behaviour (append the guest system search
  // path) — this fix is validated for the arm64 guest only.
  if (default_library_path != nullptr) {
    guest_default_path = default_library_path;
    guest_default_path += ":/system/lib64/arm64/bootstrap:/system/lib64/arm64";
  } else {
    guest_default_path = "/system/lib64/arm64/bootstrap:/system/lib64/arm64";
  }
  if (permitted_when_isolated_path != nullptr) {
    guest_permitted = permitted_when_isolated_path;
    guest_permitted += ":/system/lib64/arm64/bootstrap:/system/lib64/arm64";
  } else {
    guest_permitted = "/system/lib64/arm64/bootstrap:/system/lib64/arm64";
  }
#endif
  DIGITALIS_LOG("createNamespace guest: name=%s default_path=%s permitted=%s",
                name, guest_default_path.c_str(), guest_permitted.c_str());
  // endregion

  auto* guest_namespace = guest_loader_->CreateNamespace(name,
                                                         ld_library_path,
                                                         guest_default_path.c_str(),
                                                         type,
                                                         guest_permitted.c_str(),
                                                         parent_ns->guest_namespace);

  return CreateNativeBridgeNamespace(host_namespace, guest_namespace);
}

native_bridge_namespace_t* NdktNativeBridge::GetExportedNamespace(const char* name) {
  const std::lock_guard<std::mutex> guard(namespaces_lock_);
  auto it = exported_namespaces_.find(name);
  if (it != exported_namespaces_.end()) {
    return &it->second;
  }

  auto host_namespace = android_get_exported_namespace(name);
  auto guest_namespace = guest_loader_->GetExportedNamespace(name);

  // region digitalis
  // The guest linker config may not export all namespaces that the host config
  // does (e.g. "system", "com_android_art"). When the guest doesn't have the
  // requested namespace, fall back to the guest's "default" namespace which
  // contains the ARM64 system library search paths.
  if (guest_namespace == nullptr) {
    guest_namespace = guest_loader_->GetExportedNamespace("default");
    DIGITALIS_LOG("getExportedNamespace: guest fallback for '%s' -> default=%p", name, guest_namespace);
  }
  // endregion

  auto [insert_it, inserted] =
      exported_namespaces_.try_emplace(std::string(name),
                                       native_bridge_namespace_t{.guest_namespace = guest_namespace,
                                                                 .host_namespace = host_namespace});
  CHECK(inserted);

  return &insert_it->second;
}

bool NdktNativeBridge::InitAnonymousNamespace(const char* public_ns_sonames,
                                              const char* anon_ns_library_path) {
  return guest_loader_->InitAnonymousNamespace(public_ns_sonames, anon_ns_library_path) &&
         android_init_anonymous_namespace(public_ns_sonames, anon_ns_library_path);
}

bool NdktNativeBridge::LinkNamespaces(native_bridge_namespace_t* from,
                                      native_bridge_namespace_t* to,
                                      const char* shared_libs_sonames) {
  // region digitalis
  // Add linux-vdso.so.1 to the guest shared libs whitelist so the guest linker
  // can find the TinyLoader-loaded vDSO across namespace boundaries. Without this,
  // the guest linker loads a second copy from the filesystem which has no
  // trampolines registered, causing a null function pointer crash (SIGSEGV at
  // vDSO offset 0x800c — blr x3 with x3=0).
  std::string guest_shared_libs;
  if (shared_libs_sonames != nullptr) {
    guest_shared_libs = shared_libs_sonames;
    guest_shared_libs += ":linux-vdso.so.1";
  } else {
    guest_shared_libs = "linux-vdso.so.1";
  }
  // endregion
  return guest_loader_->LinkNamespaces(
             from->guest_namespace, to->guest_namespace, guest_shared_libs.c_str()) &&
         android_link_namespaces(from->host_namespace, to->host_namespace, shared_libs_sonames);
}

native_bridge_namespace_t* NdktNativeBridge::CreateNativeBridgeNamespace(
    android_namespace_t* host_namespace,
    android_namespace_t* guest_namespace) {
  const std::lock_guard<std::mutex> guard(namespaces_lock_);
  namespaces_.emplace_back(native_bridge_namespace_t{.guest_namespace = guest_namespace,
                                                     .host_namespace = host_namespace});
  return &namespaces_.back();
}

void ProtectMappingsFromGuest() {
  auto callback =
      [](uint64_t start, uint64_t end, uint16_t, uint64_t, ino_t, const char* libname_c_str, bool) {
        std::string_view libname(libname_c_str);
        // Per analysis in b/218772975 only libc is affected. It's occasionally either proxy libc or
        // guest libc. So we protect all libs with "libc.so" substring. At this point no app's libs
        // are loaded yet, so the app shouldn't tamper with the already loaded ones. We don't
        // protect all the already loaded libraries though since GuestMapShadow isn't optimized
        // to work with large number of entries. Also some of them could be unmapped later, which is
        // not expected for libc.so.
        if (libname.find("libc.so") != std::string_view::npos) {
          berberis::GuestMapShadow::GetInstance()->AddProtectedMapping(
              berberis::bit_cast<void*>(static_cast<uintptr_t>(start)),
              berberis::bit_cast<void*>(static_cast<uintptr_t>(end)));
        }
      };
  android::procinfo::ReadMapFile("/proc/self/maps", callback);
}

extern "C" const char* __progname;

bool NdktNativeBridge::FinalizeInit() {
  // Guest-libc is expected to be loaded along with app-process during Initialize(). At that time
  // __progname isn't yet initialized in java. So now when it should be initialized we copy it over
  // from host to guest.
  // Note that we cannot delay Initialize() (and hence guest-libc loading) until now because
  // guest_loader initialized there is then used to create and link linker namespaces.
  // We cannot cannot unload (dlclose) guest-libc after app-process loading either (intending to
  // reload it now to get the updated __progname), since guest_linker is already tightly linked with
  // it.

  // Force libc loading if it's not loaded yet to ensure the symbol is overridden.
  // We do not call LoadLibrary since it'd recurse back into FinalizeInit.
  void* libc = guest_loader_->DlOpenExt("libc.so", RTLD_NOW, nullptr);
  CHECK_NE(libc, nullptr);
  auto addr = DlSym(libc, "__progname");
  CHECK_NE(addr, berberis::kNullGuestAddr);
  memcpy(berberis::ToHostAddr<char*>(addr), &__progname, sizeof(__progname));

  // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  berberis::DigitalisInjectQtPluginPath(
      libc, [this](void* handle, const char* name) { return DlSym(handle, name); });
#endif  // defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  // endregion

  // Now, when guest libc and proxy-libc are loaded,
  // remember mappings which guest code must not tamper with.
  ProtectMappingsFromGuest();

  return true;
}

NdktNativeBridge g_ndkt_native_bridge;

// Runtime values must be non-NULL, otherwise native bridge will be disabled.
// Note, that 'supported_abis' and 'abi_count' are deprecated (b/18061712).
const struct android::NativeBridgeRuntimeValues* GetAppEnvByIsa(const char* app_isa) {
  if (app_isa == nullptr) {
    TRACE_AND_ALOGE("instruction set is null");
    return nullptr;
  }

  if (strcmp(app_isa, berberis::kGuestIsa) == 0) {
    return &berberis::kNativeBridgeRuntimeValues;
  }

  TRACE_AND_ALOGE("unknown instruction set '%s'", app_isa);
  return nullptr;
}

void SetAppPropertiesFromCodeCachePath(const char* private_dir) {
  if (private_dir == nullptr) {
    return;
  }

  // Expect private_dir to be .../<app_package>/code_cache
  std::string_view path(private_dir);
  if (!berberis::ConsumeSuffix(&path, "/code_cache")) {
    return;
  }

  berberis::SetAppPrivateDir(path);

  auto begin = path.find_last_of('/');
  if (begin == std::string_view::npos) {
    return;
  }
  berberis::SetAppPackageName(path.substr(begin + 1));
}

bool native_bridge_initialize(const android::NativeBridgeRuntimeCallbacks* runtime_cbs,
                              const char* private_dir,
                              const char* instruction_set) {
  LOG_NB("native_bridge_initialize(runtime_callbacks=%p, private_dir='%s', app_isa='%s')",
         runtime_cbs,
         private_dir ? private_dir : "(null)",
         instruction_set ? instruction_set : "(null)");
  auto* env = GetAppEnvByIsa(instruction_set);
  if (env == nullptr) {
    return false;
  }
  g_runtime_callbacks = runtime_cbs;
  SetAppPropertiesFromCodeCachePath(private_dir);
  berberis::InitBerberis();

  // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  // bionic's fdsan enforces, per host file descriptor, that the object holding
  // it (a unique_fd, Parcel, Fence, DIR*, …) is the one that closes it. That
  // invariant assumes a single party owns the fd table. Under translation the
  // table is shared between host code and the guest, which issues raw close /
  // close_range / dup3 syscalls (often aggressively, e.g. anti-tamper SDKs
  // sweeping every fd) and so can close, replace, or reuse an fd a live host
  // object still owns. The kernel-level syscall does not update the host
  // fdsan owner table, so the host object's eventual close trips fdsan and a
  // FATAL level aborts the whole process. The per-syscall emulation
  // (RunGuestSyscall___NR_close / _close_range / _dup3) already avoids
  // destroying live host-owned fds, but it cannot close every race or
  // host-internal path. Demote fdsan to warn-once for the guest process so a
  // genuine cross-boundary ownership mismatch logs instead of killing the app
  // — matching the non-FATAL default real Android apps run with, while keeping
  // the diagnostic. Host-only proxy bugs still surface as the logged warning.
  android_fdsan_set_error_level(ANDROID_FDSAN_ERROR_LEVEL_WARN_ONCE);
#endif
  // endregion

  char version[PROP_VALUE_MAX];
  if (__system_property_get("ro.berberis.version", version)) {
    TRACE_AND_ALOGI("Initialized Berberis (%s), version %s", env->os_arch, version);
  } else {
    TRACE_AND_ALOGI("Initialized Berberis (%s)", env->os_arch);
  }

  std::string error_msg;
  if (!g_ndkt_native_bridge.Initialize(&error_msg)) {
    LOG_ALWAYS_FATAL("native_bridge_initialize: %s", error_msg.c_str());
  }
  return true;
}

void* native_bridge_loadLibrary(const char* libpath, int flag) {
  // We should only get here if this library is not native.
  LOG_NB("native_bridge_loadLibrary(path='%s', flag=0x%x)", libpath ? libpath : "(null)", flag);

  return g_ndkt_native_bridge.LoadLibrary(libpath, flag);
}

void* native_bridge_getTrampolineWithJNICallType(void* handle,
                                                 const char* name,
                                                 const char* shorty,
                                                 uint32_t len,
                                                 enum android::JNICallType jni_call_type) {
  LOG_NB(
      "native_bridge_getTrampolineWithJNICallType(handle=%p, name='%s', shorty='%s', len=%d, "
      "jni_call_type=%d)",
      handle,
      name ? name : "(null)",
      shorty ? shorty : "(null)",
      len,
      jni_call_type);

  if (g_ndkt_native_bridge.IsHostHandle(handle)) {
    return dlsym(handle, name);
  }

  GuestAddr guest_addr = g_ndkt_native_bridge.DlSym(handle, name);
  if (!guest_addr) {
    return nullptr;
  }

  if (shorty) {
    return const_cast<void*>(berberis::WrapGuestJNIFunction(
        guest_addr,
        shorty,
        name,
        jni_call_type != android::JNICallType::kJNICallTypeCriticalNative));
  }

  berberis::HostCode result = berberis::WrapKnownGuestFunction(guest_addr, name);
  if (result == nullptr) {
    // No wrapper is registered for this function name.
    // This usually happens for ANativeActivity_onCreate renamed with android.app.func_name.
    // TODO(b/27307664): maybe query android.app.func_name from Java and check exactly?
    TRACE("No wrapper is registered for %s, assume it's ANativeActivity_onCreate", name);
    result = berberis::WrapKnownGuestFunction(guest_addr, "ANativeActivity_onCreate");
  }
  return const_cast<void*>(result);
}

void* native_bridge_getTrampolineForFunctionPointer(const void* method,
                                                    const char* shorty,
                                                    uint32_t len,
                                                    enum android::JNICallType jni_call_type) {
  LOG_NB(
      "native_bridge_getTrampolineForFunctionPointer(method=%p, shorty='%s', len=%d, "
      "jni_call_type=%d)",
      method,
      shorty ? shorty : "(null)",
      len,
      jni_call_type);

  auto guest_addr = berberis::ToGuestAddr(method);
  if (!berberis::GuestMapShadow::GetInstance()->IsExecutable(guest_addr, 1)) {
    LOG_NB("Not executable method - assuming it's a host library");
    // This is not guest code - happens when native_bridge falls back
    // to host libraries.
    return const_cast<void*>(method);
  }

  return const_cast<void*>(berberis::WrapGuestJNIFunction(
      guest_addr,
      shorty,
      "(unknown)",
      jni_call_type != android::JNICallType::kJNICallTypeCriticalNative));
}

bool native_bridge_isNativeBridgeFunctionPointer(const void* method) {
  bool result =
      berberis::GuestMapShadow::GetInstance()->IsExecutable(berberis::ToGuestAddr(method), 1);
  LOG_NB("native_bridge_isNativeBridgeFunctionPointer(method=%p): %d", method, result);
  return result;
}

void* native_bridge_getTrampoline(void* handle,
                                  const char* name,
                                  const char* shorty,
                                  uint32_t len) {
  LOG_NB(
      "Warning: Unexpected call to native_bridge_getTrampoline (old android version?), converting "
      "to a native_bridge_getTrampolineWithJNICallType call with kJNICallTyepRegular");
  return native_bridge_getTrampolineWithJNICallType(
      handle, name, shorty, len, android::JNICallType::kJNICallTypeRegular);
}

bool native_bridge_isSupported(const char* libpath) {
  LOG_NB("native_bridge_isSupported(path='%s')", libpath ? libpath : "(null)");
  return true;
}

const struct android::NativeBridgeRuntimeValues* native_bridge_getAppEnv(
    const char* instruction_set) {
  LOG_NB("native_bridge_getAppEnv(app_isa='%s')", instruction_set ? instruction_set : "(null)");
  return GetAppEnvByIsa(instruction_set);
}

bool native_bridge_isCompatibleWith(uint32_t bridge_version) {
  LOG_NB("native_bridge_isCompatibleWith(bridge_version=%d)", bridge_version);
  return bridge_version >= kNativeBridgeCallbackMinVersion &&
         bridge_version <= kNativeBridgeCallbackMaxVersion;
}

android::NativeBridgeSignalHandlerFn native_bridge_getSignalHandler(int signal) {
  LOG_NB("native_bridge_getSignalHandler(signal=%d)", signal);
  return nullptr;
}

int native_bridge_unloadLibrary(void* handle) {
  LOG_NB("native_bridge_unloadLibrary(handle=%p)", handle);
  // TODO(b/276787500): support library unloading!
  return 0;
}

const char* native_bridge_getError() {
  LOG_NB("native_bridge_getError()");
  return g_ndkt_native_bridge.DlError();
}

bool native_bridge_isPathSupported(const char* library_path) {
  LOG_NB("native_bridge_isPathSupported(path=%s)", library_path);
  return strstr(library_path, berberis::kSupportedLibraryPathSubstring) != nullptr;
}

bool native_bridge_initAnonymousNamespace(const char* public_ns_sonames,
                                          const char* anon_ns_library_path) {
  // region digitalis
  DIGITALIS_LOG("initAnonymousNamespace: sonames=%.200s path=%s",
                public_ns_sonames ? public_ns_sonames : "(null)",
                anon_ns_library_path ? anon_ns_library_path : "(null)");
  // endregion
  LOG_NB("native_bridge_initAnonymousNamespace(public_ns_sonames=%s, anon_ns_library_path=%s)",
         public_ns_sonames,
         anon_ns_library_path);
  // region digitalis
  bool result = g_ndkt_native_bridge.InitAnonymousNamespace(public_ns_sonames, anon_ns_library_path);
  DIGITALIS_LOG("initAnonymousNamespace: result=%d", result);
  return result;
  // endregion
}

native_bridge_namespace_t* native_bridge_createNamespace(const char* name,
                                                         const char* ld_library_path,
                                                         const char* default_library_path,
                                                         uint64_t type,
                                                         const char* permitted_when_isolated_path,
                                                         native_bridge_namespace_t* parent_ns) {
  // region digitalis
  DIGITALIS_LOG("createNamespace: name=%s ld_path=%s default_path=%s type=%llu permitted=%s parent=%p",
                name ? name : "(null)",
                ld_library_path ? ld_library_path : "(null)",
                default_library_path ? default_library_path : "(null)",
                (unsigned long long)type,
                permitted_when_isolated_path ? permitted_when_isolated_path : "(null)",
                parent_ns);
  // endregion
  LOG_NB("native_bridge_createNamespace(name=%s, path=%s)", name, ld_library_path);
  auto* result = g_ndkt_native_bridge.CreateNamespace(
      name, ld_library_path, default_library_path, type, permitted_when_isolated_path, parent_ns);
  // region digitalis
  DIGITALIS_LOG("createNamespace: result=%p guest=%p", result,
                result ? result->guest_namespace : nullptr);
  // endregion
  return result;
}

bool native_bridge_linkNamespaces(native_bridge_namespace_t* from,
                                  native_bridge_namespace_t* to,
                                  const char* shared_libs_sonames) {
  // region digitalis
  DIGITALIS_LOG("linkNamespaces: from=%p(guest=%p) to=%p(guest=%p) shared_libs=%.300s",
                from, from ? from->guest_namespace : nullptr,
                to, to ? to->guest_namespace : nullptr,
                shared_libs_sonames ? shared_libs_sonames : "(null)");
  // endregion
  LOG_NB("native_bridge_linkNamespaces(from=%p, to=%p, shared_libs=%s)",
         from,
         to,
         shared_libs_sonames);

  // region digitalis
  bool result = g_ndkt_native_bridge.LinkNamespaces(from, to, shared_libs_sonames);
  DIGITALIS_LOG("linkNamespaces: result=%d", result);
  return result;
  // endregion
}

void* native_bridge_loadLibraryExt(const char* libpath, int flag, native_bridge_namespace_t* ns) {
  // region digitalis
  DIGITALIS_LOG("loadLibraryExt: path=%s ns=%p(guest=%p)",
                libpath ? libpath : "(null)",
                ns, ns ? ns->guest_namespace : nullptr);
  // endregion
  LOG_NB("native_bridge_loadLibraryExt(path=%s)", libpath);

  // region digitalis
  void* result = g_ndkt_native_bridge.LoadLibrary(libpath, flag, ns);
  DIGITALIS_LOG("loadLibraryExt: result=%p dlerror=%s", result,
                result ? "ok" : (g_ndkt_native_bridge.DlError() ? g_ndkt_native_bridge.DlError() : "(null)"));
  return result;
  // endregion
}

native_bridge_namespace_t* native_bridge_getVendorNamespace() {
  LOG_NB("native_bridge_getVendorNamespace()");
  // This method is retained for backwards compatibility.
  return g_ndkt_native_bridge.GetExportedNamespace(kVendorNamespaceName);
}

native_bridge_namespace_t* native_bridge_getExportedNamespace(const char* name) {
  LOG_NB("native_bridge_getExportedNamespace(name=%s)", name);
  // region digitalis
  auto* result = g_ndkt_native_bridge.GetExportedNamespace(name);
  DIGITALIS_LOG("getExportedNamespace: name=%s result=%p guest=%p",
                name ? name : "(null)",
                result, result ? result->guest_namespace : nullptr);
  return result;
  // endregion
}

void native_bridge_preZygoteFork() {
  // In case of app-zygote the translator could have executed some guest code
  // during app-zygote's doPreload(). Zygote's fork doesn't allow unrecognized
  // open file descriptors, so we close them.
  //
  // We assume that all guest execution has finished in doPreload() and there
  // are no background guest threads. ART ensures the fork is single-threaded by
  // calling waitUntilAllThreadsStopped() in ZygoteHooks::preZork().
  // TODO(b/188923523): Technically this happens after nativePreFork() (which
  // calls this callback), so theoretically some guest thread may still be
  // running and finishes later. If this happens to be an issue, we can call an
  // analog of waitUntilAllThreadsStopped() here. Or try to call nativePreFork()
  // after waitUntilAllThreadsStopped() in ART.

  // TODO(b/188923523): Consider moving to berberis::GuestPreZygoteFork().
  void* liblog = g_ndkt_native_bridge.LoadGuestLibrary("liblog.so", RTLD_NOLOAD);
  // Nothing to close if the guest library hasn't been loaded.
  if (liblog) {
    auto addr = g_ndkt_native_bridge.DlSym(liblog, "__android_log_close");
    CHECK_NE(addr, berberis::kNullGuestAddr);
    berberis::GuestCall call;
    call.RunVoid(addr);
  }

  berberis::PreZygoteForkUnsafe();
}

}  // namespace

namespace berberis {

const char* GetJMethodShorty(JNIEnv* env, jmethodID mid) {
  CHECK(g_runtime_callbacks);
  return (g_runtime_callbacks->getMethodShorty)(env, mid);
}

}  // namespace berberis

extern "C" {
// "NativeBridgeItf" is effectively an API (it is the name of the symbol that
// will be loaded by the native bridge library).
android::NativeBridgeCallbacks NativeBridgeItf = {
    kNativeBridgeCallbackVersion,
    &native_bridge_initialize,
    &native_bridge_loadLibrary,
    &native_bridge_getTrampoline,
    &native_bridge_isSupported,
    &native_bridge_getAppEnv,
    &native_bridge_isCompatibleWith,
    &native_bridge_getSignalHandler,
    &native_bridge_unloadLibrary,
    &native_bridge_getError,
    &native_bridge_isPathSupported,
    &native_bridge_initAnonymousNamespace,
    &native_bridge_createNamespace,
    &native_bridge_linkNamespaces,
    &native_bridge_loadLibraryExt,
    &native_bridge_getVendorNamespace,
    &native_bridge_getExportedNamespace,
    &native_bridge_preZygoteFork,
    &native_bridge_getTrampolineWithJNICallType,
    &native_bridge_getTrampolineForFunctionPointer,
    &native_bridge_isNativeBridgeFunctionPointer,
};
}  // extern "C"
