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

#ifndef BERBERIS_NATIVE_BRIDGE_ARM64_DIGITALIS_APP_SUPPORT_H_
#define BERBERIS_NATIVE_BRIDGE_ARM64_DIGITALIS_APP_SUPPORT_H_

#include <functional>
#include <string>

#include "berberis/guest_state/guest_addr.h"

// arm64-guest-only prebuilt-app support helpers extracted out of the shared
// native_bridge.cc so their libziparchive dependency and their sizeable code
// never enter the riscv64 native bridge translation unit. This whole file is
// compiled only into libberberis_native_bridge_arm64 (see native_bridge/
// Android.bp), so it carries no NATIVE_BRIDGE_GUEST_ARCH_ARM64 guard; the
// call sites in native_bridge.cc keep that guard.

namespace berberis {

// Extract <apk_path>!/<entry> to
// /data/data/<pkg>/cache/berberis_extract/<basename> and return the extracted
// path on success, or "" on any failure (caller falls through to its existing
// host-loader path). Used as a fallback when the guest dynamic linker fails to
// dlopen an in-APK library path (Qt 6 prebuilt apps). Relies on libziparchive.
std::string ExtractInApkLibToCache(const char* libpath);

// Inject QT_PLUGIN_PATH / QT_QPA_PLATFORM_PLUGIN_PATH into guest libc's environ
// for Qt-for-Android prebuilt apps, and lay down Qt-friendly plugin symlinks in
// the extract dir. `guest_libc` is the guest libc handle; `dlsym` resolves a
// symbol in that guest library — the caller supplies NdktNativeBridge::DlSym,
// which is local to native_bridge.cc and not visible here.
void DigitalisInjectQtPluginPath(
    void* guest_libc,
    const std::function<GuestAddr(void* handle, const char* name)>& dlsym);

}  // namespace berberis

#endif  // BERBERIS_NATIVE_BRIDGE_ARM64_DIGITALIS_APP_SUPPORT_H_
