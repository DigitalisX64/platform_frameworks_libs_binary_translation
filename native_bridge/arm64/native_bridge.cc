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

#include "berberis/native_bridge/native_bridge.h"

namespace berberis {

const char* kGuestIsa = "arm64";

const char* kSupportedLibraryPathSubstring = "/lib/arm64";

const android::NativeBridgeRuntimeValues kNativeBridgeRuntimeValues = {
    .os_arch = "aarch64",
    .cpu_abi = "arm64-v8a",
    .cpu_abi2 = nullptr,
    .supported_abis = nullptr,
    .abi_count = 0,
};

}  // namespace berberis
