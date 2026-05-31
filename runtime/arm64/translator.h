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

#ifndef BERBERIS_RUNTIME_ARM64_TRANSLATOR_H_
#define BERBERIS_RUNTIME_ARM64_TRANSLATOR_H_

#include <cstdint>
#include <cstdlib>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {

HostCodePiece InstallTranslated(MachineCode* machine_code,
                                GuestAddr pc,
                                size_t size,
                                const char* prefix);
std::tuple<bool, uint8_t> IsPcExecutable(GuestAddr pc, GuestMapShadow* guest_map_shadow);

void InitTranslatorArch();

}  // namespace berberis

#endif  // BERBERIS_RUNTIME_ARM64_TRANSLATOR_H_
