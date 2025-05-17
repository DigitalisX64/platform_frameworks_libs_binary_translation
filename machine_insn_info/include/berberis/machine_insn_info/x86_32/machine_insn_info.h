/*
 * Copyright (C) 2025 The Android Open Source Project
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

#ifndef BERBERIS_MACHINE_INSN_INFO_X86_32_MACHINE_INSN_INFO_H_
#define BERBERIS_MACHINE_INSN_INFO_X86_32_MACHINE_INSN_INFO_H_

#include <x86intrin.h>

#include <cstdint>

#include "berberis/machine_insn_info/x86_32_or_x86_64/machine_insn_info.h"

namespace berberis {

namespace x86_32::machine_insn_info {

// Note: normally using namespace is forbidden in headers, but these two namespaces literally
// only exist to be imported here (and in other device CPU-specific headers).

using namespace berberis::x86_32_or_x86_64::machine_insn_info;

#include "berberis/machine_insn_info/x86_32/machine_reg_class-inl.h"

}  // namespace x86_32::machine_insn_info

namespace machine_insn_info {

template <>
inline constexpr bool kIsFLAGS<x86_32::machine_insn_info::FLAGS> = true;

template <>
inline constexpr bool kIsRegister<x86_32::machine_insn_info::FLAGS> = true;

}  // namespace machine_insn_info

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_X86_32_MACHINE_INSN_INFO_H_
