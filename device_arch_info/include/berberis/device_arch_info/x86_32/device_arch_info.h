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

#ifndef BERBERIS_DEVICE_ARCH_INFO_X86_32_DEVICE_ARCH_INFO_H_
#define BERBERIS_DEVICE_ARCH_INFO_X86_32_DEVICE_ARCH_INFO_H_

#include <x86intrin.h>

#include <cstdint>

#include "berberis/assembler/x86_32.h"
#include "berberis/device_arch_info/x86_32_or_x86_64/device_arch_info.h"

namespace berberis {

namespace x86_32 {

namespace device_arch_info {

// Note: normally using namespace is forbidden in headers, but these two namespaces literally
// only exist to be imported here (and in other device CPU-specific headers).

using namespace berberis::x86_32_or_x86_64::device_arch_info;

class Cond {
 public:
  using Type = x86_32_or_x86_64::Assembler<x86_64::Assembler>::Condition;
};

// We don't currently have use-cases where call may be embedded into MachineIR and we don't know how
// to properly handle it.
// We would need to make this class “real” to be able to do that, but also would probably need
// other changes.
class ESP;

#include "berberis/device_arch_info/x86_32/machine_reg_class-inl.h"

}  // namespace device_arch_info

#include "berberis/device_arch_info/all_to_x86_32_or_x86_64/device_insn_info-inl.h"
#include "berberis/device_arch_info/x86_32/device_insn_info-inl.h"
#include "berberis/device_arch_info/x86_32_or_x86_64/device_insn_info-inl.h"

}  // namespace x86_32

namespace device_arch_info {

template <>
inline constexpr bool kIsCondition<x86_32::device_arch_info::Cond> = true;

template <>
inline constexpr bool kIsGeneralReg32<x86_32::device_arch_info::GeneralReg32> = true;

template <>
inline constexpr bool kIsFLAGS<x86_32::device_arch_info::FLAGS> = true;

template <>
inline constexpr bool kIsRegister<x86_32::device_arch_info::FLAGS> = true;

}  // namespace device_arch_info

}  // namespace berberis

#endif  // BERBERIS_DEVICE_ARCH_INFO_X86_32_DEVICE_ARCH_INFO_H_
