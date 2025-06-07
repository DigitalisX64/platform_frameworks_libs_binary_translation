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

#ifndef BERBERIS_DEVICE_ARCH_INFO_RISCV64_DEVICE_ARCH_INFO_H_
#define BERBERIS_DEVICE_ARCH_INFO_RISCV64_DEVICE_ARCH_INFO_H_

#include <cstdint>

#include "berberis/assembler/riscv.h"
#include "berberis/device_arch_info/common/device_arch_info.h"

namespace berberis {

namespace riscv64::device_arch_info {

// Note: normally using namespace is forbidden in headers, but these two namespaces literally
// only exist to be imported here (and in other device CPU-specific headers).

using namespace berberis::device_arch_info;

class BImm {
 public:
  using Type = riscv::BImmediate;
};

class CsrImm {
 public:
  using Type = riscv::CsrImmediate;
};

class IImm {
 public:
  using Type = riscv::IImmediate;
};

class JImm {
 public:
  using Type = riscv::JImmediate;
};

class PImm {
 public:
  using Type = riscv::PImmediate;
};

class SImm {
 public:
  using Type = riscv::SImmediate;
};

class Shift32Imm {
 public:
  using Type = riscv::Shift32Immediate;
};

class Shift64Imm {
 public:
  using Type = riscv::Shift64Immediate;
};

class UImm {
 public:
  using Type = riscv::UImmediate;
};

#include "berberis/device_arch_info/riscv64/machine_reg_class-inl.h"

}  // namespace riscv64::device_arch_info

namespace device_arch_info {

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::BImm> = true;

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::CsrImm> = true;

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::IImm> = true;

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::JImm> = true;

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::PImm> = true;

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::SImm> = true;

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::Shift32Imm> = true;

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::Shift64Imm> = true;

template <>
inline constexpr bool kIsImmediate<riscv64::device_arch_info::UImm> = true;

}  // namespace device_arch_info

}  // namespace berberis

#endif  // BERBERIS_DEVICE_ARCH_INFO_RISCV64_DEVICE_ARCH_INFO_H_
