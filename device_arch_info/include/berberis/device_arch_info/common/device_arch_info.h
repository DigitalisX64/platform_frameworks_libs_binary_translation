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

#ifndef BERBERIS_DEVICE_ARCH_INFO_COMMON_DEVICE_ARCH_INFO_H_
#define BERBERIS_DEVICE_ARCH_INFO_COMMON_DEVICE_ARCH_INFO_H_

#include <cstdint>

#include "berberis/base/string_literal.h"

namespace berberis::device_arch_info {

class Mem8 {
 public:
  using Type = uint8_t;
  static constexpr char kAsRegister = 'm';
};

class Mem16 {
 public:
  using Type = uint16_t;
  static constexpr char kAsRegister = 'm';
};

class Mem32 {
 public:
  using Type = uint32_t;
  static constexpr char kAsRegister = 'm';
};

class Mem64 {
 public:
  using Type = uint64_t;
  static constexpr char kAsRegister = 'm';
};

template <typename OperandClass, typename = void>
inline constexpr bool kIsGeneralReg32 = false;

template <typename OperandClass, typename = void>
inline constexpr bool kIsFLAGS = false;

template <typename OperandClass, typename = void>
inline constexpr bool kIsImmediate = false;

template <typename OperandClass, typename = void>
inline constexpr bool kIsMemoryOperand = false;

template <typename OperandClass, typename = void>
inline constexpr bool kIsRegister = !kIsImmediate<OperandClass> && !kIsMemoryOperand<OperandClass>;

template <typename OperandClass, typename = void>
inline constexpr bool kIsImplicitReg = false;

template <typename OperandClass>
inline constexpr bool
    kIsGeneralReg32<OperandClass, std::enable_if_t<sizeof(typename OperandClass::Class) >= 1>> =
        kIsGeneralReg32<typename OperandClass::Class>;

template <typename OperandClass>
inline constexpr bool
    kIsFLAGS<OperandClass, std::enable_if_t<sizeof(typename OperandClass::Class) >= 1>> =
        kIsFLAGS<typename OperandClass::Class>;

template <typename OperandClass>
inline constexpr bool
    kIsImmediate<OperandClass, std::enable_if_t<sizeof(typename OperandClass::Class) >= 1>> =
        kIsImmediate<typename OperandClass::Class>;

template <typename RegisterClass>
inline constexpr bool kIsImplicitReg<
    RegisterClass,
    std::enable_if_t<kIsRegister<RegisterClass> &&
                     std::tuple_size_v<typename RegisterClass::RegistersList> == 1>> = true;

template <typename OperandClass>
inline constexpr bool
    kIsImplicitReg<OperandClass, std::enable_if_t<sizeof(typename OperandClass::Class) >= 1>> =
        kIsImplicitReg<typename OperandClass::Class>;

template <typename OperandClass>
inline constexpr bool
    kIsMemoryOperand<OperandClass, std::enable_if_t<sizeof(typename OperandClass::Class) >= 1>> =
        kIsMemoryOperand<typename OperandClass::Class>;

template <>
inline constexpr bool kIsMemoryOperand<Mem8> = true;

template <>
inline constexpr bool kIsMemoryOperand<Mem16> = true;

template <>
inline constexpr bool kIsMemoryOperand<Mem32> = true;

template <>
inline constexpr bool kIsMemoryOperand<Mem64> = true;

// Note: value of RegBindingKind and MachineRegKind have to be the same since we convert one to
// another with a static_cast in berberis/backend/x86_64/machine_insn_intrinsics.h. We don't care
// about these values in intrinsics module, but for optimizations it's important to have LSB set
// when an instruction uses the value (which is true for kDefEarlyClobber: in that case the
// instruction sets the value and then uses it), the next bit is set when register is output and MSB
// bit is set when register is input. We have static_assert in the aforemetioned header that ensures
// that an attempt to change these two enums and make them different would lead to a compile-time
// error.
enum RegBindingKind { kDef = 2, kDefEarlyClobber = 3, kUse = 5, kUseDef = 7 };

template <typename OperandClass, RegBindingKind kUsageTemplateName>
class OperandInfo {
 public:
  using Class = OperandClass;
  static constexpr RegBindingKind kUsage = kUsageTemplateName;
  static_assert(!kIsImmediate<Class> || kUsage == kUse);
};

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoCPUIDRestriction;  // All CPUs have at least “no CPUID restriction” mode.

template <auto kEmitInsnFunc,
          StringLiteral kMnemo,
          bool kSideEffects,
          auto GetOpcode,
          typename... Types>
class DeviceInsnInfo;

template <auto kEmitInsnFunc_,
          StringLiteral kMnemo,
          bool kSideEffects_,
          auto GetOpcode,
          typename CPUIDRestriction_,
          typename... Operands_>
class DeviceInsnInfo<kEmitInsnFunc_,
                     kMnemo,
                     kSideEffects_,
                     GetOpcode,
                     CPUIDRestriction_,
                     std::tuple<Operands_...>>
    final {
 public:
  static constexpr auto kEmitInsnFunc = kEmitInsnFunc_;
  static constexpr bool kSideEffects = kSideEffects_;
  using CPUIDRestriction = CPUIDRestriction_;
  template <typename Callback, typename... Args>
  constexpr static void ProcessOperands(Callback&& callback, Args&&... args) {
    (callback(Operands_{}, std::forward<Args>(args)...), ...);
  }
  template <typename Callback, typename... Args>
  constexpr static bool VerifyOperands(Callback&& callback, Args&&... args) {
    return (callback(Operands_{}, std::forward<Args>(args)...) && ...);
  }
  template <typename Callback, typename... Args>
  constexpr static auto MakeTuplefromOperands(Callback&& callback, Args&&... args) {
    return std::tuple_cat(callback(Operands_{}, std::forward<Args>(args)...)...);
  }
  using Operands = std::tuple<Operands_...>;
};

}  // namespace berberis::device_arch_info

#endif  // BERBERIS_DEVICE_ARCH_INFO_COMMON_DEVICE_ARCH_INFO_H_
