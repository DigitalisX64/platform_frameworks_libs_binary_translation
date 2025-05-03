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

#ifndef BERBERIS_INTRINSICS_COMMON_MACHINE_INSN_INFO_H_
#define BERBERIS_INTRINSICS_COMMON_MACHINE_INSN_INFO_H_

#include <cstdint>

namespace berberis::machine_insn_info {

class FLAGS {
 public:
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 0;
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kFLAGS;
};

class Mem8 {
 public:
  using Type = uint8_t;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class Mem16 {
 public:
  using Type = uint16_t;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class Mem32 {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class Mem64 {
 public:
  using Type = uint64_t;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

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
  static_assert(!Class::kIsImmediate || kUsage == kUse);
};

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoCPUIDRestriction;  // All CPUs have at least “no CPUID restriction” mode.

template <auto kMacroInstructionTemplateName, auto kMnemo, auto GetOpcode, typename... Types>
class AsmCallInfo;

template <auto kMacroInstructionTemplateName,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestrictionTemplateValue,
          typename... OperandsTypes>
class AsmCallInfo<kMacroInstructionTemplateName,
                  kMnemo,
                  GetOpcode,
                  CPUIDRestrictionTemplateValue,
                  std::tuple<OperandsTypes...>>
    final {
 public:
  static constexpr auto kMacroInstruction = kMacroInstructionTemplateName;
  using CPUIDRestriction = CPUIDRestrictionTemplateValue;
  template <typename Callback, typename... Args>
  constexpr static void ProcessOperands(Callback&& callback, Args&&... args) {
    (callback(OperandsTypes{}, std::forward<Args>(args)...), ...);
  }
  template <typename Callback, typename... Args>
  constexpr static bool VerifyOperands(Callback&& callback, Args&&... args) {
    return (callback(OperandsTypes{}, std::forward<Args>(args)...) && ...);
  }
  template <typename Callback, typename... Args>
  constexpr static auto MakeTuplefromOperands(Callback&& callback, Args&&... args) {
    return std::tuple_cat(callback(OperandsTypes{}, std::forward<Args>(args)...)...);
  }
  using Operands = std::tuple<OperandsTypes...>;
};

}  // namespace berberis::machine_insn_info

#endif  // BERBERIS_INTRINSICS_COMMON_MACHINE_INSN_INFO_H_
