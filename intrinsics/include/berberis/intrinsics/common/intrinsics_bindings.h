/*
 * Copyright (C) 2024 The Android Open Source Project
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

#ifndef BERBERIS_INTRINSICS_COMMON_INTRINSICS_BINDINGS_H_
#define BERBERIS_INTRINSICS_COMMON_INTRINSICS_BINDINGS_H_

#include <cstdint>

#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/type_traits.h"

namespace berberis {

namespace intrinsics::bindings {

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

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class Def;
class DefEarlyClobber;
class Use;
class UseDef;

template <typename Tag, typename MachineRegKind>
constexpr auto ToRegKind() {
  if constexpr (std::is_same_v<Tag, Def>) {
    return MachineRegKind::kDef;
  } else if constexpr (std::is_same_v<Tag, DefEarlyClobber>) {
    return MachineRegKind::kDefEarlyClobber;
  } else if constexpr (std::is_same_v<Tag, Use>) {
    return MachineRegKind::kUse;
  } else if constexpr (std::is_same_v<Tag, UseDef>) {
    return MachineRegKind::kUseDef;
  } else {
    static_assert(kDependentTypeFalse<Tag>);
  }
}

template <typename Tag, typename MachineRegKind>
inline constexpr auto kRegKind = ToRegKind<Tag, MachineRegKind>();

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoCPUIDRestriction;  // All CPUs have at least “no CPUID restriction” mode.

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoNansOperation;
class PreciseNanOperationsHandling;
class ImpreciseNanOperationsHandling;

template <auto kIntrinsicTemplateName,
          auto kMacroInstructionTemplateName,
          auto kMnemo,
          typename GetOpcode,
          typename CPUIDRestrictionTemplateValue,
          typename PreciseNanOperationsHandlingTemplateValue,
          bool kSideEffectsTemplateValue,
          typename... Types>
class AsmCallInfo;

template <auto kIntrinsicTemplateName,
          auto kMacroInstructionTemplateName,
          auto kMnemo,
          typename GetOpcode,
          typename CPUIDRestrictionTemplateValue,
          typename PreciseNanOperationsHandlingTemplateValue,
          bool kSideEffectsTemplateValue,
          typename... InputArgumentsTypes,
          typename... OutputArgumentsTypes,
          typename... BindingsTypes>
class AsmCallInfo<kIntrinsicTemplateName,
                  kMacroInstructionTemplateName,
                  kMnemo,
                  GetOpcode,
                  CPUIDRestrictionTemplateValue,
                  PreciseNanOperationsHandlingTemplateValue,
                  kSideEffectsTemplateValue,
                  std::tuple<InputArgumentsTypes...>,
                  std::tuple<OutputArgumentsTypes...>,
                  BindingsTypes...>
    final {
 public:
  static constexpr auto kIntrinsic = kIntrinsicTemplateName;
  static constexpr auto kMacroInstruction = kMacroInstructionTemplateName;
  // TODO(b/260725458): Use lambda template argument after C++20 becomes available.
  template <typename Opcode>
  static constexpr auto kOpcode = GetOpcode{}.template operator()<Opcode>();
  using CPUIDRestriction = CPUIDRestrictionTemplateValue;
  using PreciseNanOperationsHandling = PreciseNanOperationsHandlingTemplateValue;
  static constexpr bool kSideEffects = kSideEffectsTemplateValue;
  static constexpr const char* InputArgumentsTypeNames[] = {
      TypeTraits<InputArgumentsTypes>::kName...};
  static constexpr const char* OutputArgumentsTypeNames[] = {
      TypeTraits<OutputArgumentsTypes>::kName...};
  template <typename Callback, typename... Args>
  constexpr static void ProcessBindings(Callback&& callback, Args&&... args) {
    (callback(ArgTraits<BindingsTypes>(), std::forward<Args>(args)...), ...);
  }
  template <typename Callback, typename... Args>
  constexpr static bool VerifyBindings(Callback&& callback, Args&&... args) {
    return (callback(ArgTraits<BindingsTypes>(), std::forward<Args>(args)...) && ...);
  }
  template <typename Callback, typename... Args>
  constexpr static auto MakeTuplefromBindings(Callback&& callback, Args&&... args) {
    return std::tuple_cat(callback(ArgTraits<BindingsTypes>(), std::forward<Args>(args)...)...);
  }
  using InputArguments = std::tuple<InputArgumentsTypes...>;
  using OutputArguments = std::tuple<OutputArgumentsTypes...>;
  using Bindings = std::tuple<BindingsTypes...>;
  using IntrinsicType = std::conditional_t<std::tuple_size_v<OutputArguments> == 0,
                                           void (*)(InputArgumentsTypes...),
                                           OutputArguments (*)(InputArgumentsTypes...)>;
  template <template <typename, auto, auto, typename...> typename MachineInsnType,
            template <typename...>
            typename ConstructorArgs,
            typename Opcode>
  using MachineInsn = MachineInsnType<AsmCallInfo,
                                      kMnemo,
                                      kOpcode<Opcode>,
                                      ConstructorArgs<BindingsTypes...>,
                                      BindingsTypes...>;
};

}  // namespace intrinsics::bindings

template <typename AsmCallInfo>
constexpr void AssignRegisterNumbers(int* register_numbers) {
  // Assign number for output (and temporary) arguments.
  std::size_t id = 0;
  int arg_counter = 0;
  AsmCallInfo::ProcessBindings([&id, &arg_counter, &register_numbers](auto arg) {
    if constexpr (!IsImmediate(decltype(arg)::arg_info)) {
      using RegisterClass = typename decltype(arg)::RegisterClass;
      if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
        if constexpr (!std::is_same_v<typename decltype(arg)::Usage, intrinsics::bindings::Use>) {
          register_numbers[arg_counter] = id++;
        }
        ++arg_counter;
      }
    }
  });
  // Assign numbers for input arguments.
  arg_counter = 0;
  AsmCallInfo::ProcessBindings([&id, &arg_counter, &register_numbers](auto arg) {
    if constexpr (!IsImmediate(decltype(arg)::arg_info)) {
      using RegisterClass = typename decltype(arg)::RegisterClass;
      if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
        if constexpr (std::is_same_v<typename decltype(arg)::Usage, intrinsics::bindings::Use>) {
          register_numbers[arg_counter] = id++;
        }
        ++arg_counter;
      }
    }
  });
}

template <typename AsmCallInfo>
constexpr void CheckIntrinsicHasFlagsBinding(bool& expect_flags) {
  AsmCallInfo::ProcessBindings([&expect_flags](auto arg) {
    if constexpr (!IsImmediate(decltype(arg)::arg_info)) {
      using RegisterClass = typename decltype(arg)::RegisterClass;
      if constexpr (std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
        expect_flags = true;
      }
    }
  });
}

template <typename AsmCallInfo, typename AssemblerType>
constexpr void CallAssembler(AssemblerType* as, int* register_numbers) {
  int arg_counter = 0;
  AsmCallInfo::ProcessBindings([&arg_counter, &as, register_numbers](auto arg) {
    if constexpr (!IsImmediate(decltype(arg)::arg_info)) {
      using RegisterClass = typename decltype(arg)::RegisterClass;
      if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
        if constexpr (RegisterClass::kAsRegister != 'm') {
          if constexpr (RegisterClass::kIsImplicitReg) {
            if constexpr (RegisterClass::kAsRegister == 'a') {
              as->gpr_a = typename AssemblerType::Register(register_numbers[arg_counter]);
            } else if constexpr (RegisterClass::kAsRegister == 'b') {
              as->gpr_b = typename AssemblerType::Register(register_numbers[arg_counter]);
            } else if constexpr (RegisterClass::kAsRegister == 'c') {
              as->gpr_c = typename AssemblerType::Register(register_numbers[arg_counter]);
            } else {
              static_assert(RegisterClass::kAsRegister == 'd');
              as->gpr_d = typename AssemblerType::Register(register_numbers[arg_counter]);
            }
          }
        }
        ++arg_counter;
      }
    }
  });
  as->gpr_macroassembler_constants = typename AssemblerType::Register(arg_counter);
  arg_counter = 0;
  int scratch_counter = 0;
  std::apply(
      AsmCallInfo::kMacroInstruction,
      std::tuple_cat(
          std::tuple<AssemblerType&>{*as},
          AsmCallInfo::MakeTuplefromBindings(
              [&as, &arg_counter, &scratch_counter, register_numbers](auto arg) {
                if constexpr (IsImmediate(decltype(arg)::arg_info)) {
                  // TODO(b/394278175): We don't have access to the value of the immediate argument
                  // here. The value of the immediate argument often decides which instructions in
                  // an intrinsic are called, by being used in conditional statements. We need to
                  // make sure that all possible instructions in the intrinsic are executed when
                  // using VerifierAssembler on inline-only intrinsics. For now, we set immediate
                  // argument to 2, since it generally covers most instructions in inline-only
                  // intrinsics.
                  return std::tuple{2};
                } else {
                  using RegisterClass = typename decltype(arg)::RegisterClass;
                  if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
                    if constexpr (RegisterClass::kAsRegister == 'm') {
                      if (scratch_counter == 0) {
                        as->gpr_macroassembler_scratch =
                            typename AssemblerType::Register(arg_counter++);
                      } else if (scratch_counter == 1) {
                        as->gpr_macroassembler_scratch2 =
                            typename AssemblerType::Register(arg_counter++);
                      } else {
                        FATAL("Only two scratch registers are supported for now");
                      }
                      // Note: as->gpr_scratch in combination with offset is treated by text
                      // assembler specially.  We rely on offset set here to be the same as
                      // scratch2 address in scratch buffer.
                      return std::tuple{typename AssemblerType::Operand{
                          .base = as->gpr_scratch,
                          .disp = static_cast<int32_t>(config::kScratchAreaSlotSize *
                                                       scratch_counter++)}};
                    } else if constexpr (RegisterClass::kIsImplicitReg) {
                      ++arg_counter;
                      return std::tuple{};
                    } else {
                      return std::tuple{register_numbers[arg_counter++]};
                    }
                  } else {
                    return std::tuple{};
                  }
                }
              })));
}

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_COMMON_INTRINSICS_BINDINGS_H_
