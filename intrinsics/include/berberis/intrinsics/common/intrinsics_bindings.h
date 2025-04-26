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
#include "berberis/intrinsics/common/machine_insn_info.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/type_traits.h"

namespace berberis {

namespace intrinsics::bindings {

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoNansOperation;
class PreciseNanOperationsHandling;
class ImpreciseNanOperationsHandling;

template <auto kIntrinsicTemplateName,
          auto kMacroInstructionTemplateName,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestrictionTemplateValue,
          typename PreciseNanOperationsHandlingTemplateValue,
          bool kSideEffectsTemplateValue,
          typename... Types>
class IntrinsicBindingInfo;

template <auto kIntrinsicTemplateName,
          auto kMacroInstructionTemplateName,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestrictionTemplateValue,
          typename PreciseNanOperationsHandlingTemplateValue,
          bool kSideEffectsTemplateValue,
          typename... InputArgumentsTypes,
          typename... OutputArgumentsTypes,
          typename... BindingsTypes>
class IntrinsicBindingInfo<kIntrinsicTemplateName,
                           kMacroInstructionTemplateName,
                           kMnemo,
                           GetOpcode,
                           CPUIDRestrictionTemplateValue,
                           PreciseNanOperationsHandlingTemplateValue,
                           kSideEffectsTemplateValue,
                           std::tuple<InputArgumentsTypes...>,
                           std::tuple<OutputArgumentsTypes...>,
                           std::tuple<BindingsTypes...>>
    final {
 public:
  static constexpr auto kIntrinsic = kIntrinsicTemplateName;
  static constexpr auto kMacroInstruction = kMacroInstructionTemplateName;
  template <typename Opcode>
  static constexpr auto kOpcode = GetOpcode.template operator()<Opcode>();
  using CPUIDRestriction = CPUIDRestrictionTemplateValue;
  using PreciseNanOperationsHandling = PreciseNanOperationsHandlingTemplateValue;
  static constexpr bool kSideEffects = kSideEffectsTemplateValue;
  static constexpr const char* InputArgumentsTypeNames[] = {
      TypeTraits<InputArgumentsTypes>::kName...};
  static constexpr const char* OutputArgumentsTypeNames[] = {
      TypeTraits<OutputArgumentsTypes>::kName...};
  template <typename Callback, typename... Args>
  constexpr static void ProcessBindings(Callback&& callback, Args&&... args) {
    (callback(BindingsTypes(), std::forward<Args>(args)...), ...);
  }
  template <typename Callback, typename... Args>
  constexpr static bool VerifyBindings(Callback&& callback, Args&&... args) {
    return (callback(BindingsTypes(), std::forward<Args>(args)...) && ...);
  }
  template <typename Callback, typename... Args>
  constexpr static auto MakeTuplefromBindings(Callback&& callback, Args&&... args) {
    return std::tuple_cat(callback(BindingsTypes(), std::forward<Args>(args)...)...);
  }
  using InputArguments = std::tuple<InputArgumentsTypes...>;
  using OutputArguments = std::tuple<OutputArgumentsTypes...>;
  using Bindings = std::tuple<BindingsTypes...>;
  using IntrinsicType = std::conditional_t<std::tuple_size_v<OutputArguments> == 0,
                                           void (*)(InputArgumentsTypes...),
                                           OutputArguments (*)(InputArgumentsTypes...)>;
};

}  // namespace intrinsics::bindings

template <typename IntrinsicBindingInfo>
constexpr void AssignRegisterNumbers(int* register_numbers) {
  // Assign number for output (and temporary) arguments.
  std::size_t id = 0;
  int arg_counter = 0;
  IntrinsicBindingInfo::ProcessBindings([&id, &arg_counter, &register_numbers](auto arg) {
    if constexpr (!IsImmediate(decltype(arg)::arg_info)) {
      using RegisterClass = typename decltype(arg)::RegisterClass;
      if constexpr (!std::is_same_v<RegisterClass, machine_insn_info::FLAGS>) {
        if constexpr (decltype(arg)::kUsage != machine_insn_info::kUse) {
          register_numbers[arg_counter] = id++;
        }
        ++arg_counter;
      }
    }
  });
  // Assign numbers for input arguments.
  arg_counter = 0;
  IntrinsicBindingInfo::ProcessBindings([&id, &arg_counter, &register_numbers](auto arg) {
    if constexpr (!IsImmediate(decltype(arg)::arg_info)) {
      using RegisterClass = typename decltype(arg)::RegisterClass;
      if constexpr (!std::is_same_v<RegisterClass, machine_insn_info::FLAGS>) {
        if constexpr (decltype(arg)::kUsage == machine_insn_info::kUse) {
          register_numbers[arg_counter] = id++;
        }
        ++arg_counter;
      }
    }
  });
}

template <typename AsmCallInfo>
constexpr bool CheckIntrinsicHasFlagsBinding() {
  bool expect_flags = false;
  AsmCallInfo::ProcessBindings([&expect_flags](auto arg) constexpr {
    if constexpr (!IsImmediate(decltype(arg)::arg_info)) {
      using RegisterClass = typename decltype(arg)::RegisterClass;
      if constexpr (std::is_same_v<RegisterClass, machine_insn_info::FLAGS>) {
        expect_flags = true;
      }
    }
  });
  return expect_flags;
}

template <typename IntrinsicBindingInfo, typename AssemblerType>
constexpr void CallVerifierAssembler(AssemblerType* as, int* register_numbers) {
  int arg_counter = 0;
  IntrinsicBindingInfo::ProcessBindings([&arg_counter, &as, register_numbers](auto arg) {
    if constexpr (!IsImmediate(decltype(arg)::arg_info)) {
      using RegisterClass = typename decltype(arg)::RegisterClass;
      if constexpr (!std::is_same_v<RegisterClass, machine_insn_info::FLAGS>) {
        if constexpr (RegisterClass::kAsRegister != 'm') {
          if constexpr (RegisterClass::kIsImplicitReg) {
            if constexpr (RegisterClass::kAsRegister == 'a') {
              as->gpr_a =
                  typename AssemblerType::Register{register_numbers[arg_counter], arg.kUsage};
            } else if constexpr (RegisterClass::kAsRegister == 'b') {
              as->gpr_b =
                  typename AssemblerType::Register{register_numbers[arg_counter], arg.kUsage};
            } else if constexpr (RegisterClass::kAsRegister == 'c') {
              as->gpr_c =
                  typename AssemblerType::Register{register_numbers[arg_counter], arg.kUsage};
            } else {
              static_assert(RegisterClass::kAsRegister == 'd');
              as->gpr_d =
                  typename AssemblerType::Register{register_numbers[arg_counter], arg.kUsage};
            }
          }
        }
        ++arg_counter;
      }
    }
  });
  // Macroassembler constants register points to the constant pool. Intrinsics can read from it
  // but shouldn't change it's address, that's why it's always kUse.
  as->gpr_macroassembler_constants =
      typename AssemblerType::Register{arg_counter, machine_insn_info::kUse};
  arg_counter = 0;
  int scratch_counter = 0;
  std::apply(
      IntrinsicBindingInfo::kMacroInstruction,
      std::tuple_cat(
          std::tuple<AssemblerType&>{*as},
          IntrinsicBindingInfo::MakeTuplefromBindings([&as,
                                                       &arg_counter,
                                                       &scratch_counter,
                                                       register_numbers](auto arg) {
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
              if constexpr (!std::is_same_v<RegisterClass, machine_insn_info::FLAGS>) {
                if constexpr (RegisterClass::kAsRegister == 'm') {
                  static_assert(decltype(arg)::kUsage == machine_insn_info::kDefEarlyClobber);
                  if (scratch_counter == 0) {
                    as->gpr_macroassembler_scratch = typename AssemblerType::Register(
                        arg_counter++, machine_insn_info::kDefEarlyClobber);
                  } else if (scratch_counter == 1) {
                    as->gpr_macroassembler_scratch2 = typename AssemblerType::Register(
                        arg_counter++, machine_insn_info::kDefEarlyClobber);
                  } else {
                    FATAL("Only two scratch registers are supported for now");
                  }
                  // Note: as->gpr_scratch in combination with offset is treated by text
                  // assembler specially.  We rely on offset set here to be the same as
                  // scratch2 address in scratch buffer.
                  return std::tuple{typename AssemblerType::Operand{
                      .base = as->gpr_scratch,
                      .disp =
                          static_cast<int32_t>(config::kScratchAreaSlotSize * scratch_counter++)}};
                } else if constexpr (RegisterClass::kIsImplicitReg) {
                  ++arg_counter;
                  return std::tuple{};
                } else {
                  if constexpr (RegisterClass::kAsRegister == 'q' ||
                                RegisterClass::kAsRegister == 'r') {
                    return std::tuple{typename AssemblerType::Register{
                        register_numbers[arg_counter++], arg.kUsage}};
                  } else if constexpr (RegisterClass::kAsRegister == 'x') {
                    return std::tuple{typename AssemblerType::XRegister{
                        register_numbers[arg_counter++], arg.kUsage}};
                  } else {
                    static_assert(kDependentValueFalse<RegisterClass::kAsRegister>);
                  }
                }
              } else {
                return std::tuple{};
              }
            }
          })));
}

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_COMMON_INTRINSICS_BINDINGS_H_
