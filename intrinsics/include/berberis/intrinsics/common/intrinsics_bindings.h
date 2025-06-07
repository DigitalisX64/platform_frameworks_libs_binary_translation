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
#include "berberis/base/string_literal.h"
#include "berberis/device_arch_info/common/device_arch_info.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/type_traits.h"

namespace berberis {

namespace intrinsics::bindings {

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoNansOperation;
class PreciseNanOperationsHandling;
class ImpreciseNanOperationsHandling;

template <StringLiteral kIntrinsic, typename... Types>
class IntrinsicBindingInfo;

template <StringLiteral kIntrinsic_,
          auto kEmitInsnFunc_,
          StringLiteral kMnemo,
          auto GetOpcode,
          typename CPUIDRestriction_,
          typename PreciseNanOperationsHandling_,
          bool kSideEffects_,
          typename... InputArgumentsTypes,
          typename... OutputArgumentsTypes,
          typename... BindingsTypes,
          typename... OperandsTypes>
class IntrinsicBindingInfo<kIntrinsic_,
                           PreciseNanOperationsHandling_,
                           std::tuple<InputArgumentsTypes...>,
                           std::tuple<OutputArgumentsTypes...>,
                           std::tuple<BindingsTypes...>,
                           device_arch_info::DeviceInsnInfo<kEmitInsnFunc_,
                                                            kMnemo,
                                                            kSideEffects_,
                                                            GetOpcode,
                                                            CPUIDRestriction_,
                                                            std::tuple<OperandsTypes...>>>
    final {
 public:
  static constexpr auto kIntrinsic = kIntrinsic_;
  static constexpr auto kEmitInsnFunc = kEmitInsnFunc_;
  template <typename Opcode>
  static constexpr auto kOpcode = GetOpcode.template operator()<Opcode>();
  using CPUIDRestriction = CPUIDRestriction_;
  using PreciseNanOperationsHandling = PreciseNanOperationsHandling_;
  static constexpr bool kSideEffects = kSideEffects_;
  static constexpr const char* InputArgumentsTypeNames[] = {
      TypeTraits<InputArgumentsTypes>::kName...};
  static constexpr const char* OutputArgumentsTypeNames[] = {
      TypeTraits<OutputArgumentsTypes>::kName...};
  template <typename Callback, typename... Args>
  constexpr static void ProcessBindings(Callback&& callback, Args&&... args) {
    (callback.template operator()<BindingsTypes, OperandsTypes>(std::forward<Args>(args)...), ...);
  }
  template <typename Callback, typename... Args>
  constexpr static bool VerifyBindings(Callback&& callback, Args&&... args) {
    return (
        callback.template operator()<BindingsTypes, OperandsTypes>(std::forward<Args>(args)...) &&
        ...);
  }
  template <typename Callback, typename... Args>
  constexpr static auto MakeTuplefromBindings(Callback&& callback, Args&&... args) {
    return std::tuple_cat(
        callback.template operator()<BindingsTypes, OperandsTypes>(std::forward<Args>(args)...)...);
  }
  using InputArguments = std::tuple<InputArgumentsTypes...>;
  using OutputArguments = std::tuple<OutputArgumentsTypes...>;
  using Bindings = std::tuple<BindingsTypes...>;
  using Operands = std::tuple<OperandsTypes...>;
  using IntrinsicType = std::conditional_t<std::tuple_size_v<OutputArguments> == 0,
                                           void (*)(InputArgumentsTypes...),
                                           OutputArguments (*)(InputArgumentsTypes...)>;
  using DeviceInsnInfo = device_arch_info::
      DeviceInsnInfo<kEmitInsnFunc, kMnemo, kSideEffects_, GetOpcode, CPUIDRestriction, Operands>;
};

}  // namespace intrinsics::bindings

template <typename IntrinsicBindingInfo, typename AssemblerType>
constexpr void Check32BitRegistersAreZeroExtended(AssemblerType* as) {
  int id = 0;
  IntrinsicBindingInfo::ProcessBindings([&as, &id]<typename Binding, typename Operand> {
    if constexpr (!device_arch_info::kIsImmediate<Operand> &&
                  !device_arch_info::kIsFLAGS<Operand>) {
      if constexpr (HaveOutput(Binding::kArgInfo)) {
        static_assert(Operand::kUsage != device_arch_info::kUse);
        if constexpr (device_arch_info::kIsGeneralReg32<Operand>) {
          as->Check32BitRegisterIsZeroExtended(id);
        }
        id++;
      }
    }
  });
}

template <typename IntrinsicBindingInfo>
constexpr void AssignRegisterNumbers(int* register_numbers) {
  // Assign number for output (and temporary) arguments.
  std::size_t id = 0;
  int arg_counter = 0;
  IntrinsicBindingInfo::ProcessBindings(
      [&id, &arg_counter, &register_numbers]<typename Binding, typename Operand> {
        if constexpr (!device_arch_info::kIsImmediate<Operand> &&
                      !device_arch_info::kIsFLAGS<Operand>) {
          if constexpr (Operand::kUsage != device_arch_info::kUse) {
            register_numbers[arg_counter] = id++;
          }
          ++arg_counter;
        }
      });
  // Assign numbers for input arguments.
  arg_counter = 0;
  IntrinsicBindingInfo::ProcessBindings(
      [&id, &arg_counter, &register_numbers]<typename Binding, typename Operand> {
        if constexpr (!device_arch_info::kIsImmediate<Operand> &&
                      !device_arch_info::kIsFLAGS<Operand>) {
          if constexpr (Operand::kUsage == device_arch_info::kUse) {
            register_numbers[arg_counter] = id++;
          }
          ++arg_counter;
        }
      });
}

template <typename DeviceInsnInfo>
constexpr bool CheckIntrinsicHasFlagsBinding() {
  bool expect_flags = false;
  DeviceInsnInfo::ProcessBindings([&expect_flags]<typename Binding, typename Operand> {
    if constexpr (device_arch_info::kIsFLAGS<Operand>) {
      expect_flags = true;
    }
  });
  return expect_flags;
}

template <typename IntrinsicBindingInfo, typename AssemblerType>
constexpr void CallVerifierAssembler(AssemblerType* as, int* register_numbers) {
  int arg_counter = 0;
  IntrinsicBindingInfo::ProcessBindings(
      [&arg_counter, &as, register_numbers]<typename Binding, typename Operand> {
        if constexpr (device_arch_info::kIsImplicitReg<Operand> &&
                      !device_arch_info::kIsFLAGS<Operand>) {
          as->*(Operand::Class::template kAssemblerRegisterPointer<AssemblerType>) =
              typename AssemblerType::Register{register_numbers[arg_counter], Operand::kUsage};
        }
        ++arg_counter;
      });
  // Macroassembler constants register points to the constant pool. Intrinsics can read from it
  // but shouldn't change it's address, that's why it's always kUse.
  as->gpr_macroassembler_constants =
      typename AssemblerType::Register{arg_counter, device_arch_info::kUse};
  arg_counter = 0;
  int scratch_counter = 0;
  std::apply(
      IntrinsicBindingInfo::kEmitInsnFunc,
      std::tuple_cat(
          std::tuple<AssemblerType&>{*as},
          IntrinsicBindingInfo::MakeTuplefromBindings(
              [&as,
               &arg_counter,
               &scratch_counter,
               register_numbers]<typename Binding, typename Operand> {
                if constexpr (device_arch_info::kIsImmediate<Operand>) {
                  // TODO(b/394278175): We don't have access to the value of the immediate argument
                  // here. The value of the immediate argument often decides which instructions in
                  // an intrinsic are called, by being used in conditional statements. We need to
                  // make sure that all possible instructions in the intrinsic are executed when
                  // using VerifierAssembler on inline-only intrinsics. For now, we set immediate
                  // argument to 2, since it generally covers most instructions in inline-only
                  // intrinsics.
                  return std::tuple{2};
                } else if constexpr (device_arch_info::kIsMemoryOperand<Operand>) {
                  static_assert(Operand::kUsage == device_arch_info::kDefEarlyClobber);
                  if (scratch_counter == 0) {
                    as->gpr_macroassembler_scratch = typename AssemblerType::Register(
                        arg_counter++, device_arch_info::kDefEarlyClobber);
                  } else if (scratch_counter == 1) {
                    as->gpr_macroassembler_scratch2 = typename AssemblerType::Register(
                        arg_counter++, device_arch_info::kDefEarlyClobber);
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
                } else {
                  if constexpr (!device_arch_info::kIsFLAGS<Operand>) {
                    if constexpr (device_arch_info::kIsImplicitReg<Operand>) {
                      ++arg_counter;
                      return std::tuple{};
                    } else {
                      using RegisterClass = Operand::Class;
                      if constexpr (RegisterClass::kAsRegister == 'q' ||
                                    RegisterClass::kAsRegister == 'r') {
                        return std::tuple{typename AssemblerType::Register{
                            register_numbers[arg_counter++], Operand::kUsage}};
                      } else if constexpr (RegisterClass::kAsRegister == 'x') {
                        return std::tuple{typename AssemblerType::XRegister{
                            register_numbers[arg_counter++], Operand::kUsage}};
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
