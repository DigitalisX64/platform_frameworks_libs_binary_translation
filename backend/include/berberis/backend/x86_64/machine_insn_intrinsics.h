/*
 * Copyright (C) 2023 The Android Open Source Project
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

#ifndef BERBERIS_BACKEND_X86_64_MACHINE_INSN_INTRINSICS_H_
#define BERBERIS_BACKEND_X86_64_MACHINE_INSN_INTRINSICS_H_

#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

#include "berberis/backend/code_emitter.h"
#include "berberis/backend/common/machine_ir.h"
#include "berberis/backend/x86_64/code_debug.h"
#include "berberis/backend/x86_64/code_emit.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/base/dependent_false.h"
#include "berberis/base/stringprintf.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/intrinsics_bindings.h"

namespace berberis::x86_64 {

// tuple_cat for types, to help remove filtered out types below.
template <typename... Ts>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

// Predicate to determine whether type T has a RegisterClass alias.
template <class, class = void>
struct has_reg_class_impl : std::false_type {};
template <class T>
struct has_reg_class_impl<T, std::void_t<typename T::RegisterClass>> : std::true_type {};
template <typename T>
using has_reg_class_t = has_reg_class_impl<T>;

// Filter out types from Ts... that do not satisfy the predicate, collect them
// into a tuple.
template <template <typename> typename Predicate, typename... Ts>
using filter_t =
    tuple_cat_t<std::conditional_t<Predicate<Ts>::value, std::tuple<Ts>, std::tuple<>>...>;

// Convert Binding into constructor argument(s).
template <typename T, typename = void>
struct ConstructorArg;

// Immediates expand into their class type.
template <typename T>
struct ConstructorArg<ArgTraits<T>, std::enable_if_t<ArgTraits<T>::Class::kIsImmediate, void>> {
  using type = std::tuple<typename ArgTraits<T>::Class::Type>;
};

// Mem ops expand into base register and disp.
template <typename T>
struct ConstructorArg<ArgTraits<T>,
                      std::enable_if_t<!ArgTraits<T>::Class::kIsImmediate &&
                                           ArgTraits<T>::RegisterClass::kAsRegister == 'm',
                                       void>> {
  static_assert(ArgTraits<T>::kUsage == intrinsics::bindings::kDefEarlyClobber);
  // Need to emit base register AND disp.
  using type = std::tuple<MachineReg, int32_t>;
};

// Everything else expands into a MachineReg.
template <typename T>
struct ConstructorArg<ArgTraits<T>,
                      std::enable_if_t<!ArgTraits<T>::Class::kIsImmediate &&
                                           ArgTraits<T>::RegisterClass::kAsRegister != 'm',
                                       void>> {
  using type = std::tuple<MachineReg>;
};

template <typename T>
using constructor_one_arg_t = typename ConstructorArg<ArgTraits<T>>::type;

// Use this alias to generate constructor Args from bindings.
template <typename... T>
using constructor_args_t = tuple_cat_t<constructor_one_arg_t<T>...>;

// Predicate to determine whether type T is a memory access arg.
template <class, class = void>
struct is_mem_impl : std::false_type {};
template <class T>
struct is_mem_impl<
    T,
    std::enable_if_t<!T::Class::kIsImmediate && T::RegisterClass::kAsRegister == 'm', void>>
    : std::true_type {};
template <typename T>
using is_mem_t = is_mem_impl<T>;

template <typename... Bindings>
constexpr size_t mem_count_v = std::tuple_size_v<filter_t<is_mem_t, ArgTraits<Bindings>...>>;

template <size_t N, typename... Bindings>
constexpr bool has_n_mem_v = mem_count_v<Bindings...> > (N - 1);

template <typename IntrinsicBindingInfo>
class MachineInsn;

// Use specialization to extract the tuple parameter pack generated from constructor_args_t above.
template <auto kIntrinsic,
          auto kMacroInstruction,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestriction,
          typename PreciseNanOperationsHandling,
          bool kSideEffects,
          typename... InputArguments,
          typename... OutputArguments,
          typename... Bindings>
class MachineInsn<intrinsics::bindings::IntrinsicBindingInfo<kIntrinsic,
                                                             kMacroInstruction,
                                                             kMnemo,
                                                             GetOpcode,
                                                             CPUIDRestriction,
                                                             PreciseNanOperationsHandling,
                                                             kSideEffects,
                                                             std::tuple<InputArguments...>,
                                                             std::tuple<OutputArguments...>,
                                                             std::tuple<Bindings...>>>
    final : public MachineInsnX86_64 {
 private:
  template <typename>
  struct GenMachineInsnInfoT;
  // We want to filter out any bindings that are not used for Register args.
  using RegBindings = filter_t<has_reg_class_t, ArgTraits<Bindings>...>;

 public:
  // This static simplifies constructing this MachineInsn in intrinsic implementations.
  static constexpr MachineInsn* (MachineIRBuilder::*kGenFunc)(constructor_args_t<Bindings...>) =
      &MachineIRBuilder::template Gen<MachineInsn>;

  explicit MachineInsn(constructor_args_t<Bindings...> args) : MachineInsnX86_64(&kInfo) {
    std::apply(
        [this](auto... args) {
          this->ProcessArgs<0 /* reg_idx */, 0 /* disp_idx */, Bindings...>(args...);
        },
        args);
  }

  static constexpr MachineInsnInfo kInfo = GenMachineInsnInfoT<RegBindings>::value;

  static constexpr int NumRegOperands() { return kInfo.num_reg_operands; }
  static constexpr const MachineRegKind& RegKindAt(int i) { return kInfo.reg_kinds[i]; }

  std::string GetDebugString() const override {
    std::string s(kMnemo);
    // Code below assumes that we have at most two memory operands.
    static_assert(([]<typename Binding> {
                    if constexpr (!ArgTraits<Binding>::Class::kIsImmediate) {
                      return ArgTraits<Binding>::Class::kAsRegister == 'm';
                    }
                    return false;
                  }.template operator()<Bindings>() +
                   ... + 0) <= 2);
    size_t arg_idx{}, reg_idx{}, disp_idx{};
    (
        [&s, &arg_idx, &reg_idx, &disp_idx, this]<typename Binding> {
          s += " ";
          if (arg_idx > 0) {
            s += ", ";
          }
          if constexpr (ArgTraits<Binding>::Class::kIsImmediate) {
            // Without static_cast here compilation fails with extemely criptic error message:
            // error: implicit instantiation of undefined template 'berberis::InOutArg<0, 0, …>'
            //   …
            // machine_insn_intrinsics.h:161:18: note: in instantiation of template class
            //  'std::tuple<berberis::InOutArg<0, 0, …>, berberis::InArg<1, …>' requested here
            // 187            s += GetImmOperandDebugString(this);
            //
            // Same below.
            s += GetImmOperandDebugString(static_cast<const MachineInsnX86_64*>(this));
            arg_idx++;
          } else if constexpr (ArgTraits<Binding>::Class::kAsRegister == 'm') {
            if (disp_idx == 0) {
              s += GetBaseDispMemOperandDebugString(static_cast<const MachineInsnX86_64*>(this),
                                                    reg_idx);
            } else /* disp_idx == 1 */ {
              s += StringPrintf(
                  "[%s + 0x%x]",
                  GetRegOperandDebugString(static_cast<const MachineInsnX86_64*>(this), reg_idx)
                      .c_str(),
                  disp2());
            }
            arg_idx++, reg_idx++, disp_idx++;
          } else if constexpr (ArgTraits<Binding>::RegisterClass::kIsImplicitReg) {
            s += GetImplicitRegOperandDebugString(static_cast<const MachineInsnX86_64*>(this),
                                                  reg_idx);
            arg_idx++, reg_idx++;
          } else {
            s += GetRegOperandDebugString(static_cast<const MachineInsnX86_64*>(this), reg_idx);
            arg_idx++, reg_idx++;
          }
        }.template operator()<Bindings>(),
        ...);

    if (this->recovery_pc()) {
      s += StringPrintf(" <0x%" PRIxPTR ">", this->recovery_pc());
    }
    return s;
  }

  void Emit(CodeEmitter* as) const override {
    // Code below assumes that we have at most two memory operands.
    static_assert(([]<typename Binding> {
                    if constexpr (!ArgTraits<Binding>::Class::kIsImmediate) {
                      return ArgTraits<Binding>::Class::kAsRegister == 'm';
                    }
                    return false;
                  }.template operator()<Bindings>() +
                   ... + 0) <= 2);
    size_t reg_idx{}, disp_idx{};
    std::apply(
        kMacroInstruction,
        std::tuple_cat(
            std::tuple<CodeEmitter&>{*as}, [&reg_idx, &disp_idx, this]<typename Binding> {
              if constexpr (ArgTraits<Binding>::Class::kIsImmediate) {
                return std::tuple{
                    static_cast<constructor_one_arg_t<Binding>>(MachineInsnX86_64::imm())};
              } else if constexpr (ArgTraits<Binding>::RegisterClass::kAsRegister == 'x') {
                return std::tuple{GetXReg(this->RegAt(reg_idx++))};
              } else if constexpr (ArgTraits<Binding>::RegisterClass::kAsRegister == 'r' ||
                                   ArgTraits<Binding>::RegisterClass::kAsRegister == 'q') {
                return std::tuple{GetGReg(this->RegAt(reg_idx++))};
              } else if constexpr (ArgTraits<Binding>::RegisterClass::kAsRegister == 'm' &&
                                   ArgTraits<Binding>::kUsage ==
                                       intrinsics::bindings::kDefEarlyClobber) {
                disp_idx++;
                if (disp_idx == 1) {
                  return std::tuple{Assembler::Operand{.base = GetGReg(this->RegAt(reg_idx++)),
                                                       .disp = static_cast<int32_t>(disp())}};
                } else /* disp_idx == 2 */ {
                  return std::tuple{Assembler::Operand{.base = GetGReg(this->RegAt(reg_idx++)),
                                                       .disp = static_cast<int32_t>(disp2())}};
                }
              } else if constexpr (ArgTraits<Binding>::RegisterClass::kIsImplicitReg) {
                return std::tuple{};
              } else {
                static_assert(kDependentTypeFalse<Binding>);
              }
            }.template operator()<Bindings>()...));
  }

 private:
  template <size_t, size_t, typename...>
  void ProcessArgs() {}

  template <size_t reg_idx,
            size_t disp_idx,
            typename B,
            typename... BindingsRest,
            typename T,
            typename... Args>
  auto ProcessArgs(T arg, Args... args) -> std::enable_if_t<ArgTraits<B>::Class::kIsImmediate> {
    this->set_imm(arg);
    ProcessArgs<reg_idx, disp_idx, BindingsRest...>(args...);
  }

  template <size_t reg_idx,
            size_t disp_idx,
            typename B,
            typename... BindingsRest,
            typename T,
            typename... Args>
  auto ProcessArgs(T arg, Args... args)
      -> std::enable_if_t<ArgTraits<B>::RegisterClass::kAsRegister != 'm'> {
    static_assert(std::is_same_v<MachineReg, T>);
    this->SetRegAt(reg_idx, arg);
    ProcessArgs<reg_idx + 1, disp_idx, BindingsRest...>(args...);
  }

  template <size_t reg_idx,
            size_t disp_idx,
            typename B,
            typename... BindingsRest,
            typename T1,
            typename T2,
            typename... Args>
  auto ProcessArgs(T1 base, T2 disp, Args... args)
      -> std::enable_if_t<ArgTraits<B>::RegisterClass::kAsRegister == 'm'> {
    // Only tmp memory args are supported.
    static_assert(ArgTraits<B>::arg_info.arg_type == ArgInfo::TMP_ARG);
    this->SetRegAt(reg_idx, base);
    if constexpr (disp_idx == 0) {
      this->set_disp(disp);
    } else if constexpr (disp_idx == 1) {
      this->set_disp2(disp);
    } else {
      static_assert(kDependentValueFalse<disp_idx>);
    }
    ProcessArgs<reg_idx + 1, disp_idx + 1, BindingsRest...>(args...);
  }

  static constexpr auto GetInsnKind() {
    if constexpr (kSideEffects) {
      return kMachineInsnSideEffects;
    } else {
      return kMachineInsnDefault;
    }
  }

  template <typename T, typename = void>
  struct RegInfo;
  template <typename T>
  struct RegInfo<T, std::enable_if_t<T::RegisterClass::kAsRegister != 'm', void>> {
    static constexpr auto kRegClass = &T::RegisterClass::template kRegClass<MachineInsnX86_64>;
    static constexpr auto kRegKind = static_cast<MachineRegKind::StandardAccess>(T::kUsage);
    static_assert(MachineRegKind::kDef ==
                  static_cast<MachineRegKind::StandardAccess>(intrinsics::bindings::kDef));
    static_assert(MachineRegKind::kDefEarlyClobber == static_cast<MachineRegKind::StandardAccess>(
                                                          intrinsics::bindings::kDefEarlyClobber));
    static_assert(MachineRegKind::kUse ==
                  static_cast<MachineRegKind::StandardAccess>(intrinsics::bindings::kUse));
    static_assert(MachineRegKind::kUseDef ==
                  static_cast<MachineRegKind::StandardAccess>(intrinsics::bindings::kUseDef));
  };
  template <typename T>
  struct RegInfo<T, std::enable_if_t<T::RegisterClass::kAsRegister == 'm', void>> {
    static_assert(T::kUsage == intrinsics::bindings::kDefEarlyClobber);
    static constexpr auto kRegClass = &kGeneralReg32;
    static constexpr auto kRegKind = MachineRegKind::kUse;
  };

  template <typename... T>
  struct GenMachineInsnInfoT<std::tuple<T...>> {
    static constexpr MachineInsnInfo value =
        MachineInsnInfo({GetOpcode.template operator()<MachineOpcode>(),
                         sizeof...(T),
                         {{RegInfo<T>::kRegClass, RegInfo<T>::kRegKind}...},
                         GetInsnKind()});
  };
};

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_MACHINE_INSN_INTRINSICS_H_
