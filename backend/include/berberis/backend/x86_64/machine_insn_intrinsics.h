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

// Predicate to determine whether type T uses a register alias.
template <class, class = void>
struct has_reg_class_impl : std::false_type {};
template <class T>
struct has_reg_class_impl<T, std::enable_if_t<!T::Class::kIsImmediate>> : std::true_type {};
template <typename T>
using has_reg_class_t = has_reg_class_impl<T>;

// Filter out types from Ts... that do not satisfy the predicate, collect them
// into a tuple.
template <template <typename> typename Predicate, typename... Ts>
using filter_t =
    tuple_cat_t<std::conditional_t<Predicate<Ts>::value, std::tuple<Ts>, std::tuple<>>...>;

// Convert Operands into constructor argument(s).
template <typename T, typename = void>
struct ConstructorArg;

// Immediates expand into their class type.
template <typename O>
struct ConstructorArg<O, std::enable_if_t<O::Class::kIsImmediate>> {
  using type = std::tuple<typename O::Class::Type>;
};

// Mem ops expand into base register and disp.
template <typename O>
struct ConstructorArg<O,
                      std::enable_if_t<!O::Class::kIsImmediate && O::Class::kAsRegister == 'm'>> {
  static_assert(O::kUsage == machine_insn_info::kDefEarlyClobber);
  // Need to emit base register AND disp.
  using type = std::tuple<MachineReg, int32_t>;
};

// Everything else expands into a MachineReg.
template <typename O>
struct ConstructorArg<O,
                      std::enable_if_t<!O::Class::kIsImmediate && O::Class::kAsRegister != 'm'>> {
  using type = std::tuple<MachineReg>;
};

template <typename O>
using constructor_one_arg_t = typename ConstructorArg<O>::type;

// Use this alias to generate constructor Args from operands.
template <typename... O>
using constructor_args_t = tuple_cat_t<constructor_one_arg_t<O>...>;

// Predicate to determine whether type T is a memory access arg.
template <class, class = void>
struct is_mem_impl : std::false_type {};
template <class T>
struct is_mem_impl<T, std::enable_if_t<!T::Class::kIsImmediate && T::Class::kAsRegister == 'm'>>
    : std::true_type {};
template <typename T>
using is_mem_t = is_mem_impl<T>;

template <typename... Operands>
constexpr size_t mem_count_v = std::tuple_size_v<filter_t<is_mem_t, Operands...>>;

template <size_t N, typename... Operands>
constexpr bool has_n_mem_v = mem_count_v<Operands...> > (N - 1);

template <typename IntrinsicBindingInfo, bool kSideEffects>
class MachineInsn;

// Use specialization to extract the tuple parameter pack generated from constructor_args_t above.
template <auto kMacroInstruction,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestriction,
          typename... Operands,
          bool kSideEffects>
class MachineInsn<machine_insn_info::AsmCallInfo<kMacroInstruction,
                                                 kMnemo,
                                                 GetOpcode,
                                                 CPUIDRestriction,
                                                 std::tuple<Operands...>>,
                  kSideEffects>
    final : public MachineInsnX86_64 {
 private:
  template <typename>
  struct GenMachineInsnInfoT;
  // We want to filter out any operands that are not used for Register args.
  using RegOperands = filter_t<has_reg_class_t, Operands...>;

 public:
  // This static simplifies constructing this MachineInsn in intrinsic implementations.
  static constexpr MachineInsn* (MachineIRBuilder::*kGenFunc)(constructor_args_t<Operands...>) =
      &MachineIRBuilder::template Gen<MachineInsn>;

  explicit MachineInsn(constructor_args_t<Operands...> args) : MachineInsnX86_64(&kInfo) {
    std::apply(
        [this](auto... args) {
          this->ProcessArgs<0 /* reg_idx */, 0 /* disp_idx */, Operands...>(args...);
        },
        args);
  }

  static constexpr MachineInsnInfo kInfo = GenMachineInsnInfoT<RegOperands>::value;

  static constexpr int NumRegOperands() { return kInfo.num_reg_operands; }
  static constexpr const MachineRegKind& RegKindAt(int i) { return kInfo.reg_kinds[i]; }

  std::string GetDebugString() const override {
    std::string s(kMnemo);
    // Code below assumes that we have at most two memory operands.
    static_assert(([]<typename Operand> {
                    if constexpr (!Operand::Class::kIsImmediate) {
                      return Operand::Class::kAsRegister == 'm';
                    }
                    return false;
                  }.template operator()<Operands>() +
                   ... + 0) <= 2);
    size_t arg_idx{}, reg_idx{}, disp_idx{};
    (
        [&s, &arg_idx, &reg_idx, &disp_idx, this]<typename Operand> {
          s += " ";
          if (arg_idx > 0) {
            s += ", ";
          }
          if constexpr (Operand::Class::kIsImmediate) {
            s += GetImmOperandDebugString(this);
            arg_idx++;
          } else if constexpr (Operand::Class::kAsRegister == 'm') {
            if (disp_idx == 0) {
              s += GetBaseDispMemOperandDebugString(this, reg_idx);
            } else /* disp_idx == 1 */ {
              s += StringPrintf(
                  "[%s + 0x%x]", GetRegOperandDebugString(this, reg_idx).c_str(), disp2());
            }
            arg_idx++, reg_idx++, disp_idx++;
          } else if constexpr (Operand::Class::kIsImplicitReg) {
            s += GetImplicitRegOperandDebugString(this, reg_idx);
            arg_idx++, reg_idx++;
          } else {
            s += GetRegOperandDebugString(this, reg_idx);
            arg_idx++, reg_idx++;
          }
        }.template operator()<Operands>(),
        ...);

    if (this->recovery_pc()) {
      s += StringPrintf(" <0x%" PRIxPTR ">", this->recovery_pc());
    }
    return s;
  }

  void Emit(CodeEmitter* as) const override {
    // Code below assumes that we have at most two memory operands.
    static_assert(([]<typename Operand> {
                    if constexpr (!Operand::Class::kIsImmediate) {
                      return Operand::Class::kAsRegister == 'm';
                    }
                    return false;
                  }.template operator()<Operands>() +
                   ... + 0) <= 2);
    size_t reg_idx{}, disp_idx{};
    std::apply(
        kMacroInstruction,
        std::tuple_cat(
            std::tuple<CodeEmitter&>{*as}, [&reg_idx, &disp_idx, this]<typename Operand> {
              // Suppress spurious warnings.
              // See https://github.com/llvm/llvm-project/issues/34798#issuecomment-980989495
              (void)reg_idx;
              (void)disp_idx;
              if constexpr (Operands::Class::kIsImmediate) {
                return std::tuple{
                    static_cast<constructor_one_arg_t<Operands>>(MachineInsnX86_64::imm())};
              } else if constexpr (Operand::Class::kAsRegister == 'x') {
                return std::tuple{GetXReg(this->RegAt(reg_idx++))};
              } else if constexpr (Operand::Class::kAsRegister == 'r' ||
                                   Operand::Class::kAsRegister == 'q') {
                return std::tuple{GetGReg(this->RegAt(reg_idx++))};
              } else if constexpr (Operand::Class::kAsRegister == 'm' &&
                                   Operand::kUsage == machine_insn_info::kDefEarlyClobber) {
                disp_idx++;
                if (disp_idx == 1) {
                  return std::tuple{Assembler::Operand{.base = GetGReg(this->RegAt(reg_idx++)),
                                                       .disp = static_cast<int32_t>(disp())}};
                } else /* disp_idx == 2 */ {
                  return std::tuple{Assembler::Operand{.base = GetGReg(this->RegAt(reg_idx++)),
                                                       .disp = static_cast<int32_t>(disp2())}};
                }
              } else if constexpr (Operand::Class::kIsImplicitReg) {
                return std::tuple{};
              } else {
                static_assert(kDependentTypeFalse<Operand>);
              }
            }.template operator()<Operands>()...));
  }

 private:
  template <size_t, size_t, typename...>
  void ProcessArgs() {}

  template <size_t reg_idx,
            size_t disp_idx,
            typename O,
            typename... OperandsRest,
            typename T,
            typename... Args>
  auto ProcessArgs(T arg, Args... args) -> std::enable_if_t<O::Class::kIsImmediate> {
    this->set_imm(arg);
    ProcessArgs<reg_idx, disp_idx, OperandsRest...>(args...);
  }

  template <size_t reg_idx,
            size_t disp_idx,
            typename O,
            typename... OperandsRest,
            typename T,
            typename... Args>
  auto ProcessArgs(T arg, Args... args) -> std::enable_if_t<O::Class::kAsRegister != 'm'> {
    static_assert(std::is_same_v<MachineReg, T>);
    this->SetRegAt(reg_idx, arg);
    ProcessArgs<reg_idx + 1, disp_idx, OperandsRest...>(args...);
  }

  template <size_t reg_idx,
            size_t disp_idx,
            typename O,
            typename... OperandsRest,
            typename T1,
            typename T2,
            typename... Args>
  auto ProcessArgs(T1 base,
                   T2 disp,
                   Args... args) -> std::enable_if_t<O::Class::kAsRegister == 'm'> {
    // Only tmp memory args are supported.
    this->SetRegAt(reg_idx, base);
    if constexpr (disp_idx == 0) {
      this->set_disp(disp);
    } else if constexpr (disp_idx == 1) {
      this->set_disp2(disp);
    } else {
      static_assert(kDependentValueFalse<disp_idx>);
    }
    ProcessArgs<reg_idx + 1, disp_idx + 1, OperandsRest...>(args...);
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
  struct RegInfo<T, std::enable_if_t<T::Class::kAsRegister != 'm', void>> {
    static constexpr auto kRegClass = &T::Class::template kRegClass<MachineInsnX86_64>;
    static constexpr auto kRegKind = static_cast<MachineRegKind::StandardAccess>(T::kUsage);
    static_assert(MachineRegKind::kDef ==
                  static_cast<MachineRegKind::StandardAccess>(machine_insn_info::kDef));
    static_assert(MachineRegKind::kDefEarlyClobber ==
                  static_cast<MachineRegKind::StandardAccess>(machine_insn_info::kDefEarlyClobber));
    static_assert(MachineRegKind::kUse ==
                  static_cast<MachineRegKind::StandardAccess>(machine_insn_info::kUse));
    static_assert(MachineRegKind::kUseDef ==
                  static_cast<MachineRegKind::StandardAccess>(machine_insn_info::kUseDef));
  };
  template <typename T>
  struct RegInfo<T, std::enable_if_t<T::Class::kAsRegister == 'm', void>> {
    static_assert(T::kUsage == machine_insn_info::kDefEarlyClobber);
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
