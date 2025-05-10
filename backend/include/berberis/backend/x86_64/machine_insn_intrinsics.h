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
  // Note: memory operands accept register and offset, thus they are included.
  using RegOperands = decltype(std::tuple_cat(
      std::declval<std::conditional_t<machine_insn_info::kIsRegister<Operands> ||
                                          machine_insn_info::kIsMemoryOperand<Operands>,
                                      std::tuple<Operands>,
                                      std::tuple<>>>()...));
  // Note: immediates accept appropriate type, register operands includes only register while memory
  // operand needs both base register and offset.
  using ConstructorArgs = decltype(std::tuple_cat(
      std::declval<std::conditional_t<machine_insn_info::kIsImmediate<Operands>,
                                      std::tuple<typename Operands::Class::Type>,
                                      std::conditional_t<machine_insn_info::kIsRegister<Operands>,
                                                         std::tuple<MachineReg>,
                                                         std::tuple<MachineReg, int32_t>>>>()...));

 public:
  // This static simplifies constructing this MachineInsn in intrinsic implementations.
  static constexpr MachineInsn* (MachineIRBuilder::*kGenFunc)(ConstructorArgs) =
      &MachineIRBuilder::template Gen<MachineInsn>;

  explicit MachineInsn(ConstructorArgs args) : MachineInsnX86_64(&kInfo) {
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
    static_assert((machine_insn_info::kIsMemoryOperand<Operands> + ... + 0) <= 2);
    size_t arg_idx{}, reg_idx{}, disp_idx{};
    (
        [&s, &arg_idx, &reg_idx, &disp_idx, this]<typename Operand> {
          s += " ";
          if (arg_idx > 0) {
            s += ", ";
          }
          if constexpr (machine_insn_info::kIsImmediate<Operand>) {
            s += GetImmOperandDebugString(this);
            arg_idx++;
          } else if constexpr (machine_insn_info::kIsMemoryOperand<Operand>) {
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
    static_assert((machine_insn_info::kIsMemoryOperand<Operands> + ... + 0) <= 2);
    size_t reg_idx{}, disp_idx{};
    std::apply(
        kMacroInstruction,
        std::tuple_cat(
            std::tuple<CodeEmitter&>{*as}, [&reg_idx, &disp_idx, this]<typename Operand> {
              // Suppress spurious warnings.
              // See https://github.com/llvm/llvm-project/issues/34798#issuecomment-980989495
              (void)reg_idx;
              (void)disp_idx;
              if constexpr (machine_insn_info::kIsImmediate<Operands>) {
                return std::tuple{
                    static_cast<typename Operands::Class::Type>(MachineInsnX86_64::imm())};
              } else if constexpr (machine_insn_info::kIsMemoryOperand<Operands> &&
                                   Operand::kUsage == machine_insn_info::kDefEarlyClobber) {
                disp_idx++;
                if (disp_idx == 1) {
                  return std::tuple{Assembler::Operand{.base = GetGReg(this->RegAt(reg_idx++)),
                                                       .disp = static_cast<int32_t>(disp())}};
                } else /* disp_idx == 2 */ {
                  return std::tuple{Assembler::Operand{.base = GetGReg(this->RegAt(reg_idx++)),
                                                       .disp = static_cast<int32_t>(disp2())}};
                }
              } else if constexpr (Operand::Class::kAsRegister == 'x') {
                return std::tuple{GetXReg(this->RegAt(reg_idx++))};
              } else if constexpr (Operand::Class::kAsRegister == 'r' ||
                                   Operand::Class::kAsRegister == 'q') {
                return std::tuple{GetGReg(this->RegAt(reg_idx++))};
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
            typename Operand,
            typename... OperandsRest,
            typename Arg,
            typename... Args>
  auto ProcessArgs(Arg arg,
                   Args... args) -> std::enable_if_t<machine_insn_info::kIsImmediate<Operand>> {
    this->set_imm(arg);
    ProcessArgs<reg_idx, disp_idx, OperandsRest...>(args...);
  }

  template <size_t reg_idx,
            size_t disp_idx,
            typename Operand,
            typename... OperandsRest,
            typename Arg,
            typename... Args>
  auto ProcessArgs(Arg arg,
                   Args... args) -> std::enable_if_t<machine_insn_info::kIsRegister<Operand>> {
    static_assert(std::is_same_v<MachineReg, Arg>);
    this->SetRegAt(reg_idx, arg);
    ProcessArgs<reg_idx + 1, disp_idx, OperandsRest...>(args...);
  }

  template <size_t reg_idx,
            size_t disp_idx,
            typename Operand,
            typename... OperandsRest,
            typename Arg1,
            typename Arg2,
            typename... Args>
  auto ProcessArgs(Arg1 base,
                   Arg2 disp,
                   Args... args) -> std::enable_if_t<machine_insn_info::kIsMemoryOperand<Operand>> {
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

  template <typename Operand, typename = void>
  struct RegInfo;
  template <typename Operand>
  struct RegInfo<Operand, std::enable_if_t<machine_insn_info::kIsRegister<Operand>>> {
    static constexpr auto kRegClass = &Operand::Class::template kRegClass<MachineInsnX86_64>;
    static constexpr auto kRegKind = static_cast<MachineRegKind::StandardAccess>(Operand::kUsage);
    static_assert(MachineRegKind::kDef ==
                  static_cast<MachineRegKind::StandardAccess>(machine_insn_info::kDef));
    static_assert(MachineRegKind::kDefEarlyClobber ==
                  static_cast<MachineRegKind::StandardAccess>(machine_insn_info::kDefEarlyClobber));
    static_assert(MachineRegKind::kUse ==
                  static_cast<MachineRegKind::StandardAccess>(machine_insn_info::kUse));
    static_assert(MachineRegKind::kUseDef ==
                  static_cast<MachineRegKind::StandardAccess>(machine_insn_info::kUseDef));
  };
  template <typename Operand>
  struct RegInfo<Operand, std::enable_if_t<machine_insn_info::kIsMemoryOperand<Operand>>> {
    static_assert(Operand::kUsage == machine_insn_info::kDefEarlyClobber);
    static constexpr auto kRegClass = &kGeneralReg32;
    static constexpr auto kRegKind = MachineRegKind::kUse;
  };

  template <typename... Operand>
  struct GenMachineInsnInfoT<std::tuple<Operand...>> {
    static constexpr MachineInsnInfo value =
        MachineInsnInfo({GetOpcode.template operator()<MachineOpcode>(),
                         sizeof...(Operand),
                         {{RegInfo<Operand>::kRegClass, RegInfo<Operand>::kRegKind}...},
                         GetInsnKind()});
  };
};

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_MACHINE_INSN_INTRINSICS_H_
