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
#include "berberis/backend/x86_64/code_debug.h"
#include "berberis/backend/x86_64/code_emit.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/dependent_false.h"
#include "berberis/base/stringprintf.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/intrinsics_bindings.h"

namespace berberis::x86_64 {

struct MemoryOperand {
  MachineReg base = kInvalidMachineReg;
  MachineReg index = kInvalidMachineReg;
  Assembler::ScaleFactor scale = Assembler::kTimesOne;
  // Note: x86-64 only supports 64bit offset in one instruction: movabs – and that one may only be
  // be used to move a value to or from RAX. We don't use it in our code anywhere and it would be
  // better to treat it as a special case, rather than pretend that other instruction may support
  // 64bit offset.
  int32_t disp = 0;
};

template <typename IntrinsicBindingInfo>
class MachineInsnOperandsHelper;

template <auto kEmitInsnFunc,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestriction,
          typename... Operands,
          bool kSideEffects>
class MachineInsnOperandsHelper<device_arch_info::DeviceInsnInfo<kEmitInsnFunc,
                                                                 kMnemo,
                                                                 kSideEffects,
                                                                 GetOpcode,
                                                                 CPUIDRestriction,
                                                                 std::tuple<Operands...>>>
    final {
 public:
  // We want to filter out any operands that are not used for Register args.
  // Note: memory operands accept register and offset, thus they are included.
  using RegOperandsTuple = decltype(std::tuple_cat(
      std::declval<std::conditional_t<device_arch_info::kIsRegister<Operands> ||
                                          device_arch_info::kIsMemoryOperand<Operands>,
                                      std::tuple<Operands>,
                                      std::tuple<>>>()...));
  // Note: immediates accept appropriate type, register operands includes only register while memory
  // operand needs both base register and offset.
  using ConstructorArgsTuple = decltype(std::tuple_cat(
      std::declval<std::conditional_t<device_arch_info::kIsImmediate<Operands>,
                                      std::tuple<typename Operands::Class::Type>,
                                      std::conditional_t<device_arch_info::kIsRegister<Operands>,
                                                         std::tuple<MachineReg>,
                                                         std::tuple<const MemoryOperand&>>>>()...));
};

template <typename IntrinsicBindingInfo,
          typename = typename MachineInsnOperandsHelper<IntrinsicBindingInfo>::RegOperandsTuple,
          typename = typename MachineInsnOperandsHelper<IntrinsicBindingInfo>::ConstructorArgsTuple>
class MachineInsn;

template <auto kEmitInsnFunc,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestriction,
          typename... Operands,
          typename... RegOperands,
          typename... ConstructorArgs,
          bool kSideEffects>
class MachineInsn<device_arch_info::DeviceInsnInfo<kEmitInsnFunc,
                                                   kMnemo,
                                                   kSideEffects,
                                                   GetOpcode,
                                                   CPUIDRestriction,
                                                   std::tuple<Operands...>>,
                  std::tuple<RegOperands...>,
                  std::tuple<ConstructorArgs...>>
    final : public MachineInsnX86_64 {
 private:
  template <auto>
  static constexpr MachineInsnInfo GenMachineInsnInfo();

 public:
  // This static simplifies constructing this MachineInsn in intrinsic implementations.
  template <typename MachineIRBuilder>
  static constexpr MachineInsn* (MachineIRBuilder::*kGenFunc)(ConstructorArgs...) =
      &MachineIRBuilder::template Gen<MachineInsn>;

  using DeviceInsnInfo = device_arch_info::DeviceInsnInfo<kEmitInsnFunc,
                                                          kMnemo,
                                                          kSideEffects,
                                                          GetOpcode,
                                                          CPUIDRestriction,
                                                          std::tuple<Operands...>>;

  explicit MachineInsn(ConstructorArgs... args) : MachineInsnX86_64(&GenMachineInsnInfo(args...)) {
    constexpr int kImmediateOperandsCount = (device_arch_info::kIsImmediate<Operands> + ... + 0);
    static_assert(kImmediateOperandsCount <= 1);
    constexpr int kMemoryOperandsCount = (device_arch_info::kIsMemoryOperand<Operands> + ... + 0);
    static_assert(kMemoryOperandsCount <= 2);
    size_t reg_idx{}, mem_idx{};
    (
        [&reg_idx, &mem_idx, this]<typename Operand, typename ConstructorArg>(ConstructorArg arg) {
          if constexpr (device_arch_info::kIsImmediate<Operand>) {
            MachineInsnX86_64::set_imm(arg);
          } else if constexpr (device_arch_info::kIsRegister<Operand>) {
            static_assert(std::is_same_v<MachineReg, ConstructorArg>);
            MachineInsnX86_64::SetRegAt(reg_idx++, arg);
          } else if constexpr (device_arch_info::kIsMemoryOperand<Operand>) {
            static_assert(std::is_same_v<const MemoryOperand&, ConstructorArg>);
            if (arg.base != kInvalidMachineReg) {
              MachineInsnX86_64::SetRegAt(reg_idx++, arg.base);
            }
            if (arg.index != kInvalidMachineReg) {
              MachineInsnX86_64::SetRegAt(reg_idx++, arg.index);
            }
            if (++mem_idx == 1) {
              MachineInsnX86_64::set_disp(arg.disp);
              MachineInsnX86_64::set_scale(arg.scale);
            } else if (mem_idx == 2) {
              MachineInsnX86_64::set_disp2(arg.disp);
              MachineInsnX86_64::set_scale2(arg.scale);
            }
          }
        }.template operator()<Operands, ConstructorArgs>(args),
        ...);
  }

  template <auto BaseIndexRegistersUsed = std::array<bool, 0>{}>
  static constexpr MachineInsnInfo kInfos = GenMachineInsnInfo<BaseIndexRegistersUsed>();
  // TODO(399130034): This field should be removed when all clients would switch to kInfos and/or
  // DeviceInsnInfo.
  static constexpr const MachineInsnInfo& kInfo = kInfos<>;

  static constexpr int NumRegOperands() { return kInfo.num_reg_operands; }
  static constexpr const MachineRegKind& RegKindAt(int i) { return kInfo.reg_kinds[i]; }

  std::string GetDebugString() const override {
    std::string s(kMnemo);
    // Code below assumes that we have at most two memory operands.
    static_assert((device_arch_info::kIsMemoryOperand<Operands> + ... + 0) <= 2);
    size_t arg_idx{}, reg_idx{}, mem_idx{};
    (
        [&s, &arg_idx, &reg_idx, &mem_idx, this]<typename Operand> {
          s += " ";
          if (arg_idx > 0) {
            s += ", ";
          }
          if constexpr (device_arch_info::kIsImmediate<Operand>) {
            s += GetImmOperandDebugString(this);
          } else if constexpr (device_arch_info::kIsMemoryOperand<Operand>) {
            auto [has_base, has_index] = OpcodeHasMemoryBaseIndex(mem_idx++);
            if (mem_idx == 1) {
              if (has_base) {
                if (has_index) {
                  s += GetBaseIndexDispMemOperandDebugString(this, reg_idx);
                  reg_idx += 2;
                } else {
                  s += GetBaseDispMemOperandDebugString(this, reg_idx++);
                }
              } else if (has_index) {
                s += GetIndexDispMemOperandDebugString(this, reg_idx++);
              } else {
                s += GetAbsoluteMemOperandDebugString(this);
              }
            } else /* mem_idx == 2 */ {
              if (has_base) {
                if (has_index) {
                  s += StringPrintf("[%s + %s * %d + 0x%x]",
                                    GetRegOperandDebugString(this, reg_idx).c_str(),
                                    GetRegOperandDebugString(this, reg_idx + 1).c_str(),
                                    1 << MachineInsnX86_64::scale2(),
                                    MachineInsnX86_64::disp2());
                  reg_idx += 2;
                } else {
                  s += StringPrintf(
                      "[%s + 0x%x]", GetRegOperandDebugString(this, reg_idx++).c_str(), disp2());
                }
              } else if (has_index) {
                s += StringPrintf("[%s * %d + 0x%x]",
                                  GetRegOperandDebugString(this, reg_idx++).c_str(),
                                  1 << MachineInsnX86_64::scale2(),
                                  MachineInsnX86_64::disp2());
              } else {
                s += StringPrintf("[0x%x]", MachineInsnX86_64::disp2());
              }
            }
          } else if constexpr (device_arch_info::kIsImplicitReg<Operand>) {
            s += GetImplicitRegOperandDebugString(this, reg_idx++);
          } else {
            s += GetRegOperandDebugString(this, reg_idx++);
          }
          arg_idx++;
        }.template operator()<Operands>(),
        ...);

    if (MachineInsnX86_64::recovery_pc()) {
      s += StringPrintf(" <0x%" PRIxPTR ">", MachineInsnX86_64::recovery_pc());
    }
    return s;
  }

  void Emit(CodeEmitter* as) const override {
    // Code below assumes that we have at most two memory operands.
    static_assert((device_arch_info::kIsMemoryOperand<Operands> + ... + 0) <= 2);
    size_t reg_idx{}, mem_idx{};
    std::apply(
        kEmitInsnFunc,
        std::tuple_cat(std::tuple<CodeEmitter&>{*as}, [&reg_idx, &mem_idx, this]<typename Operand> {
          // Suppress spurious warnings.
          // See https://github.com/llvm/llvm-project/issues/34798#issuecomment-980989495
          (void)reg_idx;
          (void)mem_idx;
          if constexpr (device_arch_info::kIsImmediate<Operands>) {
            return std::tuple{
                static_cast<typename Operands::Class::Type>(MachineInsnX86_64::imm())};
          } else if constexpr (device_arch_info::kIsMemoryOperand<Operands>) {
            auto [has_base, has_index] = OpcodeHasMemoryBaseIndex(mem_idx++);
            Assembler::Operand operand;
            if (has_base) {
              operand.base = GetGReg(MachineInsnX86_64::RegAt(reg_idx++));
            }
            if (has_index) {
              operand.index = GetGReg(MachineInsnX86_64::RegAt(reg_idx++));
            }
            if (mem_idx == 1) {
              if (has_index) {
                operand.scale = scale();
              }
              operand.disp = static_cast<int32_t>(disp());
            } else /* mem_idx == 2 */ {
              if (has_index) {
                operand.scale = scale2();
              }
              operand.disp = static_cast<int32_t>(disp2());
            }
            return std::tuple{operand};
          } else if constexpr (device_arch_info::kIsImplicitReg<Operand>) {
            return std::tuple{};
          } else if constexpr (Operand::Class::kAsRegister == 'x') {
            return std::tuple{GetXReg(MachineInsnX86_64::RegAt(reg_idx++))};
          } else if constexpr (Operand::Class::kAsRegister == 'r' ||
                               Operand::Class::kAsRegister == 'q') {
            return std::tuple{GetGReg(MachineInsnX86_64::RegAt(reg_idx++))};
          } else {
            static_assert(kDependentTypeFalse<Operand>);
          }
        }.template operator()<Operands>()...));
  }

 private:
  // Ensure that bits that we are using to split opcodes are not used by opcode already.
  // Note: we need to do that with all opcodes, including opcodes without memory operands,
  // to guarantee that memory-using opcodes don't clash with memory non-using opcodes.
  static_assert(!(static_cast<int>(GetOpcode.template operator()<MachineOpcode>()) &
                  ((~0) << kLowMachineOpcodeBits)));

  static constexpr auto GetInsnKind() {
    if constexpr (kSideEffects) {
      return kMachineInsnSideEffects;
    } else {
      return kMachineInsnDefault;
    }
  }

  constexpr std::pair<bool, bool> OpcodeHasMemoryBaseIndex(size_t mem_operand_idx) const {
    int base_index_info = opcode() >> (kLowMachineOpcodeBits + mem_operand_idx * 2);
    return {base_index_info & 1, base_index_info & 2};
  }

  static const MachineInsnInfo& GenMachineInsnInfo(ConstructorArgs... args) {
    constexpr int kMemoryOperandsCount =
        (std::is_same_v<ConstructorArgs, const MemoryOperand&> + ... + 0);
    static_assert(kMemoryOperandsCount <= 2);
    if constexpr (kMemoryOperandsCount == 0) {
      return kInfos<>;
    } else {
      const MachineInsnInfo* const* ptr;
      if constexpr (kMemoryOperandsCount == 1) {
        static constexpr const MachineInsnInfo* array[4] = {&kInfos<std::array{false, false}>,
                                                            &kInfos<std::array{true, false}>,
                                                            &kInfos<std::array{false, true}>,
                                                            &kInfos<std::array{true, true}>};
        ptr = array;
      } else {
        static constexpr const MachineInsnInfo* array[16] = {
            &kInfos<std::array{false, false, false, false}>,
            &kInfos<std::array{true, false, false, false}>,
            &kInfos<std::array{false, true, false, false}>,
            &kInfos<std::array{true, true, false, false}>,
            &kInfos<std::array{false, false, true, false}>,
            &kInfos<std::array{true, false, true, false}>,
            &kInfos<std::array{false, true, true, false}>,
            &kInfos<std::array{true, true, true, false}>,
            &kInfos<std::array{false, false, false, true}>,
            &kInfos<std::array{true, false, false, true}>,
            &kInfos<std::array{false, true, false, true}>,
            &kInfos<std::array{true, true, false, true}>,
            &kInfos<std::array{false, false, true, true}>,
            &kInfos<std::array{true, false, true, true}>,
            &kInfos<std::array{false, true, true, true}>,
            &kInfos<std::array{true, true, true, true}>};
        ptr = array;
      }
      size_t index = 0;
      size_t current_bit = 1;
      (
          [&index, &current_bit]<typename ConstructorArg>(ConstructorArg arg) {
            if constexpr (std::is_same_v<ConstructorArg, const MemoryOperand&>) {
              if (arg.base != kInvalidMachineReg) {
                index |= current_bit;
              }
              current_bit <<= 1;
              if (arg.index != kInvalidMachineReg) {
                index |= current_bit;
              }
              current_bit <<= 1;
            }
          }.template operator()<ConstructorArgs>(args),
          ...);
      return *ptr[index];
    }
  }
};

template <auto kEmitInsnFunc,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestriction,
          typename... Operands,
          typename... RegOperands,
          typename... ConstructorArgs,
          bool kSideEffects>
template <auto BaseIndexRegistersUsed>
constexpr MachineInsnInfo MachineInsn<device_arch_info::DeviceInsnInfo<kEmitInsnFunc,
                                                                       kMnemo,
                                                                       kSideEffects,
                                                                       GetOpcode,
                                                                       CPUIDRestriction,
                                                                       std::tuple<Operands...>>,
                                      std::tuple<RegOperands...>,
                                      std::tuple<ConstructorArgs...>>::GenMachineInsnInfo() {
  MachineInsnInfo result = {
    .opcode = GetOpcode.template operator()<MachineOpcode>(),
    .kind = GetInsnKind()
  };
  size_t mem_operand_bit_pos = 0;
  (
      [&opcode = result.opcode,
       &mem_operand_bit_pos,
       &num_reg_operands = result.num_reg_operands,
       &reg_kinds = result.reg_kinds]<typename Operand> {
        if constexpr (device_arch_info::kIsRegister<Operand>) {
          static_assert(MachineRegKind::kDef ==
                        static_cast<MachineRegKind::StandardAccess>(device_arch_info::kDef));
          static_assert(
              MachineRegKind::kDefEarlyClobber ==
              static_cast<MachineRegKind::StandardAccess>(device_arch_info::kDefEarlyClobber));
          static_assert(MachineRegKind::kUse ==
                        static_cast<MachineRegKind::StandardAccess>(device_arch_info::kUse));
          static_assert(MachineRegKind::kUseDef ==
                        static_cast<MachineRegKind::StandardAccess>(device_arch_info::kUseDef));
          reg_kinds[num_reg_operands++] = {
              &kRegisterClass<typename Operand::Class>,
              static_cast<MachineRegKind::StandardAccess>(Operand::kUsage)};
        } else {
          static_assert(device_arch_info::kIsMemoryOperand<Operand>);
          // Note: normally size of array should match number of memory operands, but that's not
          // true for kInfo where it's zero.
          // TODO(399130034): remove std::size when kInfo is removed.
          if (std::size(BaseIndexRegistersUsed) > mem_operand_bit_pos &&
              BaseIndexRegistersUsed[mem_operand_bit_pos]) {
            reg_kinds[num_reg_operands++] = {&kGeneralReg64, MachineRegKind::kUse};
            opcode = static_cast<MachineOpcode>(
                opcode | (1 << (kLowMachineOpcodeBits + mem_operand_bit_pos)));
          }
          mem_operand_bit_pos++;
          if (std::size(BaseIndexRegistersUsed) > mem_operand_bit_pos &&
              BaseIndexRegistersUsed[mem_operand_bit_pos]) {
            reg_kinds[num_reg_operands++] = {&kGeneralReg64, MachineRegKind::kUse};
            opcode = static_cast<MachineOpcode>(
                opcode | (1 << (kLowMachineOpcodeBits + mem_operand_bit_pos)));
          }
          mem_operand_bit_pos++;
        }
      }.template operator()<RegOperands>(),
      ...);
  return result;
}

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_MACHINE_INSN_INTRINSICS_H_
