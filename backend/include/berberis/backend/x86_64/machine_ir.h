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

// x86_64 machine IR interface.

#ifndef BERBERIS_BACKEND_X86_64_MACHINE_IR_H_
#define BERBERIS_BACKEND_X86_64_MACHINE_IR_H_

#include <array>
#include <bit>
#include <cstdint>
#include <string>

#include "berberis/assembler/x86_64.h"
#include "berberis/backend/code_emitter.h"
#include "berberis/backend/common/machine_ir.h"  // IWYU pragma: export.
#include "berberis/backend/x86_64/code_emit.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/base/stringprintf.h"
#include "berberis/device_arch_info/x86_64/device_arch_info.h"
#include "berberis/guest_state/guest_state_arch.h"

namespace berberis {

// Some instructions form groups. E.g. memory-accesses typically have 4 versions: Absolute, Base,
// Index Base+Index.
//
// To ensure that there enough bits to separate these versions we reserve top 8 bits.
inline constexpr int kLowMachineOpcodeBits = 24;

enum MachineOpcode : int {
  kMachineOpUndefined = 0,
  kMachineOpCallImm,
  kMachineOpCallImmArg,
  kMachineOpPseudoBranch,
  kMachineOpPseudoCondBranch,
  kMachineOpPseudoCopy,
  kMachineOpPseudoDefReg,
  kMachineOpPseudoDefXReg,
  kMachineOpPseudoIndirectJump,
  kMachineOpPseudoJump,
  kMachineOpPseudoReadFlags,
  kMachineOpPseudoWriteFlags,
#include "machine_opcode_x86_64-inl.h"  // NOLINT generated file!
};

namespace x86_64 {

class MachineRegs {
 public:
  static constexpr MachineReg kR8{1};
  static constexpr MachineReg kR9{2};
  static constexpr MachineReg kR10{3};
  static constexpr MachineReg kR11{4};
  static constexpr MachineReg kRSI{5};
  static constexpr MachineReg kRDI{6};
  static constexpr MachineReg kRAX{7};
  static constexpr MachineReg kRBX{8};
  static constexpr MachineReg kRCX{9};
  static constexpr MachineReg kRDX{10};
  static constexpr MachineReg kRBP{11};
  static constexpr MachineReg kRSP{12};
  static constexpr MachineReg kR12{13};
  static constexpr MachineReg kR13{14};
  static constexpr MachineReg kR14{15};
  static constexpr MachineReg kR15{16};
  static constexpr MachineReg kFLAGS{19};
  static constexpr MachineReg kXMM0{20};
  static constexpr MachineReg kXMM1{21};
  static constexpr MachineReg kXMM2{22};
  static constexpr MachineReg kXMM3{23};
  static constexpr MachineReg kXMM4{24};
  static constexpr MachineReg kXMM5{25};
  static constexpr MachineReg kXMM6{26};
  static constexpr MachineReg kXMM7{27};
  static constexpr MachineReg kXMM8{28};
  static constexpr MachineReg kXMM9{29};
  static constexpr MachineReg kXMM10{30};
  static constexpr MachineReg kXMM11{31};
  static constexpr MachineReg kXMM12{32};
  static constexpr MachineReg kXMM13{33};
  static constexpr MachineReg kXMM14{34};
  static constexpr MachineReg kXMM15{35};
};

inline constexpr auto kMachineRegFLAGS = MachineRegs::kFLAGS;
inline constexpr auto kMachineRegRBP = MachineRegs::kRBP;
inline constexpr auto kMachineRegRSP = MachineRegs::kRSP;

inline bool IsGReg(MachineReg r) {
  return r.reg() >= MachineRegs::kR8.reg() && r.reg() <= MachineRegs::kR15.reg();
}

inline bool IsXReg(MachineReg r) {
  return r.reg() >= MachineRegs::kXMM0.reg() && r.reg() <= MachineRegs::kXMM15.reg();
}

// rax, rdi, rsi, rdx, rcx, r8-r11, xmm0-xmm15, flags
const int kMaxMachineRegOperands = 26;

// Context loads and stores use rbp as base.
inline constexpr auto kCPUStatePointer = MachineRegs::kRBP;

struct MachineInsnInfo {
  MachineOpcode opcode;
  int num_reg_operands;
  MachineRegKind reg_kinds[kMaxMachineRegOperands];
  MachineInsnKind kind;
  constexpr int InputRegistersCount() const {
    int result = 0;
    for (int index = 0; index < num_reg_operands; ++index) {
      if (reg_kinds[index].IsInput()) {
        result++;
      }
    }
    return result;
  }
  constexpr int OutputRegistersCount() const {
    int result = 0;
    for (int index = 0; index < num_reg_operands; ++index) {
      if (reg_kinds[index].IsDef()) {
        result++;
      }
    }
    return result;
  }
};

template <typename MachineInsnInfoClass>
constexpr MachineRegClass MachineRegClassFromMachineInsnInfoClass() {
  return []<typename... RegisterClass>(const std::tuple<RegisterClass...>&) -> MachineRegClass {
    return {
        .debug_name = MachineInsnInfoClass::kName,
        .reg_size = MachineInsnInfoClass::kSizeInBits / 8,
        .reg_mask = ((1ULL << RegisterClass::template kMachineRegId<MachineRegs>.reg()) | ...),
        .num_regs = sizeof...(RegisterClass),
        .regs = {RegisterClass::template kMachineRegId<MachineRegs>...},
    };
  }(typename MachineInsnInfoClass::RegistersList());
}

template <typename MachineInsnInfoClass>
inline constexpr MachineRegClass kRegisterClass =
    MachineRegClassFromMachineInsnInfoClass<MachineInsnInfoClass>();

inline constexpr auto& kRAX = kRegisterClass<device_arch_info::RAX>;
inline constexpr auto& kGeneralReg32 = kRegisterClass<device_arch_info::GeneralReg32>;
inline constexpr auto& kGeneralReg64 = kRegisterClass<device_arch_info::GeneralReg64>;
inline constexpr auto& kReg32 = kRegisterClass<device_arch_info::Reg32>;
inline constexpr auto& kReg64 = kRegisterClass<device_arch_info::Reg64>;
inline constexpr auto& kXmmReg = kRegisterClass<device_arch_info::XmmReg>;
inline constexpr auto& kFLAGS = kRegisterClass<device_arch_info::FLAGS>;

class MachineInsnX86_64 : public MachineInsn {
 public:
  MachineInsnX86_64(const MachineInsnX86_64& other) : MachineInsn(other) {
    for (int i = 0; i < kMaxMachineRegOperands; i++) {
      regs_[i] = other.regs_[i];
    }
    scale_ = other.scale_;
    scale2_ = other.scale2_;
    disp_ = other.disp_;
    disp2_ = other.disp2_;
    imm_ = other.imm_;
    cond_ = other.cond_;

    SetRegs(regs_);
  }

  ~MachineInsnX86_64() override {
    // No code here - will never be called!
  }

  Assembler::ScaleFactor scale() const { return scale_; }

  Assembler::ScaleFactor scale2() const { return scale2_; }

  uint32_t disp() const { return disp_; }

  uint32_t disp2() const { return disp2_; }

  Assembler::Condition cond() const { return cond_; }

  uint64_t imm() const { return imm_; }

 protected:
  explicit MachineInsnX86_64(const MachineInsnInfo* info)
      : MachineInsn(info->opcode, info->num_reg_operands, info->reg_kinds, regs_, info->kind),
        scale_(Assembler::kTimesOne) {}

  void set_scale(Assembler::ScaleFactor scale) { scale_ = scale; }

  void set_scale2(Assembler::ScaleFactor scale2) { scale2_ = scale2; }

  void set_disp(uint32_t disp) { disp_ = disp; }

  void set_disp2(uint32_t disp2) { disp2_ = disp2; }

  void set_cond(Assembler::Condition cond) { cond_ = cond; }

  void set_imm(uint64_t imm) { imm_ = imm; }

 private:
  MachineReg regs_[kMaxMachineRegOperands];
  uint32_t disp_;
  Assembler::ScaleFactor scale_;
  Assembler::ScaleFactor scale2_;
  Assembler::Condition cond_;
  uint32_t disp2_;
  uint64_t imm_;
};

// Syntax sugar.
inline const MachineInsnX86_64* AsMachineInsnX86_64(const MachineInsn* insn) {
  return static_cast<const MachineInsnX86_64*>(insn);
}

inline MachineInsnX86_64* AsMachineInsnX86_64(MachineInsn* insn) {
  return static_cast<MachineInsnX86_64*>(insn);
}

// Clobbered registers are described as DEF'ed.
// TODO(b/232598137): implement simpler support for clobbered registers?
class CallImm : public MachineInsnX86_64 {
 public:
  enum class RegType {
    kIntType,
    kXmmType,
  };

  static constexpr RegType kIntRegType = RegType::kIntType;
  static constexpr RegType kXmmRegType = RegType::kXmmType;

  struct Arg {
    MachineReg reg;
    RegType reg_type;
  };

  explicit CallImm(uint64_t imm);

  [[nodiscard]] static int GetIntArgIndex(int i);
  [[nodiscard]] static int GetXmmArgIndex(int i);
  [[nodiscard]] static int GetFlagsArgIndex();

  [[nodiscard]] MachineReg IntResultAt(int i) const;
  [[nodiscard]] MachineReg XmmResultAt(int i) const;

  [[nodiscard]] std::string GetDebugString() const override;
  void Emit(CodeEmitter* as) const override;
  void EnableCustomAVX256ABI() { custom_avx256_abi_ = true; };

 private:
  bool custom_avx256_abi_;
};

// An auxiliary instruction to express data-flow for CallImm arguments.  It uses the same vreg as
// the corresponding operand in CallImm. The specific hard register assigned is defined by the
// register class of CallImm operand. MachineIRBuilder adds an extra PseudoCopy before this insn in
// case the same vreg holds values for several arguments (with non-intersecting register classes).
class CallImmArg : public MachineInsnX86_64 {
 public:
  explicit CallImmArg(MachineReg arg, CallImm::RegType reg_type);

  std::string GetDebugString() const override;
  void Emit(CodeEmitter*) const override{
      // It's an auxiliary instruction. Do not emit.
  };
};

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
      std::declval<std::conditional_t<device_arch_info::kIsCondition<Operands> ||
                                          device_arch_info::kIsImmediate<Operands>,
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
  static constexpr std::array<MachineInsnInfo,
                              1 << (2 * (device_arch_info::kIsMemoryOperand<Operands> + ... + 0))>
  GenMachineInsnInfos();

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
    constexpr int kConditionalsOperandsCount = (device_arch_info::kIsCondition<Operands> + ... + 0);
    static_assert(kConditionalsOperandsCount <= 1);
    constexpr int kImmediateOperandsCount = (device_arch_info::kIsImmediate<Operands> + ... + 0);
    static_assert(kImmediateOperandsCount <= 1);
    constexpr int kMemoryOperandsCount = (device_arch_info::kIsMemoryOperand<Operands> + ... + 0);
    static_assert(kMemoryOperandsCount <= 2);
    size_t reg_idx{}, mem_idx{};
    (
        [&reg_idx, &mem_idx, this]<typename Operand, typename ConstructorArg>(ConstructorArg arg) {
          if constexpr (device_arch_info::kIsCondition<Operand>) {
            MachineInsnX86_64::set_cond(arg);
          } else if constexpr (device_arch_info::kIsImmediate<Operand>) {
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

  static constexpr std::array<MachineInsnInfo,
                              1 << (2 * (device_arch_info::kIsMemoryOperand<Operands> + ... + 0))>
      kInfos = GenMachineInsnInfos();
  // Note: kInfo has well-defined meaning – it's information about intrinsic with all MemoryOperand
  // types ignored.
  // This is useful not only for instructions without operands, but also for SSA form: since these
  // registers that are passed into MemoryOperand are always kUse and never kDef or kUseDef we may
  // ignore them in our analysis.
  static constexpr const MachineInsnInfo& kInfo = kInfos[0];

  int NumRegOperands() {
    constexpr int kMemoryOperandsCount = (device_arch_info::kIsMemoryOperand<Operands> + ... + 0);
    if constexpr (kMemoryOperandsCount == 0) {
      return kInfo.num_reg_operands;
    } else {
      return kInfo.num_reg_operands + std::popcount(opcode() >> kLowMachineOpcodeBits);
    }
  }

  const MachineRegKind& RegKindAt(int i) {
    constexpr int kMemoryOperandsCount = (device_arch_info::kIsMemoryOperand<Operands> + ... + 0);
    constexpr int kMemoryOperandsCountMask = (1 << (2 * kMemoryOperandsCount)) - 1;
    return kInfos[(opcode() >> kLowMachineOpcodeBits) & kMemoryOperandsCountMask].reg_kinds[i];
  }

  std::string GetDebugString() const override {
    std::string s(kMnemo);
    // Code below assumes that we have at most two memory operands.
    static_assert((device_arch_info::kIsMemoryOperand<Operands> + ... + 0) <= 2);
    size_t arg_idx{}, reg_idx{}, mem_idx{};
    (
        [&s, &arg_idx, &reg_idx, &mem_idx, this]<typename Operand> {
          if (arg_idx == 0) {
            s += " ";
          } else {
            s += ", ";
          }
          if constexpr (device_arch_info::kIsCondition<Operand>) {
            s += GetCondName(cond());
          } else if constexpr (device_arch_info::kIsImmediate<Operand>) {
            s += StringPrintf("0x%" PRIx64, imm());
          } else if constexpr (device_arch_info::kIsMemoryOperand<Operand>) {
            auto [has_base, has_index] = OpcodeHasMemoryBaseIndex(mem_idx++);
            int32_t scale;
            if (has_index) {
              scale =
                  1 << (mem_idx == 1 ? MachineInsnX86_64::scale() : MachineInsnX86_64::scale2());
            }
            int32_t disp = mem_idx == 1 ? MachineInsnX86_64::disp() : MachineInsnX86_64::disp2();
            if (has_base) {
              if (has_index) {
                s += StringPrintf("[%s + %s * %d + 0x%x]",
                                  GetRegOperandDebugString(this, reg_idx).c_str(),
                                  GetRegOperandDebugString(this, reg_idx + 1).c_str(),
                                  scale,
                                  disp);
                reg_idx += 2;
              } else {
                s += StringPrintf(
                    "[%s + 0x%x]", GetRegOperandDebugString(this, reg_idx++).c_str(), disp);
              }
            } else if (has_index) {
              s += StringPrintf("[%s * %d + 0x%x]",
                                GetRegOperandDebugString(this, reg_idx++).c_str(),
                                scale,
                                disp);
            } else {
              s += StringPrintf("[0x%x]", disp);
            }
          } else if constexpr (device_arch_info::kIsImplicitReg<Operand>) {
            s += StringPrintf("(%s)", GetRegOperandDebugString(this, reg_idx++).c_str());
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
          if constexpr (device_arch_info::kIsCondition<Operands>) {
            return std::tuple{MachineInsnX86_64::cond()};
          } else if constexpr (device_arch_info::kIsImmediate<Operands>) {
            return std::tuple{MachineInsnX86_64::imm()};
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
            return reg_idx++, std::tuple{};
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
      return kInfos[0];
    } else {
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
      return kInfos[index];
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

template <auto kEmitInsnFunc,
          auto kMnemo,
          auto GetOpcode,
          typename CPUIDRestriction,
          typename... Operands,
          typename... RegOperands,
          typename... ConstructorArgs,
          bool kSideEffects>
constexpr std::array<MachineInsnInfo,
                     1 << (2 * (device_arch_info::kIsMemoryOperand<Operands> + ... + 0))>
MachineInsn<device_arch_info::DeviceInsnInfo<kEmitInsnFunc,
                                             kMnemo,
                                             kSideEffects,
                                             GetOpcode,
                                             CPUIDRestriction,
                                             std::tuple<Operands...>>,
            std::tuple<RegOperands...>,
            std::tuple<ConstructorArgs...>>::GenMachineInsnInfos() {
  constexpr int kMemoryOperandsCount = (device_arch_info::kIsMemoryOperand<Operands> + ... + 0);
  if constexpr (kMemoryOperandsCount == 0) {
    return {GenMachineInsnInfo<std::array<bool, 0>{}>()};
  } else if constexpr (kMemoryOperandsCount == 1) {
    return {GenMachineInsnInfo<std::array{false, false}>(),
            GenMachineInsnInfo<std::array{true, false}>(),
            GenMachineInsnInfo<std::array{false, true}>(),
            GenMachineInsnInfo<std::array{true, true}>()};
  } else if constexpr (kMemoryOperandsCount == 2) {
    return {GenMachineInsnInfo<std::array{false, false, false, false}>(),
            GenMachineInsnInfo<std::array{true, false, false, false}>(),
            GenMachineInsnInfo<std::array{false, true, false, false}>(),
            GenMachineInsnInfo<std::array{true, true, false, false}>(),
            GenMachineInsnInfo<std::array{false, false, true, false}>(),
            GenMachineInsnInfo<std::array{true, false, true, false}>(),
            GenMachineInsnInfo<std::array{false, true, true, false}>(),
            GenMachineInsnInfo<std::array{true, true, true, false}>(),
            GenMachineInsnInfo<std::array{false, false, false, true}>(),
            GenMachineInsnInfo<std::array{true, false, false, true}>(),
            GenMachineInsnInfo<std::array{false, true, false, true}>(),
            GenMachineInsnInfo<std::array{true, true, false, true}>(),
            GenMachineInsnInfo<std::array{false, false, true, true}>(),
            GenMachineInsnInfo<std::array{true, false, true, true}>(),
            GenMachineInsnInfo<std::array{false, true, true, true}>(),
            GenMachineInsnInfo<std::array{true, true, true, true}>()};
  } else {
    static_assert(kDependentTypeFalse<std::tuple<Operands...>>);
  }
}

class MachineIR : public berberis::MachineIR {
 public:
  enum class BasicBlockOrder {
    kUnordered,
    kReversePostOrder,
  };

  explicit MachineIR(Arena* arena, int num_vreg = 0)
      : berberis::MachineIR(arena, num_vreg, 0), bb_order_(BasicBlockOrder::kUnordered) {}

  void AddEdge(MachineBasicBlock* src, MachineBasicBlock* dst) {
    MachineEdge* edge = NewInArena<MachineEdge>(arena(), arena(), src, dst);
    src->out_edges().push_back(edge);
    dst->in_edges().push_back(edge);
    bb_order_ = BasicBlockOrder::kUnordered;
  }

  [[nodiscard]] static bool IsCPUStateGet(berberis::MachineInsn* insn) {
    if (insn->opcode() != kMachineOpMovqRegMemBaseDisp &&
        insn->opcode() != kMachineOpMovdqaXRegMemBaseDisp &&
        insn->opcode() != kMachineOpMovwRegMemBaseDisp &&
        insn->opcode() != kMachineOpMovsdXRegMemBaseDisp) {
      return false;
    }

    auto x86_insn = AsMachineInsnX86_64(insn);

    // Check that it is not for ThreadState fields outside of CPUState.
    if (x86_insn->disp() >= sizeof(CPUState)) {
      return false;
    }

    // reservation_value is loaded in HeavyOptimizerFrontend::AtomicLoad and written
    // in HeavyOptimizerFrontend::AtomicStore partially (for performance
    // reasons), which is not supported by our context optimizer.
    auto reservation_value_offset = offsetof(ThreadState, cpu.reservation_value);
    if (x86_insn->disp() >= reservation_value_offset &&
        x86_insn->disp() < reservation_value_offset + sizeof(Reservation)) {
      return false;
    }

    return x86_insn->RegAt(1) == kCPUStatePointer;
  }

  [[nodiscard]] static bool IsCPUStatePut(berberis::MachineInsn* insn) {
    if (insn->opcode() != kMachineOpMovqMemBaseDispReg &&
        insn->opcode() != kMachineOpMovdqaMemBaseDispXReg &&
        insn->opcode() != kMachineOpMovwMemBaseDispReg &&
        insn->opcode() != kMachineOpMovsdMemBaseDispXReg) {
      return false;
    }

    auto x86_insn = AsMachineInsnX86_64(insn);

    // Check that it is not for ThreadState fields outside of CPUState.
    if (x86_insn->disp() >= sizeof(CPUState)) {
      return false;
    }

    // reservation_value is loaded in HeavyOptimizerFrontend::AtomicLoad and written
    // in HeavyOptimizerFrontend::AtomicStore partially (for performance
    // reasons), which is not supported by our context optimizer.
    auto reservation_value_offset = offsetof(ThreadState, cpu.reservation_value);
    if (x86_insn->disp() >= reservation_value_offset &&
        x86_insn->disp() < reservation_value_offset + sizeof(Reservation)) {
      return false;
    }

    return x86_insn->RegAt(0) == kCPUStatePointer;
  }

  [[nodiscard]] MachineBasicBlock* NewBasicBlock() {
    return NewInArena<MachineBasicBlock>(arena(), arena(), ReserveBasicBlockId());
  }

  // Instruction iterators are preserved after splitting basic block and moving
  // instructions to the new basic block.
  [[nodiscard]] MachineBasicBlock* SplitBasicBlock(MachineBasicBlock* bb,
                                                   MachineInsnList::iterator insn_it) {
    MachineBasicBlock* new_bb = NewBasicBlock();

    new_bb->insn_list().splice(
        new_bb->insn_list().begin(), bb->insn_list(), insn_it, bb->insn_list().end());
    bb->insn_list().push_back(NewInsn<PseudoBranch>(new_bb));

    // Relink out edges from bb.
    for (auto out_edge : bb->out_edges()) {
      out_edge->set_src(new_bb);
    }
    new_bb->out_edges().swap(bb->out_edges());

    AddEdge(bb, new_bb);
    bb_list().push_back(new_bb);
    return new_bb;
  }

  [[nodiscard]] static bool IsControlTransfer(berberis::MachineInsn* insn) {
    return insn->opcode() == kMachineOpPseudoBranch ||
           insn->opcode() == kMachineOpPseudoCondBranch ||
           insn->opcode() == kMachineOpPseudoIndirectJump || insn->opcode() == kMachineOpPseudoJump;
  }

  [[nodiscard]] BasicBlockOrder bb_order() const { return bb_order_; }

  void set_bb_order(BasicBlockOrder order) { bb_order_ = order; }

  using berberis::MachineIR::NewInsn;

  template <typename T, typename... Args>
  [[nodiscard]] T* NewInsn(Args... args) {
    return berberis::MachineIR::template NewInsn<T, Args...>(args...);
  }

  template <template <typename> typename InsnType>
  using MachineInsnType =
      MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>;

  template <template <typename> typename InsnType, size_t N>
  using GenArg = std::tuple_element_t<
      N,
      typename MachineInsnOperandsHelper<typename InsnType<
          typename CodeEmitter::Assemblers>::DeviceInsnInfo>::ConstructorArgsTuple>;

  template <template <typename> typename InsnType>
  [[nodiscard]] auto NewInsn()
      -> std::enable_if_t<
          std::tuple_size_v<typename MachineInsnOperandsHelper<typename InsnType<
              typename CodeEmitter::Assemblers>::DeviceInsnInfo>::ConstructorArgsTuple> == 0,
          MachineInsnType<InsnType>*> {
    return NewInsn<MachineInsnType<InsnType>>();
  }

  template <template <typename> typename InsnType>
  [[nodiscard]] auto NewInsn(GenArg<InsnType, 0> arg0)
      -> std::enable_if_t<
          std::tuple_size_v<typename MachineInsnOperandsHelper<typename InsnType<
              typename CodeEmitter::Assemblers>::DeviceInsnInfo>::ConstructorArgsTuple> == 1,
          MachineInsnType<InsnType>*> {
    return NewInsn<MachineInsnType<InsnType>, GenArg<InsnType, 0>>(arg0);
  }

  template <template <typename> typename InsnType>
  [[nodiscard]] auto NewInsn(GenArg<InsnType, 0> arg0, GenArg<InsnType, 1> arg1)
      -> std::enable_if_t<
          std::tuple_size_v<typename MachineInsnOperandsHelper<typename InsnType<
              typename CodeEmitter::Assemblers>::DeviceInsnInfo>::ConstructorArgsTuple> == 2,
          MachineInsnType<InsnType>*> {
    return NewInsn<MachineInsnType<InsnType>, GenArg<InsnType, 0>, GenArg<InsnType, 1>>(arg0, arg1);
  }

  template <template <typename> typename InsnType>
  [[nodiscard]] auto NewInsn(GenArg<InsnType, 0> arg0,
                             GenArg<InsnType, 1> arg1,
                             GenArg<InsnType, 2> arg2)
      -> std::enable_if_t<
          std::tuple_size_v<typename MachineInsnOperandsHelper<typename InsnType<
              typename CodeEmitter::Assemblers>::DeviceInsnInfo>::ConstructorArgsTuple> == 3,
          MachineInsnType<InsnType>*> {
    return NewInsn<MachineInsnType<InsnType>,
                   GenArg<InsnType, 0>,
                   GenArg<InsnType, 1>,
                   GenArg<InsnType, 2>>(arg0, arg1, arg2);
  }

  template <template <typename> typename InsnType>
  [[nodiscard]] auto NewInsn(GenArg<InsnType, 0> arg0,
                             GenArg<InsnType, 1> arg1,
                             GenArg<InsnType, 2> arg2,
                             GenArg<InsnType, 3> arg3)
      -> std::enable_if_t<
          std::tuple_size_v<typename MachineInsnOperandsHelper<typename InsnType<
              typename CodeEmitter::Assemblers>::DeviceInsnInfo>::ConstructorArgsTuple> == 4,
          MachineInsnType<InsnType>*> {
    return NewInsn<MachineInsnType<InsnType>,
                   GenArg<InsnType, 0>,
                   GenArg<InsnType, 1>,
                   GenArg<InsnType, 2>,
                   GenArg<InsnType, 3>>(arg0, arg1, arg2, arg3);
  }

  template <template <typename> typename InsnType>
  [[nodiscard]] auto NewInsn(GenArg<InsnType, 0> arg0,
                             GenArg<InsnType, 1> arg1,
                             GenArg<InsnType, 2> arg2,
                             GenArg<InsnType, 3> arg3,
                             GenArg<InsnType, 4> arg4)
      -> std::enable_if_t<
          std::tuple_size_v<typename MachineInsnOperandsHelper<typename InsnType<
              typename CodeEmitter::Assemblers>::DeviceInsnInfo>::ConstructorArgsTuple> == 5,
          MachineInsnType<InsnType>*> {
    return NewInsn<MachineInsnType<InsnType>,
                   GenArg<InsnType, 0>,
                   GenArg<InsnType, 1>,
                   GenArg<InsnType, 2>,
                   GenArg<InsnType, 3>,
                   GenArg<InsnType, 4>>(arg0, arg1, arg2, arg3, arg4);
  }

  template <template <typename> typename InsnType>
  [[nodiscard]] auto NewInsn(GenArg<InsnType, 0> arg0,
                             GenArg<InsnType, 1> arg1,
                             GenArg<InsnType, 2> arg2,
                             GenArg<InsnType, 3> arg3,
                             GenArg<InsnType, 4> arg4,
                             GenArg<InsnType, 5> arg5)
      -> std::enable_if_t<
          std::tuple_size_v<typename MachineInsnOperandsHelper<typename InsnType<
              typename CodeEmitter::Assemblers>::DeviceInsnInfo>::ConstructorArgsTuple> == 6,
          MachineInsnType<InsnType>*> {
    return NewInsn<MachineInsnType<InsnType>,
                   GenArg<InsnType, 0>,
                   GenArg<InsnType, 1>,
                   GenArg<InsnType, 2>,
                   GenArg<InsnType, 3>,
                   GenArg<InsnType, 4>,
                   GenArg<InsnType, 5>>(arg0, arg1, arg2, arg3, arg4, arg5);
  }

  template <template <typename> typename InsnType, typename... Args>
  [[nodiscard]] MachineInsnType<InsnType>* NewInsn(Args... args) {
    return NewInsn<MachineInsnType<InsnType>, Args...>(args...);
  }

 private:
  BasicBlockOrder bb_order_;
};

}  // namespace x86_64

}  // namespace berberis

#endif  // BERBERIS_BACKEND_X86_64_MACHINE_IR_H_
