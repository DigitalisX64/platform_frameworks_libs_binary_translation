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

#include <cstdint>
#include <string>

#include "berberis/assembler/x86_64.h"
#include "berberis/backend/code_emitter.h"
#include "berberis/backend/common/machine_ir.h"  // IWYU pragma: export.
#include "berberis/base/arena_alloc.h"
#include "berberis/device_arch_info/x86_64/device_arch_info.h"
#include "berberis/guest_state/guest_state_arch.h"

namespace berberis {

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

// Some instructions form groups. E.g. memory-accesses typically have 4 versions: Absolute, Base,
// Index Base+Index.
//
// To ensure that there enough bits to separate these versions we reserve top 8 bits.
inline constexpr int kLowMachineOpcodeBits = 24;

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

  bool IsCPUStateGet() {
    if (opcode() != kMachineOpMovqRegMemBaseDisp && opcode() != kMachineOpMovdqaXRegMemBaseDisp &&
        opcode() != kMachineOpMovwRegMemBaseDisp && opcode() != kMachineOpMovsdXRegMemBaseDisp) {
      return false;
    }

    // Check that it is not for ThreadState fields outside of CPUState.
    if (disp() >= sizeof(CPUState)) {
      return false;
    }

    // reservation_value is loaded in HeavyOptimizerFrontend::AtomicLoad and written
    // in HeavyOptimizerFrontend::AtomicStore partially (for performance
    // reasons), which is not supported by our context optimizer.
    auto reservation_value_offset = offsetof(ThreadState, cpu.reservation_value);
    if (disp() >= reservation_value_offset &&
        disp() < reservation_value_offset + sizeof(Reservation)) {
      return false;
    }

    return RegAt(1) == kCPUStatePointer;
  }

  bool IsCPUStatePut() {
    if (opcode() != kMachineOpMovqMemBaseDispReg && opcode() != kMachineOpMovdqaMemBaseDispXReg &&
        opcode() != kMachineOpMovwMemBaseDispReg && opcode() != kMachineOpMovsdMemBaseDispXReg) {
      return false;
    }

    // Check that it is not for ThreadState fields outside of CPUState.
    if (disp() >= sizeof(CPUState)) {
      return false;
    }

    // reservation_value is loaded in HeavyOptimizerFrontend::AtomicLoad and written
    // in HeavyOptimizerFrontend::AtomicStore partially (for performance
    // reasons), which is not supported by our context optimizer.
    auto reservation_value_offset = offsetof(ThreadState, cpu.reservation_value);
    if (disp() >= reservation_value_offset &&
        disp() < reservation_value_offset + sizeof(Reservation)) {
      return false;
    }

    return RegAt(0) == kCPUStatePointer;
  }

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

using MachineInsnForArch = MachineInsnX86_64;

#include "gen_machine_ir_x86_64-inl.h"  // NOLINT generated file!

class MachineInfo {
 public:
#include "machine_info_x86_64-inl.h"  // NOLINT generated file!
};

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

  [[nodiscard]] bool IsCPUStateGet(MachineInsn* insn) const {
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

  [[nodiscard]] bool IsCPUStatePut(MachineInsn* insn) const {
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

  [[nodiscard]] static bool IsControlTransfer(MachineInsn* insn) {
    return insn->opcode() == kMachineOpPseudoBranch ||
           insn->opcode() == kMachineOpPseudoCondBranch ||
           insn->opcode() == kMachineOpPseudoIndirectJump || insn->opcode() == kMachineOpPseudoJump;
  }

  [[nodiscard]] BasicBlockOrder bb_order() const { return bb_order_; }

  void set_bb_order(BasicBlockOrder order) { bb_order_ = order; }

 private:
  BasicBlockOrder bb_order_;
};

}  // namespace x86_64

}  // namespace berberis

#endif  // BERBERIS_BACKEND_X86_64_MACHINE_IR_H_
