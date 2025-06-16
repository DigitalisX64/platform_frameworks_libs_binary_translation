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

#include "frontend.h"

#include <cstddef>

#include "berberis/assembler/x86_64.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/guest_state/guest_state_arch.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/memory_region_reservation.h"
#include "berberis/runtime_primitives/platform.h"

namespace berberis {

using BranchOpcode = HeavyOptimizerFrontend::Decoder::BranchOpcode;
using FpRegister = HeavyOptimizerFrontend::FpRegister;
using Register = HeavyOptimizerFrontend::Register;

void HeavyOptimizerFrontend::CompareAndBranch(BranchOpcode opcode,
                                              Register arg1,
                                              Register arg2,
                                              int16_t offset) {
  auto ir = builder_.ir();
  auto cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* else_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, else_bb);

  builder_.Gen<PseudoCondBranch>(
      ToAssemblerCond(opcode), then_bb, else_bb, std::get<0>(Gen<x86_64::CmpqRegReg>(arg1, arg2)));

  builder_.StartBasicBlock(then_bb);
  GenJump(pc_ + offset);

  builder_.StartBasicBlock(else_bb);
}

void HeavyOptimizerFrontend::Branch(int32_t offset) {
  is_uncond_branch_ = true;
  GenJump(pc_ + offset);
}

void HeavyOptimizerFrontend::BranchRegister(Register src, int16_t offset) {
  is_uncond_branch_ = true;
  Register target = src;
  // Avoid the extra insn if unneeded.
  if (offset == 0) {
    // TODO(b/232598137) Maybe move this to translation cache?
    target = std::get<0>(Gen<x86_64::AndqRegImm>(target, ~int32_t{1}));
  } else {
    // TODO(b/232598137) Maybe move this to translation cache?
    target = std::get<0>(Gen<x86_64::AddqRegImm>(src, offset));
    target = std::get<0>(Gen<x86_64::AndqRegImm, kNoSSA>(target, ~int32_t{1}));
  }
  ExitRegionIndirect(target);
}

x86_64::Assembler::Condition HeavyOptimizerFrontend::ToAssemblerCond(BranchOpcode opcode) {
  switch (opcode) {
    case BranchOpcode::kBeq:
      return x86_64::Assembler::Condition::kEqual;
    case BranchOpcode::kBne:
      return x86_64::Assembler::Condition::kNotEqual;
    case BranchOpcode::kBlt:
      return x86_64::Assembler::Condition::kLess;
    case BranchOpcode::kBge:
      return x86_64::Assembler::Condition::kGreaterEqual;
    case BranchOpcode::kBltu:
      return x86_64::Assembler::Condition::kBelow;
    case BranchOpcode::kBgeu:
      return x86_64::Assembler::Condition::kAboveEqual;
  }
}

Register HeavyOptimizerFrontend::GetImm(uint64_t imm) {
  return std::get<0>(Gen<x86_64::MovqRegImm>(imm));
}

Register HeavyOptimizerFrontend::AllocTempReg() {
  return builder_.ir()->AllocVReg();
}

SimdReg HeavyOptimizerFrontend::AllocTempSimdReg() {
  return SimdReg{builder_.ir()->AllocVReg()};
}

void HeavyOptimizerFrontend::GenJump(GuestAddr target) {
  auto map_it = branch_targets_.find(target);
  if (map_it == branch_targets_.end()) {
    // Remember that this address was taken to help region formation. If we
    // translate it later the data will be overwritten with the actual location.
    branch_targets_[target] = MachineInsnPosition{};
  }

  // Checking pending signals only on back jumps guarantees no infinite loops
  // without pending signal checks.
  auto kind = target <= GetInsnAddr() ? PseudoJump::Kind::kJumpWithPendingSignalsCheck
                                      : PseudoJump::Kind::kJumpWithoutPendingSignalsCheck;

  builder_.Gen<PseudoJump>(target, kind);
}

void HeavyOptimizerFrontend::ExitGeneratedCode(GuestAddr target) {
  builder_.Gen<PseudoJump>(target, PseudoJump::Kind::kExitGeneratedCode);
}

void HeavyOptimizerFrontend::ExitRegionIndirect(Register target) {
  builder_.Gen<PseudoIndirectJump>(target);
}

void HeavyOptimizerFrontend::Undefined() {
  success_ = false;
  ExitGeneratedCode(GetInsnAddr());
  // We don't require region to end here as control flow may jump around
  // the undefined instruction, so handle it as an unconditional branch.
  is_uncond_branch_ = true;
}

bool HeavyOptimizerFrontend::IsRegionEndReached() const {
  if (!is_uncond_branch_) {
    return false;
  }

  auto map_it = branch_targets_.find(GetInsnAddr());
  // If this instruction following an unconditional branch isn't reachable by
  // some other branch - it's a region end.
  return map_it == branch_targets_.end();
}

void HeavyOptimizerFrontend::ResolveJumps() {
  if (!config::kLinkJumpsWithinRegion) {
    return;
  }
  auto ir = builder_.ir();

  MachineBasicBlockList bb_list_copy(ir->bb_list());
  for (auto bb : bb_list_copy) {
    if (bb->is_recovery()) {
      // Recovery blocks must exit region, do not try to resolve it into a local branch.
      continue;
    }

    const MachineInsn* last_insn = bb->insn_list().back();
    if (last_insn->opcode() != kMachineOpPseudoJump) {
      continue;
    }

    auto* jump = static_cast<const PseudoJump*>(last_insn);
    if (jump->kind() == PseudoJump::Kind::kSyscall ||
        jump->kind() == PseudoJump::Kind::kExitGeneratedCode) {
      // Syscall or generated code exit must always exit region.
      continue;
    }

    GuestAddr target = jump->target();
    auto map_it = branch_targets_.find(target);
    // All PseudoJump insns must add their targets to branch_targets.
    CHECK(map_it != branch_targets_.end());

    MachineInsnPosition pos = map_it->second;
    MachineBasicBlock* target_containing_bb = pos.first;
    if (!target_containing_bb) {
      // Branch target is not in the current region
      continue;
    }

    CHECK(pos.second.has_value());
    auto target_insn_it = pos.second.value();
    MachineBasicBlock* target_bb;
    if (target_insn_it == target_containing_bb->insn_list().begin()) {
      // We don't need to split if target_insn_it is at the beginning of target_containing_bb.
      target_bb = target_containing_bb;
    } else {
      // target_bb is split from target_containing_bb.
      target_bb = ir->SplitBasicBlock(target_containing_bb, target_insn_it);
      UpdateBranchTargetsAfterSplit(target, target_containing_bb, target_bb);

      // Make sure target_bb is also considered for jump resolution. Otherwise we may leave code
      // referenced by it unlinked from the rest of the IR.
      bb_list_copy.push_back(target_bb);

      // If bb is equal to target_containing_bb, then the branch instruction at the end of bb
      // is moved to the new target_bb, so we replace the instruction at the end of the
      // target_bb instead of bb.
      if (bb == target_containing_bb) {
        bb = target_bb;
      }
    }

    ReplaceJumpWithBranch(bb, target_bb);
  }
}

void HeavyOptimizerFrontend::ReplaceJumpWithBranch(MachineBasicBlock* bb,
                                                   MachineBasicBlock* target_bb) {
  auto ir = builder_.ir();
  const auto* last_insn = bb->insn_list().back();
  CHECK_EQ(last_insn->opcode(), kMachineOpPseudoJump);
  auto* jump = static_cast<const PseudoJump*>(last_insn);
  GuestAddr target = static_cast<const PseudoJump*>(jump)->target();
  // Do not invalidate this iterator as it may be a target for another jump.
  // Instead overwrite the instruction.
  auto jump_it = std::prev(bb->insn_list().end());

  if (jump->kind() == PseudoJump::Kind::kJumpWithoutPendingSignalsCheck) {
    // Simple branch for forward jump.
    *jump_it = ir->NewInsn<PseudoBranch>(target_bb);
    ir->AddEdge(bb, target_bb);
  } else {
    CHECK(jump->kind() == PseudoJump::Kind::kJumpWithPendingSignalsCheck);
    // See EmitCheckSignalsAndMaybeReturn.
    auto* exit_bb = ir->NewBasicBlock();
    // Note that we intentionally don't mark exit_bb as recovery and therefore don't request its
    // reordering away from hot code spots. target_bb is a back branch and is unlikely to be a
    // fall-through jump for the current bb. At the same time exit_bb can be a fall-through jump
    // and benchmarks benefit from it.
    const size_t offset = offsetof(ThreadState, pending_signals_status);
    auto* cmpb = ir->NewInsn<x86_64::CmpbOpImm>({.base = x86_64::kMachineRegRBP, .disp = offset},
                                                kPendingSignalsPresent,
                                                GetFlagsRegister());
    *jump_it = cmpb;
    auto* cond_branch = ir->NewInsn<PseudoCondBranch>(
        x86_64::Assembler::Condition::kEqual, exit_bb, target_bb, GetFlagsRegister());
    bb->insn_list().push_back(cond_branch);

    builder_.StartBasicBlock(exit_bb);
    ExitGeneratedCode(target);

    ir->AddEdge(bb, exit_bb);
    ir->AddEdge(bb, target_bb);
  }
}

void HeavyOptimizerFrontend::UpdateBranchTargetsAfterSplit(GuestAddr addr,
                                                           const MachineBasicBlock* old_bb,
                                                           MachineBasicBlock* new_bb) {
  auto map_it = branch_targets_.find(addr);
  CHECK(map_it != branch_targets_.end());
  while (map_it != branch_targets_.end() && map_it->second.first == old_bb) {
    map_it->second.first = new_bb;
    map_it++;
  }
}

Register HeavyOptimizerFrontend::GetReg(uint8_t reg) {
  CHECK_LT(reg, kNumGuestRegs);
  Register dst = AllocTempReg();
  builder_.GenGet(dst, GetThreadStateRegOffset(reg));
  return dst;
}

void HeavyOptimizerFrontend::SetReg(uint8_t reg, Register value) {
  CHECK_LT(reg, kNumGuestRegs);
  if (success()) {
    builder_.GenPut(GetThreadStateRegOffset(reg), value);
  }
}

FpRegister HeavyOptimizerFrontend::GetFpReg(uint8_t reg) {
  FpRegister result = AllocTempSimdReg();
  builder_.GenGetSimd<8>(result.machine_reg(), GetThreadStateFRegOffset(reg));
  return result;
}

void HeavyOptimizerFrontend::Nop() {}

Register HeavyOptimizerFrontend::Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
  using OpOpcode = Decoder::OpOpcode;
  using Condition = x86_64::Assembler::Condition;
  switch (opcode) {
    case OpOpcode::kAdd:
      return std::get<0>(Gen<x86_64::AddqRegReg>(arg1, arg2));
    case OpOpcode::kSub:
      return std::get<0>(Gen<x86_64::SubqRegReg>(arg1, arg2));
    case OpOpcode::kAnd:
      return std::get<0>(Gen<x86_64::AndqRegReg>(arg1, arg2));
    case OpOpcode::kOr:
      return std::get<0>(Gen<x86_64::OrqRegReg>(arg1, arg2));
    case OpOpcode::kXor:
      return std::get<0>(Gen<x86_64::XorqRegReg>(arg1, arg2));
    case OpOpcode::kSll:
      return std::get<0>(Gen<x86_64::ShlqRegReg>(arg1, arg2));
    case OpOpcode::kSrl:
      return std::get<0>(Gen<x86_64::ShrqRegReg>(arg1, arg2));
    case OpOpcode::kSra:
      return std::get<0>(Gen<x86_64::SarqRegReg>(arg1, arg2));
    case OpOpcode::kSlt:
      return std::get<0>(
          Gen<x86_64::MovzxbqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::SetccReg, kNoSSA>(
              Condition::kLess, std::get<0>(Gen<x86_64::CmpqRegReg>(arg1, arg2))))));
    case OpOpcode::kSltu:
      return std::get<0>(
          Gen<x86_64::MovzxbqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::SetccReg, kNoSSA>(
              Condition::kBelow, std::get<0>(Gen<x86_64::CmpqRegReg>(arg1, arg2))))));
    case OpOpcode::kMul:
      return std::get<0>(Gen<x86_64::ImulqRegReg>(arg1, arg2));
    case OpOpcode::kMulh:
      return std::get<1>(Gen<x86_64::ImulqRegRegReg>(arg1, arg2));
    case OpOpcode::kMulhsu: {
      auto [low, high, mul_flags] = Gen<x86_64::MulqRegRegReg>(arg1, arg2);
      auto [adjust, imul_flags] =
          Gen<x86_64::ImulqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::SarqRegImm>(arg1, 63)), arg2);
      return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(adjust, high));
    }
    case OpOpcode::kMulhu:
      return std::get<1>(Gen<x86_64::MulqRegRegReg>(arg1, arg2));
    case OpOpcode::kAndn:
      if (host_platform::kHasBMI) {
        return std::get<0>(Gen<x86_64::AndnqRegRegReg>(arg2, arg1));
      } else {
        return std::get<0>(
            Gen<x86_64::AndqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::NotqReg>(arg2)), arg1));
      }
      break;
    case OpOpcode::kOrn:
      return std::get<0>(
          Gen<x86_64::OrqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::NotqReg>(arg2)), arg1));
    case OpOpcode::kXnor:
      return std::get<0>(
          Gen<x86_64::NotqReg, kNoSSA>(std::get<0>(Gen<x86_64::XorqRegReg>(arg1, arg2))));
    default:
      Undefined();
      return {};
  }
}

Register HeavyOptimizerFrontend::Op32(Decoder::Op32Opcode opcode, Register arg1, Register arg2) {
  using Op32Opcode = Decoder::Op32Opcode;
  switch (opcode) {
    case Op32Opcode::kAddw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::AddlRegReg>(arg1, arg2))));
    case Op32Opcode::kSubw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::SublRegReg>(arg1, arg2))));
    case Op32Opcode::kSllw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::ShllRegReg>(arg1, arg2))));
    case Op32Opcode::kSrlw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::ShrlRegReg>(arg1, arg2))));
    case Op32Opcode::kSraw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::SarlRegReg>(arg1, arg2))));
    case Op32Opcode::kMulw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::ImullRegReg>(arg1, arg2))));
    default:
      Undefined();
      return {};
  }
}

Register HeavyOptimizerFrontend::OpImm(Decoder::OpImmOpcode opcode, Register arg, int16_t imm) {
  using OpImmOpcode = Decoder::OpImmOpcode;
  using Condition = x86_64::Assembler::Condition;
  switch (opcode) {
    case OpImmOpcode::kAddi:
      return std::get<0>(Gen<x86_64::AddqRegImm>(arg, imm));
    case OpImmOpcode::kSlti:
      return std::get<0>(
          Gen<x86_64::MovsxbqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::SetccReg, kNoSSA>(
              Condition::kLess, std::get<0>(Gen<x86_64::CmpqRegImm>(arg, imm))))));
    case OpImmOpcode::kSltiu:
      return std::get<0>(
          Gen<x86_64::MovsxbqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::SetccReg, kNoSSA>(
              Condition::kBelow, std::get<0>(Gen<x86_64::CmpqRegImm>(arg, imm))))));
    case OpImmOpcode::kXori:
      return std::get<0>(Gen<x86_64::XorqRegImm>(arg, imm));
    case OpImmOpcode::kOri:
      return std::get<0>(Gen<x86_64::OrqRegImm>(arg, imm));
    case OpImmOpcode::kAndi:
      return std::get<0>(Gen<x86_64::AndqRegImm>(arg, imm));
    default:
      Undefined();
      return {};
  }
}

Register HeavyOptimizerFrontend::OpImm32(Decoder::OpImm32Opcode opcode, Register arg, int16_t imm) {
  switch (opcode) {
    case Decoder::OpImm32Opcode::kAddiw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::AddlRegImm>(arg, imm))));
    default:
      Undefined();
      return {};
  }
}

Register HeavyOptimizerFrontend::Slli(Register arg, int8_t imm) {
  return std::get<0>(Gen<x86_64::ShlqRegImm>(arg, imm));
}

Register HeavyOptimizerFrontend::Srli(Register arg, int8_t imm) {
  return std::get<0>(Gen<x86_64::ShrqRegImm>(arg, imm));
}

Register HeavyOptimizerFrontend::Srai(Register arg, int8_t imm) {
  return std::get<0>(Gen<x86_64::SarqRegImm>(arg, imm));
}

Register HeavyOptimizerFrontend::ShiftImm32(Decoder::ShiftImm32Opcode opcode,
                                            Register arg,
                                            uint16_t imm) {
  using ShiftImm32Opcode = Decoder::ShiftImm32Opcode;
  switch (opcode) {
    case ShiftImm32Opcode::kSlliw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::ShllRegImm>(arg, imm))));
    case ShiftImm32Opcode::kSrliw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::ShrlRegImm>(arg, imm))));
    case ShiftImm32Opcode::kSraiw:
      return std::get<0>(
          Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::SarlRegImm>(arg, imm))));
    default:
      Undefined();
      return {};
  }
}

Register HeavyOptimizerFrontend::Rori(Register arg, int8_t shamt) {
  return std::get<0>(Gen<x86_64::RorqRegImm>(arg, shamt));
}

Register HeavyOptimizerFrontend::Roriw(Register arg, int8_t shamt) {
  return std::get<0>(
      Gen<x86_64::MovsxlqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::RorlRegImm>(arg, shamt))));
}

Register HeavyOptimizerFrontend::Lui(int32_t imm) {
  return std::get<0>(Gen<x86_64::MovqRegImm>(imm));
}

Register HeavyOptimizerFrontend::Auipc(int32_t imm) {
  return std::get<0>(Gen<x86_64::AddqRegImm>(GetImm(GetInsnAddr()), imm));
}

void HeavyOptimizerFrontend::Store(Decoder::MemoryDataOperandType operand_type,
                                   Register arg,
                                   int16_t offset,
                                   Register data) {
  int32_t sx_offset{offset};
  StoreWithoutRecovery(operand_type, arg, sx_offset, data);
  GenRecoveryBlockForLastInsn();
}

Register HeavyOptimizerFrontend::Load(Decoder::LoadOperandType operand_type,
                                      Register arg,
                                      int16_t offset) {
  int32_t sx_offset{offset};
  auto res = LoadWithoutRecovery(operand_type, arg, sx_offset);
  GenRecoveryBlockForLastInsn();
  return res;
}

void HeavyOptimizerFrontend::GenRecoveryBlockForLastInsn() {
  // TODO(b/311240558) Accurate Sigsegv?
  auto* ir = builder_.ir();
  auto* current_bb = builder_.bb();
  auto* continue_bb = ir->NewBasicBlock();
  auto* recovery_bb = ir->NewBasicBlock();
  ir->AddEdge(current_bb, continue_bb);
  ir->AddEdge(current_bb, recovery_bb);

  builder_.SetRecoveryPointAtLastInsn(recovery_bb);

  // Note, even though there are two bb successors, we only explicitly branch to
  // the continue_bb, since jump to the recovery_bb is set up by the signal
  // handler.
  builder_.Gen<PseudoBranch>(continue_bb);

  builder_.StartBasicBlock(recovery_bb);
  ExitGeneratedCode(GetInsnAddr());

  builder_.StartBasicBlock(continue_bb);
}

//
//  Methods that are not part of SemanticsListener implementation.
//
void HeavyOptimizerFrontend::StartInsn() {
  if (is_uncond_branch_) {
    auto* ir = builder_.ir();
    builder_.StartBasicBlock(ir->NewBasicBlock());
  }

  is_uncond_branch_ = false;
  // The iterators in branch_targets are the last iterators before generating an insn.
  // We advance iterators by one step in Finalize(), as we'll use it to iterate
  // the sub-list of instructions starting from the first one for the given
  // guest address.

  // If a basic block is empty before generating insn, an empty optional typed
  // value is returned. We will resolve it to the first insn of the basic block
  // in Finalize().
  branch_targets_[GetInsnAddr()] = builder_.GetMachineInsnPosition();
}

void HeavyOptimizerFrontend::Finalize(GuestAddr stop_pc) {
  // Make sure the last basic block isn't empty before fixing iterators in
  // branch_targets.
  if (builder_.bb()->insn_list().empty() ||
      !builder_.ir()->IsControlTransfer(builder_.bb()->insn_list().back())) {
    GenJump(stop_pc);
  }

  // This loop advances the iterators in the branch_targets by one. Because in
  // StartInsn(), we saved the iterator to the last insn before we generate the
  // first insn for each guest address. If an insn is saved as an empty optional,
  // then the basic block is empty before we generate the first insn for the
  // guest address. So we resolve it to the first insn in the basic block.
  for (auto& [unused_address, insn_pos] : branch_targets_) {
    auto& [bb, insn_it] = insn_pos;
    if (!bb) {
      // Branch target is not in the current region.
      continue;
    }

    if (insn_it.has_value()) {
      insn_it.value()++;
    } else {
      // Make sure bb isn't still empty.
      CHECK(!bb->insn_list().empty());
      insn_it = bb->insn_list().begin();
    }
  }

  ResolveJumps();
}

Register HeavyOptimizerFrontend::LoadWithoutRecovery(Decoder::LoadOperandType operand_type,
                                                     Register base,
                                                     int32_t disp) {
  switch (operand_type) {
    case Decoder::LoadOperandType::k8bitUnsigned:
      return std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = base, .disp = disp}));
    case Decoder::LoadOperandType::k16bitUnsigned:
      return std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = base, .disp = disp}));
    case Decoder::LoadOperandType::k32bitUnsigned:
      return std::get<0>(Gen<x86_64::MovlRegOp>({.base = base, .disp = disp}));
    case Decoder::LoadOperandType::k64bit:
      return std::get<0>(Gen<x86_64::MovqRegOp>({.base = base, .disp = disp}));
    case Decoder::LoadOperandType::k8bitSigned:
      return std::get<0>(Gen<x86_64::MovsxbqRegOp>({.base = base, .disp = disp}));
    case Decoder::LoadOperandType::k16bitSigned:
      return std::get<0>(Gen<x86_64::MovsxwqRegOp>({.base = base, .disp = disp}));
    case Decoder::LoadOperandType::k32bitSigned:
      return std::get<0>(Gen<x86_64::MovsxlqRegOp>({.base = base, .disp = disp}));
    default:
      Undefined();
      return {};
  }
}

Register HeavyOptimizerFrontend::LoadWithoutRecovery(Decoder::LoadOperandType operand_type,
                                                     Register base,
                                                     Register index,
                                                     int32_t disp) {
  switch (operand_type) {
    case Decoder::LoadOperandType::k8bitUnsigned:
      return std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = base, .index = index, .disp = disp}));
    case Decoder::LoadOperandType::k16bitUnsigned:
      return std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = base, .index = index, .disp = disp}));
    case Decoder::LoadOperandType::k32bitUnsigned:
      return std::get<0>(Gen<x86_64::MovlRegOp>({.base = base, .index = index, .disp = disp}));
    case Decoder::LoadOperandType::k64bit:
      return std::get<0>(Gen<x86_64::MovqRegOp>({.base = base, .index = index, .disp = disp}));
    case Decoder::LoadOperandType::k8bitSigned:
      return std::get<0>(Gen<x86_64::MovsxbqRegOp>({.base = base, .index = index, .disp = disp}));
    case Decoder::LoadOperandType::k16bitSigned:
      return std::get<0>(Gen<x86_64::MovsxwqRegOp>({.base = base, .index = index, .disp = disp}));
    case Decoder::LoadOperandType::k32bitSigned:
      return std::get<0>(Gen<x86_64::MovsxlqRegOp>({.base = base, .index = index, .disp = disp}));
    default:
      Undefined();
      return {};
  }
}

Register HeavyOptimizerFrontend::UpdateCsr(Decoder::CsrOpcode opcode, Register arg, Register csr) {
  switch (opcode) {
    case Decoder::CsrOpcode::kCsrrs:
      return std::get<0>(Gen<x86_64::OrqRegReg>(arg, csr));
    case Decoder::CsrOpcode::kCsrrc:
      if (host_platform::kHasBMI) {
        return std::get<0>(Gen<x86_64::AndnqRegRegReg>(arg, csr));
      } else {
        return std::get<0>(
            Gen<x86_64::AndqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::NotqReg>(arg)), csr));
      }
    default:
      Undefined();
      return {};
  }
}

Register HeavyOptimizerFrontend::UpdateCsr(Decoder::CsrImmOpcode opcode, int8_t imm, Register csr) {
  switch (opcode) {
    case Decoder::CsrImmOpcode::kCsrrwi:
      return std::get<0>(Gen<x86_64::MovlRegImm>(imm));
    case Decoder::CsrImmOpcode::kCsrrsi:
      return std::get<0>(
          Gen<x86_64::OrqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::MovlRegImm>(imm)), csr));
    case Decoder::CsrImmOpcode::kCsrrci:
      return std::get<0>(
          Gen<x86_64::AndqRegReg, kNoSSA>(std::get<0>(Gen<x86_64::MovqRegImm>(~imm)), csr));
    default:
      Undefined();
      return {};
  }
}

void HeavyOptimizerFrontend::StoreWithoutRecovery(Decoder::MemoryDataOperandType operand_type,
                                                  Register base,
                                                  int32_t disp,
                                                  Register data) {
  switch (operand_type) {
    case Decoder::MemoryDataOperandType::k8bit:
      Gen<x86_64::MovbOpReg>({.base = base, .disp = disp}, data);
      break;
    case Decoder::MemoryDataOperandType::k16bit:
      Gen<x86_64::MovwOpReg>({.base = base, .disp = disp}, data);
      break;
    case Decoder::MemoryDataOperandType::k32bit:
      Gen<x86_64::MovlOpReg>({.base = base, .disp = disp}, data);
      break;
    case Decoder::MemoryDataOperandType::k64bit:
      Gen<x86_64::MovqOpReg>({.base = base, .disp = disp}, data);
      break;
    default:
      return Undefined();
  }
}

void HeavyOptimizerFrontend::StoreWithoutRecovery(Decoder::MemoryDataOperandType operand_type,
                                                  Register base,
                                                  Register index,
                                                  int32_t disp,
                                                  Register data) {
  switch (operand_type) {
    case Decoder::MemoryDataOperandType::k8bit:
      Gen<x86_64::MovbOpReg>(
          {.base = base, .index = index, x86_64::Assembler::kTimesOne, .disp = disp}, data);
      break;
    case Decoder::MemoryDataOperandType::k16bit:
      Gen<x86_64::MovwOpReg>(
          {.base = base, .index = index, x86_64::Assembler::kTimesOne, .disp = disp}, data);
      break;
    case Decoder::MemoryDataOperandType::k32bit:
      Gen<x86_64::MovlOpReg>(
          {.base = base, .index = index, x86_64::Assembler::kTimesOne, .disp = disp}, data);
      break;
    case Decoder::MemoryDataOperandType::k64bit:
      Gen<x86_64::MovqOpReg>(
          {.base = base, .index = index, x86_64::Assembler::kTimesOne, .disp = disp}, data);
      break;
    default:
      return Undefined();
  }
}

// Ordering affecting I/O devices is not relevant to user-space code thus we just ignore bits
// related to devices I/O.
void HeavyOptimizerFrontend::Fence(Decoder::FenceOpcode /* opcode */,
                                   Register /* src */,
                                   bool sw,
                                   bool sr,
                                   bool /* so */,
                                   bool /* si */,
                                   bool pw,
                                   bool pr,
                                   bool /* po */,
                                   bool /* pi */) {
  // Two types of fences (total store ordering fence and normal fence) are supposed to be
  // processed differently, but only for the “read_fence && write_fence” case (otherwise total
  // store ordering fence becomes normal fence for the “forward compatibility”), yet because x86
  // doesn't distinguish between these two types of fences and since we are supposed to map all
  // not-yet defined fences to normal fence (again, for the “forward compatibility”) it's Ok to
  // just ignore opcode field.
  bool read_fence = sr | pr;
  bool write_fence = sw | pw;
  if (read_fence) {
    if (write_fence) {
      Gen<x86_64::Mfence>();
    } else {
      Gen<x86_64::Lfence>();
    }
  } else if (write_fence) {
    Gen<x86_64::Sfence>();
  }
}

void HeavyOptimizerFrontend::MemoryRegionReservationLoad(Register aligned_addr) {
  // Store aligned_addr in CPUState.
  int32_t address_offset = GetThreadStateReservationAddressOffset();
  Gen<x86_64::MovqOpReg>({.base = x86_64::kMachineRegRBP, .disp = address_offset}, aligned_addr);

  // MemoryRegionReservation::SetOwner(aligned_addr, &(state->cpu)).
  builder_.GenCallImm(bit_cast<uint64_t>(&MemoryRegionReservation::SetOwner),
                      GetFlagsRegister(),
                      std::array<x86_64::CallImm::Arg, 2>{{
                          {aligned_addr, x86_64::CallImm::kIntRegType},
                          {x86_64::kMachineRegRBP, x86_64::CallImm::kIntRegType},
                      }});

  // Load reservation value and store it in CPUState.
  auto [reservation] = Gen<x86_64::MovqRegOp>({.base = aligned_addr});
  int32_t value_offset = GetThreadStateReservationValueOffset();
  Gen<x86_64::MovqOpReg>({.base = x86_64::kMachineRegRBP, .disp = value_offset}, reservation);
}

Register HeavyOptimizerFrontend::MemoryRegionReservationExchange(Register aligned_addr,
                                                                 Register curr_reservation_value) {
  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  auto* addr_match_bb = ir->NewBasicBlock();
  auto* failure_bb = ir->NewBasicBlock();
  auto* continue_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, addr_match_bb);
  ir->AddEdge(cur_bb, failure_bb);
  ir->AddEdge(failure_bb, continue_bb);
  Register result = AllocTempReg();

  // MemoryRegionReservation::Clear.
  int32_t address_offset = GetThreadStateReservationAddressOffset();
  auto [stored_aligned_addr] =
      Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = address_offset});
  Gen<x86_64::MovqOpImm>({.base = x86_64::kMachineRegRBP, .disp = address_offset}, kNullGuestAddr);
  // Compare aligned_addr to the one in CPUState.
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNotEqual,
      failure_bb,
      addr_match_bb,
      std::get<0>(Gen<x86_64::CmpqRegReg>(stored_aligned_addr, aligned_addr)));

  builder_.StartBasicBlock(addr_match_bb);
  // Load new reservation value into integer register where CmpXchgq expects it.
  int32_t value_offset = GetThreadStateReservationValueOffset();
  auto [new_reservation_value] =
      Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = value_offset});

  MemoryRegionReservationSwapWithLockedOwner(
      aligned_addr, curr_reservation_value, new_reservation_value, failure_bb);

  ir->AddEdge(builder_.bb(), continue_bb);
  // Pseudo-def for use-def operand of XOR to make sure data-flow is integrate.
  builder_.Gen<PseudoDefReg>(result);
  builder_.Gen<x86_64::XorqRegReg>(result, result, GetFlagsRegister());
  builder_.Gen<PseudoBranch>(continue_bb);

  builder_.StartBasicBlock(failure_bb);
  builder_.Gen<x86_64::MovqRegImm>(result, 1);
  builder_.Gen<PseudoBranch>(continue_bb);

  builder_.StartBasicBlock(continue_bb);

  return result;
}

void HeavyOptimizerFrontend::MemoryRegionReservationSwapWithLockedOwner(
    Register aligned_addr,
    Register curr_reservation_value,
    Register new_reservation_value,
    MachineBasicBlock* failure_bb) {
  auto* ir = builder_.ir();
  auto* lock_success_bb = ir->NewBasicBlock();
  auto* swap_success_bb = ir->NewBasicBlock();
  ir->AddEdge(builder_.bb(), lock_success_bb);
  ir->AddEdge(builder_.bb(), failure_bb);
  ir->AddEdge(lock_success_bb, swap_success_bb);
  ir->AddEdge(lock_success_bb, failure_bb);

  // lock_entry = MemoryRegionReservation::TryLock(aligned_addr, &(state->cpu)).
  auto* call = builder_.GenCallImm(bit_cast<uint64_t>(&MemoryRegionReservation::TryLock),
                                   GetFlagsRegister(),
                                   std::array<x86_64::CallImm::Arg, 2>{{
                                       {aligned_addr, x86_64::CallImm::kIntRegType},
                                       {x86_64::kMachineRegRBP, x86_64::CallImm::kIntRegType},
                                   }});
  Register lock_entry = AllocTempReg();
  // Limit life-time of a narrow reg-class call result.
  builder_.Gen<PseudoCopy>(lock_entry, call->IntResultAt(0), 8);
  builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kZero,
                                 failure_bb,
                                 lock_success_bb,
                                 std::get<0>(Gen<x86_64::TestqRegReg>(lock_entry, lock_entry)));

  builder_.StartBasicBlock(lock_success_bb);
  MachineReg host_flags;
  std::tie(curr_reservation_value, host_flags) = Gen<x86_64::LockCmpXchgqRegOpReg>(
      curr_reservation_value, {.base = aligned_addr}, new_reservation_value);

  // MemoryRegionReservation::Unlock(lock_entry)
  Gen<x86_64::MovqOpImm>({.base = lock_entry}, 0);
  // Zero-flag is set if CmpXchg is successful.
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNotZero, failure_bb, swap_success_bb, host_flags);

  builder_.StartBasicBlock(swap_success_bb);
}

}  // namespace berberis
