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

#include "berberis/backend/x86_64/insn_folding.h"

#include <cstdint>
#include <tuple>

#include "berberis/backend/x86_64/machine_ir.h"

#include "berberis/backend/code_emitter.h"  // for CodeEmitter::Condition
#include "berberis/base/algorithm.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/logging.h"

namespace berberis::x86_64 {

void DefMap::MapDefRegs(MachineInsnList::iterator insn_it) {
  const MachineInsn* insn = *insn_it;
  for (int op = 0; op < insn->NumRegOperands(); ++op) {
    MachineReg reg = insn->RegAt(op);
    if (insn->RegKindAt(op).RegClass()->IsSubsetOf(&x86_64::kFLAGS)) {
      if (flags_reg_ == kInvalidMachineReg) {
        flags_reg_ = reg;
      }
      // Some optimizations assume flags is the same virtual register everywhere.
      CHECK(reg == flags_reg_);
    }
    if (insn->RegKindAt(op).IsDef()) {
      Set(reg, insn_it);
    }
  }
}

void DefMap::ProcessInsn(MachineInsnList::iterator insn_it) {
  MapDefRegs(insn_it);
  ++index_;
}

void DefMap::Initialize() {
  std::fill(def_map_.begin(), def_map_.end(), std::pair(std::nullopt, 0));
  flags_reg_ = kInvalidMachineReg;
  index_ = 0;
}

std::optional<uint64_t> InsnFolding::GetImmValueIfPossible(MachineReg reg) const {
  auto [general_insn_it, _] = FindNonPseudoCopyDef(reg);
  if (!general_insn_it.has_value()) {
    return std::nullopt;
  }
  const MachineInsn* general_insn = *general_insn_it.value();
  const auto* insn = AsMachineInsnX86_64(general_insn);
  if (insn->opcode() == kMachineOpMovqRegImm) {
    return insn->imm();
  } else if (insn->opcode() == kMachineOpMovlRegImm) {
    // Take into account zero-extension by MOVL.
    return static_cast<uint64_t>(static_cast<uint32_t>(insn->imm()));
  }
  return std::nullopt;
}

MachineInsn* InsnFolding::NewImmInsnFromRegInsn(const MachineInsn* insn, int32_t imm32) {
  MachineInsn* folded_insn;
  switch (insn->opcode()) {
    case kMachineOpAddqRegReg:
      folded_insn = machine_ir_->NewInsn<AddqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpSubqRegReg:
      folded_insn = machine_ir_->NewInsn<SubqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpCmpqRegReg:
      folded_insn = machine_ir_->NewInsn<CmpqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpOrqRegReg:
      folded_insn = machine_ir_->NewInsn<OrqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpXorqRegReg:
      folded_insn = machine_ir_->NewInsn<XorqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpAndqRegReg:
      folded_insn = machine_ir_->NewInsn<AndqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpTestqRegReg:
      folded_insn = machine_ir_->NewInsn<TestqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpShlqRegReg:
      folded_insn = machine_ir_->NewInsn<ShlqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpShrqRegReg:
      folded_insn = machine_ir_->NewInsn<ShrqRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpMovlRegReg:
      folded_insn = machine_ir_->NewInsn<MovlRegImm>(insn->RegAt(0), imm32);
      break;
    case kMachineOpAddlRegReg:
      folded_insn = machine_ir_->NewInsn<AddlRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpSublRegReg:
      folded_insn = machine_ir_->NewInsn<SublRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpCmplRegReg:
      folded_insn = machine_ir_->NewInsn<CmplRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpOrlRegReg:
      folded_insn = machine_ir_->NewInsn<OrlRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpXorlRegReg:
      folded_insn = machine_ir_->NewInsn<XorlRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpAndlRegReg:
      folded_insn = machine_ir_->NewInsn<AndlRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpTestlRegReg:
      folded_insn = machine_ir_->NewInsn<TestlRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpShllRegReg:
      folded_insn = machine_ir_->NewInsn<ShllRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpShrlRegReg:
      folded_insn = machine_ir_->NewInsn<ShrlRegImm>(insn->RegAt(0), imm32, insn->RegAt(2));
      break;
    case kMachineOpMovlMemBaseDispReg:
      folded_insn = machine_ir_->NewInsn<MovlMemBaseDispImm>(
          insn->RegAt(0), AsMachineInsnX86_64(insn)->disp(), imm32);
      break;
    case kMachineOpMovqMemBaseDispReg:
      folded_insn = machine_ir_->NewInsn<MovqMemBaseDispImm>(
          insn->RegAt(0), AsMachineInsnX86_64(insn)->disp(), imm32);
      break;
    default:
      LOG_ALWAYS_FATAL("unexpected opcode");
  }
  // Inherit the additional attributes.
  folded_insn->set_recovery_bb(insn->recovery_bb());
  folded_insn->set_recovery_pc(insn->recovery_pc());
  return folded_insn;
}

std::tuple<std::optional<MachineInsnList::iterator>, int> InsnFolding::FindNonPseudoCopyDef(
    MachineReg src_reg) const {
  auto [def_insn_it, def_insn_pos] = def_map_.Get(src_reg);
  while (def_insn_it.has_value()) {
    const MachineInsn* def_insn = *def_insn_it.value();
    if (def_insn->opcode() != kMachineOpPseudoCopy) {
      return {def_insn_it, def_insn_pos};
    }
    std::tie(def_insn_it, def_insn_pos) = def_map_.Get(def_insn->RegAt(1), def_insn_pos);
  }
  return {std::nullopt, 0};
}

bool InsnFolding::IsWritingSameFlagsValue(MachineInsnList::iterator write_flags_insn_it) const {
  const MachineInsn* write_flags_insn = *write_flags_insn_it;
  CHECK(write_flags_insn && write_flags_insn->opcode() == kMachineOpPseudoWriteFlags);
  MachineReg src_reg = write_flags_insn->RegAt(0);
  auto [def_insn_it, def_insn_pos] = FindNonPseudoCopyDef(src_reg);
  if (!def_insn_it.has_value()) {
    return false;
  }
  const MachineInsn* def_insn = *def_insn_it.value();
  if (def_insn->opcode() != kMachineOpPseudoReadFlags) {
    return false;
  }
  // Instruction is PseudoReadFlags.
  if (write_flags_insn->RegAt(1) != def_insn->RegAt(1)) {
    return false;
  }
  auto [flag_def_insn, _] = def_map_.Get(write_flags_insn->RegAt(1), def_insn_pos);
  return flag_def_insn.has_value();
}

template <bool kIsInput64Bit>
std::tuple<FoldingType, MachineInsn*> InsnFolding::TryFoldImmediateInput(
    MachineInsnList::iterator insn_it) {
  const MachineInsn* insn = *insn_it;
  auto src1 = insn->RegAt(1);
  std::optional<uint64_t> imm64_1 = GetImmValueIfPossible(src1);
  if (!imm64_1.has_value()) {
    return {FoldingType::kImpossible, nullptr};
  }

  auto src0 = insn->RegAt(0);
  std::optional<uint64_t> imm64_0 = GetImmValueIfPossible(src0);
  if (imm64_0.has_value()) {
    // Both operands are immediates. This insn can be folded into one Movq.
    if (insn->opcode() == kMachineOpAndqRegReg || insn->opcode() == kMachineOpAndlRegReg ||
        insn->opcode() == kMachineOpOrqRegReg || insn->opcode() == kMachineOpOrlRegReg ||
        insn->opcode() == kMachineOpXorqRegReg || insn->opcode() == kMachineOpXorlRegReg ||
        insn->opcode() == kMachineOpAddqRegReg || insn->opcode() == kMachineOpAddlRegReg ||
        insn->opcode() == kMachineOpSubqRegReg || insn->opcode() == kMachineOpSublRegReg ||
        insn->opcode() == kMachineOpShlqRegReg || insn->opcode() == kMachineOpShllRegReg ||
        insn->opcode() == kMachineOpShrqRegReg || insn->opcode() == kMachineOpShrlRegReg) {
      return {FoldingType::kInsertInsn,
              NewInsnFromTwoImmediatesOperation(insn, imm64_0.value(), imm64_1.value())};
    }
  }

  // MovqRegReg is the only instruction that can encode full 64-bit immediate.
  if (insn->opcode() == kMachineOpMovqRegReg) {
    return {FoldingType::kReplaceInsn,
            machine_ir_->NewInsn<MovqRegImm>(insn->RegAt(0), imm64_1.value())};
  }

  int64_t signed_imm = bit_cast<int64_t>(imm64_1.value());
  int32_t signed_imm32 = static_cast<int32_t>(signed_imm);
  if (!kIsInput64Bit) {
    // Use the lower half of the register as the immediate operand.
    return {FoldingType::kReplaceInsn, NewImmInsnFromRegInsn(insn, signed_imm32)};
  }

  // Except for MOVQ x86 doesn't allow to encode 64-bit immediates. That said,
  // we can encode 32-bit immediates that are sign-extended by hardware to
  // 64-bit during instruction execution.
  if (signed_imm == static_cast<int64_t>(signed_imm32)) {
    return {FoldingType::kReplaceInsn, NewImmInsnFromRegInsn(insn, signed_imm32)};
  }

  return {FoldingType::kImpossible, nullptr};
}

MachineInsn* InsnFolding::NewInsnFromTwoImmediatesOperation(const MachineInsn* insn,
                                                            uint64_t imm1,
                                                            uint64_t imm2) {
  switch (insn->opcode()) {
    case kMachineOpShllRegImm:
    case kMachineOpShllRegReg:
      // In 32 bit shift operations, count operand is masked to size 5 bits.
      return machine_ir_->NewInsn<MovlRegImm>(insn->RegAt(0),
                                              static_cast<uint32_t>(imm1 << (imm2 % 32)));
    case kMachineOpShlqRegImm:
    case kMachineOpShlqRegReg:
      // In 64 bit shift operations, count operand is masked to size 6 bits.
      return machine_ir_->NewInsn<MovqRegImm>(insn->RegAt(0), imm1 << (imm2 % 64));
    case kMachineOpShrlRegImm:
    case kMachineOpShrlRegReg:
      // In 32 bit shift operations, count operand is masked to size 5 bits.
      return machine_ir_->NewInsn<MovlRegImm>(insn->RegAt(0),
                                              static_cast<uint32_t>(imm1 >> (imm2 % 32)));
    case kMachineOpShrqRegImm:
    case kMachineOpShrqRegReg:
      // In 64 bit shift operations, count operand is masked to size 6 bits.
      return machine_ir_->NewInsn<MovqRegImm>(insn->RegAt(0), imm1 >> (imm2 % 64));
    case kMachineOpAndlRegImm:
    case kMachineOpAndlRegReg:
      return machine_ir_->NewInsn<MovlRegImm>(insn->RegAt(0), static_cast<uint32_t>(imm1 & imm2));
    case kMachineOpAndqRegImm:
    case kMachineOpAndqRegReg:
      return machine_ir_->NewInsn<MovqRegImm>(insn->RegAt(0), imm1 & imm2);
    case kMachineOpOrlRegImm:
    case kMachineOpOrlRegReg:
      return machine_ir_->NewInsn<MovlRegImm>(insn->RegAt(0), static_cast<uint32_t>(imm1 | imm2));
    case kMachineOpOrqRegImm:
    case kMachineOpOrqRegReg:
      return machine_ir_->NewInsn<MovqRegImm>(insn->RegAt(0), imm1 | imm2);
    case kMachineOpXorlRegImm:
    case kMachineOpXorlRegReg:
      return machine_ir_->NewInsn<MovlRegImm>(insn->RegAt(0), static_cast<uint32_t>(imm1 ^ imm2));
    case kMachineOpXorqRegImm:
    case kMachineOpXorqRegReg:
      return machine_ir_->NewInsn<MovqRegImm>(insn->RegAt(0), imm1 ^ imm2);
    case kMachineOpAddlRegImm:
    case kMachineOpAddlRegReg:
      return machine_ir_->NewInsn<MovlRegImm>(insn->RegAt(0), static_cast<uint32_t>(imm1 + imm2));
    case kMachineOpAddqRegImm:
    case kMachineOpAddqRegReg:
      return machine_ir_->NewInsn<MovqRegImm>(insn->RegAt(0), imm1 + imm2);
    case kMachineOpSublRegImm:
    case kMachineOpSublRegReg:
      return machine_ir_->NewInsn<MovlRegImm>(insn->RegAt(0), static_cast<uint32_t>(imm1 - imm2));
    case kMachineOpSubqRegImm:
    case kMachineOpSubqRegReg:
      return machine_ir_->NewInsn<MovqRegImm>(insn->RegAt(0), imm1 - imm2);
    default:
      LOG_ALWAYS_FATAL("unexpected opcode");
      return nullptr;
  }
}

std::tuple<FoldingType, MachineInsn*> InsnFolding::TryFoldTwoImmediates(
    MachineInsnList::iterator insn_it) {
  const MachineInsn* insn = *insn_it;
  CHECK_GE(insn->NumRegOperands(), 2);
  MachineReg imm1_reg = insn->RegAt(0);
  std::optional<uint64_t> imm1 = GetImmValueIfPossible(imm1_reg);
  if (!imm1.has_value()) {
    return {FoldingType::kImpossible, nullptr};
  }
  uint64_t imm2 = AsMachineInsnX86_64(insn)->imm();
  // Check no value loss when imm2 is represented using 32 bits.
  CHECK(imm2 == static_cast<uint64_t>(static_cast<int32_t>(imm2)));
  // Rest of IR may use the value of flags set by current insn. Therefore, we don't remove
  // current insn, rather simply insert the folded insn. The dead code eliminator will
  // remove the current insn if possible.
  return {FoldingType::kInsertInsn, NewInsnFromTwoImmediatesOperation(insn, imm1.value(), imm2)};
}

std::tuple<FoldingType, MachineInsn*> InsnFolding::TryFoldRedundantMovl(
    MachineInsnList::iterator insn_it) {
  const MachineInsn* insn = *insn_it;
  CHECK_EQ(insn->opcode(), kMachineOpMovlRegReg);
  auto src = insn->RegAt(1);
  auto [def_insn_it, _] = FindNonPseudoCopyDef(src);
  if (!def_insn_it.has_value()) {
    return {FoldingType::kImpossible, nullptr};
  }
  const MachineInsn* def_insn = *def_insn_it.value();

  // If the definition of src clears its upper half, then we can replace MOVL with PseudoCopy.
  switch (def_insn->opcode()) {
    case kMachineOpMovlRegReg:
    case kMachineOpMovlRegMemAbsolute:
    case kMachineOpMovlRegMemBaseDisp:
    case kMachineOpMovlRegMemIndexDisp:
    case kMachineOpMovlRegMemBaseIndexDisp:
    case kMachineOpAndlRegReg:
    case kMachineOpXorlRegReg:
    case kMachineOpOrlRegReg:
    case kMachineOpSublRegReg:
    case kMachineOpAddlRegReg:
    case kMachineOpShrdlRegRegImm:
      return {FoldingType::kReplaceInsn, machine_ir_->NewInsn<PseudoCopy>(insn->RegAt(0), src, 4)};
    default:
      return {FoldingType::kImpossible, nullptr};
  }
}

template <bool kBMI, bool kIsInput64Bit>
std::tuple<FoldingType, MachineInsn*> InsnFolding::TryFoldCountLeadingZeros(
    MachineInsnList::iterator insn_it,
    const MachineBasicBlock* bb) {
  const MachineInsn* insn = *insn_it;
  const MachineOpcode clz_insn_opcode =
      kBMI            ? kIsInput64Bit ? kMachineOpLzcntqRegReg : kMachineOpLzcntlRegReg
      : kIsInput64Bit ? kMachineOpCountLeadingZerosU64
                      : kMachineOpCountLeadingZerosU32;
  CHECK_EQ(insn->opcode(), clz_insn_opcode);
  MachineReg clz_src_reg = insn->RegAt(1);
  auto [def_insn_it, def_insn_pos] = FindNonPseudoCopyDef(clz_src_reg);
  if (!def_insn_it.has_value()) {
    return {FoldingType::kImpossible, nullptr};
  }
  if (def_insn_it == bb->insn_list().begin()) {
    return {FoldingType::kImpossible, nullptr};
  }
  const MachineInsn* def_insn = *def_insn_it.value();
  const MachineOpcode reverse_bits_insn_opcode =
      kIsInput64Bit ? kMachineOpReverseBitsU64 : kMachineOpReverseBitsU32;
  if (def_insn->opcode() != reverse_bits_insn_opcode) {
    return {FoldingType::kImpossible, nullptr};
  }
  const MachineInsn* reverse_bits_insn = def_insn;
  MachineInsnList::iterator insn_before_reverse_bits_it = std::prev(def_insn_it.value());
  const MachineInsn* insn_before_reverse_bits = *insn_before_reverse_bits_it;
  if (insn_before_reverse_bits->opcode() != kMachineOpPseudoCopy) {
    return {FoldingType::kImpossible, nullptr};
  }
  const MachineInsn* pseudo_copy = insn_before_reverse_bits;
  if (pseudo_copy->RegAt(0) != reverse_bits_insn->RegAt(1) ||
      pseudo_copy->RegAt(0) == pseudo_copy->RegAt(1)) {
    return {FoldingType::kImpossible, nullptr};
  }
  // If ReverseBits insn or any insn after overwrites pseudo_copy->RegAt(1), this will return
  // std::nullopt.
  if (std::get<0>(def_map_.Get(pseudo_copy->RegAt(1), def_insn_pos)) == std::nullopt) {
    return {FoldingType::kImpossible, nullptr};
  }
  MachineInsn* new_insn;
  if (kBMI) {
    if (kIsInput64Bit) {
      new_insn =
          machine_ir_->NewInsn<TzcntqRegReg>(insn->RegAt(0), pseudo_copy->RegAt(1), insn->RegAt(2));
    } else {
      new_insn =
          machine_ir_->NewInsn<TzcntlRegReg>(insn->RegAt(0), pseudo_copy->RegAt(1), insn->RegAt(2));
    }
  } else {
    if (kIsInput64Bit) {
      new_insn = machine_ir_->NewInsn<CountTrailingZerosU64>(
          insn->RegAt(0), pseudo_copy->RegAt(1), insn->RegAt(2));
    } else {
      new_insn = machine_ir_->NewInsn<CountTrailingZerosU32>(
          insn->RegAt(0), pseudo_copy->RegAt(1), insn->RegAt(2));
    }
  }
  return {FoldingType::kReplaceInsn, new_insn};
}

std::tuple<FoldingType, MachineInsn*> InsnFolding::TryFoldInsn(
    const MachineInsnList::iterator insn_it,
    const MachineBasicBlock* bb) {
  const MachineInsn* insn = *insn_it;
  switch (insn->opcode()) {
    case kMachineOpMovqMemBaseDispReg:
    case kMachineOpMovqRegReg:
    case kMachineOpAndqRegReg:
    case kMachineOpTestqRegReg:
    case kMachineOpXorqRegReg:
    case kMachineOpOrqRegReg:
    case kMachineOpSubqRegReg:
    case kMachineOpCmpqRegReg:
    case kMachineOpAddqRegReg:
    case kMachineOpShlqRegReg:
    case kMachineOpShrqRegReg:
      return TryFoldImmediateInput<true>(insn_it);
    case kMachineOpMovlRegReg: {
      auto [folding_type, folded_insn] = TryFoldImmediateInput<false>(insn_it);
      if (folding_type != FoldingType::kImpossible) {
        return {folding_type, folded_insn};
      }
      return TryFoldRedundantMovl(insn_it);
    }
    case kMachineOpMovlMemBaseDispReg:
    case kMachineOpAndlRegReg:
    case kMachineOpTestlRegReg:
    case kMachineOpXorlRegReg:
    case kMachineOpOrlRegReg:
    case kMachineOpSublRegReg:
    case kMachineOpCmplRegReg:
    case kMachineOpAddlRegReg:
    case kMachineOpShllRegReg:
    case kMachineOpShrlRegReg:
      return TryFoldImmediateInput<false>(insn_it);
    case kMachineOpPseudoWriteFlags: {
      if (IsWritingSameFlagsValue(insn_it)) {
        return {FoldingType::kRemoveInsn, nullptr};
      }
      break;
    }
    case kMachineOpShlqRegImm:
    case kMachineOpShrqRegImm:
    case kMachineOpAndqRegImm:
    case kMachineOpOrqRegImm:
    case kMachineOpXorqRegImm:
    case kMachineOpAddqRegImm:
    case kMachineOpSubqRegImm:
    case kMachineOpShllRegImm:
    case kMachineOpShrlRegImm:
    case kMachineOpAndlRegImm:
    case kMachineOpOrlRegImm:
    case kMachineOpXorlRegImm:
    case kMachineOpAddlRegImm:
    case kMachineOpSublRegImm:
      return TryFoldTwoImmediates(insn_it);
    case kMachineOpLzcntlRegReg:
      return TryFoldCountLeadingZeros<true, false>(insn_it, bb);
    case kMachineOpLzcntqRegReg:
      return TryFoldCountLeadingZeros<true, true>(insn_it, bb);
    case kMachineOpCountLeadingZerosU32:
      return TryFoldCountLeadingZeros<false, false>(insn_it, bb);
    case kMachineOpCountLeadingZerosU64:
      return TryFoldCountLeadingZeros<false, true>(insn_it, bb);
    default:
      return {FoldingType::kImpossible, nullptr};
  }
  return {FoldingType::kImpossible, nullptr};
}

void FoldInsns(MachineIR* machine_ir) {
  DefMap def_map(machine_ir->NumVReg(), machine_ir->arena());
  for (auto* bb : machine_ir->bb_list()) {
    def_map.Initialize();
    InsnFolding insn_folding(def_map, machine_ir);
    MachineInsnList& insn_list = bb->insn_list();
    for (auto insn_it = insn_list.begin(); insn_it != insn_list.end();) {
      auto [folding_type, new_insn] = insn_folding.TryFoldInsn(insn_it, bb);
      if (folding_type == FoldingType::kRemoveInsn) {
        insn_it = insn_list.erase(insn_it);
        continue;
      }

      if (folding_type == FoldingType::kReplaceInsn) {
        CHECK(new_insn);
        *insn_it = new_insn;
      } else if (folding_type == FoldingType::kInsertInsn) {
        CHECK(new_insn);
        insn_list.insert(std::next(insn_it), new_insn);
      } else {
        CHECK(folding_type == FoldingType::kImpossible);
      }
      def_map.ProcessInsn(insn_it);
      ++insn_it;
    }
  }
}

// TODO(b/179708579): Maybe combine with FoldInsns.
void FoldWriteFlags(MachineIR* machine_ir) {
  for (auto* bb : machine_ir->bb_list()) {
    CHECK(!bb->insn_list().empty());
    auto insn_it = std::prev(bb->insn_list().end());
    if ((*insn_it)->opcode() != kMachineOpPseudoCondBranch) {
      continue;
    }

    auto* branch = static_cast<PseudoCondBranch*>(*insn_it);
    const auto* write_flags = *(--insn_it);
    if (write_flags->opcode() != kMachineOpPseudoWriteFlags) {
      continue;
    }
    // There is only one flags register, so CondBranch must read flags from WriteFlags.
    MachineReg flags = write_flags->RegAt(1);
    CHECK_EQ(flags.reg(), branch->RegAt(0).reg());

    const auto& live_out = bb->live_out();
    if (Contains(live_out, flags)) {
      // Flags are living-out. Cannot remove.
      // TODO(b/179708579): This shouldn't happen. Consider conversion to an assert.
      continue;
    }

    using Cond = CodeEmitter::Condition;
    Cond new_cond = Cond::kInvalidCondition;
    PseudoWriteFlags::Flags flags_mask;

    switch (branch->cond()) {
      // Verify that the flags are within the bottom 16 bits, so we can use Testw.
      static_assert(sizeof(PseudoWriteFlags::Flags) == 2);
      case Cond::kZero:
        new_cond = Cond::kNotZero;
        flags_mask = PseudoWriteFlags::Flags::kZero;
        break;
      case Cond::kNotZero:
        new_cond = Cond::kZero;
        flags_mask = PseudoWriteFlags::Flags::kZero;
        break;
      case Cond::kCarry:
        new_cond = Cond::kNotZero;
        flags_mask = PseudoWriteFlags::Flags::kCarry;
        break;
      case Cond::kNotCarry:
        new_cond = Cond::kZero;
        flags_mask = PseudoWriteFlags::Flags::kCarry;
        break;
      case Cond::kNegative:
        new_cond = Cond::kNotZero;
        flags_mask = PseudoWriteFlags::Flags::kNegative;
        break;
      case Cond::kNotSign:
        new_cond = Cond::kZero;
        flags_mask = PseudoWriteFlags::Flags::kNegative;
        break;
      case Cond::kOverflow:
        new_cond = Cond::kNotZero;
        flags_mask = PseudoWriteFlags::Flags::kOverflow;
        break;
      case Cond::kNoOverflow:
        new_cond = Cond::kZero;
        flags_mask = PseudoWriteFlags::Flags::kOverflow;
        break;
      default:
        continue;
    }

    MachineReg flags_src = write_flags->RegAt(0);
    MachineInsn* new_write_flags =
        machine_ir->NewInsn<x86_64::TestwRegImm>(flags_src, flags_mask, flags);
    insn_it = bb->insn_list().erase(insn_it);
    bb->insn_list().insert(insn_it, new_write_flags);
    branch->set_cond(new_cond);
  }
}

}  // namespace berberis::x86_64
