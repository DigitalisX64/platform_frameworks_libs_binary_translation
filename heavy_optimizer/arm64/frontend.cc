/*
 * Copyright (C) 2026 utzcoz
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
#include <cstdint>

#include "berberis/assembler/x86_64.h"
#include "berberis/backend/common/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

using Register = HeavyOptimizerFrontend::Register;

int32_t HeavyOptimizerFrontend::GetThreadStateRegOffset(uint8_t reg) {
  return static_cast<int32_t>(offsetof(ThreadState, cpu.x[0]) + reg * sizeof(uint64_t));
}

int32_t HeavyOptimizerFrontend::GetThreadStateSpOffset() {
  return static_cast<int32_t>(offsetof(ThreadState, cpu.sp));
}

void HeavyOptimizerFrontend::GenJump(GuestAddr target) {
  auto map_it = branch_targets_.find(target);
  if (map_it == branch_targets_.end()) {
    // Remember that this address was taken to help region formation. If we
    // translate it later the data will be overwritten with the actual location.
    branch_targets_[target] = MachineInsnPosition{};
  }

  // Checking pending signals only on back jumps guarantees no infinite loops
  // without pending-signal checks.
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

// After a faulting host memory access, split off a recovery basic block that
// exits the region so the guest signal handler runs. This mechanism is
// guest-agnostic and is copied verbatim from heavy_optimizer/riscv64.
void HeavyOptimizerFrontend::GenRecoveryBlockForLastInsn() {
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
// Branches.
//

// B (unconditional). SemanticsPlayer has already written X30 for the BL form.
void HeavyOptimizerFrontend::Branch(int32_t offset) {
  if (!success()) {
    return;
  }
  is_uncond_branch_ = true;
  GenJump(GetInsnAddr() + offset);
}

// BR / RET / BLR (indirect). SemanticsPlayer has already written X30 for BLR.
// Mirrors lite_translator.h::BranchRegister, which does NOT mask the top byte
// (no TBI): it simply exits indirect to `target`.
void HeavyOptimizerFrontend::BranchRegister(Register target) {
  if (!success()) {
    return;
  }
  is_uncond_branch_ = true;
  ExitRegionIndirect(target);
}

// Materialize a 0/1 predicate for ARM64 condition `cond` from the NZCV bits in
// ThreadState.cpu.flags. Bit positions and boolean algebra mirror
// lite_translator.h::EmitJumpIfCondNotMet exactly:
//   N = bit 15, Z = bit 14, C = bit 8, V = bit 0.
// The predicate is 1 iff the condition is satisfied. We extract each needed bit
// to its low position with a shift+and and combine with and/or/xor; "not" is
// xor with 1. kAl/kNv are unconditional and are handled by the caller before
// reaching here.
Register HeavyOptimizerFrontend::EmitArmCondPredicate(Decoder::Condition cond) {
  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));
  // Load the 16-bit NZCV word from ThreadState.cpu.flags. We use the 16-bit
  // MovwRegOp form (not MovzxwlRegOp) on purpose: RemoveLoopGuestContextAccesses
  // only recognizes MovwRegMemBaseDisp as a guest-context read of a 16-bit
  // field, so for an in-region loop the flags read must use this opcode to stay
  // consistent with the MovwOpReg write EmitMaterializeNZCV emits (otherwise the
  // optimizer caches the flag write in a register and the read keeps loading a
  // stale memory value, wedging the loop). MovwRegOp leaves the upper bits
  // untouched, so mask to the low 16 to get a clean zero-extended value.
  Register flags =
      std::get<0>(Gen<x86_64::MovwRegOp>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}));
  flags = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(flags, int32_t{0xFFFF}));

  // bit_to_low(pos): (flags >> pos) & 1, as a fresh 0/1 register.
  auto bit_to_low = [&](int8_t pos) -> Register {
    Register r = std::get<0>(Gen<x86_64::MovlRegReg>(flags));
    if (pos != 0) {
      r = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(r, pos));
    }
    return std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(r, int32_t{1}));
  };

  switch (cond) {
    case Decoder::Condition::kEq:  // Z==1
      return bit_to_low(14);
    case Decoder::Condition::kNe:  // Z==0
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(14), int32_t{1}));
    case Decoder::Condition::kCs:  // C==1
      return bit_to_low(8);
    case Decoder::Condition::kCc:  // C==0
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(8), int32_t{1}));
    case Decoder::Condition::kMi:  // N==1
      return bit_to_low(15);
    case Decoder::Condition::kPl:  // N==0
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(15), int32_t{1}));
    case Decoder::Condition::kVs:  // V==1
      return bit_to_low(0);
    case Decoder::Condition::kVc:  // V==0
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(0), int32_t{1}));
    case Decoder::Condition::kHi: {  // C==1 && Z==0
      Register c = bit_to_low(8);
      Register not_z = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(14), int32_t{1}));
      return std::get<0>(Gen<x86_64::AndlRegReg, kNoSSA>(c, not_z));
    }
    case Decoder::Condition::kLs: {  // C==0 || Z==1
      Register not_c = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(8), int32_t{1}));
      Register z = bit_to_low(14);
      return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(not_c, z));
    }
    case Decoder::Condition::kGe: {  // N==V  -> !(N^V)
      Register n = bit_to_low(15);
      Register v = bit_to_low(0);
      Register n_xor_v = std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(n, v));
      return std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(n_xor_v, int32_t{1}));
    }
    case Decoder::Condition::kLt: {  // N!=V  -> N^V
      Register n = bit_to_low(15);
      Register v = bit_to_low(0);
      return std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(n, v));
    }
    case Decoder::Condition::kGt: {  // Z==0 && N==V
      Register not_z = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(bit_to_low(14), int32_t{1}));
      Register n = bit_to_low(15);
      Register v = bit_to_low(0);
      Register n_xor_v = std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(n, v));
      Register n_eq_v = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(n_xor_v, int32_t{1}));
      return std::get<0>(Gen<x86_64::AndlRegReg, kNoSSA>(not_z, n_eq_v));
    }
    case Decoder::Condition::kLe: {  // Z==1 || N!=V
      Register z = bit_to_low(14);
      Register n = bit_to_low(15);
      Register v = bit_to_low(0);
      Register n_xor_v = std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(n, v));
      return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(z, n_xor_v));
    }
    case Decoder::Condition::kAl:
    case Decoder::Condition::kNv:
      // Unconditional: handled by the caller; never reached.
      CHECK(false);
      return flags;
  }
  CHECK(false);
  return flags;
}

// Branch to then_bb when `cond` is met, else_bb otherwise. The predicate is a
// 0/1 value; TestlRegReg sets ZF=1 when it is 0 (not taken) and ZF=0 when it is
// 1 (taken), so the then_bb is selected on kNotZero.
void HeavyOptimizerFrontend::EmitCondBranch(Decoder::Condition cond,
                                            MachineBasicBlock* then_bb,
                                            MachineBasicBlock* else_bb) {
  Register pred = EmitArmCondPredicate(cond);
  builder_.Gen<PseudoCondBranch>(x86_64::Assembler::Condition::kNotZero,
                                 then_bb,
                                 else_bb,
                                 std::get<0>(Gen<x86_64::TestlRegReg>(pred, pred)));
}

// B.cond (conditional). AL/NV are unconditional. Otherwise split into a taken
// then_bb (GenJump to the target) and a fall-through else_bb in which
// translation continues.
void HeavyOptimizerFrontend::BranchCond(Decoder::Condition cond, int32_t offset) {
  if (!success()) {
    return;
  }
  GuestAddr target = GetInsnAddr() + offset;

  // AL/NV always branch: lower as an unconditional B.
  if (cond == Decoder::Condition::kAl || cond == Decoder::Condition::kNv) {
    is_uncond_branch_ = true;
    GenJump(target);
    return;
  }

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* else_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, else_bb);

  EmitCondBranch(cond, then_bb, else_bb);

  builder_.StartBasicBlock(then_bb);
  GenJump(target);

  // Continue translating the not-taken path. A backward target is handled as an
  // in-region back-edge by GenJump+ResolveJumps (with the pending-signal
  // check), so do NOT set is_uncond_branch_/region-end here.
  builder_.StartBasicBlock(else_bb);
}

// CBZ (is_nonzero=false) / CBNZ (is_nonzero=true). Test the source for zero and
// branch like B.cond, mirroring lite_translator.h::CompareAndBranch.
void HeavyOptimizerFrontend::CompareAndBranch(bool is_nonzero,
                                              bool is_64bit,
                                              Register src,
                                              int32_t offset) {
  if (!success()) {
    return;
  }
  GuestAddr target = GetInsnAddr() + offset;

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* else_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, else_bb);

  // TEST sets ZF=1 when src is zero. CBNZ branches when nonzero (ZF==0 ->
  // kNotZero); CBZ branches when zero (ZF==1 -> kZero).
  Register flags = is_64bit ? std::get<0>(Gen<x86_64::TestqRegReg>(src, src))
                            : std::get<0>(Gen<x86_64::TestlRegReg>(src, src));
  builder_.Gen<PseudoCondBranch>(
      is_nonzero ? x86_64::Assembler::Condition::kNotZero : x86_64::Assembler::Condition::kZero,
      then_bb,
      else_bb,
      flags);

  builder_.StartBasicBlock(then_bb);
  GenJump(target);

  builder_.StartBasicBlock(else_bb);
}

// TBZ (is_nonzero=false) / TBNZ (is_nonzero=true). BT of bit `bit` of src sets
// CF; branch like B.cond, mirroring lite_translator.h::TestAndBranch. Bt is a
// 64-bit-register op, so it covers bits 0..63 directly.
void HeavyOptimizerFrontend::TestAndBranch(bool is_nonzero,
                                           Register src,
                                           uint8_t bit,
                                           int32_t offset) {
  if (!success()) {
    return;
  }
  GuestAddr target = GetInsnAddr() + offset;

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* else_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, else_bb);

  // BTQ src, bit -> CF = bit of src. TBNZ branches when the bit is set
  // (CF==1 -> kCarry); TBZ branches when clear (CF==0 -> kNotCarry).
  Register flags = std::get<0>(Gen<x86_64::BtqRegImm>(src, static_cast<int8_t>(bit)));
  builder_.Gen<PseudoCondBranch>(
      is_nonzero ? x86_64::Assembler::Condition::kCarry : x86_64::Assembler::Condition::kNotCarry,
      then_bb,
      else_bb,
      flags);

  builder_.StartBasicBlock(then_bb);
  GenJump(target);

  builder_.StartBasicBlock(else_bb);
}

// CSEL/CSINC/CSINV/CSNEG. Materialize the false case (transform(src2)) into a
// result vreg, then conditionally overwrite it with src1 when `cond` holds.
// Mirrors lite_translator.h::ConditionalSelect. AL/NV always select src1.
Register HeavyOptimizerFrontend::ConditionalSelect(Decoder::ConditionalSelectOpcode opcode,
                                                   bool is_64bit,
                                                   Register src1,
                                                   Register src2,
                                                   Decoder::Condition cond) {
  if (!success()) {
    return AllocTempReg();
  }

  // The false case = transform(src2), and the true case = src1, each produced
  // into a fresh 64-bit-wide value (a 32-bit op clears the upper half, matching
  // ARM64 W-write semantics). They are funneled into a single `result` vreg via
  // PseudoCopy so that, after the conditional overwrite, `result` holds the
  // selected operand on every control-flow edge.
  auto width_adjust = [&](Register r) -> Register {
    if (is_64bit) {
      return Copy(r);
    }
    return std::get<0>(Gen<x86_64::MovlRegReg>(r));
  };

  Register false_val;
  switch (opcode) {
    case Decoder::ConditionalSelectOpcode::kCsel:
      false_val = width_adjust(src2);
      break;
    case Decoder::ConditionalSelectOpcode::kCsinc:
      false_val = width_adjust(src2);
      if (is_64bit) {
        false_val = std::get<0>(Gen<x86_64::AddqRegImm, kNoSSA>(false_val, int32_t{1}));
      } else {
        false_val = std::get<0>(Gen<x86_64::AddlRegImm, kNoSSA>(false_val, int32_t{1}));
      }
      break;
    case Decoder::ConditionalSelectOpcode::kCsinv:
      // ~src2. The heavy IR has only a 64-bit NOT; the 32-bit case re-clears the
      // upper half with a 32-bit mov afterwards.
      false_val = Copy(src2);
      false_val = std::get<0>(Gen<x86_64::NotqReg, kNoSSA>(false_val));
      if (!is_64bit) {
        false_val = std::get<0>(Gen<x86_64::MovlRegReg>(false_val));
      }
      break;
    case Decoder::ConditionalSelectOpcode::kCsneg: {
      // -src2 = 0 - src2 (the heavy IR has no Neg op). The l-suffix subtract
      // zero-extends the 32-bit result.
      Register zero = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
      if (is_64bit) {
        false_val = std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(zero, src2));
      } else {
        false_val = std::get<0>(Gen<x86_64::SublRegReg, kNoSSA>(zero, src2));
      }
      break;
    }
  }

  Register result = AllocTempReg();
  builder_.Gen<PseudoCopy>(result, false_val, 8);

  // AL/NV: always select src1 (unconditional). No branch needed.
  if (cond == Decoder::Condition::kAl || cond == Decoder::Condition::kNv) {
    builder_.Gen<PseudoCopy>(result, width_adjust(src1), 8);
    return result;
  }

  // Conditionally overwrite result with src1 when the condition holds. then_bb
  // does the overwrite; both paths fall into merge_bb. result is the same vreg
  // written on both edges, so its value after merge_bb is the selected operand.
  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, merge_bb);

  EmitCondBranch(cond, then_bb, merge_bb);

  builder_.StartBasicBlock(then_bb);
  builder_.Gen<PseudoCopy>(result, width_adjust(src1), 8);
  ir->AddEdge(then_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
  return result;
}

// CCMP/CCMN. If `cond` holds, set NZCV from a real CMP (is_neg=false) / CMN
// (is_neg=true); otherwise set NZCV from the 4-bit nzcv immediate. Mirrors
// lite_translator.cc::ConditionalCompare with then/else/merge basic blocks.
void HeavyOptimizerFrontend::ConditionalCompare(bool is_neg,
                                                bool is_64bit,
                                                Register rn,
                                                Register rm,
                                                Decoder::Condition cond,
                                                uint8_t nzcv) {
  if (!success()) {
    return;
  }

  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));

  // Pack the immediate-path NZCV: ARM bit3=N,bit2=Z,bit1=C,bit0=V map to
  // cpu.flags N@15, Z@14, C@8, V@0 (the same layout EmitMaterializeNZCV writes).
  auto emit_immediate_path = [&]() {
    uint16_t flags_val = 0;
    if (nzcv & 0x8) {
      flags_val |= (1 << 15);  // N
    }
    if (nzcv & 0x4) {
      flags_val |= (1 << 14);  // Z
    }
    if (nzcv & 0x2) {
      flags_val |= (1 << 8);  // C
    }
    if (nzcv & 0x1) {
      flags_val |= (1 << 0);  // V
    }
    Register imm_reg = GetImm(flags_val);
    builder_.Gen<x86_64::MovwOpReg>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}, imm_reg);
  };

  // The compare path: CMP is non-destructive (CmpqRegReg only defs FLAGS); CMN
  // has no non-destructive x86 add, so add rn+rm into a scratch (never into rn,
  // which is the live guest register under register mapping) and take its flags.
  auto emit_compare_path = [&]() {
    Register flags;
    if (is_64bit) {
      if (is_neg) {
        Register tmp = Copy(rn);
        flags = std::get<1>(Gen<x86_64::AddqRegReg, kNoSSA>(tmp, rm));
      } else {
        flags = std::get<0>(Gen<x86_64::CmpqRegReg>(rn, rm));
      }
    } else {
      if (is_neg) {
        Register tmp = std::get<0>(Gen<x86_64::MovlRegReg>(rn));
        flags = std::get<1>(Gen<x86_64::AddlRegReg, kNoSSA>(tmp, rm));
      } else {
        flags = std::get<0>(Gen<x86_64::CmplRegReg>(rn, rm));
      }
    }
    EmitMaterializeNZCV(flags, /*is_sub=*/!is_neg);
  };

  // AL/NV: always the compare path (no branch).
  if (cond == Decoder::Condition::kAl || cond == Decoder::Condition::kNv) {
    emit_compare_path();
    return;
  }

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* cmp_bb = ir->NewBasicBlock();   // condition met -> real compare
  MachineBasicBlock* imm_bb = ir->NewBasicBlock();   // condition not met -> nzcv imm
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, cmp_bb);
  ir->AddEdge(cur_bb, imm_bb);

  EmitCondBranch(cond, cmp_bb, imm_bb);

  builder_.StartBasicBlock(cmp_bb);
  emit_compare_path();
  ir->AddEdge(cmp_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(imm_bb);
  emit_immediate_path();
  ir->AddEdge(imm_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
}

void HeavyOptimizerFrontend::Undefined() {
  // Idempotent: a single guest instruction can trigger several listener calls
  // (e.g. a pre/post-index access calls AddImm then Load/Store). If more than one
  // bails, only the first may append the region-exit terminator — a second exit
  // in the same basic block would make it fail CheckMachineIR.
  if (!success_) {
    return;
  }
  success_ = false;
  ExitGeneratedCode(GetInsnAddr());
  // We don't require the region to end here as control flow may jump around the
  // undefined instruction, so handle it as an unconditional branch.
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
      // Recovery blocks must exit the region, do not try to resolve into a local branch.
      continue;
    }

    const MachineInsn* last_insn = bb->insn_list().back();
    if (last_insn->opcode() != kMachineOpPseudoJump) {
      continue;
    }

    auto* jump = static_cast<const PseudoJump*>(last_insn);
    if (jump->kind() == PseudoJump::Kind::kSyscall ||
        jump->kind() == PseudoJump::Kind::kExitGeneratedCode) {
      // Syscall or generated-code exit must always exit the region.
      continue;
    }

    GuestAddr target = jump->target();
    auto map_it = branch_targets_.find(target);
    // All PseudoJump insns must add their targets to branch_targets.
    CHECK(map_it != branch_targets_.end());

    MachineInsnPosition pos = map_it->second;
    MachineBasicBlock* target_containing_bb = pos.first;
    if (!target_containing_bb) {
      // Branch target is not in the current region.
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

      // Make sure target_bb is also considered for jump resolution. Otherwise we
      // may leave code referenced by it unlinked from the rest of the IR.
      bb_list_copy.push_back(target_bb);

      // If bb is equal to target_containing_bb, then the branch instruction at
      // the end of bb is moved to the new target_bb, so we replace the
      // instruction at the end of target_bb instead of bb.
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
    // This is a backward branch resolved into an in-region target, i.e. an
    // in-region loop back-edge. Record that the region captured a hot loop.
    has_in_region_backedge_ = true;
    // See EmitCheckSignalsAndMaybeReturn.
    auto* exit_bb = ir->NewBasicBlock();
    // Note that we intentionally don't mark exit_bb as recovery and therefore
    // don't request its reordering away from hot code spots. target_bb is a
    // back branch and is unlikely to be a fall-through jump for the current bb.
    // At the same time exit_bb can be a fall-through jump and benchmarks benefit
    // from it.
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

//
// Methods that are not part of the SemanticsListener implementation.
//
void HeavyOptimizerFrontend::StartInsn() {
  if (is_uncond_branch_) {
    auto* ir = builder_.ir();
    builder_.StartBasicBlock(ir->NewBasicBlock());
  }

  is_uncond_branch_ = false;
  // The iterators in branch_targets are the last iterators before generating an
  // insn. We advance iterators by one step in Finalize(), as we'll use it to
  // iterate the sub-list of instructions starting from the first one for the
  // given guest address.
  //
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

  // This loop advances the iterators in branch_targets by one. Because in
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

}  // namespace berberis
