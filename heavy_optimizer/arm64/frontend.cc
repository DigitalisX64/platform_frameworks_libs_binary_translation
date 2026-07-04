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

// UDIV Xd/Wd, Xn/Wn, Xm/Wm. ARM division never traps: if the divisor is 0 the
// result is 0. x86 DIV #DE-faults on a zero divisor, so guard it with a branch.
// Mirrors lite_translator.h::DataProc2Src kUdiv. The dividend goes into RAX and
// the high half (zero, unsigned) into RDX; the DivRegRegReg pseudo-op binds
// those fixed registers via the Gen<> SSA wrapper, so the divisor stays a free
// vreg. A 32-bit DIV writes EAX, which zero-extends to the X register.
Register HeavyOptimizerFrontend::EmitUDiv(bool is_64bit, Register src1, Register src2) {
  if (!success()) {
    return AllocTempReg();
  }

  Register result = AllocTempReg();
  // Divide-by-zero path writes 0; the divide path overwrites result with the
  // quotient. Both edges define `result`, so it holds the right value at merge.
  Register zero = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  builder_.Gen<PseudoCopy>(result, zero, 8);

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* div_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, div_bb);
  ir->AddEdge(cur_bb, merge_bb);

  // TEST sets ZF=1 when the divisor is zero -> skip the divide (result stays 0).
  Register flags = is_64bit ? std::get<0>(Gen<x86_64::TestqRegReg>(src2, src2))
                            : std::get<0>(Gen<x86_64::TestlRegReg>(src2, src2));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kZero, merge_bb, div_bb, flags);

  builder_.StartBasicBlock(div_bb);
  // High half of the dividend is 0 for unsigned division.
  Register hi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  Register quotient;
  if (is_64bit) {
    quotient = std::get<0>(Gen<x86_64::DivqRegRegReg>(src1, hi, src2));
  } else {
    quotient = std::get<0>(Gen<x86_64::DivlRegRegReg>(src1, hi, src2));
  }
  builder_.Gen<PseudoCopy>(result, quotient, 8);
  ir->AddEdge(div_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
  return result;
}

// SDIV Xd/Wd, Xn/Wn, Xm/Wm. ARM division never traps: Rm==0 -> 0, and the
// INT_MIN/-1 overflow case -> INT_MIN (x86 IDIV #DE-faults on both). Mirrors
// lite_translator.h::DataProc2Src kSdiv. Three blocks: zero divisor (result
// stays 0), divisor==-1 (result = -Rn, which is INT_MIN for INT_MIN input and
// correct for every other Rn), and the real IDIV (RDX = sign-extension of RAX).
Register HeavyOptimizerFrontend::EmitSDiv(bool is_64bit, Register src1, Register src2) {
  if (!success()) {
    return AllocTempReg();
  }

  Register result = AllocTempReg();
  Register zero = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  builder_.Gen<PseudoCopy>(result, zero, 8);

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* nonzero_bb = ir->NewBasicBlock();
  MachineBasicBlock* neg_one_bb = ir->NewBasicBlock();
  MachineBasicBlock* div_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();

  // Divisor == 0 -> result stays 0.
  ir->AddEdge(cur_bb, merge_bb);
  ir->AddEdge(cur_bb, nonzero_bb);
  Register zflags = is_64bit ? std::get<0>(Gen<x86_64::TestqRegReg>(src2, src2))
                             : std::get<0>(Gen<x86_64::TestlRegReg>(src2, src2));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kZero, merge_bb, nonzero_bb, zflags);

  // Divisor == -1 -> result = -Rn (= INT_MIN when Rn==INT_MIN, no IDIV).
  builder_.StartBasicBlock(nonzero_bb);
  ir->AddEdge(nonzero_bb, neg_one_bb);
  ir->AddEdge(nonzero_bb, div_bb);
  Register cflags = is_64bit
                        ? std::get<0>(Gen<x86_64::CmpqRegImm>(src2, int32_t{-1}))
                        : std::get<0>(Gen<x86_64::CmplRegImm>(src2, int32_t{-1}));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kEqual, neg_one_bb, div_bb, cflags);

  // result = 0 - Rn (the heavy IR has no NEG op; a 32-bit sub zero-extends).
  builder_.StartBasicBlock(neg_one_bb);
  Register negbase = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
  Register neg;
  if (is_64bit) {
    neg = std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(negbase, src1));
  } else {
    neg = std::get<0>(Gen<x86_64::SublRegReg, kNoSSA>(negbase, src1));
  }
  builder_.Gen<PseudoCopy>(result, neg, 8);
  ir->AddEdge(neg_one_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  // Real division: RDX = sign-extension of the dividend (CQO/CDQ equivalent).
  builder_.StartBasicBlock(div_bb);
  Register quotient;
  if (is_64bit) {
    Register hi = std::get<0>(Gen<x86_64::SarqRegImm>(Copy(src1), int8_t{63}));
    quotient = std::get<0>(Gen<x86_64::IdivqRegRegReg>(src1, hi, src2));
  } else {
    Register hi = std::get<0>(Gen<x86_64::SarlRegImm>(
        std::get<0>(Gen<x86_64::MovlRegReg>(src1)), int8_t{31}));
    quotient = std::get<0>(Gen<x86_64::IdivlRegRegReg>(src1, hi, src2));
  }
  builder_.Gen<PseudoCopy>(result, quotient, 8);
  ir->AddEdge(div_bb, merge_bb);
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

// Map the x86 EFLAGS a UCOMIS{S,D} left in `flags_vreg` to ARM64 FP NZCV and
// store the packed 16-bit word to ThreadState.cpu.flags. Bit-exact with
// lite_translator.h::EmitStoreArmFpNZCV, which branches on the x86 FLAGS
// directly; here the flags are first read into a GP register (PseudoReadFlags:
// LAHF + SETO) so the branch tree can test individual bits without keeping the
// single host FLAGS live across basic-block boundaries. In that GP word:
// CF@8, PF@10, ZF@14 (OF@0, unused here). Priority PF > ZF > CF matches lite:
//   PF set   -> unordered -> NZCV C,V   (0x0101)
//   ZF set   -> equal     -> NZCV Z,C   (0x4100)
//   CF set   -> less      -> NZCV N     (0x8000)
//   else     -> greater   -> NZCV C     (0x0100)
void HeavyOptimizerFrontend::EmitStoreArmFpNZCV(Register flags_vreg) {
  if (!success()) {
    return;
  }
  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));

  Register raw = AllocTempReg();
  builder_.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, raw, flags_vreg);

  auto* ir = builder_.ir();
  MachineBasicBlock* uo_bb = ir->NewBasicBlock();       // unordered (PF)
  MachineBasicBlock* chk_eq_bb = ir->NewBasicBlock();   // test ZF
  MachineBasicBlock* eq_bb = ir->NewBasicBlock();       // equal (ZF)
  MachineBasicBlock* chk_lt_bb = ir->NewBasicBlock();   // test CF
  MachineBasicBlock* lt_bb = ir->NewBasicBlock();       // less (CF)
  MachineBasicBlock* gt_bb = ir->NewBasicBlock();       // greater (default)
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();

  // Fill a leaf block: write the packed NZCV immediate and jump to merge.
  auto store_leaf = [&](uint16_t nzcv_word, MachineBasicBlock* bb) {
    builder_.StartBasicBlock(bb);
    Register imm = GetImm(nzcv_word);
    builder_.Gen<x86_64::MovwOpReg>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}, imm);
    ir->AddEdge(bb, merge_bb);
    builder_.Gen<PseudoBranch>(merge_bb);
  };

  // PF (bit 10) set -> unordered, else fall to the ZF test.
  auto* cur_bb = builder_.bb();
  ir->AddEdge(cur_bb, uo_bb);
  ir->AddEdge(cur_bb, chk_eq_bb);
  Register pf = std::get<0>(Gen<x86_64::TestlRegImm>(raw, int32_t{1 << 10}));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNotZero, uo_bb, chk_eq_bb, pf);

  // ZF (bit 14) set -> equal, else fall to the CF test.
  builder_.StartBasicBlock(chk_eq_bb);
  ir->AddEdge(chk_eq_bb, eq_bb);
  ir->AddEdge(chk_eq_bb, chk_lt_bb);
  Register zf = std::get<0>(Gen<x86_64::TestlRegImm>(raw, int32_t{1 << 14}));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNotZero, eq_bb, chk_lt_bb, zf);

  // CF (bit 8) set -> less, else greater.
  builder_.StartBasicBlock(chk_lt_bb);
  ir->AddEdge(chk_lt_bb, lt_bb);
  ir->AddEdge(chk_lt_bb, gt_bb);
  Register cf = std::get<0>(Gen<x86_64::TestlRegImm>(raw, int32_t{1 << 8}));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNotZero, lt_bb, gt_bb, cf);

  store_leaf(0x0101, uo_bb);  // unordered: C,V
  store_leaf(0x4100, eq_bb);  // equal:     Z,C
  store_leaf(0x8000, lt_bb);  // less:      N
  store_leaf(0x0100, gt_bb);  // greater:   C

  builder_.StartBasicBlock(merge_bb);
}

// FCMP/FCMPE Sn/Dn, Sm/Dm (or #0.0). Mirrors lite_translator.h::FpCompare:
// load lane 0 of the operands, UCOMIS{S,D}, then EmitStoreArmFpNZCV. S/D only;
// FP16 (ftype 0b11) and the reserved ftype 0b10 bail to the lite tier.
void HeavyOptimizerFrontend::FpCompare(const Decoder::FpCompareArgs& args) {
  if (!success()) {
    return;
  }
  if (args.ftype != 0b00 && args.ftype != 0b01) {
    Undefined();
    return;
  }
  const bool is_double = (args.ftype == 0b01);

  FpRegister xmm_n = GetVRegScalar(args.rn, is_double);
  FpRegister xmm_m = args.with_zero ? AllocZeroedSimdReg() : GetVRegScalar(args.rm, is_double);

  Register flags = is_double
                       ? std::get<0>(Gen<x86_64::UcomisdXRegXReg>(xmm_n.machine_reg(),
                                                                  xmm_m.machine_reg()))
                       : std::get<0>(Gen<x86_64::UcomissXRegXReg>(xmm_n.machine_reg(),
                                                                  xmm_m.machine_reg()));
  EmitStoreArmFpNZCV(flags);
}

// FCCMP/FCCMPE: if `cond` holds do the FCMP compare + NZCV mapping, else write
// the 4-bit nzcv immediate straight to cpu.flags. Same then/else/merge shape as
// ConditionalCompare; the compare path's EmitStoreArmFpNZCV creates its own
// sub-tree, so the edge into merge is taken from whatever block it leaves the
// builder in. S/D only; FP16 / reserved ftype bail.
void HeavyOptimizerFrontend::FpConditionalCompare(
    const Decoder::FpConditionalCompareArgs& args) {
  if (!success()) {
    return;
  }
  if (args.ftype != 0b00 && args.ftype != 0b01) {
    Undefined();
    return;
  }
  const bool is_double = (args.ftype == 0b01);
  const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));

  // Pack the false-path NZCV immediate: bit3=N,bit2=Z,bit1=C,bit0=V map to
  // cpu.flags N@15, Z@14, C@8, V@0 (same layout EmitMaterializeNZCV writes).
  auto emit_immediate_path = [&]() {
    uint16_t flags_val = 0;
    if (args.nzcv & 0x8) {
      flags_val |= (1 << 15);  // N
    }
    if (args.nzcv & 0x4) {
      flags_val |= (1 << 14);  // Z
    }
    if (args.nzcv & 0x2) {
      flags_val |= (1 << 8);  // C
    }
    if (args.nzcv & 0x1) {
      flags_val |= (1 << 0);  // V
    }
    Register imm = GetImm(flags_val);
    builder_.Gen<x86_64::MovwOpReg>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}, imm);
  };

  auto emit_compare_path = [&]() {
    FpRegister xmm_n = GetVRegScalar(args.rn, is_double);
    FpRegister xmm_m = GetVRegScalar(args.rm, is_double);
    Register flags = is_double
                         ? std::get<0>(Gen<x86_64::UcomisdXRegXReg>(xmm_n.machine_reg(),
                                                                    xmm_m.machine_reg()))
                         : std::get<0>(Gen<x86_64::UcomissXRegXReg>(xmm_n.machine_reg(),
                                                                    xmm_m.machine_reg()));
    EmitStoreArmFpNZCV(flags);
  };

  // AL/NV: always the compare path (no predicate branch). The compare path's
  // NZCV sub-tree leaves the builder at its merge; translation continues there.
  if (args.cond == Decoder::Condition::kAl || args.cond == Decoder::Condition::kNv) {
    emit_compare_path();
    return;
  }

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* cmp_bb = ir->NewBasicBlock();
  MachineBasicBlock* imm_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, cmp_bb);
  ir->AddEdge(cur_bb, imm_bb);

  // Condition is read from the pre-compare cpu.flags (EmitCondBranch loads them
  // in cur_bb, before the compare path overwrites them).
  EmitCondBranch(args.cond, cmp_bb, imm_bb);

  builder_.StartBasicBlock(cmp_bb);
  emit_compare_path();
  // emit_compare_path built an NZCV sub-tree; connect its tail to merge.
  MachineBasicBlock* cmp_tail = builder_.bb();
  ir->AddEdge(cmp_tail, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(imm_bb);
  emit_immediate_path();
  ir->AddEdge(imm_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
}

// SCVTF (op 010) / UCVTF (op 011), unscaled (rmode == 00): convert a general
// register integer to scalar FP via x86 CVTSI2SS/SD, which reads a SIGNED
// source. Mirrors lite_translator.h::FpIntConversion's SCVTF/UCVTF path.
//   * SCVTF: the L-form sign-extends the 32-bit source (sf=0), the Q-form takes
//     the 64-bit source (sf=1) — both match ARM's signed convert directly.
//   * UCVTF sf=0 (32-bit unsigned): zero-extend with a 32-bit MOV (auto-clears
//     the upper 32) then Q-convert (value <= UINT32_MAX < INT64_MAX, exact).
//   * UCVTF sf=1 (64-bit unsigned): values < 2^63 Q-convert directly; values
//     >= 2^63 branch to the round-to-odd halve/convert/double fix-up so the
//     round-to-nearest-even result matches static_cast<float|double>(uint64_t)
//     bit-for-bit. Two paths write a shared merge XMM.
// The scalar result is committed with SetVRegScalar (upper V[] bytes zeroed).
void HeavyOptimizerFrontend::EmitScvtfUcvtf(const Decoder::FpIntConvArgs& args,
                                            bool is_double) {
  if (!success()) {
    return;
  }
  const bool is_unsigned = (args.op == 0b011);

  // rn == 31 is WZR/XZR -> 0; static_cast<FP>(0) == +0.0.
  if (args.rn == 31) {
    SetVRegScalar(args.rd, AllocZeroedSimdReg(), is_double);
    return;
  }

  Register src = GetReg(args.rn);

  // SCVTF (any sf): straight-line signed convert.
  if (!is_unsigned) {
    FpRegister xmm = AllocTempSimdReg();
    if (args.sf) {
      if (is_double) {
        builder_.Gen<x86_64::Cvtsi2sdqXRegReg>(xmm.machine_reg(), src);
      } else {
        builder_.Gen<x86_64::Cvtsi2ssqXRegReg>(xmm.machine_reg(), src);
      }
    } else {
      if (is_double) {
        builder_.Gen<x86_64::Cvtsi2sdlXRegReg>(xmm.machine_reg(), src);
      } else {
        builder_.Gen<x86_64::Cvtsi2sslXRegReg>(xmm.machine_reg(), src);
      }
    }
    SetVRegScalar(args.rd, xmm, is_double);
    return;
  }

  // UCVTF sf=0: zero-extend the 32-bit source, convert as signed 64-bit.
  if (!args.sf) {
    Register zx = std::get<0>(Gen<x86_64::MovlRegReg>(src));
    FpRegister xmm = AllocTempSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::Cvtsi2sdqXRegReg>(xmm.machine_reg(), zx);
    } else {
      builder_.Gen<x86_64::Cvtsi2ssqXRegReg>(xmm.machine_reg(), zx);
    }
    SetVRegScalar(args.rd, xmm, is_double);
    return;
  }

  // UCVTF sf=1: 64-bit unsigned. Branch on the sign bit (bit 63). Both paths
  // write the shared merge XMM `result`, defined on every edge into merge_bb.
  FpRegister result = AllocTempSimdReg();

  auto* ir = builder_.ir();
  auto* cur_bb = builder_.bb();
  MachineBasicBlock* direct_bb = ir->NewBasicBlock();
  MachineBasicBlock* fixup_bb = ir->NewBasicBlock();
  MachineBasicBlock* merge_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, direct_bb);
  ir->AddEdge(cur_bb, fixup_bb);

  // TEST sets SF = bit 63; take the fix-up path when the source is >= 2^63.
  Register flags = std::get<0>(Gen<x86_64::TestqRegReg>(src, src));
  builder_.Gen<PseudoCondBranch>(
      x86_64::Assembler::Condition::kNegative, fixup_bb, direct_bb, flags);

  // Direct: value < 2^63, signed Q-convert is exact.
  builder_.StartBasicBlock(direct_bb);
  {
    FpRegister xd = AllocTempSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::Cvtsi2sdqXRegReg>(xd.machine_reg(), src);
    } else {
      builder_.Gen<x86_64::Cvtsi2ssqXRegReg>(xd.machine_reg(), src);
    }
    builder_.Gen<x86_64::MovdqaXRegXReg>(result.machine_reg(), xd.machine_reg());
  }
  ir->AddEdge(direct_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  // Fix-up: value >= 2^63. odd = (src >> 1) | (src & 1); convert; double.
  builder_.StartBasicBlock(fixup_bb);
  {
    Register low_bit = std::get<0>(Gen<x86_64::AndqRegImm>(Copy(src), int32_t{1}));
    Register halved = std::get<0>(Gen<x86_64::ShrqRegImm>(Copy(src), int8_t{1}));
    Register odd = std::get<0>(Gen<x86_64::OrqRegReg>(halved, low_bit));
    FpRegister xf = AllocTempSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::Cvtsi2sdqXRegReg>(xf.machine_reg(), odd);
      builder_.Gen<x86_64::AddsdXRegXReg>(xf.machine_reg(), xf.machine_reg());
    } else {
      builder_.Gen<x86_64::Cvtsi2ssqXRegReg>(xf.machine_reg(), odd);
      builder_.Gen<x86_64::AddssXRegXReg>(xf.machine_reg(), xf.machine_reg());
    }
    builder_.Gen<x86_64::MovdqaXRegXReg>(result.machine_reg(), xf.machine_reg());
  }
  ir->AddEdge(fixup_bb, merge_bb);
  builder_.Gen<PseudoBranch>(merge_bb);

  builder_.StartBasicBlock(merge_bb);
  SetVRegScalar(args.rd, result, is_double);
}

// LDXR/STXR/LDAXR/STLXR (exclusive) and LDAR/STLR (acquire/release). Mirrors
// lite_translator.h::LoadStoreExclusive byte-for-byte:
//   * base is TBI-masked first (the top-byte-ignore tag is not part of the host
//     address), then used for the reservation address, the load/store, and the
//     CMPXCHG memory operand, so the heavy and lite/interp monitors agree on the
//     same guest under tagged pointers.
//   * LDAR/STLR: x86-TSO gives acquire/release for plain loads/stores, so they
//     are a plain sized Load()/Store() (both emit fault recovery internally).
//   * LDXR/LDAXR: Load() the value, then record cpu.reservation_address = base
//     and cpu.reservation_value = value (64-bit slot, matching the lite tier and
//     interpreter; only LDXP/STXP use the full 128-bit width).
//   * STXR/STLXR: if cpu.reservation_address still equals base, do a sized
//     LOCK CMPXCHG of the saved reservation_value (expected, in the accumulator)
//     against [base] with Rt as the new value; clear cpu.reservation_address;
//     Rs gets 0 on CMPXCHG success (ZF=1), 1 on CMPXCHG failure or address
//     mismatch. XZR (reg 31) reads as 0 / discards writes.
// This deliberately reuses the lite tier's reservation_address + reservation_value
// + plain CMPXCHG monitor model (NOT the riscv64 heavy tier's
// MemoryRegionReservation owner-tracking model), so the heavy, lite, and
// interpreter tiers share one monitor scheme for the same guest.
void HeavyOptimizerFrontend::LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& args,
                                                Register base) {
  if (!success()) {
    return;
  }

  // Apply the TBI mask before using base as a memory operand or reservation key.
  base = ApplyTbi(base);
  auto lss = static_cast<Decoder::LoadStoreSize>(args.size);
  const int32_t resv_addr_off = static_cast<int32_t>(offsetof(ThreadState, cpu.reservation_address));
  const int32_t resv_val_off = static_cast<int32_t>(offsetof(ThreadState, cpu.reservation_value));

  switch (args.op) {
    case Decoder::AtomicOp::kLdar: {
      // Load-acquire: x86-TSO provides acquire ordering for all loads.
      Register res = Load(lss, /*is_signed=*/false, /*is_64bit_target=*/true, base, 0);
      if (!success()) {
        return;
      }
      if (args.rt != 31) {
        SetReg(args.rt, res);
      }
      return;
    }

    case Decoder::AtomicOp::kStlr: {
      // Store-release: x86-TSO gives release ordering for free, but ARM STLR is
      // sequentially consistent (RCsc) and also orders the store before later
      // loads. x86 permits StoreLoad reordering, so emit MFENCE after the store.
      Register data = (args.rt != 31) ? GetReg(args.rt) : GetImm(0);
      if (!success()) {
        return;
      }
      Store(lss, base, 0, data);
      builder_.Gen<x86_64::Mfence>();
      return;
    }

    case Decoder::AtomicOp::kLdxr: {
      // Load-exclusive: load the value, then record the reservation.
      Register res = Load(lss, /*is_signed=*/false, /*is_64bit_target=*/true, base, 0);
      if (!success()) {
        return;
      }
      // reservation_address = base.
      builder_.Gen<x86_64::MovqOpReg>(
          {.base = x86_64::kMachineRegRBP, .disp = resv_addr_off}, base);
      // reservation_value = res (low 64 bits of the 128-bit slot, as the lite
      // tier does; the single-register forms only ever use 64 bits).
      builder_.Gen<x86_64::MovqOpReg>(
          {.base = x86_64::kMachineRegRBP, .disp = resv_val_off}, res);
      if (args.rt != 31) {
        SetReg(args.rt, res);
      }
      return;
    }

    case Decoder::AtomicOp::kStxr: {
      // Store-exclusive: compare-and-swap against the reservation. Structured to
      // mirror the riscv64 heavy frontend's MemoryRegionReservationExchange: the
      // status vreg (result) is written only in the two terminal predecessors of
      // the merge block (XOR -> 0 on success, MOVQ #1 on failure), never in the
      // entry block, so its live range is the simple two-def/one-use shape the
      // lifetime analyzer expects.
      Register new_val = (args.rt != 31) ? GetReg(args.rt) : GetImm(0);
      Register resv_addr =
          std::get<0>(Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = resv_addr_off}));
      if (!success()) {
        return;
      }

      // Clear the reservation (STXR always clears, success or not).
      builder_.GenPutImm(resv_addr_off, 0);

      Register status = AllocTempReg();
      auto* ir = builder_.ir();
      auto* cur_bb = builder_.bb();
      MachineBasicBlock* match_bb = ir->NewBasicBlock();   // reservation addr matches base
      MachineBasicBlock* fail_bb = ir->NewBasicBlock();    // addr mismatch or CMPXCHG fail
      MachineBasicBlock* swap_ok_bb = ir->NewBasicBlock();  // CMPXCHG succeeded
      MachineBasicBlock* merge_bb = ir->NewBasicBlock();
      ir->AddEdge(cur_bb, match_bb);
      ir->AddEdge(cur_bb, fail_bb);

      // if (reservation_address != base) goto fail_bb (status = 1).
      builder_.Gen<PseudoCondBranch>(
          x86_64::Assembler::Condition::kNotEqual,
          fail_bb,
          match_bb,
          std::get<0>(Gen<x86_64::CmpqRegReg>(resv_addr, base)));

      // --- match path: sized LOCK CMPXCHG(expected, [base], new_val). ---
      builder_.StartBasicBlock(match_bb);
      // Load the expected value (saved reservation) into the CMPXCHG accumulator.
      Register expected =
          std::get<0>(Gen<x86_64::MovqRegOp>({.base = x86_64::kMachineRegRBP, .disp = resv_val_off}));
      Register host_flags;
      switch (args.size) {
        case 0:
          std::tie(expected, host_flags) =
              Gen<x86_64::LockCmpXchgbRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 1:
          std::tie(expected, host_flags) =
              Gen<x86_64::LockCmpXchgwRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 2:
          std::tie(expected, host_flags) =
              Gen<x86_64::LockCmpXchglRegOpReg>(expected, {.base = base}, new_val);
          break;
        case 3:
          std::tie(expected, host_flags) =
              Gen<x86_64::LockCmpXchgqRegOpReg>(expected, {.base = base}, new_val);
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      // No fault-recovery split here (unlike Load()/Store()): the CMPXCHG runs
      // only when reservation_address == base, i.e. a prior LDXR already faulted
      // in this page, so a fault is pathological. Splitting the block would also
      // strand the CMPXCHG's FLAGS output across the recovery edge. This mirrors
      // the riscv64 heavy frontend's MemoryRegionReservationExchange, which also
      // omits recovery around the swap CMPXCHG.

      // ZF=0 (kNotZero) means CMPXCHG failed -> fail_bb (status = 1); ZF=1 ->
      // swap_ok_bb (status = 0).
      ir->AddEdge(builder_.bb(), fail_bb);
      ir->AddEdge(builder_.bb(), swap_ok_bb);
      builder_.Gen<PseudoCondBranch>(
          x86_64::Assembler::Condition::kNotZero, fail_bb, swap_ok_bb, host_flags);

      // --- success: status = 0 (XOR self, with a PseudoDef to seat the value). ---
      builder_.StartBasicBlock(swap_ok_bb);
      builder_.Gen<PseudoDefReg>(status);
      builder_.Gen<x86_64::XorqRegReg>(status, status, GetFlagsRegister());
      ir->AddEdge(swap_ok_bb, merge_bb);
      builder_.Gen<PseudoBranch>(merge_bb);

      // --- failure (addr mismatch or CMPXCHG miss): status = 1. ---
      builder_.StartBasicBlock(fail_bb);
      builder_.Gen<x86_64::MovqRegImm>(status, int64_t{1});
      ir->AddEdge(fail_bb, merge_bb);
      builder_.Gen<PseudoBranch>(merge_bb);

      builder_.StartBasicBlock(merge_bb);
      if (args.rs != 31) {
        SetReg(args.rs, status);
      }
      return;
    }

    case Decoder::AtomicOp::kCas: {
      // CAS Xs, Xt, [Xn]: compare [Xn] with Xs; on a match store Xt to [Xn];
      // the old value of [Xn] is written back to Xs. Mirrors lite_translator.h's
      // kCas: x86 LOCK CMPXCHG compares RAX with [mem], stores the source on a
      // match, and leaves the old memory value in RAX. GenRecoveryBlockForLastInsn()
      // delivers a guest fault on a bad pointer. Plain CAS never branches, so the
      // CMPXCHG's FLAGS output is discarded (std::get<0> keeps only the RAX
      // result); nothing FLAGS-class is live across the recovery edge — unlike
      // STXR, where the FLAGS feed a branch and recovery is therefore omitted.
      Register expected = (args.rs != 31) ? GetReg(args.rs) : GetImm(0);
      Register desired = (args.rt != 31) ? GetReg(args.rt) : GetImm(0);
      switch (args.size) {
        case 0:
          expected =
              std::get<0>(Gen<x86_64::LockCmpXchgbRegOpReg>(expected, {.base = base}, desired));
          break;
        case 1:
          expected =
              std::get<0>(Gen<x86_64::LockCmpXchgwRegOpReg>(expected, {.base = base}, desired));
          break;
        case 2:
          expected =
              std::get<0>(Gen<x86_64::LockCmpXchglRegOpReg>(expected, {.base = base}, desired));
          break;
        case 3:
          expected =
              std::get<0>(Gen<x86_64::LockCmpXchgqRegOpReg>(expected, {.base = base}, desired));
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      GenRecoveryBlockForLastInsn();
      // The old value is now in RAX (expected). ARM CAS Ws zero-extends the old
      // value to 64 bits; on a CMPXCHG *match* the accumulator keeps the full
      // 64-bit operand's upper bits, so re-zero-extend the sub-64 forms (mirrors
      // lite's byte/halfword AND masks and 32-bit MOVL).
      if (args.rs != 31) {
        if (args.size == 0) {
          expected =
              std::get<0>(Gen<x86_64::AndqRegImm>(expected, static_cast<int32_t>(0xFF)));
        } else if (args.size == 1) {
          expected =
              std::get<0>(Gen<x86_64::AndqRegImm>(expected, static_cast<int32_t>(0xFFFF)));
        } else if (args.size == 2) {
          expected = std::get<0>(Gen<x86_64::MovlRegReg>(expected));
        }
        SetReg(args.rs, expected);
      }
      return;
    }

    case Decoder::AtomicOp::kSwp: {
      // SWP Xs, Xt, [Xn]: atomically swap [Xn] with Xs; the old value goes to
      // Xt. Mirrors lite_translator.h's kSwp: x86 XCHG with a memory operand is
      // implicitly LOCKed. XCHG has no FLAGS operand, so the recovery split is
      // exactly as safe as a plain Store's.
      Register new_val = (args.rs != 31) ? GetReg(args.rs) : GetImm(0);
      switch (args.size) {
        case 0:
          new_val = std::get<0>(Gen<x86_64::XchgbRegOp>(new_val, {.base = base}));
          break;
        case 1:
          new_val = std::get<0>(Gen<x86_64::XchgwRegOp>(new_val, {.base = base}));
          break;
        case 2:
          new_val = std::get<0>(Gen<x86_64::XchglRegOp>(new_val, {.base = base}));
          break;
        case 3:
          new_val = std::get<0>(Gen<x86_64::XchgqRegOp>(new_val, {.base = base}));
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      GenRecoveryBlockForLastInsn();
      // byte/halfword XCHG leaves the upper bits of new_val as the original guest
      // Xs; ARM SWP Wt zero-extends the old memory value to 64. The 32-bit XCHG
      // already zero-extends (a 32-bit reg write clears bits 63:32), matching
      // lite, which only masks the byte/halfword forms.
      if (args.size == 0) {
        new_val = std::get<0>(Gen<x86_64::AndqRegImm>(new_val, static_cast<int32_t>(0xFF)));
      } else if (args.size == 1) {
        new_val = std::get<0>(Gen<x86_64::AndqRegImm>(new_val, static_cast<int32_t>(0xFFFF)));
      }
      if (args.rt != 31) {
        SetReg(args.rt, new_val);
      }
      return;
    }

    case Decoder::AtomicOp::kLdadd: {
      // LDADD Xs, Xt, [Xn]: atomically add Xs to [Xn]; the old value goes to Xt.
      // Mirrors lite_translator.h's kLdadd: x86 LOCK XADD adds the source to
      // [mem] and leaves the old memory value in the source register. ARM LDADD
      // does not set NZCV, so the XADD's FLAGS output is discarded (std::get<0>
      // keeps only the value); nothing FLAGS-class is live across the recovery
      // edge.
      Register addend = (args.rs != 31) ? GetReg(args.rs) : GetImm(0);
      // LOCK XADD models FLAGS as use_def (XADD writes all flags from the sum),
      // so the frontend Gen adapter takes an explicit FLAGS operand; the value
      // is ignored — Gen overrides it with GetFlagsRegister() — but must be
      // passed to satisfy the operand count.
      switch (args.size) {
        case 0:
          addend =
              std::get<0>(Gen<x86_64::LockXaddbOpReg>({.base = base}, addend, GetFlagsRegister()));
          break;
        case 1:
          addend =
              std::get<0>(Gen<x86_64::LockXaddwOpReg>({.base = base}, addend, GetFlagsRegister()));
          break;
        case 2:
          addend =
              std::get<0>(Gen<x86_64::LockXaddlOpReg>({.base = base}, addend, GetFlagsRegister()));
          break;
        case 3:
          addend =
              std::get<0>(Gen<x86_64::LockXaddqOpReg>({.base = base}, addend, GetFlagsRegister()));
          break;
        default:
          UndefinedReturningVoid();
          return;
      }
      GenRecoveryBlockForLastInsn();
      // byte/halfword XADD only updates the low bits of addend; ARM LDADD Wt
      // zero-extends the old memory value to 64. The 32-bit XADD already
      // zero-extends, so mirror lite and mask only the byte/halfword forms.
      if (args.size == 0) {
        addend = std::get<0>(Gen<x86_64::AndqRegImm>(addend, static_cast<int32_t>(0xFF)));
      } else if (args.size == 1) {
        addend = std::get<0>(Gen<x86_64::AndqRegImm>(addend, static_cast<int32_t>(0xFFFF)));
      }
      if (args.rt != 31) {
        SetReg(args.rt, addend);
      }
      return;
    }

    default:
      // Remaining LSE atomics (LDCLR/LDSET/LDEOR, LDSMAX/MIN, LDUMAX/MIN),
      // CASP, and the LDXP/STXP pair forms are not yet mirrored into the heavy
      // tier; bail to the lite translator (correct, just slower).
      UndefinedReturningVoid();
      return;
  }
}

void HeavyOptimizerFrontend::DataMemoryBarrier() {
  // Full DMB/DSB -> MFENCE: recover the StoreLoad ordering x86 TSO omits.
  builder_.Gen<x86_64::Mfence>();
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
