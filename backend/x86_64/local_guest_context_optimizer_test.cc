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

#include "gtest/gtest.h"

#include "berberis/backend/x86_64/local_guest_context_optimizer.h"

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

namespace {

TEST(MachineIRLocalGuestContextOptimizer, RemoveReadAfterWrite) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto reg1 = machine_ir.AllocVReg();
  auto reg2 = machine_ir.AllocVReg();
  builder.GenPut(GetThreadStateRegOffset(0), reg1);
  builder.GenGet(reg2, GetThreadStateRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL);

  auto* store_insn = *bb->insn_list().begin();
  ASSERT_EQ(store_insn->opcode(), kMachineOpMovqMemBaseDispReg);
  auto disp = x86_64::AsMachineInsnX86_64(store_insn)->disp();
  ASSERT_EQ(disp, berberis::GetThreadStateRegOffset(0));
  auto replaced_reg = store_insn->RegAt(1);
  ASSERT_EQ(store_insn->RegAt(0), x86_64::kMachineRegRBP);

  auto* load_copy_insn = *std::next(bb->insn_list().begin());
  ASSERT_EQ(load_copy_insn->opcode(), kMachineOpPseudoCopy);
  ASSERT_EQ(load_copy_insn->RegAt(0), reg2);
  ASSERT_EQ(load_copy_insn->RegAt(1), replaced_reg);
}

TEST(MachineIRLocalGuestContextOptimizer, RemoveReadAfterWriteImmediate) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto reg1 = machine_ir.AllocVReg();
  builder.GenPutImm(GetThreadStateRegOffset(0), 6);
  builder.GenGet(reg1, GetThreadStateRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL);

  auto* store_insn = *bb->insn_list().begin();
  ASSERT_EQ(store_insn->opcode(), kMachineOpMovqMemBaseDispImm);
  auto disp = x86_64::AsMachineInsnX86_64(store_insn)->disp();
  ASSERT_EQ(disp, berberis::GetThreadStateRegOffset(0));
  ASSERT_EQ(store_insn->RegAt(0), x86_64::kMachineRegRBP);

  auto* load_copy_insn = *std::next(bb->insn_list().begin());
  ASSERT_EQ(load_copy_insn->opcode(), kMachineOpMovqRegImm);
  ASSERT_EQ(load_copy_insn->RegAt(0), reg1);
  uint64_t imm = x86_64::AsMachineInsnX86_64(store_insn)->imm();
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(load_copy_insn)->imm(), imm);
}

TEST(MachineIRLocalGuestContextOptimizer, RemoveReadAfterRead) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto reg1 = machine_ir.AllocVReg();
  auto reg2 = machine_ir.AllocVReg();
  builder.GenGet(reg1, GetThreadStateRegOffset(0));
  builder.GenGet(reg2, GetThreadStateRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL);
  auto* load_insn = *bb->insn_list().begin();
  ASSERT_EQ(load_insn->opcode(), kMachineOpMovqRegMemBaseDisp);
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(load_insn)->disp(), berberis::GetThreadStateRegOffset(0));
  ASSERT_EQ(load_insn->RegAt(0), reg1);
  ASSERT_EQ(load_insn->RegAt(1), x86_64::kMachineRegRBP);

  auto* copy_insn = *std::next(bb->insn_list().begin());
  ASSERT_EQ(copy_insn->opcode(), kMachineOpPseudoCopy);
  ASSERT_EQ(copy_insn->RegAt(0), reg2);
  ASSERT_EQ(copy_insn->RegAt(1), reg1);
}

TEST(MachineIRLocalGuestContextOptimizer, RemoveWriteBeforeWrite) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto reg1 = machine_ir.AllocVReg();
  auto reg2 = machine_ir.AllocVReg();
  builder.GenPut(GetThreadStateRegOffset(0), reg1);
  builder.GenPut(GetThreadStateRegOffset(0), reg2);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 2UL);
  auto* store_insn = *bb->insn_list().begin();
  ASSERT_EQ(store_insn->opcode(), kMachineOpMovqMemBaseDispReg);
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(store_insn)->disp(), berberis::GetThreadStateRegOffset(0));
  ASSERT_EQ(store_insn->RegAt(1), reg2);
  ASSERT_EQ(store_insn->RegAt(0), x86_64::kMachineRegRBP);
}

TEST(MachineIRLocalGuestContextOptimizer, RemoveWriteImmediateBeforeWrite) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto reg1 = machine_ir.AllocVReg();
  auto reg2 = machine_ir.AllocVReg();
  builder.GenPutImm(GetThreadStateRegOffset(0), 5);
  builder.GenGet(reg2, GetThreadStateRegOffset(0));
  builder.GenPut(GetThreadStateRegOffset(0), reg1);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL);

  auto* load_insn = *bb->insn_list().begin();
  ASSERT_EQ(load_insn->opcode(), kMachineOpMovqRegImm);
  ASSERT_EQ(load_insn->RegAt(0), reg2);
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(load_insn)->imm(), 5UL);

  auto* store_insn = *std::next(bb->insn_list().begin());
  ASSERT_EQ(store_insn->opcode(), kMachineOpMovqMemBaseDispReg);
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(store_insn)->disp(), berberis::GetThreadStateRegOffset(0));
  ASSERT_EQ(store_insn->RegAt(1), reg1);
  ASSERT_EQ(store_insn->RegAt(0), x86_64::kMachineRegRBP);
}

TEST(MachineIRLocalGuestContextOptimizer, RemoveWriteBeforeWriteImmediate) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto reg1 = machine_ir.AllocVReg();
  auto reg2 = machine_ir.AllocVReg();
  builder.GenPut(GetThreadStateRegOffset(0), reg1);
  builder.GenGet(reg2, GetThreadStateRegOffset(0));
  builder.GenPutImm(GetThreadStateRegOffset(0), 5);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL);

  auto* load_insn = *bb->insn_list().begin();
  ASSERT_EQ(load_insn->opcode(), kMachineOpPseudoCopy);
  ASSERT_EQ(load_insn->RegAt(0), reg2);
  ASSERT_EQ(load_insn->RegAt(1), reg1);

  auto* store_insn = *std::next(bb->insn_list().begin());
  ASSERT_EQ(store_insn->opcode(), kMachineOpMovqMemBaseDispImm);
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(store_insn)->disp(), berberis::GetThreadStateRegOffset(0));
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(store_insn)->imm(), 5UL);
  ASSERT_EQ(store_insn->RegAt(0), x86_64::kMachineRegRBP);
}

TEST(MachineIRLocalGuestContextOptimizer, DoNotRemoveAccessToMonitorValue) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto reg1 = machine_ir.AllocVReg();
  auto reg2 = machine_ir.AllocVReg();
  constexpr auto offset = offsetof(ProcessState, cpu.reservation_value);
  builder.Gen<x86_64::MovqOpReg>({.base = x86_64::kMachineRegRBP, .disp = offset}, reg1);
  builder.Gen<x86_64::MovqOpReg>({.base = x86_64::kMachineRegRBP, .disp = offset}, reg2);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL);
  auto* store_insn_1 = *bb->insn_list().begin();
  ASSERT_EQ(store_insn_1->opcode(), kMachineOpMovqMemBaseDispReg);
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(store_insn_1)->disp(), offset);

  auto* store_insn_2 = *std::next(bb->insn_list().begin());
  ASSERT_EQ(store_insn_2->opcode(), kMachineOpMovqMemBaseDispReg);
  ASSERT_EQ(x86_64::AsMachineInsnX86_64(store_insn_2)->disp(), offset);
}

TEST(MachineIRLocalGuestContextOptimizer, LimitRegisters) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  builder.GenGet(machine_ir.AllocVReg(), GetThreadStateRegOffset(1));
  builder.GenGet(machine_ir.AllocVReg(), GetThreadStateRegOffset(1));
  builder.GenGet(machine_ir.AllocVReg(), GetThreadStateRegOffset(0));
  builder.GenGet(machine_ir.AllocVReg(), GetThreadStateRegOffset(0));
  builder.GenGet(machine_ir.AllocVReg(), GetThreadStateRegOffset(0));

  if (DoesCpuStateHaveDedicatedSimdRegs()) {
    builder.GenGet(machine_ir.AllocVReg(), GetThreadStateSimdRegOffset(5));
    builder.GenGet(machine_ir.AllocVReg(), GetThreadStateSimdRegOffset(2));
    builder.GenGet(machine_ir.AllocVReg(), GetThreadStateSimdRegOffset(5));
    builder.GenGet(machine_ir.AllocVReg(), GetThreadStateSimdRegOffset(2));
    builder.GenGet(machine_ir.AllocVReg(), GetThreadStateSimdRegOffset(0));
    builder.GenGet(machine_ir.AllocVReg(), GetThreadStateSimdRegOffset(5));
  }
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir,
                                          x86_64::OptimizeLocalParams{
                                              .general_reg_limit = 1,
                                              .simd_reg_limit = 2,
                                          });
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  // Check instructions with general regs replaced.
  auto insn_it = bb->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpMovqRegMemBaseDisp);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpMovqRegMemBaseDisp);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpMovqRegMemBaseDisp);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;

  // Check instructions with simd regs replaced.
  if (DoesCpuStateHaveDedicatedSimdRegs()) {
    ASSERT_EQ((*insn_it)->opcode(), kMachineOpMovqRegMemBaseDisp);
    insn_it++;
    ASSERT_EQ((*insn_it)->opcode(), kMachineOpMovqRegMemBaseDisp);
    insn_it++;
    ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
    insn_it++;
    ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
    insn_it++;
    ASSERT_EQ((*insn_it)->opcode(), kMachineOpMovqRegMemBaseDisp);
    insn_it++;
    ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
    insn_it++;
    ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoJump);
  }
}


// A 64-bit MOVSD load of a guest SIMD register zeroes the upper half of its
// XMM, so it does not hold the bits a later 128-bit read of the same register
// needs. Forwarding it would silently replace that register's upper half with
// zeroes, which is how a bcrypt key expansion (fmov w, s0 followed by
// eor v1.16b, v1.16b, v0.16b) came out wrong with no crash.
TEST(MachineIRLocalGuestContextOptimizer, KeepWiderGetAfterNarrowerGet) {
  if (!DoesCpuStateHaveDedicatedSimdRegs()) {
    GTEST_SKIP() << "guest has no dedicated SIMD registers";
  }
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto narrow = machine_ir.AllocVReg();
  auto wide = machine_ir.AllocVReg();
  builder.GenGetSimd<8>(narrow, GetThreadStateSimdRegOffset(0));
  builder.GenGetSimd<16>(wide, GetThreadStateSimdRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL);
  auto* narrow_insn = *bb->insn_list().begin();
  EXPECT_EQ(narrow_insn->opcode(), kMachineOpMovsdXRegMemBaseDisp);
  auto* wide_insn = *std::next(bb->insn_list().begin());
  EXPECT_EQ(wide_insn->opcode(), kMachineOpMovdqaXRegMemBaseDisp)
      << "128-bit get must not be forwarded from a 64-bit get of the same slot";
  EXPECT_EQ(wide_insn->RegAt(0), wide);
}

// Same hole on the store side: a 64-bit put writes only the lower half of the
// slot, so a following 128-bit get still has to read the upper half back from
// memory.
TEST(MachineIRLocalGuestContextOptimizer, KeepWiderGetAfterNarrowerPut) {
  if (!DoesCpuStateHaveDedicatedSimdRegs()) {
    GTEST_SKIP() << "guest has no dedicated SIMD registers";
  }
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto stored = machine_ir.AllocVReg();
  auto loaded = machine_ir.AllocVReg();
  builder.GenSetSimd<8>(GetThreadStateSimdRegOffset(0), stored);
  builder.GenGetSimd<16>(loaded, GetThreadStateSimdRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL);
  auto* load_insn = *std::next(bb->insn_list().begin());
  EXPECT_EQ(load_insn->opcode(), kMachineOpMovdqaXRegMemBaseDisp)
      << "128-bit get must not be forwarded from a 64-bit put of the same slot";
}

// A narrower store leaves the bytes an earlier wider store wrote, so the wider
// store is not dead and must survive.
TEST(MachineIRLocalGuestContextOptimizer, KeepWiderPutBeforeNarrowerPut) {
  if (!DoesCpuStateHaveDedicatedSimdRegs()) {
    GTEST_SKIP() << "guest has no dedicated SIMD registers";
  }
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto wide = machine_ir.AllocVReg();
  auto narrow = machine_ir.AllocVReg();
  builder.GenSetSimd<16>(GetThreadStateSimdRegOffset(0), wide);
  builder.GenSetSimd<8>(GetThreadStateSimdRegOffset(0), narrow);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_EQ(bb->insn_list().size(), 3UL)
      << "128-bit put is not dead: the following 64-bit put covers only half of it";
  EXPECT_EQ((*bb->insn_list().begin())->opcode(), kMachineOpMovdqaMemBaseDispXReg);
}

// The equal-width case must still be optimized, so the guard above does not
// silently disable the pass.
TEST(MachineIRLocalGuestContextOptimizer, ForwardsSameWidthSimdGet) {
  if (!DoesCpuStateHaveDedicatedSimdRegs()) {
    GTEST_SKIP() << "guest has no dedicated SIMD registers";
  }
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto first = machine_ir.AllocVReg();
  auto second = machine_ir.AllocVReg();
  builder.GenGetSimd<16>(first, GetThreadStateSimdRegOffset(0));
  builder.GenGetSimd<16>(second, GetThreadStateSimdRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::RemoveLocalGuestContextAccesses(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto* second_insn = *std::next(bb->insn_list().begin());
  EXPECT_EQ(second_insn->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(second_insn->RegAt(0), second);
  EXPECT_EQ(second_insn->RegAt(1), first);
}

// region digitalis
// A block ended by a faulting guest load: bb -> {next, recovery}, the shape
// the heavy frontends' GenRecoveryBlockForLastInsn builds.
MachineBasicBlock* GenGuestLoadSplit(x86_64::MachineIR* machine_ir,
                                             x86_64::MachineIRBuilder* builder) {
  auto* bb = builder->bb();
  auto* next = machine_ir->NewBasicBlock();
  auto* recovery = machine_ir->NewBasicBlock();
  machine_ir->AddEdge(bb, next);
  machine_ir->AddEdge(bb, recovery);
  builder->Gen<x86_64::MovqRegOp>(machine_ir->AllocVReg(), {.base = machine_ir->AllocVReg()});
  builder->SetRecoveryPointAtLastInsn(recovery);
  builder->Gen<PseudoBranch>(next);
  builder->StartBasicBlock(recovery);
  builder->Gen<PseudoJump>(kNullGuestAddr);
  builder->StartBasicBlock(next);
  return next;
}

TEST(MachineIRRecoverySplitGetForwarding, ForwardsPutAcrossSplit) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto value = machine_ir.AllocVReg();
  auto read = machine_ir.AllocVReg();
  builder.GenPut(GetThreadStateRegOffset(0), value);
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGet(read, GetThreadStateRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  // The store stays for the recovery exit; the read becomes a copy.
  EXPECT_EQ((*bb->insn_list().begin())->opcode(), kMachineOpMovqMemBaseDispReg);
  auto* get = *next->insn_list().begin();
  ASSERT_EQ(get->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(get->RegAt(0), read);
  EXPECT_EQ(get->RegAt(1), value);
}

TEST(MachineIRRecoverySplitGetForwarding, ForwardsGetAlongChain) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto first = machine_ir.AllocVReg();
  auto second = machine_ir.AllocVReg();
  builder.GenGet(first, GetThreadStateRegOffset(1));
  GenGuestLoadSplit(&machine_ir, &builder);
  auto* last = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGet(second, GetThreadStateRegOffset(1));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto* get = *last->insn_list().begin();
  ASSERT_EQ(get->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(get->RegAt(0), second);
  EXPECT_EQ(get->RegAt(1), first);
}

// A narrower vector read gets MOVQ, which zeroes the upper half like the
// MOVSD load it replaces.
TEST(MachineIRRecoverySplitGetForwarding, NarrowSimdGetZeroExtends) {
  if (!DoesCpuStateHaveDedicatedSimdRegs()) {
    GTEST_SKIP() << "guest has no dedicated SIMD registers";
  }
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto value = machine_ir.AllocVReg();
  auto low = machine_ir.AllocVReg();
  auto full = machine_ir.AllocVReg();
  builder.GenSetSimd<16>(GetThreadStateSimdRegOffset(0), value);
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGetSimd<8>(low, GetThreadStateSimdRegOffset(0));
  builder.GenGetSimd<16>(full, GetThreadStateSimdRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto it = next->insn_list().begin();
  ASSERT_EQ((*it)->opcode(), kMachineOpMovqXRegXReg);
  EXPECT_EQ((*it)->RegAt(0), low);
  EXPECT_EQ((*it)->RegAt(1), value);
  // Full vectors are not carried across splits.
  ++it;
  EXPECT_EQ((*it)->opcode(), kMachineOpMovdqaXRegMemBaseDisp);
}

// A narrow put records only its own bytes: a later full-width read loads.
TEST(MachineIRRecoverySplitGetForwarding, NarrowSimdPutDoesNotFeedWideGet) {
  if (!DoesCpuStateHaveDedicatedSimdRegs()) {
    GTEST_SKIP() << "guest has no dedicated SIMD registers";
  }
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  builder.GenSetSimd<8>(GetThreadStateSimdRegOffset(0), machine_ir.AllocVReg());
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGetSimd<16>(machine_ir.AllocVReg(), GetThreadStateSimdRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  EXPECT_EQ((*next->insn_list().begin())->opcode(), kMachineOpMovdqaXRegMemBaseDisp);
}

// A put into the upper half of a vector register (at its own offset) makes the
// whole register's recorded value stale.
TEST(MachineIRRecoverySplitGetForwarding, OverlappingPutInvalidates) {
  if (!DoesCpuStateHaveDedicatedSimdRegs()) {
    GTEST_SKIP() << "guest has no dedicated SIMD registers";
  }
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  builder.GenSetSimd<16>(GetThreadStateSimdRegOffset(0), machine_ir.AllocVReg());
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenSetSimd<8>(GetThreadStateSimdRegOffset(0) + 8, machine_ir.AllocVReg());
  builder.GenGetSimd<16>(machine_ir.AllocVReg(), GetThreadStateSimdRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  EXPECT_EQ((*std::next(next->insn_list().begin()))->opcode(), kMachineOpMovdqaXRegMemBaseDisp);
}

// A store through the state pointer that is not a modeled put, a redefined
// register and a call each end the recorded value.
TEST(MachineIRRecoverySplitGetForwarding, UnmodeledWriteRedefinitionAndCallInvalidate) {
  for (int variant = 0; variant < 3; variant++) {
    Arena arena;
    x86_64::MachineIR machine_ir(&arena);
    x86_64::MachineIRBuilder builder(&machine_ir);

    auto* bb = machine_ir.NewBasicBlock();
    builder.StartBasicBlock(bb);
    auto value = machine_ir.AllocVReg();
    builder.GenPut(GetThreadStateRegOffset(0), value);
    auto* next = GenGuestLoadSplit(&machine_ir, &builder);
    if (variant == 0) {
      builder.Gen<x86_64::MovlOpReg>(
          {.base = x86_64::kMachineRegRBP,
           .disp = static_cast<int32_t>(GetThreadStateRegOffset(0) + 4)},
          machine_ir.AllocVReg());
    } else if (variant == 1) {
      builder.Gen<x86_64::AddqRegImm>(value, 1, x86_64::kMachineRegFLAGS);
    } else {
      builder.GenCallImm(0, machine_ir.AllocVReg());
    }
    builder.GenGet(machine_ir.AllocVReg(), GetThreadStateRegOffset(0));
    builder.Gen<PseudoJump>(kNullGuestAddr);

    x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
    EXPECT_EQ((*std::next(next->insn_list().begin()))->opcode(), kMachineOpMovqRegMemBaseDisp)
        << "variant " << variant;
  }
}

// A continuation that another branch also reaches starts its own chain.
TEST(MachineIRRecoverySplitGetForwarding, JoinBlockIsNotForwarded) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  auto* other = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  builder.GenPut(GetThreadStateRegOffset(0), machine_ir.AllocVReg());
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGet(machine_ir.AllocVReg(), GetThreadStateRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);
  builder.StartBasicBlock(other);
  builder.Gen<PseudoBranch>(next);
  machine_ir.AddEdge(other, next);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  EXPECT_EQ((*next->insn_list().begin())->opcode(), kMachineOpMovqRegMemBaseDisp);
}
// A value recorded further back than the forwarding distance is reloaded, and
// the reloaded value is carried on.
TEST(MachineIRRecoverySplitGetForwarding, DistantValueIsReloadedAndCarriedOn) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  builder.GenPut(GetThreadStateRegOffset(0), machine_ir.AllocVReg());
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  for (int i = 0; i < 40; i++) {
    builder.Gen<x86_64::MovqRegImm>(machine_ir.AllocVReg(), 0);
  }
  auto reloaded = machine_ir.AllocVReg();
  auto again = machine_ir.AllocVReg();
  builder.GenGet(reloaded, GetThreadStateRegOffset(0));
  auto* last = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGet(again, GetThreadStateRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  EXPECT_EQ((*std::next(next->insn_list().begin(), 40))->opcode(), kMachineOpMovqRegMemBaseDisp);
  auto* get = *last->insn_list().begin();
  ASSERT_EQ(get->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(get->RegAt(0), again);
  EXPECT_EQ(get->RegAt(1), reloaded);
}

// A forwarded get restarts the distance from its own copy, which then feeds
// the next get.
TEST(MachineIRRecoverySplitGetForwarding, ForwardedGetRestartsDistance) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto first = machine_ir.AllocVReg();
  auto second = machine_ir.AllocVReg();
  auto third = machine_ir.AllocVReg();
  builder.GenGet(first, GetThreadStateRegOffset(1));
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  for (int i = 0; i < 20; i++) {
    builder.Gen<x86_64::MovqRegImm>(machine_ir.AllocVReg(), 0);
  }
  builder.GenGet(second, GetThreadStateRegOffset(1));
  for (int i = 0; i < 20; i++) {
    builder.Gen<x86_64::MovqRegImm>(machine_ir.AllocVReg(), 0);
  }
  builder.GenGet(third, GetThreadStateRegOffset(1));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto* get = *std::next(next->insn_list().begin(), 20);
  ASSERT_EQ(get->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(get->RegAt(1), first);
  get = *std::next(next->insn_list().begin(), 41);
  ASSERT_EQ(get->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(get->RegAt(0), third);
  EXPECT_EQ(get->RegAt(1), second);
}

// An 8-byte FP or SIMD slot, which every guest has.
size_t GetNarrowSimdSlotOffset() {
  return DoesCpuStateHaveDedicatedFpRegs() ? GetThreadStateFRegOffset(0)
                                           : GetThreadStateSimdRegOffset(0);
}

// A get in the block of the value it would read is left to the local pass,
// which forwards it with a coalescable copy; the next block still gets it.
TEST(MachineIRRecoverySplitGetForwarding, SameBlockGetIsLeftToLocalPass) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto value = machine_ir.AllocVReg();
  auto later = machine_ir.AllocVReg();
  builder.GenPut(GetThreadStateRegOffset(0), value);
  builder.GenGet(machine_ir.AllocVReg(), GetThreadStateRegOffset(0));
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGet(later, GetThreadStateRegOffset(0));
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  EXPECT_EQ((*std::next(bb->insn_list().begin()))->opcode(), kMachineOpMovqRegMemBaseDisp);
  auto* get = *next->insn_list().begin();
  ASSERT_EQ(get->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(get->RegAt(0), later);
  EXPECT_EQ(get->RegAt(1), value);
}

// An 8-byte value a get loaded (or a forwarded copy of it) already has its
// upper half zeroed and is copied as is; one a put stored is zero-extended.
TEST(MachineIRRecoverySplitGetForwarding, NarrowSimdZeroExtendsOnlyStoredValues) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto* bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  auto loaded = machine_ir.AllocVReg();
  auto first = machine_ir.AllocVReg();
  auto second = machine_ir.AllocVReg();
  auto stored = machine_ir.AllocVReg();
  auto third = machine_ir.AllocVReg();
  builder.GenGetSimd<8>(loaded, GetNarrowSimdSlotOffset());
  auto* next = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGetSimd<8>(first, GetNarrowSimdSlotOffset());
  auto* next2 = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGetSimd<8>(second, GetNarrowSimdSlotOffset());
  builder.GenSetSimd<8>(GetNarrowSimdSlotOffset(), stored);
  auto* next3 = GenGuestLoadSplit(&machine_ir, &builder);
  builder.GenGetSimd<8>(third, GetNarrowSimdSlotOffset());
  builder.Gen<PseudoJump>(kNullGuestAddr);

  x86_64::ForwardGuestContextGetsAcrossRecoverySplits(&machine_ir);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto* get = *next->insn_list().begin();
  ASSERT_EQ(get->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(get->RegAt(1), loaded);
  get = *next2->insn_list().begin();
  ASSERT_EQ(get->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(get->RegAt(0), second);
  EXPECT_EQ(get->RegAt(1), first);
  get = *next3->insn_list().begin();
  ASSERT_EQ(get->opcode(), kMachineOpMovqXRegXReg);
  EXPECT_EQ(get->RegAt(0), third);
  EXPECT_EQ(get->RegAt(1), stored);
}

// endregion digitalis

}  // namespace

}  // namespace berberis
