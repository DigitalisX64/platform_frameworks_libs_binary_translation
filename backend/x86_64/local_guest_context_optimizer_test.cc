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

}  // namespace

}  // namespace berberis
