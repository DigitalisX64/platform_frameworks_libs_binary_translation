/*
 * Copyright (C) 2025 The Android Open Source Project
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

#include <tuple>

#include "berberis/backend/x86_64/read_flags_optimizer.h"

#include "berberis/backend/common/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_analysis.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

namespace {

std::tuple<MachineBasicBlock*, MachineBasicBlock*, MachineBasicBlock*, MachineBasicBlock*>
BuildBasicLoop(MachineIR* machine_ir) {
  x86_64::MachineIRBuilder builder(machine_ir);

  auto bb0 = machine_ir->NewBasicBlock();
  auto bb1 = machine_ir->NewBasicBlock();
  auto bb2 = machine_ir->NewBasicBlock();
  auto bb3 = machine_ir->NewBasicBlock();
  machine_ir->AddEdge(bb0, bb1);
  machine_ir->AddEdge(bb1, bb2);
  machine_ir->AddEdge(bb2, bb1);
  machine_ir->AddEdge(bb2, bb3);

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoBranch>(bb1);
  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoBranch>(bb2);

  builder.StartBasicBlock(bb2);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb1, bb3, kMachineRegFLAGS);

  builder.StartBasicBlock(bb3);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  return {bb0, bb1, bb2, bb3};
}

TEST(MachineIRReadFlagsOptimizer, CheckRegsUnusedWithinInsnRangeAddsReg) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs({flags0}, machine_ir.arena());

  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(flags1, flags0, 8);
  builder.Gen<PseudoWriteFlags>(flags1, kMachineRegFLAGS);
  builder.Gen<PseudoBranch>(bb1);

  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto insn_it = bb0->insn_list().begin();
  // Skip the pseudoreadflags instruction.
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  insn_it++;
  ASSERT_FALSE(CheckRegsUnusedWithinInsnRange(insn_it, bb0->insn_list().end(), regs));
  ASSERT_TRUE(
      CheckRegsUnusedWithinInsnRange(bb1->insn_list().begin(), bb1->insn_list().end(), regs));
  ASSERT_EQ(regs.size(), 2UL);
}

TEST(MachineIRReadFlagsOptimizer, CheckRegsUnusedWithinInsnRange) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs0({flags0}, machine_ir.arena());
  ArenaVector<MachineReg> regs1({flags1}, machine_ir.arena());

  auto bb0 = machine_ir.NewBasicBlock();

  builder.StartBasicBlock(bb0);
  builder.Gen<MovqRegImm>(flags0, 123);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);
  auto insn_it = bb0->insn_list().begin();
  ASSERT_FALSE(CheckRegsUnusedWithinInsnRange(insn_it, bb0->insn_list().end(), regs0));
  ASSERT_TRUE(CheckRegsUnusedWithinInsnRange(insn_it, bb0->insn_list().end(), regs1));
  ASSERT_EQ(regs0.size(), 1UL);
}

TEST(MachineIRReadFlagsOptimizer, CheckPostLoopNodeLifetime) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  MachineReg flags_copy = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs({flags, flags_copy}, machine_ir.arena());

  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(flags_copy, flags, 8);
  builder.Gen<PseudoBranch>(bb1);

  builder.StartBasicBlock(bb1);
  builder.Gen<x86_64::AddqRegReg>(flags_copy, flags_copy, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  bb1->live_in().push_back(flags_copy);
  ASSERT_TRUE(CheckPostLoopNode(bb1, regs));

  // Should fail because flags_copy shouldln't outlive bb1.
  bb1->live_out().push_back(flags_copy);
  ASSERT_FALSE(CheckPostLoopNode(bb1, regs));
}

// CheckPostLoopNode should pass if no livein.
TEST(MachineIRReadFlagsOptimizer, CheckPostLoopNodeLiveIn) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs({flags}, machine_ir.arena());

  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  auto bb2 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);
  machine_ir.AddEdge(bb1, bb2);
  machine_ir.AddEdge(bb2, bb1);

  // This should pass even though in_edges > 1 because it has no live_in.
  ASSERT_TRUE(CheckPostLoopNode(bb1, regs));

  // Just to keep us honest that it fails.
  bb1->live_in().push_back(flags);
  ASSERT_FALSE(CheckPostLoopNode(bb1, regs));
}

// Test that CheckPostLoopNode fails when node has more than one in_edge.
TEST(MachineIRReadFlagsOptimizer, CheckPostLoopNodeInEdges) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs({flags}, machine_ir.arena());

  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  auto bb2 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);
  machine_ir.AddEdge(bb1, bb2);

  bb1->live_in().push_back(flags);
  ASSERT_TRUE(CheckPostLoopNode(bb1, regs));
  machine_ir.AddEdge(bb2, bb1);
  ASSERT_FALSE(CheckPostLoopNode(bb1, regs));
}

// Test that CheckSuccessorNode fails if we are using register in regs.
TEST(MachineIRReadFlagsOptimizer, CheckSuccessorNodeFailsIfUsingRegisters) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs({flags}, machine_ir.arena());

  auto [preloop, loop_head, loop_exit, postloop] = BuildBasicLoop(&machine_ir);
  loop_exit->live_in().push_back(flags);
  loop_exit->insn_list().insert(loop_exit->insn_list().begin(),
                                machine_ir.NewInsn<MovqRegImm>(flags, 123));

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto loop_tree = BuildLoopTree(&machine_ir);
  auto loop = loop_tree.root()->GetInnerloopNode(0)->loop();
  ASSERT_FALSE(CheckSuccessorNode(loop, loop_exit, regs));
}

TEST(MachineIRReadFlagsOptimizer, CheckSuccessorNodeFailsIfNotExit) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs({flags}, machine_ir.arena());

  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  auto bb2 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);
  machine_ir.AddEdge(bb1, bb2);
  machine_ir.AddEdge(bb2, bb1);
  bb2->live_in().push_back(flags);

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoBranch>(bb1);
  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoBranch>(bb2);
  builder.StartBasicBlock(bb2);
  builder.Gen<PseudoBranch>(bb1);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto loop_tree = BuildLoopTree(&machine_ir);
  auto loop = loop_tree.root()->GetInnerloopNode(0)->loop();
  // Should fail because not an exit node.
  ASSERT_FALSE(CheckSuccessorNode(loop, bb2, regs));
}

// Check that we test for only one in_edge.
TEST(MachineIRReadFlagsOptimizer, CheckSuccessorNodeInEdges) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs({flags}, machine_ir.arena());

  auto [preloop, loop_head, loop_exit, postloop] = BuildBasicLoop(&machine_ir);

  auto loop_tree = BuildLoopTree(&machine_ir);
  auto loop = loop_tree.root()->GetInnerloopNode(0)->loop();

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  loop_exit->live_in().push_back(flags);
  ASSERT_TRUE(CheckSuccessorNode(loop, loop_exit, regs));
  machine_ir.AddEdge(preloop, loop_exit);
  ASSERT_FALSE(CheckSuccessorNode(loop, loop_exit, regs));
}

// regs should not be live_in to other loop nodes.
TEST(MachineIRReadFlagsOptimizer, CheckSuccessorNodeLiveIn) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();
  ArenaVector<MachineReg> regs({flags0}, machine_ir.arena());

  auto [preloop, loop_head, loop_exit, postloop] = BuildBasicLoop(&machine_ir);

  loop_exit->live_in().push_back(flags0);

  loop_exit->insn_list().insert(loop_exit->insn_list().begin(),
                                machine_ir.NewInsn<PseudoCopy>(flags1, flags0, 8));

  postloop->live_in().push_back(flags1);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto loop_tree = BuildLoopTree(&machine_ir);
  auto loop = loop_tree.root()->GetInnerloopNode(0)->loop();

  ASSERT_TRUE(CheckSuccessorNode(loop, loop_exit, regs));
  // Remove flags1.
  regs.pop_back();

  // Make sure we fail if flags0 is live_in of another loop node.
  loop_head->live_in().push_back(flags0);
  ASSERT_FALSE(CheckSuccessorNode(loop, loop_exit, regs));

  // Reset state.
  loop_head->live_in().pop_back();
  regs.pop_back();

  // Make sure that we check live_in after CheckRegsUnusedWithinInsnRange.
  loop_head->live_in().push_back(flags1);
  ASSERT_FALSE(CheckSuccessorNode(loop, loop_exit, regs));
}

// Helper function to check that two instructions are the same.
void TestCopiedInstruction(MachineIR* machine_ir, MachineInsn* insn) {
  MachineReg reg = machine_ir->AllocVReg();

  auto gen = GetInsnGen(insn->opcode());
  ASSERT_TRUE(gen.has_value());
  auto* copy = gen.value()(machine_ir, insn);

  ASSERT_EQ(copy->opcode(), insn->opcode());
  ASSERT_EQ(copy->NumRegOperands(), insn->NumRegOperands());
  for (auto i = 0; i < insn->NumRegOperands(); i++) {
    ASSERT_EQ(copy->RegAt(i), insn->RegAt(i));
  }

  // Check that it's a deep copy.
  copy->SetRegAt(0, reg);
  ASSERT_NE(copy->RegAt(0), insn->RegAt(0));
  ASSERT_EQ(copy->RegAt(0), reg);
}

TEST(MachineIRReadFlagsOptimizer, GetInsnGen) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  TestCopiedInstruction(&machine_ir,
                        machine_ir.NewInsn<AddqRegReg>(
                            machine_ir.AllocVReg(), machine_ir.AllocVReg(), kMachineRegFLAGS));
  // PseudoReadFlags is a special case as it has its own member variables and
  // doesn't inherit from MachineInsnX86_64 so we test it too.
  TestCopiedInstruction(
      &machine_ir,
      machine_ir.NewInsn<PseudoReadFlags>(
          PseudoReadFlags::kWithOverflow, machine_ir.AllocVReg(), kMachineRegFLAGS));
}

TEST(MachineIRReadFlagsOptimizer, FindFlagSettingInsn) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg reg0 = machine_ir.AllocVReg();
  MachineReg reg1 = machine_ir.AllocVReg();
  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();
  MachineReg reg_with_flags0 = machine_ir.AllocVReg();

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  builder.Gen<AddqRegReg>(reg0, reg1, flags0);
  builder.Gen<SubqRegImm>(reg1, 1234, flags0);
  builder.Gen<AddqRegReg>(reg1, reg0, flags1);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, reg_with_flags0, flags0);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  // Move to PseudoReadFlags.
  auto insn_it = std::prev(bb->insn_list().end(), 2);
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);

  auto flag_setter = FindFlagSettingInsn(insn_it, bb->insn_list().begin(), flags0);
  ASSERT_TRUE(flag_setter.has_value());
  ASSERT_EQ((*flag_setter.value())->opcode(), kMachineOpSubqRegImm);

  // Test that we exit properly when we can't find the instruction.
  // Move to second AddqRegReg.
  insn_it--;
  flag_setter = FindFlagSettingInsn(insn_it, bb->insn_list().begin(), flags1);
  ASSERT_FALSE(flag_setter.has_value());
}

}  // namespace

}  // namespace berberis::x86_64
