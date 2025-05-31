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

#include <algorithm>

#include "berberis/backend/x86_64/read_flags_optimizer.h"

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_analysis.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/base/algorithm.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

namespace {

struct TestLoop {
  MachineBasicBlock* preloop;
  MachineBasicBlock* loop_head;
  MachineBasicBlock* loop_exit;
  MachineBasicBlock* postloop;
  MachineBasicBlock* successor;
  MachineBasicBlock* succ_postloop;
  MachineReg flags_reg;
  // Iterator which points to the READFLAGS instruction.
  MachineInsnList::iterator readflags_it;
};

TestLoop BuildBasicLoop(MachineIR* machine_ir) {
  x86_64::MachineIRBuilder builder(machine_ir);

  // bb0 -> bb1 -> bb2 -> bb3
  //         ^       |
  //         |----- bb4 -> bb5
  auto bb0 = machine_ir->NewBasicBlock();
  auto bb1 = machine_ir->NewBasicBlock();
  auto bb2 = machine_ir->NewBasicBlock();
  auto bb3 = machine_ir->NewBasicBlock();
  auto bb4 = machine_ir->NewBasicBlock();
  auto bb5 = machine_ir->NewBasicBlock();
  machine_ir->AddEdge(bb0, bb1);
  machine_ir->AddEdge(bb1, bb2);
  machine_ir->AddEdge(bb2, bb3);
  machine_ir->AddEdge(bb2, bb4);
  machine_ir->AddEdge(bb4, bb1);
  machine_ir->AddEdge(bb4, bb5);

  auto flags0 = machine_ir->AllocVReg();
  auto flags1 = machine_ir->AllocVReg();

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoBranch>(bb1);
  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoBranch>(bb2);

  builder.StartBasicBlock(bb2);
  builder.Gen<AddqRegReg>(machine_ir->AllocVReg(), machine_ir->AllocVReg(), kMachineRegFLAGS);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(flags1, flags0, 8);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb3, bb4, kMachineRegFLAGS);
  bb2->live_out().push_back(flags1);

  builder.StartBasicBlock(bb3);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  builder.StartBasicBlock(bb4);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb1, bb5, kMachineRegFLAGS);

  builder.StartBasicBlock(bb5);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  auto insn_it = std::next(bb2->insn_list().begin());
  CHECK_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);

  return {bb0, bb1, bb2, bb3, bb4, bb5, flags1, insn_it};
}

TEST(MachineIRReadFlagsOptimizer, CheckRegsUnusedWithinInsnRangeAddsReg) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();
  MachineRegVector regs({flags0}, machine_ir.arena());

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
  MachineRegVector regs0({flags0}, machine_ir.arena());
  MachineRegVector regs1({flags1}, machine_ir.arena());

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

TEST(MachineIRReadFlagsOptimizer, CheckPostLoopChecksRedefines) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  MachineRegVector regs({flags}, machine_ir.arena());

  auto bb0 = machine_ir.NewBasicBlock();

  bb0->live_in().push_back(flags);
  builder.StartBasicBlock(bb0);
  builder.Gen<x86_64::AddqRegReg>(flags, flags, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ASSERT_FALSE(CheckPostLoopNode(bb0, regs));
}

TEST(MachineIRReadFlagsOptimizer, CheckPostLoopNodeLifetime) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  MachineReg flags_copy = machine_ir.AllocVReg();
  MachineRegVector regs({flags, flags_copy}, machine_ir.arena());

  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(flags_copy, flags, 8);
  builder.Gen<PseudoBranch>(bb1);

  builder.StartBasicBlock(bb1);
  builder.Gen<x86_64::AddqRegReg>(machine_ir.AllocVReg(), flags_copy, kMachineRegFLAGS);
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
  MachineRegVector regs({flags}, machine_ir.arena());

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
  MachineRegVector regs({flags}, machine_ir.arena());

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
  MachineRegVector regs({flags}, machine_ir.arena());

  auto testloop = BuildBasicLoop(&machine_ir);
  testloop.loop_exit->live_in().push_back(flags);
  testloop.loop_exit->insn_list().insert(testloop.loop_exit->insn_list().begin(),
                                         machine_ir.NewInsn<MovqRegImm>(flags, 123));

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto loop_tree = BuildLoopTree(&machine_ir);
  auto loop = loop_tree.root()->GetInnerloopNode(0)->loop();
  ASSERT_FALSE(CheckSuccessorNode(loop, testloop.loop_exit, regs));
}

TEST(MachineIRReadFlagsOptimizer, CheckSuccessorNodeFailsIfNotExit) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags = machine_ir.AllocVReg();
  MachineRegVector regs({flags}, machine_ir.arena());

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

  auto testloop = BuildBasicLoop(&machine_ir);
  auto loop_tree = BuildLoopTree(&machine_ir);
  auto loop = loop_tree.root()->GetInnerloopNode(0)->loop();
  MachineRegVector regs({testloop.flags_reg}, machine_ir.arena());

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  testloop.successor->live_in().push_back(testloop.flags_reg);
  ASSERT_TRUE(CheckSuccessorNode(loop, testloop.successor, regs));
  machine_ir.AddEdge(testloop.preloop, testloop.successor);
  ASSERT_FALSE(CheckSuccessorNode(loop, testloop.successor, regs));
}

// regs should not be live_in to other loop nodes.
TEST(MachineIRReadFlagsOptimizer, CheckSuccessorNodeLiveIn) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();
  MachineRegVector regs({flags0}, machine_ir.arena());

  auto testloop = BuildBasicLoop(&machine_ir);

  testloop.loop_exit->live_in().push_back(flags0);
  testloop.loop_exit->insn_list().insert(testloop.loop_exit->insn_list().begin(),
                                         machine_ir.NewInsn<PseudoCopy>(flags1, flags0, 8));

  testloop.postloop->live_in().push_back(flags1);
  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto loop_tree = BuildLoopTree(&machine_ir);
  auto loop = loop_tree.root()->GetInnerloopNode(0)->loop();

  ASSERT_TRUE(CheckSuccessorNode(loop, testloop.loop_exit, regs));
  // Remove flags1.
  regs.pop_back();

  // Make sure we fail if flags0 is live_in of another loop node.
  testloop.successor->live_in().push_back(flags0);
  ASSERT_FALSE(CheckSuccessorNode(loop, testloop.loop_exit, regs));

  // Reset state.
  testloop.successor->live_in().pop_back();
  regs.pop_back();

  // Make sure that we check live_in after CheckRegsUnusedWithinInsnRange.
  testloop.successor->live_in().push_back(flags1);
  ASSERT_FALSE(CheckSuccessorNode(loop, testloop.loop_exit, regs));
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

TEST(MachineIRReadFlagsOptimizer, InsertFlagGenInstructionsAddsCmc) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto flags0 = machine_ir.AllocVReg();
  auto input0 = machine_ir.AllocVReg();
  auto input1 = machine_ir.AllocVReg();

  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);

  builder.StartBasicBlock(bb0);
  builder.Gen<AddqRegReg>(input0, input1, kMachineRegFLAGS);
  builder.Gen<Cmc>(kMachineRegFLAGS);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS);
  builder.Gen<PseudoBranch>(bb1);

  builder.StartBasicBlock(bb1);
  builder.Gen<MovqRegReg>(flags0, flags0);

  auto context = ReadFlagsOptContext{
      bb1, std::next(bb0->insn_list().begin(), 2), FlagSettingInsn{bb0->insn_list().begin(), true}};
  InsertFlagGenInstructions(
      &machine_ir,
      context,
      bb1->insn_list().begin(),
      ArenaMap<MachineReg, MachineReg>({{input0, input0}, {input1, input1}}, machine_ir.arena()),
      flags0);

  auto insn_it = bb1->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
  auto flags_reg = (*insn_it)->RegAt(2);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpCmc);
  ASSERT_EQ((*insn_it)->RegAt(0), flags_reg);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  ASSERT_EQ((*insn_it)->RegAt(1), flags_reg);
  insn_it++;
}

TEST(MachineIRReadFlagsOptimizer, InsertFlagGenInstructionsSavesFlagReg) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  auto testloop = BuildBasicLoop(&machine_ir);

  // register we save flags to in BuildBasicLoop.
  auto flags = (*std::next(testloop.loop_exit->insn_list().begin()))->RegAt(0);
  auto input0 = (*testloop.loop_exit->insn_list().begin())->RegAt(0);
  auto input1 = (*testloop.loop_exit->insn_list().begin())->RegAt(1);

  // Note the instructions are inserted in reverse order.
  testloop.postloop->insn_list().push_front(machine_ir.NewInsn<PseudoReadFlags>(
      PseudoReadFlags::kWithOverflow, machine_ir.AllocVReg(), kMachineRegFLAGS));
  testloop.postloop->insn_list().push_front(machine_ir.NewInsn<MovqRegReg>(flags, flags));

  auto context =
      ReadFlagsOptContext{testloop.postloop,
                          std::next(testloop.loop_exit->insn_list().begin()),
                          FlagSettingInsn{testloop.loop_exit->insn_list().begin(), false}};
  InsertFlagGenInstructions(
      &machine_ir,
      context,
      testloop.postloop->insn_list().begin(),
      ArenaMap<MachineReg, MachineReg>({{input0, input0}, {input1, input1}}, machine_ir.arena()),
      flags);

  // Check that read and write flags instructions inserted.
  ASSERT_EQ((*testloop.postloop->insn_list().begin())->opcode(), kMachineOpPseudoReadFlags);
  ASSERT_EQ((*std::next(testloop.postloop->insn_list().begin(), 4))->opcode(),
            kMachineOpPseudoWriteFlags);

  // Now test that we don't insert read/write flags when we don't need to.
  testloop.postloop->insn_list().clear();
  testloop.postloop->insn_list().push_front(machine_ir.NewInsn<PseudoJump>(kNullGuestAddr));
  testloop.postloop->insn_list().push_front(machine_ir.NewInsn<MovqRegReg>(flags, flags));

  InsertFlagGenInstructions(
      &machine_ir,
      context,
      testloop.postloop->insn_list().begin(),
      ArenaMap<MachineReg, MachineReg>({{input0, input0}, {input1, input1}}, machine_ir.arena()),
      flags);

  ASSERT_EQ((*testloop.postloop->insn_list().begin())->opcode(), kMachineOpPseudoCopy);
}

// Tests that IsEligibleReadFlags makes sure the flag register isn't used in the
// exit node.
TEST(MachineIRReadFlagsOptimizer, IsEligibleReadFlagChecksFlagsNotUsedInExitNode) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  auto testloop = BuildBasicLoop(&machine_ir);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);
  auto loop_tree = BuildLoopTree(&machine_ir);
  auto res = IsEligibleReadFlag(&machine_ir,
                                loop_tree.root()->GetInnerloopNode(0)->loop(),
                                testloop.loop_exit,
                                testloop.readflags_it);
  ASSERT_TRUE(res.has_value());

  testloop.loop_exit->insn_list().push_back(
      machine_ir.NewInsn<PseudoWriteFlags>(testloop.flags_reg, kMachineRegFLAGS));
  res = IsEligibleReadFlag(&machine_ir,
                           loop_tree.root()->GetInnerloopNode(0)->loop(),
                           testloop.loop_exit,
                           testloop.readflags_it);
  ASSERT_FALSE(res.has_value());
}

// Tests that IsEligibleReadFlags checks post loop node.
TEST(MachineIRReadFlagsOptimizer, IsEligibleReadFlagChecksPostloopNode) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  auto testloop = BuildBasicLoop(&machine_ir);
  MachineReg flags_copy = machine_ir.AllocVReg();

  testloop.postloop->live_in().push_back(testloop.flags_reg);
  testloop.postloop->insn_list().push_front(
      machine_ir.NewInsn<PseudoCopy>(flags_copy, testloop.flags_reg, 8));

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);
  auto loop_tree = BuildLoopTree(&machine_ir);
  auto res = IsEligibleReadFlag(&machine_ir,
                                loop_tree.root()->GetInnerloopNode(0)->loop(),
                                testloop.loop_exit,
                                testloop.readflags_it);
  ASSERT_TRUE(res.has_value());

  // Make postloop node fail by having the copy be live_out.
  testloop.postloop->live_out().push_back(testloop.flags_reg);
  res = IsEligibleReadFlag(&machine_ir,
                           loop_tree.root()->GetInnerloopNode(0)->loop(),
                           testloop.loop_exit,
                           testloop.readflags_it);
  ASSERT_FALSE(res.has_value());
}

// Tests that IsEligibleReadFlags checks loop successor node.
TEST(MachineIRReadFlagsOptimizer, IsEligibleReadFlagChecksSuccessorNode) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  auto testloop = BuildBasicLoop(&machine_ir);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);
  auto loop_tree = BuildLoopTree(&machine_ir);
  auto res = IsEligibleReadFlag(&machine_ir,
                                loop_tree.root()->GetInnerloopNode(0)->loop(),
                                testloop.loop_exit,
                                testloop.readflags_it);
  ASSERT_TRUE(res.has_value());

  // Make successor fail by accessing the register.
  testloop.successor->live_in().push_back(testloop.flags_reg);
  testloop.successor->insn_list().push_front(
      machine_ir.NewInsn<PseudoWriteFlags>(machine_ir.AllocVReg(), testloop.flags_reg));
  res = IsEligibleReadFlag(&machine_ir,
                           loop_tree.root()->GetInnerloopNode(0)->loop(),
                           testloop.loop_exit,
                           testloop.readflags_it);
  ASSERT_FALSE(res.has_value());
}

// Tests that IsEligibleReadFlags checks successor's postloop node.
TEST(MachineIRReadFlagsOptimizer, IsEligibleReadFlagChecksSuccPostLoopNode) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  auto testloop = BuildBasicLoop(&machine_ir);
  MachineReg flags_copy = machine_ir.AllocVReg();

  testloop.successor->live_in().push_back(testloop.flags_reg);
  testloop.successor->insn_list().push_front(
      machine_ir.NewInsn<PseudoCopy>(flags_copy, testloop.flags_reg, 8));
  testloop.successor->live_out().push_back(flags_copy);
  testloop.succ_postloop->live_in().push_back(flags_copy);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);
  auto loop_tree = BuildLoopTree(&machine_ir);
  auto res = IsEligibleReadFlag(&machine_ir,
                                loop_tree.root()->GetInnerloopNode(0)->loop(),
                                testloop.loop_exit,
                                testloop.readflags_it);
  ASSERT_TRUE(res.has_value());

  // succ_postloop should fail if it lets flags_copy be live_out.
  testloop.succ_postloop->live_out().push_back(flags_copy);
  res = IsEligibleReadFlag(&machine_ir,
                           loop_tree.root()->GetInnerloopNode(0)->loop(),
                           testloop.loop_exit,
                           testloop.readflags_it);
  ASSERT_FALSE(res.has_value());
}

// Tests that IsEligibleReadFlags returns the right instruction.
TEST(MachineIRReadFlagsOptimizer, IsEligibleReadFlagReturnsSetter) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  auto testloop = BuildBasicLoop(&machine_ir);
  testloop.loop_exit->insn_list().push_front(
      machine_ir.NewInsn<SubqRegImm>(machine_ir.AllocVReg(), 121, testloop.flags_reg));

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);
  auto loop_tree = BuildLoopTree(&machine_ir);

  auto insn_it = std::next(testloop.loop_exit->insn_list().begin(), 2);
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  auto res = IsEligibleReadFlag(
      &machine_ir, loop_tree.root()->GetInnerloopNode(0)->loop(), testloop.loop_exit, insn_it);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ((*res.value().insn)->opcode(), kMachineOpAddqRegReg);
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
  ASSERT_EQ((*flag_setter.value().insn)->opcode(), kMachineOpSubqRegImm);

  // Test that we exit properly when we can't find the instruction.
  // Move to second AddqRegReg.
  insn_it--;
  flag_setter = FindFlagSettingInsn(insn_it, bb->insn_list().begin(), flags1);
  ASSERT_FALSE(flag_setter.has_value());
}

TEST(MachineIRReadFlagsOptimizer, FindFlagSettingInsnSetsCmc) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  auto bb = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb);
  builder.Gen<AddqRegReg>(machine_ir.AllocVReg(), machine_ir.AllocVReg(), kMachineRegFLAGS);
  builder.Gen<Cmc>(kMachineRegFLAGS);
  builder.Gen<PseudoReadFlags>(
      PseudoReadFlags::kWithOverflow, machine_ir.AllocVReg(), kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  auto flag_setter = FindFlagSettingInsn(
      std::next(bb->insn_list().begin(), 2), bb->insn_list().begin(), kMachineRegFLAGS);
  ASSERT_TRUE(flag_setter.has_value());
  ASSERT_TRUE(flag_setter.value().cmc);
}

TEST(MachineIRReadFlagsOptimizer, NeedsToSaveFlags) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  // Not used so shouldn't need to save.
  auto bb0 = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb0);
  builder.Gen<MovqRegReg>(machine_ir.AllocVReg(), machine_ir.AllocVReg());
  builder.Gen<PseudoJump>(kNullGuestAddr);
  ASSERT_FALSE(NeedsToSaveFlags(bb0, bb0->insn_list().begin()).has_value());

  // Flags are read so should be saved.
  auto bb1 = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb1);
  builder.Gen<MovqRegReg>(machine_ir.AllocVReg(), machine_ir.AllocVReg());
  builder.Gen<PseudoReadFlags>(
      PseudoReadFlags::kWithOverflow, machine_ir.AllocVReg(), kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  ASSERT_EQ(NeedsToSaveFlags(bb1, bb1->insn_list().begin()).value(), kMachineRegFLAGS);

  // Flags used but clobbered beforehand so shouldn't need to be saved.
  auto bb2 = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb2);
  builder.Gen<MovqRegReg>(machine_ir.AllocVReg(), machine_ir.AllocVReg());
  builder.Gen<SubqRegImm>(machine_ir.AllocVReg(), 1, kMachineRegFLAGS);
  builder.Gen<PseudoReadFlags>(
      PseudoReadFlags::kWithOverflow, machine_ir.AllocVReg(), kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  ASSERT_FALSE(NeedsToSaveFlags(bb2, bb2->insn_list().begin()).has_value());
}

TEST(MachineIRReadFlagsOptimizer, RemoveEligibleReadFlagsInLoopTree) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg scratch = machine_ir.AllocVReg();
  // flags0 used to test whether we remove from outer loops.
  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags00 = machine_ir.AllocVReg();
  // flags1 used to test whether we remove from inner loop.
  MachineReg flags1 = machine_ir.AllocVReg();
  MachineReg flags11 = machine_ir.AllocVReg();

  //         |-------|
  // bb0 -> bb1 --> bb2 <-> bb3 -> bb4
  //         |
  //        bb5
  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  auto bb2 = machine_ir.NewBasicBlock();
  auto bb3 = machine_ir.NewBasicBlock();
  auto bb4 = machine_ir.NewBasicBlock();
  auto bb5 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);
  machine_ir.AddEdge(bb1, bb2);
  machine_ir.AddEdge(bb1, bb5);
  machine_ir.AddEdge(bb2, bb1);
  machine_ir.AddEdge(bb2, bb3);
  machine_ir.AddEdge(bb3, bb2);
  machine_ir.AddEdge(bb3, bb4);

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoBranch>(bb1);

  builder.StartBasicBlock(bb1);
  builder.Gen<x86_64::AddqRegReg>(scratch, scratch, kMachineRegFLAGS);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(flags00, flags0, 8);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb2, bb5, kMachineRegFLAGS);
  bb1->live_out().push_back(flags00);

  builder.StartBasicBlock(bb2);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb1, bb3, kMachineRegFLAGS);

  builder.StartBasicBlock(bb3);
  builder.Gen<x86_64::AddqRegReg>(scratch, scratch, kMachineRegFLAGS);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags1, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(flags11, flags1, 8);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb2, bb4, kMachineRegFLAGS);
  bb3->live_out().push_back(flags11);

  builder.StartBasicBlock(bb4);
  builder.Gen<x86_64::AddqRegReg>(machine_ir.AllocVReg(), flags11, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  bb4->live_in().push_back(flags11);

  builder.StartBasicBlock(bb5);
  builder.Gen<x86_64::AddqRegReg>(machine_ir.AllocVReg(), flags00, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  bb5->live_in().push_back(flags00);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);
  auto loop_tree = BuildLoopTree(&machine_ir);
  RemoveEligibleReadFlagsInLoopTree(&machine_ir, loop_tree.root());

  // flags0 should be removed if we correctly optimize outer loops.
  auto insn_it = std::next(bb1->insn_list().begin());
  ASSERT_NE((*insn_it)->opcode(), kMachineOpPseudoReadFlags);

  // pseudoreadflags flags1 should be removed.
  insn_it = std::next(bb3->insn_list().begin());
  ASSERT_NE((*insn_it)->opcode(), kMachineOpPseudoReadFlags);

  // Check that bb4 and bb5 have the correct instructions added.
  insn_it = bb4->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);

  insn_it = bb5->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
}

TEST(MachineIRReadFlagsOptimizer, RemoveEligibleReadFlagsExitsToOuterLoop) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg scratch = machine_ir.AllocVReg();
  // flags0 used to test whether we remove from outer loops.
  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags00 = machine_ir.AllocVReg();

  //         |-------------|
  // bb0 -> bb1 -> bb2 -> bb3 -> bb4
  //                ^--|
  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  auto bb2 = machine_ir.NewBasicBlock();
  auto bb3 = machine_ir.NewBasicBlock();
  auto bb4 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);
  machine_ir.AddEdge(bb1, bb2);
  machine_ir.AddEdge(bb2, bb2);
  machine_ir.AddEdge(bb2, bb3);
  machine_ir.AddEdge(bb3, bb1);
  machine_ir.AddEdge(bb3, bb4);

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoBranch>(bb1);

  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoBranch>(bb2);

  builder.StartBasicBlock(bb2);
  builder.Gen<x86_64::SubqRegReg>(scratch, scratch, kMachineRegFLAGS);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(flags00, flags0, 8);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb2, bb3, kMachineRegFLAGS);
  bb2->live_out().push_back(flags00);

  bb3->live_in().push_back(flags00);
  builder.StartBasicBlock(bb3);
  builder.Gen<x86_64::MovqRegReg>(machine_ir.AllocVReg(), flags00);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb1, bb4, machine_ir.AllocVReg());

  builder.StartBasicBlock(bb4);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);
  auto loop_tree = BuildLoopTree(&machine_ir);
  RemoveEligibleReadFlagsInLoopTree(&machine_ir, loop_tree.root());

  // flags0 should be removed if we correctly optimize outer loops.
  auto insn_it = std::next(bb2->insn_list().begin());
  ASSERT_NE((*insn_it)->opcode(), kMachineOpPseudoReadFlags);

  // Check that bb3 has instructions added.
  insn_it = bb3->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpSubqRegReg);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoWriteFlags);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpMovqRegReg);
}

TEST(MachineIRReadFlagsOptimizer, OptimizeReadFlags) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  auto testloop = BuildBasicLoop(&machine_ir);

  MachineReg flags_copy = machine_ir.AllocVReg();

  testloop.postloop->live_in().push_back(testloop.flags_reg);
  testloop.postloop->insn_list().push_front(
      machine_ir.NewInsn<MovqRegReg>(machine_ir.AllocVReg(), testloop.flags_reg));

  testloop.successor->live_in().push_back(testloop.flags_reg);
  testloop.successor->insn_list().push_front(
      machine_ir.NewInsn<PseudoCopy>(flags_copy, testloop.flags_reg, 8));
  testloop.successor->live_out().push_back(flags_copy);

  testloop.succ_postloop->live_in().push_back(flags_copy);
  testloop.succ_postloop->insn_list().push_front(
      machine_ir.NewInsn<MovqRegReg>(machine_ir.AllocVReg(), flags_copy));

  OptimizeReadFlags(&machine_ir);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  // Check that original PSEUDOREADFLAGS instruction is gone.
  ASSERT_TRUE(
      std::none_of(testloop.loop_exit->insn_list().begin(),
                   testloop.loop_exit->insn_list().end(),
                   [](MachineInsn* insn) { return insn->opcode() == kMachineOpPseudoReadFlags; }));

  // Check that postloop inserted the original instruction.
  auto insn_it = testloop.postloop->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);

  // Check that successor removes pseudocopy.
  insn_it = testloop.successor->insn_list().begin();
  ASSERT_NE((*insn_it)->opcode(), kMachineOpPseudoReadFlags);

  // Check that succ_postloop also has original instruction.
  insn_it = testloop.succ_postloop->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
}

TEST(MachineIRReadFlagsOptimizer, RemoveRegs) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();
  MachineReg flags2 = machine_ir.AllocVReg();
  MachineReg flags3 = machine_ir.AllocVReg();
  MachineReg flags4 = machine_ir.AllocVReg();

  MachineRegVector disallowed({flags0, flags1, flags3}, machine_ir.arena());
  MachineRegVector regs({flags0, flags1, flags2, flags3, flags4}, machine_ir.arena());

  ASSERT_TRUE(RemoveRegs(regs, disallowed));
  ASSERT_EQ(regs.size(), 2UL);
  ASSERT_TRUE(Contains(regs, flags2));
  ASSERT_TRUE(Contains(regs, flags4));
  ASSERT_FALSE(RemoveRegs(regs, disallowed));
}

TEST(MachineIRReadFlagsOptimizer, RemoveReadFlags) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags00 = machine_ir.AllocVReg();
  MachineReg flags000 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();
  MachineReg input_flag0 = machine_ir.AllocVReg();
  MachineReg input_flag1 = machine_ir.AllocVReg();

  // bb0 --> bb1 --> bb2 --> bb3
  //          ^       |
  //          |       v
  //          -------bb4 --> bb5
  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  auto bb2 = machine_ir.NewBasicBlock();
  auto bb3 = machine_ir.NewBasicBlock();
  auto bb4 = machine_ir.NewBasicBlock();
  auto bb5 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);
  machine_ir.AddEdge(bb1, bb2);
  machine_ir.AddEdge(bb2, bb3);
  machine_ir.AddEdge(bb2, bb4);
  machine_ir.AddEdge(bb4, bb1);
  machine_ir.AddEdge(bb4, bb5);

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoBranch>(bb1);

  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoBranch>(bb2);

  builder.StartBasicBlock(bb2);
  builder.Gen<AddqRegReg>(input_flag0, input_flag1, kMachineRegFLAGS);
  auto* readflag_insn =
      builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(flags00, flags0, 8);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb3, bb4, kMachineRegFLAGS);
  bb2->live_out().push_back(flags00);

  builder.StartBasicBlock(bb3);
  builder.Gen<MovqRegReg>(flags1, flags00);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  bb3->live_in().push_back(flags00);

  builder.StartBasicBlock(bb4);
  builder.Gen<PseudoCopy>(flags000, flags00, 8);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb1, bb5, kMachineRegFLAGS);
  bb4->live_in().push_back(flags00);
  bb4->live_out().push_back(flags000);

  builder.StartBasicBlock(bb5);
  builder.Gen<MovqRegReg>(flags1, flags000);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  bb5->live_in().push_back(flags000);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  RemoveReadFlags(&machine_ir,
                  ReadFlagsOptContext{
                      bb2,
                      std::find(bb2->insn_list().begin(), bb2->insn_list().end(), readflag_insn),
                      FlagSettingInsn{bb2->insn_list().begin(), false}});

  // Check ReadFlags gone.
  ASSERT_TRUE(std::none_of(bb2->insn_list().begin(), bb2->insn_list().end(), [](MachineInsn* insn) {
    return insn->opcode() == kMachineOpPseudoReadFlags;
  }));
  // Check that we created copies of input flags.
  auto insn_it = bb2->insn_list().begin();
  ASSERT_TRUE((*insn_it)->opcode() == kMachineOpPseudoCopy && (*insn_it)->RegAt(1) == input_flag0);
  insn_it++;
  ASSERT_TRUE((*insn_it)->opcode() == kMachineOpPseudoCopy && (*insn_it)->RegAt(1) == input_flag1);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);

  // Check live_in/live_out.
  ASSERT_EQ(bb2->live_out().size(), 2UL);
  ASSERT_EQ(std::find(bb2->live_out().begin(), bb2->live_out().end(), flags0),
            bb2->live_out().end());
  ASSERT_EQ(bb3->live_in().size(), 2UL);
  ASSERT_EQ(bb4->live_in().size(), 2UL);
  ASSERT_EQ(bb4->live_out().size(), 2UL);
  ASSERT_EQ(bb5->live_in().size(), 2UL);
  ASSERT_EQ(bb1->live_in().size(), 0UL);

  // Check that we create the instruction to set flags.
  insn_it = bb3->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  ASSERT_TRUE(Contains(bb3->live_in(), (*insn_it)->RegAt(1)));
  auto input_copy = (*insn_it)->RegAt(0);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
  ASSERT_FALSE(Contains(bb3->live_in(), (*insn_it)->RegAt(0)));
  ASSERT_EQ(input_copy, (*insn_it)->RegAt(0));
  ASSERT_TRUE(Contains(bb3->live_in(), (*insn_it)->RegAt(1)));
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  ASSERT_EQ((*insn_it)->RegAt(0), flags00);

  insn_it = bb5->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  ASSERT_TRUE(Contains(bb5->live_in(), (*insn_it)->RegAt(1)));
  input_copy = (*insn_it)->RegAt(0);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
  ASSERT_FALSE(Contains(bb5->live_in(), (*insn_it)->RegAt(0)));
  ASSERT_EQ(input_copy, (*insn_it)->RegAt(0));
  ASSERT_TRUE(Contains(bb5->live_in(), (*insn_it)->RegAt(1)));
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  ASSERT_EQ((*insn_it)->RegAt(0), flags000);
}

TEST(MachineIRReadFlagsOptimizer, ReplaceFlagRegistersRecursesOnNeighbors) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg input0 = machine_ir.AllocVReg();
  MachineReg input00 = machine_ir.AllocVReg();

  // bb0 <-> bb2
  //  |-> bb1
  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  auto bb2 = machine_ir.NewBasicBlock();
  auto bb3 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);
  machine_ir.AddEdge(bb0, bb2);
  machine_ir.AddEdge(bb2, bb0);

  builder.StartBasicBlock(bb0);
  auto* flag_set_insn = builder.Gen<SubqRegImm>(input0, 12, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(input00, input0, 8);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, bb1, bb2, kMachineRegFLAGS);

  bb1->live_in().push_back(flags0);
  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoWriteFlags>(flags0, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  bb2->live_in().push_back(flags0);
  builder.StartBasicBlock(bb2);
  builder.Gen<PseudoWriteFlags>(flags0, kMachineRegFLAGS);
  builder.Gen<PseudoBranch>(bb0);

  ReplaceFlagRegisters(
      &machine_ir,
      ReadFlagsOptContext{
          bb0,
          MachineInsnList{{machine_ir.NewInsn<PseudoReadFlags>(
                              PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS)},
                          machine_ir.arena()}
              .begin(),
          FlagSettingInsn{bb0->insn_list().begin(), false},
      },
      bb0->insn_list().begin(),
      MachineRegVector({flags0}, machine_ir.arena()),
      ArenaMap<MachineReg, MachineReg>({{input0, input00}}, machine_ir.arena()),
      flag_set_insn);

  // Make sure that ReplaceFlagRegisters modifies bb1 and bb2.
  ASSERT_EQ((*std::next(bb1->insn_list().begin()))->opcode(), kMachineOpSubqRegImm);
  ASSERT_EQ((*std::next(bb2->insn_list().begin()))->opcode(), kMachineOpSubqRegImm);
}

TEST(MachineIRReadFlagsOptimizer, ReplaceFlagRegistersReplacesInstructions) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags00 = machine_ir.AllocVReg();
  MachineReg input0 = machine_ir.AllocVReg();
  MachineReg input00 = machine_ir.AllocVReg();
  auto bb0 = machine_ir.NewBasicBlock();
  auto bb1 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);

  builder.StartBasicBlock(bb0);
  auto* flag_set_insn = builder.Gen<SubqRegImm>(input0, 12, kMachineRegFLAGS);
  builder.Gen<PseudoCopy>(input00, input0, 8);
  builder.Gen<PseudoBranch>(bb1);

  bb1->live_in().push_back(flags0);
  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoCopy>(flags00, flags0, 8);
  builder.Gen<PseudoWriteFlags>(flags00, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ReplaceFlagRegisters(
      &machine_ir,
      ReadFlagsOptContext{
          bb0,
          MachineInsnList{{machine_ir.NewInsn<PseudoReadFlags>(
                              PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS)},
                          machine_ir.arena()}
              .begin(),
          FlagSettingInsn{bb0->insn_list().begin(), false},
      },
      bb0->insn_list().begin(),
      MachineRegVector({flags0}, machine_ir.arena()),
      ArenaMap<MachineReg, MachineReg>({{input0, input00}}, machine_ir.arena()),
      flag_set_insn);

  auto insns = bb1->insn_list().begin();
  ASSERT_EQ((*insns)->opcode(), kMachineOpPseudoCopy);
  ASSERT_EQ((*insns)->RegAt(1), input00);
  insns++;
  auto input000 = (*insns)->RegAt(0);
  ASSERT_EQ((*insns)->opcode(), kMachineOpSubqRegImm);
  ASSERT_EQ((*insns)->RegAt(0), input000);
  auto sub_flag_reg = (*insns)->RegAt(1);
  insns++;
  ASSERT_EQ((*insns)->opcode(), kMachineOpPseudoReadFlags);
  ASSERT_EQ((*insns)->RegAt(0).reg(), flags00.reg());
  ASSERT_EQ((*insns)->RegAt(1), sub_flag_reg);
}

TEST(MachineIRReadFlagsOptimizer, ReplaceFlagRegistersUpdatesLiveInOut) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags00 = machine_ir.AllocVReg();
  MachineReg input0 = machine_ir.AllocVReg();
  MachineReg input00 = machine_ir.AllocVReg();
  MachineReg input1 = machine_ir.AllocVReg();
  MachineReg input11 = machine_ir.AllocVReg();

  auto bb0 = machine_ir.NewBasicBlock();

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoCopy>(flags00, flags0, 8);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  bb0->live_in().push_back(flags0);
  bb0->live_out().push_back(flags00);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ReplaceFlagRegisters(
      &machine_ir,
      ReadFlagsOptContext{
          bb0,
          MachineInsnList{{machine_ir.NewInsn<PseudoReadFlags>(
                              PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS)},
                          machine_ir.arena()}
              .begin(),
          FlagSettingInsn{
              MachineInsnList{{machine_ir.NewInsn<AddqRegReg>(input0, input1, kMachineRegFLAGS)},
                              machine_ir.arena()}
                  .begin(),
              false},
      },
      bb0->insn_list().begin(),
      MachineRegVector({flags0}, machine_ir.arena()),
      ArenaMap<MachineReg, MachineReg>({{input0, input00}, {input1, input11}}, machine_ir.arena()),
      nullptr);

  ASSERT_EQ(bb0->live_in().size(), 2UL);
  ASSERT_TRUE(Contains(bb0->live_in(), input00));
  ASSERT_TRUE(Contains(bb0->live_in(), input11));
}

TEST(MachineIRReadFlagsOptimizer, ReplaceFlagRegistersDeletesCopies) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg flags00 = machine_ir.AllocVReg();
  MachineReg flags000 = machine_ir.AllocVReg();
  MachineReg flags1 = machine_ir.AllocVReg();

  auto bb0 = machine_ir.NewBasicBlock();

  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoCopy>(flags00, flags0, 8);
  builder.Gen<PseudoCopy>(flags000, flags00, 8);
  builder.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, flags1, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ReplaceFlagRegisters(
      &machine_ir,
      ReadFlagsOptContext{
          bb0,
          MachineInsnList{{machine_ir.NewInsn<PseudoReadFlags>(
                              PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS)},
                          machine_ir.arena()}
              .begin(),
          FlagSettingInsn{
              MachineInsnList{{machine_ir.NewInsn<AddqRegReg>(flags0, flags0, kMachineRegFLAGS)},
                              machine_ir.arena()}
                  .begin(),
              false},
      },
      bb0->insn_list().begin(),
      MachineRegVector({flags0}, machine_ir.arena()),
      ArenaMap<MachineReg, MachineReg>(machine_ir.arena()),
      nullptr);
  ASSERT_TRUE(std::none_of(bb0->insn_list().begin(), bb0->insn_list().end(), [](MachineInsn* insn) {
    return insn->opcode() == kMachineOpPseudoCopy;
  }));
  ASSERT_TRUE(std::any_of(bb0->insn_list().begin(), bb0->insn_list().end(), [](MachineInsn* insn) {
    return insn->opcode() == kMachineOpPseudoReadFlags;
  }));
}

// Make sure we make copies of any registers which are written to.
TEST(MachineIRReadFlagsOptimizer, ReplaceFlagRegistersCopiesDefRegisters) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg input0 = machine_ir.AllocVReg();
  MachineReg input00 = machine_ir.AllocVReg();
  MachineReg input1 = machine_ir.AllocVReg();
  MachineReg input11 = machine_ir.AllocVReg();

  auto bb0 = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoWriteFlags>(flags0, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ReplaceFlagRegisters(
      &machine_ir,
      ReadFlagsOptContext{
          bb0,
          MachineInsnList{{machine_ir.NewInsn<PseudoReadFlags>(
                              PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS)},
                          machine_ir.arena()}
              .begin(),
          FlagSettingInsn{
              MachineInsnList{{machine_ir.NewInsn<AddqRegReg>(input0, input1, kMachineRegFLAGS)},
                              machine_ir.arena()}
                  .begin(),
              false},
      },
      bb0->insn_list().begin(),
      MachineRegVector({flags0}, machine_ir.arena()),
      ArenaMap<MachineReg, MachineReg>(
          {
              {input0, input00},
              {input1, input11},
          },
          machine_ir.arena()),
      nullptr);

  auto insns = bb0->insn_list().begin();
  ASSERT_EQ((*insns)->opcode(), kMachineOpPseudoCopy);
  ASSERT_EQ((*insns)->RegAt(1), input00);
  auto input000 = (*insns)->RegAt(0);
  insns++;
  ASSERT_EQ((*insns)->opcode(), kMachineOpAddqRegReg);
  ASSERT_EQ((*insns)->RegAt(0), input000);
  ASSERT_NE((*insns)->RegAt(1), input1);
}

// Test that ReplaceFlagRegisters won't insert instructions multiple times for
// the same register.
TEST(MachineIRReadFlagsOptimizer, ReplaceFlagRegistersWithDuplicates) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  x86_64::MachineIRBuilder builder(&machine_ir);

  MachineReg flags0 = machine_ir.AllocVReg();
  MachineReg input0 = machine_ir.AllocVReg();

  auto bb0 = machine_ir.NewBasicBlock();
  builder.StartBasicBlock(bb0);
  builder.Gen<SubqRegReg>(flags0, flags0, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  ASSERT_EQ(x86_64::CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  ReplaceFlagRegisters(
      &machine_ir,
      ReadFlagsOptContext{
          bb0,
          MachineInsnList{{machine_ir.NewInsn<PseudoReadFlags>(
                              PseudoReadFlags::kWithOverflow, flags0, kMachineRegFLAGS)},
                          machine_ir.arena()}
              .begin(),
          FlagSettingInsn{
              MachineInsnList{{machine_ir.NewInsn<AddqRegReg>(input0, input0, kMachineRegFLAGS)},
                              machine_ir.arena()}
                  .begin(),
              false},
      },
      bb0->insn_list().begin(),
      MachineRegVector({flags0}, machine_ir.arena()),
      ArenaMap<MachineReg, MachineReg>(
          {
              {input0, input0},
          },
          machine_ir.arena()),
      nullptr);

  auto insn_it = bb0->insn_list().begin();
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoCopy);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpAddqRegReg);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoReadFlags);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpSubqRegReg);
  insn_it++;
  ASSERT_EQ((*insn_it)->opcode(), kMachineOpPseudoJump);
}

}  // namespace

}  // namespace berberis::x86_64
