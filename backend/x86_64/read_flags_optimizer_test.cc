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

#include "berberis/backend/x86_64/read_flags_optimizer.h"

#include "berberis/backend/common/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/base/arena_vector.h"

namespace berberis::x86_64 {

namespace {

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

}  // namespace

}  // namespace berberis::x86_64
