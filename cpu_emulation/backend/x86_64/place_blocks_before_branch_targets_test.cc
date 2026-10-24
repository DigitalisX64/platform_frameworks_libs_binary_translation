#include "gtest/gtest.h"

#include <vector>

#include "berberis/backend/code_emitter.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/backend/x86_64/place_blocks_before_branch_targets.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis::x86_64 {

namespace {

std::vector<const MachineBasicBlock*> Layout(const MachineIR& machine_ir) {
  return {machine_ir.bb_list().begin(), machine_ir.bb_list().end()};
}

// bb0 enters the loop at its header bb3, which the builder laid out after the
// exit block; the latch bb1 branches back to bb3 or leaves through bb2.
TEST(MachineIRPlaceBlocksBeforeBranchTargetsTest, MovesLoopHeaderBeforeItsBody) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb0 = machine_ir.NewBasicBlock();
  auto* bb1 = machine_ir.NewBasicBlock();
  auto* bb2 = machine_ir.NewBasicBlock();
  auto* bb3 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb3);
  machine_ir.AddEdge(bb1, bb3);
  machine_ir.AddEdge(bb1, bb2);
  machine_ir.AddEdge(bb3, bb1);

  MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoBranch>(bb3);
  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoCondBranch>(
      CodeEmitter::Condition::kZero, bb3, bb2, kMachineRegFLAGS);
  builder.StartBasicBlock(bb2);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  builder.StartBasicBlock(bb3);
  builder.Gen<PseudoBranch>(bb1);
  ASSERT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);

  PlaceBlocksBeforeBranchTargets(&machine_ir);

  EXPECT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);
  EXPECT_EQ(Layout(machine_ir), (std::vector<const MachineBasicBlock*>{bb0, bb3, bb1, bb2}));
}

// Moving bb1 before bb3 would let bb0 fall through to bb2 instead, and bb2
// would stop falling through to bb3: no jump is saved, so nothing moves.
TEST(MachineIRPlaceBlocksBeforeBranchTargetsTest, KeepsLayoutWhenNoJumpIsSaved) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb0 = machine_ir.NewBasicBlock();
  auto* bb1 = machine_ir.NewBasicBlock();
  auto* bb2 = machine_ir.NewBasicBlock();
  auto* bb3 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb1);
  machine_ir.AddEdge(bb0, bb2);
  machine_ir.AddEdge(bb1, bb3);
  machine_ir.AddEdge(bb2, bb3);

  MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoCondBranch>(
      CodeEmitter::Condition::kZero, bb1, bb2, kMachineRegFLAGS);
  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoBranch>(bb3);
  builder.StartBasicBlock(bb2);
  builder.Gen<PseudoBranch>(bb3);
  builder.StartBasicBlock(bb3);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  ASSERT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);

  PlaceBlocksBeforeBranchTargets(&machine_ir);

  EXPECT_EQ(Layout(machine_ir), (std::vector<const MachineBasicBlock*>{bb0, bb1, bb2, bb3}));
}

// The entry block stays first even though it jumps over bb1, and a recovery
// block that branches back is not pulled out of the cold tail.
TEST(MachineIRPlaceBlocksBeforeBranchTargetsTest, KeepsEntryAndRecoveryBlocksInPlace) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb0 = machine_ir.NewBasicBlock();
  auto* bb1 = machine_ir.NewBasicBlock();
  auto* bb2 = machine_ir.NewBasicBlock();
  auto* bb3 = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(bb0, bb2);
  machine_ir.AddEdge(bb1, bb2);
  machine_ir.AddEdge(bb3, bb2);

  MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(bb0);
  builder.Gen<PseudoBranch>(bb2);
  builder.StartBasicBlock(bb1);
  builder.Gen<PseudoBranch>(bb2);
  builder.StartBasicBlock(bb2);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  builder.StartBasicBlock(bb3);
  builder.Gen<PseudoBranch>(bb2);
  bb3->MarkAsRecovery();

  PlaceBlocksBeforeBranchTargets(&machine_ir);

  EXPECT_EQ(Layout(machine_ir), (std::vector<const MachineBasicBlock*>{bb0, bb1, bb2, bb3}));
}

}  // namespace

}  // namespace berberis::x86_64
