#include "gtest/gtest.h"

#include <cstdint>
#include <iterator>
#include <limits>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/backend/x86_64/use_lea_for_copied_adds.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis::x86_64 {

namespace {

class MachineIRUseLeaForCopiedAddsTest : public ::testing::Test {
 protected:
  MachineIRUseLeaForCopiedAddsTest() : machine_ir_(&arena_), builder_(&machine_ir_) {
    bb_ = machine_ir_.NewBasicBlock();
    builder_.StartBasicBlock(bb_);
    copied_ = machine_ir_.AllocVReg();
    dst_ = machine_ir_.AllocVReg();
    other_ = machine_ir_.AllocVReg();
    flags_ = machine_ir_.AllocVReg();
    builder_.Gen<MovqRegImm>(copied_, 0x1000);
    builder_.Gen<MovqRegImm>(other_, 0x20);
  }

  // Ends the block with reads of copied_ and dst_.
  void Finish() {
    MachineReg sink = machine_ir_.AllocVReg();
    builder_.Gen<MovqRegReg>(sink, copied_);
    builder_.Gen<MovqRegReg>(sink, dst_);
    builder_.Gen<PseudoJump>(kNullGuestAddr);
    ASSERT_EQ(CheckMachineIR(machine_ir_), kMachineIRCheckSuccess);
  }

  const berberis::MachineInsn* InsnAt(size_t index) const {
    auto it = bb_->insn_list().begin();
    std::advance(it, index);
    return *it;
  }

  void ExpectUnchanged(size_t size, size_t op_index, MachineOpcode opcode) {
    EXPECT_EQ(bb_->insn_list().size(), size);
    EXPECT_EQ(InsnAt(op_index)->opcode(), opcode);
  }

  Arena arena_;
  MachineIR machine_ir_;
  MachineIRBuilder builder_;
  MachineBasicBlock* bb_;
  MachineReg copied_;
  MachineReg dst_;
  MachineReg other_;
  MachineReg flags_;
};

TEST_F(MachineIRUseLeaForCopiedAddsTest, MergesCopyIntoAddImmediate) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  EXPECT_EQ(CheckMachineIR(machine_ir_), kMachineIRCheckSuccess);
  ASSERT_EQ(bb_->insn_list().size(), 6u);
  const berberis::MachineInsn* lea = InsnAt(2);
  ASSERT_EQ(lea->opcode(), kMachineOpLeaqRegMemBaseDisp);
  EXPECT_EQ(lea->RegAt(0), dst_);
  EXPECT_EQ(lea->RegAt(1), copied_);
  EXPECT_EQ(AsMachineInsnX86_64(lea)->disp(), 16u);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, MergesCopyIntoSubImmediate) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<SubqRegImm>(dst_, 8, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ASSERT_EQ(bb_->insn_list().size(), 6u);
  const berberis::MachineInsn* lea = InsnAt(2);
  ASSERT_EQ(lea->opcode(), kMachineOpLeaqRegMemBaseDisp);
  EXPECT_EQ(lea->RegAt(1), copied_);
  EXPECT_EQ(AsMachineInsnX86_64(lea)->disp(), static_cast<uint32_t>(-8));
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, MergesCopyIntoAddRegister) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddqRegReg>(dst_, other_, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  EXPECT_EQ(CheckMachineIR(machine_ir_), kMachineIRCheckSuccess);
  ASSERT_EQ(bb_->insn_list().size(), 6u);
  const berberis::MachineInsn* lea = InsnAt(2);
  ASSERT_EQ(lea->opcode(), kMachineOpLeaqRegMemBaseIndexDisp);
  EXPECT_EQ(lea->RegAt(0), dst_);
  EXPECT_EQ(lea->RegAt(1), copied_);
  EXPECT_EQ(lea->RegAt(2), other_);
  EXPECT_EQ(AsMachineInsnX86_64(lea)->scale(), Assembler::kTimesOne);
  EXPECT_EQ(AsMachineInsnX86_64(lea)->disp(), 0u);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, AddOfDestinationToItselfDoublesSource) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddqRegReg>(dst_, dst_, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ASSERT_EQ(bb_->insn_list().size(), 6u);
  const berberis::MachineInsn* lea = InsnAt(2);
  ASSERT_EQ(lea->opcode(), kMachineOpLeaqRegMemBaseIndexDisp);
  EXPECT_EQ(lea->RegAt(1), copied_);
  EXPECT_EQ(lea->RegAt(2), copied_);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, MergesWhenFlagsAreOverwrittenBeforeUse) {
  MachineReg word = machine_ir_.AllocVReg();
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  builder_.Gen<CmpqRegReg>(copied_, other_, flags_);
  builder_.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, word, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  EXPECT_EQ(InsnAt(2)->opcode(), kMachineOpLeaqRegMemBaseDisp);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsAddWhoseFlagsAreRead) {
  MachineReg word = machine_ir_.AllocVReg();
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  builder_.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, word, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(8u, 3, kMachineOpAddqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsAddWhoseCarryIsRead) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  builder_.Gen<AdcqRegImm>(other_, 0, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(8u, 3, kMachineOpAddqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsAddWhoseFlagsLiveOut) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  Finish();
  bb_->live_out().push_back(flags_);

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(7u, 3, kMachineOpAddqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsAddWritingHardFlagsAtBlockEnd) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddqRegImm>(dst_, 16, kMachineRegFLAGS);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(7u, 3, kMachineOpAddqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsCopyWhenSourceRedefined) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<MovqRegImm>(copied_, 0);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(8u, 4, kMachineOpAddqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsCopyWhenDestinationReadFirst) {
  MachineReg sink = machine_ir_.AllocVReg();
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<MovqRegReg>(sink, dst_);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(8u, 4, kMachineOpAddqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsAddWithoutCopy) {
  builder_.Gen<MovqRegImm>(dst_, 1);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(7u, 3, kMachineOpAddqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsCopyOfVectorRegister) {
  MachineReg vector = machine_ir_.AllocVReg();
  builder_.Gen<PseudoDefXReg>(vector);
  builder_.Gen<PseudoCopy>(dst_, vector, 8);
  builder_.Gen<AddqRegImm>(dst_, 16, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(8u, 4, kMachineOpAddqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsSubOfMostNegativeImmediate) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<SubqRegImm>(dst_, std::numeric_limits<int32_t>::min(), flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(7u, 3, kMachineOpSubqRegImm);
}

TEST_F(MachineIRUseLeaForCopiedAddsTest, KeepsThirtyTwoBitAdd) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 8);
  builder_.Gen<AddlRegImm>(dst_, 16, flags_);
  Finish();

  UseLeaForCopiedAdds(&machine_ir_);

  ExpectUnchanged(7u, 3, kMachineOpAddlRegImm);
}

}  // namespace

}  // namespace berberis::x86_64
