#include "gtest/gtest.h"

#include <cstdint>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/backend/x86_64/use_vex_three_operand_forms.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis::x86_64 {

namespace {

class MachineIRUseVexThreeOperandFormsTest : public ::testing::Test {
 protected:
  MachineIRUseVexThreeOperandFormsTest() : machine_ir_(&arena_), builder_(&machine_ir_) {
    bb_ = machine_ir_.NewBasicBlock();
    builder_.StartBasicBlock(bb_);
    copied_ = machine_ir_.AllocVReg();
    dst_ = machine_ir_.AllocVReg();
    other_ = machine_ir_.AllocVReg();
    builder_.Gen<PseudoDefXReg>(copied_);
    builder_.Gen<PseudoDefXReg>(other_);
  }

  // Ends the block with reads of copied_ and dst_.
  void Finish() {
    MachineReg sink = machine_ir_.AllocVReg();
    builder_.Gen<MovdqaXRegXReg>(sink, copied_);
    builder_.Gen<MovdqaXRegXReg>(sink, dst_);
    builder_.Gen<PseudoJump>(kNullGuestAddr);
    ASSERT_EQ(CheckMachineIR(machine_ir_), kMachineIRCheckSuccess);
  }

  const berberis::MachineInsn* InsnAt(size_t index) const {
    auto it = bb_->insn_list().begin();
    std::advance(it, index);
    return *it;
  }

  Arena arena_;
  MachineIR machine_ir_;
  MachineIRBuilder builder_;
  MachineBasicBlock* bb_;
  MachineReg copied_;
  MachineReg dst_;
  MachineReg other_;
};

TEST_F(MachineIRUseVexThreeOperandFormsTest, MergesCopyIntoRegisterForm) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 16);
  builder_.Gen<PsubdXRegXReg>(dst_, other_);
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  EXPECT_EQ(CheckMachineIR(machine_ir_), kMachineIRCheckSuccess);
  ASSERT_EQ(bb_->insn_list().size(), 6u);
  const berberis::MachineInsn* op = InsnAt(2);
  ASSERT_EQ(op->opcode(), kMachineOpVpsubdXRegXRegXReg);
  EXPECT_EQ(op->RegAt(0), dst_);
  EXPECT_EQ(op->RegAt(1), copied_);
  EXPECT_EQ(op->RegAt(2), other_);
  EXPECT_TRUE(op->RegKindAt(0).IsDef());
  EXPECT_FALSE(op->RegKindAt(0).IsUse());
}

TEST_F(MachineIRUseVexThreeOperandFormsTest, MergesCopyIntoImmediateForm) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 16);
  builder_.Gen<PsrldqXRegImm>(dst_, int8_t{8});
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  EXPECT_EQ(CheckMachineIR(machine_ir_), kMachineIRCheckSuccess);
  ASSERT_EQ(bb_->insn_list().size(), 6u);
  const berberis::MachineInsn* op = InsnAt(2);
  ASSERT_EQ(op->opcode(), kMachineOpVpsrldqXRegXRegImm);
  EXPECT_EQ(op->RegAt(0), dst_);
  EXPECT_EQ(op->RegAt(1), copied_);
  EXPECT_EQ(AsMachineInsnX86_64(op)->imm(), 8u);
}

TEST_F(MachineIRUseVexThreeOperandFormsTest, MergesCopyAcrossUnrelatedInstructions) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 16);
  builder_.Gen<PaddqXRegXReg>(other_, copied_);
  builder_.Gen<PmaddwdXRegXReg>(dst_, other_);
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  EXPECT_EQ(CheckMachineIR(machine_ir_), kMachineIRCheckSuccess);
  ASSERT_EQ(bb_->insn_list().size(), 7u);
  EXPECT_EQ(InsnAt(2)->opcode(), kMachineOpPaddqXRegXReg);
  const berberis::MachineInsn* op = InsnAt(3);
  ASSERT_EQ(op->opcode(), kMachineOpVpmaddwdXRegXRegXReg);
  EXPECT_EQ(op->RegAt(1), copied_);
  EXPECT_EQ(op->RegAt(2), other_);
}

TEST_F(MachineIRUseVexThreeOperandFormsTest, ReadsCopiedRegisterForOwnDestination) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 16);
  builder_.Gen<PhadddXRegXReg>(dst_, dst_);
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  EXPECT_EQ(CheckMachineIR(machine_ir_), kMachineIRCheckSuccess);
  const berberis::MachineInsn* op = InsnAt(2);
  ASSERT_EQ(op->opcode(), kMachineOpVphadddXRegXRegXReg);
  EXPECT_EQ(op->RegAt(0), dst_);
  EXPECT_EQ(op->RegAt(1), copied_);
  EXPECT_EQ(op->RegAt(2), copied_);
}

TEST_F(MachineIRUseVexThreeOperandFormsTest, KeepsScalarOperationInSourceOrder) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 16);
  builder_.Gen<SubsdXRegXReg>(dst_, other_);
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  const berberis::MachineInsn* op = InsnAt(2);
  ASSERT_EQ(op->opcode(), kMachineOpVsubsdXRegXRegXReg);
  EXPECT_EQ(op->RegAt(1), copied_);
  EXPECT_EQ(op->RegAt(2), other_);
}

TEST_F(MachineIRUseVexThreeOperandFormsTest, KeepsOperationWhoseCopiedRegisterChanges) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 16);
  builder_.Gen<PxorXRegXReg>(copied_, copied_);
  builder_.Gen<PadddXRegXReg>(dst_, other_);
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  ASSERT_EQ(InsnAt(2)->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(InsnAt(4)->opcode(), kMachineOpPadddXRegXReg);
}

TEST_F(MachineIRUseVexThreeOperandFormsTest, KeepsOperationWhoseDestinationIsUsedBefore) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 16);
  builder_.Gen<PaddqXRegXReg>(other_, dst_);
  builder_.Gen<PadddXRegXReg>(dst_, other_);
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  ASSERT_EQ(InsnAt(2)->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(InsnAt(4)->opcode(), kMachineOpPadddXRegXReg);
}

TEST_F(MachineIRUseVexThreeOperandFormsTest, KeepsOperationWithoutCopy) {
  builder_.Gen<PseudoDefXReg>(dst_);
  builder_.Gen<PadddXRegXReg>(dst_, other_);
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  EXPECT_EQ(InsnAt(2)->opcode(), kMachineOpPseudoDefXReg);
  EXPECT_EQ(InsnAt(3)->opcode(), kMachineOpPadddXRegXReg);
}

TEST_F(MachineIRUseVexThreeOperandFormsTest, KeepsOperationWithoutAvxForm) {
  builder_.Gen<PseudoCopy>(dst_, copied_, 16);
  builder_.Gen<AesencXRegXReg>(dst_, other_);
  Finish();

  UseVexThreeOperandForms(&machine_ir_);

  ASSERT_EQ(InsnAt(2)->opcode(), kMachineOpPseudoCopy);
  EXPECT_EQ(InsnAt(3)->opcode(), kMachineOpAesencXRegXReg);
}

}  // namespace

}  // namespace berberis::x86_64
