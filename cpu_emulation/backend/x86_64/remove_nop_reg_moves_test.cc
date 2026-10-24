#include "gtest/gtest.h"

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/backend/x86_64/remove_nop_reg_moves.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis::x86_64 {

namespace {

constexpr auto kMachineRegRBX = MachineRegs::kRBX;
constexpr auto kMachineRegRCX = MachineRegs::kRCX;
constexpr auto kMachineRegXMM0 = MachineRegs::kXMM0;
constexpr auto kMachineRegXMM3 = MachineRegs::kXMM3;
constexpr auto kMachineRegXMM4 = MachineRegs::kXMM4;
constexpr auto kMachineRegXMM5 = MachineRegs::kXMM5;

TEST(MachineIRRemoveNopRegMovesTest, RemovesHostRegisterMovedToItself) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb = machine_ir.NewBasicBlock();
  MachineIRBuilder builder(&machine_ir);

  builder.StartBasicBlock(bb);
  builder.Gen<MovdqaXRegXReg>(kMachineRegXMM3, kMachineRegXMM3);
  builder.Gen<MovsdXRegXReg>(kMachineRegXMM4, kMachineRegXMM4);
  builder.Gen<MovssXRegXReg>(kMachineRegXMM5, kMachineRegXMM5);
  builder.Gen<MovqRegReg>(kMachineRegRBX, kMachineRegRBX);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  ASSERT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);

  RemoveNopRegMoves(&machine_ir);

  EXPECT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);
  ASSERT_EQ(bb->insn_list().size(), 1u);
  EXPECT_EQ(bb->insn_list().front()->opcode(), kMachineOpPseudoJump);
}

TEST(MachineIRRemoveNopRegMovesTest, KeepsMovesThatChangeTheirDestination) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb = machine_ir.NewBasicBlock();
  MachineIRBuilder builder(&machine_ir);

  builder.StartBasicBlock(bb);
  // Different registers.
  builder.Gen<MovdqaXRegXReg>(kMachineRegXMM3, kMachineRegXMM4);
  builder.Gen<MovqRegReg>(kMachineRegRBX, kMachineRegRCX);
  // MOVL clears bits 32..63 of its destination.
  builder.Gen<MovlRegReg>(kMachineRegRBX, kMachineRegRBX);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  ASSERT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);

  RemoveNopRegMoves(&machine_ir);

  ASSERT_EQ(bb->insn_list().size(), 4u);
  auto it = bb->insn_list().begin();
  EXPECT_EQ((*it++)->opcode(), kMachineOpMovdqaXRegXReg);
  EXPECT_EQ((*it++)->opcode(), kMachineOpMovqRegReg);
  EXPECT_EQ((*it++)->opcode(), kMachineOpMovlRegReg);
  EXPECT_EQ((*it)->opcode(), kMachineOpPseudoJump);
}

TEST(MachineIRRemoveNopRegMovesTest, KeepsVirtualRegisterMovedToItself) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb = machine_ir.NewBasicBlock();
  MachineIRBuilder builder(&machine_ir);

  MachineReg vreg = machine_ir.AllocVReg();
  builder.StartBasicBlock(bb);
  // Before allocation this defines vreg.
  builder.Gen<MovdqaXRegXReg>(vreg, vreg);
  builder.Gen<MovdqaXRegXReg>(kMachineRegXMM0, vreg);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  ASSERT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);

  RemoveNopRegMoves(&machine_ir);

  EXPECT_EQ(bb->insn_list().size(), 3u);
}

}  // namespace

}  // namespace berberis::x86_64
