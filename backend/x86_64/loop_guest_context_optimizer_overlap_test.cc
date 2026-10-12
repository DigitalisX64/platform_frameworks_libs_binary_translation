// Loop guest context optimizer: slots it must keep in memory because they are
// accessed at more than one width or overlap another accessed slot's bytes.

#include "gtest/gtest.h"

#include "berberis/backend/x86_64/loop_guest_context_optimizer.h"

#include "berberis/backend/code_emitter.h"
#include "berberis/backend/x86_64/machine_ir_analysis.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/guest_state/guest_state_opaque.h"

namespace berberis::x86_64 {

namespace {

bool HasSimdSlots() {
  return DoesCpuStateHaveDedicatedSimdRegs() || DoesCpuStateHaveDedicatedVecRegs();
}

int32_t SimdSlotOffset(int index) {
  return DoesCpuStateHaveDedicatedSimdRegs() ? GetThreadStateSimdRegOffset(index)
                                             : GetThreadStateVRegOffset(index);
}

TEST(MachineIRLoopGuestContextOptimizerOverlap, FindUnmappableOffsets) {
  if (!HasSimdSlots()) {
    GTEST_SKIP() << "Guest CPU doesn't have SIMD registers";
  }
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* body = machine_ir.NewBasicBlock();
  MachineIRBuilder builder(&machine_ir);
  auto gp = machine_ir.AllocVReg();
  auto xmm = machine_ir.AllocVReg();

  builder.StartBasicBlock(body);
  // Slot 0: 64-bit and 128-bit SIMD accesses.
  builder.GenSetSimd<8>(SimdSlotOffset(0), xmm);
  builder.GenGetSimd<16>(xmm, SimdSlotOffset(0));
  // Slot 1: 128-bit, with its upper half also accessed at its own offset.
  builder.GenGetSimd<16>(xmm, SimdSlotOffset(1));
  builder.GenSetSimd<8>(SimdSlotOffset(1) + 8, xmm);
  // Slot 2 and a general register: one width each, no overlap.
  builder.GenGetSimd<16>(xmm, SimdSlotOffset(2));
  builder.GenSetSimd<16>(SimdSlotOffset(2), xmm);
  builder.GenGet(gp, GetThreadStateRegOffset(0));
  builder.GenPut(GetThreadStateRegOffset(0), gp);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  Loop loop({body}, machine_ir.arena());
  auto unmappable = FindUnmappableOffsets(&machine_ir, &loop);
  EXPECT_TRUE(unmappable.at(SimdSlotOffset(0)));
  EXPECT_TRUE(unmappable.at(SimdSlotOffset(1)));
  EXPECT_TRUE(unmappable.at(SimdSlotOffset(1) + 8));
  EXPECT_FALSE(unmappable.at(SimdSlotOffset(2)));
  EXPECT_FALSE(unmappable.at(GetThreadStateRegOffset(0)));
}

// The upper half's reader comes first, so the slots are not mixed-width at
// either offset: only the overlap keeps them in memory.
TEST(MachineIRLoopGuestContextOptimizerOverlap, OverlappingSlotsStayInMemory) {
  if (!HasSimdSlots()) {
    GTEST_SKIP() << "Guest CPU doesn't have SIMD registers";
  }
  Arena arena;
  MachineIR machine_ir(&arena);

  auto* preloop = machine_ir.NewBasicBlock();
  auto* body = machine_ir.NewBasicBlock();
  auto* afterloop = machine_ir.NewBasicBlock();
  machine_ir.AddEdge(preloop, body);
  machine_ir.AddEdge(body, body);
  machine_ir.AddEdge(body, afterloop);

  auto upper = machine_ir.AllocVReg();
  auto wide = machine_ir.AllocVReg();
  const auto offset = SimdSlotOffset(0);

  MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(preloop);
  builder.Gen<PseudoBranch>(body);

  builder.StartBasicBlock(body);
  builder.GenGetSimd<8>(upper, offset + 8);
  builder.GenSetSimd<16>(offset, wide);
  builder.Gen<PseudoCondBranch>(CodeEmitter::Condition::kZero, body, afterloop, kMachineRegFLAGS);

  builder.StartBasicBlock(afterloop);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  Loop loop({body}, machine_ir.arena());
  OptimizeLoop(&machine_ir, &loop);
  ASSERT_EQ(CheckMachineIR(machine_ir), x86_64::kMachineIRCheckSuccess);

  EXPECT_EQ(preloop->insn_list().size(), 1UL);
  ASSERT_EQ(body->insn_list().size(), 3UL);
  auto it = body->insn_list().begin();
  EXPECT_EQ((*it)->opcode(), kMachineOpMovsdXRegMemBaseDisp);
  EXPECT_EQ(AsMachineInsnX86_64(*it)->disp(), static_cast<uint32_t>(offset + 8));
  it++;
  EXPECT_EQ((*it)->opcode(), kMachineOpMovdqaMemBaseDispXReg);
  EXPECT_EQ(AsMachineInsnX86_64(*it)->disp(), static_cast<uint32_t>(offset));
  EXPECT_EQ(afterloop->insn_list().size(), 1UL);
}

}  // namespace

}  // namespace berberis::x86_64
