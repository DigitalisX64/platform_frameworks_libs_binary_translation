// RenameCopyUses on instructions that define a register they also read
// through a separate operand: the use is read before the def is written, so
// it still sees the copy, unless the def is early-clobber.

#include "gtest/gtest.h"

#include <string>

#include "berberis/backend/code_emitter.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/backend/x86_64/machine_ir_check.h"
#include "berberis/backend/x86_64/machine_ir_opt.h"
#include "berberis/backend/x86_64/rename_copy_uses.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis::x86_64 {

namespace {

// Reads operand 0 and writes operand 1 before it has read operand 0, like an
// intrinsic whose scratch output is written early.
class EarlyClobberTestInsn : public MachineInsnX86_64 {
 public:
  EarlyClobberTestInsn(MachineReg use, MachineReg clobber) : MachineInsnX86_64(&kInfo) {
    SetRegAt(0, use);
    SetRegAt(1, clobber);
  }
  EarlyClobberTestInsn(const EarlyClobberTestInsn& other) : MachineInsnX86_64(other) {}

  std::string GetDebugString() const override { return "EARLY_CLOBBER_TEST"; }
  void Emit(CodeEmitter*) const override {}
  MachineInsn* Clone(Arena* arena) const override {
    return NewInArena<EarlyClobberTestInsn>(arena, *this);
  }

 private:
  static constexpr MachineInsnInfo kInfo = MachineInsnInfo(
      {kMachineOpAddqRegReg,
       2,
       {{&kGeneralReg64, MachineRegKind::kUse}, {&kGeneralReg64, MachineRegKind::kDefEarlyClobber}},
       kMachineInsnDefault});
};

int CountCopies(MachineBasicBlock* bb) {
  int copies = 0;
  for (auto* insn : bb->insn_list()) {
    if (insn->is_copy()) {
      copies++;
    }
  }
  return copies;
}

TEST(MachineIRRenameCopyUsesOrderTest, UseOfRegisterTheSameInsnRedefines) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb = machine_ir.NewBasicBlock();
  MachineIRBuilder builder(&machine_ir);

  MachineReg src = machine_ir.AllocVReg();
  MachineReg copy = machine_ir.AllocVReg();
  MachineReg out = machine_ir.AllocVReg();

  builder.StartBasicBlock(bb);
  builder.Gen<MovdqaXRegXReg>(src, src);
  builder.Gen<PseudoCopy>(copy, src, 16);
  // pmovsxbw copy, copy: widens in place, as the frontend's narrow-to-wide lowerings do.
  auto* widen = builder.Gen<PmovsxbwXRegXReg>(copy, copy);
  builder.Gen<PmullwXRegXReg>(copy, src);
  builder.Gen<MovdqaXRegXReg>(out, copy);
  builder.Gen<PseudoJump>(kNullGuestAddr);
  ASSERT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);

  RenameCopyUses(&machine_ir);
  RemoveDeadCode(&machine_ir);

  EXPECT_EQ(widen->RegAt(0), copy);
  EXPECT_EQ(widen->RegAt(1), src);
  EXPECT_EQ(CountCopies(bb), 0);
  EXPECT_EQ(CheckMachineIR(machine_ir), kMachineIRCheckSuccess);
}

TEST(MachineIRRenameCopyUsesOrderTest, UseNotRenamedToRegisterTheSameInsnRedefinesLater) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb = machine_ir.NewBasicBlock();
  MachineIRBuilder builder(&machine_ir);

  MachineReg src = machine_ir.AllocVReg();
  MachineReg copy = machine_ir.AllocVReg();
  MachineReg sum = machine_ir.AllocVReg();

  builder.StartBasicBlock(bb);
  builder.Gen<PseudoCopy>(copy, src, 8);
  builder.Gen<SubqRegImm>(src, 1, kMachineRegFLAGS);
  // The copy's source changed after the copy: the old value lives only in `copy`.
  auto* add = builder.Gen<AddqRegReg>(sum, copy, kMachineRegFLAGS);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  RenameCopyUses(&machine_ir);

  EXPECT_EQ(add->RegAt(1), copy);
}

TEST(MachineIRRenameCopyUsesOrderTest, UseNotRenamedToEarlyClobberDef) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* bb = machine_ir.NewBasicBlock();
  MachineIRBuilder builder(&machine_ir);

  MachineReg src = machine_ir.AllocVReg();
  MachineReg copy = machine_ir.AllocVReg();

  builder.StartBasicBlock(bb);
  builder.Gen<PseudoCopy>(copy, src, 8);
  // The clobber is written before `copy` is read, so `copy` must not become `src`.
  auto* insn = builder.Gen<EarlyClobberTestInsn>(copy, src);
  builder.Gen<PseudoJump>(kNullGuestAddr);

  RenameCopyUses(&machine_ir);

  EXPECT_EQ(insn->RegAt(0), copy);
  EXPECT_EQ(insn->RegAt(1), src);
}

}  // namespace

}  // namespace berberis::x86_64
