#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

#include "berberis/assembler/machine_code.h"
#include "berberis/backend/code_emitter.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/arena_alloc.h"

namespace berberis::x86_64 {

namespace {

using Condition = CodeEmitter::Condition;

std::vector<uint8_t> Bytes(const MachineCode& machine_code) {
  const auto* begin = machine_code.AddrAs<uint8_t>(0);
  return {begin, begin + machine_code.code_offset()};
}

enum class Arm { kThen, kElse, kOther };

struct Layout {
  bool then_bound;
  bool else_bound;
  Arm next;
};

// Emits `cond ? then : else` with the given arms already bound (laid out
// before the branch) and the given next block.
std::vector<uint8_t> EmitCondBranch(Condition cond, Layout layout) {
  Arena arena;
  MachineIR machine_ir(&arena);
  auto* then_bb = machine_ir.NewBasicBlock();
  auto* else_bb = machine_ir.NewBasicBlock();
  auto* other_bb = machine_ir.NewBasicBlock();
  auto* insn = machine_ir.NewInsn<PseudoCondBranch>(cond, then_bb, else_bb, kMachineRegFLAGS);

  MachineCode machine_code;
  CodeEmitter as(&machine_code, 0, machine_ir.NumBasicBlocks(), machine_ir.arena());
  if (layout.then_bound) {
    as.Bind(as.GetLabelAt(then_bb->id()));
  }
  if (layout.else_bound) {
    as.Bind(as.GetLabelAt(else_bb->id()));
  }
  const MachineBasicBlock* next = layout.next == Arm::kThen   ? then_bb
                                  : layout.next == Arm::kElse ? else_bb
                                                              : other_bb;
  as.set_next_label(as.GetLabelAt(next->id()));
  insn->Emit(&as);
  return Bytes(machine_code);
}

enum class Jump { kJccThen, kJccNotElse, kJmpThen, kJmpElse };

// The same layout, with the given jumps emitted by hand.
std::vector<uint8_t> EmitExpected(Condition cond, Layout layout, std::vector<Jump> jumps) {
  Arena arena;
  MachineCode machine_code;
  CodeEmitter as(&machine_code, 0, 2, &arena);
  auto* then_label = as.GetLabelAt(0);
  auto* else_label = as.GetLabelAt(1);
  if (layout.then_bound) {
    as.Bind(then_label);
  }
  if (layout.else_bound) {
    as.Bind(else_label);
  }
  for (auto jump : jumps) {
    switch (jump) {
      case Jump::kJccThen:
        as.Jcc(cond, *then_label);
        break;
      case Jump::kJccNotElse:
        as.Jcc(ToReverseCond(cond), *else_label);
        break;
      case Jump::kJmpThen:
        as.Jmp(*then_label);
        break;
      case Jump::kJmpElse:
        as.Jmp(*else_label);
        break;
    }
  }
  return Bytes(machine_code);
}

// A loop back edge whose exit is laid out elsewhere: the Jcc takes the back
// edge, and the exit gets the JMP.
TEST(PseudoCondBranchEmitTest, JccTakesBackwardElseArm) {
  Layout layout{.then_bound = false, .else_bound = true, .next = Arm::kOther};
  EXPECT_EQ(EmitCondBranch(Condition::kEqual, layout),
            EmitExpected(Condition::kEqual, layout, {Jump::kJccNotElse, Jump::kJmpThen}));
  EXPECT_NE(EmitCondBranch(Condition::kEqual, layout),
            EmitExpected(Condition::kEqual, layout, {Jump::kJccThen, Jump::kJmpElse}));
}

TEST(PseudoCondBranchEmitTest, JccTakesBackwardThenArm) {
  Layout layout{.then_bound = true, .else_bound = false, .next = Arm::kOther};
  EXPECT_EQ(EmitCondBranch(Condition::kBelow, layout),
            EmitExpected(Condition::kBelow, layout, {Jump::kJccThen, Jump::kJmpElse}));
}

TEST(PseudoCondBranchEmitTest, BothArmsForwardKeepCondition) {
  Layout layout{.then_bound = false, .else_bound = false, .next = Arm::kOther};
  EXPECT_EQ(EmitCondBranch(Condition::kEqual, layout),
            EmitExpected(Condition::kEqual, layout, {Jump::kJccThen, Jump::kJmpElse}));
}

TEST(PseudoCondBranchEmitTest, BothArmsBackwardKeepCondition) {
  Layout layout{.then_bound = true, .else_bound = true, .next = Arm::kOther};
  EXPECT_EQ(EmitCondBranch(Condition::kEqual, layout),
            EmitExpected(Condition::kEqual, layout, {Jump::kJccThen, Jump::kJmpElse}));
}

TEST(PseudoCondBranchEmitTest, ElseFallsThrough) {
  Layout layout{.then_bound = false, .else_bound = false, .next = Arm::kElse};
  EXPECT_EQ(EmitCondBranch(Condition::kEqual, layout),
            EmitExpected(Condition::kEqual, layout, {Jump::kJccThen}));
}

TEST(PseudoCondBranchEmitTest, ThenFallsThroughWhileElseIsBackward) {
  Layout layout{.then_bound = false, .else_bound = true, .next = Arm::kThen};
  EXPECT_EQ(EmitCondBranch(Condition::kEqual, layout),
            EmitExpected(Condition::kEqual, layout, {Jump::kJccNotElse}));
}

}  // namespace

}  // namespace berberis::x86_64
