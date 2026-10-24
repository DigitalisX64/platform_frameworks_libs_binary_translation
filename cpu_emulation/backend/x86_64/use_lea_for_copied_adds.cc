#include "berberis/backend/x86_64/use_lea_for_copied_adds.h"

#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/checks.h"

namespace berberis::x86_64 {

namespace {

bool References(const berberis::MachineInsn* insn, MachineReg reg) {
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegAt(i) == reg) {
      return true;
    }
  }
  return false;
}

bool Defines(const berberis::MachineInsn* insn, MachineReg reg) {
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegAt(i) == reg && insn->RegKindAt(i).IsDef()) {
      return true;
    }
  }
  return false;
}

bool Uses(const berberis::MachineInsn* insn, MachineReg reg) {
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegAt(i) == reg && insn->RegKindAt(i).IsUse()) {
      return true;
    }
  }
  return false;
}

// Whether nothing reads the flags that the instruction at it writes: the next
// instruction in bb that touches them only writes them, or none does and they
// do not live out of bb.
bool AreFlagsDeadAfter(const MachineBasicBlock* bb,
                       MachineInsnList::const_iterator it,
                       MachineReg flags) {
  for (++it; it != bb->insn_list().end(); ++it) {
    if (Uses(*it, flags)) {
      return false;
    }
    if (Defines(*it, flags)) {
      return true;
    }
  }
  if (!flags.IsVReg()) {
    return false;
  }
  for (auto reg : bb->live_out()) {
    if (reg == flags) {
      return false;
    }
  }
  return true;
}

// Whether each virtual register is known to live in a general register: some
// instruction other than a copy gives it a general register class, and none
// gives it a class without general registers. LEA reads its sources from
// general registers, so a source with any other class would leave the
// allocator no register for it.
std::vector<bool> FindGeneralRegs(const MachineIR* machine_ir) {
  std::vector<bool> general(machine_ir->NumVReg(), false);
  std::vector<bool> other(machine_ir->NumVReg(), false);
  for (const auto* bb : machine_ir->bb_list()) {
    for (const auto* insn : bb->insn_list()) {
      if (insn->opcode() == kMachineOpPseudoCopy) {
        continue;
      }
      for (int i = 0; i < insn->NumRegOperands(); i++) {
        MachineReg reg = insn->RegAt(i);
        if (!reg.IsVReg()) {
          continue;
        }
        if (insn->RegKindAt(i).RegClass()->IsSubsetOf(&kGeneralReg64)) {
          general[reg.GetVRegIndex()] = true;
        } else {
          other[reg.GetVRegIndex()] = true;
        }
      }
    }
  }
  for (size_t i = 0; i < general.size(); i++) {
    general[i] = general[i] && !other[i];
  }
  return general;
}

// The LEA computing op with a in place of op's destination, or nullptr if op is
// not an addition LEA can express.
berberis::MachineInsn* NewLea(MachineIR* machine_ir,
                              const berberis::MachineInsn* op,
                              MachineReg a) {
  MachineReg dst = op->RegAt(0);
  switch (op->opcode()) {
    case kMachineOpAddqRegImm: {
      auto imm = static_cast<int32_t>(AsMachineInsnX86_64(op)->imm());
      return machine_ir->NewInsn<LeaqRegOp>(dst, {.base = a, .disp = imm});
    }
    case kMachineOpSubqRegImm: {
      auto imm = static_cast<int32_t>(AsMachineInsnX86_64(op)->imm());
      if (imm == std::numeric_limits<int32_t>::min()) {
        return nullptr;
      }
      return machine_ir->NewInsn<LeaqRegOp>(dst, {.base = a, .disp = -imm});
    }
    case kMachineOpAddqRegReg: {
      // ADDQ d, d adds a to itself. Any other b is a general register already.
      MachineReg b = op->RegAt(1) == dst ? a : op->RegAt(1);
      if (!b.IsVReg()) {
        return nullptr;
      }
      return machine_ir->NewInsn<LeaqRegOp>(dst, {.base = a, .index = b});
    }
    default:
      return nullptr;
  }
}

int FlagsOperand(MachineOpcode opcode) {
  switch (opcode) {
    case kMachineOpAddqRegImm:
    case kMachineOpSubqRegImm:
      return 1;
    case kMachineOpAddqRegReg:
      return 2;
    default:
      return -1;
  }
}

// Merges the copy that starts op's destination into op. Returns true if it did.
bool TryMerge(MachineIR* machine_ir,
              const std::vector<bool>& general,
              MachineBasicBlock* bb,
              MachineInsnList::iterator op_it) {
  berberis::MachineInsn* op = *op_it;
  int flags_operand = FlagsOperand(op->opcode());
  if (flags_operand < 0) {
    return false;
  }
  MachineReg dst = op->RegAt(0);
  if (!dst.IsVReg() || !AreFlagsDeadAfter(bb, op_it, op->RegAt(flags_operand))) {
    return false;
  }
  // The latest earlier instruction that touches dst must be the copy into it,
  // and the copied register must keep its value up to op.
  auto it = op_it;
  while (it != bb->insn_list().begin()) {
    --it;
    if (!References(*it, dst)) {
      continue;
    }
    berberis::MachineInsn* copy = *it;
    if (copy->opcode() != kMachineOpPseudoCopy || copy->RegAt(0) != dst) {
      return false;
    }
    MachineReg src = copy->RegAt(1);
    if (!src.IsVReg() || src == dst || !general[src.GetVRegIndex()]) {
      return false;
    }
    for (auto between = std::next(it); between != op_it; ++between) {
      if (Defines(*between, src)) {
        return false;
      }
    }
    berberis::MachineInsn* lea = NewLea(machine_ir, op, src);
    if (!lea) {
      return false;
    }
    bb->insn_list().erase(it);
    *op_it = lea;
    return true;
  }
  return false;
}

}  // namespace

void UseLeaForCopiedAdds(MachineIR* machine_ir) {
  std::vector<bool> general = FindGeneralRegs(machine_ir);
  for (auto* bb : machine_ir->bb_list()) {
    for (auto op_it = bb->insn_list().begin(); op_it != bb->insn_list().end(); ++op_it) {
      TryMerge(machine_ir, general, bb, op_it);
    }
  }
}

}  // namespace berberis::x86_64
