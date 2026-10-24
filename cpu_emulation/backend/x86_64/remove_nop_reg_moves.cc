#include "berberis/backend/x86_64/remove_nop_reg_moves.h"

#include "berberis/backend/x86_64/machine_ir.h"

namespace berberis::x86_64 {

namespace {

bool IsNopRegMove(const berberis::MachineInsn* insn) {
  switch (insn->opcode()) {
    case kMachineOpMovdqaXRegXReg:
    case kMachineOpMovsdXRegXReg:
    case kMachineOpMovssXRegXReg:
    case kMachineOpMovqRegReg:
      break;
    default:
      return false;
  }
  // Before allocation a vreg moved to itself is how a lowering gives a vreg a
  // definition; only an allocated host register moved to itself is a no-op.
  return insn->RegAt(0).IsHardReg() && insn->RegAt(0) == insn->RegAt(1);
}

}  // namespace

void RemoveNopRegMoves(MachineIR* machine_ir) {
  for (auto* machine_bb : machine_ir->bb_list()) {
    machine_bb->insn_list().remove_if(IsNopRegMove);
  }
}

}  // namespace berberis::x86_64
