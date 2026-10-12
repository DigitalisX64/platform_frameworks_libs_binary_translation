#ifndef BERBERIS_BACKEND_X86_64_REMOVE_NOP_REG_MOVES_H_
#define BERBERIS_BACKEND_X86_64_REMOVE_NOP_REG_MOVES_H_

#include "berberis/backend/x86_64/machine_ir.h"

namespace berberis::x86_64 {

// After register allocation, removes explicit register moves whose source and
// destination got the same host register and whose result equals their input
// (MOVDQA, MOVSD, MOVSS between XMM registers, 64-bit MOVQ). MOVL is kept: it
// clears the upper half of its destination. The PseudoCopy counterpart is
// RemoveNopPseudoCopy.
void RemoveNopRegMoves(MachineIR* machine_ir);

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_REMOVE_NOP_REG_MOVES_H_
