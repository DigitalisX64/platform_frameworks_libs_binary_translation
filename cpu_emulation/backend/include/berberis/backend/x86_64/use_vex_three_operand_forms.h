#ifndef BERBERIS_BACKEND_X86_64_USE_VEX_THREE_OPERAND_FORMS_H_
#define BERBERIS_BACKEND_X86_64_USE_VEX_THREE_OPERAND_FORMS_H_

#include "berberis/backend/x86_64/machine_ir.h"

namespace berberis::x86_64 {

// Before register allocation, merges a copy into the two-address SSE
// operation that overwrites it, using the operation's three-operand AVX form:
//
//   PseudoCopy d, a          (d not used before op, a not redefined)
//   ...
//   OP d, b                  (or OP d, imm)
// into
//   ...
//   VOP d, a, b              (or VOP d, a, imm)
//
// The AVX form reads a and writes d, so d no longer has to start as a copy of
// a: the allocator drops the move whenever a stays live past the operation.
// The caller runs this pass only on hosts with AVX.
void UseVexThreeOperandForms(MachineIR* machine_ir);

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_USE_VEX_THREE_OPERAND_FORMS_H_
