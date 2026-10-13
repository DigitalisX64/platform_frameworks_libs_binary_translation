#ifndef BERBERIS_BACKEND_X86_64_USE_LEA_FOR_COPIED_ADDS_H_
#define BERBERIS_BACKEND_X86_64_USE_LEA_FOR_COPIED_ADDS_H_

#include "berberis/backend/x86_64/machine_ir.h"

namespace berberis::x86_64 {

// Before register allocation, merges a copy into the 64-bit addition that
// overwrites it when nothing reads the flags the addition writes:
//
//   PseudoCopy d, a          (d not used before the add, a not redefined)
//   ...
//   ADDQ d, imm              (or SUBQ d, imm, or ADDQ d, b)
// into
//   ...
//   LEAQ d, [a + imm]        (or [a - imm], or [a + b])
//
// LEA reads a and writes d, so the allocator drops the move whenever a stays
// live past the addition, and it leaves the host flags alone.
void UseLeaForCopiedAdds(MachineIR* machine_ir);

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_USE_LEA_FOR_COPIED_ADDS_H_
