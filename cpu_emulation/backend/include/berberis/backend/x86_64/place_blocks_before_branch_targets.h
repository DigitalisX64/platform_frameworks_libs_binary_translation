#ifndef BERBERIS_BACKEND_X86_64_PLACE_BLOCKS_BEFORE_BRANCH_TARGETS_H_
#define BERBERIS_BACKEND_X86_64_PLACE_BLOCKS_BEFORE_BRANCH_TARGETS_H_

#include "berberis/backend/x86_64/machine_ir.h"

namespace berberis::x86_64 {

// Moves a basic block that ends in an unconditional PseudoBranch to just
// before the block it branches to, so the emitter can drop that JMP, when
// fewer jumps run after the move. Each jump is weighted by the loop nesting
// depth of the block that runs it. Only the layout changes: every block keeps
// its explicit terminator, the entry block stays first and recovery blocks
// stay where they are.
void PlaceBlocksBeforeBranchTargets(MachineIR* machine_ir);

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_PLACE_BLOCKS_BEFORE_BRANCH_TARGETS_H_
