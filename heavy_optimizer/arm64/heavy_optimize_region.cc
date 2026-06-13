/*
 * Copyright (C) 2026 utzcoz
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "berberis/heavy_optimizer/arm64/heavy_optimize_region.h"

#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/macros.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

// Phase 0 scaffolding: the ARM64 optimizing frontend is not implemented yet.
// Always bail (translate zero instructions) so the two-gear runtime falls back
// to the lite translator. The build/runtime plumbing is exercised; the frontend
// (decoder -> MachineIR, mirroring heavy_optimizer/riscv64/frontend.{h,cc}) and
// the x86_64::GenCode call land in later phases.
std::tuple<GuestAddr, bool, size_t> HeavyOptimizeRegion(GuestAddr pc,
                                                        MachineCode* machine_code,
                                                        const HeavyOptimizeParams& params) {
  UNUSED(machine_code, params);
  return {pc, /*success=*/false, /*number_of_instructions=*/0};
}

}  // namespace berberis
