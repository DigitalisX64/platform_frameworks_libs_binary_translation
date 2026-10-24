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

#ifndef BERBERIS_HEAVY_OPTIMIZER_ARM64_HEAVY_OPTIMIZE_REGION_H_
#define BERBERIS_HEAVY_OPTIMIZER_ARM64_HEAVY_OPTIMIZE_REGION_H_

#include <cstddef>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

struct HeavyOptimizeParams {
  // Regions longer than ~200 insns are rare and the LivenessAnalyzer's memory
  // use grows with region length, so cap region size like the riscv64 tier.
  size_t max_number_of_instructions = 200;
  GuestAddr end_pc = GetGuestAddrRangeEnd();
};

// Optimizing (second-gear) translation of the ARM64 region at `pc` into
// `machine_code`. Returns {stop_pc, success, number_of_instructions}:
//   success == false && stop_pc == pc  -> could not translate the first
//   instruction; the caller must fall back to the lite tier / interpreter.
//
// The frontend (heavy_optimizer/arm64/frontend.{h,cc}) drives the ARM64 decoder
// through the SemanticsPlayer into x86_64 MachineIR. Instructions it does not yet
// translate make it bail, so a geared-up region falls back to the lite tier.
// `out_has_in_region_backedge`, if non-null, is set to whether the region
// captured an in-region loop back-edge (a hot loop translated without
// per-iteration region-exit dispatch). The runtime uses it to decide whether a
// small or partially-translated (bailed) region is still worth installing heavy.
std::tuple<GuestAddr, bool, size_t> HeavyOptimizeRegion(
    GuestAddr pc,
    MachineCode* machine_code,
    const HeavyOptimizeParams& params = HeavyOptimizeParams(),
    bool* out_has_in_region_backedge = nullptr);

}  // namespace berberis

#endif  // BERBERIS_HEAVY_OPTIMIZER_ARM64_HEAVY_OPTIMIZE_REGION_H_
