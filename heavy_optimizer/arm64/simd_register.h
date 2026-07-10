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

#ifndef BERBERIS_HEAVY_OPTIMIZER_ARM64_SIMD_REGISTER_H_
#define BERBERIS_HEAVY_OPTIMIZER_ARM64_SIMD_REGISTER_H_

// Guest-agnostic MachineReg wrapper for SIMD/FP values. The SimdReg class body
// was identical to heavy_optimizer/riscv64/simd_register.h (differing only by
// copyright/guard/comment) and silently drifted. Forward to the riscv64 copy
// rather than forking it. Do NOT copy content back here - edit the riscv64
// file; both guests share it.
#include "../riscv64/simd_register.h"

#endif  // BERBERIS_HEAVY_OPTIMIZER_ARM64_SIMD_REGISTER_H_
