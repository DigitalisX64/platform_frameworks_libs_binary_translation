/*
 * Copyright (C) 2023 The Android Open Source Project
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

#ifndef BERBERIS_HEAVY_OPTIMIZER_ARM64_CALL_INTRINSIC_H_
#define BERBERIS_HEAVY_OPTIMIZER_ARM64_CALL_INTRINSIC_H_

// Guest-agnostic host x86_64 intrinsic->SSE call lowering. This was
// byte-identical (modulo include-guard names) to
// heavy_optimizer/riscv64/call_intrinsic.h and silently drifted. Forward to
// the riscv64 copy rather than forking it. Do NOT copy content back here -
// edit the riscv64 file; both guests share it.
#include "../riscv64/call_intrinsic.h"

#endif  // BERBERIS_HEAVY_OPTIMIZER_ARM64_CALL_INTRINSIC_H_
