/*
 * Copyright (C) 2024 The Android Open Source Project
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

#ifndef RISCV64_TO_ARM64_BERBERIS_INTRINSICS_VECTOR_INTRINSICS_H_
#define RISCV64_TO_ARM64_BERBERIS_INTRINSICS_VECTOR_INTRINSICS_H_

#include <cstdlib>

#include "berberis/intrinsics/simd_register.h"

namespace berberis::intrinsics {

template <typename ElementType>
[[nodiscard, gnu::pure]] inline std::tuple<SIMD128Register> BitMaskToSimdMask(
    [[maybe_unused]] size_t mask) {
  SIMD128Register result;
  abort();
  return {result};
}

}  // namespace berberis::intrinsics

#include "berberis/intrinsics/riscv64_to_all/vector_intrinsics.h"

#endif  // RISCV64_TO_ARM64_BERBERIS_INTRINSICS_VECTOR_INTRINSICS_H_