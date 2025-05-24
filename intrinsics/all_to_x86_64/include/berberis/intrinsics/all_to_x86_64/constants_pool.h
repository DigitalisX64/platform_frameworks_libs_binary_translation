/*
 * Copyright (C) 2020 The Android Open Source Project
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

#ifndef BERBERIS_INTRINSICS_ALL_TO_X86_64_CONSTANTS_POOL_H_
#define BERBERIS_INTRINSICS_ALL_TO_X86_64_CONSTANTS_POOL_H_

#include <cinttypes>

#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/all_to_x86_32_or_x86_64/constants_pool.h"
#include "berberis/intrinsics/common/constants_pool.h"

namespace berberis::constants_pool {

BERBERIS_CONST_EXTERN(uint64_t{64});
BERBERIS_CONST_EXTERN(uint64_t{127});

// Helper constant for BsrToClz conversion. 63 for int32_t, 127 for int64_t.
template <>
inline const int32_t& kBsrToClz<int64_t> = kConst<uint64_t{127}>;

// Helper constant for width of the type. 32 for int32_t, 64 for int64_t.
template <>
inline const int32_t& kWidthInBits<int64_t> = kConst<uint64_t{64}>;

}  // namespace berberis::constants_pool

#endif  // BERBERIS_INTRINSICS_ALL_TO_X86_64_CONSTANTS_POOL_H_
