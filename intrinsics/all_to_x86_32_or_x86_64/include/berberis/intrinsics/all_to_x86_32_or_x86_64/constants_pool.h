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

#ifndef BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_CONSTANTS_POOL_H_
#define BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_CONSTANTS_POOL_H_

#include <cinttypes>

#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/common/constants_pool.h"

namespace berberis::constants_pool {

// 64 bit constants for use with arithmetic operations.
// Used because only 32 bit immediates are supported on x86-64.

template <auto Value>
struct Const {};

// Specialize Const<Value> using an out-of-line definition.
#define BERBERIS_CONST_EXTERN(Value) \
  template <>                        \
  struct Const<Value> {              \
    static const int32_t kValue;     \
  }

// Specialize Const<Value> using a reference to another constant's int32_t address.
#define BERBERIS_CONST_ALIAS(Value, Alias)          \
  template <>                                       \
  struct Const<Value> {                             \
    static constexpr const int32_t& kValue = Alias; \
  }

template <auto Value>
inline const int32_t& kConst = Const<Value>::kValue;

BERBERIS_CONST_EXTERN(uint32_t{32});
BERBERIS_CONST_EXTERN(uint32_t{63});

// Helper constant for BsrToClz conversion. 63 for int32_t, 127 for int64_t.
template <typename IntType>
inline constexpr int32_t kBsrToClz = kImpossibleTypeConst<IntType>;
template <>
inline const int32_t kBsrToClz<int32_t> = kConst<uint32_t{63}>;

// Helper constant for width of the type. 32 for int32_t, 64 for int64_t.
template <typename IntType>
inline constexpr int32_t kWidthInBits = kImpossibleTypeConst<IntType>;
template <>
inline const int32_t kWidthInBits<int32_t> = kConst<uint32_t{32}>;

}  // namespace berberis::constants_pool

namespace berberis::constants_offsets {

template <typename IntType>
inline constexpr TypeConstantAccessor<&constants_pool::kBsrToClz<IntType>> kBsrToClz{};

template <typename IntType>
inline constexpr TypeConstantAccessor<&constants_pool::kWidthInBits<IntType>> kWidthInBits{};

}  // namespace berberis::constants_offsets

#endif  // BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_CONSTANTS_POOL_H_
