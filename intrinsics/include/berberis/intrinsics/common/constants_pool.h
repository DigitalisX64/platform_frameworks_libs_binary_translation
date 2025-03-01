/*
 * Copyright (C) 2025 The Android Open Source Project
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

#ifndef BERBERIS_INTRINSICS_COMMON_CONSTANTS_POOL_H_
#define BERBERIS_INTRINSICS_COMMON_CONSTANTS_POOL_H_

#include <stdint.h>
#include <bit>

#include <type_traits>

namespace berberis {

namespace constants_pool {

#if defined(__i386__) || defined(__x86_64__)
using ConstPoolAddrType = int32_t;
#else
using ConstPoolAddrType = intptr_t;
#endif

// Vector constants, that is: constants are repeated to fill 128bit SIMD register.
template <auto Value, typename = void>
struct VectorConst {};

template <auto Value>
inline const int32_t& kVectorConst = VectorConst<Value>::kValue;

template <auto Value>
struct VectorConst<Value,
                   std::enable_if_t<std::is_unsigned_v<std::remove_cvref_t<decltype(Value)>>>> {
  static constexpr const ConstPoolAddrType& kValue =
      kVectorConst<static_cast<std::make_signed_t<std::remove_cvref_t<decltype(Value)>>>(Value)>;
};

template <float Value>
struct VectorConst<Value> {
  static constexpr const ConstPoolAddrType& kValue = kVectorConst<std::bit_cast<int32_t>(Value)>;
};

template <double Value>
struct VectorConst<Value> {
  static constexpr const ConstPoolAddrType& kValue = kVectorConst<std::bit_cast<int64_t>(Value)>;
};

}  // namespace constants_pool

namespace constants_offsets {

// constants_offsets namespace includes compile-time versions of constants used in macro assembler
// functions. This allows the static verifier assembler to use static versions of the macro-
// assembly functions.
using ConstPoolAddrType = constants_pool::ConstPoolAddrType;

template <const int32_t* constant_addr>
class ConstantAccessor {
 public:
  constexpr operator ConstPoolAddrType() const {
    if (std::is_constant_evaluated()) {
      return 0;
    } else {
      return *constant_addr;
    }
  }
};

template <const auto Value>
class TypeConstantAccessor {
 public:
  constexpr operator ConstPoolAddrType() const {
    if (std::is_constant_evaluated()) {
      return 0;
    } else {
      return *Value;
    }
  }
};

template <const auto Value>
class VectorConstantAccessor {
 public:
  constexpr operator ConstPoolAddrType() const {
    if (std::is_constant_evaluated()) {
      return 0;
    } else {
      return constants_pool::VectorConst<Value>::kValue;
    }
  }
};

}  // namespace constants_offsets

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_COMMON_CONSTANTS_POOL_H_
