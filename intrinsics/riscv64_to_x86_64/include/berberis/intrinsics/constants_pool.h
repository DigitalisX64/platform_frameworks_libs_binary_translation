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

#ifndef RISCV64_TO_X86_64_BERBERIS_INTRINSICS_CONSTANTS_POOL_H_
#define RISCV64_TO_X86_64_BERBERIS_INTRINSICS_CONSTANTS_POOL_H_

#include <cinttypes>

#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/common/constants_pool.h"
#include "berberis/intrinsics/common/intrinsics_float.h"

namespace berberis::constants_pool {

// Specialize VectorConst<Value> using an out-of-line definition.
#pragma push_macro("VECTOR_CONST_EXTERN")
#define VECTOR_CONST_EXTERN(Value) \
  template <>                      \
  struct VectorConst<Value> {      \
    static const int32_t kValue;   \
  }

// Specialize VectorConst<Value> using a reference to another constant's int32_t address.
#pragma push_macro("VECTOR_CONST_ALIAS")
#define VECTOR_CONST_ALIAS(Value, Alias)            \
  template <>                                       \
  struct VectorConst<Value> {                       \
    static constexpr const int32_t& kValue = Alias; \
  }

VECTOR_CONST_EXTERN(int8_t{-128});
VECTOR_CONST_EXTERN(int8_t{127});
VECTOR_CONST_EXTERN(int16_t{-0x8000});
VECTOR_CONST_EXTERN(int16_t{0x7fff});
VECTOR_CONST_EXTERN(int32_t{static_cast<int32_t>(-0x8000'0000)});
VECTOR_CONST_EXTERN(int32_t{-0x0080'0000});
VECTOR_CONST_EXTERN(int32_t{0x3f80'0000});
VECTOR_CONST_EXTERN(int32_t{0x7f80'0000});
VECTOR_CONST_EXTERN(int32_t{0x7fff'ffff});
VECTOR_CONST_EXTERN(int64_t{static_cast<int64_t>(-0x8000'0000'0000'0000)});
VECTOR_CONST_EXTERN(int64_t{0x3ff0'0000'0000'0000});
VECTOR_CONST_EXTERN(int64_t{0x7ff0'0000'0000'0000});
VECTOR_CONST_EXTERN(int64_t{0x7fff'ffff'ffff'ffff});
VECTOR_CONST_EXTERN(int64_t{-0x0010'0000'0000'0000});
VECTOR_CONST_EXTERN(uint64_t{0x0000'0000'0000'0000});

VECTOR_CONST_ALIAS(int8_t{0x00}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
VECTOR_CONST_ALIAS(uint8_t{0x00}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
VECTOR_CONST_ALIAS(int16_t{0x0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
VECTOR_CONST_ALIAS(uint16_t{0x0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
VECTOR_CONST_ALIAS(uint8_t{127}, kVectorConst<int8_t{127}>);
VECTOR_CONST_ALIAS(uint8_t{128}, kVectorConst<int8_t{-128}>);
VECTOR_CONST_ALIAS(uint16_t{0x7fff}, kVectorConst<int16_t{0x7fff}>);
VECTOR_CONST_ALIAS(uint16_t{0x8000}, kVectorConst<int16_t{-0x8000}>);
VECTOR_CONST_ALIAS(int32_t{0x0000'0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
VECTOR_CONST_ALIAS(uint32_t{0x0000'0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
VECTOR_CONST_ALIAS(uint32_t{0x3f80'0000}, kVectorConst<int32_t{0x3f80'0000}>);
VECTOR_CONST_ALIAS(uint32_t{0x7f80'0000}, kVectorConst<int32_t{0x7f80'0000}>);
VECTOR_CONST_ALIAS(uint32_t{0x7fff'ffff}, kVectorConst<int32_t{0x7fff'ffff}>);
VECTOR_CONST_ALIAS(uint32_t{0x8000'0000},
                   kVectorConst<int32_t{static_cast<int32_t>(-0x8000'0000)}>);
VECTOR_CONST_ALIAS(uint32_t{0xff80'0000}, kVectorConst<int32_t{-0x0080'0000}>);
VECTOR_CONST_ALIAS(int64_t{0x0000'0000'0000'0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);

VECTOR_CONST_EXTERN(uint64_t{0x7fc'00000'7fc'00000});

VECTOR_CONST_ALIAS(uint64_t{0x7ff0'0000'0000'0000}, kVectorConst<int64_t{0x7ff0'0000'0000'0000}>);

VECTOR_CONST_EXTERN(uint64_t{0x7ff8'0000'0000'0000});

VECTOR_CONST_ALIAS(uint64_t{0xfff0'0000'0000'0000}, kVectorConst<int64_t{-0x0010'0000'0000'0000}>);

VECTOR_CONST_EXTERN(uint64_t{0xffff'ffff'0000'0000});
VECTOR_CONST_EXTERN(uint64_t{0xffff'ffff'7fc0'0000});
VECTOR_CONST_EXTERN(uint64_t{0xffff'ffff'ffff'ffff});

VECTOR_CONST_ALIAS(int8_t{-1}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
VECTOR_CONST_ALIAS(uint8_t{0xff}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
VECTOR_CONST_ALIAS(int16_t{-1}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
VECTOR_CONST_ALIAS(uint16_t{0xffff}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
VECTOR_CONST_ALIAS(int32_t{-1}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
VECTOR_CONST_ALIAS(uint32_t{0xffff'ffff}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
VECTOR_CONST_ALIAS(int64_t{-1}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);

#pragma pop_macro("VECTOR_CONST_EXTERN")
#pragma pop_macro("VECTOR_CONST_ALIAS")

// 64 bit constants for use with arithmetic operations.
// Used because only 32 bit immediates are supported on x86-64.

template <auto Value>
struct Const {};

// Specialize Const<Value> using an out-of-line definition.
#pragma push_macro("CONST_EXTERN")
#define CONST_EXTERN(Value)      \
  template <>                    \
  struct Const<Value> {          \
    static const int32_t kValue; \
  }

// Specialize Const<Value> using a reference to another constant's int32_t address.
#pragma push_macro("CONST_ALIAS")
#define CONST_ALIAS(Value, Alias)                   \
  template <>                                       \
  struct Const<Value> {                             \
    static constexpr const int32_t& kValue = Alias; \
  }

template <auto Value>
inline const int32_t& kConst = Const<Value>::kValue;

CONST_EXTERN(uint32_t{32});
CONST_EXTERN(uint32_t{63});
CONST_EXTERN(uint64_t{64});
CONST_EXTERN(uint64_t{127});

CONST_ALIAS(int8_t{0x00}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
CONST_ALIAS(uint8_t{0x00}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
CONST_ALIAS(int16_t{0x0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
CONST_ALIAS(uint16_t{0x0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
CONST_ALIAS(int32_t{0x0000'0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
CONST_ALIAS(uint32_t{0x0000'0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
CONST_ALIAS(int64_t{0x0000'0000'0000'0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);
CONST_ALIAS(uint64_t{0x0000'0000'0000'0000}, kVectorConst<uint64_t{0x0000'0000'0000'0000}>);

CONST_EXTERN(uint64_t{0x8000'0000'0000'00ff});

CONST_ALIAS(int8_t{-1}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
CONST_ALIAS(uint8_t{0xff}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
CONST_ALIAS(int16_t{-1}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
CONST_ALIAS(uint16_t{0xffff}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
CONST_ALIAS(int32_t{-1}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
CONST_ALIAS(uint32_t{0xffff'ffff}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
CONST_ALIAS(int64_t{-1}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);
CONST_ALIAS(uint64_t{0xffff'ffff'ffff'ffff}, kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}>);

#pragma pop_macro("CONST_EXTERN")
#pragma pop_macro("CONST_ALIAS")

// Constant suitable for NaN boxing of RISC-V 32bit float with PXor.
// Note: technically we only need to Nan-box Float32 since we don't support Float16 yet.
template <typename FloatType>
inline constexpr int32_t kNanBox = kImpossibleTypeConst<FloatType>;
template <>
inline const int32_t& kNanBox<intrinsics::Float32> = kVectorConst<uint64_t{0xffff'ffff'0000'0000}>;

// Canonically Nan boxed canonical NaN.
// Note: technically we only need to Nan-box Float32 since we don't support Float16 yet.
template <typename FloatType>
inline constexpr int32_t kNanBoxedNans = kImpossibleTypeConst<FloatType>;
template <>
inline const int32_t& kNanBoxedNans<intrinsics::Float32> =
    kVectorConst<uint64_t{0xffff'ffff'7fc0'0000}>;

// Canonical NaNs. Float32 and Float64 are supported.
template <typename FloatType>
inline constexpr int32_t kCanonicalNans = kImpossibleTypeConst<FloatType>;
template <>
inline const int32_t& kCanonicalNans<intrinsics::Float32> =
    kVectorConst<uint64_t{0x7fc0'0000'7fc0'0000}>;
template <>
inline const int32_t& kCanonicalNans<intrinsics::Float64> =
    kVectorConst<uint64_t{0x7ff8'0000'0000'0000}>;

// Helper constant for BsrToClz conversion. 63 for int32_t, 127 for int64_t.
template <typename IntType>
inline constexpr int32_t kBsrToClz = kImpossibleTypeConst<IntType>;
template <>
inline const int32_t kBsrToClz<int32_t> = kConst<uint32_t{63}>;
template <>
inline const int32_t kBsrToClz<int64_t> = kConst<uint64_t{127}>;

// Helper constant for width of the type. 32 for int32_t, 64 for int64_t.
template <typename IntType>
inline constexpr int32_t kWidthInBits = kImpossibleTypeConst<IntType>;
template <>
inline const int32_t kWidthInBits<int32_t> = kConst<uint32_t{32}>;
template <>
inline const int32_t kWidthInBits<int64_t> = kConst<uint64_t{64}>;

extern const int32_t kRiscVToX87Exceptions;
extern const int32_t kX87ToRiscVExceptions;

extern const int32_t kVid64Bit;
extern const int32_t kVid32Bit;
extern const int32_t kVid16Bit;
extern const int32_t kVid8Bit;

extern const int32_t kBitMaskTable;
extern const int32_t kBitMaskTo32bitMask;
extern const int32_t kBitMaskTo16bitMask;
extern const int32_t kBitMaskTo8bitMask;

extern const int32_t kPMovmskwToPMovmskb;
extern const int32_t kPMovmskdToPMovmskb;
extern const int32_t kPMovmskqToPMovmskb;

}  // namespace berberis::constants_pool

namespace berberis::constants_offsets {

inline constexpr ConstantAccessor<&constants_pool::kRiscVToX87Exceptions> kRiscVToX87Exceptions;

inline constexpr ConstantAccessor<&constants_pool::kX87ToRiscVExceptions> kX87ToRiscVExceptions;

template <typename IntType>
inline constexpr TypeConstantAccessor<&constants_pool::kBsrToClz<IntType>> kBsrToClz{};

template <typename FloatType>
inline constexpr TypeConstantAccessor<&constants_pool::kCanonicalNans<FloatType>> kCanonicalNans{};

template <typename FloatType>
inline constexpr TypeConstantAccessor<&constants_pool::kNanBox<FloatType>> kNanBox{};

template <typename FloatType>
inline constexpr TypeConstantAccessor<&constants_pool::kNanBoxedNans<FloatType>> kNanBoxedNans{};

template <typename IntType>
inline constexpr TypeConstantAccessor<&constants_pool::kWidthInBits<IntType>> kWidthInBits{};

template <auto Value>
inline constexpr VectorConstantAccessor<Value> kVectorConst{};

}  // namespace berberis::constants_offsets

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_CONSTANTS_POOL_H_
