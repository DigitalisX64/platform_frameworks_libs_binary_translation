/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef BERBERIS_INTRINSICS_COMMON_INTRINSICS_H_
#define BERBERIS_INTRINSICS_COMMON_INTRINSICS_H_

#include <cstdint>

#include "berberis/base/checks.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/common/intrinsics_float.h"  // Float16/Float32/Float64

namespace berberis {

class SIMD128Register;

namespace intrinsics {

enum EnumFromTemplateType : uint8_t {
  kInt8T = 1,
  kUInt8T = 0,
  kInt16T = 3,
  kUInt16T = 2,
  kInt32T = 5,
  kUInt32T = 4,
  kInt64T = 7,
  kUInt64T = 6,
  kFloat16 = 10,
  kFloat32 = 12,
  kFloat64 = 14,
  kSIMD128Register = 16,
};

constexpr EnumFromTemplateType EnumFromTemplateTypeToFloat(EnumFromTemplateType value) {
  DCHECK(value >= kUInt16T && value <= kInt64T);
  return EnumFromTemplateType{static_cast<uint8_t>((value & 0x6) + 8)};
}

template <EnumFromTemplateType kValue>
EnumFromTemplateType kFloat = EnumFromTemplateTypeToFloat(kValue);

constexpr EnumFromTemplateType EnumFromTemplateTypeToInt(EnumFromTemplateType value) {
  DCHECK((value >= kFloat16 && value <= kFloat64) && !(value & 1));
  return EnumFromTemplateType{static_cast<uint8_t>(value - 8)};
}

template <EnumFromTemplateType kValue>
EnumFromTemplateType kInt = EnumFromTemplateTypeToInt(kValue);

constexpr EnumFromTemplateType EnumFromTemplateTypeToNarrow(EnumFromTemplateType value) {
  DCHECK((value >= kUInt16T && value <= kInt64T) ||
         ((value >= kFloat32 && value <= kFloat64) && !(value & 1)));
  return EnumFromTemplateType{static_cast<uint8_t>(value - 2)};
}

template <EnumFromTemplateType kValue>
EnumFromTemplateType kNarrow = EnumFromTemplateTypeToNarrow(kValue);

constexpr EnumFromTemplateType EnumFromTemplateTypeToSigned(EnumFromTemplateType value) {
  DCHECK(value <= kInt64T);
  return EnumFromTemplateType{static_cast<uint8_t>(value | 1)};
}

template <EnumFromTemplateType kValue>
EnumFromTemplateType kSigned = EnumFromTemplateTypeToSigned(kValue);

constexpr EnumFromTemplateType EnumFromTemplateTypeToUnsigned(EnumFromTemplateType value) {
  DCHECK(value <= kInt64T);
  return EnumFromTemplateType{static_cast<uint8_t>(value & ~1)};
}

template <EnumFromTemplateType kValue>
EnumFromTemplateType kUnsigned = EnumFromTemplateTypeToUnsigned(kValue);

constexpr EnumFromTemplateType EnumFromTemplateTypeToWide(EnumFromTemplateType value) {
  DCHECK(value <= kInt32T || ((value >= kFloat16 && value <= kFloat32) && !(value & 1)));
  return EnumFromTemplateType{static_cast<uint8_t>(value + 2)};
}

template <EnumFromTemplateType kValue>
EnumFromTemplateType kWide = EnumFromTemplateTypeToWide(kValue);

template <typename Type>
constexpr EnumFromTemplateType TypeToEnumFromTemplateType() {
  if constexpr (std::is_same_v<int8_t, std::decay_t<Type>>) {
    return EnumFromTemplateType::kInt8T;
  } else if constexpr (std::is_same_v<uint8_t, std::decay_t<Type>>) {
    return EnumFromTemplateType::kUInt8T;
  } else if constexpr (std::is_same_v<int16_t, std::decay_t<Type>>) {
    return EnumFromTemplateType::kUInt16T;
  } else if constexpr (std::is_same_v<uint16_t, std::decay_t<Type>>) {
    return EnumFromTemplateType::kUInt16T;
  } else if constexpr (std::is_same_v<int32_t, std::decay_t<Type>>) {
    return EnumFromTemplateType::kUInt32T;
  } else if constexpr (std::is_same_v<uint32_t, std::decay_t<Type>>) {
    return EnumFromTemplateType::kUInt32T;
  } else if constexpr (std::is_same_v<int64_t, std::decay_t<Type>>) {
    return EnumFromTemplateType::kUInt64T;
  } else if constexpr (std::is_same_v<uint64_t, std::decay_t<Type>>) {
    return EnumFromTemplateType::kUInt64T;
  } else if constexpr (std::is_same_v<Float16, std::decay_t<Type>>) {
    return EnumFromTemplateType::kFloat16;
  } else if constexpr (std::is_same_v<Float32, std::decay_t<Type>>) {
    return EnumFromTemplateType::kFloat32;
  } else if constexpr (std::is_same_v<Float64, std::decay_t<Type>>) {
    return EnumFromTemplateType::kFloat64;
  } else if constexpr (std::is_same_v<Float64, std::decay_t<Type>>) {
    return EnumFromTemplateType::kSIMD128Register;
  } else {
    static_assert(kDependentTypeFalse<Type>);
  }
}

template <typename Type>
constexpr EnumFromTemplateType kEnumFromTemplateType = TypeToEnumFromTemplateType<Type>();

template <enum EnumFromTemplateType>
class TemplateTypeFromEnumHelper;

#pragma push_macro("DEFINE_TEMPLATE_TYPE_FROM_ENUM")
#undef DEFINE_TEMPLATE_TYPE_FROM_ENUM
#define DEFINE_TEMPLATE_TYPE_FROM_ENUM(kEnumValue, TemplateType) \
  template <>                                                    \
  class TemplateTypeFromEnumHelper<kEnumValue> {                 \
   public:                                                       \
    using Type = TemplateType;                                   \
  }

DEFINE_TEMPLATE_TYPE_FROM_ENUM(kInt8T, int8_t);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kUInt8T, uint8_t);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kInt16T, int16_t);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kUInt16T, uint16_t);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kInt32T, int32_t);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kUInt32T, uint32_t);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kInt64T, int64_t);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kUInt64T, uint64_t);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kFloat16, Float16);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kFloat32, Float32);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kFloat64, Float64);
DEFINE_TEMPLATE_TYPE_FROM_ENUM(kSIMD128Register, SIMD128Register);

#pragma pop_macro("DEFINE_TEMPLATE_TYPE_FROM_ENUM")

template <enum EnumFromTemplateType kEnumValue>
using TemplateTypeFromEnum = TemplateTypeFromEnumHelper<kEnumValue>::Type;

// A solution for the inability to call generic implementation from specialization.
// Declaration:
//   template <typename Type,
//             int size,
//             enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
//   inline std::tuple<SIMD128Register> VectorMultiplyByScalarInt(SIMD128Register op1,
//                                                                SIMD128Register op2);
// Normal use only specifies two arguments, e.g. VectorMultiplyByScalarInt<uint32_t, 2>,
// but assembler implementation can (if SSE 4.1 is not available) do the following call:
//   return VectorMultiplyByScalarInt<uint32_t, 2, kUseCppImplementation>(in0, in1);
//
// Because PreferredIntrinsicsImplementation argument has non-default value we have call to the
// generic C-based implementation here.

enum PreferredIntrinsicsImplementation {
  kUseAssemblerImplementationIfPossible,
  kUseCppImplementation
};

}  // namespace intrinsics

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_COMMON_INTRINSICS_H_
