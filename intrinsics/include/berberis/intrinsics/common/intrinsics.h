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

// Value that's passed as argument of function or lambda couldn't be constexpr, but if it's
// passed as part of argument type then it's different.
// Class Value is empty, but carries the required information in its type.
// It can also be automatically converted into value of the specified type when needed.
// That way we can pass argument into a template as normal, non-template argument.
template <auto ValueParam>
class Value {
 public:
  using ValueType = std::remove_cvref_t<decltype(ValueParam)>;
  static constexpr auto kValue = ValueParam;
  constexpr operator ValueType() const { return kValue; }
};

enum TemplateTypeId : uint8_t {
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

constexpr TemplateTypeId TemplateTypeIdToFloat(TemplateTypeId value) {
  DCHECK(value >= kUInt16T && value <= kInt64T);
  return TemplateTypeId{static_cast<uint8_t>((value & 0x6) + 8)};
}

constexpr TemplateTypeId TemplateTypeIdToInt(TemplateTypeId value) {
  DCHECK((value >= kFloat16 && value <= kFloat64) && !(value & 1));
  return TemplateTypeId{static_cast<uint8_t>(value - 8)};
}

constexpr TemplateTypeId TemplateTypeIdToNarrow(TemplateTypeId value) {
  DCHECK((value >= kUInt16T && value <= kInt64T) ||
         ((value >= kFloat32 && value <= kFloat64) && !(value & 1)));
  return TemplateTypeId{static_cast<uint8_t>(value - 2)};
}

constexpr TemplateTypeId TemplateTypeIdToSigned(TemplateTypeId value) {
  DCHECK(value <= kInt64T);
  return TemplateTypeId{static_cast<uint8_t>(value | 1)};
}

constexpr int TemplateTypeIdSizeOf(TemplateTypeId value) {
  if (value == kSIMD128Register) {
    return 16;
  }
  return 1 << ((value & 0b110) >> 1);
}

constexpr TemplateTypeId TemplateTypeIdToUnsigned(TemplateTypeId value) {
  DCHECK(value <= kInt64T);
  return TemplateTypeId{static_cast<uint8_t>(value & ~1)};
}

constexpr TemplateTypeId TemplateTypeIdToWide(TemplateTypeId value) {
  DCHECK(value <= kInt32T || ((value >= kFloat16 && value <= kFloat32) && !(value & 1)));
  return TemplateTypeId{static_cast<uint8_t>(value + 2)};
}

template <typename Type>
constexpr TemplateTypeId IdFromType() {
  if constexpr (std::is_same_v<int8_t, std::decay_t<Type>>) {
    return TemplateTypeId::kInt8T;
  } else if constexpr (std::is_same_v<uint8_t, std::decay_t<Type>>) {
    return TemplateTypeId::kUInt8T;
  } else if constexpr (std::is_same_v<int16_t, std::decay_t<Type>>) {
    return TemplateTypeId::kInt16T;
  } else if constexpr (std::is_same_v<uint16_t, std::decay_t<Type>>) {
    return TemplateTypeId::kUInt16T;
  } else if constexpr (std::is_same_v<int32_t, std::decay_t<Type>>) {
    return TemplateTypeId::kInt32T;
  } else if constexpr (std::is_same_v<uint32_t, std::decay_t<Type>>) {
    return TemplateTypeId::kUInt32T;
  } else if constexpr (std::is_same_v<int64_t, std::decay_t<Type>>) {
    return TemplateTypeId::kInt64T;
  } else if constexpr (std::is_same_v<uint64_t, std::decay_t<Type>>) {
    return TemplateTypeId::kUInt64T;
  } else if constexpr (std::is_same_v<Float16, std::decay_t<Type>>) {
    return TemplateTypeId::kFloat16;
  } else if constexpr (std::is_same_v<Float32, std::decay_t<Type>>) {
    return TemplateTypeId::kFloat32;
  } else if constexpr (std::is_same_v<Float64, std::decay_t<Type>>) {
    return TemplateTypeId::kFloat64;
  } else if constexpr (std::is_same_v<SIMD128Register, std::decay_t<Type>>) {
    return TemplateTypeId::kSIMD128Register;
  } else {
    static_assert(kDependentTypeFalse<Type>);
  }
}

template <typename Type>
constexpr TemplateTypeId kIdFromType = IdFromType<Type>();

constexpr TemplateTypeId IntSizeToTemplateTypeId(uint8_t size, bool is_signed = false) {
  DCHECK(std::has_single_bit(size));
  DCHECK(size < 16);
  return static_cast<TemplateTypeId>((std::countr_zero(size) << 1) + is_signed);
}

template <enum TemplateTypeId>
class TypeFromIdHelper;

#pragma push_macro("DEFINE_TEMPLATE_TYPE_FROM_ENUM")
#undef DEFINE_TEMPLATE_TYPE_FROM_ENUM
#define DEFINE_TEMPLATE_TYPE_FROM_ENUM(kEnumValue, TemplateType) \
  template <>                                                    \
  class TypeFromIdHelper<kEnumValue> {                           \
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

template <enum TemplateTypeId kEnumValue>
using TypeFromId = TypeFromIdHelper<kEnumValue>::Type;

// If we carry TemplateTypeId then we can do the exact same manipulations wuth it as with
// normal value, but also can get actual type from it and do appropriate operations:
// make signed, make unsigned, widen, narrow, etc.
template <TemplateTypeId ValueParam>
class Value<ValueParam> {
 public:
  using Type = TypeFromId<ValueParam>;
  using ValueType = TemplateTypeId;
  static constexpr auto kValue = ValueParam;
  constexpr operator TemplateTypeId() const { return kValue; }
};

#pragma push_macro("DEFINE_VALUE_FUNCTION")
#undef DEFINE_VALUE_FUNCTION
#define DEFINE_VALUE_FUNCTION(FunctionName)                                   \
  template <TemplateTypeId ValueParam>                                        \
  constexpr Value<FunctionName(ValueParam)> FunctionName(Value<ValueParam>) { \
    return {};                                                                \
  }

DEFINE_VALUE_FUNCTION(TemplateTypeIdToFloat)
DEFINE_VALUE_FUNCTION(TemplateTypeIdToInt)
DEFINE_VALUE_FUNCTION(TemplateTypeIdToNarrow)
DEFINE_VALUE_FUNCTION(TemplateTypeIdToSigned)
DEFINE_VALUE_FUNCTION(TemplateTypeIdSizeOf)
DEFINE_VALUE_FUNCTION(TemplateTypeIdToUnsigned)
DEFINE_VALUE_FUNCTION(TemplateTypeIdToWide)

#pragma pop_macro("DEFINE_VALUE_FUNCTION")

#pragma push_macro("DEFINE_VALUE_OPERATOR")
#undef DEFINE_VALUE_OPERATOR
#define DEFINE_VALUE_OPERATOR(operator_name)                                       \
  template <auto ValueParam1, auto ValueParam2>                                    \
  constexpr Value<(ValueParam1 operator_name ValueParam2)> operator operator_name( \
      Value<ValueParam1>, Value<ValueParam2>) {                                    \
    return {};                                                                     \
  }

DEFINE_VALUE_OPERATOR(+)
DEFINE_VALUE_OPERATOR(-)
DEFINE_VALUE_OPERATOR(*)
DEFINE_VALUE_OPERATOR(/)
DEFINE_VALUE_OPERATOR(<<)
DEFINE_VALUE_OPERATOR(>>)
DEFINE_VALUE_OPERATOR(==)
DEFINE_VALUE_OPERATOR(!=)
DEFINE_VALUE_OPERATOR(>)
DEFINE_VALUE_OPERATOR(<)
DEFINE_VALUE_OPERATOR(<=)
DEFINE_VALUE_OPERATOR(>=)
DEFINE_VALUE_OPERATOR(&&)
DEFINE_VALUE_OPERATOR(||)

#pragma pop_macro("DEFINE_VALUE_OPERATOR")

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
