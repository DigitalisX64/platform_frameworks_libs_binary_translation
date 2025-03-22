/*
 * Copyright (C) 2013 The Android Open Source Project
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

#include "gtest/gtest.h"

#include <cstdint>
#include <type_traits>

#include "berberis/intrinsics/common/intrinsics.h"
#include "berberis/intrinsics/simd_register.h"

namespace berberis::intrinsics {

static_assert(kIdFromType<int8_t> == kInt8T);
static_assert(kIdFromType<uint8_t> == kUInt8T);
static_assert(kIdFromType<int16_t> == kInt16T);
static_assert(kIdFromType<uint16_t> == kUInt16T);
static_assert(kIdFromType<int32_t> == kInt32T);
static_assert(kIdFromType<uint32_t> == kUInt32T);
static_assert(kIdFromType<int64_t> == kInt64T);
static_assert(kIdFromType<uint64_t> == kUInt64T);
static_assert(kIdFromType<Float16> == kFloat16);
static_assert(kIdFromType<Float32> == kFloat32);
static_assert(kIdFromType<Float64> == kFloat64);
static_assert(kIdFromType<SIMD128Register> == kSIMD128Register);

static_assert(std::is_same_v<TypeFromId<kInt8T>, int8_t>);
static_assert(std::is_same_v<TypeFromId<kUInt8T>, uint8_t>);
static_assert(std::is_same_v<TypeFromId<kInt16T>, int16_t>);
static_assert(std::is_same_v<TypeFromId<kUInt16T>, uint16_t>);
static_assert(std::is_same_v<TypeFromId<kInt32T>, int32_t>);
static_assert(std::is_same_v<TypeFromId<kUInt32T>, uint32_t>);
static_assert(std::is_same_v<TypeFromId<kInt64T>, int64_t>);
static_assert(std::is_same_v<TypeFromId<kUInt64T>, uint64_t>);
static_assert(std::is_same_v<TypeFromId<kSIMD128Register>, SIMD128Register>);

static_assert(TemplateTypeIdToFloat(kInt16T) == kFloat16);
static_assert(TemplateTypeIdToFloat(kUInt16T) == kFloat16);
static_assert(TemplateTypeIdToFloat(kInt32T) == kFloat32);
static_assert(TemplateTypeIdToFloat(kUInt32T) == kFloat32);
static_assert(TemplateTypeIdToFloat(kInt64T) == kFloat64);
static_assert(TemplateTypeIdToFloat(kUInt64T) == kFloat64);

static_assert(TemplateTypeIdToInt(kFloat16) == kUInt16T);
static_assert(TemplateTypeIdToInt(kFloat32) == kUInt32T);
static_assert(TemplateTypeIdToInt(kFloat64) == kUInt64T);

static_assert(TemplateTypeIdToNarrow(kInt16T) == kInt8T);
static_assert(TemplateTypeIdToNarrow(kUInt16T) == kUInt8T);
static_assert(TemplateTypeIdToNarrow(kInt32T) == kInt16T);
static_assert(TemplateTypeIdToNarrow(kUInt32T) == kUInt16T);
static_assert(TemplateTypeIdToNarrow(kInt64T) == kInt32T);
static_assert(TemplateTypeIdToNarrow(kUInt64T) == kUInt32T);
static_assert(TemplateTypeIdToNarrow(kFloat32) == kFloat16);
static_assert(TemplateTypeIdToNarrow(kFloat64) == kFloat32);

static_assert(TemplateTypeIdToSigned(kInt8T) == kInt8T);
static_assert(TemplateTypeIdToSigned(kUInt8T) == kInt8T);
static_assert(TemplateTypeIdToSigned(kInt16T) == kInt16T);
static_assert(TemplateTypeIdToSigned(kUInt16T) == kInt16T);
static_assert(TemplateTypeIdToSigned(kInt32T) == kInt32T);
static_assert(TemplateTypeIdToSigned(kUInt32T) == kInt32T);
static_assert(TemplateTypeIdToSigned(kInt64T) == kInt64T);
static_assert(TemplateTypeIdToSigned(kUInt64T) == kInt64T);

static_assert(TemplateTypeIdSizeOf(kInt8T) == 1);
static_assert(TemplateTypeIdSizeOf(kUInt8T) == 1);
static_assert(TemplateTypeIdSizeOf(kInt16T) == 2);
static_assert(TemplateTypeIdSizeOf(kUInt16T) == 2);
static_assert(TemplateTypeIdSizeOf(kInt32T) == 4);
static_assert(TemplateTypeIdSizeOf(kUInt32T) == 4);
static_assert(TemplateTypeIdSizeOf(kInt64T) == 8);
static_assert(TemplateTypeIdSizeOf(kUInt64T) == 8);
static_assert(TemplateTypeIdSizeOf(kFloat16) == 2);
static_assert(TemplateTypeIdSizeOf(kFloat32) == 4);
static_assert(TemplateTypeIdSizeOf(kFloat64) == 8);
static_assert(TemplateTypeIdSizeOf(kSIMD128Register) == 16);

static_assert(TemplateTypeIdToUnsigned(kInt8T) == kUInt8T);
static_assert(TemplateTypeIdToUnsigned(kUInt8T) == kUInt8T);
static_assert(TemplateTypeIdToUnsigned(kInt16T) == kUInt16T);
static_assert(TemplateTypeIdToUnsigned(kUInt16T) == kUInt16T);
static_assert(TemplateTypeIdToUnsigned(kInt32T) == kUInt32T);
static_assert(TemplateTypeIdToUnsigned(kUInt32T) == kUInt32T);
static_assert(TemplateTypeIdToUnsigned(kInt64T) == kUInt64T);
static_assert(TemplateTypeIdToUnsigned(kUInt64T) == kUInt64T);

static_assert(TemplateTypeIdToWide(kInt8T) == kInt16T);
static_assert(TemplateTypeIdToWide(kUInt8T) == kUInt16T);
static_assert(TemplateTypeIdToWide(kInt16T) == kInt32T);
static_assert(TemplateTypeIdToWide(kUInt16T) == kUInt32T);
static_assert(TemplateTypeIdToWide(kInt32T) == kInt64T);
static_assert(TemplateTypeIdToWide(kUInt32T) == kUInt64T);
static_assert(TemplateTypeIdToWide(kFloat16) == kFloat32);
static_assert(TemplateTypeIdToWide(kFloat32) == kFloat64);

}  // namespace berberis::intrinsics
