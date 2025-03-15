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

static_assert(kEnumFromTemplateType<int8_t> == kInt8T);
static_assert(kEnumFromTemplateType<uint8_t> == kUInt8T);
static_assert(kEnumFromTemplateType<int16_t> == kInt16T);
static_assert(kEnumFromTemplateType<uint16_t> == kUInt16T);
static_assert(kEnumFromTemplateType<int32_t> == kInt32T);
static_assert(kEnumFromTemplateType<uint32_t> == kUInt32T);
static_assert(kEnumFromTemplateType<int64_t> == kInt64T);
static_assert(kEnumFromTemplateType<uint64_t> == kUInt64T);
static_assert(kEnumFromTemplateType<Float16> == kFloat16);
static_assert(kEnumFromTemplateType<Float32> == kFloat32);
static_assert(kEnumFromTemplateType<Float64> == kFloat64);
static_assert(kEnumFromTemplateType<SIMD128Register> == kSIMD128Register);

static_assert(std::is_same_v<TemplateTypeFromEnum<kInt8T>, int8_t>);
static_assert(std::is_same_v<TemplateTypeFromEnum<kUInt8T>, uint8_t>);
static_assert(std::is_same_v<TemplateTypeFromEnum<kInt16T>, int16_t>);
static_assert(std::is_same_v<TemplateTypeFromEnum<kUInt16T>, uint16_t>);
static_assert(std::is_same_v<TemplateTypeFromEnum<kInt32T>, int32_t>);
static_assert(std::is_same_v<TemplateTypeFromEnum<kUInt32T>, uint32_t>);
static_assert(std::is_same_v<TemplateTypeFromEnum<kInt64T>, int64_t>);
static_assert(std::is_same_v<TemplateTypeFromEnum<kUInt64T>, uint64_t>);
static_assert(std::is_same_v<TemplateTypeFromEnum<kSIMD128Register>, SIMD128Register>);

static_assert(EnumFromTemplateTypeToFloat(kInt16T) == kFloat16);
static_assert(EnumFromTemplateTypeToFloat(kUInt16T) == kFloat16);
static_assert(EnumFromTemplateTypeToFloat(kInt32T) == kFloat32);
static_assert(EnumFromTemplateTypeToFloat(kUInt32T) == kFloat32);
static_assert(EnumFromTemplateTypeToFloat(kInt64T) == kFloat64);
static_assert(EnumFromTemplateTypeToFloat(kUInt64T) == kFloat64);

static_assert(EnumFromTemplateTypeToInt(kFloat16) == kUInt16T);
static_assert(EnumFromTemplateTypeToInt(kFloat32) == kUInt32T);
static_assert(EnumFromTemplateTypeToInt(kFloat64) == kUInt64T);

static_assert(EnumFromTemplateTypeToNarrow(kInt16T) == kInt8T);
static_assert(EnumFromTemplateTypeToNarrow(kUInt16T) == kUInt8T);
static_assert(EnumFromTemplateTypeToNarrow(kInt32T) == kInt16T);
static_assert(EnumFromTemplateTypeToNarrow(kUInt32T) == kUInt16T);
static_assert(EnumFromTemplateTypeToNarrow(kInt64T) == kInt32T);
static_assert(EnumFromTemplateTypeToNarrow(kUInt64T) == kUInt32T);
static_assert(EnumFromTemplateTypeToNarrow(kFloat32) == kFloat16);
static_assert(EnumFromTemplateTypeToNarrow(kFloat64) == kFloat32);

static_assert(EnumFromTemplateTypeToSigned(kInt8T) == kInt8T);
static_assert(EnumFromTemplateTypeToSigned(kUInt8T) == kInt8T);
static_assert(EnumFromTemplateTypeToSigned(kInt16T) == kInt16T);
static_assert(EnumFromTemplateTypeToSigned(kUInt16T) == kInt16T);
static_assert(EnumFromTemplateTypeToSigned(kInt32T) == kInt32T);
static_assert(EnumFromTemplateTypeToSigned(kUInt32T) == kInt32T);
static_assert(EnumFromTemplateTypeToSigned(kInt64T) == kInt64T);
static_assert(EnumFromTemplateTypeToSigned(kUInt64T) == kInt64T);

static_assert(EnumFromTemplateTypeSizeOf(kInt8T) == 1);
static_assert(EnumFromTemplateTypeSizeOf(kUInt8T) == 1);
static_assert(EnumFromTemplateTypeSizeOf(kInt16T) == 2);
static_assert(EnumFromTemplateTypeSizeOf(kUInt16T) == 2);
static_assert(EnumFromTemplateTypeSizeOf(kInt32T) == 4);
static_assert(EnumFromTemplateTypeSizeOf(kUInt32T) == 4);
static_assert(EnumFromTemplateTypeSizeOf(kInt64T) == 8);
static_assert(EnumFromTemplateTypeSizeOf(kUInt64T) == 8);
static_assert(EnumFromTemplateTypeSizeOf(kFloat16) == 2);
static_assert(EnumFromTemplateTypeSizeOf(kFloat32) == 4);
static_assert(EnumFromTemplateTypeSizeOf(kFloat64) == 8);
static_assert(EnumFromTemplateTypeSizeOf(kSIMD128Register) == 16);

static_assert(EnumFromTemplateTypeToUnsigned(kInt8T) == kUInt8T);
static_assert(EnumFromTemplateTypeToUnsigned(kUInt8T) == kUInt8T);
static_assert(EnumFromTemplateTypeToUnsigned(kInt16T) == kUInt16T);
static_assert(EnumFromTemplateTypeToUnsigned(kUInt16T) == kUInt16T);
static_assert(EnumFromTemplateTypeToUnsigned(kInt32T) == kUInt32T);
static_assert(EnumFromTemplateTypeToUnsigned(kUInt32T) == kUInt32T);
static_assert(EnumFromTemplateTypeToUnsigned(kInt64T) == kUInt64T);
static_assert(EnumFromTemplateTypeToUnsigned(kUInt64T) == kUInt64T);

static_assert(EnumFromTemplateTypeToWide(kInt8T) == kInt16T);
static_assert(EnumFromTemplateTypeToWide(kUInt8T) == kUInt16T);
static_assert(EnumFromTemplateTypeToWide(kInt16T) == kInt32T);
static_assert(EnumFromTemplateTypeToWide(kUInt16T) == kUInt32T);
static_assert(EnumFromTemplateTypeToWide(kInt32T) == kInt64T);
static_assert(EnumFromTemplateTypeToWide(kUInt32T) == kUInt64T);
static_assert(EnumFromTemplateTypeToWide(kFloat16) == kFloat32);
static_assert(EnumFromTemplateTypeToWide(kFloat32) == kFloat64);

}  // namespace berberis::intrinsics
