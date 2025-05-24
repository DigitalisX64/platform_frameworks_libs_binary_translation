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

#ifndef BERBERIS_INTRINSICS_RISCV64_TO_ALL_VECTOR_INTRINSICS_H_
#define BERBERIS_INTRINSICS_RISCV64_TO_ALL_VECTOR_INTRINSICS_H_

#include <algorithm>
#include <climits>  // CHAR_BIT
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "berberis/base/bit_util.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/intrinsics.h"        // PreferredIntrinsicsImplementation
#if defined(__aarch64__)
#include "berberis/intrinsics/common/intrinsics_float.h"
#include "berberis/intrinsics/vector_intrinsics.h"
#else
#include "berberis/intrinsics/intrinsics_float.h"  // Float32/Float64
#endif
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/type_traits.h"

namespace berberis::intrinsics {

enum class TailProcessing {
  kUndisturbed = 0,
  kAgnostic = 1,
};

enum class InactiveProcessing {
  kUndisturbed = 0,
  kAgnostic = 1,
};

enum class NoInactiveProcessing {
  kNoInactiveProcessing = 0,
};

template <typename ElementType>
[[nodiscard]] inline std::tuple<NoInactiveProcessing> FullMaskForRegister(NoInactiveProcessing) {
  return {NoInactiveProcessing{}};
}

template <typename ElementType>
[[nodiscard]] inline std::tuple<
    std::conditional_t<sizeof(ElementType) == sizeof(Int8), RawInt16, RawInt8>>
FullMaskForRegister(SIMD128Register) {
  if constexpr (sizeof(ElementType) == sizeof(uint8_t)) {
    return {{0xffff}};
  } else if constexpr (sizeof(ElementType) == sizeof(uint16_t)) {
    return {{0xff}};
  } else if constexpr (sizeof(ElementType) == sizeof(uint32_t)) {
    return {{0xf}};
  } else if constexpr (sizeof(ElementType) == sizeof(uint64_t)) {
    return {{0x3}};
  } else {
    static_assert(kDependentTypeFalse<ElementType>, "Unsupported vector element type");
  }
}

template <typename ElementType>
[[nodiscard]] inline std::tuple<NoInactiveProcessing> MaskForRegisterInSequence(
    NoInactiveProcessing,
    size_t) {
  return {NoInactiveProcessing{}};
}

template <typename ElementType>
[[nodiscard]] inline std::tuple<
    std::conditional_t<sizeof(ElementType) == sizeof(Int8), RawInt16, RawInt8>>
MaskForRegisterInSequence(SIMD128Register mask, size_t register_in_sequence) {
  if constexpr (sizeof(ElementType) == sizeof(uint8_t)) {
    return {mask.Get<RawInt16>(register_in_sequence)};
  } else if constexpr (sizeof(ElementType) == sizeof(uint16_t)) {
    return {mask.Get<RawInt8>(register_in_sequence)};
  } else if constexpr (sizeof(ElementType) == sizeof(uint32_t)) {
    return {RawInt8{TruncateTo<UInt8>(mask.Get<UInt32>(0) >> UInt64(register_in_sequence * 4)) &
                    UInt8{0b1111}}};
  } else if constexpr (sizeof(ElementType) == sizeof(uint64_t)) {
    return {RawInt8{TruncateTo<UInt8>(mask.Get<UInt32>(0) >> UInt64(register_in_sequence * 2)) &
                    UInt8{0b11}}};
  } else {
    static_assert(kDependentTypeFalse<ElementType>, "Unsupported vector element type");
  }
}

// Naïve implementation for tests.  Also used on not-x86 platforms.
[[nodiscard]] inline std::tuple<SIMD128Register> MakeBitmaskFromVlForTests(size_t vl) {
  if (vl == 128) {
    return {SIMD128Register(__int128(0))};
  } else {
    return {SIMD128Register((~__int128(0)) << vl)};
  }
}

#ifndef __x86_64__
[[nodiscard]] inline std::tuple<SIMD128Register> MakeBitmaskFromVl(size_t vl) {
  return {MakeBitmaskFromVlForTests(vl)};
}
#endif

template <typename ElementType>
[[nodiscard]] inline std::tuple<SIMD128Register> MakeBitmaskFromVl(size_t vl) {
  return MakeBitmaskFromVl(vl * sizeof(ElementType) * CHAR_BIT);
}

// Naïve implementation for tests.  Also used on not-x86 platforms.
template <typename ElementType>
[[nodiscard]] inline std::tuple<SIMD128Register> BitMaskToSimdMaskForTests(size_t mask) {
  constexpr ElementType kZeroValue = ElementType{0};
  constexpr ElementType kFillValue = ~ElementType{0};
  SIMD128Register result;
  for (size_t index = 0; index < sizeof(SIMD128Register) / sizeof(ElementType); ++index) {
    size_t bit = 1 << index;
    if (mask & bit) {
      result.Set(kFillValue, index);
    } else {
      result.Set(kZeroValue, index);
    }
  }
  return {result};
}

#if !defined(__x86_64__) && !defined(__aarch64__)
template <typename ElementType>
[[nodiscard]] inline std::tuple<SIMD128Register> BitMaskToSimdMask(size_t mask) {
  return {BitMaskToSimdMaskForTests<ElementType>(mask)};
}
#endif

// Naïve implementation for tests.  Also used on not-x86 platforms.
template <typename ElementType>
[[nodiscard]] inline std::tuple<
    std::conditional_t<sizeof(ElementType) == sizeof(Int8), RawInt16, RawInt8>>
SimdMaskToBitMaskForTests(SIMD128Register simd_mask) {
  using ResultType = std::conditional_t<sizeof(ElementType) == sizeof(Int8), UInt16, UInt8>;
  ResultType mask{0};
  constexpr ResultType kElementsCount{
      static_cast<uint8_t>(sizeof(SIMD128Register) / sizeof(ElementType))};
  for (ResultType index{0}; index < kElementsCount; index += ResultType{1}) {
    if (simd_mask.Get<ElementType>(index) != ElementType{}) {
      mask |= ResultType{1} << ResultType{index};
    }
  }
  return mask;
}

#ifndef __SSSE3__
template <typename ElementType>
[[nodiscard]] inline std::tuple<
    std::conditional_t<sizeof(ElementType) == sizeof(Int8), RawInt16, RawInt8>>
SimdMaskToBitMask(SIMD128Register simd_mask) {
  return SimdMaskToBitMaskForTests<ElementType>(simd_mask);
}
#endif

#if !defined(__aarch64__)
template <auto kElement>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorMaskedElementToForTests(
    SIMD128Register simd_mask,
    SIMD128Register result) {
  using ElementType = decltype(kElement);
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType);
  for (size_t index = 0; index < kElementsCount; ++index) {
    if (!simd_mask.Get<ElementType>(index)) {
      result.Set(kElement, index);
    }
  }
  return result;
}

#ifndef __x86_64__
template <typename ElementType>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorMaskedElementTo(SIMD128Register simd_mask,
                                                                       SIMD128Register result) {
  return VectorMaskedElementToForTests(simd_mask, result);
}
#endif

#endif

// For instructions that operate on carry bits, expands single bit from mask register
//     into vector argument
template <typename ElementType, TailProcessing vta, auto vma>
std::tuple<SIMD128Register> GetMaskVectorArgument(SIMD128Register mask, size_t index) {
  using MaskType = std::conditional_t<sizeof(ElementType) == sizeof(Int8), UInt16, UInt8>;
  auto register_mask = std::get<0>(MaskForRegisterInSequence<ElementType>(mask, index));
  return BitMaskToSimdMaskForTests<ElementType>(Int64{MaskType{register_mask}});
}

template <typename ElementType>
[[nodiscard]] inline ElementType VectorElement(SIMD128Register src, int index) {
  return src.Get<ElementType>(index);
}

template <typename ElementType>
[[nodiscard]] inline ElementType VectorElement(ElementType src, int) {
  return src;
}

template <typename ElementType>
[[nodiscard]] inline std::tuple<SIMD128Register> VMovTopHalfToBottom(SIMD128Register src) {
  return {SIMD128Register{src.Get<uint64_t>(1)}};
}

template <typename ElementType>
[[nodiscard]] inline std::tuple<SIMD128Register> VMergeBottomHalfToTop(SIMD128Register bottom,
                                                                       SIMD128Register top) {
  SIMD128Register result{bottom};
  result.Set<uint64_t>(top.Get<uint64_t>(0), 1);
  return result;
}

// Naïve implementation for tests.  Also used on not-x86 platforms.
template <auto kDefaultElement>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorBroadcastForTests() {
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof kDefaultElement;
  SIMD128Register dest;
  for (size_t index = 0; index < kElementsCount; ++index) {
    dest.Set(kDefaultElement, index);
  }
  return dest;
}

#ifndef __x86_64__
template <auto kDefaultElement>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorBroadcast() {
  return VectorBroadcastForTests<kDefaultElement>();
}
#endif

template <auto kDefaultElement, TailProcessing vta, NoInactiveProcessing = NoInactiveProcessing{}>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorMasking(SIMD128Register result,
                                                               int vstart,
                                                               int vl) {
  constexpr int kElementsCount = static_cast<int>(sizeof(SIMD128Register) / sizeof kDefaultElement);
  if (vstart < 0) {
    vstart = 0;
  }
  if (vl < 0) {
    vl = 0;
  }
  if (vl > kElementsCount) {
    vl = kElementsCount;
  }
  if constexpr (kDefaultElement == decltype(kDefaultElement){}) {
    if (vstart == 0) [[likely]] {
      if (vl != kElementsCount) [[unlikely]] {
        const auto [tail_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vl);
        result &= ~tail_bitmask;
      }
    } else if (vstart >= vl) [[unlikely]] {
      // Note: vstart <= vl here because RISC-V instructions don't alter the result if vstart >= vl.
      // But when vstart is so big that it's larger than kElementsCount and vl is also larger than
      // kElementsCount we hit that corner case and return zero if that happens.
      result = SIMD128Register{};
    } else {
      // Note: vstart < vl here because RISC-V instructions don't alter the result if vstart >= vl.
      CHECK_LT(vstart, vl);
      const auto [start_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vstart);
      const auto [tail_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vl);
      result &= start_bitmask;
      result &= ~tail_bitmask;
    }
  } else if constexpr (kDefaultElement == ~decltype(kDefaultElement){}) {
    if (vstart == 0) [[likely]] {
      if (vl != kElementsCount) [[unlikely]] {
        const auto [tail_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vl);
        result |= tail_bitmask;
      }
    } else if (vstart >= vl) [[unlikely]] {
      // Note: vstart <= vl here because RISC-V instructions don't alter the result if vstart >= vl.
      // But when vstart is so big that it's larger than kElementsCount and vl is also larger than
      // kElementsCount we hit that corner case and return zero if that happens.
      result = ~SIMD128Register{};
    } else {
      // Note: vstart < vl here because RISC-V instructions don't alter the result if vstart >= vl.
      CHECK_LT(vstart, vl);
      const auto [start_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vstart);
      const auto [tail_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vl);
      result |= ~start_bitmask;
      result |= tail_bitmask;
    }
  } else {
    const std::tuple<SIMD128Register>& dest = VectorBroadcast<kDefaultElement>();
    if (vstart == 0) [[likely]] {
      if (vl != kElementsCount) [[unlikely]] {
        const auto [tail_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vl);
        result &= ~tail_bitmask;
        result |= (std::get<0>(dest) & tail_bitmask);
      }
    } else if (vstart >= vl) [[unlikely]] {
      // Note: vstart <= vl here because RISC-V instructions don't alter the result if vstart >= vl.
      // But when vstart is so big that it's larger than kElementsCount and vl is also larger than
      // kElementsCount we hit that corner case and return dest if that happens.
      result = std::get<0>(dest);
    } else {
      const auto [start_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vstart);
      const auto [tail_bitmask] = MakeBitmaskFromVl<decltype(kDefaultElement)>(vl);
      result &= start_bitmask;
      result &= ~tail_bitmask;
      result |= (std::get<0>(dest) & (~start_bitmask | tail_bitmask));
    }
  }
  return result;
}

template <auto kDefaultElement,
          TailProcessing vta,
          auto vma = NoInactiveProcessing{},
          typename MaskType>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorMasking(SIMD128Register result,
                                                               int vstart,
                                                               int vl,
                                                               MaskType mask) {
  static_assert((std::is_same_v<decltype(vma), NoInactiveProcessing> &&
                 std::is_same_v<MaskType, NoInactiveProcessing>) ||
                (std::is_same_v<decltype(vma), InactiveProcessing> &&
                 (std::is_same_v<MaskType, RawInt8> || std::is_same_v<MaskType, RawInt16>)));
  if constexpr (std::is_same_v<decltype(vma), InactiveProcessing>) {
    const auto [simd_mask] = BitMaskToSimdMask<decltype(kDefaultElement)>(
        static_cast<typename MaskType::BaseType>(mask));
    if constexpr (kDefaultElement == ~decltype(kDefaultElement){}) {
      result |= ~simd_mask;
    } else {
      result &= simd_mask;
      if constexpr (kDefaultElement != decltype(kDefaultElement){}) {
        const std::tuple<SIMD128Register>& dest = VectorBroadcast<kDefaultElement>();
        result |= std::get<0>(dest) & ~simd_mask;
      }
    }
  }
  return VectorMasking<kDefaultElement, vta>(result, vstart, vl);
}

template <typename ElementType, TailProcessing vta, NoInactiveProcessing = NoInactiveProcessing{}>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorMasking(
    SIMD128Register dest,
    SIMD128Register result,
    int vstart,
    int vl,
    NoInactiveProcessing /*mask*/ = NoInactiveProcessing{}) {
  constexpr int kElementsCount = static_cast<int>(sizeof(SIMD128Register) / sizeof(ElementType));
  if (vstart < 0) {
    vstart = 0;
  }
  if (vl < 0) {
    vl = 0;
  }
  if (vl > kElementsCount) {
    vl = kElementsCount;
  }
  if (vstart == 0) [[likely]] {
    if (vl == kElementsCount) [[likely]] {
      return result;
    }
    const auto [tail_bitmask] = MakeBitmaskFromVl<ElementType>(vl);
    if constexpr (vta == TailProcessing::kAgnostic) {
      dest = result | tail_bitmask;
    } else {
      dest = (dest & tail_bitmask) | (result & ~tail_bitmask);
    }
  } else if (vstart < vl) [[likely]] {
    // Note: vstart <= vl here because RISC-V instructions don't alter the result if vstart >= vl.
    // But when vstart is so big that it's larger than kElementsCount and vl is also larger than
    // kElementsCount we hit that corner case and return dest if that happens.
    const auto [start_bitmask] = MakeBitmaskFromVl<ElementType>(vstart);
    const auto [tail_bitmask] = MakeBitmaskFromVl<ElementType>(vl);
    if constexpr (vta == TailProcessing::kAgnostic) {
      dest = (dest & ~start_bitmask) | (result & start_bitmask) | tail_bitmask;
    } else {
      dest = (dest & (~start_bitmask | tail_bitmask)) | (result & start_bitmask & ~tail_bitmask);
    }
  } else if constexpr (vta == TailProcessing::kAgnostic) {
    if (vstart == vl) {
      // Corners case where vstart == vl may happen because of vslideup:
      //   https://github.com/riscv/riscv-v-spec/issues/263
      const auto [tail_bitmask] = MakeBitmaskFromVl<ElementType>(vl);
      dest |= tail_bitmask;
    }
  }
  return {dest};
}

template <typename ElementType,
          TailProcessing vta,
          auto vma = NoInactiveProcessing{},
          typename MaskType = NoInactiveProcessing>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorMasking(
    SIMD128Register dest,
    SIMD128Register result,
    SIMD128Register result_mask,
    int vstart,
    int vl,
    MaskType mask = NoInactiveProcessing{}) {
  static_assert((std::is_same_v<decltype(vma), NoInactiveProcessing> &&
                 std::is_same_v<MaskType, NoInactiveProcessing>) ||
                (std::is_same_v<decltype(vma), InactiveProcessing> &&
                 (std::is_same_v<MaskType, RawInt8> || std::is_same_v<MaskType, RawInt16>)));
  if constexpr (std::is_same_v<decltype(vma), InactiveProcessing>) {
    const auto [simd_mask] =
        BitMaskToSimdMask<ElementType>(static_cast<typename MaskType::BaseType>(mask));
    if (vma == InactiveProcessing::kAgnostic) {
      result |= ~simd_mask;
    } else {
      result = (result & simd_mask) | (result_mask & ~simd_mask);
    }
  }
  return VectorMasking<ElementType, vta>(dest, result, vstart, vl);
}

template <typename ElementType, TailProcessing vta, InactiveProcessing vma, typename MaskType>
[[nodiscard]] inline std::tuple<SIMD128Register> VectorMasking(SIMD128Register dest,
                                                               SIMD128Register result,
                                                               int vstart,
                                                               int vl,
                                                               MaskType mask) {
  return VectorMasking<ElementType, vta, vma>(dest,
                                              result,
                                              /*result_mask=*/dest,
                                              vstart,
                                              vl,
                                              mask);
}

template <typename ElementType, typename... ParameterType>
inline constexpr bool kIsAllowedArgumentForVector =
    ((std::is_same_v<ParameterType, SIMD128Register> ||
      std::is_same_v<ParameterType, ElementType>) &&
     ...);

// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType, typename Lambda, typename... ParameterType>
inline std::tuple<SIMD128Register> VectorProcessing(Lambda lambda, ParameterType... parameters) {
  static_assert(kIsAllowedArgumentForVector<ElementType, ParameterType...>);
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType);
  for (size_t index = 0; index < kElementsCount; ++index) {
    result.Set(lambda(VectorElement<ElementType>(parameters, index)...), index);
  }
  return result;
}

// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType, typename Lambda, typename ResultType, typename... ParameterType>
inline std::tuple<ResultType> VectorProcessingReduce(Lambda lambda,
                                                     ResultType init,
                                                     ParameterType... parameters) {
  static_assert(std::is_same_v<ResultType, ElementType> ||
                std::is_same_v<ResultType, WideType<ElementType>>);
  static_assert(kIsAllowedArgumentForVector<ElementType, ParameterType...>);
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType);
  for (size_t index = 0; index < kElementsCount; ++index) {
    if constexpr (std::is_same_v<ResultType, WideType<ElementType>>) {
      init = lambda(init, Widen(VectorElement<ElementType>(parameters, index)...));
    } else {
      init = lambda(init, VectorElement<ElementType>(parameters, index)...);
    }
  }
  return init;
}

// SEW = 2*SEW op SEW
// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType, typename Lambda, typename ParameterType1, typename ParameterType2>
inline std::tuple<SIMD128Register> VectorArithmeticNarrowwv(Lambda lambda,
                                                            ParameterType1 src1,
                                                            ParameterType2 src2) {
  static_assert(kIsAllowedArgumentForVector<ElementType, ParameterType1, ParameterType2>);
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType) / 2;
  for (size_t index = 0; index < kElementsCount; ++index) {
    result.Set(Narrow(lambda(VectorElement<WideType<ElementType>>(src1, index),
                             Widen(VectorElement<ElementType>(src2, index)))),
               index);
  }
  return result;
}

// 2*SEW = SEW op SEW
// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType, typename Lambda, typename... ParameterType>
inline std::tuple<SIMD128Register> VectorArithmeticWidenvv(Lambda lambda,
                                                           ParameterType... parameters) {
  static_assert(kIsAllowedArgumentForVector<ElementType, ParameterType...>);
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType) / 2;
  for (size_t index = 0; index < kElementsCount; ++index) {
    result.Set(lambda(Widen(VectorElement<ElementType>(parameters, index))...), index);
  }
  return result;
}

// 2*SEW = SEW op SEW op 2*SEW
// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType,
          typename Lambda,
          typename ParameterType1,
          typename ParameterType2,
          typename ParameterType3>
inline std::tuple<SIMD128Register> VectorArithmeticWidenvvw(Lambda lambda,
                                                            ParameterType1 src1,
                                                            ParameterType2 src2,
                                                            ParameterType3 src3) {
  static_assert(
      kIsAllowedArgumentForVector<ElementType, ParameterType1, ParameterType2, ParameterType3>);
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType) / 2;
  for (size_t index = 0; index < kElementsCount; ++index) {
    result.Set(lambda(Widen(VectorElement<ElementType>(src1, index)),
                      Widen(VectorElement<ElementType>(src2, index)),
                      VectorElement<WideType<ElementType>>(src3, index)),
               index);
  }
  return result;
}

// SEW = 2*SEW op SEW
// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType, typename Lambda, typename ParameterType1, typename ParameterType2>
inline std::tuple<SIMD128Register> VectorArithmeticWidenwv(Lambda lambda,
                                                           ParameterType1 src1,
                                                           ParameterType2 src2) {
  static_assert(kIsAllowedArgumentForVector<ElementType, ParameterType1, ParameterType2>);
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType) / 2;
  for (size_t index = 0; index < kElementsCount; ++index) {
    result.Set(lambda(VectorElement<WideType<ElementType>>(src1, index),
                      Widen(VectorElement<ElementType>(src2, index))),
               index);
  }
  return result;
}

template <typename ElementType>
SIMD128Register VectorExtend(SIMD128Register src) {
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType) / 2;
  for (size_t index = 0; index < kElementsCount; ++index) {
    result.Set(Widen(VectorElement<ElementType>(src, index)), index);
  }
  return result;
}

template <typename ElementType,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vextf2(SIMD128Register src) {
  using SourceElementType = NarrowType<ElementType>;
  return {VectorExtend<SourceElementType>(src)};
}

template <typename ElementType,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vextf4(SIMD128Register src) {
  using WideSourceElementType = NarrowType<ElementType>;
  using SourceElementType = NarrowType<WideSourceElementType>;
  return {VectorExtend<WideSourceElementType>(VectorExtend<SourceElementType>(src))};
}

template <typename ElementType,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vextf8(SIMD128Register src) {
  using WideWideSourceElementType = NarrowType<ElementType>;
  return {
      VectorExtend<WideWideSourceElementType>(std::get<0>(Vextf4<WideWideSourceElementType>(src)))};
}

template <typename ElementType,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> VidvForTests(size_t index) {
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType);
  ElementType element = {static_cast<typename ElementType::BaseType>(index * kElementsCount)};
  for (size_t index = 0; index < kElementsCount; ++index) {
    result.Set(element, index);
    element += ElementType{1};
  }
  return result;
}

// Handles "slide up" for a single destination register. Effectively copies the last offset elements
// in [kElementsCount - offset, kElementsCount) of src1 followed by the first [0, kElementsCount -
// offset) elements of src2 into the result.
//
// This leaves result looking like
//
//     result = {
//         src1[kElementsCount-offset+0],
//         src1[kElementsCount-offset+1],
//         ...,
//         src1[kElementsCount-offset+(offset-1),
//         src2[0],
//         src2[1],
//         ...,
//         src2[kElementsCount-offset-1]
//     };
template <typename ElementType>
inline std::tuple<SIMD128Register> VectorSlideUp(size_t offset,
                                                 SIMD128Register src1,
                                                 SIMD128Register src2) {
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType);
  CHECK_LT(offset, kElementsCount);
  for (size_t index = 0; index < offset; ++index) {
    result.Set(VectorElement<ElementType>(src1, kElementsCount - offset + index), index);
  }
  for (size_t index = offset; index < kElementsCount; ++index) {
    result.Set(VectorElement<ElementType>(src2, index - offset), index);
  }
  return result;
}

// Handles "slide down" for a single destination register. Effectively copies the elements in
// [offset, kElementsCount) of src1 followed by the [0, kElementsCount - offset) elements of src2
// into the result.
//
// This leaves result looking like
//
//     result = {
//         [0] = src1[offset+0],
//         [1] = src1[offset+1],
//         ...,
//         [kElementsCount-offset-1] = src1[kElementsCount-1],
//         [kElementsCount-offset] = src2[0],
//         [kElementsCount-offset+1] = src2[1],
//         ...,
//         [kElementsCount-offset+(offset-1)] = src2[kElementsCount-offset-1]
//     };
template <typename ElementType>
inline std::tuple<SIMD128Register> VectorSlideDown(size_t offset,
                                                   SIMD128Register src1,
                                                   SIMD128Register src2) {
  SIMD128Register result;
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType);
  CHECK_LT(offset, kElementsCount);
  for (size_t index = 0; index < kElementsCount - offset; ++index) {
    result.Set(VectorElement<ElementType>(src1, offset + index), index);
  }
  for (size_t index = kElementsCount - offset; index < kElementsCount; ++index) {
    result.Set(VectorElement<ElementType>(src2, index - (kElementsCount - offset)), index);
  }
  return result;
}

template <enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vcpopm(SIMD128Register simd_src) {
  UInt128 src = simd_src.Get<UInt128>();
  return Popcount(src);
}

template <enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vfirstm(SIMD128Register simd_src) {
  UInt128 src = simd_src.Get<UInt128>();
  if (src == UInt128{0}) {
    return ~UInt128{0};
  }
  return CountRZero(src);
}

#ifndef __x86_64__
template <typename ElementType,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vidv(size_t index) {
  return VidvForTests<ElementType>(index);
}
#endif

template <enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vmsifm(SIMD128Register simd_src) {
  Int128 src = simd_src.Get<Int128>();
  return {(src - Int128{1}) ^ src};
}

template <enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vmsbfm(SIMD128Register simd_src) {
  Int128 src = simd_src.Get<Int128>();
  if (src == Int128{0}) {
    return {~Int128{0}};
  }
  return {std::get<0>(Vmsifm(simd_src)).Get<Int128>() >> Int128{1}};
}

template <enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vmsofm(SIMD128Register simd_src) {
  return {std::get<0>(Vmsbfm(simd_src)) ^ std::get<0>(Vmsifm(simd_src))};
}

template <typename ElementType,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register, size_t> Viotam(SIMD128Register simd_src, size_t counter) {
  constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(ElementType);
  __uint128_t src = simd_src.Get<__uint128_t>();
  SIMD128Register result;
  for (size_t index = 0; index < kElementsCount; ++index) {
    typename Wrapping<typename ElementType::BaseType>::UnsignedType value{
        static_cast<typename ElementType::BaseType>(counter)};
    result.Set(value, index);
    counter += static_cast<size_t>(src & 1);
    src >>= 1;
  }
  return {result, counter};
}

template <typename TargetElementType,
          typename SourceElementType,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
inline std::tuple<SIMD128Register> Vfcvtv(int8_t rm, int8_t frm, SIMD128Register src) {
  SIMD128Register result;
  size_t kElementsCount = std::min(sizeof(SIMD128Register) / sizeof(TargetElementType),
                                   sizeof(SIMD128Register) / sizeof(SourceElementType));
  for (size_t index = 0; index < kElementsCount; ++index) {
    if constexpr (!std::is_same_v<TargetElementType, Float16> &&
                  !std::is_same_v<TargetElementType, Float32> &&
                  !std::is_same_v<TargetElementType, Float64>) {
      result.Set(
          std::get<0>(FCvtFloatToInteger<typename TargetElementType::BaseType, SourceElementType>(
              rm, frm, src.Get<SourceElementType>(index))),
          index);
    } else if constexpr (!std::is_same_v<SourceElementType, Float16> &&
                         !std::is_same_v<SourceElementType, Float32> &&
                         !std::is_same_v<SourceElementType, Float64>) {
      result.Set(
          std::get<0>(FCvtIntegerToFloat<TargetElementType, typename SourceElementType::BaseType>(
              rm, frm, src.Get<typename SourceElementType::BaseType>(index))),
          index);
    } else {
      result.Set(std::get<0>(FCvtFloatToFloat<TargetElementType, SourceElementType>(
                     rm, frm, src.Get<SourceElementType>(index))),
                 index);
    }
  }
  return result;
}

// With wide intrinsics multiplication we may do sign-extension or zero-extension, but some
// intrinsics need mix: Signed * Unsigned. We narrow down value and then extend it again.
// Compiler is smart enough to eliminate dead code.
template <typename ElementType>
std::tuple<ElementType> WideMultiplySignedUnsigned(ElementType arg1, ElementType arg2) {
  return BitCastToUnsigned(Widen(BitCastToSigned(Narrow(arg1)))) *
         Widen(BitCastToUnsigned(Narrow(arg2)));
}

#define DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS(...) __VA_ARGS__
#define DEFINE_ARITHMETIC_INTRINSIC(Name, arithmetic, parameters, capture, arguments)             \
  template <typename ElementType,                                                                 \
            enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>       \
  inline std::tuple<SIMD128Register> Name(DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS parameters) { \
    return VectorProcessing<ElementType>(                                                         \
        [DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS capture](auto... args) {                       \
          static_assert((std::is_same_v<decltype(args), ElementType> && ...));                    \
          arithmetic;                                                                             \
        },                                                                                        \
        DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS arguments);                                     \
  }

#define DEFINE_1OP_ARITHMETIC_INTRINSIC_V(name, ...)                 \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##v, return ({ __VA_ARGS__; }); \
                              , (SIMD128Register src), (), (src))

#define DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(name, ...)                 \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##vv, return ({ __VA_ARGS__; }); \
                              , (SIMD128Register src1, SIMD128Register src2), (), (src1, src2))

#define DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(name, ...)                                             \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##vv, return ({ __VA_ARGS__; });                             \
                              ,                                                                   \
                              (SIMD128Register src1, SIMD128Register src2, SIMD128Register src3), \
                              (),                                                                 \
                              (src1, src2, src3))

#define DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(name, ...)                            \
  DEFINE_ARITHMETIC_INTRINSIC(                                                        \
      V##name##vv, return ({ __VA_ARGS__; });                                         \
      ,                                                                               \
      (int8_t csr, SIMD128Register src1, SIMD128Register src2, SIMD128Register src3), \
      (csr),                                                                          \
      (src1, src2, src3))

#define DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(name, ...)                 \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##vx, return ({ __VA_ARGS__; }); \
                              , (SIMD128Register src1, ElementType src2), (), (src1, src2))

#define DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(name, ...) \
  DEFINE_ARITHMETIC_INTRINSIC(                        \
      V##name##vx, return ({ __VA_ARGS__; });         \
      , (SIMD128Register src1, ElementType src2, SIMD128Register src3), (), (src1, src2, src3))

#define DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(name, ...)                        \
  DEFINE_ARITHMETIC_INTRINSIC(                                                    \
      V##name##vf, return ({ __VA_ARGS__; });                                     \
      ,                                                                           \
      (int8_t csr, SIMD128Register src1, ElementType src2, SIMD128Register src3), \
      (csr),                                                                      \
      (src1, src2, src3))

#define DEFINE_1OP_ARITHMETIC_INTRINSIC_X(name, ...) \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##x, return ({ __VA_ARGS__; });, (ElementType src), (), (src))

#define DEFINE_1OP_1CSR_ARITHMETIC_INTRINSIC_V(name, ...)            \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##v, return ({ __VA_ARGS__; }); \
                              , (int8_t csr, SIMD128Register src), (csr), (src))

#define DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VF(name, ...) \
  DEFINE_ARITHMETIC_INTRINSIC(                             \
      V##name##vf, return ({ __VA_ARGS__; });              \
      , (int8_t csr, SIMD128Register src1, ElementType src2), (csr), (src1, src2))

#define DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(name, ...) \
  DEFINE_ARITHMETIC_INTRINSIC(                             \
      V##name##vv, return ({ __VA_ARGS__; });              \
      , (int8_t csr, SIMD128Register src1, SIMD128Register src2), (csr), (src1, src2))

#define DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VX(name, ...) \
  DEFINE_ARITHMETIC_INTRINSIC(                             \
      V##name##vx, return ({ __VA_ARGS__; });              \
      , (int8_t csr, SIMD128Register src1, ElementType src2), (csr), (src1, src2))

#define DEFINE_ARITHMETIC_REDUCE_INTRINSIC(Name, arithmetic, parameters, capture, arguments) \
  template <typename ElementType,                                                            \
            typename ResultType = ElementType,                                               \
            enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>  \
  inline std::tuple<ResultType> Name(DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS parameters) { \
    return VectorProcessingReduce<ElementType>(                                              \
        [DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS capture](auto... args) {                  \
          static_assert((std::is_same_v<decltype(args), ResultType> && ...));                \
          arithmetic;                                                                        \
        },                                                                                   \
        DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS arguments);                                \
  }

#define DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(name, ...)                        \
  DEFINE_ARITHMETIC_REDUCE_INTRINSIC(V##name##vs, return ({ __VA_ARGS__; }); \
                                     , (ResultType init, SIMD128Register src), (), (init, src))

#define DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VS(name, ...) \
  DEFINE_ARITHMETIC_REDUCE_INTRINSIC(                      \
      Vfred##name##vs, return ({ __VA_ARGS__; });          \
      , (int8_t csr, ResultType init, SIMD128Register src), (csr), (init, src))

#define DEFINE_W_ARITHMETIC_INTRINSIC(Name, Pattern, arithmetic, parameters, capture, arguments)  \
  template <typename ElementType,                                                                 \
            enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>       \
  inline std::tuple<SIMD128Register> Name(DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS parameters) { \
    return VectorArithmetic##Pattern<ElementType>(                                                \
        [DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS capture](auto... args) {                       \
          static_assert((std::is_same_v<decltype(args), WideType<ElementType>> && ...));          \
          arithmetic;                                                                             \
        },                                                                                        \
        DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS arguments);                                     \
  }

#define DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VV(name, ...)                       \
  DEFINE_W_ARITHMETIC_INTRINSIC(Vw##name##vv, Widenvv, return ({ __VA_ARGS__; }); \
                                , (SIMD128Register src1, SIMD128Register src2), (), (src1, src2))

#define DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VV(name, ...) \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                 \
      Vfw##name##vv, Widenvv, return ({ __VA_ARGS__; });         \
      , (int8_t csr, SIMD128Register src1, SIMD128Register src2), (csr), (src1, src2))

#define DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VF(name, ...) \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                 \
      Vfw##name##vf, Widenvv, return ({ __VA_ARGS__; });         \
      , (int8_t csr, SIMD128Register src1, ElementType src2), (csr), (src1, src2))

#define DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_WV(name, ...) \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                 \
      Vfw##name##wv, Widenwv, return ({ __VA_ARGS__; });         \
      , (int8_t csr, SIMD128Register src1, SIMD128Register src2), (csr), (src1, src2))

#define DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_WF(name, ...) \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                 \
      Vfw##name##wf, Widenwv, return ({ __VA_ARGS__; });         \
      , (int8_t csr, SIMD128Register src1, ElementType src2), (csr), (src1, src2))

#define DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VVW(name, ...)              \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                          \
      Vw##name##vv, Widenvvw, return ({ __VA_ARGS__; });                  \
      ,                                                                   \
      (SIMD128Register src1, SIMD128Register src2, SIMD128Register src3), \
      (),                                                                 \
      (src1, src2, src3))

#define DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VX(name, ...)                       \
  DEFINE_W_ARITHMETIC_INTRINSIC(Vw##name##vx, Widenvv, return ({ __VA_ARGS__; }); \
                                , (SIMD128Register src1, ElementType src2), (), (src1, src2))

#define DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VXW(name, ...) \
  DEFINE_W_ARITHMETIC_INTRINSIC(                             \
      Vw##name##vx, Widenvvw, return ({ __VA_ARGS__; });     \
      , (SIMD128Register src1, ElementType src2, SIMD128Register src3), (), (src1, src2, src3))

#define DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VVW(name, ...)                     \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                                      \
      Vfw##name##vv, Widenvvw, return ({ __VA_ARGS__; });                             \
      ,                                                                               \
      (int8_t csr, SIMD128Register src1, SIMD128Register src2, SIMD128Register src3), \
      (csr),                                                                          \
      (src1, src2, src3))

#define DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VXW(name, ...)                 \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                                  \
      Vfw##name##vf, Widenvvw, return ({ __VA_ARGS__; });                         \
      ,                                                                           \
      (int8_t csr, SIMD128Register src1, ElementType src2, SIMD128Register src3), \
      (csr),                                                                      \
      (src1, src2, src3))

#define DEFINE_2OP_NARROW_ARITHMETIC_INTRINSIC_WV(name, ...)                       \
  DEFINE_W_ARITHMETIC_INTRINSIC(Vn##name##wv, Narrowwv, return ({ __VA_ARGS__; }); \
                                , (SIMD128Register src1, SIMD128Register src2), (), (src1, src2))

#define DEFINE_2OP_NARROW_ARITHMETIC_INTRINSIC_WX(name, ...)                       \
  DEFINE_W_ARITHMETIC_INTRINSIC(Vn##name##wx, Narrowwv, return ({ __VA_ARGS__; }); \
                                , (SIMD128Register src1, ElementType src2), (), (src1, src2))

#define DEFINE_2OP_1CSR_NARROW_ARITHMETIC_INTRINSIC_WV(name, ...) \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                  \
      Vn##name##wv, Narrowwv, return ({ __VA_ARGS__; });          \
      , (int8_t csr, SIMD128Register src1, SIMD128Register src2), (csr), (src1, src2))

#define DEFINE_2OP_1CSR_NARROW_ARITHMETIC_INTRINSIC_WX(name, ...) \
  DEFINE_W_ARITHMETIC_INTRINSIC(                                  \
      Vn##name##wx, Narrowwv, return ({ __VA_ARGS__; });          \
      , (int8_t csr, SIMD128Register src1, ElementType src2), (csr), (src1, src2))

#define DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VV(name, ...)                       \
  DEFINE_W_ARITHMETIC_INTRINSIC(Vw##name##vv, Widenvv, return ({ __VA_ARGS__; }); \
                                , (SIMD128Register src1, SIMD128Register src2), (), (src1, src2))

#define DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_WV(name, ...)                       \
  DEFINE_W_ARITHMETIC_INTRINSIC(Vw##name##wv, Widenwv, return ({ __VA_ARGS__; }); \
                                , (SIMD128Register src1, SIMD128Register src2), (), (src1, src2))

#define DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_WX(name, ...)                       \
  DEFINE_W_ARITHMETIC_INTRINSIC(Vw##name##wx, Widenwv, return ({ __VA_ARGS__; }); \
                                , (SIMD128Register src1, ElementType src2), (), (src1, src2))

#define DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VX(name, ...)                       \
  DEFINE_W_ARITHMETIC_INTRINSIC(Vw##name##vx, Widenvv, return ({ __VA_ARGS__; }); \
                                , (SIMD128Register src1, ElementType src2), (), (src1, src2))

DEFINE_1OP_ARITHMETIC_INTRINSIC_V(copy, auto [arg] = std::tuple{args...}; arg)
DEFINE_1OP_ARITHMETIC_INTRINSIC_X(copy, auto [arg] = std::tuple{args...}; arg)
DEFINE_1OP_ARITHMETIC_INTRINSIC_V(brev8, std::get<0>((Brev8(args...))))
DEFINE_1OP_ARITHMETIC_INTRINSIC_V(frsqrt7, RSqrtEstimate(args...))
DEFINE_1OP_ARITHMETIC_INTRINSIC_V(
    fclass,
    static_cast<typename TypeTraits<ElementType>::Int>(std::get<0>(FClass(args...))))

DEFINE_1OP_1CSR_ARITHMETIC_INTRINSIC_V(fsqrt,
                                       CanonicalizeNanTuple(FSqrt(FPFlags::DYN, csr, args...)))

DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(add, (args + ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(add, (args + ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(redsum, (args + ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(rsub, auto [arg1, arg2] = std::tuple{args...}; (arg2 - arg1))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(sub, (args - ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(sub, (args - ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(and, (args & ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(and, (args & ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(redand, (args & ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(or, (args | ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(or, (args | ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(redor, (args | ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(xor, (args ^ ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(xor, (args ^ ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(redxor, (args ^ ...))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(
    aadd,
    ElementType{std::get<0>(Aadd(csr, static_cast<typename ElementType::BaseType>(args)...))})
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VX(
    aadd,
    ElementType{std::get<0>(Aadd(csr, static_cast<typename ElementType::BaseType>(args)...))})

DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(smul, auto [arg1, arg2] = std::tuple{args...}; ElementType{
    Narrow(Saturating{std::get<0>(Roundoff(
        csr,
        static_cast<typename WideType<ElementType>::BaseType>(Widen(arg1) * Widen(arg2)),
        static_cast<typename WideType<ElementType>::BaseType>((sizeof(ElementType) * CHAR_BIT) -
                                                              1)))})})

DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VX(smul, auto [arg1, arg2] = std::tuple{args...}; ElementType{
    Narrow(Saturating{std::get<0>(Roundoff(
        csr,
        static_cast<typename WideType<ElementType>::BaseType>(Widen(arg1) * Widen(arg2)),
        static_cast<typename WideType<ElementType>::BaseType>((sizeof(ElementType) * CHAR_BIT) -
                                                              1)))})})

DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(
    ssr,
    ElementType{std::get<0>(Roundoff(csr, static_cast<typename ElementType::BaseType>(args)...))})

DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VX(
    ssr,
    ElementType{std::get<0>(Roundoff(csr, static_cast<typename ElementType::BaseType>(args)...))})

DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(fadd,
                                        CanonicalizeNanTuple(FAdd(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VF(fadd,
                                        CanonicalizeNanTuple(FAdd(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VV(
    add,
    CanonicalizeNanTuple(FAdd(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VF(
    add,
    CanonicalizeNanTuple(FAdd(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VV(
    sub,
    CanonicalizeNanTuple(FSub(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VF(
    sub,
    CanonicalizeNanTuple(FSub(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VV(
    mul,
    CanonicalizeNanTuple(FMul(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VF(
    mul,
    CanonicalizeNanTuple(FMul(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_WV(
    add,
    CanonicalizeNanTuple(FAdd(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_WF(
    add,
    CanonicalizeNanTuple(FAdd(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_WV(
    sub,
    CanonicalizeNanTuple(FSub(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_WF(
    sub,
    CanonicalizeNanTuple(FSub(FPFlags::DYN, csr, args...)))

DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(fsub,
                                        CanonicalizeNanTuple(FSub(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VF(fsub,
                                        CanonicalizeNanTuple(FSub(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VF(frsub, auto [arg1, arg2] = std::tuple{args...};
                                        CanonicalizeNanTuple(FSub(FPFlags::DYN, csr, arg2, arg1)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VS(osum,
                                        CanonicalizeNanTuple(FAdd(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VS(usum,
                                        CanonicalizeNanTuple(FAdd(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(
    asub,
    ElementType{std::get<0>(Asub(csr, static_cast<typename ElementType::BaseType>(args)...))})
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VX(
    asub,
    ElementType{std::get<0>(Asub(csr, static_cast<typename ElementType::BaseType>(args)...))})
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VF(fmul,
                                        CanonicalizeNanTuple(FMul(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(fmul,
                                        CanonicalizeNanTuple(FMul(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VF(fdiv,
                                        CanonicalizeNanTuple(FDiv(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV(fdiv,
                                        CanonicalizeNanTuple(FDiv(FPFlags::DYN, csr, args...)))
DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VF(frdiv, auto [arg1, arg2] = std::tuple{args...};
                                        CanonicalizeNanTuple(FDiv(FPFlags::DYN, csr, arg2, arg1)))
// SIMD mask either includes results with all bits set to 0 or all bits set to 1.
// This way it may be used with VAnd and VAndN operations to perform masking.
// Such comparison is effectively one instruction of x86-64 (via SSE or AVX) but
// to achieve it we need to multiply bool result by (~IntType{0}) or (~ElementType{0}).
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(feq, using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(std::get<0>(Feq(args...))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(feq, using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(std::get<0>(Feq(args...))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(fne, using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(!std::get<0>(Feq(args...))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fne, using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(!std::get<0>(Feq(args...))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(flt, using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(std::get<0>(Flt(args...))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(flt, using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(std::get<0>(Flt(args...))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(fle, using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(std::get<0>(Fle(args...))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fle, using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(std::get<0>(Fle(args...))))
// Note: for floating point numbers Flt(b, a) and !Fle(a, b) produce different and incompatible
// results. IEEE754-2008 defined NOT (!=) predicate as negation of EQ (==) predicate while GT (>)
// and GE (>=) are not negations of LE (<) or GT (<=) predicated but instead use swap of arguments.
// Note that scalar form includes only three predicates (Feq, Fle, Fgt) while vector form includes
// Vmfgt.vf and Vmfge.vf instructions only for vector+scalar case (vector+vector case is supposed
// to be handled by swapping arguments). More here: https://github.com/riscv/riscv-v-spec/issues/300
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fgt, auto [arg1, arg2] = std::tuple{args...};
                                   using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(std::get<0>(Flt(arg2, arg1))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fge, auto [arg1, arg2] = std::tuple{args...};
                                   using IntType = typename TypeTraits<ElementType>::Int;
                                   (~IntType{0}) * IntType(std::get<0>(Fle(arg2, arg1))))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(adc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (arg2 + arg1 - arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(adc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (arg2 + arg1 - arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(sbc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (arg2 - arg1 + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(sbc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (arg2 - arg1 + arg3))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(
    seq,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args == ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(
    seq,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args == ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(
    sne,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args != ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(
    sne,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args != ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(
    slt,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args < ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(
    slt,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args < ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(
    sle,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args <= ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(
    sle,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args <= ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(
    sgt,
    (~ElementType{0}) * ElementType{static_cast<typename ElementType::BaseType>((args > ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(sl, auto [arg1, arg2] = std::tuple{args...}; (arg1 << arg2))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(sl, auto [arg1, arg2] = std::tuple{args...}; (arg1 << arg2))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(sr, auto [arg1, arg2] = std::tuple{args...}; (arg1 >> arg2))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(sr, auto [arg1, arg2] = std::tuple{args...}; (arg1 >> arg2))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(macc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   ((arg2 * arg1) + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(macc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   ((arg2 * arg1) + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(nmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (-(arg2 * arg1) + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(nmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (-(arg2 * arg1) + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(madd, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   ((arg2 * arg3) + arg1))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(madd, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   ((arg2 * arg3) + arg1))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(nmsub, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (-(arg2 * arg3) + arg1))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(nmsub, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (-(arg2 * arg3) + arg1))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(fmacc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FMAdd(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(fmacc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FMAdd(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(fmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FMSub(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(fmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FMSub(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(fmadd, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FMAdd(FPFlags::DYN, csr, arg3, arg2, arg1)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(fmadd, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FMAdd(FPFlags::DYN, csr, arg3, arg2, arg1)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(fmsub, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FMSub(FPFlags::DYN, csr, arg3, arg2, arg1)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(fmsub, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FMSub(FPFlags::DYN, csr, arg3, arg2, arg1)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(fnmacc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FNMSub(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(fnmacc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FNMSub(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(fnmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FNMAdd(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(fnmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FNMAdd(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(fnmadd, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FNMSub(FPFlags::DYN, csr, arg3, arg2, arg1)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(fnmadd, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FNMSub(FPFlags::DYN, csr, arg3, arg2, arg1)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV(fnmsub, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FNMAdd(FPFlags::DYN, csr, arg3, arg2, arg1)))
DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF(fnmsub, auto [arg1, arg2, arg3] = std::tuple{args...};
                                        std::get<0>(FNMAdd(FPFlags::DYN, csr, arg3, arg2, arg1)))

DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(fmin, std::get<0>(FMin(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fmin, std::get<0>(FMin(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(fredmin, std::get<0>(FMin(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(fmax, std::get<0>(FMax(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fmax, std::get<0>(FMax(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(fredmax, std::get<0>(FMax(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(fsgnj, std::get<0>(FSgnj(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fsgnj, std::get<0>(FSgnj(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(fsgnjn, std::get<0>(FSgnjn(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fsgnjn, std::get<0>(FSgnjn(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(fsgnjx, std::get<0>(FSgnjx(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(fsgnjx, std::get<0>(FSgnjx(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(min, std::min(args...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(min, std::min(args...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(redmin, std::min(args...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(max, std::max(args...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(max, std::max(args...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VS(redmax, std::max(args...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(mul, auto [arg1, arg2] = std::tuple{args...}; (arg2 * arg1))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(mul, auto [arg1, arg2] = std::tuple{args...}; (arg2 * arg1))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(mulh, auto [arg1, arg2] = std::tuple{args...};
                                   NarrowTopHalf(Widen(arg2) * Widen(arg1)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(mulh, auto [arg1, arg2] = std::tuple{args...};
                                   NarrowTopHalf(Widen(arg2) * Widen(arg1)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(mulhsu, auto [arg1, arg2] = std::tuple{args...};
                                   NarrowTopHalf(BitCastToUnsigned(Widen(BitCastToSigned(arg1))) *
                                                 Widen(BitCastToUnsigned(arg2))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(mulhsu, auto [arg1, arg2] = std::tuple{args...};
                                   NarrowTopHalf(BitCastToUnsigned(Widen(BitCastToSigned(arg1))) *
                                                 Widen(BitCastToUnsigned(arg2))))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(
    div,
    ElementType{std::get<0>(DivRiscV(static_cast<typename ElementType::BaseType>(args)...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(
    div,
    ElementType{std::get<0>(DivRiscV(static_cast<typename ElementType::BaseType>(args)...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(
    rem,
    ElementType{std::get<0>(RemRiscV(static_cast<typename ElementType::BaseType>(args)...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(
    rem,
    ElementType{std::get<0>(RemRiscV(static_cast<typename ElementType::BaseType>(args)...))})

DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VV(add, (args + ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VX(add, (args + ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_WV(add, (args + ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_WX(add, (args + ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VV(sub, (args - ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VX(sub, (args - ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_WV(sub, (args - ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_WX(sub, (args - ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VV(mul, (args * ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VV(mulsu, std::get<0>(WideMultiplySignedUnsigned(args...)))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VX(mul, (args * ...))
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VX(mulsu, std::get<0>(WideMultiplySignedUnsigned(args...)))

DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VVW(macc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                          (arg1 * arg2) + arg3)
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VXW(macc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                          (arg1 * arg2) + arg3)
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VVW(maccsu, auto [arg1, arg2, arg3] = std::tuple{args...};
                                          (std::get<0>(WideMultiplySignedUnsigned(arg2, arg1))) +
                                          arg3)
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VXW(maccsu, auto [arg1, arg2, arg3] = std::tuple{args...};
                                          (std::get<0>(WideMultiplySignedUnsigned(arg2, arg1))) +
                                          arg3)
DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VXW(maccus, auto [arg1, arg2, arg3] = std::tuple{args...};
                                          (std::get<0>(WideMultiplySignedUnsigned(arg1, arg2))) +
                                          arg3)
DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VVW(
    macc, auto [arg1, arg2, arg3] = std::tuple{args...};
    std::get<0>(FMAdd(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VXW(
    macc, auto [arg1, arg2, arg3] = std::tuple{args...};
    std::get<0>(FMAdd(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VVW(
    nmacc, auto [arg1, arg2, arg3] = std::tuple{args...};
    std::get<0>(FNMSub(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VXW(
    nmacc, auto [arg1, arg2, arg3] = std::tuple{args...};
    std::get<0>(FNMSub(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VVW(
    msac, auto [arg1, arg2, arg3] = std::tuple{args...};
    std::get<0>(FMSub(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VXW(
    msac, auto [arg1, arg2, arg3] = std::tuple{args...};
    std::get<0>(FMSub(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VVW(
    nmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
    std::get<0>(FNMAdd(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VXW(
    nmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
    std::get<0>(FNMAdd(FPFlags::DYN, csr, arg2, arg1, arg3)))
DEFINE_2OP_NARROW_ARITHMETIC_INTRINSIC_WV(sr, auto [arg1, arg2] = std::tuple{args...};
                                          (arg1 >> arg2))
DEFINE_2OP_NARROW_ARITHMETIC_INTRINSIC_WX(sr, auto [arg1, arg2] = std::tuple{args...};
                                          (arg1 >> arg2))
DEFINE_2OP_1CSR_NARROW_ARITHMETIC_INTRINSIC_WV(
    clip,
    WideType<ElementType>{(std::get<0>(
        Roundoff(csr, static_cast<typename WideType<ElementType>::BaseType>(args)...)))})
DEFINE_2OP_1CSR_NARROW_ARITHMETIC_INTRINSIC_WX(
    clip,
    WideType<ElementType>{(std::get<0>(
        Roundoff(csr, static_cast<typename WideType<ElementType>::BaseType>(args)...)))})

#undef DEFINE_ARITHMETIC_INTRINSIC
#undef DEFINE_W_ARITHMETIC_INTRINSIC
#undef DEFINE_ARITHMETIC_REDUCE_INTRINSIC
#undef DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS
#undef DEFINE_1OP_ARITHMETIC_INTRINSIC_V
#undef DEFINE_2OP_ARITHMETIC_INTRINSIC_VS
#undef DEFINE_2OP_ARITHMETIC_INTRINSIC_VV
#undef DEFINE_3OP_ARITHMETIC_INTRINSIC_VV
#undef DEFINE_2OP_ARITHMETIC_INTRINSIC_VX
#undef DEFINE_3OP_ARITHMETIC_INTRINSIC_VX
#undef DEFINE_1OP_ARITHMETIC_INTRINSIC_X
#undef DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VF
#undef DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VS
#undef DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VX
#undef DEFINE_2OP_1CSR_ARITHMETIC_INTRINSIC_VV
#undef DEFINE_2OP_NARROW_ARITHMETIC_INTRINSIC_WV
#undef DEFINE_2OP_NARROW_ARITHMETIC_INTRINSIC_WX
#undef DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VV
#undef DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VF
#undef DEFINE_3OP_1CSR_ARITHMETIC_INTRINSIC_VV
#undef DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VV
#undef DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VF
#undef DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_WV
#undef DEFINE_2OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_WF
#undef DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VVW
#undef DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_WV
#undef DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_WX
#undef DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VX
#undef DEFINE_2OP_WIDEN_ARITHMETIC_INTRINSIC_VXW
#undef DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VVW
#undef DEFINE_3OP_1CSR_WIDEN_ARITHMETIC_INTRINSIC_VXW

}  // namespace berberis::intrinsics

#endif  // BERBERIS_INTRINSICS_RISCV64_TO_ALL_VECTOR_INTRINSICS_H_
