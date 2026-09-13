/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef BERBERIS_GUEST_ABI_GUEST_PARAMS_ARCH_H_
#define BERBERIS_GUEST_ABI_GUEST_PARAMS_ARCH_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "berberis/base/dependent_false.h"
#include "berberis/calling_conventions/calling_conventions_arm64.h"
#include "berberis/guest_abi/guest_abi.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

class GuestVAListParams;

// End of workaroud

// region digitalis
// Integer and enum results narrower than 64 bits.
//
// AAPCS64 leaves the unused bits of the result register unspecified, but the code
// guest callers are actually compiled from does not treat them that way. clang and
// gcc materialise such a result with a w-register write (`mov w0, #v`), which leaves
// bool and unsigned values zero-extended and signed 8- and 16-bit values sign-extended
// to 32 bits, with the upper half of x0 clear. Callers depend on it: rustc emits
// `cbz w0` after a `jboolean ExceptionCheck()`. Writing only sizeof(T) bytes into x0
// left the remaining bytes holding whatever x0 carried on entry -- the first argument,
// e.g. a JNIEnv* -- so a host `false` read back as true in the guest.
//
// Return() hands out this slot instead of the raw byte-wide location so every write
// produces the value a real w-register write would.
template <typename Type, bool = std::is_enum_v<Type>>
struct GuestNarrowResultRepr {
  using type = Type;
};

template <typename Type>
struct GuestNarrowResultRepr<Type, true> {
  using type = std::underlying_type_t<Type>;
};

template <typename Type>
inline constexpr bool kIsGuestNarrowIntegerResult =
    (std::is_integral_v<Type> || std::is_enum_v<Type>) && sizeof(Type) < sizeof(uint64_t);

template <typename Type>
class GuestNarrowIntegerResult {
 public:
  using Repr = typename GuestNarrowResultRepr<Type>::type;

  GuestNarrowIntegerResult& operator=(Type value) {
    reg_ = Extend(static_cast<Repr>(value));
    return *this;
  }
  GuestNarrowIntegerResult& operator=(const GuestType<Type>& value) {
    return *this = static_cast<Type>(value);
  }
  operator Type() const { return static_cast<Type>(static_cast<Repr>(reg_)); }

 private:
  static constexpr uint64_t Extend(Repr value) {
    if constexpr (std::is_signed_v<Repr>) {
      return static_cast<uint32_t>(static_cast<int32_t>(value));
    } else {
      return static_cast<uint64_t>(value);
    }
  }

  uint64_t reg_;
};
// endregion

class GuestParamsAndReturnHelper : protected GuestAbi {
 protected:
  template <typename Type>
  constexpr static auto* ParamLocationAddress(uint64_t* x,
                                              __uint128_t* v,
                                              uint8_t* s,
                                              arm64::ArgLocation loc) {
    void* address = nullptr;
    if (loc.kind == arm64::kArgLocationStack) {
      address = s + loc.offset;
    } else if (loc.kind == arm64::kArgLocationInt) {
      address = x + loc.offset;
    } else if (loc.kind == arm64::kArgLocationSimd) {
      address = v + loc.offset;
    } else {
      LOG_ALWAYS_FATAL("Unknown ArgumentKind");
    }

    static_assert(!std::is_same_v<Type, void>);
    if constexpr (GuestArgumentInfo<Type>::kArgumentClass == ArgumentClass::kLargeStructType) {
      return *reinterpret_cast<typename GuestArgumentInfo<Type>::GuestType**>(address);
    } else {
      return reinterpret_cast<typename GuestArgumentInfo<Type>::GuestType*>(address);
    }
  }
};

// GuestArguments is typesafe wrapper around GuestArgumentBuffer.
// Usage looks like this:
//   GuestParamsAndReturn<double(int, double, int, double)> params(state);
//   int x = params.Params<0>();
//   float y = params.Params<1>();
//   params.Return() = x * y;

template <typename, GuestAbi::CallingConventionsVariant = GuestAbi::kAapcs64>
class GuestParamsAndReturn;  // Not defined.

template <typename ReturnType, typename... ParamType, bool kNoexcept>
class GuestParamsAndReturn<ReturnType(ParamType...) noexcept(kNoexcept), GuestAbi::kAapcs64>
    : GuestParamsAndReturnHelper {
 public:
  GuestParamsAndReturn(ThreadState* state)
      : x_(state->cpu.x), v_(state->cpu.v), s_(ToHostAddr<uint8_t>(state->cpu.sp)) {}

  template <size_t index>
  auto* Params() const {
    static_assert(index < sizeof...(ParamType));
    return this->ParamLocationAddress<std::tuple_element_t<index, std::tuple<ParamType...>>>(
        x_, v_, s_, kParamsLocations[index]);
  }

  auto* Return() const {
    // region digitalis - narrow integer results fill all of x0; see GuestNarrowIntegerResult.
    // The brace-less else keeps the upstream return below as the else-branch, unchanged.
    if constexpr (kIsGuestNarrowIntegerResult<ReturnType>) {
      return reinterpret_cast<GuestNarrowIntegerResult<ReturnType>*>(x_ + kReturnLocation.offset);
    } else
    // endregion
    return this->ParamLocationAddress<ReturnType>(x_, v_, s_, kReturnLocation);
  }

 private:
  friend class GuestVAListParams;
  friend class GuestParamsAndReturn<ReturnType(ParamType..., ...) noexcept(kNoexcept),
                                    GuestAbi::kAapcs64>;

  constexpr static const std::tuple<arm64::CallingConventions,
                                    std::array<arm64::ArgLocation, sizeof...(ParamType)>>
  ParamsInfoHelper() {
    struct {
      const ArgumentClass kArgumentClass;
      const unsigned kSize;
      const unsigned kAlignment;
    } const kArgumentsInfo[] = {{.kArgumentClass = GuestArgumentInfo<ParamType>::kArgumentClass,
                                 .kSize = GuestArgumentInfo<ParamType>::kSize,
                                 .kAlignment = GuestArgumentInfo<ParamType>::kAlignment}...};

    arm64::CallingConventions conv;
    std::array<arm64::ArgLocation, sizeof...(ParamType)> result{};
    for (const auto& kArgInfo : kArgumentsInfo) {
      if (kArgInfo.kArgumentClass == ArgumentClass::kInteger ||
          kArgInfo.kArgumentClass == ArgumentClass::kLargeStructType) {
        result[&kArgInfo - kArgumentsInfo] =
            conv.GetNextIntArgLoc(kArgInfo.kSize, kArgInfo.kAlignment);
      } else if (kArgInfo.kArgumentClass == ArgumentClass::kVFP) {
        result[&kArgInfo - kArgumentsInfo] =
            conv.GetNextFpArgLoc(kArgInfo.kSize, kArgInfo.kAlignment);
      } else {
        LOG_ALWAYS_FATAL("Unsupported ArgumentClass");
      }
    }

    return {conv, result};
  }

  constexpr static arm64::ArgLocation ResultInfoHelper() {
    arm64::CallingConventions conv;
    using ReturnInfo = GuestArgumentInfo<ReturnType>;
    if constexpr (std::is_same_v<ReturnType, void>) {
      return {arm64::kArgLocationNone, 0};
    } else if constexpr (ReturnInfo::kArgumentClass == ArgumentClass::kInteger) {
      return conv.GetIntResLoc(ReturnInfo::kSize);
    } else if constexpr (ReturnInfo::kArgumentClass == ArgumentClass::kVFP) {
      return conv.GetFpResLoc(ReturnInfo::kSize);
    } else if constexpr (ReturnInfo::kArgumentClass == ArgumentClass::kLargeStructType) {
      static_assert(GuestArgumentInfo<ReturnType>::kSize == 8);
      static_assert(GuestArgumentInfo<ReturnType>::kAlignment == 8);
      // Note: neither arm64::CallingConventions nor GuestArgumentBuffer have support for it ATM.
      // Handle it here till that would be fixed.
      return {arm64::kArgLocationInt, 8};  // Note: that's x8 register, not size or alignment.
    } else {
      static_assert(kDependentTypeFalse<ReturnType>, "Unsupported ArgumentClass");
    }
  }

  constexpr static arm64::CallingConventions kVAStartBase = std::get<0>(ParamsInfoHelper());

  constexpr static std::array<arm64::ArgLocation, sizeof...(ParamType)> kParamsLocations =
      std::get<1>(ParamsInfoHelper());

  constexpr static arm64::ArgLocation kReturnLocation = ResultInfoHelper();

  uint64_t* x_;
  __uint128_t* v_;
  uint8_t* s_;
};

// Partial specialization for GuestParamsAndReturn<FunctionToPointer> - it acts the same
// as the corresponding GuestParamsAndReturn<Function>.
template <typename ReturnType, typename... ParamType, bool kNoexcept>
class GuestParamsAndReturn<ReturnType (*)(ParamType...) noexcept(kNoexcept), GuestAbi::kAapcs64>
    : public GuestParamsAndReturn<ReturnType(ParamType...) noexcept(kNoexcept),
                                  GuestAbi::kAapcs64> {
 public:
  GuestParamsAndReturn(ThreadState* state)
      : GuestParamsAndReturn<ReturnType(ParamType...) noexcept(kNoexcept), GuestAbi::kAapcs64>(
            state) {}
};

// Partial specialization for GuestParamsAndReturn<Function> for functions with variadic
// arguents - it acts the same as the corresponding one without it.
template <typename ReturnType, typename... ParamType, bool kNoexcept>
class GuestParamsAndReturn<ReturnType(ParamType..., ...) noexcept(kNoexcept), GuestAbi::kAapcs64>
    : public GuestParamsAndReturn<ReturnType(ParamType...) noexcept(kNoexcept),
                                  GuestAbi::kAapcs64> {
 public:
  GuestParamsAndReturn(ThreadState* state)
      : GuestParamsAndReturn<ReturnType(ParamType...) noexcept(kNoexcept), GuestAbi::kAapcs64>(
            state) {}

 private:
  friend class GuestVAListParams;
  constexpr static arm64::CallingConventions kVAStartBase =
      std::get<0>(GuestParamsAndReturn<ReturnType(ParamType...) noexcept(kNoexcept),
                                       GuestAbi::kAapcs64>::ParamsInfoHelper());
};

// Partial specialization for GuestParamsAndReturn<FunctionToPointer> for functions with
// variadic arguents - it acts the same as the corresponding one without it.
template <typename ReturnType, typename... ParamType, bool kNoexcept>
class GuestParamsAndReturn<ReturnType (*)(ParamType..., ...) noexcept(kNoexcept),
                           GuestAbi::kAapcs64>
    : public GuestParamsAndReturn<ReturnType(ParamType..., ...) noexcept(kNoexcept),
                                  GuestAbi::kAapcs64> {
 public:
  GuestParamsAndReturn(ThreadState* state)
      : GuestParamsAndReturn<ReturnType(ParamType..., ...) noexcept(kNoexcept), GuestAbi::kAapcs64>(
            state) {}
};

template <typename FunctionType,
          GuestAbi::CallingConventionsVariant kCallingConventionsVariant = GuestAbi::kDefaultAbi>
class GuestParamsValues;

class GuestVAListParams : GuestParamsAndReturnHelper {
 public:
  template <typename Func, GuestAbi::CallingConventionsVariant kCallingConventionsVariant>
  GuestVAListParams(const GuestParamsValues<Func, kCallingConventionsVariant>&& named_parameters)
      : calling_conventions_(named_parameters.kVAStartBase),
        x_(named_parameters.x_),
        v_(named_parameters.v_),
        s_(named_parameters.s_) {}

  // Extract parameters from va_list.
  // On arm64, va_list is a struct passed by pointer.
  GuestVAListParams(GuestAddr va_ptr)
      : calling_conventions_(GuestVaListToIntOffset(va_ptr), GuestVaListToSimdOffset(va_ptr)),
        x_(GuestVaListToX(va_ptr)),
        v_(GuestVaListToV(va_ptr)),
        s_(GuestVaListToStack(va_ptr)) {}

  template <typename T>
  typename GuestArgumentInfo<T>::GuestType& GetParam() {
    if constexpr (GuestArgumentInfo<T>::kArgumentClass == ArgumentClass::kInteger ||
                  GuestArgumentInfo<T>::kArgumentClass == ArgumentClass::kLargeStructType) {
      return *this->ParamLocationAddress<T>(
          x_,
          v_,
          s_,
          calling_conventions_.GetNextIntArgLoc(GuestArgumentInfo<T>::kSize,
                                                GuestArgumentInfo<T>::kAlignment));
    } else if constexpr (GuestArgumentInfo<T>::kArgumentClass == ArgumentClass::kVFP) {
      return *this->ParamLocationAddress<T>(
          x_,
          v_,
          s_,
          calling_conventions_.GetNextFpArgLoc(GuestArgumentInfo<T>::kSize,
                                               GuestArgumentInfo<T>::kAlignment));
    } else {
      static_assert(kDependentTypeFalse<T>, "Unsupported ArgumentClass");
    }
  }

  template <typename T>
  T* GetPointerParam() {
    return ToHostAddr<T>(GetParam<GuestAddr>());
  }

 private:
  // See "Procedure Call Standard for the ARM 64-bit Architecture" (AAPCS64).
  struct Guest_va_list {
    GuestAddr __stack;   // next stack param
    GuestAddr __gr_top;  // end of GP arg reg save area
    GuestAddr __vr_top;  // end of FP/SIMD arg reg save area
    int32_t __gr_offs;   // offset from gr_top to next GP register arg
    int32_t __vr_offs;   // offset from vr_top to next FP/SIMD register arg
  };

  uint64_t* GuestVaListToX(GuestAddr va_ptr) {
    Guest_va_list* va_list = ToHostAddr<Guest_va_list>(va_ptr);
    return ToHostAddr<uint64_t>(va_list->__gr_top - 8 * sizeof(uint64_t));
  }

  unsigned GuestVaListToIntOffset(GuestAddr va_ptr) {
    Guest_va_list* va_list = ToHostAddr<Guest_va_list>(va_ptr);
    CHECK_LE(va_list->__gr_offs, 0);
    CHECK((-va_list->__gr_offs) % sizeof(uint64_t) == 0);
    return 8 - ((-va_list->__gr_offs) / sizeof(uint64_t));
  }

  __uint128_t* GuestVaListToV(GuestAddr va_ptr) {
    Guest_va_list* va_list = ToHostAddr<Guest_va_list>(va_ptr);
    return ToHostAddr<__uint128_t>(va_list->__vr_top - 8 * sizeof(__uint128_t));
  }

  unsigned GuestVaListToSimdOffset(GuestAddr va_ptr) {
    Guest_va_list* va_list = ToHostAddr<Guest_va_list>(va_ptr);
    CHECK_LE(va_list->__vr_offs, 0);
    CHECK((-va_list->__vr_offs) % sizeof(__uint128_t) == 0);
    return 8 - ((-va_list->__vr_offs) / sizeof(__uint128_t));
  }

  uint8_t* GuestVaListToStack(GuestAddr va_ptr) {
    Guest_va_list* va_list = ToHostAddr<Guest_va_list>(va_ptr);
    return ToHostAddr<uint8_t>(va_list->__stack);
  }

  arm64::CallingConventions calling_conventions_;

  uint64_t* x_;
  __uint128_t* v_;
  uint8_t* s_;
};

}  // namespace berberis

#endif  //  BERBERIS_GUEST_ABI_GUEST_PARAMS_ARCH_H_
