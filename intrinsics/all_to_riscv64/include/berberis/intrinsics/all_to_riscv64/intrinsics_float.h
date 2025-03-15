/*
 * Copyright (C) 2024 The Android Open Source Project
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

#ifndef BERBERIS_INTRINSICS_ALL_TO_RISCV64_INTRINSICS_FLOAT_H_
#define BERBERIS_INTRINSICS_ALL_TO_RISCV64_INTRINSICS_FLOAT_H_

#include <cinttypes>
#include <cmath>

#include "berberis/base/bit_util.h"
#include "berberis/base/logging.h"
#include "berberis/intrinsics/common/intrinsics_float.h"  // Float32/Float64
#include "berberis/intrinsics/guest_rounding_modes.h"     // FE_HOSTROUND/FE_TIESAWAY

namespace berberis::intrinsics {

#define MAKE_BINARY_OPERATOR(guest_name, operator_name, assignment_name)                         \
                                                                                                 \
  inline Float32 operator operator_name(const Float32& v1, const Float32& v2) {                  \
    Float32 result;                                                                              \
    asm("f" #guest_name ".s %0, %1, %2" : "=f"(result.value_) : "f"(v1.value_), "f"(v2.value_)); \
    return result;                                                                               \
  }                                                                                              \
                                                                                                 \
  inline Float32& operator assignment_name(Float32 & v1, const Float32 & v2) {                   \
    asm("f" #guest_name ".s %0, %1, %2" : "=f"(v1.value_) : "f"(v1.value_), "f"(v2.value_));     \
    return v1;                                                                                   \
  }                                                                                              \
                                                                                                 \
  inline Float64 operator operator_name(const Float64& v1, const Float64& v2) {                  \
    Float64 result;                                                                              \
    asm("f" #guest_name ".d %0, %1, %2" : "=f"(result.value_) : "f"(v1.value_), "f"(v2.value_)); \
    return result;                                                                               \
  }                                                                                              \
                                                                                                 \
  inline Float64& operator assignment_name(Float64 & v1, const Float64 & v2) {                   \
    asm("f" #guest_name ".d %0, %1, %2" : "=f"(v1.value_) : "f"(v1.value_), "f"(v2.value_));     \
    return v1;                                                                                   \
  }

MAKE_BINARY_OPERATOR(add, +, +=)
MAKE_BINARY_OPERATOR(sub, -, -=)
MAKE_BINARY_OPERATOR(mul, *, *=)
MAKE_BINARY_OPERATOR(div, /, /=)

#undef MAKE_BINARY_OPERATOR

inline bool operator<(const Float32& v1, const Float32& v2) {
  bool result;
  asm("flt.s %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator<(const Float64& v1, const Float64& v2) {
  bool result;
  asm("flt.d %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator>(const Float32& v1, const Float32& v2) {
  bool result;
  asm("flt.s %0, %1, %2" : "=r"(result) : "f"(v2.value_), "f"(v1.value_));
  return result;
}

inline bool operator>(const Float64& v1, const Float64& v2) {
  bool result;
  asm("flt.d %0, %1, %2" : "=r"(result) : "f"(v2.value_), "f"(v1.value_));
  return result;
}

inline bool operator<=(const Float32& v1, const Float32& v2) {
  bool result;
  asm("fle.s %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator<=(const Float64& v1, const Float64& v2) {
  bool result;
  asm("fle.d %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator>=(const Float32& v1, const Float32& v2) {
  bool result;
  asm("fle.s %0, %1, %2" : "=r"(result) : "f"(v2.value_), "f"(v1.value_));
  return result;
}

inline bool operator>=(const Float64& v1, const Float64& v2) {
  bool result;
  asm("fle.d %0, %1, %2" : "=r"(result) : "f"(v2.value_), "f"(v1.value_));
  return result;
}

inline bool operator==(const Float32& v1, const Float32& v2) {
  bool result;
  asm("feq.s %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator==(const Float64& v1, const Float64& v2) {
  bool result;
  asm("feq.d %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator!=(const Float32& v1, const Float32& v2) {
  bool result;
  asm("feq.s %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return !result;
}

inline bool operator!=(const Float64& v1, const Float64& v2) {
  bool result;
  asm("feq.d %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return !result;
}

// It's NOT safe to use ANY functions which return float or double.  That's because IA32 ABI uses
// x87 stack to pass arguments (and does that even with -mfpmath=sse) and NaN float and
// double values would be corrupted if pushed on it.

inline Float32 Negative(const Float32& v) {
  Float32 result;
  asm("fneg.s %0, %1" : "=f"(result.value_) : "f"(v.value_));
  return result;
}

inline Float64 Negative(const Float64& v) {
  Float64 result;
  asm("fneg.d %0, %1" : "=f"(result.value_) : "f"(v.value_));
  return result;
}

inline Float32 FPRound(const Float32& value, int round_control) {
  // RISC-V doesn't have any instructions that can be used used to implement FPRound efficiently
  // because conversion to integer returns an actual int (int32_t or int64_t) and that fails for
  // values that are larger than 1/ϵ – but all such values couldn't have fraction parts which means
  // that we may return them unmodified and only deal with small values that fit into int32_t below.
  Float32 result = value;
  // First of all we need to obtain positive value.
  Float32 positive_value;
  asm("fabs.s %0, %1" : "=f"(positive_value.value_) : "f"(result.value_));
  // Compare that positive value to 1/ϵ and return values that are not smaller unmodified.
  // Note: that includes ±∞ and NaNs!
  int64_t compare_result;
  asm("flt.s %0, %1, %2"
      : "=r"(compare_result)
      : "f"(positive_value.value_), "f"(float{1 / std::numeric_limits<float>::epsilon()}));
  if (compare_result == 0) [[unlikely]] {
    return result;
  }
  // Note: here we are dealing only with “small” values that can fit into int32_t.
  switch (round_control) {
    case FE_HOSTROUND:
      asm("fcvt.w.s %1, %2, dyn\n"
          "fcvt.s.w %0, %1, dyn"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_TONEAREST:
      asm("fcvt.w.s %1, %2, rne\n"
          "fcvt.s.w %0, %1, rne"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_DOWNWARD:
      asm("fcvt.w.s %1, %2, rdn\n"
          "fcvt.s.w %0, %1, rdn"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_UPWARD:
      asm("fcvt.w.s %1, %2, rup\n"
          "fcvt.s.w %0, %1, rup"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_TOWARDZERO:
      asm("fcvt.w.s %1, %2, rtz\n"
          "fcvt.s.w %0, %1, rtz"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_TIESAWAY:
      // Convert positive value to integer with rounding up.
      asm("fcvt.w.s %0, %1, rup" : "=r"(compare_result) : "f"(positive_value.value_));
      // Subtract .5 from the rounded avlue and compare to the previously calculated positive value.
      // Note: here we don't have to deal with infinities, NaNs, values that are too large, etc,
      // since they are all handled above before we reach that line.
      // But coding that in C++ gives compiler opportunity to use Zfa, if it's enabled.
      if (positive_value.value_ ==
          static_cast<float>(static_cast<float>(static_cast<int32_t>(compare_result)) - 0.5f)) {
        // If they are equal then we already have the final result (but without correct sign bit).
        // Thankfully RISC-V includes operation that can be used to pick sign from original value.
        result.value_ = static_cast<float>(static_cast<int32_t>(compare_result));
      } else {
        // Otherwise we may now use conversion to nearest.
        asm("fcvt.w.s %1, %2, rne\n"
            "fcvt.s.w %0, %1, rne"
            : "=f"(result.value_), "=r"(compare_result)
            : "f"(result.value_));
      }
      break;
    default:
      FATAL("Unknown round_control in FPRound!");
  }
  // Pick sign from original value. This is needed for -0 corner cases and ties away.
  asm("fsgnj.s %0, %1, %2" : "=f"(result.value_) : "f"(result.value_), "f"(value.value_));
  return result;
}

inline Float64 FPRound(const Float64& value, int round_control) {
  // RISC-V doesn't have any instructions that can be used used to implement FPRound efficiently
  // because conversion to integer returns an actual int (int32_t or int64_t) and that fails for
  // values that are larger than 1/ϵ – but all such values couldn't have fraction parts which means
  // that we may return them unmodified and only deal with small values that fit into int64_t below.
  Float64 result = value;
  // First of all we need to obtain positive value.
  Float64 positive_value;
  asm("fabs.d %0, %1" : "=f"(positive_value.value_) : "f"(result.value_));
  // Compare that positive value to 1/ϵ and return values that are not smaller unmodified.
  // Note: that includes ±∞ and NaNs!
  int64_t compare_result;
  asm("flt.d %0, %1, %2"
      : "=r"(compare_result)
      : "f"(positive_value.value_), "f"(1 / std::numeric_limits<double>::epsilon()));
  if (compare_result == 0) [[unlikely]] {
    return result;
  }
  // Note: here we are dealing only with “small” values that can fit into int32_t.
  switch (round_control) {
    case FE_HOSTROUND:
      asm("fcvt.l.d %1, %2, dyn\n"
          "fcvt.d.l %0, %1, dyn"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_TONEAREST:
      asm("fcvt.l.d %1, %2, rne\n"
          "fcvt.d.l %0, %1, rne"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_DOWNWARD:
      asm("fcvt.l.d %1, %2, rdn\n"
          "fcvt.d.l %0, %1, rdn"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_UPWARD:
      asm("fcvt.l.d %1, %2, rup\n"
          "fcvt.d.l %0, %1, rup"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_TOWARDZERO:
      asm("fcvt.l.d %1, %2, rtz\n"
          "fcvt.d.l %0, %1, rtz"
          : "=f"(result.value_), "=r"(compare_result)
          : "f"(result.value_));
      break;
    case FE_TIESAWAY:
      // Convert positive value to integer with rounding up.
      asm("fcvt.l.d %0, %1, rup" : "=r"(compare_result) : "f"(positive_value.value_));
      // Subtract .5 from the rounded value and compare to the previously calculated positive value.
      // Note: here we don't have to deal with infinities, NaNs, values that are too large, etc,
      // since they are all handled above before we reach that line.
      // But coding that in C++ gives compiler opportunity to use Zfa, if it's enabled.
      if (positive_value.value_ == static_cast<double>(compare_result) - 0.5) {
        // If they are equal then we already have the final result (but without correct sign bit).
        // Thankfully RISC-V includes operation that can be used to pick sign from original value.
        result.value_ = static_cast<double>(compare_result);
      } else {
        // Otherwise we may now use conversion to nearest.
        asm("fcvt.l.d %1, %2, rne\n"
            "fcvt.d.l %0, %1, rne"
            : "=f"(result.value_), "=r"(compare_result)
            : "f"(result.value_));
      }
      break;
    default:
      FATAL("Unknown round_control in FPRound!");
  }
  // Pick sign from original value. This is needed for -0 corner cases and ties away.
  asm("fsgnj.d %0, %1, %2" : "=f"(result.value_) : "f"(result.value_), "f"(value.value_));
  return result;
}

#undef ROUND_FLOAT

}  // namespace berberis::intrinsics

#endif  // BERBERIS_INTRINSICS_ALL_TO_RISCV64_INTRINSICS_FLOAT_H_
