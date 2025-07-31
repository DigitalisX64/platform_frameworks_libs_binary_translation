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

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <utility>

namespace berberis {

namespace {

template <typename T>
constexpr T BitUtilLog2(T x) {
  return __builtin_ctz(x);
}

using UInt8x16Tuple = std::tuple<uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t,
                                 uint8_t>;
using UInt16x8Tuple =
    std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>;
using UInt32x4Tuple = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
using UInt64x2Tuple = std::tuple<uint64_t, uint64_t>;

enum PrintModeEndianess { kLittleEndian, kBigEndian };

// A wrapper around __uint128 which can be constructed from a pair of uint64_t literals.
class SIMD128Register {
 public:
  SIMD128Register() {};

  constexpr SIMD128Register(UInt8x16Tuple u8) : uint8_{u8} {};
  constexpr SIMD128Register(UInt16x8Tuple u16) : uint16_{u16} {};
  constexpr SIMD128Register(UInt32x4Tuple u32) : uint32_{u32} {};
  constexpr SIMD128Register(UInt64x2Tuple u64) : uint64_{u64} {};
  constexpr SIMD128Register(__int128_t i128) : i128_{i128} {};
  constexpr SIMD128Register(uint8_t u128) : u128_{u128} {};
  constexpr SIMD128Register(uint16_t u128) : u128_{u128} {};
  constexpr SIMD128Register(uint32_t u128) : u128_{u128} {};
  constexpr SIMD128Register(uint64_t u128) : u128_{u128} {};
  constexpr SIMD128Register(__uint128_t u128) : u128_{u128} {};

  [[nodiscard]] constexpr __uint128_t Get() const { return u128_; }

  constexpr SIMD128Register& operator=(const SIMD128Register& other) {
    u128_ = other.u128_;
    return *this;
  };
  constexpr SIMD128Register& operator&=(const SIMD128Register& other) {
    u128_ &= other.u128_;
    return *this;
  }
  constexpr SIMD128Register& operator|=(const SIMD128Register& other) {
    u128_ |= other.u128_;
    return *this;
  }

  constexpr bool operator==(const SIMD128Register& other) const { return u128_ == other.u128_; }
  constexpr bool operator!=(const SIMD128Register& other) const { return u128_ != other.u128_; }
  constexpr SIMD128Register operator>>(size_t shift_amount) const { return u128_ >> shift_amount; }
  constexpr SIMD128Register operator<<(size_t shift_amount) const { return u128_ << shift_amount; }
  constexpr SIMD128Register operator&(SIMD128Register other) const { return u128_ & other.u128_; }
  constexpr SIMD128Register operator|(SIMD128Register other) const { return u128_ | other.u128_; }
  constexpr SIMD128Register operator^(SIMD128Register other) const { return u128_ ^ other.u128_; }
  constexpr SIMD128Register operator~() const { return ~u128_; }
  friend std::ostream& operator<<(std::ostream& os, const SIMD128Register& simd);

  template <size_t N>
  std::ostream& Print(std::ostream& os) const {
    if constexpr (kSimd128PrintMode == kBigEndian) {
      os << std::uppercase << std::hex << std::setw(4) << std::setfill('0') << std::get<N>(uint16_);
      if constexpr (N > 0) {
        os << '\'';
      }
    } else {
      os << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<int>(std::get<N * 2>(uint8_));
      os << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<int>(std::get<N * 2 + 1>(uint8_));
      if constexpr (N < 7) {
        os << '\'';
      }
    }
    return os;
  }

  template <size_t... N>
  std::ostream& PrintEach(std::ostream& os, std::index_sequence<N...>) const {
    os << "0x";
    if constexpr (kSimd128PrintMode == kBigEndian) {
      (Print<7 - N>(os), ...);
    } else {
      (Print<N>(os), ...);
    }
    return os;
  }

 private:
  union {
#ifdef __GNUC__
    [[gnu::may_alias]] UInt8x16Tuple uint8_;
    [[gnu::may_alias]] UInt16x8Tuple uint16_;
    [[gnu::may_alias]] UInt32x4Tuple uint32_;
    [[gnu::may_alias]] UInt64x2Tuple uint64_;
    [[gnu::may_alias]] __int128_t i128_;
    [[gnu::may_alias]] __uint128_t u128_;
#endif
  };

  // Support for BIG_ENDIAN or LITTLE_ENDIAN printing of SIMD128Register values. Change this value
  // if you want to see failure results in LITTLE_ENDIAN.
  static constexpr const PrintModeEndianess kSimd128PrintMode = kBigEndian;
};

// Helps produce easy to read output on failed tests.
std::ostream& operator<<(std::ostream& os, const SIMD128Register& simd) {
  return simd.PrintEach(os, std::make_index_sequence<8>());
}

const SIMD128Register kVectorCalculationsSourceLegacy[16] = {
    {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908}},
    {{0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918}},
    {{0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928}},
    {{0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938}},
    {{0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948}},
    {{0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958}},
    {{0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968}},
    {{0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},

    {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211}},
    {{0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231}},
    {{0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251}},
    {{0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271}},
    {{0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291}},
    {{0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1}},
    {{0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1}},
    {{0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
};

const SIMD128Register kVectorCalculationsSource[16] = {
    {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908}},
    {{0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918}},
    {{0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928}},
    {{0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938}},
    {{0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948}},
    {{0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958}},
    {{0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968}},
    {{0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},

    {{0x9e0c'9a09'9604'9200, 0x8e1c'8a18'8614'8211}},
    {{0xbe2c'ba29'b624'b220, 0xae3c'aa38'a634'a231}},
    {{0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251}},
    {{0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271}},
    {{0x1e8c'1a89'1684'1280, 0x0e9c'0a98'0694'0291}},
    {{0x3eac'3aa9'36a4'32a0, 0x2ebc'2ab8'26b4'22b1}},
    {{0x5ecc'5ac9'56c4'52c0, 0x4edc'4ad8'46d4'42d1}},
    {{0x7eec'7ae9'76e4'72e0, 0x6efc'6af8'66f4'62f1}},
};

const SIMD128Register kVectorComparisonSource[16] = {
    {{0xf005'f005'f005'f005, 0xffff'ffff'4040'4040}},
    {{0xffff'ffff'40b4'40b4, 0xffff'ffff'40b4'0000}},
    {{0x4016'4016'4016'4016, 0x4016'8000'0000'0000}},
    {{0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111}},
    {{0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6}},
    {{0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5}},
    {{0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb}},
    {{0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},

    {{0xf005'f005'f005'f005, 0xffff'ffff'4040'4040}},
    {{0x1111'1111'1111'1111, 0x1111'1111'1111'1111}},
    {{0xfff1'fff1'fff1'fff1, 0xfff1'fff1'fff1'fff1}},
    {{0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271}},
    {{0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291}},
    {{0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1}},
    {{0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1}},
    {{0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
};

namespace VXRMFlags {

inline constexpr uint64_t RNU = 0b00;
inline constexpr uint64_t RNE = 0b01;
inline constexpr uint64_t RDN = 0b10;
inline constexpr uint64_t ROD = 0b11;

}  // namespace VXRMFlags

// Easily recognizable bit pattern for target register.
const SIMD128Register kUndisturbedResult{{0x5555'5555'5555'5555, 0x5555'5555'5555'5555}};

SIMD128Register GetAgnosticResult() {
  static const bool kRvvAgnosticIsUndisturbed = getenv("RVV_AGNOSTIC_IS_UNDISTURBED") != nullptr;
  if (kRvvAgnosticIsUndisturbed) {
    return kUndisturbedResult;
  }
  return {{~uint64_t{0U}, ~uint64_t{0U}}};
}

const SIMD128Register kAgnosticResult = GetAgnosticResult();

// Mask in form suitable for storing in v0 and use in v0.t form.
const SIMD128Register kMask{{0xd5ad'd6b5'ad6b'b5ad, 0x6af7'57bb'deed'7bb5}};
// Mask used with vsew = 0 (8bit) elements.
const SIMD128Register kMaskInt8[8] = {
    {{255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255}},
    {{255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255}},
    {{255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255}},
    {{255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 0, 255, 255}},
    {{255, 0, 255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0}},
    {{255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 255}},
    {{255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0}},
    {{255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 0}},
};
// Mask used with vsew = 1 (16bit) elements.
const SIMD128Register kMaskInt16[8] = {
    {{0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff}},
    {{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff}},
    {{0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000}},
    {{0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff}},
    {{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff}},
    {{0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff}},
    {{0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff}},
    {{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff}},
};
// Mask used with vsew = 2 (32bit) elements.
const SIMD128Register kMaskInt32[8] = {
    {{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff}},
    {{0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}},
    {{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000}},
    {{0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}},
    {{0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}},
    {{0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0x0000'0000}},
    {{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff}},
    {{0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}},
};
// Mask used with vsew = 3 (64bit) elements.
const SIMD128Register kMaskInt64[8] = {
    {{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff}},
    {{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000}},
    {{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff}},
};
// To verify operations without masking.
const SIMD128Register kNoMask[8] = {
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
};

// Half of sub-register lmul.
const SIMD128Register kFractionMaskInt8[5] = {
    {{255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},                // Half of 1/8 reg = 1/16
    {{255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},              // Half of 1/4 reg = 1/8
    {{255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},          // Half of 1/2 reg = 1/4
    {{255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0}},  // Half of full reg = 1/2
    {{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},  // Full reg
};

SIMD128Register MakeBitmaskFromVl(size_t vl) {
  if (vl == 128) {
    return {__uint128_t(0)};
  } else {
    return {(~__int128(0)) << vl};
  }
}

template <typename ElementType>
auto MaskForElem() {
  if constexpr (std::is_same_v<ElementType, uint8_t>) {
    return kMaskInt8;
  } else if constexpr (std::is_same_v<ElementType, uint16_t>) {
    return kMaskInt16;
  } else if constexpr (std::is_same_v<ElementType, uint32_t>) {
    return kMaskInt32;
  } else if constexpr (std::is_same_v<ElementType, uint64_t>) {
    return kMaskInt64;
  } else {
    static_assert(false);
  }
}

template <bool kIsMasked, typename ElementType>
auto MaskForElemIfMasked() {
  if constexpr (!kIsMasked) {
    return kNoMask;
  } else {
    return MaskForElem<ElementType>();
  }
}

using ExecInsnFunc = void (*)();

void RunTwoVectorArgsOneRes(ExecInsnFunc exec_insn,
                            const SIMD128Register* src,
                            SIMD128Register* res,
                            uint64_t vtype,
                            uint64_t vlmax) {
  uint64_t vstart, vl;
  // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
  // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
  asm(  // Load arguments and undisturbed result.
      "vsetvli t0, zero, e64, m8, ta, ma\n\t"
      "vle64.v v8, (%[res])\n\t"
      "vle64.v v16, (%[src])\n\t"
      "addi t0, %[src], 128\n\t"
      "vle64.v v24, (t0)\n\t"
      // Load mask.
      "vsetvli t0, zero, e64, m1, ta, ma\n\t"
      "vle64.v v0, (%[mask])\n\t"
      // Execute tested instruction.
      "vsetvl t0, zero, %[vtype]\n\t"
      "jalr %[exec_insn]\n\t"
      // Save vstart and vl just after insn execution for checks.
      "csrr %[vstart], vstart\n\t"
      "csrr %[vl], vl\n\t"
      // Store the result.
      "vsetvli t0, zero, e64, m8, ta, ma\n\t"
      "vse64.v v8, (%[res])\n\t"
      : [vstart] "=&r"(vstart), [vl] "=&r"(vl)
      : [exec_insn] "r"(exec_insn),
        [src] "r"(src),
        [res] "r"(res),
        [vtype] "r"(vtype),
        [mask] "r"(&kMask)
      : "t0",
        "ra",
        "v0",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31",
        "memory");
  // Every vector instruction must set vstart to 0, but shouldn't touch vl.
  EXPECT_EQ(vstart, 0);
  EXPECT_EQ(vl, vlmax);
}

// Supports ExecInsnFuncs that fit the following [inputs...] -> output formats:
//   vector -> vector
//   vector, vector -> vector
//   vector, scalar -> vector
//   vector, float -> vector
// Vectors will be used in v16 first, then v24
// scalar and float will be filled from scalar_src, and will use t0 and ft0,
// respectively.
void RunCommonVectorFunc(ExecInsnFunc exec_insn,
                         const SIMD128Register* src,
                         SIMD128Register* res,
                         uint64_t* scalar_int_res,
                         uint64_t* scalar_float_res,
                         uint64_t scalar_src,
                         uint64_t vstart,
                         uint64_t vtype,
                         uint64_t vlin) {
  uint64_t unused_local;
  if (!scalar_int_res) {
    scalar_int_res = &unused_local;
  }
  if (!scalar_float_res) {
    scalar_float_res = &unused_local;
  }
  SIMD128Register unused_simd_local[8];
  if (!res) {
    res = &unused_simd_local[0];
  }

  uint64_t vl = vlin;
  // Mask register is, unconditionally, v0, and we need 8 or 24 to handle full 8-registers
  // inputs thus we use v8..v15 for destination and place sources into v24..v31.
  asm(  // Load arguments and undisturbed result.
      "vsetvli t0, zero, e64, m8, ta, ma\n\t"
      "vle64.v v8, (%[res])\n\t"
      "vle64.v v16, (%[src])\n\t"
      "addi t0, %[src], 128\n\t"
      "vle64.v v24, (t0)\n\t"
      // Load mask.
      "vsetvli t0, zero, e64, m1, ta, ma\n\t"
      "vle64.v v0, (%[mask])\n\t"
      // Execute tested instruction.
      "vsetvl t0, %[vl], %[vtype]\n\t"
      "csrw vstart, %[vstart]\n\t"
      "mv t0, %[scalar_src]\n\t"
      "fmv.d.x ft0, %[scalar_src]\n\t"
      "jalr %[exec_insn]\n\t"
      // Save t0, ft0 just after insn execution for return.
      "sd t0, %[scalar_int]\n\t"
      "fsd ft0, %[scalar_float]\n\t"
      // Save vstart and vl just after insn execution for checks.
      "csrr %[vstart], vstart\n\t"
      "csrr %[vl], vl\n\t"
      // Store the result.
      "vsetvli t0, zero, e64, m8, ta, ma\n\t"
      "vse64.v v8, (%[res])\n\t"
      : [vstart] "=r"(vstart), [vl] "=r"(vl)
      : [exec_insn] "r"(exec_insn),
        [src] "r"(src),
        [res] "r"(res),
        [vtype] "r"(vtype),
        "0"(vstart),
        "1"(vl),
        [mask] "r"(&kMask),
        [scalar_src] "r"(scalar_src),
        [scalar_float] "m"(*scalar_float_res),
        [scalar_int] "m"(*scalar_int_res)
      : "t0",
        "ra",
        "ft0",
        "v0",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31",
        "memory");
  // Every vector instruction must set vstart to 0, but shouldn't touch vl.
  EXPECT_EQ(vstart, 0);
  EXPECT_EQ(vl, vlin);
}

enum class TestVectorInstructionKind { kInteger, kFloat };
enum class TestVectorInstructionMode { kDefault, kWidening, kNarrowing, kVMerge };

template <TestVectorInstructionKind kTestVectorInstructionKind,
          TestVectorInstructionMode kTestVectorInstructionMode,
          typename... ExpectedResultType,
          size_t... kResultsCount>
void TestVectorInstructionInternal(ExecInsnFunc exec_insn,
                                   ExecInsnFunc exec_masked_insn,
                                   const SIMD128Register dst_result,
                                   const SIMD128Register (&source)[16],
                                   const ExpectedResultType (&... expected_result)[kResultsCount]) {
  auto Verify = [&source, dst_result](ExecInsnFunc exec_insn,
                                      uint8_t vsew,
                                      const auto& expected_result,
                                      const auto& mask) {
    uint64_t scalar_src = 0;
    if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kInteger) {
      // Set t0 for vx instructions.
      scalar_src = 0xaaaa'aaaa'aaaa'aaaa;
    } else {
      // We only support Float32/Float64 for float instructions, but there are conversion
      // instructions that work with double width floats.
      // These instructions never use float registers though and thus we don't need to store
      // anything into ft0 register, if they are used.
      // For Float32/Float64 case we load 5.625 of the appropriate type into ft0.
      ASSERT_LE(vsew, 3);
      if (vsew == 2) {
        scalar_src = 0xffff'ffff'40b4'0000;  // float 5.625
      } else if (vsew == 3) {
        scalar_src = 0x4016'8000'0000'0000;  // double 5.625
      }
    }
    for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
      if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kNarrowing ||
                    kTestVectorInstructionMode == TestVectorInstructionMode::kWidening) {
        // Incompatible vlmul for narrowing.
        if (vlmul == 3) {
          continue;
        }
      }
      for (uint8_t vta = 0; vta < 2; ++vta) {
        for (uint8_t vma = 0; vma < 2; ++vma) {
          uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
          uint64_t vlmax = 0;
          asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
          // Incompatible vsew and vlmax. Skip it.
          if (vlmax == 0) {
            continue;
          }
          uint8_t emul =
              (vlmul + (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening)) &
              0b111;

          // To make tests quick enough we don't test vstart and vl change with small register
          // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
          // register, last register and half of next-to last register.
          // Don't use vlmul == 3 because that one may not be supported if instruction widens the
          // result.
          uint64_t vstart;
          uint64_t vl;
          if (emul == 2) {
            vstart = vlmax / 8;
            vl = (vlmax * 5) / 8;
          } else {
            vstart = 0;
            vl = vlmax;
          }

          SIMD128Register result[8];
          // Set expected_result vector registers into 0b01010101… pattern.
          // Set undisturbed result vector registers.
          std::fill_n(result, 8, dst_result);

          RunCommonVectorFunc(
              exec_insn, &source[0], &result[0], nullptr, nullptr, scalar_src, vstart, vtype, vl);

          // Values for inactive elements (i.e. corresponding mask bit is 0).
          SIMD128Register expected_inactive[8];
          if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kVMerge) {
            // vs2 is the start of the source vector register group.
            // Note: copy_n input/output args are backwards compared to fill_n below.
            std::copy_n(source, 8, expected_inactive);
          } else {
            // For most instructions, follow basic inactive processing rules based on vma flag.
            std::fill_n(expected_inactive, 8, (vma ? kAgnosticResult : dst_result));
          }

          if (emul < 4) {
            for (size_t index = 0; index < 1 << emul; ++index) {
              if (index == 0 && emul == 2) {
                EXPECT_EQ(result[index],
                          ((dst_result & kFractionMaskInt8[3]) |
                           (SIMD128Register{expected_result[index]} & mask[index] &
                            ~kFractionMaskInt8[3]) |
                           (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3])));
              } else if (index == 2 && emul == 2) {
                EXPECT_EQ(result[index],
                          ((SIMD128Register{expected_result[index]} & mask[index] &
                            kFractionMaskInt8[3]) |
                           (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                           ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[3])));
              } else if (index == 3 && emul == 2 && vta) {
                EXPECT_EQ(result[index], kAgnosticResult);
              } else if (index == 3 && emul == 2) {
                EXPECT_EQ(result[index], dst_result);
              } else {
                EXPECT_EQ(result[index],
                          ((SIMD128Register{expected_result[index]} & mask[index]) |
                           ((expected_inactive[index] & ~mask[index]))));
              }
            }
          } else {
            EXPECT_EQ(
                result[0],
                ((SIMD128Register{expected_result[0]} & mask[0] & kFractionMaskInt8[emul - 4]) |
                 (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[emul - 4]) |
                 ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[emul - 4])));
          }
        }
      }
    }
  };

  ((Verify(exec_insn,
           BitUtilLog2(sizeof(std::tuple_element_t<0, ExpectedResultType>)) -
               (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening),
           expected_result,
           kNoMask),
    Verify(exec_masked_insn,
           BitUtilLog2(sizeof(std::tuple_element_t<0, ExpectedResultType>)) -
               (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening),
           expected_result,
           MaskForElem<std::tuple_element_t<0, ExpectedResultType>>())),
   ...);
}

template <TestVectorInstructionKind kTestVectorInstructionKind,
          TestVectorInstructionMode kTestVectorInstructionMode,
          typename... ExpectedResultType,
          size_t... kResultsCount>
void TestVectorInstruction(ExecInsnFunc exec_insn,
                           ExecInsnFunc exec_masked_insn,
                           const SIMD128Register (&source)[16],
                           const ExpectedResultType (&... expected_result)[kResultsCount]) {
  TestVectorInstructionInternal<kTestVectorInstructionKind, kTestVectorInstructionMode>(
      exec_insn, exec_masked_insn, kUndisturbedResult, source, expected_result...);
}

void TestVectorInstruction(ExecInsnFunc exec_insn,
                           ExecInsnFunc exec_masked_insn,
                           const UInt8x16Tuple (&expected_result_int8)[8],
                           const UInt16x8Tuple (&expected_result_int16)[8],
                           const UInt32x4Tuple (&expected_result_int32)[8],
                           const UInt64x2Tuple (&expected_result_int64)[8],
                           const SIMD128Register (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kDefault>(
      exec_insn,
      exec_masked_insn,
      source,
      expected_result_int8,
      expected_result_int16,
      expected_result_int32,
      expected_result_int64);
}

void TestVectorFloatInstruction(ExecInsnFunc exec_insn,
                                ExecInsnFunc exec_masked_insn,
                                const UInt32x4Tuple (&expected_result_int32)[8],
                                const UInt64x2Tuple (&expected_result_int64)[8],
                                const SIMD128Register (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kDefault>(
      exec_insn, exec_masked_insn, source, expected_result_int32, expected_result_int64);
}

void TestNarrowingVectorInstruction(ExecInsnFunc exec_insn,
                                    ExecInsnFunc exec_masked_insn,
                                    const UInt8x16Tuple (&expected_result_int8)[4],
                                    const UInt16x8Tuple (&expected_result_int16)[4],
                                    const UInt32x4Tuple (&expected_result_int32)[4],
                                    const SIMD128Register (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kNarrowing>(
      exec_insn,
      exec_masked_insn,
      source,
      expected_result_int8,
      expected_result_int16,
      expected_result_int32);
}

void TestNarrowingVectorFloatInstruction(ExecInsnFunc exec_insn,
                                         ExecInsnFunc exec_masked_insn,
                                         const UInt32x4Tuple (&expected_result_int32)[4],
                                         const SIMD128Register (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kNarrowing>(
      exec_insn, exec_masked_insn, source, expected_result_int32);
}

void TestNarrowingVectorFloatInstruction(ExecInsnFunc exec_insn,
                                         ExecInsnFunc exec_masked_insn,
                                         const UInt16x8Tuple (&expected_result_int16)[4],
                                         const UInt32x4Tuple (&expected_result_int32)[4],
                                         const SIMD128Register (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kNarrowing>(
      exec_insn, exec_masked_insn, source, expected_result_int16, expected_result_int32);
}

void TestWideningVectorFloatInstruction(ExecInsnFunc exec_insn,
                                        ExecInsnFunc exec_masked_insn,
                                        const UInt64x2Tuple (&expected_result_int64)[8],
                                        const SIMD128Register (&source)[16],
                                        SIMD128Register dst_result = kUndisturbedResult) {
  TestVectorInstructionInternal<TestVectorInstructionKind::kFloat,
                                TestVectorInstructionMode::kWidening>(
      exec_insn, exec_masked_insn, dst_result, source, expected_result_int64);
}

void TestWideningVectorFloatInstruction(ExecInsnFunc exec_insn,
                                        ExecInsnFunc exec_masked_insn,
                                        const UInt32x4Tuple (&expected_result_int32)[8],
                                        const UInt64x2Tuple (&expected_result_int64)[8],
                                        const SIMD128Register (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kWidening>(
      exec_insn, exec_masked_insn, source, expected_result_int32, expected_result_int64);
}

void TestWideningVectorInstruction(ExecInsnFunc exec_insn,
                                   ExecInsnFunc exec_masked_insn,
                                   const UInt16x8Tuple (&expected_result_int16)[8],
                                   const UInt32x4Tuple (&expected_result_int32)[8],
                                   const UInt64x2Tuple (&expected_result_int64)[8],
                                   const SIMD128Register (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kWidening>(
      exec_insn,
      exec_masked_insn,
      source,
      expected_result_int16,
      expected_result_int32,
      expected_result_int64);
}

template <TestVectorInstructionMode kTestVectorInstructionMode, typename... ExpectedResultType>
void TestVectorReductionInstruction(
    ExecInsnFunc exec_insn,
    ExecInsnFunc exec_masked_insn,
    const SIMD128Register (&source)[16],
    std::tuple<const ExpectedResultType (&)[8],
               const ExpectedResultType (&)[8]>... expected_result) {
  // Each expected_result input to this function is the vd[0] value of the reduction, for each
  // of the possible vlmul, i.e. expected_result_vd0_int8[n] = vd[0], int8, no mask, vlmul=n.
  //
  // As vlmul=4 is reserved, expected_result_vd0_*[4] is ignored.
  auto Verify =
      [&source](ExecInsnFunc exec_insn, uint8_t vsew, uint8_t vlmul, const auto& expected_result) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          for (uint8_t vma = 0; vma < 2; ++vma) {
            uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
            uint64_t vlmax = 0;
            asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
            if (vlmax == 0) {
              continue;
            }

            SIMD128Register result[8];
            // Set undisturbed result vector registers.
            for (size_t index = 0; index < 8; ++index) {
              result[index] = kUndisturbedResult;
            }

            // Exectations for reductions are for swapped source arguments.
            SIMD128Register two_sources[16]{};
            memcpy(&two_sources[0], &source[8], sizeof(two_sources[0]) * 8);
            memcpy(&two_sources[8], &source[0], sizeof(two_sources[0]) * 8);

            RunTwoVectorArgsOneRes(exec_insn, &two_sources[0], &result[0], vtype, vlmax);

            // Reduction instructions are unique in that they produce a scalar
            // output to a single vector register as opposed to a register group.
            // This allows us to take some short-cuts when validating:
            //
            // - The mask setting is only useful during computation, as the body
            // of the destination is always only element 0, which will always be
            // written to, regardless of mask setting.
            // - The tail is guaranteed to be 1..VLEN/SEW, so the vlmul setting
            // does not affect the elements that the tail policy applies to in the
            // destination register.

            // Verify that the destination register holds the reduction in the
            // first element and the tail policy applies to the remaining.
            SIMD128Register expected_result_register = vta ? kAgnosticResult : kUndisturbedResult;
            size_t result_bits = 8 << vsew;
            if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening) {
              result_bits *= 2;
            }
            expected_result_register &= ~SIMD128Register{(__int128_t{1} << result_bits) - 1};
            expected_result_register |= SIMD128Register{expected_result};
            EXPECT_EQ(result[0], expected_result_register) << " vtype=" << vtype;

            // Verify all non-destination registers are undisturbed.
            for (size_t index = 1; index < 8; ++index) {
              EXPECT_EQ(result[index], kUndisturbedResult)
                  << " index=" << index << " vtype=" << vtype;
            }
          }
        }
      };

  auto GetVsew = [](size_t result_size) -> uint8_t {
    size_t sew = result_size;
    if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening) {
      sew /= 2;
    }
    return BitUtilLog2(sew);
  };

  for (int vlmul = 0; vlmul < 8; vlmul++) {
    ((Verify(exec_insn,
             GetVsew(sizeof(ExpectedResultType)),
             vlmul,
             std::get<0>(expected_result)[vlmul]),
      Verify(exec_masked_insn,
             GetVsew(sizeof(ExpectedResultType)),
             vlmul,
             std::get<1>(expected_result)[vlmul])),
     ...);
  }
}

void TestVectorReductionInstruction(ExecInsnFunc exec_insn,
                                    ExecInsnFunc exec_masked_insn,
                                    const uint32_t (&expected_result_vd0_int32)[8],
                                    const uint64_t (&expected_result_vd0_int64)[8],
                                    const uint32_t (&expected_result_vd0_with_mask_int32)[8],
                                    const uint64_t (&expected_result_vd0_with_mask_int64)[8],
                                    const SIMD128Register (&source)[16]) {
  TestVectorReductionInstruction<TestVectorInstructionMode::kDefault>(
      exec_insn,
      exec_masked_insn,
      source,
      std::tuple<const uint32_t(&)[8], const uint32_t(&)[8]>{expected_result_vd0_int32,
                                                             expected_result_vd0_with_mask_int32},
      std::tuple<const uint64_t(&)[8], const uint64_t(&)[8]>{expected_result_vd0_int64,
                                                             expected_result_vd0_with_mask_int64});
}

void TestVectorReductionInstruction(ExecInsnFunc exec_insn,
                                    ExecInsnFunc exec_masked_insn,
                                    const uint8_t (&expected_result_vd0_int8)[8],
                                    const uint16_t (&expected_result_vd0_int16)[8],
                                    const uint32_t (&expected_result_vd0_int32)[8],
                                    const uint64_t (&expected_result_vd0_int64)[8],
                                    const uint8_t (&expected_result_vd0_with_mask_int8)[8],
                                    const uint16_t (&expected_result_vd0_with_mask_int16)[8],
                                    const uint32_t (&expected_result_vd0_with_mask_int32)[8],
                                    const uint64_t (&expected_result_vd0_with_mask_int64)[8],
                                    const SIMD128Register (&source)[16]) {
  TestVectorReductionInstruction<TestVectorInstructionMode::kDefault>(
      exec_insn,
      exec_masked_insn,
      source,
      std::tuple<const uint8_t(&)[8], const uint8_t(&)[8]>{expected_result_vd0_int8,
                                                           expected_result_vd0_with_mask_int8},
      std::tuple<const uint16_t(&)[8], const uint16_t(&)[8]>{expected_result_vd0_int16,
                                                             expected_result_vd0_with_mask_int16},
      std::tuple<const uint32_t(&)[8], const uint32_t(&)[8]>{expected_result_vd0_int32,
                                                             expected_result_vd0_with_mask_int32},
      std::tuple<const uint64_t(&)[8], const uint64_t(&)[8]>{expected_result_vd0_int64,
                                                             expected_result_vd0_with_mask_int64});
}

void TestWideningVectorReductionInstruction(
    ExecInsnFunc exec_insn,
    ExecInsnFunc exec_masked_insn,
    const uint64_t (&expected_result_vd0_int64)[8],
    const uint64_t (&expected_result_vd0_with_mask_int64)[8],
    const SIMD128Register (&source)[16]) {
  TestVectorReductionInstruction<TestVectorInstructionMode::kWidening>(
      exec_insn,
      exec_masked_insn,
      source,
      std::tuple<const uint64_t(&)[8], const uint64_t(&)[8]>{expected_result_vd0_int64,
                                                             expected_result_vd0_with_mask_int64});
}

void TestWideningVectorReductionInstruction(
    ExecInsnFunc exec_insn,
    ExecInsnFunc exec_masked_insn,
    const uint16_t (&expected_result_vd0_int16)[8],
    const uint32_t (&expected_result_vd0_int32)[8],
    const uint64_t (&expected_result_vd0_int64)[8],
    const uint16_t (&expected_result_vd0_with_mask_int16)[8],
    const uint32_t (&expected_result_vd0_with_mask_int32)[8],
    const uint64_t (&expected_result_vd0_with_mask_int64)[8],
    const SIMD128Register (&source)[16]) {
  TestVectorReductionInstruction<TestVectorInstructionMode::kWidening>(
      exec_insn,
      exec_masked_insn,
      source,
      std::tuple<const uint16_t(&)[8], const uint16_t(&)[8]>{expected_result_vd0_int16,
                                                             expected_result_vd0_with_mask_int16},
      std::tuple<const uint32_t(&)[8], const uint32_t(&)[8]>{expected_result_vd0_int32,
                                                             expected_result_vd0_with_mask_int32},
      std::tuple<const uint64_t(&)[8], const uint64_t(&)[8]>{expected_result_vd0_int64,
                                                             expected_result_vd0_with_mask_int64});
}

template <bool kIsMasked, typename... ExpectedResultType, size_t... kResultsCount>
void TestVectorIota(ExecInsnFunc exec_insn,
                    const SIMD128Register (&source)[16],
                    const ExpectedResultType (&... expected_result)[kResultsCount]) {
  auto Verify = [&source](ExecInsnFunc exec_insn,
                          uint8_t vsew,
                          const auto& expected_result,
                          auto elem_mask) {
    for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
      for (uint8_t vta = 0; vta < 2; ++vta) {
        for (uint8_t vma = 0; vma < 2; ++vma) {
          uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
          uint64_t vlmax = 0;
          asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
          if (vlmax == 0) {
            continue;
          }

          for (uint8_t vl = 0; vl < vlmax; vl += vlmax) {
            // To make tests quick enough we don't test vl change with small register sets. Only
            // with vlmul == 2 (4 registers) we set vl to skip last register and half of next-to
            // last register.
            uint64_t vlin;
            if (vlmul == 2 && vl == vlmax) {
              vlin = 5 * vlmax / 8;
            } else {
              vlin = vl;
            }

            SIMD128Register result[8];
            // Set expected_result vector registers into 0b01010101… pattern.
            // Set undisturbed result vector registers.
            std::fill_n(result, 8, kUndisturbedResult);

            RunCommonVectorFunc(
                exec_insn, &source[0], &result[0], nullptr, nullptr, 0, 0, vtype, vlin);

            SIMD128Register expected_inactive[8];
            std::fill_n(expected_inactive, 8, (vma ? kAgnosticResult : kUndisturbedResult));

            // vl of 0 should never change dst registers
            if (vl == 0) {
              for (size_t index = 0; index < 8; ++index) {
                EXPECT_EQ(result[index], kUndisturbedResult);
              }
            } else if (vlmul < 4) {
              for (size_t index = 0; index < 1 << vlmul; ++index) {
                for (size_t index = 0; index < 1 << vlmul; ++index) {
                  if (index == 2 && vlmul == 2) {
                    EXPECT_EQ(
                        result[index],
                        ((SIMD128Register{expected_result[index]} & elem_mask[index] &
                          kFractionMaskInt8[3]) |
                         (expected_inactive[index] & ~elem_mask[index] & kFractionMaskInt8[3]) |
                         ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])));
                  } else if (index == 3 && vlmul == 2) {
                    EXPECT_EQ(result[index], vta ? kAgnosticResult : kUndisturbedResult);
                  } else {
                    EXPECT_EQ(result[index],
                              ((SIMD128Register{expected_result[index]} & elem_mask[index]) |
                               (expected_inactive[index] & ~elem_mask[index])));
                  }
                }
              }
            } else {
              // vlmul >= 4 only uses 1 register
              EXPECT_EQ(
                  result[0],
                  ((SIMD128Register{expected_result[0]} & elem_mask[0] &
                    kFractionMaskInt8[vlmul - 4]) |
                   (expected_inactive[0] & ~elem_mask[0] & kFractionMaskInt8[vlmul - 4]) |
                   ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[vlmul - 4])));
            }
          }
        }
      }
    }
  };

  (Verify(exec_insn,
          BitUtilLog2(sizeof(std::tuple_element_t<0, ExpectedResultType>)),
          expected_result,
          MaskForElemIfMasked<kIsMasked, std::tuple_element_t<0, ExpectedResultType>>()),
   ...);
}

template <bool kIsMasked>
void TestVectorIota(ExecInsnFunc exec_insn,
                    const UInt8x16Tuple (&expected_result_int8)[8],
                    const UInt16x8Tuple (&expected_result_int16)[8],
                    const UInt32x4Tuple (&expected_result_int32)[8],
                    const UInt64x2Tuple (&expected_result_int64)[8],
                    const SIMD128Register (&source)[16]) {
  TestVectorIota<kIsMasked>(exec_insn,
                            source,
                            expected_result_int8,
                            expected_result_int16,
                            expected_result_int32,
                            expected_result_int64);
}

void TestExtendingVectorInstruction(ExecInsnFunc exec_insn,
                                    ExecInsnFunc exec_masked_insn,
                                    const UInt16x8Tuple (&expected_result_int16)[8],
                                    const UInt32x4Tuple (&expected_result_int32)[8],
                                    const UInt64x2Tuple (&expected_result_int64)[8],
                                    const SIMD128Register (&source)[16],
                                    const uint8_t factor) {
  auto Verify = [&source, &factor](ExecInsnFunc exec_insn,
                                   uint8_t vsew,
                                   uint8_t vlmul_max,
                                   const auto& expected_result,
                                   auto mask) {
    EXPECT_TRUE((factor == 2) || (factor == 4) || (factor == 8));
    for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
      if (vlmul == 3) {
        continue;
      }
      for (uint8_t vta = 0; vta < 2; ++vta) {
        for (uint8_t vma = 0; vma < 2; ++vma) {
          uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
          uint64_t vlmax = 0;
          asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
          // Incompatible vsew and vlmax. Skip it.
          if (vlmax == 0) {
            continue;
          }
          // To make tests quick enough we don't test vstart and vl change with small register
          // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of
          // first
          // register and half of last register.
          // Don't use vlmul == 3 because that one may not be supported if instruction widens
          // the result.
          uint64_t vstart;
          uint64_t vl;
          if (vlmul == 2) {
            vstart = vlmax / 8;
            vl = (vlmax * 5) / 8;
          } else {
            vstart = 0;
            vl = vlmax;
          }

          SIMD128Register result[8];
          // Set expected_result vector registers into 0b01010101… pattern.
          // Set undisturbed result vector registers.
          std::fill_n(result, 8, kUndisturbedResult);

          RunCommonVectorFunc(
              exec_insn, &source[0], &result[0], nullptr, nullptr, 0, vstart, vtype, vl);

          // Values for inactive elements (i.e. corresponding mask bit is 0).
          const size_t n = std::size(source) * 2;
          SIMD128Register expected_inactive[n];
          // For most instructions, follow basic inactive processing rules based on vma flag.
          std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));

          if (vlmul < 4) {
            for (size_t index = 0; index < 1 << vlmul; ++index) {
              if (index == 0 && vlmul == 2) {
                EXPECT_EQ(result[index],
                          (kUndisturbedResult & kFractionMaskInt8[3]) |
                              (SIMD128Register{expected_result[index]} & mask[index] &
                               ~kFractionMaskInt8[3]) |
                              (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3]));
              } else if (index == 2 && vlmul == 2) {
                EXPECT_EQ(
                    result[index],
                    (SIMD128Register{expected_result[index]} & mask[index] & kFractionMaskInt8[3]) |
                        (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                        ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3]));
              } else if (index == 3 && vlmul == 2 && vta) {
                EXPECT_EQ(result[index], kAgnosticResult);
              } else if (index == 3 && vlmul == 2) {
                EXPECT_EQ(result[index], kUndisturbedResult);
              } else {
                EXPECT_EQ(result[index],
                          (SIMD128Register{expected_result[index]} & mask[index]) |
                              (expected_inactive[index] & ~mask[index]));
              }
            }
          } else {
            EXPECT_EQ(
                result[0],
                (SIMD128Register{expected_result[0]} & mask[0] & kFractionMaskInt8[vlmul - 4]) |
                    (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[vlmul - 4]) |
                    ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[vlmul - 4]));
          }
        }
      }
    }
  };

  if (factor == 2) {
    Verify(exec_masked_insn, 1, 8, expected_result_int16, kMaskInt16);
    Verify(exec_insn, 1, 8, expected_result_int16, kNoMask);
  }
  if (factor == 2 || factor == 4) {
    Verify(exec_masked_insn, 2, 8, expected_result_int32, kMaskInt32);
    Verify(exec_insn, 2, 8, expected_result_int32, kNoMask);
  }
  Verify(exec_masked_insn, 3, 8, expected_result_int64, kMaskInt64);
  Verify(exec_insn, 3, 8, expected_result_int64, kNoMask);
}

// Unlike regular arithmetic instructions, the result of a permutation
// instruction depends also on vlmul.  Also, the vslideup specs mention that
// the destination vector remains unchanged the first |offset| elements (in
// effect, the offset acts akin to vstart), in those cases skip can be used
// to specify how many elements' mask will be skipped (counting from the
// beginning, should be the same as the offset).
//
// If |ignore_vma_for_last| is true, an inactive element at vl-1 will be
// treated as if vma=0 (Undisturbed).
// If |last_elem_is_reg1| is true, the last element of the vector in
// expected_result (that is, at vl-1) will be expected to be the same as
// |regx1| when VL < VMAX and said element is active.
template <TestVectorInstructionKind kTestVectorInstructionKind,
          typename... ExpectedResultType,
          size_t... kResultsCount>
void TestVectorPermutationInstruction(
    ExecInsnFunc exec_insn,
    ExecInsnFunc exec_masked_insn,
    const SIMD128Register (&source)[16],
    uint8_t vlmul,
    uint64_t skip,
    bool ignore_vma_for_last,
    bool last_elem_is_reg1,
    uint64_t regt0,
    const ExpectedResultType (&... expected_result)[kResultsCount]) {
  auto Verify = [&source, vlmul, regt0, skip, ignore_vma_for_last, last_elem_is_reg1](
                    ExecInsnFunc exec_insn,
                    uint8_t vsew,
                    const auto& expected_result_raw,
                    auto mask) {
    uint64_t scalar_src = 0;
    if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
      (void)(regt0);
      // We only support Float32/Float64 for float instructions, but there are conversion
      // instructions that work with double width floats.
      // These instructions never use float registers though and thus we don't need to store
      // anything into f1 register, if they are used.
      // For Float32/Float64 case we load 5.625 of the appropriate type into ft0.
      ASSERT_LE(vsew, 3);
      if (vsew == 2) {
        scalar_src = 0xffff'ffff'40b4'0000;  // float 5.625
      } else if (vsew == 3) {
        scalar_src = 0x4016'8000'0000'0000;  // double 5.625
      }
    } else {
      // Set t0 for vx instructions.
      scalar_src = regt0;
    }

    const size_t kElementSize = 1 << vsew;
    size_t num_regs = 1 << vlmul;
    if (vlmul > 3) {
      num_regs = 1;
    }
    // Values for which the mask is not applied due to being before the offset when doing
    // vslideup.
    SIMD128Register skip_mask[num_regs];
    int64_t toskip = skip;
    for (size_t index = 0; index < num_regs && toskip > 0; ++index) {
      size_t skip_bits = toskip * kElementSize * 8;
      skip_mask[index] = ~MakeBitmaskFromVl(skip_bits > 128 ? 128 : skip_bits);
      toskip -= 16 / kElementSize;
    }

    for (uint8_t vta = 0; vta < 2; ++vta) {
      for (uint8_t vma = 0; vma < 2; ++vma) {
        uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
        uint64_t vlmax = 0;
        asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
        // Incompatible vsew and vlmax. Skip it.
        if (vlmax == 0) {
          continue;
        }

        // To make tests quick enough we don't test vstart and vl change with small register
        // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
        // register, last register and half of next-to last register.
        // Don't use vlmul == 3 because that one may not be supported if instruction widens the
        // result.
        uint64_t vstart;
        uint64_t vl;
        if (vlmul == 2) {
          vstart = vlmax / 8;
          vl = (vlmax * 5) / 8;
        } else {
          vstart = 0;
          vl = vlmax;
        }

        SIMD128Register result[8];
        // Set expected_result vector registers into 0b01010101… pattern.
        // Set undisturbed result vector registers.
        std::fill_n(result, 8, kUndisturbedResult);

        uint64_t int_res, float_res;
        RunCommonVectorFunc(
            exec_insn, &source[0], &result[0], &int_res, &float_res, scalar_src, vstart, vtype, vl);

        const size_t n = std::size(source);
        // Values for inactive elements (i.e. corresponding mask bit is 0).
        SIMD128Register expected_inactive[n];
        // For most instructions, follow basic inactive processing rules based on vma flag.
        std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));

        const size_t kElementsPerRegister = 16 / kElementSize;
        // TODO is this vl the input vl or vl after commands run?
        const size_t last_reg = (vl - 1) / kElementsPerRegister;
        const size_t last_elem = (vl - 1) % kElementsPerRegister;
        const auto mask_for_vl = MakeBitmaskFromVl(last_elem * kElementSize * 8);
        if (vma && ignore_vma_for_last) {
          // Set expected value for inactive element at vl-1 to Undisturbed.
          expected_inactive[last_reg] =
              (expected_inactive[last_reg] & ~mask_for_vl) | (kUndisturbedResult & mask_for_vl);
        }

        SIMD128Register expected_result[std::size(expected_result_raw)];
        for (size_t index = 0; index < std::size(expected_result_raw); ++index) {
          expected_result[index] = SIMD128Register{expected_result_raw[index]};
        }

        if (vlmul == 2 && last_elem_is_reg1) {
          switch (kElementSize) {
            case 1:
              expected_result[last_reg] = static_cast<__uint128_t>(static_cast<uint8_t>(int_res))
                                          << (last_elem * 8);
              break;
            case 2:
              expected_result[last_reg] = static_cast<__uint128_t>(static_cast<uint16_t>(int_res))
                                          << (last_elem * 16);
              break;
            case 4:
              if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
                expected_result[last_reg] =
                    static_cast<__uint128_t>(static_cast<uint32_t>(float_res)) << (last_elem * 32);
              } else {
                expected_result[last_reg] = static_cast<__uint128_t>(static_cast<uint32_t>(int_res))
                                            << (last_elem * 32);
              }
              break;
            case 8:
              if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
                expected_result[last_reg] =
                    static_cast<__uint128_t>(static_cast<uint64_t>(float_res)) << (last_elem * 64);
              } else {
                expected_result[last_reg] = static_cast<__uint128_t>(static_cast<uint64_t>(int_res))
                                            << (last_elem * 64);
              }
              break;
            default:
              FAIL() << "Element size is " << kElementSize;
          }
        }

        if (vlmul < 4) {
          for (size_t index = 0; index < num_regs; ++index) {
            if (index == 0 && vlmul == 2) {
              EXPECT_EQ(result[index],
                        (kUndisturbedResult & kFractionMaskInt8[3]) |
                            (expected_result[index] & (mask[index] | skip_mask[index]) &
                             ~kFractionMaskInt8[3]) |
                            (expected_inactive[index] & ~mask[index] & ~skip_mask[index] &
                             ~kFractionMaskInt8[3]));
            } else if (index == 2 && vlmul == 2) {
              EXPECT_EQ(result[index],
                        (expected_result[index] & (mask[index] | skip_mask[index]) &
                         kFractionMaskInt8[3]) |
                            (expected_inactive[index] & ~mask[index] & ~skip_mask[index] &
                             kFractionMaskInt8[3]) |
                            ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3]));
            } else if (index == 3 && vlmul == 2) {
              EXPECT_EQ(result[index], vta ? kAgnosticResult : kUndisturbedResult);
            } else {
              EXPECT_EQ(result[index],
                        (expected_result[index] & (mask[index] | skip_mask[index])) |
                            (expected_inactive[index] & ~(mask[index] | skip_mask[index])));
            }
          }
        } else {
          SIMD128Register affected_part{expected_result[0] &
                                        ((mask[0] & kFractionMaskInt8[vlmul - 4]) | skip_mask[0])};
          SIMD128Register masked_part{expected_inactive[0] & ~mask[0] & ~skip_mask[0] &
                                      kFractionMaskInt8[vlmul - 4]};
          SIMD128Register tail_part{(vta ? kAgnosticResult : kUndisturbedResult) &
                                    ~kFractionMaskInt8[vlmul - 4]};

          EXPECT_EQ(result[0], affected_part | masked_part | tail_part)
              << "vlmul=" << uint32_t{vlmul} << " vsew=" << uint32_t{vsew}
              << " vma=" << uint32_t{vma} << " vl=" << vl << " vstart=" << vstart
              << " affected_part=" << affected_part;
        }
      }
    }
  };

  // Test with and without masking enabled.
  (Verify(exec_masked_insn,
          BitUtilLog2(sizeof(std::tuple_element_t<0, ExpectedResultType>)),
          expected_result,
          MaskForElem<std::tuple_element_t<0, ExpectedResultType>>()),
   ...);
  (Verify(exec_insn,
          BitUtilLog2(sizeof(std::tuple_element_t<0, ExpectedResultType>)),
          expected_result,
          kNoMask),
   ...);
}

void TestVectorPermutationInstruction(ExecInsnFunc exec_insn,
                                      ExecInsnFunc exec_masked_insn,
                                      const UInt8x16Tuple (&expected_result_int8)[8],
                                      const UInt16x8Tuple (&expected_result_int16)[8],
                                      const UInt32x4Tuple (&expected_result_int32)[8],
                                      const UInt64x2Tuple (&expected_result_int64)[8],
                                      const SIMD128Register (&source)[16],
                                      uint8_t vlmul,
                                      uint64_t regt0 = 0x0,
                                      uint64_t skip = 0,
                                      bool ignore_vma_for_last = false,
                                      bool last_elem_is_x1 = false) {
  TestVectorPermutationInstruction<TestVectorInstructionKind::kInteger>(exec_insn,
                                                                        exec_masked_insn,
                                                                        source,
                                                                        vlmul,
                                                                        skip,
                                                                        ignore_vma_for_last,
                                                                        last_elem_is_x1,
                                                                        regt0,
                                                                        expected_result_int8,
                                                                        expected_result_int16,
                                                                        expected_result_int32,
                                                                        expected_result_int64);
}

template <typename... ExpectedResultType>
void TestVectorMaskTargetInstruction(ExecInsnFunc exec_insn,
                                     ExecInsnFunc exec_masked_insn,
                                     const SIMD128Register (&source)[16],
                                     const ExpectedResultType(&... expected_result)) {
  auto Verify = [&source](
                    ExecInsnFunc exec_insn, uint8_t vsew, const auto& expected_result, auto mask) {
    // Set t0 for vx instructions.
    uint64_t scalar_src = 0xaaaa'aaaa'aaaa'aaaa;

    // Set ft0 for vf instructions.
    if (vsew == 2) {
      scalar_src = 0xffff'ffff'40b4'0000;  // float 5.625
    } else if (vsew == 3) {
      scalar_src = 0x4016'8000'0000'0000;  // double 5.625
    }
    for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
      for (uint8_t vta = 0; vta < 2; ++vta) {  // vta should be ignored but we test both values!
        for (uint8_t vma = 0; vma < 2; ++vma) {
          uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
          uint64_t vlmax = 0;
          asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
          // Incompatible vsew and vlmax. Skip it.
          if (vlmax == 0) {
            continue;
          }

          // To make tests quick enough we don't test vstart and vl change with small register
          // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
          // register, last register and half of next-to last register.
          // Don't use vlmul == 3 because that one may not be supported if instruction widens the
          // result.
          uint64_t vstart;
          uint64_t vl;
          if (vlmul == 2) {
            vstart = vlmax / 8;
            vl = (vlmax * 5) / 8;
          } else {
            vstart = 0;
            vl = vlmax;
          }

          SIMD128Register result[8];
          // Set expected_result vector registers into 0b01010101… pattern.
          // Set undisturbed result vector registers.
          std::fill_n(result, 8, kUndisturbedResult);

          RunCommonVectorFunc(
              exec_insn, &source[0], &result[0], nullptr, nullptr, scalar_src, vstart, vtype, vl);

          SIMD128Register expected_result_in_register(expected_result);
          expected_result_in_register = (expected_result_in_register & mask) |
                                        ((vma ? kAgnosticResult : kUndisturbedResult) & ~mask);
          // Mask registers are always processing tail like vta is set.
          if (vlmax != 128) {
            const SIMD128Register vl_mask = MakeBitmaskFromVl(vl);
            expected_result_in_register =
                (kAgnosticResult & vl_mask) | (expected_result_in_register & ~vl_mask);
          }
          if (vlmul == 2) {
            const SIMD128Register start_mask = MakeBitmaskFromVl(vstart);
            expected_result_in_register =
                (kUndisturbedResult & ~start_mask) | (expected_result_in_register & start_mask);
          }
          EXPECT_EQ(result[0], expected_result_in_register)
              << "vlmul=" << uint32_t{vlmul} << " vsew=" << uint32_t{vsew}
              << " vma=" << uint32_t{vma} << " vl=" << vl << " vstart=" << vstart;
        }
      }
    }
  };

  ((Verify(exec_insn,
           BitUtilLog2(sizeof(__uint128_t) / sizeof(ExpectedResultType)),
           expected_result,
           kNoMask[0]),
    Verify(exec_masked_insn,
           BitUtilLog2(sizeof(__uint128_t) / sizeof(ExpectedResultType)),
           expected_result,
           kMask)),
   ...);
}

void TestVectorMaskTargetInstruction(ExecInsnFunc exec_insn,
                                     ExecInsnFunc exec_masked_insn,
                                     const uint32_t expected_result_int32,
                                     const uint16_t expected_result_int64,
                                     const SIMD128Register (&source)[16]) {
  TestVectorMaskTargetInstruction(
      exec_insn, exec_masked_insn, source, expected_result_int32, expected_result_int64);
}

void TestVectorMaskTargetInstruction(ExecInsnFunc exec_insn,
                                     ExecInsnFunc exec_masked_insn,
                                     const UInt8x16Tuple expected_result_int8,
                                     const uint64_t expected_result_int16,
                                     const uint32_t expected_result_int32,
                                     const uint16_t expected_result_int64,
                                     const SIMD128Register (&source)[16]) {
  TestVectorMaskTargetInstruction(exec_insn,
                                  exec_masked_insn,
                                  source,
                                  expected_result_int8,
                                  expected_result_int16,
                                  expected_result_int32,
                                  expected_result_int64);
}

// clang-format off
#define DEFINE_TWO_ARG_ONE_RES_FUNCTION(Name, Asm) \
  [[gnu::naked]] void Exec##Name() {               \
    asm(#Asm " v8, v16, v24\n\t"                   \
        "ret\n\t");                                \
  }                                                \
  [[gnu::naked]] void ExecMasked##Name() {         \
    asm(#Asm " v8, v16, v24, v0.t\n\t"             \
        "ret\n\t");                                \
  }
// clang-format on

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredsum, vredsum.vs)

TEST(InlineAsmTestRiscv64, TestVredsum) {
  TestVectorReductionInstruction(
      ExecVredsum,
      ExecMaskedVredsum,
      // expected_result_vd0_int8
      {242, 228, 200, 144, /* unused */ 0, 146, 44, 121},
      // expected_result_vd0_int16
      {0x0172, 0x82e4, 0x88c8, 0xa090, /* unused */ 0, 0x1300, 0xa904, 0xe119},
      // expected_result_vd0_int32
      {0xcb44'b932,
       0x9407'71e4,
       0xa70e'64c8,
       0xd312'5090,
       /* unused */ 0,
       /* unused */ 0,
       0x1907'1300,
       0xb713'ad09},
      // expected_result_vd0_int64
      {0xb32f'a926'9f1b'9511,
       0x1f99'0d88'fb74'e962,
       0xb92c'970e'74e8'52c4,
       0xef4e'ad14'6aca'2888,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x2513'1f0e'1907'1300},
      // expected_result_vd0_with_mask_int8
      {39, 248, 142, 27, /* unused */ 0, 0, 154, 210},
      // expected_result_vd0_with_mask_int16
      {0x5f45, 0xc22f, 0x99d0, 0x98bf, /* unused */ 0, 0x1300, 0x1300, 0x4b15},
      // expected_result_vd0_with_mask_int32
      {0x2d38'1f29,
       0x99a1'838a,
       0x1989'ef5c,
       0x9cf4'4aa1,
       /* unused */ 0,
       /* unused */ 0,
       0x1907'1300,
       0x1907'1300},
      // expected_result_vd0_with_mask_int64
      {0x2513'1f0e'1907'1300,
       0x917c'8370'7560'6751,
       0x4e56'3842'222a'0c13,
       0xc833'9e0e'73df'49b5,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x2513'1f0e'1907'1300},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vwredsumu, vwredsumu.vs)

TEST(InlineAsmTestRiscv64, TestVwredsumu) {
  TestWideningVectorReductionInstruction(
      ExecVwredsumu,
      ExecMaskedVwredsumu,
      // expected_result_vd0_int16
      {0x85f2, 0x8ce4, 0xa0c8, 0xc090, /* unused */ 0, 0x8192, 0x822c, 0x8379},
      // expected_result_vd0_int32
      {0x8307'0172,
       0x830c'82e4,
       0x831a'88c8,
       0x8322'a090,
       /* unused */ 0,
       0x8303'1300,
       0x8303'a904,
       0x8304'e119},
      // expected_result_vd0_int64
      {0x8706'8506'cb44'b932,
       0x8706'8509'9407'71e4,
       0x8706'8510'a70e'64c8,
       0x8706'8514'd312'5090,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8505'1907'1300,
       0x8706'8505'b713'ad09},
      // expected_result_vd0_with_mask_int16
      {0x8427, 0x88f8, 0x948e, 0xab1b, /* unused */ 0, 0x8100, 0x819a, 0x82d2},
      // expected_result_vd0_with_mask_int32
      {0x8305'5f45,
       0x8308'c22f,
       0x8311'99d0,
       0x8316'98bf,
       /* unused */ 0,
       0x8303'1300,
       0x8303'1300,
       0x8304'4b15},
      // expected_result_vd0_with_mask_int64
      {0x8706'8506'2d38'1f29,
       0x8706'8507'99a1'838a,
       0x8706'850c'1989'ef5c,
       0x8706'850e'9cf4'4aa1,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8505'1907'1300,
       0x8706'8505'1907'1300},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vwredsum, vwredsum.vs)

TEST(InlineAsmTestRiscv64, TestVwredsum) {
  TestWideningVectorReductionInstruction(
      ExecVwredsum,
      ExecMaskedVwredsum,
      // expected_result_vd0_int16
      {0x7df2, 0x7ce4, 0x80c8, 0x8090, /* unused */ 0, 0x8092, 0x802c, 0x7f79},
      // expected_result_vd0_int32
      {0x82ff'0172,
       0x82fc'82e4,
       0x82fa'88c8,
       0x8302'a090,
       /* unused */ 0,
       0x8302'1300,
       0x8301'a904,
       0x8300'e119},
      // expected_result_vd0_int64
      {0x8706'8502'cb44'b932,
       0x8706'8501'9407'71e4,
       0x8706'8500'a70e'64c8,
       0x8706'8504'd312'5090,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'1907'1300,
       0x8706'8503'b713'ad09},
      // expected_result_vd0_with_mask_int16
      {0x7f27, 0x7df8, 0x818e, 0x811b, /* unused */ 0, 0x8100, 0x809a, 0x7fd2},
      // expected_result_vd0_with_mask_int32
      {0x8300'5f45,
       0x82fe'c22f,
       0x82fd'99d0,
       0x8302'98bf,
       /* unused */ 0,
       0x8302'1300,
       0x8302'1300,
       0x8301'4b15},
      // expected_result_vd0_with_mask_int64
      {0x8706'8503'2d38'1f29,
       0x8706'8502'99a1'838a,
       0x8706'8502'1989'ef5c,
       0x8706'8504'9cf4'4aa1,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'1907'1300,
       0x8706'8504'1907'1300},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfredosum, vfredosum.vs)

TEST(InlineAsmTestRiscv64, TestVfredosum) {
  TestVectorReductionInstruction(ExecVfredosum,
                                 ExecMaskedVfredosum,
                                 // expected_result_vd0_int32
                                 {0x9e0c'9a8e,
                                  0xbe2c'bace,
                                  0xfe6c'fb4e,
                                  0x7e6b'fc4d,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9e0c'9a8e},
                                 // expected_result_vd0_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xfe6c'fa69'f664'f260,
                                  0x7eec'5def'0cee'0dee,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 // expected_result_vd0_with_mask_int32
                                 {0x9604'929d,
                                  0xbe2c'ba29,
                                  0xfe6c'fb4e,
                                  0x7e6b'fa84,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9604'9200},
                                 // expected_result_vd0_with_mask_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xee7c'ea78'e674'e271,
                                  0x6efc'4e0d'ee0d'ee0f,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfredusum, vfredusum.vs)

// Currently Vfredusum is implemented as Vfredosum (as explicitly permitted by RVV 1.0).
// If we would implement some speedups which would change results then we may need to alter tests.
TEST(InlineAsmTestRiscv64, TestVfredusum) {
  TestVectorReductionInstruction(ExecVfredusum,
                                 ExecMaskedVfredusum,
                                 // expected_result_vd0_int32
                                 {0x9e0c'9a8e,
                                  0xbe2c'bace,
                                  0xfe6c'fb4e,
                                  0x7e6b'fc4d,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9e0c'9a8e},
                                 // expected_result_vd0_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xfe6c'fa69'f664'f260,
                                  0x7eec'5def'0cee'0dee,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 // expected_result_vd0_with_mask_int32
                                 {0x9604'929d,
                                  0xbe2c'ba29,
                                  0xfe6c'fb4e,
                                  0x7e6b'fa84,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9604'9200},
                                 // expected_result_vd0_with_mask_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xee7c'ea78'e674'e271,
                                  0x6efc'4e0d'ee0d'ee0f,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfwredusum, vfwredusum.vs)

// We currently don't support half-precision (16-bit) floats, so only check 32-bit to 64-bit
// widening.
TEST(InlineAsmTestRiscv64, TestVfwredusum) {
  TestWideningVectorReductionInstruction(ExecVfwredusum,
                                         ExecMaskedVfwredusum,
                                         // expected_result_vd0_int64
                                         {0xbbc1'9351'b253'9156,
                                          0xbfc5'9759'b65b'955e,
                                          0xc7cd'9f69'be6b'9d6e,
                                          0x47cd'7f89'9e8b'7d8d,
                                          /* unused */ 0,
                                          /* unused */ 0,
                                          0xbac0'9240'0000'0000,
                                          0xbbc1'9351'b240'0000},
                                         // expected_result_vd0_with_mask_int64
                                         {0xbac0'9253'9155'9042,
                                          0xbfc5'9745'2017'9547,
                                          0xc7cd'9f69'be6b'9d4f,
                                          0x47cd'7f50'81d3'7d6f,
                                          /* unused */ 0,
                                          /* unused */ 0,
                                          0xbac0'9240'0000'0000,
                                          0xbac0'9240'0000'0000},
                                         kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfwredosum, vfwredosum.vs)

// We currently don't support half-precision (16-bit) floats, so only check 32-bit to 64-bit
// widening.
TEST(InlineAsmTestRiscv64, TestVfwredosum) {
  TestWideningVectorReductionInstruction(ExecVfwredosum,
                                         ExecMaskedVfwredosum,
                                         // expected_result_vd0_int64
                                         {0xbbc1'9351'b253'9156,
                                          0xbfc5'9759'b65b'955e,
                                          0xc7cd'9f69'be6b'9d6e,
                                          0x47cd'7f89'9e8b'7d8d,
                                          /* unused */ 0,
                                          /* unused */ 0,
                                          0xbac0'9240'0000'0000,
                                          0xbbc1'9351'b240'0000},
                                         // expected_result_vd0_with_mask_int64
                                         {0xbac0'9253'9155'9042,
                                          0xbfc5'9745'2017'9547,
                                          0xc7cd'9f69'be6b'9d4f,
                                          0x47cd'7f50'81d3'7d6f,
                                          /* unused */ 0,
                                          /* unused */ 0,
                                          0xbac0'9240'0000'0000,
                                          0xbac0'9240'0000'0000},
                                         kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredand, vredand.vs)

TEST(InlineAsmTestRiscv64, TestVredand) {
  TestVectorReductionInstruction(
      ExecVredand,
      ExecMaskedVredand,
      // expected_result_vd0_int8
      {0, 0, 0, 0, /* unused */ 0, 0, 0, 0},
      // expected_result_vd0_int16
      {0x8000, 0x8000, 0x8000, 0x0000, /* unused */ 0, 0x8000, 0x8000, 0x8000},
      // expected_result_vd0_int32
      {0x8200'8000,
       0x8200'8000,
       0x8200'8000,
       0x0200'0000,
       /* unused */ 0,
       /* unused */ 0,
       0x8200'8000,
       0x8200'8000},
      // expected_result_vd0_int64
      {0x8604'8000'8200'8000,
       0x8604'8000'8200'8000,
       0x8604'8000'8200'8000,
       0x0604'0000'0200'0000,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8604'8000'8200'8000},
      // expected_result_vd0_with_mask_int8
      {0, 0, 0, 0, /* unused */ 0, 0, 0, 0},
      // expected_result_vd0_with_mask_int16
      {0x8000, 0x8000, 0x8000, 0x0000, /* unused */ 0, 0x8000, 0x8000, 0x8000},
      // expected_result_vd0_with_mask_int32
      {0x8200'8000,
       0x8200'8000,
       0x8200'8000,
       0x0200'0000,
       /* unused */ 0,
       /* unused */ 0,
       0x8200'8000,
       0x8200'8000},
      // expected_result_vd0_with_mask_int64
      {0x8604'8000'8200'8000,
       0x8604'8000'8200'8000,
       0x8604'8000'8200'8000,
       0x0604'0000'0200'0000,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8604'8000'8200'8000},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredor, vredor.vs)

TEST(InlineAsmTestRiscv64, TestVredor) {
  TestVectorReductionInstruction(
      ExecVredor,
      ExecMaskedVredor,
      // expected_result_vd0_int8
      {159, 191, 255, 255, /* unused */ 0, 146, 150, 159},
      // expected_result_vd0_int16
      {0x9f1d, 0xbf3d, 0xff7d, 0xfffd, /* unused */ 0, 0x9300, 0x9704, 0x9f0d},
      // expected_result_vd0_int32
      {0x9f1e'9b19,
       0xbf3e'bb39,
       0xff7e'fb79,
       0xfffe'fbf9,
       /* unused */ 0,
       /* unused */ 0,
       0x9706'9300,
       0x9f0e'9b09},
      // expected_result_vd0_int64
      {0x9f1e'9f1d'9716'9311,
       0xbf3e'bf3d'b736'b331,
       0xff7e'ff7d'f776'f371,
       0xfffe'fffd'f7f6'f3f1,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9f0e'9f0d'9706'9300},
      // expected_result_vd0_with_mask_int8
      {159, 191, 255, 255, /* unused */ 0, 0, 150, 158},
      // expected_result_vd0_with_mask_int16
      {0x9f1d, 0xbf3d, 0xff7d, 0xfffd, /* unused */ 0, 0x9300, 0x9300, 0x9f0d},
      // expected_result_vd0_with_mask_int32
      {0x9f1e'9b19,
       0xbf3e'bb39,
       0xff7e'fb79,
       0xfffe'fbf9,
       /* unused */ 0,
       /* unused */ 0,
       0x9706'9300,
       0x9706'9300},
      // expected_result_vd0_with_mask_int64
      {0x9f0e'9f0d'9706'9300,
       0xbf3e'bf3d'b736'b331,
       0xff7e'ff7d'f776'f371,
       0xfffe'fffd'f7f6'f3f1,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9f0e'9f0d'9706'9300},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredxor, vredxor.vs)

TEST(InlineAsmTestRiscv64, TestVredxor) {
  TestVectorReductionInstruction(
      ExecVredxor,
      ExecMaskedVredxor,
      // expected_result_vd0_int8
      {0, 0, 0, 0, /* unused */ 0, 146, 0, 1},
      // expected_result_vd0_int16
      {0x8100, 0x8100, 0x8100, 0x8100, /* unused */ 0, 0x1300, 0x8504, 0x8101},
      // expected_result_vd0_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       0x1506'1300,
       0x8b0a'8909},
      // expected_result_vd0_int64
      {0x9716'9515'9312'9111,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x190a'1f0d'1506'1300},
      // expected_result_vd0_with_mask_int8
      {143, 154, 150, 43, /* unused */ 0, 0, 146, 150},
      // expected_result_vd0_with_mask_int16
      {0x1f0d, 0xbd3d, 0x9514, 0x8d0d, /* unused */ 0, 0x1300, 0x1300, 0x1705},
      // expected_result_vd0_with_mask_int32
      {0x1d0e'1b09,
       0x0d1e'0b18,
       0xfb7a'f978,
       0xab2a'a929,
       /* unused */ 0,
       /* unused */ 0,
       0x1506'1300,
       0x1506'1300},
      // expected_result_vd0_with_mask_int64
      {0x190a'1f0d'1506'1300,
       0x091a'0f1c'0516'0311,
       0x293a'2f3c'2536'2331,
       0x77f6'75f5'73f2'71f1,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x190a'1f0d'1506'1300},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredminu, vredminu.vs)

TEST(InlineAsmTestRiscv64, TestVredminu) {
  TestVectorReductionInstruction(
      ExecVredminu,
      ExecMaskedVredminu,
      // expected_result_vd0_int8
      {0, 0, 0, 0, /* unused */ 0, 0, 0, 0},
      // expected_result_vd0_int16
      {0x8100, 0x8100, 0x8100, 0x0291, /* unused */ 0, 0x8100, 0x8100, 0x8100},
      // expected_result_vd0_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x0694'0291,
       /* unused */ 0,
       /* unused */ 0,
       0x8302'8100,
       0x8302'8100},
      // expected_result_vd0_int64
      {0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x0e9c'0a98'0694'0291,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'8302'8100},
      // expected_result_vd0_with_mask_int8
      {0, 0, 0, 0, /* unused */ 0, 0, 0, 0},
      // expected_result_vd0_with_mask_int16
      {0x8100, 0x8100, 0x8100, 0x0291, /* unused */ 0, 0x8100, 0x8100, 0x8100},
      // expected_result_vd0_with_mask_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x0e9c'0a98,
       /* unused */ 0,
       /* unused */ 0,
       0x8302'8100,
       0x8302'8100},
      // expected_result_vd0_with_mask_int64
      {0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x1e8c'1a89'1684'1280,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'8302'8100},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredmin, vredmin.vs)

TEST(InlineAsmTestRiscv64, TestVredmin) {
  TestVectorReductionInstruction(
      ExecVredmin,
      ExecMaskedVredmin,
      // expected_result_vd0_int8
      {130, 130, 130, 128, /* unused */ 0, 146, 146, 146},
      // expected_result_vd0_int16
      {0x8100, 0x8100, 0x8100, 0x8100, /* unused */ 0, 0x8100, 0x8100, 0x8100},
      // expected_result_vd0_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       0x8302'8100,
       0x8302'8100},
      // expected_result_vd0_int64
      {0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'8302'8100},
      // expected_result_vd0_with_mask_int8
      {138, 138, 138, 128, /* unused */ 0, 0, 150, 150},
      // expected_result_vd0_with_mask_int16
      {0x8100, 0x8100, 0x8100, 0x8100, /* unused */ 0, 0x8100, 0x8100, 0x8100},
      // expected_result_vd0_with_mask_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       0x8302'8100,
       0x8302'8100},
      // expected_result_vd0_with_mask_int64
      {0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'8302'8100},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfredmin, vfredmin.vs)

TEST(InlineAsmTestRiscv64, TestVfredmin) {
  TestVectorReductionInstruction(ExecVfredmin,
                                 ExecMaskedVfredmin,
                                 // expected_result_vd0_int32
                                 {0x9e0c'9a09,
                                  0xbe2c'ba29,
                                  0xfe6c'fa69,
                                  0xfe6c'fa69,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9e0c'9a09},
                                 // expected_result_vd0_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xfe6c'fa69'f664'f260,
                                  0xfe6c'fa69'f664'f260,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 // expected_result_vd0_with_mask_int32
                                 {0x9604'9200,
                                  0xbe2c'ba29,
                                  0xfe6c'fa69,
                                  0xfe6c'fa69,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9604'9200},
                                 // expected_result_vd0_with_mask_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xee7c'ea78'e674'e271,
                                  0xee7c'ea78'e674'e271,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredmaxu, vredmaxu.vs)

TEST(InlineAsmTestRiscv64, TestVredmaxu) {
  TestVectorReductionInstruction(
      ExecVredmaxu,
      ExecMaskedVredmaxu,
      // expected_result_vd0_int8
      {158, 190, 254, 254, /* unused */ 0, 146, 150, 158},
      // expected_result_vd0_int16
      {0x9e0c, 0xbe2c, 0xfe6c, 0xfe6c, /* unused */ 0, 0x9200, 0x9604, 0x9e0c},
      // expected_result_vd0_int32
      {0x9e0c'9a09,
       0xbe2c'ba29,
       0xfe6c'fa69,
       0xfe6c'fa69,
       /* unused */ 0,
       /* unused */ 0,
       0x9604'9200,
       0x9e0c'9a09},
      // expected_result_vd0_int64
      {0x9e0c'9a09'9604'9200,
       0xbe2c'ba29'b624'b220,
       0xfe6c'fa69'f664'f260,
       0xfe6c'fa69'f664'f260,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9e0c'9a09'9604'9200},
      // expected_result_vd0_with_mask_int8
      {158, 186, 254, 254, /* unused */ 0, 0, 150, 158},
      // expected_result_vd0_with_mask_int16
      {0x9e0c, 0xba29, 0xfe6c, 0xfe6c, /* unused */ 0, 0x9200, 0x9200, 0x9e0c},
      // expected_result_vd0_with_mask_int32
      {0x9604'9200,
       0xbe2c'ba29,
       0xfe6c'fa69,
       0xfe6c'fa69,
       /* unused */ 0,
       /* unused */ 0,
       0x9604'9200,
       0x9604'9200},
      // expected_result_vd0_with_mask_int64
      {0x9e0c'9a09'9604'9200,
       0xbe2c'ba29'b624'b220,
       0xee7c'ea78'e674'e271,
       0xee7c'ea78'e674'e271,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9e0c'9a09'9604'9200},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredmax, vredmax.vs)

TEST(InlineAsmTestRiscv64, TestVredmax) {
  TestVectorReductionInstruction(
      ExecVredmax,
      ExecMaskedVredmax,
      // expected_result_vd0_int8
      {28, 60, 124, 126, /* unused */ 0, 0, 4, 12},
      // expected_result_vd0_int16
      {0x9e0c, 0xbe2c, 0xfe6c, 0x7eec, /* unused */ 0, 0x9200, 0x9604, 0x9e0c},
      // expected_result_vd0_int32
      {0x9e0c'9a09,
       0xbe2c'ba29,
       0xfe6c'fa69,
       0x7eec'7ae9,
       /* unused */ 0,
       /* unused */ 0,
       0x9604'9200,
       0x9e0c'9a09},
      // expected_result_vd0_int64
      {0x9e0c'9a09'9604'9200,
       0xbe2c'ba29'b624'b220,
       0xfe6c'fa69'f664'f260,
       0x7eec'7ae9'76e4'72e0,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9e0c'9a09'9604'9200},
      // expected_result_vd0_with_mask_int8
      {24, 52, 124, 126, /* unused */ 0, 0, 4, 4},
      // expected_result_vd0_with_mask_int16
      {0x9e0c, 0xba29, 0xfe6c, 0x7ae9, /* unused */ 0, 0x9200, 0x9200, 0x9e0c},
      // expected_result_vd0_with_mask_int32
      {0x9604'9200,
       0xbe2c'ba29,
       0xfe6c'fa69,
       0x7eec'7ae9,
       /* unused */ 0,
       /* unused */ 0,
       0x9604'9200,
       0x9604'9200},
      // expected_result_vd0_with_mask_int64
      {0x9e0c'9a09'9604'9200,
       0xbe2c'ba29'b624'b220,
       0xee7c'ea78'e674'e271,
       0x6efc'6af8'66f4'62f1,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9e0c'9a09'9604'9200},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfredmax, vfredmax.vs)

TEST(InlineAsmTestRiscv64, TestVfredmax) {
  TestVectorReductionInstruction(ExecVfredmax,
                                 ExecMaskedVfredmax,
                                 // expected_result_vd0_int32
                                 {0x8302'8100,
                                  0x8302'8100,
                                  0x8302'8100,
                                  0x7eec'7ae9,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x8302'8100,
                                  0x8302'8100},
                                 // expected_result_vd0_int64
                                 {0x8706'8504'8302'8100,
                                  0x8706'8504'8302'8100,
                                  0x8706'8504'8302'8100,
                                  0x7eec'7ae9'76e4'72e0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x8706'8504'8302'8100},
                                 // expected_result_vd0_with_mask_int32
                                 {0x8302'8100,
                                  0x8302'8100,
                                  0x8302'8100,
                                  0x7eec'7ae9,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x8302'8100,
                                  0x8302'8100},
                                 // expected_result_vd0_with_mask_int64
                                 {0x8706'8504'8302'8100,
                                  0x8706'8504'8302'8100,
                                  0x8706'8504'8302'8100,
                                  0x6efc'6af8'66f4'62f1,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x8706'8504'8302'8100},
                                 kVectorCalculationsSource);
}

#undef DEFINE_TWO_ARG_ONE_RES_FUNCTION

[[gnu::naked]] void ExecVfsqrtv() {
  asm("vfsqrt.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfsqrtv() {
  asm("vfsqrt.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVfsqrtv) {
  TestVectorFloatInstruction(ExecVfsqrtv,
                             ExecMaskedVfsqrtv,
                             {{0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x2b02'052b, 0x2f05'ea47, 0x2309'a451, 0x270d'53b1},
                              {0x3b10'f937, 0x3f14'7a09, 0x3317'd8b1, 0x371b'31d0},
                              {0x4b1e'85c1, 0x4f21'bb83, 0x4324'd4da, 0x4727'ebbf},
                              {0x5b2b'0054, 0x5f2d'fb2f, 0x5330'dd9e, 0x5733'bf97}},
                             {{0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x2f3d'fd15'c59f'19b3, 0x2745'2e80'5593'4661},
                              {0x3f4e'0e34'c013'd37a, 0x3755'3a9e'ffea'ec9f},
                              {0x4f5e'1f49'ff52'69b6, 0x4765'46b6'c2dc'cddd},
                              {0x5f6e'3055'93df'fb07, 0x5775'52c7'aa27'df73}},
                             kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVfcvtxufv() {
  asm("vfcvt.xu.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtxufv() {
  asm("vfcvt.xu.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtxfv() {
  asm("vfcvt.x.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtxfv() {
  asm("vfcvt.x.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtfxuv() {
  asm("vfcvt.f.xu.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtfxuv() {
  asm("vfcvt.f.xu.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtfxv() {
  asm("vfcvt.f.x.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtfxv() {
  asm("vfcvt.f.x.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtrtzxuf() {
  asm("vfcvt.rtz.xu.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtrtzxuf() {
  asm("vfcvt.rtz.xu.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtrtzxf() {
  asm("vfcvt.rtz.x.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtrtzxf() {
  asm("vfcvt.rtz.x.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtxufv() {
  asm("vfwcvt.xu.f.v v8, v28\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtxufv() {
  asm("vfwcvt.xu.f.v v8, v28, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtxfv() {
  asm("vfwcvt.x.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtxfv() {
  asm("vfwcvt.x.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtffv() {
  asm("vfwcvt.f.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtffv() {
  asm("vfwcvt.f.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtfxuv() {
  asm("vfwcvt.f.xu.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtfxuv() {
  asm("vfwcvt.f.xu.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtfxv() {
  asm("vfwcvt.f.x.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtfxv() {
  asm("vfwcvt.f.x.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtrtzxuf() {
  asm("vfwcvt.rtz.xu.f.v v8, v28\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtrtzxuf() {
  asm("vfwcvt.rtz.xu.f.v v8, v28, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtrtzxf() {
  asm("vfwcvt.rtz.x.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtrtzxf() {
  asm("vfwcvt.rtz.x.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtxufw() {
  asm("vfncvt.xu.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtxufw() {
  asm("vfncvt.xu.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtxfw() {
  asm("vfncvt.x.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtxfw() {
  asm("vfncvt.x.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtffw() {
  asm("vfncvt.f.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtffw() {
  asm("vfncvt.f.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtfxuw() {
  asm("vfncvt.f.xu.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtfxuw() {
  asm("vfncvt.f.xu.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtfxw() {
  asm("vfncvt.f.x.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtfxw() {
  asm("vfncvt.f.x.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtrtzxuf() {
  asm("vfncvt.rtz.xu.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtrtzxuf() {
  asm("vfncvt.rtz.xu.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtrtzxfw() {
  asm("vfncvt.rtz.x.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtrtzxfw() {
  asm("vfncvt.rtz.x.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVfcvtxfv) {
  TestVectorFloatInstruction(ExecVfcvtxufv,
                             ExecMaskedVfcvtxufv,
                             {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0xffff'ffff, 0xffff'ffff, 0x0000'6a21, 0x6e25'6c00},
                              {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                             {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
                              {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtxfv,
                             ExecMaskedVfcvtxfv,
                             {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x8000'0000, 0x8000'0000, 0xffff'cacf, 0xc8cd'6a00},
                              {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x7fff'ffff, 0x7fff'ffff, 0x0000'6a21, 0x6e25'6c00},
                              {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff}},
                             {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff},
                              {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtfxuv,
                             ExecMaskedVfcvtfxuv,
                             {{0x4f16'0492, 0x4f1e'0c9a, 0x4f06'1482, 0x4f0e'1c8a},
                              {0x4f36'24b2, 0x4f3e'2cba, 0x4f26'34a2, 0x4f2e'3caa},
                              {0x4f56'44d2, 0x4f5e'4cda, 0x4f46'54c2, 0x4f4e'5cca},
                              {0x4f76'64f2, 0x4f7e'6cfa, 0x4f66'74e2, 0x4f6e'7cea},
                              {0x4db4'2094, 0x4df4'60d4, 0x4cd2'8052, 0x4d69'c0aa},
                              {0x4e5a'90ca, 0x4e7a'b0eb, 0x4e1a'd08b, 0x4e3a'f0ab},
                              {0x4ead'88a6, 0x4ebd'98b6, 0x4e8d'a886, 0x4e9d'b896},
                              {0x4eed'c8e6, 0x4efd'd8f6, 0x4ecd'e8c6, 0x4edd'f8d6}},
                             {{0x43e3'c193'4132'c092, 0x43e1'c391'4310'c290},
                              {0x43e7'c597'4536'c496, 0x43e5'c795'4714'c694},
                              {0x43eb'c99b'493a'c89a, 0x43e9'cb99'4b18'ca98},
                              {0x43ef'cd9f'4d3e'cc9e, 0x43ed'cf9d'4f1c'ce9c},
                              {0x43be'8c1a'8916'8412, 0x43ad'3815'300d'2805},
                              {0x43cf'561d'549b'5219, 0x43c7'5e15'5c13'5a11},
                              {0x43d7'b316'b255'b115, 0x43d3'b712'b611'b511},
                              {0x43df'bb1e'ba5d'b91d, 0x43db'bf1a'be19'bd19}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtfxv,
                             ExecMaskedVfcvtfxv,
                             {{0xced3'f6dc, 0xcec3'e6cc, 0xcef3'd6fc, 0xcee3'c6ec},
                              {0xce93'b69c, 0xce83'a68c, 0xceb3'96bc, 0xcea3'86ac},
                              {0xce26'ecb7, 0xce06'cc97, 0xce66'acf7, 0xce46'8cd7},
                              {0xcd19'b0da, 0xcbc9'82cc, 0xcdcc'58ec, 0xcd8c'18ac},
                              {0x4db4'2094, 0x4df4'60d4, 0x4cd2'8052, 0x4d69'c0aa},
                              {0x4e5a'90ca, 0x4e7a'b0eb, 0x4e1a'd08b, 0x4e3a'f0ab},
                              {0x4ead'88a6, 0x4ebd'98b6, 0x4e8d'a886, 0x4e9d'b896},
                              {0x4eed'c8e6, 0x4efd'd8f6, 0x4ecd'e8c6, 0x4edd'f8d6}},
                             {{0xc3d8'7cd9'7d9a'7edc, 0xc3dc'78dd'79de'7adf},
                              {0xc3d0'74d1'7592'76d3, 0xc3d4'70d5'71d6'72d7},
                              {0xc3c0'd992'db14'dd97, 0xc3c8'd19a'd39c'd59f},
                              {0xc379'3059'6099'b0da, 0xc3b1'8315'8719'8b1e},
                              {0x43be'8c1a'8916'8412, 0x43ad'3815'300d'2805},
                              {0x43cf'561d'549b'5219, 0x43c7'5e15'5c13'5a11},
                              {0x43d7'b316'b255'b115, 0x43d3'b712'b611'b511},
                              {0x43df'bb1e'ba5d'b91d, 0x43db'bf1a'be19'bd19}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtrtzxuf,
                             ExecMaskedVfcvtrtzxuf,
                             {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0xffff'ffff, 0xffff'ffff, 0x0000'6a21, 0x6e25'6c00},
                              {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                             {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
                              {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtrtzxf,
                             ExecMaskedVfcvtrtzxf,
                             {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x8000'0000, 0x8000'0000, 0xffff'cad0, 0xc8cd'6a00},
                              {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x7fff'ffff, 0x7fff'ffff, 0x0000'6a21, 0x6e25'6c00},
                              {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff}},
                             {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff},
                              {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff}},
                             kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtxufv,
                                     ExecMaskedVfwcvtxufv,
                                     {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'6229'6000'0000, 0x662d'6480'0000'0000},
                                      {0x0000'0000'0000'6a21, 0x0000'0000'6e25'6c00},
                                      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
                                      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtxfv,
                                     ExecMaskedVfwcvtxfv,
                                     {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0xffff'cecb'7000'0000, 0xccc9'6dc0'0000'0000},
                                      {0xffff'ffff'ffff'cacf, 0xffff'ffff'c8cd'6a00},
                                      {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                                      {0x8000'0000'0000'0000, 0x8000'0000'0000'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtffv,
                                     ExecMaskedVfwcvtffv,
                                     {{0xbac0'9240'0000'0000, 0xbbc1'9341'2000'0000},
                                      {0xb8c2'9042'2000'0000, 0xb9c3'9143'0000'0000},
                                      {0xbec4'9644'0000'0000, 0xbfc5'9745'2000'0000},
                                      {0xbcc6'9446'2000'0000, 0xbdc7'9547'0000'0000},
                                      {0xc2c8'9a48'0000'0000, 0xc3c9'9b49'2000'0000},
                                      {0xc0ca'984a'2000'0000, 0xc1cb'994b'0000'0000},
                                      {0xc6cc'9e4c'0000'0000, 0xc7cd'9f4d'2000'0000},
                                      {0xc4ce'9c4e'2000'0000, 0xc5cf'9d4f'0000'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtfxuv,
                                     ExecMaskedVfwcvtfxuv,
                                     {{0x4712'0000, 0x4716'0400, 0x471a'0900, 0x471e'0c00},
                                      {0x4702'1100, 0x4706'1400, 0x470a'1800, 0x470e'1c00},
                                      {0x4732'2000, 0x4736'2400, 0x473a'2900, 0x473e'2c00},
                                      {0x4722'3100, 0x4726'3400, 0x472a'3800, 0x472e'3c00},
                                      {0x4752'4000, 0x4756'4400, 0x475a'4900, 0x475e'4c00},
                                      {0x4742'5100, 0x4746'5400, 0x474a'5800, 0x474e'5c00},
                                      {0x4772'6000, 0x4776'6400, 0x477a'6900, 0x477e'6c00},
                                      {0x4762'7100, 0x4766'7400, 0x476a'7800, 0x476e'7c00}},
                                     {{0x41e2'c092'4000'0000, 0x41e3'c193'4120'0000},
                                      {0x41e0'c290'4220'0000, 0x41e1'c391'4300'0000},
                                      {0x41e6'c496'4400'0000, 0x41e7'c597'4520'0000},
                                      {0x41e4'c694'4620'0000, 0x41e5'c795'4700'0000},
                                      {0x41ea'c89a'4800'0000, 0x41eb'c99b'4920'0000},
                                      {0x41e8'ca98'4a20'0000, 0x41e9'cb99'4b00'0000},
                                      {0x41ee'cc9e'4c00'0000, 0x41ef'cd9f'4d20'0000},
                                      {0x41ec'ce9c'4e20'0000, 0x41ed'cf9d'4f00'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtfxv,
                                     ExecMaskedVfwcvtfxv,
                                     {{0xc6dc'0000, 0xc6d3'f800, 0xc6cb'ee00, 0xc6c3'e800},
                                      {0xc6fb'de00, 0xc6f3'd800, 0xc6eb'd000, 0xc6e3'c800},
                                      {0xc69b'c000, 0xc693'b800, 0xc68b'ae00, 0xc683'a800},
                                      {0xc6bb'9e00, 0xc6b3'9800, 0xc6ab'9000, 0xc6a3'8800},
                                      {0xc637'0000, 0xc626'f000, 0xc616'dc00, 0xc606'd000},
                                      {0xc676'bc00, 0xc666'b000, 0xc656'a000, 0xc646'9000},
                                      {0xc55a'0000, 0xc519'c000, 0xc4b2'e000, 0xc3ca'0000},
                                      {0xc5ec'7800, 0xc5cc'6000, 0xc5ac'4000, 0xc58c'2000}},
                                     {{0xc1da'7edb'8000'0000, 0xc1d8'7cd9'7dc0'0000},
                                      {0xc1de'7adf'7bc0'0000, 0xc1dc'78dd'7a00'0000},
                                      {0xc1d2'76d3'7800'0000, 0xc1d0'74d1'75c0'0000},
                                      {0xc1d6'72d7'73c0'0000, 0xc1d4'70d5'7200'0000},
                                      {0xc1c4'dd96'e000'0000, 0xc1c0'd992'db80'0000},
                                      {0xc1cc'd59e'd780'0000, 0xc1c8'd19a'd400'0000},
                                      {0xc1a3'361b'4000'0000, 0xc179'3059'7000'0000},
                                      {0xc1b9'8b1d'8f00'0000, 0xc1b1'8315'8800'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtrtzxuf,
                                     ExecMaskedVfwcvtrtzxuf,
                                     {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'6229'6000'0000, 0x662d'6480'0000'0000},
                                      {0x0000'0000'0000'6a21, 0x0000'0000'6e25'6c00},
                                      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
                                      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtrtzxf,
                                     ExecMaskedVfwcvtrtzxf,
                                     {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0xffff'cecb'7000'0000, 0xccc9'6dc0'0000'0000},
                                      {0xffff'ffff'ffff'cad0, 0xffff'ffff'c8cd'6a00},
                                      {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                                      {0x8000'0000'0000'0000, 0x8000'0000'0000'0000}},
                                     kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(
      ExecVfncvtxufw,
      ExecMaskedVfncvtxufw,
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0xffff, 0xffff, 0x6a21, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(
      ExecVfncvtxfw,
      ExecMaskedVfncvtxfw,
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x8000, 0x8000, 0xcacf, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x7fff, 0x7fff, 0x6a21, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff}},
      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(ExecVfncvtffw,
                                      ExecMaskedVfncvtffw,
                                      {{0x8000'0000, 0x8000'0000, 0xb165'd14e, 0x8000'0000},
                                       {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000},
                                       {0x0000'0000, 0x0000'0000, 0x3561'd54a, 0x0000'0000},
                                       {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                                      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(ExecVfncvtfxuw,
                                      ExecMaskedVfncvtfxuw,
                                      {{0x5f1e'0c9a, 0x5f0e'1c8a, 0x5f3e'2cba, 0x5f2e'3caa},
                                       {0x5f5e'4cda, 0x5f4e'5cca, 0x5f7e'6cfa, 0x5f6e'7cea},
                                       {0x5df4'60d4, 0x5d69'c0aa, 0x5e7a'b0eb, 0x5e3a'f0ab},
                                       {0x5ebd'98b6, 0x5e9d'b896, 0x5efd'd8f6, 0x5edd'f8d6}},
                                      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(ExecVfncvtfxw,
                                      ExecMaskedVfncvtfxw,
                                      {{0xdec3'e6cc, 0xdee3'c6ec, 0xde83'a68c, 0xdea3'86ac},
                                       {0xde06'cc97, 0xde46'8cd7, 0xdbc9'82cb, 0xdd8c'18ac},
                                       {0x5df4'60d4, 0x5d69'c0aa, 0x5e7a'b0eb, 0x5e3a'f0ab},
                                       {0x5ebd'98b6, 0x5e9d'b896, 0x5efd'd8f6, 0x5edd'f8d6}},
                                      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(
      ExecVfncvtrtzxuf,
      ExecMaskedVfncvtrtzxuf,
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0xffff, 0xffff, 0x6a21, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(
      ExecVfncvtrtzxfw,
      ExecMaskedVfncvtrtzxfw,
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x8000, 0x8000, 0xcad0, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x7fff, 0x7fff, 0x6a21, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff}},
      kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVid() {
  asm("vid.v v8\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVid() {
  asm("vid.v v8, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVid) {
  TestVectorInstruction(
      ExecVid,
      ExecMaskedVid,
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
       {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
       {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
       {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79},
       {80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95},
       {96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111},
       {112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}},
      {{0, 1, 2, 3, 4, 5, 6, 7},
       {8, 9, 10, 11, 12, 13, 14, 15},
       {16, 17, 18, 19, 20, 21, 22, 23},
       {24, 25, 26, 27, 28, 29, 30, 31},
       {32, 33, 34, 35, 36, 37, 38, 39},
       {40, 41, 42, 43, 44, 45, 46, 47},
       {48, 49, 50, 51, 52, 53, 54, 55},
       {56, 57, 58, 59, 60, 61, 62, 63}},
      {{0, 1, 2, 3},
       {4, 5, 6, 7},
       {8, 9, 10, 11},
       {12, 13, 14, 15},
       {16, 17, 18, 19},
       {20, 21, 22, 23},
       {24, 25, 26, 27},
       {28, 29, 30, 31}},
      {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecViotam() {
  asm("viota.m v8, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedViotam() {
  asm("viota.m v8, v16, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestIota) {
  TestVectorIota<false>(ExecViotam,
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1},
                         {2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5},
                         {6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9},
                         {10, 10, 11, 12, 12, 12, 12, 12, 12, 13, 14, 15, 15, 15, 15, 15},
                         {16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19},
                         {20, 20, 21, 21, 22, 22, 22, 22, 22, 23, 24, 24, 25, 25, 25, 25},
                         {26, 26, 26, 27, 28, 28, 28, 28, 28, 29, 29, 30, 31, 31, 31, 31},
                         {32, 32, 33, 34, 35, 35, 35, 35, 35, 36, 37, 38, 39, 39, 39, 39}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0002, 0x0002, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003},
                         {0x0003, 0x0004, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005},
                         {0x0006, 0x0006, 0x0006, 0x0007, 0x0007, 0x0007, 0x0007, 0x0007},
                         {0x0007, 0x0008, 0x0008, 0x0009, 0x0009, 0x0009, 0x0009, 0x0009},
                         {0x000a, 0x000a, 0x000b, 0x000c, 0x000c, 0x000c, 0x000c, 0x000c},
                         {0x000c, 0x000d, 0x000e, 0x000f, 0x000f, 0x000f, 0x000f, 0x000f}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0002, 0x0000'0002, 0x0000'0003, 0x0000'0003},
                         {0x0000'0003, 0x0000'0003, 0x0000'0003, 0x0000'0003},
                         {0x0000'0003, 0x0000'0004, 0x0000'0005, 0x0000'0005},
                         {0x0000'0005, 0x0000'0005, 0x0000'0005, 0x0000'0005}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorCalculationsSource);
  TestVectorIota<true>(ExecMaskedViotam,
                       {{0, 0x55, 0, 0, 0x55, 0, 0x55, 0, 0, 0x55, 1, 0x55, 1, 1, 0x55, 1},
                        {2, 2, 0x55, 3, 0x55, 3, 3, 0x55, 3, 0x55, 4, 4, 0x55, 4, 0x55, 4},
                        {5, 0x55, 5, 0x55, 6, 6, 0x55, 6, 0x55, 6, 6, 0x55, 7, 0x55, 7, 7},
                        {8, 0x55, 8, 9, 0x55, 9, 0x55, 9, 9, 0x55, 10, 0x55, 11, 0x55, 11, 11},
                        {12, 0x55, 12, 0x55, 12, 12, 0x55, 12, 12, 13, 0x55, 13, 14, 14, 14, 0x55},
                        {14, 0x55, 14, 14, 0x55, 15, 15, 15, 0x55, 15, 16, 16, 17, 0x55, 17, 17},
                        {18, 18, 0x55, 18, 19, 19, 0x55, 19, 19, 20, 20, 0x55, 21, 0x55, 21, 0x55},
                        {21, 21, 22, 0x55, 23, 23, 23, 23, 0x55, 23, 0x55, 24, 0x55, 25, 25, 0x55}},
                       {{0x0000, 0x5555, 0x0000, 0x0000, 0x5555, 0x0000, 0x5555, 0x0000},
                        {0x0000, 0x5555, 0x0001, 0x5555, 0x0001, 0x0001, 0x5555, 0x0001},
                        {0x0002, 0x0002, 0x5555, 0x0003, 0x5555, 0x0003, 0x0003, 0x5555},
                        {0x0003, 0x5555, 0x0004, 0x0004, 0x5555, 0x0004, 0x5555, 0x0004},
                        {0x0005, 0x5555, 0x0005, 0x5555, 0x0006, 0x0006, 0x5555, 0x0006},
                        {0x5555, 0x0006, 0x0006, 0x5555, 0x0007, 0x5555, 0x0007, 0x0007},
                        {0x0008, 0x5555, 0x0008, 0x0009, 0x5555, 0x0009, 0x5555, 0x0009},
                        {0x0009, 0x5555, 0x000a, 0x5555, 0x000b, 0x5555, 0x000b, 0x000b}},
                       {{0x0000'0000, 0x5555'5555, 0x0000'0000, 0x0000'0000},
                        {0x5555'5555, 0x0000'0000, 0x5555'5555, 0x0000'0000},
                        {0x0000'0000, 0x5555'5555, 0x0000'0001, 0x5555'5555},
                        {0x0000'0001, 0x0000'0001, 0x5555'5555, 0x0000'0001},
                        {0x0000'0002, 0x0000'0002, 0x5555'5555, 0x0000'0003},
                        {0x5555'5555, 0x0000'0003, 0x0000'0003, 0x5555'5555},
                        {0x0000'0003, 0x5555'5555, 0x0000'0004, 0x0000'0004},
                        {0x5555'5555, 0x0000'0004, 0x5555'5555, 0x0000'0004}},
                       {{0x0000'0000'0000'0000, 0x5555'5555'5555'5555},
                        {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                        {0x5555'5555'5555'5555, 0x0000'0000'0000'0000},
                        {0x5555'5555'5555'5555, 0x0000'0000'0000'0000},
                        {0x0000'0000'0000'0000, 0x5555'5555'5555'5555},
                        {0x0000'0000'0000'0001, 0x5555'5555'5555'5555},
                        {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                        {0x5555'5555'5555'5555, 0x0000'0000'0000'0001}},
                       kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVrsubvx() {
  asm("vrsub.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVrsubvx() {
  asm("vrsub.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVrsubvi() {
  asm("vrsub.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVrsubvi() {
  asm("vrsub.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfrsubvf() {
  asm("vfrsub.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfrsubvf() {
  asm("vfrsub.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVrsub) {
  TestVectorInstruction(
      ExecVrsubvx,
      ExecMaskedVrsubvx,
      {{170, 41, 168, 39, 166, 37, 164, 35, 162, 33, 160, 31, 158, 29, 156, 27},
       {154, 25, 152, 23, 150, 21, 148, 19, 146, 17, 144, 15, 142, 13, 140, 11},
       {138, 9, 136, 7, 134, 5, 132, 3, 130, 1, 128, 255, 126, 253, 124, 251},
       {122, 249, 120, 247, 118, 245, 116, 243, 114, 241, 112, 239, 110, 237, 108, 235},
       {106, 233, 104, 231, 102, 229, 100, 227, 98, 225, 96, 223, 94, 221, 92, 219},
       {90, 217, 88, 215, 86, 213, 84, 211, 82, 209, 80, 207, 78, 205, 76, 203},
       {74, 201, 72, 199, 70, 197, 68, 195, 66, 193, 64, 191, 62, 189, 60, 187},
       {58, 185, 56, 183, 54, 181, 52, 179, 50, 177, 48, 175, 46, 173, 44, 171}},
      {{0x29aa, 0x27a8, 0x25a6, 0x23a4, 0x21a2, 0x1fa0, 0x1d9e, 0x1b9c},
       {0x199a, 0x1798, 0x1596, 0x1394, 0x1192, 0x0f90, 0x0d8e, 0x0b8c},
       {0x098a, 0x0788, 0x0586, 0x0384, 0x0182, 0xff80, 0xfd7e, 0xfb7c},
       {0xf97a, 0xf778, 0xf576, 0xf374, 0xf172, 0xef70, 0xed6e, 0xeb6c},
       {0xe96a, 0xe768, 0xe566, 0xe364, 0xe162, 0xdf60, 0xdd5e, 0xdb5c},
       {0xd95a, 0xd758, 0xd556, 0xd354, 0xd152, 0xcf50, 0xcd4e, 0xcb4c},
       {0xc94a, 0xc748, 0xc546, 0xc344, 0xc142, 0xbf40, 0xbd3e, 0xbb3c},
       {0xb93a, 0xb738, 0xb536, 0xb334, 0xb132, 0xaf30, 0xad2e, 0xab2c}},
      {{0x27a8'29aa, 0x23a4'25a6, 0x1fa0'21a2, 0x1b9c'1d9e},
       {0x1798'199a, 0x1394'1596, 0x0f90'1192, 0x0b8c'0d8e},
       {0x0788'098a, 0x0384'0586, 0xff80'0182, 0xfb7b'fd7e},
       {0xf777'f97a, 0xf373'f576, 0xef6f'f172, 0xeb6b'ed6e},
       {0xe767'e96a, 0xe363'e566, 0xdf5f'e162, 0xdb5b'dd5e},
       {0xd757'd95a, 0xd353'd556, 0xcf4f'd152, 0xcb4b'cd4e},
       {0xc747'c94a, 0xc343'c546, 0xbf3f'c142, 0xbb3b'bd3e},
       {0xb737'b93a, 0xb333'b536, 0xaf2f'b132, 0xab2b'ad2e}},
      {{0x23a4'25a6'27a8'29aa, 0x1b9c'1d9e'1fa0'21a2},
       {0x1394'1596'1798'199a, 0x0b8c'0d8e'0f90'1192},
       {0x0384'0586'0788'098a, 0xfb7b'fd7d'ff80'0182},
       {0xf373'f575'f777'f97a, 0xeb6b'ed6d'ef6f'f172},
       {0xe363'e565'e767'e96a, 0xdb5b'dd5d'df5f'e162},
       {0xd353'd555'd757'd95a, 0xcb4b'cd4d'cf4f'd152},
       {0xc343'c545'c747'c94a, 0xbb3b'bd3d'bf3f'c142},
       {0xb333'b535'b737'b93a, 0xab2b'ad2d'af2f'b132}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVrsubvi,
      ExecMaskedVrsubvi,
      {{245, 116, 243, 114, 241, 112, 239, 110, 237, 108, 235, 106, 233, 104, 231, 102},
       {229, 100, 227, 98, 225, 96, 223, 94, 221, 92, 219, 90, 217, 88, 215, 86},
       {213, 84, 211, 82, 209, 80, 207, 78, 205, 76, 203, 74, 201, 72, 199, 70},
       {197, 68, 195, 66, 193, 64, 191, 62, 189, 60, 187, 58, 185, 56, 183, 54},
       {181, 52, 179, 50, 177, 48, 175, 46, 173, 44, 171, 42, 169, 40, 167, 38},
       {165, 36, 163, 34, 161, 32, 159, 30, 157, 28, 155, 26, 153, 24, 151, 22},
       {149, 20, 147, 18, 145, 16, 143, 14, 141, 12, 139, 10, 137, 8, 135, 6},
       {133, 4, 131, 2, 129, 0, 127, 254, 125, 252, 123, 250, 121, 248, 119, 246}},
      {{0x7ef5, 0x7cf3, 0x7af1, 0x78ef, 0x76ed, 0x74eb, 0x72e9, 0x70e7},
       {0x6ee5, 0x6ce3, 0x6ae1, 0x68df, 0x66dd, 0x64db, 0x62d9, 0x60d7},
       {0x5ed5, 0x5cd3, 0x5ad1, 0x58cf, 0x56cd, 0x54cb, 0x52c9, 0x50c7},
       {0x4ec5, 0x4cc3, 0x4ac1, 0x48bf, 0x46bd, 0x44bb, 0x42b9, 0x40b7},
       {0x3eb5, 0x3cb3, 0x3ab1, 0x38af, 0x36ad, 0x34ab, 0x32a9, 0x30a7},
       {0x2ea5, 0x2ca3, 0x2aa1, 0x289f, 0x269d, 0x249b, 0x2299, 0x2097},
       {0x1e95, 0x1c93, 0x1a91, 0x188f, 0x168d, 0x148b, 0x1289, 0x1087},
       {0x0e85, 0x0c83, 0x0a81, 0x087f, 0x067d, 0x047b, 0x0279, 0x0077}},
      {{0x7cfd'7ef5, 0x78f9'7af1, 0x74f5'76ed, 0x70f1'72e9},
       {0x6ced'6ee5, 0x68e9'6ae1, 0x64e5'66dd, 0x60e1'62d9},
       {0x5cdd'5ed5, 0x58d9'5ad1, 0x54d5'56cd, 0x50d1'52c9},
       {0x4ccd'4ec5, 0x48c9'4ac1, 0x44c5'46bd, 0x40c1'42b9},
       {0x3cbd'3eb5, 0x38b9'3ab1, 0x34b5'36ad, 0x30b1'32a9},
       {0x2cad'2ea5, 0x28a9'2aa1, 0x24a5'269d, 0x20a1'2299},
       {0x1c9d'1e95, 0x1899'1a91, 0x1495'168d, 0x1091'1289},
       {0x0c8d'0e85, 0x0889'0a81, 0x0485'067d, 0x0081'0279}},
      {{0x78f9'7afb'7cfd'7ef5, 0x70f1'72f3'74f5'76ed},
       {0x68e9'6aeb'6ced'6ee5, 0x60e1'62e3'64e5'66dd},
       {0x58d9'5adb'5cdd'5ed5, 0x50d1'52d3'54d5'56cd},
       {0x48c9'4acb'4ccd'4ec5, 0x40c1'42c3'44c5'46bd},
       {0x38b9'3abb'3cbd'3eb5, 0x30b1'32b3'34b5'36ad},
       {0x28a9'2aab'2cad'2ea5, 0x20a1'22a3'24a5'269d},
       {0x1899'1a9b'1c9d'1e95, 0x1091'1293'1495'168d},
       {0x0889'0a8b'0c8d'0e85, 0x0081'0283'0485'067d}},
      kVectorCalculationsSourceLegacy);

  TestVectorFloatInstruction(ExecVfrsubvf,
                             ExecMaskedVfrsubvf,
                             {{0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0017, 0x40b4'1757, 0x40cb'd7a8},
                              {0x4348'6140, 0x4746'cae4, 0x4b4a'c94e, 0x4f4e'cd4c},
                              {0x5352'd150, 0x5756'd554, 0x5b5a'd958, 0x5f5e'dd5c},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'807a'f4f2'eceb},
                              {0x4746'c544'c342'c140, 0x4f4e'cd4c'cb4a'c948},
                              {0x5756'd554'd352'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVaddvv() {
  asm("vadd.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaddvv() {
  asm("vadd.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaddvx() {
  asm("vadd.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaddvx() {
  asm("vadd.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaddvi() {
  asm("vadd.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaddvi() {
  asm("vadd.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsadduvv() {
  asm("vsaddu.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsadduvv() {
  asm("vsaddu.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsadduvx() {
  asm("vsaddu.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsadduvx() {
  asm("vsaddu.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsadduvi() {
  asm("vsaddu.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsadduvi() {
  asm("vsaddu.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsaddvv() {
  asm("vsadd.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsaddvv() {
  asm("vsadd.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsaddvx() {
  asm("vsadd.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsaddvx() {
  asm("vsadd.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsaddvi() {
  asm("vsadd.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsaddvi() {
  asm("vsadd.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfaddvv() {
  asm("vfadd.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfaddvv() {
  asm("vfadd.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfaddvf() {
  asm("vfadd.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfaddvf() {
  asm("vfadd.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwaddvv() {
  asm("vfwadd.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwaddvv() {
  asm("vfwadd.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwaddwv() {
  asm("vfwadd.wv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwaddwv() {
  asm("vfwadd.wv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwaddwf() {
  asm("vfwadd.wf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwaddwf() {
  asm("vfwadd.wf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVadd) {
  TestVectorInstruction(
      ExecVaddvv,
      ExecMaskedVaddvv,
      {{0, 131, 6, 137, 13, 143, 18, 149, 25, 155, 30, 161, 36, 167, 42, 173},
       {48, 179, 54, 185, 61, 191, 66, 197, 73, 203, 78, 209, 84, 215, 90, 221},
       {96, 227, 102, 233, 109, 239, 114, 245, 121, 251, 126, 1, 132, 7, 138, 13},
       {144, 19, 150, 25, 157, 31, 162, 37, 169, 43, 174, 49, 180, 55, 186, 61},
       {192, 67, 198, 73, 205, 79, 210, 85, 217, 91, 222, 97, 228, 103, 234, 109},
       {240, 115, 246, 121, 253, 127, 2, 133, 9, 139, 14, 145, 20, 151, 26, 157},
       {32, 163, 38, 169, 45, 175, 50, 181, 57, 187, 62, 193, 68, 199, 74, 205},
       {80, 211, 86, 217, 93, 223, 98, 229, 105, 235, 110, 241, 116, 247, 122, 253}},
      {{0x8300, 0x8906, 0x8f0d, 0x9512, 0x9b19, 0xa11e, 0xa724, 0xad2a},
       {0xb330, 0xb936, 0xbf3d, 0xc542, 0xcb49, 0xd14e, 0xd754, 0xdd5a},
       {0xe360, 0xe966, 0xef6d, 0xf572, 0xfb79, 0x017e, 0x0784, 0x0d8a},
       {0x1390, 0x1996, 0x1f9d, 0x25a2, 0x2ba9, 0x31ae, 0x37b4, 0x3dba},
       {0x43c0, 0x49c6, 0x4fcd, 0x55d2, 0x5bd9, 0x61de, 0x67e4, 0x6dea},
       {0x73f0, 0x79f6, 0x7ffd, 0x8602, 0x8c09, 0x920e, 0x9814, 0x9e1a},
       {0xa420, 0xaa26, 0xb02d, 0xb632, 0xbc39, 0xc23e, 0xc844, 0xce4a},
       {0xd450, 0xda56, 0xe05d, 0xe662, 0xec69, 0xf26e, 0xf874, 0xfe7a}},
      {{0x8906'8300, 0x9512'8f0d, 0xa11e'9b19, 0xad2a'a724},
       {0xb936'b330, 0xc542'bf3d, 0xd14e'cb49, 0xdd5a'd754},
       {0xe966'e360, 0xf572'ef6d, 0x017e'fb79, 0x0d8b'0784},
       {0x1997'1390, 0x25a3'1f9d, 0x31af'2ba9, 0x3dbb'37b4},
       {0x49c7'43c0, 0x55d3'4fcd, 0x61df'5bd9, 0x6deb'67e4},
       {0x79f7'73f0, 0x8603'7ffd, 0x920f'8c09, 0x9e1b'9814},
       {0xaa27'a420, 0xb633'b02d, 0xc23f'bc39, 0xce4b'c844},
       {0xda57'd450, 0xe663'e05d, 0xf26f'ec69, 0xfe7b'f874}},
      {{0x9512'8f0d'8906'8300, 0xad2a'a724'a11e'9b19},
       {0xc542'bf3d'b936'b330, 0xdd5a'd754'd14e'cb49},
       {0xf572'ef6d'e966'e360, 0x0d8b'0785'017e'fb79},
       {0x25a3'1f9e'1997'1390, 0x3dbb'37b5'31af'2ba9},
       {0x55d3'4fce'49c7'43c0, 0x6deb'67e5'61df'5bd9},
       {0x8603'7ffe'79f7'73f0, 0x9e1b'9815'920f'8c09},
       {0xb633'b02e'aa27'a420, 0xce4b'c845'c23f'bc39},
       {0xe663'e05e'da57'd450, 0xfe7b'f875'f26f'ec69}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVaddvx,
      ExecMaskedVaddvx,
      {{170, 43, 172, 45, 174, 47, 176, 49, 178, 51, 180, 53, 182, 55, 184, 57},
       {186, 59, 188, 61, 190, 63, 192, 65, 194, 67, 196, 69, 198, 71, 200, 73},
       {202, 75, 204, 77, 206, 79, 208, 81, 210, 83, 212, 85, 214, 87, 216, 89},
       {218, 91, 220, 93, 222, 95, 224, 97, 226, 99, 228, 101, 230, 103, 232, 105},
       {234, 107, 236, 109, 238, 111, 240, 113, 242, 115, 244, 117, 246, 119, 248, 121},
       {250, 123, 252, 125, 254, 127, 0, 129, 2, 131, 4, 133, 6, 135, 8, 137},
       {10, 139, 12, 141, 14, 143, 16, 145, 18, 147, 20, 149, 22, 151, 24, 153},
       {26, 155, 28, 157, 30, 159, 32, 161, 34, 163, 36, 165, 38, 167, 40, 169}},
      {{0x2baa, 0x2dac, 0x2fae, 0x31b0, 0x33b2, 0x35b4, 0x37b6, 0x39b8},
       {0x3bba, 0x3dbc, 0x3fbe, 0x41c0, 0x43c2, 0x45c4, 0x47c6, 0x49c8},
       {0x4bca, 0x4dcc, 0x4fce, 0x51d0, 0x53d2, 0x55d4, 0x57d6, 0x59d8},
       {0x5bda, 0x5ddc, 0x5fde, 0x61e0, 0x63e2, 0x65e4, 0x67e6, 0x69e8},
       {0x6bea, 0x6dec, 0x6fee, 0x71f0, 0x73f2, 0x75f4, 0x77f6, 0x79f8},
       {0x7bfa, 0x7dfc, 0x7ffe, 0x8200, 0x8402, 0x8604, 0x8806, 0x8a08},
       {0x8c0a, 0x8e0c, 0x900e, 0x9210, 0x9412, 0x9614, 0x9816, 0x9a18},
       {0x9c1a, 0x9e1c, 0xa01e, 0xa220, 0xa422, 0xa624, 0xa826, 0xaa28}},
      {{0x2dad'2baa, 0x31b1'2fae, 0x35b5'33b2, 0x39b9'37b6},
       {0x3dbd'3bba, 0x41c1'3fbe, 0x45c5'43c2, 0x49c9'47c6},
       {0x4dcd'4bca, 0x51d1'4fce, 0x55d5'53d2, 0x59d9'57d6},
       {0x5ddd'5bda, 0x61e1'5fde, 0x65e5'63e2, 0x69e9'67e6},
       {0x6ded'6bea, 0x71f1'6fee, 0x75f5'73f2, 0x79f9'77f6},
       {0x7dfd'7bfa, 0x8201'7ffe, 0x8605'8402, 0x8a09'8806},
       {0x8e0d'8c0a, 0x9211'900e, 0x9615'9412, 0x9a19'9816},
       {0x9e1d'9c1a, 0xa221'a01e, 0xa625'a422, 0xaa29'a826}},
      {{0x31b1'2faf'2dad'2baa, 0x39b9'37b7'35b5'33b2},
       {0x41c1'3fbf'3dbd'3bba, 0x49c9'47c7'45c5'43c2},
       {0x51d1'4fcf'4dcd'4bca, 0x59d9'57d7'55d5'53d2},
       {0x61e1'5fdf'5ddd'5bda, 0x69e9'67e7'65e5'63e2},
       {0x71f1'6fef'6ded'6bea, 0x79f9'77f7'75f5'73f2},
       {0x8201'7fff'7dfd'7bfa, 0x8a09'8807'8605'8402},
       {0x9211'900f'8e0d'8c0a, 0x9a19'9817'9615'9412},
       {0xa221'a01f'9e1d'9c1a, 0xaa29'a827'a625'a422}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVaddvi,
      ExecMaskedVaddvi,
      {{245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255, 128, 1, 130, 3, 132},
       {5, 134, 7, 136, 9, 138, 11, 140, 13, 142, 15, 144, 17, 146, 19, 148},
       {21, 150, 23, 152, 25, 154, 27, 156, 29, 158, 31, 160, 33, 162, 35, 164},
       {37, 166, 39, 168, 41, 170, 43, 172, 45, 174, 47, 176, 49, 178, 51, 180},
       {53, 182, 55, 184, 57, 186, 59, 188, 61, 190, 63, 192, 65, 194, 67, 196},
       {69, 198, 71, 200, 73, 202, 75, 204, 77, 206, 79, 208, 81, 210, 83, 212},
       {85, 214, 87, 216, 89, 218, 91, 220, 93, 222, 95, 224, 97, 226, 99, 228},
       {101, 230, 103, 232, 105, 234, 107, 236, 109, 238, 111, 240, 113, 242, 115, 244}},
      {{0x80f5, 0x82f7, 0x84f9, 0x86fb, 0x88fd, 0x8aff, 0x8d01, 0x8f03},
       {0x9105, 0x9307, 0x9509, 0x970b, 0x990d, 0x9b0f, 0x9d11, 0x9f13},
       {0xa115, 0xa317, 0xa519, 0xa71b, 0xa91d, 0xab1f, 0xad21, 0xaf23},
       {0xb125, 0xb327, 0xb529, 0xb72b, 0xb92d, 0xbb2f, 0xbd31, 0xbf33},
       {0xc135, 0xc337, 0xc539, 0xc73b, 0xc93d, 0xcb3f, 0xcd41, 0xcf43},
       {0xd145, 0xd347, 0xd549, 0xd74b, 0xd94d, 0xdb4f, 0xdd51, 0xdf53},
       {0xe155, 0xe357, 0xe559, 0xe75b, 0xe95d, 0xeb5f, 0xed61, 0xef63},
       {0xf165, 0xf367, 0xf569, 0xf76b, 0xf96d, 0xfb6f, 0xfd71, 0xff73}},
      {{0x8302'80f5, 0x8706'84f9, 0x8b0a'88fd, 0x8f0e'8d01},
       {0x9312'9105, 0x9716'9509, 0x9b1a'990d, 0x9f1e'9d11},
       {0xa322'a115, 0xa726'a519, 0xab2a'a91d, 0xaf2e'ad21},
       {0xb332'b125, 0xb736'b529, 0xbb3a'b92d, 0xbf3e'bd31},
       {0xc342'c135, 0xc746'c539, 0xcb4a'c93d, 0xcf4e'cd41},
       {0xd352'd145, 0xd756'd549, 0xdb5a'd94d, 0xdf5e'dd51},
       {0xe362'e155, 0xe766'e559, 0xeb6a'e95d, 0xef6e'ed61},
       {0xf372'f165, 0xf776'f569, 0xfb7a'f96d, 0xff7e'fd71}},
      {{0x8706'8504'8302'80f5, 0x8f0e'8d0c'8b0a'88fd},
       {0x9716'9514'9312'9105, 0x9f1e'9d1c'9b1a'990d},
       {0xa726'a524'a322'a115, 0xaf2e'ad2c'ab2a'a91d},
       {0xb736'b534'b332'b125, 0xbf3e'bd3c'bb3a'b92d},
       {0xc746'c544'c342'c135, 0xcf4e'cd4c'cb4a'c93d},
       {0xd756'd554'd352'd145, 0xdf5e'dd5c'db5a'd94d},
       {0xe766'e564'e362'e155, 0xef6e'ed6c'eb6a'e95d},
       {0xf776'f574'f372'f165, 0xff7e'fd7c'fb7a'f96d}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVsadduvv,
      ExecMaskedVsadduvv,
      {{0, 255, 6, 255, 13, 255, 18, 255, 25, 255, 30, 255, 36, 255, 42, 255},
       {48, 255, 54, 255, 61, 255, 66, 255, 73, 255, 78, 255, 84, 255, 90, 255},
       {96, 255, 102, 255, 109, 255, 114, 255, 121, 255, 126, 255, 132, 255, 138, 255},
       {144, 255, 150, 255, 157, 255, 162, 255, 169, 255, 174, 255, 180, 255, 186, 255},
       {192, 211, 198, 217, 205, 223, 210, 229, 217, 203, 222, 209, 228, 215, 234, 221},
       {240, 255, 246, 255, 253, 255, 255, 255, 255, 251, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xd3c0, 0xd9c6, 0xdfcd, 0xe5d2, 0xcbd9, 0xd1de, 0xd7e4, 0xddea},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xfc09, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xd9c6'd3c0, 0xe5d2'dfcd, 0xd1de'cbd9, 0xddea'd7e4},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xe5d2'dfcd'd9c6'd3c0, 0xddea'd7e4'd1de'cbd9},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsadduvx,
      ExecMaskedVsadduvx,
      {{170, 255, 172, 255, 174, 255, 176, 255, 178, 255, 180, 255, 182, 255, 184, 255},
       {186, 255, 188, 255, 190, 255, 192, 255, 194, 255, 196, 255, 198, 255, 200, 255},
       {202, 255, 204, 255, 206, 255, 208, 255, 210, 255, 212, 255, 214, 255, 216, 255},
       {218, 255, 220, 255, 222, 255, 224, 255, 226, 255, 228, 255, 230, 255, 232, 255},
       {234, 255, 236, 255, 238, 255, 240, 255, 242, 255, 244, 255, 246, 255, 248, 255},
       {250, 255, 252, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsadduvi,
      ExecMaskedVsadduvi,
      {{245, 255, 247, 255, 249, 255, 251, 255, 253, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsaddvv,
      ExecMaskedVsaddvv,
      {{0, 128, 6, 128, 13, 128, 18, 128, 25, 128, 30, 128, 36, 128, 42, 128},
       {48, 128, 54, 128, 61, 128, 66, 128, 73, 128, 78, 128, 84, 128, 90, 128},
       {96, 128, 102, 128, 109, 128, 114, 133, 121, 128, 126, 128, 127, 128, 127, 128},
       {127, 163, 127, 169, 127, 175, 127, 181, 127, 155, 127, 161, 127, 167, 127, 173},
       {192, 211, 198, 217, 205, 223, 210, 229, 217, 203, 222, 209, 228, 215, 234, 221},
       {240, 3, 246, 9, 253, 15, 2, 21, 9, 251, 14, 1, 20, 7, 26, 13},
       {32, 51, 38, 57, 45, 63, 50, 69, 57, 43, 62, 49, 68, 55, 74, 61},
       {80, 99, 86, 105, 93, 111, 98, 117, 105, 91, 110, 97, 116, 103, 122, 109}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8572, 0x8000, 0x8000, 0x8000, 0x8000},
       {0xa390, 0xa996, 0xaf9d, 0xb5a2, 0x9ba9, 0xa1ae, 0xa7b4, 0xadba},
       {0xd3c0, 0xd9c6, 0xdfcd, 0xe5d2, 0xcbd9, 0xd1de, 0xd7e4, 0xddea},
       {0x03f0, 0x09f6, 0x0ffd, 0x1602, 0xfc09, 0x020e, 0x0814, 0x0e1a},
       {0x3420, 0x3a26, 0x402d, 0x4632, 0x2c39, 0x323e, 0x3844, 0x3e4a},
       {0x6450, 0x6a56, 0x705d, 0x7662, 0x5c69, 0x626e, 0x6874, 0x6e7a}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8573'7f6d, 0x8000'0000, 0x8000'0000},
       {0xa997'a390, 0xb5a3'af9d, 0xa1af'9ba9, 0xadbb'a7b4},
       {0xd9c6'd3c0, 0xe5d2'dfcd, 0xd1de'cbd9, 0xddea'd7e4},
       {0x09f7'03f0, 0x1603'0ffd, 0x020e'fc09, 0x0e1b'0814},
       {0x3a27'3420, 0x4633'402d, 0x323f'2c39, 0x3e4b'3844},
       {0x6a57'6450, 0x7663'705d, 0x626f'5c69, 0x6e7b'6874}},
      {{0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8573'7f6e'7967'7360, 0x8000'0000'0000'0000},
       {0xb5a3'af9e'a997'a390, 0xadbb'a7b5'a1af'9ba9},
       {0xe5d2'dfcd'd9c6'd3c0, 0xddea'd7e4'd1de'cbd9},
       {0x1603'0ffe'09f7'03f0, 0x0e1b'0815'020e'fc09},
       {0x4633'402e'3a27'3420, 0x3e4b'3845'323f'2c39},
       {0x7663'705e'6a57'6450, 0x6e7b'6875'626f'5c69}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsaddvx,
      ExecMaskedVsaddvx,
      {{170, 128, 172, 128, 174, 128, 176, 128, 178, 128, 180, 128, 182, 128, 184, 128},
       {186, 128, 188, 128, 190, 128, 192, 128, 194, 128, 196, 128, 198, 128, 200, 128},
       {202, 128, 204, 128, 206, 128, 208, 128, 210, 128, 212, 128, 214, 128, 216, 128},
       {218, 128, 220, 128, 222, 128, 224, 128, 226, 128, 228, 128, 230, 128, 232, 128},
       {234, 128, 236, 128, 238, 128, 240, 128, 242, 128, 244, 128, 246, 128, 248, 128},
       {250, 128, 252, 128, 254, 128, 0, 129, 2, 131, 4, 133, 6, 135, 8, 137},
       {10, 139, 12, 141, 14, 143, 16, 145, 18, 147, 20, 149, 22, 151, 24, 153},
       {26, 155, 28, 157, 30, 159, 32, 161, 34, 163, 36, 165, 38, 167, 40, 169}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8200, 0x8402, 0x8604, 0x8806, 0x8a08},
       {0x8c0a, 0x8e0c, 0x900e, 0x9210, 0x9412, 0x9614, 0x9816, 0x9a18},
       {0x9c1a, 0x9e1c, 0xa01e, 0xa220, 0xa422, 0xa624, 0xa826, 0xaa28}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8201'7ffe, 0x8605'8402, 0x8a09'8806},
       {0x8e0d'8c0a, 0x9211'900e, 0x9615'9412, 0x9a19'9816},
       {0x9e1d'9c1a, 0xa221'a01e, 0xa625'a422, 0xaa29'a826}},
      {{0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8201'7fff'7dfd'7bfa, 0x8a09'8807'8605'8402},
       {0x9211'900f'8e0d'8c0a, 0x9a19'9817'9615'9412},
       {0xa221'a01f'9e1d'9c1a, 0xaa29'a827'a625'a422}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsaddvi,
      ExecMaskedVsaddvi,
      {{245, 128, 247, 128, 249, 128, 251, 128, 253, 128, 255, 128, 1, 130, 3, 132},
       {5, 134, 7, 136, 9, 138, 11, 140, 13, 142, 15, 144, 17, 146, 19, 148},
       {21, 150, 23, 152, 25, 154, 27, 156, 29, 158, 31, 160, 33, 162, 35, 164},
       {37, 166, 39, 168, 41, 170, 43, 172, 45, 174, 47, 176, 49, 178, 51, 180},
       {53, 182, 55, 184, 57, 186, 59, 188, 61, 190, 63, 192, 65, 194, 67, 196},
       {69, 198, 71, 200, 73, 202, 75, 204, 77, 206, 79, 208, 81, 210, 83, 212},
       {85, 214, 87, 216, 89, 218, 91, 220, 93, 222, 95, 224, 97, 226, 99, 228},
       {101, 230, 103, 232, 105, 234, 107, 236, 109, 238, 111, 240, 113, 242, 115, 244}},
      {{0x80f5, 0x82f7, 0x84f9, 0x86fb, 0x88fd, 0x8aff, 0x8d01, 0x8f03},
       {0x9105, 0x9307, 0x9509, 0x970b, 0x990d, 0x9b0f, 0x9d11, 0x9f13},
       {0xa115, 0xa317, 0xa519, 0xa71b, 0xa91d, 0xab1f, 0xad21, 0xaf23},
       {0xb125, 0xb327, 0xb529, 0xb72b, 0xb92d, 0xbb2f, 0xbd31, 0xbf33},
       {0xc135, 0xc337, 0xc539, 0xc73b, 0xc93d, 0xcb3f, 0xcd41, 0xcf43},
       {0xd145, 0xd347, 0xd549, 0xd74b, 0xd94d, 0xdb4f, 0xdd51, 0xdf53},
       {0xe155, 0xe357, 0xe559, 0xe75b, 0xe95d, 0xeb5f, 0xed61, 0xef63},
       {0xf165, 0xf367, 0xf569, 0xf76b, 0xf96d, 0xfb6f, 0xfd71, 0xff73}},
      {{0x8302'80f5, 0x8706'84f9, 0x8b0a'88fd, 0x8f0e'8d01},
       {0x9312'9105, 0x9716'9509, 0x9b1a'990d, 0x9f1e'9d11},
       {0xa322'a115, 0xa726'a519, 0xab2a'a91d, 0xaf2e'ad21},
       {0xb332'b125, 0xb736'b529, 0xbb3a'b92d, 0xbf3e'bd31},
       {0xc342'c135, 0xc746'c539, 0xcb4a'c93d, 0xcf4e'cd41},
       {0xd352'd145, 0xd756'd549, 0xdb5a'd94d, 0xdf5e'dd51},
       {0xe362'e155, 0xe766'e559, 0xeb6a'e95d, 0xef6e'ed61},
       {0xf372'f165, 0xf776'f569, 0xfb7a'f96d, 0xff7e'fd71}},
      {{0x8706'8504'8302'80f5, 0x8f0e'8d0c'8b0a'88fd},
       {0x9716'9514'9312'9105, 0x9f1e'9d1c'9b1a'990d},
       {0xa726'a524'a322'a115, 0xaf2e'ad2c'ab2a'a91d},
       {0xb736'b534'b332'b125, 0xbf3e'bd3c'bb3a'b92d},
       {0xc746'c544'c342'c135, 0xcf4e'cd4c'cb4a'c93d},
       {0xd756'd554'd352'd145, 0xdf5e'dd5c'db5a'd94d},
       {0xe766'e564'e362'e155, 0xef6e'ed6c'eb6a'e95d},
       {0xf776'f574'f372'f165, 0xff7e'fd7c'fb7a'f96d}},
      kVectorCalculationsSource);

  TestVectorFloatInstruction(ExecVfaddvv,
                             ExecMaskedVfaddvv,
                             {{0x9604'9200, 0x9e0c'9a09, 0x8b0a'ae29, 0x8f35'af92},
                              {0xb624'b220, 0xbe2c'ba29, 0xa634'a233, 0xae3c'aa38},
                              {0xd644'd240, 0xde4c'da49, 0xc654'c251, 0xce5c'ca58},
                              {0xf664'f260, 0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e4fe, 0xeb6a'e968, 0xef6e'ed6c},
                              {0x76e2'8cfd, 0x7eec'78fb, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x9e0c'9a09'9604'9200, 0x8f0e'8d45'9f3b'9531},
                              {0xbe2c'ba29'b624'b220, 0xae3c'aa38'a634'a231},
                              {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
                              {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0x7eec'7ae9'76e4'72e0, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfaddvf,
                             ExecMaskedVfaddvf,
                             {{0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b3'ffe9, 0x40b3'e8a9, 0x409c'2858},
                              {0xc33d'2140, 0xc746'bfa4, 0xcb4a'c942, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'7f85'0b0d'1315},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwaddvv,
                                     ExecMaskedVfwaddvv,
                                     {{0xbac0'9240'0000'4140, 0xbbc1'9341'2000'0043},
                                      {0xb961'55c5'1088'0000, 0xb9e6'b5f2'4000'0000},
                                      {0xbec4'9644'0000'0000, 0xbfc5'9745'2000'0000},
                                      {0xbcc6'9446'6d4c'8c00, 0xbdc7'9547'004f'4e8e},
                                      {0xc2c8'9a48'0000'0000, 0xc3c9'9b49'2000'0000},
                                      {0xc0ca'984a'2000'0000, 0xc1cb'994b'0000'0000},
                                      {0xc6cc'9e4c'0000'0000, 0xc7cd'9f4d'2000'0000},
                                      {0xc4ce'9c4e'2000'0000, 0xc5cf'9d4f'0000'0000}},
                                     kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwaddwv,
                                     ExecMaskedVfwaddwv,
                                     {{0xbac0'9240'0000'0000, 0xbbc1'9341'2000'0000},
                                      {0xb8c2'9042'2000'0000, 0xb9c3'9143'0000'0000},
                                      {0xbec4'9644'0000'0000, 0xbfc5'9745'2000'0000},
                                      {0xbcc6'9446'2000'0000, 0xbf3e'bd3c'ea65'4738},
                                      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                                      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                                      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                                      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                                     kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwaddwf,
                                     ExecMaskedVfwaddwf,
                                     {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                      {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                      {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                      {0x4016'8000'0000'0000, 0x4016'7f85'0b0d'1315},
                                      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                                      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                                      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                                      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                                     kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVsubvv() {
  asm("vsub.vv v8, v16, v24\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVsubvv() {
  asm("vsub.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVsubvx() {
  asm("vsub.vx v8, v16, t0\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVsubvx() {
  asm("vsub.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVssubuvv() {
  asm("vssubu.vv v8, v16, v24\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVssubuvv() {
  asm("vssubu.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVssubuvx() {
  asm("vssubu.vx v8, v16, t0\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVssubuvx() {
  asm("vssubu.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVssubvv() {
  asm("vssub.vv v8, v16, v24\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVssubvv() {
  asm("vssub.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVssubvx() {
  asm("vssub.vx v8, v16, t0\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVssubvx() {
  asm("vssub.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfsubvv() {
  asm("vfsub.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfsubvv() {
  asm("vfsub.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfsubvf() {
  asm("vfsub.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfsubvf() {
  asm("vfsub.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwsubvv() {
  asm("vfwsub.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwsubvv() {
  asm("vfwsub.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwsubvf() {
  asm("vfwsub.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwsubvf() {
  asm("vfwsub.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwsubwv() {
  asm("vfwsub.wv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwsubwv() {
  asm("vfwsub.wv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwsubwf() {
  asm("vfwsub.wf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwsubwf() {
  asm("vfwsub.wf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVsub) {
  TestVectorInstruction(
      ExecVsubvv,
      ExecMaskedVsubvv,
      {{0, 127, 254, 125, 251, 123, 250, 121, 247, 119, 246, 117, 244, 115, 242, 113},
       {240, 111, 238, 109, 235, 107, 234, 105, 231, 103, 230, 101, 228, 99, 226, 97},
       {224, 95, 222, 93, 219, 91, 218, 89, 215, 87, 214, 85, 212, 83, 210, 81},
       {208, 79, 206, 77, 203, 75, 202, 73, 199, 71, 198, 69, 196, 67, 194, 65},
       {192, 63, 190, 61, 187, 59, 186, 57, 183, 55, 182, 53, 180, 51, 178, 49},
       {176, 47, 174, 45, 171, 43, 170, 41, 167, 39, 166, 37, 164, 35, 162, 33},
       {160, 31, 158, 29, 155, 27, 154, 25, 151, 23, 150, 21, 148, 19, 146, 17},
       {144, 15, 142, 13, 139, 11, 138, 9, 135, 7, 134, 5, 132, 3, 130, 1}},
      {{0x7f00, 0x7cfe, 0x7afb, 0x78fa, 0x76f7, 0x74f6, 0x72f4, 0x70f2},
       {0x6ef0, 0x6cee, 0x6aeb, 0x68ea, 0x66e7, 0x64e6, 0x62e4, 0x60e2},
       {0x5ee0, 0x5cde, 0x5adb, 0x58da, 0x56d7, 0x54d6, 0x52d4, 0x50d2},
       {0x4ed0, 0x4cce, 0x4acb, 0x48ca, 0x46c7, 0x44c6, 0x42c4, 0x40c2},
       {0x3ec0, 0x3cbe, 0x3abb, 0x38ba, 0x36b7, 0x34b6, 0x32b4, 0x30b2},
       {0x2eb0, 0x2cae, 0x2aab, 0x28aa, 0x26a7, 0x24a6, 0x22a4, 0x20a2},
       {0x1ea0, 0x1c9e, 0x1a9b, 0x189a, 0x1697, 0x1496, 0x1294, 0x1092},
       {0x0e90, 0x0c8e, 0x0a8b, 0x088a, 0x0687, 0x0486, 0x0284, 0x0082}},
      {{0x7cfe'7f00, 0x78fa'7afb, 0x74f6'76f7, 0x70f2'72f4},
       {0x6cee'6ef0, 0x68ea'6aeb, 0x64e6'66e7, 0x60e2'62e4},
       {0x5cde'5ee0, 0x58da'5adb, 0x54d6'56d7, 0x50d2'52d4},
       {0x4cce'4ed0, 0x48ca'4acb, 0x44c6'46c7, 0x40c2'42c4},
       {0x3cbe'3ec0, 0x38ba'3abb, 0x34b6'36b7, 0x30b2'32b4},
       {0x2cae'2eb0, 0x28aa'2aab, 0x24a6'26a7, 0x20a2'22a4},
       {0x1c9e'1ea0, 0x189a'1a9b, 0x1496'1697, 0x1092'1294},
       {0x0c8e'0e90, 0x088a'0a8b, 0x0486'0687, 0x0082'0284}},
      {{0x78fa'7afb'7cfe'7f00, 0x70f2'72f4'74f6'76f7},
       {0x68ea'6aeb'6cee'6ef0, 0x60e2'62e4'64e6'66e7},
       {0x58da'5adb'5cde'5ee0, 0x50d2'52d4'54d6'56d7},
       {0x48ca'4acb'4cce'4ed0, 0x40c2'42c4'44c6'46c7},
       {0x38ba'3abb'3cbe'3ec0, 0x30b2'32b4'34b6'36b7},
       {0x28aa'2aab'2cae'2eb0, 0x20a2'22a4'24a6'26a7},
       {0x189a'1a9b'1c9e'1ea0, 0x1092'1294'1496'1697},
       {0x088a'0a8b'0c8e'0e90, 0x0082'0284'0486'0687}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVsubvx,
      ExecMaskedVsubvx,
      {{86, 215, 88, 217, 90, 219, 92, 221, 94, 223, 96, 225, 98, 227, 100, 229},
       {102, 231, 104, 233, 106, 235, 108, 237, 110, 239, 112, 241, 114, 243, 116, 245},
       {118, 247, 120, 249, 122, 251, 124, 253, 126, 255, 128, 1, 130, 3, 132, 5},
       {134, 7, 136, 9, 138, 11, 140, 13, 142, 15, 144, 17, 146, 19, 148, 21},
       {150, 23, 152, 25, 154, 27, 156, 29, 158, 31, 160, 33, 162, 35, 164, 37},
       {166, 39, 168, 41, 170, 43, 172, 45, 174, 47, 176, 49, 178, 51, 180, 53},
       {182, 55, 184, 57, 186, 59, 188, 61, 190, 63, 192, 65, 194, 67, 196, 69},
       {198, 71, 200, 73, 202, 75, 204, 77, 206, 79, 208, 81, 210, 83, 212, 85}},
      {{0xd656, 0xd858, 0xda5a, 0xdc5c, 0xde5e, 0xe060, 0xe262, 0xe464},
       {0xe666, 0xe868, 0xea6a, 0xec6c, 0xee6e, 0xf070, 0xf272, 0xf474},
       {0xf676, 0xf878, 0xfa7a, 0xfc7c, 0xfe7e, 0x0080, 0x0282, 0x0484},
       {0x0686, 0x0888, 0x0a8a, 0x0c8c, 0x0e8e, 0x1090, 0x1292, 0x1494},
       {0x1696, 0x1898, 0x1a9a, 0x1c9c, 0x1e9e, 0x20a0, 0x22a2, 0x24a4},
       {0x26a6, 0x28a8, 0x2aaa, 0x2cac, 0x2eae, 0x30b0, 0x32b2, 0x34b4},
       {0x36b6, 0x38b8, 0x3aba, 0x3cbc, 0x3ebe, 0x40c0, 0x42c2, 0x44c4},
       {0x46c6, 0x48c8, 0x4aca, 0x4ccc, 0x4ece, 0x50d0, 0x52d2, 0x54d4}},
      {{0xd857'd656, 0xdc5b'da5a, 0xe05f'de5e, 0xe463'e262},
       {0xe867'e666, 0xec6b'ea6a, 0xf06f'ee6e, 0xf473'f272},
       {0xf877'f676, 0xfc7b'fa7a, 0x007f'fe7e, 0x0484'0282},
       {0x0888'0686, 0x0c8c'0a8a, 0x1090'0e8e, 0x1494'1292},
       {0x1898'1696, 0x1c9c'1a9a, 0x20a0'1e9e, 0x24a4'22a2},
       {0x28a8'26a6, 0x2cac'2aaa, 0x30b0'2eae, 0x34b4'32b2},
       {0x38b8'36b6, 0x3cbc'3aba, 0x40c0'3ebe, 0x44c4'42c2},
       {0x48c8'46c6, 0x4ccc'4aca, 0x50d0'4ece, 0x54d4'52d2}},
      {{0xdc5b'da59'd857'd656, 0xe463'e261'e05f'de5e},
       {0xec6b'ea69'e867'e666, 0xf473'f271'f06f'ee6e},
       {0xfc7b'fa79'f877'f676, 0x0484'0282'007f'fe7e},
       {0x0c8c'0a8a'0888'0686, 0x1494'1292'1090'0e8e},
       {0x1c9c'1a9a'1898'1696, 0x24a4'22a2'20a0'1e9e},
       {0x2cac'2aaa'28a8'26a6, 0x34b4'32b2'30b0'2eae},
       {0x3cbc'3aba'38b8'36b6, 0x44c4'42c2'40c0'3ebe},
       {0x4ccc'4aca'48c8'46c6, 0x54d4'52d2'50d0'4ece}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVssubuvv,
                        ExecMaskedVssubuvv,
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 5, 0, 3, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 175, 0, 173, 0, 171, 0, 169, 0, 199, 0, 197, 0, 195, 0, 193},
                         {0, 159, 0, 157, 0, 155, 0, 153, 0, 183, 0, 181, 0, 179, 0, 177},
                         {0, 143, 0, 141, 0, 139, 0, 137, 0, 167, 0, 165, 0, 163, 0, 161},
                         {0, 127, 0, 125, 0, 123, 0, 121, 0, 151, 0, 149, 0, 147, 0, 145}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x06f7, 0x04f6, 0x02f4, 0x00f2},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0xaec0, 0xacbe, 0xaabb, 0xa8ba, 0xc6b7, 0xc4b6, 0xc2b4, 0xc0b2},
                         {0x9eb0, 0x9cae, 0x9aab, 0x98aa, 0xb6a7, 0xb4a6, 0xb2a4, 0xb0a2},
                         {0x8ea0, 0x8c9e, 0x8a9b, 0x889a, 0xa697, 0xa496, 0xa294, 0xa092},
                         {0x7e90, 0x7c8e, 0x7a8b, 0x788a, 0x9687, 0x9486, 0x9284, 0x9082}},
                        {{0x0000'0000, 0x0000'0000, 0x04f6'06f7, 0x00f2'02f4},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0xacbe'aec0, 0xa8ba'aabb, 0xc4b6'c6b7, 0xc0b2'c2b4},
                         {0x9cae'9eb0, 0x98aa'9aab, 0xb4a6'b6a7, 0xb0a2'b2a4},
                         {0x8c9e'8ea0, 0x889a'8a9b, 0xa496'a697, 0xa092'a294},
                         {0x7c8e'7e90, 0x788a'7a8b, 0x9486'9687, 0x9082'9284}},
                        {{0x0000'0000'0000'0000, 0x00f2'02f4'04f6'06f7},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0xa8ba'aabb'acbe'aec0, 0xc0b2'c2b4'c4b6'c6b7},
                         {0x98aa'9aab'9cae'9eb0, 0xb0a2'b2a4'b4a6'b6a7},
                         {0x889a'8a9b'8c9e'8ea0, 0xa092'a294'a496'a697},
                         {0x788a'7a8b'7c8e'7e90, 0x9082'9284'9486'9687}},
                        kVectorCalculationsSource);
  TestVectorInstruction(ExecVssubuvx,
                        ExecMaskedVssubuvx,
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 5},
                         {0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21},
                         {0, 23, 0, 25, 0, 27, 0, 29, 0, 31, 0, 33, 0, 35, 0, 37},
                         {0, 39, 0, 41, 0, 43, 0, 45, 0, 47, 0, 49, 0, 51, 0, 53},
                         {0, 55, 0, 57, 0, 59, 0, 61, 0, 63, 0, 65, 0, 67, 0, 69},
                         {0, 71, 0, 73, 0, 75, 0, 77, 0, 79, 0, 81, 0, 83, 0, 85}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0080, 0x0282, 0x0484},
                         {0x0686, 0x0888, 0x0a8a, 0x0c8c, 0x0e8e, 0x1090, 0x1292, 0x1494},
                         {0x1696, 0x1898, 0x1a9a, 0x1c9c, 0x1e9e, 0x20a0, 0x22a2, 0x24a4},
                         {0x26a6, 0x28a8, 0x2aaa, 0x2cac, 0x2eae, 0x30b0, 0x32b2, 0x34b4},
                         {0x36b6, 0x38b8, 0x3aba, 0x3cbc, 0x3ebe, 0x40c0, 0x42c2, 0x44c4},
                         {0x46c6, 0x48c8, 0x4aca, 0x4ccc, 0x4ece, 0x50d0, 0x52d2, 0x54d4}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x007f'fe7e, 0x0484'0282},
                         {0x0888'0686, 0x0c8c'0a8a, 0x1090'0e8e, 0x1494'1292},
                         {0x1898'1696, 0x1c9c'1a9a, 0x20a0'1e9e, 0x24a4'22a2},
                         {0x28a8'26a6, 0x2cac'2aaa, 0x30b0'2eae, 0x34b4'32b2},
                         {0x38b8'36b6, 0x3cbc'3aba, 0x40c0'3ebe, 0x44c4'42c2},
                         {0x48c8'46c6, 0x4ccc'4aca, 0x50d0'4ece, 0x54d4'52d2}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0484'0282'007f'fe7e},
                         {0x0c8c'0a8a'0888'0686, 0x1494'1292'1090'0e8e},
                         {0x1c9c'1a9a'1898'1696, 0x24a4'22a2'20a0'1e9e},
                         {0x2cac'2aaa'28a8'26a6, 0x34b4'32b2'30b0'2eae},
                         {0x3cbc'3aba'38b8'36b6, 0x44c4'42c2'40c0'3ebe},
                         {0x4ccc'4aca'48c8'46c6, 0x54d4'52d2'50d0'4ece}},
                        kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVssubvv,
      ExecMaskedVssubvv,
      {{0, 239, 254, 237, 251, 235, 250, 233, 247, 7, 246, 5, 244, 3, 242, 1},
       {240, 223, 238, 221, 235, 219, 234, 217, 231, 247, 230, 245, 228, 243, 226, 241},
       {224, 207, 222, 205, 219, 203, 218, 201, 215, 231, 214, 229, 212, 227, 210, 225},
       {208, 191, 206, 189, 203, 187, 202, 185, 199, 215, 198, 213, 196, 211, 194, 209},
       {127, 175, 127, 173, 127, 171, 127, 169, 127, 199, 127, 197, 127, 195, 127, 193},
       {127, 159, 127, 157, 127, 155, 127, 153, 127, 183, 127, 181, 127, 179, 127, 177},
       {127, 143, 127, 141, 127, 139, 127, 137, 127, 167, 127, 165, 127, 163, 127, 161},
       {127, 128, 127, 128, 127, 128, 127, 128, 127, 151, 127, 149, 127, 147, 127, 145}},
      {{0xef00, 0xecfe, 0xeafb, 0xe8fa, 0x06f7, 0x04f6, 0x02f4, 0x00f2},
       {0xdef0, 0xdcee, 0xdaeb, 0xd8ea, 0xf6e7, 0xf4e6, 0xf2e4, 0xf0e2},
       {0xcee0, 0xccde, 0xcadb, 0xc8da, 0xe6d7, 0xe4d6, 0xe2d4, 0xe0d2},
       {0xbed0, 0xbcce, 0xbacb, 0xb8ca, 0xd6c7, 0xd4c6, 0xd2c4, 0xd0c2},
       {0xaec0, 0xacbe, 0xaabb, 0xa8ba, 0xc6b7, 0xc4b6, 0xc2b4, 0xc0b2},
       {0x9eb0, 0x9cae, 0x9aab, 0x98aa, 0xb6a7, 0xb4a6, 0xb2a4, 0xb0a2},
       {0x8ea0, 0x8c9e, 0x8a9b, 0x889a, 0xa697, 0xa496, 0xa294, 0xa092},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x9687, 0x9486, 0x9284, 0x9082}},
      {{0xecfd'ef00, 0xe8f9'eafb, 0x04f6'06f7, 0x00f2'02f4},
       {0xdced'def0, 0xd8e9'daeb, 0xf4e5'f6e7, 0xf0e1'f2e4},
       {0xccdd'cee0, 0xc8d9'cadb, 0xe4d5'e6d7, 0xe0d1'e2d4},
       {0xbccd'bed0, 0xb8c9'bacb, 0xd4c5'd6c7, 0xd0c1'd2c4},
       {0xacbe'aec0, 0xa8ba'aabb, 0xc4b6'c6b7, 0xc0b2'c2b4},
       {0x9cae'9eb0, 0x98aa'9aab, 0xb4a6'b6a7, 0xb0a2'b2a4},
       {0x8c9e'8ea0, 0x889a'8a9b, 0xa496'a697, 0xa092'a294},
       {0x8000'0000, 0x8000'0000, 0x9486'9687, 0x9082'9284}},
      {{0xe8f9'eafa'ecfd'ef00, 0x00f2'02f4'04f6'06f7},
       {0xd8e9'daea'dced'def0, 0xf0e1'f2e3'f4e5'f6e7},
       {0xc8d9'cada'ccdd'cee0, 0xe0d1'e2d3'e4d5'e6d7},
       {0xb8c9'baca'bccd'bed0, 0xd0c1'd2c3'd4c5'd6c7},
       {0xa8ba'aabb'acbe'aec0, 0xc0b2'c2b4'c4b6'c6b7},
       {0x98aa'9aab'9cae'9eb0, 0xb0a2'b2a4'b4a6'b6a7},
       {0x889a'8a9b'8c9e'8ea0, 0xa092'a294'a496'a697},
       {0x8000'0000'0000'0000, 0x9082'9284'9486'9687}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVssubvx,
      ExecMaskedVssubvx,
      {{86, 215, 88, 217, 90, 219, 92, 221, 94, 223, 96, 225, 98, 227, 100, 229},
       {102, 231, 104, 233, 106, 235, 108, 237, 110, 239, 112, 241, 114, 243, 116, 245},
       {118, 247, 120, 249, 122, 251, 124, 253, 126, 255, 127, 1, 127, 3, 127, 5},
       {127, 7, 127, 9, 127, 11, 127, 13, 127, 15, 127, 17, 127, 19, 127, 21},
       {127, 23, 127, 25, 127, 27, 127, 29, 127, 31, 127, 33, 127, 35, 127, 37},
       {127, 39, 127, 41, 127, 43, 127, 45, 127, 47, 127, 49, 127, 51, 127, 53},
       {127, 55, 127, 57, 127, 59, 127, 61, 127, 63, 127, 65, 127, 67, 127, 69},
       {127, 71, 127, 73, 127, 75, 127, 77, 127, 79, 127, 81, 127, 83, 127, 85}},
      {{0xd656, 0xd858, 0xda5a, 0xdc5c, 0xde5e, 0xe060, 0xe262, 0xe464},
       {0xe666, 0xe868, 0xea6a, 0xec6c, 0xee6e, 0xf070, 0xf272, 0xf474},
       {0xf676, 0xf878, 0xfa7a, 0xfc7c, 0xfe7e, 0x0080, 0x0282, 0x0484},
       {0x0686, 0x0888, 0x0a8a, 0x0c8c, 0x0e8e, 0x1090, 0x1292, 0x1494},
       {0x1696, 0x1898, 0x1a9a, 0x1c9c, 0x1e9e, 0x20a0, 0x22a2, 0x24a4},
       {0x26a6, 0x28a8, 0x2aaa, 0x2cac, 0x2eae, 0x30b0, 0x32b2, 0x34b4},
       {0x36b6, 0x38b8, 0x3aba, 0x3cbc, 0x3ebe, 0x40c0, 0x42c2, 0x44c4},
       {0x46c6, 0x48c8, 0x4aca, 0x4ccc, 0x4ece, 0x50d0, 0x52d2, 0x54d4}},
      {{0xd857'd656, 0xdc5b'da5a, 0xe05f'de5e, 0xe463'e262},
       {0xe867'e666, 0xec6b'ea6a, 0xf06f'ee6e, 0xf473'f272},
       {0xf877'f676, 0xfc7b'fa7a, 0x007f'fe7e, 0x0484'0282},
       {0x0888'0686, 0x0c8c'0a8a, 0x1090'0e8e, 0x1494'1292},
       {0x1898'1696, 0x1c9c'1a9a, 0x20a0'1e9e, 0x24a4'22a2},
       {0x28a8'26a6, 0x2cac'2aaa, 0x30b0'2eae, 0x34b4'32b2},
       {0x38b8'36b6, 0x3cbc'3aba, 0x40c0'3ebe, 0x44c4'42c2},
       {0x48c8'46c6, 0x4ccc'4aca, 0x50d0'4ece, 0x54d4'52d2}},
      {{0xdc5b'da59'd857'd656, 0xe463'e261'e05f'de5e},
       {0xec6b'ea69'e867'e666, 0xf473'f271'f06f'ee6e},
       {0xfc7b'fa79'f877'f676, 0x0484'0282'007f'fe7e},
       {0x0c8c'0a8a'0888'0686, 0x1494'1292'1090'0e8e},
       {0x1c9c'1a9a'1898'1696, 0x24a4'22a2'20a0'1e9e},
       {0x2cac'2aaa'28a8'26a6, 0x34b4'32b2'30b0'2eae},
       {0x3cbc'3aba'38b8'36b6, 0x44c4'42c2'40c0'3ebe},
       {0x4ccc'4aca'48c8'46c6, 0x54d4'52d2'50d0'4ece}},
      kVectorCalculationsSource);

  TestVectorFloatInstruction(ExecVfsubvv,
                             ExecMaskedVfsubvv,
                             {{0x1604'9200, 0x1e0c'9a09, 0x8b0a'63e7, 0x8ece'd50c},
                              {0x3624'b220, 0x3e2c'ba29, 0x2634'a22f, 0x2e3c'aa38},
                              {0x5644'd240, 0x5e4c'da49, 0x4654'c251, 0x4e5c'ca58},
                              {0x7664'f260, 0x7e6c'fa69, 0x6674'e271, 0x6e7c'ea78},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e5ca, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf6e6'58c3, 0xfeec'7cd7, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x1e0c'9a09'9604'9200, 0x8f0e'8cd3'76d9'7cdf},
                              {0x3e2c'ba29'b624'b220, 0x2e3c'aa38'a634'a231},
                              {0x5e4c'da49'd644'd240, 0x4e5c'ca58'c654'c251},
                              {0x7e6c'fa69'f664'f260, 0x6e7c'ea78'e674'e271},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xfeec'7ae9'76e4'72e0, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfsubvf,
                             ExecMaskedVfsubvf,
                             {{0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000},
                              {0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000},
                              {0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000},
                              {0xc0b4'0000, 0xc0b4'0017, 0xc0b4'1757, 0xc0cb'd7a8},
                              {0xc348'6140, 0xc746'cae4, 0xcb4a'c94e, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                              {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                              {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                              {0xc016'8000'0000'0000, 0xc016'807a'f4f2'eceb},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwsubvv,
                                     ExecMaskedVfwsubvv,
                                     {{0x3ac0'923f'ffff'bec0, 0x3bc1'9341'1fff'ffbd},
                                      {0xb961'4c7c'ef78'0000, 0xb9d9'daa1'8000'0000},
                                      {0x3ec4'9644'0000'0000, 0x3fc5'9745'2000'0000},
                                      {0x3cc6'9445'd2b3'7400, 0x3dc7'9546'ffb0'b172},
                                      {0x42c8'9a48'0000'0000, 0x43c9'9b49'2000'0000},
                                      {0x40ca'984a'2000'0000, 0x41cb'994b'0000'0000},
                                      {0x46cc'9e4c'0000'0000, 0x47cd'9f4d'2000'0000},
                                      {0x44ce'9c4e'2000'0000, 0x45cf'9d4f'0000'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwsubvf,
                                     ExecMaskedVfwsubvf,
                                     {{0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0003},
                                      {0xc016'8000'0000'02ab, 0xc016'8000'0002'bab5},
                                      {0xc016'8000'02ca'c4c0, 0xc016'8002'dad4'd000},
                                      {0xc016'82ea'e4e0'0000, 0xc019'7af4'f000'0000}},
                                     kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwsubwv,
                                     ExecMaskedVfwsubwv,
                                     {{0x3ac0'9240'0000'0000, 0x3bc1'9341'2000'0000},
                                      {0x38c2'9042'2000'0000, 0x39c3'9143'0000'0000},
                                      {0x3ec4'9644'0000'0000, 0x3fc5'9745'2000'0000},
                                      {0x3cc6'9446'2000'0000, 0xbf3e'bd3c'8c10'2b38},
                                      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                                      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                                      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                                      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwsubwf,
                                     ExecMaskedVfwsubwf,
                                     {{0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'807a'f4f2'eceb},
                                      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                                      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                                      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                                      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                                     kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVandvv() {
  asm("vand.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVandvv() {
  asm("vand.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVandvx() {
  asm("vand.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVandvx() {
  asm("vand.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVandvi() {
  asm("vand.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVandvi() {
  asm("vand.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVand) {
  TestVectorInstruction(
      ExecVandvv,
      ExecMaskedVandvv,
      {{0, 0, 0, 2, 0, 0, 4, 6, 0, 0, 0, 2, 8, 8, 12, 14},
       {0, 0, 0, 2, 0, 0, 4, 6, 16, 16, 16, 18, 24, 24, 28, 30},
       {0, 0, 0, 2, 0, 0, 4, 6, 0, 0, 0, 2, 8, 8, 12, 14},
       {32, 32, 32, 34, 32, 32, 36, 38, 48, 48, 48, 50, 56, 56, 60, 62},
       {0, 128, 0, 130, 0, 128, 4, 134, 0, 128, 0, 130, 8, 136, 12, 142},
       {0, 128, 0, 130, 0, 128, 4, 134, 16, 144, 16, 146, 24, 152, 28, 158},
       {64, 192, 64, 194, 64, 192, 68, 198, 64, 192, 64, 194, 72, 200, 76, 206},
       {96, 224, 96, 226, 96, 224, 100, 230, 112, 240, 112, 242, 120, 248, 124, 254}},
      {{0x0000, 0x0200, 0x0000, 0x0604, 0x0000, 0x0200, 0x0808, 0x0e0c},
       {0x0000, 0x0200, 0x0000, 0x0604, 0x1010, 0x1210, 0x1818, 0x1e1c},
       {0x0000, 0x0200, 0x0000, 0x0604, 0x0000, 0x0200, 0x0808, 0x0e0c},
       {0x2020, 0x2220, 0x2020, 0x2624, 0x3030, 0x3230, 0x3838, 0x3e3c},
       {0x8000, 0x8200, 0x8000, 0x8604, 0x8000, 0x8200, 0x8808, 0x8e0c},
       {0x8000, 0x8200, 0x8000, 0x8604, 0x9010, 0x9210, 0x9818, 0x9e1c},
       {0xc040, 0xc240, 0xc040, 0xc644, 0xc040, 0xc240, 0xc848, 0xce4c},
       {0xe060, 0xe260, 0xe060, 0xe664, 0xf070, 0xf270, 0xf878, 0xfe7c}},
      {{0x0200'0000, 0x0604'0000, 0x0200'0000, 0x0e0c'0808},
       {0x0200'0000, 0x0604'0000, 0x1210'1010, 0x1e1c'1818},
       {0x0200'0000, 0x0604'0000, 0x0200'0000, 0x0e0c'0808},
       {0x2220'2020, 0x2624'2020, 0x3230'3030, 0x3e3c'3838},
       {0x8200'8000, 0x8604'8000, 0x8200'8000, 0x8e0c'8808},
       {0x8200'8000, 0x8604'8000, 0x9210'9010, 0x9e1c'9818},
       {0xc240'c040, 0xc644'c040, 0xc240'c040, 0xce4c'c848},
       {0xe260'e060, 0xe664'e060, 0xf270'f070, 0xfe7c'f878}},
      {{0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000},
       {0x0604'0000'0200'0000, 0x1e1c'1818'1210'1010},
       {0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000},
       {0x2624'2020'2220'2020, 0x3e3c'3838'3230'3030},
       {0x8604'8000'8200'8000, 0x8e0c'8808'8200'8000},
       {0x8604'8000'8200'8000, 0x9e1c'9818'9210'9010},
       {0xc644'c040'c240'c040, 0xce4c'c848'c240'c040},
       {0xe664'e060'e260'e060, 0xfe7c'f878'f270'f070}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVandvx,
                        ExecMaskedVandvx,
                        {{0, 128, 2, 130, 0, 128, 2, 130, 8, 136, 10, 138, 8, 136, 10, 138},
                         {0, 128, 2, 130, 0, 128, 2, 130, 8, 136, 10, 138, 8, 136, 10, 138},
                         {32, 160, 34, 162, 32, 160, 34, 162, 40, 168, 42, 170, 40, 168, 42, 170},
                         {32, 160, 34, 162, 32, 160, 34, 162, 40, 168, 42, 170, 40, 168, 42, 170},
                         {0, 128, 2, 130, 0, 128, 2, 130, 8, 136, 10, 138, 8, 136, 10, 138},
                         {0, 128, 2, 130, 0, 128, 2, 130, 8, 136, 10, 138, 8, 136, 10, 138},
                         {32, 160, 34, 162, 32, 160, 34, 162, 40, 168, 42, 170, 40, 168, 42, 170},
                         {32, 160, 34, 162, 32, 160, 34, 162, 40, 168, 42, 170, 40, 168, 42, 170}},
                        {{0x8000, 0x8202, 0x8000, 0x8202, 0x8808, 0x8a0a, 0x8808, 0x8a0a},
                         {0x8000, 0x8202, 0x8000, 0x8202, 0x8808, 0x8a0a, 0x8808, 0x8a0a},
                         {0xa020, 0xa222, 0xa020, 0xa222, 0xa828, 0xaa2a, 0xa828, 0xaa2a},
                         {0xa020, 0xa222, 0xa020, 0xa222, 0xa828, 0xaa2a, 0xa828, 0xaa2a},
                         {0x8000, 0x8202, 0x8000, 0x8202, 0x8808, 0x8a0a, 0x8808, 0x8a0a},
                         {0x8000, 0x8202, 0x8000, 0x8202, 0x8808, 0x8a0a, 0x8808, 0x8a0a},
                         {0xa020, 0xa222, 0xa020, 0xa222, 0xa828, 0xaa2a, 0xa828, 0xaa2a},
                         {0xa020, 0xa222, 0xa020, 0xa222, 0xa828, 0xaa2a, 0xa828, 0xaa2a}},
                        {{0x8202'8000, 0x8202'8000, 0x8a0a'8808, 0x8a0a'8808},
                         {0x8202'8000, 0x8202'8000, 0x8a0a'8808, 0x8a0a'8808},
                         {0xa222'a020, 0xa222'a020, 0xaa2a'a828, 0xaa2a'a828},
                         {0xa222'a020, 0xa222'a020, 0xaa2a'a828, 0xaa2a'a828},
                         {0x8202'8000, 0x8202'8000, 0x8a0a'8808, 0x8a0a'8808},
                         {0x8202'8000, 0x8202'8000, 0x8a0a'8808, 0x8a0a'8808},
                         {0xa222'a020, 0xa222'a020, 0xaa2a'a828, 0xaa2a'a828},
                         {0xa222'a020, 0xa222'a020, 0xaa2a'a828, 0xaa2a'a828}},
                        {{0x8202'8000'8202'8000, 0x8a0a'8808'8a0a'8808},
                         {0x8202'8000'8202'8000, 0x8a0a'8808'8a0a'8808},
                         {0xa222'a020'a222'a020, 0xaa2a'a828'aa2a'a828},
                         {0xa222'a020'a222'a020, 0xaa2a'a828'aa2a'a828},
                         {0x8202'8000'8202'8000, 0x8a0a'8808'8a0a'8808},
                         {0x8202'8000'8202'8000, 0x8a0a'8808'8a0a'8808},
                         {0xa222'a020'a222'a020, 0xaa2a'a828'aa2a'a828},
                         {0xa222'a020'a222'a020, 0xaa2a'a828'aa2a'a828}},
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVandvi,
      ExecMaskedVandvi,
      {{0, 129, 0, 129, 4, 133, 4, 133, 0, 129, 0, 129, 4, 133, 4, 133},
       {16, 145, 16, 145, 20, 149, 20, 149, 16, 145, 16, 145, 20, 149, 20, 149},
       {32, 161, 32, 161, 36, 165, 36, 165, 32, 161, 32, 161, 36, 165, 36, 165},
       {48, 177, 48, 177, 52, 181, 52, 181, 48, 177, 48, 177, 52, 181, 52, 181},
       {64, 193, 64, 193, 68, 197, 68, 197, 64, 193, 64, 193, 68, 197, 68, 197},
       {80, 209, 80, 209, 84, 213, 84, 213, 80, 209, 80, 209, 84, 213, 84, 213},
       {96, 225, 96, 225, 100, 229, 100, 229, 96, 225, 96, 225, 100, 229, 100, 229},
       {112, 241, 112, 241, 116, 245, 116, 245, 112, 241, 112, 241, 116, 245, 116, 245}},
      {{0x8100, 0x8300, 0x8504, 0x8704, 0x8900, 0x8b00, 0x8d04, 0x8f04},
       {0x9110, 0x9310, 0x9514, 0x9714, 0x9910, 0x9b10, 0x9d14, 0x9f14},
       {0xa120, 0xa320, 0xa524, 0xa724, 0xa920, 0xab20, 0xad24, 0xaf24},
       {0xb130, 0xb330, 0xb534, 0xb734, 0xb930, 0xbb30, 0xbd34, 0xbf34},
       {0xc140, 0xc340, 0xc544, 0xc744, 0xc940, 0xcb40, 0xcd44, 0xcf44},
       {0xd150, 0xd350, 0xd554, 0xd754, 0xd950, 0xdb50, 0xdd54, 0xdf54},
       {0xe160, 0xe360, 0xe564, 0xe764, 0xe960, 0xeb60, 0xed64, 0xef64},
       {0xf170, 0xf370, 0xf574, 0xf774, 0xf970, 0xfb70, 0xfd74, 0xff74}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8900, 0x8f0e'8d04},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9910, 0x9f1e'9d14},
       {0xa322'a120, 0xa726'a524, 0xab2a'a920, 0xaf2e'ad24},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b930, 0xbf3e'bd34},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c940, 0xcf4e'cd44},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd950, 0xdf5e'dd54},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e960, 0xef6e'ed64},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f970, 0xff7e'fd74}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8900},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9910},
       {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a920},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b930},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c940},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd950},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e960},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f970}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVorvv() {
  asm("vor.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVorvv() {
  asm("vor.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVorvx() {
  asm("vor.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVorvx() {
  asm("vor.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVorvi() {
  asm("vor.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVorvi() {
  asm("vor.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVor) {
  TestVectorInstruction(
      ExecVorvv,
      ExecMaskedVorvv,
      {{0, 131, 6, 135, 13, 143, 14, 143, 25, 155, 30, 159, 28, 159, 30, 159},
       {48, 179, 54, 183, 61, 191, 62, 191, 57, 187, 62, 191, 60, 191, 62, 191},
       {96, 227, 102, 231, 109, 239, 110, 239, 121, 251, 126, 255, 124, 255, 126, 255},
       {112, 243, 118, 247, 125, 255, 126, 255, 121, 251, 126, 255, 124, 255, 126, 255},
       {192, 195, 198, 199, 205, 207, 206, 207, 217, 219, 222, 223, 220, 223, 222, 223},
       {240, 243, 246, 247, 253, 255, 254, 255, 249, 251, 254, 255, 252, 255, 254, 255},
       {224, 227, 230, 231, 237, 239, 238, 239, 249, 251, 254, 255, 252, 255, 254, 255},
       {240, 243, 246, 247, 253, 255, 254, 255, 249, 251, 254, 255, 252, 255, 254, 255}},
      {{0x8300, 0x8706, 0x8f0d, 0x8f0e, 0x9b19, 0x9f1e, 0x9f1c, 0x9f1e},
       {0xb330, 0xb736, 0xbf3d, 0xbf3e, 0xbb39, 0xbf3e, 0xbf3c, 0xbf3e},
       {0xe360, 0xe766, 0xef6d, 0xef6e, 0xfb79, 0xff7e, 0xff7c, 0xff7e},
       {0xf370, 0xf776, 0xff7d, 0xff7e, 0xfb79, 0xff7e, 0xff7c, 0xff7e},
       {0xc3c0, 0xc7c6, 0xcfcd, 0xcfce, 0xdbd9, 0xdfde, 0xdfdc, 0xdfde},
       {0xf3f0, 0xf7f6, 0xfffd, 0xfffe, 0xfbf9, 0xfffe, 0xfffc, 0xfffe},
       {0xe3e0, 0xe7e6, 0xefed, 0xefee, 0xfbf9, 0xfffe, 0xfffc, 0xfffe},
       {0xf3f0, 0xf7f6, 0xfffd, 0xfffe, 0xfbf9, 0xfffe, 0xfffc, 0xfffe}},
      {{0x8706'8300, 0x8f0e'8f0d, 0x9f1e'9b19, 0x9f1e'9f1c},
       {0xb736'b330, 0xbf3e'bf3d, 0xbf3e'bb39, 0xbf3e'bf3c},
       {0xe766'e360, 0xef6e'ef6d, 0xff7e'fb79, 0xff7e'ff7c},
       {0xf776'f370, 0xff7e'ff7d, 0xff7e'fb79, 0xff7e'ff7c},
       {0xc7c6'c3c0, 0xcfce'cfcd, 0xdfde'dbd9, 0xdfde'dfdc},
       {0xf7f6'f3f0, 0xfffe'fffd, 0xfffe'fbf9, 0xfffe'fffc},
       {0xe7e6'e3e0, 0xefee'efed, 0xfffe'fbf9, 0xfffe'fffc},
       {0xf7f6'f3f0, 0xfffe'fffd, 0xfffe'fbf9, 0xfffe'fffc}},
      {{0x8f0e'8f0d'8706'8300, 0x9f1e'9f1c'9f1e'9b19},
       {0xbf3e'bf3d'b736'b330, 0xbf3e'bf3c'bf3e'bb39},
       {0xef6e'ef6d'e766'e360, 0xff7e'ff7c'ff7e'fb79},
       {0xff7e'ff7d'f776'f370, 0xff7e'ff7c'ff7e'fb79},
       {0xcfce'cfcd'c7c6'c3c0, 0xdfde'dfdc'dfde'dbd9},
       {0xfffe'fffd'f7f6'f3f0, 0xfffe'fffc'fffe'fbf9},
       {0xefee'efed'e7e6'e3e0, 0xfffe'fffc'fffe'fbf9},
       {0xfffe'fffd'f7f6'f3f0, 0xfffe'fffc'fffe'fbf9}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVorvx,
      ExecMaskedVorvx,
      {{170, 171, 170, 171, 174, 175, 174, 175, 170, 171, 170, 171, 174, 175, 174, 175},
       {186, 187, 186, 187, 190, 191, 190, 191, 186, 187, 186, 187, 190, 191, 190, 191},
       {170, 171, 170, 171, 174, 175, 174, 175, 170, 171, 170, 171, 174, 175, 174, 175},
       {186, 187, 186, 187, 190, 191, 190, 191, 186, 187, 186, 187, 190, 191, 190, 191},
       {234, 235, 234, 235, 238, 239, 238, 239, 234, 235, 234, 235, 238, 239, 238, 239},
       {250, 251, 250, 251, 254, 255, 254, 255, 250, 251, 250, 251, 254, 255, 254, 255},
       {234, 235, 234, 235, 238, 239, 238, 239, 234, 235, 234, 235, 238, 239, 238, 239},
       {250, 251, 250, 251, 254, 255, 254, 255, 250, 251, 250, 251, 254, 255, 254, 255}},
      {{0xabaa, 0xabaa, 0xafae, 0xafae, 0xabaa, 0xabaa, 0xafae, 0xafae},
       {0xbbba, 0xbbba, 0xbfbe, 0xbfbe, 0xbbba, 0xbbba, 0xbfbe, 0xbfbe},
       {0xabaa, 0xabaa, 0xafae, 0xafae, 0xabaa, 0xabaa, 0xafae, 0xafae},
       {0xbbba, 0xbbba, 0xbfbe, 0xbfbe, 0xbbba, 0xbbba, 0xbfbe, 0xbfbe},
       {0xebea, 0xebea, 0xefee, 0xefee, 0xebea, 0xebea, 0xefee, 0xefee},
       {0xfbfa, 0xfbfa, 0xfffe, 0xfffe, 0xfbfa, 0xfbfa, 0xfffe, 0xfffe},
       {0xebea, 0xebea, 0xefee, 0xefee, 0xebea, 0xebea, 0xefee, 0xefee},
       {0xfbfa, 0xfbfa, 0xfffe, 0xfffe, 0xfbfa, 0xfbfa, 0xfffe, 0xfffe}},
      {{0xabaa'abaa, 0xafae'afae, 0xabaa'abaa, 0xafae'afae},
       {0xbbba'bbba, 0xbfbe'bfbe, 0xbbba'bbba, 0xbfbe'bfbe},
       {0xabaa'abaa, 0xafae'afae, 0xabaa'abaa, 0xafae'afae},
       {0xbbba'bbba, 0xbfbe'bfbe, 0xbbba'bbba, 0xbfbe'bfbe},
       {0xebea'ebea, 0xefee'efee, 0xebea'ebea, 0xefee'efee},
       {0xfbfa'fbfa, 0xfffe'fffe, 0xfbfa'fbfa, 0xfffe'fffe},
       {0xebea'ebea, 0xefee'efee, 0xebea'ebea, 0xefee'efee},
       {0xfbfa'fbfa, 0xfffe'fffe, 0xfbfa'fbfa, 0xfffe'fffe}},
      {{0xafae'afae'abaa'abaa, 0xafae'afae'abaa'abaa},
       {0xbfbe'bfbe'bbba'bbba, 0xbfbe'bfbe'bbba'bbba},
       {0xafae'afae'abaa'abaa, 0xafae'afae'abaa'abaa},
       {0xbfbe'bfbe'bbba'bbba, 0xbfbe'bfbe'bbba'bbba},
       {0xefee'efee'ebea'ebea, 0xefee'efee'ebea'ebea},
       {0xfffe'fffe'fbfa'fbfa, 0xfffe'fffe'fbfa'fbfa},
       {0xefee'efee'ebea'ebea, 0xefee'efee'ebea'ebea},
       {0xfffe'fffe'fbfa'fbfa, 0xfffe'fffe'fbfa'fbfa}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVorvi,
      ExecMaskedVorvi,
      {{245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255}},
      {{0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff}},
      {{0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd}},
      {{0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVxorvv() {
  asm("vxor.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVxorvv() {
  asm("vxor.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVxorvx() {
  asm("vxor.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVxorvx() {
  asm("vxor.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVxorvi() {
  asm("vxor.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVxorvi() {
  asm("vxor.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVxor) {
  TestVectorInstruction(
      ExecVxorvv,
      ExecMaskedVxorvv,
      {{0, 131, 6, 133, 13, 143, 10, 137, 25, 155, 30, 157, 20, 151, 18, 145},
       {48, 179, 54, 181, 61, 191, 58, 185, 41, 171, 46, 173, 36, 167, 34, 161},
       {96, 227, 102, 229, 109, 239, 106, 233, 121, 251, 126, 253, 116, 247, 114, 241},
       {80, 211, 86, 213, 93, 223, 90, 217, 73, 203, 78, 205, 68, 199, 66, 193},
       {192, 67, 198, 69, 205, 79, 202, 73, 217, 91, 222, 93, 212, 87, 210, 81},
       {240, 115, 246, 117, 253, 127, 250, 121, 233, 107, 238, 109, 228, 103, 226, 97},
       {160, 35, 166, 37, 173, 47, 170, 41, 185, 59, 190, 61, 180, 55, 178, 49},
       {144, 19, 150, 21, 157, 31, 154, 25, 137, 11, 142, 13, 132, 7, 130, 1}},
      {{0x8300, 0x8506, 0x8f0d, 0x890a, 0x9b19, 0x9d1e, 0x9714, 0x9112},
       {0xb330, 0xb536, 0xbf3d, 0xb93a, 0xab29, 0xad2e, 0xa724, 0xa122},
       {0xe360, 0xe566, 0xef6d, 0xe96a, 0xfb79, 0xfd7e, 0xf774, 0xf172},
       {0xd350, 0xd556, 0xdf5d, 0xd95a, 0xcb49, 0xcd4e, 0xc744, 0xc142},
       {0x43c0, 0x45c6, 0x4fcd, 0x49ca, 0x5bd9, 0x5dde, 0x57d4, 0x51d2},
       {0x73f0, 0x75f6, 0x7ffd, 0x79fa, 0x6be9, 0x6dee, 0x67e4, 0x61e2},
       {0x23a0, 0x25a6, 0x2fad, 0x29aa, 0x3bb9, 0x3dbe, 0x37b4, 0x31b2},
       {0x1390, 0x1596, 0x1f9d, 0x199a, 0x0b89, 0x0d8e, 0x0784, 0x0182}},
      {{0x8506'8300, 0x890a'8f0d, 0x9d1e'9b19, 0x9112'9714},
       {0xb536'b330, 0xb93a'bf3d, 0xad2e'ab29, 0xa122'a724},
       {0xe566'e360, 0xe96a'ef6d, 0xfd7e'fb79, 0xf172'f774},
       {0xd556'd350, 0xd95a'df5d, 0xcd4e'cb49, 0xc142'c744},
       {0x45c6'43c0, 0x49ca'4fcd, 0x5dde'5bd9, 0x51d2'57d4},
       {0x75f6'73f0, 0x79fa'7ffd, 0x6dee'6be9, 0x61e2'67e4},
       {0x25a6'23a0, 0x29aa'2fad, 0x3dbe'3bb9, 0x31b2'37b4},
       {0x1596'1390, 0x199a'1f9d, 0x0d8e'0b89, 0x0182'0784}},
      {{0x890a'8f0d'8506'8300, 0x9112'9714'9d1e'9b19},
       {0xb93a'bf3d'b536'b330, 0xa122'a724'ad2e'ab29},
       {0xe96a'ef6d'e566'e360, 0xf172'f774'fd7e'fb79},
       {0xd95a'df5d'd556'd350, 0xc142'c744'cd4e'cb49},
       {0x49ca'4fcd'45c6'43c0, 0x51d2'57d4'5dde'5bd9},
       {0x79fa'7ffd'75f6'73f0, 0x61e2'67e4'6dee'6be9},
       {0x29aa'2fad'25a6'23a0, 0x31b2'37b4'3dbe'3bb9},
       {0x199a'1f9d'1596'1390, 0x0182'0784'0d8e'0b89}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVxorvx,
      ExecMaskedVxorvx,
      {{170, 43, 168, 41, 174, 47, 172, 45, 162, 35, 160, 33, 166, 39, 164, 37},
       {186, 59, 184, 57, 190, 63, 188, 61, 178, 51, 176, 49, 182, 55, 180, 53},
       {138, 11, 136, 9, 142, 15, 140, 13, 130, 3, 128, 1, 134, 7, 132, 5},
       {154, 27, 152, 25, 158, 31, 156, 29, 146, 19, 144, 17, 150, 23, 148, 21},
       {234, 107, 232, 105, 238, 111, 236, 109, 226, 99, 224, 97, 230, 103, 228, 101},
       {250, 123, 248, 121, 254, 127, 252, 125, 242, 115, 240, 113, 246, 119, 244, 117},
       {202, 75, 200, 73, 206, 79, 204, 77, 194, 67, 192, 65, 198, 71, 196, 69},
       {218, 91, 216, 89, 222, 95, 220, 93, 210, 83, 208, 81, 214, 87, 212, 85}},
      {{0x2baa, 0x29a8, 0x2fae, 0x2dac, 0x23a2, 0x21a0, 0x27a6, 0x25a4},
       {0x3bba, 0x39b8, 0x3fbe, 0x3dbc, 0x33b2, 0x31b0, 0x37b6, 0x35b4},
       {0x0b8a, 0x0988, 0x0f8e, 0x0d8c, 0x0382, 0x0180, 0x0786, 0x0584},
       {0x1b9a, 0x1998, 0x1f9e, 0x1d9c, 0x1392, 0x1190, 0x1796, 0x1594},
       {0x6bea, 0x69e8, 0x6fee, 0x6dec, 0x63e2, 0x61e0, 0x67e6, 0x65e4},
       {0x7bfa, 0x79f8, 0x7ffe, 0x7dfc, 0x73f2, 0x71f0, 0x77f6, 0x75f4},
       {0x4bca, 0x49c8, 0x4fce, 0x4dcc, 0x43c2, 0x41c0, 0x47c6, 0x45c4},
       {0x5bda, 0x59d8, 0x5fde, 0x5ddc, 0x53d2, 0x51d0, 0x57d6, 0x55d4}},
      {{0x29a8'2baa, 0x2dac'2fae, 0x21a0'23a2, 0x25a4'27a6},
       {0x39b8'3bba, 0x3dbc'3fbe, 0x31b0'33b2, 0x35b4'37b6},
       {0x0988'0b8a, 0x0d8c'0f8e, 0x0180'0382, 0x0584'0786},
       {0x1998'1b9a, 0x1d9c'1f9e, 0x1190'1392, 0x1594'1796},
       {0x69e8'6bea, 0x6dec'6fee, 0x61e0'63e2, 0x65e4'67e6},
       {0x79f8'7bfa, 0x7dfc'7ffe, 0x71f0'73f2, 0x75f4'77f6},
       {0x49c8'4bca, 0x4dcc'4fce, 0x41c0'43c2, 0x45c4'47c6},
       {0x59d8'5bda, 0x5ddc'5fde, 0x51d0'53d2, 0x55d4'57d6}},
      {{0x2dac'2fae'29a8'2baa, 0x25a4'27a6'21a0'23a2},
       {0x3dbc'3fbe'39b8'3bba, 0x35b4'37b6'31b0'33b2},
       {0x0d8c'0f8e'0988'0b8a, 0x0584'0786'0180'0382},
       {0x1d9c'1f9e'1998'1b9a, 0x1594'1796'1190'1392},
       {0x6dec'6fee'69e8'6bea, 0x65e4'67e6'61e0'63e2},
       {0x7dfc'7ffe'79f8'7bfa, 0x75f4'77f6'71f0'73f2},
       {0x4dcc'4fce'49c8'4bca, 0x45c4'47c6'41c0'43c2},
       {0x5ddc'5fde'59d8'5bda, 0x55d4'57d6'51d0'53d2}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVxorvi,
      ExecMaskedVxorvi,
      {{245, 116, 247, 118, 241, 112, 243, 114, 253, 124, 255, 126, 249, 120, 251, 122},
       {229, 100, 231, 102, 225, 96, 227, 98, 237, 108, 239, 110, 233, 104, 235, 106},
       {213, 84, 215, 86, 209, 80, 211, 82, 221, 92, 223, 94, 217, 88, 219, 90},
       {197, 68, 199, 70, 193, 64, 195, 66, 205, 76, 207, 78, 201, 72, 203, 74},
       {181, 52, 183, 54, 177, 48, 179, 50, 189, 60, 191, 62, 185, 56, 187, 58},
       {165, 36, 167, 38, 161, 32, 163, 34, 173, 44, 175, 46, 169, 40, 171, 42},
       {149, 20, 151, 22, 145, 16, 147, 18, 157, 28, 159, 30, 153, 24, 155, 26},
       {133, 4, 135, 6, 129, 0, 131, 2, 141, 12, 143, 14, 137, 8, 139, 10}},
      {{0x7ef5, 0x7cf7, 0x7af1, 0x78f3, 0x76fd, 0x74ff, 0x72f9, 0x70fb},
       {0x6ee5, 0x6ce7, 0x6ae1, 0x68e3, 0x66ed, 0x64ef, 0x62e9, 0x60eb},
       {0x5ed5, 0x5cd7, 0x5ad1, 0x58d3, 0x56dd, 0x54df, 0x52d9, 0x50db},
       {0x4ec5, 0x4cc7, 0x4ac1, 0x48c3, 0x46cd, 0x44cf, 0x42c9, 0x40cb},
       {0x3eb5, 0x3cb7, 0x3ab1, 0x38b3, 0x36bd, 0x34bf, 0x32b9, 0x30bb},
       {0x2ea5, 0x2ca7, 0x2aa1, 0x28a3, 0x26ad, 0x24af, 0x22a9, 0x20ab},
       {0x1e95, 0x1c97, 0x1a91, 0x1893, 0x169d, 0x149f, 0x1299, 0x109b},
       {0x0e85, 0x0c87, 0x0a81, 0x0883, 0x068d, 0x048f, 0x0289, 0x008b}},
      {{0x7cfd'7ef5, 0x78f9'7af1, 0x74f5'76fd, 0x70f1'72f9},
       {0x6ced'6ee5, 0x68e9'6ae1, 0x64e5'66ed, 0x60e1'62e9},
       {0x5cdd'5ed5, 0x58d9'5ad1, 0x54d5'56dd, 0x50d1'52d9},
       {0x4ccd'4ec5, 0x48c9'4ac1, 0x44c5'46cd, 0x40c1'42c9},
       {0x3cbd'3eb5, 0x38b9'3ab1, 0x34b5'36bd, 0x30b1'32b9},
       {0x2cad'2ea5, 0x28a9'2aa1, 0x24a5'26ad, 0x20a1'22a9},
       {0x1c9d'1e95, 0x1899'1a91, 0x1495'169d, 0x1091'1299},
       {0x0c8d'0e85, 0x0889'0a81, 0x0485'068d, 0x0081'0289}},
      {{0x78f9'7afb'7cfd'7ef5, 0x70f1'72f3'74f5'76fd},
       {0x68e9'6aeb'6ced'6ee5, 0x60e1'62e3'64e5'66ed},
       {0x58d9'5adb'5cdd'5ed5, 0x50d1'52d3'54d5'56dd},
       {0x48c9'4acb'4ccd'4ec5, 0x40c1'42c3'44c5'46cd},
       {0x38b9'3abb'3cbd'3eb5, 0x30b1'32b3'34b5'36bd},
       {0x28a9'2aab'2cad'2ea5, 0x20a1'22a3'24a5'26ad},
       {0x1899'1a9b'1c9d'1e95, 0x1091'1293'1495'169d},
       {0x0889'0a8b'0c8d'0e85, 0x0081'0283'0485'068d}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVmfeqvv() {
  asm("vmfeq.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmfeqvv() {
  asm("vmfeq.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVmfeqvf() {
  asm("vmfeq.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmfeqvf() {
  asm("vmfeq.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmfeq) {
  TestVectorMaskTargetInstruction(
      ExecVmfeqvv, ExecMaskedVmfeqvv, 0x0000'0007, 0x0001, kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      ExecVmfeqvf, ExecMaskedVmfeqvf, 0x0000'0040, 0x0020, kVectorComparisonSource);
}

[[gnu::naked]] void ExecVaadduvv() {
  asm("vaaddu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaadduvv() {
  asm("vaaddu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaadduvx() {
  asm("vaaddu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaadduvx() {
  asm("vaaddu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaaddvv() {
  asm("vaadd.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaaddvv() {
  asm("vaadd.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaaddvx() {
  asm("vaadd.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaaddvx() {
  asm("vaadd.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVasubuvv() {
  asm("vasubu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVasubuvv() {
  asm("vasubu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVasubuvx() {
  asm("vasubu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVasubuvx() {
  asm("vasubu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVasubvv() {
  asm("vasub.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVasubvv() {
  asm("vasub.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVasubvx() {
  asm("vasub.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVasubvx() {
  asm("vasub.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipuwi() {
  asm("vnclipu.wi  v8, v16, 0xa\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipuwi() {
  asm("vnclipu.wi  v8, v16, 0xa, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipwi() {
  asm("vnclip.wi  v8, v16, 0xa\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipwi() {
  asm("vnclip.wi  v8, v16, 0xa, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipuwx() {
  asm("vnclipu.wx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipuwx() {
  asm("vnclipu.wx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipwx() {
  asm("vnclip.wx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipwx() {
  asm("vnclip.wx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipuwv() {
  asm("vnclipu.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipuwv() {
  asm("vnclipu.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipwv() {
  asm("vnclip.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipwv() {
  asm("vnclip.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsmulvv() {
  asm("vsmul.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsmulvv() {
  asm("vsmul.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsmulvx() {
  asm("vsmul.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsmulvx() {
  asm("vsmul.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssrlvv() {
  asm("vssrl.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssrlvv() {
  asm("vssrl.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssrlvx() {
  asm("vssrl.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssrlvx() {
  asm("vssrl.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssrlvi() {
  asm("vssrl.vi  v8, v16, 0xa\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssrlvi() {
  asm("vssrl.vi  v8, v16, 0xa, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssravv() {
  asm("vssra.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssravv() {
  asm("vssra.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssravx() {
  asm("vssra.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssravx() {
  asm("vssra.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssravi() {
  asm("vssra.vi  v8, v16, 0xa\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssravi() {
  asm("vssra.vi  v8, v16, 0xa, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestRDN) {
  uint64_t vxrm;
  asm volatile(
      "csrr %0, vxrm\n\t"
      "csrwi vxrm, %c1\n\t"
      : "=r"(vxrm)
      : "i"(VXRMFlags::RDN));
  TestVectorInstruction(
      ExecVaadduvv,
      ExecMaskedVaadduvv,
      {{0, 137, 3, 140, 6, 143, 9, 146, 12, 133, 15, 136, 18, 139, 21, 142},
       {24, 161, 27, 164, 30, 167, 33, 170, 36, 157, 39, 160, 42, 163, 45, 166},
       {48, 185, 51, 188, 54, 191, 57, 194, 60, 181, 63, 184, 66, 187, 69, 190},
       {72, 209, 75, 212, 78, 215, 81, 218, 84, 205, 87, 208, 90, 211, 93, 214},
       {96, 105, 99, 108, 102, 111, 105, 114, 108, 101, 111, 104, 114, 107, 117, 110},
       {120, 129, 123, 132, 126, 135, 129, 138, 132, 125, 135, 128, 138, 131, 141, 134},
       {144, 153, 147, 156, 150, 159, 153, 162, 156, 149, 159, 152, 162, 155, 165, 158},
       {168, 177, 171, 180, 174, 183, 177, 186, 180, 173, 183, 176, 186, 179, 189, 182}},
      {{0x8980, 0x8c83, 0x8f86, 0x9289, 0x858c, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79e, 0xaaa1, 0x9da4, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb6, 0xc2b9, 0xb5bc, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7ce, 0xdad1, 0xcdd4, 0xd0d7, 0xd3da, 0xd6dd},
       {0x69e0, 0x6ce3, 0x6fe6, 0x72e9, 0x65ec, 0x68ef, 0x6bf2, 0x6ef5},
       {0x81f8, 0x84fb, 0x87fe, 0x8b01, 0x7e04, 0x8107, 0x840a, 0x870d},
       {0x9a10, 0x9d13, 0xa016, 0xa319, 0x961c, 0x991f, 0x9c22, 0x9f25},
       {0xb228, 0xb52b, 0xb82e, 0xbb31, 0xae34, 0xb137, 0xb43a, 0xb73d}},
      {{0x8c83'8980, 0x9289'8f86, 0x888f'858c, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79e, 0xa0a7'9da4, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb6, 0xb8bf'b5bc, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7ce, 0xd0d7'cdd4, 0xd6dd'd3da},
       {0x6ce3'69e0, 0x72e9'6fe6, 0x68ef'65ec, 0x6ef5'6bf2},
       {0x84fb'81f8, 0x8b01'87fe, 0x8107'7e04, 0x870d'840a},
       {0x9d13'9a10, 0xa319'a016, 0x991f'961c, 0x9f25'9c22},
       {0xb52b'b228, 0xbb31'b82e, 0xb137'ae34, 0xb73d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858c},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da4},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bc},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd4},
       {0x72e9'6fe6'ece3'69e0, 0x6ef5'6bf2'68ef'65ec},
       {0x8b01'87ff'04fb'81f8, 0x870d'840a'8107'7e04},
       {0xa319'a017'1d13'9a10, 0x9f25'9c22'991f'961c},
       {0xbb31'b82f'352b'b228, 0xb73d'b43a'b137'ae34}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaadduvx,
      ExecMaskedVaadduvx,
      {{85, 149, 86, 150, 87, 151, 88, 152, 89, 153, 90, 154, 91, 155, 92, 156},
       {93, 157, 94, 158, 95, 159, 96, 160, 97, 161, 98, 162, 99, 163, 100, 164},
       {101, 165, 102, 166, 103, 167, 104, 168, 105, 169, 106, 170, 107, 171, 108, 172},
       {109, 173, 110, 174, 111, 175, 112, 176, 113, 177, 114, 178, 115, 179, 116, 180},
       {117, 181, 118, 182, 119, 183, 120, 184, 121, 185, 122, 186, 123, 187, 124, 188},
       {125, 189, 126, 190, 127, 191, 128, 192, 129, 193, 130, 194, 131, 195, 132, 196},
       {133, 197, 134, 198, 135, 199, 136, 200, 137, 201, 138, 202, 139, 203, 140, 204},
       {141, 205, 142, 206, 143, 207, 144, 208, 145, 209, 146, 210, 147, 211, 148, 212}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvv,
      ExecMaskedVaaddvv,
      {{0, 137, 3, 140, 6, 143, 9, 146, 12, 133, 15, 136, 18, 139, 21, 142},
       {24, 161, 27, 164, 30, 167, 33, 170, 36, 157, 39, 160, 42, 163, 45, 166},
       {48, 185, 51, 188, 54, 191, 57, 194, 60, 181, 63, 184, 66, 187, 69, 190},
       {72, 209, 75, 212, 78, 215, 81, 218, 84, 205, 87, 208, 90, 211, 93, 214},
       {224, 233, 227, 236, 230, 239, 233, 242, 236, 229, 239, 232, 242, 235, 245, 238},
       {248, 1, 251, 4, 254, 7, 1, 10, 4, 253, 7, 0, 10, 3, 13, 6},
       {16, 25, 19, 28, 22, 31, 25, 34, 28, 21, 31, 24, 34, 27, 37, 30},
       {40, 49, 43, 52, 46, 55, 49, 58, 52, 45, 55, 48, 58, 51, 61, 54}},
      {{0x8980, 0x8c83, 0x8f86, 0x9289, 0x858c, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79e, 0xaaa1, 0x9da4, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb6, 0xc2b9, 0xb5bc, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7ce, 0xdad1, 0xcdd4, 0xd0d7, 0xd3da, 0xd6dd},
       {0xe9e0, 0xece3, 0xefe6, 0xf2e9, 0xe5ec, 0xe8ef, 0xebf2, 0xeef5},
       {0x01f8, 0x04fb, 0x07fe, 0x0b01, 0xfe04, 0x0107, 0x040a, 0x070d},
       {0x1a10, 0x1d13, 0x2016, 0x2319, 0x161c, 0x191f, 0x1c22, 0x1f25},
       {0x3228, 0x352b, 0x382e, 0x3b31, 0x2e34, 0x3137, 0x343a, 0x373d}},
      {{0x8c83'8980, 0x9289'8f86, 0x888f'858c, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79e, 0xa0a7'9da4, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb6, 0xb8bf'b5bc, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7ce, 0xd0d7'cdd4, 0xd6dd'd3da},
       {0xece3'69e0, 0xf2e9'6fe6, 0xe8ef'65ec, 0xeef5'6bf2},
       {0x04fb'81f8, 0x0b01'87fe, 0x0107'7e04, 0x070d'840a},
       {0x1d13'9a10, 0x2319'a016, 0x191f'961c, 0x1f25'9c22},
       {0x352b'b228, 0x3b31'b82e, 0x3137'ae34, 0x373d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858c},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da4},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bc},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd4},
       {0xf2e9'6fe6'ece3'69e0, 0xeef5'6bf2'68ef'65ec},
       {0x0b01'87ff'04fb'81f8, 0x070d'840a'8107'7e04},
       {0x2319'a017'1d13'9a10, 0x1f25'9c22'991f'961c},
       {0x3b31'b82f'352b'b228, 0x373d'b43a'b137'ae34}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvx,
      ExecMaskedVaaddvx,
      {{213, 149, 214, 150, 215, 151, 216, 152, 217, 153, 218, 154, 219, 155, 220, 156},
       {221, 157, 222, 158, 223, 159, 224, 160, 225, 161, 226, 162, 227, 163, 228, 164},
       {229, 165, 230, 166, 231, 167, 232, 168, 233, 169, 234, 170, 235, 171, 236, 172},
       {237, 173, 238, 174, 239, 175, 240, 176, 241, 177, 242, 178, 243, 179, 244, 180},
       {245, 181, 246, 182, 247, 183, 248, 184, 249, 185, 250, 186, 251, 187, 252, 188},
       {253, 189, 254, 190, 255, 191, 0, 192, 1, 193, 2, 194, 3, 195, 4, 196},
       {5, 197, 6, 198, 7, 199, 8, 200, 9, 201, 10, 202, 11, 203, 12, 204},
       {13, 205, 14, 206, 15, 207, 16, 208, 17, 209, 18, 210, 19, 211, 20, 212}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvv,
      ExecMaskedVasubuvv,
      {{0, 247, 255, 246, 253, 245, 253, 244, 251, 3, 251, 2, 250, 1, 249, 0},
       {248, 239, 247, 238, 245, 237, 245, 236, 243, 251, 243, 250, 242, 249, 241, 248},
       {240, 231, 239, 230, 237, 229, 237, 228, 235, 243, 235, 242, 234, 241, 233, 240},
       {232, 223, 231, 222, 229, 221, 229, 220, 227, 235, 227, 234, 226, 233, 225, 232},
       {224, 87, 223, 86, 221, 85, 221, 84, 219, 99, 219, 98, 218, 97, 217, 96},
       {216, 79, 215, 78, 213, 77, 213, 76, 211, 91, 211, 90, 210, 89, 209, 88},
       {208, 71, 207, 70, 205, 69, 205, 68, 203, 83, 203, 82, 202, 81, 201, 80},
       {200, 63, 199, 62, 197, 61, 197, 60, 195, 75, 195, 74, 194, 73, 193, 72}},
      {{0xf780, 0xf67f, 0xf57d, 0xf47d, 0x037b, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed75, 0xec75, 0xfb73, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56d, 0xe46d, 0xf36b, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd65, 0xdc65, 0xeb63, 0xea63, 0xe962, 0xe861},
       {0x5760, 0x565f, 0x555d, 0x545d, 0x635b, 0x625b, 0x615a, 0x6059},
       {0x4f58, 0x4e57, 0x4d55, 0x4c55, 0x5b53, 0x5a53, 0x5952, 0x5851},
       {0x4750, 0x464f, 0x454d, 0x444d, 0x534b, 0x524b, 0x514a, 0x5049},
       {0x3f48, 0x3e47, 0x3d45, 0x3c45, 0x4b43, 0x4a43, 0x4942, 0x4841}},
      {{0xf67e'f780, 0xf47c'f57d, 0x027b'037b, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed75, 0xfa72'fb73, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56d, 0xf26a'f36b, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd65, 0xea62'eb63, 0xe860'e962},
       {0x565f'5760, 0x545d'555d, 0x625b'635b, 0x6059'615a},
       {0x4e57'4f58, 0x4c55'4d55, 0x5a53'5b53, 0x5851'5952},
       {0x464f'4750, 0x444d'454d, 0x524b'534b, 0x5049'514a},
       {0x3e47'3f48, 0x3c45'3d45, 0x4a43'4b43, 0x4841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037b},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb73},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36b},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb63},
       {0x545d'555d'd65f'5760, 0x6059'615a'625b'635b},
       {0x4c55'4d55'ce57'4f58, 0x5851'5952'5a53'5b53},
       {0x444d'454d'c64f'4750, 0x5049'514a'524b'534b},
       {0x3c45'3d45'be47'3f48, 0x4841'4942'4a43'4b43}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvx,
      ExecMaskedVasubuvx,
      {{171, 235, 172, 236, 173, 237, 174, 238, 175, 239, 176, 240, 177, 241, 178, 242},
       {179, 243, 180, 244, 181, 245, 182, 246, 183, 247, 184, 248, 185, 249, 186, 250},
       {187, 251, 188, 252, 189, 253, 190, 254, 191, 255, 192, 0, 193, 1, 194, 2},
       {195, 3, 196, 4, 197, 5, 198, 6, 199, 7, 200, 8, 201, 9, 202, 10},
       {203, 11, 204, 12, 205, 13, 206, 14, 207, 15, 208, 16, 209, 17, 210, 18},
       {211, 19, 212, 20, 213, 21, 214, 22, 215, 23, 216, 24, 217, 25, 218, 26},
       {219, 27, 220, 28, 221, 29, 222, 30, 223, 31, 224, 32, 225, 33, 226, 34},
       {227, 35, 228, 36, 229, 37, 230, 38, 231, 39, 232, 40, 233, 41, 234, 42}},
      {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
       {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
       {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
       {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
       {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
       {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
       {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
       {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
      {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
       {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
       {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
       {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
       {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
       {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
       {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
       {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
      {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
       {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
       {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
       {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
       {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
       {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
       {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
       {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubvv,
      ExecMaskedVasubvv,
      {{0, 247, 255, 246, 253, 245, 253, 244, 251, 3, 251, 2, 250, 1, 249, 0},
       {248, 239, 247, 238, 245, 237, 245, 236, 243, 251, 243, 250, 242, 249, 241, 248},
       {240, 231, 239, 230, 237, 229, 237, 228, 235, 243, 235, 242, 234, 241, 233, 240},
       {232, 223, 231, 222, 229, 221, 229, 220, 227, 235, 227, 234, 226, 233, 225, 232},
       {96, 215, 95, 214, 93, 213, 93, 212, 91, 227, 91, 226, 90, 225, 89, 224},
       {88, 207, 87, 206, 85, 205, 85, 204, 83, 219, 83, 218, 82, 217, 81, 216},
       {80, 199, 79, 198, 77, 197, 77, 196, 75, 211, 75, 210, 74, 209, 73, 208},
       {72, 191, 71, 190, 69, 189, 69, 188, 67, 203, 67, 202, 66, 201, 65, 200}},
      {{0xf780, 0xf67f, 0xf57d, 0xf47d, 0x037b, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed75, 0xec75, 0xfb73, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56d, 0xe46d, 0xf36b, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd65, 0xdc65, 0xeb63, 0xea63, 0xe962, 0xe861},
       {0xd760, 0xd65f, 0xd55d, 0xd45d, 0xe35b, 0xe25b, 0xe15a, 0xe059},
       {0xcf58, 0xce57, 0xcd55, 0xcc55, 0xdb53, 0xda53, 0xd952, 0xd851},
       {0xc750, 0xc64f, 0xc54d, 0xc44d, 0xd34b, 0xd24b, 0xd14a, 0xd049},
       {0xbf48, 0xbe47, 0xbd45, 0xbc45, 0xcb43, 0xca43, 0xc942, 0xc841}},
      {{0xf67e'f780, 0xf47c'f57d, 0x027b'037b, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed75, 0xfa72'fb73, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56d, 0xf26a'f36b, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd65, 0xea62'eb63, 0xe860'e962},
       {0xd65f'5760, 0xd45d'555d, 0xe25b'635b, 0xe059'615a},
       {0xce57'4f58, 0xcc55'4d55, 0xda53'5b53, 0xd851'5952},
       {0xc64f'4750, 0xc44d'454d, 0xd24b'534b, 0xd049'514a},
       {0xbe47'3f48, 0xbc45'3d45, 0xca43'4b43, 0xc841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037b},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb73},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36b},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb63},
       {0xd45d'555d'd65f'5760, 0xe059'615a'625b'635b},
       {0xcc55'4d55'ce57'4f58, 0xd851'5952'5a53'5b53},
       {0xc44d'454d'c64f'4750, 0xd049'514a'524b'534b},
       {0xbc45'3d45'be47'3f48, 0xc841'4942'4a43'4b43}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVasubvx,
                        ExecMaskedVasubvx,
                        {{43, 235, 44, 236, 45, 237, 46, 238, 47, 239, 48, 240, 49, 241, 50, 242},
                         {51, 243, 52, 244, 53, 245, 54, 246, 55, 247, 56, 248, 57, 249, 58, 250},
                         {59, 251, 60, 252, 61, 253, 62, 254, 63, 255, 64, 0, 65, 1, 66, 2},
                         {67, 3, 68, 4, 69, 5, 70, 6, 71, 7, 72, 8, 73, 9, 74, 10},
                         {75, 11, 76, 12, 77, 13, 78, 14, 79, 15, 80, 16, 81, 17, 82, 18},
                         {83, 19, 84, 20, 85, 21, 86, 22, 87, 23, 88, 24, 89, 25, 90, 26},
                         {91, 27, 92, 28, 93, 29, 94, 30, 95, 31, 96, 32, 97, 33, 98, 34},
                         {99, 35, 100, 36, 101, 37, 102, 38, 103, 39, 104, 40, 105, 41, 106, 42}},
                        {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
                         {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
                         {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
                         {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
                         {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
                         {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
                         {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
                         {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
                        {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
                         {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
                         {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
                         {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
                         {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
                         {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
                         {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
                         {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
                        {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
                         {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
                         {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
                         {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
                         {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
                         {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
                         {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
                         {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
                        kVectorCalculationsSource);
  TestNarrowingVectorInstruction(ExecVnclipuwi,
                                 ExecMaskedVnclipuwi,
                                 {{32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39},
                                  {40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47},
                                  {48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55},
                                  {56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                                 kVectorCalculationsSource);
  TestNarrowingVectorInstruction(
      ExecVnclipwi,
      ExecMaskedVnclipwi,
      {{224, 224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231},
       {232, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239},
       {240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247},
       {248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(ExecVnclipuwx,
                                 ExecMaskedVnclipuwx,
                                 {{32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39},
                                  {40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47},
                                  {48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55},
                                  {56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0x0021'c1a1, 0x0023'c3a3, 0x0025'c5a5, 0x0027'c7a7},
                                  {0x0029'c9a9, 0x002b'cbab, 0x002d'cdad, 0x002f'cfaf},
                                  {0x0031'd1b1, 0x0033'd3b3, 0x0035'd5b5, 0x0037'd7b7},
                                  {0x0039'd9b9, 0x003b'dbbb, 0x003d'ddbd, 0x003f'dfbf}},
                                 kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwx,
      ExecMaskedVnclipwx,
      {{224, 224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231},
       {232, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239},
       {240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247},
       {248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0xffe1'c1a1, 0xffe3'c3a3, 0xffe5'c5a5, 0xffe7'c7a7},
       {0xffe9'c9a9, 0xffeb'cbab, 0xffed'cdad, 0xffef'cfaf},
       {0xfff1'd1b1, 0xfff3'd3b3, 0xfff5'd5b5, 0xfff7'd7b7},
       {0xfff9'd9b9, 0xfffb'dbbb, 0xfffd'ddbd, 0xffff'dfbf}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipuwv,
      ExecMaskedVnclipuwv,
      {{255, 255, 255, 255, 68, 34, 8, 2, 255, 255, 255, 255, 153, 38, 9, 2},
       {255, 255, 255, 255, 84, 42, 10, 2, 255, 255, 255, 255, 185, 46, 11, 2},
       {255, 255, 255, 255, 100, 50, 12, 3, 255, 255, 255, 255, 217, 54, 13, 3},
       {255, 255, 255, 255, 116, 58, 14, 3, 255, 255, 255, 255, 249, 62, 15, 3}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0x4989, 0x0971, 0x009b, 0x0009},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x5999, 0x0b73, 0x00bb, 0x000b},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x69a9, 0x0d75, 0x00db, 0x000d},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x79b9, 0x0f77, 0x00fb, 0x000f}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xa726'a524, 0x0057'9756, 0x0000'5b9b, 0x0000'00bf},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xe766'e564, 0x0077'b776, 0x0000'7bbb, 0x0000'00ff}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwv,
      ExecMaskedVnclipwv,
      {{128, 128, 128, 128, 196, 226, 248, 254, 128, 128, 128, 128, 153, 230, 249, 254},
       {128, 128, 128, 128, 212, 234, 250, 254, 128, 128, 128, 128, 185, 238, 251, 254},
       {128, 128, 128, 128, 228, 242, 252, 255, 128, 128, 128, 128, 217, 246, 253, 255},
       {128, 128, 128, 157, 244, 250, 254, 255, 128, 128, 128, 221, 249, 254, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0xc989, 0xf971, 0xff9b, 0xfff9},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xd999, 0xfb73, 0xffbb, 0xfffb},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xe9a9, 0xfd75, 0xffdb, 0xfffd},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xf9b9, 0xff77, 0xfffb, 0xffff}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xa726'a524, 0xffd7'9756, 0xffff'db9b, 0xffff'ffbf},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xe766'e564, 0xfff7'b776, 0xffff'fbbb, 0xffff'ffff}},
      kVectorCalculationsSource);

  TestVectorInstruction(
      ExecVsmulvv,
      ExecMaskedVsmulvv,
      {{0, 109, 0, 103, 0, 98, 0, 92, 1, 117, 1, 111, 2, 106, 3, 100},
       {4, 67, 5, 63, 6, 58, 7, 54, 9, 75, 10, 71, 12, 66, 14, 62},
       {16, 34, 18, 30, 20, 27, 22, 23, 25, 42, 27, 38, 30, 35, 33, 31},
       {36, 8, 39, 6, 42, 3, 45, 1, 49, 16, 52, 14, 56, 11, 60, 9},
       {192, 247, 192, 245, 192, 244, 192, 242, 193, 255, 193, 253, 194, 252, 195, 250},
       {196, 237, 197, 237, 198, 236, 199, 236, 201, 245, 202, 245, 204, 244, 206, 244},
       {208, 236, 210, 236, 213, 237, 214, 237, 217, 244, 219, 244, 222, 245, 225, 245},
       {228, 242, 231, 244, 235, 245, 237, 247, 241, 250, 244, 252, 248, 253, 252, 255}},
      {{0x6d24, 0x677e, 0x61f8, 0x5c94, 0x750c, 0x6f68, 0x69e3, 0x647e},
       {0x437e, 0x3eda, 0x3a56, 0x35f4, 0x4b6a, 0x46c8, 0x4245, 0x3de3},
       {0x21e9, 0x1e47, 0x1ac5, 0x1765, 0x29d9, 0x2639, 0x22b8, 0x1f57},
       {0x0863, 0x05c4, 0x0344, 0x00e5, 0x1058, 0x0db9, 0x0b3b, 0x08dc},
       {0xf6ee, 0xf550, 0xf3d2, 0xf276, 0xfee7, 0xfd4a, 0xfbcd, 0xfa71},
       {0xed88, 0xeced, 0xec71, 0xec17, 0xf585, 0xf4eb, 0xf470, 0xf415},
       {0xec33, 0xec9a, 0xed20, 0xedc7, 0xf434, 0xf49b, 0xf523, 0xf5ca},
       {0xf2ee, 0xf456, 0xf5df, 0xf788, 0xfaf3, 0xfc5c, 0xfde5, 0xff8f}},
      {{0x677d'76ae, 0x5c93'1930, 0x6f67'3830, 0x647d'dbb6},
       {0x3eda'09c6, 0x35f3'b250, 0x46c7'cf50, 0x3de2'78dd},
       {0x1e46'b4fd, 0x1764'638f, 0x2638'7e8f, 0x1f57'2e25},
       {0x05c3'7854, 0x00e5'2cef, 0x0db9'45ef, 0x08db'fb8c},
       {0xf550'cd46, 0xf276'7fe1, 0xfd4a'8ed9, 0xfa71'4276},
       {0xeced'a0be, 0xec17'5961, 0xf4eb'6659, 0xf416'1ffe},
       {0xec9a'8c56, 0xedc8'4b00, 0xf49c'55f8, 0xf5cb'15a6},
       {0xf457'900d, 0xf789'54c0, 0xfc5d'5db8, 0xff90'236d}},
      {{0x5c93'192f'ccd4'7781, 0x647d'dbb5'bb66'23af},
       {0x35f3'b24f'43d0'aa38, 0x3de2'78dd'1a4e'4256},
       {0x1764'638e'e2fd'152f, 0x1f57'2e24'a166'993d},
       {0x00e5'2cee'aa59'b866, 0x08db'fb8c'50af'2864},
       {0xf276'7fe1'80cf'f441, 0xfa71'4276'eef1'1fff},
       {0xec17'5961'584c'a798, 0xf416'1ffe'ae59'bf46},
       {0xedc8'4b01'57f9'9330, 0xf5cb'15a6'95f2'96ce},
       {0xf789'54c1'7fd6'b708, 0xff90'236e'a5bb'a696}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVsmulvx,
                        ExecMaskedVsmulvx,
                        {{0, 85, 254, 83, 253, 82, 251, 81, 250, 79, 249, 78, 247, 77, 246, 75},
                         {245, 74, 243, 73, 242, 71, 241, 70, 239, 69, 238, 67, 237, 66, 235, 65},
                         {234, 63, 233, 62, 231, 61, 230, 59, 229, 58, 227, 57, 226, 55, 225, 54},
                         {223, 53, 222, 51, 221, 50, 219, 49, 218, 47, 217, 46, 215, 45, 214, 43},
                         {213, 42, 211, 40, 210, 39, 208, 38, 207, 36, 206, 35, 204, 34, 203, 32},
                         {202, 31, 200, 30, 199, 28, 198, 27, 196, 26, 195, 24, 194, 23, 192, 22},
                         {191, 20, 190, 19, 188, 18, 187, 16, 186, 15, 184, 14, 183, 12, 182, 11},
                         {180, 10, 179, 8, 178, 7, 176, 6, 175, 4, 174, 3, 172, 2, 171, 0}},
                        {{0x54ab, 0x5354, 0x51fd, 0x50a7, 0x4f50, 0x4df9, 0x4ca3, 0x4b4c},
                         {0x49f5, 0x489f, 0x4748, 0x45f1, 0x449b, 0x4344, 0x41ed, 0x4097},
                         {0x3f40, 0x3de9, 0x3c93, 0x3b3c, 0x39e5, 0x388f, 0x3738, 0x35e1},
                         {0x348b, 0x3334, 0x31dd, 0x3087, 0x2f30, 0x2dd9, 0x2c83, 0x2b2c},
                         {0x29d5, 0x287e, 0x2728, 0x25d1, 0x247a, 0x2324, 0x21cd, 0x2076},
                         {0x1f20, 0x1dc9, 0x1c72, 0x1b1c, 0x19c5, 0x186e, 0x1718, 0x15c1},
                         {0x146a, 0x1314, 0x11bd, 0x1066, 0x0f10, 0x0db9, 0x0c62, 0x0b0c},
                         {0x09b5, 0x085e, 0x0708, 0x05b1, 0x045a, 0x0304, 0x01ad, 0x0056}},
                        {{0x5353'aa00, 0x50a6'51fd, 0x4df8'f9fb, 0x4b4b'a1f8},
                         {0x489e'49f5, 0x45f0'f1f3, 0x4343'99f0, 0x4096'41ed},
                         {0x3de8'e9eb, 0x3b3b'91e8, 0x388e'39e5, 0x35e0'e1e3},
                         {0x3333'89e0, 0x3086'31dd, 0x2dd8'd9db, 0x2b2b'81d8},
                         {0x287e'29d5, 0x25d0'd1d2, 0x2323'79d0, 0x2076'21cd},
                         {0x1dc8'c9ca, 0x1b1b'71c8, 0x186e'19c5, 0x15c0'c1c2},
                         {0x1313'69c0, 0x1066'11bd, 0x0db8'b9ba, 0x0b0b'61b8},
                         {0x085e'09b5, 0x05b0'b1b2, 0x0303'59b0, 0x0056'01ad}},
                        {{0x50a6'51fc'fdfe'54ab, 0x4b4b'a1f7'a34e'4f50},
                         {0x45f0'f1f2'489e'49f5, 0x4096'41ec'edee'449b},
                         {0x3b3b'91e7'933e'3f40, 0x35e0'e1e2'388e'39e5},
                         {0x3086'31dc'ddde'348b, 0x2b2b'81d7'832e'2f30},
                         {0x25d0'd1d2'287e'29d5, 0x2076'21cc'cdce'247a},
                         {0x1b1b'71c7'731e'1f20, 0x15c0'c1c2'186e'19c5},
                         {0x1066'11bc'bdbe'146a, 0x0b0b'61b7'630e'0f10},
                         {0x05b0'b1b2'085e'09b5, 0x0056'01ac'adae'045a}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvv,
                        ExecMaskedVssrlvv,
                        {{0, 32, 0, 2, 2, 33, 0, 2, 4, 34, 0, 2, 12, 35, 0, 2},
                         {16, 36, 1, 2, 10, 37, 1, 2, 12, 38, 1, 2, 28, 39, 1, 2},
                         {32, 40, 2, 2, 18, 41, 2, 2, 20, 42, 2, 2, 44, 43, 2, 2},
                         {48, 44, 3, 2, 26, 45, 3, 2, 28, 46, 3, 2, 60, 47, 3, 2},
                         {64, 48, 4, 3, 34, 49, 4, 3, 36, 50, 4, 3, 76, 51, 4, 3},
                         {80, 52, 5, 3, 42, 53, 5, 3, 44, 54, 5, 3, 92, 55, 5, 3},
                         {96, 56, 6, 3, 50, 57, 6, 3, 52, 58, 6, 3, 108, 59, 6, 3},
                         {112, 60, 7, 3, 58, 61, 7, 3, 60, 62, 7, 3, 124, 63, 7, 3}},
                        {{0x8100, 0x0830, 0x0042, 0x0008, 0x4484, 0x08b0, 0x008d, 0x0008},
                         {0x9110, 0x0931, 0x004a, 0x0009, 0x4c8c, 0x09b1, 0x009d, 0x0009},
                         {0xa120, 0x0a32, 0x0052, 0x000a, 0x5494, 0x0ab2, 0x00ad, 0x000a},
                         {0xb130, 0x0b33, 0x005a, 0x000b, 0x5c9c, 0x0bb3, 0x00bd, 0x000b},
                         {0xc140, 0x0c34, 0x0062, 0x000c, 0x64a4, 0x0cb4, 0x00cd, 0x000c},
                         {0xd150, 0x0d35, 0x006a, 0x000d, 0x6cac, 0x0db5, 0x00dd, 0x000d},
                         {0xe160, 0x0e36, 0x0072, 0x000e, 0x74b4, 0x0eb6, 0x00ed, 0x000e},
                         {0xf170, 0x0f37, 0x007a, 0x000f, 0x7cbc, 0x0fb7, 0x00fd, 0x000f}},
                        {{0x8302'8100, 0x0043'8342, 0x0000'4585, 0x0000'008f},
                         {0x9312'9110, 0x004b'8b4a, 0x0000'4d8d, 0x0000'009f},
                         {0xa322'a120, 0x0053'9352, 0x0000'5595, 0x0000'00af},
                         {0xb332'b130, 0x005b'9b5a, 0x0000'5d9d, 0x0000'00bf},
                         {0xc342'c140, 0x0063'a362, 0x0000'65a5, 0x0000'00cf},
                         {0xd352'd150, 0x006b'ab6a, 0x0000'6dad, 0x0000'00df},
                         {0xe362'e160, 0x0073'b372, 0x0000'75b5, 0x0000'00ef},
                         {0xf372'f170, 0x007b'bb7a, 0x0000'7dbd, 0x0000'00ff}},
                        {{0x8706'8504'8302'8100, 0x0000'4787'4686'4585},
                         {0x0000'0000'9716'9514, 0x0000'0000'0000'4f8f},
                         {0xa726'a524'a322'a120, 0x0000'5797'5696'5595},
                         {0x0000'0000'b736'b534, 0x0000'0000'0000'5f9f},
                         {0xc746'c544'c342'c140, 0x0000'67a7'66a6'65a5},
                         {0x0000'0000'd756'd554, 0x0000'0000'0000'6faf},
                         {0xe766'e564'e362'e160, 0x0000'77b7'76b6'75b5},
                         {0x0000'0000'f776'f574, 0x0000'0000'0000'7fbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvx,
                        ExecMaskedVssrlvx,
                        {{0, 32, 0, 32, 1, 33, 1, 33, 2, 34, 2, 34, 3, 35, 3, 35},
                         {4, 36, 4, 36, 5, 37, 5, 37, 6, 38, 6, 38, 7, 39, 7, 39},
                         {8, 40, 8, 40, 9, 41, 9, 41, 10, 42, 10, 42, 11, 43, 11, 43},
                         {12, 44, 12, 44, 13, 45, 13, 45, 14, 46, 14, 46, 15, 47, 15, 47},
                         {16, 48, 16, 48, 17, 49, 17, 49, 18, 50, 18, 50, 19, 51, 19, 51},
                         {20, 52, 20, 52, 21, 53, 21, 53, 22, 54, 22, 54, 23, 55, 23, 55},
                         {24, 56, 24, 56, 25, 57, 25, 57, 26, 58, 26, 58, 27, 59, 27, 59},
                         {28, 60, 28, 60, 29, 61, 29, 61, 30, 62, 30, 62, 31, 63, 31, 63}},
                        {{0x0020, 0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023},
                         {0x0024, 0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027},
                         {0x0028, 0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b},
                         {0x002c, 0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f},
                         {0x0030, 0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033},
                         {0x0034, 0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037},
                         {0x0038, 0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b},
                         {0x003c, 0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0000'0000'0021'c1a1, 0x0000'0000'0023'c3a3},
                         {0x0000'0000'0025'c5a5, 0x0000'0000'0027'c7a7},
                         {0x0000'0000'0029'c9a9, 0x0000'0000'002b'cbab},
                         {0x0000'0000'002d'cdad, 0x0000'0000'002f'cfaf},
                         {0x0000'0000'0031'd1b1, 0x0000'0000'0033'd3b3},
                         {0x0000'0000'0035'd5b5, 0x0000'0000'0037'd7b7},
                         {0x0000'0000'0039'd9b9, 0x0000'0000'003b'dbbb},
                         {0x0000'0000'003d'ddbd, 0x0000'0000'003f'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvi,
                        ExecMaskedVssrlvi,
                        {{0, 32, 0, 32, 1, 33, 1, 33, 2, 34, 2, 34, 3, 35, 3, 35},
                         {4, 36, 4, 36, 5, 37, 5, 37, 6, 38, 6, 38, 7, 39, 7, 39},
                         {8, 40, 8, 40, 9, 41, 9, 41, 10, 42, 10, 42, 11, 43, 11, 43},
                         {12, 44, 12, 44, 13, 45, 13, 45, 14, 46, 14, 46, 15, 47, 15, 47},
                         {16, 48, 16, 48, 17, 49, 17, 49, 18, 50, 18, 50, 19, 51, 19, 51},
                         {20, 52, 20, 52, 21, 53, 21, 53, 22, 54, 22, 54, 23, 55, 23, 55},
                         {24, 56, 24, 56, 25, 57, 25, 57, 26, 58, 26, 58, 27, 59, 27, 59},
                         {28, 60, 28, 60, 29, 61, 29, 61, 30, 62, 30, 62, 31, 63, 31, 63}},
                        {{0x0020, 0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023},
                         {0x0024, 0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027},
                         {0x0028, 0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b},
                         {0x002c, 0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f},
                         {0x0030, 0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033},
                         {0x0034, 0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037},
                         {0x0038, 0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b},
                         {0x003c, 0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0021'c1a1'4120'c0a0, 0x0023'c3a3'4322'c2a2},
                         {0x0025'c5a5'4524'c4a4, 0x0027'c7a7'4726'c6a6},
                         {0x0029'c9a9'4928'c8a8, 0x002b'cbab'4b2a'caaa},
                         {0x002d'cdad'4d2c'ccac, 0x002f'cfaf'4f2e'ceae},
                         {0x0031'd1b1'5130'd0b0, 0x0033'd3b3'5332'd2b2},
                         {0x0035'd5b5'5534'd4b4, 0x0037'd7b7'5736'd6b6},
                         {0x0039'd9b9'5938'd8b8, 0x003b'dbbb'5b3a'daba},
                         {0x003d'ddbd'5d3c'dcbc, 0x003f'dfbf'5f3e'debe}},
                        kVectorCalculationsSource);
  TestVectorInstruction(ExecVssravv,
                        ExecMaskedVssravv,
                        {{0, 224, 0, 254, 2, 225, 0, 254, 4, 226, 0, 254, 12, 227, 0, 254},
                         {16, 228, 1, 254, 10, 229, 1, 254, 12, 230, 1, 254, 28, 231, 1, 254},
                         {32, 232, 2, 254, 18, 233, 2, 254, 20, 234, 2, 254, 44, 235, 2, 254},
                         {48, 236, 3, 254, 26, 237, 3, 254, 28, 238, 3, 254, 60, 239, 3, 254},
                         {64, 240, 4, 255, 34, 241, 4, 255, 36, 242, 4, 255, 76, 243, 4, 255},
                         {80, 244, 5, 255, 42, 245, 5, 255, 44, 246, 5, 255, 92, 247, 5, 255},
                         {96, 248, 6, 255, 50, 249, 6, 255, 52, 250, 6, 255, 108, 251, 6, 255},
                         {112, 252, 7, 255, 58, 253, 7, 255, 60, 254, 7, 255, 124, 255, 7, 255}},
                        {{0x8100, 0xf830, 0xffc2, 0xfff8, 0xc484, 0xf8b0, 0xff8d, 0xfff8},
                         {0x9110, 0xf931, 0xffca, 0xfff9, 0xcc8c, 0xf9b1, 0xff9d, 0xfff9},
                         {0xa120, 0xfa32, 0xffd2, 0xfffa, 0xd494, 0xfab2, 0xffad, 0xfffa},
                         {0xb130, 0xfb33, 0xffda, 0xfffb, 0xdc9c, 0xfbb3, 0xffbd, 0xfffb},
                         {0xc140, 0xfc34, 0xffe2, 0xfffc, 0xe4a4, 0xfcb4, 0xffcd, 0xfffc},
                         {0xd150, 0xfd35, 0xffea, 0xfffd, 0xecac, 0xfdb5, 0xffdd, 0xfffd},
                         {0xe160, 0xfe36, 0xfff2, 0xfffe, 0xf4b4, 0xfeb6, 0xffed, 0xfffe},
                         {0xf170, 0xff37, 0xfffa, 0xffff, 0xfcbc, 0xffb7, 0xfffd, 0xffff}},
                        {{0x8302'8100, 0xffc3'8342, 0xffff'c585, 0xffff'ff8f},
                         {0x9312'9110, 0xffcb'8b4a, 0xffff'cd8d, 0xffff'ff9f},
                         {0xa322'a120, 0xffd3'9352, 0xffff'd595, 0xffff'ffaf},
                         {0xb332'b130, 0xffdb'9b5a, 0xffff'dd9d, 0xffff'ffbf},
                         {0xc342'c140, 0xffe3'a362, 0xffff'e5a5, 0xffff'ffcf},
                         {0xd352'd150, 0xffeb'ab6a, 0xffff'edad, 0xffff'ffdf},
                         {0xe362'e160, 0xfff3'b372, 0xffff'f5b5, 0xffff'ffef},
                         {0xf372'f170, 0xfffb'bb7a, 0xffff'fdbd, 0xffff'ffff}},
                        {{0x8706'8504'8302'8100, 0xffff'c787'4686'4585},
                         {0xffff'ffff'9716'9514, 0xffff'ffff'ffff'cf8f},
                         {0xa726'a524'a322'a120, 0xffff'd797'5696'5595},
                         {0xffff'ffff'b736'b534, 0xffff'ffff'ffff'df9f},
                         {0xc746'c544'c342'c140, 0xffff'e7a7'66a6'65a5},
                         {0xffff'ffff'd756'd554, 0xffff'ffff'ffff'efaf},
                         {0xe766'e564'e362'e160, 0xffff'f7b7'76b6'75b5},
                         {0xffff'ffff'f776'f574, 0xffff'ffff'ffff'ffbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravx,
                        ExecMaskedVssravx,
                        {{0, 224, 0, 224, 1, 225, 1, 225, 2, 226, 2, 226, 3, 227, 3, 227},
                         {4, 228, 4, 228, 5, 229, 5, 229, 6, 230, 6, 230, 7, 231, 7, 231},
                         {8, 232, 8, 232, 9, 233, 9, 233, 10, 234, 10, 234, 11, 235, 11, 235},
                         {12, 236, 12, 236, 13, 237, 13, 237, 14, 238, 14, 238, 15, 239, 15, 239},
                         {16, 240, 16, 240, 17, 241, 17, 241, 18, 242, 18, 242, 19, 243, 19, 243},
                         {20, 244, 20, 244, 21, 245, 21, 245, 22, 246, 22, 246, 23, 247, 23, 247},
                         {24, 248, 24, 248, 25, 249, 25, 249, 26, 250, 26, 250, 27, 251, 27, 251},
                         {28, 252, 28, 252, 29, 253, 29, 253, 30, 254, 30, 254, 31, 255, 31, 255}},
                        {{0xffe0, 0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3},
                         {0xffe4, 0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7},
                         {0xffe8, 0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb},
                         {0xffec, 0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef},
                         {0xfff0, 0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3},
                         {0xfff4, 0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7},
                         {0xfff8, 0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb},
                         {0xfffc, 0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffff'ffff'ffe1'c1a1, 0xffff'ffff'ffe3'c3a3},
                         {0xffff'ffff'ffe5'c5a5, 0xffff'ffff'ffe7'c7a7},
                         {0xffff'ffff'ffe9'c9a9, 0xffff'ffff'ffeb'cbab},
                         {0xffff'ffff'ffed'cdad, 0xffff'ffff'ffef'cfaf},
                         {0xffff'ffff'fff1'd1b1, 0xffff'ffff'fff3'd3b3},
                         {0xffff'ffff'fff5'd5b5, 0xffff'ffff'fff7'd7b7},
                         {0xffff'ffff'fff9'd9b9, 0xffff'ffff'fffb'dbbb},
                         {0xffff'ffff'fffd'ddbd, 0xffff'ffff'ffff'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravi,
                        ExecMaskedVssravi,
                        {{0, 224, 0, 224, 1, 225, 1, 225, 2, 226, 2, 226, 3, 227, 3, 227},
                         {4, 228, 4, 228, 5, 229, 5, 229, 6, 230, 6, 230, 7, 231, 7, 231},
                         {8, 232, 8, 232, 9, 233, 9, 233, 10, 234, 10, 234, 11, 235, 11, 235},
                         {12, 236, 12, 236, 13, 237, 13, 237, 14, 238, 14, 238, 15, 239, 15, 239},
                         {16, 240, 16, 240, 17, 241, 17, 241, 18, 242, 18, 242, 19, 243, 19, 243},
                         {20, 244, 20, 244, 21, 245, 21, 245, 22, 246, 22, 246, 23, 247, 23, 247},
                         {24, 248, 24, 248, 25, 249, 25, 249, 26, 250, 26, 250, 27, 251, 27, 251},
                         {28, 252, 28, 252, 29, 253, 29, 253, 30, 254, 30, 254, 31, 255, 31, 255}},
                        {{0xffe0, 0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3},
                         {0xffe4, 0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7},
                         {0xffe8, 0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb},
                         {0xffec, 0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef},
                         {0xfff0, 0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3},
                         {0xfff4, 0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7},
                         {0xfff8, 0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb},
                         {0xfffc, 0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffe1'c1a1'4120'c0a0, 0xffe3'c3a3'4322'c2a2},
                         {0xffe5'c5a5'4524'c4a4, 0xffe7'c7a7'4726'c6a6},
                         {0xffe9'c9a9'4928'c8a8, 0xffeb'cbab'4b2a'caaa},
                         {0xffed'cdad'4d2c'ccac, 0xffef'cfaf'4f2e'ceae},
                         {0xfff1'd1b1'5130'd0b0, 0xfff3'd3b3'5332'd2b2},
                         {0xfff5'd5b5'5534'd4b4, 0xfff7'd7b7'5736'd6b6},
                         {0xfff9'd9b9'5938'd8b8, 0xfffb'dbbb'5b3a'daba},
                         {0xfffd'ddbd'5d3c'dcbc, 0xffff'dfbf'5f3e'debe}},
                        kVectorCalculationsSource);
  asm volatile("csrw vxrm, %0\n\t" ::"r"(vxrm));
}

TEST(InlineAsmTestRiscv64, TestRNE) {
  uint64_t vxrm;
  asm volatile(
      "csrr %0, vxrm\n\t"
      "csrwi vxrm, %c1\n\t"
      : "=r"(vxrm)
      : "i"(VXRMFlags::RNE));
  TestVectorInstruction(
      ExecVaadduvv,
      ExecMaskedVaadduvv,
      {{0, 138, 3, 140, 6, 144, 9, 146, 12, 134, 15, 136, 18, 140, 21, 142},
       {24, 162, 27, 164, 30, 168, 33, 170, 36, 158, 39, 160, 42, 164, 45, 166},
       {48, 186, 51, 188, 54, 192, 57, 194, 60, 182, 63, 184, 66, 188, 69, 190},
       {72, 210, 75, 212, 78, 216, 81, 218, 84, 206, 87, 208, 90, 212, 93, 214},
       {96, 106, 99, 108, 102, 112, 105, 114, 108, 102, 111, 104, 114, 108, 117, 110},
       {120, 130, 123, 132, 126, 136, 129, 138, 132, 126, 135, 128, 138, 132, 141, 134},
       {144, 154, 147, 156, 150, 160, 153, 162, 156, 150, 159, 152, 162, 156, 165, 158},
       {168, 178, 171, 180, 174, 184, 177, 186, 180, 174, 183, 176, 186, 180, 189, 182}},
      {{0x8980, 0x8c83, 0x8f86, 0x9289, 0x858c, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79e, 0xaaa1, 0x9da4, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb6, 0xc2b9, 0xb5bc, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7ce, 0xdad1, 0xcdd4, 0xd0d7, 0xd3da, 0xd6dd},
       {0x69e0, 0x6ce3, 0x6fe6, 0x72e9, 0x65ec, 0x68ef, 0x6bf2, 0x6ef5},
       {0x81f8, 0x84fb, 0x87fe, 0x8b01, 0x7e04, 0x8107, 0x840a, 0x870d},
       {0x9a10, 0x9d13, 0xa016, 0xa319, 0x961c, 0x991f, 0x9c22, 0x9f25},
       {0xb228, 0xb52b, 0xb82e, 0xbb31, 0xae34, 0xb137, 0xb43a, 0xb73d}},
      {{0x8c83'8980, 0x9289'8f86, 0x888f'858c, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79e, 0xa0a7'9da4, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb6, 0xb8bf'b5bc, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7ce, 0xd0d7'cdd4, 0xd6dd'd3da},
       {0x6ce3'69e0, 0x72e9'6fe6, 0x68ef'65ec, 0x6ef5'6bf2},
       {0x84fb'81f8, 0x8b01'87fe, 0x8107'7e04, 0x870d'840a},
       {0x9d13'9a10, 0xa319'a016, 0x991f'961c, 0x9f25'9c22},
       {0xb52b'b228, 0xbb31'b82e, 0xb137'ae34, 0xb73d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858c},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da4},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bc},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd4},
       {0x72e9'6fe6'ece3'69e0, 0x6ef5'6bf2'68ef'65ec},
       {0x8b01'87ff'04fb'81f8, 0x870d'840a'8107'7e04},
       {0xa319'a017'1d13'9a10, 0x9f25'9c22'991f'961c},
       {0xbb31'b82f'352b'b228, 0xb73d'b43a'b137'ae34}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaadduvx,
      ExecMaskedVaadduvx,
      {{85, 150, 86, 150, 87, 152, 88, 152, 89, 154, 90, 154, 91, 156, 92, 156},
       {93, 158, 94, 158, 95, 160, 96, 160, 97, 162, 98, 162, 99, 164, 100, 164},
       {101, 166, 102, 166, 103, 168, 104, 168, 105, 170, 106, 170, 107, 172, 108, 172},
       {109, 174, 110, 174, 111, 176, 112, 176, 113, 178, 114, 178, 115, 180, 116, 180},
       {117, 182, 118, 182, 119, 184, 120, 184, 121, 186, 122, 186, 123, 188, 124, 188},
       {125, 190, 126, 190, 127, 192, 128, 192, 129, 194, 130, 194, 131, 196, 132, 196},
       {133, 198, 134, 198, 135, 200, 136, 200, 137, 202, 138, 202, 139, 204, 140, 204},
       {141, 206, 142, 206, 143, 208, 144, 208, 145, 210, 146, 210, 147, 212, 148, 212}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvv,
      ExecMaskedVaaddvv,
      {{0, 138, 3, 140, 6, 144, 9, 146, 12, 134, 15, 136, 18, 140, 21, 142},
       {24, 162, 27, 164, 30, 168, 33, 170, 36, 158, 39, 160, 42, 164, 45, 166},
       {48, 186, 51, 188, 54, 192, 57, 194, 60, 182, 63, 184, 66, 188, 69, 190},
       {72, 210, 75, 212, 78, 216, 81, 218, 84, 206, 87, 208, 90, 212, 93, 214},
       {224, 234, 227, 236, 230, 240, 233, 242, 236, 230, 239, 232, 242, 236, 245, 238},
       {248, 2, 251, 4, 254, 8, 1, 10, 4, 254, 7, 0, 10, 4, 13, 6},
       {16, 26, 19, 28, 22, 32, 25, 34, 28, 22, 31, 24, 34, 28, 37, 30},
       {40, 50, 43, 52, 46, 56, 49, 58, 52, 46, 55, 48, 58, 52, 61, 54}},
      {{0x8980, 0x8c83, 0x8f86, 0x9289, 0x858c, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79e, 0xaaa1, 0x9da4, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb6, 0xc2b9, 0xb5bc, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7ce, 0xdad1, 0xcdd4, 0xd0d7, 0xd3da, 0xd6dd},
       {0xe9e0, 0xece3, 0xefe6, 0xf2e9, 0xe5ec, 0xe8ef, 0xebf2, 0xeef5},
       {0x01f8, 0x04fb, 0x07fe, 0x0b01, 0xfe04, 0x0107, 0x040a, 0x070d},
       {0x1a10, 0x1d13, 0x2016, 0x2319, 0x161c, 0x191f, 0x1c22, 0x1f25},
       {0x3228, 0x352b, 0x382e, 0x3b31, 0x2e34, 0x3137, 0x343a, 0x373d}},
      {{0x8c83'8980, 0x9289'8f86, 0x888f'858c, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79e, 0xa0a7'9da4, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb6, 0xb8bf'b5bc, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7ce, 0xd0d7'cdd4, 0xd6dd'd3da},
       {0xece3'69e0, 0xf2e9'6fe6, 0xe8ef'65ec, 0xeef5'6bf2},
       {0x04fb'81f8, 0x0b01'87fe, 0x0107'7e04, 0x070d'840a},
       {0x1d13'9a10, 0x2319'a016, 0x191f'961c, 0x1f25'9c22},
       {0x352b'b228, 0x3b31'b82e, 0x3137'ae34, 0x373d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858c},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da4},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bc},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd4},
       {0xf2e9'6fe6'ece3'69e0, 0xeef5'6bf2'68ef'65ec},
       {0x0b01'87ff'04fb'81f8, 0x070d'840a'8107'7e04},
       {0x2319'a017'1d13'9a10, 0x1f25'9c22'991f'961c},
       {0x3b31'b82f'352b'b228, 0x373d'b43a'b137'ae34}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvx,
      ExecMaskedVaaddvx,
      {{213, 150, 214, 150, 215, 152, 216, 152, 217, 154, 218, 154, 219, 156, 220, 156},
       {221, 158, 222, 158, 223, 160, 224, 160, 225, 162, 226, 162, 227, 164, 228, 164},
       {229, 166, 230, 166, 231, 168, 232, 168, 233, 170, 234, 170, 235, 172, 236, 172},
       {237, 174, 238, 174, 239, 176, 240, 176, 241, 178, 242, 178, 243, 180, 244, 180},
       {245, 182, 246, 182, 247, 184, 248, 184, 249, 186, 250, 186, 251, 188, 252, 188},
       {253, 190, 254, 190, 255, 192, 0, 192, 1, 194, 2, 194, 3, 196, 4, 196},
       {5, 198, 6, 198, 7, 200, 8, 200, 9, 202, 10, 202, 11, 204, 12, 204},
       {13, 206, 14, 206, 15, 208, 16, 208, 17, 210, 18, 210, 19, 212, 20, 212}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvv,
      ExecMaskedVasubuvv,
      {{0, 248, 255, 246, 254, 246, 253, 244, 252, 4, 251, 2, 250, 2, 249, 0},
       {248, 240, 247, 238, 246, 238, 245, 236, 244, 252, 243, 250, 242, 250, 241, 248},
       {240, 232, 239, 230, 238, 230, 237, 228, 236, 244, 235, 242, 234, 242, 233, 240},
       {232, 224, 231, 222, 230, 222, 229, 220, 228, 236, 227, 234, 226, 234, 225, 232},
       {224, 88, 223, 86, 222, 86, 221, 84, 220, 100, 219, 98, 218, 98, 217, 96},
       {216, 80, 215, 78, 214, 78, 213, 76, 212, 92, 211, 90, 210, 90, 209, 88},
       {208, 72, 207, 70, 206, 70, 205, 68, 204, 84, 203, 82, 202, 82, 201, 80},
       {200, 64, 199, 62, 198, 62, 197, 60, 196, 76, 195, 74, 194, 74, 193, 72}},
      {{0xf780, 0xf67f, 0xf57e, 0xf47d, 0x037c, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed76, 0xec75, 0xfb74, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56e, 0xe46d, 0xf36c, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd66, 0xdc65, 0xeb64, 0xea63, 0xe962, 0xe861},
       {0x5760, 0x565f, 0x555e, 0x545d, 0x635c, 0x625b, 0x615a, 0x6059},
       {0x4f58, 0x4e57, 0x4d56, 0x4c55, 0x5b54, 0x5a53, 0x5952, 0x5851},
       {0x4750, 0x464f, 0x454e, 0x444d, 0x534c, 0x524b, 0x514a, 0x5049},
       {0x3f48, 0x3e47, 0x3d46, 0x3c45, 0x4b44, 0x4a43, 0x4942, 0x4841}},
      {{0xf67e'f780, 0xf47c'f57e, 0x027b'037c, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed76, 0xfa72'fb74, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56e, 0xf26a'f36c, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd66, 0xea62'eb64, 0xe860'e962},
       {0x565f'5760, 0x545d'555e, 0x625b'635c, 0x6059'615a},
       {0x4e57'4f58, 0x4c55'4d56, 0x5a53'5b54, 0x5851'5952},
       {0x464f'4750, 0x444d'454e, 0x524b'534c, 0x5049'514a},
       {0x3e47'3f48, 0x3c45'3d46, 0x4a43'4b44, 0x4841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037c},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb74},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36c},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb64},
       {0x545d'555d'd65f'5760, 0x6059'615a'625b'635c},
       {0x4c55'4d55'ce57'4f58, 0x5851'5952'5a53'5b54},
       {0x444d'454d'c64f'4750, 0x5049'514a'524b'534c},
       {0x3c45'3d45'be47'3f48, 0x4841'4942'4a43'4b44}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvx,
      ExecMaskedVasubuvx,
      {{171, 236, 172, 236, 173, 238, 174, 238, 175, 240, 176, 240, 177, 242, 178, 242},
       {179, 244, 180, 244, 181, 246, 182, 246, 183, 248, 184, 248, 185, 250, 186, 250},
       {187, 252, 188, 252, 189, 254, 190, 254, 191, 0, 192, 0, 193, 2, 194, 2},
       {195, 4, 196, 4, 197, 6, 198, 6, 199, 8, 200, 8, 201, 10, 202, 10},
       {203, 12, 204, 12, 205, 14, 206, 14, 207, 16, 208, 16, 209, 18, 210, 18},
       {211, 20, 212, 20, 213, 22, 214, 22, 215, 24, 216, 24, 217, 26, 218, 26},
       {219, 28, 220, 28, 221, 30, 222, 30, 223, 32, 224, 32, 225, 34, 226, 34},
       {227, 36, 228, 36, 229, 38, 230, 38, 231, 40, 232, 40, 233, 42, 234, 42}},
      {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
       {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
       {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
       {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
       {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
       {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
       {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
       {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
      {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
       {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
       {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
       {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
       {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
       {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
       {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
       {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
      {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
       {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
       {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
       {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
       {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
       {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
       {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
       {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubvv,
      ExecMaskedVasubvv,
      {{0, 248, 255, 246, 254, 246, 253, 244, 252, 4, 251, 2, 250, 2, 249, 0},
       {248, 240, 247, 238, 246, 238, 245, 236, 244, 252, 243, 250, 242, 250, 241, 248},
       {240, 232, 239, 230, 238, 230, 237, 228, 236, 244, 235, 242, 234, 242, 233, 240},
       {232, 224, 231, 222, 230, 222, 229, 220, 228, 236, 227, 234, 226, 234, 225, 232},
       {96, 216, 95, 214, 94, 214, 93, 212, 92, 228, 91, 226, 90, 226, 89, 224},
       {88, 208, 87, 206, 86, 206, 85, 204, 84, 220, 83, 218, 82, 218, 81, 216},
       {80, 200, 79, 198, 78, 198, 77, 196, 76, 212, 75, 210, 74, 210, 73, 208},
       {72, 192, 71, 190, 70, 190, 69, 188, 68, 204, 67, 202, 66, 202, 65, 200}},
      {{0xf780, 0xf67f, 0xf57e, 0xf47d, 0x037c, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed76, 0xec75, 0xfb74, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56e, 0xe46d, 0xf36c, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd66, 0xdc65, 0xeb64, 0xea63, 0xe962, 0xe861},
       {0xd760, 0xd65f, 0xd55e, 0xd45d, 0xe35c, 0xe25b, 0xe15a, 0xe059},
       {0xcf58, 0xce57, 0xcd56, 0xcc55, 0xdb54, 0xda53, 0xd952, 0xd851},
       {0xc750, 0xc64f, 0xc54e, 0xc44d, 0xd34c, 0xd24b, 0xd14a, 0xd049},
       {0xbf48, 0xbe47, 0xbd46, 0xbc45, 0xcb44, 0xca43, 0xc942, 0xc841}},
      {{0xf67e'f780, 0xf47c'f57e, 0x027b'037c, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed76, 0xfa72'fb74, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56e, 0xf26a'f36c, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd66, 0xea62'eb64, 0xe860'e962},
       {0xd65f'5760, 0xd45d'555e, 0xe25b'635c, 0xe059'615a},
       {0xce57'4f58, 0xcc55'4d56, 0xda53'5b54, 0xd851'5952},
       {0xc64f'4750, 0xc44d'454e, 0xd24b'534c, 0xd049'514a},
       {0xbe47'3f48, 0xbc45'3d46, 0xca43'4b44, 0xc841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037c},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb74},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36c},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb64},
       {0xd45d'555d'd65f'5760, 0xe059'615a'625b'635c},
       {0xcc55'4d55'ce57'4f58, 0xd851'5952'5a53'5b54},
       {0xc44d'454d'c64f'4750, 0xd049'514a'524b'534c},
       {0xbc45'3d45'be47'3f48, 0xc841'4942'4a43'4b44}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVasubvx,
                        ExecMaskedVasubvx,
                        {{43, 236, 44, 236, 45, 238, 46, 238, 47, 240, 48, 240, 49, 242, 50, 242},
                         {51, 244, 52, 244, 53, 246, 54, 246, 55, 248, 56, 248, 57, 250, 58, 250},
                         {59, 252, 60, 252, 61, 254, 62, 254, 63, 0, 64, 0, 65, 2, 66, 2},
                         {67, 4, 68, 4, 69, 6, 70, 6, 71, 8, 72, 8, 73, 10, 74, 10},
                         {75, 12, 76, 12, 77, 14, 78, 14, 79, 16, 80, 16, 81, 18, 82, 18},
                         {83, 20, 84, 20, 85, 22, 86, 22, 87, 24, 88, 24, 89, 26, 90, 26},
                         {91, 28, 92, 28, 93, 30, 94, 30, 95, 32, 96, 32, 97, 34, 98, 34},
                         {99, 36, 100, 36, 101, 38, 102, 38, 103, 40, 104, 40, 105, 42, 106, 42}},
                        {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
                         {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
                         {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
                         {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
                         {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
                         {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
                         {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
                         {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
                        {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
                         {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
                         {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
                         {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
                         {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
                         {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
                         {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
                         {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
                        {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
                         {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
                         {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
                         {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
                         {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
                         {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
                         {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
                         {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
                        kVectorCalculationsSource);
  TestNarrowingVectorInstruction(ExecVnclipuwi,
                                 ExecMaskedVnclipuwi,
                                 {{32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40},
                                  {40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48},
                                  {48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56},
                                  {56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                                 kVectorCalculationsSource);
  TestNarrowingVectorInstruction(
      ExecVnclipwi,
      ExecMaskedVnclipwi,
      {{224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232},
       {232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239, 240},
       {240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 248},
       {248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255, 0}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(ExecVnclipuwx,
                                 ExecMaskedVnclipuwx,
                                 {{32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40},
                                  {40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48},
                                  {48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56},
                                  {56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0x0021'c1a1, 0x0023'c3a3, 0x0025'c5a5, 0x0027'c7a7},
                                  {0x0029'c9a9, 0x002b'cbab, 0x002d'cdad, 0x002f'cfaf},
                                  {0x0031'd1b1, 0x0033'd3b3, 0x0035'd5b5, 0x0037'd7b7},
                                  {0x0039'd9b9, 0x003b'dbbb, 0x003d'ddbd, 0x003f'dfbf}},
                                 kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwx,
      ExecMaskedVnclipwx,
      {{224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232},
       {232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239, 240},
       {240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 248},
       {248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255, 0}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0xffe1'c1a1, 0xffe3'c3a3, 0xffe5'c5a5, 0xffe7'c7a7},
       {0xffe9'c9a9, 0xffeb'cbab, 0xffed'cdad, 0xffef'cfaf},
       {0xfff1'd1b1, 0xfff3'd3b3, 0xfff5'd5b5, 0xfff7'd7b7},
       {0xfff9'd9b9, 0xfffb'dbbb, 0xfffd'ddbd, 0xffff'dfbf}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipuwv,
      ExecMaskedVnclipuwv,
      {{255, 255, 255, 255, 69, 35, 9, 2, 255, 255, 255, 255, 153, 39, 10, 2},
       {255, 255, 255, 255, 85, 43, 11, 3, 255, 255, 255, 255, 185, 47, 12, 3},
       {255, 255, 255, 255, 101, 51, 13, 3, 255, 255, 255, 255, 217, 55, 14, 3},
       {255, 255, 255, 255, 117, 59, 15, 4, 255, 255, 255, 255, 249, 63, 16, 4}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0x4989, 0x0971, 0x009b, 0x000a},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x5999, 0x0b73, 0x00bb, 0x000c},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x69a9, 0x0d75, 0x00db, 0x000e},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x79b9, 0x0f77, 0x00fb, 0x0010}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xa726'a525, 0x0057'9757, 0x0000'5b9b, 0x0000'00bf},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xe766'e565, 0x0077'b777, 0x0000'7bbb, 0x0000'00ff}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwv,
      ExecMaskedVnclipwv,
      {{128, 128, 128, 128, 197, 227, 249, 254, 128, 128, 128, 128, 153, 231, 250, 254},
       {128, 128, 128, 128, 213, 235, 251, 255, 128, 128, 128, 128, 185, 239, 252, 255},
       {128, 128, 128, 128, 229, 243, 253, 255, 128, 128, 128, 128, 217, 247, 254, 255},
       {128, 128, 128, 158, 245, 251, 255, 0, 128, 128, 128, 222, 249, 255, 0, 0}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0xc989, 0xf971, 0xff9b, 0xfffa},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xd999, 0xfb73, 0xffbb, 0xfffc},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xe9a9, 0xfd75, 0xffdb, 0xfffe},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xf9b9, 0xff77, 0xfffb, 0x0000}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xa726'a525, 0xffd7'9757, 0xffff'db9b, 0xffff'ffbf},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xe766'e565, 0xfff7'b777, 0xffff'fbbb, 0xffff'ffff}},
      kVectorCalculationsSource);

  TestVectorInstruction(
      ExecVsmulvv,
      ExecMaskedVsmulvv,
      {{0, 109, 0, 104, 0, 98, 1, 93, 1, 117, 2, 112, 2, 106, 3, 101},
       {4, 68, 5, 63, 6, 59, 8, 54, 9, 76, 11, 71, 12, 67, 14, 62},
       {16, 34, 18, 31, 21, 27, 23, 24, 25, 42, 28, 39, 30, 35, 33, 32},
       {36, 9, 39, 6, 43, 4, 46, 1, 49, 17, 53, 14, 56, 12, 60, 9},
       {192, 247, 192, 246, 193, 244, 193, 243, 194, 255, 194, 254, 194, 252, 195, 251},
       {196, 238, 197, 237, 199, 237, 200, 236, 202, 246, 203, 245, 204, 245, 206, 244},
       {208, 236, 210, 237, 213, 237, 215, 238, 218, 244, 220, 245, 222, 245, 225, 246},
       {228, 243, 231, 244, 235, 246, 238, 247, 242, 251, 245, 252, 248, 254, 252, 255}},
      {{0x6d24, 0x677e, 0x61f8, 0x5c94, 0x750c, 0x6f68, 0x69e3, 0x647f},
       {0x437f, 0x3edb, 0x3a57, 0x35f5, 0x4b6b, 0x46c9, 0x4246, 0x3de3},
       {0x21e9, 0x1e48, 0x1ac5, 0x1765, 0x29da, 0x2639, 0x22b8, 0x1f58},
       {0x0864, 0x05c4, 0x0344, 0x00e6, 0x1058, 0x0dba, 0x0b3b, 0x08dd},
       {0xf6ee, 0xf551, 0xf3d3, 0xf276, 0xfee7, 0xfd4b, 0xfbce, 0xfa71},
       {0xed89, 0xeced, 0xec72, 0xec17, 0xf586, 0xf4eb, 0xf470, 0xf416},
       {0xec34, 0xec9a, 0xed21, 0xedc8, 0xf435, 0xf49c, 0xf523, 0xf5cb},
       {0xf2ee, 0xf457, 0xf5df, 0xf788, 0xfaf4, 0xfc5d, 0xfde6, 0xff8f}},
      {{0x677d'76af, 0x5c93'1931, 0x6f67'3831, 0x647d'dbb7},
       {0x3eda'09c6, 0x35f3'b250, 0x46c7'cf50, 0x3de2'78de},
       {0x1e46'b4fd, 0x1764'6390, 0x2638'7e90, 0x1f57'2e25},
       {0x05c3'7855, 0x00e5'2cef, 0x0db9'45ef, 0x08db'fb8d},
       {0xf550'cd47, 0xf276'7fe1, 0xfd4a'8eda, 0xfa71'4277},
       {0xeced'a0be, 0xec17'5961, 0xf4eb'6659, 0xf416'1ffe},
       {0xec9a'8c56, 0xedc8'4b01, 0xf49c'55f9, 0xf5cb'15a6},
       {0xf457'900e, 0xf789'54c1, 0xfc5d'5db9, 0xff90'236e}},
      {{0x5c93'192f'ccd4'7781, 0x647d'dbb5'bb66'23af},
       {0x35f3'b24f'43d0'aa38, 0x3de2'78dd'1a4e'4256},
       {0x1764'638e'e2fd'152f, 0x1f57'2e24'a166'993d},
       {0x00e5'2cee'aa59'b866, 0x08db'fb8c'50af'2865},
       {0xf276'7fe1'80cf'f441, 0xfa71'4276'eef1'1fff},
       {0xec17'5961'584c'a799, 0xf416'1ffe'ae59'bf47},
       {0xedc8'4b01'57f9'9330, 0xf5cb'15a6'95f2'96cf},
       {0xf789'54c1'7fd6'b708, 0xff90'236e'a5bb'a696}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVsmulvx,
                        ExecMaskedVsmulvx,
                        {{0, 85, 255, 84, 253, 83, 252, 81, 251, 80, 249, 79, 248, 77, 247, 76},
                         {245, 75, 244, 73, 243, 72, 241, 71, 240, 69, 239, 68, 237, 67, 236, 65},
                         {234, 64, 233, 62, 232, 61, 230, 60, 229, 58, 228, 57, 226, 56, 225, 54},
                         {224, 53, 222, 52, 221, 50, 220, 49, 218, 48, 217, 46, 216, 45, 214, 44},
                         {213, 42, 212, 41, 210, 40, 209, 38, 208, 37, 206, 36, 205, 34, 204, 33},
                         {202, 32, 201, 30, 200, 29, 198, 28, 197, 26, 196, 25, 194, 24, 193, 22},
                         {192, 21, 190, 19, 189, 18, 187, 17, 186, 15, 185, 14, 183, 13, 182, 11},
                         {181, 10, 179, 9, 178, 7, 177, 6, 175, 5, 174, 3, 173, 2, 171, 1}},
                        {{0x54ab, 0x5355, 0x51fe, 0x50a7, 0x4f51, 0x4dfa, 0x4ca3, 0x4b4d},
                         {0x49f6, 0x489f, 0x4749, 0x45f2, 0x449b, 0x4345, 0x41ee, 0x4097},
                         {0x3f40, 0x3dea, 0x3c93, 0x3b3c, 0x39e6, 0x388f, 0x3738, 0x35e2},
                         {0x348b, 0x3334, 0x31de, 0x3087, 0x2f30, 0x2dda, 0x2c83, 0x2b2c},
                         {0x29d6, 0x287f, 0x2728, 0x25d2, 0x247b, 0x2324, 0x21ce, 0x2077},
                         {0x1f20, 0x1dca, 0x1c73, 0x1b1c, 0x19c6, 0x186f, 0x1718, 0x15c2},
                         {0x146b, 0x1314, 0x11bd, 0x1067, 0x0f10, 0x0db9, 0x0c63, 0x0b0c},
                         {0x09b5, 0x085f, 0x0708, 0x05b1, 0x045b, 0x0304, 0x01ad, 0x0057}},
                        {{0x5353'aa01, 0x50a6'51fe, 0x4df8'f9fb, 0x4b4b'a1f9},
                         {0x489e'49f6, 0x45f0'f1f3, 0x4343'99f1, 0x4096'41ee},
                         {0x3de8'e9eb, 0x3b3b'91e8, 0x388e'39e6, 0x35e0'e1e3},
                         {0x3333'89e0, 0x3086'31de, 0x2dd8'd9db, 0x2b2b'81d8},
                         {0x287e'29d6, 0x25d0'd1d3, 0x2323'79d0, 0x2076'21ce},
                         {0x1dc8'c9cb, 0x1b1b'71c8, 0x186e'19c6, 0x15c0'c1c3},
                         {0x1313'69c0, 0x1066'11bd, 0x0db8'b9bb, 0x0b0b'61b8},
                         {0x085e'09b5, 0x05b0'b1b3, 0x0303'59b0, 0x0056'01ad}},
                        {{0x50a6'51fc'fdfe'54ab, 0x4b4b'a1f7'a34e'4f51},
                         {0x45f0'f1f2'489e'49f6, 0x4096'41ec'edee'449b},
                         {0x3b3b'91e7'933e'3f40, 0x35e0'e1e2'388e'39e6},
                         {0x3086'31dc'ddde'348b, 0x2b2b'81d7'832e'2f30},
                         {0x25d0'd1d2'287e'29d6, 0x2076'21cc'cdce'247b},
                         {0x1b1b'71c7'731e'1f20, 0x15c0'c1c2'186e'19c6},
                         {0x1066'11bc'bdbe'146b, 0x0b0b'61b7'630e'0f10},
                         {0x05b0'b1b2'085e'09b5, 0x0056'01ac'adae'045b}},
                        kVectorCalculationsSource);
  TestVectorInstruction(ExecVssrlvv,
                        ExecMaskedVssrlvv,
                        {{0, 32, 0, 2, 2, 33, 0, 2, 4, 34, 1, 2, 12, 35, 1, 2},
                         {16, 36, 1, 2, 10, 37, 1, 2, 12, 38, 2, 2, 28, 39, 2, 2},
                         {32, 40, 2, 3, 18, 41, 2, 3, 20, 42, 3, 3, 44, 43, 3, 3},
                         {48, 44, 3, 3, 26, 45, 3, 3, 28, 46, 4, 3, 60, 47, 4, 3},
                         {64, 48, 4, 3, 34, 49, 4, 3, 36, 50, 5, 3, 76, 51, 5, 3},
                         {80, 52, 5, 3, 42, 53, 5, 3, 44, 54, 6, 3, 92, 55, 6, 3},
                         {96, 56, 6, 4, 50, 57, 6, 4, 52, 58, 7, 4, 108, 59, 7, 4},
                         {112, 60, 7, 4, 58, 61, 7, 4, 60, 62, 8, 4, 124, 63, 8, 4}},
                        {{0x8100, 0x0830, 0x0043, 0x0008, 0x4484, 0x08b1, 0x008d, 0x0009},
                         {0x9110, 0x0931, 0x004b, 0x0009, 0x4c8c, 0x09b2, 0x009d, 0x000a},
                         {0xa120, 0x0a32, 0x0053, 0x000a, 0x5494, 0x0ab3, 0x00ad, 0x000b},
                         {0xb130, 0x0b33, 0x005b, 0x000b, 0x5c9c, 0x0bb4, 0x00bd, 0x000c},
                         {0xc140, 0x0c34, 0x0063, 0x000c, 0x64a4, 0x0cb5, 0x00cd, 0x000d},
                         {0xd150, 0x0d35, 0x006b, 0x000d, 0x6cac, 0x0db6, 0x00dd, 0x000e},
                         {0xe160, 0x0e36, 0x0073, 0x000e, 0x74b4, 0x0eb7, 0x00ed, 0x000f},
                         {0xf170, 0x0f37, 0x007b, 0x000f, 0x7cbc, 0x0fb8, 0x00fd, 0x0010}},
                        {{0x8302'8100, 0x0043'8343, 0x0000'4585, 0x0000'008f},
                         {0x9312'9110, 0x004b'8b4b, 0x0000'4d8d, 0x0000'009f},
                         {0xa322'a120, 0x0053'9353, 0x0000'5595, 0x0000'00af},
                         {0xb332'b130, 0x005b'9b5b, 0x0000'5d9d, 0x0000'00bf},
                         {0xc342'c140, 0x0063'a363, 0x0000'65a5, 0x0000'00cf},
                         {0xd352'd150, 0x006b'ab6b, 0x0000'6dad, 0x0000'00df},
                         {0xe362'e160, 0x0073'b373, 0x0000'75b5, 0x0000'00ef},
                         {0xf372'f170, 0x007b'bb7b, 0x0000'7dbd, 0x0000'00ff}},
                        {{0x8706'8504'8302'8100, 0x0000'4787'4686'4585},
                         {0x0000'0000'9716'9515, 0x0000'0000'0000'4f8f},
                         {0xa726'a524'a322'a120, 0x0000'5797'5696'5595},
                         {0x0000'0000'b736'b535, 0x0000'0000'0000'5f9f},
                         {0xc746'c544'c342'c140, 0x0000'67a7'66a6'65a5},
                         {0x0000'0000'd756'd555, 0x0000'0000'0000'6faf},
                         {0xe766'e564'e362'e160, 0x0000'77b7'76b6'75b5},
                         {0x0000'0000'f776'f575, 0x0000'0000'0000'7fbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvx,
                        ExecMaskedVssrlvx,
                        {{0, 32, 0, 33, 1, 33, 2, 34, 2, 34, 2, 35, 3, 35, 4, 36},
                         {4, 36, 4, 37, 5, 37, 6, 38, 6, 38, 6, 39, 7, 39, 8, 40},
                         {8, 40, 8, 41, 9, 41, 10, 42, 10, 42, 10, 43, 11, 43, 12, 44},
                         {12, 44, 12, 45, 13, 45, 14, 46, 14, 46, 14, 47, 15, 47, 16, 48},
                         {16, 48, 16, 49, 17, 49, 18, 50, 18, 50, 18, 51, 19, 51, 20, 52},
                         {20, 52, 20, 53, 21, 53, 22, 54, 22, 54, 22, 55, 23, 55, 24, 56},
                         {24, 56, 24, 57, 25, 57, 26, 58, 26, 58, 26, 59, 27, 59, 28, 60},
                         {28, 60, 28, 61, 29, 61, 30, 62, 30, 62, 30, 63, 31, 63, 32, 64}},
                        {{0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023, 0x0024},
                         {0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027, 0x0028},
                         {0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b, 0x002c},
                         {0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f, 0x0030},
                         {0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033, 0x0034},
                         {0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037, 0x0038},
                         {0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b, 0x003c},
                         {0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f, 0x0040}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0000'0000'0021'c1a1, 0x0000'0000'0023'c3a3},
                         {0x0000'0000'0025'c5a5, 0x0000'0000'0027'c7a7},
                         {0x0000'0000'0029'c9a9, 0x0000'0000'002b'cbab},
                         {0x0000'0000'002d'cdad, 0x0000'0000'002f'cfaf},
                         {0x0000'0000'0031'd1b1, 0x0000'0000'0033'd3b3},
                         {0x0000'0000'0035'd5b5, 0x0000'0000'0037'd7b7},
                         {0x0000'0000'0039'd9b9, 0x0000'0000'003b'dbbb},
                         {0x0000'0000'003d'ddbd, 0x0000'0000'003f'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvi,
                        ExecMaskedVssrlvi,
                        {{0, 32, 0, 33, 1, 33, 2, 34, 2, 34, 2, 35, 3, 35, 4, 36},
                         {4, 36, 4, 37, 5, 37, 6, 38, 6, 38, 6, 39, 7, 39, 8, 40},
                         {8, 40, 8, 41, 9, 41, 10, 42, 10, 42, 10, 43, 11, 43, 12, 44},
                         {12, 44, 12, 45, 13, 45, 14, 46, 14, 46, 14, 47, 15, 47, 16, 48},
                         {16, 48, 16, 49, 17, 49, 18, 50, 18, 50, 18, 51, 19, 51, 20, 52},
                         {20, 52, 20, 53, 21, 53, 22, 54, 22, 54, 22, 55, 23, 55, 24, 56},
                         {24, 56, 24, 57, 25, 57, 26, 58, 26, 58, 26, 59, 27, 59, 28, 60},
                         {28, 60, 28, 61, 29, 61, 30, 62, 30, 62, 30, 63, 31, 63, 32, 64}},
                        {{0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023, 0x0024},
                         {0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027, 0x0028},
                         {0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b, 0x002c},
                         {0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f, 0x0030},
                         {0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033, 0x0034},
                         {0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037, 0x0038},
                         {0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b, 0x003c},
                         {0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f, 0x0040}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0021'c1a1'4120'c0a0, 0x0023'c3a3'4322'c2a2},
                         {0x0025'c5a5'4524'c4a4, 0x0027'c7a7'4726'c6a6},
                         {0x0029'c9a9'4928'c8a8, 0x002b'cbab'4b2a'caaa},
                         {0x002d'cdad'4d2c'ccac, 0x002f'cfaf'4f2e'ceae},
                         {0x0031'd1b1'5130'd0b0, 0x0033'd3b3'5332'd2b2},
                         {0x0035'd5b5'5534'd4b4, 0x0037'd7b7'5736'd6b6},
                         {0x0039'd9b9'5938'd8b8, 0x003b'dbbb'5b3a'daba},
                         {0x003d'ddbd'5d3c'dcbc, 0x003f'dfbf'5f3e'debe}},
                        kVectorCalculationsSource);
  TestVectorInstruction(ExecVssravv,
                        ExecMaskedVssravv,
                        {{0, 224, 0, 254, 2, 225, 0, 254, 4, 226, 1, 254, 12, 227, 1, 254},
                         {16, 228, 1, 254, 10, 229, 1, 254, 12, 230, 2, 254, 28, 231, 2, 254},
                         {32, 232, 2, 255, 18, 233, 2, 255, 20, 234, 3, 255, 44, 235, 3, 255},
                         {48, 236, 3, 255, 26, 237, 3, 255, 28, 238, 4, 255, 60, 239, 4, 255},
                         {64, 240, 4, 255, 34, 241, 4, 255, 36, 242, 5, 255, 76, 243, 5, 255},
                         {80, 244, 5, 255, 42, 245, 5, 255, 44, 246, 6, 255, 92, 247, 6, 255},
                         {96, 248, 6, 0, 50, 249, 6, 0, 52, 250, 7, 0, 108, 251, 7, 0},
                         {112, 252, 7, 0, 58, 253, 7, 0, 60, 254, 8, 0, 124, 255, 8, 0}},
                        {{0x8100, 0xf830, 0xffc3, 0xfff8, 0xc484, 0xf8b1, 0xff8d, 0xfff9},
                         {0x9110, 0xf931, 0xffcb, 0xfff9, 0xcc8c, 0xf9b2, 0xff9d, 0xfffa},
                         {0xa120, 0xfa32, 0xffd3, 0xfffa, 0xd494, 0xfab3, 0xffad, 0xfffb},
                         {0xb130, 0xfb33, 0xffdb, 0xfffb, 0xdc9c, 0xfbb4, 0xffbd, 0xfffc},
                         {0xc140, 0xfc34, 0xffe3, 0xfffc, 0xe4a4, 0xfcb5, 0xffcd, 0xfffd},
                         {0xd150, 0xfd35, 0xffeb, 0xfffd, 0xecac, 0xfdb6, 0xffdd, 0xfffe},
                         {0xe160, 0xfe36, 0xfff3, 0xfffe, 0xf4b4, 0xfeb7, 0xffed, 0xffff},
                         {0xf170, 0xff37, 0xfffb, 0xffff, 0xfcbc, 0xffb8, 0xfffd, 0x0000}},
                        {{0x8302'8100, 0xffc3'8343, 0xffff'c585, 0xffff'ff8f},
                         {0x9312'9110, 0xffcb'8b4b, 0xffff'cd8d, 0xffff'ff9f},
                         {0xa322'a120, 0xffd3'9353, 0xffff'd595, 0xffff'ffaf},
                         {0xb332'b130, 0xffdb'9b5b, 0xffff'dd9d, 0xffff'ffbf},
                         {0xc342'c140, 0xffe3'a363, 0xffff'e5a5, 0xffff'ffcf},
                         {0xd352'd150, 0xffeb'ab6b, 0xffff'edad, 0xffff'ffdf},
                         {0xe362'e160, 0xfff3'b373, 0xffff'f5b5, 0xffff'ffef},
                         {0xf372'f170, 0xfffb'bb7b, 0xffff'fdbd, 0xffff'ffff}},
                        {{0x8706'8504'8302'8100, 0xffff'c787'4686'4585},
                         {0xffff'ffff'9716'9515, 0xffff'ffff'ffff'cf8f},
                         {0xa726'a524'a322'a120, 0xffff'd797'5696'5595},
                         {0xffff'ffff'b736'b535, 0xffff'ffff'ffff'df9f},
                         {0xc746'c544'c342'c140, 0xffff'e7a7'66a6'65a5},
                         {0xffff'ffff'd756'd555, 0xffff'ffff'ffff'efaf},
                         {0xe766'e564'e362'e160, 0xffff'f7b7'76b6'75b5},
                         {0xffff'ffff'f776'f575, 0xffff'ffff'ffff'ffbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravx,
                        ExecMaskedVssravx,
                        {{0, 224, 0, 225, 1, 225, 2, 226, 2, 226, 2, 227, 3, 227, 4, 228},
                         {4, 228, 4, 229, 5, 229, 6, 230, 6, 230, 6, 231, 7, 231, 8, 232},
                         {8, 232, 8, 233, 9, 233, 10, 234, 10, 234, 10, 235, 11, 235, 12, 236},
                         {12, 236, 12, 237, 13, 237, 14, 238, 14, 238, 14, 239, 15, 239, 16, 240},
                         {16, 240, 16, 241, 17, 241, 18, 242, 18, 242, 18, 243, 19, 243, 20, 244},
                         {20, 244, 20, 245, 21, 245, 22, 246, 22, 246, 22, 247, 23, 247, 24, 248},
                         {24, 248, 24, 249, 25, 249, 26, 250, 26, 250, 26, 251, 27, 251, 28, 252},
                         {28, 252, 28, 253, 29, 253, 30, 254, 30, 254, 30, 255, 31, 255, 32, 0}},
                        {{0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3, 0xffe4},
                         {0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7, 0xffe8},
                         {0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb, 0xffec},
                         {0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef, 0xfff0},
                         {0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3, 0xfff4},
                         {0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7, 0xfff8},
                         {0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb, 0xfffc},
                         {0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff, 0x0000}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffff'ffff'ffe1'c1a1, 0xffff'ffff'ffe3'c3a3},
                         {0xffff'ffff'ffe5'c5a5, 0xffff'ffff'ffe7'c7a7},
                         {0xffff'ffff'ffe9'c9a9, 0xffff'ffff'ffeb'cbab},
                         {0xffff'ffff'ffed'cdad, 0xffff'ffff'ffef'cfaf},
                         {0xffff'ffff'fff1'd1b1, 0xffff'ffff'fff3'd3b3},
                         {0xffff'ffff'fff5'd5b5, 0xffff'ffff'fff7'd7b7},
                         {0xffff'ffff'fff9'd9b9, 0xffff'ffff'fffb'dbbb},
                         {0xffff'ffff'fffd'ddbd, 0xffff'ffff'ffff'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravi,
                        ExecMaskedVssravi,
                        {{0, 224, 0, 225, 1, 225, 2, 226, 2, 226, 2, 227, 3, 227, 4, 228},
                         {4, 228, 4, 229, 5, 229, 6, 230, 6, 230, 6, 231, 7, 231, 8, 232},
                         {8, 232, 8, 233, 9, 233, 10, 234, 10, 234, 10, 235, 11, 235, 12, 236},
                         {12, 236, 12, 237, 13, 237, 14, 238, 14, 238, 14, 239, 15, 239, 16, 240},
                         {16, 240, 16, 241, 17, 241, 18, 242, 18, 242, 18, 243, 19, 243, 20, 244},
                         {20, 244, 20, 245, 21, 245, 22, 246, 22, 246, 22, 247, 23, 247, 24, 248},
                         {24, 248, 24, 249, 25, 249, 26, 250, 26, 250, 26, 251, 27, 251, 28, 252},
                         {28, 252, 28, 253, 29, 253, 30, 254, 30, 254, 30, 255, 31, 255, 32, 0}},
                        {{0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3, 0xffe4},
                         {0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7, 0xffe8},
                         {0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb, 0xffec},
                         {0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef, 0xfff0},
                         {0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3, 0xfff4},
                         {0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7, 0xfff8},
                         {0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb, 0xfffc},
                         {0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff, 0x0000}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffe1'c1a1'4120'c0a0, 0xffe3'c3a3'4322'c2a2},
                         {0xffe5'c5a5'4524'c4a4, 0xffe7'c7a7'4726'c6a6},
                         {0xffe9'c9a9'4928'c8a8, 0xffeb'cbab'4b2a'caaa},
                         {0xffed'cdad'4d2c'ccac, 0xffef'cfaf'4f2e'ceae},
                         {0xfff1'd1b1'5130'd0b0, 0xfff3'd3b3'5332'd2b2},
                         {0xfff5'd5b5'5534'd4b4, 0xfff7'd7b7'5736'd6b6},
                         {0xfff9'd9b9'5938'd8b8, 0xfffb'dbbb'5b3a'daba},
                         {0xfffd'ddbd'5d3c'dcbc, 0xffff'dfbf'5f3e'debe}},
                        kVectorCalculationsSource);
  asm volatile("csrw vxrm, %0\n\t" ::"r"(vxrm));
}

TEST(InlineAsmTestRiscv64, TestRNU) {
  uint64_t vxrm;
  asm volatile(
      "csrr %0, vxrm\n\t"
      "csrwi vxrm, %c1\n\t"
      : "=r"(vxrm)
      : "i"(VXRMFlags::RNU));
  TestVectorInstruction(
      ExecVaadduvv,
      ExecMaskedVaadduvv,
      {{0, 138, 3, 141, 7, 144, 9, 147, 13, 134, 15, 137, 18, 140, 21, 143},
       {24, 162, 27, 165, 31, 168, 33, 171, 37, 158, 39, 161, 42, 164, 45, 167},
       {48, 186, 51, 189, 55, 192, 57, 195, 61, 182, 63, 185, 66, 188, 69, 191},
       {72, 210, 75, 213, 79, 216, 81, 219, 85, 206, 87, 209, 90, 212, 93, 215},
       {96, 106, 99, 109, 103, 112, 105, 115, 109, 102, 111, 105, 114, 108, 117, 111},
       {120, 130, 123, 133, 127, 136, 129, 139, 133, 126, 135, 129, 138, 132, 141, 135},
       {144, 154, 147, 157, 151, 160, 153, 163, 157, 150, 159, 153, 162, 156, 165, 159},
       {168, 178, 171, 181, 175, 184, 177, 187, 181, 174, 183, 177, 186, 180, 189, 183}},
      {{0x8980, 0x8c83, 0x8f87, 0x9289, 0x858d, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79f, 0xaaa1, 0x9da5, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb7, 0xc2b9, 0xb5bd, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7cf, 0xdad1, 0xcdd5, 0xd0d7, 0xd3da, 0xd6dd},
       {0x69e0, 0x6ce3, 0x6fe7, 0x72e9, 0x65ed, 0x68ef, 0x6bf2, 0x6ef5},
       {0x81f8, 0x84fb, 0x87ff, 0x8b01, 0x7e05, 0x8107, 0x840a, 0x870d},
       {0x9a10, 0x9d13, 0xa017, 0xa319, 0x961d, 0x991f, 0x9c22, 0x9f25},
       {0xb228, 0xb52b, 0xb82f, 0xbb31, 0xae35, 0xb137, 0xb43a, 0xb73d}},
      {{0x8c83'8980, 0x9289'8f87, 0x888f'858d, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79f, 0xa0a7'9da5, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb7, 0xb8bf'b5bd, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7cf, 0xd0d7'cdd5, 0xd6dd'd3da},
       {0x6ce3'69e0, 0x72e9'6fe7, 0x68ef'65ed, 0x6ef5'6bf2},
       {0x84fb'81f8, 0x8b01'87ff, 0x8107'7e05, 0x870d'840a},
       {0x9d13'9a10, 0xa319'a017, 0x991f'961d, 0x9f25'9c22},
       {0xb52b'b228, 0xbb31'b82f, 0xb137'ae35, 0xb73d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858d},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da5},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bd},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd5},
       {0x72e9'6fe6'ece3'69e0, 0x6ef5'6bf2'68ef'65ed},
       {0x8b01'87ff'04fb'81f8, 0x870d'840a'8107'7e05},
       {0xa319'a017'1d13'9a10, 0x9f25'9c22'991f'961d},
       {0xbb31'b82f'352b'b228, 0xb73d'b43a'b137'ae35}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaadduvx,
      ExecMaskedVaadduvx,
      {{85, 150, 86, 151, 87, 152, 88, 153, 89, 154, 90, 155, 91, 156, 92, 157},
       {93, 158, 94, 159, 95, 160, 96, 161, 97, 162, 98, 163, 99, 164, 100, 165},
       {101, 166, 102, 167, 103, 168, 104, 169, 105, 170, 106, 171, 107, 172, 108, 173},
       {109, 174, 110, 175, 111, 176, 112, 177, 113, 178, 114, 179, 115, 180, 116, 181},
       {117, 182, 118, 183, 119, 184, 120, 185, 121, 186, 122, 187, 123, 188, 124, 189},
       {125, 190, 126, 191, 127, 192, 128, 193, 129, 194, 130, 195, 131, 196, 132, 197},
       {133, 198, 134, 199, 135, 200, 136, 201, 137, 202, 138, 203, 139, 204, 140, 205},
       {141, 206, 142, 207, 143, 208, 144, 209, 145, 210, 146, 211, 147, 212, 148, 213}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvv,
      ExecMaskedVaaddvv,
      {{0, 138, 3, 141, 7, 144, 9, 147, 13, 134, 15, 137, 18, 140, 21, 143},
       {24, 162, 27, 165, 31, 168, 33, 171, 37, 158, 39, 161, 42, 164, 45, 167},
       {48, 186, 51, 189, 55, 192, 57, 195, 61, 182, 63, 185, 66, 188, 69, 191},
       {72, 210, 75, 213, 79, 216, 81, 219, 85, 206, 87, 209, 90, 212, 93, 215},
       {224, 234, 227, 237, 231, 240, 233, 243, 237, 230, 239, 233, 242, 236, 245, 239},
       {248, 2, 251, 5, 255, 8, 1, 11, 5, 254, 7, 1, 10, 4, 13, 7},
       {16, 26, 19, 29, 23, 32, 25, 35, 29, 22, 31, 25, 34, 28, 37, 31},
       {40, 50, 43, 53, 47, 56, 49, 59, 53, 46, 55, 49, 58, 52, 61, 55}},
      {{0x8980, 0x8c83, 0x8f87, 0x9289, 0x858d, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79f, 0xaaa1, 0x9da5, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb7, 0xc2b9, 0xb5bd, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7cf, 0xdad1, 0xcdd5, 0xd0d7, 0xd3da, 0xd6dd},
       {0xe9e0, 0xece3, 0xefe7, 0xf2e9, 0xe5ed, 0xe8ef, 0xebf2, 0xeef5},
       {0x01f8, 0x04fb, 0x07ff, 0x0b01, 0xfe05, 0x0107, 0x040a, 0x070d},
       {0x1a10, 0x1d13, 0x2017, 0x2319, 0x161d, 0x191f, 0x1c22, 0x1f25},
       {0x3228, 0x352b, 0x382f, 0x3b31, 0x2e35, 0x3137, 0x343a, 0x373d}},
      {{0x8c83'8980, 0x9289'8f87, 0x888f'858d, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79f, 0xa0a7'9da5, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb7, 0xb8bf'b5bd, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7cf, 0xd0d7'cdd5, 0xd6dd'd3da},
       {0xece3'69e0, 0xf2e9'6fe7, 0xe8ef'65ed, 0xeef5'6bf2},
       {0x04fb'81f8, 0x0b01'87ff, 0x0107'7e05, 0x070d'840a},
       {0x1d13'9a10, 0x2319'a017, 0x191f'961d, 0x1f25'9c22},
       {0x352b'b228, 0x3b31'b82f, 0x3137'ae35, 0x373d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858d},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da5},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bd},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd5},
       {0xf2e9'6fe6'ece3'69e0, 0xeef5'6bf2'68ef'65ed},
       {0x0b01'87ff'04fb'81f8, 0x070d'840a'8107'7e05},
       {0x2319'a017'1d13'9a10, 0x1f25'9c22'991f'961d},
       {0x3b31'b82f'352b'b228, 0x373d'b43a'b137'ae35}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvx,
      ExecMaskedVaaddvx,
      {{213, 150, 214, 151, 215, 152, 216, 153, 217, 154, 218, 155, 219, 156, 220, 157},
       {221, 158, 222, 159, 223, 160, 224, 161, 225, 162, 226, 163, 227, 164, 228, 165},
       {229, 166, 230, 167, 231, 168, 232, 169, 233, 170, 234, 171, 235, 172, 236, 173},
       {237, 174, 238, 175, 239, 176, 240, 177, 241, 178, 242, 179, 243, 180, 244, 181},
       {245, 182, 246, 183, 247, 184, 248, 185, 249, 186, 250, 187, 251, 188, 252, 189},
       {253, 190, 254, 191, 255, 192, 0, 193, 1, 194, 2, 195, 3, 196, 4, 197},
       {5, 198, 6, 199, 7, 200, 8, 201, 9, 202, 10, 203, 11, 204, 12, 205},
       {13, 206, 14, 207, 15, 208, 16, 209, 17, 210, 18, 211, 19, 212, 20, 213}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvv,
      ExecMaskedVasubuvv,
      {{0, 248, 255, 247, 254, 246, 253, 245, 252, 4, 251, 3, 250, 2, 249, 1},
       {248, 240, 247, 239, 246, 238, 245, 237, 244, 252, 243, 251, 242, 250, 241, 249},
       {240, 232, 239, 231, 238, 230, 237, 229, 236, 244, 235, 243, 234, 242, 233, 241},
       {232, 224, 231, 223, 230, 222, 229, 221, 228, 236, 227, 235, 226, 234, 225, 233},
       {224, 88, 223, 87, 222, 86, 221, 85, 220, 100, 219, 99, 218, 98, 217, 97},
       {216, 80, 215, 79, 214, 78, 213, 77, 212, 92, 211, 91, 210, 90, 209, 89},
       {208, 72, 207, 71, 206, 70, 205, 69, 204, 84, 203, 83, 202, 82, 201, 81},
       {200, 64, 199, 63, 198, 62, 197, 61, 196, 76, 195, 75, 194, 74, 193, 73}},
      {{0xf780, 0xf67f, 0xf57e, 0xf47d, 0x037c, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed76, 0xec75, 0xfb74, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56e, 0xe46d, 0xf36c, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd66, 0xdc65, 0xeb64, 0xea63, 0xe962, 0xe861},
       {0x5760, 0x565f, 0x555e, 0x545d, 0x635c, 0x625b, 0x615a, 0x6059},
       {0x4f58, 0x4e57, 0x4d56, 0x4c55, 0x5b54, 0x5a53, 0x5952, 0x5851},
       {0x4750, 0x464f, 0x454e, 0x444d, 0x534c, 0x524b, 0x514a, 0x5049},
       {0x3f48, 0x3e47, 0x3d46, 0x3c45, 0x4b44, 0x4a43, 0x4942, 0x4841}},
      {{0xf67e'f780, 0xf47c'f57e, 0x027b'037c, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed76, 0xfa72'fb74, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56e, 0xf26a'f36c, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd66, 0xea62'eb64, 0xe860'e962},
       {0x565f'5760, 0x545d'555e, 0x625b'635c, 0x6059'615a},
       {0x4e57'4f58, 0x4c55'4d56, 0x5a53'5b54, 0x5851'5952},
       {0x464f'4750, 0x444d'454e, 0x524b'534c, 0x5049'514a},
       {0x3e47'3f48, 0x3c45'3d46, 0x4a43'4b44, 0x4841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037c},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb74},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36c},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb64},
       {0x545d'555d'd65f'5760, 0x6059'615a'625b'635c},
       {0x4c55'4d55'ce57'4f58, 0x5851'5952'5a53'5b54},
       {0x444d'454d'c64f'4750, 0x5049'514a'524b'534c},
       {0x3c45'3d45'be47'3f48, 0x4841'4942'4a43'4b44}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvx,
      ExecMaskedVasubuvx,
      {{171, 236, 172, 237, 173, 238, 174, 239, 175, 240, 176, 241, 177, 242, 178, 243},
       {179, 244, 180, 245, 181, 246, 182, 247, 183, 248, 184, 249, 185, 250, 186, 251},
       {187, 252, 188, 253, 189, 254, 190, 255, 191, 0, 192, 1, 193, 2, 194, 3},
       {195, 4, 196, 5, 197, 6, 198, 7, 199, 8, 200, 9, 201, 10, 202, 11},
       {203, 12, 204, 13, 205, 14, 206, 15, 207, 16, 208, 17, 209, 18, 210, 19},
       {211, 20, 212, 21, 213, 22, 214, 23, 215, 24, 216, 25, 217, 26, 218, 27},
       {219, 28, 220, 29, 221, 30, 222, 31, 223, 32, 224, 33, 225, 34, 226, 35},
       {227, 36, 228, 37, 229, 38, 230, 39, 231, 40, 232, 41, 233, 42, 234, 43}},
      {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
       {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
       {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
       {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
       {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
       {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
       {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
       {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
      {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
       {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
       {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
       {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
       {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
       {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
       {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
       {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
      {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
       {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
       {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
       {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
       {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
       {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
       {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
       {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubvv,
      ExecMaskedVasubvv,
      {{0, 248, 255, 247, 254, 246, 253, 245, 252, 4, 251, 3, 250, 2, 249, 1},
       {248, 240, 247, 239, 246, 238, 245, 237, 244, 252, 243, 251, 242, 250, 241, 249},
       {240, 232, 239, 231, 238, 230, 237, 229, 236, 244, 235, 243, 234, 242, 233, 241},
       {232, 224, 231, 223, 230, 222, 229, 221, 228, 236, 227, 235, 226, 234, 225, 233},
       {96, 216, 95, 215, 94, 214, 93, 213, 92, 228, 91, 227, 90, 226, 89, 225},
       {88, 208, 87, 207, 86, 206, 85, 205, 84, 220, 83, 219, 82, 218, 81, 217},
       {80, 200, 79, 199, 78, 198, 77, 197, 76, 212, 75, 211, 74, 210, 73, 209},
       {72, 192, 71, 191, 70, 190, 69, 189, 68, 204, 67, 203, 66, 202, 65, 201}},
      {{0xf780, 0xf67f, 0xf57e, 0xf47d, 0x037c, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed76, 0xec75, 0xfb74, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56e, 0xe46d, 0xf36c, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd66, 0xdc65, 0xeb64, 0xea63, 0xe962, 0xe861},
       {0xd760, 0xd65f, 0xd55e, 0xd45d, 0xe35c, 0xe25b, 0xe15a, 0xe059},
       {0xcf58, 0xce57, 0xcd56, 0xcc55, 0xdb54, 0xda53, 0xd952, 0xd851},
       {0xc750, 0xc64f, 0xc54e, 0xc44d, 0xd34c, 0xd24b, 0xd14a, 0xd049},
       {0xbf48, 0xbe47, 0xbd46, 0xbc45, 0xcb44, 0xca43, 0xc942, 0xc841}},
      {{0xf67e'f780, 0xf47c'f57e, 0x027b'037c, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed76, 0xfa72'fb74, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56e, 0xf26a'f36c, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd66, 0xea62'eb64, 0xe860'e962},
       {0xd65f'5760, 0xd45d'555e, 0xe25b'635c, 0xe059'615a},
       {0xce57'4f58, 0xcc55'4d56, 0xda53'5b54, 0xd851'5952},
       {0xc64f'4750, 0xc44d'454e, 0xd24b'534c, 0xd049'514a},
       {0xbe47'3f48, 0xbc45'3d46, 0xca43'4b44, 0xc841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037c},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb74},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36c},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb64},
       {0xd45d'555d'd65f'5760, 0xe059'615a'625b'635c},
       {0xcc55'4d55'ce57'4f58, 0xd851'5952'5a53'5b54},
       {0xc44d'454d'c64f'4750, 0xd049'514a'524b'534c},
       {0xbc45'3d45'be47'3f48, 0xc841'4942'4a43'4b44}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVasubvx,
                        ExecMaskedVasubvx,
                        {{43, 236, 44, 237, 45, 238, 46, 239, 47, 240, 48, 241, 49, 242, 50, 243},
                         {51, 244, 52, 245, 53, 246, 54, 247, 55, 248, 56, 249, 57, 250, 58, 251},
                         {59, 252, 60, 253, 61, 254, 62, 255, 63, 0, 64, 1, 65, 2, 66, 3},
                         {67, 4, 68, 5, 69, 6, 70, 7, 71, 8, 72, 9, 73, 10, 74, 11},
                         {75, 12, 76, 13, 77, 14, 78, 15, 79, 16, 80, 17, 81, 18, 82, 19},
                         {83, 20, 84, 21, 85, 22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27},
                         {91, 28, 92, 29, 93, 30, 94, 31, 95, 32, 96, 33, 97, 34, 98, 35},
                         {99, 36, 100, 37, 101, 38, 102, 39, 103, 40, 104, 41, 105, 42, 106, 43}},
                        {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
                         {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
                         {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
                         {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
                         {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
                         {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
                         {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
                         {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
                        {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
                         {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
                         {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
                         {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
                         {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
                         {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
                         {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
                         {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
                        {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
                         {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
                         {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
                         {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
                         {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
                         {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
                         {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
                         {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
                        kVectorCalculationsSource);
  TestNarrowingVectorInstruction(ExecVnclipuwi,
                                 ExecMaskedVnclipuwi,
                                 {{32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40},
                                  {40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48},
                                  {48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56},
                                  {56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                                 kVectorCalculationsSource);
  TestNarrowingVectorInstruction(
      ExecVnclipwi,
      ExecMaskedVnclipwi,
      {{224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232},
       {232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239, 240},
       {240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 248},
       {248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255, 0}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(ExecVnclipuwx,
                                 ExecMaskedVnclipuwx,
                                 {{32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40},
                                  {40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48},
                                  {48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56},
                                  {56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0x0021'c1a1, 0x0023'c3a3, 0x0025'c5a5, 0x0027'c7a7},
                                  {0x0029'c9a9, 0x002b'cbab, 0x002d'cdad, 0x002f'cfaf},
                                  {0x0031'd1b1, 0x0033'd3b3, 0x0035'd5b5, 0x0037'd7b7},
                                  {0x0039'd9b9, 0x003b'dbbb, 0x003d'ddbd, 0x003f'dfbf}},
                                 kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwx,
      ExecMaskedVnclipwx,
      {{224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232},
       {232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239, 240},
       {240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 248},
       {248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255, 0}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0xffe1'c1a1, 0xffe3'c3a3, 0xffe5'c5a5, 0xffe7'c7a7},
       {0xffe9'c9a9, 0xffeb'cbab, 0xffed'cdad, 0xffef'cfaf},
       {0xfff1'd1b1, 0xfff3'd3b3, 0xfff5'd5b5, 0xfff7'd7b7},
       {0xfff9'd9b9, 0xfffb'dbbb, 0xfffd'ddbd, 0xffff'dfbf}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipuwv,
      ExecMaskedVnclipuwv,
      {{255, 255, 255, 255, 69, 35, 9, 2, 255, 255, 255, 255, 153, 39, 10, 2},
       {255, 255, 255, 255, 85, 43, 11, 3, 255, 255, 255, 255, 185, 47, 12, 3},
       {255, 255, 255, 255, 101, 51, 13, 3, 255, 255, 255, 255, 217, 55, 14, 3},
       {255, 255, 255, 255, 117, 59, 15, 4, 255, 255, 255, 255, 249, 63, 16, 4}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0x4989, 0x0971, 0x009b, 0x000a},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x5999, 0x0b73, 0x00bb, 0x000c},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x69a9, 0x0d75, 0x00db, 0x000e},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x79b9, 0x0f77, 0x00fb, 0x0010}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xa726'a525, 0x0057'9757, 0x0000'5b9b, 0x0000'00bf},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xe766'e565, 0x0077'b777, 0x0000'7bbb, 0x0000'00ff}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwv,
      ExecMaskedVnclipwv,
      {{128, 128, 128, 128, 197, 227, 249, 254, 128, 128, 128, 128, 153, 231, 250, 254},
       {128, 128, 128, 128, 213, 235, 251, 255, 128, 128, 128, 128, 185, 239, 252, 255},
       {128, 128, 128, 128, 229, 243, 253, 255, 128, 128, 128, 128, 217, 247, 254, 255},
       {128, 128, 128, 158, 245, 251, 255, 0, 128, 128, 128, 222, 249, 255, 0, 0}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0xc989, 0xf971, 0xff9b, 0xfffa},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xd999, 0xfb73, 0xffbb, 0xfffc},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xe9a9, 0xfd75, 0xffdb, 0xfffe},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xf9b9, 0xff77, 0xfffb, 0x0000}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xa726'a525, 0xffd7'9757, 0xffff'db9b, 0xffff'ffbf},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xe766'e565, 0xfff7'b777, 0xffff'fbbb, 0xffff'ffff}},
      kVectorCalculationsSource);

  TestVectorInstruction(
      ExecVsmulvv,
      ExecMaskedVsmulvv,
      {{0, 109, 0, 104, 0, 98, 1, 93, 1, 117, 2, 112, 2, 106, 3, 101},
       {4, 68, 5, 63, 6, 59, 8, 54, 9, 76, 11, 71, 12, 67, 14, 62},
       {16, 34, 18, 31, 21, 27, 23, 24, 25, 42, 28, 39, 30, 35, 33, 32},
       {36, 9, 39, 6, 43, 4, 46, 1, 49, 17, 53, 14, 56, 12, 60, 9},
       {192, 247, 192, 246, 193, 244, 193, 243, 194, 255, 194, 254, 194, 252, 195, 251},
       {196, 238, 197, 237, 199, 237, 200, 236, 202, 246, 203, 245, 204, 245, 206, 244},
       {208, 236, 210, 237, 213, 237, 215, 238, 218, 244, 220, 245, 222, 245, 225, 246},
       {228, 243, 231, 244, 235, 246, 238, 247, 242, 251, 245, 252, 248, 254, 252, 255}},
      {{0x6d24, 0x677e, 0x61f8, 0x5c94, 0x750c, 0x6f68, 0x69e3, 0x647f},
       {0x437f, 0x3edb, 0x3a57, 0x35f5, 0x4b6b, 0x46c9, 0x4246, 0x3de3},
       {0x21e9, 0x1e48, 0x1ac5, 0x1765, 0x29da, 0x2639, 0x22b8, 0x1f58},
       {0x0864, 0x05c4, 0x0344, 0x00e6, 0x1058, 0x0dba, 0x0b3b, 0x08dd},
       {0xf6ee, 0xf551, 0xf3d3, 0xf276, 0xfee7, 0xfd4b, 0xfbce, 0xfa71},
       {0xed89, 0xeced, 0xec72, 0xec17, 0xf586, 0xf4eb, 0xf470, 0xf416},
       {0xec34, 0xec9a, 0xed21, 0xedc8, 0xf435, 0xf49c, 0xf523, 0xf5cb},
       {0xf2ee, 0xf457, 0xf5df, 0xf788, 0xfaf4, 0xfc5d, 0xfde6, 0xff8f}},
      {{0x677d'76af, 0x5c93'1931, 0x6f67'3831, 0x647d'dbb7},
       {0x3eda'09c6, 0x35f3'b250, 0x46c7'cf50, 0x3de2'78de},
       {0x1e46'b4fd, 0x1764'6390, 0x2638'7e90, 0x1f57'2e25},
       {0x05c3'7855, 0x00e5'2cef, 0x0db9'45ef, 0x08db'fb8d},
       {0xf550'cd47, 0xf276'7fe1, 0xfd4a'8eda, 0xfa71'4277},
       {0xeced'a0be, 0xec17'5961, 0xf4eb'6659, 0xf416'1ffe},
       {0xec9a'8c56, 0xedc8'4b01, 0xf49c'55f9, 0xf5cb'15a6},
       {0xf457'900e, 0xf789'54c1, 0xfc5d'5db9, 0xff90'236e}},
      {{0x5c93'192f'ccd4'7781, 0x647d'dbb5'bb66'23af},
       {0x35f3'b24f'43d0'aa38, 0x3de2'78dd'1a4e'4256},
       {0x1764'638e'e2fd'152f, 0x1f57'2e24'a166'993d},
       {0x00e5'2cee'aa59'b866, 0x08db'fb8c'50af'2865},
       {0xf276'7fe1'80cf'f441, 0xfa71'4276'eef1'1fff},
       {0xec17'5961'584c'a799, 0xf416'1ffe'ae59'bf47},
       {0xedc8'4b01'57f9'9330, 0xf5cb'15a6'95f2'96cf},
       {0xf789'54c1'7fd6'b708, 0xff90'236e'a5bb'a696}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVsmulvx,
                        ExecMaskedVsmulvx,
                        {{0, 85, 255, 84, 253, 83, 252, 81, 251, 80, 249, 79, 248, 77, 247, 76},
                         {245, 75, 244, 73, 243, 72, 241, 71, 240, 69, 239, 68, 237, 67, 236, 65},
                         {235, 64, 233, 62, 232, 61, 230, 60, 229, 58, 228, 57, 226, 56, 225, 54},
                         {224, 53, 222, 52, 221, 50, 220, 49, 218, 48, 217, 46, 216, 45, 214, 44},
                         {213, 42, 212, 41, 210, 40, 209, 38, 208, 37, 206, 36, 205, 34, 204, 33},
                         {202, 32, 201, 30, 200, 29, 198, 28, 197, 26, 196, 25, 194, 24, 193, 22},
                         {192, 21, 190, 19, 189, 18, 187, 17, 186, 15, 185, 14, 183, 13, 182, 11},
                         {181, 10, 179, 9, 178, 7, 177, 6, 175, 5, 174, 3, 173, 2, 171, 1}},
                        {{0x54ab, 0x5355, 0x51fe, 0x50a7, 0x4f51, 0x4dfa, 0x4ca3, 0x4b4d},
                         {0x49f6, 0x489f, 0x4749, 0x45f2, 0x449b, 0x4345, 0x41ee, 0x4097},
                         {0x3f40, 0x3dea, 0x3c93, 0x3b3c, 0x39e6, 0x388f, 0x3738, 0x35e2},
                         {0x348b, 0x3334, 0x31de, 0x3087, 0x2f30, 0x2dda, 0x2c83, 0x2b2c},
                         {0x29d6, 0x287f, 0x2728, 0x25d2, 0x247b, 0x2324, 0x21ce, 0x2077},
                         {0x1f20, 0x1dca, 0x1c73, 0x1b1c, 0x19c6, 0x186f, 0x1718, 0x15c2},
                         {0x146b, 0x1314, 0x11bd, 0x1067, 0x0f10, 0x0db9, 0x0c63, 0x0b0c},
                         {0x09b5, 0x085f, 0x0708, 0x05b1, 0x045b, 0x0304, 0x01ad, 0x0057}},
                        {{0x5353'aa01, 0x50a6'51fe, 0x4df8'f9fb, 0x4b4b'a1f9},
                         {0x489e'49f6, 0x45f0'f1f3, 0x4343'99f1, 0x4096'41ee},
                         {0x3de8'e9eb, 0x3b3b'91e8, 0x388e'39e6, 0x35e0'e1e3},
                         {0x3333'89e0, 0x3086'31de, 0x2dd8'd9db, 0x2b2b'81d8},
                         {0x287e'29d6, 0x25d0'd1d3, 0x2323'79d0, 0x2076'21ce},
                         {0x1dc8'c9cb, 0x1b1b'71c8, 0x186e'19c6, 0x15c0'c1c3},
                         {0x1313'69c0, 0x1066'11bd, 0x0db8'b9bb, 0x0b0b'61b8},
                         {0x085e'09b5, 0x05b0'b1b3, 0x0303'59b0, 0x0056'01ad}},
                        {{0x50a6'51fc'fdfe'54ab, 0x4b4b'a1f7'a34e'4f51},
                         {0x45f0'f1f2'489e'49f6, 0x4096'41ec'edee'449b},
                         {0x3b3b'91e7'933e'3f40, 0x35e0'e1e2'388e'39e6},
                         {0x3086'31dc'ddde'348b, 0x2b2b'81d7'832e'2f30},
                         {0x25d0'd1d2'287e'29d6, 0x2076'21cc'cdce'247b},
                         {0x1b1b'71c7'731e'1f20, 0x15c0'c1c2'186e'19c6},
                         {0x1066'11bc'bdbe'146b, 0x0b0b'61b7'630e'0f10},
                         {0x05b0'b1b2'085e'09b5, 0x0056'01ac'adae'045b}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvv,
                        ExecMaskedVssrlvv,
                        {{0, 32, 0, 2, 2, 33, 0, 2, 4, 34, 1, 2, 12, 35, 1, 2},
                         {16, 36, 1, 2, 10, 37, 1, 2, 12, 38, 2, 2, 28, 39, 2, 2},
                         {32, 40, 2, 3, 18, 41, 2, 3, 20, 42, 3, 3, 44, 43, 3, 3},
                         {48, 44, 3, 3, 26, 45, 3, 3, 28, 46, 4, 3, 60, 47, 4, 3},
                         {64, 48, 4, 3, 34, 49, 4, 3, 36, 50, 5, 3, 76, 51, 5, 3},
                         {80, 52, 5, 3, 42, 53, 5, 3, 44, 54, 6, 3, 92, 55, 6, 3},
                         {96, 56, 6, 4, 50, 57, 6, 4, 52, 58, 7, 4, 108, 59, 7, 4},
                         {112, 60, 7, 4, 58, 61, 7, 4, 60, 62, 8, 4, 124, 63, 8, 4}},
                        {{0x8100, 0x0830, 0x0043, 0x0008, 0x4484, 0x08b1, 0x008d, 0x0009},
                         {0x9110, 0x0931, 0x004b, 0x0009, 0x4c8c, 0x09b2, 0x009d, 0x000a},
                         {0xa120, 0x0a32, 0x0053, 0x000a, 0x5494, 0x0ab3, 0x00ad, 0x000b},
                         {0xb130, 0x0b33, 0x005b, 0x000b, 0x5c9c, 0x0bb4, 0x00bd, 0x000c},
                         {0xc140, 0x0c34, 0x0063, 0x000c, 0x64a4, 0x0cb5, 0x00cd, 0x000d},
                         {0xd150, 0x0d35, 0x006b, 0x000d, 0x6cac, 0x0db6, 0x00dd, 0x000e},
                         {0xe160, 0x0e36, 0x0073, 0x000e, 0x74b4, 0x0eb7, 0x00ed, 0x000f},
                         {0xf170, 0x0f37, 0x007b, 0x000f, 0x7cbc, 0x0fb8, 0x00fd, 0x0010}},
                        {{0x8302'8100, 0x0043'8343, 0x0000'4585, 0x0000'008f},
                         {0x9312'9110, 0x004b'8b4b, 0x0000'4d8d, 0x0000'009f},
                         {0xa322'a120, 0x0053'9353, 0x0000'5595, 0x0000'00af},
                         {0xb332'b130, 0x005b'9b5b, 0x0000'5d9d, 0x0000'00bf},
                         {0xc342'c140, 0x0063'a363, 0x0000'65a5, 0x0000'00cf},
                         {0xd352'd150, 0x006b'ab6b, 0x0000'6dad, 0x0000'00df},
                         {0xe362'e160, 0x0073'b373, 0x0000'75b5, 0x0000'00ef},
                         {0xf372'f170, 0x007b'bb7b, 0x0000'7dbd, 0x0000'00ff}},
                        {{0x8706'8504'8302'8100, 0x0000'4787'4686'4585},
                         {0x0000'0000'9716'9515, 0x0000'0000'0000'4f8f},
                         {0xa726'a524'a322'a120, 0x0000'5797'5696'5595},
                         {0x0000'0000'b736'b535, 0x0000'0000'0000'5f9f},
                         {0xc746'c544'c342'c140, 0x0000'67a7'66a6'65a5},
                         {0x0000'0000'd756'd555, 0x0000'0000'0000'6faf},
                         {0xe766'e564'e362'e160, 0x0000'77b7'76b6'75b5},
                         {0x0000'0000'f776'f575, 0x0000'0000'0000'7fbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvx,
                        ExecMaskedVssrlvx,
                        {{0, 32, 1, 33, 1, 33, 2, 34, 2, 34, 3, 35, 3, 35, 4, 36},
                         {4, 36, 5, 37, 5, 37, 6, 38, 6, 38, 7, 39, 7, 39, 8, 40},
                         {8, 40, 9, 41, 9, 41, 10, 42, 10, 42, 11, 43, 11, 43, 12, 44},
                         {12, 44, 13, 45, 13, 45, 14, 46, 14, 46, 15, 47, 15, 47, 16, 48},
                         {16, 48, 17, 49, 17, 49, 18, 50, 18, 50, 19, 51, 19, 51, 20, 52},
                         {20, 52, 21, 53, 21, 53, 22, 54, 22, 54, 23, 55, 23, 55, 24, 56},
                         {24, 56, 25, 57, 25, 57, 26, 58, 26, 58, 27, 59, 27, 59, 28, 60},
                         {28, 60, 29, 61, 29, 61, 30, 62, 30, 62, 31, 63, 31, 63, 32, 64}},
                        {{0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023, 0x0024},
                         {0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027, 0x0028},
                         {0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b, 0x002c},
                         {0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f, 0x0030},
                         {0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033, 0x0034},
                         {0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037, 0x0038},
                         {0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b, 0x003c},
                         {0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f, 0x0040}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0000'0000'0021'c1a1, 0x0000'0000'0023'c3a3},
                         {0x0000'0000'0025'c5a5, 0x0000'0000'0027'c7a7},
                         {0x0000'0000'0029'c9a9, 0x0000'0000'002b'cbab},
                         {0x0000'0000'002d'cdad, 0x0000'0000'002f'cfaf},
                         {0x0000'0000'0031'd1b1, 0x0000'0000'0033'd3b3},
                         {0x0000'0000'0035'd5b5, 0x0000'0000'0037'd7b7},
                         {0x0000'0000'0039'd9b9, 0x0000'0000'003b'dbbb},
                         {0x0000'0000'003d'ddbd, 0x0000'0000'003f'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvi,
                        ExecMaskedVssrlvi,
                        {{0, 32, 1, 33, 1, 33, 2, 34, 2, 34, 3, 35, 3, 35, 4, 36},
                         {4, 36, 5, 37, 5, 37, 6, 38, 6, 38, 7, 39, 7, 39, 8, 40},
                         {8, 40, 9, 41, 9, 41, 10, 42, 10, 42, 11, 43, 11, 43, 12, 44},
                         {12, 44, 13, 45, 13, 45, 14, 46, 14, 46, 15, 47, 15, 47, 16, 48},
                         {16, 48, 17, 49, 17, 49, 18, 50, 18, 50, 19, 51, 19, 51, 20, 52},
                         {20, 52, 21, 53, 21, 53, 22, 54, 22, 54, 23, 55, 23, 55, 24, 56},
                         {24, 56, 25, 57, 25, 57, 26, 58, 26, 58, 27, 59, 27, 59, 28, 60},
                         {28, 60, 29, 61, 29, 61, 30, 62, 30, 62, 31, 63, 31, 63, 32, 64}},
                        {{0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023, 0x0024},
                         {0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027, 0x0028},
                         {0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b, 0x002c},
                         {0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f, 0x0030},
                         {0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033, 0x0034},
                         {0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037, 0x0038},
                         {0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b, 0x003c},
                         {0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f, 0x0040}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0021'c1a1'4120'c0a0, 0x0023'c3a3'4322'c2a2},
                         {0x0025'c5a5'4524'c4a4, 0x0027'c7a7'4726'c6a6},
                         {0x0029'c9a9'4928'c8a8, 0x002b'cbab'4b2a'caaa},
                         {0x002d'cdad'4d2c'ccac, 0x002f'cfaf'4f2e'ceae},
                         {0x0031'd1b1'5130'd0b0, 0x0033'd3b3'5332'd2b2},
                         {0x0035'd5b5'5534'd4b4, 0x0037'd7b7'5736'd6b6},
                         {0x0039'd9b9'5938'd8b8, 0x003b'dbbb'5b3a'daba},
                         {0x003d'ddbd'5d3c'dcbc, 0x003f'dfbf'5f3e'debe}},
                        kVectorCalculationsSource);
  TestVectorInstruction(ExecVssravv,
                        ExecMaskedVssravv,
                        {{0, 224, 0, 254, 2, 225, 0, 254, 4, 226, 1, 254, 12, 227, 1, 254},
                         {16, 228, 1, 254, 10, 229, 1, 254, 12, 230, 2, 254, 28, 231, 2, 254},
                         {32, 232, 2, 255, 18, 233, 2, 255, 20, 234, 3, 255, 44, 235, 3, 255},
                         {48, 236, 3, 255, 26, 237, 3, 255, 28, 238, 4, 255, 60, 239, 4, 255},
                         {64, 240, 4, 255, 34, 241, 4, 255, 36, 242, 5, 255, 76, 243, 5, 255},
                         {80, 244, 5, 255, 42, 245, 5, 255, 44, 246, 6, 255, 92, 247, 6, 255},
                         {96, 248, 6, 0, 50, 249, 6, 0, 52, 250, 7, 0, 108, 251, 7, 0},
                         {112, 252, 7, 0, 58, 253, 7, 0, 60, 254, 8, 0, 124, 255, 8, 0}},
                        {{0x8100, 0xf830, 0xffc3, 0xfff8, 0xc484, 0xf8b1, 0xff8d, 0xfff9},
                         {0x9110, 0xf931, 0xffcb, 0xfff9, 0xcc8c, 0xf9b2, 0xff9d, 0xfffa},
                         {0xa120, 0xfa32, 0xffd3, 0xfffa, 0xd494, 0xfab3, 0xffad, 0xfffb},
                         {0xb130, 0xfb33, 0xffdb, 0xfffb, 0xdc9c, 0xfbb4, 0xffbd, 0xfffc},
                         {0xc140, 0xfc34, 0xffe3, 0xfffc, 0xe4a4, 0xfcb5, 0xffcd, 0xfffd},
                         {0xd150, 0xfd35, 0xffeb, 0xfffd, 0xecac, 0xfdb6, 0xffdd, 0xfffe},
                         {0xe160, 0xfe36, 0xfff3, 0xfffe, 0xf4b4, 0xfeb7, 0xffed, 0xffff},
                         {0xf170, 0xff37, 0xfffb, 0xffff, 0xfcbc, 0xffb8, 0xfffd, 0x0000}},
                        {{0x8302'8100, 0xffc3'8343, 0xffff'c585, 0xffff'ff8f},
                         {0x9312'9110, 0xffcb'8b4b, 0xffff'cd8d, 0xffff'ff9f},
                         {0xa322'a120, 0xffd3'9353, 0xffff'd595, 0xffff'ffaf},
                         {0xb332'b130, 0xffdb'9b5b, 0xffff'dd9d, 0xffff'ffbf},
                         {0xc342'c140, 0xffe3'a363, 0xffff'e5a5, 0xffff'ffcf},
                         {0xd352'd150, 0xffeb'ab6b, 0xffff'edad, 0xffff'ffdf},
                         {0xe362'e160, 0xfff3'b373, 0xffff'f5b5, 0xffff'ffef},
                         {0xf372'f170, 0xfffb'bb7b, 0xffff'fdbd, 0xffff'ffff}},
                        {{0x8706'8504'8302'8100, 0xffff'c787'4686'4585},
                         {0xffff'ffff'9716'9515, 0xffff'ffff'ffff'cf8f},
                         {0xa726'a524'a322'a120, 0xffff'd797'5696'5595},
                         {0xffff'ffff'b736'b535, 0xffff'ffff'ffff'df9f},
                         {0xc746'c544'c342'c140, 0xffff'e7a7'66a6'65a5},
                         {0xffff'ffff'd756'd555, 0xffff'ffff'ffff'efaf},
                         {0xe766'e564'e362'e160, 0xffff'f7b7'76b6'75b5},
                         {0xffff'ffff'f776'f575, 0xffff'ffff'ffff'ffbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravx,
                        ExecMaskedVssravx,
                        {{0, 224, 1, 225, 1, 225, 2, 226, 2, 226, 3, 227, 3, 227, 4, 228},
                         {4, 228, 5, 229, 5, 229, 6, 230, 6, 230, 7, 231, 7, 231, 8, 232},
                         {8, 232, 9, 233, 9, 233, 10, 234, 10, 234, 11, 235, 11, 235, 12, 236},
                         {12, 236, 13, 237, 13, 237, 14, 238, 14, 238, 15, 239, 15, 239, 16, 240},
                         {16, 240, 17, 241, 17, 241, 18, 242, 18, 242, 19, 243, 19, 243, 20, 244},
                         {20, 244, 21, 245, 21, 245, 22, 246, 22, 246, 23, 247, 23, 247, 24, 248},
                         {24, 248, 25, 249, 25, 249, 26, 250, 26, 250, 27, 251, 27, 251, 28, 252},
                         {28, 252, 29, 253, 29, 253, 30, 254, 30, 254, 31, 255, 31, 255, 32, 0}},
                        {{0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3, 0xffe4},
                         {0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7, 0xffe8},
                         {0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb, 0xffec},
                         {0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef, 0xfff0},
                         {0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3, 0xfff4},
                         {0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7, 0xfff8},
                         {0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb, 0xfffc},
                         {0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff, 0x0000}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffff'ffff'ffe1'c1a1, 0xffff'ffff'ffe3'c3a3},
                         {0xffff'ffff'ffe5'c5a5, 0xffff'ffff'ffe7'c7a7},
                         {0xffff'ffff'ffe9'c9a9, 0xffff'ffff'ffeb'cbab},
                         {0xffff'ffff'ffed'cdad, 0xffff'ffff'ffef'cfaf},
                         {0xffff'ffff'fff1'd1b1, 0xffff'ffff'fff3'd3b3},
                         {0xffff'ffff'fff5'd5b5, 0xffff'ffff'fff7'd7b7},
                         {0xffff'ffff'fff9'd9b9, 0xffff'ffff'fffb'dbbb},
                         {0xffff'ffff'fffd'ddbd, 0xffff'ffff'ffff'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravi,
                        ExecMaskedVssravi,
                        {{0, 224, 1, 225, 1, 225, 2, 226, 2, 226, 3, 227, 3, 227, 4, 228},
                         {4, 228, 5, 229, 5, 229, 6, 230, 6, 230, 7, 231, 7, 231, 8, 232},
                         {8, 232, 9, 233, 9, 233, 10, 234, 10, 234, 11, 235, 11, 235, 12, 236},
                         {12, 236, 13, 237, 13, 237, 14, 238, 14, 238, 15, 239, 15, 239, 16, 240},
                         {16, 240, 17, 241, 17, 241, 18, 242, 18, 242, 19, 243, 19, 243, 20, 244},
                         {20, 244, 21, 245, 21, 245, 22, 246, 22, 246, 23, 247, 23, 247, 24, 248},
                         {24, 248, 25, 249, 25, 249, 26, 250, 26, 250, 27, 251, 27, 251, 28, 252},
                         {28, 252, 29, 253, 29, 253, 30, 254, 30, 254, 31, 255, 31, 255, 32, 0}},
                        {{0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3, 0xffe4},
                         {0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7, 0xffe8},
                         {0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb, 0xffec},
                         {0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef, 0xfff0},
                         {0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3, 0xfff4},
                         {0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7, 0xfff8},
                         {0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb, 0xfffc},
                         {0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff, 0x0000}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffe1'c1a1'4120'c0a0, 0xffe3'c3a3'4322'c2a2},
                         {0xffe5'c5a5'4524'c4a4, 0xffe7'c7a7'4726'c6a6},
                         {0xffe9'c9a9'4928'c8a8, 0xffeb'cbab'4b2a'caaa},
                         {0xffed'cdad'4d2c'ccac, 0xffef'cfaf'4f2e'ceae},
                         {0xfff1'd1b1'5130'd0b0, 0xfff3'd3b3'5332'd2b2},
                         {0xfff5'd5b5'5534'd4b4, 0xfff7'd7b7'5736'd6b6},
                         {0xfff9'd9b9'5938'd8b8, 0xfffb'dbbb'5b3a'daba},
                         {0xfffd'ddbd'5d3c'dcbc, 0xffff'dfbf'5f3e'debe}},
                        kVectorCalculationsSource);
  asm volatile("csrw vxrm, %0\n\t" ::"r"(vxrm));
}

TEST(InlineAsmTestRiscv64, TestROD) {
  uint64_t vxrm;
  asm volatile(
      "csrr %0, vxrm\n\t"
      "csrwi vxrm, %c1\n\t"
      : "=r"(vxrm)
      : "i"(VXRMFlags::ROD));
  TestVectorInstruction(
      ExecVaadduvv,
      ExecMaskedVaadduvv,
      {{0, 137, 3, 141, 7, 143, 9, 147, 13, 133, 15, 137, 18, 139, 21, 143},
       {24, 161, 27, 165, 31, 167, 33, 171, 37, 157, 39, 161, 42, 163, 45, 167},
       {48, 185, 51, 189, 55, 191, 57, 195, 61, 181, 63, 185, 66, 187, 69, 191},
       {72, 209, 75, 213, 79, 215, 81, 219, 85, 205, 87, 209, 90, 211, 93, 215},
       {96, 105, 99, 109, 103, 111, 105, 115, 109, 101, 111, 105, 114, 107, 117, 111},
       {120, 129, 123, 133, 127, 135, 129, 139, 133, 125, 135, 129, 138, 131, 141, 135},
       {144, 153, 147, 157, 151, 159, 153, 163, 157, 149, 159, 153, 162, 155, 165, 159},
       {168, 177, 171, 181, 175, 183, 177, 187, 181, 173, 183, 177, 186, 179, 189, 183}},
      {{0x8980, 0x8c83, 0x8f87, 0x9289, 0x858d, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79f, 0xaaa1, 0x9da5, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb7, 0xc2b9, 0xb5bd, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7cf, 0xdad1, 0xcdd5, 0xd0d7, 0xd3da, 0xd6dd},
       {0x69e0, 0x6ce3, 0x6fe7, 0x72e9, 0x65ed, 0x68ef, 0x6bf2, 0x6ef5},
       {0x81f8, 0x84fb, 0x87ff, 0x8b01, 0x7e05, 0x8107, 0x840a, 0x870d},
       {0x9a10, 0x9d13, 0xa017, 0xa319, 0x961d, 0x991f, 0x9c22, 0x9f25},
       {0xb228, 0xb52b, 0xb82f, 0xbb31, 0xae35, 0xb137, 0xb43a, 0xb73d}},
      {{0x8c83'8980, 0x9289'8f87, 0x888f'858d, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79f, 0xa0a7'9da5, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb7, 0xb8bf'b5bd, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7cf, 0xd0d7'cdd5, 0xd6dd'd3da},
       {0x6ce3'69e0, 0x72e9'6fe7, 0x68ef'65ed, 0x6ef5'6bf2},
       {0x84fb'81f8, 0x8b01'87ff, 0x8107'7e05, 0x870d'840a},
       {0x9d13'9a10, 0xa319'a017, 0x991f'961d, 0x9f25'9c22},
       {0xb52b'b228, 0xbb31'b82f, 0xb137'ae35, 0xb73d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858d},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da5},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bd},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd5},
       {0x72e9'6fe6'ece3'69e0, 0x6ef5'6bf2'68ef'65ed},
       {0x8b01'87ff'04fb'81f8, 0x870d'840a'8107'7e05},
       {0xa319'a017'1d13'9a10, 0x9f25'9c22'991f'961d},
       {0xbb31'b82f'352b'b228, 0xb73d'b43a'b137'ae35}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaadduvx,
      ExecMaskedVaadduvx,
      {{85, 149, 86, 151, 87, 151, 88, 153, 89, 153, 90, 155, 91, 155, 92, 157},
       {93, 157, 94, 159, 95, 159, 96, 161, 97, 161, 98, 163, 99, 163, 100, 165},
       {101, 165, 102, 167, 103, 167, 104, 169, 105, 169, 106, 171, 107, 171, 108, 173},
       {109, 173, 110, 175, 111, 175, 112, 177, 113, 177, 114, 179, 115, 179, 116, 181},
       {117, 181, 118, 183, 119, 183, 120, 185, 121, 185, 122, 187, 123, 187, 124, 189},
       {125, 189, 126, 191, 127, 191, 128, 193, 129, 193, 130, 195, 131, 195, 132, 197},
       {133, 197, 134, 199, 135, 199, 136, 201, 137, 201, 138, 203, 139, 203, 140, 205},
       {141, 205, 142, 207, 143, 207, 144, 209, 145, 209, 146, 211, 147, 211, 148, 213}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvv,
      ExecMaskedVaaddvv,
      {{0, 137, 3, 141, 7, 143, 9, 147, 13, 133, 15, 137, 18, 139, 21, 143},
       {24, 161, 27, 165, 31, 167, 33, 171, 37, 157, 39, 161, 42, 163, 45, 167},
       {48, 185, 51, 189, 55, 191, 57, 195, 61, 181, 63, 185, 66, 187, 69, 191},
       {72, 209, 75, 213, 79, 215, 81, 219, 85, 205, 87, 209, 90, 211, 93, 215},
       {224, 233, 227, 237, 231, 239, 233, 243, 237, 229, 239, 233, 242, 235, 245, 239},
       {248, 1, 251, 5, 255, 7, 1, 11, 5, 253, 7, 1, 10, 3, 13, 7},
       {16, 25, 19, 29, 23, 31, 25, 35, 29, 21, 31, 25, 34, 27, 37, 31},
       {40, 49, 43, 53, 47, 55, 49, 59, 53, 45, 55, 49, 58, 51, 61, 55}},
      {{0x8980, 0x8c83, 0x8f87, 0x9289, 0x858d, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79f, 0xaaa1, 0x9da5, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb7, 0xc2b9, 0xb5bd, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7cf, 0xdad1, 0xcdd5, 0xd0d7, 0xd3da, 0xd6dd},
       {0xe9e0, 0xece3, 0xefe7, 0xf2e9, 0xe5ed, 0xe8ef, 0xebf2, 0xeef5},
       {0x01f8, 0x04fb, 0x07ff, 0x0b01, 0xfe05, 0x0107, 0x040a, 0x070d},
       {0x1a10, 0x1d13, 0x2017, 0x2319, 0x161d, 0x191f, 0x1c22, 0x1f25},
       {0x3228, 0x352b, 0x382f, 0x3b31, 0x2e35, 0x3137, 0x343a, 0x373d}},
      {{0x8c83'8980, 0x9289'8f87, 0x888f'858d, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79f, 0xa0a7'9da5, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb7, 0xb8bf'b5bd, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7cf, 0xd0d7'cdd5, 0xd6dd'd3da},
       {0xece3'69e0, 0xf2e9'6fe7, 0xe8ef'65ed, 0xeef5'6bf2},
       {0x04fb'81f8, 0x0b01'87ff, 0x0107'7e05, 0x070d'840a},
       {0x1d13'9a10, 0x2319'a017, 0x191f'961d, 0x1f25'9c22},
       {0x352b'b228, 0x3b31'b82f, 0x3137'ae35, 0x373d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858d},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da5},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bd},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd5},
       {0xf2e9'6fe6'ece3'69e0, 0xeef5'6bf2'68ef'65ed},
       {0x0b01'87ff'04fb'81f8, 0x070d'840a'8107'7e05},
       {0x2319'a017'1d13'9a10, 0x1f25'9c22'991f'961d},
       {0x3b31'b82f'352b'b228, 0x373d'b43a'b137'ae35}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvx,
      ExecMaskedVaaddvx,
      {{213, 149, 214, 151, 215, 151, 216, 153, 217, 153, 218, 155, 219, 155, 220, 157},
       {221, 157, 222, 159, 223, 159, 224, 161, 225, 161, 226, 163, 227, 163, 228, 165},
       {229, 165, 230, 167, 231, 167, 232, 169, 233, 169, 234, 171, 235, 171, 236, 173},
       {237, 173, 238, 175, 239, 175, 240, 177, 241, 177, 242, 179, 243, 179, 244, 181},
       {245, 181, 246, 183, 247, 183, 248, 185, 249, 185, 250, 187, 251, 187, 252, 189},
       {253, 189, 254, 191, 255, 191, 0, 193, 1, 193, 2, 195, 3, 195, 4, 197},
       {5, 197, 6, 199, 7, 199, 8, 201, 9, 201, 10, 203, 11, 203, 12, 205},
       {13, 205, 14, 207, 15, 207, 16, 209, 17, 209, 18, 211, 19, 211, 20, 213}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvv,
      ExecMaskedVasubuvv,
      {{0, 247, 255, 247, 253, 245, 253, 245, 251, 3, 251, 3, 250, 1, 249, 1},
       {248, 239, 247, 239, 245, 237, 245, 237, 243, 251, 243, 251, 242, 249, 241, 249},
       {240, 231, 239, 231, 237, 229, 237, 229, 235, 243, 235, 243, 234, 241, 233, 241},
       {232, 223, 231, 223, 229, 221, 229, 221, 227, 235, 227, 235, 226, 233, 225, 233},
       {224, 87, 223, 87, 221, 85, 221, 85, 219, 99, 219, 99, 218, 97, 217, 97},
       {216, 79, 215, 79, 213, 77, 213, 77, 211, 91, 211, 91, 210, 89, 209, 89},
       {208, 71, 207, 71, 205, 69, 205, 69, 203, 83, 203, 83, 202, 81, 201, 81},
       {200, 63, 199, 63, 197, 61, 197, 61, 195, 75, 195, 75, 194, 73, 193, 73}},
      {{0xf780, 0xf67f, 0xf57d, 0xf47d, 0x037b, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed75, 0xec75, 0xfb73, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56d, 0xe46d, 0xf36b, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd65, 0xdc65, 0xeb63, 0xea63, 0xe962, 0xe861},
       {0x5760, 0x565f, 0x555d, 0x545d, 0x635b, 0x625b, 0x615a, 0x6059},
       {0x4f58, 0x4e57, 0x4d55, 0x4c55, 0x5b53, 0x5a53, 0x5952, 0x5851},
       {0x4750, 0x464f, 0x454d, 0x444d, 0x534b, 0x524b, 0x514a, 0x5049},
       {0x3f48, 0x3e47, 0x3d45, 0x3c45, 0x4b43, 0x4a43, 0x4942, 0x4841}},
      {{0xf67e'f780, 0xf47c'f57d, 0x027b'037b, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed75, 0xfa72'fb73, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56d, 0xf26a'f36b, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd65, 0xea62'eb63, 0xe860'e962},
       {0x565f'5760, 0x545d'555d, 0x625b'635b, 0x6059'615a},
       {0x4e57'4f58, 0x4c55'4d55, 0x5a53'5b53, 0x5851'5952},
       {0x464f'4750, 0x444d'454d, 0x524b'534b, 0x5049'514a},
       {0x3e47'3f48, 0x3c45'3d45, 0x4a43'4b43, 0x4841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037b},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb73},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36b},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb63},
       {0x545d'555d'd65f'5760, 0x6059'615a'625b'635b},
       {0x4c55'4d55'ce57'4f58, 0x5851'5952'5a53'5b53},
       {0x444d'454d'c64f'4750, 0x5049'514a'524b'534b},
       {0x3c45'3d45'be47'3f48, 0x4841'4942'4a43'4b43}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvx,
      ExecMaskedVasubuvx,
      {{171, 235, 172, 237, 173, 237, 174, 239, 175, 239, 176, 241, 177, 241, 178, 243},
       {179, 243, 180, 245, 181, 245, 182, 247, 183, 247, 184, 249, 185, 249, 186, 251},
       {187, 251, 188, 253, 189, 253, 190, 255, 191, 255, 192, 1, 193, 1, 194, 3},
       {195, 3, 196, 5, 197, 5, 198, 7, 199, 7, 200, 9, 201, 9, 202, 11},
       {203, 11, 204, 13, 205, 13, 206, 15, 207, 15, 208, 17, 209, 17, 210, 19},
       {211, 19, 212, 21, 213, 21, 214, 23, 215, 23, 216, 25, 217, 25, 218, 27},
       {219, 27, 220, 29, 221, 29, 222, 31, 223, 31, 224, 33, 225, 33, 226, 35},
       {227, 35, 228, 37, 229, 37, 230, 39, 231, 39, 232, 41, 233, 41, 234, 43}},
      {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
       {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
       {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
       {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
       {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
       {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
       {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
       {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
      {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
       {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
       {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
       {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
       {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
       {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
       {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
       {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
      {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
       {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
       {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
       {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
       {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
       {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
       {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
       {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubvv,
      ExecMaskedVasubvv,
      {{0, 247, 255, 247, 253, 245, 253, 245, 251, 3, 251, 3, 250, 1, 249, 1},
       {248, 239, 247, 239, 245, 237, 245, 237, 243, 251, 243, 251, 242, 249, 241, 249},
       {240, 231, 239, 231, 237, 229, 237, 229, 235, 243, 235, 243, 234, 241, 233, 241},
       {232, 223, 231, 223, 229, 221, 229, 221, 227, 235, 227, 235, 226, 233, 225, 233},
       {96, 215, 95, 215, 93, 213, 93, 213, 91, 227, 91, 227, 90, 225, 89, 225},
       {88, 207, 87, 207, 85, 205, 85, 205, 83, 219, 83, 219, 82, 217, 81, 217},
       {80, 199, 79, 199, 77, 197, 77, 197, 75, 211, 75, 211, 74, 209, 73, 209},
       {72, 191, 71, 191, 69, 189, 69, 189, 67, 203, 67, 203, 66, 201, 65, 201}},
      {{0xf780, 0xf67f, 0xf57d, 0xf47d, 0x037b, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed75, 0xec75, 0xfb73, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56d, 0xe46d, 0xf36b, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd65, 0xdc65, 0xeb63, 0xea63, 0xe962, 0xe861},
       {0xd760, 0xd65f, 0xd55d, 0xd45d, 0xe35b, 0xe25b, 0xe15a, 0xe059},
       {0xcf58, 0xce57, 0xcd55, 0xcc55, 0xdb53, 0xda53, 0xd952, 0xd851},
       {0xc750, 0xc64f, 0xc54d, 0xc44d, 0xd34b, 0xd24b, 0xd14a, 0xd049},
       {0xbf48, 0xbe47, 0xbd45, 0xbc45, 0xcb43, 0xca43, 0xc942, 0xc841}},
      {{0xf67e'f780, 0xf47c'f57d, 0x027b'037b, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed75, 0xfa72'fb73, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56d, 0xf26a'f36b, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd65, 0xea62'eb63, 0xe860'e962},
       {0xd65f'5760, 0xd45d'555d, 0xe25b'635b, 0xe059'615a},
       {0xce57'4f58, 0xcc55'4d55, 0xda53'5b53, 0xd851'5952},
       {0xc64f'4750, 0xc44d'454d, 0xd24b'534b, 0xd049'514a},
       {0xbe47'3f48, 0xbc45'3d45, 0xca43'4b43, 0xc841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037b},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb73},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36b},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb63},
       {0xd45d'555d'd65f'5760, 0xe059'615a'625b'635b},
       {0xcc55'4d55'ce57'4f58, 0xd851'5952'5a53'5b53},
       {0xc44d'454d'c64f'4750, 0xd049'514a'524b'534b},
       {0xbc45'3d45'be47'3f48, 0xc841'4942'4a43'4b43}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVasubvx,
                        ExecMaskedVasubvx,
                        {{43, 235, 44, 237, 45, 237, 46, 239, 47, 239, 48, 241, 49, 241, 50, 243},
                         {51, 243, 52, 245, 53, 245, 54, 247, 55, 247, 56, 249, 57, 249, 58, 251},
                         {59, 251, 60, 253, 61, 253, 62, 255, 63, 255, 64, 1, 65, 1, 66, 3},
                         {67, 3, 68, 5, 69, 5, 70, 7, 71, 7, 72, 9, 73, 9, 74, 11},
                         {75, 11, 76, 13, 77, 13, 78, 15, 79, 15, 80, 17, 81, 17, 82, 19},
                         {83, 19, 84, 21, 85, 21, 86, 23, 87, 23, 88, 25, 89, 25, 90, 27},
                         {91, 27, 92, 29, 93, 29, 94, 31, 95, 31, 96, 33, 97, 33, 98, 35},
                         {99, 35, 100, 37, 101, 37, 102, 39, 103, 39, 104, 41, 105, 41, 106, 43}},
                        {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
                         {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
                         {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
                         {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
                         {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
                         {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
                         {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
                         {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
                        {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
                         {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
                         {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
                         {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
                         {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
                         {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
                         {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
                         {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
                        {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
                         {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
                         {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
                         {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
                         {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
                         {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
                         {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
                         {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
                        kVectorCalculationsSource);
  TestNarrowingVectorInstruction(ExecVnclipuwi,
                                 ExecMaskedVnclipuwi,
                                 {{33, 33, 33, 33, 35, 35, 35, 35, 37, 37, 37, 37, 39, 39, 39, 39},
                                  {41, 41, 41, 41, 43, 43, 43, 43, 45, 45, 45, 45, 47, 47, 47, 47},
                                  {49, 49, 49, 49, 51, 51, 51, 51, 53, 53, 53, 53, 55, 55, 55, 55},
                                  {57, 57, 57, 57, 59, 59, 59, 59, 61, 61, 61, 61, 63, 63, 63, 63}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                                 kVectorCalculationsSource);
  TestNarrowingVectorInstruction(
      ExecVnclipwi,
      ExecMaskedVnclipwi,
      {{225, 225, 225, 225, 227, 227, 227, 227, 229, 229, 229, 229, 231, 231, 231, 231},
       {233, 233, 233, 233, 235, 235, 235, 235, 237, 237, 237, 237, 239, 239, 239, 239},
       {241, 241, 241, 241, 243, 243, 243, 243, 245, 245, 245, 245, 247, 247, 247, 247},
       {249, 249, 249, 249, 251, 251, 251, 251, 253, 253, 253, 253, 255, 255, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(ExecVnclipuwx,
                                 ExecMaskedVnclipuwx,
                                 {{33, 33, 33, 33, 35, 35, 35, 35, 37, 37, 37, 37, 39, 39, 39, 39},
                                  {41, 41, 41, 41, 43, 43, 43, 43, 45, 45, 45, 45, 47, 47, 47, 47},
                                  {49, 49, 49, 49, 51, 51, 51, 51, 53, 53, 53, 53, 55, 55, 55, 55},
                                  {57, 57, 57, 57, 59, 59, 59, 59, 61, 61, 61, 61, 63, 63, 63, 63}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0x0021'c1a1, 0x0023'c3a3, 0x0025'c5a5, 0x0027'c7a7},
                                  {0x0029'c9a9, 0x002b'cbab, 0x002d'cdad, 0x002f'cfaf},
                                  {0x0031'd1b1, 0x0033'd3b3, 0x0035'd5b5, 0x0037'd7b7},
                                  {0x0039'd9b9, 0x003b'dbbb, 0x003d'ddbd, 0x003f'dfbf}},
                                 kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwx,
      ExecMaskedVnclipwx,
      {{225, 225, 225, 225, 227, 227, 227, 227, 229, 229, 229, 229, 231, 231, 231, 231},
       {233, 233, 233, 233, 235, 235, 235, 235, 237, 237, 237, 237, 239, 239, 239, 239},
       {241, 241, 241, 241, 243, 243, 243, 243, 245, 245, 245, 245, 247, 247, 247, 247},
       {249, 249, 249, 249, 251, 251, 251, 251, 253, 253, 253, 253, 255, 255, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0xffe1'c1a1, 0xffe3'c3a3, 0xffe5'c5a5, 0xffe7'c7a7},
       {0xffe9'c9a9, 0xffeb'cbab, 0xffed'cdad, 0xffef'cfaf},
       {0xfff1'd1b1, 0xfff3'd3b3, 0xfff5'd5b5, 0xfff7'd7b7},
       {0xfff9'd9b9, 0xfffb'dbbb, 0xfffd'ddbd, 0xffff'dfbf}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipuwv,
      ExecMaskedVnclipuwv,
      {{255, 255, 255, 255, 69, 35, 9, 3, 255, 255, 255, 255, 153, 39, 9, 3},
       {255, 255, 255, 255, 85, 43, 11, 3, 255, 255, 255, 255, 185, 47, 11, 3},
       {255, 255, 255, 255, 101, 51, 13, 3, 255, 255, 255, 255, 217, 55, 13, 3},
       {255, 255, 255, 255, 117, 59, 15, 3, 255, 255, 255, 255, 249, 63, 15, 3}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0x4989, 0x0971, 0x009b, 0x0009},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x5999, 0x0b73, 0x00bb, 0x000b},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x69a9, 0x0d75, 0x00db, 0x000d},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x79b9, 0x0f77, 0x00fb, 0x000f}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xa726'a525, 0x0057'9757, 0x0000'5b9b, 0x0000'00bf},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xe766'e565, 0x0077'b777, 0x0000'7bbb, 0x0000'00ff}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwv,
      ExecMaskedVnclipwv,
      {{128, 128, 128, 128, 197, 227, 249, 255, 128, 128, 128, 128, 153, 231, 249, 255},
       {128, 128, 128, 128, 213, 235, 251, 255, 128, 128, 128, 128, 185, 239, 251, 255},
       {128, 128, 128, 128, 229, 243, 253, 255, 128, 128, 128, 128, 217, 247, 253, 255},
       {128, 128, 128, 157, 245, 251, 255, 255, 128, 128, 128, 221, 249, 255, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0xc989, 0xf971, 0xff9b, 0xfff9},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xd999, 0xfb73, 0xffbb, 0xfffb},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xe9a9, 0xfd75, 0xffdb, 0xfffd},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xf9b9, 0xff77, 0xfffb, 0xffff}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xa726'a525, 0xffd7'9757, 0xffff'db9b, 0xffff'ffbf},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xe766'e565, 0xfff7'b777, 0xffff'fbbb, 0xffff'ffff}},
      kVectorCalculationsSource);

  TestVectorInstruction(
      ExecVsmulvv,
      ExecMaskedVsmulvv,
      {{0, 109, 1, 103, 1, 99, 1, 93, 1, 117, 1, 111, 3, 107, 3, 101},
       {4, 67, 5, 63, 7, 59, 7, 55, 9, 75, 11, 71, 13, 67, 15, 63},
       {16, 35, 19, 31, 21, 27, 23, 23, 25, 43, 27, 39, 31, 35, 33, 31},
       {36, 9, 39, 7, 43, 3, 45, 1, 49, 17, 53, 15, 57, 11, 61, 9},
       {192, 247, 193, 245, 193, 245, 193, 243, 193, 255, 193, 253, 195, 253, 195, 251},
       {196, 237, 197, 237, 199, 237, 199, 237, 201, 245, 203, 245, 205, 245, 207, 245},
       {208, 237, 211, 237, 213, 237, 215, 237, 217, 245, 219, 245, 223, 245, 225, 245},
       {228, 243, 231, 245, 235, 245, 237, 247, 241, 251, 245, 253, 249, 253, 253, 255}},
      {{0x6d24, 0x677f, 0x61f9, 0x5c95, 0x750d, 0x6f69, 0x69e3, 0x647f},
       {0x437f, 0x3edb, 0x3a57, 0x35f5, 0x4b6b, 0x46c9, 0x4245, 0x3de3},
       {0x21e9, 0x1e47, 0x1ac5, 0x1765, 0x29d9, 0x2639, 0x22b9, 0x1f57},
       {0x0863, 0x05c5, 0x0345, 0x00e5, 0x1059, 0x0db9, 0x0b3b, 0x08dd},
       {0xf6ef, 0xf551, 0xf3d3, 0xf277, 0xfee7, 0xfd4b, 0xfbcd, 0xfa71},
       {0xed89, 0xeced, 0xec71, 0xec17, 0xf585, 0xf4eb, 0xf471, 0xf415},
       {0xec33, 0xec9b, 0xed21, 0xedc7, 0xf435, 0xf49b, 0xf523, 0xf5cb},
       {0xf2ef, 0xf457, 0xf5df, 0xf789, 0xfaf3, 0xfc5d, 0xfde5, 0xff8f}},
      {{0x677d'76af, 0x5c93'1931, 0x6f67'3831, 0x647d'dbb7},
       {0x3eda'09c7, 0x35f3'b251, 0x46c7'cf51, 0x3de2'78dd},
       {0x1e46'b4fd, 0x1764'638f, 0x2638'7e8f, 0x1f57'2e25},
       {0x05c3'7855, 0x00e5'2cef, 0x0db9'45ef, 0x08db'fb8d},
       {0xf550'cd47, 0xf276'7fe1, 0xfd4a'8ed9, 0xfa71'4277},
       {0xeced'a0bf, 0xec17'5961, 0xf4eb'6659, 0xf416'1fff},
       {0xec9a'8c57, 0xedc8'4b01, 0xf49c'55f9, 0xf5cb'15a7},
       {0xf457'900d, 0xf789'54c1, 0xfc5d'5db9, 0xff90'236d}},
      {{0x5c93'192f'ccd4'7781, 0x647d'dbb5'bb66'23af},
       {0x35f3'b24f'43d0'aa39, 0x3de2'78dd'1a4e'4257},
       {0x1764'638e'e2fd'152f, 0x1f57'2e24'a166'993d},
       {0x00e5'2cee'aa59'b867, 0x08db'fb8c'50af'2865},
       {0xf276'7fe1'80cf'f441, 0xfa71'4276'eef1'1fff},
       {0xec17'5961'584c'a799, 0xf416'1ffe'ae59'bf47},
       {0xedc8'4b01'57f9'9331, 0xf5cb'15a6'95f2'96cf},
       {0xf789'54c1'7fd6'b709, 0xff90'236e'a5bb'a697}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVsmulvx,
                        ExecMaskedVsmulvx,
                        {{0, 85, 255, 83, 253, 83, 251, 81, 251, 79, 249, 79, 247, 77, 247, 75},
                         {245, 75, 243, 73, 243, 71, 241, 71, 239, 69, 239, 67, 237, 67, 235, 65},
                         {235, 63, 233, 63, 231, 61, 231, 59, 229, 59, 227, 57, 227, 55, 225, 55},
                         {223, 53, 223, 51, 221, 51, 219, 49, 219, 47, 217, 47, 215, 45, 215, 43},
                         {213, 43, 211, 41, 211, 39, 209, 39, 207, 37, 207, 35, 205, 35, 203, 33},
                         {203, 31, 201, 31, 199, 29, 199, 27, 197, 27, 195, 25, 195, 23, 193, 23},
                         {191, 21, 191, 19, 189, 19, 187, 17, 187, 15, 185, 15, 183, 13, 183, 11},
                         {181, 11, 179, 9, 179, 7, 177, 7, 175, 5, 175, 3, 173, 3, 171, 1}},
                        {{0x54ab, 0x5355, 0x51fd, 0x50a7, 0x4f51, 0x4df9, 0x4ca3, 0x4b4d},
                         {0x49f5, 0x489f, 0x4749, 0x45f1, 0x449b, 0x4345, 0x41ed, 0x4097},
                         {0x3f41, 0x3de9, 0x3c93, 0x3b3d, 0x39e5, 0x388f, 0x3739, 0x35e1},
                         {0x348b, 0x3335, 0x31dd, 0x3087, 0x2f31, 0x2dd9, 0x2c83, 0x2b2d},
                         {0x29d5, 0x287f, 0x2729, 0x25d1, 0x247b, 0x2325, 0x21cd, 0x2077},
                         {0x1f21, 0x1dc9, 0x1c73, 0x1b1d, 0x19c5, 0x186f, 0x1719, 0x15c1},
                         {0x146b, 0x1315, 0x11bd, 0x1067, 0x0f11, 0x0db9, 0x0c63, 0x0b0d},
                         {0x09b5, 0x085f, 0x0709, 0x05b1, 0x045b, 0x0305, 0x01ad, 0x0057}},
                        {{0x5353'aa01, 0x50a6'51fd, 0x4df8'f9fb, 0x4b4b'a1f9},
                         {0x489e'49f5, 0x45f0'f1f3, 0x4343'99f1, 0x4096'41ed},
                         {0x3de8'e9eb, 0x3b3b'91e9, 0x388e'39e5, 0x35e0'e1e3},
                         {0x3333'89e1, 0x3086'31dd, 0x2dd8'd9db, 0x2b2b'81d9},
                         {0x287e'29d5, 0x25d0'd1d3, 0x2323'79d1, 0x2076'21cd},
                         {0x1dc8'c9cb, 0x1b1b'71c9, 0x186e'19c5, 0x15c0'c1c3},
                         {0x1313'69c1, 0x1066'11bd, 0x0db8'b9bb, 0x0b0b'61b9},
                         {0x085e'09b5, 0x05b0'b1b3, 0x0303'59b1, 0x0056'01ad}},
                        {{0x50a6'51fc'fdfe'54ab, 0x4b4b'a1f7'a34e'4f51},
                         {0x45f0'f1f2'489e'49f5, 0x4096'41ec'edee'449b},
                         {0x3b3b'91e7'933e'3f41, 0x35e0'e1e2'388e'39e5},
                         {0x3086'31dc'ddde'348b, 0x2b2b'81d7'832e'2f31},
                         {0x25d0'd1d2'287e'29d5, 0x2076'21cc'cdce'247b},
                         {0x1b1b'71c7'731e'1f21, 0x15c0'c1c2'186e'19c5},
                         {0x1066'11bc'bdbe'146b, 0x0b0b'61b7'630e'0f11},
                         {0x05b0'b1b2'085e'09b5, 0x0056'01ac'adae'045b}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvv,
                        ExecMaskedVssrlvv,
                        {{0, 33, 1, 3, 2, 33, 1, 3, 4, 35, 1, 3, 12, 35, 1, 3},
                         {16, 37, 1, 3, 10, 37, 1, 3, 12, 39, 1, 3, 28, 39, 1, 3},
                         {32, 41, 3, 3, 18, 41, 3, 3, 20, 43, 3, 3, 44, 43, 3, 3},
                         {48, 45, 3, 3, 26, 45, 3, 3, 28, 47, 3, 3, 60, 47, 3, 3},
                         {64, 49, 5, 3, 34, 49, 5, 3, 36, 51, 5, 3, 76, 51, 5, 3},
                         {80, 53, 5, 3, 42, 53, 5, 3, 44, 55, 5, 3, 92, 55, 5, 3},
                         {96, 57, 7, 3, 50, 57, 7, 3, 52, 59, 7, 3, 108, 59, 7, 3},
                         {112, 61, 7, 3, 58, 61, 7, 3, 60, 63, 7, 3, 124, 63, 7, 3}},
                        {{0x8100, 0x0831, 0x0043, 0x0009, 0x4484, 0x08b1, 0x008d, 0x0009},
                         {0x9110, 0x0931, 0x004b, 0x0009, 0x4c8c, 0x09b1, 0x009d, 0x0009},
                         {0xa120, 0x0a33, 0x0053, 0x000b, 0x5494, 0x0ab3, 0x00ad, 0x000b},
                         {0xb130, 0x0b33, 0x005b, 0x000b, 0x5c9c, 0x0bb3, 0x00bd, 0x000b},
                         {0xc140, 0x0c35, 0x0063, 0x000d, 0x64a4, 0x0cb5, 0x00cd, 0x000d},
                         {0xd150, 0x0d35, 0x006b, 0x000d, 0x6cac, 0x0db5, 0x00dd, 0x000d},
                         {0xe160, 0x0e37, 0x0073, 0x000f, 0x74b4, 0x0eb7, 0x00ed, 0x000f},
                         {0xf170, 0x0f37, 0x007b, 0x000f, 0x7cbc, 0x0fb7, 0x00fd, 0x000f}},
                        {{0x8302'8100, 0x0043'8343, 0x0000'4585, 0x0000'008f},
                         {0x9312'9110, 0x004b'8b4b, 0x0000'4d8d, 0x0000'009f},
                         {0xa322'a120, 0x0053'9353, 0x0000'5595, 0x0000'00af},
                         {0xb332'b130, 0x005b'9b5b, 0x0000'5d9d, 0x0000'00bf},
                         {0xc342'c140, 0x0063'a363, 0x0000'65a5, 0x0000'00cf},
                         {0xd352'd150, 0x006b'ab6b, 0x0000'6dad, 0x0000'00df},
                         {0xe362'e160, 0x0073'b373, 0x0000'75b5, 0x0000'00ef},
                         {0xf372'f170, 0x007b'bb7b, 0x0000'7dbd, 0x0000'00ff}},
                        {{0x8706'8504'8302'8100, 0x0000'4787'4686'4585},
                         {0x0000'0000'9716'9515, 0x0000'0000'0000'4f8f},
                         {0xa726'a524'a322'a120, 0x0000'5797'5696'5595},
                         {0x0000'0000'b736'b535, 0x0000'0000'0000'5f9f},
                         {0xc746'c544'c342'c140, 0x0000'67a7'66a6'65a5},
                         {0x0000'0000'd756'd555, 0x0000'0000'0000'6faf},
                         {0xe766'e564'e362'e160, 0x0000'77b7'76b6'75b5},
                         {0x0000'0000'f776'f575, 0x0000'0000'0000'7fbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvx,
                        ExecMaskedVssrlvx,
                        {{0, 33, 1, 33, 1, 33, 1, 33, 2, 35, 3, 35, 3, 35, 3, 35},
                         {4, 37, 5, 37, 5, 37, 5, 37, 6, 39, 7, 39, 7, 39, 7, 39},
                         {8, 41, 9, 41, 9, 41, 9, 41, 10, 43, 11, 43, 11, 43, 11, 43},
                         {12, 45, 13, 45, 13, 45, 13, 45, 14, 47, 15, 47, 15, 47, 15, 47},
                         {16, 49, 17, 49, 17, 49, 17, 49, 18, 51, 19, 51, 19, 51, 19, 51},
                         {20, 53, 21, 53, 21, 53, 21, 53, 22, 55, 23, 55, 23, 55, 23, 55},
                         {24, 57, 25, 57, 25, 57, 25, 57, 26, 59, 27, 59, 27, 59, 27, 59},
                         {28, 61, 29, 61, 29, 61, 29, 61, 30, 63, 31, 63, 31, 63, 31, 63}},
                        {{0x0021, 0x0021, 0x0021, 0x0021, 0x0023, 0x0023, 0x0023, 0x0023},
                         {0x0025, 0x0025, 0x0025, 0x0025, 0x0027, 0x0027, 0x0027, 0x0027},
                         {0x0029, 0x0029, 0x0029, 0x0029, 0x002b, 0x002b, 0x002b, 0x002b},
                         {0x002d, 0x002d, 0x002d, 0x002d, 0x002f, 0x002f, 0x002f, 0x002f},
                         {0x0031, 0x0031, 0x0031, 0x0031, 0x0033, 0x0033, 0x0033, 0x0033},
                         {0x0035, 0x0035, 0x0035, 0x0035, 0x0037, 0x0037, 0x0037, 0x0037},
                         {0x0039, 0x0039, 0x0039, 0x0039, 0x003b, 0x003b, 0x003b, 0x003b},
                         {0x003d, 0x003d, 0x003d, 0x003d, 0x003f, 0x003f, 0x003f, 0x003f}},
                        {{0x0020'c0a1, 0x0021'c1a1, 0x0022'c2a3, 0x0023'c3a3},
                         {0x0024'c4a5, 0x0025'c5a5, 0x0026'c6a7, 0x0027'c7a7},
                         {0x0028'c8a9, 0x0029'c9a9, 0x002a'caab, 0x002b'cbab},
                         {0x002c'ccad, 0x002d'cdad, 0x002e'ceaf, 0x002f'cfaf},
                         {0x0030'd0b1, 0x0031'd1b1, 0x0032'd2b3, 0x0033'd3b3},
                         {0x0034'd4b5, 0x0035'd5b5, 0x0036'd6b7, 0x0037'd7b7},
                         {0x0038'd8b9, 0x0039'd9b9, 0x003a'dabb, 0x003b'dbbb},
                         {0x003c'dcbd, 0x003d'ddbd, 0x003e'debf, 0x003f'dfbf}},
                        {{0x0000'0000'0021'c1a1, 0x0000'0000'0023'c3a3},
                         {0x0000'0000'0025'c5a5, 0x0000'0000'0027'c7a7},
                         {0x0000'0000'0029'c9a9, 0x0000'0000'002b'cbab},
                         {0x0000'0000'002d'cdad, 0x0000'0000'002f'cfaf},
                         {0x0000'0000'0031'd1b1, 0x0000'0000'0033'd3b3},
                         {0x0000'0000'0035'd5b5, 0x0000'0000'0037'd7b7},
                         {0x0000'0000'0039'd9b9, 0x0000'0000'003b'dbbb},
                         {0x0000'0000'003d'ddbd, 0x0000'0000'003f'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvi,
                        ExecMaskedVssrlvi,
                        {{0, 33, 1, 33, 1, 33, 1, 33, 2, 35, 3, 35, 3, 35, 3, 35},
                         {4, 37, 5, 37, 5, 37, 5, 37, 6, 39, 7, 39, 7, 39, 7, 39},
                         {8, 41, 9, 41, 9, 41, 9, 41, 10, 43, 11, 43, 11, 43, 11, 43},
                         {12, 45, 13, 45, 13, 45, 13, 45, 14, 47, 15, 47, 15, 47, 15, 47},
                         {16, 49, 17, 49, 17, 49, 17, 49, 18, 51, 19, 51, 19, 51, 19, 51},
                         {20, 53, 21, 53, 21, 53, 21, 53, 22, 55, 23, 55, 23, 55, 23, 55},
                         {24, 57, 25, 57, 25, 57, 25, 57, 26, 59, 27, 59, 27, 59, 27, 59},
                         {28, 61, 29, 61, 29, 61, 29, 61, 30, 63, 31, 63, 31, 63, 31, 63}},
                        {{0x0021, 0x0021, 0x0021, 0x0021, 0x0023, 0x0023, 0x0023, 0x0023},
                         {0x0025, 0x0025, 0x0025, 0x0025, 0x0027, 0x0027, 0x0027, 0x0027},
                         {0x0029, 0x0029, 0x0029, 0x0029, 0x002b, 0x002b, 0x002b, 0x002b},
                         {0x002d, 0x002d, 0x002d, 0x002d, 0x002f, 0x002f, 0x002f, 0x002f},
                         {0x0031, 0x0031, 0x0031, 0x0031, 0x0033, 0x0033, 0x0033, 0x0033},
                         {0x0035, 0x0035, 0x0035, 0x0035, 0x0037, 0x0037, 0x0037, 0x0037},
                         {0x0039, 0x0039, 0x0039, 0x0039, 0x003b, 0x003b, 0x003b, 0x003b},
                         {0x003d, 0x003d, 0x003d, 0x003d, 0x003f, 0x003f, 0x003f, 0x003f}},
                        {{0x0020'c0a1, 0x0021'c1a1, 0x0022'c2a3, 0x0023'c3a3},
                         {0x0024'c4a5, 0x0025'c5a5, 0x0026'c6a7, 0x0027'c7a7},
                         {0x0028'c8a9, 0x0029'c9a9, 0x002a'caab, 0x002b'cbab},
                         {0x002c'ccad, 0x002d'cdad, 0x002e'ceaf, 0x002f'cfaf},
                         {0x0030'd0b1, 0x0031'd1b1, 0x0032'd2b3, 0x0033'd3b3},
                         {0x0034'd4b5, 0x0035'd5b5, 0x0036'd6b7, 0x0037'd7b7},
                         {0x0038'd8b9, 0x0039'd9b9, 0x003a'dabb, 0x003b'dbbb},
                         {0x003c'dcbd, 0x003d'ddbd, 0x003e'debf, 0x003f'dfbf}},
                        {{0x0021'c1a1'4120'c0a1, 0x0023'c3a3'4322'c2a3},
                         {0x0025'c5a5'4524'c4a5, 0x0027'c7a7'4726'c6a7},
                         {0x0029'c9a9'4928'c8a9, 0x002b'cbab'4b2a'caab},
                         {0x002d'cdad'4d2c'ccad, 0x002f'cfaf'4f2e'ceaf},
                         {0x0031'd1b1'5130'd0b1, 0x0033'd3b3'5332'd2b3},
                         {0x0035'd5b5'5534'd4b5, 0x0037'd7b7'5736'd6b7},
                         {0x0039'd9b9'5938'd8b9, 0x003b'dbbb'5b3a'dabb},
                         {0x003d'ddbd'5d3c'dcbd, 0x003f'dfbf'5f3e'debf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravv,
                        ExecMaskedVssravv,
                        {{0, 225, 1, 255, 2, 225, 1, 255, 4, 227, 1, 255, 12, 227, 1, 255},
                         {16, 229, 1, 255, 10, 229, 1, 255, 12, 231, 1, 255, 28, 231, 1, 255},
                         {32, 233, 3, 255, 18, 233, 3, 255, 20, 235, 3, 255, 44, 235, 3, 255},
                         {48, 237, 3, 255, 26, 237, 3, 255, 28, 239, 3, 255, 60, 239, 3, 255},
                         {64, 241, 5, 255, 34, 241, 5, 255, 36, 243, 5, 255, 76, 243, 5, 255},
                         {80, 245, 5, 255, 42, 245, 5, 255, 44, 247, 5, 255, 92, 247, 5, 255},
                         {96, 249, 7, 255, 50, 249, 7, 255, 52, 251, 7, 255, 108, 251, 7, 255},
                         {112, 253, 7, 255, 58, 253, 7, 255, 60, 255, 7, 255, 124, 255, 7, 255}},
                        {{0x8100, 0xf831, 0xffc3, 0xfff9, 0xc484, 0xf8b1, 0xff8d, 0xfff9},
                         {0x9110, 0xf931, 0xffcb, 0xfff9, 0xcc8c, 0xf9b1, 0xff9d, 0xfff9},
                         {0xa120, 0xfa33, 0xffd3, 0xfffb, 0xd494, 0xfab3, 0xffad, 0xfffb},
                         {0xb130, 0xfb33, 0xffdb, 0xfffb, 0xdc9c, 0xfbb3, 0xffbd, 0xfffb},
                         {0xc140, 0xfc35, 0xffe3, 0xfffd, 0xe4a4, 0xfcb5, 0xffcd, 0xfffd},
                         {0xd150, 0xfd35, 0xffeb, 0xfffd, 0xecac, 0xfdb5, 0xffdd, 0xfffd},
                         {0xe160, 0xfe37, 0xfff3, 0xffff, 0xf4b4, 0xfeb7, 0xffed, 0xffff},
                         {0xf170, 0xff37, 0xfffb, 0xffff, 0xfcbc, 0xffb7, 0xfffd, 0xffff}},
                        {{0x8302'8100, 0xffc3'8343, 0xffff'c585, 0xffff'ff8f},
                         {0x9312'9110, 0xffcb'8b4b, 0xffff'cd8d, 0xffff'ff9f},
                         {0xa322'a120, 0xffd3'9353, 0xffff'd595, 0xffff'ffaf},
                         {0xb332'b130, 0xffdb'9b5b, 0xffff'dd9d, 0xffff'ffbf},
                         {0xc342'c140, 0xffe3'a363, 0xffff'e5a5, 0xffff'ffcf},
                         {0xd352'd150, 0xffeb'ab6b, 0xffff'edad, 0xffff'ffdf},
                         {0xe362'e160, 0xfff3'b373, 0xffff'f5b5, 0xffff'ffef},
                         {0xf372'f170, 0xfffb'bb7b, 0xffff'fdbd, 0xffff'ffff}},
                        {{0x8706'8504'8302'8100, 0xffff'c787'4686'4585},
                         {0xffff'ffff'9716'9515, 0xffff'ffff'ffff'cf8f},
                         {0xa726'a524'a322'a120, 0xffff'd797'5696'5595},
                         {0xffff'ffff'b736'b535, 0xffff'ffff'ffff'df9f},
                         {0xc746'c544'c342'c140, 0xffff'e7a7'66a6'65a5},
                         {0xffff'ffff'd756'd555, 0xffff'ffff'ffff'efaf},
                         {0xe766'e564'e362'e160, 0xffff'f7b7'76b6'75b5},
                         {0xffff'ffff'f776'f575, 0xffff'ffff'ffff'ffbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravx,
                        ExecMaskedVssravx,
                        {{0, 225, 1, 225, 1, 225, 1, 225, 2, 227, 3, 227, 3, 227, 3, 227},
                         {4, 229, 5, 229, 5, 229, 5, 229, 6, 231, 7, 231, 7, 231, 7, 231},
                         {8, 233, 9, 233, 9, 233, 9, 233, 10, 235, 11, 235, 11, 235, 11, 235},
                         {12, 237, 13, 237, 13, 237, 13, 237, 14, 239, 15, 239, 15, 239, 15, 239},
                         {16, 241, 17, 241, 17, 241, 17, 241, 18, 243, 19, 243, 19, 243, 19, 243},
                         {20, 245, 21, 245, 21, 245, 21, 245, 22, 247, 23, 247, 23, 247, 23, 247},
                         {24, 249, 25, 249, 25, 249, 25, 249, 26, 251, 27, 251, 27, 251, 27, 251},
                         {28, 253, 29, 253, 29, 253, 29, 253, 30, 255, 31, 255, 31, 255, 31, 255}},
                        {{0xffe1, 0xffe1, 0xffe1, 0xffe1, 0xffe3, 0xffe3, 0xffe3, 0xffe3},
                         {0xffe5, 0xffe5, 0xffe5, 0xffe5, 0xffe7, 0xffe7, 0xffe7, 0xffe7},
                         {0xffe9, 0xffe9, 0xffe9, 0xffe9, 0xffeb, 0xffeb, 0xffeb, 0xffeb},
                         {0xffed, 0xffed, 0xffed, 0xffed, 0xffef, 0xffef, 0xffef, 0xffef},
                         {0xfff1, 0xfff1, 0xfff1, 0xfff1, 0xfff3, 0xfff3, 0xfff3, 0xfff3},
                         {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff7, 0xfff7, 0xfff7, 0xfff7},
                         {0xfff9, 0xfff9, 0xfff9, 0xfff9, 0xfffb, 0xfffb, 0xfffb, 0xfffb},
                         {0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xffff, 0xffff, 0xffff, 0xffff}},
                        {{0xffe0'c0a1, 0xffe1'c1a1, 0xffe2'c2a3, 0xffe3'c3a3},
                         {0xffe4'c4a5, 0xffe5'c5a5, 0xffe6'c6a7, 0xffe7'c7a7},
                         {0xffe8'c8a9, 0xffe9'c9a9, 0xffea'caab, 0xffeb'cbab},
                         {0xffec'ccad, 0xffed'cdad, 0xffee'ceaf, 0xffef'cfaf},
                         {0xfff0'd0b1, 0xfff1'd1b1, 0xfff2'd2b3, 0xfff3'd3b3},
                         {0xfff4'd4b5, 0xfff5'd5b5, 0xfff6'd6b7, 0xfff7'd7b7},
                         {0xfff8'd8b9, 0xfff9'd9b9, 0xfffa'dabb, 0xfffb'dbbb},
                         {0xfffc'dcbd, 0xfffd'ddbd, 0xfffe'debf, 0xffff'dfbf}},
                        {{0xffff'ffff'ffe1'c1a1, 0xffff'ffff'ffe3'c3a3},
                         {0xffff'ffff'ffe5'c5a5, 0xffff'ffff'ffe7'c7a7},
                         {0xffff'ffff'ffe9'c9a9, 0xffff'ffff'ffeb'cbab},
                         {0xffff'ffff'ffed'cdad, 0xffff'ffff'ffef'cfaf},
                         {0xffff'ffff'fff1'd1b1, 0xffff'ffff'fff3'd3b3},
                         {0xffff'ffff'fff5'd5b5, 0xffff'ffff'fff7'd7b7},
                         {0xffff'ffff'fff9'd9b9, 0xffff'ffff'fffb'dbbb},
                         {0xffff'ffff'fffd'ddbd, 0xffff'ffff'ffff'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravi,
                        ExecMaskedVssravi,
                        {{0, 225, 1, 225, 1, 225, 1, 225, 2, 227, 3, 227, 3, 227, 3, 227},
                         {4, 229, 5, 229, 5, 229, 5, 229, 6, 231, 7, 231, 7, 231, 7, 231},
                         {8, 233, 9, 233, 9, 233, 9, 233, 10, 235, 11, 235, 11, 235, 11, 235},
                         {12, 237, 13, 237, 13, 237, 13, 237, 14, 239, 15, 239, 15, 239, 15, 239},
                         {16, 241, 17, 241, 17, 241, 17, 241, 18, 243, 19, 243, 19, 243, 19, 243},
                         {20, 245, 21, 245, 21, 245, 21, 245, 22, 247, 23, 247, 23, 247, 23, 247},
                         {24, 249, 25, 249, 25, 249, 25, 249, 26, 251, 27, 251, 27, 251, 27, 251},
                         {28, 253, 29, 253, 29, 253, 29, 253, 30, 255, 31, 255, 31, 255, 31, 255}},
                        {{0xffe1, 0xffe1, 0xffe1, 0xffe1, 0xffe3, 0xffe3, 0xffe3, 0xffe3},
                         {0xffe5, 0xffe5, 0xffe5, 0xffe5, 0xffe7, 0xffe7, 0xffe7, 0xffe7},
                         {0xffe9, 0xffe9, 0xffe9, 0xffe9, 0xffeb, 0xffeb, 0xffeb, 0xffeb},
                         {0xffed, 0xffed, 0xffed, 0xffed, 0xffef, 0xffef, 0xffef, 0xffef},
                         {0xfff1, 0xfff1, 0xfff1, 0xfff1, 0xfff3, 0xfff3, 0xfff3, 0xfff3},
                         {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff7, 0xfff7, 0xfff7, 0xfff7},
                         {0xfff9, 0xfff9, 0xfff9, 0xfff9, 0xfffb, 0xfffb, 0xfffb, 0xfffb},
                         {0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xffff, 0xffff, 0xffff, 0xffff}},
                        {{0xffe0'c0a1, 0xffe1'c1a1, 0xffe2'c2a3, 0xffe3'c3a3},
                         {0xffe4'c4a5, 0xffe5'c5a5, 0xffe6'c6a7, 0xffe7'c7a7},
                         {0xffe8'c8a9, 0xffe9'c9a9, 0xffea'caab, 0xffeb'cbab},
                         {0xffec'ccad, 0xffed'cdad, 0xffee'ceaf, 0xffef'cfaf},
                         {0xfff0'd0b1, 0xfff1'd1b1, 0xfff2'd2b3, 0xfff3'd3b3},
                         {0xfff4'd4b5, 0xfff5'd5b5, 0xfff6'd6b7, 0xfff7'd7b7},
                         {0xfff8'd8b9, 0xfff9'd9b9, 0xfffa'dabb, 0xfffb'dbbb},
                         {0xfffc'dcbd, 0xfffd'ddbd, 0xfffe'debf, 0xffff'dfbf}},
                        {{0xffe1'c1a1'4120'c0a1, 0xffe3'c3a3'4322'c2a3},
                         {0xffe5'c5a5'4524'c4a5, 0xffe7'c7a7'4726'c6a7},
                         {0xffe9'c9a9'4928'c8a9, 0xffeb'cbab'4b2a'caab},
                         {0xffed'cdad'4d2c'ccad, 0xffef'cfaf'4f2e'ceaf},
                         {0xfff1'd1b1'5130'd0b1, 0xfff3'd3b3'5332'd2b3},
                         {0xfff5'd5b5'5534'd4b5, 0xfff7'd7b7'5736'd6b7},
                         {0xfff9'd9b9'5938'd8b9, 0xfffb'dbbb'5b3a'dabb},
                         {0xfffd'ddbd'5d3c'dcbd, 0xffff'dfbf'5f3e'debf}},
                        kVectorCalculationsSource);
  asm volatile("csrw vxrm, %0\n\t" ::"r"(vxrm));
}

[[gnu::naked]] void ExecVfdivvv() {
  asm("vfdiv.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfdivvv() {
  asm("vfdiv.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfdivvf() {
  asm("vfdiv.vf  v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfdivvf() {
  asm("vfdiv.vf  v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfrdivvf() {
  asm("vfrdiv.vf  v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfrdivvf() {
  asm("vfrdiv.vf  v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVdivuvv() {
  asm("vdivu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVdivuvv() {
  asm("vdivu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVdivvv() {
  asm("vdiv.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVdivvv() {
  asm("vdiv.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVdivuvx() {
  asm("vdivu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVdivuvx() {
  asm("vdivu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVdivvx() {
  asm("vdiv.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVdivvx() {
  asm("vdiv.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVdiv) {
  TestVectorFloatInstruction(ExecVfdivvv,
                             ExecMaskedVfdivvv,
                             {{0xbc7d'1561, 0xb875'e8eb, 0xb46f'83ae, 0xb069'c6ec},
                              {0xac64'99e5, 0xa85f'e87d, 0xa45b'a22e, 0xa057'b943},
                              {0x9c54'2241, 0x9850'd382, 0x944d'c4d2, 0x904a'ef31},
                              {0x8c48'4c98, 0x8845'd7d4, 0x8443'8c62, 0x8060'b328},
                              {0x7c3c'206f, 0x7835'0888, 0x742e'b4f5, 0x7029'0782},
                              {0x6c23'e7dc, 0x681f'423a, 0x641b'0659, 0x6017'26b8},
                              {0x5c13'980b, 0x5810'50ca, 0x540d'48e2, 0x500a'7968},
                              {0x4c07'dc6c, 0x4805'6ccb, 0x4403'260f, 0x4001'0454}},
                             {{0xb8e9'b361'617b'3332, 0xb0e1'64f5'e24e'7813},
                              {0xa8d9'a850'c33d'b3c7, 0xa0d1'5a44'cf64'f786},
                              {0x98c9'9d59'4646'6ce0, 0x90c1'4fab'f702'438e},
                              {0x88b9'927a'9559'd99b, 0x80b1'452b'0727'cc70},
                              {0x78a9'87b4'5cbc'33ee, 0x70a1'3ac1'af47'5dc7},
                              {0x6899'7d06'4a29'0e6f, 0x6091'306f'a03b'130b},
                              {0x5889'7270'0ccb'2650, 0x5081'2634'8c3d'81c9},
                              {0x4879'67f1'5534'6be6, 0x4071'1c10'26e2'17fd}},
                             kVectorCalculationsSourceLegacy);
  TestVectorFloatInstruction(ExecVfdivvf,
                             ExecMaskedVfdivvf,
                             {{0x81b9'9b06, 0x85bf'5117, 0x89c5'0728, 0x8dca'bd39},
                              {0x91d0'734a, 0x95d6'295b, 0x99db'df6c, 0x9de1'957d},
                              {0xa1e7'4b8e, 0xa5ed'019f, 0xa9f2'b7b0, 0xadf8'6dc1},
                              {0xb1fe'23d2, 0xb601'ecf2, 0xba04'c7fa, 0xbe07'a303},
                              {0xc20a'7e0b, 0xc60d'5914, 0xca10'341c, 0xce13'0f25},
                              {0xd215'ea2e, 0xd618'c536, 0xda1b'a03f, 0xde1e'7b47},
                              {0xe221'5650, 0xe624'3158, 0xea27'0c61, 0xee29'e769},
                              {0xf22c'c272, 0xf62f'9d7a, 0xfa32'7883, 0xfe35'538b}},
                             {{0x86e0'0391'6e3a'ab61, 0x8ee5'b9a2'8501'cd89},
                              {0x96f0'0efd'9068'39a5, 0x9ef5'c50e'a72f'5bcd},
                              {0xa700'1a69'b295'c7e9, 0xaf05'd07a'c95c'ea11},
                              {0xb710'25d5'd4c3'562e, 0xbf15'dbe6'eb8a'7855},
                              {0xc720'3141'f6f0'e472, 0xcf25'e753'0db8'069a},
                              {0xd730'3cae'191e'72b6, 0xdf35'f2bf'2fe5'94de},
                              {0xe740'481a'3b4c'00fa, 0xef45'fe2b'5213'2322},
                              {0xf750'5386'5d79'8f3f, 0xff56'0997'7440'b166}},
                             kVectorCalculationsSourceLegacy);
  TestVectorFloatInstruction(ExecVfrdivvf,
                             ExecMaskedVfrdivvf,
                             {{0xfd30'8be3, 0xf92b'46b3, 0xf526'4fba, 0xf121'a05c},
                              {0xed1d'32b7, 0xe919'0189, 0xe515'081b, 0xe111'4231},
                              {0xdd0d'abfb, 0xd90a'4206, 0xd507'0132, 0xd103'e6a8},
                              {0xcd00'efd5, 0xc8fc'34b9, 0xc4f6'c831, 0xc0f1'9620},
                              {0xbcec'9ae8, 0xb8e7'd337, 0xb4e3'3bfd, 0xb0de'd268},
                              {0xacda'93da, 0xa8d6'7deb, 0xa4d2'8e5f, 0xa0ce'c322},
                              {0x9ccb'1a46, 0x98c7'9200, 0x94c4'28a6, 0x90c0'dca6},
                              {0x8cbd'ac8d, 0x88ba'96ff, 0x84b7'9ab5, 0x80b4'b67d}},
                             {{0xf8ff'f8de'ba96'50ff, 0xf0f7'9132'204d'3f73},
                              {0xe8ef'e220'dbd5'38e4, 0xe0e7'84d4'8fe3'51e3},
                              {0xd8df'cb83'4048'7bb7, 0xd0d7'7883'f290'f6d8},
                              {0xc8cf'b505'a379'43a3, 0xc0c7'6c40'3409'4932},
                              {0xb8bf'9ea7'c1b1'e9b5, 0xb0b7'6009'4029'bc36},
                              {0xa8af'8869'57fb'4e5c, 0xa0a7'53df'02f9'ad62},
                              {0x989f'724a'241a'3d11, 0x9097'47c1'68a9'f793},
                              {0x888f'5c49'e48c'db01, 0x8087'3bb0'5d94'877b}},
                             kVectorCalculationsSourceLegacy);

  TestVectorInstruction(ExecVdivuvv,
                        ExecMaskedVdivuvv,
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {10, 3, 10, 3, 15, 15, 15, 15, 0, 0, 10, 3, 15, 15, 15, 15},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0003, 0x0003, 0x000f, 0x000f, 0x0000, 0x0003, 0x000f, 0x000f},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0003, 0x0000'000f, 0x0000'0003, 0x0000'000f},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'000e, 0x0000'0000'0000'000e},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);

  TestVectorInstruction(ExecVdivvv,
                        ExecMaskedVdivvv,
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {252, 3, 252, 3, 0, 0, 0, 0, 0, 0, 252, 3, 0, 0, 0, 0},
                         {255, 192, 255, 192, 255, 192, 255, 192, 0, 0, 0, 0, 0, 128, 255, 192},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2},
                         {2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7, 8, 10, 14, 21, 43}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0003, 0x0003, 0x0000, 0x0000, 0x0000, 0x0003, 0x0000, 0x0000},
                         {0xfbbb, 0xfbbb, 0xfbbb, 0xfbbb, 0x0000, 0x0000, 0x0888, 0xfbbb},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0002, 0x0001, 0x0002},
                         {0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0009, 0x0011, 0x0055}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0003, 0x0000'0000, 0x0000'0003, 0x0000'0000},
                         {0xffff'fb6d, 0xffff'fb6d, 0x0000'0000, 0xffff'fb6d},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0002, 0x0000'0002},
                         {0x0000'0003, 0x0000'0005, 0x0000'0009, 0x0000'0055}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0xffff'ffff'ffff'fb6d, 0xffff'ffff'ffff'fb6d},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0002},
                         {0x0000'0000'0000'0005, 0x0000'0000'0000'0055}},
                        kVectorComparisonSource);

  TestVectorInstruction(ExecVdivuvx,
                        ExecMaskedVdivuvx,
                        {{0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                         {1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0001, 0x0001, 0x0000, 0x0000, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0001},
                         {0x0000'0000, 0x0000'0001, 0x0000'0000, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);

  TestVectorInstruction(ExecVdivvx,
                        ExecMaskedVdivvx,
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}

[[gnu::naked]] void ExecVmaccvv() {
  asm("vmacc.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmaccvv() {
  asm("vmacc.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVmaccvx() {
  asm("vmacc.vx  v8, t0, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmaccvx() {
  asm("vmacc.vx  v8, t0, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVwmaccuvv() {
  asm("vwmaccu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmaccuvv() {
  asm("vwmaccu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVwmaccuvx() {
  asm("vwmaccu.vx  v8, t0, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmaccuvx() {
  asm("vwmaccu.vx  v8, t0, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVwmaccvv() {
  asm("vwmacc.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmaccvv() {
  asm("vwmacc.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVwmaccvx() {
  asm("vwmacc.vx  v8, t0, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmaccvx() {
  asm("vwmacc.vx  v8, t0, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVwmaccusvx() {
  asm("vwmaccus.vx  v8, t0, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmaccusvx() {
  asm("vwmaccus.vx  v8, t0, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVwmaccsuvv() {
  asm("vwmaccsu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmaccsuvv() {
  asm("vwmaccsu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVwmaccsuvx() {
  asm("vwmaccsu.vx  v8, t0, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmaccsuvx() {
  asm("vwmaccsu.vx  v8, t0, v16, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmacc) {
  TestVectorInstruction(
      ExecVmaccvv,
      ExecMaskedVmaccvv,
      {{85, 231, 93, 23, 121, 87, 157, 167, 221, 231, 29, 23, 117, 87, 221, 167},
       {85, 39, 221, 215, 137, 151, 29, 103, 237, 39, 157, 215, 117, 151, 93, 103},
       {85, 103, 93, 151, 153, 215, 157, 39, 253, 103, 29, 151, 117, 215, 221, 39},
       {85, 167, 221, 87, 169, 23, 29, 231, 13, 167, 157, 87, 117, 23, 93, 231},
       {85, 231, 93, 23, 185, 87, 157, 167, 29, 231, 29, 23, 117, 87, 221, 167},
       {85, 39, 221, 215, 201, 151, 29, 103, 45, 39, 157, 215, 117, 151, 93, 103},
       {85, 103, 93, 151, 217, 215, 157, 39, 61, 103, 29, 151, 117, 215, 221, 39},
       {85, 167, 221, 87, 233, 23, 29, 231, 77, 167, 157, 87, 117, 23, 93, 231}},
      {{0x5555, 0x8d5d, 0x6a79, 0x5d9d, 0x7edd, 0x6e1d, 0x0675, 0xbedd},
       {0x9755, 0xcfdd, 0xbd89, 0xa11d, 0xd2ed, 0xb29d, 0x4b75, 0x045d},
       {0xdd55, 0x165d, 0x1499, 0xe89d, 0x2afd, 0xfb1d, 0x9475, 0x4ddd},
       {0x2755, 0x60dd, 0x6fa9, 0x341d, 0x870d, 0x479d, 0xe175, 0x9b5d},
       {0x7555, 0xaf5d, 0xceb9, 0x839d, 0xe71d, 0x981d, 0x3275, 0xecdd},
       {0xc755, 0x01dd, 0x31c9, 0xd71d, 0x4b2d, 0xec9d, 0x8775, 0x425d},
       {0x1d55, 0x585d, 0x98d9, 0x2e9d, 0xb33d, 0x451d, 0xe075, 0x9bdd},
       {0x7755, 0xb2dd, 0x03e9, 0x8a1d, 0x1f4d, 0xa19d, 0x3d75, 0xf95d}},
      {{0xc6e7'5555, 0xb4c4'6a79, 0xcf3e'7edd, 0xac0b'0675},
       {0x66c4'9755, 0x67b3'bd89, 0x832d'd2ed, 0x52ec'4b75},
       {0x12a9'dd55, 0x26ab'1499, 0x4325'2afd, 0x05d5'9475},
       {0xca97'2755, 0xf1aa'6fa9, 0x0f24'870d, 0xc4c6'e175},
       {0xcd4c'7555, 0x036d'ceb9, 0x1de3'e71d, 0xc274'3275},
       {0x8d39'c755, 0xd66d'31c9, 0xf1e3'4b2d, 0x8965'8775},
       {0x592f'1d55, 0xb574'98d9, 0xd1ea'b33d, 0x5c5e'e075},
       {0x312c'7755, 0xa084'03e9, 0xbdfa'1f4d, 0x3b60'3d75}},
      {{0xeeb7'f4ac'c6e7'5555, 0xf26a'9b12'cf3e'7edd},
       {0x5d63'0378'66c4'9755, 0x691b'afe2'832d'd2ed},
       {0xe826'2654'12a9'dd55, 0xfbe4'd8c2'4325'2afd},
       {0x8f01'5d3f'ca97'2755, 0xaac6'15b2'0f24'870d},
       {0x082d'23b8'cd4c'7555, 0x13df'd21f'1de3'e71d},
       {0xb708'62a4'8d39'c755, 0xcac1'170e'f1e3'4b2d},
       {0x81fb'b5a0'592f'1d55, 0x9dba'700e'd1ea'b33d},
       {0x6907'1cac'312c'7755, 0x8ccb'dd1e'bdfa'1f4d}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVmaccvx,
      ExecMaskedVmaccvx,
      {{85, 255, 169, 83, 253, 167, 81, 251, 165, 79, 249, 163, 77, 247, 161, 75},
       {245, 159, 73, 243, 157, 71, 241, 155, 69, 239, 153, 67, 237, 151, 65, 235},
       {149, 63, 233, 147, 61, 231, 145, 59, 229, 143, 57, 227, 141, 55, 225, 139},
       {53, 223, 137, 51, 221, 135, 49, 219, 133, 47, 217, 131, 45, 215, 129, 43},
       {213, 127, 41, 211, 125, 39, 209, 123, 37, 207, 121, 35, 205, 119, 33, 203},
       {117, 31, 201, 115, 29, 199, 113, 27, 197, 111, 25, 195, 109, 23, 193, 107},
       {21, 191, 105, 19, 189, 103, 17, 187, 101, 15, 185, 99, 13, 183, 97, 11},
       {181, 95, 9, 179, 93, 7, 177, 91, 5, 175, 89, 3, 173, 87, 1, 171}},
      {{0xff55, 0xa8a9, 0x51fd, 0xfb51, 0xa4a5, 0x4df9, 0xf74d, 0xa0a1},
       {0x49f5, 0xf349, 0x9c9d, 0x45f1, 0xef45, 0x9899, 0x41ed, 0xeb41},
       {0x9495, 0x3de9, 0xe73d, 0x9091, 0x39e5, 0xe339, 0x8c8d, 0x35e1},
       {0xdf35, 0x8889, 0x31dd, 0xdb31, 0x8485, 0x2dd9, 0xd72d, 0x8081},
       {0x29d5, 0xd329, 0x7c7d, 0x25d1, 0xcf25, 0x7879, 0x21cd, 0xcb21},
       {0x7475, 0x1dc9, 0xc71d, 0x7071, 0x19c5, 0xc319, 0x6c6d, 0x15c1},
       {0xbf15, 0x6869, 0x11bd, 0xbb11, 0x6465, 0x0db9, 0xb70d, 0x6061},
       {0x09b5, 0xb309, 0x5c5d, 0x05b1, 0xaf05, 0x5859, 0x01ad, 0xab01}},
      {{0xa8a8'ff55, 0x50a6'51fd, 0xf8a3'a4a5, 0xa0a0'f74d},
       {0x489e'49f5, 0xf09b'9c9d, 0x9898'ef45, 0x4096'41ed},
       {0xe893'9495, 0x9090'e73d, 0x388e'39e5, 0xe08b'8c8d},
       {0x8888'df35, 0x3086'31dd, 0xd883'8485, 0x8080'd72d},
       {0x287e'29d5, 0xd07b'7c7d, 0x7878'cf25, 0x2076'21cd},
       {0xc873'7475, 0x7070'c71d, 0x186e'19c5, 0xc06b'6c6d},
       {0x6868'bf15, 0x1066'11bd, 0xb863'6465, 0x6060'b70d},
       {0x085e'09b5, 0xb05b'5c5d, 0x5858'af05, 0x0056'01ad}},
      {{0xfb50'fca7'a8a8'ff55, 0xa0a0'f74c'f8a3'a4a5},
       {0x45f0'f1f2'489e'49f5, 0xeb40'ec97'9898'ef45},
       {0x9090'e73c'e893'9495, 0x35e0'e1e2'388e'39e5},
       {0xdb30'dc87'8888'df35, 0x8080'd72c'd883'8485},
       {0x25d0'd1d2'287e'29d5, 0xcb20'cc77'7878'cf25},
       {0x7070'c71c'c873'7475, 0x15c0'c1c2'186e'19c5},
       {0xbb10'bc67'6868'bf15, 0x6060'b70c'b863'6465},
       {0x05b0'b1b2'085e'09b5, 0xab00'ac57'5858'af05}},
      kVectorCalculationsSource);
  TestWideningVectorInstruction(ExecVwmaccuvv,
                                ExecMaskedVwmaccuvv,
                                {{0x5555, 0x9ee7, 0x555d, 0xa217, 0x5579, 0xa557, 0x559d, 0xa8a7},
                                 {0x55dd, 0x9ae7, 0x561d, 0x9e17, 0x5675, 0xa157, 0x56dd, 0xa4a7},
                                 {0x5755, 0xba27, 0x57dd, 0xbdd7, 0x5889, 0xc197, 0x591d, 0xc567},
                                 {0x59ed, 0xb627, 0x5a9d, 0xb9d7, 0x5b75, 0xbd97, 0x5c5d, 0xc167},
                                 {0x5d55, 0xd967, 0x5e5d, 0xdd97, 0x5f99, 0xe1d7, 0x609d, 0xe627},
                                 {0x61fd, 0xd567, 0x631d, 0xd997, 0x6475, 0xddd7, 0x65dd, 0xe227},
                                 {0x6755, 0xfca7, 0x68dd, 0x0157, 0x6aa9, 0x0617, 0x6c1d, 0x0ae7},
                                 {0x6e0d, 0xf8a7, 0x6f9d, 0xfd57, 0x7175, 0x0217, 0x735d, 0x06e7}},
                                {{0x9ee7'5555, 0xa21a'8d5d, 0xa55e'6a79, 0xa8b1'5d9d},
                                 {0x9af4'7edd, 0x9e27'6e1d, 0xa16b'0675, 0xa4be'bedd},
                                 {0xba44'9755, 0xbdf8'cfdd, 0xc1bd'bd89, 0xc591'a11d},
                                 {0xb653'd2ed, 0xba07'b29d, 0xbdcc'4b75, 0xc1a1'045d},
                                 {0xd9a9'dd55, 0xdddf'165d, 0xe225'1499, 0xe679'e89d},
                                 {0xd5bb'2afd, 0xd9ef'fb1d, 0xde35'9475, 0xe28b'4ddd},
                                 {0xfd17'2755, 0x01cd'60dd, 0x0694'6fa9, 0x0b6a'341d},
                                 {0xf92a'870d, 0xfde0'479d, 0x02a6'e175, 0x077d'9b5d}},
                                {{0xa21b'23ac'c6e7'5555, 0xa8b2'00fa'b4c4'6a79},
                                 {0x9e27'fc86'cf3e'7edd, 0xa4bf'5a54'ac0b'0675},
                                 {0xbdf9'9d68'66c4'9755, 0xc592'7dba'67b3'bd89},
                                 {0xba08'7846'832d'd2ed, 0xc1a1'd918'52ec'4b75},
                                 {0xdde0'2334'12a9'dd55, 0xe67b'068a'26ab'1499},
                                 {0xd9f1'0016'4325'2afd, 0xe28c'63ec'05d5'9475},
                                 {0x01ce'b50f'ca97'2755, 0x0b6b'9b69'f1aa'6fa9},
                                 {0xfde1'93f6'0f24'870d, 0x077e'facf'c4c6'e175}},
                                kVectorCalculationsSource);
  TestWideningVectorInstruction(ExecVwmaccuvx,
                                ExecMaskedVwmaccuvx,
                                {{0x5555, 0xaaff, 0x56a9, 0xac53, 0x57fd, 0xada7, 0x5951, 0xaefb},
                                 {0x5aa5, 0xb04f, 0x5bf9, 0xb1a3, 0x5d4d, 0xb2f7, 0x5ea1, 0xb44b},
                                 {0x5ff5, 0xb59f, 0x6149, 0xb6f3, 0x629d, 0xb847, 0x63f1, 0xb99b},
                                 {0x6545, 0xbaef, 0x6699, 0xbc43, 0x67ed, 0xbd97, 0x6941, 0xbeeb},
                                 {0x6a95, 0xc03f, 0x6be9, 0xc193, 0x6d3d, 0xc2e7, 0x6e91, 0xc43b},
                                 {0x6fe5, 0xc58f, 0x7139, 0xc6e3, 0x728d, 0xc837, 0x73e1, 0xc98b},
                                 {0x7535, 0xcadf, 0x7689, 0xcc33, 0x77dd, 0xcd87, 0x7931, 0xcedb},
                                 {0x7a85, 0xd02f, 0x7bd9, 0xd183, 0x7d2d, 0xd2d7, 0x7e81, 0xd42b}},
                                {{0xab54'ff55, 0xacab'a8a9, 0xae02'51fd, 0xaf58'fb51},
                                 {0xb0af'a4a5, 0xb206'4df9, 0xb35c'f74d, 0xb4b3'a0a1},
                                 {0xb60a'49f5, 0xb760'f349, 0xb8b7'9c9d, 0xba0e'45f1},
                                 {0xbb64'ef45, 0xbcbb'9899, 0xbe12'41ed, 0xbf68'eb41},
                                 {0xc0bf'9495, 0xc216'3de9, 0xc36c'e73d, 0xc4c3'9091},
                                 {0xc61a'39e5, 0xc770'e339, 0xc8c7'8c8d, 0xca1e'35e1},
                                 {0xcb74'df35, 0xcccb'8889, 0xce22'31dd, 0xcf78'db31},
                                 {0xd0cf'8485, 0xd226'2dd9, 0xd37c'd72d, 0xd4d3'8081}},
                                {{0xacac'55ff'a8a8'ff55, 0xaf59'ae02'50a6'51fd},
                                 {0xb207'0604'f8a3'a4a5, 0xb4b4'5e07'a0a0'f74d},
                                 {0xb761'b60a'489e'49f5, 0xba0f'0e0c'f09b'9c9d},
                                 {0xbcbc'660f'9898'ef45, 0xbf69'be12'4096'41ed},
                                 {0xc217'1614'e893'9495, 0xc4c4'6e17'9090'e73d},
                                 {0xc771'c61a'388e'39e5, 0xca1f'1e1c'e08b'8c8d},
                                 {0xcccc'761f'8888'df35, 0xcf79'ce22'3086'31dd},
                                 {0xd227'2624'd883'8485, 0xd4d4'7e27'8080'd72d}},
                                kVectorCalculationsSource);
  TestWideningVectorInstruction(ExecVwmaccvv,
                                ExecMaskedVwmaccvv,
                                {{0x5555, 0x8be7, 0x555d, 0x8917, 0x5579, 0x8657, 0x559d, 0x83a7},
                                 {0x55dd, 0x8fe7, 0x561d, 0x8d17, 0x5675, 0x8a57, 0x56dd, 0x87a7},
                                 {0x5755, 0x7727, 0x57dd, 0x74d7, 0x5889, 0x7297, 0x591d, 0x7067},
                                 {0x59ed, 0x7b27, 0x5a9d, 0x78d7, 0x5b75, 0x7697, 0x5c5d, 0x7467},
                                 {0x5d55, 0x6667, 0x5e5d, 0x6497, 0x5f99, 0x62d7, 0x609d, 0x6127},
                                 {0x61fd, 0x6a67, 0x631d, 0x6897, 0x6475, 0x66d7, 0x65dd, 0x6527},
                                 {0x6755, 0x59a7, 0x68dd, 0x5857, 0x6aa9, 0x5717, 0x6c1d, 0x55e7},
                                 {0x6e0d, 0x5da7, 0x6f9d, 0x5c57, 0x7175, 0x5b17, 0x735d, 0x59e7}},
                                {{0x8be7'5555, 0x8914'8d5d, 0x8651'6a79, 0x839f'5d9d},
                                 {0x8fdb'7edd, 0x8d09'6e1d, 0x8a47'0675, 0x8794'bedd},
                                 {0x7714'9755, 0x74c2'cfdd, 0x7280'bd89, 0x704f'a11d},
                                 {0x7b0a'd2ed, 0x78b9'b29d, 0x7678'4b75, 0x7447'045d},
                                 {0x6649'dd55, 0x6479'165d, 0x62b8'1499, 0x6107'e89d},
                                 {0x6a42'2afd, 0x6871'fb1d, 0x66b1'9475, 0x6501'4ddd},
                                 {0x5987'2755, 0x5837'60dd, 0x56f7'6fa9, 0x55c8'341d},
                                 {0x5d81'870d, 0x5c32'479d, 0x5af2'e175, 0x59c3'9b5d}},
                                {{0x8914'10ac'c6e7'5555, 0x839e'e1ed'b4c4'6a79},
                                 {0x8d08'f16d'cf3e'7edd, 0x8794'4330'ac0b'0675},
                                 {0x74c2'5a38'66c4'9755, 0x704f'2e7d'67b3'bd89},
                                 {0x78b9'3cfd'832d'd2ed, 0x7446'91c4'52ec'4b75},
                                 {0x6478'afd4'12a9'dd55, 0x6107'871d'26ab'1499},
                                 {0x6871'949d'4325'2afd, 0x6500'ec68'05d5'9475},
                                 {0x5837'117f'ca97'2755, 0x55c7'ebcc'f1aa'6fa9},
                                 {0x5c31'f84d'0f24'870d, 0x59c3'531b'c4c6'e175}},
                                kVectorCalculationsSource);
  TestWideningVectorInstruction(ExecVwmaccvx,
                                ExecMaskedVwmaccvx,
                                {{0x5555, 0x7fff, 0x54a9, 0x7f53, 0x53fd, 0x7ea7, 0x5351, 0x7dfb},
                                 {0x52a5, 0x7d4f, 0x51f9, 0x7ca3, 0x514d, 0x7bf7, 0x50a1, 0x7b4b},
                                 {0x4ff5, 0x7a9f, 0x4f49, 0x79f3, 0x4e9d, 0x7947, 0x4df1, 0x789b},
                                 {0x4d45, 0x77ef, 0x4c99, 0x7743, 0x4bed, 0x7697, 0x4b41, 0x75eb},
                                 {0x4a95, 0x753f, 0x49e9, 0x7493, 0x493d, 0x73e7, 0x4891, 0x733b},
                                 {0x47e5, 0x728f, 0x4739, 0x71e3, 0x468d, 0x7137, 0x45e1, 0x708b},
                                 {0x4535, 0x6fdf, 0x4489, 0x6f33, 0x43dd, 0x6e87, 0x4331, 0x6ddb},
                                 {0x4285, 0x6d2f, 0x41d9, 0x6c83, 0x412d, 0x6bd7, 0x4081, 0x6b2b}},
                                {{0x7faa'ff55, 0x7eff'a8a9, 0x7e54'51fd, 0x7da8'fb51},
                                 {0x7cfd'a4a5, 0x7c52'4df9, 0x7ba6'f74d, 0x7afb'a0a1},
                                 {0x7a50'49f5, 0x79a4'f349, 0x78f9'9c9d, 0x784e'45f1},
                                 {0x77a2'ef45, 0x76f7'9899, 0x764c'41ed, 0x75a0'eb41},
                                 {0x74f5'9495, 0x744a'3de9, 0x739e'e73d, 0x72f3'9091},
                                 {0x7248'39e5, 0x719c'e339, 0x70f1'8c8d, 0x7046'35e1},
                                 {0x6f9a'df35, 0x6eef'8889, 0x6e44'31dd, 0x6d98'db31},
                                 {0x6ced'8485, 0x6c42'2dd9, 0x6b96'd72d, 0x6aeb'8081}},
                                {{0x7eff'2a55'a8a8'ff55, 0x7da8'7e54'50a6'51fd},
                                 {0x7c51'd252'f8a3'a4a5, 0x7afb'2651'a0a0'f74d},
                                 {0x79a4'7a50'489e'49f5, 0x784d'ce4e'f09b'9c9d},
                                 {0x76f7'224d'9898'ef45, 0x75a0'764c'4096'41ed},
                                 {0x7449'ca4a'e893'9495, 0x72f3'1e49'9090'e73d},
                                 {0x719c'7248'388e'39e5, 0x7045'c646'e08b'8c8d},
                                 {0x6eef'1a45'8888'df35, 0x6d98'6e44'3086'31dd},
                                 {0x6c41'c242'd883'8485, 0x6aeb'1641'8080'd72d}},
                                kVectorCalculationsSource);
  TestWideningVectorInstruction(ExecVwmaccusvx,
                                ExecMaskedVwmaccusvx,
                                {{0x5555, 0x00ff, 0x56a9, 0x0253, 0x57fd, 0x03a7, 0x5951, 0x04fb},
                                 {0x5aa5, 0x064f, 0x5bf9, 0x07a3, 0x5d4d, 0x08f7, 0x5ea1, 0x0a4b},
                                 {0x5ff5, 0x0b9f, 0x6149, 0x0cf3, 0x629d, 0x0e47, 0x63f1, 0x0f9b},
                                 {0x6545, 0x10ef, 0x6699, 0x1243, 0x67ed, 0x1397, 0x6941, 0x14eb},
                                 {0x6a95, 0x163f, 0x6be9, 0x1793, 0x6d3d, 0x18e7, 0x6e91, 0x1a3b},
                                 {0x6fe5, 0x1b8f, 0x7139, 0x1ce3, 0x728d, 0x1e37, 0x73e1, 0x1f8b},
                                 {0x7535, 0x20df, 0x7689, 0x2233, 0x77dd, 0x2387, 0x7931, 0x24db},
                                 {0x7a85, 0x262f, 0x7bd9, 0x2783, 0x7d2d, 0x28d7, 0x7e81, 0x2a2b}},
                                {{0x00aa'ff55, 0x0201'a8a9, 0x0358'51fd, 0x04ae'fb51},
                                 {0x0605'a4a5, 0x075c'4df9, 0x08b2'f74d, 0x0a09'a0a1},
                                 {0x0b60'49f5, 0x0cb6'f349, 0x0e0d'9c9d, 0x0f64'45f1},
                                 {0x10ba'ef45, 0x1211'9899, 0x1368'41ed, 0x14be'eb41},
                                 {0x1615'9495, 0x176c'3de9, 0x18c2'e73d, 0x1a19'9091},
                                 {0x1b70'39e5, 0x1cc6'e339, 0x1e1d'8c8d, 0x1f74'35e1},
                                 {0x20ca'df35, 0x2221'8889, 0x2378'31dd, 0x24ce'db31},
                                 {0x2625'8485, 0x277c'2dd9, 0x28d2'd72d, 0x2a29'8081}},
                                {{0x0201'ab55'a8a8'ff55, 0x04af'0358'50a6'51fd},
                                 {0x075c'5b5a'f8a3'a4a5, 0x0a09'b35d'a0a0'f74d},
                                 {0x0cb7'0b60'489e'49f5, 0x0f64'6362'f09b'9c9d},
                                 {0x1211'bb65'9898'ef45, 0x14bf'1368'4096'41ed},
                                 {0x176c'6b6a'e893'9495, 0x1a19'c36d'9090'e73d},
                                 {0x1cc7'1b70'388e'39e5, 0x1f74'7372'e08b'8c8d},
                                 {0x2221'cb75'8888'df35, 0x24cf'2378'3086'31dd},
                                 {0x277c'7b7a'd883'8485, 0x2a29'd37d'8080'd72d}},
                                kVectorCalculationsSource);
  TestWideningVectorInstruction(ExecVwmaccsuvv,
                                ExecMaskedVwmaccsuvv,
                                {{0x5555, 0x0ce7, 0x555d, 0x0c17, 0x5579, 0x0b57, 0x559d, 0x0aa7},
                                 {0x55dd, 0x18e7, 0x561d, 0x1817, 0x5675, 0x1757, 0x56dd, 0x16a7},
                                 {0x5755, 0x0827, 0x57dd, 0x07d7, 0x5889, 0x0797, 0x591d, 0x0767},
                                 {0x59ed, 0x1427, 0x5a9d, 0x13d7, 0x5b75, 0x1397, 0x5c5d, 0x1367},
                                 {0x5d55, 0x0767, 0x5e5d, 0x0797, 0x5f99, 0x07d7, 0x609d, 0x0827},
                                 {0x61fd, 0x1367, 0x631d, 0x1397, 0x6475, 0x13d7, 0x65dd, 0x1427},
                                 {0x6755, 0x0aa7, 0x68dd, 0x0b57, 0x6aa9, 0x0c17, 0x6c1d, 0x0ce7},
                                 {0x6e0d, 0x16a7, 0x6f9d, 0x1757, 0x7175, 0x1817, 0x735d, 0x18e7}},
                                {{0x0ce7'5555, 0x0c16'8d5d, 0x0b55'6a79, 0x0aa5'5d9d},
                                 {0x18e3'7edd, 0x1813'6e1d, 0x1753'0675, 0x16a2'bedd},
                                 {0x0824'9755, 0x07d4'cfdd, 0x0794'bd89, 0x0765'a11d},
                                 {0x1422'd2ed, 0x13d3'b29d, 0x1394'4b75, 0x1365'045d},
                                 {0x0769'dd55, 0x079b'165d, 0x07dc'1499, 0x082d'e89d},
                                 {0x136a'2afd, 0x139b'fb1d, 0x13dd'9475, 0x142f'4ddd},
                                 {0x0ab7'2755, 0x0b69'60dd, 0x0c2b'6fa9, 0x0cfe'341d},
                                 {0x16b9'870d, 0x176c'479d, 0x182e'e175, 0x1901'9b5d}},
                                {{0x0c16'91ac'c6e7'5555, 0x0aa5'66f1'b4c4'6a79},
                                 {0x1813'7a75'cf3e'7edd, 0x16a2'd03c'ac0b'0675},
                                 {0x07d4'eb48'66c4'9755, 0x0765'c391'67b3'bd89},
                                 {0x13d3'd615'832d'd2ed, 0x1365'2ee0'52ec'4b75},
                                 {0x079b'50f4'12a9'dd55, 0x082e'2c41'26ab'1499},
                                 {0x139c'3dc5'4325'2afd, 0x142f'9994'05d5'9475},
                                 {0x0b69'c2af'ca97'2755, 0x0cfe'a100'f1aa'6fa9},
                                 {0x176c'b185'0f24'870d, 0x1902'1057'c4c6'e175}},
                                kVectorCalculationsSource);
  TestWideningVectorInstruction(ExecVwmaccsuvx,
                                ExecMaskedVwmaccsuvx,
                                {{0x5555, 0x29ff, 0x54a9, 0x2953, 0x53fd, 0x28a7, 0x5351, 0x27fb},
                                 {0x52a5, 0x274f, 0x51f9, 0x26a3, 0x514d, 0x25f7, 0x50a1, 0x254b},
                                 {0x4ff5, 0x249f, 0x4f49, 0x23f3, 0x4e9d, 0x2347, 0x4df1, 0x229b},
                                 {0x4d45, 0x21ef, 0x4c99, 0x2143, 0x4bed, 0x2097, 0x4b41, 0x1feb},
                                 {0x4a95, 0x1f3f, 0x49e9, 0x1e93, 0x493d, 0x1de7, 0x4891, 0x1d3b},
                                 {0x47e5, 0x1c8f, 0x4739, 0x1be3, 0x468d, 0x1b37, 0x45e1, 0x1a8b},
                                 {0x4535, 0x19df, 0x4489, 0x1933, 0x43dd, 0x1887, 0x4331, 0x17db},
                                 {0x4285, 0x172f, 0x41d9, 0x1683, 0x412d, 0x15d7, 0x4081, 0x152b}},
                                {{0x2a54'ff55, 0x29a9'a8a9, 0x28fe'51fd, 0x2852'fb51},
                                 {0x27a7'a4a5, 0x26fc'4df9, 0x2650'f74d, 0x25a5'a0a1},
                                 {0x24fa'49f5, 0x244e'f349, 0x23a3'9c9d, 0x22f8'45f1},
                                 {0x224c'ef45, 0x21a1'9899, 0x20f6'41ed, 0x204a'eb41},
                                 {0x1f9f'9495, 0x1ef4'3de9, 0x1e48'e73d, 0x1d9d'9091},
                                 {0x1cf2'39e5, 0x1c46'e339, 0x1b9b'8c8d, 0x1af0'35e1},
                                 {0x1a44'df35, 0x1999'8889, 0x18ee'31dd, 0x1842'db31},
                                 {0x1797'8485, 0x16ec'2dd9, 0x1640'd72d, 0x1595'8081}},
                                {{0x29a9'd4ff'a8a8'ff55, 0x2853'28fe'50a6'51fd},
                                 {0x26fc'7cfc'f8a3'a4a5, 0x25a5'd0fb'a0a0'f74d},
                                 {0x244f'24fa'489e'49f5, 0x22f8'78f8'f09b'9c9d},
                                 {0x21a1'ccf7'9898'ef45, 0x204b'20f6'4096'41ed},
                                 {0x1ef4'74f4'e893'9495, 0x1d9d'c8f3'9090'e73d},
                                 {0x1c47'1cf2'388e'39e5, 0x1af0'70f0'e08b'8c8d},
                                 {0x1999'c4ef'8888'df35, 0x1843'18ee'3086'31dd},
                                 {0x16ec'6cec'd883'8485, 0x1595'c0eb'8080'd72d}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVmaddvv() {
  asm("vmadd.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmaddvv() {
  asm("vmadd.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVmaddvx() {
  asm("vmadd.vx  v8, t0, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmaddvx() {
  asm("vmadd.vx  v8, t0, v16, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmadd) {
  TestVectorInstruction(
      ExecVmaddvv,
      ExecMaskedVmaddvv,
      {{0, 103, 174, 21, 93, 195, 10, 113, 185, 255, 102, 173, 20, 91, 194, 9},
       {112, 215, 30, 133, 205, 51, 122, 225, 41, 111, 214, 29, 132, 203, 50, 121},
       {224, 71, 142, 245, 61, 163, 234, 81, 153, 223, 70, 141, 244, 59, 162, 233},
       {80, 183, 254, 101, 173, 19, 90, 193, 9, 79, 182, 253, 100, 171, 18, 89},
       {192, 39, 110, 213, 29, 131, 202, 49, 121, 191, 38, 109, 212, 27, 130, 201},
       {48, 151, 222, 69, 141, 243, 58, 161, 233, 47, 150, 221, 68, 139, 242, 57},
       {160, 7, 78, 181, 253, 99, 170, 17, 89, 159, 6, 77, 180, 251, 98, 169},
       {16, 119, 190, 37, 109, 211, 26, 129, 201, 15, 118, 189, 36, 107, 210, 25}},
      {{0x6700, 0xbfae, 0x185d, 0x710a, 0xa9b9, 0x0266, 0x5b14, 0xb3c2},
       {0x2c70, 0x851e, 0xddcd, 0x367a, 0x6f29, 0xc7d6, 0x2084, 0x7932},
       {0xf1e0, 0x4a8e, 0xa33d, 0xfbea, 0x3499, 0x8d46, 0xe5f4, 0x3ea2},
       {0xb750, 0x0ffe, 0x68ad, 0xc15a, 0xfa09, 0x52b6, 0xab64, 0x0412},
       {0x7cc0, 0xd56e, 0x2e1d, 0x86ca, 0xbf79, 0x1826, 0x70d4, 0xc982},
       {0x4230, 0x9ade, 0xf38d, 0x4c3a, 0x84e9, 0xdd96, 0x3644, 0x8ef2},
       {0x07a0, 0x604e, 0xb8fd, 0x11aa, 0x4a59, 0xa306, 0xfbb4, 0x5462},
       {0xcd10, 0x25be, 0x7e6d, 0xd71a, 0x0fc9, 0x6876, 0xc124, 0x19d2}},
      {{0xbfae'6700, 0x1bb5'185d, 0x57bb'a9b9, 0xb3c2'5b14},
       {0x2fc9'2c70, 0x8bcf'ddcd, 0xc7d6'6f29, 0x23dd'2084},
       {0x9fe3'f1e0, 0xfbea'a33d, 0x37f1'3499, 0x93f7'e5f4},
       {0x0ffe'b750, 0x6c05'68ad, 0xa80b'fa09, 0x0412'ab64},
       {0x8018'7cc0, 0xdc1f'2e1d, 0x1825'bf79, 0x742c'70d4},
       {0xf033'4230, 0x4c39'f38d, 0x8840'84e9, 0xe447'3644},
       {0x604e'07a0, 0xbc54'b8fd, 0xf85b'4a59, 0x5461'fbb4},
       {0xd068'cd10, 0x2c6f'7e6d, 0x6876'0fc9, 0xc47c'c124}},
      {{0x710a'6db2'bfae'6700, 0xb3c2'5b14'57bb'a9b9},
       {0x367a'8878'2fc9'2c70, 0x7932'75d9'c7d6'6f29},
       {0xfbea'a33d'9fe3'f1e0, 0x3ea2'909f'37f1'3499},
       {0xc15a'be03'0ffe'b750, 0x0412'ab64'a80b'fa09},
       {0x86c9'd8c7'8018'7cc0, 0xc981'c629'1825'bf79},
       {0x4c39'f38c'f033'4230, 0x8ef1'e0ee'8840'84e9},
       {0x11aa'0e52'604e'07a0, 0x5461'fbb3'f85b'4a59},
       {0xd71a'2917'd068'cd10, 0x19d2'1679'6876'0fc9}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVmaddvx,
      ExecMaskedVmaddvx,
      {{114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255, 128, 1},
       {130, 3, 132, 5, 134, 7, 136, 9, 138, 11, 140, 13, 142, 15, 144, 17},
       {146, 19, 148, 21, 150, 23, 152, 25, 154, 27, 156, 29, 158, 31, 160, 33},
       {162, 35, 164, 37, 166, 39, 168, 41, 170, 43, 172, 45, 174, 47, 176, 49},
       {178, 51, 180, 53, 182, 55, 184, 57, 186, 59, 188, 61, 190, 63, 192, 65},
       {194, 67, 196, 69, 198, 71, 200, 73, 202, 75, 204, 77, 206, 79, 208, 81},
       {210, 83, 212, 85, 214, 87, 216, 89, 218, 91, 220, 93, 222, 95, 224, 97},
       {226, 99, 228, 101, 230, 103, 232, 105, 234, 107, 236, 109, 238, 111, 240, 113}},
      {{0x9d72, 0x9f74, 0xa176, 0xa378, 0xa57a, 0xa77c, 0xa97e, 0xab80},
       {0xad82, 0xaf84, 0xb186, 0xb388, 0xb58a, 0xb78c, 0xb98e, 0xbb90},
       {0xbd92, 0xbf94, 0xc196, 0xc398, 0xc59a, 0xc79c, 0xc99e, 0xcba0},
       {0xcda2, 0xcfa4, 0xd1a6, 0xd3a8, 0xd5aa, 0xd7ac, 0xd9ae, 0xdbb0},
       {0xddb2, 0xdfb4, 0xe1b6, 0xe3b8, 0xe5ba, 0xe7bc, 0xe9be, 0xebc0},
       {0xedc2, 0xefc4, 0xf1c6, 0xf3c8, 0xf5ca, 0xf7cc, 0xf9ce, 0xfbd0},
       {0xfdd2, 0xffd4, 0x01d6, 0x03d8, 0x05da, 0x07dc, 0x09de, 0x0be0},
       {0x0de2, 0x0fe4, 0x11e6, 0x13e8, 0x15ea, 0x17ec, 0x19ee, 0x1bf0}},
      {{0xf4c9'9d72, 0xf8cd'a176, 0xfcd1'a57a, 0x00d5'a97e},
       {0x04d9'ad82, 0x08dd'b186, 0x0ce1'b58a, 0x10e5'b98e},
       {0x14e9'bd92, 0x18ed'c196, 0x1cf1'c59a, 0x20f5'c99e},
       {0x24f9'cda2, 0x28fd'd1a6, 0x2d01'd5aa, 0x3105'd9ae},
       {0x3509'ddb2, 0x390d'e1b6, 0x3d11'e5ba, 0x4115'e9be},
       {0x4519'edc2, 0x491d'f1c6, 0x4d21'f5ca, 0x5125'f9ce},
       {0x5529'fdd2, 0x592e'01d6, 0x5d32'05da, 0x6136'09de},
       {0x653a'0de2, 0x693e'11e6, 0x6d42'15ea, 0x7146'19ee}},
      {{0xa378'4c20'f4c9'9d72, 0xab80'5428'fcd1'a57a},
       {0xb388'5c31'04d9'ad82, 0xbb90'6439'0ce1'b58a},
       {0xc398'6c41'14e9'bd92, 0xcba0'7449'1cf1'c59a},
       {0xd3a8'7c51'24f9'cda2, 0xdbb0'8459'2d01'd5aa},
       {0xe3b8'8c61'3509'ddb2, 0xebc0'9469'3d11'e5ba},
       {0xf3c8'9c71'4519'edc2, 0xfbd0'a479'4d21'f5ca},
       {0x03d8'ac81'5529'fdd2, 0x0be0'b489'5d32'05da},
       {0x13e8'bc91'653a'0de2, 0x1bf0'c499'6d42'15ea}},
      kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVmaxvv() {
  asm("vmax.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmaxvv() {
  asm("vmax.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVmaxvx() {
  asm("vmax.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmaxvx() {
  asm("vmax.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmax) {
  TestVectorInstruction(
      ExecVmaxvv,
      ExecMaskedVmaxvv,
      {{0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126},
       {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
       {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
       {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
       {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}},
      {{0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
       {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
       {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
       {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
      {{0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
      {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVmaxvx,
      ExecMaskedVmaxvx,
      {{0, 170, 2, 170, 4, 170, 6, 170, 8, 170, 10, 170, 12, 170, 14, 170},
       {16, 170, 18, 170, 20, 170, 22, 170, 24, 170, 26, 170, 28, 170, 30, 170},
       {32, 170, 34, 170, 36, 170, 38, 170, 40, 170, 42, 171, 44, 173, 46, 175},
       {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
       {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
       {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
       {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
       {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}},
      {{0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xab2a, 0xad2c, 0xaf2e},
       {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
       {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
       {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
       {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
       {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
      {{0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xab2a'a928, 0xaf2e'ad2c},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaf2e'ad2c'ab2a'a928},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVmaxuvv() {
  asm("vmaxu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmaxuvv() {
  asm("vmaxu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVmaxuvx() {
  asm("vmaxu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmaxuvx() {
  asm("vmaxu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmaxu) {
  TestVectorInstruction(
      ExecVmaxuvv,
      ExecMaskedVmaxuvv,
      {{0, 129, 4, 131, 9, 133, 12, 135, 17, 137, 20, 139, 24, 141, 28, 143},
       {32, 145, 36, 147, 41, 149, 44, 151, 49, 153, 52, 155, 56, 157, 60, 159},
       {64, 161, 68, 163, 73, 165, 76, 167, 81, 169, 84, 171, 88, 173, 92, 175},
       {96, 177, 100, 179, 105, 181, 108, 183, 113, 185, 116, 187, 120, 189, 124, 191},
       {128, 193, 132, 195, 137, 197, 140, 199, 145, 201, 148, 203, 152, 205, 156, 207},
       {160, 209, 164, 211, 169, 213, 172, 215, 177, 217, 180, 219, 184, 221, 188, 223},
       {192, 225, 196, 227, 201, 229, 204, 231, 209, 233, 212, 235, 216, 237, 220, 239},
       {224, 241, 228, 243, 233, 245, 236, 247, 241, 249, 244, 251, 248, 253, 252, 255}},
      {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
       {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
       {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
       {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
       {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
       {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
       {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
       {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
       {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
       {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVmaxuvx,
      ExecMaskedVmaxuvx,
      {{170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 171, 170, 173, 170, 175},
       {170, 177, 170, 179, 170, 181, 170, 183, 170, 185, 170, 187, 170, 189, 170, 191},
       {170, 193, 170, 195, 170, 197, 170, 199, 170, 201, 170, 203, 170, 205, 170, 207},
       {170, 209, 170, 211, 170, 213, 170, 215, 170, 217, 170, 219, 170, 221, 170, 223},
       {170, 225, 170, 227, 170, 229, 170, 231, 170, 233, 170, 235, 170, 237, 170, 239},
       {170, 241, 170, 243, 170, 245, 170, 247, 170, 249, 170, 251, 170, 253, 170, 255}},
      {{0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xab2a, 0xad2c, 0xaf2e},
       {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
       {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
       {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
       {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
       {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
      {{0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xab2a'a928, 0xaf2e'ad2c},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaf2e'ad2c'ab2a'a928},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVminvv() {
  asm("vmin.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVminvv() {
  asm("vmin.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVminvx() {
  asm("vmin.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVminvx() {
  asm("vmin.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmin) {
  TestVectorInstruction(
      ExecVminvv,
      ExecMaskedVminvv,
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
       {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
       {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
       {128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158},
       {160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190},
       {192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222},
       {224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254}},
      {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
       {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
       {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
       {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
       {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
       {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
       {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVminvx,
      ExecMaskedVminvx,
      {{170, 129, 170, 131, 170, 133, 170, 135, 170, 137, 170, 139, 170, 141, 170, 143},
       {170, 145, 170, 147, 170, 149, 170, 151, 170, 153, 170, 155, 170, 157, 170, 159},
       {170, 161, 170, 163, 170, 165, 170, 167, 170, 169, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170}},
      {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
       {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
       {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
       {0xa322'a120, 0xa726'a524, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
       {0xa726'a524'a322'a120, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVminuvv() {
  asm("vminu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVminuvv() {
  asm("vminu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVminuvx() {
  asm("vminu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVminuvx() {
  asm("vminu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVminu) {
  TestVectorInstruction(
      ExecVminuvv,
      ExecMaskedVminuvv,
      {{0, 2, 2, 6, 4, 10, 6, 14, 8, 18, 10, 22, 12, 26, 14, 30},
       {16, 34, 18, 38, 20, 42, 22, 46, 24, 50, 26, 54, 28, 58, 30, 62},
       {32, 66, 34, 70, 36, 74, 38, 78, 40, 82, 42, 86, 44, 90, 46, 94},
       {48, 98, 50, 102, 52, 106, 54, 110, 56, 114, 58, 118, 60, 122, 62, 126},
       {64, 130, 66, 134, 68, 138, 70, 142, 72, 146, 74, 150, 76, 154, 78, 158},
       {80, 162, 82, 166, 84, 170, 86, 174, 88, 178, 90, 182, 92, 186, 94, 190},
       {96, 194, 98, 198, 100, 202, 102, 206, 104, 210, 106, 214, 108, 218, 110, 222},
       {112, 226, 114, 230, 116, 234, 118, 238, 120, 242, 122, 246, 124, 250, 126, 254}},
      {{0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc}},
      {{0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
      {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
       {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVminuvx,
      ExecMaskedVminuvx,
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
       {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 170, 44, 170, 46, 170},
       {48, 170, 50, 170, 52, 170, 54, 170, 56, 170, 58, 170, 60, 170, 62, 170},
       {64, 170, 66, 170, 68, 170, 70, 170, 72, 170, 74, 170, 76, 170, 78, 170},
       {80, 170, 82, 170, 84, 170, 86, 170, 88, 170, 90, 170, 92, 170, 94, 170},
       {96, 170, 98, 170, 100, 170, 102, 170, 104, 170, 106, 170, 108, 170, 110, 170},
       {112, 170, 114, 170, 116, 170, 118, 170, 120, 170, 122, 170, 124, 170, 126, 170}},
      {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
       {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
       {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
       {0xa322'a120, 0xa726'a524, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
       {0xa726'a524'a322'a120, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVmulvv() {
  asm("vmul.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmulvv() {
  asm("vmul.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVmulvx() {
  asm("vmul.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmulvx() {
  asm("vmul.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfmulvv() {
  asm("vfmul.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfmulvv() {
  asm("vfmul.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfmulvf() {
  asm("vfmul.vf  v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfmulvf() {
  asm("vfmul.vf  v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwmulvv() {
  asm("vfwmul.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwmulvv() {
  asm("vfwmul.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwmulvf() {
  asm("vfwmul.vf  v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwmulvf() {
  asm("vfwmul.vf  v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmul) {
  TestVectorInstruction(ExecVmulvv,
                        ExecMaskedVmulvv,
                        {{0, 2, 8, 18, 36, 50, 72, 98, 136, 162, 200, 242, 32, 82, 136, 194},
                         {0, 66, 136, 210, 52, 114, 200, 34, 152, 226, 72, 178, 32, 146, 8, 130},
                         {0, 130, 8, 146, 68, 178, 72, 226, 168, 34, 200, 114, 32, 210, 136, 66},
                         {0, 194, 136, 82, 84, 242, 200, 162, 184, 98, 72, 50, 32, 18, 8, 2},
                         {0, 2, 8, 18, 100, 50, 72, 98, 200, 162, 200, 242, 32, 82, 136, 194},
                         {0, 66, 136, 210, 116, 114, 200, 34, 216, 226, 72, 178, 32, 146, 8, 130},
                         {0, 130, 8, 146, 132, 178, 72, 226, 232, 34, 200, 114, 32, 210, 136, 66},
                         {0, 194, 136, 82, 148, 242, 200, 162, 248, 98, 72, 50, 32, 18, 8, 2}},
                        {{0x0000, 0x1808, 0xd524, 0xa848, 0xa988, 0xb8c8, 0x7120, 0x4988},
                         {0x4200, 0x5a88, 0x2834, 0xebc8, 0xfd98, 0xfd48, 0xb620, 0x8f08},
                         {0x8800, 0xa108, 0x7f44, 0x3348, 0x55a8, 0x45c8, 0xff20, 0xd888},
                         {0xd200, 0xeb88, 0xda54, 0x7ec8, 0xb1b8, 0x9248, 0x4c20, 0x2608},
                         {0x2000, 0x3a08, 0x3964, 0xce48, 0x11c8, 0xe2c8, 0x9d20, 0x7788},
                         {0x7200, 0x8c88, 0x9c74, 0x21c8, 0x75d8, 0x3748, 0xf220, 0xcd08},
                         {0xc800, 0xe308, 0x0384, 0x7948, 0xdde8, 0x8fc8, 0x4b20, 0x2688},
                         {0x2200, 0x3d88, 0x6e94, 0xd4c8, 0x49f8, 0xec48, 0xa820, 0x8408}},
                        {{0x0902'0000, 0x749c'd524, 0x5df5'a988, 0xb900'7120},
                         {0x9fd6'4200, 0x1e83'2834, 0x0add'fd98, 0x58da'b620},
                         {0x42b2'8800, 0xd471'7f44, 0xc3ce'55a8, 0x04bc'ff20},
                         {0xf196'd200, 0x9667'da54, 0x88c6'b1b8, 0xbca7'4c20},
                         {0xac83'2000, 0x6466'3964, 0x59c7'11c8, 0x8099'9d20},
                         {0x7377'7200, 0x3e6c'9c74, 0x36cf'75d8, 0x5093'f220},
                         {0x4673'c800, 0x247b'0384, 0x1fdf'dde8, 0x2c96'4b20},
                         {0x2578'2200, 0x1691'6e94, 0x14f8'49f8, 0x14a0'a820}},
                        {{0xfc4e'ad16'0902'0000, 0xa697'acf5'5df5'a988},
                         {0x4fde'a9cf'9fd6'4200, 0x0833'b3b7'0add'fd98},
                         {0xbf86'ba99'42b2'8800, 0x85e7'ce88'c3ce'55a8},
                         {0x4b46'df72'f196'd200, 0x1fb3'fd6a'88c6'b1b8},
                         {0xf31f'185c'ac83'2000, 0xd598'405c'59c7'11c8},
                         {0xb70f'6556'7377'7200, 0xa794'975e'36cf'75d8},
                         {0x9717'c660'4673'c800, 0x95a9'0270'1fdf'dde8},
                         {0x9338'3b7a'2578'2200, 0x9fd5'8192'14f8'49f8}},
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVmulvx,
      ExecMaskedVmulvx,
      {{0, 170, 84, 254, 168, 82, 252, 166, 80, 250, 164, 78, 248, 162, 76, 246},
       {160, 74, 244, 158, 72, 242, 156, 70, 240, 154, 68, 238, 152, 66, 236, 150},
       {64, 234, 148, 62, 232, 146, 60, 230, 144, 58, 228, 142, 56, 226, 140, 54},
       {224, 138, 52, 222, 136, 50, 220, 134, 48, 218, 132, 46, 216, 130, 44, 214},
       {128, 42, 212, 126, 40, 210, 124, 38, 208, 122, 36, 206, 120, 34, 204, 118},
       {32, 202, 116, 30, 200, 114, 28, 198, 112, 26, 196, 110, 24, 194, 108, 22},
       {192, 106, 20, 190, 104, 18, 188, 102, 16, 186, 100, 14, 184, 98, 12, 182},
       {96, 10, 180, 94, 8, 178, 92, 6, 176, 90, 4, 174, 88, 2, 172, 86}},
      {{0xaa00, 0x5354, 0xfca8, 0xa5fc, 0x4f50, 0xf8a4, 0xa1f8, 0x4b4c},
       {0xf4a0, 0x9df4, 0x4748, 0xf09c, 0x99f0, 0x4344, 0xec98, 0x95ec},
       {0x3f40, 0xe894, 0x91e8, 0x3b3c, 0xe490, 0x8de4, 0x3738, 0xe08c},
       {0x89e0, 0x3334, 0xdc88, 0x85dc, 0x2f30, 0xd884, 0x81d8, 0x2b2c},
       {0xd480, 0x7dd4, 0x2728, 0xd07c, 0x79d0, 0x2324, 0xcc78, 0x75cc},
       {0x1f20, 0xc874, 0x71c8, 0x1b1c, 0xc470, 0x6dc4, 0x1718, 0xc06c},
       {0x69c0, 0x1314, 0xbc68, 0x65bc, 0x0f10, 0xb864, 0x61b8, 0x0b0c},
       {0xb460, 0x5db4, 0x0708, 0xb05c, 0x59b0, 0x0304, 0xac58, 0x55ac}},
      {{0x5353'aa00, 0xfb50'fca8, 0xa34e'4f50, 0x4b4b'a1f8},
       {0xf348'f4a0, 0x9b46'4748, 0x4343'99f0, 0xeb40'ec98},
       {0x933e'3f40, 0x3b3b'91e8, 0xe338'e490, 0x8b36'3738},
       {0x3333'89e0, 0xdb30'dc88, 0x832e'2f30, 0x2b2b'81d8},
       {0xd328'd480, 0x7b26'2728, 0x2323'79d0, 0xcb20'cc78},
       {0x731e'1f20, 0x1b1b'71c8, 0xc318'c470, 0x6b16'1718},
       {0x1313'69c0, 0xbb10'bc68, 0x630e'0f10, 0x0b0b'61b8},
       {0xb308'b460, 0x5b06'0708, 0x0303'59b0, 0xab00'ac58}},
      {{0xa5fb'a752'5353'aa00, 0x4b4b'a1f7'a34e'4f50},
       {0xf09b'9c9c'f348'f4a0, 0x95eb'9742'4343'99f0},
       {0x3b3b'91e7'933e'3f40, 0xe08b'8c8c'e338'e490},
       {0x85db'8732'3333'89e0, 0x2b2b'81d7'832e'2f30},
       {0xd07b'7c7c'd328'd480, 0x75cb'7722'2323'79d0},
       {0x1b1b'71c7'731e'1f20, 0xc06b'6c6c'c318'c470},
       {0x65bb'6712'1313'69c0, 0x0b0b'61b7'630e'0f10},
       {0xb05b'5c5c'b308'b460, 0x55ab'5702'0303'59b0}},
      kVectorCalculationsSourceLegacy);
  TestVectorFloatInstruction(ExecVfmulvv,
                             ExecMaskedVfmulvv,
                             {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
                              {0x8000'02f0, 0x85ca'89ec, 0x91d9'a3e9, 0x9de9'3ee6},
                              {0xa9f9'5ae5, 0xb604'fbf4, 0xc20d'8af5, 0xce16'5a77},
                              {0xda1f'6a7a, 0xe628'bafe, 0xf232'4c02, 0xfe3c'1d87},
                              {0x0a49'9dd9, 0x165a'3ee4, 0x226b'60ef, 0x2e7d'03f9},
                              {0x3a87'9403, 0x4690'e68c, 0x529a'7994, 0x5ea4'4d1d},
                              {0x6aae'6126, 0x76b8'b5b2, 0x7f80'0000, 0x7f80'0000},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x8553'e032'b59e'2bf7, 0x9d6b'012b'925d'8532},
                              {0xb584'0511'cdec'af2c, 0xcd9b'2e22'd263'd03f},
                              {0xe5b4'2a11'269b'b302, 0xfdcb'5b3a'52ca'9bed},
                              {0x15e4'4f30'bfab'3779, 0x2dfb'8872'1391'e83b},
                              {0x4614'7470'991b'3c90, 0x5e2b'b5ca'14b9'b52b},
                              {0x7644'99d0'b2eb'c249, 0x7ff0'0000'0000'0000},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSourceLegacy);
  TestVectorFloatInstruction(ExecVfmulvf,
                             ExecMaskedVfmulvf,
                             {{0x8437'8568, 0x883d'2b0e, 0x8c42'd0b3, 0x9048'7659},
                              {0x944e'1bfe, 0x9853'c1a4, 0x9c59'674a, 0xa05f'0cef},
                              {0xa464'b295, 0xa86a'583b, 0xac6f'fde0, 0xb075'a386},
                              {0xb47b'492c, 0xb880'7769, 0xbc83'4a3b, 0xc086'1d0e},
                              {0xc488'efe1, 0xc88b'c2b4, 0xcc8e'9587, 0xd091'6859},
                              {0xd494'3b2c, 0xd897'0dff, 0xdc99'e0d2, 0xe09c'b3a5},
                              {0xe49f'8678, 0xe8a2'594a, 0xeca5'2c1d, 0xf0a7'fef0},
                              {0xf4aa'd1c3, 0xf8ad'a496, 0xfcb0'7768, 0xff80'0000}},
                             {{0x872f'ab0e'583b'8568, 0x8f35'7b2c'd1c3'685a},
                              {0x973f'c1a4'eed2'1bfe, 0x9f45'8678'1d0e'b3a5},
                              {0xa74f'd83b'8568'b295, 0xaf55'91c3'6859'fef0},
                              {0xb75f'eed2'1bff'492c, 0xbf65'9d0e'b3a5'4a3b},
                              {0xc770'02b4'594a'efe1, 0xcf75'a859'fef0'9587},
                              {0xd780'0dff'a496'3b2c, 0xdf85'b3a5'4a3b'e0d2},
                              {0xe790'194a'efe1'8678, 0xef95'bef0'9587'2c1d},
                              {0xf7a0'2496'3b2c'd1c3, 0xffa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSourceLegacy);
  TestWideningVectorFloatInstruction(ExecVfwmulvv,
                                     ExecMaskedVfwmulvv,
                                     {{0x3330'e53c'6480'0000, 0x34b2'786b'bbc5'4900},
                                      {0x3234'1766'da4a'6200, 0x33b5'cab6'2d6c'4800},
                                      {0x3937'92ba'5bd0'8000, 0x3ab9'666a'779a'0d00},
                                      {0x383b'4565'd61f'6600, 0x39bd'3935'e5bd'8800},
                                      {0x3f3f'423b'5522'0000, 0x40c0'ab36'1ab7'e880},
                                      {0x3e41'bab3'e9fa'b500, 0x3fc2'd4dc'5007'e400},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwmulvf,
                                     ExecMaskedVfwmulvf,
                                     {{0xb886'f0ad'0000'0000, 0xb907'a561'b400'0000},
                                      {0xb988'5a16'6800'0000, 0xba09'0ecb'1c00'0000},
                                      {0xba89'c37f'd000'0000, 0xbb0a'7834'8400'0000},
                                      {0xbb8b'2ce9'3800'0000, 0xbc0b'e19d'ec00'0000},
                                      {0xbc8c'9652'a000'0000, 0xbd0d'4b07'5400'0000},
                                      {0xbd8d'ffbc'0800'0000, 0xbe0e'b470'bc00'0000},
                                      {0xbe8f'6925'7000'0000, 0xbf10'0eed'1200'0000},
                                      {0xbf90'6947'6c00'0000, 0xc010'c3a1'c600'0000}},
                                     kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVmulhvv() {
  asm("vmulh.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmulhvv() {
  asm("vmulh.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVmulhvx() {
  asm("vmulh.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmulhvx() {
  asm("vmulh.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmulh) {
  TestVectorInstruction(ExecVmulhvv,
                        ExecMaskedVmulhvv,
                        {{0, 255, 0, 253, 0, 251, 0, 249, 0, 247, 0, 245, 1, 244, 1, 242},
                         {2, 241, 2, 239, 3, 238, 3, 237, 4, 235, 5, 234, 6, 233, 7, 232},
                         {8, 231, 9, 230, 10, 229, 11, 228, 12, 228, 13, 227, 15, 226, 16, 226},
                         {18, 225, 19, 225, 21, 224, 22, 224, 24, 224, 26, 224, 28, 224, 30, 224},
                         {224, 31, 224, 29, 224, 27, 224, 25, 224, 23, 224, 21, 225, 20, 225, 18},
                         {226, 17, 226, 15, 227, 14, 227, 13, 228, 11, 229, 10, 230, 9, 231, 8},
                         {232, 7, 233, 6, 234, 5, 235, 4, 236, 4, 237, 3, 239, 2, 240, 2},
                         {242, 1, 243, 1, 245, 0, 246, 0, 248, 0, 250, 0, 252, 0, 254, 0}},
                        {{0xff02, 0xfd10, 0xfb2d, 0xf95c, 0xf79a, 0xf5e9, 0xf448, 0xf2b7},
                         {0xf136, 0xefc5, 0xee64, 0xed13, 0xebd2, 0xeaa2, 0xe982, 0xe872},
                         {0xe772, 0xe682, 0xe5a2, 0xe4d3, 0xe413, 0xe364, 0xe2c4, 0xe235},
                         {0xe1b6, 0xe147, 0xe0e8, 0xe09a, 0xe05b, 0xe02d, 0xe00f, 0xe001},
                         {0x1ec3, 0x1cd3, 0x1af3, 0x1923, 0x1764, 0x15b4, 0x1415, 0x1286},
                         {0x1107, 0x0f98, 0x0e39, 0x0ceb, 0x0bac, 0x0a7e, 0x095f, 0x0851},
                         {0x0753, 0x0665, 0x0588, 0x04ba, 0x03fc, 0x034f, 0x02b2, 0x0225},
                         {0x01a8, 0x013b, 0x00de, 0x0091, 0x0055, 0x0028, 0x000c, 0x0000}},
                        {{0xfd10'1a16, 0xf95c'aad6, 0xf5e9'bc58, 0xf2b7'4e9b},
                         {0xefc5'619f, 0xed13'f564, 0xeaa3'09ea, 0xe872'9f31},
                         {0xe682'b539, 0xe4d3'4c01, 0xe364'638b, 0xe235'fbd7},
                         {0xe148'14e2, 0xe09a'aeaf, 0xe02d'c93d, 0xe001'648c},
                         {0x1cd2'bf5c, 0x1923'5829, 0x15b4'71b7, 0x1286'0c06},
                         {0x0f98'2716, 0x0cea'c2e7, 0x0a7d'df79, 0x0851'7ccc},
                         {0x0665'9ae0, 0x04ba'39b5, 0x034f'594b, 0x0224'f9a2},
                         {0x013b'1aba, 0x0091'bc93, 0x0028'df2d, 0x0000'8288}},
                        {{0xf95c'aad6'78f5'63b8, 0xf2b7'4e9b'bf9d'55cb},
                         {0xed13'f564'2968'6900, 0xe872'9f31'6a0c'5913},
                         {0xe4d3'4c01'edf3'8a67, 0xe235'fbd7'2893'787a},
                         {0xe09a'aeaf'c696'c7ef, 0xe001'648c'fb32'b402},
                         {0x1923'5828'f00f'6056, 0x1286'0c06'169f'4261},
                         {0x0cea'c2e6'e0d2'c60e, 0x0851'7ccc'015e'a619},
                         {0x04ba'39b4'e5ae'47e6, 0x0224'f9a2'0036'25f1},
                         {0x0091'bc92'fea1'e5de, 0x0000'8288'1325'c1e9}},
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVmulhvx,
      ExecMaskedVmulhvx,
      {{0, 42, 255, 41, 254, 41, 253, 40, 253, 39, 252, 39, 251, 38, 251, 37},
       {250, 37, 249, 36, 249, 35, 248, 35, 247, 34, 247, 33, 246, 33, 245, 32},
       {245, 31, 244, 31, 243, 30, 243, 29, 242, 29, 241, 28, 241, 27, 240, 27},
       {239, 26, 239, 25, 238, 25, 237, 24, 237, 23, 236, 23, 235, 22, 235, 21},
       {234, 21, 233, 20, 233, 19, 232, 19, 231, 18, 231, 17, 230, 17, 229, 16},
       {229, 15, 228, 15, 227, 14, 227, 13, 226, 13, 225, 12, 225, 11, 224, 11},
       {223, 10, 223, 9, 222, 9, 221, 8, 221, 7, 220, 7, 219, 6, 219, 5},
       {218, 5, 217, 4, 217, 3, 216, 3, 215, 2, 215, 1, 214, 1, 213, 0}},
      {{0x2a55, 0x29aa, 0x28fe, 0x2853, 0x27a8, 0x26fc, 0x2651, 0x25a6},
       {0x24fa, 0x244f, 0x23a4, 0x22f8, 0x224d, 0x21a2, 0x20f6, 0x204b},
       {0x1fa0, 0x1ef4, 0x1e49, 0x1d9e, 0x1cf2, 0x1c47, 0x1b9c, 0x1af0},
       {0x1a45, 0x199a, 0x18ee, 0x1843, 0x1798, 0x16ec, 0x1641, 0x1596},
       {0x14ea, 0x143f, 0x1394, 0x12e8, 0x123d, 0x1192, 0x10e6, 0x103b},  // NOTYPO
       {0x0f90, 0x0ee4, 0x0e39, 0x0d8e, 0x0ce2, 0x0c37, 0x0b8c, 0x0ae0},
       {0x0a35, 0x098a, 0x08de, 0x0833, 0x0788, 0x06dc, 0x0631, 0x0586},
       {0x04da, 0x042f, 0x0384, 0x02d8, 0x022d, 0x0182, 0x00d6, 0x002b}},
      {{0x29a9'd500, 0x2853'28fe, 0x26fc'7cfd, 0x25a5'd0fc},
       {0x244f'24fa, 0x22f8'78f9, 0x21a1'ccf8, 0x204b'20f6},
       {0x1ef4'74f5, 0x1d9d'c8f4, 0x1c47'1cf2, 0x1af0'70f1},
       {0x1999'c4f0, 0x1843'18ee, 0x16ec'6ced, 0x1595'c0ec},
       {0x143f'14ea, 0x12e8'68e9, 0x1191'bce8, 0x103b'10e6},  // NOTYPO
       {0x0ee4'64e5, 0x0d8d'b8e4, 0x0c37'0ce2, 0x0ae0'60e1},
       {0x0989'b4e0, 0x0833'08de, 0x06dc'5cdd, 0x0585'b0dc},
       {0x042f'04da, 0x02d8'58d9, 0x0181'acd8, 0x002b'00d6}},
      {{0x2853'28fe'7eff'2a55, 0x25a5'd0fb'd1a7'27a8},
       {0x22f8'78f9'244f'24fa, 0x204b'20f6'76f7'224d},
       {0x1d9d'c8f3'c99f'1fa0, 0x1af0'70f1'1c47'1cf2},
       {0x1843'18ee'6eef'1a45, 0x1595'c0eb'c197'1798},
       {0x12e8'68e9'143f'14ea, 0x103b'10e6'66e7'123d},  // NOTYPO
       {0x0d8d'b8e3'b98f'0f90, 0x0ae0'60e1'0c37'0ce2},
       {0x0833'08de'5edf'0a35, 0x0585'b0db'b187'0788},
       {0x02d8'58d9'042f'04da, 0x002b'00d6'56d7'022d}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVmulhuvv() {
  asm("vmulhu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmulhuvv() {
  asm("vmulhu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVmulhuvx() {
  asm("vmulhu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmulhuvx() {
  asm("vmulhu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmulhu) {
  TestVectorInstruction(
      ExecVmulhuvv,
      ExecMaskedVmulhuvv,
      {{0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 1, 14, 1, 16},
       {2, 19, 2, 21, 3, 24, 3, 27, 4, 29, 5, 32, 6, 35, 7, 38},
       {8, 41, 9, 44, 10, 47, 11, 50, 12, 54, 13, 57, 15, 60, 16, 64},
       {18, 67, 19, 71, 21, 74, 22, 78, 24, 82, 26, 86, 28, 90, 30, 94},
       {32, 98, 34, 102, 36, 106, 38, 110, 40, 114, 42, 118, 45, 123, 47, 127},
       {50, 132, 52, 136, 55, 141, 57, 146, 60, 150, 63, 155, 66, 160, 69, 165},
       {72, 170, 75, 175, 78, 180, 81, 185, 84, 191, 87, 196, 91, 201, 94, 207},
       {98, 212, 101, 218, 105, 223, 108, 229, 112, 235, 116, 241, 120, 247, 124, 253}},
      {{0x0102, 0x0314, 0x0536, 0x0768, 0x09ab, 0x0bfd, 0x0e60, 0x10d3},
       {0x1356, 0x15e9, 0x188d, 0x1b3f, 0x1e03, 0x20d6, 0x23ba, 0x26ae},
       {0x29b2, 0x2cc6, 0x2feb, 0x331f, 0x3664, 0x39b8, 0x3d1c, 0x4091},
       {0x4416, 0x47ab, 0x4b51, 0x4f06, 0x52cc, 0x56a1, 0x5a87, 0x5e7d},
       {0x6283, 0x6699, 0x6ac0, 0x6ef5, 0x733d, 0x7792, 0x7bf9, 0x8070},
       {0x84f7, 0x898e, 0x8e36, 0x92ed, 0x97b5, 0x9c8c, 0xa173, 0xa66b},
       {0xab73, 0xb08b, 0xb5b5, 0xbaec, 0xc035, 0xc58d, 0xcaf6, 0xd06f},
       {0xd5f8, 0xdb91, 0xe13b, 0xe6f3, 0xecbe, 0xf296, 0xf880, 0xfe7a}},
      {{0x0314'1c16, 0x0768'b4df, 0x0bfd'ce69, 0x10d3'68b3},
       {0x15e9'83bf, 0x1b40'1f8d, 0x20d7'3c1b, 0x26ae'd969},
       {0x2cc6'f779, 0x331f'964a, 0x39b8'b5dc, 0x4092'562f},
       {0x47ac'7742, 0x4f07'1918, 0x56a2'3bae, 0x5e7d'df04},
       {0x669a'031c, 0x6ef6'a7f6, 0x7793'cd90, 0x8071'73ea},
       {0x898f'9b06, 0x92ee'42e4, 0x9c8d'6b82, 0xa66d'14e0},
       {0xb08d'3f00, 0xbaed'e9e2, 0xc58f'1584, 0xd070'c1e6},
       {0xdb92'ef0a, 0xe6f5'9cf0, 0xf298'cb96, 0xfe7c'7afc}},
      {{0x0768'b4df'7ef9'65b8, 0x10d3'68b3'd5b1'67dc},
       {0x1b40'1f8d'4f8c'8b20, 0x26ae'd969'a040'8b44},
       {0x331f'964b'3437'cca7, 0x4092'562f'7ee7'cacb},
       {0x4f07'1919'2cfb'2a4f, 0x5e7d'df05'71a7'2673},
       {0x6ef6'a7f7'39d6'a416, 0x8071'73eb'787e'9e3a},
       {0x92ee'42e5'5aca'39fe, 0xa66d'14e1'936e'3222},
       {0xbaed'e9e3'8fd5'ec06, 0xd070'c1e7'c275'e22a},
       {0xe6f5'9cf1'd8f9'ba2e, 0xfe7c'7afe'0595'ae52}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVmulhuvx,
                        ExecMaskedVmulhuvx,
                        {{0, 85, 1, 86, 2, 88, 3, 89, 5, 90, 6, 92, 7, 93, 9, 94},
                         {10, 96, 11, 97, 13, 98, 14, 100, 15, 101, 17, 102, 18, 104, 19, 105},
                         {21, 106, 22, 108, 23, 109, 25, 110, 26, 112, 27, 113, 29, 114, 30, 116},
                         {31, 117, 33, 118, 34, 120, 35, 121, 37, 122, 38, 124, 39, 125, 41, 126},
                         {42, 128, 43, 129, 45, 130, 46, 132, 47, 133, 49, 134, 50, 136, 51, 137},
                         {53, 138, 54, 140, 55, 141, 57, 142, 58, 144, 59, 145, 61, 146, 62, 148},
                         {63, 149, 65, 150, 66, 152, 67, 153, 69, 154, 70, 156, 71, 157, 73, 158},
                         {74, 160, 75, 161, 77, 162, 78, 164, 79, 165, 81, 166, 82, 168, 83, 169}},
                        {{0x55ff, 0x5756, 0x58ac, 0x5a03, 0x5b5a, 0x5cb0, 0x5e07, 0x5f5e},
                         {0x60b4, 0x620b, 0x6362, 0x64b8, 0x660f, 0x6766, 0x68bc, 0x6a13},
                         {0x6b6a, 0x6cc0, 0x6e17, 0x6f6e, 0x70c4, 0x721b, 0x7372, 0x74c8},
                         {0x761f, 0x7776, 0x78cc, 0x7a23, 0x7b7a, 0x7cd0, 0x7e27, 0x7f7e},
                         {0x80d4, 0x822b, 0x8382, 0x84d8, 0x862f, 0x8786, 0x88dc, 0x8a33},
                         {0x8b8a, 0x8ce0, 0x8e37, 0x8f8e, 0x90e4, 0x923b, 0x9392, 0x94e8},
                         {0x963f, 0x9796, 0x98ec, 0x9a43, 0x9b9a, 0x9cf0, 0x9e47, 0x9f9e},
                         {0xa0f4, 0xa24b, 0xa3a2, 0xa4f8, 0xa64f, 0xa7a6, 0xa8fc, 0xaa53}},
                        {{0x5757'00aa, 0x5a04'58ac, 0x5cb1'b0af, 0x5f5f'08b2},
                         {0x620c'60b4, 0x64b9'b8b7, 0x6767'10ba, 0x6a14'68bc},
                         {0x6cc1'c0bf, 0x6f6f'18c2, 0x721c'70c4, 0x74c9'c8c7},
                         {0x7777'20ca, 0x7a24'78cc, 0x7cd1'd0cf, 0x7f7f'28d2},
                         {0x822c'80d4, 0x84d9'd8d7, 0x8787'30da, 0x8a34'88dc},
                         {0x8ce1'e0df, 0x8f8f'38e2, 0x923c'90e4, 0x94e9'e8e7},
                         {0x9797'40ea, 0x9a44'98ec, 0x9cf1'f0ef, 0x9f9f'48f2},
                         {0xa24c'a0f4, 0xa4f9'f8f7, 0xa7a7'50fa, 0xaa54'a8fc}},
                        {{0x5a04'58ad'acac'55ff, 0x5f5f'08b3'075c'5b5a},
                         {0x64b9'b8b8'620c'60b4, 0x6a14'68bd'bcbc'660f},
                         {0x6f6f'18c3'176c'6b6a, 0x74c9'c8c8'721c'70c4},
                         {0x7a24'78cd'cccc'761f, 0x7f7f'28d3'277c'7b7a},
                         {0x84d9'd8d8'822c'80d4, 0x8a34'88dd'dcdc'862f},
                         {0x8f8f'38e3'378c'8b8a, 0x94e9'e8e8'923c'90e4},
                         {0x9a44'98ed'ecec'963f, 0x9f9f'48f3'479c'9b9a},
                         {0xa4f9'f8f8'a24c'a0f4, 0xaa54'a8fd'fcfc'a64f}},
                        kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVnmsacvv() {
  asm("vnmsac.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnmsacvv() {
  asm("vnmsac.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnmsacvx() {
  asm("vnmsac.vx  v8, t0, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnmsacvx() {
  asm("vnmsac.vx  v8, t0, v16, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVnmsac) {
  TestVectorInstruction(ExecVnmsacvv,
                        ExecMaskedVnmsacvv,
                        {{85, 195, 77, 147, 49, 83, 13, 3, 205, 195, 141, 147, 53, 83, 205, 3},
                         {85, 131, 205, 211, 33, 19, 141, 67, 189, 131, 13, 211, 53, 19, 77, 67},
                         {85, 67, 77, 19, 17, 211, 13, 131, 173, 67, 141, 19, 53, 211, 205, 131},
                         {85, 3, 205, 83, 1, 147, 141, 195, 157, 3, 13, 83, 53, 147, 77, 195},
                         {85, 195, 77, 147, 241, 83, 13, 3, 141, 195, 141, 147, 53, 83, 205, 3},
                         {85, 131, 205, 211, 225, 19, 141, 67, 125, 131, 13, 211, 53, 19, 77, 67},
                         {85, 67, 77, 19, 209, 211, 13, 131, 109, 67, 141, 19, 53, 211, 205, 131},
                         {85, 3, 205, 83, 193, 147, 141, 195, 93, 3, 13, 83, 53, 147, 77, 195}},
                        {{0x5555, 0x1d4d, 0x4031, 0x4d0d, 0x2bcd, 0x3c8d, 0xa435, 0xebcd},
                         {0x1355, 0xdacd, 0xed21, 0x098d, 0xd7bd, 0xf80d, 0x5f35, 0xa64d},
                         {0xcd55, 0x944d, 0x9611, 0xc20d, 0x7fad, 0xaf8d, 0x1635, 0x5ccd},
                         {0x8355, 0x49cd, 0x3b01, 0x768d, 0x239d, 0x630d, 0xc935, 0x0f4d},
                         {0x3555, 0xfb4d, 0xdbf1, 0x270d, 0xc38d, 0x128d, 0x7835, 0xbdcd},
                         {0xe355, 0xa8cd, 0x78e1, 0xd38d, 0x5f7d, 0xbe0d, 0x2335, 0x684d},
                         {0x8d55, 0x524d, 0x11d1, 0x7c0d, 0xf76d, 0x658d, 0xca35, 0x0ecd},
                         {0x3355, 0xf7cd, 0xa6c1, 0x208d, 0x8b5d, 0x090d, 0x6d35, 0xb14d}},
                        {{0xe3c3'5555, 0xf5e6'4031, 0xdb6c'2bcd, 0xfe9f'a435},
                         {0x43e6'1355, 0x42f6'ed21, 0x277c'd7bd, 0x57be'5f35},
                         {0x9800'cd55, 0x83ff'9611, 0x6785'7fad, 0xa4d5'1635},
                         {0xe013'8355, 0xb900'3b01, 0x9b86'239d, 0xe5e3'c935},
                         {0xdd5e'3555, 0xa73c'dbf1, 0x8cc6'c38d, 0xe836'7835},
                         {0x1d70'e355, 0xd43d'78e1, 0xb8c7'5f7d, 0x2145'2335},
                         {0x517b'8d55, 0xf536'11d1, 0xd8bf'f76d, 0x4e4b'ca35},
                         {0x797e'3355, 0x0a26'a6c1, 0xecb0'8b5d, 0x6f4a'6d35}},
                        {{0xbbf2'b5fd'e3c3'5555, 0xb840'0f97'db6c'2bcd},
                         {0x4d47'a732'43e6'1355, 0x418e'fac8'277c'd7bd},
                         {0xc284'8456'9800'cd55, 0xaec5'd1e8'6785'7fad},
                         {0x1ba9'4d6a'e013'8355, 0xffe4'94f8'9b86'239d},
                         {0xa27d'86f1'dd5e'3555, 0x96ca'd88b'8cc6'c38d},
                         {0xf3a2'4806'1d70'e355, 0xdfe9'939b'b8c7'5f7d},
                         {0x28ae'f50a'517b'8d55, 0x0cf0'3a9b'd8bf'f76d},
                         {0x41a3'8dfe'797e'3355, 0x1dde'cd8b'ecb0'8b5d}},
                        kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVnmsacvx,
      ExecMaskedVnmsacvx,
      {{85, 171, 1, 87, 173, 3, 89, 175, 5, 91, 177, 7, 93, 179, 9, 95},
       {181, 11, 97, 183, 13, 99, 185, 15, 101, 187, 17, 103, 189, 19, 105, 191},
       {21, 107, 193, 23, 109, 195, 25, 111, 197, 27, 113, 199, 29, 115, 201, 31},
       {117, 203, 33, 119, 205, 35, 121, 207, 37, 123, 209, 39, 125, 211, 41, 127},
       {213, 43, 129, 215, 45, 131, 217, 47, 133, 219, 49, 135, 221, 51, 137, 223},
       {53, 139, 225, 55, 141, 227, 57, 143, 229, 59, 145, 231, 61, 147, 233, 63},
       {149, 235, 65, 151, 237, 67, 153, 239, 69, 155, 241, 71, 157, 243, 73, 159},
       {245, 75, 161, 247, 77, 163, 249, 79, 165, 251, 81, 167, 253, 83, 169, 255}},
      {{0xab55, 0x0201, 0x58ad, 0xaf59, 0x0605, 0x5cb1, 0xb35d, 0x0a09},
       {0x60b5, 0xb761, 0x0e0d, 0x64b9, 0xbb65, 0x1211, 0x68bd, 0xbf69},
       {0x1615, 0x6cc1, 0xc36d, 0x1a19, 0x70c5, 0xc771, 0x1e1d, 0x74c9},
       {0xcb75, 0x2221, 0x78cd, 0xcf79, 0x2625, 0x7cd1, 0xd37d, 0x2a29},
       {0x80d5, 0xd781, 0x2e2d, 0x84d9, 0xdb85, 0x3231, 0x88dd, 0xdf89},
       {0x3635, 0x8ce1, 0xe38d, 0x3a39, 0x90e5, 0xe791, 0x3e3d, 0x94e9},
       {0xeb95, 0x4241, 0x98ed, 0xef99, 0x4645, 0x9cf1, 0xf39d, 0x4a49},
       {0xa0f5, 0xf7a1, 0x4e4d, 0xa4f9, 0xfba5, 0x5251, 0xa8fd, 0xffa9}},
      {{0x0201'ab55, 0x5a04'58ad, 0xb207'0605, 0x0a09'b35d},
       {0x620c'60b5, 0xba0f'0e0d, 0x1211'bb65, 0x6a14'68bd},
       {0xc217'1615, 0x1a19'c36d, 0x721c'70c5, 0xca1f'1e1d},
       {0x2221'cb75, 0x7a24'78cd, 0xd227'2625, 0x2a29'd37d},
       {0x822c'80d5, 0xda2f'2e2d, 0x3231'db85, 0x8a34'88dd},
       {0xe237'3635, 0x3a39'e38d, 0x923c'90e5, 0xea3f'3e3d},
       {0x4241'eb95, 0x9a44'98ed, 0xf247'4645, 0x4a49'f39d},
       {0xa24c'a0f5, 0xfa4f'4e4d, 0x5251'fba5, 0xaa54'a8fd}},
      {{0xaf59'ae03'0201'ab55, 0x0a09'b35d'b207'0605},
       {0x64b9'b8b8'620c'60b5, 0xbf69'be13'1211'bb65},
       {0x1a19'c36d'c217'1615, 0x74c9'c8c8'721c'70c5},
       {0xcf79'ce23'2221'cb75, 0x2a29'd37d'd227'2625},
       {0x84d9'd8d8'822c'80d5, 0xdf89'de33'3231'db85},
       {0x3a39'e38d'e237'3635, 0x94e9'e8e8'923c'90e5},
       {0xef99'ee43'4241'eb95, 0x4a49'f39d'f247'4645},
       {0xa4f9'f8f8'a24c'a0f5, 0xffa9'fe53'5251'fba5}},
      kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVnmsubvv() {
  asm("vnmsub.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnmsubvv() {
  asm("vnmsub.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnmsubvx() {
  asm("vnmsub.vx  v8, t0, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnmsubvx() {
  asm("vnmsub.vx  v8, t0, v16, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVnmsub) {
  TestVectorInstruction(
      ExecVnmsubvv,
      ExecMaskedVnmsubvv,
      {{0, 189, 90, 23, 181, 113, 14, 203, 105, 5, 194, 95, 28, 185, 118, 19},
       {208, 141, 42, 231, 133, 65, 222, 155, 57, 213, 146, 47, 236, 137, 70, 227},
       {160, 93, 250, 183, 85, 17, 174, 107, 9, 165, 98, 255, 188, 89, 22, 179},
       {112, 45, 202, 135, 37, 225, 126, 59, 217, 117, 50, 207, 140, 41, 230, 131},
       {64, 253, 154, 87, 245, 177, 78, 11, 169, 69, 2, 159, 92, 249, 182, 83},
       {16, 205, 106, 39, 197, 129, 30, 219, 121, 21, 210, 111, 44, 201, 134, 35},
       {224, 157, 58, 247, 149, 81, 238, 171, 73, 229, 162, 63, 252, 153, 86, 243},
       {176, 109, 10, 199, 101, 33, 190, 123, 25, 181, 114, 15, 204, 105, 38, 195}},
      {{0xbd00, 0x6c5a, 0x1bb5, 0xcb0e, 0x5a69, 0x09c2, 0xb91c, 0x6876},
       {0x37d0, 0xe72a, 0x9685, 0x45de, 0xd539, 0x8492, 0x33ec, 0xe346},
       {0xb2a0, 0x61fa, 0x1155, 0xc0ae, 0x5009, 0xff62, 0xaebc, 0x5e16},
       {0x2d70, 0xdcca, 0x8c25, 0x3b7e, 0xcad9, 0x7a32, 0x298c, 0xd8e6},
       {0xa840, 0x579a, 0x06f5, 0xb64e, 0x45a9, 0xf502, 0xa45c, 0x53b6},
       {0x2310, 0xd26a, 0x81c5, 0x311e, 0xc079, 0x6fd2, 0x1f2c, 0xce86},
       {0x9de0, 0x4d3a, 0xfc95, 0xabee, 0x3b49, 0xeaa2, 0x99fc, 0x4956},
       {0x18b0, 0xc80a, 0x7765, 0x26be, 0xb619, 0x6572, 0x14cc, 0xc426}},
      {{0x6c5a'bd00, 0x2064'1bb5, 0xb46d'5a69, 0x6876'b91c},
       {0x3c80'37d0, 0xf089'9685, 0x8492'd539, 0x389c'33ec},
       {0x0ca5'b2a0, 0xc0af'1155, 0x54b8'5009, 0x08c1'aebc},
       {0xdccb'2d70, 0x90d4'8c25, 0x24dd'cad9, 0xd8e7'298c},
       {0xacef'a840, 0x60f9'06f5, 0xf502'45a9, 0xa90b'a45c},
       {0x7d15'2310, 0x311e'81c5, 0xc527'c079, 0x7931'1f2c},
       {0x4d3a'9de0, 0x0143'fc95, 0x954d'3b49, 0x4956'99fc},
       {0x1d60'18b0, 0xd169'7765, 0x6572'b619, 0x197c'14cc}},
      {{0xcb0e'c660'6c5a'bd00, 0x6876'b91c'b46d'5a69},
       {0x45de'ebdb'3c80'37d0, 0xe346'de97'8492'd539},
       {0xc0af'1156'0ca5'b2a0, 0x5e17'0412'54b8'5009},
       {0x3b7f'36d0'dccb'2d70, 0xd8e7'298d'24dd'cad9},
       {0xb64e'5c4a'acef'a840, 0x53b6'4f06'f502'45a9},
       {0x311e'81c5'7d15'2310, 0xce86'7481'c527'c079},
       {0xabee'a740'4d3a'9de0, 0x4956'99fc'954d'3b49},
       {0x26be'ccbb'1d60'18b0, 0xc426'bf77'6572'b619}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVnmsubvx,
      ExecMaskedVnmsubvx,
      {{142, 15, 144, 17, 146, 19, 148, 21, 150, 23, 152, 25, 154, 27, 156, 29},
       {158, 31, 160, 33, 162, 35, 164, 37, 166, 39, 168, 41, 170, 43, 172, 45},
       {174, 47, 176, 49, 178, 51, 180, 53, 182, 55, 184, 57, 186, 59, 188, 61},
       {190, 63, 192, 65, 194, 67, 196, 69, 198, 71, 200, 73, 202, 75, 204, 77},
       {206, 79, 208, 81, 210, 83, 212, 85, 214, 87, 216, 89, 218, 91, 220, 93},
       {222, 95, 224, 97, 226, 99, 228, 101, 230, 103, 232, 105, 234, 107, 236, 109},
       {238, 111, 240, 113, 242, 115, 244, 117, 246, 119, 248, 121, 250, 123, 252, 125},
       {254, 127, 0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141}},
      {{0x648e, 0x6690, 0x6892, 0x6a94, 0x6c96, 0x6e98, 0x709a, 0x729c},
       {0x749e, 0x76a0, 0x78a2, 0x7aa4, 0x7ca6, 0x7ea8, 0x80aa, 0x82ac},
       {0x84ae, 0x86b0, 0x88b2, 0x8ab4, 0x8cb6, 0x8eb8, 0x90ba, 0x92bc},
       {0x94be, 0x96c0, 0x98c2, 0x9ac4, 0x9cc6, 0x9ec8, 0xa0ca, 0xa2cc},
       {0xa4ce, 0xa6d0, 0xa8d2, 0xaad4, 0xacd6, 0xaed8, 0xb0da, 0xb2dc},
       {0xb4de, 0xb6e0, 0xb8e2, 0xbae4, 0xbce6, 0xbee8, 0xc0ea, 0xc2ec},
       {0xc4ee, 0xc6f0, 0xc8f2, 0xcaf4, 0xccf6, 0xcef8, 0xd0fa, 0xd2fc},
       {0xd4fe, 0xd700, 0xd902, 0xdb04, 0xdd06, 0xdf08, 0xe10a, 0xe30c}},
      {{0x113b'648e, 0x153f'6892, 0x1943'6c96, 0x1d47'709a},
       {0x214b'749e, 0x254f'78a2, 0x2953'7ca6, 0x2d57'80aa},
       {0x315b'84ae, 0x355f'88b2, 0x3963'8cb6, 0x3d67'90ba},
       {0x416b'94be, 0x456f'98c2, 0x4973'9cc6, 0x4d77'a0ca},
       {0x517b'a4ce, 0x557f'a8d2, 0x5983'acd6, 0x5d87'b0da},
       {0x618b'b4de, 0x658f'b8e2, 0x6993'bce6, 0x6d97'c0ea},
       {0x719b'c4ee, 0x759f'c8f2, 0x79a3'ccf6, 0x7da7'd0fa},
       {0x81ab'd4fe, 0x85af'd902, 0x89b3'dd06, 0x8db7'e10a}},
      {{0x6a94'bde8'113b'648e, 0x729c'c5f0'1943'6c96},
       {0x7aa4'cdf8'214b'749e, 0x82ac'd600'2953'7ca6},
       {0x8ab4'de08'315b'84ae, 0x92bc'e610'3963'8cb6},
       {0x9ac4'ee18'416b'94be, 0xa2cc'f620'4973'9cc6},
       {0xaad4'fe28'517b'a4ce, 0xb2dd'0630'5983'acd6},
       {0xbae5'0e38'618b'b4de, 0xc2ed'1640'6993'bce6},
       {0xcaf5'1e48'719b'c4ee, 0xd2fd'2650'79a3'ccf6},
       {0xdb05'2e58'81ab'd4fe, 0xe30d'3660'89b3'dd06}},
      kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVnsrawi() {
  asm("vnsra.wi  v8, v16, 3\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnsrawi() {
  asm("vnsra.wi  v8, v16, 3, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnsrawx() {
  asm("vnsra.wx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnsrawx() {
  asm("vnsra.wx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnsrawv() {
  asm("vnsra.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnsrawv() {
  asm("vnsra.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVnsra) {
  TestNarrowingVectorInstruction(
      ExecVnsrawi,
      ExecMaskedVnsrawi,
      {{32, 96, 160, 224, 33, 97, 161, 225, 34, 98, 162, 226, 35, 99, 163, 227},
       {36, 100, 164, 228, 37, 101, 165, 229, 38, 102, 166, 230, 39, 103, 167, 231},
       {40, 104, 168, 232, 41, 105, 169, 233, 42, 106, 170, 234, 43, 107, 171, 235},
       {44, 108, 172, 236, 45, 109, 173, 237, 46, 110, 174, 238, 47, 111, 175, 239}},
      {{0x5020, 0xd0a0, 0x5121, 0xd1a1, 0x5222, 0xd2a2, 0x5323, 0xd3a3},
       {0x5424, 0xd4a4, 0x5525, 0xd5a5, 0x5626, 0xd6a6, 0x5727, 0xd7a7},
       {0x5828, 0xd8a8, 0x5929, 0xd9a9, 0x5a2a, 0xdaaa, 0x5b2b, 0xdbab},
       {0x5c2c, 0xdcac, 0x5d2d, 0xddad, 0x5e2e, 0xdeae, 0x5f2f, 0xdfaf}},
      {{0x9060'5020, 0x9161'5121, 0x9262'5222, 0x9363'5323},
       {0x9464'5424, 0x9565'5525, 0x9666'5626, 0x9767'5727},
       {0x9868'5828, 0x9969'5929, 0x9a6a'5a2a, 0x9b6b'5b2b},
       {0x9c6c'5c2c, 0x9d6d'5d2d, 0x9e6e'5e2e, 0x9f6f'5f2f}},
      kVectorCalculationsSourceLegacy);
  TestNarrowingVectorInstruction(
      ExecVnsrawx,
      ExecMaskedVnsrawx,
      {{224, 224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231},
       {232, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239},
       {240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247},
       {248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255}},
      {{0xc0a0, 0xc1a1, 0xc2a2, 0xc3a3, 0xc4a4, 0xc5a5, 0xc6a6, 0xc7a7},
       {0xc8a8, 0xc9a9, 0xcaaa, 0xcbab, 0xccac, 0xcdad, 0xceae, 0xcfaf},
       {0xd0b0, 0xd1b1, 0xd2b2, 0xd3b3, 0xd4b4, 0xd5b5, 0xd6b6, 0xd7b7},
       {0xd8b8, 0xd9b9, 0xdaba, 0xdbbb, 0xdcbc, 0xddbd, 0xdebe, 0xdfbf}},
      {{0xffe1'c1a1, 0xffe3'c3a3, 0xffe5'c5a5, 0xffe7'c7a7},
       {0xffe9'c9a9, 0xffeb'cbab, 0xffed'cdad, 0xffef'cfaf},
       {0xfff1'd1b1, 0xfff3'd3b3, 0xfff5'd5b5, 0xfff7'd7b7},
       {0xfff9'd9b9, 0xfffb'dbbb, 0xfffd'ddbd, 0xffff'dfbf}},
      kVectorCalculationsSourceLegacy);
  TestNarrowingVectorInstruction(
      ExecVnsrawv,
      ExecMaskedVnsrawv,
      {{0, 192, 80, 28, 196, 226, 248, 254, 136, 196, 81, 92, 153, 230, 249, 254},
       {32, 200, 82, 156, 212, 234, 250, 254, 152, 204, 83, 220, 185, 238, 251, 254},
       {64, 208, 84, 29, 228, 242, 252, 255, 168, 212, 85, 93, 217, 246, 253, 255},
       {96, 216, 86, 157, 244, 250, 254, 255, 184, 220, 87, 221, 249, 254, 255, 255}},
      {{0x8100, 0x6850, 0x8544, 0xf0e8, 0xc989, 0xf971, 0xff9b, 0xfff9},
       {0xa120, 0x6a52, 0x9554, 0xf2ea, 0xd999, 0xfb73, 0xffbb, 0xfffb},
       {0xc140, 0x6c54, 0xa564, 0xf4ec, 0xe9a9, 0xfd75, 0xffdb, 0xfffd},
       {0xe160, 0x6e56, 0xb574, 0xf6ee, 0xf9b9, 0xff77, 0xfffb, 0xffff}},
      {{0x8302'8100, 0x8645'8544, 0x4a8a'4989, 0x1e9d'1c9b},
       {0xa726'a524, 0xffd7'9756, 0xffff'db9b, 0xffff'ffbf},
       {0xc342'c140, 0xa665'a564, 0x6aaa'69a9, 0x5edd'5cdb},
       {0xe766'e564, 0xfff7'b776, 0xffff'fbbb, 0xffff'ffff}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVnsrlwi() {
  asm("vnsrl.wi  v8, v16, 3\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnsrlwi() {
  asm("vnsrl.wi  v8, v16, 3, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnsrlwx() {
  asm("vnsrl.wx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnsrlwx() {
  asm("vnsrl.wx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnsrlwv() {
  asm("vnsrl.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnsrlwv() {
  asm("vnsrl.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVnsrl) {
  TestNarrowingVectorInstruction(
      ExecVnsrlwi,
      ExecMaskedVnsrlwi,
      {{32, 96, 160, 224, 33, 97, 161, 225, 34, 98, 162, 226, 35, 99, 163, 227},
       {36, 100, 164, 228, 37, 101, 165, 229, 38, 102, 166, 230, 39, 103, 167, 231},
       {40, 104, 168, 232, 41, 105, 169, 233, 42, 106, 170, 234, 43, 107, 171, 235},
       {44, 108, 172, 236, 45, 109, 173, 237, 46, 110, 174, 238, 47, 111, 175, 239}},
      {{0x5020, 0xd0a0, 0x5121, 0xd1a1, 0x5222, 0xd2a2, 0x5323, 0xd3a3},
       {0x5424, 0xd4a4, 0x5525, 0xd5a5, 0x5626, 0xd6a6, 0x5727, 0xd7a7},
       {0x5828, 0xd8a8, 0x5929, 0xd9a9, 0x5a2a, 0xdaaa, 0x5b2b, 0xdbab},
       {0x5c2c, 0xdcac, 0x5d2d, 0xddad, 0x5e2e, 0xdeae, 0x5f2f, 0xdfaf}},
      {{0x9060'5020, 0x9161'5121, 0x9262'5222, 0x9363'5323},
       {0x9464'5424, 0x9565'5525, 0x9666'5626, 0x9767'5727},
       {0x9868'5828, 0x9969'5929, 0x9a6a'5a2a, 0x9b6b'5b2b},
       {0x9c6c'5c2c, 0x9d6d'5d2d, 0x9e6e'5e2e, 0x9f6f'5f2f}},
      kVectorCalculationsSourceLegacy);
  TestNarrowingVectorInstruction(ExecVnsrlwx,
                                 ExecMaskedVnsrlwx,
                                 {{32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39},
                                  {40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47},
                                  {48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55},
                                  {56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63}},
                                 {{0xc0a0, 0xc1a1, 0xc2a2, 0xc3a3, 0xc4a4, 0xc5a5, 0xc6a6, 0xc7a7},
                                  {0xc8a8, 0xc9a9, 0xcaaa, 0xcbab, 0xccac, 0xcdad, 0xceae, 0xcfaf},
                                  {0xd0b0, 0xd1b1, 0xd2b2, 0xd3b3, 0xd4b4, 0xd5b5, 0xd6b6, 0xd7b7},
                                  {0xd8b8, 0xd9b9, 0xdaba, 0xdbbb, 0xdcbc, 0xddbd, 0xdebe, 0xdfbf}},
                                 {{0x0021'c1a1, 0x0023'c3a3, 0x0025'c5a5, 0x0027'c7a7},
                                  {0x0029'c9a9, 0x002b'cbab, 0x002d'cdad, 0x002f'cfaf},
                                  {0x0031'd1b1, 0x0033'd3b3, 0x0035'd5b5, 0x0037'd7b7},
                                  {0x0039'd9b9, 0x003b'dbbb, 0x003d'ddbd, 0x003f'dfbf}},
                                 kVectorCalculationsSourceLegacy);
  TestNarrowingVectorInstruction(
      ExecVnsrlwv,
      ExecMaskedVnsrlwv,
      {{0, 192, 80, 28, 68, 34, 8, 2, 136, 196, 81, 92, 153, 38, 9, 2},
       {32, 200, 82, 156, 84, 42, 10, 2, 152, 204, 83, 220, 185, 46, 11, 2},
       {64, 208, 84, 29, 100, 50, 12, 3, 168, 212, 85, 93, 217, 54, 13, 3},
       {96, 216, 86, 157, 116, 58, 14, 3, 184, 220, 87, 221, 249, 62, 15, 3}},
      {{0x8100, 0x6850, 0x8544, 0xf0e8, 0x4989, 0x0971, 0x009b, 0x0009},
       {0xa120, 0x6a52, 0x9554, 0xf2ea, 0x5999, 0x0b73, 0x00bb, 0x000b},
       {0xc140, 0x6c54, 0xa564, 0xf4ec, 0x69a9, 0x0d75, 0x00db, 0x000d},
       {0xe160, 0x6e56, 0xb574, 0xf6ee, 0x79b9, 0x0f77, 0x00fb, 0x000f}},
      {{0x8302'8100, 0x8645'8544, 0x4a8a'4989, 0x1e9d'1c9b},
       {0xa726'a524, 0x0057'9756, 0x0000'5b9b, 0x0000'00bf},
       {0xc342'c140, 0xa665'a564, 0x6aaa'69a9, 0x5edd'5cdb},
       {0xe766'e564, 0x0077'b776, 0x0000'7bbb, 0x0000'00ff}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVremuvv() {
  asm("vremu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVremuvv() {
  asm("vremu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVremuvx() {
  asm("vremu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVremuvx() {
  asm("vremu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVremvv() {
  asm("vrem.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVremvv() {
  asm("vrem.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVremvx() {
  asm("vrem.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVremvx() {
  asm("vrem.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVrem) {
  TestVectorInstruction(
      ExecVremuvv,
      ExecMaskedVremuvv,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {10, 13, 10, 13, 0, 0, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0},
       {22, 64, 22, 64, 22, 64, 22, 64, 0, 0, 0, 0, 0, 128, 22, 64},
       {74, 72, 70, 68, 65, 64, 62, 60, 17, 17, 17, 17, 17, 17, 17, 17},
       {116, 125, 112, 121, 107, 117, 104, 113, 101, 109, 98, 105, 94, 101, 90, 97},
       {84, 93, 80, 89, 79, 85, 76, 81, 68, 77, 65, 73, 61, 69, 57, 65},
       {187, 187, 187, 169, 187, 187, 187, 169, 187, 187, 187, 169, 187, 187, 187, 169},
       {169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0d81, 0x0d81, 0x0000, 0x0000, 0x0000, 0x0d81, 0x0000, 0x0000},
       {0x4016, 0x4016, 0x4016, 0x4016, 0x0000, 0x0000, 0x8000, 0x4016},
       {0x484a, 0x4446, 0x4041, 0x3c3e, 0x1111, 0x1111, 0x1111, 0x1111},
       {0x7d74, 0x7970, 0x756b, 0x7168, 0x6d65, 0x6962, 0x655e, 0x615a},
       {0x5d54, 0x5950, 0x554f, 0x514c, 0x4d44, 0x4941, 0x453d, 0x4139},
       {0xbbbb, 0xa9bb, 0xbbbb, 0xa9bb, 0xbbbb, 0xa9bb, 0xbbbb, 0xa9bb},
       {0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0d81'0d81, 0x0000'0000, 0x0d80'cccd, 0x0000'0000},
       {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
       {0x4446'484a, 0x3c3e'4041, 0x1111'1111, 0x1111'1111},
       {0x7970'7d74, 0x7168'756b, 0x6962'6d65, 0x615a'655e},
       {0x5950'5d54, 0x514c'554f, 0x4941'4d44, 0x4139'453d},
       {0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb},
       {0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x1111'1110'51c5'51c6, 0x1111'1110'51c5'1112},
       {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
       {0x3c3e'4041'4446'484a, 0x1111'1111'1111'1111},
       {0x7168'756b'7970'7d74, 0x615a'655e'6962'6d65},
       {0x514c'554f'5950'5d54, 0x4139'453d'4941'4d44},
       {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
       {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},
      kVectorComparisonSource);

  TestVectorInstruction(
      ExecVremuvx,
      ExecMaskedVremuvx,
      {{5, 70, 5, 70, 5, 70, 5, 70, 64, 64, 64, 64, 85, 85, 85, 85},
       {10, 64, 10, 64, 85, 85, 85, 85, 0, 0, 10, 64, 85, 85, 85, 85},
       {22, 64, 22, 64, 22, 64, 22, 64, 0, 0, 0, 0, 0, 128, 22, 64},
       {0, 0, 0, 0, 0, 0, 0, 0, 17, 17, 17, 17, 17, 17, 17, 17},
       {74, 85, 74, 85, 74, 85, 74, 85, 76, 85, 76, 85, 76, 85, 76, 85},
       {74, 85, 74, 85, 78, 85, 78, 85, 75, 85, 75, 85, 75, 85, 75, 85},
       {17, 17, 17, 169, 17, 17, 17, 169, 17, 17, 17, 169, 17, 17, 17, 169},
       {169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169}},
      {{0x455b, 0x455b, 0x455b, 0x455b, 0x4040, 0x4040, 0x5555, 0x5555},
       {0x40b4, 0x40b4, 0x5555, 0x5555, 0x0000, 0x40b4, 0x5555, 0x5555},
       {0x4016, 0x4016, 0x4016, 0x4016, 0x0000, 0x0000, 0x8000, 0x4016},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x1111, 0x1111, 0x1111, 0x1111},
       {0x554a, 0x554a, 0x554a, 0x554a, 0x554c, 0x554c, 0x554c, 0x554c},
       {0x554a, 0x554a, 0x554e, 0x554e, 0x554b, 0x554b, 0x554b, 0x554b},
       {0x1111, 0xa9bb, 0x1111, 0xa9bb, 0x1111, 0xa9bb, 0x1111, 0xa9bb},
       {0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9}},
      {{0x455b'455b, 0x455b'455b, 0x4040'4040, 0x5555'5555},
       {0x40b4'40b4, 0x5555'5555, 0x40b4'0000, 0x5555'5555},
       {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
       {0x0000'0000, 0x0000'0000, 0x1111'1111, 0x1111'1111},
       {0x554a'554a, 0x554a'554a, 0x554c'554c, 0x554c'554c},
       {0x554a'554a, 0x554e'554e, 0x554b'554b, 0x554b'554b},
       {0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb},
       {0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9}},
      {{0x455b'455b'455b'455b, 0x5555'5554'9595'9596},
       {0x5555'5554'9609'960a, 0x5555'5554'9609'5556},
       {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
       {0x0000'0000'0000'0000, 0x1111'1111'1111'1111},
       {0x554a'554a'554a'554a, 0x554c'554c'554c'554c},
       {0x554e'554e'554a'554a, 0x554b'554b'554b'554b},
       {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
       {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},
      kVectorComparisonSource);

  TestVectorInstruction(
      ExecVremvv,
      ExecMaskedVremvv,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {248, 13, 248, 13, 255, 255, 255, 255, 0, 0, 248, 13, 255, 255, 255, 255},
       {7, 0, 7, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0},
       {170, 170, 170, 170, 170, 170, 170, 170, 17, 17, 17, 17, 17, 17, 17, 17},
       {244, 255, 244, 255, 244, 255, 244, 255, 246, 255, 246, 255, 246, 255, 246, 255},
       {244, 255, 244, 255, 248, 255, 248, 255, 245, 255, 245, 255, 245, 255, 245, 255},
       {251, 249, 247, 227, 242, 241, 239, 219, 234, 233, 231, 253, 227, 225, 223, 237},
       {233, 229, 253, 247, 238, 235, 249, 241, 244, 253, 253, 249, 249, 253, 253, 255}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0d81, 0x0d81, 0xffff, 0xffff, 0x0000, 0x0d81, 0xffff, 0xffff},
       {0x000b, 0x000b, 0x000b, 0x000b, 0x0000, 0x0000, 0xfff8, 0x000b},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0x1111, 0x1111, 0x1111, 0x1111},
       {0xfff4, 0xfff4, 0xfff4, 0xfff4, 0xfff6, 0xfff6, 0xfff6, 0xfff6},
       {0xfff4, 0xfff4, 0xfff8, 0xfff8, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xf8fb, 0xe2f7, 0xf0f2, 0xdaef, 0xe8ea, 0xfc13, 0xe0e3, 0xec03},
       {0xe3e9, 0xf4fd, 0xfe05, 0xff0d, 0xf803, 0xfb15, 0xff31, 0xfffd}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0d81'0d81, 0xffff'ffff, 0x0d80'cccd, 0xffff'ffff},
       {0x000b'fb79, 0x000b'fb79, 0x0000'0000, 0x000c'3b63},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0x1111'1111, 0x1111'1111},
       {0xfff4'fff4, 0xfff4'fff4, 0xfff6'fff6, 0xfff6'fff6},
       {0xfff4'fff4, 0xfff8'fff8, 0xfff5'fff5, 0xfff5'fff5},
       {0xe2f6'f8fb, 0xdaee'f0f2, 0xfc12'1619, 0xec02'060b},
       {0xf4fb'0109, 0xff09'131c, 0xfb0d'1f30, 0xffaa'5551}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0xffff'ffff'40b4'40b4, 0xffff'ffff'40b4'0000},
       {0x000c'000c'000b'fb79, 0x000c'3ff5'bff5'bb63},
       {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
       {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
       {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
       {0xdaee'f0f1'e2f6'f8fb, 0xec02'0609'fc12'1619},
       {0xff09'1318'2731'3b49, 0xffaa'54ff'aa54'ffa4}},
      kVectorComparisonSource);

  TestVectorInstruction(
      ExecVremvx,
      ExecMaskedVremvx,
      {{5, 240, 5, 240, 5, 240, 5, 240, 64, 64, 64, 64, 255, 255, 255, 255},
       {180, 64, 180, 64, 255, 255, 255, 255, 0, 0, 180, 64, 255, 255, 255, 255},
       {22, 64, 22, 64, 22, 64, 22, 64, 0, 0, 0, 0, 0, 214, 22, 64},
       {0, 0, 0, 0, 0, 0, 0, 0, 17, 17, 17, 17, 17, 17, 17, 17},
       {244, 255, 244, 255, 244, 255, 244, 255, 246, 255, 246, 255, 246, 255, 246, 255},
       {244, 255, 244, 255, 248, 255, 248, 255, 245, 255, 245, 255, 245, 255, 245, 255},
       {187, 187, 187, 255, 187, 187, 187, 255, 187, 187, 187, 255, 187, 187, 187, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},
      {{0xf005, 0xf005, 0xf005, 0xf005, 0x4040, 0x4040, 0xffff, 0xffff},
       {0x40b4, 0x40b4, 0xffff, 0xffff, 0x0000, 0x40b4, 0xffff, 0xffff},
       {0x4016, 0x4016, 0x4016, 0x4016, 0x0000, 0x0000, 0xd556, 0x4016},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x1111, 0x1111, 0x1111, 0x1111},
       {0xfff4, 0xfff4, 0xfff4, 0xfff4, 0xfff6, 0xfff6, 0xfff6, 0xfff6},
       {0xfff4, 0xfff4, 0xfff8, 0xfff8, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xbbbb, 0xff11, 0xbbbb, 0xff11, 0xbbbb, 0xff11, 0xbbbb, 0xff11},
       {0xfeff, 0xfeff, 0xfeff, 0xfeff, 0xfeff, 0xfeff, 0xfeff, 0xfeff}},
      {{0xf005'f005, 0xf005'f005, 0x4040'4040, 0xffff'ffff},
       {0x40b4'40b4, 0xffff'ffff, 0x40b4'0000, 0xffff'ffff},
       {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
       {0x0000'0000, 0x0000'0000, 0x1111'1111, 0x1111'1111},
       {0xfff4'fff4, 0xfff4'fff4, 0xfff6'fff6, 0xfff6'fff6},
       {0xfff4'fff4, 0xfff8'fff8, 0xfff5'fff5, 0xfff5'fff5},
       {0xff11'1111, 0xff11'1111, 0xff11'1111, 0xff11'1111},
       {0xfefe'feff, 0xfefe'feff, 0xfefe'feff, 0xfefe'feff}},
      {{0xf005'f005'f005'f005, 0xffff'ffff'4040'4040},
       {0xffff'ffff'40b4'40b4, 0xffff'ffff'40b4'0000},
       {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
       {0x0000'0000'0000'0000, 0x1111'1111'1111'1111},
       {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
       {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
       {0xff11'1110'ff11'1111, 0xff11'1110'ff11'1111},
       {0xfefe'fefe'fefe'feff, 0xfefe'fefe'fefe'feff}},
      kVectorComparisonSource);
}

[[gnu::naked]] void ExecVslide1upvx() {
  asm("vslide1up.vx  v8, v24, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVslide1upvx() {
  asm("vslide1up.vx  v8, v24, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVslide1up) {
  TestVectorInstruction(
      ExecVslide1upvx,
      ExecMaskedVslide1upvx,
      {{0xaa, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {30, 32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60},
       {62, 64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92},
       {94, 96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124},
       {126, 128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156},
       {158, 160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188},
       {190, 192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220},
       {222, 224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252}},
      {{0xaaaa, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x1e1c, 0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38},
       {0x3e3c, 0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58},
       {0x5e5c, 0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78},
       {0x7e7c, 0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98},
       {0x9e9c, 0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8},
       {0xbebc, 0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8},
       {0xdedc, 0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8}},
      {{0xaaaa'aaaa, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x1e1c'1a18, 0x2624'2220, 0x2e2c'2a29, 0x3634'3231},
       {0x3e3c'3a38, 0x4644'4240, 0x4e4c'4a49, 0x5654'5251},
       {0x5e5c'5a58, 0x6664'6260, 0x6e6c'6a69, 0x7674'7271},
       {0x7e7c'7a78, 0x8684'8280, 0x8e8c'8a89, 0x9694'9291},
       {0x9e9c'9a98, 0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1},
       {0xbebc'bab8, 0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1},
       {0xdedc'dad8, 0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0x0e0c'0a09'0604'0200},
       {0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x8e8c'8a89'8684'8280},
       {0x9e9c'9a98'9694'9291, 0xaeac'aaa9'a6a4'a2a0},
       {0xbebc'bab8'b6b4'b2b1, 0xcecc'cac9'c6c4'c2c0},
       {0xdedc'dad8'd6d4'd2d1, 0xeeec'eae9'e6e4'e2e0}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVsllvv() {
  asm("vsll.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsllvv() {
  asm("vsll.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsllvx() {
  asm("vsll.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsllvx() {
  asm("vsll.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsllvi() {
  asm("vsll.vi  v8, v16, 0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsllvi() {
  asm("vsll.vi  v8, v16, 0xb, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVsll) {
  TestVectorInstruction(
      ExecVsllvv,
      ExecMaskedVsllvv,
      {{0, 4, 32, 192, 8, 20, 96, 192, 16, 36, 160, 192, 12, 52, 224, 192},
       {16, 68, 32, 192, 40, 84, 96, 192, 48, 100, 160, 192, 28, 116, 224, 192},
       {32, 132, 32, 192, 72, 148, 96, 192, 80, 164, 160, 192, 44, 180, 224, 192},
       {48, 196, 32, 192, 104, 212, 96, 192, 112, 228, 160, 192, 60, 244, 224, 192},
       {64, 4, 32, 192, 136, 20, 96, 192, 144, 36, 160, 192, 76, 52, 224, 192},
       {80, 68, 32, 192, 168, 84, 96, 192, 176, 100, 160, 192, 92, 116, 224, 192},
       {96, 132, 32, 192, 200, 148, 96, 192, 208, 164, 160, 192, 108, 180, 224, 192},
       {112, 196, 32, 192, 232, 212, 96, 192, 240, 228, 160, 192, 124, 244, 224, 192}},
      {{0x8100, 0x3020, 0x0800, 0x6000, 0x1210, 0xb0a0, 0x0c00, 0xe000},
       {0x9110, 0x3120, 0x2800, 0x6000, 0x3230, 0xb1a0, 0x1c00, 0xe000},
       {0xa120, 0x3220, 0x4800, 0x6000, 0x5250, 0xb2a0, 0x2c00, 0xe000},
       {0xb130, 0x3320, 0x6800, 0x6000, 0x7270, 0xb3a0, 0x3c00, 0xe000},
       {0xc140, 0x3420, 0x8800, 0x6000, 0x9290, 0xb4a0, 0x4c00, 0xe000},
       {0xd150, 0x3520, 0xa800, 0x6000, 0xb2b0, 0xb5a0, 0x5c00, 0xe000},
       {0xe160, 0x3620, 0xc800, 0x6000, 0xd2d0, 0xb6a0, 0x6c00, 0xe000},
       {0xf170, 0x3720, 0xe800, 0x6000, 0xf2f0, 0xb7a0, 0x7c00, 0xe000}},
      {{0x8302'8100, 0x0d0a'0800, 0x1210'0000, 0x0c00'0000},
       {0x9312'9110, 0x2d2a'2800, 0x3230'0000, 0x1c00'0000},
       {0xa322'a120, 0x4d4a'4800, 0x5250'0000, 0x2c00'0000},
       {0xb332'b130, 0x6d6a'6800, 0x7270'0000, 0x3c00'0000},
       {0xc342'c140, 0x8d8a'8800, 0x9290'0000, 0x4c00'0000},
       {0xd352'd150, 0xadaa'a800, 0xb2b0'0000, 0x5c00'0000},
       {0xe362'e160, 0xcdca'c800, 0xd2d0'0000, 0x6c00'0000},
       {0xf372'f170, 0xedea'e800, 0xf2f0'0000, 0x7c00'0000}},
      {{0x8706'8504'8302'8100, 0x1a19'1615'1210'0000},
       {0x9312'9110'0000'0000, 0x3230'0000'0000'0000},
       {0xa726'a524'a322'a120, 0x5a59'5655'5250'0000},
       {0xb332'b130'0000'0000, 0x7270'0000'0000'0000},
       {0xc746'c544'c342'c140, 0x9a99'9695'9290'0000},
       {0xd352'd150'0000'0000, 0xb2b0'0000'0000'0000},
       {0xe766'e564'e362'e160, 0xdad9'd6d5'd2d0'0000},
       {0xf372'f170'0000'0000, 0xf2f0'0000'0000'0000}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVsllvx,
      ExecMaskedVsllvx,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188},
       {192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252},
       {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188},
       {192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252}},
      {{0x0000, 0x0800, 0x1000, 0x1800, 0x2000, 0x2800, 0x3000, 0x3800},
       {0x4000, 0x4800, 0x5000, 0x5800, 0x6000, 0x6800, 0x7000, 0x7800},
       {0x8000, 0x8800, 0x9000, 0x9800, 0xa000, 0xa800, 0xb000, 0xb800},
       {0xc000, 0xc800, 0xd000, 0xd800, 0xe000, 0xe800, 0xf000, 0xf800},
       {0x0000, 0x0800, 0x1000, 0x1800, 0x2000, 0x2800, 0x3000, 0x3800},
       {0x4000, 0x4800, 0x5000, 0x5800, 0x6000, 0x6800, 0x7000, 0x7800},
       {0x8000, 0x8800, 0x9000, 0x9800, 0xa000, 0xa800, 0xb000, 0xb800},
       {0xc000, 0xc800, 0xd000, 0xd800, 0xe000, 0xe800, 0xf000, 0xf800}},
      {{0x0a04'0000, 0x1a14'1000, 0x2a24'2000, 0x3a34'3000},
       {0x4a44'4000, 0x5a54'5000, 0x6a64'6000, 0x7a74'7000},
       {0x8a84'8000, 0x9a94'9000, 0xaaa4'a000, 0xbab4'b000},
       {0xcac4'c000, 0xdad4'd000, 0xeae4'e000, 0xfaf4'f000},
       {0x0b05'0000, 0x1b15'1000, 0x2b25'2000, 0x3b35'3000},
       {0x4b45'4000, 0x5b55'5000, 0x6b65'6000, 0x7b75'7000},
       {0x8b85'8000, 0x9b95'9000, 0xaba5'a000, 0xbbb5'b000},
       {0xcbc5'c000, 0xdbd5'd000, 0xebe5'e000, 0xfbf5'f000}},
      {{0x0a04'0000'0000'0000, 0x2a24'2000'0000'0000},
       {0x4a44'4000'0000'0000, 0x6a64'6000'0000'0000},
       {0x8a84'8000'0000'0000, 0xaaa4'a000'0000'0000},
       {0xcac4'c000'0000'0000, 0xeae4'e000'0000'0000},
       {0x0b05'0000'0000'0000, 0x2b25'2000'0000'0000},
       {0x4b45'4000'0000'0000, 0x6b65'6000'0000'0000},
       {0x8b85'8000'0000'0000, 0xaba5'a000'0000'0000},
       {0xcbc5'c000'0000'0000, 0xebe5'e000'0000'0000}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVsllvi,
      ExecMaskedVsllvi,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248}},
      {{0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000}},
      {{0x1408'0000, 0x3428'2000, 0x5448'4000, 0x7468'6000},
       {0x9488'8000, 0xb4a8'a000, 0xd4c8'c000, 0xf4e8'e000},
       {0x1509'0000, 0x3529'2000, 0x5549'4000, 0x7569'6000},
       {0x9589'8000, 0xb5a9'a000, 0xd5c9'c000, 0xf5e9'e000},
       {0x160a'0000, 0x362a'2000, 0x564a'4000, 0x766a'6000},
       {0x968a'8000, 0xb6aa'a000, 0xd6ca'c000, 0xf6ea'e000},
       {0x170b'0000, 0x372b'2000, 0x574b'4000, 0x776b'6000},
       {0x978b'8000, 0xb7ab'a000, 0xd7cb'c000, 0xf7eb'e000}},
      {{0x3428'2418'1408'0000, 0x7468'6458'5448'4000},
       {0xb4a8'a498'9488'8000, 0xf4e8'e4d8'd4c8'c000},
       {0x3529'2519'1509'0000, 0x7569'6559'5549'4000},
       {0xb5a9'a599'9589'8000, 0xf5e9'e5d9'd5c9'c000},
       {0x362a'261a'160a'0000, 0x766a'665a'564a'4000},
       {0xb6aa'a69a'968a'8000, 0xf6ea'e6da'd6ca'c000},
       {0x372b'271b'170b'0000, 0x776b'675b'574b'4000},
       {0xb7ab'a79b'978b'8000, 0xf7eb'e7db'd7cb'c000}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVsravv() {
  asm("vsra.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsravv() {
  asm("vsra.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsravx() {
  asm("vsra.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsravx() {
  asm("vsra.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsravi() {
  asm("vsra.vi  v8, v16, 0x3\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsravi() {
  asm("vsra.vi  v8, v16, 0x3, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVsra) {
  TestVectorInstruction(ExecVsravv,
                        ExecMaskedVsravv,
                        {{0, 224, 0, 254, 2, 225, 0, 254, 4, 226, 0, 254, 12, 227, 0, 254},
                         {16, 228, 1, 254, 10, 229, 1, 254, 12, 230, 1, 254, 28, 231, 1, 254},
                         {32, 232, 2, 254, 18, 233, 2, 254, 20, 234, 2, 254, 44, 235, 2, 254},
                         {48, 236, 3, 254, 26, 237, 3, 254, 28, 238, 3, 254, 60, 239, 3, 254},
                         {64, 240, 4, 255, 34, 241, 4, 255, 36, 242, 4, 255, 76, 243, 4, 255},
                         {80, 244, 5, 255, 42, 245, 5, 255, 44, 246, 5, 255, 92, 247, 5, 255},
                         {96, 248, 6, 255, 50, 249, 6, 255, 52, 250, 6, 255, 108, 251, 6, 255},
                         {112, 252, 7, 255, 58, 253, 7, 255, 60, 254, 7, 255, 124, 255, 7, 255}},
                        {{0x8100, 0xf830, 0xffc2, 0xfff8, 0xc484, 0xf8b0, 0xff8d, 0xfff8},
                         {0x9110, 0xf931, 0xffca, 0xfff9, 0xcc8c, 0xf9b1, 0xff9d, 0xfff9},
                         {0xa120, 0xfa32, 0xffd2, 0xfffa, 0xd494, 0xfab2, 0xffad, 0xfffa},
                         {0xb130, 0xfb33, 0xffda, 0xfffb, 0xdc9c, 0xfbb3, 0xffbd, 0xfffb},
                         {0xc140, 0xfc34, 0xffe2, 0xfffc, 0xe4a4, 0xfcb4, 0xffcd, 0xfffc},
                         {0xd150, 0xfd35, 0xffea, 0xfffd, 0xecac, 0xfdb5, 0xffdd, 0xfffd},
                         {0xe160, 0xfe36, 0xfff2, 0xfffe, 0xf4b4, 0xfeb6, 0xffed, 0xfffe},
                         {0xf170, 0xff37, 0xfffa, 0xffff, 0xfcbc, 0xffb7, 0xfffd, 0xffff}},
                        {{0x8302'8100, 0xffc3'8342, 0xffff'c585, 0xffff'ff8f},
                         {0x9312'9110, 0xffcb'8b4a, 0xffff'cd8d, 0xffff'ff9f},
                         {0xa322'a120, 0xffd3'9352, 0xffff'd595, 0xffff'ffaf},
                         {0xb332'b130, 0xffdb'9b5a, 0xffff'dd9d, 0xffff'ffbf},
                         {0xc342'c140, 0xffe3'a362, 0xffff'e5a5, 0xffff'ffcf},
                         {0xd352'd150, 0xffeb'ab6a, 0xffff'edad, 0xffff'ffdf},
                         {0xe362'e160, 0xfff3'b372, 0xffff'f5b5, 0xffff'ffef},
                         {0xf372'f170, 0xfffb'bb7a, 0xffff'fdbd, 0xffff'ffff}},
                        {{0x8706'8504'8302'8100, 0xffff'c787'4686'4585},
                         {0xffff'ffff'9716'9514, 0xffff'ffff'ffff'cf8f},
                         {0xa726'a524'a322'a120, 0xffff'd797'5696'5595},
                         {0xffff'ffff'b736'b534, 0xffff'ffff'ffff'df9f},
                         {0xc746'c544'c342'c140, 0xffff'e7a7'66a6'65a5},
                         {0xffff'ffff'd756'd554, 0xffff'ffff'ffff'efaf},
                         {0xe766'e564'e362'e160, 0xffff'f7b7'76b6'75b5},
                         {0xffff'ffff'f776'f574, 0xffff'ffff'ffff'ffbf}},
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVsravx,
                        ExecMaskedVsravx,
                        {{0, 224, 0, 224, 1, 225, 1, 225, 2, 226, 2, 226, 3, 227, 3, 227},
                         {4, 228, 4, 228, 5, 229, 5, 229, 6, 230, 6, 230, 7, 231, 7, 231},
                         {8, 232, 8, 232, 9, 233, 9, 233, 10, 234, 10, 234, 11, 235, 11, 235},
                         {12, 236, 12, 236, 13, 237, 13, 237, 14, 238, 14, 238, 15, 239, 15, 239},
                         {16, 240, 16, 240, 17, 241, 17, 241, 18, 242, 18, 242, 19, 243, 19, 243},
                         {20, 244, 20, 244, 21, 245, 21, 245, 22, 246, 22, 246, 23, 247, 23, 247},
                         {24, 248, 24, 248, 25, 249, 25, 249, 26, 250, 26, 250, 27, 251, 27, 251},
                         {28, 252, 28, 252, 29, 253, 29, 253, 30, 254, 30, 254, 31, 255, 31, 255}},
                        {{0xffe0, 0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3},
                         {0xffe4, 0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7},
                         {0xffe8, 0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb},
                         {0xffec, 0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef},
                         {0xfff0, 0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3},
                         {0xfff4, 0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7},
                         {0xfff8, 0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb},
                         {0xfffc, 0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffff'ffff'ffe1'c1a1, 0xffff'ffff'ffe3'c3a3},
                         {0xffff'ffff'ffe5'c5a5, 0xffff'ffff'ffe7'c7a7},
                         {0xffff'ffff'ffe9'c9a9, 0xffff'ffff'ffeb'cbab},
                         {0xffff'ffff'ffed'cdad, 0xffff'ffff'ffef'cfaf},
                         {0xffff'ffff'fff1'd1b1, 0xffff'ffff'fff3'd3b3},
                         {0xffff'ffff'fff5'd5b5, 0xffff'ffff'fff7'd7b7},
                         {0xffff'ffff'fff9'd9b9, 0xffff'ffff'fffb'dbbb},
                         {0xffff'ffff'fffd'ddbd, 0xffff'ffff'ffff'dfbf}},
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVsravi,
                        ExecMaskedVsravi,
                        {{0, 240, 0, 240, 0, 240, 0, 240, 1, 241, 1, 241, 1, 241, 1, 241},
                         {2, 242, 2, 242, 2, 242, 2, 242, 3, 243, 3, 243, 3, 243, 3, 243},
                         {4, 244, 4, 244, 4, 244, 4, 244, 5, 245, 5, 245, 5, 245, 5, 245},
                         {6, 246, 6, 246, 6, 246, 6, 246, 7, 247, 7, 247, 7, 247, 7, 247},
                         {8, 248, 8, 248, 8, 248, 8, 248, 9, 249, 9, 249, 9, 249, 9, 249},
                         {10, 250, 10, 250, 10, 250, 10, 250, 11, 251, 11, 251, 11, 251, 11, 251},
                         {12, 252, 12, 252, 12, 252, 12, 252, 13, 253, 13, 253, 13, 253, 13, 253},
                         {14, 254, 14, 254, 14, 254, 14, 254, 15, 255, 15, 255, 15, 255, 15, 255}},
                        {{0xf020, 0xf060, 0xf0a0, 0xf0e0, 0xf121, 0xf161, 0xf1a1, 0xf1e1},
                         {0xf222, 0xf262, 0xf2a2, 0xf2e2, 0xf323, 0xf363, 0xf3a3, 0xf3e3},
                         {0xf424, 0xf464, 0xf4a4, 0xf4e4, 0xf525, 0xf565, 0xf5a5, 0xf5e5},
                         {0xf626, 0xf666, 0xf6a6, 0xf6e6, 0xf727, 0xf767, 0xf7a7, 0xf7e7},
                         {0xf828, 0xf868, 0xf8a8, 0xf8e8, 0xf929, 0xf969, 0xf9a9, 0xf9e9},
                         {0xfa2a, 0xfa6a, 0xfaaa, 0xfaea, 0xfb2b, 0xfb6b, 0xfbab, 0xfbeb},
                         {0xfc2c, 0xfc6c, 0xfcac, 0xfcec, 0xfd2d, 0xfd6d, 0xfdad, 0xfded},
                         {0xfe2e, 0xfe6e, 0xfeae, 0xfeee, 0xff2f, 0xff6f, 0xffaf, 0xffef}},
                        {{0xf060'5020, 0xf0e0'd0a0, 0xf161'5121, 0xf1e1'd1a1},
                         {0xf262'5222, 0xf2e2'd2a2, 0xf363'5323, 0xf3e3'd3a3},
                         {0xf464'5424, 0xf4e4'd4a4, 0xf565'5525, 0xf5e5'd5a5},
                         {0xf666'5626, 0xf6e6'd6a6, 0xf767'5727, 0xf7e7'd7a7},
                         {0xf868'5828, 0xf8e8'd8a8, 0xf969'5929, 0xf9e9'd9a9},
                         {0xfa6a'5a2a, 0xfaea'daaa, 0xfb6b'5b2b, 0xfbeb'dbab},
                         {0xfc6c'5c2c, 0xfcec'dcac, 0xfd6d'5d2d, 0xfded'ddad},
                         {0xfe6e'5e2e, 0xfeee'deae, 0xff6f'5f2f, 0xffef'dfaf}},
                        {{0xf0e0'd0a0'9060'5020, 0xf1e1'd1a1'9161'5121},
                         {0xf2e2'd2a2'9262'5222, 0xf3e3'd3a3'9363'5323},
                         {0xf4e4'd4a4'9464'5424, 0xf5e5'd5a5'9565'5525},
                         {0xf6e6'd6a6'9666'5626, 0xf7e7'd7a7'9767'5727},
                         {0xf8e8'd8a8'9868'5828, 0xf9e9'd9a9'9969'5929},
                         {0xfaea'daaa'9a6a'5a2a, 0xfbeb'dbab'9b6b'5b2b},
                         {0xfcec'dcac'9c6c'5c2c, 0xfded'ddad'9d6d'5d2d},
                         {0xfeee'deae'9e6e'5e2e, 0xffef'dfaf'9f6f'5f2f}},
                        kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVsrlvv() {
  asm("vsrl.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsrlvv() {
  asm("vsrl.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsrlvx() {
  asm("vsrl.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsrlvx() {
  asm("vsrl.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsrlvi() {
  asm("vsrl.vi  v8, v16, 0x3\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsrlvi() {
  asm("vsrl.vi  v8, v16, 0x3, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVsrl) {
  TestVectorInstruction(ExecVsrlvv,
                        ExecMaskedVsrlvv,
                        {{0, 32, 0, 2, 2, 33, 0, 2, 4, 34, 0, 2, 12, 35, 0, 2},
                         {16, 36, 1, 2, 10, 37, 1, 2, 12, 38, 1, 2, 28, 39, 1, 2},
                         {32, 40, 2, 2, 18, 41, 2, 2, 20, 42, 2, 2, 44, 43, 2, 2},
                         {48, 44, 3, 2, 26, 45, 3, 2, 28, 46, 3, 2, 60, 47, 3, 2},
                         {64, 48, 4, 3, 34, 49, 4, 3, 36, 50, 4, 3, 76, 51, 4, 3},
                         {80, 52, 5, 3, 42, 53, 5, 3, 44, 54, 5, 3, 92, 55, 5, 3},
                         {96, 56, 6, 3, 50, 57, 6, 3, 52, 58, 6, 3, 108, 59, 6, 3},
                         {112, 60, 7, 3, 58, 61, 7, 3, 60, 62, 7, 3, 124, 63, 7, 3}},
                        {{0x8100, 0x0830, 0x0042, 0x0008, 0x4484, 0x08b0, 0x008d, 0x0008},
                         {0x9110, 0x0931, 0x004a, 0x0009, 0x4c8c, 0x09b1, 0x009d, 0x0009},
                         {0xa120, 0x0a32, 0x0052, 0x000a, 0x5494, 0x0ab2, 0x00ad, 0x000a},
                         {0xb130, 0x0b33, 0x005a, 0x000b, 0x5c9c, 0x0bb3, 0x00bd, 0x000b},
                         {0xc140, 0x0c34, 0x0062, 0x000c, 0x64a4, 0x0cb4, 0x00cd, 0x000c},
                         {0xd150, 0x0d35, 0x006a, 0x000d, 0x6cac, 0x0db5, 0x00dd, 0x000d},
                         {0xe160, 0x0e36, 0x0072, 0x000e, 0x74b4, 0x0eb6, 0x00ed, 0x000e},
                         {0xf170, 0x0f37, 0x007a, 0x000f, 0x7cbc, 0x0fb7, 0x00fd, 0x000f}},
                        {{0x8302'8100, 0x0043'8342, 0x0000'4585, 0x0000'008f},
                         {0x9312'9110, 0x004b'8b4a, 0x0000'4d8d, 0x0000'009f},
                         {0xa322'a120, 0x0053'9352, 0x0000'5595, 0x0000'00af},
                         {0xb332'b130, 0x005b'9b5a, 0x0000'5d9d, 0x0000'00bf},
                         {0xc342'c140, 0x0063'a362, 0x0000'65a5, 0x0000'00cf},
                         {0xd352'd150, 0x006b'ab6a, 0x0000'6dad, 0x0000'00df},
                         {0xe362'e160, 0x0073'b372, 0x0000'75b5, 0x0000'00ef},
                         {0xf372'f170, 0x007b'bb7a, 0x0000'7dbd, 0x0000'00ff}},
                        {{0x8706'8504'8302'8100, 0x0000'4787'4686'4585},
                         {0x0000'0000'9716'9514, 0x0000'0000'0000'4f8f},
                         {0xa726'a524'a322'a120, 0x0000'5797'5696'5595},
                         {0x0000'0000'b736'b534, 0x0000'0000'0000'5f9f},
                         {0xc746'c544'c342'c140, 0x0000'67a7'66a6'65a5},
                         {0x0000'0000'd756'd554, 0x0000'0000'0000'6faf},
                         {0xe766'e564'e362'e160, 0x0000'77b7'76b6'75b5},
                         {0x0000'0000'f776'f574, 0x0000'0000'0000'7fbf}},
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVsrlvx,
                        ExecMaskedVsrlvx,
                        {{0, 32, 0, 32, 1, 33, 1, 33, 2, 34, 2, 34, 3, 35, 3, 35},
                         {4, 36, 4, 36, 5, 37, 5, 37, 6, 38, 6, 38, 7, 39, 7, 39},
                         {8, 40, 8, 40, 9, 41, 9, 41, 10, 42, 10, 42, 11, 43, 11, 43},
                         {12, 44, 12, 44, 13, 45, 13, 45, 14, 46, 14, 46, 15, 47, 15, 47},
                         {16, 48, 16, 48, 17, 49, 17, 49, 18, 50, 18, 50, 19, 51, 19, 51},
                         {20, 52, 20, 52, 21, 53, 21, 53, 22, 54, 22, 54, 23, 55, 23, 55},
                         {24, 56, 24, 56, 25, 57, 25, 57, 26, 58, 26, 58, 27, 59, 27, 59},
                         {28, 60, 28, 60, 29, 61, 29, 61, 30, 62, 30, 62, 31, 63, 31, 63}},
                        {{0x0020, 0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023},
                         {0x0024, 0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027},
                         {0x0028, 0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b},
                         {0x002c, 0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f},
                         {0x0030, 0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033},
                         {0x0034, 0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037},
                         {0x0038, 0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b},
                         {0x003c, 0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0000'0000'0021'c1a1, 0x0000'0000'0023'c3a3},
                         {0x0000'0000'0025'c5a5, 0x0000'0000'0027'c7a7},
                         {0x0000'0000'0029'c9a9, 0x0000'0000'002b'cbab},
                         {0x0000'0000'002d'cdad, 0x0000'0000'002f'cfaf},
                         {0x0000'0000'0031'd1b1, 0x0000'0000'0033'd3b3},
                         {0x0000'0000'0035'd5b5, 0x0000'0000'0037'd7b7},
                         {0x0000'0000'0039'd9b9, 0x0000'0000'003b'dbbb},
                         {0x0000'0000'003d'ddbd, 0x0000'0000'003f'dfbf}},
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVsrlvi,
                        ExecMaskedVsrlvi,
                        {{0, 16, 0, 16, 0, 16, 0, 16, 1, 17, 1, 17, 1, 17, 1, 17},
                         {2, 18, 2, 18, 2, 18, 2, 18, 3, 19, 3, 19, 3, 19, 3, 19},
                         {4, 20, 4, 20, 4, 20, 4, 20, 5, 21, 5, 21, 5, 21, 5, 21},
                         {6, 22, 6, 22, 6, 22, 6, 22, 7, 23, 7, 23, 7, 23, 7, 23},
                         {8, 24, 8, 24, 8, 24, 8, 24, 9, 25, 9, 25, 9, 25, 9, 25},
                         {10, 26, 10, 26, 10, 26, 10, 26, 11, 27, 11, 27, 11, 27, 11, 27},
                         {12, 28, 12, 28, 12, 28, 12, 28, 13, 29, 13, 29, 13, 29, 13, 29},
                         {14, 30, 14, 30, 14, 30, 14, 30, 15, 31, 15, 31, 15, 31, 15, 31}},
                        {{0x1020, 0x1060, 0x10a0, 0x10e0, 0x1121, 0x1161, 0x11a1, 0x11e1},
                         {0x1222, 0x1262, 0x12a2, 0x12e2, 0x1323, 0x1363, 0x13a3, 0x13e3},
                         {0x1424, 0x1464, 0x14a4, 0x14e4, 0x1525, 0x1565, 0x15a5, 0x15e5},
                         {0x1626, 0x1666, 0x16a6, 0x16e6, 0x1727, 0x1767, 0x17a7, 0x17e7},
                         {0x1828, 0x1868, 0x18a8, 0x18e8, 0x1929, 0x1969, 0x19a9, 0x19e9},
                         {0x1a2a, 0x1a6a, 0x1aaa, 0x1aea, 0x1b2b, 0x1b6b, 0x1bab, 0x1beb},
                         {0x1c2c, 0x1c6c, 0x1cac, 0x1cec, 0x1d2d, 0x1d6d, 0x1dad, 0x1ded},
                         {0x1e2e, 0x1e6e, 0x1eae, 0x1eee, 0x1f2f, 0x1f6f, 0x1faf, 0x1fef}},
                        {{0x1060'5020, 0x10e0'd0a0, 0x1161'5121, 0x11e1'd1a1},
                         {0x1262'5222, 0x12e2'd2a2, 0x1363'5323, 0x13e3'd3a3},
                         {0x1464'5424, 0x14e4'd4a4, 0x1565'5525, 0x15e5'd5a5},
                         {0x1666'5626, 0x16e6'd6a6, 0x1767'5727, 0x17e7'd7a7},
                         {0x1868'5828, 0x18e8'd8a8, 0x1969'5929, 0x19e9'd9a9},
                         {0x1a6a'5a2a, 0x1aea'daaa, 0x1b6b'5b2b, 0x1beb'dbab},
                         {0x1c6c'5c2c, 0x1cec'dcac, 0x1d6d'5d2d, 0x1ded'ddad},
                         {0x1e6e'5e2e, 0x1eee'deae, 0x1f6f'5f2f, 0x1fef'dfaf}},
                        {{0x10e0'd0a0'9060'5020, 0x11e1'd1a1'9161'5121},
                         {0x12e2'd2a2'9262'5222, 0x13e3'd3a3'9363'5323},
                         {0x14e4'd4a4'9464'5424, 0x15e5'd5a5'9565'5525},
                         {0x16e6'd6a6'9666'5626, 0x17e7'd7a7'9767'5727},
                         {0x18e8'd8a8'9868'5828, 0x19e9'd9a9'9969'5929},
                         {0x1aea'daaa'9a6a'5a2a, 0x1beb'dbab'9b6b'5b2b},
                         {0x1cec'dcac'9c6c'5c2c, 0x1ded'ddad'9d6d'5d2d},
                         {0x1eee'deae'9e6e'5e2e, 0x1fef'dfaf'9f6f'5f2f}},
                        kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVwaddvv() {
  asm("vwadd.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwaddvv() {
  asm("vwadd.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwadd) {
  TestWideningVectorInstruction(ExecVwaddvv,
                                ExecMaskedVwaddvv,
                                {{0x0000, 0xff13, 0x0006, 0xff19, 0x000d, 0xff1f, 0x0012, 0xff25},
                                 {0x0019, 0xff0b, 0x001e, 0xff11, 0x0024, 0xff17, 0x002a, 0xff1d},
                                 {0x0030, 0xff43, 0x0036, 0xff49, 0x003d, 0xff4f, 0x0042, 0xff55},
                                 {0x0049, 0xff3b, 0x004e, 0xff41, 0x0054, 0xff47, 0x005a, 0xff4d},
                                 {0x0060, 0xff73, 0x0066, 0xff79, 0x006d, 0xff7f, 0x0072, 0xff85},
                                 {0x0079, 0xff6b, 0x007e, 0xff71, 0x0084, 0xff77, 0x008a, 0xff7d},
                                 {0x0090, 0xffa3, 0x0096, 0xffa9, 0x009d, 0xffaf, 0x00a2, 0xffb5},
                                 {0x00a9, 0xff9b, 0x00ae, 0xffa1, 0x00b4, 0xffa7, 0x00ba, 0xffad}},
                                {{0xffff'1300, 0xffff'1906, 0xffff'1f0d, 0xffff'2512},
                                 {0xffff'0b19, 0xffff'111e, 0xffff'1724, 0xffff'1d2a},
                                 {0xffff'4330, 0xffff'4936, 0xffff'4f3d, 0xffff'5542},
                                 {0xffff'3b49, 0xffff'414e, 0xffff'4754, 0xffff'4d5a},
                                 {0xffff'7360, 0xffff'7966, 0xffff'7f6d, 0xffff'8572},
                                 {0xffff'6b79, 0xffff'717e, 0xffff'7784, 0xffff'7d8a},
                                 {0xffff'a390, 0xffff'a996, 0xffff'af9d, 0xffff'b5a2},
                                 {0xffff'9ba9, 0xffff'a1ae, 0xffff'a7b4, 0xffff'adba}},
                                {{0xffff'ffff'1907'1300, 0xffff'ffff'2513'1f0d},
                                 {0xffff'ffff'111f'0b19, 0xffff'ffff'1d2b'1724},
                                 {0xffff'ffff'4937'4330, 0xffff'ffff'5543'4f3d},
                                 {0xffff'ffff'414f'3b49, 0xffff'ffff'4d5b'4754},
                                 {0xffff'ffff'7967'7360, 0xffff'ffff'8573'7f6d},
                                 {0xffff'ffff'717f'6b79, 0xffff'ffff'7d8b'7784},
                                 {0xffff'ffff'a997'a390, 0xffff'ffff'b5a3'af9d},
                                 {0xffff'ffff'a1af'9ba9, 0xffff'ffff'adbb'a7b4}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwadduvv() {
  asm("vwaddu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwadduvv() {
  asm("vwaddu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwaddu) {
  TestWideningVectorInstruction(ExecVwadduvv,
                                ExecMaskedVwadduvv,
                                {{0x0000, 0x0113, 0x0006, 0x0119, 0x000d, 0x011f, 0x0012, 0x0125},
                                 {0x0019, 0x010b, 0x001e, 0x0111, 0x0024, 0x0117, 0x002a, 0x011d},
                                 {0x0030, 0x0143, 0x0036, 0x0149, 0x003d, 0x014f, 0x0042, 0x0155},
                                 {0x0049, 0x013b, 0x004e, 0x0141, 0x0054, 0x0147, 0x005a, 0x014d},
                                 {0x0060, 0x0173, 0x0066, 0x0179, 0x006d, 0x017f, 0x0072, 0x0185},
                                 {0x0079, 0x016b, 0x007e, 0x0171, 0x0084, 0x0177, 0x008a, 0x017d},
                                 {0x0090, 0x01a3, 0x0096, 0x01a9, 0x009d, 0x01af, 0x00a2, 0x01b5},
                                 {0x00a9, 0x019b, 0x00ae, 0x01a1, 0x00b4, 0x01a7, 0x00ba, 0x01ad}},
                                {{0x0001'1300, 0x0001'1906, 0x0001'1f0d, 0x0001'2512},
                                 {0x0001'0b19, 0x0001'111e, 0x0001'1724, 0x0001'1d2a},
                                 {0x0001'4330, 0x0001'4936, 0x0001'4f3d, 0x0001'5542},
                                 {0x0001'3b49, 0x0001'414e, 0x0001'4754, 0x0001'4d5a},
                                 {0x0001'7360, 0x0001'7966, 0x0001'7f6d, 0x0001'8572},
                                 {0x0001'6b79, 0x0001'717e, 0x0001'7784, 0x0001'7d8a},
                                 {0x0001'a390, 0x0001'a996, 0x0001'af9d, 0x0001'b5a2},
                                 {0x0001'9ba9, 0x0001'a1ae, 0x0001'a7b4, 0x0001'adba}},
                                {{0x0000'0001'1907'1300, 0x0000'0001'2513'1f0d},
                                 {0x0000'0001'111f'0b19, 0x0000'0001'1d2b'1724},
                                 {0x0000'0001'4937'4330, 0x0000'0001'5543'4f3d},
                                 {0x0000'0001'414f'3b49, 0x0000'0001'4d5b'4754},
                                 {0x0000'0001'7967'7360, 0x0000'0001'8573'7f6d},
                                 {0x0000'0001'717f'6b79, 0x0000'0001'7d8b'7784},
                                 {0x0000'0001'a997'a390, 0x0000'0001'b5a3'af9d},
                                 {0x0000'0001'a1af'9ba9, 0x0000'0001'adbb'a7b4}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwadduvx() {
  asm("vwaddu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwadduvx() {
  asm("vwaddu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwadduvx) {
  TestWideningVectorInstruction(ExecVwadduvx,
                                ExecMaskedVwadduvx,
                                {{0x00aa, 0x012b, 0x00ac, 0x012d, 0x00ae, 0x012f, 0x00b0, 0x0131},
                                 {0x00b2, 0x0133, 0x00b4, 0x0135, 0x00b6, 0x0137, 0x00b8, 0x0139},
                                 {0x00ba, 0x013b, 0x00bc, 0x013d, 0x00be, 0x013f, 0x00c0, 0x0141},
                                 {0x00c2, 0x0143, 0x00c4, 0x0145, 0x00c6, 0x0147, 0x00c8, 0x0149},
                                 {0x00ca, 0x014b, 0x00cc, 0x014d, 0x00ce, 0x014f, 0x00d0, 0x0151},
                                 {0x00d2, 0x0153, 0x00d4, 0x0155, 0x00d6, 0x0157, 0x00d8, 0x0159},
                                 {0x00da, 0x015b, 0x00dc, 0x015d, 0x00de, 0x015f, 0x00e0, 0x0161},
                                 {0x00e2, 0x0163, 0x00e4, 0x0165, 0x00e6, 0x0167, 0x00e8, 0x0169}},
                                {{0x0001'2baa, 0x0001'2dac, 0x0001'2fae, 0x0001'31b0},
                                 {0x0001'33b2, 0x0001'35b4, 0x0001'37b6, 0x0001'39b8},
                                 {0x0001'3bba, 0x0001'3dbc, 0x0001'3fbe, 0x0001'41c0},
                                 {0x0001'43c2, 0x0001'45c4, 0x0001'47c6, 0x0001'49c8},
                                 {0x0001'4bca, 0x0001'4dcc, 0x0001'4fce, 0x0001'51d0},
                                 {0x0001'53d2, 0x0001'55d4, 0x0001'57d6, 0x0001'59d8},
                                 {0x0001'5bda, 0x0001'5ddc, 0x0001'5fde, 0x0001'61e0},
                                 {0x0001'63e2, 0x0001'65e4, 0x0001'67e6, 0x0001'69e8}},
                                {{0x0000'0001'2dad'2baa, 0x0000'0001'31b1'2fae},
                                 {0x0000'0001'35b5'33b2, 0x0000'0001'39b9'37b6},
                                 {0x0000'0001'3dbd'3bba, 0x0000'0001'41c1'3fbe},
                                 {0x0000'0001'45c5'43c2, 0x0000'0001'49c9'47c6},
                                 {0x0000'0001'4dcd'4bca, 0x0000'0001'51d1'4fce},
                                 {0x0000'0001'55d5'53d2, 0x0000'0001'59d9'57d6},
                                 {0x0000'0001'5ddd'5bda, 0x0000'0001'61e1'5fde},
                                 {0x0000'0001'65e5'63e2, 0x0000'0001'69e9'67e6}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwadduwv() {
  asm("vwaddu.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwadduwv() {
  asm("vwaddu.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwadduwv) {
  TestWideningVectorInstruction(ExecVwadduwv,
                                ExecMaskedVwadduwv,
                                {{0x8100, 0x8394, 0x8508, 0x879c, 0x8911, 0x8ba4, 0x8d18, 0x8fac},
                                 {0x9121, 0x9394, 0x9528, 0x979c, 0x9930, 0x9ba4, 0x9d38, 0x9fac},
                                 {0xa140, 0xa3d4, 0xa548, 0xa7dc, 0xa951, 0xabe4, 0xad58, 0xafec},
                                 {0xb161, 0xb3d4, 0xb568, 0xb7dc, 0xb970, 0xbbe4, 0xbd78, 0xbfec},
                                 {0xc180, 0xc414, 0xc588, 0xc81c, 0xc991, 0xcc24, 0xcd98, 0xd02c},
                                 {0xd1a1, 0xd414, 0xd5a8, 0xd81c, 0xd9b0, 0xdc24, 0xddb8, 0xe02c},
                                 {0xe1c0, 0xe454, 0xe5c8, 0xe85c, 0xe9d1, 0xec64, 0xedd8, 0xf06c},
                                 {0xf1e1, 0xf454, 0xf5e8, 0xf85c, 0xf9f0, 0xfc64, 0xfdf8, 0x006c}},
                                {{0x8303'1300, 0x8707'1b08, 0x8b0b'2311, 0x8f0f'2b18},
                                 {0x9313'1321, 0x9717'1b28, 0x9b1b'2330, 0x9f1f'2b38},
                                 {0xa323'5340, 0xa727'5b48, 0xab2b'6351, 0xaf2f'6b58},
                                 {0xb333'5361, 0xb737'5b68, 0xbb3b'6370, 0xbf3f'6b78},
                                 {0xc343'9380, 0xc747'9b88, 0xcb4b'a391, 0xcf4f'ab98},
                                 {0xd353'93a1, 0xd757'9ba8, 0xdb5b'a3b0, 0xdf5f'abb8},
                                 {0xe363'd3c0, 0xe767'dbc8, 0xeb6b'e3d1, 0xef6f'ebd8},
                                 {0xf373'd3e1, 0xf777'dbe8, 0xfb7b'e3f0, 0xff7f'ebf8}},
                                {{0x8706'8505'1907'1300, 0x8f0e'8d0d'2917'2311},
                                 {0x9716'9515'1927'1321, 0x9f1e'9d1d'2937'2330},
                                 {0xa726'a525'5947'5340, 0xaf2e'ad2d'6957'6351},
                                 {0xb736'b535'5967'5361, 0xbf3e'bd3d'6977'6370},
                                 {0xc746'c545'9987'9380, 0xcf4e'cd4d'a997'a391},
                                 {0xd756'd555'99a7'93a1, 0xdf5e'dd5d'a9b7'a3b0},
                                 {0xe766'e565'd9c7'd3c0, 0xef6e'ed6d'e9d7'e3d1},
                                 {0xf776'f575'd9e7'd3e1, 0xff7e'fd7d'e9f7'e3f0}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwadduwx() {
  asm("vwaddu.wx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwadduwx() {
  asm("vwaddu.wx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwadduwx) {
  TestWideningVectorInstruction(ExecVwadduwx,
                                ExecMaskedVwadduwx,
                                {{0x81aa, 0x83ac, 0x85ae, 0x87b0, 0x89b2, 0x8bb4, 0x8db6, 0x8fb8},
                                 {0x91ba, 0x93bc, 0x95be, 0x97c0, 0x99c2, 0x9bc4, 0x9dc6, 0x9fc8},
                                 {0xa1ca, 0xa3cc, 0xa5ce, 0xa7d0, 0xa9d2, 0xabd4, 0xadd6, 0xafd8},
                                 {0xb1da, 0xb3dc, 0xb5de, 0xb7e0, 0xb9e2, 0xbbe4, 0xbde6, 0xbfe8},
                                 {0xc1ea, 0xc3ec, 0xc5ee, 0xc7f0, 0xc9f2, 0xcbf4, 0xcdf6, 0xcff8},
                                 {0xd1fa, 0xd3fc, 0xd5fe, 0xd800, 0xda02, 0xdc04, 0xde06, 0xe008},
                                 {0xe20a, 0xe40c, 0xe60e, 0xe810, 0xea12, 0xec14, 0xee16, 0xf018},
                                 {0xf21a, 0xf41c, 0xf61e, 0xf820, 0xfa22, 0xfc24, 0xfe26, 0x0028}},
                                {{0x8303'2baa, 0x8707'2fae, 0x8b0b'33b2, 0x8f0f'37b6},
                                 {0x9313'3bba, 0x9717'3fbe, 0x9b1b'43c2, 0x9f1f'47c6},
                                 {0xa323'4bca, 0xa727'4fce, 0xab2b'53d2, 0xaf2f'57d6},
                                 {0xb333'5bda, 0xb737'5fde, 0xbb3b'63e2, 0xbf3f'67e6},
                                 {0xc343'6bea, 0xc747'6fee, 0xcb4b'73f2, 0xcf4f'77f6},
                                 {0xd353'7bfa, 0xd757'7ffe, 0xdb5b'8402, 0xdf5f'8806},
                                 {0xe363'8c0a, 0xe767'900e, 0xeb6b'9412, 0xef6f'9816},
                                 {0xf373'9c1a, 0xf777'a01e, 0xfb7b'a422, 0xff7f'a826}},
                                {{0x8706'8505'2dad'2baa, 0x8f0e'8d0d'35b5'33b2},
                                 {0x9716'9515'3dbd'3bba, 0x9f1e'9d1d'45c5'43c2},
                                 {0xa726'a525'4dcd'4bca, 0xaf2e'ad2d'55d5'53d2},
                                 {0xb736'b535'5ddd'5bda, 0xbf3e'bd3d'65e5'63e2},
                                 {0xc746'c545'6ded'6bea, 0xcf4e'cd4d'75f5'73f2},
                                 {0xd756'd555'7dfd'7bfa, 0xdf5e'dd5d'8605'8402},
                                 {0xe766'e565'8e0d'8c0a, 0xef6e'ed6d'9615'9412},
                                 {0xf776'f575'9e1d'9c1a, 0xff7e'fd7d'a625'a422}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwaddvx() {
  asm("vwadd.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwaddvx() {
  asm("vwadd.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwaddvx) {
  TestWideningVectorInstruction(ExecVwaddvx,
                                ExecMaskedVwaddvx,
                                {{0xffaa, 0xff2b, 0xffac, 0xff2d, 0xffae, 0xff2f, 0xffb0, 0xff31},
                                 {0xffb2, 0xff33, 0xffb4, 0xff35, 0xffb6, 0xff37, 0xffb8, 0xff39},
                                 {0xffba, 0xff3b, 0xffbc, 0xff3d, 0xffbe, 0xff3f, 0xffc0, 0xff41},
                                 {0xffc2, 0xff43, 0xffc4, 0xff45, 0xffc6, 0xff47, 0xffc8, 0xff49},
                                 {0xffca, 0xff4b, 0xffcc, 0xff4d, 0xffce, 0xff4f, 0xffd0, 0xff51},
                                 {0xffd2, 0xff53, 0xffd4, 0xff55, 0xffd6, 0xff57, 0xffd8, 0xff59},
                                 {0xffda, 0xff5b, 0xffdc, 0xff5d, 0xffde, 0xff5f, 0xffe0, 0xff61},
                                 {0xffe2, 0xff63, 0xffe4, 0xff65, 0xffe6, 0xff67, 0xffe8, 0xff69}},
                                {{0xffff'2baa, 0xffff'2dac, 0xffff'2fae, 0xffff'31b0},
                                 {0xffff'33b2, 0xffff'35b4, 0xffff'37b6, 0xffff'39b8},
                                 {0xffff'3bba, 0xffff'3dbc, 0xffff'3fbe, 0xffff'41c0},
                                 {0xffff'43c2, 0xffff'45c4, 0xffff'47c6, 0xffff'49c8},
                                 {0xffff'4bca, 0xffff'4dcc, 0xffff'4fce, 0xffff'51d0},
                                 {0xffff'53d2, 0xffff'55d4, 0xffff'57d6, 0xffff'59d8},
                                 {0xffff'5bda, 0xffff'5ddc, 0xffff'5fde, 0xffff'61e0},
                                 {0xffff'63e2, 0xffff'65e4, 0xffff'67e6, 0xffff'69e8}},
                                {{0xffff'ffff'2dad'2baa, 0xffff'ffff'31b1'2fae},
                                 {0xffff'ffff'35b5'33b2, 0xffff'ffff'39b9'37b6},
                                 {0xffff'ffff'3dbd'3bba, 0xffff'ffff'41c1'3fbe},
                                 {0xffff'ffff'45c5'43c2, 0xffff'ffff'49c9'47c6},
                                 {0xffff'ffff'4dcd'4bca, 0xffff'ffff'51d1'4fce},
                                 {0xffff'ffff'55d5'53d2, 0xffff'ffff'59d9'57d6},
                                 {0xffff'ffff'5ddd'5bda, 0xffff'ffff'61e1'5fde},
                                 {0xffff'ffff'65e5'63e2, 0xffff'ffff'69e9'67e6}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwaddwv() {
  asm("vwadd.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwaddwv() {
  asm("vwadd.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwaddwv) {
  TestWideningVectorInstruction(ExecVwaddwv,
                                ExecMaskedVwaddwv,
                                {{0x8100, 0x8294, 0x8508, 0x869c, 0x8911, 0x8aa4, 0x8d18, 0x8eac},
                                 {0x9121, 0x9294, 0x9528, 0x969c, 0x9930, 0x9aa4, 0x9d38, 0x9eac},
                                 {0xa140, 0xa2d4, 0xa548, 0xa6dc, 0xa951, 0xaae4, 0xad58, 0xaeec},
                                 {0xb161, 0xb2d4, 0xb568, 0xb6dc, 0xb970, 0xbae4, 0xbd78, 0xbeec},
                                 {0xc180, 0xc314, 0xc588, 0xc71c, 0xc991, 0xcb24, 0xcd98, 0xcf2c},
                                 {0xd1a1, 0xd314, 0xd5a8, 0xd71c, 0xd9b0, 0xdb24, 0xddb8, 0xdf2c},
                                 {0xe1c0, 0xe354, 0xe5c8, 0xe75c, 0xe9d1, 0xeb64, 0xedd8, 0xef6c},
                                 {0xf1e1, 0xf354, 0xf5e8, 0xf75c, 0xf9f0, 0xfb64, 0xfdf8, 0xff6c}},
                                {{0x8302'1300, 0x8706'1b08, 0x8b0a'2311, 0x8f0e'2b18},
                                 {0x9312'1321, 0x9716'1b28, 0x9b1a'2330, 0x9f1e'2b38},
                                 {0xa322'5340, 0xa726'5b48, 0xab2a'6351, 0xaf2e'6b58},
                                 {0xb332'5361, 0xb736'5b68, 0xbb3a'6370, 0xbf3e'6b78},
                                 {0xc342'9380, 0xc746'9b88, 0xcb4a'a391, 0xcf4e'ab98},
                                 {0xd352'93a1, 0xd756'9ba8, 0xdb5a'a3b0, 0xdf5e'abb8},
                                 {0xe362'd3c0, 0xe766'dbc8, 0xeb6a'e3d1, 0xef6e'ebd8},
                                 {0xf372'd3e1, 0xf776'dbe8, 0xfb7a'e3f0, 0xff7e'ebf8}},
                                {{0x8706'8504'1907'1300, 0x8f0e'8d0c'2917'2311},
                                 {0x9716'9514'1927'1321, 0x9f1e'9d1c'2937'2330},
                                 {0xa726'a524'5947'5340, 0xaf2e'ad2c'6957'6351},
                                 {0xb736'b534'5967'5361, 0xbf3e'bd3c'6977'6370},
                                 {0xc746'c544'9987'9380, 0xcf4e'cd4c'a997'a391},
                                 {0xd756'd554'99a7'93a1, 0xdf5e'dd5c'a9b7'a3b0},
                                 {0xe766'e564'd9c7'd3c0, 0xef6e'ed6c'e9d7'e3d1},
                                 {0xf776'f574'd9e7'd3e1, 0xff7e'fd7c'e9f7'e3f0}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwaddwx() {
  asm("vwadd.wx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwaddwx() {
  asm("vwadd.wx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwaddwx) {
  TestWideningVectorInstruction(ExecVwaddwx,
                                ExecMaskedVwaddwx,
                                {{0x80aa, 0x82ac, 0x84ae, 0x86b0, 0x88b2, 0x8ab4, 0x8cb6, 0x8eb8},
                                 {0x90ba, 0x92bc, 0x94be, 0x96c0, 0x98c2, 0x9ac4, 0x9cc6, 0x9ec8},
                                 {0xa0ca, 0xa2cc, 0xa4ce, 0xa6d0, 0xa8d2, 0xaad4, 0xacd6, 0xaed8},
                                 {0xb0da, 0xb2dc, 0xb4de, 0xb6e0, 0xb8e2, 0xbae4, 0xbce6, 0xbee8},
                                 {0xc0ea, 0xc2ec, 0xc4ee, 0xc6f0, 0xc8f2, 0xcaf4, 0xccf6, 0xcef8},
                                 {0xd0fa, 0xd2fc, 0xd4fe, 0xd700, 0xd902, 0xdb04, 0xdd06, 0xdf08},
                                 {0xe10a, 0xe30c, 0xe50e, 0xe710, 0xe912, 0xeb14, 0xed16, 0xef18},
                                 {0xf11a, 0xf31c, 0xf51e, 0xf720, 0xf922, 0xfb24, 0xfd26, 0xff28}},
                                {{0x8302'2baa, 0x8706'2fae, 0x8b0a'33b2, 0x8f0e'37b6},
                                 {0x9312'3bba, 0x9716'3fbe, 0x9b1a'43c2, 0x9f1e'47c6},
                                 {0xa322'4bca, 0xa726'4fce, 0xab2a'53d2, 0xaf2e'57d6},
                                 {0xb332'5bda, 0xb736'5fde, 0xbb3a'63e2, 0xbf3e'67e6},
                                 {0xc342'6bea, 0xc746'6fee, 0xcb4a'73f2, 0xcf4e'77f6},
                                 {0xd352'7bfa, 0xd756'7ffe, 0xdb5a'8402, 0xdf5e'8806},
                                 {0xe362'8c0a, 0xe766'900e, 0xeb6a'9412, 0xef6e'9816},
                                 {0xf372'9c1a, 0xf776'a01e, 0xfb7a'a422, 0xff7e'a826}},
                                {{0x8706'8504'2dad'2baa, 0x8f0e'8d0c'35b5'33b2},
                                 {0x9716'9514'3dbd'3bba, 0x9f1e'9d1c'45c5'43c2},
                                 {0xa726'a524'4dcd'4bca, 0xaf2e'ad2c'55d5'53d2},
                                 {0xb736'b534'5ddd'5bda, 0xbf3e'bd3c'65e5'63e2},
                                 {0xc746'c544'6ded'6bea, 0xcf4e'cd4c'75f5'73f2},
                                 {0xd756'd554'7dfd'7bfa, 0xdf5e'dd5c'8605'8402},
                                 {0xe766'e564'8e0d'8c0a, 0xef6e'ed6c'9615'9412},
                                 {0xf776'f574'9e1d'9c1a, 0xff7e'fd7c'a625'a422}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwmulvv() {
  asm("vwmul.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmulvv() {
  asm("vwmul.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwmul) {
  TestWideningVectorInstruction(ExecVwmulvv,
                                ExecMaskedVwmulvv,
                                {{0x0000, 0x3692, 0x0008, 0x33c2, 0x0024, 0x3102, 0x0048, 0x2e52},
                                 {0x0088, 0x3a92, 0x00c8, 0x37c2, 0x0120, 0x3502, 0x0188, 0x3252},
                                 {0x0200, 0x21d2, 0x0288, 0x1f82, 0x0334, 0x1d42, 0x03c8, 0x1b12},
                                 {0x0498, 0x25d2, 0x0548, 0x2382, 0x0620, 0x2142, 0x0708, 0x1f12},
                                 {0x0800, 0x1112, 0x0908, 0x0f42, 0x0a44, 0x0d82, 0x0b48, 0x0bd2},
                                 {0x0ca8, 0x1512, 0x0dc8, 0x1342, 0x0f20, 0x1182, 0x1088, 0x0fd2},
                                 {0x1200, 0x0452, 0x1388, 0x0302, 0x1554, 0x01c2, 0x16c8, 0x0092},
                                 {0x18b8, 0x0852, 0x1a48, 0x0702, 0x1c20, 0x05c2, 0x1e08, 0x0492}},
                                {{0x3692'0000, 0x33bf'3808, 0x30fc'1524, 0x2e4a'0848},
                                 {0x3a86'2988, 0x37b4'18c8, 0x34f1'b120, 0x323f'6988},
                                 {0x21bf'4200, 0x1f6d'7a88, 0x1d2b'6834, 0x1afa'4bc8},
                                 {0x25b5'7d98, 0x2364'5d48, 0x2122'f620, 0x1ef1'af08},
                                 {0x10f4'8800, 0x0f23'c108, 0x0d62'bf44, 0x0bb2'9348},
                                 {0x14ec'd5a8, 0x131c'a5c8, 0x115c'3f20, 0x0fab'f888},
                                 {0x0431'd200, 0x02e2'0b88, 0x01a2'1a54, 0x0072'dec8},
                                 {0x082c'31b8, 0x06dc'f248, 0x059d'8c20, 0x046e'4608}},
                                {{0x33be'bb57'7192'0000, 0x2e49'8c98'5f6f'1524},
                                 {0x37b3'9c18'79e9'2988, 0x323e'eddb'56b5'b120},
                                 {0x1f6d'04e3'116f'4200, 0x1af9'd928'125e'6834},
                                 {0x2363'e7a8'2dd8'7d98, 0x1ef1'3c6e'fd96'f620},
                                 {0x0f23'5a7e'bd54'8800, 0x0bb2'31c7'd155'bf44},
                                 {0x131c'3f47'edcf'd5a8, 0x0fab'9712'b080'3f20},
                                 {0x02e1'bc2a'7541'd200, 0x0072'9677'9c55'1a54},
                                 {0x06dc'a2f7'b9cf'31b8, 0x046d'fdc6'6f71'8c20}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwmulsuvv() {
  asm("vwmulsu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmulsuvv() {
  asm("vwmulsu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwmulsu) {
  TestWideningVectorInstruction(ExecVwmulsuvv,
                                ExecMaskedVwmulsuvv,
                                {{0x0000, 0xb792, 0x0008, 0xb6c2, 0x0024, 0xb602, 0x0048, 0xb552},
                                 {0x0088, 0xc392, 0x00c8, 0xc2c2, 0x0120, 0xc202, 0x0188, 0xc152},
                                 {0x0200, 0xb2d2, 0x0288, 0xb282, 0x0334, 0xb242, 0x03c8, 0xb212},
                                 {0x0498, 0xbed2, 0x0548, 0xbe82, 0x0620, 0xbe42, 0x0708, 0xbe12},
                                 {0x0800, 0xb212, 0x0908, 0xb242, 0x0a44, 0xb282, 0x0b48, 0xb2d2},
                                 {0x0ca8, 0xbe12, 0x0dc8, 0xbe42, 0x0f20, 0xbe82, 0x1088, 0xbed2},
                                 {0x1200, 0xb552, 0x1388, 0xb602, 0x1554, 0xb6c2, 0x16c8, 0xb792},
                                 {0x18b8, 0xc152, 0x1a48, 0xc202, 0x1c20, 0xc2c2, 0x1e08, 0xc392}},
                                {{0xb792'0000, 0xb6c1'3808, 0xb600'1524, 0xb550'0848},
                                 {0xc38e'2988, 0xc2be'18c8, 0xc1fd'b120, 0xc14d'6988},
                                 {0xb2cf'4200, 0xb27f'7a88, 0xb23f'6834, 0xb210'4bc8},
                                 {0xbecd'7d98, 0xbe7e'5d48, 0xbe3e'f620, 0xbe0f'af08},
                                 {0xb214'8800, 0xb245'c108, 0xb286'bf44, 0xb2d8'9348},
                                 {0xbe14'd5a8, 0xbe46'a5c8, 0xbe88'3f20, 0xbed9'f888},
                                 {0xb561'd200, 0xb614'0b88, 0xb6d6'1a54, 0xb7a8'dec8},
                                 {0xc164'31b8, 0xc216'f248, 0xc2d9'8c20, 0xc3ac'4608}},
                                {{0xb6c1'3c57'7192'0000, 0xb550'119c'5f6f'1524},
                                 {0xc2be'2520'79e9'2988, 0xc14d'7ae7'56b5'b120},
                                 {0xb27f'95f3'116f'4200, 0xb210'6e3c'125e'6834},
                                 {0xbe7e'80c0'2dd8'7d98, 0xbe0f'd98a'fd96'f620},
                                 {0xb245'fb9e'bd54'8800, 0xb2d8'd6eb'd155'bf44},
                                 {0xbe46'e86f'edcf'd5a8, 0xbeda'443e'b080'3f20},
                                 {0xb614'6d5a'7541'd200, 0xb7a9'4bab'9c55'1a54},
                                 {0xc217'5c2f'b9cf'31b8, 0xc3ac'bb02'6f71'8c20}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwmulsuvx() {
  asm("vwmulsu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmulsuvx() {
  asm("vwmulsu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwmulsuvx) {
  TestWideningVectorInstruction(ExecVwmulsuvx,
                                ExecMaskedVwmulsuvx,
                                {{0x0000, 0xabaa, 0x0154, 0xacfe, 0x02a8, 0xae52, 0x03fc, 0xafa6},
                                 {0x0550, 0xb0fa, 0x06a4, 0xb24e, 0x07f8, 0xb3a2, 0x094c, 0xb4f6},
                                 {0x0aa0, 0xb64a, 0x0bf4, 0xb79e, 0x0d48, 0xb8f2, 0x0e9c, 0xba46},
                                 {0x0ff0, 0xbb9a, 0x1144, 0xbcee, 0x1298, 0xbe42, 0x13ec, 0xbf96},
                                 {0x1540, 0xc0ea, 0x1694, 0xc23e, 0x17e8, 0xc392, 0x193c, 0xc4e6},
                                 {0x1a90, 0xc63a, 0x1be4, 0xc78e, 0x1d38, 0xc8e2, 0x1e8c, 0xca36},
                                 {0x1fe0, 0xcb8a, 0x2134, 0xccde, 0x2288, 0xce32, 0x23dc, 0xcf86},
                                 {0x2530, 0xd0da, 0x2684, 0xd22e, 0x27d8, 0xd382, 0x292c, 0xd4d6}},
                                {{0xab55'aa00, 0xacac'5354, 0xae02'fca8, 0xaf59'a5fc},
                                 {0xb0b0'4f50, 0xb206'f8a4, 0xb35d'a1f8, 0xb4b4'4b4c},
                                 {0xb60a'f4a0, 0xb761'9df4, 0xb8b8'4748, 0xba0e'f09c},
                                 {0xbb65'99f0, 0xbcbc'4344, 0xbe12'ec98, 0xbf69'95ec},
                                 {0xc0c0'3f40, 0xc216'e894, 0xc36d'91e8, 0xc4c4'3b3c},
                                 {0xc61a'e490, 0xc771'8de4, 0xc8c8'3738, 0xca1e'e08c},
                                 {0xcb75'89e0, 0xcccc'3334, 0xce22'dc88, 0xcf79'85dc},
                                 {0xd0d0'2f30, 0xd226'd884, 0xd37d'81d8, 0xd4d4'2b2c}},
                                {{0xacac'5600'5353'aa00, 0xaf59'ae02'fb50'fca8},
                                 {0xb207'0605'a34e'4f50, 0xb4b4'5e08'4b4b'a1f8},
                                 {0xb761'b60a'f348'f4a0, 0xba0f'0e0d'9b46'4748},
                                 {0xbcbc'6610'4343'99f0, 0xbf69'be12'eb40'ec98},
                                 {0xc217'1615'933e'3f40, 0xc4c4'6e18'3b3b'91e8},
                                 {0xc771'c61a'e338'e490, 0xca1f'1e1d'8b36'3738},
                                 {0xcccc'7620'3333'89e0, 0xcf79'ce22'db30'dc88},
                                 {0xd227'2625'832e'2f30, 0xd4d4'7e28'2b2b'81d8}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwmuluvv() {
  asm("vwmulu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmuluvv() {
  asm("vwmulu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwmulu) {
  TestWideningVectorInstruction(ExecVwmuluvv,
                                ExecMaskedVwmuluvv,
                                {{0x0000, 0x4992, 0x0008, 0x4cc2, 0x0024, 0x5002, 0x0048, 0x5352},
                                 {0x0088, 0x4592, 0x00c8, 0x48c2, 0x0120, 0x4c02, 0x0188, 0x4f52},
                                 {0x0200, 0x64d2, 0x0288, 0x6882, 0x0334, 0x6c42, 0x03c8, 0x7012},
                                 {0x0498, 0x60d2, 0x0548, 0x6482, 0x0620, 0x6842, 0x0708, 0x6c12},
                                 {0x0800, 0x8412, 0x0908, 0x8842, 0x0a44, 0x8c82, 0x0b48, 0x90d2},
                                 {0x0ca8, 0x8012, 0x0dc8, 0x8442, 0x0f20, 0x8882, 0x1088, 0x8cd2},
                                 {0x1200, 0xa752, 0x1388, 0xac02, 0x1554, 0xb0c2, 0x16c8, 0xb592},
                                 {0x18b8, 0xa352, 0x1a48, 0xa802, 0x1c20, 0xacc2, 0x1e08, 0xb192}},
                                {{0x4992'0000, 0x4cc5'3808, 0x5009'1524, 0x535c'0848},
                                 {0x459f'2988, 0x48d2'18c8, 0x4c15'b120, 0x4f69'6988},
                                 {0x64ef'4200, 0x68a3'7a88, 0x6c68'6834, 0x703c'4bc8},
                                 {0x60fe'7d98, 0x64b2'5d48, 0x6876'f620, 0x6c4b'af08},
                                 {0x8454'8800, 0x8889'c108, 0x8ccf'bf44, 0x9124'9348},
                                 {0x8065'd5a8, 0x849a'a5c8, 0x88e0'3f20, 0x8d35'f888},
                                 {0xa7c1'd200, 0xac78'0b88, 0xb13f'1a54, 0xb614'dec8},
                                 {0xa3d5'31b8, 0xa88a'f248, 0xad51'8c20, 0xb228'4608}},
                                {{0x4cc5'ce57'7192'0000, 0x535c'aba5'5f6f'1524},
                                 {0x48d2'a731'79e9'2988, 0x4f6a'04ff'56b5'b120},
                                 {0x68a4'4813'116f'4200, 0x703d'2865'125e'6834},
                                 {0x64b3'22f1'2dd8'7d98, 0x6c4c'83c2'fd96'f620},
                                 {0x888a'cdde'bd54'8800, 0x9125'b134'd155'bf44},
                                 {0x849b'aac0'edcf'd5a8, 0x8d37'0e96'b080'3f20},
                                 {0xac79'5fba'7541'd200, 0xb616'4614'9c55'1a54},
                                 {0xa88c'3ea0'b9cf'31b8, 0xb229'a57a'6f71'8c20}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwmuluvx() {
  asm("vwmulu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmuluvx() {
  asm("vwmulu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwmuluvx) {
  TestWideningVectorInstruction(ExecVwmuluvx,
                                ExecMaskedVwmuluvx,
                                {{0x0000, 0x55aa, 0x0154, 0x56fe, 0x02a8, 0x5852, 0x03fc, 0x59a6},
                                 {0x0550, 0x5afa, 0x06a4, 0x5c4e, 0x07f8, 0x5da2, 0x094c, 0x5ef6},
                                 {0x0aa0, 0x604a, 0x0bf4, 0x619e, 0x0d48, 0x62f2, 0x0e9c, 0x6446},
                                 {0x0ff0, 0x659a, 0x1144, 0x66ee, 0x1298, 0x6842, 0x13ec, 0x6996},
                                 {0x1540, 0x6aea, 0x1694, 0x6c3e, 0x17e8, 0x6d92, 0x193c, 0x6ee6},
                                 {0x1a90, 0x703a, 0x1be4, 0x718e, 0x1d38, 0x72e2, 0x1e8c, 0x7436},
                                 {0x1fe0, 0x758a, 0x2134, 0x76de, 0x2288, 0x7832, 0x23dc, 0x7986},
                                 {0x2530, 0x7ada, 0x2684, 0x7c2e, 0x27d8, 0x7d82, 0x292c, 0x7ed6}},
                                {{0x55ff'aa00, 0x5756'5354, 0x58ac'fca8, 0x5a03'a5fc},
                                 {0x5b5a'4f50, 0x5cb0'f8a4, 0x5e07'a1f8, 0x5f5e'4b4c},
                                 {0x60b4'f4a0, 0x620b'9df4, 0x6362'4748, 0x64b8'f09c},
                                 {0x660f'99f0, 0x6766'4344, 0x68bc'ec98, 0x6a13'95ec},
                                 {0x6b6a'3f40, 0x6cc0'e894, 0x6e17'91e8, 0x6f6e'3b3c},
                                 {0x70c4'e490, 0x721b'8de4, 0x7372'3738, 0x74c8'e08c},
                                 {0x761f'89e0, 0x7776'3334, 0x78cc'dc88, 0x7a23'85dc},
                                 {0x7b7a'2f30, 0x7cd0'd884, 0x7e27'81d8, 0x7f7e'2b2c}},
                                {{0x5757'00aa'5353'aa00, 0x5a04'58ac'fb50'fca8},
                                 {0x5cb1'b0af'a34e'4f50, 0x5f5f'08b2'4b4b'a1f8},
                                 {0x620c'60b4'f348'f4a0, 0x64b9'b8b7'9b46'4748},
                                 {0x6767'10ba'4343'99f0, 0x6a14'68bc'eb40'ec98},
                                 {0x6cc1'c0bf'933e'3f40, 0x6f6f'18c2'3b3b'91e8},
                                 {0x721c'70c4'e338'e490, 0x74c9'c8c7'8b36'3738},
                                 {0x7777'20ca'3333'89e0, 0x7a24'78cc'db30'dc88},
                                 {0x7cd1'd0cf'832e'2f30, 0x7f7f'28d2'2b2b'81d8}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwmulvx() {
  asm("vwmul.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwmulvx() {
  asm("vwmul.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwmulvx) {
  TestWideningVectorInstruction(ExecVwmulvx,
                                ExecMaskedVwmulvx,
                                {{0x0000, 0x2aaa, 0xff54, 0x29fe, 0xfea8, 0x2952, 0xfdfc, 0x28a6},
                                 {0xfd50, 0x27fa, 0xfca4, 0x274e, 0xfbf8, 0x26a2, 0xfb4c, 0x25f6},
                                 {0xfaa0, 0x254a, 0xf9f4, 0x249e, 0xf948, 0x23f2, 0xf89c, 0x2346},
                                 {0xf7f0, 0x229a, 0xf744, 0x21ee, 0xf698, 0x2142, 0xf5ec, 0x2096},
                                 {0xf540, 0x1fea, 0xf494, 0x1f3e, 0xf3e8, 0x1e92, 0xf33c, 0x1de6},
                                 {0xf290, 0x1d3a, 0xf1e4, 0x1c8e, 0xf138, 0x1be2, 0xf08c, 0x1b36},
                                 {0xefe0, 0x1a8a, 0xef34, 0x19de, 0xee88, 0x1932, 0xeddc, 0x1886},
                                 {0xed30, 0x17da, 0xec84, 0x172e, 0xebd8, 0x1682, 0xeb2c, 0x15d6}},
                                {{0x2a55'aa00, 0x29aa'5354, 0x28fe'fca8, 0x2853'a5fc},
                                 {0x27a8'4f50, 0x26fc'f8a4, 0x2651'a1f8, 0x25a6'4b4c},
                                 {0x24fa'f4a0, 0x244f'9df4, 0x23a4'4748, 0x22f8'f09c},
                                 {0x224d'99f0, 0x21a2'4344, 0x20f6'ec98, 0x204b'95ec},
                                 {0x1fa0'3f40, 0x1ef4'e894, 0x1e49'91e8, 0x1d9e'3b3c},
                                 {0x1cf2'e490, 0x1c47'8de4, 0x1b9c'3738, 0x1af0'e08c},
                                 {0x1a45'89e0, 0x199a'3334, 0x18ee'dc88, 0x1843'85dc},
                                 {0x1798'2f30, 0x16ec'd884, 0x1641'81d8, 0x1596'2b2c}},
                                {{0x29a9'd500'5353'aa00, 0x2853'28fe'fb50'fca8},
                                 {0x26fc'7cfd'a34e'4f50, 0x25a5'd0fc'4b4b'a1f8},
                                 {0x244f'24fa'f348'f4a0, 0x22f8'78f9'9b46'4748},
                                 {0x21a1'ccf8'4343'99f0, 0x204b'20f6'eb40'ec98},
                                 {0x1ef4'74f5'933e'3f40, 0x1d9d'c8f4'3b3b'91e8},
                                 {0x1c47'1cf2'e338'e490, 0x1af0'70f1'8b36'3738},
                                 {0x1999'c4f0'3333'89e0, 0x1843'18ee'db30'dc88},
                                 {0x16ec'6ced'832e'2f30, 0x1595'c0ec'2b2b'81d8}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwsubuvv() {
  asm("vwsubu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwsubuvv() {
  asm("vwsubu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwsubuvv) {
  TestWideningVectorInstruction(ExecVwsubuvv,
                                ExecMaskedVwsubuvv,
                                {{0x0000, 0xffef, 0xfffe, 0xffed, 0xfffb, 0xffeb, 0xfffa, 0xffe9},
                                 {0xfff7, 0x0007, 0xfff6, 0x0005, 0xfff4, 0x0003, 0xfff2, 0x0001},
                                 {0xfff0, 0xffdf, 0xffee, 0xffdd, 0xffeb, 0xffdb, 0xffea, 0xffd9},
                                 {0xffe7, 0xfff7, 0xffe6, 0xfff5, 0xffe4, 0xfff3, 0xffe2, 0xfff1},
                                 {0xffe0, 0xffcf, 0xffde, 0xffcd, 0xffdb, 0xffcb, 0xffda, 0xffc9},
                                 {0xffd7, 0xffe7, 0xffd6, 0xffe5, 0xffd4, 0xffe3, 0xffd2, 0xffe1},
                                 {0xffd0, 0xffbf, 0xffce, 0xffbd, 0xffcb, 0xffbb, 0xffca, 0xffb9},
                                 {0xffc7, 0xffd7, 0xffc6, 0xffd5, 0xffc4, 0xffd3, 0xffc2, 0xffd1}},
                                {{0xffff'ef00, 0xffff'ecfe, 0xffff'eafb, 0xffff'e8fa},
                                 {0x0000'06f7, 0x0000'04f6, 0x0000'02f4, 0x0000'00f2},
                                 {0xffff'def0, 0xffff'dcee, 0xffff'daeb, 0xffff'd8ea},
                                 {0xffff'f6e7, 0xffff'f4e6, 0xffff'f2e4, 0xffff'f0e2},
                                 {0xffff'cee0, 0xffff'ccde, 0xffff'cadb, 0xffff'c8da},
                                 {0xffff'e6d7, 0xffff'e4d6, 0xffff'e2d4, 0xffff'e0d2},
                                 {0xffff'bed0, 0xffff'bcce, 0xffff'bacb, 0xffff'b8ca},
                                 {0xffff'd6c7, 0xffff'd4c6, 0xffff'd2c4, 0xffff'd0c2}},
                                {{0xffff'ffff'ecfd'ef00, 0xffff'ffff'e8f9'eafb},
                                 {0x0000'0000'04f6'06f7, 0x0000'0000'00f2'02f4},
                                 {0xffff'ffff'dced'def0, 0xffff'ffff'd8e9'daeb},
                                 {0xffff'ffff'f4e5'f6e7, 0xffff'ffff'f0e1'f2e4},
                                 {0xffff'ffff'ccdd'cee0, 0xffff'ffff'c8d9'cadb},
                                 {0xffff'ffff'e4d5'e6d7, 0xffff'ffff'e0d1'e2d4},
                                 {0xffff'ffff'bccd'bed0, 0xffff'ffff'b8c9'bacb},
                                 {0xffff'ffff'd4c5'd6c7, 0xffff'ffff'd0c1'd2c4}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwsubuvx() {
  asm("vwsubu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwsubuvx() {
  asm("vwsubu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwsubuvx) {
  TestWideningVectorInstruction(ExecVwsubuvx,
                                ExecMaskedVwsubuvx,
                                {{0xff56, 0xffd7, 0xff58, 0xffd9, 0xff5a, 0xffdb, 0xff5c, 0xffdd},
                                 {0xff5e, 0xffdf, 0xff60, 0xffe1, 0xff62, 0xffe3, 0xff64, 0xffe5},
                                 {0xff66, 0xffe7, 0xff68, 0xffe9, 0xff6a, 0xffeb, 0xff6c, 0xffed},
                                 {0xff6e, 0xffef, 0xff70, 0xfff1, 0xff72, 0xfff3, 0xff74, 0xfff5},
                                 {0xff76, 0xfff7, 0xff78, 0xfff9, 0xff7a, 0xfffb, 0xff7c, 0xfffd},
                                 {0xff7e, 0xffff, 0xff80, 0x0001, 0xff82, 0x0003, 0xff84, 0x0005},
                                 {0xff86, 0x0007, 0xff88, 0x0009, 0xff8a, 0x000b, 0xff8c, 0x000d},
                                 {0xff8e, 0x000f, 0xff90, 0x0011, 0xff92, 0x0013, 0xff94, 0x0015}},
                                {{0xffff'd656, 0xffff'd858, 0xffff'da5a, 0xffff'dc5c},
                                 {0xffff'de5e, 0xffff'e060, 0xffff'e262, 0xffff'e464},
                                 {0xffff'e666, 0xffff'e868, 0xffff'ea6a, 0xffff'ec6c},
                                 {0xffff'ee6e, 0xffff'f070, 0xffff'f272, 0xffff'f474},
                                 {0xffff'f676, 0xffff'f878, 0xffff'fa7a, 0xffff'fc7c},
                                 {0xffff'fe7e, 0x0000'0080, 0x0000'0282, 0x0000'0484},
                                 {0x0000'0686, 0x0000'0888, 0x0000'0a8a, 0x0000'0c8c},
                                 {0x0000'0e8e, 0x0000'1090, 0x0000'1292, 0x0000'1494}},
                                {{0xffff'ffff'd857'd656, 0xffff'ffff'dc5b'da5a},
                                 {0xffff'ffff'e05f'de5e, 0xffff'ffff'e463'e262},
                                 {0xffff'ffff'e867'e666, 0xffff'ffff'ec6b'ea6a},
                                 {0xffff'ffff'f06f'ee6e, 0xffff'ffff'f473'f272},
                                 {0xffff'ffff'f877'f676, 0xffff'ffff'fc7b'fa7a},
                                 {0x0000'0000'007f'fe7e, 0x0000'0000'0484'0282},
                                 {0x0000'0000'0888'0686, 0x0000'0000'0c8c'0a8a},
                                 {0x0000'0000'1090'0e8e, 0x0000'0000'1494'1292}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwsubuwv() {
  asm("vwsubu.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwsubuwv() {
  asm("vwsubu.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwsubuwv) {
  TestWideningVectorInstruction(ExecVwsubuwv,
                                ExecMaskedVwsubuwv,
                                {{0x8100, 0x8270, 0x8500, 0x8670, 0x88ff, 0x8a70, 0x8d00, 0x8e70},
                                 {0x90ff, 0x9290, 0x9500, 0x9690, 0x9900, 0x9a90, 0x9d00, 0x9e90},
                                 {0xa100, 0xa270, 0xa500, 0xa670, 0xa8ff, 0xaa70, 0xad00, 0xae70},
                                 {0xb0ff, 0xb290, 0xb500, 0xb690, 0xb900, 0xba90, 0xbd00, 0xbe90},
                                 {0xc100, 0xc270, 0xc500, 0xc670, 0xc8ff, 0xca70, 0xcd00, 0xce70},
                                 {0xd0ff, 0xd290, 0xd500, 0xd690, 0xd900, 0xda90, 0xdd00, 0xde90},
                                 {0xe100, 0xe270, 0xe500, 0xe670, 0xe8ff, 0xea70, 0xed00, 0xee70},
                                 {0xf0ff, 0xf290, 0xf500, 0xf690, 0xf900, 0xfa90, 0xfd00, 0xfe90}},
                                {{0x8301'ef00, 0x8705'ef00, 0x8b09'eeff, 0x8f0d'ef00},
                                 {0x9312'0eff, 0x9716'0f00, 0x9b1a'0f00, 0x9f1e'0f00},
                                 {0xa321'ef00, 0xa725'ef00, 0xab29'eeff, 0xaf2d'ef00},
                                 {0xb332'0eff, 0xb736'0f00, 0xbb3a'0f00, 0xbf3e'0f00},
                                 {0xc341'ef00, 0xc745'ef00, 0xcb49'eeff, 0xcf4d'ef00},
                                 {0xd352'0eff, 0xd756'0f00, 0xdb5a'0f00, 0xdf5e'0f00},
                                 {0xe361'ef00, 0xe765'ef00, 0xeb69'eeff, 0xef6d'ef00},
                                 {0xf372'0eff, 0xf776'0f00, 0xfb7a'0f00, 0xff7e'0f00}},
                                {{0x8706'8503'ecfd'ef00, 0x8f0e'8d0b'ecfd'eeff},
                                 {0x9716'9514'0cfe'0eff, 0x9f1e'9d1c'0cfe'0f00},
                                 {0xa726'a523'ecfd'ef00, 0xaf2e'ad2b'ecfd'eeff},
                                 {0xb736'b534'0cfe'0eff, 0xbf3e'bd3c'0cfe'0f00},
                                 {0xc746'c543'ecfd'ef00, 0xcf4e'cd4b'ecfd'eeff},
                                 {0xd756'd554'0cfe'0eff, 0xdf5e'dd5c'0cfe'0f00},
                                 {0xe766'e563'ecfd'ef00, 0xef6e'ed6b'ecfd'eeff},
                                 {0xf776'f574'0cfe'0eff, 0xff7e'fd7c'0cfe'0f00}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwsubuwx() {
  asm("vwsubu.wx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwsubuwx() {
  asm("vwsubu.wx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwsubuwx) {
  TestWideningVectorInstruction(ExecVwsubuwx,
                                ExecMaskedVwsubuwx,
                                {{0x8056, 0x8258, 0x845a, 0x865c, 0x885e, 0x8a60, 0x8c62, 0x8e64},
                                 {0x9066, 0x9268, 0x946a, 0x966c, 0x986e, 0x9a70, 0x9c72, 0x9e74},
                                 {0xa076, 0xa278, 0xa47a, 0xa67c, 0xa87e, 0xaa80, 0xac82, 0xae84},
                                 {0xb086, 0xb288, 0xb48a, 0xb68c, 0xb88e, 0xba90, 0xbc92, 0xbe94},
                                 {0xc096, 0xc298, 0xc49a, 0xc69c, 0xc89e, 0xcaa0, 0xcca2, 0xcea4},
                                 {0xd0a6, 0xd2a8, 0xd4aa, 0xd6ac, 0xd8ae, 0xdab0, 0xdcb2, 0xdeb4},
                                 {0xe0b6, 0xe2b8, 0xe4ba, 0xe6bc, 0xe8be, 0xeac0, 0xecc2, 0xeec4},
                                 {0xf0c6, 0xf2c8, 0xf4ca, 0xf6cc, 0xf8ce, 0xfad0, 0xfcd2, 0xfed4}},
                                {{0x8301'd656, 0x8705'da5a, 0x8b09'de5e, 0x8f0d'e262},
                                 {0x9311'e666, 0x9715'ea6a, 0x9b19'ee6e, 0x9f1d'f272},
                                 {0xa321'f676, 0xa725'fa7a, 0xab29'fe7e, 0xaf2e'0282},
                                 {0xb332'0686, 0xb736'0a8a, 0xbb3a'0e8e, 0xbf3e'1292},
                                 {0xc342'1696, 0xc746'1a9a, 0xcb4a'1e9e, 0xcf4e'22a2},
                                 {0xd352'26a6, 0xd756'2aaa, 0xdb5a'2eae, 0xdf5e'32b2},
                                 {0xe362'36b6, 0xe766'3aba, 0xeb6a'3ebe, 0xef6e'42c2},
                                 {0xf372'46c6, 0xf776'4aca, 0xfb7a'4ece, 0xff7e'52d2}},
                                {{0x8706'8503'd857'd656, 0x8f0e'8d0b'e05f'de5e},
                                 {0x9716'9513'e867'e666, 0x9f1e'9d1b'f06f'ee6e},
                                 {0xa726'a523'f877'f676, 0xaf2e'ad2c'007f'fe7e},
                                 {0xb736'b534'0888'0686, 0xbf3e'bd3c'1090'0e8e},
                                 {0xc746'c544'1898'1696, 0xcf4e'cd4c'20a0'1e9e},
                                 {0xd756'd554'28a8'26a6, 0xdf5e'dd5c'30b0'2eae},
                                 {0xe766'e564'38b8'36b6, 0xef6e'ed6c'40c0'3ebe},
                                 {0xf776'f574'48c8'46c6, 0xff7e'fd7c'50d0'4ece}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwsubvv() {
  asm("vwsub.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwsubvv() {
  asm("vwsub.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwsubvv) {
  TestWideningVectorInstruction(ExecVwsubvv,
                                ExecMaskedVwsubvv,
                                {{0x0000, 0xffef, 0xfffe, 0xffed, 0xfffb, 0xffeb, 0xfffa, 0xffe9},
                                 {0xfff7, 0x0007, 0xfff6, 0x0005, 0xfff4, 0x0003, 0xfff2, 0x0001},
                                 {0xfff0, 0xffdf, 0xffee, 0xffdd, 0xffeb, 0xffdb, 0xffea, 0xffd9},
                                 {0xffe7, 0xfff7, 0xffe6, 0xfff5, 0xffe4, 0xfff3, 0xffe2, 0xfff1},
                                 {0xffe0, 0xffcf, 0xffde, 0xffcd, 0xffdb, 0xffcb, 0xffda, 0xffc9},
                                 {0xffd7, 0xffe7, 0xffd6, 0xffe5, 0xffd4, 0xffe3, 0xffd2, 0xffe1},
                                 {0xffd0, 0xffbf, 0xffce, 0xffbd, 0xffcb, 0xffbb, 0xffca, 0xffb9},
                                 {0xffc7, 0xffd7, 0xffc6, 0xffd5, 0xffc4, 0xffd3, 0xffc2, 0xffd1}},
                                {{0xffff'ef00, 0xffff'ecfe, 0xffff'eafb, 0xffff'e8fa},
                                 {0x0000'06f7, 0x0000'04f6, 0x0000'02f4, 0x0000'00f2},
                                 {0xffff'def0, 0xffff'dcee, 0xffff'daeb, 0xffff'd8ea},
                                 {0xffff'f6e7, 0xffff'f4e6, 0xffff'f2e4, 0xffff'f0e2},
                                 {0xffff'cee0, 0xffff'ccde, 0xffff'cadb, 0xffff'c8da},
                                 {0xffff'e6d7, 0xffff'e4d6, 0xffff'e2d4, 0xffff'e0d2},
                                 {0xffff'bed0, 0xffff'bcce, 0xffff'bacb, 0xffff'b8ca},
                                 {0xffff'd6c7, 0xffff'd4c6, 0xffff'd2c4, 0xffff'd0c2}},
                                {{0xffff'ffff'ecfd'ef00, 0xffff'ffff'e8f9'eafb},
                                 {0x0000'0000'04f6'06f7, 0x0000'0000'00f2'02f4},
                                 {0xffff'ffff'dced'def0, 0xffff'ffff'd8e9'daeb},
                                 {0xffff'ffff'f4e5'f6e7, 0xffff'ffff'f0e1'f2e4},
                                 {0xffff'ffff'ccdd'cee0, 0xffff'ffff'c8d9'cadb},
                                 {0xffff'ffff'e4d5'e6d7, 0xffff'ffff'e0d1'e2d4},
                                 {0xffff'ffff'bccd'bed0, 0xffff'ffff'b8c9'bacb},
                                 {0xffff'ffff'd4c5'd6c7, 0xffff'ffff'd0c1'd2c4}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwsubvx() {
  asm("vwsub.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwsubvx() {
  asm("vwsub.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwsubvx) {
  TestWideningVectorInstruction(ExecVwsubvx,
                                ExecMaskedVwsubvx,
                                {{0x0056, 0xffd7, 0x0058, 0xffd9, 0x005a, 0xffdb, 0x005c, 0xffdd},
                                 {0x005e, 0xffdf, 0x0060, 0xffe1, 0x0062, 0xffe3, 0x0064, 0xffe5},
                                 {0x0066, 0xffe7, 0x0068, 0xffe9, 0x006a, 0xffeb, 0x006c, 0xffed},
                                 {0x006e, 0xffef, 0x0070, 0xfff1, 0x0072, 0xfff3, 0x0074, 0xfff5},
                                 {0x0076, 0xfff7, 0x0078, 0xfff9, 0x007a, 0xfffb, 0x007c, 0xfffd},
                                 {0x007e, 0xffff, 0x0080, 0x0001, 0x0082, 0x0003, 0x0084, 0x0005},
                                 {0x0086, 0x0007, 0x0088, 0x0009, 0x008a, 0x000b, 0x008c, 0x000d},
                                 {0x008e, 0x000f, 0x0090, 0x0011, 0x0092, 0x0013, 0x0094, 0x0015}},
                                {{0xffff'd656, 0xffff'd858, 0xffff'da5a, 0xffff'dc5c},
                                 {0xffff'de5e, 0xffff'e060, 0xffff'e262, 0xffff'e464},
                                 {0xffff'e666, 0xffff'e868, 0xffff'ea6a, 0xffff'ec6c},
                                 {0xffff'ee6e, 0xffff'f070, 0xffff'f272, 0xffff'f474},
                                 {0xffff'f676, 0xffff'f878, 0xffff'fa7a, 0xffff'fc7c},
                                 {0xffff'fe7e, 0x0000'0080, 0x0000'0282, 0x0000'0484},
                                 {0x0000'0686, 0x0000'0888, 0x0000'0a8a, 0x0000'0c8c},
                                 {0x0000'0e8e, 0x0000'1090, 0x0000'1292, 0x0000'1494}},
                                {{0xffff'ffff'd857'd656, 0xffff'ffff'dc5b'da5a},
                                 {0xffff'ffff'e05f'de5e, 0xffff'ffff'e463'e262},
                                 {0xffff'ffff'e867'e666, 0xffff'ffff'ec6b'ea6a},
                                 {0xffff'ffff'f06f'ee6e, 0xffff'ffff'f473'f272},
                                 {0xffff'ffff'f877'f676, 0xffff'ffff'fc7b'fa7a},
                                 {0x0000'0000'007f'fe7e, 0x0000'0000'0484'0282},
                                 {0x0000'0000'0888'0686, 0x0000'0000'0c8c'0a8a},
                                 {0x0000'0000'1090'0e8e, 0x0000'0000'1494'1292}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwsubwv() {
  asm("vwsub.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwsubwv() {
  asm("vwsub.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwsubwv) {
  TestWideningVectorInstruction(ExecVwsubwv,
                                ExecMaskedVwsubwv,
                                {{0x8100, 0x8370, 0x8500, 0x8770, 0x88ff, 0x8b70, 0x8d00, 0x8f70},
                                 {0x90ff, 0x9390, 0x9500, 0x9790, 0x9900, 0x9b90, 0x9d00, 0x9f90},
                                 {0xa100, 0xa370, 0xa500, 0xa770, 0xa8ff, 0xab70, 0xad00, 0xaf70},
                                 {0xb0ff, 0xb390, 0xb500, 0xb790, 0xb900, 0xbb90, 0xbd00, 0xbf90},
                                 {0xc100, 0xc370, 0xc500, 0xc770, 0xc8ff, 0xcb70, 0xcd00, 0xcf70},
                                 {0xd0ff, 0xd390, 0xd500, 0xd790, 0xd900, 0xdb90, 0xdd00, 0xdf90},
                                 {0xe100, 0xe370, 0xe500, 0xe770, 0xe8ff, 0xeb70, 0xed00, 0xef70},
                                 {0xf0ff, 0xf390, 0xf500, 0xf790, 0xf900, 0xfb90, 0xfd00, 0xff90}},
                                {{0x8302'ef00, 0x8706'ef00, 0x8b0a'eeff, 0x8f0e'ef00},
                                 {0x9313'0eff, 0x9717'0f00, 0x9b1b'0f00, 0x9f1f'0f00},
                                 {0xa322'ef00, 0xa726'ef00, 0xab2a'eeff, 0xaf2e'ef00},
                                 {0xb333'0eff, 0xb737'0f00, 0xbb3b'0f00, 0xbf3f'0f00},
                                 {0xc342'ef00, 0xc746'ef00, 0xcb4a'eeff, 0xcf4e'ef00},
                                 {0xd353'0eff, 0xd757'0f00, 0xdb5b'0f00, 0xdf5f'0f00},
                                 {0xe362'ef00, 0xe766'ef00, 0xeb6a'eeff, 0xef6e'ef00},
                                 {0xf373'0eff, 0xf777'0f00, 0xfb7b'0f00, 0xff7f'0f00}},
                                {{0x8706'8504'ecfd'ef00, 0x8f0e'8d0c'ecfd'eeff},
                                 {0x9716'9515'0cfe'0eff, 0x9f1e'9d1d'0cfe'0f00},
                                 {0xa726'a524'ecfd'ef00, 0xaf2e'ad2c'ecfd'eeff},
                                 {0xb736'b535'0cfe'0eff, 0xbf3e'bd3d'0cfe'0f00},
                                 {0xc746'c544'ecfd'ef00, 0xcf4e'cd4c'ecfd'eeff},
                                 {0xd756'd555'0cfe'0eff, 0xdf5e'dd5d'0cfe'0f00},
                                 {0xe766'e564'ecfd'ef00, 0xef6e'ed6c'ecfd'eeff},
                                 {0xf776'f575'0cfe'0eff, 0xff7e'fd7d'0cfe'0f00}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVwsubwx() {
  asm("vwsub.wx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVwsubwx() {
  asm("vwsub.wx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVwsubwx) {
  TestWideningVectorInstruction(ExecVwsubwx,
                                ExecMaskedVwsubwx,
                                {{0x8156, 0x8358, 0x855a, 0x875c, 0x895e, 0x8b60, 0x8d62, 0x8f64},
                                 {0x9166, 0x9368, 0x956a, 0x976c, 0x996e, 0x9b70, 0x9d72, 0x9f74},
                                 {0xa176, 0xa378, 0xa57a, 0xa77c, 0xa97e, 0xab80, 0xad82, 0xaf84},
                                 {0xb186, 0xb388, 0xb58a, 0xb78c, 0xb98e, 0xbb90, 0xbd92, 0xbf94},
                                 {0xc196, 0xc398, 0xc59a, 0xc79c, 0xc99e, 0xcba0, 0xcda2, 0xcfa4},
                                 {0xd1a6, 0xd3a8, 0xd5aa, 0xd7ac, 0xd9ae, 0xdbb0, 0xddb2, 0xdfb4},
                                 {0xe1b6, 0xe3b8, 0xe5ba, 0xe7bc, 0xe9be, 0xebc0, 0xedc2, 0xefc4},
                                 {0xf1c6, 0xf3c8, 0xf5ca, 0xf7cc, 0xf9ce, 0xfbd0, 0xfdd2, 0xffd4}},
                                {{0x8302'd656, 0x8706'da5a, 0x8b0a'de5e, 0x8f0e'e262},
                                 {0x9312'e666, 0x9716'ea6a, 0x9b1a'ee6e, 0x9f1e'f272},
                                 {0xa322'f676, 0xa726'fa7a, 0xab2a'fe7e, 0xaf2f'0282},
                                 {0xb333'0686, 0xb737'0a8a, 0xbb3b'0e8e, 0xbf3f'1292},
                                 {0xc343'1696, 0xc747'1a9a, 0xcb4b'1e9e, 0xcf4f'22a2},
                                 {0xd353'26a6, 0xd757'2aaa, 0xdb5b'2eae, 0xdf5f'32b2},
                                 {0xe363'36b6, 0xe767'3aba, 0xeb6b'3ebe, 0xef6f'42c2},
                                 {0xf373'46c6, 0xf777'4aca, 0xfb7b'4ece, 0xff7f'52d2}},
                                {{0x8706'8504'd857'd656, 0x8f0e'8d0c'e05f'de5e},
                                 {0x9716'9514'e867'e666, 0x9f1e'9d1c'f06f'ee6e},
                                 {0xa726'a524'f877'f676, 0xaf2e'ad2d'007f'fe7e},
                                 {0xb736'b535'0888'0686, 0xbf3e'bd3d'1090'0e8e},
                                 {0xc746'c545'1898'1696, 0xcf4e'cd4d'20a0'1e9e},
                                 {0xd756'd555'28a8'26a6, 0xdf5e'dd5d'30b0'2eae},
                                 {0xe766'e565'38b8'36b6, 0xef6e'ed6d'40c0'3ebe},
                                 {0xf776'f575'48c8'46c6, 0xff7e'fd7d'50d0'4ece}},
                                kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVzextvf8() {
  asm("vzext.vf8  v8, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVzextvf8() {
  asm("vzext.vf8  v8, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsextvf8() {
  asm("vsext.vf8  v8, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsextvf8() {
  asm("vsext.vf8  v8, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVzextvf4() {
  asm("vzext.vf4  v8, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVzextvf4() {
  asm("vzext.vf4  v8, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsextvf4() {
  asm("vsext.vf4  v8, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsextvf4() {
  asm("vsext.vf4  v8, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVzextvf2() {
  asm("vzext.vf2  v8, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVzextvf2() {
  asm("vzext.vf2  v8, v16, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsextvf2() {
  asm("vsext.vf2  v8, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsextvf2() {
  asm("vsext.vf2  v8, v16, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVXext) {
  TestExtendingVectorInstruction(ExecVzextvf8,
                                 ExecMaskedVzextvf8,
                                 {},
                                 {},
                                 {{0x0000'0000'0000'0000, 0x0000'0000'0000'0081},
                                  {0x0000'0000'0000'0002, 0x0000'0000'0000'0083},
                                  {0x0000'0000'0000'0004, 0x0000'0000'0000'0085},
                                  {0x0000'0000'0000'0006, 0x0000'0000'0000'0087},
                                  {0x0000'0000'0000'0008, 0x0000'0000'0000'0089},
                                  {0x0000'0000'0000'000a, 0x0000'0000'0000'008b},
                                  {0x0000'0000'0000'000c, 0x0000'0000'0000'008d},
                                  {0x0000'0000'0000'000e, 0x0000'0000'0000'008f}},
                                 kVectorCalculationsSource,
                                 8);

  TestExtendingVectorInstruction(ExecVsextvf8,
                                 ExecMaskedVsextvf8,
                                 {},
                                 {},
                                 {{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ff81},
                                  {0x0000'0000'0000'0002, 0xffff'ffff'ffff'ff83},
                                  {0x0000'0000'0000'0004, 0xffff'ffff'ffff'ff85},
                                  {0x0000'0000'0000'0006, 0xffff'ffff'ffff'ff87},
                                  {0x0000'0000'0000'0008, 0xffff'ffff'ffff'ff89},
                                  {0x0000'0000'0000'000a, 0xffff'ffff'ffff'ff8b},
                                  {0x0000'0000'0000'000c, 0xffff'ffff'ffff'ff8d},
                                  {0x0000'0000'0000'000e, 0xffff'ffff'ffff'ff8f}},
                                 kVectorCalculationsSource,
                                 8);

  TestExtendingVectorInstruction(ExecVzextvf4,
                                 ExecMaskedVzextvf4,
                                 {},
                                 {{0x0000'0000, 0x0000'0081, 0x0000'0002, 0x0000'0083},
                                  {0x0000'0004, 0x0000'0085, 0x0000'0006, 0x0000'0087},
                                  {0x0000'0008, 0x0000'0089, 0x0000'000a, 0x0000'008b},
                                  {0x0000'000c, 0x0000'008d, 0x0000'000e, 0x0000'008f},
                                  {0x0000'0010, 0x0000'0091, 0x0000'0012, 0x0000'0093},
                                  {0x0000'0014, 0x0000'0095, 0x0000'0016, 0x0000'0097},
                                  {0x0000'0018, 0x0000'0099, 0x0000'001a, 0x0000'009b},
                                  {0x0000'001c, 0x0000'009d, 0x0000'001e, 0x0000'009f}},
                                 {{0x0000'0000'0000'8100, 0x0000'0000'0000'8302},
                                  {0x0000'0000'0000'8504, 0x0000'0000'0000'8706},
                                  {0x0000'0000'0000'8908, 0x0000'0000'0000'8b0a},
                                  {0x0000'0000'0000'8d0c, 0x0000'0000'0000'8f0e},
                                  {0x0000'0000'0000'9110, 0x0000'0000'0000'9312},
                                  {0x0000'0000'0000'9514, 0x0000'0000'0000'9716},
                                  {0x0000'0000'0000'9918, 0x0000'0000'0000'9b1a},
                                  {0x0000'0000'0000'9d1c, 0x0000'0000'0000'9f1e}},
                                 kVectorCalculationsSource,
                                 4);

  TestExtendingVectorInstruction(ExecVsextvf4,
                                 ExecMaskedVsextvf4,
                                 {},
                                 {{0x0000'0000, 0xffff'ff81, 0x0000'0002, 0xffff'ff83},
                                  {0x0000'0004, 0xffff'ff85, 0x0000'0006, 0xffff'ff87},
                                  {0x0000'0008, 0xffff'ff89, 0x0000'000a, 0xffff'ff8b},
                                  {0x0000'000c, 0xffff'ff8d, 0x0000'000e, 0xffff'ff8f},
                                  {0x0000'0010, 0xffff'ff91, 0x0000'0012, 0xffff'ff93},
                                  {0x0000'0014, 0xffff'ff95, 0x0000'0016, 0xffff'ff97},
                                  {0x0000'0018, 0xffff'ff99, 0x0000'001a, 0xffff'ff9b},
                                  {0x0000'001c, 0xffff'ff9d, 0x0000'001e, 0xffff'ff9f}},
                                 {{0xffff'ffff'ffff'8100, 0xffff'ffff'ffff'8302},
                                  {0xffff'ffff'ffff'8504, 0xffff'ffff'ffff'8706},
                                  {0xffff'ffff'ffff'8908, 0xffff'ffff'ffff'8b0a},
                                  {0xffff'ffff'ffff'8d0c, 0xffff'ffff'ffff'8f0e},
                                  {0xffff'ffff'ffff'9110, 0xffff'ffff'ffff'9312},
                                  {0xffff'ffff'ffff'9514, 0xffff'ffff'ffff'9716},
                                  {0xffff'ffff'ffff'9918, 0xffff'ffff'ffff'9b1a},
                                  {0xffff'ffff'ffff'9d1c, 0xffff'ffff'ffff'9f1e}},
                                 kVectorCalculationsSource,
                                 4);

  TestExtendingVectorInstruction(ExecVzextvf2,
                                 ExecMaskedVzextvf2,
                                 {{0x0000, 0x0081, 0x0002, 0x0083, 0x0004, 0x0085, 0x0006, 0x0087},
                                  {0x0008, 0x0089, 0x000a, 0x008b, 0x000c, 0x008d, 0x000e, 0x008f},
                                  {0x0010, 0x0091, 0x0012, 0x0093, 0x0014, 0x0095, 0x0016, 0x0097},
                                  {0x0018, 0x0099, 0x001a, 0x009b, 0x001c, 0x009d, 0x001e, 0x009f},
                                  {0x0020, 0x00a1, 0x0022, 0x00a3, 0x0024, 0x00a5, 0x0026, 0x00a7},
                                  {0x0028, 0x00a9, 0x002a, 0x00ab, 0x002c, 0x00ad, 0x002e, 0x00af},
                                  {0x0030, 0x00b1, 0x0032, 0x00b3, 0x0034, 0x00b5, 0x0036, 0x00b7},
                                  {0x0038, 0x00b9, 0x003a, 0x00bb, 0x003c, 0x00bd, 0x003e, 0x00bf}},
                                 {{0x0000'8100, 0x0000'8302, 0x0000'8504, 0x0000'8706},
                                  {0x0000'8908, 0x0000'8b0a, 0x0000'8d0c, 0x0000'8f0e},
                                  {0x0000'9110, 0x0000'9312, 0x0000'9514, 0x0000'9716},
                                  {0x0000'9918, 0x0000'9b1a, 0x0000'9d1c, 0x0000'9f1e},
                                  {0x0000'a120, 0x0000'a322, 0x0000'a524, 0x0000'a726},
                                  {0x0000'a928, 0x0000'ab2a, 0x0000'ad2c, 0x0000'af2e},
                                  {0x0000'b130, 0x0000'b332, 0x0000'b534, 0x0000'b736},
                                  {0x0000'b938, 0x0000'bb3a, 0x0000'bd3c, 0x0000'bf3e}},
                                 {{0x0000'0000'8302'8100, 0x0000'0000'8706'8504},
                                  {0x0000'0000'8b0a'8908, 0x0000'0000'8f0e'8d0c},
                                  {0x0000'0000'9312'9110, 0x0000'0000'9716'9514},
                                  {0x0000'0000'9b1a'9918, 0x0000'0000'9f1e'9d1c},
                                  {0x0000'0000'a322'a120, 0x0000'0000'a726'a524},
                                  {0x0000'0000'ab2a'a928, 0x0000'0000'af2e'ad2c},
                                  {0x0000'0000'b332'b130, 0x0000'0000'b736'b534},
                                  {0x0000'0000'bb3a'b938, 0x0000'0000'bf3e'bd3c}},
                                 kVectorCalculationsSource,
                                 2);

  TestExtendingVectorInstruction(ExecVsextvf2,
                                 ExecMaskedVsextvf2,
                                 {{0x0000, 0xff81, 0x0002, 0xff83, 0x0004, 0xff85, 0x0006, 0xff87},
                                  {0x0008, 0xff89, 0x000a, 0xff8b, 0x000c, 0xff8d, 0x000e, 0xff8f},
                                  {0x0010, 0xff91, 0x0012, 0xff93, 0x0014, 0xff95, 0x0016, 0xff97},
                                  {0x0018, 0xff99, 0x001a, 0xff9b, 0x001c, 0xff9d, 0x001e, 0xff9f},
                                  {0x0020, 0xffa1, 0x0022, 0xffa3, 0x0024, 0xffa5, 0x0026, 0xffa7},
                                  {0x0028, 0xffa9, 0x002a, 0xffab, 0x002c, 0xffad, 0x002e, 0xffaf},
                                  {0x0030, 0xffb1, 0x0032, 0xffb3, 0x0034, 0xffb5, 0x0036, 0xffb7},
                                  {0x0038, 0xffb9, 0x003a, 0xffbb, 0x003c, 0xffbd, 0x003e, 0xffbf}},
                                 {{0xffff'8100, 0xffff'8302, 0xffff'8504, 0xffff'8706},
                                  {0xffff'8908, 0xffff'8b0a, 0xffff'8d0c, 0xffff'8f0e},
                                  {0xffff'9110, 0xffff'9312, 0xffff'9514, 0xffff'9716},
                                  {0xffff'9918, 0xffff'9b1a, 0xffff'9d1c, 0xffff'9f1e},
                                  {0xffff'a120, 0xffff'a322, 0xffff'a524, 0xffff'a726},
                                  {0xffff'a928, 0xffff'ab2a, 0xffff'ad2c, 0xffff'af2e},
                                  {0xffff'b130, 0xffff'b332, 0xffff'b534, 0xffff'b736},
                                  {0xffff'b938, 0xffff'bb3a, 0xffff'bd3c, 0xffff'bf3e}},
                                 {{0xffff'ffff'8302'8100, 0xffff'ffff'8706'8504},
                                  {0xffff'ffff'8b0a'8908, 0xffff'ffff'8f0e'8d0c},
                                  {0xffff'ffff'9312'9110, 0xffff'ffff'9716'9514},
                                  {0xffff'ffff'9b1a'9918, 0xffff'ffff'9f1e'9d1c},
                                  {0xffff'ffff'a322'a120, 0xffff'ffff'a726'a524},
                                  {0xffff'ffff'ab2a'a928, 0xffff'ffff'af2e'ad2c},
                                  {0xffff'ffff'b332'b130, 0xffff'ffff'b736'b534},
                                  {0xffff'ffff'bb3a'b938, 0xffff'ffff'bf3e'bd3c}},
                                 kVectorCalculationsSource,
                                 2);
}

[[gnu::naked]] void ExecVmulhsuvv() {
  asm("vmulhsu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmulhsuvv() {
  asm("vmulhsu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVmulhsuvx() {
  asm("vmulhsu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVmulhsuvx() {
  asm("vmulhsu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVmulhsu) {
  TestVectorInstruction(
      ExecVmulhsuvv,
      ExecMaskedVmulhsuvv,
      {{0, 255, 0, 253, 0, 251, 0, 249, 0, 247, 0, 245, 1, 244, 1, 242},
       {2, 241, 2, 239, 3, 238, 3, 237, 4, 235, 5, 234, 6, 233, 7, 232},
       {8, 231, 9, 230, 10, 229, 11, 228, 12, 228, 13, 227, 15, 226, 16, 226},
       {18, 225, 19, 225, 21, 224, 22, 224, 24, 224, 26, 224, 28, 224, 30, 224},
       {32, 224, 34, 224, 36, 224, 38, 224, 40, 224, 42, 224, 45, 225, 47, 225},
       {50, 226, 52, 226, 55, 227, 57, 228, 60, 228, 63, 229, 66, 230, 69, 231},
       {72, 232, 75, 233, 78, 234, 81, 235, 84, 237, 87, 238, 91, 239, 94, 241},
       {98, 242, 101, 244, 105, 245, 108, 247, 112, 249, 116, 251, 120, 253, 124, 255}},
      {{0xff02, 0xfd10, 0xfb2d, 0xf95c, 0xf79a, 0xf5e9, 0xf448, 0xf2b7},
       {0xf136, 0xefc5, 0xee64, 0xed13, 0xebd2, 0xeaa2, 0xe982, 0xe872},
       {0xe772, 0xe682, 0xe5a2, 0xe4d3, 0xe413, 0xe364, 0xe2c4, 0xe235},
       {0xe1b6, 0xe147, 0xe0e8, 0xe09a, 0xe05b, 0xe02d, 0xe00f, 0xe001},
       {0xe003, 0xe015, 0xe037, 0xe069, 0xe0ac, 0xe0fe, 0xe161, 0xe1d4},
       {0xe257, 0xe2ea, 0xe38d, 0xe441, 0xe504, 0xe5d8, 0xe6bb, 0xe7af},
       {0xe8b3, 0xe9c7, 0xeaec, 0xec20, 0xed64, 0xeeb9, 0xf01e, 0xf193},
       {0xf318, 0xf4ad, 0xf652, 0xf807, 0xf9cd, 0xfba2, 0xfd88, 0xff7e}},
      {{0xfd10'1a16, 0xf95c'aad6, 0xf5e9'bc58, 0xf2b7'4e9b},
       {0xefc5'619f, 0xed13'f564, 0xeaa3'09ea, 0xe872'9f31},
       {0xe682'b539, 0xe4d3'4c01, 0xe364'638b, 0xe235'fbd7},
       {0xe148'14e2, 0xe09a'aeaf, 0xe02d'c93d, 0xe001'648c},
       {0xe015'809c, 0xe06a'1d6d, 0xe0ff'3aff, 0xe1d4'd952},
       {0xe2ea'f866, 0xe441'983b, 0xe5d8'b8d1, 0xe7b0'5a28},
       {0xe9c8'7c40, 0xec21'1f19, 0xeeba'42b3, 0xf193'e70e},
       {0xf4ae'0c2a, 0xf808'b207, 0xfba3'd8a5, 0xff7f'8004}},
      {{0xf95c'aad6'78f5'63b8, 0xf2b7'4e9b'bf9d'55cb},
       {0xed13'f564'2968'6900, 0xe872'9f31'6a0c'5913},
       {0xe4d3'4c01'edf3'8a67, 0xe235'fbd7'2893'787a},
       {0xe09a'aeaf'c696'c7ef, 0xe001'648c'fb32'b402},
       {0xe06a'1d6d'b352'2196, 0xe1d4'd952'e1ea'0ba9},
       {0xe441'983b'b425'975e, 0xe7b0'5a28'dcb9'7f71},
       {0xec21'1f19'c911'2946, 0xf193'e70e'eba1'0f59},
       {0xf808'b207'f214'd74e, 0xff7f'8005'0ea0'bb61}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVmulhsuvx,
                        ExecMaskedVmulhsuvx,
                        {{0, 171, 1, 172, 2, 174, 3, 175, 5, 176, 6, 178, 7, 179, 9, 180},
                         {10, 182, 11, 183, 13, 184, 14, 186, 15, 187, 17, 188, 18, 190, 19, 191},
                         {21, 192, 22, 194, 23, 195, 25, 196, 26, 198, 27, 199, 29, 200, 30, 202},
                         {31, 203, 33, 204, 34, 206, 35, 207, 37, 208, 38, 210, 39, 211, 41, 212},
                         {42, 214, 43, 215, 45, 216, 46, 218, 47, 219, 49, 220, 50, 222, 51, 223},
                         {53, 224, 54, 226, 55, 227, 57, 228, 58, 230, 59, 231, 61, 232, 62, 234},
                         {63, 235, 65, 236, 66, 238, 67, 239, 69, 240, 70, 242, 71, 243, 73, 244},
                         {74, 246, 75, 247, 77, 248, 78, 250, 79, 251, 81, 252, 82, 254, 83, 255}},
                        {{0xab55, 0xacac, 0xae02, 0xaf59, 0xb0b0, 0xb206, 0xb35d, 0xb4b4},
                         {0xb60a, 0xb761, 0xb8b8, 0xba0e, 0xbb65, 0xbcbc, 0xbe12, 0xbf69},
                         {0xc0c0, 0xc216, 0xc36d, 0xc4c4, 0xc61a, 0xc771, 0xc8c8, 0xca1e},
                         {0xcb75, 0xcccc, 0xce22, 0xcf79, 0xd0d0, 0xd226, 0xd37d, 0xd4d4},
                         {0xd62a, 0xd781, 0xd8d8, 0xda2e, 0xdb85, 0xdcdc, 0xde32, 0xdf89},
                         {0xe0e0, 0xe236, 0xe38d, 0xe4e4, 0xe63a, 0xe791, 0xe8e8, 0xea3e},
                         {0xeb95, 0xecec, 0xee42, 0xef99, 0xf0f0, 0xf246, 0xf39d, 0xf4f4},
                         {0xf64a, 0xf7a1, 0xf8f8, 0xfa4e, 0xfba5, 0xfcfc, 0xfe52, 0xffa9}},
                        {{0xacac'5600, 0xaf59'ae02, 0xb207'0605, 0xb4b4'5e08},
                         {0xb761'b60a, 0xba0f'0e0d, 0xbcbc'6610, 0xbf69'be12},
                         {0xc217'1615, 0xc4c4'6e18, 0xc771'c61a, 0xca1f'1e1d},
                         {0xcccc'7620, 0xcf79'ce22, 0xd227'2625, 0xd4d4'7e28},
                         {0xd781'd62a, 0xda2f'2e2d, 0xdcdc'8630, 0xdf89'de32},
                         {0xe237'3635, 0xe4e4'8e38, 0xe791'e63a, 0xea3f'3e3d},
                         {0xecec'9640, 0xef99'ee42, 0xf247'4645, 0xf4f4'9e48},
                         {0xf7a1'f64a, 0xfa4f'4e4d, 0xfcfc'a650, 0xffa9'fe52}},
                        {{0xaf59'ae03'0201'ab55, 0xb4b4'5e08'5cb1'b0b0},
                         {0xba0f'0e0d'b761'b60a, 0xbf69'be13'1211'bb65},
                         {0xc4c4'6e18'6cc1'c0c0, 0xca1f'1e1d'c771'c61a},
                         {0xcf79'ce23'2221'cb75, 0xd4d4'7e28'7cd1'd0d0},
                         {0xda2f'2e2d'd781'd62a, 0xdf89'de33'3231'db85},
                         {0xe4e4'8e38'8ce1'e0e0, 0xea3f'3e3d'e791'e63a},
                         {0xef99'ee43'4241'eb95, 0xf4f4'9e48'9cf1'f0f0},
                         {0xfa4f'4e4d'f7a1'f64a, 0xffa9'fe53'5251'fba5}},
                        kVectorCalculationsSourceLegacy);
}

// TODO(b/301577077): Add vi tests with non-zero shift.
[[gnu::naked]] void ExecVslidedownvi() {
  asm("vslidedown.vi  v8, v24, 0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVslidedownvi() {
  asm("vslidedown.vi  v8, v24, 0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVslidedownvx() {
  asm("vslidedown.vx  v8, v24, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVslidedownvx() {
  asm("vslidedown.vx  v8, v24, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVslidedown) {
  // With slide offset equal zero, this is equivalent to Vmv.
  TestVectorInstruction(
      ExecVslidedownvi,
      ExecMaskedVslidedownvi,
      {{0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126},
       {128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158},
       {160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190},
       {192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222},
       {224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254}},
      {{0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc}},
      {{0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
      {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
       {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
      kVectorCalculationsSourceLegacy);

  // VLMUL = 0.
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x1e1c'1a18'1614'1211, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{17, 18, 20, 22, 24, 26, 28, 30, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 1
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/1,
      /*skip=*/0);
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 2
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 64},
       {66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94, 96},
       {98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x4240},
       {0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c, 0x6260},
       {0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x4644'4240},
       {0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58, 0x6664'6260},
       {0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 74, 76, 78},
       {81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 105, 106, 108, 110},
       {113, 114, 116, 118, 120, 122, 124, 126, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 3
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 64},
       {66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94, 96},
       {98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126, 128},
       {130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158, 160},
       {162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190, 192},
       {194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222, 224},
       {226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254, 0}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x4240},
       {0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c, 0x6260},
       {0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c, 0x8280},
       {0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c, 0xa2a0},
       {0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc, 0xc2c0},
       {0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc, 0xe2e0},
       {0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc, 0x0000}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x4644'4240},
       {0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58, 0x6664'6260},
       {0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78, 0x8684'8280},
       {0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98, 0xa6a4'a2a0},
       {0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8, 0xc6c4'c2c0},
       {0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8, 0xe6e4'e2e0},
       {0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8, 0x0000'0000}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x8e8c'8a89'8684'8280},
       {0x9e9c'9a98'9694'9291, 0xaeac'aaa9'a6a4'a2a0},
       {0xbebc'bab8'b6b4'b2b1, 0xcecc'cac9'c6c4'c2c0},
       {0xdedc'dad8'd6d4'd2d1, 0xeeec'eae9'e6e4'e2e0},
       {0xfefc'faf8'f6f4'f2f1, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 74, 76, 78},
       {81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 105, 106, 108, 110},
       {113, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 137, 138, 140, 142},
       {145, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 169, 170, 172, 174},
       {177, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 201, 202, 204, 206},
       {209, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 233, 234, 236, 238},
       {241, 242, 244, 246, 248, 250, 252, 254, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 4
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/4,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/4,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 5
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/5,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/5,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 6
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0604, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/6,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/6,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 7
  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{2, 4, 6, 9, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0604, 0x0a09, 0x0e0c, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0e0c'0a09, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/7,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      ExecVslidedownvx,
      ExecMaskedVslidedownvx,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/7,
      /*regx1=*/8,
      /*skip=*/0);
}

[[gnu::naked]] void ExecVslideupvi() {
  asm("vslideup.vi  v8, v24, 0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVslideupvi() {
  asm("vslideup.vi  v8, v24, 0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVslideupvx() {
  asm("vslideup.vx  v8, v24, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVslideupvx() {
  asm("vslideup.vx  v8, v24, t0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVslideup) {
  // With slide offset equal zero, this is equivalent to Vmv.
  TestVectorInstruction(
      ExecVslideupvi,
      ExecMaskedVslideupvi,
      {{0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126},
       {128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158},
       {160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190},
       {192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222},
       {224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254}},
      {{0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc}},
      {{0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
      {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
       {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
      kVectorCalculationsSourceLegacy);

  // VLMUL = 0.
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x5555'5555'5555'5555, 0x0e0c'0a09'0604'0200},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/1,
      /*skip=*/1);
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 2, 4, 6, 9, 10, 12, 14},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 1
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {30, 32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x1e1c, 0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x1e1c'1a18, 0x2624'2220, 0x2e2c'2a29, 0x3634'3231},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x5555'5555'5555'5555, 0x0e0c'0a09'0604'0200},
       {0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/1,
      /*skip=*/1);
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 2, 4, 6, 9, 10, 12, 14},
       {17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
       {0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 2
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {30, 32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60},
       {62, 64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92},
       {94, 96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x1e1c, 0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38},
       {0x3e3c, 0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58},
       {0x5e5c, 0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x1e1c'1a18, 0x2624'2220, 0x2e2c'2a29, 0x3634'3231},
       {0x3e3c'3a38, 0x4644'4240, 0x4e4c'4a49, 0x5654'5251},
       {0x5e5c'5a58, 0x6664'6260, 0x6e6c'6a69, 0x7674'7271},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x5555'5555'5555'5555, 0x0e0c'0a09'0604'0200},
       {0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/1,
      /*skip=*/1);

  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 2, 4, 6, 9, 10, 12, 14},
       {17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 74, 76, 78},
       {81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 105, 106, 108, 110},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
       {0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 3
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {30, 32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60},
       {62, 64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92},
       {94, 96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124},
       {126, 128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156},
       {158, 160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188},
       {190, 192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220},
       {222, 224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x1e1c, 0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38},
       {0x3e3c, 0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58},
       {0x5e5c, 0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78},
       {0x7e7c, 0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98},
       {0x9e9c, 0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8},
       {0xbebc, 0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8},
       {0xdedc, 0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8}},
      {{0x5555'5555, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x1e1c'1a18, 0x2624'2220, 0x2e2c'2a29, 0x3634'3231},
       {0x3e3c'3a38, 0x4644'4240, 0x4e4c'4a49, 0x5654'5251},
       {0x5e5c'5a58, 0x6664'6260, 0x6e6c'6a69, 0x7674'7271},
       {0x7e7c'7a78, 0x8684'8280, 0x8e8c'8a89, 0x9694'9291},
       {0x9e9c'9a98, 0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1},
       {0xbebc'bab8, 0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1},
       {0xdedc'dad8, 0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1}},
      {{0x5555'5555'5555'5555, 0x0e0c'0a09'0604'0200},
       {0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x8e8c'8a89'8684'8280},
       {0x9e9c'9a98'9694'9291, 0xaeac'aaa9'a6a4'a2a0},
       {0xbebc'bab8'b6b4'b2b1, 0xcecc'cac9'c6c4'c2c0},
       {0xdedc'dad8'd6d4'd2d1, 0xeeec'eae9'e6e4'e2e0}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/1,
      /*skip=*/1);

  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 2, 4, 6, 9, 10, 12, 14},
       {17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 74, 76, 78},
       {81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 105, 106, 108, 110},
       {113, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 137, 138, 140, 142},
       {145, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 169, 170, 172, 174},
       {177, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 201, 202, 204, 206},
       {209, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 233, 234, 236, 238}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
       {0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc}},
      {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8}},
      {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 4
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/4,
      /*regx1=*/1,
      /*skip=*/1);

  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/4,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 5
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/5,
      /*regx1=*/1,
      /*skip=*/1);

  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/5,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 6
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x0200, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/6,
      /*regx1=*/1,
      /*skip=*/1);

  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 85, 85, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x5555, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/6,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 7
  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 0, 2, 4, 6, 9, 10, 12, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x0604'0200, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x5555'5555'5555'5555, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/7,
      /*regx1=*/1,
      /*skip=*/1);

  TestVectorPermutationInstruction(
      ExecVslideupvx,
      ExecMaskedVslideupvx,
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x5555'5555, 0x5555'5555, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x5555'5555'5555'5555, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/7,
      /*regx1=*/8,
      /*skip=*/8);
}

[[gnu::naked]] void ExecVfirstm() {
  asm("vfirst.m t0, v16\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVfirstm) {
  SIMD128Register source[16]{};
  SIMD128Register unused_result[8]{};
  // Setup 8-bit elements in 8 registers to involve all 128 bits in vl. Masks and tail processing
  // aren't used in this test.
  uint64_t vtype = 3;
  for (uint64_t bit = 0; bit < 128; bit++) {
    uint64_t result{~uint64_t{0}};
    source[0] = __uint128_t{1} << bit;
    RunCommonVectorFunc(
        &ExecVfirstm, &source[0], nullptr, &result, nullptr, 0, 0, vtype, /*vl*/ 128);
    EXPECT_EQ(result, bit);
  }

  // Also test the zero special case.
  uint64_t result{0};
  source[0] = __uint128_t{0};
  RunCommonVectorFunc(&ExecVfirstm, &source[0], nullptr, &result, nullptr, 0, 0, vtype, /*vl*/ 128);
  EXPECT_EQ(result, ~uint64_t{0});
}

}  // namespace

}  // namespace berberis
