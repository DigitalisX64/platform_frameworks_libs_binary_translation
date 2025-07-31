/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include "berberis/intrinsics/constants_pool.h"

#include <inttypes.h>
#include <sys/mman.h>

#include "berberis/base/bit_util.h"
#include "berberis/base/mmap.h"
#include "berberis/base/struct_check.h"
#include "berberis/intrinsics/simd_register.h"

namespace berberis::constants_pool {

// All constants we refer in macroinstructions are collected in MacroAssemblerConstants.
// This allows us:
//   1. To save memory (they are not duplicated when reused).
//   2. Make it possible to access in text assembler without complex dance with hash-tables.
//   3. Allocate below-2GB-copy in x86_64 mode easily.
struct MacroAssemblerConstants {
  alignas(16) const uint64_t kNanBoxFloat32[2] = {0xffff'ffff'0000'0000, 0xffff'ffff'0000'0000};
  alignas(16) const uint64_t kNanBoxedNansFloat32[2] = {0xffff'ffff'7fc0'0000,
                                                        0xffff'ffff'7fc0'0000};
  alignas(16) const uint32_t kCanonicalNansFloat32[4] = {0x7fc0'0000,
                                                         0x7fc0'0000,
                                                         0x7fc0'0000,
                                                         0x7fc0'0000};
  alignas(16) const uint64_t kCanonicalNansFloat64[2] = {0x7ff8'0000'0000'0000,
                                                         0x7ff8'0000'0000'0000};
  alignas(16) const uint32_t kFloat32One[4] = {0x3f80'0000, 0x3f80'0000, 0x3f80'0000, 0x3f80'0000};
  alignas(16) const uint64_t kFloat64One[2] = {0x3ff0'0000'0000'0000, 0x3ff0'0000'0000'0000};
  alignas(16) const uint32_t kFloat32PInf[4] = {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000};
  alignas(16) const uint32_t kFloat32NInf[4] = {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000};
  alignas(16) const uint64_t kFloat64PInf[2] = {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000};
  alignas(16) const uint64_t kFloat64NInf[2] = {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000};
  alignas(16) const int8_t kMinInt8[16] = {
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
      -128,
  };
  alignas(16) const int8_t kMaxInt8[16] =
      {127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127};
  alignas(16) const int16_t kMinInt16[8] =
      {-0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000};
  alignas(16)
      const int16_t kMaxInt16[8] = {0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff};
  alignas(16) const int32_t kMinInt32[4] = {
      static_cast<int32_t>(-0x8000'0000),
      static_cast<int32_t>(-0x8000'0000),
      static_cast<int32_t>(-0x8000'0000),
      static_cast<int32_t>(-0x8000'0000),
  };
  alignas(16) const int32_t kMaxInt32[4] = {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff};
  alignas(16) const int64_t kMinInt64[2] = {
      static_cast<int64_t>(-0x8000'0000'0000'0000),
      static_cast<int64_t>(-0x8000'0000'0000'0000),
  };
  alignas(16) const int64_t kMaxInt64[2] = {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff};
  int64_t kBsrToClzInt64 = 127;
  int64_t kWidthInBits64 = 64;
  int32_t kBsrToClzInt32 = 63;
  int32_t kWidthInBits32 = 32;
  // 64 bit constants for use with arithmetic operations.
  // Used because only 32 bit immediates are supported on x86-64.
  int64_t k0x8000_0000_0000_00ff = 0x8000'0000'0000'00ff;
  alignas(16) const int8_t kPMovmskwToPMovmskb[16] =
      {0, 2, 4, 6, 8, 10, 12, 14, -63, -24, -19, -27, -28, -128, -128, -128};
  alignas(16) const int8_t kPMovmskdToPMovmskb[16] =
      {0, 4, 8, 12, -128, -128, -128, -128, -51, -17, -24, -31, -19, -27, -28, -128};
  alignas(16) const int8_t kPMovmskqToPMovmskb[16] =
      {0, 8, -128, -128, -128, -128, -128, -128, -57, -24, -31, -6, -7, -128, -128, -128};
  alignas(16) const uint8_t kRiscVToX87Exceptions[32] = {
      0x00, 0x20, 0x10, 0x30, 0x08, 0x28, 0x18, 0x38, 0x04, 0x24, 0x14,
      0x34, 0x0c, 0x2c, 0x1c, 0x3c, 0x01, 0x21, 0x11, 0x31, 0x09, 0x29,
      0x19, 0x39, 0x05, 0x25, 0x15, 0x35, 0x0d, 0x2d, 0x1d, 0x3d};
  alignas(16) const uint8_t kX87ToRiscVExceptions[64] = {
      0x00, 0x10, 0x00, 0x10, 0x08, 0x18, 0x08, 0x18, 0x04, 0x14, 0x04, 0x14, 0x0c,
      0x1c, 0x0c, 0x1c, 0x02, 0x12, 0x02, 0x12, 0x0a, 0x1a, 0x0a, 0x1a, 0x06, 0x16,
      0x06, 0x16, 0x0e, 0x1e, 0x0e, 0x1e, 0x01, 0x11, 0x01, 0x11, 0x09, 0x19, 0x09,
      0x19, 0x05, 0x15, 0x05, 0x15, 0x0d, 0x1d, 0x0d, 0x1d, 0x03, 0x13, 0x03, 0x13,
      0x0b, 0x1b, 0x0b, 0x1b, 0x07, 0x17, 0x07, 0x17, 0x0f, 0x1f, 0x0f, 0x1f};
  // This table represents exactly what you see: 𝟏𝟐𝟖 + 𝐍 unset bits and then 𝟏𝟐𝟖 - 𝐍 set bits for 𝐍
  // in range from 𝟎 to 𝟕.  The last 𝟏𝟐𝟖 bits from line 𝐍 then it's mask for 𝐯𝐥 equal to 𝐍 and if
  // you shift start address down by 𝐌  bytes then you get mask for 𝟖 * 𝐌 + 𝐍 bits.
  // Using this approach we may load appropriate bitmask from memory with one load instruction and
  // the table itself takes 𝟐𝟓𝟔 bytes.
  // Since valid values for 𝐯𝐥 are from 𝟎 to 𝐯𝐥 𝟎 𝟏𝟐𝟖 (including 𝟏𝟐𝟖), 𝐌 may be from 𝟎 to 𝟏𝟔,
  // that's why we need 16 zero bytes in that table.
  // On AMD CPUs with misalignsse feature we may access data from that table without using movups,
  // and on all CPUs we may use 2KiB table instead.
  // But for now we are using movups and small table as probably-adequate compromise.
  alignas(16) const uint64_t kBitMaskTable[8][4] = {
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'fffe, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'fffc, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'fff8, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'fff0, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffe0, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffc0, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'ff80, 0xffff'ffff'ffff'ffff},
  };
  // RISC-V manual strongly implies that vid.v may be implemented similarly to viota.m
  // This may be true for hardware implementation, but in software vid.v may be implemented with a
  // simple precomputed table which implementation of viota.m is much more tricky and slow.
  // Here are precomputed values for Vid.v
  alignas(16) Int64x2 kVid64Bit[8] = {
      {0, 1},
      {2, 3},
      {4, 5},
      {6, 7},
      {8, 9},
      {10, 11},
      {12, 13},
      {14, 15},
  };
  alignas(16) Int32x4 kVid32Bit[8] = {
      {0, 1, 2, 3},
      {4, 5, 6, 7},
      {8, 9, 10, 11},
      {12, 13, 14, 15},
      {16, 17, 18, 19},
      {20, 21, 22, 23},
      {24, 25, 26, 27},
      {28, 29, 30, 31},
  };
  alignas(16) Int16x8 kVid16Bit[8] = {
      {0, 1, 2, 3, 4, 5, 6, 7},
      {8, 9, 10, 11, 12, 13, 14, 15},
      {16, 17, 18, 19, 20, 21, 22, 23},
      {24, 25, 26, 27, 28, 29, 30, 31},
      {32, 33, 34, 35, 36, 37, 38, 39},
      {40, 41, 42, 43, 44, 45, 46, 47},
      {48, 49, 50, 51, 52, 53, 54, 55},
      {56, 57, 58, 59, 60, 61, 62, 63},
  };
  alignas(16) Int8x16 kVid8Bit[8] = {
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
      {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
      {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
      {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79},
      {80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95},
      {96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111},
      {112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127},
  };
  alignas(16) const uint64_t kBitMaskTo32bitMask[4] = {
      0x0000'0000'0000'0000,
      0x0000'0000'ffff'ffff,
      0xffff'ffff'0000'0000,
      0xffff'ffff'ffff'ffff,
  };
  alignas(16) const uint64_t kBitMaskTo16bitMask[16] = {
      0x0000'0000'0000'0000,
      0x0000'0000'0000'ffff,
      0x0000'0000'ffff'0000,
      0x0000'0000'ffff'ffff,
      0x0000'ffff'0000'0000,
      0x0000'ffff'0000'ffff,
      0x0000'ffff'ffff'0000,
      0x0000'ffff'ffff'ffff,
      0xffff'0000'0000'0000,
      0xffff'0000'0000'ffff,
      0xffff'0000'ffff'0000,
      0xffff'0000'ffff'ffff,
      0xffff'ffff'0000'0000,
      0xffff'ffff'0000'ffff,
      0xffff'ffff'ffff'0000,
      0xffff'ffff'ffff'ffff,
  };
  alignas(16) const uint64_t kBitMaskTo8bitMask[256] = {
      0x0000'0000'0000'0000, 0x0000'0000'0000'00ff, 0x0000'0000'0000'ff00, 0x0000'0000'0000'ffff,
      0x0000'0000'00ff'0000, 0x0000'0000'00ff'00ff, 0x0000'0000'00ff'ff00, 0x0000'0000'00ff'ffff,
      0x0000'0000'ff00'0000, 0x0000'0000'ff00'00ff, 0x0000'0000'ff00'ff00, 0x0000'0000'ff00'ffff,
      0x0000'0000'ffff'0000, 0x0000'0000'ffff'00ff, 0x0000'0000'ffff'ff00, 0x0000'0000'ffff'ffff,
      0x0000'00ff'0000'0000, 0x0000'00ff'0000'00ff, 0x0000'00ff'0000'ff00, 0x0000'00ff'0000'ffff,
      0x0000'00ff'00ff'0000, 0x0000'00ff'00ff'00ff, 0x0000'00ff'00ff'ff00, 0x0000'00ff'00ff'ffff,
      0x0000'00ff'ff00'0000, 0x0000'00ff'ff00'00ff, 0x0000'00ff'ff00'ff00, 0x0000'00ff'ff00'ffff,
      0x0000'00ff'ffff'0000, 0x0000'00ff'ffff'00ff, 0x0000'00ff'ffff'ff00, 0x0000'00ff'ffff'ffff,
      0x0000'ff00'0000'0000, 0x0000'ff00'0000'00ff, 0x0000'ff00'0000'ff00, 0x0000'ff00'0000'ffff,
      0x0000'ff00'00ff'0000, 0x0000'ff00'00ff'00ff, 0x0000'ff00'00ff'ff00, 0x0000'ff00'00ff'ffff,
      0x0000'ff00'ff00'0000, 0x0000'ff00'ff00'00ff, 0x0000'ff00'ff00'ff00, 0x0000'ff00'ff00'ffff,
      0x0000'ff00'ffff'0000, 0x0000'ff00'ffff'00ff, 0x0000'ff00'ffff'ff00, 0x0000'ff00'ffff'ffff,
      0x0000'ffff'0000'0000, 0x0000'ffff'0000'00ff, 0x0000'ffff'0000'ff00, 0x0000'ffff'0000'ffff,
      0x0000'ffff'00ff'0000, 0x0000'ffff'00ff'00ff, 0x0000'ffff'00ff'ff00, 0x0000'ffff'00ff'ffff,
      0x0000'ffff'ff00'0000, 0x0000'ffff'ff00'00ff, 0x0000'ffff'ff00'ff00, 0x0000'ffff'ff00'ffff,
      0x0000'ffff'ffff'0000, 0x0000'ffff'ffff'00ff, 0x0000'ffff'ffff'ff00, 0x0000'ffff'ffff'ffff,
      0x00ff'0000'0000'0000, 0x00ff'0000'0000'00ff, 0x00ff'0000'0000'ff00, 0x00ff'0000'0000'ffff,
      0x00ff'0000'00ff'0000, 0x00ff'0000'00ff'00ff, 0x00ff'0000'00ff'ff00, 0x00ff'0000'00ff'ffff,
      0x00ff'0000'ff00'0000, 0x00ff'0000'ff00'00ff, 0x00ff'0000'ff00'ff00, 0x00ff'0000'ff00'ffff,
      0x00ff'0000'ffff'0000, 0x00ff'0000'ffff'00ff, 0x00ff'0000'ffff'ff00, 0x00ff'0000'ffff'ffff,
      0x00ff'00ff'0000'0000, 0x00ff'00ff'0000'00ff, 0x00ff'00ff'0000'ff00, 0x00ff'00ff'0000'ffff,
      0x00ff'00ff'00ff'0000, 0x00ff'00ff'00ff'00ff, 0x00ff'00ff'00ff'ff00, 0x00ff'00ff'00ff'ffff,
      0x00ff'00ff'ff00'0000, 0x00ff'00ff'ff00'00ff, 0x00ff'00ff'ff00'ff00, 0x00ff'00ff'ff00'ffff,
      0x00ff'00ff'ffff'0000, 0x00ff'00ff'ffff'00ff, 0x00ff'00ff'ffff'ff00, 0x00ff'00ff'ffff'ffff,
      0x00ff'ff00'0000'0000, 0x00ff'ff00'0000'00ff, 0x00ff'ff00'0000'ff00, 0x00ff'ff00'0000'ffff,
      0x00ff'ff00'00ff'0000, 0x00ff'ff00'00ff'00ff, 0x00ff'ff00'00ff'ff00, 0x00ff'ff00'00ff'ffff,
      0x00ff'ff00'ff00'0000, 0x00ff'ff00'ff00'00ff, 0x00ff'ff00'ff00'ff00, 0x00ff'ff00'ff00'ffff,
      0x00ff'ff00'ffff'0000, 0x00ff'ff00'ffff'00ff, 0x00ff'ff00'ffff'ff00, 0x00ff'ff00'ffff'ffff,
      0x00ff'ffff'0000'0000, 0x00ff'ffff'0000'00ff, 0x00ff'ffff'0000'ff00, 0x00ff'ffff'0000'ffff,
      0x00ff'ffff'00ff'0000, 0x00ff'ffff'00ff'00ff, 0x00ff'ffff'00ff'ff00, 0x00ff'ffff'00ff'ffff,
      0x00ff'ffff'ff00'0000, 0x00ff'ffff'ff00'00ff, 0x00ff'ffff'ff00'ff00, 0x00ff'ffff'ff00'ffff,
      0x00ff'ffff'ffff'0000, 0x00ff'ffff'ffff'00ff, 0x00ff'ffff'ffff'ff00, 0x00ff'ffff'ffff'ffff,
      0xff00'0000'0000'0000, 0xff00'0000'0000'00ff, 0xff00'0000'0000'ff00, 0xff00'0000'0000'ffff,
      0xff00'0000'00ff'0000, 0xff00'0000'00ff'00ff, 0xff00'0000'00ff'ff00, 0xff00'0000'00ff'ffff,
      0xff00'0000'ff00'0000, 0xff00'0000'ff00'00ff, 0xff00'0000'ff00'ff00, 0xff00'0000'ff00'ffff,
      0xff00'0000'ffff'0000, 0xff00'0000'ffff'00ff, 0xff00'0000'ffff'ff00, 0xff00'0000'ffff'ffff,
      0xff00'00ff'0000'0000, 0xff00'00ff'0000'00ff, 0xff00'00ff'0000'ff00, 0xff00'00ff'0000'ffff,
      0xff00'00ff'00ff'0000, 0xff00'00ff'00ff'00ff, 0xff00'00ff'00ff'ff00, 0xff00'00ff'00ff'ffff,
      0xff00'00ff'ff00'0000, 0xff00'00ff'ff00'00ff, 0xff00'00ff'ff00'ff00, 0xff00'00ff'ff00'ffff,
      0xff00'00ff'ffff'0000, 0xff00'00ff'ffff'00ff, 0xff00'00ff'ffff'ff00, 0xff00'00ff'ffff'ffff,
      0xff00'ff00'0000'0000, 0xff00'ff00'0000'00ff, 0xff00'ff00'0000'ff00, 0xff00'ff00'0000'ffff,
      0xff00'ff00'00ff'0000, 0xff00'ff00'00ff'00ff, 0xff00'ff00'00ff'ff00, 0xff00'ff00'00ff'ffff,
      0xff00'ff00'ff00'0000, 0xff00'ff00'ff00'00ff, 0xff00'ff00'ff00'ff00, 0xff00'ff00'ff00'ffff,
      0xff00'ff00'ffff'0000, 0xff00'ff00'ffff'00ff, 0xff00'ff00'ffff'ff00, 0xff00'ff00'ffff'ffff,
      0xff00'ffff'0000'0000, 0xff00'ffff'0000'00ff, 0xff00'ffff'0000'ff00, 0xff00'ffff'0000'ffff,
      0xff00'ffff'00ff'0000, 0xff00'ffff'00ff'00ff, 0xff00'ffff'00ff'ff00, 0xff00'ffff'00ff'ffff,
      0xff00'ffff'ff00'0000, 0xff00'ffff'ff00'00ff, 0xff00'ffff'ff00'ff00, 0xff00'ffff'ff00'ffff,
      0xff00'ffff'ffff'0000, 0xff00'ffff'ffff'00ff, 0xff00'ffff'ffff'ff00, 0xff00'ffff'ffff'ffff,
      0xffff'0000'0000'0000, 0xffff'0000'0000'00ff, 0xffff'0000'0000'ff00, 0xffff'0000'0000'ffff,
      0xffff'0000'00ff'0000, 0xffff'0000'00ff'00ff, 0xffff'0000'00ff'ff00, 0xffff'0000'00ff'ffff,
      0xffff'0000'ff00'0000, 0xffff'0000'ff00'00ff, 0xffff'0000'ff00'ff00, 0xffff'0000'ff00'ffff,
      0xffff'0000'ffff'0000, 0xffff'0000'ffff'00ff, 0xffff'0000'ffff'ff00, 0xffff'0000'ffff'ffff,
      0xffff'00ff'0000'0000, 0xffff'00ff'0000'00ff, 0xffff'00ff'0000'ff00, 0xffff'00ff'0000'ffff,
      0xffff'00ff'00ff'0000, 0xffff'00ff'00ff'00ff, 0xffff'00ff'00ff'ff00, 0xffff'00ff'00ff'ffff,
      0xffff'00ff'ff00'0000, 0xffff'00ff'ff00'00ff, 0xffff'00ff'ff00'ff00, 0xffff'00ff'ff00'ffff,
      0xffff'00ff'ffff'0000, 0xffff'00ff'ffff'00ff, 0xffff'00ff'ffff'ff00, 0xffff'00ff'ffff'ffff,
      0xffff'ff00'0000'0000, 0xffff'ff00'0000'00ff, 0xffff'ff00'0000'ff00, 0xffff'ff00'0000'ffff,
      0xffff'ff00'00ff'0000, 0xffff'ff00'00ff'00ff, 0xffff'ff00'00ff'ff00, 0xffff'ff00'00ff'ffff,
      0xffff'ff00'ff00'0000, 0xffff'ff00'ff00'00ff, 0xffff'ff00'ff00'ff00, 0xffff'ff00'ff00'ffff,
      0xffff'ff00'ffff'0000, 0xffff'ff00'ffff'00ff, 0xffff'ff00'ffff'ff00, 0xffff'ff00'ffff'ffff,
      0xffff'ffff'0000'0000, 0xffff'ffff'0000'00ff, 0xffff'ffff'0000'ff00, 0xffff'ffff'0000'ffff,
      0xffff'ffff'00ff'0000, 0xffff'ffff'00ff'00ff, 0xffff'ffff'00ff'ff00, 0xffff'ffff'00ff'ffff,
      0xffff'ffff'ff00'0000, 0xffff'ffff'ff00'00ff, 0xffff'ffff'ff00'ff00, 0xffff'ffff'ff00'ffff,
      0xffff'ffff'ffff'0000, 0xffff'ffff'ffff'00ff, 0xffff'ffff'ffff'ff00, 0xffff'ffff'ffff'ffff,
  };
};

// Make sure Layout is the same in 32-bit mode and 64-bit mode.
CHECK_STRUCT_LAYOUT(MacroAssemblerConstants, 27520, 128);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kNanBoxFloat32, 0, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kNanBoxedNansFloat32,
                            kNanBoxFloat32,
                            128,
                            128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kCanonicalNansFloat32,
                            kNanBoxedNansFloat32,
                            128,
                            128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kCanonicalNansFloat64,
                            kCanonicalNansFloat32,
                            128,
                            128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kFloat32One, kCanonicalNansFloat64, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kFloat64One, kFloat32One, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kFloat32PInf, kFloat64One, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kFloat32NInf, kFloat32PInf, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kFloat64PInf, kFloat32NInf, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kFloat64NInf, kFloat64PInf, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kMinInt8, kFloat64NInf, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kMaxInt8, kMinInt8, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kMinInt16, kMaxInt8, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kMaxInt16, kMinInt16, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kMinInt32, kMaxInt16, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kMaxInt32, kMinInt32, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kMinInt64, kMaxInt32, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kMaxInt64, kMinInt64, 128, 128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kBsrToClzInt64, kMaxInt64, 128, 64);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kWidthInBits64, kBsrToClzInt64, 64, 64);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kBsrToClzInt32, kWidthInBits64, 64, 32);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kWidthInBits32, kBsrToClzInt32, 32, 32);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            k0x8000_0000_0000_00ff,
                            kWidthInBits32,
                            32,
                            64);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kPMovmskwToPMovmskb,
                            k0x8000_0000_0000_00ff,
                            64,
                            128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kPMovmskdToPMovmskb,
                            kPMovmskwToPMovmskb,
                            128,
                            128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kPMovmskqToPMovmskb,
                            kPMovmskdToPMovmskb,
                            128,
                            128);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kRiscVToX87Exceptions,
                            kPMovmskqToPMovmskb,
                            128,
                            256);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kX87ToRiscVExceptions,
                            kRiscVToX87Exceptions,
                            256,
                            512);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kBitMaskTable,
                            kX87ToRiscVExceptions,
                            512,
                            2048);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kVid64Bit, kBitMaskTable, 2048, 1024);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kVid32Bit, kVid64Bit, 1024, 1024);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kVid16Bit, kVid32Bit, 1024, 1024);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kVid8Bit, kVid16Bit, 1024, 1024);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants, kBitMaskTo32bitMask, kVid8Bit, 1024, 256);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kBitMaskTo16bitMask,
                            kBitMaskTo32bitMask,
                            256,
                            1024);
CHECK_FIELD_LAYOUT_RELATIVE(MacroAssemblerConstants,
                            kBitMaskTo8bitMask,
                            kBitMaskTo16bitMask,
                            1024,
                            16384);

// Note: because we have aligned fields and thus padding in that data structure
// value-initialization is both slower and larger than copy-initialization for
// that structure.
//
// Also assembler intrinsics for interpreter use kMacroAssemblerConstants
// directly (they couldn't use consts below because these addresses are only
// known during runtime)
extern const MacroAssemblerConstants kBerberisMacroAssemblerConstants
    __attribute__((visibility("hidden")));
const MacroAssemblerConstants kBerberisMacroAssemblerConstants;

namespace {

int32_t GetConstants() {
  static const MacroAssemblerConstants* Constants =
      new (mmap(nullptr,
                AlignUpPageSize(sizeof(MacroAssemblerConstants)),
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT,
                -1,
                0)) MacroAssemblerConstants(kBerberisMacroAssemblerConstants);
  // Note that we are returning only 32-bit address here, but it's guaranteed to
  // be enough since struct is in low memory because of MAP_32BIT flag.
  return bit_cast<intptr_t>(Constants);
}

}  // namespace

extern const int32_t kBerberisMacroAssemblerConstantsRelocated;
const int32_t kBerberisMacroAssemblerConstantsRelocated = GetConstants();

#pragma push_macro("VECTOR_CONST_EXTERN")
#define VECTOR_CONST_EXTERN(Value, Name, Offset) \
  const int32_t VectorConst<Value>::kValue =     \
      GetConstants() + offsetof(MacroAssemblerConstants, Name) + Offset;

VECTOR_CONST_EXTERN(int8_t{-128}, kMinInt8, 0);
VECTOR_CONST_EXTERN(int8_t{127}, kMaxInt8, 0);
VECTOR_CONST_EXTERN(int16_t{-0x8000}, kMinInt16, 0);
VECTOR_CONST_EXTERN(int16_t{0x7fff}, kMaxInt16, 0);
VECTOR_CONST_EXTERN(int32_t{static_cast<int32_t>(-0x8000'0000)}, kMinInt32, 0);
VECTOR_CONST_EXTERN(int32_t{0x3f80'0000}, kFloat32One, 0);
VECTOR_CONST_EXTERN(int32_t{0x7f80'0000}, kFloat32PInf, 0);
VECTOR_CONST_EXTERN(int32_t{0x7fff'ffff}, kMaxInt32, 0);
VECTOR_CONST_EXTERN(int32_t{-0x0080'0000}, kFloat32NInf, 0);
VECTOR_CONST_EXTERN(int64_t{static_cast<int64_t>(-0x8000'0000'0000'0000)}, kMinInt64, 0);
VECTOR_CONST_EXTERN(int64_t{0x3ff0'0000'0000'0000}, kFloat64One, 0);
VECTOR_CONST_EXTERN(int64_t{0x7ff0'0000'0000'0000}, kFloat64PInf, 0);
VECTOR_CONST_EXTERN(int64_t{0x7fff'ffff'ffff'ffff}, kMaxInt64, 0);
VECTOR_CONST_EXTERN(int64_t{-0x0010'0000'0000'0000}, kFloat64NInf, 0);
VECTOR_CONST_EXTERN(uint64_t{0x0000'0000'0000'0000}, kBitMaskTable, 0);
VECTOR_CONST_EXTERN(uint64_t{0xffff'ffff'ffff'ffff}, kBitMaskTable, 16);
VECTOR_CONST_EXTERN(uint64_t{0xffff'ffff'0000'0000}, kNanBoxFloat32, 0);
VECTOR_CONST_EXTERN(uint64_t{0xffff'ffff'7fc0'0000}, kNanBoxedNansFloat32, 0);
VECTOR_CONST_EXTERN(uint64_t{0x7fc0'0000'7fc0'0000}, kCanonicalNansFloat32, 0);
VECTOR_CONST_EXTERN(uint64_t{0x7ff8'0000'0000'0000}, kCanonicalNansFloat64, 0);

#pragma pop_macro("VECTOR_CONST_EXTERN")

#pragma push_macro("CONST_EXTERN")
#define CONST_EXTERN(Value, Name, Offset) \
  const int32_t Const<Value>::kValue =    \
      GetConstants() + offsetof(MacroAssemblerConstants, Name) + Offset;

CONST_EXTERN(uint64_t{127}, kBsrToClzInt64, 0);
CONST_EXTERN(uint64_t{64}, kWidthInBits64, 0);
CONST_EXTERN(uint32_t{63}, kBsrToClzInt32, 0);
CONST_EXTERN(uint32_t{32}, kWidthInBits32, 0);
CONST_EXTERN(uint64_t{0x8000'0000'0000'00ff}, k0x8000_0000_0000_00ff, 0);

#pragma pop_macro("CONST_EXTERN")

const int32_t kRiscVToX87Exceptions =
    GetConstants() + offsetof(MacroAssemblerConstants, kRiscVToX87Exceptions);
const int32_t kX87ToRiscVExceptions =
    GetConstants() + offsetof(MacroAssemblerConstants, kX87ToRiscVExceptions);
const int32_t kBitMaskTable = GetConstants() + offsetof(MacroAssemblerConstants, kBitMaskTable);
const int32_t kVid64Bit = GetConstants() + offsetof(MacroAssemblerConstants, kVid64Bit);
const int32_t kVid32Bit = GetConstants() + offsetof(MacroAssemblerConstants, kVid32Bit);
const int32_t kVid16Bit = GetConstants() + offsetof(MacroAssemblerConstants, kVid16Bit);
const int32_t kVid8Bit = GetConstants() + offsetof(MacroAssemblerConstants, kVid8Bit);
const int32_t kBitMaskTo32bitMask =
    GetConstants() + offsetof(MacroAssemblerConstants, kBitMaskTo32bitMask);
const int32_t kBitMaskTo16bitMask =
    GetConstants() + offsetof(MacroAssemblerConstants, kBitMaskTo16bitMask);
const int32_t kBitMaskTo8bitMask =
    GetConstants() + offsetof(MacroAssemblerConstants, kBitMaskTo8bitMask);
const int32_t kPMovmskwToPMovmskb =
    GetConstants() + offsetof(MacroAssemblerConstants, kPMovmskwToPMovmskb);
const int32_t kPMovmskdToPMovmskb =
    GetConstants() + offsetof(MacroAssemblerConstants, kPMovmskdToPMovmskb);
const int32_t kPMovmskqToPMovmskb =
    GetConstants() + offsetof(MacroAssemblerConstants, kPMovmskqToPMovmskb);

}  // namespace berberis::constants_pool
