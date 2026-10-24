/*
 * Copyright (C) 2025 The Android Open Source Project
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

#ifndef BERBERIS_DEVICE_ARCH_INFO_ALL_TO_X86_32_OR_x86_64_DEVICE_ARCH_INFO_H_
#define BERBERIS_DEVICE_ARCH_INFO_ALL_TO_X86_32_OR_x86_64_DEVICE_ARCH_INFO_H_

#include <x86intrin.h>

#include <cstdint>

#include "berberis/assembler/x86_32_or_x86_64.h"
#include "berberis/device_arch_info/common/device_arch_info.h"

namespace berberis {

namespace x86_32_or_x86_64::device_arch_info {

// Note: normally using namespace is forbidden in headers, but these two namespaces literally
// only exist to be imported here (and in other device CPU-specific headers).

using namespace berberis::device_arch_info;

class Cond {
 public:
  using Type = Condition;
};

class Imm2 {
 public:
  using Type = int8_t;
};

class Imm8 {
 public:
  using Type = int8_t;
};

class Imm16 {
 public:
  using Type = int16_t;
};

class Imm32 {
 public:
  using Type = int32_t;
};

class Imm64 {
 public:
  using Type = int64_t;
};

class ImmPCode {
 public:
  using Type = const void*;
};

class Mem {
 public:
  // Mem can only be used as fake, non-memory Leal/Leaq argument, but having type here simplifies
  // metaprogramming: it can not be used as actual type of variable or parameter, but can be used
  // with std::conditional_t to pick some other type.
  using Type = void;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class MemX87 {
 public:
  // MemX87 can only be used as temporary argument, but having type here simplifies metaprogramming:
  // it can not be used as actual type of variable or parameter, but can be used with
  // std::conditional_t to pick some other type.
  using Type = void;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class VecMem32 {
 public:
  using Type = uint32_t;
  static constexpr char kAsRegister = 'm';
};

class VecMem64 {
 public:
  using Type = uint64_t;
  static constexpr char kAsRegister = 'm';
};

class VecMem128 {
 public:
  using Type = __m128;
  static constexpr char kAsRegister = 'm';
};

class VecMem256 {
 public:
#ifdef __AVX__
  using Type = __m256;
#endif
  static constexpr char kAsRegister = 'm';
};

// We don't currently have use-cases where instructions that use these register classes can be used
// with MachineIR.
// We would need to make this classes “real” to be able to do that, but also would probably need
// other changes.
class CC;
class GeneralReg;
class Label;
class MemX8716;
class MemX8732;
class MemX8764;
class MemX8780;
class RSP;
class RegX87;
class SW;
class ST;
class ST1;

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class Has3DNOW;
class Has3DNOWP;
class HasADX;
class HasAES;
class HasAESAVX;
class HasAMXBF16;
class HasAMXFP16;
class HasAMXINT8;
class HasAMXTILE;
class HasAVX;
class HasAVX2;
class HasAVX5124FMAPS;
class HasAVX5124VNNIW;
class HasAVX512BF16;
class HasAVX512BITALG;
class HasAVX512BW;
class HasAVX512CD;
class HasAVX512DQ;
class HasAVX512ER;
class HasAVX512F;
class HasAVX512FP16;
class HasAVX512IFMA;
class HasAVX512PF;
class HasAVX512VBMI;
class HasAVX512VBMI2;
class HasAVX512VL;
class HasAVX512VNNI;
class HasAVX512VPOPCNTDQ;
class HasBMI;
class HasBMI2;
class HasCLMUL;
class HasCLMULAVX;
class HasCMOV;
class HasCMPXCHG16B;
class HasCMPXCHG8B;
class HasF16C;
class HasFMA;
class HasFMA4;
class HasFXSAVE;
class HasLZCNT;
// BMI2 is set and PDEP/PEXT are ok to use. See more here:
//   https://twitter.com/instlatx64/status/1322503571288559617
class HashPDEP;
class HasPOPCNT;
class HasRDSEED;
class HasSERIALIZE;
class HasSHA;
class HasSSE;
class HasSSE2;
class HasSSE3;
class HasSSE4_1;
class HasSSE4_2;
class HasSSE4a;
class HasSSSE3;
class HasTBM;
class HasVAES;
class HasVPCLMULQD;
class HasX87;
class HasCustomCapability;
class IsAuthenticAMD;

}  // namespace x86_32_or_x86_64::device_arch_info

namespace device_arch_info {

template <>
inline constexpr bool kIsCondition<x86_32_or_x86_64::device_arch_info::Cond> = true;

template <>
inline constexpr bool kIsImmediate<x86_32_or_x86_64::device_arch_info::Imm2> = true;

template <>
inline constexpr bool kIsImmediate<x86_32_or_x86_64::device_arch_info::Imm8> = true;

template <>
inline constexpr bool kIsImmediate<x86_32_or_x86_64::device_arch_info::Imm16> = true;

template <>
inline constexpr bool kIsImmediate<x86_32_or_x86_64::device_arch_info::Imm32> = true;

template <>
inline constexpr bool kIsImmediate<x86_32_or_x86_64::device_arch_info::Imm64> = true;

template <>
inline constexpr bool kIsImmediate<x86_32_or_x86_64::device_arch_info::ImmPCode> = true;

template <>
inline constexpr bool kIsMemoryOperand<x86_32_or_x86_64::device_arch_info::Mem> = true;

template <>
inline constexpr bool kIsMemoryOperand<x86_32_or_x86_64::device_arch_info::MemX87> = true;

template <>
inline constexpr bool kIsMemoryOperand<x86_32_or_x86_64::device_arch_info::VecMem32> = true;

template <>
inline constexpr bool kIsMemoryOperand<x86_32_or_x86_64::device_arch_info::VecMem64> = true;

template <>
inline constexpr bool kIsMemoryOperand<x86_32_or_x86_64::device_arch_info::VecMem128> = true;

template <>
inline constexpr bool kIsMemoryOperand<x86_32_or_x86_64::device_arch_info::VecMem256> = true;

}  // namespace device_arch_info

}  // namespace berberis

#endif  // BERBERIS_DEVICE_ARCH_INFO_ALL_TO_X86_32_OR_x86_64_DEVICE_ARCH_INFO_H_
