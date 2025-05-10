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

#ifndef BERBERIS_MACHINE_INSN_INFO_ALL_TO_X86_32_OR_x86_64_MACHINE_INSN_INFO_H_
#define BERBERIS_MACHINE_INSN_INFO_ALL_TO_X86_32_OR_x86_64_MACHINE_INSN_INFO_H_

#include <xmmintrin.h>

#include <cstdint>

#include "berberis/machine_insn_info/common/machine_insn_info.h"

namespace berberis::x86_32_or_x86_64::machine_insn_info_backend {

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

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoCPUIDRestriction;
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

}  // namespace berberis::x86_32_or_x86_64::machine_insn_info_backend

#endif  // BERBERIS_MACHINE_INSN_INFO_ALL_TO_X86_32_OR_x86_64_MACHINE_INSN_INFO_H_
