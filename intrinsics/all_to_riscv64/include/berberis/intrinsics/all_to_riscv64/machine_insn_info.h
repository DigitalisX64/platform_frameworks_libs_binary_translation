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

#ifndef BERBERIS_INTRINSICS_COMMON_TO_RISCV_MACHINE_INSN_INFO_H_
#define BERBERIS_INTRINSICS_COMMON_TO_RISCV_MACHINE_INSN_INFO_H_

#include <cstdint>

#include "berberis/assembler/riscv.h"
#include "berberis/intrinsics/common/machine_insn_info.h"

namespace berberis::machine_insn_info {

class BImm {
 public:
  using Type = riscv::BImmediate;
  static constexpr bool kIsImmediate = true;
};

class CsrImm {
 public:
  using Type = riscv::CsrImmediate;
  static constexpr bool kIsImmediate = true;
};

class GeneralReg {
 public:
  using Type = uint64_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'r';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kGeneralReg;
};

class IImm {
 public:
  using Type = riscv::IImmediate;
  static constexpr bool kIsImmediate = true;
};

class JImm {
 public:
  using Type = riscv::JImmediate;
  static constexpr bool kIsImmediate = true;
};

class PImm {
 public:
  using Type = riscv::PImmediate;
  static constexpr bool kIsImmediate = true;
};

class SImm {
 public:
  using Type = riscv::SImmediate;
  static constexpr bool kIsImmediate = true;
};

class Shift32Imm {
 public:
  using Type = riscv::Shift32Immediate;
  static constexpr bool kIsImmediate = true;
};

class Shift64Imm {
 public:
  using Type = riscv::Shift64Immediate;
  static constexpr bool kIsImmediate = true;
};

class UImm {
 public:
  using Type = riscv::UImmediate;
  static constexpr bool kIsImmediate = true;
};

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoCPUIDRestriction;

}  // namespace berberis::machine_insn_info

#endif  // BERBERIS_INTRINSICS_COMMON_TO_RISCV_MACHINE_INSN_INFO_H_
