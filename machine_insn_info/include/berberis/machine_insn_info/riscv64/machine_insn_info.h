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

#ifndef BERBERIS_MACHINE_INSN_INFO_RISCV64_MACHINE_INSN_INFO_H_
#define BERBERIS_MACHINE_INSN_INFO_RISCV64_MACHINE_INSN_INFO_H_

#include <cstdint>

#include "berberis/assembler/riscv.h"
#include "berberis/machine_insn_info/common/machine_insn_info.h"

namespace berberis::riscv64::machine_insn_info_backend {

// Note: normally using namespace is forbidden in headers, but these two namespaces literally
// only exist to be imported here (and in other device CPU-specific headers).

using namespace berberis::machine_insn_info_backend;

class BImm {
 public:
  using Type = riscv::BImmediate;
};

class CsrImm {
 public:
  using Type = riscv::CsrImmediate;
};

class IImm {
 public:
  using Type = riscv::IImmediate;
};

class JImm {
 public:
  using Type = riscv::JImmediate;
};

class PImm {
 public:
  using Type = riscv::PImmediate;
};

class SImm {
 public:
  using Type = riscv::SImmediate;
};

class Shift32Imm {
 public:
  using Type = riscv::Shift32Immediate;
};

class Shift64Imm {
 public:
  using Type = riscv::Shift64Immediate;
};

class UImm {
 public:
  using Type = riscv::UImmediate;
};

}  // namespace berberis::riscv64::machine_insn_info_backend

#endif  // BERBERIS_MACHINE_INSN_INFO_RISCV64_MACHINE_INSN_INFO_H_
