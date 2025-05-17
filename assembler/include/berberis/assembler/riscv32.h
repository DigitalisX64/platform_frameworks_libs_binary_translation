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

// Assembler to produce RV32 instructions (no ABI version). Somewhat influenced by V8 assembler.

#ifndef BERBERIS_ASSEMBLER_RISCV32_H_
#define BERBERIS_ASSEMBLER_RISCV32_H_

#include <type_traits>  // std::is_same

#include "berberis/assembler/riscv.h"

namespace berberis::riscv32 {

class Assembler : public riscv::Assembler<Assembler> {
 public:
  using BaseAssembler = riscv::Assembler<Assembler>;
  using FinalAssembler = Assembler;

  explicit Assembler(MachineCode* code) : BaseAssembler(code) {}

  using ShiftImmediate = BaseAssembler::Shift32Immediate;

  // Don't use templates here to enable implicit conversions.
#define BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(IntType)                                \
  static constexpr std::optional<ShiftImmediate> MakeShiftImmediate(IntType value) { \
    return BaseAssembler::MakeShift32Immediate(value);                               \
  }
  BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(int8_t)
  BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(uint8_t)
  BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(int16_t)
  BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(uint16_t)
  BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(int32_t)
  BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(uint32_t)
  BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(int64_t)
  BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE(uint64_t)
#undef BERBERIS_DEFINE_MAKE_SHIFT_IMMEDIATE

  friend BaseAssembler;

// Instructions.
#include "berberis/assembler/gen_assembler_riscv32-inl.h"  // NOLINT generated file!

 private:
  Assembler() = delete;
  Assembler(const Assembler&) = delete;
  Assembler(Assembler&&) = delete;
  void operator=(const Assembler&) = delete;
  void operator=(Assembler&&) = delete;
  friend BaseAssembler;
};

}  // namespace berberis::riscv32

#endif  // BERBERIS_ASSEMBLER_RISCV32_H_
