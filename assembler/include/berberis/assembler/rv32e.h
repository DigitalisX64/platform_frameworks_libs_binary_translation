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

// Assembler to produce RV32 instructions (EABI version). Somewhat influenced by V8 assembler.

#ifndef BERBERIS_ASSEMBLER_RV32E_H_
#define BERBERIS_ASSEMBLER_RV32E_H_

#include <type_traits>  // std::is_same

#include "berberis/assembler/riscv32.h"

namespace berberis::rv32e {

class Assembler : public ::berberis::riscv32::Assembler {
 public:
  using BaseAssembler = riscv::Assembler<::berberis::riscv32::Assembler>;
  using FinalAssembler = berberis::riscv32::Assembler;

  explicit Assembler(MachineCode* code) : berberis::riscv32::Assembler(code) {}

  // Registers available used on “small” CPUs (with 16 general purpose registers) and “big” CPUs (32
  // general purpose registers).
  static constexpr Register ra{1};
  static constexpr Register sp{2};
  static constexpr Register gp{3};
  static constexpr Register tp{4};
  static constexpr Register t0{5};
  static constexpr Register s3{6};
  static constexpr Register s4{7};
  static constexpr Register s0{8};
  static constexpr Register s1{9};
  static constexpr Register a0{10};
  static constexpr Register a1{11};
  static constexpr Register a2{12};
  static constexpr Register a3{13};
  static constexpr Register s2{14};
  static constexpr Register t1{15};

  // Register only available on “big” CPUs (with 32 gneral purpose registers).
  static constexpr Register s5{16};
  static constexpr Register s6{17};
  static constexpr Register s7{18};
  static constexpr Register s8{19};
  static constexpr Register s9{20};
  static constexpr Register s10{21};
  static constexpr Register s11{22};
  static constexpr Register s12{23};
  static constexpr Register s13{24};
  static constexpr Register s14{25};
  static constexpr Register s15{26};
  static constexpr Register s16{27};
  static constexpr Register s17{28};
  static constexpr Register s18{29};
  static constexpr Register s19{30};
  static constexpr Register s20{31};

 private:
  Assembler() = delete;
  Assembler(const Assembler&) = delete;
  Assembler(Assembler&&) = delete;
  void operator=(const Assembler&) = delete;
  void operator=(Assembler&&) = delete;
};

}  // namespace berberis::rv32e

#endif  // BERBERIS_ASSEMBLER_RV32E_H_
