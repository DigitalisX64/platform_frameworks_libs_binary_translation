/*
 * Copyright (C) 2020 The Android Open Source Project
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

#ifndef RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_BITMANIP_IMPL_H_
#define RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_BITMANIP_IMPL_H_

#include <climits>

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/macro_assembler.h"

namespace berberis {

template <typename Assembler>
template <typename IntType>
constexpr void MacroAssembler<Assembler>::Clz(Register result, Register src) {
  Bsr<IntType>(result, src);
  Cmov<IntType>(Condition::kZero, result, {.disp = constants_offsets::kBsrToClz<IntType>});
  Xor<IntType>(result, sizeof(IntType) * CHAR_BIT - 1);
}

template <typename Assembler>
template <typename IntType>
constexpr void MacroAssembler<Assembler>::Ctz(Register result, Register src) {
  Bsf<IntType>(result, src);
  Cmov<IntType>(Condition::kZero, result, {.disp = constants_offsets::kWidthInBits<IntType>});
}

template <typename Assembler>
template <typename IntType>
constexpr void MacroAssembler<Assembler>::Max(Register result, Register src1, Register src2) {
  Mov<IntType>(result, src1);
  Cmp<IntType>(src1, src2);
  if constexpr (std::is_signed_v<IntType>) {
    Cmov<IntType>(Condition::kLess, result, src2);
  } else {
    Cmov<IntType>(Condition::kBelow, result, src2);
  }
}

template <typename Assembler>
template <typename IntType>
constexpr void MacroAssembler<Assembler>::Min(Register result, Register src1, Register src2) {
  Mov<IntType>(result, src1);
  Cmp<IntType>(src1, src2);
  if constexpr (std::is_signed_v<IntType>) {
    Cmov<IntType>(Condition::kGreater, result, src2);
  } else {
    Cmov<IntType>(Condition::kAbove, result, src2);
  }
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Orcb(XMMRegister result) {
  Pcmpeqb(result, {.disp = constants_offsets::kVectorConst<uint8_t{0}>});
  PNot(result);
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::OrcbAVX(XMMRegister result, XMMRegister src) {
  Vpcmpeqb(result, src, {.disp = constants_offsets::kVectorConst<uint8_t{0}>});
  Vpnot(result, result);
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Adduw(Register result, Register src) {
  Movl(result, result);
  Leaq(result, {.base = src, .index = result, .scale = Assembler::kTimesOne});
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Sh1adduw(Register result, Register src) {
  Movl(result, result);
  Leaq(result, {.base = src, .index = result, .scale = Assembler::kTimesTwo});
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Sh2adduw(Register result, Register src) {
  Movl(result, result);
  Leaq(result, {.base = src, .index = result, .scale = Assembler::kTimesFour});
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Sh3adduw(Register result, Register src) {
  Movl(result, result);
  Leaq(result, {.base = src, .index = result, .scale = Assembler::kTimesEight});
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Sh1add(Register result, Register src) {
  Leaq(result, {.base = src, .index = result, .scale = Assembler::kTimesTwo});
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Sh2add(Register result, Register src) {
  Leaq(result, {.base = src, .index = result, .scale = Assembler::kTimesFour});
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Sh3add(Register result, Register src) {
  Leaq(result, {.base = src, .index = result, .scale = Assembler::kTimesEight});
}

template <typename Assembler>
constexpr void MacroAssembler<Assembler>::Bext(Register result, Register src1, Register src2) {
  Btq(src1, src2);
  Movl(result, 0);
  Setcc(Condition::kCarry, result);
}

}  // namespace berberis

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_BITMANIP_IMPL_H_
