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

#ifndef RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_ARITH_IMPL_H_
#define RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_ARITH_IMPL_H_

#include <climits>
#include <type_traits>

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/macro_assembler.h"

namespace berberis {

// Divisor comes in "src", dividend comes in gpr_a, result is returned in gpr_a.
// gpr_d and FLAGS are clobbered by that macroinstruction.
template <typename Assembler>
template <typename IntType>
constexpr void MacroAssembler<Assembler>::MacroDiv(Register src) {
  Label* zero = MakeLabel();
  Label* done = MakeLabel();
  Test<IntType>(src, src);
  Jcc(Condition::kZero, *zero);

  if constexpr (std::is_signed_v<IntType>) {
    Label* do_idiv = MakeLabel();
    // If min int32_t/int64_t is divided by -1 then in risc-v the result is
    // the dividend, but x86 will raise an exception. Handle this case separately.
    Cmp<IntType>(src, int8_t{-1});
    Jcc(Condition::kNotEqual, *do_idiv);

    if constexpr (std::is_same_v<IntType, int64_t>) {
      Cmp<IntType>(gpr_a,
                   {.disp = constants_offsets::kVectorConst<std::numeric_limits<IntType>::min()>});
    } else {
      Cmp<IntType>(gpr_a, std::numeric_limits<IntType>::min());
    }
    Jcc(Condition::kEqual, *done);

    Bind(do_idiv);
    // If we are dealing with 8-bit signed case then we need to sign-extend %al into %ax.
    if constexpr (std::is_same_v<IntType, int8_t>) {
      Cbw();
    // We need to sign-extend gpr_a into gpr_d to ensure 32bit/64-bit/128-bit dividend is correct.
    } else if constexpr (std::is_same_v<IntType, int16_t>) {
      Cwd();
    } else if constexpr (std::is_same_v<IntType, int32_t>) {
      Cdq();
    } else if constexpr (std::is_same_v<IntType, int64_t>) {
      Cqo();
    } else {
      static_assert(kDependentTypeFalse<IntType>, "Unsupported format");
    }
  } else if constexpr (std::is_same_v<IntType, uint8_t>) {
    // For 8bit unsigned case we need “xor %ah, %ah” instruction, but our assembler doesn't support
    // %ah register. Use .byte to emit the required machine code.
    TwoByte(uint16_t{0xe430});
  } else {
    // We need to zero-extend eax into dx/edx/rdx to ensure 32-bit/64-bit/128-bit dividend is
    // correct.
    Xor<uint32_t>(gpr_d, gpr_d);
  }

  Div<IntType>(src);
  Jmp(*done);

  Bind(zero);
  Mov<IntType>(gpr_a, int64_t{-1});

  Bind(done);
}

// Divisor comes in "src", dividend comes in gpr_a.
// For 16/32/64-bit: remainder is returned in gpr_d. gpr_a and FLAGS are clobbered.
// For 8-bit: remainder is returned in gpr_a. FLAGS are clobbered.
template <typename Assembler>
template <typename IntType>
constexpr void MacroAssembler<Assembler>::MacroRem(Register src) {
  Label* zero = MakeLabel();
  Label* overflow = MakeLabel();
  Label* done = MakeLabel();
  Test<IntType>(src, src);
  Jcc(Condition::kZero, *zero);

  if constexpr (std::is_signed_v<IntType>) {
    Label* do_idiv = MakeLabel();
    // If min int32_t/int64_t is divided by -1 then in risc-v the result is
    // the dividend, but x86 will raise an exception. Handle this case separately.
    Cmp<IntType>(src, int8_t{-1});
    Jcc(Condition::kNotEqual, *do_idiv);

    if constexpr (std::is_same_v<IntType, int64_t>) {
      Cmp<IntType>(gpr_a,
                   {.disp = constants_offsets::kVectorConst<std::numeric_limits<IntType>::min()>});
    } else {
      Cmp<IntType>(gpr_a, std::numeric_limits<IntType>::min());
    }
    Jcc(Condition::kEqual, *overflow);

    Bind(do_idiv);
    // If we are dealing with 8-bit signed case then we need to sign-extend %al into %ax.
    if constexpr (std::is_same_v<IntType, int8_t>) {
      Cbw();
      // We need to sign-extend gpr_a into gpr_d to ensure 32bit/64-bit/128-bit dividend is correct.
    } else if constexpr (std::is_same_v<IntType, int16_t>) {
      Cwd();
    } else if constexpr (std::is_same_v<IntType, int32_t>) {
      Cdq();
    } else if constexpr (std::is_same_v<IntType, int64_t>) {
      Cqo();
    } else {
      static_assert(kDependentTypeFalse<IntType>, "Unsupported format");
    }
  } else if constexpr (std::is_same_v<IntType, uint8_t>) {
    // For 8bit unsigned case we need “xor %ah, %ah” instruction, but our assembler doesn't support
    // %ah register. Use .byte to emit the required machine code.
    TwoByte(uint16_t{0xe430});
  } else {
    // We need to zero-extend eax into dx/edx/rdx to ensure 32-bit/64-bit/128-bit dividend is
    // correct.
    Xor<uint32_t>(gpr_d, gpr_d);
  }

  Div<IntType>(src);
  if constexpr (std::is_same_v<IntType, uint8_t> || std::is_same_v<IntType, int8_t>) {
    // For 8bit case the result is in %ah, but our assembler doesn't support
    // %ah register. move %ah to %al
    TwoByte(uint16_t{0xe086});
  }
  Jmp(*done);

  Bind(zero);
  if constexpr (!std::is_same_v<IntType, uint8_t> && !std::is_same_v<IntType, int8_t>) {
    Mov<IntType>(gpr_d, gpr_a);
  }
  Jmp(*done);

  Bind(overflow);
  if constexpr (std::is_same_v<IntType, uint8_t> || std::is_same_v<IntType, int8_t>) {
    Xor<int8_t>(gpr_a, gpr_a);
  } else {
    Xor<IntType>(gpr_d, gpr_d);
  }
  Bind(done);
}
}  // namespace berberis

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_ARITH_IMPL_H_
