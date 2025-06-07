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

#ifndef BERBERIS_INTRINSICS_ALL_TO_X86_64_MACRO_ASSEMBLER_IMPL_H_
#define BERBERIS_INTRINSICS_ALL_TO_X86_64_MACRO_ASSEMBLER_IMPL_H_

#include "berberis/intrinsics/all_to_x86_64/constants_pool.h"
#include "berberis/intrinsics/all_to_x86_64/macro_assembler.h"

namespace berberis {

template <typename Assembler, typename AssemblerBase, typename SpecificMacroAssembler>
template <typename IntType>
constexpr void MacroAssemblerX86_64GuestAgnostic<Assembler, AssemblerBase, SpecificMacroAssembler>::
    CountLeadingZeros(Register result, Register src) {
  AssemblerBase::template CountLeadingZeros<IntType>(result, src);
}

template <typename Assembler, typename AssemblerBase, typename SpecificMacroAssembler>
template <typename IntType>
constexpr void MacroAssemblerX86_64GuestAgnostic<Assembler, AssemblerBase, SpecificMacroAssembler>::
    CountTrailingZeros(Register result, Register src) {
  AssemblerBase::template CountTrailingZeros<IntType>(result, src);
}

template <typename Assembler, typename AssemblerBase, typename SpecificMacroAssembler>
constexpr void
MacroAssemblerX86_64GuestAgnostic<Assembler, AssemblerBase, SpecificMacroAssembler>::ReverseBitsU64(
    Register dst,
    Register src,
    Register tmp) {
  Mov<uint64_t>(tmp, 0x5555'5555'5555'5555);
  Mov<uint64_t>(dst, src);
  Shr<uint64_t>(src, int8_t{1});
  And<uint64_t>(dst, tmp);
  And<uint64_t>(src, tmp);
  Mov<uint64_t>(tmp, 0x3333'3333'3333'3333);
  Shl<uint64_t>(dst, int8_t{1});
  Or<uint64_t>(src, dst);
  Mov<uint64_t>(dst, src);
  Shr<uint64_t>(src, int8_t{2});
  And<uint64_t>(dst, tmp);
  And<uint64_t>(src, tmp);
  Mov<uint64_t>(tmp, 0x0f0f'0f0f'0f0f'0f0f);
  Shl<uint64_t>(dst, int8_t{2});
  Or<uint64_t>(src, dst);
  Mov<uint64_t>(dst, src);
  Shr<uint64_t>(src, int8_t{4});
  And<uint64_t>(dst, tmp);
  And<uint64_t>(src, tmp);
  Shl<uint64_t>(dst, int8_t{4});
  Or<uint64_t>(dst, src);
  Bswap<uint64_t>(dst);
}

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_ALL_TO_X86_64_MACRO_ASSEMBLER_IMPL_H_
