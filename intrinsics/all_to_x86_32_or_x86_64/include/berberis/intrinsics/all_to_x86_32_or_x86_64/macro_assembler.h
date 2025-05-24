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

#ifndef BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_MACRO_ASSEMBLER_H_
#define BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_MACRO_ASSEMBLER_H_

#include <limits.h>
#include <type_traits>  // is_same_v

// Don't include arch-dependent parts because macro-assembler doesn't depend on implementation of
// Float32/Float64 types but can be compiled for different architecture (soong's host architecture,
// not device architecture AKA berberis' host architecture).
#include "berberis/intrinsics/common/intrinsics_float.h"

namespace berberis {

class CompilerHooks;

// When CRTP is used the derived class, supplied as template argument, can be used in the
// *implementation* of base class, but couldn't be used as part of the *interface*.
//
// And macro-assembler *interface* depends on the Assembler, that's why it's supplied separately.
//
// And we also need base class to form hierarchy.
//
// Details at go/berberis-macroassembler-mixins
template <typename AssemblerT, typename AssemblerBaseT, typename SpecificMacroAssemblerT>
class MacroAssemblerX86GuestAgnostic : public AssemblerBaseT {
 public:
  using Assembler = AssemblerBaseT;
  using AssemblerBase = AssemblerBaseT;
  using SpecificMacroAssembler = SpecificMacroAssemblerT;

#define IMPORT_ASSEMBLER_FUNCTIONS
#include "berberis/assembler/gen_assembler_x86_common-using-inl.h"
#undef IMPORT_ASSEMBLER_FUNCTIONS

#define DEFINE_MACRO_ASSEMBLER_GENERIC_FUNCTIONS
#include "berberis/intrinsics/all_to_x86_32_or_x86_64/macro_assembler-inl.h"
#undef DEFINE_MACRO_ASSEMBLER_GENERIC_FUNCTIONS

#include "berberis/intrinsics/all_to_x86_32_or_x86_64/macro_assembler_interface-inl.h"  // NOLINT generated file

  template <typename... Аrgs>
  constexpr explicit MacroAssemblerX86GuestAgnostic(Аrgs&&... args)
      : Assembler(std::forward<Аrgs>(args)...) {}
};

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_MACRO_ASSEMBLER_H_
