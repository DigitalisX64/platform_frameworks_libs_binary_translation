/*
 * Copyright (C) 2023 The Android Open Source Project
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

#ifndef BERBERIS_INTRINSICS_RISCV64_TO_X86_64_TEXT_ASSEMBLER_H_
#define BERBERIS_INTRINSICS_RISCV64_TO_X86_64_TEXT_ASSEMBLER_H_

#include <stdio.h>

#include "berberis/intrinsics/all_to_x86_32_or_x86_64/text_assembler_x86_32_and_x86_64.h"

namespace berberis {

class TextAssembler : public x86_32_and_x86_64::TextAssembler<TextAssembler> {
 public:
  using BaseAssembler = x86_32_and_x86_64::TextAssembler<TextAssembler>;
  using FinalAssembler = TextAssembler;

  TextAssembler(int indent, FILE* out) : BaseAssembler(indent, out) {}

// Instructions.
#include "gen_text_assembler_x86_64-inl.h"  // NOLINT generated file

  // Unhide Movq(Mem, XMMReg) and Movq(XMMReg, Mem) hidden by Movq(Reg, Imm) and many others.
  using BaseAssembler::Movq;

  static constexpr char kArchName[] = "riscv64";
  static constexpr char kArchGuard[] = "RISCV64_TO_X86_64";
  static constexpr char kNamespaceName[] = "berberis";

 protected:
  using RegisterDefaultBit = RegisterTemplate<kRsp, 'q'>;

 private:
  TextAssembler() = delete;
  TextAssembler(const TextAssembler&) = delete;
  TextAssembler(TextAssembler&&) = delete;
  void operator=(const TextAssembler&) = delete;
  void operator=(TextAssembler&&) = delete;
  using DerivedAssemblerType = TextAssembler;

  friend BaseAssembler;
};

void MakeGetSetFPEnvironment(FILE* out) {
  fprintf(out,
          R"STRING(
// On platforms that we care about (Bionic, GLibc, MUSL, even x86-64 MacOS) exceptions are
// taken directly from x86 status word or MXCSR.
//
// The only exception seems to be MSVC and it can be detected with this simple check.
#if (FE_INVALID == 0x01) && (FE_DIVBYZERO == 0x04) && (FE_OVERFLOW == 0x08) && \
    (FE_UNDERFLOW == 0x10) && (FE_INEXACT == 0x20)

inline std::tuple<uint64_t> FeGetExceptions() {
  return reinterpret_cast<const char*>(&constants_pool::kBerberisMacroAssemblerConstants)
      [%1$d + fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT)];
}

inline void FeSetExceptions(uint64_t exceptions) {
  const fexcept_t x87_flag = reinterpret_cast<const char*>(
      &constants_pool::kBerberisMacroAssemblerConstants)[%2$d + exceptions];
  fesetexceptflag(&x87_flag, FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
}

inline void FeSetExceptionsImm(uint8_t exceptions) {
  FeSetExceptions(exceptions);
}

#else

#error Unsupported libc.

#endif
)STRING",
          constants_pool::GetOffset(constants_pool::kX87ToRiscVExceptions),
          constants_pool::GetOffset(constants_pool::kRiscVToX87Exceptions));
}

void MakeExtraGuestFunctions(FILE* out) {
  MakeGetSetFPEnvironment(out);
}

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_RISCV64_TO_X86_64_TEXT_ASSEMBLER_H_
