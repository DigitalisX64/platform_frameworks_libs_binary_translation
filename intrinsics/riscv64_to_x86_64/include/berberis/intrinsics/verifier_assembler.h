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

#ifndef RISCV64_TO_X86_64_INTRINSICS_VERIFIER_ASSEMBLER_H_
#define RISCV64_TO_X86_64_INTRINSICS_VERIFIER_ASSEMBLER_H_

#include <stdio.h>

#include "berberis/intrinsics/all_to_x86_32_or_x86_64/verifier_assembler_x86_32_and_x86_64.h"

namespace berberis {

class VerifierAssembler : public x86_32_and_x86_64::VerifierAssembler<VerifierAssembler> {
 public:
  using BaseAssembler = x86_32_and_x86_64::VerifierAssembler<VerifierAssembler>;
  using FinalAssembler = VerifierAssembler;

  constexpr VerifierAssembler([[maybe_unused]] int indent, [[maybe_unused]] FILE* out)
      : BaseAssembler() {}
  constexpr VerifierAssembler() : BaseAssembler() {}

// Instructions.
#include "gen_verifier_assembler_x86_64-inl.h"  // NOLINT generated file

  // Unhide Movq(Mem, XMMReg) and Movq(XMMReg, Mem) hidden by Movq(Reg, Imm) and many others.
  using BaseAssembler::Movq;

 protected:
  using RegisterDefaultBit = RegisterTemplate<kRsp, 'q'>;

 private:
  VerifierAssembler(const VerifierAssembler&) = delete;
  VerifierAssembler(VerifierAssembler&&) = delete;
  void operator=(const VerifierAssembler&) = delete;
  void operator=(VerifierAssembler&&) = delete;
  using DerivedAssemblerType = VerifierAssembler;

  friend BaseAssembler;
};

}  // namespace berberis

#endif  // RISCV64_TO_X86_64_INTRINSICS_VERIFIER_ASSEMBLER_H_
