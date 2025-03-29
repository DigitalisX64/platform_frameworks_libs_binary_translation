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

#include "gtest/gtest.h"

#include "berberis/intrinsics/all_to_x86_32_or_x86_64/intrinsics_bindings.h"
#include "berberis/intrinsics/all_to_x86_32_or_x86_64/intrinsics_float.h"
#include "berberis/intrinsics/all_to_x86_32_or_x86_64/verifier_assembler_x86_32_and_x86_64.h"

namespace berberis {

namespace {

template <typename Assembler>
class MacroAssembler : public Assembler {
 public:
  using MacroAssemblers = std::tuple<MacroAssembler<Assembler>,
                                     typename Assembler::BaseAssembler,
                                     typename Assembler::FinalAssembler>;
  template <typename... Args>
  constexpr explicit MacroAssembler(Args&&... args) : Assembler(std::forward<Args>(args)...) {}

#define IMPORT_ASSEMBLER_FUNCTIONS
#include "berberis/assembler/gen_assembler_x86_common-using-inl.h"
#undef IMPORT_ASSEMBLER_FUNCTIONS

#define DEFINE_MACRO_ASSEMBLER_GENERIC_FUNCTIONS
#include "berberis/intrinsics/all_to_x86_32_or_x86_64/macro_assembler-inl.h"
#undef DEFINE_MACRO_ASSEMBLER_GENERIC_FUNCTIONS

  constexpr void ExecuteSSE3Instruction(XMMRegister dst, XMMRegister src1) { Haddpd(dst, src1); }
};

class VerifierAssembler : public x86_32_and_x86_64::VerifierAssembler<VerifierAssembler> {
 public:
  using BaseAssembler = x86_32_and_x86_64::VerifierAssembler<VerifierAssembler>;
  using FinalAssembler = VerifierAssembler;

  constexpr VerifierAssembler() : BaseAssembler() {}

 private:
  VerifierAssembler(const VerifierAssembler&) = delete;
  VerifierAssembler(VerifierAssembler&&) = delete;
  void operator=(const VerifierAssembler&) = delete;
  void operator=(VerifierAssembler&&) = delete;
  using DerivedAssemblerType = VerifierAssembler;

  friend BaseAssembler;
};

template <typename AsmCallInfo>
constexpr void VerifyIntrinsic() {
  int register_numbers[std::tuple_size_v<typename AsmCallInfo::Bindings> == 0
                           ? 1
                           : std::tuple_size_v<typename AsmCallInfo::Bindings>];
  AssignRegisterNumbers<AsmCallInfo>(register_numbers);
  MacroAssembler<VerifierAssembler> as;
  CallVerifierAssembler<AsmCallInfo, MacroAssembler<VerifierAssembler>>(&as, register_numbers);
  // Verify CPU vendor and SSE restrictions.
  as.CheckCPUIDRestriction<typename AsmCallInfo::CPUIDRestriction>();
}

template <typename AsmCallInfo>
constexpr bool CallVerifyIntrinsic() {
  VerifyIntrinsic<AsmCallInfo>();
  return true;
}

static constexpr const char BINDING_NAME[] = "TestInstruction";
static constexpr const char BINDING_MNEMO[] = "TEST_0";

using MacroAssemblers = MacroAssembler<VerifierAssembler>::MacroAssemblers;

TEST(VERIFIER_ASSEMBLER, TestCorrectCPUID) {
  using AsmCallInfo = intrinsics::bindings::AsmCallInfo<
      BINDING_NAME,
      static_cast<void (std::tuple_element_t<0, MacroAssemblers>::*)(
          typename std::tuple_element_t<0, MacroAssemblers>::XMMRegister,
          typename std::tuple_element_t<0, MacroAssemblers>::XMMRegister)>(
          &std::tuple_element_t<0, MacroAssemblers>::ExecuteSSE3Instruction),
      BINDING_MNEMO,
      void,
      intrinsics::bindings::HasSSE3,
      intrinsics::bindings::NoNansOperation,
      false,
      std::tuple<SIMD128Register, SIMD128Register>,
      std::tuple<SIMD128Register>,
      InOutArg<0, 0, intrinsics::bindings::XmmReg, intrinsics::bindings::Def>,
      InArg<1, intrinsics::bindings::XmmReg, intrinsics::bindings::Use>>;

  ASSERT_TRUE(CallVerifyIntrinsic<AsmCallInfo>());
}

TEST(VERIFIER_ASSEMBLER, TestIncorrectCPUID) {
  using AsmCallInfo = intrinsics::bindings::AsmCallInfo<
      BINDING_NAME,
      static_cast<void (std::tuple_element_t<0, MacroAssemblers>::*)(
          typename std::tuple_element_t<0, MacroAssemblers>::XMMRegister,
          typename std::tuple_element_t<0, MacroAssemblers>::XMMRegister)>(
          &std::tuple_element_t<0, MacroAssemblers>::ExecuteSSE3Instruction),
      BINDING_MNEMO,
      void,
      intrinsics::bindings::NoCPUIDRestriction,
      intrinsics::bindings::NoNansOperation,
      false,
      std::tuple<SIMD128Register, SIMD128Register>,
      std::tuple<SIMD128Register>,
      InOutArg<0, 0, intrinsics::bindings::XmmReg, intrinsics::bindings::Def>,
      InArg<1, intrinsics::bindings::XmmReg, intrinsics::bindings::Use>>;

  ASSERT_DEATH(CallVerifyIntrinsic<AsmCallInfo>(), "error: expect_sse3 != need_sse3");
}

}  // namespace

}  // namespace berberis
