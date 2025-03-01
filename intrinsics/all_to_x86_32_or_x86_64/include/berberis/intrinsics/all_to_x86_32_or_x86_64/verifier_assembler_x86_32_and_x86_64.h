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

#ifndef BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_x86_64_VERIFIER_ASSEMBLER_COMMON_H_
#define BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_x86_64_VERIFIER_ASSEMBLER_COMMON_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/all_to_x86_32_or_x86_64/intrinsics_bindings.h"

namespace berberis {

namespace x86_32_and_x86_64 {

template <typename DerivedAssemblerType>
class VerifierAssembler {
 public:
  // Condition class - 16 x86 conditions.
  enum class Condition {
    kOverflow = 0,
    kNoOverflow = 1,
    kBelow = 2,
    kAboveEqual = 3,
    kEqual = 4,
    kNotEqual = 5,
    kBelowEqual = 6,
    kAbove = 7,
    kNegative = 8,
    kPositiveOrZero = 9,
    kParityEven = 10,
    kParityOdd = 11,
    kLess = 12,
    kGreaterEqual = 13,
    kLessEqual = 14,
    kGreater = 15,

    // aka...
    kCarry = kBelow,
    kNotCarry = kAboveEqual,
    kZero = kEqual,
    kNotZero = kNotEqual,
    kSign = kNegative,
    kNotSign = kPositiveOrZero
  };

  enum ScaleFactor {
    kTimesOne = 0,
    kTimesTwo = 1,
    kTimesFour = 2,
    kTimesEight = 3,
    // All our target systems use 32-bit pointers.
    kTimesPointerSize = kTimesFour
  };

  struct Label {
    size_t id;
    bool bound = false;
  };

  struct Operand;

  class Register {
   public:
    constexpr Register(int arg_no) : arg_no_(arg_no) {}
    constexpr int arg_no() const {
      CHECK_NE(arg_no_, kNoRegister);
      return arg_no_;
    }

    constexpr bool operator==(const Register& other) const { return arg_no() == other.arg_no(); }
    constexpr bool operator!=(const Register& other) const { return arg_no() != other.arg_no(); }

    static constexpr int kNoRegister = -1;
    static constexpr int kStackPointer = -2;
    // Used in Operand to deal with references to scratch area.
    static constexpr int kScratchPointer = -3;

   private:
    friend struct Operand;

    // Register number created during creation of assembler call.
    // See arg['arm_register'] in _gen_c_intrinsic_body in gen_intrinsics.py
    //
    // Default value (-1) means it's not assigned yet (thus couldn't be used).
    int arg_no_;
  };

  class X87Register {
   public:
    constexpr X87Register(int arg_no) : arg_no_(arg_no) {}
    int arg_no() const {
      CHECK_NE(arg_no_, kNoRegister);
      return arg_no_;
    }

    constexpr bool operator==(const X87Register& other) const { return arg_no_ == other.arg_no_; }
    constexpr bool operator!=(const X87Register& other) const { return arg_no_ != other.arg_no_; }

   private:
    // Register number created during creation of assembler call.
    // See arg['arm_register'] in _gen_c_intrinsic_body in gen_intrinsics.py
    //
    // Default value (-1) means it's not assigned yet (thus couldn't be used).
    static constexpr int kNoRegister = -1;
    int arg_no_;
  };

  template <int kBits>
  class SIMDRegister {
   public:
    friend class SIMDRegister<384 - kBits>;
    constexpr SIMDRegister(int arg_no) : arg_no_(arg_no) {}
    int arg_no() const {
      CHECK_NE(arg_no_, kNoRegister);
      return arg_no_;
    }

    constexpr bool operator==(const SIMDRegister& other) const {
      return arg_no() == other.arg_no();
    }
    constexpr bool operator!=(const SIMDRegister& other) const {
      return arg_no() != other.arg_no();
    }

    constexpr auto To128Bit() const {
      return std::enable_if_t<kBits != 128, SIMDRegister<128>>{arg_no_};
    }
    constexpr auto To256Bit() const {
      return std::enable_if_t<kBits != 256, SIMDRegister<256>>{arg_no_};
    }

   private:
    // Register number created during creation of assembler call.
    // See arg['arm_register'] in _gen_c_intrinsic_body in gen_intrinsics.py
    //
    // Default value (-1) means it's not assigned yet (thus couldn't be used).
    static constexpr int kNoRegister = -1;
    int arg_no_;
  };

  using XMMRegister = SIMDRegister<128>;
  using YMMRegister = SIMDRegister<256>;

  struct Operand {
    Register base = Register{Register::kNoRegister};
    Register index = Register{Register::kNoRegister};
    ScaleFactor scale = kTimesOne;
    int32_t disp = 0;
  };

  constexpr VerifierAssembler() {}

  // These start as Register::kNoRegister but can be changed if they are used as arguments to
  // something else.
  // If they are not coming as arguments then using them is compile-time error!
  Register gpr_a{Register::kNoRegister};
  Register gpr_b{Register::kNoRegister};
  Register gpr_c{Register::kNoRegister};
  Register gpr_d{Register::kNoRegister};
  // Note: stack pointer is not reflected in list of arguments, intrinsics use
  // it implicitly.
  Register gpr_s{Register::kStackPointer};
  // Used in Operand as pseudo-register to temporary operand.
  Register gpr_scratch{Register::kScratchPointer};

  // In x86-64 case we could refer to kBerberisMacroAssemblerConstants via %rip.
  // In x86-32 mode, on the other hand, we need complex dance to access it via GOT.
  // Intrinsics which use these constants receive it via additional parameter - and
  // we need to know if it's needed or not.
  Register gpr_macroassembler_constants{Register::kNoRegister};
  bool need_gpr_macroassembler_constants() const { return need_gpr_macroassembler_constants_; }

  Register gpr_macroassembler_scratch{Register::kNoRegister};
  bool need_gpr_macroassembler_scratch() const { return need_gpr_macroassembler_scratch_; }
  Register gpr_macroassembler_scratch2{Register::kNoRegister};

  bool need_aesavx = false;
  bool need_aes = false;
  bool need_avx = false;
  bool need_avx2 = false;
  bool need_bmi = false;
  bool need_bmi2 = false;
  bool need_clmulavx = false;
  bool need_clmul = false;
  bool need_f16c = false;
  bool need_fma = false;
  bool need_fma4 = false;
  bool need_lzcnt = false;
  bool need_popcnt = false;
  bool need_sse3 = false;
  bool need_ssse3 = false;
  bool need_sse4_1 = false;
  bool need_sse4_2 = false;
  bool need_vaes = false;
  bool need_vpclmulqd = false;
  bool has_custom_capability = false;

  bool defines_flags = false;

  constexpr void Bind([[maybe_unused]] Label* label) {}

  // Currently label_ is meaningless. Verifier assembler does not yet have a need for it.
  constexpr Label* MakeLabel() { return &label_; }

  template <typename... Args>
  constexpr void Byte([[maybe_unused]] Args... args) {
    static_assert((std::is_same_v<Args, uint8_t> && ...));
  }

  template <typename... Args>
  constexpr void TwoByte([[maybe_unused]] Args... args) {
    static_assert((std::is_same_v<Args, uint16_t> && ...));
  }

  template <typename... Args>
  constexpr void FourByte([[maybe_unused]] Args... args) {
    static_assert((std::is_same_v<Args, uint32_t> && ...));
  }

  template <typename... Args>
  constexpr void EigthByte([[maybe_unused]] Args... args) {
    static_assert((std::is_same_v<Args, uint64_t> && ...));
  }

  constexpr void P2Align([[maybe_unused]] uint32_t m) {}

  // Verify CPU vendor and SSE restrictions.
  template <typename CPUIDRestriction>
  constexpr void CheckCPUIDRestriction() {
    constexpr bool expect_bmi = std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasBMI>;
    constexpr bool expect_f16c = std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasF16C>;
    constexpr bool expect_fma = std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasFMA>;
    constexpr bool expect_fma4 = std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasFMA4>;
    constexpr bool expect_lzcnt = std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasLZCNT>;
    constexpr bool expect_vaes = std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasVAES>;
    constexpr bool expect_vpclmulqd =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasVPCLMULQD>;
    constexpr bool expect_aesavx =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasAESAVX> || expect_vaes;
    constexpr bool expect_aes =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasAES> || expect_aesavx;
    constexpr bool expect_clmulavx =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasCLMULAVX> || expect_vpclmulqd;
    constexpr bool expect_clmul =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasCLMUL> || expect_clmulavx;
    constexpr bool expect_popcnt =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasPOPCNT>;
    constexpr bool expect_avx = std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasAVX> ||
                                expect_aesavx || expect_clmulavx || expect_f16c || expect_fma ||
                                expect_fma4;
    constexpr bool expect_sse4_2 =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSE4_2> || expect_aes ||
        expect_clmul || expect_avx;
    constexpr bool expect_sse4_1 =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSE4_1> || expect_sse4_2;
    constexpr bool expect_ssse3 =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSSE3> || expect_sse4_1;
    constexpr bool expect_sse3 =
        std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSE3> || expect_ssse3;

    if (expect_aesavx != need_aesavx) {
      printf("error: expect_aesavx != need_aesavx\n");
    }
    if (expect_aes != need_aes) {
      printf("error: expect_aes != need_aes\n");
    }
    if (expect_avx != need_avx) {
      printf("error: expect_avx != need_avx\n");
    }
    if (expect_bmi != need_bmi) {
      printf("error: expect_bmi != need_bmi\n");
    }
    if (expect_clmulavx != need_clmulavx) {
      printf("error: expect_clmulavx != need_clmulavx\n");
    }
    if (expect_clmul != need_clmul) {
      printf("error: expect_clmul != need_clmul\n");
    }
    if (expect_f16c != need_f16c) {
      printf("error: expect_f16c != need_f16c\n");
    }
    if (expect_fma != need_fma) {
      printf("error: expect_fma != need_fma\n");
    }
    if (expect_fma4 != need_fma4) {
      printf("error: expect_fma4 != need_fma4\n");
    }
    if (expect_lzcnt != need_lzcnt) {
      printf("error: expect_lzcnt != need_lzcnt\n");
    }
    if (expect_popcnt != need_popcnt) {
      printf("error: expect_popcnt != need_popcnt\n");
    }
    if (expect_sse3 != need_sse3) {
      printf("error: expect_sse3 != need_sse3\n");
    }
    if (expect_ssse3 != need_ssse3) {
      printf("error: expect_ssse3 != need_ssse3\n");
    }
    if (expect_sse4_1 != need_sse4_1) {
      printf("error: expect_sse4_1 != need_sse4_1\n");
    }
    if (expect_sse4_2 != need_sse4_2) {
      printf("error: expect_sse4_2 != need_sse4_2\n");
    }
    if (expect_vaes != need_vaes) {
      printf("error: expect_vaes != need_vaes\n");
    }
    if (expect_vpclmulqd != need_vpclmulqd) {
      printf("error: expect_vpclmulqd != need_vpclmulqd\n");
    }
  }

  constexpr void CheckFlagsBinding(bool expect_flags) {
    if (expect_flags != defines_flags) {
      printf("error: expect_flags != defines_flags\n");
    }
  }

// Instructions.
#include "gen_verifier_assembler_common_x86-inl.h"  // NOLINT generated file

 protected:
  bool need_gpr_macroassembler_constants_ = false;
  bool need_gpr_macroassembler_scratch_ = false;

  template <const char* kSpPrefix, char kRegisterPrefix>
  class RegisterTemplate {
   public:
    explicit constexpr RegisterTemplate(Register reg) : reg_(reg) {}

   private:
    Register reg_;
  };

  constexpr static char kSpl[] = "%%spl";
  using Register8Bit = RegisterTemplate<kSpl, 'b'>;
  constexpr static char kSp[] = "%%sp";
  using Register16Bit = RegisterTemplate<kSp, 'w'>;
  constexpr static char kEsp[] = "%%esp";
  using Register32Bit = RegisterTemplate<kEsp, 'k'>;
  constexpr static char kRsp[] = "%%rsp";
  using Register64Bit = RegisterTemplate<kRsp, 'q'>;

  constexpr void SetRequiredFeatureAESAVX() {
    need_aesavx = true;
    SetRequiredFeatureAES();
    SetRequiredFeatureAVX();
  }

  constexpr void SetRequiredFeatureAES() {
    need_aes = true;
    SetRequiredFeatureSSE4_2();
  }

  constexpr void SetRequiredFeatureAVX() {
    need_avx = true;
    SetRequiredFeatureSSE4_2();
  }

  constexpr void SetRequiredFeatureAVX2() {
    need_avx2 = true;
    SetRequiredFeatureAVX();
  }

  constexpr void SetRequiredFeatureBMI() { need_bmi = true; }

  constexpr void SetRequiredFeatureBMI2() { need_bmi2 = true; }

  constexpr void SetRequiredFeatureCLMULAVX() {
    need_clmulavx = true;
    SetRequiredFeatureCLMUL();
    SetRequiredFeatureAVX();
  }

  constexpr void SetRequiredFeatureCLMUL() {
    need_clmul = true;
    SetRequiredFeatureSSE4_2();
  }

  constexpr void SetRequiredFeatureF16C() {
    need_f16c = true;
    SetRequiredFeatureAVX();
  }

  constexpr void SetRequiredFeatureFMA() {
    need_fma = true;
    SetRequiredFeatureAVX();
  }

  constexpr void SetRequiredFeatureFMA4() {
    need_fma4 = true;
    SetRequiredFeatureAVX();
  }

  constexpr void SetRequiredFeatureLZCNT() { need_lzcnt = true; }

  constexpr void SetRequiredFeaturePOPCNT() { need_popcnt = true; }

  constexpr void SetRequiredFeatureSSE3() {
    need_sse3 = true;
    // Note: we assume that SSE2 is always available thus we don't have have_sse2 or have_sse1
    // variables.
  }

  constexpr void SetRequiredFeatureSSSE3() {
    need_ssse3 = true;
    SetRequiredFeatureSSE3();
  }

  constexpr void SetRequiredFeatureSSE4_1() {
    need_sse4_1 = true;
    SetRequiredFeatureSSSE3();
  }

  constexpr void SetRequiredFeatureSSE4_2() {
    need_sse4_2 = true;
    SetRequiredFeatureSSE4_1();
  }

  constexpr void SetRequiredFeatureVAES() {
    need_vaes = true;
    SetRequiredFeatureAESAVX();
  }

  constexpr void SetRequiredFeatureVPCLMULQD() {
    need_vpclmulqd = true;
    SetRequiredFeatureCLMULAVX();
  }

  constexpr void SetHasCustomCapability() { has_custom_capability = true; }

  constexpr void SetDefinesFLAGS() { defines_flags = true; }

  template <typename... Args>
  constexpr void Instruction(const char* name, Condition cond, const Args&... args);

  template <typename... Args>
  constexpr void Instruction(const char* name, const Args&... args);

  void EmitString() {}

  void EmitString([[maybe_unused]] const std::string& s) {}

  template <typename... Args>
  void EmitString([[maybe_unused]] const std::string& s, [[maybe_unused]] const Args&... args) {}

 private:
  Label label_;

  VerifierAssembler(const VerifierAssembler&) = delete;
  VerifierAssembler(VerifierAssembler&&) = delete;
  void operator=(const VerifierAssembler&) = delete;
  void operator=(VerifierAssembler&&) = delete;
};

template <typename DerivedAssemblerType>
template <typename... Args>
constexpr void VerifierAssembler<DerivedAssemblerType>::Instruction(
    [[maybe_unused]] const char* name,
    [[maybe_unused]] Condition cond,
    [[maybe_unused]] const Args&... args) {}

template <typename DerivedAssemblerType>
template <typename... Args>
constexpr void VerifierAssembler<DerivedAssemblerType>::Instruction(
    [[maybe_unused]] const char* name,
    [[maybe_unused]] const Args&... args) {}

}  // namespace x86_32_and_x86_64

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_x86_64_VERIFIER_ASSEMBLER_COMMON_H_
