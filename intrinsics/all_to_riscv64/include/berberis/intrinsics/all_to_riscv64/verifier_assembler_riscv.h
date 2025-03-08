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

#ifndef BERBERIS_INTRINSICS_COMMON_TO_RISCV_VERIFIER_ASSEMBLER_COMMON_H_
#define BERBERIS_INTRINSICS_COMMON_TO_RISCV_VERIFIER_ASSEMBLER_COMMON_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

#include "berberis/assembler/riscv.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/all_to_riscv64/intrinsics_bindings.h"
#include "berberis/intrinsics/common/intrinsics_bindings.h"

namespace berberis {

namespace riscv {

template <typename DerivedAssemblerType>
class VerifierAssembler {
 public:
  using Condition = riscv::Condition;
  using Csr = riscv::Csr;
  using Rounding = riscv::Rounding;

  struct Label {
    size_t id;
    bool bound = false;
  };

  template <typename RegisterType, typename ImmediateType>
  struct Operand;

  class Register {
   public:
    constexpr Register() : arg_no_(kNoRegister) {}
    constexpr Register(int arg_no) : arg_no_(arg_no) {}
    constexpr Register(int arg_no,
                       [[maybe_unused]] intrinsics::bindings::RegBindingKind binding_kind)
        : arg_no_(arg_no) {}

    int arg_no() const {
      CHECK_NE(arg_no_, kNoRegister);
      return arg_no_;
    }

    friend bool operator==(const Register&, const Register&) = default;

    static constexpr int kNoRegister = -1;
    static constexpr int kStackPointer = -2;
    // Used in Operand to deal with references to scratch area.
    static constexpr int kScratchPointer = -3;
    static constexpr int kZeroRegister = -4;

   private:
    template <typename RegisterType, typename ImmediateType>
    friend struct Operand;

    // Register number created during creation of assembler call.
    // See arg['arm_register'] in _gen_c_intrinsic_body in gen_intrinsics.py
    //
    // Default value (-1) means it's not assigned yet (thus couldn't be used).
    int arg_no_;
  };

  class FpRegister {
   public:
    constexpr FpRegister() : arg_no_(kNoRegister) {}
    constexpr FpRegister(int arg_no) : arg_no_(arg_no) {}
    int arg_no() const {
      CHECK_NE(arg_no_, kNoRegister);
      return arg_no_;
    }

    friend bool operator==(const FpRegister&, const FpRegister&) = default;

   private:
    // Register number created during creation of assembler call.
    // See arg['arm_register'] in _gen_c_intrinsic_body in gen_intrinsics.py
    //
    // Default value (-1) means it's not assigned yet (thus couldn't be used).
    static constexpr int kNoRegister = -1;
    int arg_no_;
  };

  template <typename RegisterType, typename ImmediateType>
  struct Operand {
    RegisterType base{0};
    ImmediateType disp = 0;
  };

  using BImmediate = riscv::BImmediate;
  using CsrImmediate = riscv::CsrImmediate;
  using IImmediate = riscv::IImmediate;
  using Immediate = riscv::Immediate;
  using JImmediate = riscv::JImmediate;
  using Shift32Immediate = riscv::Shift32Immediate;
  using Shift64Immediate = riscv::Shift64Immediate;
  using PImmediate = riscv::PImmediate;
  using SImmediate = riscv::SImmediate;
  using UImmediate = riscv::UImmediate;

  using XRegister = Register;

  constexpr VerifierAssembler() {}

  // Verify CPU vendor and SSE restrictions.
  template <typename CPUIDRestriction>
  constexpr void CheckCPUIDRestriction() {}

  constexpr void CheckFlagsBinding([[maybe_unused]] bool expect_flags) {}

  constexpr void CheckAppropriateDefEarlyClobbers() {}

  // Translate CPU restrictions into string.
  template <typename CPUIDRestriction>
  static constexpr const char* kCPUIDRestrictionString =
      DerivedAssemblerType::template CPUIDRestrictionToString<CPUIDRestriction>();

  // RISC-V doesn't have “a”, “b”, “c”, or “d” registers, but we need these to be able to compile
  // the code generator.
  template <char kConstraint>
  class UnsupportedRegister {
   public:
    UnsupportedRegister operator=(Register) {
      LOG_ALWAYS_FATAL("Registers of the class “%c” don't exist on RISC-V", kConstraint);
    }
  };

  UnsupportedRegister<'a'> gpr_a;
  UnsupportedRegister<'b'> gpr_b;
  UnsupportedRegister<'c'> gpr_c;
  UnsupportedRegister<'d'> gpr_d;
  // Note: stack pointer is not reflected in list of arguments, intrinsics use
  // it implicitly.
  Register gpr_s{Register::kStackPointer};
  // Used in Operand as pseudo-register to temporary operand.
  Register gpr_scratch{Register::kScratchPointer};
  // Intrinsics which use these constants receive it via additional parameter - and
  // we need to know if it's needed or not.
  Register gpr_macroassembler_constants{};
  bool need_gpr_macroassembler_constants() const { return need_gpr_macroassembler_constants_; }

  Register gpr_macroassembler_scratch{};
  bool need_gpr_macroassembler_scratch() const { return need_gpr_macroassembler_scratch_; }
  Register gpr_macroassembler_scratch2{};

  Register zero{Register::kZeroRegister};

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

// Instructions.
#include "gen_verifier_assembler_common_riscv-inl.h"  // NOLINT generated file

 protected:
  template <typename CPUIDRestriction>
  static constexpr const char* CPUIDRestrictionToString() {
    if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::NoCPUIDRestriction>) {
      return nullptr;
    } else {
      static_assert(kDependentTypeFalse<CPUIDRestriction>);
    }
  }

  bool need_gpr_macroassembler_constants_ = false;
  bool need_gpr_macroassembler_scratch_ = false;

  template <typename Arg>
  constexpr void RegisterDef([[maybe_unused]] Arg reg) {}

  template <typename Arg>
  constexpr void RegisterUse([[maybe_unused]] Arg reg) {}

 private:
  Label label_;

  VerifierAssembler(const VerifierAssembler&) = delete;
  VerifierAssembler(VerifierAssembler&&) = delete;
  void operator=(const VerifierAssembler&) = delete;
  void operator=(VerifierAssembler&&) = delete;
};

}  // namespace riscv

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_COMMON_TO_RISCV_VERIFIER_ASSEMBLER_COMMON_H_
