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

#ifndef BERBERIS_INTRINSICS_COMMON_TO_RISCV_TEXT_ASSEMBLER_COMMON_H_
#define BERBERIS_INTRINSICS_COMMON_TO_RISCV_TEXT_ASSEMBLER_COMMON_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>

#include "berberis/assembler/riscv.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/all_to_riscv64/intrinsics_bindings.h"

namespace berberis {

namespace constants_pool {

extern const intptr_t kBerberisMacroAssemblerConstantsRelocated;

inline intptr_t GetOffset(intptr_t address) {
  return address - constants_pool::kBerberisMacroAssemblerConstantsRelocated;
}

}  // namespace constants_pool

namespace riscv {

#define BERBERIS_DEFINE_TO_FAS_ARGUMENT(Immediate)                         \
  template <typename MacroAssembler>                                       \
  inline std::string ToGasArgument(Immediate immediate, MacroAssembler*) { \
    return "$" + std::to_string(static_cast<int32_t>(immediate));          \
  }
BERBERIS_DEFINE_TO_FAS_ARGUMENT(BImmediate)
BERBERIS_DEFINE_TO_FAS_ARGUMENT(CsrImmediate)
BERBERIS_DEFINE_TO_FAS_ARGUMENT(IImmediate)
BERBERIS_DEFINE_TO_FAS_ARGUMENT(JImmediate)
BERBERIS_DEFINE_TO_FAS_ARGUMENT(PImmediate)
BERBERIS_DEFINE_TO_FAS_ARGUMENT(Shift32Immediate)
BERBERIS_DEFINE_TO_FAS_ARGUMENT(Shift64Immediate)
BERBERIS_DEFINE_TO_FAS_ARGUMENT(SImmediate)
BERBERIS_DEFINE_TO_FAS_ARGUMENT(UImmediate)
#undef BERBERIS_DEFINE_TO_FAS_ARGUMENT

template <typename MacroAssembler>
inline std::string ToGasArgument(Rounding rm, MacroAssembler*) {
  switch (rm) {
    case Rounding::kRne:
      return "rne";
    case Rounding::kRtz:
      return "rtz";
    case Rounding::kRdn:
      return "rdn";
    case Rounding::kRup:
      return "ruo";
    case Rounding::kRmm:
      return "rmm";
    case Rounding::kDyn:
      return "dyn";
    default:
      LOG_ALWAYS_FATAL("Unsupported rounding mode %d", rm);
  }
}

template <typename DerivedAssemblerType>
class TextAssembler {
 public:
  using Condition = riscv::Condition;
  using Csr = riscv::Csr;
  using Rounding = riscv::Rounding;

  struct Label {
    size_t id;
    bool bound = false;

    template <typename MacroAssembler>
    friend std::string ToGasArgument(const Label& label, MacroAssembler*) {
      return std::to_string(label.id) + (label.bound ? "b" : "f");
    }
  };

  template <typename RegisterType, typename ImmediateType>
  struct Operand;

  class Register {
   public:
    constexpr Register() : arg_no_(kNoRegister) {}
    constexpr Register(int arg_no) : arg_no_(arg_no) {}
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

    template <typename MacroAssembler>
    friend const std::string ToGasArgument(const Register& reg, MacroAssembler*) {
      if (reg.arg_no_ == kZeroRegister) {
        return "zero";
      }

      return '%' + std::to_string(reg.arg_no());
    }

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

    template <typename MacroAssembler>
    friend const std::string ToGasArgument(const FpRegister& reg, MacroAssembler*) {
      return '%' + std::to_string(reg.arg_no());
    }

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

    template <typename MacroAssembler>
    friend const std::string ToGasArgument(const Operand& op, MacroAssembler* as) {
      std::string result{};
      result = '(' + ToGasArgument(op.base, as) + ')';
      int32_t disp = static_cast<int32_t>(op.disp);
      if (disp) {
        result = ToGasArgument(disp, as) + result;
      }
      return result;
    }
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

  TextAssembler(int indent, FILE* out) : indent_(indent), out_(out) {}

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

  void Bind(Label* label) {
    CHECK_EQ(label->bound, false);
    fprintf(out_, "%*s\"%zd:\\n\"\n", indent_ + 2, "", label->id);
    label->bound = true;
  }

  Label* MakeLabel() {
    labels_allocated_.push_back({labels_allocated_.size()});
    return &labels_allocated_.back();
  }

  template <typename... Args>
  void Byte(Args... args) {
    static_assert((std::is_same_v<Args, uint8_t> && ...));
    bool print_kwd = true;
    fprintf(out_, "%*s\"", indent_ + 2, "");
    (fprintf(out_, "%s%" PRIu8, print_kwd ? print_kwd = false, ".byte " : ", ", args), ...);
    fprintf(out_, "\\n\"\n");
  }

  template <typename... Args>
  void TwoByte(Args... args) {
    static_assert((std::is_same_v<Args, uint16_t> && ...));
    bool print_kwd = true;
    fprintf(out_, "%*s\"", indent_ + 2, "");
    (fprintf(out_, "%s%" PRIu16, print_kwd ? print_kwd = false, ".2byte " : ", ", args), ...);
    fprintf(out_, "\\n\"\n");
  }

  template <typename... Args>
  void FourByte(Args... args) {
    static_assert((std::is_same_v<Args, uint32_t> && ...));
    bool print_kwd = true;
    fprintf(out_, "%*s\"", indent_ + 2, "");
    (fprintf(out_, "%s%" PRIu32, print_kwd ? print_kwd = false, ".4byte " : ", ", args), ...);
    fprintf(out_, "\\n\"\n");
  }

  template <typename... Args>
  void EigthByte(Args... args) {
    static_assert((std::is_same_v<Args, uint64_t> && ...));
    bool print_kwd = true;
    fprintf(out_, "%*s\"", indent_ + 2, "");
    (fprintf(out_, "%s%" PRIu64, print_kwd ? print_kwd = false, ".8byte " : ", ", args), ...);
    fprintf(out_, "\\n\"\n");
  }

  void P2Align(uint32_t m) {
    fprintf(out_, "%*s\".p2align %u\\n\"\n", indent_ + 2, "", m);
  }

// Instructions.
#include "gen_text_assembler_common_riscv-inl.h"  // NOLINT generated file

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

  template <typename... Args>
  void Instruction(const char* name, Condition cond, const Args&... args);

  template <typename... Args>
  void Instruction(const char* name, const Args&... args);

  void EmitString() {}

  void EmitString(const std::string& s) { fprintf(out_, "%s", s.c_str()); }

  template <typename... Args>
  void EmitString(const std::string& s, const Args&... args) {
    fprintf(out_, "%s, ", s.c_str());
    EmitString(args...);
  }

 protected:
  int indent_;
  FILE* out_;

 private:
  std::deque<Label> labels_allocated_;

  TextAssembler() = delete;
  TextAssembler(const TextAssembler&) = delete;
  TextAssembler(TextAssembler&&) = delete;
  void operator=(const TextAssembler&) = delete;
  void operator=(TextAssembler&&) = delete;
};

template <typename Arg, typename MacroAssembler>
inline std::string ToGasArgument(const Arg& arg, MacroAssembler*) {
  return "$" + std::to_string(arg);
}

template <typename DerivedAssemblerType>
template <typename... Args>
inline void TextAssembler<DerivedAssemblerType>::Instruction(const char* name,
                                                             Condition cond,
                                                             const Args&... args) {
  char name_with_condition[8] = {};
  CHECK_EQ(strcmp(name, "Bcc"), 0);

  switch (cond) {
    case Condition::kEqual:
      strcat(name_with_condition, "eq");
      break;
    case Condition::kNotEqual:
      strcat(name_with_condition, "ne");
      break;
    case Condition::kLess:
      strcat(name_with_condition, "lt");
      break;
    case Condition::kGreaterEqual:
      strcat(name_with_condition, "ge");
      break;
    case Condition::kBelow:
      strcat(name_with_condition, "ltu");
      break;
    case Condition::kAboveEqual:
      strcat(name_with_condition, "geu");
      break;
    default:
      LOG_ALWAYS_FATAL("Unsupported condition %d", cond);
  }
  Instruction(name_with_condition, args...);
}

template <typename DerivedAssemblerType>
template <typename... Args>
inline void TextAssembler<DerivedAssemblerType>::Instruction(const char* name,
                                                             const Args&... args) {
  int name_length = strlen(name);
  fprintf(out_, "%*s\"%.*s ", indent_ + 2, "", name_length, name);
  EmitString(ToGasArgument(args, this)...);
  fprintf(out_, "\\n\"\n");
}

}  // namespace riscv

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_COMMON_TO_RISCV_TEXT_ASSEMBLER_COMMON_H_
