/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_x86_64_TEXT_ASSEMBLER_COMMON_H_
#define BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_x86_64_TEXT_ASSEMBLER_COMMON_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>

#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/all_to_x86_32_or_x86_64/intrinsics_bindings.h"

namespace berberis {

namespace constants_pool {

// Note: kBerberisMacroAssemblerConstantsRelocated is the same as original,
// unrelocated version in 32-bit world.  But in 64-bit world it's copy on the first 2GiB.
//
// Our builder could be built as 64-bit binary thus we must not mix them.
//
// Note: we have CHECK_*_LAYOUT tests in macro_assembler_common_x86.cc to make sure
// offsets produced by 64-bit builder are usable in 32-bit libberberis.so

extern const int32_t kBerberisMacroAssemblerConstantsRelocated;

inline int32_t GetOffset(int32_t address) {
  return address - constants_pool::kBerberisMacroAssemblerConstantsRelocated;
}

}  // namespace constants_pool

namespace x86_32_and_x86_64 {

template <typename DerivedAssemblerType>
class TextAssembler {
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

    template <typename MacroAssembler>
    friend std::string ToGasArgument(const Label& label, MacroAssembler*) {
      return std::to_string(label.id) + (label.bound ? "b" : "f");
    }
  };

  struct Operand;

  class Register {
   public:
    constexpr Register(int arg_no) : arg_no_(arg_no) {}
    int arg_no() const {
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

    template <typename MacroAssembler>
    friend const std::string ToGasArgument(const X87Register& reg, MacroAssembler*) {
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

    template <typename MacroAssembler>
    friend const std::string ToGasArgument(const SIMDRegister& reg, MacroAssembler*) {
      if constexpr (kBits == 128) {
        return "%x" + std::to_string(reg.arg_no());
      } else if constexpr (kBits == 256) {
        return "%t" + std::to_string(reg.arg_no());
      } else if constexpr (kBits == 512) {
        return "%g" + std::to_string(reg.arg_no());
      } else {
        static_assert(kDependentValueFalse<kBits>);
      }
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

    template <typename MacroAssembler>
    friend const std::string ToGasArgument(const Operand& op, MacroAssembler* as) {
      std::string result{};
      if (op.base.arg_no_ == Register::kNoRegister and op.index.arg_no_ == Register::kNoRegister) {
        as->need_gpr_macroassembler_constants_ = true;
        result =
            std::to_string(constants_pool::GetOffset(op.disp)) + " + " +
            ToGasArgument(
                typename DerivedAssemblerType::RegisterDefaultBit(as->gpr_macroassembler_constants),
                as);
      } else if (op.base.arg_no_ == Register::kScratchPointer) {
        CHECK(op.index.arg_no_ == Register::kNoRegister);
        // Only support two pointers to scratch area for now.
        if (op.disp == 0) {
          result = '%' + std::to_string(as->gpr_macroassembler_scratch.arg_no());
        } else if (op.disp == config::kScratchAreaSlotSize) {
          result = '%' + std::to_string(as->gpr_macroassembler_scratch2.arg_no());
        } else {
          FATAL("Only two scratch registers are supported for now");
        }
      } else {
        if (op.base.arg_no_ != Register::kNoRegister) {
          result = ToGasArgument(typename DerivedAssemblerType::RegisterDefaultBit(op.base), as);
        }
        if (op.index.arg_no_ != Register::kNoRegister) {
          result += ',' +
                    ToGasArgument(typename DerivedAssemblerType::RegisterDefaultBit(op.index), as) +
                    ',' + std::to_string(1 << op.scale);
        }
        result = '(' + result + ')';
        if (op.disp) {
          result = std::to_string(op.disp) + result;
        }
      }
      return result;
    }
  };

  TextAssembler(int indent, FILE* out) : indent_(indent), out_(out) {}

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

  // Translate CPU restrictions into string.
  template <typename CPUIDRestriction>
  static constexpr const char* kCPUIDRestrictionString =
      DerivedAssemblerType::template CPUIDRestrictionToString<CPUIDRestriction>();

// Instructions.
#include "gen_text_assembler_common_x86-inl.h"  // NOLINT generated file

 protected:
  template <typename CPUIDRestriction>
  static constexpr const char* CPUIDRestrictionToString() {
    if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::NoCPUIDRestriction>) {
      return nullptr;
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::IsAuthenticAMD>) {
      return "host_platform::kIsAuthenticAMD";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasAES>) {
      return "host_platform::kHasAES";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasAESAVX>) {
      return "host_platform::kHasAES && host_platform::kHasAVX";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasAVX>) {
      return "host_platform::kHasAVX";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasBMI>) {
      return "host_platform::kHasBMI";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasF16C>) {
      return "host_platform::kHasF16C";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasCLMUL>) {
      return "host_platform::kHasCLMUL";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasCLMULAVX>) {
      return "host_platform::kHasCLMUL && host_platform::kHasAVX";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasFMA>) {
      return "host_platform::kHasFMA";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasFMA4>) {
      return "host_platform::kHasFMA4";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasLZCNT>) {
      return "host_platform::kHasLZCNT";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasPOPCNT>) {
      return "host_platform::kHasPOPCNT";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSE3>) {
      return "host_platform::kHasSSE3";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSSE3>) {
      return "host_platform::kHasSSSE3";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSE4_1>) {
      return "host_platform::kHasSSE4_1";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSE4_2>) {
      return "host_platform::kHasSSE4_2";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasSSSE3>) {
      return "host_platform::kHasSSSE3";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasVAES>) {
      return "host_platform::kHasVAES";
    } else if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::HasVPCLMULQD>) {
      return "host_platform::kHasVPCLMULQD";
    } else if constexpr (std::is_same_v<CPUIDRestriction,
                                        intrinsics::bindings::HasCustomCapability>) {
      return "host_platform::kHasCustomCapability";
    } else {
      static_assert(kDependentTypeFalse<CPUIDRestriction>);
    }
  }

  bool need_gpr_macroassembler_constants_ = false;
  bool need_gpr_macroassembler_scratch_ = false;

  template <const char* kSpPrefix, char kRegisterPrefix>
  class RegisterTemplate {
   public:
    explicit RegisterTemplate(Register reg) : reg_(reg) {}

    template <typename MacroAssembler>
    friend const std::string ToGasArgument(const RegisterTemplate& reg, MacroAssembler*) {
      if (reg.reg_.arg_no() == Register::kStackPointer) {
        return kSpPrefix;
      } else {
        if (kRegisterPrefix) {
          return std::string({'%', kRegisterPrefix}) + std::to_string(reg.reg_.arg_no());
        } else {
          return '%' + std::to_string(reg.reg_.arg_no());
        }
      }
    }

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

  void SetRequiredFeatureAESAVX() {}

  void SetRequiredFeatureAES() {}

  void SetRequiredFeatureAVX() {}

  void SetRequiredFeatureAVX2() {}

  void SetRequiredFeatureBMI() {}

  void SetRequiredFeatureBMI2() {}

  void SetRequiredFeatureCLMULAVX() {}

  void SetRequiredFeatureCLMUL() {}

  void SetRequiredFeatureF16C() {}

  void SetRequiredFeatureFMA() {}

  void SetRequiredFeatureFMA4() {}

  void SetRequiredFeatureLZCNT() {}

  void SetRequiredFeaturePOPCNT() {}

  void SetRequiredFeatureSSE3() {}

  void SetRequiredFeatureSSSE3() {}

  void SetRequiredFeatureSSE4_1() {}

  void SetRequiredFeatureSSE4_2() {}

  void SetRequiredFeatureVAES() {}

  void SetRequiredFeatureVPCLMULQD() {}

  void SetHasCustomCapability() {}

  template <typename... Args>
  void Instruction(const char* name, Condition cond, const Args&... args);

  template <typename... Args>
  void Instruction(const char* name, const Args&... args);

  void EmitString() {}

  void EmitString(const std::string& s) { fprintf(out_, "%s", s.c_str()); }

  template <typename... Args>
  void EmitString(const std::string& s, const Args&... args) {
    EmitString(args...);
    fprintf(out_, ", %s", s.c_str());
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
  if (strcmp(name, "Cmovw") == 0 || strcmp(name, "Cmovl") == 0 || strcmp(name, "Cmovq") == 0) {
    strcpy(name_with_condition, "Cmov");
  } else if (strcmp(name, "Jcc") == 0) {
    strcpy(name_with_condition, "J");
  } else {
    CHECK(strcmp(name, "Setcc") == 0);
    strcpy(name_with_condition, "Set");
  }
  switch (cond) {
    case Condition::kOverflow:
      strcat(name_with_condition, "o");
      break;
    case Condition::kNoOverflow:
      strcat(name_with_condition, "no");
      break;
    case Condition::kBelow:
      strcat(name_with_condition, "b");
      break;
    case Condition::kAboveEqual:
      strcat(name_with_condition, "ae");
      break;
    case Condition::kEqual:
      strcat(name_with_condition, "e");
      break;
    case Condition::kNotEqual:
      strcat(name_with_condition, "ne");
      break;
    case Condition::kBelowEqual:
      strcat(name_with_condition, "be");
      break;
    case Condition::kAbove:
      strcat(name_with_condition, "a");
      break;
    case Condition::kNegative:
      strcat(name_with_condition, "s");
      break;
    case Condition::kPositiveOrZero:
      strcat(name_with_condition, "ns");
      break;
    case Condition::kParityEven:
      strcat(name_with_condition, "p");
      break;
    case Condition::kParityOdd:
      strcat(name_with_condition, "np");
      break;
    case Condition::kLess:
      strcat(name_with_condition, "l");
      break;
    case Condition::kGreaterEqual:
      strcat(name_with_condition, "ge");
      break;
    case Condition::kLessEqual:
      strcat(name_with_condition, "le");
      break;
    case Condition::kGreater:
      strcat(name_with_condition, "g");
      break;
  }
  Instruction(name_with_condition, args...);
}

template <typename DerivedAssemblerType>
template <typename... Args>
inline void TextAssembler<DerivedAssemblerType>::Instruction(const char* name,
                                                             const Args&... args) {
  for (auto it : std::array<std::tuple<const char*, const char*>, 22>{
           {// Note: SSE doesn't include simple register-to-register move instruction.
            // You are supposed to use one of half-dozen variants depending on what you
            // are doing.
            //
            // Pseudoinstructions with embedded "lock" prefix.
            {"Lock Xaddb", "Lock; Xaddb"},
            {"Lock Xaddw", "Lock; Xaddw"},
            {"Lock Xaddl", "Lock; Xaddl"},
            {"Lock Xaddq", "Lock; Xaddq"},
            {"Lock CmpXchg8b", "Lock; CmpXchg8b"},
            {"Lock CmpXchg16b", "Lock; CmpXchg16b"},
            {"Lock CmpXchgb", "Lock; CmpXchgb"},
            {"Lock CmpXchgl", "Lock; CmpXchgl"},
            {"Lock CmpXchgq", "Lock; CmpXchgq"},
            {"Lock CmpXchgw", "Lock; CmpXchgw"},
            // Our assembler has Pmov instruction which is supposed to pick the best
            // option - but currently we just map Pmov to Movaps.
            {"Pmov", "Movaps"},
            // These instructions use different names in our assembler than in GNU AS.
            {"Movdq", "Movaps"},
            {"Movsxbl", "Movsbl"},
            {"Movsxbq", "Movsbq"},
            {"Movsxwl", "Movswl"},
            {"Movsxwq", "Movswq"},
            {"Movsxlq", "Movslq"},
            {"Movzxbl", "Movzbl"},
            {"Movzxbq", "Movzbq"},
            {"Movzxwl", "Movzwl"},
            {"Movzxwq", "Movzwq"},
            {"Movzxlq", "Movzlq"}}}) {
    if (strcmp(name, std::get<0>(it)) == 0) {
      name = std::get<1>(it);
      break;
    }
  }

  int name_length = strlen(name);
  auto cl_register = "";
  if (name_length > 4 && strcmp(name + (name_length - 4), "ByCl") == 0) {
    name_length -= 4;
    cl_register = " %%cl,";
  }

  fprintf(out_, "%*s\"%.*s%s ", indent_ + 2, "", name_length, name, cl_register);
  EmitString(ToGasArgument(args, this)...);
  fprintf(out_, "\\n\"\n");
}

}  // namespace x86_32_and_x86_64

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_x86_64_TEXT_ASSEMBLER_COMMON_H_
