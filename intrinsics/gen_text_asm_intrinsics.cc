/*
 * Copyright (C) 2019 The Android Open Source Project
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

#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/intrinsics/common/intrinsics_bindings.h"
#include "berberis/intrinsics/common/intrinsics_float.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/type_traits.h"
#include "berberis/intrinsics/verifier_assembler.h"

#include "text_assembler.h"

namespace berberis {

template <typename AsmCallInfo>
void GenerateOutputVariables(FILE* out, int indent);
template <typename AsmCallInfo>
void GenerateTemporaries(FILE* out, int indent);
template <typename AsmCallInfo>
void GenerateInShadows(FILE* out, int indent);
template <typename AsmCallInfo>
void GenerateAssemblerOuts(FILE* out, int indent);
template <typename AsmCallInfo>
void GenerateAssemblerIns(FILE* out,
                          int indent,
                          int* register_numbers,
                          bool need_gpr_macroassembler_scratch,
                          bool need_gpr_macroassembler_constants);
template <typename AsmCallInfo>
void GenerateOutShadows(FILE* out, int indent);
template <typename AsmCallInfo>
void GenerateElementsList(FILE* out,
                          int indent,
                          const std::string& prefix,
                          const std::string& suffix,
                          const std::vector<std::string>& elements);
template <typename AsmCallInfo, typename Arg>
constexpr bool NeedInputShadow(Arg arg);
template <typename AsmCallInfo, typename Arg>
constexpr bool NeedOutputShadow(Arg arg);

template <typename AsmCallInfo>
void GenerateFunctionHeader(FILE* out, int indent) {
  if (strchr(AsmCallInfo::kIntrinsic, '<')) {
    fprintf(out, "template <>\n");
  }
  std::string prefix;
  if constexpr (std::tuple_size_v<typename AsmCallInfo::OutputArguments> == 0) {
    prefix = "inline void " + std::string(AsmCallInfo::kIntrinsic) + "(";
  } else {
    const char* prefix_of_prefix = "inline std::tuple<";
    for (const char* type_name : AsmCallInfo::OutputArgumentsTypeNames) {
      prefix += prefix_of_prefix + std::string(type_name);
      prefix_of_prefix = ", ";
    }
    prefix += "> " + std::string(AsmCallInfo::kIntrinsic) + "(";
  }
  std::vector<std::string> ins;
  for (const char* type_name : AsmCallInfo::InputArgumentsTypeNames) {
    ins.push_back("[[maybe_unused]] " + std::string(type_name) + " in" +
                  std::to_string(ins.size()));
  }
  GenerateElementsList<AsmCallInfo>(out, indent, prefix, ") {", ins);
  fprintf(out,
          "  [[maybe_unused]]  alignas(berberis::config::kScratchAreaAlign)"
          " uint8_t scratch[berberis::config::kScratchAreaSize];\n");
  fprintf(out,
          "  [[maybe_unused]] auto& scratch2 ="
          " scratch[berberis::config::kScratchAreaSlotSize];\n");
}

template <typename AsmCallInfo>
constexpr void CallAssembler(MacroAssembler<TextAssembler>* as, int* register_numbers) {
  int arg_counter = 0;
  AsmCallInfo::ProcessBindings([&arg_counter, &as, register_numbers](auto arg) {
    using RegisterClass = typename decltype(arg)::RegisterClass;
    if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
      if constexpr (RegisterClass::kAsRegister != 'm') {
        if constexpr (RegisterClass::kIsImplicitReg) {
          if constexpr (RegisterClass::kAsRegister == 'a') {
            as->gpr_a =
                typename MacroAssembler<TextAssembler>::Register(register_numbers[arg_counter]);
          } else if constexpr (RegisterClass::kAsRegister == 'b') {
            as->gpr_b =
                typename MacroAssembler<TextAssembler>::Register(register_numbers[arg_counter]);
          } else if constexpr (RegisterClass::kAsRegister == 'c') {
            as->gpr_c =
                typename MacroAssembler<TextAssembler>::Register(register_numbers[arg_counter]);
          } else {
            static_assert(RegisterClass::kAsRegister == 'd');
            as->gpr_d =
                typename MacroAssembler<TextAssembler>::Register(register_numbers[arg_counter]);
          }
        }
      }
      ++arg_counter;
    }
  });
  as->gpr_macroassembler_constants = typename MacroAssembler<TextAssembler>::Register(arg_counter);
  arg_counter = 0;
  int scratch_counter = 0;
  std::apply(AsmCallInfo::kMacroInstruction,
             std::tuple_cat(
                 std::tuple<MacroAssembler<TextAssembler>&>{*as},
                 AsmCallInfo::MakeTuplefromBindings(
                     [&as, &arg_counter, &scratch_counter, register_numbers](auto arg) {
                       using RegisterClass = typename decltype(arg)::RegisterClass;
                       if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
                         if constexpr (RegisterClass::kAsRegister == 'm') {
                           if (scratch_counter == 0) {
                             as->gpr_macroassembler_scratch =
                                 typename MacroAssembler<TextAssembler>::Register(arg_counter++);
                           } else if (scratch_counter == 1) {
                             as->gpr_macroassembler_scratch2 =
                                 typename MacroAssembler<TextAssembler>::Register(arg_counter++);
                           } else {
                             FATAL("Only two scratch registers are supported for now");
                           }
                           // Note: as->gpr_scratch in combination with offset is treated by text
                           // assembler specially.  We rely on offset set here to be the same as
                           // scratch2 address in scratch buffer.
                           return std::tuple{typename MacroAssembler<TextAssembler>::Operand{
                               .base = as->gpr_scratch,
                               .disp = static_cast<int32_t>(config::kScratchAreaSlotSize *
                                                            scratch_counter++)}};
                         } else if constexpr (RegisterClass::kIsImplicitReg) {
                           ++arg_counter;
                           return std::tuple{};
                         } else {
                           return std::tuple{register_numbers[arg_counter++]};
                         }
                       } else {
                         return std::tuple{};
                       }
                     })));
}

template <typename AsmCallInfo>
void GenerateFunctionBody(FILE* out, int indent) {
  // Declare out variables.
  GenerateOutputVariables<AsmCallInfo>(out, indent);
  // Declare temporary variables.
  GenerateTemporaries<AsmCallInfo>(out, indent);
  // We need "shadow variables" for ins of types: Float32, Float64 and SIMD128Register.
  // This is because assembler does not accept these arguments for XMMRegisters and
  // we couldn't use "float"/"double" function arguments because if ABI issues.
  GenerateInShadows<AsmCallInfo>(out, indent);
  // Even if we don't pass any registers we need to allocate at least one element.
  int register_numbers[std::tuple_size_v<typename AsmCallInfo::Bindings> == 0
                           ? 1
                           : std::tuple_size_v<typename AsmCallInfo::Bindings>];
  // Assign numbers to registers - we need to pass them to assembler and then, later,
  // to Generator of Input Variable line.
  AssignRegisterNumbers<AsmCallInfo>(register_numbers);
  // Print opening line for asm call.
  if constexpr (AsmCallInfo::kSideEffects) {
    fprintf(out, "%*s__asm__ __volatile__(\n", indent, "");
  } else {
    fprintf(out, "%*s__asm__(\n", indent, "");
  }
  // Call text assembler to produce the body of an asm call.
  MacroAssembler<TextAssembler> as(indent, out);
  CallAssembler<AsmCallInfo>(&as, register_numbers);
  // Assembler instruction outs.
  GenerateAssemblerOuts<AsmCallInfo>(out, indent);
  // Assembler instruction ins.
  GenerateAssemblerIns<AsmCallInfo>(out,
                                    indent,
                                    register_numbers,
                                    as.need_gpr_macroassembler_scratch(),
                                    as.need_gpr_macroassembler_constants());
  // Close asm call.
  fprintf(out, "%*s);\n", indent, "");
  // Generate copies from shadows to outputs.
  GenerateOutShadows<AsmCallInfo>(out, indent);
  // Return value from function.
  if constexpr (std::tuple_size_v<typename AsmCallInfo::OutputArguments> > 0) {
    std::vector<std::string> outs;
    for (std::size_t id = 0; id < std::tuple_size_v<typename AsmCallInfo::OutputArguments>; ++id) {
      outs.push_back("out" + std::to_string(id));
    }
    GenerateElementsList<AsmCallInfo>(out, indent, "return {", "};", outs);
  }
}

template <typename AsmCallInfo>
void GenerateOutputVariables(FILE* out, int indent) {
  std::size_t id = 0;
  for (const char* type_name : AsmCallInfo::OutputArgumentsTypeNames) {
    fprintf(out, "%*s%s out%zd;\n", indent, "", type_name, id++);
  }
}

template <typename AsmCallInfo>
void GenerateTemporaries(FILE* out, int indent) {
  std::size_t id = 0;
  AsmCallInfo::ProcessBindings([out, &id, indent](auto arg) {
    using RegisterClass = typename decltype(arg)::RegisterClass;
    if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
      if constexpr (!HaveInput(arg.arg_info) && !HaveOutput(arg.arg_info)) {
        static_assert(
            std::is_same_v<typename decltype(arg)::Usage, intrinsics::bindings::Def> ||
            std::is_same_v<typename decltype(arg)::Usage, intrinsics::bindings::DefEarlyClobber>);
        fprintf(out,
                "%*s%s tmp%zd;\n",
                indent,
                "",
                TypeTraits<typename RegisterClass::Type>::kName,
                id++);
      }
    }
  });
}

template <typename AsmCallInfo>
void GenerateInShadows(FILE* out, int indent) {
  AsmCallInfo::ProcessBindings([out, indent](auto arg) {
    using RegisterClass = typename decltype(arg)::RegisterClass;
    if constexpr (RegisterClass::kAsRegister == 'm') {
      // Only temporary memory scratch area is supported.
      static_assert(!HaveInput(arg.arg_info) && !HaveOutput(arg.arg_info));
    } else if constexpr (RegisterClass::kAsRegister == 'r') {
      // TODO(b/138439904): remove when clang handling of 'r' constraint would be fixed.
      if constexpr (NeedInputShadow<AsmCallInfo>(arg)) {
        fprintf(out, "%2$*1$suint32_t in%3$d_shadow = in%3$d;\n", indent, "", arg.arg_info.from);
      }
      if constexpr (NeedOutputShadow<AsmCallInfo>(arg)) {
        fprintf(out, "%*suint32_t out%d_shadow;\n", indent, "", arg.arg_info.to);
      }
    } else if constexpr (RegisterClass::kAsRegister == 'x') {
      if constexpr (HaveInput(arg.arg_info)) {
        using Type = std::tuple_element_t<arg.arg_info.from, typename AsmCallInfo::InputArguments>;
        const char* type_name = TypeTraits<Type>::kName;
        const char* xmm_type_name;
        const char* expanded = "";
        // Types allowed for 'x' restriction are float, double and __m128/__m128i/__m128d
        // First two work for {,u}int32_t and {,u}int64_t, but small integer types must be expanded.
        if constexpr (std::is_integral_v<Type> && sizeof(Type) < sizeof(int32_t)) {
          fprintf(
              out, "%2$*1$suint32_t in%3$d_expanded = in%3$d;\n", indent, "", arg.arg_info.from);
          type_name = TypeTraits<uint32_t>::kName;
          xmm_type_name =
              TypeTraits<typename TypeTraits<typename TypeTraits<uint32_t>::Float>::Raw>::kName;
          expanded = "_expanded";
        } else if constexpr (std::is_integral_v<Type>) {
          // {,u}int32_t and {,u}int64_t have to be converted to float/double.
          xmm_type_name =
              TypeTraits<typename TypeTraits<typename TypeTraits<Type>::Float>::Raw>::kName;
        } else if constexpr (std::is_same_v<Type, intrinsics::Float16>) {
          // It's a bit strange that _Float16 is not accepted in XMM register, but it's also not
          // clear if it's a bug or not. Just use __m128 for now.
          fprintf(out, "%2$*1$s__m128 in%3$d_expanded;\n", indent, "", arg.arg_info.from);
          fprintf(out,
                  "%2$*1$smemcpy(&in%3$d_expanded, &in%3$d, sizeof(Float16));\n",
                  indent,
                  "",
                  arg.arg_info.from);
          type_name = "__m128";
          xmm_type_name = "__m128";
          expanded = "_expanded";
        } else {
          // Float32/Float64 can not be used, we need to use raw float/double.
          xmm_type_name = TypeTraits<typename TypeTraits<Type>::Raw>::kName;
        }
        fprintf(out, "%*s%s in%d_shadow;\n", indent, "", xmm_type_name, arg.arg_info.from);
        fprintf(out,
                "%*sstatic_assert(sizeof(%s) == sizeof(%s));\n",
                indent,
                "",
                type_name,
                xmm_type_name);
        // Note: it's not safe to use bit_cast here till we have std::bit_cast from C++20.
        // If optimizer wouldn't be enabled (e.g. if code is compiled with -O0) then bit_cast
        // would use %st on 32-bit platform which destroys NaNs.
        fprintf(out,
                "%2$*1$smemcpy(&in%3$d_shadow, &in%3$d%4$s, sizeof(%5$s));\n",
                indent,
                "",
                arg.arg_info.from,
                expanded,
                xmm_type_name);
      }
      if constexpr (HaveOutput(arg.arg_info)) {
        using Type = std::tuple_element_t<arg.arg_info.to, typename AsmCallInfo::OutputArguments>;
        const char* xmm_type_name;
        // {,u}int32_t and {,u}int64_t have to be converted to float/double.
        if constexpr (std::is_integral_v<Type>) {
          xmm_type_name =
              TypeTraits<typename TypeTraits<typename TypeTraits<Type>::Float>::Raw>::kName;
        } else if constexpr (std::is_same_v<Type, intrinsics::Float16>) {
          // It's a bit strange that _Float16 is not accepted in XMM register, but it's also not
          // clear if it's a bug or not. Just use __m128 for now.
          xmm_type_name = "__m128";
        } else {
          // Float32/Float64 can not be used, we need to use raw float/double.
          xmm_type_name = TypeTraits<typename TypeTraits<Type>::Raw>::kName;
        }
        fprintf(out, "%*s%s out%d_shadow;\n", indent, "", xmm_type_name, arg.arg_info.to);
      }
    }
  });
}

template <typename AsmCallInfo>
void GenerateAssemblerOuts(FILE* out, int indent) {
  std::vector<std::string> outs;
  int tmp_id = 0;
  AsmCallInfo::ProcessBindings([&outs, &tmp_id](auto arg) {
    using RegisterClass = typename decltype(arg)::RegisterClass;
    if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS> &&
                  !std::is_same_v<typename decltype(arg)::Usage, intrinsics::bindings::Use>) {
      std::string out = "\"=";
      if constexpr (std::is_same_v<typename decltype(arg)::Usage,
                                   intrinsics::bindings::DefEarlyClobber>) {
        out += "&";
      }
      out += RegisterClass::kAsRegister;
      if constexpr (HaveOutput(arg.arg_info)) {
        bool need_shadow = NeedOutputShadow<AsmCallInfo>(arg);
        out += "\"(out" + std::to_string(arg.arg_info.to) + (need_shadow ? "_shadow)" : ")");
      } else if constexpr (HaveInput(arg.arg_info)) {
        bool need_shadow = NeedInputShadow<AsmCallInfo>(arg);
        out += "\"(in" + std::to_string(arg.arg_info.from) + (need_shadow ? "_shadow)" : ")");
      } else {
        out += "\"(tmp" + std::to_string(tmp_id++) + ")";
      }
      outs.push_back(out);
    }
  });
  GenerateElementsList<AsmCallInfo>(out, indent, "  : ", "", outs);
}

template <typename AsmCallInfo>
void GenerateAssemblerIns(FILE* out,
                          int indent,
                          int* register_numbers,
                          bool need_gpr_macroassembler_scratch,
                          bool need_gpr_macroassembler_constants) {
  std::vector<std::string> ins;
  AsmCallInfo::ProcessBindings([&ins](auto arg) {
    using RegisterClass = typename decltype(arg)::RegisterClass;
    if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS> &&
                  std::is_same_v<typename decltype(arg)::Usage, intrinsics::bindings::Use>) {
      ins.push_back("\"" + std::string(1, RegisterClass::kAsRegister) + "\"(in" +
                    std::to_string(arg.arg_info.from) +
                    (NeedInputShadow<AsmCallInfo>(arg) ? "_shadow)" : ")"));
    }
  });
  if (need_gpr_macroassembler_scratch) {
    ins.push_back("\"m\"(scratch), \"m\"(scratch2)");
  }
  if (need_gpr_macroassembler_constants) {
    ins.push_back(
        "\"m\"(*reinterpret_cast<const char*>(&constants_pool::kBerberisMacroAssemblerConstants))");
  }
  int arg_counter = 0;
  AsmCallInfo::ProcessBindings([&ins, &arg_counter, register_numbers](auto arg) {
    using RegisterClass = typename decltype(arg)::RegisterClass;
    if constexpr (!std::is_same_v<RegisterClass, intrinsics::bindings::FLAGS>) {
      if constexpr (HaveInput(arg.arg_info) &&
                    !std::is_same_v<typename decltype(arg)::Usage, intrinsics::bindings::Use>) {
        ins.push_back("\"" + std::to_string(register_numbers[arg_counter]) + "\"(in" +
                      std::to_string(arg.arg_info.from) +
                      (NeedInputShadow<AsmCallInfo>(arg) ? "_shadow)" : ")"));
      }
      ++arg_counter;
    }
  });
  GenerateElementsList<AsmCallInfo>(out, indent, "  : ", "", ins);
}

template <typename AsmCallInfo>
void GenerateOutShadows(FILE* out, int indent) {
  AsmCallInfo::ProcessBindings([out, indent](auto arg) {
    using RegisterClass = typename decltype(arg)::RegisterClass;
    if constexpr (RegisterClass::kAsRegister == 'r') {
      // TODO(b/138439904): remove when clang handling of 'r' constraint would be fixed.
      if constexpr (HaveOutput(arg.arg_info)) {
        using Type = std::tuple_element_t<arg.arg_info.to, typename AsmCallInfo::OutputArguments>;
        if constexpr (sizeof(Type) == sizeof(uint8_t)) {
          fprintf(out, "%2$*1$sout%3$d = out%3$d_shadow;\n", indent, "", arg.arg_info.to);
        }
      }
    } else if constexpr (RegisterClass::kAsRegister == 'x') {
      if constexpr (HaveOutput(arg.arg_info)) {
        using Type = std::tuple_element_t<arg.arg_info.to, typename AsmCallInfo::OutputArguments>;
        const char* type_name = TypeTraits<Type>::kName;
        const char* xmm_type_name;
        // {,u}int32_t and {,u}int64_t have to be converted to float/double.
        if constexpr (std::is_integral_v<Type>) {
          xmm_type_name =
              TypeTraits<typename TypeTraits<typename TypeTraits<Type>::Float>::Raw>::kName;
        } else {
          // Float32/Float64 can not be used, we need to use raw float/double.
          xmm_type_name = TypeTraits<typename TypeTraits<Type>::Raw>::kName;
        }
        // It's a bit strange that _Float16 is not accepted in XMM register, but it's also not
        // clear if it's a bug or not. We use __m128 for now and that means size of types don't
        // match here.
        if constexpr (!std::is_same_v<Type, intrinsics::Float16>) {
          fprintf(out,
                  "%*sstatic_assert(sizeof(%s) == sizeof(%s));\n",
                  indent,
                  "",
                  type_name,
                  xmm_type_name);
        }
        // Note: it's not safe to use bit_cast here till we have std::bit_cast from C++20.
        // If optimizer wouldn't be enabled (e.g. if code is compiled with -O0) then bit_cast
        // would use %st on 32-bit platform which destroys NaNs.
        fprintf(out,
                "%2$*1$smemcpy(&out%3$d, &out%3$d_shadow, sizeof(%4$s));\n",
                indent,
                "",
                arg.arg_info.to,
                xmm_type_name);
      }
    }
  });
}

template <typename AsmCallInfo>
void GenerateElementsList(FILE* out,
                          int indent,
                          const std::string& prefix,
                          const std::string& suffix,
                          const std::vector<std::string>& elements) {
  std::size_t length = prefix.length() + suffix.length();
  if (elements.size() == 0) {
    fprintf(out, "%*s%s%s\n", indent, "", prefix.c_str(), suffix.c_str());
    return;
  }
  for (const auto& element : elements) {
    length += element.length() + 2;
  }
  for (const auto& element : elements) {
    if (&element == &elements[0]) {
      fprintf(out, "%*s%s%s", indent, "", prefix.c_str(), element.c_str());
    } else {
      if (length <= 102) {
        fprintf(out, ", %s", element.c_str());
      } else {
        fprintf(out, ",\n%*s%s", static_cast<int>(prefix.length()) + indent, "", element.c_str());
      }
    }
  }
  fprintf(out, "%s\n", suffix.c_str());
}

template <typename AsmCallInfo, typename Arg>
constexpr bool NeedInputShadow(Arg arg) {
  using RegisterClass = typename Arg::RegisterClass;
  // Without shadow clang silently converts 'r' restriction into 'q' restriction which
  // is wrong: if %ah or %bh is picked we would produce incorrect result here.
  // TODO(b/138439904): remove when clang handling of 'r' constraint would be fixed.
  if constexpr (RegisterClass::kAsRegister == 'r' && HaveInput(arg.arg_info)) {
    // Only 8-bit registers are special because each 16-bit registers include two of them
    // (%al/%ah, %cl/%ch, %dl/%dh, %bl/%bh).
    // Mix of 16-bit and 64-bit registers doesn't trigger bug in Clang.
    if constexpr (sizeof(std::tuple_element_t<arg.arg_info.from,
                                              typename AsmCallInfo::InputArguments>) ==
                  sizeof(uint8_t)) {
      return true;
    }
  } else if constexpr (RegisterClass::kAsRegister == 'x') {
    return true;
  }
  return false;
}

template <typename AsmCallInfo, typename Arg>
constexpr bool NeedOutputShadow(Arg arg) {
  using RegisterClass = typename Arg::RegisterClass;
  // Without shadow clang silently converts 'r' restriction into 'q' restriction which
  // is wrong: if %ah or %bh is picked we would produce incorrect result here.
  // TODO(b/138439904): remove when clang handling of 'r' constraint would be fixed.
  if constexpr (RegisterClass::kAsRegister == 'r' && HaveOutput(arg.arg_info)) {
    // Only 8-bit registers are special because each some 16-bit registers include two of
    // them (%al/%ah, %cl/%ch, %dl/%dh, %bl/%bh).
    // Mix of 16-bit and 64-bit registers don't trigger bug in Clang.
    if constexpr (sizeof(std::tuple_element_t<arg.arg_info.to,
                                              typename AsmCallInfo::OutputArguments>) ==
                  sizeof(uint8_t)) {
      return true;
    }
  } else if constexpr (RegisterClass::kAsRegister == 'x') {
    return true;
  }
  return false;
}

#include "text_asm_intrinsics_process_bindings-inl.h"

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

  // Verify that intrinsic's bindings correctly states that intrinsic uses/doesn't use FLAGS
  // register.
  bool expect_flags = false;
  CheckIntrinsicHasFlagsBinding<AsmCallInfo>(expect_flags);
  as.CheckFlagsBinding(expect_flags);
  as.CheckAppropriateDefEarlyClobbers();
  as.CheckLabelsAreBound();
}

constexpr bool VerifyTextAsmIntrinsics() {
  ProcessAllBindings<MacroAssembler<VerifierAssembler>::MacroAssemblers>(
      [](auto&& asm_call_generator) {
        using AsmCallInfo = std::decay_t<decltype(asm_call_generator)>;
        VerifyIntrinsic<AsmCallInfo>();
      });
  return true;
}

void GenerateTextAsmIntrinsics(FILE* out) {
  // Verifier assembler verifies that CPU vendor and SSE restrictions for intrinsics are defined
  // correctly.
  static_assert(VerifyTextAsmIntrinsics());

  // Note: nullptr means "NoCPUIDRestriction", other values are only assigned in one place below
  // since the code in this function mostly cares only about three cases:
  //   • There are no CPU restrictions.
  //   • There are CPU restrictions but they are the same as in previous case (which is error).
  //   • There are new CPU restrictions.
  const char* cpuid_restriction = nullptr /* NoCPUIDRestriction */;
  bool if_opened = false;
  std::string running_name;
  ProcessAllBindings<MacroAssembler<TextAssembler>::MacroAssemblers>(
      [&running_name, &if_opened, &cpuid_restriction, out](auto&& asm_call_generator) {
        using AsmCallInfo = std::decay_t<decltype(asm_call_generator)>;
        std::string full_name = std::string(asm_call_generator.kIntrinsic,
                                            std::strlen(asm_call_generator.kIntrinsic) - 1) +
                                ", kUseCppImplementation>";
        if (size_t arguments_count = std::tuple_size_v<typename AsmCallInfo::InputArguments>) {
          full_name += "(in0";
          for (size_t i = 1; i < arguments_count; ++i) {
            full_name += ", in" + std::to_string(i);
          }
          full_name += ")";
        } else {
          full_name += "()";
        }
        if (full_name != running_name) {
          if (if_opened) {
            if (cpuid_restriction) {
              fprintf(out, "  } else {\n    return %s;\n", running_name.c_str());
              cpuid_restriction = nullptr /* NoCPUIDRestriction */;
            }
            if_opened = false;
            fprintf(out, "  }\n");
          }
          // Final line of function.
          if (!running_name.empty()) {
            fprintf(out, "};\n\n");
          }
          GenerateFunctionHeader<AsmCallInfo>(out, 0);
          running_name = full_name;
        }
        using CPUIDRestriction = AsmCallInfo::CPUIDRestriction;
        // Note: this series of "if constexpr" expressions is the only place where cpuid_restriction
        // may get a concrete non-zero value;
        if constexpr (std::is_same_v<CPUIDRestriction, intrinsics::bindings::NoCPUIDRestriction>) {
          if (cpuid_restriction) {
            fprintf(out, "  } else {\n");
            cpuid_restriction = nullptr;
          }
        } else {
          if (if_opened) {
            fprintf(out, "  } else if (");
          } else {
            fprintf(out, "  if (");
            if_opened = true;
          }
          cpuid_restriction = TextAssembler::kCPUIDRestrictionString<CPUIDRestriction>;
          fprintf(out, "%s) {\n", cpuid_restriction);
        }
        GenerateFunctionBody<AsmCallInfo>(out, 2 + 2 * if_opened);
      });
  if (if_opened) {
    fprintf(out, "  }\n");
  }
  // Final line of function.
  if (!running_name.empty()) {
    fprintf(out, "};\n\n");
  }
}

}  // namespace berberis

int main(int argc, char* argv[]) {
  FILE* out = argc > 1 ? fopen(argv[1], "w") : stdout;
  fprintf(out,
          R"STRING(
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

// This file automatically generated by gen_text_asm_intrinsics.cc
// DO NOT EDIT!

#ifndef %2$s_%3$s_INTRINSICS_INTRINSICS_H_
#define %2$s_%3$s_INTRINSICS_INTRINSICS_H_

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include "berberis/base/config.h"
#include "berberis/runtime_primitives/platform.h"
#include "%3$s/intrinsics/%1$s_to_all/intrinsics.h"
#include "%3$s/intrinsics/vector_intrinsics.h"

namespace berberis::constants_pool {

struct MacroAssemblerConstants;

extern const MacroAssemblerConstants kBerberisMacroAssemblerConstants
    __attribute__((visibility("hidden")));

}  // namespace berberis::constants_pool

namespace %3$s {

namespace constants_pool {

%4$s

}  // namespace constants_pool

namespace intrinsics {
)STRING",
          berberis::TextAssembler::kArchName,
          berberis::TextAssembler::kArchGuard,
          berberis::TextAssembler::kNamespaceName,
          strcmp(berberis::TextAssembler::kNamespaceName, "berberis")
              ? "using berberis::constants_pool::kBerberisMacroAssemblerConstants;"
              : "");

  berberis::GenerateTextAsmIntrinsics(out);
  berberis::MakeExtraGuestFunctions(out);

  fprintf(out,
          R"STRING(
}  // namespace intrinsics

}  // namespace %2$s

#endif /* %1$s_%2$s_INTRINSICS_INTRINSICS_H_ */
)STRING",
          berberis::TextAssembler::kArchGuard,
          berberis::TextAssembler::kNamespaceName);

  fclose(out);
  return 0;
}
