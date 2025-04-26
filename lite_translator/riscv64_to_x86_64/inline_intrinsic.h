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

#ifndef BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_INLINE_INTRINSIC_H_
#define BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_INLINE_INTRINSIC_H_

#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>

#include "berberis/assembler/x86_64.h"
#include "berberis/base/checks.h"
#include "berberis/base/dependent_false.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/intrinsics/guest_cpu_flags.h"
#include "berberis/intrinsics/intrinsics_process_bindings.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/runtime_primitives/platform.h"

namespace berberis::inline_intrinsic {

template <auto kFunction,
          typename RegAlloc,
          typename SIMDRegAlloc,
          typename AssemblerResType,
          typename... AssemblerArgType>
bool TryInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                        RegAlloc&& reg_alloc,
                        SIMDRegAlloc&& simd_reg_alloc,
                        AssemblerResType result,
                        AssemblerArgType... args);

template <auto kFunc>
class InlineIntrinsic {
 public:
  template <typename RegAlloc, typename SIMDRegAlloc, typename ResType, typename... ArgType>
  static bool TryInlineWithHostRounding(MacroAssembler<x86_64::Assembler>& as,
                                        RegAlloc&& reg_alloc,
                                        SIMDRegAlloc&& simd_reg_alloc,
                                        ResType result,
                                        ArgType... args) {
    std::tuple args_tuple = std::make_tuple(args...);
    if constexpr (IsTagEq<&intrinsics::FMul<intrinsics::Float64>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FMulHostRounding<intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FMul<intrinsics::Float32>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FMulHostRounding<intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FAdd<intrinsics::Float64>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FAddHostRounding<intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FAdd<intrinsics::Float32>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FAddHostRounding<intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FSub<intrinsics::Float64>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FSubHostRounding<intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FSub<intrinsics::Float32>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FSubHostRounding<intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FDiv<intrinsics::Float64>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FDivHostRounding<intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FDiv<intrinsics::Float32>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FDivHostRounding<intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FCvtFloatToInteger<int64_t, intrinsics::Float64>>) {
      auto [rm, frm, src] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<
          &intrinsics::FCvtFloatToIntegerHostRounding<int64_t, intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src);
    } else if constexpr (IsTagEq<&intrinsics::FCvtFloatToInteger<int64_t, intrinsics::Float32>>) {
      auto [rm, frm, src] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<
          &intrinsics::FCvtFloatToIntegerHostRounding<int64_t, intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src);
    } else if constexpr (IsTagEq<&intrinsics::FCvtFloatToInteger<int32_t, intrinsics::Float64>>) {
      auto [rm, frm, src] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<
          &intrinsics::FCvtFloatToIntegerHostRounding<int32_t, intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src);
    } else if constexpr (IsTagEq<&intrinsics::FCvtFloatToInteger<int32_t, intrinsics::Float32>>) {
      auto [rm, frm, src] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<
          &intrinsics::FCvtFloatToIntegerHostRounding<int32_t, intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src);
    }
    return false;
  }

 private:
  template <auto kFunction>
  class FunctionCompareTag;

  template <auto kOtherFunction>
  static constexpr bool IsTagEq =
      std::is_same_v<FunctionCompareTag<kFunc>, FunctionCompareTag<kOtherFunction>>;
};

template <typename format, typename DestType, typename SrcType>
auto Mov(MacroAssembler<x86_64::Assembler>& as, DestType dest, SrcType src)
    -> decltype(std::declval<MacroAssembler<x86_64::Assembler>>()
                    .Mov<format>(std::declval<DestType>(), std::declval<SrcType>())) {
  if constexpr (std::is_integral_v<format>) {
    return as.template Mov<format>(dest, src);
  } else if (host_platform::kHasAVX) {
    return as.template Vmov<format>(dest, src);
  } else {
    return as.template Mov<format>(dest, src);
  }
}

template <typename format, typename DestType, typename SrcType>
auto Mov(MacroAssembler<x86_64::Assembler>& as, DestType dest, SrcType src)
    -> decltype(std::declval<MacroAssembler<x86_64::Assembler>>()
                    .Movs<format>(std::declval<DestType>(), std::declval<SrcType>())) {
  if (host_platform::kHasAVX) {
    if constexpr (std::is_same_v<DestType, MacroAssembler<x86_64::Assembler>::XMMRegister> &&
                  std::is_same_v<SrcType, MacroAssembler<x86_64::Assembler>::XMMRegister>) {
      return as.template Vmovs<format>(dest, dest, src);
    } else {
      return as.template Vmovs<format>(dest, src);
    }
  } else {
    return as.template Movs<format>(dest, src);
  }
}

template <auto kFunction,
          typename RegAlloc,
          typename SIMDRegAlloc,
          typename AssemblerResType,
          typename... AssemblerArgType>
class TryBindingBasedInlineIntrinsic {
  template <auto kFunctionForFriend,
            typename RegAllocForFriend,
            typename SIMDRegAllocForFriend,
            typename AssemblerResTypeForFriend,
            typename... AssemblerArgTypeForFriend>
  friend bool TryInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                                 RegAllocForFriend&& reg_alloc,
                                 SIMDRegAllocForFriend&& simd_reg_alloc,
                                 AssemblerResTypeForFriend result,
                                 AssemblerArgTypeForFriend... args);
  template <auto kFunc,
            typename MacroAssembler,
            typename Result,
            typename Callback,
            typename... Args>
  friend constexpr Result intrinsics::bindings::ProcessBindings(Callback callback,
                                                                Result def_result,
                                                                Args&&... args);
  template <auto kIntrinsicTemplateName,
            auto kMacroInstructionTemplateName,
            auto kMnemo,
            auto GetOpcode,
            typename kCPUIDRestrictionTemplateValue,
            typename kPreciseNanOperationsHandlingTemplateValue,
            bool kSideEffectsTemplateValue,
            typename... Types>
  friend class intrinsics::bindings::IntrinsicBindingInfo;

  TryBindingBasedInlineIntrinsic() = delete;
  TryBindingBasedInlineIntrinsic(const TryBindingBasedInlineIntrinsic&) = delete;
  TryBindingBasedInlineIntrinsic(TryBindingBasedInlineIntrinsic&&) = default;
  TryBindingBasedInlineIntrinsic& operator=(const TryBindingBasedInlineIntrinsic&) = delete;
  TryBindingBasedInlineIntrinsic& operator=(TryBindingBasedInlineIntrinsic&&) = default;

  TryBindingBasedInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                                 RegAlloc& reg_alloc,
                                 SIMDRegAlloc& simd_reg_alloc,
                                 AssemblerResType result,
                                 AssemblerArgType... args)
      : as_(as),
        reg_alloc_(reg_alloc),
        simd_reg_alloc_(simd_reg_alloc),
        result_{result},
        input_args_(std::tuple{args...}),
        success_(intrinsics::bindings::ProcessBindings<
                 kFunction,
                 typename MacroAssembler<x86_64::Assembler>::MacroAssemblers,
                 bool,
                 TryBindingBasedInlineIntrinsic&>(*this, false)) {}
  operator bool() { return success_; }

  template <typename IntrinsicBindingInfo>
  std::optional<bool> /*ProcessBindingsClient*/ operator()(IntrinsicBindingInfo asm_call_info) {
    static_assert(
        std::is_same_v<decltype(kFunction), typename IntrinsicBindingInfo::IntrinsicType>);
    static_assert(std::is_same_v<typename IntrinsicBindingInfo::PreciseNanOperationsHandling,
                                 intrinsics::bindings::NoNansOperation>);
    using CPUIDRestriction = IntrinsicBindingInfo::CPUIDRestriction;
    if constexpr (std::is_same_v<CPUIDRestriction, machine_insn_info::HasAVX>) {
      if (!host_platform::kHasAVX) {
        return {};
      }
    } else if constexpr (std::is_same_v<CPUIDRestriction, machine_insn_info::HasBMI>) {
      if (!host_platform::kHasBMI) {
        return {};
      }
    } else if constexpr (std::is_same_v<CPUIDRestriction, machine_insn_info::HasFMA>) {
      if (!host_platform::kHasFMA) {
        return {};
      }
    } else if constexpr (std::is_same_v<CPUIDRestriction, machine_insn_info::HasLZCNT>) {
      if (!host_platform::kHasLZCNT) {
        return {};
      }
    } else if constexpr (std::is_same_v<CPUIDRestriction, machine_insn_info::HasPOPCNT>) {
      if (!host_platform::kHasPOPCNT) {
        return {};
      }
    } else if constexpr (std::is_same_v<CPUIDRestriction, machine_insn_info::NoCPUIDRestriction>) {
      // No restrictions. Do nothing.
    } else {
      static_assert(kDependentValueFalse<IntrinsicBindingInfo::kCPUIDRestriction>);
    }
    std::apply(
        IntrinsicBindingInfo::kMacroInstruction,
        std::tuple_cat(
            std::tuple<MacroAssembler<x86_64::Assembler>&>{as_},
            IntrinsicBindingInfo::template MakeTuplefromBindings<TryBindingBasedInlineIntrinsic&>(
                *this, asm_call_info)));
    if constexpr (std::tuple_size_v<typename IntrinsicBindingInfo::OutputArguments> == 0) {
      // No return value. Do nothing.
    } else if constexpr (std::tuple_size_v<typename IntrinsicBindingInfo::OutputArguments> == 1) {
      using ReturnType = std::tuple_element_t<0, typename IntrinsicBindingInfo::OutputArguments>;
      if constexpr (std::is_integral_v<ReturnType>) {
        if (result_reg_ != x86_64::Assembler::no_register) {
          Mov<ReturnType>(as_, result_, result_reg_);
          CHECK_EQ(result_xmm_reg_, x86_64::Assembler::no_xmm_register);
        } else if (result_xmm_reg_ != x86_64::Assembler::no_xmm_register) {
          Mov<typename TypeTraits<ReturnType>::Float>(as_, result_, result_xmm_reg_);
          CHECK_EQ(result_reg_, x86_64::Assembler::no_register);
        }
      } else {
        CHECK_EQ(result_reg_, x86_64::Assembler::no_register);
        CHECK_EQ(result_xmm_reg_, x86_64::Assembler::no_xmm_register);
      }
      if constexpr (std::is_integral_v<ReturnType> && sizeof(ReturnType) < sizeof(std::int32_t)) {
        // Don't handle these types just yet. We are not sure how to expand them and there
        // are no examples.
        static_assert(kDependentTypeFalse<ReturnType>);
      }
      if constexpr (std::is_same_v<ReturnType, int32_t> || std::is_same_v<ReturnType, uint32_t>) {
        // Expans 32 bit values as signed. Even if actual results are processed as unsigned!
        as_.Expand<int64_t, std::make_signed_t<ReturnType>>(result_, result_);
      } else if constexpr (std::is_integral_v<ReturnType> &&
                           sizeof(ReturnType) == sizeof(std::int64_t)) {
        // Do nothing, we have already produced expanded value.
      } else if constexpr (std::is_same_v<ReturnType, intrinsics::Float32> ||
                           std::is_same_v<ReturnType, intrinsics::Float64>) {
        // Do nothing, NaN boxing is handled by semantics player.
      } else {
        static_assert(kDependentTypeFalse<ReturnType>);
      }
    } else {
      static_assert(kDependentTypeFalse<typename IntrinsicBindingInfo::OutputArguments>);
    }
    return {true};
  }

  template <typename ArgBinding, typename OperandInfo, typename IntrinsicBindingInfo>
  auto /*MakeTuplefromBindingsClient*/ operator()(ArgTraits<ArgBinding, OperandInfo>,
                                                  IntrinsicBindingInfo) {
    static constexpr const auto& arg_info = ArgTraits<ArgBinding, OperandInfo>::arg_info;
    if constexpr (arg_info.arg_type == ArgInfo::IMM_ARG) {
      return ProcessArgInput<ArgBinding, OperandInfo, IntrinsicBindingInfo>(reg_alloc_);
    } else {
      using RegisterClass = typename ArgTraits<ArgBinding, OperandInfo>::RegisterClass;
      if constexpr (RegisterClass::kAsRegister == 'x') {
        return ProcessArgInput<ArgBinding, OperandInfo, IntrinsicBindingInfo>(simd_reg_alloc_);
      } else {
        return ProcessArgInput<ArgBinding, OperandInfo, IntrinsicBindingInfo>(reg_alloc_);
      }
    }
  }

  template <typename ArgBinding,
            typename OperandInfo,
            typename IntrinsicBindingInfo,
            typename RegAllocForArg>
  auto ProcessArgInput(RegAllocForArg&& reg_alloc) {
    static constexpr const auto& arg_info = ArgTraits<ArgBinding, OperandInfo>::arg_info;
    if constexpr (arg_info.arg_type == ArgInfo::IMM_ARG) {
      return std::tuple{std::get<arg_info.from>(input_args_)};
    } else {
      using RegisterClass = typename ArgTraits<ArgBinding, OperandInfo>::RegisterClass;
      static constexpr auto kUsage = ArgTraits<ArgBinding, OperandInfo>::kUsage;
      if constexpr (arg_info.arg_type == ArgInfo::IN_ARG) {
        using Type =
            std::tuple_element_t<arg_info.from, typename IntrinsicBindingInfo::InputArguments>;
        if constexpr (RegisterClass::kAsRegister == 'x' && std::is_integral_v<Type>) {
          auto reg = reg_alloc();
          Mov<typename TypeTraits<int64_t>::Float>(as_, reg, std::get<arg_info.from>(input_args_));
          return std::tuple{reg};
        } else {
          static_assert(kUsage == machine_insn_info::kUse);
          static_assert(!RegisterClass::kIsImplicitReg);
          return std::tuple{std::get<arg_info.from>(input_args_)};
        }
      } else if constexpr (arg_info.arg_type == ArgInfo::IN_OUT_ARG) {
        using Type =
            std::tuple_element_t<arg_info.from, typename IntrinsicBindingInfo::InputArguments>;
        static_assert(kUsage == machine_insn_info::kUseDef);
        static_assert(!RegisterClass::kIsImplicitReg);
        if constexpr (RegisterClass::kAsRegister == 'x' && std::is_integral_v<Type>) {
          static_assert(std::is_integral_v<
                        std::tuple_element_t<arg_info.to,
                                             typename IntrinsicBindingInfo::OutputArguments>>);
          CHECK_EQ(result_xmm_reg_, x86_64::Assembler::no_xmm_register);
          result_xmm_reg_ = reg_alloc();
          Mov<typename TypeTraits<int64_t>::Float>(
              as_, result_xmm_reg_, std::get<arg_info.from>(input_args_));
          return std::tuple{result_xmm_reg_};
        } else {
          Mov<std::tuple_element_t<arg_info.from, typename IntrinsicBindingInfo::InputArguments>>(
              as_, result_, std::get<arg_info.from>(input_args_));
          return std::tuple{result_};
        }
      } else if constexpr (arg_info.arg_type == ArgInfo::IN_TMP_ARG) {
        if constexpr (RegisterClass::kAsRegister == 'c') {
          Mov<std::tuple_element_t<arg_info.from, typename IntrinsicBindingInfo::InputArguments>>(
              as_, as_.rcx, std::get<arg_info.from>(input_args_));
          return std::tuple{};
        } else if constexpr (RegisterClass::kAsRegister == 'a') {
          Mov<std::tuple_element_t<arg_info.from, typename IntrinsicBindingInfo::InputArguments>>(
              as_, as_.rax, std::get<arg_info.from>(input_args_));
          return std::tuple{};
        } else {
          static_assert(kUsage == machine_insn_info::kUseDef);
          static_assert(!RegisterClass::kIsImplicitReg);
          auto reg = reg_alloc();
          Mov<std::tuple_element_t<arg_info.from, typename IntrinsicBindingInfo::InputArguments>>(
              as_, reg, std::get<arg_info.from>(input_args_));
          return std::tuple{reg};
        }
      } else if constexpr (arg_info.arg_type == ArgInfo::IN_OUT_TMP_ARG) {
        using Type =
            std::tuple_element_t<arg_info.from, typename IntrinsicBindingInfo::InputArguments>;
        static_assert(kUsage == machine_insn_info::kUseDef);
        static_assert(RegisterClass::kIsImplicitReg);
        if constexpr (RegisterClass::kAsRegister == 'a') {
          CHECK_EQ(result_reg_, x86_64::Assembler::no_register);
          Mov<Type>(as_, as_.rax, std::get<arg_info.from>(input_args_));
          result_reg_ = as_.rax;
          return std::tuple{};
        } else {
          static_assert(kDependentValueFalse<arg_info.arg_type>);
        }
      } else if constexpr (arg_info.arg_type == ArgInfo::OUT_ARG) {
        using Type =
            std::tuple_element_t<arg_info.to, typename IntrinsicBindingInfo::OutputArguments>;
        static_assert(kUsage == machine_insn_info::kDef ||
                      kUsage == machine_insn_info::kDefEarlyClobber);
        if constexpr (RegisterClass::kAsRegister == 'a') {
          CHECK_EQ(result_reg_, x86_64::Assembler::no_register);
          result_reg_ = as_.rax;
          return std::tuple{};
        } else if constexpr (RegisterClass::kAsRegister == 'c') {
          CHECK_EQ(result_reg_, x86_64::Assembler::no_register);
          result_reg_ = as_.rcx;
          return std::tuple{};
        } else {
          static_assert(!RegisterClass::kIsImplicitReg);
          if constexpr (RegisterClass::kAsRegister == 'x' && std::is_integral_v<Type>) {
            CHECK_EQ(result_xmm_reg_, x86_64::Assembler::no_xmm_register);
            result_xmm_reg_ = reg_alloc();
            return std::tuple{result_xmm_reg_};
          } else {
            return std::tuple{result_};
          }
        }
      } else if constexpr (arg_info.arg_type == ArgInfo::OUT_TMP_ARG) {
        if constexpr (RegisterClass::kAsRegister == 'd') {
          result_reg_ = as_.rdx;
          return std::tuple{};
        } else {
          static_assert(kDependentValueFalse<arg_info.arg_type>);
        }
      } else if constexpr (arg_info.arg_type == ArgInfo::TMP_ARG) {
        static_assert(kUsage == machine_insn_info::kDef ||
                      kUsage == machine_insn_info::kDefEarlyClobber);
        if constexpr (RegisterClass::kAsRegister == 'm') {
          if (scratch_arg_ >= config::kScratchAreaSize / config::kScratchAreaSlotSize) {
            FATAL("Only two scratch registers are supported for now");
          }
          return std::tuple{x86_64::Assembler::Operand{
              .base = as_.rbp,
              .disp = static_cast<int>(offsetof(ThreadState, intrinsics_scratch_area) +
                                       config::kScratchAreaSlotSize * scratch_arg_++)}};
        } else if constexpr (RegisterClass::kIsImplicitReg) {
          return std::tuple{};
        } else {
          return std::tuple{reg_alloc()};
        }
      } else {
        static_assert(kDependentValueFalse<arg_info.arg_type>);
      }
    }
  }

 private:
  MacroAssembler<x86_64::Assembler>& as_;
  RegAlloc& reg_alloc_;
  SIMDRegAlloc& simd_reg_alloc_;
  AssemblerResType result_;
  x86_64::Assembler::Register result_reg_ = x86_64::Assembler::no_register;
  x86_64::Assembler::XMMRegister result_xmm_reg_ = x86_64::Assembler::no_xmm_register;
  std::tuple<AssemblerArgType...> input_args_;
  uint32_t scratch_arg_ = 0;
  bool success_;
};

template <auto kFunction,
          typename RegAlloc,
          typename SIMDRegAlloc,
          typename AssemblerResType,
          typename... AssemblerArgType>
bool TryInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                        RegAlloc&& reg_alloc,
                        SIMDRegAlloc&& simd_reg_alloc,
                        AssemblerResType result,
                        AssemblerArgType... args) {
  if (InlineIntrinsic<kFunction>::TryInlineWithHostRounding(
          as, reg_alloc, simd_reg_alloc, result, args...)) {
    return true;
  }

  return TryBindingBasedInlineIntrinsic<kFunction,
                                        RegAlloc,
                                        SIMDRegAlloc,
                                        AssemblerResType,
                                        AssemblerArgType...>(
      as, reg_alloc, simd_reg_alloc, result, args...);
}

}  // namespace berberis::inline_intrinsic

#endif  // BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_CALL_INTRINSIC_H_
