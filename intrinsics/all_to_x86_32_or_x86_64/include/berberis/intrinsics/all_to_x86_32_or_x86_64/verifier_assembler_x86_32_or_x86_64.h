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

#ifndef BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_VERIFIER_ASSEMBLER_COMMON_H_
#define BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_VERIFIER_ASSEMBLER_COMMON_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "berberis/assembler/x86_32_or_x86_64.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/dependent_false.h"
#include "berberis/device_arch_info/x86_32_or_x86_64/device_arch_info.h"
#include "berberis/intrinsics/common/intrinsics_bindings.h"

namespace berberis {

namespace x86_32_or_x86_64 {

template <typename DerivedAssemblerType>
class VerifierAssembler {
 public:
  using Condition = x86_32_or_x86_64::Condition;

  using ScaleFactor = x86_32_or_x86_64::ScaleFactor;
  static constexpr ScaleFactor kTimesOne = ScaleFactor::kTimesOne;
  static constexpr ScaleFactor kTimesTwo = ScaleFactor::kTimesTwo;
  static constexpr ScaleFactor kTimesFour = ScaleFactor::kTimesFour;
  static constexpr ScaleFactor kTimesEight = ScaleFactor::kTimesEight;

  struct Label {
    size_t id;
    int index = -1;
    bool bound = false;
  };

  struct Operand;

  class Register {
   public:
    constexpr Register(std::optional<Register> reg)
        : Register(reg.has_value() ? *reg : (FATAL("attempt to use undeclared register"), *reg)) {}
    constexpr Register(int arg_no, device_arch_info::RegBindingKind binding_kind)
        : arg_no_(arg_no), binding_kind_(binding_kind) {}

    constexpr int arg_no() const {
      CHECK_NE(arg_no_, kNoRegister);
      return arg_no_;
    }

    constexpr bool register_initialised() const { return (arg_no_ != kNoRegister); }

    constexpr bool operator==(const Register& other) const { return arg_no() == other.arg_no(); }
    constexpr bool operator!=(const Register& other) const { return arg_no() != other.arg_no(); }

    static constexpr int kNoRegister = -1;
    static constexpr int kStackPointer = -2;
    // Used in Operand to deal with references to scratch area.
    static constexpr int kScratchPointer = -3;

    constexpr device_arch_info::RegBindingKind get_binding_kind() const { return binding_kind_; }

   private:
    friend struct Operand;

    // Register number created during creation of assembler call.
    // See arg['arm_register'] in _gen_c_intrinsic_body in gen_intrinsics.py
    //
    // Default value (-1) means it's not assigned yet (thus couldn't be used).
    int arg_no_;
    device_arch_info::RegBindingKind binding_kind_;
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

    constexpr SIMDRegister(int arg_no, device_arch_info::RegBindingKind binding_kind)
        : arg_no_(arg_no), binding_kind_(binding_kind) {}

    constexpr int arg_no() const {
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
      return std::enable_if_t<kBits != 128, SIMDRegister<128>>{arg_no_, binding_kind_};
    }
    constexpr auto To256Bit() const {
      return std::enable_if_t<kBits != 256, SIMDRegister<256>>{arg_no_, binding_kind_};
    }

    constexpr device_arch_info::RegBindingKind get_binding_kind() const { return binding_kind_; }

   private:
    // Register number created during creation of assembler call.
    // See arg['arm_register'] in _gen_c_intrinsic_body in gen_intrinsics.py
    //
    // Default value (-1) means it's not assigned yet (thus couldn't be used).
    static constexpr int kNoRegister = -1;
    int arg_no_;
    device_arch_info::RegBindingKind binding_kind_;
  };

  using XMMRegister = SIMDRegister<128>;
  using YMMRegister = SIMDRegister<256>;

  using XRegister = XMMRegister;

  struct Operand {
    std::optional<Register> base{};
    std::optional<Register> index{};
    ScaleFactor scale = kTimesOne;
    int32_t disp = 0;
  };

  constexpr VerifierAssembler() {}

  // These start as Register::kNoRegister but can be changed if they are used as arguments to
  // something else.
  // If they are not coming as arguments then using them is compile-time error!
  std::optional<Register> gpr_a{};
  std::optional<Register> gpr_b{};
  std::optional<Register> gpr_c{};
  std::optional<Register> gpr_d{};
  // Note: stack pointer is not reflected in list of arguments, intrinsics use it implicitly.
  // It's also always defined on the entrance to intrinsics and, if modified, has to be restored.
  // But kUse/kDef is not precise enough to describe “this register could be touched but has to be
  // restored” requirement, thus we define it as kUseDef.
  Register gpr_s{Register::kStackPointer, device_arch_info::kUseDef};
  // Used in Operand as pseudo-register to temporary operand.
  std::optional<Register> gpr_scratch{};

  // In x86-64 case we could refer to kBerberisMacroAssemblerConstants via %rip.
  // In x86-32 mode, on the other hand, we need complex dance to access it via GOT.
  // Intrinsics which use these constants receive it via additional parameter - and
  // we need to know if it's needed or not.
  std::optional<Register> gpr_macroassembler_constants{};
  bool need_gpr_macroassembler_constants() const { return need_gpr_macroassembler_constants_; }

  std::optional<Register> gpr_macroassembler_scratch{};
  bool need_gpr_macroassembler_scratch() const { return need_gpr_macroassembler_scratch_; }
  std::optional<Register> gpr_macroassembler_scratch2{};

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
  bool need_sse_or_sse2 = false;
  bool need_sse3 = false;
  bool need_ssse3 = false;
  bool need_sse4_1 = false;
  bool need_sse4_2 = false;
  bool need_vaes = false;
  bool need_vpclmulqd = false;
  bool has_custom_capability = false;

  bool defines_flags = false;

  bool intrinsic_is_non_linear = false;

  // We assume that maximum number of XMM/general/fixed registers binded to the intrinsic is 16.
  // VerifierAssembler thus assumes arg_no will never be higher than this number. We use arrays of
  // size 16 to track individual registers. If there is a register with an arg_no higher than 16, we
  // will see a compiler error, since we detect out-of-bounds access to the array in constexpr.
  static constexpr int kMaxRegisters = 16;

  // Verifier Assmebler checks that 'def' or 'def_early_clober' XMM registers aren't read before
  // they are written to, unless they are used in a dependency breaking instruction. However, many
  // intrinsics first use and define an XMM register in a non dependency breaking instruction. This
  // check is default disabled, but can be enabled to view and manually check these intrinsics.
  static constexpr bool kCheckDefOrDefEarlyClobberXMMRegistersAreWrittenBeforeRead = false;

  class RegisterUsageFlags {
   public:
    constexpr void CheckValidRegisterUse(bool is_fixed) {
      if (intrinsic_defined_def_general_register ||
          (intrinsic_defined_def_fixed_register && !is_fixed)) {
        FATAL(
            "error: intrinsic used a 'use' general register after writing to a 'def' general "
            "register");
      }
    }

    constexpr void CheckValidXMMRegisterUse() {
      if (intrinsic_defined_def_xmm_register) {
        FATAL(
            "error: intrinsic used a 'use' xmm register after writing to a 'def' xmm "
            "register");
      }
    }

    constexpr void CheckAppropriateDefEarlyClobbers() {
      for (int i = 0; i < kMaxRegisters; i++) {
        if (intrinsic_defined_def_early_clobber_fixed_register.at(i) &&
            !valid_def_early_clobber_register.at(i)) {
          FATAL(
              "error: intrinsic never used a 'use' general register after writing to a "
              "'def_early_clobber' fixed register");
        }
        if (intrinsic_defined_def_early_clobber_general_register.at(i) &&
            !valid_def_early_clobber_register.at(i)) {
          FATAL(
              "error: intrinsic never used a 'use' general/fixed register after writing to a "
              "'def_early_clobber' general register");
        }
        if (intrinsic_defined_def_early_clobber_xmm_register.at(i) &&
            !valid_def_early_clobber_register.at(i)) {
          FATAL(
              "error: intrinsic never used a 'use' xmm register after writing to a "
              "'def_early_clobber' xmm register");
        }
      }
    }

    constexpr void CheckValidDefOrDefEarlyClobberRegisterUse(int reg_arg_no) {
      if (!intrinsic_defined_def_or_def_early_clobber_register.at(reg_arg_no)) {
        FATAL("error: intrinsic read a def/def_early_clobber register before writing to it");
      }
    }

    constexpr void UpdateIntrinsicRegisterDef(bool is_fixed) {
      if (is_fixed) {
        intrinsic_defined_def_fixed_register = true;
      } else {
        intrinsic_defined_def_general_register = true;
      }
    }

    constexpr void UpdateIntrinsicDefOrDefEarlyClobberRegister(int reg_arg_no) {
      intrinsic_defined_def_or_def_early_clobber_register.at(reg_arg_no) = true;
    }

    constexpr void UpdateIntrinsicRegisterDefEarlyClobber(int reg_arg_no, bool is_fixed) {
      if (is_fixed) {
        intrinsic_defined_def_early_clobber_fixed_register.at(reg_arg_no) = true;
      } else {
        intrinsic_defined_def_early_clobber_general_register.at(reg_arg_no) = true;
      }
    }

    constexpr void UpdateIntrinsicRegisterUse(bool is_fixed) {
      for (int i = 0; i < kMaxRegisters; i++) {
        if (intrinsic_defined_def_early_clobber_general_register.at(i)) {
          valid_def_early_clobber_register.at(i) = true;
        }
        if (intrinsic_defined_def_early_clobber_fixed_register.at(i) && !is_fixed) {
          valid_def_early_clobber_register.at(i) = true;
        }
      }
    }

    constexpr void UpdateIntrinsicXMMRegisterDef() { intrinsic_defined_def_xmm_register = true; }

    constexpr void UpdateIntrinsicXMMRegisterDefEarlyClobber(int reg_arg_no) {
      intrinsic_defined_def_early_clobber_xmm_register.at(reg_arg_no) = true;
    }

    constexpr void UpdateIntrinsicXMMRegisterUse() {
      for (int i = 0; i < kMaxRegisters; i++) {
        if (intrinsic_defined_def_early_clobber_xmm_register.at(i)) {
          valid_def_early_clobber_register.at(i) = true;
        }
      }
    }

    constexpr void Update32BitGeneralRegisterExtension(int reg_arg_no, bool is_zero_extended) {
      if (is_zero_extended) {
        zero_extended_32_bit_register.at(reg_arg_no) = true;
      } else {
        zero_extended_32_bit_register.at(reg_arg_no) = false;
      }
    }

    constexpr void Update32BitFixedRegisterExtension(int fixed_reg_index, bool is_zero_extended) {
      if (is_zero_extended) {
        zero_extended_fixed_register.at(fixed_reg_index) = true;
      } else {
        zero_extended_fixed_register.at(fixed_reg_index) = false;
      }
    }

    enum {
      kFixedRegisterShift,
      kGeneralRegisterShift,
      kXMMRegisterShift,
      kNumStateBits,
    };

    constexpr int GetNonLinearUseDefState() {
      int state = 0;
      if (intrinsic_defined_def_fixed_register) {
        state += 1 << kFixedRegisterShift;
      }
      if (intrinsic_defined_def_general_register) {
        state += 1 << kGeneralRegisterShift;
      }
      if (intrinsic_defined_def_xmm_register) {
        state += 1 << kXMMRegisterShift;
      }
      return state;
    }

    constexpr void Check32BitRegisterIsZeroExtended(int reg_no) {
      if (!zero_extended_32_bit_register.at(reg_no)) {
        FATAL("error: intrinsic didn't zero extend 32 bit output register");
      }
    }

    constexpr void Check32BitFixedRegisterIsZeroExtended(int fixed_reg_index) {
      if (!zero_extended_fixed_register.at(fixed_reg_index)) {
        if (fixed_reg_index == 0) {
          FATAL("error: intrinsic didn't zero extend 32 bit EAX output");
        }
        if (fixed_reg_index == 1) {
          FATAL("error: intrinsic didn't zero extend 32 bit EBX output");
        }
        if (fixed_reg_index == 2) {
          FATAL("error: intrinsic didn't zero extend 32 bit ECX output");
        }
        if (fixed_reg_index == 3) {
          FATAL("error: intrinsic didn't zero extend 32 bit EDX output");
        }
      }
    }

   private:
    bool intrinsic_defined_def_fixed_register = false;
    bool intrinsic_defined_def_general_register = false;
    bool intrinsic_defined_def_xmm_register = false;

    std::array<bool, kMaxRegisters> intrinsic_defined_def_or_def_early_clobber_register{};

    std::array<bool, kMaxRegisters> intrinsic_defined_def_early_clobber_fixed_register{};
    std::array<bool, kMaxRegisters> intrinsic_defined_def_early_clobber_general_register{};
    std::array<bool, kMaxRegisters> intrinsic_defined_def_early_clobber_xmm_register{};

    std::array<bool, kMaxRegisters> valid_def_early_clobber_register{};

    std::array<bool, kMaxRegisters> zero_extended_32_bit_register{};
    std::array<bool, 4> zero_extended_fixed_register{};  // {gpr_a, gpr_b, gpr_c, gpr_d}
  };

  RegisterUsageFlags register_usage_flags;

  struct Instruction {
    constexpr void UpdateInstructionRegisterDef(bool is_fixed) {
      if (is_fixed) {
        instruction_defined_def_fixed_register = true;
      } else {
        instruction_defined_def_general_register = true;
      }
    }

    constexpr void UpdateInstructionXMMRegisterDef() {
      instruction_defined_def_xmm_register = true;
    }

    constexpr void UpdateInstructionRegisterUse(bool is_fixed) {
      if (is_fixed) {
        instruction_used_use_fixed_register = true;
      } else {
        instruction_used_use_general_register = true;
      }
    }

    constexpr void UpdateInstructionXMMRegisterUse() { instruction_used_use_xmm_register = true; }

    constexpr void UpdateInstructionGeneralRegisterZeroExtension(int reg_no,
                                                                 bool is_zero_extended) {
      if (is_zero_extended) {
        instruction_zero_extended_general_register.at(reg_no) = true;
      } else {
        instruction_defined_general_register_no_zero_extension.at(reg_no) = true;
      }
    }

    constexpr bool CheckVisited(RegisterUsageFlags use_def_flags) {
      return use_def_state_checked.at(use_def_flags.GetNonLinearUseDefState());
    }

    constexpr void SetVisited(RegisterUsageFlags use_def_flags) {
      use_def_state_checked.at(use_def_flags.GetNonLinearUseDefState()) = true;
    }

    constexpr bool CheckVisited(bool register_currently_zero_extended) {
      if (!register_currently_zero_extended) {
        return zero_extension_register_checked.at(0);
      } else {
        return zero_extension_register_checked.at(1);
      }
    }

    constexpr void SetVisited(bool register_currently_zero_extended) {
      if (!register_currently_zero_extended) {
        zero_extension_register_checked.at(0) = true;
      } else {
        zero_extension_register_checked.at(1) = true;
      }
    }

    constexpr void ProcessInstructionUseDefs(RegisterUsageFlags& use_def_flags) {
      if (instruction_used_use_fixed_register) {
        use_def_flags.CheckValidRegisterUse(true);
      }
      if (instruction_used_use_general_register) {
        use_def_flags.CheckValidRegisterUse(false);
      }
      if (instruction_used_use_xmm_register) {
        use_def_flags.CheckValidXMMRegisterUse();
      }
      if (instruction_defined_def_fixed_register) {
        use_def_flags.UpdateIntrinsicRegisterDef(true);
      }
      if (instruction_defined_def_general_register) {
        use_def_flags.UpdateIntrinsicRegisterDef(false);
      }
      if (instruction_defined_def_xmm_register) {
        use_def_flags.UpdateIntrinsicXMMRegisterDef();
      }
    }

    constexpr void ProcessInstructionZeroExtension(bool& register_currently_zero_extended,
                                                   int reg_no) {
      if (instruction_defined_general_register_no_zero_extension.at(reg_no)) {
        register_currently_zero_extended = false;
        return;
      }
      if (instruction_zero_extended_general_register.at(reg_no)) {
        register_currently_zero_extended = true;
      }
    }

    bool instruction_defined_def_fixed_register = false;
    bool instruction_defined_def_general_register = false;
    bool instruction_defined_def_xmm_register = false;

    bool instruction_used_use_fixed_register = false;
    bool instruction_used_use_general_register = false;
    bool instruction_used_use_xmm_register = false;

    std::array<bool, kMaxRegisters> instruction_defined_general_register_no_zero_extension{};
    std::array<bool, kMaxRegisters> instruction_zero_extended_general_register{};

    bool is_unconditional_jump = false;
    bool is_conditional_jump = false;
    Label* jump_target = nullptr;

    // The check for each instruction is fully defined by prior `def` register flags.
    // When we reach an instruction by different paths, we may arrive with different 'def' flags. We
    // use this array to memorize which `def` combinations we have checked already.
    //
    // The state to keep track of is whether a 'def' register of each of the three types (general,
    // fixed and xmm) has been written in the intrinsic yet. Thus, there are 2^3 = 8 possible states
    // of an instruction.
    std::array<bool, 1 << RegisterUsageFlags::kNumStateBits> use_def_state_checked{};

    std::array<bool, 2> zero_extension_register_checked{};
  };

  constexpr void CheckAppropriateDefEarlyClobbers() {
    if (intrinsic_is_non_linear) {
      return;
    }
    register_usage_flags.CheckAppropriateDefEarlyClobbers();
  }

  constexpr void Check32BitRegisterIsZeroExtended(int reg_no) {
    if (!intrinsic_is_non_linear) {
      register_usage_flags.Check32BitRegisterIsZeroExtended(reg_no);
    } else {
      CheckInstructionZeroExtensionRecursive(0, false, reg_no);
    }
  }

  constexpr void Check32BitFixedRegisterIsZeroExtended(int fixed_reg_index) {
    if (intrinsic_is_non_linear) {
      return;
    }
    register_usage_flags.Check32BitFixedRegisterIsZeroExtended(fixed_reg_index);
  }

  constexpr void CheckLabelsAreBound() {
    if (!intrinsic_is_non_linear) {
      return;
    }
    for (int i = 0; i < num_instructions_; i++) {
      if (instructions.at(i).is_conditional_jump || instructions.at(i).is_unconditional_jump) {
        if (instructions.at(i).jump_target->bound == false) {
          FATAL("error: intrinsic jumps to a label that was never bound");
        }
      }
    }
  }

  constexpr void CheckNonLinearIntrinsicsUseDefRegisters() {
    if (!intrinsic_is_non_linear) {
      return;
    }
    // Uses DFS to check that a 'use' register is never used after a 'def' register is written on
    // all paths of a non-linear intrinsic.
    RegisterUsageFlags use_def_flags{};
    CheckInstructionUseDefRegistersRecursive(0, use_def_flags);
  }

  constexpr void CheckInstructionUseDefRegistersRecursive(int current_instruction,
                                                          RegisterUsageFlags use_def_flags) {
    CHECK_LE(current_instruction, num_instructions_);
    if (current_instruction == num_instructions_) {
      // Reached end of intrinsic.
      return;
    }
    if (instructions.at(current_instruction).CheckVisited(use_def_flags)) {
      // Already visited this instruction with the same use_def state.
      return;
    }
    instructions.at(current_instruction).SetVisited(use_def_flags);
    instructions.at(current_instruction).ProcessInstructionUseDefs(use_def_flags);
    if (instructions.at(current_instruction).is_unconditional_jump ||
        instructions.at(current_instruction).is_conditional_jump) {
      // Explore execution path given that jump is taken.
      CheckInstructionUseDefRegistersRecursive(
          instructions.at(current_instruction).jump_target->index, use_def_flags);
    }
    if (instructions.at(current_instruction).is_unconditional_jump) {
      return;
    }
    // Explore execution path given that we move to the next instruction.
    CheckInstructionUseDefRegistersRecursive(current_instruction + 1, use_def_flags);
  }

  constexpr void CheckInstructionZeroExtensionRecursive(int current_instruction,
                                                        bool register_currently_zero_extended,
                                                        int reg_no) {
    CHECK_LE(current_instruction, num_instructions_);
    if (current_instruction == num_instructions_) {
      // Reached end of intrinsic. Check if register has ended up zero extended.
      if (!register_currently_zero_extended) {
        FATAL("error: intrinsic didn't zero extend 32 bit output register");
      }
      return;
    }
    if (instructions.at(current_instruction).CheckVisited(register_currently_zero_extended)) {
      // Already visited this instruction with the same use_def state.
      return;
    }
    instructions.at(current_instruction).SetVisited(register_currently_zero_extended);
    instructions.at(current_instruction)
        .ProcessInstructionZeroExtension(register_currently_zero_extended, reg_no);
    if (instructions.at(current_instruction).is_unconditional_jump ||
        instructions.at(current_instruction).is_conditional_jump) {
      // Explore execution path given that jump is taken.
      CheckInstructionZeroExtensionRecursive(
          instructions.at(current_instruction).jump_target->index,
          register_currently_zero_extended,
          reg_no);
    }
    if (instructions.at(current_instruction).is_unconditional_jump) {
      return;
    }
    // Explore execution path given that we move to the next instruction.
    CheckInstructionZeroExtensionRecursive(
        current_instruction + 1, register_currently_zero_extended, reg_no);
  }

  constexpr void Bind(Label* label) {
    CHECK_EQ(label->bound, false);
    label->index = num_instructions_;
    label->bound = true;
  }

  constexpr Label* MakeLabel() {
    intrinsic_is_non_linear = true;
    labels_.at(num_labels_) = {{num_labels_}};
    return &labels_.at(num_labels_++);
  }

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
    // Technically AVX implies SSE but mixing AVX and SSE instructions can cause a performance
    // penalty. Thus, we first ensure that AVX-using intrinsics don't use SSE instructions, before
    // propagating required feature dependencies correctly.
    if (need_avx && need_sse_or_sse2) {
      FATAL("error: intrinsic used both AVX and SSE instructions");
    }

    constexpr bool expect_bmi = std::is_same_v<CPUIDRestriction, device_arch_info::HasBMI>;
    constexpr bool expect_f16c = std::is_same_v<CPUIDRestriction, device_arch_info::HasF16C>;
    constexpr bool expect_fma = std::is_same_v<CPUIDRestriction, device_arch_info::HasFMA>;
    constexpr bool expect_fma4 = std::is_same_v<CPUIDRestriction, device_arch_info::HasFMA4>;
    constexpr bool expect_lzcnt = std::is_same_v<CPUIDRestriction, device_arch_info::HasLZCNT>;
    constexpr bool expect_vaes = std::is_same_v<CPUIDRestriction, device_arch_info::HasVAES>;
    constexpr bool expect_vpclmulqd =
        std::is_same_v<CPUIDRestriction, device_arch_info::HasVPCLMULQD>;
    constexpr bool expect_aesavx =
        std::is_same_v<CPUIDRestriction, device_arch_info::HasAESAVX> || expect_vaes;
    constexpr bool expect_aes = std::is_same_v<CPUIDRestriction, device_arch_info::HasAES>;
    constexpr bool expect_clmulavx =
        std::is_same_v<CPUIDRestriction, device_arch_info::HasCLMULAVX> || expect_vpclmulqd;
    constexpr bool expect_clmul = std::is_same_v<CPUIDRestriction, device_arch_info::HasCLMUL>;
    constexpr bool expect_popcnt = std::is_same_v<CPUIDRestriction, device_arch_info::HasPOPCNT>;
    constexpr bool expect_avx = std::is_same_v<CPUIDRestriction, device_arch_info::HasAVX> ||
                                expect_aesavx || expect_clmulavx || expect_f16c || expect_fma ||
                                expect_fma4;
    constexpr bool expect_sse4_2 =
        std::is_same_v<CPUIDRestriction, device_arch_info::HasSSE4_2> || expect_aes || expect_clmul;
    constexpr bool expect_sse4_1 =
        std::is_same_v<CPUIDRestriction, device_arch_info::HasSSE4_1> || expect_sse4_2;
    constexpr bool expect_ssse3 =
        std::is_same_v<CPUIDRestriction, device_arch_info::HasSSSE3> || expect_sse4_1;
    constexpr bool expect_sse3 =
        std::is_same_v<CPUIDRestriction, device_arch_info::HasSSE3> || expect_ssse3;

    // Note that we don't check SSE or SSE2, since we assume SSE2 is always available.

    if (expect_aesavx != need_aesavx) {
      FATAL("error: expect_aesavx != need_aesavx");
    }
    if (expect_aes != need_aes) {
      FATAL("error: expect_aes != need_aes");
    }
    if (expect_avx != need_avx) {
      FATAL("error: expect_avx != need_avx");
    }
    if (expect_bmi != need_bmi) {
      FATAL("error: expect_bmi != need_bmi");
    }
    if (expect_clmulavx != need_clmulavx) {
      FATAL("error: expect_clmulavx != need_clmulavx");
    }
    if (expect_clmul != need_clmul) {
      FATAL("error: expect_clmul != need_clmul");
    }
    if (expect_f16c != need_f16c) {
      FATAL("error: expect_f16c != need_f16c");
    }
    if (expect_fma != need_fma) {
      FATAL("error: expect_fma != need_fma");
    }
    if (expect_fma4 != need_fma4) {
      FATAL("error: expect_fma4 != need_fma4");
    }
    if (expect_lzcnt != need_lzcnt) {
      FATAL("error: expect_lzcnt != need_lzcnt");
    }
    if (expect_popcnt != need_popcnt) {
      FATAL("error: expect_popcnt != need_popcnt");
    }
    if (expect_sse3 != need_sse3) {
      FATAL("error: expect_sse3 != need_sse3");
    }
    if (expect_ssse3 != need_ssse3) {
      FATAL("error: expect_ssse3 != need_ssse3");
    }
    if (expect_sse4_1 != need_sse4_1) {
      FATAL("error: expect_sse4_1 != need_sse4_1");
    }
    if (expect_sse4_2 != need_sse4_2) {
      FATAL("error: expect_sse4_2 != need_sse4_2");
    }
    if (expect_vaes != need_vaes) {
      FATAL("error: expect_vaes != need_vaes");
    }
    if (expect_vpclmulqd != need_vpclmulqd) {
      FATAL("error: expect_vpclmulqd != need_vpclmulqd");
    }
  }

  constexpr void CheckFlagsBinding(bool expect_flags) {
    if (expect_flags != defines_flags) {
      FATAL("error: expect_flags != defines_flags");
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
    SetRequiredFeatureAVX();
  }

  constexpr void SetRequiredFeatureAES() {
    need_aes = true;
    SetRequiredFeatureSSE4_2();
  }

  constexpr void SetRequiredFeatureAVX() {
    // Technically AVX implies SSE but mixing AVX and SSE instructions can cause a performance
    // penalty. Thus, we first ensure that AVX-using intrinsics don't use SSE instructions, before
    // propagating required feature dependencies correctly.
    need_avx = true;
  }

  constexpr void SetRequiredFeatureAVX2() {
    need_avx2 = true;
    SetRequiredFeatureAVX();
  }

  constexpr void SetRequiredFeatureBMI() { need_bmi = true; }

  constexpr void SetRequiredFeatureBMI2() { need_bmi2 = true; }

  constexpr void SetRequiredFeatureCLMULAVX() {
    need_clmulavx = true;
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

  constexpr void SetRequiredFeatureSSEOrSSE2() { need_sse_or_sse2 = true; }

  constexpr void SetRequiredFeatureSSE3() {
    need_sse3 = true;
    SetRequiredFeatureSSEOrSSE2();
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

  constexpr bool RegisterIsFixed(Register reg) {
    if (gpr_a.has_value()) {
      if (reg == gpr_a) return true;
    }
    if (gpr_b.has_value()) {
      if (reg == gpr_b) return true;
    }
    if (gpr_c.has_value()) {
      if (reg == gpr_c) return true;
    }
    if (gpr_d.has_value()) {
      if (reg == gpr_d) return true;
    }
    if (reg == gpr_s) return true;
    return false;
  }

  constexpr bool GetFixedRegisterIndex(Register reg) {
    if (gpr_a.has_value()) {
      if (reg == gpr_a) return 0;
    }
    if (gpr_b.has_value()) {
      if (reg == gpr_b) return 1;
    }
    if (gpr_c.has_value()) {
      if (reg == gpr_c) return 2;
    }
    if (gpr_d.has_value()) {
      if (reg == gpr_d) return 3;
    }
    if (reg == gpr_s) return 4;
    FATAL("Register is not fixed or is not permitted for use in this intrinsic");
  }

  constexpr void RegisterDef(Register reg, bool is_zero_extended = false) {
    if (reg.get_binding_kind() == device_arch_info::kUse) {
      FATAL("error: intrinsic defined a 'use' register");
    }
    if (reg.get_binding_kind() == device_arch_info::kDef ||
        reg.get_binding_kind() == device_arch_info::kDefEarlyClobber) {
      register_usage_flags.UpdateIntrinsicDefOrDefEarlyClobberRegister(reg.arg_no());
    }
    if (reg.get_binding_kind() == device_arch_info::kDef) {
      instructions.at(num_instructions_).UpdateInstructionRegisterDef(RegisterIsFixed(reg));
      register_usage_flags.UpdateIntrinsicRegisterDef(RegisterIsFixed(reg));
    } else if (reg.get_binding_kind() == device_arch_info::kDefEarlyClobber) {
      register_usage_flags.UpdateIntrinsicRegisterDefEarlyClobber(reg.arg_no(),
                                                                  RegisterIsFixed(reg));
    }
    if (!RegisterIsFixed(reg)) {
      register_usage_flags.Update32BitGeneralRegisterExtension(reg.arg_no(), is_zero_extended);
      instructions.at(num_instructions_)
          .UpdateInstructionGeneralRegisterZeroExtension(reg.arg_no(), is_zero_extended);
    } else {
      int fixed_reg_index = GetFixedRegisterIndex(reg);
      // The stack pointer (gpr_s) is also a fixed register, but it is not part of the
      // zero_extended_fixed_register array, and we don't handle it here.
      if (fixed_reg_index < 4) {
        register_usage_flags.Update32BitFixedRegisterExtension(fixed_reg_index, is_zero_extended);
      }
    }
  }

  constexpr void RegisterDef(XMMRegister reg, [[maybe_unused]] bool is_zero_extended = false) {
    if (reg.get_binding_kind() == device_arch_info::kUse) {
      FATAL("error: intrinsic defined a 'use' XMM register");
    }
    if (reg.get_binding_kind() == device_arch_info::kDef ||
        reg.get_binding_kind() == device_arch_info::kDefEarlyClobber) {
      register_usage_flags.UpdateIntrinsicDefOrDefEarlyClobberRegister(reg.arg_no());
    }
    if (reg.get_binding_kind() == device_arch_info::kDef) {
      instructions.at(num_instructions_).UpdateInstructionXMMRegisterDef();
      register_usage_flags.UpdateIntrinsicXMMRegisterDef();
    } else if (reg.get_binding_kind() == device_arch_info::kDefEarlyClobber) {
      register_usage_flags.UpdateIntrinsicXMMRegisterDefEarlyClobber(reg.arg_no());
    }
  }

  constexpr void RegisterUse(Register reg) {
    if (reg.get_binding_kind() == device_arch_info::kUse) {
      instructions.at(num_instructions_).UpdateInstructionRegisterUse(RegisterIsFixed(reg));
    }
    if (intrinsic_is_non_linear) {
      return;
    }
    if (reg.get_binding_kind() == device_arch_info::kUse) {
      register_usage_flags.CheckValidRegisterUse(RegisterIsFixed(reg));
      register_usage_flags.UpdateIntrinsicRegisterUse(RegisterIsFixed(reg));
    }
    if (reg.get_binding_kind() == device_arch_info::kDef ||
        reg.get_binding_kind() == device_arch_info::kDefEarlyClobber) {
      register_usage_flags.CheckValidDefOrDefEarlyClobberRegisterUse(reg.arg_no());
    }
  }

  constexpr void RegisterUse(XMMRegister reg) {
    if (reg.get_binding_kind() == device_arch_info::kUse) {
      instructions.at(num_instructions_).UpdateInstructionXMMRegisterUse();
    }
    if (intrinsic_is_non_linear) {
      return;
    }
    if (reg.get_binding_kind() == device_arch_info::kUse) {
      register_usage_flags.CheckValidXMMRegisterUse();
      register_usage_flags.UpdateIntrinsicXMMRegisterUse();
    }
    if (!kCheckDefOrDefEarlyClobberXMMRegistersAreWrittenBeforeRead) {
      return;
    }
    if (reg.get_binding_kind() == device_arch_info::kDef ||
        reg.get_binding_kind() == device_arch_info::kDefEarlyClobber) {
      register_usage_flags.CheckValidDefOrDefEarlyClobberRegisterUse(reg.arg_no());
    }
  }

  template <typename RegisterType>
  constexpr void HandleDefOrDefEarlyClobberRegisterReset(RegisterType reg1, RegisterType reg2) {
    if (reg1 == reg2 && (reg1.get_binding_kind() == device_arch_info::kDef ||
                         reg1.get_binding_kind() == device_arch_info::kDefEarlyClobber)) {
      register_usage_flags.UpdateIntrinsicDefOrDefEarlyClobberRegister(reg1.arg_no());
    }
  }

  constexpr void HandleDefOrDefEarlyClobberRegisterReset(XMMRegister reg1,
                                                         XMMRegister reg2,
                                                         XMMRegister reg3) {
    if (reg2 == reg3 && (reg1.get_binding_kind() == device_arch_info::kDef ||
                         reg1.get_binding_kind() == device_arch_info::kDefEarlyClobber)) {
      register_usage_flags.UpdateIntrinsicDefOrDefEarlyClobberRegister(reg1.arg_no());
    }
  }

  constexpr void HandleConditionalJump(const Label& label) {
    instructions.at(num_instructions_).is_conditional_jump = true;
    instructions.at(num_instructions_).jump_target = const_cast<Label*>(&label);
  }

  constexpr void HandleUnconditionalJump(const Label& label) {
    instructions.at(num_instructions_).is_unconditional_jump = true;
    instructions.at(num_instructions_).jump_target = const_cast<Label*>(&label);
  }

  constexpr void HandleUnconditionalJumpRegister() {
    FATAL("error: intrinsic does jump to register");
  }

  constexpr void EndInstruction() { num_instructions_++; }

 private:
  // Time complexity of checking correct use/def register bindings for non linear intrinsics is 2^n.
  // Therefore, we only handle intrinsics with maximum of 5 labels. Also, no intrinsics exist with >
  // 5 labels, so we can use this array for all intrinsics.
  static constexpr int kMaxLabels = 5;
  std::array<Label, kMaxLabels> labels_{};
  size_t num_labels_ = 0;

  int num_instructions_ = 0;
  static constexpr int kMaxInstructions = 300;
  std::array<Instruction, kMaxInstructions> instructions{};

  VerifierAssembler(const VerifierAssembler&) = delete;
  VerifierAssembler(VerifierAssembler&&) = delete;
  void operator=(const VerifierAssembler&) = delete;
  void operator=(VerifierAssembler&&) = delete;
};

}  // namespace x86_32_or_x86_64

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_ALL_TO_X86_32_OR_X86_64_VERIFIER_ASSEMBLER_COMMON_H_
