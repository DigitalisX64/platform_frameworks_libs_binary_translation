/*
 * Copyright (C) 2014 The Android Open Source Project
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

// Assembler to produce x86 instructions. Somewhat influenced by V8 assembler.

#ifndef BERBERIS_ASSEMBLER_X86_32_H_
#define BERBERIS_ASSEMBLER_X86_32_H_

#include <type_traits>  // std::is_same

#include "berberis/assembler/x86_32_or_x86_64.h"
#include "berberis/base/bit_util.h"

namespace berberis {

namespace x86_32 {

class Assembler : public x86_32_or_x86_64::Assembler<Assembler> {
 public:
  using BaseAssembler = x86_32_or_x86_64::Assembler<Assembler>;
  using FinalAssembler = Assembler;

  explicit Assembler(MachineCode* code) : BaseAssembler(code) {}

  static constexpr Register no_register{0x80};
  static constexpr Register eax{0};
  static constexpr Register ecx{1};
  static constexpr Register edx{2};
  static constexpr Register ebx{3};
  static constexpr Register esp{4};
  static constexpr Register ebp{5};
  static constexpr Register esi{6};
  static constexpr Register edi{7};

  static constexpr XMMRegister xmm0{0};
  static constexpr XMMRegister xmm1{1};
  static constexpr XMMRegister xmm2{2};
  static constexpr XMMRegister xmm3{3};
  static constexpr XMMRegister xmm4{4};
  static constexpr XMMRegister xmm5{5};
  static constexpr XMMRegister xmm6{6};
  static constexpr XMMRegister xmm7{7};

  static constexpr YMMRegister no_ymm_register{0x80};
  static constexpr YMMRegister ymm0{0};
  static constexpr YMMRegister ymm1{1};
  static constexpr YMMRegister ymm2{2};
  static constexpr YMMRegister ymm3{3};
  static constexpr YMMRegister ymm4{4};
  static constexpr YMMRegister ymm5{5};
  static constexpr YMMRegister ymm6{6};
  static constexpr YMMRegister ymm7{7};

  // Macroassembler uses these names to support both x86-32 and x86-64 modes.
  static constexpr Register gpr_a{0};
  static constexpr Register gpr_c{1};
  static constexpr Register gpr_d{2};
  static constexpr Register gpr_b{3};
  static constexpr Register gpr_s{4};

// Instructions.
#include "berberis/assembler/gen_assembler_x86_32-inl.h"  // NOLINT generated file!

  // Unhide Decl(Mem) hidden by Decl(Reg).
  using BaseAssembler::Decl;

  // Unhide Decw(Mem) hidden by Decw(Reg).
  using BaseAssembler::Decw;

  // Unhide Incl(Mem) hidden by Incl(Reg).
  using BaseAssembler::Incl;

  // Unhide Incw(Mem) hidden by Incw(Reg).
  using BaseAssembler::Incw;

  // Unhide Movb(Reg, Reg) hidden by special versions below.
  using BaseAssembler::Movb;

  // Movb in 32-bit mode has certain optimizations not available in x86-64 mode
  constexpr void Movb(Register dest, const Operand& src) {
    if (IsAccumulator(dest) && src.base == no_register && src.index == no_register) {
      EmitInstruction<0xA0>(src.disp);
    } else {
      BaseAssembler::Movb(dest, src);
    }
  }

  constexpr void Movb(const Operand& dest, Register src) {
    if (dest.base == no_register && dest.index == no_register && IsAccumulator(src)) {
      EmitInstruction<0xA2>(dest.disp);
    } else {
      BaseAssembler::Movb(dest, src);
    }
  }

  // Unhide Movw(Reg, Reg) hidden by special versions below.
  using BaseAssembler::Movw;

  // Movw in 32-bit mode has certain optimizations not available in x86-64 mode
  constexpr void Movw(Register dest, const Operand& src) {
    if (IsAccumulator(dest) && src.base == no_register && src.index == no_register) {
      EmitInstruction<0x66, 0xA1>(src.disp);
    } else {
      BaseAssembler::Movw(dest, src);
    }
  }

  constexpr void Movw(const Operand& dest, Register src) {
    if (dest.base == no_register && dest.index == no_register && IsAccumulator(src)) {
      EmitInstruction<0x66, 0xA3>(dest.disp);
    } else {
      BaseAssembler::Movw(dest, src);
    }
  }

  // Unhide Movl(Reg, Reg) hidden by special versions below.
  using BaseAssembler::Movl;

  // Movl in 32-bit mode has certain optimizations not available in x86-64 mode
  constexpr void Movl(Register dest, const Operand& src) {
    if (IsAccumulator(dest) && src.base == no_register && src.index == no_register) {
      EmitInstruction<0xA1>(src.disp);
    } else {
      BaseAssembler::Movl(dest, src);
    }
  }

  constexpr void Movl(const Operand& dest, Register src) {
    if (dest.base == no_register && dest.index == no_register && IsAccumulator(src)) {
      EmitInstruction<0xA3>(dest.disp);
    } else {
      BaseAssembler::Movl(dest, src);
    }
  }

  // Unhide Vmov*(Mem, Reg) hidden by Vmov*(Reg, Reg).
  using BaseAssembler::Vmovapd;
  using BaseAssembler::Vmovaps;
  using BaseAssembler::Vmovdqa;
  using BaseAssembler::Vmovdqu;
  using BaseAssembler::Vmovq;
  using BaseAssembler::Vmovsd;
  using BaseAssembler::Vmovss;

  // TODO(b/127356868): decide what to do with these functions when cross-arch assembler is used.

#if defined(__i386__)

  // Unside Call(Reg), hidden by special version below.
  using BaseAssembler::Call;

  void Call(const void* target) {
    Emit8(0xe8);
    Emit32(0xcccccccc);
    // Set last 4 bytes to displacement from current pc to 'target'.
    AddRelocation(pc() - 4, RelocationType::RelocAbsToDisp32, pc(), bit_cast<intptr_t>(target));
  }

  // Unside Jcc(Label), hidden by special version below.
  using BaseAssembler::Jcc;

  // Make sure only type void* can be passed to function below, not Label* or any other pointer.
  template <typename T>
  auto Jcc(Condition cc, T* target) -> void = delete;

  template <typename T>
  auto Jcc(Condition cc, T target)
      -> std::enable_if_t<std::is_integral_v<T> && sizeof(uintptr_t) < sizeof(T)> = delete;

  void Jcc(Condition cc, uintptr_t target) {
    if (cc == Condition::kAlways) {
      Jmp(target);
      return;
    } else if (cc == Condition::kNever) {
      return;
    }
    CHECK_EQ(0, static_cast<uint8_t>(cc) & 0xF0);
    Emit8(0x0F);
    Emit8(0x80 | static_cast<uint8_t>(cc));
    Emit32(0xcccccccc);
    // Set last 4 bytes to displacement from current pc to 'target'.
    AddRelocation(pc() - 4, RelocationType::RelocAbsToDisp32, pc(), bit_cast<intptr_t>(target));
  }

  void Jcc(Condition cc, const void* target) { Jcc(cc, bit_cast<uintptr_t>(target)); }

  // Unside Jmp(Reg), hidden by special version below.
  using BaseAssembler::Jmp;

  // Make sure only type void* can be passed to function below, not Label* or any other pointer.
  template <typename T>
  auto Jmp(T* target) -> void = delete;

  template <typename T>
  auto Jmp(T target)
      -> std::enable_if_t<std::is_integral_v<T> && sizeof(uintptr_t) < sizeof(T)> = delete;

  void Jmp(uintptr_t target) {
    Emit8(0xe9);
    Emit32(0xcccccccc);
    // Set last 4 bytes to displacement from current pc to 'target'.
    AddRelocation(pc() - 4, RelocationType::RelocAbsToDisp32, pc(), bit_cast<intptr_t>(target));
  }

  void Jmp(const void* target) { Jmp(bit_cast<uintptr_t>(target)); }

#endif  // defined(__i386__)

 private:
  Assembler() = delete;
  Assembler(const Assembler&) = delete;
  Assembler(Assembler&&) = delete;
  void operator=(const Assembler&) = delete;
  void operator=(Assembler&&) = delete;
  using DerivedAssemblerType = Assembler;

  static Register Accumulator() { return eax; }
  static bool IsAccumulator(Register reg) { return reg == eax; }

  // Check if a given type is "a register with size" (for EmitInstruction).
  template <typename ArgumentType>
  struct IsRegister {
    static constexpr bool value =
        std::is_same_v<ArgumentType, Register8Bit> || std::is_same_v<ArgumentType, Register32Bit>;
  };

  // Check if a given type is "a memory operand with size" (for EmitInstruction).
  template <typename ArgumentType>
  struct IsMemoryOperand {
    static constexpr bool value = std::is_same_v<ArgumentType, Memory32Bit>;
  };

  // Check if a given type is "a memory operand with size" (for EmitInstruction).
  template <typename ArgumentType>
  struct IsLabelOperand {
    static constexpr bool value = std::is_same_v<ArgumentType, Label32Bit>;
  };

  template <typename... ArgumentsType>
  void EmitRex(ArgumentsType...) {
    // There is no REX in 32-bit mode thus we don't need to do anything here.
  }

  template <typename RegisterType>
  [[nodiscard]] static bool IsSwapProfitable(RegisterType /*rm_arg*/, RegisterType /*vex_arg*/) {
    // In 32bit mode swapping register to shorten VEX prefix is not possible.
    return false;
  }

  template <uint8_t byte1,
            uint8_t byte2,
            uint8_t byte3,
            bool reg_is_opcode_extension,
            typename... ArgumentsTypes>
  void EmitVex(ArgumentsTypes... arguments) {
    constexpr auto registers_count = kCountArguments<IsRegister, ArgumentsTypes...>;
    constexpr auto operands_count = kCountArguments<IsMemoryOperand, ArgumentsTypes...>;
    constexpr auto labels_count = kCountArguments<IsLabelOperand, ArgumentsTypes...>;
    constexpr auto vvvv_parameter = 2 - reg_is_opcode_extension - operands_count - labels_count;
    int vvvv = 0;
    if constexpr (registers_count > vvvv_parameter) {
      vvvv = ArgumentByType<vvvv_parameter, IsRegister>(arguments...).num_;
    }
    // Note that ¬R is always 1 in x86-32 mode but it's not set in JSON.
    // This means that 2nd byte of 3-byte vex is always the same in 32bit mode (but 3rd byte of
    // unfolded version and 2nd byte of folded one may still need to handle vvvv argument).
    if (byte1 == 0xC4 && byte2 == 0b0'0'0'00001 && (byte3 & 0b1'0000'0'00) == 0) {
      Emit16((0x80c5 | (byte3 << 8) | 0b0'1111'0'00'00000000) ^ (vvvv << 11));
    } else {
      Emit8(byte1);
      // Note that ¬R/¬X/¬B are always 1 in x86-32 mode. But they are specified as 0 in JSON.
      Emit16(((byte2 | 0b111'00000) | (byte3 << 8) | 0b0'1111'000'00000000) ^ (vvvv << 11));
    }
  }

  template <typename ArgumentType>
  void EmitRegisterInOpcode(uint8_t opcode, ArgumentType argument) {
    Emit8(opcode | argument.num_);
  }

  template <typename ArgumentType1, typename ArgumentType2>
  void EmitModRM(ArgumentType1 argument1, ArgumentType2 argument2) {
    Emit8(0xC0 | (argument1.num_ << 3) | argument2.num_);
  }

  template <typename ArgumentType>
  void EmitModRM(uint8_t opcode_extension, ArgumentType argument) {
    CHECK_LE(opcode_extension, 7);
    Emit8(0xC0 | (opcode_extension << 3) | argument.num_);
  }

  template <typename ArgumentType>
  void EmitOperandOp(ArgumentType argument, Operand operand) {
    EmitOperandOp(static_cast<int>(argument.num_), operand);
  }

  template <size_t kImmediatesSize, typename ArgumentType>
  void EmitRipOp(ArgumentType argument, const Label& label) {
    EmitRipOp<kImmediatesSize>(static_cast<int>(argument.num_), label);
  }

  // Emit the ModR/M byte, and optionally the SIB byte and
  // 1- or 4-byte offset for a memory operand.  Also used to encode
  // a three-bit opcode extension into the ModR/M byte.
  constexpr void EmitOperandOp(int num_ber, const Operand& addr);
  // Helper functions to handle various ModR/M and SIB combinations.
  // Should *only* be called from EmitOperandOp!
  constexpr void EmitIndexDispOperand(int reg, const Operand& addr);
  template <typename ArgType, void (AssemblerBase::*)(ArgType)>
  constexpr void EmitBaseIndexDispOperand(int base_modrm_and_sib, const Operand& addr);
  // Emit ModR/M for rip-addressig.
  template <size_t kImmediatesSize>
  constexpr void EmitRipOp(int num_, const Label& label);

  friend BaseAssembler;
};

// This function looks big, but when we are emitting Operand with fixed registers
// (which is the most common case) all "if"s below are calculated statically which
// makes effective size of that function very small.
//
// But for this to happen function have to be inline and in header.
constexpr inline void Assembler::EmitOperandOp(int num_ber, const Operand& addr) {
  // Additional info (register num_ber, etc) is limited to 3 bits.
  CHECK_LE(unsigned(num_ber), 7);

  // Reg field must be shifted by 3 bits.
  int reg = num_ber << 3;

  // On x86 %esp cannot be index, only base.
  CHECK(addr.index != esp);

  // If base is not %esp and we don't have index, then we don't have SIB byte.
  // All other cases have "ModR/M" and SIB bytes.
  if (addr.base != esp && addr.index == no_register) {
    // If we have base register then we could use the same logic as for other common cases.
    if (addr.base != no_register) {
      EmitBaseIndexDispOperand<uint8_t, &Assembler::Emit8>(addr.base.num_ | reg, addr);
    } else {
      Emit8(0x05 | reg);
      Emit32(addr.disp);
    }
  } else if (addr.index == no_register) {
    // Note: when ModR/M and SIB are used "no index" is encoded as if %esp is used in place of
    // index (that's why %esp couldn't be used as index - see check above).
    EmitBaseIndexDispOperand<int16_t, &Assembler::Emit16>(0x2004 | (addr.base.num_ << 8) | reg,
                                                          addr);
  } else if (addr.base == no_register) {
    EmitIndexDispOperand(reg, addr);
  } else {
    EmitBaseIndexDispOperand<int16_t, &Assembler::Emit16>(
        0x04 | (addr.scale << 14) | (addr.index.num_ << 11) | (addr.base.num_ << 8) | reg, addr);
  }
}

constexpr inline void Assembler::EmitIndexDispOperand(int reg, const Operand& addr) {
  // We only have index here, no base, use SIB but put %ebp in "base" field.
  Emit16(0x0504 | (addr.scale << 14) | (addr.index.num_ << 11) | reg);
  Emit32(addr.disp);
}

template <typename ArgType, void (AssemblerBase::* EmitBase)(ArgType)>
constexpr inline void Assembler::EmitBaseIndexDispOperand(int base_modrm_and_sib,
                                                          const Operand& addr) {
  if (addr.disp == 0 && addr.base != ebp) {
    // We can omit zero displacement only if base isn't %ebp
    (this->*EmitBase)(base_modrm_and_sib);
  } else if (IsInRange<int8_t>(addr.disp)) {
    // If disp could it in byte then use byte-disp.
    (this->*EmitBase)(base_modrm_and_sib | 0x40);
    Emit8(addr.disp);
  } else {
    // Otherwise use full-disp.
    (this->*EmitBase)(base_modrm_and_sib | 0x80);
    Emit32(addr.disp);
  }
}

}  // namespace x86_32

}  // namespace berberis

#endif  // BERBERIS_ASSEMBLER_X86_32_H_
