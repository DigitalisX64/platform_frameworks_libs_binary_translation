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

// Assembler to produce x86-64 instructions. Somewhat influenced by V8 assembler.

#ifndef BERBERIS_ASSEMBLER_X86_64_H_
#define BERBERIS_ASSEMBLER_X86_64_H_

#include <type_traits>  // std::is_same

#include "berberis/assembler/x86_32_or_x86_64.h"
#include "berberis/base/logging.h"

namespace berberis {

namespace x86_64 {

class Assembler : public x86_32_or_x86_64::Assembler<Assembler> {
 public:
  using BaseAssembler = x86_32_or_x86_64::Assembler<Assembler>;
  using FinalAssembler = Assembler;

  explicit Assembler(MachineCode* code) : BaseAssembler(code) {}

  static constexpr Register no_register{0x80};
  static constexpr Register rax{0};
  static constexpr Register rcx{1};
  static constexpr Register rdx{2};
  static constexpr Register rbx{3};
  static constexpr Register rsp{4};
  static constexpr Register rbp{5};
  static constexpr Register rsi{6};
  static constexpr Register rdi{7};
  static constexpr Register r8{8};
  static constexpr Register r9{9};
  static constexpr Register r10{10};
  static constexpr Register r11{11};
  static constexpr Register r12{12};
  static constexpr Register r13{13};
  static constexpr Register r14{14};
  static constexpr Register r15{15};

  static constexpr XMMRegister no_xmm_register{0x80};
  static constexpr XMMRegister xmm0{0};
  static constexpr XMMRegister xmm1{1};
  static constexpr XMMRegister xmm2{2};
  static constexpr XMMRegister xmm3{3};
  static constexpr XMMRegister xmm4{4};
  static constexpr XMMRegister xmm5{5};
  static constexpr XMMRegister xmm6{6};
  static constexpr XMMRegister xmm7{7};
  static constexpr XMMRegister xmm8{8};
  static constexpr XMMRegister xmm9{9};
  static constexpr XMMRegister xmm10{10};
  static constexpr XMMRegister xmm11{11};
  static constexpr XMMRegister xmm12{12};
  static constexpr XMMRegister xmm13{13};
  static constexpr XMMRegister xmm14{14};
  static constexpr XMMRegister xmm15{15};

  static constexpr YMMRegister no_ymm_register{0x80};
  static constexpr YMMRegister ymm0{0};
  static constexpr YMMRegister ymm1{1};
  static constexpr YMMRegister ymm2{2};
  static constexpr YMMRegister ymm3{3};
  static constexpr YMMRegister ymm4{4};
  static constexpr YMMRegister ymm5{5};
  static constexpr YMMRegister ymm6{6};
  static constexpr YMMRegister ymm7{7};
  static constexpr YMMRegister ymm8{8};
  static constexpr YMMRegister ymm9{9};
  static constexpr YMMRegister ymm10{10};
  static constexpr YMMRegister ymm11{11};
  static constexpr YMMRegister ymm12{12};
  static constexpr YMMRegister ymm13{13};
  static constexpr YMMRegister ymm14{14};
  static constexpr YMMRegister ymm15{15};

  // Macroassembler uses these names to support both x86-32 and x86-64 modes.
  static constexpr Register gpr_a{0};
  static constexpr Register gpr_c{1};
  static constexpr Register gpr_d{2};
  static constexpr Register gpr_b{3};
  static constexpr Register gpr_s{4};

// Instructions.
#include "berberis/assembler/gen_assembler_x86_64-inl.h"  // NOLINT generated file!

  // Historical curiosity: x86-32 mode has Movq for memory-to-xmm operations.
  // x86-64 added another one, with different opcode but since they are functionally equivalent
  // GNU Assembler and Clang use old one both in 32-bit mode and 64-bit mode thus we are doing
  // the same.

  // Unhide Decl(Mem) hidden by Decl(Reg).
  using BaseAssembler::Decl;

  // Unhide Decw(Mem) hidden by Decw(Reg).
  using BaseAssembler::Decw;

  // Unhide Incl(Mem) hidden by Incl(Reg).
  using BaseAssembler::Incl;

  // Unhide Incw(Mem) hidden by Incw(Reg).
  using BaseAssembler::Incw;

  // Unhide Movq(Mem, XMMReg) and Movq(XMMReg, Mem) hidden by Movq(Reg, Imm) and many others.
  using BaseAssembler::Movq;

  // Unhide Xchgl(Mem, Reg) hidden by modified version below.
  using BaseAssembler::Xchgl;

  // Unhide Vmov*(Mem, Reg) hidden by Vmov*(Reg, Reg).
  using BaseAssembler::Vmovapd;
  using BaseAssembler::Vmovaps;
  using BaseAssembler::Vmovdqa;
  using BaseAssembler::Vmovdqu;
  using BaseAssembler::Vmovq;
  using BaseAssembler::Vmovsd;
  using BaseAssembler::Vmovss;

  void Xchgl(Register dest, Register src) {
    // In 32-bit mode "xchgl %eax, %eax" did nothing and was often reused as "nop".
    //
    // On x86-64 "xchgl %eax, %eax" clears top half of %eax register, but having single-byte nop
    // is too convenient, thus, as special exception, 0x90 is not interpreted as "xchgl %eax, %eax",
    // but was kept as "nop" - thus longer encoding for "xchgl %eax, %eax" must be used.

    if (IsAccumulator(src) && IsAccumulator(dest)) {
      Emit16(0xc087);
    } else {
      BaseAssembler::Xchgl(dest, src);
    }
  }

  // TODO(b/127356868): decide what to do with these functions when cross-arch assembler is used.

#ifdef __amd64__

  // Unhide Call(Reg), hidden by special version below.
  using BaseAssembler::Call;

  void Call(const void* target) {
    // There are no call instruction with properties we need thus we emulate it.
    // This is what the following code looks like when decoded with objdump (if
    // target address is 0x1234'5678'9abc'def0):
    //   0: ff 15 02 00 00 00        callq  *0x2(%rip) # 0x8
    //   6: eb 08                    jmp    0x10
    //   8: f0 de bc 9a 78 56 34 12  lock fidivrs 0x1234'5678(%rdx,%rbx,4)
    // First we do call - with address taken from last 8 bytes, then we jump over
    // these 8 bytes.
    Emit64(0x08eb'0000'0002'15ff);
    Emit64(bit_cast<int64_t>(target));
  }

  // Unhide Jcc(Label), hidden by special version below.
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
    CHECK_EQ(0, static_cast<uint8_t>(cc) & 0xf0);
    // There are no Jcc instruction with properties we need thus we emulate it.
    // This is what the following code looks like when decoded with objdump (if
    // target address is 0x1234'5678'9abc'def0):
    //   0: 75 0e                   jne    0x10
    //   2: ff 25 00 00 00 00       jmpq   *0x0(%rip) # 0x8
    //   8: f0 de bc 9a 78 56 34 12 lock fidivrs 0x1234'5678(%rdx,%rbx,4)
    // We are doing relative jump for the inverted condition (because Jcc could
    // only jump ±2GiB and in 64 bit mode which is not enough to reach arbitrary
    // address), then jmpq with address stored right after jmpq.
    Emit64(0x0000'0000'25ff'0e70 | static_cast<int8_t>(ToReverseCond(cc)));
    Emit64(bit_cast<int64_t>(target));
  }

  // Emit short relative jcc to an absolute address.
  //
  // This is used to shorten jcc in the code installed in lower 2G address space.
  // Use this if the target is also within this address space.
  void Jcc32(Condition cc, uintptr_t target) {
    if (cc == Condition::kAlways) {
      Jmp32(target);
      return;
    } else if (cc == Condition::kNever) {
      return;
    }
    CHECK_EQ(static_cast<uint8_t>(cc) & 0xf0, 0);
    Emit8(0x0f);
    Emit8(0x80 | static_cast<uint8_t>(cc));
    Emit32(0xcccc'cccc);
    // Set last 4 bytes to displacement from current pc to 'target'.
    AddRelocation(pc() - 4, RelocationType::RelocAbsToDisp32, pc(), bit_cast<intptr_t>(target));
  }

  void Jcc(Condition cc, const void* target) { Jcc(cc, bit_cast<uintptr_t>(target)); }

  // Unhide Jmp(Reg), hidden by special version below.
  using BaseAssembler::Jmp;

  // Make sure only type void* can be passed to function below, not Label* or any other pointer.
  template <typename T>
  auto Jmp(T* target) -> void = delete;

  template <typename T>
  auto Jmp(T target)
      -> std::enable_if_t<std::is_integral_v<T> && sizeof(uintptr_t) < sizeof(T)> = delete;

  void Jmp(uintptr_t target) {
    // There are no jump instruction with properties we need thus we emulate it.
    // This is what the following code looks like when decoded with objdump (if
    // target address is 0x1234'5678'9abc'def0):
    //   0: ff 25 00 00 00 00       jmpq   *0x0(%rip) # 0x6
    //   6: f0 de bc 9a 78 56 34 12 lock fidivrs 0x1234'5678(%rdx,%rbx,4)
    // We are doing jump to the address stored right after jmpq using %rip-relative
    // addressing (with offset 0).
    Emit16(0x25ff);
    Emit32(0x0000'0000);
    Emit64(bit_cast<int64_t>(target));
  }

  // Emit short relative jump to an absolute address.
  //
  // This is used to shorten jmps in the code installed in lower 2G address space.
  // Use this if the target is also within this address space.
  void Jmp32(uintptr_t target) {
    Emit8(0xe9);
    Emit32(0xcccc'cccc);
    AddRelocation(pc() - 4, RelocationType::RelocAbsToDisp32, pc(), target);
  }

  void Jmp(const void* target) { Jmp(bit_cast<uintptr_t>(target)); }

#endif

 private:
  Assembler() = delete;
  Assembler(const Assembler&) = delete;
  Assembler(Assembler&&) = delete;
  void operator=(const Assembler&) = delete;
  void operator=(Assembler&&) = delete;
  using DerivedAssemblerType = Assembler;

  static Register Accumulator() { return rax; }
  static bool IsAccumulator(Register reg) { return reg == rax; }

  struct Register64Bit {
    explicit constexpr Register64Bit(Register reg) : num_(reg.num_) {}
    uint8_t num_;
  };

  struct Memory64Bit {
    explicit Memory64Bit(const Operand& op) : operand(op) {}
    Operand operand;
  };

  struct Label64Bit {
    explicit Label64Bit(const LabelOperand& l) : label(l.label) {}
    const Label& label;
  };

  // This type is only used by CmpXchg16b and acts similarly to Memory64Bit there.
  using Memory128Bit = Memory64Bit;
  using Label128Bit = Label64Bit;

  // Check if a given type is "a register with size" (for EmitInstruction).
  template <typename ArgumentType>
  struct IsRegister {
    static constexpr bool value = std::is_same_v<ArgumentType, Register8Bit> ||
                                  std::is_same_v<ArgumentType, Register32Bit> ||
                                  std::is_same_v<ArgumentType, Register64Bit>;
  };

  // Check if a given type is "a memory operand with size" (for EmitInstruction).
  template <typename ArgumentType>
  struct IsMemoryOperand {
    static constexpr bool value =
        std::is_same_v<ArgumentType, Memory32Bit> || std::is_same_v<ArgumentType, Memory64Bit>;
  };

  template <typename ArgumentType>
  struct IsLabelOperand {
    static constexpr bool value =
        std::is_same_v<ArgumentType, Label32Bit> || std::is_same_v<ArgumentType, Label64Bit>;
  };

  template <typename... ArgumentsTypes>
  void EmitRex(ArgumentsTypes... arguments) {
    constexpr auto registers_count = kCountArguments<IsRegister, ArgumentsTypes...>;
    constexpr auto operands_count = kCountArguments<IsMemoryOperand, ArgumentsTypes...>;
    static_assert(registers_count + operands_count <= 2,
                  "Only two-arguments instructions are supported, not VEX or EVEX");
    uint8_t rex = 0;
    if constexpr (registers_count == 2) {
      rex = Rex<0b0100>(ArgumentByType<0, IsRegister>(arguments...)) |
            Rex<0b0001>(ArgumentByType<1, IsRegister>(arguments...));
    } else if constexpr (registers_count == 1 && operands_count == 1) {
      rex = Rex<0b0100>(ArgumentByType<0, IsRegister>(arguments...)) |
            Rex(ArgumentByType<0, IsMemoryOperand>(arguments...));
    } else if constexpr (registers_count == 1) {
      rex = Rex<0b0001>(ArgumentByType<0, IsRegister>(arguments...));
    } else if constexpr (operands_count == 1) {
      rex = Rex(ArgumentByType<0, IsMemoryOperand>(arguments...));
    }
    if (rex) {
      Emit8(rex);
    }
  }

  template <uint8_t base_rex, typename ArgumentType>
  uint8_t Rex(ArgumentType argument) {
    if (argument.num_ & 0b1000) {
      // 64-bit argument requires REX.W bit
      if (std::is_same_v<ArgumentType, Register64Bit>) {
        return 0b0100'1000 | base_rex;
      }
      return 0b0100'0000 | base_rex;
    }
    // 8-bit argument requires REX (even if without any bits).
    if (std::is_same_v<ArgumentType, Register8Bit> && argument.num_ > 3) {
      return 0b0100'0000;
    }
    if (std::is_same_v<ArgumentType, Register64Bit>) {
      return 0b0100'1000;
    }
    return 0;
  }

  uint8_t Rex(Operand operand) {
    // REX.B and REX.X always come from operand.
    uint8_t rex = ((operand.base.num_ & 0b1000) >> 3) | ((operand.index.num_ & 0b1000) >> 2);
    if (rex) {
      // We actually need rex byte here.
      return 0b0100'0000 | rex;
    } else {
      return 0;
    }
  }

  uint8_t Rex(Memory32Bit operand) { return Rex(operand.operand); }

  uint8_t Rex(Memory64Bit operand) {
    // 64-bit argument requires REX.W bit - and thus REX itself.
    return 0b0100'1000 | Rex(operand.operand);
  }

  template <typename RegisterType>
  [[nodiscard]] static bool IsSwapProfitable(RegisterType rm_arg, RegisterType vex_arg) {
    // In 64bit mode we may use more compact encoding if operand encoded in rm is low register.
    // Return true if we may achieve that by swapping arguments.
    return rm_arg.num_ >= 8 && vex_arg.num_ < 8;
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
    auto vex2 = byte2 | 0b111'00000;
    if constexpr (operands_count == 1) {
      auto operand = ArgumentByType<0, IsMemoryOperand>(arguments...);
      vex2 ^= (operand.operand.base.num_ & 0b1000) << 2;
      vex2 ^= (operand.operand.index.num_ & 0b1000) << 3;
      if constexpr (!reg_is_opcode_extension) {
        vex2 ^= (ArgumentByType<0, IsRegister>(arguments...).num_ & 0b1000) << 4;
      }
    } else if constexpr (labels_count == 1) {
      if constexpr (!reg_is_opcode_extension) {
        vex2 ^= (ArgumentByType<0, IsRegister>(arguments...).num_ & 0b1000) << 4;
      }
    } else if constexpr (registers_count > 0) {
      if constexpr (reg_is_opcode_extension) {
        vex2 ^= (ArgumentByType<0, IsRegister>(arguments...).num_ & 0b1000) << 2;
      } else {
        vex2 ^= (ArgumentByType<0, IsRegister>(arguments...).num_ & 0b1000) << 4;
        vex2 ^= (ArgumentByType<1, IsRegister>(arguments...).num_ & 0b1000) << 2;
      }
    }
    if (byte1 == 0xc4 && (vex2 & 0b0'1'1'11111) == 0b0'1'1'00001 && (byte3 & 0b1'0000'0'00) == 0) {
      Emit16((0xc5 | ((vex2 & 0b1'0'0'00000) << 8) | (byte3 << 8) |
              0b0'1111'000'00000000) ^ (vvvv << 11));
    } else {
      Emit8(byte1);
      Emit16((vex2 | (byte3 << 8) | 0b0'1111'000'00000000) ^ (vvvv << 11));
    }
  }

  template <typename ArgumentType>
  void EmitRegisterInOpcode(uint8_t opcode, ArgumentType argument) {
    Emit8(opcode | (argument.num_ & 0b111));
  }

  template <typename ArgumentType1, typename ArgumentType2>
  void EmitModRM(ArgumentType1 argument1, ArgumentType2 argument2) {
    Emit8(0xc0 | ((argument1.num_ & 0b111) << 3) | (argument2.num_ & 0b111));
  }

  template <typename ArgumentType>
  void EmitModRM(uint8_t opcode_extension, ArgumentType argument) {
    CHECK_LE(opcode_extension, 0b111);
    Emit8(0xc0 | (opcode_extension << 3) | (argument.num_ & 0b111));
  }

  template <typename ArgumentType>
  constexpr void EmitOperandOp(ArgumentType argument, Operand operand) {
    EmitOperandOp(static_cast<int>(argument.num_ & 0b111), operand);
  }

  template <size_t kImmediatesSize, typename ArgumentType>
  constexpr void EmitRipOp(ArgumentType argument, const Label& label) {
    EmitRipOp<kImmediatesSize>(static_cast<int>(argument.num_) & 0b111, label);
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

  // On x86 %rsp cannot be index, only base.
  CHECK(addr.index != rsp);

  // If base is not %rsp/r12 and we don't have index, then we don't have SIB byte.
  // All other cases have "ModR/M" and SIB bytes.
  if (addr.base != rsp && addr.base != r12 && addr.index == no_register) {
    // If we have base register then we could use the same logic as for other common cases.
    if (addr.base != no_register) {
      EmitBaseIndexDispOperand<uint8_t, &Assembler::Emit8>((addr.base.num_ & 7) | reg, addr);
    } else {
      Emit16(0x2504 | reg);
      Emit32(addr.disp);
    }
  } else if (addr.index == no_register) {
    // Note: when ModR/M and SIB are used "no index" is encoded as if %rsp is used in place of
    // index (that's why %rsp couldn't be used as index - see check above).
    EmitBaseIndexDispOperand<int16_t, &Assembler::Emit16>(
        0x2004 | ((addr.base.num_ & 7) << 8) | reg, addr);
  } else if (addr.base == no_register) {
    EmitIndexDispOperand(reg, addr);
  } else {
    EmitBaseIndexDispOperand<int16_t, &Assembler::Emit16>(0x04 | (addr.scale << 14) |
                                                              ((addr.index.num_ & 7) << 11) |
                                                              ((addr.base.num_ & 7) << 8) | reg,
                                                          addr);
  }
}

constexpr inline void Assembler::EmitIndexDispOperand(int reg, const Operand& addr) {
  // We only have index here, no base, use SIB but put %rbp in "base" field.
  Emit16(0x0504 | (addr.scale << 14) | ((addr.index.num_ & 7) << 11) | reg);
  Emit32(addr.disp);
}

template <size_t kImmediatesSize>
constexpr inline void Assembler::EmitRipOp(int num_, const Label& label) {
  Emit8(0x05 | (num_ << 3));
  jumps_.push_back(Jump{&label, pc(), false});
  Emit32(0xffff'fffc - kImmediatesSize);
}

template <typename ArgType, void (AssemblerBase::* EmitBase)(ArgType)>
constexpr inline void Assembler::EmitBaseIndexDispOperand(int base_modrm_and_sib,
                                                          const Operand& addr) {
  if (addr.disp == 0 && addr.base != rbp && addr.base != r13) {
    // We can omit zero displacement only if base isn't %rbp/%r13
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

constexpr inline void Assembler::Movq(Register dest, int64_t imm64) {
  if (IsInRange<uint32_t>(imm64)) {
    // Shorter encoding.
    Movl(dest, static_cast<uint32_t>(imm64));
  } else if (IsInRange<int32_t>(imm64)) {
    // Slightly longer encoding.
    EmitInstruction<0xc7, 0x00>(Register64Bit(dest), static_cast<int32_t>(imm64));
  } else {
    // Longest encoding.
    EmitInstruction<0xb8>(Register64Bit(dest), imm64);
  }
}

constexpr inline void Assembler::Vmovapd(XMMRegister arg0, XMMRegister arg1) {
  if (IsSwapProfitable(arg1, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x01, 0x29>(VectorRegister128Bit(arg1),
                                                   VectorRegister128Bit(arg0));
  }
  EmitInstruction<0xc4, 0x01, 0x01, 0x28>(VectorRegister128Bit(arg0), VectorRegister128Bit(arg1));
}

constexpr inline void Assembler::Vmovapd(YMMRegister arg0, YMMRegister arg1) {
  if (IsSwapProfitable(arg1, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x05, 0x29>(VectorRegister256Bit(arg1),
                                                   VectorRegister256Bit(arg0));
  }
  EmitInstruction<0xc4, 0x01, 0x05, 0x28>(VectorRegister256Bit(arg0), VectorRegister256Bit(arg1));
}

constexpr inline void Assembler::Vmovaps(XMMRegister arg0, XMMRegister arg1) {
  if (IsSwapProfitable(arg1, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x00, 0x29>(VectorRegister128Bit(arg1),
                                                   VectorRegister128Bit(arg0));
  }
  EmitInstruction<0xc4, 0x01, 0x00, 0x28>(VectorRegister128Bit(arg0), VectorRegister128Bit(arg1));
}

constexpr inline void Assembler::Vmovaps(YMMRegister arg0, YMMRegister arg1) {
  if (IsSwapProfitable(arg1, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x04, 0x29>(VectorRegister256Bit(arg1),
                                                   VectorRegister256Bit(arg0));
  }
  EmitInstruction<0xc4, 0x01, 0x04, 0x28>(VectorRegister256Bit(arg0), VectorRegister256Bit(arg1));
}

constexpr inline void Assembler::Vmovdqa(XMMRegister arg0, XMMRegister arg1) {
  if (IsSwapProfitable(arg1, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x01, 0x7f>(VectorRegister128Bit(arg1),
                                                   VectorRegister128Bit(arg0));
  }
  EmitInstruction<0xc4, 0x01, 0x01, 0x6f>(VectorRegister128Bit(arg0), VectorRegister128Bit(arg1));
}

constexpr inline void Assembler::Vmovdqa(YMMRegister arg0, YMMRegister arg1) {
  if (IsSwapProfitable(arg1, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x05, 0x7f>(VectorRegister256Bit(arg1),
                                                   VectorRegister256Bit(arg0));
  }
  EmitInstruction<0xc4, 0x01, 0x05, 0x6f>(VectorRegister256Bit(arg0), VectorRegister256Bit(arg1));
}

constexpr inline void Assembler::Vmovdqu(XMMRegister arg0, XMMRegister arg1) {
  if (IsSwapProfitable(arg1, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x02, 0x7f>(VectorRegister128Bit(arg1),
                                                   VectorRegister128Bit(arg0));
  }
  EmitInstruction<0xc4, 0x01, 0x02, 0x6f>(VectorRegister128Bit(arg0), VectorRegister128Bit(arg1));
}

constexpr inline void Assembler::Vmovdqu(YMMRegister arg0, YMMRegister arg1) {
  if (IsSwapProfitable(arg1, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x06, 0x7f>(VectorRegister256Bit(arg1),
                                                   VectorRegister256Bit(arg0));
  }
  EmitInstruction<0xc4, 0x01, 0x06, 0x6f>(VectorRegister256Bit(arg0), VectorRegister256Bit(arg1));
}

constexpr inline void Assembler::Vmovsd(XMMRegister arg0, XMMRegister arg1, XMMRegister arg2) {
  if (IsSwapProfitable(arg2, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x03, 0x11>(
        VectorRegister128Bit(arg2), VectorRegister128Bit(arg0), VectorRegister128Bit(arg1));
  }
  EmitInstruction<0xc4, 0x01, 0x03, 0x10>(
      VectorRegister128Bit(arg0), VectorRegister128Bit(arg2), VectorRegister128Bit(arg1));
}

constexpr inline void Assembler::Vmovss(XMMRegister arg0, XMMRegister arg1, XMMRegister arg2) {
  if (IsSwapProfitable(arg2, arg0)) {
    return EmitInstruction<0xc4, 0x01, 0x02, 0x11>(
        VectorRegister128Bit(arg2), VectorRegister128Bit(arg0), VectorRegister128Bit(arg1));
  }
  EmitInstruction<0xc4, 0x01, 0x02, 0x10>(
      VectorRegister128Bit(arg0), VectorRegister128Bit(arg2), VectorRegister128Bit(arg1));
}

constexpr inline void Assembler::Xchgq(Register dest, Register src) {
  // We compare output to that from clang and thus want to produce the same code.
  // 0x48 0x90 is suboptimal encoding for that operation (pure 0x90 does the same
  // and this is what gcc + gas are producing), but this is what clang <= 8 does.
  if (IsAccumulator(src) && IsAccumulator(dest)) {
    Emit8(0x90);
  } else if (IsAccumulator(src) || IsAccumulator(dest)) {
    Register other = IsAccumulator(src) ? dest : src;
    EmitInstruction<0x90>(Register64Bit(other));
  } else {
  // Clang 8 (after r330298) puts dest before src.  We are comparing output
  // to clang in exhaustive test thus we want to match clang behavior exactly.
  EmitInstruction<0x87>(Register64Bit(dest), Register64Bit(src));
  }
}

}  // namespace x86_64

}  // namespace berberis

#endif  // BERBERIS_ASSEMBLER_X86_64_H_
