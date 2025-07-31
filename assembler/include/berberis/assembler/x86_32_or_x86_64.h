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

#ifndef BERBERIS_ASSEMBLER_X86_32_OR_X86_64_H_
#define BERBERIS_ASSEMBLER_X86_32_OR_X86_64_H_

#include <cstddef>  // std::size_t
#include <cstdint>
#include <type_traits>  // std::enable_if, std::is_integral

#include "berberis/assembler/common.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"

namespace berberis {

// Assembler includes implementation of most x86 assembler instructions.
//
// x86-32 and x86-64 assemblers are nearly identical, but difference lies in handling
// of very low-level instruction details: almost all instructions on x86-64 could include
// REX byte which is needed if new registers (%r8 to %r15 or %xmm8 to %xmm15) are used.
//
// To handle that difference efficiently Assembler is CRTP class: it's parameterized
// by its own descendant and pull certain functions (e.g. GetHighBit or Rex8Size) from
// its implementation.
//
// Certain functions are only implemented by its descendant (since there are instructions
// which only exist in x86-32 mode and instructions which only exist in x86-64 mode).

namespace x86_32 {

class Assembler;

}  // namespace x86_32

namespace x86_64 {

class Assembler;

}  // namespace x86_64

namespace x86_32_or_x86_64 {

enum class Condition : int8_t {
  kInvalidCondition = -1,

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
  kAlways = 16,
  kNever = 17,

  // aka...
  kCarry = kBelow,
  kNotCarry = kAboveEqual,
  kZero = kEqual,
  kNotZero = kNotEqual,
  kSign = kNegative,
  kNotSign = kPositiveOrZero
};

constexpr const char* GetCondName(Condition cond) {
  switch (cond) {
    case Condition::kOverflow:
      return "O";
    case Condition::kNoOverflow:
      return "NO";
    case Condition::kBelow:
      return "B";
    case Condition::kAboveEqual:
      return "AE";
    case Condition::kEqual:
      return "Z";
    case Condition::kNotEqual:
      return "NZ";
    case Condition::kBelowEqual:
      return "BE";
    case Condition::kAbove:
      return "A";
    case Condition::kNegative:
      return "N";
    case Condition::kPositiveOrZero:
      return "PL";
    case Condition::kParityEven:
      return "PE";
    case Condition::kParityOdd:
      return "PO";
    case Condition::kLess:
      return "LS";
    case Condition::kGreaterEqual:
      return "GE";
    case Condition::kLessEqual:
      return "LE";
    case Condition::kGreater:
      return "GT";
    default:
      return "??";
  }
}

enum ScaleFactor : int8_t { kTimesOne = 0, kTimesTwo = 1, kTimesFour = 2, kTimesEight = 3 };

template <typename DerivedAssemblerType>
class Assembler : public AssemblerBase {
 public:
  explicit Assembler(MachineCode* code) : AssemblerBase(code) {}

  using Condition = x86_32_or_x86_64::Condition;

  class Register {
   public:
    constexpr bool operator==(const Register&) const = default;
    constexpr uint8_t GetPhysicalIndex() { return num_; }
    friend constexpr uint8_t ValueForFmtSpec(Register value) { return value.num_; }
    friend class Assembler<DerivedAssemblerType>;
    friend class x86_32::Assembler;
    friend class x86_64::Assembler;

   private:
    explicit constexpr Register(uint8_t num) : num_(num) {}
    uint8_t num_;
  };

  class X87Register {
   public:
    constexpr bool operator==(const X87Register&) const = default;
    constexpr uint8_t GetPhysicalIndex() { return num_; }
    friend constexpr uint8_t ValueForFmtSpec(X87Register value) { return value.num_; }
    friend class Assembler<DerivedAssemblerType>;
    friend class x86_32::Assembler;
    friend class x86_64::Assembler;

   private:
    explicit constexpr X87Register(uint8_t num) : num_(num) {}
    uint8_t num_;
  };

  static constexpr X87Register st{0};
  static constexpr X87Register st0{0};
  static constexpr X87Register st1{1};
  static constexpr X87Register st2{2};
  static constexpr X87Register st3{3};
  static constexpr X87Register st4{4};
  static constexpr X87Register st5{5};
  static constexpr X87Register st6{6};
  static constexpr X87Register st7{7};

  template <int kBits>
  class SIMDRegister {
   public:
    constexpr bool operator==(const SIMDRegister&) const = default;
    constexpr uint8_t GetPhysicalIndex() { return num_; }
    friend constexpr uint8_t ValueForFmtSpec(SIMDRegister value) { return value.num_; }
    friend class Assembler<DerivedAssemblerType>;
    friend class x86_32::Assembler;
    friend class x86_64::Assembler;
    friend class SIMDRegister<384 - kBits>;

    constexpr auto To128Bit() const {
      return std::enable_if_t < kBits != 128, SIMDRegister < 128 >> {num_};
    }
    constexpr auto To256Bit() const {
      return std::enable_if_t < kBits != 256, SIMDRegister < 256 >> {num_};
    }

   private:
    explicit constexpr SIMDRegister(uint8_t num) : num_(num) {}
    uint8_t num_;
  };

  using XMMRegister = SIMDRegister<128>;
  using YMMRegister = SIMDRegister<256>;

  using ScaleFactor = x86_32_or_x86_64::ScaleFactor;
  static constexpr ScaleFactor kTimesOne = ScaleFactor::kTimesOne;
  static constexpr ScaleFactor kTimesTwo = ScaleFactor::kTimesTwo;
  static constexpr ScaleFactor kTimesFour = ScaleFactor::kTimesFour;
  static constexpr ScaleFactor kTimesEight = ScaleFactor::kTimesEight;

  struct Operand {
    constexpr bool operator==(const Operand&) const = default;
    constexpr uint8_t rex() const {
      return DerivedAssemblerType::kIsX86_64
                 ? ((index.num_ & 0x08) >> 2) | ((base.num_ & 0x08) >> 3)
                 : 0;
    }

    constexpr bool RequiresRex() const {
      return DerivedAssemblerType::kIsX86_64 ? ((index.num_ & 0x08) | (base.num_ & 0x08)) : false;
    }

    Register base = DerivedAssemblerType::no_register;
    Register index = DerivedAssemblerType::no_register;
    ScaleFactor scale = kTimesOne;
    int32_t disp = 0;
  };

  struct LabelOperand {
    const Label& label;
  };

  // Macro operations.
  void Finalize() { ResolveJumps(); }

  void P2Align(uint32_t m) {
    uint32_t mask = m - 1;
    uint32_t addr = pc();
    Nop((m - (addr & mask)) & mask);
  }

  void Nop(uint32_t bytes) {
    static const uint32_t kNumNops = 15;
    static const uint8_t nop1[] = {0x90};
    static const uint8_t nop2[] = {0x66, 0x90};
    static const uint8_t nop3[] = {0x0f, 0x1f, 0x00};
    static const uint8_t nop4[] = {0x0f, 0x1f, 0x40, 0x00};
    static const uint8_t nop5[] = {0x0f, 0x1f, 0x44, 0x00, 0x00};
    static const uint8_t nop6[] = {0x66, 0x0f, 0x1f, 0x44, 0x00, 0x0};
    static const uint8_t nop7[] = {0x0f, 0x1f, 0x80, 0x00, 0x00, 0x0, 0x00};
    static const uint8_t nop8[] = {0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};
    static const uint8_t nop9[] = {0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};
    static const uint8_t nop10[] = {0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};
    static const uint8_t nop11[] = {
        0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};
    static const uint8_t nop12[] = {
        0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};
    static const uint8_t nop13[] = {
        0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};
    static const uint8_t nop14[] = {
        0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};
    static const uint8_t nop15[] = {
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};

    static const uint8_t* nops[kNumNops] = {nop1,
                                            nop2,
                                            nop3,
                                            nop4,
                                            nop5,
                                            nop6,
                                            nop7,
                                            nop8,
                                            nop9,
                                            nop10,
                                            nop11,
                                            nop12,
                                            nop13,
                                            nop14,
                                            nop15};
    // Common case.
    if (bytes == 1) {
      Emit8(nop1[0]);
      return;
    }

    while (bytes > 0) {
      uint32_t len = bytes;
      if (len > kNumNops) {
        len = kNumNops;
      }
      EmitSequence(nops[len - 1], len);
      bytes -= len;
    }
  }

// Instructions.
#include "berberis/assembler/gen_assembler_x86_32_or_x86_64-inl.h"  // NOLINT generated file

  // Flow control.
  void JmpRel(int32_t offset) {
    CHECK_GE(offset, INT32_MIN + 2);
    int32_t short_offset = offset - 2;
    if (IsInRange<int8_t>(short_offset)) {
      Emit8(0xeb);
      Emit8(static_cast<int8_t>(short_offset));
    } else {
      CHECK_GE(offset, INT32_MIN + 5);
      Emit8(0xe9);
      Emit32(offset - 5);
    }
  }

  void Call(int32_t offset) {
    CHECK_GE(offset, INT32_MIN + 5);
    Emit8(0xe8);
    Emit32(offset - 5);
  }

  void JccRel(Condition cc, int32_t offset) {
    if (cc == Condition::kAlways) {
      JmpRel(offset);
      return;
    }
    if (cc == Condition::kNever) {
      return;
    }
    CHECK_EQ(0, static_cast<uint8_t>(cc) & 0xf0);
    CHECK_GE(offset, INT32_MIN + 2);
    int32_t short_offset = offset - 2;
    if (IsInRange<int8_t>(short_offset)) {
      Emit8(0x70 | static_cast<uint8_t>(cc));
      Emit8(static_cast<int8_t>(short_offset));
    } else {
      CHECK_GE(offset, INT32_MIN + 6);
      Emit8(0x0f);
      Emit8(0x80 | static_cast<uint8_t>(cc));
      Emit32(offset - 6);
    }
  }

 protected:
  // Helper types to distinguish argument types.
  struct Register8Bit {
    explicit constexpr Register8Bit(Register reg) : num_(reg.num_) {}
    uint8_t num_;
  };

  // Any register number that doesn't need special processing.
  struct SizeAgnosticRegister {
    explicit constexpr SizeAgnosticRegister(Register reg) : num_(reg.num_) {}
    explicit constexpr SizeAgnosticRegister(XMMRegister reg) : num_(reg.num_) {}
    explicit constexpr SizeAgnosticRegister(YMMRegister reg) : num_(reg.num_) {}
    uint8_t num_;
  };

  // 16-bit, 32bit, 128-bit, and 256bit vector registers don't need special rules.
  using Register16Bit = SizeAgnosticRegister;
  using Register32Bit = SizeAgnosticRegister;
  using VectorRegister128Bit = SizeAgnosticRegister;
  using VectorRegister256Bit = SizeAgnosticRegister;
  // Certain instructions (Enter/Leave, Jcc/Jmp/Loop, Call/Ret, Push/Pop) always operate
  // on registers of default size (32-bit in 32-bit mode, 64-bit in 64-bit mode (see
  // "Instructions Not Requiring REX Prefix in 64-Bit Mode" table in 24594 AMD Manual)
  // Map these to SizeAgnosticRegister, too, since they don't need REX.W even in 64-bit mode.
  //
  // x87 instructions fall into that category, too, since they were not expanded in x86-64 mode.
  using RegisterDefaultBit = SizeAgnosticRegister;

  // Any memory address that doesn't need special processing.
  struct SizeAgnosticMemory {
    explicit SizeAgnosticMemory(const Operand& op) : operand(op) {}
    Operand operand;
  };

  // 8-bit, 16-bit, 128-bit memory behave the same as 32-bit memory.
  // Only 64-bit memory is different.
  using Memory8Bit = SizeAgnosticMemory;
  using Memory16Bit = SizeAgnosticMemory;
  using Memory32Bit = SizeAgnosticMemory;
  // Some instructions have memory operand that have unspecified size (lea, prefetch, etc),
  // they are encoded like SizeAgnosticMemory, anyway.
  using MemoryDefaultBit = SizeAgnosticMemory;
  // X87 instructions always use the same encoding - even for 64-bit or 28-bytes
  // memory operands (like in fldenv/fnstenv)
  using MemoryX87 = SizeAgnosticMemory;
  using MemoryX8716Bit = SizeAgnosticMemory;
  using MemoryX8732Bit = SizeAgnosticMemory;
  using MemoryX8764Bit = SizeAgnosticMemory;
  using MemoryX8780Bit = SizeAgnosticMemory;
  // Most vector instructions don't need to use REX.W to access 64-bit or 128-bit memory.
  using VectorMemory32Bit = SizeAgnosticMemory;
  using VectorMemory64Bit = SizeAgnosticMemory;
  using VectorMemory128Bit = SizeAgnosticMemory;
  using VectorMemory256Bit = SizeAgnosticMemory;

  // Labels types for memory quantities.  Note that names are similar to the ones before because
  // they are autogenerated.  E.g. VectorLabel32Bit should be read as “VECTOR's operation LABEL
  // for 32-BIT quantity in memory”.
  struct Label32Bit {
    explicit Label32Bit(const struct LabelOperand& l) : label(l.label) {}
    const Label& label;
  };

  // 8-bit, 16-bit, 128-bit memory behave the same as 32-bit memory.
  // Only 64-bit memory is different.
  using Label8Bit = Label32Bit;
  using Label16Bit = Label32Bit;
  // Some instructions have memory operand that have unspecified size (lea, prefetch, etc),
  // they are encoded like Label32Bit, anyway.
  using LabelDefaultBit = Label32Bit;
  // X87 instructions always use the same encoding - even for 64-bit or 28-bytes
  // memory operands (like in fldenv/fnstenv)
  using LabelX87 = Label32Bit;
  using LabelX8716Bit = Label32Bit;
  using LabelX8732Bit = Label32Bit;
  using LabelX8764Bit = Label32Bit;
  using LabelX8780Bit = Label32Bit;
  // Most vector instructions don't need to use REX.W to access 64-bit or 128-bit memory.
  using VectorLabel32Bit = Label32Bit;
  using VectorLabel64Bit = Label32Bit;
  using VectorLabel128Bit = Label32Bit;

  static constexpr bool IsLegacyPrefix(int code) {
    // Legacy prefixes used as opcode extensions in SSE.
    // Lock is used by cmpxchg.
    return (code == 0x66) || (code == 0xf2) || (code == 0xf3) || (code == 0xf0);
  }

  // Delegate check to Assembler::template IsRegister.
  template <typename ArgumentType>
  struct IsCondition {
    static constexpr bool value = std::is_same_v<ArgumentType, Condition>;
  };

  template <typename ArgumentType>
  struct IsRegister {
    static constexpr bool value = DerivedAssemblerType::template IsRegister<ArgumentType>::value ||
                                  std::is_same_v<ArgumentType, X87Register>;
  };

  template <typename ArgumentType>
  struct IsMemoryOperand {
    static constexpr bool value =
        DerivedAssemblerType::template IsMemoryOperand<ArgumentType>::value;
  };

  template <typename ArgumentType>
  struct IsLabelOperand {
    static constexpr bool value =
        DerivedAssemblerType::template IsLabelOperand<ArgumentType>::value;
  };

  template <typename ArgumentType>
  struct IsImmediate {
    static constexpr bool value =
        std::is_integral_v<ArgumentType> &&
        ((sizeof(ArgumentType) == sizeof(int8_t)) || (sizeof(ArgumentType) == sizeof(int16_t)) ||
         (sizeof(ArgumentType) == sizeof(int32_t)) || (sizeof(ArgumentType) == sizeof(int64_t)));
  };

  // Count number of arguments selected by Predicate.
  template <template <typename> typename Predicate, typename... ArgumentTypes>
  static constexpr std::size_t kCountArguments =
      ((Predicate<ArgumentTypes>::value ? 1 : 0) + ... + 0);

  // Extract arguments selected by Predicate.
  //
  // Note: This interface begs for the trick used in EmitFunctionTypeHelper in make_intrinsics.cc
  // in conjunction with structured bindings.
  //
  // Unfortunately returning std::tuple slows down AssemblerTest by about 30% when libc++ and clang
  // are used together (no slowdown on GCC, no slowdown on clang+libstdc++).
  //
  // TODO(http://b/140721204): refactor when it would be safe to return std::tuple from function.
  //
  template <std::size_t index,
            template <typename>
            typename Predicate,
            typename ArgumentType,
            typename... ArgumentTypes>
  static constexpr auto ArgumentByType(ArgumentType argument, ArgumentTypes... arguments) {
    if constexpr (Predicate<std::decay_t<ArgumentType>>::value) {
      if constexpr (index == 0) {
        return argument;
      } else {
        return ArgumentByType<index - 1, Predicate>(arguments...);
      }
    } else {
      return ArgumentByType<index, Predicate>(arguments...);
    }
  }

  // Emit immediates - they always come at the end and don't affect anything except rip-addressig.
  static constexpr void EmitImmediates() {}

  template <typename FirstArgumentType, typename... ArgumentTypes>
  void EmitImmediates(FirstArgumentType first_argument, ArgumentTypes... other_arguments) {
    if constexpr (std::is_integral_v<FirstArgumentType> &&
                  sizeof(FirstArgumentType) == sizeof(int8_t)) {
      Emit8(first_argument);
    } else if constexpr (std::is_integral_v<FirstArgumentType> &&
                         sizeof(FirstArgumentType) == sizeof(int16_t)) {
      Emit16(first_argument);
    } else if constexpr (std::is_integral_v<FirstArgumentType> &&
                         sizeof(FirstArgumentType) == sizeof(int32_t)) {
      Emit32(first_argument);
    } else if constexpr (std::is_integral_v<FirstArgumentType> &&
                         sizeof(FirstArgumentType) == sizeof(int64_t)) {
      Emit64(first_argument);
    }
    EmitImmediates(other_arguments...);
  }

  template <typename ArgumentType>
  static constexpr size_t ImmediateSize() {
    if constexpr (std::is_integral_v<ArgumentType> && sizeof(ArgumentType) == sizeof(int8_t)) {
      return 1;
    } else if constexpr (std::is_integral_v<ArgumentType> &&
                         sizeof(ArgumentType) == sizeof(int16_t)) {
      return 2;
    } else if constexpr (std::is_integral_v<ArgumentType> &&
                         sizeof(ArgumentType) == sizeof(int32_t)) {
      return 4;
    } else if constexpr (std::is_integral_v<ArgumentType> &&
                         sizeof(ArgumentType) == sizeof(int64_t)) {
      return 8;
    } else {
      static_assert(!std::is_integral_v<ArgumentType>);
      return 0;
    }
  }

  template <typename... ArgumentTypes>
  static constexpr size_t ImmediatesSize() {
    return (ImmediateSize<ArgumentTypes>() + ... + 0);
  }

  // Note: We may need separate x87 EmitInstruction if we would want to support
  // full set of x86 instructions.
  //
  // That's because 8087 was completely separate piece of silicone which was only
  // partially driven by 8086:
  //     https://en.wikipedia.org/wiki/Intel_8087
  //
  // In particular it had the following properties:
  //   1. It had its own separate subset of opcodes - because it did its own decoding.
  //   2. It had separate set of registers and could *only* access these.
  //   2a. The 8086, in turn, *couldn't* access these registers at all.
  //   3. To access memory it was designed to take address from address bus.
  //
  // This means that:
  //   1. x87 instructions are easily recognizable - all instructions with opcodes 0xd8
  //      to 0xdf are x87 instructions, all instructions with other opcodes are not.
  //   2. We could be sure that x87 registers would only be used with x87 instructions
  //      and other types of registers wouldn't be used with these.
  //   3. We still would use normal registers for memory access, but REX.W bit wouldn't
  //      be used for 64-bit quantities, whether they are floating point numbers or integers.
  //
  // Right now we only use EmitInstruction to emit x87 instructions which are using memory
  // operands - and it works well enough for that because of #3.

  // If you want to understand how this function works (and how helper function like Vex and
  // Rex work), you need good understanding of AMD/Intel Instruction format.
  //
  // Intel manual includes the most precise explanation, but it's VERY hard to read.
  //
  // AMD manual is much easier to read, but it doesn't include description of EVEX
  // instructions and is less precise. Diagram on page 2 of Volume 3 is especially helpful:
  //   https://www.amd.com/system/files/TechDocs/24594.pdf#page=42
  //
  // And the most concise (albeit unofficial) in on osdev Wiki:
  //   https://wiki.osdev.org/X86-64_Instruction_Encoding

  // Note: if you change this function (or any of the helper functions) then remove --fast
  // option from ExhaustiveAssemblerTest to run full blackbox comparison to clang.

  template <uint8_t... kOpcodes, typename... ArgumentsTypes>
  void EmitInstruction(ArgumentsTypes... arguments) {
    static constexpr auto kOpcodesArray = std::array{kOpcodes...};
    static constexpr size_t kLegacyPrefixesCount = []() {
      size_t legacy_prefixes_count = 0;
      for (legacy_prefixes_count = 0; IsLegacyPrefix(kOpcodesArray[legacy_prefixes_count]);
           ++legacy_prefixes_count) {
      }
      return legacy_prefixes_count;
    }();
    for (size_t legacy_prefixes_index = 0; legacy_prefixes_index < kLegacyPrefixesCount;
         ++legacy_prefixes_index) {
      Emit8(kOpcodesArray[legacy_prefixes_index]);
    }
    // We don't yet support any XOP-encoded instructions, but they are 100% identical to vex ones,
    // except they are using 0x8f prefix, not 0xc4 prefix.
    constexpr auto kVexOrXop = []() {
      if constexpr (std::size(kOpcodesArray) < kLegacyPrefixesCount + 3) {
        return false;
        // Note that JSON files use AMD approach: bytes are specified as in AMD manual (only we are
        // replacing ¬R/¬X/¬B and vvvv bits with zeros).
        //
        // In particular it means that vex-encoded instructions should be specified with 0xc4 even
        // if they are always emitted with 0xc4-to-0xc5 folding.
      } else if constexpr (kOpcodesArray[kLegacyPrefixesCount] == 0xc4 ||
                           kOpcodesArray[kLegacyPrefixesCount] == 0x8f) {
        return true;
      }
      return false;
    }();
    constexpr auto conditions_count = kCountArguments<IsCondition, ArgumentsTypes...>;
    constexpr auto operands_count = kCountArguments<IsMemoryOperand, ArgumentsTypes...>;
    constexpr auto labels_count = kCountArguments<IsLabelOperand, ArgumentsTypes...>;
    constexpr auto registers_count = kCountArguments<IsRegister, ArgumentsTypes...>;
    // We need to know if Reg field (in ModRM byte) is an opcode extension or if opcode extension
    // goes into the immediate field.
    constexpr auto reg_is_opcode_extension =
        (registers_count + operands_count > 0) &&
        (registers_count + operands_count + labels_count <
         2 + kVexOrXop * (std::size(kOpcodesArray) - kLegacyPrefixesCount - 4));
    static_assert((registers_count + operands_count + labels_count + conditions_count +
                   kCountArguments<IsImmediate, ArgumentsTypes...>) == sizeof...(ArgumentsTypes),
                  "Only registers (with specified size), Operands (with specified size), "
                  "Conditions, and Immediates are supported.");
    static_assert(operands_count <= 1, "Only one operand is allowed in instruction.");
    static_assert(labels_count <= 1, "Only one label is allowed in instruction.");
    // 0x0f is an opcode extension, if it's not there then we only have one byte opcode.
    const size_t kPrefixesAndOpcodeExtensionsCount = []() {
      if constexpr (kVexOrXop) {
        static_assert(conditions_count == 0,
                      "No conditionals are supported in vex/xop instructions.");
        static_assert((registers_count + operands_count + labels_count) <= 4,
                      "Up to four-arguments in vex/xop instructions are supported.");
        return kLegacyPrefixesCount + 3;
      } else {
        static_assert(conditions_count <= 1, "Only one condition is allowed in instruction.");
        static_assert((registers_count + operands_count + labels_count) <= 2,
                      "Only two-arguments legacy instructions are supported.");
        if constexpr (kOpcodesArray[kLegacyPrefixesCount] == 0x0f) {
          if constexpr (kOpcodesArray[kLegacyPrefixesCount + 1] == 0x38 ||
                        kOpcodesArray[kLegacyPrefixesCount + 1] == 0x3a) {
            return kLegacyPrefixesCount + 2;
          }
          return kLegacyPrefixesCount + 1;
        }
        return kLegacyPrefixesCount;
      }
    }();
    if constexpr (kVexOrXop) {
      static_cast<DerivedAssemblerType*>(this)
          ->template EmitVex<kOpcodesArray[kLegacyPrefixesCount],
                             kOpcodesArray[kLegacyPrefixesCount + 1],
                             kOpcodesArray[kLegacyPrefixesCount + 2],
                             reg_is_opcode_extension>(arguments...);
    } else {
      static_cast<DerivedAssemblerType*>(this)->EmitRex(arguments...);
      for (size_t extension_opcode_index = kLegacyPrefixesCount;
           extension_opcode_index < kPrefixesAndOpcodeExtensionsCount;
           ++extension_opcode_index) {
        Emit8(kOpcodesArray[extension_opcode_index]);
      }
    }
    // These are older 8086 instructions which encode register number in the opcode itself.
    if constexpr (registers_count == 1 && operands_count == 0 && labels_count == 0 &&
                  std::size(kOpcodesArray) == kPrefixesAndOpcodeExtensionsCount + 1) {
      static_cast<DerivedAssemblerType*>(this)->EmitRegisterInOpcode(
          kOpcodesArray[kPrefixesAndOpcodeExtensionsCount],
          ArgumentByType<0, IsRegister>(arguments...));
      EmitImmediates(arguments...);
    } else {
      // Emit "main" single-byte opcode.
      if constexpr (conditions_count == 1) {
        auto condition_code = static_cast<uint8_t>(ArgumentByType<0, IsCondition>(arguments...));
        CHECK_EQ(0, condition_code & 0xf0);
        Emit8(kOpcodesArray[kPrefixesAndOpcodeExtensionsCount] | condition_code);
      } else {
        Emit8(kOpcodesArray[kPrefixesAndOpcodeExtensionsCount]);
      }
      if constexpr (reg_is_opcode_extension) {
        if constexpr (operands_count == 1) {
          static_cast<DerivedAssemblerType*>(this)->EmitOperandOp(
              static_cast<int>(kOpcodesArray[kPrefixesAndOpcodeExtensionsCount + 1]),
              ArgumentByType<0, IsMemoryOperand>(arguments...).operand);
        } else if constexpr (labels_count == 1) {
          static_cast<DerivedAssemblerType*>(this)
              ->template EmitRipOp<ImmediatesSize<ArgumentsTypes...>()>(
                  static_cast<int>(kOpcodesArray[kPrefixesAndOpcodeExtensionsCount + 1]),
                  ArgumentByType<0, IsLabelOperand>(arguments...).label);
        } else {
          static_cast<DerivedAssemblerType*>(this)->EmitModRM(
              kOpcodesArray[kPrefixesAndOpcodeExtensionsCount + 1],
              ArgumentByType<0, IsRegister>(arguments...));
        }
      } else if constexpr (registers_count > 0) {
        if constexpr (operands_count == 1) {
          static_cast<DerivedAssemblerType*>(this)->EmitOperandOp(
              ArgumentByType<0, IsRegister>(arguments...),
              ArgumentByType<0, IsMemoryOperand>(arguments...).operand);
        } else if constexpr (labels_count == 1) {
          static_cast<DerivedAssemblerType*>(this)
              ->template EmitRipOp<ImmediatesSize<ArgumentsTypes...>()>(
                  ArgumentByType<0, IsRegister>(arguments...),
                  ArgumentByType<0, IsLabelOperand>(arguments...).label);
        } else {
          static_cast<DerivedAssemblerType*>(this)->EmitModRM(
              ArgumentByType<0, IsRegister>(arguments...),
              ArgumentByType<1, IsRegister>(arguments...));
        }
      }
      // If reg is an opcode extension then we already used that element.
      if constexpr (reg_is_opcode_extension) {
        static_assert(std::size(kOpcodesArray) == kPrefixesAndOpcodeExtensionsCount + 2);
      } else if constexpr (std::size(kOpcodesArray) > kPrefixesAndOpcodeExtensionsCount + 1) {
        // Final opcode byte(s) - they are in the place where immediate is expected.
        // Cmpsps/Cmppd and 3DNow! instructions are using it.
        static_assert(std::size(kOpcodesArray) == kPrefixesAndOpcodeExtensionsCount + 2);
        Emit8(kOpcodesArray[kPrefixesAndOpcodeExtensionsCount + 1]);
      }
      if constexpr (registers_count + operands_count + labels_count == 4) {
        if constexpr (kCountArguments<IsImmediate, ArgumentsTypes...> == 1) {
          Emit8((ArgumentByType<registers_count - 1, IsRegister>(arguments...).num_ << 4) |
                ArgumentByType<0, IsImmediate>(arguments...));
        } else {
          static_assert(kCountArguments<IsImmediate, ArgumentsTypes...> == 0);
          Emit8(ArgumentByType<registers_count - 1, IsRegister>(arguments...).num_ << 4);
        }
      } else {
        EmitImmediates(arguments...);
      }
    }
  }

  // Normally instruction arguments come in the following order: vex, rm, reg, imm.
  // But certain instructions can have swapped arguments in a different order.
  // In addition to that we have special case where two arguments may need to be swapped
  // to reduce encoding size.

  template <uint8_t... kOpcodes,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename... ArgumentsTypes>
  void EmitRegToRmInstruction(ArgumentsType0&& argument0,
                              ArgumentsType1&& argument1,
                              ArgumentsTypes&&... arguments) {
    return EmitInstruction<kOpcodes...>(std::forward<ArgumentsType1>(argument1),
                                        std::forward<ArgumentsType0>(argument0),
                                        std::forward<ArgumentsTypes>(arguments)...);
  }

  template <uint8_t... kOpcodes,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename... ArgumentsTypes>
  void EmitRmToVexInstruction(ArgumentsType0&& argument0,
                              ArgumentsType1&& argument1,
                              ArgumentsTypes&&... arguments) {
    return EmitInstruction<kOpcodes...>(std::forward<ArgumentsType1>(argument1),
                                        std::forward<ArgumentsType0>(argument0),
                                        std::forward<ArgumentsTypes>(arguments)...);
  }

  // If vex operand is one of first 8 registers and rm operand is not then swapping these two
  // operands produces more compact encoding.
  // This only works with commutative instructions from first opcode map.
  template <uint8_t... kOpcodes,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename ArgumentsType2,
            typename... ArgumentsTypes>
  void EmitOptimizableUsingCommutationInstruction(ArgumentsType0&& argument0,
                                                  ArgumentsType1&& argument1,
                                                  ArgumentsType2&& argument2,
                                                  ArgumentsTypes&&... arguments) {
    if constexpr (std::is_same_v<ArgumentsType2, ArgumentsType1>) {
      if (DerivedAssemblerType::IsSwapProfitable(std::forward<ArgumentsType2>(argument2),
                                                 std::forward<ArgumentsType1>(argument1))) {
        return EmitInstruction<kOpcodes...>(std::forward<ArgumentsType0>(argument0),
                                            std::forward<ArgumentsType1>(argument1),
                                            std::forward<ArgumentsType2>(argument2),
                                            std::forward<ArgumentsTypes>(arguments)...);
      }
    }
    return EmitInstruction<kOpcodes...>(std::forward<ArgumentsType0>(argument0),
                                        std::forward<ArgumentsType2>(argument2),
                                        std::forward<ArgumentsType1>(argument1),
                                        std::forward<ArgumentsTypes>(arguments)...);
  }

  template <uint8_t... kOpcodes,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename ArgumentsType2,
            typename ArgumentsType3,
            typename... ArgumentsTypes>
  void EmitVexImmRmToRegInstruction(ArgumentsType0&& argument0,
                                    ArgumentsType1&& argument1,
                                    ArgumentsType2&& argument2,
                                    ArgumentsType3&& argument3,
                                    ArgumentsTypes&&... arguments) {
    return EmitInstruction<kOpcodes...>(std::forward<ArgumentsType0>(argument0),
                                        std::forward<ArgumentsType3>(argument3),
                                        std::forward<ArgumentsType1>(argument1),
                                        std::forward<ArgumentsType2>(argument2),
                                        std::forward<ArgumentsTypes>(arguments)...);
  }

  template <uint8_t... kOpcodes,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename ArgumentsType2,
            typename... ArgumentsTypes>
  void EmitVexRegToRmInstruction(ArgumentsType0&& argument0,
                                 ArgumentsType1&& argument1,
                                 ArgumentsType2&& argument2,
                                 ArgumentsTypes&&... arguments) {
    return EmitInstruction<kOpcodes...>(std::forward<ArgumentsType1>(argument2),
                                        std::forward<ArgumentsType0>(argument0),
                                        std::forward<ArgumentsType2>(argument1),
                                        std::forward<ArgumentsTypes>(arguments)...);
  }

  template <uint8_t... kOpcodes,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename ArgumentsType2,
            typename ArgumentsType3,
            typename... ArgumentsTypes>
  void EmitVexRmImmToRegInstruction(ArgumentsType0&& argument0,
                                    ArgumentsType1&& argument1,
                                    ArgumentsType2&& argument2,
                                    ArgumentsType3&& argument3,
                                    ArgumentsTypes&&... arguments) {
    return EmitInstruction<kOpcodes...>(std::forward<ArgumentsType0>(argument0),
                                        std::forward<ArgumentsType2>(argument2),
                                        std::forward<ArgumentsType1>(argument1),
                                        std::forward<ArgumentsType3>(argument3),
                                        std::forward<ArgumentsTypes>(arguments)...);
  }

  template <uint8_t... kOpcodes,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename ArgumentsType2,
            typename... ArgumentsTypes>
  void EmitVexRmToRegInstruction(ArgumentsType0&& argument0,
                                 ArgumentsType1&& argument1,
                                 ArgumentsType2&& argument2,
                                 ArgumentsTypes&&... arguments) {
    return EmitInstruction<kOpcodes...>(std::forward<ArgumentsType0>(argument0),
                                        std::forward<ArgumentsType2>(argument2),
                                        std::forward<ArgumentsType1>(argument1),
                                        std::forward<ArgumentsTypes>(arguments)...);
  }

  constexpr void ResolveJumps();

 private:
  Assembler() = delete;
  Assembler(const Assembler&) = delete;
  Assembler(Assembler&&) = delete;
  void operator=(const Assembler&) = delete;
  void operator=(Assembler&&) = delete;
};

template <typename DerivedAssemblerType>
constexpr inline void Assembler<DerivedAssemblerType>::Pmov(XMMRegister dest, XMMRegister src) {
  // SSE does not have operations for register-to-register integer move and
  // Intel explicitly recommends to use pshufd instead on Pentium4:
  //   See https://software.intel.com/en-us/articles/
  //               fast-simd-integer-move-for-the-intel-pentiumr-4-processor
  // These recommendations are CPU-dependent, though, thus we will need to
  // investigate this question further before we could decide when to use
  // movaps (or movapd) and when to use pshufd.
  //
  // TODO(khim): investigate performance problems related to integer MOVs
  Movaps(dest, src);
}

template <typename DerivedAssemblerType>
constexpr inline void Assembler<DerivedAssemblerType>::Call(const Label& label) {
  if (label.IsBound()) {
    int32_t offset = label.position() - pc();
    Call(offset);
  } else {
    Emit8(0xe8);
    Emit32(0xffff'fffc);
    jumps_.push_back(Jump{&label, pc() - 4, false});
  }
}

template <typename DerivedAssemblerType>
constexpr inline void Assembler<DerivedAssemblerType>::Jcc(Condition cc, const Label& label) {
  if (cc == Condition::kAlways) {
    Jmp(label);
    return;
  } else if (cc == Condition::kNever) {
    return;
  }
  CHECK_EQ(0, static_cast<uint8_t>(cc) & 0xf0);
  // TODO(eaeltsin): may be remove IsBound case?
  // Then jcc by label will be of fixed size (5 bytes)
  if (label.IsBound()) {
    int32_t offset = label.position() - pc();
    JccRel(cc, offset);
  } else {
    Emit16(0x800f | (static_cast<uint8_t>(cc) << 8));
    Emit32(0xffff'fffc);
    jumps_.push_back(Jump{&label, pc() - 4, false});
  }
}

template <typename DerivedAssemblerType>
constexpr inline void Assembler<DerivedAssemblerType>::Jmp(const Label& label) {
  // TODO(eaeltsin): may be remove IsBound case?
  // Then jmp by label will be of fixed size (5 bytes)
  if (label.IsBound()) {
    int32_t offset = label.position() - pc();
    JmpRel(offset);
  } else {
    Emit8(0xe9);
    Emit32(0xffff'fffc);
    jumps_.push_back(Jump{&label, pc() - 4, false});
  }
}

template <typename DerivedAssemblerType>
constexpr inline void Assembler<DerivedAssemblerType>::ResolveJumps() {
  for (const auto& jump : jumps_) {
    const Label* label = jump.label;
    uint32_t pc = jump.pc;
    CHECK(label->IsBound());
    if (jump.is_recovery) {
      // Add pc -> label correspondence to recovery map.
      AddRelocation(0, RelocationType::RelocRecoveryPoint, pc, label->position());
    } else {
      int32_t offset = label->position() - pc;
      *AddrAs<int32_t>(pc) += offset;
    }
  }
}

// Code size optimized instructions: they have different variants depending on registers used.

template <typename DerivedAssemblerType>
constexpr inline void Assembler<DerivedAssemblerType>::Xchgl(Register dest, Register src) {
  if (DerivedAssemblerType::IsAccumulator(src) || DerivedAssemblerType::IsAccumulator(dest)) {
    Register other = DerivedAssemblerType::IsAccumulator(src) ? dest : src;
    EmitInstruction<0x90>(SizeAgnosticRegister(other));
  } else {
    // Clang 8 (after r330298) puts dest before src.  We are comparing output
    // to clang in exhaustive test thus we want to match clang behavior exactly.
    EmitInstruction<0x87>(SizeAgnosticRegister(dest), SizeAgnosticRegister(src));
  }
}

}  // namespace x86_32_or_x86_64

}  // namespace berberis

#endif  // BERBERIS_ASSEMBLER_X86_32_OR_X86_64_H_
