// region digitalis
/*
 * Copyright (C) 2026 utzcoz
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

#ifndef BERBERIS_DECODER_ARM64_DECODER_H_
#define BERBERIS_DECODER_ARM64_DECODER_H_

#include <climits>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "berberis/base/checks.h"

namespace berberis {

// Decode() method takes a sequence of bytes and decodes it into the instruction opcode and fields.
// The InsnConsumer's method corresponding to the decoded opcode is called with the decoded fields
// as an argument. Returned is the instruction size (always 4 for ARM64).
template <class InsnConsumer>
class Decoder {
 public:
  explicit Decoder(InsnConsumer* insn_consumer) : insn_consumer_(insn_consumer) {}

  // https://eel.is/c++draft/enum#dcl.enum-8
  // For an enumeration whose underlying type is fixed, the values of the enumeration are the values
  // of the underlying type.

  // To ensure that there are no surprises we specify that type in all enums below.

  //
  // Condition codes for B.cond etc.
  //
  enum class Condition : uint8_t {
    kEq = 0b0000,   // Equal (Z==1)
    kNe = 0b0001,   // Not equal (Z==0)
    kCs = 0b0010,   // Carry set / unsigned higher or same (C==1)
    kCc = 0b0011,   // Carry clear / unsigned lower (C==0)
    kMi = 0b0100,   // Minus / negative (N==1)
    kPl = 0b0101,   // Plus / positive or zero (N==0)
    kVs = 0b0110,   // Overflow (V==1)
    kVc = 0b0111,   // No overflow (V==0)
    kHi = 0b1000,   // Unsigned higher (C==1 && Z==0)
    kLs = 0b1001,   // Unsigned lower or same (C==0 || Z==1)
    kGe = 0b1010,   // Signed greater than or equal (N==V)
    kLt = 0b1011,   // Signed less than (N!=V)
    kGt = 0b1100,   // Signed greater than (Z==0 && N==V)
    kLe = 0b1101,   // Signed less than or equal (Z==1 || N!=V)
    kAl = 0b1110,   // Always
    kNv = 0b1111,   // Always (alternative encoding)
  };

  //
  // Logical immediate opcodes.
  //
  enum class LogicalImmOpcode : uint8_t {
    kAnd = 0b00,
    kOrr = 0b01,
    kEor = 0b10,
    kAnds = 0b11,
  };

  //
  // Logical shifted register opcodes.
  //
  enum class LogicalShiftedRegOpcode : uint8_t {
    kAnd = 0b00,
    kOrr = 0b01,
    kEor = 0b10,
    kAnds = 0b11,
  };

  //
  // Shift types for data processing register.
  //
  enum class ShiftType : uint8_t {
    kLsl = 0b00,
    kLsr = 0b01,
    kAsr = 0b10,
    kRor = 0b11,
  };

  //
  // Move wide opcodes.
  //
  enum class MoveWideOpcode : uint8_t {
    kMovn = 0b00,
    kMovz = 0b10,
    kMovk = 0b11,
  };

  //
  // Add/Sub shifted register opcodes.
  //
  enum class AddSubShiftedRegOpcode : uint8_t {
    kAdd = 0b0,
    kSub = 0b1,
  };

  //
  // Bitfield opcodes.
  //
  enum class BitfieldOpcode : uint8_t {
    kSbfm = 0b00,
    kBfm = 0b01,
    kUbfm = 0b10,
  };

  //
  // Data Processing (2-source) opcodes.
  //
  enum class DataProc2SrcOpcode : uint8_t {
    kUdiv = 0b000010,
    kSdiv = 0b000011,
    kLslv = 0b001000,
    kLsrv = 0b001001,
    kAsrv = 0b001010,
    kRorv = 0b001011,
    // region digitalis - CRC32 opcodes
    kCrc32b = 0b010000,
    kCrc32h = 0b010001,
    kCrc32w = 0b010010,
    kCrc32x = 0b010011,
    kCrc32cb = 0b010100,
    kCrc32ch = 0b010101,
    kCrc32cw = 0b010110,
    kCrc32cx = 0b010111,
    // endregion
  };

  //
  // Data Processing (3-source) opcodes.
  //
  enum class DataProc3SrcOpcode : uint8_t {
    kMadd = 0b000,
    kMsub = 0b001,
    kSmaddl = 0b010,
    kSmsubl = 0b011,
    kSmulh = 0b100,
    kUmaddl = 0b101,
    kUmsubl = 0b110,
    kUmulh = 0b111,
  };

  //
  // Conditional select opcodes.
  //
  enum class ConditionalSelectOpcode : uint8_t {
    kCsel = 0b00,
    kCsinc = 0b01,
    kCsinv = 0b10,
    kCsneg = 0b11,
  };

  //
  // Load/Store size encoding.
  //
  enum class LoadStoreSize : uint8_t {
    k8bit = 0b00,
    k16bit = 0b01,
    k32bit = 0b10,
    k64bit = 0b11,
  };

  //
  // System register encodings (for MRS/MSR).
  //
  enum class SystemReg : uint16_t {
    kTpidrEl0 = 0xDE82,   // TPIDR_EL0: op0=3, op1=3, CRn=13, CRm=0, op2=2
    kNzcv = 0xDA10,       // NZCV: op0=3, op1=3, CRn=4, CRm=2, op2=0
    kFpcr = 0xDA20,       // FPCR: op0=3, op1=3, CRn=4, CRm=4, op2=0
    kFpsr = 0xDA21,       // FPSR: op0=3, op1=3, CRn=4, CRm=4, op2=1
    // region digitalis
    kCtrEl0 = 0xD807,     // CTR_EL0: op0=3, op1=3, CRn=0, CRm=0, op2=7
    kDczidEl0 = 0xD80F,   // DCZID_EL0: op0=3, op1=3, CRn=0, CRm=0, op2=7 (alt)
    // MIDR_EL1: op0=3, op1=0, CRn=0, CRm=0, op2=0
    //   sysreg = (3<<14)|(0<<11)|(0<<7)|(0<<3)|0 = 0xC000
    // Read by code that decides whether to use vectorised fast paths;
    // returning a real ARM CPU id (Cortex-A53 here) keeps Bionic and
    // third-party compression libs (Facebook superpack, etc.) on the
    // expected fast paths instead of the SIGILL-handler-driven probe
    // fallbacks that can spin in detection loops.
    kMidrEl1 = 0xC000,
    // endregion
  };

  //
  // Args structs for each instruction group.
  //

  struct AddSubImmArgs {
    uint8_t dst;       // Rd (0-30, or 31 for SP/ZR depending on context)
    uint8_t src;       // Rn (0-30, or 31 for SP)
    uint32_t imm;      // 12-bit immediate, optionally shifted left by 12
    bool is_64bit;     // sf bit: true for X registers, false for W
    bool is_sub;       // true for SUB, false for ADD
    bool set_flags;    // S bit: true to set NZCV flags (ADDS/SUBS/CMP/CMN)
  };

  struct LogicalImmArgs {
    LogicalImmOpcode opcode;
    uint8_t dst;
    uint8_t src;
    uint64_t imm;      // Decoded bitmask immediate
    bool is_64bit;
  };

  struct MoveWideArgs {
    MoveWideOpcode opcode;
    uint8_t dst;
    uint16_t imm;      // 16-bit immediate
    uint8_t shift;     // Amount to shift: 0, 16, 32, or 48
    bool is_64bit;
  };

  struct PcRelAddrArgs {
    uint8_t dst;
    int64_t offset;    // PC-relative offset
    bool is_adrp;      // true for ADRP (page-aligned), false for ADR
  };

  struct BitfieldArgs {
    BitfieldOpcode opcode;
    uint8_t dst;
    uint8_t src;
    uint8_t immr;
    uint8_t imms;
    bool is_64bit;
  };

  struct BranchImmArgs {
    int32_t offset;    // PC-relative offset (signed, in bytes)
    bool is_link;      // true for BL, false for B
  };

  struct BranchCondArgs {
    Condition cond;
    int32_t offset;    // PC-relative offset (signed, in bytes)
  };

  struct BranchRegArgs {
    uint8_t src;       // Register containing target address
    uint8_t link_reg;  // 0 for BR/RET, 30 for BLR
    bool is_link;      // true for BLR
    bool is_ret;       // true for RET
  };

  struct CompareAndBranchArgs {
    uint8_t src;       // Register to test
    int32_t offset;    // PC-relative offset (signed, in bytes)
    bool is_64bit;
    bool is_nonzero;   // true for CBNZ, false for CBZ
  };

  struct TestAndBranchArgs {
    uint8_t src;       // Register to test
    uint8_t bit;       // Bit number to test (0-63)
    int32_t offset;    // PC-relative offset (signed, in bytes)
    bool is_nonzero;   // true for TBNZ, false for TBZ
  };

  struct LoadStoreImmArgs {
    uint8_t rt;        // Destination/source register
    uint8_t rn;        // Base address register (31 = SP)
    int32_t offset;    // Byte offset (pre-scaled)
    LoadStoreSize size;
    bool is_store;
    bool is_signed;    // For loads: sign-extend the value
    bool is_64bit_target;  // For signed loads: extend to 64-bit
  };

  struct LoadStorePairArgs {
    uint8_t rt1;       // First register
    uint8_t rt2;       // Second register
    uint8_t rn;        // Base address register (31 = SP)
    int32_t offset;    // Byte offset (pre-scaled)
    LoadStoreSize size; // k32bit or k64bit
    bool is_store;
    bool is_preindex;
    bool is_postindex;
  };

  struct LoadStoreRegArgs {
    uint8_t rt;
    uint8_t rn;
    uint8_t rm;
    // region digitalis - raw 3-bit ARMv8 option field:
    // 000=UXTB, 001=UXTH, 010=UXTW, 011=LSL (UXTX), 100=SXTB,
    // 101=SXTH, 110=SXTW, 111=SXTX. Only 010/011/110/111 are valid for
    // a load/store address; the encoded option is preserved so the JIT
    // and interpreter can apply the correct 32->64 extension before
    // shifting/adding to the base. Throwing this away (treating SXTW or
    // UXTW as LSL) silently uses bits[63:32] of the X register that
    // backs the W offset, corrupting addresses for sign-extended or
    // unclean upper-half offsets. See libsuperpack-jni.so hot loops at
    // 0x34488 / 0x344a4 where ldrb [..., w, uxtw] is the entire decode
    // inner kernel.
    uint8_t extend_type;
    // endregion
    uint8_t shift_amount;
    LoadStoreSize size;
    bool is_store;
    bool is_signed;
    bool is_64bit_target;
  };

  struct SvcArgs {
    uint16_t imm;
  };

  struct MrsArgs {
    uint8_t dst;
    SystemReg sysreg;
  };

  struct MsrArgs {
    uint8_t src;
    SystemReg sysreg;
  };

  struct LogicalShiftedRegArgs {
    LogicalShiftedRegOpcode opcode;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    ShiftType shift_type;
    uint8_t shift_amount;
    bool is_64bit;
    bool invert;       // N bit: invert src2 (BIC, ORN, EON, BICS)
  };

  struct AddSubShiftedRegArgs {
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    ShiftType shift_type;
    uint8_t shift_amount;
    bool is_64bit;
    bool is_sub;
    bool set_flags;
  };

  struct AddSubExtendedRegArgs {
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    uint8_t extend_type;  // 3-bit extend type
    uint8_t shift_amount; // 0-4
    bool is_64bit;
    bool is_sub;
    bool set_flags;
  };

  struct ConditionalSelectArgs {
    ConditionalSelectOpcode opcode;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    Condition cond;
    bool is_64bit;
  };

  struct DataProc2SrcArgs {
    DataProc2SrcOpcode opcode;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    bool is_64bit;
  };

  struct DataProc3SrcArgs {
    DataProc3SrcOpcode opcode;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    uint8_t src3;      // Ra: addend register
    bool is_64bit;
  };

  struct SystemArgs {
    uint32_t insn;     // Full instruction for unrecognized system instructions.
  };

  // region digitalis
  //
  // SIMD/FP load/store size (extends LoadStoreSize with 128-bit).
  //
  enum class SimdLoadStoreSize : uint8_t {
    k8bit = 0,
    k16bit = 1,
    k32bit = 2,
    k64bit = 3,
    k128bit = 4,
  };

  struct SimdModifiedImmArgs {
    uint8_t rd;
    uint8_t cmode;
    uint8_t op;
    uint8_t abc;
    uint8_t defgh;
    bool q;
  };

  struct SimdLoadStoreImmArgs {
    uint8_t rt;
    uint8_t rn;
    int64_t offset;
    SimdLoadStoreSize size;
    bool is_store;
  };

  struct SimdLoadStorePairArgs {
    uint8_t rt1;
    uint8_t rt2;
    uint8_t rn;
    int64_t offset;
    SimdLoadStoreSize size;
    bool is_store;
    bool is_preindex;
    bool is_postindex;
  };

  struct SimdLoadStoreRegArgs {
    uint8_t rt;
    uint8_t rn;
    uint8_t rm;
    // region digitalis - raw 3-bit ARMv8 option field for the offset
    // register, same encoding as LoadStoreRegArgs::extend_type. See the
    // comment there.
    uint8_t extend_type;
    // endregion
    uint8_t shift_amount;
    SimdLoadStoreSize size;
    bool is_store;
  };

  struct FpIntConvArgs {
    uint8_t rd;
    uint8_t rn;
    uint8_t opcode;   // sf:ftype:rmode:opcode combined
    bool sf;          // true = 64-bit int
    uint8_t ftype;    // 00=S, 01=D, 10=H, 11=Q(128)
    uint8_t rmode;
    uint8_t op;
  };

  // region digitalis
  enum class FpFixedPointOp : uint8_t {
    kScvtf,   // Signed fixed-point to FP
    kUcvtf,   // Unsigned fixed-point to FP
    kFcvtzs,  // FP to signed fixed-point
    kFcvtzu,  // FP to unsigned fixed-point
  };

  struct FpFixedPointArgs {
    uint8_t rd;
    uint8_t rn;
    FpFixedPointOp op;
    bool sf;          // true = 64-bit integer
    uint8_t ftype;    // 00=S, 01=D
    uint8_t fbits;    // Number of fractional bits (1..32 for sf=0, 1..64 for sf=1)
  };
  // endregion

  struct ConditionalCompareArgs {
    uint8_t rn;        // First operand register
    uint8_t rm_or_imm; // Second operand (register or 5-bit immediate)
    uint8_t nzcv;      // Flags to set if condition is false
    Condition cond;    // Condition to evaluate
    bool is_64bit;
    bool is_imm;       // true = immediate, false = register
    bool is_neg;       // true = CCMN, false = CCMP
  };

  enum class AtomicOp : uint8_t {
    kLdxr,     // Load exclusive
    kStxr,     // Store exclusive (result in Rs)
    kLdar,     // Load acquire
    kStlr,     // Store release
    kCas,      // Compare and swap
    kSwp,      // Swap
    kLdadd,    // Atomic add
    kLdclr,    // Atomic bit clear
    kLdset,    // Atomic bit set
    kLdeor,    // Atomic exclusive or
  };

  struct LoadStoreExclusiveArgs {
    AtomicOp op;
    uint8_t rt;        // Data register
    uint8_t rn;        // Base address register (31=SP)
    uint8_t rs;        // Status(STXR) or swap/compare(CAS/SWP) register
    uint8_t size;      // 0=8bit, 1=16bit, 2=32bit, 3=64bit
    bool acquire;      // Acquire semantics
    bool release;      // Release semantics
  };

  enum class AdvSimdCopyOpcode : uint8_t {
    kDupElement,   // DUP (element): duplicate Vn element to all lanes of Vd
    kDupGeneral,   // DUP (general): duplicate Xn/Wn to all lanes of Vd
    kInsGeneral,   // INS (general): insert Xn/Wn into Vd element
    kSmov,         // SMOV: signed move from Vn element to Xd/Wd
    kUmov,         // UMOV: unsigned move from Vn element to Xd/Wd
    kInsElement,   // INS (element): copy Vn element to Vd element
    // region digitalis - scalar SIMD copy (DUP scalar / MOV Vd, Vn[index])
    kDupScalar,    // DUP (scalar) / MOV scalar: copy one Vn[index] element into bottom of Vd, zero upper
    // endregion
  };

  struct AdvSimdCopyArgs {
    AdvSimdCopyOpcode opcode;
    uint8_t rd;        // Destination register (SIMD for DUP/INS, GP for SMOV/UMOV)
    uint8_t rn;        // Source register (SIMD for DUP/SMOV/UMOV, GP for DUP-general/INS-general)
    uint8_t imm5;      // Encodes element size and index
    uint8_t imm4;      // For INS (element): source index
    bool q;            // Q bit: 0=64-bit vector, 1=128-bit vector
  };

  // region digitalis
  //
  // AdvSIMD three same opcodes.
  // Encoding: 0 Q U 01110 size 1 Rm opcode 1 Rn Rd
  //
  enum class AdvSimdThreeSameOpcode : uint8_t {
    kAnd,       // AND (vector): U=0, size=00, opcode=00011
    kBic,       // BIC (vector): U=0, size=01, opcode=00011
    kOrr,       // ORR (vector): U=0, size=10, opcode=00011
    kOrn,       // ORN (vector): U=0, size=11, opcode=00011
    kEor,       // EOR (vector): U=1, size=00, opcode=00011
    kBsl,       // BSL (vector): U=1, size=01, opcode=00011
    kBit,       // BIT (vector): U=1, size=10, opcode=00011
    kBif,       // BIF (vector): U=1, size=11, opcode=00011
    kAdd,       // ADD (vector): U=0, opcode=10000
    kSub,       // SUB (vector): U=1, opcode=10000
    kCmeq,      // CMEQ (vector): U=1, opcode=10001
    kCmtst,     // CMTST (vector): U=0, opcode=10001
    kSmax,      // SMAX (vector): U=0, opcode=01100
    kSmin,      // SMIN (vector): U=0, opcode=01101
    kUmax,      // UMAX (vector): U=1, opcode=01100
    kUmin,      // UMIN (vector): U=1, opcode=01101
    // region digitalis
    kCmgt,      // CMGT (vector, register): U=0, opcode=00110
    kCmhi,      // CMHI (vector, register): U=1, opcode=00110 (unsigned >)
    kCmge,      // CMGE (vector, register): U=0, opcode=00111
    kCmhs,      // CMHS (vector, register): U=1, opcode=00111 (unsigned >=)
    // endregion
    kShadd,     // SHADD (vector): U=0, opcode=00000
    kUhadd,     // UHADD (vector): U=1, opcode=00000
    kSqadd,     // SQADD (vector): U=0, opcode=00001
    kUqadd,     // UQADD (vector): U=1, opcode=00001
    kSrhadd,    // SRHADD (vector): U=0, opcode=00010
    kUrhadd,    // URHADD (vector): U=1, opcode=00010
    kShsub,     // SHSUB (vector): U=0, opcode=00100
    kUhsub,     // UHSUB (vector): U=1, opcode=00100
    kSqsub,     // SQSUB (vector): U=0, opcode=00101
    kUqsub,     // UQSUB (vector): U=1, opcode=00101
    kSshl,      // SSHL (vector): U=0, opcode=01000
    kUshl,      // USHL (vector): U=1, opcode=01000
    kSqshl,     // SQSHL (vector): U=0, opcode=01001
    kUqshl,     // UQSHL (vector): U=1, opcode=01001
    kSrshl,     // SRSHL (vector): U=0, opcode=01010
    kUrshl,     // URSHL (vector): U=1, opcode=01010
    kSqrshl,    // SQRSHL (vector): U=0, opcode=01011
    kUqrshl,    // UQRSHL (vector): U=1, opcode=01011
    kAddp,      // ADDP (vector): U=0, opcode=10111
    kMul,       // MUL (vector): U=0, opcode=10011
    kMla,       // MLA (vector): U=0, opcode=10010
    kMls,       // MLS (vector): U=1, opcode=10010
    // region digitalis
    kSmaxp,     // SMAXP (vector): U=0, opcode=10100
    kUmaxp,     // UMAXP (vector): U=1, opcode=10100
    kSminp,     // SMINP (vector): U=0, opcode=10101
    kUminp,     // UMINP (vector): U=1, opcode=10101
    // FP three-same (vector). The encoding actually uses bit 23 as
    // op_high (0 = FADD/FMUL/FMLA half; 1 = FSUB/FMLS half) plus bit
    // 22 as sz (0 = single 32-bit; 1 = double 64-bit). The args.size
    // field is repurposed to carry sz alone (so 0b00 = single,
    // 0b01 = double); the decoder distinguishes FSUB / FMLS from
    // FADD / FMLA by selecting a different enum value rather than
    // leaking op_high through args.
    kFaddV,     // FADD  (vector): op_high=0, opcode=11010, U=0
    kFsubV,     // FSUB  (vector): op_high=1, opcode=11010, U=0
    kFmulV,     // FMUL  (vector): op_high=0, opcode=11011, U=1
    kFmlaV,     // FMLA  (vector): op_high=0, opcode=11001, U=0
    kFmlsV,     // FMLS  (vector): op_high=1, opcode=11001, U=0
    kFmaxnmV,   // FMAXNM (vector): op_high=0, opcode=11000, U=0
    kFminnmV,   // FMINNM (vector): op_high=1, opcode=11000, U=0
    kFmaxV,     // FMAX   (vector): op_high=0, opcode=11110, U=0
    kFminV,     // FMIN   (vector): op_high=1, opcode=11110, U=0
    kFdivV,     // FDIV   (vector): op_high=0, opcode=11111, U=1
    kFcmeqV,    // FCMEQ  (vector): op_high=0, opcode=11100, U=0
    kFcmgeV,    // FCMGE  (vector): op_high=0, opcode=11100, U=1
    kFcmgtV,    // FCMGT  (vector): op_high=1, opcode=11100, U=1
    kFacgeV,    // FACGE  (vector): op_high=0, opcode=11101, U=1
    kFacgtV,    // FACGT  (vector): op_high=1, opcode=11101, U=1
    // endregion
  };

  struct AdvSimdThreeSameArgs {
    AdvSimdThreeSameOpcode opcode;
    uint8_t rd;        // Destination SIMD register
    uint8_t rn;        // First source SIMD register
    uint8_t rm;        // Second source SIMD register
    uint8_t size;      // Element size: 00=8b, 01=16b, 10=32b, 11=64b
    bool q;            // Q bit: 0=64-bit vector (D regs), 1=128-bit vector (Q regs)
  };
  // endregion

  // region digitalis
  //
  // AdvSIMD three different (widening/narrowing) opcodes.
  // Encoding: 0 Q U 01110 size 1 Rm opcode 00 Rn Rd
  //   Widening ops: narrow inputs (size) -> wide output (2*size)
  //
  enum class AdvSimdThreeDiffOpcode : uint8_t {
    kSaddl,    // SADDL/SADDL2: U=0, opcode=0000
    kUaddl,    // UADDL/UADDL2: U=1, opcode=0000
    kSsubl,    // SSUBL/SSUBL2: U=0, opcode=0010
    kUsubl,    // USUBL/USUBL2: U=1, opcode=0010
    kSmlal,    // SMLAL/SMLAL2: U=0, opcode=1000
    kUmlal,    // UMLAL/UMLAL2: U=1, opcode=1000
    kSmlsl,    // SMLSL/SMLSL2: U=0, opcode=1010
    kUmlsl,    // UMLSL/UMLSL2: U=1, opcode=1010
    kSmull,    // SMULL/SMULL2: U=0, opcode=1100
    kUmull,    // UMULL/UMULL2: U=1, opcode=1100
    kSabdl,    // SABDL/SABDL2: U=0, opcode=0111
    kUabdl,    // UABDL/UABDL2: U=1, opcode=0111
    kSabal,    // SABAL/SABAL2: U=0, opcode=0101
    kUabal,    // UABAL/UABAL2: U=1, opcode=0101
    // region digitalis - wide add/sub: Vd(wide) = Vn(wide) op extend(Vm(narrow))
    kSaddw,    // SADDW/SADDW2: U=0, opcode=0001
    kUaddw,    // UADDW/UADDW2: U=1, opcode=0001
    kSsubw,    // SSUBW/SSUBW2: U=0, opcode=0011
    kUsubw,    // USUBW/USUBW2: U=1, opcode=0011
    // endregion
  };

  struct AdvSimdThreeDiffArgs {
    AdvSimdThreeDiffOpcode opcode;
    uint8_t rd;        // Destination SIMD register (wide)
    uint8_t rn;        // First source SIMD register (narrow)
    uint8_t rm;        // Second source SIMD register (narrow)
    uint8_t size;      // Input element size: 00=8b, 01=16b, 10=32b
    bool q;            // Q=0: lower half (base), Q=1: upper half (2 variant)
  };

  //
  // AdvSIMD load/store single structure.
  // Covers: LD1/ST1 to one lane, LD1R-LD4R (replicate to all lanes).
  //
  enum class AdvSimdSingleStructOp : uint8_t {
    kLd1r,   // Load single element and replicate to all lanes (1 register)
    kLd2r,   // Load single element and replicate (2 registers)
    kLd3r,   // Load single element and replicate (3 registers)
    kLd4r,   // Load single element and replicate (4 registers)
    kLd1,    // Load single element to one lane (1 register)
    kLd2,    // Load single element to one lane (2 registers)
    kLd3,    // Load single element to one lane (3 registers)
    kLd4,    // Load single element to one lane (4 registers)
    kSt1,    // Store single element from one lane (1 register)
    kSt2,    // Store single element from one lane (2 registers)
    kSt3,    // Store single element from one lane (3 registers)
    kSt4,    // Store single element from one lane (4 registers)
  };

  struct AdvSimdSingleStructArgs {
    AdvSimdSingleStructOp op;
    uint8_t rt;        // First SIMD register
    uint8_t rn;        // Base address register
    uint8_t rm;        // Post-index register (31 = immediate, 0 = no post-index)
    uint8_t size;      // Element size: 00=B, 01=H, 10=S, 11=D
    uint8_t index;     // Lane index (for non-replicate ops)
    uint8_t num_regs;  // Number of registers (1-4)
    bool q;            // Vector arrangement qualifier
    bool postindex;    // Has post-index
    bool is_replicate; // True for LD1R-LD4R
  };
  // endregion

  // region digitalis
  //
  // FP data-processing (1 source) opcodes.
  //
  enum class FpDataProc1Opcode : uint8_t {
    kFmov = 0b000000,
    kFabs = 0b000001,
    kFneg = 0b000010,
    kFsqrt = 0b000011,
    kFcvtToOther1 = 0b000100,  // FCVT to the other single/double
    kFcvtToOther2 = 0b000101,  // FCVT to half or from half
    kFrintn = 0b001000,
    kFrintp = 0b001001,
    kFrintm = 0b001010,
    kFrintz = 0b001011,
    kFrinta = 0b001100,
    kFrintx = 0b001110,
    kFrinti = 0b001111,
  };

  struct FpDataProc1Args {
    uint8_t rd;
    uint8_t rn;
    uint8_t ftype;    // 00=S, 01=D, 11=H
    uint8_t opcode;   // raw 6-bit opcode
  };

  //
  // FP data-processing (2 source) opcodes.
  //
  enum class FpDataProc2Opcode : uint8_t {
    kFmul = 0b0000,
    kFdiv = 0b0001,
    kFadd = 0b0010,
    kFsub = 0b0011,
    kFmax = 0b0100,
    kFmin = 0b0101,
    kFmaxnm = 0b0110,
    kFminnm = 0b0111,
    kFnmul = 0b1000,
  };

  struct FpDataProc2Args {
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t ftype;    // 00=S, 01=D
    uint8_t opcode;   // raw 4-bit opcode
  };

  //
  // FP compare args.
  //
  struct FpCompareArgs {
    uint8_t rn;
    uint8_t rm;
    uint8_t ftype;    // 00=S, 01=D
    bool with_zero;   // true = compare with 0.0
    bool signal_nans; // true = FCMPE (signal all NaNs)
  };

  //
  // AdvSIMD two-reg misc opcodes.
  //
  enum class AdvSimdTwoRegMiscOpcode : uint8_t {
    kRev64,
    kRev32,
    kRev16,
    kSaddlp,
    kUaddlp,
    kCls,
    kClz,
    kCnt,
    kNot,     // also RBIT for different U
    kSadalp,
    kUadalp,
    kAbs,
    kNeg,
    kCmgtZero,
    kCmgeZero,
    kCmeqZero,
    kCmleZero,
    kCmltZero,
    kXtn,
    kSqxtn,
    kUqxtn,
    kFcvtn,
    kFcvtl,
    kFabs,
    kFneg,
    // region digitalis
    // Across-lanes reductions share the two-reg-misc dispatch path but are
    // distinguished by bit20=1 (across-lanes group) vs bit20=0 (two-reg-misc).
    kAddv,      // ADDV: U=0, opcode=11011
    kSmaxv,     // SMAXV: U=0, opcode=01010
    kUmaxv,     // UMAXV: U=1, opcode=01010
    kSminv,     // SMINV: U=0, opcode=11010
    kUminv,     // UMINV: U=1, opcode=11010
    kSuqadd,    // SUQADD: U=0, opcode=00011, bit20=0 (signed sat acc of unsigned)
    kUsqadd,    // USQADD: U=1, opcode=00011, bit20=0 (unsigned sat acc of signed)
    kSaddlv,    // SADDLV: U=0, opcode=00011, bit20=1 (signed add long across)
    kUaddlv,    // UADDLV: U=1, opcode=00011, bit20=1 (unsigned add long across)
    kScvtfV,    // SCVTF (vector, integer): U=0, opcode=11101, bit23=0
    kUcvtfV,    // UCVTF (vector, integer): U=1, opcode=11101, bit23=0
    // endregion
  };

  struct AdvSimdTwoRegMiscArgs {
    AdvSimdTwoRegMiscOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t size;     // element size: 00=8b, 01=16b, 10=32b, 11=64b
    bool q;           // Q bit: 0=64-bit vector, 1=128-bit vector
    bool u;           // U bit from encoding
  };

  // region digitalis
  //
  // AdvSIMD scalar two-reg misc opcodes.
  //
  enum class AdvSimdScalarTwoRegMiscOpcode : uint8_t {
    kScvtf,   // SCVTF (scalar): signed int → float
    kUcvtf,   // UCVTF (scalar): unsigned int → float
    kFcvtzs,  // FCVTZS (scalar): float → signed int, round toward zero
    kFcvtzu,  // FCVTZU (scalar): float → unsigned int, round toward zero
  };

  struct AdvSimdScalarTwoRegMiscArgs {
    AdvSimdScalarTwoRegMiscOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t size;   // 0=single, 1=double
  };
  // endregion

  // region digitalis
  //
  // AdvSIMD scalar three same opcodes.
  // Encoding: 01 U 11110 size 1 Rm opcode 1 Rn Rd
  // Scalar variants — operate on the bottom element of each register; only
  // size=11 (D-form, 64-bit) is typically defined for the integer arithmetic
  // family (ADD/SUB/CMxx/SHL).
  //
  enum class AdvSimdScalarThreeSameOpcode : uint8_t {
    kAdd,     // ADD  (scalar, D): U=0, opcode=10000
    kSub,     // SUB  (scalar, D): U=1, opcode=10000
    kCmgt,    // CMGT (scalar, D): U=0, opcode=00110 (signed >)
    kCmhi,    // CMHI (scalar, D): U=1, opcode=00110 (unsigned >)
    kCmge,    // CMGE (scalar, D): U=0, opcode=00111 (signed >=)
    kCmhs,    // CMHS (scalar, D): U=1, opcode=00111 (unsigned >=)
    kCmtst,   // CMTST(scalar, D): U=0, opcode=10001 (bitwise AND != 0)
    kCmeq,    // CMEQ (scalar, D): U=1, opcode=10001
    kSshl,    // SSHL (scalar, D): U=0, opcode=01000 (signed shift left)
    kUshl,    // USHL (scalar, D): U=1, opcode=01000 (unsigned shift left)
    // region digitalis - FP scalar three same.
    // For FP variants, size bits [23:22] = 1x where bit22 (sz) selects S (0) or D (1).
    kFabd,    // FABD (scalar, FP): U=1, bit23=1, opcode=11010
    kFcmgt,   // FCMGT (scalar, FP): U=1, bit23=1, opcode=11100
    kFcmge,   // FCMGE (scalar, FP): U=1, bit23=0, opcode=11100
    kFcmeq,   // FCMEQ (scalar, FP): U=0, bit23=0, opcode=11100
    kFacgt,   // FACGT (scalar, FP): U=1, bit23=1, opcode=11101
    kFacge,   // FACGE (scalar, FP): U=1, bit23=0, opcode=11101
    // endregion
  };

  struct AdvSimdScalarThreeSameArgs {
    AdvSimdScalarThreeSameOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t size;   // integer: full size field; FP: 0 -> S (32-bit), 1 -> D (64-bit)
  };

  // AdvSIMD scalar pairwise opcodes.
  // Encoding: 01 U 11110 size 11000 opcode 10 Rn Rd
  // Reads a pair of esize elements from Vn (low pair) and combines them into
  // a single scalar in Vd.
  enum class AdvSimdScalarPairwiseOpcode : uint8_t {
    kAddp,    // ADDP (scalar): U=0, size=11, opcode=11011 -> d-form
    // (FMAXNMP/FMAXP/FMINNMP/FMINP scalar variants live here too, but only
    //  ADDP scalar is needed at present.)
  };

  struct AdvSimdScalarPairwiseArgs {
    AdvSimdScalarPairwiseOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t size;
  };
  // endregion

  //
  // AdvSIMD shift by immediate opcodes.
  //
  enum class AdvSimdShiftImmOpcode : uint8_t {
    kSshr,
    kUshr,
    kSsra,
    kUsra,
    kSrshr,
    kUrshr,
    kSrsra,
    kUrsra,
    kSri,
    kShl,
    kSli,
    kSqshl,
    kUqshl,
    kSqshlu,
    kShrn,
    kRshrn,
    kSqshrn,
    kUqshrn,
    kSshll,
    kUshll,
  };

  struct AdvSimdShiftImmArgs {
    AdvSimdShiftImmOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t immh;     // immh field (bits[22:19])
    uint8_t immb;     // immb field (bits[18:16])
    bool q;           // Q bit
    bool u;           // U bit
  };
  // endregion

  // region digitalis
  //
  // AdvSIMD vector x indexed element opcodes.
  //
  enum class AdvSimdVecXIdxOpcode : uint8_t {
    kFmla,    // FMLA (by element)
    kFmls,    // FMLS (by element)
    kFmul,    // FMUL (by element)
    kMul,     // MUL (by element)
    kMla,     // MLA (by element)
    kMls,     // MLS (by element)
  };

  struct AdvSimdVecXIdxArgs {
    AdvSimdVecXIdxOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;       // indexed source register
    uint8_t index;    // element index within rm
    uint8_t size;     // 01=16b, 10=32b, 11=64b
    bool q;
  };
  // endregion

  // Signextend bits from size to the corresponding signed type of sizeof(Type) size.
  template <unsigned size, typename Type>
  static auto SignExtend(const Type val) {
    static_assert(std::is_integral_v<Type>, "Only integral types are supported");
    static_assert(size > 0 && size < (sizeof(Type) * CHAR_BIT), "Invalid size value");
    using SignedType = std::make_signed_t<Type>;
    struct {
      SignedType val : size;
    } holder = {.val = static_cast<SignedType>(val)};
    return static_cast<SignedType>(holder.val);
  }

  uint8_t Decode(const uint16_t* code) {
    // ARM64 instructions are always 4 bytes. The pointer may not be 4-byte aligned.
    memcpy(&code_, code, sizeof(code_));
    DecodeInstruction();
    return 4;
  }

 private:
  template <uint32_t start, uint32_t size>
  auto GetBits() const {
    static_assert((start + size) <= 32 && size > 0, "Invalid start or size value");
    using ResultType = std::conditional_t<
        size == 1,
        bool,
        std::conditional_t<size <= 8, uint8_t, std::conditional_t<size <= 16, uint16_t, uint32_t>>>;
    uint32_t shifted_val = code_ << (32 - start - size);
    return static_cast<ResultType>(shifted_val >> (32 - size));
  }

  void Undefined() { insn_consumer_->Undefined(); }

  // Decode the bitmask immediate encoding used by logical immediate instructions.
  // Returns the 64-bit immediate value.  The ARM64 encoding uses N, immr, imms fields.
  static bool DecodeBitmaskImmediate(uint32_t n, uint32_t immr, uint32_t imms, bool is_64bit,
                                     uint64_t* result) {
    // Determine the element size.
    unsigned len;
    if (n == 1) {
      len = 6;
    } else {
      // Find the highest bit set in (imms ^ 0x3f) that is part of the size field.
      uint32_t pattern = (~imms & 0x3f);
      if (pattern == 0) {
        return false;
      }
      len = 0;
      uint32_t tmp = pattern;
      while (tmp >>= 1) {
        len++;
      }
      if (len < 1) {
        return false;
      }
    }

    unsigned esize = 1u << len;
    unsigned levels = esize - 1;
    unsigned s = imms & levels;
    unsigned r = immr & levels;

    if (s == levels) {
      return false;  // Reserved encoding.
    }

    // region digitalis
    // Create a mask of (s + 1) ones. Handle s=63 to avoid UB from 1<<64.
    // uint64_t welem = (1ULL << (s + 1)) - 1;
    uint64_t welem = (s >= 63) ? ~0ULL : ((1ULL << (s + 1)) - 1);
    // endregion

    // region digitalis
    // Rotate right by r within esize bits.
    // // welem = ((welem >> r) | (welem << (esize - r))) & ((1ULL << esize) - 1);
    if (r != 0) {
      uint64_t mask = (esize >= 64) ? ~0ULL : ((1ULL << esize) - 1);
      welem = ((welem >> r) | (welem << (esize - r))) & mask;
    }
    // endregion

    // Replicate the esize-bit pattern to fill 64 bits.
    uint64_t imm = 0;
    for (unsigned i = 0; i < 64; i += esize) {
      imm |= welem << i;
    }

    if (!is_64bit) {
      imm &= 0xFFFFFFFFULL;
    }

    *result = imm;
    return true;
  }

  //
  // Top-level instruction dispatch.
  // ARM64 top-level encoding groups are determined by bits [28:25] (op0).
  //
  void DecodeInstruction() {
    uint8_t op0 = GetBits<25, 4>();

    switch (op0) {
      case 0b0000:
        // Reserved / unallocated.
        Undefined();
        break;
      case 0b1000:
      case 0b1001:
        // Data Processing - Immediate (op0 = 100x).
        DecodeDataProcessingImmediate();
        break;
      case 0b1010:
      case 0b1011:
        // Branches, Exception Generating, and System instructions (op0 = 101x).
        DecodeBranchExceptionSystem();
        break;
      case 0b0100:
      case 0b0110:
      case 0b1100:
      case 0b1110:
        // Loads and Stores (op0 = x1x0).
        DecodeLoadStore();
        break;
      case 0b0101:
      case 0b1101:
        // Data Processing - Register (op0 = x101).
        DecodeDataProcessingRegister();
        break;
      case 0b0111:
      case 0b1111:
        // region digitalis
        // SIMD & FP (op0 = x111).
        // // TODO: Implement SIMD/FP decoding.
        // Undefined();
        DecodeSimdFp();
        // endregion
        break;
      default:
        Undefined();
        break;
    }
  }

  //
  // Data Processing - Immediate.
  //
  void DecodeDataProcessingImmediate() {
    uint8_t op0 = GetBits<23, 3>();

    switch (op0) {
      case 0b000:
      case 0b001:
        // PC-rel. addressing: ADR, ADRP.
        DecodePcRelAddr();
        break;
      case 0b010:
        // Add/subtract (immediate).
        DecodeAddSubImmediate();
        break;
      case 0b011:
        // Add/subtract (immediate, with tags) - not implemented.
        Undefined();
        break;
      case 0b100:
        // Logical (immediate).
        DecodeLogicalImmediate();
        break;
      case 0b101:
        // Move wide (immediate).
        DecodeMoveWide();
        break;
      case 0b110:
        // Bitfield.
        DecodeBitfield();
        break;
      case 0b111:
        // region digitalis
        // Extract (EXTR/ROR).
        DecodeExtract();
        // endregion
        break;
      default:
        Undefined();
        break;
    }
  }

  void DecodePcRelAddr() {
    bool is_adrp = GetBits<31, 1>();
    uint8_t rd = GetBits<0, 5>();
    uint32_t immlo = GetBits<29, 2>();
    uint32_t immhi = GetBits<5, 19>();
    int64_t offset = SignExtend<21>(static_cast<uint32_t>((immhi << 2) | immlo));

    if (is_adrp) {
      offset <<= 12;  // Page-aligned: shift by 12.
    }

    const PcRelAddrArgs args = {
        .dst = rd,
        .offset = offset,
        .is_adrp = is_adrp,
    };
    insn_consumer_->PcRelAddr(args);
  }

  void DecodeAddSubImmediate() {
    bool sf = GetBits<31, 1>();
    bool is_sub = GetBits<30, 1>();
    bool set_flags = GetBits<29, 1>();
    uint8_t shift = GetBits<22, 1>();  // 0 or 1 (shift left by 0 or 12)
    uint32_t imm12 = GetBits<10, 12>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    uint32_t imm = shift ? (imm12 << 12) : imm12;

    const AddSubImmArgs args = {
        .dst = rd,
        .src = rn,
        .imm = imm,
        .is_64bit = sf,
        .is_sub = is_sub,
        .set_flags = set_flags,
    };
    insn_consumer_->AddSubImm(args);
  }

  void DecodeLogicalImmediate() {
    bool sf = GetBits<31, 1>();
    uint8_t opc = GetBits<29, 2>();
    uint8_t n = GetBits<22, 1>();
    uint8_t immr = GetBits<16, 6>();
    uint8_t imms = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // N must be 0 for 32-bit operations.
    if (!sf && n) {
      return Undefined();
    }

    uint64_t imm;
    if (!DecodeBitmaskImmediate(n, immr, imms, sf, &imm)) {
      return Undefined();
    }

    const LogicalImmArgs args = {
        .opcode = LogicalImmOpcode{opc},
        .dst = rd,
        .src = rn,
        .imm = imm,
        .is_64bit = sf,
    };
    insn_consumer_->LogicalImm(args);
  }

  void DecodeMoveWide() {
    bool sf = GetBits<31, 1>();
    uint8_t opc = GetBits<29, 2>();
    uint8_t hw = GetBits<21, 2>();
    uint16_t imm16 = GetBits<5, 16>();
    uint8_t rd = GetBits<0, 5>();

    // Validate: for 32-bit, hw must be 0 or 1. opc=01 is reserved.
    if (!sf && (hw >= 2)) {
      return Undefined();
    }
    if (opc == 0b01) {
      return Undefined();
    }

    const MoveWideArgs args = {
        .opcode = MoveWideOpcode{opc},
        .dst = rd,
        .imm = imm16,
        .shift = static_cast<uint8_t>(hw * 16),
        .is_64bit = sf,
    };
    insn_consumer_->MoveWide(args);
  }

  void DecodeBitfield() {
    bool sf = GetBits<31, 1>();
    uint8_t opc = GetBits<29, 2>();
    bool n = GetBits<22, 1>();
    uint8_t immr = GetBits<16, 6>();
    uint8_t imms = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // opc=11 is reserved.
    if (opc == 0b11) {
      return Undefined();
    }

    // N must match sf.
    if (sf != n) {
      return Undefined();
    }

    const BitfieldArgs args = {
        .opcode = BitfieldOpcode{opc},
        .dst = rd,
        .src = rn,
        .immr = immr,
        .imms = imms,
        .is_64bit = sf,
    };
    insn_consumer_->Bitfield(args);
  }

  //
  // Branches, Exception Generating, and System instructions.
  //
  void DecodeBranchExceptionSystem() {
    uint8_t op0 = GetBits<29, 3>();

    switch (op0) {
      case 0b000:
      case 0b100:
        // Unconditional branch (immediate): B or BL.
        DecodeBranchImm();
        break;
      case 0b010:
        // Conditional branch: B.cond.
        DecodeBranchCond();
        break;
      case 0b001:
      case 0b101:
        // region digitalis
        // Both CBZ/CBNZ and TBZ/TBNZ have op0=x01.
        // Distinguish by bit25: 0=CBZ/CBNZ, 1=TBZ/TBNZ.
        // // Compare and branch: CBZ/CBNZ.
        // DecodeCompareAndBranch();
        if (GetBits<25, 1>()) {
          DecodeTestAndBranch();
        } else {
          DecodeCompareAndBranch();
        }
        // endregion
        break;
      case 0b011:
      case 0b111:
        // region digitalis
        // // Test and branch: TBZ/TBNZ.
        // DecodeTestAndBranch();
        // Same dispatch as above — both x01 and x11 can reach here.
        if (GetBits<25, 1>()) {
          DecodeTestAndBranch();
        } else {
          DecodeCompareAndBranch();
        }
        // endregion
        break;
      case 0b110: {
        // This group contains: Unconditional branch (register), Exception generation, System.
        uint32_t opc_top = GetBits<22, 4>();
        if (opc_top == 0b0000) {
          // Exception generation (SVC, HVC, SMC, BRK, HLT, DCPS).
          DecodeExceptionGeneration();
        } else if (opc_top == 0b0100) {
          // System (MSR, MRS, NOP, DMB, DSB, ISB, SYS, SYSL).
          DecodeSystem();
        } else if ((opc_top & 0b1000) != 0) {
          // Unconditional branch (register): BR, BLR, RET.
          DecodeBranchReg();
        } else {
          Undefined();
        }
        break;
      }
      default:
        Undefined();
        break;
    }
  }

  void DecodeBranchImm() {
    bool is_link = GetBits<31, 1>();  // 0=B, 1=BL
    uint32_t imm26 = GetBits<0, 26>();
    int32_t offset = SignExtend<28>(static_cast<uint32_t>(imm26 << 2));

    const BranchImmArgs args = {
        .offset = offset,
        .is_link = is_link,
    };
    insn_consumer_->BranchImm(args);
  }

  void DecodeBranchCond() {
    uint32_t imm19 = GetBits<5, 19>();
    uint8_t cond = GetBits<0, 4>();
    int32_t offset = SignExtend<21>(static_cast<uint32_t>(imm19 << 2));

    const BranchCondArgs args = {
        .cond = Condition{cond},
        .offset = offset,
    };
    insn_consumer_->BranchCond(args);
  }

  void DecodeCompareAndBranch() {
    bool sf = GetBits<31, 1>();
    bool is_nonzero = GetBits<24, 1>();
    uint32_t imm19 = GetBits<5, 19>();
    uint8_t rt = GetBits<0, 5>();
    int32_t offset = SignExtend<21>(static_cast<uint32_t>(imm19 << 2));

    const CompareAndBranchArgs args = {
        .src = rt,
        .offset = offset,
        .is_64bit = sf,
        .is_nonzero = is_nonzero,
    };
    insn_consumer_->CompareAndBranch(args);
  }

  void DecodeTestAndBranch() {
    bool b5 = GetBits<31, 1>();
    bool is_nonzero = GetBits<24, 1>();
    uint8_t b40 = GetBits<19, 5>();
    uint32_t imm14 = GetBits<5, 14>();
    uint8_t rt = GetBits<0, 5>();
    int32_t offset = SignExtend<16>(static_cast<uint32_t>(imm14 << 2));
    uint8_t bit_num = static_cast<uint8_t>((b5 ? 32 : 0) | b40);

    const TestAndBranchArgs args = {
        .src = rt,
        .bit = bit_num,
        .offset = offset,
        .is_nonzero = is_nonzero,
    };
    insn_consumer_->TestAndBranch(args);
  }

  void DecodeExceptionGeneration() {
    uint8_t opc = GetBits<21, 3>();
    uint16_t imm16 = GetBits<5, 16>();
    uint8_t ll = GetBits<0, 2>();

    // SVC: opc=000, ll=01
    if (opc == 0b000 && ll == 0b01) {
      const SvcArgs args = {
          .imm = imm16,
      };
      insn_consumer_->Svc(args);
      return;
    }
    // Other exception instructions not implemented yet.
    Undefined();
  }

  void DecodeSystem() {
    uint8_t l = GetBits<21, 1>();    // 0=MSR, 1=MRS
    uint8_t op0 = GetBits<19, 2>();
    uint8_t op1 = GetBits<16, 3>();
    uint8_t crn = GetBits<12, 4>();
    uint8_t crm = GetBits<8, 4>();
    uint8_t op2 = GetBits<5, 3>();
    uint8_t rt = GetBits<0, 5>();

    // NOP and other HINT instructions: SYS with CRn=0010, op0=00.
    if (op0 == 0b00 && l == 0) {
      if (crn == 0b0010) {
        // HINT instructions: NOP is CRm=0000, op2=000, Rt=11111.
        // DMB: CRn=0011, CRm=barrier_type, op2=001
        // DSB: CRn=0011, CRm=barrier_type, op2=100
        // ISB: CRn=0011, CRm=barrier_type, op2=110
        insn_consumer_->Nop();
        return;
      }
      if (crn == 0b0011) {
        // Barrier instructions: DMB, DSB, ISB.
        // For now, treat as NOP (barriers are handled by host memory model).
        insn_consumer_->Nop();
        return;
      }
    }

    // region digitalis
    // SYS instructions (op0=01): cache maintenance (DC, IC), TLB ops, etc.
    // Safe to treat as NOP in interpreter.
    if (op0 == 0b01 && l == 0) {
      insn_consumer_->Nop();
      return;
    }
    // SYSL instructions (op0=01, l=1): also safe as NOP.
    if (op0 == 0b01 && l == 1) {
      insn_consumer_->Nop();
      return;
    }
    // endregion

    // MRS/MSR with op0 >= 2 (system register access).
    if (op0 >= 2) {
      // Encode system register as: op0:op1:CRn:CRm:op2.
      uint16_t sysreg = static_cast<uint16_t>(
          (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2);

      if (l) {
        // MRS Xt, <sysreg>
        const MrsArgs args = {
            .dst = rt,
            .sysreg = SystemReg{sysreg},
        };
        insn_consumer_->Mrs(args);
      } else {
        // MSR <sysreg>, Xt
        const MsrArgs args = {
            .src = rt,
            .sysreg = SystemReg{sysreg},
        };
        insn_consumer_->Msr(args);
      }
      return;
    }

    Undefined();
  }

  void DecodeBranchReg() {
    uint8_t opc = GetBits<21, 4>();
    uint8_t rn = GetBits<5, 5>();

    switch (opc) {
      case 0b0000: {
        // BR Xn
        const BranchRegArgs args = {
            .src = rn,
            .link_reg = 0,
            .is_link = false,
            .is_ret = false,
        };
        insn_consumer_->BranchReg(args);
        break;
      }
      case 0b0001: {
        // BLR Xn
        const BranchRegArgs args = {
            .src = rn,
            .link_reg = 30,
            .is_link = true,
            .is_ret = false,
        };
        insn_consumer_->BranchReg(args);
        break;
      }
      case 0b0010: {
        // RET {Xn} (default Xn = X30)
        const BranchRegArgs args = {
            .src = rn,
            .link_reg = 0,
            .is_link = false,
            .is_ret = true,
        };
        insn_consumer_->BranchReg(args);
        break;
      }
      default:
        Undefined();
        break;
    }
  }

  //
  // Loads and Stores.
  //
  void DecodeLoadStore() {
    // ARM64 Load/Store encoding: top bits[28:25] = x1x0.
    // Further decoded by bit[29], bit[28:26], bit[24:23], bit[11:10].
    //
    // Major sub-groups by bits[29:27] and bit[24]:
    //   bit[29]=0, bit[26]=0, bit[24]=0: Load/store no-allocate pair / LDP/STP
    //   bit[29]=1, bit[26]=0: Load/store register
    //   etc.
    //
    // Simplified dispatch:

    uint8_t op_29 = GetBits<29, 1>();
    uint8_t op_28_27 = GetBits<27, 2>();
    uint8_t op_26 = GetBits<26, 1>();
    uint8_t op_24 = GetBits<24, 1>();
    uint8_t op_23 = GetBits<23, 1>();
    uint8_t op4 = GetBits<10, 2>();

    // region digitalis
    // Load/store exclusive/atomic: bit29=0, op_28_27=01, op_26=0
    // Includes: LDXR, STXR, LDAXR, STLXR, LDAR, STLR, CAS, CASP
    if (op_28_27 == 0b01 && op_26 == 0 && op_29 == 0) {
      DecodeLoadStoreExclusive();
      return;
    }
    // endregion

    // Load/store pair: bit29=1, op_28_27=01, op_26=0
    // Encoding: opc[31:30] 101 0 0xx xxxxxxx (pairs)
    if (op_28_27 == 0b01 && op_26 == 0) {
      DecodeLoadStorePair();
      return;
    }

    // region digitalis
    // AdvSIMD LD/ST: bit29=0, bits[28:27]=01, bit[26]=1
    //   bit[24]=0: multiple structures (LD1-4, ST1-4)
    //   bit[24]=1: single structure (LD1/ST1 to one lane, LD1R-LD4R replicate)
    if (op_29 == 0 && op_28_27 == 0b01 && op_26 == 1) {
      if (!op_24) {
        DecodeAdvSimdMultiStruct();
      } else {
        DecodeAdvSimdSingleStruct();
      }
      return;
    }
    // SIMD/FP load/store pair: bit29=1, bits[28:27]=01, bit[26]=1
    if (op_29 == 1 && op_28_27 == 0b01 && op_26 == 1) {
      DecodeSimdLoadStorePair();
      return;
    }
    // endregion

    // region digitalis
    // SIMD/FP load/store register (various): bits[29:27] = x11, bit[26]=1
    if (op_28_27 == 0b11 && op_26 == 1) {
      if (op_24) {
        DecodeSimdLoadStoreUnsignedImm();
        return;
      }
      if (op4 == 0b01) {
        DecodeSimdLoadStoreImmPostPreIndex(false);
        return;
      }
      if (op4 == 0b11) {
        DecodeSimdLoadStoreImmPostPreIndex(true);
        return;
      }
      if (op4 == 0b00) {
        // Unscaled immediate (LDUR/STUR for SIMD).
        DecodeSimdLoadStoreUnscaled();
        return;
      }
      if (op4 == 0b10) {
        DecodeSimdLoadStoreRegOffset();
        return;
      }
    }
    // endregion

    // Load/store register (various): bits[29:27] = x11, bit[26]=0
    // Encoding: size[31:30] 111 0 00xx ... (unscaled/pre/post)
    //           size[31:30] 111 0 01xx ... (unsigned offset)
    if (op_28_27 == 0b11 && op_26 == 0) {
      if (op_24) {
        // bit[24]=1: Load/store register (unsigned immediate).
        DecodeLoadStoreUnsignedImm();
        return;
      }
      // bit[24]=0: Sub-dispatch on bits[11:10].
      if (op4 == 0b01) {
        // Post-index.
        DecodeLoadStoreImmPostPreIndex(false);
        return;
      }
      if (op4 == 0b11) {
        // Pre-index.
        DecodeLoadStoreImmPostPreIndex(true);
        return;
      }
      if (op4 == 0b00) {
        // region digitalis
        // bit[21]=1: Atomic memory operations (SWP, LDADD, etc.)
        // bit[21]=0: Unscaled immediate (LDUR/STUR).
        if (GetBits<21, 1>()) {
          DecodeAtomicMemoryOp();
        } else {
          DecodeLoadStoreUnscaled();
        }
        // endregion
        return;
      }
      if (op4 == 0b10) {
        // Load/store register (register offset).
        DecodeLoadStoreRegOffset();
        return;
      }
    }

    // Catch-all for other load/store variants not yet implemented (SIMD, exclusive, etc.).
    Undefined();
  }

  void DecodeLoadStoreUnsignedImm() {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint32_t imm12 = GetBits<10, 12>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    // region digitalis - PRFM (immediate) is size=0b11, opc=0b10. The
    // previous check (opc=0b11) never fired and let PRFM execute as
    // LDRSW Xt, [Xn, #imm12] with Rt=0 (the prefetch type), silently
    // clobbering X0 with eight bytes from [Xn + imm12]. NOP it.
    if (size == 0b11 && opc == 0b10) {
      insn_consumer_->Nop();
      return;
    }
    // endregion

    bool is_store = ((opc & 0b01) == 0) && ((opc & 0b10) == 0);
    bool is_signed = (opc & 0b10) != 0;
    bool is_64bit_target = (opc & 0b01) != 0;

    // opc encoding:
    // 00 = STR
    // 01 = LDR
    // 10 = LDRS (sign-extend to 64-bit)  [size=11 -> PRFM, handled above]
    // 11 = LDRS (sign-extend to 32-bit)
    if (opc == 0b00) {
      is_store = true;
    } else if (opc == 0b01) {
      is_store = false;
      is_signed = false;
    } else if (opc == 0b10) {
      is_store = false;
      is_signed = true;
      is_64bit_target = true;
    } else {
      is_store = false;
      is_signed = true;
      is_64bit_target = false;
    }

    // Scale offset by access size.
    int32_t offset = static_cast<int32_t>(imm12 << size);

    const LoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = offset,
        .size = LoadStoreSize{size},
        .is_store = is_store,
        .is_signed = is_signed,
        .is_64bit_target = is_64bit_target,
    };
    insn_consumer_->LoadStoreImm(args);
  }

  void DecodeLoadStoreImmPostPreIndex(bool is_preindex) {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint32_t imm9 = GetBits<12, 9>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    int32_t offset = SignExtend<9>(imm9);

    bool is_store;
    bool is_signed;
    bool is_64bit_target;

    if (opc == 0b00) {
      is_store = true;
      is_signed = false;
      is_64bit_target = false;
    } else if (opc == 0b01) {
      is_store = false;
      is_signed = false;
      is_64bit_target = false;
    } else if (opc == 0b10) {
      is_store = false;
      is_signed = true;
      is_64bit_target = true;
    } else {
      is_store = false;
      is_signed = true;
      is_64bit_target = false;
    }

    // For pre/post index, we encode using LoadStoreImm and the interpreter handles the writeback.
    // But for now, we treat them as simple offset (TODO: proper pre/post-index support).
    const LoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = offset,
        .size = LoadStoreSize{size},
        .is_store = is_store,
        .is_signed = is_signed,
        .is_64bit_target = is_64bit_target,
    };

    if (is_preindex) {
      insn_consumer_->LoadStoreImmPreIndex(args);
    } else {
      insn_consumer_->LoadStoreImmPostIndex(args);
    }
  }

  void DecodeLoadStoreUnscaled() {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint32_t imm9 = GetBits<12, 9>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    // region digitalis - PRFUM (prefetch unscaled) shares this encoding with
    // size=0b11, opc=0b10. NOP it; otherwise it would be decoded as LDURSW
    // into Rt (prefetch type code), clobbering the destination register.
    if (size == 0b11 && opc == 0b10) {
      insn_consumer_->Nop();
      return;
    }
    // endregion

    int32_t offset = SignExtend<9>(imm9);

    bool is_store;
    bool is_signed;
    bool is_64bit_target;

    if (opc == 0b00) {
      is_store = true;
      is_signed = false;
      is_64bit_target = false;
    } else if (opc == 0b01) {
      is_store = false;
      is_signed = false;
      is_64bit_target = false;
    } else if (opc == 0b10) {
      is_store = false;
      is_signed = true;
      is_64bit_target = true;
    } else {
      is_store = false;
      is_signed = true;
      is_64bit_target = false;
    }

    const LoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = offset,
        .size = LoadStoreSize{size},
        .is_store = is_store,
        .is_signed = is_signed,
        .is_64bit_target = is_64bit_target,
    };
    insn_consumer_->LoadStoreImm(args);
  }

  void DecodeLoadStoreRegOffset() {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t option = GetBits<13, 3>();
    bool s_bit = GetBits<12, 1>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    // region digitalis - PRFM (register) shares this encoding with
    // size=0b11, opc=0b10. Without this guard, the decoder treats the
    // prefetch as an LDRSW into Rt (where Rt is the prefetch type code,
    // typically 0 = pldl1keep), silently clobbering X0 with eight bytes
    // from [Xn, Xm]. Observed as bad x0 (e.g. 0x000000XX_00070001) in
    // libsuperpack-jni.so's Brotli/SP2 decompressor on Facebook startup.
    // PRFM has no architectural side effects we need to emulate; NOP it.
    if (size == 0b11 && opc == 0b10) {
      insn_consumer_->Nop();
      return;
    }
    // endregion

    bool is_store;
    bool is_signed;
    bool is_64bit_target;

    if (opc == 0b00) {
      is_store = true;
      is_signed = false;
      is_64bit_target = false;
    } else if (opc == 0b01) {
      is_store = false;
      is_signed = false;
      is_64bit_target = false;
    } else if (opc == 0b10) {
      is_store = false;
      is_signed = true;
      is_64bit_target = true;
    } else {
      is_store = false;
      is_signed = true;
      is_64bit_target = false;
    }

    uint8_t shift_amount = s_bit ? size : 0;

    // region digitalis - Validate option field. Only word-or-larger
    // offsets are encodable: 010=UXTW, 011=LSL/UXTX, 110=SXTW, 111=SXTX.
    // Other options are UNDEFINED per ARMv8.
    switch (option) {
      case 0b010:
      case 0b011:
      case 0b110:
      case 0b111:
        break;
      default:
        return Undefined();
    }

    const LoadStoreRegArgs args = {
        .rt = rt,
        .rn = rn,
        .rm = rm,
        .extend_type = option,
        .shift_amount = shift_amount,
        .size = LoadStoreSize{size},
        .is_store = is_store,
        .is_signed = is_signed,
        .is_64bit_target = is_64bit_target,
    };
    // endregion
    insn_consumer_->LoadStoreReg(args);
  }

  void DecodeLoadStorePair() {
    bool is_64bit = GetBits<31, 1>();
    uint8_t type = GetBits<23, 2>(); // 01=post-index, 10=signed-offset, 11=pre-index
    bool is_load = GetBits<22, 1>();
    uint32_t imm7 = GetBits<15, 7>();
    uint8_t rt2 = GetBits<10, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt1 = GetBits<0, 5>();

    LoadStoreSize size = is_64bit ? LoadStoreSize::k64bit : LoadStoreSize::k32bit;
    uint8_t scale = is_64bit ? 3 : 2;
    int32_t offset = SignExtend<7>(imm7) << scale;

    bool is_preindex = (type == 0b11);
    bool is_postindex = (type == 0b01);

    const LoadStorePairArgs args = {
        .rt1 = rt1,
        .rt2 = rt2,
        .rn = rn,
        .offset = offset,
        .size = size,
        .is_store = !is_load,
        .is_preindex = is_preindex,
        .is_postindex = is_postindex,
    };
    insn_consumer_->LoadStorePair(args);
  }

  // region digitalis
  //
  // SIMD & FP - top-level decode for op0 = x111.
  //
  void DecodeSimdFp() {
    bool bit31 = GetBits<31, 1>();

    // AdvSIMD modified immediate: bit31=0, bits[28:24]=01111, bits[23:19]=00000, bit10=1
    if (!bit31 && GetBits<24, 5>() == 0b01111 && GetBits<19, 5>() == 0 && GetBits<10, 1>()) {
      DecodeAdvSimdModifiedImm();
      return;
    }

    // region digitalis
    // AdvSIMD scalar two-reg misc: bit31=0, bit30=1, bits[28:24]=11110, bits[21:17]=10000, bits[11:10]=10
    // Must be checked BEFORE FpDataProc1/FpIntConversion/FpDataProc2 because all share bits[28:24]=11110,
    // but scalar SIMD has bit30=1 while scalar FP has bit30=0.
    if (!bit31 && GetBits<30, 1>() && GetBits<24, 5>() == 0b11110 &&
        GetBits<17, 5>() == 0b10000 && GetBits<10, 2>() == 0b10) {
      DecodeAdvSimdScalarTwoRegMisc();
      return;
    }

    // AdvSIMD scalar copy (DUP scalar / MOV Vd, Vn[index]):
    //   bit31=0, bit30=1, op=bit29=0, bits[28:24]=11110, bits[23:21]=000,
    //   bit15=0, imm4=bits[14:11]=0000, bit10=1
    // Distinct from scalar two-reg misc (bits[21:17]=10000, bits[11:10]=10).
    // Distinct from FP scalar ops (those have bit30=0).
    if (!bit31 && GetBits<30, 1>() && !GetBits<29, 1>() &&
        GetBits<24, 5>() == 0b11110 && GetBits<21, 3>() == 0b000 &&
        !GetBits<15, 1>() && GetBits<11, 4>() == 0b0000 && GetBits<10, 1>()) {
      DecodeAdvSimdScalarCopy();
      return;
    }

    // AdvSIMD scalar pairwise:
    //   bit31=0, bit30=1, bits[28:24]=11110, bits[21:17]=11000, bits[11:10]=10
    // Must be checked BEFORE scalar three same (which only requires bit21=1).
    if (!bit31 && GetBits<30, 1>() && GetBits<24, 5>() == 0b11110 &&
        GetBits<17, 5>() == 0b11000 && GetBits<10, 2>() == 0b10) {
      DecodeAdvSimdScalarPairwise();
      return;
    }

    // AdvSIMD scalar three same:
    //   bit31=0, bit30=1, bits[28:24]=11110, bit21=1, bit10=1
    // Must be checked AFTER scalar two-reg misc (which requires
    // bits[20:17]=0000) to avoid mis-routing — for scalar three same we
    // require Rm != 0 conceptually, but the safer way is just ordering
    // and demanding bits[14:11] (opcode field) is non-zero in a way
    // that doesn't match two-reg-misc opcode shape.
    if (!bit31 && GetBits<30, 1>() && GetBits<24, 5>() == 0b11110 &&
        GetBits<21, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdScalarThreeSame();
      return;
    }

    // region digitalis
    // FP <-> fixed-point conversion: bits[28:24]=11110, bit21=0
    // Must be checked BEFORE all bit21=1 FP checks.
    // Encoding: sf 0 S 11110 ftype 0 rmode opcode scale Rn Rd
    if (GetBits<24, 5>() == 0b11110 && !GetBits<21, 1>()) {
      DecodeFpFixedPointConversion();
      return;
    }
    // endregion

    // Floating-point data-processing (1 source): bits[28:24]=11110, bit21=1, bits[14:10]=10000
    // Must be checked BEFORE FpIntConversion because both share bits[28:24]=11110 and bit21=1,
    // but FpDataProc1 has bits[14:10]=10000 while FpIntConversion has bits[15:10]=000000.
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() &&
        GetBits<10, 5>() == 0b10000) {
      DecodeFpDataProc1();
      return;
    }

    // Floating-point <-> integer conversion: bits[28:24]=11110, bit21=1, bits[15:10]=000000
    // bit31=sf can be 0 or 1 (GP register size).
    if (GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 6>() == 0b000000) {
      DecodeFpIntConversion();
      return;
    }

    // region digitalis
    // FMOV (scalar, immediate): bit31=0, bits[28:24]=11110, bit21=1, bits[12:10]=100, bits[9:5]=00000
    // Encoding: 0 0 0 11110 ftype 1 imm8 100 00000 Rd
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() &&
        GetBits<10, 3>() == 0b100 && GetBits<5, 5>() == 0b00000) {
      DecodeFpMovImmediate();
      return;
    }
    // endregion

    // Floating-point data-processing (2 source): bit31=0, bits[28:24]=11110, bit21=1, bits[11:10]=10
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b10) {
      DecodeFpDataProc2();
      return;
    }
    // endregion

    // Floating-point compare: bit31=0, bits[28:24]=11110, bit21=1, bits[13:10]=1000
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 4>() == 0b1000) {
      DecodeFpCompare();
      return;
    }

    // region digitalis
    // FCSEL: bit31=0, bits[28:24]=11110, bit21=1, bits[11:10]=11
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b11) {
      DecodeFpCondSelect();
      return;
    }

    // Floating-point data-processing (3 source): bit31=0, bits[28:24]=11111
    // FMADD, FMSUB, FNMADD, FNMSUB
    if (!bit31 && GetBits<24, 5>() == 0b11111) {
      DecodeFpDataProc3();
      return;
    }
    // endregion

    // region digitalis
    // AdvSIMD three different: bit31=0, bits[28:24]=01110, bit21=1, bits[11:10]=00
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b00) {
      DecodeAdvSimdThreeDiff();
      return;
    }
    // endregion

    // AdvSIMD three same: bit31=0, bits[28:24]=01110, bit21=1, bit10=1
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<21, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdThreeSame();
      return;
    }

    // region digitalis
    // AdvSIMD permute (UZP1, TRN1, ZIP1, UZP2, TRN2, ZIP2):
    // bit31=0, bits[28:24]=01110, bit21=0, bit15=0, bits[11:10]=10
    // Must be checked BEFORE two-reg misc since both share bits[11:10]=10 but
    // permute has bit21=0 while two-reg misc has bit21=1 (bits[21:17]=10000).
    if (!bit31 && GetBits<24, 5>() == 0b01110 && !GetBits<21, 1>() &&
        !GetBits<15, 1>() && GetBits<10, 2>() == 0b10) {
      uint8_t opcode = GetBits<12, 3>();
      insn_consumer_->AdvSimdPermute(
          GetBits<0, 5>(),   // rd
          GetBits<5, 5>(),   // rn
          GetBits<16, 5>(),  // rm
          GetBits<22, 2>(),  // size
          opcode,            // permute opcode (001=UZP1,010=TRN1,011=ZIP1,101=UZP2,110=TRN2,111=ZIP2)
          GetBits<30, 1>()); // q
      return;
    }
    // endregion

    // AdvSIMD two-reg misc: bit31=0, bits[28:24]=01110, bit21=1, bit17=0, bits[11:10]=10
    if (!bit31 && GetBits<24, 5>() == 0b01110 && !GetBits<17, 1>() && GetBits<10, 2>() == 0b10) {
      DecodeAdvSimdTwoRegMisc();
      return;
    }

    // AdvSIMD copy (DUP, INS, SMOV, UMOV): bit31=0, bits[28:24]=01110, bit21=0, bit15=0, bit10=1
    if (!bit31 && GetBits<24, 5>() == 0b01110 && !GetBits<21, 1>() && !GetBits<15, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdCopy();
      return;
    }

    // region digitalis
    // AdvSIMD vector x indexed element: bit31=0, bits[28:24]=01111, bit10=0
    if (!bit31 && GetBits<24, 5>() == 0b01111 && !GetBits<10, 1>()) {
      DecodeAdvSimdVecXIndexedElement();
      return;
    }
    // endregion

    // AdvSIMD shift by immediate: bit31=0, bits[28:24]=01111, bit10=1, immh!=0000
    if (!bit31 && GetBits<24, 5>() == 0b01111 && GetBits<10, 1>() && GetBits<19, 4>() != 0) {
      DecodeAdvSimdShiftByImm();
      return;
    }

    // region digitalis
    // AdvSIMD extract (EXT): bit31=0, bits[28:24]=01110, bits[23:22]=00, bit21=0, bit15=0, bit10=0
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<22, 2>() == 0 &&
        !GetBits<21, 1>() && !GetBits<15, 1>() && !GetBits<10, 1>()) {
      // EXT Vd.<T>, Vn.<T>, Vm.<T>, #index
      insn_consumer_->AdvSimdExtract(
          GetBits<0, 5>(),   // rd
          GetBits<5, 5>(),   // rn
          GetBits<16, 5>(),  // rm
          GetBits<11, 4>(),  // imm4 (byte index)
          GetBits<30, 1>()); // q
      return;
    }
    // endregion

    Undefined();
  }

  //
  // AdvSIMD modified immediate (MOVI, MVNI, ORR imm, BIC imm, FMOV imm).
  //
  void DecodeAdvSimdModifiedImm() {
    bool q = GetBits<30, 1>();
    uint8_t op = GetBits<29, 1>();
    uint8_t abc = GetBits<16, 3>();
    uint8_t cmode = GetBits<12, 4>();
    uint8_t defgh = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    const SimdModifiedImmArgs args = {
        .rd = rd,
        .cmode = cmode,
        .op = op,
        .abc = abc,
        .defgh = defgh,
        .q = q,
    };
    insn_consumer_->SimdModifiedImm(args);
  }

  //
  // SIMD/FP load/store (unsigned immediate) - bit[26]=1 variant.
  //
  void DecodeSimdLoadStoreUnsignedImm() {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint32_t imm12 = GetBits<10, 12>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    SimdLoadStoreSize ls_size;
    uint8_t scale;
    bool is_store;

    // SIMD/FP encoding: size:opc determines register width
    if (size == 0b00 && opc == 0b00) { ls_size = SimdLoadStoreSize::k8bit; scale = 1; is_store = true; }
    else if (size == 0b00 && opc == 0b01) { ls_size = SimdLoadStoreSize::k8bit; scale = 1; is_store = false; }
    else if (size == 0b01 && opc == 0b00) { ls_size = SimdLoadStoreSize::k16bit; scale = 2; is_store = true; }
    else if (size == 0b01 && opc == 0b01) { ls_size = SimdLoadStoreSize::k16bit; scale = 2; is_store = false; }
    else if (size == 0b10 && opc == 0b00) { ls_size = SimdLoadStoreSize::k32bit; scale = 4; is_store = true; }
    else if (size == 0b10 && opc == 0b01) { ls_size = SimdLoadStoreSize::k32bit; scale = 4; is_store = false; }
    else if (size == 0b11 && opc == 0b00) { ls_size = SimdLoadStoreSize::k64bit; scale = 8; is_store = true; }
    else if (size == 0b11 && opc == 0b01) { ls_size = SimdLoadStoreSize::k64bit; scale = 8; is_store = false; }
    else if (size == 0b00 && opc == 0b10) { ls_size = SimdLoadStoreSize::k128bit; scale = 16; is_store = true; }
    else if (size == 0b00 && opc == 0b11) { ls_size = SimdLoadStoreSize::k128bit; scale = 16; is_store = false; }
    else { Undefined(); return; }

    const SimdLoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = static_cast<int64_t>(imm12) * scale,
        .size = ls_size,
        .is_store = is_store,
    };
    insn_consumer_->SimdLoadStoreImm(args);
  }

  //
  // SIMD/FP load/store (pre/post index) - bit[26]=1 variant.
  //
  void DecodeSimdLoadStoreImmPostPreIndex(bool is_preindex) {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint32_t imm9 = GetBits<12, 9>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    SimdLoadStoreSize ls_size;
    bool is_store;

    if (size == 0b00 && opc == 0b00) { ls_size = SimdLoadStoreSize::k8bit; is_store = true; }
    else if (size == 0b00 && opc == 0b01) { ls_size = SimdLoadStoreSize::k8bit; is_store = false; }
    else if (size == 0b01 && opc == 0b00) { ls_size = SimdLoadStoreSize::k16bit; is_store = true; }
    else if (size == 0b01 && opc == 0b01) { ls_size = SimdLoadStoreSize::k16bit; is_store = false; }
    else if (size == 0b10 && opc == 0b00) { ls_size = SimdLoadStoreSize::k32bit; is_store = true; }
    else if (size == 0b10 && opc == 0b01) { ls_size = SimdLoadStoreSize::k32bit; is_store = false; }
    else if (size == 0b11 && opc == 0b00) { ls_size = SimdLoadStoreSize::k64bit; is_store = true; }
    else if (size == 0b11 && opc == 0b01) { ls_size = SimdLoadStoreSize::k64bit; is_store = false; }
    else if (size == 0b00 && opc == 0b10) { ls_size = SimdLoadStoreSize::k128bit; is_store = true; }
    else if (size == 0b00 && opc == 0b11) { ls_size = SimdLoadStoreSize::k128bit; is_store = false; }
    else { Undefined(); return; }

    int32_t offset = SignExtend<9>(imm9);

    const SimdLoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = offset,
        .size = ls_size,
        .is_store = is_store,
    };

    if (is_preindex) {
      insn_consumer_->SimdLoadStoreImmPreIndex(args);
    } else {
      insn_consumer_->SimdLoadStoreImmPostIndex(args);
    }
  }

  //
  // SIMD/FP load/store (unscaled immediate) - bit[26]=1 variant.
  //
  void DecodeSimdLoadStoreUnscaled() {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint32_t imm9 = GetBits<12, 9>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    SimdLoadStoreSize ls_size;
    bool is_store;

    if (size == 0b00 && opc == 0b00) { ls_size = SimdLoadStoreSize::k8bit; is_store = true; }
    else if (size == 0b00 && opc == 0b01) { ls_size = SimdLoadStoreSize::k8bit; is_store = false; }
    else if (size == 0b01 && opc == 0b00) { ls_size = SimdLoadStoreSize::k16bit; is_store = true; }
    else if (size == 0b01 && opc == 0b01) { ls_size = SimdLoadStoreSize::k16bit; is_store = false; }
    else if (size == 0b10 && opc == 0b00) { ls_size = SimdLoadStoreSize::k32bit; is_store = true; }
    else if (size == 0b10 && opc == 0b01) { ls_size = SimdLoadStoreSize::k32bit; is_store = false; }
    else if (size == 0b11 && opc == 0b00) { ls_size = SimdLoadStoreSize::k64bit; is_store = true; }
    else if (size == 0b11 && opc == 0b01) { ls_size = SimdLoadStoreSize::k64bit; is_store = false; }
    else if (size == 0b00 && opc == 0b10) { ls_size = SimdLoadStoreSize::k128bit; is_store = true; }
    else if (size == 0b00 && opc == 0b11) { ls_size = SimdLoadStoreSize::k128bit; is_store = false; }
    else { Undefined(); return; }

    int32_t offset = SignExtend<9>(imm9);

    const SimdLoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = offset,
        .size = ls_size,
        .is_store = is_store,
    };
    insn_consumer_->SimdLoadStoreImm(args);
  }

  //
  // SIMD/FP load/store (register offset) - bit[26]=1 variant.
  //
  void DecodeSimdLoadStoreRegOffset() {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t option = GetBits<13, 3>();
    uint8_t s_bit = GetBits<12, 1>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    SimdLoadStoreSize ls_size;
    bool is_store;
    uint8_t shift_amount = 0;

    if (size == 0b00 && opc == 0b00) { ls_size = SimdLoadStoreSize::k8bit; is_store = true; if (s_bit) shift_amount = 0; }
    else if (size == 0b00 && opc == 0b01) { ls_size = SimdLoadStoreSize::k8bit; is_store = false; if (s_bit) shift_amount = 0; }
    else if (size == 0b01 && opc == 0b00) { ls_size = SimdLoadStoreSize::k16bit; is_store = true; if (s_bit) shift_amount = 1; }
    else if (size == 0b01 && opc == 0b01) { ls_size = SimdLoadStoreSize::k16bit; is_store = false; if (s_bit) shift_amount = 1; }
    else if (size == 0b10 && opc == 0b00) { ls_size = SimdLoadStoreSize::k32bit; is_store = true; if (s_bit) shift_amount = 2; }
    else if (size == 0b10 && opc == 0b01) { ls_size = SimdLoadStoreSize::k32bit; is_store = false; if (s_bit) shift_amount = 2; }
    else if (size == 0b11 && opc == 0b00) { ls_size = SimdLoadStoreSize::k64bit; is_store = true; if (s_bit) shift_amount = 3; }
    else if (size == 0b11 && opc == 0b01) { ls_size = SimdLoadStoreSize::k64bit; is_store = false; if (s_bit) shift_amount = 3; }
    else if (size == 0b00 && opc == 0b10) { ls_size = SimdLoadStoreSize::k128bit; is_store = true; if (s_bit) shift_amount = 4; }
    else if (size == 0b00 && opc == 0b11) { ls_size = SimdLoadStoreSize::k128bit; is_store = false; if (s_bit) shift_amount = 4; }
    else { Undefined(); return; }

    // region digitalis - Reject UNDEFINED option encodings for the
    // offset register (only 010/011/110/111 are valid SIMD load/store
    // forms). Preserve the option so the handler can apply the right
    // extension.
    switch (option) {
      case 0b010:
      case 0b011:
      case 0b110:
      case 0b111:
        break;
      default:
        Undefined();
        return;
    }

    const SimdLoadStoreRegArgs args = {
        .rt = rt,
        .rn = rn,
        .rm = rm,
        .extend_type = option,
        .shift_amount = shift_amount,
        .size = ls_size,
        .is_store = is_store,
    };
    // endregion
    insn_consumer_->SimdLoadStoreReg(args);
  }

  //
  // SIMD/FP load/store pair - bit[26]=1 variant.
  //
  // region digitalis
  //
  // AdvSIMD load/store multiple structures (LD1/ST1 with 1-4 registers).
  //
  // No post-index: 0 Q 001100 0 L 0 00000 opcode size Rn Rt
  // Post-index:    0 Q 001100 1 L 0 Rm    opcode size Rn Rt
  //   L = bit22 (1=load), Rm = bits[20:16] (11111 = imm post-index)
  //   opcode = bits[15:12]: 0111=1reg, 1010=2regs, 0110=3regs, 0010=4regs
  //   size = bits[11:10]
  //
  void DecodeAdvSimdMultiStruct() {
    bool q = GetBits<30, 1>();
    bool postindex = GetBits<23, 1>();
    bool is_load = GetBits<22, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode = GetBits<12, 4>();
    uint8_t size = GetBits<10, 2>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    uint8_t num_regs;
    bool is_interleaved;
    switch (opcode) {
      // LD1 / ST1 with N contiguous registers (no de-interleave).
      case 0b0111: num_regs = 1; is_interleaved = false; break;
      case 0b1010: num_regs = 2; is_interleaved = false; break;
      case 0b0110: num_regs = 3; is_interleaved = false; break;
      case 0b0010: num_regs = 4; is_interleaved = false; break;
      // region digitalis - LD2 / LD3 / LD4 (de-interleaving on load,
      // interleaving on store). Distinct from LDn-with-1-reg-N-times.
      case 0b1000: num_regs = 2; is_interleaved = true; break;
      case 0b0100: num_regs = 3; is_interleaved = true; break;
      case 0b0000: num_regs = 4; is_interleaved = true; break;
      // endregion
      default: Undefined(); return;
    }

    insn_consumer_->AdvSimdMultiStruct(rt, rn, num_regs, size, q, !is_load, postindex, rm,
                                       is_interleaved);
  }

  //
  // AdvSIMD load/store single structure.
  // Encoding: 0 Q 001101 P L R Rm opcode S size Rn Rt
  //   where P = bit[23] (post-indexed if 1)
  //   L = bit[22] (load/store), R = bit[21] (register count modifier)
  //   Rm = bits[20:16] (post-index register, 11111 = immediate)
  //   opcode = bits[15:13], S = bit[12], size = bits[11:10]
  //
  void DecodeAdvSimdSingleStruct() {
    bool q = GetBits<30, 1>();
    bool postindex = GetBits<23, 1>();
    bool is_load = GetBits<22, 1>();
    bool r = GetBits<21, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode = GetBits<13, 3>();
    uint8_t s_bit = GetBits<12, 1>();
    uint8_t size = GetBits<10, 2>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    // Replicate loads: opcode=110 or 111 with L=1
    if (opcode >= 0b110 && is_load) {
      // LD1R/LD2R/LD3R/LD4R
      // opcode=110: 1-reg(R=0) or 2-reg(R=1)
      // opcode=111: 3-reg(R=0) or 4-reg(R=1)
      uint8_t num_regs;
      AdvSimdSingleStructOp op;
      if (opcode == 0b110 && !r) {
        num_regs = 1; op = AdvSimdSingleStructOp::kLd1r;
      } else if (opcode == 0b110 && r) {
        num_regs = 2; op = AdvSimdSingleStructOp::kLd2r;
      } else if (opcode == 0b111 && !r) {
        num_regs = 3; op = AdvSimdSingleStructOp::kLd3r;
      } else {
        num_regs = 4; op = AdvSimdSingleStructOp::kLd4r;
      }

      // Element size is encoded in 'size' field directly.
      // S must be 0 for replicate loads.
      if (s_bit) { Undefined(); return; }

      const AdvSimdSingleStructArgs args = {
          .op = op,
          .rt = rt,
          .rn = rn,
          .rm = rm,
          .size = size,
          .index = 0,
          .num_regs = num_regs,
          .q = q,
          .postindex = postindex,
          .is_replicate = true,
      };
      insn_consumer_->AdvSimdSingleStruct(args);
      return;
    }

    // Non-replicate LD/ST single element to one lane.
    // Decode element size and index from opcode/S/size/Q.
    uint8_t elem_size;
    uint8_t index;
    switch (opcode) {
      case 0b000:  // Byte
        elem_size = 0;  // B
        index = (q << 3) | (s_bit << 2) | size;
        break;
      case 0b010:  // Halfword
        if (size & 1) { Undefined(); return; }
        elem_size = 1;  // H
        index = (q << 2) | (s_bit << 1) | (size >> 1);
        break;
      case 0b100:  // Word or Doubleword
        if (size == 0b00) {
          elem_size = 2;  // S
          index = (q << 1) | s_bit;
        } else if (size == 0b01 && !s_bit) {
          elem_size = 3;  // D
          index = q;
        } else {
          Undefined(); return;
        }
        break;
      default:
        Undefined(); return;
    }

    // Determine number of registers from opcode and R bit.
    uint8_t num_regs;
    AdvSimdSingleStructOp op;
    if (!r) {
      switch (opcode) {
        case 0b000: case 0b010: case 0b100:
          num_regs = 1;
          op = is_load ? AdvSimdSingleStructOp::kLd1 : AdvSimdSingleStructOp::kSt1;
          break;
        default: Undefined(); return;
      }
    } else {
      switch (opcode) {
        case 0b000: case 0b010: case 0b100:
          num_regs = 2;
          op = is_load ? AdvSimdSingleStructOp::kLd2 : AdvSimdSingleStructOp::kSt2;
          break;
        default: Undefined(); return;
      }
    }

    const AdvSimdSingleStructArgs args = {
        .op = op,
        .rt = rt,
        .rn = rn,
        .rm = rm,
        .size = elem_size,
        .index = index,
        .num_regs = num_regs,
        .q = q,
        .postindex = postindex,
        .is_replicate = false,
    };
    insn_consumer_->AdvSimdSingleStruct(args);
  }
  // endregion

  void DecodeSimdLoadStorePair() {
    uint8_t opc = GetBits<30, 2>();
    uint8_t type = GetBits<23, 2>();
    bool is_load = GetBits<22, 1>();
    uint32_t imm7 = GetBits<15, 7>();
    uint8_t rt2 = GetBits<10, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt1 = GetBits<0, 5>();

    SimdLoadStoreSize ls_size;
    uint8_t scale;
    // opc determines size: 00=S(32), 01=D(64), 10=Q(128)
    switch (opc) {
      case 0b00: ls_size = SimdLoadStoreSize::k32bit; scale = 2; break;
      case 0b01: ls_size = SimdLoadStoreSize::k64bit; scale = 3; break;
      case 0b10: ls_size = SimdLoadStoreSize::k128bit; scale = 4; break;
      default: Undefined(); return;
    }

    int32_t offset = SignExtend<7>(imm7) << scale;

    bool is_preindex = (type == 0b11);
    bool is_postindex = (type == 0b01);

    const SimdLoadStorePairArgs args = {
        .rt1 = rt1,
        .rt2 = rt2,
        .rn = rn,
        .offset = offset,
        .size = ls_size,
        .is_store = !is_load,
        .is_preindex = is_preindex,
        .is_postindex = is_postindex,
    };
    insn_consumer_->SimdLoadStorePair(args);
  }

  //
  // FP <-> integer conversion (FMOV, SCVTF, UCVTF, FCVTZS, FCVTZU, etc.)
  //
  void DecodeFpIntConversion() {
    bool sf = GetBits<31, 1>();
    uint8_t ftype = GetBits<22, 2>();
    uint8_t rmode = GetBits<19, 2>();
    uint8_t opcode = GetBits<16, 3>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    const FpIntConvArgs args = {
        .rd = rd,
        .rn = rn,
        .opcode = opcode,
        .sf = sf,
        .ftype = ftype,
        .rmode = rmode,
        .op = opcode,
    };
    insn_consumer_->FpIntConversion(args);
  }

  // region digitalis
  // FMOV (scalar, immediate): Dd/Sd = VFPExpandImm(imm8)
  void DecodeFpMovImmediate() {
    uint8_t ftype = GetBits<22, 2>();
    uint8_t imm8 = GetBits<13, 8>();
    uint8_t rd = GetBits<0, 5>();
    insn_consumer_->FpMovImmediate(rd, imm8, ftype);
  }

  // FCSEL: Floating-point conditional select
  // Encoding: 0 0 0 11110 ftype 1 Rm cond 11 Rn Rd
  void DecodeFpCondSelect() {
    uint8_t ftype = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t cond = GetBits<12, 4>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    insn_consumer_->FpCondSelect(rd, rn, rm, ftype, static_cast<Condition>(cond));
  }

  // FP <-> fixed-point conversion: SCVTF, UCVTF, FCVTZS, FCVTZU (scalar, fixed-point)
  // Encoding: sf 0 S 11110 ftype 0 rmode opcode scale Rn Rd
  void DecodeFpFixedPointConversion() {
    bool sf = GetBits<31, 1>();
    uint8_t ftype = GetBits<22, 2>();
    uint8_t rmode = GetBits<19, 2>();
    uint8_t opcode = GetBits<16, 3>();
    uint8_t scale = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    uint8_t fbits = 64 - scale;

    FpFixedPointOp op;
    if (rmode == 0b00 && opcode == 0b010) {
      op = FpFixedPointOp::kScvtf;
    } else if (rmode == 0b00 && opcode == 0b011) {
      op = FpFixedPointOp::kUcvtf;
    } else if (rmode == 0b11 && opcode == 0b000) {
      op = FpFixedPointOp::kFcvtzs;
    } else if (rmode == 0b11 && opcode == 0b001) {
      op = FpFixedPointOp::kFcvtzu;
    } else {
      Undefined();
      return;
    }

    const FpFixedPointArgs args = {
        .rd = rd,
        .rn = rn,
        .op = op,
        .sf = sf,
        .ftype = ftype,
        .fbits = fbits,
    };
    insn_consumer_->FpFixedPointConversion(args);
  }

  // FP data-processing (3 source): FMADD, FMSUB, FNMADD, FNMSUB
  // Encoding: 0 0 0 11111 ftype o1 Rm o0 Ra Rn Rd
  void DecodeFpDataProc3() {
    uint8_t ftype = GetBits<22, 2>();
    bool o1 = GetBits<21, 1>();    // 0=FMADD/FMSUB, 1=FNMADD/FNMSUB
    uint8_t rm = GetBits<16, 5>();
    bool o0 = GetBits<15, 1>();    // 0=FMADD/FNMADD, 1=FMSUB/FNMSUB
    uint8_t ra = GetBits<10, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    insn_consumer_->FpDataProc3(rd, rn, rm, ra, ftype, o1, o0);
  }
  // endregion

  //
  // Stub decoders for SIMD/FP instruction groups (dispatch to Undefined for now,
  // will be implemented as needed).
  //
  void DecodeExtract() {
    bool sf = GetBits<31, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t imms = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    // EXTR Xd, Xn, Xm, #lsb: result = (Xn:Xm) >> lsb
    // For 32-bit: result = (Wn:Wm) >> lsb (lsb < 32)
    // When Rn == Rm, this is ROR.
    // Implement by passing to Bitfield with BFM opcode (won't work) or directly.
    // Simplest: pass as a new callback.
    // For now, use inline computation in the decoder:
    // Since the interpreter doesn't have a dedicated Extract, let's add inline.
    // Actually, let's just pass the args to a callback.
    struct ExtractArgs { uint8_t rd, rn, rm; uint8_t lsb; bool is_64bit; };
    // We can't add new callback without modifying all three files.
    // Simpler: compute the result inline in the interpreter via existing callbacks.
    // EXTR Xd, Xn, Xm, #lsb = concatenate Xn:Xm and extract bits[lsb+regsize-1:lsb]
    // For ROR (Rn==Rm): result = (Xn >> lsb) | (Xn << (regsize - lsb))
    // For general EXTR: result = (Xm >> lsb) | (Xn << (regsize - lsb))
    // Use BFM semantics: compute result directly
    // Since we need to pass the result, let's abuse MoveWide or use Bitfield:
    // Actually, EXTR with Rn==Rm is ROR, which Bitfield can express as UBFM.
    // For general EXTR, we need both Rn and Rm.
    // Simplest: implement inline by getting registers and setting result.
    // But we'd need the Interpreter to have access to two source registers.
    // Let's use the semantics player pattern: get two registers, compute, set.

    // For now, implement EXTR inline through BFM-like callback.
    // This is a hack but works: pass Rn through GetReg, Rm through GetReg,
    // compute result, set Rd.
    // Actually, let me just add a simple Extr callback.
    // I'll define it without a new args struct by using existing infrastructure.

    // Quick implementation: compute and emit via listener directly
    // The SemanticsPlayer will call listener_->Extr(...)
    insn_consumer_->Extr(rd, rn, rm, imms, sf);
  }

  // region digitalis
  // Data Processing (1-source): CLZ, CLS, RBIT, REV, REV16, REV32
  void DecodeDataProc1Src() {
    bool sf = GetBits<31, 1>();
    uint8_t opcode2 = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    // Dispatch based on opcode2
    // 000000 = RBIT, 000001 = REV16, 000010 = REV32(32-bit)/REV(64-bit),
    // 000011 = REV(64-bit only), 000100 = CLZ, 000101 = CLS
    insn_consumer_->DataProc1Src(rd, rn, opcode2, sf);
  }
  // endregion

  void DecodeConditionalCompare() {
    bool sf = GetBits<31, 1>();
    bool is_neg = !GetBits<30, 1>();  // bit30: 1=CCMP, 0=CCMN
    bool is_imm = GetBits<11, 1>();
    uint8_t rm_or_imm = GetBits<16, 5>();
    uint8_t cond = GetBits<12, 4>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t nzcv = GetBits<0, 4>();

    const ConditionalCompareArgs args = {
        .rn = rn,
        .rm_or_imm = rm_or_imm,
        .nzcv = nzcv,
        .cond = Condition{cond},
        .is_64bit = sf,
        .is_imm = is_imm,
        .is_neg = is_neg,
    };
    insn_consumer_->ConditionalCompare(args);
  }
  // endregion

  // region digitalis
  //
  // FP data-processing (2 source).
  // Encoding: M S 11110 ftype 1 Rm opcode 10 Rn Rd
  //
  void DecodeFpDataProc2() {
    uint8_t ftype = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode = GetBits<12, 4>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // Only ftype 00 (single) and 01 (double) supported.
    if (ftype >= 2) { Undefined(); return; }

    // Validate opcode range.
    if (opcode > 0b1000) { Undefined(); return; }

    const FpDataProc2Args args = {
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .ftype = ftype,
        .opcode = opcode,
    };
    insn_consumer_->FpDataProc2(args);
  }

  //
  // FP data-processing (1 source).
  // Encoding: M S 11110 ftype 1 opcode 10000 Rn Rd
  //
  void DecodeFpDataProc1() {
    uint8_t ftype = GetBits<22, 2>();
    uint8_t opcode = GetBits<15, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    const FpDataProc1Args args = {
        .rd = rd,
        .rn = rn,
        .ftype = ftype,
        .opcode = opcode,
    };
    insn_consumer_->FpDataProc1(args);
  }

  //
  // FP compare.
  // Encoding: M S 11110 ftype 1 Rm op 1000 Rn opcode2
  //   opcode2[0] = with_zero, opcode2[4] = signal_nans (FCMPE)
  //
  void DecodeFpCompare() {
    uint8_t ftype = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t opcode2 = GetBits<0, 5>();

    // Only ftype 00 (single) and 01 (double) supported.
    if (ftype >= 2) { Undefined(); return; }

    bool with_zero = (opcode2 & 0b01000) != 0;
    bool signal_nans = (opcode2 & 0b10000) != 0;

    const FpCompareArgs args = {
        .rn = rn,
        .rm = rm,
        .ftype = ftype,
        .with_zero = with_zero,
        .signal_nans = signal_nans,
    };
    insn_consumer_->FpCompare(args);
  }
  // endregion
  // region digitalis
  //
  // AdvSIMD three same.
  // Encoding: 0 Q U 01110 size 1 Rm opcode 1 Rn Rd
  //   Q = bit30, U = bit29
  //   size = bits[23:22], Rm = bits[20:16]
  //   opcode = bits[15:11], Rn = bits[9:5], Rd = bits[4:0]
  //
  void DecodeAdvSimdThreeSame() {
    bool q = GetBits<30, 1>();
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode = GetBits<11, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdThreeSameOpcode op;

    if (opcode == 0b00011) {
      // Logic group: operation selected by U and size.
      if (!u) {
        switch (size) {
          case 0b00: op = AdvSimdThreeSameOpcode::kAnd; break;
          case 0b01: op = AdvSimdThreeSameOpcode::kBic; break;
          case 0b10: op = AdvSimdThreeSameOpcode::kOrr; break;
          default:   op = AdvSimdThreeSameOpcode::kOrn; break;
        }
      } else {
        switch (size) {
          case 0b00: op = AdvSimdThreeSameOpcode::kEor; break;
          case 0b01: op = AdvSimdThreeSameOpcode::kBsl; break;
          case 0b10: op = AdvSimdThreeSameOpcode::kBit; break;
          default:   op = AdvSimdThreeSameOpcode::kBif; break;
        }
      }
    } else if (opcode == 0b10000) {
      op = u ? AdvSimdThreeSameOpcode::kSub : AdvSimdThreeSameOpcode::kAdd;
    } else if (opcode == 0b10001) {
      op = u ? AdvSimdThreeSameOpcode::kCmeq : AdvSimdThreeSameOpcode::kCmtst;
    // region digitalis
    } else if (opcode == 0b00110) {
      op = u ? AdvSimdThreeSameOpcode::kCmhi : AdvSimdThreeSameOpcode::kCmgt;
    } else if (opcode == 0b00111) {
      op = u ? AdvSimdThreeSameOpcode::kCmhs : AdvSimdThreeSameOpcode::kCmge;
    } else if (opcode == 0b01100) {
      op = u ? AdvSimdThreeSameOpcode::kUmax : AdvSimdThreeSameOpcode::kSmax;
    } else if (opcode == 0b01101) {
      op = u ? AdvSimdThreeSameOpcode::kUmin : AdvSimdThreeSameOpcode::kSmin;
    // endregion
    } else if (opcode == 0b00000) {
      op = u ? AdvSimdThreeSameOpcode::kUhadd : AdvSimdThreeSameOpcode::kShadd;
    } else if (opcode == 0b00001) {
      op = u ? AdvSimdThreeSameOpcode::kUqadd : AdvSimdThreeSameOpcode::kSqadd;
    } else if (opcode == 0b00010) {
      op = u ? AdvSimdThreeSameOpcode::kUrhadd : AdvSimdThreeSameOpcode::kSrhadd;
    } else if (opcode == 0b00100) {
      op = u ? AdvSimdThreeSameOpcode::kUhsub : AdvSimdThreeSameOpcode::kShsub;
    } else if (opcode == 0b00101) {
      op = u ? AdvSimdThreeSameOpcode::kUqsub : AdvSimdThreeSameOpcode::kSqsub;
    } else if (opcode == 0b01000) {
      op = u ? AdvSimdThreeSameOpcode::kUshl : AdvSimdThreeSameOpcode::kSshl;
    } else if (opcode == 0b01001) {
      op = u ? AdvSimdThreeSameOpcode::kUqshl : AdvSimdThreeSameOpcode::kSqshl;
    } else if (opcode == 0b01010) {
      op = u ? AdvSimdThreeSameOpcode::kUrshl : AdvSimdThreeSameOpcode::kSrshl;
    } else if (opcode == 0b01011) {
      op = u ? AdvSimdThreeSameOpcode::kUqrshl : AdvSimdThreeSameOpcode::kSqrshl;
    } else if (opcode == 0b10111) {
      if (u) { Undefined(); return; }
      op = AdvSimdThreeSameOpcode::kAddp;
    } else if (opcode == 0b10011) {
      if (u) { Undefined(); return; }
      op = AdvSimdThreeSameOpcode::kMul;
    } else if (opcode == 0b10010) {
      op = u ? AdvSimdThreeSameOpcode::kMls : AdvSimdThreeSameOpcode::kMla;
    // region digitalis
    } else if (opcode == 0b10100) {
      op = u ? AdvSimdThreeSameOpcode::kUmaxp : AdvSimdThreeSameOpcode::kSmaxp;
    } else if (opcode == 0b10101) {
      op = u ? AdvSimdThreeSameOpcode::kUminp : AdvSimdThreeSameOpcode::kSminp;
    } else if ((opcode & 0b11000) == 0b11000) {
      // FP three-same (vector). bits[23] = op_high, bits[22] = sz.
      // The 'size' field as read above is {op_high, sz} for this encoding;
      // split it out and pass only sz through args.size so the interpreter
      // can dispatch element width purely from args.size.
      bool op_high = (size >> 1) & 1;
      uint8_t sz = size & 1;
      bool ok = true;
      if (!op_high) {
        switch (opcode) {
          case 0b11010:
            if (u) { ok = false; break; }   // FADDP not implemented
            op = AdvSimdThreeSameOpcode::kFaddV;
            break;
          case 0b11011:
            if (!u) { ok = false; break; }  // FMULX not implemented
            op = AdvSimdThreeSameOpcode::kFmulV;
            break;
          case 0b11001:
            if (u) { ok = false; break; }   // U=1 reserved here
            op = AdvSimdThreeSameOpcode::kFmlaV;
            break;
          // region digitalis - FMAXNM/FMAX (op_high=0, U=0); FDIV (U=1);
          // FCMEQ (U=0)/FCMGE (U=1)/FACGE (U=1) at opcode 11100/11101.
          case 0b11000:
            if (u) { ok = false; break; }   // FMAXNMP — not implemented
            op = AdvSimdThreeSameOpcode::kFmaxnmV;
            break;
          case 0b11100:
            op = u ? AdvSimdThreeSameOpcode::kFcmgeV : AdvSimdThreeSameOpcode::kFcmeqV;
            break;
          case 0b11101:
            if (!u) { ok = false; break; }  // FRECPS — not implemented
            op = AdvSimdThreeSameOpcode::kFacgeV;
            break;
          case 0b11110:
            if (u) { ok = false; break; }   // FMAXP — not implemented
            op = AdvSimdThreeSameOpcode::kFmaxV;
            break;
          case 0b11111:
            if (!u) { ok = false; break; }  // FRECPS-low — not implemented
            op = AdvSimdThreeSameOpcode::kFdivV;
            break;
          // endregion
          default: ok = false; break;
        }
      } else {
        switch (opcode) {
          case 0b11010:
            if (u) { ok = false; break; }   // FABD not implemented
            op = AdvSimdThreeSameOpcode::kFsubV;
            break;
          case 0b11001:
            if (u) { ok = false; break; }   // U=1 reserved here
            op = AdvSimdThreeSameOpcode::kFmlsV;
            break;
          // region digitalis - FMINNM/FMIN (op_high=1, U=0); FCMGT (U=1)/FACGT (U=1).
          case 0b11000:
            if (u) { ok = false; break; }   // FMINNMP — not implemented
            op = AdvSimdThreeSameOpcode::kFminnmV;
            break;
          case 0b11100:
            if (!u) { ok = false; break; }  // op_high=1,U=0 undef at opcode 11100
            op = AdvSimdThreeSameOpcode::kFcmgtV;
            break;
          case 0b11101:
            if (!u) { ok = false; break; }  // FRSQRTS — not implemented
            op = AdvSimdThreeSameOpcode::kFacgtV;
            break;
          case 0b11110:
            if (u) { ok = false; break; }   // FMINP — not implemented
            op = AdvSimdThreeSameOpcode::kFminV;
            break;
          // endregion
          default: ok = false; break;
        }
      }
      if (!ok) {
        Undefined();
        return;
      }
      // sz=1 (double) requires Q=1.
      if (sz && !q) {
        Undefined();
        return;
      }
      // Pass sz as args.size (0 = single -> 4-byte elements,
      // 1 = double -> 8-byte elements). Interpreter dispatches on this.
      size = sz;
    // endregion
    } else {
      Undefined();
      return;
    }

    // size=11 (64-bit elements) requires Q=1 for most opcodes except logic ops.
    if (size == 0b11 && !q && opcode != 0b00011) {
      Undefined();
      return;
    }

    const AdvSimdThreeSameArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .size = size,
        .q = q,
    };
    insn_consumer_->AdvSimdThreeSame(args);
  }
  // endregion

  // region digitalis
  //
  // AdvSIMD three different (widening/narrowing).
  // Encoding: 0 Q U 01110 size 1 Rm opcode 00 Rn Rd
  //   Q = bit30, U = bit29
  //   size = bits[23:22], Rm = bits[20:16]
  //   opcode = bits[15:12], Rn = bits[9:5], Rd = bits[4:0]
  //
  void DecodeAdvSimdThreeDiff() {
    bool q = GetBits<30, 1>();
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode = GetBits<12, 4>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // size=11 is reserved for three-different.
    if (size == 0b11) {
      Undefined();
      return;
    }

    AdvSimdThreeDiffOpcode op;

    switch (opcode) {
      case 0b0000:
        op = u ? AdvSimdThreeDiffOpcode::kUaddl : AdvSimdThreeDiffOpcode::kSaddl;
        break;
      // region digitalis - wide add/sub variants (opcode 0001/0011)
      case 0b0001:
        op = u ? AdvSimdThreeDiffOpcode::kUaddw : AdvSimdThreeDiffOpcode::kSaddw;
        break;
      case 0b0011:
        op = u ? AdvSimdThreeDiffOpcode::kUsubw : AdvSimdThreeDiffOpcode::kSsubw;
        break;
      // endregion
      case 0b0010:
        op = u ? AdvSimdThreeDiffOpcode::kUsubl : AdvSimdThreeDiffOpcode::kSsubl;
        break;
      case 0b0101:
        op = u ? AdvSimdThreeDiffOpcode::kUabal : AdvSimdThreeDiffOpcode::kSabal;
        break;
      case 0b0111:
        op = u ? AdvSimdThreeDiffOpcode::kUabdl : AdvSimdThreeDiffOpcode::kSabdl;
        break;
      case 0b1000:
        op = u ? AdvSimdThreeDiffOpcode::kUmlal : AdvSimdThreeDiffOpcode::kSmlal;
        break;
      case 0b1010:
        op = u ? AdvSimdThreeDiffOpcode::kUmlsl : AdvSimdThreeDiffOpcode::kSmlsl;
        break;
      case 0b1100:
        op = u ? AdvSimdThreeDiffOpcode::kUmull : AdvSimdThreeDiffOpcode::kSmull;
        break;
      default:
        Undefined();
        return;
    }

    const AdvSimdThreeDiffArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .size = size,
        .q = q,
    };
    insn_consumer_->AdvSimdThreeDiff(args);
  }
  // endregion

  // region digitalis
  //
  // AdvSIMD two-reg misc.
  // Encoding: 0 Q U 01110 size 10000 opcode 10 Rn Rd
  //
  void DecodeAdvSimdTwoRegMisc() {
    bool q = GetBits<30, 1>();
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t opcode = GetBits<12, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdTwoRegMiscOpcode op;

    switch (opcode) {
      case 0b00000:
        op = u ? AdvSimdTwoRegMiscOpcode::kRev32 : AdvSimdTwoRegMiscOpcode::kRev64;
        break;
      case 0b00001:
        if (!u) { Undefined(); return; }  // opcode=00001 U=0 is unallocated
        op = AdvSimdTwoRegMiscOpcode::kRev16;
        break;
      case 0b00010:
        op = u ? AdvSimdTwoRegMiscOpcode::kUaddlp : AdvSimdTwoRegMiscOpcode::kSaddlp;
        break;
      // region digitalis - opcode 00011 covers two distinct families:
      //  bit20=0 (bits[21:17]=10000) -> two-reg-misc SUQADD (U=0) / USQADD (U=1)
      //  bit20=1 (bits[21:17]=11000) -> across-lanes  SADDLV (U=0) / UADDLV (U=1)
      // Observed `uaddlv h0, v0.8b` (insn 0x2e303800) in WhatsApp's
      // libar-bundle3.so JNI_OnLoad path.
      case 0b00011:
        if (GetBits<20, 1>()) {
          op = u ? AdvSimdTwoRegMiscOpcode::kUaddlv : AdvSimdTwoRegMiscOpcode::kSaddlv;
          if (size == 0b11) { Undefined(); return; }  // no 64-bit element
        } else {
          op = u ? AdvSimdTwoRegMiscOpcode::kUsqadd : AdvSimdTwoRegMiscOpcode::kSuqadd;
        }
        break;
      // endregion
      case 0b00100:
        op = u ? AdvSimdTwoRegMiscOpcode::kClz : AdvSimdTwoRegMiscOpcode::kCls;
        break;
      case 0b00101:
        if (!u) {
          op = AdvSimdTwoRegMiscOpcode::kCnt;  // U=0, size=00
        } else {
          op = AdvSimdTwoRegMiscOpcode::kNot;  // U=1, size=00 => NOT; size=01 => RBIT
        }
        break;
      case 0b00110:
        op = u ? AdvSimdTwoRegMiscOpcode::kUadalp : AdvSimdTwoRegMiscOpcode::kSadalp;
        break;
      case 0b01000:
        op = u ? AdvSimdTwoRegMiscOpcode::kCmgeZero : AdvSimdTwoRegMiscOpcode::kCmgtZero;
        break;
      case 0b01001:
        op = u ? AdvSimdTwoRegMiscOpcode::kCmleZero : AdvSimdTwoRegMiscOpcode::kCmeqZero;
        break;
      case 0b01010:
        // region digitalis - bit20 distinguishes across-lanes (1) from two-reg-misc (0).
        // Across-lanes opcode=01010 is SMAXV (U=0) / UMAXV (U=1).
        // Two-reg-misc opcode=01010 is CMLT zero (U=0 only; U=1 unallocated).
        if (GetBits<20, 1>()) {
          op = u ? AdvSimdTwoRegMiscOpcode::kUmaxv : AdvSimdTwoRegMiscOpcode::kSmaxv;
          // SMAXV/UMAXV are only defined for size 00/01/10 with Q matching, but the
          // interpreter validates size; reject 64-bit element which has no encoding.
          if (size == 0b11) { Undefined(); return; }
        } else {
          if (u) { Undefined(); return; }
          op = AdvSimdTwoRegMiscOpcode::kCmltZero;
        }
        break;
      case 0b11010:
        // Across-lanes opcode=11010 is SMINV (U=0) / UMINV (U=1).
        // bit20 must be 1; bit20=0 with this opcode is unallocated in two-reg-misc.
        if (!GetBits<20, 1>()) { Undefined(); return; }
        op = u ? AdvSimdTwoRegMiscOpcode::kUminv : AdvSimdTwoRegMiscOpcode::kSminv;
        if (size == 0b11) { Undefined(); return; }
        break;
        // endregion
      case 0b01011:
        op = u ? AdvSimdTwoRegMiscOpcode::kNeg : AdvSimdTwoRegMiscOpcode::kAbs;
        break;
      case 0b10010:
        if (u) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kXtn;
        break;
      case 0b10100:
        if (u) {
          op = AdvSimdTwoRegMiscOpcode::kUqxtn;
        } else {
          op = AdvSimdTwoRegMiscOpcode::kSqxtn;
        }
        break;
      case 0b10110:
        if (u) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kFcvtn;
        break;
      case 0b10111:
        if (u) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kFcvtl;
        break;
      case 0b01111:
        op = u ? AdvSimdTwoRegMiscOpcode::kFneg : AdvSimdTwoRegMiscOpcode::kFabs;
        break;
      // region digitalis - SCVTF/UCVTF (vector, integer): opcode=11101, bit23=0.
      // (bit23=1 with this opcode is FRECPE/FRSQRTE — not implemented here.)
      // Observed `ucvtf v0.4s, v0.4s` (insn 0x6e21d800) in WhatsApp's
      // libar-bundle3.so init path.
      case 0b11101:
        if (GetBits<23, 1>()) { Undefined(); return; }  // FRECPE/FRSQRTE - TODO
        op = u ? AdvSimdTwoRegMiscOpcode::kUcvtfV : AdvSimdTwoRegMiscOpcode::kScvtfV;
        break;
      // endregion
      // region digitalis
      case 0b11011:
        // ADDV is in the across-lanes group (bit20=1), not two-reg-misc (bit20=0).
        if (!GetBits<20, 1>()) { Undefined(); return; }
        if (u) { Undefined(); return; }  // ADDV is U=0 only
        if (size == 0b11) { Undefined(); return; }  // No 64-bit element ADDV
        op = AdvSimdTwoRegMiscOpcode::kAddv;
        break;
      // endregion
      default:
        Undefined();
        return;
    }

    const AdvSimdTwoRegMiscArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .size = size,
        .q = q,
        .u = u,
    };
    insn_consumer_->AdvSimdTwoRegMisc(args);
  }
  // endregion
  // region digitalis
  //
  // AdvSIMD scalar two-reg misc.
  // Encoding: 01 U 11110 size 10000 opcode 10 Rn Rd
  //
  void DecodeAdvSimdScalarTwoRegMisc() {
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t opcode = GetBits<12, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdScalarTwoRegMiscOpcode op;

    switch (opcode) {
      case 0b11101:
        // SCVTF (U=0) / UCVTF (U=1): integer → float, scalar
        // size: 0=single, 1=double
        if (size >= 2) { Undefined(); return; }
        op = u ? AdvSimdScalarTwoRegMiscOpcode::kUcvtf
               : AdvSimdScalarTwoRegMiscOpcode::kScvtf;
        break;
      case 0b11011:
        // FCVTZS (U=0) / FCVTZU (U=1): float → integer, round toward zero
        if (size >= 2) { Undefined(); return; }
        op = u ? AdvSimdScalarTwoRegMiscOpcode::kFcvtzu
               : AdvSimdScalarTwoRegMiscOpcode::kFcvtzs;
        break;
      default:
        Undefined();
        return;
    }

    const AdvSimdScalarTwoRegMiscArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .size = static_cast<uint8_t>(size & 1),  // 0=single, 1=double
    };
    insn_consumer_->AdvSimdScalarTwoRegMisc(args);
  }
  // endregion
  // region digitalis
  //
  // AdvSIMD scalar three same: ADD/SUB/CMxx/SSHL/USHL on scalar D-form.
  // Encoding: 01 U 11110 size 1 Rm opcode 1 Rn Rd
  //
  void DecodeAdvSimdScalarThreeSame() {
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode = GetBits<11, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdScalarThreeSameOpcode op;
    uint8_t out_size = size;
    bool is_fp = false;

    // FP scalar three-same opcodes live in the same encoding class but the
    // size field is interpreted as bit23=Fp-discriminator (1), bit22=sz.
    // Distinguish FP by opcode: 11010 (FABD), 11100 (FCMxx), 11101 (FAC..).
    if (opcode == 0b11010 || opcode == 0b11100 || opcode == 0b11101) {
      is_fp = true;
      bool bit23 = (size >> 1) & 1;
      uint8_t sz = size & 1;  // 0 -> S, 1 -> D
      switch (opcode) {
        case 0b11010:
          if (!u || !bit23) { Undefined(); return; }
          op = AdvSimdScalarThreeSameOpcode::kFabd;
          break;
        case 0b11100:
          if (u && bit23) op = AdvSimdScalarThreeSameOpcode::kFcmgt;
          else if (u && !bit23) op = AdvSimdScalarThreeSameOpcode::kFcmge;
          else if (!u && !bit23) op = AdvSimdScalarThreeSameOpcode::kFcmeq;
          else { Undefined(); return; }
          break;
        case 0b11101:
          if (u && bit23) op = AdvSimdScalarThreeSameOpcode::kFacgt;
          else if (u && !bit23) op = AdvSimdScalarThreeSameOpcode::kFacge;
          else { Undefined(); return; }
          break;
        default:
          Undefined();
          return;
      }
      out_size = sz;
    } else {
      // Integer scalar three same — D-form only.
      if (size != 0b11) { Undefined(); return; }

      switch (opcode) {
        case 0b00110:
          op = u ? AdvSimdScalarThreeSameOpcode::kCmhi
                 : AdvSimdScalarThreeSameOpcode::kCmgt;
          break;
        case 0b00111:
          op = u ? AdvSimdScalarThreeSameOpcode::kCmhs
                 : AdvSimdScalarThreeSameOpcode::kCmge;
          break;
        case 0b01000:
          op = u ? AdvSimdScalarThreeSameOpcode::kUshl
                 : AdvSimdScalarThreeSameOpcode::kSshl;
          break;
        case 0b10000:
          op = u ? AdvSimdScalarThreeSameOpcode::kSub
                 : AdvSimdScalarThreeSameOpcode::kAdd;
          break;
        case 0b10001:
          op = u ? AdvSimdScalarThreeSameOpcode::kCmeq
                 : AdvSimdScalarThreeSameOpcode::kCmtst;
          break;
        default:
          Undefined();
          return;
      }
    }
    (void)is_fp;

    const AdvSimdScalarThreeSameArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .size = out_size,
    };
    insn_consumer_->AdvSimdScalarThreeSame(args);
  }
  // endregion

  // region digitalis
  //
  // AdvSIMD scalar pairwise.
  // Encoding: 01 U 11110 size 11000 opcode 10 Rn Rd
  // For now only ADDP scalar (U=0, size=11, opcode=11011) is supported.
  //
  void DecodeAdvSimdScalarPairwise() {
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t opcode = GetBits<12, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    if (u || opcode != 0b11011 || size != 0b11) {
      Undefined();
      return;
    }

    const AdvSimdScalarPairwiseArgs args = {
        .opcode = AdvSimdScalarPairwiseOpcode::kAddp,
        .rd = rd,
        .rn = rn,
        .size = size,
    };
    insn_consumer_->AdvSimdScalarPairwise(args);
  }
  // endregion
  // region digitalis
  //
  // AdvSIMD scalar copy: DUP (scalar), aka MOV Vd, Vn[index].
  // Encoding: 0 1 0 11110 000 imm5 0 0000 1 Rn Rd
  // Element size derived from imm5 like the vector copy form:
  //   imm5[0]=1 -> B, imm5[1:0]=10 -> H, imm5[2:0]=100 -> S, imm5[3:0]=1000 -> D.
  // Result: copy Vn[index] (one esize-byte element) into bottom of Vd; upper bits zero.
  //
  void DecodeAdvSimdScalarCopy() {
    uint8_t imm5 = GetBits<16, 5>();
    uint8_t imm4 = GetBits<11, 4>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // Validate imm5: must encode a valid element size.
    if ((imm5 & 0xF) == 0) {
      Undefined();
      return;
    }

    // Reuse the vector-copy args plumbing with kDupScalar opcode.
    // q=false signals "scalar" semantics (zero-extend element to 128-bit).
    const AdvSimdCopyArgs args = {
        .opcode = AdvSimdCopyOpcode::kDupScalar,
        .rd = rd,
        .rn = rn,
        .imm5 = imm5,
        .imm4 = imm4,
        .q = false,
    };
    insn_consumer_->AdvSimdCopy(args);
  }
  // endregion
  // region digitalis
  //
  // AdvSIMD copy (DUP, INS, SMOV, UMOV).
  //
  // Encoding: 0 Q op 01110 000 imm5 0 imm4 1 Rn Rd
  //   bit31=0, bits[28:24]=01110, bits[23:21]=000, bit15=0, bit10=1
  //   Q = bit30, op = bit29
  //   imm5 = bits[20:16], imm4 = bits[14:11]
  //   Rn = bits[9:5], Rd = bits[4:0]
  //
  void DecodeAdvSimdCopy() {
    bool q = GetBits<30, 1>();
    uint8_t op = GetBits<29, 1>();
    uint8_t imm5 = GetBits<16, 5>();
    uint8_t imm4 = GetBits<11, 4>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdCopyOpcode opcode;

    if (op == 0) {
      // op=0: DUP (element), DUP (general), INS (general), SMOV, UMOV
      switch (imm4) {
        case 0b0000:
          // DUP (element): imm4=0000
          opcode = AdvSimdCopyOpcode::kDupElement;
          break;
        case 0b0001:
          // DUP (general): imm4=0001
          opcode = AdvSimdCopyOpcode::kDupGeneral;
          break;
        case 0b0011:
          // INS (general): op=0, imm4=0011, Q must be 1
          if (!q) { Undefined(); return; }
          opcode = AdvSimdCopyOpcode::kInsGeneral;
          break;
        case 0b0101:
          // SMOV: imm4=0101
          opcode = AdvSimdCopyOpcode::kSmov;
          break;
        case 0b0111:
          // UMOV: imm4=0111
          opcode = AdvSimdCopyOpcode::kUmov;
          break;
        default:
          Undefined();
          return;
      }
    } else {
      // op=1: INS (element) -- imm4 encodes source element index. Q must be 1.
      if (!q) { Undefined(); return; }
      opcode = AdvSimdCopyOpcode::kInsElement;
    }

    // Validate imm5: must have at least one bit set in [3:0] to encode a valid element size.
    if ((imm5 & 0xF) == 0) {
      Undefined();
      return;
    }

    const AdvSimdCopyArgs args = {
        .opcode = opcode,
        .rd = rd,
        .rn = rn,
        .imm5 = imm5,
        .imm4 = imm4,
        .q = q,
    };
    insn_consumer_->AdvSimdCopy(args);
  }
  // endregion
  // region digitalis
  //
  // AdvSIMD vector x indexed element.
  // Encoding: 0 Q U 01111 size L M Rm opcode H 0 Rn Rd
  //
  void DecodeAdvSimdVecXIndexedElement() {
    bool q = GetBits<30, 1>();
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t L = GetBits<21, 1>();
    uint8_t M = GetBits<20, 1>();
    uint8_t Rm4 = GetBits<16, 4>();
    uint8_t opcode = GetBits<12, 4>();
    uint8_t H = GetBits<11, 1>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    uint8_t rm;
    uint8_t index;

    if (size == 0b10) {
      // 32-bit: Vm = M:Rm, index = H:L
      rm = (M << 4) | Rm4;
      index = (H << 1) | L;
    } else if (size == 0b11) {
      // 64-bit: Vm = M:Rm, index = H
      rm = (M << 4) | Rm4;
      index = H;
    } else {
      // 16-bit or reserved.
      Undefined();
      return;
    }

    AdvSimdVecXIdxOpcode op;
    switch (opcode) {
      case 0b0001:
        if (u) { Undefined(); return; }
        op = AdvSimdVecXIdxOpcode::kFmla;
        break;
      case 0b0101:
        if (u) { Undefined(); return; }
        op = AdvSimdVecXIdxOpcode::kFmls;
        break;
      case 0b1001:
        op = u ? AdvSimdVecXIdxOpcode::kFmul : AdvSimdVecXIdxOpcode::kFmul;
        break;
      case 0b1000:
        if (u) {
          op = AdvSimdVecXIdxOpcode::kMla;
        } else {
          op = AdvSimdVecXIdxOpcode::kMul;
        }
        break;
      case 0b0100:
        if (u) {
          op = AdvSimdVecXIdxOpcode::kMls;
        } else {
          Undefined(); return;
        }
        break;
      case 0b0000:
        if (u) {
          op = AdvSimdVecXIdxOpcode::kMla;
        } else {
          Undefined(); return;
        }
        break;
      default:
        Undefined();
        return;
    }

    const AdvSimdVecXIdxArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .index = index,
        .size = size,
        .q = q,
    };
    insn_consumer_->AdvSimdVecXIndexedElement(args);
  }
  // endregion

  // region digitalis
  //
  // AdvSIMD shift by immediate.
  // Encoding: 0 Q U 011110 immh:immb opcode 1 Rn Rd
  //   immh = bits[22:19], immb = bits[18:16]
  //   opcode = bits[15:11]
  //
  void DecodeAdvSimdShiftByImm() {
    bool q = GetBits<30, 1>();
    bool u = GetBits<29, 1>();
    uint8_t immh = GetBits<19, 4>();
    uint8_t immb = GetBits<16, 3>();
    uint8_t opcode = GetBits<11, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // immh=0000 is reserved (encodes AdvSIMD modified immediate instead).
    if (immh == 0) { Undefined(); return; }

    AdvSimdShiftImmOpcode op;

    switch (opcode) {
      case 0b00000:
        op = u ? AdvSimdShiftImmOpcode::kUshr : AdvSimdShiftImmOpcode::kSshr;
        break;
      case 0b00010:
        op = u ? AdvSimdShiftImmOpcode::kUsra : AdvSimdShiftImmOpcode::kSsra;
        break;
      case 0b00100:
        op = u ? AdvSimdShiftImmOpcode::kUrshr : AdvSimdShiftImmOpcode::kSrshr;
        break;
      case 0b00110:
        op = u ? AdvSimdShiftImmOpcode::kUrsra : AdvSimdShiftImmOpcode::kSrsra;
        break;
      case 0b01000:
        if (u) {
          op = AdvSimdShiftImmOpcode::kSri;
        } else {
          Undefined(); return;
        }
        break;
      case 0b01010:
        if (u) {
          op = AdvSimdShiftImmOpcode::kSli;
        } else {
          op = AdvSimdShiftImmOpcode::kShl;
        }
        break;
      case 0b01110:
        if (u) {
          op = AdvSimdShiftImmOpcode::kSqshlu;
        } else {
          op = AdvSimdShiftImmOpcode::kSqshl;
        }
        break;
      case 0b01100:
        op = u ? AdvSimdShiftImmOpcode::kUqshl : AdvSimdShiftImmOpcode::kSqshl;
        break;
      case 0b10000:
        if (!u) {
          op = AdvSimdShiftImmOpcode::kShrn;
        } else {
          op = AdvSimdShiftImmOpcode::kSqshrn;
        }
        break;
      case 0b10001:
        if (!u) {
          op = AdvSimdShiftImmOpcode::kRshrn;
        } else {
          op = AdvSimdShiftImmOpcode::kUqshrn;
        }
        break;
      case 0b10100:
        op = u ? AdvSimdShiftImmOpcode::kUshll : AdvSimdShiftImmOpcode::kSshll;
        break;
      default:
        Undefined();
        return;
    }

    const AdvSimdShiftImmArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .immh = immh,
        .immb = immb,
        .q = q,
        .u = u,
    };
    insn_consumer_->AdvSimdShiftByImm(args);
  }
  // endregion

  //
  // Load/store exclusive, ordered, and CAS.
  // bit29=0, op_28_27=01, op_26=0.
  //
  void DecodeLoadStoreExclusive() {
    uint8_t size = GetBits<30, 2>();
    uint8_t o2 = GetBits<23, 1>();
    uint8_t L = GetBits<22, 1>();
    uint8_t o1 = GetBits<21, 1>();
    uint8_t rs = GetBits<16, 5>();
    uint8_t o0 = GetBits<15, 1>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    // region digitalis
    // o2 distinguishes exclusive (o2=0) from ordered/CAS (o2=1):
    //   o2=0, o1=0: STXR/LDXR/STLXR/LDAXR (exclusive)
    //   o2=0, o1=1: CASP (exclusive pair CAS)
    //   o2=1, o1=0: STLR/LDAR (ordered, non-exclusive)
    //   o2=1, o1=1: CAS/CASA/CASL/CASAL
    // endregion

    LoadStoreExclusiveArgs args;
    args.rt = rt;
    args.rn = rn;
    args.rs = rs;
    args.size = size;
    args.acquire = false;
    args.release = false;

    if (o1 == 0) {
      // region digitalis - check o2 to distinguish STLR/LDAR from STXR/LDXR
      if (o2) {
        // o2=1, o1=0: LDAR/STLR (ordered, non-exclusive)
        if (L) {
          args.op = AtomicOp::kLdar;
          args.acquire = true;
        } else {
          args.op = AtomicOp::kStlr;
          args.release = true;
        }
      } else {
        // o2=0, o1=0: LDXR/STXR family (exclusive)
        if (L) {
          args.op = AtomicOp::kLdxr;
          args.acquire = (o0 != 0);  // LDAXR
        } else {
          args.op = AtomicOp::kStxr;
          args.release = (o0 != 0);  // STLXR
        }
      }
      // endregion
    } else {
      // o1=1: CAS family
      args.op = AtomicOp::kCas;
      args.acquire = (L != 0);   // CASA/CASAL
      args.release = (o0 != 0);  // CASL/CASAL
    }
    insn_consumer_->LoadStoreExclusive(args);
  }

  //
  // Atomic memory operations: SWP, LDADD, LDCLR, LDSET, LDEOR, etc.
  // op_28_27=11, op_26=0, bit24=0, bit21=1.
  //
  void DecodeAtomicMemoryOp() {
    uint8_t size = GetBits<30, 2>();
    uint8_t A = GetBits<23, 1>();     // Acquire
    uint8_t R = GetBits<22, 1>();     // Release
    uint8_t rs = GetBits<16, 5>();
    uint8_t o3 = GetBits<15, 1>();    // Distinguishes SWP (o3=1) from LD* (o3=0)
    uint8_t opc = GetBits<12, 3>();   // Operation type within group
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    LoadStoreExclusiveArgs args;
    args.rt = rt;
    args.rn = rn;
    args.rs = rs;
    args.size = size;
    args.acquire = (A != 0);
    args.release = (R != 0);

    // ARM64 atomic memory ops: o3 + opc determine the operation.
    // o3=0: LDADD/LDCLR/LDSET/LDEOR (opc selects which)
    // o3=1, opc=000: SWP
    uint8_t full_op = (o3 << 3) | opc;
    switch (full_op) {
      case 0b0000: args.op = AtomicOp::kLdadd; break;
      case 0b0001: args.op = AtomicOp::kLdclr; break;
      case 0b0010: args.op = AtomicOp::kLdset; break;
      case 0b0011: args.op = AtomicOp::kLdeor; break;
      case 0b1000: args.op = AtomicOp::kSwp; break;
      default: Undefined(); return;
    }
    insn_consumer_->LoadStoreExclusive(args);
  }
  // endregion

  //
  // Data Processing - Register.
  //
  void DecodeDataProcessingRegister() {
    // ARM64 Data Processing (Register) encoding.
    // Top-level: bits[28:25] = x101 (already dispatched here).
    // Sub-groups determined by bit[28], bit[24], bit[21].
    bool op1_high = GetBits<28, 1>();  // bit 28
    bool op2_high = GetBits<24, 1>();  // bit 24
    bool op2_low = GetBits<21, 1>();   // bit 21

    if (!op1_high) {
      // bit[28] = 0
      if (!op2_high) {
        // bit[28]=0, bit[24]=0: Logical (shifted register).
        DecodeLogicalShiftedReg();
        return;
      }
      // bit[28]=0, bit[24]=1: Add/sub (shifted or extended register).
      if (!op2_low) {
        // bit[21]=0: Add/sub (shifted register).
        DecodeAddSubShiftedReg();
      } else {
        // bit[21]=1: Add/sub (extended register).
        DecodeAddSubExtendedReg();
      }
      return;
    }

    // bit[28] = 1
    if (!op2_high) {
      // bit[28]=1, bit[24]=0
      // region digitalis
      uint8_t op2_bits = GetBits<21, 3>();  // bits [23:21]

      // // uint8_t op2_bits = GetBits<21, 4>();  // bits [24:21]
      if (op2_bits == 0b000) {
        // region digitalis
        // Add/sub with carry: ADC, ADCS, SBC, SBCS
        {
          bool sf = GetBits<31, 1>();
          bool op = GetBits<30, 1>();   // 0=ADC, 1=SBC
          bool s = GetBits<29, 1>();    // set flags
          uint8_t rm = GetBits<16, 5>();
          uint8_t rn = GetBits<5, 5>();
          uint8_t rd = GetBits<0, 5>();
          insn_consumer_->AddSubWithCarry(rd, rn, rm, sf, op, s);
        }
        // endregion
        return;
      }
      // region digitalis
      if (op2_bits == 0b010) {
        // Conditional compare (register/immediate).
        DecodeConditionalCompare();
        return;
      }
      // if ((op2_bits & 0b0110) == 0b0100) {
      if (op2_bits == 0b100) {
        // Conditional select.
        DecodeConditionalSelect();
        return;
      }
      if (op2_bits == 0b110) {
        // region digitalis
        // Distinguish 2-source (bit30=0) from 1-source (bit30=1)
        if (GetBits<30, 1>()) {
          // Data processing (1-source): CLZ, CLS, RBIT, REV, REV16, REV32
          DecodeDataProc1Src();
        } else {
          // Data processing (2-source): UDIV, SDIV, LSLV, LSRV, ASRV, RORV
          DecodeDataProc2Src();
        }
        // endregion
        return;
      }
      // endregion
      Undefined();
      return;
    }

    // region digitalis
    // bit[28]=1, bit[24]=1: always Data processing (3-source).
    // DataProc2Src has bits[28:24]=11010 (bit24=0), dispatched above.
    // // bit[21]=0: Data processing (2-source).
    // // bit[21]=1: Data processing (3-source).
    DecodeDataProc3Src();
    // endregion
  }

  void DecodeLogicalShiftedReg() {
    bool sf = GetBits<31, 1>();
    uint8_t opc = GetBits<29, 2>();
    uint8_t shift = GetBits<22, 2>();
    bool n = GetBits<21, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t imm6 = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // For 32-bit, imm6 must be < 32.
    if (!sf && (imm6 >= 32)) {
      return Undefined();
    }

    const LogicalShiftedRegArgs args = {
        .opcode = LogicalShiftedRegOpcode{opc},
        .dst = rd,
        .src1 = rn,
        .src2 = rm,
        .shift_type = ShiftType{shift},
        .shift_amount = imm6,
        .is_64bit = sf,
        .invert = n,
    };
    insn_consumer_->LogicalShiftedReg(args);
  }

  void DecodeAddSubShiftedReg() {
    bool sf = GetBits<31, 1>();
    bool is_sub = GetBits<30, 1>();
    bool set_flags = GetBits<29, 1>();
    uint8_t shift = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t imm6 = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // shift == 11 is reserved.
    if (shift == 0b11) {
      return Undefined();
    }

    // For 32-bit, imm6 must be < 32.
    if (!sf && (imm6 >= 32)) {
      return Undefined();
    }

    const AddSubShiftedRegArgs args = {
        .dst = rd,
        .src1 = rn,
        .src2 = rm,
        .shift_type = ShiftType{shift},
        .shift_amount = imm6,
        .is_64bit = sf,
        .is_sub = is_sub,
        .set_flags = set_flags,
    };
    insn_consumer_->AddSubShiftedReg(args);
  }

  void DecodeAddSubExtendedReg() {
    bool sf = GetBits<31, 1>();
    bool is_sub = GetBits<30, 1>();
    bool set_flags = GetBits<29, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t option = GetBits<13, 3>();
    uint8_t imm3 = GetBits<10, 3>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // imm3 must be <= 4.
    if (imm3 > 4) {
      return Undefined();
    }

    const AddSubExtendedRegArgs args = {
        .dst = rd,
        .src1 = rn,
        .src2 = rm,
        .extend_type = option,
        .shift_amount = imm3,
        .is_64bit = sf,
        .is_sub = is_sub,
        .set_flags = set_flags,
    };
    insn_consumer_->AddSubExtendedReg(args);
  }

  void DecodeConditionalSelect() {
    bool sf = GetBits<31, 1>();
    bool op = GetBits<30, 1>();
    bool s = GetBits<29, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t cond = GetBits<12, 4>();
    uint8_t op2 = GetBits<10, 2>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // S must be 0.
    if (s) {
      return Undefined();
    }

    // op2 bit 1 must be 0.
    if (op2 & 0b10) {
      return Undefined();
    }

    uint8_t opcode = static_cast<uint8_t>((op << 1) | (op2 & 0b01));

    const ConditionalSelectArgs args = {
        .opcode = ConditionalSelectOpcode{opcode},
        .dst = rd,
        .src1 = rn,
        .src2 = rm,
        .cond = Condition{cond},
        .is_64bit = sf,
    };
    insn_consumer_->ConditionalSelect(args);
  }

  void DecodeDataProc2Src() {
    bool sf = GetBits<31, 1>();
    bool s = GetBits<29, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // S must be 0 for these instructions.
    if (s) {
      return Undefined();
    }

    const DataProc2SrcArgs args = {
        .opcode = DataProc2SrcOpcode{opcode},
        .dst = rd,
        .src1 = rn,
        .src2 = rm,
        .is_64bit = sf,
    };
    insn_consumer_->DataProc2Src(args);
  }

  void DecodeDataProc3Src() {
    bool sf = GetBits<31, 1>();
    uint8_t op31 = GetBits<21, 3>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t o0 = GetBits<15, 1>();
    uint8_t ra = GetBits<10, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // region digitalis
    // Encode opcode from op31[1:0] and o0.
    // ARM64 DataProc3Src: op31[2] is the unsigned flag (U), op31[1:0]+o0 select the operation.
    // Map: MADD=000, MSUB=001, SMADDL=010, SMSUBL=011, SMULH=100,
    //       UMADDL=101, UMSUBL=110, UMULH=111
    // For unsigned variants (op31[2]=1), add 5 to the signed equivalent.
    uint8_t op_low = (op31 & 3);  // bits[1:0] of op31
    bool is_unsigned = (op31 & 4) != 0;  // bit[2] of op31
    uint8_t opcode;
    if (!is_unsigned) {
      opcode = static_cast<uint8_t>((op_low << 1) | o0);
    } else {
      // Unsigned: UMADDL(5), UMSUBL(6), UMULH(7)
      opcode = static_cast<uint8_t>(5 + (op_low << 1) + o0 - 2);
      // Actually: op31=101,o0=0 -> UMADDL=5; op31=101,o0=1 -> UMSUBL=6
      //           op31=110,o0=0 -> UMULH=7
      // Simpler: UMADDL = op_low=01,o0=0 -> (1<<1)|0 + 4 = 6? No...
      // Let me just map directly:
      opcode = static_cast<uint8_t>(((op31 & 3) << 1) | o0);
      if (is_unsigned) opcode += 4; // shift unsigned by 4 (SMADDL=2 -> UMADDL=6?)
      // Hmm this doesn't match the enum either.
    }
    // Actually the simplest fix: just use the original formula but cap at 7
    opcode = static_cast<uint8_t>((op31 << 1) | o0);
    // Map the ARM64 encoding to our enum:
    // op31=000,o0=0 -> 0 (kMadd) ✓
    // op31=000,o0=1 -> 1 (kMsub) ✓
    // op31=001,o0=0 -> 2 (kSmaddl) ✓
    // op31=001,o0=1 -> 3 (kSmsubl) ✓
    // op31=010,o0=0 -> 4 (kSmulh) ✓
    // op31=101,o0=0 -> 10 ← PROBLEM! Should be kUmaddl=5
    // op31=101,o0=1 -> 11 ← Should be kUmsubl=6
    // op31=110,o0=0 -> 12 ← Should be kUmulh=7
    // Fix: if op31 >= 4, remap
    if (op31 >= 4) {
      opcode = static_cast<uint8_t>(5 + ((op31 & 3) << 1) + o0 - 2);
      // op31=101: 5 + (1<<1) + 0 - 2 = 5 (kUmaddl) ✓
      // op31=101: 5 + (1<<1) + 1 - 2 = 6 (kUmsubl) ✓
      // op31=110: 5 + (2<<1) + 0 - 2 = 7 (kUmulh) ✓
    }
    // endregion

    const DataProc3SrcArgs args = {
        .opcode = DataProc3SrcOpcode{opcode},
        .dst = rd,
        .src1 = rn,
        .src2 = rm,
        .src3 = ra,
        .is_64bit = sf,
    };
    insn_consumer_->DataProc3Src(args);
  }

  uint32_t code_;
  InsnConsumer* insn_consumer_;
};

}  // namespace berberis

#endif  // BERBERIS_DECODER_ARM64_DECODER_H_
// endregion
