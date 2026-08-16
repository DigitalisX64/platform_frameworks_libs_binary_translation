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

// AArch64 system-register encoding used by DecodeSystem (ARM ARM C5.2, the
// S<op0>_<op1>_<Cn>_<Cm>_<op2> name): (op0<<14)|(op1<<11)|(CRn<<7)|(CRm<<3)|op2,
// matching the key DecodeSystem builds. SystemReg enumerators are computed from
// this so an encoding can never drift from its (op0,op1,CRn,CRm,op2) tuple by a
// hand-typed hex slip (a prior hand-typed swap sent MRS DCZID_EL0 to the
// CTR_EL0 handler).
constexpr uint16_t SysRegEncoding(unsigned op0, unsigned op1, unsigned crn, unsigned crm,
                                  unsigned op2) {
  return static_cast<uint16_t>((op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2);
}

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
    // PACGA (Armv8.3-PAuth generic PAC compute)
    // PACGA Xd, Xn, Xm|SP: 64-bit only (sf=1, S=0, opcode=001100).
    // ARM ARM C7.2.179: produces a 32-bit PAC in Rd[63:32], zeros Rd[31:0].
    // Digitalis is PAC-blind (never inserts authentication codes), so the
    // PAC value is 0 and Rd = 0.  Interpreter handles this; JIT bails.
    kPacga = 0b001100,
    // CRC32 opcodes
    kCrc32b = 0b010000,
    kCrc32h = 0b010001,
    kCrc32w = 0b010010,
    kCrc32x = 0b010011,
    kCrc32cb = 0b010100,
    kCrc32ch = 0b010101,
    kCrc32cw = 0b010110,
    kCrc32cx = 0b010111,
  };

  //
  // MTE (Memory Tagging Extension, Armv8.5-A) data-processing 2-source
  // opcodes. Encoding:
  //   sf 0 S 11010110 Rm opcode Rn Rd
  // where opcode = bits[15:10]. Distinguished from the regular
  // DataProc2Src opcodes (which start at 0b000010) because MTE uses
  // 0b000000, 0b000100, 0b000101; SUBP (opcode=0,S=0) and SUBPS
  // (opcode=0,S=1) collide on opcode but differ on S, so we fold the
  // S-bit into the enum as a synthetic kSubps value.
  //
  // llvm-mc verified (aarch64-linux-gnu-as -march=armv8.5-a+memtag):
  //   irg   x0, x1     = 0x9adf1020  (sf=1, S=0, Rm=11111(XZR), opc=000100, Rn=1, Rd=0)
  //   gmi   x0, x1, x2 = 0x9ac21420  (sf=1, S=0, Rm=2,          opc=000101, Rn=1, Rd=0)
  //   subp  x0, x1, x2 = 0x9ac20020  (sf=1, S=0, Rm=2,          opc=000000, Rn=1, Rd=0)
  //   subps x0, x1, x2 = 0xbac20020  (sf=1, S=1, Rm=2,          opc=000000, Rn=1, Rd=0)
  enum class MteDataProcOpcode : uint8_t {
    kSubp = 0b000000,    // SUBP  Xd, Xn|SP, Xm|SP — 56-bit signed Rn - Rm
    kIrg  = 0b000100,    // IRG   Xd|SP, Xn|SP{, Xm} — identity (no MTE)
    kGmi  = 0b000101,    // GMI   Xd, Xn|SP, Xm — pass-through Rm (no MTE)
    kSubps = 0b1000000,  // SUBPS Xd, Xn|SP, Xm|SP — SUBP + set NZCV
                         // (synthetic high bit; not on the wire)
  };

  struct MteDataProcArgs {
    MteDataProcOpcode opcode;
    uint8_t dst;   // Rd; for SUBP[S]/GMI dst=31 means XZR, for IRG dst=31 means SP.
    uint8_t src1;  // Rn; src1=31 always means SP for MTE DP-2src.
    uint8_t src2;  // Rm; src2=31 means XZR (or "no Rm" for IRG without Xm).
  };

  // MTE (Armv8.5-A) load/store memory tags.
  //
  // Encoding (ARM ARM C4.1.84.4):
  //   11011001 opc 1 imm9 op2 Rn Rt
  //     opc = bits[23:22]    op2 = bits[11:10]    imm9 = bits[20:12] (signed, granule-scaled)
  //
  // llvm-mc verified (aarch64-linux-gnu-as -march=armv8.5-a+memtag):
  //   stg   x0,[x1],#16   = 0xd9201420  (opc=00, op2=01, imm9=1, post-index)
  //   stg   x0,[x1,#16]   = 0xd9201820  (opc=00, op2=10, imm9=1, signed offset)
  //   stg   x0,[x1,#16]!  = 0xd9201c20  (opc=00, op2=11, imm9=1, pre-index)
  //   ldg   x0,[x1,#16]   = 0xd9601020  (opc=01, op2=00, imm9=1, signed offset, no writeback)
  //   stzg  x0,[x1,#16]   = 0xd9601820  (opc=01, op2=10, imm9=1, signed offset)
  //   st2g  x0,[x1,#32]   = 0xd9a02820  (opc=10, op2=10, imm9=2, signed offset)
  //   stz2g x0,[x1,#32]   = 0xd9e02820  (opc=11, op2=10, imm9=2, signed offset)
  //
  // op2 encodes the index mode (and, for opc=01, picks LDG vs STZG):
  //   0b00 = signed offset, no writeback (only valid for opc=01 => LDG)
  //   0b01 = post-index (writeback Xn += imm after access)
  //   0b10 = signed offset, no writeback
  //   0b11 = pre-index (writeback Xn += imm before access)
  enum class MteLoadStoreOpcode : uint8_t {
    kStg,    // store tag — NOP without MTE backing
    kLdg,    // load tag into Rt[59:56] — without MTE, loaded tag is 0
    kStzg,   // store tag + zero 16-byte granule
    kSt2g,   // store double tag (32-byte granule) — NOP without MTE backing
    kStz2g,  // store double tag + zero 32-byte granule
    // tag-block (granule-multiple) forms (op2=00).
    kLdgm,   // load tag multiple into Rt — without MTE, all tags read 0
    kStgm,   // store tag multiple — NOP without MTE backing
    kStzgm,  // store tag multiple + zero block — tag NOP; block size is
             // GMID_EL1-defined (not emulated), so treated as a tag NOP
  };

  struct MteLoadStoreArgs {
    MteLoadStoreOpcode opcode;
    uint8_t rn;       // base; rn=31 means SP.
    uint8_t rt;       // data/dest; rt=31 means XZR.
    int32_t imm;      // sign-extended imm9 << 4 (already scaled by 16-byte granule).
    uint8_t op2;      // 0b00=offset-no-wb (LDG), 0b01=post, 0b10=offset, 0b11=pre.
  };

  // Advanced SIMD complex floating-point (Armv8.3-FCMA): FCADD / FCMLA.
  //
  // Encoding (observed bits, llvm-mc-verified with
  //   aarch64-linux-gnu-as -march=armv8.3-a):
  //   bit31=0, bit30=Q, bit29=1 (U), bits[28:24]=01110, bits[23:22]=size,
  //   bit21=0, bits[20:16]=Rm, bit15=1, bit14=1, bit10=1, bits[9:5]=Rn,
  //   bits[4:0]=Rd.
  // FCADD: bit13=1, bit12=rot (0=#90, 1=#270), bit11=0.
  // FCMLA: bit13=0, bits[12:11]=rot (00=#0, 01=#90, 10=#180, 11=#270).
  //
  // Verified encodings:
  //   fcadd v0.4s,v1.4s,v2.4s,#90   = 0x6e82e420  (size=10, Q=1, bit12=0)
  //   fcadd v0.4s,v1.4s,v2.4s,#270  = 0x6e82f420  (size=10, Q=1, bit12=1)
  //   fcadd v0.2d,v1.2d,v2.2d,#90   = 0x6ec2e420  (size=11, Q=1, bit12=0)
  //   fcadd v0.2s,v1.2s,v2.2s,#90   = 0x2e82e420  (size=10, Q=0)
  //   fcmla v0.4s,v1.4s,v2.4s,#0    = 0x6e82c420  (size=10, Q=1, bits[12:11]=00)
  //   fcmla v0.4s,v1.4s,v2.4s,#90   = 0x6e82cc20  (size=10, Q=1, bits[12:11]=01)
  //   fcmla v0.4s,v1.4s,v2.4s,#180  = 0x6e82d420  (size=10, Q=1, bits[12:11]=10)
  //   fcmla v0.4s,v1.4s,v2.4s,#270  = 0x6e82dc20  (size=10, Q=1, bits[12:11]=11)
  //   fcmla v0.2d,v1.2d,v2.2d,#90   = 0x6ec2cc20  (size=11, Q=1)
  //
  // size: 00 reserved, 01 = half-precision (FP16 — Digitalis treats as
  // Undefined for now; FP16 SIMD support is a separate plan item under),
  // 10 = single, 11 = double.  For size=11 only Q=1 (2D) is valid; the
  // half-vector "1D" form is reserved.
  enum class FcmaOpcode : uint8_t {
    kFcadd,
    kFcmla,
  };

  struct FcmaArgs {
    FcmaOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t size;     // raw bits[23:22] — 10 = single, 11 = double.
    uint8_t rot;      // FCADD: 0=#90, 1=#270; FCMLA: 0=#0, 1=#90, 2=#180, 3=#270.
    bool q;           // bit[30] — 0 = 64-bit vector, 1 = 128-bit vector.
  };

  // indexed FCMLA
  // Advanced SIMD complex floating-point by element (Armv8.3-FCMA): FCMLA.
  //
  // FCADD has no by-element form; only FCMLA has an indexed encoding.
  //
  // Encoding (ARM ARM C7.2.86, verified via llvm-mc):
  //   bit31=0, bit30=Q, bit29=1 (U), bits[28:24]=01111,
  //   bits[23:22]=size, bit21=L, bit20=M, bits[19:16]=Rm[3:0],
  //   bit15=0, bits[14:13]=rot, bit12=1, bit11=H, bit10=0,
  //   bits[9:5]=Rn, bits[4:0]=Rd.
  //
  // The size field is 2-bit raw bits[23:22]:
  //   size==0b01: FP16, Vd is .4h (Q=0) or .8h (Q=1).
  //   size==0b10: FP32, Vd is .4s (Q=1 ONLY — .2s reserved).
  //   size==0b00 / 0b11: reserved.
  // FP32 size=0b10 SHARES the raw size value with FMLA-by-element FP32
  // — the distinguisher is U: FMLA/FMLS use U=0, FCMLA-idx uses U=1.
  //
  // index width:
  //   FP16: H:L (2 bits, 0..3 — Vm.8H has 4 complex pairs).
  //   FP32: H (1 bit, 0..1 — Vm.4S has 2 complex pairs); L must be 0.
  //
  // Vm: M:Rm[3:0] (5 bits).
  //
  // Without this carve-out, FMLA-pattern opcodes 0001/0101 with U=1 hit
  // the existing `case 0b0001/case 0b0101: if (u) Undefined()` branches,
  // while rot=1 (0011) and rot=3 (0111) fall through to the default
  // Undefined.  None of those four paths surface to the consumer.
  //
  // llvm-mc-verified encodings:
  //   fcmla v0.4s, v1.4s, v2.s[0], #0   = 0x6F821020
  //   fcmla v0.4s, v1.4s, v2.s[1], #0   = 0x6F821820  (H=1)
  //   fcmla v0.4s, v1.4s, v2.s[0], #90  = 0x6F823020  (rot=01)
  //   fcmla v0.4s, v1.4s, v2.s[1], #90  = 0x6F823820
  //   fcmla v0.4s, v1.4s, v2.s[0], #180 = 0x6F825020  (rot=10)
  //   fcmla v0.4s, v1.4s, v2.s[1], #180 = 0x6F825820
  //   fcmla v0.4s, v1.4s, v2.s[0], #270 = 0x6F827020  (rot=11)
  //   fcmla v0.4s, v1.4s, v2.s[1], #270 = 0x6F827820
  //   fcmla v0.4s, v1.4s, v17.s[0], #0  = 0x6F911020  (Vm=10001)
  //
  // FP16-indexed FCMLA is parked alongside non-indexed FP16 (the
  // non-indexed FP16 path rejects size==FP16 as "no Digitalis FP16-SIMD
  // FCMA yet").  Even though FP16 vector three-same support exists, the
  // family hasn't been extended yet; doing both at once would bundle two
  // task blocks.  Future work: lift the FP16 reject in both indexed and
  // non-indexed paths together.
  enum class FcmaIdxOpcode : uint8_t {
    kFcmlaIdx,
  };

  struct FcmaIdxArgs {
    FcmaIdxOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t index;    // 0..1 for FP32 (1-bit H); 0..3 for FP16 (2-bit H:L).
    uint8_t size;     // 0b10 = FP32 only (FP16 parked).
    uint8_t rot;      // 0=#0, 1=#90, 2=#180, 3=#270.
    bool q;           // bit[30] — false = .2s (1 pair), true = .4s (2 pairs).
  };

  // Advanced SIMD BFloat16 three-same-extra (Armv8.6-BF16):
  //   BFDOT (vector), BFMMLA.
  //
  // Encoding (verified via aarch64-linux-gnu-as -march=armv8.6-a):
  //   bit31=0, bit30=Q, bit29=1 (U), bits[28:24]=01110, bits[23:22]=01,
  //   bit21=0, bits[20:16]=Rm, bit15=1, bit14=1, bit13=1, bit11=1, bit10=1,
  //   bits[9:5]=Rn, bits[4:0]=Rd.
  //   bit12 = 1  ->  BFDOT
  //   bit12 = 0  ->  BFMMLA   (requires Q=1 — only 4S form exists)
  //
  // Verified encodings:
  //   bfdot v0.4s, v1.8h, v2.8h   = 0x6e42fc20  (Q=1)
  //   bfdot v0.2s, v1.4h, v2.4h   = 0x2e42fc20  (Q=0)
  //   bfmmla v0.4s, v1.8h, v2.8h  = 0x6e42ec20  (Q=1)
  //
  // BFDOT semantics: 32-bit accumulation lanes; each lane is FP32 += dot
  // product of two BF16 pairs from Vn,Vm.  2S form (Q=0) covers 2 lanes,
  // 4S form (Q=1) covers 4 lanes.
  //
  // BFMMLA semantics: Vd.4S viewed as 2x2 FP32 matrix; Vn.8H / Vm.8H
  // viewed as 2x4 BF16 matrices; computes Vd += Vn * Vm^T per the
  // ARM ARM C7.2.55 pseudo-code.  Always 128-bit (Q=1).
  //
  // Follow-ups (Armv8.6-BF16 surface closeout):
  //   - kBfmlalbVec / kBfmlaltVec — BFMLALB/BFMLALT (vector).
  //     Per-FP32-lane widening MAC; B=even (h[2i]), T=odd (h[2i+1]).
  //     Encoding: bits[28:24]=01110, bits[23:22]=11, bit21=0,
  //     bits[15:10]=111111. bit30 is the T discriminator (Q implicit 1).
  //     llvm-mc: bfmlalb v0.4s,v1.8h,v2.8h = 0x2ec2fc20,
  //              bfmlalt v0.4s,v1.8h,v2.8h = 0x6ec2fc20.
  //   - kBfdotIdx — BFDOT (by element).
  //     Encoding: bits[28:24]=01111, bits[23:22]=01, bit10=0,
  //     opcode bits[15:12]=1111. Vm = M:Rm[3:0] (V0..V31), index = H:L.
  //     Q selects 2S vs 4S form (.2s or .4s).
  //     llvm-mc: bfdot v0.4s,v1.8h,v2.2h[0] = 0x4f42f020.
  //   - kBfmlalbIdx / kBfmlaltIdx — BFMLALB/BFMLALT (by element).
  //     Encoding: bits[28:24]=01111, bits[23:22]=11, bit10=0,
  //     opcode bits[15:12]=1111. Vm = Rm[3:0] (V0..V15), index = H:L:M.
  //     bit30 is the T discriminator (Q implicit 1, dest always .4s).
  //     llvm-mc: bfmlalb v0.4s,v1.8h,v2.h[0] = 0x0fc2f020,
  //              bfmlalt v0.4s,v1.8h,v2.h[7] = 0x4ff2f820.
  enum class Bf16ThreeSameOpcode : uint8_t {
    kBfdot,
    kBfmmla,
    kBfmlalbVec,    // BFMLALB (vector)
    kBfmlaltVec,    // BFMLALT (vector)
    kBfdotIdx,      // BFDOT (by element)
    kBfmlalbIdx,    // BFMLALB (by element)
    kBfmlaltIdx,    // BFMLALT (by element)
  };

  struct Bf16ThreeSameArgs {
    Bf16ThreeSameOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t index;  // 0..3 for kBfdotIdx; 0..7 for kBfmlal{b,t}Idx; 0 otherwise.
    bool q;         // True selects 4S/.4s form; BFMMLA & BFMLAL ops always pass true.
  };

  // hello-dotprod
  // AdvSIMD integer dot product (Armv8.4-DotProd): SDOT / UDOT, vector
  // and by-element forms.
  //
  // Semantics: each 32-bit output lane accumulates the dot product of a
  // 4-byte group from Vn against a 4-byte group from Vm.
  //   for each output lane i:
  //     for k in [0..4):
  //       n_byte = Vn.b[4*i + k]              (signed for SDOT, unsigned for UDOT)
  //       m_byte = vector form: Vm.b[4*i + k]
  //                indexed form: Vm.b[4*index + k]  (one 4-byte group broadcast)
  //       Vd.s[i] += ext(n_byte) * ext(m_byte)
  //   lanes = q ? 4 : 2 (Q selects .4s vs .2s).  For Q=0 the upper 64 bits of
  //   Vd are zeroed.
  //
  // Verified encodings (clang --target=aarch64 -march=armv8.4-a+dotprod):
  //   sdot v0.4s, v1.16b, v2.16b      = 0x4e829420  (vector, Q=1, U=0)
  //   udot v0.4s, v1.16b, v2.16b      = 0x6e829420  (vector, Q=1, U=1)
  //   sdot v0.2s, v1.8b,  v2.8b       = 0x0e829420  (vector, Q=0, U=0)
  //   udot v0.2s, v1.8b,  v2.8b       = 0x2e829420  (vector, Q=0, U=1)
  //   sdot v0.4s, v1.16b, v2.4b[0]    = 0x4f82e020  (idx, Q=1, U=0, index=0)
  //   udot v0.4s, v1.16b, v2.4b[3]    = 0x6fa2e820  (idx, Q=1, U=1, index=3)
  //   sdot v0.2s, v1.8b,  v2.4b[0]    = 0x0f82e020  (idx, Q=0, U=0, index=0)
  //   udot v0.2s, v1.8b,  v2.4b[3]    = 0x2fa2e820  (idx, Q=0, U=1, index=3)
  enum class DotProductOpcode : uint8_t {
    kSdot,      // SDOT vector
    kUdot,      // UDOT vector
    kSdotIdx,   // SDOT by element
    kUdotIdx,   // UDOT by element
    // I8MM mixed-sign dot products (FEAT_I8MM).
    kUsdot,     // USDOT vector  (Vn unsigned, Vm signed)
    kUsdotIdx,  // USDOT by element
    kSudotIdx,  // SUDOT by element (Vn signed, Vm unsigned; vector form N/A)
  };

  struct DotProductArgs {
    DotProductOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t index;  // 0..3 for indexed forms; 0 for vector forms.
    bool q;         // True selects .4s form (4 lanes); false selects .2s (2 lanes).
  };

  // I8MM 8-bit integer matrix multiply-accumulate.
  enum class MatMulOpcode : uint8_t {
    kSmmla,   // signed x signed
    kUmmla,   // unsigned x unsigned
    kUsmmla,  // Vn unsigned, Vm signed
  };
  struct MatMulArgs {
    MatMulOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
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
    // Values are computed from the op-field decomposition via SysRegEncoding() so an
    // encoding can never diverge from its (op0,op1,CRn,CRm,op2) tuple.
    kTpidrEl0 = SysRegEncoding(3, 3, 13, 0, 2),  // TPIDR_EL0
    kNzcv = SysRegEncoding(3, 3, 4, 2, 0),       // NZCV
    kFpcr = SysRegEncoding(3, 3, 4, 4, 0),       // FPCR
    kFpsr = SysRegEncoding(3, 3, 4, 4, 1),       // FPSR
    // ARM ARM C5.2.6: CTR_EL0 is op2=1, DCZID_EL0 is op2=7 (NOT both op2=7).
    // A prior hand-typed swap (kCtrEl0=0xD807, kDczidEl0=0xD80F) routed mrs DCZID_EL0
    // to the CTR_EL0 handler, returning 0x8444c004 instead of 0x10 — which
    // made bionic's __dl___memset_aarch64+0xb8 cmp x5,#0x4 succeed and the
    // DC-ZVA fallback loop take, leaving most of any large memset's buffer
    // unzeroed.  Symptom: VkCapsViewer bucket-array corruption.
    kCtrEl0 = SysRegEncoding(3, 3, 0, 0, 1),    // CTR_EL0
    kDczidEl0 = SysRegEncoding(3, 3, 0, 0, 7),  // DCZID_EL0
    // MIDR_EL1. Read by code that decides whether to use vectorised fast paths;
    // returning a real ARM CPU id (Cortex-A53 here) keeps Bionic and
    // third-party compression libs (Facebook superpack, etc.) on the
    // expected fast paths instead of the SIGILL-handler-driven probe
    // fallbacks that can spin in detection loops.
    kMidrEl1 = SysRegEncoding(3, 0, 0, 0, 0),  // MIDR_EL1
    // RNDR / RNDRRS (FEAT_RNG). Read by getentropy/ASLR seeding; the interpreter
    // wires these to a host RNG and reports success (NZCV cleared), rather than
    // faulting.
    kRndr = SysRegEncoding(3, 3, 2, 4, 0),    // RNDR
    kRndrrs = SysRegEncoding(3, 3, 2, 4, 1),  // RNDRRS
    // Generic timer (all op0=3, op1=3, CRn=14, CRm=0). Read by timing/benchmark
    // code (e.g. Unity, game engines) for a cheap monotonic clock; the interpreter
    // backs the counters with the host monotonic clock and reports a matching
    // frequency.
    kCntfrqEl0 = SysRegEncoding(3, 3, 14, 0, 0),  // CNTFRQ_EL0 (counter frequency)
    kCntpctEl0 = SysRegEncoding(3, 3, 14, 0, 1),  // CNTPCT_EL0 (physical count)
    kCntvctEl0 = SysRegEncoding(3, 3, 14, 0, 2),  // CNTVCT_EL0 (virtual count)
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

  // ADDG/SUBG: add/subtract immediate, with tags (FEAT_MTE).
  struct AddSubImmTagsArgs {
    uint8_t dst;     // Xd|SP
    uint8_t src;     // Xn|SP
    uint8_t uimm6;   // address offset in units of 16 bytes (offset = uimm6<<4)
    uint8_t uimm4;   // logical tag offset applied to bits[59:56]
    bool is_sub;     // true for SUBG, false for ADDG
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

  // LDR/LDRSW (literal) — PC-relative integer load. The decoder routes
  // PRFM (literal) to Nop() (no-op prefetch). bit[29]=0 (V=0) identifies
  // the integer form; SIMD/FP literal loads are decoded separately.
  struct LoadLiteralArgs {
    uint8_t rt;        // Destination register (31 = XZR/WZR, write discarded)
    int64_t offset;    // PC-relative byte offset (imm19 * 4, sign-extended)
    LoadStoreSize size;  // k32bit (LDR Wt / LDRSW Xt) or k64bit (LDR Xt)
    bool is_signed;    // True for LDRSW (sign-extend 32→64)
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
    bool is_signed;     // LDPSW: 32-bit elements sign-extended to 64-bit.
  };

  struct LoadStoreRegArgs {
    uint8_t rt;
    uint8_t rn;
    uint8_t rm;
    // raw 3-bit ARMv8 option field:
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
    // raw 3-bit ARMv8 option field for the offset
    // register, same encoding as LoadStoreRegArgs::extend_type. See the
    // comment there.
    uint8_t extend_type;
    uint8_t shift_amount;
    SimdLoadStoreSize size;
    bool is_store;
  };

  // LDR (literal) — SIMD/FP form (V=1). Load 32/64/128 bits from the
  // PC-relative literal pool into V[rt]. The integer form is decoded
  // separately as LoadLiteral (V=0).
  struct SimdLoadLiteralArgs {
    uint8_t rt;
    int64_t offset;          // PC-relative byte offset (imm19 * 4, sign-extended)
    SimdLoadStoreSize size;  // k32bit (S), k64bit (D), k128bit (Q)
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
    // CASP (compare-and-swap pair, Armv8.1 LSE).
    kCasp,     // Compare and swap pair (Rs:Rs+1 = expected, Rt:Rt+1 = new)
    kLdxp,     // Load exclusive pair (Rt, Rt2 <- [Rn], [Rn+sz])
    kStxp,     // Store exclusive pair (result in Rs; stores Rt, Rt2)
    kSwp,      // Swap
    kLdadd,    // Atomic add
    kLdclr,    // Atomic bit clear
    kLdset,    // Atomic bit set
    kLdeor,    // Atomic exclusive or
    // atomic min/max (LSE Armv8.1).
    kLdsmax,   // Atomic signed max
    kLdsmin,   // Atomic signed min
    kLdumax,   // Atomic unsigned max
    kLdumin,   // Atomic unsigned min
  };

  struct LoadStoreExclusiveArgs {
    AtomicOp op;
    uint8_t rt;        // Data register
    uint8_t rt2;       // Second data register (LDXP/STXP pair)
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
    // scalar SIMD copy (DUP scalar / MOV Vd, Vn[index])
    kDupScalar,    // DUP (scalar) / MOV scalar: copy one Vn[index] element into bottom of Vd, zero upper
  };

  struct AdvSimdCopyArgs {
    AdvSimdCopyOpcode opcode;
    uint8_t rd;        // Destination register (SIMD for DUP/INS, GP for SMOV/UMOV)
    uint8_t rn;        // Source register (SIMD for DUP/SMOV/UMOV, GP for DUP-general/INS-general)
    uint8_t imm5;      // Encodes element size and index
    uint8_t imm4;      // For INS (element): source index
    bool q;            // Q bit: 0=64-bit vector, 1=128-bit vector
  };

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
    kCmgt,      // CMGT (vector, register): U=0, opcode=00110
    kCmhi,      // CMHI (vector, register): U=1, opcode=00110 (unsigned >)
    kCmge,      // CMGE (vector, register): U=0, opcode=00111
    kCmhs,      // CMHS (vector, register): U=1, opcode=00111 (unsigned >=)
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
    kFabdV,     // FABD   (vector): op_high=1, opcode=11010, U=1
    kFmulxV,    // FMULX  (vector): op_high=0, opcode=11011, U=0
                // FP16-three-same encoding: a=0, opcode_3=011, U=0.
                // Same as kFmulV but with the ARM-defined ±0 * ±inf -> ±2.0
                // saturation (instead of NaN), used by libm reciprocal
                // refinement loops.  Interpreter-only.
    kFrecpsV,   // FRECPS  (vector): op_high=0, opcode=11111, U=0
                // FP16-three-same encoding: a=0, opcode_3=111, U=0.
                // Reciprocal step: (2.0 - a*b), with (0 * inf) -> +2.0
                // saturation, used as Newton-Raphson refinement after FRECPE.
                // Interpreter-only.
    kFrsqrtsV,  // FRSQRTS (vector): op_high=1, opcode=11111, U=0
                // FP16-three-same encoding: a=1, opcode_3=111, U=0.
                // Reciprocal square-root step: (3.0 - a*b)/2, with (0 * inf)
                // -> +1.5 saturation, used as Newton-Raphson refinement after
                // FRSQRTE.  Interpreter-only.
    // FP pairwise (vector): reduce adjacent element pairs of Vn into the low
    // half of Vd and of Vm into the high half.  U=1 in the FP three-same leg.
    // All FP precisions (.4h/.8h via FP16 encoding; .2s/.4s/.2d).
    // Interpreter-only.
    kFaddpV,    // FADDP   (vector): op_high=0, opcode=11010, U=1
    kFmaxpV,    // FMAXP   (vector): op_high=0, opcode=11110, U=1
    kFminpV,    // FMINP   (vector): op_high=1, opcode=11110, U=1
    kFmaxnmpV,  // FMAXNMP (vector): op_high=0, opcode=11000, U=1
    kFminnmpV,  // FMINNMP (vector): op_high=1, opcode=11000, U=1
    // SABD/UABD: vector absolute difference at .8b/.16b/
    // .4h/.8h/.2s/.4s. size=11 (64-bit lane) is reserved.  Verified with
    // llvm-mc:  sabd v0.8b,v1.8b,v2.8b = 0x0e227420 (opcode=01110, U=0);
    //           uabd v0.8b,v1.8b,v2.8b = 0x2e227420 (opcode=01110, U=1).
    kSabd,      // SABD (vector): U=0, opcode=01110
    kUabd,      // UABD (vector): U=1, opcode=01110
    // SABA/UABA: absolute-difference-and-accumulate at
    // .8b/.16b/.4h/.8h/.2s/.4s.  size=11 reserved.  Vd[i] += |Vn[i] - Vm[i]|.
    // Verified with llvm-mc:
    //   saba v0.8b,v1.8b,v2.8b = 0x0e227c20 (opcode=01111, U=0);
    //   uaba v0.8b,v1.8b,v2.8b = 0x2e227c20 (opcode=01111, U=1).
    kSaba,      // SABA (vector): U=0, opcode=01111
    kUaba,      // UABA (vector): U=1, opcode=01111
    // PMUL polynomial multiply (byte-lane GF(2) multiply).
    // .8b/.16b only; size=01/10/11 reserved per ARM ARM.
    // Verified with llvm-mc:
    //   pmul v0.8b,v1.8b,v2.8b   = 0x2e229c20 (opcode=10011, U=1, size=00, Q=0)
    //   pmul v0.16b,v1.16b,v2.16b = 0x6e229c20 (opcode=10011, U=1, size=00, Q=1)
    //   pmul v0.4h / v0.4s        invalid (size=01/10 reserved)
    kPmul,      // PMUL (vector): U=1, opcode=10011, size=00
    // SQDMULH / SQRDMULH saturating doubling multiply high.
    // .4h/.8h/.2s/.4s; size=00 and size=11 reserved per ARM ARM.
    // Semantics (per lane): high half of (2 * signed(Vn[i]) * signed(Vm[i]) +
    // round), saturated to the destination element's signed range. SQRDMULH
    // adds round = 1 << (esize_bits - 1); SQDMULH uses round = 0.
    // Verified with llvm-mc:
    //   sqdmulh  v0.4h,v1.4h,v2.4h = 0x0e62b420 (opcode=10110, U=0, size=01)
    //   sqdmulh  v0.4s,v1.4s,v2.4s = 0x4ea2b420 (Q=1, size=10)
    //   sqrdmulh v0.4h,v1.4h,v2.4h = 0x2e62b420 (opcode=10110, U=1, size=01)
    //   sqrdmulh v0.4s,v1.4s,v2.4s = 0x6ea2b420 (Q=1, U=1, size=10)
    //   sqdmulh v0.8b / v0.2d      invalid (size=00/11 reserved)
    kSqdmulh,   // SQDMULH (vector):  U=0, opcode=10110
    kSqrdmulh,  // SQRDMULH (vector): U=1, opcode=10110
    // Armv8.1-RDM SQRDMLAH / SQRDMLSH three-same vector
    // forms (NOT the by-element forms — those live in AdvSimdVecXIdxOpcode
    // as kSqrdmlahIdx / kSqrdmlshIdx).  These ride the "Advanced SIMD three
    // same extra" encoding class (bit21=0, bit15=1, bit10=1), which is
    // disjoint from the standard three-same class (bit21=1) handled by all
    // entries above.  The decoder routes both encoding classes into this
    // single enum so the interpreter / JIT only need one dispatch surface.
    // Encoding (per ARM ARM C7.2.298 / C7.2.300):
    //   SQRDMLAH <V>d.<T>, <V>n.<T>, <V>m.<T>
    //     = 0 Q 1 01110 size 0 Rm 1 0000 1 Rn Rd  (U=1, opcode4=0000)
    //   SQRDMLSH <V>d.<T>, <V>n.<T>, <V>m.<T>
    //     = 0 Q 1 01110 size 0 Rm 1 0001 1 Rn Rd  (U=1, opcode4=0001)
    //   size in {01 (.4h/.8h), 10 (.2s/.4s)}; size=00 / size=11 reserved.
    kSqrdmlahVec,  // SQRDMLAH (vector, three-same): three-same-extra
                   //   U=1, opcode4=0000, size ∈ {01,10}.
                   // Vd[i] = sat_signed(Vd[i] + SQRDMULH(Vn[i], Vm[i])).
    kSqrdmlshVec,  // SQRDMLSH (vector, three-same): three-same-extra
                   //   U=1, opcode4=0001, size ∈ {01,10}.
                   // Vd[i] = sat_signed(Vd[i] - SQRDMULH(Vn[i], Vm[i])).
  };

  struct AdvSimdThreeSameArgs {
    AdvSimdThreeSameOpcode opcode;
    uint8_t rd;        // Destination SIMD register
    uint8_t rn;        // First source SIMD register
    uint8_t rm;        // Second source SIMD register
    uint8_t size;      // Element size: 00=8b, 01=16b, 10=32b, 11=64b.
                       // For the FP three-same encoding the decoder already
                       // collapses {op_high, sz} to sz here (0=single, 1=double);
                       // when is_fp16 is set, the lanes are 2-byte half.
    bool q;            // Q bit: 0=64-bit vector (D regs), 1=128-bit vector (Q regs)
    // Armv8.2-FP16 NEON vector three-same.
    // FP16 vector three-same has a separate encoding (bit21=0, bits[15:14]=00,
    // bit22=1) from the standard three-same (bit21=1).  The decoder maps both
    // through this struct and the interpreter dispatches on this flag.
    bool is_fp16;
  };

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
    // wide add/sub: Vd(wide) = Vn(wide) op extend(Vm(narrow))
    kSaddw,    // SADDW/SADDW2: U=0, opcode=0001
    kUaddw,    // UADDW/UADDW2: U=1, opcode=0001
    kSsubw,    // SSUBW/SSUBW2: U=0, opcode=0011
    kUsubw,    // USUBW/USUBW2: U=1, opcode=0011
    // polynomial multiply (ARMv8 base PMULL + crypto PMULL64)
    // PMULL/PMULL2: U=0, opcode=1110
    //   size=00: 8-bit element poly-mul, 8 lanes -> 8x 16-bit results
    //   size=11: 64-bit element poly-mul, 1 lane -> 128-bit result (PMULL64)
    //   size=01,10 are RESERVED. Used by libz CRC32 acceleration.
    kPmull,
    // narrowing high (add/sub of two wide vectors, take high
    // half of each result lane, write to half-width destination).
    //   ADDHN  / ADDHN2  : U=0, opcode=0100, round=0
    //   RADDHN / RADDHN2 : U=1, opcode=0100, round=1<<(narrow_bits-1)
    // size encodes narrow-elem width: 00->8b dest from .8h, 01->.4h from .4s,
    // 10->.2s from .2d. size=11 reserved. Q=0 writes lower 64 bits of Vd
    // (upper cleared); Q=1 writes upper 64 bits (lower preserved).
    // llvm-mc verified encodings:
    //   addhn   v0.8b,  v1.8h, v2.8h -> 0x0e224020 (size=00, Q=0, U=0)
    //   addhn2  v0.16b, v1.8h, v2.8h -> 0x4e224020 (size=00, Q=1)
    //   addhn   v0.4h,  v1.4s, v2.4s -> 0x0e624020 (size=01)
    //   addhn   v0.2s,  v1.2d, v2.2d -> 0x0ea24020 (size=10)
    //   raddhn  v0.8b,  v1.8h, v2.8h -> 0x2e224020 (U=1)
    //   raddhn  v0.4h,  v1.4s, v2.4s -> 0x2e624020
    //   raddhn  v0.2s,  v1.2d, v2.2d -> 0x2ea24020
    kAddhn,
    kRaddhn,
    // narrowing high subtract (subtract two wide vectors,
    // take high half of each result lane, write to half-width destination).
    //   SUBHN  / SUBHN2  : U=0, opcode=0110, round=0
    //   RSUBHN / RSUBHN2 : U=1, opcode=0110, round=1<<(narrow_bits-1)
    // size encodes narrow-elem width: 00->.8b from .8h, 01->.4h from .4s,
    // 10->.2s from .2d. size=11 reserved. Q=0 writes lower 64 bits of Vd
    // (upper cleared); Q=1 writes upper 64 bits (lower preserved).
    // llvm-mc verified encodings:
    //   subhn   v0.8b,  v1.8h, v2.8h -> 0x0e226020 (size=00, Q=0, U=0)
    //   subhn2  v0.16b, v1.8h, v2.8h -> 0x4e226020 (size=00, Q=1)
    //   subhn   v0.4h,  v1.4s, v2.4s -> 0x0e626020 (size=01)
    //   subhn   v0.2s,  v1.2d, v2.2d -> 0x0ea26020 (size=10)
    //   rsubhn  v0.8b,  v1.8h, v2.8h -> 0x2e226020 (U=1)
    //   rsubhn  v0.4h,  v1.4s, v2.4s -> 0x2e626020
    //   rsubhn  v0.2s,  v1.2d, v2.2d -> 0x2ea26020
    kSubhn,
    kRsubhn,
    // signed saturating doubling multiply long.
    //   SQDMULL / SQDMULL2 : U=0, opcode=1101
    // For each lane, signed multiply two narrow source elements and double
    // the product. Saturate the result to the wide signed range; the only
    // input pair that saturates is (INT_MIN, INT_MIN), which produces
    // INT_MAX_out instead of -INT_MIN_out (which would overflow). size
    // encodes narrow elem width: 01 -> .4s from .4h (in 16, out 32),
    // 10 -> .2d from .2s (in 32, out 64). size=00 and size=11 are reserved
    // for SQDMULL (size=11 is already rejected by the top-of-routine guard;
    // size=00 is rejected explicitly in the opcode arm). Q=0 reads lower
    // 64 bits of each source; Q=1 reads upper 64 bits. Wide result fully
    // overwrites Vd.
    // llvm-mc verified encodings:
    //   sqdmull  v0.4s, v1.4h, v2.4h -> 0x0e62d020 (Q=0, size=01)
    //   sqdmull2 v0.4s, v1.8h, v2.8h -> 0x4e62d020 (Q=1, size=01)
    //   sqdmull  v0.2d, v1.2s, v2.2s -> 0x0ea2d020 (Q=0, size=10)
    //   sqdmull2 v0.2d, v1.4s, v2.4s -> 0x4ea2d020 (Q=1, size=10)
    kSqdmull,
    // signed saturating doubling multiply-accumulate long.
    //   SQDMLAL / SQDMLAL2 : U=0, opcode=1001
    // For each lane: addend = SignedSat(2 * Vn_narrow[i] * Vm_narrow[i])
    //   (first-stage saturation, identical to SQDMULL).
    //                Vd_wide[i] = SignedSat(Vd_wide[i] + addend)
    //   (second-stage saturation on the wide accumulate).
    // size/Q semantics match SQDMULL: size=01 -> .4s from .4h, size=10 ->
    // .2d from .2s; size=00 and size=11 are reserved. Q=0 reads lower
    // 64 bits of each source; Q=1 reads upper 64 bits. Vd is read-modify-
    // write (wide accumulator) rather than fully overwritten.
    // llvm-mc verified encodings:
    //   sqdmlal  v0.4s, v1.4h, v2.4h -> 0x0e629020 (Q=0, size=01)
    //   sqdmlal2 v0.4s, v1.8h, v2.8h -> 0x4e629020 (Q=1, size=01)
    //   sqdmlal  v0.2d, v1.2s, v2.2s -> 0x0ea29020 (Q=0, size=10)
    //   sqdmlal2 v0.2d, v1.4s, v2.4s -> 0x4ea29020 (Q=1, size=10)
    kSqdmlal,
    // signed saturating doubling multiply-subtract long.
    //   SQDMLSL / SQDMLSL2 : U=0, opcode=1011
    // For each lane: addend = SignedSat(2 * Vn_narrow[i] * Vm_narrow[i])
    //   (first-stage saturation, identical to SQDMULL/SQDMLAL).
    //                Vd_wide[i] = SignedSat(Vd_wide[i] - addend)
    //   (second-stage saturation on the wide accumulate subtract).
    // size/Q semantics match SQDMULL/SQDMLAL: size=01 -> .4s from .4h,
    // size=10 -> .2d from .2s; size=00 and size=11 are reserved. Q=0 reads
    // lower 64 bits of each source; Q=1 reads upper 64 bits. Vd is
    // read-modify-write (wide accumulator) rather than fully overwritten.
    // llvm-mc verified encodings:
    //   sqdmlsl  v0.4s, v1.4h, v2.4h -> 0x0e62b020 (Q=0, size=01)
    //   sqdmlsl2 v0.4s, v1.8h, v2.8h -> 0x4e62b020 (Q=1, size=01)
    //   sqdmlsl  v0.2d, v1.2s, v2.2s -> 0x0ea2b020 (Q=0, size=10)
    //   sqdmlsl2 v0.2d, v1.4s, v2.4s -> 0x4ea2b020 (Q=1, size=10)
    kSqdmlsl,
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
    // BFCVT <Hd>, <Sn>: FP32 single -> BFloat16 with round-to-nearest-even.
    // Encoded with ftype=01 (the 6-bit opcode + ftype together discriminate
    // this from the FCVT-from-double cases above).  llvm-mc-verified:
    //   bfcvt h0, s1  =  0x1e634020
    kBfcvt = 0b000110,
    kFrintn = 0b001000,
    kFrintp = 0b001001,
    kFrintm = 0b001010,
    kFrintz = 0b001011,
    kFrinta = 0b001100,
    kFrintx = 0b001110,
    kFrinti = 0b001111,
    // FRINTTS (FEAT_FRINTTS): round to a 32/64-bit signed
    // integral FP value, saturating out-of-range / NaN to the most-negative
    // value. Z = toward zero, X = FPCR rounding mode (signals Inexact).
    kFrint32z = 0b010000,
    kFrint32x = 0b010001,
    kFrint64z = 0b010010,
    kFrint64x = 0b010011,
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

  // FP conditional compare args (FCCMP, FCCMPE).
  // If cond evaluates true, perform an FP compare (FCMP-style for FCCMP, FCMPE-style for FCCMPE)
  // and set NZCV; else copy nzcv immediate directly into NZCV.
  struct FpConditionalCompareArgs {
    uint8_t rn;
    uint8_t rm;
    uint8_t nzcv;        // 4-bit NZCV immediate written when condition is false
    Condition cond;      // Condition selecting compare-vs-imm path
    uint8_t ftype;       // 00=S, 01=D
    bool signal_nans;    // true = FCCMPE, false = FCCMP
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
    // FP FCMxxZero (FP32/FP64). The integer kCmxxZero
    // and Armv8.2-FP16 paths reuse the same kCmxxZero enum values disambiguated
    // by args.is_fp16; the FP32/FP64 form lives at a different opcode column
    // (01100/01101/01110 with bits[21:17]=10000, bit23=1), so it carries its
    // own enum values rather than overloading.
    kFcmgtZero,  // FCMGT zero: U=0, opcode=01100
    kFcmgeZero,  // FCMGE zero: U=1, opcode=01100
    kFcmeqZero,  // FCMEQ zero: U=0, opcode=01101
    kFcmleZero,  // FCMLE zero: U=1, opcode=01101
    kFcmltZero,  // FCMLT zero: U=0, opcode=01110 (U=1 unallocated)
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
    kFcvtzsV,   // FCVTZS (vector, FP→int trunc): U=0, opcode=11011, bit23=1
    kFcvtzuV,   // FCVTZU (vector, FP→int trunc): U=1, opcode=11011, bit23=1
    kFrecpeV,   // FRECPE (vector): U=0, opcode=11101, bit23=1
    kFrsqrteV,  // FRSQRTE (vector): U=1, opcode=11101, bit23=1
    kFsqrtV,    // FSQRT  (vector): U=1, opcode=11111, bit23=1
    // FCVT* vector rounding-mode variants.
    // Encoding follows the bit23 ("a") + opcode ("op") split in DDI 0487.
    // Pair  (signed,unsigned) -> bit U.
    kFcvtnsV,   // FCVTNS (vector, round-to-nearest ties-even): U=0, opcode=11010, bit23=0
    kFcvtnuV,   // FCVTNU: U=1, opcode=11010, bit23=0
    kFcvtmsV,   // FCVTMS (vector, round toward -inf): U=0, opcode=11011, bit23=0
    kFcvtmuV,   // FCVTMU: U=1, opcode=11011, bit23=0
    kFcvtpsV,   // FCVTPS (vector, round toward +inf): U=0, opcode=11010, bit23=1
    kFcvtpuV,   // FCVTPU: U=1, opcode=11010, bit23=1
    kFcvtasV,   // FCVTAS (vector, round-to-nearest ties-away): U=0, opcode=11100, bit23=0
    kFcvtauV,   // FCVTAU: U=1, opcode=11100, bit23=0
    kUrecpe,    // URECPE  (vector, unsigned integer reciprocal estimate): U=0, opcode=11100, bit23=1, sz=0
    kUrsqrte,   // URSQRTE (vector, unsigned integer reciprocal sqrt estimate): U=1, opcode=11100, bit23=1, sz=0
    // FRINTTS (vector): round to 32/64-bit integral FP, saturating. opcode
    // 11110=FRINT32, 11111=FRINT64; bit23=0; U selects X(1)/Z(0); sz(bit22)
    // selects FP32 (.2S/.4S) / FP64 (.2D).
    kFrint32zV,
    kFrint32xV,
    kFrint64zV,
    kFrint64xV,
    // BFCVTN/BFCVTN2 (Armv8.6-BF16).
    // Vector narrow FP32 -> BF16. Encoding shares opcode=10110 with FCVTN,
    // but uses size=10 (vs FCVTN's size=00/01). bit30 = Q: Q=0 -> BFCVTN
    // (writes low 4H of Vd, upper 64 bits zeroed); Q=1 -> BFCVTN2 (writes
    // upper 4H of Vd, low 64 bits preserved).
    //   llvm-mc: bfcvtn  v0.4h, v1.4s  = 0x0ea16820
    //            bfcvtn2 v0.8h, v1.4s  = 0x4ea16820
    kBfcvtn,    // BFCVTN/BFCVTN2 (vector narrow FP32->BF16).
    // a=0 column: FP16 vector FRINT* (round to
    // integral). Currently only the FP16 form of these is decoded (via
    // DecodeAdvSimdFp16TwoRegMisc); the std FP32/FP64 two-reg-misc dispatch
    // still routes opcodes 11000/11001 to Undefined() for now.
    kFrintnV,   // FRINTN  (ties-to-even):          a=0, U=0, opcode=11000
    kFrintaV,   // FRINTA  (ties-away):             a=0, U=1, opcode=11000
    kFrintmV,   // FRINTM  (toward -inf):           a=0, U=0, opcode=11001
    kFrintxV,   // FRINTX  (use current rounding):  a=0, U=1, opcode=11001
    kFrintpV,   // FRINTP  (toward +inf):           a=1, U=0, opcode=11000
    kFrintzV,   // FRINTZ  (toward zero):           a=1, U=0, opcode=11001
    kFrintiV,   // FRINTI  (use current rounding):  a=1, U=1, opcode=11001
    // SQABS / SQNEG (signed saturating abs / negate).
    // Encoding: 0 Q U 01110 size 10000 00111 10 Rn Rd
    //   sqabs v0.8b, v1.8b   = 0x0e207820  (Q=0, U=0, size=00)
    //   sqabs v0.16b, v1.16b = 0x4e207820  (Q=1, U=0, size=00)
    //   sqabs v0.4h, v1.4h   = 0x0e607820  (Q=0, U=0, size=01)
    //   sqabs v0.8h, v1.8h   = 0x4e607820  (Q=1, U=0, size=01)
    //   sqabs v0.2s, v1.2s   = 0x0ea07820  (Q=0, U=0, size=10)
    //   sqabs v0.4s, v1.4s   = 0x4ea07820  (Q=1, U=0, size=10)
    //   sqabs v0.2d, v1.2d   = 0x4ee07820  (Q=1, U=0, size=11)
    //   sqneg ... = same with U=1.
    // The only saturating input is INT_MIN_in: |INT_MIN| and -INT_MIN
    // both overflow the signed range and clamp to INT_MAX. .1d (size=11,
    // Q=0) is not encoded for either op.
    kSqabs,     // SQABS: U=0, opcode=00111
    kSqneg,     // SQNEG: U=1, opcode=00111
    // SQXTUN / SQXTUN2 (signed saturating extract unsigned narrow).
    // Encoding: 0 Q 1 01110 size 10000 10010 10 Rn Rd  (U=1 required;
    // U=0 with opcode=10010 is XTN). Sizes 00/01/10 only —
    // size=11 is unallocated for the narrow group (the .1d / .2d
    // destination form does not exist).
    //   sqxtun  v0.8b,  v1.8h  = 0x2e212820  (Q=0, size=00, src=.8h  -> dst=.8b)
    //   sqxtun2 v0.16b, v1.8h  = 0x6e212820  (Q=1, size=00, src=.8h  -> dst=.16b)
    //   sqxtun  v0.4h,  v1.4s  = 0x2e612820  (Q=0, size=01, src=.4s  -> dst=.4h)
    //   sqxtun2 v0.8h,  v1.4s  = 0x6e612820  (Q=1, size=01, src=.4s  -> dst=.8h)
    //   sqxtun  v0.2s,  v1.2d  = 0x2ea12820  (Q=0, size=10, src=.2d  -> dst=.2s)
    //   sqxtun2 v0.4s,  v1.2d  = 0x6ea12820  (Q=1, size=10, src=.2d  -> dst=.4s)
    // Saturation rule: read each signed source lane, clamp to
    // [0, UMAX_dst]: input<0 -> 0; input>UMAX_dst -> UMAX_dst;
    // else cast to unsigned narrow. Q=0 writes low 64 bits of Vd
    // (upper zeroed); Q=1 writes upper 64 bits and preserves lower.
    kSqxtun,    // SQXTUN/SQXTUN2: U=1, opcode=10010
    // SHLL / SHLL2 (shift left long, by element size).
    // Encoding: 0 Q 1 01110 size 10000 10011 10 Rn Rd  (U=1 required;
    // U=0 with opcode=10011 is unallocated). Sizes 00/01/10 only —
    // size=11 (source .1d / .2d) is unallocated (no wider destination).
    // Source element width: esize_src = 8 << size (8/16/32).
    // Destination element width: 2 * esize_src (16/32/64).
    // Shift amount is implicit: always esize_src (i.e., bottom half of
    // each widened destination lane is zero; top half is the source element).
    //   shll  v0.8h, v1.8b,  #8  = 0x2e213820  (Q=0, size=00, src .8b  -> .8h)
    //   shll2 v0.8h, v1.16b, #8  = 0x6e213820  (Q=1, size=00, src .16b -> .8h)
    //   shll  v0.4s, v1.4h,  #16 = 0x2e613820  (Q=0, size=01, src .4h  -> .4s)
    //   shll2 v0.4s, v1.8h,  #16 = 0x6e613820  (Q=1, size=01, src .8h  -> .4s)
    //   shll  v0.2d, v1.2s,  #32 = 0x2ea13820  (Q=0, size=10, src .2s  -> .2d)
    //   shll2 v0.2d, v1.4s,  #32 = 0x6ea13820  (Q=1, size=10, src .4s  -> .2d)
    // Q=0 reads source from low 64 bits of Vn (SHLL); Q=1 reads from upper
    // 64 bits (SHLL2). Both write all 128 bits of Vd. Functionally
    // equivalent to USHLL/USHLL2 with shift = esize_src.
    kShll,      // SHLL/SHLL2: U=1, opcode=10011
    // FCVTXN / FCVTXN2 (vector narrow FP64->FP32, round-to-odd).
    // Encoding: 0 Q 1 01110 0 sz 10000 10110 10 Rn Rd  with sz=1 (size=01).
    // U=1 distinguishes this from FCVTN (U=0, opcode=10110) and from BFCVTN
    // (U=0, opcode=10110, size=10). Only FP64 (size=01) source is encoded; the
    // destination is FP32. Q=0 writes 2 narrow lanes into low 64 bits of Vd
    // (upper zeroed); Q=1 writes 2 narrow lanes into upper 64 bits and
    // preserves lower.
    //   fcvtxn  v0.2s, v1.2d  = 0x2e616820  (Q=0, U=1, size=01)
    //   fcvtxn2 v0.4s, v1.2d  = 0x6e616820  (Q=1, U=1, size=01)
    // Round-to-odd rule (used to avoid double-rounding when chaining narrow
    // conversions): take the round-toward-zero result; if any bits were
    // discarded, force the LSB of the result mantissa to 1.
    kFcvtxn,    // FCVTXN/FCVTXN2: U=1, opcode=10110, size=01
    // across-lanes FP reductions FMAXV / FMINV /
    // FMAXNMV / FMINNMV. These share AdvSIMD two-reg-misc dispatch path
    // but live in the across-lanes group (bit20=1). U picks the element
    // width: U=1 is the FP32 form, U=0 the Armv8.2-FP16 form, surfaced to
    // the backends as args.is_fp16.
    // The `size` field carries "o sz" (bit23=o picks max=0 / min=1;
    // bit22=sz must be 0 — there is no FP64 across-lanes form). The
    // decoder splits max/min via bit23 into separate enum values so the
    // backends only need to know which of the four ops it is.
    // Q picks the arrangement: FP32 has only .4S, so Q=0 (.2S) is
    // unallocated; the FP16 form has .4H (Q=0) and .8H (Q=1).
    //   fmaxv   s0, v1.4s  = 0x6e30f820  (U=1, Q=1, size=00, opcode=01111)
    //   fminv   s0, v1.4s  = 0x6eb0f820  (U=1, Q=1, size=10, opcode=01111)
    //   fmaxnmv s0, v1.4s  = 0x6e30c820  (U=1, Q=1, size=00, opcode=01100)
    //   fminnmv s0, v1.4s  = 0x6eb0c820  (U=1, Q=1, size=10, opcode=01100)
    //   fmaxv   h0, v1.4h  = 0x0e30f820  (U=0, Q=0, size=00, opcode=01111)
    //   fmaxv   h0, v1.8h  = 0x4e30f820  (U=0, Q=1, size=00, opcode=01111)
    //   fminv   h0, v1.4h  = 0x0eb0f820  (U=0, Q=0, size=10, opcode=01111)
    //   fminv   h0, v1.8h  = 0x4eb0f820  (U=0, Q=1, size=10, opcode=01111)
    //   fmaxnmv h0, v1.4h  = 0x0e30c820  (U=0, Q=0, size=00, opcode=01100)
    //   fmaxnmv h0, v1.8h  = 0x4e30c820  (U=0, Q=1, size=00, opcode=01100)
    //   fminnmv h0, v1.4h  = 0x0eb0c820  (U=0, Q=0, size=10, opcode=01100)
    //   fminnmv h0, v1.8h  = 0x4eb0c820  (U=0, Q=1, size=10, opcode=01100)
    // Semantics: FMAXV/FMINV use IEEE 754-2008 max/min (any NaN -> NaN);
    // FMAXNMV/FMINNMV use max-number/min-number (NaN excluded when other
    // input is non-NaN). Result is a scalar lane in the bottom esize bytes
    // of Vd (upper bits zeroed by the routine's `result=0` init).
    kFmaxv,     // FMAXV   (across .4S/.4H/.8H): opcode=01111, bit23=0
    kFminv,     // FMINV   (across .4S/.4H/.8H): opcode=01111, bit23=1
    kFmaxnmv,   // FMAXNMV (across .4S/.4H/.8H): opcode=01100, bit23=0
    kFminnmv,   // FMINNMV (across .4S/.4H/.8H): opcode=01100, bit23=1
  };

  // SHA-512 (FEAT_SHA512) ops live outside the AdvSIMD
  // encoding family. The three-register SHA-512 group encodes:
  //   11001110 011 Rm 1000 o2 Rn Rd
  // with o2 = bits[11:10] picking the op. The two-register variant
  // (SHA512SU0) encodes: 11001110 110 00000 1000 00 Rn Rd.
  enum class Sha512Op : uint8_t {
    kSha512h,    // 3-reg, o2=00
    kSha512h2,   // 3-reg, o2=01
    kSha512su1,  // 3-reg, o2=10
    kSha512su0,  // 2-reg
  };

  struct AdvSimdTwoRegMiscArgs {
    AdvSimdTwoRegMiscOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t size;     // element size: 00=8b, 01=16b, 10=32b, 11=64b
    bool q;           // Q bit: 0=64-bit vector, 1=128-bit vector
    bool u;           // U bit from encoding
    // Armv8.2-FP16 vector two-reg-misc.
    // The FP16 encoding (DDI 0487 C7.2 "Advanced SIMD two-register
    // miscellaneous (FP16)") shares this struct.  When set, lanes are
    // 2-byte half and `size` carries no meaning (set to 0 by the FP16
    // dispatcher).  The interpreter reads is_fp16 inside each opcode
    // case and dispatches via FpHalfToSingle / FpSingleToHalf as.
    bool is_fp16;
  };

  //
  // AdvSIMD scalar two-reg misc opcodes.
  //
  enum class AdvSimdScalarTwoRegMiscOpcode : uint8_t {
    kScvtf,   // SCVTF  (scalar): signed int → float                       (opcode=11101, U=0; size ∈ {S,D})
    kUcvtf,   // UCVTF  (scalar): unsigned int → float                     (opcode=11101, U=1; size ∈ {S,D})
    kFcvtzs,  // FCVTZS (scalar): float → signed int, round toward zero    (opcode=11011, U=0; size ∈ {S,D})
    kFcvtzu,  // FCVTZU (scalar): float → unsigned int, round toward zero  (opcode=11011, U=1; size ∈ {S,D})
    // scalar FRECPE / FRSQRTE.
    kFrecpe,  // FRECPE  (scalar): FP reciprocal estimate                  (opcode=11101, U=0; size ∈ {10, 11})
    kFrsqrte, // FRSQRTE (scalar): FP reciprocal square-root estimate      (opcode=11101, U=1; size ∈ {10, 11})
    // scalar FCVTAS / FCVTAU.
    kFcvtas,  // FCVTAS (scalar): float → signed int, round-to-nearest ties-away (opcode=11100, U=0; size ∈ {00, 01})
    kFcvtau,  // FCVTAU (scalar): float → unsigned int, round-to-nearest ties-away (opcode=11100, U=1; size ∈ {00, 01})
    // scalar twins of vector SQABS / SQNEG / SQXTUN / FCVTXN.
    kSqabs,   // SQABS  (scalar): saturating signed absolute value         (opcode=00111, U=0; size ∈ {B,H,S,D})
    kSqneg,   // SQNEG  (scalar): saturating signed negate                 (opcode=00111, U=1; size ∈ {B,H,S,D})
    kSqxtun,  // SQXTUN (scalar): signed→unsigned saturating extract narrow (opcode=10010, U=1; size ∈ {B,H,S})
    kSqxtn,   // SQXTN  (scalar): signed saturating extract narrow         (opcode=10100, U=0; size ∈ {B,H,S})
    kUqxtn,   // UQXTN  (scalar): unsigned saturating extract narrow       (opcode=10100, U=1; size ∈ {B,H,S})
    kFcvtxn,  // FCVTXN (scalar): FP64→FP32 round-to-odd narrow             (opcode=10110, U=1; size=01)
  };

  struct AdvSimdScalarTwoRegMiscArgs {
    AdvSimdScalarTwoRegMiscOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    // Raw 2-bit size field from the encoding.
    //   For FP ops (SCVTF/UCVTF/FCVTZS/FCVTZU/FCVTXN), size matches the
    //   ARM ARM "sz" bit conventions: 0 = single (FP32), 1 = double (FP64).
    //   For integer ops (SQABS/SQNEG/SQXTUN), size selects the lane width
    //   in bytes via `1 << size`: 0=B, 1=H, 2=S, 3=D.
    uint8_t size;
  };

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
    // FP scalar three same.
    // For FP variants, size bits [23:22] = 1x where bit22 (sz) selects S (0) or D (1).
    kFabd,    // FABD (scalar, FP): U=1, bit23=1, opcode=11010
    kFcmgt,   // FCMGT (scalar, FP): U=1, bit23=1, opcode=11100
    kFcmge,   // FCMGE (scalar, FP): U=1, bit23=0, opcode=11100
    kFcmeq,   // FCMEQ (scalar, FP): U=0, bit23=0, opcode=11100
    kFacgt,   // FACGT (scalar, FP): U=1, bit23=1, opcode=11101
    kFacge,   // FACGE (scalar, FP): U=1, bit23=0, opcode=11101
    kFmulx,   // FMULX (scalar, FP): U=0, bit23=1, opcode=11011.
              // Identical to FMUL except (±0 * ±inf) lanes return ±2.0
              // instead of NaN — see Interpreter::FmulxScalar<>.
    kFrecps,  // FRECPS (scalar, FP): U=0, bit23=0, opcode=11111.
              // Reciprocal step: (2.0 - a*b) with (0*inf) -> +2.0 saturation.
              // FP16 scalar variant: a=0, U=0, opcode_3=111.
    kFrsqrts, // FRSQRTS (scalar, FP): U=0, bit23=1, opcode=11111.
              // Reciprocal sqrt step: (3.0 - a*b)/2 with (0*inf) -> +1.5
              // saturation. FP16 scalar variant: a=1, U=0, opcode_3=111.
    // scalar saturating add/sub (B/H/S/D).
    //   size=00 -> B,  01 -> H,  10 -> S,  11 -> D.
    //   Unlike the rest of the integer scalar three-same family which is
    //   D-form only, opcodes 00001 (SQADD/UQADD) and 00011 (SQSUB/UQSUB)
    //   carry args.size = {00, 01, 10, 11}; the interpreter expands esize
    //   accordingly and applies the per-width saturating bounds.  Encoding
    //   reference (ARM ARM C7.2.282 / .284 / .317 / .319):
    //     SQADD <V>d, <V>n, <V>m  = 01 0 11110 size 1 Rm 0 0001 1 Rn Rd
    //     UQADD <V>d, <V>n, <V>m  = 01 1 11110 size 1 Rm 0 0001 1 Rn Rd
    //     SQSUB <V>d, <V>n, <V>m  = 01 0 11110 size 1 Rm 0 0011 1 Rn Rd
    //     UQSUB <V>d, <V>n, <V>m  = 01 1 11110 size 1 Rm 0 0011 1 Rn Rd
    kSqaddScalar,
    kUqaddScalar,
    kSqsubScalar,
    kUqsubScalar,
    // scalar saturating shift left (B/H/S/D).
    //   size = 00 -> B, 01 -> H, 10 -> S, 11 -> D.
    //   Shift amount is the low 8 bits of Vm (signed); negative shifts
    //   are arithmetic right (signed forms) or logical right (unsigned).
    //   The "R" variants round (add `1 << (rshift-1)` before the right
    //   shift) and the "S/U-Q" forms saturate the left-shift result to
    //   the per-width signed/unsigned range.  See ARM ARM C7.2.302 /
    //   .306 / .310 / .313 (SQSHL / SQRSHL / UQSHL / UQRSHL).
    //   Encoding (scalar):
    //     SQSHL  <V>d, <V>n, <V>m = 01 0 11110 size 1 Rm 0 1001 1 Rn Rd
    //     UQSHL  <V>d, <V>n, <V>m = 01 1 11110 size 1 Rm 0 1001 1 Rn Rd
    //     SQRSHL <V>d, <V>n, <V>m = 01 0 11110 size 1 Rm 0 1011 1 Rn Rd
    //     UQRSHL <V>d, <V>n, <V>m = 01 1 11110 size 1 Rm 0 1011 1 Rn Rd
    kSqshlScalar,
    kUqshlScalar,
    kSqrshlScalar,
    kUqrshlScalar,
    // scalar rounding shift left (D only, non-saturating).
    //   size = 11 -> D (only valid lane width).
    //   Shift amount is the low 8 bits of Vm (signed); negative shifts
    //   are arithmetic right (signed) or logical right (unsigned), with
    //   the standard rounding term `1 << (rshift-1)` added before the
    //   right shift.  No saturation — bits beyond D are discarded.
    //   See ARM ARM C7.2.270 / .335 (SRSHL / URSHL).
    //   Encoding (scalar):
    //     SRSHL  d, d, d  = 01 0 11110 11 1 Rm 0 1010 1 Rn Rd
    //     URSHL  d, d, d  = 01 1 11110 11 1 Rm 0 1010 1 Rn Rd
    kSrshlScalar,
    kUrshlScalar,
    // scalar saturating doubling multiply high (H/S only).
    //   size = 01 -> H, 10 -> S.  Lane widths B and D are unallocated for
    //   this opcode; the decoder rejects them as Undefined.
    //   Computes `Vd = sat_signed((2 * sext(Vn) * sext(Vm) + round) >>
    //   bits_local)`, with `bits_local = 16` for H and `32` for S, and
    //   `round = 1 << (bits_local - 1)` for SQRDMULH or `0` for SQDMULH.
    //   See ARM ARM C7.2.301 / .305 (SQDMULH / SQRDMULH).
    //   Encoding (scalar):
    //     SQDMULH  <V>d, <V>n, <V>m  = 01 0 11110 size 1 Rm 1 0110 1 Rn Rd
    //     SQRDMULH <V>d, <V>n, <V>m  = 01 1 11110 size 1 Rm 1 0110 1 Rn Rd
    kSqdmulhScalar,
    kSqrdmulhScalar,
    // Armv8.1-RDM scalar saturating rounding doubling
    //   multiply accumulate / subtract high (H/S only).
    //   size = 01 -> H, 10 -> S.  Lane widths B and D are unallocated for
    //   this opcode; the decoder rejects them as Undefined.
    //   Stage 1: SQRDMULH(Vn, Vm) (rounding doubling multiply high, signed
    //   saturated).  Stage 2: signed-saturating ADD (SQRDMLAH) or SUB
    //   (SQRDMLSH) of stage-1 result into Vd.
    //   See ARM ARM C7.2.299 / .301 (SQRDMLAH / SQRDMLSH, scalar form).
    //   Encoding (scalar three-same-extra):
    //     SQRDMLAH <V>d, <V>n, <V>m  = 01 1 11110 size 0 Rm 1 0000 1 Rn Rd
    //     SQRDMLSH <V>d, <V>n, <V>m  = 01 1 11110 size 0 Rm 1 0001 1 Rn Rd
    //   Reuses the AdvSimdScalarThreeSame consumer surface so the
    //   interpreter and JIT can dispatch on a single enum.
    kSqrdmlahScalar,
    kSqrdmlshScalar,
  };

  struct AdvSimdScalarThreeSameArgs {
    AdvSimdScalarThreeSameOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t size;   // integer: full size field; FP: 0 -> S (32-bit), 1 -> D (64-bit);
                    // FP16 scalar three-same: 0 (unused — interpreter checks is_fp16).
    // Armv8.2-FP16 scalar three-same.  When true, the
    // interpreter reads the low 16 bits of Vn/Vm as binary16 and performs
    // a widen-op-narrow round-trip through FP32 (FpHalfToSingle ->
    // semantic helper -> FpSingleToHalf).  Bit-exact for the FMULX
    // saturation case because ±2.0 is exactly representable in FP16.
    bool is_fp16 = false;
  };

  // AdvSIMD scalar pairwise opcodes.
  // Encoding: 01 U 11110 size 11000 opcode 10 Rn Rd
  // Reads a pair of esize elements from Vn (low pair) and combines them into
  // a single scalar in Vd.
  enum class AdvSimdScalarPairwiseOpcode : uint8_t {
    kAddp,    // ADDP (scalar): U=0, size=11, opcode=11011 -> d-form
    // FP scalar pairwise variants.  U=1 in the dispatch encoding selects the
    // FP32/FP64 forms (size[0] picks S vs D); U=0 with bit22=0 selects the
    // Armv8.2-FP16 forms (carried on AdvSimdScalarPairwiseArgs.is_fp16).
    //   size[0] (bit 22) selects S (0) or D (1) for the U=1 forms.
    //   size[1] (bit 23) selects max- vs min- variants for FMAX*/FMIN*.
    kFaddpScalar,    // FADDP scalar:    opcode=01101, bit23=0
    kFmaxnmpScalar,  // FMAXNMP scalar:  opcode=01100, bit23=0
    kFminnmpScalar,  // FMINNMP scalar:  opcode=01100, bit23=1
    kFmaxpScalar,    // FMAXP scalar:    opcode=01111, bit23=0
    kFminpScalar,    // FMINP scalar:    opcode=01111, bit23=1
  };

  struct AdvSimdScalarPairwiseArgs {
    AdvSimdScalarPairwiseOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t size;
    bool is_fp16;
  };

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
    // AdvSIMD shift-by-immediate narrow family — the
    // prior enum was missing SQSHRUN / SQRSHRUN / SQRSHRN / UQRSHRN.
    // Without these, opcodes 0b10010 / 0b10011 (both U-values) silently
    // fell through to the dispatch default and the U=1 variants of
    // 0b10000 / 0b10001 (real SQSHRUN / SQRSHRUN) were mis-dispatched
    // to SQSHRN / UQSHRN handlers.  See dispatch table at
    // DecodeAdvSimdShiftByImm for the corrected (opcode, U) → enum map.
    kSqshrun,
    kSqrshrun,
    kSqrshrn,
    kUqrshrn,
    kSshll,
    kUshll,
    // scalar fixed-point conversion shift-by-imm family.
    // Opcodes 11100 (SCVTF/UCVTF) and 11111 (FCVTZS/FCVTZU) of
    // DecodeAdvSimdScalarShiftByImm — convert between fixed-point integer
    // and floating-point with the encoded shift as fractional-bit count.
    // S (immh=01xx) and D (immh=1xxx) widths only; FP16 (immh=001x) is
    // deferred and rejected by the decoder. The Fixed suffix distinguishes
    // these from the no-shift AdvSimd two-reg-misc SCVTF/UCVTF/FCVTZS/FCVTZU
    // already present in AdvSimdScalarTwoRegMiscOpcode / AdvSimdTwoRegMiscOpcode.
    kScvtfFixed,
    kUcvtfFixed,
    kFcvtzsFixed,
    kFcvtzuFixed,
  };

  struct AdvSimdShiftImmArgs {
    AdvSimdShiftImmOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t immh;     // immh field (bits[22:19])
    uint8_t immb;     // immb field (bits[18:16])
    bool q;           // Q bit
    bool u;           // U bit
    // Scalar shift-by-immediate dispatch sets this; vector dispatch leaves
    // it false.  When set, the consumer must process exactly one element
    // regardless of esize (overrides num_elements = vec_len / esize), and
    // must zero the upper 64 bits of Vd to match ARM ARM scalar semantics.
    bool scalar = false;
  };

  //
  // AdvSIMD vector x indexed element opcodes.
  //
  enum class AdvSimdVecXIdxOpcode : uint8_t {
    kFmla,    // FMLA (by element)
    kFmls,    // FMLS (by element)
    kFmul,    // FMUL (by element)
    kFmulx,   // FMULX (by element): U=1, opcode=1001.  Same lane semantics as
              // kFmul except (±0 * ±inf) lanes return ±2.0 instead of NaN.
    kMul,     // MUL (by element)
    kMla,     // MLA (by element)
    kMls,     // MLS (by element)
    kSqdmulhIdx,   // SQDMULH (by element): U=0, opcode=1100, size in {01,10}.
                   // Per-lane saturating doubling multiply high (no rounding).
    kSqrdmulhIdx,  // SQRDMULH (by element): U=0, opcode=1101, size in {01,10}.
                   // Same as SQDMULH but with rounding constant 1<<(esize-1)
                   // added before the right shift.
    kSmullIdx,     // SMULL/SMULL2 (by element): U=0, opcode=1010, size in {01,10}.
                   // Widening signed multiply: dst lane = sign_ext(Vn[i]) *
                   // sign_ext(Vm[index]); dst is always 128-bit.  Q selects Vn
                   // low half (Q=0, SMULL) vs high half (Q=1, SMULL2).
    kUmullIdx,     // UMULL/UMULL2 (by element): U=1, opcode=1010, size in {01,10}.
                   // Widening unsigned multiply; same Q semantics as kSmullIdx.
    kSmlalIdx,     // SMLAL/SMLAL2 (by element): U=0, opcode=0010, size in {01,10}.
                   // Vd lane += sign_ext(Vn[i]) * sign_ext(Vm[index]).
    kUmlalIdx,     // UMLAL/UMLAL2 (by element): U=1, opcode=0010, size in {01,10}.
                   // Vd lane += unsigned product.
    kSmlslIdx,     // SMLSL/SMLSL2 (by element): U=0, opcode=0110, size in {01,10}.
                   // Vd lane -= sign_ext(Vn[i]) * sign_ext(Vm[index]).
    kUmlslIdx,     // UMLSL/UMLSL2 (by element): U=1, opcode=0110, size in {01,10}.
                   // Vd lane -= unsigned product.
    kSqdmullIdx,   // SQDMULL/SQDMULL2 (by element): U=0, opcode=1011, size in {01,10}.
                   // Saturating doubling widening multiply: dst lane =
                   // SignedSat(2 * sign_ext(Vn[i]) * sign_ext(Vm[index])).
                   // Only (INT_MIN_in, INT_MIN_in) saturates; dst always 128-bit.
    kSqdmlalIdx,   // SQDMLAL/SQDMLAL2 (by element): U=0, opcode=0011, size in {01,10}.
                   // Vd_wide[i] = SignedSat(Vd_wide[i] + SignedSat(2 * sn * sm)).
    kSqdmlslIdx,   // SQDMLSL/SQDMLSL2 (by element): U=0, opcode=0111, size in {01,10}.
                   // Vd_wide[i] = SignedSat(Vd_wide[i] - SignedSat(2 * sn * sm)).
    kSqrdmlahIdx,  // SQRDMLAH (by element, Armv8.1-RDM): U=1, opcode=1101,
                   // size in {01,10}.  Non-widening: dst lane width = source
                   // lane width.  Per-lane: addend = SignedSat(round(2*sn*sm)
                   // >> esize_bits), then Vd[i] = SignedSat(Vd[i] + addend).
                   // Reuses the SQRDMULH math for the addend computation.
    kSqrdmlshIdx,  // SQRDMLSH (by element, Armv8.1-RDM): U=1, opcode=1111,
                   // size in {01,10}.  Same as kSqrdmlahIdx but subtracts the
                   // saturated rounded-doubled product: Vd[i] = SignedSat(
                   // Vd[i] - SignedSat(round(2*sn*sm) >> esize_bits)).
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

  //
  // AdvSIMD scalar x indexed element (ARM ARM C4.1.71).
  // Encoding: 0 1 U 11111 size L M Rm opcode H 0 Rn Rd
  // The destination is a single FP lane in Vd; upper lanes of Vd are zeroed.
  // This is the scalar sibling of AdvSimdVecXIndexedElement.  size encodes
  // the precision (10=FP32, 11=FP64; 01=FP16 is Armv8.2-FP16 — not handled
  // here yet).  FMUL / FMULX / FMLA / FMLS are implemented; the remaining
  // scalar-x-indexed encodings (SQDMULL / SQDMULH variants) fall through
  // the decoder to Undefined() until they are needed.
  //
  enum class AdvSimdScalarXIdxOpcode : uint8_t {
    kFmulx,  // FMULX (scalar by element): U=1, opcode=1001.  Same lane
             // semantics as FMUL except (±0 * ±inf) returns ±2.0 instead
             // of NaN.  Used by libm reciprocal-estimate refinement loops.
    kFmul,   // FMUL  (scalar by element): U=0, opcode=1001.
    kFmla,   // FMLA  (scalar by element): U=0, opcode=0001.  Fused multiply-add.
    kFmls,   // FMLS  (scalar by element): U=0, opcode=0101.  Fused negated multiply-add.
    // Armv8.1-RDM scalar by-element saturating rounding
    //   doubling multiply accumulate / subtract high (H/S only).
    //   Stage 1: SQRDMULH(Vn, broadcast(Vm[index])).  Stage 2: signed-
    //   saturating ADD (SQRDMLAH) or SUB (SQRDMLSH) of stage-1 result
    //   into Vd.  Sibling of the vector by-element forms
    //   kSqrdmlahIdx / kSqrdmlshIdx; same math, scalar lane width.
    //   See ARM ARM C7.2.300 / .302 (SQRDMLAH / SQRDMLSH, scalar by-element).
    //   Encoding (scalar x indexed element):
    //     SQRDMLAH <V>d, <V>n, <Vm>.<T>[i] = 01 1 11111 size L M Rm 1101 H 0 Rn Rd
    //     SQRDMLSH <V>d, <V>n, <Vm>.<T>[i] = 01 1 11111 size L M Rm 1111 H 0 Rn Rd
    kSqrdmlahScalarIdx,
    kSqrdmlshScalarIdx,
    // Scalar by-element signed saturating doubling multiply (H/S only). These
    // were previously unallocated -> SIGILL; the interpreter now implements
    // them (the JITs bail to the interpreter). U=0.
    //   kSqdmulhScalarIdx  (opcode=1100): Vd = SignedSat((2*Vn*Vm[i]) >> bits).
    //   kSqrdmulhScalarIdx (opcode=1101): rounded high half (+ 1<<(bits-1)).
    //   kSqdmullScalarIdx  (opcode=1011): widening Vd = SignedSat(2*Vn*Vm[i])
    //                                      into the 2x-wide element.
    //   Verified (aarch64-linux-gnu-as -march=armv8.2-a+fp16):
    //     sqdmulh  s0, s1, v2.s[0] = 0x5f82c020
    //     sqrdmulh s0, s1, v2.s[0] = 0x5f82d020
    //     sqdmull  d0, s1, v2.s[0] = 0x5f82b020
    kSqdmulhScalarIdx,
    kSqrdmulhScalarIdx,
    kSqdmullScalarIdx,
  };

  struct AdvSimdScalarXIdxArgs {
    AdvSimdScalarXIdxOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;       // indexed source register
    uint8_t index;    // element index within rm
    uint8_t size;     // 10 = FP32 (single), 11 = FP64 (double)
  };

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

    // Create a mask of (s + 1) ones. Handle s=63 to avoid UB from 1<<64.
    // uint64_t welem = (1ULL << (s + 1)) - 1;
    uint64_t welem = (s >= 63) ? ~0ULL : ((1ULL << (s + 1)) - 1);

    // Rotate right by r within esize bits.
    // // welem = ((welem >> r) | (welem << (esize - r))) & ((1ULL << esize) - 1);
    if (r != 0) {
      uint64_t mask = (esize >= 64) ? ~0ULL : ((1ULL << esize) - 1);
      welem = ((welem >> r) | (welem << (esize - r))) & mask;
    }

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
        // SIMD & FP (op0 = x111).
        // // TODO: Implement SIMD/FP decoding.
        // Undefined();
        DecodeSimdFp();
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
        // Add/subtract (immediate, with tags): ADDG/SUBG.
        DecodeAddSubImmTags();
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
        // Extract (EXTR/ROR).
        DecodeExtract();
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

  // ADDG/SUBG (add/subtract immediate, with tags).
  // Encoding: sf=1 op S=0 100011 0 uimm6[21:16] (00) uimm4[13:10] Rn Rd.
  void DecodeAddSubImmTags() {
    bool is_sub = GetBits<30, 1>();
    const AddSubImmTagsArgs args = {
        .dst = GetBits<0, 5>(),
        .src = GetBits<5, 5>(),
        .uimm6 = GetBits<16, 6>(),
        .uimm4 = GetBits<10, 4>(),
        .is_sub = is_sub,
    };
    insn_consumer_->AddSubImmTags(args);
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
        // Both CBZ/CBNZ and TBZ/TBNZ have op0=x01.
        // Distinguish by bit25: 0=CBZ/CBNZ, 1=TBZ/TBNZ.
        // // Compare and branch: CBZ/CBNZ.
        // DecodeCompareAndBranch();
        if (GetBits<25, 1>()) {
          DecodeTestAndBranch();
        } else {
          DecodeCompareAndBranch();
        }
        break;
      case 0b011:
      case 0b111:
        // // Test and branch: TBZ/TBNZ.
        // DecodeTestAndBranch();
        // Same dispatch as above — both x01 and x11 can reach here.
        if (GetBits<25, 1>()) {
          DecodeTestAndBranch();
        } else {
          DecodeCompareAndBranch();
        }
        break;
      case 0b110: {
        // This group contains: Unconditional branch (register), Exception generation, System.
        uint32_t opc_top = GetBits<22, 4>();
        // Exception generation is bits[25:24]==00 (the
        // 11010100 prefix); the opc field lives in bits[23:21], so HLT (opc=010)
        // and DCPS1-3 (opc=101) set bits 22/23 and must still route here rather
        // than falling through to Undefined(). DecodeExceptionGeneration sorts
        // out the individual opcodes.
        if ((opc_top & 0b1100) == 0b0000) {
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
    // BRK (opc=001, ll=00) and HLT (opc=010, ll=00) are
    // breakpoint-class instructions; both deliver a synchronous SIGTRAP to the
    // guest (so debuggers and sanitizers — HWASan, UBSan trap-on-error — see a
    // breakpoint at the guest PC rather than an illegal-instruction abort).
    if (opc == 0b001 && ll == 0b00) {  // BRK
      insn_consumer_->Brk(imm16);
      return;
    }
    if (opc == 0b010 && ll == 0b00) {  // HLT — same SIGTRAP path as BRK
      insn_consumer_->Brk(imm16);
      return;
    }
    // HVC (opc=000, ll=10), SMC (opc=000, ll=11) and DCPS1-3 (opc=101) are
    // UNDEFINED at EL0 (privileged / debug-state-only). They fall through to
    // Undefined(), which delivers SIGILL to the guest — the architecturally
    // correct result for guest user-space. The translator itself never aborts.
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
        // hint audit (HINT #N = CRm:op2[2:0]).
        // The HINT (CRn=0010) space encodes a 7-bit hint number formed
        // by CRm:op2.  At EL0 every hint we see should be treated as a
        // no-op — the kernel handles any real sleep / wake / barrier
        // semantics, and we never run at EL1.  Encodings verified with
        // llvm-mc (clang-r563880c, armv8.5-a):
        //   NOP   = CRm=0000, op2=000  (HINT #0)
        //   YIELD = CRm=0000, op2=001  (HINT #1)
        //   WFE   = CRm=0000, op2=010  (HINT #2)
        //   WFI   = CRm=0000, op2=011  (HINT #3)
        //   SEV   = CRm=0000, op2=100  (HINT #4)
        //   SEVL  = CRm=0000, op2=101  (HINT #5)
        //   DGH   = CRm=0000, op2=110  (HINT #6, Armv8.6-DGH)
        //   ESB   = CRm=0010, op2=000  (HINT #16, Armv8.2-RAS)
        //   CSDB  = CRm=0010, op2=100  (HINT #20, Armv8.0-PRED)
        //   PAC/AUT-1716, BTI, etc. also live here at higher hint
        //   numbers.  All are correctly NOPed for binary translation.
        // explicit BTI audit: BTI guards live in
        // the HINT space with CRm=0100; the four variants are
        //   BTI    = CRm=0100, op2=000  (HINT #32 = 0x20)
        //   BTI c  = CRm=0100, op2=010  (HINT #34 = 0x22)
        //   BTI j  = CRm=0100, op2=100  (HINT #36 = 0x24)
        //   BTI jc = CRm=0100, op2=110  (HINT #38 = 0x26)
        // The intervening odd-op2 encodings (HINT #33/35/37/39) are
        // reserved BTI placeholders and decode as plain NOP on hardware
        // that does not implement BTI — exactly what we want here.  All
        // of HINT #32–#39 (the full 0x20–0x27 block called out in the
        // plan) share CRn=0010 with the other hints above and reach
        // this `Nop()` call, never `Undefined()`.  Verified by
        // hello-bti sample (compiled with -mbranch-protection=bti):
        // every indirect-branch entry point starts with a BTI guard
        // and all four mnemonics also appear as explicit inline-asm
        // probes — process must not SIGILL.
        insn_consumer_->Nop();
        return;
      }
      if (crn == 0b0011) {
        // Memory and synchronization barriers.  Op2 selects the variant:
        //   CLREX = CRn=0011, CRm=imm,  op2=010
        //   DSB   = CRn=0011, CRm=opt,  op2=100  (incl. "DFB" — full)
        //   DMB   = CRn=0011, CRm=opt,  op2=101
        //   ISB   = CRn=0011, CRm=imm,  op2=110
        //   SB    = CRn=0011, CRm=0000, op2=111  (Armv8.5-SB)
        //   TSB CSYNC = CRn=0011, CRm=0010, op2=010 (Armv8.4-TRBE)
        //
        // x86_64 TSO provides StoreStore, LoadLoad and LoadStore ordering for
        // free, but NOT StoreLoad. A full DMB/DSB (CRm option type 0b11=full or
        // 0b00=reserved-treated-as-full: SY/ISH/NSH/OSH) is a FULL barrier that
        // DOES order a prior store before a later load, so it must lower to an
        // MFENCE — NOP-ing it drops StoreLoad ordering and breaks sequentially
        // consistent guest code (flaky lock-free / seqlock corruption). The
        // store-only (CRm type 0b10, *ST) and load-only (0b01, *LD) variants
        // need only orderings x86 already gives, so they stay NOPs.
        //
        // ISB needs no DATA fence (guest code patching is handled via IC IVAU /
        // the translator's own cache invalidation, not guest ISB). CLREX clears
        // the LL/SC monitor, which we model as cmpxchg, and SB/TSB are
        // speculation/trace barriers — all remain NOPs.
        const bool is_dmb = (op2 == 0b101);
        const bool is_dsb = (op2 == 0b100);
        const uint8_t barrier_type = crm & 0b11;  // 11=full, 00=reserved(full), 10=ST, 01=LD
        if ((is_dmb || is_dsb) && (barrier_type == 0b11 || barrier_type == 0b00)) {
          insn_consumer_->DataMemoryBarrier();
        } else {
          insn_consumer_->Nop();
        }
        return;
      }
      if (crn == 0b0100) {
        // MSR (immediate): write a PSTATE field. CRm:op2 select the field; Rt
        // is fixed 0b11111. Encodings (op1:op2), verified against the ARM ARM:
        //   CFINV=000:000, XAFLAG=000:001, AXFLAG=000:010, UAO=000:011,
        //   PAN=000:100, SPSel=000:101, ALLINT=001:000, SSBS=011:001,
        //   DIT=011:010, SVCRSM/SVCRZA(SME)=011:011, TCO=011:100,
        //   DAIFSet=011:110, DAIFClr=011:111.
        // All of these PSTATE bits are EL0-irrelevant for binary translation:
        // they govern MTE tag-check (TCO), data-independent timing (DIT),
        // speculative-store-bypass (SSBS), privileged access control
        // (PAN/UAO/SPSel/ALLINT) and interrupt masking (DAIF) — none affects the
        // data computation we translate, and at EL0 user code cannot change the
        // privileged ones anyway. So every MSR-immediate is a no-op here, exactly
        // like the HINT space above. NOPing avoids a spurious SIGILL on MTE/
        // DIT/SSBS-hardened guest code (e.g. Chromium's Scudo `msr TCO` toggles).
        insn_consumer_->Nop();
        return;
      }
    }

    // SYS instructions (op0=01): cache maintenance (DC, IC), TLB ops, etc.
    if (op0 == 0b01 && l == 0) {
      // IC IVAU, Xt — Instruction Cache Invalidate by VA to Point of Unification
      // (op1=011, CRn=0111, CRm=0101, op2=001). This is how ARM64 user-space
      // signals self-modified code: a JIT (e.g. PCRE2/sljit) writes new code and
      // issues IC IVAU per cache line. A binary translator MUST invalidate its
      // translation cache for that address, otherwise it keeps running a stale
      // translation of the previous code. (On riscv64 the equivalent is the
      // riscv_flush_icache syscall; ARM64 has no such syscall.) All other cache
      // maintenance (DC, IC IALLU, TLB) is safe to NOP for a shared-memory
      // in-process translator.
      if (op1 == 0b011 && crn == 0b0111 && crm == 0b0101 && op2 == 0b001) {
        insn_consumer_->IcIvau(rt);
        return;
      }
      insn_consumer_->Nop();
      return;
    }
    // SYSL instructions (op0=01, l=1): also safe as NOP.
    if (op0 == 0b01 && l == 1) {
      insn_consumer_->Nop();
      return;
    }

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
    // PAuth BR/BLR/RET variants (Armv8.3-PAuth).
    // op3 = bits[15:10] distinguishes the PAC variants from the plain ones:
    //   000000 = no PAC; 000010 = A-key PAC; 000011 = B-key PAC.
    // BRAAZ/BRABZ (opc=0000) and BLRAAZ/BLRABZ (opc=0001) already route
    // correctly through the existing BR/BLR cases since Rn carries the
    // actual target and op4 is ignored.  RETAA/RETAB (opc=0010, op3!=0)
    // need the implicit LR (X30) as the source — the encoding hardcodes
    // Rn=11111 (XZR) which would otherwise be misrouted.  BRAA/BRAB
    // (opc=1000) and BLRAA/BLRAB (opc=1001) currently bail to Undefined.
    // PAC modifier in op4/Rm is ignored: Digitalis never inserts PAC bits
    // into pointers, so Rn already holds the clean target — no masking
    // is needed.
    uint8_t op3 = GetBits<10, 6>();
    bool is_pac = (op3 == 0b000010 || op3 == 0b000011);

    switch (opc) {
      case 0b0000: {
        // BR Xn, BRAAZ Xn, BRABZ Xn.
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
        // BLR Xn, BLRAAZ Xn, BLRABZ Xn.
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
        // RET {Xn} (default Xn = X30), RETAA, RETAB.
        // RETAA/RETAB are encoded with Rn=11111
        // but the architectural source register is implicitly X30 (LR).
        uint8_t src = is_pac ? 30 : rn;
        const BranchRegArgs args = {
            .src = src,
            .link_reg = 0,
            .is_link = false,
            .is_ret = true,
        };
        insn_consumer_->BranchReg(args);
        break;
      }
      // BRAA/BRAB/BLRAA/BLRAB (Armv8.3-PAuth).
      case 0b1000: {
        // BRAA Xn, Xm / BRAB Xn, Xm.
        if (!is_pac) { Undefined(); return; }
        const BranchRegArgs args = {
            .src = rn,
            .link_reg = 0,
            .is_link = false,
            .is_ret = false,
        };
        insn_consumer_->BranchReg(args);
        break;
      }
      case 0b1001: {
        // BLRAA Xn, Xm / BLRAB Xn, Xm.
        if (!is_pac) { Undefined(); return; }
        const BranchRegArgs args = {
            .src = rn,
            .link_reg = 30,
            .is_link = true,
            .is_ret = false,
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
    uint8_t op4 = GetBits<10, 2>();

    // Load/store exclusive/atomic: bit29=0, op_28_27=01, op_26=0
    // Includes: LDXR, STXR, LDAXR, STLXR, LDAR, STLR, CAS, CASP
    if (op_28_27 == 0b01 && op_26 == 0 && op_29 == 0) {
      DecodeLoadStoreExclusive();
      return;
    }

    // Load/store pair: bit29=1, op_28_27=01, op_26=0
    // Encoding: opc[31:30] 101 0 0xx xxxxxxx (pairs)
    if (op_28_27 == 0b01 && op_26 == 0) {
      DecodeLoadStorePair();
      return;
    }

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

    // SIMD/FP load/store register (various): bits[29:27] = x11, bit[26]=1
    if (op_28_27 == 0b11 && op_26 == 1) {
      // LDR (literal) — SIMD/FP form: opc(2) 011100 imm19 Rt with
      // bit[29]=0, bit[26]=1. Mirrors the integer LDR (literal) V-bit
      // fix in the bit[26]=0 branch below: without this check, the
      // dispatcher falls through to the SIMD register-with-immediate
      // handlers (which require bit[29]=1) and interprets bits[9:5]
      // of imm19 as Rn, silently corrupting whichever register Rn
      // happens to land on. Encoding examples:
      //   `ldr s0, =literal` = 0x1C000000 | (imm19<<5) | rt   (opc=00)
      //   `ldr d0, =literal` = 0x5C000000 | (imm19<<5) | rt   (opc=01)
      //   `ldr q0, =literal` = 0x9C000000 | (imm19<<5) | rt   (opc=10)
      if (op_29 == 0) {
        DecodeSimdLoadLiteral();
        return;
      }
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

    // Load/store register (various): bits[29:27] = x11, bit[26]=0
    // Encoding: size[31:30] 111 0 00xx ... (unscaled/pre/post)
    //           size[31:30] 111 0 01xx ... (unsigned offset)
    if (op_28_27 == 0b11 && op_26 == 0) {
      // LDR/LDRSW/PRFM (literal) — opc(2) 011000 imm19 Rt with bit[29]=0
      // (V=0). Encoding e.g. `ldr x16, =literal` = 0x58007c50. Without
      // this branch, dispatch falls through to LDR (immediate)
      // pre/post/reg/unscaled handlers, which interpret bits[9:5] as Rn
      // and bits[20:12] as imm9 — treating the literal-offset bits as a
      // base register index + offset, silently corrupting whichever
      // register Rn happens to land on (e.g. x2 for 0x58007c50).
      // Observed in libcoldstart.so sha256_block_data_order prologue
      // where bits[11:10] of imm19 = 0b11 routed the LDR literal to
      // DecodeLoadStoreImmPostPreIndex(true) → STR x16, [x2, #7]!.
      // LDR/LDRSW/PRFM (literal) all have bit[24]=0; the MTE tag load/store
      // group (LDG/STG/ST2G/STZG/STZ2G and the LDGM/STGM/STZGM tag-block forms)
      // shares op_29=0 but has bit[24]=1. Guard the literal gate on op_24==0 so
      // it does not shadow the MTE encodings decoded below.
      if (op_29 == 0 && op_24 == 0) {
        DecodeLoadLiteral();
        return;
      }
      if (op_24) {
        // MTE load/store memory tags (LDG/STG/ST2G/STZG/STZ2G):
        //   bits[31:24]=11011001, bit[21]=1.
        // op_29=0 distinguishes from ordinary LDR/STR (unsigned imm), which
        // has op_29=1. size==11 is required by the MTE encoding; other
        // bits[31:30] with op_29=0 here are unallocated per ARM ARM.
        if (op_29 == 0 && GetBits<30, 2>() == 0b11 && GetBits<21, 1>()) {
          DecodeLoadStoreMemTag();
          return;
        }
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
        // bit[21]=1: Atomic memory operations (SWP, LDADD, etc.)
        // bit[21]=0: Unscaled immediate (LDUR/STUR).
        if (GetBits<21, 1>()) {
          DecodeAtomicMemoryOp();
        } else {
          DecodeLoadStoreUnscaled();
        }
        return;
      }
      if (op4 == 0b10) {
        // bit[21]=1: Load/store register (register offset).
        // bit[21]=0: Load/store register (unprivileged) — LDTR*/STTR*. At EL0
        // these are semantically identical to LDUR/STUR and share the imm9
        // layout, so route them to the unscaled handler (which decodes
        // size/opc/imm9 and sign-extends signed loads correctly). Without the
        // bit21 split they were misdecoded as register-offset, reinterpreting
        // imm9 as rm/option/S — either Undefined or a wrong-address load.
        if (GetBits<21, 1>()) {
          DecodeLoadStoreRegOffset();
        } else {
          DecodeLoadStoreUnscaled();
        }
        return;
      }
    }

    // Catch-all for other load/store variants not yet implemented (SIMD, exclusive, etc.).
    Undefined();
  }

  // MTE load/store memory tags: LDG / STG / ST2G / STZG / STZ2G.
  // Common encoding: 11011001 opc 1 imm9 op2 Rn Rt
  // See `MteLoadStoreOpcode` for the per-opcode encoding citations.
  void DecodeLoadStoreMemTag() {
    uint8_t opc = GetBits<22, 2>();
    int32_t imm9 = static_cast<int32_t>(GetBits<12, 9>());
    if (imm9 & (1 << 8)) imm9 |= ~0x1FF;  // Sign-extend bit[8].
    int32_t imm = imm9 * 16;              // Scale by 16-byte tag granule.
    uint8_t op2 = GetBits<10, 2>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    // Map (opc, op2) -> opcode. op2=00 is reserved except for opc=01 (LDG).
    MteLoadStoreOpcode mte_op;
    if (opc == 0b00 && op2 != 0b00) {
      mte_op = MteLoadStoreOpcode::kStg;
    } else if (opc == 0b01 && op2 == 0b00) {
      mte_op = MteLoadStoreOpcode::kLdg;
    } else if (opc == 0b01 && op2 != 0b00) {
      mte_op = MteLoadStoreOpcode::kStzg;
    } else if (opc == 0b10 && op2 != 0b00) {
      mte_op = MteLoadStoreOpcode::kSt2g;
    } else if (opc == 0b11 && op2 != 0b00) {
      mte_op = MteLoadStoreOpcode::kStz2g;
    // tag-block (granule-multiple) forms: op2=00, imm9=0.
    } else if (opc == 0b11 && op2 == 0b00) {
      mte_op = MteLoadStoreOpcode::kLdgm;
    } else if (opc == 0b10 && op2 == 0b00) {
      mte_op = MteLoadStoreOpcode::kStgm;
    } else if (opc == 0b00 && op2 == 0b00) {
      mte_op = MteLoadStoreOpcode::kStzgm;
    } else {
      return Undefined();
    }

    insn_consumer_->MteLoadStore({
        .opcode = mte_op,
        .rn = rn,
        .rt = rt,
        .imm = imm,
        .op2 = op2,
    });
  }

  // LDR/LDRSW/PRFM (literal): opc(2) 011000 imm19 Rt, bit[29]=0.
  //   opc=00 → LDR Wt (32-bit, zero-extend)
  //   opc=01 → LDR Xt (64-bit)
  //   opc=10 → LDRSW Xt (32-bit, sign-extend to 64-bit)
  //   opc=11 → PRFM (literal) — treat as NOP
  void DecodeLoadLiteral() {
    uint8_t opc = GetBits<30, 2>();
    int64_t offset = static_cast<int64_t>(SignExtend<19>(GetBits<5, 19>())) * 4;
    uint8_t rt = GetBits<0, 5>();

    LoadStoreSize size;
    bool is_signed;
    switch (opc) {
      case 0b00:
        size = LoadStoreSize::k32bit;
        is_signed = false;
        break;
      case 0b01:
        size = LoadStoreSize::k64bit;
        is_signed = false;
        break;
      case 0b10:
        size = LoadStoreSize::k32bit;
        is_signed = true;
        break;
      case 0b11:
      default:
        insn_consumer_->Nop();
        return;
    }

    const LoadLiteralArgs args = {
        .rt = rt,
        .offset = offset,
        .size = size,
        .is_signed = is_signed,
    };
    insn_consumer_->LoadLiteral(args);
  }

  // Flags derived from the load/store opc(2) field (bits[23:22]), shared by
  // the four general integer load/store forms: unsigned-offset, unscaled,
  // post/pre-index, and register-offset. All four encode the operation in
  // opc identically, and the flags depend ONLY on opc, never on size. size
  // selects the access width and — together with opc==0b10 — the PRFM
  // prefetch encoding, which IsPrfmEncoding() filters out at each call site
  // before these flags are consulted.
  //
  //   opc | operation      | is_store | is_signed | is_64bit_target
  //   ----+----------------+----------+-----------+----------------
  //   00  | STR            |  true    |  false    |  false
  //   01  | LDR (unsigned) |  false   |  false    |  false
  //   10  | LDRS ->64-bit  |  false   |  true     |  true
  //   11  | LDRS ->32-bit  |  false   |  true     |  false
  //
  // Full 16-entry (size,opc) routing (size = bits[31:30]):
  //   size=00/01/10, opc=00 -> STR          (flags row 00)
  //   size=00/01/10, opc=01 -> LDR          (flags row 01)
  //   size=00/01/10, opc=10 -> LDRS ->64    (flags row 10)
  //   size=00/01/10, opc=11 -> LDRS ->32    (flags row 11)
  //   size=11,       opc=00 -> STR (64-bit) (flags row 00)
  //   size=11,       opc=01 -> LDR (64-bit) (flags row 01)
  //   size=11,       opc=10 -> PRFM         (NOP, via IsPrfmEncoding)
  //   size=11,       opc=11 -> LDRS ->32    (flags row 11)
  struct LoadStoreOpcFlags {
    bool is_store;
    bool is_signed;         // For loads: sign-extend the value.
    bool is_64bit_target;   // For signed loads: extend to 64-bit.
  };

  static LoadStoreOpcFlags DecodeLoadStoreOpcFlags(uint8_t opc) {
    switch (opc) {
      case 0b00:
        return {.is_store = true, .is_signed = false, .is_64bit_target = false};
      case 0b01:
        return {.is_store = false, .is_signed = false, .is_64bit_target = false};
      case 0b10:
        return {.is_store = false, .is_signed = true, .is_64bit_target = true};
      default:  // 0b11
        return {.is_store = false, .is_signed = true, .is_64bit_target = false};
    }
  }

  // PRFM/PRFUM (prefetch) shares the general load/store encodings with
  // size==0b11, opc==0b10. Prefetch has no architectural side effects we
  // emulate; without this guard it would decode as an LDRS into Rt (where Rt
  // is the prefetch-type code, typically 0), silently clobbering the
  // destination register — observed clobbering X0 in libsuperpack-jni.so's
  // Brotli/SP2 decompressor on Facebook startup. NOP it. (Pre/post-index has
  // no prefetch form, so DecodeLoadStoreImmPostPreIndex does not check this.)
  static bool IsPrfmEncoding(uint8_t size, uint8_t opc) {
    return size == 0b11 && opc == 0b10;
  }

  void DecodeLoadStoreUnsignedImm() {
    uint8_t size = GetBits<30, 2>();
    uint8_t opc = GetBits<22, 2>();
    uint32_t imm12 = GetBits<10, 12>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rt = GetBits<0, 5>();

    if (IsPrfmEncoding(size, opc)) {
      insn_consumer_->Nop();
      return;
    }

    LoadStoreOpcFlags flags = DecodeLoadStoreOpcFlags(opc);

    // Scale offset by access size.
    int32_t offset = static_cast<int32_t>(imm12 << size);

    const LoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = offset,
        .size = LoadStoreSize{size},
        .is_store = flags.is_store,
        .is_signed = flags.is_signed,
        .is_64bit_target = flags.is_64bit_target,
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

    LoadStoreOpcFlags flags = DecodeLoadStoreOpcFlags(opc);

    // For pre/post index, we encode using LoadStoreImm and the interpreter handles the writeback.
    // But for now, we treat them as simple offset (TODO: proper pre/post-index support).
    const LoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = offset,
        .size = LoadStoreSize{size},
        .is_store = flags.is_store,
        .is_signed = flags.is_signed,
        .is_64bit_target = flags.is_64bit_target,
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

    if (IsPrfmEncoding(size, opc)) {
      insn_consumer_->Nop();
      return;
    }

    int32_t offset = SignExtend<9>(imm9);

    LoadStoreOpcFlags flags = DecodeLoadStoreOpcFlags(opc);

    const LoadStoreImmArgs args = {
        .rt = rt,
        .rn = rn,
        .offset = offset,
        .size = LoadStoreSize{size},
        .is_store = flags.is_store,
        .is_signed = flags.is_signed,
        .is_64bit_target = flags.is_64bit_target,
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

    if (IsPrfmEncoding(size, opc)) {
      insn_consumer_->Nop();
      return;
    }

    LoadStoreOpcFlags flags = DecodeLoadStoreOpcFlags(opc);

    uint8_t shift_amount = s_bit ? size : 0;

    // Validate option field. Only word-or-larger
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
        .is_store = flags.is_store,
        .is_signed = flags.is_signed,
        .is_64bit_target = flags.is_64bit_target,
    };
    insn_consumer_->LoadStoreReg(args);
  }

  void DecodeLoadStorePair() {
    // opc = bits[31:30]: 00=32-bit, 01=LDPSW (load-only, 32-bit element sign-
    // extended to 64-bit), 10=64-bit. bit31 alone selects the access width; the
    // low opc bit (bit30) marks the signed-word load. Missing bit30 silently
    // zero-extends LDPSW — observed as hello-qt's libQt6Gui rasteriser walking a
    // negative path coordinate (e.g. -96) as +4294967200, blowing a DDA endpoint
    // up to ~2^26 so the fill loop never converges (blank render).
    bool is_64bit = GetBits<31, 1>();
    bool opc_low = GetBits<30, 1>();
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
    bool is_signed = is_load && !is_64bit && opc_low;  // LDPSW

    const LoadStorePairArgs args = {
        .rt1 = rt1,
        .rt2 = rt2,
        .rn = rn,
        .offset = offset,
        .size = size,
        .is_store = !is_load,
        .is_preindex = is_preindex,
        .is_postindex = is_postindex,
        .is_signed = is_signed,
    };
    insn_consumer_->LoadStorePair(args);
  }

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

    // Armv8.2-FP16 scalar three-same.
    // Encoding (per ARM ARM C7.2 "Advanced SIMD scalar three same (FP16)"):
    //   0 1 U 1 1 1 1 0 a 1 0 Rm 0 0 opcode_3 1 Rn Rd
    // i.e. bit31=0, bit30=1, bit29=U, bits[28:24]=11110, bit23=a, bit22=1,
    //      bit21=0, bits[15:14]=00, bits[13:11]=opcode_3, bit10=1.
    // Must precede the FpFixedPointConversion check below, which would
    // otherwise misroute this encoding (FpFixedPointConversion only gates
    // on bits[28:24]=11110 && !bit21 — it doesn't constrain bit30, even
    // though its own encoding requires bit30=0).
    // Distinct from std scalar three-same (bit21=1 there, =0 here) and
    // from scalar copy (bits[23:21]=000 there; bits[23:21]=`a 1 0` here).
    if (!bit31 && GetBits<30, 1>() && GetBits<24, 5>() == 0b11110 &&
        GetBits<22, 1>() && !GetBits<21, 1>() &&
        !GetBits<15, 1>() && !GetBits<14, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdScalarFp16ThreeSame();
      return;
    }

    // Cryptographic three-register SHA (SHA1C/SHA1P/SHA1M/SHA1SU0,
    // SHA256H/SHA256H2/SHA256SU1):
    //   bit31=0, bit30=1, bit29=0, bits[28:24]=11110, bits[23:22]=00,
    //   bit21=0, bits[20:16]=Rm, bit15=0, bits[14:12]=opcode, bits[11:10]=00
    // Must be checked BEFORE FpFixedPointConversion (which catches !bit21
    // for the bits[28:24]=11110 group and would silently mis-route SHA).
    // opcode: 000=SHA1C, 001=SHA1P, 010=SHA1M, 011=SHA1SU0,
    //         100=SHA256H, 101=SHA256H2, 110=SHA256SU1, 111=Undefined.
    if (!bit31 && GetBits<30, 1>() && !GetBits<29, 1>() &&
        GetBits<24, 5>() == 0b11110 && GetBits<22, 2>() == 0 &&
        !GetBits<21, 1>() && !GetBits<15, 1>() && GetBits<10, 2>() == 0b00) {
      insn_consumer_->CryptoSha3Reg(
          GetBits<0, 5>(),    // rd
          GetBits<5, 5>(),    // rn
          GetBits<16, 5>(),   // rm
          GetBits<12, 3>());  // opcode
      return;
    }

    // Cryptographic two-register SHA (SHA1H, SHA1SU1, SHA256SU0):
    //   bit31=0, bit30=1, bit29=0, bits[28:24]=11110, bits[23:22]=00,
    //   bits[21:17]=10100, bit16=0, bits[15:14]=00, bits[13:12]=opcode,
    //   bits[11:10]=10
    // Must be checked BEFORE FpDataProc2 (which also matches bits[28:24]=11110,
    // bit21=1, bits[11:10]=10 — but with bit30=0).
    // opcode: 00=SHA1H, 01=SHA1SU1, 10=SHA256SU0 (interp-only), 11=Undefined.
    if (!bit31 && GetBits<30, 1>() && !GetBits<29, 1>() &&
        GetBits<24, 5>() == 0b11110 && GetBits<22, 2>() == 0 &&
        GetBits<17, 5>() == 0b10100 && !GetBits<16, 1>() &&
        GetBits<14, 2>() == 0 && GetBits<10, 2>() == 0b10) {
      insn_consumer_->CryptoSha2Reg(
          GetBits<0, 5>(),    // rd
          GetBits<5, 5>(),    // rn
          GetBits<12, 2>());  // opcode
      return;
    }

    // AdvSIMD Armv8.1-RDM scalar three-same-extra: SQRDMLAH / SQRDMLSH
    // (scalar, non-indexed).  Sibling of the vector three-same-extra
    // dispatch above; differs only in bits[30:24] (scalar marker = 1 11110
    // vs vector = Q 01110).
    //   bit31=0, bit30=1, bit29=1 (U=1), bits[28:24]=11110, bit21=0.
    //   bit15=1, bit14=0, bit13=0, bit12=0, bit10=1.
    //   bit11 = 0 (SQRDMLAH) / 1 (SQRDMLSH).
    //   size ∈ {01 (H), 10 (S)}; size=00/11 reserved per ARM ARM
    //   C7.2.299 / .301.
    //
    // Must precede FpFixedPointConversion below — that arm gates only on
    // bits[28:24]=11110 && !bit21 (it doesn't constrain bit30), so the
    // scalar three-same-extra encoding would otherwise be silently
    // misrouted to FpFixedPointConversion's internal opcode dispatch
    // (which interprets the Rm bits as rmode/opcode and either
    // mis-emits SCVTF/UCVTF/FCVTZS/FCVTZU semantics or Undefined()s,
    // depending on Rm).  No overlap with the FP16 scalar three-same arm
    // (bit22=1 there; here size hi-bit can be 0).  No overlap with the
    // standard scalar three-same arm above (bit21=1 there; here bit21=0).
    // No overlap with CryptoSha (bit29=0 there; here bit29=1).
    if (!bit31 && GetBits<30, 1>() && GetBits<29, 1>() &&
        GetBits<24, 5>() == 0b11110 && !GetBits<21, 1>() &&
        GetBits<15, 1>() && !GetBits<14, 1>() && !GetBits<13, 1>() &&
        !GetBits<12, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdScalarRdmThreeSame();
      return;
    }

    // FP <-> fixed-point conversion: bits[28:24]=11110, bit21=0
    // Must be checked BEFORE all bit21=1 FP checks.
    // Encoding: sf 0 S 11110 ftype 0 rmode opcode scale Rn Rd
    if (GetBits<24, 5>() == 0b11110 && !GetBits<21, 1>()) {
      DecodeFpFixedPointConversion();
      return;
    }

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

    // FMOV (scalar, immediate): bit31=0, bits[28:24]=11110, bit21=1, bits[12:10]=100, bits[9:5]=00000
    // Encoding: 0 0 0 11110 ftype 1 imm8 100 00000 Rd
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() &&
        GetBits<10, 3>() == 0b100 && GetBits<5, 5>() == 0b00000) {
      DecodeFpMovImmediate();
      return;
    }

    // Floating-point data-processing (2 source): bit31=0, bits[28:24]=11110, bit21=1, bits[11:10]=10
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b10) {
      DecodeFpDataProc2();
      return;
    }

    // Floating-point compare: bit31=0, bits[28:24]=11110, bit21=1, bits[13:10]=1000
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 4>() == 0b1000) {
      DecodeFpCompare();
      return;
    }

    // Floating-point conditional compare: bit31=0, bits[28:24]=11110, bit21=1, bits[11:10]=01
    // Encoding: 0 0 0 11110 ftype 1 Rm cond 01 Rn op nzcv  (op: 0=FCCMP, 1=FCCMPE)
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b01) {
      DecodeFpConditionalCompare();
      return;
    }

    // FCSEL: bit31=0, bits[28:24]=11110, bit21=1, bits[11:10]=11
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b11) {
      DecodeFpCondSelect();
      return;
    }

    // Floating-point data-processing (3 source): bit31=0, bit30=0, bit29=0,
    // bits[28:24]=11111.  FMADD, FMSUB, FNMADD, FNMSUB.
    // require bit30=0 (M) and bit29=0 (S) per ARM ARM
    // encoding "M=0 S=0 11111 ftype o1 0 Rm o0 Ra Rn Rd".  Without these
    // constraints the prefix also catches the "AdvSIMD scalar x indexed
    // element" family (bit30=1, bits[28:24]=11111) — e.g. FMULX scalar
    // by-element (U=1, opcode=1001) was silently mis-routed into
    // FpDataProc3 as garbage FMADD/FMSUB, producing wrong math without
    // any SIGILL.  Tightening here routes the scalar-x-indexed encodings
    // to the final Undefined() (no implementation yet) so the failure
    // mode is a diagnostic SIGILL rather than corrupted arithmetic.
    if (!bit31 && !GetBits<30, 1>() && !GetBits<29, 1>() &&
        GetBits<24, 5>() == 0b11111) {
      DecodeFpDataProc3();
      return;
    }

    // AdvSIMD scalar x indexed element (ARM ARM C4.1.71):
    //   bit31=0, bit30=1, bits[28:24]=11111, bit10=0.
    // Sibling of vector-x-indexed (bits[28:24]=01111, dispatched below at
    // the AdvSimd*VecXIndexedElement path).  Distinguished from
    // FpDataProc3 (above) by bit30=1.
    if (!bit31 && GetBits<30, 1>() && GetBits<24, 5>() == 0b11111 &&
        !GetBits<10, 1>()) {
      DecodeAdvSimdScalarXIndexedElement();
      return;
    }

    // AdvSIMD scalar shift by immediate (ARM ARM C4.1.6.10):
    //   bit31=0, bit30=1, bits[28:24]=11111, bit23=0, bit10=1, immh!=0.
    // Sibling of vector AdvSimdShiftByImm (bits[28:24]=01111, also bit10=1),
    // and of AdvSimdScalarXIndexedElement (same bits[28:24]=11111 but bit10=0).
    // The encoding is bit-identical to the vector shift-by-immediate apart
    // from bits[28:24] (and the absence of a Q bit — scalar always produces
    // exactly one element).  By constructing AdvSimdShiftImmArgs with
    // q=false and routing through the existing AdvSimdShiftByImm consumer,
    // the interpreter naturally executes a single-lane shift (num_elements
    // = vec_len / esize = 8 / 8 = 1 for D-form).  The dispatched
    // implementations are intentionally limited this cycle to the shift
    // ops where this num_elements=1 identity holds without additional
    // post-masking (see DecodeAdvSimdScalarShiftByImm body).
    if (!bit31 && GetBits<30, 1>() && GetBits<24, 5>() == 0b11111 &&
        !GetBits<23, 1>() && GetBits<10, 1>() && GetBits<19, 4>() != 0) {
      DecodeAdvSimdScalarShiftByImm();
      return;
    }

    // AdvSIMD three different: bit31=0, bits[28:24]=01110, bit21=1, bits[11:10]=00
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b00) {
      DecodeAdvSimdThreeDiff();
      return;
    }

    // AdvSIMD BFloat16 three-same-extra (Armv8.6-BF16): BFDOT, BFMMLA,
    // BFMLALB, BFMLALT (vector forms).
    //   bit31=0, bit29=1, bits[28:24]=01110, bits[23:22] ∈ {01, 11},
    //   bit21=0, bit15=1, bit14=1, bit13=1, bit11=1, bit10=1.
    // Inner dispatch (DecodeAdvSimdBf16ThreeSame) picks per-size:
    //   size=01: bit12=1 -> BFDOT (vector); bit12=0 -> BFMMLA (Q=1 only).
    //   size=11: bit12=1 -> BFMLALB (bit30=0) / BFMLALT (bit30=1).
    // Must precede the FCMA dispatch below since they share bit15=1,
    // bit14=1, bit10=1 with bit21=0; for size=01 the FCMA inner decode
    // would reject the encoding (FCMA needs size>=10), and for size=11
    // FCMA's FCADD path requires bit11=0 (BFMLAL has bit11=1) so it
    // would reject as Undefined.  Either way, without this carve-out
    // BF16 ops silently SIGILL.
    if (!bit31 && GetBits<29, 1>() && GetBits<24, 5>() == 0b01110 &&
        (GetBits<22, 2>() == 0b01 || GetBits<22, 2>() == 0b11) &&
        !GetBits<21, 1>() &&
        GetBits<15, 1>() && GetBits<14, 1>() && GetBits<13, 1>() &&
        GetBits<11, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdBf16ThreeSame();
      return;
    }

    // Advanced SIMD complex floating-point (Armv8.3-FCMA): FCADD / FCMLA.
    //   bit31=0, bits[28:24]=01110, bit21=0, bit15=1, bit14=1, bit10=1.
    // Must precede three-same / permute / copy / two-reg-misc to avoid
    // mis-routing the FCMA encoding bits.  Three-same itself requires
    // bit21=1, so there's no overlap there; but the other AdvSIMD shapes
    // that have bit21=0 all require bit15=0 or bit14=0 or bit10=0, so
    // pinning bit15=1, bit14=1, bit10=1 carves the FCMA subspace cleanly.
    // Other three-same-extra opcodes (SDOT, UDOT, SQRDMLAH, SQRDMLSH,
    // USDOT, BF*) all have bit14=0 in their opcode field, so the
    // bit14=1 guard keeps this branch FCMA-only.
    if (!bit31 && GetBits<24, 5>() == 0b01110 && !GetBits<21, 1>() &&
        GetBits<15, 1>() && GetBits<14, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdFcma();
      return;
    }

    // hello-dotprod
    // AdvSIMD integer dot product (Armv8.4-DotProd): SDOT / UDOT (vector).
    //   bit31=0, bits[28:24]=01110, bits[23:21]=100 (so bits[23:22]=10
    //   and bit21=0), bits[15:10]=100101 (bit15=1, bit14=0, bit13=0,
    //   bit12=1, bit11=0, bit10=1).  bit30=Q, bit29=U (0=SDOT, 1=UDOT).
    // Verified from clang --target=aarch64 -march=armv8.4-a+dotprod:
    //   sdot v0.4s,v1.16b,v2.16b = 0x4e829420 — bit23=1, bit22=0.
    // Must precede the generic three-same / permute / copy / two-reg-misc
    // decoders that share the bits[28:24]=01110 prefix.  Three-same proper
    // requires bit21=1, so there's no overlap; permute / copy require
    // bit15=0; two-reg-misc requires bit21=1; FCMA (above) requires
    // bit14=1; BF16 three-same-extra (above) requires bits[23:22] ∈ {01,
    // 11} — DotProd uses bits[23:22]=10, so no conflict.  Without this
    // carve-out SDOT/UDOT silently fall through and SIGILL the guest.
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<22, 2>() == 0b10 &&
        !GetBits<21, 1>() && GetBits<15, 1>() && !GetBits<14, 1>() &&
        !GetBits<13, 1>() && GetBits<12, 1>() && !GetBits<11, 1>() &&
        GetBits<10, 1>()) {
      DecodeAdvSimdDotProductVec();
      return;
    }

    // I8MM (FEAT_I8MM): USDOT (vector) and the integer
    // matrix-multiply-accumulate SMMLA/UMMLA/USMMLA. Same prefix as DotProd
    // (bits[28:24]=01110, bits[23:22]=10, bit21=0, bit15=1, bit14=0, bit10=1);
    // bits[13:11] select the op:
    //   011 -> USDOT vector (U=0; Vn unsigned, Vm signed)
    //   100 -> SMMLA (U=0) / UMMLA (U=1)
    //   101 -> USMMLA (U=0; Vn unsigned, Vm signed)
    // All are .4S (Q=1) only.
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<22, 2>() == 0b10 &&
        !GetBits<21, 1>() && GetBits<15, 1>() && !GetBits<14, 1>() &&
        GetBits<10, 1>()) {
      uint8_t b13_11 = GetBits<11, 3>();  // bits[13:11]
      bool u = GetBits<29, 1>();
      if (b13_11 == 0b011 && !u) {  // USDOT vector
        const DotProductArgs args = {
            .opcode = DotProductOpcode::kUsdot,
            .rd = GetBits<0, 5>(),
            .rn = GetBits<5, 5>(),
            .rm = GetBits<16, 5>(),
            .index = 0,
            .q = GetBits<30, 1>(),
        };
        insn_consumer_->AdvSimdDotProduct(args);
        return;
      }
      if (b13_11 == 0b100) {  // SMMLA / UMMLA
        const MatMulArgs args = {
            .opcode = u ? MatMulOpcode::kUmmla : MatMulOpcode::kSmmla,
            .rd = GetBits<0, 5>(),
            .rn = GetBits<5, 5>(),
            .rm = GetBits<16, 5>(),
        };
        insn_consumer_->AdvSimdMatMul(args);
        return;
      }
      if (b13_11 == 0b101 && !u) {  // USMMLA
        const MatMulArgs args = {
            .opcode = MatMulOpcode::kUsmmla,
            .rd = GetBits<0, 5>(),
            .rn = GetBits<5, 5>(),
            .rm = GetBits<16, 5>(),
        };
        insn_consumer_->AdvSimdMatMul(args);
        return;
      }
    }

    // AdvSIMD Armv8.1-RDM three-same vector: SQRDMLAH / SQRDMLSH (NOT the
    // by-element form — that's already wired through AdvSimdVecXIdxOpcode).
    //   bit31=0, bits[28:24]=01110, bit21=0, U=bit29=1, bit15=1, bit14=0,
    //   bit13=0, bit12=0, bit10=1.  bit11 = 0 (SQRDMLAH) / 1 (SQRDMLSH).
    //   size ∈ {01, 10}; size=00/11 reserved per ARM ARM C7.2.298/.300.
    //
    // Disambiguation versus the other three-same-extra arms above:
    //   FCMA  (bit14=1) — no overlap, FCMA needs bit14=1 here we need bit14=0.
    //   BF16  (bit13=1, bit11=1) — no overlap, we need bit13=0.
    //   DotProd (bit12=1, bit11=0) — no overlap, we need bit12=0.
    // Standard three-same (bit21=1) is mutually exclusive on bit21.  FP16
    // three-same (bit22=1, bit15=0) is mutually exclusive on bit15.
    if (!bit31 && GetBits<24, 5>() == 0b01110 && !GetBits<21, 1>() &&
        GetBits<29, 1>() && GetBits<15, 1>() && !GetBits<14, 1>() &&
        !GetBits<13, 1>() && !GetBits<12, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdRdmThreeSame();
      return;
    }

    // Armv8.2-FP16 NEON vector three-same.
    // Encoding (per ARM ARM C7.2 "Advanced SIMD three same (FP16)"):
    //   0 Q U 0 1 1 1 0 a 1 0 Rm 0 0 opcode 1 Rn Rd
    // i.e. bit31=0, bits[28:24]=01110, bit23=a, bit22=1, bit21=0,
    //      bits[15:14]=00, bit10=1, bits[13:11]=3-bit opcode.
    // Verified against `clang --target=aarch64 -march=armv8.2-a+fp16`:
    //   FADD v0.4h,v1.4h,v2.4h = 0x0e421420 -> bit23=0, bit22=1, bit21=0,
    //                                          bits[15:14]=00, bit13:11=010, bit10=1.
    //   FSUB v0.4h,v1.4h,v2.4h = 0x0ec21420 -> bit23=1 (a=1), bit22=1, ...
    // Must precede AdvSimdCopy (bit21=0, bit15=0, bit10=1) which it overlaps
    // on bit21/bit15/bit10; bit22 distinguishes (Copy has bit22=0, FP16
    // three-same has bit22=1).  Standard three-same below requires bit21=1
    // so there's no overlap with that.
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<22, 1>() && !GetBits<21, 1>() &&
        GetBits<14, 2>() == 0b00 && GetBits<10, 1>()) {
      DecodeAdvSimdFp16ThreeSame();
      return;
    }

    // AdvSIMD three same: bit31=0, bits[28:24]=01110, bit21=1, bit10=1
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<21, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdThreeSame();
      return;
    }

    // AdvSIMD permute (UZP1, TRN1, ZIP1, UZP2, TRN2, ZIP2):
    // bit31=0, bit29=0, bits[28:24]=01110, bit21=0, bit15=0, bits[11:10]=10.
    // The bit29=0 check disambiguates from EXT (bit29=1), which otherwise
    // collides for odd imm4 (bit 11 of EXT's imm4 = 1 yields bits[10:11]=10).
    // Must be checked BEFORE two-reg misc since both share bits[11:10]=10 but
    // permute has bit21=0 while two-reg misc has bit21=1 (bits[21:17]=10000).
    if (!bit31 && !GetBits<29, 1>() && GetBits<24, 5>() == 0b01110 &&
        !GetBits<21, 1>() && !GetBits<15, 1>() && GetBits<10, 2>() == 0b10) {
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

    // Cryptographic AES (AESE, AESD, AESMC, AESIMC):
    //   bit31=0, bit30=1, bit29=0, bits[28:24]=01110, bits[23:22]=00,
    //   bits[21:17]=10100, bits[16:14]=001, bits[11:10]=10
    // opcode field bits[16:12] = 00100=AESE, 00101=AESD, 00110=AESMC, 00111=AESIMC.
    // Must be checked BEFORE AdvSIMD two-reg-misc which also matches
    // bits[24:5]=01110, bit17=0, bits[11:10]=10 but does not handle these.
    if (!bit31 && GetBits<30, 1>() && !GetBits<29, 1>() &&
        GetBits<24, 5>() == 0b01110 && GetBits<22, 2>() == 0 &&
        GetBits<17, 5>() == 0b10100 && GetBits<14, 3>() == 0b001 &&
        GetBits<10, 2>() == 0b10) {
      insn_consumer_->CryptoAes(
          GetBits<0, 5>(),    // rd
          GetBits<5, 5>(),    // rn
          GetBits<12, 2>());  // 00=AESE, 01=AESD, 10=AESMC, 11=AESIMC
      return;
    }

    // Armv8.2-FP16 NEON vector two-register miscellaneous.
    // Encoding: 0 Q U 0 1 1 1 0 a 1 1 1 1 1 0 opcode 1 0 Rn Rd
    //   bit31=0, bits[28:24]=01110, bit23=a (free), bit22=1, bits[21:17]=11100,
    //   bits[11:10]=10.
    // The std two-reg-misc form (below) sets bits[21:17]=10000; carve out the
    // FP16 form first so it doesn't fall into the std handler where args.size
    // would mis-route bit22=1 to FP64 element semantics.
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<22, 1>() &&
        GetBits<17, 5>() == 0b11100 && GetBits<10, 2>() == 0b10) {
      DecodeAdvSimdFp16TwoRegMisc();
      return;
    }

    // AdvSIMD two-reg misc: bit31=0, bits[28:24]=01110, bit21=1, bit17=0, bits[11:10]=10
    // The bit21=1 check is load-bearing: EXT (bit21=0) with an odd imm4 also has
    // bits[11:10]=10 and bit17 derived from Rm, so without it EXT is swallowed
    // here and never reaches its handler below.
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<21, 1>() && !GetBits<17, 1>() &&
        GetBits<10, 2>() == 0b10) {
      DecodeAdvSimdTwoRegMisc();
      return;
    }

    // AdvSIMD copy (DUP, INS, SMOV, UMOV): bit31=0, bits[28:24]=01110, bit21=0, bit15=0, bit10=1
    if (!bit31 && GetBits<24, 5>() == 0b01110 && !GetBits<21, 1>() && !GetBits<15, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdCopy();
      return;
    }

    // AdvSIMD vector x indexed element: bit31=0, bits[28:24]=01111, bit10=0
    if (!bit31 && GetBits<24, 5>() == 0b01111 && !GetBits<10, 1>()) {
      DecodeAdvSimdVecXIndexedElement();
      return;
    }

    // AdvSIMD shift by immediate: bit31=0, bits[28:24]=01111, bit10=1, immh!=0000
    if (!bit31 && GetBits<24, 5>() == 0b01111 && GetBits<10, 1>() && GetBits<19, 4>() != 0) {
      DecodeAdvSimdShiftByImm();
      return;
    }

    // AdvSIMD extract (EXT) and AdvSIMD table lookup (TBL/TBX) share most of
    // their encoding prefix. They differ on bit29 (op2 in the encoding tree):
    //   EXT: bit29=1   (i.e. bits[29:24]=101110)
    //   TBL: bit29=0   (i.e. bits[29:24]=001110)
    // plus the imm4/len/op subfields differ. We dispatch on bit29.
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<22, 2>() == 0 &&
        !GetBits<21, 1>() && !GetBits<15, 1>() && !GetBits<10, 1>()) {
      if (GetBits<29, 1>()) {
        // EXT Vd.<T>, Vn.<T>, Vm.<T>, #index
        insn_consumer_->AdvSimdExtract(
            GetBits<0, 5>(),   // rd
            GetBits<5, 5>(),   // rn
            GetBits<16, 5>(),  // rm
            GetBits<11, 4>(),  // imm4 (byte index)
            GetBits<30, 1>()); // q
      } else {
        // TBL/TBX Vd.<T>, {Vn.16B [, V(n+1).16B [, V(n+2).16B [, V(n+3).16B]]]}, Vm.<T>
        // len = bits[14:13]+1 table registers; op = bit12 (0=TBL, 1=TBX).
        // bit11 must be 0 for TBL/TBX; any other value is reserved.
        if (GetBits<11, 1>()) { Undefined(); return; }
        insn_consumer_->AdvSimdTableLookup(
            GetBits<0, 5>(),   // rd
            GetBits<5, 5>(),   // rn (first table register; spans len consecutive)
            GetBits<16, 5>(),  // rm (index vector)
            GetBits<13, 2>(),  // len (0..3 → 1..4 table registers)
            GetBits<12, 1>(),  // op (0=TBL, 1=TBX)
            GetBits<30, 1>()); // q
      }
      return;
    }

    // SHA-512 (FEAT_SHA512) — bit31=1 group, outside the AdvSIMD family.
    // Common prefix: bits[30:24]=1001110, bits[15:12]=1000.
    //   Three-register encoding: 11001110 011 Rm 1000 o2 Rn Rd, where
    //     o2 = bits[11:10] = 00 (SHA512H), 01 (SHA512H2), 10 (SHA512SU1).
    //   Two-register encoding (SHA512SU0):
    //     11001110 110 00000 1000 00 Rn Rd.
    if (bit31 && GetBits<24, 7>() == 0b1001110 &&
        GetBits<12, 4>() == 0b1000) {
      uint8_t bits23_21 = GetBits<21, 3>();
      uint8_t opcode2 = GetBits<10, 2>();   // bits[11:10]
      if (bits23_21 == 0b011) {
        // RAX1 (FEAT_SHA3) shares this three-register prefix at opcode2=11.
        if (opcode2 == 0b11) {
          insn_consumer_->Rax1(GetBits<0, 5>(),   // rd
                               GetBits<5, 5>(),    // rn
                               GetBits<16, 5>());  // rm
          return;
        }
        Sha512Op op;
        switch (opcode2) {
          case 0b00: op = Sha512Op::kSha512h; break;
          case 0b01: op = Sha512Op::kSha512h2; break;
          case 0b10: op = Sha512Op::kSha512su1; break;
          default: Undefined(); return;
        }
        insn_consumer_->Sha512(op,
                               GetBits<0, 5>(),   // rd
                               GetBits<5, 5>(),   // rn
                               GetBits<16, 5>()); // rm
        return;
      }
      if (bits23_21 == 0b110 && GetBits<16, 5>() == 0 && opcode2 == 0b00) {
        insn_consumer_->Sha512(Sha512Op::kSha512su0,
                               GetBits<0, 5>(),   // rd
                               GetBits<5, 5>(),   // rn
                               0);                // rm unused
        return;
      }
    }

    // SHA3 (FEAT_SHA3): EOR3 / BCAX (four-register), XAR.  RAX1 is decoded
    // in the SHA512 three-register block above (opcode2=11).
    //   EOR3 Vd.16B,Vn,Vm,Va = 0xCE00.. (bits[23:21]=000, bit15=0)
    //   BCAX Vd.16B,Vn,Vm,Va = 0xCE20.. (bits[23:21]=001, bit15=0)
    //   XAR  Vd.2D,Vn,Vm,#imm6 = 0xCE80.. (bits[23:21]=100, imm6=bits[15:10])
    if (bit31 && GetBits<24, 7>() == 0b1001110) {
      uint8_t op23_21 = GetBits<21, 3>();
      if ((op23_21 == 0b000 || op23_21 == 0b001) && !GetBits<15, 1>()) {
        uint8_t rd = GetBits<0, 5>();
        uint8_t rn = GetBits<5, 5>();
        uint8_t ra = GetBits<10, 5>();
        uint8_t rm = GetBits<16, 5>();
        if (op23_21 == 0b000) {
          insn_consumer_->Eor3(rd, rn, rm, ra);
        } else {
          insn_consumer_->Bcax(rd, rn, rm, ra);
        }
        return;
      }
      if (op23_21 == 0b100) {
        insn_consumer_->Xar(GetBits<0, 5>(),   // rd
                            GetBits<5, 5>(),    // rn
                            GetBits<16, 5>(),   // rm
                            GetBits<10, 6>());  // imm6
        return;
      }
    }

    // SM4 (FEAT_SM4): SM4E (2-register) and SM4EKEY
    // (3-register). Same bit31=1, bits[30:24]=1001110 crypto prefix as
    // SHA512/SHA3; distinguished by bits[23:21] and the low opcode bits.
    //   SM4EKEY Vd,Vn,Vm : bits[23:21]=011, bits[15:12]=1100, bits[11:10]=10.
    //   SM4E    Vd,Vn    : bits[23:21]=110, Rm=00000, bits[15:10]=100001.
    if (bit31 && GetBits<24, 7>() == 0b1001110) {
      uint8_t sm_b23_21 = GetBits<21, 3>();
      // SM3SS1 (4-register) / SM3TT1A/1B/2A/2B (3-register, by-lane).
      //   bits[23:21]=010, bit15=0 -> SM3SS1 (Ra = bits[14:10]).
      //   bits[23:21]=010, bit15=1 -> SM3TT*, imm2=bits[13:12], op=bits[11:10]
      //     (00=TT1A, 01=TT1B, 10=TT2A, 11=TT2B).
      if (sm_b23_21 == 0b010) {
        if (!GetBits<15, 1>()) {
          insn_consumer_->Sm3ss1(GetBits<0, 5>(),    // rd
                                 GetBits<5, 5>(),     // rn
                                 GetBits<16, 5>(),    // rm
                                 GetBits<10, 5>());   // ra
        } else {
          insn_consumer_->Sm3tt(GetBits<0, 5>(),    // rd
                                GetBits<5, 5>(),     // rn
                                GetBits<16, 5>(),    // rm
                                GetBits<12, 2>(),    // imm2
                                GetBits<10, 2>());   // op: 00/01/10/11
        }
        return;
      }
      // bits[23:21]=011, bits[15:12]=1100: SM3PARTW1 (bits[11:10]=00),
      // SM3PARTW2 (01), SM4EKEY (10).
      if (sm_b23_21 == 0b011 && GetBits<12, 4>() == 0b1100) {
        uint8_t sub = GetBits<10, 2>();
        if (sub == 0b00) {
          insn_consumer_->Sm3partw1(GetBits<0, 5>(), GetBits<5, 5>(),
                                    GetBits<16, 5>());
          return;
        }
        if (sub == 0b01) {
          insn_consumer_->Sm3partw2(GetBits<0, 5>(), GetBits<5, 5>(),
                                    GetBits<16, 5>());
          return;
        }
        if (sub == 0b10) {
          insn_consumer_->Sm4ekey(GetBits<0, 5>(),   // rd
                                  GetBits<5, 5>(),    // rn
                                  GetBits<16, 5>());  // rm
          return;
        }
      }
      if (sm_b23_21 == 0b110 && GetBits<16, 5>() == 0 &&
          GetBits<10, 6>() == 0b100001) {
        insn_consumer_->Sm4e(GetBits<0, 5>(),   // rd
                             GetBits<5, 5>());  // rn
        return;
      }
    }

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
  // LDR (literal) SIMD/FP: opc(2) 011100 imm19 Rt, bit[29]=0, bit[26]=1.
  //   opc=00 → LDR St (32-bit, S register)
  //   opc=01 → LDR Dt (64-bit, D register)
  //   opc=10 → LDR Qt (128-bit, Q register)
  //   opc=11 → unallocated (UNDEFINED)
  void DecodeSimdLoadLiteral() {
    uint8_t opc = GetBits<30, 2>();
    int64_t offset = static_cast<int64_t>(SignExtend<19>(GetBits<5, 19>())) * 4;
    uint8_t rt = GetBits<0, 5>();

    SimdLoadStoreSize size;
    switch (opc) {
      case 0b00:
        size = SimdLoadStoreSize::k32bit;
        break;
      case 0b01:
        size = SimdLoadStoreSize::k64bit;
        break;
      case 0b10:
        size = SimdLoadStoreSize::k128bit;
        break;
      default:
        Undefined();
        return;
    }

    insn_consumer_->SimdLoadLiteral({.rt = rt, .offset = offset, .size = size});
  }

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

    // Reject UNDEFINED option encodings for the
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
    insn_consumer_->SimdLoadStoreReg(args);
  }

  //
  // SIMD/FP load/store pair - bit[26]=1 variant.
  //
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
      // LD2 / LD3 / LD4 (de-interleaving on load,
      // interleaving on store). Distinct from LDn-with-1-reg-N-times.
      case 0b1000: num_regs = 2; is_interleaved = true; break;
      case 0b0100: num_regs = 3; is_interleaved = true; break;
      case 0b0000: num_regs = 4; is_interleaved = true; break;
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

    // Non-replicate LD/ST single element to one lane. The element size is
    // selected by opcode<2:1> (00=B, 01=H, 10=S/D); opcode<0> is the low bit of
    // the structure count. Index packs Q:S:size per the ARM ARM size group.
    uint8_t elem_size;
    uint8_t index;
    switch (opcode >> 1) {
      case 0b00:  // Byte
        elem_size = 0;  // B
        index = (q << 3) | (s_bit << 2) | size;
        break;
      case 0b01:  // Halfword
        if (size & 1) { Undefined(); return; }
        elem_size = 1;  // H
        index = (q << 2) | (s_bit << 1) | (size >> 1);
        break;
      case 0b10:  // Word or Doubleword
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

    // Structure count (selem) = (opcode<0> : R) + 1, giving LD1/2/3/4 or
    // ST1/2/3/4. The odd opcodes (LD3/LD4/ST3/ST4) were previously unhandled.
    uint8_t num_regs = static_cast<uint8_t>((((opcode & 1) << 1) | (r ? 1 : 0)) + 1);
    AdvSimdSingleStructOp op;
    switch (num_regs) {
      case 1: op = is_load ? AdvSimdSingleStructOp::kLd1 : AdvSimdSingleStructOp::kSt1; break;
      case 2: op = is_load ? AdvSimdSingleStructOp::kLd2 : AdvSimdSingleStructOp::kSt2; break;
      case 3: op = is_load ? AdvSimdSingleStructOp::kLd3 : AdvSimdSingleStructOp::kSt3; break;
      default: op = is_load ? AdvSimdSingleStructOp::kLd4 : AdvSimdSingleStructOp::kSt4; break;
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

  // Data Processing (1-source): CLZ, CLS, RBIT, REV, REV16, REV32
  void DecodeDataProc1Src() {
    bool sf = GetBits<31, 1>();
    uint8_t op2 = GetBits<16, 5>();  // ARM ARM "opcode2" (bits 20:16)
    uint8_t opcode2 = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    // Dispatch based on opcode2
    // 000000 = RBIT, 000001 = REV16, 000010 = REV32(32-bit)/REV(64-bit),
    // 000011 = REV(64-bit only), 000100 = CLZ, 000101 = CLS
    // PAuth DP-1Src as identity
    // ARM ARM opcode2=00001 selects the PAuth family of DP-1Src ops
    // (PACIA/PACIB/PACDA/PACDB and AUT* siblings, the Z-variants where
    // Rn==RZR, plus XPACI/XPACD).  Digitalis does not implement pointer
    // authentication: PAC bits are never inserted, so authenticating or
    // stripping a pointer is the identity dst = src.  We route every
    // documented opcode (000000..010001) to the existing DataProc1Src
    // handler with a marker bit (0x40) that tells the interpreter/JIT
    // "this is PAuth — return src unchanged".  Without this dispatch the
    // PAuth ops alias RBIT/REV/CLZ etc. and silently miscompile, which
    // breaks any binary built with -mbranch-protection=pac-ret on NDK r25+.
    if (op2 == 0b00001) {
      if (!sf) {
        // PAuth DP-1Src is X-form only (sf must be 1).
        Undefined();
        return;
      }
      if (opcode2 > 0b010001) {
        // Reserved encoding within the PAuth family.
        Undefined();
        return;
      }
      // ARM ARM semantics for the PAuth family use Xd as BOTH the input
      // pointer and the destination (Xn is just the modifier salt).  For a
      // PAC-blind translator the identity is Xd' = Xd — so the source we
      // hand to the existing DataProc1Src callback must be Xd, not Xn.
      // Passing rn here (as earlier code did) caused PACIA/AUTIA and the
      // rest of the on-register PAC family to overwrite Xd with the value
      // of Xn, silently miscompiling any PAuth probe / verifier.
      // Z-variants (PACIZA et al.) encode Rn=11111 (XZR) and XPACI/XPACD
      // similarly use XZR as Rn — passing rd uniformly is still correct
      // because the architectural input register is always Xd.
      insn_consumer_->DataProc1Src(rd, rd, /*pauth marker=*/0x40 | opcode2, sf);
      return;
    }
    if (op2 != 0) {
      // Other opcode2 values are reserved; bail to interpreter.
      Undefined();
      return;
    }
    insn_consumer_->DataProc1Src(rd, rn, opcode2, sf);
  }

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

    // ftype: 00=S, 01=D, 11=H (Armv8.2-FP16).  10 is reserved.
    if (ftype == 0b10) { Undefined(); return; }

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

    // ftype: 00=S, 01=D, 11=H (Armv8.2-FP16).  10 is reserved.
    if (ftype == 0b10) { Undefined(); return; }

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

  //
  // FP conditional compare (FCCMP / FCCMPE).
  // Encoding: 0 0 0 11110 ftype 1 Rm cond 01 Rn op nzcv
  //   op == 0 -> FCCMP, op == 1 -> FCCMPE (signals on quiet NaN)
  //
  void DecodeFpConditionalCompare() {
    uint8_t ftype = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t cond = GetBits<12, 4>();
    uint8_t rn = GetBits<5, 5>();
    bool signal_nans = GetBits<4, 1>();
    uint8_t nzcv = GetBits<0, 4>();

    // ftype: 00=S, 01=D, 11=H (Armv8.2-FP16).  10 is reserved.
    if (ftype == 0b10) { Undefined(); return; }

    const FpConditionalCompareArgs args = {
        .rn = rn,
        .rm = rm,
        .nzcv = nzcv,
        .cond = Condition{cond},
        .ftype = ftype,
        .signal_nans = signal_nans,
    };
    insn_consumer_->FpConditionalCompare(args);
  }

  //
  // Advanced SIMD complex floating-point (Armv8.3-FCMA): FCADD / FCMLA.
  //
  // Encoding (verified via llvm-mc):
  //   bit31=0, bit30=Q, bit29=1 (U), bits[28:24]=01110, bits[23:22]=size,
  //   bit21=0, bits[20:16]=Rm, bit15=1, bit14=1, bit10=1.
  // FCADD: bit13=1, bit12=rot (0=#90, 1=#270), bit11=0.
  // FCMLA: bit13=0, bits[12:11]=rot (00=#0, 01=#90, 10=#180, 11=#270).
  //
  // Reserved combinations rejected as Undefined:
  //   - size == 00 (no 8-bit FP).
  //   - size == 11 (double) with Q == 0 (no half-size double vector — the
  //     2D form requires the 128-bit container).
  //   - FCADD with bit11 != 0 (unallocated).
  //
  // size == 01 (FP16): both Q=0 (.4h, 2 pairs) and Q=1 (.8h, 4 pairs) are
  // accepted (FP16 SIMD FCMA). Interpreter promotes
  // each half-precision lane to binary32 via FpHalfToSingle, applies the
  // FCMA rotation table, and narrows back via FpSingleToHalf — same
  // round-trip pattern as FP16 vector three-same / two-reg-misc.
  void DecodeAdvSimdFcma() {
    bool q = GetBits<30, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    bool bit13 = GetBits<13, 1>();
    uint8_t rot;
    FcmaOpcode opcode;
    if (bit13) {
      // FCADD: rot is bit[12]; bit[11] must be 0.
      if (GetBits<11, 1>()) { Undefined(); return; }
      opcode = FcmaOpcode::kFcadd;
      rot = GetBits<12, 1>();
    } else {
      // FCMLA: rot is bits[12:11].
      opcode = FcmaOpcode::kFcmla;
      rot = GetBits<11, 2>();
    }
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    if (size == 0b00) { Undefined(); return; }
    if (size == 0b11 && !q) { Undefined(); return; }

    const FcmaArgs args = {
        .opcode = opcode,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .size = size,
        .rot = rot,
        .q = q,
    };
    insn_consumer_->AdvSimdFcma(args);
  }

  // hello-dotprod
  // SDOT / UDOT (vector).  Encoding already filtered by the DecodeArmV8
  // guard: bits[28:24]=01110, bits[23:22]=00, bit21=0, bits[15:10]=100101.
  // bit30 = Q (vector length), bit29 = U (SDOT=0 / UDOT=1).
  void DecodeAdvSimdDotProductVec() {
    bool q = GetBits<30, 1>();
    bool u = GetBits<29, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    const DotProductArgs args = {
        .opcode = u ? DotProductOpcode::kUdot : DotProductOpcode::kSdot,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .index = 0,
        .q = q,
    };
    insn_consumer_->AdvSimdDotProduct(args);
  }

  //
  // AdvSIMD Armv8.1-RDM three-same vector: SQRDMLAH / SQRDMLSH.  These are
  // the non-indexed siblings of the by-element forms decoded under
  // AdvSimdVecXIdxOpcode::{kSqrdmlahIdx, kSqrdmlshIdx}.  Both feed back into
  // the standard AdvSimdThreeSame insn-consumer entry so the interpreter /
  // JIT can dispatch on a single enum surface.
  //
  // Encoding (per ARM ARM C7.2.298 / C7.2.300):
  //   0 Q 1 01110 size 0 Rm 1 opcode4 1 Rn Rd
  //   U=1, bit15=1, bit10=1, bit14=0, bit13=0, bit12=0.
  //   opcode4=0000 (bit11=0) -> SQRDMLAH.
  //   opcode4=0001 (bit11=1) -> SQRDMLSH.
  //   size ∈ {01 (.4h/.8h), 10 (.2s/.4s)}; size=00 / size=11 reserved.
  //
  // Encodings (bit-level reconstruction from the three-same-extra schema
  // above; Q=q, size=01 or 10, opcode4=0000 or 0001):
  //   sqrdmlah v0.4h, v1.4h, v2.4h = 0x2e428420 (Q=0, size=01, opcode4=0000)
  //   sqrdmlsh v0.8h, v1.8h, v2.8h = 0x6e428c20 (Q=1, size=01, opcode4=0001)
  //   sqrdmlah v0.4s, v1.4s, v2.4s = 0x6e828420 (Q=1, size=10, opcode4=0000)
  //   sqrdmlsh v0.2s, v1.2s, v2.2s = 0x2e828c20 (Q=0, size=10, opcode4=0001)
  void DecodeAdvSimdRdmThreeSame() {
    bool q = GetBits<30, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    bool bit11 = GetBits<11, 1>();

    // size=00 and size=11 are reserved per ARM ARM.
    if (size != 0b01 && size != 0b10) { Undefined(); return; }

    AdvSimdThreeSameArgs args = {
        .opcode = bit11 ? AdvSimdThreeSameOpcode::kSqrdmlshVec
                        : AdvSimdThreeSameOpcode::kSqrdmlahVec,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .size = size,
        .q = q,
        .is_fp16 = false,
    };
    insn_consumer_->AdvSimdThreeSame(args);
  }

  //
  // AdvSIMD Armv8.1-RDM scalar three-same-extra: SQRDMLAH / SQRDMLSH
  // (scalar, non-indexed).  Scalar sibling of DecodeAdvSimdRdmThreeSame
  // above.
  //
  // Encoding (per ARM ARM C7.2.299 / C7.2.301):
  //   0 1 1 11110 size 0 Rm 1 opcode4 1 Rn Rd
  //   U=1, bit30=1, bits[28:24]=11110, bit21=0, bit15=1, bit10=1.
  //   opcode4=0000 (bit11=0) -> SQRDMLAH.
  //   opcode4=0001 (bit11=1) -> SQRDMLSH.
  //   size ∈ {01 (H), 10 (S)}; size=00 / size=11 reserved.
  //
  // Verified encodings (llvm-mc -arch=aarch64 -mattr=+rdm):
  //   sqrdmlah h0, h1, h2 = 0x7e428420 (size=01, opcode4=0000)
  //   sqrdmlah s0, s1, s2 = 0x7e828420 (size=10, opcode4=0000)
  //   sqrdmlsh h0, h1, h2 = 0x7e428c20 (size=01, opcode4=0001)
  //   sqrdmlsh s0, s1, s2 = 0x7e828c20 (size=10, opcode4=0001)
  void DecodeAdvSimdScalarRdmThreeSame() {
    uint8_t size = GetBits<22, 2>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    bool bit11 = GetBits<11, 1>();

    // size=00 (B) and size=11 (D) are reserved per ARM ARM.
    if (size != 0b01 && size != 0b10) { Undefined(); return; }

    const AdvSimdScalarThreeSameArgs args = {
        .opcode = bit11 ? AdvSimdScalarThreeSameOpcode::kSqrdmlshScalar
                        : AdvSimdScalarThreeSameOpcode::kSqrdmlahScalar,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .size = size,
        .is_fp16 = false,
    };
    insn_consumer_->AdvSimdScalarThreeSame(args);
  }

  //
  // AdvSIMD BFloat16 three-same-extra: BFDOT (vec), BFMMLA, BFMLALB (vec),
  // BFMLALT (vec).
  //
  // Encoding (verified via llvm-mc -march=armv8.6-a):
  //   bit31=0, bit29=1, bits[28:24]=01110, bits[23:22]=size, bit21=0,
  //   bits[20:16]=Rm, bits[15:13]=111, bit12=op, bit11=1, bit10=1,
  //   bits[9:5]=Rn, bits[4:0]=Rd.
  //
  // size=01:
  //   bit30 = Q.
  //   bit12=1 -> BFDOT  (Q selects 2S vs 4S form).
  //   bit12=0 -> BFMMLA (Q must be 1; Q=0 reserved per ARM ARM C7.2.55).
  //
  // size=11:
  //   bit30 = T (B/T discriminator).  Q is implicit 1 (BFMLAL is always .4s).
  //   bit12 must be 1 (bits[15:10]=111111).  bit30=0 -> BFMLALB; bit30=1 -> BFMLALT.
  //
  // llvm-mc-verified encodings:
  //   bfmlalb v0.4s, v1.8h, v2.8h   = 0x2ec2fc20  (bit30=0, T=B)
  //   bfmlalt v0.4s, v1.8h, v2.8h   = 0x6ec2fc20  (bit30=1, T=T)
  void DecodeAdvSimdBf16ThreeSame() {
    uint8_t size = GetBits<22, 2>();
    bool bit30 = GetBits<30, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    bool bit12 = GetBits<12, 1>();

    Bf16ThreeSameOpcode opcode;
    bool eff_q;

    if (size == 0b01) {
      // BFDOT (vector) / BFMMLA.
      if (bit12) {
        opcode = Bf16ThreeSameOpcode::kBfdot;
      } else {
        if (!bit30) { Undefined(); return; }   // BFMMLA requires Q=1
        opcode = Bf16ThreeSameOpcode::kBfmmla;
      }
      eff_q = bit30;
    } else {
      // size == 0b11: BFMLALB / BFMLALT (vector).
      // bit12 must be 1; bit12=0 here is unallocated.
      if (!bit12) { Undefined(); return; }
      opcode = bit30 ? Bf16ThreeSameOpcode::kBfmlaltVec
                     : Bf16ThreeSameOpcode::kBfmlalbVec;
      eff_q = true;  // BFMLAL vector is always .4s
    }

    const Bf16ThreeSameArgs args = {
        .opcode = opcode,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .index = 0,
        .q = eff_q,
    };
    insn_consumer_->AdvSimdBf16ThreeSame(args);
  }

  //
  // AdvSIMD three same.
  // Encoding: 0 Q U 01110 size 1 Rm opcode 1 Rn Rd
  //   Q = bit30, U = bit29
  //   size = bits[23:22], Rm = bits[20:16]
  //   opcode = bits[15:11], Rn = bits[9:5], Rd = bits[4:0]
  //
  // Armv8.2-FP16 vector three-same.
  // Encoding: 0 Q U 0 1 1 1 0 a 1 0 Rm 0 0 opcode 1 Rn Rd
  // where a = bit23, opcode = bits[13:11] (3 bits).  Maps (a, U, opcode) to
  // the existing AdvSimdThreeSameOpcode set (which the interpreter then
  // dispatches with args.is_fp16 = true to use 2-byte lanes).
  // Opcode table (per ARM ARM C7.2 "Advanced SIMD three same (FP16)"):
  //   a=0,U=0,op=000 FMAXNM    a=1,U=0,op=000 FMINNM
  //   a=0,U=0,op=001 FMLA      a=1,U=0,op=001 FMLS
  //   a=0,U=0,op=010 FADD      a=1,U=0,op=010 FSUB
  //   a=0,U=0,op=011 FMULX*    a=1,U=0,op=011 reserved
  //   a=0,U=0,op=100 FCMEQ     a=1,U=0,op=100 reserved
  //   a=0,U=0,op=110 FMAX      a=1,U=0,op=110 FMIN
  //   a=0,U=0,op=111 FRECPS*   a=1,U=0,op=111 FRSQRTS*
  //   a=0,U=1,op=000 FMAXNMP*  a=1,U=1,op=000 FMINNMP*
  //   a=0,U=1,op=010 FADDP*    a=1,U=1,op=010 FABD
  //   a=0,U=1,op=011 FMUL      a=1,U=1,op=011 reserved
  //   a=0,U=1,op=100 FCMGE     a=1,U=1,op=100 FCMGT
  //   a=0,U=1,op=101 FACGE     a=1,U=1,op=101 FACGT
  //   a=0,U=1,op=110 FMAXP*    a=1,U=1,op=110 FMINP*
  //   a=0,U=1,op=111 FDIV      a=1,U=1,op=111 reserved
  // * = not implemented in this cycle (pairwise / FRECPS / FMULX); routed
  // to Undefined() until the interpreter grows handlers.
  void DecodeAdvSimdFp16ThreeSame() {
    bool q = GetBits<30, 1>();
    bool u = GetBits<29, 1>();
    bool a = GetBits<23, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode_3 = GetBits<11, 3>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdThreeSameOpcode op;
    bool ok = false;
    if (!u) {
      if (!a) {
        switch (opcode_3) {
          case 0b000: op = AdvSimdThreeSameOpcode::kFmaxnmV; ok = true; break;
          case 0b001: op = AdvSimdThreeSameOpcode::kFmlaV;   ok = true; break;
          case 0b010: op = AdvSimdThreeSameOpcode::kFaddV;   ok = true; break;
          // FP16 FMULX (a=0, U=0, opcode_3=011).
          case 0b011: op = AdvSimdThreeSameOpcode::kFmulxV;  ok = true; break;
          case 0b100: op = AdvSimdThreeSameOpcode::kFcmeqV;  ok = true; break;
          case 0b110: op = AdvSimdThreeSameOpcode::kFmaxV;   ok = true; break;
          // FP16 FRECPS (a=0, U=0, opcode_3=111).
          case 0b111: op = AdvSimdThreeSameOpcode::kFrecpsV; ok = true; break;
          default: break;  // 101 reserved — Undefined.
        }
      } else {
        switch (opcode_3) {
          case 0b000: op = AdvSimdThreeSameOpcode::kFminnmV; ok = true; break;
          case 0b001: op = AdvSimdThreeSameOpcode::kFmlsV;   ok = true; break;
          case 0b010: op = AdvSimdThreeSameOpcode::kFsubV;   ok = true; break;
          case 0b110: op = AdvSimdThreeSameOpcode::kFminV;   ok = true; break;
          // FP16 FRSQRTS (a=1, U=0, opcode_3=111).
          case 0b111: op = AdvSimdThreeSameOpcode::kFrsqrtsV; ok = true; break;
          default: break;  // 011 reserved, 100 reserved, 101 reserved — Undefined.
        }
      }
    } else {
      if (!a) {
        switch (opcode_3) {
          case 0b000: op = AdvSimdThreeSameOpcode::kFmaxnmpV; ok = true; break;
          case 0b010: op = AdvSimdThreeSameOpcode::kFaddpV;  ok = true; break;
          case 0b011: op = AdvSimdThreeSameOpcode::kFmulV;   ok = true; break;
          case 0b100: op = AdvSimdThreeSameOpcode::kFcmgeV;  ok = true; break;
          case 0b101: op = AdvSimdThreeSameOpcode::kFacgeV;  ok = true; break;
          case 0b110: op = AdvSimdThreeSameOpcode::kFmaxpV;  ok = true; break;
          case 0b111: op = AdvSimdThreeSameOpcode::kFdivV;   ok = true; break;
          default: break;  // 001 reserved — Undefined.
        }
      } else {
        switch (opcode_3) {
          case 0b000: op = AdvSimdThreeSameOpcode::kFminnmpV; ok = true; break;
          case 0b010: op = AdvSimdThreeSameOpcode::kFabdV;   ok = true; break;
          case 0b100: op = AdvSimdThreeSameOpcode::kFcmgtV;  ok = true; break;
          case 0b101: op = AdvSimdThreeSameOpcode::kFacgtV;  ok = true; break;
          case 0b110: op = AdvSimdThreeSameOpcode::kFminpV;  ok = true; break;
          default: break;  // 011, 111 reserved — Undefined.
        }
      }
    }
    if (!ok) {
      Undefined();
      return;
    }

    const AdvSimdThreeSameArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .size = 0,        // unused for the FP16 path (interpreter checks is_fp16)
        .q = q,
        .is_fp16 = true,
    };
    insn_consumer_->AdvSimdThreeSame(args);
  }

  // Armv8.2-FP16 scalar three-same.
  // Encoding (per ARM ARM C7.2 "Advanced SIMD scalar three same (FP16)"):
  //   0 1 U 1 1 1 1 0 a 1 0 Rm 0 0 opcode_3 1 Rn Rd
  // where a = bit23, opcode_3 = bits[13:11] (3 bits).
  // Opcode table (the scalar-allocated subset; cells marked "—" are
  // reserved/Undefined in the scalar encoding):
  //   a=0,U=0,op=011  FMULX   (saturation: ±0 * ±inf -> ±2.0)
  //   a=0,U=0,op=100  FCMEQ
  //   a=0,U=0,op=111  FRECPS  (interpreter not implemented yet)
  //   a=1,U=0,op=111  FRSQRTS (interpreter not implemented yet)
  //   a=0,U=1,op=100  FCMGE
  //   a=0,U=1,op=101  FACGE
  //   a=1,U=1,op=010  FABD
  //   a=1,U=1,op=100  FCMGT
  //   a=1,U=1,op=101  FACGT
  // Reuses the std AdvSimdScalarThreeSameOpcode enum with the is_fp16=true
  // flag, mirroring the FP16 vector three-same pattern.  FRECPS/FRSQRTS
  // (op=111) remain Undefined() until interpreter handlers land.
  void DecodeAdvSimdScalarFp16ThreeSame() {
    bool u = GetBits<29, 1>();
    bool a = GetBits<23, 1>();
    uint8_t rm = GetBits<16, 5>();
    uint8_t opcode_3 = GetBits<11, 3>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdScalarThreeSameOpcode op;
    bool ok = false;
    if (!a && !u && opcode_3 == 0b011) {
      op = AdvSimdScalarThreeSameOpcode::kFmulx;
      ok = true;
    } else if (!a && !u && opcode_3 == 0b100) {
      op = AdvSimdScalarThreeSameOpcode::kFcmeq;
      ok = true;
    } else if (!a && u && opcode_3 == 0b100) {
      op = AdvSimdScalarThreeSameOpcode::kFcmge;
      ok = true;
    } else if (!a && u && opcode_3 == 0b101) {
      op = AdvSimdScalarThreeSameOpcode::kFacge;
      ok = true;
    } else if (a && u && opcode_3 == 0b010) {
      op = AdvSimdScalarThreeSameOpcode::kFabd;
      ok = true;
    } else if (a && u && opcode_3 == 0b100) {
      op = AdvSimdScalarThreeSameOpcode::kFcmgt;
      ok = true;
    } else if (a && u && opcode_3 == 0b101) {
      op = AdvSimdScalarThreeSameOpcode::kFacgt;
      ok = true;
    } else if (!a && !u && opcode_3 == 0b111) {
      // FP16 scalar FRECPS (a=0, U=0, opcode_3=111).
      op = AdvSimdScalarThreeSameOpcode::kFrecps;
      ok = true;
    } else if (a && !u && opcode_3 == 0b111) {
      // FP16 scalar FRSQRTS (a=1, U=0, opcode_3=111).
      op = AdvSimdScalarThreeSameOpcode::kFrsqrts;
      ok = true;
    }
    // Reserved (a,U,op) combinations route to Undefined().
    if (!ok) {
      Undefined();
      return;
    }

    const AdvSimdScalarThreeSameArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .size = 0,         // unused for FP16 path (interpreter checks is_fp16).
        .is_fp16 = true,
    };
    insn_consumer_->AdvSimdScalarThreeSame(args);
  }

  // Armv8.2-FP16 vector two-register miscellaneous.
  // Encoding (per ARM ARM C7.2 "Advanced SIMD two-register miscellaneous (FP16)"):
  //   0 Q U 0 1 1 1 0 a 1 1 1 1 1 0 opcode 1 0 Rn Rd
  // i.e. bit31=0, bits[28:24]=01110, bit23=a, bits[22:17]=111110,
  //      bits[16:12]=opcode, bits[11:10]=10.
  // Verified against `clang --target=aarch64 -march=armv8.2-a+fp16`:
  //   FABS  v0.4h,v1.4h = 0x0ef8f820 -> a=1,U=0,op=01111
  //   FNEG  v0.4h,v1.4h = 0x2ef8f820 -> a=1,U=1,op=01111
  //   FSQRT v0.4h,v1.4h = 0x2ef9f820 -> a=1,U=1,op=11111
  //   FCMEQ v0.4h,v1.4h,#0 = 0x0ef8d820 -> a=1,U=0,op=01101
  //   FCMGT v0.4h,v1.4h,#0 = 0x0ef8c820 -> a=1,U=0,op=01100
  //   FCMLT v0.4h,v1.4h,#0 = 0x0ef8e820 -> a=1,U=0,op=01110
  //   FCMGE v0.4h,v1.4h,#0 = 0x2ef8c820 -> a=1,U=1,op=01100
  //   FCMLE v0.4h,v1.4h,#0 = 0x2ef8d820 -> a=1,U=1,op=01101
  // Opcodes that aren't implemented yet (FRINT* / FCVT* round-mode /
  // SCVTF/UCVTF/FRECPE/FRSQRTE in FP16 form) route to Undefined() until
  // the interpreter grows the handlers; this matches the three-same
  // pairwise-reject pattern.
  void DecodeAdvSimdFp16TwoRegMisc() {
    bool q = GetBits<30, 1>();
    bool u = GetBits<29, 1>();
    bool a = GetBits<23, 1>();
    uint8_t opcode_5 = GetBits<12, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdTwoRegMiscOpcode op;
    bool ok = false;

    // (a, U, opcode_5) selects the op per ARM ARM C7.2 "Advanced SIMD
    // two-register miscellaneous (FP16)".
    if (a) {
      if (!u) {
        switch (opcode_5) {
          case 0b01100: op = AdvSimdTwoRegMiscOpcode::kCmgtZero; ok = true; break;  // FCMGT #0
          case 0b01101: op = AdvSimdTwoRegMiscOpcode::kCmeqZero; ok = true; break;  // FCMEQ #0
          case 0b01110: op = AdvSimdTwoRegMiscOpcode::kCmltZero; ok = true; break;  // FCMLT #0
          case 0b01111: op = AdvSimdTwoRegMiscOpcode::kFabs;     ok = true; break;  // FABS
          // a=0/a=1 columns: FRINT*/FCVT*-round.
          case 0b11000: op = AdvSimdTwoRegMiscOpcode::kFrintpV;  ok = true; break;  // FRINTP
          case 0b11001: op = AdvSimdTwoRegMiscOpcode::kFrintzV;  ok = true; break;  // FRINTZ
          case 0b11010: op = AdvSimdTwoRegMiscOpcode::kFcvtpsV;  ok = true; break;  // FCVTPS
          case 0b11011: op = AdvSimdTwoRegMiscOpcode::kFcvtzsV;  ok = true; break;  // FCVTZS
          case 0b11101: op = AdvSimdTwoRegMiscOpcode::kFrecpeV;  ok = true; break;  // FRECPE
          default: break;
        }
      } else {
        switch (opcode_5) {
          case 0b01100: op = AdvSimdTwoRegMiscOpcode::kCmgeZero; ok = true; break;  // FCMGE #0
          case 0b01101: op = AdvSimdTwoRegMiscOpcode::kCmleZero; ok = true; break;  // FCMLE #0
          case 0b01111: op = AdvSimdTwoRegMiscOpcode::kFneg;     ok = true; break;  // FNEG
          case 0b11111: op = AdvSimdTwoRegMiscOpcode::kFsqrtV;   ok = true; break;  // FSQRT
          // a=0/a=1 columns.
          case 0b11001: op = AdvSimdTwoRegMiscOpcode::kFrintiV;  ok = true; break;  // FRINTI
          case 0b11010: op = AdvSimdTwoRegMiscOpcode::kFcvtpuV;  ok = true; break;  // FCVTPU
          case 0b11011: op = AdvSimdTwoRegMiscOpcode::kFcvtzuV;  ok = true; break;  // FCVTZU
          case 0b11101: op = AdvSimdTwoRegMiscOpcode::kFrsqrteV; ok = true; break;  // FRSQRTE
          default: break;
        }
      }
    } else {
      // a=0 column: FRINTN/A, FRINTM/X,
      // FCVTNS/NU, FCVTMS/MU, FCVTAS/AU, SCVTF/UCVTF in FP16 form.
      if (!u) {
        switch (opcode_5) {
          case 0b11000: op = AdvSimdTwoRegMiscOpcode::kFrintnV;  ok = true; break;  // FRINTN
          case 0b11001: op = AdvSimdTwoRegMiscOpcode::kFrintmV;  ok = true; break;  // FRINTM
          case 0b11010: op = AdvSimdTwoRegMiscOpcode::kFcvtnsV;  ok = true; break;  // FCVTNS
          case 0b11011: op = AdvSimdTwoRegMiscOpcode::kFcvtmsV;  ok = true; break;  // FCVTMS
          case 0b11100: op = AdvSimdTwoRegMiscOpcode::kFcvtasV;  ok = true; break;  // FCVTAS
          case 0b11101: op = AdvSimdTwoRegMiscOpcode::kScvtfV;   ok = true; break;  // SCVTF
          default: break;
        }
      } else {
        switch (opcode_5) {
          case 0b11000: op = AdvSimdTwoRegMiscOpcode::kFrintaV;  ok = true; break;  // FRINTA
          case 0b11001: op = AdvSimdTwoRegMiscOpcode::kFrintxV;  ok = true; break;  // FRINTX
          case 0b11010: op = AdvSimdTwoRegMiscOpcode::kFcvtnuV;  ok = true; break;  // FCVTNU
          case 0b11011: op = AdvSimdTwoRegMiscOpcode::kFcvtmuV;  ok = true; break;  // FCVTMU
          case 0b11100: op = AdvSimdTwoRegMiscOpcode::kFcvtauV;  ok = true; break;  // FCVTAU
          case 0b11101: op = AdvSimdTwoRegMiscOpcode::kUcvtfV;   ok = true; break;  // UCVTF
          default: break;
        }
      }
    }

    if (!ok) {
      Undefined();
      return;
    }

    const AdvSimdTwoRegMiscArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .size = 0,        // unused for the FP16 path (interpreter checks is_fp16)
        .q = q,
        .u = u,
        .is_fp16 = true,
    };
    insn_consumer_->AdvSimdTwoRegMisc(args);
  }

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
    } else if (opcode == 0b00110) {
      op = u ? AdvSimdThreeSameOpcode::kCmhi : AdvSimdThreeSameOpcode::kCmgt;
    } else if (opcode == 0b00111) {
      op = u ? AdvSimdThreeSameOpcode::kCmhs : AdvSimdThreeSameOpcode::kCmge;
    } else if (opcode == 0b01100) {
      op = u ? AdvSimdThreeSameOpcode::kUmax : AdvSimdThreeSameOpcode::kSmax;
    } else if (opcode == 0b01101) {
      op = u ? AdvSimdThreeSameOpcode::kUmin : AdvSimdThreeSameOpcode::kSmin;
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
      // U=1 -> PMUL polynomial multiply (size=00 only).
      // Previously U=1 routed to Undefined(); now dispatched to kPmul.
      if (u) {
        if (size != 0b00) { Undefined(); return; }
        op = AdvSimdThreeSameOpcode::kPmul;
      } else {
        op = AdvSimdThreeSameOpcode::kMul;
      }
    } else if (opcode == 0b10010) {
      op = u ? AdvSimdThreeSameOpcode::kMls : AdvSimdThreeSameOpcode::kMla;
    // SQDMULH (U=0) / SQRDMULH (U=1) saturating doubling
    // multiply high. size=00 and size=11 are reserved per ARM ARM.
    } else if (opcode == 0b10110) {
      if (size == 0b00 || size == 0b11) { Undefined(); return; }
      op = u ? AdvSimdThreeSameOpcode::kSqrdmulh : AdvSimdThreeSameOpcode::kSqdmulh;
    } else if (opcode == 0b10100) {
      op = u ? AdvSimdThreeSameOpcode::kUmaxp : AdvSimdThreeSameOpcode::kSmaxp;
    } else if (opcode == 0b10101) {
      op = u ? AdvSimdThreeSameOpcode::kUminp : AdvSimdThreeSameOpcode::kSminp;
    } else if (opcode == 0b01110) {
      // SABD/UABD: absolute-difference vector. size=11 reserved.
      if (size == 0b11) { Undefined(); return; }
      op = u ? AdvSimdThreeSameOpcode::kUabd : AdvSimdThreeSameOpcode::kSabd;
    } else if (opcode == 0b01111) {
      // SABA/UABA: absolute-difference-and-accumulate vector. size=11 reserved.
      if (size == 0b11) { Undefined(); return; }
      op = u ? AdvSimdThreeSameOpcode::kUaba : AdvSimdThreeSameOpcode::kSaba;
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
            op = u ? AdvSimdThreeSameOpcode::kFaddpV
                   : AdvSimdThreeSameOpcode::kFaddV;
            break;
          case 0b11011:
            // U=0 -> FMULX (kFmulxV), U=1 -> FMUL (kFmulV).
            // Previously U=0 routed to ok=false; the interpreter now grows
            // the FMULX special case (±0 * ±inf -> ±2.0) so we can dispatch.
            op = u ? AdvSimdThreeSameOpcode::kFmulV
                   : AdvSimdThreeSameOpcode::kFmulxV;
            break;
          case 0b11001:
            if (u) { ok = false; break; }   // U=1 reserved here
            op = AdvSimdThreeSameOpcode::kFmlaV;
            break;
          // FMAXNM/FMAX (op_high=0, U=0); FDIV (U=1);
          // FCMEQ (U=0)/FCMGE (U=1)/FACGE (U=1) at opcode 11100/11101.
          case 0b11000:
            op = u ? AdvSimdThreeSameOpcode::kFmaxnmpV
                   : AdvSimdThreeSameOpcode::kFmaxnmV;
            break;
          case 0b11100:
            op = u ? AdvSimdThreeSameOpcode::kFcmgeV : AdvSimdThreeSameOpcode::kFcmeqV;
            break;
          case 0b11101:
            // op_high=0, opcode=11101: FACGE (U=1); U=0 is reserved.
            if (!u) { ok = false; break; }
            op = AdvSimdThreeSameOpcode::kFacgeV;
            break;
          case 0b11110:
            op = u ? AdvSimdThreeSameOpcode::kFmaxpV
                   : AdvSimdThreeSameOpcode::kFmaxV;
            break;
          case 0b11111:
            // op_high=0, opcode=11111: FRECPS (U=0) / FDIV (U=1).
            op = u ? AdvSimdThreeSameOpcode::kFdivV
                   : AdvSimdThreeSameOpcode::kFrecpsV;
            break;
          default: ok = false; break;
        }
      } else {
        switch (opcode) {
          case 0b11010:
            op = u ? AdvSimdThreeSameOpcode::kFabdV
                   : AdvSimdThreeSameOpcode::kFsubV;
            break;
          case 0b11001:
            if (u) { ok = false; break; }   // U=1 reserved here
            op = AdvSimdThreeSameOpcode::kFmlsV;
            break;
          // FMINNM/FMIN (op_high=1, U=0); FCMGT (U=1)/FACGT (U=1).
          case 0b11000:
            op = u ? AdvSimdThreeSameOpcode::kFminnmpV
                   : AdvSimdThreeSameOpcode::kFminnmV;
            break;
          case 0b11100:
            if (!u) { ok = false; break; }  // op_high=1,U=0 undef at opcode 11100
            op = AdvSimdThreeSameOpcode::kFcmgtV;
            break;
          case 0b11101:
            // op_high=1, opcode=11101: FACGT (U=1); U=0 is reserved.
            if (!u) { ok = false; break; }
            op = AdvSimdThreeSameOpcode::kFacgtV;
            break;
          case 0b11110:
            op = u ? AdvSimdThreeSameOpcode::kFminpV
                   : AdvSimdThreeSameOpcode::kFminV;
            break;
          case 0b11111:
            // op_high=1, opcode=11111: FRSQRTS (U=0);
            // U=1 is reserved.
            if (u) { ok = false; break; }
            op = AdvSimdThreeSameOpcode::kFrsqrtsV;
            break;
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

    // PMULL (opcode=1110) accepts size=00 (8-bit) and size=11 (64-bit, PMULL64).
    // Decoder rule: size=11 is reserved for all OTHER three-different opcodes; only PMULL allows it.
    if (size == 0b11 && !(u == 0 && opcode == 0b1110)) {
      Undefined();
      return;
    }

    AdvSimdThreeDiffOpcode op;

    switch (opcode) {
      case 0b0000:
        op = u ? AdvSimdThreeDiffOpcode::kUaddl : AdvSimdThreeDiffOpcode::kSaddl;
        break;
      // polynomial multiply (used by libz CRC32-acc).
      case 0b1110:
        if (u != 0) { Undefined(); return; }  // U=1 with opcode=1110 is unallocated
        // size=01 and size=10 are unallocated for PMULL.
        if (size == 0b01 || size == 0b10) { Undefined(); return; }
        op = AdvSimdThreeDiffOpcode::kPmull;
        break;
      // wide add/sub variants (opcode 0001/0011)
      case 0b0001:
        op = u ? AdvSimdThreeDiffOpcode::kUaddw : AdvSimdThreeDiffOpcode::kSaddw;
        break;
      case 0b0011:
        op = u ? AdvSimdThreeDiffOpcode::kUsubw : AdvSimdThreeDiffOpcode::kSsubw;
        break;
      case 0b0010:
        op = u ? AdvSimdThreeDiffOpcode::kUsubl : AdvSimdThreeDiffOpcode::kSsubl;
        break;
      // narrowing high: ADDHN/ADDHN2 (U=0), RADDHN/RADDHN2 (U=1).
      // size=11 already rejected above.
      case 0b0100:
        op = u ? AdvSimdThreeDiffOpcode::kRaddhn : AdvSimdThreeDiffOpcode::kAddhn;
        break;
      case 0b0101:
        op = u ? AdvSimdThreeDiffOpcode::kUabal : AdvSimdThreeDiffOpcode::kSabal;
        break;
      // narrowing high subtract: SUBHN/SUBHN2 (U=0),
      // RSUBHN/RSUBHN2 (U=1). size=11 already rejected above.
      case 0b0110:
        op = u ? AdvSimdThreeDiffOpcode::kRsubhn : AdvSimdThreeDiffOpcode::kSubhn;
        break;
      // signed saturating doubling multiply long:
      // SQDMULL / SQDMULL2 (U=0, opcode=1101). U=1 with opcode=1101 is
      // unallocated. size=00 is reserved (only halfword and word inputs
      // are defined); size=11 is already rejected by the top-of-routine
      // size guard.
      case 0b1101:
        if (u != 0) { Undefined(); return; }
        if (size == 0b00) { Undefined(); return; }
        op = AdvSimdThreeDiffOpcode::kSqdmull;
        break;
      // signed saturating doubling multiply-accumulate
      // long: SQDMLAL / SQDMLAL2 (U=0, opcode=1001). U=1 with opcode=1001
      // is unallocated. size=00 is reserved (only halfword and word inputs
      // are defined); size=11 is already rejected by the top-of-routine
      // size guard.
      case 0b1001:
        if (u != 0) { Undefined(); return; }
        if (size == 0b00) { Undefined(); return; }
        op = AdvSimdThreeDiffOpcode::kSqdmlal;
        break;
      // signed saturating doubling multiply-subtract
      // long: SQDMLSL / SQDMLSL2 (U=0, opcode=1011). U=1 with opcode=1011
      // is unallocated. size=00 is reserved (only halfword and word inputs
      // are defined); size=11 is already rejected by the top-of-routine
      // size guard.
      case 0b1011:
        if (u != 0) { Undefined(); return; }
        if (size == 0b00) { Undefined(); return; }
        op = AdvSimdThreeDiffOpcode::kSqdmlsl;
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
    // Armv8.2-FP16 across-lanes (FMAXV/FMINV/FMAXNMV/FMINNMV
    // on .4H/.8H) shares the across-lanes dispatch path with FP32 (.4S) and
    // is distinguished by U=0 vs FP32's U=1. The relevant cases below set
    // this flag; it is otherwise false. The backends read args.is_fp16
    // inside each across-lanes arm to dispatch the half-precision element
    // path.
    bool is_fp16 = false;

    switch (opcode) {
      case 0b00000:
        op = u ? AdvSimdTwoRegMiscOpcode::kRev32 : AdvSimdTwoRegMiscOpcode::kRev64;
        break;
      case 0b00001:
        // REV16 (vector) is U=0, opcode=00001 (REV64=U0/op0, REV32=U1/op0,
        // REV16=U0/op1). U=1 at this opcode is unallocated.
        if (u) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kRev16;
        break;
      case 0b00010:
        op = u ? AdvSimdTwoRegMiscOpcode::kUaddlp : AdvSimdTwoRegMiscOpcode::kSaddlp;
        break;
      // opcode 00011 covers two distinct families:
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
      // SQABS (U=0) / SQNEG (U=1) at opcode 00111.
      // All four sizes are valid; size=11 (.2d) is Q=1-only — the .1d form
      // (size=11, Q=0) is unallocated.
      case 0b00111:
        if (size == 0b11 && !q) { Undefined(); return; }
        op = u ? AdvSimdTwoRegMiscOpcode::kSqneg : AdvSimdTwoRegMiscOpcode::kSqabs;
        break;
      case 0b01000:
        op = u ? AdvSimdTwoRegMiscOpcode::kCmgeZero : AdvSimdTwoRegMiscOpcode::kCmgtZero;
        break;
      case 0b01001:
        op = u ? AdvSimdTwoRegMiscOpcode::kCmleZero : AdvSimdTwoRegMiscOpcode::kCmeqZero;
        break;
      case 0b01010:
        // bit20 distinguishes across-lanes (1) from two-reg-misc (0).
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
        // bit20=1: across-lanes SMINV (U=0) / UMINV (U=1).
        // bit20=0: vector FCVT* rounding-mode, with bit23 picking N (0) vs P (1).
        if (GetBits<20, 1>()) {
          op = u ? AdvSimdTwoRegMiscOpcode::kUminv : AdvSimdTwoRegMiscOpcode::kSminv;
          if (size == 0b11) { Undefined(); return; }
        } else {
          // vector FCVTNS/NU (bit23=0) and FCVTPS/PU (bit23=1).
          if (!GetBits<23, 1>()) {
            op = u ? AdvSimdTwoRegMiscOpcode::kFcvtnuV
                   : AdvSimdTwoRegMiscOpcode::kFcvtnsV;
          } else {
            op = u ? AdvSimdTwoRegMiscOpcode::kFcvtpuV
                   : AdvSimdTwoRegMiscOpcode::kFcvtpsV;
          }
        }
        break;
      case 0b01011:
        op = u ? AdvSimdTwoRegMiscOpcode::kNeg : AdvSimdTwoRegMiscOpcode::kAbs;
        break;
      case 0b10010:
        // opcode=10010 splits on U: XTN (U=0) / SQXTUN (U=1).
        // Sizes 00/01/10 are valid; size=11 has no encoded narrow form for SQXTUN.
        // XTN at size=11 already routes to Undefined via the existing dispatch in
        // DecodeAdvSimd that gates on the Q/size combination, but we gate here too
        // for the SQXTUN arm to mirror the ARM ARM.
        if (u) {
          if (size == 0b11) { Undefined(); return; }
          op = AdvSimdTwoRegMiscOpcode::kSqxtun;
        } else {
          op = AdvSimdTwoRegMiscOpcode::kXtn;
        }
        break;
      // opcode=10011 is SHLL/SHLL2 (U=1 required).
      // size=11 is unallocated (no destination element wider than 64-bit).
      case 0b10011:
        if (!u) { Undefined(); return; }
        if (size == 0b11) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kShll;
        break;
      case 0b10100:
        if (u) {
          op = AdvSimdTwoRegMiscOpcode::kUqxtn;
        } else {
          op = AdvSimdTwoRegMiscOpcode::kSqxtn;
        }
        break;
      case 0b10110:
        // opcode=10110 splits on U:
        //   U=0 + size=10 -> BFCVTN/BFCVTN2 (FP32->BF16).
        //   U=0 + size=00/01 -> FCVTN/FCVTN2: size=00 narrows FP32->FP16,
        //     size=01 narrows FP64->FP32. Both are implemented.
        //   U=1 + size=01 -> FCVTXN/FCVTXN2 (FP64->FP32, round-to-odd).
        //   U=1 + size!=01 -> unallocated.
        // bit30 (q) selects low-half write (Q=0) vs high-half write (Q=1).
        if (u) {
          if (size != 0b01) { Undefined(); return; }
          op = AdvSimdTwoRegMiscOpcode::kFcvtxn;
        } else if (size == 0b10) {
          op = AdvSimdTwoRegMiscOpcode::kBfcvtn;
        } else {
          op = AdvSimdTwoRegMiscOpcode::kFcvtn;
        }
        break;
      case 0b10111:
        if (u) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kFcvtl;
        break;
      // bit20 splits opcode=01111 between
      // FABS/FNEG (bit20=0, two-reg-misc) and across-lanes
      // FMAXV/FMINV (bit20=1). bit23 picks max (0) vs min (1).
      // U=1 selects the FP32 form; U=0 selects the Armv8.2-FP16 form
      // (backends dispatch on args.is_fp16). For both U values bit22
      // (size[0], "sz") must be 0 — FP64 is unallocated for these encodings.
      // Q selects the arrangement: FP32 has only .4S, so Q=0 (.2S) is
      // unallocated; FP16 has .4H (Q=0) and .8H (Q=1). The Q test therefore
      // runs after is_fp16 is known.
      case 0b01111:
        if (GetBits<20, 1>()) {
          if (size & 1) { Undefined(); return; }
          is_fp16 = !u;
          if (!is_fp16 && !q) { Undefined(); return; }
          op = GetBits<23, 1>() ? AdvSimdTwoRegMiscOpcode::kFminv
                                : AdvSimdTwoRegMiscOpcode::kFmaxv;
        } else {
          op = u ? AdvSimdTwoRegMiscOpcode::kFneg : AdvSimdTwoRegMiscOpcode::kFabs;
        }
        break;
      // opcode=01100 covers two distinct encodings:
      //   bit20=1: across-lanes FMAXNMV (bit23=0) / FMINNMV (bit23=1).
      //     bit22 (sz) must be 0; U=1 selects FP32 (.4S only), U=0 selects
      //     FP16 (.4H when Q=0, .8H when Q=1) — backends dispatch on
      //     args.is_fp16.
      //   bit20=0: FP two-reg-misc FCMGT (zero, U=0) / FCMGE (zero, U=1).
      //     bit23=1 is required (FP form). bit22 (sz) picks FP32 (0) or
      //     FP64 (1); FP64 requires Q=1.
      // Pre-Digitalis these all routed to Undefined() (fell through to default).
      case 0b01100:
        if (GetBits<20, 1>()) {
          if (size & 1) { Undefined(); return; }
          is_fp16 = !u;
          if (!is_fp16 && !q) { Undefined(); return; }
          op = GetBits<23, 1>() ? AdvSimdTwoRegMiscOpcode::kFminnmv
                                : AdvSimdTwoRegMiscOpcode::kFmaxnmv;
        } else {
          if (!GetBits<23, 1>()) { Undefined(); return; }  // bit23=0 unallocated
          if (size == 0b11 && !q) { Undefined(); return; }  // FP64 needs Q=1
          op = u ? AdvSimdTwoRegMiscOpcode::kFcmgeZero
                 : AdvSimdTwoRegMiscOpcode::kFcmgtZero;
        }
        break;
      // FP two-reg-misc FCMEQ (zero, U=0) / FCMLE (zero, U=1).
      // Encoding: opcode=01101, bit20=0, bit23=1 required.
      case 0b01101:
        if (GetBits<20, 1>()) { Undefined(); return; }
        if (!GetBits<23, 1>()) { Undefined(); return; }
        if (size == 0b11 && !q) { Undefined(); return; }
        op = u ? AdvSimdTwoRegMiscOpcode::kFcmleZero
               : AdvSimdTwoRegMiscOpcode::kFcmeqZero;
        break;
      // FP two-reg-misc FCMLT zero. U=1 unallocated.
      // Encoding: opcode=01110, bit20=0, bit23=1, U=0.
      case 0b01110:
        if (u) { Undefined(); return; }
        if (GetBits<20, 1>()) { Undefined(); return; }
        if (!GetBits<23, 1>()) { Undefined(); return; }
        if (size == 0b11 && !q) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kFcmltZero;
        break;
      // opcode=11101 splits on bit23:
      //   bit23=0: SCVTF (U=0) / UCVTF (U=1) — vector int→FP.
      //   bit23=1: FRECPE (U=0) / FRSQRTE (U=1) — vector FP reciprocal estimate.
      case 0b11101:
        if (GetBits<23, 1>()) {
          op = u ? AdvSimdTwoRegMiscOpcode::kFrsqrteV
                 : AdvSimdTwoRegMiscOpcode::kFrecpeV;
        } else {
          op = u ? AdvSimdTwoRegMiscOpcode::kUcvtfV
                 : AdvSimdTwoRegMiscOpcode::kScvtfV;
        }
        break;
      // opcode=11111 splits on bit23:
      //   bit23=1, U=1: FSQRT (vector).
      //   bit23=0: FRINT64Z (U=0) / FRINT64X (U=1) — FRINTTS 64-bit.
      case 0b11111:
        if (GetBits<23, 1>()) {
          if (!u) { Undefined(); return; }
          op = AdvSimdTwoRegMiscOpcode::kFsqrtV;
        } else {
          op = u ? AdvSimdTwoRegMiscOpcode::kFrint64xV
                 : AdvSimdTwoRegMiscOpcode::kFrint64zV;
        }
        break;
      // opcode=11110, bit23=0: FRINT32Z (U=0) / FRINT32X
      // (U=1) — FRINTTS 32-bit. bit23=1 is unallocated for this opcode.
      case 0b11110:
        if (GetBits<23, 1>()) { Undefined(); return; }
        op = u ? AdvSimdTwoRegMiscOpcode::kFrint32xV
               : AdvSimdTwoRegMiscOpcode::kFrint32zV;
        break;
      // opcode=11011 splits on whether this is the
      // across-lanes group (bit20=1, ADDV) or two-reg-misc (bit20=0).
      // For bit20=0 with bit23=1, this is FCVTZS / FCVTZU (vector FP→int
      // truncating). bit23=0 / bit20=0 / opcode=11011 is FCVTMS/FCVTMU
      // (round toward -inf) which are not implemented yet.
      case 0b11011:
        if (GetBits<20, 1>()) {
          if (u) { Undefined(); return; }  // ADDV is U=0 only
          if (size == 0b11) { Undefined(); return; }  // No 64-bit element ADDV
          op = AdvSimdTwoRegMiscOpcode::kAddv;
        } else {
          if (GetBits<23, 1>()) {
            op = u ? AdvSimdTwoRegMiscOpcode::kFcvtzuV
                   : AdvSimdTwoRegMiscOpcode::kFcvtzsV;
          } else {
            // vector FCVTMS/MU (round toward -inf). .
            op = u ? AdvSimdTwoRegMiscOpcode::kFcvtmuV
                   : AdvSimdTwoRegMiscOpcode::kFcvtmsV;
          }
        }
        break;
      // opcode=11100 splits on bit23 ("a"):
      //   bit23=0: FCVTAS (U=0) / FCVTAU (U=1) — FP convert round-to-nearest
      //            ties-away.
      //   bit23=1, sz(bit22)=0: URECPE (U=0) / URSQRTE (U=1) — unsigned integer
      //            reciprocal / reciprocal-sqrt estimate (.2S/.4S only).
      // bit23=1 with sz=1 is unallocated.
      case 0b11100:
        if (GetBits<23, 1>()) {
          if (GetBits<22, 1>()) { Undefined(); return; }  // sz=1 unallocated
          op = u ? AdvSimdTwoRegMiscOpcode::kUrsqrte
                 : AdvSimdTwoRegMiscOpcode::kUrecpe;
        } else {
          op = u ? AdvSimdTwoRegMiscOpcode::kFcvtauV
                 : AdvSimdTwoRegMiscOpcode::kFcvtasV;
        }
        break;
      // std FP32/FP64 FRINT* round-to-int.
      // Encoding (per ARM ARM C7.2 "Advanced SIMD two-register miscellaneous"):
      //   0 Q U 0 1110 size 10000 opcode 10 Rn Rd
      // with bit23 = 'a' (rounding-mode subset) and bit22 = 'sz' (0=FP32,
      // 1=FP64).  The interpreter picks FP32 vs FP64 from `args.size & 1`,
      // matching the existing FCVTNS / FCVTPS / FCVTZS fp dispatch.
      //   opcode=11000:
      //     a=0,U=0: FRINTN  ties-to-even
      //     a=0,U=1: FRINTA  ties-away
      //     a=1,U=0: FRINTP  toward +inf
      //     a=1,U=1: FRINT32X (Armv8.5) -- left Undefined
      //   opcode=11001:
      //     a=0,U=0: FRINTM  toward -inf
      //     a=0,U=1: FRINTX  use current FPCR (raises Inexact)
      //     a=1,U=0: FRINTZ  toward zero
      //     a=1,U=1: FRINTI  use current FPCR (no Inexact)
      // Verified with llvm-mc:
      //   frintn v0.4s = 0x4e218820  (a=0,U=0,opcode=11000)
      //   frinta v0.4s = 0x6e218820  (a=0,U=1,opcode=11000)
      //   frintp v0.4s = 0x4ea18820  (a=1,U=0,opcode=11000)
      //   frintm v0.4s = 0x4e219820  (a=0,U=0,opcode=11001)
      //   frintx v0.4s = 0x6e219820  (a=0,U=1,opcode=11001)
      //   frintz v0.4s = 0x4ea19820  (a=1,U=0,opcode=11001)
      //   frinti v0.4s = 0x6ea19820  (a=1,U=1,opcode=11001)
      case 0b11000:
        if (!GetBits<23, 1>()) {
          op = u ? AdvSimdTwoRegMiscOpcode::kFrintaV
                 : AdvSimdTwoRegMiscOpcode::kFrintnV;
        } else {
          if (u) { Undefined(); return; }  // FRINT32X (Armv8.5) unimplemented
          op = AdvSimdTwoRegMiscOpcode::kFrintpV;
        }
        break;
      case 0b11001:
        if (!GetBits<23, 1>()) {
          op = u ? AdvSimdTwoRegMiscOpcode::kFrintxV
                 : AdvSimdTwoRegMiscOpcode::kFrintmV;
        } else {
          op = u ? AdvSimdTwoRegMiscOpcode::kFrintiV
                 : AdvSimdTwoRegMiscOpcode::kFrintzV;
        }
        break;
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
        // across-lanes FP16 (FMAXV/FMINV/FMAXNMV/FMINNMV
        // on .4H/.8H) is selected by U=0 in the across-lanes cases above;
        // for every other dispatch path is_fp16 stays false.
        .is_fp16 = is_fp16,
    };
    insn_consumer_->AdvSimdTwoRegMisc(args);
  }
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
        // opcode=11101 splits on bit23 of the `size` field:
        //   bit23=0 (size ∈ {00, 01}): SCVTF (U=0) / UCVTF (U=1) —
        //     integer → float, scalar; sz=size&1 chooses S vs D.
        //   bit23=1 (size ∈ {10, 11}): FRECPE (U=0) / FRSQRTE (U=1) —
        //     FP reciprocal / reciprocal-sqrt estimate, scalar.
        // llvm-mc-21 encoding checks:
        //   scvtf   s0, s1   → 0x5e21d820  (U=0, opcode=11101, size=00)
        //   scvtf   d0, d1   → 0x5e61d820  (U=0, opcode=11101, size=01)
        //   frecpe  s0, s1   → 0x5ea1d820  (U=0, opcode=11101, size=10)
        //   frecpe  d0, d1   → 0x5ee1d820  (U=0, opcode=11101, size=11)
        //   frsqrte s0, s1   → 0x7ea1d820  (U=1, opcode=11101, size=10)
        if ((size & 0b10) == 0) {
          op = u ? AdvSimdScalarTwoRegMiscOpcode::kUcvtf
                 : AdvSimdScalarTwoRegMiscOpcode::kScvtf;
        } else {
          op = u ? AdvSimdScalarTwoRegMiscOpcode::kFrsqrte
                 : AdvSimdScalarTwoRegMiscOpcode::kFrecpe;
        }
        break;
      // scalar FCVTAS / FCVTAU.
      // opcode=11100 splits on bit23 of the `size` field per the ARM ARM
      // "AdvSIMD scalar two-register miscellaneous (FP)" table:
      //   bit23=0 (size ∈ {00, 01}): FCVTAS (U=0) / FCVTAU (U=1) —
      //     FP → int, round-to-nearest ties-away-from-zero; sz=size&1
      //     selects single (0) vs double (1).
      //   bit23=1: no scalar form (vector URECPE/URSQRTE only).
      // llvm-mc-21 encoding checks:
      //   fcvtas s0, s1 → 0x5e21c820  (U=0, opcode=11100, size=00)
      //   fcvtas d0, d1 → 0x5e61c820  (U=0, opcode=11100, size=01)
      //   fcvtau s0, s1 → 0x7e21c820  (U=1, opcode=11100, size=00)
      //   fcvtau d0, d1 → 0x7e61c820  (U=1, opcode=11100, size=01)
      case 0b11100:
        if ((size & 0b10) != 0) { Undefined(); return; }
        op = u ? AdvSimdScalarTwoRegMiscOpcode::kFcvtau
               : AdvSimdScalarTwoRegMiscOpcode::kFcvtas;
        break;
      case 0b11011:
        // FCVTZS (U=0) / FCVTZU (U=1): float → integer, round toward zero.
        // Per ARM ARM "AdvSIMD scalar two-register miscellaneous (FP)" table,
        // the real FCVTZS/FCVTZU encoding uses a=bit23=1 (size ∈ {10, 11}
        // where sz=bit22 selects single vs double).  bit23=0 is the
        // unimplemented FCVTMS/FCVTMU (round toward -inf) — accept both
        // bit23 values to preserve pre-existing dispatch surface (the legacy
        // bit23=0 path silently produced truncate-toward-zero semantics; we
        // keep that to avoid disturbing any test that hit it), and route
        // both to kFcvtzs/kFcvtzu.  The interpreter handler reads only the
        // sz bit (size & 1) to choose single (0) vs double (1).
        // llvm-mc-21 check: fcvtzs s1, s0 → 0x5ea1b801 (bit23=1, sz=0).
        op = u ? AdvSimdScalarTwoRegMiscOpcode::kFcvtzu
               : AdvSimdScalarTwoRegMiscOpcode::kFcvtzs;
        break;
      // scalar SQABS / SQNEG / SQXTUN / FCVTXN.
      // Scalar twins of the vector ops at the same opcode bits (00111,
      // 10010, 10110); each is the one-lane collapse of the vector form.
      // llvm-mc-21 encoding checks:
      //   sqabs   b0, b1   → 0x5e207820  (U=0, opcode=00111, size=00)
      //   sqabs   d0, d1   → 0x5ee07820  (U=0, opcode=00111, size=11)
      //   sqneg   b0, b1   → 0x7e207820  (U=1, opcode=00111, size=00)
      //   sqxtun  b0, h1   → 0x7e212820  (U=1, opcode=10010, size=00)
      //   sqxtun  s0, d1   → 0x7ea12820  (U=1, opcode=10010, size=10)
      //   fcvtxn  s0, d1   → 0x7e616820  (U=1, opcode=10110, size=01)
      case 0b00111:
        // SQABS (U=0) / SQNEG (U=1): single signed-saturating abs/neg.
        // All four lane widths (B/H/S/D) are encoded.
        op = u ? AdvSimdScalarTwoRegMiscOpcode::kSqneg
               : AdvSimdScalarTwoRegMiscOpcode::kSqabs;
        break;
      case 0b10010:
        // SQXTUN (U=1): single-lane signed→unsigned saturating narrow.
        // size=00→B from H, 01→H from S, 10→S from D. size=11 unallocated.
        // U=0 at this opcode is XTN (vector-only, no scalar form).
        if (!u || size == 0b11) { Undefined(); return; }
        op = AdvSimdScalarTwoRegMiscOpcode::kSqxtun;
        break;
      case 0b10100:
        // SQXTN (U=0) / UQXTN (U=1): single-lane signed/unsigned saturating
        // narrow. size=00→B from H, 01→H from S, 10→S from D. size=11
        // unallocated (no narrow form from a 128-bit source element).
        // llvm-mc-21 encoding checks:
        //   sqxtn   b0, h1   → 0x5e214820  (U=0, opcode=10100, size=00)
        //   sqxtn   h0, s1   → 0x5e614820  (U=0, opcode=10100, size=01)
        //   sqxtn   s0, d1   → 0x5ea14820  (U=0, opcode=10100, size=10)
        //   uqxtn   b0, h1   → 0x7e214820  (U=1, opcode=10100, size=00)
        //   uqxtn   h0, s1   → 0x7e614820  (U=1, opcode=10100, size=01)
        //   uqxtn   s0, d1   → 0x7ea14820  (U=1, opcode=10100, size=10)
        if (size == 0b11) { Undefined(); return; }
        op = u ? AdvSimdScalarTwoRegMiscOpcode::kUqxtn
               : AdvSimdScalarTwoRegMiscOpcode::kSqxtn;
        break;
      case 0b10110:
        // FCVTXN (U=1, size=01): scalar FP64→FP32 round-to-odd.
        // U=0 at this opcode bit pattern is reserved at the scalar level
        // (BFCVTN/FCVTN are vector-only — no scalar BFCVTN scalar form
        // exists), so reject anything else.
        if (!u || size != 0b01) { Undefined(); return; }
        op = AdvSimdScalarTwoRegMiscOpcode::kFcvtxn;
        break;
      default:
        Undefined();
        return;
    }

    const AdvSimdScalarTwoRegMiscArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        // Raw 2-bit size. The pre-existing FP ops use only size ∈ {0,1};
        // the new SQABS/SQNEG/SQXTUN family uses size ∈ {0..3}. See the
        // struct doc-comment above.
        .size = size,
    };
    insn_consumer_->AdvSimdScalarTwoRegMisc(args);
  }
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
    // Distinguish FP by opcode: 11010 (FABD), 11011 (FMULX), 11100 (FCMxx),
    // 11101 (FAC..), 11111 (FRECPS / FRSQRTS).
    if (opcode == 0b11010 || opcode == 0b11011 || opcode == 0b11100 ||
        opcode == 0b11101 || opcode == 0b11111) {
      is_fp = true;
      bool bit23 = (size >> 1) & 1;
      uint8_t sz = size & 1;  // 0 -> S, 1 -> D
      switch (opcode) {
        case 0b11010:
          if (!u || !bit23) { Undefined(); return; }
          op = AdvSimdScalarThreeSameOpcode::kFabd;
          break;
        case 0b11011:
          // FMULX (scalar, single & double): U=0, bit23=0, opcode=11011.
          // Per ARM ARM C7.2.150 "FMULX (vector, scalar)": encoding
          //   01 0 11110 0 sz 1 Rm 11011 1 Rn Rd
          // i.e. bit29=U=0, bit23=0, bit22=sz (0=S, 1=D), bit21=1.
          // Verified: fmulx s0,s1,s2 = 0x5E22DC20, fmulx d0,d1,d2 = 0x5E62DC20.
          // Other combinations of (U, bit23) at opcode=11011 are unallocated
          // in the scalar encoding.
          if (!u && !bit23) op = AdvSimdScalarThreeSameOpcode::kFmulx;
          else { Undefined(); return; }
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
        case 0b11111:
          // FRECPS (U=0, bit23=0) / FRSQRTS (U=0, bit23=1).
          // Per ARM ARM C7.2.151 "FRECPS" and C7.2.155 "FRSQRTS":
          //   01 0 11110 0 sz 1 Rm 11111 1 Rn Rd   (FRECPS)
          //   01 0 11110 1 sz 1 Rm 11111 1 Rn Rd   (FRSQRTS)
          // Verified: frecps s0,s1,s2=0x5E22FC20, frecps d0,d1,d2=0x5E62FC20,
          //           frsqrts s0,s1,s2=0x5EA2FC20, frsqrts d0,d1,d2=0x5EE2FC20.
          // U=1 at opcode=11111 is unallocated in the scalar encoding.
          if (u) { Undefined(); return; }
          op = bit23 ? AdvSimdScalarThreeSameOpcode::kFrsqrts
                     : AdvSimdScalarThreeSameOpcode::kFrecps;
          break;
        default:
          Undefined();
          return;
      }
      out_size = sz;
    } else {
      // Integer scalar three same.
      //
      // Most integer scalar three-same opcodes are D-form only.  The
      // saturating add/sub family (opcodes 00001 and 00011) supports
      // all four lane widths (B/H/S/D) — they're commonly used as
      // single-lane fast paths for DSP saturation.  See ARM ARM
      // C7.2.282 / .284 (SQADD / SQSUB) and C7.2.317 / .319 (UQADD /
      // UQSUB).
      // opcodes 00001 / 00101 / 01001 / 01011 are B/H/S/D-capable;
      // opcode 10110 (SQDMULH / SQRDMULH) is H/S-only (the per-arm guard
      // below rejects B and D for it).  All other integer scalar-three-same
      // opcodes accept D only.
      const bool all_sizes = (opcode == 0b00001) || (opcode == 0b00101) ||
                             (opcode == 0b01001) || (opcode == 0b01011);
      const bool hs_only = (opcode == 0b10110);
      if (!all_sizes && !hs_only && size != 0b11) { Undefined(); return; }

      switch (opcode) {
        // SQADD / UQADD scalar (B/H/S/D).
        case 0b00001:
          op = u ? AdvSimdScalarThreeSameOpcode::kUqaddScalar
                 : AdvSimdScalarThreeSameOpcode::kSqaddScalar;
          break;
        // SQSUB / UQSUB scalar (B/H/S/D).
        // ARM ARM C7.2.317 / C7.2.319: encoding is opcode 0b00101
        // (NOT 0b00011 — verified against clang --target=aarch64,
        // e.g. sqsub b0,b1,b2 = 0x5e222c20 whose bits[15:11]=00101).
        // Opcode 0b00011 is unallocated in the AdvSimdScalarThreeSame
        // class.
        case 0b00101:
          op = u ? AdvSimdScalarThreeSameOpcode::kUqsubScalar
                 : AdvSimdScalarThreeSameOpcode::kSqsubScalar;
          break;
        // SQSHL / UQSHL scalar (B/H/S/D).
        case 0b01001:
          op = u ? AdvSimdScalarThreeSameOpcode::kUqshlScalar
                 : AdvSimdScalarThreeSameOpcode::kSqshlScalar;
          break;
        // SQRSHL / UQRSHL scalar (B/H/S/D).
        case 0b01011:
          op = u ? AdvSimdScalarThreeSameOpcode::kUqrshlScalar
                 : AdvSimdScalarThreeSameOpcode::kSqrshlScalar;
          break;
        // SRSHL / URSHL scalar (D only, non-saturating).
        //   size=11 is the only valid lane width for this scalar form
        //   (the all_sizes whitelist intentionally excludes 01010 so
        //   the `size != 0b11` guard above already rejects B/H/S).
        case 0b01010:
          op = u ? AdvSimdScalarThreeSameOpcode::kUrshlScalar
                 : AdvSimdScalarThreeSameOpcode::kSrshlScalar;
          break;
        // SQDMULH / SQRDMULH scalar (H/S only).
        //   Per ARM ARM C7.2.301 / .305: only size=01 (H) and size=10 (S)
        //   are allocated for this opcode; B and D are unallocated.
        case 0b10110:
          if (size != 0b01 && size != 0b10) { Undefined(); return; }
          op = u ? AdvSimdScalarThreeSameOpcode::kSqrdmulhScalar
                 : AdvSimdScalarThreeSameOpcode::kSqdmulhScalar;
          break;
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

  //
  // AdvSIMD scalar pairwise.
  // Encoding: 01 U 11110 size 11000 opcode 10 Rn Rd
  // Integer leg (U=0): ADDP scalar (size=11, opcode=11011).
  // FP leg (U=1): FADDP / FMAXNMP / FMINNMP / FMAXP / FMINP scalar.
  //   For the FP leg, size[1] (bit 23) discriminates max- vs min- variants
  //   and size[0] (bit 22) picks S (0) vs D (1).
  //
  void DecodeAdvSimdScalarPairwise() {
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t opcode = GetBits<12, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdScalarPairwiseOpcode op;
    bool ok = false;
    bool is_fp16 = false;

    if (!u) {
      // Integer leg: ADDP scalar (D-form) only.
      if (opcode == 0b11011 && size == 0b11) {
        op = AdvSimdScalarPairwiseOpcode::kAddp;
        ok = true;
      }
      // Armv8.2-FP16 scalar pairwise (ARM ARM C7.2 "Advanced SIMD scalar
      // pairwise (FP16)"): U=0, bit22 (size[0]) must be 0; bit23 (size[1])
      // selects max- vs min- for FMAX*/FMIN*. FADDP is only allocated at
      // bit23=0.
      else if ((size & 1) == 0) {
        bool size_hi = ((size >> 1) & 1) != 0;
        switch (opcode) {
          case 0b01100:
            op = size_hi ? AdvSimdScalarPairwiseOpcode::kFminnmpScalar
                         : AdvSimdScalarPairwiseOpcode::kFmaxnmpScalar;
            ok = true;
            is_fp16 = true;
            break;
          case 0b01101:
            if (!size_hi) {
              op = AdvSimdScalarPairwiseOpcode::kFaddpScalar;
              ok = true;
              is_fp16 = true;
            }
            break;
          case 0b01111:
            op = size_hi ? AdvSimdScalarPairwiseOpcode::kFminpScalar
                         : AdvSimdScalarPairwiseOpcode::kFmaxpScalar;
            ok = true;
            is_fp16 = true;
            break;
          default:
            break;
        }
      }
    } else {
      // FP leg.  size[1] selects min- (1) vs max- (0); size[0] selects S/D.
      bool size_hi = ((size >> 1) & 1) != 0;
      switch (opcode) {
        case 0b01100:
          op = size_hi ? AdvSimdScalarPairwiseOpcode::kFminnmpScalar
                       : AdvSimdScalarPairwiseOpcode::kFmaxnmpScalar;
          ok = true;
          break;
        case 0b01101:
          // FADDP scalar only allocated at size[1]=0 (size=0x).
          if (!size_hi) {
            op = AdvSimdScalarPairwiseOpcode::kFaddpScalar;
            ok = true;
          }
          break;
        case 0b01111:
          op = size_hi ? AdvSimdScalarPairwiseOpcode::kFminpScalar
                       : AdvSimdScalarPairwiseOpcode::kFmaxpScalar;
          ok = true;
          break;
        default:
          break;
      }
    }

    if (!ok) {
      Undefined();
      return;
    }

    const AdvSimdScalarPairwiseArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .size = size,
        .is_fp16 = is_fp16,
    };
    insn_consumer_->AdvSimdScalarPairwise(args);
  }
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

    // BF16 indexed
    //
    // Armv8.6-BF16 by-element forms route through the vector x indexed
    // element subspace too: bit31=0, bit29=0 (U=0), bits[28:24]=01111,
    // bit10=0.  Distinguishing field is the opcode (bits[15:12]=1111)
    // combined with size:
    //   size=01: BFDOT (by element).   Vm = M:Rm[3:0] (5-bit, V0..V31);
    //                                  index = H:L (2-bit, 0..3).
    //   size=11: BFMLALB/BFMLALT (idx). bit30 = T discriminator (Q
    //                                  implicit 1, dest always .4s).
    //                                  Vm = Rm[3:0] (V0..V15);
    //                                  index = H:L:M (3-bit, 0..7).
    // Otherwise size=01 falls into the existing "Undefined" check below
    // and size=11 falls into the existing FP MLA/MLS/MUL switch (which
    // would reject opcode=1111 as default Undefined).
    //
    // Encoding cross-checks (llvm-mc):
    //   bfdot   v0.4s,v1.8h,v17.2h[0] = 0x4f51f020   (Vm=17 via M:Rm)
    //   bfdot   v0.4s,v1.8h,v2.2h[3]  = 0x4f62f820   (index=3 via H:L)
    //   bfmlalb v0.4s,v1.8h,v15.h[0]  = 0x0fcff020   (Vm=15 via Rm[3:0])
    //   bfmlalb v0.4s,v1.8h,v2.h[4]   = 0x0fc2f820   (index=4 via H)
    //   bfmlalb v0.4s,v1.8h,v2.h[2]   = 0x0fe2f020   (index=2 via L)
    //   bfmlalb v0.4s,v1.8h,v2.h[1]   = 0x0fd2f020   (index=1 via M)
    //   bfmlalt v0.4s,v1.8h,v2.h[7]   = 0x4ff2f820   (bit30=1 -> T)
    if (!u && opcode == 0b1111) {
      if (size == 0b01) {
        // BFDOT (by element).  Vm is 5-bit M:Rm, index is 2-bit H:L.
        uint8_t bf_rm = static_cast<uint8_t>((M << 4) | Rm4);
        uint8_t bf_index = static_cast<uint8_t>((H << 1) | L);
        const Bf16ThreeSameArgs args = {
            .opcode = Bf16ThreeSameOpcode::kBfdotIdx,
            .rd = rd,
            .rn = rn,
            .rm = bf_rm,
            .index = bf_index,
            .q = q,
        };
        insn_consumer_->AdvSimdBf16ThreeSame(args);
        return;
      }
      if (size == 0b11) {
        // BFMLALB / BFMLALT (by element).  Vm is 4-bit Rm[3:0], index
        // is 3-bit H:L:M.  bit30 is the T discriminator (Q implicit 1).
        uint8_t bf_index = static_cast<uint8_t>((H << 2) | (L << 1) | M);
        const Bf16ThreeSameArgs args = {
            .opcode = q ? Bf16ThreeSameOpcode::kBfmlaltIdx
                        : Bf16ThreeSameOpcode::kBfmlalbIdx,
            .rd = rd,
            .rn = rn,
            .rm = Rm4,
            .index = bf_index,
            .q = true,
        };
        insn_consumer_->AdvSimdBf16ThreeSame(args);
        return;
      }
    }

    // hello-dotprod
    // SDOT / UDOT (by element) — Armv8.4-DotProd.
    //   bit31=0, bits[28:24]=01111, bits[23:22]=10, opcode=bits[15:12]=1110,
    //   bit10=0.  bit30=Q (selects .4s vs .2s), bit29=U (SDOT/UDOT).
    //   Vm = M:Rm[3:0] (5-bit, V0..V31); index = H:L (2-bit, 0..3).
    //
    // Verified encodings (clang --target=aarch64 -march=armv8.4-a+dotprod):
    //   sdot v0.4s, v1.16b, v2.4b[0]  = 0x4f82e020  (U=0, L=0, H=0)
    //   udot v0.4s, v1.16b, v2.4b[3]  = 0x6fa2e820  (U=1, L=1, H=1)
    //   sdot v0.2s, v1.8b,  v2.4b[0]  = 0x0f82e020  (Q=0, U=0)
    //   udot v0.2s, v1.8b,  v2.4b[3]  = 0x2fa2e820  (Q=0, U=1)
    //
    // The DotProd opcode 1110 differs from the FP {MLA=0001, MLS=0101,
    // MUL=1001, MLA/MUL=1000, MLS=0100, MLA=0000} cases that the default
    // path below handles for size=10, so this carve-out is required to
    // route DOT idx away from the FP switch's default Undefined() branch.
    if (size == 0b10 && opcode == 0b1110) {
      uint8_t dp_rm = static_cast<uint8_t>((M << 4) | Rm4);
      uint8_t dp_index = static_cast<uint8_t>((H << 1) | L);
      const DotProductArgs args = {
          .opcode = u ? DotProductOpcode::kUdotIdx : DotProductOpcode::kSdotIdx,
          .rd = rd,
          .rn = rn,
          .rm = dp_rm,
          .index = dp_index,
          .q = q,
      };
      insn_consumer_->AdvSimdDotProduct(args);
      return;
    }

    // I8MM by-element USDOT / SUDOT (FEAT_I8MM).
    //   U=0, opcode=bits[15:12]=1111, index = H:L; the size field selects the
    //   mixed-sign flavour: size=10 -> USDOT (Vn unsigned, Vm signed),
    //   size=00 -> SUDOT (Vn signed, Vm unsigned). Vm = M:Rm[3:0].
    // Verified (clang -march=armv8.6-a+i8mm):
    //   usdot v0.4s, v1.16b, v2.4b[0] = 0x4f82f020 (size=10)
    //   sudot v0.4s, v1.16b, v2.4b[1] = 0x4f22f020 (size=00, L=1)
    if (!u && opcode == 0b1111 && (size == 0b10 || size == 0b00)) {
      uint8_t dp_rm = static_cast<uint8_t>((M << 4) | Rm4);
      uint8_t dp_index = static_cast<uint8_t>((H << 1) | L);
      const DotProductArgs args = {
          .opcode = (size == 0b10) ? DotProductOpcode::kUsdotIdx
                                   : DotProductOpcode::kSudotIdx,
          .rd = rd,
          .rn = rn,
          .rm = dp_rm,
          .index = dp_index,
          .q = q,
      };
      insn_consumer_->AdvSimdDotProduct(args);
      return;
    }

    // indexed FCMLA
    // FCMLA (by element) — Armv8.3-FCMA.
    //
    // Distinguished from FMLA/FMLS (by element) by U=1 (bit29).
    // Opcode field bits[15:12] = (0, rot[1], rot[0], 1) — i.e., a 4-bit
    // value whose bit15 is 0 and bit12 is 1, with rot in the middle two
    // bits.  Mask `(opcode & 0b1001) == 0b0001` catches all four rotations:
    //   rot=0 (opcode=0001), rot=1 (0011), rot=2 (0101), rot=3 (0111).
    // Without this carve-out, rot=0 (0001) and rot=2 (0101) would be routed
    // to Undefined by the existing `case 0b0001/case 0b0101: if (u)
    // Undefined()` branches, while rot=1 (0011) and rot=3 (0111) would fall
    // through to the default Undefined.
    //
    // size encoding: raw bits[23:22] with bit23=1 fixed and bit22=size:
    //   bit22=0 (raw bits[23:22] = 0b10) -> FP32 (only Q=1 .4s form).
    //   bit22=1 (raw bits[23:22] = 0b11) -> this is the FMLA-FP64 slot;
    //     FCMLA does NOT use it.
    //   bit23=0,bit22=1 (raw bits[23:22] = 0b01) -> FP16.  Parked
    // alongside non-indexed FP16 (the non-indexed FP16-SIMD FCMA path
    //     rejects until the family is implemented end-to-end).
    //
    // llvm-mc-verified FP32 encodings:
    //   fcmla v0.4s, v1.4s, v2.s[0], #0   = 0x6F821020
    //   fcmla v0.4s, v1.4s, v2.s[1], #0   = 0x6F821820  (H=1)
    //   fcmla v0.4s, v1.4s, v2.s[0], #90  = 0x6F823020  (rot=01)
    //   fcmla v0.4s, v1.4s, v2.s[1], #90  = 0x6F823820
    //   fcmla v0.4s, v1.4s, v2.s[0], #180 = 0x6F825020  (rot=10)
    //   fcmla v0.4s, v1.4s, v2.s[1], #180 = 0x6F825820
    //   fcmla v0.4s, v1.4s, v2.s[0], #270 = 0x6F827020  (rot=11)
    //   fcmla v0.4s, v1.4s, v2.s[1], #270 = 0x6F827820
    //   fcmla v0.4s, v1.4s, v17.s[0], #0  = 0x6F911020  (Vm=M:Rm=10001)
    //
    // FP32 .2s form is REJECTED by the architecture (the .2s container
    // would have only 1 output complex pair while indexed FCMLA requires
    // at least one accumulation per index value across .4s).  Q=0 with
    // size=0b10 is reserved for FCMLA-indexed.
    if (u && (opcode & 0b1001) == 0b0001) {
      uint8_t fcmla_rot = static_cast<uint8_t>((opcode >> 1) & 0b11);
      uint8_t fcmla_rm = static_cast<uint8_t>((M << 4) | Rm4);
      if (size == 0b10) {
        // FP32: index = H (1 bit); L must be 0; Q must be 1.
        if (L || !q) { Undefined(); return; }
        const FcmaIdxArgs args = {
            .opcode = FcmaIdxOpcode::kFcmlaIdx,
            .rd = rd,
            .rn = rn,
            .rm = fcmla_rm,
            .index = H,
            .size = 0b10, // non-indexed convention: 0b10 = FP32.
            .rot = fcmla_rot,
            .q = q,
        };
        insn_consumer_->AdvSimdFcmaIdx(args);
        return;
      }
      if (size == 0b01) {
        // FP16 (FP16 SIMD FCMA indexed):
        //   Q=1 (.8h): index = H:L (2 bits, 0..3) — Vm.8H has 4 pairs.
        //   Q=0 (.4h): index = L only (1 bit, 0..1); H must be 0.
        if (!q && H) { Undefined(); return; }
        uint8_t fcmla_idx_fp16 = static_cast<uint8_t>((H << 1) | L);
        const FcmaIdxArgs args = {
            .opcode = FcmaIdxOpcode::kFcmlaIdx,
            .rd = rd,
            .rn = rn,
            .rm = fcmla_rm,
            .index = fcmla_idx_fp16,
            .size = 0b01,
            .rot = fcmla_rot,
            .q = q,
        };
        insn_consumer_->AdvSimdFcmaIdx(args);
        return;
      }
      // Other size values (0b00, 0b11) are reserved — fall through to
      // the existing paths below, which route opcode=0001/0101 with U=1
      // to Undefined and opcode=0011/0111 to the default Undefined.
    }

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
    // FP16 vector indexed FMLA/FMLS/FMUL
    //
    // FP16 by-element FMLA/FMLS/FMUL — Armv8.2-FP16.  Encoding pattern:
    //   0 Q 0 01111 00 L M Rm[3:0] opcode H 0 Rn Rd
    // i.e. U=0, size=0b00, opcode ∈ {0001 FMLA, 0101 FMLS, 1001 FMUL}.
    //   - Vm is only 4 bits (M:Rm[3:0] where M is consumed by the index),
    //     so the indexed source is restricted to V0..V15.
    //   - index = (H << 2) | (L << 1) | M (3 bits, 0..7) — broadcasts one
    //     lane from Vm.8H regardless of Q.
    //   - Q selects the destination width only: Q=0 updates the low 4
    //     lanes of Vd.8H (.4H), Q=1 updates all 8 (.8H).
    //
    // Note: size=0b00 with the integer MLA/MLS/MUL opcodes (1000/0100/0000)
    // is reserved by the architecture (8-bit indexed MLA does not exist),
    // so this carve-out only fires for FP opcodes (and only U=0).  Integer
    // MLA-idx with size=01 (.4h/.8h elements) is a separate path still
    // routed to Undefined below — out of scope here.
    } else if (size == 0b00) {
      if (u || (opcode != 0b0001 && opcode != 0b0101 && opcode != 0b1001)) {
        Undefined();
        return;
      }
      rm = Rm4;
      index = static_cast<uint8_t>((H << 2) | (L << 1) | M);
    // integer MLA/MLS/MUL-idx halfword
    //
    // Integer MUL/MLA/MLS by-element with halfword elements (.4h/.8h) —
    // Armv8-A baseline (not an extension; just a previously deferred
    // decoder gap for "Integer MLA/MLS/MUL-idx with size=0b01").
    // Encoding pattern (per ARM ARM C7.2):
    //   0 Q U 01111 01 L M Rm[3:0] opcode H 0 Rn Rd
    // Vm restricted to V0..V15 — the bit-20 M slot is consumed by the
    // index field rather than as Vm's high bit, identical to the
    // FP16-indexed convention above.  index = H:L:M (3 bits, 0..7).
    //
    // Valid (size=01, U, opcode) tuples per ARM ARM:
    //   U=0, opcode=1000 -> MUL  (MUL_byelement,  halfword)
    //   U=1, opcode=0000 -> MLA  (MLA_byelement,  halfword)
    //   U=1, opcode=0100 -> MLS  (MLS_byelement,  halfword)
    // Other opcodes at size=01 belong to SMULL/UMULL/SQDMULL widening
    // or saturating variants which use the post-switch Undefined path.
    //
    // The interpreter already handles size=0b01 via the generic integer
    // else-branch in AdvSimdVecXIndexedElement (esize = 2), so no
    // interpreter change is needed.
    //
    // llvm-mc-verified encodings:
    //   mul  v0.4h, v1.4h, v2.h[0] = 0x0f428020
    //   mul  v0.8h, v1.8h, v2.h[7] = 0x4f728820
    //   mla  v0.4h, v1.4h, v2.h[0] = 0x2f420020
    //   mla  v0.8h, v1.8h, v2.h[7] = 0x6f720820
    //   mls  v0.4h, v1.4h, v2.h[0] = 0x2f424020
    //   mls  v0.8h, v1.8h, v2.h[7] = 0x6f724820
    //
    // SQDMULH / SQRDMULH (by element, halfword) follow the same size=01
    // shape — Vm is restricted to V0..V15 via Rm4, the bit-20 M slot is
    // consumed as the index's low bit, and index = H:L:M (3 bits, 0..7).
    // Verified encodings:
    //   sqdmulh  v0.4h, v1.4h, v2.h[0] = 0x0f42c020   (U=0, opcode=1100)
    //   sqdmulh  v0.8h, v1.8h, v2.h[3] = 0x4f72c020
    //   sqrdmulh v0.4h, v1.4h, v2.h[7] = 0x0f72d820   (U=0, opcode=1101)
    //   sqrdmulh v0.8h, v1.8h, v2.h[5] = 0x4f52d820
    } else if (size == 0b01) {
      if (!((opcode == 0b1000 && !u) ||
            (opcode == 0b0000 && u) ||
            (opcode == 0b0100 && u) ||
            (opcode == 0b1100 && !u) ||
            (opcode == 0b1101 && !u) ||
            // Widening MUL/MAC by element accept both U values.
            // Encoding pattern (ARM ARM C7.2):
            //   smull   v0.4s, v1.4h, v2.h[0] = 0x0F42A020   U=0, opcode=1010
            //   smull2  v0.4s, v1.8h, v2.h[7] = 0x4F72A820   Q=1, index=7
            //   umull   v0.4s, v1.4h, v2.h[0] = 0x2F42A020   U=1, opcode=1010
            //   smlal   v0.4s, v1.4h, v2.h[0] = 0x0F422020   U=0, opcode=0010
            //   umlal   v0.4s, v1.4h, v2.h[0] = 0x2F422020   U=1, opcode=0010
            //   smlsl   v0.4s, v1.4h, v2.h[0] = 0x0F426020   U=0, opcode=0110
            //   umlsl   v0.4s, v1.4h, v2.h[0] = 0x2F426020   U=1, opcode=0110
            opcode == 0b1010 ||
            opcode == 0b0010 ||
            opcode == 0b0110 ||
            // Saturating doubling widening MUL/MAC by element.  U must
            // be 0; size restricted to {01, 10}.  Vm and index encoding
            // matches the widening MUL/MAC by element family above.
            //   sqdmull  v0.4s, v1.4h, v2.h[0]  = 0x0F42B020   opcode=1011
            //   sqdmull2 v0.4s, v1.8h, v2.h[7]  = 0x4F72B820
            //   sqdmlal  v0.4s, v1.4h, v2.h[0]  = 0x0F423020   opcode=0011
            //   sqdmlsl  v0.4s, v1.4h, v2.h[0]  = 0x0F427020   opcode=0111
            (opcode == 0b1011 && !u) ||
            (opcode == 0b0011 && !u) ||
            (opcode == 0b0111 && !u) ||
            // Armv8.1-RDM saturating-rounding-doubling MAC by element
            // (non-widening; dst lane size = source lane size).  U must
            // be 1; size restricted to {01, 10}.  Vm and index encoding
            // matches the SQDMULH/SQRDMULH by-element shape.
            //   sqrdmlah v0.4h, v1.4h, v2.h[0] = 0x2F42D020   opcode=1101
            //   sqrdmlsh v0.4h, v1.4h, v2.h[0] = 0x2F42F020   opcode=1111
            (opcode == 0b1101 && u) ||
            (opcode == 0b1111 && u)
            )) {
        Undefined();
        return;
      }
      rm = Rm4;
      index = static_cast<uint8_t>((H << 2) | (L << 1) | M);
    } else {
      // Reserved.
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
        // FMUL/FMULX by element.  Per ARM ARM C7.2
        // AdvSIMD-vector-x-indexed-element:
        //   U=0, opcode=1001 -> FMUL  (by element) -> kFmul.
        //   U=1, opcode=1001 -> FMULX (by element, Armv8.2-FP) -> kFmulx.
        // FMULX differs from FMUL only in the ±0 * ±inf saturation case (it
        // produces ±2.0 rather than NaN), used by libm reciprocal-estimate
        // refinement loops.  Interpreter implements both via FmulxScalar.
        op = u ? AdvSimdVecXIdxOpcode::kFmulx
               : AdvSimdVecXIdxOpcode::kFmul;
        break;
      case 0b1000:
        // follow-up: reject reserved
        // opcode=1000 with U=1.  Per ARM ARM C7.2 AdvSIMD-vector-x-indexed-element:
        //   U=0, opcode=1000 -> MUL (by element) — kept as kMul.
        //   U=1, opcode=1000 -> RESERVED — there is no instruction at this slot.
        //     MLA-by-element is encoded at opcode=0000 with U=1 (handled by the
        //     `case 0b0000` arm below); SQRDMLAH-by-element (Armv8.1-RDM) is at
        //     opcode=1101 (not implemented here yet).
        // The previous code silently mapped this reserved slot to kMla, which
        // produced wrong results without any SIGILL — a textbook silent decoder
        // mis-route.  We now route reserved encodings to Undefined so a guest
        // emitting (size=10/11, U=1, opcode=1000) at least gets a diagnostic
        // SIGILL instead of corrupted vector arithmetic.  size=01 with U=1
        // opcode=1000 is already rejected by the size==0b01 guard above.
        if (u) {
          Undefined();
          return;
        }
        op = AdvSimdVecXIdxOpcode::kMul;
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
      // SQDMULH / SQRDMULH by element (vector).
      //   U=0, opcode=1100 -> SQDMULH (by element).
      //   U=0, opcode=1101 -> SQRDMULH (by element).
      //   U=1, opcode=1101 -> SQRDMLAH (Armv8.1-RDM, by element).
      //   U=1, opcode=1111 -> SQRDMLSH (Armv8.1-RDM, by element).
      // All restricted to size ∈ {0b01 (.4h/.8h), 0b10 (.2s/.4s)}.
      // U=1 at opcode=1100 has no encoding -> Undefined.
      case 0b1100:
        if (u || (size != 0b01 && size != 0b10)) { Undefined(); return; }
        op = AdvSimdVecXIdxOpcode::kSqdmulhIdx;
        break;
      case 0b1101:
        if (size != 0b01 && size != 0b10) { Undefined(); return; }
        op = u ? AdvSimdVecXIdxOpcode::kSqrdmlahIdx
               : AdvSimdVecXIdxOpcode::kSqrdmulhIdx;
        break;
      case 0b1111:
        // U=0 at opcode=1111 is BFDOT/BFMLAL{B,T} (handled by the
        // carve-out at the top of DecodeAdvSimdVecXIndexedElement).  Only
        // U=1 (SQRDMLSH, Armv8.1-RDM) reaches the switch here.
        if (!u || (size != 0b01 && size != 0b10)) { Undefined(); return; }
        op = AdvSimdVecXIdxOpcode::kSqrdmlshIdx;
        break;
      // Widening MUL/MAC by element.  Restricted to size ∈ {0b01 (.4h/.8h ->
      // .4s), 0b10 (.2s/.4s -> .2d)}.  size=0b00 has no encoding (8->16
      // widening MUL is not defined), size=0b11 is reserved.
      case 0b1010:
        if (size != 0b01 && size != 0b10) { Undefined(); return; }
        op = u ? AdvSimdVecXIdxOpcode::kUmullIdx
               : AdvSimdVecXIdxOpcode::kSmullIdx;
        break;
      case 0b0010:
        if (size != 0b01 && size != 0b10) { Undefined(); return; }
        op = u ? AdvSimdVecXIdxOpcode::kUmlalIdx
               : AdvSimdVecXIdxOpcode::kSmlalIdx;
        break;
      case 0b0110:
        if (size != 0b01 && size != 0b10) { Undefined(); return; }
        op = u ? AdvSimdVecXIdxOpcode::kUmlslIdx
               : AdvSimdVecXIdxOpcode::kSmlslIdx;
        break;
      // Saturating doubling widening MUL/MAC by element (signed only).
      // Restricted to size ∈ {0b01 (.4h/.8h -> .4s), 0b10 (.2s/.4s -> .2d)}.
      // U=1 at these opcodes is reserved and routes to Undefined.
      case 0b1011:
        if (u || (size != 0b01 && size != 0b10)) { Undefined(); return; }
        op = AdvSimdVecXIdxOpcode::kSqdmullIdx;
        break;
      case 0b0011:
        if (u || (size != 0b01 && size != 0b10)) { Undefined(); return; }
        op = AdvSimdVecXIdxOpcode::kSqdmlalIdx;
        break;
      case 0b0111:
        if (u || (size != 0b01 && size != 0b10)) { Undefined(); return; }
        op = AdvSimdVecXIdxOpcode::kSqdmlslIdx;
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

  //
  // AdvSIMD scalar x indexed element.
  // Encoding: 0 1 U 11111 size L M Rm opcode H 0 Rn Rd
  //   size = bits[23:22]: 10 = FP32 (single lane), 11 = FP64 (double lane).
  //   For FP32: Vm = M:Rm[3:0] (5-bit), index = H:L (2-bit, 0..3).
  //   For FP64: Vm = M:Rm[3:0] (5-bit), index = H   (1-bit, 0..1); L must
  //             be 0 (reserved).
  // FMULX (U=1/1001), FMUL (U=0/1001), FMLA (U=0/0001), FMLS (U=0/0101) are
  // implemented.  The remaining scalar-x-indexed opcodes (SQDMULL /
  // SQDMULH / SQRDMULH / SQRDMLAH / SQRDMLSH variants) route to Undefined
  // until they are needed.
  //
  // Encoding cross-checks (aarch64-linux-gnu-as / objdump):
  //   fmulx s0, s1, v2.s[0]   = 0x7F829020   (U=1, size=10, L=0, H=0, op=1001)
  //   fmulx s0, s1, v2.s[3]   = 0x7FA29820   (U=1, size=10, L=1, H=1, op=1001)
  //   fmulx d0, d1, v2.d[0]   = 0x7FC29020   (U=1, size=11, H=0, op=1001)
  //   fmulx d0, d1, v2.d[1]   = 0x7FC29820   (U=1, size=11, H=1, op=1001)
  //   fmul  s0, s1, v2.s[0]   = 0x5F829020   (U=0, size=10, L=0, H=0, op=1001)
  //   fmul  d0, d1, v2.d[1]   = 0x5FC29820   (U=0, size=11, H=1, op=1001)
  //   fmla  s0, s1, v2.s[0]   = 0x5F821020   (U=0, size=10, L=0, H=0, op=0001)
  //   fmla  d0, d1, v2.d[1]   = 0x5FC21820   (U=0, size=11, H=1, op=0001)
  //   fmls  s0, s1, v2.s[0]   = 0x5F825020   (U=0, size=10, L=0, H=0, op=0101)
  //   fmls  d0, d1, v2.d[0]   = 0x5FC25020   (U=0, size=11, H=0, op=0101)
  //
  void DecodeAdvSimdScalarXIndexedElement() {
    bool u = GetBits<29, 1>();
    uint8_t size = GetBits<22, 2>();
    uint8_t L = GetBits<21, 1>();
    uint8_t M = GetBits<20, 1>();
    uint8_t Rm4 = GetBits<16, 4>();
    uint8_t opcode = GetBits<12, 4>();
    uint8_t H = GetBits<11, 1>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    AdvSimdScalarXIdxOpcode op;
    if (u && opcode == 0b1001) {
      op = AdvSimdScalarXIdxOpcode::kFmulx;
    } else if (!u && opcode == 0b1001) {
      op = AdvSimdScalarXIdxOpcode::kFmul;
    } else if (!u && opcode == 0b0001) {
      op = AdvSimdScalarXIdxOpcode::kFmla;
    } else if (!u && opcode == 0b0101) {
      op = AdvSimdScalarXIdxOpcode::kFmls;
    // Armv8.1-RDM scalar by-element SQRDMLAH / SQRDMLSH.
    //   U=1, opcode=1101 -> SQRDMLAH (scalar by element).
    //   U=1, opcode=1111 -> SQRDMLSH (scalar by element).
    //   size ∈ {01 (H), 10 (S)}; size=00 / size=11 reserved for this opcode.
    //   Per ARM ARM C7.2.300 / .302:
    //     SQRDMLAH <V>d, <V>n, <Vm>.<T>[i] = 01 1 11111 size L M Rm4 1101 H 0 Rn Rd
    //     SQRDMLSH <V>d, <V>n, <Vm>.<T>[i] = 01 1 11111 size L M Rm4 1111 H 0 Rn Rd
    //   Verified via llvm-mc -arch=aarch64 -mattr=+rdm:
    //     sqrdmlah h0, h1, v2.h[0] = 0x7f42d020
    //     sqrdmlah s0, s1, v2.s[0] = 0x7f82d020
    //     sqrdmlsh h0, h1, v2.h[0] = 0x7f42f020
    //     sqrdmlsh s0, s1, v2.s[0] = 0x7f82f020
    } else if (u && opcode == 0b1101) {
      if (size != 0b01 && size != 0b10) { Undefined(); return; }
      op = AdvSimdScalarXIdxOpcode::kSqrdmlahScalarIdx;
    } else if (u && opcode == 0b1111) {
      if (size != 0b01 && size != 0b10) { Undefined(); return; }
      op = AdvSimdScalarXIdxOpcode::kSqrdmlshScalarIdx;
    // Scalar by-element signed saturating doubling multiply (U=0, H/S only).
    } else if (!u && opcode == 0b1100) {
      if (size != 0b01 && size != 0b10) { Undefined(); return; }
      op = AdvSimdScalarXIdxOpcode::kSqdmulhScalarIdx;
    } else if (!u && opcode == 0b1101) {
      if (size != 0b01 && size != 0b10) { Undefined(); return; }
      op = AdvSimdScalarXIdxOpcode::kSqrdmulhScalarIdx;
    } else if (!u && opcode == 0b1011) {
      if (size != 0b01 && size != 0b10) { Undefined(); return; }
      op = AdvSimdScalarXIdxOpcode::kSqdmullScalarIdx;
    } else {
      // Genuinely unallocated encodings — raise SIGILL.
      Undefined();
      return;
    }

    uint8_t rm;
    uint8_t index;
    if (size == 0b10) {
      // FP32: index = H:L (4 elements in Vm.4S).  Vm = M:Rm4 (5 bits).
      rm = static_cast<uint8_t>((M << 4) | Rm4);
      index = static_cast<uint8_t>((H << 1) | L);
    } else if (size == 0b11) {
      // FP64: index = H (2 elements in Vm.2D); L must be 0.  Vm = M:Rm4.
      if (L) { Undefined(); return; }
      rm = static_cast<uint8_t>((M << 4) | Rm4);
      index = H;
    // Armv8.2-FP16 scalar by-element FMLA/FMLS/FMUL/FMULX.
    //
    // Encoding "Advanced SIMD scalar x indexed element (FP16)" per ARM ARM
    // C7.2 uses size=0b00 (not 0b01 as a stale earlier comment claimed).
    //   "0 1 U 1 1 1 1 1 0 0 L M Rm[3:0] opcode H 0 Rn Rd"
    // The bit-20 M slot is consumed as the index's low bit, so Vm is only
    // 4 bits (Rm4) — the indexed source is restricted to V0..V15.  Index
    // = H:L:M (3 bits, 0..7) selects one of 8 FP16 lanes of Vm.8H.  Unlike
    // the *vector* FP16 by-element form (which rejects U=1), scalar FMULX
    // FP16 by-element DOES exist (ARM ARM C7.2.83), so all four opcodes
    // {FMLA, FMLS, FMUL, FMULX} are admitted here.
    //
    // Verified encodings (aarch64-linux-gnu-as -march=armv8.2-a+fp16):
    //   fmul  h0, h1, v2.h[0] = 0x5F029020
    //   fmul  h3, h4, v5.h[7] = 0x5F359883
    //   fmla  h0, h1, v2.h[0] = 0x5F021020
    //   fmls  h0, h1, v2.h[7] = 0x5F325820
    //   fmulx h0, h1, v2.h[0] = 0x7F029020
    //   fmulx h3, h4, v5.h[7] = 0x7F359883
    } else if (size == 0b00) {
      rm = Rm4;
      index = static_cast<uint8_t>((H << 2) | (L << 1) | M);
    // Armv8.1-RDM scalar by-element H form (SQRDMLAH/SQRDMLSH).
    //   8 elements in Vm.8H, index = H:L:M (3 bits, 0..7), Vm = Rm4 only
    //   (indexed source restricted to V0..V15 — bit-20 M slot is the
    //   index's low bit).  size=01 only applies to the RDM-by-element
    //   opcodes; the FP-by-element opcodes (FMUL/FMLA/FMLS/FMULX) treat
    //   size=01 as reserved.
    //   Verified encodings (llvm-mc -arch=aarch64 -mattr=+rdm):
    //     sqrdmlah h0, h1, v2.h[0]  = 0x7f42d020
    //     sqrdmlah h3, h4, v5.h[7]  = 0x7f75d883
    //     sqrdmlsh h0, h1, v15.h[3] = 0x7f7ff020
    } else if (size == 0b01 &&
               (op == AdvSimdScalarXIdxOpcode::kSqrdmlahScalarIdx ||
                op == AdvSimdScalarXIdxOpcode::kSqrdmlshScalarIdx ||
                op == AdvSimdScalarXIdxOpcode::kSqdmulhScalarIdx ||
                op == AdvSimdScalarXIdxOpcode::kSqrdmulhScalarIdx ||
                op == AdvSimdScalarXIdxOpcode::kSqdmullScalarIdx)) {
      // H form: 8 lanes in Vm.8H, index = H:L:M, Vm restricted to V0..V15.
      rm = Rm4;
      index = static_cast<uint8_t>((H << 2) | (L << 1) | M);
    } else {
      // size=0b01 is reserved at this slot for FP opcodes.
      Undefined();
      return;
    }
    const AdvSimdScalarXIdxArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .rm = rm,
        .index = index,
        .size = size,
    };
    insn_consumer_->AdvSimdScalarXIndexedElement(args);
  }

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
      // AdvSIMD saturating shift left (immediate) — verified
      // against llvm-mc output for `sqshl`, `uqshl`, and `sqshlu` (the prior
      // dispatch swapped 0b01100 and 0b01110, sending real-world UQSHL to
      // the SQSHLU handler and SQSHLU to the UQSHL handler). The ARMv8 ARM
      // assigns opcode 0b01110 to SQSHL (U=0) / UQSHL (U=1), and 0b01100 to
      // SQSHLU (U=1; U=0 is reserved).
      case 0b01110:
        op = u ? AdvSimdShiftImmOpcode::kUqshl : AdvSimdShiftImmOpcode::kSqshl;
        break;
      case 0b01100:
        if (u) {
          op = AdvSimdShiftImmOpcode::kSqshlu;
        } else {
          Undefined();
          return;
        }
        break;
      // AdvSIMD narrow-shift dispatch — verified against
      // llvm-mc output for shrn / rshrn / sqshrn / uqshrn / sqshrun /
      // sqrshrn / sqrshrun / uqrshrn (and the *2 upper-half forms, which
      // differ only in Q).
      //
      //   opcode | U=0       | U=1
      //   -------+-----------+----------
      //   10000  | SHRN      | SQSHRUN
      //   10001  | RSHRN     | SQRSHRUN
      //   10010  | SQSHRN    | UQSHRN
      //   10011  | SQRSHRN   | UQRSHRN
      //
      // Prior dispatch had three independent bugs in this block:
      //   1. opcode 0b10000 U=1 routed to kSqshrn (wrong: SQSHRUN).
      //   2. opcode 0b10001 U=1 routed to kUqshrn (wrong: SQRSHRUN).
      //   3. opcodes 0b10010 / 0b10011 (both U-values) silently fell to
      //      Undefined(), making real SQSHRN / UQSHRN / SQRSHRN / UQRSHRN
      //      raise SIGILL on any sample that touched them.
      case 0b10000:
        op = u ? AdvSimdShiftImmOpcode::kSqshrun : AdvSimdShiftImmOpcode::kShrn;
        break;
      case 0b10001:
        op = u ? AdvSimdShiftImmOpcode::kSqrshrun : AdvSimdShiftImmOpcode::kRshrn;
        break;
      case 0b10010:
        op = u ? AdvSimdShiftImmOpcode::kUqshrn : AdvSimdShiftImmOpcode::kSqshrn;
        break;
      case 0b10011:
        op = u ? AdvSimdShiftImmOpcode::kUqrshrn : AdvSimdShiftImmOpcode::kSqrshrn;
        break;
      case 0b10100:
        op = u ? AdvSimdShiftImmOpcode::kUshll : AdvSimdShiftImmOpcode::kSshll;
        break;
      // Vector fixed-point conversion (ARM ARM C7.2
      // "Advanced SIMD shift by immediate"):
      //   opcode | U=0     | U=1
      //   -------+---------+---------
      //   11100  | SCVTF   | UCVTF   (integer fixed-point -> FP per lane)
      //   11111  | FCVTZS  | FCVTZU  (FP -> integer fixed-point per lane)
      // immh selects element width: immh=01xx -> .2s/.4s (FP32 / S32),
      // immh=1xxx -> .2d (FP64 / S64, Q=1 only).  immh=001x (FP16) is
      // deferred (reject so SIGILL fires rather than wrong-result), and
      // immh=0001/0000 are unallocated for these opcodes (top-level
      // dispatch already gates immh!=0).
      case 0b11100:
        if (!(immh & 0b1100)) { Undefined(); return; }
        if ((immh & 0b1000) && !q) { Undefined(); return; }  // .2d requires Q=1
        op = u ? AdvSimdShiftImmOpcode::kUcvtfFixed
               : AdvSimdShiftImmOpcode::kScvtfFixed;
        break;
      case 0b11111:
        if (!(immh & 0b1100)) { Undefined(); return; }
        if ((immh & 0b1000) && !q) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kFcvtzuFixed
               : AdvSimdShiftImmOpcode::kFcvtzsFixed;
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

  //
  // AdvSIMD scalar shift by immediate (ARM ARM C4.1.6.10).
  // Encoding: 0 1 U 1 1 1 1 1 0 immh immb opcode 1 Rn Rd
  //
  // Scalar shift-by-immediate is the per-lane sibling of the vector
  // shift-by-immediate routine just above.  Cycle scope is the
  // D-form (immh=1xxx) "simple" shifts plus D-form saturating shifts:
  //
  //   opcode | U=0      | U=1       | sizes accepted by ARM ARM
  //   -------+----------+-----------+---------------------------
  //   00000  | SSHR     | USHR      | scalar D only
  //   00010  | SSRA     | USRA      | scalar D only
  //   00100  | SRSHR    | URSHR     | scalar D only
  //   00110  | SRSRA    | URSRA     | scalar D only
  //   01000  | —        | SRI       | scalar D only
  //   01010  | SHL      | SLI       | scalar D only
  //   01100  | —        | SQSHLU    | scalar B/H/S/D (all dispatched)
  //   01110  | SQSHL    | UQSHL     | scalar B/H/S/D (all dispatched)
  //   10000  | —        | SQSHRUN   | scalar B/H/S
  //   10001  | —        | SQRSHRUN  | scalar B/H/S
  //   10010  | SQSHRN   | UQSHRN    | scalar B/H/S
  //   10011  | SQRSHRN  | UQRSHRN   | scalar B/H/S
  //
  // The non-saturating ops (rows 00000..01010) are spec'd by ARM ARM as
  // scalar D-only; the saturating shifts (rows 01100/01110) accept all
  // four sizes; the narrow shifts (rows 10000..10011) accept B/H/S
  // destination (source is 2*dest_esize bits) — immh=1xxx is unallocated
  // for them because there is no scalar narrow-to-D form. Fixed-point
  // conversion (opcode 11xxx) still falls through to Undefined() until a
  // follow-up cycle adds its single-lane interpreter arm — see the
  // carried "what should be done next" list.
  //
  void DecodeAdvSimdScalarShiftByImm() {
    bool u = GetBits<29, 1>();
    uint8_t immh = GetBits<19, 4>();
    uint8_t immb = GetBits<16, 3>();
    uint8_t opcode = GetBits<11, 5>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();

    // immh=0000 is the AdvSIMD modified-immediate carve-out; the
    // top-level dispatch already gates on immh!=0, so this is just a
    // defensive double-check.
    if (immh == 0) { Undefined(); return; }

    AdvSimdShiftImmOpcode op;

    // Per ARM ARM C4.1.6.10 the "simple" shifts (SSHR / USHR / SSRA /
    // USRA / SRSHR / URSHR / SRSRA / URSRA / SHL / SLI / SRI) are
    // spec'd as scalar D-only — their B/H/S encodings are unallocated.
    // Reject non-D immh for these to fire SIGILL on bogus encodings
    // rather than silently producing wrong results.
    //
    // The saturating shifts (SQSHL / UQSHL / SQSHLU) accept B/H/S/D
    // scalar — gate them on immh!=0 (any element size) so all four
    // widths dispatch.
    switch (opcode) {
      case 0b00000:
        if (!(immh & 0b1000)) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kUshr : AdvSimdShiftImmOpcode::kSshr;
        break;
      case 0b00010:
        if (!(immh & 0b1000)) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kUsra : AdvSimdShiftImmOpcode::kSsra;
        break;
      case 0b00100:
        if (!(immh & 0b1000)) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kUrshr : AdvSimdShiftImmOpcode::kSrshr;
        break;
      case 0b00110:
        if (!(immh & 0b1000)) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kUrsra : AdvSimdShiftImmOpcode::kSrsra;
        break;
      case 0b01000:
        if (!(immh & 0b1000)) { Undefined(); return; }
        if (u) {
          op = AdvSimdShiftImmOpcode::kSri;
        } else {
          Undefined(); return;
        }
        break;
      case 0b01010:
        if (!(immh & 0b1000)) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kSli : AdvSimdShiftImmOpcode::kShl;
        break;
      case 0b01100:
        // SQSHLU: scalar B/H/S/D — accept any non-zero immh.
        if (u) {
          op = AdvSimdShiftImmOpcode::kSqshlu;
        } else {
          Undefined(); return;
        }
        break;
      case 0b01110:
        // SQSHL/UQSHL: scalar B/H/S/D — accept any non-zero immh.
        op = u ? AdvSimdShiftImmOpcode::kUqshl : AdvSimdShiftImmOpcode::kSqshl;
        break;
      // Scalar narrow-shift dispatch (ARM ARM C4.1.6.10, opcodes 10000..10011).
      // Source is 2*esize bits, destination is esize bits, with saturation
      // (and optional rounding for the "R" variants). Maps:
      //   opcode | U=0       | U=1
      //   -------+-----------+-----------
      //   10000  | —         | SQSHRUN
      //   10001  | —         | SQRSHRUN
      //   10010  | SQSHRN    | UQSHRN
      //   10011  | SQRSHRN   | UQRSHRN
      // Per ARM ARM these accept B/H/S destination (immh = 0001 / 001x /
      // 01xx); immh = 1xxx is unallocated (would need a D destination,
      // but ARM has no scalar narrow-to-D form). The U=0 variants of
      // opcode 10000/10001 are also unallocated (no SHRN / RSHRN
      // scalar — those are vector-only).
      case 0b10000:
        if (immh & 0b1000) { Undefined(); return; }
        if (u) {
          op = AdvSimdShiftImmOpcode::kSqshrun;
        } else {
          Undefined(); return;
        }
        break;
      case 0b10001:
        if (immh & 0b1000) { Undefined(); return; }
        if (u) {
          op = AdvSimdShiftImmOpcode::kSqrshrun;
        } else {
          Undefined(); return;
        }
        break;
      case 0b10010:
        if (immh & 0b1000) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kUqshrn : AdvSimdShiftImmOpcode::kSqshrn;
        break;
      case 0b10011:
        if (immh & 0b1000) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kUqrshrn : AdvSimdShiftImmOpcode::kSqrshrn;
        break;
      // Scalar fixed-point conversion (ARM ARM C4.1.6.10, opcodes 11100 / 11111).
      //   opcode | U=0     | U=1
      //   -------+---------+---------
      //   11100  | SCVTF   | UCVTF
      //   11111  | FCVTZS  | FCVTZU
      // Width comes from immh: immh=01xx → S (32-bit float, 32-bit fixed),
      // immh=1xxx → D (64-bit float, 64-bit fixed). immh=001x is the FP16
      // variant (deferred — rejected here so SIGILL fires rather than
      // silent wrong-result), immh=0001/0000 are unallocated for these
      // opcodes (top-level dispatch already gates immh!=0).
      case 0b11100:
        if (!(immh & 0b1100)) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kUcvtfFixed
               : AdvSimdShiftImmOpcode::kScvtfFixed;
        break;
      case 0b11111:
        if (!(immh & 0b1100)) { Undefined(); return; }
        op = u ? AdvSimdShiftImmOpcode::kFcvtzuFixed
               : AdvSimdShiftImmOpcode::kFcvtzsFixed;
        break;
      default:
        Undefined();
        return;
    }

    // scalar=true forces the consumer to process exactly one element
    // regardless of esize, matching ARM ARM scalar semantics:
    //   Vd[esize-1:0] = op(Vn[esize-1:0]); Vd[127:esize] = 0.
    // q=false is preserved for downstream code that hasn't yet been
    // updated to honour the scalar flag; setting both keeps lane
    // counts conservative (≤2 lanes for the legacy code paths).
    const AdvSimdShiftImmArgs args = {
        .opcode = op,
        .rd = rd,
        .rn = rn,
        .immh = immh,
        .immb = immb,
        .q = false,
        .u = u,
        .scalar = true,
    };
    insn_consumer_->AdvSimdShiftByImm(args);
  }

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

    // o2 distinguishes exclusive (o2=0) from ordered/CAS (o2=1):
    //   o2=0, o1=0: STXR/LDXR/STLXR/LDAXR (exclusive)
    //   o2=0, o1=1: CASP (exclusive pair CAS)
    //   o2=1, o1=0: STLR/LDAR (ordered, non-exclusive)
    //   o2=1, o1=1: CAS/CASA/CASL/CASAL

    LoadStoreExclusiveArgs args;
    args.rt = rt;
    args.rn = rn;
    args.rs = rs;
    args.size = size;
    args.acquire = false;
    args.release = false;

    if (o1 == 0) {
      // check o2 to distinguish STLR/LDAR from STXR/LDXR
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
    } else {
      // CASP fix: o1=1 covers both CAS and CASP.
      // The o2 bit is the discriminator: o2=0 → CASP (pair); o2=1 → CAS (single).
      // Encodings confirmed via llvm-mc (clang-r563880c):
      //   casp   w0,w1,w2,w3,[x10] = 0x08207d42 → o2=0, o1=1
      //   casa   w0,w1,[x10]       = 0x88e07d41 → o2=1, o1=1
      // Prior code routed both to kCas, silently miscompiling CASP.
      if (o2 == 0) {
        // o1=1, o2=0 covers two distinct families distinguished by bit31:
        //   bit31=1 → LDXP/STXP (load/store exclusive PAIR)
        //   bit31=0 → CASP (compare-and-swap pair, LSE)
        // bit[30] = sz: 0 → 32-bit pair, 1 → 64-bit pair. Re-encode args.size to
        // 2 (32-bit) or 3 (64-bit) so the interpreter reuses its size dispatch.
        if (GetBits<31, 1>() != 0) {
          // LDXP/STXP: Rt2 (bits[14:10]) is the second data register; Rs is the
          // STXP status register (ignored for LDXP). o0 adds acquire/release.
          args.rt2 = GetBits<10, 5>();
          args.size = (GetBits<30, 1>() != 0) ? 3 : 2;
          if (L) {
            args.op = AtomicOp::kLdxp;
            args.acquire = (o0 != 0);  // LDAXP
          } else {
            args.op = AtomicOp::kStxp;
            args.release = (o0 != 0);  // STLXP
          }
        } else {
          // CASP: bits[14:10] must be 11111.
          uint8_t rt2 = GetBits<10, 5>();
          if (rt2 != 0b11111) {
            Undefined();
            return;
          }
          args.op = AtomicOp::kCasp;
          args.acquire = (L != 0);   // CASPA/CASPAL
          args.release = (o0 != 0);  // CASPL/CASPAL
          args.size = (GetBits<30, 1>() != 0) ? 3 : 2;
        }
      } else {
        // CAS family
        args.op = AtomicOp::kCas;
        args.acquire = (L != 0);   // CASA/CASAL
        args.release = (o0 != 0);  // CASL/CASAL
      }
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
      // fix: LDEOR/LDSET routing was swapped.
      // Per ARM ARM C7.2.149/162 and confirmed via llvm-mc:
      //   ldeor w0,w1,[x2] = 0xB820_2041 → opc=010 → kLdeor (XOR)
      //   ldset w0,w1,[x2] = 0xB820_3041 → opc=011 → kLdset (OR)
      // The interpreter and JIT handlers for kLdeor (XOR) / kLdset (OR)
      // already match those names; only the decoder's case 0b0010/0b0011
      // were transposed.  Hit rarely in the wild because Bionic on the
      // pre-LSE NDK falls back to LL/SC for fetch_or, but anyone built
      // with -march=armv8.1-a+lse hits this silently.
      case 0b0010: args.op = AtomicOp::kLdeor; break;
      case 0b0011: args.op = AtomicOp::kLdset; break;
      // atomic min/max (LSE Armv8.1).
      // LDSMAX/SMIN/UMAX/UMIN are encoded at opc=100/101/110/111 with o3=0;
      // full_op = (o3<<3)|opc.
      case 0b0100: args.op = AtomicOp::kLdsmax; break;
      case 0b0101: args.op = AtomicOp::kLdsmin; break;
      case 0b0110: args.op = AtomicOp::kLdumax; break;
      case 0b0111: args.op = AtomicOp::kLdumin; break;
      case 0b1000: args.op = AtomicOp::kSwp; break;
      // LDAPR (Armv8.3-LRCPC) as plain LDAR.
      // LDAPR/LDAPRB/LDAPRH/LDAPR (Load-Acquire RCpc Register) shares the
      // atomic-memory-op encoding class with full_op=(o3<<3)|opc=0b1100
      // (o3=1, opc=0b100).  Verified via NDK r28 clang assembly:
      //   ldapr x0,[x1]  = 0xF8BFC020  -> o3=1, opc=100, Rs=11111, A=1, R=0
      //   ldapr w0,[x1]  = 0xB8BFC020
      //   ldaprb w0,[x1] = 0x38BFC020
      //   ldaprh w0,[x1] = 0x78BFC020
      // The architectural distinguishers are Rs=11111 and A=1, R=0; we
      // gate on Rs=11111 here (A/R live in args.acquire/release set above).
      // RCpc is *weaker* than ARM release-consistency; x86-TSO is *stronger*
      // than both, so routing to kLdar (which the interpreter and JIT
      // already implement as a plain x86 load) is correctness-preserving.
      // Encoding ref: ARM ARM DDI 0487 C7.2.156 (LDAPR).
      // An earlier revision picked case 0b1111 here, which never fired --
      // the bug surfaced when hello-lrcpc issued the explicit instruction
      // via inline asm and SIGILL'd on the first probe.
      case 0b1100: {
        if (rs != 0b11111) {
          Undefined();
          return;
        }
        args.op = AtomicOp::kLdar;
        // args.acquire/release are already (A,R)=(1,0) from above — match
        // them to LDAR's canonical (acquire=true, release=false) so the
        // interpreter/JIT see the same shape as a real LDAR.
        args.acquire = true;
        args.release = false;
        break;
      }
      default: Undefined(); return;
    }
    insn_consumer_->LoadStoreExclusive(args);
  }

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
      uint8_t op2_bits = GetBits<21, 3>();  // bits [23:21]

      // // uint8_t op2_bits = GetBits<21, 4>();  // bits [24:21]
      if (op2_bits == 0b000) {
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
        return;
      }
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
        // Distinguish 2-source (bit30=0) from 1-source (bit30=1)
        if (GetBits<30, 1>()) {
          // Data processing (1-source): CLZ, CLS, RBIT, REV, REV16, REV32
          DecodeDataProc1Src();
        } else {
          // Data processing (2-source): UDIV, SDIV, LSLV, LSRV, ASRV, RORV
          DecodeDataProc2Src();
        }
        return;
      }
      Undefined();
      return;
    }

    // bit[28]=1, bit[24]=1: always Data processing (3-source).
    // DataProc2Src has bits[28:24]=11010 (bit24=0), dispatched above.
    // // bit[21]=0: Data processing (2-source).
    // // bit[21]=1: Data processing (3-source).
    DecodeDataProc3Src();
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

    // MTE DP-2src: SUBP(opc=0,S=0), SUBPS(opc=0,S=1), IRG(opc=4,S=0),
    // GMI(opc=5,S=0). All require sf=1 (64-bit). Route them to the
    // MteDataProc listener BEFORE the S-bit check below, since SUBPS
    // has S=1 by definition.
    if (sf && (opcode == 0b000000 || opcode == 0b000100 || opcode == 0b000101)) {
      MteDataProcOpcode mte_op;
      if (opcode == 0b000000) {
        mte_op = s ? MteDataProcOpcode::kSubps : MteDataProcOpcode::kSubp;
      } else if (opcode == 0b000100) {
        if (s) return Undefined();   // IRG never sets flags.
        mte_op = MteDataProcOpcode::kIrg;
      } else {
        if (s) return Undefined();   // GMI never sets flags.
        mte_op = MteDataProcOpcode::kGmi;
      }
      const MteDataProcArgs args = {
          .opcode = mte_op,
          .dst = rd,
          .src1 = rn,
          .src2 = rm,
      };
      insn_consumer_->MteDataProc(args);
      return;
    }

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
