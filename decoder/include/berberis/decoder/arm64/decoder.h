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

  // region digitalis
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
  };

  struct MteLoadStoreArgs {
    MteLoadStoreOpcode opcode;
    uint8_t rn;       // base; rn=31 means SP.
    uint8_t rt;       // data/dest; rt=31 means XZR.
    int32_t imm;      // sign-extended imm9 << 4 (already scaled by 16-byte granule).
    uint8_t op2;      // 0b00=offset-no-wb (LDG), 0b01=post, 0b10=offset, 0b11=pre.
  };
  // endregion

  // region digitalis
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
  // endregion

  // region digitalis indexed FCMLA
  // Advanced SIMD complex floating-point by element (Armv8.3-FCMA): FCMLA.
  //
  // FCADD has no by-element form; only FCMLA has an indexed encoding.
  //
  // Encoding (ARM ARM C7.2.86, verified via llvm-mc — handoff-58):
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
  // llvm-mc-verified encodings (handoff-58):
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
  // FP16-indexed FCMLA is parked alongside non-indexed FP16 (handoff-49
  // rejects size==FP16 as "no Digitalis FP16-SIMD FCMA yet").  Even
  // though handoff-57 added FP16 vector three-same support (), the
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
  // endregion

  // region digitalis
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
  // Handoff-51 follow-ups (Armv8.6-BF16 surface closeout):
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
  // endregion

  // region digitalis hello-dotprod
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
  };

  struct DotProductArgs {
    DotProductOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t index;  // 0..3 for indexed forms; 0 for vector forms.
    bool q;         // True selects .4s form (4 lanes); false selects .2s (2 lanes).
  };
  // endregion

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
    // region digitalis CASP (compare-and-swap pair, Armv8.1 LSE).
    kCasp,     // Compare and swap pair (Rs:Rs+1 = expected, Rt:Rt+1 = new)
    // endregion
    kSwp,      // Swap
    kLdadd,    // Atomic add
    kLdclr,    // Atomic bit clear
    kLdset,    // Atomic bit set
    kLdeor,    // Atomic exclusive or
    // region digitalis atomic min/max (LSE Armv8.1).
    kLdsmax,   // Atomic signed max
    kLdsmin,   // Atomic signed min
    kLdumax,   // Atomic unsigned max
    kLdumin,   // Atomic unsigned min
    // endregion
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
    kFabdV,     // FABD   (vector): op_high=1, opcode=11010, U=1
    // region digitalis
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
    // endregion
    // region digitalis - SABD/UABD: vector absolute difference at .8b/.16b/
    // .4h/.8h/.2s/.4s. size=11 (64-bit lane) is reserved.  Verified with
    // llvm-mc:  sabd v0.8b,v1.8b,v2.8b = 0x0e227420 (opcode=01110, U=0);
    //           uabd v0.8b,v1.8b,v2.8b = 0x2e227420 (opcode=01110, U=1).
    kSabd,      // SABD (vector): U=0, opcode=01110
    kUabd,      // UABD (vector): U=1, opcode=01110
    // endregion
    // region digitalis - SABA/UABA: absolute-difference-and-accumulate at
    // .8b/.16b/.4h/.8h/.2s/.4s.  size=11 reserved.  Vd[i] += |Vn[i] - Vm[i]|.
    // Verified with llvm-mc:
    //   saba v0.8b,v1.8b,v2.8b = 0x0e227c20 (opcode=01111, U=0);
    //   uaba v0.8b,v1.8b,v2.8b = 0x2e227c20 (opcode=01111, U=1).
    kSaba,      // SABA (vector): U=0, opcode=01111
    kUaba,      // UABA (vector): U=1, opcode=01111
    // endregion
    // region digitalis - PMUL polynomial multiply (byte-lane GF(2) multiply).
    // .8b/.16b only; size=01/10/11 reserved per ARM ARM.
    // Verified with llvm-mc:
    //   pmul v0.8b,v1.8b,v2.8b   = 0x2e229c20 (opcode=10011, U=1, size=00, Q=0)
    //   pmul v0.16b,v1.16b,v2.16b = 0x6e229c20 (opcode=10011, U=1, size=00, Q=1)
    //   pmul v0.4h / v0.4s        invalid (size=01/10 reserved)
    kPmul,      // PMUL (vector): U=1, opcode=10011, size=00
    // endregion
    // region digitalis - SQDMULH / SQRDMULH saturating doubling multiply high.
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
    // endregion
    // endregion
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
    // region digitalis: Armv8.2-FP16 NEON vector three-same.
    // FP16 vector three-same has a separate encoding (bit21=0, bits[15:14]=00,
    // bit22=1) from the standard three-same (bit21=1).  The decoder maps both
    // through this struct and the interpreter dispatches on this flag.
    bool is_fp16;
    // endregion
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
    // region digitalis - polynomial multiply (ARMv8 base PMULL + crypto PMULL64)
    // PMULL/PMULL2: U=0, opcode=1110
    //   size=00: 8-bit element poly-mul, 8 lanes -> 8x 16-bit results
    //   size=11: 64-bit element poly-mul, 1 lane -> 128-bit result (PMULL64)
    //   size=01,10 are RESERVED. Used by libz CRC32 acceleration.
    kPmull,
    // endregion
    // region digitalis - narrowing high (add/sub of two wide vectors, take high
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
    // endregion
    // region digitalis - narrowing high subtract (subtract two wide vectors,
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
    // endregion
    // region digitalis - signed saturating doubling multiply long.
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
    // endregion
    // region digitalis - signed saturating doubling multiply-accumulate long.
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
    // endregion
    // region digitalis - signed saturating doubling multiply-subtract long.
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
    // region digitalis
    // BFCVT <Hd>, <Sn>: FP32 single -> BFloat16 with round-to-nearest-even.
    // Encoded with ftype=01 (the 6-bit opcode + ftype together discriminate
    // this from the FCVT-from-double cases above).  llvm-mc-verified:
    //   bfcvt h0, s1  =  0x1e634020
    kBfcvt = 0b000110,
    // endregion
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

  // region digitalis
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
  // endregion

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
    kFcvtzsV,   // FCVTZS (vector, FP→int trunc): U=0, opcode=11011, bit23=1
    kFcvtzuV,   // FCVTZU (vector, FP→int trunc): U=1, opcode=11011, bit23=1
    kFrecpeV,   // FRECPE (vector): U=0, opcode=11101, bit23=1
    kFrsqrteV,  // FRSQRTE (vector): U=1, opcode=11101, bit23=1
    kFsqrtV,    // FSQRT  (vector): U=1, opcode=11111, bit23=1
    // region digitalis - FCVT* vector rounding-mode variants.
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
    // endregion
    // region digitalis BFCVTN/BFCVTN2 (Armv8.6-BF16).
    // Vector narrow FP32 -> BF16. Encoding shares opcode=10110 with FCVTN,
    // but uses size=10 (vs FCVTN's size=00/01). bit30 = Q: Q=0 -> BFCVTN
    // (writes low 4H of Vd, upper 64 bits zeroed); Q=1 -> BFCVTN2 (writes
    // upper 4H of Vd, low 64 bits preserved).
    //   llvm-mc: bfcvtn  v0.4h, v1.4s  = 0x0ea16820
    //            bfcvtn2 v0.8h, v1.4s  = 0x4ea16820
    kBfcvtn,    // BFCVTN/BFCVTN2 (vector narrow FP32->BF16).
    // endregion
    // region digitalis a=0 column: FP16 vector FRINT* (round to
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
    // endregion
    // region digitalis - SQABS / SQNEG (signed saturating abs / negate).
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
    // endregion
    // endregion
  };

  // region digitalis - SHA-512 (FEAT_SHA512) ops live outside the AdvSIMD
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
  // endregion

  struct AdvSimdTwoRegMiscArgs {
    AdvSimdTwoRegMiscOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t size;     // element size: 00=8b, 01=16b, 10=32b, 11=64b
    bool q;           // Q bit: 0=64-bit vector, 1=128-bit vector
    bool u;           // U bit from encoding
    // region digitalis: Armv8.2-FP16 vector two-reg-misc.
    // The FP16 encoding (DDI 0487 C7.2 "Advanced SIMD two-register
    // miscellaneous (FP16)") shares this struct.  When set, lanes are
    // 2-byte half and `size` carries no meaning (set to 0 by the FP16
    // dispatcher).  The interpreter reads is_fp16 inside each opcode
    // case and dispatches via FpHalfToSingle / FpSingleToHalf as.
    bool is_fp16;
    // endregion
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
    kFmulx,   // FMULX (scalar, FP): U=0, bit23=1, opcode=11011.
              // Identical to FMUL except (±0 * ±inf) lanes return ±2.0
              // instead of NaN — see Interpreter::FmulxScalar<>.
    kFrecps,  // FRECPS (scalar, FP): U=0, bit23=0, opcode=11111.
              // Reciprocal step: (2.0 - a*b) with (0*inf) -> +2.0 saturation.
              // FP16 scalar variant: a=0, U=0, opcode_3=111.
    kFrsqrts, // FRSQRTS (scalar, FP): U=0, bit23=1, opcode=11111.
              // Reciprocal sqrt step: (3.0 - a*b)/2 with (0*inf) -> +1.5
              // saturation. FP16 scalar variant: a=1, U=0, opcode_3=111.
    // endregion
  };

  struct AdvSimdScalarThreeSameArgs {
    AdvSimdScalarThreeSameOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;
    uint8_t size;   // integer: full size field; FP: 0 -> S (32-bit), 1 -> D (64-bit);
                    // FP16 scalar three-same: 0 (unused — interpreter checks is_fp16).
    // region digitalis: Armv8.2-FP16 scalar three-same.  When true, the
    // interpreter reads the low 16 bits of Vn/Vm as binary16 and performs
    // a widen-op-narrow round-trip through FP32 (FpHalfToSingle ->
    // semantic helper -> FpSingleToHalf).  Bit-exact for the FMULX
    // saturation case because ±2.0 is exactly representable in FP16.
    bool is_fp16 = false;
    // endregion
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
    // region digitalis: AdvSIMD shift-by-immediate narrow family — the
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
    // endregion
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
    // region digitalis
    kFmulx,   // FMULX (by element): U=1, opcode=1001.  Same lane semantics as
              // kFmul except (±0 * ±inf) lanes return ±2.0 instead of NaN.
    // endregion
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

  // region digitalis
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
  };

  struct AdvSimdScalarXIdxArgs {
    AdvSimdScalarXIdxOpcode opcode;
    uint8_t rd;
    uint8_t rn;
    uint8_t rm;       // indexed source register
    uint8_t index;    // element index within rm
    uint8_t size;     // 10 = FP32 (single), 11 = FP64 (double)
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
        // region digitalis hint audit (HINT #N = CRm:op2[2:0]).
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
        // explicit BTI audit (handoff-45): BTI guards live in
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
        // endregion
        insn_consumer_->Nop();
        return;
      }
      if (crn == 0b0011) {
        // region digitalis barrier audit (CRn=0011).
        // Memory and synchronization barriers.  Op2 selects the variant:
        //   CLREX = CRn=0011, CRm=imm,  op2=010
        //   DSB   = CRn=0011, CRm=opt,  op2=100  (incl. "DFB" — full)
        //   DMB   = CRn=0011, CRm=opt,  op2=101
        //   ISB   = CRn=0011, CRm=imm,  op2=110
        //   SB    = CRn=0011, CRm=0000, op2=111  (Armv8.5-SB)
        //   TSB CSYNC = CRn=0011, CRm=0010, op2=010 (Armv8.4-TRBE)
        // x86_64 already provides total store order with locked atomics
        // (see kCas/kSwp/kLdadd handlers in interpreter.h and
        // lite_translator.h: every LSE op uses `lock` or `xchg`).  ISB
        // is unnecessary because we never patch code in-flight from the
        // guest's perspective; the JIT cache is invalidated through
        // the translator's own mechanism, not via guest ISB.  CLREX
        // clears the LL/SC exclusive monitor, which we don't model
        // (Digitalis emulates LDXR/STXR pairs as cmpxchg, see
        //). Routing the whole CRn=0011 class to Nop is therefore
        // correct.
        // endregion
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
    // region digitalis PAuth BR/BLR/RET variants (Armv8.3-PAuth).
    // op3 = bits[15:10] distinguishes the PAC variants from the plain ones:
    //   000000 = no PAC; 000010 = A-key PAC; 000011 = B-key PAC.
    // BRAAZ/BRABZ (opc=0000) and BLRAAZ/BLRABZ (opc=0001) already route
    // correctly through the existing BR/BLR cases since Rn carries the
    // actual target and op4 is ignored.  RETAA/RETAB (opc=0010, op3!=0)
    // need the implicit LR (X30) as the source — the encoding hardcodes
    // Rn=11111 (XZR) which would otherwise be misrouted.  BRAA/BRAB
    // (opc=1000) and BLRAA/BLRAB (opc=1001) currently bail to Undefined.
    // PAC modifier in op4/Rm is ignored: Digitalis never inserts PAC bits
    // into pointers (see handoff-30), so Rn already holds the clean
    // target — no masking is needed.
    uint8_t op3 = GetBits<10, 6>();
    bool is_pac = (op3 == 0b000010 || op3 == 0b000011);
    // endregion

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
        // region digitalis: RETAA/RETAB are encoded with Rn=11111
        // but the architectural source register is implicitly X30 (LR).
        uint8_t src = is_pac ? 30 : rn;
        // endregion
        const BranchRegArgs args = {
            .src = src,
            .link_reg = 0,
            .is_link = false,
            .is_ret = true,
        };
        insn_consumer_->BranchReg(args);
        break;
      }
      // region digitalis BRAA/BRAB/BLRAA/BLRAB (Armv8.3-PAuth).
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
      // endregion
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
        // region digitalis
        // MTE load/store memory tags (LDG/STG/ST2G/STZG/STZ2G):
        //   bits[31:24]=11011001, bit[21]=1.
        // op_29=0 distinguishes from ordinary LDR/STR (unsigned imm), which
        // has op_29=1. size==11 is required by the MTE encoding; other
        // bits[31:30] with op_29=0 here are unallocated per ARM ARM.
        if (op_29 == 0 && GetBits<30, 2>() == 0b11 && GetBits<21, 1>()) {
          DecodeLoadStoreMemTag();
          return;
        }
        // endregion
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

  // region digitalis
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
  // endregion

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

    // region digitalis: Armv8.2-FP16 scalar three-same.
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
    // endregion

    // region digitalis
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
    // endregion

    // region digitalis
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
    // endregion

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
    // Floating-point conditional compare: bit31=0, bits[28:24]=11110, bit21=1, bits[11:10]=01
    // Encoding: 0 0 0 11110 ftype 1 Rm cond 01 Rn op nzcv  (op: 0=FCCMP, 1=FCCMPE)
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b01) {
      DecodeFpConditionalCompare();
      return;
    }
    // endregion

    // region digitalis
    // FCSEL: bit31=0, bits[28:24]=11110, bit21=1, bits[11:10]=11
    if (!bit31 && GetBits<24, 5>() == 0b11110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b11) {
      DecodeFpCondSelect();
      return;
    }

    // Floating-point data-processing (3 source): bit31=0, bit30=0, bit29=0,
    // bits[28:24]=11111.  FMADD, FMSUB, FNMADD, FNMSUB.
    // region digitalis: require bit30=0 (M) and bit29=0 (S) per ARM ARM
    // encoding "M=0 S=0 11111 ftype o1 0 Rm o0 Ra Rn Rd".  Without these
    // constraints the prefix also catches the "AdvSIMD scalar x indexed
    // element" family (bit30=1, bits[28:24]=11111) — e.g. FMULX scalar
    // by-element (U=1, opcode=1001) was silently mis-routed into
    // FpDataProc3 as garbage FMADD/FMSUB, producing wrong math without
    // any SIGILL.  Tightening here routes the scalar-x-indexed encodings
    // to the final Undefined() (no implementation yet) so the failure
    // mode is a diagnostic SIGILL rather than corrupted arithmetic.
    // endregion
    if (!bit31 && !GetBits<30, 1>() && !GetBits<29, 1>() &&
        GetBits<24, 5>() == 0b11111) {
      DecodeFpDataProc3();
      return;
    }
    // endregion

    // region digitalis
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
    // endregion

    // region digitalis
    // AdvSIMD three different: bit31=0, bits[28:24]=01110, bit21=1, bits[11:10]=00
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<21, 1>() && GetBits<10, 2>() == 0b00) {
      DecodeAdvSimdThreeDiff();
      return;
    }
    // endregion

    // region digitalis
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
    // endregion

    // region digitalis
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
    // endregion

    // region digitalis hello-dotprod
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
    // endregion

    // region digitalis: Armv8.2-FP16 NEON vector three-same.
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
    // endregion

    // AdvSIMD three same: bit31=0, bits[28:24]=01110, bit21=1, bit10=1
    if (!bit31 && GetBits<24, 5>() == 0b01110 && GetBits<21, 1>() && GetBits<10, 1>()) {
      DecodeAdvSimdThreeSame();
      return;
    }

    // region digitalis
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
    // endregion

    // region digitalis
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
    // endregion

    // region digitalis
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
    // endregion

    // region digitalis: Armv8.2-FP16 NEON vector two-register miscellaneous.
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
    // endregion

    // region digitalis
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
    uint8_t op2 = GetBits<16, 5>();  // ARM ARM "opcode2" (bits 20:16)
    uint8_t opcode2 = GetBits<10, 6>();
    uint8_t rn = GetBits<5, 5>();
    uint8_t rd = GetBits<0, 5>();
    // Dispatch based on opcode2
    // 000000 = RBIT, 000001 = REV16, 000010 = REV32(32-bit)/REV(64-bit),
    // 000011 = REV(64-bit only), 000100 = CLZ, 000101 = CLS
    // region digitalis PAuth DP-1Src as identity
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
      // Passing rn here (as the pre-handoff-41 code did) caused PACIA/AUTIA
      // and the rest of the on-register PAC family to overwrite Xd with the
      // value of Xn, silently miscompiling any PAuth probe / verifier.
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
    // endregion
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

    // region digitalis
    // ftype: 00=S, 01=D, 11=H (Armv8.2-FP16).  10 is reserved.
    if (ftype == 0b10) { Undefined(); return; }
    // endregion

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

    // region digitalis
    // ftype: 00=S, 01=D, 11=H (Armv8.2-FP16).  10 is reserved.
    if (ftype == 0b10) { Undefined(); return; }
    // endregion

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

    // Only ftype 00 (single) and 01 (double) supported.
    if (ftype >= 2) { Undefined(); return; }

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
  // endregion

  // region digitalis
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
  // accepted (handoff-61, FP16 SIMD FCMA). Interpreter promotes
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
  // endregion

  // region digitalis hello-dotprod
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
  // endregion

  // region digitalis
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
  // llvm-mc-verified encodings (handoff-51):
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
  // endregion

  // region digitalis
  //
  // AdvSIMD three same.
  // Encoding: 0 Q U 01110 size 1 Rm opcode 1 Rn Rd
  //   Q = bit30, U = bit29
  //   size = bits[23:22], Rm = bits[20:16]
  //   opcode = bits[15:11], Rn = bits[9:5], Rd = bits[4:0]
  //
  // region digitalis: Armv8.2-FP16 vector three-same.
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
          // region digitalis: FP16 FMULX (a=0, U=0, opcode_3=011).
          case 0b011: op = AdvSimdThreeSameOpcode::kFmulxV;  ok = true; break;
          // endregion
          case 0b100: op = AdvSimdThreeSameOpcode::kFcmeqV;  ok = true; break;
          case 0b110: op = AdvSimdThreeSameOpcode::kFmaxV;   ok = true; break;
          // region digitalis: FP16 FRECPS (a=0, U=0, opcode_3=111).
          case 0b111: op = AdvSimdThreeSameOpcode::kFrecpsV; ok = true; break;
          // endregion
          default: break;  // 101 reserved — Undefined.
        }
      } else {
        switch (opcode_3) {
          case 0b000: op = AdvSimdThreeSameOpcode::kFminnmV; ok = true; break;
          case 0b001: op = AdvSimdThreeSameOpcode::kFmlsV;   ok = true; break;
          case 0b010: op = AdvSimdThreeSameOpcode::kFsubV;   ok = true; break;
          case 0b110: op = AdvSimdThreeSameOpcode::kFminV;   ok = true; break;
          // region digitalis: FP16 FRSQRTS (a=1, U=0, opcode_3=111).
          case 0b111: op = AdvSimdThreeSameOpcode::kFrsqrtsV; ok = true; break;
          // endregion
          default: break;  // 011 reserved, 100 reserved, 101 reserved — Undefined.
        }
      }
    } else {
      if (!a) {
        switch (opcode_3) {
          case 0b011: op = AdvSimdThreeSameOpcode::kFmulV;   ok = true; break;
          case 0b100: op = AdvSimdThreeSameOpcode::kFcmgeV;  ok = true; break;
          case 0b101: op = AdvSimdThreeSameOpcode::kFacgeV;  ok = true; break;
          case 0b111: op = AdvSimdThreeSameOpcode::kFdivV;   ok = true; break;
          default: break;  // 000 FMAXNMP, 010 FADDP, 110 FMAXP — Undefined.
        }
      } else {
        switch (opcode_3) {
          case 0b010: op = AdvSimdThreeSameOpcode::kFabdV;   ok = true; break;
          case 0b100: op = AdvSimdThreeSameOpcode::kFcmgtV;  ok = true; break;
          case 0b101: op = AdvSimdThreeSameOpcode::kFacgtV;  ok = true; break;
          default: break;  // 000 FMINNMP, 011 reserved, 110 FMINP, 111 reserved — Undefined.
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
  // endregion

  // region digitalis: Armv8.2-FP16 scalar three-same.
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
      // region digitalis: FP16 scalar FRECPS (a=0, U=0, opcode_3=111).
      op = AdvSimdScalarThreeSameOpcode::kFrecps;
      ok = true;
      // endregion
    } else if (a && !u && opcode_3 == 0b111) {
      // region digitalis: FP16 scalar FRSQRTS (a=1, U=0, opcode_3=111).
      op = AdvSimdScalarThreeSameOpcode::kFrsqrts;
      ok = true;
      // endregion
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
  // endregion

  // region digitalis: Armv8.2-FP16 vector two-register miscellaneous.
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
  // pairwise-reject pattern (handoff-57).
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
          // region digitalis a=0/a=1 columns: FRINT*/FCVT*-round.
          case 0b11000: op = AdvSimdTwoRegMiscOpcode::kFrintpV;  ok = true; break;  // FRINTP
          case 0b11001: op = AdvSimdTwoRegMiscOpcode::kFrintzV;  ok = true; break;  // FRINTZ
          case 0b11010: op = AdvSimdTwoRegMiscOpcode::kFcvtpsV;  ok = true; break;  // FCVTPS
          case 0b11011: op = AdvSimdTwoRegMiscOpcode::kFcvtzsV;  ok = true; break;  // FCVTZS
          case 0b11101: op = AdvSimdTwoRegMiscOpcode::kFrecpeV;  ok = true; break;  // FRECPE
          // endregion
          default: break;
        }
      } else {
        switch (opcode_5) {
          case 0b01100: op = AdvSimdTwoRegMiscOpcode::kCmgeZero; ok = true; break;  // FCMGE #0
          case 0b01101: op = AdvSimdTwoRegMiscOpcode::kCmleZero; ok = true; break;  // FCMLE #0
          case 0b01111: op = AdvSimdTwoRegMiscOpcode::kFneg;     ok = true; break;  // FNEG
          case 0b11111: op = AdvSimdTwoRegMiscOpcode::kFsqrtV;   ok = true; break;  // FSQRT
          // region digitalis a=0/a=1 columns.
          case 0b11001: op = AdvSimdTwoRegMiscOpcode::kFrintiV;  ok = true; break;  // FRINTI
          case 0b11010: op = AdvSimdTwoRegMiscOpcode::kFcvtpuV;  ok = true; break;  // FCVTPU
          case 0b11011: op = AdvSimdTwoRegMiscOpcode::kFcvtzuV;  ok = true; break;  // FCVTZU
          case 0b11101: op = AdvSimdTwoRegMiscOpcode::kFrsqrteV; ok = true; break;  // FRSQRTE
          // endregion
          default: break;
        }
      }
    } else {
      // region digitalis a=0 column: FRINTN/A, FRINTM/X,
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
      // endregion
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
  // endregion

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
      // region digitalis: U=1 -> PMUL polynomial multiply (size=00 only).
      // Previously U=1 routed to Undefined(); now dispatched to kPmul.
      if (u) {
        if (size != 0b00) { Undefined(); return; }
        op = AdvSimdThreeSameOpcode::kPmul;
      } else {
        op = AdvSimdThreeSameOpcode::kMul;
      }
      // endregion
    } else if (opcode == 0b10010) {
      op = u ? AdvSimdThreeSameOpcode::kMls : AdvSimdThreeSameOpcode::kMla;
    // region digitalis: SQDMULH (U=0) / SQRDMULH (U=1) saturating doubling
    // multiply high. size=00 and size=11 are reserved per ARM ARM.
    } else if (opcode == 0b10110) {
      if (size == 0b00 || size == 0b11) { Undefined(); return; }
      op = u ? AdvSimdThreeSameOpcode::kSqrdmulh : AdvSimdThreeSameOpcode::kSqdmulh;
    // endregion
    // region digitalis
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
            if (u) { ok = false; break; }   // FADDP not implemented
            op = AdvSimdThreeSameOpcode::kFaddV;
            break;
          case 0b11011:
            // region digitalis: U=0 -> FMULX (kFmulxV), U=1 -> FMUL (kFmulV).
            // Previously U=0 routed to ok=false; the interpreter now grows
            // the FMULX special case (±0 * ±inf -> ±2.0) so we can dispatch.
            op = u ? AdvSimdThreeSameOpcode::kFmulV
                   : AdvSimdThreeSameOpcode::kFmulxV;
            // endregion
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
            // op_high=0, opcode=11101: FACGE (U=1); U=0 is reserved.
            if (!u) { ok = false; break; }
            op = AdvSimdThreeSameOpcode::kFacgeV;
            break;
          case 0b11110:
            if (u) { ok = false; break; }   // FMAXP — not implemented
            op = AdvSimdThreeSameOpcode::kFmaxV;
            break;
          case 0b11111:
            // region digitalis: op_high=0, opcode=11111: FRECPS (U=0) / FDIV (U=1).
            op = u ? AdvSimdThreeSameOpcode::kFdivV
                   : AdvSimdThreeSameOpcode::kFrecpsV;
            // endregion
            break;
          // endregion
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
            // op_high=1, opcode=11101: FACGT (U=1); U=0 is reserved.
            if (!u) { ok = false; break; }
            op = AdvSimdThreeSameOpcode::kFacgtV;
            break;
          case 0b11110:
            if (u) { ok = false; break; }   // FMINP — not implemented
            op = AdvSimdThreeSameOpcode::kFminV;
            break;
          case 0b11111:
            // region digitalis: op_high=1, opcode=11111: FRSQRTS (U=0);
            // U=1 is reserved.
            if (u) { ok = false; break; }
            op = AdvSimdThreeSameOpcode::kFrsqrtsV;
            // endregion
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

    // region digitalis - PMULL (opcode=1110) accepts size=00 (8-bit) and size=11 (64-bit, PMULL64).
    // Decoder rule: size=11 is reserved for all OTHER three-different opcodes; only PMULL allows it.
    if (size == 0b11 && !(u == 0 && opcode == 0b1110)) {
      Undefined();
      return;
    }
    // endregion

    AdvSimdThreeDiffOpcode op;

    switch (opcode) {
      case 0b0000:
        op = u ? AdvSimdThreeDiffOpcode::kUaddl : AdvSimdThreeDiffOpcode::kSaddl;
        break;
      // region digitalis - polynomial multiply (used by libz CRC32-acc).
      case 0b1110:
        if (u != 0) { Undefined(); return; }  // U=1 with opcode=1110 is unallocated
        // size=01 and size=10 are unallocated for PMULL.
        if (size == 0b01 || size == 0b10) { Undefined(); return; }
        op = AdvSimdThreeDiffOpcode::kPmull;
        break;
      // endregion
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
      // region digitalis - narrowing high: ADDHN/ADDHN2 (U=0), RADDHN/RADDHN2 (U=1).
      // size=11 already rejected above.
      case 0b0100:
        op = u ? AdvSimdThreeDiffOpcode::kRaddhn : AdvSimdThreeDiffOpcode::kAddhn;
        break;
      // endregion
      case 0b0101:
        op = u ? AdvSimdThreeDiffOpcode::kUabal : AdvSimdThreeDiffOpcode::kSabal;
        break;
      // region digitalis - narrowing high subtract: SUBHN/SUBHN2 (U=0),
      // RSUBHN/RSUBHN2 (U=1). size=11 already rejected above.
      case 0b0110:
        op = u ? AdvSimdThreeDiffOpcode::kRsubhn : AdvSimdThreeDiffOpcode::kSubhn;
        break;
      // endregion
      // region digitalis - signed saturating doubling multiply long:
      // SQDMULL / SQDMULL2 (U=0, opcode=1101). U=1 with opcode=1101 is
      // unallocated. size=00 is reserved (only halfword and word inputs
      // are defined); size=11 is already rejected by the top-of-routine
      // size guard.
      case 0b1101:
        if (u != 0) { Undefined(); return; }
        if (size == 0b00) { Undefined(); return; }
        op = AdvSimdThreeDiffOpcode::kSqdmull;
        break;
      // endregion
      // region digitalis - signed saturating doubling multiply-accumulate
      // long: SQDMLAL / SQDMLAL2 (U=0, opcode=1001). U=1 with opcode=1001
      // is unallocated. size=00 is reserved (only halfword and word inputs
      // are defined); size=11 is already rejected by the top-of-routine
      // size guard.
      case 0b1001:
        if (u != 0) { Undefined(); return; }
        if (size == 0b00) { Undefined(); return; }
        op = AdvSimdThreeDiffOpcode::kSqdmlal;
        break;
      // endregion
      // region digitalis - signed saturating doubling multiply-subtract
      // long: SQDMLSL / SQDMLSL2 (U=0, opcode=1011). U=1 with opcode=1011
      // is unallocated. size=00 is reserved (only halfword and word inputs
      // are defined); size=11 is already rejected by the top-of-routine
      // size guard.
      case 0b1011:
        if (u != 0) { Undefined(); return; }
        if (size == 0b00) { Undefined(); return; }
        op = AdvSimdThreeDiffOpcode::kSqdmlsl;
        break;
      // endregion
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
      // region digitalis - SQABS (U=0) / SQNEG (U=1) at opcode 00111.
      // All four sizes are valid; size=11 (.2d) is Q=1-only — the .1d form
      // (size=11, Q=0) is unallocated.
      case 0b00111:
        if (size == 0b11 && !q) { Undefined(); return; }
        op = u ? AdvSimdTwoRegMiscOpcode::kSqneg : AdvSimdTwoRegMiscOpcode::kSqabs;
        break;
      // endregion
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
        // bit20=1: across-lanes SMINV (U=0) / UMINV (U=1).
        // bit20=0: vector FCVT* rounding-mode, with bit23 picking N (0) vs P (1).
        if (GetBits<20, 1>()) {
          op = u ? AdvSimdTwoRegMiscOpcode::kUminv : AdvSimdTwoRegMiscOpcode::kSminv;
          if (size == 0b11) { Undefined(); return; }
        } else {
          // region digitalis - vector FCVTNS/NU (bit23=0) and FCVTPS/PU (bit23=1).
          if (!GetBits<23, 1>()) {
            op = u ? AdvSimdTwoRegMiscOpcode::kFcvtnuV
                   : AdvSimdTwoRegMiscOpcode::kFcvtnsV;
          } else {
            op = u ? AdvSimdTwoRegMiscOpcode::kFcvtpuV
                   : AdvSimdTwoRegMiscOpcode::kFcvtpsV;
          }
          // endregion
        }
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
        // region digitalis BFCVTN/BFCVTN2 share opcode=10110
        // with FCVTN; distinguished by size=10 (vs FCVTN's size=00/01).
        // bit30 (q) selects BFCVTN (low half write) vs BFCVTN2 (high half).
        if (size == 0b10) {
          op = AdvSimdTwoRegMiscOpcode::kBfcvtn;
        } else {
          op = AdvSimdTwoRegMiscOpcode::kFcvtn;
        }
        // endregion
        break;
      case 0b10111:
        if (u) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kFcvtl;
        break;
      case 0b01111:
        op = u ? AdvSimdTwoRegMiscOpcode::kFneg : AdvSimdTwoRegMiscOpcode::kFabs;
        break;
      // region digitalis - opcode=11101 splits on bit23:
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
      // endregion
      // region digitalis - opcode=11111 with bit23=1, U=1 is FSQRT (vector).
      // No defined encoding for U=0 / opcode=11111 in two-reg-misc.
      case 0b11111:
        if (!u) { Undefined(); return; }
        if (!GetBits<23, 1>()) { Undefined(); return; }
        op = AdvSimdTwoRegMiscOpcode::kFsqrtV;
        break;
      // endregion
      // region digitalis - opcode=11011 splits on whether this is the
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
            // region digitalis - vector FCVTMS/MU (round toward -inf). .
            op = u ? AdvSimdTwoRegMiscOpcode::kFcvtmuV
                   : AdvSimdTwoRegMiscOpcode::kFcvtmsV;
            // endregion
          }
        }
        break;
      // endregion
      // region digitalis - vector FCVTAS/AU (round-to-nearest ties-away).
      // Encoding: opcode=11100, bit23=0. bit23=1 is unallocated.
      case 0b11100:
        if (GetBits<23, 1>()) { Undefined(); return; }
        op = u ? AdvSimdTwoRegMiscOpcode::kFcvtauV
               : AdvSimdTwoRegMiscOpcode::kFcvtasV;
        break;
      // endregion
      // region digitalis: std FP32/FP64 FRINT* round-to-int.
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
          // region digitalis: FRECPS (U=0, bit23=0) / FRSQRTS (U=0, bit23=1).
          // Per ARM ARM C7.2.151 "FRECPS" and C7.2.155 "FRSQRTS":
          //   01 0 11110 0 sz 1 Rm 11111 1 Rn Rd   (FRECPS)
          //   01 0 11110 1 sz 1 Rm 11111 1 Rn Rd   (FRSQRTS)
          // Verified: frecps s0,s1,s2=0x5E22FC20, frecps d0,d1,d2=0x5E62FC20,
          //           frsqrts s0,s1,s2=0x5EA2FC20, frsqrts d0,d1,d2=0x5EE2FC20.
          // U=1 at opcode=11111 is unallocated in the scalar encoding.
          if (u) { Undefined(); return; }
          op = bit23 ? AdvSimdScalarThreeSameOpcode::kFrsqrts
                     : AdvSimdScalarThreeSameOpcode::kFrecps;
          // endregion
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

    // region digitalis BF16 indexed
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
    // Encoding cross-checks (llvm-mc, handoff-51):
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
    // endregion

    // region digitalis hello-dotprod
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
    // endregion

    // region digitalis indexed FCMLA
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
    // alongside non-indexed FP16 (handoff-49 rejects FP16-SIMD
    //     FCMA until the family is implemented end-to-end).
    //
    // llvm-mc-verified FP32 encodings (handoff-58 derivation):
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
        // FP16 (handoff-61, FP16 SIMD FCMA indexed):
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
    // endregion

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
    // region digitalis FP16 vector indexed FMLA/FMLS/FMUL (handoff-62)
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
    // routed to Undefined below — out of scope for handoff-62.
    } else if (size == 0b00) {
      if (u || (opcode != 0b0001 && opcode != 0b0101 && opcode != 0b1001)) {
        Undefined();
        return;
      }
      rm = Rm4;
      index = static_cast<uint8_t>((H << 2) | (L << 1) | M);
    // endregion
    // region digitalis - Plan §H-followup: integer MLA/MLS/MUL-idx halfword (handoff-64)
    //
    // Integer MUL/MLA/MLS by-element with halfword elements (.4h/.8h) —
    // Armv8-A baseline (not an extension; just a previously deferred
    // decoder gap, see handoff-63 standing item "Integer MLA/MLS/MUL-idx
    // with size=0b01").  Encoding pattern (per ARM ARM C7.2):
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
    // llvm-mc-verified encodings (handoff-64):
    //   mul  v0.4h, v1.4h, v2.h[0] = 0x0f428020
    //   mul  v0.8h, v1.8h, v2.h[7] = 0x4f728820
    //   mla  v0.4h, v1.4h, v2.h[0] = 0x2f420020
    //   mla  v0.8h, v1.8h, v2.h[7] = 0x6f720820
    //   mls  v0.4h, v1.4h, v2.h[0] = 0x2f424020
    //   mls  v0.8h, v1.8h, v2.h[7] = 0x6f724820
    } else if (size == 0b01) {
      if (!((opcode == 0b1000 && !u) ||
            (opcode == 0b0000 && u) ||
            (opcode == 0b0100 && u))) {
        Undefined();
        return;
      }
      rm = Rm4;
      index = static_cast<uint8_t>((H << 2) | (L << 1) | M);
    // endregion
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
        // region digitalis: FMUL/FMULX by element.  Per ARM ARM C7.2
        // AdvSIMD-vector-x-indexed-element:
        //   U=0, opcode=1001 -> FMUL  (by element) -> kFmul.
        //   U=1, opcode=1001 -> FMULX (by element, Armv8.2-FP) -> kFmulx.
        // FMULX differs from FMUL only in the ±0 * ±inf saturation case (it
        // produces ±2.0 rather than NaN), used by libm reciprocal-estimate
        // refinement loops.  Interpreter implements both via FmulxScalar.
        op = u ? AdvSimdVecXIdxOpcode::kFmulx
               : AdvSimdVecXIdxOpcode::kFmul;
        break;
        // endregion
      case 0b1000:
        // region digitalis follow-up (handoff-67): reject reserved
        // opcode=1000 with U=1.  Per ARM ARM C7.2 AdvSIMD-vector-x-indexed-element:
        //   U=0, opcode=1000 -> MUL (by element) — kept as kMul.
        //   U=1, opcode=1000 -> RESERVED — there is no instruction at this slot.
        //     MLA-by-element is encoded at opcode=0000 with U=1 (handled by the
        //     `case 0b0000` arm below); SQRDMLAH-by-element (Armv8.1-RDM) is at
        //     opcode=1101 (not implemented anywhere here yet).
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
        // endregion
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
    } else {
      // Remaining opcodes (SQDMULL/SQDMULH/SQRDMULH/SQRDMLAH/SQRDMLSH) are
      // not implemented — raise SIGILL.
      Undefined();
      return;
    }

    uint8_t rm = static_cast<uint8_t>((M << 4) | Rm4);
    uint8_t index;
    if (size == 0b10) {
      // FP32: index = H:L (4 elements in Vm.4S).
      index = static_cast<uint8_t>((H << 1) | L);
    } else if (size == 0b11) {
      // FP64: index = H (2 elements in Vm.2D); L must be 0.
      if (L) { Undefined(); return; }
      index = H;
    } else {
      // FP16 (size=01) is Armv8.2-FP16 scalar-x-indexed — not handled
      // yet.  size=00 is reserved at this slot.
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
      // region digitalis: AdvSIMD saturating shift left (immediate) — verified
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
      // endregion
      // region digitalis: AdvSIMD narrow-shift dispatch — verified against
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
      // endregion
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
      // region digitalis CASP fix: o1=1 covers both CAS and CASP.
      // The o2 bit is the discriminator: o2=0 → CASP (pair); o2=1 → CAS (single).
      // Encodings confirmed via llvm-mc (clang-r563880c):
      //   casp   w0,w1,w2,w3,[x10] = 0x08207d42 → o2=0, o1=1
      //   casa   w0,w1,[x10]       = 0x88e07d41 → o2=1, o1=1
      // Prior code routed both to kCas, silently miscompiling CASP.
      if (o2 == 0) {
        // CASP: bit[31] must be 0 and bits[14:10] must be 11111.
        // bit[30] = sz: 0 → 32-bit pair, 1 → 64-bit pair.
        // Re-encode args.size to 2 (32-bit) or 3 (64-bit) so the
        // interpreter/JIT can reuse their existing size dispatch.
        uint8_t rt2 = GetBits<10, 5>();
        if (GetBits<31, 1>() != 0 || rt2 != 0b11111) {
          Undefined();
          return;
        }
        args.op = AtomicOp::kCasp;
        args.acquire = (L != 0);   // CASPA/CASPAL
        args.release = (o0 != 0);  // CASPL/CASPAL
        args.size = (GetBits<30, 1>() != 0) ? 3 : 2;
      } else {
        // CAS family
        args.op = AtomicOp::kCas;
        args.acquire = (L != 0);   // CASA/CASAL
        args.release = (o0 != 0);  // CASL/CASAL
      }
      // endregion
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
      // region digitalis fix: LDEOR/LDSET routing was swapped.
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
      // endregion
      case 0b1000: args.op = AtomicOp::kSwp; break;
      // region digitalis LDAPR (Armv8.3-LRCPC) as plain LDAR.
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
      // Handoff-31 picked case 0b1111 here, which never fired -- the bug
      // surfaced when hello-lrcpc (handoff-42) issued the explicit
      // instruction via inline asm and SIGILL'd on the first probe.
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
      // endregion
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

    // region digitalis
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
    // endregion

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
