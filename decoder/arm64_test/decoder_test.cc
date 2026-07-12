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

// Always-on unit test for the ARM64 instruction decoder: direct
// (encoding -> expected callback + key fields) checks.
//
// Silent decoder mis-dispatch is the project's #1 historical bug class: an
// instruction that shares an encoding prefix with a neighbor gets routed to the
// wrong handler and produces a WRONG RESULT with no crash until a memory
// boundary is hit. Past examples that shipped and had to be chased down on
// device: LDPSW ignoring opc bit30 (zero-extending a negative offset), a
// hand-typed swap sending MRS DCZID_EL0 to the CTR_EL0 handler, CMGT decoded as
// an SMAX-adjacent opcode, and the AES crypto ops being swallowed by the
// AdvSIMD two-register-miscellaneous handler they must be checked before.
//
// Unlike decoder_objdump_diff_test.cc (a diagnostic keyed off an env var that
// harvests real-library encodings), this test is self-contained and always
// runs: every row is a fixed 32-bit encoding with a hand-authored expectation.
//
// Encoding provenance: every encoding below was produced OFFLINE with the
// pinned toolchain and pasted in with its assembly source as a comment, so the
// test itself invokes no external tool. The recipe (run by the author, not by
// the test) was, for a file `corpus.s` beginning with
// `.arch armv8.5-a+crypto+sha3+sha2+fp16+rng+memtag` and one instruction per
// line:
//
//   prebuilts/clang/host/linux-x86/clang-r547379/bin/clang \
//       --target=aarch64-linux-gnu -c corpus.s -o corpus.o
//   prebuilts/clang/host/linux-x86/clang-r547379/bin/llvm-objdump -d corpus.o
//
// (Reserved/undefined rows use `.inst 0x...` since they do not assemble from a
// mnemonic.) The <unknown>/reserved rows are additionally guaranteed by the
// decoder contract: DecodeInstruction() routes op0 (bits[28:25]) == 0b0000
// unconditionally to Undefined().

#include "gtest/gtest.h"

#include <cstdint>

#include "berberis/decoder/arm64/decoder.h"

namespace berberis {

namespace {

// A complete, self-contained recording InsnConsumer. Decoder<Recorder>
// instantiates references to EVERY consumer method it can dispatch to, so every
// callback must be defined even if a given corpus never reaches it. Each
// callback records its name; the callbacks whose families this test field-
// checks also copy the salient (mis-dispatch-sensitive) fields into scalar
// members. A fresh Recorder is used per decode, so the default member
// initializers reset all state.
class Recorder {
 public:
  using Decoder = berberis::Decoder<Recorder>;

  const char* name = nullptr;  // decoded handler name
  // Operand registers (-1 == not recorded by the firing callback).
  int rd = -1, rn = -1, rm = -1, rt = -1, rt2 = -1, rs = -1;
  int size = -1;       // LoadStore size in BITS (8/16/32/64) OR raw element-size
                       // field (0..3) — documented per family at the use site.
  int opcode = -1;     // handler-specific opcode enum, cast to int.
  int sysreg = -1;     // SystemReg value (MRS/MSR).
  int atomic_op = -1;  // AtomicOp value (exclusive / LSE atomics).
  bool is_store = false, is_signed = false, is_64bit_target = false;
  bool q = false, u = false, is_fp16 = false, is_adrp = false, scalar = false;
  bool is_preindex = false, is_postindex = false;

  static int SzBits(Decoder::LoadStoreSize s) {
    switch (s) {
      case Decoder::LoadStoreSize::k8bit:
        return 8;
      case Decoder::LoadStoreSize::k16bit:
        return 16;
      case Decoder::LoadStoreSize::k32bit:
        return 32;
      case Decoder::LoadStoreSize::k64bit:
        return 64;
    }
    return -1;
  }

  // --- load/store (field-checked) ---
  void LoadStoreImm(const Decoder::LoadStoreImmArgs& a) {
    name = "LoadStoreImm";
    rt = a.rt;
    rn = a.rn;
    size = SzBits(a.size);
    is_store = a.is_store;
    is_signed = a.is_signed;
    is_64bit_target = a.is_64bit_target;
  }
  void LoadStoreImmPreIndex(const Decoder::LoadStoreImmArgs& a) {
    name = "LoadStoreImmPreIndex";
    rt = a.rt;
    rn = a.rn;
    size = SzBits(a.size);
    is_store = a.is_store;
    is_signed = a.is_signed;
  }
  void LoadStoreImmPostIndex(const Decoder::LoadStoreImmArgs& a) {
    name = "LoadStoreImmPostIndex";
    rt = a.rt;
    rn = a.rn;
    size = SzBits(a.size);
    is_store = a.is_store;
    is_signed = a.is_signed;
  }
  void LoadStorePair(const Decoder::LoadStorePairArgs& a) {
    name = "LoadStorePair";
    rt = a.rt1;
    rt2 = a.rt2;
    rn = a.rn;
    size = SzBits(a.size);
    is_store = a.is_store;
    is_signed = a.is_signed;
    is_preindex = a.is_preindex;
    is_postindex = a.is_postindex;
  }
  void LoadStoreReg(const Decoder::LoadStoreRegArgs& a) {
    name = "LoadStoreReg";
    rt = a.rt;
    rn = a.rn;
    rm = a.rm;
    size = SzBits(a.size);
    is_store = a.is_store;
    is_signed = a.is_signed;
    is_64bit_target = a.is_64bit_target;
  }
  void LoadLiteral(const Decoder::LoadLiteralArgs& a) {
    name = "LoadLiteral";
    rt = a.rt;
    size = SzBits(a.size);
    is_signed = a.is_signed;
  }
  void LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& a) {
    name = "LoadStoreExclusive";
    atomic_op = static_cast<int>(a.op);
    rt = a.rt;
    rt2 = a.rt2;
    rn = a.rn;
    rs = a.rs;
    size = a.size;  // raw 0..3 (0=8b,1=16b,2=32b,3=64b)
  }

  // --- SIMD load/store (name only) ---
  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs&) { name = "SimdLoadStoreImm"; }
  void SimdLoadStoreImmPreIndex(const Decoder::SimdLoadStoreImmArgs&) {
    name = "SimdLoadStoreImmPreIndex";
  }
  void SimdLoadStoreImmPostIndex(const Decoder::SimdLoadStoreImmArgs&) {
    name = "SimdLoadStoreImmPostIndex";
  }
  void SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs&) { name = "SimdLoadStorePair"; }
  void SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs&) { name = "SimdLoadStoreReg"; }
  void SimdLoadLiteral(const Decoder::SimdLoadLiteralArgs&) { name = "SimdLoadLiteral"; }
  void AdvSimdMultiStruct(uint8_t, uint8_t, uint8_t, uint8_t, bool, bool, bool, uint8_t, bool) {
    name = "AdvSimdMultiStruct";
  }
  void AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs&) { name = "AdvSimdSingleStruct"; }

  // --- SIMD data-processing (field-checked families record opcode/size/q/u) ---
  void AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& a) {
    name = "AdvSimdShiftByImm";
    opcode = static_cast<int>(a.opcode);
    rd = a.rd;
    rn = a.rn;
    q = a.q;
    u = a.u;
    scalar = a.scalar;
  }
  void AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& a) {
    name = "AdvSimdThreeDiff";
    opcode = static_cast<int>(a.opcode);
    rd = a.rd;
    rn = a.rn;
    rm = a.rm;
    q = a.q;
  }
  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& a) {
    name = "AdvSimdTwoRegMisc";
    opcode = static_cast<int>(a.opcode);
    rd = a.rd;
    rn = a.rn;
    size = a.size;  // raw element-size field (0..3)
    q = a.q;
    u = a.u;
    is_fp16 = a.is_fp16;
  }
  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& a) {
    name = "AdvSimdThreeSame";
    opcode = static_cast<int>(a.opcode);
    rd = a.rd;
    rn = a.rn;
    rm = a.rm;
    size = a.size;  // raw element-size field (0..3)
    q = a.q;
    is_fp16 = a.is_fp16;
  }
  void AdvSimdCopy(const Decoder::AdvSimdCopyArgs& a) {
    name = "AdvSimdCopy";
    opcode = static_cast<int>(a.opcode);
  }
  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs&) {
    name = "AdvSimdScalarThreeSame";
  }
  void AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs&) {
    name = "AdvSimdScalarTwoRegMisc";
  }
  void AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs&) {
    name = "AdvSimdScalarPairwise";
  }
  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs&) {
    name = "AdvSimdVecXIndexedElement";
  }
  void AdvSimdScalarXIndexedElement(const Decoder::AdvSimdScalarXIdxArgs&) {
    name = "AdvSimdScalarXIndexedElement";
  }
  void AdvSimdTableLookup(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, bool) {
    name = "AdvSimdTableLookup";
  }
  void AdvSimdPermute(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, bool) { name = "AdvSimdPermute"; }
  void AdvSimdExtract(uint8_t, uint8_t, uint8_t, uint8_t, bool) { name = "AdvSimdExtract"; }
  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs&) { name = "SimdModifiedImm"; }
  void AdvSimdDotProduct(const Decoder::DotProductArgs&) { name = "AdvSimdDotProduct"; }
  void AdvSimdMatMul(const Decoder::MatMulArgs&) { name = "AdvSimdMatMul"; }
  void AdvSimdFcma(const Decoder::FcmaArgs&) { name = "AdvSimdFcma"; }
  void AdvSimdFcmaIdx(const Decoder::FcmaIdxArgs&) { name = "AdvSimdFcmaIdx"; }
  void AdvSimdBf16ThreeSame(const Decoder::Bf16ThreeSameArgs&) { name = "AdvSimdBf16ThreeSame"; }

  // --- integer / branch / system / fp (name, plus a few field-checked) ---
  void AddSubImm(const Decoder::AddSubImmArgs&) { name = "AddSubImm"; }
  void AddSubImmTags(const Decoder::AddSubImmTagsArgs&) { name = "AddSubImmTags"; }
  void LogicalImm(const Decoder::LogicalImmArgs&) { name = "LogicalImm"; }
  void MoveWide(const Decoder::MoveWideArgs&) { name = "MoveWide"; }
  void PcRelAddr(const Decoder::PcRelAddrArgs& a) {
    name = "PcRelAddr";
    rd = a.dst;
    is_adrp = a.is_adrp;
  }
  void Bitfield(const Decoder::BitfieldArgs&) { name = "Bitfield"; }
  void Extr(uint8_t, uint8_t, uint8_t, uint8_t, bool) { name = "Extr"; }
  void BranchImm(const Decoder::BranchImmArgs&) { name = "BranchImm"; }
  void BranchCond(const Decoder::BranchCondArgs&) { name = "BranchCond"; }
  void BranchReg(const Decoder::BranchRegArgs&) { name = "BranchReg"; }
  void CompareAndBranch(const Decoder::CompareAndBranchArgs&) { name = "CompareAndBranch"; }
  void TestAndBranch(const Decoder::TestAndBranchArgs&) { name = "TestAndBranch"; }
  void Svc(const Decoder::SvcArgs&) { name = "Svc"; }
  void Brk(uint16_t) { name = "Brk"; }
  void Mrs(const Decoder::MrsArgs& a) {
    name = "Mrs";
    rd = a.dst;
    sysreg = static_cast<int>(a.sysreg);
  }
  void Msr(const Decoder::MsrArgs& a) {
    name = "Msr";
    rn = a.src;
    sysreg = static_cast<int>(a.sysreg);
  }
  void LogicalShiftedReg(const Decoder::LogicalShiftedRegArgs&) { name = "LogicalShiftedReg"; }
  void AddSubShiftedReg(const Decoder::AddSubShiftedRegArgs&) { name = "AddSubShiftedReg"; }
  void AddSubExtendedReg(const Decoder::AddSubExtendedRegArgs&) { name = "AddSubExtendedReg"; }
  void AddSubWithCarry(uint8_t, uint8_t, uint8_t, bool, bool, bool) { name = "AddSubWithCarry"; }
  void ConditionalSelect(const Decoder::ConditionalSelectArgs&) { name = "ConditionalSelect"; }
  void ConditionalCompare(const Decoder::ConditionalCompareArgs&) { name = "ConditionalCompare"; }
  void DataProc1Src(uint8_t, uint8_t, uint8_t, bool) { name = "DataProc1Src"; }
  void DataProc2Src(const Decoder::DataProc2SrcArgs&) { name = "DataProc2Src"; }
  void DataProc3Src(const Decoder::DataProc3SrcArgs&) { name = "DataProc3Src"; }
  void Nop() { name = "Nop"; }
  void DataMemoryBarrier() { name = "DataMemoryBarrier"; }
  void IcIvau(uint8_t) { name = "IcIvau"; }
  void MteDataProc(const Decoder::MteDataProcArgs&) { name = "MteDataProc"; }
  void MteLoadStore(const Decoder::MteLoadStoreArgs&) { name = "MteLoadStore"; }
  void FpIntConversion(const Decoder::FpIntConvArgs&) { name = "FpIntConversion"; }
  void FpMovImmediate(uint8_t, uint8_t, uint8_t) { name = "FpMovImmediate"; }
  void FpDataProc1(const Decoder::FpDataProc1Args&) { name = "FpDataProc1"; }
  void FpDataProc2(const Decoder::FpDataProc2Args&) { name = "FpDataProc2"; }
  void FpDataProc3(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, bool, bool) { name = "FpDataProc3"; }
  void FpCompare(const Decoder::FpCompareArgs&) { name = "FpCompare"; }
  void FpConditionalCompare(const Decoder::FpConditionalCompareArgs&) { name = "FpConditionalCompare"; }
  void FpCondSelect(uint8_t, uint8_t, uint8_t, uint8_t, Decoder::Condition) { name = "FpCondSelect"; }
  void FpFixedPointConversion(const Decoder::FpFixedPointArgs&) { name = "FpFixedPointConversion"; }
  void Sha512(Decoder::Sha512Op, uint8_t, uint8_t, uint8_t) { name = "Sha512"; }
  void Eor3(uint8_t, uint8_t, uint8_t, uint8_t) { name = "Eor3"; }
  void Bcax(uint8_t, uint8_t, uint8_t, uint8_t) { name = "Bcax"; }
  void Rax1(uint8_t, uint8_t, uint8_t) { name = "Rax1"; }
  void Xar(uint8_t, uint8_t, uint8_t, uint8_t) { name = "Xar"; }
  void Sm4e(uint8_t, uint8_t) { name = "Sm4e"; }
  void Sm4ekey(uint8_t, uint8_t, uint8_t) { name = "Sm4ekey"; }
  void Sm3ss1(uint8_t, uint8_t, uint8_t, uint8_t) { name = "Sm3ss1"; }
  void Sm3tt(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t) { name = "Sm3tt"; }
  void Sm3partw1(uint8_t, uint8_t, uint8_t) { name = "Sm3partw1"; }
  void Sm3partw2(uint8_t, uint8_t, uint8_t) { name = "Sm3partw2"; }
  void CryptoAes(uint8_t rd_arg, uint8_t rn_arg, uint8_t opcode_arg) {
    name = "CryptoAes";
    rd = rd_arg;
    rn = rn_arg;
    opcode = opcode_arg;  // 00=AESE, 01=AESD, 10=AESMC, 11=AESIMC
  }
  void CryptoSha3Reg(uint8_t, uint8_t, uint8_t, uint8_t) { name = "CryptoSha3Reg"; }
  void CryptoSha2Reg(uint8_t, uint8_t, uint8_t) { name = "CryptoSha2Reg"; }
  void Undefined() { name = "Undefined"; }
};

using Dec = Decoder<Recorder>;

// Decode one 32-bit encoding through a fresh Recorder.
Recorder Decode(uint32_t enc) {
  Recorder rec;
  Dec dec(&rec);
  dec.Decode(reinterpret_cast<const uint16_t*>(&enc));
  return rec;
}

// ============================================================================
// Load/store pair — LDPSW vs LDP (opc/bit30). The historical bug: the decoder
// ignored opc bit30, so LDPSW zero-extended instead of sign-extending, turning
// a negative 32-bit path coordinate into +4.29e9 and blowing a fill loop.
// ============================================================================
TEST(Arm64DecoderTest, LoadStorePair_LdpswVsLdp) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    const char* handler;
    int size_bits;  // 32 or 64
    bool is_signed;
    bool is_store;
    bool pre;
    bool post;
    int rt1, rt2, rn;
  };
  const Row rows[] = {
      // ldpsw x0, x1, [x2, #8]      (signed, load, offset)
      {0x69410440, "ldpsw x0,x1,[x2,#8]", "LoadStorePair", 32, true, false, false, false, 0, 1, 2},
      // ldp   w3, w4, [x5, #8]      (unsigned 32-bit load)
      {0x294110a3, "ldp w3,w4,[x5,#8]", "LoadStorePair", 32, false, false, false, false, 3, 4, 5},
      // ldp   x6, x7, [x8, #16]     (unsigned 64-bit load)
      {0xa9411d06, "ldp x6,x7,[x8,#16]", "LoadStorePair", 64, false, false, false, false, 6, 7, 8},
      // stp   w9, w10, [x11, #8]    (32-bit store)
      {0x29012969, "stp w9,w10,[x11,#8]", "LoadStorePair", 32, false, true, false, false, 9, 10, 11},
      // stp   x12, x13, [x14, #16]  (64-bit store)
      {0xa90135cc, "stp x12,x13,[x14,#16]", "LoadStorePair", 64, false, true, false, false, 12, 13, 14},
      // ldpsw x15, x16, [x17], #8   (post-index, signed)
      {0x68c1422f, "ldpsw x15,x16,[x17],#8", "LoadStorePair", 32, true, false, false, true, 15, 16, 17},
      // ldp   w18, w19, [x20], #8   (post-index, unsigned)
      {0x28c14e92, "ldp w18,w19,[x20],#8", "LoadStorePair", 32, false, false, false, true, 18, 19, 20},
      // ldpsw x21, x22, [x23, #8]!  (pre-index, signed)
      {0x69c15af5, "ldpsw x21,x22,[x23,#8]!", "LoadStorePair", 32, true, false, true, false, 21, 22, 23},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, r.handler);
    EXPECT_EQ(rec.size, r.size_bits);
    EXPECT_EQ(rec.is_signed, r.is_signed);
    EXPECT_EQ(rec.is_store, r.is_store);
    EXPECT_EQ(rec.is_preindex, r.pre);
    EXPECT_EQ(rec.is_postindex, r.post);
    EXPECT_EQ(rec.rt, r.rt1);
    EXPECT_EQ(rec.rt2, r.rt2);
    EXPECT_EQ(rec.rn, r.rn);
  }
}

// ============================================================================
// Load/store unsigned-immediate — every (size, opc) combination, including the
// signed loads (ldrs*), the 32-vs-64-bit signed-load target width, and PRFM
// (which must decode to Nop, not a load).
// ============================================================================
TEST(Arm64DecoderTest, LoadStoreImm_SizeOpcCombos) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    const char* handler;
    int size_bits;
    bool is_signed;
    bool is_store;
    bool x_target;  // is_64bit_target (only meaningful for signed loads)
    int rt, rn;
  };
  const Row rows[] = {
      // ldrb  w0, [x1, #1]
      {0x39400420, "ldrb w0,[x1,#1]", "LoadStoreImm", 8, false, false, false, 0, 1},
      // ldrh  w2, [x3, #2]
      {0x79400462, "ldrh w2,[x3,#2]", "LoadStoreImm", 16, false, false, false, 2, 3},
      // ldr   w4, [x5, #4]
      {0xb94004a4, "ldr w4,[x5,#4]", "LoadStoreImm", 32, false, false, false, 4, 5},
      // ldr   x6, [x7, #8]
      {0xf94004e6, "ldr x6,[x7,#8]", "LoadStoreImm", 64, false, false, false, 6, 7},
      // ldrsb x8, [x9, #1]   (signed load to X)
      {0x39800528, "ldrsb x8,[x9,#1]", "LoadStoreImm", 8, true, false, true, 8, 9},
      // ldrsh x10, [x11, #2] (signed load to X)
      {0x7980056a, "ldrsh x10,[x11,#2]", "LoadStoreImm", 16, true, false, true, 10, 11},
      // ldrsw x12, [x13, #4] (signed load to X)
      {0xb98005ac, "ldrsw x12,[x13,#4]", "LoadStoreImm", 32, true, false, true, 12, 13},
      // ldrsb w14, [x15, #1] (signed load to W)
      {0x39c005ee, "ldrsb w14,[x15,#1]", "LoadStoreImm", 8, true, false, false, 14, 15},
      // ldrsh w16, [x17, #2] (signed load to W)
      {0x79c00630, "ldrsh w16,[x17,#2]", "LoadStoreImm", 16, true, false, false, 16, 17},
      // strb  w18, [x19, #1]
      {0x39000672, "strb w18,[x19,#1]", "LoadStoreImm", 8, false, true, false, 18, 19},
      // strh  w20, [x21, #2]
      {0x790006b4, "strh w20,[x21,#2]", "LoadStoreImm", 16, false, true, false, 20, 21},
      // str   w22, [x23, #4]
      {0xb90006f6, "str w22,[x23,#4]", "LoadStoreImm", 32, false, true, false, 22, 23},
      // str   x24, [x25, #8]
      {0xf9000738, "str x24,[x25,#8]", "LoadStoreImm", 64, false, true, false, 24, 25},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, r.handler);
    EXPECT_EQ(rec.size, r.size_bits);
    EXPECT_EQ(rec.is_signed, r.is_signed);
    EXPECT_EQ(rec.is_store, r.is_store);
    EXPECT_EQ(rec.rt, r.rt);
    EXPECT_EQ(rec.rn, r.rn);
    if (r.is_signed) {
      EXPECT_EQ(rec.is_64bit_target, r.x_target);
    }
  }

  // prfm pldl1keep, [x26, #8] — PRFM (immediate) is an unsigned-imm load-store
  // encoding that MUST decode to Nop (no-op prefetch), never a real load.
  Recorder prfm = Decode(0xf9800740);
  EXPECT_STREQ(prfm.name, "Nop") << "prfm pldl1keep,[x26,#8]";
}

// ============================================================================
// MRS — system-register reads. The (op0,op1,CRn,CRm,op2) tuple must map to the
// right SystemReg enumerator. A prior hand-typed swap sent MRS DCZID_EL0 to the
// CTR_EL0 handler, which corrupted large memsets. Asserting against the named
// enumerators (whose values are computed from the tuple) catches any drift.
// ============================================================================
TEST(Arm64DecoderTest, Mrs_SystemRegs) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    Dec::SystemReg sysreg;
    int rd;
  };
  const Row rows[] = {
      {0xd53bd040, "mrs x0,tpidr_el0", Dec::SystemReg::kTpidrEl0, 0},
      {0xd53b0021, "mrs x1,ctr_el0", Dec::SystemReg::kCtrEl0, 1},
      {0xd53b00e2, "mrs x2,dczid_el0", Dec::SystemReg::kDczidEl0, 2},
      {0xd53b4203, "mrs x3,nzcv", Dec::SystemReg::kNzcv, 3},
      {0xd53b4404, "mrs x4,fpcr", Dec::SystemReg::kFpcr, 4},
      {0xd53b4425, "mrs x5,fpsr", Dec::SystemReg::kFpsr, 5},
      {0xd5380006, "mrs x6,midr_el1", Dec::SystemReg::kMidrEl1, 6},
      {0xd53be047, "mrs x7,cntvct_el0", Dec::SystemReg::kCntvctEl0, 7},
      {0xd53be008, "mrs x8,cntfrq_el0", Dec::SystemReg::kCntfrqEl0, 8},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, "Mrs");
    EXPECT_EQ(rec.sysreg, static_cast<int>(r.sysreg));
    EXPECT_EQ(rec.rd, r.rd);
  }
}

TEST(Arm64DecoderTest, Msr_SystemRegs) {
  // msr nzcv, x9
  Recorder m1 = Decode(0xd51b4209);
  EXPECT_STREQ(m1.name, "Msr") << "msr nzcv,x9";
  EXPECT_EQ(m1.sysreg, static_cast<int>(Dec::SystemReg::kNzcv));
  EXPECT_EQ(m1.rn, 9);
  // msr fpcr, x10
  Recorder m2 = Decode(0xd51b440a);
  EXPECT_STREQ(m2.name, "Msr") << "msr fpcr,x10";
  EXPECT_EQ(m2.sysreg, static_cast<int>(Dec::SystemReg::kFpcr));
  EXPECT_EQ(m2.rn, 10);
}

// ============================================================================
// Crypto AES vs its AdvSIMD two-register-miscellaneous neighbors. AESE/AESD/
// AESMC/AESIMC share bits[24:5]=01110 and bits[11:10]=10 with two-reg-misc and
// MUST be checked first; a mis-order swallows them into the two-reg-misc
// handler that cannot handle them. REV64/REV32/CNT are genuine two-reg-misc
// ops on the same prefix that must still route there.
// ============================================================================
TEST(Arm64DecoderTest, AesVsTwoRegMisc) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    const char* handler;
    int aes_opcode;  // -1 when not CryptoAes
    int rd, rn;
  };
  const Row rows[] = {
      {0x4e284820, "aese v0.16b,v1.16b", "CryptoAes", 0, 0, 1},
      {0x4e285862, "aesd v2.16b,v3.16b", "CryptoAes", 1, 2, 3},
      {0x4e2868a4, "aesmc v4.16b,v5.16b", "CryptoAes", 2, 4, 5},
      {0x4e2878e6, "aesimc v6.16b,v7.16b", "CryptoAes", 3, 6, 7},
      {0x4e200928, "rev64 v8.16b,v9.16b", "AdvSimdTwoRegMisc", -1, 8, 9},
      {0x6e20096a, "rev32 v10.16b,v11.16b", "AdvSimdTwoRegMisc", -1, 10, 11},
      {0x4e2059ac, "cnt v12.16b,v13.16b", "AdvSimdTwoRegMisc", -1, 12, 13},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, r.handler);
    EXPECT_EQ(rec.rd, r.rd);
    EXPECT_EQ(rec.rn, r.rn);
    if (r.aes_opcode >= 0) {
      EXPECT_EQ(rec.opcode, r.aes_opcode);
    }
  }
}

// ============================================================================
// AdvSIMD three-same integer — the CMGT-vs-SMAX opcode-mapping class. Every
// mnemonic must map to its exact AdvSimdThreeSameOpcode enumerator (asserted by
// name, so a decoder mis-map and a reordered enum can't both silently agree).
// ============================================================================
TEST(Arm64DecoderTest, ThreeSameInteger_OpcodeMapping) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    Dec::AdvSimdThreeSameOpcode opcode;
    int rd, rn, rm, size, q;  // size = raw element-size field; .4s -> 2, .16b -> 0
  };
  const Row rows[] = {
      {0x4ea23420, "cmgt v0.4s,v1.4s,v2.4s", Dec::AdvSimdThreeSameOpcode::kCmgt, 0, 1, 2, 2, 1},
      {0x4ea53c83, "cmge v3.4s,v4.4s,v5.4s", Dec::AdvSimdThreeSameOpcode::kCmge, 3, 4, 5, 2, 1},
      {0x4ea864e6, "smax v6.4s,v7.4s,v8.4s", Dec::AdvSimdThreeSameOpcode::kSmax, 6, 7, 8, 2, 1},
      {0x4eab6d49, "smin v9.4s,v10.4s,v11.4s", Dec::AdvSimdThreeSameOpcode::kSmin, 9, 10, 11, 2, 1},
      {0x6eae65ac, "umax v12.4s,v13.4s,v14.4s", Dec::AdvSimdThreeSameOpcode::kUmax, 12, 13, 14, 2, 1},
      {0x4eb1860f, "add v15.4s,v16.4s,v17.4s", Dec::AdvSimdThreeSameOpcode::kAdd, 15, 16, 17, 2, 1},
      {0x6eb48672, "sub v18.4s,v19.4s,v20.4s", Dec::AdvSimdThreeSameOpcode::kSub, 18, 19, 20, 2, 1},
      {0x6eb78ed5, "cmeq v21.4s,v22.4s,v23.4s", Dec::AdvSimdThreeSameOpcode::kCmeq, 21, 22, 23, 2, 1},
      {0x4eba9f38, "mul v24.4s,v25.4s,v26.4s", Dec::AdvSimdThreeSameOpcode::kMul, 24, 25, 26, 2, 1},
      {0x4e3d1f9b, "and v27.16b,v28.16b,v29.16b", Dec::AdvSimdThreeSameOpcode::kAnd, 27, 28, 29, 0, 1},
      {0x4ea01ffe, "orr v30.16b,v31.16b,v0.16b", Dec::AdvSimdThreeSameOpcode::kOrr, 30, 31, 0, 2, 1},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, "AdvSimdThreeSame");
    EXPECT_EQ(rec.opcode, static_cast<int>(r.opcode));
    EXPECT_EQ(rec.rd, r.rd);
    EXPECT_EQ(rec.rn, r.rn);
    EXPECT_EQ(rec.rm, r.rm);
    EXPECT_EQ(rec.size, r.size);
    EXPECT_EQ(rec.q, static_cast<bool>(r.q));
    EXPECT_FALSE(rec.is_fp16);
  }
}

// ============================================================================
// LSE atomics — SWP vs LDADD and the full opcode fan-out. All decode to
// LoadStoreExclusive; the AtomicOp discriminator must match the mnemonic (SWP
// mis-mapped to LDADD, or vice versa, produces wrong memory contents silently).
// ============================================================================
TEST(Arm64DecoderTest, Atomics_SwpVsLdadd) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    Dec::AtomicOp op;
    int rs, rt, rn;
  };
  const Row rows[] = {
      {0xf8208041, "swp x0,x1,[x2]", Dec::AtomicOp::kSwp, 0, 1, 2},
      {0xf82300a4, "ldadd x3,x4,[x5]", Dec::AtomicOp::kLdadd, 3, 4, 5},
      {0xf8261107, "ldclr x6,x7,[x8]", Dec::AtomicOp::kLdclr, 6, 7, 8},
      {0xf829316a, "ldset x9,x10,[x11]", Dec::AtomicOp::kLdset, 9, 10, 11},
      {0xf82c21cd, "ldeor x12,x13,[x14]", Dec::AtomicOp::kLdeor, 12, 13, 14},
      {0xc8af7e30, "cas x15,x16,[x17]", Dec::AtomicOp::kCas, 15, 16, 17},
      {0xf8324293, "ldsmax x18,x19,[x20]", Dec::AtomicOp::kLdsmax, 18, 19, 20},
      {0xf83552f6, "ldsmin x21,x22,[x23]", Dec::AtomicOp::kLdsmin, 21, 22, 23},
      {0xf8386359, "ldumax x24,x25,[x26]", Dec::AtomicOp::kLdumax, 24, 25, 26},
      {0xf83b73bc, "ldumin x27,x28,[x29]", Dec::AtomicOp::kLdumin, 27, 28, 29},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, "LoadStoreExclusive");
    EXPECT_EQ(rec.atomic_op, static_cast<int>(r.op));
    EXPECT_EQ(rec.rs, r.rs);
    EXPECT_EQ(rec.rt, r.rt);
    EXPECT_EQ(rec.rn, r.rn);
    EXPECT_EQ(rec.size, 3);  // all rows are 64-bit (size field == 3)
  }
}

// ============================================================================
// Load/store exclusive (non-LSE) — LDXR/STXR/LDAR/STLR.
// ============================================================================
TEST(Arm64DecoderTest, Exclusive) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    Dec::AtomicOp op;
  };
  const Row rows[] = {
      {0xc85f7c20, "ldxr x0,[x1]", Dec::AtomicOp::kLdxr},
      {0xc8027c83, "stxr w2,x3,[x4]", Dec::AtomicOp::kStxr},
      {0xc8dffcc5, "ldar x5,[x6]", Dec::AtomicOp::kLdar},
      {0xc89ffd07, "stlr x7,[x8]", Dec::AtomicOp::kStlr},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, "LoadStoreExclusive");
    EXPECT_EQ(rec.atomic_op, static_cast<int>(r.op));
  }
}

// ============================================================================
// AdvSIMD three-different (widening/narrowing) — must route to AdvSimdThreeDiff
// (bits[11:10]==00 distinguishes it from three-same's ==01/etc.), and PMULL is
// on the same prefix.
// ============================================================================
TEST(Arm64DecoderTest, ThreeDiff) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    Dec::AdvSimdThreeDiffOpcode opcode;
    int rd, rn, rm, q;
  };
  const Row rows[] = {
      {0x0e620020, "saddl v0.4s,v1.4h,v2.4h", Dec::AdvSimdThreeDiffOpcode::kSaddl, 0, 1, 2, 0},
      {0x2e650083, "uaddl v3.4s,v4.4h,v5.4h", Dec::AdvSimdThreeDiffOpcode::kUaddl, 3, 4, 5, 0},
      {0x0e68c0e6, "smull v6.4s,v7.4h,v8.4h", Dec::AdvSimdThreeDiffOpcode::kSmull, 6, 7, 8, 0},
      {0x2e6bc149, "umull v9.4s,v10.4h,v11.4h", Dec::AdvSimdThreeDiffOpcode::kUmull, 9, 10, 11, 0},
      {0x0e6e41ac, "addhn v12.4h,v13.4s,v14.4s", Dec::AdvSimdThreeDiffOpcode::kAddhn, 12, 13, 14, 0},
      {0x0e31e20f, "pmull v15.8h,v16.8b,v17.8b", Dec::AdvSimdThreeDiffOpcode::kPmull, 15, 16, 17, 0},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, "AdvSimdThreeDiff");
    EXPECT_EQ(rec.opcode, static_cast<int>(r.opcode));
    EXPECT_EQ(rec.rd, r.rd);
    EXPECT_EQ(rec.rn, r.rn);
    EXPECT_EQ(rec.rm, r.rm);
    EXPECT_EQ(rec.q, static_cast<bool>(r.q));
  }
}

// ============================================================================
// AdvSIMD shift-by-immediate — SHL/SSHR/USHR/SHRN/SSHLL. Signedness (u) and the
// opcode enumerator distinguish them from each other and from three-same.
// ============================================================================
TEST(Arm64DecoderTest, ShiftByImm) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    Dec::AdvSimdShiftImmOpcode opcode;
    bool u;
    int rd, rn;
  };
  const Row rows[] = {
      {0x4f235420, "shl v0.4s,v1.4s,#3", Dec::AdvSimdShiftImmOpcode::kShl, false, 0, 1},
      {0x4f3d0462, "sshr v2.4s,v3.4s,#3", Dec::AdvSimdShiftImmOpcode::kSshr, false, 2, 3},
      {0x6f3d04a4, "ushr v4.4s,v5.4s,#3", Dec::AdvSimdShiftImmOpcode::kUshr, true, 4, 5},
      {0x0f1d84e6, "shrn v6.4h,v7.4s,#3", Dec::AdvSimdShiftImmOpcode::kShrn, false, 6, 7},
      {0x0f13a528, "sshll v8.4s,v9.4h,#3", Dec::AdvSimdShiftImmOpcode::kSshll, false, 8, 9},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, "AdvSimdShiftByImm");
    EXPECT_EQ(rec.opcode, static_cast<int>(r.opcode));
    EXPECT_EQ(rec.u, r.u);
    EXPECT_EQ(rec.rd, r.rd);
    EXPECT_EQ(rec.rn, r.rn);
    EXPECT_FALSE(rec.scalar);  // vector forms
  }
}

// ============================================================================
// Scalar vs vector three-same (bit28). `add d0,d1,d2` (scalar) and
// `add v0.2d,v1.2d,v2.2d` (vector) differ only in bit28 and must route to
// different handlers.
// ============================================================================
TEST(Arm64DecoderTest, ScalarVsVectorThreeSame) {
  // add d0, d1, d2 — scalar (bit28 == 1).
  Recorder sc = Decode(0x5ee28420);
  EXPECT_STREQ(sc.name, "AdvSimdScalarThreeSame") << "add d0,d1,d2";
  // add v0.2d, v1.2d, v2.2d — vector (bit28 == 0).
  Recorder ve = Decode(0x4ee28420);
  EXPECT_STREQ(ve.name, "AdvSimdThreeSame") << "add v0.2d,v1.2d,v2.2d";
  EXPECT_EQ(ve.opcode, static_cast<int>(Dec::AdvSimdThreeSameOpcode::kAdd));
  EXPECT_EQ(ve.size, 3);  // .2d -> size field 3
  EXPECT_TRUE(ve.q);
}

// ============================================================================
// FP16 vs FP32/FP64 three-same. The FP16 vector form has a separate encoding
// (bit21==0, bits[15:14]==00, bit22==1) that the decoder must flag as is_fp16
// rather than mis-routing bit22 to FP64 element semantics.
// ============================================================================
TEST(Arm64DecoderTest, Fp16VsFp32Fp64) {
  // fadd v0.4s, v1.4s, v2.4s — FP32 (sz collapsed to 0), not fp16.
  Recorder s = Decode(0x4e22d420);
  EXPECT_STREQ(s.name, "AdvSimdThreeSame") << "fadd v0.4s,...";
  EXPECT_EQ(s.opcode, static_cast<int>(Dec::AdvSimdThreeSameOpcode::kFaddV));
  EXPECT_FALSE(s.is_fp16);
  EXPECT_EQ(s.size, 0);
  // fadd v3.2d, v4.2d, v5.2d — FP64 (sz collapsed to 1), not fp16.
  Recorder d = Decode(0x4e65d483);
  EXPECT_STREQ(d.name, "AdvSimdThreeSame") << "fadd v3.2d,...";
  EXPECT_EQ(d.opcode, static_cast<int>(Dec::AdvSimdThreeSameOpcode::kFaddV));
  EXPECT_FALSE(d.is_fp16);
  EXPECT_EQ(d.size, 1);
  // fadd v6.4h, v7.4h, v8.4h — FP16 vector form.
  Recorder h = Decode(0x0e4814e6);
  EXPECT_STREQ(h.name, "AdvSimdThreeSame") << "fadd v6.4h,...";
  EXPECT_EQ(h.opcode, static_cast<int>(Dec::AdvSimdThreeSameOpcode::kFaddV));
  EXPECT_TRUE(h.is_fp16);
}

// ============================================================================
// ADR vs ADRP (bit31 / the PC-relative page flag).
// ============================================================================
TEST(Arm64DecoderTest, AdrVsAdrp) {
  // adr x0, #16
  Recorder adr = Decode(0x10000080);
  EXPECT_STREQ(adr.name, "PcRelAddr") << "adr x0,#16";
  EXPECT_FALSE(adr.is_adrp);
  EXPECT_EQ(adr.rd, 0);
  // adrp x1, #0
  Recorder adrp = Decode(0x90000001);
  EXPECT_STREQ(adrp.name, "PcRelAddr") << "adrp x1,#0";
  EXPECT_TRUE(adrp.is_adrp);
  EXPECT_EQ(adrp.rd, 1);
}

// ============================================================================
// AdvSIMD two-register-miscellaneous — XTN (narrowing), NEG, ABS. XTN in
// particular must not be confused with the three-same/three-diff neighbors.
// ============================================================================
TEST(Arm64DecoderTest, TwoRegMisc) {
  struct Row {
    uint32_t enc;
    const char* asmtext;
    Dec::AdvSimdTwoRegMiscOpcode opcode;
    int rd, rn;
  };
  const Row rows[] = {
      {0x0e612820, "xtn v0.4h,v1.4s", Dec::AdvSimdTwoRegMiscOpcode::kXtn, 0, 1},
      {0x6ea0b862, "neg v2.4s,v3.4s", Dec::AdvSimdTwoRegMiscOpcode::kNeg, 2, 3},
      {0x4ea0b8a4, "abs v4.4s,v5.4s", Dec::AdvSimdTwoRegMiscOpcode::kAbs, 4, 5},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.asmtext);
    EXPECT_STREQ(rec.name, "AdvSimdTwoRegMisc");
    EXPECT_EQ(rec.opcode, static_cast<int>(r.opcode));
    EXPECT_EQ(rec.rd, r.rd);
    EXPECT_EQ(rec.rn, r.rn);
  }
}

// ============================================================================
// Reserved / undefined encodings. DecodeInstruction() routes op0 (bits[28:25])
// == 0b0000 unconditionally to Undefined(); these are the reserved-space
// encodings (UDF and friends) that must never be mistaken for a real op.
// ============================================================================
TEST(Arm64DecoderTest, ReservedUndefined) {
  struct Row {
    uint32_t enc;
    const char* note;
  };
  const Row rows[] = {
      {0x00000000, "udf #0"},
      {0x0000dead, "udf #0xdead"},
      {0x01ffffff, "reserved (op0==0b0000)"},
  };
  for (const auto& r : rows) {
    Recorder rec = Decode(r.enc);
    SCOPED_TRACE(r.note);
    EXPECT_STREQ(rec.name, "Undefined");
  }
}

}  // namespace

}  // namespace berberis
