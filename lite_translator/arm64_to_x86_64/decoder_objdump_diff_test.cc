// Copyright (C) 2026 utzcoz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Decoder-vs-objdump differential. The glyph-render bug in Chromium under
// translation is shared (interpreter == JIT) and in the JIT-compiled rasterizer,
// which means the only way both tiers can be wrong identically is a DECODER
// mis-dispatch (both go through Decoder<SemanticsPlayer<...>>). This test reads a
// file of (encoding, objdump-mnemonic) pairs harvested from real shipped
// binaries and, for every encoding, records which decoder handler +
// sign/size/store fields Berberis picks, then flags any case where Berberis's
// decode is INCONSISTENT with objdump's mnemonic (a mis-dispatch).
//
// The corpus is committed next to this test and used by default, so the check
// runs as part of the normal host suite. DIGITALIS_RAST_ENC overrides the path
// for a larger local sweep (harvest_encodings.py --cap 0 emits the full set).

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "android-base/file.h"

#include "berberis/decoder/arm64/decoder.h"

namespace berberis {

namespace {

// Committed test data. Soong's `data:` keeps the source-relative path, so these
// install under <test binary dir>/decoder/arm64_test/.
constexpr const char* kCorpusFile = "decoder/arm64_test/arm64_decoder_encodings.txt";
constexpr const char* kUndefinedBaselineFile =
    "decoder/arm64_test/arm64_decoder_undefined_baseline.txt";

// Records the decoder's chosen handler + key discriminating fields. One instance
// is reused per Decode() call (cleared by re-decode). Every InsnConsumer method
// the arm64 decoder can call is implemented; most just record the handler name,
// the load/store + suspect-SIMD ones also record fields a mis-dispatch corrupts.
class Recorder {
 public:
  using Decoder = berberis::Decoder<Recorder>;
  std::string h;  // handler + fields

  static const char* Sz(Decoder::LoadStoreSize s) {
    switch (s) {
      case Decoder::LoadStoreSize::k8bit: return "8";
      case Decoder::LoadStoreSize::k16bit: return "16";
      case Decoder::LoadStoreSize::k32bit: return "32";
      case Decoder::LoadStoreSize::k64bit: return "64";
      default: return "?";
    }
  }
  static const char* SSz(Decoder::SimdLoadStoreSize s) {
    switch (s) {
      case Decoder::SimdLoadStoreSize::k8bit: return "8";
      case Decoder::SimdLoadStoreSize::k16bit: return "16";
      case Decoder::SimdLoadStoreSize::k32bit: return "32";
      case Decoder::SimdLoadStoreSize::k64bit: return "64";
      case Decoder::SimdLoadStoreSize::k128bit: return "128";
      default: return "?";
    }
  }

  // --- load/store (the LDPSW / LDTR class — record size/signed/store) ---
  void LoadStoreImm(const Decoder::LoadStoreImmArgs& a) {
    h = std::string("LSImm sz=") + Sz(a.size) + " sgn=" + (a.is_signed ? "1" : "0") +
        " st=" + (a.is_store ? "1" : "0") + " x=" + (a.is_64bit_target ? "1" : "0");
  }
  void LoadStoreImmPreIndex(const Decoder::LoadStoreImmArgs& a) {
    h = std::string("LSImmPre sz=") + Sz(a.size) + " sgn=" + (a.is_signed ? "1" : "0") +
        " st=" + (a.is_store ? "1" : "0");
  }
  void LoadStoreImmPostIndex(const Decoder::LoadStoreImmArgs& a) {
    h = std::string("LSImmPost sz=") + Sz(a.size) + " sgn=" + (a.is_signed ? "1" : "0") +
        " st=" + (a.is_store ? "1" : "0");
  }
  void LoadStorePair(const Decoder::LoadStorePairArgs& a) {
    h = std::string("LSPair sz=") + Sz(a.size) + " sgn=" + (a.is_signed ? "1" : "0") +
        " st=" + (a.is_store ? "1" : "0") + " pre=" + (a.is_preindex ? "1" : "0") +
        " post=" + (a.is_postindex ? "1" : "0");
  }
  void LoadStoreReg(const Decoder::LoadStoreRegArgs& a) {
    h = std::string("LSReg sz=") + Sz(a.size) + " sgn=" + (a.is_signed ? "1" : "0") +
        " st=" + (a.is_store ? "1" : "0") + " x=" + (a.is_64bit_target ? "1" : "0");
  }
  // Record the sign: `ldrsw <Xt>, <label>` is a literal load, and its
  // sign-extension is the same class of bug as the LDPSW one, so it is worth
  // actually checking rather than accepting any LoadLiteral as correct.
  void LoadLiteral(const Decoder::LoadLiteralArgs& a) {
    h = std::string("LoadLiteral sgn=") + (a.is_signed ? "1" : "0");
  }
  void LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs&) { h = "LSExcl"; }
  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& a) {
    h = std::string("SLSImm sz=") + SSz(a.size) + " st=" + (a.is_store ? "1" : "0");
  }
  void SimdLoadStoreImmPreIndex(const Decoder::SimdLoadStoreImmArgs&) { h = "SLSImmPre"; }
  void SimdLoadStoreImmPostIndex(const Decoder::SimdLoadStoreImmArgs&) { h = "SLSImmPost"; }
  void SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs& a) {
    h = std::string("SLSPair sz=") + SSz(a.size) + " st=" + (a.is_store ? "1" : "0");
  }
  void SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs& a) {
    h = std::string("SLSReg sz=") + SSz(a.size) + " st=" + (a.is_store ? "1" : "0");
  }
  void SimdLoadLiteral(const Decoder::SimdLoadLiteralArgs&) { h = "SimdLoadLiteral"; }
  void AdvSimdMultiStruct(uint8_t, uint8_t, uint8_t, uint8_t, bool, bool, bool, uint8_t, bool) {
    h = "AdvSimdMultiStruct";
  }
  void AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs&) { h = "AdvSimdSingleStruct"; }

  // --- suspect SIMD families (narrowing / saturating / permute) — record opcode ---
  void AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& a) {
    h = std::string("SimdShiftImm op=") + std::to_string((int)a.opcode) +
        " u=" + (a.u ? "1" : "0") + " sc=" + (a.scalar ? "1" : "0");
  }
  void AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& a) {
    h = std::string("Simd3Diff op=") + std::to_string((int)a.opcode) + " q=" + (a.q ? "1" : "0");
  }
  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& a) {
    h = std::string("Simd2RegMisc op=") + std::to_string((int)a.opcode) + " u=" + (a.u ? "1" : "0");
  }
  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& a) {
    h = std::string("Simd3Same op=") + std::to_string((int)a.opcode);
  }
  void AdvSimdCopy(const Decoder::AdvSimdCopyArgs& a) {
    h = std::string("SimdCopy op=") + std::to_string((int)a.opcode);
  }
  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs&) { h = "SimdSc3Same"; }
  void AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs&) { h = "SimdSc2RegMisc"; }
  void AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs&) { h = "SimdScPairwise"; }
  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs&) { h = "SimdVecXIdx"; }
  void AdvSimdScalarXIndexedElement(const Decoder::AdvSimdScalarXIdxArgs&) { h = "SimdScXIdx"; }
  void AdvSimdTableLookup(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, bool) { h = "SimdTBL"; }
  void AdvSimdPermute(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, bool) { h = "SimdPermute"; }
  void AdvSimdExtract(uint8_t, uint8_t, uint8_t, uint8_t, bool) { h = "SimdEXT"; }
  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs&) { h = "SimdModImm"; }
  void AdvSimdDotProduct(const Decoder::DotProductArgs&) { h = "SimdDot"; }
  void AdvSimdMatMul(const Decoder::MatMulArgs&) { h = "SimdMatMul"; }
  void AdvSimdFcma(const Decoder::FcmaArgs&) { h = "SimdFcma"; }
  void AdvSimdFcmaIdx(const Decoder::FcmaIdxArgs&) { h = "SimdFcmaIdx"; }
  void AdvSimdBf16ThreeSame(const Decoder::Bf16ThreeSameArgs&) { h = "SimdBf16"; }

  // --- integer / branch / fp (handler name only) ---
  void AddSubImm(const Decoder::AddSubImmArgs&) { h = "AddSubImm"; }
  void AddSubImmTags(const Decoder::AddSubImmTagsArgs&) { h = "AddSubImmTags"; }
  void LogicalImm(const Decoder::LogicalImmArgs&) { h = "LogicalImm"; }
  void MoveWide(const Decoder::MoveWideArgs&) { h = "MoveWide"; }
  void PcRelAddr(const Decoder::PcRelAddrArgs& a) {
    h = std::string("PcRelAddr adrp=") + (a.is_adrp ? "1" : "0");
  }
  void Bitfield(const Decoder::BitfieldArgs&) { h = "Bitfield"; }
  void Extr(uint8_t, uint8_t, uint8_t, uint8_t, bool) { h = "Extr"; }
  void BranchImm(const Decoder::BranchImmArgs&) { h = "BranchImm"; }
  void BranchCond(const Decoder::BranchCondArgs&) { h = "BranchCond"; }
  void BranchReg(const Decoder::BranchRegArgs&) { h = "BranchReg"; }
  void CompareAndBranch(const Decoder::CompareAndBranchArgs&) { h = "CompareAndBranch"; }
  void TestAndBranch(const Decoder::TestAndBranchArgs&) { h = "TestAndBranch"; }
  void Svc(const Decoder::SvcArgs&) { h = "Svc"; }
  void Brk(uint16_t) { h = "Brk"; }
  void Mrs(const Decoder::MrsArgs&) { h = "Mrs"; }
  void Msr(const Decoder::MsrArgs&) { h = "Msr"; }
  void LogicalShiftedReg(const Decoder::LogicalShiftedRegArgs&) { h = "LogicalShiftedReg"; }
  void AddSubShiftedReg(const Decoder::AddSubShiftedRegArgs&) { h = "AddSubShiftedReg"; }
  void AddSubExtendedReg(const Decoder::AddSubExtendedRegArgs&) { h = "AddSubExtendedReg"; }
  void AddSubWithCarry(uint8_t, uint8_t, uint8_t, bool, bool, bool) { h = "AddSubWithCarry"; }
  void ConditionalSelect(const Decoder::ConditionalSelectArgs&) { h = "ConditionalSelect"; }
  void ConditionalCompare(const Decoder::ConditionalCompareArgs&) { h = "ConditionalCompare"; }
  void DataProc1Src(uint8_t, uint8_t, uint8_t, bool) { h = "DataProc1Src"; }
  void DataProc2Src(const Decoder::DataProc2SrcArgs&) { h = "DataProc2Src"; }
  void DataProc3Src(const Decoder::DataProc3SrcArgs&) { h = "DataProc3Src"; }
  void Nop() { h = "Nop"; }
  void DataMemoryBarrier() { h = "DataMemoryBarrier"; }
  void IcIvau(uint8_t) { h = "IcIvau"; }
  void MteDataProc(const Decoder::MteDataProcArgs&) { h = "MteDataProc"; }
  void MteLoadStore(const Decoder::MteLoadStoreArgs&) { h = "MteLoadStore"; }
  void FpIntConversion(const Decoder::FpIntConvArgs&) { h = "FpIntConversion"; }
  void FpMovImmediate(uint8_t, uint8_t, uint8_t) { h = "FpMovImmediate"; }
  void FpDataProc1(const Decoder::FpDataProc1Args&) { h = "FpDataProc1"; }
  void FpDataProc2(const Decoder::FpDataProc2Args&) { h = "FpDataProc2"; }
  void FpDataProc3(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, bool, bool) { h = "FpDataProc3"; }
  void FpCompare(const Decoder::FpCompareArgs&) { h = "FpCompare"; }
  void FpConditionalCompare(const Decoder::FpConditionalCompareArgs&) { h = "FpCondCompare"; }
  void FpCondSelect(uint8_t, uint8_t, uint8_t, uint8_t, Decoder::Condition) { h = "FpCondSelect"; }
  void FpFixedPointConversion(const Decoder::FpFixedPointArgs&) { h = "FpFixedPoint"; }
  void Sha512(Decoder::Sha512Op, uint8_t, uint8_t, uint8_t) { h = "Sha512"; }
  void Eor3(uint8_t, uint8_t, uint8_t, uint8_t) { h = "Eor3"; }
  void Bcax(uint8_t, uint8_t, uint8_t, uint8_t) { h = "Bcax"; }
  void Rax1(uint8_t, uint8_t, uint8_t) { h = "Rax1"; }
  void Xar(uint8_t, uint8_t, uint8_t, uint8_t) { h = "Xar"; }
  void Sm4e(uint8_t, uint8_t) { h = "Sm4e"; }
  void Sm4ekey(uint8_t, uint8_t, uint8_t) { h = "Sm4ekey"; }
  void Sm3ss1(uint8_t, uint8_t, uint8_t, uint8_t) { h = "Sm3ss1"; }
  void Sm3tt(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t) { h = "Sm3tt"; }
  void Sm3partw1(uint8_t, uint8_t, uint8_t) { h = "Sm3partw1"; }
  void Sm3partw2(uint8_t, uint8_t, uint8_t) { h = "Sm3partw2"; }
  void CryptoAes(uint8_t, uint8_t, uint8_t) { h = "CryptoAes"; }
  void CryptoSha3Reg(uint8_t, uint8_t, uint8_t, uint8_t) { h = "CryptoSha3Reg"; }
  void CryptoSha2Reg(uint8_t, uint8_t, uint8_t) { h = "CryptoSha2Reg"; }
  void Undefined() { h = "UNDEF"; }
};

// Consistency rule: does Berberis's recorded decode `h` agree with objdump's
// mnemonic `mn` for the SUSPECT (positional-corruption) classes? Returns a
// non-empty reason string when they DISAGREE, "" when consistent or not a
// suspect class. We only assert on classes a mis-dispatch would silently
// corrupt (signed loads, load/store-pair sign, the unscaled/unpriv variants),
// since a full 245-mnemonic map is unnecessary and noisy.
std::string Inconsistent(const std::string& mn, const std::string& h) {
  // An encoding the decoder does not implement is a coverage question, not a
  // mis-dispatch; UndefinedCoverageBaseline owns it. Without this every
  // unimplemented extension would also be reported here as its category being
  // "decoded as [UNDEF]", which buries the real mis-dispatches.
  if (h == "UNDEF") return "";
  // Match a whole field token: a leading space (or string start) avoids "st=1"
  // matching inside "post=1" and "st=0" inside "post=0".
  auto has = [&](const char* s) {
    std::string needle = std::string(" ") + s;
    return h.find(needle) != std::string::npos || h.rfind(s, 0) == 0;
  };
  // Signed-load mnemonics MUST decode to a load with sgn=1.
  if (mn.rfind("ldrs", 0) == 0 || mn.rfind("ldurs", 0) == 0 || mn == "ldpsw" ||
      mn == "ldtrs" || mn.rfind("ldtrs", 0) == 0) {
    if (!has("sgn=1")) return "signed-load decoded sgn!=1";
    if (has("st=1")) return "signed-load decoded as store";
  }
  // Plain (unsigned) integer loads must NOT be sgn=1 and must be st=0.
  if (mn == "ldr" || mn == "ldrb" || mn == "ldrh" || mn == "ldur" || mn == "ldurb" ||
      mn == "ldurh" || mn == "ldp" || mn == "ldtr" || mn == "ldtrb" || mn == "ldtrh") {
    if ((has("LSImm") || has("LSReg") || has("LSPair")) && has("st=1"))
      return "load decoded as store";
    if ((has("LSImm") || has("LSReg") || has("LSPair")) && has("sgn=1"))
      return "unsigned load decoded sgn=1";
  }
  // Stores must decode st=1.
  if (mn == "str" || mn == "strb" || mn == "strh" || mn == "stur" || mn == "sturb" ||
      mn == "sturh" || mn == "stp" || mn == "sttr") {
    if ((has("LSImm") || has("LSReg") || has("LSPair")) && has("st=0"))
      return "store decoded as load";
  }
  // ADRP vs ADR.
  if (mn == "adrp" && has("PcRelAddr") && has("adrp=0")) return "adrp decoded as adr";
  if (mn == "adr" && has("PcRelAddr") && has("adrp=1")) return "adr decoded as adrp";

  // --- SIMD category dispatch (fragmentation-relevant: narrowing / permute /
  // copy / extract). Strip a trailing "2" (upper-half variant) for matching. ---
  auto base = mn;
  if (base.size() > 1 && base.back() == '2') base.pop_back();
  auto in = [&](std::initializer_list<const char*> set) {
    for (auto* s : set)
      if (base == s) return true;
    return false;
  };
  auto cat = [&](const char* c) { return h.rfind(c, 0) == 0; };  // h starts with category
  // narrowing rounding/saturating shift-right (AdvSimdShiftByImm)
  // SQSHL and UQSHL are the two mnemonics in this list with BOTH a shift-by-
  // immediate form and a three-same REGISTER form (`sqshl v0.4s, v1.4s, v2.4s`),
  // so Simd3Same is a correct decode for them --- the three-same table below
  // maps them to opcodes 32/33. Every other mnemonic here is immediate-only.
  if (in({"shrn", "rshrn", "sqshrn", "sqrshrn", "uqshrn", "uqrshrn", "sqshrun", "sqrshrun",
          "sshr", "ushr", "srshr", "urshr", "ssra", "usra", "srsra", "ursra", "shl", "sli",
          "sri", "sshll", "ushll", "sqshl", "uqshl", "sqshlu"}) &&
      !cat("SimdShiftImm") && !cat("SimdSc") &&
      !(in({"sqshl", "uqshl"}) && cat("Simd3Same")))
    return "shift-imm decoded as [" + h + "]";
  // narrowing add/sub high-half + widening add/sub (AdvSimdThreeDiff; no indexed
  // form, so strictly three-diff)
  if (in({"addhn", "subhn", "raddhn", "rsubhn", "saddl", "ssubl", "uaddl", "usubl", "saddw",
          "ssubw", "uaddw", "usubw", "pmull"}) &&
      !cat("Simd3Diff"))
    return "three-diff decoded as [" + h + "]";
  // multiply-long: objdump prints the same mnemonic for the SIMD vector
  // (three-diff), SIMD by-element (indexed), AND scalar-integer-GP
  // (DataProc3Src, e.g. `smull x0,w1,w2`) forms — accept any of them.
  if (in({"smlal", "umlal", "smlsl", "umlsl", "smull", "umull", "sqdmlal", "sqdmlsl", "sqdmull"}) &&
      !cat("Simd3Diff") && !cat("SimdVecXIdx") && !cat("SimdSc") && !cat("DataProc3Src"))
    return "mul-long decoded as [" + h + "]";
  // saturating extract-narrow + xtn (AdvSimdTwoRegMisc)
  if (in({"xtn", "sqxtn", "uqxtn", "sqxtun"}) && !cat("Simd2RegMisc") && !cat("SimdSc"))
    return "xtn decoded as [" + h + "]";
  // permute / table / extract / copy (byte-rearrange — the strongest fragmentation class)
  if (in({"tbl", "tbx"}) && !cat("SimdTBL")) return "tbl decoded as [" + h + "]";
  if (in({"zip1", "zip2", "uzp1", "uzp2", "trn1", "trn2"}) && !cat("SimdPermute"))
    return "permute decoded as [" + h + "]";
  if (base == "ext" && !cat("SimdEXT")) return "ext decoded as [" + h + "]";
  if (in({"dup", "ins", "smov", "umov"}) && !cat("SimdCopy") && !cat("SimdSc"))
    return "copy decoded as [" + h + "]";

  // WITHIN-category three-same opcode check (the "CMGT-vs-SMAX" mis-map class).
  // For integer three-same mnemonics, the decoder's opcode enum value MUST match
  // the mnemonic. Parse "Simd3Same op=N" and compare to the expected enum value.
  if (cat("Simd3Same op=")) {
    int op = std::atoi(h.c_str() + h.find("op=") + 3);
    static const std::pair<const char*, int> kTS[] = {
        {"and", 0},   {"bic", 1},   {"orr", 2},   {"orn", 3},    {"eor", 4},    {"bsl", 5},
        {"bit", 6},   {"bif", 7},   {"add", 8},   {"sub", 9},    {"cmeq", 10},  {"cmtst", 11},
        {"smax", 12}, {"smin", 13}, {"umax", 14}, {"umin", 15},  {"cmgt", 16},  {"cmhi", 17},
        {"cmge", 18}, {"cmhs", 19}, {"shadd", 20},{"uhadd", 21}, {"sqadd", 22}, {"uqadd", 23},
        {"srhadd", 24},{"urhadd", 25},{"shsub", 26},{"uhsub", 27},{"sqsub", 28},{"uqsub", 29},
        {"sshl", 30}, {"ushl", 31}, {"sqshl", 32},{"uqshl", 33}, {"srshl", 34}, {"urshl", 35},
        {"sqrshl", 36},{"uqrshl", 37},{"addp", 38},{"mul", 39},  {"mla", 40},   {"mls", 41},
        {"smaxp", 42},{"umaxp", 43},{"sminp", 44},{"uminp", 45}};
    for (auto& e : kTS)
      if (base == e.first && op != e.second)
        return "three-same " + base + " decoded as opcode " + std::to_string(op) +
               " (expected " + std::to_string(e.second) + ")";
  }
  // NOTE: "decodes to UNDEF" is deliberately NOT a mis-dispatch. On a corpus
  // harvested from modern libraries most UNDEFs are ISA extensions outside the
  // implemented baseline (SVE/SVE2, SME/SME2, FP8, FEAT_FHM, FEAT_FAMINMAX,
  // FEAT_THE), where UNDEF is the correct answer. Undefined coverage is tracked
  // separately, against a committed baseline, by UndefinedCoverageBaseline.
  return "";
}

struct Entry {
  uint32_t enc;
  std::string mnemonic;
};

// Locates the committed corpus. DIGITALIS_RAST_ENC overrides it, which is how a
// much larger local sweep is run (the committed file is capped per mnemonic to
// stay a reviewable size; harvest_encodings.py --cap 0 produces the full set).
std::string CorpusPath(const char* name) {
  if (const char* env = std::getenv("DIGITALIS_RAST_ENC")) return env;
  return android::base::GetExecutableDirectory() + "/" + name;
}

std::vector<Entry> LoadCorpus(const std::string& path) {
  std::vector<Entry> out;
  std::ifstream f(path);
  if (!f.good()) return out;
  std::string line;
  while (std::getline(f, line)) {
    // Skip comments and blanks. Without this the header lines parse as data and
    // report as bogus "UNDEF vs ARM64" mismatches.
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string enc_hex, mn;
    if (!(ss >> enc_hex >> mn)) continue;
    out.push_back({static_cast<uint32_t>(std::strtoul(enc_hex.c_str(), nullptr, 16)), mn});
  }
  return out;
}

std::string DecodeOf(uint32_t enc) {
  Recorder rec;
  Recorder::Decoder dec(&rec);
  rec.h.clear();
  dec.Decode(reinterpret_cast<const uint16_t*>(&enc));
  return rec.h;
}

// Mis-dispatch gate. This is the check no JIT-vs-interpreter differential can
// make: both tiers decode through the same Decoder, so a decoder that routes an
// encoding to the wrong handler is invisible to them and produces identical
// wrong answers. objdump is an independent implementation of the same ARM ARM
// tables, so a disagreement here is a decoder bug in almost every case.
TEST(DecoderObjdumpDiff, NoMisdispatch) {
  auto corpus = LoadCorpus(CorpusPath(kCorpusFile));
  ASSERT_FALSE(corpus.empty()) << "corpus not found next to the test binary";
  int mismatches = 0;
  for (const auto& e : corpus) {
    std::string h = DecodeOf(e.enc);
    std::string why = Inconsistent(e.mnemonic, h);
    if (!why.empty()) {
      mismatches++;
      if (mismatches <= 80) {
        ADD_FAILURE() << std::hex << "0x" << e.enc << std::dec << " objdump=" << e.mnemonic
                      << " berberis=[" << h << "]  <-- " << why;
      }
    }
  }
  fprintf(stderr, "DecoderObjdumpDiff: checked=%zu mismatches=%d\n", corpus.size(), mismatches);
  EXPECT_EQ(mismatches, 0);
}

// Undefined-coverage baseline.
//
// An encoding that objdump names but Berberis decodes as Undefined is a
// coverage gap, not a mis-dispatch. On a corpus harvested from current
// libraries most such gaps are ISA extensions Digitalis does not implement
// (SVE/SVE2, SME/SME2, FP8, FEAT_FHM, FEAT_FAMINMAX, FEAT_THE), where Undefined
// is exactly right. Failing on those would make the gate useless.
//
// So the assertion is on the SET of mnemonics that decode to Undefined, against
// a committed baseline, and it is deliberately asymmetric:
//   * a mnemonic that newly decodes to Undefined FAILS --- that is a decoder
//     regression, something that used to be handled and no longer is;
//   * a baseline mnemonic that now decodes is reported, not failed --- that is
//     someone implementing an instruction, and the test should not block it.
//     Prune the baseline file in the same commit.
TEST(DecoderObjdumpDiff, UndefinedCoverageBaseline) {
  auto corpus = LoadCorpus(CorpusPath(kCorpusFile));
  ASSERT_FALSE(corpus.empty()) << "corpus not found next to the test binary";

  // The baseline always sits next to the test binary; DIGITALIS_RAST_ENC only
  // overrides the corpus, never the baseline.
  std::set<std::string> baseline;
  {
    std::ifstream f(android::base::GetExecutableDirectory() + "/" + kUndefinedBaselineFile);
    ASSERT_TRUE(f.good()) << "missing undefined-coverage baseline";
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      baseline.insert(line);
    }
  }

  std::set<std::string> undefined_now;
  for (const auto& e : corpus) {
    if (DecodeOf(e.enc) == "UNDEF") undefined_now.insert(e.mnemonic);
  }

  std::vector<std::string> regressed, implemented;
  for (const auto& mn : undefined_now)
    if (!baseline.count(mn)) regressed.push_back(mn);
  for (const auto& mn : baseline)
    if (!undefined_now.count(mn)) implemented.push_back(mn);

  for (const auto& mn : implemented) {
    fprintf(stderr,
            "DecoderObjdumpDiff: '%s' now decodes -- drop it from %s\n",
            mn.c_str(),
            kUndefinedBaselineFile);
  }
  std::string detail;
  for (const auto& mn : regressed) detail += " " + mn;
  EXPECT_TRUE(regressed.empty())
      << "these mnemonics newly decode to Undefined (decoder coverage regression):" << detail;
}

// MSR (immediate) to a PSTATE field (CRn=0b0100) must decode to Nop, not
// Undefined. Found via the rasterizer-range decoder diff: Chromium's MTE-
// hardened Scudo toggles `msr TCO, #imm`, which previously decoded as
// Undefined (a latent SIGILL had that dead MTE path ever executed). All PSTATE
// immediates are EL0-irrelevant for translation and must be NOPs.
TEST(DecoderObjdumpDiff, MsrImmediatePstateIsNop) {
  // Base MSR-immediate encoding; insert op1/CRm/op2. Verified: TCO #0 == the
  // 0xD503409F seen in libchrome's disassembly.
  auto msr_imm = [](uint32_t op1, uint32_t crm, uint32_t op2) -> uint32_t {
    return 0xD500401Fu | (op1 << 16) | (crm << 8) | (op2 << 5);
  };
  struct Case {
    const char* name;
    uint32_t op1;
    uint32_t op2;
  };
  // (op1, op2) per the ARM ARM PSTATE field selectors.
  const Case cases[] = {
      {"UAO", 0b000, 0b011},     {"PAN", 0b000, 0b100},  {"SPSel", 0b000, 0b101},
      {"SSBS", 0b011, 0b001},    {"DIT", 0b011, 0b010},  {"TCO", 0b011, 0b100},
      {"DAIFSet", 0b011, 0b110}, {"DAIFClr", 0b011, 0b111},
  };
  EXPECT_EQ(msr_imm(0b011, 0, 0b100), 0xD503409Fu);  // TCO #0 ground truth
  for (const auto& c : cases) {
    for (uint32_t crm = 0; crm < 16; ++crm) {
      uint32_t enc = msr_imm(c.op1, crm, c.op2);
      Recorder rec;
      Recorder::Decoder dec(&rec);
      rec.h.clear();
      dec.Decode(reinterpret_cast<const uint16_t*>(&enc));
      EXPECT_EQ(rec.h, "Nop") << "msr " << c.name << ", #" << crm << " (0x" << std::hex << enc
                              << ") decoded as [" << rec.h << "]";
    }
  }
}

}  // namespace

}  // namespace berberis
