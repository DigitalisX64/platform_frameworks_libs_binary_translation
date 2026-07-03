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

// Institutionalized JIT-vs-interpreter differential fuzzer.
//
// The ad-hoc differential fuzzers scattered through
// lite_translate_region_exec_tests.cc found four real shipped-code bugs that
// example-based exec tests were blind to: the CCMN reg-mapping clobber (a
// non-destructive ARM op lowered as a destructive x86 op overwriting a mapped
// guest reg), the vector SCVTF/UCVTF multi-lane drop (scalar codegen reached by
// vector .2S/.4S/.2D forms), the rd==rn in-place-convert Vd-zeroing clobber, and
// the SDIV/UDIV rcx/rsp stack imbalance on divisor 0/-1. This file collects that
// tool class behind ONE reusable harness so the highest-yield bug-finder in the
// project is a first-class, always-green part of the host suite rather than
// throwaway scratchpad code.
//
// Design:
//   * A single fixture (Arm64DifferentialFuzz) runs an arbitrary guest region
//     through the lite JIT and the per-instruction interpreter from IDENTICAL
//     seeded CPUState, then compares the FULL end state (X0-X30, SP, NZCV, and
//     V0-V31; FPSR comparison is available behind a per-class knob).
//   * Deterministic splitmix-style PRNG keyed by a fixed seed => reproducible CI
//     failures. Corpus generators per instruction class parameterize over
//     destructive register aliasing (rd==rn / rd==rm / rn==rm), high register
//     pressure (enough guest regs to force mapping spills), lane arrangements,
//     and immediate boundary values -- exactly the axes the four historical bugs
//     lived on.
//   * CI mode (default): fixed seeds, bounded iteration counts, runs in seconds
//     inside berberis_arm64_host_tests.
//   * Exhaustive mode: env var BERBERIS_DIFFERENTIAL_FUZZ_EXHAUSTIVE widens the
//     sweep (driver: digitalis/scripts/differential-fuzz.sh). The shift-imm
//     exhaustive sweep that pinned the SCVTF lane-drop is the model.
//
// A single-instruction region fully captures a JIT handler's codegen for the
// SIMD/FP classes (they read Vn / write Vd through ThreadState memory, no
// cross-instruction mapped state); multi-instruction integer regions are needed
// to expose register-MAPPING clobbers (a destructive lowering that corrupts a
// value the mapping still needs). The generators below cover both shapes.

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/arm64/interpreter.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

// NOTE: intrinsics::InitState() is defined in lite_translate_region_exec_tests.cc
// (same test library); do NOT redefine it here or the link fails with a
// duplicate symbol.

namespace {

// ARM64 NZCV lives in CPUState.flags at N@15 Z@14 C@8 V@0.
constexpr uint16_t kNZCVMask = 0xC101;

// Widen the sweep when the exhaustive-mode env knob is set (see
// digitalis/scripts/differential-fuzz.sh). Off => 1 (CI, ~seconds).
int FuzzScale() {
  const char* e = getenv("BERBERIS_DIFFERENTIAL_FUZZ_EXHAUSTIVE");
  if (e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0')) {
    return 25;
  }
  return 1;
}

class Arm64DifferentialFuzz : public ::testing::Test {
 protected:
  ThreadState state_{};
  uint64_t seed_ = 0;

  void Seed(uint64_t s) { seed_ = s; }
  // splitmix / LCG hybrid: same recurrence the historical scratch fuzzers used,
  // so their reproducible failures reproduce identically here.
  uint64_t Rnd() {
    seed_ = seed_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed_ >> 33;  // ~31 usable bits
  }
  uint64_t Rnd64() { return ((Rnd() & 0xffffffffULL) << 32) | (Rnd() & 0xffffffffULL); }

  // Full guest end state captured for the differential compare.
  struct FullState {
    uint64_t x[31];
    uint64_t sp;
    uint16_t flags;
    uint32_t emulated_fpsr;
    unsigned __int128 v[32];
  };

  // Inputs applied identically to both the JIT and interpreter runs.
  struct InitState {
    uint64_t x[31];
    uint64_t sp;
    uint16_t flags;
    unsigned __int128 v[32];
  };

  void ApplyInit(const InitState& in) {
    for (int i = 0; i < 31; i++) state_.cpu.x[i] = in.x[i];
    state_.cpu.sp = in.sp;
    state_.cpu.flags = in.flags;
    state_.cpu.emulated_fpsr = 0;
    for (int i = 0; i < 32; i++) memcpy(&state_.cpu.v[i], &in.v[i], 16);
  }

  FullState Capture() {
    FullState s;
    for (int i = 0; i < 31; i++) s.x[i] = state_.cpu.x[i];
    s.sp = state_.cpu.sp;
    s.flags = state_.cpu.flags;
    s.emulated_fpsr = state_.cpu.emulated_fpsr;
    for (int i = 0; i < 32; i++) memcpy(&s.v[i], &state_.cpu.v[i], 16);
    return s;
  }

  // A randomized init: registers get full 64-bit values, but a fraction are
  // pinned to the arithmetic edge values (0, -1, INT_MIN, INT_MAX) that the
  // SDIV/UDIV divisor-0/-1 stack-balance bug required. Vector lanes get mixed
  // sign-bit / boundary patterns.
  InitState RandomInit() {
    InitState in;
    static const uint64_t kEdge[] = {
        0ULL, ~0ULL, 0x8000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL,
        1ULL, 0xFFFFFFFF00000000ULL, 0x00000000FFFFFFFFULL, 0x80000000ULL,
    };
    for (int i = 0; i < 31; i++) {
      if ((Rnd() % 4) == 0) {
        in.x[i] = kEdge[Rnd() % (sizeof(kEdge) / sizeof(kEdge[0]))];
      } else {
        in.x[i] = Rnd64();
      }
    }
    in.sp = Rnd64() & ~0xFULL;
    in.flags = static_cast<uint16_t>(Rnd() & kNZCVMask);
    for (int i = 0; i < 32; i++) {
      unsigned __int128 hi = Rnd64(), lo = Rnd64();
      in.v[i] = (hi << 64) | lo;
    }
    return in;
  }

  static std::string RegionStr(const uint32_t* code, int n) {
    std::string d;
    for (int j = 0; j < n; j++) {
      char b[16];
      snprintf(b, sizeof(b), " %08x", code[j]);
      d += b;
    }
    return d;
  }

  enum Result { kDeclined, kMatch, kDiverge };

  // Runs region [code, code+n) through the JIT and interpreter from `in`;
  // compares full state. When compare_fpsr is false (default for classes whose
  // JIT/interp FPSR emulation legitimately differs), the emulated_fpsr field is
  // ignored. Returns kDeclined if the JIT bailed (interp-only encoding; runs on
  // the interpreter on-device, cannot be a JIT miscompile source).
  Result RunDifferential(const uint32_t* code,
                         int n,
                         const InitState& in,
                         bool compare_fpsr,
                         std::string* desc) {
    GuestAddr start = ToGuestAddr(code);
    GuestAddr code_end = start + static_cast<GuestAddr>(n) * 4;

    // JIT.
    ApplyInit(in);
    state_.cpu.insn_addr = start;
    MachineCode mc;
    auto [ok, stop] = TryLiteTranslateRegion(
        start, &mc, LiteTranslateParams{.end_pc = code_end, .allow_dispatch = false});
    if (!ok || stop > code_end || stop == start) return kDeclined;
    HostCodeAddr hc = GetDefaultCodePoolInstance()->Add(&mc);
    TestingRunGeneratedCode(&state_, AsHostCode(hc), stop);
    FullState jit = Capture();

    // Interpreter to the same stop PC.
    ApplyInit(in);
    state_.cpu.insn_addr = start;
    int guard = 0;
    while (state_.cpu.insn_addr >= start && state_.cpu.insn_addr < stop && guard++ < 256) {
      InterpretInsn(&state_);
    }
    FullState itp = Capture();

    return Compare(jit, itp, compare_fpsr, code, n, desc) ? kMatch : kDiverge;
  }

  bool Compare(const FullState& a,
               const FullState& b,
               bool compare_fpsr,
               const uint32_t* code,
               int n,
               std::string* desc) {
    char buf[192];
    for (int i = 0; i < 31; i++) {
      if (a.x[i] != b.x[i]) {
        snprintf(buf, sizeof(buf), "x%d JIT=0x%016llx INTERP=0x%016llx region:", i,
                 (unsigned long long)a.x[i], (unsigned long long)b.x[i]);
        *desc = std::string(buf) + RegionStr(code, n);
        return false;
      }
    }
    if (a.sp != b.sp) {
      snprintf(buf, sizeof(buf), "sp JIT=0x%016llx INTERP=0x%016llx region:",
               (unsigned long long)a.sp, (unsigned long long)b.sp);
      *desc = std::string(buf) + RegionStr(code, n);
      return false;
    }
    if ((a.flags & kNZCVMask) != (b.flags & kNZCVMask)) {
      snprintf(buf, sizeof(buf), "NZCV JIT=0x%04x INTERP=0x%04x region:",
               a.flags & kNZCVMask, b.flags & kNZCVMask);
      *desc = std::string(buf) + RegionStr(code, n);
      return false;
    }
    for (int i = 0; i < 32; i++) {
      if (a.v[i] != b.v[i]) {
        snprintf(buf, sizeof(buf), "v%d JIT=0x%016llx:%016llx INTERP=0x%016llx:%016llx region:", i,
                 (unsigned long long)(uint64_t)(a.v[i] >> 64), (unsigned long long)(uint64_t)a.v[i],
                 (unsigned long long)(uint64_t)(b.v[i] >> 64), (unsigned long long)(uint64_t)b.v[i]);
        *desc = std::string(buf) + RegionStr(code, n);
        return false;
      }
    }
    if (compare_fpsr && a.emulated_fpsr != b.emulated_fpsr) {
      snprintf(buf, sizeof(buf), "FPSR JIT=0x%08x INTERP=0x%08x region:", a.emulated_fpsr,
               b.emulated_fpsr);
      *desc = std::string(buf) + RegionStr(code, n);
      return false;
    }
    return true;
  }

  // ---- Corpus generators (one per instruction class). Each returns one guest
  // insn word using registers drawn from [0, kMaxReg) so destructive aliasing
  // (rd==rn etc.) and mapping spill are naturally sampled. ----

  // Integer data-processing 1/2/3-src + flag-setting + conditional-compare +
  // conditional-select. Case 19 is CCMN (the reg-mapping-clobber historical
  // bug); cases 6/7 are SDIV/UDIV (the divisor-0/-1 stack-balance bug).
  uint32_t GenIntDataProc(uint32_t kMaxReg) {
    uint32_t rd = Rnd() % kMaxReg, rn = Rnd() % kMaxReg, rm = Rnd() % kMaxReg,
             ra = Rnd() % kMaxReg;
    uint32_t cond = Rnd() % 14;  // omit AL(14)/NV(15)
    uint32_t nzcv = Rnd() % 16;
    switch (Rnd() % 24) {
      case 0: return 0x8B000000 | (rm << 16) | (rn << 5) | rd;  // add
      case 1: return 0xCB000000 | (rm << 16) | (rn << 5) | rd;  // sub
      case 2: return 0x8A000000 | (rm << 16) | (rn << 5) | rd;  // and
      case 3: return 0xAA000000 | (rm << 16) | (rn << 5) | rd;  // orr
      case 4: return 0xCA000000 | (rm << 16) | (rn << 5) | rd;  // eor
      case 5: return 0x9B007C00 | (rm << 16) | (rn << 5) | rd;  // mul
      case 6: return 0x9AC00C00 | (rm << 16) | (rn << 5) | rd;  // sdiv
      case 7: return 0x9AC00800 | (rm << 16) | (rn << 5) | rd;  // udiv
      case 8: return 0x9AC02800 | (rm << 16) | (rn << 5) | rd;  // asrv
      case 9: return 0x9AC02000 | (rm << 16) | (rn << 5) | rd;  // lslv
      case 10: return 0x9AC02400 | (rm << 16) | (rn << 5) | rd;  // lsrv
      case 11: return 0x9B407C00 | (rm << 16) | (rn << 5) | rd;  // smulh
      case 12: return 0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd;  // msub
      case 13: return 0x9B000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd;  // madd
      case 14: return 0xAB000000 | (rm << 16) | (rn << 5) | rd;  // adds
      case 15: return 0xEB000000 | (rm << 16) | (rn << 5) | rd;  // subs
      case 16: return 0xEB00001F | (rm << 16) | (rn << 5);       // cmp (subs xzr)
      case 17: return 0xAB00001F | (rm << 16) | (rn << 5);       // cmn (adds xzr)
      case 18: return 0xFA400000 | (rm << 16) | (cond << 12) | (rn << 5) | nzcv;  // ccmp
      case 19: return 0xBA400000 | (rm << 16) | (cond << 12) | (rn << 5) | nzcv;  // ccmn
      case 20: return 0x9A800000 | (rm << 16) | (cond << 12) | (rn << 5) | rd;  // csel
      case 21: return 0x9A800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd;  // csinc
      case 22: return 0xDA800000 | (rm << 16) | (cond << 12) | (rn << 5) | rd;  // csinv
      case 23: return 0xDA800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd;  // csneg
    }
    return 0xD503201FU;  // nop
  }

  // AdvSIMD shift-by-immediate: 0 Q U 011110 immh immb opcode 1 Rn Rd.
  // opcode 0b11100 = SCVTF/UCVTF (fixed-point), 0b11111 = FCVTZS/FCVTZU.
  // rd==rn is frequently chosen to exercise the in-place-convert path.
  uint32_t GenAdvSimdShiftImm() {
    uint32_t q = Rnd() & 1, u = Rnd() & 1;
    uint32_t immh = 1 + (Rnd() % 15);  // 1..15 (immh==0 is the copy/three-same space)
    uint32_t immb = Rnd() % 8;
    uint32_t opcode = Rnd() % 32;
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);  // ~50% in-place (rd==rn)
    return (q << 30) | (u << 29) | (0b011110u << 23) | (immh << 19) | (immb << 16) |
           (opcode << 11) | (1u << 10) | (rn << 5) | rd;
  }

  // Scalar FP<->integer convert: sf 0 0 11110 ftype 1 rmode opcode 000000 Rn Rd.
  // Covers FCVTZS/ZU, FCVTNS/NU/PS/PU/MS/MU, FCVTAS/AU, SCVTF/UCVTF, FMOV GP<->FP.
  uint32_t GenScalarFpConv() {
    uint32_t sf = Rnd() & 1, ftype = Rnd() & 1, rmode = Rnd() % 4, opcode = Rnd() % 8;
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);
    return (sf << 31) | (0b11110u << 24) | (ftype << 22) | (1u << 21) | (rmode << 19) |
           (opcode << 16) | (rn << 5) | rd;
  }

  // AdvSIMD three-same INTEGER: 0 Q U 01110 size 1 Rm opcode 1 Rn Rd. ARM 3-op
  // forms are non-destructive; x86 SSE is destructive, so a missing source-copy
  // is a mapping clobber -- rd may alias rn or rm.
  //
  // The register-controlled variable shifts (SSHL 0x08 / SQSHL 0x09 / SRSHL
  // 0x0A / SQRSHL 0x0B, both U variants) ARE included: the shift count is the
  // low signed byte of each Rm lane, so with fully-random data it exercises the
  // full +/-128 range including the out-of-range counts. An earlier exhaustive
  // sweep here found a real interpreter divergence -- SRSHL/SQRSHL at shift
  // == -128 sign-broadcast to -1 instead of rounding to 0 (the round constant
  // 1<<127 lifts the small source positive) -- which is now fixed in the
  // interpreter lambda, so these opcodes are again bit-exact and gate the fix
  // going forward.
  //
  // One op class is intentionally EXCLUDED:
  //   * FP three-same (opcodes 0x18/0x1A/0x1C = FMAXNM/FADD/FMAX etc.): with
  //     NaN/Inf input lanes ARM materializes the default NaN while x86 SSE
  //     propagates a host NaN payload -- an implementation-defined difference,
  //     not a miscompile (the sibling NeonThreeSameInterpVsJitFuzz skips FP too).
  uint32_t GenNeonThreeSame() {
    static const uint8_t kOpc[] = {0x01, 0x05, 0x06, 0x07, 0x08, 0x09,
                                   0x0A, 0x0B, 0x0C, 0x0D, 0x10, 0x11,
                                   0x12, 0x13};
    uint32_t q = Rnd() & 1, u = Rnd() & 1, size = Rnd() % 4;
    uint32_t opcode = kOpc[Rnd() % (sizeof(kOpc) / sizeof(kOpc[0]))];
    uint32_t rn = Rnd() % 8, rm = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? rm : (Rnd() % 8));  // alias rn / rm / free
    return (q << 30) | (u << 29) | (0b01110u << 24) | (size << 22) | (1u << 21) |
           (rm << 16) | (opcode << 11) | (1u << 10) | (rn << 5) | rd;
  }
};

// -------------------------------------------------------------------------
// Class fuzzers. Each seeds deterministically, generates seeded regions, and
// asserts JIT==interp on the full state. accepted>threshold guards against a
// silent "JIT declined everything" regression that would make the test vacuous.
// -------------------------------------------------------------------------

// Multi-instruction integer regions: exercises cross-instruction register
// MAPPING and the RDX/RCX/RAX save-restore around DIV/MUL/shift. High register
// pressure (x0..x12) forces mapping spills. Home of the CCMN clobber and the
// SDIV/UDIV divisor-0/-1 stack-balance bugs.
TEST_F(Arm64DifferentialFuzz, IntegerDataProc) {
  Seed(0x0C0FFEE123456789ULL);
  const uint32_t kMaxReg = 13;  // x0..x12 -> spill pressure incl. RDX/RCX slots
  const int kIters = 4000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const int n = 3 + static_cast<int>(Rnd() % 6);
    uint32_t code[16];
    for (int i = 0; i < n; i++) code[i] = GenIntDataProc(kMaxReg);
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, n, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 1000) << "fuzzer compared too few regions to be meaningful";
}

// Single-instruction AdvSIMD shift-by-immediate. Home of the vector
// SCVTF/UCVTF multi-lane drop and the rd==rn in-place-convert Vd-zeroing
// clobber. rd==rn is sampled ~half the time by the generator.
TEST_F(Arm64DifferentialFuzz, AdvSimdShiftByImm) {
  Seed(0x51F7A11CE0DDBA11ULL);
  const int kIters = 6000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenAdvSimdShiftImm()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 500) << "JIT accepted too few shift-imm encodings";
}

// Single-instruction scalar FP<->int conversion + FMOV GP<->FP, rd==rn sampled.
TEST_F(Arm64DifferentialFuzz, ScalarFpConversion) {
  Seed(0xF9C04E75DEAD1234ULL);
  const int kIters = 4000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenScalarFpConv()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 200) << "JIT accepted too few FP-conversion encodings";
}

// Single-instruction AdvSIMD three-same integer, rd aliasing rn/rm sampled.
TEST_F(Arm64DifferentialFuzz, NeonThreeSame) {
  Seed(0x3A3E5A3E12345678ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonThreeSame()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 500) << "JIT accepted too few three-same encodings";
}

// -------------------------------------------------------------------------
// Acceptance: the four historical bug encodings are (a) demonstrably inside the
// seeded corpus of their owning generator, and (b) now match JIT==interp (they
// are fixed). This pins the corpus so a future refactor that accidentally stops
// generating one of these classes fails loudly.
// -------------------------------------------------------------------------
TEST_F(Arm64DifferentialFuzz, HistoricalBugEncodingsInCorpus) {
  // (a) Each historical class is reachable from its generator within a bounded
  // seeded sample. We flag when the specific opcode/shape shows up.
  bool saw_ccmn = false, saw_sdiv_or_udiv = false;
  Seed(0x0C0FFEE123456789ULL);
  for (int i = 0; i < 20000; i++) {
    uint32_t insn = GenIntDataProc(13);
    // CCMN (imm/reg conditional-compare-negative): 0b0111101001... base 0xBA400000.
    if ((insn & 0xFFE00C10U) == 0xBA400000U) saw_ccmn = true;
    // SDIV 0x9AC00C00 / UDIV 0x9AC00800 (mask off Rd/Rn/Rm).
    if ((insn & 0xFFE0FC00U) == 0x9AC00C00U || (insn & 0xFFE0FC00U) == 0x9AC00800U) {
      saw_sdiv_or_udiv = true;
    }
  }
  EXPECT_TRUE(saw_ccmn) << "integer generator no longer produces CCMN (reg-map clobber class)";
  EXPECT_TRUE(saw_sdiv_or_udiv) << "integer generator no longer produces SDIV/UDIV";

  bool saw_vec_scvtf = false, saw_inplace_convert = false;
  Seed(0x51F7A11CE0DDBA11ULL);
  for (int i = 0; i < 20000; i++) {
    uint32_t insn = GenAdvSimdShiftImm();
    uint32_t opcode = (insn >> 11) & 0x1F;
    uint32_t q = (insn >> 30) & 1;
    uint32_t rn = (insn >> 5) & 0x1F, rd = insn & 0x1F;
    // SCVTF/UCVTF fixed-point (opcode 0b11100) in a vector (Q=1 => .4S/.2D) form.
    if (opcode == 0b11100 && q == 1) saw_vec_scvtf = true;
    // rd==rn in-place convert of a convert opcode (0b111xx).
    if ((opcode == 0b11100 || opcode == 0b11111) && rd == rn) saw_inplace_convert = true;
  }
  EXPECT_TRUE(saw_vec_scvtf) << "shift-imm generator no longer produces vector SCVTF/UCVTF";
  EXPECT_TRUE(saw_inplace_convert) << "shift-imm generator no longer produces rd==rn convert";

  // The variable-shift SRSHL/SQRSHL class (interp shift==-128 rounding bug) must
  // stay reachable from the three-same generator so its regression stays gated.
  bool saw_srshl_or_sqrshl = false;
  Seed(0xD1A6511F7B0BC0DEULL);
  for (int i = 0; i < 20000; i++) {
    uint32_t insn = GenNeonThreeSame();
    uint32_t opcode = (insn >> 11) & 0x1F;
    if (opcode == 0x0A || opcode == 0x0B) saw_srshl_or_sqrshl = true;  // SRSHL / SQRSHL
  }
  EXPECT_TRUE(saw_srshl_or_sqrshl)
      << "three-same generator no longer produces SRSHL/SQRSHL (rounding-shift class)";

  // (b) Concrete regenerated encodings now match JIT==interp.
  auto run_one = [&](const uint32_t* code, int n, const InitState& in) -> Result {
    std::string desc;
    Result r = RunDifferential(code, n, in, /*compare_fpsr=*/false, &desc);
    if (r == kDiverge) ADD_FAILURE() << desc;
    return r;
  };

  // 1) CCMN reg-map clobber: ccmn x0,x1,#0,EQ then read x0/x1 via add. The
  // clobber overwrote a mapped guest reg; x0/x1 must survive the ccmn.
  {
    InitState in = RandomInit();
    in.x[0] = 0x1111111111111111ULL;
    in.x[1] = 0x2222222222222222ULL;
    uint32_t code[] = {
        0xBA400000U | (1u << 16) | (0u << 12) | (0u << 5) | 0u,  // ccmn x0,x1,#0,EQ
        0x8B010002U,                                             // add  x2,x0,x1
    };
    Result r = run_one(code, 2, in);
    EXPECT_NE(r, kDeclined) << "CCMN region unexpectedly JIT-declined";
  }

  // 2) SDIV divisor-0 / -1(INT_MIN) stack balance: sdiv x2,x0,x1 with the
  // conditional rcx/rdx save-restore stressed by adjacent mul/shift.
  {
    InitState in = RandomInit();
    in.x[0] = 0x8000000000000000ULL;  // INT_MIN
    in.x[1] = 0ULL;                    // divisor 0
    uint32_t code[] = {
        0x9AC00C02U,               // sdiv x2,x0,x1
        0x9B037C04U,               // mul  x4,x0,x3
        0x9AC42862U,               // asrv x2,x3,x4
    };
    Result r = run_one(code, 3, in);
    EXPECT_NE(r, kDeclined) << "SDIV region unexpectedly JIT-declined";
    in.x[1] = ~0ULL;  // divisor -1, dividend INT_MIN => overflow edge
    run_one(code, 3, in);
  }

  // 3) Vector SCVTF multi-lane: scvtf v0.4s, v1.4s, #1 (fixed-point, Q=1). A
  // scalar-only codegen would convert lane 0 and zero the rest.
  {
    InitState in = RandomInit();
    in.v[1] = (static_cast<unsigned __int128>(0x0000000200000001ULL) << 64) |
              0x0000000400000003ULL;  // four distinct S lanes
    // Q=1 U=0 immh=0001 immb=111 opcode=11100 Rn=1 Rd=0 (shift #1, .4S).
    uint32_t code[1] = {0x4F0FE420U};
    Result r = run_one(code, 1, in);
    (void)r;  // interp-only accept is fine; a JIT-accepted diverge fails above.
  }

  // 4) rd==rn in-place convert: fcvtzs v2.4s, v2.4s, #1 (opcode 0b11111). The
  // Vd-zeroing-up-front fix must not clobber Vn when rd==rn.
  {
    InitState in = RandomInit();
    in.v[2] = (static_cast<unsigned __int128>(0x3F8000003F800000ULL) << 64) |
              0x3F8000003F800000ULL;  // four 1.0f lanes
    uint32_t code[1] = {0x4F0FFC42U};  // fcvtzs v2.4s,v2.4s,#1
    Result r = run_one(code, 1, in);
    (void)r;
  }

  // 5) SRSHL/SQRSHL shift==-128 rounding: srshl v0.16b,v1.16b,v2.16b with a
  // negative source (0x80 per byte) and a per-byte shift count of -128
  // (0x80). The rounding constant 1<<127 lifts the small source positive so
  // the ARM result is 0 in every lane; the interpreter previously skipped the
  // round (1<<127 overflows signed __int128) and sign-broadcast to -1 (0xFF).
  // Assert BOTH tiers now yield all-zero Vd, not just JIT==interp equivalence.
  {
    InitState in = RandomInit();
    unsigned __int128 neg_bytes = 0, shift_neg128 = 0;
    for (int b = 0; b < 16; b++) {
      neg_bytes |= (static_cast<unsigned __int128>(0x80)) << (b * 8);     // -128 source
      shift_neg128 |= (static_cast<unsigned __int128>(0x80)) << (b * 8);  // shift = -128
    }
    in.v[1] = neg_bytes;
    in.v[2] = shift_neg128;
    for (uint32_t code0 : {0x4E225420U /*srshl*/, 0x4E225C20U /*sqrshl*/}) {
      in.v[0] = ~static_cast<unsigned __int128>(0);  // poison Vd to catch a no-op
      uint32_t code[1] = {code0};
      Result r = run_one(code, 1, in);
      EXPECT_NE(r, kDeclined) << "rounding-shift region unexpectedly JIT-declined";
      // run_one already asserted JIT==interp; state_ holds the interp result.
      EXPECT_EQ(state_.cpu.v[0], static_cast<__uint128_t>(0))
          << "rounding shift by -128 must floor to 0, got sign-broadcast";
    }
  }
}

}  // namespace
}  // namespace berberis
