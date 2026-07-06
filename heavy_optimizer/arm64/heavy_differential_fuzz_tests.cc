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

// Institutionalized HEAVY-optimizer-vs-interpreter differential fuzzer
// (region-level).
//
// The sibling lite-tier fuzzer (lite_translator/arm64_to_x86_64/
// differential_fuzz_tests.cc) compares the single-pass lite JIT against the
// per-instruction interpreter and caught four shipped bugs. That harness is
// blind to the second-gear heavy optimizer, whose miscompiles are typically
// *region-structural*: they need multiple instructions, a real register
// mapping, and the liveness/allocation machinery of a full region to surface
// (a single-instruction exec test passes while a multi-instruction region
// diverges). The pairwise SMAXP/SMINP/UMAXP/UMINP deinterleave lowering is the
// motivating example: it passed every per-op host exec test AND the renderer
// gate AND the sample suite, yet deterministically miscompiled a NetEase region
// under two-gear, and was reverted for lack of exactly this tool.
//
// Design mirrors the lite fuzzer so their reproducible failures reproduce
// identically:
//   * A single fixture (Arm64HeavyDifferentialFuzz) runs an arbitrary guest
//     REGION through the heavy optimizer (HeavyOptimizeRegion) and the
//     per-instruction interpreter from IDENTICAL seeded CPUState, then compares
//     the FULL end state (X0-X30, SP, NZCV, V0-V31; FPSR behind a per-class
//     knob).
//   * Deterministic splitmix/LCG PRNG keyed by a fixed seed => reproducible CI
//     failures.  Corpus generators parameterize over destructive register
//     aliasing (rd==rn / rd==rm), high register pressure, lane arrangements and
//     immediate boundary values.
//   * A region is compared only when the heavy tier translated it WHOLE (ok and
//     the region stop reached end_pc): a heavy bail falls back to lite/interp
//     on-device and cannot be a heavy miscompile source, so it is skipped
//     (kDeclined), exactly as the lite harness skips a lite decline.
//   * CI mode (default): fixed seeds, bounded iterations, runs in seconds inside
//     berberis_arm64_host_tests.  Exhaustive mode
//     (BERBERIS_DIFFERENTIAL_FUZZ_EXHAUSTIVE) widens the sweep.
//
// Unlike the lite harness's single-instruction SIMD regions (a lite SIMD/FP
// handler's codegen is fully captured by one instruction), THIS harness leans on
// MULTI-instruction regions: that is where a heavy destructive-lowering clobber
// (a value the register mapping still needs, overwritten by an in-place x86 op)
// or a deinterleave/back-edge structural bug becomes observable.

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/heavy_optimizer/arm64/heavy_optimize_region.h"
#include "berberis/interpreter/arm64/interpreter.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_exec_region.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

// ARM64 NZCV lives in CPUState.flags at N@15 Z@14 C@8 V@0.
constexpr uint16_t kNZCVMask = 0xC101;

// Widen the sweep when the exhaustive-mode env knob is set (shared with the lite
// fuzzer's digitalis/scripts/differential-fuzz.sh). Off => 1 (CI, ~seconds).
int FuzzScale() {
  const char* e = getenv("BERBERIS_DIFFERENTIAL_FUZZ_EXHAUSTIVE");
  if (e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0')) {
    return 25;
  }
  return 1;
}

class Arm64HeavyDifferentialFuzz : public ::testing::Test {
 protected:
  ThreadState state_{};
  uint64_t seed_ = 0;

  void Seed(uint64_t s) { seed_ = s; }
  // splitmix / LCG hybrid: same recurrence the lite fuzzer uses.
  uint64_t Rnd() {
    seed_ = seed_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed_ >> 33;  // ~31 usable bits
  }
  uint64_t Rnd64() { return ((Rnd() & 0xffffffffULL) << 32) | (Rnd() & 0xffffffffULL); }

  struct FullState {
    uint64_t x[31];
    uint64_t sp;
    uint16_t flags;
    uint32_t emulated_fpsr;
    unsigned __int128 v[32];
  };

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

  // Randomized init: registers get full 64-bit values, a fraction pinned to the
  // arithmetic edge values (0, -1, INT_MIN, INT_MAX); vector lanes get mixed
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

  // Runs region [code, code+n) through the HEAVY optimizer and the interpreter
  // from `in`; compares full state. Returns kDeclined when the heavy tier did
  // not translate the region WHOLE (a bail: on-device it falls back to lite /
  // interp, so it cannot be a heavy miscompile source).
  Result RunDifferential(const uint32_t* code,
                         int n,
                         const InitState& in,
                         bool compare_fpsr,
                         std::string* desc) {
    GuestAddr start = ToGuestAddr(code);
    GuestAddr code_end = start + static_cast<GuestAddr>(n) * 4;

    // Heavy optimizer. Returns {stop_pc, success, number_of_instructions}.
    ApplyInit(in);
    state_.cpu.insn_addr = start;
    MachineCode mc;
    auto [stop, ok, num] =
        HeavyOptimizeRegion(start, &mc, HeavyOptimizeParams{.end_pc = code_end});
    // Only compare a region the heavy tier translated in full: whole region
    // consumed (stop == code_end), success flagged, and every instruction
    // translated. A partial/bailed region runs on lite/interp on-device.
    if (!ok || stop != code_end || num != static_cast<size_t>(n)) {
      return kDeclined;
    }

    // The TranslationCache is a process-global singleton shared across tests; a
    // prior test may have left a stale entry colliding with this PC window.
    // Clear it so the dispatcher does not reach a stale entry at `stop`.
    TranslationCache::GetInstance()->InvalidateGuestRange(start, code_end + 4);

    ScopedExecRegion exec(&mc);
    TestingRunGeneratedCode(&state_, exec.get(), stop);
    FullState heavy = Capture();

    // Interpreter to the same stop PC.
    ApplyInit(in);
    state_.cpu.insn_addr = start;
    int guard = 0;
    while (state_.cpu.insn_addr >= start && state_.cpu.insn_addr < stop && guard++ < 256) {
      InterpretInsn(&state_);
    }
    FullState itp = Capture();

    return Compare(heavy, itp, compare_fpsr, code, n, desc) ? kMatch : kDiverge;
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
        snprintf(buf, sizeof(buf), "x%d HEAVY=0x%016llx INTERP=0x%016llx region:", i,
                 (unsigned long long)a.x[i], (unsigned long long)b.x[i]);
        *desc = std::string(buf) + RegionStr(code, n);
        return false;
      }
    }
    if (a.sp != b.sp) {
      snprintf(buf, sizeof(buf), "sp HEAVY=0x%016llx INTERP=0x%016llx region:",
               (unsigned long long)a.sp, (unsigned long long)b.sp);
      *desc = std::string(buf) + RegionStr(code, n);
      return false;
    }
    if ((a.flags & kNZCVMask) != (b.flags & kNZCVMask)) {
      snprintf(buf, sizeof(buf), "NZCV HEAVY=0x%04x INTERP=0x%04x region:",
               a.flags & kNZCVMask, b.flags & kNZCVMask);
      *desc = std::string(buf) + RegionStr(code, n);
      return false;
    }
    for (int i = 0; i < 32; i++) {
      if (a.v[i] != b.v[i]) {
        snprintf(buf, sizeof(buf), "v%d HEAVY=0x%016llx:%016llx INTERP=0x%016llx:%016llx region:", i,
                 (unsigned long long)(uint64_t)(a.v[i] >> 64), (unsigned long long)(uint64_t)a.v[i],
                 (unsigned long long)(uint64_t)(b.v[i] >> 64), (unsigned long long)(uint64_t)b.v[i]);
        *desc = std::string(buf) + RegionStr(code, n);
        return false;
      }
    }
    if (compare_fpsr && a.emulated_fpsr != b.emulated_fpsr) {
      snprintf(buf, sizeof(buf), "FPSR HEAVY=0x%08x INTERP=0x%08x region:", a.emulated_fpsr,
               b.emulated_fpsr);
      *desc = std::string(buf) + RegionStr(code, n);
      return false;
    }
    return true;
  }

  // ---- Corpus generators. Registers are drawn from a small pool so
  // destructive aliasing (rd==rn / rd==rm) and mapping spill are naturally
  // sampled. ----

  // Integer data-processing (2-src register forms + variable shifts + div) that
  // the heavy tier lowers. Multi-instruction regions of these exercise the
  // register MAPPING: a non-destructive ARM 3-operand op lowered as a
  // destructive in-place x86 op would clobber a value the mapping still needs.
  uint32_t GenIntDataProc(uint32_t kMaxReg) {
    uint32_t rd = Rnd() % kMaxReg, rn = Rnd() % kMaxReg, rm = Rnd() % kMaxReg;
    uint32_t sf = Rnd() & 1;  // 0 => W (32-bit), 1 => X (64-bit)
    uint32_t sf31 = sf << 31;
    switch (Rnd() % 12) {
      case 0: return sf31 | 0x0B000000U | (rm << 16) | (rn << 5) | rd;  // add
      case 1: return sf31 | 0x4B000000U | (rm << 16) | (rn << 5) | rd;  // sub
      case 2: return sf31 | 0x0A000000U | (rm << 16) | (rn << 5) | rd;  // and
      case 3: return sf31 | 0x2A000000U | (rm << 16) | (rn << 5) | rd;  // orr
      case 4: return sf31 | 0x4A000000U | (rm << 16) | (rn << 5) | rd;  // eor
      case 5: return sf31 | 0x1B007C00U | (rm << 16) | (rn << 5) | rd;  // mul (madd xzr)
      case 6: return sf31 | 0x1AC00C00U | (rm << 16) | (rn << 5) | rd;  // sdiv
      case 7: return sf31 | 0x1AC00800U | (rm << 16) | (rn << 5) | rd;  // udiv
      case 8: return sf31 | 0x1AC02000U | (rm << 16) | (rn << 5) | rd;  // lslv
      case 9: return sf31 | 0x1AC02400U | (rm << 16) | (rn << 5) | rd;  // lsrv
      case 10: return sf31 | 0x1AC02800U | (rm << 16) | (rn << 5) | rd;  // asrv
      default: return sf31 | 0x1AC02C00U | (rm << 16) | (rn << 5) | rd;  // rorv
    }
  }

  // AdvSIMD three-same INTEGER, restricted to the opcodes the HEAVY tier lowers
  // (CMGT/CMGE/SMAX/SMIN/ADD-SUB/CMEQ-CMTST/MLA-MLS/MUL-PMUL). Non-.2D sizes
  // only for the min/max/mul forms that lack a 64-bit-lane SSE op; ADD/SUB allow
  // all sizes. rd may alias rn or rm to sample the destructive-lowering clobber.
  uint32_t GenNeonThreeSame() {
    // opcode, allow_2d
    static const struct {
      uint8_t opcode;
      bool allow_2d;
    } kOpc[] = {
        {0x06, false},  // CMGT
        {0x07, false},  // CMGE
        {0x0C, false},  // SMAX
        {0x0D, false},  // SMIN
        {0x10, true},   // ADD (U=0) / SUB (U=1)
        {0x11, false},  // CMTST (U=0) / CMEQ (U=1)
        {0x12, false},  // MLA (U=1) / MLS ; keep byte/half/word
        {0x13, false},  // MUL (U=0) / PMUL (U=1, size=00)
        {0x14, false},  // SMAXP (U=0) / UMAXP (U=1) ; byte/half/word (.2D bails)
        {0x15, false},  // SMINP (U=0) / UMINP (U=1)
        {0x17, true},   // ADDP (U=0 only); allow .2D (size=11, Q=1)
    };
    const auto& sel = kOpc[Rnd() % (sizeof(kOpc) / sizeof(kOpc[0]))];
    uint32_t q = Rnd() & 1, u = Rnd() & 1;
    uint32_t size = sel.allow_2d ? (Rnd() % 4) : (Rnd() % 3);
    // PMUL (opcode 0x13, U=1) is size=00 only.
    if (sel.opcode == 0x13 && u == 1) size = 0;
    // ADDP (opcode 0x17) is U=0 only; U=1 is decoder-Undefined.
    if (sel.opcode == 0x17) u = 0;
    uint32_t rn = Rnd() % 8, rm = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? rm : (Rnd() % 8));  // alias rn / rm / free
    return (q << 30) | (u << 29) | (0b01110u << 24) | (size << 22) | (1u << 21) |
           (rm << 16) | (static_cast<uint32_t>(sel.opcode) << 11) | (1u << 10) | (rn << 5) | rd;
  }
};

// -------------------------------------------------------------------------
// Class fuzzers. Each seeds deterministically, generates seeded regions and
// asserts HEAVY==interp on the full state. compared>threshold guards against a
// silent "heavy declined everything" regression that would make the test
// vacuous.
// -------------------------------------------------------------------------

// Multi-instruction integer regions: exposes a heavy destructive-lowering
// register-mapping clobber (a value still live in the mapping, overwritten).
TEST_F(Arm64HeavyDifferentialFuzz, IntDataProcRegion) {
  Seed(0x11EA0912345678ABULL);
  const int kIters = 4000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    int n = 2 + (Rnd() % 5);  // 2..6 instructions
    // 13 mapped guest regs => enough to force mapping spills in a small pool.
    uint32_t kMaxReg = 13;
    uint32_t code[6];
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
  EXPECT_GT(compared, 200) << "heavy accepted too few integer regions";
}

// Single-instruction AdvSIMD three-same integer, rd aliasing rn/rm sampled.
TEST_F(Arm64HeavyDifferentialFuzz, NeonThreeSame) {
  Seed(0x3A3E5A3E90ABCDEFULL);
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
  EXPECT_GT(compared, 300) << "heavy accepted too few three-same encodings";
}

// Multi-instruction AdvSIMD three-same regions: the SIMD analogue of the
// integer region test and the exact class a region-structural SIMD miscompile
// (the reverted pairwise deinterleave lowering) lives in. Vector data-flow
// between consecutive NEON ops stresses the heavy SIMD register mapping.
TEST_F(Arm64HeavyDifferentialFuzz, NeonThreeSameRegion) {
  Seed(0x5E0A5E0AFEDCBA98ULL);
  const int kIters = 4000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    int n = 2 + (Rnd() % 3);  // 2..4 instructions
    uint32_t code[4];
    for (int i = 0; i < n; i++) code[i] = GenNeonThreeSame();
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, n, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 100) << "heavy accepted too few three-same regions";
}

// Acceptance: the generators reach the register-aliasing and multi-instruction
// shapes the harness exists to stress, so a future refactor that silently stops
// producing them fails loudly rather than making the fuzzer vacuous.
TEST_F(Arm64HeavyDifferentialFuzz, GeneratorCoverage) {
  bool saw_alias_rd_rn = false, saw_alias_rd_rm = false, saw_add_2d = false;
  bool saw_addp_2d = false, saw_pairwise_minmax = false;
  Seed(0xC0FFEE0011223344ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonThreeSame();
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F, rm = (insn >> 16) & 0x1F;
    uint32_t opcode = (insn >> 11) & 0x1F, size = (insn >> 22) & 3, q = (insn >> 30) & 1;
    if (rd == rn) saw_alias_rd_rn = true;
    if (rd == rm) saw_alias_rd_rm = true;
    if (opcode == 0x10 && size == 3) saw_add_2d = true;  // ADD/SUB .2D
    if (opcode == 0x17 && size == 3 && q == 1) saw_addp_2d = true;  // ADDP .2D
    if (opcode == 0x14 || opcode == 0x15) saw_pairwise_minmax = true;  // S/U MAXP/MINP
  }
  EXPECT_TRUE(saw_alias_rd_rn) << "three-same generator no longer produces rd==rn (clobber class)";
  EXPECT_TRUE(saw_alias_rd_rm) << "three-same generator no longer produces rd==rm (clobber class)";
  EXPECT_TRUE(saw_add_2d) << "three-same generator no longer produces ADD/SUB .2D";
  EXPECT_TRUE(saw_addp_2d) << "three-same generator no longer produces ADDP .2D";
  EXPECT_TRUE(saw_pairwise_minmax) << "three-same generator no longer produces pairwise min/max";

  bool saw_div = false, saw_var_shift = false;
  Seed(0xD00D1E0055667788ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenIntDataProc(13);
    uint32_t masked = insn & 0x7FE0FC00U;  // ignore sf, Rn, Rd
    if (masked == 0x1AC00C00U || masked == 0x1AC00800U) saw_div = true;         // sdiv/udiv
    if (masked == 0x1AC02000U || masked == 0x1AC02400U) saw_var_shift = true;   // lslv/lsrv
  }
  EXPECT_TRUE(saw_div) << "integer generator no longer produces SDIV/UDIV";
  EXPECT_TRUE(saw_var_shift) << "integer generator no longer produces variable shifts";
}

// Regression pin for the store/load-forwarding stale-vreg bug that this harness
// surfaced (fixed in RemoveLocalGuestContextAccesses): a destructive three-same
// op (SUB, `psubd xn, xm`) overwrites the XMM that a prior GetSimd cached for a
// guest V-reg still needed by a later instruction, and the local-guest-context
// optimizer forwarded the stale (mutated) register to the later GetSimd. The
// 2-insn region below deterministically reproduced HEAVY=0 vs INTERP=0xFFFFFFFF
// before the fix.
TEST_F(Arm64HeavyDifferentialFuzz, StaleForwardedVRegRegression) {
  static const uint32_t code[] = {
      0x2ea58405,  // SUB  v5.2s, v0.2s, v5.2s  (destructive: result in v5, clobbers cached v0)
      0x2ea034a0,  // CMHI v0.2s, v5.2s, v0.2s  (reads the still-live original v0)
  };
  InitState in = RandomInit();
  in.v[0] = (unsigned __int128)0x0000000000000005ULL;  // lane0=5
  in.v[5] = (unsigned __int128)0x00000000000000faULL;  // lane0=250
  // SUB -> v5 lane0 = 5-250 = 0xFFFFFF0B; CMHI (v5 >u v0) lane0 = 0xFFFFFFFF.
  std::string desc;
  Result r = RunDifferential(code, 2, in, /*compare_fpsr=*/false, &desc);
  EXPECT_NE(r, kDeclined) << "SUB;CMHI region unexpectedly heavy-declined";
  EXPECT_NE(r, kDiverge) << desc;
  EXPECT_EQ((uint64_t)state_.cpu.v[0], 0x00000000ffffffffULL)
      << "CMHI must read the original v0, not the SUB result";
}

}  // namespace
}  // namespace berberis
