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
        {0x03, true},   // logical group: AND/BIC/ORR/ORN (U=0, size 00/01/10/11)
                        //                EOR/BSL/BIT/BIF (U=1, size 00/01/10/11)
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

  // AdvSIMD vector x indexed element, integer MUL/MLA/MLS by element — the
  // subset the heavy tier lowers (halfword size=01 index 0..7, word size=10
  // index 0..3). rd may alias rn or the indexed Vm to sample destructive
  // clobber. Encoding: 0 Q U 01111 size L M Rm opcode H 0 Rn Rd, with
  //   MUL: U=0 opcode=1000, MLA: U=1 opcode=0000, MLS: U=1 opcode=0100.
  uint32_t GenNeonVecXIdxMul() {
    static const struct {
      uint32_t u;
      uint32_t opc;
    } kOpc[] = {{0, 0b1000}, {1, 0b0000}, {1, 0b0100}};
    const auto& sel = kOpc[Rnd() % 3];
    uint32_t q = Rnd() & 1;
    uint32_t size = 1 + (Rnd() & 1);  // 01 (H) or 10 (S)
    uint32_t vm = Rnd() % 8;          // <16 keeps halfword Rm valid; M=0 for word
    uint32_t rn = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? vm : (Rnd() % 8));
    uint32_t H, L, M, Rm;
    if (size == 1) {  // halfword: index 0..7 = H:L:M, Vm 0..15
      uint32_t index = Rnd() % 8;
      H = index >> 2;
      L = (index >> 1) & 1;
      M = index & 1;
      Rm = vm;
    } else {  // word: index 0..3 = H:L, Vm = M:Rm
      uint32_t index = Rnd() % 4;
      H = index >> 1;
      L = index & 1;
      M = 0;
      Rm = vm;
    }
    return (q << 30) | (sel.u << 29) | (0b01111u << 24) | (size << 22) |
           (L << 21) | (M << 20) | (Rm << 16) | (sel.opc << 12) | (H << 11) |
           (rn << 5) | rd;
  }

  // Widening MUL/MAC by element: SMULL/UMULL/SMLAL/UMLAL/SMLSL/UMLSL.
  //   SMULL U=0 opc=1010 | UMULL U=1 opc=1010
  //   SMLAL U=0 opc=0010 | UMLAL U=1 opc=0010
  //   SMLSL U=0 opc=0110 | UMLSL U=1 opc=0110
  // size 01 (.4h/.8h -> .4s) or 10 (.2s/.4s -> .2d); Q selects the source half.
  uint32_t GenNeonVecXIdxMull() {
    static const struct {
      uint32_t u;
      uint32_t opc;
    } kOpc[] = {{0, 0b1010}, {1, 0b1010}, {0, 0b0010},
                {1, 0b0010}, {0, 0b0110}, {1, 0b0110}};
    const auto& sel = kOpc[Rnd() % 6];
    uint32_t q = Rnd() & 1;
    uint32_t size = 1 + (Rnd() & 1);  // 01 (H) or 10 (S)
    uint32_t vm = Rnd() % 8;          // <16 keeps halfword Rm valid; M=0 for word
    uint32_t rn = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? vm : (Rnd() % 8));
    uint32_t H, L, M, Rm;
    if (size == 1) {  // halfword: index 0..7 = H:L:M, Vm 0..15
      uint32_t index = Rnd() % 8;
      H = index >> 2;
      L = (index >> 1) & 1;
      M = index & 1;
      Rm = vm;
    } else {  // word: index 0..3 = H:L, Vm = M:Rm
      uint32_t index = Rnd() % 4;
      H = index >> 1;
      L = index & 1;
      M = 0;
      Rm = vm;
    }
    return (q << 30) | (sel.u << 29) | (0b01111u << 24) | (size << 22) |
           (L << 21) | (M << 20) | (Rm << 16) | (sel.opc << 12) | (H << 11) |
           (rn << 5) | rd;
  }

  // Saturating doubling widening MUL/MAC by element: SQDMULL/SQDMLAL/SQDMLSL.
  //   SQDMULL opc=1011 | SQDMLAL opc=0011 | SQDMLSL opc=0111 (U=0, signed only)
  // size 01 (.4h/.8h -> .4s) or 10 (.2s/.4s -> .2d); Q selects the source half.
  // Same index/Vm layout as GenNeonVecXIdxMull.
  uint32_t GenNeonVecXIdxSqdmull() {
    static const uint32_t kOpc[] = {0b1011, 0b0011, 0b0111};
    const uint32_t opc = kOpc[Rnd() % 3];
    uint32_t q = Rnd() & 1;
    uint32_t size = 1 + (Rnd() & 1);  // 01 (H) or 10 (S)
    uint32_t vm = Rnd() % 8;          // <16 keeps halfword Rm valid; M=0 for word
    uint32_t rn = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? vm : (Rnd() % 8));
    uint32_t H, L, M, Rm;
    if (size == 1) {  // halfword: index 0..7 = H:L:M, Vm 0..15
      uint32_t index = Rnd() % 8;
      H = index >> 2;
      L = (index >> 1) & 1;
      M = index & 1;
      Rm = vm;
    } else {  // word: index 0..3 = H:L, Vm = M:Rm
      uint32_t index = Rnd() % 4;
      H = index >> 1;
      L = index & 1;
      M = 0;
      Rm = vm;
    }
    return (q << 30) | (0u << 29) | (0b01111u << 24) | (size << 22) |
           (L << 21) | (M << 20) | (Rm << 16) | (opc << 12) | (H << 11) |
           (rn << 5) | rd;
  }

  // SHL Vd.T, Vn.T, #shift (AdvSIMD shift-by-immediate, U=0 opcode=01010).
  // Covers every size class the heavy tier lowers: byte immh=0001 (this
  // cycle's new PSLLW + per-byte-AND arm), half immh=001x, word immh=01xx,
  // double immh=1xxx. shift ∈ [0, esize-1] via immh:immb - esize. rd samples
  // rd==rn to stress destructive clobber.
  uint32_t GenNeonShlByImm() {
    uint32_t q = Rnd() & 1;
    uint32_t esize_sel = Rnd() % 4;  // 0=byte 1=half 2=word 3=double
    uint32_t immh, immb;
    if (esize_sel == 0) {  // byte: esize 8, immh=0001, shift 0..7
      immh = 0b0001;
      immb = Rnd() % 8;
    } else {
      uint32_t esize = 8u << esize_sel;              // 16, 32, 64
      uint32_t immh_immb = esize + (Rnd() % esize);  // [esize, 2*esize-1]
      immh = (immh_immb >> 3) & 0xF;
      immb = immh_immb & 0x7;
    }
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);
    return (q << 30) | (0u << 29) | (0b01111u << 24) | (0u << 23) |
           (immh << 19) | (immb << 16) | (0b01010u << 11) | (1u << 10) |
           (rn << 5) | rd;
  }

  // SUQADD/USQADD Vd.T, Vn.T (AdvSIMD two-register misc, opcode=00011).
  //   SUQADD U=0 (signed sat acc of unsigned) | USQADD U=1 (unsigned sat acc of
  //   signed). size 00 (.8B/.16B), 01 (.4H/.8H), 10 (.2S/.4S) are heavy-lowered;
  //   size 11 (.1D/.2D) bails to lite and is excluded here. rd samples rd==rn to
  //   stress the destructive read-modify-write accumulate.
  uint32_t GenNeonSuqadd() {
    uint32_t u = Rnd() & 1;      // 0=SUQADD, 1=USQADD
    uint32_t size = Rnd() % 3;   // 00 (B), 01 (H), 10 (S)
    uint32_t q = Rnd() & 1;
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);
    return (q << 30) | (u << 29) | (0b01110u << 24) | (size << 22) |
           (0b10000u << 17) | (0b00011u << 12) | (0b10u << 10) | (rn << 5) | rd;
  }

  // AdvSIMD three-same FP fused multiply-accumulate FMLA/FMLS (.2S/.4S FP32,
  // .2D FP64). Encoding: Q 0 01110 <fmls> <double> 1 Rm 11001 1 Rn Rd, where
  // bit23 selects FMLS(1)/FMLA(0) and bit22 selects double(1)/single(0). .1D
  // (double && !Q) is reserved, so Q is forced to 1 for the double form. The
  // FP16 forms live in a different encoding block and are not produced here.
  // rd may alias rn/rm to sample the destructive accumulate. The companion
  // test seeds finite floats so the interpreter reference and x86 FMA agree
  // bit-for-bit (random NaN payloads propagate differently on ARM vs x86 FMA).
  uint32_t GenNeonFmla() {
    uint32_t is_fmls = Rnd() & 1;
    uint32_t is_double = Rnd() & 1;
    uint32_t q = is_double ? 1u : (Rnd() & 1);  // .1D reserved
    uint32_t rn = Rnd() % 8, rm = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? rm : (Rnd() % 8));
    return (q << 30) | (0b01110u << 24) | (is_fmls << 23) | (is_double << 22) |
           (1u << 21) | (rm << 16) | (0b11001u << 11) | (1u << 10) | (rn << 5) | rd;
  }

  // AdvSIMD vector x indexed-element FP FMUL/FMLA/FMLS (.2S/.4S FP32 size=10,
  // .2D FP64 size=11). U=0; opcode FMUL=1001, FMLA=0001, FMLS=0101. FP32 index
  // = H:L (0..3), Vm = M:Rm; FP64 index = H (0..1), L=0, Vm = M:Rm. .2D is
  // reserved with Q=0, so Q is forced to 1 for the double form. FMULX (U=1) and
  // FP16 (size=00) still bail to lite and are not produced here. rd samples
  // rd==rn / rd==Vm to stress the destructive accumulate.
  uint32_t GenNeonVecXIdxFmul() {
    static const uint32_t kOpc[] = {0b1001, 0b0001, 0b0101};  // FMUL, FMLA, FMLS
    const uint32_t opc = kOpc[Rnd() % 3];
    uint32_t is_double = Rnd() & 1;
    uint32_t size = is_double ? 0b11u : 0b10u;
    uint32_t q = is_double ? 1u : (Rnd() & 1);  // .2D reserved with Q=0
    uint32_t vm = Rnd() % 8;                     // M=0, Rm=vm (<8)
    uint32_t rn = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? vm : (Rnd() % 8));
    uint32_t H, L, M = 0, Rm = vm;
    if (is_double) {  // FP64: index 0..1 = H, L must be 0
      uint32_t index = Rnd() % 2;
      H = index;
      L = 0;
    } else {  // FP32: index 0..3 = H:L
      uint32_t index = Rnd() % 4;
      H = index >> 1;
      L = index & 1;
    }
    return (q << 30) | (0u << 29) | (0b01111u << 24) | (size << 22) |
           (L << 21) | (M << 20) | (Rm << 16) | (opc << 12) | (H << 11) |
           (rn << 5) | rd;
  }

  // A random 128-bit V-register value carrying finite (non-NaN, non-inf,
  // small-magnitude) FP lanes so FMA can never manufacture a NaN and the
  // ARM-vs-x86 NaN-propagation divergence never fires. FP32: 4 lanes; FP64: 2.
  unsigned __int128 RandomFiniteFpVReg(bool is_double) {
    auto small_f32 = [&]() -> uint32_t {
      // magnitude < 1024, ~4 fractional bits, random sign — product of two
      // stays < 2^20 so FMA never overflows to inf.
      float f = static_cast<float>(static_cast<int32_t>(Rnd() % 32768) - 16384) /
                16.0f;
      uint32_t bits;
      memcpy(&bits, &f, 4);
      return bits;
    };
    auto small_f64 = [&]() -> uint64_t {
      double d = static_cast<double>(static_cast<int64_t>(Rnd64() % 33554432) -
                                     16777216) /
                 16.0;
      uint64_t bits;
      memcpy(&bits, &d, 8);
      return bits;
    };
    if (is_double) {
      unsigned __int128 lo = small_f64(), hi = small_f64();
      return (hi << 64) | lo;
    }
    uint64_t l0 = small_f32(), l1 = small_f32(), l2 = small_f32(), l3 = small_f32();
    unsigned __int128 lo = (l1 << 32) | l0, hi = (l3 << 32) | l2;
    return (hi << 64) | lo;
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

// Single-instruction AdvSIMD vector x indexed-element MUL/MLA/MLS by element,
// rd aliasing rn / the indexed Vm sampled.
TEST_F(Arm64HeavyDifferentialFuzz, NeonVecXIdxMul) {
  Seed(0x1DCE1DCE13572468ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonVecXIdxMul()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few by-element MUL/MLA/MLS encodings";
}

// Single-instruction AdvSIMD vector x indexed-element widening MUL/MAC by
// element (SMULL/UMULL/SMLAL/UMLAL/SMLSL/UMLSL), rd aliasing rn / the indexed
// Vm sampled. Exercises the PMOVSX/PMOVZX widen + PMULLD (size=01) and
// PMULDQ/PMULUDQ (size=10) heavy lowering, including the Q=1 high-half select.
TEST_F(Arm64HeavyDifferentialFuzz, NeonVecXIdxMull) {
  Seed(0x2EDF2EDF2468ACE0ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonVecXIdxMull()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few widening by-element encodings";
}

// Single-instruction saturating doubling widening by-element:
// SQDMULL/SQDMLAL/SQDMLSL. Exercises the doubling-overflow saturation corner
// (Vn.lane == Vm.lane == INT_MIN) and the SQDMLAL/SQDMLSL signed saturating
// accumulate, at both size=01 (PMOVSXWD+PMULLD, PCMPEQD corner) and size=10
// (PMOVSXDQ+PMULDQ, PCMPEQQ corner + PCMPGTQ accumulate), Q=1 high-half select.
TEST_F(Arm64HeavyDifferentialFuzz, NeonVecXIdxSqdmull) {
  Seed(0x59D115A7C0DE1234ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonVecXIdxSqdmull()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few SQDMULL by-element encodings";
}

// Single-instruction SHL-by-immediate across every size class the heavy tier
// lowers (byte via PSLLW + per-byte AND mask, half/word/double via
// PSLL{W,D,Q}). Byte was the only SHL arm still bailing before this cycle.
TEST_F(Arm64HeavyDifferentialFuzz, NeonShlByImm) {
  Seed(0x5417B00B12345678ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonShlByImm()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few SHL-by-immediate encodings";
}

// Single-instruction SUQADD/USQADD across byte/halfword/word lanes. Exercises
// the mixed-sign saturating accumulate: byte/halfword widen+PACK{US,SS} path and
// the word 64-bit widen + PCMPGTQ-clamp path, including the destructive rd==rn
// accumulate. All three sizes bailed to lite before this cycle.
TEST_F(Arm64HeavyDifferentialFuzz, NeonSuqadd) {
  Seed(0x5A7DFACE0BADCAFEULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonSuqadd()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few SUQADD/USQADD encodings";
}

// FP fused multiply-accumulate FMLA/FMLS (.2S/.4S FP32, .2D FP64). The heavy
// tier now lowers these to x86 FMA3 packed VF(N)MADD231P{S,D} (was: bail to
// lite). Inputs are seeded with finite floats (RandomFiniteFpVReg) so the
// interpreter's fused result and the x86 FMA result agree bit-for-bit — with
// random NaN payloads the ARM-vs-x86 FMA NaN-propagation rules diverge and the
// full-state compare would false-positive. Exercises FMLA/FMLS × single/double
// × Q and the destructive rd==rn / rd==rm accumulate.
TEST_F(Arm64HeavyDifferentialFuzz, NeonFmla) {
  Seed(0xF31AACC00FEEDBADULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t insn = GenNeonFmla();
    uint32_t code[1] = {insn};
    const bool is_double = ((insn >> 22) & 1) != 0;
    InitState in = RandomInit();
    for (int i = 0; i < 32; i++) in.v[i] = RandomFiniteFpVReg(is_double);
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few FMLA/FMLS encodings";
}

// FP by-element FMUL/FMLA/FMLS (.2S/.4S FP32, .2D FP64). The heavy tier now
// broadcasts Vm.lane[index] with PSHUFD and lowers to packed MULP{S,D} (FMUL)
// or x86 FMA3 VF(N)MADD231P{S,D} (FMLA/FMLS, single rounding) — was: bail to
// lite. Inputs are seeded with finite floats (RandomFiniteFpVReg) so the
// interpreter's result and the x86 result agree bit-for-bit (random NaN
// payloads propagate differently on ARM vs x86). Exercises FMUL/FMLA/FMLS ×
// single/double × Q × index and the destructive rd==rn / rd==Vm accumulate.
TEST_F(Arm64HeavyDifferentialFuzz, NeonVecXIdxFmul) {
  Seed(0xF3B1DECC1DE50FF1ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t insn = GenNeonVecXIdxFmul();
    uint32_t code[1] = {insn};
    const bool is_double = ((insn >> 22) & 1) != 0;
    InitState in = RandomInit();
    for (int i = 0; i < 32; i++) in.v[i] = RandomFiniteFpVReg(is_double);
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few by-element FMUL/FMLA/FMLS";
}

// Acceptance: the generators reach the register-aliasing and multi-instruction
// shapes the harness exists to stress, so a future refactor that silently stops
// producing them fails loudly rather than making the fuzzer vacuous.
TEST_F(Arm64HeavyDifferentialFuzz, GeneratorCoverage) {
  bool saw_alias_rd_rn = false, saw_alias_rd_rm = false, saw_add_2d = false;
  bool saw_addp_2d = false, saw_pairwise_minmax = false, saw_plain_minmax = false;
  bool saw_bic = false, saw_orn = false, saw_cmtst = false;
  Seed(0xC0FFEE0011223344ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonThreeSame();
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F, rm = (insn >> 16) & 0x1F;
    uint32_t opcode = (insn >> 11) & 0x1F, size = (insn >> 22) & 3, q = (insn >> 30) & 1;
    uint32_t u = (insn >> 29) & 1;
    if (rd == rn) saw_alias_rd_rn = true;
    if (rd == rm) saw_alias_rd_rm = true;
    if (opcode == 0x10 && size == 3) saw_add_2d = true;  // ADD/SUB .2D
    if (opcode == 0x17 && size == 3 && q == 1) saw_addp_2d = true;  // ADDP .2D
    if (opcode == 0x14 || opcode == 0x15) saw_pairwise_minmax = true;  // S/U MAXP/MINP
    if (opcode == 0x0C || opcode == 0x0D) saw_plain_minmax = true;  // S/U MAX/MIN
    if (opcode == 0x03 && u == 0 && size == 1) saw_bic = true;  // BIC
    if (opcode == 0x03 && u == 0 && size == 3) saw_orn = true;  // ORN
    if (opcode == 0x11 && u == 0) saw_cmtst = true;             // CMTST
  }
  EXPECT_TRUE(saw_alias_rd_rn) << "three-same generator no longer produces rd==rn (clobber class)";
  EXPECT_TRUE(saw_alias_rd_rm) << "three-same generator no longer produces rd==rm (clobber class)";
  EXPECT_TRUE(saw_add_2d) << "three-same generator no longer produces ADD/SUB .2D";
  EXPECT_TRUE(saw_addp_2d) << "three-same generator no longer produces ADDP .2D";
  EXPECT_TRUE(saw_pairwise_minmax) << "three-same generator no longer produces pairwise min/max";
  EXPECT_TRUE(saw_plain_minmax) << "three-same generator no longer produces plain min/max";
  EXPECT_TRUE(saw_bic) << "three-same generator no longer produces BIC";
  EXPECT_TRUE(saw_orn) << "three-same generator no longer produces ORN";
  EXPECT_TRUE(saw_cmtst) << "three-same generator no longer produces CMTST";

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

  bool saw_idx_mul = false, saw_idx_mla = false, saw_idx_mls = false;
  bool saw_idx_half = false, saw_idx_word = false;
  Seed(0xBEEF1DEA0F0F0F0FULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonVecXIdxMul();
    uint32_t u = (insn >> 29) & 1, opcode = (insn >> 12) & 0xF, size = (insn >> 22) & 3;
    if (u == 0 && opcode == 0b1000) saw_idx_mul = true;  // MUL
    if (u == 1 && opcode == 0b0000) saw_idx_mla = true;  // MLA
    if (u == 1 && opcode == 0b0100) saw_idx_mls = true;  // MLS
    if (size == 1) saw_idx_half = true;
    if (size == 2) saw_idx_word = true;
  }
  EXPECT_TRUE(saw_idx_mul) << "by-element generator no longer produces MUL";
  EXPECT_TRUE(saw_idx_mla) << "by-element generator no longer produces MLA";
  EXPECT_TRUE(saw_idx_mls) << "by-element generator no longer produces MLS";
  EXPECT_TRUE(saw_idx_half) << "by-element generator no longer produces halfword";
  EXPECT_TRUE(saw_idx_word) << "by-element generator no longer produces word";

  bool saw_mull_smull = false, saw_mull_umull = false, saw_mull_smlal = false;
  bool saw_mull_umlsl = false, saw_mull_q1 = false, saw_mull_word = false;
  Seed(0xACE0ACE01234FEDCULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonVecXIdxMull();
    uint32_t u = (insn >> 29) & 1, opcode = (insn >> 12) & 0xF;
    uint32_t size = (insn >> 22) & 3, q = (insn >> 30) & 1;
    if (u == 0 && opcode == 0b1010) saw_mull_smull = true;  // SMULL
    if (u == 1 && opcode == 0b1010) saw_mull_umull = true;  // UMULL
    if (u == 0 && opcode == 0b0010) saw_mull_smlal = true;  // SMLAL
    if (u == 1 && opcode == 0b0110) saw_mull_umlsl = true;  // UMLSL
    if (q == 1) saw_mull_q1 = true;                         // *2 high-half select
    if (size == 2) saw_mull_word = true;                    // word source -> .2d
  }
  EXPECT_TRUE(saw_mull_smull) << "widening generator no longer produces SMULL";
  EXPECT_TRUE(saw_mull_umull) << "widening generator no longer produces UMULL";
  EXPECT_TRUE(saw_mull_smlal) << "widening generator no longer produces SMLAL";
  EXPECT_TRUE(saw_mull_umlsl) << "widening generator no longer produces UMLSL";
  EXPECT_TRUE(saw_mull_q1) << "widening generator no longer produces the *2 high-half select";
  EXPECT_TRUE(saw_mull_word) << "widening generator no longer produces word sources";

  bool saw_sq_mull = false, saw_sq_mlal = false, saw_sq_mlsl = false;
  bool saw_sq_q1 = false, saw_sq_word = false;
  Seed(0x59D115A700FEDCBAULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonVecXIdxSqdmull();
    uint32_t opcode = (insn >> 12) & 0xF;
    uint32_t size = (insn >> 22) & 3, q = (insn >> 30) & 1;
    if (opcode == 0b1011) saw_sq_mull = true;  // SQDMULL
    if (opcode == 0b0011) saw_sq_mlal = true;  // SQDMLAL
    if (opcode == 0b0111) saw_sq_mlsl = true;  // SQDMLSL
    if (q == 1) saw_sq_q1 = true;              // *2 high-half select
    if (size == 2) saw_sq_word = true;         // word source -> .2d
  }
  EXPECT_TRUE(saw_sq_mull) << "SQDMULL generator no longer produces SQDMULL";
  EXPECT_TRUE(saw_sq_mlal) << "SQDMULL generator no longer produces SQDMLAL";
  EXPECT_TRUE(saw_sq_mlsl) << "SQDMULL generator no longer produces SQDMLSL";
  EXPECT_TRUE(saw_sq_q1) << "SQDMULL generator no longer produces the *2 high-half select";
  EXPECT_TRUE(saw_sq_word) << "SQDMULL generator no longer produces word sources";

  bool saw_shl_byte = false, saw_shl_half = false, saw_shl_word = false;
  bool saw_shl_dbl = false, saw_shl_alias = false;
  Seed(0x5417C0DE0BADF00DULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonShlByImm();
    uint32_t immh = (insn >> 19) & 0xF;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (immh == 0b0001) saw_shl_byte = true;               // .8B/.16B (new arm)
    else if ((immh & 0b1110) == 0b0010) saw_shl_half = true;  // 0010/0011
    else if ((immh & 0b1100) == 0b0100) saw_shl_word = true;  // 0100..0111
    else if (immh & 0b1000) saw_shl_dbl = true;              // 1xxx
    if (rd == rn) saw_shl_alias = true;
  }
  EXPECT_TRUE(saw_shl_byte) << "SHL generator no longer produces byte lanes";
  EXPECT_TRUE(saw_shl_half) << "SHL generator no longer produces halfword lanes";
  EXPECT_TRUE(saw_shl_word) << "SHL generator no longer produces word lanes";
  EXPECT_TRUE(saw_shl_dbl) << "SHL generator no longer produces doubleword lanes";
  EXPECT_TRUE(saw_shl_alias) << "SHL generator no longer produces rd==rn (clobber class)";

  bool saw_suqadd = false, saw_usqadd = false, saw_sq_byte = false;
  bool saw_sq_half = false, saw_sq_word2 = false, saw_sq_alias = false;
  Seed(0x5A7DADD00FF1CEE5ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonSuqadd();
    uint32_t u = (insn >> 29) & 1, size = (insn >> 22) & 3;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (u == 0) saw_suqadd = true;  // SUQADD
    if (u == 1) saw_usqadd = true;  // USQADD
    if (size == 0) saw_sq_byte = true;
    if (size == 1) saw_sq_half = true;
    if (size == 2) saw_sq_word2 = true;
    if (rd == rn) saw_sq_alias = true;
  }
  EXPECT_TRUE(saw_suqadd) << "sat-accumulate generator no longer produces SUQADD";
  EXPECT_TRUE(saw_usqadd) << "sat-accumulate generator no longer produces USQADD";
  EXPECT_TRUE(saw_sq_byte) << "sat-accumulate generator no longer produces byte lanes";
  EXPECT_TRUE(saw_sq_half) << "sat-accumulate generator no longer produces halfword lanes";
  EXPECT_TRUE(saw_sq_word2) << "sat-accumulate generator no longer produces word lanes";
  EXPECT_TRUE(saw_sq_alias) << "sat-accumulate generator no longer produces rd==rn (accumulate clobber)";

  bool saw_fmla = false, saw_fmls = false, saw_fma_single = false;
  bool saw_fma_double = false, saw_fma_alias = false;
  Seed(0x0FAA110022003300ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonFmla();
    uint32_t is_fmls = (insn >> 23) & 1, is_double = (insn >> 22) & 1;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (!is_fmls) saw_fmla = true;
    if (is_fmls) saw_fmls = true;
    if (!is_double) saw_fma_single = true;
    if (is_double) saw_fma_double = true;
    if (rd == rn) saw_fma_alias = true;
  }
  EXPECT_TRUE(saw_fmla) << "FMA generator no longer produces FMLA";
  EXPECT_TRUE(saw_fmls) << "FMA generator no longer produces FMLS";
  EXPECT_TRUE(saw_fma_single) << "FMA generator no longer produces single-precision";
  EXPECT_TRUE(saw_fma_double) << "FMA generator no longer produces double-precision";
  EXPECT_TRUE(saw_fma_alias) << "FMA generator no longer produces rd==rn (accumulate clobber)";

  bool saw_ifmul = false, saw_ifmla = false, saw_ifmls = false;
  bool saw_ifp_single = false, saw_ifp_double = false, saw_ifp_alias = false;
  Seed(0x1DF9C0DE5EEDBEEFULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonVecXIdxFmul();
    uint32_t opcode = (insn >> 12) & 0xF, size = (insn >> 22) & 3;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (opcode == 0b1001) saw_ifmul = true;  // FMUL
    if (opcode == 0b0001) saw_ifmla = true;  // FMLA
    if (opcode == 0b0101) saw_ifmls = true;  // FMLS
    if (size == 2) saw_ifp_single = true;    // FP32
    if (size == 3) saw_ifp_double = true;    // FP64
    if (rd == rn) saw_ifp_alias = true;
  }
  EXPECT_TRUE(saw_ifmul) << "by-element FP generator no longer produces FMUL";
  EXPECT_TRUE(saw_ifmla) << "by-element FP generator no longer produces FMLA";
  EXPECT_TRUE(saw_ifmls) << "by-element FP generator no longer produces FMLS";
  EXPECT_TRUE(saw_ifp_single) << "by-element FP generator no longer produces FP32";
  EXPECT_TRUE(saw_ifp_double) << "by-element FP generator no longer produces FP64";
  EXPECT_TRUE(saw_ifp_alias) << "by-element FP generator no longer produces rd==rn (accumulate clobber)";
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
