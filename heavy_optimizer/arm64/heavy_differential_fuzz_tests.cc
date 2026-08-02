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
#include <vector>

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
constexpr uint16_t kNZCVMask = kFlagsNZCVMask;

// Widen the sweep when the exhaustive-mode env knob is set (shared with the lite
// fuzzer's long-sweep driver script). Off => 1 (CI, ~seconds).
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

  // Byte-lane (.8B/.16B) SLI/SRI (insert) and SRSHR/URSHR/SRSRA/URSRA (rounding
  // right shift + accumulate) shift-by-immediate.  All six now lower in the
  // heavy tier (byte insert via PSLLW/PSRLW + per-byte mask + PANDN/POR; byte
  // rounding via PMOVSX/PMOVZX widen + word-shift round-bit + PACK{SS,US}WB,
  // plus PADDB for the accumulate forms).  immh is forced to byte (0b0001);
  // immb in [0,7] covers SLI shift 0..7 and every right-shift cnt 1..8.  rd
  // samples rd==rn to stress the destructive insert/accumulate.
  uint32_t GenNeonByteShiftInsertRounding() {
    static const struct { uint32_t u; uint32_t opcode; } kForms[] = {
        {1, 0b01010},  // SLI
        {1, 0b01000},  // SRI
        {0, 0b00100},  // SRSHR
        {1, 0b00100},  // URSHR
        {0, 0b00110},  // SRSRA
        {1, 0b00110},  // URSRA
    };
    const auto& f = kForms[Rnd() % 6];
    uint32_t q = Rnd() & 1;
    uint32_t immh = 0b0001;
    uint32_t immb = Rnd() % 8;
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);
    return (q << 30) | (f.u << 29) | (0b01111u << 24) | (0u << 23) |
           (immh << 19) | (immb << 16) | (f.opcode << 11) | (1u << 10) |
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

  // FMULX (by element): U=1, opcode=1001, size=10 (.2S/.4S) or 11 (.2D). Same
  // encoding shape as GenNeonVecXIdxFmul's FMUL arm but with the U bit set. The
  // saturation path only fires on (+-0 * +-inf); the paired test seeds those
  // special lanes explicitly (RandomFmulxFpVReg).
  uint32_t GenNeonVecXIdxFmulx() {
    uint32_t is_double = Rnd() & 1;
    uint32_t size = is_double ? 0b11u : 0b10u;
    uint32_t q = is_double ? 1u : (Rnd() & 1);  // .2D reserved with Q=0
    uint32_t vm = Rnd() % 8;                     // M=0, Rm=vm (<8)
    uint32_t rn = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? vm : (Rnd() % 8));
    uint32_t H, L, M = 0, Rm = vm;
    if (is_double) {  // FP64: index 0..1 = H, L must be 0
      H = Rnd() % 2;
      L = 0;
    } else {  // FP32: index 0..3 = H:L
      uint32_t index = Rnd() % 4;
      H = index >> 1;
      L = index & 1;
    }
    return (q << 30) | (1u << 29) | (0b01111u << 24) | (size << 22) |
           (L << 21) | (M << 20) | (Rm << 16) | (0b1001u << 12) | (H << 11) |
           (rn << 5) | rd;
  }

  // A random 128-bit V-register whose lanes are drawn from {+0, -0, +inf, -inf,
  // small-finite} — but NEVER NaN. This exercises FMULX's (+-0 * +-inf)->+-2.0
  // saturation lanes while keeping the ARM-vs-x86 result bit-identical: the only
  // way a*b produces a NaN from these inputs is exactly the +-0*+-inf case,
  // which both tiers turn into +-2.0, so no random NaN payload ever diverges.
  unsigned __int128 RandomFmulxFpVReg(bool is_double) {
    auto pick32 = [&]() -> uint32_t {
      switch (Rnd() % 5) {
        case 0: return 0x00000000u;  // +0
        case 1: return 0x80000000u;  // -0
        case 2: return 0x7F800000u;  // +inf
        case 3: return 0xFF800000u;  // -inf
        default: {
          float f = static_cast<float>(static_cast<int32_t>(Rnd() % 32768) - 16384) /
                    16.0f;
          uint32_t bits;
          memcpy(&bits, &f, 4);
          return bits;
        }
      }
    };
    auto pick64 = [&]() -> uint64_t {
      switch (Rnd() % 5) {
        case 0: return 0x0000000000000000ULL;  // +0
        case 1: return 0x8000000000000000ULL;  // -0
        case 2: return 0x7FF0000000000000ULL;  // +inf
        case 3: return 0xFFF0000000000000ULL;  // -inf
        default: {
          double d = static_cast<double>(
                         static_cast<int64_t>(Rnd64() % 33554432) - 16777216) /
                     16.0;
          uint64_t bits;
          memcpy(&bits, &d, 8);
          return bits;
        }
      }
    };
    if (is_double) {
      unsigned __int128 lo = pick64(), hi = pick64();
      return (hi << 64) | lo;
    }
    uint64_t l0 = pick32(), l1 = pick32(), l2 = pick32(), l3 = pick32();
    unsigned __int128 lo = (l1 << 32) | l0, hi = (l3 << 32) | l2;
    return (hi << 64) | lo;
  }

  // AdvSIMD scalar two-reg-misc FCVTAS / FCVTAU (S/D): float -> integer,
  // round-to-nearest ties-away. Encoding:
  //   (1<<30)|(U<<29)|(0b11110<<24)|(size<<22)|(1<<21)|(0b11100<<12)|(0b10<<10)
  //   |(Rn<<5)|Rd,  U: 0=FCVTAS 1=FCVTAU; size: 0=S(FP32) 1=D(FP64).
  // Heavy JITs the FP32 (S) form; the FP64 (D) form bails to lite and is
  // declined by the differential (so ~half the produced insns are declined,
  // which the compared>threshold guard tolerates). rd samples rd==rn.
  uint32_t GenNeonScalarFcvta() {
    uint32_t is_unsigned = Rnd() & 1;   // 0=FCVTAS, 1=FCVTAU
    uint32_t size = Rnd() & 1;          // 0=S (heavy JITs), 1=D (declined)
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);  // sample destructive rd==rn
    return (1u << 30) | (is_unsigned << 29) | (0b11110u << 24) | (size << 22) |
           (1u << 21) | (0b11100u << 12) | (0b10u << 10) | (rn << 5) | rd;
  }

  // A random 128-bit V-register whose FP32 lane 0 (the only lane a scalar
  // FCVTAS/FCVTAU reads) is drawn from the corner set: NaN (-> 0), +-inf,
  // +-2^31 / 2^32 (signed/unsigned overflow saturation), +-N.5 half-integers
  // (the ties-away rounding case), and ordinary fractional finite values. The
  // upper lanes are seeded too but ignored by the scalar op.
  unsigned __int128 RandomFcvtaFpVReg() {
    auto pick32 = [&]() -> uint32_t {
      switch (Rnd() % 9) {
        case 0: return 0x7FC00000u;  // NaN -> 0
        case 1: return 0x7F800000u;  // +inf
        case 2: return 0xFF800000u;  // -inf
        case 3: return 0x4F000000u;  // 2^31 (INT32 positive overflow)
        case 4: return 0x4F800000u;  // 2^32 (UINT32 overflow)
        case 5: return 0xCF000000u;  // -2^31
        case 6: {  // +-N.5 half-integer (ties-away boundary), |N| < 2048
          int32_t n = static_cast<int32_t>(Rnd() % 4096) - 2048;
          float f = static_cast<float>(n) + 0.5f;
          uint32_t bits;
          memcpy(&bits, &f, 4);
          return bits;
        }
        default: {
          float f = static_cast<float>(static_cast<int32_t>(Rnd() % 262144) -
                                       131072) /
                    8.0f;
          uint32_t bits;
          memcpy(&bits, &f, 4);
          return bits;
        }
      }
    };
    uint64_t l0 = pick32(), l1 = pick32(), l2 = pick32(), l3 = pick32();
    unsigned __int128 lo = (l1 << 32) | l0, hi = (l3 << 32) | l2;
    return (hi << 64) | lo;
  }

  // AdvSIMD vector two-reg-misc FRINTN/FRINTM/FRINTP/FRINTZ/FRINTX/FRINTI
  // (FP32 .2S/.4S): round float to integral float with a fixed rounding mode.
  // Encoding:
  //   (Q<<30)|(U<<29)|(0b01110<<24)|(size<<22)|(0b10000<<17)|(opcode<<12)
  //   |(0b10<<10)|(Rn<<5)|Rd
  // Six FP32 variants (verified vs aarch64-linux-gnu-as):
  //   FRINTN U0 op0x18 sz00 (0x4e218820), FRINTM U0 op0x19 sz00 (0x4e219820),
  //   FRINTP U0 op0x18 sz10 (0x4ea18820), FRINTZ U0 op0x19 sz10 (0x4ea19820),
  //   FRINTX U1 op0x19 sz00 (0x6e219820), FRINTI U1 op0x19 sz10 (0x6ea19820).
  // Heavy JITs the FP32 form as a single ROUNDPS; a .2D form (sz low bit=1, which
  // requires Q=1) bails to lite and is declined. rd samples rd==rn; Q samples
  // .2S/.4S.
  uint32_t GenNeonFrintV() {
    struct V { uint32_t u, opcode, size; };
    static const V kVariants[6] = {
        {0, 0x18, 0b00},  // FRINTN (nearest-even)
        {0, 0x19, 0b00},  // FRINTM (toward -inf)
        {0, 0x18, 0b10},  // FRINTP (toward +inf)
        {0, 0x19, 0b10},  // FRINTZ (toward zero)
        {1, 0x19, 0b00},  // FRINTX (use FPCR, signals inexact)
        {1, 0x19, 0b10},  // FRINTI (use FPCR)
    };
    const V& v = kVariants[Rnd() % 6];
    uint32_t size = v.size;
    uint32_t q = Rnd() & 1;
    // ~1 in 5: promote to a .2D form (sz low bit set, Q=1) which heavy declines.
    if ((Rnd() % 5) == 0) { size |= 1u; q = 1; }
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);  // sample destructive rd==rn
    return (q << 30) | (v.u << 29) | (0b01110u << 24) | (size << 22) |
           (0b10000u << 17) | (v.opcode << 12) | (0b10u << 10) | (rn << 5) | rd;
  }

  // Scalar FP data-processing (1 source): FRINT{N,M,P,Z,X,I} (round to integral
  // float) and FCVT between FP32 and FP64. Encoding:
  //   (0b11110<<24)|(ftype<<22)|(1<<21)|(opcode<<15)|(0b10000<<10)|(Rn<<5)|Rd
  // (verified vs aarch64-linux-gnu-as: frintx s1,s1=0x1e274021,
  //  fcvt d0,s0=0x1e22c000, fcvt s0,d0=0x1e624000). Heavy JITs the FRINT set at
  // S/D via ROUNDPS(lane0)/ROUNDSD, and FCVT S<->D via CVTSS2SD/CVTSD2SS. FRINTA
  // (0b001100, ties-to-away, needs the copysign(0.5) sequence) and FSQRT
  // (0b000011, no SQRT op in this tier) bail to lite and are declined. rd samples
  // rd==rn.
  uint32_t GenScalarFrintFcvt() {
    uint32_t opcode, ftype;
    switch (Rnd() % 10) {
      case 0: opcode = 0b001000; ftype = Rnd() & 1; break;  // FRINTN (S or D)
      case 1: opcode = 0b001001; ftype = Rnd() & 1; break;  // FRINTP
      case 2: opcode = 0b001010; ftype = Rnd() & 1; break;  // FRINTM
      case 3: opcode = 0b001011; ftype = Rnd() & 1; break;  // FRINTZ
      case 4: opcode = 0b001110; ftype = Rnd() & 1; break;  // FRINTX
      case 5: opcode = 0b001111; ftype = Rnd() & 1; break;  // FRINTI
      case 6: opcode = 0b000101; ftype = 0b00; break;       // FCVT Dd, Sn (single->double)
      case 7: opcode = 0b000100; ftype = 0b01; break;       // FCVT Sd, Dn (double->single)
      case 8: opcode = 0b001100; ftype = Rnd() & 1; break;  // FRINTA (heavy declines)
      default: opcode = 0b000011; ftype = Rnd() & 1; break; // FSQRT (heavy declines)
    }
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);  // sample destructive rd==rn
    return (0b11110u << 24) | (ftype << 22) | (1u << 21) | (opcode << 15) |
           (0b10000u << 10) | (rn << 5) | rd;
  }

  // FCSEL Sd|Dd, Sn, Sm, cond (scalar FP conditional select):
  //   0001_1110_ftype_1_Rm_cond_11_Rn_Rd  (S base 0x1E200C00, D base 0x1E600C00).
  // A pure bitwise select of the chosen source's low lane, so random (even
  // NaN/Inf) V lanes are safe -- there is no FP arithmetic to diverge on. cond
  // samples the whole 0..15 range (AL/NV are the unconditional select-Vn path).
  // rd aliases rn/rm to sample the destructive in-place form the heavy blend
  // must not corrupt (it mutates the guest-context GET result of Vn in place).
  uint32_t GenFcsel(uint32_t kMaxV) {
    uint32_t ftype = (Rnd() & 1) ? 0b01 : 0b00;  // D or S (FP16/reserved bail)
    uint32_t rn = Rnd() % kMaxV;
    uint32_t rm = Rnd() % kMaxV;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? rm : (Rnd() % kMaxV));
    uint32_t cond = Rnd() & 0xF;
    return (0b11110u << 24) | (ftype << 22) | (1u << 21) | (rm << 16) | (cond << 12) |
           (0b11u << 10) | (rn << 5) | rd;
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

  // AdvSIMD three-same INTEGER across the FULL opcode ladder (0x00..0x17): the
  // halving/rounding-halving add-sub family (SHADD/SRHADD/SHSUB), the saturating
  // add/sub/shift ladder (SQADD/SQSUB/SQSHL/SQRSHL and rounding SRSHL/SQRSHL),
  // the compare family (CMGT/CMGE/CMHI/CMHS/CMTST/CMEQ), min/max and pairwise
  // min/max (SMAX/SMIN/SMAXP/SMINP), logical (AND/BIC/ORR/ORN/EOR/BSL/BIT/BIF),
  // MUL/PMUL/MLA/MLS, SQDMULH/SQRDMULH, SABD/SABA and ADD/SUB/ADDP. The existing
  // GenNeonThreeSame is a narrower curated set of already-lowered opcodes; this
  // sweeps the whole ladder so a heavy miscompile in a less-common opcode (or a
  // heavy accept of a reserved-size form) is caught. rd aliases rn/rm to sample
  // the destructive lowering.
  //
  // FP three-same (opcode & 0b11000 == 0b11000, i.e. FADD/FMAX/FMLA/… at
  // 0x18..0x1F) is EXCLUDED: with random NaN/Inf lanes ARM materializes the
  // default NaN while x86 SSE propagates a host payload -- implementation-defined,
  // not a miscompile (identical rationale to the lite fuzzer's FP exclusion).
  uint32_t GenNeonThreeSameBroad() {
    // opcode; smax = sweep size in [0, smax]. Per-opcode reserved-size rules are
    // applied as fix-ups below (SQDMULH H/S-only, PMUL size=00, ADDP U=0).
    static const struct {
      uint8_t opcode;
      uint8_t smax;
    } kOpc[] = {
        {0x00, 2},  // SHADD/UHADD
        {0x01, 3},  // SQADD/UQADD
        {0x02, 2},  // SRHADD/URHADD
        {0x03, 3},  // logical: AND/BIC/ORR/ORN (U=0), EOR/BSL/BIT/BIF (U=1)
        {0x04, 2},  // SHSUB/UHSUB
        {0x05, 3},  // SQSUB/UQSUB
        {0x06, 3},  // CMGT/CMHI
        {0x07, 3},  // CMGE/CMHS
        {0x08, 3},  // SSHL/USHL
        {0x09, 3},  // SQSHL/UQSHL
        {0x0A, 3},  // SRSHL/URSHL
        {0x0B, 3},  // SQRSHL/UQRSHL
        {0x0C, 2},  // SMAX/UMAX
        {0x0D, 2},  // SMIN/UMIN
        {0x0E, 2},  // SABD/UABD (size=11 reserved)
        {0x0F, 2},  // SABA/UABA (size=11 reserved)
        {0x10, 3},  // ADD/SUB
        {0x11, 3},  // CMTST/CMEQ
        {0x12, 2},  // MLA/MLS
        {0x13, 2},  // MUL (U=0) / PMUL (U=1, size=00)
        {0x14, 2},  // SMAXP/UMAXP
        {0x15, 2},  // SMINP/UMINP
        {0x16, 2},  // SQDMULH/SQRDMULH (size 00/11 reserved -> clamped below)
        {0x17, 3},  // ADDP (U=0 only)
    };
    const auto& sel = kOpc[Rnd() % (sizeof(kOpc) / sizeof(kOpc[0]))];
    uint32_t q = Rnd() & 1, u = Rnd() & 1;
    uint32_t size = Rnd() % (sel.smax + 1);
    if (sel.opcode == 0x16) size = 1 + (Rnd() & 1);  // SQDMULH/SQRDMULH: 01/10 only
    if (sel.opcode == 0x13 && u == 1) size = 0;       // PMUL is size=00 only
    if (sel.opcode == 0x17) u = 0;                    // ADDP is U=0 only
    uint32_t rn = Rnd() % 8, rm = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? rm : (Rnd() % 8));  // alias rn / rm / free
    return (q << 30) | (u << 29) | (0b01110u << 24) | (size << 22) | (1u << 21) |
           (rm << 16) | (static_cast<uint32_t>(sel.opcode) << 11) | (1u << 10) |
           (rn << 5) | rd;
  }

  // AdvSIMD two-register miscellaneous INTEGER forms:
  //   0 Q U 01110 size 10000 opcode 10 Rn Rd  (bit20=0: the two-reg-misc space,
  // not the bit20=1 across-lanes reductions). The heavy tier lowers a broad slice
  // of this class (REV16/32/64, ABS/NEG, CLS/CLZ, CNT/NOT, S/U-ADDLP, S/U-ADALP,
  // SUQADD/USQADD, XTN/SQXTUN/SQXTN/UQXTN, the integer CMxx #0 compares), so the
  // accept rate is high; SQABS/SQNEG/RBIT bail (kDeclined). rd aliases rn ~half
  // the time to sample the destructive in-place lowering.
  //
  // FP two-reg-misc forms are EXCLUDED for the same NaN-payload / rounding-mode
  // reason as the FP three-same class; the FP->int and FRINT forms are pinned by
  // the dedicated GenNeonScalarFcvta / GenNeonFrintV / GenScalarFrintFcvt
  // generators with corner-value seeds.
  uint32_t GenNeonTwoRegMisc() {
    static const struct {
      uint8_t opcode;
      int8_t fixed_u;    // -1 = sweep U
      int8_t smax;       // sweep size in [0, smax]
      int8_t fixed_size; // -1 = sweep
    } kEnc[] = {
        {0b00000, -1, 1, -1},  // REV64 (U=0) / REV32 (U=1); size 00/01
        {0b00001,  0, 0,  0},  // REV16 (U=0, size=00)
        {0b00010, -1, 2, -1},  // SADDLP / UADDLP
        {0b00011, -1, 2, -1},  // SUQADD / USQADD (bit20=0)
        {0b00100, -1, 2, -1},  // CLS / CLZ
        {0b00101,  0, 0,  0},  // CNT  (U=0, size=00)
        {0b00101,  1, 0,  0},  // NOT  (U=1, size=00)
        {0b00101,  1, 1,  1},  // RBIT (U=1, size=01)
        {0b00110, -1, 2, -1},  // SADALP / UADALP
        {0b00111, -1, 3, -1},  // SQABS / SQNEG (.2D=>Q=1)
        {0b01000, -1, 3, -1},  // CMGT #0 / CMGE #0 (integer)
        {0b01001, -1, 3, -1},  // CMEQ #0 / CMLE #0 (integer)
        {0b01010,  0, 3, -1},  // CMLT #0 (U=0 only, integer)
        {0b01011, -1, 3, -1},  // ABS / NEG
        {0b10010, -1, 2, -1},  // XTN / SQXTUN
        {0b10100, -1, 2, -1},  // SQXTN / UQXTN
    };
    const auto& e = kEnc[Rnd() % (sizeof(kEnc) / sizeof(kEnc[0]))];
    uint32_t u = (e.fixed_u < 0) ? (Rnd() & 1) : static_cast<uint32_t>(e.fixed_u);
    uint32_t size = (e.fixed_size < 0) ? (Rnd() % (e.smax + 1))
                                       : static_cast<uint32_t>(e.fixed_size);
    uint32_t q = (size == 3) ? 1u : (Rnd() & 1);  // .2D needs Q=1
    uint32_t rn = Rnd() % 8;
    uint32_t rd = (Rnd() & 1) ? rn : (Rnd() % 8);  // ~50% in-place (rd==rn)
    return (q << 30) | (u << 29) | (0b01110u << 24) | (size << 22) | (1u << 21) |
           (static_cast<uint32_t>(e.opcode) << 12) | (0b10u << 10) | (rn << 5) | rd;
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

// Single-instruction AdvSIMD three-same integer across the FULL opcode ladder
// (0x00..0x17). Broader than NeonThreeSame's curated set: catches a heavy
// miscompile (or a wrong accept of a reserved-size form) anywhere in the ladder.
TEST_F(Arm64HeavyDifferentialFuzz, NeonThreeSameBroad) {
  Seed(0x3B0AD5A311223344ULL);
  const int kIters = 6000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonThreeSameBroad()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 200) << "heavy accepted too few broad three-same encodings";
}

// Multi-instruction broad three-same regions: vector data-flow between
// consecutive ladder ops stresses the heavy SIMD register mapping across the
// full opcode set (the region analogue of NeonThreeSameBroad).
TEST_F(Arm64HeavyDifferentialFuzz, NeonThreeSameBroadRegion) {
  Seed(0x3B0AD5A355667788ULL);
  const int kIters = 4000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    int n = 2 + (Rnd() % 3);  // 2..4 instructions
    uint32_t code[4];
    for (int i = 0; i < n; i++) code[i] = GenNeonThreeSameBroad();
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, n, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 100) << "heavy accepted too few broad three-same regions";
}

// Single-instruction AdvSIMD two-register-misc integer forms. The heavy tier
// lowers a broad slice (REV/ABS/NEG/CLS/CLZ/CNT/NOT/ADDLP/ADALP/SUQADD/XTN/
// SQXTUN/SQXTN/UQXTN/CMxx#0); SQABS/SQNEG/RBIT bail. rd==rn sampled.
TEST_F(Arm64HeavyDifferentialFuzz, NeonTwoRegMisc) {
  Seed(0x2E600F15C0DE1234ULL);
  const int kIters = 6000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonTwoRegMisc()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 200) << "heavy accepted too few two-reg-misc encodings";
}

// Multi-instruction two-reg-misc integer regions: chains ladder ops so a value a
// prior op cached in an XMM is still needed by a later one -- the destructive
// in-place lowering clobber shape, in the two-reg-misc class.
TEST_F(Arm64HeavyDifferentialFuzz, NeonTwoRegMiscRegion) {
  Seed(0x2E600F1555667788ULL);
  const int kIters = 4000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    int n = 2 + (Rnd() % 3);  // 2..4 instructions
    uint32_t code[4];
    for (int i = 0; i < n; i++) code[i] = GenNeonTwoRegMisc();
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, n, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 100) << "heavy accepted too few two-reg-misc regions";
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

// Byte-lane (.8B/.16B) SLI/SRI insert and SRSHR/URSHR/SRSRA/URSRA rounding
// shift-by-immediate, full-state differential vs the interpreter.  These byte
// forms (except URSHR) bailed to lite before this cycle; the heavy tier now
// lowers them.  Includes the destructive rd==rn insert/accumulate.
TEST_F(Arm64HeavyDifferentialFuzz, NeonByteShiftInsertRounding) {
  Seed(0xB17E5417C0FFEE00ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonByteShiftInsertRounding()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few byte insert/rounding shift encodings";
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

// FP by-element FMULX (.2S/.4S FP32, .2D FP64). The heavy tier now lowers FMULX
// as FMUL plus the (+-0 * +-inf)->+-2.0 saturation blend (PSHUFD broadcast,
// MULP{S,D}, CMPUNORDP{S,D} special-lane detection, sign-mask + bits(+2.0)
// select) — was: bail to lite. Inputs are drawn from {+-0, +-inf, small-finite}
// (RandomFmulxFpVReg, never NaN) so the saturation lanes actually fire while the
// ARM-vs-x86 result stays bit-identical. Exercises FMULX × single/double × Q ×
// index and the destructive rd==rn / rd==Vm accumulate.
TEST_F(Arm64HeavyDifferentialFuzz, NeonVecXIdxFmulx) {
  Seed(0x2A17FEEDF00DBA11ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t insn = GenNeonVecXIdxFmulx();
    uint32_t code[1] = {insn};
    const bool is_double = ((insn >> 22) & 1) != 0;
    InitState in = RandomInit();
    for (int i = 0; i < 32; i++) in.v[i] = RandomFmulxFpVReg(is_double);
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few by-element FMULX";
}

// Scalar FCVTAS / FCVTAU (S): float -> integer, round-to-nearest ties-away. The
// heavy tier now lowers the FP32 (S) form as the FRINTA copysign(0.5) trick +
// the FCVTZS (signed) / FCVTZU (unsigned) saturation/NaN fix-up on a
// lane-0-scrubbed source — was: bail to lite. The FP64 (D) form still bails and
// is declined. Inputs (RandomFcvtaFpVReg) seed the corner set — NaN, +-inf,
// signed/unsigned overflow, +-N.5 half-integers (ties-away), finite fractions —
// so the saturation and tie paths actually fire, HEAVY==interp on full state.
TEST_F(Arm64HeavyDifferentialFuzz, NeonScalarFcvta) {
  Seed(0xFC77A5FC77A50011ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonScalarFcvta()};
    InitState in = RandomInit();
    for (int i = 0; i < 32; i++) in.v[i] = RandomFcvtaFpVReg();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few scalar FCVTAS/FCVTAU";
}

// Vector FRINTN/FRINTM/FRINTP/FRINTZ/FRINTX/FRINTI (FP32 .2S/.4S): round float
// to integral float. Heavy now lowers the FP32 form as a single SSE4.1 ROUNDPS
// with the matching rounding imm (no int saturation fix-up — the result stays
// FP) — was: bail to lite. The .2D form bails and is declined. Inputs
// (RandomFcvtaFpVReg) seed every FP32 lane from the corner set — NaN/+-inf
// (propagate unchanged), 2^31/2^32 magnitudes (already integral), +-N.5
// half-integers (the nearest-even boundary for FRINTN), and finite fractions —
// so each rounding mode's boundary fires and HEAVY==interp on full state.
TEST_F(Arm64HeavyDifferentialFuzz, NeonFrintV) {
  Seed(0xF817E5F817E50022ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenNeonFrintV()};
    InitState in = RandomInit();
    for (int i = 0; i < 32; i++) in.v[i] = RandomFcvtaFpVReg();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few vector FRINT";
}

// Scalar FRINT{N,M,P,Z,X,I} (S/D) and FCVT between FP32 and FP64. Heavy now
// lowers the FRINT set as a single ROUNDPS-on-lane-0 (S) / ROUNDSD (D) with the
// matching rounding imm (no int saturation — the result stays FP), and FCVT
// S<->D as one CVTSS2SD / CVTSD2SS into a zeroed dst — was: bail to lite. FRINTA
// (ties-to-away) and FSQRT still bail and are declined. Inputs (RandomFcvtaFp
// VReg) seed every FP32 lane from the corner set — NaN/+-inf (propagate
// unchanged), 2^31/2^32 magnitudes (already integral), +-N.5 half-integers (the
// nearest-even boundary for FRINTN), and finite fractions; the low 64 bits also
// serve as the double operand for the D-form. HEAVY==interp on full state.
TEST_F(Arm64HeavyDifferentialFuzz, ScalarFrintFcvt) {
  Seed(0xF817E5F817E50044ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenScalarFrintFcvt()};
    InitState in = RandomInit();
    for (int i = 0; i < 32; i++) in.v[i] = RandomFcvtaFpVReg();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few scalar FRINT/FCVT";
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

  bool saw_fmulx_u = false, saw_fmulx_single = false, saw_fmulx_double = false;
  bool saw_fmulx_alias = false;
  Seed(0x5A7C0FFEE0DDBA11ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonVecXIdxFmulx();
    uint32_t u = (insn >> 29) & 1, opcode = (insn >> 12) & 0xF, size = (insn >> 22) & 3;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (u == 1 && opcode == 0b1001) saw_fmulx_u = true;  // FMULX (U=1)
    if (size == 2) saw_fmulx_single = true;              // FP32
    if (size == 3) saw_fmulx_double = true;              // FP64
    if (rd == rn) saw_fmulx_alias = true;
  }
  EXPECT_TRUE(saw_fmulx_u) << "by-element FMULX generator no longer sets U=1/opcode=1001";
  EXPECT_TRUE(saw_fmulx_single) << "by-element FMULX generator no longer produces FP32";
  EXPECT_TRUE(saw_fmulx_double) << "by-element FMULX generator no longer produces FP64";
  EXPECT_TRUE(saw_fmulx_alias) << "by-element FMULX generator no longer produces rd==rn (accumulate clobber)";

  bool saw_fcvtas = false, saw_fcvtau = false, saw_fcvta_s = false;
  bool saw_fcvta_d = false, saw_fcvta_alias = false;
  Seed(0xFCA7A5FCA7A50022ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonScalarFcvta();
    uint32_t u = (insn >> 29) & 1, size = (insn >> 22) & 1;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (u == 0) saw_fcvtas = true;   // FCVTAS
    if (u == 1) saw_fcvtau = true;   // FCVTAU
    if (size == 0) saw_fcvta_s = true;   // S (heavy JITs)
    if (size == 1) saw_fcvta_d = true;   // D (declined)
    if (rd == rn) saw_fcvta_alias = true;
  }
  EXPECT_TRUE(saw_fcvtas) << "scalar FCVTA generator no longer produces FCVTAS";
  EXPECT_TRUE(saw_fcvtau) << "scalar FCVTA generator no longer produces FCVTAU";
  EXPECT_TRUE(saw_fcvta_s) << "scalar FCVTA generator no longer produces S (heavy-JIT) form";
  EXPECT_TRUE(saw_fcvta_d) << "scalar FCVTA generator no longer produces D (declined) form";
  EXPECT_TRUE(saw_fcvta_alias) << "scalar FCVTA generator no longer produces rd==rn";

  bool saw_frintn = false, saw_frinti = false, saw_frint_fp32 = false;
  bool saw_frint_2d = false, saw_frint_alias = false;
  Seed(0xF817E5F817E50033ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenNeonFrintV();
    uint32_t u = (insn >> 29) & 1, opcode = (insn >> 12) & 0x1F;
    uint32_t size = (insn >> 22) & 3;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (u == 0 && opcode == 0x18 && (size & 1) == 0) saw_frintn = true;  // FRINTN/P
    if (u == 1 && opcode == 0x19) saw_frinti = true;                     // FRINTX/I
    if ((size & 1) == 0) saw_frint_fp32 = true;   // FP32 (heavy JITs)
    if ((size & 1) == 1) saw_frint_2d = true;     // .2D (declined)
    if (rd == rn) saw_frint_alias = true;
  }
  EXPECT_TRUE(saw_frintn) << "vector FRINT generator no longer produces FRINTN/FRINTP";
  EXPECT_TRUE(saw_frinti) << "vector FRINT generator no longer produces FRINTX/FRINTI";
  EXPECT_TRUE(saw_frint_fp32) << "vector FRINT generator no longer produces FP32 (heavy-JIT) form";
  EXPECT_TRUE(saw_frint_2d) << "vector FRINT generator no longer produces .2D (declined) form";
  EXPECT_TRUE(saw_frint_alias) << "vector FRINT generator no longer produces rd==rn";

  bool saw_sfrint = false, saw_sfcvt_sd = false, saw_sfcvt_ds = false;
  bool saw_sfrint_d = false, saw_sdeclined = false, saw_sfrint_alias = false;
  Seed(0xF817E5F817E50055ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenScalarFrintFcvt();
    uint32_t ftype = (insn >> 22) & 3, opcode = (insn >> 15) & 0x3F;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (opcode >= 0b001000 && opcode <= 0b001111 && opcode != 0b001100) saw_sfrint = true;  // FRINT{N,M,P,Z,X,I}
    if (opcode == 0b000101 && ftype == 0b00) saw_sfcvt_sd = true;  // FCVT Dd,Sn
    if (opcode == 0b000100 && ftype == 0b01) saw_sfcvt_ds = true;  // FCVT Sd,Dn
    if (opcode >= 0b001000 && opcode <= 0b001111 && opcode != 0b001100 && ftype == 0b01) saw_sfrint_d = true;  // D-form FRINT
    if (opcode == 0b001100 || opcode == 0b000011) saw_sdeclined = true;  // FRINTA/FSQRT declined
    if (rd == rn) saw_sfrint_alias = true;
  }
  EXPECT_TRUE(saw_sfrint) << "scalar FRINT/FCVT generator no longer produces FRINT{N..I}";
  EXPECT_TRUE(saw_sfcvt_sd) << "scalar FRINT/FCVT generator no longer produces FCVT Dd,Sn";
  EXPECT_TRUE(saw_sfcvt_ds) << "scalar FRINT/FCVT generator no longer produces FCVT Sd,Dn";
  EXPECT_TRUE(saw_sfrint_d) << "scalar FRINT/FCVT generator no longer produces the D-form (ROUNDSD) path";
  EXPECT_TRUE(saw_sdeclined) << "scalar FRINT/FCVT generator no longer produces the FRINTA/FSQRT declined forms";
  EXPECT_TRUE(saw_sfrint_alias) << "scalar FRINT/FCVT generator no longer produces rd==rn";

  bool bd_shadd = false, bd_sat = false, bd_shift = false, bd_minmax = false;
  bool bd_addp_2d = false, bd_pmul = false, bd_dmulh_hs = false, bd_alias = false;
  Seed(0x3B0AD5A300FEDCBAULL);
  for (int i = 0; i < 60000; i++) {
    uint32_t insn = GenNeonThreeSameBroad();
    uint32_t opcode = (insn >> 11) & 0x1F, u = (insn >> 29) & 1;
    uint32_t size = (insn >> 22) & 3, q = (insn >> 30) & 1;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F, rm = (insn >> 16) & 0x1F;
    if (opcode == 0x00) bd_shadd = true;                       // SHADD/UHADD
    if (opcode == 0x01 || opcode == 0x05) bd_sat = true;       // SQADD/SQSUB
    if (opcode >= 0x08 && opcode <= 0x0B) bd_shift = true;     // S/U SHL ladder
    if (opcode == 0x0C || opcode == 0x0D) bd_minmax = true;    // SMAX/SMIN
    if (opcode == 0x17 && u == 0 && size == 3 && q == 1) bd_addp_2d = true;  // ADDP .2D
    if (opcode == 0x13 && u == 1 && size == 0) bd_pmul = true; // PMUL
    if (opcode == 0x16 && (size == 1 || size == 2)) bd_dmulh_hs = true;  // SQDMULH H/S
    if (rd == rn || rd == rm) bd_alias = true;
  }
  EXPECT_TRUE(bd_shadd) << "broad three-same generator no longer produces SHADD/UHADD";
  EXPECT_TRUE(bd_sat) << "broad three-same generator no longer produces SQADD/SQSUB";
  EXPECT_TRUE(bd_shift) << "broad three-same generator no longer produces the S/U SHL ladder";
  EXPECT_TRUE(bd_minmax) << "broad three-same generator no longer produces SMAX/SMIN";
  EXPECT_TRUE(bd_addp_2d) << "broad three-same generator no longer produces ADDP .2D";
  EXPECT_TRUE(bd_pmul) << "broad three-same generator no longer produces PMUL (size=00)";
  EXPECT_TRUE(bd_dmulh_hs) << "broad three-same generator no longer produces SQDMULH H/S";
  EXPECT_TRUE(bd_alias) << "broad three-same generator no longer produces rd==rn/rd==rm";

  bool trm_cmeqz = false, trm_rev = false, trm_abs = false, trm_narrow = false;
  bool trm_2d = false, trm_alias = false;
  Seed(0x2E600F1500C0FFEEULL);
  for (int i = 0; i < 60000; i++) {
    uint32_t insn = GenNeonTwoRegMisc();
    uint32_t opcode = (insn >> 12) & 0x1F, u = (insn >> 29) & 1;
    uint32_t size = (insn >> 22) & 3, q = (insn >> 30) & 1;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (opcode == 0b01001 && u == 0) trm_cmeqz = true;         // CMEQ #0 (integer)
    if (opcode == 0b00000) trm_rev = true;                     // REV64/REV32
    if (opcode == 0b01011) trm_abs = true;                     // ABS/NEG
    if (opcode == 0b10010 || opcode == 0b10100) trm_narrow = true;  // XTN/SQXTN/UQXTN
    if (size == 3 && q == 1) trm_2d = true;                    // .2D form
    if (rd == rn) trm_alias = true;
  }
  EXPECT_TRUE(trm_cmeqz) << "two-reg-misc generator no longer produces CMEQ #0";
  EXPECT_TRUE(trm_rev) << "two-reg-misc generator no longer produces REV64/REV32";
  EXPECT_TRUE(trm_abs) << "two-reg-misc generator no longer produces ABS/NEG";
  EXPECT_TRUE(trm_narrow) << "two-reg-misc generator no longer produces the narrowing forms";
  EXPECT_TRUE(trm_2d) << "two-reg-misc generator no longer produces the .2D form";
  EXPECT_TRUE(trm_alias) << "two-reg-misc generator no longer produces rd==rn";
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

// Single FCSEL: sanity that the heavy tier accepts scalar FP conditional select
// and matches the interpreter for a random NZCV / random V-register state.
TEST_F(Arm64HeavyDifferentialFuzz, FcselSingle) {
  Seed(0xFC5E10001111ABCDULL);
  const int kIters = 4000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenFcsel(8)};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) ADD_FAILURE() << "iter " << iter << " " << desc;
  }
  EXPECT_GT(compared, 300) << "heavy accepted too few FCSEL encodings";
}

// Pure-FCSEL regions (2..6 selects over V0..V7 with rd aliasing rn/rm). This is
// the region-structural analogue that reproduced the original miscompile: the
// blend's destructive ops overwrite a GET'd guest V-reg still read by a later
// select, so a stale-forwarded vreg diverges heavy from interp. The private-temp
// blend keeps the GET results pristine; this pins that they stay so.
TEST_F(Arm64HeavyDifferentialFuzz, FcselRegion) {
  Seed(0xFC5E12340000BEEFULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    int n = 2 + (Rnd() % 5);  // 2..6 selects
    uint32_t code[6];
    for (int i = 0; i < n; i++) code[i] = GenFcsel(8);
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, n, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) ADD_FAILURE() << "iter " << iter << " " << desc;
  }
  EXPECT_GT(compared, 200) << "heavy accepted too few FCSEL regions";
}

// FCSEL interleaved with integer data-proc (flag-setters + 13 mapped GP regs =
// register pressure) and scalar FRINT/FCVT (FP temps). Exercises the "register
// pressure / instruction interleaving / flags liveness" region interactions the
// original bail commit blamed, cross-checking the select against the interpreter
// in the presence of surrounding NZCV writers and spill-forcing pressure.
TEST_F(Arm64HeavyDifferentialFuzz, FcselMixedRegion) {
  Seed(0xFC5E5A5A99887766ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    int n = 3 + (Rnd() % 4);  // 3..6 instructions
    uint32_t code[6];
    for (int i = 0; i < n; i++) {
      switch (Rnd() % 3) {
        case 0: code[i] = GenFcsel(8); break;
        case 1: code[i] = GenIntDataProc(13); break;  // flags + GP pressure
        default: code[i] = GenScalarFrintFcvt(); break;
      }
    }
    // Guarantee at least one FCSEL so the region is on-topic.
    code[Rnd() % n] = GenFcsel(8);
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, n, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) ADD_FAILURE() << "iter " << iter << " " << desc;
  }
  EXPECT_GT(compared, 100) << "heavy accepted too few mixed FCSEL regions";
}

// Replay of a real Blowfish (bcrypt) round chain: the unrolled
//   lsr/ubfiz -> and #0x3fc -> ldr [base, index] -> add/eor
// pattern an ARM64 bcrypt implementation emits. Every S-box index is masked to
// <= 0x3fc inside the region, so pointing the four table bases and SP at one
// scratch buffer keeps all loads in bounds regardless of the incoming state.
TEST_F(Arm64HeavyDifferentialFuzz, BlowfishRoundChain) {
  static const uint32_t code[] = {
      0x53067dd1, 0xd37e1dc0, 0x4a0e002e, 0xb950a3e1, 0x927e1def, 0x927e1e10,
      0x927e1e31, 0xb86f694f, 0xb8706990, 0xb8716931, 0xb8606900, 0x0b0f020f,
      0x4a1101ef, 0x0b0001ef, 0x4a0f016b, 0x530e7d6f, 0x53167d70, 0x53067d71,
      0xd37e1d60, 0x4a0b002b, 0xb950a7e1, 0x927e1def, 0x927e1e10, 0x927e1e31,
      0xb86f694f, 0xb8706990, 0xb8716931, 0xb8606900, 0x0b0f020f, 0x4a1101ef,
      0x0b0001ef, 0x4a0f01ce, 0x530e7dcf, 0x53167dd0, 0x53067dd1, 0xd37e1dc0,
      0x4a0e002e, 0xb950abe1, 0x927e1def, 0x927e1e10, 0x927e1e31, 0xb86f694f,
      0xb8706990, 0xb8716931, 0xb8606900, 0x0b0f020f, 0x4a1101ef, 0x0b0001ef,
  };
  constexpr int kInsns = sizeof(code) / sizeof(code[0]);
  // Scratch table: large enough for the SP-relative P-box loads (max +0x10a8).
  static uint64_t table[0x4000 / 8];
  Seed(0xB10F15400DDEEFFULL);
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
    table[i] = Rnd64();
  }
  auto base = static_cast<uint64_t>(ToGuestAddr(table));

  int compared = 0;
  for (int iter = 0; iter < 200 * FuzzScale(); iter++) {
    InitState in = RandomInit();
    // S-box bases and the P-box frame all point into the scratch table.
    in.x[8] = in.x[9] = in.x[10] = in.x[12] = base;
    in.sp = base;
    std::string desc;
    Result r = RunDifferential(code, kInsns, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) ADD_FAILURE() << "iter " << iter << " " << desc;
  }
  EXPECT_GT(compared, 0) << "heavy declined every Blowfish round region";
}

// NZCV must survive a long run of ARM64 instructions that do not set flags but
// whose x86 counterparts clobber EFLAGS. The bcrypt key-expansion loop sets
// flags with a CMP at the loop top and consumes them with a conditional branch
// ~280 instructions later, so the flag state has to stay live across the whole
// body.
TEST_F(Arm64HeavyDifferentialFuzz, NzcvSurvivesLongFlagClobberRun) {
  Seed(0x2C0FFEE5A17ULL);
  static uint64_t table[0x1000 / 8];
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
    table[i] = Rnd64();
  }
  auto base = static_cast<uint64_t>(ToGuestAddr(table));

  for (int run_len : {8, 32, 96, 190}) {
    std::vector<uint32_t> code;
    // CMP x15, #0xff4  (sets NZCV)
    code.push_back(0xf13fd1ffu);
    // Filler that must not disturb NZCV: EOR/ADD (w-form) and an S-box load.
    for (int i = 0; i < run_len; i++) {
      switch (i % 4) {
        case 0: code.push_back(0x4a1101efu); break;  // eor w15, w15, w17
        case 1: code.push_back(0x0b0f020fu); break;  // add w15, w16, w15
        case 2: code.push_back(0x927e1e10u); break;  // and x16, x16, #0x3fc
        default: code.push_back(0xb8706950u); break;  // ldr w16, [x10, x16]
      }
    }
    // CSET w0, lo  -- consumes the NZCV set by the CMP above.
    code.push_back(0x1a9f27e0u);

    int compared = 0;
    for (int iter = 0; iter < 60; iter++) {
      InitState in = RandomInit();
      in.x[10] = base;
      in.x[16] = Rnd64() & 0x3fc;
      std::string desc;
      Result r = RunDifferential(
          code.data(), static_cast<int>(code.size()), in, /*compare_fpsr=*/false, &desc);
      if (r == kDeclined) continue;
      compared++;
      if (r == kDiverge) ADD_FAILURE() << "run_len " << run_len << " iter " << iter << " " << desc;
    }
    EXPECT_GT(compared, 0) << "heavy declined every region at run_len " << run_len;
  }
}


// Replay of the two REAL bcrypt key-expansion regions from a shipping ARM64
// libauth.so. The guest loop body is 281 instructions, above the heavy tier's
// 200-instruction region cap, so it splits: the CMP that sets NZCV sits at the
// top of the first region and the conditional branch that consumes it sits at
// the end of the second. Every S-box index is masked to <= 0x3fc in-region, so
// pointing the table bases and SP at one scratch buffer keeps loads in bounds.
TEST_F(Arm64HeavyDifferentialFuzz, BcryptKeyExpansionRealRegionA) {
  static const uint32_t code[] = {
      0xb9506bec, 0xb9506fe2, 0xf13fd1ff, 0x4a10018c, 0x4a0b004b, 0xb95073e2,
      0x530e7d90, 0x53167d91, 0x53067d80, 0xd37e1d81, 0x4a0c004c, 0xb95077e2,
      0x927e1e10, 0x927e1e31, 0x927e1c00, 0xb8706950, 0xb87169b1, 0xb8606920,
      0xb8616901, 0x0b100230, 0x4a000210, 0x0b010210, 0x4a10016b, 0x530e7d70,
      0x53167d71, 0x53067d60, 0xd37e1d61, 0x4a0b004b, 0xb9507be2, 0x927e1e10,
      0x927e1e31, 0x927e1c00, 0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901,
      0x0b100230, 0x4a000210, 0x0b010210, 0x4a10018c, 0x530e7d90, 0x53167d91,
      0x53067d80, 0xd37e1d81, 0x4a0c004c, 0xb9507fe2, 0x927e1e10, 0x927e1e31,
      0x927e1c00, 0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901, 0x0b100230,
      0x4a000210, 0x0b010210, 0x4a10016b, 0x530e7d70, 0x53167d71, 0x53067d60,
      0xd37e1d61, 0x4a0b004b, 0xb95083e2, 0x927e1e10, 0x927e1e31, 0x927e1c00,
      0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901, 0x0b100230, 0x4a000210,
      0x0b010210, 0x4a10018c, 0x530e7d90, 0x53167d91, 0x53067d80, 0xd37e1d81,
      0x4a0c004c, 0xb95087e2, 0x927e1e10, 0x927e1e31, 0x927e1c00, 0xb8706950,
      0xb87169b1, 0xb8606920, 0xb8616901, 0x0b100230, 0x4a000210, 0x0b010210,
      0x4a10016b, 0x530e7d70, 0x53167d71, 0x53067d60, 0xd37e1d61, 0x4a0b004b,
      0xb9508be2, 0x927e1e10, 0x927e1e31, 0x927e1c00, 0xb8706950, 0xb87169b1,
      0xb8606920, 0xb8616901, 0x0b100230, 0x4a000210, 0x0b010210, 0x4a10018c,
      0x530e7d90, 0x53167d91, 0x53067d80, 0xd37e1d81, 0x4a0c004c, 0xb9508fe2,
      0x927e1e10, 0x927e1e31, 0x927e1c00, 0xb8706950, 0xb87169b1, 0xb8606920,
      0xb8616901, 0x0b100230, 0x4a000210, 0x0b010210, 0x4a10016b, 0x530e7d70,
      0x53167d71, 0x53067d60, 0xd37e1d61, 0x4a0b004b, 0xb95093e2, 0x927e1e10,
      0x927e1e31, 0x927e1c00, 0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901,
      0x0b100230, 0x4a000210, 0x0b010210, 0x4a10018c, 0x530e7d90, 0x53167d91,
      0x53067d80, 0xd37e1d81, 0x4a0c004c, 0xb95097e2, 0x927e1e10, 0x927e1e31,
      0x927e1c00, 0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901, 0x0b100230,
      0x4a000210, 0x0b010210, 0x4a10016b, 0x530e7d70, 0x53167d71, 0x53067d60,
      0xd37e1d61, 0x4a0b004b, 0xb9509be2, 0x927e1e10, 0x927e1e31, 0x927e1c00,
      0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901, 0x0b100230, 0x4a000210,
      0x0b010210, 0x4a10018c, 0x530e7d90, 0x53167d91, 0x53067d80, 0xd37e1d81,
      0x4a0c004c, 0xb9509fe2, 0x927e1e10, 0x927e1e31, 0x927e1c00, 0xb8706950,
      0xb87169b1, 0xb8606920, 0xb8616901, 0x0b100230, 0x4a000210, 0x0b010210,
      0x4a10016b, 0x530e7d70, 0x53167d71, 0x53067d60, 0xd37e1d61, 0x4a0b004b,
      0xb950a3e2, 0x927e1e10,
  };
  constexpr int kInsns = sizeof(code) / sizeof(code[0]);
  static uint64_t table[0x4000 / 8];
  Seed(0xBCB17A5E0001ULL);
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) table[i] = Rnd64();
  auto base = static_cast<uint64_t>(ToGuestAddr(table));

  int compared = 0;
  for (int iter = 0; iter < 100; iter++) {
    InitState in = RandomInit();
    in.x[8] = in.x[9] = in.x[10] = in.x[13] = base;
    in.sp = base;
    in.x[15] = (iter % 2) ? 0xff0 : 0x100;  // straddle the CMP boundary
    std::string desc;
    Result r = RunDifferential(code, kInsns, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) ADD_FAILURE() << "iter " << iter << " " << desc;
  }
  EXPECT_GT(compared, 0) << "heavy declined the real region A";
}




// The bcrypt key-expansion loop writes its results back into the very tables
// the round chain loads from (stp w,w,[x]). A miscompile of that store is
// invisible to a register-only comparison, so this replays the real region that
// ends in the write-back and diffs the scratch table as well as the registers.
TEST_F(Arm64HeavyDifferentialFuzz, BcryptKeyExpansionStoreBackRegion) {
  static const uint32_t code[] = {
      0x927e1e31, 0x927e1c00, 0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901,
      0x0b100230, 0x4a000210, 0x0b010210, 0x4a10018c, 0x530e7d90, 0x53167d91,
      0x53067d80, 0xd37e1d81, 0x4a0c004c, 0xb950a7e2, 0x927e1e10, 0x927e1e31,
      0x927e1c00, 0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901, 0x0b100230,
      0x4a000210, 0x0b010210, 0x4a100170, 0x530e7e0b, 0x53167e11, 0x53067e00,
      0xd37e1e01, 0x4a100050, 0xb950afe2, 0x927e1d6b, 0x927e1e31, 0x927e1c00,
      0xb86b694b, 0xb87169b1, 0xb8606920, 0xb8616901, 0x0b0b022b, 0x4a00016b,
      0x0b01016b, 0x4a0b018b, 0x530e7d6c, 0x53167d71, 0x53067d60, 0xd37e1d61,
      0x927e1d8c, 0x927e1e31, 0x927e1c00, 0xb86c694c, 0xb87169b1, 0xb8606920,
      0xb8616901, 0x0b0c022c, 0x4a00018c, 0x0b01018c, 0x4a0c020c, 0x530e7d90,
      0x53167d91, 0x53067d80, 0xd37e1d81, 0x927e1e10, 0x927e1e31, 0x927e1c00,
      0xb8706950, 0xb87169b1, 0xb8606920, 0xb8616901, 0x0b100230, 0xb950abf1,
      0x4a000210, 0x8b0f01a0, 0x910021ef, 0x0b010201, 0x4a0b022b, 0x4a0c0050,
      0x4a01016b, 0x29002c10,
  };
  constexpr int kInsns = sizeof(code) / sizeof(code[0]);
  constexpr size_t kTableWords = 0x4000 / 8;
  static uint64_t table[kTableWords];
  static uint64_t seed_table[kTableWords];
  static uint64_t heavy_table[kTableWords];
  Seed(0xBCB17A5E0003ULL);
  for (size_t i = 0; i < kTableWords; i++) seed_table[i] = Rnd64();
  auto base = static_cast<uint64_t>(ToGuestAddr(table));

  int compared = 0;
  for (int iter = 0; iter < 200; iter++) {
    InitState in = RandomInit();
    in.x[8] = in.x[9] = in.x[10] = in.x[13] = base;
    in.sp = base;
    in.x[15] = (Rnd() % 0x200) * 8;  // store target x0 = x13 + x15, in bounds
    // x16 and x1 are live-in S-box indices: their masking AND lives in the
    // previous region, so bound them here the way the real predecessor does.
    in.x[16] = Rnd() & 0x3fc;
    in.x[1] = Rnd() & 0x3fc;

    GuestAddr start = ToGuestAddr(code);
    GuestAddr code_end = start + static_cast<GuestAddr>(kInsns) * 4;

    memcpy(table, seed_table, sizeof(table));
    ApplyInit(in);
    state_.cpu.insn_addr = start;
    MachineCode mc;
    auto [stop, ok, num] =
        HeavyOptimizeRegion(start, &mc, HeavyOptimizeParams{.end_pc = code_end});
    if (!ok || stop != code_end || num != static_cast<size_t>(kInsns)) continue;
    TranslationCache::GetInstance()->InvalidateGuestRange(start, code_end + 4);
    ScopedExecRegion exec(&mc);
    TestingRunGeneratedCode(&state_, exec.get(), stop);
    FullState heavy = Capture();
    memcpy(heavy_table, table, sizeof(table));

    memcpy(table, seed_table, sizeof(table));
    ApplyInit(in);
    state_.cpu.insn_addr = start;
    int guard = 0;
    while (state_.cpu.insn_addr >= start && state_.cpu.insn_addr < stop && guard++ < 512) {
      InterpretInsn(&state_);
    }
    FullState itp = Capture();

    compared++;
    std::string desc;
    if (!Compare(heavy, itp, /*compare_fpsr=*/false, code, kInsns, &desc)) {
      ADD_FAILURE() << "iter " << iter << " register divergence: " << desc;
      break;
    }
    if (memcmp(heavy_table, table, sizeof(table)) != 0) {
      size_t w = 0;
      while (w < kTableWords && heavy_table[w] == table[w]) w++;
      ADD_FAILURE() << "iter " << iter << " MEMORY divergence at table word " << w
                    << " (offset 0x" << std::hex << (w * 8) << std::dec
                    << "): HEAVY=0x" << std::hex << heavy_table[w]
                    << " INTERP=0x" << table[w] << std::dec;
      break;
    }
  }
  EXPECT_GT(compared, 0) << "heavy declined the store-back region";
}






// A guest vector register read at two different widths inside one region. The
// heavy tier caches CPU-state slots per basic block, and `fmov w15, s0` reads
// V0 with a 64-bit MOVSD -- which zeroes the upper half of the XMM it lands in
// -- while the following `eor v1.16b, v1.16b, v0.16b` reads all 128 bits of the
// same V0. Forwarding the narrow load to the wide read silently substituted
// zeroes for V0's upper half, which is how a bcrypt key expansion produced a
// wrong hash with no crash and no bail.
TEST_F(Arm64HeavyDifferentialFuzz, VectorRegReadAtTwoWidths) {
  static const uint32_t code[] = {
      0x1e26000f,  // fmov w15, s0            (64-bit MOVSD get of V0)
      0x6e201c21,  // eor  v1.16b, v1.16b, v0.16b  (128-bit get of the same V0)
  };
  Seed(0x5117E0F1EE7ULL);
  int compared = 0;
  for (int iter = 0; iter < 500; iter++) {
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 2, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) ADD_FAILURE() << "iter " << iter << " " << desc;
  }
  EXPECT_GT(compared, 0) << "heavy declined the mixed-width vector region";
}

}  // namespace

// FRECPS / FRSQRTS (vector) against the interpreter.
//
// These two carry architectural special cases that the generic corpora cannot
// reach, because they only appear for particular VALUES rather than particular
// encodings: a lane whose product is 0*inf must saturate to 2.0 (FRECPS) or 1.5
// (FRSQRTS) instead of propagating the NaN the multiply produces, while a lane
// with a NaN INPUT must become the default qNaN. Distinguishing the two is the
// whole reason the lowering carries two cmpunord masks, so the corpus seeds
// lanes from a pool of specials (zeros, infinities, NaNs, denormals) often
// enough that both paths are hit, and mixes in random bits for the normal path.
//
// FP three-same is excluded from the generic NeonThreeSame generators, so
// without this the heavy lowering would have no differential coverage at all.
TEST_F(Arm64HeavyDifferentialFuzz, FrecpsFrsqrtsVector) {
  Seed(0xF2EC95A5ULL);
  const int kIters = 4000 * FuzzScale();

  // (bit23:22, is_double) for FRECPS.4S/.2D and FRSQRTS.4S/.2D.
  struct Form {
    uint32_t szbits;
    bool is_double;
  };
  static const Form kForms[] = {
      {0b00, false},  // FRECPS  .2S/.4S
      {0b01, true},   // FRECPS  .2D
      {0b10, false},  // FRSQRTS .2S/.4S
      {0b11, true},   // FRSQRTS .2D
  };
  static const uint64_t kSpecialD[] = {
      0x0000000000000000ULL,  // +0.0
      0x8000000000000000ULL,  // -0.0
      0x7FF0000000000000ULL,  // +inf
      0xFFF0000000000000ULL,  // -inf
      0x7FF8000000000000ULL,  // qNaN
      0x7FF0000000000001ULL,  // sNaN
      0x3FF0000000000000ULL,  // 1.0
      0x4000000000000000ULL,  // 2.0
      0x0000000000000001ULL,  // smallest denormal
      0x7FEFFFFFFFFFFFFFULL,  // max normal
  };
  static const uint32_t kSpecialS[] = {
      0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
      0x7F800001u, 0x3F800000u, 0x40000000u, 0x00000001u, 0x7F7FFFFFu,
  };

  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const Form& f = kForms[Rnd() % 4];
    // .1D (size=1, Q=0) is ARM-reserved.
    const bool q = f.is_double ? true : ((Rnd() & 1) != 0);
    const int rd = Rnd() % 32;
    const int rn = Rnd() % 32;
    const int rm = Rnd() % 32;
    uint32_t code[1] = {static_cast<uint32_t>(
        0x0E20FC00u | (q ? (1u << 30) : 0u) | (f.szbits << 22) |
        (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5) |
        static_cast<uint32_t>(rd))};

    InitState in = RandomInit();
    // Seed Vn/Vm with specials most of the time so 0*inf and NaN-input lanes
    // actually occur; fully random bits are overwhelmingly NaN for FP32.
    auto seed_vreg = [&](int reg) {
      uint64_t lo, hi;
      if (f.is_double) {
        lo = (Rnd() % 4) ? kSpecialD[Rnd() % 10] : Rnd64();
        hi = (Rnd() % 4) ? kSpecialD[Rnd() % 10] : Rnd64();
      } else {
        auto word = [&]() -> uint64_t {
          return (Rnd() % 4) ? kSpecialS[Rnd() % 10] : (Rnd() & 0xffffffffULL);
        };
        lo = word() | (word() << 32);
        hi = word() | (word() << 32);
      }
      in.v[reg] = (static_cast<unsigned __int128>(hi) << 64) | lo;
    };
    seed_vreg(rn);
    seed_vreg(rm);

    std::string desc;
    // FPSR is not compared: these set IOC/IDC through paths whose emulation the
    // tiers are allowed to differ on, exactly as the other FP classes here.
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
      break;
    }
  }
  fprintf(stderr, "FrecpsFrsqrtsVector: compared=%d/%d\n", compared, kIters);
  EXPECT_GT(compared, 300) << "heavy accepted too few FRECPS/FRSQRTS encodings";
}


// BFCVTN / BFCVTN2 against the interpreter.
//
// BF16 is the top 16 bits of an FP32, so the conversion is a round-to-nearest-
// even of the discarded half, with one exception: a NaN input must yield a
// quiet BF16 NaN, because rounding can carry all the way into the exponent and
// turn a NaN into an infinity. The corpus therefore leans on values where the
// low half is exactly a tie (0x8000) with both even and odd keep-bits, on
// 0x7F7FFFFF where the round carries out of the mantissa entirely, and on the
// NaN/infinity encodings either side of that boundary.
//
// BFCVTN2 additionally has to preserve the low 64 bits of Vd, which only shows
// up when Vd is seeded with something non-zero -- RandomInit does that, and both
// legs start from the same Vd.
TEST_F(Arm64HeavyDifferentialFuzz, BfcvtnVector) {
  Seed(0xBF16C0DEULL);
  const int kIters = 4000 * FuzzScale();
  static const uint32_t kSpecialS[] = {
      0x00000000u,  // +0.0
      0x80000000u,  // -0.0
      0x7F800000u,  // +inf
      0xFF800000u,  // -inf
      0x7FC00000u,  // qNaN
      0x7F800001u,  // sNaN (rounding alone would carry it to +inf)
      0x7F7FFFFFu,  // max normal: rounds up out of the mantissa
      0x3F808000u,  // exact tie, keep-bit even
      0x3F818000u,  // exact tie, keep-bit odd
      0x00000001u,  // smallest denormal
      0x3F800000u,  // 1.0
      0xBF800000u,  // -1.0
  };
  constexpr int kNumSpecial = sizeof(kSpecialS) / sizeof(kSpecialS[0]);

  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const bool q = (Rnd() & 1) != 0;  // BFCVTN2 when set
    const int rd = Rnd() % 32;
    const int rn = Rnd() % 32;
    uint32_t code[1] = {static_cast<uint32_t>(0x0EA16800u | (q ? (1u << 30) : 0u) |
                                              (static_cast<uint32_t>(rn) << 5) |
                                              static_cast<uint32_t>(rd))};

    InitState in = RandomInit();
    auto word = [&]() -> uint64_t {
      return (Rnd() % 3) ? kSpecialS[Rnd() % kNumSpecial] : (Rnd() & 0xffffffffULL);
    };
    const uint64_t lo = word() | (word() << 32);
    const uint64_t hi = word() | (word() << 32);
    in.v[rn] = (static_cast<unsigned __int128>(hi) << 64) | lo;

    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
      break;
    }
  }
  fprintf(stderr, "BfcvtnVector: compared=%d/%d\n", compared, kIters);
  EXPECT_GT(compared, 300) << "heavy accepted too few BFCVTN encodings";
}


// SRSHL / SQRSHL (vector, .2S/.4S) against the interpreter.
//
// The shift operand is the SIGNED LOW BYTE of each Vm lane -- the upper 24 bits
// are ignored -- so the corpus deliberately seeds shift lanes whose upper bits
// are garbage. The boundary shifts are where the arms of the lowering meet:
// +31/+32 (left shift into and past the sign bit; VPSLLVD zeroing must match
// ARM), -31/-32/-33 (rounding right shift at and past the element width, where
// the result collapses to 0) and -128 (the most negative byte). Saturation
// lanes (INT32_MIN/INT32_MAX inputs with small left shifts) drive SQRSHL's
// back-shift overflow detector.
TEST_F(Arm64HeavyDifferentialFuzz, SrshlSqrshlVector32) {
  Seed(0x5259A57BULL);
  const int kIters = 4000 * FuzzScale();
  static const int8_t kShifts[] = {0,  1,  2,  30,  31,  32,  33,  63, 127,
                                   -1, -2, -30, -31, -32, -33, -64, -127, -128};
  constexpr int kNumShifts = sizeof(kShifts) / sizeof(kShifts[0]);
  static const uint32_t kValues[] = {
      0x00000000u, 0x00000001u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu,
      0x40000000u, 0xC0000000u, 0x00008000u, 0xAAAAAAAAu, 0x55555555u,
  };
  constexpr int kNumValues = sizeof(kValues) / sizeof(kValues[0]);

  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const bool sat = (Rnd() & 1) != 0;  // SQRSHL vs SRSHL
    const bool q = (Rnd() & 1) != 0;
    const int rd = Rnd() % 32;
    int rn = Rnd() % 32;
    int rm = Rnd() % 32;
    if (rm == rn) rm = (rm + 1) % 32;
    // U=0, size=10; opcode 0b01010 (SRSHL) / 0b01011 (SQRSHL) at bits[15:11],
    // bit10 set (three-same). Corpus ground truth: 0x4e6054c6 == srshl.
    uint32_t code[1] = {static_cast<uint32_t>(
        0x0EA05400u | (q ? (1u << 30) : 0u) | (sat ? (1u << 11) : 0u) |
        (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5) |
        static_cast<uint32_t>(rd))};

    InitState in = RandomInit();
    auto data_word = [&]() -> uint64_t {
      return (Rnd() % 3) ? kValues[Rnd() % kNumValues] : (Rnd() & 0xffffffffULL);
    };
    auto shift_word = [&]() -> uint64_t {
      // Boundary shift in the low byte; garbage in the upper 24 bits half the
      // time, to prove only the low byte is honoured.
      uint32_t sh = static_cast<uint8_t>(kShifts[Rnd() % kNumShifts]);
      if (Rnd() & 1) sh |= (Rnd() & 0xffffff00u);
      return sh;
    };
    in.v[rn] = (static_cast<unsigned __int128>(data_word() | (data_word() << 32)) << 64) |
               (data_word() | (data_word() << 32));
    in.v[rm] = (static_cast<unsigned __int128>(shift_word() | (shift_word() << 32)) << 64) |
               (shift_word() | (shift_word() << 32));

    std::string desc;
    // FPSR not compared: SQRSHL sets QC on saturation through a path whose
    // emulation the tiers are allowed to differ on, as with the other classes.
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
      break;
    }
  }
  fprintf(stderr, "SrshlSqrshlVector32: compared=%d/%d\n", compared, kIters);
  EXPECT_GT(compared, 300) << "heavy accepted too few SRSHL/SQRSHL encodings";
}


// SSHL / USHL (vector, .2S/.4S/.2D) against the interpreter.
//
// Same corpus philosophy as the rounding variants above, with the 64-bit lanes
// included: the bail histogram over real apps put USHL.2D at the top of the
// actionable list. The 64-bit arithmetic right shift is emulated (no VPSRAVQ
// below AVX-512) via ((v ^ m) >>l s) ^ m with m the sign smear, so the
// boundary shifts -63/-64/-65 and -128 are the lanes most worth hammering.
TEST_F(Arm64HeavyDifferentialFuzz, SshlUshlVector) {
  Seed(0x55AA55E1ULL);
  const int kIters = 4000 * FuzzScale();
  static const int8_t kShifts[] = {0,   1,   31,  32,  33,  63,   64,  65,
                                   127, -1,  -31, -32, -33, -63,  -64, -65,
                                   -127, -128};
  constexpr int kNumShifts = sizeof(kShifts) / sizeof(kShifts[0]);
  static const uint64_t kValues[] = {
      0x0000000000000000ULL, 0x0000000000000001ULL, 0x7FFFFFFFFFFFFFFFULL,
      0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0x4000000000000000ULL,
      0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL, 0x0000000080000000ULL,
      0x123456789ABCDEF0ULL,
  };
  constexpr int kNumValues = sizeof(kValues) / sizeof(kValues[0]);

  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const bool is_signed = (Rnd() & 1) != 0;  // SSHL vs USHL (U bit)
    const bool is_64 = (Rnd() & 1) != 0;
    const bool q = is_64 ? true : ((Rnd() & 1) != 0);
    const int rd = Rnd() % 32;
    int rn = Rnd() % 32;
    int rm = Rnd() % 32;
    if (rm == rn) rm = (rm + 1) % 32;
    // Three-same opcode 0b01000 at bits[15:11], bit10 set; U at bit29.
    // Ground truth: 0x6ee14400 == ushl v0.2d, v0.2d, v1.2d.
    uint32_t code[1] = {static_cast<uint32_t>(
        0x0E204400u | (q ? (1u << 30) : 0u) | (is_signed ? 0u : (1u << 29)) |
        ((is_64 ? 0b11u : 0b10u) << 22) | (static_cast<uint32_t>(rm) << 16) |
        (static_cast<uint32_t>(rn) << 5) | static_cast<uint32_t>(rd))};

    InitState in = RandomInit();
    auto data_word = [&]() -> uint64_t {
      return (Rnd() % 3) ? kValues[Rnd() % kNumValues] : Rnd64();
    };
    auto shift_lane = [&](bool wide) -> uint64_t {
      uint64_t sh = static_cast<uint8_t>(kShifts[Rnd() % kNumShifts]);
      // Garbage above the low byte half the time: only bits [7:0] are the
      // architectural shift amount.
      if (Rnd() & 1) sh |= (Rnd64() << 8) & (wide ? ~0xFFULL : 0xFFFFFF00ULL);
      return sh;
    };
    if (is_64) {
      in.v[rn] = (static_cast<unsigned __int128>(data_word()) << 64) | data_word();
      in.v[rm] = (static_cast<unsigned __int128>(shift_lane(true)) << 64) | shift_lane(true);
    } else {
      auto dw = [&]() -> uint64_t {
        return (data_word() & 0xffffffffULL) | (data_word() << 32);
      };
      auto sw = [&]() -> uint64_t {
        return (shift_lane(false) & 0xffffffffULL) | (shift_lane(false) << 32);
      };
      in.v[rn] = (static_cast<unsigned __int128>(dw()) << 64) | dw();
      in.v[rm] = (static_cast<unsigned __int128>(sw()) << 64) | sw();
    }

    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
      break;
    }
  }
  fprintf(stderr, "SshlUshlVector: compared=%d/%d\n", compared, kIters);
  EXPECT_GT(compared, 300) << "heavy accepted too few SSHL/USHL encodings";
}

// CMEQ/CMGT/CMHI/CMGE/CMHS .2D against the interpreter. The unsigned forms go
// through a sign-bias XOR before PCMPGTQ, so the corpus leans on lane pairs
// that straddle the sign boundary -- (0x8000000000000000, 1) flips its answer
// between the signed and unsigned orders -- plus equal lanes for the >= forms.
TEST_F(Arm64HeavyDifferentialFuzz, CompareVector2D) {
  Seed(0xC0DE2D2DULL);
  const int kIters = 4000 * FuzzScale();
  static const uint64_t kValues[] = {
      0x0000000000000000ULL, 0x0000000000000001ULL, 0x7FFFFFFFFFFFFFFFULL,
      0x8000000000000000ULL, 0x8000000000000001ULL, 0xFFFFFFFFFFFFFFFFULL,
      0x0123456789ABCDEFULL,
  };
  constexpr int kNumValues = sizeof(kValues) / sizeof(kValues[0]);
  // (U bit, opcode bits[15:11]) per ARM ARM three-same:
  // CMGT=U0/0b00110, CMGE=U0/0b00111, CMHI=U1/0b00110, CMHS=U1/0b00111,
  // CMEQ=U1/0b10001.
  struct Form {
    uint32_t u;
    uint32_t opc;
  };
  static const Form kForms[] = {{0, 0b00110}, {0, 0b00111}, {1, 0b00110},
                                {1, 0b00111}, {1, 0b10001}};

  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const Form& f = kForms[Rnd() % 5];
    const int rd = Rnd() % 32;
    int rn = Rnd() % 32;
    int rm = Rnd() % 32;
    uint32_t code[1] = {static_cast<uint32_t>(
        0x4EE00400u | (f.u << 29) | (f.opc << 11) | (static_cast<uint32_t>(rm) << 16) |
        (static_cast<uint32_t>(rn) << 5) | static_cast<uint32_t>(rd))};

    InitState in = RandomInit();
    auto lane = [&]() -> uint64_t {
      return (Rnd() % 4) ? kValues[Rnd() % kNumValues] : Rnd64();
    };
    in.v[rn] = (static_cast<unsigned __int128>(lane()) << 64) | lane();
    // Equal lanes a third of the time so the >= vs > distinction is exercised.
    in.v[rm] = (Rnd() % 3 == 0) ? in.v[rn]
                                : (static_cast<unsigned __int128>(lane()) << 64) | lane();

    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
      break;
    }
  }
  fprintf(stderr, "CompareVector2D: compared=%d/%d\n", compared, kIters);
  EXPECT_GT(compared, 300) << "heavy accepted too few .2D compare encodings";
}

// DUP (scalar) against the interpreter, exhaustively: every valid imm5
// (element size x index) with randomized source registers and values. The
// space is tiny, so no sampling -- all 31 imm5 shapes per iteration.
TEST_F(Arm64HeavyDifferentialFuzz, DupScalarExhaustive) {
  Seed(0xD0B5CA1AULL);
  const int kIters = 200 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    for (uint32_t imm5 = 1; imm5 < 32; imm5++) {
      const int rd = Rnd() % 32;
      const int rn = Rnd() % 32;
      // 01011110000 imm5 000001 Rn Rd; ground truth 0x5e0c0401 == mov s1,v0.s[1].
      uint32_t code[1] = {static_cast<uint32_t>(
          0x5E000400u | (imm5 << 16) | (static_cast<uint32_t>(rn) << 5) |
          static_cast<uint32_t>(rd))};
      InitState in = RandomInit();
      std::string desc;
      Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
      if (r == kDeclined) continue;
      compared++;
      if (r == kDiverge) {
        ADD_FAILURE() << "iter " << iter << " imm5=" << imm5 << " " << desc;
        return;
      }
    }
  }
  fprintf(stderr, "DupScalarExhaustive: compared=%d\n", compared);
  EXPECT_GT(compared, 1000) << "heavy accepted too few DUP-scalar encodings";
}


// SHA-256 (SHA256H / SHA256H2 / SHA256SU0 / SHA256SU1) against the interpreter.
// Bit-exact hashing is the whole point, so this is the gate that matters. The
// inputs are fully random 128-bit V-registers (the round math has no special
// values -- it is adds, rotates, and/xor over uint32 lanes), and every op is
// checked at its exact encoding (ground-truth comments below).
TEST_F(Arm64HeavyDifferentialFuzz, Sha256) {
  Seed(0x54A256ULL);
  const int kIters = 3000 * FuzzScale();
  struct Form {
    uint32_t enc_base;  // rd/rn/rm inserted at [4:0],[9:5],[20:16]
    bool three_reg;     // true: H/H2/SU1 (has rm); false: SU0 (rd,rn only)
  };
  static const Form kForms[] = {
      {0x5E004000u, true},   // sha256h   q0, q0, v0.4s   (opcode 100)
      {0x5E005000u, true},   // sha256h2  q0, q0, v0.4s   (opcode 101)
      {0x5E006000u, true},   // sha256su1 v0.4s,v0.4s,v0.4s(opcode 110)
      {0x5E282800u, false},  // sha256su0 v0.4s, v0.4s     (opcode 10)
  };
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const Form& f = kForms[Rnd() % 4];
    const int rd = Rnd() % 32;
    const int rn = Rnd() % 32;
    const int rm = Rnd() % 32;
    uint32_t enc = f.enc_base | static_cast<uint32_t>(rd) | (static_cast<uint32_t>(rn) << 5);
    if (f.three_reg) enc |= (static_cast<uint32_t>(rm) << 16);
    uint32_t code[1] = {enc};
    InitState in = RandomInit();  // random V-registers
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " enc=0x" << std::hex << enc << " " << desc;
      break;
    }
  }
  fprintf(stderr, "Sha256: compared=%d/%d\n", compared, kIters);
  EXPECT_GT(compared, 300) << "heavy accepted too few SHA-256 encodings";
}

}  // namespace berberis
