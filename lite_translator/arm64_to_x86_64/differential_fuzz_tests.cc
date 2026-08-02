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
//     sweep (long-sweep driver script). The shift-imm
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
constexpr uint16_t kNZCVMask = kFlagsNZCVMask;

// Widen the sweep when the exhaustive-mode env knob is set (see
// long-sweep driver script). Off => 1 (CI, ~seconds).
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

  // AdvSIMD two-register miscellaneous (INTEGER forms only):
  //   0 Q U 01110 size 10000 opcode 10 Rn Rd.
  // Sweeps the integer / bitwise / compare / narrowing opcode set the decoder
  // routes through DecodeAdvSimdTwoRegMisc with bit20=0 (the true two-reg-misc
  // space, NOT the bit20=1 across-lanes reductions, which have a distinct
  // reg-mapping shape). rd aliases rn ~half the time to sample the destructive
  // in-place lowering (x86 SSE is 2-operand while the ARM op is not).
  //
  // Note: today the LITE tier JITs only CMEQ #0 (kCmeqZero, the linker's
  // calculate_gnu_hash_neon path) in this class; every other opcode bails to the
  // interpreter and is kDeclined (skipped) by the differential -- so the sweep
  // guards that one JIT'd op now and auto-covers the rest as the lite handler
  // grows. The compared>threshold guard is set accordingly low below.
  //
  // FP two-register-misc forms are intentionally EXCLUDED: the FP-result ops
  // (FABS/FNEG/FRINT*/FSQRT/FCVTN/FCVTL/FRECPE/FRSQRTE/SCVTF/UCVTF) can diverge
  // JIT-vs-interp on NaN payloads / rounding-mode edges exactly like the excluded
  // FP three-same class, and the FP->int converts (FCVTZS.., FCVTAS..) are pinned
  // by the ScalarFpConversion generator and the heavy FCVTA/FRINT generators. The
  // integer set here is bit-exact under fully-random input, so any divergence is
  // a real miscompile.
  uint32_t GenAdvSimdTwoRegMisc() {
    // opcode; fixed_u (-1 = sweep U); smax (sweep size in [0, smax]); fixed_size
    // (-1 = sweep). When the chosen size is 0b11 (.2D) Q is forced to 1 -- the
    // .1D form (size=11, Q=0) is unallocated for these.
    static const struct {
      uint8_t opcode;
      int8_t fixed_u;
      int8_t smax;
      int8_t fixed_size;
    } kEnc[] = {
        {0b00000, -1, 1, -1},  // REV64 (U=0) / REV32 (U=1); size 00/01 valid for both
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

  // AdvSIMD scalar three-same (INTEGER forms only):
  //   01 U 11110 size 1 Rm opcode 1 Rn Rd  (bit30=1, the 01011110 prefix).
  // Single-lane integer ops; rd aliases rn or rm to sample the destructive
  // lowering. The lite tier JITs the whole integer set (ADD/SUB, CMxx, S/U SHL,
  // the S/U-Q saturating add/sub/shift ladder, SQDMULH/SQRDMULH), so the accept
  // rate is high.
  //
  // FP scalar-three-same (FABD/FMULX/FCMxx/FACxx/FRECPS/FRSQRTS, opcodes
  // 11010/11011/11100/11101/11111) are EXCLUDED for the same NaN-payload reason
  // as the FP three-same class.
  uint32_t GenAdvSimdScalarThreeSame() {
    // opcode; size_mode: 0 = all B/H/S/D (sweep 0..3), 1 = H/S only (01/10),
    // 2 = D only (size=11).
    static const struct {
      uint8_t opcode;
      uint8_t size_mode;
    } kEnc[] = {
        {0b00001, 0},  // SQADD / UQADD   (B/H/S/D)
        {0b00101, 0},  // SQSUB / UQSUB   (B/H/S/D)
        {0b01001, 0},  // SQSHL / UQSHL   (B/H/S/D)
        {0b01011, 0},  // SQRSHL / UQRSHL (B/H/S/D)
        {0b01010, 2},  // SRSHL / URSHL   (D only)
        {0b10110, 1},  // SQDMULH / SQRDMULH (H/S only)
        {0b00110, 2},  // CMGT / CMHI     (D only)
        {0b00111, 2},  // CMGE / CMHS     (D only)
        {0b01000, 2},  // SSHL / USHL     (D only)
        {0b10000, 2},  // ADD / SUB       (D only)
        {0b10001, 2},  // CMTST / CMEQ    (D only)
    };
    const auto& e = kEnc[Rnd() % (sizeof(kEnc) / sizeof(kEnc[0]))];
    uint32_t u = Rnd() & 1;
    uint32_t size;
    switch (e.size_mode) {
      case 1: size = 1 + (Rnd() & 1); break;  // 01 or 10
      case 2: size = 3; break;                // 11 (D)
      default: size = Rnd() % 4; break;       // 00..11
    }
    uint32_t rm = Rnd() % 8, rn = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? rm : (Rnd() % 8));  // alias rn / rm / free
    return (0b01u << 30) | (u << 29) | (0b11110u << 24) | (size << 22) | (1u << 21) |
           (rm << 16) | (static_cast<uint32_t>(e.opcode) << 11) | (1u << 10) |
           (rn << 5) | rd;
  }

  // AdvSIMD three-different (INTEGER widening / narrowing):
  //   0 Q U 01110 size Rm opcode 00 Rn Rd  (bit21=1, bits[11:10]=00).
  // SADDL/…/SMULL/UMULL, ADDHN/SUBHN narrowing-high, SQDMULL/SQDMLAL/SQDMLSL
  // saturating-doubling, PMULL polynomial. Q selects the source half (the "2"
  // forms read the upper 64 bits). All integer => bit-exact; rd aliases rn/rm to
  // sample destructive lowering. The lite tier JITs the widening-MUL family
  // ({S,U}MULL/MLAL/MLSL) at all sizes; the rest bail (kDeclined).
  uint32_t GenAdvSimdThreeDiff() {
    // opcode; umode (-1 = sweep U, 0 = force U=0); size_mode:
    //   0 = {00,01,10} (widening / narrowing-high),
    //   1 = {01,10}    (SQDMULL/SQDMLAL/SQDMLSL: size=00 reserved),
    //   2 = {00,11}    (PMULL: 8-bit and PMULL64; 01/10 unallocated).
    static const struct {
      uint8_t opcode;
      int8_t umode;
      uint8_t size_mode;
    } kEnc[] = {
        {0b0000, -1, 0},  // SADDL / UADDL
        {0b0001, -1, 0},  // SADDW / UADDW
        {0b0010, -1, 0},  // SSUBL / USUBL
        {0b0011, -1, 0},  // SSUBW / USUBW
        {0b0100, -1, 0},  // ADDHN / RADDHN
        {0b0101, -1, 0},  // SABAL / UABAL
        {0b0110, -1, 0},  // SUBHN / RSUBHN
        {0b0111, -1, 0},  // SABDL / UABDL
        {0b1000, -1, 0},  // SMLAL / UMLAL
        {0b1001,  0, 1},  // SQDMLAL (U=0)
        {0b1010, -1, 0},  // SMLSL / UMLSL
        {0b1011,  0, 1},  // SQDMLSL (U=0)
        {0b1100, -1, 0},  // SMULL / UMULL
        {0b1101,  0, 1},  // SQDMULL (U=0)
        {0b1110,  0, 2},  // PMULL / PMULL2 (U=0)
    };
    const auto& e = kEnc[Rnd() % (sizeof(kEnc) / sizeof(kEnc[0]))];
    uint32_t u = (e.umode < 0) ? (Rnd() & 1) : 0u;
    uint32_t size;
    switch (e.size_mode) {
      case 1: size = 1 + (Rnd() & 1); break;       // 01 or 10
      case 2: size = (Rnd() & 1) ? 3 : 0; break;   // 00 or 11 (PMULL64)
      default: size = Rnd() % 3; break;            // 00 / 01 / 10
    }
    uint32_t q = Rnd() & 1;
    uint32_t rm = Rnd() % 8, rn = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? rm : (Rnd() % 8));  // alias rn / rm / free
    return (q << 30) | (u << 29) | (0b01110u << 24) | (size << 22) | (1u << 21) |
           (rm << 16) | (static_cast<uint32_t>(e.opcode) << 12) | (rn << 5) | rd;
  }

  // AdvSIMD vector x indexed element (INTEGER by-element):
  //   0 Q U 01111 size L M Rm opcode H 0 Rn Rd  (bit10=0).
  // MUL/MLA/MLS, SQDMULH/SQRDMULH, SQRDMLAH/SQRDMLSH, widening SMULL/UMULL/
  // SMLAL/UMLAL/SMLSL/UMLSL, and saturating-doubling SQDMULL/SQDMLAL/SQDMLSL, by
  // a broadcast lane of Vm. size=01 (halfword) uses Vm=Rm[3:0] (V0..V15) with
  // index=H:L:M (0..7); size=10 (word) uses Vm=M:Rm[3:0] with index=H:L (0..3).
  // rd aliases rn / the indexed Vm to sample destructive lowering. The FP
  // by-element forms (FMUL/FMLA/FMLS/FMULX) and the dot-product / BF16 carve-outs
  // are EXCLUDED (FP NaN payloads + special dispatch); the heavy fuzzer covers
  // the FP forms with finite-only seeds.
  uint32_t GenAdvSimdVecXIndexedElement() {
    static const struct {
      int8_t u;
      uint8_t opcode;
    } kEnc[] = {
        {0, 0b1000},  // MUL      (U=0)
        {1, 0b0000},  // MLA      (U=1)
        {1, 0b0100},  // MLS      (U=1)
        {0, 0b1100},  // SQDMULH  (U=0)
        {0, 0b1101},  // SQRDMULH (U=0)
        {1, 0b1101},  // SQRDMLAH (U=1)
        {1, 0b1111},  // SQRDMLSH (U=1)
        {0, 0b1010},  // SMULL    (U=0)
        {1, 0b1010},  // UMULL    (U=1)
        {0, 0b0010},  // SMLAL    (U=0)
        {1, 0b0010},  // UMLAL    (U=1)
        {0, 0b0110},  // SMLSL    (U=0)
        {1, 0b0110},  // UMLSL    (U=1)
        {0, 0b1011},  // SQDMULL  (U=0)
        {0, 0b0011},  // SQDMLAL  (U=0)
        {0, 0b0111},  // SQDMLSL  (U=0)
    };
    const auto& e = kEnc[Rnd() % (sizeof(kEnc) / sizeof(kEnc[0]))];
    uint32_t u = static_cast<uint32_t>(e.u);
    uint32_t q = Rnd() & 1;
    uint32_t size = 1 + (Rnd() & 1);  // 01 (H) or 10 (S) -- all of these bail at 00/11
    uint32_t vm = Rnd() % 8;          // <16 keeps halfword Rm valid; M=0 for word
    uint32_t rn = Rnd() % 8;
    uint32_t r = Rnd() % 3;
    uint32_t rd = r == 0 ? rn : (r == 1 ? vm : (Rnd() % 8));
    uint32_t H, L, M, Rm;
    if (size == 1) {  // halfword: index 0..7 = H:L:M, Vm = Rm[3:0]
      uint32_t index = Rnd() % 8;
      H = index >> 2;
      L = (index >> 1) & 1;
      M = index & 1;
      Rm = vm;
    } else {  // word: index 0..3 = H:L, Vm = M:Rm[3:0]
      uint32_t index = Rnd() % 4;
      H = index >> 1;
      L = index & 1;
      M = 0;
      Rm = vm;
    }
    return (q << 30) | (u << 29) | (0b01111u << 24) | (size << 22) | (L << 21) |
           (M << 20) | (Rm << 16) | (static_cast<uint32_t>(e.opcode) << 12) |
           (H << 11) | (rn << 5) | rd;
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

// Single-instruction AdvSIMD two-register miscellaneous (integer forms). Today
// the lite tier JITs only CMEQ #0 in this class (linker gnu-hash path); the rest
// bail to the interpreter (kDeclined). The threshold is set low to match, and
// rises automatically for free as the lite two-reg-misc handler grows.
TEST_F(Arm64DifferentialFuzz, AdvSimdTwoRegMisc) {
  Seed(0x2E600F1512345678ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenAdvSimdTwoRegMisc()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 30) << "JIT accepted too few two-reg-misc encodings";
}

// Single-instruction AdvSIMD scalar three-same (integer forms). rd aliasing
// rn/rm sampled; the lite tier JITs the whole integer scalar ladder.
TEST_F(Arm64DifferentialFuzz, AdvSimdScalarThreeSame) {
  Seed(0x5CA1A53A5A4E5A3EULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenAdvSimdScalarThreeSame()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 200) << "JIT accepted too few scalar three-same encodings";
}

// Single-instruction AdvSIMD three-different (integer widening/narrowing). rd
// aliasing rn/rm sampled; the lite tier JITs the widening-MUL family at all
// sizes (the rest bail).
TEST_F(Arm64DifferentialFuzz, AdvSimdThreeDiff) {
  Seed(0x3D1FF00D2468ACE0ULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenAdvSimdThreeDiff()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 200) << "JIT accepted too few three-different encodings";
}

// Single-instruction AdvSIMD vector x indexed-element (integer by-element). rd
// aliasing rn / the indexed Vm sampled; the lite tier JITs the integer MUL/MLA/
// MLS by-element forms (widening / saturating variants bail).
TEST_F(Arm64DifferentialFuzz, AdvSimdVecXIndexedElement) {
  Seed(0x1DE7EC7ED00DFEEDULL);
  const int kIters = 5000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    uint32_t code[1] = {GenAdvSimdVecXIndexedElement()};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
    }
  }
  EXPECT_GT(compared, 200) << "JIT accepted too few by-element encodings";
}

// Coverage: the four new generators reach the register-aliasing and key
// opcode/size shapes they exist to stress, so a future refactor that silently
// stops producing one of them fails loudly rather than making a fuzzer vacuous.
TEST_F(Arm64DifferentialFuzz, NewFamilyGeneratorCoverage) {
  bool trm_cmeqz = false, trm_rev = false, trm_2d = false, trm_alias = false;
  Seed(0x2E600F1500C0FFEEULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenAdvSimdTwoRegMisc();
    uint32_t opcode = (insn >> 12) & 0x1F, u = (insn >> 29) & 1;
    uint32_t size = (insn >> 22) & 3, q = (insn >> 30) & 1;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (opcode == 0b01001 && u == 0) trm_cmeqz = true;         // CMEQ #0 (JIT'd)
    if (opcode == 0b00000) trm_rev = true;                     // REV64/REV32
    if (size == 3 && q == 1) trm_2d = true;                    // .2D form
    if (rd == rn) trm_alias = true;
  }
  EXPECT_TRUE(trm_cmeqz) << "two-reg-misc generator no longer produces CMEQ #0";
  EXPECT_TRUE(trm_rev) << "two-reg-misc generator no longer produces REV64/REV32";
  EXPECT_TRUE(trm_2d) << "two-reg-misc generator no longer produces the .2D form";
  EXPECT_TRUE(trm_alias) << "two-reg-misc generator no longer produces rd==rn";

  bool s3_add = false, s3_sat = false, s3_dmulh = false, s3_alias = false;
  Seed(0x5CA1A53A0BADF00DULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenAdvSimdScalarThreeSame();
    uint32_t opcode = (insn >> 11) & 0x1F;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F, rm = (insn >> 16) & 0x1F;
    if (opcode == 0b10000) s3_add = true;                      // ADD/SUB
    if (opcode == 0b00001 || opcode == 0b00101) s3_sat = true; // SQADD/SQSUB
    if (opcode == 0b10110) s3_dmulh = true;                    // SQDMULH/SQRDMULH
    if (rd == rn || rd == rm) s3_alias = true;
  }
  EXPECT_TRUE(s3_add) << "scalar three-same generator no longer produces ADD/SUB";
  EXPECT_TRUE(s3_sat) << "scalar three-same generator no longer produces SQADD/SQSUB";
  EXPECT_TRUE(s3_dmulh) << "scalar three-same generator no longer produces SQDMULH/SQRDMULH";
  EXPECT_TRUE(s3_alias) << "scalar three-same generator no longer produces rd==rn/rd==rm";

  bool td_smull = false, td_pmull = false, td_sqdmull = false, td_alias = false;
  Seed(0x3D1FF00D0F0F0F0FULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenAdvSimdThreeDiff();
    uint32_t opcode = (insn >> 12) & 0xF, u = (insn >> 29) & 1, size = (insn >> 22) & 3;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (opcode == 0b1100) td_smull = true;                     // SMULL/UMULL
    if (opcode == 0b1110 && u == 0 && size == 3) td_pmull = true;  // PMULL64
    if (opcode == 0b1101 && u == 0) td_sqdmull = true;         // SQDMULL
    if (rd == rn) td_alias = true;
  }
  EXPECT_TRUE(td_smull) << "three-different generator no longer produces SMULL/UMULL";
  EXPECT_TRUE(td_pmull) << "three-different generator no longer produces PMULL64";
  EXPECT_TRUE(td_sqdmull) << "three-different generator no longer produces SQDMULL";
  EXPECT_TRUE(td_alias) << "three-different generator no longer produces rd==rn";

  bool vx_mul = false, vx_mull = false, vx_half = false, vx_word = false, vx_alias = false;
  Seed(0x1DE7EC7E13572468ULL);
  for (int i = 0; i < 40000; i++) {
    uint32_t insn = GenAdvSimdVecXIndexedElement();
    uint32_t opcode = (insn >> 12) & 0xF, u = (insn >> 29) & 1, size = (insn >> 22) & 3;
    uint32_t rd = insn & 0x1F, rn = (insn >> 5) & 0x1F;
    if (opcode == 0b1000 && u == 0) vx_mul = true;             // MUL
    if (opcode == 0b1010) vx_mull = true;                      // SMULL/UMULL (widening)
    if (size == 1) vx_half = true;
    if (size == 2) vx_word = true;
    if (rd == rn) vx_alias = true;
  }
  EXPECT_TRUE(vx_mul) << "by-element generator no longer produces MUL";
  EXPECT_TRUE(vx_mull) << "by-element generator no longer produces widening MULL";
  EXPECT_TRUE(vx_half) << "by-element generator no longer produces halfword";
  EXPECT_TRUE(vx_word) << "by-element generator no longer produces word";
  EXPECT_TRUE(vx_alias) << "by-element generator no longer produces rd==rn";
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

// DUP (scalar) against the interpreter, exhaustively: every valid imm5
// (element size x index) with randomized registers and values. The lite
// lowering (three byte-shifts) was added alongside the heavy one -- before
// that BOTH JIT tiers bailed and every `mov s1, v0.s[1]` ran interpreted --
// so it gets the same differential treatment as any new lowering.
TEST_F(Arm64DifferentialFuzz, DupScalarExhaustive) {
  Seed(0xD0B5CA1BULL);
  const int kIters = 200 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    for (uint32_t imm5 = 1; imm5 < 32; imm5++) {
      const int rd = static_cast<int>(Rnd() % 32);
      const int rn = static_cast<int>(Rnd() % 32);
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
  fprintf(stderr, "lite DupScalarExhaustive: compared=%d\n", compared);
  EXPECT_GT(compared, 1000) << "lite accepted too few DUP-scalar encodings";
}


// SBFM (SBFX / SBFIZ / ASR / SXTB / SXTH / SXTW), exhaustively: every
// (sf, immr, imms) shape with randomized inputs. The general (non-alias)
// shapes went untranslated by the lite tier for months while heavy and the
// interpreter both handled them, so every `sbfx xN, xM, #0, #1` -- the
// carry/bool-to-mask idiom compilers emit throughout checked arithmetic and
// number parsing -- ended its region and round-tripped the interpreter. The
// tier coverage table showed it the whole time (sbfx: lite 0/128); this sweep
// pins the semantics now that the general path exists.
TEST_F(Arm64DifferentialFuzz, SbfmExhaustive) {
  Seed(0x5BF3ULL);
  int compared = 0;
  for (uint32_t sf = 0; sf < 2; sf++) {
    const uint32_t width = sf ? 64 : 32;
    for (uint32_t immr = 0; immr < width; immr++) {
      for (uint32_t imms = 0; imms < width; imms++) {
        const int rd = static_cast<int>(Rnd() % 31);
        int rn = static_cast<int>(Rnd() % 31);
        if (rn == rd) rn = (rn + 1) % 31;
        uint32_t code[1] = {0x13000000u | (sf << 31) | (sf << 22) | (immr << 16) |
                            (imms << 10) | (static_cast<uint32_t>(rn) << 5) |
                            static_cast<uint32_t>(rd)};
        InitState in = RandomInit();
        std::string desc;
        Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
        if (r == kDeclined) continue;
        compared++;
        if (r == kDiverge) {
          ADD_FAILURE() << "sf=" << sf << " immr=" << immr << " imms=" << imms << " " << desc;
          return;
        }
      }
    }
  }
  fprintf(stderr, "SbfmExhaustive: compared=%d\n", compared);
  // 64*64 + 32*32 = 5120 shapes; every one must now translate.
  EXPECT_EQ(compared, 5120) << "lite declined SBFM shapes it should translate";
}


// MRS NZCV / MSR NZCV against the interpreter, plus the save/restore pair.
// The lite recipes were shipped as admitted approximations: MRS parked C at
// bit 24 and V at bit 16 (architecturally 29 and 28), and MSR stored C and V
// into the wrong stored-flag bits entirely, so a guest's
// `mrs x; ...; msr nzcv, x` flag save/restore silently destroyed C and V --
// while the interpreter converts exactly. Real apps execute these: the
// heavy tier bails on MRS NZCV, so the lite recipe is what runs on-device.
TEST_F(Arm64DifferentialFuzz, MrsMsrNzcv) {
  Seed(0x2CFA5ULL);
  const int kIters = 2000 * FuzzScale();
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const int rd = static_cast<int>(Rnd() % 31);
    const int which = static_cast<int>(Rnd() % 3);
    uint32_t code[2];
    int n;
    if (which == 0) {
      code[0] = 0xD53B4200u | static_cast<uint32_t>(rd);  // mrs xN, nzcv
      n = 1;
    } else if (which == 1) {
      code[0] = 0xD51B4200u | static_cast<uint32_t>(rd);  // msr nzcv, xN
      n = 1;
    } else {
      code[0] = 0xD53B4200u | static_cast<uint32_t>(rd);  // save
      code[1] = 0xD51B4200u | static_cast<uint32_t>(rd);  // restore
      n = 2;
    }
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, n, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " which=" << which << " " << desc;
      return;
    }
  }
  fprintf(stderr, "MrsMsrNzcv: compared=%d\n", compared);
  EXPECT_GT(compared, 1000);
}

// MRS Xn, FPCR against the interpreter: newly JIT-resolved (reads the cached
// FPCR); the differential seeds cached_fpcr with random plausible values.
TEST_F(Arm64DifferentialFuzz, MrsFpcr) {
  Seed(0xF9C2ULL);
  int compared = 0;
  for (int iter = 0; iter < 512; iter++) {
    const int rd = static_cast<int>(Rnd() % 31);
    uint32_t code[1] = {0xD53B4400u | static_cast<uint32_t>(rd)};  // mrs xN, fpcr
    InitState in = RandomInit();
    std::string desc;
    // cached_fpcr is not part of InitState; poke it directly around the run.
    state_.cpu.cached_fpcr = static_cast<uint32_t>(Rnd64());
    uint32_t fpcr = state_.cpu.cached_fpcr;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    state_.cpu.cached_fpcr = fpcr;
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " " << desc;
      return;
    }
  }
  fprintf(stderr, "MrsFpcr: compared=%d\n", compared);
  EXPECT_GT(compared, 400);
}


// SHA-256 (SHA256H/H2/SU0/SU1) lite-vs-interpreter. Bit-exact hashing is the
// point; inputs are fully random V-registers. The lite lowering stages the
// working state in ThreadState scratch, so this also exercises that the scratch
// round-trip reconstructs the exact result.
TEST_F(Arm64DifferentialFuzz, Sha256) {
  Seed(0x54A25611ULL);
  const int kIters = 3000 * FuzzScale();
  struct Form { uint32_t base; bool three; };
  static const Form kForms[] = {
      {0x5E004000u, true},   // sha256h   (opcode 100)
      {0x5E005000u, true},   // sha256h2  (opcode 101)
      {0x5E006000u, true},   // sha256su1 (opcode 110)
      {0x5E282800u, false},  // sha256su0 (opcode 10)
  };
  int compared = 0;
  for (int iter = 0; iter < kIters; iter++) {
    const Form& f = kForms[Rnd() % 4];
    const int rd = static_cast<int>(Rnd() % 32);
    const int rn = static_cast<int>(Rnd() % 32);
    const int rm = static_cast<int>(Rnd() % 32);
    uint32_t enc = f.base | static_cast<uint32_t>(rd) | (static_cast<uint32_t>(rn) << 5);
    if (f.three) enc |= (static_cast<uint32_t>(rm) << 16);
    uint32_t code[1] = {enc};
    InitState in = RandomInit();
    std::string desc;
    Result r = RunDifferential(code, 1, in, /*compare_fpsr=*/false, &desc);
    if (r == kDeclined) continue;
    compared++;
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << iter << " enc=0x" << std::hex << enc << " " << desc;
      return;
    }
  }
  fprintf(stderr, "lite Sha256: compared=%d/%d\n", compared, kIters);
  EXPECT_GT(compared, 300) << "lite accepted too few SHA-256 encodings";
}

}  // namespace berberis
