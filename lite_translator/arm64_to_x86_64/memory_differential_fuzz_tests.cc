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

// JIT-vs-interpreter differential fuzzer for MEMORY access.
//
// The sibling register-only fuzzer (differential_fuzz_tests.cc) compares the
// full end state after a region, but every corpus it generates is
// register-to-register: it never emits a load, a store or an atomic, and it
// never compares memory. That blind spot covers the surface where several
// shipped bugs lived:
//
//   * the interpreter ignoring ARM64 top-byte-ignore, so a Scudo-tagged heap
//     pointer faulted under interpret-only while both JIT tiers masked it;
//   * LDPSW zero-extending instead of sign-extending, which wedged a libQt6Gui
//     fill loop;
//   * SWP decoded as LDADD, silently returning the wrong value.
//
// None of those are reachable without emitting memory instructions and
// comparing memory afterwards.
//
// Design follows the register fuzzer so failures reproduce the same way:
// identical seeded CPUState, deterministic PRNG, full end-state comparison --
// and additionally a byte-exact comparison of a guest-visible scratch buffer
// that both runs execute against from identical starting contents.
//
// Accesses are constrained to a window in the middle of the scratch buffer so
// that every generated address is in bounds; this fuzzer is about
// value/addressing semantics, not fault behaviour (guest signal delivery has
// its own tests).
//
// Exclusives (LDXR/STXR) are deliberately excluded: they carry monitor state
// whose JIT and interpreter models are allowed to differ in when the monitor is
// cleared, so a divergence there would not indicate a bug.

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
// (same test library); do NOT define it here or the link fails with a duplicate
// symbol.

namespace {

constexpr uint16_t kNZCVMask = kFlagsNZCVMask;

int FuzzScale() {
  const char* e = getenv("BERBERIS_DIFFERENTIAL_FUZZ_EXHAUSTIVE");
  if (e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0')) {
    return 25;
  }
  return 1;
}

// Scratch layout. Accesses are generated around kWindowBase with a reach of at
// most kWindowReach in either direction, so every address stays inside the
// buffer even for negative offsets and for post-index writeback.
constexpr size_t kScratchBytes = 4096;
constexpr size_t kWindowBase = 2048;
constexpr size_t kWindowReach = 1024;

alignas(16) uint8_t g_scratch[kScratchBytes];
alignas(16) uint8_t g_seed_mem[kScratchBytes];
alignas(16) uint8_t g_jit_mem[kScratchBytes];

class Arm64MemoryDifferentialFuzz : public ::testing::Test {
 protected:
  ThreadState state_{};
  uint64_t seed_ = 0;

  void Seed(uint64_t s) { seed_ = s; }
  uint64_t Rnd() {
    seed_ = seed_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed_ >> 33;
  }
  uint64_t Rnd64() { return ((Rnd() & 0xffffffffULL) << 32) | (Rnd() & 0xffffffffULL); }

  struct FullState {
    uint64_t x[31];
    uint64_t sp;
    uint16_t flags;
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
    for (int i = 0; i < 32; i++) memcpy(&s.v[i], &state_.cpu.v[i], 16);
    return s;
  }

  InitState RandomInit() {
    InitState in;
    for (int i = 0; i < 31; i++) in.x[i] = Rnd64();
    in.sp = Rnd64() & ~0xFULL;
    in.flags = static_cast<uint16_t>(Rnd() & kNZCVMask);
    for (int i = 0; i < 32; i++) {
      unsigned __int128 hi = Rnd64(), lo = Rnd64();
      in.v[i] = (hi << 64) | lo;
    }
    return in;
  }

  static uint64_t ScratchBase() { return static_cast<uint64_t>(ToGuestAddr(g_scratch)); }

  // Reseeds the scratch buffer deterministically and remembers the contents so
  // both runs start from exactly the same memory.
  void SeedScratch() {
    for (size_t i = 0; i < kScratchBytes; i += 8) {
      uint64_t v = Rnd64();
      memcpy(g_seed_mem + i, &v, 8);
    }
    memcpy(g_scratch, g_seed_mem, kScratchBytes);
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

  // Runs the region through the lite JIT and the interpreter from identical
  // register state AND identical scratch contents, then compares both.
  Result RunDifferential(const uint32_t* code, int n, const InitState& in, std::string* desc) {
    GuestAddr start = ToGuestAddr(code);
    GuestAddr code_end = start + static_cast<GuestAddr>(n) * 4;

    // JIT leg.
    memcpy(g_scratch, g_seed_mem, kScratchBytes);
    ApplyInit(in);
    state_.cpu.insn_addr = start;
    MachineCode mc;
    auto [ok, stop] = TryLiteTranslateRegion(
        start, &mc, LiteTranslateParams{.end_pc = code_end, .allow_dispatch = false});
    if (!ok || stop > code_end || stop == start) return kDeclined;
    HostCodeAddr hc = GetDefaultCodePoolInstance()->Add(&mc);
    TestingRunGeneratedCode(&state_, AsHostCode(hc), stop);
    FullState jit = Capture();
    memcpy(g_jit_mem, g_scratch, kScratchBytes);

    // Interpreter leg, to the same stop PC.
    memcpy(g_scratch, g_seed_mem, kScratchBytes);
    ApplyInit(in);
    state_.cpu.insn_addr = start;
    int guard = 0;
    while (state_.cpu.insn_addr >= start && state_.cpu.insn_addr < stop && guard++ < 256) {
      InterpretInsn(&state_);
    }
    FullState itp = Capture();

    return Compare(jit, itp, code, n, desc) ? kMatch : kDiverge;
  }

  bool Compare(const FullState& a,
               const FullState& b,
               const uint32_t* code,
               int n,
               std::string* desc) {
    char buf[224];
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
      snprintf(buf, sizeof(buf), "NZCV JIT=0x%04x INTERP=0x%04x region:", a.flags & kNZCVMask,
               b.flags & kNZCVMask);
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
    // Memory: the whole point of this fuzzer. A store that writes the right
    // value to the wrong address, or the wrong width to the right address,
    // shows up here and nowhere else.
    if (memcmp(g_jit_mem, g_scratch, kScratchBytes) != 0) {
      for (size_t i = 0; i < kScratchBytes; i++) {
        if (g_jit_mem[i] != g_scratch[i]) {
          snprintf(buf, sizeof(buf),
                   "memory[+%zu] JIT=0x%02x INTERP=0x%02x (offset from base %+d) region:", i,
                   g_jit_mem[i], g_scratch[i], static_cast<int>(i) - static_cast<int>(kWindowBase));
          *desc = std::string(buf) + RegionStr(code, n);
          return false;
        }
      }
    }
    return true;
  }

  // Pins `rn` at an in-window, `align`-aligned address, optionally carrying a
  // non-zero ARM64 address tag in bits [63:56].
  void PinBase(InitState* in, int rn, size_t align, bool tagged) {
    uint64_t off = kWindowBase + (Rnd() % 64) * align;
    uint64_t addr = (ScratchBase() + off) & ~(static_cast<uint64_t>(align) - 1);
    if (tagged) {
      // Any non-zero top byte. ARM64 ignores it on access; x86_64 does not even
      // consider such an address canonical, so both tiers must strip it.
      addr |= (0x11ULL + (Rnd() & 0x7f)) << 56;
    }
    in->x[rn] = addr;
  }
};

// --- encoders -------------------------------------------------------------
// Each verified against a known-good encoding in a comment, because a wrong
// encoder here would silently test the wrong instruction in both tiers and
// still pass.

// LDR/STR (immediate, unsigned offset). LDR X0,[X1] == 0xF9400020.
uint32_t EncLoadStoreImm(int size, int opc, uint32_t imm12, int rn, int rt) {
  return (static_cast<uint32_t>(size) << 30) | (0b111001u << 24) |
         (static_cast<uint32_t>(opc) << 22) | ((imm12 & 0xFFF) << 10) |
         (static_cast<uint32_t>(rn) << 5) | static_cast<uint32_t>(rt);
}

// LDR/STR (immediate, pre/post index). LDR X0,[X1],#8 == 0xF8408420.
uint32_t EncLoadStoreIdx(int size, int opc, int imm9, int idx, int rn, int rt) {
  return (static_cast<uint32_t>(size) << 30) | (0b111000u << 24) |
         (static_cast<uint32_t>(opc) << 22) | ((static_cast<uint32_t>(imm9) & 0x1FF) << 12) |
         (static_cast<uint32_t>(idx) << 10) | (static_cast<uint32_t>(rn) << 5) |
         static_cast<uint32_t>(rt);
}

// LDR/STR (register offset). LDR X0,[X1,X2] == 0xF8626820.
uint32_t EncLoadStoreReg(int size, int opc, int rm, int option, int s, int rn, int rt) {
  return (static_cast<uint32_t>(size) << 30) | (0b111000u << 24) |
         (static_cast<uint32_t>(opc) << 22) | (1u << 21) | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(option) << 13) | (static_cast<uint32_t>(s) << 12) | (0b10u << 10) |
         (static_cast<uint32_t>(rn) << 5) | static_cast<uint32_t>(rt);
}

// LDP/STP (signed offset). LDP X0,X1,[X2] == 0xA9400440; LDPSW == 0x69400440.
uint32_t EncLoadStorePair(int opc, int l, int imm7, int rt2, int rn, int rt) {
  return (static_cast<uint32_t>(opc) << 30) | (0b101u << 27) | (0b010u << 23) |
         (static_cast<uint32_t>(l) << 22) | ((static_cast<uint32_t>(imm7) & 0x7F) << 15) |
         (static_cast<uint32_t>(rt2) << 10) | (static_cast<uint32_t>(rn) << 5) |
         static_cast<uint32_t>(rt);
}

// SIMD LDR/STR (immediate, unsigned offset). LDR Q0,[X1] == 0x3DC00020.
uint32_t EncSimdLoadStoreImm(int size, int opc, uint32_t imm12, int rn, int rt) {
  return (static_cast<uint32_t>(size) << 30) | (0b111101u << 24) |
         (static_cast<uint32_t>(opc) << 22) | ((imm12 & 0xFFF) << 10) |
         (static_cast<uint32_t>(rn) << 5) | static_cast<uint32_t>(rt);
}

// LSE atomic read-modify-write. LDADD X1,X0,[X2] == 0xF8210040;
// SWP X1,X0,[X2] == 0xF8218040 (o3=1).
uint32_t EncAtomic(int size, int a, int r, int rs, int o3, int opc, int rn, int rt) {
  return (static_cast<uint32_t>(size) << 30) | (0b111000u << 24) |
         (static_cast<uint32_t>(a) << 23) | (static_cast<uint32_t>(r) << 22) | (1u << 21) |
         (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(o3) << 15) |
         (static_cast<uint32_t>(opc) << 12) | (static_cast<uint32_t>(rn) << 5) |
         static_cast<uint32_t>(rt);
}

// Encoder self-check: a bad encoder would test the wrong instruction in BOTH
// tiers and still report a match, so the encodings are pinned here.
TEST_F(Arm64MemoryDifferentialFuzz, EncodersMatchKnownEncodings) {
  EXPECT_EQ(EncLoadStoreImm(0b11, 0b01, 0, 1, 0), 0xF9400020u);  // ldr x0,[x1]
  EXPECT_EQ(EncLoadStoreImm(0b11, 0b00, 0, 1, 0), 0xF9000020u);  // str x0,[x1]
  EXPECT_EQ(EncLoadStoreImm(0b10, 0b10, 0, 1, 0), 0xB9800020u);  // ldrsw x0,[x1]
  EXPECT_EQ(EncLoadStoreIdx(0b11, 0b01, 8, 0b01, 1, 0), 0xF8408420u);      // ldr x0,[x1],#8
  EXPECT_EQ(EncLoadStoreReg(0b11, 0b01, 2, 0b011, 0, 1, 0), 0xF8626820u);  // ldr x0,[x1,x2]
  EXPECT_EQ(EncLoadStorePair(0b10, 1, 0, 1, 2, 0), 0xA9400440u);           // ldp x0,x1,[x2]
  EXPECT_EQ(EncLoadStorePair(0b01, 1, 0, 1, 2, 0), 0x69400440u);           // ldpsw x0,x1,[x2]
  EXPECT_EQ(EncSimdLoadStoreImm(0b00, 0b11, 0, 1, 0), 0x3DC00020u);        // ldr q0,[x1]
  EXPECT_EQ(EncSimdLoadStoreImm(0b00, 0b10, 0, 1, 0), 0x3D800020u);        // str q0,[x1]
  EXPECT_EQ(EncAtomic(0b11, 0, 0, 1, 0, 0b000, 2, 0), 0xF8210040u);        // ldadd x1,x0,[x2]
  EXPECT_EQ(EncAtomic(0b11, 0, 0, 1, 1, 0b000, 2, 0), 0xF8218040u);        // swp x1,x0,[x2]
}

// Scalar load/store, every size x (store, load, signed-load-to-64). The signed
// forms are the LDPSW/LDRSW sign-extension class.
TEST_F(Arm64MemoryDifferentialFuzz, ScalarLoadStoreImm) {
  Seed(0xE30A1ULL);
  int iters = 3000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    int size = Rnd() % 4;
    // opc: 00 store, 01 load, 10 signed load to 64-bit. Signed-load is not
    // defined for size=11 (64-bit), so restrict it.
    int opc = Rnd() % (size == 0b11 ? 2 : 3);
    int rt = Rnd() % 31;
    int rn = Rnd() % 31;
    if (rn == rt) rn = (rn + 1) % 31;
    size_t scale = 1u << size;
    uint32_t imm12 = Rnd() % 32;

    SeedScratch();
    InitState in = RandomInit();
    PinBase(&in, rn, scale, /*tagged=*/false);

    uint32_t code[1] = {EncLoadStoreImm(size, opc, imm12, rn, rt)};
    std::string desc;
    Result r = RunDifferential(code, 1, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "ScalarLoadStoreImm: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}

// The regression guard for top-byte-ignore: the same accesses, but the base
// register carries a non-zero address tag the way a Scudo-allocated pointer
// does. Both tiers must strip bits [63:56] at the point of access.
TEST_F(Arm64MemoryDifferentialFuzz, TaggedBaseAddressIsIgnored) {
  Seed(0x7B1A5EULL);
  int iters = 3000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    int size = Rnd() % 4;
    int opc = Rnd() % (size == 0b11 ? 2 : 3);
    int rt = Rnd() % 31;
    int rn = Rnd() % 31;
    if (rn == rt) rn = (rn + 1) % 31;
    size_t scale = 1u << size;
    uint32_t imm12 = Rnd() % 32;

    SeedScratch();
    InitState in = RandomInit();
    PinBase(&in, rn, scale, /*tagged=*/true);

    uint32_t code[1] = {EncLoadStoreImm(size, opc, imm12, rn, rt)};
    std::string desc;
    Result r = RunDifferential(code, 1, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "TaggedBaseAddressIsIgnored: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}

// LDP/STP including LDPSW, whose sign-extension wedged a real Qt fill loop.
TEST_F(Arm64MemoryDifferentialFuzz, LoadStorePair) {
  Seed(0xBA12ULL);
  int iters = 3000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    // opc: 00 = 32-bit, 01 = LDPSW (load only), 10 = 64-bit.
    int opc = (Rnd() % 3);
    if (opc == 2) opc = 0b10;
    int l = (opc == 0b01) ? 1 : static_cast<int>(Rnd() & 1);
    int rt = Rnd() % 31;
    int rt2 = Rnd() % 31;
    int rn = Rnd() % 31;
    // A load with rt == rt2, or a base that is also a destination, is
    // CONSTRAINED UNPREDICTABLE; the tiers may legitimately differ.
    if (rt2 == rt) rt2 = (rt2 + 1) % 31;
    if (rn == rt || rn == rt2) rn = (rn + 2) % 31;
    if (rn == rt || rn == rt2) continue;
    size_t scale = (opc == 0b10) ? 8 : 4;
    int imm7 = static_cast<int>(Rnd() % 16) - 8;

    SeedScratch();
    InitState in = RandomInit();
    PinBase(&in, rn, scale, /*tagged=*/false);

    uint32_t code[1] = {EncLoadStorePair(opc, l, imm7, rt2, rn, rt)};
    std::string desc;
    Result r = RunDifferential(code, 1, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "LoadStorePair: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}

// Register-offset addressing, sweeping the extend option and the scale bit.
TEST_F(Arm64MemoryDifferentialFuzz, RegisterOffsetAddressing) {
  Seed(0x8E60FFULL);
  int iters = 3000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    int size = Rnd() % 4;
    int opc = Rnd() % 2;
    int rt = Rnd() % 31;
    int rn = Rnd() % 31;
    int rm = Rnd() % 31;
    if (rn == rt) rn = (rn + 1) % 31;
    if (rm == rn || rm == rt) rm = (rm + 2) % 31;
    if (rm == rn || rm == rt) continue;
    // option 011 = LSL/UXTX (64-bit index), 010 = UXTW, 110 = SXTW, 111 = SXTX.
    static const int kOptions[] = {0b010, 0b011, 0b110, 0b111};
    int option = kOptions[Rnd() % 4];
    int s = Rnd() & 1;
    size_t scale = 1u << size;

    SeedScratch();
    InitState in = RandomInit();
    PinBase(&in, rn, scale, /*tagged=*/false);
    // Keep the computed address in the window for every extend interpretation:
    // a small non-negative index means UXTW/SXTW/UXTX/SXTX all agree.
    uint64_t idx = Rnd() % 32;
    in.x[rm] = idx;

    uint32_t code[1] = {EncLoadStoreReg(size, opc, rm, option, s, rn, rt)};
    std::string desc;
    Result r = RunDifferential(code, 1, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "RegisterOffsetAddressing: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}

// Pre/post-index, which write the base register back as well as accessing
// memory -- both effects are compared.
TEST_F(Arm64MemoryDifferentialFuzz, PrePostIndexWriteback) {
  Seed(0x1D00ULL);
  int iters = 3000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    int size = Rnd() % 4;
    int opc = Rnd() % 2;
    int rt = Rnd() % 31;
    int rn = Rnd() % 31;
    // Writeback to the loaded register is CONSTRAINED UNPREDICTABLE.
    if (rn == rt) rn = (rn + 1) % 31;
    if (rn == rt) continue;
    int idx = (Rnd() & 1) ? 0b01 : 0b11;  // post / pre
    int imm9 = static_cast<int>(Rnd() % 64) - 32;
    size_t scale = 1u << size;

    SeedScratch();
    InitState in = RandomInit();
    PinBase(&in, rn, scale, /*tagged=*/false);

    uint32_t code[1] = {EncLoadStoreIdx(size, opc, imm9, idx, rn, rt)};
    std::string desc;
    Result r = RunDifferential(code, 1, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "PrePostIndexWriteback: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}

// SIMD/FP loads and stores (S, D and Q widths). A Q store writing only 8 bytes,
// or a D load leaving the upper half stale, shows up in the memory compare.
TEST_F(Arm64MemoryDifferentialFuzz, SimdLoadStoreImm) {
  Seed(0x51D0ULL);
  int iters = 3000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    // (size, opc-load, opc-store, scale) for S, D and Q.
    struct Form {
      int size;
      int load_opc;
      int store_opc;
      size_t scale;
    };
    static const Form kForms[] = {
        {0b10, 0b01, 0b00, 4},   // S
        {0b11, 0b01, 0b00, 8},   // D
        {0b00, 0b11, 0b10, 16},  // Q
    };
    const Form& f = kForms[Rnd() % 3];
    bool store = (Rnd() & 1) != 0;
    int rt = Rnd() % 32;
    int rn = Rnd() % 31;
    uint32_t imm12 = Rnd() % 16;

    SeedScratch();
    InitState in = RandomInit();
    PinBase(&in, rn, f.scale, /*tagged=*/false);

    uint32_t code[1] = {
        EncSimdLoadStoreImm(f.size, store ? f.store_opc : f.load_opc, imm12, rn, rt)};
    std::string desc;
    Result r = RunDifferential(code, 1, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "SimdLoadStoreImm: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}

// LSE atomics: the read-modify-write families plus SWP, across the
// acquire/release suffixes. SWP-decoded-as-LDADD is the historical bug here,
// and it is only visible because both the returned value and the memory are
// compared.
TEST_F(Arm64MemoryDifferentialFuzz, LseAtomics) {
  Seed(0xA701CULL);
  int iters = 3000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    int size = (Rnd() & 1) ? 0b11 : 0b10;  // 64-bit or 32-bit
    int o3 = (Rnd() % 4 == 0) ? 1 : 0;     // SWP a quarter of the time
    int opc = o3 ? 0b000 : static_cast<int>(Rnd() % 8);
    int a = Rnd() & 1;
    int r = Rnd() & 1;
    int rs = Rnd() % 31;
    int rt = Rnd() % 31;
    int rn = Rnd() % 31;
    if (rn == rt || rn == rs) rn = (rn + 3) % 31;
    if (rn == rt || rn == rs) continue;
    size_t scale = (size == 0b11) ? 8 : 4;

    SeedScratch();
    InitState in = RandomInit();
    PinBase(&in, rn, scale, /*tagged=*/false);

    uint32_t code[1] = {EncAtomic(size, a, r, rs, o3, opc, rn, rt)};
    std::string desc;
    Result res = RunDifferential(code, 1, in, &desc);
    if (res == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (res == kMatch) compared++;
  }
  fprintf(stderr, "LseAtomics: compared=%d/%d\n", compared, iters);
  // Not EXPECT_GT: if the lite tier declines every LSE atomic they all run on
  // the interpreter on-device and cannot be a lite miscompile source.
  fprintf(stderr, "LseAtomics: (declines are legitimate)\n");
}

// Multi-instruction regions mixing loads, stores and arithmetic. A single
// instruction cannot expose a register-mapping clobber where a spilled base or
// loaded value is overwritten by a later access.
TEST_F(Arm64MemoryDifferentialFuzz, MixedMemoryRegions) {
  Seed(0xBE6100ULL);
  int iters = 1500 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    constexpr int kN = 6;
    uint32_t code[kN];
    InitState in = RandomInit();
    SeedScratch();

    // One pinned base shared by the whole region, so every access is in bounds
    // regardless of what the arithmetic instructions do to other registers.
    int base_reg = 1 + (Rnd() % 5);
    PinBase(&in, base_reg, 8, /*tagged=*/(Rnd() % 4) == 0);

    for (int j = 0; j < kN; j++) {
      if (Rnd() % 3 == 0) {
        // ADD Xd, Xn, Xm on registers that are never used as a base.
        int rd = 10 + (Rnd() % 10);
        int rn = 10 + (Rnd() % 10);
        int rm = 10 + (Rnd() % 10);
        code[j] = 0x8B000000u | (static_cast<uint32_t>(rm) << 16) |
                  (static_cast<uint32_t>(rn) << 5) | static_cast<uint32_t>(rd);
      } else {
        int size = Rnd() % 4;
        int opc = Rnd() % 2;
        int rt = 10 + (Rnd() % 10);
        uint32_t imm12 = Rnd() % 16;
        code[j] = EncLoadStoreImm(size, opc, imm12, base_reg, rt);
      }
    }

    std::string desc;
    Result r = RunDifferential(code, kN, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "MixedMemoryRegions: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}

}  // namespace


// LDP/STP with WRITEBACK (pre-index and post-index), including SP as the base.
//
// Two coverage holes met here, which is why this shipped: LoadStorePair above
// only ever emits the signed-offset form (idx=0b010), never pre-index (0b011)
// or post-index (0b001); and every generator picks rn from Rnd()%31, so r31 --
// which for load/store means SP, not XZR -- is never a base. Together that left
// `stp x29,x30,[sp,#-0x20]!` / `ldp x29,x30,[sp],#0x20`, i.e. the standard
// AArch64 function prologue and epilogue, with no differential coverage at all.
//
// The writeback must update the base (SP included) and the access must use the
// pre-index address for 0b011 and the ORIGINAL base for 0b001. RunDifferential
// compares sp and memory, so a bad writeback or a wrong access address shows up.
uint32_t EncLoadStorePairIdx(int opc, int l, int idx, int imm7, int rt2, int rn, int rt) {
  return (static_cast<uint32_t>(opc) << 30) | (0b101u << 27) |
         (static_cast<uint32_t>(idx) << 23) | (static_cast<uint32_t>(l) << 22) |
         ((static_cast<uint32_t>(imm7) & 0x7F) << 15) | (static_cast<uint32_t>(rt2) << 10) |
         (static_cast<uint32_t>(rn) << 5) | static_cast<uint32_t>(rt);
}

TEST_F(Arm64MemoryDifferentialFuzz, EncodersMatchKnownPairIdxEncodings) {
  // Ground truth from a real libAemonPlayer prologue/epilogue.
  EXPECT_EQ(EncLoadStorePairIdx(0b10, 1, 0b001, 4, 30, 31, 29), 0xA8C27BFDu);   // ldp x29,x30,[sp],#0x20
  EXPECT_EQ(EncLoadStorePairIdx(0b10, 0, 0b011, -4, 30, 31, 29), 0xA9BE7BFDu);  // stp x29,x30,[sp,#-0x20]!
}

TEST_F(Arm64MemoryDifferentialFuzz, LoadStorePairWritebackAndSpBase) {
  Seed(0x59B45EULL);
  int iters = 3000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    // opc: 00 = 32-bit, 10 = 64-bit (01 = LDPSW is load-only, covered above).
    const int opc = (Rnd() & 1) ? 0b10 : 0b00;
    const int l = static_cast<int>(Rnd() & 1);
    const int idx = (Rnd() & 1) ? 0b011 : 0b001;  // pre / post index
    // Half the time use SP as the base -- the case real prologues use.
    const bool sp_base = (Rnd() & 1) != 0;
    int rn = sp_base ? 31 : static_cast<int>(Rnd() % 31);
    int rt = static_cast<int>(Rnd() % 31);
    int rt2 = static_cast<int>(Rnd() % 31);
    // A load with rt == rt2, or writeback into a loaded register, is
    // CONSTRAINED UNPREDICTABLE; skip those rather than compare them.
    if (l == 1 && rt2 == rt) continue;
    if (l == 1 && !sp_base && (rn == rt || rn == rt2)) continue;
    const size_t scale = (opc == 0b10) ? 8 : 4;
    const int imm7 = static_cast<int>(Rnd() % 16) - 8;

    SeedScratch();
    InitState in = RandomInit();
    // Point the base into the scratch window, aligned, with room either way.
    uint64_t base = (ScratchBase() + kWindowBase) & ~(static_cast<uint64_t>(scale) - 1);
    if (sp_base) {
      in.sp = base;
    } else {
      in.x[rn] = base;
    }

    uint32_t code[1] = {EncLoadStorePairIdx(opc, l, idx, imm7, rt2, rn, rt)};
    std::string desc;
    Result r = RunDifferential(code, 1, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << " sp_base=" << sp_base << " idx=" << idx << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "LoadStorePairWritebackAndSpBase: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}


// The real AArch64 function prologue as a REGION: an SP-writeback STP followed
// by instructions that READ the updated SP.
//
// The single-instruction test above proves the writeback value is right, but
// says nothing about whether later instructions in the SAME region observe it.
// That is the region-structural question, and it is exactly the shape every
// compiled function starts with:
//     stp x29, x30, [sp, #-0x20]!   ; SP -= 0x20, writeback
//     str x19, [sp, #0x10]          ; must use the NEW SP
//     mov x29, sp                   ; ADD x29, SP, #0 -- must read the NEW SP
TEST_F(Arm64MemoryDifferentialFuzz, FunctionPrologueRegionSpPropagation) {
  Seed(0x9401060EULL);
  int iters = 2000 * FuzzScale();
  int compared = 0;
  for (int i = 0; i < iters; i++) {
    static const uint32_t kPrologue[] = {
        0xA9BE7BFDu,  // stp x29, x30, [sp, #-0x20]!
        0xF9000BF3u,  // str x19, [sp, #0x10]
        0x910003FDu,  // mov x29, sp
    };
    uint32_t code[3] = {kPrologue[0], kPrologue[1], kPrologue[2]};

    SeedScratch();
    InitState in = RandomInit();
    // SP starts in the scratch window, 16-aligned, with room below for the
    // 0x20 pre-decrement.
    in.sp = (ScratchBase() + kWindowBase) & ~static_cast<uint64_t>(15);

    std::string desc;
    Result r = RunDifferential(code, 3, in, &desc);
    if (r == kDiverge) {
      ADD_FAILURE() << "iter " << i << ": " << desc;
      return;
    }
    if (r == kMatch) compared++;
  }
  fprintf(stderr, "FunctionPrologueRegionSpPropagation: compared=%d/%d\n", compared, iters);
  EXPECT_GT(compared, 0);
}

}  // namespace berberis
