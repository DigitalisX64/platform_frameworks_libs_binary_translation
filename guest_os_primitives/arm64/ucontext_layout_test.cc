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

// §L2 box (1): audit + runtime verification that the ucontext_t passed to a
// guest signal handler matches bionic's arm64 ABI byte-for-byte. The static
// CHECK_STRUCT_LAYOUT / CHECK_FIELD_LAYOUT inside guest_context_arch.h locks
// the type layout at compile time. This test additionally verifies that
// GuestContext::Save populates each field at the correct Bionic-spec byte
// offset, and that GuestContext::Restore round-trips the state back into
// CPUState (including the FPSIMD vregs that the handler reads from the
// _aarch64_ctx chain at the start of uc_mcontext.__reserved).
//
// References for the offsets used below:
//   bionic/libc/kernel/uapi/asm-arm64/asm/sigcontext.h (struct sigcontext)
//   bionic/libc/include/sys/ucontext.h                  (typedef struct ucontext)

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>

#include "berberis/guest_state/guest_state.h"

#include "guest_context_arch.h"

namespace berberis {

namespace {

// Bionic-spec offsets inside arm64 ucontext_t.
constexpr size_t kUcMcontextOff = 176;  // After 168B header + 8B align-to-16 pad.

// Bionic-spec offsets inside arm64 struct sigcontext (relative to uc_mcontext).
constexpr size_t kFaultAddressOff = 0;
constexpr size_t kRegsOff = 8;
constexpr size_t kSpOff = kRegsOff + 31 * 8;  // 256
constexpr size_t kPcOff = kSpOff + 8;          // 264
constexpr size_t kPstateOff = kPcOff + 8;      // 272
// __reserved is __aligned__(16); 280 is not 16-aligned so the compiler
// inserts 8 bytes of padding. The array itself starts at byte 288.
constexpr size_t kReservedOff = 288;

// Bionic-spec offsets inside the first chained _aarch64_ctx block.
constexpr size_t kFpsimdHeadMagicOff = 0;
constexpr size_t kFpsimdHeadSizeOff = 4;
constexpr size_t kFpsimdFpsrOff = 8;
constexpr size_t kFpsimdFpcrOff = 12;
constexpr size_t kFpsimdVregsOff = 16;
constexpr size_t kFpsimdContextSize = 528;
constexpr uint32_t kFpsimdMagic = 0x46508001U;

// Helper: byte-precise read at a known offset within an arbitrary-aligned
// buffer. We avoid reinterpret_cast to pointer-to-T for misaligned offsets.
template <typename T>
T ReadAt(const uint8_t* base, size_t off) {
  T value;
  std::memcpy(&value, base + off, sizeof(T));
  return value;
}

void FillCpuState(CPUState* cpu) {
  for (size_t i = 0; i < 31; ++i) {
    cpu->x[i] = 0xAA00ULL + i;
  }
  cpu->sp = 0xBB00BB00BB00BB00ULL;
  cpu->insn_addr = 0xCC00CC00CC00CC00ULL;
  cpu->flags = 0xABCDU;
  // FPSR bit 27 (QC, saturation) + bit 7 (IDC) — distinct, non-zero pattern
  // that maps onto bits we actually emulate.
  cpu->emulated_fpsr = (1U << 27) | (1U << 7);
  // FPCR bits 23-22 (RMode = round toward zero) + bit 24 (FZ) + bit 15
  // (default NaN) — a realistic non-zero rounding/control pattern.
  cpu->cached_fpcr = (0x3U << 22) | (1U << 24) | (1U << 15);
  for (size_t i = 0; i < 32; ++i) {
    const __uint128_t hi = (__uint128_t)(0xF1F2F3F4F5F6F7F8ULL + i) << 64;
    cpu->v[i] = hi | (0x1000 + i);
  }
}

}  // namespace

TEST(UContextLayout, SigcontextOffsetsMatchBionic) {
  GuestContext ctx;
  CPUState cpu{};
  FillCpuState(&cpu);

  ctx.Save(&cpu);

  const uint8_t* base = static_cast<const uint8_t*>(ctx.ptr());
  const uint8_t* mctx = base + kUcMcontextOff;

  EXPECT_EQ(ReadAt<uint64_t>(mctx, kFaultAddressOff), 0u)
      << "fault_address should be zeroed by Save's memset";

  for (size_t i = 0; i < 31; ++i) {
    EXPECT_EQ(ReadAt<uint64_t>(mctx, kRegsOff + i * 8), 0xAA00ULL + i)
        << "regs[" << i << "] mismatch at byte offset " << (kRegsOff + i * 8);
  }
  EXPECT_EQ(ReadAt<uint64_t>(mctx, kSpOff), 0xBB00BB00BB00BB00ULL);
  EXPECT_EQ(ReadAt<uint64_t>(mctx, kPcOff), 0xCC00CC00CC00CC00ULL);
  // pstate is uint64; CPUState::flags is uint16, zero-extended at the boundary.
  EXPECT_EQ(ReadAt<uint64_t>(mctx, kPstateOff), 0xABCDULL);
}

TEST(UContextLayout, FpsimdEmbeddedInReservedWithBionicMagic) {
  GuestContext ctx;
  CPUState cpu{};
  FillCpuState(&cpu);

  ctx.Save(&cpu);

  const uint8_t* base = static_cast<const uint8_t*>(ctx.ptr());
  const uint8_t* reserved = base + kUcMcontextOff + kReservedOff;

  // The first chained block in __reserved must be the FPSIMD context.
  EXPECT_EQ(ReadAt<uint32_t>(reserved, kFpsimdHeadMagicOff), kFpsimdMagic)
      << "FPSIMD context must be embedded at __reserved[0] per arm64 ABI; "
         "guest handlers walking the chain expect to find it here";
  EXPECT_EQ(ReadAt<uint32_t>(reserved, kFpsimdHeadSizeOff),
            static_cast<uint32_t>(kFpsimdContextSize));

  // FPSR/FPCR must be populated from the matching CPUState fields. Bionic
  // signal handlers reading uc_mcontext to inspect saturation (QC) or the
  // rounding mode expect these at offsets 8/12 inside the FPSIMD block.
  EXPECT_EQ(ReadAt<uint32_t>(reserved, kFpsimdFpsrOff), cpu.emulated_fpsr);
  EXPECT_EQ(ReadAt<uint32_t>(reserved, kFpsimdFpcrOff), cpu.cached_fpcr);

  for (size_t i = 0; i < 32; ++i) {
    __uint128_t got =
        ReadAt<__uint128_t>(reserved, kFpsimdVregsOff + i * 16);
    EXPECT_EQ(got, cpu.v[i]) << "vregs[" << i << "] mismatch in __reserved";
  }

  // Immediately after the FPSIMD block, expect the NULL terminator
  // {magic=0, size=0} so a handler walking the chain stops cleanly.
  EXPECT_EQ(ReadAt<uint32_t>(reserved, kFpsimdContextSize + 0), 0u)
      << "missing NULL terminator magic after FPSIMD block";
  EXPECT_EQ(ReadAt<uint32_t>(reserved, kFpsimdContextSize + 4), 0u)
      << "missing NULL terminator size after FPSIMD block";
}

TEST(UContextLayout, RestoreRoundTrip) {
  GuestContext ctx;
  CPUState orig{};
  FillCpuState(&orig);

  ctx.Save(&orig);

  CPUState restored{};
  // Pre-poison to ensure Restore writes everything we care about.
  std::memset(&restored, 0x5A, sizeof(restored));
  ctx.Restore(&restored);

  for (size_t i = 0; i < 31; ++i) {
    EXPECT_EQ(restored.x[i], orig.x[i]) << "x[" << i << "] round-trip";
  }
  EXPECT_EQ(restored.sp, orig.sp);
  EXPECT_EQ(restored.insn_addr, orig.insn_addr);
  EXPECT_EQ(restored.flags, orig.flags);
  EXPECT_EQ(restored.emulated_fpsr, orig.emulated_fpsr) << "fpsr round-trip";
  EXPECT_EQ(restored.cached_fpcr, orig.cached_fpcr) << "fpcr round-trip";
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(restored.v[i], orig.v[i]) << "v[" << i << "] round-trip";
  }
}

TEST(UContextLayout, RestoreHonorsHandlerWritesToReserved) {
  // Simulates a guest signal handler that modifies vregs by writing into the
  // FPSIMD block embedded in __reserved (the Bionic-spec location). The
  // subsequent Restore must reflect the handler's changes, not the original
  // Save'd values.
  GuestContext ctx;
  CPUState orig{};
  FillCpuState(&orig);
  ctx.Save(&orig);

  uint8_t* base = static_cast<uint8_t*>(ctx.ptr());
  uint8_t* reserved = base + kUcMcontextOff + kReservedOff;

  // Handler overwrites all 32 vregs with a different sentinel.
  for (size_t i = 0; i < 32; ++i) {
    __uint128_t modified =
        ((__uint128_t)(0x1234'5678'9ABC'DEF0ULL + i) << 64) | (0x9000 + i);
    std::memcpy(reserved + kFpsimdVregsOff + i * 16, &modified, 16);
  }
  // Handler also clears fpsr saturation (QC) — a common recovery pattern —
  // and flips fpcr's rounding mode to RN (00).
  const uint32_t handler_fpsr = (1U << 7);  // only IDC, QC cleared
  const uint32_t handler_fpcr = (1U << 24) | (1U << 15);  // FZ + DN, RMode=RN
  std::memcpy(reserved + kFpsimdFpsrOff, &handler_fpsr, sizeof(handler_fpsr));
  std::memcpy(reserved + kFpsimdFpcrOff, &handler_fpcr, sizeof(handler_fpcr));

  CPUState restored{};
  ctx.Restore(&restored);

  for (size_t i = 0; i < 32; ++i) {
    const __uint128_t expected =
        ((__uint128_t)(0x1234'5678'9ABC'DEF0ULL + i) << 64) | (0x9000 + i);
    EXPECT_EQ(restored.v[i], expected)
        << "Restore must read vregs from __reserved (the handler's writes), "
           "not from a private backing copy";
  }
  EXPECT_EQ(restored.emulated_fpsr, handler_fpsr)
      << "Restore must read fpsr from __reserved so handler writes propagate";
  EXPECT_EQ(restored.cached_fpcr, handler_fpcr)
      << "Restore must read fpcr from __reserved so handler writes propagate";
}

}  // namespace berberis
