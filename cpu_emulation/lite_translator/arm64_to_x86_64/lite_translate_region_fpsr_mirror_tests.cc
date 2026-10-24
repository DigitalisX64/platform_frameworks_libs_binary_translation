// The MXCSR -> FPSR mirror at a lite region's exits: regions whose SIMD/FP
// instructions only move data cannot raise an FP exception and skip it; a
// region with FP arithmetic keeps it and reports the exception in FPSR.

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <xmmintrin.h>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

constexpr uint32_t kLdrQ0X0 = 0x3dc00000;          // ldr q0, [x0]
constexpr uint32_t kStrQ0X1 = 0x3d800020;          // str q0, [x1]
constexpr uint32_t kLdpQ2Q3X0 = 0xad400c02;        // ldp q2, q3, [x0]
constexpr uint32_t kFmovD1X2 = 0x9e670041;         // fmov d1, x2
constexpr uint32_t kFmovX3D1 = 0x9e660023;         // fmov x3, d1
constexpr uint32_t kExtV2 = 0x6e004002;            // ext v2.16b, v0.16b, v0.16b, #8
constexpr uint32_t kZip1V3 = 0x4e803803;           // zip1 v3.4s, v0.4s, v0.4s
constexpr uint32_t kDupV4 = 0x4e0c0404;            // dup v4.4s, v0.s[1]
constexpr uint32_t kMoviV5 = 0x4f000425;           // movi v5.4s, #1
constexpr uint32_t kFmovS6Imm = 0x1e2e1006;        // fmov s6, #1.0
constexpr uint32_t kFabsD7D1 = 0x1e60c027;         // fabs d7, d1
constexpr uint32_t kFnegS6S6 = 0x1e2140c6;         // fneg s6, s6
constexpr uint32_t kFdivS0S0S1 = 0x1e211800;       // fdiv s0, s0, s1
constexpr uint32_t kAddX4X4Imm1 = 0x91000484;      // add x4, x4, #1

// FPSR.DZC and the MXCSR exception flags.
constexpr uint64_t kFpsrDzc = 1u << 1;
constexpr uint32_t kMxcsrFlags = 0x3f;
constexpr uint32_t kMxcsrPe = 1u << 5;

// stmxcsr [rsp], the mirror's first instruction.
constexpr uint8_t kStmxcsrRsp[] = {0x0f, 0xae, 0x1c, 0x24};

class Arm64LiteFpsrMirrorTest : public ::testing::Test {
 protected:
  template <size_t N>
  HostCode Translate(const uint32_t (&code)[N], bool* has_mirror) {
    GuestAddr start = ToGuestAddr(code);
    MachineCode machine_code;
    auto [success, stop_pc] = TryLiteTranslateRegion(
        start, &machine_code, LiteTranslateParams{.end_pc = start + N * 4, .allow_dispatch = false});
    EXPECT_TRUE(success);
    EXPECT_EQ(stop_pc, start + N * 4);
    const uint8_t* bytes = machine_code.AddrAs<uint8_t>(0);
    const uint32_t size = machine_code.install_size();
    *has_mirror = false;
    for (uint32_t i = 0; i + sizeof(kStmxcsrRsp) <= size; ++i) {
      if (memcmp(bytes + i, kStmxcsrRsp, sizeof(kStmxcsrRsp)) == 0) {
        *has_mirror = true;
      }
    }
    return AsHostCode(GetDefaultCodePoolInstance()->Add(&machine_code));
  }

  void Run(HostCode host_code, GuestAddr stop_pc, uint32_t mxcsr_flags) {
    _mm_setcsr((_mm_getcsr() & ~kMxcsrFlags) | mxcsr_flags);
    TestingRunGeneratedCode(&state_, host_code, stop_pc);
    _mm_setcsr(_mm_getcsr() & ~kMxcsrFlags);
    EXPECT_EQ(state_.cpu.insn_addr, stop_pc);
  }

  ThreadState state_{};
};

TEST_F(Arm64LiteFpsrMirrorTest, DataMoveOnlyRegionSkipsMirror) {
  static const uint32_t code[] = {kLdrQ0X0, kLdpQ2Q3X0, kStrQ0X1,   kFmovD1X2,
                                  kFmovX3D1, kExtV2,    kZip1V3,    kDupV4,
                                  kMoviV5,  kFmovS6Imm, kFabsD7D1,  kFnegS6S6,
                                  kAddX4X4Imm1};
  bool has_mirror = true;
  HostCode host_code = Translate(code, &has_mirror);
  EXPECT_FALSE(has_mirror);

  alignas(16) static uint8_t src[32];
  alignas(16) static uint8_t dst[16];
  for (int i = 0; i < 32; ++i) {
    src[i] = static_cast<uint8_t>(i + 1);
  }
  memset(dst, 0, sizeof(dst));
  state_.cpu.x[0] = ToGuestAddr(src);
  state_.cpu.x[1] = ToGuestAddr(dst);
  state_.cpu.x[2] = 0xbff0'0000'0000'0000ULL;  // -1.0
  state_.cpu.x[4] = 41;
  state_.cpu.emulated_fpsr = 0;
  // A flag left in MXCSR by host code reaches FPSR only through a mirror.
  Run(host_code, ToGuestAddr(&code[13]), kMxcsrPe);

  EXPECT_EQ(memcmp(dst, src, sizeof(dst)), 0);
  EXPECT_EQ(state_.cpu.x[3], 0xbff0'0000'0000'0000ULL);
  EXPECT_EQ(state_.cpu.x[4], 42u);
  uint64_t d7;
  memcpy(&d7, &state_.cpu.v[7], sizeof(d7));
  EXPECT_EQ(d7, 0x3ff0'0000'0000'0000ULL);  // fabs(-1.0)
  uint32_t s6;
  memcpy(&s6, &state_.cpu.v[6], sizeof(s6));
  EXPECT_EQ(s6, 0xbf80'0000u);  // -1.0f
  EXPECT_EQ(state_.cpu.emulated_fpsr, 0u);
}

TEST_F(Arm64LiteFpsrMirrorTest, ArithmeticRegionMirrorsExceptionIntoFpsr) {
  static const uint32_t code[] = {kLdrQ0X0, kFdivS0S0S1, kAddX4X4Imm1};
  bool has_mirror = false;
  HostCode host_code = Translate(code, &has_mirror);
  EXPECT_TRUE(has_mirror);

  alignas(16) static float src[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  state_.cpu.x[0] = ToGuestAddr(src);
  memset(&state_.cpu.v[1], 0, sizeof(state_.cpu.v[1]));  // s1 = +0.0
  state_.cpu.emulated_fpsr = 0;
  Run(host_code, ToGuestAddr(&code[3]), 0);

  EXPECT_EQ(state_.cpu.emulated_fpsr & kFpsrDzc, kFpsrDzc);
}

TEST_F(Arm64LiteFpsrMirrorTest, DataMoveBeforeArithmeticKeepsMirror) {
  static const uint32_t code[] = {kLdrQ0X0, kStrQ0X1, kFdivS0S0S1};
  bool has_mirror = false;
  Translate(code, &has_mirror);
  EXPECT_TRUE(has_mirror);
}

TEST_F(Arm64LiteFpsrMirrorTest, ConversionIsNotTreatedAsDataMove) {
  constexpr uint32_t kScvtfD1X2 = 0x9e620041;  // scvtf d1, x2
  static const uint32_t code[] = {kFmovD1X2, kScvtfD1X2};
  bool has_mirror = false;
  Translate(code, &has_mirror);
  EXPECT_TRUE(has_mirror);
}

}  // namespace

}  // namespace berberis
