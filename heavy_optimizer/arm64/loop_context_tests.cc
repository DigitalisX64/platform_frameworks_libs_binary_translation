// In-region loops through the whole heavy pipeline, for guest registers the
// loop guest context optimizer must leave in memory across a loop: vector
// registers accessed at 64 and 128 bits, and the upper half of a vector
// register accessed at its own offset.

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/heavy_optimizer/arm64/heavy_optimize_region.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_exec_region.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

constexpr uint32_t kAddV1V1V2_4S = 0x4ea28421;     // add v1.4s, v1.4s, v2.4s
constexpr uint32_t kAddV1V1V2_16B = 0x4e228421;    // add v1.16b, v1.16b, v2.16b
constexpr uint32_t kFaddD3D3D1 = 0x1e612863;       // fadd d3, d3, d1
constexpr uint32_t kPmull2V3V1V1 = 0x4e21e023;     // pmull2 v3.8h, v1.16b, v1.16b
constexpr uint32_t kSmullV4V1V2 = 0x0e22c024;      // smull v4.8h, v1.8b, v2.8b
constexpr uint32_t kSmull2V3V1V2 = 0x4e22c023;     // smull2 v3.8h, v1.16b, v2.16b
constexpr uint32_t kSubsX4X4Imm1 = 0xf1000484;     // subs x4, x4, #1
constexpr uint32_t kSubsX5X5Imm1 = 0xf10004a5;     // subs x5, x5, #1
constexpr uint32_t kAddX6X6X7 = 0x8b0700c6;        // add x6, x6, x7
constexpr uint32_t kLdrS1X0 = 0xbd400001;          // ldr s1, [x0]
constexpr uint32_t kLdrS2X0Imm4 = 0xbd400402;      // ldr s2, [x0, #4]
constexpr uint32_t kFaddS3S3S1 = 0x1e212863;       // fadd s3, s3, s1
constexpr uint32_t kFaddS3S3S2 = 0x1e222863;       // fadd s3, s3, s2
constexpr uint32_t kStrS3X1 = 0xbd000023;          // str s3, [x1]
constexpr uint32_t kStrS3X1Imm4 = 0xbd000423;      // str s3, [x1, #4]
constexpr uint32_t kAddX0X0Imm8 = 0x91002000;      // add x0, x0, #8
constexpr uint32_t kAddX1X1Imm8 = 0x91002021;      // add x1, x1, #8
constexpr uint32_t kLd1V1D1X0 = 0x4d408401;        // ld1 {v1.d}[1], [x0]
constexpr uint32_t kAddV3V1V2_16B = 0x4e228423;    // add v3.16b, v1.16b, v2.16b
constexpr uint32_t kLdrX9X0 = 0xf9400009;          // ldr x9, [x0]
constexpr uint32_t kAddX6X6X9 = 0x8b0900c6;        // add x6, x6, x9

// MOVZ Xd, #imm16.
constexpr uint32_t MovzX(uint8_t rd, uint16_t imm16) {
  return 0xD2800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// B.NE .+offset
constexpr uint32_t BneBy(int32_t offset) {
  return 0x54000001 | ((static_cast<uint32_t>(offset / 4) & 0x7FFFF) << 5);
}

class Arm64HeavyLoopContextTest : public ::testing::Test {
 protected:
  template <size_t N>
  void Run(const uint32_t (&code)[N]) {
    GuestAddr start = ToGuestAddr(code);
    GuestAddr end_pc = start + N * 4;
    state_.cpu.insn_addr = start;
    MachineCode machine_code;
    bool has_backedge = false;
    auto [stop, ok, n] = HeavyOptimizeRegion(
        start, &machine_code, HeavyOptimizeParams{.end_pc = end_pc}, &has_backedge);
    ASSERT_TRUE(ok);
    ASSERT_EQ(stop, end_pc);
    ASSERT_TRUE(has_backedge);
    TranslationCache::GetInstance()->InvalidateGuestRange(start, end_pc + 4);
    ScopedExecRegion exec(&machine_code);
    TestingRunGeneratedCode(&state_, exec.get(), end_pc);
    EXPECT_EQ(state_.cpu.insn_addr, end_pc);
  }

  static __uint128_t SplatU32(uint32_t value) {
    __uint128_t result = 0;
    for (int i = 0; i < 4; i++) {
      result |= static_cast<__uint128_t>(value) << (i * 32);
    }
    return result;
  }

  ThreadState state_{};
};

// v1 is read at 64 bits by FADD and at 128 bits by the vector ADD inside the
// inner loop; the outer loop's own registers (x6, x7) compete for the same
// host registers.
TEST_F(Arm64HeavyLoopContextTest, MixedWidthVectorRegisterInLoopNest) {
  static const uint32_t code[] = {
      MovzX(4, 3),     // [0] outer count
      MovzX(5, 4),     // [1] outer top: inner count
      kAddV1V1V2_4S,   // [2] inner top
      kFaddD3D3D1,     // [3]
      kSubsX5X5Imm1,   // [4]
      BneBy(-12),      // [5] -> [2]
      kAddX6X6X7,      // [6]
      kSubsX4X4Imm1,   // [7]
      BneBy(-28),      // [8] -> [1]
  };
  state_.cpu.v[1] = 0;
  state_.cpu.v[2] = SplatU32(1);
  state_.cpu.v[3] = 0;
  state_.cpu.x[6] = 100;
  state_.cpu.x[7] = 5;
  Run(code);

  __uint128_t v1 = 0;
  double d3 = 0;
  for (int i = 0; i < 12; i++) {
    v1 += SplatU32(1);
    double d1;
    uint64_t low = static_cast<uint64_t>(v1);
    memcpy(&d1, &low, sizeof(d1));
    d3 += d1;
  }
  uint64_t d3_bits;
  memcpy(&d3_bits, &d3, sizeof(d3_bits));
  EXPECT_TRUE(state_.cpu.v[1] == v1);
  EXPECT_EQ(static_cast<uint64_t>(state_.cpu.v[3]), d3_bits);
  EXPECT_EQ(static_cast<uint64_t>(state_.cpu.v[3] >> 64), 0u);
  EXPECT_EQ(state_.cpu.x[4], 0u);
  EXPECT_EQ(state_.cpu.x[5], 0u);
  EXPECT_EQ(state_.cpu.x[6], 115u);
}

// PMULL2 reads the upper half of v1 at its own offset while the vector ADD
// rewrites all of v1: each iteration must see the bytes the ADD just wrote.
TEST_F(Arm64HeavyLoopContextTest, UpperHalfReadSeesFullWidthWrite) {
  static const uint32_t code[] = {
      kAddV1V1V2_16B,  // [0] loop top: v1.16b += 1
      kPmull2V3V1V1,   // [1] v3.8h = polynomial square of bytes 8..15 of v1
      kSubsX5X5Imm1,   // [2]
      BneBy(-12),      // [3] -> [0]
  };
  state_.cpu.v[1] = 0;
  __uint128_t ones = 0;
  for (int i = 0; i < 16; i++) {
    ones |= static_cast<__uint128_t>(1) << (i * 8);
  }
  state_.cpu.v[2] = ones;
  state_.cpu.v[3] = 0;
  state_.cpu.x[5] = 2;
  Run(code);

  // v1 bytes are 2; the carry-less square of 2 is 4 in every 16-bit lane.
  __uint128_t expected_v3 = 0;
  for (int i = 0; i < 8; i++) {
    expected_v3 |= static_cast<__uint128_t>(4) << (i * 16);
  }
  EXPECT_TRUE(state_.cpu.v[1] == ones * 2);
  EXPECT_TRUE(state_.cpu.v[3] == expected_v3);
}

// SMULL and SMULL2 widen v1 and v2 without changing them: v1 and v2 keep
// their values across iterations while the widened copies are multiplied.
TEST_F(Arm64HeavyLoopContextTest, WideningMultiplyLeavesSourcesIntact) {
  static const uint32_t code[] = {
      kAddV1V1V2_16B,  // [0] loop top: v1.16b += v2.16b
      kSmull2V3V1V2,   // [1] v3.8h = v1.b[8..15] * v2.b[8..15]
      kSmullV4V1V2,    // [2] v4.8h = v1.b[0..7] * v2.b[0..7]
      kSubsX5X5Imm1,   // [3]
      BneBy(-16),      // [4] -> [0]
  };
  uint8_t v1_bytes[16];
  uint8_t v2_bytes[16];
  for (int i = 0; i < 16; i++) {
    v1_bytes[i] = static_cast<uint8_t>(i * 13);
    v2_bytes[i] = static_cast<uint8_t>(i * 37 - 100);
  }
  memcpy(&state_.cpu.v[1], v1_bytes, sizeof(v1_bytes));
  memcpy(&state_.cpu.v[2], v2_bytes, sizeof(v2_bytes));
  state_.cpu.v[3] = 0;
  state_.cpu.v[4] = 0;
  state_.cpu.x[5] = 3;
  Run(code);

  for (int i = 0; i < 16; i++) {
    v1_bytes[i] = static_cast<uint8_t>(v1_bytes[i] + 3 * v2_bytes[i]);
  }
  int16_t expected_v3[8];
  int16_t expected_v4[8];
  for (int i = 0; i < 8; i++) {
    expected_v4[i] = static_cast<int16_t>(static_cast<int8_t>(v1_bytes[i]) *
                                          static_cast<int8_t>(v2_bytes[i]));
    expected_v3[i] = static_cast<int16_t>(static_cast<int8_t>(v1_bytes[i + 8]) *
                                          static_cast<int8_t>(v2_bytes[i + 8]));
  }
  uint8_t actual_v1[16];
  uint8_t actual_v2[16];
  int16_t actual_v3[8];
  int16_t actual_v4[8];
  memcpy(actual_v1, &state_.cpu.v[1], sizeof(actual_v1));
  memcpy(actual_v2, &state_.cpu.v[2], sizeof(actual_v2));
  memcpy(actual_v3, &state_.cpu.v[3], sizeof(actual_v3));
  memcpy(actual_v4, &state_.cpu.v[4], sizeof(actual_v4));
  EXPECT_EQ(memcmp(actual_v1, v1_bytes, sizeof(actual_v1)), 0);
  EXPECT_EQ(memcmp(actual_v2, v2_bytes, sizeof(actual_v2)), 0);
  EXPECT_EQ(memcmp(actual_v3, expected_v3, sizeof(actual_v3)), 0);
  EXPECT_EQ(memcmp(actual_v4, expected_v4, sizeof(actual_v4)), 0);
}

// Guest loads and stores end their host basic block (the fault recovery
// split), and guest-register reads are forwarded across those splits: the
// scalar accumulator s3 is written, stored to memory and read again between
// guest loads. v3 is accessed at 64 and 128 bits, so the loop pass leaves it
// in memory and only the forwarding carries it.
TEST_F(Arm64HeavyLoopContextTest, ScalarChainAcrossGuestMemoryAccesses) {
  static const uint32_t code[] = {
      kLdrS1X0,       // [0] loop top
      kFaddS3S3S1,    // [1]
      kStrS3X1,       // [2]
      kLdrS2X0Imm4,   // [3]
      kFaddS3S3S2,    // [4]
      kStrS3X1Imm4,   // [5]
      kAddX0X0Imm8,   // [6]
      kAddX1X1Imm8,   // [7]
      kSubsX5X5Imm1,  // [8]
      BneBy(-36),     // [9] -> [0]
  };
  float in[8] = {1.5f, 2.25f, -3.0f, 4.0f, 0.125f, 6.0f, -7.5f, 8.0f};
  float out[8] = {};
  state_.cpu.x[0] = ToGuestAddr(in);
  state_.cpu.x[1] = ToGuestAddr(out);
  state_.cpu.x[5] = 4;
  float s3 = 0.5f;
  uint32_t s3_bits;
  memcpy(&s3_bits, &s3, sizeof(s3_bits));
  // Garbage above the scalar: the first FADD's write must zero it.
  state_.cpu.v[3] = (static_cast<__uint128_t>(~uint64_t{0}) << 64) | s3_bits;
  Run(code);

  float expected_out[8];
  for (int i = 0; i < 8; i++) {
    s3 += in[i];
    expected_out[i] = s3;
  }
  memcpy(&s3_bits, &s3, sizeof(s3_bits));
  EXPECT_EQ(memcmp(out, expected_out, sizeof(out)), 0);
  EXPECT_TRUE(state_.cpu.v[3] == s3_bits);
  EXPECT_EQ(state_.cpu.x[0], ToGuestAddr(in + 8));
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(out + 8));
}

// A full-width write of v1, then a guest load into its upper half, then a
// full-width read: the read must see the loaded half, not the value the
// full-width write left in a host register before the split.
TEST_F(Arm64HeavyLoopContextTest, LaneLoadAfterFullWidthWriteAcrossSplit) {
  static const uint32_t code[] = {
      kAddV1V1V2_16B,  // [0] loop top: v1.16b += 1
      kLd1V1D1X0,      // [1] v1.d[1] = *x0
      kAddV3V1V2_16B,  // [2] v3.16b = v1.16b + 1
      kLdrX9X0,        // [3]
      kAddX6X6X9,      // [4]
      kSubsX5X5Imm1,   // [5]
      BneBy(-24),      // [6] -> [0]
  };
  uint64_t lane = 0x0807060504030201;
  __uint128_t ones = 0;
  for (int i = 0; i < 16; i++) {
    ones |= static_cast<__uint128_t>(1) << (i * 8);
  }
  state_.cpu.x[0] = ToGuestAddr(&lane);
  state_.cpu.x[5] = 3;
  state_.cpu.x[6] = 0;
  state_.cpu.v[1] = 0;
  state_.cpu.v[2] = ones;
  Run(code);

  // Bytes never carry here, so the byte-wise adds are plain additions.
  __uint128_t expected_v1 = (static_cast<__uint128_t>(lane) << 64) | uint64_t{0x0303030303030303};
  EXPECT_TRUE(state_.cpu.v[1] == expected_v1);
  EXPECT_TRUE(state_.cpu.v[3] == expected_v1 + ones);
  EXPECT_EQ(state_.cpu.x[6], 3 * lane);
}

// A full-width vector write, a guest load, then a 64-bit read of the same
// register: the read sees the low half the vector write produced.
TEST_F(Arm64HeavyLoopContextTest, VectorWriteThenScalarReadAcrossSplit) {
  static const uint32_t code[] = {
      kAddV1V1V2_4S,   // [0] loop top
      kLdrX9X0,        // [1]
      kFaddD3D3D1,     // [2]
      kAddX6X6X9,      // [3]
      kSubsX5X5Imm1,   // [4]
      BneBy(-20),      // [5] -> [0]
  };
  uint64_t value = 7;
  state_.cpu.x[0] = ToGuestAddr(&value);
  state_.cpu.x[5] = 5;
  state_.cpu.x[6] = 0;
  state_.cpu.v[1] = 0;
  state_.cpu.v[2] = SplatU32(1);
  state_.cpu.v[3] = 0;
  Run(code);

  __uint128_t v1 = 0;
  double d3 = 0;
  for (int i = 0; i < 5; i++) {
    v1 += SplatU32(1);
    double d1;
    uint64_t low = static_cast<uint64_t>(v1);
    memcpy(&d1, &low, sizeof(d1));
    d3 += d1;
  }
  uint64_t d3_bits;
  memcpy(&d3_bits, &d3, sizeof(d3_bits));
  EXPECT_TRUE(state_.cpu.v[1] == v1);
  EXPECT_TRUE(state_.cpu.v[3] == d3_bits);
  EXPECT_EQ(state_.cpu.x[6], 35u);
}

}  // namespace

}  // namespace berberis
