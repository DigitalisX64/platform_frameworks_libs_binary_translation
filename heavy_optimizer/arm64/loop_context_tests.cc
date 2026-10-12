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
constexpr uint32_t kSubsX4X4Imm1 = 0xf1000484;     // subs x4, x4, #1
constexpr uint32_t kSubsX5X5Imm1 = 0xf10004a5;     // subs x5, x5, #1
constexpr uint32_t kAddX6X6X7 = 0x8b0700c6;        // add x6, x6, x7

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

}  // namespace

}  // namespace berberis
