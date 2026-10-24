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

// Direct InterpretInsn golden-value coverage for interpreter-ONLY ARM64
// instruction families -- families the lite/heavy JIT bails on, so the
// JIT-vs-interpreter differential fuzzers (which use the interpreter as the
// oracle) structurally cannot exercise them. Each encoding is produced by the
// assembler (clang -target aarch64 + llvm-objdump); each golden is computed by
// an independent reference implementation of the ARM ARM semantics (integer
// reference math; struct-packed IEEE-754 for the FP families).
//
// Families: CRC32/CRC32C, TBL/TBX, ZIP/UZP/TRN permutes, across-lane
// reductions (ADDV/SMAXV/UMINV/FMAXV/...), pairwise (ADDP/SMAXP/FADDP,
// vector+scalar), SQDMULH/SQRDMULH, AES (AESE/AESD/AESMC/AESIMC), SHA-1/SHA-256,
// FCADD/FCMLA, SDOT/UDOT dot products, FCVTXN narrowing convert.

#include "gtest/gtest.h"

#include <cstdint>
#include <utility>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/arm64/interpreter.h"

namespace berberis {
namespace {

// Interpreter-only test harness. Mirrors the InterpretInsn path used by the
// lite-translator exec tests: a zeroed ThreadState (value-initialized, so the
// TLS/CPUState block starts clean), write the 32-bit instruction word, point
// insn_addr at it, and single-step the interpreter.
class Arm64InterpreterOnlyTest : public ::testing::Test {
 protected:
  void Interpret(uint32_t insn) {
    insn_ = insn;
    state_.cpu.insn_addr = ToGuestAddr(&insn_);
    InterpretInsn(&state_);
  }

  void SetVReg(int idx, uint64_t lo, uint64_t hi) {
    state_.cpu.v[idx] =
        (static_cast<__uint128_t>(hi) << 64) | static_cast<__uint128_t>(lo);
  }

  std::pair<uint64_t, uint64_t> GetVReg(int idx) const {
    __uint128_t v = state_.cpu.v[idx];
    return {static_cast<uint64_t>(v), static_cast<uint64_t>(v >> 64)};
  }

  ThreadState state_{};
  uint32_t insn_ = 0;
};

#define EXPECT_VREG(idx, lo, hi)                 \
  do {                                           \
    auto vreg_pair = GetVReg(idx);               \
    EXPECT_EQ(vreg_pair.first, (lo));            \
    EXPECT_EQ(vreg_pair.second, (hi));           \
  } while (0)


// crc32b w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32b) {
    state_.cpu.x[1] = 0x0000000012345678ULL;
    state_.cpu.x[2] = 0x00000000000000abULL;
    Interpret(0x1ac24020U);  // crc32b w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x000000001fc8b738ULL);
}

// crc32b w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32b_zero) {
    state_.cpu.x[1] = 0x0000000000000000ULL;
    state_.cpu.x[2] = 0x0000000000000000ULL;
    Interpret(0x1ac24020U);  // crc32b w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32b w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32b_ones) {
    state_.cpu.x[1] = 0x00000000ffffffffULL;
    state_.cpu.x[2] = 0x00000000000000ffULL;
    Interpret(0x1ac24020U);  // crc32b w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000ffffffULL);
}

// crc32h w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32h) {
    state_.cpu.x[1] = 0x0000000012345678ULL;
    state_.cpu.x[2] = 0x000000000000beefULL;
    Interpret(0x1ac24420U);  // crc32h w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000090d164a3ULL);
}

// crc32h w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32h_zero) {
    state_.cpu.x[1] = 0x0000000000000000ULL;
    state_.cpu.x[2] = 0x0000000000000000ULL;
    Interpret(0x1ac24420U);  // crc32h w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32h w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32h_ones) {
    state_.cpu.x[1] = 0x00000000ffffffffULL;
    state_.cpu.x[2] = 0x000000000000ffffULL;
    Interpret(0x1ac24420U);  // crc32h w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x000000000000ffffULL);
}

// crc32w w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32w) {
    state_.cpu.x[1] = 0x0000000012345678ULL;
    state_.cpu.x[2] = 0x00000000deadbeefULL;
    Interpret(0x1ac24820U);  // crc32w w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x00000000b537e7cdULL);
}

// crc32w w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32w_zero) {
    state_.cpu.x[1] = 0x0000000000000000ULL;
    state_.cpu.x[2] = 0x0000000000000000ULL;
    Interpret(0x1ac24820U);  // crc32w w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32w w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32w_ones) {
    state_.cpu.x[1] = 0x00000000ffffffffULL;
    state_.cpu.x[2] = 0x00000000ffffffffULL;
    Interpret(0x1ac24820U);  // crc32w w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32x w0, w1, x2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32x) {
    state_.cpu.x[1] = 0x0000000012345678ULL;
    state_.cpu.x[2] = 0x0123456789abcdefULL;
    Interpret(0x9ac24c20U);  // crc32x w0, w1, x2
    EXPECT_EQ(state_.cpu.x[0], 0x000000009b62eadfULL);
}

// crc32x w0, w1, x2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32x_zero) {
    state_.cpu.x[1] = 0x0000000000000000ULL;
    state_.cpu.x[2] = 0x0000000000000000ULL;
    Interpret(0x9ac24c20U);  // crc32x w0, w1, x2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32x w0, w1, x2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32x_ones) {
    state_.cpu.x[1] = 0x00000000ffffffffULL;
    state_.cpu.x[2] = 0xffffffffffffffffULL;
    Interpret(0x9ac24c20U);  // crc32x w0, w1, x2
    EXPECT_EQ(state_.cpu.x[0], 0x00000000debb20e3ULL);
}

// crc32cb w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cb) {
    state_.cpu.x[1] = 0x0000000012345678ULL;
    state_.cpu.x[2] = 0x00000000000000abULL;
    Interpret(0x1ac25020U);  // crc32cb w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x00000000c0912609ULL);
}

// crc32cb w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cb_zero) {
    state_.cpu.x[1] = 0x0000000000000000ULL;
    state_.cpu.x[2] = 0x0000000000000000ULL;
    Interpret(0x1ac25020U);  // crc32cb w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32cb w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cb_ones) {
    state_.cpu.x[1] = 0x00000000ffffffffULL;
    state_.cpu.x[2] = 0x00000000000000ffULL;
    Interpret(0x1ac25020U);  // crc32cb w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000ffffffULL);
}

// crc32ch w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32ch) {
    state_.cpu.x[1] = 0x0000000012345678ULL;
    state_.cpu.x[2] = 0x000000000000beefULL;
    Interpret(0x1ac25420U);  // crc32ch w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x00000000d78220dcULL);
}

// crc32ch w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32ch_zero) {
    state_.cpu.x[1] = 0x0000000000000000ULL;
    state_.cpu.x[2] = 0x0000000000000000ULL;
    Interpret(0x1ac25420U);  // crc32ch w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32ch w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32ch_ones) {
    state_.cpu.x[1] = 0x00000000ffffffffULL;
    state_.cpu.x[2] = 0x000000000000ffffULL;
    Interpret(0x1ac25420U);  // crc32ch w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x000000000000ffffULL);
}

// crc32cw w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cw) {
    state_.cpu.x[1] = 0x0000000012345678ULL;
    state_.cpu.x[2] = 0x00000000deadbeefULL;
    Interpret(0x1ac25820U);  // crc32cw w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x00000000f3ed4b20ULL);
}

// crc32cw w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cw_zero) {
    state_.cpu.x[1] = 0x0000000000000000ULL;
    state_.cpu.x[2] = 0x0000000000000000ULL;
    Interpret(0x1ac25820U);  // crc32cw w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32cw w0, w1, w2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cw_ones) {
    state_.cpu.x[1] = 0x00000000ffffffffULL;
    state_.cpu.x[2] = 0x00000000ffffffffULL;
    Interpret(0x1ac25820U);  // crc32cw w0, w1, w2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32cx w0, w1, x2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cx) {
    state_.cpu.x[1] = 0x0000000012345678ULL;
    state_.cpu.x[2] = 0x0123456789abcdefULL;
    Interpret(0x9ac25c20U);  // crc32cx w0, w1, x2
    EXPECT_EQ(state_.cpu.x[0], 0x00000000a3d207beULL);
}

// crc32cx w0, w1, x2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cx_zero) {
    state_.cpu.x[1] = 0x0000000000000000ULL;
    state_.cpu.x[2] = 0x0000000000000000ULL;
    Interpret(0x9ac25c20U);  // crc32cx w0, w1, x2
    EXPECT_EQ(state_.cpu.x[0], 0x0000000000000000ULL);
}

// crc32cx w0, w1, x2
TEST_F(Arm64InterpreterOnlyTest, Crc32_crc32cx_ones) {
    state_.cpu.x[1] = 0x00000000ffffffffULL;
    state_.cpu.x[2] = 0xffffffffffffffffULL;
    Interpret(0x9ac25c20U);  // crc32cx w0, w1, x2
    EXPECT_EQ(state_.cpu.x[0], 0x00000000b798b438ULL);
}

// tbl v0.8b, {v1.16b}, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Tbl_1reg_8b) {
    SetVReg(1, 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL);
    SetVReg(2, 0x0706050403020100ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x0e020020U);  // tbl v0.8b, {v1.16b}, v2.8b
    EXPECT_VREG(0, 0x1716151413121110ULL, 0x0000000000000000ULL);
}

// tbl v0.16b, {v1.16b}, v2.16b
TEST_F(Arm64InterpreterOnlyTest, Tbl_1reg_16b_oob) {
    SetVReg(1, 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL);
    SetVReg(2, 0x0903ff11100f0500ULL, 0x070e0c0a08040201ULL);
    SetVReg(0, 0xaaaaaaaaaaaaaaaaULL, 0xaaaaaaaaaaaaaaaaULL);
    Interpret(0x4e020020U);  // tbl v0.16b, {v1.16b}, v2.16b
    EXPECT_VREG(0, 0x19130000001f1510ULL, 0x171e1c1a18141211ULL);
}

// tbl v0.8b, {v1.16b, v2.16b}, v3.8b
TEST_F(Arm64InterpreterOnlyTest, Tbl_2reg_8b) {
    SetVReg(1, 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL);
    SetVReg(2, 0x2726252423222120ULL, 0x2f2e2d2c2b2a2928ULL);
    SetVReg(3, 0x110f1405201f1000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x0e032020U);  // tbl v0.8b, {v1.16b, v2.16b}, v3.8b
    EXPECT_VREG(0, 0x211f2415002f2010ULL, 0x0000000000000000ULL);
}

// tbl v0.16b, {v1.16b, v2.16b, v3.16b}, v4.16b
TEST_F(Arm64InterpreterOnlyTest, Tbl_3reg_16b) {
    SetVReg(1, 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL);
    SetVReg(2, 0x2726252423222120ULL, 0x2f2e2d2c2b2a2928ULL);
    SetVReg(3, 0x3736353433323130ULL, 0x3f3e3d3c3b3a3938ULL);
    SetVReg(4, 0x14280a302f201000ULL, 0x2107060504030201ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e044020U);  // tbl v0.16b, {v1.16b, v2.16b, v3.16b}, v4.16b
    EXPECT_VREG(0, 0x24381a003f302010ULL, 0x3117161514131211ULL);
}

// tbl v0.8b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.8b
TEST_F(Arm64InterpreterOnlyTest, Tbl_4reg_8b) {
    SetVReg(1, 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL);
    SetVReg(2, 0x2726252423222120ULL, 0x2f2e2d2c2b2a2928ULL);
    SetVReg(3, 0x3736353433323130ULL, 0x3f3e3d3c3b3a3938ULL);
    SetVReg(4, 0x4746454443424140ULL, 0x4f4e4d4c4b4a4948ULL);
    SetVReg(5, 0x370a403f30201000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x0e056020U);  // tbl v0.8b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.8b
    EXPECT_VREG(0, 0x471a004f40302010ULL, 0x0000000000000000ULL);
}

// tbx v0.8b, {v1.16b}, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Tbx_1reg_8b_keep) {
    SetVReg(1, 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL);
    SetVReg(2, 0x04ff030211100100ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x8786858483828180ULL, 0x8f8e8d8c8b8a8988ULL);
    Interpret(0x0e021020U);  // tbx v0.8b, {v1.16b}, v2.8b
    EXPECT_VREG(0, 0x1486131283821110ULL, 0x0000000000000000ULL);
}

// tbx v0.16b, {v1.16b, v2.16b}, v3.16b
TEST_F(Arm64InterpreterOnlyTest, Tbx_2reg_16b_keep) {
    SetVReg(1, 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL);
    SetVReg(2, 0x2726252423222120ULL, 0x2f2e2d2c2b2a2928ULL);
    SetVReg(3, 0x110f140521201f00ULL, 0xff07060504030201ULL);
    SetVReg(0, 0x9796959493929190ULL, 0x9f9e9d9c9b9a9998ULL);
    Interpret(0x4e033020U);  // tbx v0.16b, {v1.16b, v2.16b}, v3.16b
    EXPECT_VREG(0, 0x211f241593922f10ULL, 0x9f17161514131211ULL);
}

// zip1 v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Perm_zip1_4s) {
    SetVReg(1, 0x2222222211111111ULL, 0x4444444433333333ULL);
    SetVReg(2, 0xbbbbbbbbaaaaaaaaULL, 0xddddddddccccccccULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e823820U);  // zip1 v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0xaaaaaaaa11111111ULL, 0xbbbbbbbb22222222ULL);
}

// zip2 v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Perm_zip2_4s) {
    SetVReg(1, 0x2222222211111111ULL, 0x4444444433333333ULL);
    SetVReg(2, 0xbbbbbbbbaaaaaaaaULL, 0xddddddddccccccccULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e827820U);  // zip2 v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0xcccccccc33333333ULL, 0xdddddddd44444444ULL);
}

// uzp1 v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Perm_uzp1_4s) {
    SetVReg(1, 0x2222222211111111ULL, 0x4444444433333333ULL);
    SetVReg(2, 0xbbbbbbbbaaaaaaaaULL, 0xddddddddccccccccULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e821820U);  // uzp1 v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0x3333333311111111ULL, 0xccccccccaaaaaaaaULL);
}

// uzp2 v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Perm_uzp2_4s) {
    SetVReg(1, 0x2222222211111111ULL, 0x4444444433333333ULL);
    SetVReg(2, 0xbbbbbbbbaaaaaaaaULL, 0xddddddddccccccccULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e825820U);  // uzp2 v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0x4444444422222222ULL, 0xddddddddbbbbbbbbULL);
}

// trn1 v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Perm_trn1_4s) {
    SetVReg(1, 0x2222222211111111ULL, 0x4444444433333333ULL);
    SetVReg(2, 0xbbbbbbbbaaaaaaaaULL, 0xddddddddccccccccULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e822820U);  // trn1 v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0xaaaaaaaa11111111ULL, 0xcccccccc33333333ULL);
}

// trn2 v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Perm_trn2_4s) {
    SetVReg(1, 0x2222222211111111ULL, 0x4444444433333333ULL);
    SetVReg(2, 0xbbbbbbbbaaaaaaaaULL, 0xddddddddccccccccULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e826820U);  // trn2 v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0xbbbbbbbb22222222ULL, 0xdddddddd44444444ULL);
}

// zip1 v0.8b, v1.8b, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Perm_zip1_8b) {
    SetVReg(1, 0x0706050403020100ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x4746454443424140ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x0e023820U);  // zip1 v0.8b, v1.8b, v2.8b
    EXPECT_VREG(0, 0x4303420241014000ULL, 0x0000000000000000ULL);
}

// uzp1 v0.8b, v1.8b, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Perm_uzp1_8b) {
    SetVReg(1, 0x0706050403020100ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x4746454443424140ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x0e021820U);  // uzp1 v0.8b, v1.8b, v2.8b
    EXPECT_VREG(0, 0x4644424006040200ULL, 0x0000000000000000ULL);
}

// trn2 v0.8b, v1.8b, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Perm_trn2_8b) {
    SetVReg(1, 0x0706050403020100ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x4746454443424140ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x0e026820U);  // trn2 v0.8b, v1.8b, v2.8b
    EXPECT_VREG(0, 0x4707450543034101ULL, 0x0000000000000000ULL);
}

// addv b0, v1.8b
TEST_F(Arm64InterpreterOnlyTest, Addv_8b) {
    SetVReg(1, 0x0807060504030201ULL, 0x100f0e0d0c0b0a09ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x0e31b820U);  // addv b0, v1.8b
    EXPECT_VREG(0, 0x0000000000000024ULL, 0x0000000000000000ULL);
}

// addv b0, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Addv_16b) {
    SetVReg(1, 0x0807060504030201ULL, 0x100f0e0d0c0b0a09ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x4e31b820U);  // addv b0, v1.16b
    EXPECT_VREG(0, 0x0000000000000088ULL, 0x0000000000000000ULL);
}

// addv h0, v1.4h
TEST_F(Arm64InterpreterOnlyTest, Addv_4h) {
    SetVReg(1, 0x0400030002000100ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x0e71b820U);  // addv h0, v1.4h
    EXPECT_VREG(0, 0x0000000000000a00ULL, 0x0000000000000000ULL);
}

// addv s0, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Addv_4s) {
    SetVReg(1, 0x0000002000000010ULL, 0x0000004000000030ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x4eb1b820U);  // addv s0, v1.4s
    EXPECT_VREG(0, 0x00000000000000a0ULL, 0x0000000000000000ULL);
}

// smaxv b0, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Smaxv_16b) {
    SetVReg(1, 0x40302010fe01807fULL, 0x337e81ff00706050ULL);
    SetVReg(0, 0xaaaaaaaaaaaaaaaaULL, 0xaaaaaaaaaaaaaaaaULL);
    Interpret(0x4e30a820U);  // smaxv b0, v1.16b
    EXPECT_VREG(0, 0x000000000000007fULL, 0x0000000000000000ULL);
}

// sminv b0, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Sminv_16b) {
    SetVReg(1, 0x40302010fe01807fULL, 0x337e81ff00706050ULL);
    SetVReg(0, 0xaaaaaaaaaaaaaaaaULL, 0xaaaaaaaaaaaaaaaaULL);
    Interpret(0x4e31a820U);  // sminv b0, v1.16b
    EXPECT_VREG(0, 0x0000000000000080ULL, 0x0000000000000000ULL);
}

// umaxv b0, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Umaxv_16b) {
    SetVReg(1, 0x40302010fe01807fULL, 0x337e81ff00706050ULL);
    SetVReg(0, 0xaaaaaaaaaaaaaaaaULL, 0xaaaaaaaaaaaaaaaaULL);
    Interpret(0x6e30a820U);  // umaxv b0, v1.16b
    EXPECT_VREG(0, 0x00000000000000ffULL, 0x0000000000000000ULL);
}

// uminv b0, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Uminv_16b) {
    SetVReg(1, 0x40302010fe01807fULL, 0x337e81ff00706050ULL);
    SetVReg(0, 0xaaaaaaaaaaaaaaaaULL, 0xaaaaaaaaaaaaaaaaULL);
    Interpret(0x6e31a820U);  // uminv b0, v1.16b
    EXPECT_VREG(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
}

// smaxv s0, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Smaxv_4s) {
    SetVReg(1, 0xffffffff00000001ULL, 0x800000007fffffffULL);
    SetVReg(0, 0xaaaaaaaaaaaaaaaaULL, 0xaaaaaaaaaaaaaaaaULL);
    Interpret(0x4eb0a820U);  // smaxv s0, v1.4s
    EXPECT_VREG(0, 0x000000007fffffffULL, 0x0000000000000000ULL);
}

// sminv s0, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Sminv_4s) {
    SetVReg(1, 0xffffffff00000001ULL, 0x800000007fffffffULL);
    SetVReg(0, 0xaaaaaaaaaaaaaaaaULL, 0xaaaaaaaaaaaaaaaaULL);
    Interpret(0x4eb1a820U);  // sminv s0, v1.4s
    EXPECT_VREG(0, 0x0000000080000000ULL, 0x0000000000000000ULL);
}

// fmaxv s0, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Fmaxv_4s) {
    SetVReg(1, 0xc000000040600000ULL, 0x3f80000040e80000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6e30f820U);  // fmaxv s0, v1.4s
    EXPECT_VREG(0, 0x0000000040e80000ULL, 0x0000000000000000ULL);
}

// fminv s0, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Fminv_4s) {
    SetVReg(1, 0xc000000040600000ULL, 0x3f80000040e80000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6eb0f820U);  // fminv s0, v1.4s
    EXPECT_VREG(0, 0x00000000c0000000ULL, 0x0000000000000000ULL);
}

// fmaxnmv s0, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Fmaxnmv_4s_nan) {
    SetVReg(1, 0x7fc000003fc00000ULL, 0xc040000041100000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6e30c820U);  // fmaxnmv s0, v1.4s
    EXPECT_VREG(0, 0x0000000041100000ULL, 0x0000000000000000ULL);
}

// fminnmv s0, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Fminnmv_4s_nan) {
    SetVReg(1, 0x7fc000003fc00000ULL, 0xc040000041100000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6eb0c820U);  // fminnmv s0, v1.4s
    EXPECT_VREG(0, 0x00000000c0400000ULL, 0x0000000000000000ULL);
}

// addp v0.16b, v1.16b, v2.16b
TEST_F(Arm64InterpreterOnlyTest, Addp_16b) {
    SetVReg(1, 0x0807060504030201ULL, 0x100f0e0d0c0b0a09ULL);
    SetVReg(2, 0x2827262524232221ULL, 0x302f2e2d2c2b2a29ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e22bc20U);  // addp v0.16b, v1.16b, v2.16b
    EXPECT_VREG(0, 0x1f1b17130f0b0703ULL, 0x5f5b57534f4b4743ULL);
}

// addp v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Addp_4s) {
    SetVReg(1, 0x000000140000000aULL, 0x000000280000001eULL);
    SetVReg(2, 0x0000000200000001ULL, 0x0000000400000003ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4ea2bc20U);  // addp v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0x000000460000001eULL, 0x0000000700000003ULL);
}

// addp d0, v1.2d
TEST_F(Arm64InterpreterOnlyTest, Addp_scalar_d) {
    SetVReg(1, 0x0000000100000002ULL, 0x0000000300000004ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5ef1b820U);  // addp d0, v1.2d
    EXPECT_VREG(0, 0x0000000400000006ULL, 0x0000000000000000ULL);
}

// smaxp v0.8b, v1.8b, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Smaxp_8b) {
    SetVReg(1, 0x66554433fe01807fULL, 0x0000000000000000ULL);
    SetVReg(2, 0xc040d030e020f010ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x0e22a420U);  // smaxp v0.8b, v1.8b, v2.8b
    EXPECT_VREG(0, 0x403020106644017fULL, 0x0000000000000000ULL);
}

// sminp v0.8b, v1.8b, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Sminp_8b) {
    SetVReg(1, 0x66554433fe01807fULL, 0x0000000000000000ULL);
    SetVReg(2, 0xc040d030e020f010ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x0e22ac20U);  // sminp v0.8b, v1.8b, v2.8b
    EXPECT_VREG(0, 0xc0d0e0f05533fe80ULL, 0x0000000000000000ULL);
}

// umaxp v0.8b, v1.8b, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Umaxp_8b) {
    SetVReg(1, 0x66554433fe01807fULL, 0x0000000000000000ULL);
    SetVReg(2, 0xc040d030e020f010ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x2e22a420U);  // umaxp v0.8b, v1.8b, v2.8b
    EXPECT_VREG(0, 0xc0d0e0f06644fe80ULL, 0x0000000000000000ULL);
}

// uminp v0.8b, v1.8b, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Uminp_8b) {
    SetVReg(1, 0x66554433fe01807fULL, 0x0000000000000000ULL);
    SetVReg(2, 0xc040d030e020f010ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x2e22ac20U);  // uminp v0.8b, v1.8b, v2.8b
    EXPECT_VREG(0, 0x403020105533017fULL, 0x0000000000000000ULL);
}

// faddp v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Faddp_4s) {
    SetVReg(1, 0x400000003f800000ULL, 0x4080000040400000ULL);
    SetVReg(2, 0x41a0000041200000ULL, 0x4220000041f00000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6e22d420U);  // faddp v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0x40e0000040400000ULL, 0x428c000041f00000ULL);
}

// faddp s0, v1.2s
TEST_F(Arm64InterpreterOnlyTest, Faddp_scalar_s) {
    SetVReg(1, 0x40c0000040a00000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x7e30d820U);  // faddp s0, v1.2s
    EXPECT_VREG(0, 0x0000000041300000ULL, 0x0000000000000000ULL);
}

// faddp d0, v1.2d
TEST_F(Arm64InterpreterOnlyTest, Faddp_scalar_d) {
    SetVReg(1, 0x3ff8000000000000ULL, 0x4004000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x7e70d820U);  // faddp d0, v1.2d
    EXPECT_VREG(0, 0x4010000000000000ULL, 0x0000000000000000ULL);
}

// sqdmulh v0.4h, v1.4h, v2.4h
TEST_F(Arm64InterpreterOnlyTest, Sqdmulh_4h) {
    SetVReg(1, 0x0800100020004000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x2000200040004000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x0e62b420U);  // sqdmulh v0.4h, v1.4h, v2.4h
    EXPECT_VREG(0, 0x0200040010002000ULL, 0x0000000000000000ULL);
}

// sqrdmulh v0.4h, v1.4h, v2.4h
TEST_F(Arm64InterpreterOnlyTest, Sqrdmulh_4h) {
    SetVReg(1, 0x0800100020004000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x2000200040004000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x2e62b420U);  // sqrdmulh v0.4h, v1.4h, v2.4h
    EXPECT_VREG(0, 0x0200040010002000ULL, 0x0000000000000000ULL);
}

// sqdmulh v0.4h, v1.4h, v2.4h
TEST_F(Arm64InterpreterOnlyTest, Sqdmulh_4h_sat) {
    SetVReg(1, 0x8000800080008000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x8000800080008000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x0e62b420U);  // sqdmulh v0.4h, v1.4h, v2.4h
    EXPECT_VREG(0, 0x7fff7fff7fff7fffULL, 0x0000000000000000ULL);
}

// sqdmulh v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Sqdmulh_4s) {
    SetVReg(1, 0x7fffffff40000000ULL, 0x8000000000010000ULL);
    SetVReg(2, 0x7fffffff40000000ULL, 0x8000000000020000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4ea2b420U);  // sqdmulh v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0x7ffffffe20000000ULL, 0x7fffffff00000004ULL);
}

// sqdmulh h0, h1, h2
TEST_F(Arm64InterpreterOnlyTest, Sqdmulh_scalar_h) {
    SetVReg(1, 0x0000000000004000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x0000000000004000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5e62b420U);  // sqdmulh h0, h1, h2
    EXPECT_VREG(0, 0x0000000000002000ULL, 0x0000000000000000ULL);
}

// sqrdmulh s0, s1, s2
TEST_F(Arm64InterpreterOnlyTest, Sqrdmulh_scalar_s_sat) {
    SetVReg(1, 0x0000000080000000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x0000000080000000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x7ea2b420U);  // sqrdmulh s0, s1, s2
    // (2 * INT32_MIN * INT32_MIN + (1 << 31)) >> 32 == 1 << 31 -> saturates to INT32_MAX.
    EXPECT_VREG(0, 0x000000007fffffffULL, 0x0000000000000000ULL);
}

// Scalar by-element signed saturating doubling multiply. These encodings used
// to raise SIGILL (decoder Undefined); they now interpret.

// sqdmulh s0, s1, v2.s[0]: (2 * 2^30 * 2^30) >> 32 == 2^29.
TEST_F(Arm64InterpreterOnlyTest, SqdmulhScalarIdx_s) {
    SetVReg(1, 0x0000000040000000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x0000000040000000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5f82c020U);  // sqdmulh s0, s1, v2.s[0]
    EXPECT_VREG(0, 0x0000000020000000ULL, 0x0000000000000000ULL);
}

// sqdmulh s0, s1, v2.s[1]: selects lane 1 of Vm (lane 0 is zero, so a wrong
// lane would give 0). Proves the index math.
TEST_F(Arm64InterpreterOnlyTest, SqdmulhScalarIdx_s_lane1) {
    SetVReg(1, 0x0000000040000000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x4000000000000000ULL, 0x0000000000000000ULL);  // v2.s[1]=0x40000000
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5fa2c020U);  // sqdmulh s0, s1, v2.s[1]
    EXPECT_VREG(0, 0x0000000020000000ULL, 0x0000000000000000ULL);
}

// sqrdmulh s0, s1, v2.s[0] with INT32_MIN operands: rounded high half saturates.
TEST_F(Arm64InterpreterOnlyTest, SqrdmulhScalarIdx_s_sat) {
    SetVReg(1, 0x0000000080000000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x0000000080000000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5f82d020U);  // sqrdmulh s0, s1, v2.s[0]
    EXPECT_VREG(0, 0x000000007fffffffULL, 0x0000000000000000ULL);
}

// sqdmull d0, s1, v2.s[0]: widening 2*2^30*2^30 == 2^61 into the 64-bit lane.
TEST_F(Arm64InterpreterOnlyTest, SqdmullScalarIdx_s_to_d) {
    SetVReg(1, 0x0000000040000000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x0000000040000000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5f82b020U);  // sqdmull d0, s1, v2.s[0]
    EXPECT_VREG(0, 0x2000000000000000ULL, 0x0000000000000000ULL);
}

// sqdmull d0, s1, v2.s[0] with INT32_MIN operands: only (INT_MIN,INT_MIN)
// saturates, to INT64_MAX.
TEST_F(Arm64InterpreterOnlyTest, SqdmullScalarIdx_s_to_d_sat) {
    SetVReg(1, 0x0000000080000000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x0000000080000000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5f82b020U);  // sqdmull d0, s1, v2.s[0]
    EXPECT_VREG(0, 0x7fffffffffffffffULL, 0x0000000000000000ULL);
}

// sqdmulh h0, h1, v2.h[0]: H form, (2 * 2^14 * 2^14) >> 16 == 2^13.
TEST_F(Arm64InterpreterOnlyTest, SqdmulhScalarIdx_h) {
    SetVReg(1, 0x0000000000004000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x0000000000004000ULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5f42c020U);  // sqdmulh h0, h1, v2.h[0]
    EXPECT_VREG(0, 0x0000000000002000ULL, 0x0000000000000000ULL);
}

// sqdmull s0, h1, v2.h[3]: H->S widening with a high lane index.
TEST_F(Arm64InterpreterOnlyTest, SqdmullScalarIdx_h_to_s_lane3) {
    SetVReg(1, 0x0000000000004000ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x4000000000000000ULL, 0x0000000000000000ULL);  // v2.h[3]=0x4000
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5f72b020U);  // sqdmull s0, h1, v2.h[3]
    EXPECT_VREG(0, 0x0000000020000000ULL, 0x0000000000000000ULL);
}

// aese v0.16b, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Aese) {
    SetVReg(0, 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
    SetVReg(1, 0x0f0f0f0f0f0f0f0fULL, 0x0f0f0f0f0f0f0f0fULL);
    Interpret(0x4e284820U);  // aese v0.16b, v1.16b
    EXPECT_VREG(0, 0xfe7c6f2b636b6776ULL, 0xf201ab7b30d777c5ULL);
}

// aesd v0.16b, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Aesd) {
    SetVReg(0, 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
    SetVReg(1, 0x0f0f0f0f0f0f0f0fULL, 0x0f0f0f0f0f0f0f0fULL);
    Interpret(0x4e285820U);  // aesd v0.16b, v1.16b
    EXPECT_VREG(0, 0x3009d79ebf366afbULL, 0x8140a5d552f3a338ULL);
}

// aesmc v0.16b, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Aesmc) {
    SetVReg(1, 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x4e286820U);  // aesmc v0.16b, v1.16b
    EXPECT_VREG(0, 0x0104030605000702ULL, 0x090c0b0e0d080f0aULL);
}

// aesimc v0.16b, v1.16b
TEST_F(Arm64InterpreterOnlyTest, Aesimc) {
    SetVReg(1, 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x4e287820U);  // aesimc v0.16b, v1.16b
    EXPECT_VREG(0, 0x090c0b0e0d080f0aULL, 0x0104030605000702ULL);
}

// sha1c q0, s1, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Sha1C) {
    SetVReg(0, 0xefcdab8967452301ULL, 0x1032547698badcfeULL);
    SetVReg(1, 0x00000000c3d2e1f0ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x2222222211111111ULL, 0x4444444433333333ULL);
    Interpret(0x5e020020U);  // sha1c q0, s1, v2.4s
    EXPECT_VREG(0, 0x1caebb0f33e059c9ULL, 0xd590cc0a3dad9ec0ULL);
}

// sha1p q0, s1, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Sha1P) {
    SetVReg(0, 0xefcdab8967452301ULL, 0x1032547698badcfeULL);
    SetVReg(1, 0x00000000c3d2e1f0ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x2222222211111111ULL, 0x4444444433333333ULL);
    Interpret(0x5e021020U);  // sha1p q0, s1, v2.4s
    EXPECT_VREG(0, 0xd4d79367dc17e052ULL, 0x89335d8b5403f45eULL);
}

// sha1m q0, s1, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Sha1M) {
    SetVReg(0, 0xefcdab8967452301ULL, 0x1032547698badcfeULL);
    SetVReg(1, 0x00000000c3d2e1f0ULL, 0x0000000000000000ULL);
    SetVReg(2, 0x2222222211111111ULL, 0x4444444433333333ULL);
    Interpret(0x5e022020U);  // sha1m q0, s1, v2.4s
    EXPECT_VREG(0, 0xfd7e55a1c5943025ULL, 0xd590cc0a1dab79b9ULL);
}

// sha1h s0, s1
TEST_F(Arm64InterpreterOnlyTest, Sha1h) {
    SetVReg(1, 0x000000000000000fULL, 0x0000000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x5e280820U);  // sha1h s0, s1
    EXPECT_VREG(0, 0x00000000c0000003ULL, 0x0000000000000000ULL);
}

// sha1su0 v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Sha1su0) {
    SetVReg(0, 0x0000000200000001ULL, 0x0000000400000003ULL);
    SetVReg(1, 0x0000000600000005ULL, 0x0000000800000007ULL);
    SetVReg(2, 0x0000000a00000009ULL, 0x0000000c0000000bULL);
    Interpret(0x5e023020U);  // sha1su0 v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0x0000000c0000000bULL, 0x0000000e0000000dULL);
}

// sha1su1 v0.4s, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Sha1su1) {
    SetVReg(0, 0x0000002000000010ULL, 0x0000004000000030ULL);
    SetVReg(1, 0x0000000200000001ULL, 0x0000000400000003ULL);
    Interpret(0x5e281820U);  // sha1su1 v0.4s, v1.4s
    EXPECT_VREG(0, 0x0000004600000024ULL, 0x000000c800000068ULL);
}

// sha256h q0, q1, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Sha256h) {
    SetVReg(0, 0xbb67ae856a09e667ULL, 0xa54ff53a3c6ef372ULL);
    SetVReg(1, 0x9b05688c510e527fULL, 0x5be0cd191f83d9abULL);
    SetVReg(2, 0x71374491428a2f98ULL, 0xe9b5dba5b5c0fbcfULL);
    Interpret(0x5e024020U);  // sha256h q0, q1, v2.4s
    EXPECT_VREG(0, 0xf3dd6c3f0a24b1aaULL, 0xfc08884d7ad96290ULL);
}

// sha256h2 q0, q1, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Sha256h2) {
    SetVReg(0, 0x59f111f13956c25bULL, 0xab1c5ed5923f82a4ULL);
    SetVReg(1, 0x71374491428a2f98ULL, 0xe9b5dba5b5c0fbcfULL);
    SetVReg(2, 0x71374491428a2f98ULL, 0xe9b5dba5b5c0fbcfULL);
    Interpret(0x5e025020U);  // sha256h2 q0, q1, v2.4s
    EXPECT_VREG(0, 0xf7631015aa1b0fb6ULL, 0xffb8c754578c546fULL);
}

// sha256su0 v0.4s, v1.4s
TEST_F(Arm64InterpreterOnlyTest, Sha256su0) {
    SetVReg(0, 0x89abcdef01234567ULL, 0xfeedfacedeadbeefULL);
    SetVReg(1, 0x000000000f0f0f0fULL, 0x0000000000000000ULL);
    Interpret(0x5e282820U);  // sha256su0 v0.4s, v1.4s
    EXPECT_VREG(0, 0x357ee8fa3e8111b3ULL, 0xdb2a370adb419a06ULL);
}

// sha256su1 v0.4s, v1.4s, v2.4s
TEST_F(Arm64InterpreterOnlyTest, Sha256su1) {
    SetVReg(0, 0x2222222211111111ULL, 0x4444444433333333ULL);
    SetVReg(1, 0xaaaaaaaa00000000ULL, 0xccccccccbbbbbbbbULL);
    SetVReg(2, 0x6666666655555555ULL, 0x8888888877777777ULL);
    Interpret(0x5e026020U);  // sha256su1 v0.4s, v1.4s, v2.4s
    EXPECT_VREG(0, 0x3355555411044443ULL, 0x99a5e42eaaadabb2ULL);
}

// fcadd v0.4s, v1.4s, v2.4s, #90
TEST_F(Arm64InterpreterOnlyTest, Fcadd_4s_90) {
    SetVReg(1, 0x400000003f800000ULL, 0x4080000040400000ULL);
    SetVReg(2, 0x41a0000041200000ULL, 0x4220000041f00000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6e82e420U);  // fcadd v0.4s, v1.4s, v2.4s, #90
    EXPECT_VREG(0, 0x41400000c1980000ULL, 0x42080000c2140000ULL);
}

// fcadd v0.4s, v1.4s, v2.4s, #270
TEST_F(Arm64InterpreterOnlyTest, Fcadd_4s_270) {
    SetVReg(1, 0x400000003f800000ULL, 0x4080000040400000ULL);
    SetVReg(2, 0x41a0000041200000ULL, 0x4220000041f00000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6e82f420U);  // fcadd v0.4s, v1.4s, v2.4s, #270
    EXPECT_VREG(0, 0xc100000041a80000ULL, 0xc1d00000422c0000ULL);
}

// fcadd v0.2d, v1.2d, v2.2d, #90
TEST_F(Arm64InterpreterOnlyTest, Fcadd_2d_90) {
    SetVReg(1, 0x3ff0000000000000ULL, 0x4000000000000000ULL);
    SetVReg(2, 0x4024000000000000ULL, 0x4034000000000000ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6ec2e420U);  // fcadd v0.2d, v1.2d, v2.2d, #90
    EXPECT_VREG(0, 0xc033000000000000ULL, 0x4028000000000000ULL);
}

// fcmla v0.4s, v1.4s, v2.4s, #0
TEST_F(Arm64InterpreterOnlyTest, Fcmla_4s_0) {
    SetVReg(0, 0x4348000042c80000ULL, 0x43c8000043960000ULL);
    SetVReg(1, 0x400000003f800000ULL, 0x4080000040400000ULL);
    SetVReg(2, 0x41a0000041200000ULL, 0x4220000041f00000ULL);
    Interpret(0x6e82c420U);  // fcmla v0.4s, v1.4s, v2.4s, #0
    EXPECT_VREG(0, 0x435c000042dc0000ULL, 0x4402000043c30000ULL);
}

// fcmla v0.4s, v1.4s, v2.4s, #90
TEST_F(Arm64InterpreterOnlyTest, Fcmla_4s_90) {
    SetVReg(0, 0x4348000042c80000ULL, 0x43c8000043960000ULL);
    SetVReg(1, 0x400000003f800000ULL, 0x4080000040400000ULL);
    SetVReg(2, 0x41a0000041200000ULL, 0x4220000041f00000ULL);
    Interpret(0x6e82cc20U);  // fcmla v0.4s, v1.4s, v2.4s, #90
    EXPECT_VREG(0, 0x435c000042700000ULL, 0x44020000430c0000ULL);
}

// fcmla v0.4s, v1.4s, v2.4s, #180
TEST_F(Arm64InterpreterOnlyTest, Fcmla_4s_180) {
    SetVReg(0, 0x4348000042c80000ULL, 0x43c8000043960000ULL);
    SetVReg(1, 0x400000003f800000ULL, 0x4080000040400000ULL);
    SetVReg(2, 0x41a0000041200000ULL, 0x4220000041f00000ULL);
    Interpret(0x6e82d420U);  // fcmla v0.4s, v1.4s, v2.4s, #180
    EXPECT_VREG(0, 0x4334000042b40000ULL, 0x438c000043520000ULL);
}

// fcmla v0.4s, v1.4s, v2.4s, #270
TEST_F(Arm64InterpreterOnlyTest, Fcmla_4s_270) {
    SetVReg(0, 0x4348000042c80000ULL, 0x43c8000043960000ULL);
    SetVReg(1, 0x400000003f800000ULL, 0x4080000040400000ULL);
    SetVReg(2, 0x41a0000041200000ULL, 0x4220000041f00000ULL);
    Interpret(0x6e82dc20U);  // fcmla v0.4s, v1.4s, v2.4s, #270
    EXPECT_VREG(0, 0x43340000430c0000ULL, 0x438c000043e60000ULL);
}

// sdot v0.4s, v1.16b, v2.16b
TEST_F(Arm64InterpreterOnlyTest, Sdot_4s) {
    SetVReg(1, 0x0807060504030201ULL, 0x281e140afcfdfeffULL);
    SetVReg(2, 0x0101010102020202ULL, 0x0303030301010101ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4e829420U);  // sdot v0.4s, v1.16b, v2.16b
    EXPECT_VREG(0, 0x0000001a00000014ULL, 0x0000012cfffffff6ULL);
}

// udot v0.4s, v1.16b, v2.16b
TEST_F(Arm64InterpreterOnlyTest, Udot_4s) {
    SetVReg(1, 0x0807060504030201ULL, 0x281e140afcfdfeffULL);
    SetVReg(2, 0x0101010102020202ULL, 0x0303030301010101ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x6e829420U);  // udot v0.4s, v1.16b, v2.16b
    EXPECT_VREG(0, 0x0000001a00000014ULL, 0x0000012c000003f6ULL);
}

// sdot v0.4s, v1.16b, v2.4b[1]
TEST_F(Arm64InterpreterOnlyTest, Sdot_idx_4s) {
    SetVReg(1, 0x0807060504030201ULL, 0x281e140afcfdfeffULL);
    SetVReg(2, 0x0101010102020202ULL, 0x0303030301010101ULL);
    SetVReg(0, 0x0000000000000000ULL, 0x0000000000000000ULL);
    Interpret(0x4fa2e020U);  // sdot v0.4s, v1.16b, v2.4b[1]
    EXPECT_VREG(0, 0x0000001a0000000aULL, 0x00000064fffffff6ULL);
}

// udot v0.2s, v1.8b, v2.8b
TEST_F(Arm64InterpreterOnlyTest, Udot_2s) {
    SetVReg(1, 0x0807060504030201ULL, 0x281e140afcfdfeffULL);
    SetVReg(2, 0x0101010102020202ULL, 0x0303030301010101ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x2e829420U);  // udot v0.2s, v1.8b, v2.8b
    // UDOT accumulates: each lane is 0xFFFFFFFF (preset) + dot, mod 2^32
    // (lane0: +0x14 -> 0x13, lane1: +0x1a -> 0x19); upper half zeroed (2S form).
    EXPECT_VREG(0, 0x0000001900000013ULL, 0x0000000000000000ULL);
}

// fcvtxn v0.2s, v1.2d
TEST_F(Arm64InterpreterOnlyTest, Fcvtxn_vector) {
    SetVReg(1, 0x3ff8000000000000ULL, 0xc002000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x2e616820U);  // fcvtxn v0.2s, v1.2d
    EXPECT_VREG(0, 0xc01000003fc00000ULL, 0x0000000000000000ULL);
}

// fcvtxn s0, d1
TEST_F(Arm64InterpreterOnlyTest, Fcvtxn_scalar) {
    SetVReg(1, 0x400e000000000000ULL, 0x4022000000000000ULL);
    SetVReg(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL);
    Interpret(0x7e616820U);  // fcvtxn s0, d1
    EXPECT_VREG(0, 0x0000000040700000ULL, 0x0000000000000000ULL);
}

// --- Top-byte-ignore -------------------------------------------------------
//
// ARM64 ignores address bits [63:56] on access. Android's Scudo allocator
// relies on it: it tags heap pointers and dereferences the tagged value
// directly. x86_64 has no equivalent (such an address is not even canonical),
// so the interpreter must strip the tag exactly as LiteTranslator::ApplyTbi
// does — without it, the first Scudo allocation in any statically-linked
// binary faulted, and interpret-only could not run one at all.

TEST_F(Arm64InterpreterOnlyTest, LoadThroughTaggedPointerIgnoresTopByte) {
  uint64_t cell = 0x1122334455667788ULL;
  // Scudo's tag shape: bit 57 set in the top byte.
  uint64_t tagged = ToGuestAddr(&cell) | 0x0200'0000'0000'0000ULL;

  state_.cpu.x[1] = tagged;
  Interpret(0xf9400020);  // ldr x0, [x1]

  EXPECT_EQ(state_.cpu.x[0], 0x1122334455667788ULL);
}

TEST_F(Arm64InterpreterOnlyTest, StoreThroughTaggedPointerIgnoresTopByte) {
  uint64_t cell = 0;
  uint64_t tagged = ToGuestAddr(&cell) | 0xff00'0000'0000'0000ULL;  // all tag bits

  state_.cpu.x[0] = 0xdeadbeefcafef00dULL;
  state_.cpu.x[1] = tagged;
  Interpret(0xf9000020);  // str x0, [x1]

  EXPECT_EQ(cell, 0xdeadbeefcafef00dULL);
}

TEST_F(Arm64InterpreterOnlyTest, TaggedPointerValueSurvivesInRegister) {
  // The tag is stripped at address *use*, never written back to the register:
  // guest code computes with tagged pointers and expects to read the tag back.
  uint64_t cell = 0x4242424242424242ULL;
  uint64_t tagged = ToGuestAddr(&cell) | 0x0200'0000'0000'0000ULL;

  state_.cpu.x[1] = tagged;
  Interpret(0xf9400020);  // ldr x0, [x1]

  EXPECT_EQ(state_.cpu.x[1], tagged);
}

}  // namespace
}  // namespace berberis
