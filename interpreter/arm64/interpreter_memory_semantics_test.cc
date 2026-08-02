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

// Golden-value coverage for interpreter memory semantics.
//
// The memory differential fuzzers prove the interpreter and the JIT tiers
// AGREE. They cannot prove either is right: a misconception shared by both --
// they are generated from the same decoder and were often written from the same
// reading of the manual -- produces identical wrong answers and a passing
// differential. The interpreter is the reference tier, so the properties it
// must get right on its own are pinned here against values derived from the ARM
// ARM rather than from another tier.
//
// What is checked, and why each one has a history:
//   * sign vs zero extension per load width. LDPSW zero-extending instead of
//     sign-extending blew a libQt6Gui fill loop, and the decoder carries the
//     sign as a separate field that a mis-dispatch silently flips.
//   * little-endian byte order, which fixes what "the wrong width" even means
//     for the sign-extension cases above.
//   * unaligned access. ARM64 permits it for these forms and real code relies
//     on it, but the fuzzers pin their base registers to aligned addresses, so
//     nothing else exercises it.
//   * top-byte-ignore applied at the point of access and NEVER on write-back.
//     Masking the register value instead of the address would look correct on
//     every load and store while corrupting pointer arithmetic, because guest
//     code computes with tagged pointers and expects to read the tag back.

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/arm64/interpreter.h"

namespace berberis {

namespace {

// LDR/STR (immediate, unsigned offset). LDR X0,[X1] == 0xF9400020.
uint32_t EncLoadStoreImm(int size, int opc, uint32_t imm12, int rn, int rt) {
  return (static_cast<uint32_t>(size) << 30) | (0b111001u << 24) |
         (static_cast<uint32_t>(opc) << 22) | ((imm12 & 0xFFF) << 10) |
         (static_cast<uint32_t>(rn) << 5) | static_cast<uint32_t>(rt);
}

// LDUR/STUR and the pre/post-index forms. idx==0b00 is the unscaled (LDUR)
// variant, 0b01 post-index, 0b11 pre-index. LDUR X0,[X1,#8] == 0xF8408020.
uint32_t EncLoadStoreIdx(int size, int opc, int imm9, int idx, int rn, int rt) {
  return (static_cast<uint32_t>(size) << 30) | (0b111000u << 24) |
         (static_cast<uint32_t>(opc) << 22) | ((static_cast<uint32_t>(imm9) & 0x1FF) << 12) |
         (static_cast<uint32_t>(idx) << 10) | (static_cast<uint32_t>(rn) << 5) |
         static_cast<uint32_t>(rt);
}

// LDP/STP (signed offset). LDPSW X0,X1,[X2] == 0x69400440.
uint32_t EncLoadStorePair(int opc, int l, int imm7, int rt2, int rn, int rt) {
  return (static_cast<uint32_t>(opc) << 30) | (0b101u << 27) | (0b010u << 23) |
         (static_cast<uint32_t>(l) << 22) | ((static_cast<uint32_t>(imm7) & 0x7F) << 15) |
         (static_cast<uint32_t>(rt2) << 10) | (static_cast<uint32_t>(rn) << 5) |
         static_cast<uint32_t>(rt);
}

class Arm64InterpreterMemorySemantics : public ::testing::Test {
 protected:
  // Room for an unaligned access near the middle, well away from either end.
  alignas(16) uint8_t buf_[256] = {};
  ThreadState state_{};

  static constexpr int kBase = 1;  // x1 holds the base address
  static constexpr int kDst = 0;   // x0 is the destination / source

  void SetUp() override { memset(buf_, 0, sizeof(buf_)); }

  // Executes one instruction with x[kBase] = &buf_[offset] (optionally tagged).
  void Run(uint32_t insn, size_t offset = 0, uint64_t tag = 0) {
    static uint32_t code;
    code = insn;
    state_.cpu.x[kBase] = static_cast<uint64_t>(ToGuestAddr(buf_ + offset)) | (tag << 56);
    state_.cpu.insn_addr = ToGuestAddr(&code);
    InterpretInsn(&state_);
  }

  void RunWithValue(uint32_t insn, uint64_t value, size_t offset = 0, uint64_t tag = 0) {
    static uint32_t code;
    code = insn;
    state_.cpu.x[kBase] = static_cast<uint64_t>(ToGuestAddr(buf_ + offset)) | (tag << 56);
    state_.cpu.x[kDst] = value;
    state_.cpu.insn_addr = ToGuestAddr(&code);
    InterpretInsn(&state_);
  }
};

// Fixes what "width" means for every other test in this file.
TEST_F(Arm64InterpreterMemorySemantics, StoreIsLittleEndian) {
  // str x0, [x1]
  RunWithValue(EncLoadStoreImm(0b11, 0b00, 0, kBase, kDst), 0x0102030405060708ULL);
  const uint8_t want[8] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};
  EXPECT_EQ(memcmp(buf_, want, 8), 0);
}

TEST_F(Arm64InterpreterMemorySemantics, LdrbZeroExtends) {
  buf_[0] = 0xFF;
  // ldrb w0, [x1]
  Run(EncLoadStoreImm(0b00, 0b01, 0, kBase, kDst));
  EXPECT_EQ(state_.cpu.x[kDst], 0x00000000000000FFULL);
}

TEST_F(Arm64InterpreterMemorySemantics, LdrsbSignExtendsToSixtyFour) {
  buf_[0] = 0xFF;
  // ldrsb x0, [x1]
  Run(EncLoadStoreImm(0b00, 0b10, 0, kBase, kDst));
  EXPECT_EQ(state_.cpu.x[kDst], 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(Arm64InterpreterMemorySemantics, LdrshSignExtendsToSixtyFour) {
  const uint16_t v = 0x8000;
  memcpy(buf_, &v, sizeof(v));
  // ldrsh x0, [x1]
  Run(EncLoadStoreImm(0b01, 0b10, 0, kBase, kDst));
  EXPECT_EQ(state_.cpu.x[kDst], 0xFFFFFFFFFFFF8000ULL);
}

// The LDPSW class: a negative 32-bit value must reach 64 bits sign-extended.
TEST_F(Arm64InterpreterMemorySemantics, LdrswSignExtendsNegative) {
  const uint32_t v = 0x80000000;
  memcpy(buf_, &v, sizeof(v));
  // ldrsw x0, [x1]
  Run(EncLoadStoreImm(0b10, 0b10, 0, kBase, kDst));
  EXPECT_EQ(state_.cpu.x[kDst], 0xFFFFFFFF80000000ULL);
}

TEST_F(Arm64InterpreterMemorySemantics, LdrswLeavesPositiveAlone) {
  const uint32_t v = 0x7FFFFFFF;
  memcpy(buf_, &v, sizeof(v));
  Run(EncLoadStoreImm(0b10, 0b10, 0, kBase, kDst));
  EXPECT_EQ(state_.cpu.x[kDst], 0x000000007FFFFFFFULL);
}

// The exact shape of the Qt bug: BOTH destinations must be sign-extended, not
// just the first.
TEST_F(Arm64InterpreterMemorySemantics, LdpswSignExtendsBothDestinations) {
  const uint32_t v[2] = {0xFFFFFFF0, 0x80000001};
  memcpy(buf_, v, sizeof(v));
  // ldpsw x0, x2, [x1]
  Run(EncLoadStorePair(0b01, 1, 0, /*rt2=*/2, kBase, kDst));
  EXPECT_EQ(state_.cpu.x[kDst], 0xFFFFFFFFFFFFFFF0ULL);
  EXPECT_EQ(state_.cpu.x[2], 0xFFFFFFFF80000001ULL);
}

// ARM64 allows these unaligned; the fuzzers only ever use aligned bases.
TEST_F(Arm64InterpreterMemorySemantics, UnalignedLoadReadsStraddlingBytes) {
  const uint8_t pattern[16] = {0, 0, 0, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
                               0xFF, 0x11, 0x22, 0x33, 0, 0, 0, 0};
  memcpy(buf_, pattern, sizeof(pattern));
  // ldur x0, [x1, #3] -- starts at an odd byte, spans 8 bytes.
  // Bytes at offsets 3..10 are AA BB CC DD EE FF 11 22; little-endian makes the
  // last byte read the most significant one.
  Run(EncLoadStoreIdx(0b11, 0b01, 3, 0b00, kBase, kDst));
  EXPECT_EQ(state_.cpu.x[kDst], 0x2211FFEEDDCCBBAAULL)
      << "unaligned 8-byte load did not assemble the straddling bytes";
}

TEST_F(Arm64InterpreterMemorySemantics, UnalignedStoreWritesStraddlingBytes) {
  // stur x0, [x1, #3]
  RunWithValue(EncLoadStoreIdx(0b11, 0b00, 3, 0b00, kBase, kDst), 0x0102030405060708ULL);
  const uint8_t want[8] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};
  EXPECT_EQ(memcmp(buf_ + 3, want, 8), 0);
  EXPECT_EQ(buf_[2], 0) << "unaligned store wrote below its address";
  EXPECT_EQ(buf_[11], 0) << "unaligned store wrote past its address";
}

// Top-byte-ignore: the tag must be stripped when the address is USED.
TEST_F(Arm64InterpreterMemorySemantics, TaggedBaseLoadsTheSameValue) {
  const uint64_t v = 0xDEADBEEFCAFEF00DULL;
  memcpy(buf_, &v, sizeof(v));
  Run(EncLoadStoreImm(0b11, 0b01, 0, kBase, kDst), /*offset=*/0, /*tag=*/0xA5);
  EXPECT_EQ(state_.cpu.x[kDst], v) << "a tagged base address did not read the same memory";
}

TEST_F(Arm64InterpreterMemorySemantics, TaggedBaseStoresToTheSameAddress) {
  RunWithValue(EncLoadStoreImm(0b11, 0b00, 0, kBase, kDst),
               0x1122334455667788ULL,
               /*offset=*/0,
               /*tag=*/0x7E);
  uint64_t got = 0;
  memcpy(&got, buf_, sizeof(got));
  EXPECT_EQ(got, 0x1122334455667788ULL) << "a tagged base address did not write the same memory";
}

// ...and must NOT be stripped from the register. Guest code keeps computing
// with the tagged pointer, so masking the value on write-back would corrupt
// every subsequent access derived from it.
TEST_F(Arm64InterpreterMemorySemantics, PostIndexWritebackKeepsTheTag) {
  const uint64_t tag = 0x3C;
  const uint64_t base = static_cast<uint64_t>(ToGuestAddr(buf_)) | (tag << 56);
  // ldr x0, [x1], #8
  Run(EncLoadStoreIdx(0b11, 0b01, 8, 0b01, kBase, kDst), /*offset=*/0, tag);
  EXPECT_EQ(state_.cpu.x[kBase], base + 8)
      << "post-index writeback lost the address tag (masked the register, not the access)";
}

TEST_F(Arm64InterpreterMemorySemantics, TaggedBaseIsUnchangedByAPlainLoad) {
  const uint64_t tag = 0x99;
  const uint64_t base = static_cast<uint64_t>(ToGuestAddr(buf_)) | (tag << 56);
  Run(EncLoadStoreImm(0b11, 0b01, 0, kBase, kDst), /*offset=*/0, tag);
  EXPECT_EQ(state_.cpu.x[kBase], base) << "a load rewrote its own tagged base register";
}

}  // namespace

}  // namespace berberis
