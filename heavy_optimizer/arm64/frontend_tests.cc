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

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/heavy_optimizer/arm64/heavy_optimize_region.h"
#include "berberis/runtime_primitives/platform.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_exec_region.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

// ARM64 instruction encoding helpers (same forms as the lite-translator tests).
// MOVZ Xd, #imm16 (shift 0).
constexpr uint32_t MovzX(uint8_t rd, uint16_t imm16) {
  return 0xD2800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVZ Xd, #imm16, LSL #(hw*16).
constexpr uint32_t MovzHwX(uint8_t rd, uint16_t imm16, uint8_t hw) {
  return 0xD2800000 | (static_cast<uint32_t>(hw) << 21) |
         (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVN Xd, #imm16 (shift 0).
constexpr uint32_t MovnX(uint8_t rd, uint16_t imm16) {
  return 0x92800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVK Xd, #imm16, LSL #(hw*16).
constexpr uint32_t MovkHwX(uint8_t rd, uint16_t imm16, uint8_t hw) {
  return 0xF2800000 | (static_cast<uint32_t>(hw) << 21) |
         (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVZ Wd, #imm16 (32-bit, shift 0).
constexpr uint32_t MovzW(uint8_t rd, uint16_t imm16) {
  return 0x52800000 | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// --- Non-flag integer ALU encoders (same forms as the lite-translator tests). ---
// ADD/SUB Xd, Xn, #imm12.
constexpr uint32_t AddImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x91000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
constexpr uint32_t SubImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0xD1000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// SUBS Xd, Xn, #imm12 (flag-setting subtract).
constexpr uint32_t SubsImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0xF1000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// ADDS Xd, Xn, #imm12 (flag-setting add).
constexpr uint32_t AddsImmX(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0xB1000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// CMP Xn, #imm12 == SUBS XZR, Xn, #imm12.
constexpr uint32_t CmpImmX(uint8_t rn, uint16_t imm12) {
  return SubsImmX(31, rn, imm12);
}
// SUBS/ADDS Wd, Wn, #imm12 (32-bit, flag-setting).
constexpr uint32_t SubsImmW(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x71000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
constexpr uint32_t AddsImmW(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x31000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// ADD/SUB Wd, Wn, #imm12 (32-bit).
constexpr uint32_t AddImmW(uint8_t rd, uint8_t rn, uint16_t imm12) {
  return 0x11000000 | (static_cast<uint32_t>(imm12) << 10) | (rn << 5) | rd;
}
// ADD/SUB/AND/ORR/EOR Xd, Xn, Xm (shifted register, no shift).
constexpr uint32_t AddRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x8B000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t SubRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xCB000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t AndRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x8A000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t OrrRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xAA000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t EorRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xCA000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// BIC Xd, Xn, Xm (AND with inverted Xm).
constexpr uint32_t BicRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x8A200000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// ORN Xd, Xn, Xm (ORR with inverted Xm).
constexpr uint32_t OrnRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xAA200000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SUB Xd, Xn, Xm, LSL #shift.
constexpr uint32_t SubRegLsl(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t shift) {
  return 0xCB000000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(shift) << 10) | (rn << 5) | rd;
}
// AND Xd, Xn, Xm, LSR #shift.
constexpr uint32_t AndRegLsr(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t shift) {
  return 0x8A400000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(shift) << 10) | (rn << 5) | rd;
}
// ANDS Xd, Xn, Xm (flag-setting AND, shifted register).
constexpr uint32_t AndsRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xEA000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SUBS/ADDS Xd, Xn, Xm (flag-setting, shifted register, no shift).
constexpr uint32_t SubsRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xEB000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t AddsRegX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0xAB000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SUBS/ADDS Wd, Wn, Wm (32-bit, flag-setting, shifted register, no shift).
constexpr uint32_t SubsRegW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x6B000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t AddsRegW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x2B000000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SUBS Xd, Xn, Xm, LSL #shift.
constexpr uint32_t SubsRegLsl(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t shift) {
  return 0xEB000000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(shift) << 10) | (rn << 5) | rd;
}
// ADDS Xd, Xn, Xm, UXTB #shift (extended register, flag-setting).
constexpr uint32_t AddsExtX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t option, uint8_t shift) {
  return 0xAB200000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(option) << 13) | (static_cast<uint32_t>(shift) << 10) |
         (rn << 5) | rd;
}
// ANDS Xd, Xn, #bitmask (logical immediate, N=1 64-bit, flag-setting).
constexpr uint32_t AndsImmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xF2400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// ADD Xd, Xn, Xm, UXTB #shift (extended register).
constexpr uint32_t AddExtX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t option, uint8_t shift) {
  return 0x8B200000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(option) << 13) | (static_cast<uint32_t>(shift) << 10) |
         (rn << 5) | rd;
}
// AND/ORR/EOR Xd, Xn, #bitmask (logical immediate, N=1 64-bit).
constexpr uint32_t AndImmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0x92400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
constexpr uint32_t OrrImmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xB2400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
constexpr uint32_t EorImmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xD2400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// MADD/MSUB Xd, Xn, Xm, Xa.
constexpr uint32_t MaddX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x9B000000 | (static_cast<uint32_t>(rm) << 16) | (ra << 10) | (rn << 5) | rd;
}
constexpr uint32_t MsubX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x9B008000 | (static_cast<uint32_t>(rm) << 16) | (ra << 10) | (rn << 5) | rd;
}
// SMADDL/UMADDL Xd, Wn, Wm, Xa.
constexpr uint32_t SmaddlX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x9B200000 | (static_cast<uint32_t>(rm) << 16) | (ra << 10) | (rn << 5) | rd;
}
constexpr uint32_t UmaddlX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
  return 0x9BA00000 | (static_cast<uint32_t>(rm) << 16) | (ra << 10) | (rn << 5) | rd;
}
// LSLV/LSRV/ASRV/RORV Xd, Xn, Xm (variable shift).
constexpr uint32_t LslvX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC02000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t LsrvX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC02400 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t AsrvX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC02800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t RorvX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC02C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// UDIV Xd, Xn, Xm.
constexpr uint32_t UdivX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC00800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SMULH Xd, Xn, Xm.
constexpr uint32_t SmulhX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9B407C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SDIV Xd, Xn, Xm.
constexpr uint32_t SdivX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC00C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// UDIV Wd, Wn, Wm.
constexpr uint32_t UdivW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1AC00800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SDIV Wd, Wn, Wm.
constexpr uint32_t SdivW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1AC00C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// UMULH Xd, Xn, Xm.
constexpr uint32_t UmulhX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9BC07C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// CRC32C (Castagnoli) accumulate. Accumulator Wn and result Wd are 32-bit; the
// data operand is Wm (b/h/w) or Xm (x). Encodings verified with the aarch64
// assembler (armv8-a+crc).
//   CRC32CX Wd, Wn, Xm   CRC32CB/CH/CW Wd, Wn, Wm
constexpr uint32_t Crc32cxWWX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC05C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t Crc32cbWWW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1AC05000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t Crc32chWWW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1AC05400 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t Crc32cwWWW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1AC05800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// Reference CRC32C (Castagnoli, reflected polynomial 0x82F63B78) used to
// compute expected values without hand-coding hex. Validated in the tests
// below against the canonical CRC32C("123456789") == 0xE3069283.
inline uint32_t Crc32cRef(uint32_t crc, const uint8_t* data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) {
      crc = (crc >> 1) ^ (0x82F63B78u & (~(crc & 1u) + 1u));
    }
  }
  return crc;
}
// IEEE CRC32 (ISO-3309 poly 0x04C11DB7, reflected 0xEDB88320) — data is Wm
// (b/h/w) or Xm (x). Encodings verified with the aarch64 assembler
// (armv8-a+crc): CRC32X WWX 0x9AC04C00, CRC32B/H/W WWW 0x1AC04000/4400/4800.
constexpr uint32_t Crc32xWWX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC04C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t Crc32bWWW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1AC04000 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t Crc32hWWW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1AC04400 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
constexpr uint32_t Crc32wWWW(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x1AC04800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// Reference IEEE CRC32 (raw reflected accumulate, no ~init/~final-xor — matches
// the ARM instruction). Validated below against Python-computed vectors.
inline uint32_t Crc32Ref(uint32_t crc, const uint8_t* data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) {
      crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u));
    }
  }
  return crc;
}
// SBFM/BFM/UBFM Xd, Xn, #immr, #imms (64-bit, N=1).
constexpr uint32_t SbfmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0x93400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
constexpr uint32_t BfmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xB3400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
constexpr uint32_t UbfmX(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0xD3400000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// UBFM Wd, Wn, #immr, #imms (32-bit, N=0).
constexpr uint32_t UbfmW(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0x53000000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// SBFM Wd, Wn, #immr, #imms (32-bit, sf=0, N=0).
constexpr uint32_t SbfmW(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms) {
  return 0x13000000 | (static_cast<uint32_t>(immr) << 16) |
         (static_cast<uint32_t>(imms) << 10) | (rn << 5) | rd;
}
// ADR Xd, #offset (signed 21-bit byte offset from the instruction's PC).
constexpr uint32_t Adr(uint8_t rd, int32_t offset) {
  uint32_t imm = static_cast<uint32_t>(offset) & 0x1FFFFF;
  return 0x10000000 | ((imm & 0x3) << 29) | (((imm >> 2) & 0x7FFFF) << 5) | rd;
}
// ADRP Xd, #imm (signed 21-bit page count; target = (PC & ~0xFFF) + (imm << 12)).
constexpr uint32_t Adrp(uint8_t rd, int32_t imm) {
  uint32_t u = static_cast<uint32_t>(imm) & 0x1FFFFF;
  return 0x90000000 | ((u & 0x3) << 29) | (((u >> 2) & 0x7FFFF) << 5) | rd;
}
// MRS Xt, TPIDR_EL0.
constexpr uint32_t MrsTpidrEl0(uint8_t rt) {
  return 0xD53BD040 | rt;
}
// MRS Xt, MIDR_EL1 — a non-TPIDR system register. The heavy frontend models only
// MRS TPIDR_EL0 and declines every other MRS/MSR, so this is a stable "instruction
// that bails the optimizer" marker for the partial-region tests.
constexpr uint32_t MrsMidrEl1(uint8_t rt) {
  return 0xD5380000 | rt;
}
// EXTR Xd, Xn, Xm, #lsb (64-bit) and Wd (32-bit).
constexpr uint32_t ExtrX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t lsb) {
  return 0x93C00000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(lsb) << 10) | (rn << 5) | rd;
}
constexpr uint32_t ExtrW(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t lsb) {
  return 0x13800000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(lsb) << 10) | (rn << 5) | rd;
}
// CLZ/REV16 Xd, Xn (DP-1Src).
constexpr uint32_t ClzX(uint8_t rd, uint8_t rn) {
  return 0xDAC01000 | (rn << 5) | rd;
}
constexpr uint32_t Rev16X(uint8_t rd, uint8_t rn) {
  return 0xDAC00400 | (rn << 5) | rd;
}
// REV Xd, Xn (DP-1Src, opcode2=000011, sf=1) — 64-bit full byte reverse.
constexpr uint32_t RevX(uint8_t rd, uint8_t rn) {
  return 0xDAC00C00 | (rn << 5) | rd;
}
// REV32 Xd, Xn (DP-1Src, opcode2=000010, sf=1).
constexpr uint32_t Rev32X(uint8_t rd, uint8_t rn) {
  return 0xDAC00800 | (rn << 5) | rd;
}
// REV Wd, Wn (DP-1Src, opcode2=000010, sf=0).
constexpr uint32_t RevW(uint8_t rd, uint8_t rn) {
  return 0x5AC00800 | (rn << 5) | rd;
}
// CLS Xd, Xn (DP-1Src, opcode2=000101, sf=1).
constexpr uint32_t ClsX(uint8_t rd, uint8_t rn) {
  return 0xDAC01400 | (rn << 5) | rd;
}
// CLS Wd, Wn (DP-1Src, opcode2=000101, sf=0).
constexpr uint32_t ClsW(uint8_t rd, uint8_t rn) {
  return 0x5AC01400 | (rn << 5) | rd;
}
// RBIT Xd, Xn (DP-1Src, opcode2=000000, sf=1) — reverse bit order.
constexpr uint32_t RbitX(uint8_t rd, uint8_t rn) {
  return 0xDAC00000 | (rn << 5) | rd;
}
// RBIT Wd, Wn (DP-1Src, opcode2=000000, sf=0) — reverse low-32 bit order, zero-ext.
constexpr uint32_t RbitW(uint8_t rd, uint8_t rn) {
  return 0x5AC00000 | (rn << 5) | rd;
}

// --- Branch encoders (same forms as the lite-translator tests). ---
// B offset (unconditional, imm26 byte offset, multiple of 4).
constexpr uint32_t B(int32_t offset) {
  uint32_t imm26 = static_cast<uint32_t>(offset / 4) & 0x3FFFFFF;
  return 0x14000000 | imm26;
}
// B.cond offset (imm19 byte offset, multiple of 4).
constexpr uint32_t Bcond(uint8_t cond, int32_t offset) {
  uint32_t imm19 = static_cast<uint32_t>(offset / 4) & 0x7FFFF;
  return 0x54000000 | (imm19 << 5) | cond;
}
constexpr uint8_t kCondEQ = 0x0;
constexpr uint8_t kCondNE = 0x1;
constexpr uint8_t kCondCS = 0x2;
constexpr uint8_t kCondCC = 0x3;
constexpr uint8_t kCondMI = 0x4;
constexpr uint8_t kCondPL = 0x5;
constexpr uint8_t kCondVS = 0x6;
constexpr uint8_t kCondVC = 0x7;
constexpr uint8_t kCondHI = 0x8;
constexpr uint8_t kCondLS = 0x9;
constexpr uint8_t kCondGE = 0xA;
constexpr uint8_t kCondLT = 0xB;
constexpr uint8_t kCondGT = 0xC;
constexpr uint8_t kCondLE = 0xD;
constexpr uint8_t kCondAL = 0xE;
// CBZ/CBNZ Xt, offset (imm19 byte offset).
constexpr uint32_t CbzX(uint8_t rt, int32_t offset) {
  uint32_t imm19 = static_cast<uint32_t>(offset / 4) & 0x7FFFF;
  return 0xB4000000 | (imm19 << 5) | rt;
}
constexpr uint32_t CbnzX(uint8_t rt, int32_t offset) {
  uint32_t imm19 = static_cast<uint32_t>(offset / 4) & 0x7FFFF;
  return 0xB5000000 | (imm19 << 5) | rt;
}
// CBZ/CBNZ Wt, offset (32-bit variant: sf=0).
constexpr uint32_t CbzW(uint8_t rt, int32_t offset) {
  uint32_t imm19 = static_cast<uint32_t>(offset / 4) & 0x7FFFF;
  return 0x34000000 | (imm19 << 5) | rt;
}
// TBZ/TBNZ Xt, #bit, offset (imm14 byte offset). bit5 picks bits 32..63.
constexpr uint32_t TbzX(uint8_t rt, uint8_t bit, int32_t offset) {
  uint32_t b5 = (bit >> 5) & 1;
  uint32_t b40 = bit & 0x1F;
  uint32_t imm14 = static_cast<uint32_t>(offset / 4) & 0x3FFF;
  return 0x36000000 | (b5 << 31) | (b40 << 19) | (imm14 << 5) | rt;
}
constexpr uint32_t TbnzX(uint8_t rt, uint8_t bit, int32_t offset) {
  uint32_t b5 = (bit >> 5) & 1;
  uint32_t b40 = bit & 0x1F;
  uint32_t imm14 = static_cast<uint32_t>(offset / 4) & 0x3FFF;
  return 0x37000000 | (b5 << 31) | (b40 << 19) | (imm14 << 5) | rt;
}
// BR Xn / RET Xn (indirect).
constexpr uint32_t BrX(uint8_t rn) {
  return 0xD61F0000 | (static_cast<uint32_t>(rn) << 5);
}
constexpr uint32_t RetX(uint8_t rn) {
  return 0xD65F0000 | (static_cast<uint32_t>(rn) << 5);
}

// --- Load/store unsigned-offset encoders (same forms as the lite-translator
// tests). `imm` is the SCALED immediate (units = access size); Rn is the base.
constexpr uint32_t LdrXuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0xF9400000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t StrXuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0xF9000000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t LdrWuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0xB9400000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t StrWuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0xB9000000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t LdrbWuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0x39400000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t StrbWuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0x39000000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t LdrhWuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0x79400000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t StrhWuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0x79000000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t LdrsbXuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0x39800000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t LdrshXuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0x79800000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t LdrswXuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0xB9800000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
// LDR (literal): LDR Xt, label and LDRSW Xt, label. imm19 is the SCALED-by-4
// signed PC-relative offset.
constexpr uint32_t LdrLiteralX(uint8_t rt, int32_t off) {
  uint32_t imm19 = static_cast<uint32_t>(off / 4) & 0x7FFFF;
  return 0x58000000 | (imm19 << 5) | rt;
}
constexpr uint32_t LdrswLiteral(uint8_t rt, int32_t off) {
  uint32_t imm19 = static_cast<uint32_t>(off / 4) & 0x7FFFF;
  return 0x98000000 | (imm19 << 5) | rt;
}

// --- Conditional select / compare encoders (same forms as the lite-translator
// tests). cond is the 4-bit ARM condition (kCond* above).
// CSEL/CSINC/CSINV/CSNEG Xd, Xn, Xm, cond.
constexpr uint32_t CselX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return 0x9A800000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) | (rn << 5) | rd;
}
constexpr uint32_t CsincX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return 0x9A800400 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) | (rn << 5) | rd;
}
constexpr uint32_t CsinvX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return 0xDA800000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) | (rn << 5) | rd;
}
constexpr uint32_t CsnegX(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return 0xDA800400 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) | (rn << 5) | rd;
}
// CCMP/CCMN Xn, Xm, #nzcv, cond (register form).
constexpr uint32_t CcmpRegX(uint8_t rn, uint8_t rm, uint8_t nzcv, uint8_t cond) {
  return 0xFA400000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) | (rn << 5) | nzcv;
}
constexpr uint32_t CcmnRegX(uint8_t rn, uint8_t rm, uint8_t nzcv, uint8_t cond) {
  return 0xBA400000 | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(cond) << 12) | (rn << 5) | nzcv;
}

// --- LDP/STP encoders (signed 7-bit scaled imm; 64-bit form, scale = 8). ---
constexpr uint32_t StpX(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm7) {
  return 0xA9000000 | ((static_cast<uint32_t>(imm7) & 0x7F) << 15) |
         (static_cast<uint32_t>(rt2) << 10) | (rn << 5) | rt1;
}
constexpr uint32_t LdpX(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm7) {
  return 0xA9400000 | ((static_cast<uint32_t>(imm7) & 0x7F) << 15) |
         (static_cast<uint32_t>(rt2) << 10) | (rn << 5) | rt1;
}

// --- Register-offset load/store encoders (same forms as the lite-translator
// tests). ---
// LDR Xt, [Xn, Xm, LSL #3] (64-bit, S=1, option=011=LSL).
constexpr uint32_t LdrXregLsl3(uint8_t rt, uint8_t rn, uint8_t rm) {
  return 0xF8607800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rt;
}
// LDR Xt, [Xn, Wm, UXTW #3] (option=010=UXTW).
constexpr uint32_t LdrXregUxtw3(uint8_t rt, uint8_t rn, uint8_t rm) {
  return 0xF8605800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rt;
}
// LDR Xt, [Xn, Wm, SXTW #3] (option=110=SXTW).
constexpr uint32_t LdrXregSxtw3(uint8_t rt, uint8_t rn, uint8_t rm) {
  return 0xF860D800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rt;
}
// STR Xt, [Xn, Xm, LSL #3].
constexpr uint32_t StrXregLsl3(uint8_t rt, uint8_t rn, uint8_t rm) {
  return 0xF8207800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rt;
}
// LDR Wt, [Xn, Wm, UXTW #0] (32-bit zero-extend, no shift).
constexpr uint32_t LdrWregUxtw0(uint8_t rt, uint8_t rn, uint8_t rm) {
  return 0xB8604800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rt;
}

// Heavy-optimize and execute one instruction, mirroring the riscv64 exec-test
// harness. Returns false if the optimizing frontend bailed (didn't translate the
// instruction) so a test can assert it actually went through the JIT.
bool RunOneInstruction(ThreadState* state, GuestAddr stop_pc) {
  GuestAddr start_pc = state->cpu.insn_addr;
  MachineCode machine_code;
  auto [new_addr, success, number_of_instructions] =
      HeavyOptimizeRegion(start_pc,
                          &machine_code,
                          HeavyOptimizeParams{
                              .max_number_of_instructions = 1,
                          });
  if (!success || number_of_instructions != 1) {
    return false;
  }

  // The TranslationCache is a process-global singleton shared across tests. A
  // prior test that exited to a guest PC outside its own region (e.g.
  // BranchCondTargetOutsideRegionExits) can leave a non-default entry at an
  // address that, by static-array layout, collides with this test's stop_pc;
  // SetStop then fails to install the stop and the dispatcher reaches the stale
  // entry -> berberis_HandleNoExec with a null guest thread. Clear any stale
  // entries for this test's PC window so SetStop sees the default state.
  TranslationCache::GetInstance()->InvalidateGuestRange(start_pc, stop_pc + 4);

  ScopedExecRegion exec(&machine_code);
  TestingRunGeneratedCode(state, exec.get(), stop_pc);
  return true;
}

class Arm64HeavyOptimizerFrontendTest : public ::testing::Test {
 protected:
  ThreadState state_{};
};

TEST_F(Arm64HeavyOptimizerFrontendTest, MovzX0Basic) {
  static const uint32_t code[] = {MovzX(0, 0x1234)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.insn_addr, stop_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1234});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovzWithShift) {
  // MOVZ X1, #0xABCD, LSL #16 -> 0xABCD0000.
  static const uint32_t code[] = {MovzHwX(1, 0xABCD, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0xABCD0000});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovnBasic) {
  // MOVN X2, #0 -> ~0 = 0xFFFFFFFFFFFFFFFF.
  static const uint32_t code[] = {MovnX(2, 0)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0xFFFFFFFFFFFFFFFF});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovzW32BitZeroExtends) {
  // MOVZ W3, #0x5678 -> upper 32 bits cleared.
  static const uint32_t code[] = {MovzW(3, 0x5678)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[3], uint64_t{0x5678});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovkKeepsOtherBits) {
  // X4 = 0x1111, then MOVK X4, #0x2222, LSL #16 -> 0x22221111.
  static const uint32_t setup[] = {MovzX(4, 0x1111)};
  state_.cpu.insn_addr = ToGuestAddr(setup);
  GuestAddr setup_stop = ToGuestAddr(setup) + sizeof(setup);
  ASSERT_TRUE(RunOneInstruction(&state_, setup_stop));
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0x1111});

  static const uint32_t movk[] = {MovkHwX(4, 0x2222, 1)};
  state_.cpu.insn_addr = ToGuestAddr(movk);
  GuestAddr movk_stop = ToGuestAddr(movk) + sizeof(movk);
  ASSERT_TRUE(RunOneInstruction(&state_, movk_stop));
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0x22221111});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MovkHighWindow) {
  // X5 = 0, then MOVK X5, #0xBEEF, LSL #48 -> 0xBEEF000000000000.
  static const uint32_t setup[] = {MovzX(5, 0)};
  state_.cpu.insn_addr = ToGuestAddr(setup);
  GuestAddr setup_stop = ToGuestAddr(setup) + sizeof(setup);
  ASSERT_TRUE(RunOneInstruction(&state_, setup_stop));

  static const uint32_t movk[] = {MovkHwX(5, 0xBEEF, 3)};
  state_.cpu.insn_addr = ToGuestAddr(movk);
  GuestAddr movk_stop = ToGuestAddr(movk) + sizeof(movk);
  ASSERT_TRUE(RunOneInstruction(&state_, movk_stop));
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0xBEEF000000000000});
}

// Repro: multi-instruction regions (what the runtime actually produces) must
// pass GenCode's CheckMachineIR. The 1-insn tests above never exercised this.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiMoveWideRegion) {
  static const uint32_t code[] = {MovzX(0, 0x11), MovzX(1, 0x22)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_TRUE(ok);
  EXPECT_EQ(n, 2u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MoveWideThenBailRegion) {
  // MRS MIDR_EL1 (non-modeled system register) bails after the two MoveWides.
  static const uint32_t code[] = {MovzX(0, 0x11), MovzX(1, 0x22), MrsMidrEl1(2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // 2 MoveWide translate, the MRS bails -> partial region.
  EXPECT_EQ(n, 2u);
}

// A single guest instruction whose decode fires several listener callbacks where
// a later one bails must still produce valid IR (one region exit, no trailing
// insns) — i.e. GenCode must not abort. A SIMD LDP-Q post-index decodes to
// SimdLoadStorePair (still bails this round) followed by an AddImm writeback;
// once SimdLoadStorePair sets success_ = false the AddImm must emit nothing.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiCallbackBailIsValidIR) {
  // LDR H0, [X1], #2 (SIMD 16-bit post-index): SimdLoadStoreImm (bails on the
  // H/B sizes) then the writeback callback -- a multi-callback bail.
  static const uint32_t code[] = {0x7c402420};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  // No abort here is the assertion (GenCode runs CheckMachineIR internally).
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_FALSE(ok);
  EXPECT_EQ(n, 0u);
}

// MoveWide followed by a multi-callback bail (the common on-device prefix shape).
TEST_F(Arm64HeavyOptimizerFrontendTest, MoveWideThenMultiCallbackBail) {
  static const uint32_t code[] = {MovzX(0, 0x11),
                                  0x7c402420 /*LDR H0,[X1],#2 SIMD 16-bit post-index, bails*/};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 1u);  // the MOVZ translated; the SIMD pair load bailed without corrupting IR
}

//
// Non-flag integer ALU value checks (full pipeline via RunOneInstruction).
//

TEST_F(Arm64HeavyOptimizerFrontendTest, AddImm64) {
  static const uint32_t code[] = {AddImmX(0, 1, 0x123)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1123});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SubImm64) {
  static const uint32_t code[] = {SubImmX(0, 1, 0x10)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFF0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AddImm32ZeroExtends) {
  // ADD W0, W1, #1: result is zero-extended into X0.
  static const uint32_t code[] = {AddImmW(0, 1, 1)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;  // W1 = 0xFFFFFFFF
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});  // 0xFFFFFFFF + 1 = 0, upper cleared
}

//
// Flag-setting integer ALU: assert BOTH the result register AND cpu.flags.
// NZCV packing matches the lite translator: N=bit15, Z=bit14, C=bit8, V=bit0.
//

// SUBS producing Z=1, C=1 (equal operands -> no borrow).
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsImmEqualSetsZC) {
  // SUBS X0, X1, #5 with X1 == 5.
  static const uint32_t code[] = {SubsImmX(0, 1, 5)};
  state_.cpu.x[1] = 5;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);  // no borrow
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// SUBS producing N=1, C=0 (borrow: smaller minus larger).
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsImmBorrowSetsNC) {
  // SUBS X0, X1, #10 with X1 == 5 -> -5, borrow.
  static const uint32_t code[] = {SubsImmX(0, 1, 10)};
  state_.cpu.x[1] = 5;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFFBULL});  // -5
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);  // ARM borrow -> C=0
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// SUBS signed overflow: INT64_MIN - 1 overflows -> V=1.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsImmSignedOverflow) {
  // SUBS X0, X1, #1 with X1 == INT64_MIN -> wraps to INT64_MAX, V=1.
  static const uint32_t code[] = {SubsImmX(0, 1, 1)};
  state_.cpu.x[1] = 0x8000000000000000ULL;  // INT64_MIN
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x7FFFFFFFFFFFFFFFULL});  // INT64_MAX
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagOverflow);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);  // no unsigned borrow
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
}

// ADDS producing C=1 (unsigned carry-out).
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsImmCarryOut) {
  // ADDS X0, X1, #1 with X1 == UINT64_MAX -> 0, carry out.
  static const uint32_t code[] = {AddsImmX(0, 1, 1)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// ADDS producing V=1 (signed overflow): INT64_MAX + 1.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsImmSignedOverflow) {
  // ADDS X0, X1, #1 with X1 == INT64_MAX -> INT64_MIN, V=1, N=1.
  static const uint32_t code[] = {AddsImmX(0, 1, 1)};
  state_.cpu.x[1] = 0x7FFFFFFFFFFFFFFFULL;  // INT64_MAX
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x8000000000000000ULL});  // INT64_MIN
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagOverflow);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
}

// 32-bit SUBS (W form): borrow case with zero-extended result.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsImmW32) {
  // SUBS W0, W1, #1 with W1 == 0 -> 0xFFFFFFFF, N=1, C=0 (borrow).
  static const uint32_t code[] = {SubsImmW(0, 1, 1)};
  state_.cpu.x[1] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFULL});  // W-result zero-extended
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
}

// 32-bit ADDS (W form): carry-out at the 32-bit boundary, Z=1.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsImmW32CarryOut) {
  // ADDS W0, W1, #1 with W1 == 0xFFFFFFFF -> 0, C=1, Z=1.
  static const uint32_t code[] = {AddsImmW(0, 1, 1)};
  state_.cpu.x[1] = 0xFFFFFFFFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// CMP (SUBS to XZR): flags set, result discarded, x[0] untouched.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmpImmDiscardsResult) {
  // X0 holds a sentinel; CMP X1, #5 (== SUBS XZR, X1, #5) with X1 == 5.
  static const uint32_t code[] = {CmpImmX(1, 5)};
  state_.cpu.x[0] = 0xDEADBEEFCAFEF00DULL;  // sentinel: must not be clobbered
  state_.cpu.x[1] = 5;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xDEADBEEFCAFEF00DULL});  // untouched
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
}

// SUBS shifted register: X0 = X1 - (X2 << 4), flags set.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsRegShifted) {
  // SUBS X0, X1, X2, LSL #4 with X1 == 0x100, X2 == 0x10 -> 0x100 - 0x100 = 0.
  static const uint32_t code[] = {SubsRegLsl(0, 1, 2, 4)};
  state_.cpu.x[1] = 0x100;
  state_.cpu.x[2] = 0x10;  // << 4 = 0x100
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
}

// SUBS register (no shift): borrow case.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsRegBorrow) {
  // SUBS X0, X1, X2 with X1 < X2 -> borrow (C=0), N=1.
  static const uint32_t code[] = {SubsRegX(0, 1, 2)};
  state_.cpu.x[1] = 3;
  state_.cpu.x[2] = 10;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFF9ULL});  // -7
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
}

// ADDS register (no shift): carry-out and overflow.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsRegCarryAndOverflow) {
  // ADDS X0, X1, X2 with X1 == X2 == INT64_MIN -> 0, C=1 (carry out), V=1.
  static const uint32_t code[] = {AddsRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x8000000000000000ULL;
  state_.cpu.x[2] = 0x8000000000000000ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagOverflow);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
}

// 32-bit SUBS register: equal operands -> Z=1, C=1, zero-extended result.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubsRegW32Equal) {
  static const uint32_t code[] = {SubsRegW(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFF00001234ULL;  // W1 = 0x1234
  state_.cpu.x[2] = 0x1234;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});  // upper 32 cleared
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
}

// 32-bit ADDS register: carry-out at 32-bit boundary.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsRegW32CarryOut) {
  static const uint32_t code[] = {AddsRegW(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFFULL;
  state_.cpu.x[2] = 1;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
}

// ADDS extended register: X0 = X1 + ((X2 & 0xFF) << 2), flags set.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddsExtendedUxtb) {
  // ADDS X0, X1, X2, UXTB #2 with low byte of X2 == 0x80.
  static const uint32_t code[] = {AddsExtX(0, 1, 2, /*UXTB=*/0b000, 2)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.x[2] = 0xFFFFFF80ULL;  // low byte 0x80
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1000 + (0x80 << 2)});
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// ANDS producing Z=1 (result zero), and C==0 && V==0 always for AND.
TEST_F(Arm64HeavyOptimizerFrontendTest, AndsRegZeroSetsZ) {
  // ANDS X0, X1, X2 with disjoint bits -> 0, Z=1, C=0, V=0.
  static const uint32_t code[] = {AndsRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x00FF;
  state_.cpu.x[2] = 0xFF00;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);     // AND clears C
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);  // AND clears V
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
}

// ANDS producing N=1 (MSB set), and C==0 && V==0.
TEST_F(Arm64HeavyOptimizerFrontendTest, AndsRegNegativeSetsN) {
  // ANDS X0, X1, X2 with top bit kept -> N=1, C=0, V=0.
  static const uint32_t code[] = {AndsRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x8000000000000001ULL;
  state_.cpu.x[2] = 0x8000000000000000ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x8000000000000000ULL});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// ANDS logical immediate: Z=1, and C==0 && V==0.
TEST_F(Arm64HeavyOptimizerFrontendTest, AndsImmZeroSetsZ) {
  // ANDS X0, X1, #0xFF with X1 having no low byte -> 0, Z=1.
  static const uint32_t code[] = {AndsImmX(0, 1, 0, 7)};
  state_.cpu.x[1] = 0xAB00;  // low byte zero
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// TST (ANDS to XZR): flags set, result discarded.
TEST_F(Arm64HeavyOptimizerFrontendTest, TstImmDiscardsResult) {
  // TST X1, #0xFF == ANDS XZR, X1, #0xFF. X0 sentinel must be preserved.
  static const uint32_t code[] = {AndsImmX(31, 1, 0, 7)};
  state_.cpu.x[0] = 0x1234567890ABCDEFULL;  // sentinel
  state_.cpu.x[1] = 0xAB00;                 // low byte zero -> Z=1
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1234567890ABCDEFULL});  // untouched
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
}

// ADC/SBC (add/subtract with carry) bails: needs the carry-in from cpu.flags.
// ADC/SBC (no flags) with a carry-in: the heavy tier now mirrors the lite
// Btw/Cmc/Adc(Sbb) path instead of bailing.
TEST_F(Arm64HeavyOptimizerFrontendTest, AdcSbcCarryIn) {
  // ADC X0, X1, X2 with carry-in=1: 5 + 10 + 1 = 16 (ADC does not set flags).
  static const uint32_t adc[] = {0x9A020020u};  // ADC X0, X1, X2
  state_.cpu.flags = CPUState::kFlagCarry;
  state_.cpu.x[1] = 5;
  state_.cpu.x[2] = 10;
  state_.cpu.insn_addr = ToGuestAddr(adc);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(adc) + sizeof(adc)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{16});
  EXPECT_EQ(state_.cpu.flags, uint16_t{CPUState::kFlagCarry});  // unchanged

  // ADC with carry-in=0: 5 + 10 + 0 = 15.
  state_.cpu.flags = 0;
  state_.cpu.x[1] = 5;
  state_.cpu.x[2] = 10;
  state_.cpu.insn_addr = ToGuestAddr(adc);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(adc) + sizeof(adc)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{15});

  // SBC X0, X1, X2 with carry-in=1: 100 - 30 - (1-1) = 70.
  static const uint32_t sbc[] = {0xDA020020u};  // SBC X0, X1, X2
  state_.cpu.flags = CPUState::kFlagCarry;
  state_.cpu.x[1] = 100;
  state_.cpu.x[2] = 30;
  state_.cpu.insn_addr = ToGuestAddr(sbc);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(sbc) + sizeof(sbc)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{70});

  // SBC with carry-in=0: 100 - 30 - (1-0) = 69.
  state_.cpu.flags = 0;
  state_.cpu.x[1] = 100;
  state_.cpu.x[2] = 30;
  state_.cpu.insn_addr = ToGuestAddr(sbc);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(sbc) + sizeof(sbc)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{69});
}

// ADCS carry-out on a saturated addend (the RSA/TLS bignum case) — the naive
// carry-fold loses the carry; verify the fused Adc keeps it.
TEST_F(Arm64HeavyOptimizerFrontendTest, AdcsSaturatedCarryOut) {
  static const uint32_t adcs[] = {0xBA020020u};  // ADCS X0, X1, X2
  // carry-in=1, x1=5, x2=all-ones -> x0=5, carry-out=1, Z=0.
  state_.cpu.flags = CPUState::kFlagCarry;
  state_.cpu.x[1] = 5;
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFFFFULL;
  state_.cpu.insn_addr = ToGuestAddr(adcs);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(adcs) + sizeof(adcs)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{5});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry) << "carry-out lost";
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagZero);

  // carry-in=1, x1=0, x2=all-ones -> x0=0, carry-out=1, Z=1.
  state_.cpu.flags = CPUState::kFlagCarry;
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFFFFULL;
  state_.cpu.insn_addr = ToGuestAddr(adcs);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(adcs) + sizeof(adcs)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
}

// SBCS flags: ARM carry = "no borrow" (inverted from x86 CF).
TEST_F(Arm64HeavyOptimizerFrontendTest, SbcsBorrowFlags) {
  static const uint32_t sbcs[] = {0xFA020020u};  // SBCS X0, X1, X2
  // carry-in=1, no borrow: 100 - 30 = 70, ARM C=1.
  state_.cpu.flags = CPUState::kFlagCarry;
  state_.cpu.x[1] = 100;
  state_.cpu.x[2] = 30;
  state_.cpu.insn_addr = ToGuestAddr(sbcs);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(sbcs) + sizeof(sbcs)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{70});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);  // no borrow -> C=1
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);

  // carry-in=1, borrow: 30 - 100 = -70, ARM C=0, N=1.
  state_.cpu.flags = CPUState::kFlagCarry;
  state_.cpu.x[1] = 30;
  state_.cpu.x[2] = 100;
  state_.cpu.insn_addr = ToGuestAddr(sbcs);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(sbcs) + sizeof(sbcs)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFBAULL});  // -70
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagCarry);  // borrow -> C=0
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagNegative);
}

// 32-bit ADCS (W form) zero-extends the result, and NGC (SBC with rn=XZR).
TEST_F(Arm64HeavyOptimizerFrontendTest, Adcs32AndNgc) {
  // ADCS W0, W1, W2: carry-in=1, w1=5, w2=0xFFFFFFFF -> w0=5 (wraps), C=1.
  static const uint32_t adcs32[] = {0x3A020020u};  // ADCS W0, W1, W2
  state_.cpu.flags = CPUState::kFlagCarry;
  state_.cpu.x[1] = 5;
  state_.cpu.x[2] = 0xFFFFFFFFULL;
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  state_.cpu.insn_addr = ToGuestAddr(adcs32);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(adcs32) + sizeof(adcs32)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{5});  // upper 32 zero-extended
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);

  // NGC X0, X2 = SBC X0, XZR, X2. carry-in=1: 0 - 5 - 0 = -5.
  static const uint32_t ngc[] = {0xDA0203E0u};  // NGC X0, X2
  state_.cpu.flags = CPUState::kFlagCarry;
  state_.cpu.x[2] = 5;
  state_.cpu.insn_addr = ToGuestAddr(ngc);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ngc) + sizeof(ngc)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFFBULL});  // -5
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AddReg64) {
  static const uint32_t code[] = {AddRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x100;
  state_.cpu.x[2] = 0x023;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x123});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SubReg64) {
  static const uint32_t code[] = {SubRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x500;
  state_.cpu.x[2] = 0x123;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x500 - 0x123});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AndReg64) {
  static const uint32_t code[] = {AndRegX(0, 1, 2)};
  state_.cpu.x[1] = 0xFF0F;
  state_.cpu.x[2] = 0x0FF0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFF0F & 0x0FF0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SubRegShifted) {
  // SUB X0, X1, X2, LSL #4 -> 0x1000 - (0x10 << 4) = 0x1000 - 0x100 = 0xF00.
  static const uint32_t code[] = {SubRegLsl(0, 1, 2, 4)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.x[2] = 0x10;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xF00});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AndRegLsrShifted) {
  // AND X0, X1, X2, LSR #4.
  static const uint32_t code[] = {AndRegLsr(0, 1, 2, 4)};
  state_.cpu.x[1] = 0x0F0F;
  state_.cpu.x[2] = 0xFF00;  // >>4 = 0x0FF0
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0F0F & 0x0FF0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, OrrReg64) {
  static const uint32_t code[] = {OrrRegX(0, 1, 2)};
  state_.cpu.x[1] = 0xF0F0;
  state_.cpu.x[2] = 0x0F0F;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFF});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, EorReg64) {
  static const uint32_t code[] = {EorRegX(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFF;
  state_.cpu.x[2] = 0x0F0F;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xF0F0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BicReg64) {
  // BIC X0, X1, X2 -> X1 & ~X2.
  static const uint32_t code[] = {BicRegX(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFF;
  state_.cpu.x[2] = 0x0F0F;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xF0F0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, OrnReg64) {
  // ORN X0, X1, X2 -> X1 | ~X2.
  static const uint32_t code[] = {OrnRegX(0, 1, 2)};
  state_.cpu.x[1] = 0x00FF;
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFF00ULL;  // ~X2 = 0xFF
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFF});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AddExtendedUxtb) {
  // ADD X0, X1, X2, UXTB #2 -> X1 + ((X2 & 0xFF) << 2).
  static const uint32_t code[] = {AddExtX(0, 1, 2, /*UXTB=*/0b000, 2)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.x[2] = 0xFFFFFF80ULL;  // low byte 0x80
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1000 + (0x80 << 2)});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AddExtendedSxtb) {
  // ADD X0, X1, X2, SXTB -> X1 + sext(X2[7:0]).
  static const uint32_t code[] = {AddExtX(0, 1, 2, /*SXTB=*/0b100, 0)};
  state_.cpu.x[1] = 0x1000;
  state_.cpu.x[2] = 0x80;  // sign-extends to -128
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1000 - 128});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AndImm64) {
  // AND X0, X1, #0xFF (N=1, immr=0, imms=7).
  static const uint32_t code[] = {AndImmX(0, 1, 0, 7)};
  state_.cpu.x[1] = 0x12345678;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x78});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, OrrImm64) {
  // ORR X0, X1, #0xF (N=1, immr=0, imms=3).
  static const uint32_t code[] = {OrrImmX(0, 1, 0, 3)};
  state_.cpu.x[1] = 0x1230;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x123F});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, EorImm64) {
  // EOR X0, X1, #0xFF (N=1, immr=0, imms=7).
  static const uint32_t code[] = {EorImmX(0, 1, 0, 7)};
  state_.cpu.x[1] = 0x12FF;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1200});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Madd64) {
  // MADD X0, X1, X2, X3 -> X3 + X1*X2.
  static const uint32_t code[] = {MaddX(0, 1, 2, 3)};
  state_.cpu.x[1] = 6;
  state_.cpu.x[2] = 7;
  state_.cpu.x[3] = 5;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{5 + 6 * 7});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Msub64) {
  // MSUB X0, X1, X2, X3 -> X3 - X1*X2.
  static const uint32_t code[] = {MsubX(0, 1, 2, 3)};
  state_.cpu.x[1] = 6;
  state_.cpu.x[2] = 7;
  state_.cpu.x[3] = 100;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{100 - 6 * 7});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Smaddl) {
  // SMADDL X0, W1, W2, X3 -> X3 + sext(W1)*sext(W2).
  static const uint32_t code[] = {SmaddlX(0, 1, 2, 3)};
  state_.cpu.x[1] = 0xFFFFFFFFULL;  // W1 = -1
  state_.cpu.x[2] = 5;
  state_.cpu.x[3] = 100;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{100 - 5});  // 100 + (-1)*5
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Umaddl) {
  // UMADDL X0, W1, W2, X3 -> X3 + zext(W1)*zext(W2).
  static const uint32_t code[] = {UmaddlX(0, 1, 2, 3)};
  state_.cpu.x[1] = 0xFFFFFFFFULL;  // W1 = 4294967295
  state_.cpu.x[2] = 2;
  state_.cpu.x[3] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFULL * 2});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Umulh64) {
  // UMULH: high 64 bits of an unsigned 128-bit product.
  static const uint32_t code[] = {UmulhX(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFFFFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  // (2^64-1)^2 = 2^128 - 2^65 + 1; high 64 bits = 0xFFFFFFFFFFFFFFFE.
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFFEULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Smulh64) {
  // SMULH: high 64 bits of a signed 128-bit product. (-1) * (-1) = 1 -> hi = 0.
  static const uint32_t code[] = {SmulhX(0, 1, 2)};
  state_.cpu.x[1] = static_cast<uint64_t>(-1);
  state_.cpu.x[2] = static_cast<uint64_t>(-1);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Smulh64NegativeProduct) {
  // INT64_MIN * 2 (signed): product = -(2^64), high 64 bits = 0xFFFFFFFFFFFFFFFF.
  static const uint32_t code[] = {SmulhX(0, 1, 2)};
  state_.cpu.x[1] = 0x8000000000000000ULL;  // INT64_MIN
  state_.cpu.x[2] = 2;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFFFULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Lslv64) {
  static const uint32_t code[] = {LslvX(0, 1, 2)};
  state_.cpu.x[1] = 0x1;
  state_.cpu.x[2] = 8;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x100});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Lsrv64) {
  static const uint32_t code[] = {LsrvX(0, 1, 2)};
  state_.cpu.x[1] = 0x100;
  state_.cpu.x[2] = 4;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x10});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Asrv64) {
  static const uint32_t code[] = {AsrvX(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFF00ULL;  // -256
  state_.cpu.x[2] = 4;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFF0ULL});  // -16
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Rorv64) {
  static const uint32_t code[] = {RorvX(0, 1, 2)};
  state_.cpu.x[1] = 0x0000000000000001ULL;
  state_.cpu.x[2] = 4;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1000000000000000ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Udiv64) {
  static const uint32_t code[] = {UdivX(0, 1, 2)};
  state_.cpu.x[1] = 100;
  state_.cpu.x[2] = 7;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{100 / 7});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Udiv64ByZeroReturnsZero) {
  static const uint32_t code[] = {UdivX(0, 1, 2)};
  state_.cpu.x[1] = 100;
  state_.cpu.x[2] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Udiv32ZeroExtends) {
  static const uint32_t code[] = {UdivW(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;  // W1 = 0xFFFFFFFF = 4294967295
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFF00ULL;  // W2 = 0xFFFFFF00 = 4294967040
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  // 4294967295 / 4294967040 = 1; upper 32 bits cleared.
  EXPECT_EQ(state_.cpu.x[0], uint64_t{1});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Sdiv64) {
  static const uint32_t code[] = {SdivX(0, 1, 2)};
  state_.cpu.x[1] = static_cast<uint64_t>(-100);
  state_.cpu.x[2] = 7;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(int64_t{-100} / 7));  // -14
}

// Confirms the reference CRC32C matches the canonical published vector, so the
// expected values computed from it in the heavy tests below are trustworthy.
// Crc32cRef models the raw hardware CRC32C accumulate (the ARM CRC32C / x86
// CRC32 instruction: no initial preset, no final inversion). The canonical
// CRC-32C/ISCSI checksum of "123456789" (0xE3069283) applies init=0xFFFFFFFF
// and a final XOR of 0xFFFFFFFF, i.e. ~Crc32cRef(0xFFFFFFFF, msg).
TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32cReferenceMatchesCanonicalVector) {
  const uint8_t kCheck[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  EXPECT_EQ(~Crc32cRef(0xFFFFFFFFu, kCheck, sizeof(kCheck)), uint32_t{0xE3069283});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32cx64BitData) {
  static const uint32_t code[] = {Crc32cxWWX(0, 1, 2)};
  state_.cpu.x[1] = 0;                     // Wn accumulator
  state_.cpu.x[2] = 0x0123456789ABCDEFULL;  // Xm data
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  // The 64-bit data is consumed as 8 little-endian bytes (x86 CRC32 and ARM
  // CRC32CX process the operand identically).
  const uint8_t data[8] = {0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32cRef(0, data, 8)});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32cxNonZeroAccumulator) {
  static const uint32_t code[] = {Crc32cxWWX(0, 1, 2)};
  state_.cpu.x[1] = 0xDEADBEEF;             // Wn accumulator (non-zero)
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFFFFULL;  // Xm data
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32cRef(0xDEADBEEF, data, 8)});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32cbByteData) {
  static const uint32_t code[] = {Crc32cbWWW(0, 1, 2)};
  state_.cpu.x[1] = 0x12345678;  // Wn accumulator
  state_.cpu.x[2] = 0xAABBCCEF;  // only the low byte (0xEF) is consumed
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[1] = {0xEF};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32cRef(0x12345678, data, 1)});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32chHalfwordData) {
  static const uint32_t code[] = {Crc32chWWW(0, 1, 2)};
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 0xAABBCDEF;  // low halfword (0xCDEF) consumed, LE bytes
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[2] = {0xEF, 0xCD};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32cRef(0, data, 2)});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32cwWordData) {
  static const uint32_t code[] = {Crc32cwWWW(0, 1, 2)};
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 0x89ABCDEF;  // low word consumed, LE bytes
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[4] = {0xEF, 0xCD, 0xAB, 0x89};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32cRef(0, data, 4)});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32IeeeReferenceMatchesVectors) {
  const uint8_t b[1] = {0xAB};
  EXPECT_EQ(Crc32Ref(0, b, 1), uint32_t{0x41047A60});
  const uint8_t w[4] = {0xEF, 0xBE, 0xAD, 0xDE};   // 0xDEADBEEF, LE
  EXPECT_EQ(Crc32Ref(0, w, 4), uint32_t{0x3B1EBF03});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32xIeee64BitData) {
  if (!host_platform::kHasCLMUL) GTEST_SKIP();
  static const uint32_t code[] = {Crc32xWWX(0, 1, 2)};
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 0x0123456789ABCDEFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[8] = {0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32Ref(0, data, 8)});  // 0x21193D2E
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32xIeeeNonZeroAccumulator) {
  if (!host_platform::kHasCLMUL) GTEST_SKIP();
  static const uint32_t code[] = {Crc32xWWX(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFFF;
  state_.cpu.x[2] = 0x0123456789ABCDEFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[8] = {0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32Ref(0xFFFFFFFF, data, 8)});  // 0xBBC41DB8
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32wIeeeWordData) {
  if (!host_platform::kHasCLMUL) GTEST_SKIP();
  static const uint32_t code[] = {Crc32wWWW(0, 1, 2)};
  state_.cpu.x[1] = 0x0F0F0F0F;
  state_.cpu.x[2] = 0x12345678;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[4] = {0x78, 0x56, 0x34, 0x12};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32Ref(0x0F0F0F0F, data, 4)});  // 0xCA310EFB
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32hIeeeHalfwordData) {
  if (!host_platform::kHasCLMUL) GTEST_SKIP();
  static const uint32_t code[] = {Crc32hWWW(0, 1, 2)};
  state_.cpu.x[1] = 0;
  state_.cpu.x[2] = 0x0000BEEF;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[2] = {0xEF, 0xBE};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32Ref(0, data, 2)});  // 0xF53F71A8
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Crc32bIeeeByteData) {
  if (!host_platform::kHasCLMUL) GTEST_SKIP();
  static const uint32_t code[] = {Crc32bWWW(0, 1, 2)};
  state_.cpu.x[1] = 0x12345678;
  state_.cpu.x[2] = 0x000000CD;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  const uint8_t data[1] = {0xCD};
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{Crc32Ref(0x12345678, data, 1)});  // 0xBB197355
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Sdiv64ByZeroReturnsZero) {
  static const uint32_t code[] = {SdivX(0, 1, 2)};
  state_.cpu.x[1] = static_cast<uint64_t>(-100);
  state_.cpu.x[2] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Sdiv64IntMinByMinusOne) {
  // INT64_MIN / -1 -> INT64_MIN (ARM returns INT_MIN; x86 IDIV would #DE).
  static const uint32_t code[] = {SdivX(0, 1, 2)};
  state_.cpu.x[1] = 0x8000000000000000ULL;  // INT64_MIN
  state_.cpu.x[2] = static_cast<uint64_t>(-1);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x8000000000000000ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Sdiv32IntMinByMinusOne) {
  // INT32_MIN / -1 -> INT32_MIN, zero-extended into X0.
  static const uint32_t code[] = {SdivW(0, 1, 2)};
  state_.cpu.x[1] = 0x80000000ULL;          // W1 = INT32_MIN
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFFFFULL;  // W2 = -1
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x80000000ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Sdiv32NegByZeroReturnsZero) {
  static const uint32_t code[] = {SdivW(0, 1, 2)};
  state_.cpu.x[1] = 0xFFFFFFF0ULL;  // W1 = -16
  state_.cpu.x[2] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UbfmLsr64) {
  // LSR X0, X1, #8  == UBFM X0, X1, #8, #63.
  static const uint32_t code[] = {UbfmX(0, 1, 8, 63)};
  state_.cpu.x[1] = 0x1234567800000000ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0012345678000000ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UbfmLsl64) {
  // LSL X0, X1, #4 == UBFM X0, X1, #(64-4), #(63-4) = #60, #59.
  static const uint32_t code[] = {UbfmX(0, 1, 60, 59)};
  state_.cpu.x[1] = 0x12;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x120});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UbfmUxtbW) {
  // UXTB W0, W1 == UBFM W0, W1, #0, #7.
  static const uint32_t code[] = {UbfmW(0, 1, 0, 7)};
  state_.cpu.x[1] = 0x1234ABCD;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xCD});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UbfmExtract64) {
  // UBFX X0, X1, #8, #8 (extract 8 bits starting at bit 8) == UBFM X0,X1,#8,#15.
  static const uint32_t code[] = {UbfmX(0, 1, 8, 15)};
  state_.cpu.x[1] = 0x0000000000ABCDEFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xCD});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfmAsr64) {
  // ASR X0, X1, #4 == SBFM X0, X1, #4, #63.
  static const uint32_t code[] = {SbfmX(0, 1, 4, 63)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFF00ULL;  // -256
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFF0ULL});  // -16
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfmSxtb64) {
  // SXTB X0, W1 == SBFM X0, X1, #0, #7.
  static const uint32_t code[] = {SbfmX(0, 1, 0, 7)};
  state_.cpu.x[1] = 0x80;  // sign bit set -> -128
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFF80ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfmSxtw64) {
  // SXTW X0, W1 == SBFM X0, X1, #0, #31.
  static const uint32_t code[] = {SbfmX(0, 1, 0, 31)};
  state_.cpu.x[1] = 0x80000000ULL;  // sign bit set -> negative
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFF80000000ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfxX) {
  // SBFX X0, X1, #8, #8 (extract bits [15:8], sign-extend) == SBFM X0,X1,#8,#15.
  static const uint32_t code[] = {SbfmX(0, 1, 8, 15)};
  state_.cpu.x[1] = 0x000000000000A500ULL;  // byte at [15:8] = 0xA5 (sign bit set)
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFA5ULL});  // sign-extended
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfxXPositive) {
  // SBFX X0, X1, #8, #8 with a non-negative field (top bit clear).
  static const uint32_t code[] = {SbfmX(0, 1, 8, 15)};
  state_.cpu.x[1] = 0x0000000000007F00ULL;  // byte at [15:8] = 0x7F
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x7F});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfxWOneBit) {
  // SBFX W0, W1, #0, #1 (the bitwise-CRC inner loop's bit-0 sign-extend).
  // W-write zeroes the upper 32 bits of X0.
  static const uint32_t code[] = {SbfmW(0, 1, 0, 0)};
  state_.cpu.x[1] = 0x00000001ULL;  // bit 0 set
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x00000000FFFFFFFFULL});  // all ones in low 32
  state_.cpu.x[1] = 0x00000000ULL;  // bit 0 clear
  state_.cpu.x[0] = 0xdeadbeefdeadbeefULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AdrPositive) {
  // ADR X0, #0x100: X0 = insn_addr + 0x100.
  static const uint32_t code[] = {Adr(0, 0x100)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], ToGuestAddr(code) + 0x100);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AdrpPageAligned) {
  // ADRP X0, #2: X0 = (insn_addr & ~0xFFF) + (2 << 12).
  static const uint32_t code[] = {Adrp(0, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], (ToGuestAddr(code) & ~GuestAddr{0xFFF}) + (GuestAddr{2} << 12));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MrsTpidrEl0) {
  // MRS X0, TPIDR_EL0 reads ThreadState.tls.
  static const uint32_t code[] = {MrsTpidrEl0(0)};
  state_.tls = 0x1234567890ABCDEFULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1234567890ABCDEFULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BfmBfi64) {
  // BFI X0, X1, #8, #8: insert low 8 bits of X1 at bit 8, keep other X0 bits.
  // == BFM X0, X1, #(64-8)=#56, #(8-1)=#7.
  static const uint32_t code[] = {BfmX(0, 1, 56, 7)};
  state_.cpu.x[0] = 0xFFFFFFFFFFFFFFFFULL;
  state_.cpu.x[1] = 0xAB;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFABFFULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BfmBfxil64) {
  // BFXIL X0, X1, #8, #8: take bits[15:8] of X1 into bits[7:0] of X0, keep rest.
  // == BFM X0, X1, #immr=8, #imms=15.
  static const uint32_t code[] = {BfmX(0, 1, 8, 15)};
  state_.cpu.x[0] = 0xFFFFFFFFFFFFFF00ULL;
  state_.cpu.x[1] = 0xAB00;  // bits[15:8] = 0xAB
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFABULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, ExtrCopyLsb0) {
  // EXTR X0, X1, X2, #0 -> copy of X2.
  static const uint32_t code[] = {ExtrX(0, 1, 2, 0)};
  state_.cpu.x[1] = 0x1111111111111111ULL;
  state_.cpu.x[2] = 0x2222222222222222ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x2222222222222222ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Extr32) {
  // EXTR W0, W1, W2, #4: W0 = (W1:W2) >> 4 (low 32).
  static const uint32_t code[] = {ExtrW(0, 1, 2, 4)};
  state_.cpu.x[1] = 0x0000000F;  // W1 low nibble -> top of result
  state_.cpu.x[2] = 0xABCDEF00;  // W2
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  // (0x0000000F:0xABCDEF00) >> 4 = 0xFABCDEF0.
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFABCDEF0ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Extr64) {
  // EXTR X0, X1, X2, #20: X0 = (X1:X2) >> 20 (low 64), via 64-bit SHRD.
  static const uint32_t code[] = {ExtrX(0, 1, 2, 20)};
  state_.cpu.x[1] = 0x1122334455667788ULL;
  state_.cpu.x[2] = 0xAABBCCDDEEFF0011ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x67788AABBCCDDEEFULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Ror64ViaExtr) {
  // ROR X0, X1, #13 is EXTR X0, X1, X1, #13 (Rn == Rm) — 64-bit rotate-right.
  static const uint32_t code[] = {ExtrX(0, 1, 1, 13)};
  state_.cpu.x[1] = 0x123456789ABCDEF0ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xF78091A2B3C4D5E6ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Clz64) {
  static const uint32_t code[] = {ClzX(0, 1)};
  state_.cpu.x[1] = 0x0000000000000100ULL;  // bit 8 set -> 55 leading zeros
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  if (RunOneInstruction(&state_, stop_pc)) {
    EXPECT_EQ(state_.cpu.x[0], uint64_t{55});
  } else {
    // Host without LZCNT: CLZ bails to lite, which is also correct.
    GTEST_SKIP() << "host lacks LZCNT; CLZ bailed";
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Rev16_64) {
  // REV16 reverses bytes within each 16-bit halfword.
  static const uint32_t code[] = {Rev16X(0, 1)};
  state_.cpu.x[1] = 0x1122334455667788ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x2211443366558877ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Rev64) {
  // REV X0, X1: full 64-bit byte reverse.
  static const uint32_t code[] = {RevX(0, 1)};
  state_.cpu.x[1] = 0x1122334455667788ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x8877665544332211ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Rev32X) {
  // REV32 X0, X1: byte-reverse each 32-bit word in place (no cross-word swap).
  static const uint32_t code[] = {Rev32X(0, 1)};
  state_.cpu.x[1] = 0x1122334455667788ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x4433221188776655ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, RevW32ZeroExtends) {
  // REV W0, W1: byte-reverse the 32-bit value; upper 32 bits of X0 cleared.
  static const uint32_t code[] = {RevW(0, 1)};
  state_.cpu.x[1] = 0xFFFFFFFFAABBCCDDULL;  // dirty upper bits must not leak.
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x00000000DDCCBBAAULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Cls64) {
  // CLS X0, X1: count leading sign bits. 0xFFFF...F0000 has 47 leading sign bits.
  static const uint32_t code[] = {ClsX(0, 1)};
  state_.cpu.x[1] = 0xFFFFFFFFFFFF0000ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  if (RunOneInstruction(&state_, stop_pc)) {
    EXPECT_EQ(state_.cpu.x[0], uint64_t{47});
  } else {
    GTEST_SKIP() << "host lacks LZCNT; CLS bailed";
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Cls64AllSameBits) {
  // CLS of all-zero (and all-one) is reg_size-1 = 63.
  static const uint32_t code[] = {ClsX(0, 1)};
  state_.cpu.x[1] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  if (RunOneInstruction(&state_, stop_pc)) {
    EXPECT_EQ(state_.cpu.x[0], uint64_t{63});
  } else {
    GTEST_SKIP() << "host lacks LZCNT; CLS bailed";
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, ClsW32) {
  // CLS W0, W1: 0xFFFF0000 has 15 leading sign bits; W-write zero-extends.
  static const uint32_t code[] = {ClsW(0, 1)};
  state_.cpu.x[1] = 0xDEADBEEFFFFF0000ULL;  // dirty upper bits ignored.
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  if (RunOneInstruction(&state_, stop_pc)) {
    EXPECT_EQ(state_.cpu.x[0], uint64_t{15});
  } else {
    GTEST_SKIP() << "host lacks LZCNT; CLS bailed";
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Rbit64) {
  // RBIT X0, X1: reverse all 64 bits (SWAR bit-swap + byte reverse).
  static const uint32_t code[] = {RbitX(0, 1)};
  state_.cpu.x[1] = 0x1122334455667788ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x11EE66AA22CC4488ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, RbitW32) {
  // RBIT W0, W1: reverse the low 32 bits; upper 32 bits of X0 cleared.
  static const uint32_t code[] = {RbitW(0, 1)};
  state_.cpu.x[1] = 0xDEADBEEF12345678ULL;  // dirty upper bits must not leak.
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x000000001E6A2C48ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfizX) {
  // SBFIZ X0, X1, #8, #8 == SBFM X0, X1, #immr=56, #imms=7 (imms < immr).
  // Low 8 bits 0xAB (bit 7 set -> negative), sign-extended then <<8.
  static const uint32_t code[] = {SbfmX(0, 1, 56, 7)};
  state_.cpu.x[1] = 0xAB;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFAB00ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfizXPositiveField) {
  // SBFIZ X0, X1, #8, #8 with a non-negative field (bit 7 clear).
  static const uint32_t code[] = {SbfmX(0, 1, 56, 7)};
  state_.cpu.x[1] = 0x7F;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x7F00});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SbfizW32) {
  // SBFIZ W0, W1, #4, #4 == SBFM W0, W1, #immr=28, #imms=3 (imms < immr).
  // Low 4 bits 0xF (bit 3 set -> negative), sign-extended then <<4; W-write
  // zero-extends bits 63:32.
  static const uint32_t code[] = {SbfmW(0, 1, 28, 3)};
  state_.cpu.x[1] = 0xF;
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x00000000FFFFFFF0ULL});
}

//
// Multi-instruction region tests: these exercise GenCode's CheckMachineIR on
// real multi-insn IR (the 1-insn tests above do not).
//

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluRegionNoBail) {
  // MOVZ X0,#5; ADD X1,X0,#3; SUB X2,X1,#1; ORR X3,X2,X0.
  static const uint32_t code[] = {
      MovzX(0, 5), AddImmX(1, 0, 3), SubImmX(2, 1, 1), OrrRegX(3, 2, 0)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_TRUE(ok);
  EXPECT_EQ(n, 4u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluThenBailRegion) {
  // MOVZ; ADD; SUB; then a bailing MRS ends the region after 3 translated.
  static const uint32_t code[] = {
      MovzX(0, 0x10), AddImmX(1, 0, 4), SubImmX(2, 1, 2), MrsMidrEl1(3)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 3u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluMixedShiftRegion) {
  // A mix of shifted-register, extended-register, logical-immediate and a
  // multiply, then a bailing MRS. Exercises CheckMachineIR over the whole run.
  static const uint32_t code[] = {MovzX(0, 0x7),
                                  MovzX(1, 0x3),
                                  AddRegX(2, 0, 1),
                                  SubRegLsl(3, 2, 1, 2),
                                  AndImmX(4, 3, 0, 7),
                                  MaddX(5, 0, 1, 4),
                                  MrsMidrEl1(6)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // 6 translate, the MRS bails.
  EXPECT_EQ(n, 6u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluRegionExecutes) {
  // Full region execution end-to-end: MOVZ X0,#5; ADD X1,X0,#3; SUB X2,X1,#1.
  // Expected: X0=5, X1=8, X2=7.
  static const uint32_t code[] = {MovzX(0, 5), AddImmX(1, 0, 3), SubImmX(2, 1, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  MachineCode mc;
  auto [stop, ok, n] =
      HeavyOptimizeRegion(ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = end_pc});
  ASSERT_TRUE(ok);
  ASSERT_EQ(n, 3u);
  ScopedExecRegion exec(&mc);
  TestingRunGeneratedCode(&state_, exec.get(), end_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{5});
  EXPECT_EQ(state_.cpu.x[1], uint64_t{8});
  EXPECT_EQ(state_.cpu.x[2], uint64_t{7});
}

// A flag-setting SUBS inside a multi-insn region must produce valid IR (the
// EmitMaterializeNZCV sequence — PseudoReadFlags + AND + store — must survive
// GenCode's CheckMachineIR). A following ADC bails, ending the region.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiFlagSetterThenBailRegion) {
  // MOVZ X0,#10; SUBS X1,X0,#3; PACGA X0,X1,X0 (bails: this DataProc2Src opcode
  // is not translated by the optimizing frontend). CRC32/ADC are now translated,
  // so PACGA (pointer-auth) is the bail sentinel here.
  static const uint32_t code[] = {MovzX(0, 10), SubsImmX(1, 0, 3), 0x9AC03020u /*PACGA X0,X1,X0*/};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // MOVZ + SUBS translate, the CRC32 bails -> partial region of 2.
  EXPECT_EQ(n, 2u);
}

// Full region execution end-to-end with a flag-setter: MOVZ X0,#7; SUBS X1,X0,#7
// must leave X1==0 and the Z and C flags set in cpu.flags after the region runs.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiFlagSetterRegionExecutes) {
  static const uint32_t code[] = {MovzX(0, 7), SubsImmX(1, 0, 7)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  MachineCode mc;
  auto [stop, ok, n] =
      HeavyOptimizeRegion(ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = end_pc});
  ASSERT_TRUE(ok);
  ASSERT_EQ(n, 2u);
  ScopedExecRegion exec(&mc);
  TestingRunGeneratedCode(&state_, exec.get(), end_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{7});
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0});
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);
}

// A non-MoveWide instruction must bail out of the optimizing frontend (the
// runtime then falls back to the lite translator / interpreter).
TEST_F(Arm64HeavyOptimizerFrontendTest, NonMoveWideBails) {
  // PACGA (DataProc2Src, pointer-auth) is not translated by the optimizing
  // frontend (CRC32 now is, so it is no longer a valid bail sentinel).
  static const uint32_t code[] = {0x9AC03020};  // PACGA X0, X1, X0
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
}

//
// Branch family. Branches need multi-block regions and a real target, so these
// build a region (via HeavyOptimizeRegion) and execute it, asserting where
// cpu.insn_addr lands. The convention used below: each region is a small array
// of 32-bit instructions; `end_pc` is past the last instruction so the region
// extends through the branch; the test asserts insn_addr equals the taken
// target or the fall-through PC.
//

// Build, execute, and return where cpu.insn_addr ended up. Asserts the region
// translated (ok) and translated all `expected_insns` instructions. The region
// runs to wherever its control flow exits (a branch target outside the region,
// or end_pc as a fall-through).
GuestAddr RunRegion(ThreadState* state,
                    const uint32_t* code,
                    GuestAddr end_pc,
                    bool* ok_out = nullptr) {
  state->cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] =
      HeavyOptimizeRegion(ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = end_pc});
  if (ok_out) {
    *ok_out = ok;
  }
  if (!ok) {
    return kNullGuestAddr;
  }
  // Clear any stale process-global TranslationCache entries for this PC window
  // so SetStop(end_pc) installs cleanly (see the note in RunOneInstruction).
  TranslationCache::GetInstance()->InvalidateGuestRange(ToGuestAddr(code), end_pc + 4);
  ScopedExecRegion exec(&mc);
  TestingRunGeneratedCode(state, exec.get(), end_pc);
  return state->cpu.insn_addr;
}

// Unconditional B forward: after a MOVZ, B +8 jumps past code[2]. Because the
// fall-through (code[2]) is unreachable, the region ends at the B and exits to
// the branch target (code+0xC). The MOVZ at [0] runs; nothing past the B does.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchUnconditionalForward) {
  static const uint32_t code[] = {
      MovzX(8, 0x55),  // [0] X8 = 0x55 (runs)
      B(8),            // [1] B -> code+0xC (skips [2])
      MovzX(9, 1),     // [2] unreachable -> not translated
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  GuestAddr branch_target = ToGuestAddr(code) + 4 + 8;  // code[1] addr + 8
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, branch_target);  // region exits to the B target
  EXPECT_EQ(state_.cpu.x[8], uint64_t{0x55});  // [0] ran
  EXPECT_EQ(state_.cpu.x[9], 0u);              // [2] never ran
}

// B.cond taken: SUBS makes X0-5==0 -> Z=1; B.EQ +8 should jump over code[3].
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondEqTaken) {
  static const uint32_t code[] = {
      MovzX(0, 5),      // [0] X0 = 5
      SubsImmX(31, 0, 5),  // [1] CMP X0,#5 -> Z=1,C=1
      Bcond(kCondEQ, 8),   // [2] B.EQ -> code+0x10 (skips [3])
      MovzX(9, 1),         // [3] skipped when taken
      MovzX(10, 2),        // [4] target
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 0u);   // [3] skipped (branch taken)
  EXPECT_EQ(state_.cpu.x[10], 2u);  // [4] executed
}

// B.cond not taken: X0-6 != 0 -> Z=0; B.EQ falls through and runs code[3].
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondEqNotTaken) {
  static const uint32_t code[] = {
      MovzX(0, 5),         // [0]
      SubsImmX(31, 0, 6),  // [1] CMP X0,#6 -> Z=0 (5 != 6)
      Bcond(kCondEQ, 8),   // [2] B.EQ not taken
      MovzX(9, 1),         // [3] runs (fall-through)
      MovzX(10, 2),        // [4]
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 1u);   // [3] ran (not taken)
  EXPECT_EQ(state_.cpu.x[10], 2u);  // [4] ran
}

// Cover NE / LT / GE / HI / LS taken-and-not-taken using a parameterized
// helper. For each condition we set NZCV via a SUBS that produces a known
// relation, then assert the branch is/ isn't taken by whether code[3] ran.
struct CondCase {
  uint8_t cond;
  uint16_t lhs;       // X0 value
  uint16_t rhs_imm;   // CMP immediate
  bool expect_taken;
};

void RunCondCase(ThreadState* state, const CondCase& c) {
  const uint32_t code[] = {
      MovzX(0, c.lhs),
      SubsImmX(31, 0, c.rhs_imm),  // CMP X0, #rhs
      Bcond(c.cond, 8),            // B.cond -> skip [3] when taken
      MovzX(9, 1),                 // [3] runs only on fall-through
      MovzX(10, 2),                // [4]
  };
  // Reset the registers and flags this case touches (ThreadState itself is not
  // copy-assignable because of an atomic member, so zero fields individually).
  state->cpu.x[0] = 0;
  state->cpu.x[9] = 0;
  state->cpu.x[10] = 0;
  state->cpu.flags = 0;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  state->cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] =
      HeavyOptimizeRegion(ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = end_pc});
  ASSERT_TRUE(ok);
  ScopedExecRegion exec(&mc);
  TestingRunGeneratedCode(state, exec.get(), end_pc);
  if (c.expect_taken) {
    EXPECT_EQ(state->cpu.x[9], 0u) << "cond=" << int{c.cond} << " expected taken";
  } else {
    EXPECT_EQ(state->cpu.x[9], 1u) << "cond=" << int{c.cond} << " expected not taken";
  }
  EXPECT_EQ(state->cpu.x[10], 2u);
}

// NE: Z==0. 5 - 6 != 0 -> taken; 5 - 5 == 0 -> not taken.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondNe) {
  RunCondCase(&state_, {kCondNE, 5, 6, /*taken=*/true});
  RunCondCase(&state_, {kCondNE, 5, 5, /*taken=*/false});
}

// LT: N!=V (signed less-than). 3 - 5 < 0 -> taken; 5 - 3 > 0 -> not taken.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondLt) {
  RunCondCase(&state_, {kCondLT, 3, 5, /*taken=*/true});
  RunCondCase(&state_, {kCondLT, 5, 3, /*taken=*/false});
}

// GE: N==V (signed >=). 5 - 3 >= 0 -> taken; 3 - 5 < 0 -> not taken.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondGe) {
  RunCondCase(&state_, {kCondGE, 5, 3, /*taken=*/true});
  RunCondCase(&state_, {kCondGE, 3, 5, /*taken=*/false});
}

// HI: C==1 && Z==0 (unsigned >). 5 - 3: C=1,Z=0 -> taken; 5 - 5: Z=1 -> not.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondHi) {
  RunCondCase(&state_, {kCondHI, 5, 3, /*taken=*/true});
  RunCondCase(&state_, {kCondHI, 5, 5, /*taken=*/false});
}

// LS: C==0 || Z==1 (unsigned <=). 5 - 5: Z=1 -> taken; 5 - 3: C=1,Z=0 -> not.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondLs) {
  RunCondCase(&state_, {kCondLS, 5, 5, /*taken=*/true});
  RunCondCase(&state_, {kCondLS, 5, 3, /*taken=*/false});
}

// GT: Z==0 && N==V. 5 - 3 > 0 -> taken; 5 - 5 == 0 -> not taken.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondGt) {
  RunCondCase(&state_, {kCondGT, 5, 3, /*taken=*/true});
  RunCondCase(&state_, {kCondGT, 5, 5, /*taken=*/false});
}

// LE: Z==1 || N!=V. 5 - 5 == 0 -> taken; 5 - 3 > 0 -> not taken.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondLe) {
  RunCondCase(&state_, {kCondLE, 5, 5, /*taken=*/true});
  RunCondCase(&state_, {kCondLE, 5, 3, /*taken=*/false});
}

// CS/CC (carry). 5 - 3: no borrow -> C=1 (CS taken, CC not).
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondCsCc) {
  RunCondCase(&state_, {kCondCS, 5, 3, /*taken=*/true});
  RunCondCase(&state_, {kCondCS, 3, 5, /*taken=*/false});  // borrow -> C=0
  RunCondCase(&state_, {kCondCC, 3, 5, /*taken=*/true});   // borrow -> C=0
  RunCondCase(&state_, {kCondCC, 5, 3, /*taken=*/false});
}

// MI/PL (negative). 3 - 5 = -2 -> N=1 (MI taken, PL not).
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondMiPl) {
  RunCondCase(&state_, {kCondMI, 3, 5, /*taken=*/true});
  RunCondCase(&state_, {kCondMI, 5, 3, /*taken=*/false});
  RunCondCase(&state_, {kCondPL, 5, 3, /*taken=*/true});
  RunCondCase(&state_, {kCondPL, 3, 5, /*taken=*/false});
}

// VS/VC (overflow). ADDS INT64_MIN + INT64_MIN overflows -> V=1, so VS is taken
// and VC is not. Builds INT64_MIN with MOVZ #0x8000 LSL #48.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondVsTaken) {
  static const uint32_t code[] = {
      MovzHwX(0, 0x8000, 3),  // [0] X0 = INT64_MIN
      AddsRegX(31, 0, 0),     // [1] CMN-like: X0+X0 -> V=1 (flags only, rd=XZR)
      Bcond(kCondVS, 8),      // [2] B.VS -> taken (skips [3])
      MovzX(9, 1),            // [3] skipped when taken
      MovzX(10, 2),           // [4] target
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 0u);   // taken: [3] skipped
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondVcNotTaken) {
  static const uint32_t code[] = {
      MovzHwX(0, 0x8000, 3),  // [0] X0 = INT64_MIN
      AddsRegX(31, 0, 0),     // [1] V=1
      Bcond(kCondVC, 8),      // [2] B.VC -> NOT taken (V==1)
      MovzX(9, 1),            // [3] runs (fall-through)
      MovzX(10, 2),           // [4]
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 1u);   // not taken: [3] runs
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// AL: always taken (lowered to unconditional B). The fall-through is
// unreachable, so the region exits to the B.AL target (code+0xC).
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondAlwaysTaken) {
  static const uint32_t code[] = {
      MovzX(8, 0x77),     // [0] runs
      Bcond(kCondAL, 8),  // [1] B.AL -> code+0xC (skips [2])
      MovzX(9, 1),        // [2] unreachable
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  GuestAddr branch_target = ToGuestAddr(code) + 4 + 8;
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, branch_target);
  EXPECT_EQ(state_.cpu.x[8], uint64_t{0x77});
  EXPECT_EQ(state_.cpu.x[9], 0u);
}

// CBZ taken: X0 == 0 -> branch over code[2].
TEST_F(Arm64HeavyOptimizerFrontendTest, CompareAndBranchCbzTaken) {
  static const uint32_t code[] = {
      MovzX(0, 0),    // [0] X0 = 0
      CbzX(0, 8),     // [1] CBZ X0 -> code+0x10 (skips [2])
      MovzX(9, 1),    // [2] skipped
      MovzX(10, 2),   // [3] target
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 0u);
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// CBZ not taken: X0 != 0 -> fall through, code[2] runs.
TEST_F(Arm64HeavyOptimizerFrontendTest, CompareAndBranchCbzNotTaken) {
  static const uint32_t code[] = {
      MovzX(0, 7),
      CbzX(0, 8),
      MovzX(9, 1),  // [2] runs
      MovzX(10, 2),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 1u);
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// CBNZ taken: X0 != 0 -> branch over code[2].
TEST_F(Arm64HeavyOptimizerFrontendTest, CompareAndBranchCbnzTaken) {
  static const uint32_t code[] = {
      MovzX(0, 9),
      CbnzX(0, 8),
      MovzX(9, 1),  // [2] skipped
      MovzX(10, 2),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 0u);
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// CBZ 32-bit (W form): only the low 32 bits are tested. X0 = 0x1_0000_0000 has
// W0 == 0 -> CBZ W0 must be taken even though X0 != 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, CompareAndBranchCbzWUsesLow32) {
  static const uint32_t code[] = {
      MovzHwX(0, 1, 2),  // [0] X0 = 1 << 32 (W0 == 0)
      CbzW(0, 8),        // [1] CBZ W0 -> taken (skips [2])
      MovzX(9, 1),       // [2] skipped
      MovzX(10, 2),      // [3] target
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 0u);
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// TBZ taken: bit clear -> branch. X0 = 0b100, test bit 0 (clear) -> taken.
TEST_F(Arm64HeavyOptimizerFrontendTest, TestAndBranchTbzTaken) {
  static const uint32_t code[] = {
      MovzX(0, 0x4),   // [0] bit 0 == 0
      TbzX(0, 0, 8),   // [1] TBZ X0,#0 -> taken (skips [2])
      MovzX(9, 1),     // [2] skipped
      MovzX(10, 2),    // [3] target
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 0u);
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// TBZ not taken: bit set -> fall through. X0 = 0b1, test bit 0 (set).
TEST_F(Arm64HeavyOptimizerFrontendTest, TestAndBranchTbzNotTaken) {
  static const uint32_t code[] = {
      MovzX(0, 0x1),
      TbzX(0, 0, 8),
      MovzX(9, 1),  // [2] runs
      MovzX(10, 2),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 1u);
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// TBNZ taken: bit set -> branch. Test bit 1 of 0b10.
TEST_F(Arm64HeavyOptimizerFrontendTest, TestAndBranchTbnzTaken) {
  static const uint32_t code[] = {
      MovzX(0, 0x2),    // bit 1 set
      TbnzX(0, 1, 8),   // TBNZ X0,#1 -> taken (skips [2])
      MovzX(9, 1),
      MovzX(10, 2),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 0u);
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// TBNZ on a high bit (>=32): test bit 40 of X0 = 1<<40 -> taken. Confirms Btq
// covers the full 64-bit register (no bail for bit>=32).
TEST_F(Arm64HeavyOptimizerFrontendTest, TestAndBranchTbnzHighBit) {
  static const uint32_t code[] = {
      MovzHwX(0, 0x100, 2),  // X0 = 0x100 << 32 = 1<<40
      TbnzX(0, 40, 8),       // TBNZ X0,#40 -> taken (skips [2])
      MovzX(9, 1),
      MovzX(10, 2),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[9], 0u);
  EXPECT_EQ(state_.cpu.x[10], 2u);
}

// B.cond whose target is OUTSIDE the region must exit to that guest address
// (the common case). Region is just [SUBS-equal; B.EQ +0x40]; the target
// code+0x44 is past end_pc, so the taken branch exits to it.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondTargetOutsideRegionExits) {
  static const uint32_t code[] = {
      MovzX(0, 5),         // [0]
      SubsImmX(31, 0, 5),  // [1] Z=1
      Bcond(kCondEQ, 0x40),  // [2] B.EQ -> code + 8 + 0x40 = code+0x48 (outside)
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  GuestAddr expected_target = ToGuestAddr(code) + 8 + 0x40;
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, expected_target);
}

// In-region backward branch loop: a countdown. X0 starts at 3; the loop body
// decrements X0 and branches back while X0 != 0, then exits. The accumulator
// X1 counts iterations; after the loop X1 == 3 and X0 == 0.
//   [0] MOVZ X0, #3
//   [1] MOVZ X1, #0       (loop top = code+4)
//   loop: (code+4)
//   [2] ADD  X1, X1, #1
//   [3] SUBS X0, X0, #1   (sets Z when X0 hits 0)
//   [4] B.NE loop (-12 -> back to code+4 == [1]) ... but we want top at [2].
// Put the loop top at [1] so the back-edge from [4] targets [1]; X1 then counts
// the SUBS iterations: X0 3->2->1->0 gives X1 incremented each pass.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchCondBackwardLoop) {
  // loop top is code[1]. [4] B.NE back to code[1].
  // Iterations: enter with X0=3.
  //   pass1: X1=0; ADD X1=1; SUBS X0=2 (Z=0) -> branch back
  //   ... but X1 is re-zeroed each pass if [1] is in the loop. Keep [1] OUTSIDE
  //   the loop by targeting code[2] instead.
  static const uint32_t code[] = {
      MovzX(0, 3),         // [0] X0 = 3
      MovzX(1, 0),         // [1] X1 = 0
      AddImmX(1, 1, 1),    // [2] loop top (code+8): X1++
      SubsImmX(0, 0, 1),   // [3] X0-- (sets Z when reaches 0)
      Bcond(kCondNE, -8),  // [4] B.NE -> code+8 ([2]) while X0 != 0
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);          // loop exits by falling through [4]
  EXPECT_EQ(state_.cpu.x[0], 0u);     // counted down to zero
  EXPECT_EQ(state_.cpu.x[1], 3u);     // body ran 3 times
}

// BR (indirect): exit to the address held in a register. X5 holds an arbitrary
// guest address; BR X5 must land cpu.insn_addr there.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchRegisterBr) {
  static const uint32_t code[] = {
      MovzHwX(5, 0xBEEF, 0),  // [0] X5 = 0xBEEF (a sentinel target address)
      BrX(5),                 // [1] BR X5 -> exit indirect to 0xBEEF
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, GuestAddr{0xBEEF});
}

// RET (indirect via X30 by default, here RET X3): same indirect-exit path.
TEST_F(Arm64HeavyOptimizerFrontendTest, BranchRegisterRet) {
  static const uint32_t code[] = {
      MovzHwX(3, 0x1234, 0),  // [0] X3 = 0x1234
      RetX(3),                // [1] RET X3 -> exit indirect to 0x1234
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, GuestAddr{0x1234});
}

//
// Integer loads / stores. The base register points at a static buffer; the
// optimizing frontend's Load/Store apply TBI then emit the size/sign-appropriate
// host memory access and a recovery block for faults.
//

// LDR Xt, [Xn]: 64-bit load.
// SIMD&FP load/store, unsigned-offset immediate (imm scaled by access size).
// LDR/STR Qt = 128-bit, Dt = 64-bit, St = 32-bit.
constexpr uint32_t LdrQuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0x3DC00000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t StrQuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0x3D800000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t LdrDuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0xFD400000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
constexpr uint32_t LdrSuoff(uint8_t rt, uint8_t rn, uint16_t imm) {
  return 0xBD400000 | (static_cast<uint32_t>(imm) << 10) | (rn << 5) | rt;
}
// LDP/STP Qt1, Qt2, [Xn, #imm] (imm scaled by 16, signed imm7).
constexpr uint32_t LdpQ(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm) {
  return 0xAD400000 | ((static_cast<uint32_t>(imm) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
constexpr uint32_t StpQ(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm) {
  return 0xAD000000 | ((static_cast<uint32_t>(imm) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt1;
}

// LDP/STP Dt1, Dt2, [Xn, #imm] (imm7 scaled by 8) and St1, St2 (imm7 scaled by
// 4). imm is the raw signed 7-bit imm7 field (like LdpQ/StpQ above).
constexpr uint32_t LdpD(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm) {
  return 0x6D400000 | ((static_cast<uint32_t>(imm) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
constexpr uint32_t StpD(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm) {
  return 0x6D000000 | ((static_cast<uint32_t>(imm) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
constexpr uint32_t LdpS(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm) {
  return 0x2D400000 | ((static_cast<uint32_t>(imm) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
constexpr uint32_t StpS(uint8_t rt1, uint8_t rt2, uint8_t rn, int8_t imm) {
  return 0x2D000000 | ((static_cast<uint32_t>(imm) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt1;
}
// Ground truth: clang --target=aarch64-linux-gnu + llvm-objdump.
static_assert(StpD(0, 1, 2, 0) == 0x6d000440u);     // stp d0, d1, [x2]
static_assert(LdpD(2, 3, 2, 0) == 0x6d400c42u);     // ldp d2, d3, [x2]
static_assert(StpD(8, 9, 31, 0) == 0x6d0027e8u);    // stp d8, d9, [sp]
static_assert(LdpD(14, 15, 31, 0) == 0x6d403feeu);  // ldp d14, d15, [sp]
static_assert(StpD(10, 11, 2, 2) == 0x6d012c4au);   // stp d10, d11, [x2, #16]
static_assert(LdpD(12, 13, 2, -2) == 0x6d7f344cu);  // ldp d12, d13, [x2, #-16]
static_assert(StpS(0, 1, 2, 0) == 0x2d000440u);     // stp s0, s1, [x2]
static_assert(LdpS(2, 3, 2, 0) == 0x2d400c42u);     // ldp s2, s3, [x2]
static_assert(StpS(4, 5, 2, 2) == 0x2d011444u);     // stp s4, s5, [x2, #8]
static_assert(LdpS(6, 7, 2, -2) == 0x2d7f1c46u);    // ldp s6, s7, [x2, #-8] (imm7=-2, S-scale 4)

TEST_F(Arm64HeavyOptimizerFrontendTest, LdrQ128) {
  alignas(16) static const uint64_t buf[2] = {0x1122334455667788ULL, 0x99AABBCCDDEEFF00ULL};
  static const uint32_t code[] = {LdrQuoff(0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);  // poison
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], uint64_t{0x1122334455667788ULL});
  EXPECT_EQ(r[1], uint64_t{0x99AABBCCDDEEFF00ULL});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, StrQ128) {
  alignas(16) static uint64_t buf[2] = {0, 0};
  static const uint32_t code[] = {StrQuoff(0, 1, 0)};
  const uint64_t v[2] = {0xCAFEF00DDEADBEEFULL, 0x0123456789ABCDEFULL};
  std::memcpy(&state_.cpu.v[0], v, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], v[0]);
  EXPECT_EQ(buf[1], v[1]);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, LdrD64ZeroesUpper) {
  alignas(16) static const uint64_t buf[2] = {0x1122334455667788ULL, 0xdeadbeefdeadbeefULL};
  static const uint32_t code[] = {LdrDuoff(0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);  // poison upper 64
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], uint64_t{0x1122334455667788ULL});
  EXPECT_EQ(r[1], uint64_t{0});  // LDR D zero-extends to 128
}

TEST_F(Arm64HeavyOptimizerFrontendTest, LdrS32ZeroesUpper) {
  alignas(16) static const uint32_t buf[4] = {0xAABBCCDDu, 0x11111111u, 0x22222222u, 0x33333333u};
  static const uint32_t code[] = {LdrSuoff(0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], uint32_t{0xAABBCCDDu});
  EXPECT_EQ(r[1], uint32_t{0});
  EXPECT_EQ(r[2], uint32_t{0});
  EXPECT_EQ(r[3], uint32_t{0});
}

TEST_F(Arm64HeavyOptimizerFrontendTest, StpLdpQ128) {
  alignas(16) static uint64_t buf[4] = {0, 0, 0, 0};
  const uint64_t v0[2] = {0x1111111122222222ULL, 0x3333333344444444ULL};
  const uint64_t v1[2] = {0x5555555566666666ULL, 0x7777777788888888ULL};
  std::memcpy(&state_.cpu.v[0], v0, 16);
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t scode[] = {StpQ(0, 1, 2, 0)};
  state_.cpu.x[2] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(scode);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(scode) + sizeof(scode)));
  EXPECT_EQ(buf[0], v0[0]);
  EXPECT_EQ(buf[1], v0[1]);
  EXPECT_EQ(buf[2], v1[0]);
  EXPECT_EQ(buf[3], v1[1]);
  // LDP back into v2/v3 and verify round-trip.
  std::memset(&state_.cpu.v[2], 0xAB, 16);
  std::memset(&state_.cpu.v[3], 0xAB, 16);
  static const uint32_t lcode[] = {LdpQ(2, 3, 2, 0)};
  state_.cpu.insn_addr = ToGuestAddr(lcode);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(lcode) + sizeof(lcode)));
  uint64_t r2[2], r3[2];
  std::memcpy(r2, &state_.cpu.v[2], 16);
  std::memcpy(r3, &state_.cpu.v[3], 16);
  EXPECT_EQ(r2[0], v0[0]);
  EXPECT_EQ(r2[1], v0[1]);
  EXPECT_EQ(r3[0], v1[0]);
  EXPECT_EQ(r3[1], v1[1]);
}

// STP/LDP of a 64-bit D-pair — the callee-saved FP save/restore in nearly every
// function prologue/epilogue. STP writes two 8-byte slots; LDP reads them back
// and must zero-extend each register's upper 64 bits.
TEST_F(Arm64HeavyOptimizerFrontendTest, StpLdpD64) {
  alignas(16) static uint64_t buf[2] = {0, 0};
  const uint64_t d0 = 0x1122334455667788ULL;
  const uint64_t d1 = 0x99AABBCCDDEEFF00ULL;
  std::memset(&state_.cpu.v[0], 0xAB, 16);  // upper 64 of v0 must be ignored
  std::memset(&state_.cpu.v[1], 0xCD, 16);
  std::memcpy(&state_.cpu.v[0], &d0, 8);
  std::memcpy(&state_.cpu.v[1], &d1, 8);
  static const uint32_t scode[] = {StpD(0, 1, 2, 0)};
  state_.cpu.x[2] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(scode);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(scode) + sizeof(scode)));
  EXPECT_EQ(buf[0], d0);
  EXPECT_EQ(buf[1], d1);
  // LDP back into v2/v3, poisoned upper halves must be cleared.
  std::memset(&state_.cpu.v[2], 0xAB, 16);
  std::memset(&state_.cpu.v[3], 0xAB, 16);
  static const uint32_t lcode[] = {LdpD(2, 3, 2, 0)};
  state_.cpu.insn_addr = ToGuestAddr(lcode);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(lcode) + sizeof(lcode)));
  uint64_t r2[2], r3[2];
  std::memcpy(r2, &state_.cpu.v[2], 16);
  std::memcpy(r3, &state_.cpu.v[3], 16);
  EXPECT_EQ(r2[0], d0);
  EXPECT_EQ(r2[1], uint64_t{0});  // LDP D zero-extends to 128
  EXPECT_EQ(r3[0], d1);
  EXPECT_EQ(r3[1], uint64_t{0});
}

// STP/LDP of a 32-bit S-pair. LDP S zero-extends each register's upper 96 bits.
TEST_F(Arm64HeavyOptimizerFrontendTest, StpLdpS32) {
  alignas(16) static uint32_t buf[2] = {0, 0};
  const uint32_t s0 = 0xAABBCCDDu;
  const uint32_t s1 = 0x11223344u;
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  std::memset(&state_.cpu.v[1], 0xCD, 16);
  std::memcpy(&state_.cpu.v[0], &s0, 4);
  std::memcpy(&state_.cpu.v[1], &s1, 4);
  static const uint32_t scode[] = {StpS(0, 1, 2, 0)};
  state_.cpu.x[2] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(scode);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(scode) + sizeof(scode)));
  EXPECT_EQ(buf[0], s0);
  EXPECT_EQ(buf[1], s1);
  std::memset(&state_.cpu.v[2], 0xAB, 16);
  std::memset(&state_.cpu.v[3], 0xAB, 16);
  static const uint32_t lcode[] = {LdpS(2, 3, 2, 0)};
  state_.cpu.insn_addr = ToGuestAddr(lcode);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(lcode) + sizeof(lcode)));
  uint32_t r2[4], r3[4];
  std::memcpy(r2, &state_.cpu.v[2], 16);
  std::memcpy(r3, &state_.cpu.v[3], 16);
  EXPECT_EQ(r2[0], s0);
  EXPECT_EQ(r2[1], uint32_t{0});
  EXPECT_EQ(r2[2], uint32_t{0});
  EXPECT_EQ(r2[3], uint32_t{0});
  EXPECT_EQ(r3[0], s1);
  EXPECT_EQ(r3[1], uint32_t{0});
  EXPECT_EQ(r3[2], uint32_t{0});
  EXPECT_EQ(r3[3], uint32_t{0});
}

// AdvSIMD load/store multiple structures (LD1/ST1 contiguous form):
//   0 Q 0011 00 P L 0 Rm opcode size Rn Rt
// opcode encodes the register count (LD1/ST1): 0111=1, 1010=2, 0110=3, 0010=4.
constexpr uint32_t AdvSimdMultiEnc(bool q, bool postindex, bool is_load, uint8_t rm,
                                   uint8_t opcode, uint8_t size, uint8_t rn, uint8_t rt) {
  return 0x0C000000u | (static_cast<uint32_t>(q) << 30) |
         (static_cast<uint32_t>(postindex) << 23) | (static_cast<uint32_t>(is_load) << 22) |
         (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(opcode) << 12) |
         (static_cast<uint32_t>(size) << 10) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint8_t kLd1Op1 = 0b0111;
constexpr uint8_t kLd1Op2 = 0b1010;
constexpr uint8_t kLd1Op3 = 0b0110;
constexpr uint8_t kLd1Op4 = 0b0010;
// Ground truth from the LLVM assembler (llvm-objdump of clang-assembled LD1/ST1):
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd1Op1, 0, 1, 0) == 0x4c407020u);
static_assert(AdvSimdMultiEnc(false, false, false, 0, kLd1Op1, 0, 1, 0) == 0x0c007020u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd1Op2, 0, 1, 0) == 0x4c40a020u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd1Op3, 0, 1, 0) == 0x4c406020u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd1Op4, 0, 1, 0) == 0x4c402020u);
static_assert(AdvSimdMultiEnc(true, true, true, 0x1F, kLd1Op1, 0, 1, 0) == 0x4cdf7020u);
static_assert(AdvSimdMultiEnc(true, true, true, 2, kLd1Op2, 0, 1, 0) == 0x4cc2a020u);
static_assert(AdvSimdMultiEnc(false, true, false, 0x1F, kLd1Op2, 0, 1, 0) == 0x0c9fa020u);

// LD1 {v0.16b}, [x1]: bulk 16-byte load into v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1Single16b) {
  alignas(16) static const uint64_t buf[2] = {0x1122334455667788ULL, 0x99AABBCCDDEEFF00ULL};
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd1Op1, 0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], uint64_t{0x1122334455667788ULL});
  EXPECT_EQ(r[1], uint64_t{0x99AABBCCDDEEFF00ULL});
}

// ST1 {v0.16b}, [x1]: bulk 16-byte store from v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, St1Single16b) {
  alignas(16) static uint64_t buf[2] = {0, 0};
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, false, 0, kLd1Op1, 0, 1, 0)};
  const uint64_t v[2] = {0xCAFEF00DDEADBEEFULL, 0x0123456789ABCDEFULL};
  std::memcpy(&state_.cpu.v[0], v, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], v[0]);
  EXPECT_EQ(buf[1], v[1]);
}

// LD1 {v0.8b}, [x1] (Q=0): 8-byte load zero-extends the upper 64 of v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1Single8bZeroesUpper) {
  alignas(16) static const uint64_t buf[2] = {0x1122334455667788ULL, 0xdeadbeefdeadbeefULL};
  static const uint32_t code[] = {AdvSimdMultiEnc(false, false, true, 0, kLd1Op1, 0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], uint64_t{0x1122334455667788ULL});
  EXPECT_EQ(r[1], uint64_t{0});
}

// LD1 {v0.16b, v1.16b}, [x1]: two consecutive 16-byte vectors, contiguous.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1Two16b) {
  alignas(16) static const uint64_t buf[4] = {
      0x1111111122222222ULL, 0x3333333344444444ULL, 0x5555555566666666ULL, 0x7777777788888888ULL};
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd1Op2, 0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  std::memset(&state_.cpu.v[1], 0xCD, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r0[2], r1[2];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  EXPECT_EQ(r0[0], buf[0]);
  EXPECT_EQ(r0[1], buf[1]);
  EXPECT_EQ(r1[0], buf[2]);
  EXPECT_EQ(r1[1], buf[3]);
}

// LD1 {v0.16b - v3.16b}, [x1]: four consecutive 16-byte vectors.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1Four16b) {
  alignas(16) static const uint64_t buf[8] = {0x0000000000000001ULL, 0x0000000000000002ULL,
                                              0x0000000000000003ULL, 0x0000000000000004ULL,
                                              0x0000000000000005ULL, 0x0000000000000006ULL,
                                              0x0000000000000007ULL, 0x0000000000000008ULL};
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd1Op4, 0, 1, 0)};
  for (int i = 0; i < 4; i++) std::memset(&state_.cpu.v[i], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  for (int i = 0; i < 4; i++) {
    uint64_t r[2];
    std::memcpy(r, &state_.cpu.v[i], 16);
    EXPECT_EQ(r[0], buf[i * 2 + 0]);
    EXPECT_EQ(r[1], buf[i * 2 + 1]);
  }
}

// ST1 {v0.16b, v1.16b}, [x1]: two contiguous 16-byte stores.
TEST_F(Arm64HeavyOptimizerFrontendTest, St1Two16b) {
  alignas(16) static uint64_t buf[4] = {0, 0, 0, 0};
  const uint64_t v0[2] = {0x1111111122222222ULL, 0x3333333344444444ULL};
  const uint64_t v1[2] = {0x5555555566666666ULL, 0x7777777788888888ULL};
  std::memcpy(&state_.cpu.v[0], v0, 16);
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, false, 0, kLd1Op2, 0, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], v0[0]);
  EXPECT_EQ(buf[1], v0[1]);
  EXPECT_EQ(buf[2], v1[0]);
  EXPECT_EQ(buf[3], v1[1]);
}

// LD1 {v0.16b}, [x1], #16: immediate post-index advances x1 by 1*16.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1PostIndexImm) {
  alignas(16) static const uint64_t buf[2] = {0xAAAAAAAABBBBBBBBULL, 0xCCCCCCCCDDDDDDDDULL};
  static const uint32_t code[] = {AdvSimdMultiEnc(true, true, true, 0x1F, kLd1Op1, 0, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], buf[0]);
  EXPECT_EQ(r[1], buf[1]);
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(&buf[0]) + 16);
}

// LD1 {v0.16b, v1.16b}, [x1], x2: register post-index advances x1 by x2.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1PostIndexReg) {
  alignas(16) static const uint64_t buf[4] = {
      0x1111111122222222ULL, 0x3333333344444444ULL, 0x5555555566666666ULL, 0x7777777788888888ULL};
  static const uint32_t code[] = {AdvSimdMultiEnc(true, true, true, 2, kLd1Op2, 0, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 32;  // arbitrary stride register
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r0[2], r1[2];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  EXPECT_EQ(r0[0], buf[0]);
  EXPECT_EQ(r1[1], buf[3]);
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(&buf[0]) + 32);
}

// ST1 {v0.8b, v1.8b}, [x1], #16 (Q=0): stores the low 64 of each reg, x1 += 2*8.
TEST_F(Arm64HeavyOptimizerFrontendTest, St1Q0Two8bPostIndexImm) {
  alignas(16) static uint64_t buf[2] = {0, 0};
  const uint64_t v0[2] = {0x1122334455667788ULL, 0xdeadbeefdeadbeefULL};
  const uint64_t v1[2] = {0x99AABBCCDDEEFF00ULL, 0xdeadbeefdeadbeefULL};
  std::memcpy(&state_.cpu.v[0], v0, 16);
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t code[] = {AdvSimdMultiEnc(false, true, false, 0x1F, kLd1Op2, 0, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], v0[0]);  // low 64 of v0
  EXPECT_EQ(buf[1], v1[0]);  // low 64 of v1
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(&buf[0]) + 16);
}

// LD1 {v31.16b, v0.16b}, [x1]: register list wraps (rt+r)&31 from v31 to v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1TwoWrapV31) {
  alignas(16) static const uint64_t buf[4] = {
      0x1111111122222222ULL, 0x3333333344444444ULL, 0x5555555566666666ULL, 0x7777777788888888ULL};
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd1Op2, 0, 1, 31)};
  std::memset(&state_.cpu.v[31], 0xAB, 16);
  std::memset(&state_.cpu.v[0], 0xCD, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r31[2], r0[2];
  std::memcpy(r31, &state_.cpu.v[31], 16);
  std::memcpy(r0, &state_.cpu.v[0], 16);
  EXPECT_EQ(r31[0], buf[0]);
  EXPECT_EQ(r31[1], buf[1]);
  EXPECT_EQ(r0[0], buf[2]);
  EXPECT_EQ(r0[1], buf[3]);
}

// AdvSIMD load/store multiple structures, INTERLEAVED forms (LD2/LD3/LD4 and
// ST2/ST3/ST4). Same encoding as AdvSimdMultiEnc; the opcode field selects the
// de-/interleaving variant: 1000=2 regs, 0100=3 regs, 0000=4 regs. Element e in
// memory belongs to register (e % num_regs), lane (e / num_regs).
constexpr uint8_t kLd2Op = 0b1000;
constexpr uint8_t kLd3Op = 0b0100;
constexpr uint8_t kLd4Op = 0b0000;
// Ground truth from the LLVM assembler (llvm-objdump of clang-assembled LDn/STn):
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b00, 1, 0) == 0x4c408020u);
static_assert(AdvSimdMultiEnc(true, false, false, 0, kLd2Op, 0b00, 1, 0) == 0x4c008020u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b01, 1, 0) == 0x4c408420u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b10, 1, 0) == 0x4c408820u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b11, 1, 0) == 0x4c408c20u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd3Op, 0b10, 1, 0) == 0x4c404820u);
static_assert(AdvSimdMultiEnc(true, false, false, 0, kLd3Op, 0b10, 1, 0) == 0x4c004820u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd4Op, 0b10, 1, 0) == 0x4c400820u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd4Op, 0b00, 1, 0) == 0x4c400020u);
static_assert(AdvSimdMultiEnc(true, false, false, 0, kLd4Op, 0b00, 1, 0) == 0x4c000020u);
static_assert(AdvSimdMultiEnc(false, false, true, 0, kLd2Op, 0b00, 1, 0) == 0x0c408020u);
static_assert(AdvSimdMultiEnc(true, true, true, 0x1F, kLd2Op, 0b00, 1, 0) == 0x4cdf8020u);
static_assert(AdvSimdMultiEnc(false, true, true, 2, kLd3Op, 0b00, 1, 0) == 0x0cc24020u);
static_assert(AdvSimdMultiEnc(false, true, false, 0x1F, kLd4Op, 0b00, 1, 0) == 0x0c9f0020u);
static_assert(AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b00, 1, 31) == 0x4c40803fu);

// LD2 {v0.16b, v1.16b}, [x1]: de-interleave 32 bytes → v0=even, v1=odd bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld2_16b) {
  alignas(16) static uint8_t buf[32];
  for (int i = 0; i < 32; i++) buf[i] = static_cast<uint8_t>(i * 7 + 1);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b00, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  std::memset(&state_.cpu.v[1], 0xCD, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint8_t r0[16], r1[16];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  for (int l = 0; l < 16; l++) {
    EXPECT_EQ(r0[l], buf[l * 2 + 0]);
    EXPECT_EQ(r1[l], buf[l * 2 + 1]);
  }
}

// ST2 {v0.16b, v1.16b}, [x1]: interleave v0/v1 bytes back into 32 bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, St2_16b) {
  alignas(16) static uint8_t buf[32];
  std::memset(buf, 0, sizeof(buf));
  uint8_t v0[16], v1[16];
  for (int i = 0; i < 16; i++) {
    v0[i] = static_cast<uint8_t>(i + 1);
    v1[i] = static_cast<uint8_t>(i + 0x40);
  }
  std::memcpy(&state_.cpu.v[0], v0, 16);
  std::memcpy(&state_.cpu.v[1], v1, 16);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, false, 0, kLd2Op, 0b00, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  for (int l = 0; l < 16; l++) {
    EXPECT_EQ(buf[l * 2 + 0], v0[l]);
    EXPECT_EQ(buf[l * 2 + 1], v1[l]);
  }
}

// LD2 {v0.4s, v1.4s}, [x1]: word (esize=4) de-interleave, 8 words.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld2_4s) {
  alignas(16) static uint32_t buf[8];
  for (int i = 0; i < 8; i++) buf[i] = 0x11110000u * static_cast<uint32_t>(i + 1) + i;
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b10, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint32_t r0[4], r1[4];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  for (int l = 0; l < 4; l++) {
    EXPECT_EQ(r0[l], buf[l * 2 + 0]);
    EXPECT_EQ(r1[l], buf[l * 2 + 1]);
  }
}

// LD2 {v0.2d, v1.2d}, [x1]: dword (esize=8) de-interleave, 4 dwords.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld2_2d) {
  alignas(16) static uint64_t buf[4] = {0x1111111111111111ULL, 0x2222222222222222ULL,
                                        0x3333333333333333ULL, 0x4444444444444444ULL};
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b11, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r0[2], r1[2];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  EXPECT_EQ(r0[0], buf[0]);
  EXPECT_EQ(r0[1], buf[2]);
  EXPECT_EQ(r1[0], buf[1]);
  EXPECT_EQ(r1[1], buf[3]);
}

// LD3 {v0.4s, v1.4s, v2.4s}, [x1]: 3-way word de-interleave, 12 words.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld3_4s) {
  alignas(16) static uint32_t buf[12];
  for (int i = 0; i < 12; i++) buf[i] = 0xA0000000u + static_cast<uint32_t>(i);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd3Op, 0b10, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint32_t r0[4], r1[4], r2[4];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  std::memcpy(r2, &state_.cpu.v[2], 16);
  for (int l = 0; l < 4; l++) {
    EXPECT_EQ(r0[l], buf[l * 3 + 0]);
    EXPECT_EQ(r1[l], buf[l * 3 + 1]);
    EXPECT_EQ(r2[l], buf[l * 3 + 2]);
  }
}

// ST3 {v0.4s, v1.4s, v2.4s}, [x1]: 3-way word interleave.
TEST_F(Arm64HeavyOptimizerFrontendTest, St3_4s) {
  alignas(16) static uint32_t buf[12];
  std::memset(buf, 0, sizeof(buf));
  uint32_t v0[4], v1[4], v2[4];
  for (int i = 0; i < 4; i++) {
    v0[i] = 0x100u + i;
    v1[i] = 0x200u + i;
    v2[i] = 0x300u + i;
  }
  std::memcpy(&state_.cpu.v[0], v0, 16);
  std::memcpy(&state_.cpu.v[1], v1, 16);
  std::memcpy(&state_.cpu.v[2], v2, 16);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, false, 0, kLd3Op, 0b10, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  for (int l = 0; l < 4; l++) {
    EXPECT_EQ(buf[l * 3 + 0], v0[l]);
    EXPECT_EQ(buf[l * 3 + 1], v1[l]);
    EXPECT_EQ(buf[l * 3 + 2], v2[l]);
  }
}

// LD4 {v0.16b - v3.16b}, [x1]: 4-way byte de-interleave, 64 bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld4_16b) {
  alignas(16) static uint8_t buf[64];
  for (int i = 0; i < 64; i++) buf[i] = static_cast<uint8_t>(i * 3 + 5);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd4Op, 0b00, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[4][16];
  for (int k = 0; k < 4; k++) std::memcpy(r[k], &state_.cpu.v[k], 16);
  for (int l = 0; l < 16; l++) {
    for (int k = 0; k < 4; k++) EXPECT_EQ(r[k][l], buf[l * 4 + k]);
  }
}

// ST4 {v0.16b - v3.16b}, [x1]: 4-way byte interleave.
TEST_F(Arm64HeavyOptimizerFrontendTest, St4_16b) {
  alignas(16) static uint8_t buf[64];
  std::memset(buf, 0, sizeof(buf));
  uint8_t v[4][16];
  for (int k = 0; k < 4; k++)
    for (int i = 0; i < 16; i++) v[k][i] = static_cast<uint8_t>(k * 0x40 + i + 1);
  for (int k = 0; k < 4; k++) std::memcpy(&state_.cpu.v[k], v[k], 16);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, false, 0, kLd4Op, 0b00, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  for (int l = 0; l < 16; l++) {
    for (int k = 0; k < 4; k++) EXPECT_EQ(buf[l * 4 + k], v[k][l]);
  }
}

// LD2 {v0.8b, v1.8b}, [x1] (Q=0): de-interleave 16 bytes; upper 64 of each reg
// must be zeroed (D-register semantics).
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld2_8b_Q0ZeroesUpper) {
  alignas(16) static uint8_t buf[16];
  for (int i = 0; i < 16; i++) buf[i] = static_cast<uint8_t>(0x80 + i);
  static const uint32_t code[] = {AdvSimdMultiEnc(false, false, true, 0, kLd2Op, 0b00, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  std::memset(&state_.cpu.v[1], 0xCD, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint8_t r0[16], r1[16];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  for (int l = 0; l < 8; l++) {
    EXPECT_EQ(r0[l], buf[l * 2 + 0]);
    EXPECT_EQ(r1[l], buf[l * 2 + 1]);
  }
  for (int l = 8; l < 16; l++) {
    EXPECT_EQ(r0[l], 0);
    EXPECT_EQ(r1[l], 0);
  }
}

// LD2 {v0.8h, v1.8h}, [x1]: halfword (esize=2) de-interleave, 16 halfwords.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld2_8h) {
  alignas(16) static uint16_t buf[16];
  for (int i = 0; i < 16; i++) buf[i] = static_cast<uint16_t>(0x1000 + i * 0x111);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b01, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint16_t r0[8], r1[8];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  for (int l = 0; l < 8; l++) {
    EXPECT_EQ(r0[l], buf[l * 2 + 0]);
    EXPECT_EQ(r1[l], buf[l * 2 + 1]);
  }
}

// LD2 {v0.16b, v1.16b}, [x1], #32: immediate post-index advances x1 by 2*16.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld2_16b_PostIndexImm) {
  alignas(16) static uint8_t buf[32];
  for (int i = 0; i < 32; i++) buf[i] = static_cast<uint8_t>(i);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, true, true, 0x1F, kLd2Op, 0b00, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint8_t r0[16], r1[16];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  for (int l = 0; l < 16; l++) {
    EXPECT_EQ(r0[l], buf[l * 2 + 0]);
    EXPECT_EQ(r1[l], buf[l * 2 + 1]);
  }
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(&buf[0]) + 32);
}

// LD3 {v0.8b, v1.8b, v2.8b}, [x1], x2 (Q=0): register post-index advances by x2.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld3_8b_PostIndexReg) {
  alignas(16) static uint8_t buf[24];
  for (int i = 0; i < 24; i++) buf[i] = static_cast<uint8_t>(i + 3);
  static const uint32_t code[] = {AdvSimdMultiEnc(false, true, true, 2, kLd3Op, 0b00, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 24;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint8_t r0[16], r1[16], r2[16];
  std::memcpy(r0, &state_.cpu.v[0], 16);
  std::memcpy(r1, &state_.cpu.v[1], 16);
  std::memcpy(r2, &state_.cpu.v[2], 16);
  for (int l = 0; l < 8; l++) {
    EXPECT_EQ(r0[l], buf[l * 3 + 0]);
    EXPECT_EQ(r1[l], buf[l * 3 + 1]);
    EXPECT_EQ(r2[l], buf[l * 3 + 2]);
  }
  EXPECT_EQ(r0[8], 0);  // Q=0 upper zeroed
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(&buf[0]) + 24);
}

// ST4 {v0.8b - v3.8b}, [x1], #32 (Q=0): interleave low-8 of 4 regs, x1 += 4*8.
TEST_F(Arm64HeavyOptimizerFrontendTest, St4_8b_PostIndexImm) {
  alignas(16) static uint8_t buf[32];
  std::memset(buf, 0, sizeof(buf));
  uint8_t v[4][8];
  for (int k = 0; k < 4; k++)
    for (int i = 0; i < 8; i++) v[k][i] = static_cast<uint8_t>(k * 0x20 + i + 1);
  for (int k = 0; k < 4; k++) {
    std::memset(&state_.cpu.v[k], 0xEE, 16);
    std::memcpy(&state_.cpu.v[k], v[k], 8);
  }
  static const uint32_t code[] = {AdvSimdMultiEnc(false, true, false, 0x1F, kLd4Op, 0b00, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  for (int l = 0; l < 8; l++) {
    for (int k = 0; k < 4; k++) EXPECT_EQ(buf[l * 4 + k], v[k][l]);
  }
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(&buf[0]) + 32);
}

// LD2 {v31.16b, v0.16b}, [x1]: register list wraps (rt+r)&31 from v31 to v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld2WrapV31) {
  alignas(16) static uint8_t buf[32];
  for (int i = 0; i < 32; i++) buf[i] = static_cast<uint8_t>(i * 5 + 2);
  static const uint32_t code[] = {AdvSimdMultiEnc(true, false, true, 0, kLd2Op, 0b00, 1, 31)};
  std::memset(&state_.cpu.v[31], 0xAB, 16);
  std::memset(&state_.cpu.v[0], 0xCD, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint8_t r31[16], r0[16];
  std::memcpy(r31, &state_.cpu.v[31], 16);
  std::memcpy(r0, &state_.cpu.v[0], 16);
  for (int l = 0; l < 16; l++) {
    EXPECT_EQ(r31[l], buf[l * 2 + 0]);
    EXPECT_EQ(r0[l], buf[l * 2 + 1]);
  }
}

// AdvSIMD load/store SINGLE structure: LD1R (replicate), single-element LD1 /
// ST1 (one lane), num_regs == 1. Encoding: 0 Q 001101 P L R Rm opcode S size
// Rn Rt. opcode / S / size / Q pack the element size and lane index per the ARM
// ARM; the encoder passes these fields raw and the static_asserts pin them to
// LLVM-assembler ground truth (clang-assembled + llvm-objdump'd).
constexpr uint32_t AdvSimdSingleEnc(bool q, bool postindex, bool is_load, bool r,
                                    uint8_t rm, uint8_t opcode, bool s_bit, uint8_t size,
                                    uint8_t rn, uint8_t rt) {
  return 0x0D000000u | (static_cast<uint32_t>(q) << 30) |
         (static_cast<uint32_t>(postindex) << 23) | (static_cast<uint32_t>(is_load) << 22) |
         (static_cast<uint32_t>(r) << 21) | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(opcode) << 13) | (static_cast<uint32_t>(s_bit) << 12) |
         (static_cast<uint32_t>(size) << 10) | (static_cast<uint32_t>(rn) << 5) | rt;
}
// Ground truth (llvm-objdump of clang-assembled LD1R/LD1/ST1 single-structure):
//   ld1r  {v3.4s},  [x1]        4d40c823
//   ld1r  {v3.4s},  [x10], #4   4ddfc943
//   ld1r  {v0.8b},  [x1]        0d40c020
//   ld1r  {v0.2d},  [x1]        4d40cc20
//   ld1r  {v5.8h},  [x1], x3    4dc3c425
//   ld1   {v0.s}[2],[x1]        4d408020
//   ld1   {v0.b}[5],[x1]        0d401420
//   ld1   {v31.s}[0],[x1]       0d40803f
//   ld1   {v0.h}[3],[x1], #2    0ddf5820
//   st1   {v0.s}[3],[x1]        4d009020
//   st1   {v0.d}[1],[x1]        4d008420
//   st1   {v3.s}[1],[x10], x2   0d829143
//   st1   {v0.b}[10],[x1]       4d000820
static_assert(AdvSimdSingleEnc(true, false, true, false, 0, 0b110, false, 0b10, 1, 3) ==
              0x4d40c823u);
static_assert(AdvSimdSingleEnc(true, true, true, false, 31, 0b110, false, 0b10, 10, 3) ==
              0x4ddfc943u);
static_assert(AdvSimdSingleEnc(false, false, true, false, 0, 0b110, false, 0b00, 1, 0) ==
              0x0d40c020u);
static_assert(AdvSimdSingleEnc(true, false, true, false, 0, 0b110, false, 0b11, 1, 0) ==
              0x4d40cc20u);
static_assert(AdvSimdSingleEnc(true, true, true, false, 3, 0b110, false, 0b01, 1, 5) ==
              0x4dc3c425u);
static_assert(AdvSimdSingleEnc(true, false, true, false, 0, 0b100, false, 0b00, 1, 0) ==
              0x4d408020u);
static_assert(AdvSimdSingleEnc(false, false, true, false, 0, 0b000, true, 0b01, 1, 0) ==
              0x0d401420u);
static_assert(AdvSimdSingleEnc(false, false, true, false, 0, 0b100, false, 0b00, 1, 31) ==
              0x0d40803fu);
static_assert(AdvSimdSingleEnc(false, true, true, false, 31, 0b010, true, 0b10, 1, 0) ==
              0x0ddf5820u);
static_assert(AdvSimdSingleEnc(true, false, false, false, 0, 0b100, true, 0b00, 1, 0) ==
              0x4d009020u);
static_assert(AdvSimdSingleEnc(true, false, false, false, 0, 0b100, false, 0b01, 1, 0) ==
              0x4d008420u);
static_assert(AdvSimdSingleEnc(false, true, false, false, 2, 0b100, true, 0b00, 10, 3) ==
              0x0d829143u);
static_assert(AdvSimdSingleEnc(true, false, false, false, 0, 0b000, false, 0b10, 1, 0) ==
              0x4d000820u);

// LD1R {v3.4s}, [x1]: broadcast the 4-byte element at [x1] to all 4 S lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1r_4s) {
  alignas(16) static const uint32_t buf[1] = {0xDEADBEEFu};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(true, false, true, false, 0, 0b110, false, 0b10, 1, 3)};
  std::memset(&state_.cpu.v[3], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[3], 16);
  for (int l = 0; l < 4; l++) EXPECT_EQ(r[l], 0xDEADBEEFu);
}

// LD1R {v0.8b}, [x1] (Q=0): broadcast a byte to all 8 low lanes, upper 64 = 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1r_8b_Q0ZeroesUpper) {
  static const uint8_t buf[1] = {0x5A};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(false, false, true, false, 0, 0b110, false, 0b00, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int l = 0; l < 8; l++) EXPECT_EQ(r[l], 0x5A);
  for (int l = 8; l < 16; l++) EXPECT_EQ(r[l], 0);
}

// LD1R {v0.2d}, [x1]: broadcast an 8-byte element to both D lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1r_2d) {
  alignas(16) static const uint64_t buf[1] = {0x0123456789ABCDEFULL};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(true, false, true, false, 0, 0b110, false, 0b11, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], 0x0123456789ABCDEFULL);
  EXPECT_EQ(r[1], 0x0123456789ABCDEFULL);
}

// LD1R {v3.4s}, [x10], #4: replicate then immediate post-index advances x10 by
// esize (4). This is exactly the calculate_gnu_hash_neon tail.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1r_4s_PostIndexImm) {
  alignas(16) static const uint32_t buf[1] = {0x11223344u};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(true, true, true, false, 31, 0b110, false, 0b10, 10, 3)};
  std::memset(&state_.cpu.v[3], 0xAB, 16);
  const GuestAddr base = ToGuestAddr(&buf[0]);
  state_.cpu.x[10] = base;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[3], 16);
  for (int l = 0; l < 4; l++) EXPECT_EQ(r[l], 0x11223344u);
  EXPECT_EQ(state_.cpu.x[10], base + 4);
}

// LD1R {v5.8h}, [x1], x3: replicate a halfword to 8 lanes, register post-index.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1r_8h_PostIndexReg) {
  alignas(16) static const uint16_t buf[1] = {0xBEEF};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(true, true, true, false, 3, 0b110, false, 0b01, 1, 5)};
  std::memset(&state_.cpu.v[5], 0xAB, 16);
  const GuestAddr base = ToGuestAddr(&buf[0]);
  state_.cpu.x[1] = base;
  state_.cpu.x[3] = 64;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[5], 16);
  for (int l = 0; l < 8; l++) EXPECT_EQ(r[l], 0xBEEF);
  EXPECT_EQ(state_.cpu.x[1], base + 64);
}

// LD1 {v0.s}[2], [x1]: load one S element into lane 2, other lanes preserved.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1_Single_s_Lane2) {
  alignas(16) static const uint32_t buf[1] = {0xCAFEF00Du};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(true, false, true, false, 0, 0b100, false, 0b00, 1, 0)};
  const uint32_t init[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  std::memcpy(&state_.cpu.v[0], init, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], init[0]);
  EXPECT_EQ(r[1], init[1]);
  EXPECT_EQ(r[2], 0xCAFEF00Du);
  EXPECT_EQ(r[3], init[3]);
}

// LD1 {v0.b}[5], [x1]: load one byte into lane 5, other lanes preserved.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1_Single_b_Lane5) {
  static const uint8_t buf[1] = {0x7E};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(false, false, true, false, 0, 0b000, true, 0b01, 1, 0)};
  uint8_t init[16];
  for (int i = 0; i < 16; i++) init[i] = static_cast<uint8_t>(0x10 + i);
  std::memcpy(&state_.cpu.v[0], init, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int l = 0; l < 16; l++) {
    EXPECT_EQ(r[l], l == 5 ? uint8_t{0x7E} : init[l]);
  }
}

// LD1 {v31.s}[0], [x1]: high register (v31), lane 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1_Single_s_V31) {
  alignas(16) static const uint32_t buf[1] = {0xABCD1234u};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(false, false, true, false, 0, 0b100, false, 0b00, 1, 31)};
  const uint32_t init[4] = {0x55555555u, 0x66666666u, 0x77777777u, 0x88888888u};
  std::memcpy(&state_.cpu.v[31], init, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[31], 16);
  EXPECT_EQ(r[0], 0xABCD1234u);
  EXPECT_EQ(r[1], init[1]);
  EXPECT_EQ(r[2], init[2]);
  EXPECT_EQ(r[3], init[3]);
}

// LD1 {v0.h}[3], [x1], #2: load one halfword into lane 3, imm post-index by 2.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ld1_Single_h_Lane3_PostIndexImm) {
  alignas(16) static const uint16_t buf[1] = {0x9ABC};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(false, true, true, false, 31, 0b010, true, 0b10, 1, 0)};
  uint16_t init[8];
  for (int i = 0; i < 8; i++) init[i] = static_cast<uint16_t>(0x100 + i);
  std::memcpy(&state_.cpu.v[0], init, 16);
  const GuestAddr base = ToGuestAddr(&buf[0]);
  state_.cpu.x[1] = base;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int l = 0; l < 8; l++) {
    EXPECT_EQ(r[l], l == 3 ? uint16_t{0x9ABC} : init[l]);
  }
  EXPECT_EQ(state_.cpu.x[1], base + 2);
}

// ST1 {v0.s}[3], [x1]: store lane 3 (one S element) to memory.
TEST_F(Arm64HeavyOptimizerFrontendTest, St1_Single_s_Lane3) {
  alignas(16) static uint32_t buf[1] = {0};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(true, false, false, false, 0, 0b100, true, 0b00, 1, 0)};
  const uint32_t v[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0xC0FFEE00u};
  std::memcpy(&state_.cpu.v[0], v, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], 0xC0FFEE00u);
}

// ST1 {v0.d}[1], [x1]: store the high D lane.
TEST_F(Arm64HeavyOptimizerFrontendTest, St1_Single_d_Lane1) {
  alignas(16) static uint64_t buf[1] = {0};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(true, false, false, false, 0, 0b100, false, 0b01, 1, 0)};
  const uint64_t v[2] = {0x1111222233334444ULL, 0xDEADBEEFCAFEF00DULL};
  std::memcpy(&state_.cpu.v[0], v, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], 0xDEADBEEFCAFEF00DULL);
}

// ST1 {v3.s}[1], [x10], x2: store lane 1, register post-index by x2.
TEST_F(Arm64HeavyOptimizerFrontendTest, St1_Single_s_Lane1_PostIndexReg) {
  alignas(16) static uint32_t buf[1] = {0};
  static const uint32_t code[] = {
      AdvSimdSingleEnc(false, true, false, false, 2, 0b100, true, 0b00, 10, 3)};
  const uint32_t v[4] = {0xAAAAAAAAu, 0xBADDCAFEu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[3], v, 16);
  const GuestAddr base = ToGuestAddr(&buf[0]);
  state_.cpu.x[10] = base;
  state_.cpu.x[2] = 128;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], 0xBADDCAFEu);
  EXPECT_EQ(state_.cpu.x[10], base + 128);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, LdrX64) {
  static uint64_t buf[2] = {0x1122334455667788ULL, 0};
  static const uint32_t code[] = {LdrXuoff(0, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1122334455667788ULL});
}

// STR Xt, [Xn]: 64-bit store.
TEST_F(Arm64HeavyOptimizerFrontendTest, StrX64) {
  static uint64_t buf[1] = {0};
  static const uint32_t code[] = {StrXuoff(0, 1, 0)};
  state_.cpu.x[0] = 0xCAFEF00DDEADBEEFULL;
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(buf[0], uint64_t{0xCAFEF00DDEADBEEFULL});
}

// LDR Wt, [Xn]: 32-bit load zero-extends to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrW32ZeroExtends) {
  static uint64_t buf[1] = {0xFFFFFFFFAABBCCDDULL};
  static const uint32_t code[] = {LdrWuoff(0, 1, 0)};
  state_.cpu.x[0] = 0x1111111111111111ULL;  // preset upper bits must be cleared
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xAABBCCDDULL});  // upper 32 cleared
}

// STR Wt, [Xn]: 32-bit store writes only the low 4 bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, StrW32) {
  static uint64_t buf[1] = {0xEEEEEEEEEEEEEEEEULL};
  static const uint32_t code[] = {StrWuoff(0, 1, 0)};
  state_.cpu.x[0] = 0x99999999AABBCCDDULL;  // W0 = 0xAABBCCDD
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(buf[0], uint64_t{0xEEEEEEEEAABBCCDDULL});  // only low 4 bytes changed
}

// LDRB Wt, [Xn]: byte load zero-extends to 32 (upper bits cleared).
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrbZeroExtends) {
  static uint8_t buf[1] = {0xFE};
  static const uint32_t code[] = {LdrbWuoff(0, 1, 0)};
  state_.cpu.x[0] = 0x1234567890ABCDEFULL;  // upper bits must be cleared
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFE});
}

// STRB Wt, [Xn]: byte store writes only the low byte.
TEST_F(Arm64HeavyOptimizerFrontendTest, Strb) {
  static uint64_t buf[1] = {0xAABBCCDDEEFF1122ULL};
  static const uint32_t code[] = {StrbWuoff(0, 1, 0)};
  state_.cpu.x[0] = 0x55;
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(buf[0], uint64_t{0xAABBCCDDEEFF1155ULL});  // only low byte changed
}

// LDRSB Xt, [Xn]: signed byte load sign-extends to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrsbSignExtends) {
  static uint8_t buf[1] = {0x80};  // -128
  static const uint32_t code[] = {LdrsbXuoff(0, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFF80ULL});  // sign-extended
}

// LDRH Wt, [Xn]: halfword load zero-extends to 32.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrhZeroExtends) {
  static uint16_t buf[1] = {0xBEEF};
  static const uint32_t code[] = {LdrhWuoff(0, 1, 0)};
  state_.cpu.x[0] = 0x1234567890ABCDEFULL;  // upper bits must be cleared
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xBEEF});
}

// STRH Wt, [Xn]: halfword store writes only the low 2 bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, Strh) {
  static uint64_t buf[1] = {0xAABBCCDDEEFF1122ULL};
  static const uint32_t code[] = {StrhWuoff(0, 1, 0)};
  state_.cpu.x[0] = 0x55AA;
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(buf[0], uint64_t{0xAABBCCDDEEFF55AAULL});  // only low 2 bytes changed
}

// LDRSH Xt, [Xn]: signed halfword load sign-extends to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrshSignExtends) {
  static uint16_t buf[1] = {0x8000};  // INT16_MIN
  static const uint32_t code[] = {LdrshXuoff(0, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFF8000ULL});  // sign-extended
}

// LDRSW Xt, [Xn]: signed 32-bit load sign-extends to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrswSignExtends) {
  static uint32_t buf[1] = {0x80000000U};  // INT32_MIN
  static const uint32_t code[] = {LdrswXuoff(0, 1, 0)};
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFF80000000ULL});  // sign-extended
}

// LDR Xt, [Xn, #imm]: base + scaled immediate offset (imm scaled by 8 -> byte 8).
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrXImmOffset) {
  static uint64_t buf[2] = {0xDEAD0000DEAD0000ULL, 0x0102030405060708ULL};
  static const uint32_t code[] = {LdrXuoff(0, 1, 1)};  // [X1 + 8]
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0102030405060708ULL});
}

// LDR (literal): LDR Xt, label. The address is PC-relative and constant; the
// literal value sits in the instruction stream after the load.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrLiteral64) {
  // [0] LDR X0, #8  (offset to [2..3] = literal); [1] B over the literal;
  // [2..3] the 64-bit literal value.
  alignas(8) static const uint32_t code[] = {
      LdrLiteralX(0, 8),  // [0] load from code+8
      B(12),              // [1] branch past the literal to stop_pc
      0x55667788u,        // [2] literal low word
      0x11223344u,        // [3] literal high word
  };
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, stop_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, stop_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1122334455667788ULL});
}

// LDRSW (literal): signed 32-bit PC-relative load sign-extends to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrswLiteralSignExtends) {
  alignas(8) static const uint32_t code[] = {
      LdrswLiteral(0, 8),  // [0] load from code+8
      B(8),                // [1] branch past the literal to stop_pc
      0x80000000u,         // [2] literal (INT32_MIN)
  };
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, stop_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, stop_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFF80000000ULL});  // sign-extended
}

// A multi-instruction region exercising GenCode's CheckMachineIR: LDR; ADD; then
// a bail (RBIT). The load + add translate; the bit-reverse bails.
TEST_F(Arm64HeavyOptimizerFrontendTest, LoadAddThenBailRegion) {
  static uint64_t buf[1] = {0x1000};
  static const uint32_t code[] = {
      LdrXuoff(0, 1, 0),   // [0] X0 = [X1]
      AddImmX(2, 0, 0x24),  // [1] X2 = X0 + 0x24
      MrsMidrEl1(3),       // [2] MRS MIDR_EL1 bails
  };
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // LDR + ADD translate; MRS bails -> partial region of 2 instructions.
  EXPECT_EQ(n, 2u);
}

//
// Conditional select (CSEL / CSINC / CSINV / CSNEG). Each region seeds NZCV with
// a CMP (SUBS to XZR), then runs the conditional select and asserts X3.
//

// CSEL EQ, condition met: X0==5, CMP X0,#5 -> Z=1, CSEL selects X1 (true case).
TEST_F(Arm64HeavyOptimizerFrontendTest, CselEqTaken) {
  static const uint32_t code[] = {
      MovzX(1, 100),       // [0] X1 = 100 (true case)
      MovzX(2, 200),       // [1] X2 = 200 (false case)
      MovzX(0, 5),         // [2] X0 = 5
      SubsImmX(31, 0, 5),  // [3] CMP X0,#5 -> Z=1
      CselX(3, 1, 2, kCondEQ),  // [4] X3 = EQ ? X1 : X2
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[3], uint64_t{100});
}

// CSEL EQ, condition NOT met: X0==5, CMP X0,#6 -> Z=0, CSEL selects X2 (false).
TEST_F(Arm64HeavyOptimizerFrontendTest, CselEqNotTaken) {
  static const uint32_t code[] = {
      MovzX(1, 100),
      MovzX(2, 200),
      MovzX(0, 5),
      SubsImmX(31, 0, 6),  // Z=0 (5 != 6)
      CselX(3, 1, 2, kCondEQ),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[3], uint64_t{200});
}

// CSEL LT, condition met: 3 - 5 = -2 -> N=1,V=0 -> LT (N!=V) holds -> X1.
TEST_F(Arm64HeavyOptimizerFrontendTest, CselLtTaken) {
  static const uint32_t code[] = {
      MovzX(1, 100),
      MovzX(2, 200),
      MovzX(0, 3),
      SubsImmX(31, 0, 5),  // 3 - 5 -> N=1, V=0 -> LT holds
      CselX(3, 1, 2, kCondLT),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[3], uint64_t{100});
}

// CSINC NE, condition NOT met (Z=1): selects src2 + 1. X2=200 -> 201.
TEST_F(Arm64HeavyOptimizerFrontendTest, CsincNotTakenIncrements) {
  static const uint32_t code[] = {
      MovzX(1, 100),
      MovzX(2, 200),
      MovzX(0, 5),
      SubsImmX(31, 0, 5),       // Z=1 -> NE not met
      CsincX(3, 1, 2, kCondNE),  // X3 = NE ? X1 : X2 + 1
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[3], uint64_t{201});  // false case incremented
}

// CSINC NE, condition met (Z=0): selects src1 unmodified. X1=100.
TEST_F(Arm64HeavyOptimizerFrontendTest, CsincTakenSelectsSrc1) {
  static const uint32_t code[] = {
      MovzX(1, 100),
      MovzX(2, 200),
      MovzX(0, 5),
      SubsImmX(31, 0, 6),       // Z=0 -> NE met
      CsincX(3, 1, 2, kCondNE),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[3], uint64_t{100});  // true case unmodified
}

// CSINV NE, condition NOT met (Z=1): selects ~src2. ~200 = 0xFF..FF37.
TEST_F(Arm64HeavyOptimizerFrontendTest, CsinvNotTakenInverts) {
  static const uint32_t code[] = {
      MovzX(1, 100),
      MovzX(2, 200),
      MovzX(0, 5),
      SubsImmX(31, 0, 5),        // Z=1 -> NE not met
      CsinvX(3, 1, 2, kCondNE),  // X3 = NE ? X1 : ~X2
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[3], ~uint64_t{200});
}

// CSNEG NE, condition NOT met (Z=1): selects -src2. -200.
TEST_F(Arm64HeavyOptimizerFrontendTest, CsnegNotTakenNegates) {
  static const uint32_t code[] = {
      MovzX(1, 100),
      MovzX(2, 200),
      MovzX(0, 5),
      SubsImmX(31, 0, 5),        // Z=1 -> NE not met
      CsnegX(3, 1, 2, kCondNE),  // X3 = NE ? X1 : -X2
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[3], static_cast<uint64_t>(-int64_t{200}));
}

// CSEL AL: always selects src1 regardless of flags.
TEST_F(Arm64HeavyOptimizerFrontendTest, CselAlwaysSelectsSrc1) {
  static const uint32_t code[] = {
      MovzX(1, 100),
      MovzX(2, 200),
      CselX(3, 1, 2, kCondAL),
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[3], uint64_t{100});
}

//
// Conditional compare (CCMP / CCMN). Asserts the merged cpu.flags for both the
// condition-met (real compare) and condition-not-met (nzcv immediate) paths.
//

// CCMP condition MET: CMP X0,#5 sets Z=1 (EQ holds) -> CCMP does CMP X0,X1.
// X0==X1==5 -> the real compare sets Z=1, C=1, N=0, V=0.
TEST_F(Arm64HeavyOptimizerFrontendTest, CcmpConditionMetRealCompare) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      MovzX(1, 5),
      SubsImmX(31, 0, 5),  // CMP X0,#5 -> Z=1 (EQ condition will hold)
      CcmpRegX(0, 1, /*nzcv=*/0x0, kCondEQ),  // EQ met -> CMP X0,X1 (5==5)
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);    // 5 - 5 == 0
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);   // no borrow
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);
}

// CCMP condition NOT met: NE fails after Z=1, so NZCV = the nzcv immediate.
// nzcv = 0x6 = 0b0110 -> Z=1 (bit2), C=1 (bit1), N=0, V=0.
TEST_F(Arm64HeavyOptimizerFrontendTest, CcmpConditionNotMetUsesImmediate) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      MovzX(1, 9),
      SubsImmX(31, 0, 5),  // Z=1
      CcmpRegX(0, 1, /*nzcv=*/0x6, kCondNE),  // NE NOT met -> NZCV = 0x6
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);     // bit2 of nzcv
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);    // bit1 of nzcv
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagNegative);  // bit3 clear
  EXPECT_FALSE(state_.cpu.flags & CPUState::kFlagOverflow);  // bit0 clear
}

// CCMN condition met: CMN adds rn+rm. X0=INT64_MIN, X1=INT64_MIN -> overflow.
TEST_F(Arm64HeavyOptimizerFrontendTest, CcmnConditionMetRealCompare) {
  static const uint32_t code[] = {
      MovzHwX(0, 0x8000, 3),  // X0 = INT64_MIN
      MovzHwX(1, 0x8000, 3),  // X1 = INT64_MIN
      MovzX(2, 0),
      SubsImmX(31, 2, 0),  // CMP X2,#0 -> Z=1 (EQ holds)
      CcmnRegX(0, 1, /*nzcv=*/0x0, kCondEQ),  // EQ met -> CMN X0,X1 (overflow)
  };
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagOverflow);   // signed overflow
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagCarry);      // unsigned carry out
  EXPECT_TRUE(state_.cpu.flags & CPUState::kFlagZero);       // sum is 0
}

//
// Register-offset loads / stores. Base points at a static buffer; the offset
// register is extended (UXTW/SXTW/LSL) and shifted before being added.
//

// LDR Xt, [Xn, Xm, LSL #3]: 64-bit element indexing (scale 8).
TEST_F(Arm64HeavyOptimizerFrontendTest, LoadRegLsl3) {
  static uint64_t buf[4] = {0x1111111111111111ULL, 0x2222222222222222ULL,
                            0x3333333333333333ULL, 0x4444444444444444ULL};
  static const uint32_t code[] = {LdrXregLsl3(0, 1, 2)};  // X0 = [X1 + X2*8]
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 2;  // index 2
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x3333333333333333ULL});
}

// LDR Xt, [Xn, Wm, UXTW #3]: the W index is zero-extended (upper 32 ignored).
TEST_F(Arm64HeavyOptimizerFrontendTest, LoadRegUxtw3IgnoresUpper) {
  static uint64_t buf[4] = {0xA0, 0xA1, 0xA2, 0xA3};
  static const uint32_t code[] = {LdrXregUxtw3(0, 1, 2)};  // X0 = [X1 + UXTW(W2)*8]
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 0xFFFFFFFF00000001ULL;  // W2 == 1; upper 32 must be ignored
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xA1});  // index 1
}

// LDR Xt, [Xn, Wm, SXTW #3]: the W index is sign-extended (negative offset).
TEST_F(Arm64HeavyOptimizerFrontendTest, LoadRegSxtw3Negative) {
  static uint64_t buf[4] = {0xB0, 0xB1, 0xB2, 0xB3};
  static const uint32_t code[] = {LdrXregSxtw3(0, 1, 2)};  // X0 = [X1 + SXTW(W2)*8]
  state_.cpu.x[1] = ToGuestAddr(&buf[2]);     // base at index 2
  state_.cpu.x[2] = 0xFFFFFFFFFFFFFFFFULL;     // W2 == -1 -> sign-extends to -1
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xB1});  // index 2 + (-1) = index 1
}

// LDR Wt, [Xn, Wm, UXTW #0]: 32-bit zero-extending load, no shift.
TEST_F(Arm64HeavyOptimizerFrontendTest, LoadRegW32Uxtw0) {
  static uint32_t buf[4] = {0xC0, 0xC1, 0xC2, 0xC3};
  static const uint32_t code[] = {LdrWregUxtw0(0, 1, 2)};  // W0 = [X1 + UXTW(W2)]
  state_.cpu.x[0] = 0x1111111111111111ULL;  // upper bits must be cleared
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 8;  // byte offset 8 -> buf[2]
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xC2});  // upper 32 cleared
}

// STR Xt, [Xn, Xm, LSL #3]: register-offset store.
TEST_F(Arm64HeavyOptimizerFrontendTest, StoreRegLsl3) {
  static uint64_t buf[4] = {0, 0, 0, 0};
  static const uint32_t code[] = {StrXregLsl3(0, 1, 2)};  // [X1 + X2*8] = X0
  state_.cpu.x[0] = 0xCAFEF00DDEADBEEFULL;
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 3;  // index 3
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  ASSERT_TRUE(RunOneInstruction(&state_, stop_pc));
  EXPECT_EQ(buf[3], uint64_t{0xCAFEF00DDEADBEEFULL});
  EXPECT_EQ(buf[0], uint64_t{0});  // other slots untouched
}

//
// Load/store pair (LDP / STP).
//

// STP then LDP round-trip through a static buffer. STP X0,X1,[X2] writes two
// 64-bit slots; LDP X3,X4,[X2] reads them back.
TEST_F(Arm64HeavyOptimizerFrontendTest, StpLdpRoundTrip) {
  alignas(16) static uint64_t buf[2] = {0, 0};
  static const uint32_t code[] = {
      StpX(0, 1, 2, 0),  // [0] STP X0, X1, [X2]
      LdpX(3, 4, 2, 0),  // [1] LDP X3, X4, [X2]
  };
  state_.cpu.x[0] = 0x1122334455667788ULL;
  state_.cpu.x[1] = 0x99AABBCCDDEEFF00ULL;
  state_.cpu.x[2] = ToGuestAddr(&buf[0]);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(buf[0], uint64_t{0x1122334455667788ULL});
  EXPECT_EQ(buf[1], uint64_t{0x99AABBCCDDEEFF00ULL});
  EXPECT_EQ(state_.cpu.x[3], uint64_t{0x1122334455667788ULL});
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0x99AABBCCDDEEFF00ULL});
}

// LDP X0, X8, [X0]: the base aliases the first destination. Both halves must be
// loaded from the ORIGINAL base, so X8 = obj[1] (not *(obj[0] + 8)). Mirrors the
// lite-translator LdpBaseAliasesFirstDest regression.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdpBaseAliasesFirstDest) {
  alignas(16) static uint64_t obj[2] = {
      0xAAAABBBBCCCCDDDDULL,  // [0] becomes X0 after LDP
      0x1122334455667788ULL,  // [8] must become X8 — NOT *(obj[0]+8)
  };
  static const uint32_t code[] = {LdpX(0, 8, 0, 0)};  // LDP X0, X8, [X0]
  state_.cpu.x[0] = ToGuestAddr(&obj[0]);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  GuestAddr landed = RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(landed, end_pc);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xAAAABBBBCCCCDDDDULL});
  EXPECT_EQ(state_.cpu.x[8], uint64_t{0x1122334455667788ULL});
}

//
// Scalar floating-point. The optimizing tier lowers FADD/FSUB/FMUL/FDIV (S and
// D) through the intrinsic layer and FMOV(reg)/FABS/FNEG/FMOV-imm directly; the
// rest bail. Each region writes the source V regs via memcpy of float/double,
// runs through the JIT (ASSERT the region didn't bail), then checks the low
// 4/8 result bytes AND that the upper bytes of V[d] were zeroed.
//

// FP data-processing (2 source): 0001_1110_ftype_1_Rm_opcode_10_Rn_Rd.
// opcode[15:12]: FMUL=0000, FDIV=0001, FADD=0010, FSUB=0011. ftype +0x400000=D.
constexpr uint32_t FpDP2(uint32_t base, uint8_t rd, uint8_t rn, uint8_t rm) {
  return base | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmulS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E200800, rd, rn, rm); }
constexpr uint32_t FmulD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E600800, rd, rn, rm); }
constexpr uint32_t FdivS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E201800, rd, rn, rm); }
constexpr uint32_t FdivD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E601800, rd, rn, rm); }
constexpr uint32_t FaddS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E202800, rd, rn, rm); }
constexpr uint32_t FaddD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E602800, rd, rn, rm); }
constexpr uint32_t FsubS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E203800, rd, rn, rm); }
constexpr uint32_t FsubD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E603800, rd, rn, rm); }
// FMAX/FMIN/FMAXNM/FMINNM (opcode 0100/0101/0110/0111).
constexpr uint32_t FmaxS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E204800, rd, rn, rm); }
constexpr uint32_t FmaxD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E604800, rd, rn, rm); }
constexpr uint32_t FminS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E205800, rd, rn, rm); }
constexpr uint32_t FminD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E605800, rd, rn, rm); }
constexpr uint32_t FmaxnmS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E206800, rd, rn, rm); }
constexpr uint32_t FmaxnmD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E606800, rd, rn, rm); }
constexpr uint32_t FminnmS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E207800, rd, rn, rm); }
constexpr uint32_t FminnmD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E607800, rd, rn, rm); }
// FNMUL Sd/Dd (opcode=1000): -(n * m).
constexpr uint32_t FnmulS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E208800, rd, rn, rm); }
constexpr uint32_t FnmulD(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E608800, rd, rn, rm); }

// FP data-processing (1 source): 0001_1110_ftype_1_opcode[5:0]_10000_Rn_Rd.
// opcode[20:15]: FMOV=000000, FABS=000001, FNEG=000010, FSQRT=000011.
constexpr uint32_t FpDP1(uint32_t base, uint8_t rd, uint8_t rn) {
  return base | (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FmovRegS(uint8_t rd, uint8_t rn) { return FpDP1(0x1E204000, rd, rn); }
constexpr uint32_t FmovRegD(uint8_t rd, uint8_t rn) { return FpDP1(0x1E604000, rd, rn); }
constexpr uint32_t FabsS(uint8_t rd, uint8_t rn) { return FpDP1(0x1E20C000, rd, rn); }
constexpr uint32_t FabsD(uint8_t rd, uint8_t rn) { return FpDP1(0x1E60C000, rd, rn); }
constexpr uint32_t FnegS(uint8_t rd, uint8_t rn) { return FpDP1(0x1E214000, rd, rn); }
constexpr uint32_t FnegD(uint8_t rd, uint8_t rn) { return FpDP1(0x1E614000, rd, rn); }
// FSQRT Sd,Sn / Dd,Dn (opcode=000011) — lowered via SQRTSS/SQRTSD.
constexpr uint32_t FsqrtS(uint8_t rd, uint8_t rn) { return FpDP1(0x1E21C000, rd, rn); }
constexpr uint32_t FsqrtD(uint8_t rd, uint8_t rn) { return FpDP1(0x1E61C000, rd, rn); }
static_assert(FsqrtS(0, 1) == 0x1E21C020u);  // fsqrt s0, s1
static_assert(FsqrtD(0, 1) == 0x1E61C020u);  // fsqrt d0, d1
// FRINTA Sd,Sn / Dd,Dn (opcode=001100, round to nearest, ties away from zero).
constexpr uint32_t FrintaS(uint8_t rd, uint8_t rn) { return FpDP1(0x1E264000, rd, rn); }
constexpr uint32_t FrintaD(uint8_t rd, uint8_t rn) { return FpDP1(0x1E664000, rd, rn); }
static_assert(FrintaS(0, 1) == 0x1E264020u);  // frinta s0, s1
static_assert(FrintaD(0, 1) == 0x1E664020u);  // frinta d0, d1

// FMOV (scalar, immediate): 0001_1110_ftype_1_imm8_100_00000_Rd.
constexpr uint32_t FmovImmS(uint8_t rd, uint8_t imm8) {
  return 0x1E201000 | (static_cast<uint32_t>(imm8) << 13) | rd;
}
constexpr uint32_t FmovImmD(uint8_t rd, uint8_t imm8) {
  return 0x1E601000 | (static_cast<uint32_t>(imm8) << 13) | rd;
}

// FCSEL Sd|Dd, Sn, Sm, cond:  0001_1110_ftype_1_Rm_cond_11_Rn_Rd.
// Base: 0x1E200C00 (S) / 0x1E600C00 (D).
constexpr uint32_t FcselScalar(uint32_t base, uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return base | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(cond) << 12) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FcselS(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return FcselScalar(0x1E200C00, rd, rn, rm, cond);
}
constexpr uint32_t FcselD(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
  return FcselScalar(0x1E600C00, rd, rn, rm, cond);
}
static_assert(FcselS(0, 1, 2, 0) == 0x1E220C20u);   // fcsel s0, s1, s2, eq
static_assert(FcselD(3, 4, 5, 12) == 0x1E65CC83u);  // fcsel d3, d4, d5, gt

// --- AdvSIMD three-same INTEGER encoders. ---
// Standard three-same encoding (bit21=1):
//   0 Q U 01110 size(2) 1 Rm(5) opcode(5) 1 Rn(5) Rd(5)
// Base = bits[28:24]=01110 | bit21 | bit10 = 0x0E200400.
constexpr uint32_t AdvSimdThreeSame(
    bool q, bool u, uint8_t size, uint8_t opcode, uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E200400u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(u) << 29) |
         (static_cast<uint32_t>(size) << 22) | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(opcode) << 11) | (static_cast<uint32_t>(rn) << 5) | rd;
}
// ADD (vector): U=0, opcode=10000.
constexpr uint32_t AddVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b10000, rd, rn, rm);
}
// SUB (vector): U=1, opcode=10000.
constexpr uint32_t SubVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b10000, rd, rn, rm);
}
// MUL (vector): U=0, opcode=10011.
constexpr uint32_t MulVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b10011, rd, rn, rm);
}
// PMUL (vector, polynomial): U=1, opcode=10011, size=00.
constexpr uint32_t PmulVec(bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, /*size=*/0b00, /*opcode=*/0b10011, rd, rn, rm);
}
static_assert(PmulVec(/*q=*/true, 0, 1, 2) == 0x6e229c20u);   // pmul v0.16b,v1.16b,v2.16b
static_assert(PmulVec(/*q=*/false, 0, 1, 2) == 0x2e229c20u);  // pmul v0.8b,v1.8b,v2.8b
// MLA (vector): U=0, opcode=10010.
constexpr uint32_t MlaVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b10010, rd, rn, rm);
}
// MLS (vector): U=1, opcode=10010.
constexpr uint32_t MlsVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b10010, rd, rn, rm);
}
// CMEQ (vector, register): U=1, opcode=10001.
constexpr uint32_t CmeqVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b10001, rd, rn, rm);
}
// Saturating add/sub (vector): add opcode=00001, sub opcode=00101; U selects
// signed (SQADD/SQSUB) vs unsigned (UQADD/UQSUB).
constexpr uint32_t SqaddVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b00001, rd, rn, rm);
}
constexpr uint32_t UqaddVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b00001, rd, rn, rm);
}
constexpr uint32_t SqsubVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b00101, rd, rn, rm);
}
constexpr uint32_t UqsubVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b00101, rd, rn, rm);
}
// Logic group (opcode=00011); op selected by U and size:
//   AND: U=0, size=00.   ORR: U=0, size=10.   EOR: U=1, size=00.
constexpr uint32_t AndVec(bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, /*size=*/0b00, /*opcode=*/0b00011, rd, rn, rm);
}
constexpr uint32_t OrrVec(bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, /*size=*/0b10, /*opcode=*/0b00011, rd, rn, rm);
}
constexpr uint32_t EorVec(bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, /*size=*/0b00, /*opcode=*/0b00011, rd, rn, rm);
}
// Bitwise-select group (opcode=00011, U=1); op selected by size:
//   BSL: size=01.   BIT: size=10.   BIF: size=11.
constexpr uint32_t BslVec(bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, /*size=*/0b01, /*opcode=*/0b00011, rd, rn, rm);
}
constexpr uint32_t BitVec(bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, /*size=*/0b10, /*opcode=*/0b00011, rd, rn, rm);
}
constexpr uint32_t BifVec(bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, /*size=*/0b11, /*opcode=*/0b00011, rd, rn, rm);
}
// FP three-same vector. The 'size' field carries {op_high, sz}: op_high (bit23)
// selects max(0)/min(1); sz (bit22) selects FP32(0)/FP64(1). FMAX/FMIN use
// opcode=11110, FMAXNM/FMINNM use opcode=11000, all U=0.
constexpr uint32_t FmaxVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/false, /*size=*/static_cast<uint8_t>(dbl), /*opcode=*/0b11110, rd, rn, rm);
}
constexpr uint32_t FminVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/false, /*size=*/static_cast<uint8_t>(0b10 | dbl), /*opcode=*/0b11110, rd, rn, rm);
}
constexpr uint32_t FmaxnmVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/false, /*size=*/static_cast<uint8_t>(dbl), /*opcode=*/0b11000, rd, rn, rm);
}
constexpr uint32_t FminnmVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/false, /*size=*/static_cast<uint8_t>(0b10 | dbl), /*opcode=*/0b11000, rd, rn, rm);
}
// FP three-same vector arithmetic. FADD op_high=0/opcode=11010/U=0;
// FSUB op_high=1/opcode=11010/U=0; FMUL op_high=0/opcode=11011/U=1;
// FDIV op_high=0/opcode=11111/U=1. size field = {op_high, sz}, sz=dbl.
constexpr uint32_t FaddVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/false, /*size=*/static_cast<uint8_t>(dbl), /*opcode=*/0b11010, rd, rn, rm);
}
constexpr uint32_t FsubVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/false, /*size=*/static_cast<uint8_t>(0b10 | dbl), /*opcode=*/0b11010, rd, rn, rm);
}
constexpr uint32_t FmulVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/true, /*size=*/static_cast<uint8_t>(dbl), /*opcode=*/0b11011, rd, rn, rm);
}
constexpr uint32_t FdivVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/true, /*size=*/static_cast<uint8_t>(dbl), /*opcode=*/0b11111, rd, rn, rm);
}
// FABD (vector): op_high=1, U=1, opcode=11010 -> |Vn - Vm| per lane.
constexpr uint32_t FabdVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/true, /*size=*/static_cast<uint8_t>(0b10 | dbl), /*opcode=*/0b11010, rd, rn, rm);
}
// FP three-same vector compares (opcode=11100). FCMEQ op_high=0/U=0;
// FCMGE op_high=0/U=1; FCMGT op_high=1/U=1. size field = {op_high, sz}, sz=dbl.
constexpr uint32_t FcmeqVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/false, /*size=*/static_cast<uint8_t>(dbl), /*opcode=*/0b11100, rd, rn, rm);
}
constexpr uint32_t FcmgeVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/true, /*size=*/static_cast<uint8_t>(dbl), /*opcode=*/0b11100, rd, rn, rm);
}
constexpr uint32_t FcmgtVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(
      q, /*u=*/true, /*size=*/static_cast<uint8_t>(0b10 | dbl), /*opcode=*/0b11100, rd, rn, rm);
}

// --- AdvSIMD two-register-miscellaneous encoders. ---
// Encoding: 0 Q U 01110 size(2) 1 0000 opcode(5) 10 Rn(5) Rd(5).
// Base (all fields zero) = bits[28:24]=01110 | bit21 | bit11 = 0x0E200800.
constexpr uint32_t AdvSimdTwoRegMisc(
    bool q, bool u, uint8_t size, uint8_t opcode, uint8_t rd, uint8_t rn) {
  return 0x0E200800u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(u) << 29) |
         (static_cast<uint32_t>(size) << 22) | (static_cast<uint32_t>(opcode) << 12) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
// REV16: U=0, opcode=00001, size=00.
constexpr uint32_t Rev16Vec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b00, /*opcode=*/0b00001, rd, rn);
}
// REV64: U=0, opcode=00000.
constexpr uint32_t Rev64Vec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b00000, rd, rn);
}
// CLZ: U=1, opcode=00100.
constexpr uint32_t ClzVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b00100, rd, rn);
}
// CLS: U=0, opcode=00100.
constexpr uint32_t ClsVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b00100, rd, rn);
}
// REV32: U=1, opcode=00000.
constexpr uint32_t Rev32Vec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b00000, rd, rn);
}
// FRINTA V (FP32, ties-away): a=0, U=1, opcode=11000, sz=0 -> size field 0b00.
constexpr uint32_t FrintaVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b00, /*opcode=*/0b11000, rd, rn);
}
static_assert(FrintaVec(/*q=*/true, 0, 1) == 0x6E218820u);   // frinta v0.4s, v1.4s
static_assert(FrintaVec(/*q=*/false, 0, 1) == 0x2E218820u);  // frinta v0.2s, v1.2s
// Vector FP two-reg-misc. size = {a=1, sz=dbl} = 0b10|dbl for FABS/FNEG/FSQRT/
// FRECPE/FRSQRTE; opcode picks the op. FABS/FRECPE U=0, FNEG/FRSQRTE U=1.
constexpr uint32_t FabsVec(bool dbl, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/static_cast<uint8_t>(0b10 | dbl),
                           /*opcode=*/0b01111, rd, rn);
}
constexpr uint32_t FnegVec(bool dbl, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/static_cast<uint8_t>(0b10 | dbl),
                           /*opcode=*/0b01111, rd, rn);
}
constexpr uint32_t FsqrtVec(bool dbl, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/static_cast<uint8_t>(0b10 | dbl),
                           /*opcode=*/0b11111, rd, rn);
}
constexpr uint32_t FrecpeVec(bool dbl, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/static_cast<uint8_t>(0b10 | dbl),
                           /*opcode=*/0b11101, rd, rn);
}
constexpr uint32_t FrsqrteVec(bool dbl, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/static_cast<uint8_t>(0b10 | dbl),
                           /*opcode=*/0b11101, rd, rn);
}
// FCVTL (opcode=10111, U=0) / FCVTN (opcode=10110, U=0), size=01 = FP32<->FP64.
// Q selects FCVTL2/FCVTN2 (high-half variant).
constexpr uint32_t FcvtlVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b01, /*opcode=*/0b10111, rd, rn);
}
constexpr uint32_t FcvtnVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b01, /*opcode=*/0b10110, rd, rn);
}
// FCVTXN (opcode=10110, U=1), size=01 = FP64->FP32 round-to-odd narrow.
// Q selects FCVTXN2 (high-half variant).
constexpr uint32_t FcvtxnVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b01, /*opcode=*/0b10110, rd, rn);
}
// SQABS (U=0) / SQNEG (U=1), opcode=00111.
constexpr uint32_t SqabsVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b00111, rd, rn);
}
constexpr uint32_t SqnegVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b00111, rd, rn);
}
// Armv8.2-FP16 vector three-same: 0 Q U 01110 a 1 0 Rm 00 opc3 1 Rn Rd.
// Base (all fields 0) = bits[28:24]=01110 | bit22 | bit10 = 0x0E400400.
constexpr uint32_t AdvSimdFp16ThreeSame(bool q, bool u, bool a, uint8_t opc3,
                                        uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E400400u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(u) << 29) |
         (static_cast<uint32_t>(a) << 23) | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(opc3) << 11) | (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FaddVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,0,0,0b010,d,n,m); }
constexpr uint32_t FsubVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,0,1,0b010,d,n,m); }
constexpr uint32_t FmulVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,1,0,0b011,d,n,m); }
constexpr uint32_t FdivVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,1,0,0b111,d,n,m); }
constexpr uint32_t FmaxVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,0,0,0b110,d,n,m); }
constexpr uint32_t FminVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,0,1,0b110,d,n,m); }
constexpr uint32_t FmaxnmVecH(bool q, uint8_t d, uint8_t n, uint8_t m) { return AdvSimdFp16ThreeSame(q,0,0,0b000,d,n,m); }
constexpr uint32_t FminnmVecH(bool q, uint8_t d, uint8_t n, uint8_t m) { return AdvSimdFp16ThreeSame(q,0,1,0b000,d,n,m); }
constexpr uint32_t FabdVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,1,1,0b010,d,n,m); }
constexpr uint32_t FcmeqVecH(bool q, uint8_t d, uint8_t n, uint8_t m)  { return AdvSimdFp16ThreeSame(q,0,0,0b100,d,n,m); }
constexpr uint32_t FcmgeVecH(bool q, uint8_t d, uint8_t n, uint8_t m)  { return AdvSimdFp16ThreeSame(q,1,0,0b100,d,n,m); }
constexpr uint32_t FcmgtVecH(bool q, uint8_t d, uint8_t n, uint8_t m)  { return AdvSimdFp16ThreeSame(q,1,1,0b100,d,n,m); }
constexpr uint32_t FacgeVecH(bool q, uint8_t d, uint8_t n, uint8_t m)  { return AdvSimdFp16ThreeSame(q,1,0,0b101,d,n,m); }
constexpr uint32_t FacgtVecH(bool q, uint8_t d, uint8_t n, uint8_t m)  { return AdvSimdFp16ThreeSame(q,1,1,0b101,d,n,m); }
constexpr uint32_t FmulxVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,0,0,0b011,d,n,m); }
constexpr uint32_t FaddpVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,1,0,0b010,d,n,m); }
constexpr uint32_t FmaxpVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,1,0,0b110,d,n,m); }
constexpr uint32_t FminpVecH(bool q, uint8_t d, uint8_t n, uint8_t m)   { return AdvSimdFp16ThreeSame(q,1,1,0b110,d,n,m); }
constexpr uint32_t FmaxnmpVecH(bool q, uint8_t d, uint8_t n, uint8_t m) { return AdvSimdFp16ThreeSame(q,1,0,0b000,d,n,m); }
constexpr uint32_t FminnmpVecH(bool q, uint8_t d, uint8_t n, uint8_t m) { return AdvSimdFp16ThreeSame(q,1,1,0b000,d,n,m); }
static_assert(FmulxVecH(false,0,1,2)   == 0x0E421C20u);
static_assert(FmulxVecH(true,0,1,2)    == 0x4E421C20u);
static_assert(FaddpVecH(false,0,1,2)   == 0x2E421420u);
static_assert(FaddpVecH(true,0,1,2)    == 0x6E421420u);
static_assert(FmaxpVecH(false,0,1,2)   == 0x2E423420u);
static_assert(FmaxpVecH(true,0,1,2)    == 0x6E423420u);
static_assert(FminpVecH(false,0,1,2)   == 0x2EC23420u);
static_assert(FmaxnmpVecH(false,0,1,2) == 0x2E420420u);
static_assert(FminnmpVecH(false,0,1,2) == 0x2EC20420u);
static_assert(FaddVecH(false,0,1,2)  == 0x0E421420u);  // fadd v0.4h,v1.4h,v2.4h
static_assert(FaddVecH(true,0,1,2)   == 0x4E421420u);  // fadd v0.8h,v1.8h,v2.8h
static_assert(FsubVecH(false,0,1,2)  == 0x0EC21420u);
static_assert(FmulVecH(false,0,1,2)  == 0x2E421C20u);
static_assert(FdivVecH(true,0,1,2)   == 0x6E423C20u);
static_assert(FmaxVecH(false,0,1,2)  == 0x0E423420u);
static_assert(FminVecH(false,0,1,2)  == 0x0EC23420u);
static_assert(FmaxnmVecH(false,0,1,2)== 0x0E420420u);
static_assert(FminnmVecH(false,0,1,2)== 0x0EC20420u);
static_assert(FabdVecH(false,0,1,2)  == 0x2EC21420u);
static_assert(FcmeqVecH(true,0,1,2)  == 0x4E422420u);
static_assert(FcmgeVecH(false,0,1,2) == 0x2E422420u);
static_assert(FcmgtVecH(true,0,1,2)  == 0x6EC22420u);
static_assert(FacgeVecH(false,0,1,2) == 0x2E422C20u);
static_assert(FacgtVecH(false,0,1,2) == 0x2EC22C20u);

// Armv8.2-FP16 vector two-reg misc: 0 Q U 01110 a 11110 opc5 10 Rn Rd.
// Base (all fields 0, assembler-verified) = bits[28:24]=01110 | bits[22:19]=1111 |
// bit11 = 0x0E780800.
constexpr uint32_t AdvSimdFp16TwoRegMisc(bool q, bool u, bool a, uint8_t opc5,
                                         uint8_t rd, uint8_t rn) {
  return 0x0E780800u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(u) << 29) |
         (static_cast<uint32_t>(a) << 23) | (static_cast<uint32_t>(opc5) << 12) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
constexpr uint32_t FabsVecH(bool q, uint8_t d, uint8_t n)      { return AdvSimdFp16TwoRegMisc(q,0,1,0b01111,d,n); }
constexpr uint32_t FnegVecH(bool q, uint8_t d, uint8_t n)      { return AdvSimdFp16TwoRegMisc(q,1,1,0b01111,d,n); }
constexpr uint32_t FsqrtVecH(bool q, uint8_t d, uint8_t n)     { return AdvSimdFp16TwoRegMisc(q,1,1,0b11111,d,n); }
constexpr uint32_t FrintnVecH(bool q, uint8_t d, uint8_t n)    { return AdvSimdFp16TwoRegMisc(q,0,0,0b11000,d,n); }
constexpr uint32_t FrintmVecH(bool q, uint8_t d, uint8_t n)    { return AdvSimdFp16TwoRegMisc(q,0,0,0b11001,d,n); }
constexpr uint32_t FrintpVecH(bool q, uint8_t d, uint8_t n)    { return AdvSimdFp16TwoRegMisc(q,0,1,0b11000,d,n); }
constexpr uint32_t FrintzVecH(bool q, uint8_t d, uint8_t n)    { return AdvSimdFp16TwoRegMisc(q,0,1,0b11001,d,n); }
constexpr uint32_t FrintxVecH(bool q, uint8_t d, uint8_t n)    { return AdvSimdFp16TwoRegMisc(q,1,0,0b11001,d,n); }
constexpr uint32_t FrintiVecH(bool q, uint8_t d, uint8_t n)    { return AdvSimdFp16TwoRegMisc(q,1,1,0b11001,d,n); }
constexpr uint32_t FrintaVecH(bool q, uint8_t d, uint8_t n)    { return AdvSimdFp16TwoRegMisc(q,1,0,0b11000,d,n); }
constexpr uint32_t FcmgtZeroVecH(bool q, uint8_t d, uint8_t n) { return AdvSimdFp16TwoRegMisc(q,0,1,0b01100,d,n); }
constexpr uint32_t FcmeqZeroVecH(bool q, uint8_t d, uint8_t n) { return AdvSimdFp16TwoRegMisc(q,0,1,0b01101,d,n); }
constexpr uint32_t FcmltZeroVecH(bool q, uint8_t d, uint8_t n) { return AdvSimdFp16TwoRegMisc(q,0,1,0b01110,d,n); }
constexpr uint32_t FcmgeZeroVecH(bool q, uint8_t d, uint8_t n) { return AdvSimdFp16TwoRegMisc(q,1,1,0b01100,d,n); }
constexpr uint32_t FcmleZeroVecH(bool q, uint8_t d, uint8_t n) { return AdvSimdFp16TwoRegMisc(q,1,1,0b01101,d,n); }
static_assert(FabsVecH(false,0,1)     == 0x0EF8F820u);  // fabs v0.4h,v1.4h
static_assert(FnegVecH(false,0,1)     == 0x2EF8F820u);
static_assert(FsqrtVecH(false,0,1)    == 0x2EF9F820u);
static_assert(FrintnVecH(false,0,1)   == 0x0E798820u);
static_assert(FrintmVecH(true,0,1)    == 0x4E799820u);
static_assert(FrintpVecH(false,0,1)   == 0x0EF98820u);
static_assert(FrintzVecH(false,0,1)   == 0x0EF99820u);
static_assert(FrintaVecH(false,0,1)   == 0x2E798820u);
static_assert(FrintxVecH(false,0,1)   == 0x2E799820u);
static_assert(FrintiVecH(false,0,1)   == 0x2EF99820u);
static_assert(FcmeqZeroVecH(false,0,1)== 0x0EF8D820u);  // fcmeq v0.4h,v1.4h,#0.0
static_assert(FcmgtZeroVecH(false,0,1)== 0x0EF8C820u);
static_assert(FcmgeZeroVecH(false,0,1)== 0x2EF8C820u);
static_assert(FcmltZeroVecH(false,0,1)== 0x0EF8E820u);
static_assert(FcmleZeroVecH(false,0,1)== 0x2EF8D820u);

// URECPE (U=0) / URSQRTE (U=1), opcode=11100, size=10 (32-bit lanes only).
constexpr uint32_t UrecpeVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b10, /*opcode=*/0b11100, rd, rn);
}
constexpr uint32_t UrsqrteVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b10, /*opcode=*/0b11100, rd, rn);
}
static_assert(UrecpeVec(true, 0, 1) == 0x4EA1C820u);   // urecpe v0.4s, v1.4s
static_assert(UrsqrteVec(true, 0, 1) == 0x6EA1C820u);  // ursqrte v0.4s, v1.4s
static_assert(UrecpeVec(false, 0, 1) == 0x0EA1C820u);  // urecpe v0.2s, v1.2s
static_assert(UrsqrteVec(false, 0, 1) == 0x2EA1C820u); // ursqrte v0.2s, v1.2s
// FP three-same abs-compares FACGE (opcode=11101, op_high=0, U=1) /
// FACGT (opcode=11101, op_high=1, U=1). size = {op_high, sz=dbl}.
constexpr uint32_t FacgeVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, /*size=*/static_cast<uint8_t>(dbl),
                          /*opcode=*/0b11101, rd, rn, rm);
}
constexpr uint32_t FacgtVec(bool dbl, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, /*size=*/static_cast<uint8_t>(0b10 | dbl),
                          /*opcode=*/0b11101, rd, rn, rm);
}
// Encodings verified with aarch64-linux-gnu-as + llvm-objdump:
static_assert(FabsVec(false, true, 0, 1) == 0x4EA0F820u);   // fabs v0.4s, v1.4s
static_assert(FabsVec(true, true, 0, 1) == 0x4EE0F820u);    // fabs v0.2d, v1.2d
static_assert(FnegVec(false, true, 0, 1) == 0x6EA0F820u);   // fneg v0.4s, v1.4s
static_assert(FnegVec(true, true, 0, 1) == 0x6EE0F820u);    // fneg v0.2d, v1.2d
static_assert(FsqrtVec(false, true, 0, 1) == 0x6EA1F820u);  // fsqrt v0.4s, v1.4s
static_assert(FsqrtVec(true, true, 0, 1) == 0x6EE1F820u);   // fsqrt v0.2d, v1.2d
static_assert(FsqrtVec(false, false, 0, 1) == 0x2EA1F820u); // fsqrt v0.2s, v1.2s
static_assert(FrecpeVec(false, true, 0, 1) == 0x4EA1D820u); // frecpe v0.4s, v1.4s
static_assert(FrsqrteVec(false, true, 0, 1) == 0x6EA1D820u);// frsqrte v0.4s, v1.4s
static_assert(FrecpeVec(true, true, 0, 1) == 0x4EE1D820u);  // frecpe v0.2d, v1.2d
static_assert(FrsqrteVec(true, true, 0, 1) == 0x6EE1D820u); // frsqrte v0.2d, v1.2d
static_assert(FcvtlVec(false, 0, 1) == 0x0E617820u);        // fcvtl v0.2d, v1.2s
static_assert(FcvtlVec(true, 0, 1) == 0x4E617820u);         // fcvtl2 v0.2d, v1.4s
static_assert(FcvtnVec(false, 0, 1) == 0x0E616820u);        // fcvtn v0.2s, v1.2d
static_assert(FcvtnVec(true, 0, 1) == 0x4E616820u);         // fcvtn2 v0.4s, v1.2d
static_assert(FcvtxnVec(false, 0, 1) == 0x2E616820u);       // fcvtxn v0.2s, v1.2d
static_assert(FcvtxnVec(true, 0, 1) == 0x6E616820u);        // fcvtxn2 v0.4s, v1.2d
static_assert(SqabsVec(0b10, true, 0, 1) == 0x4EA07820u);   // sqabs v0.4s, v1.4s
static_assert(SqnegVec(0b10, true, 0, 1) == 0x6EA07820u);   // sqneg v0.4s, v1.4s
static_assert(SqabsVec(0b00, false, 0, 1) == 0x0E207820u);  // sqabs v0.8b, v1.8b
static_assert(FacgeVec(false, true, 0, 1, 2) == 0x6E22EC20u);// facge v0.4s,v1,v2
static_assert(FacgtVec(false, true, 0, 1, 2) == 0x6EA2EC20u);// facgt v0.4s,v1,v2
static_assert(FacgeVec(true, true, 0, 1, 2) == 0x6E62EC20u); // facge v0.2d,v1,v2
// EXT Vd.<T>, Vn.<T>, Vm.<T>, #index:
//   0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd
constexpr uint32_t ExtVec(bool q, uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm4) {
  return 0x2e000000u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(imm4) << 11) | (static_cast<uint32_t>(rn) << 5) | rd;
}
// CNT: U=0, opcode=00101, size=00.
constexpr uint32_t CntVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b00, /*opcode=*/0b00101, rd, rn);
}
// NOT: U=1, opcode=00101, size=00.
constexpr uint32_t NotVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b00, /*opcode=*/0b00101, rd, rn);
}
// RBIT: U=1, opcode=00101, size=01.
constexpr uint32_t RbitVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b01, /*opcode=*/0b00101, rd, rn);
}
// NEG: U=1, opcode=01011.
constexpr uint32_t NegVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b01011, rd, rn);
}
// ABS: U=0, opcode=01011.
constexpr uint32_t AbsVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b01011, rd, rn);
}
// ADDV Vd, Vn.<T> (AdvSIMD across lanes): 0 Q 0 01110 size 11000 11011 10 Rn Rd.
// Base (all fields zero, size=00, rd=rn=0) = 0x0E31B800.
constexpr uint32_t AddvVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return 0x0E31B800u | (static_cast<uint32_t>(q) << 30) |
         (static_cast<uint32_t>(size) << 22) | (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(AddvVec(0b00, /*q=*/true, 0, 1) == 0x4e31b820u);   // addv b0, v1.16b
static_assert(AddvVec(0b00, /*q=*/false, 0, 1) == 0x0e31b820u);  // addv b0, v1.8b
static_assert(AddvVec(0b01, /*q=*/true, 0, 1) == 0x4e71b820u);   // addv h0, v1.8h
static_assert(AddvVec(0b01, /*q=*/false, 0, 1) == 0x0e71b820u);  // addv h0, v1.4h
static_assert(AddvVec(0b10, /*q=*/true, 0, 1) == 0x4eb1b820u);   // addv s0, v1.4s
// SADDLV/UADDLV Vd, Vn.<T>: 0 Q U 01110 size 11000 00011 10 Rn Rd.
// U=0 -> SADDLV (signed), U=1 -> UADDLV. Base (size=00, rd=rn=0) = 0x0E303800.
constexpr uint32_t AddlvVec(bool is_signed, uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return 0x0E303800u | (static_cast<uint32_t>(q) << 30) |
         (static_cast<uint32_t>(!is_signed) << 29) | (static_cast<uint32_t>(size) << 22) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(AddlvVec(false, 0b00, /*q=*/false, 0, 1) == 0x2e303820u);  // uaddlv h0, v1.8b
static_assert(AddlvVec(false, 0b00, /*q=*/true, 0, 1) == 0x6e303820u);   // uaddlv h0, v1.16b
static_assert(AddlvVec(false, 0b01, /*q=*/false, 0, 1) == 0x2e703820u);  // uaddlv s0, v1.4h
static_assert(AddlvVec(false, 0b01, /*q=*/true, 0, 1) == 0x6e703820u);   // uaddlv s0, v1.8h
static_assert(AddlvVec(false, 0b10, /*q=*/true, 0, 1) == 0x6eb03820u);   // uaddlv d0, v1.4s
static_assert(AddlvVec(true, 0b00, /*q=*/false, 0, 1) == 0x0e303820u);   // saddlv h0, v1.8b
static_assert(AddlvVec(true, 0b00, /*q=*/true, 0, 1) == 0x4e303820u);    // saddlv h0, v1.16b
static_assert(AddlvVec(true, 0b01, /*q=*/false, 0, 1) == 0x0e703820u);   // saddlv s0, v1.4h
static_assert(AddlvVec(true, 0b01, /*q=*/true, 0, 1) == 0x4e703820u);    // saddlv s0, v1.8h
static_assert(AddlvVec(true, 0b10, /*q=*/true, 0, 1) == 0x4eb03820u);    // saddlv d0, v1.4s
// SMAXV/SMINV/UMAXV/UMINV Vd, Vn.<T>: 0 Q U 01110 size 11000 opcode 10 Rn Rd.
// opcode = 01010 (max) or 11010 (min); U = 0 (signed) / 1 (unsigned).
// Base (max, signed, size=00, rd=rn=0) = 0x0E30A800.
constexpr uint32_t MaxminvVec(bool is_max, bool is_signed, uint8_t size, bool q,
                              uint8_t rd, uint8_t rn) {
  const uint32_t opcode = is_max ? 0b01010u : 0b11010u;
  return 0x0E300800u | (static_cast<uint32_t>(q) << 30) |
         (static_cast<uint32_t>(!is_signed) << 29) |
         (static_cast<uint32_t>(size) << 22) | (opcode << 12) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(MaxminvVec(true, true, 0b00, /*q=*/true, 0, 1) == 0x4e30a820u);    // smaxv b0, v1.16b
static_assert(MaxminvVec(true, true, 0b00, /*q=*/false, 0, 1) == 0x0e30a820u);   // smaxv b0, v1.8b
static_assert(MaxminvVec(true, true, 0b01, /*q=*/true, 0, 1) == 0x4e70a820u);    // smaxv h0, v1.8h
static_assert(MaxminvVec(true, true, 0b01, /*q=*/false, 0, 1) == 0x0e70a820u);   // smaxv h0, v1.4h
static_assert(MaxminvVec(true, true, 0b10, /*q=*/true, 0, 1) == 0x4eb0a820u);    // smaxv s0, v1.4s
static_assert(MaxminvVec(false, true, 0b00, /*q=*/true, 0, 1) == 0x4e31a820u);   // sminv b0, v1.16b
static_assert(MaxminvVec(false, true, 0b01, /*q=*/true, 0, 1) == 0x4e71a820u);   // sminv h0, v1.8h
static_assert(MaxminvVec(false, true, 0b10, /*q=*/true, 0, 1) == 0x4eb1a820u);   // sminv s0, v1.4s
static_assert(MaxminvVec(true, false, 0b00, /*q=*/true, 0, 1) == 0x6e30a820u);   // umaxv b0, v1.16b
static_assert(MaxminvVec(true, false, 0b00, /*q=*/false, 0, 1) == 0x2e30a820u);  // umaxv b0, v1.8b
static_assert(MaxminvVec(true, false, 0b01, /*q=*/true, 0, 1) == 0x6e70a820u);   // umaxv h0, v1.8h
static_assert(MaxminvVec(true, false, 0b10, /*q=*/true, 0, 1) == 0x6eb0a820u);   // umaxv s0, v1.4s
static_assert(MaxminvVec(false, false, 0b00, /*q=*/true, 0, 1) == 0x6e31a820u);  // uminv b0, v1.16b
static_assert(MaxminvVec(false, false, 0b01, /*q=*/true, 0, 1) == 0x6e71a820u);  // uminv h0, v1.8h
static_assert(MaxminvVec(false, false, 0b10, /*q=*/true, 0, 1) == 0x6eb1a820u);  // uminv s0, v1.4s
// FMAXV/FMINV/FMAXNMV/FMINNMV Sd, Vn.4S (FP32 across-lanes):
// 0 Q(1) U(1) 01110 sz(bit23=1 for min) 0 11000 opcode 10 Rn Rd.
// opcode = 01111 (FMAXV/FMINV) or 01100 (FMAXNMV/FMINNMV).
constexpr uint32_t FmaxvVec(bool is_max, bool is_nm, uint8_t rd, uint8_t rn) {
  const uint32_t opcode = is_nm ? 0b01100u : 0b01111u;
  const uint32_t bit23 = is_max ? 0u : 1u;  // size high bit selects min.
  return 0x6E300800u | (bit23 << 23) | (opcode << 12) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(FmaxvVec(true, false, 0, 1) == 0x6e30f820u);   // fmaxv s0, v1.4s
static_assert(FmaxvVec(false, false, 0, 1) == 0x6eb0f820u);  // fminv s0, v1.4s
static_assert(FmaxvVec(true, true, 0, 1) == 0x6e30c820u);    // fmaxnmv s0, v1.4s
static_assert(FmaxvVec(false, true, 0, 1) == 0x6eb0c820u);   // fminnmv s0, v1.4s
// FMAXV/FMINV/FMAXNMV/FMINNMV Hd, Vn.4H/.8H (Armv8.2-FP16 across-lanes):
// 0 Q U(0) 01110 sz(bit23=1 for min) 0 11000 opcode 10 Rn Rd.
// opcode = 01111 (FMAXV/FMINV) or 01100 (FMAXNMV/FMINNMV). U=0 is what selects
// the half-precision form; unlike the FP32 (.4S) form, Q=0 is a valid encoding
// and selects .4H.
constexpr uint32_t FmaxvVecH(bool is_max, bool is_nm, bool q, uint8_t rd, uint8_t rn) {
  const uint32_t opcode = is_nm ? 0b01100u : 0b01111u;
  const uint32_t bit23 = is_max ? 0u : 1u;  // size high bit selects min.
  return 0x0E300800u | (static_cast<uint32_t>(q) << 30) | (bit23 << 23) |
         (opcode << 12) | (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(FmaxvVecH(true, false, true, 0, 1) == 0x4e30f820u);    // fmaxv   h0, v1.8h
static_assert(FmaxvVecH(true, false, false, 0, 1) == 0x0e30f820u);   // fmaxv   h0, v1.4h
static_assert(FmaxvVecH(false, false, true, 0, 1) == 0x4eb0f820u);   // fminv   h0, v1.8h
static_assert(FmaxvVecH(false, false, false, 0, 1) == 0x0eb0f820u);  // fminv   h0, v1.4h
static_assert(FmaxvVecH(true, true, true, 0, 1) == 0x4e30c820u);     // fmaxnmv h0, v1.8h
static_assert(FmaxvVecH(true, true, false, 0, 1) == 0x0e30c820u);    // fmaxnmv h0, v1.4h
static_assert(FmaxvVecH(false, true, true, 0, 1) == 0x4eb0c820u);    // fminnmv h0, v1.8h
static_assert(FmaxvVecH(false, true, false, 0, 1) == 0x0eb0c820u);   // fminnmv h0, v1.4h
// FCVTZS/FCVTZU V Vd.<T>, Vn.<T> (FP32): two-reg-misc, opcode=11011, size=10
// (bit23=1 selects round-toward-zero); U=0 signed / U=1 unsigned.
constexpr uint32_t FcvtzVec(bool is_unsigned, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/is_unsigned, /*size=*/0b10, /*opcode=*/0b11011, rd, rn);
}
static_assert(FcvtzVec(false, true, 0, 1) == 0x4ea1b820u);   // fcvtzs v0.4s, v1.4s
static_assert(FcvtzVec(false, false, 0, 1) == 0x0ea1b820u);  // fcvtzs v0.2s, v1.2s
static_assert(FcvtzVec(true, true, 0, 1) == 0x6ea1b820u);    // fcvtzu v0.4s, v1.4s
static_assert(FcvtzVec(true, false, 0, 1) == 0x2ea1b820u);   // fcvtzu v0.2s, v1.2s
// Vector round-mode FP->int converts (FP32 .2S/.4S). U selects signed/unsigned,
// opcode/size select the rounding mode: FCVTN* (RNE) opcode=11010 size=x0;
// FCVTP* (+inf) opcode=11010 size=x1(bit23=1); FCVTM* (-inf) opcode=11011;
// FCVTA* (ties-away) opcode=11100.
constexpr uint32_t FcvtnsVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b00, /*opcode=*/0b11010, rd, rn);
}
constexpr uint32_t FcvtpsVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b10, /*opcode=*/0b11010, rd, rn);
}
constexpr uint32_t FcvtmsVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b00, /*opcode=*/0b11011, rd, rn);
}
constexpr uint32_t FcvtnuVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b00, /*opcode=*/0b11010, rd, rn);
}
constexpr uint32_t FcvtpuVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b10, /*opcode=*/0b11010, rd, rn);
}
constexpr uint32_t FcvtmuVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b00, /*opcode=*/0b11011, rd, rn);
}
constexpr uint32_t FcvtasVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, /*size=*/0b00, /*opcode=*/0b11100, rd, rn);
}
constexpr uint32_t FcvtauVec(bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, /*size=*/0b00, /*opcode=*/0b11100, rd, rn);
}
static_assert(FcvtnsVec(true, 0, 1) == 0x4e21a820u);   // fcvtns v0.4s, v1.4s
static_assert(FcvtnsVec(false, 0, 1) == 0x0e21a820u);  // fcvtns v0.2s, v1.2s
static_assert(FcvtpsVec(true, 0, 1) == 0x4ea1a820u);   // fcvtps v0.4s, v1.4s
static_assert(FcvtmsVec(true, 0, 1) == 0x4e21b820u);   // fcvtms v0.4s, v1.4s
static_assert(FcvtnuVec(true, 0, 1) == 0x6e21a820u);   // fcvtnu v0.4s, v1.4s
static_assert(FcvtpuVec(true, 0, 1) == 0x6ea1a820u);   // fcvtpu v0.4s, v1.4s
static_assert(FcvtmuVec(true, 0, 1) == 0x6e21b820u);   // fcvtmu v0.4s, v1.4s
static_assert(FcvtasVec(true, 0, 1) == 0x4e21c820u);   // fcvtas v0.4s, v1.4s
static_assert(FcvtasVec(false, 0, 1) == 0x0e21c820u);  // fcvtas v0.2s, v1.2s
static_assert(FcvtauVec(true, 0, 1) == 0x6e21c820u);   // fcvtau v0.4s, v1.4s
// SCVTF/UCVTF V Vd.<T>, Vn.<T> (FP32 .2S/.4S): int->FP, two-reg-misc,
// opcode=11101, size=00 (bit23=0 selects int->FP; bit23=1 is FRECPE/FRSQRTE);
// U=0 signed / U=1 unsigned.
constexpr uint32_t ScvtfVec(bool is_unsigned, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/is_unsigned, /*size=*/0b00, /*opcode=*/0b11101, rd, rn);
}
static_assert(ScvtfVec(false, true, 0, 1) == 0x4e21d820u);   // scvtf v0.4s, v1.4s
static_assert(ScvtfVec(false, false, 0, 1) == 0x0e21d820u);  // scvtf v0.2s, v1.2s
static_assert(ScvtfVec(true, true, 0, 1) == 0x6e21d820u);    // ucvtf v0.4s, v1.4s
static_assert(ScvtfVec(true, false, 0, 1) == 0x2e21d820u);   // ucvtf v0.2s, v1.2s
// SADDLP/UADDLP/SADALP/UADALP Vd.<Ta>, Vn.<Tb>: two-reg-misc, opcode=00010
// (add-long-pairwise) or 00110 (accumulate); U=0 signed / U=1 unsigned.
constexpr uint32_t AddlpVec(bool is_signed, bool is_accum, uint8_t size, bool q,
                            uint8_t rd, uint8_t rn) {
  const uint8_t opcode = is_accum ? 0b00110 : 0b00010;
  return AdvSimdTwoRegMisc(q, /*u=*/!is_signed, size, opcode, rd, rn);
}
static_assert(AddlpVec(true, false, 0b00, /*q=*/false, 0, 1) == 0x0e202820u);  // saddlp v0.4h,v1.8b
static_assert(AddlpVec(true, false, 0b00, /*q=*/true, 0, 1) == 0x4e202820u);   // saddlp v0.8h,v1.16b
static_assert(AddlpVec(true, false, 0b01, /*q=*/false, 0, 1) == 0x0e602820u);  // saddlp v0.2s,v1.4h
static_assert(AddlpVec(true, false, 0b10, /*q=*/true, 0, 1) == 0x4ea02820u);   // saddlp v0.2d,v1.4s
static_assert(AddlpVec(true, false, 0b10, /*q=*/false, 0, 1) == 0x0ea02820u);  // saddlp v0.1d,v1.2s
static_assert(AddlpVec(false, false, 0b00, /*q=*/false, 0, 1) == 0x2e202820u); // uaddlp v0.4h,v1.8b
static_assert(AddlpVec(false, false, 0b00, /*q=*/true, 0, 1) == 0x6e202820u);  // uaddlp v0.8h,v1.16b
static_assert(AddlpVec(false, false, 0b01, /*q=*/false, 0, 1) == 0x2e602820u); // uaddlp v0.2s,v1.4h
static_assert(AddlpVec(false, false, 0b10, /*q=*/true, 0, 1) == 0x6ea02820u);  // uaddlp v0.2d,v1.4s
static_assert(AddlpVec(true, true, 0b00, /*q=*/false, 0, 1) == 0x0e206820u);   // sadalp v0.4h,v1.8b
static_assert(AddlpVec(true, true, 0b01, /*q=*/false, 0, 1) == 0x0e606820u);   // sadalp v0.2s,v1.4h
static_assert(AddlpVec(false, true, 0b00, /*q=*/true, 0, 1) == 0x6e206820u);   // uadalp v0.8h,v1.16b
static_assert(AddlpVec(false, true, 0b10, /*q=*/true, 0, 1) == 0x6ea06820u);   // uadalp v0.2d,v1.4s
// XTN/XTN2: U=0, opcode=10010.
constexpr uint32_t XtnVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b10010, rd, rn);
}
// SQXTUN/SQXTUN2: U=1, opcode=10010.
constexpr uint32_t SqxtunVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b10010, rd, rn);
}
// SHLL/SHLL2: U=1, opcode=10011.
constexpr uint32_t ShllVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b10011, rd, rn);
}
// AdvSIMD shift by immediate (vector): 0 Q U 011110 immh immb opcode 1 Rn Rd.
constexpr uint32_t AdvSimdShiftImm(
    bool q, bool u, uint8_t immh, uint8_t immb, uint8_t opcode, uint8_t rd, uint8_t rn) {
  return 0x0F000400u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(u) << 29) |
         (static_cast<uint32_t>(immh) << 19) | (static_cast<uint32_t>(immb) << 16) |
         (static_cast<uint32_t>(opcode) << 11) | (static_cast<uint32_t>(rn) << 5) | rd;
}
// USHLL/USHLL2: U=1, opcode=10100.
constexpr uint32_t UshllVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b10100, rd, rn);
}
// SSHLL/SSHLL2: U=0, opcode=10100.
constexpr uint32_t SshllVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b10100, rd, rn);
}
// SHL: U=0, opcode=01010.
constexpr uint32_t ShlVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b01010, rd, rn);
}
// USHR: U=1, opcode=00000.
constexpr uint32_t UshrVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b00000, rd, rn);
}
// SSHR: U=0, opcode=00000.
constexpr uint32_t SshrVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b00000, rd, rn);
}
// SSRA (shift-accumulate): U=0, opcode=00010.
constexpr uint32_t SsraVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b00010, rd, rn);
}
// USRA (shift-accumulate): U=1, opcode=00010.
constexpr uint32_t UsraVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b00010, rd, rn);
}
// SLI (shift-left-insert): U=1, opcode=01010.
constexpr uint32_t SliVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b01010, rd, rn);
}
// SRI (shift-right-insert): U=1, opcode=01000.
constexpr uint32_t SriVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b01000, rd, rn);
}
// SRSHR (signed rounding right shift): U=0, opcode=00100.
constexpr uint32_t SrshrVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b00100, rd, rn);
}
// URSHR (unsigned rounding right shift): U=1, opcode=00100.
constexpr uint32_t UrshrVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b00100, rd, rn);
}
// SRSRA (signed rounding right shift accumulate): U=0, opcode=00110.
constexpr uint32_t SrsraVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b00110, rd, rn);
}
// URSRA (unsigned rounding right shift accumulate): U=1, opcode=00110.
constexpr uint32_t UrsraVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b00110, rd, rn);
}
// SQSHL (signed saturating shift left): U=0, opcode=01110.
constexpr uint32_t SqshlVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b01110, rd, rn);
}
// UQSHL (unsigned saturating shift left): U=1, opcode=01110.
constexpr uint32_t UqshlVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b01110, rd, rn);
}
// SQSHLU (signed saturating shift left unsigned): U=1, opcode=01100.
constexpr uint32_t SqshluVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b01100, rd, rn);
}
// SHRN/SHRN2: U=0, opcode=10000.
constexpr uint32_t ShrnVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b10000, rd, rn);
}
// RSHRN/RSHRN2: U=0, opcode=10001.
constexpr uint32_t RshrnVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b10001, rd, rn);
}
// SQSHRN/SQSHRN2 (signed saturating narrow): U=0, opcode=10010.
constexpr uint32_t SqshrnVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b10010, rd, rn);
}
// UQSHRN/UQSHRN2 (unsigned saturating narrow): U=1, opcode=10010.
constexpr uint32_t UqshrnVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b10010, rd, rn);
}
// SQSHRUN/SQSHRUN2 (signed saturating narrow to unsigned): U=1, opcode=10000.
constexpr uint32_t SqshrunVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b10000, rd, rn);
}
// SQRSHRN/SQRSHRN2 (signed rounding saturating narrow): U=0, opcode=10011.
constexpr uint32_t SqrshrnVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/false, immh, immb, /*opcode=*/0b10011, rd, rn);
}
// UQRSHRN/UQRSHRN2 (unsigned rounding saturating narrow): U=1, opcode=10011.
constexpr uint32_t UqrshrnVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b10011, rd, rn);
}
// SQRSHRUN/SQRSHRUN2 (signed rounding saturating narrow to unsigned): U=1,
// opcode=10001.
constexpr uint32_t SqrshrunVec(bool q, uint8_t immh, uint8_t immb, uint8_t rd, uint8_t rn) {
  return AdvSimdShiftImm(q, /*u=*/true, immh, immb, /*opcode=*/0b10001, rd, rn);
}
// SQXTN/SQXTN2: U=0, opcode=10100.
constexpr uint32_t SqxtnVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b10100, rd, rn);
}
// UQXTN/UQXTN2: U=1, opcode=10100.
constexpr uint32_t UqxtnVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b10100, rd, rn);
}
// CMEQ #0: U=0, opcode=01001.
constexpr uint32_t CmeqZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b01001, rd, rn);
}
// CMGT #0: U=0, opcode=01000.
constexpr uint32_t CmgtZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b01000, rd, rn);
}
// CMGE #0: U=1, opcode=01000.
constexpr uint32_t CmgeZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b01000, rd, rn);
}
// CMLE #0: U=1, opcode=01001.
constexpr uint32_t CmleZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b01001, rd, rn);
}
// CMLT #0: U=0, opcode=01010 (bit20=0 selects two-reg-misc, not SMAXV).
constexpr uint32_t CmltZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b01010, rd, rn);
}
// FP compare-against-zero (two-reg-misc, bit20=0, bit23=1 required → size&0b10).
// FP32 uses size=0b10, FP64 uses size=0b11 (Q=1 only).
// FCMGT #0.0: U=0, opcode=01100.  FCMGE #0.0: U=1, opcode=01100.
constexpr uint32_t FcmgtZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b01100, rd, rn);
}
constexpr uint32_t FcmgeZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b01100, rd, rn);
}
// FCMEQ #0.0: U=0, opcode=01101.  FCMLE #0.0: U=1, opcode=01101.
constexpr uint32_t FcmeqZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b01101, rd, rn);
}
constexpr uint32_t FcmleZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/true, size, /*opcode=*/0b01101, rd, rn);
}
// FCMLT #0.0: U=0, opcode=01110.
constexpr uint32_t FcmltZeroVec(uint8_t size, bool q, uint8_t rd, uint8_t rn) {
  return AdvSimdTwoRegMisc(q, /*u=*/false, size, /*opcode=*/0b01110, rd, rn);
}
// CMGE (vector, register, signed >=): U=0, opcode=00111.
constexpr uint32_t CmgeVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b00111, rd, rn, rm);
}
// CMHI (vector, register, unsigned >): U=1, opcode=00110.
constexpr uint32_t CmhiVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b00110, rd, rn, rm);
}
// CMHS (vector, register, unsigned >=): U=1, opcode=00111.
constexpr uint32_t CmhsVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b00111, rd, rn, rm);
}
// SMAX/SMIN (signed): U=0, opcode=01100/01101. UMAX/UMIN (unsigned): U=1.
constexpr uint32_t SmaxVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b01100, rd, rn, rm);
}
constexpr uint32_t SminVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b01101, rd, rn, rm);
}
constexpr uint32_t UmaxVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b01100, rd, rn, rm);
}
constexpr uint32_t UminVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b01101, rd, rn, rm);
}
static_assert(SmaxVec(0b00, /*q=*/true, 0, 1, 2) == 0x4e226420u);  // smax v0.16b,v1,v2
static_assert(SminVec(0b01, /*q=*/true, 0, 1, 2) == 0x4e626c20u);  // smin v0.8h,v1,v2
static_assert(UmaxVec(0b10, /*q=*/true, 0, 1, 2) == 0x6ea26420u);  // umax v0.4s,v1,v2
static_assert(UminVec(0b00, /*q=*/true, 0, 1, 2) == 0x6e226c20u);  // umin v0.16b,v1,v2
// SABD/UABD (abs diff): U=0/1, opcode=01110. SABA/UABA (abs diff accumulate):
// U=0/1, opcode=01111.
constexpr uint32_t SabdVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b01110, rd, rn, rm);
}
constexpr uint32_t UabdVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b01110, rd, rn, rm);
}
constexpr uint32_t SabaVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b01111, rd, rn, rm);
}
constexpr uint32_t UabaVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b01111, rd, rn, rm);
}
static_assert(SabdVec(0b00, /*q=*/true, 0, 1, 2) == 0x4e227420u);  // sabd v0.16b,v1,v2
static_assert(UabdVec(0b01, /*q=*/true, 0, 1, 2) == 0x6e627420u);  // uabd v0.8h,v1,v2
static_assert(SabaVec(0b10, /*q=*/true, 0, 1, 2) == 0x4ea27c20u);  // saba v0.4s,v1,v2
static_assert(UabaVec(0b00, /*q=*/true, 0, 1, 2) == 0x6e227c20u);  // uaba v0.16b,v1,v2

// Halving add/sub: SHADD/UHADD (opcode 0b00000), SRHADD/URHADD (opcode
// 0b00010), SHSUB/UHSUB (opcode 0b00100); U picks signed(0)/unsigned(1).
// Encodings assembler-verified (llvm).
constexpr uint32_t ShaddVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b00000, rd, rn, rm);
}
constexpr uint32_t UhaddVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b00000, rd, rn, rm);
}
constexpr uint32_t SrhaddVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b00010, rd, rn, rm);
}
constexpr uint32_t UrhaddVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b00010, rd, rn, rm);
}
constexpr uint32_t ShsubVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b00100, rd, rn, rm);
}
constexpr uint32_t UhsubVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b00100, rd, rn, rm);
}
static_assert(ShaddVec(0b01, /*q=*/true, 0, 1, 2) == 0x4e620420u);   // shadd v0.8h,v1,v2
static_assert(UhaddVec(0b00, /*q=*/true, 0, 1, 2) == 0x6e220420u);   // uhadd v0.16b,v1,v2
static_assert(UhaddVec(0b10, /*q=*/true, 0, 1, 2) == 0x6ea20420u);   // uhadd v0.4s,v1,v2
static_assert(SrhaddVec(0b10, /*q=*/true, 0, 1, 2) == 0x4ea21420u);  // srhadd v0.4s,v1,v2
static_assert(UrhaddVec(0b00, /*q=*/false, 0, 1, 2) == 0x2e221420u); // urhadd v0.8b,v1,v2
static_assert(ShsubVec(0b01, /*q=*/true, 0, 1, 2) == 0x4e622420u);   // shsub v0.8h,v1,v2
static_assert(UhsubVec(0b00, /*q=*/true, 0, 1, 2) == 0x6e222420u);   // uhsub v0.16b,v1,v2
static_assert(UhsubVec(0b10, /*q=*/true, 0, 1, 2) == 0x6ea22420u);   // uhsub v0.4s,v1,v2

// SQDMULH (opcode 0b10110, U=0) / SQRDMULH (opcode 0b10110, U=1): saturating
// doubling multiply-high, rounding variant. Encodings assembler-verified.
constexpr uint32_t SqdmulhVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b10110, rd, rn, rm);
}
constexpr uint32_t SqrdmulhVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b10110, rd, rn, rm);
}
static_assert(SqdmulhVec(0b01, /*q=*/true, 0, 1, 2) == 0x4e62b420u);   // sqdmulh v0.8h,v1,v2
static_assert(SqrdmulhVec(0b01, /*q=*/true, 0, 1, 2) == 0x6e62b420u);  // sqrdmulh v0.8h,v1,v2
static_assert(SqdmulhVec(0b10, /*q=*/true, 0, 1, 2) == 0x4ea2b420u);   // sqdmulh v0.4s,v1,v2
static_assert(SqrdmulhVec(0b10, /*q=*/true, 0, 1, 2) == 0x6ea2b420u);  // sqrdmulh v0.4s,v1,v2
static_assert(SqdmulhVec(0b10, /*q=*/false, 0, 1, 2) == 0x0ea2b420u);  // sqdmulh v0.2s,v1,v2

// AdvSIMD scalar three-same: 01 U 11110 size 1 Rm opcode(5) 1 Rn Rd.
constexpr uint32_t AdvSimdScalarThreeSame(
    bool u, uint8_t size, uint8_t opcode, uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x5E200400u | (static_cast<uint32_t>(u) << 29) |
         (static_cast<uint32_t>(size) << 22) | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(opcode) << 11) | (static_cast<uint32_t>(rn) << 5) | rd;
}
// SQDMULH / SQRDMULH (scalar, H/S): opcode=0b10110, U selects the rounding
// variant. Encodings assembler-verified.
constexpr uint32_t SqdmulhScalar(uint8_t size, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(/*u=*/false, size, /*opcode=*/0b10110, rd, rn, rm);
}
constexpr uint32_t SqrdmulhScalar(uint8_t size, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(/*u=*/true, size, /*opcode=*/0b10110, rd, rn, rm);
}
static_assert(SqdmulhScalar(0b01, 0, 1, 2) == 0x5e62b420u);   // sqdmulh h0,h1,h2
static_assert(SqrdmulhScalar(0b01, 0, 1, 2) == 0x7e62b420u);  // sqrdmulh h0,h1,h2
static_assert(SqdmulhScalar(0b10, 0, 1, 2) == 0x5ea2b420u);   // sqdmulh s0,s1,s2
static_assert(SqrdmulhScalar(0b10, 0, 1, 2) == 0x7ea2b420u);  // sqrdmulh s0,s1,s2

// Scalar saturating add/sub / shifts. opcode: SQADD/UQADD=0b00001,
// SQSUB/UQSUB=0b00101, SSHL/USHL=0b01000, SRSHL/URSHL=0b01010.
constexpr uint32_t SqaddScalar(uint8_t size, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(false, size, 0b00001, rd, rn, rm);
}
constexpr uint32_t UqaddScalar(uint8_t size, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(true, size, 0b00001, rd, rn, rm);
}
constexpr uint32_t SqsubScalar(uint8_t size, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(false, size, 0b00101, rd, rn, rm);
}
constexpr uint32_t UqsubScalar(uint8_t size, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(true, size, 0b00101, rd, rn, rm);
}
constexpr uint32_t SshlScalarD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(false, 0b11, 0b01000, rd, rn, rm);
}
constexpr uint32_t UshlScalarD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(true, 0b11, 0b01000, rd, rn, rm);
}
constexpr uint32_t SrshlScalarD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(false, 0b11, 0b01010, rd, rn, rm);
}
constexpr uint32_t UrshlScalarD(uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(true, 0b11, 0b01010, rd, rn, rm);
}
static_assert(SqaddScalar(0b00, 0, 1, 2) == 0x5e220c20u);  // sqadd b0,b1,b2
static_assert(SqaddScalar(0b01, 0, 1, 2) == 0x5e620c20u);  // sqadd h0,h1,h2
static_assert(SqaddScalar(0b10, 0, 1, 2) == 0x5ea20c20u);  // sqadd s0,s1,s2
static_assert(SqaddScalar(0b11, 0, 1, 2) == 0x5ee20c20u);  // sqadd d0,d1,d2
static_assert(UqaddScalar(0b00, 0, 1, 2) == 0x7e220c20u);  // uqadd b0,b1,b2
static_assert(UqaddScalar(0b11, 0, 1, 2) == 0x7ee20c20u);  // uqadd d0,d1,d2
static_assert(SqsubScalar(0b10, 0, 1, 2) == 0x5ea22c20u);  // sqsub s0,s1,s2
static_assert(SqsubScalar(0b11, 0, 1, 2) == 0x5ee22c20u);  // sqsub d0,d1,d2
static_assert(UqsubScalar(0b10, 0, 1, 2) == 0x7ea22c20u);  // uqsub s0,s1,s2
static_assert(UqsubScalar(0b11, 0, 1, 2) == 0x7ee22c20u);  // uqsub d0,d1,d2
static_assert(SshlScalarD(0, 1, 2)  == 0x5ee24420u);  // sshl  d0,d1,d2
static_assert(UshlScalarD(0, 1, 2)  == 0x7ee24420u);  // ushl  d0,d1,d2
static_assert(SrshlScalarD(0, 1, 2) == 0x5ee25420u);  // srshl d0,d1,d2
static_assert(UrshlScalarD(0, 1, 2) == 0x7ee25420u);  // urshl d0,d1,d2

// FP scalar three-same (S/D). FMULX/FRECPS opcode=0b11011/0b11111 size 0b00(S)/
// 0b01(D); FRSQRTS opcode=0b11111 size 0b10(S)/0b11(D); FABD opcode=0b11010
// size 0b10(S)/0b11(D).
constexpr uint32_t FmulxScalar(bool is_d, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(false, is_d ? 0b01 : 0b00, 0b11011, rd, rn, rm);
}
constexpr uint32_t FrecpsScalar(bool is_d, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(false, is_d ? 0b01 : 0b00, 0b11111, rd, rn, rm);
}
constexpr uint32_t FrsqrtsScalar(bool is_d, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(false, is_d ? 0b11 : 0b10, 0b11111, rd, rn, rm);
}
constexpr uint32_t FabdScalar(bool is_d, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdScalarThreeSame(true, is_d ? 0b11 : 0b10, 0b11010, rd, rn, rm);
}
static_assert(FmulxScalar(false, 0, 1, 2) == 0x5e22dc20u);   // fmulx s0,s1,s2
static_assert(FmulxScalar(true,  0, 1, 2) == 0x5e62dc20u);   // fmulx d0,d1,d2
static_assert(FrecpsScalar(false, 0, 1, 2) == 0x5e22fc20u);  // frecps s0,s1,s2
static_assert(FrecpsScalar(true,  0, 1, 2) == 0x5e62fc20u);  // frecps d0,d1,d2
static_assert(FrsqrtsScalar(false, 0, 1, 2) == 0x5ea2fc20u); // frsqrts s0,s1,s2
static_assert(FrsqrtsScalar(true,  0, 1, 2) == 0x5ee2fc20u); // frsqrts d0,d1,d2
static_assert(FabdScalar(false, 0, 1, 2) == 0x7ea2d420u);    // fabd s0,s1,s2
static_assert(FabdScalar(true,  0, 1, 2) == 0x7ee2d420u);    // fabd d0,d1,d2

// SQRDMLAH/SQRDMLSH scalar (three-same-extra, H/S). base 0x7E008400,
// size 0b01(H)/0b10(S), sub bit at bit11.
constexpr uint32_t SqrdmlScalar(bool is_sub, uint8_t size, uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x7E008400u | (static_cast<uint32_t>(size) << 22) |
         (static_cast<uint32_t>(is_sub) << 11) | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(SqrdmlScalar(false, 0b01, 0, 1, 2) == 0x7e428420u);  // sqrdmlah h0,h1,h2
static_assert(SqrdmlScalar(false, 0b10, 0, 1, 2) == 0x7e828420u);  // sqrdmlah s0,s1,s2
static_assert(SqrdmlScalar(true,  0b01, 0, 1, 2) == 0x7e428c20u);  // sqrdmlsh h0,h1,h2
static_assert(SqrdmlScalar(true,  0b10, 0, 1, 2) == 0x7e828c20u);  // sqrdmlsh s0,s1,s2

// AdvSIMD scalar two-register misc: 01 U 11110 sz(1) 1 00001 opcode(5) 10 Rn Rd
// (sz occupies bit22; the "1 00001" pattern occupies bits[23:17] with bit23=1).
// Scalar FCVTZS/FCVTZU are opcode=0b11011; sz=0 selects S(FP32), sz=1 D(FP64).
constexpr uint32_t FcvtzScalar(bool is_unsigned, bool is_double, uint8_t rd, uint8_t rn) {
  return 0x5ea1b800u | (static_cast<uint32_t>(is_unsigned) << 29) |
         (static_cast<uint32_t>(is_double) << 22) | (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(FcvtzScalar(false, false, 0, 1) == 0x5ea1b820u);  // fcvtzs s0, s1
static_assert(FcvtzScalar(true, false, 0, 1) == 0x7ea1b820u);   // fcvtzu s0, s1
static_assert(FcvtzScalar(false, true, 0, 1) == 0x5ee1b820u);   // fcvtzs d0, d1
static_assert(FcvtzScalar(true, true, 0, 1) == 0x7ee1b820u);    // fcvtzu d0, d1

// AdvSIMD scalar two-register misc SCVTF/UCVTF (int->float): opcode=0b11101,
// bit23=0 so size ∈ {00,01}; sz=bit22 selects S(0)/D(1); U=bit29 selects
// signed(0)/unsigned(1).  llvm-mc-21 encoding checks below.
constexpr uint32_t CvtfScalar(bool is_unsigned, bool is_double, uint8_t rd, uint8_t rn) {
  return 0x5e21d800u | (static_cast<uint32_t>(is_unsigned) << 29) |
         (static_cast<uint32_t>(is_double) << 22) | (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(CvtfScalar(false, false, 0, 1) == 0x5e21d820u);  // scvtf s0, s1
static_assert(CvtfScalar(true, false, 0, 1) == 0x7e21d820u);   // ucvtf s0, s1
static_assert(CvtfScalar(false, true, 0, 1) == 0x5e61d820u);   // scvtf d0, d1
static_assert(CvtfScalar(true, true, 0, 1) == 0x7e61d820u);    // ucvtf d0, d1

// AdvSIMD scalar two-register misc saturating extract-narrow SQXTN/UQXTN/SQXTUN:
// 01 U 11110 size 1 0000 opcode(5) 10 Rn Rd. size selects the dst lane width
// (00=B from H, 01=H from S, 10=S from D). SQXTN/UQXTN opcode=0b10100 (U=0/1),
// SQXTUN opcode=0b10010 (U=1). llvm-mc-21 encoding checks below.
constexpr uint32_t SqxtnScalar(bool u, uint8_t size, uint8_t opcode, uint8_t rd, uint8_t rn) {
  return 0x5e200800u | (static_cast<uint32_t>(u) << 29) |
         (static_cast<uint32_t>(size) << 22) | (static_cast<uint32_t>(opcode) << 12) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(SqxtnScalar(false, 0b00, 0b10100, 0, 1) == 0x5e214820u);  // sqxtn  b0, h1
static_assert(SqxtnScalar(false, 0b01, 0b10100, 0, 1) == 0x5e614820u);  // sqxtn  h0, s1
static_assert(SqxtnScalar(false, 0b10, 0b10100, 0, 1) == 0x5ea14820u);  // sqxtn  s0, d1
static_assert(SqxtnScalar(true, 0b00, 0b10100, 0, 1) == 0x7e214820u);   // uqxtn  b0, h1
static_assert(SqxtnScalar(true, 0b01, 0b10100, 0, 1) == 0x7e614820u);   // uqxtn  h0, s1
static_assert(SqxtnScalar(true, 0b10, 0b10100, 0, 1) == 0x7ea14820u);   // uqxtn  s0, d1
static_assert(SqxtnScalar(true, 0b00, 0b10010, 0, 1) == 0x7e212820u);   // sqxtun b0, h1
static_assert(SqxtnScalar(true, 0b10, 0b10010, 0, 1) == 0x7ea12820u);   // sqxtun s0, d1

// AdvSIMD scalar two-reg misc SQABS (U=0) / SQNEG (U=1), opcode=00111:
// 01 U 11110 size 10000 00111 10 Rn Rd.  size: 00=B,01=H,10=S,11=D.
// Encodings confirmed with clang --target=aarch64 + llvm-objdump.
constexpr uint32_t SqabsScalar(bool is_neg, uint8_t size, uint8_t rd, uint8_t rn) {
  return 0x5e207800u | (static_cast<uint32_t>(is_neg) << 29) |
         (static_cast<uint32_t>(size) << 22) | (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(SqabsScalar(false, 0b00, 0, 1) == 0x5e207820u);  // sqabs b0, b1
static_assert(SqabsScalar(false, 0b01, 0, 1) == 0x5e607820u);  // sqabs h0, h1
static_assert(SqabsScalar(false, 0b10, 0, 1) == 0x5ea07820u);  // sqabs s0, s1
static_assert(SqabsScalar(false, 0b11, 0, 1) == 0x5ee07820u);  // sqabs d0, d1
static_assert(SqabsScalar(true,  0b00, 0, 1) == 0x7e207820u);  // sqneg b0, b1
static_assert(SqabsScalar(true,  0b01, 0, 1) == 0x7e607820u);  // sqneg h0, h1
static_assert(SqabsScalar(true,  0b10, 0, 1) == 0x7ea07820u);  // sqneg s0, s1
static_assert(SqabsScalar(true,  0b11, 0, 1) == 0x7ee07820u);  // sqneg d0, d1

// AdvSIMD three different: 0 Q U 01110 size 1 Rm opcode(4) 00 Rn Rd.
constexpr uint32_t AdvSimdThreeDiff(
    bool q, bool u, uint8_t size, uint8_t opcode, uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x0E200000u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(u) << 29) |
         (static_cast<uint32_t>(size) << 22) | (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(opcode) << 12) | (static_cast<uint32_t>(rn) << 5) | rd;
}
// {S,U}MULL/{,2}: opcode=1100 (U picks sign).  {S,U}MLAL: opcode=1000.
// {S,U}MLSL: opcode=1010.
constexpr uint32_t UmullVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b1100, rd, rn, rm);
}
constexpr uint32_t SmullVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b1100, rd, rn, rm);
}
constexpr uint32_t UmlalVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b1000, rd, rn, rm);
}
constexpr uint32_t SmlalVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b1000, rd, rn, rm);
}
constexpr uint32_t UmlslVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b1010, rd, rn, rm);
}
constexpr uint32_t SmlslVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b1010, rd, rn, rm);
}
// Widening add/sub: {S,U}ADDL opcode=0000, {S,U}SUBL opcode=0010,
// {S,U}ADDW opcode=0001, {S,U}SUBW opcode=0011 (U picks sign).
constexpr uint32_t SaddlVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b0000, rd, rn, rm);
}
constexpr uint32_t UaddlVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b0000, rd, rn, rm);
}
constexpr uint32_t SsublVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b0010, rd, rn, rm);
}
constexpr uint32_t UsublVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b0010, rd, rn, rm);
}
constexpr uint32_t SaddwVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b0001, rd, rn, rm);
}
constexpr uint32_t UaddwVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b0001, rd, rn, rm);
}
constexpr uint32_t SsubwVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b0011, rd, rn, rm);
}
constexpr uint32_t UsubwVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b0011, rd, rn, rm);
}
// Narrowing high: ADDHN opcode=0100/U=0, RADDHN opcode=0100/U=1,
// SUBHN opcode=0110/U=0, RSUBHN opcode=0110/U=1.
constexpr uint32_t AddhnVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b0100, rd, rn, rm);
}
constexpr uint32_t RaddhnVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b0100, rd, rn, rm);
}
constexpr uint32_t SubhnVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b0110, rd, rn, rm);
}
constexpr uint32_t RsubhnVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b0110, rd, rn, rm);
}
// Polynomial multiply long: PMULL/PMULL2 opcode=1110, U=0.  size=00 is the
// .8H poly8 widening form; size=11 is PMULL64 (.1Q from two .1D lanes).
constexpr uint32_t PmullVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b1110, rd, rn, rm);
}
static_assert(PmullVec(0b00, false, 0, 1, 2) == 0x0E22E020u);  // pmull  v0.8h,v1.8b,v2.8b
static_assert(PmullVec(0b00, true, 0, 1, 2) == 0x4E22E020u);   // pmull2 v0.8h,v1.16b,v2.16b
static_assert(PmullVec(0b11, false, 0, 1, 2) == 0x0EE2E020u);  // pmull  v0.1q,v1.1d,v2.1d
static_assert(PmullVec(0b11, true, 0, 1, 2) == 0x4EE2E020u);   // pmull2 v0.1q,v1.2d,v2.2d
// Saturating doubling widening: SQDMULL opcode=1101, SQDMLAL opcode=1001,
// SQDMLSL opcode=1011 (all U=0).
constexpr uint32_t SqdmullVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b1101, rd, rn, rm);
}
constexpr uint32_t SqdmlalVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b1001, rd, rn, rm);
}
constexpr uint32_t SqdmlslVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b1011, rd, rn, rm);
}

// Absolute-difference-long: SABDL/UABDL opcode=0111, SABAL/UABAL opcode=0101.
constexpr uint32_t SabdlVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b0111, rd, rn, rm);
}
constexpr uint32_t UabdlVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b0111, rd, rn, rm);
}
constexpr uint32_t SabalVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/false, size, /*opcode=*/0b0101, rd, rn, rm);
}
constexpr uint32_t UabalVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeDiff(q, /*u=*/true, size, /*opcode=*/0b0101, rd, rn, rm);
}

// Helpers to write/read the scalar lane of a guest V register and to read its
// upper bytes (which an ARM scalar-FP write must zero).
void SetVf32(ThreadState* s, unsigned reg, float v) {
  std::memset(&s->cpu.v[reg], 0xAB, sizeof(s->cpu.v[reg]));  // poison upper bytes
  std::memcpy(&s->cpu.v[reg], &v, sizeof(v));
}
void SetVf64(ThreadState* s, unsigned reg, double v) {
  std::memset(&s->cpu.v[reg], 0xAB, sizeof(s->cpu.v[reg]));  // poison upper bytes
  std::memcpy(&s->cpu.v[reg], &v, sizeof(v));
}
float GetVf32(const ThreadState* s, unsigned reg) {
  float v;
  std::memcpy(&v, &s->cpu.v[reg], sizeof(v));
  return v;
}
double GetVf64(const ThreadState* s, unsigned reg) {
  double v;
  std::memcpy(&v, &s->cpu.v[reg], sizeof(v));
  return v;
}
// Upper 96 bits (FP32 scalar) / upper 64 bits (FP64 scalar) of V[reg].
uint64_t VUpperHi64(const ThreadState* s, unsigned reg) {
  uint64_t hi;
  std::memcpy(&hi, reinterpret_cast<const uint8_t*>(&s->cpu.v[reg]) + 8, sizeof(hi));
  return hi;
}
uint32_t VWord1(const ThreadState* s, unsigned reg) {
  uint32_t w;
  std::memcpy(&w, reinterpret_cast<const uint8_t*>(&s->cpu.v[reg]) + 4, sizeof(w));
  return w;
}

// Write/read the full 128-bit guest V register as two 64-bit halves (little
// endian: lo = bytes[0..7], hi = bytes[8..15]).
void SetV128(ThreadState* s, unsigned reg, uint64_t lo, uint64_t hi) {
  std::memcpy(reinterpret_cast<uint8_t*>(&s->cpu.v[reg]), &lo, sizeof(lo));
  std::memcpy(reinterpret_cast<uint8_t*>(&s->cpu.v[reg]) + 8, &hi, sizeof(hi));
}
uint64_t VLo64(const ThreadState* s, unsigned reg) {
  uint64_t lo;
  std::memcpy(&lo, reinterpret_cast<const uint8_t*>(&s->cpu.v[reg]), sizeof(lo));
  return lo;
}
// Pack two FP32 values into one 64-bit half (little-endian: lo in bits 0..31).
uint64_t Pack2xF32(float lo, float hi) {
  uint32_t a, b;
  std::memcpy(&a, &lo, sizeof(a));
  std::memcpy(&b, &hi, sizeof(b));
  return static_cast<uint64_t>(a) | (static_cast<uint64_t>(b) << 32);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FaddS) {
  static const uint32_t code[] = {FaddS(0, 1, 2)};
  SetVf32(&state_, 1, 1.5f);
  SetVf32(&state_, 2, 2.25f);
  SetVf32(&state_, 0, 99.0f);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 3.75f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);          // upper word zeroed
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);      // upper 64 bits zeroed
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FsubS) {
  static const uint32_t code[] = {FsubS(0, 1, 2)};
  SetVf32(&state_, 1, 5.0f);
  SetVf32(&state_, 2, 1.25f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 3.75f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmulS) {
  static const uint32_t code[] = {FmulS(0, 1, 2)};
  SetVf32(&state_, 1, 0.5f);
  SetVf32(&state_, 2, 3.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 1.5f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FdivS) {
  static const uint32_t code[] = {FdivS(0, 1, 2)};
  SetVf32(&state_, 1, 9.0f);
  SetVf32(&state_, 2, 4.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 2.25f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FaddD) {
  static const uint32_t code[] = {FaddD(0, 1, 2)};
  SetVf64(&state_, 1, 1.5);
  SetVf64(&state_, 2, 2.25);
  SetVf64(&state_, 0, 99.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 3.75);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);  // upper 64 bits zeroed
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FsubD) {
  static const uint32_t code[] = {FsubD(0, 1, 2)};
  SetVf64(&state_, 1, 5.0);
  SetVf64(&state_, 2, 1.25);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 3.75);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmulD) {
  static const uint32_t code[] = {FmulD(0, 1, 2)};
  SetVf64(&state_, 1, 0.5);
  SetVf64(&state_, 2, 3.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 1.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FdivD) {
  static const uint32_t code[] = {FdivD(0, 1, 2)};
  SetVf64(&state_, 1, 9.0);
  SetVf64(&state_, 2, 4.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 2.25);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMOV Sd, Sn: bit-exact copy of lane 0, upper bytes zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovRegS) {
  static const uint32_t code[] = {FmovRegS(0, 1)};
  SetVf32(&state_, 1, -7.5f);
  SetVf32(&state_, 0, 1.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), -7.5f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmovRegD) {
  static const uint32_t code[] = {FmovRegD(0, 1)};
  SetVf64(&state_, 1, -7.5);
  SetVf64(&state_, 0, 1.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), -7.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMOV (general): move between a general register and a scalar FP register.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovGenDFromX) {
  // FMOV D0, X1: X -> D (lane 0), upper 64 bits zeroed.
  static const uint32_t code[] = {0x9E670020u};  // fmov d0, x1
  state_.cpu.x[1] = 0x1122334455667788ULL;
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // must be overwritten
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1122334455667788ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmovGenXFromD) {
  // FMOV X0, D1: D (lane 0) -> X.
  static const uint32_t code[] = {0x9E660020u};  // fmov x0, d1
  SetV128(&state_, 1, 0xAABBCCDDEEFF0011ULL, 0x7777777777777777ULL);
  state_.cpu.x[0] = 0xFFFFFFFFFFFFFFFFULL;  // dirty, must be overwritten
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0xAABBCCDDEEFF0011ULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmovGenSFromW) {
  // FMOV S0, W1: low 32 of X1 -> S (lane 0), upper bytes zeroed.
  static const uint32_t code[] = {0x1E270020u};  // fmov s0, w1
  state_.cpu.x[1] = 0xFFFFFFFF1234CAFEULL;  // only low 32 used
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000001234CAFEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmovGenWFromS) {
  // FMOV W0, S1: low 32 of D1 -> W (zero-extended into X0).
  static const uint32_t code[] = {0x1E260020u};  // fmov w0, s1
  SetV128(&state_, 1, 0xAAAAAAAADEADBEEFULL, 0x7777777777777777ULL);
  state_.cpu.x[0] = 0xFFFFFFFFFFFFFFFFULL;  // dirty upper must be cleared
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0x00000000DEADBEEFULL);
}

// FABS Sd, Sn: clear sign bit.
TEST_F(Arm64HeavyOptimizerFrontendTest, FabsS) {
  static const uint32_t code[] = {FabsS(0, 1)};
  SetVf32(&state_, 1, -3.5f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 3.5f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FNEG Sd, Sn: flip sign bit.
TEST_F(Arm64HeavyOptimizerFrontendTest, FnegS) {
  static const uint32_t code[] = {FnegS(0, 1)};
  SetVf32(&state_, 1, 3.5f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), -3.5f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FABS Dd, Dn: clear sign bit (high word).
TEST_F(Arm64HeavyOptimizerFrontendTest, FabsD) {
  static const uint32_t code[] = {FabsD(0, 1)};
  SetVf64(&state_, 1, -3.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 3.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FNEG Dd, Dn: flip sign bit (high word).
TEST_F(Arm64HeavyOptimizerFrontendTest, FnegD) {
  static const uint32_t code[] = {FnegD(0, 1)};
  SetVf64(&state_, 1, 3.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), -3.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FRINTA (round to nearest, ties AWAY from zero). x86 has no ROUND* imm for
// ties-away, so the heavy tier uses copysign(0.5)+truncate with a branchless
// magnitude gate (add 0.5 only where |x| < 2^23 / 2^52). Mirrors the lite
// scalar FRINTA cases.
//
// Halfway tie: FRINTA(2.5) -> 3.0 (away), NOT 2.0 (RNE).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaSTiesAwayPos) {
  static const uint32_t code[] = {FrintaS(0, 1)};
  SetVf32(&state_, 1, 2.5f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 3.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaSTiesAwayNeg) {
  static const uint32_t code[] = {FrintaS(0, 1)};
  SetVf32(&state_, 1, -2.5f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), -3.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// Non-tie: FRINTA(0.4) -> 0.0, FRINTA(0.6) -> 1.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaSNonTie) {
  static const uint32_t code[] = {FrintaS(0, 1)};
  SetVf32(&state_, 1, 0.4f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 0.0f);
  SetVf32(&state_, 1, 0.6f);
  ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 1.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// Magnitude-gate edge: 2^23+1 is already an integer with FP step >= 1, so the
// 0.5 addend must be gated off (else RNE would bump the odd value to even).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaSOddIntegerAboveThreshold) {
  static const uint32_t code[] = {FrintaS(0, 1)};
  SetVf32(&state_, 1, 8388609.0f);  // 2^23 + 1, exactly representable
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 8388609.0f);  // unchanged (NOT 8388610.0f)
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FRINTA(0.5d) -> 1.0d (ties away).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaDTiesAway) {
  static const uint32_t code[] = {FrintaD(0, 1)};
  SetVf64(&state_, 1, 0.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 1.0);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaDNegTiesAway) {
  static const uint32_t code[] = {FrintaD(0, 1)};
  SetVf64(&state_, 1, -3.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), -4.0);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FP64 magnitude-gate edge: 2^52+1 must be left untouched.
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaDOddIntegerAboveThreshold) {
  static const uint32_t code[] = {FrintaD(0, 1)};
  SetVf64(&state_, 1, 4503599627370497.0);  // 2^52 + 1, exactly representable
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 4503599627370497.0);  // unchanged
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// Vector FRINTA .4S: all four FP32 lanes rounded ties-away independently.
// Lanes exercise pos/neg ties (2.5->3, -2.5->-3) and non-ties (0.4->0, 0.6->1).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaVec4S) {
  static const uint32_t code[] = {FrintaVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.5f, -2.5f), Pack2xF32(0.4f, 0.6f));
  SetV128(&state_, 0, 0xDEADBEEFDEADBEEFull, 0xDEADBEEFDEADBEEFull);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), Pack2xF32(3.0f, -3.0f));
  EXPECT_EQ(VUpperHi64(&state_, 0), Pack2xF32(0.0f, 1.0f));
}

// Vector FRINTA .2S: 64-bit form must zero the upper 64 bits of Vd (Q=0).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaVec2S) {
  static const uint32_t code[] = {FrintaVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(3.5f, -0.5f), 0xDEADBEEFDEADBEEFull);
  SetV128(&state_, 0, 0xDEADBEEFDEADBEEFull, 0xDEADBEEFDEADBEEFull);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), Pack2xF32(4.0f, -1.0f));  // 3.5->4 (tie away), -0.5->-1
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);                 // Q=0 zeroes upper 64
}

// Vector FRINTA .4S magnitude-gate: 2^23+1 (odd integer, FP step >= 1) must be
// left untouched in every lane (the 0.5 addend is gated off).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaVec4SOddIntegerAboveThreshold) {
  static const uint32_t code[] = {FrintaVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(8388609.0f, -8388609.0f), Pack2xF32(1.5f, 8388609.0f));
  SetV128(&state_, 0, 0xDEADBEEFDEADBEEFull, 0xDEADBEEFDEADBEEFull);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), Pack2xF32(8388609.0f, -8388609.0f));  // unchanged
  EXPECT_EQ(VUpperHi64(&state_, 0), Pack2xF32(2.0f, 8388609.0f));    // 1.5->2, gated
}

// FMOV Sd, #1.0 (imm8 = 0x70 encodes +1.0 in single precision).
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovImmSOne) {
  static const uint32_t code[] = {FmovImmS(0, 0x70)};
  SetVf32(&state_, 0, 99.0f);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 1.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMOV Dd, #-2.0 (imm8 = 0x80 encodes -2.0 in double precision: VFPExpandImm
// sign=1, exp=1024 -> 2^1).
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovImmDNegTwo) {
  static const uint32_t code[] = {FmovImmD(0, 0x80)};
  SetVf64(&state_, 0, 99.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), -2.0);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCSEL Sd|Dd, Sn, Sm, cond. The heavy tier lowers this branchlessly (0/-1
// mask + PAND/PANDN/POR blend). Each region seeds NZCV with a real compare
// (mirrors the CSEL tests above), then selects, and asserts the chosen operand
// plus the architectural zeroing of V[rd]'s upper bytes.

// EQ taken: X0==5, CMP X0,#5 -> Z=1 -> Vd = Vn. Vd's upper bytes must zero.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselSEqTaken) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),             // Z=1
      FcselS(0, 1, 2, kCondEQ),  // EQ -> Vn (1.25f)
  };
  SetVf32(&state_, 1, 1.25f);
  SetVf32(&state_, 2, 2.5f);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 1.25f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// EQ not taken: CMP X0,#6 -> Z=0 -> Vd = Vm.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselSEqNotTaken) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 6),             // Z=0
      FcselS(0, 1, 2, kCondEQ),  // EQ false -> Vm (2.5f)
  };
  SetVf32(&state_, 1, 1.25f);
  SetVf32(&state_, 2, 2.5f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 2.5f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// LT taken (D): 3 - 5 -> N=1, V=0 -> LT (N!=V) holds -> Vn.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselDLtTaken) {
  static const uint32_t code[] = {
      MovzX(0, 3),
      CmpImmX(0, 5),             // N=1, V=0
      FcselD(0, 1, 2, kCondLT),  // LT -> Vn (3.14159)
  };
  SetVf64(&state_, 1, 3.14159);
  SetVf64(&state_, 2, 2.71828);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 3.14159);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// GE not taken (D): N=1, V=0 -> GE (N==V) fails -> Vm.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselDGeNotTaken) {
  static const uint32_t code[] = {
      MovzX(0, 3),
      CmpImmX(0, 5),             // N=1, V=0
      FcselD(0, 1, 2, kCondGE),  // GE false -> Vm (2.71828)
  };
  SetVf64(&state_, 1, 3.14159);
  SetVf64(&state_, 2, 2.71828);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 2.71828);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// HI taken (D): 10 - 5 -> C=1 (no borrow), Z=0 -> HI (C&&!Z) holds -> Vn.
// Exercises the compound-condition predicate path.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselDHiTaken) {
  static const uint32_t code[] = {
      MovzX(0, 10),
      CmpImmX(0, 5),             // C=1, Z=0
      FcselD(0, 1, 2, kCondHI),  // HI -> Vn (100.5)
  };
  SetVf64(&state_, 1, 100.5);
  SetVf64(&state_, 2, -0.25);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 100.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// AL (cond 0xE) is unconditional and always selects Vn regardless of flags.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselSAlwaysVn) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 6),             // Z=0 (would fail a real condition)
      FcselS(0, 1, 2, kCondAL),  // AL -> Vn (-7.5f) unconditionally
  };
  SetVf32(&state_, 1, -7.5f);
  SetVf32(&state_, 2, 2.5f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), -7.5f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCSEL is a bitwise select, not a numeric one: selecting -0.0 over +0.0 must
// preserve the sign bit (they compare numerically equal). EQ taken -> Vn=-0.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselDBitExactNegZero) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),             // Z=1
      FcselD(0, 1, 2, kCondEQ),  // EQ -> Vn (-0.0)
  };
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xABABABABABABABABULL);  // -0.0 + poison
  SetVf64(&state_, 2, 0.0);                                            // +0.0
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x8000000000000000ULL);  // sign bit preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// --- Region-interaction stress tests. ---
// The isolated tests above always PASS even with a broken lowering because a
// straight-line select-and-store region has no optimization opportunity to
// exploit. These regions REUSE an FCSEL source operand later in the same
// region, so the heavy optimizer's local guest-context forwarding
// (RemoveLocalGuestContextAccesses) may reuse the vreg that held V[rn]. If the
// blend mutates that vreg in place (a non-SSA read-modify-write), the forwarded
// value is the clobbered mask-blend result, not the original V[rn] -- a
// deterministic miscompile that only surfaces when a live source is re-read.

// FCSEL reads V1; a later FADD re-reads V1. V4 must be 2*V1 (V1 unclobbered).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselDSourceReusedByLaterFadd) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),             // Z=1 -> EQ taken
      FcselD(3, 1, 2, kCondEQ),  // V3 = EQ ? V1 : V2 = V1
      FaddD(4, 1, 1),            // V4 = V1 + V1  (re-reads V1)
  };
  SetVf64(&state_, 1, 3.5);
  SetVf64(&state_, 2, 9.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 3), 3.5);  // selected V1
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 4), 7.0);  // 2*V1, proves V1 survived
}

// Not-taken variant: FCSEL selects V2 but clobbers the vreg holding V1's load;
// the later FADD on V1 must still see the true V1.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselDSourceReusedNotTaken) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 6),             // Z=0 -> EQ not taken
      FcselD(3, 1, 2, kCondEQ),  // V3 = V2
      FaddD(4, 1, 1),            // V4 = 2*V1
  };
  SetVf64(&state_, 1, 3.5);
  SetVf64(&state_, 2, 9.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 3), 9.0);  // selected V2
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 4), 7.0);  // 2*V1
}

// Two FCSELs sharing source V1 (both taken). The second must see the original
// V1, not the first select's in-place-blended vreg.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselDTwoSelectsSharedSource) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),             // Z=1 -> EQ taken
      FcselD(3, 1, 2, kCondEQ),  // V3 = V1
      FcselD(4, 1, 5, kCondEQ),  // V4 = V1  (re-reads V1)
  };
  SetVf64(&state_, 1, 3.5);
  SetVf64(&state_, 2, 9.0);
  SetVf64(&state_, 5, -2.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 3), 3.5);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 4), 3.5);
}

// The FALSE-operand (V2) is also mutated in place (PANDN). Re-read V2 after the
// select and confirm it survived.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselDFalseOperandReused) {
  static const uint32_t code[] = {
      MovzX(0, 5),
      CmpImmX(0, 5),             // Z=1 -> EQ taken -> selects V1
      FcselD(3, 1, 2, kCondEQ),  // V3 = V1, but V2 is PANDN'd in place
      FaddD(4, 2, 2),            // V4 = 2*V2 (re-reads V2)
  };
  SetVf64(&state_, 1, 3.5);
  SetVf64(&state_, 2, 9.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 3), 3.5);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 4), 18.0);  // 2*V2, proves V2 survived
}

// FCSEL between two flag-consumers sharing the same NZCV. FCSEL's predicate
// arithmetic (SUBQ etc.) clobbers host EFLAGS; a LATER integer CSEL must still
// read the guest NZCV that the CMP materialized, not stale host flags.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcselBetweenFlagConsumers) {
  static const uint32_t code[] = {
      MovzX(1, 100),
      MovzX(2, 200),
      MovzX(0, 5),
      SubsImmX(31, 0, 5),        // CMP X0,#5 -> Z=1
      FcselD(3, 4, 5, kCondEQ),  // V3 = EQ ? V4 : V5 (clobbers host EFLAGS)
      CselX(6, 1, 2, kCondEQ),   // X6 = EQ ? X1 : X2  -> must be 100
  };
  SetVf64(&state_, 4, 1.5);
  SetVf64(&state_, 5, 2.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 3), 1.5);   // EQ -> V4
  EXPECT_EQ(state_.cpu.x[6], uint64_t{100});    // EQ -> X1, flags intact
}

// REPRO CANDIDATE: loop whose FCSEL source V1 is loop-invariant. The loop
// guest-context optimizer hoists GET(V1); if the blend mutates that hoisted
// vreg in place, iteration 2+ reads corrupted data. Not-taken so mask=0
// (the destructive case: Pand(vn,0) zeroes vn).
TEST_F(Arm64HeavyOptimizerFrontendTest, ReproLoopFcselInvariantSource) {
  static const uint32_t code[] = {
      MovzX(0, 3),            // [0] X0 = 3 (counter)
      // loop top = [1] (code+4)
      CmpImmX(5, 6),          // [1] CMP X5,#6 -> Z=0 (EQ not taken), X5==5
      FcselD(4, 1, 2, kCondEQ),  // [2] d4 = EQ ? V1 : V2 = V2; mutates hoisted V1
      FaddD(3, 3, 1),         // [3] acc(V3) += V1
      SubsImmX(0, 0, 1),      // [4] X0--
      Bcond(kCondNE, -16),    // [5] (code+0x14) B.NE -> code+4 ([1])
  };
  state_.cpu.x[5] = 5;
  SetVf64(&state_, 1, 1.5);
  SetVf64(&state_, 2, 9.0);
  SetVf64(&state_, 3, 0.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 4), 9.0);  // last select -> V2
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 3), 4.5);  // 3 * V1, proves V1 not corrupted
}

// REPRO CANDIDATE: in-place FCSEL (rd==rn) in a loop. d1 = cond ? d1 : d2.
// Taken every iteration keeps d1 == V1; a broken in-place blend on the hoisted
// GET diverges.
TEST_F(Arm64HeavyOptimizerFrontendTest, ReproLoopFcselInPlaceRdEqRn) {
  static const uint32_t code[] = {
      MovzX(0, 3),            // [0] X0 = 3
      // loop top = [1]
      CmpImmX(5, 5),          // [1] Z=1 (EQ taken), X5==5
      FcselD(1, 1, 2, kCondEQ),  // [2] d1 = EQ ? d1 : d2 = d1 (in place)
      FaddD(3, 3, 1),         // [3] acc += d1
      SubsImmX(0, 0, 1),      // [4] X0--
      Bcond(kCondNE, -16),    // [5] back to [1]
  };
  state_.cpu.x[5] = 5;
  SetVf64(&state_, 1, 2.0);
  SetVf64(&state_, 2, 100.0);
  SetVf64(&state_, 3, 0.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 1), 2.0);  // stays V1
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 3), 6.0);  // 3 * V1
}

// REPRO CANDIDATE: alternating select feeding an accumulator, source re-read.
TEST_F(Arm64HeavyOptimizerFrontendTest, ReproLoopFcselFalseThenReadTrue) {
  static const uint32_t code[] = {
      MovzX(0, 4),            // [0] X0 = 4
      CmpImmX(5, 6),          // [1] Z=0 -> not taken
      FcselD(4, 1, 2, kCondEQ),  // [2] d4 = V2; hoisted V1 clobbered if broken
      FaddD(3, 3, 4),         // [3] acc += d4 (= V2)
      FaddD(3, 3, 1),         // [4] acc += V1 (re-read hoisted V1)
      SubsImmX(0, 0, 1),      // [5] X0--
      Bcond(kCondNE, -20),    // [6] (code+0x18) -> code+4 ([1])
  };
  state_.cpu.x[5] = 5;
  SetVf64(&state_, 1, 1.0);
  SetVf64(&state_, 2, 10.0);
  SetVf64(&state_, 3, 0.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // Each iter adds V2 + V1 = 11; 4 iters -> 44.
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 3), 44.0);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfSFromX) {
  static const uint32_t code[] = {0x9e220020u};  // scvtf s0, x1
  state_.cpu.x[1] = static_cast<uint64_t>(int64_t{-1000003});
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), static_cast<float>(int64_t{-1000003}));
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfDFromW) {
  static const uint32_t code[] = {0x1e620020u};  // scvtf d0, w1
  state_.cpu.x[1] = 0xFFFFFF85ULL;               // W1 = -123 (signed)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), static_cast<double>(int32_t{-123}));
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfSFromWzr) {
  static const uint32_t code[] = {0x1e2203e0u};  // scvtf s0, wzr
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 0.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfSFromW) {
  static const uint32_t code[] = {0x1e230020u};  // ucvtf s0, w1
  state_.cpu.x[1] = 0xFFFFFFFFULL;               // W1 = 4294967295 (unsigned)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), static_cast<float>(uint32_t{0xFFFFFFFFu}));
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfDFromW) {
  static const uint32_t code[] = {0x1e630020u};  // ucvtf d0, w1
  state_.cpu.x[1] = 0x1FFFFFFFFULL;              // upper bits ignored; W1 = 0xFFFFFFFF
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), static_cast<double>(uint32_t{0xFFFFFFFFu}));
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfDFromXSmall) {
  // Source < 2^63: the direct Q-convert path.
  static const uint32_t code[] = {0x9e630020u};  // ucvtf d0, x1
  state_.cpu.x[1] = 5ULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 5.0);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfDFromXLarge) {
  // Source >= 2^63: exercises the round-to-odd halve/convert/double fix-up.
  static const uint32_t code[] = {0x9e630020u};  // ucvtf d0, x1
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;       // UINT64_MAX
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), static_cast<double>(uint64_t{0xFFFFFFFFFFFFFFFFULL}));
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfSFromXLarge) {
  static const uint32_t code[] = {0x9e230020u};  // ucvtf s0, x1
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;       // UINT64_MAX
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), static_cast<float>(uint64_t{0xFFFFFFFFFFFFFFFFULL}));
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// ---------------------------------------------------------------------------
// FCVTZS / FCVTZU (scalar FP -> integer, truncate toward zero). Each result is
// cross-checked against the ARM by-sign saturation rules the heavy fix-up ladder
// rebuilds (NaN -> 0; positive overflow -> INT_MAX/UINT_MAX; negative overflow
// -> INT_MIN; FCVTZU of a negative -> 0).
// ---------------------------------------------------------------------------

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsWFromS) {
  static const uint32_t code[] = {0x1e380020u};  // fcvtzs w0, s1
  SetVf32(&state_, 1, 12.9f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 12u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsXFromD) {
  static const uint32_t code[] = {0x9e780020u};  // fcvtzs x0, d1
  SetVf64(&state_, 1, -12.9);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(int64_t{-12}));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsWFromD) {
  static const uint32_t code[] = {0x1e780020u};  // fcvtzs w0, d1
  SetVf64(&state_, 1, 100.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 100u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsWFromSNan) {
  static const uint32_t code[] = {0x1e380020u};  // fcvtzs w0, s1
  SetV128(&state_, 1, 0x7FC00000ULL, 0ULL);      // lane 0 = FP32 qNaN
  state_.cpu.x[0] = 0xdeadbeefULL;               // dirty, must be overwritten
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);  // NaN -> 0
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsWFromSPosOverflow) {
  static const uint32_t code[] = {0x1e380020u};  // fcvtzs w0, s1
  SetVf32(&state_, 1, 1e30f);                     // >> INT32_MAX
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(uint32_t{0x7FFFFFFFu}));  // INT32_MAX
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsXFromDNegOverflow) {
  static const uint32_t code[] = {0x9e780020u};  // fcvtzs x0, d1
  SetVf64(&state_, 1, -1e30);                     // << INT64_MIN
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(INT64_MIN));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuWFromS) {
  static const uint32_t code[] = {0x1e390020u};  // fcvtzu w0, s1
  SetVf32(&state_, 1, 100.9f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 100u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuWFromSNeg) {
  static const uint32_t code[] = {0x1e390020u};  // fcvtzu w0, s1
  SetVf32(&state_, 1, -5.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);  // negative -> 0
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuWFromSNan) {
  static const uint32_t code[] = {0x1e390020u};  // fcvtzu w0, s1
  SetV128(&state_, 1, 0x7FC00000ULL, 0ULL);      // lane 0 = FP32 qNaN
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);  // NaN -> 0
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuWFromDPosOverflow) {
  static const uint32_t code[] = {0x1e790020u};  // fcvtzu w0, d1
  SetVf64(&state_, 1, 5e9);                       // > UINT32_MAX (~4.29e9)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFu});  // UINT32_MAX
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuXFromDDirect) {
  static const uint32_t code[] = {0x9e790020u};  // fcvtzu x0, d1
  SetVf64(&state_, 1, 1000.0);                    // < 2^63, direct Q-convert
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 1000u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuXFromDInRange) {
  static const uint32_t code[] = {0x9e790020u};  // fcvtzu x0, d1
  SetVf64(&state_, 1, 1.0e19);                    // in [2^63, 2^64): offset trick
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(1.0e19));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuXFromDSatMax) {
  static const uint32_t code[] = {0x9e790020u};  // fcvtzu x0, d1
  SetVf64(&state_, 1, 2.0e19);                    // >= 2^64: saturate
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0xFFFFFFFFFFFFFFFFULL);  // UINT64_MAX
}

// Scalar rounding FP->int conversions (FCVTNS/PS/MS + unsigned). Encoding:
// sf 0 0 11110 type 1 rmode opcode 000000 Rn Rd; rmode = 00 (RNE) / 01 (+inf)
// / 10 (-inf), opcode = 000 (signed) / 001 (unsigned).
constexpr uint32_t FcvtScalar(bool sf,
                              uint8_t type,
                              uint8_t rmode,
                              uint8_t opcode,
                              uint8_t rn,
                              uint8_t rd) {
  return 0x1e200000u | (static_cast<uint32_t>(sf) << 31) | (static_cast<uint32_t>(type) << 22) |
         (static_cast<uint32_t>(rmode) << 19) | (static_cast<uint32_t>(opcode) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
static_assert(FcvtScalar(false, 0b00, 0b00, 0b000, 1, 0) == 0x1e200020u);  // fcvtns w0, s1
static_assert(FcvtScalar(true, 0b01, 0b01, 0b000, 1, 0) == 0x9e680020u);   // fcvtps x0, d1
static_assert(FcvtScalar(false, 0b00, 0b10, 0b000, 1, 0) == 0x1e300020u);  // fcvtms w0, s1
static_assert(FcvtScalar(false, 0b00, 0b00, 0b001, 1, 0) == 0x1e210020u);  // fcvtnu w0, s1
static_assert(FcvtScalar(false, 0b00, 0b01, 0b001, 1, 0) == 0x1e290020u);  // fcvtpu w0, s1
static_assert(FcvtScalar(true, 0b01, 0b10, 0b001, 1, 0) == 0x9e710020u);   // fcvtmu x0, d1
static_assert(FcvtScalar(false, 0b00, 0b00, 0b100, 1, 0) == 0x1e240020u);  // fcvtas w0, s1
static_assert(FcvtScalar(false, 0b00, 0b00, 0b101, 1, 0) == 0x1e250020u);  // fcvtau w0, s1

// FMOV between GP and V.D[1] (top half). ftype=10, rmode=01, op=110 (to GP) /
// op=111 (from GP). Assembler-verified.
constexpr uint32_t FmovXFromVd1(uint8_t rd, uint8_t rn) {  // fmov Xd, Vn.D[1]
  return FcvtScalar(true, 0b10, 0b01, 0b110, rn, rd);
}
constexpr uint32_t FmovVd1FromX(uint8_t rd, uint8_t rn) {  // fmov Vd.D[1], Xn
  return FcvtScalar(true, 0b10, 0b01, 0b111, rn, rd);
}
static_assert(FmovXFromVd1(0, 1) == 0x9eae0020u);   // fmov x0, v1.d[1]
static_assert(FmovXFromVd1(3, 4) == 0x9eae0083u);   // fmov x3, v4.d[1]
static_assert(FmovVd1FromX(0, 1) == 0x9eaf0020u);   // fmov v0.d[1], x1
static_assert(FmovVd1FromX(0, 2) == 0x9eaf0040u);   // fmov v0.d[1], x2

// BFCVT Hd, Sn (FP32 -> BF16). Assembler-verified.
constexpr uint32_t Bfcvt(uint8_t rd, uint8_t rn) { return FpDP1(0x1E634000, rd, rn); }
static_assert(Bfcvt(0, 1) == 0x1e634020u);
static_assert(Bfcvt(5, 6) == 0x1e6340c5u);

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtnsWFromS) {
  static const uint32_t code[] = {0x1e200020u};  // fcvtns w0, s1 (round-to-nearest ties-even)
  SetVf32(&state_, 1, 2.5f);                      // ties to even -> 2
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 2u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtnsWFromSTiesUp) {
  static const uint32_t code[] = {0x1e200020u};  // fcvtns w0, s1
  SetVf32(&state_, 1, 3.5f);                      // ties to even -> 4
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 4u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtpsXFromD) {
  static const uint32_t code[] = {0x9e680020u};  // fcvtps x0, d1 (toward +inf)
  SetVf64(&state_, 1, 12.1);                       // ceil -> 13
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 13u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtpsXFromDNeg) {
  static const uint32_t code[] = {0x9e680020u};  // fcvtps x0, d1
  SetVf64(&state_, 1, -12.9);                      // ceil(-12.9) -> -12
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(int64_t{-12}));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtmsWFromS) {
  static const uint32_t code[] = {0x1e300020u};  // fcvtms w0, s1 (toward -inf)
  SetVf32(&state_, 1, -12.1f);                     // floor -> -13
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(uint32_t{static_cast<uint32_t>(-13)}));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtpuWFromS) {
  static const uint32_t code[] = {0x1e290020u};  // fcvtpu w0, s1 (toward +inf, unsigned)
  SetVf32(&state_, 1, 12.1f);                      // ceil -> 13
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 13u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtmuWFromSNegSat) {
  static const uint32_t code[] = {FcvtScalar(false, 0b00, 0b10, 0b001, 1, 0)};  // fcvtmu w0, s1
  SetVf32(&state_, 1, -0.5f);                      // floor(-0.5) = -1 -> unsigned saturate to 0
  state_.cpu.x[0] = 0xdeadbeefULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtnsWFromSNan) {
  static const uint32_t code[] = {0x1e200020u};  // fcvtns w0, s1
  SetV128(&state_, 1, 0x7FC00000ULL, 0ULL);      // lane 0 = FP32 qNaN
  state_.cpu.x[0] = 0xdeadbeefULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);  // NaN -> 0 (saturation ladder still fires after ROUND)
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtpsWFromSPosOverflow) {
  static const uint32_t code[] = {FcvtScalar(false, 0b00, 0b01, 0b000, 1, 0)};  // fcvtps w0, s1
  SetVf32(&state_, 1, 1e30f);                     // >> INT32_MAX
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(uint32_t{0x7FFFFFFFu}));  // INT32_MAX
}

// FCVTAS/FCVTAU: round-to-nearest, ties AWAY from zero (distinct from fcvtns'
// ties-to-even). x86 has no ties-away mode; the heavy path adds copysign(0.5,x)
// gated off at |x| >= 2^23, then truncates.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasWFromSTieAway) {
  static const uint32_t code[] = {0x1e240020u};  // fcvtas w0, s1
  SetVf32(&state_, 1, 2.5f);                      // ties away -> 3 (fcvtns would give 2)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 3u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasWFromSNegTieAway) {
  static const uint32_t code[] = {0x1e240020u};  // fcvtas w0, s1
  SetVf32(&state_, 1, -2.5f);                     // ties away -> -3 (W-write zero-extends)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(uint32_t{static_cast<uint32_t>(-3)}));
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasWFromSHalf) {
  static const uint32_t code[] = {0x1e240020u};  // fcvtas w0, s1
  SetVf32(&state_, 1, 0.5f);                      // ties away -> 1
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 1u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtauWFromSTieAway) {
  static const uint32_t code[] = {0x1e250020u};  // fcvtau w0, s1
  SetVf32(&state_, 1, 2.5f);                      // ties away -> 3
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 3u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtauWFromSNegSat) {
  static const uint32_t code[] = {0x1e250020u};  // fcvtau w0, s1
  SetVf32(&state_, 1, -0.5f);                     // ties away -> -1 -> unsigned saturate to 0
  state_.cpu.x[0] = 0xdeadbeefULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasWFromSNan) {
  static const uint32_t code[] = {0x1e240020u};  // fcvtas w0, s1
  SetV128(&state_, 1, 0x7FC00000ULL, 0ULL);      // lane 0 = FP32 qNaN
  state_.cpu.x[0] = 0xdeadbeefULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);  // NaN -> 0 (addend gated off, saturation ladder fires)
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasWFromSLargeGated) {
  static const uint32_t code[] = {0x1e240020u};  // fcvtas w0, s1
  SetVf32(&state_, 1, 8388609.0f);               // 2^23+1: already integer, addend gated -> 8388609
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 8388609u);
}

// --- FCVTAS/FCVTAU FP64 (ties-away, GP dest) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasWFromDTieAway) {
  static const uint32_t code[] = {0x1e640020u};  // fcvtas w0, d1
  SetVf64(&state_, 1, 2.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 3u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasWFromDHalf) {
  static const uint32_t code[] = {0x1e640020u};  // fcvtas w0, d1
  SetVf64(&state_, 1, 0.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 1u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasXFromDOnePointFive) {
  static const uint32_t code[] = {0x9e640020u};  // fcvtas x0, d1
  SetVf64(&state_, 1, 1.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 2u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasXFromDNegHalf) {
  static const uint32_t code[] = {0x9e640020u};  // fcvtas x0, d1
  SetVf64(&state_, 1, -0.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(int64_t{-1}));
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasWFromDNegTwoPointFive) {
  static const uint32_t code[] = {0x1e640020u};  // fcvtas w0, d1
  SetVf64(&state_, 1, -2.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(uint32_t{static_cast<uint32_t>(-3)}));
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasXFromDLargeGated) {
  static const uint32_t code[] = {0x9e640020u};  // fcvtas x0, d1
  SetVf64(&state_, 1, 4503599627370497.0);         // 2^52 + 1 (already integer)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 4503599627370497ull);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasXFromDPosOverflow) {
  static const uint32_t code[] = {0x9e640020u};  // fcvtas x0, d1
  SetVf64(&state_, 1, 1e300);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(INT64_MAX));
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasXFromDNan) {
  static const uint32_t code[] = {0x9e640020u};  // fcvtas x0, d1
  SetV128(&state_, 1, 0x7FF8000000000000ull, 0ull);  // FP64 qNaN in lane 0
  state_.cpu.x[0] = 0xdeadbeefull;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtauXFromDTieAway) {
  static const uint32_t code[] = {0x9e650020u};  // fcvtau x0, d1
  SetVf64(&state_, 1, 2.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 3u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtauXFromDNegSat) {
  static const uint32_t code[] = {0x9e650020u};  // fcvtau x0, d1
  SetVf64(&state_, 1, -0.5);
  state_.cpu.x[0] = 0xdeadbeefull;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtauXFromDPosOverflow) {
  static const uint32_t code[] = {0x9e650020u};  // fcvtau x0, d1
  SetVf64(&state_, 1, 1e300);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], ~uint64_t{0});
}

// --- FMOV V.D[1] top-half moves ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovXFromVd1) {
  static const uint32_t code[] = {0x9eae0020u};  // fmov x0, v1.d[1]
  SetV128(&state_, 1, 0x1111111111111111ull, 0xCAFEF00DDEADBEEFull);
  state_.cpu.x[0] = 0xdeadbeefull;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0xCAFEF00DDEADBEEFull);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovVd1FromXPreservesLow) {
  static const uint32_t code[] = {0x9eaf0020u};  // fmov v0.d[1], x1
  SetV128(&state_, 0, 0xABCDEF0123456789ull, 0x2222222222222222ull);
  state_.cpu.x[1] = 0x1122334455667788ull;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xABCDEF0123456789ull);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x1122334455667788ull);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovVd1FromXzr) {
  static const uint32_t code[] = {FmovVd1FromX(0, 31)};  // fmov v0.d[1], xzr
  SetV128(&state_, 0, 0xABCDEF0123456789ull, 0x7777777777777777ull);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xABCDEF0123456789ull);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ull);
}

// --- BFCVT scalar (FP32 -> BF16) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, BfcvtOne) {
  static const uint32_t code[] = {0x1e634020u};  // bfcvt h0, s1
  SetV128(&state_, 1, 0x000000003F800000ull, 0xAAAAAAAAAAAAAAAAull);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3F80ull);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ull);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, BfcvtTieToEvenStays) {
  static const uint32_t code[] = {0x1e634020u};  // bfcvt h0, s1
  SetV128(&state_, 1, 0x000000003F808000ull, 0ull);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3F80ull);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, BfcvtTieToEvenUp) {
  static const uint32_t code[] = {0x1e634020u};  // bfcvt h0, s1
  SetV128(&state_, 1, 0x000000003F818000ull, 0ull);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3F82ull);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, BfcvtRoundUpNonTie) {
  static const uint32_t code[] = {0x1e634020u};  // bfcvt h0, s1
  SetV128(&state_, 1, 0x000000003F808001ull, 0ull);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3F81ull);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, BfcvtSnanQuieted) {
  static const uint32_t code[] = {0x1e634020u};  // bfcvt h0, s1
  SetV128(&state_, 1, 0x000000007F800001ull, 0ull);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FC0ull);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, BfcvtQnan) {
  static const uint32_t code[] = {0x1e634020u};  // bfcvt h0, s1
  SetV128(&state_, 1, 0x000000007FC00000ull, 0ull);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FC0ull);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuXFromSInRange) {
  static const uint32_t code[] = {0x9e390020u};  // fcvtzu x0, s1
  SetVf32(&state_, 1, 1.0e19f);                   // in [2^63, 2^64): offset trick
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(1.0e19f));
}

// ---------------------------------------------------------------------------
// SCVTF / UCVTF / FCVTZS / FCVTZU (scalar FP <-> fixed-point). fbits scales the
// result by 2^±fbits (SCVTF/UCVTF divide the FP by 2^fbits; FCVTZS/FCVTZU
// pre-multiply the FP source by 2^fbits before the truncating cvtt). Each chosen
// value is a discriminator: an unscaled (broken) lowering yields the wrong
// answer (e.g. scvtf s0,w1,#4 with w1=40 must be 2.5, not 40.0).
// ---------------------------------------------------------------------------

TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfFixedSFromW) {
  static const uint32_t code[] = {0x1e02f020u};  // scvtf s0, w1, #4
  state_.cpu.x[1] = 40u;                          // 40 / 2^4 = 2.5
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 2.5f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfFixedDFromX) {
  static const uint32_t code[] = {0x9e43e020u};  // ucvtf d0, x1, #8
  state_.cpu.x[1] = 384u;                         // 384 / 2^8 = 1.5
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 1.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfFixedDFromW) {
  static const uint32_t code[] = {0x1e42f420u};  // scvtf d0, w1, #3
  state_.cpu.x[1] = 20u;                          // 20 / 2^3 = 2.5
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 2.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsFixedWFromS) {
  static const uint32_t code[] = {0x1e18f020u};  // fcvtzs w0, s1, #4
  SetVf32(&state_, 1, 1.5f);                      // trunc(1.5 * 2^4) = 24
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 24u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuFixedXFromD) {
  static const uint32_t code[] = {0x9e59e020u};  // fcvtzu x0, d1, #8
  SetVf64(&state_, 1, 2.5);                       // trunc(2.5 * 2^8) = 640
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 640u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsFixedXFromDNeg) {
  static const uint32_t code[] = {0x9e58f820u};  // fcvtzs x0, d1, #2
  SetVf64(&state_, 1, -3.25);                     // trunc(-3.25 * 2^2) = -13
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], static_cast<uint64_t>(int64_t{-13}));
}

// ---------------------------------------------------------------------------
// FMADD / FMSUB / FNMADD / FNMSUB (FP data-processing, 3 source). All computed
// via the fused x86 FMA3 231-form ops (single rounding, matching ARM).
// ---------------------------------------------------------------------------

TEST_F(Arm64HeavyOptimizerFrontendTest, FmaddS) {
  static const uint32_t code[] = {0x1f020c20u};  // fmadd s0, s1, s2, s3
  SetVf32(&state_, 1, 2.0f);
  SetVf32(&state_, 2, 3.0f);
  SetVf32(&state_, 3, 10.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 16.0f);  // 10 + 2*3
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmaddD) {
  static const uint32_t code[] = {0x1f420c20u};  // fmadd d0, d1, d2, d3
  SetVf64(&state_, 1, 2.0);
  SetVf64(&state_, 2, 3.0);
  SetVf64(&state_, 3, 10.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 16.0);  // 10 + 2*3
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmsubS) {
  static const uint32_t code[] = {0x1f028c20u};  // fmsub s0, s1, s2, s3
  SetVf32(&state_, 1, 2.0f);
  SetVf32(&state_, 2, 3.0f);
  SetVf32(&state_, 3, 10.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 4.0f);  // 10 - 2*3
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FnmaddD) {
  static const uint32_t code[] = {0x1f620c20u};  // fnmadd d0, d1, d2, d3
  SetVf64(&state_, 1, 2.0);
  SetVf64(&state_, 2, 3.0);
  SetVf64(&state_, 3, 10.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), -16.0);  // -(10 + 2*3)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FnmsubS) {
  static const uint32_t code[] = {0x1f228c20u};  // fnmsub s0, s1, s2, s3
  SetVf32(&state_, 1, 2.0f);
  SetVf32(&state_, 2, 3.0f);
  SetVf32(&state_, 3, 10.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), -4.0f);  // 2*3 - 10
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// Multi-instruction FP region: chained FADD/FMUL across S and D, with an FMOV
// reg in the middle, all in one JIT region. Exercises that scalar V-reg
// read/write + zeroing compose correctly across several ops.
TEST_F(Arm64HeavyOptimizerFrontendTest, FpMultiInstructionRegion) {
  static const uint32_t code[] = {
      FaddS(0, 1, 2),   // S0 = S1 + S2 = 1.0 + 2.0 = 3.0
      FmulS(0, 0, 3),   // S0 = S0 * S3 = 3.0 * 4.0 = 12.0
      FmovRegS(4, 0),   // S4 = S0 = 12.0
      FaddD(5, 6, 7),   // D5 = D6 + D7 = 10.0 + 0.5 = 10.5
  };
  SetVf32(&state_, 1, 1.0f);
  SetVf32(&state_, 2, 2.0f);
  SetVf32(&state_, 3, 4.0f);
  SetVf64(&state_, 6, 10.0);
  SetVf64(&state_, 7, 0.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 12.0f);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 4), 12.0f);
  EXPECT_EQ(VUpperHi64(&state_, 4), 0u);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 5), 10.5);
  EXPECT_EQ(VUpperHi64(&state_, 5), 0u);
}

// FMAX (a 2-source FP op the optimizing tier does NOT handle) must bail: the
// region translates 0 instructions for a lone FMAX.
// FMAX/FMIN (NaN-propagating) and FMAXNM/FMINNM (NaN-suppressing) scalar S/D.
// Common cases where ARM semantics and the x86 idiom agree are asserted; the
// +-0 sign corner is intentionally not pinned (it matches the lite tier's
// MAX|MAX|POR idiom, whatever that yields, keeping the two tiers consistent).

TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxS) {
  static const uint32_t code[] = {FmaxS(0, 1, 2)};
  SetVf32(&state_, 1, 1.0f);
  SetVf32(&state_, 2, 2.0f);
  SetVf32(&state_, 0, 99.0f);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 2.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxD) {
  static const uint32_t code[] = {FmaxD(0, 1, 2)};
  SetVf64(&state_, 1, 1.5);
  SetVf64(&state_, 2, -3.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 1.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FminS) {
  static const uint32_t code[] = {FminS(0, 1, 2)};
  SetVf32(&state_, 1, 1.0f);
  SetVf32(&state_, 2, 2.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 1.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FminD) {
  static const uint32_t code[] = {FminD(0, 1, 2)};
  SetVf64(&state_, 1, 1.5);
  SetVf64(&state_, 2, -3.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), -3.0);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMAX propagates a NaN operand (ARM IEEE 754-2008 maxNum-less semantics).
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxNaNPropagatesS) {
  static const uint32_t code[] = {FmaxS(0, 1, 2)};
  SetV128(&state_, 1, 0x7FC00000u, 0);  // QNaN in lane 0
  SetVf32(&state_, 2, 3.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(static_cast<uint32_t>(VLo64(&state_, 0)), 0x7FC00000u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmS) {
  static const uint32_t code[] = {FmaxnmS(0, 1, 2)};
  SetVf32(&state_, 1, 1.0f);
  SetVf32(&state_, 2, 2.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 2.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMAXNM suppresses a single NaN and returns the number.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmNaNSuppressesS) {
  static const uint32_t code[] = {FmaxnmS(0, 1, 2)};
  SetV128(&state_, 1, 0x7FC00000u, 0);  // QNaN in lane 0
  SetVf32(&state_, 2, 3.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 3.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmNaNSuppressesD) {
  static const uint32_t code[] = {FmaxnmD(0, 1, 2)};
  SetV128(&state_, 1, 0x7FF8000000000000ull, 0);  // QNaN in lane 0
  SetVf64(&state_, 2, 2.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 2.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmS) {
  static const uint32_t code[] = {FminnmS(0, 1, 2)};
  SetVf32(&state_, 1, 4.0f);
  SetVf32(&state_, 2, 7.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 4.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmNaNSuppressesD) {
  static const uint32_t code[] = {FminnmD(0, 1, 2)};
  SetV128(&state_, 1, 0x7FF8000000000000ull, 0);  // QNaN in lane 0
  SetVf64(&state_, 2, -5.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), -5.0);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// --- Vector FMAX/FMIN/FMAXNM/FMINNM (AdvSimdThreeSame) heavy mirror. ---

// FMAX v0.4s, v1.4s, v2.4s: per-lane max of four FP32 lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxVec4S) {
  static const uint32_t code[] = {FmaxVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  // v1 lanes 0..3 = 1.0, -2.0, 3.5, -10.0 ; v2 = 2.0, -3.0, 3.0, 5.0.
  SetV128(&state_, 1, 0xC00000003F800000ULL, 0xC120000040600000ULL);
  SetV128(&state_, 2, 0xC040000040000000ULL, 0x40A0000040400000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FMAX -> 2.0, -2.0, 3.5, 5.0.
  EXPECT_EQ(VLo64(&state_, 0), 0xC000000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x40A0000040600000ULL);
}

// FMIN v0.4s, v1.4s, v2.4s: per-lane min of the same four FP32 lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminVec4S) {
  static const uint32_t code[] = {FminVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xC00000003F800000ULL, 0xC120000040600000ULL);
  SetV128(&state_, 2, 0xC040000040000000ULL, 0x40A0000040400000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FMIN -> 1.0, -3.0, 3.0, -10.0.
  EXPECT_EQ(VLo64(&state_, 0), 0xC04000003F800000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC120000040400000ULL);
}

// FMAX v0.2s, v1.2s, v2.2s (Q=0): only two lanes, Vd[127:64] zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxVec2S) {
  static const uint32_t code[] = {FmaxVec(/*dbl=*/false, /*q=*/false, 0, 1, 2)};
  // v1 lanes 0,1 = 1.0, -5.0 ; v2 = 2.0, -2.0.
  SetV128(&state_, 1, 0xC0A000003F800000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0xC000000040000000ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FMAX -> 2.0, -2.0 ; upper 64 zeroed.
  EXPECT_EQ(VLo64(&state_, 0), 0xC000000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMAX v0.2d, v1.2d, v2.2d: per-lane max of two FP64 lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxVec2D) {
  static const uint32_t code[] = {FmaxVec(/*dbl=*/true, /*q=*/true, 0, 1, 2)};
  // v1 = 1.5, -3.0 ; v2 = 2.5, -10.0.
  SetV128(&state_, 1, 0x3FF8000000000000ULL, 0xC008000000000000ULL);
  SetV128(&state_, 2, 0x4004000000000000ULL, 0xC024000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FMAX -> 2.5, -3.0.
  EXPECT_EQ(VLo64(&state_, 0), 0x4004000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC008000000000000ULL);
}

// FADD v0.4s, v1.4s, v2.4s: per-lane FP32 add.
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddVec4S) {
  static const uint32_t code[] = {FaddVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  // v1 = 1.0, -2.0, 3.5, -10.0 ; v2 = 2.0, -3.0, 3.0, 5.0.
  SetV128(&state_, 1, 0xC00000003F800000ULL, 0xC120000040600000ULL);
  SetV128(&state_, 2, 0xC040000040000000ULL, 0x40A0000040400000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FADD -> 3.0, -5.0, 6.5, -5.0.
  EXPECT_EQ(VLo64(&state_, 0), 0xC0A0000040400000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC0A0000040D00000ULL);
}

// FSUB v0.4s, v1.4s, v2.4s: per-lane FP32 subtract.
TEST_F(Arm64HeavyOptimizerFrontendTest, FsubVec4S) {
  static const uint32_t code[] = {FsubVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xC00000003F800000ULL, 0xC120000040600000ULL);
  SetV128(&state_, 2, 0xC040000040000000ULL, 0x40A0000040400000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FSUB -> -1.0, 1.0, 0.5, -15.0.
  EXPECT_EQ(VLo64(&state_, 0), 0x3F800000BF800000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC17000003F000000ULL);
}

// FMUL v0.4s, v1.4s, v2.4s (U=1): per-lane FP32 multiply.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulVec4S) {
  static const uint32_t code[] = {FmulVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xC00000003F800000ULL, 0xC120000040600000ULL);
  SetV128(&state_, 2, 0xC040000040000000ULL, 0x40A0000040400000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FMUL -> 2.0, 6.0, 10.5, -50.0.
  EXPECT_EQ(VLo64(&state_, 0), 0x40C0000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC248000041280000ULL);
}

// FDIV v0.2d, v1.2d, v2.2d (U=1, FP64): per-lane double divide.
TEST_F(Arm64HeavyOptimizerFrontendTest, FdivVec2D) {
  static const uint32_t code[] = {FdivVec(/*dbl=*/true, /*q=*/true, 0, 1, 2)};
  // v1 = 6.0, -8.0 ; v2 = 2.0, 4.0.
  SetV128(&state_, 1, 0x4018000000000000ULL, 0xC020000000000000ULL);
  SetV128(&state_, 2, 0x4000000000000000ULL, 0x4010000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FDIV -> 3.0, -2.0.
  EXPECT_EQ(VLo64(&state_, 0), 0x4008000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC000000000000000ULL);
}

// FADD v0.2s, v1.2s, v2.2s (Q=0): two FP32 lanes, Vd[127:64] zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddVec2S) {
  static const uint32_t code[] = {FaddVec(/*dbl=*/false, /*q=*/false, 0, 1, 2)};
  // v1 lanes 0,1 = 1.0, 2.0 ; v2 = 10.0, 20.0.
  SetV128(&state_, 1, 0x400000003F800000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x41A0000041200000ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FADD -> 11.0, 22.0 ; upper 64 zeroed.
  EXPECT_EQ(VLo64(&state_, 0), 0x41B0000041300000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// Shared FP32 .4S operands for the compare tests:
//   v1 = 1.0, 2.0, 3.0, NaN ; v2 = 1.0, 5.0, 2.0, 1.0.
// NaN lane exercises ARM's unordered-is-false rule (all three ops -> 0).
static constexpr uint64_t kCmpV1Lo = 0x400000003F800000ULL;  // 1.0, 2.0
static constexpr uint64_t kCmpV1Hi = 0x7FC0000040400000ULL;  // 3.0, NaN
static constexpr uint64_t kCmpV2Lo = 0x40A000003F800000ULL;  // 1.0, 5.0
static constexpr uint64_t kCmpV2Hi = 0x3F80000040000000ULL;  // 2.0, 1.0

// FCMEQ v0.4s, v1.4s, v2.4s: per-lane FP32 equal -> all-ones/zero mask.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmeqVec4S) {
  static const uint32_t code[] = {FcmeqVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, kCmpV1Lo, kCmpV1Hi);
  SetV128(&state_, 2, kCmpV2Lo, kCmpV2Hi);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // eq: 1==1 T, 2==5 F, 3==2 F, NaN==1 F.
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCMGE v0.4s, v1.4s, v2.4s (U=1): per-lane FP32 >= -> all-ones/zero mask.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgeVec4S) {
  static const uint32_t code[] = {FcmgeVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, kCmpV1Lo, kCmpV1Hi);
  SetV128(&state_, 2, kCmpV2Lo, kCmpV2Hi);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // ge: 1>=1 T, 2>=5 F, 3>=2 T, NaN>=1 F.
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFFULL);
}

// FCMGT v0.4s, v1.4s, v2.4s (op_high=1, U=1): per-lane FP32 > -> mask.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgtVec4S) {
  static const uint32_t code[] = {FcmgtVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, kCmpV1Lo, kCmpV1Hi);
  SetV128(&state_, 2, kCmpV2Lo, kCmpV2Hi);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // gt: 1>1 F, 2>5 F, 3>2 T, NaN>1 F.
  EXPECT_EQ(VLo64(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFFULL);
}

// FABD v0.4s, v1.4s, v2.4s (op_high=1, U=1): per-lane |v1 - v2| FP32.
TEST_F(Arm64HeavyOptimizerFrontendTest, FabdVec4S) {
  static const uint32_t code[] = {FabdVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  // v1 = 1.0, -2.0, 3.5, -10.0 ; v2 = 2.0, -3.0, 3.0, 5.0.
  SetV128(&state_, 1, 0xC00000003F800000ULL, 0xC120000040600000ULL);
  SetV128(&state_, 2, 0xC040000040000000ULL, 0x40A0000040400000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // |1-2|=1, |-2- -3|=1, |3.5-3|=0.5, |-10-5|=15.
  EXPECT_EQ(VLo64(&state_, 0), 0x3F8000003F800000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x417000003F000000ULL);
}

// FABD v0.2d, v1.2d, v2.2d (FP64): per-lane |v1 - v2|.
TEST_F(Arm64HeavyOptimizerFrontendTest, FabdVec2D) {
  static const uint32_t code[] = {FabdVec(/*dbl=*/true, /*q=*/true, 0, 1, 2)};
  // v1 = 6.0, -8.0 ; v2 = 2.0, 4.0.
  SetV128(&state_, 1, 0x4018000000000000ULL, 0xC020000000000000ULL);
  SetV128(&state_, 2, 0x4000000000000000ULL, 0x4010000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // |6-2|=4.0, |-8-4|=12.0.
  EXPECT_EQ(VLo64(&state_, 0), 0x4010000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4028000000000000ULL);
}

// FABD v0.2s, v1.2s, v2.2s (Q=0): two FP32 lanes, Vd[127:64] zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FabdVec2S) {
  static const uint32_t code[] = {FabdVec(/*dbl=*/false, /*q=*/false, 0, 1, 2)};
  // v1 lanes 0,1 = 1.0, 2.0 ; v2 = 10.0, 20.0.
  SetV128(&state_, 1, 0x400000003F800000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x41A0000041200000ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // |1-10|=9.0, |2-20|=18.0 ; upper 64 zeroed.
  EXPECT_EQ(VLo64(&state_, 0), 0x4190000041100000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// ARM FABD is FPAbs(FPSub); a NaN difference keeps its payload with the sign
// bit cleared (result high bit == 0).
TEST_F(Arm64HeavyOptimizerFrontendTest, FabdVec4SNaNSignCleared) {
  static const uint32_t code[] = {FabdVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  // v1 lane0 = -qNaN (0xFFC00000), lane1 = 3.0 ; v2 lane0 = 1.0, lane1 = 9.0.
  SetV128(&state_, 1, 0x40400000FFC00000ULL, 0ULL);
  SetV128(&state_, 2, 0x411000003F800000ULL, 0ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lane0 = |NaN - 1| -> 0x7FC00000 (sign cleared) ; lane1 = |3-9| = 6.0.
  EXPECT_EQ(VLo64(&state_, 0), 0x40C000007FC00000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FABS v0.4s: clear the sign bit of each FP32 lane.
TEST_F(Arm64HeavyOptimizerFrontendTest, FabsVec4S) {
  static const uint32_t code[] = {FabsVec(/*dbl=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xC00000003F800000ULL, 0x40800000C0600000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x400000003F800000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4080000040600000ULL);
}

// FABS v0.2d: clear the sign bit of each FP64 lane.
TEST_F(Arm64HeavyOptimizerFrontendTest, FabsVec2D) {
  static const uint32_t code[] = {FabsVec(/*dbl=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xBFF8000000000000ULL, 0x4004000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3FF8000000000000ULL);  // 1.5
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4004000000000000ULL);  // 2.5
}

// FNEG v0.4s: flip the sign bit of each FP32 lane.
TEST_F(Arm64HeavyOptimizerFrontendTest, FnegVec4S) {
  static const uint32_t code[] = {FnegVec(/*dbl=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xC00000003F800000ULL, 0xC080000040600000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x40000000BF800000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x40800000C0600000ULL);
}

// FNEG v0.2d.
TEST_F(Arm64HeavyOptimizerFrontendTest, FnegVec2D) {
  static const uint32_t code[] = {FnegVec(/*dbl=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x3FF8000000000000ULL, 0xC004000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xBFF8000000000000ULL);  // -1.5
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4004000000000000ULL);  // 2.5
}

// FSQRT v0.4s: per-lane sqrt (perfect squares).
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtVec4S) {
  static const uint32_t code[] = {FsqrtVec(/*dbl=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4110000040800000ULL, 0x41C8000041800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4040000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x40A0000040800000ULL);
}

// FSQRT v0.2d.
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtVec2D) {
  static const uint32_t code[] = {FsqrtVec(/*dbl=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4010000000000000ULL, 0x4022000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4000000000000000ULL);  // 2.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4008000000000000ULL);  // 3.0
}

// FSQRT v0.2s (Q=0): two FP32 lanes; upper 64 zeroed by SetVRegFull.
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtVec2S) {
  static const uint32_t code[] = {FsqrtVec(/*dbl=*/false, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x4110000040800000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4040000040000000ULL);  // {2.0, 3.0}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FACGE v0.4s: |v1| >= |v2| per lane -> all-ones/zero mask.
TEST_F(Arm64HeavyOptimizerFrontendTest, FacgeVec4S) {
  static const uint32_t code[] = {FacgeVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x40000000C0400000ULL, 0x40A00000BF800000ULL);
  SetV128(&state_, 2, 0xC000000040000000ULL, 0xC0A0000040800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);  // {T,T}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFF00000000ULL);  // {F,T}
}

// FACGT v0.4s: |v1| > |v2| (same inputs) -> {3>2 T, 2>2 F, 1>4 F, 5>5 F}.
TEST_F(Arm64HeavyOptimizerFrontendTest, FacgtVec4S) {
  static const uint32_t code[] = {FacgtVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x40000000C0400000ULL, 0x40A00000BF800000ULL);
  SetV128(&state_, 2, 0xC000000040000000ULL, 0xC0A0000040800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);  // {T,F}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);  // {F,F}
}

// FACGE v0.2d: |v1| >= |v2| FP64.
TEST_F(Arm64HeavyOptimizerFrontendTest, FacgeVec2D) {
  static const uint32_t code[] = {FacgeVec(/*dbl=*/true, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xC008000000000000ULL, 0x4000000000000000ULL);
  SetV128(&state_, 2, 0x4000000000000000ULL, 0xC010000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCVTL v0.2d, v1.2s (Q=0): widen the low 2 FP32 -> 2 FP64 (full 16-byte result).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtlVec2D) {
  static const uint32_t code[] = {FcvtlVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xC02000003FC00000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3FF8000000000000ULL);  // 1.5d
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC004000000000000ULL);  // -2.5d
}

// FCVTL2 v0.2d, v1.4s (Q=1): widen the high 2 FP32 -> 2 FP64.
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcvtl2Vec2D) {
  static const uint32_t code[] = {FcvtlVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x2222222222222222ULL, 0xBF00000040400000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4008000000000000ULL);  // 3.0d
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBFE0000000000000ULL);  // -0.5d
}

// FCVTN v0.2s, v1.2d (Q=0): narrow 2 FP64 -> 2 FP32 in low 64, upper zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtnVec2S) {
  static const uint32_t code[] = {FcvtnVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x3FF8000000000000ULL, 0xC004000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC02000003FC00000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCVTN2 v0.4s, v1.2d (Q=1): narrow into Vd's high 64, preserving the low 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcvtn2Vec4S) {
  static const uint32_t code[] = {FcvtnVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4008000000000000ULL, 0xBFE0000000000000ULL);
  SetV128(&state_, 0, 0x1111111122222222ULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111122222222ULL);  // preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBF00000040400000ULL);
}

// FCVTXN v0.2s, v1.2d (Q=0): round-to-odd narrow of 2 doubles. lane0 = 1.5 is exactly
// representable -> 0x3FC00000 with NO odd-forcing (proves exact lanes pass through);
// lane1 = 1.0 + 2^-25 (0x3FF0000008000000) is inexact: RNE would give 0x3F800000
// (even LSB), round-to-odd forces bit0 -> 0x3F800001. Upper 64 of Vd is zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtxnVec2S) {
  static const uint32_t code[] = {FcvtxnVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x3FF8000000000000ULL, 0x3FF0000008000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3F8000013FC00000ULL);  // {0x3FC00000, 0x3F800001}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCVTXN Q=0, overflow and NaN. lane0 = 1e300 (0x7E37E43C8800759C) overflows FP32:
// round-to-odd clamps toward zero to +FP32_MAX 0x7F7FFFFF. lane1 = qNaN
// (0x7FF8000000000000) -> default qNaN 0x7FC00000 (NOT odd-forced).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtxnVecOverflowNaN) {
  static const uint32_t code[] = {FcvtxnVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x7E37E43C8800759CULL, 0x7FF8000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FC000007F7FFFFFULL);  // {0x7F7FFFFF, 0x7FC00000}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCVTXN Q=0, negative lanes: sign must be preserved. lane0 = -(1.0 + 2^-25)
// (0xBFF0000008000000) inexact -> 0xBF800001 (odd-forced, sign kept); lane1 = -1e300
// (0xFE37E43C8800759C) overflow -> -FP32_MAX 0xFF7FFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtxnVecNegative) {
  static const uint32_t code[] = {FcvtxnVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xBFF0000008000000ULL, 0xFE37E43C8800759CULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF7FFFFFBF800001ULL);  // {0xBF800001, 0xFF7FFFFF}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCVTXN2 v0.4s, v1.2d (Q=1): narrow into Vd's HIGH 64, preserving the low 64.
// Same inputs as FcvtxnVec2S -> {0x3FC00000, 0x3F800001} land in the upper half.
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcvtxn2Vec4S) {
  static const uint32_t code[] = {FcvtxnVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x3FF8000000000000ULL, 0x3FF0000008000000ULL);
  SetV128(&state_, 0, 0x1111111122222222ULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111122222222ULL);          // preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x3F8000013FC00000ULL);     // {0x3FC00000, 0x3F800001}
}

// FRECPE v0.4s: 1.0/x per lane (full precision, matches interpreter).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrecpeVec4S) {
  static const uint32_t code[] = {FrecpeVec(/*dbl=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4080000040000000ULL, 0x3F000000C1000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // {0.5, 0.25, -0.125, 2.0} = {0x3F000000,0x3E800000,0xBE000000,0x40000000}.
  EXPECT_EQ(VLo64(&state_, 0), 0x3E8000003F000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x40000000BE000000ULL);
}

// FRSQRTE v0.4s: 1/sqrt(x), with (x<=0 || NaN) -> default qNaN (0x7FC00000).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrsqrteVec4S) {
  static const uint32_t code[] = {FrsqrteVec(/*dbl=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4180000040800000ULL, 0x3E800000BF800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // {0.5, 0.25, qNaN, 2.0} = {0x3F000000,0x3E800000,0x7FC00000,0x40000000}.
  EXPECT_EQ(VLo64(&state_, 0), 0x3E8000003F000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x400000007FC00000ULL);
}

// FRECPE v0.2d (incl. a negative lane: 1/-4 = -0.25).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrecpeVec2D) {
  static const uint32_t code[] = {FrecpeVec(/*dbl=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4000000000000000ULL, 0xC010000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3FE0000000000000ULL);   // 0.5
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBFD0000000000000ULL);  // -0.25
}

// FRSQRTE v0.2d: negative lane -> default qNaN (0x7FF8000000000000).
TEST_F(Arm64HeavyOptimizerFrontendTest, FrsqrteVec2DNaN) {
  static const uint32_t code[] = {FrsqrteVec(/*dbl=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xC000000000000000ULL, 0x4010000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FF8000000000000ULL);   // qNaN
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x3FE0000000000000ULL);  // 0.5
}

// SQABS v0.4s: per-lane saturating signed abs; INT_MIN -> INT_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqabsVec4S) {
  static const uint32_t code[] = {SqabsVec(/*size=*/0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFF900000005ULL, 0x0000000380000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000700000005ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000037FFFFFFFULL);
}

// SQNEG v0.4s: per-lane saturating signed negate; INT_MIN -> INT_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqnegVec4S) {
  static const uint32_t code[] = {SqnegVec(/*size=*/0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFF900000005ULL, 0x0000000380000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000007FFFFFFFBULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFD7FFFFFFFULL);
}

// SQABS v0.8b (Q=0): saturating byte abs; -128 -> 127; upper 64 zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqabsVec8B) {
  static const uint32_t code[] = {SqabsVec(/*size=*/0b00, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFE8005FF7F80FC03ULL, 0x1111111111111111ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x027F05017F7F0403ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCMGE v0.2s, v1.2s, v2.2s (Q=0): two FP32 lanes, Vd[127:64] zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgeVec2S) {
  static const uint32_t code[] = {FcmgeVec(/*dbl=*/false, /*q=*/false, 0, 1, 2)};
  // v1 = 3.0, 1.0 ; v2 = 2.0, 1.0.
  SetV128(&state_, 1, 0x3F80000040400000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x3F80000040000000ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // ge: 3>=2 T, 1>=1 T ; upper 64 zeroed.
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FCMGT v0.2d, v1.2d, v2.2d (op_high=1, U=1, FP64): per-lane double >.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgtVec2D) {
  static const uint32_t code[] = {FcmgtVec(/*dbl=*/true, /*q=*/true, 0, 1, 2)};
  // v1 = 5.0, 2.0 ; v2 = 3.0, 2.0.
  SetV128(&state_, 1, 0x4014000000000000ULL, 0x4000000000000000ULL);
  SetV128(&state_, 2, 0x4008000000000000ULL, 0x4000000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // gt: 5>3 T, 2>2 F.
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMINNM v0.2d, v1.2d, v2.2d: per-lane number-min of two FP64 lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmVec2D) {
  static const uint32_t code[] = {FminnmVec(/*dbl=*/true, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x3FF8000000000000ULL, 0xC008000000000000ULL);  // 1.5, -3.0
  SetV128(&state_, 2, 0x4004000000000000ULL, 0xC024000000000000ULL);  // 2.5, -10.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FMINNM -> 1.5, -10.0.
  EXPECT_EQ(VLo64(&state_, 0), 0x3FF8000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC024000000000000ULL);
}

// FMAXNM v0.4s: suppresses a single NaN lane and returns the other operand.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmVecNaNSuppress4S) {
  static const uint32_t code[] = {FmaxnmVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  // v1 lanes = qNaN, 5.0, 1.0, 2.0 ; v2 = 3.0, qNaN, 7.0, 1.0.
  SetV128(&state_, 1, 0x40A000007FC00000ULL, 0x400000003F800000ULL);
  SetV128(&state_, 2, 0x7FC0000040400000ULL, 0x3F80000040E00000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // FMAXNM -> 3.0, 5.0, 7.0, 2.0.
  EXPECT_EQ(VLo64(&state_, 0), 0x40A0000040400000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4000000040E00000ULL);
}

// FMAX v0.4s propagates a NaN lane (any NaN input -> NaN out) per ARM ARM.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxVecNaNPropagate4S) {
  static const uint32_t code[] = {FmaxVec(/*dbl=*/false, /*q=*/true, 0, 1, 2)};
  // v1 lane0 = qNaN, rest 1.0 ; v2 lane0 = 3.0, rest 2.0.
  SetV128(&state_, 1, 0x3F8000007FC00000ULL, 0x3F8000003F800000ULL);
  SetV128(&state_, 2, 0x4000000040400000ULL, 0x4000000040000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lane0 = NaN (0x7FC00000 after the double negation); lanes 1..3 = max(1,2)=2.
  EXPECT_EQ(static_cast<uint32_t>(VLo64(&state_, 0)), 0x7FC00000u);
  EXPECT_EQ(VLo64(&state_, 0) >> 32, 0x40000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4000000040000000ULL);
}

// FMAX signed-zero tie: ARM FMAX(+-0) = AND-of-signs -> +0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxVecZeroTie2S) {
  static const uint32_t code[] = {FmaxVec(/*dbl=*/false, /*q=*/false, 0, 1, 2)};
  // v1 lanes = +0.0, -0.0 ; v2 lanes = -0.0, +0.0.
  SetV128(&state_, 1, 0x8000000000000000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x0000000080000000ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // Both lanes -> +0.0 (0x00000000); upper 64 zeroed.
  EXPECT_EQ(VLo64(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMIN signed-zero tie: ARM FMIN(+-0) = OR-of-signs -> -0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminVecZeroTie2S) {
  static const uint32_t code[] = {FminVec(/*dbl=*/false, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0x1111111111111111ULL);  // +0.0, -0.0
  SetV128(&state_, 2, 0x0000000080000000ULL, 0x2222222222222222ULL);  // -0.0, +0.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // Both lanes -> -0.0 (0x80000000); upper 64 zeroed.
  EXPECT_EQ(VLo64(&state_, 0), 0x8000000080000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FNMUL Sd,Sn,Sm (opcode 1000): -(n * m). Positive product -> negated.
TEST_F(Arm64HeavyOptimizerFrontendTest, FnmulS) {
  static const uint32_t code[] = {FnmulS(0, 1, 2)};
  SetVf32(&state_, 1, 3.0f);
  SetVf32(&state_, 2, 4.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), -12.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FNMUL with one negative operand: -((-2)*3) = +6 (the sign flip re-negates a
// negative product back to positive), confirming it is a true sign-bit XOR.
TEST_F(Arm64HeavyOptimizerFrontendTest, FnmulSNegOperand) {
  static const uint32_t code[] = {FnmulS(0, 1, 2)};
  SetVf32(&state_, 1, -2.0f);
  SetVf32(&state_, 2, 3.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 6.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FnmulD) {
  static const uint32_t code[] = {FnmulD(0, 1, 2)};
  SetVf64(&state_, 1, 0.5);
  SetVf64(&state_, 2, 3.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), -1.5);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, FnmulDNegOperand) {
  static const uint32_t code[] = {FnmulD(0, 1, 2)};
  SetVf64(&state_, 1, -4.0);
  SetVf64(&state_, 2, 2.5);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 10.0);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FSQRT Sd,Sn: SQRTSS is the exact IEEE-754 square root (RNE), matching ARM
// FSQRT under the default FPCR rounding mode. sqrt(4.0f) == 2.0f exactly; the
// upper 96 bits of Vd must be zeroed by the scalar (upper-zeroing) store.
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtS) {
  static const uint32_t code[] = {FsqrtS(0, 1)};
  SetVf32(&state_, 1, 4.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 2.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FSQRT Sd,Sn non-perfect-square: sqrt(2.0f) rounds to nearest-even.
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtSInexact) {
  static const uint32_t code[] = {FsqrtS(0, 1)};
  SetVf32(&state_, 1, 2.0f);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 1.4142135f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FSQRT Dd,Dn: SQRTSD, FP64. sqrt(2.0) rounds to nearest-even.
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtD) {
  static const uint32_t code[] = {FsqrtD(0, 1)};
  SetVf64(&state_, 1, 2.0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 1.4142135623730951);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

//
// FCMP / FCMPE / FCCMP: compare two scalar FP regs (or #0.0) and set ARM NZCV
// in cpu.flags. Each test drives the full heavy pipeline (RunRegion) and asserts
// the four flag bits. NZCV layout: N@bit15, Z@bit14, C@bit8, V@bit0.
//

// FCMP/FCMPE: 0001_1110_ftype_1_Rm_00_1000_Rn_opcode2.
// opcode2[4:0]: FCMP=00000, FCMP#0=01000, FCMPE=10000, FCMPE#0=11000.
constexpr uint32_t FcmpS(uint8_t rn, uint8_t rm) {
  return 0x1E202000u | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5);
}
constexpr uint32_t FcmpD(uint8_t rn, uint8_t rm) {
  return 0x1E602000u | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5);
}
constexpr uint32_t FcmpZeroS(uint8_t rn) { return 0x1E202000u | (static_cast<uint32_t>(rn) << 5) | 0x08u; }
constexpr uint32_t FcmpZeroD(uint8_t rn) { return 0x1E602000u | (static_cast<uint32_t>(rn) << 5) | 0x08u; }
constexpr uint32_t FcmpeS(uint8_t rn, uint8_t rm) {
  return 0x1E202000u | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5) | 0x10u;
}
// FCMP Hn, Hm (ftype=0b11). Lowered in-tier via F16C widening when present.
constexpr uint32_t FcmpH(uint8_t rn, uint8_t rm) {
  return 0x1EE02000u | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(rn) << 5);
}

// FCCMP/FCCMPE: 0001_1110_ftype_1_Rm_cond_01_Rn_op_nzcv. op(bit4): 0=FCCMP,1=FCCMPE.
constexpr uint32_t FccmpS(uint8_t rn, uint8_t rm, uint8_t nzcv, uint8_t cond) {
  return 0x1E200400u | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(cond) << 12) |
         (static_cast<uint32_t>(rn) << 5) | nzcv;
}
constexpr uint32_t FccmpD(uint8_t rn, uint8_t rm, uint8_t nzcv, uint8_t cond) {
  return 0x1E600400u | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(cond) << 12) |
         (static_cast<uint32_t>(rn) << 5) | nzcv;
}

// FP16 (ftype=0b11) scalar forms: S-form base | 0xC00000.
constexpr uint32_t FmulH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE00800, rd, rn, rm); }
constexpr uint32_t FdivH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE01800, rd, rn, rm); }
constexpr uint32_t FaddH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE02800, rd, rn, rm); }
constexpr uint32_t FsubH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE03800, rd, rn, rm); }
constexpr uint32_t FmaxH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE04800, rd, rn, rm); }
constexpr uint32_t FminH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE05800, rd, rn, rm); }
constexpr uint32_t FmaxnmH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE06800, rd, rn, rm); }
constexpr uint32_t FminnmH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE07800, rd, rn, rm); }
constexpr uint32_t FnmulH(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1EE08800, rd, rn, rm); }
constexpr uint32_t FmovRegH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE04000, rd, rn); }
constexpr uint32_t FabsH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE0C000, rd, rn); }
constexpr uint32_t FnegH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE14000, rd, rn); }
constexpr uint32_t FsqrtH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE1C000, rd, rn); }
constexpr uint32_t FrintaH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE64000, rd, rn); }
constexpr uint32_t FrintnH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE44000, rd, rn); }
constexpr uint32_t FrintmH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE54000, rd, rn); }
constexpr uint32_t FrintpH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE4C000, rd, rn); }
constexpr uint32_t FrintzH(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE5C000, rd, rn); }
// FCVT: to-S=0x20000, to-D=0x28000, to-H=0x38000 (opcode<<15) added to the base.
constexpr uint32_t FcvtHtoS(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE24000, rd, rn); }  // fcvt s,h
constexpr uint32_t FcvtHtoD(uint8_t rd, uint8_t rn) { return FpDP1(0x1EE2C000, rd, rn); }  // fcvt d,h
constexpr uint32_t FcvtStoH(uint8_t rd, uint8_t rn) { return FpDP1(0x1E23C000, rd, rn); }  // fcvt h,s
constexpr uint32_t FcvtDtoH(uint8_t rd, uint8_t rn) { return FpDP1(0x1E63C000, rd, rn); }  // fcvt h,d
constexpr uint32_t FmovImmH(uint8_t rd, uint8_t imm8) {
  return 0x1EE01000u | (static_cast<uint32_t>(imm8) << 13) | rd;
}
static_assert(FaddH(0, 1, 2) == 0x1EE22820u);
static_assert(FsqrtH(0, 1) == 0x1EE1C020u);
static_assert(FcvtHtoS(0, 1) == 0x1EE24020u);
static_assert(FcvtStoH(0, 1) == 0x1E23C020u);
static_assert(FmovImmH(0, 0x70) == 0x1EEE1000u);  // fmov h0,#1.0
// FCCMP Hn,Hm: S-form base | 0xC00000.
constexpr uint32_t FccmpH(uint8_t rn, uint8_t rm, uint8_t nzcv, uint8_t cond) {
  return 0x1EE00400u | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(cond) << 12) |
         (static_cast<uint32_t>(rn) << 5) | nzcv;
}
static_assert(FccmpH(1, 2, 0, 0) == 0x1EE20420u);
// FCMP Hn,#0.0 (ftype=0b11).
constexpr uint32_t FcmpZeroH(uint8_t rn) {
  return 0x1EE02000u | (static_cast<uint32_t>(rn) << 5) | 0x08u;
}

// Write/read the 16-bit scalar half of a guest V register.
inline void SetVu16(ThreadState* s, unsigned reg, uint16_t h) {
  std::memset(&s->cpu.v[reg], 0xAB, sizeof(s->cpu.v[reg]));  // poison upper bytes
  std::memcpy(&s->cpu.v[reg], &h, sizeof(h));
}
inline uint16_t VHalf0(const ThreadState* s, unsigned reg) {
  uint16_t h;
  std::memcpy(&h, &s->cpu.v[reg], sizeof(h));
  return h;
}
inline uint32_t VWord0(const ThreadState* s, unsigned reg) {
  uint32_t w;
  std::memcpy(&w, &s->cpu.v[reg], sizeof(w));
  return w;
}
// A clean FP16 scalar result: bits[15:0]==expect, bits[127:16]==0.
#define EXPECT_HALF_CLEAN(s, reg, expect)                          \
  do {                                                             \
    EXPECT_EQ(VHalf0(&(s), (reg)), static_cast<uint16_t>(expect)); \
    EXPECT_EQ(VWord0(&(s), (reg)) >> 16, 0u);                      \
    EXPECT_EQ(VWord1(&(s), (reg)), 0u);                            \
    EXPECT_EQ(VUpperHi64(&(s), (reg)), 0u);                        \
  } while (0)

float MakeNanF() {
  uint32_t bits = 0x7FC00000u;
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

// ARM64 condition field values.
constexpr uint8_t kCondEq = 0;   // Z==1
constexpr uint8_t kCondNe = 1;   // Z==0
constexpr uint8_t kCondAl = 14;  // always

// Assert cpu.flags carries exactly the given NZCV bits.
void ExpectNZCV(const ThreadState& s, bool n, bool z, bool c, bool v) {
  EXPECT_EQ(static_cast<bool>(s.cpu.flags & CPUState::kFlagNegative), n);
  EXPECT_EQ(static_cast<bool>(s.cpu.flags & CPUState::kFlagZero), z);
  EXPECT_EQ(static_cast<bool>(s.cpu.flags & CPUState::kFlagCarry), c);
  EXPECT_EQ(static_cast<bool>(s.cpu.flags & CPUState::kFlagOverflow), v);
}

// FCMP greater (Sn > Sm): NZCV = 0,0,1,0 (C only).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpSGreater) {
  static const uint32_t code[] = {FcmpS(1, 2)};
  SetVf32(&state_, 1, 2.0f);
  SetVf32(&state_, 2, 1.0f);
  state_.cpu.flags = 0xFFFF;  // prove the store clears N/Z/V
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/false, /*c=*/true, /*v=*/false);
}

// FCMP less (Sn < Sm): NZCV = 1,0,0,0 (N only).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpSLess) {
  static const uint32_t code[] = {FcmpS(1, 2)};
  SetVf32(&state_, 1, 1.0f);
  SetVf32(&state_, 2, 2.0f);
  state_.cpu.flags = 0xFFFF;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/true, /*z=*/false, /*c=*/false, /*v=*/false);
}

// FCMP equal: NZCV = 0,1,1,0 (Z,C).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpSEqual) {
  static const uint32_t code[] = {FcmpS(1, 2)};
  SetVf32(&state_, 1, 2.5f);
  SetVf32(&state_, 2, 2.5f);
  state_.cpu.flags = 0;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/true, /*c=*/true, /*v=*/false);
}

// FCMP unordered (Sn is NaN): NZCV = 0,0,1,1 (C,V).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpSUnordered) {
  static const uint32_t code[] = {FcmpS(1, 2)};
  SetVf32(&state_, 1, MakeNanF());
  SetVf32(&state_, 2, 1.0f);
  state_.cpu.flags = 0;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/false, /*c=*/true, /*v=*/true);
}

// FCMP D-form, less: exercises the UCOMISD path.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpDLess) {
  static const uint32_t code[] = {FcmpD(3, 4)};
  SetVf64(&state_, 3, -5.0);
  SetVf64(&state_, 4, 5.0);
  state_.cpu.flags = 0xFFFF;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/true, /*z=*/false, /*c=*/false, /*v=*/false);
}

// FCMP Sn, #0.0 (with_zero): negative operand -> less -> N.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpZeroSNegative) {
  static const uint32_t code[] = {FcmpZeroS(5)};
  SetVf32(&state_, 5, -1.0f);
  state_.cpu.flags = 0xFFFF;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/true, /*z=*/false, /*c=*/false, /*v=*/false);
}

// FCMP Dn, #0.0: exactly 0.0 -> equal -> Z,C.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpZeroDEqual) {
  static const uint32_t code[] = {FcmpZeroD(6)};
  SetVf64(&state_, 6, 0.0);
  state_.cpu.flags = 0;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/true, /*c=*/true, /*v=*/false);
}

// FCMPE produces the same NZCV as FCMP for ordered operands.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpeSGreater) {
  static const uint32_t code[] = {FcmpeS(3, 4)};
  SetVf32(&state_, 3, 9.0f);
  SetVf32(&state_, 4, 4.0f);
  state_.cpu.flags = 0xFFFF;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/false, /*c=*/true, /*v=*/false);
}

// FCCMP, condition TRUE (eq with Z pre-set): takes the compare path.
// 3.0 > 1.0 -> greater -> C.
TEST_F(Arm64HeavyOptimizerFrontendTest, FccmpSCondTrueCompares) {
  static const uint32_t code[] = {FccmpS(1, 2, /*nzcv=*/0x0, kCondEq)};
  SetVf32(&state_, 1, 3.0f);
  SetVf32(&state_, 2, 1.0f);
  state_.cpu.flags = CPUState::kFlagZero;  // eq holds
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/false, /*c=*/true, /*v=*/false);
}

// FCCMP, condition FALSE (eq with Z clear): writes the nzcv immediate #0b1010
// (N=1,Z=0,C=1,V=0) verbatim, ignoring the operands.
TEST_F(Arm64HeavyOptimizerFrontendTest, FccmpSCondFalseWritesImm) {
  static const uint32_t code[] = {FccmpS(1, 2, /*nzcv=*/0b1010, kCondEq)};
  SetVf32(&state_, 1, 3.0f);  // would be "greater" if the compare ran
  SetVf32(&state_, 2, 1.0f);
  state_.cpu.flags = 0;  // eq fails (Z clear)
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/true, /*z=*/false, /*c=*/true, /*v=*/false);
}

// FCCMP with AL always takes the compare path. D-form, equal -> Z,C.
TEST_F(Arm64HeavyOptimizerFrontendTest, FccmpDAlwaysCompares) {
  static const uint32_t code[] = {FccmpD(5, 6, /*nzcv=*/0b0001, kCondAl)};
  SetVf64(&state_, 5, 7.5);
  SetVf64(&state_, 6, 7.5);
  state_.cpu.flags = 0;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/true, /*c=*/true, /*v=*/false);
}

// FCCMP, condition FALSE via NE (Z pre-set so ne fails): imm #0b0110 (Z,C).
TEST_F(Arm64HeavyOptimizerFrontendTest, FccmpSNeCondFalseWritesImm) {
  static const uint32_t code[] = {FccmpS(1, 2, /*nzcv=*/0b0110, kCondNe)};
  SetVf32(&state_, 1, 1.0f);  // would be "less" if the compare ran
  SetVf32(&state_, 2, 2.0f);
  state_.cpu.flags = CPUState::kFlagZero;  // ne fails
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/true, /*c=*/true, /*v=*/false);
}

// --- FpDataProc2 FP16 (F16C round-trip) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FaddH(0, 1, 2)};
  SetVu16(&state_, 1, 0x3E00);  // 1.5h
  SetVu16(&state_, 2, 0x4080);  // 2.25h
  SetVu16(&state_, 0, 0x1234);  // poison dst
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4380);  // 3.75h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FsubH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FsubH(0, 1, 2)};
  SetVu16(&state_, 1, 0x4500);  // 5.0h
  SetVu16(&state_, 2, 0x3D00);  // 1.25h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4380);  // 3.75h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmulH(0, 1, 2)};
  SetVu16(&state_, 1, 0x3800);  // 0.5h
  SetVu16(&state_, 2, 0x4200);  // 3.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x3E00);  // 1.5h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FdivH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FdivH(0, 1, 2)};
  SetVu16(&state_, 1, 0x4880);  // 9.0h
  SetVu16(&state_, 2, 0x4400);  // 4.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4080);  // 2.25h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxH(0, 1, 2)};
  SetVu16(&state_, 1, 0x3E00);  // 1.5h
  SetVu16(&state_, 2, 0x4080);  // 2.25h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4080);  // 2.25h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FminH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FminH(0, 1, 2)};
  SetVu16(&state_, 1, 0x3E00);  // 1.5h
  SetVu16(&state_, 2, 0x4080);  // 2.25h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x3E00);  // 1.5h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmHNanSuppressed) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxnmH(0, 1, 2)};
  SetVu16(&state_, 1, 0x7E00);  // qNaN
  SetVu16(&state_, 2, 0x4000);  // 2.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmHNanSuppressed) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FminnmH(0, 1, 2)};
  SetVu16(&state_, 1, 0x4000);  // 2.0h
  SetVu16(&state_, 2, 0x7E00);  // qNaN
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FnmulH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FnmulH(0, 1, 2)};
  SetVu16(&state_, 1, 0x3800);  // 0.5h
  SetVu16(&state_, 2, 0x4200);  // 3.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0xBE00);  // -1.5h
}

// --- FpDataProc1 FP16 ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovRegH) {  // no F16C needed
  static const uint32_t code[] = {FmovRegH(0, 1)};
  SetVu16(&state_, 1, 0x4200);  // 3.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4200);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FabsH) {  // no F16C needed
  static const uint32_t code[] = {FabsH(0, 1)};
  SetVu16(&state_, 1, 0xC200);  // -3.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4200);  // 3.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FnegH) {  // no F16C needed
  static const uint32_t code[] = {FnegH(0, 1)};
  SetVu16(&state_, 1, 0x4000);  // 2.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0xC000);  // -2.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FsqrtH(0, 1)};
  SetVu16(&state_, 1, 0x4400);  // 4.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaHTiesAway) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintaH(0, 1)};
  SetVu16(&state_, 1, 0x4100);  // 2.5h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4200);  // 3.0h (ties away)
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintnHTiesEven) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintnH(0, 1)};
  SetVu16(&state_, 1, 0x4100);  // 2.5h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h (ties to even)
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintmHFloor) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintmH(0, 1)};
  SetVu16(&state_, 1, 0x4100);  // 2.5h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintpHCeil) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintpH(0, 1)};
  SetVu16(&state_, 1, 0x4100);  // 2.5h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4200);  // 3.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintzHTrunc) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintzH(0, 1)};
  SetVu16(&state_, 1, 0x4100);  // 2.5h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h
}

// --- FCVT half <-> single/double ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtHtoS) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcvtHtoS(0, 1)};
  SetVu16(&state_, 1, 0x4000);  // 2.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), 2.0f);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtHtoD) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcvtHtoD(0, 1)};
  SetVu16(&state_, 1, 0x4000);  // 2.0h
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), 2.0);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtStoH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcvtStoH(0, 1)};
  SetVf32(&state_, 1, 2.0f);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtDtoH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcvtDtoH(0, 1)};
  SetVf64(&state_, 1, 2.0);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h
}
// Narrow with RNE rounding: 1.5f is exactly representable in FP16 (0x3E00).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtStoHExact) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcvtStoH(0, 1)};
  SetVf32(&state_, 1, 1.5f);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x3E00);  // 1.5h
}

// --- FMOV immediate FP16 (no F16C) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovImmH) {
  static const uint32_t code[] = {FmovImmH(0, 0x70)};  // #1.0
  SetVu16(&state_, 0, 0xFFFF);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x3C00);  // 1.0h
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovImmHTwo) {
  static const uint32_t code[] = {FmovImmH(0, 0x00)};  // #2.0
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4000);  // 2.0h
}

// --- FCMP / FCCMP FP16 ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpHGreater) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmpH(1, 2)};
  SetVu16(&state_, 1, 0x4000);  // 2.0h
  SetVu16(&state_, 2, 0x3C00);  // 1.0h
  state_.cpu.flags = 0xFFFF;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/false, /*c=*/true, /*v=*/false);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpHLess) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmpH(1, 2)};
  SetVu16(&state_, 1, 0x3C00);  // 1.0h
  SetVu16(&state_, 2, 0x4000);  // 2.0h
  state_.cpu.flags = 0xFFFF;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/true, /*z=*/false, /*c=*/false, /*v=*/false);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpHEqual) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmpH(1, 2)};
  SetVu16(&state_, 1, 0x4000);  // 2.0h
  SetVu16(&state_, 2, 0x4000);  // 2.0h
  state_.cpu.flags = 0xFFFF;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/true, /*c=*/true, /*v=*/false);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpHZeroEqual) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmpZeroH(1)};
  SetVu16(&state_, 1, 0x0000);  // +0.0h
  state_.cpu.flags = 0xFFFF;
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/true, /*c=*/true, /*v=*/false);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FccmpHTrueGreater) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FccmpH(1, 2, /*nzcv=*/0, /*cond=*/kCondEq)};
  SetVu16(&state_, 1, 0x4000);  // 2.0h
  SetVu16(&state_, 2, 0x3C00);  // 1.0h
  state_.cpu.flags = CPUState::kFlagZero;  // cond EQ true -> do the compare
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/false, /*z=*/false, /*c=*/true, /*v=*/false);  // greater
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FccmpHFalseUsesImm) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FccmpH(1, 2, /*nzcv=*/0xF, /*cond=*/kCondEq)};
  SetVu16(&state_, 1, 0x4000);
  SetVu16(&state_, 2, 0x3C00);
  state_.cpu.flags = 0;  // cond EQ false -> write nzcv imm 0xF (all set)
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  ExpectNZCV(state_, /*n=*/true, /*z=*/true, /*c=*/true, /*v=*/true);
}

// With F16C the heavy tier now lowers FCMP H directly; without it, the region
// still bails to lite (n == 0).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpHBailsWithoutF16C) {
  if (host_platform::kHasF16C) GTEST_SKIP() << "F16C present; FCMP H lowers in-tier";
  static const uint32_t code[] = {FcmpH(1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// --- AdvSimdThreeSame FP16 vector (F16C round-trip) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FaddVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x3C003C003C003C00ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4500440042004000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FaddVecH(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0xC000490046004500ULL);
  SetV128(&state_, 2, 0x3C003C003C003C00ULL, 0x3C003C003C003C00ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4500440042004000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBC00498047004600ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FsubVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FsubVecH(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x4500440042004000ULL, 0xBC00498047004600ULL);
  SetV128(&state_, 2, 0x3C003C003C003C00ULL, 0x3C003C003C003C00ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4400420040003C00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC000490046004500ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmulVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x4400420040003800ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x3800400038004200ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x400046003C003E00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FdivVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FdivVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x4900460044004880ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x4000420040004400ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4500400040004080ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxVecH(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xC000450042003E00ULL, 0xBC00490046003800ULL);
  SetV128(&state_, 2, 0x3C00420040004100ULL, 0xC500400044003C00ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3C00450042004100ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBC00490046003C00ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FminVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FminVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0xC000450042003E00ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x3C00420040004100ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC000420040003E00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmVecH4HNanSuppressed) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxnmVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x3C00440040007E00ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x450047007E004200ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4500470040004200ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmVecH4HNanSuppressed) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FminnmVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x3C00440040007E00ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x450047007E004200ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3C00440040004200ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// FMAX .4H +/-0 tie must be +0h (0x0000), not -0h. Existing FmaxVecH tests used
// non-zero operands, so this tie regression slipped through.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxVecH4HZeroTie) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxVecH(/*q=*/false, 0, 1, 2)};
  // lanes n=[+0,-0,1.0,qNaN], m=[-0,+0,2.0,1.0]
  SetV128(&state_, 1, 0x7E003C0080000000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x3C00400000008000ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  // expected [+0,+0,2.0,qNaN]
  EXPECT_EQ(VLo64(&state_, 0), 0x7E00400000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// FMAXNM .4H +/-0 tie = +0h; NaN suppressed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmVecH4HZeroTie) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxnmVecH(/*q=*/false, 0, 1, 2)};
  // lanes n=[+0,-0,qNaN,1.0], m=[-0,+0,3.0,qNaN]
  SetV128(&state_, 1, 0x3C007E0080000000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x7E00420000008000ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  // expected [+0,+0,3.0,1.0]
  EXPECT_EQ(VLo64(&state_, 0), 0x3C00420000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// FMINNM .4H order-swapped tie was ALSO wrong (single MINPS -> +0); fix -> -0h.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmVecH4HZeroTieOrderSwapped) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FminnmVecH(/*q=*/false, 0, 1, 2)};
  // lanes n=[-0,+0,qNaN,2.0], m=[+0,-0,3.0,qNaN]
  SetV128(&state_, 1, 0x40007E0000008000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x7E00420080000000ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  // expected [-0,-0,3.0,2.0]
  EXPECT_EQ(VLo64(&state_, 0), 0x4000420080008000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// FMAX .8H +/-0 ties across both halves (high-half path).
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxVecH8HZeroTieBothHalves) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxVecH(/*q=*/true, 0, 1, 2)};
  // n lo=[+0,1.0,-0,2.0] hi=[-0,3.0,+0,qNaN]; m lo=[-0,2.0,+0,1.0] hi=[+0,1.0,-0,1.0]
  SetV128(&state_, 1, 0x400080003C000000ULL, 0x7E00000042008000ULL);
  SetV128(&state_, 2, 0x3C00000040008000ULL, 0x3C0080003C000000ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  // expected lo=[+0,2.0,+0,2.0] hi=[+0,3.0,+0,qNaN]
  EXPECT_EQ(VLo64(&state_, 0), 0x4000000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x7E00000042000000ULL);
}
// --- FP16 vector FMULX (F16C round-trip) ---
// Lanes: 2*3=6; (+0)*(+inf)=+2.0; (-0)*(+inf)=-2.0; 4*0.5=2.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmulxVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x4400800000004000ULL, 0x1111111111111111ULL);  // [2.0,+0,-0,4.0]
  SetV128(&state_, 2, 0x38007C007C004200ULL, 0x2222222222222222ULL);  // [3.0,+inf,+inf,0.5]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4000C00040004600ULL);  // [6.0,+2.0,-2.0,2.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmulxVecH(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x4400800000004000ULL, 0xC0003C00FC007C00ULL);
  SetV128(&state_, 2, 0x38007C007C004200ULL, 0x4200450000008000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4000C00040004600ULL);        // [6.0,+2.0,-2.0,2.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC6004500C000C000ULL);   // [-2.0,-2.0,5.0,-6.0]
}
// --- FP16 scalar FMULX (H) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxScalarH_normal) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x5E421C20u};  // fmulx h0, h1, h2
  SetV128(&state_, 1, 0x0000000000004200ULL, 0x1111111111111111ULL);  // h0=3.0
  SetV128(&state_, 2, 0x0000000000004400ULL, 0x2222222222222222ULL);  // h0=4.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000004A00ULL);   // 12.0, upper bytes zeroed
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxScalarH_zero_times_inf_pos) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x5E421C20u};  // fmulx h0, h1, h2
  SetV128(&state_, 1, 0x0000000000000000ULL, 0x1111111111111111ULL);  // h0=+0.0
  SetV128(&state_, 2, 0x0000000000007C00ULL, 0x2222222222222222ULL);  // h0=+inf
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000004000ULL);   // +2.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxScalarH_negzero_times_inf) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x5E421C20u};  // fmulx h0, h1, h2
  SetV128(&state_, 1, 0x0000000000008000ULL, 0x1111111111111111ULL);  // h0=-0.0
  SetV128(&state_, 2, 0x0000000000007C00ULL, 0x2222222222222222ULL);  // h0=+inf
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000C000ULL);   // -2.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// --- FP16 vector pairwise ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddpVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FaddpVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x4800440040003C00ULL, 0x1111111111111111ULL);  // [1,2,4,8]
  SetV128(&state_, 2, 0x3E00380045004200ULL, 0x2222222222222222ULL);  // [3,5,0.5,1.5]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x400048004A004200ULL);  // [3.0, 12.0, 8.0, 2.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxpVecH4H_ZeroTieAndNan) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxpVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x4700420080000000ULL, 0x1111111111111111ULL);  // [+0,-0,3,7]
  SetV128(&state_, 2, 0x45003C0040007E00ULL, 0x2222222222222222ULL);  // [qNaN,2,1,5]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x45007E0047000000ULL);   // [+0, 7, qNaN, 5]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmpVecH4H_ZeroTieAndNan) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxnmpVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x4700420080000000ULL, 0x1111111111111111ULL);  // [+0,-0,3,7]
  SetV128(&state_, 2, 0x45003C0040007E00ULL, 0x2222222222222222ULL);  // [qNaN,2,1,5]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4500400047000000ULL);   // [+0, 7, 2, 5]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// --- FP16 scalar pairwise (reduces Vn.h[0], Vn.h[1]) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddpScalarH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x5E30D820u};  // faddp h0, v1.2h
  SetV128(&state_, 1, 0x0000000040003C00ULL, 0x1111111111111111ULL);  // h0=1.0, h1=2.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000004200ULL);   // 3.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxpScalarH_ZeroTiePlus) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x5E30F820u};  // fmaxp h0, v1.2h
  SetV128(&state_, 1, 0x0000000080000000ULL, 0x1111111111111111ULL);  // h0=+0, h1=-0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000000ULL);   // +0.0 (tie -> +0)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FminpScalarH_ZeroTieMinus) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x5EB0F820u};  // fminp h0, v1.2h
  SetV128(&state_, 1, 0x0000000080000000ULL, 0x1111111111111111ULL);  // h0=+0, h1=-0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000008000ULL);   // -0.0 (min tie -> -0)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmpScalarH_NanSuppressed) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x5E30C820u};  // fmaxnmp h0, v1.2h
  SetV128(&state_, 1, 0x0000000044007E00ULL, 0x1111111111111111ULL);  // h0=qNaN, h1=4.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000004400ULL);   // 4.0 (NaN dropped)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmpScalarH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x5EB0C820u};  // fminnmp h0, v1.2h
  SetV128(&state_, 1, 0x0000000045004200ULL, 0x1111111111111111ULL);  // h0=3.0, h1=5.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000004200ULL);   // 3.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FabdVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FabdVecH(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0xC000490046004500ULL);
  SetV128(&state_, 2, 0x400044003C004000ULL, 0x4200490047004400ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x40003C003C003C00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x450000003C003C00ULL);
}
// --- FP16 scalar FMA (FMADD/FMSUB/FNMADD/FNMSUB Hd), fused in FP64. rn=1,rm=2,ra=3.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaddScalarH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x1FC20C20u};  // fmadd h0,h1,h2,h3
  SetV128(&state_, 1, 0x0000000000004000ULL, 0x1111111111111111ULL);  // 2.0
  SetV128(&state_, 2, 0x0000000000004200ULL, 0x2222222222222222ULL);  // 3.0
  SetV128(&state_, 3, 0x0000000000003C00ULL, 0x3333333333333333ULL);  // 1.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000004700ULL);  // 1 + 2*3 = 7.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmsubScalarH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x1FC28C20u};  // fmsub h0,h1,h2,h3
  SetV128(&state_, 1, 0x0000000000004200ULL, 0x1111111111111111ULL);  // 3.0
  SetV128(&state_, 2, 0x0000000000004000ULL, 0x2222222222222222ULL);  // 2.0
  SetV128(&state_, 3, 0x0000000000003C00ULL, 0x3333333333333333ULL);  // 1.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000C500ULL);  // 1 - 3*2 = -5.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FnmaddScalarH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x1FE20C20u};  // fnmadd h0,h1,h2,h3
  SetV128(&state_, 1, 0x0000000000004000ULL, 0x1111111111111111ULL);  // 2.0
  SetV128(&state_, 2, 0x0000000000004200ULL, 0x2222222222222222ULL);  // 3.0
  SetV128(&state_, 3, 0x0000000000003C00ULL, 0x3333333333333333ULL);  // 1.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000C700ULL);  // -(1 + 2*3) = -7.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FnmsubScalarH) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x1FE28C20u};  // fnmsub h0,h1,h2,h3
  SetV128(&state_, 1, 0x0000000000004000ULL, 0x1111111111111111ULL);  // 2.0
  SetV128(&state_, 2, 0x0000000000004200ULL, 0x2222222222222222ULL);  // 3.0
  SetV128(&state_, 3, 0x0000000000003C00ULL, 0x3333333333333333ULL);  // 1.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000004500ULL);  // 2*3 - 1 = 5.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// Fusion proof: 0x0003 (subnormal fp16 ~1.79e-7) * 2732 + 1.0. The exact fused
// value 1.000488... rounds to 0x3C01; rounding the product to fp16 FIRST (unfused)
// would give 0x3C00. Only true single-rounding fusion (FP64) yields 0x3C01.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaddScalarH_FusionSingleRounding) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x1FC20C20u};  // fmadd h0,h1,h2,h3
  SetV128(&state_, 1, 0x0000000000000003ULL, 0x1111111111111111ULL);  // subnormal
  SetV128(&state_, 2, 0x0000000000006956ULL, 0x2222222222222222ULL);  // 2732.0
  SetV128(&state_, 3, 0x0000000000003C00ULL, 0x3333333333333333ULL);  // 1.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000003C01ULL);  // fused single-round
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaddScalarH_InfPassthrough) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x1FC20C20u};  // fmadd h0,h1,h2,h3
  SetV128(&state_, 1, 0x0000000000007C00ULL, 0x1111111111111111ULL);  // +inf
  SetV128(&state_, 2, 0x0000000000004000ULL, 0x2222222222222222ULL);  // 2.0
  SetV128(&state_, 3, 0x0000000000003C00ULL, 0x3333333333333333ULL);  // 1.0
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000007C00ULL);  // +inf
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// --- FP16 vector FMLA/FMLS (.4H/.8H), fused per-lane in FP64. Vd is the accumulator.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmlaVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x0E420C20u};  // fmla v0.4h,v1.4h,v2.4h
  SetV128(&state_, 1, 0x4500000342004000ULL, 0x1111111111111111ULL);  // [2,3,subn,5]
  SetV128(&state_, 2, 0x4600695644004200ULL, 0x2222222222222222ULL);  // [3,4,2732,6]
  SetV128(&state_, 0, 0x3C003C003C003C00ULL, 0xCCCCCCCCCCCCCCCCULL);  // Vd=[1,1,1,1]
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4FC03C014A804700ULL);  // [7,13,fused,31]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmlsVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x0EC20C20u};  // fmls v0.4h,v1.4h,v2.4h
  SetV128(&state_, 1, 0x4500000342004000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x4600695644004200ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0x3C003C003C003C00ULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xCF403BFFC980C500ULL);  // Vd - Vn*Vm per lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmlaVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {0x4E420C20u};  // fmla v0.8h,v1.8h,v2.8h
  // low [2,3,subn,5] hi [7,8,1,2]; vm low [3,4,2732,6] hi [2,1,3,3]; Vd all 1.0
  SetV128(&state_, 1, 0x4500000342004000ULL, 0x40003C0048004700ULL);
  SetV128(&state_, 2, 0x4600695644004200ULL, 0x420042003C004000ULL);
  SetV128(&state_, 0, 0x3C003C003C003C00ULL, 0x3C003C003C003C00ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4FC03C014A804700ULL);        // [7,13,fused,31]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4700440048804B80ULL);   // [15,9,4,7]
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgtVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmgtVecH(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x45007E003C004000ULL, 0x49003C0040004200ULL);
  SetV128(&state_, 2, 0x45003C0040003C00ULL, 0x40003C0042004000ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFF00000000FFFFULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgeVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmgeVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x7E0042003C004000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x3C00420040003C00ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000FFFF0000FFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FacgtVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FacgtVecH(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x4500C0003C00C200ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x450040003C004000ULL, 0x2222222222222222ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// --- AdvSimdTwoRegMisc FP16 vector ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FabsVecH4HNanPayload) {  // no F16C needed; sign-only
  static const uint32_t code[] = {FabsVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x8000FD554000BE00ULL, 0x1111111111111111ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00007D5540003E00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FnegVecH4HNanPayload) {  // no F16C needed
  static const uint32_t code[] = {FnegVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x8000FD554000BE00ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00007D55C0003E00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FabsVecH8H) {  // no F16C; distinct high lanes
  static const uint32_t code[] = {FabsVecH(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x8000FD554000BE00ULL, 0x4400FC013800C200ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00007D5540003E00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x44007C0138004200ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FsqrtVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x4080340048804400ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3E00380042004000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintpVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintpVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x4200BE0041003E00ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4200BC0042004000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintmVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintmVecH(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4200BE0041003E00ULL, 0x4500C100B8003800ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4200C00040003C00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4500C200BC000000ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintzVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintzVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x4200BE0041003E00ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4200BC0040003C00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintnVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintnVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x4200BE0041003E00ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4200C00040004000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintaVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintaVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x4200BE0041003E00ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4200C00042004000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrintxVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FrintxVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x4200BE0041003E00ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4200C00040004000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
// --- FP16 compare-vs-zero (routes via integer kCmeqZero / kCmgtZero group) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmeqZeroVecH4HNegZero) {  // pins the -0.0h bug fix
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmeqZeroVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x7E003C0080000000ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgtZeroVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmgtZeroVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x7E000000BC004000ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgeZeroVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmgeZeroVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x7E00BC003C000000ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmltZeroVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmltZeroVecH(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x7E0080004000BC00ULL, 0x1111111111111111ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmleZeroVecH8H) {  // distinct high lanes
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FcmleZeroVecH(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x7E003C00BC000000ULL, 0x4500C20040008000ULL);
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000FFFF0000FFFFULL);
}

// --- URECPE / URSQRTE (unsigned integer estimate, table lookup) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, UrecpeVec4S) {
  static const uint32_t code[] = {UrecpeVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0x7FFFFFFFFF800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF800000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFF80000000ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UrecpeVec2S) {
  static const uint32_t code[] = {UrecpeVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0x1111111111111111ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF800000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UrsqrteVec4S) {
  static const uint32_t code[] = {UrsqrteVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4000000000000000ULL, 0xFF80000080000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF800000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x80000000B4800000ULL);
}

// --- FP64 FCADD / FCMLA (.2D) ---
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcadd90Vec2D) {
  static const uint32_t code[] = {0x6EC2E420u};  // fcadd v0.2d, v1.2d, v2.2d, #90
  SetV128(&state_, 1, 0x4008000000000000ULL, 0x4014000000000000ULL);  // (3, 5)
  SetV128(&state_, 2, 0x4000000000000000ULL, 0x401C000000000000ULL);  // (2, 7)
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC010000000000000ULL);       // -4.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x401C000000000000ULL);  //  7.0
}
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcadd270Vec2D) {
  static const uint32_t code[] = {0x6EC2F420u};  // fcadd v0.2d, v1.2d, v2.2d, #270
  SetV128(&state_, 1, 0x4008000000000000ULL, 0x4014000000000000ULL);
  SetV128(&state_, 2, 0x4000000000000000ULL, 0x401C000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4024000000000000ULL);       // 10.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4008000000000000ULL);  //  3.0
}
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcmla0Vec2D) {
  static const uint32_t code[] = {0x6EC2C420u};  // fcmla v0.2d, v1.2d, v2.2d, #0
  SetV128(&state_, 1, 0x4000000000000000ULL, 0x4008000000000000ULL);  // Vn=(2,3)
  SetV128(&state_, 2, 0x4010000000000000ULL, 0x4014000000000000ULL);  // Vm=(4,5)
  SetV128(&state_, 0, 0x3FF0000000000000ULL, 0x3FF0000000000000ULL);  // Vd=(1,1)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4022000000000000ULL);       //  9.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4026000000000000ULL);  // 11.0
}
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcmla90Vec2D) {
  static const uint32_t code[] = {0x6EC2CC20u};  // fcmla v0.2d, v1.2d, v2.2d, #90
  SetV128(&state_, 1, 0x4000000000000000ULL, 0x4008000000000000ULL);
  SetV128(&state_, 2, 0x4010000000000000ULL, 0x4014000000000000ULL);
  SetV128(&state_, 0, 0x3FF0000000000000ULL, 0x3FF0000000000000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC02C000000000000ULL);       // -14.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x402A000000000000ULL);  //  13.0
}
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcmla180Vec2D) {
  static const uint32_t code[] = {0x6EC2D420u};  // fcmla v0.2d, v1.2d, v2.2d, #180
  SetV128(&state_, 1, 0x4000000000000000ULL, 0x4008000000000000ULL);
  SetV128(&state_, 2, 0x4010000000000000ULL, 0x4014000000000000ULL);
  SetV128(&state_, 0, 0x3FF0000000000000ULL, 0x3FF0000000000000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC01C000000000000ULL);       // -7.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC022000000000000ULL);  // -9.0
}
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcmla270Vec2D) {
  static const uint32_t code[] = {0x6EC2DC20u};  // fcmla v0.2d, v1.2d, v2.2d, #270
  SetV128(&state_, 1, 0x4000000000000000ULL, 0x4008000000000000ULL);
  SetV128(&state_, 2, 0x4010000000000000ULL, 0x4014000000000000ULL);
  SetV128(&state_, 0, 0x3FF0000000000000ULL, 0x3FF0000000000000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4030000000000000ULL);       //  16.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC026000000000000ULL);  // -11.0
}

//
// AdvSIMD three-same INTEGER: ADD, SUB, AND, ORR, EOR, MUL. Each asserts the
// result lanes and, for the D-form (Q=0), that the upper 64 bits of Vd are
// zeroed. CMEQ, saturating, and unsupported sizes must bail.
//

// ADD .4S (Q=1): four 32-bit lane adds, full 128-bit result.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddVec4S) {
  static const uint32_t code[] = {AddVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000200000001ULL, 0x0000000400000003ULL);
  SetV128(&state_, 2, 0x0000002000000010ULL, 0x0000040000000300ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000002200000011ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000040400000303ULL);
}

// CMEQ .4S (Q=1): per-lane equality -> all-ones / zero, via PCMPEQD.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmeqVec4S) {
  static const uint32_t code[] = {0x6EA28C20u};  // cmeq v0.4s, v1.4s, v2.4s
  SetV128(&state_, 1, 0x0000000200000001ULL, 0x0000000400000003ULL);  // [1,2,3,4]
  SetV128(&state_, 2, 0x0000000900000001ULL, 0x0000000900000003ULL);  // [1,9,3,9]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);   // eq, ne
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFFULL);  // eq, ne
}

// CMEQ .16B (Q=1): per-byte equality via PCMPEQB.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmeqVec16B) {
  static const uint32_t code[] = {0x6E228C20u};  // cmeq v0.16b, v1.16b, v2.16b
  SetV128(&state_, 1, 0x1111111122222222ULL, 0x0000000000000000ULL);
  SetV128(&state_, 2, 0x1111111133333333ULL, 0x0000000000000000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF00000000ULL);   // low 4 bytes differ, high 4 equal
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);  // all equal (0 == 0)
}

// AES via host AES-NI. Expected values are the interpreter's (from-scratch,
// FIPS-197) reference outputs for the same vectors.
TEST_F(Arm64HeavyOptimizerFrontendTest, AeseNI) {
  static const uint32_t code[] = {0x4E284820u};  // aese v0.16b, v1.16b
  SetV128(&state_, 0, 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
  SetV128(&state_, 1, 0x0f0f0f0f0f0f0f0fULL, 0x0f0f0f0f0f0f0f0fULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xfe7c6f2b636b6776ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xf201ab7b30d777c5ULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AesdNI) {
  static const uint32_t code[] = {0x4E285820u};  // aesd v0.16b, v1.16b
  SetV128(&state_, 0, 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
  SetV128(&state_, 1, 0x0f0f0f0f0f0f0f0fULL, 0x0f0f0f0f0f0f0f0fULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3009d79ebf366afbULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x8140a5d552f3a338ULL);
}

// AESMC exercises the standalone-MixColumns identity AESENC(AESDECLAST(Vn,0),0).
TEST_F(Arm64HeavyOptimizerFrontendTest, AesmcNI) {
  static const uint32_t code[] = {0x4E286820u};  // aesmc v0.16b, v1.16b
  SetV128(&state_, 1, 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
  SetV128(&state_, 0, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0104030605000702ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x090c0b0e0d080f0aULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, AesimcNI) {
  static const uint32_t code[] = {0x4E287820u};  // aesimc v0.16b, v1.16b
  SetV128(&state_, 1, 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL);
  SetV128(&state_, 0, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x090c0b0e0d080f0aULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0104030605000702ULL);
}

// SDOT .4S (Q=1): 4 signed byte products summed per lane, accumulated into Vd.
// Vn bytes all 2, Vm bytes all 3 -> each lane += 4*(2*3)=24; Vd starts non-zero
// to prove accumulation.
TEST_F(Arm64HeavyOptimizerFrontendTest, SdotVec4S) {
  static const uint32_t code[] = {0x4E829420u};  // sdot v0.4s, v1.16b, v2.16b
  SetV128(&state_, 1, 0x0202020202020202ULL, 0x0202020202020202ULL);
  SetV128(&state_, 2, 0x0303030303030303ULL, 0x0303030303030303ULL);
  SetV128(&state_, 0, 0x0000000A00000005ULL, 0x0000001E00000014ULL);  // [5,10,20,30]
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // [5+24, 10+24, 20+24, 30+24] = [29,34,44,54].
  EXPECT_EQ(VLo64(&state_, 0), 0x000000220000001DULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000360000002CULL);
}

// UDOT .4S: unsigned bytes. Vn bytes 0xFF = 255 (unsigned) -> lane = 4*(255*2)=2040.
TEST_F(Arm64HeavyOptimizerFrontendTest, UdotVec4S_unsigned) {
  static const uint32_t code[] = {0x6E829420u};  // udot v0.4s, v1.16b, v2.16b
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 2, 0x0202020202020202ULL, 0x0202020202020202ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000007F8000007F8ULL);   // 2040 per lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000007F8000007F8ULL);
}

// SDOT .4S: same 0xFF bytes but signed = -1 -> lane = 4*(-1*2) = -8. Distinguishes
// the signed path from the unsigned test above.
TEST_F(Arm64HeavyOptimizerFrontendTest, SdotVec4S_signed_negative) {
  static const uint32_t code[] = {0x4E829420u};  // sdot v0.4s, v1.16b, v2.16b
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 2, 0x0202020202020202ULL, 0x0202020202020202ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFF8FFFFFFF8ULL);   // -8 per lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFF8FFFFFFF8ULL);
}

// SDOT .2S (Q=0): 2 lanes; the D-form must zero the upper 64 bits of Vd.
TEST_F(Arm64HeavyOptimizerFrontendTest, SdotVec2S_ZeroesUpper) {
  static const uint32_t code[] = {0x0E829420u};  // sdot v0.2s, v1.8b, v2.8b
  SetV128(&state_, 1, 0x0202020202020202ULL, 0x1111111111111111ULL);  // hi ignored
  SetV128(&state_, 2, 0x0303030303030303ULL, 0x2222222222222222ULL);
  // Low 64 is the .2s accumulator (start 0); only the upper 64 is poison, which
  // the D-form must zero.
  SetV128(&state_, 0, 0x0000000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000001800000018ULL);   // 24 per lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // upper zeroed
}

// SDOT by element: Vm.4b[1] (bytes 4..7) drives all 4 output lanes. Byte group 0
// is zero, so a wrong index would give 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, SdotIndexed4S) {
  static const uint32_t code[] = {0x4FA2E020u};  // sdot v0.4s, v1.16b, v2.4b[1]
  SetV128(&state_, 1, 0x0202020202020202ULL, 0x0202020202020202ULL);
  SetV128(&state_, 2, 0x0303030300000000ULL, 0ULL);  // 4b[1]={3,3,3,3}, 4b[0]=0
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000001800000018ULL);   // 4*(2*3)=24 per lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000001800000018ULL);
}

// FCADD #90 .2s (Q=0): Vd = Vn + rot90(Vm); D-form zeroes the upper 64. Complex
// pairs are [re, im] in adjacent FP32 lanes. Vn=(1,2), Vm=(3,4) -> (1-4, 2+3) =
// (-3, 5). FP32: 1.0=0x3F800000 2.0=0x40000000 3.0=0x40400000 4.0=0x40800000
// 5.0=0x40A00000 -3.0=0xC0400000.
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcadd90Vec2S) {
  static const uint32_t code[] = {0x2E82E420u};  // fcadd v0.2s, v1.2s, v2.2s, #90
  SetV128(&state_, 1, 0x400000003F800000ULL, 0x1111111111111111ULL);  // (1,2)
  SetV128(&state_, 2, 0x4080000040400000ULL, 0x2222222222222222ULL);  // (3,4)
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x40A00000C0400000ULL);   // (-3, 5)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // upper zeroed
}

// FCADD #270 .4s (Q=1): Vn=[(1,2),(10,20)], Vm=[(3,4),(5,6)]; rot270 gives
// (re+im, im-re) -> [(5,-1),(16,15)]. 10.0=0x41200000 20.0=0x41A00000
// 6.0=0x40C00000 -1.0=0xBF800000 16.0=0x41800000 15.0=0x41700000.
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcadd270Vec4S) {
  static const uint32_t code[] = {0x6E82F420u};  // fcadd v0.4s, v1.4s, v2.4s, #270
  SetV128(&state_, 1, 0x400000003F800000ULL, 0x41A0000041200000ULL);  // (1,2),(10,20)
  SetV128(&state_, 2, 0x4080000040400000ULL, 0x40C0000040A00000ULL);  // (3,4),(5,6)
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xBF80000040A00000ULL);   // (5, -1)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4170000041800000ULL);  // (16, 15)
}

// FCMLA #0 .4s: Vd += Vn_re * Vm (per pair). Vn=(2,3) both pairs, Vm=(5,7),
// Vd=(1,1) -> (1+2*5, 1+2*7) = (11,15). 7.0=0x40E00000 11.0=0x41300000.
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcmla0Vec4S) {
  static const uint32_t code[] = {0x6E82C420u};  // fcmla v0.4s, v1.4s, v2.4s, #0
  SetV128(&state_, 1, 0x4040000040000000ULL, 0x4040000040000000ULL);  // (2,3),(2,3)
  SetV128(&state_, 2, 0x40E0000040A00000ULL, 0x40E0000040A00000ULL);  // (5,7),(5,7)
  SetV128(&state_, 0, 0x3F8000003F800000ULL, 0x3F8000003F800000ULL);  // (1,1),(1,1)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4170000041300000ULL);   // (11, 15)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4170000041300000ULL);
}

// FCMLA #90 .4s: Vd += rot90 term = (-Vn_im*Vm_im, Vn_im*Vm_re). Vn=(2,3),
// Vm=(5,7), Vd=(1,1) -> (1-3*7, 1+3*5) = (-20,16). -20.0=0xC1A00000
// 16.0=0x41800000.
TEST_F(Arm64HeavyOptimizerFrontendTest, Fcmla90Vec4S) {
  static const uint32_t code[] = {0x6E82CC20u};  // fcmla v0.4s, v1.4s, v2.4s, #90
  SetV128(&state_, 1, 0x4040000040000000ULL, 0x4040000040000000ULL);  // (2,3),(2,3)
  SetV128(&state_, 2, 0x40E0000040A00000ULL, 0x40E0000040A00000ULL);  // (5,7),(5,7)
  SetV128(&state_, 0, 0x3F8000003F800000ULL, 0x3F8000003F800000ULL);  // (1,1),(1,1)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x41800000C1A00000ULL);   // (-20, 16)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x41800000C1A00000ULL);
}

// ---- I8MM matrix multiply (SMMLA/UMMLA/USMMLA) ----
// Vn = 2 rows of 8 int8 (row0=low64, row1=hi64); Vm likewise (its rows are the
// result's columns). Output lane (2i+j) += dot(Vn row i, Vm row j) over 8 bytes.
// Lane order: [ (0,0), (0,1), (1,0), (1,1) ].

// SMMLA: Vn row0=1, row1=2; Vm row0=3, row1=4. Each dot = 8*(a*b).
//   (0,0)=8*1*3=24, (0,1)=8*1*4=32, (1,0)=8*2*3=48, (1,1)=8*2*4=64.
TEST_F(Arm64HeavyOptimizerFrontendTest, SmmlaBasic) {
  static const uint32_t code[] = {0x4E82A420u};  // smmla v0.4s, v1.16b, v2.16b
  SetV128(&state_, 1, 0x0101010101010101ULL, 0x0202020202020202ULL);
  SetV128(&state_, 2, 0x0303030303030303ULL, 0x0404040404040404ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000002000000018ULL);      // [24, 32]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000004000000030ULL);  // [48, 64]
}

// UMMLA: Vn row0=255 (0xFF unsigned), row1=1; Vm row0=2, row1=3.
//   (0,0)=8*255*2=4080, (0,1)=8*255*3=6120, (1,0)=8*1*2=16, (1,1)=8*1*3=24.
TEST_F(Arm64HeavyOptimizerFrontendTest, UmmlaUnsigned) {
  static const uint32_t code[] = {0x6E82A420u};  // ummla v0.4s, v1.16b, v2.16b
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0x0101010101010101ULL);
  SetV128(&state_, 2, 0x0202020202020202ULL, 0x0303030303030303ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000017E800000FF0ULL);      // [4080, 6120]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000001800000010ULL);  // [16, 24]
}

// USMMLA (Vn unsigned, Vm signed): Vn row0=2, row1=3; Vm row0=-1 (0xFF signed),
// row1=2.  (0,0)=8*2*-1=-16, (0,1)=8*2*2=32, (1,0)=8*3*-1=-24, (1,1)=8*3*2=48.
TEST_F(Arm64HeavyOptimizerFrontendTest, UsmmlaMixedSign) {
  static const uint32_t code[] = {0x4E82AC20u};  // usmmla v0.4s, v1.16b, v2.16b
  SetV128(&state_, 1, 0x0202020202020202ULL, 0x0303030303030303ULL);
  SetV128(&state_, 2, 0xFFFFFFFFFFFFFFFFULL, 0x0202020202020202ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000020FFFFFFF0ULL);      // [-16, 32]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000030FFFFFFE8ULL);  // [-24, 48]
}

// ---- USDOT / SUDOT (I8MM mixed-sign dot) — regression: heavy already lowers ----
// USDOT vec: Vn unsigned 0xFF=255, Vm signed 0xFF=-1 -> lane = 4*(255*-1) = -1020.
TEST_F(Arm64HeavyOptimizerFrontendTest, UsdotVec4S_MixedSign) {
  static const uint32_t code[] = {0x4E829C20u};  // usdot v0.4s, v1.16b, v2.16b
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 2, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFC04FFFFFC04ULL);      // -1020 per lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFC04FFFFFC04ULL);
}

// SUDOT idx (Vn signed, Vm unsigned): Vn 0xFF=-1, Vm.4b[1]=0xFF=255 -> 4*(-1*255)=-1020.
TEST_F(Arm64HeavyOptimizerFrontendTest, SudotIndexed4S_MixedSign) {
  static const uint32_t code[] = {0x4F22F020u};  // sudot v0.4s, v1.16b, v2.4b[1]
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 2, 0xFFFFFFFF00000000ULL, 0ULL);  // 4b[1]={0xFF*4}, 4b[0]=0
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFC04FFFFFC04ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFC04FFFFFC04ULL);
}

// ---- FCMLA by-element FP32 (.4s, Q=1) ----
// rot #0, index=1 -> m = Vm.s[2..3] = (5,7). Vd += Vn_re * m.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmlaIdx0_Idx1) {
  static const uint32_t code[] = {0x6F821820u};  // fcmla v0.4s, v1.4s, v2.s[1], #0
  SetV128(&state_, 1, 0x4040000040000000ULL, 0x40A0000040800000ULL);  // (2,3),(4,5)
  SetV128(&state_, 2, 0x4110000041100000ULL, 0x40E0000040A00000ULL);  // (9,9),(5,7)
  SetV128(&state_, 0, 0x3F8000003F800000ULL, 0x3F8000003F800000ULL);  // (1,1),(1,1)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4170000041300000ULL);      // (11, 15)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x41E8000041A80000ULL);  // (21, 29)
}

// rot #90, index=0 -> m = Vm.s[0..1] = (5,7). Vd.re += -Vn_im*m_im; Vd.im += Vn_im*m_re.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmlaIdx90_Idx0) {
  static const uint32_t code[] = {0x6F823020u};  // fcmla v0.4s, v1.4s, v2.s[0], #90
  SetV128(&state_, 1, 0x4040000040000000ULL, 0x40A0000040800000ULL);  // (2,3),(4,5)
  SetV128(&state_, 2, 0x40E0000040A00000ULL, 0xDEADBEEFDEADBEEFULL);  // (5,7),poison
  SetV128(&state_, 0, 0x3F8000003F800000ULL, 0x3F8000003F800000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x41800000C1A00000ULL);      // (-20, 16)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x41D00000C2080000ULL);  // (-34, 26)
}

// rot #180, index=1 -> m = (5,7). Vd.re += -Vn_re*m_re; Vd.im += -Vn_re*m_im.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmlaIdx180_Idx1) {
  static const uint32_t code[] = {0x6F825820u};  // fcmla v0.4s, v1.4s, v2.s[1], #180
  SetV128(&state_, 1, 0x4040000040000000ULL, 0x40A0000040800000ULL);  // (2,3),(4,5)
  SetV128(&state_, 2, 0xCAFEBABECAFEBABEULL, 0x40E0000040A00000ULL);  // poison,(5,7)
  SetV128(&state_, 0, 0x3F8000003F800000ULL, 0x3F8000003F800000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC1500000C1100000ULL);      // (-9, -13)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC1D80000C1980000ULL);  // (-19, -27)
}

// rot #270, index=1 -> m = (5,7). Vd.re += Vn_im*m_im; Vd.im += -Vn_im*m_re.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmlaIdx270_Idx1) {
  static const uint32_t code[] = {0x6F827820u};  // fcmla v0.4s, v1.4s, v2.s[1], #270
  SetV128(&state_, 1, 0x4040000040000000ULL, 0x40A0000040800000ULL);  // (2,3),(4,5)
  SetV128(&state_, 2, 0xCAFEBABECAFEBABEULL, 0x40E0000040A00000ULL);  // poison,(5,7)
  SetV128(&state_, 0, 0x3F8000003F800000ULL, 0x3F8000003F800000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC160000041B00000ULL);      // (22, -14)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC1C0000042100000ULL);  // (36, -24)
}

// ---- BFDOT ----
TEST_F(Arm64HeavyOptimizerFrontendTest, BfdotVec4S) {
  static const uint32_t code[] = {0x6E42FC20u};  // bfdot v0.4s, v1.8h, v2.8h
  SetV128(&state_, 1, 0x4000400040004000ULL, 0x4000400040004000ULL);  // 2.0 x8
  SetV128(&state_, 2, 0x3F803F8040404040ULL, 0x3F003F0040804080ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4080000041400000ULL);      // [12.0, 4.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4000000041800000ULL);  // [16.0, 2.0]
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BfdotVec2S_ZeroesUpper) {
  static const uint32_t code[] = {0x2E42FC20u};  // bfdot v0.2s, v1.4h, v2.4h
  SetV128(&state_, 1, 0x4000400040004000ULL, 0x1111111111111111ULL);  // hi ignored
  SetV128(&state_, 2, 0x3F803F8040404040ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0, 0xBBBBBBBBBBBBBBBBULL);  // upper poison must be zeroed
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4080000041400000ULL);      // [12.0, 4.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // upper zeroed
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BfdotIndexed4S) {
  static const uint32_t code[] = {0x4F62F820u};  // bfdot v0.4s, v1.8h, v2.2h[3]
  SetV128(&state_, 1, 0x4000400040004000ULL, 0x4000400040004000ULL);  // 2.0 x8
  SetV128(&state_, 2, 0xCAFEBABECAFEBABEULL, 0x40804040DEADBEEFULL);  // dword3=(3.0,4.0)
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4160000041600000ULL);      // 14.0 per lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4160000041600000ULL);
}

// ---- BFMMLA (.4s, Q=1 only) ----
TEST_F(Arm64HeavyOptimizerFrontendTest, BfmmlaBasic) {
  static const uint32_t code[] = {0x6E42EC20u};  // bfmmla v0.4s, v1.8h, v2.8h
  SetV128(&state_, 1, 0x4000400040004000ULL, 0x4040404040404040ULL);  // row0=2.0 row1=3.0
  SetV128(&state_, 2, 0x3F803F803F803F80ULL, 0x4080408040804080ULL);  // row0=1.0 row1=4.0
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4200000041000000ULL);      // [8, 32]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4240000041400000ULL);  // [12, 48]
}

// ---- BFMLALB / BFMLALT (vector) ----
TEST_F(Arm64HeavyOptimizerFrontendTest, BfmlalbVec) {
  static const uint32_t code[] = {0x2EC2FC20u};  // bfmlalb v0.4s, v1.8h, v2.8h
  SetV128(&state_, 1, 0xFFFF4040FFFF4000ULL, 0xFFFF3F80FFFF4080ULL);
  SetV128(&state_, 2, 0xFFFF4000FFFF3F80ULL, 0xFFFF40C0FFFF3F00ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x40C0000040000000ULL);      // [2.0, 6.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x40C0000040000000ULL);  // [2.0, 6.0]
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BfmlaltVec) {
  static const uint32_t code[] = {0x6EC2FC20u};  // bfmlalt v0.4s, v1.8h, v2.8h
  SetV128(&state_, 1, 0x4040FFFF4000FFFFULL, 0x3F80FFFF4080FFFFULL);
  SetV128(&state_, 2, 0x4000FFFF3F80FFFFULL, 0x40C0FFFF3F00FFFFULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x40C0000040000000ULL);      // [2.0, 6.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x40C0000040000000ULL);
}

// ---- BFMLALB / BFMLALT (by element) ----
TEST_F(Arm64HeavyOptimizerFrontendTest, BfmlalbIndexed) {
  static const uint32_t code[] = {0x0FC2F020u};  // bfmlalb v0.4s, v1.8h, v2.h[0]
  SetV128(&state_, 1, 0xFFFF4040FFFF4000ULL, 0xFFFF3F80FFFF4080ULL);  // even=[2,3,4,1]
  SetV128(&state_, 2, 0xCAFEBABECAFE4040ULL, 0xDEADBEEFDEADBEEFULL);  // h[0]=3.0
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4110000040C00000ULL);      // [6.0, 9.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4040000041400000ULL);  // [12.0, 3.0]
}

TEST_F(Arm64HeavyOptimizerFrontendTest, BfmlaltIndexed) {
  static const uint32_t code[] = {0x4FF2F820u};  // bfmlalt v0.4s, v1.8h, v2.h[7]
  SetV128(&state_, 1, 0x4040FFFF4000FFFFULL, 0x3F80FFFF4080FFFFULL);  // odd=[2,3,4,1]
  SetV128(&state_, 2, 0xCAFEBABECAFEBABEULL, 0x4000BEEFDEADBEEFULL);  // h[7]=2.0
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x40C0000040800000ULL);      // [4.0, 6.0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4000000041000000ULL);  // [8.0, 2.0]
}

// ===================================================================
// Heavy byte-lane MUL/MLA/MLS + scalar-SIMD coverage (slice-bytelane).
// ===================================================================

// (a) Byte MUL .16B: per-byte product truncated to 8 bits. 0x03*0x07=0x15.
TEST_F(Arm64HeavyOptimizerFrontendTest, MulVec16B) {
  static const uint32_t code[] = {MulVec(0b00, /*q=*/true, 0, 1, 2)};  // 0x4e229c20
  SetV128(&state_, 1, 0x0303030303030303ULL, 0x0303030303030303ULL);
  SetV128(&state_, 2, 0x0707070707070707ULL, 0x0707070707070707ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1515151515151515ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x1515151515151515ULL);
}

// (a) Byte MUL .16B truncation corner: 0xFF*0xFF=0xFE01, low byte 0x01.
TEST_F(Arm64HeavyOptimizerFrontendTest, MulVec16BTruncate) {
  static const uint32_t code[] = {MulVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 2, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0101010101010101ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0101010101010101ULL);
}

// (a) Byte MLA .16B: Vd += Vn*Vm per byte. 0x02 + 0x03*0x04 = 0x0E.
TEST_F(Arm64HeavyOptimizerFrontendTest, MlaVec16B) {
  static const uint32_t code[] = {MlaVec(0b00, /*q=*/true, 0, 1, 2)};  // 0x4e229420
  SetV128(&state_, 1, 0x0303030303030303ULL, 0x0303030303030303ULL);
  SetV128(&state_, 2, 0x0404040404040404ULL, 0x0404040404040404ULL);
  SetV128(&state_, 0, 0x0202020202020202ULL, 0x0202020202020202ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0E0E0E0E0E0E0E0EULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0E0E0E0E0E0E0E0EULL);
}

// (a) Byte MLS .8B (Q=0): Vd -= Vn*Vm per byte, upper 64 zeroed.
// 0x20 - 0x03*0x05 = 0x20 - 0x0F = 0x11.
TEST_F(Arm64HeavyOptimizerFrontendTest, MlsVec8BUpperZero) {
  static const uint32_t code[] = {MlsVec(0b00, /*q=*/false, 0, 1, 2)};  // 0x2e229420
  SetV128(&state_, 1, 0x0303030303030303ULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0505050505050505ULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 0, 0x2020202020202020ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111111111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (b) Byte SSRA .16B #3: Vd += SSHR(Vn,3) per byte (signed arith). Vn=0x80
// (-128), SSHR>>3 = -16 = 0xF0; Vd=0x00 -> 0xF0. Exercises sign extension.
TEST_F(Arm64HeavyOptimizerFrontendTest, SsraVec16B) {
  static const uint32_t code[] = {0x4f0d1420u};  // ssra v0.16b, v1.16b, #3
  SetV128(&state_, 1, 0x8080808080808080ULL, 0x8080808080808080ULL);
  SetV128(&state_, 0, 0x0000000000000000ULL, 0x0000000000000000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xF0F0F0F0F0F0F0F0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xF0F0F0F0F0F0F0F0ULL);
}

// (b) Byte USRA .8B #5 (Q=0): Vd += USHR(Vn,5) per byte (logical). Vn=0xFF
// (255), USHR>>5 = 7; Vd=0x01 -> 0x08. Upper 64 zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, UsraVec8BUpperZero) {
  static const uint32_t code[] = {0x2f0b1420u};  // usra v0.8b, v1.8b, #5
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 0, 0x0101010101010101ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0808080808080808ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (b) Byte URSHR .16B #2: round((Vn+2)>>2) per byte. Vn=0xFF -> round(255/4) =
// round(63.75) = 64 = 0x40. .16B is Q=1, so all 16 bytes are written.
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshrVec16B) {
  static const uint32_t code[] = {0x6f0e2420u};  // urshr v0.16b, v1.16b, #2
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 0, 0xCDCDCDCDCDCDCDCDULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4040404040404040ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4040404040404040ULL);
}

// (c) Scalar ADD D: 64-bit add on lane 0, upper zeroed. 0xFFF..F + 2 = 1.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddScalarD) {
  static const uint32_t code[] = {0x5ee28420u};  // add d0, d1, d2
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0000000000000002ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (c) Scalar SUB D: 5 - 8 = -3 = 0xFFFFFFFFFFFFFFFD, upper zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubScalarD) {
  static const uint32_t code[] = {0x7ee28420u};  // sub d0, d1, d2
  SetV128(&state_, 1, 0x0000000000000005ULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0000000000000008ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFDULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (c) Scalar CMEQ D: equal -> all ones, upper zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmeqScalarD) {
  if (!host_platform::kHasSSE4_1) GTEST_SKIP() << "host lacks SSE4.1; PCMPEQQ bailed";
  static const uint32_t code[] = {0x7ee28c20u};  // cmeq d0, d1, d2
  SetV128(&state_, 1, 0x123456789ABCDEF0ULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x123456789ABCDEF0ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (c) Scalar CMGT D (signed >): 5 > 3 -> all ones.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmgtScalarD) {
  if (!host_platform::kHasSSE4_2) GTEST_SKIP() << "host lacks SSE4.2; PCMPGTQ bailed";
  static const uint32_t code[] = {0x5ee23420u};  // cmgt d0, d1, d2
  SetV128(&state_, 1, 0x0000000000000005ULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0000000000000003ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (c) Scalar CMHI D (unsigned >): 0xFFF..F (unsigned max) > 1 -> all ones.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmhiScalarD) {
  if (!host_platform::kHasSSE4_2) GTEST_SKIP() << "host lacks SSE4.2; PCMPGTQ bailed";
  static const uint32_t code[] = {0x7ee23420u};  // cmhi d0, d1, d2
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0000000000000001ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (c) Scalar CMGE D (signed >=): 3 >= 3 -> all ones.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmgeScalarD) {
  if (!host_platform::kHasSSE4_2) GTEST_SKIP() << "host lacks SSE4.2; PCMPGTQ bailed";
  static const uint32_t code[] = {0x5ee23c20u};  // cmge d0, d1, d2
  SetV128(&state_, 1, 0x0000000000000003ULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0000000000000003ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (c) Scalar CMHS D (unsigned >=): max >= 1 -> all ones.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmhsScalarD) {
  if (!host_platform::kHasSSE4_2) GTEST_SKIP() << "host lacks SSE4.2; PCMPGTQ bailed";
  static const uint32_t code[] = {0x7ee23c20u};  // cmhs d0, d1, d2
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0000000000000001ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (c) Scalar CMTST D: (Vn & Vm) != 0 -> all ones. 0x0F & 0x08 = 0x08 != 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmtstScalarD) {
  if (!host_platform::kHasSSE4_1) GTEST_SKIP() << "host lacks SSE4.1; PCMPEQQ bailed";
  static const uint32_t code[] = {0x5ee28c20u};  // cmtst d0, d1, d2
  SetV128(&state_, 1, 0x000000000000000FULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0000000000000008ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (c) Scalar CMTST D zero case: 0x0F & 0x00 = 0 -> zero.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmtstScalarDZero) {
  if (!host_platform::kHasSSE4_1) GTEST_SKIP() << "host lacks SSE4.1; PCMPEQQ bailed";
  static const uint32_t code[] = {0x5ee28c20u};  // cmtst d0, d1, d2
  SetV128(&state_, 1, 0x000000000000000FULL, 0xCDCDCDCDCDCDCDCDULL);
  SetV128(&state_, 2, 0x0000000000000000ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (d) Scalar ADDP D: Vd.D = Vn.D[0] + Vn.D[1] = 5 + 7 = 0x0C, upper zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddpScalarD) {
  static const uint32_t code[] = {0x5ef1b820u};  // addp d0, v1.2d
  SetV128(&state_, 1, 0x0000000000000005ULL, 0x0000000000000007ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000000CULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (d) Scalar FADDP S: Vn.2s lanes 1.0 + 2.0 = 3.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddpScalarS) {
  static const uint32_t code[] = {0x7e30d820u};  // faddp s0, v1.2s
  SetV128(&state_, 1, 0x400000003F800000ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040400000ULL);  // 3.0 in lane0, [63:32]=0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (d) Scalar FADDP D: Vn.2d lanes 1.0 + 2.0 = 3.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FaddpScalarD) {
  static const uint32_t code[] = {0x7e70d820u};  // faddp d0, v1.2d
  SetV128(&state_, 1, 0x3FF0000000000000ULL, 0x4000000000000000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4008000000000000ULL);  // 3.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (d) Scalar FMAXP S: max(2.0, 3.0) = 3.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxpScalarS) {
  static const uint32_t code[] = {0x7e30f820u};  // fmaxp s0, v1.2s
  SetV128(&state_, 1, 0x4040000040000000ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040400000ULL);  // 3.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (d) Scalar FMINP D: min(5.0, 2.0) = 2.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminpScalarD) {
  static const uint32_t code[] = {0x7ef0f820u};  // fminp d0, v1.2d
  SetV128(&state_, 1, 0x4014000000000000ULL, 0x4000000000000000ULL);  // 5.0, 2.0
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4000000000000000ULL);  // 2.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (d) Scalar FMAXNMP S: NaN-suppressing max(NaN, 2.0) = 2.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmpScalarSNaN) {
  static const uint32_t code[] = {0x7e30c820u};  // fmaxnmp s0, v1.2s
  SetV128(&state_, 1, 0x400000007FC00000ULL, 0xCDCDCDCDCDCDCDCDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040000000ULL);  // 2.0 (NaN suppressed)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (e) Scalar FMUL S by element[0]: 3.0 * 2.0 = 6.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulScalarIdxS) {
  static const uint32_t code[] = {0x5f829020u};  // fmul s0, s1, v2.s[0]
  SetV128(&state_, 1, 0x0000000040400000ULL, 0xCDCDCDCDCDCDCDCDULL);  // 3.0
  SetV128(&state_, 2, 0x0000000040000000ULL, 0xCDCDCDCDCDCDCDCDULL);  // s[0]=2.0
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040C00000ULL);  // 6.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (e) Scalar FMUL D by element[1]: 3.0 * 2.0 = 6.0 (index 1 in high qword).
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulScalarIdxD1) {
  static const uint32_t code[] = {0x5fc29820u};  // fmul d0, d1, v2.d[1]
  SetV128(&state_, 1, 0x4008000000000000ULL, 0xCDCDCDCDCDCDCDCDULL);  // 3.0
  SetV128(&state_, 2, 0xCDCDCDCDCDCDCDCDULL, 0x4000000000000000ULL);  // d[1]=2.0
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4018000000000000ULL);  // 6.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (e) Scalar FMULX S by element[3]: (+0 * +inf) special -> +2.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxScalarIdxSZeroInf) {
  static const uint32_t code[] = {0x7fa29820u};  // fmulx s0, s1, v2.s[3]
  SetV128(&state_, 1, 0x0000000000000000ULL, 0xCDCDCDCDCDCDCDCDULL);  // lane0 = +0.0
  SetV128(&state_, 2, 0xCDCDCDCDCDCDCDCDULL, 0x7F80000000000000ULL);  // s[3]=+inf
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040000000ULL);  // +2.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (e) Scalar FMLA S by element[2]: Vd + Vn*Vm[2] = 1.0 + 3.0*2.0 = 7.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmlaScalarIdxS) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "host lacks FMA; FMLA bailed";
  static const uint32_t code[] = {0x5f821820u};  // fmla s0, s1, v2.s[2]
  SetV128(&state_, 0, 0x000000003F800000ULL, 0xCDCDCDCDCDCDCDCDULL);  // Vd lane0 = 1.0
  SetV128(&state_, 1, 0x0000000040400000ULL, 0xCDCDCDCDCDCDCDCDULL);  // Vn lane0 = 3.0
  SetV128(&state_, 2, 0xCDCDCDCDCDCDCDCDULL, 0x0000000040000000ULL);  // s[2]=2.0 (dword2)
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040E00000ULL);  // 7.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// (e) Scalar FMLS D by element[0]: Vd - Vn*Vm[0] = 10.0 - 3.0*2.0 = 4.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmlsScalarIdxD) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "host lacks FMA; FMLS bailed";
  static const uint32_t code[] = {0x5fc25020u};  // fmls d0, d1, v2.d[0]
  SetV128(&state_, 0, 0x4024000000000000ULL, 0xCDCDCDCDCDCDCDCDULL);  // Vd = 10.0
  SetV128(&state_, 1, 0x4008000000000000ULL, 0xCDCDCDCDCDCDCDCDULL);  // Vn = 3.0
  SetV128(&state_, 2, 0x4000000000000000ULL, 0xCDCDCDCDCDCDCDCDULL);  // d[0]=2.0
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4010000000000000ULL);  // 4.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// CMGT .4S (Q=1): per-lane signed greater-than via PCMPGTD.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmgtVec4S) {
  static const uint32_t code[] = {0x4EA23420u};  // cmgt v0.4s, v1.4s, v2.4s
  SetV128(&state_, 1, 0xFFFFFFFF00000005ULL, 0x0000000700000002ULL);  // [5,-1,2,7]
  SetV128(&state_, 2, 0x0000000300000003ULL, 0xFFFFFFFF00000002ULL);  // [3,3,2,-1]
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);   // 5>3 T, -1>3 F
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFF00000000ULL);  // 2>2 F, 7>-1 T
}

// CMGT .8H (Q=1): per-lane signed greater-than via PCMPGTW.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmgtVec8H) {
  static const uint32_t code[] = {0x4E623420u};  // cmgt v0.8h, v1.8h, v2.8h
  SetV128(&state_, 1, 0x00000064FFFB000AULL, 0x0000000000000000ULL);  // [10,-5,100,0]
  SetV128(&state_, 2, 0x0001006400050005ULL, 0x0000000000000000ULL);  // [5,5,100,1]
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFFULL);   // only lane0 (10>5) true
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// INS (general): insert a GP register into one vector lane, preserving the rest.
TEST_F(Arm64HeavyOptimizerFrontendTest, InsGenS) {
  static const uint32_t code[] = {0x4E141C20u};  // mov v0.s[2], w1
  SetV128(&state_, 0, 0x2222222211111111ULL, 0x4444444433333333ULL);  // [11,22,33,44]
  state_.cpu.x[1] = 0xFFFFFFFFDEADBEEFULL;  // only W1 (low 32) inserted
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x2222222211111111ULL);          // lanes 0,1 preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x44444444DEADBEEFULL);     // lane 2 = W1, lane 3 preserved
}

TEST_F(Arm64HeavyOptimizerFrontendTest, InsGenD) {
  static const uint32_t code[] = {0x4E181C20u};  // mov v0.d[1], x1
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  state_.cpu.x[1] = 0x1122334455667788ULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xAAAAAAAAAAAAAAAAULL);       // low 64 preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x1122334455667788ULL);  // upper 64 = X1
}

TEST_F(Arm64HeavyOptimizerFrontendTest, InsGenB) {
  static const uint32_t code[] = {0x4E0B1C20u};  // mov v0.b[5], w1
  SetV128(&state_, 0, 0x1111111111111111ULL, 0x2222222222222222ULL);
  state_.cpu.x[1] = 0x000000AB;  // low byte inserted
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111AB1111111111ULL);   // byte 5 = 0xAB
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x2222222222222222ULL);  // upper preserved
}

TEST_F(Arm64HeavyOptimizerFrontendTest, InsGenH) {
  static const uint32_t code[] = {0x4E0E1C20u};  // mov v0.h[3], w1
  SetV128(&state_, 0, 0x1111111111111111ULL, 0x2222222222222222ULL);
  state_.cpu.x[1] = 0x0000BEEF;  // low halfword inserted
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xBEEF111111111111ULL);   // halfword 3 = 0xBEEF
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x2222222222222222ULL);  // upper preserved
}

// ADD .8H (Q=1): eight 16-bit lane adds via PADDW.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddVec8H) {
  static const uint32_t code[] = {AddVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0004000300020001ULL, 0x0008000700060005ULL);
  SetV128(&state_, 2, 0x0040003000200010ULL, 0x0080007000600050ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0044003300220011ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0088007700660055ULL);
}

// ADD .2S (Q=0): D-form — two 32-bit adds, upper 64 bits of Vd must be zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddVec2SUpperZero) {
  static const uint32_t code[] = {AddVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000200000001ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x0000002000000010ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000002200000011ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);  // D-form clears the upper 64 bits
}

// SUB .4S (Q=1): four 32-bit lane subtracts via PSUBD.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubVec4S) {
  static const uint32_t code[] = {SubVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000002200000011ULL, 0x0000004400000033ULL);
  SetV128(&state_, 2, 0x0000000200000001ULL, 0x0000000400000003ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000002000000010ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000004000000030ULL);
}

// SUB .2S (Q=0): D-form upper-zero check (with lane borrow producing 0xFFFFFFFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, SubVec2SUpperZero) {
  static const uint32_t code[] = {SubVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000000000005ULL, 0x9999999999999999ULL);
  SetV128(&state_, 2, 0x0000000000000007ULL, 0x8888888888888888ULL);
  SetV128(&state_, 0, 0xEEEEEEEEEEEEEEEEULL, 0xFFFFFFFFFFFFFFFFULL);  // poison upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFEULL);  // 5-7 = -2 in low lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// AND .16B (Q=1): full 128-bit bitwise AND (element-size independent).
TEST_F(Arm64HeavyOptimizerFrontendTest, AndVec16B) {
  static const uint32_t code[] = {AndVec(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFF00FF00FF00FF00ULL, 0x0F0F0F0F0F0F0F0FULL);
  SetV128(&state_, 2, 0x0FF00FF00FF00FF0ULL, 0xFFFF0000FFFF0000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF00FF00FF00FF00ULL & 0x0FF00FF00FF00FF0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0F0F0F0F0F0F0F0FULL & 0xFFFF0000FFFF0000ULL);
}

// AND .8B (Q=0): D-form upper-zero check for a bitwise op.
TEST_F(Arm64HeavyOptimizerFrontendTest, AndVec8BUpperZero) {
  static const uint32_t code[] = {AndVec(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0xFF00FF00FF00FF00ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x0FF00FF00FF00FF0ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xABABABABABABABABULL, 0xCDCDCDCDCDCDCDCDULL);  // poison upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF00FF00FF00FF00ULL & 0x0FF00FF00FF00FF0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// ORR .16B (Q=1): full 128-bit bitwise OR.
TEST_F(Arm64HeavyOptimizerFrontendTest, OrrVec16B) {
  static const uint32_t code[] = {OrrVec(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFF00FF00FF00FF00ULL, 0x0F0F0F0F0F0F0F0FULL);
  SetV128(&state_, 2, 0x00FF00FF00FF00FFULL, 0xF0F0F0F0F0F0F0F0ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
}

// ORR with rn==rm is the AdvSIMD MOV (vector) alias: Vd = Vn. The .8B (Q=0)
// form must zero the upper 64 bits.
TEST_F(Arm64HeavyOptimizerFrontendTest, OrrVecMovAlias8B) {
  static const uint32_t code[] = {OrrVec(/*q=*/false, 0, 1, 1)};  // MOV V0.8B, V1.8B
  SetV128(&state_, 1, 0x0102030405060708ULL, 0x1112131415161718ULL);
  SetV128(&state_, 0, 0x9999999999999999ULL, 0x8888888888888888ULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0102030405060708ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// EOR .16B (Q=1): full 128-bit bitwise XOR.
TEST_F(Arm64HeavyOptimizerFrontendTest, EorVec16B) {
  static const uint32_t code[] = {EorVec(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFF00FF00FF00FF00ULL, 0x0F0F0F0F0F0F0F0FULL);
  SetV128(&state_, 2, 0xFFFFFFFFFFFFFFFFULL, 0x00FF00FF00FF00FFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF00FF00FF00FF00ULL ^ 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0F0F0F0F0F0F0F0FULL ^ 0x00FF00FF00FF00FFULL);
}

// BSL .16B (Q=1): Vd is the select mask — result bit = Vd ? Vn : Vm.
TEST_F(Arm64HeavyOptimizerFrontendTest, BslVec16B) {
  static const uint32_t code[] = {BslVec(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAAAAULL, 0x00000000FFFFFFFFULL);  // Vn
  SetV128(&state_, 2, 0x5555555555555555ULL, 0xFFFFFFFF00000000ULL);  // Vm
  SetV128(&state_, 0, 0xFF00FF00FF00FF00ULL, 0x0F0F0F0F0F0F0F0FULL);  // Vd = mask
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (Vd & Vn) | (~Vd & Vm)
  EXPECT_EQ(VLo64(&state_, 0), 0xAA55AA55AA55AA55ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xF0F0F0F00F0F0F0FULL);
}

// BIT .16B (Q=1): "insert if true" — where Vm=1 take Vn, else keep Vd.
TEST_F(Arm64HeavyOptimizerFrontendTest, BitVec16B) {
  static const uint32_t code[] = {BitVec(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAAAAULL, 0x1234567812345678ULL);  // Vn
  SetV128(&state_, 2, 0xFF00FF00FF00FF00ULL, 0x00000000FFFFFFFFULL);  // Vm = mask
  SetV128(&state_, 0, 0x5555555555555555ULL, 0xCCCCCCCCCCCCCCCCULL);  // Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (Vm & Vn) | (~Vm & Vd)
  EXPECT_EQ(VLo64(&state_, 0), 0xAA55AA55AA55AA55ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xCCCCCCCC12345678ULL);
}

// BIF .16B (Q=1): "insert if false" — where Vm=0 take Vn, else keep Vd. Exercises
// the PANDN fold and the res=xd store path.
TEST_F(Arm64HeavyOptimizerFrontendTest, BifVec16B) {
  static const uint32_t code[] = {BifVec(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAAAAULL, 0x1234567812345678ULL);  // Vn
  SetV128(&state_, 2, 0xFF00FF00FF00FF00ULL, 0x00000000FFFFFFFFULL);  // Vm = mask
  SetV128(&state_, 0, 0x5555555555555555ULL, 0xCCCCCCCCCCCCCCCCULL);  // Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (Vm & Vd) | (~Vm & Vn)
  EXPECT_EQ(VLo64(&state_, 0), 0x55AA55AA55AA55AAULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x12345678CCCCCCCCULL);
}

// BSL .8B (Q=0): D-form upper-zero check for the res=xn store path.
TEST_F(Arm64HeavyOptimizerFrontendTest, BslVec8BUpperZero) {
  static const uint32_t code[] = {BslVec(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAAAAULL, 0x1111111111111111ULL);  // Vn
  SetV128(&state_, 2, 0x5555555555555555ULL, 0x2222222222222222ULL);  // Vm
  SetV128(&state_, 0, 0xFF00FF00FF00FF00ULL, 0xCDCDCDCDCDCDCDCDULL);  // Vd = mask, poison upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xAA55AA55AA55AA55ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// BIF .8B (Q=0): D-form upper-zero check for the res=xd store path.
TEST_F(Arm64HeavyOptimizerFrontendTest, BifVec8BUpperZero) {
  static const uint32_t code[] = {BifVec(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAAAAULL, 0x1111111111111111ULL);  // Vn
  SetV128(&state_, 2, 0xFF00FF00FF00FF00ULL, 0x2222222222222222ULL);  // Vm = mask
  SetV128(&state_, 0, 0x5555555555555555ULL, 0xCDCDCDCDCDCDCDCDULL);  // Vd, poison upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x55AA55AA55AA55AAULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// MUL .4S (Q=1): four 32-bit lane products via PMULLD.
TEST_F(Arm64HeavyOptimizerFrontendTest, MulVec4S) {
  static const uint32_t code[] = {MulVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000300000002ULL, 0x0000000500000004ULL);
  SetV128(&state_, 2, 0x0000000700000006ULL, 0x0000000900000008ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lanes: 2*6=12 (0xC), 3*7=21 (0x15), 4*8=32 (0x20), 5*9=45 (0x2D)
  EXPECT_EQ(VLo64(&state_, 0), 0x000000150000000CULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000002D00000020ULL);
}

// MUL .8H (Q=1): eight 16-bit lane products via PMULLW (low 16 bits per lane).
TEST_F(Arm64HeavyOptimizerFrontendTest, MulVec8H) {
  static const uint32_t code[] = {MulVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0004000300020001ULL, 0x0008000700060005ULL);
  SetV128(&state_, 2, 0x0002000200020002ULL, 0x0002000200020002ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0008000600040002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0010000E000C000AULL);
}

// MUL .2S (Q=0): D-form upper-zero check.
TEST_F(Arm64HeavyOptimizerFrontendTest, MulVec2SUpperZero) {
  static const uint32_t code[] = {MulVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000300000002ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x0000000700000006ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xABABABABABABABABULL, 0xCDCDCDCDCDCDCDCDULL);  // poison upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000150000000CULL);  // 2*6=12, 3*7=21
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// Multi-instruction integer-SIMD region: a chain of ADD/SUB/MUL/EOR across
// several V registers, exercising store-to-load forwarding of a just-written
// V register inside one JIT region.
TEST_F(Arm64HeavyOptimizerFrontendTest, IntSimdMultiInstructionRegion) {
  static const uint32_t code[] = {
      AddVec(0b10, /*q=*/true, 0, 1, 2),  // V0.4S = V1 + V2
      MulVec(0b10, /*q=*/true, 0, 0, 3),  // V0.4S = V0 * V3
      EorVec(/*q=*/true, 4, 0, 5),        // V4.16B = V0 ^ V5
  };
  SetV128(&state_, 1, 0x0000000200000001ULL, 0x0000000400000003ULL);
  SetV128(&state_, 2, 0x0000000200000003ULL, 0x0000000400000005ULL);  // V1+V2 lanes: 4,4,8,8
  SetV128(&state_, 3, 0x0000000200000002ULL, 0x0000000200000002ULL);  // *2 lanes: 8,8,16,16
  SetV128(&state_, 5, 0ULL, 0ULL);                                    // XOR 0 = identity
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 4), 0x0000000800000008ULL);
  EXPECT_EQ(VUpperHi64(&state_, 4), 0x0000001000000010ULL);
}

// CMEQ .2D (64-bit elements) must bail: PCMPEQQ is not in the backend allowlist.
// (The B/H/S forms are now lowered via PCMPEQB/W/D — see CmeqVec4S/16B.)
// CMEQ .2D translates via PCMPEQQ (used to bail before the 64-bit compare
// forms were lowered).
TEST_F(Arm64HeavyOptimizerFrontendTest, CmeqVec2D) {
  static const uint32_t code[] = {CmeqVec(0b11, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0123456789ABCDEFULL, 0x8000000000000000ULL);
  SetV128(&state_, 2, 0x0123456789ABCDEFULL, 0x8000000000000001ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), ~0ULL);       // equal lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ULL);   // differing lane
}

// MUL .2D (64-bit elements) must bail: there is no packed 64-bit multiply.
TEST_F(Arm64HeavyOptimizerFrontendTest, MulVec2DBails) {
  static const uint32_t code[] = {MulVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// ADD .2D (Q=1): two 64-bit lanes via Paddq; low lane exercises 64-bit wrap.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddVec2D) {
  static const uint32_t code[] = {AddVec(0b11, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0x00000000FFFFFFFFULL);
  SetV128(&state_, 2, 0x0000000000000002ULL, 0x0000000000000001ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000001ULL);   // wraps within the lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000100000000ULL);
}

// ADD .16B (Q=1): per-byte add via Paddb; low LSB byte exercises byte wrap.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddVec16B) {
  static const uint32_t code[] = {AddVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x01020304050607FFULL, 0x1112131415161718ULL);
  SetV128(&state_, 2, 0x1010101010101002ULL, 0x2020202020202020ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1112131415161701ULL);   // 0xFF+0x02 -> 0x01, no carry-out
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x3132333435363738ULL);
}

// ADD .8B (Q=0): D-form upper-zero check for Paddb.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddVec8BUpperZero) {
  static const uint32_t code[] = {AddVec(0b00, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x0102030405060708ULL, 0x9999999999999999ULL);
  SetV128(&state_, 2, 0x1010101010101010ULL, 0x8888888888888888ULL);
  SetV128(&state_, 0, 0xEEEEEEEEEEEEEEEEULL, 0xFFFFFFFFFFFFFFFFULL);  // poison upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1112131415161718ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// SUB .16B (Q=1): per-byte sub via Psubb; low LSB byte exercises byte borrow.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubVec16B) {
  static const uint32_t code[] = {SubVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x1112131415161700ULL, 0x3132333435363738ULL);
  SetV128(&state_, 2, 0x1010101010101001ULL, 0x2020202020202020ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x01020304050607FFULL);   // 0x00-0x01 -> 0xFF, no borrow-out
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x1112131415161718ULL);
}

// SUB .8H (Q=1): per-halfword sub via Psubw; low lane exercises halfword borrow.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubVec8H) {
  static const uint32_t code[] = {SubVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x000A000B000C0000ULL, 0x0014001300120011ULL);
  SetV128(&state_, 2, 0x0001000200030001ULL, 0x0004000300020001ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000900090009FFFFULL);   // 0x0000-0x0001 -> 0xFFFF
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0010001000100010ULL);
}

// SUB .2D (Q=1): two 64-bit lanes via Psubq; low lane exercises 64-bit borrow.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubVec2D) {
  static const uint32_t code[] = {SubVec(0b11, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000000000000ULL, 0x0000000100000000ULL);
  SetV128(&state_, 2, 0x0000000000000001ULL, 0x0000000000000001ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);   // 0 - 1 wraps within the lane
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFFULL);
}

// MLA .4S (Q=1): Vd += Vn*Vm per 32-bit lane via Pmulld + Paddd.
TEST_F(Arm64HeavyOptimizerFrontendTest, MlaVec4S) {
  static const uint32_t code[] = {MlaVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000300000002ULL, 0x0000000500000004ULL);
  SetV128(&state_, 2, 0x0000000700000006ULL, 0x0000000900000008ULL);
  SetV128(&state_, 0, 0x0000000000000010ULL, 0x0000010000000100ULL);  // accumulator
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lane0: 0x10 + 2*6=12 -> 0x1C; lane1: 0 + 3*7=21 -> 0x15
  EXPECT_EQ(VLo64(&state_, 0), 0x000000150000001CULL);
  // lane2: 0x100 + 4*8=32 -> 0x120; lane3: 0x100 + 5*9=45 -> 0x12D
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000012D00000120ULL);
}

// MLA .8H (Q=1): eight 16-bit lane accumulates via Pmullw + Paddw (low 16 bits).
TEST_F(Arm64HeavyOptimizerFrontendTest, MlaVec8H) {
  static const uint32_t code[] = {MlaVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0004000300020001ULL, 0x0008000700060005ULL);
  SetV128(&state_, 2, 0x0002000200020002ULL, 0x0002000200020002ULL);
  SetV128(&state_, 0, 0x0001000100010001ULL, 0x0001000100010001ULL);  // accumulator
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products 2,4,6,8 / 10,12,14,16 plus 1 each
  EXPECT_EQ(VLo64(&state_, 0), 0x0009000700050003ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0011000F000D000BULL);
}

// MLS .4S (Q=1): Vd -= Vn*Vm per 32-bit lane via Pmulld + Psubd.
TEST_F(Arm64HeavyOptimizerFrontendTest, MlsVec4S) {
  static const uint32_t code[] = {MlsVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000300000002ULL, 0x0000000500000004ULL);
  SetV128(&state_, 2, 0x0000000700000006ULL, 0x0000000900000008ULL);
  SetV128(&state_, 0, 0x0000010000000100ULL, 0x0000020000000200ULL);  // accumulator
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lane0: 0x100 - 12 -> 0xF4; lane1: 0x100 - 21 -> 0xEB
  EXPECT_EQ(VLo64(&state_, 0), 0x000000EB000000F4ULL);
  // lane2: 0x200 - 32 -> 0x1E0; lane3: 0x200 - 45 -> 0x1D3
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000001D3000001E0ULL);
}

// MLS .8H (Q=1): halfword subtract-accumulate; low lane exercises 16-bit borrow.
TEST_F(Arm64HeavyOptimizerFrontendTest, MlsVec8H) {
  static const uint32_t code[] = {MlsVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0004000300020001ULL, 0x0008000700060005ULL);
  SetV128(&state_, 2, 0x0002000200020002ULL, 0x0002000200020002ULL);
  SetV128(&state_, 0, 0x0000000000000000ULL, 0x0020002000200020ULL);  // accumulator
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products 2,4,6,8; low lanes 0-{2,4,6,8} wrap to 0xFFFE/FFFC/FFFA/FFF8
  EXPECT_EQ(VLo64(&state_, 0), 0xFFF8FFFAFFFCFFFEULL);
  // high lanes 0x20 - {10,12,14,16} -> 0x16,0x14,0x12,0x10
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0010001200140016ULL);
}

// MLA .2S (Q=0): D-form upper-zero check for the accumulate path.
TEST_F(Arm64HeavyOptimizerFrontendTest, MlaVec2SUpperZero) {
  static const uint32_t code[] = {MlaVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000300000002ULL, 0x1111111111111111ULL);
  SetV128(&state_, 2, 0x0000000700000006ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0x0000010000000100ULL, 0xCDCDCDCDCDCDCDCDULL);  // poison upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lane0: 0x100 + 12 -> 0x10C; lane1: 0x100 + 21 -> 0x115
  EXPECT_EQ(VLo64(&state_, 0), 0x000001150000010CULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// MLA .2D (64-bit elements) must bail: there is no packed 64-bit multiply.
TEST_F(Arm64HeavyOptimizerFrontendTest, MlaVec2DBails) {
  static const uint32_t code[] = {MlaVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// MLS .16B (byte elements): now lowered in-tier via the PMOVZXBW/PMULLW/PACKUSWB
// byte-multiply recipe (region translates rather than bailing).
TEST_F(Arm64HeavyOptimizerFrontendTest, MlsVec16BLowered) {
  static const uint32_t code[] = {MlsVec(0b00, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 1u);
}

// SQADD .16B (Q=1): per-byte signed saturating add to [-128,127]. Mix of
// non-saturating and saturating lanes (0x7F+1 -> 0x7F, 0x80+0x80 -> 0x80).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddVec16B) {
  static const uint32_t code[] = {SqaddVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x7F7F7F7F10101010ULL, 0x8080808040404040ULL);  // Vn
  SetV128(&state_, 2, 0x0101010105050505ULL, 0x80808080C0C0C0C0ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7F7F7F7F15151515ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x8080808000000000ULL);
}

// SQADD .4S (Q=1): 32-bit signed saturating add via the emulation path. Lanes
// exercise no-overflow, positive-overflow (INT_MAX+1), and negative-overflow
// (INT_MIN+-1).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddVec4S) {
  static const uint32_t code[] = {SqaddVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x7FFFFFFF00000005ULL, 0x8000000012345678ULL);  // Vn
  SetV128(&state_, 2, 0x00000001FFFFFFFBULL, 0xFFFFFFFF00000008ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFF00000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x8000000012345680ULL);
}

// UQADD .16B (Q=1): per-byte unsigned saturating add to 255.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqaddVec16B) {
  static const uint32_t code[] = {UqaddVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFF10101010ULL, 0x8080808080808080ULL);  // Vn
  SetV128(&state_, 2, 0x0101010120202020ULL, 0x8080808080808080ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF30303030ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
}

// UQADD .4S (Q=1): 32-bit unsigned saturating add via PMAXUD overflow detect.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqaddVec4S) {
  static const uint32_t code[] = {UqaddVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFF00000010ULL, 0x800000007FFFFFFFULL);  // Vn
  SetV128(&state_, 2, 0x0000000500000020ULL, 0x8000000000000001ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF00000030ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFF80000000ULL);
}

// SQSUB .8H (Q=1): per-halfword signed saturating subtract.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqsubVec8H) {
  static const uint32_t code[] = {SqsubVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x7FFF000080001000ULL, 0x80007FFF00000000ULL);  // Vn
  SetV128(&state_, 2, 0x0001000100012000ULL, 0x8000800000000000ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFEFFFF8000F000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00007FFF00000000ULL);
}

// SQSUB .4S (Q=1): 32-bit signed saturating subtract via the emulation path.
// Lanes: no-overflow, negative-overflow (INT_MIN-1), positive-overflow
// (INT_MAX--1).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqsubVec4S) {
  static const uint32_t code[] = {SqsubVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0x7FFFFFFF12345678ULL);  // Vn
  SetV128(&state_, 2, 0x0000000100000005ULL, 0xFFFFFFFF00000008ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x80000000FFFFFFFBULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x7FFFFFFF12345670ULL);
}

// UQSUB .16B (Q=1): per-byte unsigned saturating subtract to 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqsubVec16B) {
  static const uint32_t code[] = {UqsubVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x8040201008040201ULL, 0xFFFFFFFFFFFFFFFFULL);  // Vn
  SetV128(&state_, 2, 0x0102040810204080ULL, 0x0102030405060708ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7F3E1C0800000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFEFDFCFBFAF9F8F7ULL);
}

// UQSUB .4S (Q=1): 32-bit unsigned saturating subtract via PMINUD mask.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqsubVec4S) {
  static const uint32_t code[] = {UqsubVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000500000010ULL, 0xFFFFFFFF80000000ULL);  // Vn
  SetV128(&state_, 2, 0x0000000A00000003ULL, 0x0000000180000001ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000000DULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFE00000000ULL);
}

// SQADD .2S (Q=0): the 32-bit emulation path must zero Vd[127:64] via the
// D-form SetVRegFull merge.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddVec2SUpperZero) {
  static const uint32_t code[] = {SqaddVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x7FFFFFFF00000005ULL, 0x1111111111111111ULL);  // Vn (upper ignored)
  SetV128(&state_, 2, 0x00000001FFFFFFFBULL, 0x2222222222222222ULL);  // Vm (upper ignored)
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0xCDCDCDCDCDCDCDCDULL);  // poison Vd upper
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFF00000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);  // upper 64 zeroed
}

// SQADD .2D (size=11, 64-bit) has no packed saturating qword path in either
// tier and must bail to lite (which routes it to the interpreter).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddVec2DBails) {
  static const uint32_t code[] = {SqaddVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SMAX .16B (size=00, Q=1): per-byte signed max. Needs SSE4.1 (PMAXSB).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmaxVec16B) {
  static const uint32_t code[] = {SmaxVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x7F8001FF10F00005ULL, 0x807F00FE01020304ULL);  // Vn
  SetV128(&state_, 2, 0x01FF7F80F0100500ULL, 0x7F80FF0004030201ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFF7FFF10100505ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x7F7F000004030304ULL);
}

// SMIN .8H (size=01, Q=1): per-halfword signed min. SSE2 (PMINSW).
TEST_F(Arm64HeavyOptimizerFrontendTest, SminVec8H) {
  static const uint32_t code[] = {SminVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x7FFF80000001FFFFULL, 0x123456789ABCDEF0ULL);  // Vn
  SetV128(&state_, 2, 0x0001FFFF7FFF8000ULL, 0x0000111122223333ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0001800000018000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000011119ABCDEF0ULL);
}

// UMAX .4S (size=10, Q=1): per-dword unsigned max. Needs SSE4.1 (PMAXUD).
TEST_F(Arm64HeavyOptimizerFrontendTest, UmaxVec4S) {
  static const uint32_t code[] = {UmaxVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFF00000001ULL, 0x8000000012345678ULL);  // Vn
  SetV128(&state_, 2, 0x0000000200000000ULL, 0x7FFFFFFF12345679ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF00000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x8000000012345679ULL);
}

// UMIN .16B (size=00, Q=1): per-byte unsigned min. SSE2 (PMINUB).
TEST_F(Arm64HeavyOptimizerFrontendTest, UminVec16B) {
  static const uint32_t code[] = {UminVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFF00807F01FE1020ULL, 0x0102030405060708ULL);  // Vn
  SetV128(&state_, 2, 0x00FF7F80FE012010ULL, 0x0807060504030201ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00007F7F01011010ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0102030404030201ULL);
}

// SMAX .2S (size=10, Q=0): per-dword signed max, upper 64 bits zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, SmaxVec2SUpperZero) {
  static const uint32_t code[] = {SmaxVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x7FFFFFFF80000000ULL, 0xAAAAAAAABBBBBBBBULL);  // Vn
  SetV128(&state_, 2, 0x00000001FFFFFFFFULL, 0xCCCCCCCCDDDDDDDDULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SMAX .2D (size=11, 64-bit) has no packed SSE min/max qword op and must bail
// to lite (which routes it to the interpreter).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmaxVec2DBails) {
  static const uint32_t code[] = {SmaxVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SABD .16B (size=00, Q=1): per-byte signed absolute difference. Needs SSE4.1
// (PMAXSB/PMINSB). Recipe is max(a,b) - min(a,b) on the signed interpretation.
TEST_F(Arm64HeavyOptimizerFrontendTest, SabdVec16B) {
  static const uint32_t code[] = {SabdVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x7F8001FF10F00005ULL, 0x807F00FE01020304ULL);  // Vn
  SetV128(&state_, 2, 0x01FF7F80F0100500ULL, 0x7F80FF0004030201ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7E7F7E7F20200505ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFF010203010103ULL);
}

// PMUL .16B (size=00, Q=1): per-byte polynomial (carry-less GF(2)) multiply,
// low 8 bits. Both 64-bit halves active. Expected values computed by the
// reference carry-less byte multiply (0xFF*0xFF -> 0x55, etc.).
TEST_F(Arm64HeavyOptimizerFrontendTest, PmulVec16B) {
  static const uint32_t code[] = {PmulVec(/*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFF807F0102030405ULL, 0x1122334455667788ULL);  // Vn
  SetV128(&state_, 2, 0xFF01FF80F0110503ULL, 0x8899AABBCCDDEEFFULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x5580D580E033140FULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x08121E2C3C2E2A78ULL);
}

// PMUL .8B (size=00, Q=0): D-form — only the low 64 bits are computed and
// Vd[127:64] is zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, PmulVec8B) {
  static const uint32_t code[] = {PmulVec(/*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0xFF807F0102030405ULL, 0x1122334455667788ULL);  // Vn
  SetV128(&state_, 2, 0xFF01FF80F0110503ULL, 0x8899AABBCCDDEEFFULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x5580D580E033140FULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);  // upper 64 bits zeroed
}

// UABD .8H (size=01, Q=1): per-halfword unsigned absolute difference. Needs
// SSE4.1 (PMAXUW/PMINUW).
TEST_F(Arm64HeavyOptimizerFrontendTest, UabdVec8H) {
  static const uint32_t code[] = {UabdVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFF000180007FFFULL, 0x123456789ABCDEF0ULL);  // Vn
  SetV128(&state_, 2, 0x0001FFFF7FFF8000ULL, 0x0000111122223333ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFEFFFE00010001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x12344567789AABBDULL);
}

// SABA .4S (size=10, Q=1): per-dword signed abs-diff accumulated into Vd. Needs
// SSE4.1 (PMAXSD/PMINSD). Vd is read and PADDD'd with the abs-diff.
TEST_F(Arm64HeavyOptimizerFrontendTest, SabaVec4S) {
  static const uint32_t code[] = {SabaVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x7FFFFFFF80000000ULL, 0x0000000A00000064ULL);  // Vn
  SetV128(&state_, 2, 0x0000000100000005ULL, 0xFFFFFFFF00000032ULL);  // Vm
  SetV128(&state_, 0, 0x0000000100000002ULL, 0x0000000300000004ULL);  // Vd accum
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFF80000007ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000E00000036ULL);
}

// UABA .16B (size=00, Q=1): per-byte unsigned abs-diff accumulated into Vd. SSE2
// (PMAXUB/PMINUB + PADDB).
TEST_F(Arm64HeavyOptimizerFrontendTest, UabaVec16B) {
  static const uint32_t code[] = {UabaVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFF00807F01FE1020ULL, 0x0102030405060708ULL);  // Vn
  SetV128(&state_, 2, 0x00FF7F80FE012010ULL, 0x0807060504030201ULL);  // Vm
  SetV128(&state_, 0, 0x0101010101010101ULL, 0x1010101010101010ULL);  // Vd accum
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000202FEFE1111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x1715131111131517ULL);
}

// SABD .2S (size=10, Q=0): per-dword signed abs-diff, upper 64 bits zeroed.
// Exercises the INT_MIN-vs-INT_MAX wrap where a naive PSUB+sign-mask abs fails
// but max-minus-min is correct.
TEST_F(Arm64HeavyOptimizerFrontendTest, SabdVec2SUpperZero) {
  static const uint32_t code[] = {SabdVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x7FFFFFFF80000000ULL, 0xAAAAAAAABBBBBBBBULL);  // Vn
  SetV128(&state_, 2, 0x00000001FFFFFFFFULL, 0xCCCCCCCCDDDDDDDDULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFE7FFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SABD .2D (size=11) has no packed SSE min/max qword op and must bail to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, SabdVec2DBails) {
  static const uint32_t code[] = {SabdVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// Halving add/sub. Expected constants generated by an independent ARM-semantics
// reference (floor-division by 2). See lite_translator.h for the recipes.

// SHADD .8H (size=01, Q=1): signed halving add via the bitwise identity
// (a&b)+((a^b)>>1) with arithmetic shift. Includes INT16 corner lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, ShaddVec8H) {
  static const uint32_t code[] = {ShaddVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x00040002FFFF0006ULL, 0x7FFF800000030005ULL);  // Vn
  SetV128(&state_, 2, 0x00010003FFFD0002ULL, 0x7FFF800000000005ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00020002FFFE0004ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x7FFF800000010005ULL);
}

// UHADD .16B (size=00, Q=1): unsigned halving add via the widen/PADDW/PSRLW/
// PACKUSWB byte path. Needs SSE4.1 (PMOVZXBW).
TEST_F(Arm64HeavyOptimizerFrontendTest, UhaddVec16B) {
  static const uint32_t code[] = {UhaddVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x03FE7F80011000FFULL, 0x0807060504030201ULL);  // Vn
  SetV128(&state_, 2, 0x050201800220FFFFULL, 0x0102030405060708ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0480408001187FFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0404040404040404ULL);
}

// UHADD .4S (size=10, Q=1): unsigned halving add via the bitwise word path.
// Lane 1 (0xFFFFFFFF + 2) proves the bitwise identity avoids the PADDD carry
// overflow that a naive add would lose.
TEST_F(Arm64HeavyOptimizerFrontendTest, UhaddVec4S) {
  static const uint32_t code[] = {UhaddVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFF00000005ULL, 0x800000007FFFFFFFULL);  // Vn
  SetV128(&state_, 2, 0x0000000200000002ULL, 0x8000000180000000ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x8000000000000003ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x800000007FFFFFFFULL);
}

// SRHADD .4S (size=10, Q=1): signed rounding halving add via the bitwise
// identity (a|b)-((a^b)>>1) with arithmetic shift. INT_MIN/INT_MAX corners.
TEST_F(Arm64HeavyOptimizerFrontendTest, SrhaddVec4S) {
  static const uint32_t code[] = {SrhaddVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFF00000005ULL, 0x800000007FFFFFFFULL);  // Vn
  SetV128(&state_, 2, 0xFFFFFFFF00000002ULL, 0x800000007FFFFFFFULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF00000004ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x800000007FFFFFFFULL);
}

// URHADD .8B (size=00, Q=0): unsigned rounding halving add via the widen byte
// path (the +1 materialized by PCMPEQW+PSUBW). Upper 64 bits zeroed by Q=0.
TEST_F(Arm64HeavyOptimizerFrontendTest, UrhaddVec8B) {
  static const uint32_t code[] = {UrhaddVec(0b00, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x05FE037F100100FFULL, 0xDEADBEEFCAFEF00DULL);  // Vn (hi unused)
  SetV128(&state_, 2, 0x05010480200200FFULL, 0x0123456789ABCDEFULL);  // Vm (hi unused)
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x05800480180200FFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SHSUB .8H (size=01, Q=1): signed halving sub via widen-to-dword, PSUBD,
// PSRAD, PACKSSDW. Lanes 2/3 span the full INT16 range (needs 32-bit widening).
TEST_F(Arm64HeavyOptimizerFrontendTest, ShsubVec8H) {
  static const uint32_t code[] = {ShsubVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x80007FFF0004000AULL, 0x00000064FFFF0005ULL);  // Vn
  SetV128(&state_, 2, 0x7FFF8000000A0004ULL, 0x0001000100010005ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x80007FFFFFFD0003ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFF0031FFFF0000ULL);
}

// UHSUB .16B (size=00, Q=1): unsigned halving sub via widen byte, PSUBW,
// PSRLW, low-byte mask, PACKUSWB. Exercises the modular a<b lanes that the
// mask keeps from saturating.
TEST_F(Arm64HeavyOptimizerFrontendTest, UhsubVec16B) {
  static const uint32_t code[] = {UhsubVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x03FE05018010FF00ULL, 0xAA807F10F0020040ULL);  // Vn
  SetV128(&state_, 2, 0x08020580012000FFULL, 0x557F80F010000140ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFD7E00C03FF87F80ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x2A00FF907001FF00ULL);
}

// UHSUB .4S (size=10, Q=1): unsigned halving sub via widen-to-qword, PSUBQ,
// PSRLQ, PSHUFD 0x88 gather + PUNPCKLQDQ. Lanes 2/3 span the full UINT32 range.
TEST_F(Arm64HeavyOptimizerFrontendTest, UhsubVec4S) {
  static const uint32_t code[] = {UhsubVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000500000064ULL, 0x00000001FFFFFFFFULL);  // Vn
  SetV128(&state_, 2, 0x0000006400000005ULL, 0xFFFFFFFF00000002ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFD00000002FULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x800000017FFFFFFEULL);
}

// SHADD .2D (size=11) is reserved by the ARM ARM and must bail to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, ShaddVec2DBails) {
  static const uint32_t code[] = {ShaddVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SQDMULH .8H (size=01, Q=1): per-halfword saturating doubling multiply-high.
// Includes the INT16_MIN*INT16_MIN corner (lane -0x8000*-0x8000 -> 0x7FFF).
// Expected values computed against the ARM semantics reference.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmulhVec8H) {
  static const uint32_t code[] = {SqdmulhVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x8000000140007FFFULL, 0x0102030400020003ULL);  // Vn
  SetV128(&state_, 2, 0x80007FFF40000002ULL, 0x7FFF400000010002ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFF000020000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0101018200000000ULL);
}

// SQRDMULH .8H (size=01, Q=1): rounding variant (PMULHRSW + corner fixup).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrdmulhVec8H) {
  static const uint32_t code[] = {SqrdmulhVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x8000000140007FFFULL, 0x0102030400020003ULL);  // Vn
  SetV128(&state_, 2, 0x80007FFF40000002ULL, 0x7FFF400000010002ULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFF000120000002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0102018200000000ULL);
}

// SQDMULH .4S (size=10, Q=1): per-dword form (PMULDQ widen). Needs SSE4.1.
// Includes the INT32_MIN*INT32_MIN corner (-0x80000000^2 -> 0x7FFFFFFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmulhVec4S) {
  static const uint32_t code[] = {SqdmulhVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x800000017FFFFFFFULL, 0x0000000A00000064ULL);  // Vn
  SetV128(&state_, 2, 0x800000027FFFFFFFULL, 0x00000005FFFFFFFFULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFD7FFFFFFEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFFULL);
}

// SQRDMULH .4S (size=10, Q=1): rounding variant (+2^31 per lane).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrdmulhVec4S) {
  static const uint32_t code[] = {SqrdmulhVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x800000017FFFFFFFULL, 0x0000000A00000064ULL);  // Vn
  SetV128(&state_, 2, 0x800000027FFFFFFFULL, 0x00000005FFFFFFFFULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFD7FFFFFFEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SQDMULH .2S (size=10, Q=0): lower 64 bits only; upper 64 zeroed by the Q=0
// D-form merge.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmulhVec2SUpperZero) {
  static const uint32_t code[] = {SqdmulhVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x800000017FFFFFFFULL, 0x0000000A00000064ULL);  // Vn
  SetV128(&state_, 2, 0x800000027FFFFFFFULL, 0x00000005FFFFFFFFULL);  // Vm
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFD7FFFFFFEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SQDMULH .2D (size=11) is reserved by the decoder and must bail (0 insns).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmulhVec2DBails) {
  static const uint32_t code[] = {SqdmulhVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// ---- Scalar saturating add/sub (SQADD/UQADD/SQSUB/UQSUB, B/H/S/D) ----
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddScalarBSat) {
  static const uint32_t code[] = {SqaddScalar(0b00, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAA7FULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBBBBBBBB01ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000007FULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UqaddScalarBSat) {
  static const uint32_t code[] = {UqaddScalar(0b00, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAAFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBBBBBBBB01ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000000000FFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddScalarHSat) {
  static const uint32_t code[] = {SqaddScalar(0b01, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAAAAA7FFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBBBBBB0001ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000007FFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddScalarSSat) {
  static const uint32_t code[] = {SqaddScalar(0b10, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA7FFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB00000001ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000007FFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UqaddScalarSSat) {
  static const uint32_t code[] = {UqaddScalar(0b10, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAAFFFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB00000001ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddScalarDSat) {
  static const uint32_t code[] = {SqaddScalar(0b11, 0, 1, 2)};
  SetV128(&state_, 1, 0x7FFFFFFFFFFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000001ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UqaddScalarDSat) {
  static const uint32_t code[] = {UqaddScalar(0b11, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000001ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SqsubScalarDSat) {
  static const uint32_t code[] = {SqsubScalar(0b11, 0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000001ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x8000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UqsubScalarSUnderflow) {
  static const uint32_t code[] = {UqsubScalar(0b10, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA00000003ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB00000007ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UqsubScalarDInRange) {
  static const uint32_t code[] = {UqsubScalar(0b11, 0, 1, 2)};
  SetV128(&state_, 1, 0x000000000000000AULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000005ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x5ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddScalarDInRange) {
  static const uint32_t code[] = {SqaddScalar(0b11, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000000000003ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000004ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// ---- Scalar shifts (D-form): SSHL/USHL/SRSHL/URSHL ----
TEST_F(Arm64HeavyOptimizerFrontendTest, UshlScalarDNegRight) {
  static const uint32_t code[] = {UshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0xFF00000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000FCULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0FF0000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UshlScalarDLeft) {
  static const uint32_t code[] = {UshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x0000000000000001ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000004ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x10ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UshlScalarDBigLeftZero) {
  static const uint32_t code[] = {UshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x00000000000000FFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000040ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UshlScalarDBigRightZero) {
  static const uint32_t code[] = {UshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000C0ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SshlScalarDNegArith) {
  static const uint32_t code[] = {SshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000FFULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SshlScalarDBigRightSignBcast) {
  static const uint32_t code[] = {SshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000C0ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SshlScalarDBigLeftZero) {
  static const uint32_t code[] = {SshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000041ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshlScalarDRoundUp) {
  static const uint32_t code[] = {UrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x0000000000000001ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000FFULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshlScalarDNoRound) {
  static const uint32_t code[] = {UrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x0000000000000002ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000FFULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshlScalarDEq64Bit63) {
  static const uint32_t code[] = {UrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000C0ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshlScalarDEq64Zero) {
  static const uint32_t code[] = {UrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x7FFFFFFFFFFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000C0ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshlScalarDBigZero) {
  static const uint32_t code[] = {UrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000BFULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SrshlScalarDRoundNegOne) {
  static const uint32_t code[] = {SrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000FFULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SrshlScalarDMinRight) {
  static const uint32_t code[] = {SrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000FFULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SrshlScalarDBigRightZero) {
  static const uint32_t code[] = {SrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x00000000000000C0ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SrshlScalarDLeft) {
  static const uint32_t code[] = {SrshlScalarD(0, 1, 2)};
  SetV128(&state_, 1, 0x0000000000000005ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x0000000000000002ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x14ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// ---- Scalar FP (FABD/FMULX/FRECPS/FRSQRTS, S/D) ----
TEST_F(Arm64HeavyOptimizerFrontendTest, FabdScalarS) {
  static const uint32_t code[] = {FabdScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA40400000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB40A00000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FabdScalarD) {
  static const uint32_t code[] = {FabdScalar(true, 0, 1, 2)};
  SetV128(&state_, 1, 0x4008000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x4014000000000000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxScalarS) {
  static const uint32_t code[] = {FmulxScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA40000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB40400000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040C00000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxScalarSZeroInfPos) {
  static const uint32_t code[] = {FmulxScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA00000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB7F800000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxScalarSZeroInfNeg) {
  static const uint32_t code[] = {FmulxScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA80000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB7F800000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000C0000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FmulxScalarDZeroInf) {
  static const uint32_t code[] = {FmulxScalar(true, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x7FF0000000000000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrecpsScalarS) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "no FMA";
  static const uint32_t code[] = {FrecpsScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA40400000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB40800000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000C1200000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrecpsScalarSZeroInf) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "no FMA";
  static const uint32_t code[] = {FrecpsScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA00000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB7F800000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrecpsScalarSNaN) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "no FMA";
  static const uint32_t code[] = {FrecpsScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA7FC00000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB40000000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000007FC00000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrecpsScalarD) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "no FMA";
  static const uint32_t code[] = {FrecpsScalar(true, 0, 1, 2)};
  SetV128(&state_, 1, 0x4008000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x4010000000000000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC024000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrsqrtsScalarS) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "no FMA";
  static const uint32_t code[] = {FrsqrtsScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA40000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB40800000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000C0200000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrsqrtsScalarSZeroInf) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "no FMA";
  static const uint32_t code[] = {FrsqrtsScalar(false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA00000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB7F800000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000003FC00000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, FrsqrtsScalarD) {
  if (!host_platform::kHasFMA) GTEST_SKIP() << "no FMA";
  static const uint32_t code[] = {FrsqrtsScalar(true, 0, 1, 2)};
  SetV128(&state_, 1, 0x4000000000000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0x4010000000000000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC004000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// ---- SQRDMLAH / SQRDMLSH (H/S) ----
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrdmlahScalarH) {
  static const uint32_t code[] = {SqrdmlScalar(false, 0b01, 0, 1, 2)};
  SetV128(&state_, 1, 0x1111222233334000ULL, 0xAAAAAAAAAAAAAAAAULL);
  SetV128(&state_, 2, 0x4444555566664000ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFE0001ULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000002001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrdmlahScalarHSat) {
  static const uint32_t code[] = {SqrdmlScalar(false, 0b01, 0, 1, 2)};
  SetV128(&state_, 1, 0x1111222233338000ULL, 0xAAAAAAAAAAAAAAAAULL);
  SetV128(&state_, 2, 0x4444555566668000ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFE0100ULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000007FFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrdmlshScalarH) {
  static const uint32_t code[] = {SqrdmlScalar(true, 0b01, 0, 1, 2)};
  SetV128(&state_, 1, 0x1111222233334000ULL, 0xAAAAAAAAAAAAAAAAULL);
  SetV128(&state_, 2, 0x4444555566664000ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFE0001ULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000E001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrdmlahScalarS) {
  static const uint32_t code[] = {SqrdmlScalar(false, 0b10, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA40000000ULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 2, 0xBBBBBBBB40000000ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEF00000001ULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false; RunRegion(&state_, code, end_pc, &ok); ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000020000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQDMULH/SQRDMULH (H/S): only lane 0 participates; upper lanes of Vn/Vm
// are garbage and must be scrubbed, and Vd[127:esize] must read back 0.
// Expected values computed from the ARM semantics reference (NOT the codegen):
// H a=0x8000 b=0x8000 -> corner-saturated 0x7FFF; H SQRDMULH a=0x7FFF b=2 -> 2;
// S a=0x80000000 b=0x80000000 -> corner 0x7FFFFFFF; S SQRDMULH a=0x7FFFFFFF b=2
// -> 2.

// SQDMULH scalar H, INT16_MIN*INT16_MIN corner -> INT16_MAX (0x7FFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmulhScalarHCorner) {
  static const uint32_t code[] = {SqdmulhScalar(0b01, 0, 1, 2)};
  SetV128(&state_, 1, 0x1111222233338000ULL, 0xAAAAAAAAAAAAAAAAULL);  // Vn.h[0]=0x8000
  SetV128(&state_, 2, 0x4444555566668000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vm.h[0]=0x8000
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000007FFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SQRDMULH scalar H rounding path (PMULHRSW): a=0x7FFF, b=2 -> 2.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrdmulhScalarH) {
  static const uint32_t code[] = {SqrdmulhScalar(0b01, 0, 1, 2)};
  SetV128(&state_, 1, 0x1111222233337FFFULL, 0xAAAAAAAAAAAAAAAAULL);  // Vn.h[0]=0x7FFF
  SetV128(&state_, 2, 0x4444555566660002ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vm.h[0]=0x0002
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SQDMULH scalar S, INT32_MIN*INT32_MIN corner -> INT32_MAX (0x7FFFFFFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmulhScalarSCorner) {
  static const uint32_t code[] = {SqdmulhScalar(0b10, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA80000000ULL, 0xCCCCCCCCCCCCCCCCULL);  // Vn.s[0]=0x80000000
  SetV128(&state_, 2, 0xBBBBBBBB80000000ULL, 0xDDDDDDDDDDDDDDDDULL);  // Vm.s[0]=0x80000000
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000007FFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SQRDMULH scalar S rounding path (+2^31): a=0x7FFFFFFF, b=2 -> 2.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrdmulhScalarS) {
  static const uint32_t code[] = {SqrdmulhScalar(0b10, 0, 1, 2)};
  SetV128(&state_, 1, 0xAAAAAAAA7FFFFFFFULL, 0xCCCCCCCCCCCCCCCCULL);  // Vn.s[0]=0x7FFFFFFF
  SetV128(&state_, 2, 0xBBBBBBBB00000002ULL, 0xDDDDDDDDDDDDDDDDULL);  // Vm.s[0]=0x00000002
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar S (round toward zero, signed).  Vn.s[0] = 2.75f -> 2.  The
// upper source lanes are poisoned to prove the lane-0 scrub (they must not
// leak into Vd[63:32]).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarSInRange) {
  static const uint32_t code[] = {FcvtzScalar(false, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAA40300000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=2.75f
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar S: Vn.s[0] = -3.0f -> -3 (0xFFFFFFFD).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarSNeg) {
  static const uint32_t code[] = {FcvtzScalar(false, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAAC0400000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=-3.0f
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFDULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar S: NaN -> 0 (x86 CVTTPS2DQ would give 0x80000000).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarSNan) {
  static const uint32_t code[] = {FcvtzScalar(false, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAA7FC00000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=NaN
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar S: +Inf -> INT32_MAX (0x7FFFFFFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarSPosOverflow) {
  static const uint32_t code[] = {FcvtzScalar(false, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAA7F800000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=+Inf
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000007FFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar S: -Inf -> INT32_MIN (0x80000000).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarSNegOverflow) {
  static const uint32_t code[] = {FcvtzScalar(false, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAAFF800000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=-Inf
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000080000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZU scalar S (round toward zero, unsigned): Vn.s[0] = 3.0f -> 3.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuScalarSInRange) {
  static const uint32_t code[] = {FcvtzScalar(true, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAA40400000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=3.0f
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000003ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZU scalar S: negative input clamps to 0 (MAXPS(src,0)).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuScalarSNeg) {
  static const uint32_t code[] = {FcvtzScalar(true, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAABF800000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=-1.0f
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZU scalar S: 3e9f is in the unsigned range (>2^31) -> 0xB2D05E00.  Tests
// the subtract-2^31 offset path.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuScalarSBigInRange) {
  static const uint32_t code[] = {FcvtzScalar(true, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAA4F32D05EULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=3e9f
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000B2D05E00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZU scalar S: +Inf saturates to UINT32_MAX (0xFFFFFFFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuScalarSSatMax) {
  static const uint32_t code[] = {FcvtzScalar(true, false, 0, 1)};
  SetV128(&state_, 1, 0xAAAAAAAA7F800000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=+Inf
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar D (FP64 -> signed int64, round toward zero).  Vn.d[0] = 2.75
// -> 2.  Routes through the GP-register EmitFcvtz (dest_to_simd), mirroring
// the lite `.d` lowering.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarDInRange) {
  static const uint32_t code[] = {FcvtzScalar(false, true, 0, 1)};  // fcvtzs d0, d1
  SetV128(&state_, 1, 0x4006000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=2.75
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar D: -3.0 -> -3 (0xFFFFFFFFFFFFFFFD).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarDNeg) {
  static const uint32_t code[] = {FcvtzScalar(false, true, 0, 1)};  // fcvtzs d0, d1
  SetV128(&state_, 1, 0xC008000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=-3.0
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFDULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar D: NaN -> 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarDNan) {
  static const uint32_t code[] = {FcvtzScalar(false, true, 0, 1)};  // fcvtzs d0, d1
  SetV128(&state_, 1, 0x7FF8000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=NaN
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZS scalar D: +Inf -> INT64_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsScalarDPosOverflow) {
  static const uint32_t code[] = {FcvtzScalar(false, true, 0, 1)};  // fcvtzs d0, d1
  SetV128(&state_, 1, 0x7FF0000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=+Inf
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZU scalar D (FP64 -> unsigned int64, round toward zero): 3.0 -> 3.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuScalarDInRange) {
  static const uint32_t code[] = {FcvtzScalar(true, true, 0, 1)};  // fcvtzu d0, d1
  SetV128(&state_, 1, 0x4008000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=3.0
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000003ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZU scalar D: negative input clamps to 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuScalarDNeg) {
  static const uint32_t code[] = {FcvtzScalar(true, true, 0, 1)};  // fcvtzu d0, d1
  SetV128(&state_, 1, 0xBFF0000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=-1.0
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZU scalar D: 2^63 is in [2^63, 2^64) -> 0x8000000000000000.  Exercises the
// subtract-2^63 / cvtt / set-bit63 fix-up path.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuScalarDBigInRange) {
  static const uint32_t code[] = {FcvtzScalar(true, true, 0, 1)};  // fcvtzu d0, d1
  SetV128(&state_, 1, 0x43E0000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=2^63
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x8000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// FCVTZU scalar D: +Inf saturates to UINT64_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuScalarDSatMax) {
  static const uint32_t code[] = {FcvtzScalar(true, true, 0, 1)};  // fcvtzu d0, d1
  SetV128(&state_, 1, 0x7FF0000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=+Inf
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SCVTF scalar S (signed int32 -> FP32).  Vn.s[0] = 5 -> 5.0f (0x40A00000).
// The upper source lanes are poisoned (0xAAAAAAAA / high half) to prove the
// lane-0 scrub does not leak into Vd[63:32].
TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfScalarSInRange) {
  static const uint32_t code[] = {CvtfScalar(false, false, 0, 1)};  // scvtf s0, s1
  SetV128(&state_, 1, 0xAAAAAAAA00000005ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=5
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040A00000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SCVTF scalar S: Vn.s[0] = -1 (0xFFFFFFFF) -> -1.0f (0xBF800000).
TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfScalarSNeg) {
  static const uint32_t code[] = {CvtfScalar(false, false, 0, 1)};  // scvtf s0, s1
  SetV128(&state_, 1, 0xAAAAAAAAFFFFFFFFULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=-1
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000BF800000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// UCVTF scalar S (unsigned int32 -> FP32): Vn.s[0] = 5 -> 5.0f (0x40A00000).
TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfScalarSInRange) {
  static const uint32_t code[] = {CvtfScalar(true, false, 0, 1)};  // ucvtf s0, s1
  SetV128(&state_, 1, 0xAAAAAAAA00000005ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=5
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040A00000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// UCVTF scalar S: Vn.s[0] = 0x80000000 (unsigned 2^31) -> 2^31 (0x4F000000).
// Exercises the MSB-set 2^32-addend path (signed CVTDQ2PS gives -2^31; adding
// 2^32 restores the unsigned value).
TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfScalarSBig) {
  static const uint32_t code[] = {CvtfScalar(true, false, 0, 1)};  // ucvtf s0, s1
  SetV128(&state_, 1, 0xAAAAAAAA80000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=2^31
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000004F000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SCVTF scalar D (signed int64 -> FP64).  Vn.d[0] = 5 -> 5.0 (0x4014000000000000).
// Routes through the GP-register EmitScvtfUcvtf (src_from_simd).
TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfScalarDInRange) {
  static const uint32_t code[] = {CvtfScalar(false, true, 0, 1)};  // scvtf d0, d1
  SetV128(&state_, 1, 0x0000000000000005ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=5
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);  // poison Vd
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4014000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// SCVTF scalar D: Vn.d[0] = -1 (0xFFFFFFFFFFFFFFFF) -> -1.0 (0xBFF0000000000000).
TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfScalarDNeg) {
  static const uint32_t code[] = {CvtfScalar(false, true, 0, 1)};  // scvtf d0, d1
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=-1
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xBFF0000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// UCVTF scalar D (unsigned int64 -> FP64): Vn.d[0] = 5 -> 5.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfScalarDInRange) {
  static const uint32_t code[] = {CvtfScalar(true, true, 0, 1)};  // ucvtf d0, d1
  SetV128(&state_, 1, 0x0000000000000005ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=5
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4014000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// UCVTF scalar D: Vn.d[0] = 0x8000000000000000 (unsigned 2^63) -> 2^63
// (0x43E0000000000000).  Exercises the bit63-set round-to-odd fix-up path.
TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfScalarDBig) {
  static const uint32_t code[] = {CvtfScalar(true, true, 0, 1)};  // ucvtf d0, d1
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.d[0]=2^63
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x43E0000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQXTN S (size=10, D->S): in-range 64-bit signed value narrows exactly.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtnScalarSInRange) {
  static const uint32_t code[] = {SqxtnScalar(false, 0b10, 0b10100, 0, 1)};  // sqxtn s0, d1
  SetV128(&state_, 1, 0x0000000012345678ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000012345678ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQXTN S: +2^32 > INT32_MAX -> 0x7FFFFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtnScalarSPosOverflow) {
  static const uint32_t code[] = {SqxtnScalar(false, 0b10, 0b10100, 0, 1)};  // sqxtn s0, d1
  SetV128(&state_, 1, 0x0000000100000000ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000007FFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQXTN S: -2^32 < INT32_MIN -> 0x80000000.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtnScalarSNegOverflow) {
  static const uint32_t code[] = {SqxtnScalar(false, 0b10, 0b10100, 0, 1)};  // sqxtn s0, d1
  SetV128(&state_, 1, 0xFFFFFFFF00000000ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000080000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQXTN B (size=00, H->B): 0x1234 signed > 127 -> 0x7F. Upper source
// lanes are poisoned to prove the lane-0 scrub does not leak.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtnScalarBSat) {
  static const uint32_t code[] = {SqxtnScalar(false, 0b00, 0b10100, 0, 1)};  // sqxtn b0, h1
  SetV128(&state_, 1, 0xAAAAAAAAAAAA1234ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000007FULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar UQXTN S (size=10, D->S unsigned): value > UINT32_MAX -> 0xFFFFFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqxtnScalarSSat) {
  static const uint32_t code[] = {SqxtnScalar(true, 0b10, 0b10100, 0, 1)};  // uqxtn s0, d1
  SetV128(&state_, 1, 0x0000000123456789ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar UQXTN H (size=01, S->H unsigned): value > UINT16_MAX -> 0xFFFF. Upper
// source lanes poisoned.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqxtnScalarHSat) {
  static const uint32_t code[] = {SqxtnScalar(true, 0b01, 0b10100, 0, 1)};  // uqxtn h0, s1
  SetV128(&state_, 1, 0xBBBBBBBB0001FFFFULL, 0xCCCCCCCCCCCCCCCCULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQXTUN S (size=10, D->S signed->unsigned): negative source -> 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtunScalarSNeg) {
  static const uint32_t code[] = {SqxtnScalar(true, 0b10, 0b10010, 0, 1)};  // sqxtun s0, d1
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQXTUN B (size=00, H->B signed->unsigned): 200 in [0,255] -> 0xC8.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtunScalarBInRange) {
  static const uint32_t code[] = {SqxtnScalar(true, 0b00, 0b10010, 0, 1)};  // sqxtun b0, h1
  SetV128(&state_, 1, 0xCCCCCCCCCCCC00C8ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000000000C8ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQABS S (size=10): |5| = 5.  Upper source lanes poisoned to prove the
// lane-0 scrub does not leak into Vd.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqabsScalarSPos) {
  static const uint32_t code[] = {SqabsScalar(false, 0b10, 0, 1)};  // sqabs s0, s1
  SetV128(&state_, 1, 0xAAAAAAAA00000005ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=5
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000005ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQABS S: |-7| (0xFFFFFFF9) = 7.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqabsScalarSNeg) {
  static const uint32_t code[] = {SqabsScalar(false, 0b10, 0, 1)};  // sqabs s0, s1
  SetV128(&state_, 1, 0xAAAAAAAAFFFFFFF9ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=-7
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000007ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQABS S saturation: |INT32_MIN| overflows -> INT32_MAX (0x7FFFFFFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqabsScalarSSat) {
  static const uint32_t code[] = {SqabsScalar(false, 0b10, 0, 1)};  // sqabs s0, s1
  SetV128(&state_, 1, 0xAAAAAAAA80000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=INT32_MIN
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000007FFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQNEG S: -(5) = -5 (0xFFFFFFFB).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqnegScalarSPos) {
  static const uint32_t code[] = {SqabsScalar(true, 0b10, 0, 1)};  // sqneg s0, s1
  SetV128(&state_, 1, 0xAAAAAAAA00000005ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=5
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFBULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQNEG S saturation: -(INT32_MIN) overflows -> INT32_MAX (0x7FFFFFFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqnegScalarSSat) {
  static const uint32_t code[] = {SqabsScalar(true, 0b10, 0, 1)};  // sqneg s0, s1
  SetV128(&state_, 1, 0xAAAAAAAA80000000ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.s[0]=INT32_MIN
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000007FFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQABS B (size=00) saturation: |INT8_MIN| (0x80) -> INT8_MAX (0x7F),
// stored in the low byte with the upper bytes zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqabsScalarBSat) {
  static const uint32_t code[] = {SqabsScalar(false, 0b00, 0, 1)};  // sqabs b0, b1
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAA80ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.b[0]=0x80
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000007FULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQNEG H (size=01): -(3) = -3 (0xFFFD) in the low halfword.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqnegScalarHNeg) {
  static const uint32_t code[] = {SqabsScalar(true, 0b01, 0, 1)};  // sqneg h0, h1
  SetV128(&state_, 1, 0xAAAAAAAAAAAA0003ULL, 0xBBBBBBBBBBBBBBBBULL);  // Vn.h[0]=3
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFDULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0ULL);
}

// Scalar SQABS/SQNEG D (size=11) needs 64-bit-lane saturation -> heavy bails.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqabsScalarDBails) {
  static const uint32_t code[] = {SqabsScalar(false, 0b11, 0, 1)};  // sqabs d0, d1
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// Scalar SQDMULH B (size=00) is unallocated for this opcode and must bail.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmulhScalarBBails) {
  static const uint32_t code[] = {SqdmulhScalar(0b00, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

//
// AdvSIMD modified-immediate (MOVI/MVNI/FMOV-vec/ORR/BIC) and DUP (general).
//

// 0 Q op 0111100000 abc(3) cmode(4) 01 defgh(5) Rd. Base = 0x0F000400.
constexpr uint32_t SimdModImm(bool q, uint8_t op, uint8_t cmode, uint8_t imm8, uint8_t rd) {
  uint8_t abc = (imm8 >> 5) & 0x7;
  uint8_t defgh = imm8 & 0x1F;
  return 0x0F000400u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(op) << 29) |
         (static_cast<uint32_t>(abc) << 16) | (static_cast<uint32_t>(cmode) << 12) |
         (static_cast<uint32_t>(defgh) << 5) | rd;
}
// AdvSIMD copy (DUP general): 0 Q 0 01110000 imm5(5) 0 0001 1 Rn Rd. Base = 0x0E000C00.
constexpr uint32_t DupGen(bool q, uint8_t imm5, uint8_t rd, uint8_t rn) {
  return 0x0E000C00u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(imm5) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}
// DUP (element): 0 Q 0 01110000 imm5(5) 0 0000 1 Rn Rd. Base = 0x0E000400.
constexpr uint32_t DupElem(bool q, uint8_t imm5, uint8_t rd, uint8_t rn) {
  return 0x0E000400u | (static_cast<uint32_t>(q) << 30) | (static_cast<uint32_t>(imm5) << 16) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// movi v0.2d, #0 (0x6f00e400) — the most-frequent heavy bail. All zero.
TEST_F(Arm64HeavyOptimizerFrontendTest, MoviV2DZero) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/1, /*cmode=*/0xE, 0x00, 0)};
  ASSERT_EQ(code[0], 0x6f00e400u);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// movi v0.2d, #0xffffffffffffffff (Q=1, op=1, cmode=1110, imm8=0xff).
TEST_F(Arm64HeavyOptimizerFrontendTest, MoviV2DAllOnes) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/1, /*cmode=*/0xE, 0xFF, 0)};
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
}

// movi d0, #0xff00ff00ff00ff00 (Q=0 scalar D, op=1, cmode=1110, imm8=0xaa) — D-form upper zero.
TEST_F(Arm64HeavyOptimizerFrontendTest, MoviDScalarUpperZero) {
  static const uint32_t code[] = {SimdModImm(/*q=*/false, /*op=*/1, /*cmode=*/0xE, 0xAA, 0)};
  ASSERT_EQ(code[0], 0x2f05e540u);
  SetV128(&state_, 0, 0x1111111111111111ULL, 0x2222222222222222ULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF00FF00FF00FF00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);  // D-form zeroes the upper 64 bits
}

// movi v3.4s, #0xab, lsl #16 (Q=1, op=0, cmode=0100).
TEST_F(Arm64HeavyOptimizerFrontendTest, MoviV4SLsl16) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/0, /*cmode=*/0x4, 0xAB, 3)};
  ASSERT_EQ(code[0], 0x4f054563u);
  SetV128(&state_, 3, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 3), 0x00AB000000AB0000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 3), 0x00AB000000AB0000ULL);
}

// movi v0.16b, #0x55 (Q=1, op=0, cmode=1110).
TEST_F(Arm64HeavyOptimizerFrontendTest, MoviV16B) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/0, /*cmode=*/0xE, 0x55, 0)};
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x5555555555555555ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x5555555555555555ULL);
}

// mvni v5.4s, #1, lsl #24 (Q=1, op=1, cmode=0110, imm8=1) -> ~(0x01000000 per lane).
TEST_F(Arm64HeavyOptimizerFrontendTest, MvniV4SLsl24) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/1, /*cmode=*/0x6, 0x01, 5)};
  ASSERT_EQ(code[0], 0x6f006425u);
  SetV128(&state_, 5, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 5), 0xFEFFFFFFFEFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 5), 0xFEFFFFFFFEFFFFFFULL);
}

// fmov v7.4s, #2.0 (Q=1, op=0, cmode=1111, imm8=0x00) -> 0x40000000 per S lane.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovVecV4S) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/0, /*cmode=*/0xF, 0x00, 7)};
  ASSERT_EQ(code[0], 0x4f00f407u);
  SetV128(&state_, 7, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 7), 0x4000000040000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 7), 0x4000000040000000ULL);
}

// fmov v9.2d, #-1.5 (Q=1, op=1, cmode=1111, imm8=0xf8) -> 0xBFF8000000000000 per D lane.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmovVecV2D) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/1, /*cmode=*/0xF, 0xF8, 9)};
  ASSERT_EQ(code[0], 0x6f07f709u);
  SetV128(&state_, 9, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 9), 0xBFF8000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 9), 0xBFF8000000000000ULL);
}

// orr v0.4s, #0xab, lsl #8 (Q=1, op=0, cmode=0011) — read-modify-write OR.
TEST_F(Arm64HeavyOptimizerFrontendTest, OrrVecImm) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/0, /*cmode=*/0x3, 0xAB, 0)};
  ASSERT_EQ(code[0], 0x4f053560u);
  SetV128(&state_, 0, 0x0000000100000002ULL, 0x0000000300000004ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  const uint64_t imm = 0x0000AB000000AB00ULL;  // 0xab << 8 per S lane
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000100000002ULL | imm);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000300000004ULL | imm);
}

// bic v0.4s, #0xab, lsl #8 (Q=1, op=1, cmode=0011) — read-modify-write AND-NOT.
TEST_F(Arm64HeavyOptimizerFrontendTest, BicVecImm) {
  static const uint32_t code[] = {SimdModImm(/*q=*/true, /*op=*/1, /*cmode=*/0x3, 0xAB, 0)};
  ASSERT_EQ(code[0], 0x6f053560u);
  SetV128(&state_, 0, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  const uint64_t imm = 0x0000AB000000AB00ULL;
  EXPECT_EQ(VLo64(&state_, 0), ~imm);
  EXPECT_EQ(VUpperHi64(&state_, 0), ~imm);
}

// dup v8.2d, x10 — 64-bit broadcast (must use MOVQ, not MOVD).
TEST_F(Arm64HeavyOptimizerFrontendTest, DupGenV2D) {
  static const uint32_t code[] = {DupGen(/*q=*/true, /*imm5=*/0x08, 8, 10)};
  ASSERT_EQ(code[0], 0x4e080d48u);
  state_.cpu.x[10] = 0x1122334455667788ULL;
  SetV128(&state_, 8, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 8), 0x1122334455667788ULL);
  EXPECT_EQ(VUpperHi64(&state_, 8), 0x1122334455667788ULL);  // full 64 bits, not truncated
}

// dup v6.4s, w9 — 32-bit broadcast (Q=1).
TEST_F(Arm64HeavyOptimizerFrontendTest, DupGenV4S) {
  static const uint32_t code[] = {DupGen(/*q=*/true, /*imm5=*/0x04, 6, 9)};
  ASSERT_EQ(code[0], 0x4e040d26u);
  state_.cpu.x[9] = 0xDEADBEEFCAFEBABEULL;  // only low 32 (0xCAFEBABE) used
  SetV128(&state_, 6, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 6), 0xCAFEBABECAFEBABEULL);
  EXPECT_EQ(VUpperHi64(&state_, 6), 0xCAFEBABECAFEBABEULL);
}

// dup v0.2s, w1 — 32-bit broadcast, D-form (Q=0) upper-zero.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupGenV2SUpperZero) {
  static const uint32_t code[] = {DupGen(/*q=*/false, /*imm5=*/0x04, 0, 1)};
  ASSERT_EQ(code[0], 0x0e040c20u);
  state_.cpu.x[1] = 0x00000000ABCD1234ULL;
  SetV128(&state_, 0, 0x9999999999999999ULL, 0x8888888888888888ULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xABCD1234ABCD1234ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);  // D-form zeroes the upper 64 bits
}

// dup v0.8h, w1 — 16-bit broadcast (Q=1).
TEST_F(Arm64HeavyOptimizerFrontendTest, DupGenV8H) {
  static const uint32_t code[] = {DupGen(/*q=*/true, /*imm5=*/0x02, 0, 1)};
  ASSERT_EQ(code[0], 0x4e020c20u);
  state_.cpu.x[1] = 0x000000000000ABCDULL;
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xABCDABCDABCDABCDULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xABCDABCDABCDABCDULL);
}

// dup v0.16b, w1 — 8-bit broadcast (Q=1) via PSHUFB.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupGenV16B) {
  static const uint32_t code[] = {DupGen(/*q=*/true, /*imm5=*/0x01, 0, 1)};
  ASSERT_EQ(code[0], 0x4e010c20u);
  state_.cpu.x[1] = 0x00000000000000A5ULL;
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xA5A5A5A5A5A5A5A5ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xA5A5A5A5A5A5A5A5ULL);
}

// dup v0.8b, w1 — 8-bit broadcast, D-form (Q=0) upper-zero.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupGenV8BUpperZero) {
  static const uint32_t code[] = {DupGen(/*q=*/false, /*imm5=*/0x01, 0, 1)};
  ASSERT_EQ(code[0], 0x0e010c20u);
  state_.cpu.x[1] = 0x00000000000000A5ULL;
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xA5A5A5A5A5A5A5A5ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// dup v0.2d, xzr — XZR source broadcasts zero (compiler vector-zero idiom).
TEST_F(Arm64HeavyOptimizerFrontendTest, DupGenV2DXzr) {
  static const uint32_t code[] = {DupGen(/*q=*/true, /*imm5=*/0x08, 0, 31)};
  SetV128(&state_, 0, 0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// DUP (element) must bail: needs PSHUFD/PSHUFLW not in the backend allowlist.
// DUP (element): broadcast Vn.<T>[index] to every lane of Vd.
// dup v0.16b, v1.b[5] — byte broadcast (Q=1). byte[5] of V1 = 0x0D.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupElemV16B) {
  static const uint32_t code[] = {DupElem(/*q=*/true, /*imm5=*/0x0B, 0, 1)};
  ASSERT_EQ(code[0], 0x4e0b0420u);
  SetV128(&state_, 1, 0x0F0E0D0C0B0A0908ULL, 0x1F1E1D1C1B1A1918ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0D0D0D0D0D0D0D0DULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0D0D0D0D0D0D0D0DULL);
}

// dup v0.8b, v1.b[2] — byte broadcast, D-form (Q=0) upper-zero. byte[2] = 0x0A.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupElemV8BUpperZero) {
  static const uint32_t code[] = {DupElem(/*q=*/false, /*imm5=*/0x05, 0, 1)};
  ASSERT_EQ(code[0], 0x0e050420u);
  SetV128(&state_, 1, 0x0F0E0D0C0B0A0908ULL, 0x1F1E1D1C1B1A1918ULL);
  SetV128(&state_, 0, 0x9999999999999999ULL, 0x8888888888888888ULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0A0A0A0A0A0A0A0AULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);  // D-form zeroes the upper 64 bits
}

// dup v0.8h, v1.h[3] — halfword broadcast (Q=1). h[3] = 0x0004.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupElemV8H) {
  static const uint32_t code[] = {DupElem(/*q=*/true, /*imm5=*/0x0E, 0, 1)};
  ASSERT_EQ(code[0], 0x4e0e0420u);
  SetV128(&state_, 1, 0x0004000300020001ULL, 0x0008000700060005ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0004000400040004ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0004000400040004ULL);
}

// dup v0.4s, v1.s[2] — word broadcast (Q=1). s[2] = 0x33333333.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupElemV4S) {
  static const uint32_t code[] = {DupElem(/*q=*/true, /*imm5=*/0x14, 0, 1)};
  ASSERT_EQ(code[0], 0x4e140420u);
  SetV128(&state_, 1, 0x2222222211111111ULL, 0x4444444433333333ULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3333333333333333ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x3333333333333333ULL);
}

// dup v0.2d, v1.d[1] — doubleword broadcast, high lane (Q=1). d[1] = 0xBBBB...
TEST_F(Arm64HeavyOptimizerFrontendTest, DupElemV2DIdx1) {
  static const uint32_t code[] = {DupElem(/*q=*/true, /*imm5=*/0x18, 0, 1)};
  ASSERT_EQ(code[0], 0x4e180420u);
  SetV128(&state_, 1, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xBBBBBBBBBBBBBBBBULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBBBBBBBBBBBBBBBBULL);
}

// dup v0.2d, v1.d[0] — doubleword broadcast, low lane (Q=1). d[0] full 64 bits.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupElemV2DIdx0) {
  static const uint32_t code[] = {DupElem(/*q=*/true, /*imm5=*/0x08, 0, 1)};
  ASSERT_EQ(code[0], 0x4e080420u);
  SetV128(&state_, 1, 0x1122334455667788ULL, 0x99999999AAAAAAAAULL);
  SetV128(&state_, 0, 0, 0);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1122334455667788ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x1122334455667788ULL);
}

// dup v0.1d, v1.d[0] (esize=D, Q=0) is ARM-reserved: the heavy tier bails.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupElemV1DReservedBails) {
  static const uint32_t code[] = {DupElem(/*q=*/false, /*imm5=*/0x08, 0, 1)};
  ASSERT_EQ(code[0], 0x0e080420u);
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// dup v0.1d, x1 is ARM-reserved (esize=D, Q=0): the heavy tier bails.
TEST_F(Arm64HeavyOptimizerFrontendTest, DupGenV1DReservedBails) {
  static const uint32_t code[] = {DupGen(/*q=*/false, /*imm5=*/0x08, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

//
// Load/store-exclusive + acquire/release. Field layout: size[31:30],
// 0010000[29:23], o2[23], L[22], o1[21], Rs[20:16], o0[15], Rt2[14:10]=11111,
// Rn[9:5], Rt[4:0].
//
constexpr uint32_t LdxrX(uint8_t rt, uint8_t rn) {
  return 0xC85F7C00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t StxrX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xC8007C00u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdaxrX(uint8_t rt, uint8_t rn) {
  return 0xC85FFC00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t StlxrX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xC800FC00u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdxrW(uint8_t rt, uint8_t rn) {
  return 0x885F7C00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t StxrW(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x88007C00u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdxrB(uint8_t rt, uint8_t rn) {
  return 0x085F7C00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t StxrB(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x08007C00u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdxrH(uint8_t rt, uint8_t rn) {
  return 0x485F7C00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t StxrH(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x48007C00u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdarX(uint8_t rt, uint8_t rn) {
  return 0xC8DFFC00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t StlrX(uint8_t rt, uint8_t rn) {
  return 0xC89FFC00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdarW(uint8_t rt, uint8_t rn) {
  return 0x88DFFC00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t StlrW(uint8_t rt, uint8_t rn) {
  return 0x889FFC00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdarB(uint8_t rt, uint8_t rn) {
  return 0x08DFFC00u | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t StlrB(uint8_t rt, uint8_t rn) {
  return 0x089FFC00u | (static_cast<uint32_t>(rn) << 5) | rt;
}

// Single-threaded LDXR-then-STXR round-trip (64-bit): LDXR sets the reservation,
// STXR succeeds (no intervening write), stores Rt, and returns status 0 in Rs.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdxrStxr64RoundTrip) {
  alignas(8) static uint64_t buf = 0x1111222233334444ULL;
  static const uint32_t ldxr_code[] = {LdxrX(0, 1)};    // LDXR X0, [X1]
  static const uint32_t stxr_code[] = {StxrX(2, 3, 1)};  // STXR W2, X3, [X1]

  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.x[3] = 0xAABBCCDDEEFF0011ULL;  // value to store

  state_.cpu.insn_addr = ToGuestAddr(ldxr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ldxr_code) + sizeof(ldxr_code)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1111222233334444ULL});      // loaded value
  EXPECT_EQ(state_.cpu.reservation_address, ToGuestAddr(&buf));
  EXPECT_EQ(static_cast<uint64_t>(state_.cpu.reservation_value),
            uint64_t{0x1111222233334444ULL});

  state_.cpu.insn_addr = ToGuestAddr(stxr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stxr_code) + sizeof(stxr_code)));
  EXPECT_EQ(buf, uint64_t{0xAABBCCDDEEFF0011ULL});  // memory updated
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});          // status = success
  EXPECT_EQ(state_.cpu.reservation_address, GuestAddr{0});  // reservation cleared
}

// STXR with no live reservation (address mismatch) fails: memory unchanged, status 1.
TEST_F(Arm64HeavyOptimizerFrontendTest, StxrWithoutReservationFails) {
  alignas(8) static uint64_t buf = 0xDEADBEEFCAFEF00DULL;
  static const uint32_t stxr_code[] = {StxrX(2, 3, 1)};  // STXR W2, X3, [X1]

  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.x[3] = 0x0;
  state_.cpu.reservation_address = GuestAddr{0};  // no reservation
  state_.cpu.reservation_value = 0;

  state_.cpu.insn_addr = ToGuestAddr(stxr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stxr_code) + sizeof(stxr_code)));
  EXPECT_EQ(buf, uint64_t{0xDEADBEEFCAFEF00DULL});  // memory NOT modified
  EXPECT_EQ(state_.cpu.x[2], uint64_t{1});          // status = fail
}

// STXR fails when memory changed under the reservation (CMPXCHG mismatch):
// address still matches but the saved value no longer equals memory.
TEST_F(Arm64HeavyOptimizerFrontendTest, StxrStaleValueFails) {
  alignas(8) static uint64_t buf = 0x0102030405060708ULL;
  static const uint32_t stxr_code[] = {StxrX(2, 3, 1)};  // STXR W2, X3, [X1]

  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.x[3] = 0x9999999999999999ULL;
  state_.cpu.reservation_address = ToGuestAddr(&buf);    // address matches
  state_.cpu.reservation_value = 0xBADBADBADBADBADBULL;  // but value is stale

  state_.cpu.insn_addr = ToGuestAddr(stxr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stxr_code) + sizeof(stxr_code)));
  EXPECT_EQ(buf, uint64_t{0x0102030405060708ULL});  // memory NOT modified
  EXPECT_EQ(state_.cpu.x[2], uint64_t{1});          // status = fail
}

// 32-bit LDXR/STXR round-trip: STXR W writes only the low 32 bits.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdxrStxr32RoundTrip) {
  alignas(8) static uint64_t buf = 0xFFFFFFFF12345678ULL;
  static const uint32_t ldxr_code[] = {LdxrW(0, 1)};    // LDXR W0, [X1]
  static const uint32_t stxr_code[] = {StxrW(2, 3, 1)};  // STXR W2, W3, [X1]

  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.x[3] = 0xAAAAAAAAABCDEF99ULL;  // only low 32 stored

  state_.cpu.insn_addr = ToGuestAddr(ldxr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ldxr_code) + sizeof(ldxr_code)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x12345678});  // zero-extended

  state_.cpu.insn_addr = ToGuestAddr(stxr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stxr_code) + sizeof(stxr_code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFFABCDEF99ULL});  // only low 32 changed
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});
}

// LDAXR/STLXR (acquire/release exclusive) behave identically to LDXR/STXR for
// the monitor; the ordering is a no-op on x86-TSO.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdaxrStlxrRoundTrip) {
  alignas(8) static uint64_t buf = 0x5555666677778888ULL;
  static const uint32_t ldaxr_code[] = {LdaxrX(0, 1)};    // LDAXR X0, [X1]
  static const uint32_t stlxr_code[] = {StlxrX(2, 3, 1)};  // STLXR W2, X3, [X1]

  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.x[3] = 0x1234567890ABCDEFULL;

  state_.cpu.insn_addr = ToGuestAddr(ldaxr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ldaxr_code) + sizeof(ldaxr_code)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x5555666677778888ULL});

  state_.cpu.insn_addr = ToGuestAddr(stlxr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stlxr_code) + sizeof(stlxr_code)));
  EXPECT_EQ(buf, uint64_t{0x1234567890ABCDEFULL});
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});
}

// LDAR (this is the 0xc8dffd08 hot bail): plain acquire load. STLR: release store.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdarStlr64) {
  alignas(8) static uint64_t buf = 0xCAFEBABEF00DFACEULL;
  static const uint32_t ldar_code[] = {LdarX(8, 1)};  // LDAR X8, [X1]
  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(ldar_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ldar_code) + sizeof(ldar_code)));
  EXPECT_EQ(state_.cpu.x[8], uint64_t{0xCAFEBABEF00DFACEULL});

  alignas(8) static uint64_t obuf = 0;
  static const uint32_t stlr_code[] = {StlrX(8, 1)};  // STLR X8, [X1]
  state_.cpu.x[1] = ToGuestAddr(&obuf);
  state_.cpu.x[8] = 0x0011223344556677ULL;
  state_.cpu.insn_addr = ToGuestAddr(stlr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stlr_code) + sizeof(stlr_code)));
  EXPECT_EQ(obuf, uint64_t{0x0011223344556677ULL});
}

// LDARB / STLRB (8-bit acquire/release): byte zero-extended on load.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdarbStlrb8) {
  alignas(8) static uint64_t buf = 0xFFFFFFFFFFFFFF5AULL;
  static const uint32_t ldarb_code[] = {LdarB(0, 1)};  // LDARB W0, [X1]
  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(ldarb_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ldarb_code) + sizeof(ldarb_code)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x5A});  // zero-extended byte
}

// STLRB (8-bit release store): writes only the low byte of [Xn].
TEST_F(Arm64HeavyOptimizerFrontendTest, Stlrb8) {
  alignas(8) static uint64_t buf = 0xFFFFFFFFFFFFFFFFULL;
  static const uint32_t stlrb_code[] = {StlrB(8, 1)};  // STLRB W8, [X1]
  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.x[8] = 0xAB;
  state_.cpu.insn_addr = ToGuestAddr(stlrb_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stlrb_code) + sizeof(stlrb_code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFFFFFFFFABULL});  // only low byte changed
}

// LDAR W / STLR W (32-bit acquire/release): load zero-extends, store writes low 32.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdarStlr32) {
  alignas(8) static uint64_t buf = 0xFFFFFFFF89ABCDEFULL;
  static const uint32_t ldar_code[] = {LdarW(0, 1)};  // LDAR W0, [X1]
  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(ldar_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ldar_code) + sizeof(ldar_code)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x89ABCDEF});  // zero-extended

  alignas(8) static uint64_t obuf = 0xFFFFFFFFFFFFFFFFULL;
  static const uint32_t stlr_code[] = {StlrW(8, 1)};  // STLR W8, [X1]
  state_.cpu.x[1] = ToGuestAddr(&obuf);
  state_.cpu.x[8] = 0x1122334455667788ULL;  // only low 32 stored
  state_.cpu.insn_addr = ToGuestAddr(stlr_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stlr_code) + sizeof(stlr_code)));
  EXPECT_EQ(obuf, uint64_t{0xFFFFFFFF55667788ULL});  // only low 32 changed
}

// 8-bit LDXRB/STXRB round-trip exercises the narrow (AL) LOCK CMPXCHGB path.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdxrbStxrb8RoundTrip) {
  alignas(8) static uint64_t buf = 0xAABBCCDDEEFF0042ULL;  // low byte 0x42
  static const uint32_t ldxrb_code[] = {LdxrB(0, 1)};    // LDXRB W0, [X1]
  static const uint32_t stxrb_code[] = {StxrB(2, 3, 1)};  // STXRB W2, W3, [X1]
  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.x[3] = 0x99;

  state_.cpu.insn_addr = ToGuestAddr(ldxrb_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ldxrb_code) + sizeof(ldxrb_code)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x42});  // zero-extended byte

  state_.cpu.insn_addr = ToGuestAddr(stxrb_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stxrb_code) + sizeof(stxrb_code)));
  EXPECT_EQ(buf, uint64_t{0xAABBCCDDEEFF0099ULL});  // only low byte changed
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});          // success
}

// 16-bit LDXRH/STXRH round-trip exercises the narrow (AX) LOCK CMPXCHGW path.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdxrhStxrh16RoundTrip) {
  alignas(8) static uint64_t buf = 0xAABBCCDDEEFF1234ULL;  // low halfword 0x1234
  static const uint32_t ldxrh_code[] = {LdxrH(0, 1)};    // LDXRH W0, [X1]
  static const uint32_t stxrh_code[] = {StxrH(2, 3, 1)};  // STXRH W2, W3, [X1]
  state_.cpu.x[1] = ToGuestAddr(&buf);
  state_.cpu.x[3] = 0x9988;

  state_.cpu.insn_addr = ToGuestAddr(ldxrh_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ldxrh_code) + sizeof(ldxrh_code)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x1234});  // zero-extended halfword

  state_.cpu.insn_addr = ToGuestAddr(stxrh_code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(stxrh_code) + sizeof(stxrh_code)));
  EXPECT_EQ(buf, uint64_t{0xAABBCCDDEEFF9988ULL});  // only low halfword changed
  EXPECT_EQ(state_.cpu.x[2], uint64_t{0});          // success
}

// LSE single-register atomics: CAS / SWP / LDADD, heavy-tier mirror of the lite
// lowering (LOCK CMPXCHG / XCHG / LOCK XADD). Encodings verified with clang
// -march=armv8.1-a+lse:
//   cas   x5,x1,[x2] = 0xc8a57c41   cas   w5,w1,[x2] = 0x88a57c41
//   swp   x5,x1,[x2] = 0xf8258041   swp   w5,w1,[x2] = 0xb8258041
//   ldadd x5,x1,[x2] = 0xf8250041   ldadd w5,w1,[x2] = 0xb8250041
constexpr uint32_t CasX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xC8A07C00u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t CasW(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x88A07C00u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t SwpX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8208000u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t SwpW(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xB8208000u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdaddX(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xF8200000u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdaddW(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0xB8200000u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
// Byte forms (size=00) — exercise the byte AND-0xFF zero-extension path.
//   casb w5,w1,[x2]=0x08a57c41  swpb w5,w1,[x2]=0x38258041  ldaddb=0x38250041
constexpr uint32_t CasB(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x08A07C00u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t SwpB(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x38208000u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}
constexpr uint32_t LdaddB(uint8_t rs, uint8_t rt, uint8_t rn) {
  return 0x38200000u | (static_cast<uint32_t>(rs) << 16) | (static_cast<uint32_t>(rn) << 5) | rt;
}

// CAS (64-bit) success: [Xn] equals the expected value in Xs, so Xt is stored
// and the old value is returned in Xs.
TEST_F(Arm64HeavyOptimizerFrontendTest, Cas64Match) {
  alignas(8) static uint64_t buf = 0x1111222233334444ULL;
  static const uint32_t code[] = {CasX(5, 1, 2)};  // CAS X5, X1, [X2]
  state_.cpu.x[5] = 0x1111222233334444ULL;          // expected == memory
  state_.cpu.x[1] = 0xAABBCCDDEEFF0011ULL;          // desired
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xAABBCCDDEEFF0011ULL});   // stored on match
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0x1111222233334444ULL});  // old value
}

// CAS (64-bit) mismatch: [Xn] differs from Xs, so memory is untouched and Xs is
// updated with the actual old value.
TEST_F(Arm64HeavyOptimizerFrontendTest, Cas64Mismatch) {
  alignas(8) static uint64_t buf = 0x1111222233334444ULL;
  static const uint32_t code[] = {CasX(5, 1, 2)};  // CAS X5, X1, [X2]
  state_.cpu.x[5] = 0xDEADBEEFDEADBEEFULL;          // expected != memory
  state_.cpu.x[1] = 0xAABBCCDDEEFF0011ULL;          // desired
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x1111222233334444ULL});  // untouched
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0x1111222233334444ULL});  // actual old value
}

// CAS (32-bit) match: only the low 32 bits are compared/stored; the old value
// returned in Ws is zero-extended to 64 bits.
TEST_F(Arm64HeavyOptimizerFrontendTest, Cas32MatchZeroExtends) {
  alignas(8) static uint64_t buf = 0xFFFFFFFF89ABCDEFULL;  // low32 = 0x89ABCDEF
  static const uint32_t code[] = {CasW(5, 1, 2)};  // CAS W5, W1, [X2]
  state_.cpu.x[5] = 0x1111111189ABCDEFULL;          // low32 expected == memory low32
  state_.cpu.x[1] = 0x2222222212345678ULL;          // desired low32 = 0x12345678
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFF12345678ULL});  // only low32 stored
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0x89ABCDEF});  // old value zero-extended
}

// SWP (64-bit): swap Xs into [Xn], old value to Xt.
TEST_F(Arm64HeavyOptimizerFrontendTest, Swp64) {
  alignas(8) static uint64_t buf = 0x1111222233334444ULL;
  static const uint32_t code[] = {SwpX(5, 1, 2)};  // SWP X5, X1, [X2]
  state_.cpu.x[5] = 0xAABBCCDDEEFF0011ULL;          // new value
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xAABBCCDDEEFF0011ULL});   // memory = Xs
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0x1111222233334444ULL});  // old value to Xt
}

// SWP (32-bit): only the low 32 bits swap; old value in Wt is zero-extended.
TEST_F(Arm64HeavyOptimizerFrontendTest, Swp32ZeroExtends) {
  alignas(8) static uint64_t buf = 0xFFFFFFFF89ABCDEFULL;
  static const uint32_t code[] = {SwpW(5, 1, 2)};  // SWP W5, W1, [X2]
  state_.cpu.x[5] = 0x1234567812345678ULL;          // new low32 = 0x12345678
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFF12345678ULL});  // only low32 written
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0x89ABCDEF});  // old value zero-extended
}

// LSE bitwise fetch-and-op via the heavy CMPXCHG retry loop. Encoders: base
// F8203000 (LDSET) / F8201000 (LDCLR) / F8202000 (LDEOR); W form uses B8...
// LDSET X1, X0, [X2]: [mem] |= Xs; old -> Xt.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldset64) {
  alignas(8) static uint64_t buf = 0x00FF00FF00FF00FFULL;
  static const uint32_t code[] = {0xF8213040u};  // ldset x1, x0, [x2]
  state_.cpu.x[1] = 0xFF00FF00FF00FF00ULL;         // Xs (mask)
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFFFFFFFFFFULL});          // old | Xs
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x00FF00FF00FF00FFULL});  // old -> Xt
}

// LDCLR X1, X0, [X2]: [mem] &= ~Xs; old -> Xt.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldclr64) {
  alignas(8) static uint64_t buf = 0xFFFFFFFFFFFFFFFFULL;
  static const uint32_t code[] = {0xF8211040u};  // ldclr x1, x0, [x2]
  state_.cpu.x[1] = 0xFF00FF00FF00FF00ULL;
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x00FF00FF00FF00FFULL});          // old & ~Xs
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFFFULL});  // old -> Xt
}

// LDEOR X1, X0, [X2]: [mem] ^= Xs; old -> Xt.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldeor64) {
  alignas(8) static uint64_t buf = 0x0F0F0F0F0F0F0F0FULL;
  static const uint32_t code[] = {0xF8212040u};  // ldeor x1, x0, [x2]
  state_.cpu.x[1] = 0xFFFFFFFF00000000ULL;
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xF0F0F0F00F0F0F0FULL});          // old ^ Xs
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0F0F0F0F0F0F0F0FULL});  // old -> Xt
}

// LDSET (32-bit): only low 32 written; old value in Wt zero-extended to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldset32ZeroExtends) {
  alignas(8) static uint64_t buf = 0xFFFFFFFF00FF00FFULL;  // low32 = 0x00FF00FF
  static const uint32_t code[] = {0xB8213040u};  // ldset w1, w0, [x2]
  state_.cpu.x[1] = 0x00000000FF00FF00ULL;         // Ws low32 = 0xFF00FF00
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFFFFFFFFFFULL});          // low32 |= Ws; high32 untouched
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0000000000FF00FFULL});  // old Wt zero-extended
}

// LSE min/max via the CMPXCHG loop + select diamond. Encoders: base F8204000
// (LDSMAX) / F8205000 (LDSMIN) / F8206000 (LDUMAX) / F8207000 (LDUMIN).
// LDSMAX X1, X0, [X2]: [mem]=max_signed([mem],Xs). mem=-1, Xs=1 -> signed max=1.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldsmax64Signed) {
  alignas(8) static uint64_t buf = 0xFFFFFFFFFFFFFFFFULL;  // -1 signed
  static const uint32_t code[] = {0xF8214040u};  // ldsmax x1, x0, [x2]
  state_.cpu.x[1] = 0x0000000000000001ULL;         // Xs = 1
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x0000000000000001ULL});         // max(-1, 1) = 1
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFFFULL});  // old -> Xt
}

// LDUMAX with the same bits: unsigned max keeps the large value (distinguishes
// the unsigned path from the signed test above).
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldumax64Unsigned) {
  alignas(8) static uint64_t buf = 0xFFFFFFFFFFFFFFFFULL;  // UINT64_MAX unsigned
  static const uint32_t code[] = {0xF8216040u};  // ldumax x1, x0, [x2]
  state_.cpu.x[1] = 0x0000000000000001ULL;
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFFFFFFFFFFULL});          // max(MAX, 1) = MAX
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0xFFFFFFFFFFFFFFFFULL});
}

// LDSMIN X1, X0, [X2]: signed min. mem=5, Xs=-1 -> min=-1.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldsmin64Signed) {
  alignas(8) static uint64_t buf = 0x0000000000000005ULL;
  static const uint32_t code[] = {0xF8215040u};  // ldsmin x1, x0, [x2]
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;         // Xs = -1
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFFFFFFFFFFULL});          // min(5, -1) = -1
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0000000000000005ULL});  // old -> Xt
}

// LDUMIN with the same bits: unsigned min keeps the small value.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldumin64Unsigned) {
  alignas(8) static uint64_t buf = 0x0000000000000005ULL;
  static const uint32_t code[] = {0xF8217040u};  // ldumin x1, x0, [x2]
  state_.cpu.x[1] = 0xFFFFFFFFFFFFFFFFULL;         // UINT64_MAX
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x0000000000000005ULL});         // min(5, MAX) = 5
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0000000000000005ULL});
}

// LDSMAX (32-bit signed): only low 32 compared/written; old Wt zero-extended.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldsmax32Signed) {
  alignas(8) static uint64_t buf = 0xAAAAAAAAFFFFFFFFULL;  // low32 = -1 signed
  static const uint32_t code[] = {0xB8214040u};  // ldsmax w1, w0, [x2]
  state_.cpu.x[1] = 0x0000000000000001ULL;         // Ws = 1
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xAAAAAAAA00000001ULL});         // low32 max(-1,1)=1; high32 kept
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x00000000FFFFFFFFULL});  // old Wt zero-extended
}

// CASP (32-bit pair) via packed 64-bit LOCK CMPXCHG. Rs:Rs+1 (W4,W5) = expected,
// Rt:Rt+1 (W6,W7) = new. Match -> swap; the old pair returns to Rs:Rs+1.
TEST_F(Arm64HeavyOptimizerFrontendTest, Casp32Match) {
  alignas(8) static uint64_t buf = (uint64_t{0x22222222ULL} << 32) | 0x11111111ULL;
  static const uint32_t code[] = {0x08247C46u};  // casp w4, w5, w6, w7, [x2]
  state_.cpu.x[4] = 0x11111111ULL;  // expected.lo (matches)
  state_.cpu.x[5] = 0x22222222ULL;  // expected.hi (matches)
  state_.cpu.x[6] = 0xAAAAAAAAULL;  // new.lo
  state_.cpu.x[7] = 0xBBBBBBBBULL;  // new.hi
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, (uint64_t{0xBBBBBBBBULL} << 32) | 0xAAAAAAAAULL);
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0x11111111ULL});  // old.lo zero-extended
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0x22222222ULL});  // old.hi zero-extended
}

// CASP mismatch: memory unchanged; Rs:Rs+1 get the actual prior pair.
TEST_F(Arm64HeavyOptimizerFrontendTest, Casp32Mismatch) {
  alignas(8) static uint64_t buf = (uint64_t{0xCAFEBABEULL} << 32) | 0xDEADBEEFULL;
  static const uint32_t code[] = {0x08247C46u};  // casp w4, w5, w6, w7, [x2]
  state_.cpu.x[4] = 0x12345678ULL;  // expected.lo (does NOT match)
  state_.cpu.x[5] = 0x9ABCDEF0ULL;  // expected.hi (does NOT match)
  state_.cpu.x[6] = 0xAAAAAAAAULL;
  state_.cpu.x[7] = 0xBBBBBBBBULL;
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, (uint64_t{0xCAFEBABEULL} << 32) | 0xDEADBEEFULL);  // unchanged
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0xDEADBEEFULL});  // actual old.lo
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0xCAFEBABEULL});  // actual old.hi
}

// LDADD (64-bit): [Xn] += Xs; old value to Xt.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldadd64) {
  alignas(8) static uint64_t buf = 0x0000000000000100ULL;
  static const uint32_t code[] = {LdaddX(5, 1, 2)};  // LDADD X5, X1, [X2]
  state_.cpu.x[5] = 0x0000000000000023ULL;            // addend
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x0000000000000123ULL});    // 0x100 + 0x23
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0x0000000000000100ULL});  // old value to Xt
}

// LDADD (32-bit): 32-bit add; old value in Wt zero-extended to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldadd32ZeroExtends) {
  alignas(8) static uint64_t buf = 0xFFFFFFFF00000100ULL;  // low32 = 0x100
  static const uint32_t code[] = {LdaddW(5, 1, 2)};  // LDADD W5, W1, [X2]
  state_.cpu.x[5] = 0x0000000000000023ULL;            // addend low32 = 0x23
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xFFFFFFFF00000123ULL});    // only low32 updated
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0x00000100});   // old value zero-extended
}

// Byte CAS match: exercises the AND-0xFF zero-extension of the old value. Only
// the low byte is compared/stored; the returned old value is zero-extended.
TEST_F(Arm64HeavyOptimizerFrontendTest, Cas8MatchZeroExtends) {
  alignas(8) static uint64_t buf = 0x1122334455667789ULL;  // low byte = 0x89
  static const uint32_t code[] = {CasB(5, 1, 2)};  // CASB W5, W1, [X2]
  state_.cpu.x[5] = 0xFFFFFFFFFFFFFF89ULL;          // expected low byte == 0x89
  state_.cpu.x[1] = 0xAAAAAAAAAAAAAAABULL;          // desired low byte = 0xAB
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x11223344556677ABULL});  // only low byte stored
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0x89});        // old value zero-extended
}

// Byte SWP: exercises the AND-0xFF zero-extension.
TEST_F(Arm64HeavyOptimizerFrontendTest, Swp8ZeroExtends) {
  alignas(8) static uint64_t buf = 0x1122334455667789ULL;  // low byte = 0x89
  static const uint32_t code[] = {SwpB(5, 1, 2)};  // SWPB W5, W1, [X2]
  state_.cpu.x[5] = 0x11111111111111CDULL;          // new low byte = 0xCD
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x11223344556677CDULL});  // only low byte written
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0x89});        // old value zero-extended
}

// Byte LDADD: exercises the AND-0xFF zero-extension; add wraps within the byte.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldadd8ZeroExtends) {
  alignas(8) static uint64_t buf = 0x11223344556677F0ULL;  // low byte = 0xF0
  static const uint32_t code[] = {LdaddB(5, 1, 2)};  // LDADDB W5, W1, [X2]
  state_.cpu.x[5] = 0x0000000000000015ULL;            // addend low byte = 0x15
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x1122334455667705ULL});    // 0xF0 + 0x15 = 0x105 -> 0x05
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0xF0});          // old value zero-extended
}

// XZR forms: CAS/SWP/LDADD with Rs or Rt == 31 read as zero / discard.
TEST_F(Arm64HeavyOptimizerFrontendTest, SwpXzrDiscardsOldValue) {
  alignas(8) static uint64_t buf = 0x1111222233334444ULL;
  static const uint32_t code[] = {SwpX(5, 31, 2)};  // SWP X5, XZR, [X2]
  state_.cpu.x[5] = 0xAABBCCDDEEFF0011ULL;           // new value
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xAABBCCDDEEFF0011ULL});    // memory still written
}

// ---- (a) byte/half LSE bitwise via the sized CMPXCHG retry loop --------------

// LDSETB: [mem].b |= Ws.b; old byte -> Wt zero-extended.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldset8ZeroExtends) {
  alignas(8) static uint64_t buf = 0xAABBCCDDEEFF0011ULL;  // low byte = 0x11
  static const uint32_t code[] = {0x38213040u};  // ldsetb w1, w0, [x2]
  state_.cpu.x[1] = 0x00000000000000F0ULL;         // Ws byte = 0xF0
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xAABBCCDDEEFF00F1ULL});         // only low byte OR'd (0x11|0xF0)
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0000000000000011ULL});  // old byte zero-extended
}

// LDCLRB: [mem].b &= ~Ws.b; upper bytes untouched.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldclr8ZeroExtends) {
  alignas(8) static uint64_t buf = 0x1122334455667788ULL;  // low byte = 0x88
  static const uint32_t code[] = {0x38211040u};  // ldclrb w1, w0, [x2]
  state_.cpu.x[1] = 0x000000000000000FULL;         // clear low 4 bits (~0x0F = 0xF0 mask)
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x1122334455667780ULL});         // 0x88 & ~0x0F = 0x80
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0000000000000088ULL});  // old byte
}

// LDEORH: [mem].h ^= Ws.h; halfword truncation + zero-extension.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldeor16ZeroExtends) {
  alignas(8) static uint64_t buf = 0xDEADBEEF0000FFFFULL;  // low half = 0xFFFF
  static const uint32_t code[] = {0x78212040u};  // ldeorh w1, w0, [x2]
  state_.cpu.x[1] = 0x000000000000FF00ULL;         // Ws half = 0xFF00
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xDEADBEEF000000FFULL});         // 0xFFFF ^ 0xFF00 = 0x00FF
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x000000000000FFFFULL});  // old half zero-extended
}

// ---- (a) byte/half LSE min/max: signed-vs-unsigned distinguishing cases ------

// LDSMAXB signed: mem=0xFF(=-1), Xs=1 -> signed max = 1 written.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldsmax8Signed) {
  alignas(8) static uint64_t buf = 0x11223344556677FFULL;  // low byte 0xFF = -1 signed
  static const uint32_t code[] = {0x38214040u};  // ldsmaxb w1, w0, [x2]
  state_.cpu.x[1] = 0x0000000000000001ULL;         // Ws = 1
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x1122334455667701ULL});         // max_s(-1,1)=1; upper bytes kept
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x00000000000000FFULL});  // old byte zero-extended
}

// LDUMAXB unsigned, SAME bytes: unsigned max = 0xFF (255) keeps memory.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldumax8Unsigned) {
  alignas(8) static uint64_t buf = 0x11223344556677FFULL;  // low byte 0xFF = 255 unsigned
  static const uint32_t code[] = {0x38216040u};  // ldumaxb w1, w0, [x2]
  state_.cpu.x[1] = 0x0000000000000001ULL;
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0x11223344556677FFULL});         // max_u(255,1)=255; unchanged
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x00000000000000FFULL});
}

// LDSMINH signed: mem=5, Xs=0xFFFF(=-1 signed half) -> signed min = -1 (stored as 0xFFFF).
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldsmin16Signed) {
  alignas(8) static uint64_t buf = 0xAAAAAAAAAAAA0005ULL;  // low half = 5
  static const uint32_t code[] = {0x78215040u};  // ldsminh w1, w0, [x2]
  state_.cpu.x[1] = 0x000000000000FFFFULL;         // Ws half = -1 signed
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xAAAAAAAAAAAAFFFFULL});         // min_s(5,-1) = -1
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0000000000000005ULL});  // old half zero-extended
}

// LDUMINH unsigned, SAME bits: unsigned min(5, 65535) = 5 keeps memory.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldumin16Unsigned) {
  alignas(8) static uint64_t buf = 0xAAAAAAAAAAAA0005ULL;  // low half = 5
  static const uint32_t code[] = {0x78217040u};  // lduminh w1, w0, [x2]
  state_.cpu.x[1] = 0x000000000000FFFFULL;         // Ws half = 65535 unsigned
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf, uint64_t{0xAAAAAAAAAAAA0005ULL});         // min_u(5,65535)=5; unchanged
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0000000000000005ULL});
}

// ---- (b) 128-bit pair — CMPXCHG16B (validate 4-fixed-reg allocation) ---------

// CASP (64-bit pair) match: X4:X5 == [mem], so X6:X7 stored; old pair -> X4:X5.
TEST_F(Arm64HeavyOptimizerFrontendTest, Casp64Match) {
  alignas(16) static uint64_t buf[2] = {0x1111111122222222ULL, 0x3333333344444444ULL};
  static const uint32_t code[] = {0x48247C46u};  // casp x4, x5, x6, x7, [x2]
  state_.cpu.x[4] = 0x1111111122222222ULL;  // expected.lo (matches buf[0])
  state_.cpu.x[5] = 0x3333333344444444ULL;  // expected.hi (matches buf[1])
  state_.cpu.x[6] = 0xAAAAAAAAAAAAAAAAULL;  // desired.lo
  state_.cpu.x[7] = 0xBBBBBBBBBBBBBBBBULL;  // desired.hi
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], uint64_t{0xAAAAAAAAAAAAAAAAULL});  // stored on match
  EXPECT_EQ(buf[1], uint64_t{0xBBBBBBBBBBBBBBBBULL});
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0x1111111122222222ULL});  // old.lo
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0x3333333344444444ULL});  // old.hi
}

// CASP (64-bit pair) mismatch: memory untouched; X4:X5 get the actual old pair.
TEST_F(Arm64HeavyOptimizerFrontendTest, Casp64Mismatch) {
  alignas(16) static uint64_t buf[2] = {0xDEADBEEFDEADBEEFULL, 0xCAFEBABECAFEBABEULL};
  static const uint32_t code[] = {0x48247C46u};  // casp x4, x5, x6, x7, [x2]
  state_.cpu.x[4] = 0x0000000000000001ULL;  // expected.lo (mismatch)
  state_.cpu.x[5] = 0x0000000000000002ULL;  // expected.hi (mismatch)
  state_.cpu.x[6] = 0xAAAAAAAAAAAAAAAAULL;
  state_.cpu.x[7] = 0xBBBBBBBBBBBBBBBBULL;
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], uint64_t{0xDEADBEEFDEADBEEFULL});  // unchanged
  EXPECT_EQ(buf[1], uint64_t{0xCAFEBABECAFEBABEULL});
  EXPECT_EQ(state_.cpu.x[4], uint64_t{0xDEADBEEFDEADBEEFULL});  // actual old.lo
  EXPECT_EQ(state_.cpu.x[5], uint64_t{0xCAFEBABECAFEBABEULL});  // actual old.hi
}

// LDXP (64-bit pair): loads X0:X1 and arms the monitor.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ldxp64) {
  alignas(16) static uint64_t buf[2] = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
  static const uint32_t code[] = {0xC87F0440u};  // ldxp x0, x1, [x2]
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(state_.cpu.x[0], uint64_t{0x0123456789ABCDEFULL});  // low 64
  EXPECT_EQ(state_.cpu.x[1], uint64_t{0xFEDCBA9876543210ULL});  // high 64
  EXPECT_EQ(state_.cpu.reservation_address, ToGuestAddr(&buf));
}

// STXP (64-bit pair) success: reservation matches [mem], so X0:X1 is stored and
// Rs (W3) = 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, Stxp64Success) {
  alignas(16) static uint64_t buf[2] = {0x1111111111111111ULL, 0x2222222222222222ULL};
  static const uint32_t code[] = {0xC8230440u};  // stxp w3, x0, x1, [x2]
  state_.cpu.x[0] = 0xAAAAAAAAAAAAAAAAULL;  // new.lo
  state_.cpu.x[1] = 0xBBBBBBBBBBBBBBBBULL;  // new.hi
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.reservation_address = ToGuestAddr(&buf);
  state_.cpu.reservation_value =
      (static_cast<unsigned __int128>(0x2222222222222222ULL) << 64) | 0x1111111111111111ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], uint64_t{0xAAAAAAAAAAAAAAAAULL});  // stored
  EXPECT_EQ(buf[1], uint64_t{0xBBBBBBBBBBBBBBBBULL});
  EXPECT_EQ(state_.cpu.x[3], uint64_t{0});             // success status
}

// STXP (64-bit pair) failure: memory changed under us (reservation_value stale),
// so CMPXCHG16B fails, memory is untouched, Rs (W3) = 1.
TEST_F(Arm64HeavyOptimizerFrontendTest, Stxp64FailStale) {
  alignas(16) static uint64_t buf[2] = {0x9999999999999999ULL, 0x8888888888888888ULL};
  static const uint32_t code[] = {0xC8230440u};  // stxp w3, x0, x1, [x2]
  state_.cpu.x[0] = 0xAAAAAAAAAAAAAAAAULL;
  state_.cpu.x[1] = 0xBBBBBBBBBBBBBBBBULL;
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.reservation_address = ToGuestAddr(&buf);
  // reservation_value is what LDXP *thought* was there; memory now differs.
  state_.cpu.reservation_value =
      (static_cast<unsigned __int128>(0x2222222222222222ULL) << 64) | 0x1111111111111111ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], uint64_t{0x9999999999999999ULL});  // untouched
  EXPECT_EQ(buf[1], uint64_t{0x8888888888888888ULL});
  EXPECT_EQ(state_.cpu.x[3], uint64_t{1});             // failure status
}

// STXP failure: reservation address does not match base -> store fails, W3 = 1.
TEST_F(Arm64HeavyOptimizerFrontendTest, Stxp64FailAddrMismatch) {
  alignas(16) static uint64_t buf[2] = {0x1111111111111111ULL, 0x2222222222222222ULL};
  static const uint32_t code[] = {0xC8230440u};  // stxp w3, x0, x1, [x2]
  state_.cpu.x[0] = 0xAAAAAAAAAAAAAAAAULL;
  state_.cpu.x[1] = 0xBBBBBBBBBBBBBBBBULL;
  state_.cpu.x[2] = ToGuestAddr(&buf);
  state_.cpu.reservation_address = ToGuestAddr(&buf) + 0x1000;  // different address
  state_.cpu.reservation_value =
      (static_cast<unsigned __int128>(0x2222222222222222ULL) << 64) | 0x1111111111111111ULL;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], uint64_t{0x1111111111111111ULL});  // untouched (no CMPXCHG at all)
  EXPECT_EQ(buf[1], uint64_t{0x2222222222222222ULL});
  EXPECT_EQ(state_.cpu.x[3], uint64_t{1});             // failure status
}

//
// AdvSIMD two-register-miscellaneous heavy-tier mirror: REV16, CNT, NOT, RBIT,
// NEG, ABS. Each drives the full pipeline via RunRegion (ok stays true only if
// the region translated rather than bailing).
//

// REV16 .16B (Q=1): reverse bytes within each 16-bit lane.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rev16Vec16B) {
  static const uint32_t code[] = {Rev16Vec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x1122334455667788ULL, 0x99AABBCCDDEEFF00ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x2211443366558877ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xAA99CCBBEEDD00FFULL);
}

// REV64 .16B (size=00, Q=1): byte-reverse each 64-bit doubleword.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rev64Vec16B) {
  static const uint32_t code[] = {Rev64Vec(0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7766554433221100ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFEEDDCCBBAA9988ULL);
}

// REV64 .8H (size=01, Q=1): reverse the four halfwords in each doubleword.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rev64Vec8H) {
  static const uint32_t code[] = {Rev64Vec(0b01, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x6677445522330011ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xEEFFCCDDAABB8899ULL);
}

// REV64 .4S (size=10, Q=1): swap the two 32-bit words in each doubleword.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rev64Vec4S) {
  static const uint32_t code[] = {Rev64Vec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4455667700112233ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xCCDDEEFF8899AABBULL);
}

// REV64 .2S (size=10, Q=0): low doubleword only, upper 64 zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rev64Vec2S) {
  static const uint32_t code[] = {Rev64Vec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4455667700112233ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ULL);
}

// REV32 .16B (size=00, Q=1): byte-reverse each 32-bit word.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rev32Vec16B) {
  static const uint32_t code[] = {Rev32Vec(0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3322110077665544ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBBAA9988FFEEDDCCULL);
}

// REV32 .8H (size=01, Q=1): swap the two halfwords in each 32-bit word.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rev32Vec8H) {
  static const uint32_t code[] = {Rev32Vec(0b01, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x2233001166774455ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xAABB8899EEFFCCDDULL);
}

// REV32 .8B (size=00, Q=0): low word-group only, upper 64 zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rev32Vec8B) {
  static const uint32_t code[] = {Rev32Vec(0b00, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x3322110077665544ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ULL);
}

// EXT .8B (size implied, Q=0), #3: window from the Vm:Vn concatenation with Vn
// low. Result bytes 5..7 must come from Vm; upper 64 zeroed. Same vector as the
// lite ExtByte8bInterpreter test (insn 0x2e021820).
TEST_F(Arm64HeavyOptimizerFrontendTest, ExtVec8B) {
  static const uint32_t code[] = {ExtVec(/*q=*/false, 0, 1, 2, /*imm4=*/3)};
  SetV128(&state_, 1, 0x8138268683868942ULL, 0x7741559918559252ULL);  // Vn; hi ignored
  SetV128(&state_, 2, 0x3622262609912460ULL, 0x8051243884390451ULL);  // Vm
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);   // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x9124608138268683ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ULL);
}

// EXT .16B (Q=1), #4: window from the 32-byte Vm:Vn concatenation starting at
// byte 4 (Vn low, Vm high).
TEST_F(Arm64HeavyOptimizerFrontendTest, ExtVec16B) {
  static const uint32_t code[] = {ExtVec(/*q=*/true, 0, 1, 2, /*imm4=*/4)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);  // Vn
  SetV128(&state_, 2, 0x0102030405060708ULL, 0x090A0B0C0D0E0F10ULL);  // Vm
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);   // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xCCDDEEFF00112233ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x050607088899AABBULL);
}

// EXT .16B (Q=1), #0: index 0 copies Vn verbatim (fast path, no byte shift).
TEST_F(Arm64HeavyOptimizerFrontendTest, ExtVec16BIndex0) {
  static const uint32_t code[] = {ExtVec(/*q=*/true, 0, 1, 2, /*imm4=*/0)};
  SetV128(&state_, 1, 0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL);  // Vn
  SetV128(&state_, 2, 0x0102030405060708ULL, 0x090A0B0C0D0E0F10ULL);  // Vm
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);   // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0011223344556677ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x8899AABBCCDDEEFFULL);
}

// ZIP1/ZIP2, UZP1/UZP2, TRN1/TRN2 vector permute — heavy-tier exec tests.
// Encodings and inputs/expected values mirror the verified lite exec tests
// (lite_translate_region_exec_tests.cc); each exercises a distinct heavy
// lowering path (PUNPCK, PACKUS, SHUFPS, PSHUFB+POR, PSHUFD+PUNPCKLDQ).
// v1 = Vn, v2 = Vm, v0 = Vd (poisoned 0xAA to catch missing writes/upper-zero).
constexpr uint32_t kZip1Vec16B = 0x4e023820u;
constexpr uint32_t kZip2Vec16B = 0x4e027820u;
constexpr uint32_t kZip1Vec8H  = 0x4e423820u;
constexpr uint32_t kZip1Vec4S  = 0x4e823820u;
constexpr uint32_t kZip2Vec4S  = 0x4e827820u;
constexpr uint32_t kZip1Vec2D  = 0x4ec23820u;
constexpr uint32_t kZip2Vec2D  = 0x4ec27820u;
constexpr uint32_t kZip1Vec8B  = 0x0e023820u;
constexpr uint32_t kZip2Vec8B  = 0x0e027820u;
constexpr uint32_t kUzp1Vec16B = 0x4e021820u;
constexpr uint32_t kUzp2Vec16B = 0x4e025820u;
constexpr uint32_t kUzp1Vec8H  = 0x4e421820u;
constexpr uint32_t kUzp1Vec4S  = 0x4e821820u;
constexpr uint32_t kUzp2Vec4S  = 0x4e825820u;
constexpr uint32_t kUzp1Vec2D  = 0x4ec21820u;
constexpr uint32_t kUzp1Vec8B  = 0x0e021820u;
constexpr uint32_t kUzp1Vec2S  = 0x0e821820u;
constexpr uint32_t kTrn1Vec16B = 0x4e022820u;
constexpr uint32_t kTrn2Vec16B = 0x4e026820u;
constexpr uint32_t kTrn1Vec8H  = 0x4e422820u;
constexpr uint32_t kTrn2Vec8H  = 0x4e426820u;
constexpr uint32_t kTrn1Vec4S  = 0x4e822820u;
constexpr uint32_t kTrn2Vec4S  = 0x4e826820u;
constexpr uint32_t kTrn1Vec2D  = 0x4ec22820u;
constexpr uint32_t kTrn1Vec8B  = 0x0e022820u;
constexpr uint32_t kTrn1Vec2S  = 0x0e822820u;

// ---- ZIP ----

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip1Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec16B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[2 * i], vn[i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[i]) << "lane " << i;
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip2Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec16B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[2 * i], vn[8 + i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[8 + i]) << "lane " << i;
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip1Vec8H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x0908,0x0B0A,0x0D0C,0x0F0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x1918,0x1B1A,0x1D1C,0x1F1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec8H};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i], vn[i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[i]) << "lane " << i;
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip1Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec4S};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], vn[1]);
  EXPECT_EQ(r[3], vm[1]);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip2Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec4S};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[2]);
  EXPECT_EQ(r[1], vm[2]);
  EXPECT_EQ(r[2], vn[3]);
  EXPECT_EQ(r[3], vm[3]);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip1Vec2D) {
  SetV128(&state_, 1, 0x1111111111111111ULL, 0x2222222222222222ULL);
  SetV128(&state_, 2, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);
  static const uint32_t code[] = {kZip1Vec2D};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111111111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xAAAAAAAAAAAAAAAAULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip2Vec2D) {
  SetV128(&state_, 1, 0x1111111111111111ULL, 0x2222222222222222ULL);
  SetV128(&state_, 2, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);
  static const uint32_t code[] = {kZip2Vec2D};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x2222222222222222ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBBBBBBBBBBBBBBBBULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip1Vec8B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip1Vec8B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i], vn[i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[i]) << "lane " << i;
  }
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Zip2Vec8B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kZip2Vec8B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i], vn[4 + i]) << "lane " << i;
    EXPECT_EQ(r[2 * i + 1], vm[4 + i]) << "lane " << i;
  }
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

// ---- UZP ----

TEST_F(Arm64HeavyOptimizerFrontendTest, Uzp1Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec16B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[i], vn[2 * i]) << "vn byte " << i;
    EXPECT_EQ(r[8 + i], vm[2 * i]) << "vm byte " << i;
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Uzp2Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec16B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[i], vn[2 * i + 1]) << "vn byte " << i;
    EXPECT_EQ(r[8 + i], vm[2 * i + 1]) << "vm byte " << i;
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Uzp1Vec8H) {
  // High-bit halfwords pin PACKUSDW: each masked dword is in 0..0xFFFF so
  // saturation must be a no-op.
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x8908,0x8B0A,0xFD0C,0xFF0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x9918,0x9B1A,0xED1C,0xEF1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec8H};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[i], vn[2 * i]) << "vn halfword " << i;
    EXPECT_EQ(r[4 + i], vm[2 * i]) << "vm halfword " << i;
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Uzp1Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec4S};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vn[2]);
  EXPECT_EQ(r[2], vm[0]);
  EXPECT_EQ(r[3], vm[2]);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Uzp2Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp2Vec4S};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vn[3]);
  EXPECT_EQ(r[2], vm[1]);
  EXPECT_EQ(r[3], vm[3]);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Uzp1Vec2D) {
  SetV128(&state_, 1, 0x1111111111111111ULL, 0x2222222222222222ULL);
  SetV128(&state_, 2, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);
  static const uint32_t code[] = {kUzp1Vec2D};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111111111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xAAAAAAAAAAAAAAAAULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Uzp1Vec8B) {
  // Upper 8 bytes of vn/vm are garbage and MUST NOT leak into the result.
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec8B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vn[2]);
  EXPECT_EQ(r[2], vn[4]);
  EXPECT_EQ(r[3], vn[6]);
  EXPECT_EQ(r[4], vm[0]);
  EXPECT_EQ(r[5], vm[2]);
  EXPECT_EQ(r[6], vm[4]);
  EXPECT_EQ(r[7], vm[6]);
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Uzp1Vec2S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0xDEADBEEFu, 0xCAFEBABEu};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0x12345678u, 0x9ABCDEF0u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kUzp1Vec2S};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}

// ---- TRN ----

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn1Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec16B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[2 * i], vn[2 * i]) << "lane " << (2 * i);
    EXPECT_EQ(r[2 * i + 1], vm[2 * i]) << "lane " << (2 * i + 1);
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn2Vec16B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec16B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(r[2 * i], vn[2 * i + 1]) << "lane " << (2 * i);
    EXPECT_EQ(r[2 * i + 1], vm[2 * i + 1]) << "lane " << (2 * i + 1);
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn1Vec8H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x8908,0x8B0A,0xFD0C,0xFF0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x9918,0x9B1A,0xED1C,0xEF1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec8H};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i], vn[2 * i]) << "lane " << (2 * i);
    EXPECT_EQ(r[2 * i + 1], vm[2 * i]) << "lane " << (2 * i + 1);
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn2Vec8H) {
  uint16_t vn[8] = {0x0100,0x0302,0x0504,0x0706,0x8908,0x8B0A,0xFD0C,0xFF0E};
  uint16_t vm[8] = {0x1110,0x1312,0x1514,0x1716,0x9918,0x9B1A,0xED1C,0xEF1E};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec8H};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint16_t r[8];
  std::memcpy(r, &state_.cpu.v[0], 16);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r[2 * i], vn[2 * i + 1]) << "lane " << (2 * i);
    EXPECT_EQ(r[2 * i + 1], vm[2 * i + 1]) << "lane " << (2 * i + 1);
  }
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn1Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec4S};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], vn[2]);
  EXPECT_EQ(r[3], vm[2]);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn2Vec4S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn2Vec4S};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[1]);
  EXPECT_EQ(r[1], vm[1]);
  EXPECT_EQ(r[2], vn[3]);
  EXPECT_EQ(r[3], vm[3]);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn1Vec2D) {
  SetV128(&state_, 1, 0x1111111111111111ULL, 0x2222222222222222ULL);
  SetV128(&state_, 2, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);
  static const uint32_t code[] = {kTrn1Vec2D};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111111111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xAAAAAAAAAAAAAAAAULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn1Vec8B) {
  uint8_t vn[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                    0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8};
  uint8_t vm[16] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                    0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec8B};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t r[16];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], vn[2]);
  EXPECT_EQ(r[3], vm[2]);
  EXPECT_EQ(r[4], vn[4]);
  EXPECT_EQ(r[5], vm[4]);
  EXPECT_EQ(r[6], vn[6]);
  EXPECT_EQ(r[7], vm[6]);
  for (int i = 8; i < 16; ++i) EXPECT_EQ(r[i], 0u) << "upper byte " << i;
}

TEST_F(Arm64HeavyOptimizerFrontendTest, Trn1Vec2S) {
  uint32_t vn[4] = {0x11111111u, 0x22222222u, 0xDEADBEEFu, 0xCAFEBABEu};
  uint32_t vm[4] = {0xAAAAAAAAu, 0xBBBBBBBBu, 0x12345678u, 0x9ABCDEF0u};
  std::memcpy(&state_.cpu.v[1], vn, 16);
  std::memcpy(&state_.cpu.v[2], vm, 16);
  std::memset(&state_.cpu.v[0], 0xAA, 16);
  static const uint32_t code[] = {kTrn1Vec2S};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint32_t r[4];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], vn[0]);
  EXPECT_EQ(r[1], vm[0]);
  EXPECT_EQ(r[2], 0u);
  EXPECT_EQ(r[3], 0u);
}

// AdvSimdTableLookup heavy-tier exec tests — TBL/TBX (8B/16B forms, 1..4 table
// registers). Encodings/inputs/expected values mirror the validated lite
// exec tests verbatim (JIT-vs-interpreter proven there).
//   Encoding: bits[31]=0, bits[30]=Q, bits[29:24]=001110, bits[23:22]=00,
//   bit21=0, bits[20:16]=Rm, bit15=0, bits[14:13]=len, bit12=op, bits[11:10]=00,
//   bits[9:5]=Rn, bits[4:0]=Rd. Q=0 → 8B form, Q=1 → 16B form.
//   len ∈ {0,1,2,3} → 1..4 table registers; op: 0=TBL, 1=TBX.
constexpr uint32_t TblTbxEnc(uint8_t q, uint8_t rd, uint8_t rn, uint8_t rm,
                             uint8_t len, uint8_t op) {
  return (static_cast<uint32_t>(q) << 30) | 0x0E000000u |
         (static_cast<uint32_t>(rm) << 16) |
         (static_cast<uint32_t>(len) << 13) |
         (static_cast<uint32_t>(op) << 12) |
         (static_cast<uint32_t>(rn) << 5) | rd;
}

// TBL Vd.16B, {Vn.16B}, Vm.16B — identity permutation selects Vn unchanged.
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl16BIdentity) {
  uint8_t table_bytes[16];
  uint8_t idx_bytes[16];
  for (int i = 0; i < 16; ++i) {
    table_bytes[i] = static_cast<uint8_t>(0xA0 + i);
    idx_bytes[i]   = static_cast<uint8_t>(i);
  }
  std::memcpy(&state_.cpu.v[1], table_bytes, 16);
  std::memcpy(&state_.cpu.v[2], idx_bytes, 16);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {TblTbxEnc(1, 0, 1, 2, 0, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out[i], table_bytes[i]) << "byte " << i;
  }
}

// TBL — out-of-range index gives 0 (NOT a wrapped lookup).
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl16BOutOfRangeZeros) {
  uint8_t table_bytes[16];
  uint8_t idx_bytes[16];
  for (int i = 0; i < 16; ++i) {
    table_bytes[i] = static_cast<uint8_t>(0x10 + i);
  }
  for (int i = 0; i < 8; ++i)  idx_bytes[i] = static_cast<uint8_t>(i);
  for (int i = 8; i < 16; ++i) idx_bytes[i] = static_cast<uint8_t>(16 + i);
  std::memcpy(&state_.cpu.v[1], table_bytes, 16);
  std::memcpy(&state_.cpu.v[2], idx_bytes, 16);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {TblTbxEnc(1, 0, 1, 2, 0, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out[i], table_bytes[i]) << "byte " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(out[i], 0u) << "byte " << i << " (out-of-range must be 0)";
  }
}

// TBL — high-bit indices (128..255) yield 0 (native PSHUFB behaviour).
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl16BHighBitIndexZeros) {
  uint8_t table_bytes[16];
  uint8_t idx_bytes[16];
  for (int i = 0; i < 16; ++i) {
    table_bytes[i] = static_cast<uint8_t>(0xC0 + i);
    idx_bytes[i]   = static_cast<uint8_t>(0x80 | i);
  }
  std::memcpy(&state_.cpu.v[1], table_bytes, 16);
  std::memcpy(&state_.cpu.v[2], idx_bytes, 16);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {TblTbxEnc(1, 0, 1, 2, 0, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(reinterpret_cast<const uint8_t*>(&state_.cpu.v[0])[i], 0u) << "byte " << i;
  }
}

// TBL Vd.8B (Q=0): only low 8 bytes computed; upper 8 zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl8BUpperZero) {
  uint8_t table_bytes[16];
  uint8_t idx_bytes[16];
  for (int i = 0; i < 16; ++i) {
    table_bytes[i] = static_cast<uint8_t>(0xE0 + i);
    idx_bytes[i]   = static_cast<uint8_t>(i);
  }
  std::memcpy(&state_.cpu.v[1], table_bytes, 16);
  std::memcpy(&state_.cpu.v[2], idx_bytes, 16);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {TblTbxEnc(0, 0, 1, 2, 0, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out[i], table_bytes[i]) << "byte " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(out[i], 0u) << "byte " << i << " (upper must be 0)";
  }
}

// TBX — out-of-range index preserves Vd (the key TBX-vs-TBL difference).
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbx16BPreservesVdOnOOR) {
  uint8_t table_bytes[16];
  uint8_t idx_bytes[16];
  uint8_t vd_initial[16];
  for (int i = 0; i < 16; ++i) {
    table_bytes[i] = static_cast<uint8_t>(0x40 + i);
    vd_initial[i]  = static_cast<uint8_t>(0x90 + i);
  }
  for (int i = 0; i < 8; ++i)  idx_bytes[i] = static_cast<uint8_t>(i);
  for (int i = 8; i < 16; ++i) idx_bytes[i] = static_cast<uint8_t>(16 + i);
  std::memcpy(&state_.cpu.v[0], vd_initial, 16);
  std::memcpy(&state_.cpu.v[1], table_bytes, 16);
  std::memcpy(&state_.cpu.v[2], idx_bytes, 16);
  static const uint32_t code[] = {TblTbxEnc(1, 0, 1, 2, 0, 1)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out[i], table_bytes[i]) << "byte " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(out[i], vd_initial[i]) << "byte " << i << " (TBX preserves Vd)";
  }
}

// TBX Vd.8B (Q=0): preserved bytes in low 8 honour Vd; upper 8 always 0.
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbx8BUpperZeroDespiteVd) {
  uint8_t table_bytes[16];
  uint8_t idx_bytes[16];
  uint8_t vd_initial[16];
  for (int i = 0; i < 16; ++i) {
    table_bytes[i] = static_cast<uint8_t>(0x50 + i);
    vd_initial[i]  = static_cast<uint8_t>(0xA0 + i);
    idx_bytes[i]   = static_cast<uint8_t>(0xFF);
  }
  std::memcpy(&state_.cpu.v[0], vd_initial, 16);
  std::memcpy(&state_.cpu.v[1], table_bytes, 16);
  std::memcpy(&state_.cpu.v[2], idx_bytes, 16);
  static const uint32_t code[] = {TblTbxEnc(0, 0, 1, 2, 0, 1)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out[i], vd_initial[i]) << "byte " << i;
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_EQ(out[i], 0u) << "byte " << i;
  }
}

// TBL 2-reg table: indices 0..15 → Vn, 16..31 → V(n+1).
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl16B2RegSpan) {
  uint8_t t0[16], t1[16], idx[16];
  for (int i = 0; i < 16; ++i) {
    t0[i]  = static_cast<uint8_t>(0x20 + i);
    t1[i]  = static_cast<uint8_t>(0x60 + i);
    idx[i] = static_cast<uint8_t>(2 * i);
  }
  std::memcpy(&state_.cpu.v[1], t0,  16);
  std::memcpy(&state_.cpu.v[2], t1,  16);
  std::memcpy(&state_.cpu.v[3], idx, 16);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {TblTbxEnc(1, 0, 1, 3, 1, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    int j = 2 * i;
    uint8_t expected = (j < 16) ? t0[j] : t1[j - 16];
    EXPECT_EQ(out[i], expected) << "byte " << i << " idx=" << j;
  }
}

// TBL 3-reg table: transitions around register boundaries (15/16, 31/32).
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl16B3RegBoundaries) {
  uint8_t t0[16], t1[16], t2[16], idx[16];
  for (int i = 0; i < 16; ++i) {
    t0[i] = static_cast<uint8_t>(0x10 + i);
    t1[i] = static_cast<uint8_t>(0x50 + i);
    t2[i] = static_cast<uint8_t>(0x90 + i);
  }
  uint8_t test_idx[16] = {14, 15, 16, 17, 30, 31, 32, 33,
                          47, 48, 50, 60, 0, 16, 32, 47};
  std::memcpy(idx, test_idx, 16);
  std::memcpy(&state_.cpu.v[1], t0,  16);
  std::memcpy(&state_.cpu.v[2], t1,  16);
  std::memcpy(&state_.cpu.v[3], t2,  16);
  std::memcpy(&state_.cpu.v[4], idx, 16);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {TblTbxEnc(1, 0, 1, 4, 2, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  auto lookup = [&](int j) -> uint8_t {
    if (j < 16)      return t0[j];
    else if (j < 32) return t1[j - 16];
    else if (j < 48) return t2[j - 32];
    else             return 0;
  };
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out[i], lookup(test_idx[i])) << "byte " << i;
  }
}

// TBL 4-reg table: all four registers plus upper-OOR (64..127) zero case.
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl16B4RegFull) {
  uint8_t t[4][16], idx[16];
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < 16; ++i) {
      t[r][i] = static_cast<uint8_t>((r << 4) | i);
    }
  }
  uint8_t test_idx[16] = {0, 16, 32, 48, 15, 31, 47, 63,
                          64, 100, 1, 17, 33, 49, 60, 70};
  std::memcpy(idx, test_idx, 16);
  std::memcpy(&state_.cpu.v[1], t[0], 16);
  std::memcpy(&state_.cpu.v[2], t[1], 16);
  std::memcpy(&state_.cpu.v[3], t[2], 16);
  std::memcpy(&state_.cpu.v[4], t[3], 16);
  std::memcpy(&state_.cpu.v[5], idx, 16);
  state_.cpu.v[0] = ~__uint128_t{0};
  static const uint32_t code[] = {TblTbxEnc(1, 0, 1, 5, 3, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  for (int i = 0; i < 16; ++i) {
    uint8_t expected = (test_idx[i] < 64) ? test_idx[i] : 0u;
    EXPECT_EQ(out[i], expected) << "byte " << i;
  }
}

// TBX 2-reg: in-range bytes get the lookup; OOR bytes preserve Vd.
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbx16B2RegMixed) {
  uint8_t t0[16], t1[16], vd_initial[16], idx[16];
  for (int i = 0; i < 16; ++i) {
    t0[i]         = static_cast<uint8_t>(0x30 + i);
    t1[i]         = static_cast<uint8_t>(0x70 + i);
    vd_initial[i] = static_cast<uint8_t>(0xB0 + i);
  }
  uint8_t test_idx[16] = {0, 1, 16, 17, 32, 200, 33, 100,
                          15, 31, 14, 30, 64, 0, 50, 31};
  std::memcpy(idx, test_idx, 16);
  std::memcpy(&state_.cpu.v[0], vd_initial, 16);
  std::memcpy(&state_.cpu.v[1], t0, 16);
  std::memcpy(&state_.cpu.v[2], t1, 16);
  std::memcpy(&state_.cpu.v[3], idx, 16);
  static const uint32_t code[] = {TblTbxEnc(1, 0, 1, 3, 1, 1)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[0], 16);
  auto lookup = [&](int j, int byte_i) -> uint8_t {
    if (j < 16)      return t0[j];
    else if (j < 32) return t1[j - 16];
    else             return vd_initial[byte_i];
  };
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out[i], lookup(test_idx[i], i)) << "byte " << i;
  }
}

// TBL with Vd == Vn (in-place table): table read before Vd written.
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl16BInPlaceVdEqualsVn) {
  uint8_t table_bytes[16];
  uint8_t idx_bytes[16];
  for (int i = 0; i < 16; ++i) {
    table_bytes[i] = static_cast<uint8_t>(0xC0 + i);
    idx_bytes[i]   = static_cast<uint8_t>(15 - i);
  }
  std::memcpy(&state_.cpu.v[5], table_bytes, 16);
  std::memcpy(&state_.cpu.v[2], idx_bytes, 16);
  static const uint32_t code[] = {TblTbxEnc(1, 5, 5, 2, 0, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[5], 16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out[i], table_bytes[15 - i]) << "byte " << i;
  }
}

// TBL with Vd == Vm (in-place index): index read before Vd written.
TEST_F(Arm64HeavyOptimizerFrontendTest, Tbl16BInPlaceVdEqualsVm) {
  uint8_t table_bytes[16];
  uint8_t idx_bytes[16];
  for (int i = 0; i < 16; ++i) {
    table_bytes[i] = static_cast<uint8_t>(0xD0 + i);
    idx_bytes[i]   = static_cast<uint8_t>((i * 3) & 0x0F);
  }
  std::memcpy(&state_.cpu.v[1], table_bytes, 16);
  std::memcpy(&state_.cpu.v[7], idx_bytes, 16);
  static const uint32_t code[] = {TblTbxEnc(1, 7, 1, 7, 0, 0)};
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  uint8_t out[16];
  std::memcpy(out, &state_.cpu.v[7], 16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out[i], table_bytes[idx_bytes[i]]) << "byte " << i;
  }
}

// UMOV / SMOV / INS (element). Vn (v1) throughout for the register-move tests:
//   lo = 0xF0E1D2C3B4A59687  -> b0=0x87 b1=0x96 b2=0xA5 b3=0xB4 b4=0xC3
//                              b5=0xD2 b6=0xE1 b7=0xF0
//   hi = 0x0F1E2D3C4B5A6978  -> b8=0x78 b9=0x69 b10=0x5A b11=0x4B ...
//   h1=0xB4A5 h3=0xF0E1  s1=0xF0E1D2C3 s2=0x4B5A6978  d1=0x0F1E2D3C4B5A6978
constexpr uint64_t kCopyVnLo = 0xF0E1D2C3B4A59687ULL;
constexpr uint64_t kCopyVnHi = 0x0F1E2D3C4B5A6978ULL;

// UMOV Wd, Vn.B[3]: byte 3 (0xB4) zero-extended into W0 (X0 upper = 0).
TEST_F(Arm64HeavyOptimizerFrontendTest, UmovWB3) {
  static const uint32_t code[] = {0x0e073c20u};  // umov w0, v1.b[3]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0x00000000000000B4ULL);
}

// UMOV Wd, Vn.H[1]: halfword 1 (0xB4A5) zero-extended.
TEST_F(Arm64HeavyOptimizerFrontendTest, UmovWH1) {
  static const uint32_t code[] = {0x0e063c20u};  // umov w0, v1.h[1]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0x000000000000B4A5ULL);
}

// UMOV Wd, Vn.S[2]: word 2 (0x4B5A6978) zero-extended.
TEST_F(Arm64HeavyOptimizerFrontendTest, UmovWS2) {
  static const uint32_t code[] = {0x0e143c20u};  // umov w0, v1.s[2]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  state_.cpu.x[0] = 0xDEADBEEFDEADBEEFULL;
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0x000000004B5A6978ULL);
}

// UMOV Xd, Vn.D[1]: full doubleword 1.
TEST_F(Arm64HeavyOptimizerFrontendTest, UmovXD1) {
  static const uint32_t code[] = {0x4e183c20u};  // umov x0, v1.d[1]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0x0F1E2D3C4B5A6978ULL);
}

// SMOV Wd, Vn.B[2]: byte 2 (0xA5, negative) sign-extended to 32 (W0 upper zero).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmovWB2) {
  static const uint32_t code[] = {0x0e052c20u};  // smov w0, v1.b[2]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0x00000000FFFFFFA5ULL);
}

// SMOV Xd, Vn.B[0]: byte 0 (0x87, negative) sign-extended to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, SmovXB0) {
  static const uint32_t code[] = {0x4e012c20u};  // smov x0, v1.b[0]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0xFFFFFFFFFFFFFF87ULL);
}

// SMOV Xd, Vn.B[8]: byte 8 (0x78, positive) sign-extended to 64 (stays positive).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmovXB8) {
  static const uint32_t code[] = {0x4e112c20u};  // smov x0, v1.b[8]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0x0000000000000078ULL);
}

// SMOV Wd, Vn.H[3]: halfword 3 (0xF0E1, negative) sign-extended to 32.
TEST_F(Arm64HeavyOptimizerFrontendTest, SmovWH3) {
  static const uint32_t code[] = {0x0e0e2c20u};  // smov w0, v1.h[3]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0x00000000FFFFF0E1ULL);
}

// SMOV Xd, Vn.H[3]: halfword 3 (0xF0E1, negative) sign-extended to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, SmovXH3) {
  static const uint32_t code[] = {0x4e0e2c20u};  // smov x0, v1.h[3]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0xFFFFFFFFFFFFF0E1ULL);
}

// SMOV Xd, Vn.S[1]: word 1 (0xF0E1D2C3, negative) sign-extended to 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, SmovXS1) {
  static const uint32_t code[] = {0x4e0c2c20u};  // smov x0, v1.s[1]
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(state_.cpu.x[0], 0xFFFFFFFFF0E1D2C3ULL);
}

// INS Vd.B[5], Vn.B[10]: byte 10 of v1 (0x5A) into byte 5 of v0; rest preserved.
TEST_F(Arm64HeavyOptimizerFrontendTest, InsElemB5FromB10) {
  static const uint32_t code[] = {0x6e0b5420u};  // ins v0.b[5], v1.b[10]
  SetV128(&state_, 0, 0x1111111111111111ULL, 0x2222222222222222ULL);
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x11115A1111111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x2222222222222222ULL);
}

// INS Vd.H[3], Vn.H[1]: halfword 1 of v1 (0xB4A5) into halfword 3 of v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, InsElemH3FromH1) {
  static const uint32_t code[] = {0x6e0e1420u};  // ins v0.h[3], v1.h[1]
  SetV128(&state_, 0, 0x1111111111111111ULL, 0x2222222222222222ULL);
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xB4A5111111111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x2222222222222222ULL);
}

// INS Vd.S[2], Vn.S[0]: word 0 of v1 (0xB4A59687) into word 2 of v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, InsElemS2FromS0) {
  static const uint32_t code[] = {0x6e140420u};  // ins v0.s[2], v1.s[0]
  SetV128(&state_, 0, 0x1111111111111111ULL, 0x2222222222222222ULL);
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111111111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x22222222B4A59687ULL);
}

// INS Vd.D[1], Vn.D[0]: doubleword 0 of v1 into the upper 64 bits of v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, InsElemD1FromD0) {
  static const uint32_t code[] = {0x6e180420u};  // ins v0.d[1], v1.d[0]
  SetV128(&state_, 0, 0x1111111111111111ULL, 0x2222222222222222ULL);
  SetV128(&state_, 1, kCopyVnLo, kCopyVnHi);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111111111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xF0E1D2C3B4A59687ULL);
}

// INS Vd.S[0], Vd.S[3]: rd == rn self-INS — word 3 (0x22222222) into word 0.
// Loading Vn before storing Vd is what makes the aliasing case correct.
TEST_F(Arm64HeavyOptimizerFrontendTest, InsElemSelfS0FromS3) {
  static const uint32_t code[] = {0x6e046400u};  // ins v0.s[0], v0.s[3]
  SetV128(&state_, 0, 0x1111111111111111ULL, 0x2222222222222222ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1111111122222222ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x2222222222222222ULL);
}

// CNT .16B (Q=1): per-byte population count.
TEST_F(Arm64HeavyOptimizerFrontendTest, CntVec16B) {
  static const uint32_t code[] = {CntVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0102030405060708ULL, 0xFFFF0000FF00FFFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0101020102020301ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0808000008000808ULL);
}

// CNT .8B (Q=0): per-byte popcount, upper 64 bits zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, CntVec8B) {
  static const uint32_t code[] = {CntVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0102030405060708ULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0101020102020301ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ULL);
}

// NOT .16B (Q=1): per-lane bitwise complement.
TEST_F(Arm64HeavyOptimizerFrontendTest, NotVec16B) {
  static const uint32_t code[] = {NotVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x00FF00FF00FF00FFULL, 0x123456789ABCDEF0ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF00FF00FF00FF00ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xEDCBA9876543210FULL);
}

// RBIT .8B (Q=0): per-byte bit reversal, upper 64 bits zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, RbitVec8B) {
  static const uint32_t code[] = {RbitVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0102040880402010ULL, 0xAAAAAAAAAAAAAAAAULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x8040201001020408ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ULL);
}

// NEG .4S (Q=1): per-32-bit-lane negation via PSUBD.
TEST_F(Arm64HeavyOptimizerFrontendTest, NegVec4S) {
  static const uint32_t code[] = {NegVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0000000200000001ULL, 0xFFFFFFFB00000005ULL);  // [1,2,5,-5]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFEFFFFFFFFULL);   // [-1,-2]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000005FFFFFFFBULL);  // [-5,5]
}

// NEG .2D (Q=1): per-64-bit-lane negation via PSUBQ.
TEST_F(Arm64HeavyOptimizerFrontendTest, NegVec2D) {
  static const uint32_t code[] = {NegVec(0b11, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0000000000000003ULL, 0xFFFFFFFFFFFFFFFFULL);  // [3,-1]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFDULL);   // -3
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000001ULL);  // 1
}

// ABS .16B (Q=1): per-byte absolute value via PCMPGTB + PSUBB. INT8_MIN stays.
TEST_F(Arm64HeavyOptimizerFrontendTest, AbsVec16B) {
  static const uint32_t code[] = {AbsVec(0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x01FF02FE03FD04FCULL, 0x8000000000000080ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0101020203030404ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x8000000000000080ULL);  // INT8_MIN preserved
}

// ABS .8H (Q=1): per-16-bit-lane absolute value via PCMPGTW + PSUBW.
TEST_F(Arm64HeavyOptimizerFrontendTest, AbsVec8H) {
  static const uint32_t code[] = {AbsVec(0b01, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFF0002FFFE0001ULL, 0x8000000000008000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0001000200020001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x8000000000008000ULL);  // INT16_MIN preserved
}

// ABS .4S (Q=1): per-32-bit-lane absolute value via PCMPGTD + PSUBD.
TEST_F(Arm64HeavyOptimizerFrontendTest, AbsVec4S) {
  static const uint32_t code[] = {AbsVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFE00000005ULL, 0x80000000FFFFFFFFULL);  // [5,-2,-1,INT_MIN]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000200000005ULL);   // [5,2]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x8000000000000001ULL);  // [1,INT_MIN preserved]
}

// ---- AdvSIMD across-lanes reductions (heavy mirror): ADDV / SADDLV / UADDLV. ----

// ADDV b0, v1.16b: sum all 16 bytes (bytes 1..16 -> 136 = 0x88) into the low
// byte, every other byte of Vd zeroed. Exercises the PSADBW + high-qword fold.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddvVec16B) {
  static const uint32_t code[] = {AddvVec(0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0102030405060708ULL, 0x090A0B0C0D0E0F10ULL);  // bytes 1..16
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000088ULL);  // 136
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// ADDV s0, v1.4s: sum 4 dwords (1+2+3+4 = 10) into the low dword.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddvVec4S) {
  static const uint32_t code[] = {AddvVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0000000200000001ULL, 0x0000000400000003ULL);  // dwords 1,2,3,4
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000000AULL);  // 10
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UADDLV h0, v1.16b: zero-extend 16 bytes and sum. Low qword bytes all 1 (sum 8),
// high qword bytes all 2 (sum 16) -> 24; verifies both qword partial sums fold in.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlvVec16B) {
  static const uint32_t code[] = {AddlvVec(false, 0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0101010101010101ULL, 0x0202020202020202ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000018ULL);  // 24
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UADDLV s0, v1.4h (Q=0): zero-extend 4 low halfwords (1+2+3+4 = 10) to 32-bit.
// Upper 64 of Vn is don't-care and must be ignored.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlvVec4H) {
  static const uint32_t code[] = {AddlvVec(false, 0b01, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0004000300020001ULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000000AULL);  // 10
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UADDLV d0, v1.4s: zero-extend 4 dwords (each 0xFFFFFFFF) and sum to a 64-bit
// result 4*0xFFFFFFFF = 0x3FFFFFFFC. Exercises the .4S->D widen + qword fold.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlvVec4S) {
  static const uint32_t code[] = {AddlvVec(false, 0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000003FFFFFFFCULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SADDLV h0, v1.8b (Q=0): sign-extend 8 low bytes {-1,-2,-3,-4,0,0,0,0} and sum
// to -10 = 0xFFF6 (16-bit). Catches a zero-extend regression (would go positive).
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddlvVec8B) {
  static const uint32_t code[] = {AddlvVec(true, 0b00, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x00000000FCFDFEFFULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFF6ULL);  // -10 as u16
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SADDLV h0, v1.16b (Q=1): byte -1 at lane0, byte -2 at lane8, rest 0 -> -3 =
// 0xFFFD. Exercises the dual-half PMOVSXBW widen + PADDW path for .16B.
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddlvVec16B) {
  static const uint32_t code[] = {AddlvVec(true, 0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x00000000000000FFULL, 0x00000000000000FEULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFDULL);  // -3 as u16
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UADDLV s0, v1.8h (Q=1): zero-extend 8 halfwords (each 0xFFFF) and sum to
// 8*65535 = 524280 = 0x7FFF8. Exercises the dual-half PMOVZXWD widen for .8H.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlvVec8H) {
  static const uint32_t code[] = {AddlvVec(false, 0b01, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000007FFF8ULL);  // 524280
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// ---- AdvSIMD across-lanes max/min reductions (heavy mirror):
//      SMAXV / SMINV / UMAXV / UMINV. ----
//
// Byte tests share one source with a signed/unsigned-distinguishing lane set:
//   bytes = {0x7F, 0x80, 1, 2, 3, 4, 5, 6, 0x40..0x47}
//   signed:   max=+127 (0x7F)  min=-128 (0x80)
//   unsigned: max=128  (0x80)  min=1    (0x01)

TEST_F(Arm64HeavyOptimizerFrontendTest, SmaxvVec16B) {
  static const uint32_t code[] = {MaxminvVec(true, true, 0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x060504030201807FULL, 0x4746454443424140ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000007FULL);  // signed max +127
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, SminvVec16B) {
  static const uint32_t code[] = {MaxminvVec(false, true, 0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x060504030201807FULL, 0x4746454443424140ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000080ULL);  // signed min -128
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UmaxvVec16B) {
  static const uint32_t code[] = {MaxminvVec(true, false, 0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x060504030201807FULL, 0x4746454443424140ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000080ULL);  // unsigned max 128
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, UminvVec16B) {
  static const uint32_t code[] = {MaxminvVec(false, false, 0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x060504030201807FULL, 0x4746454443424140ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000001ULL);  // unsigned min 1
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UMAXV h0, v1.8h: unsigned max of {0x7FFF,0x8000,2,3,4,5,6,7} = 0x8000 (32768,
// unsigned) — a signed reduction would pick 0x7FFF (32767).
TEST_F(Arm64HeavyOptimizerFrontendTest, UmaxvVec8H) {
  static const uint32_t code[] = {MaxminvVec(true, false, 0b01, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0003000280007FFFULL, 0x0007000600050004ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000008000ULL);  // 32768
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SMAXV h0, v1.4h (Q=0): signed max of the low 4 halfwords
// {-32767, 5, 28672, -1} = 28672 (0x7000). Upper 64 of Vn is don't-care and
// must be neutralized by the low-qword replication.
TEST_F(Arm64HeavyOptimizerFrontendTest, SmaxvVec4H) {
  static const uint32_t code[] = {MaxminvVec(true, true, 0b01, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFFFF700000058001ULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000007000ULL);  // 28672
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SMAXV s0, v1.4s: signed max of {INT32_MIN, 5, INT32_MAX, -1} = INT32_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, SmaxvVec4S) {
  static const uint32_t code[] = {MaxminvVec(true, true, 0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0000000580000000ULL, 0xFFFFFFFF7FFFFFFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000007FFFFFFFULL);  // INT32_MAX
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UMINV s0, v1.4s: unsigned min of {16, 5, 3, 0xFFFFFFFF} = 3.
TEST_F(Arm64HeavyOptimizerFrontendTest, UminvVec4S) {
  static const uint32_t code[] = {MaxminvVec(false, false, 0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0000000500000010ULL, 0xFFFFFFFF00000003ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000003ULL);  // 3
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// ---- FMAXV/FMINV/FMAXNMV/FMINNMV FP32 across-lanes reduction (heavy). ----
// Lane packing (.4S): VLo64 = {lane1:lane0}, VUpperHi64 = {lane3:lane2}.
// Test vector lanes = {3.0, 1.0, -2.0, 5.0}:
//   3.0 = 0x40400000, 1.0 = 0x3F800000, -2.0 = 0xC0000000, 5.0 = 0x40A00000.

// FMAXV s0, v1.4s: NaN-propagating max of {3,1,-2,5} = 5.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxvVec4S) {
  static const uint32_t code[] = {FmaxvVec(/*is_max=*/true, /*is_nm=*/false, 0, 1)};
  SetV128(&state_, 1, 0x3F80000040400000ULL, 0x40A00000C0000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040A00000ULL);  // 5.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// FMINV s0, v1.4s: NaN-propagating min of {3,1,-2,5} = -2.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminvVec4S) {
  static const uint32_t code[] = {FmaxvVec(/*is_max=*/false, /*is_nm=*/false, 0, 1)};
  SetV128(&state_, 1, 0x3F80000040400000ULL, 0x40A00000C0000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000C0000000ULL);  // -2.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// FMAXNMV s0, v1.4s: NaN-suppressing max of {3,1,NaN,5} = 5.0 (NaN ignored).
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmvVec4S) {
  static const uint32_t code[] = {FmaxvVec(/*is_max=*/true, /*is_nm=*/true, 0, 1)};
  // lane2 = 0x7FC00000 (qNaN), lane3 = 5.0.
  SetV128(&state_, 1, 0x3F80000040400000ULL, 0x40A000007FC00000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000040A00000ULL);  // 5.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// FMINNMV s0, v1.4s: NaN-suppressing min of {3,1,NaN,-2} = -2.0 (NaN ignored).
TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmvVec4S) {
  static const uint32_t code[] = {FmaxvVec(/*is_max=*/false, /*is_nm=*/true, 0, 1)};
  // lane2 = 0x7FC00000 (qNaN), lane3 = -2.0.
  SetV128(&state_, 1, 0x3F80000040400000ULL, 0xC00000007FC00000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000C0000000ULL);  // -2.0
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// FMAXV s0, v1.4s with a NaN lane: NaN-propagating max of {3,1,NaN,5} -> NaN.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxvVec4SNanPropagates) {
  static const uint32_t code[] = {FmaxvVec(/*is_max=*/true, /*is_nm=*/false, 0, 1)};
  SetV128(&state_, 1, 0x3F80000040400000ULL, 0x40A000007FC00000ULL);  // lane2 = qNaN
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  const uint32_t res = static_cast<uint32_t>(VLo64(&state_, 0));
  EXPECT_EQ(res & 0x7F800000u, 0x7F800000u);  // exponent all ones
  EXPECT_NE(res & 0x007FFFFFu, 0u);            // nonzero mantissa -> NaN
  EXPECT_EQ(VLo64(&state_, 0) >> 32, 0x0ULL);  // upper 96 bits zeroed
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// ---- FMAXV/FMINV/FMAXNMV/FMINNMV FP16 across-lanes reduction (heavy, F16C). ----
// Lane packing (.8H): VLo64 = lanes 0..3, VUpperHi64 = lanes 4..7, low half first.
// Half constants: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200, 4.0=0x4400, 5.0=0x4500,
// 6.0=0x4600, 7.0=0x4700, 8.0=0x4800, -2.0=0xC000, -3.0=0xC200, -4.0=0xC400,
// -8.0=0xC800, +0.0=0x0000, -0.0=0x8000, qNaN=0x7E00.
// The .8H vector below is {1,2,3,4, 5,6,7,-2}: max = 7.0, min = -2.0.

// FMAXV h0, v1.8h: NaN-propagating max over 8 halves = 7.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxvVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/true, /*is_nm=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0xC000470046004500ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4700);  // 7.0h
}

// FMINV h0, v1.8h: NaN-propagating min over 8 halves = -2.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminvVecH8H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/false, /*is_nm=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0xC000470046004500ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0xC000);  // -2.0h
}

// FMAXNMV h0, v1.8h with lane5 = qNaN: the NaN lane is ignored, max = 7.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxnmvVecH8HNanSuppressed) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/true, /*is_nm=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0xC00047007E004500ULL);  // lane5 = qNaN
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4700);  // 7.0h
}

// FMINNMV h0, v1.8h with lane5 = qNaN: the NaN lane is ignored, min = -2.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminnmvVecH8HNanSuppressed) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/false, /*is_nm=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0xC00047007E004500ULL);  // lane5 = qNaN
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0xC000);  // -2.0h
}

// FMAXV h0, v1.8h with a NaN lane: NaN propagates to the reduction result. The
// surviving payload is whatever the POR idiom leaves, so only NaN-ness is checked
// (same contract as the FP32 FmaxvVec4SNanPropagates test).
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxvVecH8HNanPropagates) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/true, /*is_nm=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0xC00047007E004500ULL);  // lane5 = qNaN
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  const uint16_t res = VHalf0(&state_, 0);
  EXPECT_EQ(res & 0x7C00u, 0x7C00u);  // exponent all ones
  EXPECT_NE(res & 0x03FFu, 0u);       // nonzero mantissa -> NaN
  EXPECT_EQ(VWord0(&state_, 0) >> 16, 0u);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMINV h0, v1.8h with a NaN lane: NaN propagates through the min idiom too.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminvVecH8HNanPropagates) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/false, /*is_nm=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0xC00047007E004500ULL);  // lane5 = qNaN
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  const uint16_t res = VHalf0(&state_, 0);
  EXPECT_EQ(res & 0x7C00u, 0x7C00u);
  EXPECT_NE(res & 0x03FFu, 0u);
  EXPECT_EQ(VWord0(&state_, 0) >> 16, 0u);
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

// FMAXV h0, v1.4h: only lanes 0..3 {1,2,3,4} participate -> 4.0. Lanes 4..7 are all
// 5.0..8.0, so a leak from the upper half would show up as 8.0 instead.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxvVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/true, /*is_nm=*/false, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x4400420040003C00ULL, 0x4800470046004500ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x4400);  // 4.0h
}

// FMINV h0, v1.4h: lanes 0..3 are {1,-3,3,4} -> -3.0. Lanes 4..7 are all -8.0, so a
// leak from the upper half would show up as -8.0.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminvVecH4H) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/false, /*is_nm=*/false, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x44004200C2003C00ULL, 0xC800C800C800C800ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0xC200);  // -3.0h
}

// FMAXV h0, v1.4h over {-0,-0,-0,+0}: ARM's FPMax gives sign = AND of the input
// signs when every operand is zero, so the result is +0.0, not -0.0. Lanes 4..7 hold
// 4.0 to catch an upper-half leak.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxvVecH4HZeroSigns) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/true, /*is_nm=*/false, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0000800080008000ULL, 0x4400440044004400ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x0000);  // +0.0h
}

// FMINV h0, v1.4h over {+0,+0,+0,-0}: FPMin gives sign = OR of the input signs, so
// the result is -0.0. Lanes 4..7 hold -4.0 to catch an upper-half leak.
TEST_F(Arm64HeavyOptimizerFrontendTest, FminvVecH4HZeroSigns) {
  if (!host_platform::kHasF16C) GTEST_SKIP() << "no F16C";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/false, /*is_nm=*/false, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0xC400C400C400C400ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  bool ok = false;
  RunRegion(&state_, code, ToGuestAddr(code) + sizeof(code), &ok);
  ASSERT_TRUE(ok);
  EXPECT_HALF_CLEAN(state_, 0, 0x8000);  // -0.0h
}

// Without F16C the FP16 across-lanes form has no in-tier lowering and the region
// bails to lite (n == 0), which is correct-but-slow rather than wrong.
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxvVecHBailsWithoutF16C) {
  if (host_platform::kHasF16C) GTEST_SKIP() << "F16C present; FMAXV H lowers in-tier";
  static const uint32_t code[] = {FmaxvVecH(/*is_max=*/true, /*is_nm=*/false, /*q=*/true, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// ---- FCVTZS/FCVTZU V (vector FP32 -> int, round toward zero) heavy. ----

// FCVTZS v0.4s, v1.4s: truncate {2.7, -2.7, 100.9, -100.9} -> {2, -2, 100, -100}.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsVec4S) {
  static const uint32_t code[] = {FcvtzVec(/*is_unsigned=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.7f, -2.7f), Pack2xF32(100.9f, -100.9f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFE00000002ULL);   // {2, -2}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFF9C00000064ULL);  // {100, -100}
}

// FCVTZS v0.2s, v1.2s: only lanes 0,1 converted; upper 64 bits zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsVec2S) {
  static const uint32_t code[] = {FcvtzVec(/*is_unsigned=*/false, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.7f, -2.7f), Pack2xF32(100.9f, -100.9f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFE00000002ULL);   // {2, -2}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // .2S upper zeroed
}

// FCVTZS saturation: NaN -> 0, +overflow -> INT32_MAX, -overflow -> INT32_MIN.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzsVec4SSaturate) {
  static const uint32_t code[] = {FcvtzVec(/*is_unsigned=*/false, /*q=*/true, 0, 1)};
  // lane0 = qNaN, lane1 = +1e20 (> INT32_MAX), lane2 = -1e20 (< INT32_MIN),
  // lane3 = 5.0.
  SetV128(&state_, 1, Pack2xF32(std::numeric_limits<float>::quiet_NaN(), 1e20f),
          Pack2xF32(-1e20f, 5.0f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFF00000000ULL);   // {0, INT32_MAX}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000580000000ULL);  // {INT32_MIN, 5}
}

// FCVTZU v0.4s, v1.4s: {2.0, 100.0, 1.5*2^31, -1.0} -> {2, 100, 0xC0000000, 0}.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuVec4S) {
  static const uint32_t code[] = {FcvtzVec(/*is_unsigned=*/true, /*q=*/true, 0, 1)};
  // 1.5*2^31 = 3221225472 = 0xC0000000 (exactly representable in FP32).
  SetV128(&state_, 1, Pack2xF32(2.0f, 100.0f), Pack2xF32(3221225472.0f, -1.0f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000006400000002ULL);   // {2, 100}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000C0000000ULL);  // {0xC0000000, 0}
}

// FCVTZU saturation: negative -> 0, NaN -> 0, > 2^32 -> UINT32_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtzuVec4SSaturate) {
  static const uint32_t code[] = {FcvtzVec(/*is_unsigned=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(-1.0f, std::numeric_limits<float>::quiet_NaN()),
          Pack2xF32(1e20f, 5.0f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000000ULL);   // {0, 0}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000005FFFFFFFFULL);  // {UINT32_MAX, 5}
}

// ---- Vector SCVTF/UCVTF int->FP converts (heavy). ----

// SCVTF v0.4s, v1.4s: int32 {2, -2, 100, -100} -> FP32 {2.0, -2.0, 100.0, -100.0}.
TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfVec4S) {
  static const uint32_t code[] = {ScvtfVec(/*is_unsigned=*/false, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFE00000002ULL, 0xFFFFFF9C00000064ULL);  // {2,-2},{100,-100}
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC000000040000000ULL);       // {2.0, -2.0}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xC2C8000042C80000ULL);  // {100.0, -100.0}
}

// SCVTF v0.2s, v1.2s: only lanes 0,1 converted; upper 64 bits zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfVec2S) {
  static const uint32_t code[] = {ScvtfVec(/*is_unsigned=*/false, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFE00000002ULL, 0xBBBBBBBBBBBBBBBBULL);  // {2,-2}, junk
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xC000000040000000ULL);       // {2.0, -2.0}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // .2S upper zeroed
}

// UCVTF v0.4s, v1.4s: u32 {2, 2^31, 100, 0xFFFFFFFF} -> {2.0, 2^31, 100.0, 2^32}.
// Lanes with bit31 set (2^31, 0xFFFFFFFF) exercise the +2^32 addend path;
// 0xFFFFFFFF rounds to 2^32 (0x4F800000) under round-to-nearest-even.
TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfVec4S) {
  static const uint32_t code[] = {ScvtfVec(/*is_unsigned=*/true, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x8000000000000002ULL, 0xFFFFFFFF00000064ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4F00000040000000ULL);       // {2.0, 2^31}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4F80000042C80000ULL);  // {100.0, 2^32}
}

// UCVTF v0.2s, v1.2s: bit31-set lane + upper-zero check.
TEST_F(Arm64HeavyOptimizerFrontendTest, UcvtfVec2S) {
  static const uint32_t code[] = {ScvtfVec(/*is_unsigned=*/true, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x8000000000000002ULL, 0xBBBBBBBBBBBBBBBBULL);  // {2, 2^31}, junk
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x4F00000040000000ULL);       // {2.0, 2^31}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // .2S upper zeroed
}

// ---- Vector round-mode FP->int converts FCVTN*/FCVTP*/FCVTM*/FCVTA* (heavy). ----
// Same {2.7, -2.7, 100.9, -100.9} inputs distinguish the three signed rounding
// modes; the saturation fix-up is byte-identical to FCVTZS/FCVTZU (already
// saturate-tested above).

// FCVTNS (round-to-nearest ties-even): {2.7,-2.7,100.9,-100.9} -> {3,-3,101,-101}.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtnsVec4S) {
  static const uint32_t code[] = {FcvtnsVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.7f, -2.7f), Pack2xF32(100.9f, -100.9f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFD00000003ULL);       // {3, -3}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFF9B00000065ULL);  // {101, -101}
}

// FCVTPS (round toward +inf / ceil): -> {3,-2,101,-100}.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtpsVec4S) {
  static const uint32_t code[] = {FcvtpsVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.7f, -2.7f), Pack2xF32(100.9f, -100.9f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFE00000003ULL);       // {3, -2}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFF9C00000065ULL);  // {101, -100}
}

// FCVTMS (round toward -inf / floor): -> {2,-3,100,-101}. .2s zeroes upper 64.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtmsVec2S) {
  static const uint32_t code[] = {FcvtmsVec(/*q=*/false, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.7f, -2.7f), Pack2xF32(100.9f, -100.9f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFD00000002ULL);       // {2, -3}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // .2S upper zeroed
}

// FCVTNU: {2.7,100.9,5.2,200.8} -> {3,101,5,201} (all positive).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtnuVec4S) {
  static const uint32_t code[] = {FcvtnuVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.7f, 100.9f), Pack2xF32(5.2f, 200.8f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000006500000003ULL);       // {3, 101}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000C900000005ULL);  // {5, 201}
}

// FCVTPU (ceil): -> {3,101,6,201}.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtpuVec4S) {
  static const uint32_t code[] = {FcvtpuVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.7f, 100.9f), Pack2xF32(5.2f, 200.8f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000006500000003ULL);       // {3, 101}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000C900000006ULL);  // {6, 201}
}

// FCVTMU (floor): -> {2,100,5,200}. Also verify negative -> 0 clamp.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtmuVec4S) {
  static const uint32_t code[] = {FcvtmuVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.7f, 100.9f), Pack2xF32(-5.2f, 200.8f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000006400000002ULL);       // {2, 100}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000C800000000ULL);  // {floor(-5.2)=-6 -> 0, 200}
}

// FCVTAS (round-to-nearest ties-away): ties {2.5,-2.5,3.5,-3.5} -> {3,-3,4,-4}
// (ties-even would give lane0=2; this pins the ties-away rounding).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtasVec4S) {
  static const uint32_t code[] = {FcvtasVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.5f, -2.5f), Pack2xF32(3.5f, -3.5f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFD00000003ULL);       // {3, -3}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFC00000004ULL);  // {4, -4}
}

// FCVTAU (ties-away, unsigned): {2.5,3.5,0.5,200.5} -> {3,4,1,201}.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcvtauVec4S) {
  static const uint32_t code[] = {FcvtauVec(/*q=*/true, 0, 1)};
  SetV128(&state_, 1, Pack2xF32(2.5f, 3.5f), Pack2xF32(0.5f, 200.5f));
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000400000003ULL);       // {3, 4}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000C900000001ULL);  // {1, 201}
}

// ---- SADDLP/UADDLP/SADALP/UADALP pairwise long add / accumulate (heavy). ----

// UADDLP v0.4h, v1.8b (Q=0): unsigned pair-sum of bytes
// {0xFF,0x01,0x80,0x80,0x10,0x20,0x00,0xFF} -> {0x100,0x100,0x30,0xFF}.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlpVec4H) {
  static const uint32_t code[] = {AddlpVec(false, false, 0b00, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFF002010808001FFULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00FF003001000100ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SADDLP v0.4h, v1.8b (Q=0): SAME input, signed pair-sum
// {(-1)+1, (-128)+(-128), 16+32, 0+(-1)} = {0, -256, 48, -1}.
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddlpVec4H) {
  static const uint32_t code[] = {AddlpVec(true, false, 0b00, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFF002010808001FFULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFF0030FF000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UADDLP v0.8h, v1.16b (Q=1): full-128 unsigned pair-sum. Low half as above;
// high bytes {1,2,3,4,5,6,7,8} -> {3,7,0xB,0xF}.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlpVec8H) {
  static const uint32_t code[] = {AddlpVec(false, false, 0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFF002010808001FFULL, 0x0807060504030201ULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00FF003001000100ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000F000B00070003ULL);
}

// UADDLP v0.2s, v1.4h (Q=0): unsigned pair-sum of halfwords
// {0xFFFF,0x0001,0x8000,0x8000} -> {0x10000, 0x10000}.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlpVec2S) {
  static const uint32_t code[] = {AddlpVec(false, false, 0b01, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x800080000001FFFFULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0001000000010000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SADDLP v0.2s, v1.4h (Q=0): SAME input, signed pair-sum (PMADDWD path)
// {(-1)+1, (-32768)+(-32768)} = {0, -65536}.
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddlpVec2S) {
  static const uint32_t code[] = {AddlpVec(true, false, 0b01, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x800080000001FFFFULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFF000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UADDLP v0.2d, v1.4s (Q=1): unsigned pair-sum of words
// {0xFFFFFFFF,0x1,0x80000000,0x80000000} -> {0x100000000, 0x100000000}.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlpVec2D) {
  static const uint32_t code[] = {AddlpVec(false, false, 0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x00000001FFFFFFFFULL, 0x8000000080000000ULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000100000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000100000000ULL);
}

// SADDLP v0.2d, v1.4s (Q=1): SAME input, signed pair-sum (PMOVSXDQ path)
// {(-1)+1, (-2^31)+(-2^31)} = {0, -2^32}.
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddlpVec2D) {
  static const uint32_t code[] = {AddlpVec(true, false, 0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x00000001FFFFFFFFULL, 0x8000000080000000ULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFF00000000ULL);
}

// SADDLP v0.1d, v1.2s (Q=0): signed pair-sum of 2 words, high garbage lane
// must be masked. {(-2^31)+(-1)} = -2147483649 = 0xFFFFFFFF7FFFFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddlpVec1D) {
  static const uint32_t code[] = {AddlpVec(true, false, 0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFF80000000ULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF7FFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SADALP v0.2s, v1.4h (Q=0): signed pair-sum {2+3, (-1)+(-2)} = {5, -3},
// accumulated into Vd {0x10, 0x100} -> {0x15, 0xFD}.
TEST_F(Arm64HeavyOptimizerFrontendTest, SadalpVec2S) {
  static const uint32_t code[] = {AddlpVec(true, true, 0b01, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFFFEFFFF00030002ULL, 0xAAAAAAAAAAAAAAAAULL);  // hi = poison
  SetV128(&state_, 0, 0x0000010000000010ULL, 0xDDDDDDDDDDDDDDDDULL);  // acc + poison hi
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000FD00000015ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UADALP v0.2d, v1.4s (Q=1): unsigned pair-sum {2+3, 4+5} = {5, 9},
// accumulated into Vd {0x100, 0x1000} -> {0x105, 0x1009}.
TEST_F(Arm64HeavyOptimizerFrontendTest, UadalpVec2D) {
  static const uint32_t code[] = {AddlpVec(false, true, 0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0000000300000002ULL, 0x0000000500000004ULL);
  SetV128(&state_, 0, 0x0000000000000100ULL, 0x0000000000001000ULL);  // acc
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000105ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000001009ULL);
}

// ---- AdvSIMD two-reg-misc widening/narrowing (heavy mirror). ----

// XTN v0.8b, v1.8h (size=00, Q=0): truncate 8 halfwords to their low bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, XtnVec8B) {
  static const uint32_t code[] = {XtnVec(0b00, /*q=*/false, 0, 1)};
  // halfwords h0..h7 = 0102 ABCD 1234 00FF | 5566 7788 99AA BBCC.
  SetV128(&state_, 1, 0x00FF1234ABCD0102ULL, 0xBBCC99AA77885566ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xCCAA8866FF34CD02ULL);   // low bytes of each hword
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zeroed
}

// XTN2 v0.16b, v1.8h (size=00, Q=1): narrowed bytes into Vd.high, Vd.low kept.
TEST_F(Arm64HeavyOptimizerFrontendTest, Xtn2Vec16B) {
  static const uint32_t code[] = {XtnVec(0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x00FF1234ABCD0102ULL, 0xBBCC99AA77885566ULL);
  SetV128(&state_, 0, 0x1122334455667788ULL, 0xBBBBBBBBBBBBBBBBULL);  // low preserved
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1122334455667788ULL);   // Vd.low preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xCCAA8866FF34CD02ULL);  // narrowed into high
}

// XTN v0.4h, v1.4s (size=01, Q=0): truncate 4 dwords to their low halfwords.
TEST_F(Arm64HeavyOptimizerFrontendTest, XtnVec4H) {
  static const uint32_t code[] = {XtnVec(0b01, /*q=*/false, 0, 1)};
  // dwords 1111AAAA 2222BBBB | 3333CCCC 4444DDDD.
  SetV128(&state_, 1, 0x2222BBBB1111AAAAULL, 0x4444DDDD3333CCCCULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xDDDDCCCCBBBBAAAAULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// XTN v0.2s, v1.2d (size=10, Q=0): PSHUFD gathers the low dword of each qword.
TEST_F(Arm64HeavyOptimizerFrontendTest, XtnVec2S) {
  static const uint32_t code[] = {XtnVec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x1234567890ABCDEFULL, 0xFEDCBA9876543210ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7654321090ABCDEFULL);   // [d0.lo32, d1.lo32]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SHLL v0.8h, v1.8b, #8 (size=00, Q=0): zero-extend low 8 bytes to words << 8.
TEST_F(Arm64HeavyOptimizerFrontendTest, ShllVec8H) {
  static const uint32_t code[] = {ShllVec(0b00, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0807060504030201ULL, 0x1817161514131211ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0400030002000100ULL);   // words 0100 0200 0300 0400
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0800070006000500ULL);
}

// SHLL2 v0.8h, v1.16b, #8 (size=00, Q=1): widen Vn's HIGH 8 bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, Shll2Vec8H) {
  static const uint32_t code[] = {ShllVec(0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0807060504030201ULL, 0x100F0E0D0C0B0A09ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0C000B000A000900ULL);   // words 0900 0A00 0B00 0C00
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x10000F000E000D00ULL);
}

// SHLL v0.4s, v1.4h, #16 (size=01, Q=0): zero-extend low 4 hwords to dwords << 16.
TEST_F(Arm64HeavyOptimizerFrontendTest, ShllVec4S) {
  static const uint32_t code[] = {ShllVec(0b01, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0004000300020001ULL, 0x0008000700060005ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0002000000010000ULL);   // dwords 00010000 00020000
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0004000000030000ULL);
}

// SHLL v0.2d, v1.2s, #32 (size=10, Q=0): zero-extend low 2 dwords to qwords << 32.
TEST_F(Arm64HeavyOptimizerFrontendTest, ShllVec2D) {
  static const uint32_t code[] = {ShllVec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0000000A00000005ULL, 0xDEADBEEFCAFEBABEULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000500000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000A00000000ULL);
}

// UXTL v0.8h, v1.8b, #0 (USHLL #0, immh=0001, immb=0): zero-extend low 8 bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, UxtlVec8H) {
  static const uint32_t code[] = {UshllVec(/*q=*/false, /*immh=*/0b0001, /*immb=*/0, 0, 1)};
  SetV128(&state_, 1, 0x0807060504030201ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0004000300020001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0008000700060005ULL);
}

// SXTL v0.4s, v1.4h, #0 (SSHLL #0, immh=0010, immb=0): sign-extend low 4 hwords.
TEST_F(Arm64HeavyOptimizerFrontendTest, SxtlVec4S) {
  static const uint32_t code[] = {SshllVec(/*q=*/false, /*immh=*/0b0010, /*immb=*/0, 0, 1)};
  // hwords lane0..3 = 0001 FFFF 8000 7FFF.
  SetV128(&state_, 1, 0x7FFF8000FFFF0001ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF00000001ULL);   // words 00000001 FFFFFFFF
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00007FFFFFFF8000ULL);  // words FFFF8000 00007FFF
}

// USHLL2 v0.4s, v1.8h, #3 (q=1, immh=0010, immb=3): widen UPPER 4 hwords << 3.
TEST_F(Arm64HeavyOptimizerFrontendTest, Ushll2Vec4S) {
  static const uint32_t code[] = {UshllVec(/*q=*/true, /*immh=*/0b0010, /*immb=*/3, 0, 1)};
  // upper hwords lane4..7 = 0001 0002 0003 0004.
  SetV128(&state_, 1, 0xBBBBBBBBBBBBBBBBULL, 0x0004000300020001ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000001000000008ULL);   // 0x0001<<3, 0x0002<<3
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000002000000018ULL);  // 0x0003<<3, 0x0004<<3
}

// SHL v0.8h, v1.8h, #4 (q=1, immh=0010, immb=4): left-shift 8 hwords by 4.
TEST_F(Arm64HeavyOptimizerFrontendTest, ShlVec8H) {
  static const uint32_t code[] = {ShlVec(/*q=*/true, /*immh=*/0b0010, /*immb=*/4, 0, 1)};
  SetV128(&state_, 1, 0x1000010000100001ULL, 0x800000FF00030002ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000100001000010ULL);   // 1<<4,10<<4,100<<4,1000<<4(=0)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000FF000300020ULL);  // 2<<4,3<<4,FF<<4,8000<<4(=0)
}

// USHR v0.4s, v1.4s, #8 (q=1, immh=0111, immb=0): logical right-shift 4 words by 8.
TEST_F(Arm64HeavyOptimizerFrontendTest, UshrVec4S) {
  static const uint32_t code[] = {UshrVec(/*q=*/true, /*immh=*/0b0111, /*immb=*/0, 0, 1)};
  SetV128(&state_, 1, 0xFF00000000000100ULL, 0x800000000000FF00ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00FF000000000001ULL);   // 0x100>>8, 0xFF000000>>8
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00800000000000FFULL);  // 0xFF00>>8, 0x80000000>>8
}

// SSHR v0.4h, v1.4h, #2 (q=0, immh=0011, immb=6): arith right-shift 4 hwords, upper zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, SshrVec4H) {
  static const uint32_t code[] = {SshrVec(/*q=*/false, /*immh=*/0b0011, /*immb=*/6, 0, 1)};
  // hwords lane0..3 = 0004 FFFC(-4) 8000(-32768) 000C(12).
  SetV128(&state_, 1, 0x000C8000FFFC0004ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0003E000FFFF0001ULL);   // 1, -1, -8192(0xE000), 3
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// SSHR v0.16b, v1.16b, #3 (q=1, immh=0001, immb=5): arith right-shift 16 bytes.
// Exercises the byte-lane path (PMOVSXBW + PSRAW + PACKSSWB) across both
// 64-bit halves — previously bailed to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, SshrVec16B) {
  static const uint32_t code[] = {SshrVec(/*q=*/true, /*immh=*/0b0001, /*immb=*/5, 0, 1)};
  // lo bytes: 80(-128) 7F(127) FF(-1) 08(8) 40(64) C0(-64) 01(1) FE(-2).
  // hi bytes: 10 20 30 40 50 60 70 88(-120).
  SetV128(&state_, 1, 0xFE01C04008FF7F80ULL, 0x8870605040302010ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lo: F0 0F FF 01 08 F8 00 FF ; hi: 02 04 06 08 0A 0C 0E F1.
  EXPECT_EQ(VLo64(&state_, 0), 0xFF00F80801FF0FF0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xF10E0C0A08060402ULL);
}

// USHR v0.16b, v1.16b, #2 (q=1, immh=0001, immb=6): logical right-shift 16
// bytes (PMOVZXBW + PSRLW + PACKUSWB across both halves).
TEST_F(Arm64HeavyOptimizerFrontendTest, UshrVec16B) {
  static const uint32_t code[] = {UshrVec(/*q=*/true, /*immh=*/0b0001, /*immb=*/6, 0, 1)};
  // lo bytes: 80(128) FF(255) 04 03 40(64) C0(192) 01 FC(252).
  // hi bytes: 08 10 20 40 80(128) F0(240) 05 07.
  SetV128(&state_, 1, 0xFC01C0400304FF80ULL, 0x0705F08040201008ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lo: 20 3F 01 00 10 30 00 3F ; hi: 02 04 08 10 20 3C 01 01.
  EXPECT_EQ(VLo64(&state_, 0), 0x3F00301000013F20ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x01013C2010080402ULL);
}

// SSHR v0.8b, v1.8b, #1 (q=0, immh=0001, immb=7): .8B arith right-shift, upper
// 64 bits of Vd must be zeroed (D-form).
TEST_F(Arm64HeavyOptimizerFrontendTest, SshrVec8B) {
  static const uint32_t code[] = {SshrVec(/*q=*/false, /*immh=*/0b0001, /*immb=*/7, 0, 1)};
  // lo bytes: 80(-128) 7F(127) FE(-2) 02 FF(-1) 10(16) 10(16) 81(-127).
  SetV128(&state_, 1, 0x811010FF02FE7F80ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // C0 3F FF 01 FF 08 08 C0.
  EXPECT_EQ(VLo64(&state_, 0), 0xC00808FF01FF3FC0ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// USHR v0.8b, v1.8b, #4 (q=0, immh=0001, immb=4): .8B logical right-shift,
// upper 64 bits of Vd zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, UshrVec8B) {
  static const uint32_t code[] = {UshrVec(/*q=*/false, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  // lo bytes: F0(240) FF(255) 80(128) 10(16) 01 A0(160) 0F 88(136).
  SetV128(&state_, 1, 0x880FA0011080FFF0ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // 0F 0F 08 01 00 0A 00 08.
  EXPECT_EQ(VLo64(&state_, 0), 0x08000A0001080F0FULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// SSRA v0.4s, v1.4s, #4 (q=1, immh=0111, immb=4): Vd += SSHR(Vn, 4) per word.
TEST_F(Arm64HeavyOptimizerFrontendTest, SsraVec4S) {
  static const uint32_t code[] = {SsraVec(/*q=*/true, /*immh=*/0b0111, /*immb=*/4, 0, 1)};
  // Vn words 0x00000080(128) 0xFFFFFF80(-128) 0x80000000(INT_MIN) 0x0000000C(12).
  SetV128(&state_, 1, 0xFFFFFF8000000080ULL, 0x0000000C80000000ULL);
  // Vd words 1, 2, 3, 4.
  SetV128(&state_, 0, 0x0000000200000001ULL, 0x0000000400000003ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // SSHR by 4: 8, -8, 0xF8000000, 0. Vd += that -> 9, -6, 0xF8000003, 4.
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFA00000009ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000004F8000003ULL);
}

// SSRA v0.4h, v1.4h, #2 (q=0, immh=0011, immb=6): D-form accumulate, upper zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, SsraVec4H) {
  static const uint32_t code[] = {SsraVec(/*q=*/false, /*immh=*/0b0011, /*immb=*/6, 0, 1)};
  // Vn hwords 0x0004(4) 0xFFFC(-4) 0x8000(-32768) 0x000C(12).
  SetV128(&state_, 1, 0x000C8000FFFC0004ULL, 0xBBBBBBBBBBBBBBBBULL);
  // Vd hwords 1, 2, 3, 4.
  SetV128(&state_, 0, 0x0004000300020001ULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // SSHR by 2: 1, -1, -8192(0xE000), 3. Vd += that -> 2, 1, 0xE003, 7.
  EXPECT_EQ(VLo64(&state_, 0), 0x0007E00300010002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// USRA v0.4s, v1.4s, #8 (q=1, immh=0111, immb=0): Vd += USHR(Vn, 8) per word.
TEST_F(Arm64HeavyOptimizerFrontendTest, UsraVec4S) {
  static const uint32_t code[] = {UsraVec(/*q=*/true, /*immh=*/0b0111, /*immb=*/0, 0, 1)};
  // Vn words 0x00000100 0xFF000000 0x0000FF00 0x80000000.
  SetV128(&state_, 1, 0xFF00000000000100ULL, 0x800000000000FF00ULL);
  // Vd words 0x10, 0x20, 0x30, 0x40.
  SetV128(&state_, 0, 0x0000002000000010ULL, 0x0000004000000030ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // USHR by 8: 0x01, 0x00FF0000, 0xFF, 0x00800000. Vd += that.
  EXPECT_EQ(VLo64(&state_, 0), 0x00FF002000000011ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x008000400000012FULL);
}

// USRA v0.2d, v1.2d, #4 (q=1, immh=1111, immb=4): Vd += USHR(Vn, 4) per dword (PADDQ).
TEST_F(Arm64HeavyOptimizerFrontendTest, UsraVec2D) {
  static const uint32_t code[] = {UsraVec(/*q=*/true, /*immh=*/0b1111, /*immb=*/4, 0, 1)};
  SetV128(&state_, 1, 0x0000000000000100ULL, 0xFF00000000000000ULL);
  SetV128(&state_, 0, 0x0000000000000001ULL, 0x0000000000000002ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // USHR by 4: 0x10, 0x0FF0000000000000. Vd += that -> 0x11, 0x0FF0000000000002.
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000011ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0FF0000000000002ULL);
}

// SSRA .2D needs PSRAQ (AVX-512F-VL) — must bail to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, SsraVec2DBails) {
  static const uint32_t code[] = {SsraVec(/*q=*/true, /*immh=*/0b1000, /*immb=*/0, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// USRA .16B byte lane: now lowered in-tier via PMOVZXBW/PSRLW/PACKUSWB + PADDB
// (region translates rather than bailing).
TEST_F(Arm64HeavyOptimizerFrontendTest, UsraVec16BLowered) {
  static const uint32_t code[] = {UsraVec(/*q=*/true, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 1u);
}

// SLI v0.4s, v1.4s, #8 (q=1, immh=0101, immb=0): Vd<i> = (Vn<i>><<8) with Vd's
// low 8 bits preserved per word.
TEST_F(Arm64HeavyOptimizerFrontendTest, SliVec4S) {
  static const uint32_t code[] = {SliVec(/*q=*/true, /*immh=*/0b0101, /*immb=*/0, 0, 1)};
  // Vn words 0x00000001 0x000000FF 0x12345678 0x80000001.
  SetV128(&state_, 1, 0x000000FF00000001ULL, 0x8000000112345678ULL);
  // Vd words 0xAABBCCDD 0x11223344 0x55667788 0x99AABBCC.
  SetV128(&state_, 0, 0x11223344AABBCCDDULL, 0x99AABBCC55667788ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (Vn<<8)|(Vd&0xFF): 0x000001DD 0x0000FF44 0x34567888 0x000001CC.
  EXPECT_EQ(VLo64(&state_, 0), 0x0000FF44000001DDULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000001CC34567888ULL);
}

// SLI v0.2s, v1.2s, #0 (q=0, immh=0100, immb=0): shift 0 -> whole Vn inserted,
// D-form upper zeroed. Exercises the inv==esize boundary (clears all of Vd).
TEST_F(Arm64HeavyOptimizerFrontendTest, SliVec2SZero) {
  static const uint32_t code[] = {SliVec(/*q=*/false, /*immh=*/0b0100, /*immb=*/0, 0, 1)};
  SetV128(&state_, 1, 0xDEADBEEF12345678ULL, 0xAAAAAAAAAAAAAAAAULL);
  SetV128(&state_, 0, 0xBBBBBBBBAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xDEADBEEF12345678ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// SRI v0.4h, v1.4h, #4 (q=0, immh=0011, immb=4): Vd<i> = USHR(Vn<i>,4) with
// Vd's high 4 bits preserved per hword; D-form upper zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, SriVec4H) {
  static const uint32_t code[] = {SriVec(/*q=*/false, /*immh=*/0b0011, /*immb=*/4, 0, 1)};
  // Vn hwords 0x1234 0xFF00 0x000F 0x8421.
  SetV128(&state_, 1, 0x8421000FFF001234ULL, 0xBBBBBBBBBBBBBBBBULL);
  // Vd hwords 0xAAAA 0xBBBB 0xCCCC 0xDDDD.
  SetV128(&state_, 0, 0xDDDDCCCCBBBBAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (Vd&0xF000)|(Vn>>4): 0xA123 0xBFF0 0xC000 0xD842.
  EXPECT_EQ(VLo64(&state_, 0), 0xD842C000BFF0A123ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// SRI v0.2d, v1.2d, #4 (q=1, immh=1111, immb=4): Vd<i> = USHR(Vn<i>,4) with
// Vd's high 4 bits preserved per dword (PSRLQ/PSLLQ/POR).
TEST_F(Arm64HeavyOptimizerFrontendTest, SriVec2D) {
  static const uint32_t code[] = {SriVec(/*q=*/true, /*immh=*/0b1111, /*immb=*/4, 0, 1)};
  SetV128(&state_, 1, 0x0000000000000100ULL, 0xFF00000000000000ULL);
  SetV128(&state_, 0, 0x1111111111111111ULL, 0xF222222222222222ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (Vd&0xF00..0)|(Vn>>4): 0x1000000000000010, 0xFFF0000000000000.
  EXPECT_EQ(VLo64(&state_, 0), 0x1000000000000010ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFF0000000000000ULL);
}

// SRI v0.4s, v1.4s, #32 (q=1, immh=0100, immb=0): shift==esize -> USHR gives 0,
// all of Vd preserved. Exercises the inv==0 boundary (Vd unchanged).
TEST_F(Arm64HeavyOptimizerFrontendTest, SriVec4SFull) {
  static const uint32_t code[] = {SriVec(/*q=*/true, /*immh=*/0b0100, /*immb=*/0, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  SetV128(&state_, 0, 0x2222222211111111ULL, 0x4444444433333333ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x2222222211111111ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x4444444433333333ULL);
}

// SLI .16B byte lane has no x86 packed byte shift — must bail to lite.
// SLI v0.16b, v1.16b, #4 now lowers in the heavy tier (byte form). Per byte:
// result = ((Vn << 4) & 0xF0) | (Vd & 0x0F). Vn=0x05 -> 0x50; Vd=0xFF low4=0x0F
// -> 0x5F in every one of the 16 byte lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, SliVec16BLowered) {
  static const uint32_t code[] = {SliVec(/*q=*/true, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  SetV128(&state_, 1, 0x0505050505050505ULL, 0x0505050505050505ULL);
  SetV128(&state_, 0, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x5F5F5F5F5F5F5F5FULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x5F5F5F5F5F5F5F5FULL);
}

// URSHR v0.4s, v1.4s, #4 (q=1, immh=0111, immb=4): Vd<i> = (Vn<i>+8)>>4 unsigned.
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshrVec4S) {
  static const uint32_t code[] = {UrshrVec(/*q=*/true, /*immh=*/0b0111, /*immb=*/4, 0, 1)};
  // Vn words 0x00000080(128) 0x000000FF(255) 0xFFFFFFF8 0x0000000C(12).
  SetV128(&state_, 1, 0x000000FF00000080ULL, 0x0000000CFFFFFFF8ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (Vn+8)>>4: 8, 16, 0x10000000, 1.
  EXPECT_EQ(VLo64(&state_, 0), 0x0000001000000008ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000110000000ULL);
}

// SRSHR v0.4h, v1.4h, #2 (q=0, immh=0011, immb=6): signed rounding, D-form upper zero.
TEST_F(Arm64HeavyOptimizerFrontendTest, SrshrVec4H) {
  static const uint32_t code[] = {SrshrVec(/*q=*/false, /*immh=*/0b0011, /*immb=*/6, 0, 1)};
  // Vn hwords 0x0006(6) 0xFFFA(-6) 0x8000(-32768) 0x000E(14).
  SetV128(&state_, 1, 0x000E8000FFFA0006ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // floor((Vn+2)/4): 2, -1(0xFFFF), -8192(0xE000), 4.
  EXPECT_EQ(VLo64(&state_, 0), 0x0004E000FFFF0002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// URSRA v0.4s, v1.4s, #8 (q=1, immh=0111, immb=0): Vd += (Vn+128)>>8 unsigned.
TEST_F(Arm64HeavyOptimizerFrontendTest, UrsraVec4S) {
  static const uint32_t code[] = {UrsraVec(/*q=*/true, /*immh=*/0b0111, /*immb=*/0, 0, 1)};
  // Vn words 0x00000180 0xFF000080 0x0000FF00 0x80000040.
  SetV128(&state_, 1, 0xFF00008000000180ULL, 0x800000400000FF00ULL);
  // Vd words 0x10, 0x20, 0x30, 0x40.
  SetV128(&state_, 0, 0x0000002000000010ULL, 0x0000004000000030ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // URSHR by 8: 2, 0x00FF0001, 0xFF, 0x00800000. Vd += that.
  EXPECT_EQ(VLo64(&state_, 0), 0x00FF002100000012ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x008000400000012FULL);
}

// URSHR v0.2d, v1.2d, #4 (q=1, immh=1111, immb=4): unsigned rounding .2D (PSRLQ path).
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshrVec2D) {
  static const uint32_t code[] = {UrshrVec(/*q=*/true, /*immh=*/0b1111, /*immb=*/4, 0, 1)};
  SetV128(&state_, 1, 0x0000000000000188ULL, 0xFF00000000000008ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (Vn+8)>>4: (0x188+8)>>4=0x19, (0xFF00..08+8)>>4=0x0FF0000000000001.
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000019ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0FF0000000000001ULL);
}

// SRSRA v0.4h, v1.4h, #2 (q=0, immh=0011, immb=6): signed rounding accumulate, D-form.
TEST_F(Arm64HeavyOptimizerFrontendTest, SrsraVec4H) {
  static const uint32_t code[] = {SrsraVec(/*q=*/false, /*immh=*/0b0011, /*immb=*/6, 0, 1)};
  // Vn hwords 0x0006(6) 0xFFFA(-6) 0x8000(-32768) 0x000E(14).
  SetV128(&state_, 1, 0x000E8000FFFA0006ULL, 0xBBBBBBBBBBBBBBBBULL);
  // Vd hwords 1, 2, 3, 4.
  SetV128(&state_, 0, 0x0004000300020001ULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // SRSHR by 2: 2, -1, -8192, 4. Vd += that -> 3, 1, 0xE003, 8.
  EXPECT_EQ(VLo64(&state_, 0), 0x0008E00300010003ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// SRSHR .2D needs PSRAQ (AVX-512F-VL) — must bail to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, SrshrVec2DBails) {
  static const uint32_t code[] = {SrshrVec(/*q=*/true, /*immh=*/0b1000, /*immb=*/0, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// URSHR .16B byte lane: now lowered in-tier via the PMOVZXBW round-bracket
// recipe (region translates rather than bailing).
TEST_F(Arm64HeavyOptimizerFrontendTest, UrshrVec16BLowered) {
  static const uint32_t code[] = {UrshrVec(/*q=*/true, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 1u);
}

// UQSHL v0.4s, v1.4s, #28 (q=1, immh=0111, immb=4): unsigned saturating left.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqshlVec4S) {
  static const uint32_t code[] = {UqshlVec(/*q=*/true, /*immh=*/0b0111, /*immb=*/4, 0, 1)};
  // Vn words 1, 16, 15, 0.  1<<28 fits; 16<<28 overflows -> UINT32_MAX;
  // 15<<28 fits; 0 stays 0.
  SetV128(&state_, 1, 0x0000001000000001ULL, 0x000000000000000FULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // words: 0x10000000, 0xFFFFFFFF(sat), 0xF0000000, 0.
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF10000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000F0000000ULL);
}

// SQSHL v0.4h, v1.4h, #12 (q=0, immh=0011, immb=4): signed saturating left, D-form.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqshlVec4H) {
  static const uint32_t code[] = {SqshlVec(/*q=*/false, /*immh=*/0b0011, /*immb=*/4, 0, 1)};
  // Vn hwords 1, 7, 8, -1.  1<<12,7<<12 fit; 8<<12 overflows INT16_MAX(+) ->
  // 0x7FFF; -1<<12=-4096 fits (0xF000).
  SetV128(&state_, 1, 0xFFFF000800070001ULL, 0xBBBBBBBBBBBBBBBBULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xCCCCCCCCCCCCCCCCULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // hwords: 0x1000, 0x7000, 0x7FFF(sat), 0xF000.
  EXPECT_EQ(VLo64(&state_, 0), 0xF0007FFF70001000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // D-form upper zero
}

// SQSHLU v0.4s, v1.4s, #28 (q=1, immh=0111, immb=4): signed src, unsigned sat.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqshluVec4S) {
  static const uint32_t code[] = {SqshluVec(/*q=*/true, /*immh=*/0b0111, /*immb=*/4, 0, 1)};
  // Vn words 1, 16, -16(0xFFFFFFF0), 15.  1<<28 fits; 16<<28 overflows ->
  // UINT32_MAX; negative -> 0; 15<<28 fits.
  SetV128(&state_, 1, 0x0000001000000001ULL, 0x0000000FFFFFFFF0ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // words: 0x10000000, 0xFFFFFFFF(sat), 0(neg->0), 0xF0000000.
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF10000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xF000000000000000ULL);
}

// UQSHL v0.2d, v1.2d, #60 (q=1, immh=1111, immb=4): unsigned saturating .2D (PSLLQ/PSRLQ/PCMPEQQ).
TEST_F(Arm64HeavyOptimizerFrontendTest, UqshlVec2D) {
  static const uint32_t code[] = {UqshlVec(/*q=*/true, /*immh=*/0b1111, /*immb=*/4, 0, 1)};
  // Vn dwords 1, 16.  1<<60 fits; 16<<60 overflows -> UINT64_MAX.
  SetV128(&state_, 1, 0x0000000000000001ULL, 0x0000000000000010ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x1000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);  // sat
}

// SQSHL .2D needs PSRAQ (AVX-512F-VL) for the signed recovery — must bail to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqshlVec2DBails) {
  static const uint32_t code[] = {SqshlVec(/*q=*/true, /*immh=*/0b1000, /*immb=*/0, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// UQSHL .16B byte lane has no x86 packed byte shift — must bail to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqshlVec16BBails) {
  static const uint32_t code[] = {UqshlVec(/*q=*/true, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SHRN v0.8b, v1.8h, #4 (immh=0001, immb=4, Q=0): shift-right narrow 16->8.
TEST_F(Arm64HeavyOptimizerFrontendTest, ShrnVec8B) {
  static const uint32_t code[] = {ShrnVec(/*q=*/false, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  // Vn.8h lanes 0-3: 0010 00F0 0100 0FF0 ; lanes 4-7: 1230 FF00 00A0 00B0.
  SetV128(&state_, 1, 0x0FF0010000F00010ULL, 0x00B000A0FF001230ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // lane>>4 low byte: 01 0F 10 FF 23 F0 0A 0B.
  EXPECT_EQ(VLo64(&state_, 0), 0x0B0AF023FF100F01ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zero
}

// RSHRN2 v0.8h, v1.4s, #4 (immh=0011, immb=4, Q=1): rounding narrow 32->16 into
// the upper half; exercises the wrap-on-overflow-is-benign edge on lane 3.
TEST_F(Arm64HeavyOptimizerFrontendTest, Rshrn2Vec8H) {
  static const uint32_t code[] = {RshrnVec(/*q=*/true, /*immh=*/0b0011, /*immb=*/4, 0, 1)};
  // Vn.4s words: 0x00000078 0x00000008 0x00012345 0xFFFFFFF8. round=8, shift=4.
  SetV128(&state_, 1, 0x0000000800000078ULL, 0xFFFFFFF800012345ULL);
  // Vd low 64 preserved; upper is overwritten by the narrowed lanes.
  SetV128(&state_, 0, 0x1111222233334444ULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (word+8)>>4 low 16: 0x0008 0x0001 0x1234 0x0000 (lane3 wraps but bit is dropped).
  EXPECT_EQ(VLo64(&state_, 0), 0x1111222233334444ULL);       // Vd[63:0] preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000123400010008ULL);
}

// RSHRN v0.2s, v1.2d, #4 (immh=0111, immb=4, Q=0): rounding narrow 64->32 (PSRLQ/PADDQ path).
TEST_F(Arm64HeavyOptimizerFrontendTest, RshrnVec2S) {
  static const uint32_t code[] = {RshrnVec(/*q=*/false, /*immh=*/0b0111, /*immb=*/4, 0, 1)};
  // Vn.2d dwords: 0x78, 0x12345678. round=8, shift=4.
  SetV128(&state_, 1, 0x0000000000000078ULL, 0x0000000012345678ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (dword+8)>>4 low 32: 0x00000008, 0x01234568.
  EXPECT_EQ(VLo64(&state_, 0), 0x0123456800000008ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zero
}

// SHRN with immh bit3 set (immh=1000) is RESERVED for narrowing shifts — bail to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, ShrnVecReservedBails) {
  static const uint32_t code[] = {ShrnVec(/*q=*/false, /*immh=*/0b1000, /*immb=*/4, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SQSHRN v0.8b, v1.8h, #4 (immh=0001, immb=4, Q=0): signed saturating narrow
// 16->8 with arithmetic right shift + PMINSW/PMAXSW clamp to [-128,127].
TEST_F(Arm64HeavyOptimizerFrontendTest, SqshrnVec8B) {
  static const uint32_t code[] = {SqshrnVec(/*q=*/false, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  // Vn.8h lanes 0-3: 0300 7FFF 8000 FFF0 ; lanes 4-7: 0800 F800 0000 FFFF.
  SetV128(&state_, 1, 0xFFF080007FFF0300ULL, 0xFFFF0000F8000800ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (sarw lane, 4) clamped [-128,127], low byte per lane:
  //   48->30, 2047->7F(sat), -2048->80(sat min), -1->FF, 128->7F(sat),
  //   -128->80, 0->00, -1->FF.
  EXPECT_EQ(VLo64(&state_, 0), 0xFF00807FFF807F30ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zero
}

// UQSHRN v0.4h, v1.4s, #12 (immh=0010, immb=4, Q=0): unsigned saturating narrow
// 32->16 with logical right shift + PMINUD clamp to 0xFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqshrnVec4H) {
  static const uint32_t code[] = {UqshrnVec(/*q=*/false, /*immh=*/0b0010, /*immb=*/4, 0, 1)};
  // Vn.4s words: 0x00012345 0xFFFFFFFF 0x0FFFF000 0x00007000. shift=12.
  SetV128(&state_, 1, 0xFFFFFFFF00012345ULL, 0x000070000FFFF000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (word>>12) clamped 0xFFFF, low 16 per lane: 0x12, 0xFFFFF->0xFFFF(sat),
  //   0xFFFF, 0x07.
  EXPECT_EQ(VLo64(&state_, 0), 0x0007FFFFFFFF0012ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zero
}

// SQSHRUN2 v0.16b, v1.8h, #4 (immh=0001, immb=4, Q=1): signed->unsigned
// saturating narrow 16->8 into the upper half; PMAXSW-vs-0 pins negatives to 0,
// PMINSW caps positives at 0xFF; Vd[63:0] preserved.
TEST_F(Arm64HeavyOptimizerFrontendTest, Sqshrun2Vec16B) {
  static const uint32_t code[] = {SqshrunVec(/*q=*/true, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  // Vn.8h lanes 0-3: 0300 F800 7FFF 0FF0 ; lanes 4-7: FFFF 0100 1000 0000.
  SetV128(&state_, 1, 0x0FF07FFFF8000300ULL, 0x000010000100FFFFULL);
  // Vd low 64 preserved; upper is overwritten by the narrowed lanes.
  SetV128(&state_, 0, 0x1111222233334444ULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (sarw lane, 4) clamped [0,255], low byte per lane:
  //   48->30, -128->00, 2047->FF(sat), 255->FF, -1->00, 16->10, 256->FF(sat),
  //   0->00.
  EXPECT_EQ(VLo64(&state_, 0), 0x1111222233334444ULL);       // Vd[63:0] preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00FF1000FFFF0030ULL);
}

// SQSHRN v0.2s, v1.2d, #28 (immh=0100, immb=4, Q=0): src=64 needs PSRAQ
// (AVX-512F-VL) which baseline SSE lacks — heavy bails to lite.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqshrnVecSrc64Bails) {
  static const uint32_t code[] = {SqshrnVec(/*q=*/false, /*immh=*/0b0100, /*immb=*/4, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SQRSHRN v0.8b, v1.8h, #4 (immh=0001, immb=4, Q=0): signed rounding saturating
// narrow 16->8. PADDSW pre-add of (1<<3)=8 (saturating), PSRAW #4, PMINSW/PMAXSW
// clamp to [-128,127].
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrshrnVec8B) {
  static const uint32_t code[] = {SqrshrnVec(/*q=*/false, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  // Vn.8h lanes 0-3: 0078 07F8 7FFF 0088 ; lanes 4-7: FFF8 FF88 8000 0000.
  SetV128(&state_, 1, 0x00887FFF07F80078ULL, 0x00008000FF88FFF8ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // ((lane + 8) >>4 signed) clamped [-128,127], low byte per lane:
  //   120->8, 2040->127->7F, 32767(PADDSW-sat->32767)->2047->7F, 136->9,
  //   -8->0, -120->-7->F9, -32760->-2048->-128->80, 0->0.
  EXPECT_EQ(VLo64(&state_, 0), 0x0080F900097F7F08ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zero
}

// UQRSHRN v0.4h, v1.4s, #12 (immh=0010, immb=4, Q=0): unsigned rounding
// saturating narrow 32->16. PMINUD pre-clamp to (0xFFFFFFFF-0x800), PADDD
// +0x800, PSRLD #12, PMINUD clamp to 0xFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqrshrnVec4H) {
  static const uint32_t code[] = {UqrshrnVec(/*q=*/false, /*immh=*/0b0010, /*immb=*/4, 0, 1)};
  // Vn.4s words: 0x00001800 0xFFFFFFFF | 0x00002800 0x00000700. shift=12.
  SetV128(&state_, 1, 0xFFFFFFFF00001800ULL, 0x0000070000002800ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // ((word + 0x800) >>12 unsigned) clamped 0xFFFF, low 16 per lane:
  //   0x1800->2, 0xFFFFFFFF(preclamp+add->0xFFFFFFFF)>>12=0xFFFFF->0xFFFF,
  //   0x2800->3, 0x0700->0.
  EXPECT_EQ(VLo64(&state_, 0), 0x00000003FFFF0002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zero
}

// SQRSHRUN2 v0.16b, v1.8h, #4 (immh=0001, immb=4, Q=1): signed->unsigned
// rounding saturating narrow 16->8 into the upper half. Carry-bit identity
// ((x+8)>>4 == (x>>4)+bit3(x)) avoids premature saturation; PMAXSW-vs-0 then
// PMINSW cap to [0,255]; Vd[63:0] preserved.
TEST_F(Arm64HeavyOptimizerFrontendTest, Sqrshrun2Vec16B) {
  static const uint32_t code[] = {SqrshrunVec(/*q=*/true, /*immh=*/0b0001, /*immb=*/4, 0, 1)};
  // Vn.8h lanes 0-3: 0078 FF88 07FF 0FF8 ; lanes 4-7: FFFF 0100 0808 0000.
  SetV128(&state_, 1, 0x0FF807FFFF880078ULL, 0x000008080100FFFFULL);
  // Vd low 64 preserved; upper is overwritten by the narrowed lanes.
  SetV128(&state_, 0, 0x1111222233334444ULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // ((lane + 8) >>4 signed) clamped [0,255], low byte per lane:
  //   120->8, -120->-7->0, 2047->128->80, 4088->256->255->FF,
  //   -1->0->0, 256->16->10, 2056->129->81, 0->0.
  EXPECT_EQ(VLo64(&state_, 0), 0x1111222233334444ULL);       // Vd[63:0] preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00811000FF800008ULL);
}

// SQRSHRN v0.2s, v1.2d, #28 (immh=0100, immb=4, Q=0): src=64 needs PSRAQ which
// baseline SSE lacks — heavy bails to lite (same as the non-rounding sibling).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqrshrnVecSrc64Bails) {
  static const uint32_t code[] = {SqrshrnVec(/*q=*/false, /*immh=*/0b0100, /*immb=*/4, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SQXTN v0.8b, v1.8h (size=00, Q=0): signed saturating narrow (PACKSSWB).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtnVec8B) {
  static const uint32_t code[] = {SqxtnVec(0b00, /*q=*/false, 0, 1)};
  // hwords 0001 007F 0080 FFFF | FF80 FF00 0100 8000.
  SetV128(&state_, 1, 0xFFFF0080007F0001ULL, 0x80000100FF00FF80ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x807F8080FF7F7F01ULL);   // signed-sat bytes
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UQXTN v0.8b, v1.8h (size=00, Q=0): unsigned saturating narrow (PMINUW+PACKUSWB).
TEST_F(Arm64HeavyOptimizerFrontendTest, UqxtnVec8B) {
  static const uint32_t code[] = {UqxtnVec(0b00, /*q=*/false, 0, 1)};
  // hwords 0001 00FF 0100 00AB | FFFF 0080 1234 0000.
  SetV128(&state_, 1, 0x00AB010000FF0001ULL, 0x000012340080FFFFULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00FF80FFABFFFF01ULL);   // clamp-to-255 bytes
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SQXTUN v0.8b, v1.8h (size=00, Q=0): signed->unsigned saturating narrow (PACKUSWB).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtunVec8B) {
  static const uint32_t code[] = {SqxtunVec(0b00, /*q=*/false, 0, 1)};
  // hwords 0001 00FF 0100 FFFF | 007F 8000 00AB 7FFF.
  SetV128(&state_, 1, 0xFFFF010000FF0001ULL, 0x7FFF00AB8000007FULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFAB007F00FFFF01ULL);   // neg->0, >255->255
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SQXTN v0.4h, v1.4s (size=01, Q=0): signed saturating narrow 4 dwords (PACKSSDW).
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtnVec4H) {
  static const uint32_t code[] = {SqxtnVec(0b01, /*q=*/false, 0, 1)};
  // dwords 00000001 00007FFF | 00008000 FFFF8000.
  SetV128(&state_, 1, 0x00007FFF00000001ULL, 0xFFFF800000008000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x80007FFF7FFF0001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SQXTN2 v0.16b, v1.8h (size=00, Q=1): saturating narrow into Vd.high, Vd.low kept.
TEST_F(Arm64HeavyOptimizerFrontendTest, Sqxtn2Vec16B) {
  static const uint32_t code[] = {SqxtnVec(0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFF0080007F0001ULL, 0x80000100FF00FF80ULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEBABEULL, 0xBBBBBBBBBBBBBBBBULL);  // low preserved
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xDEADBEEFCAFEBABEULL);   // Vd.low preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x807F8080FF7F7F01ULL);  // narrowed into high
}

// SQXTN v0.2s, v1.2d (size=10, Q=0): signed saturating 64->32 narrow via the
// PCMPGTQ clamp/blend. lane0 = -2^31-1 (< INT32_MIN) -> INT32_MIN 0x80000000;
// lane1 = +2^31 (> INT32_MAX) -> INT32_MAX 0x7FFFFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtnVec2S) {
  static const uint32_t code[] = {SqxtnVec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFF7FFFFFFFULL, 0x0000000080000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFF80000000ULL);   // {INT32_MIN, INT32_MAX}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// UQXTN v0.2s, v1.2d (size=10, Q=0): unsigned saturating 64->32 narrow.
// lane0 = 0x00000000ABCDEF01 (high32==0) passes; lane1 = 0x0000000100000000
// (high32!=0) -> UINT32_MAX 0xFFFFFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, UqxtnVec2S) {
  static const uint32_t code[] = {UqxtnVec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x00000000ABCDEF01ULL, 0x0000000100000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFABCDEF01ULL);   // {0xABCDEF01, UINT32_MAX}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SQXTUN v0.2s, v1.2d (size=10, Q=0): signed->unsigned saturating 64->32 narrow.
// lane0 = -1 (negative) -> 0; lane1 = 0x0000000123456789 (> UINT32_MAX) ->
// UINT32_MAX 0xFFFFFFFF.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqxtunVec2S) {
  static const uint32_t code[] = {SqxtunVec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFFFFFFFFFFULL, 0x0000000123456789ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF00000000ULL);   // {0, UINT32_MAX}
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SQXTN2 v0.4s, v1.2d (size=10, Q=1): narrow into Vd.high, Vd.low preserved.
TEST_F(Arm64HeavyOptimizerFrontendTest, Sqxtn2Vec4S) {
  static const uint32_t code[] = {SqxtnVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFF7FFFFFFFULL, 0x0000000080000000ULL);
  SetV128(&state_, 0, 0xDEADBEEFCAFEBABEULL, 0xBBBBBBBBBBBBBBBBULL);  // low preserved
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xDEADBEEFCAFEBABEULL);   // Vd.low preserved
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x7FFFFFFF80000000ULL);  // narrowed into high
}

// CLZ v0.4s, v1.4s (size=10, Q=1): per-lane 32-bit count-leading-zeros.
// lanes [1, 0x8000, 0x80000000, 0] -> CLZ [31, 16, 0, 32].
TEST_F(Arm64HeavyOptimizerFrontendTest, ClzVec4S) {
  static const uint32_t code[] = {ClzVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0000800000000001ULL, 0x0000000080000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000100000001FULL);      // [31, 16]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000002000000000ULL);  // [0, 32]
}

// CLZ v0.8b, v1.8b (size=00, Q=0): per-byte count-leading-zeros; upper 64 zeroed.
// bytes [01,80,FF,00,10,08,40,7F] -> CLZ [7,0,0,8,3,4,1,1].
TEST_F(Arm64HeavyOptimizerFrontendTest, ClzVec8B) {
  static const uint32_t code[] = {ClzVec(0b00, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x7F40081000FF8001ULL, 0x1111111111111111ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0101040308000007ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zeroed
}

// CLS v0.8h, v1.8h (size=01, Q=1): per-halfword count-leading-sign-bits.
// lanes [0001,FFFF,8000,4000, 0000,7FFF,C000,2000] -> CLS [14,15,0,0, 15,0,1,1].
TEST_F(Arm64HeavyOptimizerFrontendTest, ClsVec8H) {
  static const uint32_t code[] = {ClsVec(0b01, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x40008000FFFF0001ULL, 0x2000C0007FFF0000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000000F000EULL);      // [14,15,0,0]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000100010000000FULL);  // [15,0,1,1]
}

// CLS v0.2s, v1.2s (size=10, Q=0): 32-bit count-leading-sign-bits; upper zeroed.
// lanes [00000001, FFFFFFF0] -> CLS [30, 27].
TEST_F(Arm64HeavyOptimizerFrontendTest, ClsVec2S) {
  static const uint32_t code[] = {ClsVec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFF000000001ULL, 0x2222222222222222ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000001B0000001EULL);      // [30, 27]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zeroed
}

// SADDLP v0.8h, v1.16b (Q=1): full-128 signed byte->half pair-sum. Low bytes
// {0xFF,0x01,0x80,0x80,0x10,0x20,0x00,0xFF} -> {0, -256, 48, -1}; high bytes
// {0x7F,0x7F,0x01,0xFF,0x00,0x80,0x40,0x40} -> {254, 0, -128, 128}.
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddlpVec8H) {
  static const uint32_t code[] = {AddlpVec(true, false, 0b00, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFF002010808001FFULL, 0x40408000FF017F7FULL);
  SetV128(&state_, 0, 0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFF0030FF000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0080FF80000000FEULL);
}

// ---- AdvSIMD two-reg-misc compare-against-zero (heavy mirror, wave 2). ----

// CMEQ #0 .4S (Q=1): per-lane equal-to-zero via PCMPEQD against a zeroed reg.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmeqZeroVec4S) {
  static const uint32_t code[] = {CmeqZeroVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x0000000500000000ULL, 0x00000000FFFFFFFFULL);  // [0,5,-1,0]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);   // [T,F]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFF00000000ULL);  // [F,T]
}

// CMEQ #0 .2S (Q=0): the upper 64 bits of Vd must be zeroed by SetVRegFull.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmeqZeroVec2S) {
  static const uint32_t code[] = {CmeqZeroVec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x0000000500000000ULL, 0x1111111111111111ULL);  // [0,5]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);   // [T,F]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zeroed
}

// CMGT #0 .8H (Q=1): per-lane signed greater-than-zero via PCMPGTW(Vn, 0).
TEST_F(Arm64HeavyOptimizerFrontendTest, CmgtZeroVec8H) {
  static const uint32_t code[] = {CmgtZeroVec(0b01, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFF9C0000FFFF0005ULL, 0x0000000000000000ULL);  // [5,-1,0,-100 | 0,0,0,0]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000FFFFULL);   // only lane0 (5>0)
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// CMGE #0 .16B (Q=1): per-byte signed >= 0 via NOT(PCMPGTB(0, Vn)).
TEST_F(Arm64HeavyOptimizerFrontendTest, CmgeZeroVec16B) {
  static const uint32_t code[] = {CmgeZeroVec(0b00, /*q=*/true, 0, 1)};
  // low bytes [0x00,0x7F,0x80,0xFF,0x01,0x40,0xC0,0x00]; high 8 bytes all 0.
  SetV128(&state_, 1, 0x00C04001FF807F00ULL, 0x0000000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF00FFFF0000FFFFULL);   // [T,T,F,F,T,T,F,T]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);  // all bytes >=0
}

// CMLE #0 .16B (Q=1): per-byte signed <= 0 via NOT(PCMPGTB(Vn, 0)).
TEST_F(Arm64HeavyOptimizerFrontendTest, CmleZeroVec16B) {
  static const uint32_t code[] = {CmleZeroVec(0b00, /*q=*/true, 0, 1)};
  // low bytes [0x00,0x01,0x7F,0x80,0xFF,0x40,0xC0,0x00]; high 8 bytes all 0.
  SetV128(&state_, 1, 0x00C040FF807F0100ULL, 0x0000000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFF00FFFF0000FFULL);   // [T,F,F,T,T,F,T,T]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);  // all bytes <=0
}

// CMLT #0 .4S (Q=1): per-lane signed < 0 via PCMPGTD(0, Vn).
TEST_F(Arm64HeavyOptimizerFrontendTest, CmltZeroVec4S) {
  static const uint32_t code[] = {CmltZeroVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0xFFFFFFFF00000005ULL, 0x8000000000000000ULL);  // [5,-1,0,INT_MIN]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF00000000ULL);   // [F,T]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFF00000000ULL);  // [F,T]
}

// ---- AdvSIMD two-reg-misc FP compare-against-zero (heavy). ----
//
// FP32 .4S test vector, lanes [0.0, 5.0, -1.0, NaN]:
//   lane0=0.0f=0x00000000, lane1=5.0f=0x40A00000,
//   lane2=-1.0f=0xBF800000, lane3=NaN=0x7FC00000.
//   Lo64 = 0x40A0000000000000, Hi64 = 0x7FC00000BF800000.
// SSE ordered compares return FALSE on the NaN lane for every predicate.

// FCMEQ #0.0 .4S: lane==0 -> [T,F,F,F].
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmeqZeroVec4S) {
  static const uint32_t code[] = {FcmeqZeroVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x40A0000000000000ULL, 0x7FC00000BF800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);     // [T,F]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // [F,F]
}

// FCMGT #0.0 .4S: lane>0 -> [F,T,F,F].
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgtZeroVec4S) {
  static const uint32_t code[] = {FcmgtZeroVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x40A0000000000000ULL, 0x7FC00000BF800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFF00000000ULL);     // [F,T]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // [F,F]
}

// FCMGE #0.0 .4S: lane>=0 -> [T,T,F,F].
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgeZeroVec4S) {
  static const uint32_t code[] = {FcmgeZeroVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x40A0000000000000ULL, 0x7FC00000BF800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);     // [T,T]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // [F,F]
}

// FCMLT #0.0 .4S: lane<0 -> [F,F,T,F].
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmltZeroVec4S) {
  static const uint32_t code[] = {FcmltZeroVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x40A0000000000000ULL, 0x7FC00000BF800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000000ULL);     // [F,F]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFFULL);  // [T,F]
}

// FCMLE #0.0 .4S: lane<=0 -> [T,F,T,F].
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmleZeroVec4S) {
  static const uint32_t code[] = {FcmleZeroVec(0b10, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x40A0000000000000ULL, 0x7FC00000BF800000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);     // [T,F]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFFULL);  // [T,F]
}

// FCMEQ #0.0 .2S (Q=0): two FP32 lanes, Vd[127:64] must be zeroed.
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmeqZeroVec2S) {
  static const uint32_t code[] = {FcmeqZeroVec(0b10, /*q=*/false, 0, 1)};
  SetV128(&state_, 1, 0x40A0000000000000ULL, 0x1111111111111111ULL);  // [0.0, 5.0]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);     // [T,F]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // Q=0 upper zeroed
}

// FCMGT #0.0 .2D (FP64, Q=1): lanes [3.0, -2.0] -> [T,F].
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmgtZeroVec2D) {
  static const uint32_t code[] = {FcmgtZeroVec(0b11, /*q=*/true, 0, 1)};
  SetV128(&state_, 1, 0x4008000000000000ULL, 0xC000000000000000ULL);  // [3.0d, -2.0d]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);     // lane0 (3>0) T
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // lane1 (-2>0) F
}

// CMEQ/CMGT #0 .2D now lower (PCMPEQQ / PCMPGTQ); the frontend must translate
// them instead of bailing. Value-level checks live in Int64LaneMisc.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmeqZeroVec2DTranslates) {
  static const uint32_t code[] = {CmeqZeroVec(0b11, /*q=*/true, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_TRUE(ok);
  EXPECT_EQ(n, 1u);
}
TEST_F(Arm64HeavyOptimizerFrontendTest, CmgtZeroVec2DTranslates) {
  static const uint32_t code[] = {CmgtZeroVec(0b11, /*q=*/true, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_TRUE(ok);
  EXPECT_EQ(n, 1u);
}

// ---- AdvSIMD three-same signed/unsigned compares (heavy mirror, wave 2). ----

// CMGE .4S (Q=1): signed >= via NOT(PCMPGTD(Vm, Vn)).
TEST_F(Arm64HeavyOptimizerFrontendTest, CmgeVec4S) {
  static const uint32_t code[] = {CmgeVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFFFFFF00000005ULL, 0x0000000700000002ULL);  // [5,-1,2,7]
  SetV128(&state_, 2, 0x0000000300000003ULL, 0xFFFFFFFF00000002ULL);  // [3,3,2,-1]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFFULL);   // 5>=3 T, -1>=3 F
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);  // 2>=2 T, 7>=-1 T
}

// CMHI .16B (Q=1): unsigned > via sign-bias (GPR broadcast) + PCMPGTB.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmhiVec16B) {
  static const uint32_t code[] = {CmhiVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x55AA000110FF8000ULL, 0x0000000000000000ULL);
  SetV128(&state_, 2, 0x54AA000010FE7F01ULL, 0x0000000000000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFF0000FF00FFFF00ULL);   // [F,T,T,F,T,F,F,T]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // 0 >u 0 all false
}

// CMHI .4S (Q=1): unsigned > via sign-bias (PSLLD imm) + PCMPGTD.
// Exercises the newly added PslldXRegImm LIR op.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmhiVec4S) {
  static const uint32_t code[] = {CmhiVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000005ULL, 0x00000000FFFFFFFFULL);  // [5,2^31,-1,0]
  SetV128(&state_, 2, 0x7FFFFFFF00000003ULL, 0x00000001FFFFFFFFULL);  // [3,2^31-1,-1,1]
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);   // 5>3 T, 2^31>2^31-1 T
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);  // 0xFFFFFFFF>0xFFFFFFFF F, 0>1 F
}

// CMHS .8H (Q=1): unsigned >= via sign-bias (PSLLW imm) + PCMPGTW + invert.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmhsVec8H) {
  static const uint32_t code[] = {CmhsVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xFFFF000080000005ULL, 0x8000FFFF00000001ULL);
  SetV128(&state_, 2, 0xFFFF00017FFF0005ULL, 0x8001000000000002ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFF0000FFFFFFFFULL);   // [T,T,F,T]
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000FFFFFFFF0000ULL);  // [F,T,T,F]
}

// CMHI .2D (unsigned >) translates via sign-biased PCMPGTQ. The lane pair
// 0x8000000000000000 vs 1 is the discriminating case: unsigned says greater,
// signed says less, so an unbiased PCMPGTQ would flip it.
TEST_F(Arm64HeavyOptimizerFrontendTest, CmhiVec2D) {
  static const uint32_t code[] = {CmhiVec(0b11, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x8000000000000000ULL, 0x0000000000000005ULL);
  SetV128(&state_, 2, 0x0000000000000001ULL, 0x0000000000000005ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), ~0ULL);       // 0x80.. >u 1
  EXPECT_EQ(VUpperHi64(&state_, 0), 0ULL);   // 5 >u 5 is false
}

// AdvSimdThreeDiff widening multiply-accumulate (heavy tier). Result always
// fills 128 bits (8H/4S/2D). Signed vs unsigned diverge on high-bit inputs;
// the "2" (Q=1) forms take the upper 64 of Vn/Vm; MLAL/MLSL wrap (no
// saturation). Expected values match lite_translator.h::AdvSimdThreeDiff.

// UMULL .8H: 8x (unsigned byte * byte) -> 16-bit lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, UmullVec8H) {
  static const uint32_t code[] = {UmullVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vn.8B={2,3,0xFF,0x10,0,1,4,5} Vm.8B={4,5,2,0x10,7,9,3,2}
  SetV128(&state_, 1, 0x0504010010FF0302ULL, 0xEEEEEEEEEEEEEEEEULL);  // upper ignored
  SetV128(&state_, 2, 0x0203090710020504ULL, 0xDDDDDDDDDDDDDDDDULL);  // upper ignored
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={8,15,510,256,0,9,12,10}
  EXPECT_EQ(VLo64(&state_, 0), 0x010001FE000F0008ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000A000C00090000ULL);
}

// SMULL .8H: signed byte products (distinguishes from UMULL via 0xFF/0x80).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmullVec8H) {
  static const uint32_t code[] = {SmullVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vn.8B={0xFF,0x80,0x7F,0x02,0x00,0xFE,0x10,0x01}
  SetV128(&state_, 1, 0x0110FE00027F80FFULL, 0xEEEEEEEEEEEEEEEEULL);
  // Vm.8B={0x02,0x02,0x02,0x03,0x05,0xFF,0x10,0x7F}
  SetV128(&state_, 2, 0x7F10FF0503020202ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={-2,-256,254,6,0,2,256,127}
  EXPECT_EQ(VLo64(&state_, 0), 0x000600FEFF00FFFEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x007F010000020000ULL);
}

// UMULL .4S: 4x (unsigned halfword * halfword) -> 32-bit lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, UmullVec4S) {
  static const uint32_t code[] = {UmullVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vn.4H={0xFFFF,0x1234,0x0002,0x0100} Vm.4H={0xFFFF,0x0002,0x0003,0x0010}
  SetV128(&state_, 1, 0x010000021234FFFFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x001000030002FFFFULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={0xFFFE0001,0x2468,6,0x1000}
  EXPECT_EQ(VLo64(&state_, 0), 0x00002468FFFE0001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000100000000006ULL);
}

// SMULL .4S: signed halfword products (0xFFFF=-1, 0x8000=INT16_MIN).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmullVec4S) {
  static const uint32_t code[] = {SmullVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vn.4H={0xFFFF,0x8000,0x7FFF,0x0002} Vm.4H={2,2,2,3}
  SetV128(&state_, 1, 0x00027FFF8000FFFFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0003000200020002ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={-2,-65536,65534,6}
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFF0000FFFFFFFEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000060000FFFEULL);
}

// UMULL .2D: 2x (unsigned word * word) -> 64-bit lanes (PMULUDQ path).
TEST_F(Arm64HeavyOptimizerFrontendTest, UmullVec2D) {
  static const uint32_t code[] = {UmullVec(0b10, /*q=*/false, 0, 1, 2)};
  // Vn.2S={0xFFFFFFFF,0x00010000} Vm.2S={0xFFFFFFFF,0x00000003}
  SetV128(&state_, 1, 0x00010000FFFFFFFFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x00000003FFFFFFFFULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={0xFFFFFFFE00000001, 0x30000}
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFE00000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000030000ULL);
}

// SMULL .2D: signed word products (INT32_MIN, -1) (PMULDQ path).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmullVec2D) {
  static const uint32_t code[] = {SmullVec(0b10, /*q=*/false, 0, 1, 2)};
  // Vn.2S={0xFFFFFFFF(-1),0x80000000} Vm.2S={2,2}
  SetV128(&state_, 1, 0x80000000FFFFFFFFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0000000200000002ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={-2, -4294967296}
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFFFFFFFFFFEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFF00000000ULL);
}

// UMULL2 .8H: Q=1 takes bytes 8..15 of Vn/Vm; low halves must be ignored.
TEST_F(Arm64HeavyOptimizerFrontendTest, Umull2Vec8H) {
  static const uint32_t code[] = {UmullVec(0b00, /*q=*/true, 0, 1, 2)};
  // Vn.16B hi={1,2,3,4,5,6,7,8} Vm.16B hi={2,2,2,2,2,2,2,2}
  SetV128(&state_, 1, 0xDEADBEEFDEADBEEFULL, 0x0807060504030201ULL);
  SetV128(&state_, 2, 0xCAFECAFECAFECAFEULL, 0x0202020202020202ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={2,4,6,8,10,12,14,16}
  EXPECT_EQ(VLo64(&state_, 0), 0x0008000600040002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0010000E000C000AULL);
}

// SMULL2 .4S: Q=1 + signed halfword products from the upper half.
TEST_F(Arm64HeavyOptimizerFrontendTest, Smull2Vec4S) {
  static const uint32_t code[] = {SmullVec(0b01, /*q=*/true, 0, 1, 2)};
  // Vn.8H hi={0xFFFF,0x0003,0x8000,0x0002} Vm.8H hi={2,2,2,2}
  SetV128(&state_, 1, 0xDEADBEEFDEADBEEFULL, 0x000280000003FFFFULL);
  SetV128(&state_, 2, 0x1111111111111111ULL, 0x0002000200020002ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={-2,6,-65536,4}
  EXPECT_EQ(VLo64(&state_, 0), 0x00000006FFFFFFFEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000004FFFF0000ULL);
}

// PMULL64 (.1Q from Vn.D[0] x Vm.D[0]) — single PCLMULQDQ, imm 0x00.
// clmul(0x1000000000000001, 0x1000000000000001) = (x^60+1)^2 = x^120+1 in
// GF(2): bit 120 -> high qword bit 56 (0x0100000000000000), bit 0 -> low = 1.
TEST_F(Arm64HeavyOptimizerFrontendTest, Pmull64) {
  static const uint32_t code[] = {PmullVec(0b11, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x1000000000000001ULL, 0xDEADBEEFDEADBEEFULL);  // D[1] ignored
  SetV128(&state_, 2, 0x1000000000000001ULL, 0xBBBBBBBBBBBBBBBBULL);  // D[1] ignored
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);  // poison dst
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0100000000000000ULL);
}

// PMULL2 64 (.1Q from Vn.D[1] x Vm.D[1]) — PCLMULQDQ imm 0x11 (high x high).
// Same operands as above in the HIGH qwords; D[0] must be ignored.
TEST_F(Arm64HeavyOptimizerFrontendTest, Pmull2_64UsesHighHalf) {
  static const uint32_t code[] = {PmullVec(0b11, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xDEADBEEFDEADBEEFULL, 0x1000000000000001ULL);  // D[0] ignored
  SetV128(&state_, 2, 0xBBBBBBBBBBBBBBBBULL, 0x1000000000000001ULL);  // D[0] ignored
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0100000000000000ULL);
}

// PMULL .8H poly8 widening (8 independent 8-bit carryless products -> 16-bit
// lanes). Same operands/expected as the validated lite Pmull8hPoly8 test.
TEST_F(Arm64HeavyOptimizerFrontendTest, Pmull8hPoly8) {
  static const uint32_t code[] = {PmullVec(0b00, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0xAA07108001FFCA53ULL, 0xDEADBEEFDEADBEEFULL);  // high ignored
  SetV128(&state_, 2, 0x5507108002FF53CAULL, 0xBBBBBBBBBBBBBBBBULL);  // high ignored
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000255553F7E3F7EULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x2222001501004000ULL);
}

// PMULL2 .8H poly8 — same products but from the HIGH 8 bytes of Vn/Vm.
TEST_F(Arm64HeavyOptimizerFrontendTest, Pmull2_8hPoly8UsesHighHalf) {
  static const uint32_t code[] = {PmullVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0xDEADBEEFDEADBEEFULL, 0xAA07108001FFCA53ULL);  // low ignored
  SetV128(&state_, 2, 0xBBBBBBBBBBBBBBBBULL, 0x5507108002FF53CAULL);  // low ignored
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_EQ(VLo64(&state_, 0), 0x000255553F7E3F7EULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x2222001501004000ULL);
}

// UMLAL .4S: accumulate into Vd (32-bit wrap, no saturation).
TEST_F(Arm64HeavyOptimizerFrontendTest, UmlalVec4S) {
  static const uint32_t code[] = {UmlalVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vd.4S={10,20,0xFFFFFFFF,100} Vn.4H={3,4,5,6} Vm.4H={2,2,2,2}
  SetV128(&state_, 0, 0x000000140000000AULL, 0x00000064FFFFFFFFULL);
  SetV128(&state_, 1, 0x0006000500040003ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0002000200020002ULL, 0xDDDDDDDDDDDDDDDDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={6,8,10,12}; Vd+={16,28,9(wrap),112}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000001C00000010ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000007000000009ULL);
}

// SMLAL .8H: signed accumulate (16-bit wrap).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmlalVec8H) {
  static const uint32_t code[] = {SmlalVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vd.8H={0x10,0x20,0xFFFF,0,1,2,3,4}
  SetV128(&state_, 0, 0x0000FFFF00200010ULL, 0x0004000300020001ULL);
  // Vn.8B={0xFF,0x02,0x80,0x03,0x04,0x05,0x06,0x07}
  SetV128(&state_, 1, 0x07060504038002FFULL, 0xEEEEEEEEEEEEEEEEULL);
  // Vm.8B={0x02,0x03,0x02,0x04,0x01,0x01,0x01,0x01}
  SetV128(&state_, 2, 0x0101010104020302ULL, 0xDDDDDDDDDDDDDDDDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={-2,6,-256,12,4,5,6,7}; Vd+={0x000E,0x0026,0xFEFF,0x000C,5,7,9,0x0B}
  EXPECT_EQ(VLo64(&state_, 0), 0x000CFEFF0026000EULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000B000900070005ULL);
}

// SMLSL .4S: signed subtract-accumulate (Vd -= products).
TEST_F(Arm64HeavyOptimizerFrontendTest, SmlslVec4S) {
  static const uint32_t code[] = {SmlslVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vd.4S={100,50,0,10} Vn.4H={0xFFFF,3,5,0x7FFF} Vm.4H={2,2,2,2}
  SetV128(&state_, 0, 0x0000003200000064ULL, 0x0000000A00000000ULL);
  SetV128(&state_, 1, 0x7FFF00050003FFFFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0002000200020002ULL, 0xDDDDDDDDDDDDDDDDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={-2,6,10,65534}; Vd-={102,44,-10,-65524}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000002C00000066ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFF000CFFFFFFF6ULL);
}

// UMLSL .2D: unsigned subtract-accumulate (PMULUDQ + PSUBQ, 64-bit wrap).
TEST_F(Arm64HeavyOptimizerFrontendTest, UmlslVec2D) {
  static const uint32_t code[] = {UmlslVec(0b10, /*q=*/false, 0, 1, 2)};
  // Vd.2D={0x0000000200000003,5} Vn.2S={0xFFFFFFFF,2} Vm.2S={2,3}
  SetV128(&state_, 0, 0x0000000200000003ULL, 0x0000000000000005ULL);
  SetV128(&state_, 1, 0x00000002FFFFFFFFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0000000300000002ULL, 0xDDDDDDDDDDDDDDDDULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // products={0x1FFFFFFFE,6}; Vd-={5, -1(wrap)}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000005ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFFFFFFFFFFFFFULL);
}

// AdvSimdThreeDiff widening add/sub (heavy tier). Result always fills 128 bits
// (8H/4S/2D). The L-forms widen both narrow sources; the W-forms take Vn as an
// already-wide 128-bit vector and widen only Vm. Q=1 ("2") selects the upper 64
// of the narrow sources. Expected values match lite_translator.h::AdvSimdThreeDiff.

// UADDL .8H: 8x (uint8 + uint8) -> 16-bit lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, UaddlVec8H) {
  static const uint32_t code[] = {UaddlVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vn.8B={0xFF,0x01,0x80,0x10,0x00,0x7F,0x02,0x05}
  SetV128(&state_, 1, 0x05027F00108001FFULL, 0xEEEEEEEEEEEEEEEEULL);
  // Vm.8B={0x01,0x02,0x80,0xF0,0xFF,0x01,0x03,0x0A}
  SetV128(&state_, 2, 0x0A0301FFF0800201ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // sums={0x100,0x003,0x100,0x100,0x0FF,0x080,0x005,0x00F}
  EXPECT_EQ(VLo64(&state_, 0), 0x0100010000030100ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000F0005008000FFULL);
}

// SADDL .8H: signed byte add-long (0xFF=-1, 0x80=-128).
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddlVec8H) {
  static const uint32_t code[] = {SaddlVec(0b00, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x05027F00108001FFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0A0301FFF0800201ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // sums(signed)={0,3,-256,0,-1,128,5,15}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000FF0000030000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000F00050080FFFFULL);
}

// USUBL .4S: 4x (uint16 - uint16) -> 32-bit lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, UsublVec4S) {
  static const uint32_t code[] = {UsublVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vn.4H={0x0001,0x8000,0xFFFF,0x1234}
  SetV128(&state_, 1, 0x1234FFFF80000001ULL, 0xEEEEEEEEEEEEEEEEULL);
  // Vm.4H={0x0002,0x0001,0x0001,0x1234}
  SetV128(&state_, 2, 0x1234000100010002ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // diffs(unsigned)={-1,0x7FFF,0xFFFE,0}
  EXPECT_EQ(VLo64(&state_, 0), 0x00007FFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000000000FFFEULL);
}

// SSUBL .4S: signed halfword sub-long (0x8000=INT16_MIN).
TEST_F(Arm64HeavyOptimizerFrontendTest, SsublVec4S) {
  static const uint32_t code[] = {SsublVec(0b01, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x1234FFFF80000001ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x1234000100010002ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // diffs(signed)={-1,-32769,-2,0}
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFF7FFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFEULL);
}

// UADDL2 .2D: unsigned word add-long, upper 64 of Vn/Vm (Q=1) -> 64-bit lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, Uaddl2Vec2D) {
  static const uint32_t code[] = {UaddlVec(0b10, /*q=*/true, 0, 1, 2)};
  // Vn.4S upper={0xFFFFFFFF,0x00000003}; lower ignored.
  SetV128(&state_, 1, 0x1111111111111111ULL, 0x00000003FFFFFFFFULL);
  // Vm.4S upper={0x00000002,0xFFFFFFFF}; lower ignored.
  SetV128(&state_, 2, 0x2222222222222222ULL, 0xFFFFFFFF00000002ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // sums={0x1_00000001, 0x1_00000002}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000100000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000100000002ULL);
}

// SADDW .8H: Vn is a full .8H (128-bit) + SignExtend(Vm.8B).
TEST_F(Arm64HeavyOptimizerFrontendTest, SaddwVec8H) {
  static const uint32_t code[] = {SaddwVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vn.8H={0x0100,0x0003,0x0100,0x0100,0x00FF,0x0080,0x0005,0x000F}
  SetV128(&state_, 1, 0x0100010000030100ULL, 0x000F0005008000FFULL);
  // Vm.8B(signed)={1,-1,2,-128,16,1,-1,10}
  SetV128(&state_, 2, 0x0AFF01108002FF01ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // sums={0x101,0x002,0x102,0x080,0x10F,0x081,0x004,0x019}
  EXPECT_EQ(VLo64(&state_, 0), 0x0080010200020101ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x001900040081010FULL);
}

// SSUBW .4S: Vn is a full .4S (128-bit) - SignExtend(Vm.4H).
TEST_F(Arm64HeavyOptimizerFrontendTest, SsubwVec4S) {
  static const uint32_t code[] = {SsubwVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vn.4S={0x00001000,0xFFFFFFFF,0x00000000,0x80000000}
  SetV128(&state_, 1, 0xFFFFFFFF00001000ULL, 0x8000000000000000ULL);
  // Vm.4H(signed)={1,-1,-32768,32767}
  SetV128(&state_, 2, 0x7FFF8000FFFF0001ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // diffs={0xFFF,0,0x8000,0x7FFF8001}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000FFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x7FFF800100008000ULL);
}

// UADDW2 .4S: Vn full .4S + ZeroExtend(upper Vm.8H) (Q=1).
TEST_F(Arm64HeavyOptimizerFrontendTest, Uaddw2Vec4S) {
  static const uint32_t code[] = {UaddwVec(0b01, /*q=*/true, 0, 1, 2)};
  // Vn.4S={0x00010000,0x00000001,0xFFFFFFFE,0x12345678}
  SetV128(&state_, 1, 0x0000000100010000ULL, 0x12345678FFFFFFFEULL);
  // Vm.8H upper={0x0002,0xFFFF,0x0001,0x1000}; lower ignored.
  SetV128(&state_, 2, 0x2222222222222222ULL, 0x10000001FFFF0002ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // sums={0x00010002,0x00010000,0xFFFFFFFF,0x12346678}
  EXPECT_EQ(VLo64(&state_, 0), 0x0001000000010002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x12346678FFFFFFFFULL);
}

// UABDL .8H: unsigned byte absolute-difference-long -> 16-bit lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, UabdlVec8H) {
  static const uint32_t code[] = {UabdlVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vn.8B={0xFF,0x01,0x80,0x10,0x00,0x7F,0x02,0x05}
  SetV128(&state_, 1, 0x05027F00108001FFULL, 0xEEEEEEEEEEEEEEEEULL);
  // Vm.8B={0x01,0x02,0x80,0xF0,0xFF,0x01,0x03,0x0A}
  SetV128(&state_, 2, 0x0A0301FFF0800201ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // |diff|(unsigned)={254,1,0,224,255,126,1,5}
  EXPECT_EQ(VLo64(&state_, 0), 0x00E00000000100FEULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00050001007E00FFULL);
}

// SABDL .8H: signed byte absolute-difference-long (0xFF=-1, 0x80=-128).
TEST_F(Arm64HeavyOptimizerFrontendTest, SabdlVec8H) {
  static const uint32_t code[] = {SabdlVec(0b00, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x05027F00108001FFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0A0301FFF0800201ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // Vn(signed)={-1,1,-128,16,0,127,2,5}; Vm(signed)={1,2,-128,-16,-1,1,3,10}
  // |diff|={2,1,0,32,1,126,1,5}
  EXPECT_EQ(VLo64(&state_, 0), 0x0020000000010002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00050001007E0001ULL);
}

// UABDL .4S: unsigned halfword abs-diff-long -> 32-bit lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, UabdlVec4S) {
  static const uint32_t code[] = {UabdlVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vn.4H={0x0001,0x8000,0xFFFF,0x1234}
  SetV128(&state_, 1, 0x1234FFFF80000001ULL, 0xEEEEEEEEEEEEEEEEULL);
  // Vm.4H={0x0002,0x0001,0x0001,0x1234}
  SetV128(&state_, 2, 0x1234000100010002ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // |diff|(unsigned)={1,0x7FFF,0xFFFE,0}
  EXPECT_EQ(VLo64(&state_, 0), 0x00007FFF00000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x000000000000FFFEULL);
}

// SABDL .4S: signed halfword abs-diff-long (0x8000=INT16_MIN).
TEST_F(Arm64HeavyOptimizerFrontendTest, SabdlVec4S) {
  static const uint32_t code[] = {SabdlVec(0b01, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x1234FFFF80000001ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x1234000100010002ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // Vn(signed)={1,-32768,-1,0x1234}; Vm(signed)={2,1,1,0x1234}
  // |diff|={1,32769,2,0}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000800100000001ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000002ULL);
}

// UABDL2 .2D: unsigned word abs-diff-long on the UPPER half (Q=1) -> 64-bit
// lanes. Exercises the size=10 Pcmpgtq signed-abs path.
TEST_F(Arm64HeavyOptimizerFrontendTest, Uabdl2Vec2D) {
  static const uint32_t code[] = {UabdlVec(0b10, /*q=*/true, 0, 1, 2)};
  // Vn.4S upper={0xFFFFFFFF,0x00000003}; lower ignored.
  SetV128(&state_, 1, 0x1111111111111111ULL, 0x00000003FFFFFFFFULL);
  // Vm.4S upper={0x00000002,0xFFFFFFFF}; lower ignored.
  SetV128(&state_, 2, 0x2222222222222222ULL, 0xFFFFFFFF00000002ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // |0xFFFFFFFF-2|=0xFFFFFFFD; |3-0xFFFFFFFF|=0xFFFFFFFC (unsigned widen).
  EXPECT_EQ(VLo64(&state_, 0), 0x00000000FFFFFFFDULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000000FFFFFFFCULL);
}

// SABDL2 .2D: signed word abs-diff-long on the UPPER half (Q=1) -> 64-bit lanes.
TEST_F(Arm64HeavyOptimizerFrontendTest, Sabdl2Vec2D) {
  static const uint32_t code[] = {SabdlVec(0b10, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x1111111111111111ULL, 0x00000003FFFFFFFFULL);
  SetV128(&state_, 2, 0x2222222222222222ULL, 0xFFFFFFFF00000002ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // Vn(signed words)={-1,3}; Vm(signed words)={2,-1}; |diff|={3,4}.
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000003ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000004ULL);
}

// SABAL .8H: signed byte abs-diff-long, accumulated into Vd.8H.
TEST_F(Arm64HeavyOptimizerFrontendTest, SabalVec8H) {
  static const uint32_t code[] = {SabalVec(0b00, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x05027F00108001FFULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0A0301FFF0800201ULL, 0xDDDDDDDDDDDDDDDDULL);
  // Vd.8H accumulator={0x0010,0x0020,0x0030,0x0040,0x0001,0x0002,0x0003,0x0004}
  SetV128(&state_, 0, 0x0040003000200010ULL, 0x0004000300020001ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // |diff|={2,1,0,32,1,126,1,5} + Vd -> {0x12,0x21,0x30,0x60,0x02,0x80,0x04,0x09}
  EXPECT_EQ(VLo64(&state_, 0), 0x0060003000210012ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0009000400800002ULL);
}

// UABAL .4S: unsigned halfword abs-diff-long, accumulated into Vd.4S.
TEST_F(Arm64HeavyOptimizerFrontendTest, UabalVec4S) {
  static const uint32_t code[] = {UabalVec(0b01, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x1234FFFF80000001ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x1234000100010002ULL, 0xDDDDDDDDDDDDDDDDULL);
  // Vd.4S accumulator={0x00000010,0x00000020,0x00000100,0x00000200}
  SetV128(&state_, 0, 0x0000002000000010ULL, 0x0000020000000100ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // |diff|={1,0x7FFF,0xFFFE,0} + Vd -> {0x11,0x801F,0x100FE,0x200}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000801F00000011ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x00000200000100FEULL);
}

// USUBW .8H: Vn full .8H - ZeroExtend(Vm.8B).
TEST_F(Arm64HeavyOptimizerFrontendTest, UsubwVec8H) {
  static const uint32_t code[] = {UsubwVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vn.8H={0x0100,0x0003,0x0100,0x0100,0x00FF,0x0080,0x0005,0x000F}
  SetV128(&state_, 1, 0x0100010000030100ULL, 0x000F0005008000FFULL);
  // Vm.8B(unsigned)={1,255,2,128,16,1,255,10}
  SetV128(&state_, 2, 0x0AFF01108002FF01ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // diffs={0x0FF,0xFF04,0x0FE,0x080,0x0EF,0x07F,0xFF06,0x005}
  EXPECT_EQ(VLo64(&state_, 0), 0x008000FEFF0400FFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0005FF06007F00EFULL);
}

// AdvSimdThreeDiff narrowing-high ADDHN/SUBHN/RADDHN/RSUBHN (heavy tier). Add or
// subtract the two wide-lane sources, then take the HIGH half of each lane as
// the narrow result. Q=0 zero-extends the upper 64; Q=1 ("2") merges into
// Vd.high. Rounding variants add 1<<(esize-1) of the SOURCE lane first. Expected
// values match lite_translator.h::AdvSimdThreeDiff.

// ADDHN .8B <- .8H: high byte of each 16-bit sum.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddhnVec8B) {
  static const uint32_t code[] = {AddhnVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vn.8H={0x0100,0x0200,0xFF00,0x1234,0x00FF,0x8000,0x0080,0xABCD}
  SetV128(&state_, 1, 0x1234FF0002000100ULL, 0xABCD0080800000FFULL);
  // Vm.8H={0x0100,0x0100,0x0100,0x1000,0x0001,0x8000,0x0080,0x1111}
  SetV128(&state_, 2, 0x1000010001000100ULL, 0x1111008080000001ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // high-bytes={0x02,0x03,0x00,0x22,0x01,0x00,0x01,0xBC}
  EXPECT_EQ(VLo64(&state_, 0), 0xBC01000122000302ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SUBHN .4H <- .4S: bits[31:16] of each 32-bit difference.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubhnVec4H) {
  static const uint32_t code[] = {SubhnVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vn.4S={0x00030000,0x12340000,0xFFFF0000,0x00018000}
  SetV128(&state_, 1, 0x1234000000030000ULL, 0x00018000FFFF0000ULL);
  // Vm.4S={0x00010000,0x00340000,0x00010000,0x00008000}
  SetV128(&state_, 2, 0x0034000000010000ULL, 0x0000800000010000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // diffs>>16={0x0002,0x1200,0xFFFE,0x0001}
  EXPECT_EQ(VLo64(&state_, 0), 0x0001FFFE12000002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// ADDHN .2S <- .2D: bits[63:32] of each 64-bit sum (PSHUFD gather, no 64->32 pack).
TEST_F(Arm64HeavyOptimizerFrontendTest, AddhnVec2S) {
  static const uint32_t code[] = {AddhnVec(0b10, /*q=*/false, 0, 1, 2)};
  // Vn.2D={0x0000000100000000,0xFFFFFFFF00000000}
  SetV128(&state_, 1, 0x0000000100000000ULL, 0xFFFFFFFF00000000ULL);
  // Vm.2D={0x0000000200000000,0x0000000100000000}
  SetV128(&state_, 2, 0x0000000200000000ULL, 0x0000000100000000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // sums>>32={0x00000003,0x00000000}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000000000003ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// RADDHN .4H <- .4S: rounding add (bias 1<<15 per 32-bit lane) then >>16.
TEST_F(Arm64HeavyOptimizerFrontendTest, RaddhnVec4H) {
  static const uint32_t code[] = {RaddhnVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vn.4S={0x00018000,0x00027FFF,0x00000000,0xFFFF8000}
  SetV128(&state_, 1, 0x00027FFF00018000ULL, 0xFFFF800000000000ULL);
  // Vm.4S={0x00000000,0x00000000,0x00008000,0x00000000}
  SetV128(&state_, 2, 0x0000000000000000ULL, 0x0000000000008000ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (sum+0x8000)>>16={0x0002,0x0002,0x0001,0x0000}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000000100020002ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// ADDHN2 .16B <- .8H (Q=1): result merges into Vd.high, Vd.low preserved.
TEST_F(Arm64HeavyOptimizerFrontendTest, Addhn2Vec16B) {
  static const uint32_t code[] = {AddhnVec(0b00, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x1234FF0002000100ULL, 0xABCD0080800000FFULL);
  SetV128(&state_, 2, 0x1000010001000100ULL, 0x1111008080000001ULL);
  SetV128(&state_, 0, 0x1122334455667788ULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // same high-bytes as AddhnVec8B, now in Vd.high; Vd.low preserved.
  EXPECT_EQ(VLo64(&state_, 0), 0x1122334455667788ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xBC01000122000302ULL);
}

// RSUBHN .8B <- .8H: rounding subtract (bias 1<<7 per 16-bit lane) then >>8.
TEST_F(Arm64HeavyOptimizerFrontendTest, RsubhnVec8B) {
  static const uint32_t code[] = {RsubhnVec(0b00, /*q=*/false, 0, 1, 2)};
  // Vn.8H={0x0180,0x0200,0x00FF,0x8000,0x0100,0x1234,0xFF80,0x0080}
  SetV128(&state_, 1, 0x800000FF02000180ULL, 0x0080FF8012340100ULL);
  // Vm.8H={0x0000,0x0080,0x0000,0x0000,0x0080,0x0034,0x0000,0x0000}
  SetV128(&state_, 2, 0x0000000000800000ULL, 0x0000000000340080ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // (diff+0x80)>>8={0x02,0x02,0x01,0x80,0x01,0x12,0x00,0x01}
  EXPECT_EQ(VLo64(&state_, 0), 0x0100120180010202ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000000ULL);
}

// SQDMULL .4S <- .4H: signed saturating doubling multiply long. The
// (-32768)*(-32768) lane doubles to 2^31 and saturates to INT32_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmullVec4S) {
  static const uint32_t code[] = {SqdmullVec(0b01, /*q=*/false, 0, 1, 2)};
  // Vn.4H={3,-4,-32768,100}  Vm.4H={5,7,-32768,-100}
  SetV128(&state_, 1, 0x00648000FFFC0003ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0xFF9C800000070005ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // 2*prod={30,-56,SAT->0x7FFFFFFF,-20000}
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFC80000001EULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFB1E07FFFFFFFULL);
}

// SQDMLAL .4S: signed saturating doubling multiply-accumulate. The
// 5+INT32_MAX lane overflows on the add and saturates to INT32_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmlalVec4S) {
  static const uint32_t code[] = {SqdmlalVec(0b01, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x00648000FFFC0003ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0xFF9C800000070005ULL, 0xDDDDDDDDDDDDDDDDULL);
  // Vd.4S={10,100,5,0x7FFFFF00}; P={30,-56,0x7FFFFFFF,-20000}
  SetV128(&state_, 0, 0x000000640000000AULL, 0x7FFFFF0000000005ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // {40,44,SAT->0x7FFFFFFF,0x7FFFB0E0}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000002C00000028ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x7FFFB0E07FFFFFFFULL);
}

// SQDMLSL .4S: signed saturating doubling multiply-subtract. The
// INT32_MIN-INT32_MAX lane underflows on the sub and saturates to INT32_MIN.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmlslVec4S) {
  static const uint32_t code[] = {SqdmlslVec(0b01, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x00648000FFFC0003ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0xFF9C800000070005ULL, 0xDDDDDDDDDDDDDDDDULL);
  // Vd.4S={100,10,INT32_MIN,0x80000100}; P={30,-56,0x7FFFFFFF,-20000}
  SetV128(&state_, 0, 0x0000000A00000064ULL, 0x8000010080000000ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // {70,66,SAT->0x80000000,0x80004F20}
  EXPECT_EQ(VLo64(&state_, 0), 0x0000004200000046ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x80004F2080000000ULL);
}

// SQDMULL2 .4S: Q=1 takes the upper 64 bits (bytes 8..15) of Vn/Vm as .4H.
TEST_F(Arm64HeavyOptimizerFrontendTest, Sqdmull2Vec4S) {
  static const uint32_t code[] = {SqdmullVec(0b01, /*q=*/true, 0, 1, 2)};
  SetV128(&state_, 1, 0x1111111111111111ULL, 0x00648000FFFC0003ULL);
  SetV128(&state_, 2, 0x2222222222222222ULL, 0xFF9C800000070005ULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // same result as SqdmullVec4S (upper-half sources)
  EXPECT_EQ(VLo64(&state_, 0), 0xFFFFFFC80000001EULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0xFFFFB1E07FFFFFFFULL);
}

// SQDMULL .2D <- .2S: 64-bit manual saturation. The INT32_MIN*INT32_MIN lane
// doubles to 2^63 and saturates to INT64_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmullVec2D) {
  static const uint32_t code[] = {SqdmullVec(0b10, /*q=*/false, 0, 1, 2)};
  // Vn.2S={3,INT32_MIN}  Vm.2S={5,INT32_MIN}
  SetV128(&state_, 1, 0x8000000000000003ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x8000000000000005ULL, 0xDDDDDDDDDDDDDDDDULL);
  SetV128(&state_, 0, 0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // {30, SAT->INT64_MAX}
  EXPECT_EQ(VLo64(&state_, 0), 0x000000000000001EULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x7FFFFFFFFFFFFFFFULL);
}

// SQDMLAL .2D: 64-bit saturating accumulate. INT64_MAX + 2^32 overflows on the
// add and saturates to INT64_MAX.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmlalVec2D) {
  static const uint32_t code[] = {SqdmlalVec(0b10, /*q=*/false, 0, 1, 2)};
  // Vn.2S={2^30,3}  Vm.2S={2,5}; P={2^32,30}
  SetV128(&state_, 1, 0x0000000340000000ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0000000500000002ULL, 0xDDDDDDDDDDDDDDDDULL);
  // Vd.2D={INT64_MAX,100}
  SetV128(&state_, 0, 0x7FFFFFFFFFFFFFFFULL, 0x0000000000000064ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // {SAT->INT64_MAX, 130}
  EXPECT_EQ(VLo64(&state_, 0), 0x7FFFFFFFFFFFFFFFULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000082ULL);
}

// SQDMLSL .2D: 64-bit saturating subtract. INT64_MIN - 2^32 underflows on the
// sub and saturates to INT64_MIN.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqdmlslVec2D) {
  static const uint32_t code[] = {SqdmlslVec(0b10, /*q=*/false, 0, 1, 2)};
  SetV128(&state_, 1, 0x0000000340000000ULL, 0xEEEEEEEEEEEEEEEEULL);
  SetV128(&state_, 2, 0x0000000500000002ULL, 0xDDDDDDDDDDDDDDDDULL);
  // Vd.2D={INT64_MIN,100}; P={2^32,30}
  SetV128(&state_, 0, 0x8000000000000000ULL, 0x0000000000000064ULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  // {SAT->INT64_MIN, 70}
  EXPECT_EQ(VLo64(&state_, 0), 0x8000000000000000ULL);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0x0000000000000046ULL);
}

// LDR/STR (SIMD&FP, register offset):
//   size(31:30) 111 V=1(26) 00 opc(23:22) 1(21) Rm(20:16) option(15:13) S(12)
//   10(11:10) Rn(9:5) Rt(4:0).
// opc selects the operation and the 128-bit form (128-bit load opc=11 / store
// opc=10; non-128 load opc=01 / store opc=00); size selects the transfer width
// (128->00, 64->11, 32->10). option is the extend (LSL/UXTX=011, UXTW=010,
// SXTW=110) and S enables the log2(bytes) shift. Encodings verified against the
// LLVM assembler (disassembly inline).
constexpr uint32_t SimdLdStRegOffEnc(uint8_t size_field,
                                     uint8_t opc,
                                     uint8_t rm,
                                     uint8_t option,
                                     uint8_t s_bit,
                                     uint8_t rn,
                                     uint8_t rt) {
  return (uint32_t{size_field} << 30) | (0b111u << 27) | (1u << 26) | (uint32_t{opc} << 22) |
         (1u << 21) | (uint32_t{rm} << 16) | (uint32_t{option} << 13) | (uint32_t{s_bit} << 12) |
         (0b10u << 10) | (uint32_t{rn} << 5) | uint32_t{rt};
}
// 128-bit (Q): size=00, load opc=11 / store opc=10.
static_assert(SimdLdStRegOffEnc(0b00, 0b11, 21, 0b011, 0, 13, 0) == 0x3cf569a0u);  // ldr q0,[x13,x21]
static_assert(SimdLdStRegOffEnc(0b00, 0b10, 3, 0b011, 0, 0, 0) == 0x3ca36800u);    // str q0,[x0,x3]
static_assert(SimdLdStRegOffEnc(0b00, 0b11, 2, 0b011, 0, 1, 0) == 0x3ce26820u);    // ldr q0,[x1,x2]
static_assert(SimdLdStRegOffEnc(0b00, 0b10, 2, 0b011, 0, 1, 0) == 0x3ca26820u);    // str q0,[x1,x2]
static_assert(SimdLdStRegOffEnc(0b00, 0b11, 2, 0b011, 1, 1, 0) == 0x3ce27820u);    // ldr q0,[x1,x2,lsl#4]
static_assert(SimdLdStRegOffEnc(0b00, 0b11, 2, 0b011, 0, 1, 31) == 0x3ce2683fu);   // ldr q31,[x1,x2]
// 64-bit (D): size=11, load opc=01 / store opc=00.
static_assert(SimdLdStRegOffEnc(0b11, 0b01, 2, 0b011, 0, 1, 0) == 0xfc626820u);    // ldr d0,[x1,x2]
static_assert(SimdLdStRegOffEnc(0b11, 0b00, 2, 0b011, 0, 1, 0) == 0xfc226820u);    // str d0,[x1,x2]
static_assert(SimdLdStRegOffEnc(0b11, 0b01, 6, 0b010, 1, 5, 3) == 0xfc6658a3u);    // ldr d3,[x5,w6,uxtw#3]
// 32-bit (S): size=10, load opc=01 / store opc=00.
static_assert(SimdLdStRegOffEnc(0b10, 0b01, 2, 0b011, 0, 1, 0) == 0xbc626820u);    // ldr s0,[x1,x2]

// ldr q0, [x1, x2]: 128-bit register-offset load; x2=16 selects buf[2..3].
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrQRegOffset) {
  alignas(16) static const uint64_t buf[4] = {0x1111111122222222ULL, 0x3333333344444444ULL,
                                              0x5555555566666666ULL, 0x7777777788888888ULL};
  static const uint32_t code[] = {SimdLdStRegOffEnc(0b00, 0b11, 2, 0b011, 0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 16;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], buf[2]);
  EXPECT_EQ(r[1], buf[3]);
}

// str q0, [x1, x2]: 128-bit register-offset store; x2=16 writes buf[2..3].
TEST_F(Arm64HeavyOptimizerFrontendTest, StrQRegOffset) {
  alignas(16) static uint64_t buf[4] = {0, 0, 0, 0};
  static const uint32_t code[] = {SimdLdStRegOffEnc(0b00, 0b10, 2, 0b011, 0, 1, 0)};
  const uint64_t v[2] = {0xCAFEF00DDEADBEEFULL, 0x0123456789ABCDEFULL};
  std::memcpy(&state_.cpu.v[0], v, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 16;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], 0ULL);  // untouched
  EXPECT_EQ(buf[1], 0ULL);
  EXPECT_EQ(buf[2], v[0]);
  EXPECT_EQ(buf[3], v[1]);
}

// ldr d0, [x1, x2]: 64-bit register-offset load zero-extends the upper 64 of v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrDRegOffsetZeroesUpper) {
  alignas(16) static const uint64_t buf[2] = {0x1122334455667788ULL, 0xdeadbeefdeadbeefULL};
  static const uint32_t code[] = {SimdLdStRegOffEnc(0b11, 0b01, 2, 0b011, 0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], buf[0]);
  EXPECT_EQ(r[1], 0ULL);
}

// str d0, [x1, x2]: 64-bit register-offset store writes only the low 8 bytes.
TEST_F(Arm64HeavyOptimizerFrontendTest, StrDRegOffset) {
  alignas(16) static uint64_t buf[2] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
  static const uint32_t code[] = {SimdLdStRegOffEnc(0b11, 0b00, 2, 0b011, 0, 1, 0)};
  const uint64_t v[2] = {0x0011223344556677ULL, 0x8899AABBCCDDEEFFULL};
  std::memcpy(&state_.cpu.v[0], v, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  EXPECT_EQ(buf[0], v[0]);                       // low 8 written
  EXPECT_EQ(buf[1], 0xFFFFFFFFFFFFFFFFULL);      // high 8 untouched
}

// ldr s0, [x1, x2]: 32-bit register-offset load zero-extends the upper 96 of v0.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrSRegOffsetZeroesUpper) {
  alignas(16) static const uint32_t buf[4] = {0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
  static const uint32_t code[] = {SimdLdStRegOffEnc(0b10, 0b01, 2, 0b011, 0, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], uint64_t{0x11223344u});
  EXPECT_EQ(r[1], 0ULL);
}

// ldr q0, [x1, x2, lsl #4]: the S bit scales x2 by 16, so x2=1 -> addr = x1 + 16.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrQRegOffsetLslShift) {
  alignas(16) static const uint64_t buf[4] = {0x1111111111111111ULL, 0x2222222222222222ULL,
                                              0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL};
  static const uint32_t code[] = {SimdLdStRegOffEnc(0b00, 0b11, 2, 0b011, 1, 1, 0)};
  std::memset(&state_.cpu.v[0], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 1;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[0], 16);
  EXPECT_EQ(r[0], buf[2]);
  EXPECT_EQ(r[1], buf[3]);
}

// ldr d3, [x5, w6, uxtw #3]: UXTW takes only the low 32 bits of x6 and scales by
// 8. x6=0xFFFFFFFF00000002 -> index 2 -> addr = x5 + 16 -> buf[2].
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrDRegOffsetUxtw) {
  alignas(16) static const uint64_t buf[4] = {0xDEAD0000ULL, 0xBEEF1111ULL, 0xF00D2222ULL,
                                              0xCAFE3333ULL};
  static const uint32_t code[] = {SimdLdStRegOffEnc(0b11, 0b01, 6, 0b010, 1, 5, 3)};
  std::memset(&state_.cpu.v[3], 0xAB, 16);
  state_.cpu.x[5] = ToGuestAddr(&buf[0]);
  state_.cpu.x[6] = 0xFFFFFFFF00000002ULL;  // only low 32 (=2) used by UXTW
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[3], 16);
  EXPECT_EQ(r[0], buf[2]);
  EXPECT_EQ(r[1], 0ULL);
}

// ldr q31, [x1, x2]: high vector register.
TEST_F(Arm64HeavyOptimizerFrontendTest, LdrQ31RegOffset) {
  alignas(16) static const uint64_t buf[2] = {0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL};
  static const uint32_t code[] = {SimdLdStRegOffEnc(0b00, 0b11, 2, 0b011, 0, 1, 31)};
  std::memset(&state_.cpu.v[31], 0xAB, 16);
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.x[2] = 0;
  state_.cpu.insn_addr = ToGuestAddr(code);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(code) + sizeof(code)));
  uint64_t r[2];
  std::memcpy(r, &state_.cpu.v[31], 16);
  EXPECT_EQ(r[0], buf[0]);
  EXPECT_EQ(r[1], buf[1]);
}

}  // namespace


// ST2-ST4 / LD2-LD4 single-lane and LD2R-LD4R replicate forms in the heavy
// tier (mirrors the lite lowerings; previously num_regs==1 only). Encodings
// objdump-verified.
TEST_F(Arm64HeavyOptimizerFrontendTest, StNSingleStructLane) {
  auto set_lane32 = [&](int vreg, int lane, uint32_t val) {
    std::memcpy(reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + lane * 4, &val, 4);
  };
  auto set_lane16 = [&](int vreg, int lane, uint16_t val) {
    std::memcpy(reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + lane * 2, &val, 2);
  };
  auto set_lane64 = [&](int vreg, int lane, uint64_t val) {
    std::memcpy(reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + lane * 8, &val, 8);
  };

  // st2 {v4.s-v5.s}[0], [x6]
  alignas(16) uint32_t mem2s[2] = {0, 0};
  set_lane32(4, 0, 0x41424344u);
  set_lane32(5, 0, 0x45464748u);
  state_.cpu.x[6] = ToGuestAddr(&mem2s[0]);
  static const uint32_t st2s[] = {0x0d2080c4U};
  state_.cpu.insn_addr = ToGuestAddr(st2s);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(st2s) + sizeof(st2s)));
  EXPECT_EQ(mem2s[0], 0x41424344u);
  EXPECT_EQ(mem2s[1], 0x45464748u);

  // st4 {v24.h-v27.h}[0], [x8]
  alignas(16) uint16_t mem4h[4] = {0, 0, 0, 0};
  set_lane16(24, 0, 0x1111);
  set_lane16(25, 0, 0x2222);
  set_lane16(26, 0, 0x3333);
  set_lane16(27, 0, 0x4444);
  state_.cpu.x[8] = ToGuestAddr(&mem4h[0]);
  static const uint32_t st4h[] = {0x0d206118U};
  state_.cpu.insn_addr = ToGuestAddr(st4h);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(st4h) + sizeof(st4h)));
  EXPECT_EQ(mem4h[0], 0x1111);
  EXPECT_EQ(mem4h[1], 0x2222);
  EXPECT_EQ(mem4h[2], 0x3333);
  EXPECT_EQ(mem4h[3], 0x4444);

  // st3 {v0.d-v2.d}[0], [x0]
  alignas(16) uint64_t mem3d[3] = {0, 0, 0};
  set_lane64(0, 0, 0x1122334455667788ULL);
  set_lane64(1, 0, 0x99AABBCCDDEEFF00ULL);
  set_lane64(2, 0, 0x0F1E2D3C4B5A6978ULL);
  state_.cpu.x[0] = ToGuestAddr(&mem3d[0]);
  static const uint32_t st3d[] = {0x0d00a400U};
  state_.cpu.insn_addr = ToGuestAddr(st3d);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(st3d) + sizeof(st3d)));
  EXPECT_EQ(mem3d[0], 0x1122334455667788ULL);
  EXPECT_EQ(mem3d[1], 0x99AABBCCDDEEFF00ULL);
  EXPECT_EQ(mem3d[2], 0x0F1E2D3C4B5A6978ULL);

  // st2 {v31.d-v0.d}[0], [x0]: register list wraps 31 -> 0.
  alignas(16) uint64_t memwrap[2] = {0, 0};
  set_lane64(31, 0, 0xAAAAAAAAAAAAAAAAULL);
  set_lane64(0, 0, 0xBBBBBBBBBBBBBBBBULL);
  state_.cpu.x[0] = ToGuestAddr(&memwrap[0]);
  static const uint32_t st2wrap[] = {0x0d20841fU};
  state_.cpu.insn_addr = ToGuestAddr(st2wrap);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(st2wrap) + sizeof(st2wrap)));
  EXPECT_EQ(memwrap[0], 0xAAAAAAAAAAAAAAAAULL);
  EXPECT_EQ(memwrap[1], 0xBBBBBBBBBBBBBBBBULL);

  // st4 {v0.s-v3.s}[0], [x1], #16 (immediate post-index writeback)
  alignas(16) uint32_t mem4s[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) set_lane32(i, 0, 0x10203040u + i);
  state_.cpu.x[1] = ToGuestAddr(&mem4s[0]);
  static const uint32_t st4post[] = {0x0dbfa020U};
  state_.cpu.insn_addr = ToGuestAddr(st4post);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(st4post) + sizeof(st4post)));
  for (int i = 0; i < 4; i++) EXPECT_EQ(mem4s[i], 0x10203040u + i);
  EXPECT_EQ(state_.cpu.x[1], ToGuestAddr(&mem4s[0]) + 16);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, LdNSingleStructLaneAndReplicate) {
  auto lane32 = [&](int vreg, int lane) {
    uint32_t v;
    std::memcpy(&v, reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + lane * 4, 4);
    return v;
  };
  auto lane8 = [&](int vreg, int lane) {
    uint8_t v;
    std::memcpy(&v, reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + lane, 1);
    return v;
  };
  auto upper64 = [&](int vreg) {
    uint64_t v;
    std::memcpy(&v, reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + 8, 8);
    return v;
  };

  // ld2 {v4.s-v5.s}[0], [x6]: lane 0 written, other lanes preserved.
  alignas(16) static const uint32_t mem2s[2] = {0x600D600Du, 0x0DD00DD0u};
  std::memset(&state_.cpu.v[4], 0xFF, 16);
  std::memset(&state_.cpu.v[5], 0xFF, 16);
  state_.cpu.x[6] = ToGuestAddr(&mem2s[0]);
  static const uint32_t ld2s[] = {0x0d6080c4U};
  state_.cpu.insn_addr = ToGuestAddr(ld2s);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ld2s) + sizeof(ld2s)));
  EXPECT_EQ(lane32(4, 0), 0x600D600Du);
  EXPECT_EQ(lane32(5, 0), 0x0DD00DD0u);
  EXPECT_EQ(lane32(4, 1), 0xFFFFFFFFu);
  EXPECT_EQ(lane32(5, 3), 0xFFFFFFFFu);

  // ld4 {v21.b-v24.b}[12], [x2]
  alignas(16) static const uint8_t mem4b[4] = {0x5A, 0x6B, 0x7C, 0x8D};
  for (int i = 21; i <= 24; i++) std::memset(&state_.cpu.v[i], 0, 16);
  state_.cpu.x[2] = ToGuestAddr(&mem4b[0]);
  static const uint32_t ld4b[] = {0x4d603055U};
  state_.cpu.insn_addr = ToGuestAddr(ld4b);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ld4b) + sizeof(ld4b)));
  for (int i = 0; i < 4; i++) EXPECT_EQ(lane8(21 + i, 12), mem4b[i]);

  // ld2r {v0.8b-v1.8b}, [x0]: per-register broadcast, Q=0 zeroes upper half.
  alignas(16) static const uint8_t mem2b[2] = {0xAB, 0xCD};
  std::memset(&state_.cpu.v[0], 0xFF, 16);
  std::memset(&state_.cpu.v[1], 0xFF, 16);
  state_.cpu.x[0] = ToGuestAddr(&mem2b[0]);
  static const uint32_t ld2r[] = {0x0d60c000U};
  state_.cpu.insn_addr = ToGuestAddr(ld2r);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ld2r) + sizeof(ld2r)));
  for (int l = 0; l < 8; l++) {
    EXPECT_EQ(lane8(0, l), 0xAB);
    EXPECT_EQ(lane8(1, l), 0xCD);
  }
  EXPECT_EQ(upper64(0), 0u);
  EXPECT_EQ(upper64(1), 0u);

  // ld4r {v0.4s-v3.4s}, [x0]
  alignas(16) static const uint32_t mem4r[4] = {0x01010101u, 0x02020202u, 0x03030303u,
                                                0x04040404u};
  for (int i = 0; i < 4; i++) std::memset(&state_.cpu.v[i], 0, 16);
  state_.cpu.x[0] = ToGuestAddr(&mem4r[0]);
  static const uint32_t ld4r[] = {0x4d60e800U};
  state_.cpu.insn_addr = ToGuestAddr(ld4r);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(ld4r) + sizeof(ld4r)));
  for (int i = 0; i < 4; i++)
    for (int l = 0; l < 4; l++) EXPECT_EQ(lane32(i, l), mem4r[i]);
}


// 64-bit-lane integer forms previously bailed as "AVX-512-blocked" but SSE-
// emulable: CMEQ #0 (PCMPEQQ), ABS + the CMGT/CMLT #0 family (PCMPGTQ), and
// SSHR .2D via sign-mask PSRLQ/PSLLQ/POR. Encodings objdump-verified.
TEST_F(Arm64HeavyOptimizerFrontendTest, Int64LaneMisc) {
  auto set_lane64 = [&](int vreg, int lane, uint64_t val) {
    std::memcpy(reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + lane * 8, &val, 8);
  };
  auto lane64 = [&](int vreg, int lane) {
    uint64_t v;
    std::memcpy(&v, reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + lane * 8, 8);
    return v;
  };

  // cmeq v3.2d, v3.2d, #0
  set_lane64(3, 0, 0);
  set_lane64(3, 1, 0x123456789ABCDEF0ULL);
  static const uint32_t cmeqz[] = {0x4ee09863U};
  state_.cpu.insn_addr = ToGuestAddr(cmeqz);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(cmeqz) + sizeof(cmeqz)));
  EXPECT_EQ(lane64(3, 0), ~0ULL);
  EXPECT_EQ(lane64(3, 1), 0ULL);

  // abs v2.2d, v0.2d (INT64_MIN fixed point)
  set_lane64(0, 0, static_cast<uint64_t>(-42LL));
  set_lane64(0, 1, 0x8000000000000000ULL);
  static const uint32_t absd[] = {0x4ee0b802U};
  state_.cpu.insn_addr = ToGuestAddr(absd);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(absd) + sizeof(absd)));
  EXPECT_EQ(lane64(2, 0), 42ULL);
  EXPECT_EQ(lane64(2, 1), 0x8000000000000000ULL);

  // cmgt/cmlt v0.2d, #0
  set_lane64(0, 0, 5);
  set_lane64(0, 1, static_cast<uint64_t>(-5LL));
  static const uint32_t cmgtz[] = {0x4ee08800U};
  state_.cpu.insn_addr = ToGuestAddr(cmgtz);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(cmgtz) + sizeof(cmgtz)));
  EXPECT_EQ(lane64(0, 0), ~0ULL);
  EXPECT_EQ(lane64(0, 1), 0ULL);

  set_lane64(0, 0, 5);
  set_lane64(0, 1, static_cast<uint64_t>(-5LL));
  static const uint32_t cmltz[] = {0x4ee0a800U};
  state_.cpu.insn_addr = ToGuestAddr(cmltz);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(cmltz) + sizeof(cmltz)));
  EXPECT_EQ(lane64(0, 0), 0ULL);
  EXPECT_EQ(lane64(0, 1), ~0ULL);

  // sshr v0.2d, #10 and the #64 sign-fill boundary; scalar sshr d0, d1, #1.
  set_lane64(0, 0, static_cast<uint64_t>(-1024LL));
  set_lane64(0, 1, 1024ULL);
  static const uint32_t sshr10[] = {0x4f760400U};
  state_.cpu.insn_addr = ToGuestAddr(sshr10);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(sshr10) + sizeof(sshr10)));
  EXPECT_EQ(lane64(0, 0), static_cast<uint64_t>(-1LL));
  EXPECT_EQ(lane64(0, 1), 1ULL);

  set_lane64(0, 0, static_cast<uint64_t>(-7LL));
  set_lane64(0, 1, 7ULL);
  static const uint32_t sshr64[] = {0x4f400400U};
  state_.cpu.insn_addr = ToGuestAddr(sshr64);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(sshr64) + sizeof(sshr64)));
  EXPECT_EQ(lane64(0, 0), ~0ULL);
  EXPECT_EQ(lane64(0, 1), 0ULL);

  // sshr d0, d1, #1 (scalar-D: upper 64 of Vd zeroed).
  set_lane64(1, 0, static_cast<uint64_t>(-8LL));
  set_lane64(1, 1, 0xDEADBEEFDEADBEEFULL);
  set_lane64(0, 1, 0x1111111111111111ULL);
  static const uint32_t sshrd[] = {0x5f7f0420U};
  state_.cpu.insn_addr = ToGuestAddr(sshrd);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(sshrd) + sizeof(sshrd)));
  EXPECT_EQ(lane64(0, 0), static_cast<uint64_t>(-4LL));
  EXPECT_EQ(lane64(0, 1), 0ULL);
}


// SHA-1 ops in the heavy tier (mirror of the lite lowerings, built on the
// same 32-bit ALU helper the SHA-256 lowerings use). Script-computed goldens;
// encodings objdump-verified.
TEST_F(Arm64HeavyOptimizerFrontendTest, Sha1Ops) {
  auto set_v = [&](int vreg, uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    uint32_t lanes[4] = {l0, l1, l2, l3};
    std::memcpy(&state_.cpu.v[vreg], lanes, 16);
  };
  auto lane32 = [&](int vreg, int lane) {
    uint32_t v;
    std::memcpy(&v, reinterpret_cast<uint8_t*>(&state_.cpu.v[vreg]) + lane * 4, 4);
    return v;
  };
  auto reset = [&] {
    set_v(0, 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u);
    set_v(1, 0xC3D2E1F0u, 0x0BADF00Du, 0, 0);
    set_v(2, 0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
  };

  struct Case { uint32_t insn; uint32_t g[4]; };
  static const Case kCases[] = {
      {0x5e020020U, {0x33E059C9u, 0x1CAEBB0Fu, 0x3DAD9EC0u, 0xD590CC0Au}},  // sha1c
      {0x5e021020U, {0xDC17E052u, 0xD4D79367u, 0x5403F45Eu, 0x89335D8Bu}},  // sha1p
      {0x5e022020U, {0xC5943025u, 0xFD7E55A1u, 0x1DAB79B9u, 0xD590CC0Au}},  // sha1m
      {0x5e023020U, {0xEEEEEEEEu, 0xDDDDDDDDu, 0x685B0E3Du, 0x5FDBE03Fu}},  // sha1su0
  };
  for (const Case& c : kCases) {
    reset();
    uint32_t insn[1] = {c.insn};
    state_.cpu.insn_addr = ToGuestAddr(insn);
    ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(insn) + sizeof(insn)))
        << std::hex << c.insn;
    for (int l = 0; l < 4; l++)
      EXPECT_EQ(lane32(0, l), c.g[l]) << std::hex << c.insn << " lane " << l;
  }

  // sha1su1 v0.4s, v1.4s.
  set_v(0, 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u);
  set_v(1, 0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
  static const uint32_t su1[] = {0x5e281820U};
  state_.cpu.insn_addr = ToGuestAddr(su1);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(su1) + sizeof(su1)));
  EXPECT_EQ(lane32(0, 0), 0x8ACE0246u);
  EXPECT_EQ(lane32(0, 1), 0xB9FD3175u);
  EXPECT_EQ(lane32(0, 2), 0xB9FD3175u);
  EXPECT_EQ(lane32(0, 3), 0x35F8AC61u);

  // sha1h s0, s1.
  set_v(0, 0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu, 0xDDDDDDDDu);
  set_v(1, 0xC3D2E1F0u, 0, 0, 0);
  static const uint32_t h[] = {0x5e280820U};
  state_.cpu.insn_addr = ToGuestAddr(h);
  ASSERT_TRUE(RunOneInstruction(&state_, ToGuestAddr(h) + sizeof(h)));
  EXPECT_EQ(lane32(0, 0), 0x30F4B87Cu);
  EXPECT_EQ(lane32(0, 1), 0u);
  EXPECT_EQ(lane32(0, 2), 0u);
  EXPECT_EQ(lane32(0, 3), 0u);
}

}  // namespace berberis
