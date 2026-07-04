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

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/heavy_optimizer/arm64/heavy_optimize_region.h"
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
  // MOVZ X0,#10; SUBS X1,X0,#3; CRC32B W0,W1,W0 (bails: DataProc2Src is not
  // translated by the optimizing frontend). ADC is now translated, so it is no
  // longer a valid bail sentinel here.
  static const uint32_t code[] = {MovzX(0, 10), SubsImmX(1, 0, 3), 0x1AC04020u /*CRC32B W0,W1,W0*/};
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
  // CRC32 (DataProc2Src) is not translated by the optimizing frontend.
  static const uint32_t code[] = {0x1AC04020};  // CRC32B W0, W1, W0
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
// FMAX Sd,Sn,Sm (opcode=0100) — must bail in the optimizing tier.
constexpr uint32_t FmaxS(uint8_t rd, uint8_t rn, uint8_t rm) { return FpDP2(0x1E204800, rd, rn, rm); }

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
// FSQRT Sd,Sn (opcode=000011) — must bail in the optimizing tier.
constexpr uint32_t FsqrtS(uint8_t rd, uint8_t rn) { return FpDP1(0x1E21C000, rd, rn); }

// FMOV (scalar, immediate): 0001_1110_ftype_1_imm8_100_00000_Rd.
constexpr uint32_t FmovImmS(uint8_t rd, uint8_t imm8) {
  return 0x1E201000 | (static_cast<uint32_t>(imm8) << 13) | rd;
}
constexpr uint32_t FmovImmD(uint8_t rd, uint8_t imm8) {
  return 0x1E601000 | (static_cast<uint32_t>(imm8) << 13) | rd;
}

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
// CMEQ (vector, register): U=1, opcode=10001.
constexpr uint32_t CmeqVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/true, size, /*opcode=*/0b10001, rd, rn, rm);
}
// SQADD (vector, saturating): U=0, opcode=00001 — must bail.
constexpr uint32_t SqaddVec(uint8_t size, bool q, uint8_t rd, uint8_t rn, uint8_t rm) {
  return AdvSimdThreeSame(q, /*u=*/false, size, /*opcode=*/0b00001, rd, rn, rm);
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

// SCVTF / UCVTF (integer -> scalar FP). Expected values are computed with the
// host's own static_cast so each test cross-checks the heavy lowering against
// the native conversion, and the upper V[] bytes are asserted zeroed.

TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfSFromW) {
  static const uint32_t code[] = {0x1e220020u};  // scvtf s0, w1
  state_.cpu.x[1] = 0xFFFFFFF9ULL;               // W1 = -7 (signed)
  SetV128(&state_, 0, 0xDEADBEEFCAFEF00DULL, 0x0123456789ABCDEFULL);
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_FLOAT_EQ(GetVf32(&state_, 0), static_cast<float>(int32_t{-7}));
  EXPECT_EQ(VWord1(&state_, 0), 0u);
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, ScvtfDFromX) {
  static const uint32_t code[] = {0x9e620020u};  // scvtf d0, x1
  state_.cpu.x[1] = static_cast<uint64_t>(int64_t{-123456789});
  GuestAddr end_pc = ToGuestAddr(code) + sizeof(code);
  bool ok = false;
  RunRegion(&state_, code, end_pc, &ok);
  ASSERT_TRUE(ok);
  EXPECT_DOUBLE_EQ(GetVf64(&state_, 0), static_cast<double>(int64_t{-123456789}));
  EXPECT_EQ(VUpperHi64(&state_, 0), 0u);
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
TEST_F(Arm64HeavyOptimizerFrontendTest, FmaxBails) {
  static const uint32_t code[] = {FmaxS(0, 1, 2)};
  SetVf32(&state_, 1, 1.0f);
  SetVf32(&state_, 2, 2.0f);
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// FSQRT (a 1-source FP op the optimizing tier does NOT handle) must bail.
TEST_F(Arm64HeavyOptimizerFrontendTest, FsqrtBails) {
  static const uint32_t code[] = {FsqrtS(0, 1)};
  SetVf32(&state_, 1, 4.0f);
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
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
// FCMP Hn, Hm (ftype=0b11): FP16 must bail to the lite tier.
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

// FP16 FCMP must bail to the lite tier (no F16C widening ops in the heavy tier).
TEST_F(Arm64HeavyOptimizerFrontendTest, FcmpHBails) {
  static const uint32_t code[] = {FcmpH(1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
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
TEST_F(Arm64HeavyOptimizerFrontendTest, CmeqVec2DBails) {
  static const uint32_t code[] = {CmeqVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
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

// ADD .2D (64-bit elements) must bail: Paddq is not in the backend allowlist.
TEST_F(Arm64HeavyOptimizerFrontendTest, AddVec2DBails) {
  static const uint32_t code[] = {AddVec(0b11, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SUB .16B (byte elements) must bail: Psubb is not in the backend allowlist.
TEST_F(Arm64HeavyOptimizerFrontendTest, SubVec16BBails) {
  static const uint32_t code[] = {SubVec(0b00, /*q=*/true, 0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

// SQADD (a saturating three-same op) must bail to the lite translator.
TEST_F(Arm64HeavyOptimizerFrontendTest, SqaddVecBails) {
  static const uint32_t code[] = {SqaddVec(0b10, /*q=*/true, 0, 1, 2)};
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
TEST_F(Arm64HeavyOptimizerFrontendTest, DupElementBails) {
  static const uint32_t code[] = {DupElem(/*q=*/true, /*imm5=*/0x08, 0, 1)};  // dup v0.2d, v1.d[0]
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

// A two-reg-misc opcode we do NOT yet mirror (CLZ) must still bail cleanly.
TEST_F(Arm64HeavyOptimizerFrontendTest, ClzVecBails) {
  // clz v0.4s, v1.4s : U=0, opcode=00100, size=10, Q=1.
  static const uint32_t code[] = {AdvSimdTwoRegMisc(/*q=*/true, /*u=*/false, /*size=*/0b10,
                                                    /*opcode=*/0b00100, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 0u);
}

}  // namespace

}  // namespace berberis
