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
// UDIV Xd, Xn, Xm (must bail).
constexpr uint32_t UdivX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9AC00800 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
}
// SMULH Xd, Xn, Xm (must bail).
constexpr uint32_t SmulhX(uint8_t rd, uint8_t rn, uint8_t rm) {
  return 0x9B407C00 | (static_cast<uint32_t>(rm) << 16) | (rn << 5) | rd;
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
// REV Xd, Xn (DP-1Src) — must bail (no x86 BSWAP MachineIR op).
constexpr uint32_t RevX(uint8_t rd, uint8_t rn) {
  return 0xDAC00C00 | (rn << 5) | rd;
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
  // SMULH (DataProc3Src high-multiply) still bails after the two MoveWides.
  static const uint32_t code[] = {MovzX(0, 0x11), MovzX(1, 0x22), SmulhX(2, 0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // 2 MoveWide translate, the SMULH bails -> partial region.
  EXPECT_EQ(n, 2u);
}

// A single guest instruction whose decode fires several listener callbacks where
// a later one bails must still produce valid IR (one region exit, no trailing
// insns) — i.e. GenCode must not abort. A SIMD LDP-Q post-index decodes to
// SimdLoadStorePair (still bails this round) followed by an AddImm writeback;
// once SimdLoadStorePair sets success_ = false the AddImm must emit nothing.
TEST_F(Arm64HeavyOptimizerFrontendTest, MultiCallbackBailIsValidIR) {
  // LDP Q1, Q2, [X0], #32 (SIMD post-index): SimdLoadStorePair (bails) then the
  // AddImm writeback.
  static const uint32_t code[] = {0xACC10801};
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
  static const uint32_t code[] = {MovzX(0, 0x11), 0xACC10801 /*SIMD LDP-Q post-index, bails*/};
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
TEST_F(Arm64HeavyOptimizerFrontendTest, AdcBails) {
  // ADC X0, X1, X2 -> 0x9A000000 base | (rm<<16) | (rn<<5) | rd.
  static const uint32_t code[] = {0x9A020020u};  // ADC X0, X1, X2
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
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

TEST_F(Arm64HeavyOptimizerFrontendTest, SmulhBails) {
  static const uint32_t code[] = {SmulhX(0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
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

TEST_F(Arm64HeavyOptimizerFrontendTest, UdivBails) {
  static const uint32_t code[] = {UdivX(0, 1, 2)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
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

TEST_F(Arm64HeavyOptimizerFrontendTest, Extr64Bails) {
  // EXTR X0, X1, X2, #4 (64-bit non-zero lsb) bails (no 64-bit SHRD MachineIR op).
  static const uint32_t code[] = {ExtrX(0, 1, 2, 4)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
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

TEST_F(Arm64HeavyOptimizerFrontendTest, RevBails) {
  // REV needs x86 BSWAP, which has no MachineIR op here -> bails.
  static const uint32_t code[] = {RevX(0, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  GuestAddr stop_pc = ToGuestAddr(code) + sizeof(code);
  EXPECT_FALSE(RunOneInstruction(&state_, stop_pc));
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
  // MOVZ; ADD; SUB; then a bailing UDIV ends the region after 3 translated.
  static const uint32_t code[] = {
      MovzX(0, 0x10), AddImmX(1, 0, 4), SubImmX(2, 1, 2), UdivX(3, 2, 1)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  EXPECT_EQ(n, 3u);
}

TEST_F(Arm64HeavyOptimizerFrontendTest, MultiAluMixedShiftRegion) {
  // A mix of shifted-register, extended-register, logical-immediate and a
  // multiply, then a bailing SMULH. Exercises CheckMachineIR over the whole run.
  static const uint32_t code[] = {MovzX(0, 0x7),
                                  MovzX(1, 0x3),
                                  AddRegX(2, 0, 1),
                                  SubRegLsl(3, 2, 1, 2),
                                  AndImmX(4, 3, 0, 7),
                                  MaddX(5, 0, 1, 4),
                                  SmulhX(6, 5, 0)};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // 6 translate, the SMULH bails.
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
  // MOVZ X0,#10; SUBS X1,X0,#3; ADC X2,X1,X0 (bails).
  static const uint32_t code[] = {MovzX(0, 10), SubsImmX(1, 0, 3), 0x9A000022u /*ADC X2,X1,X0*/};
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // MOVZ + SUBS translate, the ADC bails -> partial region of 2.
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
// a bail (SMULH). The load + add translate; the high-multiply bails.
TEST_F(Arm64HeavyOptimizerFrontendTest, LoadAddThenBailRegion) {
  static uint64_t buf[1] = {0x1000};
  static const uint32_t code[] = {
      LdrXuoff(0, 1, 0),   // [0] X0 = [X1]
      AddImmX(2, 0, 0x24),  // [1] X2 = X0 + 0x24
      SmulhX(3, 0, 2),     // [2] SMULH bails
  };
  state_.cpu.x[1] = ToGuestAddr(&buf[0]);
  state_.cpu.insn_addr = ToGuestAddr(code);
  MachineCode mc;
  auto [stop, ok, n] = HeavyOptimizeRegion(
      ToGuestAddr(code), &mc, HeavyOptimizeParams{.end_pc = ToGuestAddr(code) + sizeof(code)});
  // LDR + ADD translate; SMULH bails -> partial region of 2 instructions.
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

}  // namespace

}  // namespace berberis
