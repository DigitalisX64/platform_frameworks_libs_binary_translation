// region digitalis
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

#include "lite_translator.h"

#include "berberis/base/checks.h"
#include "berberis/base/macros.h"
#include "berberis/code_gen_lib/code_gen_lib.h"

namespace berberis {

using Register = LiteTranslator::Register;
using Condition = LiteTranslator::Condition;

//
// Region exit methods.
//

void LiteTranslator::ExitGeneratedCode(GuestAddr target) {
  StoreMappedRegs();
  as_.Movq(as_.rax, target);
  EmitExitGeneratedCode(&as_, as_.rax);
}

void LiteTranslator::ExitRegion(GuestAddr target) {
  StoreMappedRegs();
  if (params_.allow_dispatch) {
    EmitDirectDispatch(&as_, target, /* check_pending_signals */ true);
  } else {
    as_.Movq(as_.rax, target);
    EmitExitGeneratedCode(&as_, as_.rax);
  }
}

void LiteTranslator::ExitRegionIndirect(Register target) {
  StoreMappedRegs();
  if (params_.allow_dispatch) {
    EmitIndirectDispatch(&as_, target);
  } else {
    EmitExitGeneratedCode(&as_, target);
  }
}

//
// BranchCond: evaluate ARM64 condition code from stored NZCV flags.
//
// ARM64 NZCV flags are stored in ThreadState.cpu.flags as a uint16_t.
// On x86_64 host, the bit positions are:
//   N = bit 15 (CPUState::kFlagNegative)
//   Z = bit 14 (CPUState::kFlagZero)
//   C = bit 8  (CPUState::kFlagCarry)
//   V = bit 0  (CPUState::kFlagOverflow)
//
void LiteTranslator::BranchCond(Decoder::Condition cond, int32_t offset) {
  // Load flags from ThreadState.
  int32_t flags_offset = offsetof(ThreadState, cpu.flags);
  Register flags_reg = AllocTempReg();
  as_.Movzxwl(flags_reg, {.base = Assembler::rbp, .disp = flags_offset});

  Assembler::Label* cont = as_.MakeLabel();

  // Evaluate condition and jump to continuation if NOT taken.
  // We emit the inverse check: jump over the branch-taken path.
  switch (cond) {
    case Decoder::Condition::kEq:
      // EQ: Z==1. Test bit 14.
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kNotCarry, *cont);  // skip if Z==0
      break;
    case Decoder::Condition::kNe:
      // NE: Z==0. Test bit 14.
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kCarry, *cont);  // skip if Z==1
      break;
    case Decoder::Condition::kCs:
      // CS/HS: C==1. Test bit 8.
      as_.Btl(flags_reg, static_cast<int8_t>(8));
      as_.Jcc(Condition::kNotCarry, *cont);
      break;
    case Decoder::Condition::kCc:
      // CC/LO: C==0. Test bit 8.
      as_.Btl(flags_reg, static_cast<int8_t>(8));
      as_.Jcc(Condition::kCarry, *cont);
      break;
    case Decoder::Condition::kMi:
      // MI: N==1. Test bit 15.
      as_.Btl(flags_reg, static_cast<int8_t>(15));
      as_.Jcc(Condition::kNotCarry, *cont);
      break;
    case Decoder::Condition::kPl:
      // PL: N==0. Test bit 15.
      as_.Btl(flags_reg, static_cast<int8_t>(15));
      as_.Jcc(Condition::kCarry, *cont);
      break;
    case Decoder::Condition::kVs:
      // VS: V==1. Test bit 0.
      as_.Btl(flags_reg, static_cast<int8_t>(0));
      as_.Jcc(Condition::kNotCarry, *cont);
      break;
    case Decoder::Condition::kVc:
      // VC: V==0. Test bit 0.
      as_.Btl(flags_reg, static_cast<int8_t>(0));
      as_.Jcc(Condition::kCarry, *cont);
      break;
    case Decoder::Condition::kHi: {
      // HI: C==1 && Z==0. Test C (bit 8) and Z (bit 14).
      as_.Btl(flags_reg, static_cast<int8_t>(8));
      as_.Jcc(Condition::kNotCarry, *cont);  // skip if C==0
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kCarry, *cont);     // skip if Z==1
      break;
    }
    case Decoder::Condition::kLs: {
      // LS: C==0 || Z==1.
      // Branch taken if C==0 OR Z==1. Skip (to cont) only if C==1 AND Z==0.
      Assembler::Label* taken = as_.MakeLabel();
      as_.Btl(flags_reg, static_cast<int8_t>(8));
      as_.Jcc(Condition::kNotCarry, *taken);  // C==0 -> take branch
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kNotCarry, *cont);   // C==1, Z==0 -> skip
      as_.Bind(taken);
      break;
    }
    case Decoder::Condition::kGe: {
      // GE: N==V. Extract N (bit 15) and V (bit 0), compare.
      Register tmp = AllocTempReg();
      // Get N bit into bit 0 of tmp.
      as_.Movl(tmp, flags_reg);
      as_.Shrl(tmp, static_cast<int8_t>(15));  // N is now in bit 0
      // XOR with V (bit 0 of flags_reg) -- if same, result bit 0 is 0.
      as_.Xorl(tmp, flags_reg);
      as_.Btl(tmp, static_cast<int8_t>(0));
      as_.Jcc(Condition::kCarry, *cont);  // skip if N!=V
      break;
    }
    case Decoder::Condition::kLt: {
      // LT: N!=V.
      Register tmp = AllocTempReg();
      as_.Movl(tmp, flags_reg);
      as_.Shrl(tmp, static_cast<int8_t>(15));
      as_.Xorl(tmp, flags_reg);
      as_.Btl(tmp, static_cast<int8_t>(0));
      as_.Jcc(Condition::kNotCarry, *cont);  // skip if N==V
      break;
    }
    case Decoder::Condition::kGt: {
      // GT: Z==0 && N==V.
      // Skip if Z==1 or N!=V.
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kCarry, *cont);  // skip if Z==1
      Register tmp = AllocTempReg();
      as_.Movl(tmp, flags_reg);
      as_.Shrl(tmp, static_cast<int8_t>(15));
      as_.Xorl(tmp, flags_reg);
      as_.Btl(tmp, static_cast<int8_t>(0));
      as_.Jcc(Condition::kCarry, *cont);  // skip if N!=V
      break;
    }
    case Decoder::Condition::kLe: {
      // LE: Z==1 || N!=V.
      // Branch taken if Z==1 OR N!=V. Skip only if Z==0 AND N==V.
      Assembler::Label* taken = as_.MakeLabel();
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kCarry, *taken);  // Z==1 -> take branch
      Register tmp = AllocTempReg();
      as_.Movl(tmp, flags_reg);
      as_.Shrl(tmp, static_cast<int8_t>(15));
      as_.Xorl(tmp, flags_reg);
      as_.Btl(tmp, static_cast<int8_t>(0));
      as_.Jcc(Condition::kNotCarry, *cont);  // Z==0, N==V -> skip
      as_.Bind(taken);
      break;
    }
    case Decoder::Condition::kAl:
    case Decoder::Condition::kNv:
      // AL/NV: always taken (NV behaves like AL).
      break;
  }

  // region digitalis - reverted region extension (was causing linker hang)
  is_region_end_reached_ = true;
  GuestAddr target = GetInsnAddr() + offset;
  ExitRegion(target);
  // endregion
  as_.Bind(cont);
}

//
// CCMP/CCMN: Conditional Compare.
// If condition is true: compare rn with rm (CMP or CMN) and set NZCV flags.
// If condition is false: set NZCV flags to the immediate nzcv value.
//
void LiteTranslator::ConditionalCompare(bool is_neg, bool is_64bit, Register rn, Register rm,
                                         Decoder::Condition cond, uint8_t nzcv) {
  // We need to evaluate the condition from the current flags, then either
  // do a CMP/CMN (updating flags) or set flags to the nzcv immediate.

  int32_t flags_offset = offsetof(ThreadState, cpu.flags);

  // Load current flags for condition evaluation.
  Register flags_reg = AllocTempReg();
  as_.Movzxwl(flags_reg, {.base = Assembler::rbp, .disp = flags_offset});

  Assembler::Label* cond_false = as_.MakeLabel();
  Assembler::Label* done = as_.MakeLabel();

  // Evaluate condition — jump to cond_false if NOT met.
  // Reuse the same condition evaluation logic as BranchCond.
  switch (cond) {
    case Decoder::Condition::kEq:
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kNotCarry, *cond_false);
      break;
    case Decoder::Condition::kNe:
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kCarry, *cond_false);
      break;
    case Decoder::Condition::kCs:
      as_.Btl(flags_reg, static_cast<int8_t>(8));
      as_.Jcc(Condition::kNotCarry, *cond_false);
      break;
    case Decoder::Condition::kCc:
      as_.Btl(flags_reg, static_cast<int8_t>(8));
      as_.Jcc(Condition::kCarry, *cond_false);
      break;
    case Decoder::Condition::kMi:
      as_.Btl(flags_reg, static_cast<int8_t>(15));
      as_.Jcc(Condition::kNotCarry, *cond_false);
      break;
    case Decoder::Condition::kPl:
      as_.Btl(flags_reg, static_cast<int8_t>(15));
      as_.Jcc(Condition::kCarry, *cond_false);
      break;
    case Decoder::Condition::kVs:
      as_.Btl(flags_reg, static_cast<int8_t>(0));
      as_.Jcc(Condition::kNotCarry, *cond_false);
      break;
    case Decoder::Condition::kVc:
      as_.Btl(flags_reg, static_cast<int8_t>(0));
      as_.Jcc(Condition::kCarry, *cond_false);
      break;
    case Decoder::Condition::kHi:
      as_.Btl(flags_reg, static_cast<int8_t>(8));
      as_.Jcc(Condition::kNotCarry, *cond_false);
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kCarry, *cond_false);
      break;
    case Decoder::Condition::kLs: {
      Assembler::Label* cond_true = as_.MakeLabel();
      as_.Btl(flags_reg, static_cast<int8_t>(8));
      as_.Jcc(Condition::kNotCarry, *cond_true);
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kNotCarry, *cond_false);
      as_.Bind(cond_true);
      break;
    }
    case Decoder::Condition::kGe: {
      Register tmp = AllocTempReg();
      as_.Movl(tmp, flags_reg);
      as_.Shrl(tmp, static_cast<int8_t>(15));
      as_.Xorl(tmp, flags_reg);
      as_.Btl(tmp, static_cast<int8_t>(0));
      as_.Jcc(Condition::kCarry, *cond_false);
      break;
    }
    case Decoder::Condition::kLt: {
      Register tmp = AllocTempReg();
      as_.Movl(tmp, flags_reg);
      as_.Shrl(tmp, static_cast<int8_t>(15));
      as_.Xorl(tmp, flags_reg);
      as_.Btl(tmp, static_cast<int8_t>(0));
      as_.Jcc(Condition::kNotCarry, *cond_false);
      break;
    }
    case Decoder::Condition::kGt:
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kCarry, *cond_false);
      {
        Register tmp = AllocTempReg();
        as_.Movl(tmp, flags_reg);
        as_.Shrl(tmp, static_cast<int8_t>(15));
        as_.Xorl(tmp, flags_reg);
        as_.Btl(tmp, static_cast<int8_t>(0));
        as_.Jcc(Condition::kCarry, *cond_false);
      }
      break;
    case Decoder::Condition::kLe: {
      Assembler::Label* cond_true = as_.MakeLabel();
      as_.Btl(flags_reg, static_cast<int8_t>(14));
      as_.Jcc(Condition::kCarry, *cond_true);
      Register tmp = AllocTempReg();
      as_.Movl(tmp, flags_reg);
      as_.Shrl(tmp, static_cast<int8_t>(15));
      as_.Xorl(tmp, flags_reg);
      as_.Btl(tmp, static_cast<int8_t>(0));
      as_.Jcc(Condition::kNotCarry, *cond_false);
      as_.Bind(cond_true);
      break;
    }
    case Decoder::Condition::kAl:
    case Decoder::Condition::kNv:
      // Always true — just do the compare.
      break;
  }

  // Condition is true: perform CMP (is_neg=false) or CMN (is_neg=true).
  if (is_64bit) {
    if (is_neg) {
      as_.Addq(rn, rm);  // CMN: add and check flags
    } else {
      as_.Cmpq(rn, rm);  // CMP: subtract and check flags
    }
  } else {
    if (is_neg) {
      as_.Addl(rn, rm);
    } else {
      as_.Cmpl(rn, rm);
    }
  }
  // Store the resulting x86 flags as ARM64 NZCV.
  EmitStoreArmNZCV(/*is_sub=*/!is_neg);
  as_.Jmp(*done);

  // Condition is false: set NZCV to the immediate value.
  as_.Bind(cond_false);
  {
    // ARM64 nzcv immediate: bit3=N, bit2=Z, bit1=C, bit0=V
    // Map to x86_64 flag positions: N=bit15, Z=bit14, C=bit8, V=bit0
    uint16_t flags_val = 0;
    if (nzcv & 0x8) flags_val |= (1 << 15);  // N
    if (nzcv & 0x4) flags_val |= (1 << 14);  // Z
    if (nzcv & 0x2) flags_val |= (1 << 8);   // C
    if (nzcv & 0x1) flags_val |= (1 << 0);   // V
    Register imm_reg = AllocTempReg();
    as_.Movl(imm_reg, static_cast<int32_t>(flags_val));
    as_.Movw({.base = Assembler::rbp, .disp = flags_offset}, imm_reg);
  }
  as_.Bind(done);
}

}  // namespace berberis
// endregion
