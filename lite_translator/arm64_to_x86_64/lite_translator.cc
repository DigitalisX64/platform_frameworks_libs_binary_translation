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

#include <cstddef>

#include "berberis/base/checks.h"
#include "berberis/base/macros.h"
#include "berberis/code_gen_lib/code_gen_lib.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

using Register = LiteTranslator::Register;
using Condition = LiteTranslator::Condition;

//
// Region exit methods.
//

// Capture host MXCSR cumulative exception bits and OR-mirror them into
// emulated_fpsr at ARM FPSR bit positions, before exiting the JIT region.
//
// Why at region exit, not per FP op: the System V x86_64 ABI treats MXCSR
// exception bits 0-5 as caller-saved status flags, so the C++ runtime in the
// dispatch path (berberis_HandleInterpret, InterpretBatch, ...) is permitted
// to clobber them. A lazy mirror in the interpreter MRS-FPSR handler reads
// MXCSR after that C++ has already run and may have wiped the bits set by
// JIT-emitted DIVSS/MULSS/SQRTSS/etc. Capturing here, while still in JIT
// context, preserves the cumulative state for the interpreter to read out
// of emulated_fpsr.
//
//   MXCSR[0] IE (invalid)   -> FPSR[0] IOC
//   MXCSR[1] DE (denormal)  -> FPSR[7] IDC
//   MXCSR[2] ZE (div-zero)  -> FPSR[1] DZC
//   MXCSR[3] OE (overflow)  -> FPSR[2] OFC
//   MXCSR[4] UE (underflow) -> FPSR[3] UFC
//   MXCSR[5] PE (inexact)   -> FPSR[4] IXC
//
// (ARM ARM C5.2.8 / Intel SDM 11.6.6.)
//
// The translated code below uses rax/rcx/rdx as scratches (caller-saved at the
// region boundary) and a 4-byte slot at [rsp] inside the JIT region's
// kFrameSizeAtTranslatedCode-byte frame (already reserved by
// berberis_RunGeneratedCode).
void LiteTranslator::EmitMxcsrToFpsrMirror() {
  // skip mirror entirely if no SIMD/FP op was emitted in
  // the region: MXCSR cannot have been dirtied by this region, so the mirror
  // would only OR in already-stale bits (which are no-ops) at significant
  // host cost. Integer-only hot loops like CdEntryMapZip32::AddToMap probe
  // hit 9+ region exits per iteration; this elides the ~12-insn mirror per
  // exit. Correctness: any earlier region's MXCSR bits were already mirrored
  // into emulated_fpsr at *its* exit, and the dispatch-loop C++ between
  // regions can only clear MXCSR exception bits (caller-saved per SysV
  // x86_64 ABI 3.2.1), not set them.
  if (!fp_dirty_) {
    return;
  }
  // stmxcsr [rsp]
  as_.Stmxcsr({.base = as_.rsp, .disp = 0});
  // eax = MXCSR & 0x3F  (isolate exception bits 0-5)
  as_.Movzxbl(as_.rax, {.base = as_.rsp, .disp = 0});
  as_.Andl(as_.rax, int32_t{0x3F});
  // edx = IE (bit 0 -> IOC bit 0)
  as_.Movl(as_.rdx, as_.rax);
  as_.Andl(as_.rdx, int32_t{0x01});
  // ecx = (eax >> 1) & 0x1E  -- {ZE,OE,UE,PE} bits 2..5 -> {DZC,OFC,UFC,IXC} bits 1..4
  as_.Movl(as_.rcx, as_.rax);
  as_.Shrl(as_.rcx, int8_t{1});
  as_.Andl(as_.rcx, int32_t{0x1E});
  as_.Orl(as_.rdx, as_.rcx);
  // ecx = (eax << 6) & 0x80  -- DE bit 1 -> IDC bit 7
  as_.Movl(as_.rcx, as_.rax);
  as_.Shll(as_.rcx, int8_t{6});
  as_.Andl(as_.rcx, int32_t{0x80});
  as_.Orl(as_.rdx, as_.rcx);
  // or [rbp + offsetof(cpu.emulated_fpsr)], edx
  int32_t fpsr_off = offsetof(ThreadState, cpu.emulated_fpsr);
  as_.Orl({.base = as_.rbp, .disp = fpsr_off}, as_.rdx);
}

void LiteTranslator::ExitGeneratedCode(GuestAddr target) {
  StoreMappedRegs();
  EmitMxcsrToFpsrMirror();
  as_.Movq(as_.rax, target);
  EmitExitGeneratedCode(&as_, as_.rax);
}

void LiteTranslator::Svc(uint16_t imm) {
  UNUSED(imm);
  // Translate the syscall inline rather than bailing to the interpreter. A
  // success_=false bail would end the region BEFORE the SVC, mark the SVC PC
  // kInterpreted, and route every syscall through a dispatcher round-trip plus
  // an interpreter batch — costly on the hot path since real apps issue
  // syscalls constantly (futex, ioctl, binder, IO).
  //
  // RunGuestSyscall reads and writes the guest register file in ThreadState
  // directly (x8 = number, x0-x5 = args, x0 = result) and may run a guest
  // signal handler, so flush the mapped guest registers and mirror the FP
  // status first, exactly as a region exit does. EmitSyscall then stamps
  // insn_addr, calls RunGuestSyscall, and direct-dispatches to pc+4, so this
  // ends the region (like Branch). It does NOT exit to the SVC's own PC, which
  // is what made an earlier attempt loop forever.
  StoreMappedRegs();
  EmitMxcsrToFpsrMirror();
  is_region_end_reached_ = true;
  EmitSyscall(&as_, GetInsnAddr());
}

void LiteTranslator::ExitRegion(GuestAddr target) {
  StoreMappedRegs();
  EmitMxcsrToFpsrMirror();
  if (params_.allow_dispatch) {
    EmitDirectDispatch(&as_, target, /* check_pending_signals */ true);
  } else {
    as_.Movq(as_.rax, target);
    EmitExitGeneratedCode(&as_, as_.rax);
  }
}

void LiteTranslator::ExitRegionIndirect(Register target) {
  StoreMappedRegs();
  // skip the mirror-spill scaffolding entirely when no
  // FP/SIMD work happened in this region (see EmitMxcsrToFpsrMirror comment).
  if (!fp_dirty_) {
    // Move target to rax directly; no spill needed.
    if (target != as_.rax) {
      as_.Movq(as_.rax, target);
    }
  } else {
    // Spill target across the mirror because EmitMxcsrToFpsrMirror clobbers
    // rax/rcx/rdx, and target may live in rcx or rdx (both in the allocator
    // pool). Allocate an extra 16-byte slot below rsp: lower 4 bytes for the
    // mirror's stmxcsr scratch (referenced via [rsp+0]), upper 8 bytes for
    // the spilled target. Restore rax with target's value, then restore rsp.
    as_.Subq(as_.rsp, int32_t{16});
    as_.Movq({.base = as_.rsp, .disp = 8}, target);
    EmitMxcsrToFpsrMirror();
    as_.Movq(as_.rax, {.base = as_.rsp, .disp = 8});
    as_.Addq(as_.rsp, int32_t{16});
  }
  if (params_.allow_dispatch) {
    EmitIndirectDispatch(&as_, as_.rax);
  } else {
    EmitExitGeneratedCode(&as_, as_.rax);
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
  Assembler::Label* cont = as_.MakeLabel();

  // Jump over the branch-taken path (to cont) when the condition is not
  // satisfied. EmitJumpIfCondNotMet tests cpu.flags in memory directly.
  EmitJumpIfCondNotMet(cond, *cont);

  // forward branch extension with back-edge detection
  GuestAddr target = GetInsnAddr() + offset;
  if (offset <= 0) {
    // Backward branch (or self-loop): end region to prevent infinite loops.
    is_region_end_reached_ = true;
  }
  // Taken path: always exit to branch target.
  ExitRegion(target);
  // Fall-through: continue translating if forward branch (is_region_end_reached_ not set).
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

  Assembler::Label* cond_false = as_.MakeLabel();
  Assembler::Label* done = as_.MakeLabel();

  // Evaluate condition — jump to cond_false if NOT met. Tests cpu.flags in
  // memory directly (shared with BranchCond / CSEL); flags_offset is still
  // used below to write the immediate-nzcv result on the false path.
  EmitJumpIfCondNotMet(cond, *cond_false);

  // Condition is true: perform CMP (is_neg=false) or CMN (is_neg=true).
  // CMN is non-destructive (it only sets NZCV from rn+rm), but x86 has no
  // non-destructive add, so the sum must go to a scratch register — never into
  // rn. With register mapping enabled, rn is the live x86 register for the
  // guest source register, so adding into it would silently corrupt that guest
  // register for the rest of the JIT region (CMP's Cmpq is genuinely
  // non-destructive, so only the CMN path is affected). ADD sets the same
  // EFLAGS regardless of where the result lands, so the scratch copy preserves
  // the flag semantics.
  if (is_64bit) {
    if (is_neg) {
      Register tmp_cmn = AllocTempReg();
      as_.Movq(tmp_cmn, rn);
      as_.Addq(tmp_cmn, rm);  // CMN: add into scratch, only flags matter
    } else {
      as_.Cmpq(rn, rm);  // CMP: subtract and check flags
    }
  } else {
    if (is_neg) {
      Register tmp_cmn = AllocTempReg();
      as_.Movl(tmp_cmn, rn);
      as_.Addl(tmp_cmn, rm);
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
    if (nzcv & 0x8) flags_val |= CPUState::kFlagNegative;
    if (nzcv & 0x4) flags_val |= CPUState::kFlagZero;
    if (nzcv & 0x2) flags_val |= CPUState::kFlagCarry;
    if (nzcv & 0x1) flags_val |= CPUState::kFlagOverflow;
    Register imm_reg = AllocTempReg();
    as_.Movl(imm_reg, static_cast<int32_t>(flags_val));
    as_.Movw({.base = Assembler::rbp, .disp = flags_offset}, imm_reg);
  }
  as_.Bind(done);
}

}  // namespace berberis
