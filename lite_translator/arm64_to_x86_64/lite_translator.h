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

#ifndef BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_H_
#define BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_H_

#include <cstdint>
#include <tuple>
#include <unordered_map>

#include "berberis/assembler/common.h"
#include "berberis/assembler/x86_64.h"
#include "berberis/base/checks.h"
#include "berberis/base/macros.h"
#include "berberis/decoder/arm64/decoder.h"
#include "berberis/decoder/arm64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/platform.h"

#include "allocator.h"
#include "register_maintainer.h"

namespace berberis {

class LiteTranslator {
 public:
  // Use x86_64::Assembler directly since there is no ARM64-specific MacroAssembler yet.
  using Assembler = x86_64::Assembler;
  using Decoder = Decoder<SemanticsPlayer<LiteTranslator>>;
  using Register = Assembler::Register;
  static constexpr auto no_register = Assembler::no_register;
  using SimdRegister = Assembler::XMMRegister;
  static constexpr auto no_simd_register = Assembler::no_xmm_register;
  using Condition = Assembler::Condition;

  explicit LiteTranslator(MachineCode* machine_code,
                          GuestAddr pc,
                          LiteTranslateParams params = LiteTranslateParams{})
      : as_(machine_code),
        success_(true),
        pc_(pc),
        params_(params),
        is_region_end_reached_(false),
        // track whether any SIMD/FP register was allocated.
        // If not, this region can't have dirtied MXCSR exception bits, so the
        // MXCSR -> FPSR mirror at region exit can be elided. Integer-only hot
        // loops (e.g. bionic linker CdEntryMapZip32::AddToMap hash probe) hit
        // 9+ region exits per iteration; eliding the 12-x86-insn mirror saves
        // ~100 host insns per iteration there.
        fp_dirty_(false)
        {}

  //
  // Guest state getters/setters.
  //

  GuestAddr GetInsnAddr() const { return pc_; }

  void IncrementInsnAddr(uint8_t insn_size) { pc_ += insn_size; }

  Register GetReg(uint8_t reg) {
    CHECK_LT(reg, std::size(ThreadState{}.cpu.x));
    // bail early if already in error state (e.g. from Undefined())
    if (!success()) return no_register;
    if (IsRegMappingEnabled()) {
      auto [mapped_reg, is_new_mapping] = GetMappedRegisterOrMap(reg);
      // spill to temp when permanent pool is full
      if (!success()) {
        // Pool full: clear failure and fall through to temp-based load.
        success_ = true;
      } else {
        if (is_new_mapping) {
          int32_t offset = offsetof(ThreadState, cpu.x[0]) + reg * 8;
          as_.Movq(mapped_reg, {.base = Assembler::rbp, .disp = offset});
        }
        return mapped_reg;
      }
    }
    Register result = AllocTempReg();
    int32_t offset = offsetof(ThreadState, cpu.x[0]) + reg * 8;
    as_.Movq(result, {.base = Assembler::rbp, .disp = offset});
    return result;
  }

  void SetReg(uint8_t reg, Register value) {
    CHECK_LT(reg, std::size(ThreadState{}.cpu.x));
    // bail early if already in error state (e.g. from Undefined())
    if (!success()) return;
    if (IsRegMappingEnabled()) {
      auto [mapped_reg, _] = GetMappedRegisterOrMap(reg);
      if (success()) {
        as_.Movq(mapped_reg, value);
        gp_maintainer_.NoticeModified(reg);
        return;
      }
      // spill: pool full, write through to ThreadState
      success_ = true;
    }
    int32_t offset = offsetof(ThreadState, cpu.x[0]) + reg * 8;
    as_.Movq({.base = Assembler::rbp, .disp = offset}, value);
  }

  Register GetSp() {
    Register result = AllocTempReg();
    int32_t offset = offsetof(ThreadState, cpu.sp);
    as_.Movq(result, {.base = Assembler::rbp, .disp = offset});
    return result;
  }

  void SetSp(Register value) {
    int32_t offset = offsetof(ThreadState, cpu.sp);
    as_.Movq({.base = Assembler::rbp, .disp = offset}, value);
  }

  [[nodiscard]] Register GetImm(uint64_t imm) {
    Register imm_reg = AllocTempReg();
    as_.Movq(imm_reg, imm);
    return imm_reg;
  }

  [[nodiscard]] Register Copy(Register value) {
    Register result = AllocTempReg();
    as_.Movq(result, value);
    return result;
  }

  void StoreMappedRegs() {
    if (!IsRegMappingEnabled()) {
      return;
    }
    for (unsigned i = 0; i < kNumGuestRegs; i++) {
      if (gp_maintainer_.IsModified(i)) {
        auto mapped_reg = gp_maintainer_.GetMapped(i);
        int32_t offset = offsetof(ThreadState, cpu.x[0]) + i * 8;
        as_.Movq({.base = Assembler::rbp, .disp = offset}, mapped_reg);
      }
    }
  }

  //
  // Region exit methods.
  //

  void ExitGeneratedCode(GuestAddr target);
  void ExitRegion(GuestAddr target);
  void ExitRegionIndirect(Register target);

  // Mirror host MXCSR cumulative exception bits into emulated_fpsr at ARM FPSR
  // positions. Called from every region-exit path: System V x86_64 ABI lets
  // C++ runtime callees clobber MXCSR exception bits, so the capture must
  // happen here in JIT context, not in the interpreter MRS-FPSR handler.
  void EmitMxcsrToFpsrMirror();

  //
  // Instruction implementations.
  // Each method corresponds to a SemanticsPlayer callback.
  // Unsupported instructions set success_ = false for interpreter fallback.
  //

  // integer data-processing, branches, scalar load/store, system handlers (see lite_translator_integer_ctrl.inc).
#include "lite_translator_integer_ctrl.inc"
  // SIMD complex-FP / dot-product / modified-immediate / SIMD load-store handlers (see lite_translator_simd_fp_misc.inc).
#include "lite_translator_simd_fp_misc.inc"
  // scalar floating-point handlers (see lite_translator_fp_scalar.inc).
#include "lite_translator_fp_scalar.inc"
  // SIMD copy (INS/DUP/UMOV/SMOV) handlers (see lite_translator_simd_copy.inc).
#include "lite_translator_simd_copy.inc"
  // SIMD three-same-register vector arithmetic handlers (see lite_translator_simd_three_same.inc).
#include "lite_translator_simd_three_same.inc"
  // SIMD three-different-register, extract, permute, table handlers (see lite_translator_simd_three_diff_perm.inc).
#include "lite_translator_simd_three_diff_perm.inc"
  // cryptographic extension (SHA/SM3/SM4/AES) handlers (see lite_translator_crypto.inc).
#include "lite_translator_crypto.inc"
  // SIMD multi/single structure load/store (LD1/ST1 ...) handlers (see lite_translator_simd_struct_ldst.inc).
#include "lite_translator_simd_struct_ldst.inc"
  // integer carry, one-source, extract, cond-compare, exclusive/atomic handlers (see lite_translator_int_carry_excl.inc).
#include "lite_translator_int_carry_excl.inc"
  // scalar FP data-processing / compare / estimate tables handlers (see lite_translator_fp_dataproc.inc).
#include "lite_translator_fp_dataproc.inc"
  // SIMD two-register-misc vector handlers (see lite_translator_simd_two_reg_misc.inc).
#include "lite_translator_simd_two_reg_misc.inc"
  // SIMD scalar two-register-misc handlers (see lite_translator_simd_scalar_two_reg_misc.inc).
#include "lite_translator_simd_scalar_two_reg_misc.inc"
  // SIMD scalar three-same handlers (see lite_translator_simd_scalar_three_same.inc).
#include "lite_translator_simd_scalar_three_same.inc"
  // SIMD scalar pairwise + shift-by-immediate handlers (see lite_translator_simd_scalar_pairwise_shift.inc).
#include "lite_translator_simd_scalar_pairwise_shift.inc"
  // SIMD by-element (vector + scalar indexed) handlers (see lite_translator_simd_indexed.inc).
#include "lite_translator_simd_indexed.inc"
  //
  // Accessor helpers.
  //

  [[nodiscard]] Assembler* as() { return &as_; }
  [[nodiscard]] bool success() const { return success_; }
  bool is_region_end_reached() const { return is_region_end_reached_; }

  void FreeTempRegs() {
    gp_allocator_.FreeTemps();
    simd_allocator_.FreeTemps();
  }

  // Backward-branch inlining (a per-PC label map + local jumps for in-region
  // loops) was removed: it traps the CPU in a tight loop without signal
  // checks, and an in-region back-edge variant also miscompiled codec loops.
  // Backward edges always exit the region so signals are processed.

  bool IsRegMappingEnabled() { return params_.enable_reg_mapping; }

  std::tuple<Register, bool> GetMappedRegisterOrMap(int reg) {
    if (gp_maintainer_.IsMapped(reg)) {
      return {gp_maintainer_.GetMapped(reg), false};
    }
    if (auto alloc_result = gp_allocator_.Alloc()) {
      gp_maintainer_.Map(reg, alloc_result.value());
      return {alloc_result.value(), true};
    }
    success_ = false;
    return {Assembler::no_register, false};
  }

  Register AllocTempReg() {
    if (auto reg_option = gp_allocator_.AllocTemp()) {
      return reg_option.value();
    }
    success_ = false;
    return Assembler::no_register;
  }

  SimdRegister AllocTempSimdReg() {
    // any SIMD/FP work in the region marks MXCSR as
    // possibly-dirty so the mirror at region exit isn't elided.
    fp_dirty_ = true;
    if (auto reg_option = simd_allocator_.AllocTemp()) {
      return reg_option.value();
    }
    success_ = false;
    return Assembler::no_xmm_register;
  }

  // Byte offset of guest vector register v within ThreadState.
  static constexpr int32_t VRegOffset(unsigned v) {
    return static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + v * sizeof(__uint128_t));
  }
  // Load/store a full 128-bit guest vector register.
  void LoadVReg(SimdRegister xmm, unsigned v) {
    as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = VRegOffset(v)});
  }
  void StoreVReg(unsigned v, SimdRegister xmm) {
    as_.Movdqu({.base = Assembler::rbp, .disp = VRegOffset(v)}, xmm);
  }
  // Zero the upper 64 bits (Q=0 / D-register semantics).
  void MaskLow64(SimdRegister xmm) {
    as_.Pslldq(xmm, int8_t{8});
    as_.Psrldq(xmm, int8_t{8});
  }

  // Shared RDX/RCX spill scaffolding for UDIV/SDIV. Both hand-place the
  // dividend in RDX:RAX and must preserve RDX — and RCX when the guest divisor
  // maps to RDX — across the host DIV/IDIV. ARM {U,S}DIV return 0 for a zero
  // divisor (x86 DIV/IDIV fault), so both check the divisor first. kSigned
  // selects IDIV + CQO/CDQ sign-extension and the INT_MIN/-1 special case (ARM
  // returns INT_MIN where x86 IDIV faults); the unsigned path zero-extends RDX
  // and skips that case. The emitted instruction sequence is identical to the
  // former per-opcode hand-written bodies.
  template <bool kSigned>
  void EmitDivCommon(Register res, Register src1, Register src2, bool is_64bit) {
    constexpr int32_t kSlot = sizeof(uint64_t);
    Assembler::Label* zero = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.Subq(Assembler::rsp, kSlot);
    as_.Movq({.base = Assembler::rsp}, Assembler::rdx);  // save rdx (clobbered by DIV/IDIV)
    // If src2 is rdx, save rcx and use it as the divisor, since rdx is
    // clobbered below. The rcx SAVE must happen BEFORE the divide-by-zero (and,
    // for signed, the src2==-1) branch so the stack stays balanced on those
    // paths too: the `done` block always restores rcx when src2==rdx, so a save
    // that only ran on the divide path would leave the other paths popping a
    // phantom slot (corrupting rcx/rdx and unbalancing rsp).
    if (src2 == Assembler::rdx) {
      as_.Subq(Assembler::rsp, kSlot);
      as_.Movq({.base = Assembler::rsp}, Assembler::rcx);  // save rcx
    }
    if (is_64bit) {
      as_.Testq(src2, src2);
    } else {
      as_.Testl(src2, src2);
    }
    as_.Jcc(Condition::kEqual, *zero);
    if (kSigned) {
      // INT_MIN / -1: ARM64 returns INT_MIN, x86_64 IDIV faults.
      Assembler::Label* do_div = as_.MakeLabel();
      if (is_64bit) {
        as_.Cmpq(src2, static_cast<int32_t>(-1));
      } else {
        as_.Cmpl(src2, static_cast<int32_t>(-1));
      }
      as_.Jcc(Condition::kNotEqual, *do_div);
      // src2 == -1: result = -src1 (which equals INT_MIN for INT_MIN input).
      if (is_64bit) {
        as_.Movq(res, src1);
        as_.Negq(res);
      } else {
        as_.Movl(res, src1);
        as_.Negl(res);
      }
      as_.Jmp(*done);
      as_.Bind(do_div);
    }
    // {RDX:RAX (unsigned) or sign-extended src1} / divisor → quotient in RAX.
    // Move src1 to rax BEFORE clobbering rcx/rdx (src1 might be rcx or rdx).
    as_.Movq(Assembler::rax, src1);
    // Now that src1 is safely in rax, copy the divisor out of rdx into rcx.
    if (src2 == Assembler::rdx) {
      as_.Movq(Assembler::rcx, Assembler::rdx);
    }
    if (kSigned) {
      if (is_64bit) {
        as_.Cqo();
        as_.Idivq(src2 == Assembler::rdx ? Assembler::rcx : src2);
      } else {
        as_.Cdq();
        as_.Idivl(src2 == Assembler::rdx ? Assembler::rcx : src2);
      }
    } else {
      as_.Xorl(Assembler::rdx, Assembler::rdx);
      if (is_64bit) {
        as_.Divq(src2 == Assembler::rdx ? Assembler::rcx : src2);
      } else {
        as_.Divl(src2 == Assembler::rdx ? Assembler::rcx : src2);
      }
    }
    as_.Movq(res, Assembler::rax);
    as_.Jmp(*done);
    as_.Bind(zero);
    as_.Xorl(res, res);
    as_.Bind(done);
    // Restore rcx if we saved it (src2==rdx case).
    if (src2 == Assembler::rdx) {
      if (res != Assembler::rcx) {
        as_.Movq(Assembler::rcx, {.base = Assembler::rsp});
      }
      as_.Addq(Assembler::rsp, kSlot);  // pop rcx slot
    }
    if (res == Assembler::rdx) {
      as_.Addq(Assembler::rsp, kSlot);  // discard saved rdx
    } else {
      as_.Movq(Assembler::rdx, {.base = Assembler::rsp});
      as_.Addq(Assembler::rsp, kSlot);  // restore rdx
    }
  }

  // 8-bit widen-multiply: computes the per-lane byte product of xn and xm,
  // leaving the low-8-bit-truncated result in xn. x86 has no PMULLB, so each
  // 8 bytes is widened to 16-bit words (low + high half separately), PMULLW'd,
  // masked to the low byte of each word (so PACKUSWB doesn't saturate), then
  // packed. Clobbers xm and allocates two SIMD temps. Returns false (with
  // success_ already cleared) if the temp pool is exhausted; the caller must
  // Undefined()/return without emitting more.
  [[nodiscard]] bool EmitByteMulWiden(SimdRegister xn, SimdRegister xm) {
    SimdRegister xn_hi = AllocTempSimdReg();
    SimdRegister xm_hi = AllocTempSimdReg();
    if (!success()) {
      return false;
    }
    as_.Movdqa(xn_hi, xn);
    as_.Movdqa(xm_hi, xm);
    as_.Psrldq(xn_hi, int8_t{8});
    as_.Psrldq(xm_hi, int8_t{8});
    as_.Pmovzxbw(xn, xn);
    as_.Pmovzxbw(xm, xm);
    as_.Pmovzxbw(xn_hi, xn_hi);
    as_.Pmovzxbw(xm_hi, xm_hi);
    as_.Pmullw(xn, xm);
    as_.Pmullw(xn_hi, xm_hi);
    // Reuse xm as the 0x00FF×8 mask: PCMPEQB writes all-ones, PSRLW 8
    // clears the high byte of each 16-bit lane.
    as_.Pcmpeqb(xm, xm);
    as_.Psrlw(xm, int8_t{8});
    as_.Pand(xn, xm);
    as_.Pand(xn_hi, xm);
    as_.Packuswb(xn, xn_hi);
    return true;
  }

  // explicit marker for FP-affecting paths that don't go
  // through AllocTempSimdReg (e.g. a future scalar FP path that directly uses
  // a fixed XMM register). Currently unused; kept for forward use.
  void MarkFpDirty() { fp_dirty_ = true; }
  bool fp_dirty() const { return fp_dirty_; }

 private:
  // Helper: extract ARM64 NZCV flags from x86_64 EFLAGS after ADD/SUB.
  //
  // x86_64 LAHF stores flags into AH with bit positions that happen to match
  // the ARM64 flag word layout:
  //   AX[15] = SF = ARM64 N    AX[14] = ZF = ARM64 Z    AX[8] = CF = ARM64 C
  // ARM64 V (overflow) at bit 0 is extracted via SETO.
  //
  // For SUB, ARM64 C = !x86_CF (ARM64 uses inverted borrow), so we XOR bit 8.
  // use AL for overflow (avoids clobbering rcx in allocator pool).
  // LAHF stores SF|ZF|CF to AH (bits 8-15). SETCC OF stores to AL (bits 0-7).
  // AND kFlagsNZCVMask keeps N(bit15), Z(bit14), C(bit8), V(bit0). No rcx save/restore needed.
  void EmitStoreArmNZCV(bool is_sub) {
    as_.Lahf();
    as_.Setcc(Condition::kOverflow, Assembler::rax);
    as_.Andl(Assembler::rax, static_cast<int32_t>(kFlagsNZCVMask));
    if (is_sub) {
      as_.Xorl(Assembler::rax, static_cast<int32_t>(CPUState::kFlagCarry));  // invert ARM borrow
    }
    int32_t flags_offset = offsetof(ThreadState, cpu.flags);
    as_.Movw({.base = Assembler::rbp, .disp = flags_offset}, Assembler::rax);
  }

  // Emit a jump to `not_met` when the ARM64 condition `cond` is NOT satisfied
  // by the current NZCV in ThreadState.cpu.flags. Shared by every flag
  // consumer (B.cond, CCMP/CCMN, CSEL family) — each branches over its action
  // when the condition fails, so this is the single place that decodes an ARM
  // condition.
  //
  // Single-bit conditions test cpu.flags in memory directly (Btw on
  // [rbp+flags]) and allocate no scratch register, keeping the hot branch path
  // off the GP pool (less register pressure -> fewer early region exits). Only
  // the N==V comparisons (GE/LT/GT/LE) load flags into a scratch for the XOR.
  // ARM flag bit positions in cpu.flags: N=15, Z=14, C=8, V=0. Btw sets CF to
  // the tested bit, so the Jcc polarity mirrors the previous Btl-on-register
  // form exactly.
  void EmitJumpIfCondNotMet(Decoder::Condition cond, const Assembler::Label& not_met) {
    const int32_t f = offsetof(ThreadState, cpu.flags);
    // (N xor V) in CF, then jump to not_met with the given polarity. Used by
    // the signed comparisons; needs flags in a scratch for the XOR.
    auto emit_n_xor_v = [&](Condition jump_when) {
      Register flags_reg = AllocTempReg();
      as_.Movzxwl(flags_reg, {.base = Assembler::rbp, .disp = f});
      Register tmp = AllocTempReg();
      as_.Movl(tmp, flags_reg);
      as_.Shrl(tmp, kFlagNegativeBit);  // N -> bit 0
      as_.Xorl(tmp, flags_reg);                // bit 0 = N ^ V
      as_.Btl(tmp, kFlagOverflowBit);    // CF = N ^ V
      as_.Jcc(jump_when, not_met);
    };
    switch (cond) {
      case Decoder::Condition::kEq:  // Z==1
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagZeroBit);
        as_.Jcc(Condition::kNotCarry, not_met);
        break;
      case Decoder::Condition::kNe:  // Z==0
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagZeroBit);
        as_.Jcc(Condition::kCarry, not_met);
        break;
      case Decoder::Condition::kCs:  // C==1
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagCarryBit);
        as_.Jcc(Condition::kNotCarry, not_met);
        break;
      case Decoder::Condition::kCc:  // C==0
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagCarryBit);
        as_.Jcc(Condition::kCarry, not_met);
        break;
      case Decoder::Condition::kMi:  // N==1
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagNegativeBit);
        as_.Jcc(Condition::kNotCarry, not_met);
        break;
      case Decoder::Condition::kPl:  // N==0
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagNegativeBit);
        as_.Jcc(Condition::kCarry, not_met);
        break;
      case Decoder::Condition::kVs:  // V==1
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagOverflowBit);
        as_.Jcc(Condition::kNotCarry, not_met);
        break;
      case Decoder::Condition::kVc:  // V==0
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagOverflowBit);
        as_.Jcc(Condition::kCarry, not_met);
        break;
      case Decoder::Condition::kHi:  // C==1 && Z==0
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagCarryBit);
        as_.Jcc(Condition::kNotCarry, not_met);  // C==0 -> not met
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagZeroBit);
        as_.Jcc(Condition::kCarry, not_met);     // Z==1 -> not met
        break;
      case Decoder::Condition::kLs: {  // C==0 || Z==1
        Assembler::Label* met = as_.MakeLabel();
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagCarryBit);
        as_.Jcc(Condition::kNotCarry, *met);     // C==0 -> met
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagZeroBit);
        as_.Jcc(Condition::kNotCarry, not_met);  // C==1 && Z==0 -> not met
        as_.Bind(met);
        break;
      }
      case Decoder::Condition::kGe:  // N==V
        emit_n_xor_v(Condition::kCarry);         // N!=V -> not met
        break;
      case Decoder::Condition::kLt:  // N!=V
        emit_n_xor_v(Condition::kNotCarry);      // N==V -> not met
        break;
      case Decoder::Condition::kGt:  // Z==0 && N==V
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagZeroBit);
        as_.Jcc(Condition::kCarry, not_met);     // Z==1 -> not met
        emit_n_xor_v(Condition::kCarry);         // N!=V -> not met
        break;
      case Decoder::Condition::kLe: {  // Z==1 || N!=V
        Assembler::Label* met = as_.MakeLabel();
        as_.Btw({.base = Assembler::rbp, .disp = f}, kFlagZeroBit);
        as_.Jcc(Condition::kCarry, *met);        // Z==1 -> met
        emit_n_xor_v(Condition::kNotCarry);      // Z==0 && N==V -> not met
        as_.Bind(met);
        break;
      }
      case Decoder::Condition::kAl:
      case Decoder::Condition::kNv:
        // Always met: emit nothing.
        break;
    }
  }

  // emit ARM FP-compare NZCV from x86 UCOMIS flags.
  //
  // UCOMISS/UCOMISD set ZF/PF/CF and leave SF/OF untouched, so the integer
  // EmitStoreArmNZCV path (which copies SF into ARM N and OF into ARM V)
  // produced random N and V.  Map the four ordered outcomes by jump-table:
  //   x86 ZF PF CF   ARM NZCV (bit15 N, bit14 Z, bit8 C, bit0 V)   value
  //   gt:  0 0 0  -> 0 0 1 0                                       0x0100
  //   lt:  0 0 1  -> 1 0 0 0                                       0x8000
  //   eq:  1 0 0  -> 0 1 1 0                                       0x4100
  //   uo:  1 1 1  -> 0 0 1 1                                       0x0101
  // The Movl-imm sequence below preserves EFLAGS (MOV doesn't touch them),
  // so the Jcc reads UCOMISS's flags directly.
  void EmitStoreArmFpNZCV() {
    Assembler::Label* uo_label = as_.MakeLabel();
    Assembler::Label* eq_label = as_.MakeLabel();
    Assembler::Label* lt_label = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();

    as_.Movl(Assembler::rax, static_cast<int32_t>(kFlagsFpGreater));  // gt (default)
    as_.Jcc(Condition::kParityEven, *uo_label);  // PF=1 -> unordered (NaN)
    as_.Jcc(Condition::kEqual, *eq_label);       // ZF=1 (PF=0) -> equal
    as_.Jcc(Condition::kBelow, *lt_label);       // CF=1 -> less
    as_.Jmp(*done);                              // else gt

    as_.Bind(lt_label);
    as_.Movl(Assembler::rax, static_cast<int32_t>(kFlagsFpLess));
    as_.Jmp(*done);

    as_.Bind(eq_label);
    as_.Movl(Assembler::rax, static_cast<int32_t>(kFlagsFpEqual));
    as_.Jmp(*done);

    as_.Bind(uo_label);
    as_.Movl(Assembler::rax, static_cast<int32_t>(kFlagsFpUnordered));

    as_.Bind(done);
    int32_t flags_offset = offsetof(ThreadState, cpu.flags);
    as_.Movw({.base = Assembler::rbp, .disp = flags_offset}, Assembler::rax);
  }

  // Helper: emit shift of src into dst by a compile-time constant amount.
  void EmitShift(Register dst, Register src, Decoder::ShiftType shift_type,
                 uint8_t shift_amount, bool is_64bit) {
    if (is_64bit) {
      as_.Movq(dst, src);
      if (shift_amount == 0) return;
      switch (shift_type) {
        case Decoder::ShiftType::kLsl:
          as_.Shlq(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kLsr:
          as_.Shrq(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kAsr:
          as_.Sarq(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kRor:
          as_.Rorq(dst, static_cast<int8_t>(shift_amount));
          break;
      }
    } else {
      as_.Movl(dst, src);
      if (shift_amount == 0) return;
      switch (shift_type) {
        case Decoder::ShiftType::kLsl:
          as_.Shll(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kLsr:
          as_.Shrl(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kAsr:
          as_.Sarl(dst, static_cast<int8_t>(shift_amount));
          break;
        case Decoder::ShiftType::kRor:
          as_.Rorl(dst, static_cast<int8_t>(shift_amount));
          break;
      }
    }
  }

  Assembler as_;
  bool success_;
  GuestAddr pc_;
  Allocator<Register> gp_allocator_;
  RegisterFileMaintainer<Register, kNumGuestRegs> gp_maintainer_;
  Allocator<SimdRegister> simd_allocator_;
  const LiteTranslateParams params_;
  bool is_region_end_reached_;
  // see constructor comment.
  bool fp_dirty_;
  // guest PC label map for backward branch inlining
};

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_H_
