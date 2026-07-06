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

#ifndef BERBERIS_HEAVY_OPTIMIZER_ARM64_FRONTEND_H_
#define BERBERIS_HEAVY_OPTIMIZER_ARM64_FRONTEND_H_

#include <cstddef>
#include <cstdint>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/base/arena_map.h"
#include "berberis/base/checks.h"
#include "berberis/decoder/arm64/decoder.h"
#include "berberis/decoder/arm64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_arch.h"
#include "berberis/intrinsics/intrinsics.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/runtime_primitives/platform.h"

#include "call_intrinsic.h"
#include "inline_intrinsic.h"
#include "simd_register.h"

namespace berberis {

// ARM64 optimizing-tier frontend: translates the ARM64 decoder's
// SemanticsPlayer callbacks into guest-agnostic x86_64 MachineIR. The generic
// region machinery (StartRegion / GenJump / ExitGeneratedCode / ResolveJumps /
// Finalize / StartInsn) is adapted from heavy_optimizer/riscv64/frontend.{h,cc}.
//
// Integer/branch/load-store instructions are translated to native x86_64.
// Scalar floating-point arithmetic (FADD/FSUB/FMUL/FDIV for S and D) is lowered
// through the guest-agnostic intrinsic layer (inline_intrinsic.h +
// machine_ir_intrinsic_binding.json), and FMOV/FABS/FNEG/FSQRT/FMOV-imm are
// emitted directly. Anything not handled calls Undefined() (sets success_ =
// false) so the two-gear runtime falls back to the lite translator/interpreter.
// New instructions are added here as the optimizing tier grows.
class HeavyOptimizerFrontend {
 public:
  using Decoder = Decoder<SemanticsPlayer<HeavyOptimizerFrontend>>;
  using Register = MachineReg;
  static constexpr Register no_register = MachineReg{};
  using FpRegister = SimdReg;
  static constexpr SimdReg no_fp_register = SimdReg{};
  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

  explicit HeavyOptimizerFrontend(x86_64::MachineIR* machine_ir, GuestAddr pc)
      : pc_(pc),
        success_(true),
        builder_(machine_ir),
        flag_register_(machine_ir->AllocVReg()),
        is_uncond_branch_(false),
        branch_targets_(machine_ir->arena()) {
    StartRegion();
  }

  //
  // Guest state getters/setters.
  //

  [[nodiscard]] GuestAddr GetInsnAddr() const { return pc_; }
  void IncrementInsnAddr(uint8_t insn_size) { pc_ += insn_size; }

  // Once the region has bailed (success_ == false) the IR-emitting read helpers
  // must emit NOTHING more: a single guest instruction can call several listener
  // methods (e.g. a pre/post-index access calls AddImm then Load/Store), and the
  // first bail already terminated the current basic block with an exit. Any
  // further instruction appended after that terminator — or a second exit — makes
  // the block fail CheckMachineIR. So after a bail these return an unused
  // undefined temp without generating code; SetReg/SetSp/Undefined are likewise
  // inert. The undefined temp never reaches live machine code because every
  // downstream consumer on the bailed instruction is suppressed too.
  [[nodiscard]] Register GetReg(uint8_t reg) {
    CHECK_LT(reg, kNumGuestRegs);
    Register dst = AllocTempReg();
    if (success()) {
      builder_.GenGet(dst, GetThreadStateRegOffset(reg));
    }
    return dst;
  }

  void SetReg(uint8_t reg, Register value) {
    CHECK_LT(reg, kNumGuestRegs);
    if (success()) {
      builder_.GenPut(GetThreadStateRegOffset(reg), value);
    }
  }

  [[nodiscard]] Register GetSp() {
    Register dst = AllocTempReg();
    if (success()) {
      builder_.GenGet(dst, GetThreadStateSpOffset());
    }
    return dst;
  }

  void SetSp(Register value) {
    if (success()) {
      builder_.GenPut(GetThreadStateSpOffset(), value);
    }
  }

  [[nodiscard]] Register GetImm(uint64_t imm) {
    if (!success()) {
      return AllocTempReg();
    }
    return std::get<0>(Gen<x86_64::MovqRegImm>(imm));
  }

  [[nodiscard]] Register Copy(Register value) {
    Register result = AllocTempReg();
    if (success()) {
      builder_.Gen<PseudoCopy>(result, value, 8);
    }
    return result;
  }

  [[nodiscard]] bool success() const { return success_; }

  void Undefined();
  void Nop() {}

  // DMB/DSB full barrier: lowers to MFENCE (x86 TSO lacks StoreLoad ordering).
  // The store-only/load-only barrier variants are NOPed in the decoder.
  void DataMemoryBarrier();

  //
  // Immediate-form data processing.
  //

  // ADD/SUB (immediate), including the flag-setting SUBS/ADDS/CMP/CMN forms.
  // When set_flags is true the op always emits (even imm==0) so host EFLAGS are
  // valid, then EmitMaterializeNZCV packs NZCV into cpu.flags exactly as
  // lite_translator.h::EmitStoreArmNZCV. CMP/CMN to XZR (rd==31) discard the
  // result in the SemanticsPlayer (SetRegOrIgnore). Mirrors
  // lite_translator.h::AddSubImm: a 32-bit op uses the l-suffix insns (which
  // zero-extend the upper 32 bits, matching ARM64 W-register write semantics).
  Register AddSubImm(bool is_sub, bool set_flags, bool is_64bit, Register src, uint32_t imm) {
    // A prior callback for this guest instruction may have already bailed (e.g.
    // post-index Load() bails before this AddImm-equivalent runs); emit nothing.
    if (!success()) {
      return AllocTempReg();
    }
    // The ARM imm12 fits in int32 and x86 add/sub-immediate take int32.
    if (is_64bit) {
      Register res = Copy(src);
      // When setting flags, always emit the op (even imm==0) so EFLAGS are valid.
      if (set_flags || imm != 0) {
        if (is_sub) {
          auto [r, flags] = Gen<x86_64::SubqRegImm, kNoSSA>(res, static_cast<int32_t>(imm));
          res = r;
          if (set_flags) {
            EmitMaterializeNZCV(flags, /*is_sub=*/true);
          }
        } else {
          auto [r, flags] = Gen<x86_64::AddqRegImm, kNoSSA>(res, static_cast<int32_t>(imm));
          res = r;
          if (set_flags) {
            EmitMaterializeNZCV(flags, /*is_sub=*/false);
          }
        }
      }
      return res;
    }
    // 32-bit: a 32-bit mov zero-extends src to 64, then the 32-bit op keeps the
    // upper 32 bits clear (ARM64 W-write semantics).
    Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
    if (set_flags || imm != 0) {
      if (is_sub) {
        auto [r, flags] = Gen<x86_64::SublRegImm, kNoSSA>(res, static_cast<int32_t>(imm));
        res = r;
        if (set_flags) {
          EmitMaterializeNZCV(flags, /*is_sub=*/true);
        }
      } else {
        auto [r, flags] = Gen<x86_64::AddlRegImm, kNoSSA>(res, static_cast<int32_t>(imm));
        res = r;
        if (set_flags) {
          EmitMaterializeNZCV(flags, /*is_sub=*/false);
        }
      }
    }
    return res;
  }

  Register AddSubImmTags(bool is_sub, Register src, uint8_t uimm6, uint8_t uimm4) {
    UndefinedReturningReg();
    UNUSED_ARGS(is_sub, src, uimm6, uimm4);
    return AllocTempReg();
  }

  // AND/ORR/EOR (immediate, decoded 64-bit bitmask), including ANDS/TST. x86 AND
  // clears CF and OF, so ANDS materializes ARM64 NZCV with C=0 and V=0 (is_sub
  // false, no borrow XOR). x86 logical-immediate forms only take a 32-bit
  // immediate, but the bitmask
  // immediate needs the full 64 bits, so materialize it into a register and use
  // the reg-reg forms (mirrors lite_translator.h::LogicalImm).
  Register LogicalImm(Decoder::LogicalImmOpcode opcode, bool is_64bit, Register src, uint64_t imm) {
    if (!success()) {
      return AllocTempReg();
    }
    Register imm_reg = GetImm(imm);
    if (is_64bit) {
      Register res = Copy(src);
      switch (opcode) {
        case Decoder::LogicalImmOpcode::kAnd:
          return std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, imm_reg));
        case Decoder::LogicalImmOpcode::kAnds: {
          // ANDS/TST: x86 AND clears CF and OF, so ARM64 C=0 and V=0 naturally.
          auto [r, flags] = Gen<x86_64::AndqRegReg, kNoSSA>(res, imm_reg);
          EmitMaterializeNZCV(flags, /*is_sub=*/false);
          return r;
        }
        case Decoder::LogicalImmOpcode::kOrr:
          return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(res, imm_reg));
        case Decoder::LogicalImmOpcode::kEor:
          return std::get<0>(Gen<x86_64::XorqRegReg, kNoSSA>(res, imm_reg));
        default:
          UndefinedReturningReg();
          return AllocTempReg();
      }
    }
    // 32-bit op: the l-suffix form zero-extends the result to 64 bits.
    Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
    switch (opcode) {
      case Decoder::LogicalImmOpcode::kAnd:
        return std::get<0>(Gen<x86_64::AndlRegReg, kNoSSA>(res, imm_reg));
      case Decoder::LogicalImmOpcode::kAnds: {
        auto [r, flags] = Gen<x86_64::AndlRegReg, kNoSSA>(res, imm_reg);
        EmitMaterializeNZCV(flags, /*is_sub=*/false);
        return r;
      }
      case Decoder::LogicalImmOpcode::kOrr:
        return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(res, imm_reg));
      case Decoder::LogicalImmOpcode::kEor:
        return std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(res, imm_reg));
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }

  // MOVZ / MOVN: result is compile-time known.
  Register MoveWide(Decoder::MoveWideOpcode opcode, bool is_64bit, uint16_t imm16, uint8_t shift) {
    uint64_t value = static_cast<uint64_t>(imm16) << shift;
    if (opcode == Decoder::MoveWideOpcode::kMovn) {
      value = ~value;
    }
    if (!is_64bit) {
      value &= 0xFFFFFFFFULL;
    }
    return std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(value)));
  }

  // MOVK: keep other bits of `current`, overwrite the 16-bit window at `shift`.
  // x86_64 AND/OR-immediate forms only accept a 32-bit sign-extended immediate,
  // but ~mask / value can need a full 64-bit immediate (shift up to 48), so
  // materialize each constant into a register and use the reg-reg forms,
  // mirroring the lite translator.
  Register MoveWideKeep(Register current, uint16_t imm16, uint8_t shift, bool is_64bit) {
    uint64_t mask = static_cast<uint64_t>(0xFFFF) << shift;
    uint64_t value = static_cast<uint64_t>(imm16) << shift;
    // res = current & ~mask
    Register not_mask = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(~mask)));
    Register res = std::get<0>(Gen<x86_64::AndqRegReg>(current, not_mask));
    // res = res | value
    Register value_reg = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(value)));
    res = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(res, value_reg));
    if (!is_64bit) {
      // Zero-extend 32->64 (a 32-bit mov clears the upper 32 bits).
      res = std::get<0>(Gen<x86_64::MovlRegReg, kNoSSA>(res));
    }
    return res;
  }

  // ADR / ADRP: the target is a pure function of the (translation-time-constant)
  // guest PC and the decoded offset, so materialize it as an immediate. ADRP
  // page-aligns the PC first. Mirrors lite_translator.h::PcRelAddr.
  Register PcRelAddr(bool is_adrp, int64_t offset) {
    if (!success()) {
      return AllocTempReg();
    }
    GuestAddr pc = GetInsnAddr();
    GuestAddr target = is_adrp ? ((pc & ~static_cast<GuestAddr>(0xFFF)) + offset) : (pc + offset);
    return GetImm(static_cast<uint64_t>(target));
  }

  // LDR/LDRSW (literal): load from [insn_addr + offset]. The address is constant
  // at translation time, so materialize it with GetImm and reuse Load() (which
  // applies TBI, emits the size-appropriate movzx/movsx, and sets the recovery
  // point). Mirrors lite_translator.h::LoadLiteral.
  Register LoadLiteral(Decoder::LoadStoreSize size, bool is_signed, int64_t offset) {
    if (!success()) {
      return AllocTempReg();
    }
    GuestAddr target = GetInsnAddr() + offset;
    Register addr = GetImm(target);
    bool is_64bit_target = (size == Decoder::LoadStoreSize::k64bit) || is_signed;
    return Load(size, is_signed, is_64bit_target, addr, 0);
  }

  // SBFM/UBFM/BFM (bitfield move). Mirrors lite_translator.h::Bitfield. The
  // 32-bit paths use l-suffix shifts/and (which zero-extend the upper 32 bits,
  // matching ARM64 W-register write semantics).
  Register Bitfield(Decoder::BitfieldOpcode opcode,
                    bool is_64bit,
                    Register dst_val,
                    Register src,
                    uint8_t immr,
                    uint8_t imms) {
    if (!success()) {
      return AllocTempReg();
    }
    unsigned reg_size = is_64bit ? 64 : 32;

    if (opcode == Decoder::BitfieldOpcode::kUbfm) {
      // LSR: UBFM Rd, Rn, #shift, #(regsize-1).
      if (imms == reg_size - 1) {
        if (is_64bit) {
          Register res = Copy(src);
          if (immr != 0) {
            res = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
          }
          return res;
        }
        Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        if (immr != 0) {
          res = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
        }
        return res;
      }
      // LSL: UBFM Rd, Rn, #(regsize-shift), #(regsize-1-shift).
      if (imms + 1 == immr && imms < reg_size - 1) {
        uint8_t shift = reg_size - immr;
        if (is_64bit) {
          Register res = Copy(src);
          return std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(shift)));
        }
        Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        return std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(shift)));
      }
      // UXTB: UBFM Wd, Wn, #0, #7.
      if (!is_64bit && immr == 0 && imms == 7) {
        return std::get<0>(Gen<x86_64::MovzxblRegReg>(src));
      }
      // UXTH: UBFM Wd, Wn, #0, #15.
      if (!is_64bit && immr == 0 && imms == 15) {
        return std::get<0>(Gen<x86_64::MovzxwlRegReg>(src));
      }
      // General UBFM (UBFX extract; UBFIZ insert).
      if (imms >= immr) {
        // UBFX-like: extract bits[imms:immr] of src to low bits of dst.
        unsigned width = imms - immr + 1;
        uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
        if (is_64bit) {
          Register res = Copy(src);
          if (immr != 0) {
            res = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
          }
          if (width < 64) {
            Register mask_reg = GetImm(mask);
            res = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, mask_reg));
          }
          return res;
        }
        Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        if (immr != 0) {
          res = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
        }
        if (width < 32) {
          res = std::get<0>(
              Gen<x86_64::AndlRegImm, kNoSSA>(res, static_cast<int32_t>(mask & 0xFFFFFFFFULL)));
        }
        return res;
      }
      // UBFIZ-like: extract low (imms+1) bits of src, shift left by (reg_size-immr).
      unsigned width = imms + 1;
      unsigned pos = reg_size - immr;
      uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
      if (is_64bit) {
        Register res = Copy(src);
        if (width < 64) {
          Register mask_reg = GetImm(mask);
          res = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, mask_reg));
        }
        if (pos != 0) {
          res = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(pos)));
        }
        return res;
      }
      Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      if (width < 32) {
        res = std::get<0>(
            Gen<x86_64::AndlRegImm, kNoSSA>(res, static_cast<int32_t>(mask & 0xFFFFFFFFULL)));
      }
      if (pos != 0) {
        res = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(pos)));
      }
      return res;
    }

    if (opcode == Decoder::BitfieldOpcode::kSbfm) {
      // ASR: SBFM Rd, Rn, #shift, #(regsize-1).
      if (imms == reg_size - 1) {
        if (is_64bit) {
          Register res = Copy(src);
          if (immr != 0) {
            res = std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
          }
          return res;
        }
        // 32-bit ASR: Sarl writes the low 32 and zero-extends to 64, matching
        // ARM64 W-write semantics. Do NOT sign-extend further.
        Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        if (immr != 0) {
          res = std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
        }
        return res;
      }
      // SXTB: SBFM Xd/Wd, Wn, #0, #7.
      if (immr == 0 && imms == 7) {
        if (is_64bit) {
          return std::get<0>(Gen<x86_64::MovsxbqRegReg>(src));
        }
        return std::get<0>(Gen<x86_64::MovsxblRegReg>(src));
      }
      // SXTH: SBFM Xd/Wd, Wn, #0, #15.
      if (immr == 0 && imms == 15) {
        if (is_64bit) {
          return std::get<0>(Gen<x86_64::MovsxwqRegReg>(src));
        }
        return std::get<0>(Gen<x86_64::MovsxwlRegReg>(src));
      }
      // SXTW: SBFM Xd, Wn, #0, #31.
      if (is_64bit && immr == 0 && imms == 31) {
        return std::get<0>(Gen<x86_64::MovsxlqRegReg>(src));
      }
      // General SBFX (imms >= immr): sign-extend the field src[imms:immr] down to
      // bit 0. Two shifts: LSL by (reg_size-1-imms) lands bit imms at the MSB,
      // then ASR by (reg_size-1-imms+immr) shifts the field back to bit 0 while
      // sign-extending from the field's top bit. For 32-bit the Sar writes the
      // low 32 and zero-extends to 64, matching ARM64 W-write semantics. The
      // imms < immr case (SBFIZ) is handled separately below.
      if (imms >= immr) {
        const uint8_t left = static_cast<uint8_t>(reg_size - 1 - imms);
        const uint8_t right = static_cast<uint8_t>(left + immr);
        if (is_64bit) {
          Register res = Copy(src);
          if (left != 0) {
            res = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(left)));
          }
          if (right != 0) {
            res = std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(res, static_cast<int8_t>(right)));
          }
          return res;
        }
        Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        if (left != 0) {
          res = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(left)));
        }
        if (right != 0) {
          res = std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(res, static_cast<int8_t>(right)));
        }
        return res;
      }
      // SBFIZ (imms < immr): sign-extend the low (imms+1) bits of src, then shift
      // left by lsb = reg_size - immr. Two shifts mirror the general-SBFX idiom:
      // LSL by (reg_size-1-imms) lands the field's top bit at the MSB, then ASR
      // by (immr-1-imms) sign-extends and lands bit 0 at position lsb. Since
      // imms < immr, right = immr-1-imms >= 0. For 32-bit the Sar writes the low
      // 32 and zero-extends to 64 (ARM64 W-write semantics).
      const uint8_t sbfiz_left = static_cast<uint8_t>(reg_size - 1 - imms);
      const uint8_t sbfiz_right = static_cast<uint8_t>(immr - 1 - imms);
      if (is_64bit) {
        Register res = Copy(src);
        if (sbfiz_left != 0) {
          res = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(sbfiz_left)));
        }
        if (sbfiz_right != 0) {
          res = std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(res, static_cast<int8_t>(sbfiz_right)));
        }
        return res;
      }
      Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
      if (sbfiz_left != 0) {
        res = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(sbfiz_left)));
      }
      if (sbfiz_right != 0) {
        res = std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(res, static_cast<int8_t>(sbfiz_right)));
      }
      return res;
    }

    // General BFM (BFI / BFXIL / BFC).
    //   result = (dst_val & ~mask) | (shifted_src & mask)
    if (opcode == Decoder::BitfieldOpcode::kBfm) {
      uint64_t mask;
      Register res;
      if (imms >= immr) {
        // BFXIL: extract width = imms-immr+1 bits, deposit at bit 0.
        unsigned width = imms - immr + 1;
        mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
        if (is_64bit) {
          res = Copy(src);
          if (immr != 0) {
            res = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
          }
          Register mask_reg = GetImm(mask);
          res = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, mask_reg));
        } else {
          res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
          if (immr != 0) {
            res = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(res, static_cast<int8_t>(immr)));
          }
          res = std::get<0>(
              Gen<x86_64::AndlRegImm, kNoSSA>(res, static_cast<int32_t>(mask & 0xFFFFFFFFULL)));
        }
      } else {
        // BFI / BFC: extract low width = imms+1 bits, deposit at pos.
        unsigned width = imms + 1;
        unsigned pos = reg_size - immr;
        uint64_t field_mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
        mask = (pos >= 64) ? 0 : (field_mask << pos);
        if (is_64bit) {
          res = Copy(src);
          Register fmask_reg = GetImm(field_mask);
          res = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, fmask_reg));
          if (pos != 0) {
            res = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(res, static_cast<int8_t>(pos)));
          }
        } else {
          res = std::get<0>(Gen<x86_64::MovlRegReg>(src));
          res = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(
              res, static_cast<int32_t>(field_mask & 0xFFFFFFFFULL)));
          if (pos != 0) {
            res = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(res, static_cast<int8_t>(pos)));
          }
        }
      }
      // Merge: res = res | (dst_val & ~mask). Copy dst_val into a fresh temp.
      if (is_64bit) {
        Register keep = Copy(dst_val);
        Register notmask_reg = GetImm(~mask);
        keep = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(keep, notmask_reg));
        return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(res, keep));
      }
      Register keep = std::get<0>(Gen<x86_64::MovlRegReg>(dst_val));
      keep = std::get<0>(
          Gen<x86_64::AndlRegImm, kNoSSA>(keep, static_cast<int32_t>((~mask) & 0xFFFFFFFFULL)));
      return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(res, keep));
    }

    UndefinedReturningReg();
    return AllocTempReg();
  }

  //
  // Branches.
  //
  // The direct branch family translates to MachineIR control flow exactly like
  // the riscv64 frontend: a conditional branch creates then_bb/else_bb basic
  // blocks ending in a PseudoCondBranch, the taken path GenJumps to the target
  // (which links in-region back-edges and adds the pending-signal check), and
  // the not-taken path continues translating in else_bb. An unconditional B /
  // BR sets is_uncond_branch_ so StartInsn opens a fresh block for whatever
  // follows. The ARM64 condition is evaluated from ThreadState.cpu.flags
  // (NZCV: N@15, Z@14, C@8, V@0) bit-for-bit as lite_translator's
  // EmitJumpIfCondNotMet.

  // B (unconditional). The SemanticsPlayer writes X30 for BL before this runs.
  void Branch(int32_t offset);

  // B.cond (conditional). AL/NV are unconditional.
  void BranchCond(Decoder::Condition cond, int32_t offset);

  // BR/RET/BLR (indirect). The SemanticsPlayer writes X30 for BLR before this
  // runs. Mirrors lite_translator's BranchRegister: no TBI masking of the top
  // byte (it just exits indirect to `target`).
  void BranchRegister(Register target);

  // CBZ (is_nonzero=false) / CBNZ (is_nonzero=true).
  void CompareAndBranch(bool is_nonzero, bool is_64bit, Register src, int32_t offset);

  // TBZ (is_nonzero=false) / TBNZ (is_nonzero=true).
  void TestAndBranch(bool is_nonzero, Register src, uint8_t bit, int32_t offset);

  //
  // Integer loads / stores.
  //

  // Integer load with a base+imm offset. Mirrors lite_translator.h::Load: ApplyTbi
  // masks the top byte of the address (ARM64 TBI), then a size/sign-appropriate
  // host load reads from {masked, offset}. A <32-bit unsigned load zero-extends to
  // 32 (a 32-bit dest reg clears the upper 32); a signed load sign-extends to 32 or
  // 64 per is_64bit_target; a 32-bit unsigned load zero-extends to 64 (Movl), and
  // LDRSW (signed 32->64) uses Movsxlq. GenRecoveryBlockForLastInsn() exits the
  // region if the host access faults, so the guest signal is delivered.
  Register Load(Decoder::LoadStoreSize size,
                bool is_signed,
                bool is_64bit_target,
                Register base,
                int32_t offset) {
    if (!success()) {
      return AllocTempReg();
    }
    Register masked = ApplyTbi(base);
    Register res;
    switch (size) {
      case Decoder::LoadStoreSize::k64bit:
        res = std::get<0>(Gen<x86_64::MovqRegOp>({.base = masked, .disp = offset}));
        break;
      case Decoder::LoadStoreSize::k32bit:
        if (is_signed && is_64bit_target) {
          // LDRSW: 32 -> sign-extend to 64.
          res = std::get<0>(Gen<x86_64::MovsxlqRegOp>({.base = masked, .disp = offset}));
        } else {
          // 32-bit unsigned: zero-extends to 64.
          res = std::get<0>(Gen<x86_64::MovlRegOp>({.base = masked, .disp = offset}));
        }
        break;
      case Decoder::LoadStoreSize::k16bit:
        if (is_signed) {
          if (is_64bit_target) {
            res = std::get<0>(Gen<x86_64::MovsxwqRegOp>({.base = masked, .disp = offset}));
          } else {
            res = std::get<0>(Gen<x86_64::MovsxwlRegOp>({.base = masked, .disp = offset}));
          }
        } else {
          res = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = masked, .disp = offset}));
        }
        break;
      case Decoder::LoadStoreSize::k8bit:
        if (is_signed) {
          if (is_64bit_target) {
            res = std::get<0>(Gen<x86_64::MovsxbqRegOp>({.base = masked, .disp = offset}));
          } else {
            res = std::get<0>(Gen<x86_64::MovsxblRegOp>({.base = masked, .disp = offset}));
          }
        } else {
          res = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = masked, .disp = offset}));
        }
        break;
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
    GenRecoveryBlockForLastInsn();
    return res;
  }

  // Integer store of the low `size` bytes of `data` to {ApplyTbi(base), offset}.
  // Mirrors lite_translator.h::Store. GenRecoveryBlockForLastInsn() exits the
  // region on a host fault so the guest signal handler runs.
  void Store(Decoder::LoadStoreSize size, Register base, int32_t offset, Register data) {
    if (!success()) {
      return;
    }
    Register masked = ApplyTbi(base);
    switch (size) {
      case Decoder::LoadStoreSize::k64bit:
        Gen<x86_64::MovqOpReg>({.base = masked, .disp = offset}, data);
        break;
      case Decoder::LoadStoreSize::k32bit:
        Gen<x86_64::MovlOpReg>({.base = masked, .disp = offset}, data);
        break;
      case Decoder::LoadStoreSize::k16bit:
        Gen<x86_64::MovwOpReg>({.base = masked, .disp = offset}, data);
        break;
      case Decoder::LoadStoreSize::k8bit:
        Gen<x86_64::MovbOpReg>({.base = masked, .disp = offset}, data);
        break;
      default:
        UndefinedReturningVoid();
        return;
    }
    GenRecoveryBlockForLastInsn();
  }

  // Plain 64-bit add of an immediate (used for address computation). Always
  // non-flag-setting. Mirrors lite_translator.h::AddImm.
  Register AddImm(Register base, int32_t offset) {
    // A prior callback (e.g. a post-index Load) may have already bailed; emit
    // nothing so we don't append IR after the region-exit terminator.
    if (!success()) {
      return AllocTempReg();
    }
    Register res = Copy(base);
    if (offset != 0) {
      res = std::get<0>(Gen<x86_64::AddqRegImm, kNoSSA>(res, offset));
    }
    return res;
  }

  // LDP/LDPSW: load two adjacent sized elements at {base, 0} and {base, scale}.
  // The SemanticsPlayer passes rt1/rt2 as register NUMBERS and applies any
  // pre/post-index base writeback itself; this method commits the two loaded
  // values to the destination registers. Mirrors lite_translator.h::LoadPair:
  // both halves are loaded into temps FIRST, then committed via SetReg, so when
  // a destination aliases the base register (e.g. `ldp x0, x8, [x0]`) the second
  // load still reads from the original base. LDPSW (is_signed) sign-extends each
  // 32-bit element into its 64-bit target.
  void LoadPair(Decoder::LoadStoreSize size,
                Register base,
                int32_t offset,
                uint8_t rt1,
                uint8_t rt2,
                uint8_t scale,
                bool is_signed) {
    if (!success()) {
      return;
    }
    UNUSED_ARGS(offset);  // Already applied by the SemanticsPlayer.
    bool is_64bit_target = (size == Decoder::LoadStoreSize::k64bit) || is_signed;
    Register val1 = Load(size, is_signed, is_64bit_target, base, 0);
    if (!success()) {
      return;
    }
    Register val2 = Load(size, is_signed, is_64bit_target, base, static_cast<int32_t>(scale));
    if (!success()) {
      return;
    }
    if (rt1 != 31) {
      SetReg(rt1, val1);
    }
    if (rt2 != 31) {
      SetReg(rt2, val2);
    }
  }

  // STP: store two adjacent sized elements at {base, 0} and {base, scale}.
  // The SemanticsPlayer applies any pre/post-index base writeback itself.
  // Mirrors lite_translator.h::StorePair.
  void StorePair(Decoder::LoadStoreSize size,
                 Register base,
                 int32_t offset,
                 Register data1,
                 Register data2,
                 uint8_t scale) {
    if (!success()) {
      return;
    }
    UNUSED_ARGS(offset);  // Already applied by the SemanticsPlayer.
    Store(size, base, 0, data1);
    if (!success()) {
      return;
    }
    Store(size, base, static_cast<int32_t>(scale), data2);
  }

  // LDR (register offset): address = base + extend(offset_reg) << shift_amount,
  // then the same TBI + sized/sign-appropriate access + recovery as Load().
  // Mirrors lite_translator.h::LoadReg (+ ApplyOffsetExtend). The decoder only
  // emits the word-or-larger options (010=UXTW, 011=LSL/UXTX, 110=SXTW,
  // 111=SXTX); any other option was already rejected as UNDEFINED.
  Register LoadReg(Decoder::LoadStoreSize size,
                   bool is_signed,
                   bool is_64bit_target,
                   Register base,
                   Register offset_reg,
                   uint8_t extend_type,
                   uint8_t shift_amount) {
    if (!success()) {
      return AllocTempReg();
    }
    Register addr = EmitRegOffsetAddr(base, offset_reg, extend_type, shift_amount);
    return Load(size, is_signed, is_64bit_target, addr, 0);
  }

  // STR (register offset): mirrors lite_translator.h::StoreReg (+
  // ApplyOffsetExtend).
  void StoreReg(Decoder::LoadStoreSize size,
                Register base,
                Register offset_reg,
                uint8_t extend_type,
                uint8_t shift_amount,
                Register data) {
    if (!success()) {
      return;
    }
    Register addr = EmitRegOffsetAddr(base, offset_reg, extend_type, shift_amount);
    Store(size, addr, 0, data);
  }

  // LDXR/STXR/LDAXR/STLXR (exclusive) and LDAR/STLR (acquire/release). Defined
  // in the .cc (STXR needs basic-block manipulation for the reservation-address
  // check and CMPXCHG status branch). Mirrors lite_translator.h::LoadStoreExclusive
  // exactly: LDXR records cpu.reservation_address + the 64-bit cpu.reservation_value;
  // STXR re-checks the address, does a sized LOCK CMPXCHG against the saved value,
  // and writes status 0 (success) / 1 (fail) to Rs. Acquire/release are free on
  // x86-TSO. The LSE atomics / pair / CAS / SWP ops in args.op still bail.
  void LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& args, Register base);

  //
  // System.
  //

  void Svc(uint16_t imm) {
    UndefinedReturningVoid();
    UNUSED_ARGS(imm);
  }

  void Brk(uint16_t imm) {
    UndefinedReturningVoid();
    UNUSED_ARGS(imm);
  }

  // MRS Xt, TPIDR_EL0: read the guest thread-local-storage pointer from
  // ThreadState.tls. This is the read the bionic stack-canary prologue and all
  // TLS accesses issue, so it appears in nearly every real function; without it
  // the heavy tier bails on almost all real-app code. Other system registers
  // (NZCV, the CPU-detect MIDR/ID regs, FPSR) bail to the lite tier, which
  // handles them. Mirrors lite_translator.h::Mrs for the TPIDR_EL0 case.
  Register Mrs(Decoder::SystemReg sysreg) {
    if (sysreg == Decoder::SystemReg::kTpidrEl0) {
      Register res = AllocTempReg();
      if (success()) {
        builder_.GenGet(res, static_cast<int32_t>(offsetof(ThreadState, tls)));
      }
      return res;
    }
    UndefinedReturningReg();
    UNUSED_ARGS(sysreg);
    return AllocTempReg();
  }

  void Msr(Decoder::SystemReg sysreg, Register src) {
    UndefinedReturningVoid();
    UNUSED_ARGS(sysreg, src);
  }

  void IcIvau(uint8_t rt) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rt);
  }

  //
  // Register-form data processing.
  //

  // AND/ORR/EOR/BIC/ORN/EON (shifted register), including ANDS/BICS/TST. `invert`
  // selects the BIC/ORN/EON variants (src2 is bitwise-inverted before the op).
  // x86 AND clears CF and OF, so ANDS materializes NZCV with C=0 and V=0.
  // Mirrors lite_translator.h::LogicalShiftedReg.
  Register LogicalShiftedReg(Decoder::LogicalShiftedRegOpcode opcode,
                             bool is_64bit,
                             bool invert,
                             Register src1,
                             Register src2,
                             Decoder::ShiftType shift_type,
                             uint8_t shift_amount) {
    if (!success()) {
      return AllocTempReg();
    }
    Register op2 = EmitShiftImm(src2, shift_type, shift_amount, is_64bit);
    if (invert) {
      // NotqReg is 64-bit only; for the 32-bit case the upper half is re-cleared
      // by the subsequent 32-bit op, so a 64-bit NOT is safe.
      op2 = std::get<0>(Gen<x86_64::NotqReg, kNoSSA>(op2));
    }
    if (is_64bit) {
      Register res = Copy(src1);
      switch (opcode) {
        case Decoder::LogicalShiftedRegOpcode::kAnd:
          return std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(res, op2));
        case Decoder::LogicalShiftedRegOpcode::kAnds: {
          // ANDS/BICS/TST: x86 AND clears CF and OF, so ARM64 C=0 and V=0.
          auto [r, flags] = Gen<x86_64::AndqRegReg, kNoSSA>(res, op2);
          EmitMaterializeNZCV(flags, /*is_sub=*/false);
          return r;
        }
        case Decoder::LogicalShiftedRegOpcode::kOrr:
          return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(res, op2));
        case Decoder::LogicalShiftedRegOpcode::kEor:
          return std::get<0>(Gen<x86_64::XorqRegReg, kNoSSA>(res, op2));
        default:
          UndefinedReturningReg();
          return AllocTempReg();
      }
    }
    Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
    switch (opcode) {
      case Decoder::LogicalShiftedRegOpcode::kAnd:
        return std::get<0>(Gen<x86_64::AndlRegReg, kNoSSA>(res, op2));
      case Decoder::LogicalShiftedRegOpcode::kAnds: {
        auto [r, flags] = Gen<x86_64::AndlRegReg, kNoSSA>(res, op2);
        EmitMaterializeNZCV(flags, /*is_sub=*/false);
        return r;
      }
      case Decoder::LogicalShiftedRegOpcode::kOrr:
        return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(res, op2));
      case Decoder::LogicalShiftedRegOpcode::kEor:
        return std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(res, op2));
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }

  // ADD/SUB (shifted register), including the flag-setting ADDS/SUBS/CMP/CMN
  // forms (NZCV materialized via EmitMaterializeNZCV). ROR is not a valid shift
  // for add/sub and bails. Mirrors lite_translator.h::AddSubShiftedReg.
  Register AddSubShiftedReg(bool is_sub,
                            bool set_flags,
                            bool is_64bit,
                            Register src1,
                            Register src2,
                            Decoder::ShiftType shift_type,
                            uint8_t shift_amount) {
    if (!success()) {
      return AllocTempReg();
    }
    // Validate first; emit nothing on bail. ROR is not a valid shift for ADD/SUB.
    if (shift_type == Decoder::ShiftType::kRor) {
      UndefinedReturningReg();
      return AllocTempReg();
    }
    Register op2 = EmitShiftImm(src2, shift_type, shift_amount, is_64bit);
    if (is_64bit) {
      Register res = Copy(src1);
      if (is_sub) {
        auto [r, flags] = Gen<x86_64::SubqRegReg, kNoSSA>(res, op2);
        if (set_flags) {
          EmitMaterializeNZCV(flags, /*is_sub=*/true);
        }
        return r;
      }
      auto [r, flags] = Gen<x86_64::AddqRegReg, kNoSSA>(res, op2);
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/false);
      }
      return r;
    }
    Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
    if (is_sub) {
      auto [r, flags] = Gen<x86_64::SublRegReg, kNoSSA>(res, op2);
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/true);
      }
      return r;
    }
    auto [r, flags] = Gen<x86_64::AddlRegReg, kNoSSA>(res, op2);
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/false);
    }
    return r;
  }

  // ADD/SUB (extended register), including the flag-setting ADDS/SUBS/CMP/CMN
  // forms (NZCV materialized via EmitMaterializeNZCV). Mirrors
  // lite_translator.h::AddSubExtendedReg.
  Register AddSubExtendedReg(bool is_sub,
                             bool set_flags,
                             bool is_64bit,
                             Register src1,
                             Register src2,
                             uint8_t extend_type,
                             uint8_t shift_amount) {
    if (!success()) {
      return AllocTempReg();
    }
    // Validate first; emit nothing on bail.
    if (shift_amount > 4 || extend_type > 0b111) {
      UndefinedReturningReg();
      return AllocTempReg();
    }

    // Apply extension to src2.
    // extend_type: 000=UXTB 001=UXTH 010=UXTW 011=UXTX 100=SXTB 101=SXTH
    //              110=SXTW 111=SXTX.
    Register ext;
    switch (extend_type) {
      case 0b000:  // UXTB
        ext = std::get<0>(Gen<x86_64::MovzxblRegReg>(src2));
        break;
      case 0b001:  // UXTH
        ext = std::get<0>(Gen<x86_64::MovzxwlRegReg>(src2));
        break;
      case 0b010:  // UXTW: 32-bit mov zero-extends to 64.
        ext = std::get<0>(Gen<x86_64::MovlRegReg>(src2));
        break;
      case 0b011:  // UXTX: no extension.
        ext = Copy(src2);
        break;
      case 0b100:  // SXTB
        if (is_64bit) {
          ext = std::get<0>(Gen<x86_64::MovsxbqRegReg>(src2));
        } else {
          ext = std::get<0>(Gen<x86_64::MovsxblRegReg>(src2));
        }
        break;
      case 0b101:  // SXTH
        if (is_64bit) {
          ext = std::get<0>(Gen<x86_64::MovsxwqRegReg>(src2));
        } else {
          ext = std::get<0>(Gen<x86_64::MovsxwlRegReg>(src2));
        }
        break;
      case 0b110:  // SXTW
        if (is_64bit) {
          ext = std::get<0>(Gen<x86_64::MovsxlqRegReg>(src2));
        } else {
          ext = std::get<0>(Gen<x86_64::MovlRegReg>(src2));
        }
        break;
      default:  // 0b111 SXTX: no extension.
        ext = Copy(src2);
        break;
    }

    // Apply shift.
    if (shift_amount > 0) {
      if (is_64bit) {
        ext = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(ext, static_cast<int8_t>(shift_amount)));
      } else {
        ext = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(ext, static_cast<int8_t>(shift_amount)));
      }
    }

    if (is_64bit) {
      Register res = Copy(src1);
      if (is_sub) {
        auto [r, flags] = Gen<x86_64::SubqRegReg, kNoSSA>(res, ext);
        if (set_flags) {
          EmitMaterializeNZCV(flags, /*is_sub=*/true);
        }
        return r;
      }
      auto [r, flags] = Gen<x86_64::AddqRegReg, kNoSSA>(res, ext);
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/false);
      }
      return r;
    }
    Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
    if (is_sub) {
      auto [r, flags] = Gen<x86_64::SublRegReg, kNoSSA>(res, ext);
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/true);
      }
      return r;
    }
    auto [r, flags] = Gen<x86_64::AddlRegReg, kNoSSA>(res, ext);
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/false);
    }
    return r;
  }

  // CSEL/CSINC/CSINV/CSNEG Rd, Rn, Rm, cond:
  //   Rd = cond ? Rn : transform(Rm)
  // where transform is identity (CSEL), +1 (CSINC), ~ (CSINV), or - (CSNEG).
  // Mirrors lite_translator.h::ConditionalSelect: materialize the false case
  // (transform(Rm)) into a result register, then conditionally overwrite it with
  // Rn when the condition holds. Structured with then/merge basic blocks (the
  // overwrite happens in then_bb, both paths fall into merge_bb) like BranchCond,
  // since the heavy IR has no condition-immediate CMOV adapter. AL/NV always
  // select Rn. 32-bit forms zero-extend. Defined in the .cc (needs basic-block
  // manipulation). Returns the result register.
  Register ConditionalSelect(Decoder::ConditionalSelectOpcode opcode,
                             bool is_64bit,
                             Register src1,
                             Register src2,
                             Decoder::Condition cond);

  // UDIV/SDIV. Defined in the .cc: ARM division never traps, so these wrap the
  // fixed-RDX:RAX x86 DIV/IDIV pseudo-op in guard blocks. UDIV/SDIV with Rm==0
  // return 0; SDIV INT_MIN/-1 returns INT_MIN (x86 IDIV would #DE on both).
  Register EmitUDiv(bool is_64bit, Register src1, Register src2);
  Register EmitSDiv(bool is_64bit, Register src1, Register src2);

  // LSLV/LSRV/ASRV/RORV (variable shifts) and UDIV/SDIV. CRC32/PACGA still bail.
  // Division routes to the .cc EmitUDiv/EmitSDiv helpers because the ARM
  // divide-by-zero (Rd=0) and SDIV INT_MIN/-1 (Rd=INT_MIN) guards need basic
  // blocks around the fixed-RDX:RAX x86 DIV/IDIV pseudo-op. The variable shifts
  // use the x86 shift-by-CL forms; the backend register allocator binds the
  // count operand to RCX automatically.
  Register DataProc2Src(Decoder::DataProc2SrcOpcode opcode,
                        bool is_64bit,
                        Register src1,
                        Register src2) {
    if (!success()) {
      return AllocTempReg();
    }
    switch (opcode) {
      case Decoder::DataProc2SrcOpcode::kLslv:
        if (is_64bit) {
          return std::get<0>(Gen<x86_64::ShlqRegReg>(src1, src2));
        }
        return std::get<0>(Gen<x86_64::ShllRegReg>(src1, src2));
      case Decoder::DataProc2SrcOpcode::kLsrv:
        if (is_64bit) {
          return std::get<0>(Gen<x86_64::ShrqRegReg>(src1, src2));
        }
        return std::get<0>(Gen<x86_64::ShrlRegReg>(src1, src2));
      case Decoder::DataProc2SrcOpcode::kAsrv:
        if (is_64bit) {
          return std::get<0>(Gen<x86_64::SarqRegReg>(src1, src2));
        }
        return std::get<0>(Gen<x86_64::SarlRegReg>(src1, src2));
      case Decoder::DataProc2SrcOpcode::kRorv:
        if (is_64bit) {
          return std::get<0>(Gen<x86_64::RorqRegReg>(src1, src2));
        }
        return std::get<0>(Gen<x86_64::RorlRegReg>(src1, src2));
      case Decoder::DataProc2SrcOpcode::kUdiv:
        return EmitUDiv(is_64bit, src1, src2);
      case Decoder::DataProc2SrcOpcode::kSdiv:
        return EmitSDiv(is_64bit, src1, src2);
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }

  // MADD/MSUB and the signed/unsigned widening multiply-accumulates
  // (SMADDL/SMSUBL/UMADDL/UMSUBL), plus SMULH/UMULH (high 64 bits of a 64x64
  // product via the widening x86 IMUL/MUL into RDX:RAX). Mirrors
  // lite_translator.h::DataProc3Src.
  //   MADD:  Rd = Ra + Rn * Rm     MSUB:  Rd = Ra - Rn * Rm
  //   SMADDL: Xd = Xa + sext(Wn)*sext(Wm)   (UMADDL uses zext; *SUBL subtracts)
  Register DataProc3Src(Decoder::DataProc3SrcOpcode opcode,
                        bool is_64bit,
                        Register src1,
                        Register src2,
                        Register src3) {
    if (!success()) {
      return AllocTempReg();
    }
    switch (opcode) {
      case Decoder::DataProc3SrcOpcode::kMadd: {
        if (is_64bit) {
          Register prod = std::get<0>(Gen<x86_64::ImulqRegReg>(src1, src2));
          return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(prod, src3));
        }
        Register prod = std::get<0>(Gen<x86_64::ImullRegReg>(src1, src2));
        return std::get<0>(Gen<x86_64::AddlRegReg, kNoSSA>(prod, src3));
      }
      case Decoder::DataProc3SrcOpcode::kMsub: {
        if (is_64bit) {
          Register prod = std::get<0>(Gen<x86_64::ImulqRegReg>(src1, src2));
          Register res = Copy(src3);
          return std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(res, prod));
        }
        Register prod = std::get<0>(Gen<x86_64::ImullRegReg>(src1, src2));
        Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src3));
        return std::get<0>(Gen<x86_64::SublRegReg, kNoSSA>(res, prod));
      }
      case Decoder::DataProc3SrcOpcode::kSmaddl:
      case Decoder::DataProc3SrcOpcode::kSmsubl: {
        // Sign-extend both 32-bit sources to 64-bit, then 64-bit multiply.
        Register ext1 = std::get<0>(Gen<x86_64::MovsxlqRegReg>(src1));
        Register ext2 = std::get<0>(Gen<x86_64::MovsxlqRegReg>(src2));
        Register prod = std::get<0>(Gen<x86_64::ImulqRegReg, kNoSSA>(ext1, ext2));
        Register res = Copy(src3);
        if (opcode == Decoder::DataProc3SrcOpcode::kSmaddl) {
          return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(res, prod));
        }
        return std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(res, prod));
      }
      case Decoder::DataProc3SrcOpcode::kUmaddl:
      case Decoder::DataProc3SrcOpcode::kUmsubl: {
        // Zero-extend both 32-bit sources to 64-bit (Movl), then 64-bit multiply.
        Register ext1 = std::get<0>(Gen<x86_64::MovlRegReg>(src1));
        Register ext2 = std::get<0>(Gen<x86_64::MovlRegReg>(src2));
        Register prod = std::get<0>(Gen<x86_64::ImulqRegReg, kNoSSA>(ext1, ext2));
        Register res = Copy(src3);
        if (opcode == Decoder::DataProc3SrcOpcode::kUmaddl) {
          return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(res, prod));
        }
        return std::get<0>(Gen<x86_64::SubqRegReg, kNoSSA>(res, prod));
      }
      case Decoder::DataProc3SrcOpcode::kUmulh:
        // UMULH: Xd = high 64 bits of (Xn * Xm), unsigned. MulqRegRegReg returns
        // [low(RAX), high(RDX), flags]; the high half is the result. X-form only.
        return std::get<1>(Gen<x86_64::MulqRegRegReg>(src1, src2));
      case Decoder::DataProc3SrcOpcode::kSmulh:
        // SMULH: Xd = high 64 bits of (Xn * Xm), signed (ImulqRegRegReg). X-form only.
        return std::get<1>(Gen<x86_64::ImulqRegRegReg>(src1, src2));
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }

  // ADC/ADCS/SBC/SBCS (add/subtract with the guest carry flag). Mirrors
  // lite_translator.h::AddSubWithCarry: the guest carry (ARM NZCV C =
  // cpu.flags bit 8) is loaded into x86 CF, then one x86 ADC/SBB does the
  // carry op.
  //   ADC — ARM C maps directly to x86 CF: BT bit 8 sets CF, then ADC.
  //   SBC — x86 SBB computes src1 - src2 - CF = src1 + ~src2 + (1 - CF), while
  //         ARM SBC wants src1 + ~src2 + C, so SBB needs CF = !C (CMC after the
  //         BT). On output x86 SBB's CF is the borrow; EmitMaterializeNZCV
  //         (is_sub=true) inverts it back to ARM's "carry = no borrow".
  // res is a fresh temp; the whole MOV/BT/CMC/ADC(SBB) chain stays in one basic
  // block so the host FLAGS never cross a BB boundary. MOV does not touch
  // FLAGS, so the carry loaded by BT survives to the ADC/SBB.
  Register AddSubWithCarry(Register src1, Register src2, bool is_64bit, bool is_sub, bool set_flags) {
    if (!success()) {
      return AllocTempReg();
    }
    // res = src1 (32-bit MOV for sf=0 so the later 32-bit ADC/SBB zero-extends).
    Register res = is_64bit ? Copy(src1) : std::get<0>(Gen<x86_64::MovlRegReg>(src1));
    // Load the guest carry into x86 CF. MovwRegOp matches the 16-bit flags-read
    // opcode RemoveLoopGuestContextAccesses recognizes; only bit 8 is tested, so
    // the untouched upper bits of the loaded word do not matter.
    const int32_t flags_disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags));
    Register armflags =
        std::get<0>(Gen<x86_64::MovwRegOp>({.base = x86_64::kMachineRegRBP, .disp = flags_disp}));
    Gen<x86_64::BtqRegImm>(armflags, static_cast<int8_t>(8));  // CF = bit 8 = ARM carry.
    if (is_sub) {
      Gen<x86_64::Cmc>(GetFlagsRegister());  // SBB wants CF = !(ARM carry).
    }
    if (is_64bit) {
      if (is_sub) {
        auto [r, flags] = Gen<x86_64::SbbqRegReg, kNoSSA>(res, src2, GetFlagsRegister());
        if (set_flags) {
          EmitMaterializeNZCV(flags, /*is_sub=*/true);
        }
        return r;
      }
      auto [r, flags] = Gen<x86_64::AdcqRegReg, kNoSSA>(res, src2, GetFlagsRegister());
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/false);
      }
      return r;
    }
    if (is_sub) {
      auto [r, flags] = Gen<x86_64::SbblRegReg, kNoSSA>(res, src2, GetFlagsRegister());
      if (set_flags) {
        EmitMaterializeNZCV(flags, /*is_sub=*/true);
      }
      return r;
    }
    auto [r, flags] = Gen<x86_64::AdclRegReg, kNoSSA>(res, src2, GetFlagsRegister());
    if (set_flags) {
      EmitMaterializeNZCV(flags, /*is_sub=*/false);
    }
    return r;
  }

  // RBIT/REV16/REV32/REV/CLZ/CLS. REV16/REV32/REV are byte-reversed with SWAR
  // shift/mask/or sequences (no x86 BSWAP MachineIR op). CLZ maps to LZCNT and
  // CLS to LZCNT(x ^ (x>>1))-1 (both gated on host LZCNT). RBIT has no x86
  // mapping and bails to the lite translator/interpreter. PAuth DP-1Src variants
  // (opcode2 bit 0x40) are treated as identity because Digitalis is PAC-blind.
  // Mirrors lite_translator.h::DataProc1Src.
  Register DataProc1Src(Register src, uint8_t opcode2, bool is_64bit) {
    if (!success()) {
      return AllocTempReg();
    }
    // PAuth DP-1Src: identity copy (the upper-half clear of a 32-bit mov handles
    // the sf=0 zero-extend; PAuth ops are X-form, so the Movq branch is taken).
    if (opcode2 & 0x40) {
      if (is_64bit) {
        return Copy(src);
      }
      return std::get<0>(Gen<x86_64::MovlRegReg>(src));
    }
    switch (opcode2) {
      case 0b000001: {  // REV16: reverse byte order within each 16-bit halfword.
        // res = ((src & lo_mask) << 8) | ((src & hi_mask) >> 8).
        if (is_64bit) {
          Register lo = Copy(src);
          Register lo_mask = GetImm(0x00FF00FF00FF00FFULL);
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, lo_mask));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{8}));
          Register hi = Copy(src);
          Register hi_mask = GetImm(0xFF00FF00FF00FF00ULL);
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, hi_mask));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{8}));
          return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        }
        Register lo = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(lo, 0x00FF00FF));
        lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{8}));
        Register hi = std::get<0>(Gen<x86_64::MovlRegReg>(src));
        hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, static_cast<int32_t>(0xFF00FF00)));
        hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(hi, int8_t{8}));
        return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
      }
      case 0b000010:  // REV32 (sf=1) / REV (sf=0): byte-reverse each 32-bit word.
        // No x86 BSWAP MachineIR op, so byte-swap with the SWAR shift/mask/or
        // sequence: swap bytes within each halfword, then swap halves within each
        // 32-bit word. For the X-form this byte-reverses each 32-bit word in place
        // (no cross-word swap); for the W-form it byte-reverses the 32-bit value
        // and zero-extends.
        if (is_64bit) {
          Register x = Copy(src);
          // Swap bytes within each 16-bit halfword.
          Register lo = Copy(x);
          Register m1 = GetImm(0x00FF00FF00FF00FFULL);
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, m1));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{8}));
          Register hi = Copy(x);
          Register m2 = GetImm(0xFF00FF00FF00FF00ULL);
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, m2));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{8}));
          x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
          // Swap 16-bit halves within each 32-bit word.
          lo = Copy(x);
          Register m3 = GetImm(0x0000FFFF0000FFFFULL);
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, m3));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{16}));
          hi = Copy(x);
          Register m4 = GetImm(0xFFFF0000FFFF0000ULL);
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, m4));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{16}));
          return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        } else {
          // REV Wd: byte-reverse the 32-bit word (zero-extends to 64).
          Register lo = std::get<0>(Gen<x86_64::MovlRegReg>(src));
          lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(lo, 0x00FF00FF));
          lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{8}));
          Register hi = std::get<0>(Gen<x86_64::MovlRegReg>(src));
          hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, static_cast<int32_t>(0xFF00FF00)));
          hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(hi, int8_t{8}));
          Register x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
          lo = Copy(x);
          lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{16}));
          hi = Copy(x);
          hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(hi, int8_t{16}));
          return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
        }
      case 0b000011:  // REV (64-bit full byte reverse).
        // SWAR byte-swap: halfword swap, then halfword-pair swap within words,
        // then 32-bit word swap. (No BSWAP MachineIR op.)
        if (is_64bit) {
          Register x = Copy(src);
          // Swap bytes within each 16-bit halfword.
          Register lo = Copy(x);
          Register m1 = GetImm(0x00FF00FF00FF00FFULL);
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, m1));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{8}));
          Register hi = Copy(x);
          Register m2 = GetImm(0xFF00FF00FF00FF00ULL);
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, m2));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{8}));
          x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
          // Swap 16-bit halves within each 32-bit word.
          lo = Copy(x);
          Register m3 = GetImm(0x0000FFFF0000FFFFULL);
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(lo, m3));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{16}));
          hi = Copy(x);
          Register m4 = GetImm(0xFFFF0000FFFF0000ULL);
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, m4));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{16}));
          x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
          // Swap the two 32-bit words.
          lo = Copy(x);
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{32}));
          hi = Copy(x);
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(hi, int8_t{32}));
          return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        }
        // REV is X-form only for opcode2=000011; sf=0 is not encoded. Bail safely.
        UndefinedReturningReg();
        return AllocTempReg();
      case 0b000100:  // CLZ (count leading zeros) maps directly to x86 LZCNT.
        // Without LZCNT the encoding decodes as BSR (wrong result for CLZ), so
        // bail to the lite translator (which uses a BSR-with-zero-check sequence).
        if (!host_platform::kHasLZCNT) {
          UndefinedReturningReg();
          return AllocTempReg();
        }
        if (is_64bit) {
          return std::get<0>(Gen<x86_64::LzcntqRegReg>(src));
        }
        return std::get<0>(Gen<x86_64::LzcntlRegReg>(src));
      case 0b000101:  // CLS (count leading sign bits).
        // CLS = LZCNT(x ^ (x >>arith 1)) - 1. LZCNT(0) == reg_size, so the
        // all-same-bits case yields reg_size-1 with no branch. Needs LZCNT for
        // the zero-input result; without it, bail to the lite/interp path.
        if (!host_platform::kHasLZCNT) {
          UndefinedReturningReg();
          return AllocTempReg();
        }
        if (is_64bit) {
          Register sar = Copy(src);
          sar = std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(sar, int8_t{1}));
          Register xored = Copy(src);
          xored = std::get<0>(Gen<x86_64::XorqRegReg, kNoSSA>(xored, sar));
          Register lz = std::get<0>(Gen<x86_64::LzcntqRegReg>(xored));
          return std::get<0>(Gen<x86_64::SubqRegImm, kNoSSA>(lz, int32_t{1}));
        } else {
          // 32-bit: Movl src into a clean 32-bit value first, Sarl/Lzcntl operate
          // over 32 bits and zero-extend the result to 64 (ARM64 W-write).
          Register clean = std::get<0>(Gen<x86_64::MovlRegReg>(src));
          Register sar = Copy(clean);
          sar = std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(sar, int8_t{1}));
          Register xored = Copy(clean);
          xored = std::get<0>(Gen<x86_64::XorlRegReg, kNoSSA>(xored, sar));
          Register lz = std::get<0>(Gen<x86_64::LzcntlRegReg>(xored));
          return std::get<0>(Gen<x86_64::SublRegImm, kNoSSA>(lz, int32_t{1}));
        }
      case 0b000000:  // RBIT: reverse bit order.
        // SWAR bit-reverse: swap adjacent bits, then bit-pairs, then nibbles
        // (each as ((x & m) << s) | ((x >> s) & m)), then reverse byte order
        // with the same shift/mask/or sequence as REV (no x86 BSWAP MachineIR op).
        if (is_64bit) {
          Register x = Copy(src);
          // Swap adjacent bits (mask 0x5555..., shift 1).
          Register lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x5555555555555555ULL)));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{1}));
          Register hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{1}));
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x5555555555555555ULL)));
          x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
          // Swap bit-pairs (mask 0x3333..., shift 2).
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x3333333333333333ULL)));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{2}));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{2}));
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x3333333333333333ULL)));
          x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
          // Swap nibbles (mask 0x0F0F..., shift 4).
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x0F0F0F0F0F0F0F0FULL)));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{4}));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{4}));
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x0F0F0F0F0F0F0F0FULL)));
          x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
          // Reverse byte order: halfword-byte swap, halfword-pair swap, word swap.
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x00FF00FF00FF00FFULL)));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{8}));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{8}));
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x00FF00FF00FF00FFULL)));
          x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
          lo = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(Copy(x), GetImm(0x0000FFFF0000FFFFULL)));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(lo, int8_t{16}));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{16}));
          hi = std::get<0>(Gen<x86_64::AndqRegReg, kNoSSA>(hi, GetImm(0x0000FFFF0000FFFFULL)));
          x = std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
          lo = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(Copy(x), int8_t{32}));
          hi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(Copy(x), int8_t{32}));
          return std::get<0>(Gen<x86_64::OrqRegReg, kNoSSA>(lo, hi));
        } else {
          // RBIT Wd: reverse the low 32 bits, zero-extend.
          Register x = std::get<0>(Gen<x86_64::MovlRegReg>(src));
          Register lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(x), 0x55555555));
          lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{1}));
          Register hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{1}));
          hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, 0x55555555));
          x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
          lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(x), 0x33333333));
          lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{2}));
          hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{2}));
          hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, 0x33333333));
          x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
          lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(x), 0x0F0F0F0F));
          lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{4}));
          hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{4}));
          hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, 0x0F0F0F0F));
          x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
          // Byte-reverse the 32-bit word: halfword-byte swap then halfword swap.
          lo = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(Copy(x), 0x00FF00FF));
          lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(lo, int8_t{8}));
          hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{8}));
          hi = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(hi, 0x00FF00FF));
          x = std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
          lo = std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(Copy(x), int8_t{16}));
          hi = std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(Copy(x), int8_t{16}));
          return std::get<0>(Gen<x86_64::OrlRegReg, kNoSSA>(lo, hi));
        }
      default:
        // Any remaining unhandled DP-1Src opcode: fall back to the interpreter.
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }

  // EXTR Rd, Rn, Rm, #lsb: Rd = (Rn:Rm) >> lsb. lsb==0 is a copy of Rm. Both the
  // 32-bit and 64-bit non-zero cases map to x86 SHRD (the ROR Rn==Rm alias falls
  // out for free). Mirrors lite_translator.h::Extr.
  Register Extr(Register src_n, Register src_m, uint8_t lsb, bool is_64bit) {
    if (!success()) {
      return AllocTempReg();
    }
    if (lsb == 0) {
      if (is_64bit) {
        return Copy(src_m);
      }
      return std::get<0>(Gen<x86_64::MovlRegReg>(src_m));
    }
    // SHRD dest, src, imm: dest = (src:dest) >> imm.
    // ARM EXTR Rd = (Rn:Rm) >> lsb = SHRD(Rm, Rn, lsb): dest=Rm, src=Rn.
    if (is_64bit) {
      Register res = Copy(src_m);
      return std::get<0>(Gen<x86_64::ShrdqRegRegImm, kNoSSA>(res, src_n, static_cast<int8_t>(lsb)));
    }
    Register res = std::get<0>(Gen<x86_64::MovlRegReg>(src_m));
    return std::get<0>(Gen<x86_64::ShrdlRegRegImm, kNoSSA>(res, src_n, static_cast<int8_t>(lsb)));
  }

  // CCMP/CCMN Rn, Rm, #nzcv, cond:
  //   if cond holds: NZCV = flags of (Rn - Rm) [CMP] or (Rn + Rm) [CMN]
  //   else:          NZCV = the 4-bit nzcv immediate (bit3=N,bit2=Z,bit1=C,bit0=V)
  // Mirrors lite_translator.cc::ConditionalCompare: branch on the condition
  // predicate to a compare-path bb (EmitMaterializeNZCV) vs an immediate-path bb
  // (writes the packed nzcv to cpu.flags), then merge. Defined in the .cc (needs
  // basic-block manipulation). AL/NV always take the compare path.
  void ConditionalCompare(bool is_neg,
                          bool is_64bit,
                          Register rn,
                          Register rm,
                          Decoder::Condition cond,
                          uint8_t nzcv);

  //
  // MTE.
  //

  void MteDataProc(const Decoder::MteDataProcArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void MteLoadStore(const Decoder::MteLoadStoreArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  //
  // Floating-point scalar.
  //

  // FCSEL Sd|Dd, Sn|Dn, Sm|Dm, cond:
  //   V[rd] = ZeroExtend(ConditionHolds(cond) ? V[rn] : V[rm]).
  // Lowered branchlessly (no basic-block manipulation, unlike the lite tier's
  // Jcc form) so the whole scalar select stays inside one machine BB: build a
  // full-width 0/-1 mask from the ARM condition predicate and blend the two
  // scalars with PAND/PANDN/POR. Mirrors the interpreter's FpCondSelect
  // (read the chosen source, zero-extend into V[rd]) semantically.
  //
  //   pred    = EmitArmCondPredicate(cond)   // 0/1, 1 iff cond holds
  //   mask_gp = 0 - pred                      // cond ? 0xFFFF..FFFF : 0
  //   mask    = MOVQ(mask_gp)                 // low 64 bits carry the mask
  //   vn &= mask ;  mask = ~mask & vm ;  vn |= mask   // vn = cond ? Vn : Vm
  //   SetVRegScalar(rd, vn)                   // zero-extends the low lane
  //
  // GetVRegScalar returns a fresh MOVSD-loaded temp (upper lanes zero), so the
  // two operands can be mutated in place. Only the low scalar lane needs a
  // correct mask (SetVRegScalar re-zeroes the upper bytes), so a low-64 mask
  // covers both S and D. ftype 0b11 (FP16) and reserved 0b10 bail to the lite
  // tier, whose intrinsics cover them (matches FpDataProc3's FP16 bail).
  void FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ftype, Decoder::Condition cond) {
    if (!success()) {
      return;
    }
    // Bail to the lite tier. The branchless mask-blend lowering (from the
    // FCSEL heavy commit) miscompiled inside real regions — it passed every
    // isolated per-op exec test but deterministically crashed the Chromium
    // renderer (bisected to that commit; heavy-off rendered cleanly). The
    // arithmetic and the EmitArmCondPredicate mask look correct in isolation,
    // so the fault is a region-level interaction the single-region tests don't
    // exercise; until it's reproduced and fixed with a region-level
    // differential test, FCSEL stays a heavy bail (correct-but-slow — the
    // interpreter and lite tier handle it). Do NOT re-enable a heavy FCSEL
    // lowering without a region-interaction test that reproduces the crash.
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, ftype, cond);
  }

  // FCVTZS/FCVTZU/SCVTF/UCVTF (fixed-point). The FCvt* intrinsics + cvtsi2ss
  // SSE ops are not in the ARM64 backend gen inputs (their macro-assembler defs
  // live only in riscv64_to_x86_64/macro_def.json), so bail.
  void FpFixedPointConversion(const Decoder::FpFixedPointArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  // FMADD/FMSUB/FNMADD/FNMSUB (FP data-processing, 3 source) at S/D. Lowered to
  // the x86 FMA3 231-form ops, which — like ARM's fused multiply-add — round the
  // whole a+n*m once. A plain MUL+ADD would double-round and is wrong, so a host
  // without FMA3 bails to the lite tier (which uses libc fma()/fmaf()). Mirrors
  // lite_translator.h::FpDataProc3's S/D paths. The FMA231 op is use_def on its
  // dest (the accumulator = Ra), so copy Ra into a fresh temp first to avoid
  // clobbering the guest V[ra] mapping:
  //   FMADD  (o1=0,o0=0) Ra + Rn*Rm     -> Vfmadd231  (acc + Rn*Rm)
  //   FMSUB  (o1=0,o0=1) Ra - Rn*Rm     -> Vfnmadd231 (acc + -(Rn*Rm))
  //   FNMADD (o1=1,o0=0) -(Ra + Rn*Rm)  -> Vfnmsub231 (-(Rn*Rm) - acc)
  //   FNMSUB (o1=1,o0=1) Rn*Rm - Ra     -> Vfmsub231  (Rn*Rm - acc)
  // FP16 (ftype=0b11, needs F16C widen/narrow ops not in the backend gen inputs)
  // and the reserved ftype=0b10 bail to the lite tier.
  void FpDataProc3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra, uint8_t ftype, bool o1, bool o0) {
    if (!success()) {
      return;
    }
    if (ftype != 0b00 && ftype != 0b01) {
      UndefinedReturningVoid();
      return;
    }
    if (!host_platform::kHasFMA) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_double = (ftype == 0b01);
    FpRegister xmm_n = GetVRegScalar(rn, is_double);
    FpRegister xmm_m = GetVRegScalar(rm, is_double);
    FpRegister xmm_a = GetVRegScalar(ra, is_double);
    FpRegister acc = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdqaXRegXReg>(acc.machine_reg(), xmm_a.machine_reg());
    if (!o1 && !o0) {  // FMADD
      if (is_double) {
        builder_.Gen<x86_64::Vfmadd231sdXRegXRegXReg>(
            acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
      } else {
        builder_.Gen<x86_64::Vfmadd231ssXRegXRegXReg>(
            acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
      }
    } else if (!o1 && o0) {  // FMSUB
      if (is_double) {
        builder_.Gen<x86_64::Vfnmadd231sdXRegXRegXReg>(
            acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
      } else {
        builder_.Gen<x86_64::Vfnmadd231ssXRegXRegXReg>(
            acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
      }
    } else if (o1 && !o0) {  // FNMADD
      if (is_double) {
        builder_.Gen<x86_64::Vfnmsub231sdXRegXRegXReg>(
            acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
      } else {
        builder_.Gen<x86_64::Vfnmsub231ssXRegXRegXReg>(
            acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
      }
    } else {  // FNMSUB
      if (is_double) {
        builder_.Gen<x86_64::Vfmsub231sdXRegXRegXReg>(
            acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
      } else {
        builder_.Gen<x86_64::Vfmsub231ssXRegXRegXReg>(
            acc.machine_reg(), xmm_n.machine_reg(), xmm_m.machine_reg());
      }
    }
    SetVRegScalar(rd, acc, is_double);
  }

  // FMOV (scalar, immediate): the FP constant is fully known at translation
  // time (VFPExpandImm of imm8). Zero the 16-byte V[d] slot, then store the
  // 32/64-bit constant into lane 0. Mirrors lite_translator.h::FpMovImmediate;
  // bail FP16 (ftype=0b11) and the reserved ftype=0b10.
  void FpMovImmediate(uint8_t rd, uint8_t imm8, uint8_t ftype) {
    if (!success()) {
      return;
    }
    if (ftype != 0b00 && ftype != 0b01) {
      UndefinedReturningVoid();
      return;
    }
    if (ftype == 0b00) {
      uint32_t imm32 = VFPExpandImm32(imm8);
      Register tmp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(imm32)));
      SetVRegScalarFromGp(rd, tmp, /*is_double=*/false);
    } else {
      uint64_t imm64 = VFPExpandImm64(imm8);
      Register tmp = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(imm64)));
      SetVRegScalarFromGp(rd, tmp, /*is_double=*/true);
    }
  }

  // FMOV between a general register and a scalar FP register, single (S/W) or
  // double (D/X), via x86 MOVD/MOVQ; plus SCVTF/UCVTF (integer -> FP) via the
  // x86 CVTSI2SS/SD ops (EmitScvtfUcvtf) and FCVTZS/FCVTZU (FP -> int, truncate
  // toward zero) via CVTT{SS,SD}2SI + the ARM saturation/NaN fix-up (EmitFcvtz).
  // The rmode == 01 top-half (V.D[1]) forms, the rounding FP -> int conversions
  // (FCVTNS/PS/MS/AS/...), FP16, and ftype >= 0b10 bail to the lite tier, whose
  // intrinsics cover them.
  // Guest V[] access stays in the XMM domain (GetVRegScalar / SetVRegScalar*),
  // and the GP<->XMM crossing is an explicit register move, not a forwarded
  // guest-context GET. Mirrors lite_translator.h::FpIntConversion (FMOV +
  // SCVTF/UCVTF subset).
  void FpIntConversion(const Decoder::FpIntConvArgs& args) {
    if (!success()) {
      return;
    }
    if (args.ftype != 0b00 && args.ftype != 0b01) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_double = (args.ftype == 0b01);
    if (args.rmode == 0b00 && (args.op == 0b110 || args.op == 0b111)) {
      if (args.op == 0b111) {
        // FMOV Sd, Wn / Dd, Xn: general register -> scalar FP (upper lanes zeroed).
        if (args.rn == 31) {
          SetVRegScalar(args.rd, AllocZeroedSimdReg(), is_double);  // WZR/XZR -> 0
        } else {
          SetVRegScalarFromGp(args.rd, GetReg(args.rn), is_double);
        }
      } else {
        // FMOV Wd, Sn / Xd, Dn: scalar FP -> general register.
        if (args.rd == 31) {
          return;  // WZR/XZR destination: discard.
        }
        FpRegister xmm = GetVRegScalar(args.rn, is_double);
        Register gp = is_double
                          ? std::get<0>(Gen<x86_64::MovqRegXReg>(xmm.machine_reg()))
                          : std::get<0>(Gen<x86_64::MovdRegXReg>(xmm.machine_reg()));
        SetReg(args.rd, gp);
      }
      return;
    }
    // SCVTF (op 010) / UCVTF (op 011): integer -> FP, unscaled (rmode == 00).
    if (args.rmode == 0b00 && (args.op == 0b010 || args.op == 0b011)) {
      EmitScvtfUcvtf(args, is_double);
      return;
    }
    // FCVTZS (op 000) / FCVTZU (op 001): FP -> int, truncate (rmode == 11).
    if (args.rmode == 0b11 && (args.op == 0b000 || args.op == 0b001)) {
      EmitFcvtz(args, is_double);
      return;
    }
    // Rounding FP -> int conversions FCVTNS/NU (rmode 00, round-to-nearest
    // ties-even), FCVTPS/PU (rmode 01, toward +inf) and FCVTMS/MU (rmode 10,
    // toward -inf). ARM ties-away FCVTAS/AU (op 100/101) has no x86 round mode
    // and still bails. Route through EmitFcvtz with the matching x86 ROUND imm8
    // so the shared saturation/NaN ladder handles out-of-range/NaN.
    if ((args.op == 0b000 || args.op == 0b001) &&
        (args.rmode == 0b00 || args.rmode == 0b01 || args.rmode == 0b10)) {
      const int8_t round_imm = (args.rmode == 0b00) ? int8_t{0x08}    // RNE + suppress-inexact
                               : (args.rmode == 0b01) ? int8_t{0x0A}  // toward +inf
                                                      : int8_t{0x09};  // toward -inf
      EmitFcvtz(args, is_double, round_imm);
      return;
    }
    // FCVTAS (op 100) / FCVTAU (op 101): FP->int, round-to-nearest ties-away
    // (rmode 00). x86 has no ties-away round mode; EmitFcvtz's ties_away path
    // adds copysign(0.5, x) before the truncating saturation ladder. FP32 (S)
    // only — FP64 (D) bails to lite (mirrors the FP32-only vector FCVTAS/AU).
    if ((args.op == 0b100 || args.op == 0b101) && args.rmode == 0b00) {
      if (is_double) {
        UndefinedReturningVoid();
        return;
      }
      EmitFcvtz(args, is_double, /*round_imm=*/-1, /*ties_away=*/true);
      return;
    }
    // Everything else (rmode==01 V.D[1] FMOV) bails to lite.
    UndefinedReturningVoid();
  }

  // FMOV(reg) / FABS / FNEG for FP32 (ftype=00) and FP64 (ftype=01). These are
  // pure bit operations: copy (FMOV), sign-bit clear (FABS), sign-bit flip
  // (FNEG). Everything stays in the XMM domain — read the scalar into an XMM,
  // apply the sign mask there (PAND for FABS, XORPD for FNEG against a mask XMM),
  // and write back. Keeping reads/writes of guest v[] in the XMM domain is
  // required: RemoveLocalGuestContextAccesses forwards a prior 16-byte MOVDQA
  // store of an XMM vreg straight into a same-offset GET, so a GP-domain read of
  // the same v[] slot would funnel an XMM vreg into a GP operand and fail
  // register-class intersection.
  //
  // FSQRT, FRINT*, FCVT (precision change), BFCVT, FP16, and ftype=10 bail:
  // their SSE ops (Sqrt*, Round*, Cvt*) are not allowlisted for the ARM64
  // backend.
  void FpDataProc1(const Decoder::FpDataProc1Args& args) {
    if (!success()) {
      return;
    }
    if (args.ftype != 0b00 && args.ftype != 0b01) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_double = (args.ftype == 0b01);

    // FMOV / FABS / FNEG only. Anything else bails.
    if (args.opcode != 0b000000 && args.opcode != 0b000001 && args.opcode != 0b000010) {
      UndefinedReturningVoid();
      return;
    }

    FpRegister val = GetVRegScalar(args.rn, is_double);

    if (args.opcode != 0b000000) {
      // Build the sign mask in a GP register (a fresh immediate, never a
      // forwarded guest-context value, so the GP->XMM move is conflict-free),
      // move it into an XMM, then PAND (FABS: clear sign) / XORPD (FNEG: flip
      // sign). FP32 masks live in the low 32 bits; lanes above 0 are irrelevant
      // because SetVRegScalar only commits lane 0.
      FpRegister mask = AllocTempSimdReg();
      if (is_double) {
        uint64_t m = (args.opcode == 0b000001) ? 0x7FFFFFFFFFFFFFFFULL : 0x8000000000000000ULL;
        Register gm = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(m)));
        builder_.Gen<x86_64::MovqXRegReg>(mask.machine_reg(), gm);
      } else {
        uint32_t m = (args.opcode == 0b000001) ? 0x7FFFFFFFu : 0x80000000u;
        Register gm = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(m)));
        builder_.Gen<x86_64::MovdXRegReg>(mask.machine_reg(), gm);
      }
      if (args.opcode == 0b000001) {  // FABS
        builder_.Gen<x86_64::PandXRegXReg>(val.machine_reg(), mask.machine_reg());
      } else {  // FNEG
        builder_.Gen<x86_64::XorpdXRegXReg>(val.machine_reg(), mask.machine_reg());
      }
    }
    SetVRegScalar(args.rd, val, is_double);
  }

  // FADD/FSUB/FMUL/FDIV for FP32 (ftype=00) and FP64 (ftype=01), lowered through
  // the guest-agnostic intrinsic layer (InlineIntrinsicForHeavyOptimizer ->
  // FXxxHostRounding -> SSE ADDSS/ADDSD/... per machine_ir_intrinsic_binding.json).
  // Host default rounding is round-to-nearest, matching ARM FPCR.RMode=0.
  //
  // FPSR note: heavy-optimizer FP regions do NOT yet accumulate FPSR exception
  // bits into cpu.emulated_fpsr (the FeGetExceptions/FeSetExceptions intrinsics
  // are riscv64-only macro defs and convert to the RISC-V exception bit layout,
  // not ARM's). FP *results* are correct without this; MRS-of-FPSR bails to the
  // lite translator/interpreter, which maintains the flags. This is a follow-up.
  //
  // FMUL/FDIV/FADD/FSUB, FMAX/FMIN/FMAXNM/FMINNM (opcode 0b0100..0b0111), and
  // FNMUL (0b1000) are wired here. FP16 (ftype=0b11) and the reserved ftype=0b10
  // bail: their intrinsics/SSE ops are not available in this tier.
  void FpDataProc2(const Decoder::FpDataProc2Args& args) {
    if (!success()) {
      return;
    }
    if (args.ftype != 0b00 && args.ftype != 0b01) {
      UndefinedReturningVoid();
      return;
    }
    if (args.opcode > 0b1000) {
      // Opcodes above FNMUL (0b1000) are reserved.
      UndefinedReturningVoid();
      return;
    }
    const bool is_double = (args.ftype == 0b01);
    FpRegister src1 = GetVRegScalar(args.rn, is_double);
    FpRegister src2 = GetVRegScalar(args.rm, is_double);

    // FMAX/FMIN/FMAXNM/FMINNM (0b0100..0b0111). x86 MAXP{S,D}/MINP{S,D} have
    // ARM-incompatible NaN and signed-zero semantics, so mirror the lite tier's
    // explicit sequences. Packed ops run on the scalar-loaded registers; only
    // lane 0 is written back by SetVRegScalar, so upper-lane garbage is
    // irrelevant. NaN-propagating FMAX/FMIN (ARM: any NaN in -> NaN out) use the
    // symmetric MAX|MAX|POR idiom (the POR keeps a NaN exponent if either input
    // was NaN, and the two-sided MAX makes +-0 order-independent). NaN-
    // suppressing FMAXNM/FMINNM (ARM: exactly one NaN -> the number) substitute
    // each NaN lane with the other operand — via a CMPUNORDP{S,D} self-compare
    // mask — before the MAX/MIN. Mirrors lite_translator.h's scalar path.
    if (args.opcode >= 0b0100 && args.opcode <= 0b0111) {
      const bool is_max = (args.opcode == 0b0100 || args.opcode == 0b0110);
      const bool is_nm = (args.opcode == 0b0110 || args.opcode == 0b0111);
      FpRegister n = src1;
      FpRegister m = src2;
      if (!is_nm) {
        // NaN-propagating: tmp = m; MAXP tmp,n; MAXP n,m; POR n,tmp.
        FpRegister tmp = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), m.machine_reg());
        if (is_max) {
          if (is_double) {
            builder_.Gen<x86_64::MaxpdXRegXReg>(tmp.machine_reg(), n.machine_reg());
            builder_.Gen<x86_64::MaxpdXRegXReg>(n.machine_reg(), m.machine_reg());
          } else {
            builder_.Gen<x86_64::MaxpsXRegXReg>(tmp.machine_reg(), n.machine_reg());
            builder_.Gen<x86_64::MaxpsXRegXReg>(n.machine_reg(), m.machine_reg());
          }
        } else {
          if (is_double) {
            builder_.Gen<x86_64::MinpdXRegXReg>(tmp.machine_reg(), n.machine_reg());
            builder_.Gen<x86_64::MinpdXRegXReg>(n.machine_reg(), m.machine_reg());
          } else {
            builder_.Gen<x86_64::MinpsXRegXReg>(tmp.machine_reg(), n.machine_reg());
            builder_.Gen<x86_64::MinpsXRegXReg>(n.machine_reg(), m.machine_reg());
          }
        }
        builder_.Gen<x86_64::PorXRegXReg>(n.machine_reg(), tmp.machine_reg());
        SetVRegScalar(args.rd, n, is_double);
      } else {
        // NaN-suppressing: mask_a=isnan(n), mask_b=isnan(m). Substitute the NaN
        // lanes with the other operand (an_sub = isnan(n) ? m; bn_sub =
        // isnan(m) ? n), giving sub_n = select(isnan(n), m, n) and sub_m =
        // select(isnan(m), n, m); then MAX/MIN the substituted pair.
        FpRegister mask_a = AllocTempSimdReg();
        FpRegister mask_b = AllocTempSimdReg();
        FpRegister an_sub = AllocTempSimdReg();
        FpRegister bn_sub = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(mask_a.machine_reg(), n.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(mask_b.machine_reg(), m.machine_reg());
        if (is_double) {
          builder_.Gen<x86_64::CmpunordpdXRegXReg>(mask_a.machine_reg(), mask_a.machine_reg());
          builder_.Gen<x86_64::CmpunordpdXRegXReg>(mask_b.machine_reg(), mask_b.machine_reg());
        } else {
          builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_a.machine_reg(), mask_a.machine_reg());
          builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_b.machine_reg(), mask_b.machine_reg());
        }
        builder_.Gen<x86_64::MovdqaXRegXReg>(an_sub.machine_reg(), mask_a.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(an_sub.machine_reg(), m.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(bn_sub.machine_reg(), mask_b.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(bn_sub.machine_reg(), n.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(mask_a.machine_reg(), n.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(mask_b.machine_reg(), m.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), an_sub.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(mask_b.machine_reg(), bn_sub.machine_reg());
        if (is_max) {
          if (is_double) {
            builder_.Gen<x86_64::MaxpdXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
          } else {
            builder_.Gen<x86_64::MaxpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
          }
        } else {
          if (is_double) {
            builder_.Gen<x86_64::MinpdXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
          } else {
            builder_.Gen<x86_64::MinpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
          }
        }
        SetVRegScalar(args.rd, mask_a, is_double);
      }
      return;
    }

    FpRegister result = AllocTempSimdReg();
    if (is_double) {
      switch (args.opcode) {
        case 0b0000:  // FMUL
        case 0b1000:  // FNMUL: -(n * m) — negate the product below.
          EmitFpBinop<&intrinsics::FMul<Float64>>(result, src1, src2);
          break;
        case 0b0001:  // FDIV
          EmitFpBinop<&intrinsics::FDiv<Float64>>(result, src1, src2);
          break;
        case 0b0010:  // FADD
          EmitFpBinop<&intrinsics::FAdd<Float64>>(result, src1, src2);
          break;
        case 0b0011:  // FSUB
          EmitFpBinop<&intrinsics::FSub<Float64>>(result, src1, src2);
          break;
      }
    } else {
      switch (args.opcode) {
        case 0b0000:  // FMUL
        case 0b1000:  // FNMUL: -(n * m) — negate the product below.
          EmitFpBinop<&intrinsics::FMul<Float32>>(result, src1, src2);
          break;
        case 0b0001:  // FDIV
          EmitFpBinop<&intrinsics::FDiv<Float32>>(result, src1, src2);
          break;
        case 0b0010:  // FADD
          EmitFpBinop<&intrinsics::FAdd<Float32>>(result, src1, src2);
          break;
        case 0b0011:  // FSUB
          EmitFpBinop<&intrinsics::FSub<Float32>>(result, src1, src2);
          break;
      }
    }
    if (args.opcode == 0b1000) {
      // FNMUL: flip the sign bit of the product. Build the sign mask in a fresh
      // GP register (never a forwarded guest value, so the GP->XMM move is
      // conflict-free), move it into an XMM, then XORPD. FP32 masks live in the
      // low 32 bits; upper lanes are irrelevant because SetVRegScalar commits
      // only lane 0. Mirrors lite_translator.h's FpDataProc2 FNMUL path.
      FpRegister sign = AllocTempSimdReg();
      if (is_double) {
        Register gs = std::get<0>(Gen<x86_64::MovqRegImm>(
            static_cast<int64_t>(0x8000000000000000ULL)));
        builder_.Gen<x86_64::MovqXRegReg>(sign.machine_reg(), gs);
      } else {
        Register gs = std::get<0>(Gen<x86_64::MovlRegImm>(
            static_cast<int32_t>(0x80000000u)));
        builder_.Gen<x86_64::MovdXRegReg>(sign.machine_reg(), gs);
      }
      builder_.Gen<x86_64::XorpdXRegXReg>(result.machine_reg(), sign.machine_reg());
    }
    SetVRegScalar(args.rd, result, is_double);
  }

  // FCMP/FCMPE Sn/Dn, Sm/Dm (or #0.0): compare and set NZCV. Lowers to x86
  // UCOMIS{S,D} (now allowlisted as UcomiseXRegXReg) followed by the FP-specific
  // EFLAGS->ARM-NZCV mapping (EmitStoreArmFpNZCV). Defined in the .cc (the NZCV
  // mapping is a branch tree over PF/ZF/CF and needs basic-block manipulation).
  // FP16 (ftype 0b11, would need F16C widening ops not in the backend gen
  // inputs) and the reserved ftype 0b10 bail to the lite tier.
  void FpCompare(const Decoder::FpCompareArgs& args);

  // FCCMP/FCCMPE: if `cond` holds, perform the FCMP compare + NZCV mapping;
  // otherwise write the 4-bit nzcv immediate straight to cpu.flags. Mirrors
  // lite_translator.h::FpConditionalCompare; defined in the .cc (then/else/merge
  // basic blocks, same shape as ConditionalCompare). The signal_nans (FCCMPE)
  // bit does not change the architectural NZCV output — UCOMIS already signals
  // on SNaN — so it is ignored, matching lite.
  void FpConditionalCompare(const Decoder::FpConditionalCompareArgs& args);

  // SCVTF/UCVTF Sd/Dd, Wn/Xn (unscaled, rmode == 00): convert a signed/unsigned
  // integer general register to scalar FP via x86 CVTSI2SS/SD. Defined in the .cc
  // because the sf==1 unsigned form needs a basic-block split (values >= 2^63 use
  // the round-to-odd halve/convert/double fix-up). Mirrors
  // lite_translator.h::FpIntConversion's SCVTF/UCVTF path.
  void EmitScvtfUcvtf(const Decoder::FpIntConvArgs& args, bool is_double);

  // FCVTZS (op 000) / FCVTZU (op 001), truncating (rmode == 11): FP -> integer
  // via x86 CVTT{SS,SD}2SI plus the ARM by-sign saturation / NaN fix-up ladder.
  // BB-split lowering (a shared `result` GP vreg merged via PseudoCopy). Mirrors
  // lite_translator.h::FpIntConversion's FCVTZS/FCVTZU paths.
  //
  // The rounding scalar conversions FCVTNS/NU (rmode 00), FCVTPS/PU (rmode 01)
  // and FCVTMS/MU (rmode 10) reuse this same saturation/NaN ladder: pass
  // `round_imm >= 0` (an x86 ROUND imm8 for RNE / +inf / -inf) and the source
  // is ROUND-ed to an integer-valued FP first (NaN/±Inf/sign-of-zero pass
  // through unchanged), so the truncating cvtt then yields the rounded integer
  // with the ARM out-of-range/NaN semantics preserved. FCVTAS/AU (ties-away,
  // op 100/101) have no x86 round mode; pass `ties_away = true` (FP32 only) to
  // add a copysign(0.5, x) addend — gated to 0 when |x| >= 2^23, where a 0.5
  // addend would round the wrong way — before the same truncating ladder.
  void EmitFcvtz(const Decoder::FpIntConvArgs& args,
                 bool is_double,
                 int8_t round_imm = -1,
                 bool ties_away = false);

  //
  // Advanced SIMD (Args-struct forms).
  //

  void AdvSimdFcma(const Decoder::FcmaArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdFcmaIdx(const Decoder::FcmaIdxArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdBf16ThreeSame(const Decoder::Bf16ThreeSameArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdMatMul(const Decoder::MatMulArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdDotProduct(const Decoder::DotProductArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  // AdvSIMD modified immediate: MOVI / MVNI / vector FMOV (replace forms) and
  // ORR / BIC (vector, immediate) (read-modify-write forms). The 128-bit value
  // is a pure function of (op, cmode, abc, defgh, q) via AdvSIMDExpandImm, so it
  // is computed at translation time (ExpandSimdModifiedImmJit, shared with the
  // lite translator) and materialized into V[rd].
  //
  // Materialization uses only allowlisted XMM ops: a PXOR-zeroed XMM, MOVQ to
  // load the low 64 bits from a GP reg, and (when the upper half is non-zero and
  // Q==1) PINSRQ to insert the high 64 bits. For the D-form (Q==0) only the low
  // 64 bits are inserted and the upper half stays zero (PXOR), matching ARM64
  // 64-bit-vector write semantics. The commit is a single 16-byte MOVDQA store at
  // the v[rd] displacement (GenSetSimd<16>). Mirrors lite_translator.h::Simd-
  // ModifiedImm. Bails (FP16 vector FMOV / reserved cmodes) never reach here:
  // the decoder rejects them upstream, and the lite tier handles whatever does.
  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs& args) {
    if (!success()) {
      return;
    }
    const uint8_t cmode = args.cmode;
    // ORR/BIC (vector, immediate): cmode<0>==1 and cmode<3:2>!=11. These read-
    // modify-write V[rd] with the MOVI-style (op=0) expanded immediate: ORR
    // (op=0) sets bits, BIC (op=1) clears them.
    const bool is_orr_bic = (cmode & 1) && ((cmode & 0b1100) != 0b1100);
    __uint128_t value =
        is_orr_bic
            ? ExpandSimdModifiedImmJit(0, cmode, args.abc, args.defgh, args.q)
            : ExpandSimdModifiedImmJit(args.op, cmode, args.abc, args.defgh, args.q);
    const uint64_t lo = static_cast<uint64_t>(value);
    // Q==0 operates on the low 64 bits and zeroes the upper 64 of V[rd].
    const uint64_t hi = args.q ? static_cast<uint64_t>(value >> 64) : 0;
    const int32_t off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);

    // Build the 128-bit immediate constant into a PXOR-zeroed XMM. For Q==0 the
    // high half is left zero by construction.
    FpRegister ximm = AllocZeroedSimdReg();
    if (lo != 0) {
      Register glo = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(lo)));
      builder_.Gen<x86_64::MovqXRegReg>(ximm.machine_reg(), glo);  // zero-extends upper 64
    }
    if (hi != 0) {
      Register ghi = std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(hi)));
      builder_.Gen<x86_64::PinsrqXRegRegImm>(ximm.machine_reg(), ghi, int8_t{1});
    }

    if (!is_orr_bic) {
      // MOVI / MVNI / FMOV: replace V[rd] with the constant.
      builder_.GenSetSimd<16>(off, ximm.machine_reg());
      return;
    }

    // ORR/BIC: load current V[rd] (D-form must zero-extend the low 64 so the
    // upper half ends up zeroed), then OR (set) / AND-NOT (clear) the immediate.
    FpRegister xd = AllocTempSimdReg();
    if (args.q) {
      builder_.GenGetSimd<16>(xd.machine_reg(), off);
    } else {
      // MOVSD reg<-mem zero-extends the upper 64 bits of the XMM. The MOVSD load
      // is one of the SIMD opcodes RemoveLocalGuestContextAccesses recognizes as
      // a guest-context GET, so a prior 16-byte store forwards correctly.
      xd = FpRegister{std::get<0>(Gen<x86_64::MovsdXRegOp>(
          {.base = x86_64::kMachineRegRBP, .disp = off}))};
    }
    if (args.op == 0) {
      builder_.Gen<x86_64::PorXRegXReg>(xd.machine_reg(), ximm.machine_reg());
      builder_.GenSetSimd<16>(off, xd.machine_reg());
    } else {
      // BIC: xd = ~imm & Vd. PANDN(dst, src) computes dst = ~dst & src, so with
      // dst=ximm, src=xd we get ~imm & Vd; the result lands in ximm.
      builder_.Gen<x86_64::PandnXRegXReg>(ximm.machine_reg(), xd.machine_reg());
      builder_.GenSetSimd<16>(off, ximm.machine_reg());
    }
  }

  void SimdLoadLiteral(const Decoder::SimdLoadLiteralArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  // LDR/STR (SIMD&FP, immediate): 128/64/32-bit (Q/D/S). A load zero-extends the
  // rest of the 128-bit V[rt] (MOVSD/MOVSS already clear the unused lanes; the
  // full 16-byte slot is then committed). The memory access uses the unaligned
  // MOVDQU/MOVSD/MOVSS forms (guest memory is not 16-byte aligned) and a recovery
  // block so a host fault is delivered to the guest signal handler. The V[rt]
  // slot is 16-byte aligned, so its access uses GenGetSimd/GenSetSimd (MOVDQA).
  // 8/16-bit (B/H) bail to the lite tier. Mirrors lite_translator.h::SimdLoadStoreImm.
  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& args, Register base) {
    if (!success()) {
      return;
    }
    if (args.size != Decoder::SimdLoadStoreSize::k32bit &&
        args.size != Decoder::SimdLoadStoreSize::k64bit &&
        args.size != Decoder::SimdLoadStoreSize::k128bit) {
      UndefinedReturningVoid();
      return;
    }
    Register masked = ApplyTbi(base);
    const int32_t off = static_cast<int32_t>(args.offset);
    const int32_t vreg_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rt * 16);
    FpRegister xmm = AllocTempSimdReg();
    if (args.is_store) {
      builder_.GenGetSimd<16>(xmm.machine_reg(), vreg_off);
      switch (args.size) {
        case Decoder::SimdLoadStoreSize::k128bit:
          builder_.Gen<x86_64::MovdquOpXReg>({.base = masked, .disp = off}, xmm.machine_reg());
          break;
        case Decoder::SimdLoadStoreSize::k64bit:
          builder_.Gen<x86_64::MovsdOpXReg>({.base = masked, .disp = off}, xmm.machine_reg());
          break;
        default:  // k32bit
          builder_.Gen<x86_64::MovssOpXReg>({.base = masked, .disp = off}, xmm.machine_reg());
          break;
      }
      GenRecoveryBlockForLastInsn();
    } else {
      switch (args.size) {
        case Decoder::SimdLoadStoreSize::k128bit:
          xmm = FpRegister{std::get<0>(Gen<x86_64::MovdquXRegOp>({.base = masked, .disp = off}))};
          break;
        case Decoder::SimdLoadStoreSize::k64bit:
          // MOVSD reg<-mem zero-extends the upper 64 bits of the XMM.
          xmm = FpRegister{std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = masked, .disp = off}))};
          break;
        default:  // k32bit; MOVSS reg<-mem zero-extends the upper 96 bits.
          xmm = FpRegister{std::get<0>(Gen<x86_64::MovssXRegOp>({.base = masked, .disp = off}))};
          break;
      }
      GenRecoveryBlockForLastInsn();
      builder_.GenSetSimd<16>(vreg_off, xmm.machine_reg());
    }
  }

  // LDP/STP (SIMD&FP): 128-bit Q-pair, 64-bit D-pair, and 32-bit S-pair. The two
  // elements sit at [addr] and [addr + element_size] (16/8/4). Each memory access
  // is recovery-wrapped. A load zero-extends the unused lanes (MOVDQU is full
  // width; MOVSD zeroes bits 127:64; MOVSS zeroes bits 127:32) and the full
  // 16-byte v[] slot is committed so the guest register's upper bits read as
  // zero. The 64/32-bit D/S pair is the ubiquitous callee-saved FP save/restore
  // (ldp/stp d8-d15) in function prologues/epilogues. Mirrors
  // lite_translator.h::SimdLoadStorePair.
  void SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs& args, Register addr) {
    if (!success()) {
      return;
    }
    int32_t element_size;
    switch (args.size) {
      case Decoder::SimdLoadStoreSize::k128bit: element_size = 16; break;
      case Decoder::SimdLoadStoreSize::k64bit: element_size = 8; break;
      case Decoder::SimdLoadStoreSize::k32bit: element_size = 4; break;
      default:
        UndefinedReturningVoid();
        return;
    }
    Register masked = ApplyTbi(addr);
    const int32_t v1_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rt1 * 16);
    const int32_t v2_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rt2 * 16);
    if (args.is_store) {
      FpRegister xmm1 = AllocTempSimdReg();
      FpRegister xmm2 = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xmm1.machine_reg(), v1_off);
      builder_.GenGetSimd<16>(xmm2.machine_reg(), v2_off);
      switch (args.size) {
        case Decoder::SimdLoadStoreSize::k128bit:
          builder_.Gen<x86_64::MovdquOpXReg>({.base = masked, .disp = 0}, xmm1.machine_reg());
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::MovdquOpXReg>({.base = masked, .disp = element_size},
                                             xmm2.machine_reg());
          GenRecoveryBlockForLastInsn();
          break;
        case Decoder::SimdLoadStoreSize::k64bit:
          builder_.Gen<x86_64::MovsdOpXReg>({.base = masked, .disp = 0}, xmm1.machine_reg());
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::MovsdOpXReg>({.base = masked, .disp = element_size},
                                            xmm2.machine_reg());
          GenRecoveryBlockForLastInsn();
          break;
        default:  // k32bit
          builder_.Gen<x86_64::MovssOpXReg>({.base = masked, .disp = 0}, xmm1.machine_reg());
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::MovssOpXReg>({.base = masked, .disp = element_size},
                                            xmm2.machine_reg());
          GenRecoveryBlockForLastInsn();
          break;
      }
    } else {
      FpRegister xmm1;
      FpRegister xmm2;
      switch (args.size) {
        case Decoder::SimdLoadStoreSize::k128bit:
          xmm1 = FpRegister{std::get<0>(Gen<x86_64::MovdquXRegOp>({.base = masked, .disp = 0}))};
          GenRecoveryBlockForLastInsn();
          xmm2 = FpRegister{
              std::get<0>(Gen<x86_64::MovdquXRegOp>({.base = masked, .disp = element_size}))};
          GenRecoveryBlockForLastInsn();
          break;
        case Decoder::SimdLoadStoreSize::k64bit:
          // MOVSD reg<-mem zero-extends the upper 64 bits of the XMM.
          xmm1 = FpRegister{std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = masked, .disp = 0}))};
          GenRecoveryBlockForLastInsn();
          xmm2 = FpRegister{
              std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = masked, .disp = element_size}))};
          GenRecoveryBlockForLastInsn();
          break;
        default:  // k32bit; MOVSS reg<-mem zero-extends the upper 96 bits.
          xmm1 = FpRegister{std::get<0>(Gen<x86_64::MovssXRegOp>({.base = masked, .disp = 0}))};
          GenRecoveryBlockForLastInsn();
          xmm2 = FpRegister{
              std::get<0>(Gen<x86_64::MovssXRegOp>({.base = masked, .disp = element_size}))};
          GenRecoveryBlockForLastInsn();
          break;
      }
      builder_.GenSetSimd<16>(v1_off, xmm1.machine_reg());
      builder_.GenSetSimd<16>(v2_off, xmm2.machine_reg());
    }
  }

  // LDR/STR (SIMD&FP, register offset): 128/64/32-bit (Q/D/S). Compute the
  // register-offset address (offset extend + shift + base add) with the same
  // EmitRegOffsetAddr helper the GP register-offset load/store uses, apply TBI,
  // then reuse SimdLoadStoreImm's mem<->xmm<->v[] path at disp=0. A load's
  // MOVSD/MOVSS zero-extends the unused lanes and the full 16-byte v[] slot is
  // committed; the memory access uses the unaligned MOVDQU/MOVSD/MOVSS forms and
  // a recovery block. 8/16-bit (B/H) bail to the lite tier (which covers all
  // sizes), matching SimdLoadStoreImm. Mirrors lite_translator.h::SimdLoadStoreReg.
  void SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs& args, Register base, Register offset) {
    if (!success()) {
      return;
    }
    if (args.size != Decoder::SimdLoadStoreSize::k32bit &&
        args.size != Decoder::SimdLoadStoreSize::k64bit &&
        args.size != Decoder::SimdLoadStoreSize::k128bit) {
      UndefinedReturningVoid();
      return;
    }
    Register addr = EmitRegOffsetAddr(base, offset, args.extend_type, args.shift_amount);
    Register masked = ApplyTbi(addr);
    const int32_t vreg_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rt * 16);
    FpRegister xmm = AllocTempSimdReg();
    if (args.is_store) {
      builder_.GenGetSimd<16>(xmm.machine_reg(), vreg_off);
      switch (args.size) {
        case Decoder::SimdLoadStoreSize::k128bit:
          builder_.Gen<x86_64::MovdquOpXReg>({.base = masked, .disp = 0}, xmm.machine_reg());
          break;
        case Decoder::SimdLoadStoreSize::k64bit:
          builder_.Gen<x86_64::MovsdOpXReg>({.base = masked, .disp = 0}, xmm.machine_reg());
          break;
        default:  // k32bit
          builder_.Gen<x86_64::MovssOpXReg>({.base = masked, .disp = 0}, xmm.machine_reg());
          break;
      }
      GenRecoveryBlockForLastInsn();
    } else {
      switch (args.size) {
        case Decoder::SimdLoadStoreSize::k128bit:
          xmm = FpRegister{std::get<0>(Gen<x86_64::MovdquXRegOp>({.base = masked, .disp = 0}))};
          break;
        case Decoder::SimdLoadStoreSize::k64bit:
          // MOVSD reg<-mem zero-extends the upper 64 bits of the XMM.
          xmm = FpRegister{std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = masked, .disp = 0}))};
          break;
        default:  // k32bit; MOVSS reg<-mem zero-extends the upper 96 bits.
          xmm = FpRegister{std::get<0>(Gen<x86_64::MovssXRegOp>({.base = masked, .disp = 0}))};
          break;
      }
      GenRecoveryBlockForLastInsn();
      builder_.GenSetSimd<16>(vreg_off, xmm.machine_reg());
    }
  }

  // AdvSIMD copy. DUP (general), INS (general), UMOV, SMOV, and INS (element)
  // are implemented in the optimizing tier; DUP (element) still bails to the
  // lite translator (its PSHUFB byte-broadcast mask constant / PSHUFD lane
  // select don't map cleanly onto the optimizer allowlist). The lane-select
  // moves (UMOV/SMOV/INS-element) route through PEXTR/PINSR in the XMM domain
  // for MOVDQA store/load-forwarding consistency. Mirrors
  // lite_translator.h::AdvSimdCopy.
  //
  // DUP (general) — broadcast a GP register Rn across all lanes of Vd —
  // lowers as follows:
  // imm5 encodes the element size: bit0=B(1), bit1=H(2), bit2=S(4), bit3=D(8).
  //   B: MOVD Rn->xmm, then PSHUFB with a zeroed mask broadcasts byte 0 to all 16.
  //   H: MOVD Rn->xmm (low halfword in lane 0), then PINSRW into all 8 lanes.
  //   S: MOVD Rn->xmm, then PINSRD into all 4 lanes.
  //   D: MOVQ Rn->xmm (FULL 64 bits — a 32-bit MOVD here would truncate a
  //      pointer), then PUNPCKLQDQ self duplicates the low qword to both halves.
  // Q==0 forms zero the upper 64 bits of Vd (D-register semantics): for B/H/S
  // they are built into a PXOR-zeroed XMM whose upper half is only filled for the
  // Q==1 broadcast, and the 64-bit (1D) Q==0 form is ARM-reserved and bails.
  void AdvSimdCopy(const Decoder::AdvSimdCopyArgs& args) {
    if (!success()) {
      return;
    }
    // INS (general): insert a GP register into one lane of Vd, preserving the
    // others. Read v[rd] as an XMM (GenGetSimd<16>), PINSR the GP value at the
    // lane, and store the full 128 bits back (GenSetSimd<16>) — staying in the
    // XMM domain so the MOVDQA store/load forwarding stays consistent. The lane
    // width and index come from imm5. Mirrors lite_translator.h's INS-general.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kInsGeneral) {
      const uint8_t imm5 = args.imm5;
      uint8_t esize;
      int8_t lane;
      if (imm5 & 0b00001) {
        esize = 1;
        lane = static_cast<int8_t>((imm5 >> 1) & 0xf);
      } else if (imm5 & 0b00010) {
        esize = 2;
        lane = static_cast<int8_t>((imm5 >> 2) & 0x7);
      } else if (imm5 & 0b00100) {
        esize = 4;
        lane = static_cast<int8_t>((imm5 >> 3) & 0x3);
      } else if (imm5 & 0b01000) {
        esize = 8;
        lane = static_cast<int8_t>((imm5 >> 4) & 0x1);
      } else {
        UndefinedReturningVoid();  // reserved imm5
        return;
      }
      const int32_t off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
      FpRegister xmm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xmm.machine_reg(), off);
      Register src = (args.rn < 31) ? GetReg(args.rn)
                                    : std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
      switch (esize) {
        case 1:
          builder_.Gen<x86_64::PinsrbXRegRegImm>(xmm.machine_reg(), src, lane);
          break;
        case 2:
          builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), src, lane);
          break;
        case 4:
          builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), src, lane);
          break;
        default:
          builder_.Gen<x86_64::PinsrqXRegRegImm>(xmm.machine_reg(), src, lane);
          break;
      }
      builder_.GenSetSimd<16>(off, xmm.machine_reg());
      return;
    }

    // UMOV / SMOV / INS (element): lane-select moves. Decode element size and
    // the (destination) lane index from imm5 — the same encoding kInsGeneral
    // uses. All three stay in the XMM domain (GenGetSimd<16> full loads,
    // PEXTR/PINSR, GenSetSimd<16> full stores), NOT a narrow memory access to
    // a v[] sub-lane, so the MOVDQA store/load forwarding the optimizer relies
    // on stays consistent (a partial-width access to a 16-byte v[] slot could
    // be reordered around a wide store/load of the same slot). The lite tier
    // is memory-direct instead — safe there because the single-pass lite
    // translator never reorders.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kUmov ||
        args.opcode == Decoder::AdvSimdCopyOpcode::kSmov ||
        args.opcode == Decoder::AdvSimdCopyOpcode::kInsElement) {
      const uint8_t imm5_low4 = args.imm5 & 0xf;
      uint8_t esize;
      uint8_t index;
      if (imm5_low4 & 0x1) {
        esize = 1;
        index = (args.imm5 >> 1) & 0xf;
      } else if (imm5_low4 & 0x2) {
        esize = 2;
        index = (args.imm5 >> 2) & 0x7;
      } else if (imm5_low4 & 0x4) {
        esize = 4;
        index = (args.imm5 >> 3) & 0x3;
      } else if (imm5_low4 & 0x8) {
        esize = 8;
        index = (args.imm5 >> 4) & 0x1;
      } else {
        UndefinedReturningVoid();  // reserved imm5
        return;
      }
      const int32_t vn_off =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);

      // UMOV Vn.<T>[index] -> Rd (unsigned). Canonical (esize, Q) pairs: B/H/S
      // with Q=0 (Wd), D with Q=1 (Xd). PEXTR zero-extends the extracted
      // element into the GP register (upper bits cleared) — exactly UMOV's
      // unsigned semantics; the 32-bit PEXTR forms clear the upper 32 bits,
      // matching a Wd write. rd==31 (WZR/XZR) discards the result.
      if (args.opcode == Decoder::AdvSimdCopyOpcode::kUmov) {
        const bool canonical = (esize == 8 && args.q) || (esize != 8 && !args.q);
        if (!canonical) {
          UndefinedReturningVoid();
          return;
        }
        if (args.rd < 31) {
          FpRegister xn = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
          Register res;
          switch (esize) {
            case 1:
              res = std::get<0>(
                  Gen<x86_64::PextrbRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
              break;
            case 2:
              res = std::get<0>(
                  Gen<x86_64::PextrwRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
              break;
            case 4:
              res = std::get<0>(
                  Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
              break;
            default:
              res = std::get<0>(
                  Gen<x86_64::PextrqRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
              break;
          }
          SetReg(args.rd, res);
        }
        return;
      }

      // SMOV Vn.<T>[index] -> Rd (signed). Canonical: (1,*), (2,*), (4,Q=1).
      // Extract the raw element (zero-extended by PEXTR) then re-sign-extend
      // the low esize bits to the destination width; the 32-bit Movsx*l forms
      // clear the upper 32 (Wd), the 64-bit Movsx*q forms fill it (Xd). No
      // 32-bit sign-extend-from-memory LIR op exists, so this two-step is why
      // SMOV routes through PEXTR + a RegReg sign-extend.
      if (args.opcode == Decoder::AdvSimdCopyOpcode::kSmov) {
        const bool canonical = (esize == 1) || (esize == 2) || (esize == 4 && args.q);
        if (!canonical) {
          UndefinedReturningVoid();
          return;
        }
        if (args.rd < 31) {
          FpRegister xn = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
          Register raw;
          if (esize == 4) {
            raw = std::get<0>(
                Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
          } else if (esize == 2) {
            raw = std::get<0>(
                Gen<x86_64::PextrwRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
          } else {
            raw = std::get<0>(
                Gen<x86_64::PextrbRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(index)));
          }
          Register res;
          if (esize == 1 && !args.q) {
            res = std::get<0>(Gen<x86_64::MovsxblRegReg>(raw));
          } else if (esize == 1) {
            res = std::get<0>(Gen<x86_64::MovsxbqRegReg>(raw));
          } else if (esize == 2 && !args.q) {
            res = std::get<0>(Gen<x86_64::MovsxwlRegReg>(raw));
          } else if (esize == 2) {
            res = std::get<0>(Gen<x86_64::MovsxwqRegReg>(raw));
          } else /* esize == 4 && q */ {
            res = std::get<0>(Gen<x86_64::MovsxlqRegReg>(raw));
          }
          SetReg(args.rd, res);
        }
        return;
      }

      // INS (element): Vn.<T>[src_index] -> Vd.<T>[index], other Vd lanes kept.
      // src_index comes from imm4, decoded against the same esize. Load both Vd
      // and Vn as full XMMs, PEXTR the source lane into a GP temp, PINSR it into
      // Vd at the dest lane, store Vd back. Loading both before storing is
      // correct for the rd==rn self-INS case (e.g. INS V0.S[0], V0.S[3]). The
      // decoder enforces Q=1, so the full 128-bit Vd is in play.
      uint8_t src_index;
      switch (esize) {
        case 1:
          src_index = args.imm4 & 0xF;
          break;
        case 2:
          src_index = (args.imm4 >> 1) & 0x7;
          break;
        case 4:
          src_index = (args.imm4 >> 2) & 0x3;
          break;
        default /* esize == 8 */:
          src_index = (args.imm4 >> 3) & 0x1;
          break;
      }
      const int32_t vd_off =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
      FpRegister xd = AllocTempSimdReg();
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      switch (esize) {
        case 1: {
          Register t = std::get<0>(
              Gen<x86_64::PextrbRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(src_index)));
          builder_.Gen<x86_64::PinsrbXRegRegImm>(
              xd.machine_reg(), t, static_cast<int8_t>(index));
          break;
        }
        case 2: {
          Register t = std::get<0>(
              Gen<x86_64::PextrwRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(src_index)));
          builder_.Gen<x86_64::PinsrwXRegRegImm>(
              xd.machine_reg(), t, static_cast<int8_t>(index));
          break;
        }
        case 4: {
          Register t = std::get<0>(
              Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(src_index)));
          builder_.Gen<x86_64::PinsrdXRegRegImm>(
              xd.machine_reg(), t, static_cast<int8_t>(index));
          break;
        }
        default: {
          Register t = std::get<0>(
              Gen<x86_64::PextrqRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(src_index)));
          builder_.Gen<x86_64::PinsrqXRegRegImm>(
              xd.machine_reg(), t, static_cast<int8_t>(index));
          break;
        }
      }
      builder_.GenSetSimd<16>(vd_off, xd.machine_reg());
      return;
    }

    // DUP (element): broadcast Vn.<T>[index] (one esize-byte element) to every
    // lane of Vd. Q=0 fills the low 64 and zeros the upper 64 (D-register
    // semantics); Q=1 fills all 128. imm5 encodes (esize, index) exactly as
    // UMOV/SMOV/INS-element above. Lowering mirrors lite_translator.h's
    // DUP-element:
    //   esize=1 → PSHUFB with a materialized {idx}×16 byte-index mask.
    //   esize=2 → PSHUFB with a {idx*2, idx*2+1}×8 halfword-index mask.
    //   esize=4 → PSHUFD imm = idx*0x55 (broadcast dword[idx]).
    //   esize=8 → PSHUFD imm 0x44 (idx=0) / 0xEE (idx=1); (8,Q=0)=1D reserved.
    // Both halves of the 128-bit index mask are identical, so it is built
    // qword-at-a-time via MOVQ r->xmm + PUNPCKLQDQ self-broadcast (the same
    // mask-materialization recipe used elsewhere in this file). SetVRegFull with
    // q=false zeros the upper 64 for the D-form. Stays entirely in the XMM
    // domain so MOVDQA store/load forwarding remains consistent.
    if (args.opcode == Decoder::AdvSimdCopyOpcode::kDupElement) {
      const uint8_t imm5_low4 = args.imm5 & 0xf;
      uint8_t esize;
      uint8_t index;
      if (imm5_low4 & 0x1) {
        esize = 1;
        index = (args.imm5 >> 1) & 0xf;
      } else if (imm5_low4 & 0x2) {
        esize = 2;
        index = (args.imm5 >> 2) & 0x7;
      } else if (imm5_low4 & 0x4) {
        esize = 4;
        index = (args.imm5 >> 3) & 0x3;
      } else if (imm5_low4 & 0x8) {
        esize = 8;
        index = (args.imm5 >> 4) & 0x1;
      } else {
        UndefinedReturningVoid();  // reserved imm5
        return;
      }
      // DUP Vd.1D (esize=8, Q=0) is ARM-reserved.
      if (esize == 8 && !args.q) {
        UndefinedReturningVoid();
        return;
      }
      const int32_t vn_off =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
      FpRegister xmm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xmm.machine_reg(), vn_off);
      if (esize == 1 || esize == 2) {
        uint64_t mask_qword;
        if (esize == 1) {
          mask_qword = 0x0101010101010101ULL * static_cast<uint64_t>(index);
        } else {
          const uint64_t b0 = static_cast<uint64_t>(index) * 2;
          const uint64_t pair = ((b0 + 1) << 8) | b0;
          mask_qword = pair | (pair << 16) | (pair << 32) | (pair << 48);
        }
        Register r =
            std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(mask_qword)));
        FpRegister mask = AllocTempSimdReg();
        builder_.Gen<x86_64::MovqXRegReg>(mask.machine_reg(), r);
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(mask.machine_reg(), mask.machine_reg());
        builder_.Gen<x86_64::PshufbXRegXReg>(xmm.machine_reg(), mask.machine_reg());
      } else if (esize == 4) {
        const int8_t imm = static_cast<int8_t>(static_cast<uint8_t>(index * 0x55));
        builder_.Gen<x86_64::PshufdXRegXRegImm>(xmm.machine_reg(), xmm.machine_reg(), imm);
      } else {  // esize == 8, Q=1
        const int8_t imm =
            (index == 0) ? static_cast<int8_t>(0x44) : static_cast<int8_t>(0xEE);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(xmm.machine_reg(), xmm.machine_reg(), imm);
      }
      SetVRegFull(args.rd, xmm, args.q);
      return;
    }

    if (args.opcode != Decoder::AdvSimdCopyOpcode::kDupGeneral) {
      // Any remaining AdvSimdCopy opcode (e.g. kDupScalar) is not lowered by the
      // optimizing tier; the lite translator handles it.
      UndefinedReturningVoid();
      return;
    }

    const uint8_t esize_bits = args.imm5 & 0xf;
    // 1D (Q==0, esize=D) is ARM-reserved.
    if (esize_bits == 0x08 && !args.q) {
      UndefinedReturningVoid();
      return;
    }

    // Source GP value (XZR/WZR -> 0: a common compiler idiom to zero a vector).
    Register src;
    if (args.rn < 31) {
      src = GetReg(args.rn);
    } else {
      src = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0}));
    }

    const int32_t off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);

    if (esize_bits == 0x08) {  // D (Q==1 only): 2D broadcast.
      FpRegister xmm = AllocTempSimdReg();
      builder_.Gen<x86_64::MovqXRegReg>(xmm.machine_reg(), src);          // low qword = src, upper = 0
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmm.machine_reg(), xmm.machine_reg());  // dup low qword
      builder_.GenSetSimd<16>(off, xmm.machine_reg());
      return;
    }

    if (esize_bits == 0x01) {  // B: byte broadcast.
      FpRegister xmm = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdXRegReg>(xmm.machine_reg(), src);  // byte 0 in lane 0
      // PSHUFB with an all-zero mask selects byte 0 for every output byte.
      FpRegister zero_mask = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PshufbXRegXReg>(xmm.machine_reg(), zero_mask.machine_reg());
      // PSHUFB filled all 16 bytes; for Q==0 we still must zero the upper half.
      SetVRegFull(args.rd, xmm, args.q);
      return;
    }

    if (esize_bits == 0x02) {  // H: halfword broadcast via PINSRW into every lane.
      // Build into a zeroed XMM so the Q==0 upper half stays 0 (only lanes 0..3
      // are written for the D-form; the SetVRegFull merge then drops 4..7 too).
      FpRegister xmm = AllocZeroedSimdReg();
      const int8_t n = args.q ? int8_t{8} : int8_t{4};
      for (int8_t lane = 0; lane < n; ++lane) {
        builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), src, lane);
      }
      SetVRegFull(args.rd, xmm, args.q);
      return;
    }

    // esize_bits == 0x04: S: word broadcast via PINSRD into every lane.
    FpRegister xmm = AllocZeroedSimdReg();
    const int8_t n = args.q ? int8_t{4} : int8_t{2};
    for (int8_t lane = 0; lane < n; ++lane) {
      builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), src, lane);
    }
    SetVRegFull(args.rd, xmm, args.q);
  }

  // AdvSIMD three-same INTEGER ops that lower to a single packed SSE2/SSE4.1
  // instruction: ADD, SUB, AND, ORR, EOR, MUL, and CMEQ (register). Each loads
  // the full 128-bit Vn and Vm with GenGetSimd<16> (MOVDQA), runs the packed op
  // on the host XMM, and writes the result back with GenSetSimd<16> (MOVDQA).
  // The 16-byte aligned MOVDQA access at the v[reg] displacement is the form
  // RemoveLocalGuestContextAccesses recognizes for store-to-load forwarding, so
  // reads of a just-written V register inside the region see the fresh value.
  //
  // For the D-form (Q=0) the upper 64 bits of Vd are zeroed: the result's low
  // 64 bits are merged (MOVSD reg-reg) into a PXOR-zeroed XMM, exactly the
  // single-store discipline SetVRegScalar uses, so the committed MOVDQA holds a
  // clean zero-extended 64-bit value.
  //
  // Element size comes from args.size (00=byte, 01=half, 10=word, 11=double).
  // The available packed ops constrain which sizes are handled:
  //   ADD: Paddb (8), Paddw (16), Paddd (32), Paddq (64) — all sizes handled.
  //   SUB: Psubb (8), Psubw (16), Psubd (32), Psubq (64) — all sizes handled.
  //   MUL: Pmullw (16), Pmulld (32). 8-bit and 64-bit have no packed op; bail.
  //   AND/ORR/EOR: Pand/Por/Pxor are element-size-independent (one op covers
  //     all). ORR with rn==rm is the AdvSIMD MOV (vector) alias and lowers the
  //     same way.
  //   BSL/BIT/BIF: bitwise-select — element-size-independent Pxor/Pand/Pandn
  //     sequences that read Vd; handled in a self-contained pre-switch block.
  //   CMEQ: Pcmpeqb (8), Pcmpeqw (16), Pcmpeqd (32). 64-bit (Pcmpeqq) bails.
  //   CMGT (signed): Pcmpgtb (8), Pcmpgtw (16), Pcmpgtd (32). 64-bit bails.
  //   CMGE (signed >=): NOT(Pcmpgt(Vm, Vn)); CMHI/CMHS (unsigned >, >=):
  //     sign-bias both operands then the signed Pcmpgt (+ invert for CMHS).
  //     All three are 8/16/32-bit; the 64-bit form needs PCMPGTQ and bails.
  //     Handled in a self-contained pre-switch block (result register varies).
  // Everything else (saturating, shifts, polynomial, FP, pairwise, widening,
  // CMTST, etc.) bails to the lite translator/interpreter.
  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& args) {
    if (!success()) {
      return;
    }
    const int32_t vn_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
    const int32_t vm_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);

    // PMUL polynomial multiply (vector, byte lanes). ARM ARM C7.2.219: per-lane
    // carry-less (GF(2)[x]) multiply keeping the low 8 bits. The decoder pins
    // size=00 (.8B/.16B); other sizes are reserved. Mirror of
    // lite_translator.h::AdvSimdThreeSame kPmul's per-bit unrolled SSE2 recipe
    // (no PCLMULQDQ dependency):
    //   acc = 0; bit = 0x01 (byte-replicated)
    //   for i in 0..7:
    //     shifted_a = (a << i) per byte, byte-masked with (0xFF<<i)&0xFF
    //     selector  = ((b & bit) PCMPEQB bit)   -> 0xFF per byte where bit i set
    //     acc      ^= shifted_a & selector
    //     bit      += bit   (PADDB doubles 0x01->0x02->...->0x80, no overflow)
    // PSLLW shifts whole 16-bit lanes, so the low byte's high bits spill into the
    // adjacent byte; the PAND with shift_mask = -bit re-zeros those spilled bits
    // in every byte. Q=0 (.8B) zeroes Vd[127:64] via SetVRegFull's D-form merge.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kPmul) {
      if (args.size != 0b00) {
        UndefinedReturningVoid();  // decoder pins size=00; defensive bail.
        return;
      }
      FpRegister xa = AllocTempSimdReg();
      FpRegister xb = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xa.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xb.machine_reg(), vm_off);
      // acc/zero pre-zeroed (AllocZeroedSimdReg avoids a use-before-def
      // self-PXOR on a fresh temp — see lifetime.h reg_class_ CHECK).
      FpRegister xacc = AllocZeroedSimdReg();
      FpRegister xzero = AllocZeroedSimdReg();
      // bit = 0x01 replicated across all 16 bytes (MOVQ zero-extends, then
      // PUNPCKLQDQ broadcasts the low 64 into the high 64).
      FpRegister xbit = AllocTempSimdReg();
      Register gp = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0101010101010101LL}));
      builder_.Gen<x86_64::MovqXRegReg>(xbit.machine_reg(), gp);
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xbit.machine_reg(), xbit.machine_reg());
      FpRegister xshift = AllocTempSimdReg();
      FpRegister xsel = AllocTempSimdReg();
      for (int i = 0; i < 8; ++i) {
        // shifted_a = (a << i) per byte, byte-masked.
        builder_.Gen<x86_64::MovdqaXRegXReg>(xshift.machine_reg(), xa.machine_reg());
        if (i != 0) {
          builder_.Gen<x86_64::PsllwXRegImm>(xshift.machine_reg(), static_cast<int8_t>(i));
          // shift_mask = -bit per byte = (0xFF<<i)&0xFF.
          builder_.Gen<x86_64::MovdqaXRegXReg>(xsel.machine_reg(), xzero.machine_reg());
          builder_.Gen<x86_64::PsubbXRegXReg>(xsel.machine_reg(), xbit.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(xshift.machine_reg(), xsel.machine_reg());
        }
        // selector = (b & bit) PCMPEQB bit -> 0xFF per byte where bit i set.
        builder_.Gen<x86_64::MovdqaXRegXReg>(xsel.machine_reg(), xb.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xsel.machine_reg(), xbit.machine_reg());
        builder_.Gen<x86_64::PcmpeqbXRegXReg>(xsel.machine_reg(), xbit.machine_reg());
        // acc ^= shifted_a & selector.
        builder_.Gen<x86_64::PandXRegXReg>(xshift.machine_reg(), xsel.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xacc.machine_reg(), xshift.machine_reg());
        // Double bit for the next iteration (skip after the last bit).
        if (i != 7) {
          builder_.Gen<x86_64::PaddbXRegXReg>(xbit.machine_reg(), xbit.machine_reg());
        }
      }
      SetVRegFull(args.rd, xacc, args.q);
      return;
    }

    // CMGE/CMHI/CMHS (signed >= / unsigned > / unsigned >=) — handled here as
    // self-contained sequences because their result does not always land in vn
    // (the accumulator the shared switch below assumes) and the unsigned forms
    // need a sign-bias step. x86 has no unsigned vector compare, so CMHI/CMHS
    // flip the per-lane sign bit of both operands (XOR with the width's sign
    // mask) to map the unsigned ordering onto the signed PCMPGT*. CMGE =
    // NOT(Vm > Vn); CMHS = NOT(biased Vm > biased Vn). The .2D (size=11) form
    // needs PCMPGTQ (not allowlisted) and bails to the lite tier.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmge ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhi ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhs) {
      if (args.size == 0b11) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_unsigned =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhi) ||
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhs);
      // CMHI computes (Vn > Vm) with no invert; CMGE/CMHS compute (Vm > Vn)
      // then invert into (Vn >= Vm) / (Vn >=u Vm) via XOR with all-ones.
      const bool invert =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmge) ||
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmhs);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      if (is_unsigned) {
        // Build the per-lane sign-bit mask and XOR it into both operands.
        FpRegister sign = AllocZeroedSimdReg();
        switch (args.size) {
          case 0b00: {
            // 0x80 in every byte (no PSLLB on x86): broadcast from a GPR.
            Register t = std::get<0>(
                Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0x8080808080808080ULL)));
            builder_.Gen<x86_64::MovqXRegReg>(sign.machine_reg(), t);
            builder_.Gen<x86_64::PunpcklqdqXRegXReg>(sign.machine_reg(), sign.machine_reg());
            break;
          }
          case 0b01:
            builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
            builder_.Gen<x86_64::PsllwXRegImm>(sign.machine_reg(), int8_t{15});
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign.machine_reg(), sign.machine_reg());
            builder_.Gen<x86_64::PslldXRegImm>(sign.machine_reg(), int8_t{31});
            break;
        }
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), sign.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign.machine_reg());
      }
      // CMHI (no invert): PCMPGT(xn, xm) -> result in xn.
      // CMGE/CMHS (invert): PCMPGT(xm, xn) -> result in xm, then invert.
      FpRegister res = invert ? xm : xn;
      FpRegister a = invert ? xm : xn;
      FpRegister b = invert ? xn : xm;
      switch (args.size) {
        case 0b00:
          builder_.Gen<x86_64::PcmpgtbXRegXReg>(a.machine_reg(), b.machine_reg());
          break;
        case 0b01:
          builder_.Gen<x86_64::PcmpgtwXRegXReg>(a.machine_reg(), b.machine_reg());
          break;
        default:  // 0b10
          builder_.Gen<x86_64::PcmpgtdXRegXReg>(a.machine_reg(), b.machine_reg());
          break;
      }
      if (invert) {
        FpRegister ones = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(ones.machine_reg(), ones.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(res.machine_reg(), ones.machine_reg());
      }
      SetVRegFull(args.rd, res, args.q);
      return;
    }

    // FP vector FMAX/FMIN/FMAXNM/FMINNM (.2S/.4S/.2D). x86 MAXP{S,D}/MINP{S,D}
    // disagree with ARM on both NaN and signed-zero results, so mirror
    // lite_translator.h's vector sequence exactly. Everything is computed as
    // FMIN (FMAX = -FMIN(-a,-b)): x86 MINP{S,D} returns the SECOND source on a
    // +-0 tie, which gives ARM's OR-of-signs (most-negative) for FMIN and, after
    // the double negation, AND-of-signs (most-positive) for FMAX. NaN-propagating
    // FMAX/FMIN (any NaN in -> NaN out) use the symmetric MIN|MIN|POR idiom;
    // NaN-suppressing FMAXNM/FMINNM (exactly one NaN -> the number) substitute
    // each NaN lane with the other operand via a CMPUNORDP{S,D} self-compare mask
    // before the MIN. FP16 (is_fp16, needs an F16C round-trip not in the backend
    // gen inputs) bails to lite; the decoder already filters the reserved
    // sz=1&&!Q (.1D) shape, so only .2S/.4S/.2D reach here.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxV ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminV ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminnmV) {
      if (args.is_fp16) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_max = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxV ||
                           args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV);
      const bool is_nm = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmaxnmV ||
                          args.opcode == Decoder::AdvSimdThreeSameOpcode::kFminnmV);
      const bool is_double = (args.size & 1);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      // FMAX: negate both inputs so the shared FMIN lowering computes it; the
      // per-lane sign mask (all-ones << 31/63) is XORed in now and, after the
      // MIN, XORed back out of the result.
      FpRegister sign_mask = no_fp_register;
      if (is_max) {
        // AllocZeroedSimdReg establishes a def before the all-ones self-compare
        // (a bare AllocTempSimdReg would trip the lifetime use-before-def CHECK).
        sign_mask = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(sign_mask.machine_reg(), sign_mask.machine_reg());
        if (is_double) {
          builder_.Gen<x86_64::PsllqXRegImm>(sign_mask.machine_reg(), int8_t{63});
        } else {
          builder_.Gen<x86_64::PslldXRegImm>(sign_mask.machine_reg(), int8_t{31});
        }
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), sign_mask.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), sign_mask.machine_reg());
      }
      auto gen_min = [&](FpRegister dst, FpRegister src) {
        if (is_double) {
          builder_.Gen<x86_64::MinpdXRegXReg>(dst.machine_reg(), src.machine_reg());
        } else {
          builder_.Gen<x86_64::MinpsXRegXReg>(dst.machine_reg(), src.machine_reg());
        }
      };
      auto gen_cmpunord = [&](FpRegister dst, FpRegister src) {
        if (is_double) {
          builder_.Gen<x86_64::CmpunordpdXRegXReg>(dst.machine_reg(), src.machine_reg());
        } else {
          builder_.Gen<x86_64::CmpunordpsXRegXReg>(dst.machine_reg(), src.machine_reg());
        }
      };
      FpRegister result;
      if (!is_nm) {
        // NaN-propagating: tmp = xm; MIN(tmp, xn); MIN(xn, xm); POR(xn, tmp).
        FpRegister tmp = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), xm.machine_reg());
        gen_min(tmp, xn);
        gen_min(xn, xm);
        builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), tmp.machine_reg());
        result = xn;
      } else {
        // NaN-suppressing: substitute each NaN lane with the other operand
        // (a' = select(isnan(a), b, a); b' = select(isnan(b), a, b)), then MIN.
        FpRegister mask_a = AllocTempSimdReg();
        FpRegister mask_b = AllocTempSimdReg();
        FpRegister an_sub = AllocTempSimdReg();
        FpRegister bn_sub = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(mask_a.machine_reg(), xn.machine_reg());
        gen_cmpunord(mask_a, mask_a);  // 1s where a is NaN
        builder_.Gen<x86_64::MovdqaXRegXReg>(mask_b.machine_reg(), xm.machine_reg());
        gen_cmpunord(mask_b, mask_b);  // 1s where b is NaN
        builder_.Gen<x86_64::MovdqaXRegXReg>(an_sub.machine_reg(), mask_a.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(an_sub.machine_reg(), xm.machine_reg());  // mask_a & b
        builder_.Gen<x86_64::MovdqaXRegXReg>(bn_sub.machine_reg(), mask_b.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(bn_sub.machine_reg(), xn.machine_reg());  // mask_b & a
        builder_.Gen<x86_64::PandnXRegXReg>(mask_a.machine_reg(), xn.machine_reg());  // ~mask_a & a
        builder_.Gen<x86_64::PandnXRegXReg>(mask_b.machine_reg(), xm.machine_reg());  // ~mask_b & b
        builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), an_sub.machine_reg());  // a'
        builder_.Gen<x86_64::PorXRegXReg>(mask_b.machine_reg(), bn_sub.machine_reg());  // b'
        // minab|minba|OR so the +-0 tie gets OR-of-signs (ARM's FMINNM rule);
        // operands are NaN-free here (NaN lanes were substituted above).
        FpRegister t_nm = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(t_nm.machine_reg(), mask_b.machine_reg());
        gen_min(t_nm, mask_a);      // min(b', a')
        gen_min(mask_a, mask_b);    // min(a', b')
        builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), t_nm.machine_reg());
        result = mask_a;
      }
      // Negate the FMIN result back to obtain FMAX = -FMIN(-a,-b).
      if (is_max) {
        builder_.Gen<x86_64::PxorXRegXReg>(result.machine_reg(), sign_mask.machine_reg());
      }
      // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, result, args.q);
      return;
    }

    // FP vector FADD/FSUB/FMUL/FDIV (.2S/.4S FP32, .2D FP64) lower directly to
    // SSE2 packed FP arithmetic (ADDP{S,D}/SUBP{S,D}/MULP{S,D}/DIVP{S,D}), whose
    // default-rounding IEEE-754 results match ARM's lane-for-lane. Mirrors
    // lite_translator.h's non-FP16 path exactly. FP16 (is_fp16) needs an F16C
    // round-trip absent from the backend Gen inputs and bails to lite; the
    // decoder already rejects the reserved sz=1&&!Q (.1D) shape, so only
    // .2S/.4S/.2D reach here.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFaddV ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFsubV ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFmulV ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFdivV) {
      if (args.is_fp16) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_double = (args.size & 1);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      if (is_double) {
        switch (args.opcode) {
          case Decoder::AdvSimdThreeSameOpcode::kFaddV:
            builder_.Gen<x86_64::AddpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
            break;
          case Decoder::AdvSimdThreeSameOpcode::kFsubV:
            builder_.Gen<x86_64::SubpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
            break;
          case Decoder::AdvSimdThreeSameOpcode::kFmulV:
            builder_.Gen<x86_64::MulpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
            break;
          default:  // kFdivV
            builder_.Gen<x86_64::DivpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
            break;
        }
      } else {
        switch (args.opcode) {
          case Decoder::AdvSimdThreeSameOpcode::kFaddV:
            builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
            break;
          case Decoder::AdvSimdThreeSameOpcode::kFsubV:
            builder_.Gen<x86_64::SubpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
            break;
          case Decoder::AdvSimdThreeSameOpcode::kFmulV:
            builder_.Gen<x86_64::MulpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
            break;
          default:  // kFdivV
            builder_.Gen<x86_64::DivpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
            break;
        }
      }
      // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // FP vector FCMEQ/FCMGE/FCMGT (.2S/.4S FP32, .2D FP64) produce a per-lane
    // all-ones/zero mask. Mirrors lite_translator.h's non-FP16 path: the SSE
    // legacy-encoded CMP{EQ,LT,LE}P{S,D} predicates are ordered, returning FALSE
    // (zero) for any NaN operand, exactly matching ARM's unordered-is-false rule.
    //   FCMEQ: CMPEQP* xn, xm                 -> xn = (xn == xm)
    //   FCMGE: CMPLEP* xm, xn   [result = xm] -> xm = (xm <= xn) == (xn >= xm)
    //   FCMGT: CMPLTP* xm, xn   [result = xm] -> xm = (xm <  xn) == (xn >  xm)
    // FP16 (is_fp16, needs an F16C round-trip absent from the backend Gen inputs)
    // bails to lite; the decoder already rejects the reserved sz=1&&!Q (.1D)
    // shape, so only .2S/.4S/.2D reach here. FACGE/FACGT (abs-compare) are NOT
    // handled here — they need a sign-clear pre-mask and still bail to lite.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFcmeqV ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFcmgeV ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kFcmgtV) {
      if (args.is_fp16) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_double = (args.size & 1);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      FpRegister result;
      if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFcmeqV) {
        if (is_double) {
          builder_.Gen<x86_64::CmpeqpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::CmpeqpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        result = xn;
      } else if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFcmgeV) {
        // (xm <= xn) == (xn >= xm); mask lands in xm.
        if (is_double) {
          builder_.Gen<x86_64::CmplepdXRegXReg>(xm.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::CmplepsXRegXReg>(xm.machine_reg(), xn.machine_reg());
        }
        result = xm;
      } else {  // FCMGT
        // (xm < xn) == (xn > xm); mask lands in xm.
        if (is_double) {
          builder_.Gen<x86_64::CmpltpdXRegXReg>(xm.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::CmpltpsXRegXReg>(xm.machine_reg(), xn.machine_reg());
        }
        result = xm;
      }
      // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, result, args.q);
      return;
    }

    // FP vector FABD = |a - b| (.2S/.4S FP32, .2D FP64): a packed SUB then a
    // sign-clear (bitwise AND with 0x7FFF…). ARM FABD is FPAbs(FPSub(a,b)); x86
    // SUBP{S,D} matches ARM FPSub lane-for-lane under default rounding, and
    // clearing the sign bit yields FPAbs — including on NaN, where ARM FPAbs
    // also clears the sign bit. Mirrors lite_translator.h's non-FP16 path. The
    // sign-clear mask (0x7FFFFFFF/dword FP32, 0x7FFFFFFFFFFFFFFF/qword FP64) is
    // built with the PCMPEQD-self ; PSRLD/PSRLQ 1 idiom. FP16 (needs an F16C
    // round-trip absent from the backend Gen inputs) bails to lite; the decoder
    // already rejects the reserved sz=1&&!Q (.1D) shape.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kFabdV) {
      if (args.is_fp16) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_double = (args.size & 1);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      if (is_double) {
        builder_.Gen<x86_64::SubpdXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::SubpsXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      // AllocZeroedSimdReg establishes a def before the all-ones self-compare
      // (a bare AllocTempSimdReg would trip the lifetime use-before-def CHECK).
      FpRegister mask = AllocZeroedSimdReg();
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(mask.machine_reg(), mask.machine_reg());
      if (is_double) {
        builder_.Gen<x86_64::PsrlqXRegImm>(mask.machine_reg(), int8_t{1});
      } else {
        builder_.Gen<x86_64::PsrldXRegImm>(mask.machine_reg(), int8_t{1});
      }
      builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), mask.machine_reg());
      // Q=0 (.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // Integer vector MLA/MLS (multiply-accumulate / multiply-subtract):
    //   MLA: Vd[lane] += Vn[lane] * Vm[lane]
    //   MLS: Vd[lane] -= Vn[lane] * Vm[lane]
    // at .8H/.4H (size=01, PMULLW) and .4S/.2S (size=10, PMULLD). Unlike ADD/SUB
    // above, these READ Vd as the accumulator, so they are handled here rather
    // than in the shared vn-only switch: compute the low-half product Vn*Vm, then
    // PADD (MLA) / PSUB (MLS) it into Vd. Mirrors lite_translator.h's non-byte
    // path exactly (the low 16/32 bits of the product are what ARM keeps). The
    // .16B/.8B (size=00) byte form needs the widen+PMULLW+PACKUSWB recipe and the
    // reserved .2D (size=11) form has no packed 64-bit multiply — both bail to the
    // lite tier, exactly like the heavy kMul path.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kMla ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kMls) {
      if (args.size != 0b01 && args.size != 0b10) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_mls = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kMls);
      const int32_t vd_off =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      // Low-half product Vn*Vm into xn.
      if (args.size == 0b01) {
        builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      // Accumulate into / subtract from Vd at the element width.
      if (is_mls) {
        if (args.size == 0b01) {
          builder_.Gen<x86_64::PsubwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PsubdXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
      } else {
        if (args.size == 0b01) {
          builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
      }
      // Q=0 (.4H/.2S) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xd, args.q);
      return;
    }

    // Integer vector bitwise-select BSL / BIT / BIF (opcode 0b00011, U=1; the
    // size field selects which of the three). These are element-size-independent
    // bit ops that all READ Vd, so they are handled here (like MLA/MLS) rather
    // than in the shared vn-only switch. Sequences mirror lite_translator.h
    // exactly (Pxor/Pand/Pandn are all already allowlisted; no new LIR ops):
    //   BSL  Vd = (Vd & Vn) | (~Vd & Vm) == ((Vn ^ Vm) & Vd) ^ Vm
    //   BIT  Vd = (Vm & Vn) | (~Vm & Vd) == ((Vn ^ Vd) & Vm) ^ Vd  ("if true")
    //   BIF  Vd = (Vm & Vd) | (~Vm & Vn) == Vd ^ (~Vm & (Vn ^ Vd))  ("if false")
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kBsl ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kBit ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kBif) {
      const int32_t vd_off =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      FpRegister res = xn;  // BSL/BIT accumulate into xn; BIF stores xd.
      if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kBsl) {
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());  // Vn^Vm
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xd.machine_reg());  // &Vd
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());  // ^Vm
      } else if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kBit) {
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xd.machine_reg());  // Vn^Vd
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());  // &Vm
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xd.machine_reg());  // ^Vd
      } else {  // kBif: fold the NOT into PANDN (~xm & xn).
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xd.machine_reg());   // Vn^Vd
        builder_.Gen<x86_64::PandnXRegXReg>(xm.machine_reg(), xn.machine_reg());  // ~Vm & (Vn^Vd)
        builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), xm.machine_reg());   // Vd ^ ...
        res = xd;
      }
      // Q=0 (.8B) zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, res, args.q);
      return;
    }

    // Integer vector saturating add/sub SQADD/UQADD/SQSUB/UQSUB (opcode 00001 for
    // add, 00101 for sub; U selects signed/unsigned). The 8-bit (size=00) and
    // 16-bit (size=01) forms have direct SSE2 saturating packed ops; the 32-bit
    // (size=10) forms have no native saturating dword op, so they are emulated
    // exactly as lite_translator.h does (wrap-add/sub + overflow detect +
    // saturate). These read only Vn/Vm (never Vd) but need multiple temps and a
    // per-op emulation sequence, so they are handled here rather than in the
    // shared vn-only switch. The reserved .2D (size=11, 64-bit) form has no
    // packed 64-bit saturating path in either tier and bails to lite (which in
    // turn routes it to the interpreter) — mirroring lite's `size==0b11` bail.
    // The result always lands in xn; SetVRegFull's Q=0 merge zeroes Vd[127:64].
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqadd ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqadd ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqsub ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqsub) {
      if (args.size == 0b11) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqadd) {
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PaddsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else if (args.size == 0b01) {
          builder_.Gen<x86_64::PaddswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          // 32-bit signed saturating add. sum = a + b (wrap); overflow iff
          // ~(a^b) & (a^sum) has its MSB set; sat = (a<0)?INT_MIN:INT_MAX;
          // result = sum ^ ((sum ^ sat) & ovf_mask).
          FpRegister t_sum = AllocTempSimdReg();
          FpRegister t_ovf = AllocTempSimdReg();
          FpRegister t_sat = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t_sum.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PadddXRegXReg>(t_sum.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_sum.machine_reg());
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());  // -1
          builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());   // ~(a^b)
          builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
          builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});
          builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{31});            // a<0?-1:0
          builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{1});             // 0x7FFFFFFF
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());      // sat
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_sum.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_ovf.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_sum.machine_reg());
        }
      } else if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kUqadd) {
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PaddusbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else if (args.size == 0b01) {
          builder_.Gen<x86_64::PadduswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          // 32-bit unsigned saturating add. sum = a + b (wrap); overflow iff
          // sum < a (unsigned), detected via PMAXUD: max(a,sum)==sum iff no
          // overflow; saturate overflowed lanes to UINT32_MAX.
          FpRegister t_save_a = AllocTempSimdReg();
          FpRegister t_ones = AllocZeroedSimdReg();  // def before the self-compare
          builder_.Gen<x86_64::MovdqaXRegXReg>(t_save_a.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xm.machine_reg());       // sum
          builder_.Gen<x86_64::PmaxudXRegXReg>(t_save_a.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(t_save_a.machine_reg(), xn.machine_reg());  // -1 if no ovf
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(t_ones.machine_reg(), t_ones.machine_reg());  // -1
          builder_.Gen<x86_64::PxorXRegXReg>(t_save_a.machine_reg(), t_ones.machine_reg());   // -1 if ovf
          builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), t_save_a.machine_reg());        // saturate
        }
      } else if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqsub) {
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PsubsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else if (args.size == 0b01) {
          builder_.Gen<x86_64::PsubswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          // 32-bit signed saturating sub. diff = a - b (wrap); overflow iff
          // (a^b) & (a^diff) has its MSB set; sat = (a<0)?INT_MIN:INT_MAX;
          // result = diff ^ ((diff ^ sat) & ovf_mask).
          FpRegister t_diff = AllocTempSimdReg();
          FpRegister t_ovf = AllocTempSimdReg();
          FpRegister t_sat = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t_diff.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsubdXRegXReg>(t_diff.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(t_ovf.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t_ovf.machine_reg(), xm.machine_reg());   // a^b
          builder_.Gen<x86_64::MovdqaXRegXReg>(t_sat.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t_sat.machine_reg(), t_diff.machine_reg());  // a^diff
          builder_.Gen<x86_64::PandXRegXReg>(t_ovf.machine_reg(), t_sat.machine_reg());
          builder_.Gen<x86_64::PsradXRegImm>(t_ovf.machine_reg(), int8_t{31});
          builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{31});            // a<0?-1:0
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());   // -1
          builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{1});             // 0x7FFFFFFF
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());      // sat
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_ovf.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), t_diff.machine_reg());
        }
      } else {  // kUqsub
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PsubusbXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else if (args.size == 0b01) {
          builder_.Gen<x86_64::PsubuswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          // 32-bit unsigned saturating sub. result = (a>=b) ? a-b : 0. Mask:
          // PMINUD(a,b)==b iff a>=b; AND the wrap-diff with it to zero underflow.
          FpRegister t_mask = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t_mask.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PminudXRegXReg>(t_mask.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(t_mask.machine_reg(), xm.machine_reg());  // -1 if a>=b
          builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());        // a-b
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), t_mask.machine_reg());     // zero underflow
        }
      }
      // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // Vector saturating-doubling multiply-high SQDMULH / rounding SQRDMULH
    // (opcode 10110; U selects round). Per lane: (2*Vn*Vm) >> esize, with the
    // sole INT_MIN*INT_MIN corner (== -2*minval^2 which would overflow) saturated
    // to INT_MAX. SQRDMULH adds a rounding term of 2^(esize-1) before the shift.
    // Only the 16-bit (size=01) and 32-bit (size=10) lane forms are defined; the
    // decoder reserves size 00/11, so those bail. This is a byte-for-byte mirror
    // of lite_translator.h's kSqdmulh/kSqrdmulh lowering — reads only Vn/Vm, so
    // it lives here (like SQADD) rather than the shared vn-only switch. Result
    // lands in xn (size=01) / xp_lo (size=10); SetVRegFull's Q=0 merge zeroes
    // Vd[127:64].
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqdmulh ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrdmulh) {
      const bool is_round =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSqrdmulh);
      if (args.size == 0b01) {
        FpRegister xn = AllocTempSimdReg();
        FpRegister xm = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
        if (is_round) {
          // SQRDMULH .4H/.8H via PMULHRSW + corner fixup (SSSE3). PMULHRSW
          // computes ((a*b >> 14) + 1) >> 1 = round((2*a*b)/2^16) already, so
          // only the INT16_MIN*INT16_MIN corner needs the ^0xFFFF flip.
          if (!host_platform::kHasSSSE3) {
            UndefinedReturningVoid();
            return;
          }
          FpRegister xn_corner = AllocTempSimdReg();
          FpRegister xm_corner = AllocTempSimdReg();
          FpRegister x_min = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqw idiom
          builder_.Gen<x86_64::MovdqaXRegXReg>(xn_corner.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(xm_corner.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(x_min.machine_reg(), x_min.machine_reg());
          builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});  // INT16_MIN
          builder_.Gen<x86_64::PmulhrswXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_corner.machine_reg(), x_min.machine_reg());
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_corner.machine_reg(), x_min.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(xn_corner.machine_reg(), xm_corner.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_corner.machine_reg());
          SetVRegFull(args.rd, xn, args.q);
          return;
        }
        // SQDMULH .4H/.8H via PMULHW + PMULLW combine + corner fixup (SSE2).
        // high16(2*a*b) = (high16(a*b) << 1) | (top bit of low16(a*b)).
        FpRegister xn_lo = AllocTempSimdReg();
        FpRegister xn_corner = AllocTempSimdReg();
        FpRegister xm_corner = AllocTempSimdReg();
        FpRegister x_min = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqw idiom
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn_corner.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(xm_corner.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn_lo.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmullwXRegXReg>(xn_lo.machine_reg(), xm.machine_reg());  // low16(a*b)
        builder_.Gen<x86_64::PmulhwXRegXReg>(xn.machine_reg(), xm.machine_reg());     // high16(a*b)
        builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{1});              // high<<1
        builder_.Gen<x86_64::PsrlwXRegImm>(xn_lo.machine_reg(), int8_t{15});          // low top bit
        builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xn_lo.machine_reg());     // high16(2*a*b)
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(x_min.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});          // INT16_MIN
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_corner.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_corner.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn_corner.machine_reg(), xm_corner.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_corner.machine_reg());  // INT16_MIN^0xFFFF=MAX
        SetVRegFull(args.rd, xn, args.q);
        return;
      }
      if (args.size == 0b10) {
        // size=10 .2S/.4S: PMULDQ widen (SSE4.1) + PSLLQ double + optional round
        // + corner fixup; PSHUFD 0xDD lifts each product's upper 32 bits.
        if (!host_platform::kHasSSE4_1) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        FpRegister xm = AllocTempSimdReg();
        FpRegister x_const = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqd idiom
        FpRegister corner = AllocTempSimdReg();
        FpRegister xp_lo = AllocTempSimdReg();
        FpRegister xp_hi = AllocTempSimdReg();
        FpRegister xm_hi = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
        // x_const = INT32_MIN broadcast across 4 dwords.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
        builder_.Gen<x86_64::PslldXRegImm>(x_const.machine_reg(), int8_t{31});
        // Corner: lanes where Vn.s[i] == INT32_MIN AND Vm.s[i] == INT32_MIN.
        builder_.Gen<x86_64::MovdqaXRegXReg>(corner.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(corner.machine_reg(), x_const.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(xp_lo.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xp_lo.machine_reg(), x_const.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(corner.machine_reg(), xp_lo.machine_reg());
        // Two PMULDQs reconstruct the 4 signed 32x32 -> 64 products (even lanes
        // in xp_lo, odd lanes in xp_hi).
        builder_.Gen<x86_64::MovdqaXRegXReg>(xp_lo.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmuldqXRegXReg>(xp_lo.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(xp_hi.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsrlqXRegImm>(xp_hi.machine_reg(), int8_t{32});
        builder_.Gen<x86_64::MovdqaXRegXReg>(xm_hi.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PsrlqXRegImm>(xm_hi.machine_reg(), int8_t{32});
        builder_.Gen<x86_64::PmuldqXRegXReg>(xp_hi.machine_reg(), xm_hi.machine_reg());
        // Double each 64-bit signed product.
        builder_.Gen<x86_64::PsllqXRegImm>(xp_lo.machine_reg(), int8_t{1});
        builder_.Gen<x86_64::PsllqXRegImm>(xp_hi.machine_reg(), int8_t{1});
        if (is_round) {
          // SQRDMULH: add rounding constant 2^31 = 0x80000000 per qword.
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
          builder_.Gen<x86_64::PsllqXRegImm>(x_const.machine_reg(), int8_t{63});
          builder_.Gen<x86_64::PsrlqXRegImm>(x_const.machine_reg(), int8_t{32});
          builder_.Gen<x86_64::PaddqXRegXReg>(xp_lo.machine_reg(), x_const.machine_reg());
          builder_.Gen<x86_64::PaddqXRegXReg>(xp_hi.machine_reg(), x_const.machine_reg());
        }
        // Extract the upper 32 bits of each 64-bit lane and interleave.
        builder_.Gen<x86_64::PshufdXRegXRegImm>(xp_lo.machine_reg(), xp_lo.machine_reg(), static_cast<int8_t>(0xDD));
        builder_.Gen<x86_64::PshufdXRegXRegImm>(xp_hi.machine_reg(), xp_hi.machine_reg(), static_cast<int8_t>(0xDD));
        builder_.Gen<x86_64::PunpckldqXRegXReg>(xp_lo.machine_reg(), xp_hi.machine_reg());
        // Apply corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX.
        builder_.Gen<x86_64::PxorXRegXReg>(xp_lo.machine_reg(), corner.machine_reg());
        SetVRegFull(args.rd, xp_lo, args.q);
        return;
      }
      // size=00 and size=11 reserved by the decoder; bail to lite/interp.
      UndefinedReturningVoid();
      return;
    }

    // Integer vector S{MAX,MIN}/U{MAX,MIN} (opcode 01100 max / 01101 min; U
    // selects signed/unsigned): lane-wise signed or unsigned min/max. x86 has
    // direct lane-width-matched PMAXS/PMINS/PMAXU/PMINU for 8/16/32-bit lanes;
    // the .2D (size=11) form has no SSE-era 64-bit min/max (PMAXSQ/… need
    // AVX-512F-VL) and bails to lite. Mirrors lite_translator.h's SMAX/… block
    // exactly, including the per-size SSE4.1 gate (PMAXSB/PMINSB/PMAXSD/PMINSD/
    // PMAXUW/PMINUW/PMAXUD/PMINUD need SSE4.1; PMAXSW/PMINSW/PMAXUB/PMINUB are
    // SSE2). The result lands in xn; SetVRegFull's Q=0 merge zeroes Vd[127:64].
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmin) {
      if (args.size == 0b11) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_max = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
                           args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax);
      const bool is_signed = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
                              args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin);
      const bool needs_sse4_1 = (is_signed && args.size == 0b00) ||
                                (is_signed && args.size == 0b10) ||
                                (!is_signed && args.size == 0b01) ||
                                (!is_signed && args.size == 0b10);
      if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      switch (args.size) {
        case 0b00:  // .16B / .8B
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxubXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminubXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
        case 0b01:  // .8H / .4H
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
        default:  // 0b10: .4S / .2S
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxudXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
      }
      // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // SABD/UABD (absolute difference) and SABA/UABA (absolute difference then
    // accumulate into Vd): per-lane |Vn - Vm|. Mirrors lite_translator.h's
    // kSabd/kUabd/kSaba/kUaba block exactly. ARM computes the difference in
    // extended precision then truncates the absolute value; a naive PSUB +
    // sign-mask abs mismatches on the INT_MIN-vs-INT_MAX wrap. The correct
    // modular-arithmetic recipe is per-lane max(a,b) - min(a,b) on the
    // signed (SABD/SABA) or unsigned (UABD/UABA) interpretation, so it reuses
    // the same PMAXS/PMINS/PMAXU/PMINU packed ops as the min/max block above,
    // then PSUB (and PADD into Vd for the *ABA accumulate). The .2D (size=11)
    // form is reserved by the ARM ARM and bails. Per-size SSE4.1 gate matches
    // the min/max block (PMAXSB/PMINSB/PMAXSD/PMINSD/PMAXUW/PMINUW/PMAXUD/
    // PMINUD need SSE4.1; PMAXSW/PMINSW/PMAXUB/PMINUB are SSE2).
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSabd ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUabd ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kSaba ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUaba) {
      if (args.size == 0b11) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_signed = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSabd ||
                              args.opcode == Decoder::AdvSimdThreeSameOpcode::kSaba);
      const bool is_accum = (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSaba ||
                             args.opcode == Decoder::AdvSimdThreeSameOpcode::kUaba);
      const bool needs_sse4_1 = (is_signed && args.size == 0b00) ||
                                (is_signed && args.size == 0b10) ||
                                (!is_signed && args.size == 0b01) ||
                                (!is_signed && args.size == 0b10);
      if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      FpRegister xmax = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      // xmax = max(Vn, Vm); xn = min(Vn, Vm); xmax -= xn == |Vn - Vm|.
      builder_.Gen<x86_64::MovdqaXRegXReg>(xmax.machine_reg(), xn.machine_reg());
      switch (args.size) {
        case 0b00:  // .16B / .8B
          if (is_signed) {
            builder_.Gen<x86_64::PmaxsbXRegXReg>(xmax.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PminsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PmaxubXRegXReg>(xmax.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PminubXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          builder_.Gen<x86_64::PsubbXRegXReg>(xmax.machine_reg(), xn.machine_reg());
          break;
        case 0b01:  // .8H / .4H
          if (is_signed) {
            builder_.Gen<x86_64::PmaxswXRegXReg>(xmax.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PmaxuwXRegXReg>(xmax.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          builder_.Gen<x86_64::PsubwXRegXReg>(xmax.machine_reg(), xn.machine_reg());
          break;
        default:  // 0b10: .4S / .2S
          if (is_signed) {
            builder_.Gen<x86_64::PmaxsdXRegXReg>(xmax.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PmaxudXRegXReg>(xmax.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          builder_.Gen<x86_64::PsubdXRegXReg>(xmax.machine_reg(), xn.machine_reg());
          break;
      }
      if (is_accum) {
        // Accumulate the abs-diff (xmax) into Vd at the element width.
        const int32_t vd_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PaddbXRegXReg>(xd.machine_reg(), xmax.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xmax.machine_reg());
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xmax.machine_reg());
            break;
        }
        // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
        SetVRegFull(args.rd, xd, args.q);
      } else {
        // Q=0 zeroes Vd[127:64] via SetVRegFull's D-form merge.
        SetVRegFull(args.rd, xmax, args.q);
      }
      return;
    }

    // SHADD/UHADD (halving add: (a+b)>>1), SRHADD/URHADD (rounding halving add:
    // (a+b+1)>>1), and SHSUB/UHSUB (halving sub: (a-b)>>1), signed or unsigned.
    // Mirrors lite_translator.h's kShadd/kUhadd/kSrhadd/kUrhadd/kShsub/kUhsub
    // blocks. ARM ARM C7.2 reserves the .2D (size=11) form -> bail.
    //   * Halfword/word ADD forms use the bitwise identities (proven from
    //     a+b = (a^b) + 2*(a&b) and a|b = (a^b) + (a&b)):
    //       (a+b)>>1   = (a&b) + ((a^b)>>1)   [SHADD/UHADD]
    //       (a+b+1)>>1 = (a|b) - ((a^b)>>1)   [SRHADD/URHADD]
    //     Arithmetic shift (PSRAW/PSRAD) for signed, logical (PSRLW/PSRLD) for
    //     unsigned. All SSE2, no widening.
    //   * Byte ADD forms and every SUB form widen each 64-bit half to the next
    //     lane width (PMOVSXBW/PMOVZXBW etc., SSE4.1), do PADDW/PSUBW, shift
    //     right by 1, and PACK back with signed/unsigned saturation. The +1 of
    //     the rounding-add byte form is materialized by PCMPEQW ones + PSUBW
    //     (PAVGB is not an allowlisted heavy LIR op). SUB unsigned masks the low
    //     sub-lane before PACK so a modular a<b result does not saturate up.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShadd ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUhadd ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrhadd ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUrhadd ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kShsub ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUhsub) {
      if (args.size == 0b11) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShadd) ||
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrhadd) ||
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShsub);
      const bool is_round =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSrhadd) ||
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kUrhadd);
      const bool is_sub =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kShsub) ||
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kUhsub);
      // Byte-widened path needs SSE4.1 (PMOVSX/PMOVZX + PACKUSDW at size=01).
      const bool needs_widen = is_sub || (args.size == 0b00);
      if (needs_widen && !host_platform::kHasSSE4_1) {
        UndefinedReturningVoid();
        return;
      }

      // Halfword/word ADD forms: bitwise identity, no widening.
      if (!is_sub && args.size != 0b00) {
        FpRegister xn = AllocTempSimdReg();
        FpRegister xm = AllocTempSimdReg();
        FpRegister xtmp = AllocTempSimdReg();  // (a&b) for floor, (a^b) is in xn
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
        builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
        if (is_round) {
          // (a|b) - ((a^b)>>1): xtmp = a|b, xn = (a^b)>>1, xtmp -= xn.
          builder_.Gen<x86_64::PorXRegXReg>(xtmp.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());
          if (args.size == 0b01) {
            if (is_signed) builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), int8_t{1});
            else builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{1});
            builder_.Gen<x86_64::PsubwXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
          } else {  // size == 0b10
            if (is_signed) builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{1});
            else builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{1});
            builder_.Gen<x86_64::PsubdXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
          }
        } else {
          // (a&b) + ((a^b)>>1): xtmp = a&b, xn = (a^b)>>1, xtmp += xn.
          builder_.Gen<x86_64::PandXRegXReg>(xtmp.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xm.machine_reg());
          if (args.size == 0b01) {
            if (is_signed) builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), int8_t{1});
            else builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{1});
            builder_.Gen<x86_64::PaddwXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
          } else {  // size == 0b10
            if (is_signed) builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{1});
            else builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{1});
            builder_.Gen<x86_64::PadddXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
          }
        }
        SetVRegFull(args.rd, xtmp, args.q);
        return;
      }

      // Byte ADD forms and all SUB forms: widen each 64-bit half, op, shift, pack.
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      FpRegister xn_hi = AllocTempSimdReg();
      FpRegister xm_hi = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      builder_.Gen<x86_64::MovdqaXRegXReg>(xn_hi.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xm_hi.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PsrldqXRegImm>(xn_hi.machine_reg(), int8_t{8});
      builder_.Gen<x86_64::PsrldqXRegImm>(xm_hi.machine_reg(), int8_t{8});

      if (args.size == 0b00) {
        // 8->16 widen.
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
        }
        if (is_sub) {
          builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PsubwXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
        } else {
          builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PaddwXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
          if (is_round) {
            // +1 per word: xm/xm_hi are dead; clobber to all-ones and PSUBW.
            builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PsubwXRegXReg>(xn_hi.machine_reg(), xm.machine_reg());
          }
        }
        if (is_signed) {
          builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PsrawXRegImm>(xn_hi.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PacksswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        } else {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PsrlwXRegImm>(xn_hi.machine_reg(), int8_t{1});
          if (is_sub) {
            // Mask low byte per word so PACKUSWB does not saturate the modular
            // a<b result (in [0x7F80, 0x7FFF]) up to 0xFF.
            builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PandXRegXReg>(xn_hi.machine_reg(), xm.machine_reg());
          }
          builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        }
        SetVRegFull(args.rd, xn, args.q);
        return;
      }

      // SUB size=01 (halfword): 16->32 widen, PSUBD, shift, PACKSSDW/PACKUSDW.
      if (args.size == 0b01) {
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
          builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PsubdXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
          builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PsradXRegImm>(xn_hi.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PackssdwXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
          builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PsubdXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{1});
          builder_.Gen<x86_64::PsrldXRegImm>(xn_hi.machine_reg(), int8_t{1});
          // Mask low 16 bits per dword before PACKUSDW.
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{16});
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(xn_hi.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
        }
        SetVRegFull(args.rd, xn, args.q);
        return;
      }

      // SUB size=10 (word): 32->64 widen, PSUBQ, PSRLQ 1, gather low dwords via
      // PSHUFD 0x88 + PUNPCKLQDQ. PSRLQ is logical, but the bit-63 difference vs
      // an arithmetic shift is discarded by the low-dword gather, so PMOVSXDQ vs
      // PMOVZXDQ alone distinguishes signed/unsigned.
      if (is_signed) {
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
      } else {
        builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn_hi.machine_reg(), xn_hi.machine_reg());
        builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm_hi.machine_reg(), xm_hi.machine_reg());
      }
      builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::PsubqXRegXReg>(xn_hi.machine_reg(), xm_hi.machine_reg());
      builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), int8_t{1});
      builder_.Gen<x86_64::PsrlqXRegImm>(xn_hi.machine_reg(), int8_t{1});
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                              static_cast<int8_t>(0x88));
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xn_hi.machine_reg(), xn_hi.machine_reg(),
                                              static_cast<int8_t>(0x88));
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xn_hi.machine_reg());
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // ADDP (pairwise add) vector — mirror of lite_translator.h's kAddp. Vd =
    // pair(Vn) || pair(Vm) where pair(X)[i] = X[2i] + X[2i+1]. 8H/4S map to
    // PHADDW/PHADDD directly (SSSE3 — ARM's concat-then-pair layout); 4H/2S use
    // the same op then PSHUFD 0x08 to gather the two low pair-lanes into the low
    // 64 bits; byte lanes emulate via PSRLW-8 + PADDB (even bytes hold the pair
    // sums) then truncate each halfword's low byte and PACKUSWB; .2D (size=11
    // Q=1) has no PHADDQ, so PSHUFD 0xEE + PADDQ + PUNPCKLQDQ. .1D (size=11 Q=0)
    // is ARM-reserved. SetVRegFull's Q=0 merge zeroes Vd[127:64].
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kAddp) {
      // Halfword/word forms need PHADDW/PHADDD (SSSE3); byte and .2D are SSE2.
      if ((args.size == 0b01 || args.size == 0b10) && !host_platform::kHasSSSE3) {
        UndefinedReturningVoid();
        return;
      }
      if (args.size == 0b11 && !args.q) {  // .1D reserved
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      switch (args.size) {
        case 0b00: {
          FpRegister tmp_n = AllocTempSimdReg();
          FpRegister tmp_m = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(tmp_n.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(tmp_m.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PsrlwXRegImm>(tmp_n.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsrlwXRegImm>(tmp_m.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PaddbXRegXReg>(xn.machine_reg(), tmp_n.machine_reg());
          builder_.Gen<x86_64::PaddbXRegXReg>(xm.machine_reg(), tmp_m.machine_reg());
          builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsllwXRegImm>(xm.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});
          if (args.q) {
            builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PackuswbXRegXReg>(xm.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PunpckldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
        }
        case 0b01:
          builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          if (!args.q) {
            builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                    int8_t{0x08});
          }
          break;
        case 0b10:
          builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xm.machine_reg());
          if (!args.q) {
            builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                    int8_t{0x08});
          }
          break;
        case 0b11: {  // .2D (Q=1); .1D already bailed above.
          FpRegister tmp = AllocTempSimdReg();
          builder_.Gen<x86_64::PshufdXRegXRegImm>(tmp.machine_reg(), xn.machine_reg(),
                                                  static_cast<int8_t>(0xEE));
          builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), tmp.machine_reg());
          builder_.Gen<x86_64::PshufdXRegXRegImm>(tmp.machine_reg(), xm.machine_reg(),
                                                  static_cast<int8_t>(0xEE));
          builder_.Gen<x86_64::PaddqXRegXReg>(xm.machine_reg(), tmp.machine_reg());
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        }
      }
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // SMAXP/SMINP/UMAXP/UMINP (pairwise signed/unsigned max/min) — mirror of
    // lite_translator.h. Vd = pair(Vn) || pair(Vm), pair(X)[i] = op(X[2i],
    // X[2i+1]). No horizontal-pairwise min/max exists in SSE; gather even/odd
    // lanes (PSHUFB for byte/halfword, PSHUFD 0x88/0xDD for dword), lane-wise
    // PMAX/PMIN, then concatenate the Vn and Vm partials. size=11 (.2D) needs
    // 64-bit packed min/max (AVX-512) — bail to lite/interpreter.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kSminp ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmaxp ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUminp) {
      if (args.size == 0b11) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_max =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
           args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmaxp);
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmaxp ||
           args.opcode == Decoder::AdvSimdThreeSameOpcode::kSminp);
      // Feature gate mirrors lite: PMAXSB/PMINSB/PMAXSD/PMINSD/PMAXUW/PMINUW/
      // PMAXUD/PMINUD -> SSE4.1; PMAXSW/PMINSW/PMAXUB/PMINUB -> SSE2. Byte and
      // halfword even/odd gathers need PSHUFB (SSSE3).
      const bool needs_sse4_1 =
          (is_signed && args.size == 0b00) ||
          (is_signed && args.size == 0b10) ||
          (!is_signed && args.size == 0b01) ||
          (!is_signed && args.size == 0b10);
      if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
        UndefinedReturningVoid();
        return;
      }
      const bool needs_ssse3 = (args.size == 0b00 || args.size == 0b01);
      if (needs_ssse3 && !host_platform::kHasSSSE3) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      auto pmax_pmin = [&](FpRegister dst, FpRegister src) {
        switch (args.size) {
          case 0b00:
            if (is_signed) {
              if (is_max) builder_.Gen<x86_64::PmaxsbXRegXReg>(dst.machine_reg(), src.machine_reg());
              else builder_.Gen<x86_64::PminsbXRegXReg>(dst.machine_reg(), src.machine_reg());
            } else {
              if (is_max) builder_.Gen<x86_64::PmaxubXRegXReg>(dst.machine_reg(), src.machine_reg());
              else builder_.Gen<x86_64::PminubXRegXReg>(dst.machine_reg(), src.machine_reg());
            }
            break;
          case 0b01:
            if (is_signed) {
              if (is_max) builder_.Gen<x86_64::PmaxswXRegXReg>(dst.machine_reg(), src.machine_reg());
              else builder_.Gen<x86_64::PminswXRegXReg>(dst.machine_reg(), src.machine_reg());
            } else {
              if (is_max) builder_.Gen<x86_64::PmaxuwXRegXReg>(dst.machine_reg(), src.machine_reg());
              else builder_.Gen<x86_64::PminuwXRegXReg>(dst.machine_reg(), src.machine_reg());
            }
            break;
          case 0b10:
            if (is_signed) {
              if (is_max) builder_.Gen<x86_64::PmaxsdXRegXReg>(dst.machine_reg(), src.machine_reg());
              else builder_.Gen<x86_64::PminsdXRegXReg>(dst.machine_reg(), src.machine_reg());
            } else {
              if (is_max) builder_.Gen<x86_64::PmaxudXRegXReg>(dst.machine_reg(), src.machine_reg());
              else builder_.Gen<x86_64::PminudXRegXReg>(dst.machine_reg(), src.machine_reg());
            }
            break;
        }
      };
      if (args.size == 0b00 || args.size == 0b01) {
        // Byte/halfword: build PSHUFB even/odd gather masks from immediates
        // (upper qword 0x80 => PSHUFB writes zero there), gather each operand's
        // even and odd lanes into the low 8 bytes, PMAX/PMIN them to form the
        // 8-byte partial pair(Vn) / pair(Vm), then concatenate.
        FpRegister even_mask = AllocTempSimdReg();
        FpRegister odd_mask = AllocTempSimdReg();
        FpRegister evens_n = AllocTempSimdReg();
        FpRegister odds_n = AllocTempSimdReg();
        FpRegister evens_m = AllocTempSimdReg();
        FpRegister odds_m = AllocTempSimdReg();
        int64_t even_lo, even_hi, odd_lo, odd_hi;
        if (args.size == 0b00) {
          even_lo = static_cast<int64_t>(0x0E0C0A0806040200LL);
          even_hi = static_cast<int64_t>(0x8080808080808080ULL);
          odd_lo = static_cast<int64_t>(0x0F0D0B0907050301LL);
          odd_hi = static_cast<int64_t>(0x8080808080808080ULL);
        } else {
          even_lo = static_cast<int64_t>(0x0D0C090805040100LL);
          even_hi = static_cast<int64_t>(0x8080808080808080ULL);
          odd_lo = static_cast<int64_t>(0x0F0E0B0A07060302LL);
          odd_hi = static_cast<int64_t>(0x8080808080808080ULL);
        }
        Register elo = std::get<0>(Gen<x86_64::MovqRegImm>(even_lo));
        builder_.Gen<x86_64::MovqXRegReg>(even_mask.machine_reg(), elo);
        Register ehi = std::get<0>(Gen<x86_64::MovqRegImm>(even_hi));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(even_mask.machine_reg(), ehi, int8_t{1});
        Register olo = std::get<0>(Gen<x86_64::MovqRegImm>(odd_lo));
        builder_.Gen<x86_64::MovqXRegReg>(odd_mask.machine_reg(), olo);
        Register ohi = std::get<0>(Gen<x86_64::MovqRegImm>(odd_hi));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(odd_mask.machine_reg(), ohi, int8_t{1});

        builder_.Gen<x86_64::MovdqaXRegXReg>(evens_n.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PshufbXRegXReg>(evens_n.machine_reg(), even_mask.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(odds_n.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PshufbXRegXReg>(odds_n.machine_reg(), odd_mask.machine_reg());
        pmax_pmin(evens_n, odds_n);  // low 8 bytes hold pair(Vn)

        builder_.Gen<x86_64::MovdqaXRegXReg>(evens_m.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PshufbXRegXReg>(evens_m.machine_reg(), even_mask.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(odds_m.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PshufbXRegXReg>(odds_m.machine_reg(), odd_mask.machine_reg());
        pmax_pmin(evens_m, odds_m);  // low 8 bytes hold pair(Vm)

        if (args.q) {
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(evens_n.machine_reg(), evens_m.machine_reg());
        } else {
          builder_.Gen<x86_64::PunpckldqXRegXReg>(evens_n.machine_reg(), evens_m.machine_reg());
        }
        SetVRegFull(args.rd, evens_n, args.q);
        return;
      }
      // size == 0b10 (.4S / .2S): dword pairwise via PSHUFD even/odd lift.
      // 0x88 => {d0,d2,d0,d2}; 0xDD => {d1,d3,d1,d3}; PMAX/PMIN yields pair(X)
      // replicated in both halves. Q=1 concatenates the low qwords; Q=0 packs
      // dwords 0 and 2 into positions 0,1 with a final PSHUFD 0x08.
      FpRegister evens_n = AllocTempSimdReg();
      FpRegister odds_n = AllocTempSimdReg();
      FpRegister evens_m = AllocTempSimdReg();
      FpRegister odds_m = AllocTempSimdReg();
      builder_.Gen<x86_64::PshufdXRegXRegImm>(evens_n.machine_reg(), xn.machine_reg(),
                                              static_cast<int8_t>(0x88));
      builder_.Gen<x86_64::PshufdXRegXRegImm>(odds_n.machine_reg(), xn.machine_reg(),
                                              static_cast<int8_t>(0xDD));
      pmax_pmin(evens_n, odds_n);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(evens_m.machine_reg(), xm.machine_reg(),
                                              static_cast<int8_t>(0x88));
      builder_.Gen<x86_64::PshufdXRegXRegImm>(odds_m.machine_reg(), xm.machine_reg(),
                                              static_cast<int8_t>(0xDD));
      pmax_pmin(evens_m, odds_m);
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(evens_n.machine_reg(), evens_m.machine_reg());
      if (!args.q) {
        builder_.Gen<x86_64::PshufdXRegXRegImm>(evens_n.machine_reg(), evens_n.machine_reg(),
                                                int8_t{0x08});
      }
      SetVRegFull(args.rd, evens_n, args.q);
      return;
    }

    // SMAX/SMIN/UMAX/UMIN (plain, non-pairwise lane-wise signed/unsigned
    // min/max) — mirror of lite_translator.h. x86 has direct lane-width-matched
    // PMAXS/PMINS/PMAXU/PMINU for 8/16/32-bit widths; the .2D (64-bit) lane has
    // no SSE-era op (PMAXSQ/PMINSQ/PMAXUQ/PMINUQ are AVX-512F-VL only) — bail.
    // Per-size SSE feature gate (Intel SDM Vol 2):
    //   PMAXSB/PMINSB/PMAXSD/PMINSD/PMAXUW/PMINUW/PMAXUD/PMINUD -> SSE4.1;
    //   PMAXSW/PMINSW/PMAXUB/PMINUB                              -> SSE2.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax ||
        args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmin) {
      if (args.size == 0b11) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_max =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
           args.opcode == Decoder::AdvSimdThreeSameOpcode::kUmax);
      const bool is_signed =
          (args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmax ||
           args.opcode == Decoder::AdvSimdThreeSameOpcode::kSmin);
      const bool needs_sse4_1 =
          (is_signed && args.size == 0b00) ||
          (is_signed && args.size == 0b10) ||
          (!is_signed && args.size == 0b01) ||
          (!is_signed && args.size == 0b10);
      if (needs_sse4_1 && !host_platform::kHasSSE4_1) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      switch (args.size) {
        case 0b00:  // .16B / .8B
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminsbXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxubXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminubXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
        case 0b01:  // .8H / .4H
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
        case 0b10:  // .4S / .2S
          if (is_signed) {
            if (is_max) builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            if (is_max) builder_.Gen<x86_64::PmaxudXRegXReg>(xn.machine_reg(), xm.machine_reg());
            else builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
      }
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // BIC / ORN (vector bitwise AND-NOT / OR-NOT) — mirror of lite_translator.h.
    // Element-size-independent bit ops that read only Vn/Vm, but need the PANDN /
    // ones-materialize sequences rather than a single in-place op, so they live
    // here rather than the shared vn-only switch below.
    //   BIC  Vd = Vn AND NOT Vm.  x86 PANDN(dst, src) = ~dst & src, so
    //        PANDN(xm, xn) lands ~Vm & Vn in xm.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kBic) {
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      builder_.Gen<x86_64::PandnXRegXReg>(xm.machine_reg(), xn.machine_reg());
      SetVRegFull(args.rd, xm, args.q);
      return;
    }
    //   ORN  Vd = Vn OR NOT Vm.  Materialize all-ones via self-PCMPEQD on a
    //        pre-zeroed vreg (AllocZeroedSimdReg establishes a def before the
    //        self-compare), XOR into xm to get ~Vm, then OR with xn.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kOrn) {
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      FpRegister ones = AllocZeroedSimdReg();  // def before the self-compare
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(ones.machine_reg(), ones.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xm.machine_reg(), ones.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xm.machine_reg());
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // CMTST (test bits) — mirror of lite_translator.h. Vd = (Vn & Vm) != 0 ?
    // all-ones : 0 per lane. PAND, compare the AND result against zero (yielding
    // all-ones where the lane IS zero), then invert so the non-zero lanes are
    // set. Byte/halfword/word use PCMPEQ{B,W,D} (SSE2); the .2D (size=11) form
    // uses PCMPEQQ (SSE4.1) — bail cleanly if absent, matching lite.
    if (args.opcode == Decoder::AdvSimdThreeSameOpcode::kCmtst) {
      if (args.size == 0b11 && !host_platform::kHasSSE4_1) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());  // Vn & Vm
      FpRegister z = AllocZeroedSimdReg();  // zero, proper def for the self-compare below
      switch (args.size) {
        case 0b00: builder_.Gen<x86_64::PcmpeqbXRegXReg>(xn.machine_reg(), z.machine_reg()); break;
        case 0b01: builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn.machine_reg(), z.machine_reg()); break;
        case 0b10: builder_.Gen<x86_64::PcmpeqdXRegXReg>(xn.machine_reg(), z.machine_reg()); break;
        default:   builder_.Gen<x86_64::PcmpeqqXRegXReg>(xn.machine_reg(), z.machine_reg()); break;
      }
      // z is now dead as the zero comparand; clobber it to all-ones (it already
      // has a def, so the self-PCMPEQD is legal) and XOR to invert the mask.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(z.machine_reg(), z.machine_reg());  // z = -1
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), z.machine_reg());
      SetVRegFull(args.rd, xn, args.q);
      return;
    }

    // Validate the (opcode, size) pair up front and emit nothing on bail. After
    // this switch every reachable case has a single allowlisted packed op.
    switch (args.opcode) {
      case Decoder::AdvSimdThreeSameOpcode::kAdd:
      case Decoder::AdvSimdThreeSameOpcode::kSub:
        // ADD -> Padd{b,w,d,q}; SUB -> Psub{b,w,d,q}. All four element sizes
        // (8/16/32/64) have a direct SSE2 packed op, so every size is handled
        // (mirrors lite_translator.h). The reserved .1D shape (size=11, Q=0) is
        // UNALLOCATED; like lite we don't special-case it — SetVRegFull's Q=0
        // merge just zero-extends the low 64 bits.
        break;
      case Decoder::AdvSimdThreeSameOpcode::kMul:
        if (args.size != 0b01 && args.size != 0b10) {
          UndefinedReturningVoid();
          return;
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kAnd:
      case Decoder::AdvSimdThreeSameOpcode::kOrr:
      case Decoder::AdvSimdThreeSameOpcode::kEor:
        // Bitwise: element size is irrelevant; all forms are handled.
        break;
      case Decoder::AdvSimdThreeSameOpcode::kCmeq:
      case Decoder::AdvSimdThreeSameOpcode::kCmgt:
        // CMEQ -> PCMPEQ{B,W,D}; CMGT (signed) -> PCMPGT{B,W,D}. The 64-bit (2D)
        // form needs PCMPEQQ/PCMPGTQ (SSE4.1/4.2), which are not in the backend
        // allowlist, so it bails to the lite tier.
        if (args.size == 0b11) {
          UndefinedReturningVoid();
          return;
        }
        break;
      default:
        UndefinedReturningVoid();
        return;
    }

    FpRegister vn = AllocTempSimdReg();
    FpRegister vm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(vn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(vm.machine_reg(), vm_off);

    // Run the packed op in place on vn (vn := vn OP vm).
    switch (args.opcode) {
      case Decoder::AdvSimdThreeSameOpcode::kAdd:
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PaddbXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else if (args.size == 0b01) {
          builder_.Gen<x86_64::PaddwXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else if (args.size == 0b10) {
          builder_.Gen<x86_64::PadddXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else {
          builder_.Gen<x86_64::PaddqXRegXReg>(vn.machine_reg(), vm.machine_reg());
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kSub:
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PsubbXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else if (args.size == 0b01) {
          builder_.Gen<x86_64::PsubwXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else if (args.size == 0b10) {
          builder_.Gen<x86_64::PsubdXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else {
          builder_.Gen<x86_64::PsubqXRegXReg>(vn.machine_reg(), vm.machine_reg());
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kMul:
        if (args.size == 0b01) {
          builder_.Gen<x86_64::PmullwXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmulldXRegXReg>(vn.machine_reg(), vm.machine_reg());
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kAnd:
        builder_.Gen<x86_64::PandXRegXReg>(vn.machine_reg(), vm.machine_reg());
        break;
      case Decoder::AdvSimdThreeSameOpcode::kOrr:
        builder_.Gen<x86_64::PorXRegXReg>(vn.machine_reg(), vm.machine_reg());
        break;
      case Decoder::AdvSimdThreeSameOpcode::kEor:
        builder_.Gen<x86_64::PxorXRegXReg>(vn.machine_reg(), vm.machine_reg());
        break;
      case Decoder::AdvSimdThreeSameOpcode::kCmeq:
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PcmpeqbXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else if (args.size == 0b01) {
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else {
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(vn.machine_reg(), vm.machine_reg());
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kCmgt:
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PcmpgtbXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else if (args.size == 0b01) {
          builder_.Gen<x86_64::PcmpgtwXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else {
          builder_.Gen<x86_64::PcmpgtdXRegXReg>(vn.machine_reg(), vm.machine_reg());
        }
        break;
      default:
        // Unreachable: the validation switch above already bailed.
        UndefinedReturningVoid();
        return;
    }

    SetVRegFull(args.rd, vn, args.q);
  }

  // Heavy-tier mirror of the register-domain-pure subset of
  // lite_translator.h::AdvSimdThreeDiff:
  //   {S,U}MULL{,2}, {S,U}MLAL{,2}, {S,U}MLSL{,2}   (widening multiply-accumulate)
  //   {S,U}ADDL{,2}, {S,U}SUBL{,2}                  (widening add/sub, both narrow)
  //   {S,U}ADDW{,2}, {S,U}SUBW{,2}                  (widening add/sub, Vn wide)
  // at input sizes 8/16/32 (size 00/01/10) and both Q halves. The widening
  // recipe matches lite exactly: widen the narrow sources (PMOVSX/PMOVZX per
  // sign), then for the multiply subset lane-wise multiply
  // (PMULLW / PMULLD / PMULDQ|PMULUDQ) — MLAL/MLSL load Vd and PADD/PSUB the
  // product — and for the add/sub subset PADD/PSUB the widened operands
  // directly at the wide lane width. The result always fills 128 bits
  // (8H/4S/2D), so SetVRegFull q=true regardless of Q. Q=1 ("2") forms take
  // the upper 64 of the narrow sources — bring bytes 8..15 down with PSRLDQ
  // before widening (the W-forms' Vn is already a 128-bit wide vector and is
  // loaded full, never shifted).
  //
  // Also mirrors the narrowing-high subset:
  //   ADDHN/SUBHN/RADDHN/RSUBHN — add/sub the two wide-lane sources, then take
  //   the HIGH half of each lane as the narrow result (rounding variants add a
  //   half-ulp bias first). Committed via SetVRegNarrow (Q=0 zero-extend / Q2
  //   merge into Vd.high).
  //
  // Also mirrors the saturating-doubling widening subset:
  //   SQDMULL/SQDMLAL/SQDMLSL — signed saturating doubling multiply long, with
  //   the accumulate/subtract forms saturating again on the add. size 01
  //   (.4H->.4S, manual 32-bit saturation) and size 10 (.2S->.2D, manual 64-bit
  //   saturation); size 00/11 are decoder-rejected. The remaining ThreeDiff
  //   opcodes (ABDL/ABAL, PMULL) still bail to the lite tier — emit NOTHING
  //   before a bail.
  void AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& args) {
    if (!success()) {
      return;
    }
    using Op = Decoder::AdvSimdThreeDiffOpcode;

    // Widening add/sub subset: {S,U}ADDL/{S,U}SUBL (both sources narrow) and
    // {S,U}ADDW/{S,U}SUBW (Vn already a wide-lane 128-bit vector). Handle first;
    // fall through to the multiply-accumulate subset otherwise.
    const bool is_addl = (args.opcode == Op::kSaddl || args.opcode == Op::kUaddl);
    const bool is_subl = (args.opcode == Op::kSsubl || args.opcode == Op::kUsubl);
    const bool is_addw = (args.opcode == Op::kSaddw || args.opcode == Op::kUaddw);
    const bool is_subw = (args.opcode == Op::kSsubw || args.opcode == Op::kUsubw);
    if (is_addl || is_subl || is_addw || is_subw) {
      if (args.size > 0b10) {
        UndefinedReturningVoid();
        return;
      }
      const bool addsub_signed = (args.opcode == Op::kSaddl ||
                                  args.opcode == Op::kSsubl ||
                                  args.opcode == Op::kSaddw ||
                                  args.opcode == Op::kSsubw);
      const bool is_add = (is_addl || is_addw);
      const bool n_is_wide = (is_addw || is_subw);
      const int32_t vn_off_as =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
      const int32_t vm_off_as =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);

      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off_as);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off_as);
      // Q=1 selects the upper 64 of the *narrow* sources; the W-forms' Vn is
      // already wide, so only shift it for the L-forms.
      if (args.q) {
        if (!n_is_wide) {
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
        }
        builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
      }
      // Widen Vn (skip for W-forms — Vn is already a wide-lane vector).
      if (!n_is_wide) {
        switch (args.size) {
          case 0b00:
            if (addsub_signed) builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            else builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            break;
          case 0b01:
            if (addsub_signed) builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            else builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            break;
          case 0b10:
            if (addsub_signed) builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
            else builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
            break;
        }
      }
      // Widen Vm (always narrow).
      switch (args.size) {
        case 0b00:
          if (addsub_signed) builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
          break;
        case 0b01:
          if (addsub_signed) builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
          break;
        case 0b10:
          if (addsub_signed) builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
          break;
      }
      // xn = Vn +/- Vm at the wide lane width.
      switch (args.size) {
        case 0b00:
          if (is_add) builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        case 0b01:
          if (is_add) builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        case 0b10:
          if (is_add) builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          else builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
      }
      SetVRegFull(args.rd, xn, /*q=*/true);
      return;
    }

    // SABDL/UABDL (absolute-difference-long) and SABAL/UABAL (abs-diff-long
    // accumulate). Widen both narrow sources (Q selects the low/high 64 of
    // Vn/Vm), then abs(a - b) at the widened lane width:
    //   size=00/01: max(a,b) - min(a,b) via PMAXS*/PMINS* (signed) or
    //               PMAXU*/PMINU* (unsigned), then PSUB.
    //   size=10   : no 64-bit SSE lane-wise max/min, so diff = a - b (PSUBQ
    //               after the sign/zero-widening), then a Pcmpgtq-against-zero
    //               signed-abs: mask = (0 > diff); abs = (diff ^ mask) - mask.
    // ABAL then accumulates the abs-diff into Vd. The result always fills 128
    // bits (8H/4S/2D). Mirrors lite_translator.h::AdvSimdThreeDiff's
    // SABDL/UABDL/SABAL/UABAL block size-by-size.
    if (args.opcode == Op::kSabdl || args.opcode == Op::kUabdl ||
        args.opcode == Op::kSabal || args.opcode == Op::kUabal) {
      if (args.size > 0b10) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_signed =
          (args.opcode == Op::kSabdl || args.opcode == Op::kSabal);
      const bool is_abal =
          (args.opcode == Op::kSabal || args.opcode == Op::kUabal);
      const int32_t vn_off_ab =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
      const int32_t vm_off_ab =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);
      const int32_t vd_off_ab =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);

      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off_ab);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off_ab);
      // Q=1 ("2" form) selects the upper 64 of the narrow sources.
      if (args.q) {
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
      }
      // Widen both narrow sources to the wide lane width.
      switch (args.size) {
        case 0b00:
          if (is_signed) {
            builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
          }
          break;
        case 0b01:
          if (is_signed) {
            builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
          }
          break;
        case 0b10:
          if (is_signed) {
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
          }
          break;
      }

      if (args.size == 0b10) {
        // 32->64: diff = a - b at 64-bit lane width, then signed-abs via
        // Pcmpgtq against zero. Correct for both signs because the widening
        // already injected the correct sign/zero extension.
        builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        FpRegister mask = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PcmpgtqXRegXReg>(mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), mask.machine_reg());
        builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), mask.machine_reg());
        if (is_abal) {
          FpRegister xd = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xd.machine_reg(), vd_off_ab);
          builder_.Gen<x86_64::PaddqXRegXReg>(xd.machine_reg(), xn.machine_reg());
          SetVRegFull(args.rd, xd, /*q=*/true);
        } else {
          SetVRegFull(args.rd, xn, /*q=*/true);
        }
        return;
      }

      // 8->16 / 16->32: abs diff = max(a,b) - min(a,b). Copy Vn into xmax
      // (max clobbers its dst), leaving xn to hold the min.
      FpRegister xmax = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(xmax.machine_reg(), xn.machine_reg());
      if (args.size == 0b00) {
        if (is_signed) {
          builder_.Gen<x86_64::PmaxswXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmaxuwXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        builder_.Gen<x86_64::PsubwXRegXReg>(xmax.machine_reg(), xn.machine_reg());
      } else {  // size == 0b01
        if (is_signed) {
          builder_.Gen<x86_64::PmaxsdXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmaxudXRegXReg>(xmax.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        builder_.Gen<x86_64::PsubdXRegXReg>(xmax.machine_reg(), xn.machine_reg());
      }
      if (is_abal) {
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off_ab);
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xmax.machine_reg());
        } else {
          builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xmax.machine_reg());
        }
        SetVRegFull(args.rd, xd, /*q=*/true);
      } else {
        SetVRegFull(args.rd, xmax, /*q=*/true);
      }
      return;
    }

    // ADDHN/SUBHN/RADDHN/RSUBHN — add/sub the two wide-lane sources, then take
    // the HIGH half of each lane as the narrow result. size 00/01/10 selects
    // source lane 16/32/64 -> narrow dst 8/16/32. Rounding variants add a
    // half-ulp bias (1 << (esize-1) of the SOURCE lane, i.e. bit just below the
    // >>esize truncation point) before the shift. size=00: PADDW/PSUBW, >>8,
    // PACKUSWB gathers the 8 high-bytes to the low 64. size=01: PADDD/PSUBD,
    // >>16, PACKUSDW. size=10: PADDQ/PSUBQ, >>32, PSHUFD 0b00001000 gathers
    // dwords {0,2} to the low 64 (no 64->32 pack). The narrow result is
    // committed via SetVRegNarrow (Q=0 zero-extend / Q2 merge into Vd.high).
    // Mirrors lite_translator.h::AdvSimdThreeDiff's ADDHN/SUBHN block.
    if (args.opcode == Op::kAddhn || args.opcode == Op::kSubhn ||
        args.opcode == Op::kRaddhn || args.opcode == Op::kRsubhn) {
      if (args.size > 0b10) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_sub = (args.opcode == Op::kSubhn || args.opcode == Op::kRsubhn);
      const bool is_round = (args.opcode == Op::kRaddhn || args.opcode == Op::kRsubhn);
      const int32_t vn_o =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
      const int32_t vm_o =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_o);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_o);

      // Materialize a broadcast constant (`pattern` in both qwords) for the
      // rounding bias.
      auto broadcast = [&](uint64_t pattern) -> FpRegister {
        FpRegister x = AllocTempSimdReg();
        Register gr =
            std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pattern)));
        builder_.Gen<x86_64::MovqXRegReg>(x.machine_reg(), gr);
        builder_.Gen<x86_64::PinsrqXRegRegImm>(x.machine_reg(), gr, int8_t{1});
        return x;
      };

      if (args.size == 0b10) {
        if (is_sub) {
          builder_.Gen<x86_64::PsubqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        if (is_round) {
          FpRegister xr = broadcast(uint64_t{0x0000000080000000ULL});
          builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xr.machine_reg());
        }
        builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), int8_t{32});
        builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                int8_t{0b00001000});
        SetVRegNarrow(args.rd, xn, args.q);
        return;
      }
      if (args.size == 0b00) {
        if (is_sub) {
          builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        if (is_round) {
          FpRegister xr = broadcast(uint64_t{0x0080008000800080ULL});
          builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xr.machine_reg());
        }
        builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{8});
        FpRegister xz = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
      } else {  // size == 0b01
        if (is_sub) {
          builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        if (is_round) {
          FpRegister xr = broadcast(uint64_t{0x0000800000008000ULL});
          builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xr.machine_reg());
        }
        builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{16});
        FpRegister xz = AllocZeroedSimdReg();
        builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
      }
      SetVRegNarrow(args.rd, xn, args.q);
      return;
    }

    // SQDMULL/SQDMLAL/SQDMLSL — signed saturating doubling multiply long (+
    // saturating accumulate/subtract). Line-by-line mirror of
    // lite_translator.h::AdvSimdThreeDiff's SQDMULL block. Only size=01
    // (.4H->.4S) and size=10 (.2S->.2D) are defined (size 00/11 are
    // decoder-rejected). The exact lane products are doubled with manual
    // saturation of the single INT_MIN^2 overflow lane (product == 2^30 / 2^62
    // -> SMAX), then the accumulate forms apply a second signed-saturating
    // add/sub via the (a^P)&(a^res) sign-bit overflow-detection idiom. The
    // "long" result always fills 128 bits, so SetVRegFull q=true regardless of
    // the Q ("2") bit, which only selects the upper half of the narrow sources.
    if (args.opcode == Op::kSqdmull || args.opcode == Op::kSqdmlal ||
        args.opcode == Op::kSqdmlsl) {
      if (args.size != 0b01 && args.size != 0b10) {
        UndefinedReturningVoid();
        return;
      }
      const bool is_acc = (args.opcode != Op::kSqdmull);
      const bool is_sub = (args.opcode == Op::kSqdmlsl);
      const int32_t vn_o =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
      const int32_t vm_o =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);
      const int32_t vd_o =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);

      // Materialize a broadcast constant (`pattern` in both qwords).
      auto broadcast = [&](uint64_t pattern) -> FpRegister {
        FpRegister x = AllocTempSimdReg();
        Register gr =
            std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pattern)));
        builder_.Gen<x86_64::MovqXRegReg>(x.machine_reg(), gr);
        builder_.Gen<x86_64::PinsrqXRegRegImm>(x.machine_reg(), gr, int8_t{1});
        return x;
      };

      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_o);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_o);
      // Q=1 ("2") forms take the upper 64 bits of the narrow sources.
      if (args.q) {
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
      }

      // Compute the doubled, saturated product P into xP (128 bits: 4S or 2D).
      FpRegister xP = broadcast(args.size == 0b01 ? uint64_t{0x4000000040000000ULL}
                                                  : uint64_t{0x4000000000000000ULL});
      if (args.size == 0b01) {
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
        // xP starts as INT16_MIN^2 (0x40000000 per lane); mark == lanes.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(xP.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xn.machine_reg());  // double
        FpRegister xsat = broadcast(uint64_t{0x7FFFFFFF7FFFFFFFULL});
        builder_.Gen<x86_64::PandXRegXReg>(xsat.machine_reg(), xP.machine_reg());   // SMAX in sat lanes
        builder_.Gen<x86_64::PandnXRegXReg>(xP.machine_reg(), xn.machine_reg());    // doubled in non-sat
        builder_.Gen<x86_64::PorXRegXReg>(xP.machine_reg(), xsat.machine_reg());    // xP = P
      } else {  // size == 0b10
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PmuldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        // xP starts as INT32_MIN^2 (2^62 per qword); mark == lanes.
        builder_.Gen<x86_64::PcmpeqqXRegXReg>(xP.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xn.machine_reg());  // double
        FpRegister xsat = broadcast(uint64_t{0x7FFFFFFFFFFFFFFFULL});
        builder_.Gen<x86_64::PandXRegXReg>(xsat.machine_reg(), xP.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(xP.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(xP.machine_reg(), xsat.machine_reg());
      }

      if (!is_acc) {
        SetVRegFull(args.rd, xP, /*q=*/true);
        return;
      }

      // Signed saturating accumulate: result = SignedSat(Vd +/- P). a = Vd.
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_o);
      FpRegister xres = AllocTempSimdReg();
      FpRegister xof = AllocTempSimdReg();
      FpRegister xtmp = AllocTempSimdReg();
      if (args.size == 0b01) {
        if (is_sub) {
          builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), xP.machine_reg());   // diff = a - P
          builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xP.machine_reg());     // a ^ P
          builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // a ^ diff
          builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
        } else {
          builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::PadddXRegXReg>(xres.machine_reg(), xP.machine_reg());   // sum = a + P
          builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xres.machine_reg());   // a ^ sum
          builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xP.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // P ^ sum
          builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
        }
        builder_.Gen<x86_64::PsradXRegImm>(xof.machine_reg(), int8_t{31});  // overflow lanes
        builder_.Gen<x86_64::PsradXRegImm>(xd.machine_reg(), int8_t{31});   // a's sign (0 / -1)
        FpRegister xmaxc = broadcast(uint64_t{0x7FFFFFFF7FFFFFFFULL});
        builder_.Gen<x86_64::PxorXRegXReg>(xd.machine_reg(), xmaxc.machine_reg());   // sat = (a>>31)^INT32_MAX
        builder_.Gen<x86_64::PandXRegXReg>(xd.machine_reg(), xof.machine_reg());     // sat in overflow lanes
        builder_.Gen<x86_64::PandnXRegXReg>(xof.machine_reg(), xres.machine_reg());  // result in non-overflow
        builder_.Gen<x86_64::PorXRegXReg>(xd.machine_reg(), xof.machine_reg());
        SetVRegFull(args.rd, xd, /*q=*/true);
        return;
      }
      // size == 0b10 accumulate (64-bit signed saturating).
      FpRegister xzero = AllocZeroedSimdReg();
      if (is_sub) {
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PsubqXRegXReg>(xres.machine_reg(), xP.machine_reg());   // diff = a - P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xP.machine_reg());     // a ^ P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // a ^ diff
        builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
      } else {
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xP.machine_reg());   // sum = a + P
        builder_.Gen<x86_64::MovdqaXRegXReg>(xof.machine_reg(), xd.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xof.machine_reg(), xres.machine_reg());   // a ^ sum
        builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xP.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xtmp.machine_reg(), xres.machine_reg());  // P ^ sum
        builder_.Gen<x86_64::PandXRegXReg>(xof.machine_reg(), xtmp.machine_reg());
      }
      // overflow_mask (xtmp) = all-ones per qword where xof < 0 (sign set).
      builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xzero.machine_reg());
      builder_.Gen<x86_64::PcmpgtqXRegXReg>(xtmp.machine_reg(), xof.machine_reg());
      // sat = sign(a) ^ INT64_MAX  (INT64_MIN if a<0, else INT64_MAX).
      builder_.Gen<x86_64::PcmpgtqXRegXReg>(xzero.machine_reg(), xd.machine_reg());  // all-ones where a<0
      FpRegister xmaxc = broadcast(uint64_t{0x7FFFFFFFFFFFFFFFULL});
      builder_.Gen<x86_64::PxorXRegXReg>(xzero.machine_reg(), xmaxc.machine_reg());  // xzero = sat value
      builder_.Gen<x86_64::PandXRegXReg>(xzero.machine_reg(), xtmp.machine_reg());   // sat in overflow lanes
      builder_.Gen<x86_64::PandnXRegXReg>(xtmp.machine_reg(), xres.machine_reg());   // result in non-overflow
      builder_.Gen<x86_64::PorXRegXReg>(xzero.machine_reg(), xtmp.machine_reg());
      SetVRegFull(args.rd, xzero, /*q=*/true);
      return;
    }

    const bool is_mull = (args.opcode == Op::kSmull || args.opcode == Op::kUmull);
    const bool is_mlal = (args.opcode == Op::kSmlal || args.opcode == Op::kUmlal);
    const bool is_mlsl = (args.opcode == Op::kSmlsl || args.opcode == Op::kUmlsl);
    if ((!is_mull && !is_mlal && !is_mlsl) || args.size > 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_signed = (args.opcode == Op::kSmull ||
                            args.opcode == Op::kSmlal ||
                            args.opcode == Op::kSmlsl);
    const int32_t vn_off =
        static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
    const int32_t vm_off =
        static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);

    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    if (args.q) {
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
      builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
    }
    switch (args.size) {
      case 0b00:
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxbwXRegXReg>(xm.machine_reg(), xm.machine_reg());
        }
        builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      case 0b01:
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(xm.machine_reg(), xm.machine_reg());
        }
        builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
        break;
      case 0b10:
        if (is_signed) {
          builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovsxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PmuldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PmovzxdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PmuludqXRegXReg>(xn.machine_reg(), xm.machine_reg());
        }
        break;
    }

    if (is_mull) {
      SetVRegFull(args.rd, xn, /*q=*/true);
      return;
    }

    const int32_t vd_off =
        static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
    FpRegister xd = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
    switch (args.size) {
      case 0b00:
        if (is_mlal) {
          builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PsubwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
        break;
      case 0b01:
        if (is_mlal) {
          builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PsubdXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
        break;
      case 0b10:
        if (is_mlal) {
          builder_.Gen<x86_64::PaddqXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PsubqXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
        break;
    }
    SetVRegFull(args.rd, xd, /*q=*/true);
  }

  // Heavy-tier mirror of the lite AdvSimdSingleStruct lowering
  // (lite_translator.h). Covers the same subset the lite tier does: LD1R
  // (replicate one element to every lane), single-element LD1 (load one lane),
  // and single-element ST1 (store one lane), all with num_regs == 1 — critical
  // for the dynamic linker's calculate_gnu_hash_neon tail (`ld1r v3.4s, [x10],
  // #4`), which otherwise bails heavy->lite on every symbol resolve. Every
  // element move routes mem<->lane through a GP temp (the heavy tier has no
  // memory-operand PINSR/PEXTR, only the register forms), each guest-memory
  // access carries a recovery block, and the v[] slot is always accessed
  // full-width (GenGetSimd<16>/GenSetSimd<16>) so the optimizer's 16-byte
  // slot store/load forwarding is never split by a narrow sub-lane access
  // (see the UMOV/INS-element note above). LD1R starts from a zeroed XMM and
  // only fills the active lanes, so Q=0 upper-half zeroing is automatic.
  void AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs& args) {
    if (!success()) {
      return;
    }
    using Op = Decoder::AdvSimdSingleStructOp;
    const bool is_replicate = (args.op == Op::kLd1r);
    const bool is_single_load = (args.op == Op::kLd1);
    const bool is_single_store = (args.op == Op::kSt1);
    if (!is_replicate && !is_single_load && !is_single_store) {
      UndefinedReturningVoid();
      return;
    }
    if (args.num_regs != 1 || args.size > 3) {
      UndefinedReturningVoid();
      return;
    }
    const int esize = 1 << args.size;  // 1/2/4/8 bytes (B/H/S/D).
    const int32_t vec_bytes = args.q ? 16 : 8;
    const int32_t vt_off =
        static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rt * 16);

    Register base_orig = (args.rn == 31) ? GetSp() : GetReg(args.rn);
    Register base = ApplyTbi(base_orig);

    if (is_single_store) {
      // ST1 lane: full-width v[rt] load, PEXTR lane->gp, MOV* gp->mem (+recovery).
      FpRegister xmm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xmm.machine_reg(), vt_off);
      const int8_t lane = static_cast<int8_t>(args.index);
      Register elem;
      switch (esize) {
        case 1:
          elem = std::get<0>(Gen<x86_64::PextrbRegXRegImm>(xmm.machine_reg(), lane));
          Gen<x86_64::MovbOpReg>({.base = base, .disp = 0}, elem);
          break;
        case 2:
          elem = std::get<0>(Gen<x86_64::PextrwRegXRegImm>(xmm.machine_reg(), lane));
          Gen<x86_64::MovwOpReg>({.base = base, .disp = 0}, elem);
          break;
        case 4:
          elem = std::get<0>(Gen<x86_64::PextrdRegXRegImm>(xmm.machine_reg(), lane));
          Gen<x86_64::MovlOpReg>({.base = base, .disp = 0}, elem);
          break;
        default:  // esize == 8
          elem = std::get<0>(Gen<x86_64::PextrqRegXRegImm>(xmm.machine_reg(), lane));
          Gen<x86_64::MovqOpReg>({.base = base, .disp = 0}, elem);
          break;
      }
      GenRecoveryBlockForLastInsn();
    } else if (is_replicate) {
      // LD1R: load one element mem->gp (+recovery), then PINSR it into every
      // lane. Starting from a zeroed XMM leaves the upper 64 bits zero for Q=0.
      const int num_lanes = vec_bytes / esize;
      FpRegister xmm = AllocZeroedSimdReg();
      Register elem;
      switch (esize) {
        case 1:
          elem = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = base, .disp = 0}));
          break;
        case 2:
          elem = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = base, .disp = 0}));
          break;
        case 4:
          elem = std::get<0>(Gen<x86_64::MovlRegOp>({.base = base, .disp = 0}));
          break;
        default:  // esize == 8
          elem = std::get<0>(Gen<x86_64::MovqRegOp>({.base = base, .disp = 0}));
          break;
      }
      GenRecoveryBlockForLastInsn();
      for (int l = 0; l < num_lanes; l++) {
        const int8_t lane = static_cast<int8_t>(l);
        switch (esize) {
          case 1:
            builder_.Gen<x86_64::PinsrbXRegRegImm>(xmm.machine_reg(), elem, lane);
            break;
          case 2:
            builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), elem, lane);
            break;
          case 4:
            builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), elem, lane);
            break;
          default:  // esize == 8
            builder_.Gen<x86_64::PinsrqXRegRegImm>(xmm.machine_reg(), elem, lane);
            break;
        }
      }
      builder_.GenSetSimd<16>(vt_off, xmm.machine_reg());
    } else {
      // LD1 single lane: load full-width v[rt], PINSR the loaded element into
      // lane[index] (preserving the other lanes), store back full-width.
      FpRegister xmm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xmm.machine_reg(), vt_off);
      const int8_t lane = static_cast<int8_t>(args.index);
      Register elem;
      switch (esize) {
        case 1:
          elem = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = base, .disp = 0}));
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::PinsrbXRegRegImm>(xmm.machine_reg(), elem, lane);
          break;
        case 2:
          elem = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = base, .disp = 0}));
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), elem, lane);
          break;
        case 4:
          elem = std::get<0>(Gen<x86_64::MovlRegOp>({.base = base, .disp = 0}));
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), elem, lane);
          break;
        default:  // esize == 8
          elem = std::get<0>(Gen<x86_64::MovqRegOp>({.base = base, .disp = 0}));
          GenRecoveryBlockForLastInsn();
          builder_.Gen<x86_64::PinsrqXRegRegImm>(xmm.machine_reg(), elem, lane);
          break;
      }
      builder_.GenSetSimd<16>(vt_off, xmm.machine_reg());
    }

    if (args.postindex) {
      // Writeback preserves the original (un-TBI-masked) top byte, so re-read
      // the base register rather than reusing the masked access address.
      Register reread_base = (args.rn == 31) ? GetSp() : GetReg(args.rn);
      Register new_base = Copy(reread_base);
      if (args.rm == 31) {
        // Immediate post-index: total bytes accessed = num_regs * esize.
        new_base = std::get<0>(Gen<x86_64::AddqRegImm, kNoSSA>(
            new_base, static_cast<int32_t>(args.num_regs) * esize));
      } else {
        Register rm_val = GetReg(args.rm);
        new_base = std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(new_base, rm_val));
      }
      if (args.rn == 31) {
        SetSp(new_base);
      } else {
        SetReg(args.rn, new_base);
      }
    }
  }

  // Heavy-tier mirror of the register-domain-pure AdvSIMD two-reg-misc opcodes
  // the lite translator already lowers: REV16, CNT, NOT/RBIT, NEG, ABS. All are
  // packed SSE sequences with no host FLAGS and no memory operand beyond the
  // guest v[] load/store, so they map straight onto the MachineIR builder the
  // same way AdvSimdThreeSame does. Every other opcode bails to the lite tier
  // (UndefinedReturningVoid) — emit NOTHING before a bail. Q=0 upper-half zeroing
  // is handled by SetVRegFull. The lowerings match lite_translator.h exactly.
  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& args) {
    if (!success()) {
      return;
    }
    const int32_t vn_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);

    switch (args.opcode) {
      // REV16 V.<T>, V.<T> (size=00 only): reverse byte order within each 16-bit
      // lane via (Vn << 8) | (Vn >> 8) per halfword. PSLLW/PSRLW shift the 16-bit
      // lanes; OR recombines the swapped bytes without a PSHUFB mask table.
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev16: {
        if (args.size != 0b00) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        FpRegister xt = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.Gen<x86_64::MovdqaXRegXReg>(xt.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PsrlwXRegImm>(xt.machine_reg(), int8_t{8});
        builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xt.machine_reg());
        SetVRegFull(args.rd, xn, args.q);
        return;
      }

      // REV64 V.<T>, V.<T> (U=0): reverse element order within each 64-bit
      // doubleword. size=00 byte-reverse via PSHUFB + a materialized per-lane
      // byte-index mask; size=01 halfword-reverse via PSHUFLW then PSHUFHW
      // (imm=0x1B reverses the four words in each 64-bit half); size=10
      // word-reverse via PSHUFD (imm=0xB1 swaps the two 32-bit words in each
      // 64-bit half). The Q=0 forms shuffle both halves and let SetVRegFull
      // discard the upper 64. size=11 is reserved and bails.
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev64: {
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        switch (args.size) {
          case 0b00: {
            FpRegister xmask = AllocTempSimdReg();
            Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0001020304050607LL}));
            builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), mlo);
            Register mhi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x08090A0B0C0D0E0FLL}));
            builder_.Gen<x86_64::PinsrqXRegRegImm>(xmask.machine_reg(), mhi, int8_t{1});
            builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), xmask.machine_reg());
            break;
          }
          case 0b01:
            builder_.Gen<x86_64::PshuflwXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                     int8_t{0x1B});
            builder_.Gen<x86_64::PshufhwXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                     int8_t{0x1B});
            break;
          case 0b10:
            builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                    static_cast<int8_t>(0xB1));
            break;
          default:  // 0b11 reserved
            UndefinedReturningVoid();
            return;
        }
        SetVRegFull(args.rd, xn, args.q);
        return;
      }

      // REV32 V.<T>, V.<T> (U=1): reverse element order within each 32-bit
      // word. size=00 byte-reverse via PSHUFB + mask; size=01 halfword-swap
      // via PSHUFLW then PSHUFHW (imm=0xB1 swaps the two words in each 32-bit
      // lane). size>=10 is reserved for REV32 and bails.
      case Decoder::AdvSimdTwoRegMiscOpcode::kRev32: {
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        switch (args.size) {
          case 0b00: {
            FpRegister xmask = AllocTempSimdReg();
            Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0405060700010203LL}));
            builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), mlo);
            Register mhi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0C0D0E0F08090A0BLL}));
            builder_.Gen<x86_64::PinsrqXRegRegImm>(xmask.machine_reg(), mhi, int8_t{1});
            builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), xmask.machine_reg());
            break;
          }
          case 0b01:
            builder_.Gen<x86_64::PshuflwXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                     static_cast<int8_t>(0xB1));
            builder_.Gen<x86_64::PshufhwXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                     static_cast<int8_t>(0xB1));
            break;
          default:  // size >= 0b10 reserved
            UndefinedReturningVoid();
            return;
        }
        SetVRegFull(args.rd, xn, args.q);
        return;
      }

      // CNT V.16B / V.8B (size=00): per-byte population count via the
      // Mula-Wojcik nibble-LUT (two PSHUFB lookups on the low/high nibbles,
      // summed with PADDB). Table = popcount-per-nibble; mask = 0x0F broadcast.
      case Decoder::AdvSimdTwoRegMiscOpcode::kCnt: {
        if (args.size != 0b00) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xtable_lo = AllocTempSimdReg();
        FpRegister xtable_hi = AllocTempSimdReg();
        FpRegister xmask = AllocTempSimdReg();
        FpRegister xn = AllocTempSimdReg();
        FpRegister xt_hi = AllocTempSimdReg();
        // Build the popcount nibble table {0,1,1,2,...,4} into xtable_lo.
        Register tlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0302020102010100LL}));
        builder_.Gen<x86_64::MovqXRegReg>(xtable_lo.machine_reg(), tlo);
        Register thi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0403030203020201LL}));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(xtable_lo.machine_reg(), thi, int8_t{1});
        // PSHUFB is destructive; keep a second copy of the table.
        builder_.Gen<x86_64::MovdqaXRegXReg>(xtable_hi.machine_reg(), xtable_lo.machine_reg());
        // Build the 0x0F low-nibble mask, broadcast to all 16 bytes.
        Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0F0F0F0F0F0F0F0FLL}));
        builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), mlo);
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmask.machine_reg(), xmask.machine_reg());
        // Split Vn into low and high nibbles.
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.Gen<x86_64::MovdqaXRegXReg>(xt_hi.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsrlwXRegImm>(xt_hi.machine_reg(), int8_t{4});
        builder_.Gen<x86_64::PandXRegXReg>(xt_hi.machine_reg(), xmask.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xmask.machine_reg());
        // Vd = PSHUFB(table, low_nibbles) + PSHUFB(table, high_nibbles).
        builder_.Gen<x86_64::PshufbXRegXReg>(xtable_lo.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PshufbXRegXReg>(xtable_hi.machine_reg(), xt_hi.machine_reg());
        builder_.Gen<x86_64::PaddbXRegXReg>(xtable_lo.machine_reg(), xtable_hi.machine_reg());
        SetVRegFull(args.rd, xtable_lo, args.q);
        return;
      }

      // NOT V.16B / V.8B (size=00): per-lane bitwise complement Vd = ~Vn.
      // RBIT V.16B / V.8B (size=01): per-byte bit reversal via two PSHUFB
      // nibble-LUTs (reverse_bits(b) = (reverse4(L)<<4) | reverse4(H)), the two
      // results occupying disjoint nibble positions and combined with POR.
      case Decoder::AdvSimdTwoRegMiscOpcode::kNot: {
        if (args.size == 0b00) {
          FpRegister xn = AllocTempSimdReg();
          FpRegister allones = AllocZeroedSimdReg();
          builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(allones.machine_reg(), allones.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), allones.machine_reg());
          SetVRegFull(args.rd, xn, args.q);
          return;
        }
        if (args.size == 0b01) {
          FpRegister xlow_table = AllocTempSimdReg();
          FpRegister xhigh_table = AllocTempSimdReg();
          FpRegister xmask = AllocTempSimdReg();
          FpRegister xn = AllocTempSimdReg();
          FpRegister xn_hi = AllocTempSimdReg();
          // low_table[i]  = reverse4(i)       (result in low nibble)
          Register lt_lo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0E060A020C040800LL}));
          builder_.Gen<x86_64::MovqXRegReg>(xlow_table.machine_reg(), lt_lo);
          Register lt_hi = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0F070B030D050901LL}));
          builder_.Gen<x86_64::PinsrqXRegRegImm>(xlow_table.machine_reg(), lt_hi, int8_t{1});
          // high_table[i] = reverse4(i) << 4  (result in high nibble)
          Register ht_lo =
              std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0xE060A020C0408000ULL)));
          builder_.Gen<x86_64::MovqXRegReg>(xhigh_table.machine_reg(), ht_lo);
          Register ht_hi =
              std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0xF070B030D0509010ULL)));
          builder_.Gen<x86_64::PinsrqXRegRegImm>(xhigh_table.machine_reg(), ht_hi, int8_t{1});
          // Build the 0x0F broadcast mask.
          Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(int64_t{0x0F0F0F0F0F0F0F0FLL}));
          builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), mlo);
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmask.machine_reg(), xmask.machine_reg());
          // t_lo = low nibbles, t_hi = high nibbles.
          builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
          builder_.Gen<x86_64::MovdqaXRegXReg>(xn_hi.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsrlwXRegImm>(xn_hi.machine_reg(), int8_t{4});
          builder_.Gen<x86_64::PandXRegXReg>(xn_hi.machine_reg(), xmask.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xmask.machine_reg());
          // Vd = PSHUFB(high_table, t_lo) | PSHUFB(low_table, t_hi).
          builder_.Gen<x86_64::PshufbXRegXReg>(xhigh_table.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PshufbXRegXReg>(xlow_table.machine_reg(), xn_hi.machine_reg());
          builder_.Gen<x86_64::PorXRegXReg>(xhigh_table.machine_reg(), xlow_table.machine_reg());
          SetVRegFull(args.rd, xhigh_table, args.q);
          return;
        }
        UndefinedReturningVoid();
        return;
      }

      // NEG V.<T>, V.<T>: per-lane integer negation Vd = 0 - Vn (PSUBB/W/D/Q).
      // size=11 with Q=0 (.1D) is reserved and bails.
      case Decoder::AdvSimdTwoRegMiscOpcode::kNeg: {
        if (args.size == 0b11 && !args.q) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        FpRegister xz = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PsubbXRegXReg>(xz.machine_reg(), xn.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PsubwXRegXReg>(xz.machine_reg(), xn.machine_reg());
            break;
          case 0b10:
            builder_.Gen<x86_64::PsubdXRegXReg>(xz.machine_reg(), xn.machine_reg());
            break;
          default:  // 0b11 (.2D, Q=1)
            builder_.Gen<x86_64::PsubqXRegXReg>(xz.machine_reg(), xn.machine_reg());
            break;
        }
        SetVRegFull(args.rd, xz, args.q);
        return;
      }

      // ABS V.<T>, V.<T>: per-lane integer absolute value via
      // (Vn ^ sign_mask) - sign_mask, sign_mask = PCMPGT(0, Vn) = -1 if Vn<0.
      // size=11 (.2D) needs PCMPGTQ (not allowlisted) and bails.
      case Decoder::AdvSimdTwoRegMiscOpcode::kAbs: {
        if (args.size == 0b11) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        FpRegister mask = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PcmpgtbXRegXReg>(mask.machine_reg(), xn.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PcmpgtwXRegXReg>(mask.machine_reg(), xn.machine_reg());
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PcmpgtdXRegXReg>(mask.machine_reg(), xn.machine_reg());
            break;
        }
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), mask.machine_reg());
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PsubbXRegXReg>(xn.machine_reg(), mask.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PsubwXRegXReg>(xn.machine_reg(), mask.machine_reg());
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PsubdXRegXReg>(xn.machine_reg(), mask.machine_reg());
            break;
        }
        SetVRegFull(args.rd, xn, args.q);
        return;
      }

      // CMEQ Vd.<T>, Vn.<T>, #0 — per-lane integer compare-equal against zero.
      // PCMPEQ{B,W,D} against a zeroed register. size=11 (.2D) needs PCMPEQQ
      // (not allowlisted) and bails, matching the lite translator.
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmeqZero: {
        if (args.size == 0b11) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        FpRegister xz = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PcmpeqbXRegXReg>(xn.machine_reg(), xz.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn.machine_reg(), xz.machine_reg());
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PcmpeqdXRegXReg>(xn.machine_reg(), xz.machine_reg());
            break;
        }
        SetVRegFull(args.rd, xn, args.q);
        return;
      }

      // CMGT/CMGE/CMLE/CMLT Vd.<T>, Vn.<T>, #0 — per-lane signed integer
      // compare against zero. CMGT/CMLE compute (Vn > 0) via PCMPGT(Vn, 0);
      // CMGE/CMLT compute (0 > Vn) via PCMPGT(0, Vn). CMGE = NOT(0 > Vn) and
      // CMLE = NOT(Vn > 0), inverted with XOR against all-ones. size=11 (.2D)
      // needs PCMPGTQ (not allowlisted) and bails; FP16 lanes bail too.
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCmltZero: {
        if (args.is_fp16 || args.size == 0b11) {
          UndefinedReturningVoid();
          return;
        }
        const bool n_gt_z =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmgtZero) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero);
        const bool invert =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmgeZero) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCmleZero);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xz = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        // For (Vn > 0) the result lands in xn; for (0 > Vn) it lands in xz.
        FpRegister res = n_gt_z ? xn : xz;
        FpRegister a = n_gt_z ? xn : xz;
        FpRegister b = n_gt_z ? xz : xn;
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PcmpgtbXRegXReg>(a.machine_reg(), b.machine_reg());
            break;
          case 0b01:
            builder_.Gen<x86_64::PcmpgtwXRegXReg>(a.machine_reg(), b.machine_reg());
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PcmpgtdXRegXReg>(a.machine_reg(), b.machine_reg());
            break;
        }
        if (invert) {
          FpRegister ones = AllocZeroedSimdReg();
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(ones.machine_reg(), ones.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(res.machine_reg(), ones.machine_reg());
        }
        SetVRegFull(args.rd, res, args.q);
        return;
      }

      // XTN/XTN2 Vd.<Tb>, Vn.<Ta> — truncating narrow: keep the low half of each
      // element. size=00 (8H->8B) / size=01 (4S->4H): mask off the high half of
      // every lane, then PACKUSWB/PACKUSDW into the low 64 (the mask guarantees
      // all values sit in the unsigned pack's non-saturating range, so the pack
      // is a pure truncation). size=10 (.2D->.2S): PSHUFD gathers dwords {0,2}
      // into the low 64. Q=0 zero-extends the upper 64; Q=1 (XTN2) merges into
      // Vd's high 64. size=11 is reserved and bails. Mirrors lite kXtn.
      case Decoder::AdvSimdTwoRegMiscOpcode::kXtn: {
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        if (args.size == 0b00 || args.size == 0b01) {
          FpRegister xz = AllocZeroedSimdReg();
          FpRegister xm = AllocTempSimdReg();
          Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(
              args.size == 0b00 ? int64_t{0x00FF00FF00FF00FFLL}
                                : int64_t{0x0000FFFF0000FFFFLL}));
          builder_.Gen<x86_64::MovqXRegReg>(xm.machine_reg(), mlo);
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(xn.machine_reg(), xm.machine_reg());
          if (args.size == 0b00) {
            builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
          } else {
            builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
          }
        } else if (args.size == 0b10) {
          builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(),
                                                  int8_t{0b00001000});
        } else {  // size=11 reserved
          UndefinedReturningVoid();
          return;
        }
        SetVRegNarrow(args.rd, xn, args.q);
        return;
      }

      // SHLL/SHLL2 Vd.<Ta>, Vn.<Tb>, #<esize> — shift-left-long: zero-extend each
      // narrow lane, then shift left by the source element width (8/16/32). Q=0
      // widens Vn's low 8 bytes; Q=1 (SHLL2) the high 8 bytes (PSRLDQ brings them
      // low first). The widened result always fills all 128 bits, so it is stored
      // full regardless of Q. size=11 is unallocated and bails. Mirrors lite kShll.
      case Decoder::AdvSimdTwoRegMiscOpcode::kShll: {
        if (args.size > 0b10) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        if (args.q) {
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
        }
        switch (args.size) {
          case 0b00:
            builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{8});
            break;
          case 0b01:
            builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), int8_t{16});
            break;
          default:  // 0b10
            builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsllqXRegImm>(xn.machine_reg(), int8_t{32});
            break;
        }
        SetVRegFull(args.rd, xn, /*q=*/true);
        return;
      }

      // SQXTN/UQXTN/SQXTUN Vd.<Tb>, Vn.<Ta> — saturating extract narrow. Same
      // dst-width and Q layout as XTN; the pack flavour differs by saturation:
      //   SQXTN  signed->signed    : PACKSSWB / PACKSSDW
      //   SQXTUN signed->unsigned  : PACKUSWB / PACKUSDW
      //   UQXTN  unsigned->unsigned : PMINUW/PMINUD clamp to the unsigned dst max
      //                               (keeps values in the positive signed range
      //                               so the following PACKUS is exact), then
      //                               PACKUSWB / PACKUSDW.
      // size=00 (8H->8B) and size=01 (4S->4H) use the x86 narrowing packs.
      // size=10 (.2D->.2S) has no 64->32 x86 pack, so each 64-bit lane is
      // clamped into the destination range with PCMPGTQ/PCMPEQQ (SSE4.2/4.1)
      // masked blends and the two low dwords gathered with PSHUFD — a bit-exact
      // mirror of lite kSqxtn/kUqxtn/kSqxtun's size=10 path. size=11 is reserved.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqxtn:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUqxtn:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun: {
        const auto opc = args.opcode;
        if (args.size == 0b10) {
          FpRegister x = AllocTempSimdReg();
          builder_.GenGetSimd<16>(x.machine_reg(), vn_off);
          // Broadcast a 64-bit constant into both lanes.
          auto set_const = [&](FpRegister r, int64_t v) {
            Register gp = std::get<0>(Gen<x86_64::MovqRegImm>(v));
            builder_.Gen<x86_64::MovqXRegReg>(r.machine_reg(), gp);
            builder_.Gen<x86_64::PunpcklqdqXRegXReg>(r.machine_reg(), r.machine_reg());
          };
          // x = (x & ~mask) | (val & mask), via x ^= (x ^ val) & mask.
          auto blend = [&](FpRegister val, FpRegister mask) {
            FpRegister t = AllocTempSimdReg();
            builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
            builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), val.machine_reg());
            builder_.Gen<x86_64::PandXRegXReg>(t.machine_reg(), mask.machine_reg());
            builder_.Gen<x86_64::PxorXRegXReg>(x.machine_reg(), t.machine_reg());
          };
          if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtn) {
            FpRegister c = AllocTempSimdReg();
            FpRegister m = AllocTempSimdReg();
            set_const(c, int64_t{0x000000007FFFFFFFLL});  // INT32_MAX
            builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
            builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());  // x > MAX
            blend(c, m);
            set_const(c, static_cast<int64_t>(0xFFFFFFFF80000000ULL));  // INT32_MIN
            builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), c.machine_reg());
            builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), x.machine_reg());  // x < MIN
            blend(c, m);
          } else if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun) {
            FpRegister c = AllocTempSimdReg();
            FpRegister m = AllocTempSimdReg();
            set_const(c, int64_t{0x00000000FFFFFFFFLL});  // UINT32_MAX
            builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
            builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());  // x > UMAX
            blend(c, m);
            FpRegister z = AllocZeroedSimdReg();
            builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), z.machine_reg());
            builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), x.machine_reg());  // x < 0
            builder_.Gen<x86_64::PandnXRegXReg>(m.machine_reg(), x.machine_reg());     // neg -> 0
            builder_.Gen<x86_64::MovdqaXRegXReg>(x.machine_reg(), m.machine_reg());
          } else {  // kUqxtn: unsigned uint64 -> clamp to UINT32_MAX
            FpRegister c = AllocTempSimdReg();
            FpRegister m = AllocTempSimdReg();
            set_const(c, int64_t{0x00000000FFFFFFFFLL});  // UINT32_MAX
            builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
            builder_.Gen<x86_64::PsrlqXRegImm>(m.machine_reg(), int8_t{32});  // high 32 bits
            FpRegister z = AllocZeroedSimdReg();
            builder_.Gen<x86_64::PcmpeqqXRegXReg>(m.machine_reg(), z.machine_reg());  // high32==0
            FpRegister t = AllocTempSimdReg();
            builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
            builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), c.machine_reg());
            builder_.Gen<x86_64::PandnXRegXReg>(m.machine_reg(), t.machine_reg());
            builder_.Gen<x86_64::PxorXRegXReg>(x.machine_reg(), m.machine_reg());
          }
          builder_.Gen<x86_64::PshufdXRegXRegImm>(x.machine_reg(), x.machine_reg(),
                                                  int8_t{0b00001000});
          SetVRegNarrow(args.rd, x, args.q);
          return;
        }
        if (args.size != 0b00 && args.size != 0b01) {  // size=11 reserved
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        FpRegister xz = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kUqxtn) {
          FpRegister xm = AllocTempSimdReg();
          Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(
              args.size == 0b00 ? int64_t{0x00FF00FF00FF00FFLL}
                                : int64_t{0x0000FFFF0000FFFFLL}));
          builder_.Gen<x86_64::MovqXRegReg>(xm.machine_reg(), mlo);
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
          if (args.size == 0b00) {
            builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
          } else {
            builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xm.machine_reg());
            builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
          }
        } else if (opc == Decoder::AdvSimdTwoRegMiscOpcode::kSqxtun) {
          if (args.size == 0b00) {
            builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
          } else {
            builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
          }
        } else {  // kSqxtn
          if (args.size == 0b00) {
            builder_.Gen<x86_64::PacksswbXRegXReg>(xn.machine_reg(), xz.machine_reg());
          } else {
            builder_.Gen<x86_64::PackssdwXRegXReg>(xn.machine_reg(), xz.machine_reg());
          }
        }
        SetVRegNarrow(args.rd, xn, args.q);
        return;
      }

      // CLZ / CLS V.<T>, V.<T> — per-lane count-leading-zeros / count-leading-
      // sign-bits. size=00 .8B/.16B (N=8), size=01 .4H/.8H (N=16), size=10
      // .2S/.4S (N=32); size=11 reserved. Mirrors lite kClz/kCls but uses LZCNT
      // (branchless: LZCNT_32(0)==32) instead of the lite tier's BSR + zero
      // branch. Each lane is scalarized: PEXTR{b,w,d} zero-extends the element
      // into a GP reg, LZCNTL counts leading zeros over 32 bits, and SUBL folds
      // the width correction — CLZ_N = LZCNT_32(x) - (32-N). CLS first maps
      // negative lanes to ~x (vector PCMPGT sign mask + PXOR, exactly the lite
      // preprocess) so CLS = CLZ_N(y) - 1 = LZCNT_32(y) - (33-N); the y==0 lane
      // (all-same-bits input) then yields N-1 with no branch. Bails to lite if
      // the host lacks LZCNT.
      case Decoder::AdvSimdTwoRegMiscOpcode::kClz:
      case Decoder::AdvSimdTwoRegMiscOpcode::kCls: {
        if (args.size == 0b11 || !host_platform::kHasLZCNT) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_cls = (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kCls);
        const int lane_bits = 8 << args.size;             // 8, 16, 32
        const int bytes_per_lane = 1 << args.size;        // 1, 2, 4
        const int lanes_per_vec = (args.q ? 16 : 8) / bytes_per_lane;
        // CLZ_N = LZCNT_32 - (32-lane_bits); CLS subtracts one more.
        const int32_t correction =
            static_cast<int32_t>(32 - lane_bits) + (is_cls ? 1 : 0);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xd = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        if (is_cls) {
          // y = (x < 0) ? ~x : x per lane: xsign = (0 > xn) all-ones mask, then
          // xn ^= xsign collapses negative lanes to ~x, positive lanes unchanged.
          FpRegister xsign = AllocZeroedSimdReg();
          switch (args.size) {
            case 0b00:
              builder_.Gen<x86_64::PcmpgtbXRegXReg>(xsign.machine_reg(), xn.machine_reg());
              break;
            case 0b01:
              builder_.Gen<x86_64::PcmpgtwXRegXReg>(xsign.machine_reg(), xn.machine_reg());
              break;
            default:  // 0b10
              builder_.Gen<x86_64::PcmpgtdXRegXReg>(xsign.machine_reg(), xn.machine_reg());
              break;
          }
          builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xsign.machine_reg());
        }
        for (int i = 0; i < lanes_per_vec; ++i) {
          Register lane;
          switch (args.size) {
            case 0b00:
              lane = std::get<0>(
                  Gen<x86_64::PextrbRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(i)));
              break;
            case 0b01:
              lane = std::get<0>(
                  Gen<x86_64::PextrwRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(i)));
              break;
            default:  // 0b10
              lane = std::get<0>(
                  Gen<x86_64::PextrdRegXRegImm>(xn.machine_reg(), static_cast<int8_t>(i)));
              break;
          }
          Register cnt = std::get<0>(Gen<x86_64::LzcntlRegReg>(lane));
          if (correction != 0) {
            cnt = std::get<0>(Gen<x86_64::SublRegImm, kNoSSA>(cnt, correction));
          }
          switch (args.size) {
            case 0b00:
              builder_.Gen<x86_64::PinsrbXRegRegImm>(xd.machine_reg(), cnt,
                                                     static_cast<int8_t>(i));
              break;
            case 0b01:
              builder_.Gen<x86_64::PinsrwXRegRegImm>(xd.machine_reg(), cnt,
                                                     static_cast<int8_t>(i));
              break;
            default:  // 0b10
              builder_.Gen<x86_64::PinsrdXRegRegImm>(xd.machine_reg(), cnt,
                                                     static_cast<int8_t>(i));
              break;
          }
        }
        SetVRegFull(args.rd, xd, args.q);
        return;
      }

      // ADDV Vd, Vn.<T> — sum all source lanes; the single esize-wide result
      // is written to Vd's low lane with every other byte of Vd zeroed. Mirrors
      // the validated lite lowering: PSADBW (byte), cascading PHADDW (halfword)
      // and PHADDD (word) reductions, then a PSLLDQ/PSRLDQ shuttle keeps only
      // the low result lane. size=10 Q=0 (.2S) is reserved; size=11 bails.
      case Decoder::AdvSimdTwoRegMiscOpcode::kAddv: {
        const int32_t vd_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        switch (args.size) {
          case 0b00: {
            FpRegister xz = AllocZeroedSimdReg();
            // xn := [sum(bytes 0..7), 0..., sum(bytes 8..15), 0...] (16-bit
            // sums in qword-lane positions).
            builder_.Gen<x86_64::PsadbwXRegXReg>(xn.machine_reg(), xz.machine_reg());
            if (args.q) {
              // .16B: fold the high-qword sum into the low qword.
              FpRegister xt = AllocTempSimdReg();
              builder_.Gen<x86_64::PshufdXRegXRegImm>(xt.machine_reg(),
                                                      xn.machine_reg(),
                                                      static_cast<int8_t>(0xEE));
              builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xt.machine_reg());
            }
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{15});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{15});
            break;
          }
          case 0b01: {
            // .4H: 2 cascading PHADDW; .8H: 3.
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            if (args.q) {
              builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            }
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{14});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{14});
            break;
          }
          case 0b10: {
            if (!args.q) {  // .2S reserved for ADDV.
              UndefinedReturningVoid();
              return;
            }
            // .4S: 2 cascading PHADDD.
            builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
            break;
          }
          default:
            UndefinedReturningVoid();
            return;
        }
        builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
        return;
      }

      // SADDLV / UADDLV Vd, Vn.<T> — across-lanes long sum. Each source element
      // is widened to 2*esize before summing; the single 2*esize-wide result is
      // written to Vd's low lane with all other bytes zeroed. Mirrors the
      // validated lite lowering. UADDLV at .8B/.16B reuses the PSADBW byte-sum;
      // SADDLV and all halfword/word inputs widen first via PMOVSX/PMOVZX, then
      // reduce with PHADDW/PHADDD (or PADDQ folds for the .4S->D case).
      case Decoder::AdvSimdTwoRegMiscOpcode::kSaddlv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUaddlv: {
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSaddlv);
        const int32_t vd_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        switch (args.size) {
          case 0b00: {
            // Bytes -> 16-bit sum.
            if (!is_signed) {
              FpRegister xz = AllocZeroedSimdReg();
              builder_.Gen<x86_64::PsadbwXRegXReg>(xn.machine_reg(), xz.machine_reg());
              if (args.q) {
                FpRegister xt = AllocTempSimdReg();
                builder_.Gen<x86_64::PshufdXRegXRegImm>(xt.machine_reg(),
                                                        xn.machine_reg(),
                                                        static_cast<int8_t>(0xEE));
                builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xt.machine_reg());
              }
            } else if (args.q) {
              // SADDLV .16B: widen low/high 8 bytes separately, lane-add, then
              // 3 PHADDW collapses.
              FpRegister xt = AllocTempSimdReg();
              builder_.Gen<x86_64::PmovsxbwXRegXReg>(xt.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xt.machine_reg());
              builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            } else {
              // SADDLV .8B: widen low 8 bytes; 3 PHADDW collapses.
              builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PhaddwXRegXReg>(xn.machine_reg(), xn.machine_reg());
            }
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{14});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{14});
            break;
          }
          case 0b01: {
            // Halfwords -> 32-bit sum.
            if (args.q) {
              FpRegister xt = AllocTempSimdReg();
              if (is_signed) {
                builder_.Gen<x86_64::PmovsxwdXRegXReg>(xt.machine_reg(), xn.machine_reg());
                builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
                builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
              } else {
                builder_.Gen<x86_64::PmovzxwdXRegXReg>(xt.machine_reg(), xn.machine_reg());
                builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
                builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
              }
              builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xt.machine_reg());
              builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
            } else {
              if (is_signed) {
                builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
              } else {
                builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
              }
              builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PhadddXRegXReg>(xn.machine_reg(), xn.machine_reg());
            }
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
            break;
          }
          case 0b10: {
            if (!args.q) {  // .2S reserved.
              UndefinedReturningVoid();
              return;
            }
            // .4S -> 64-bit sum: widen 4 dwords -> 4 qwords across two regs,
            // lane-add, fold hi-qword to lo-qword.
            FpRegister xt = AllocTempSimdReg();
            if (is_signed) {
              builder_.Gen<x86_64::PmovsxdqXRegXReg>(xt.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
            } else {
              builder_.Gen<x86_64::PmovzxdqXRegXReg>(xt.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
            }
            builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xt.machine_reg());
            builder_.Gen<x86_64::PshufdXRegXRegImm>(xt.machine_reg(), xn.machine_reg(),
                                                    static_cast<int8_t>(0xEE));
            builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xt.machine_reg());
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
            break;
          }
          default:
            UndefinedReturningVoid();
            return;
        }
        builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
        return;
      }

      // SMAXV / SMINV / UMAXV / UMINV Vd, Vn.<T> — across-lanes integer
      // max/min reduce. Scan all source lanes; write the single scalar
      // max/min to Vd's low lane with all other bytes zeroed. The result
      // width equals esize (unlike SADDLV/UADDLV which widens to 2*esize).
      // Mirrors the validated lite lowering: for Q=0 the low qword is
      // replicated across both halves (PSHUFD 0x44) so the don't-care upper
      // half can't poison the reduction (max(x,x)=x is idempotent), then a
      // cascading PMAX/PMIN against a shuffle-of-xn halves the surviving lane
      // count each step; a final PSLLDQ/PSRLDQ shuttle keeps only the low lane.
      case Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSminv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUminv: {
        const bool is_max =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUmaxv);
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSmaxv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSminv);
        const int32_t vd_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xt = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        if (!args.q) {
          // Replicate low qword to high qword: { dw0, dw1, dw0, dw1 } via
          // _MM_SHUFFLE(1,0,1,0) = 0x44. Neutralizes the don't-care upper half.
          builder_.Gen<x86_64::PshufdXRegXRegImm>(
              xn.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x44));
        }
        auto EmitPmaxPmin = [&](unsigned width_bits) {
          switch (width_bits) {
            case 8:
              if (is_signed) {
                if (is_max)
                  builder_.Gen<x86_64::PmaxsbXRegXReg>(xn.machine_reg(), xt.machine_reg());
                else
                  builder_.Gen<x86_64::PminsbXRegXReg>(xn.machine_reg(), xt.machine_reg());
              } else {
                if (is_max)
                  builder_.Gen<x86_64::PmaxubXRegXReg>(xn.machine_reg(), xt.machine_reg());
                else
                  builder_.Gen<x86_64::PminubXRegXReg>(xn.machine_reg(), xt.machine_reg());
              }
              break;
            case 16:
              if (is_signed) {
                if (is_max)
                  builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xt.machine_reg());
                else
                  builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xt.machine_reg());
              } else {
                if (is_max)
                  builder_.Gen<x86_64::PmaxuwXRegXReg>(xn.machine_reg(), xt.machine_reg());
                else
                  builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xt.machine_reg());
              }
              break;
            case 32:
              if (is_signed) {
                if (is_max)
                  builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xt.machine_reg());
                else
                  builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xt.machine_reg());
              } else {
                if (is_max)
                  builder_.Gen<x86_64::PmaxudXRegXReg>(xn.machine_reg(), xt.machine_reg());
                else
                  builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xt.machine_reg());
              }
              break;
          }
        };
        switch (args.size) {
          case 0b00: {
            // Bytes: 16 -> 8 -> 4 -> 2 -> 1 surviving lanes per step.
            builder_.Gen<x86_64::PshufdXRegXRegImm>(
                xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x4E));
            EmitPmaxPmin(8);
            builder_.Gen<x86_64::PshufdXRegXRegImm>(
                xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xB1));
            EmitPmaxPmin(8);
            builder_.Gen<x86_64::MovdqaXRegXReg>(xt.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrldqXRegImm>(xt.machine_reg(), int8_t{2});
            EmitPmaxPmin(8);
            builder_.Gen<x86_64::MovdqaXRegXReg>(xt.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrldqXRegImm>(xt.machine_reg(), int8_t{1});
            EmitPmaxPmin(8);
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{15});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{15});
            break;
          }
          case 0b01: {
            // Halfwords: 8 -> 4 -> 2 -> 1 surviving lanes per step.
            builder_.Gen<x86_64::PshufdXRegXRegImm>(
                xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x4E));
            EmitPmaxPmin(16);
            builder_.Gen<x86_64::PshufdXRegXRegImm>(
                xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xB1));
            EmitPmaxPmin(16);
            builder_.Gen<x86_64::MovdqaXRegXReg>(xt.machine_reg(), xn.machine_reg());
            builder_.Gen<x86_64::PsrldqXRegImm>(xt.machine_reg(), int8_t{2});
            EmitPmaxPmin(16);
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{14});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{14});
            break;
          }
          case 0b10: {
            if (!args.q) {  // .2S reserved.
              UndefinedReturningVoid();
              return;
            }
            // Dwords: 4 -> 2 -> 1 surviving lanes per step.
            builder_.Gen<x86_64::PshufdXRegXRegImm>(
                xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x4E));
            EmitPmaxPmin(32);
            builder_.Gen<x86_64::PshufdXRegXRegImm>(
                xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xB1));
            EmitPmaxPmin(32);
            builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
            builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
            break;
          }
          default:
            UndefinedReturningVoid();
            return;
        }
        builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
        return;
      }

      // FMAXV / FMINV / FMAXNMV / FMINNMV Sd, Vn.4S — floating-point across-lanes
      // reduction. Reduce the 4 FP32 lanes to a single scalar written to Sd's low
      // lane with the upper 96 bits zeroed. Two pairwise steps (PSHUFD 0x4E then
      // 0xB1) each fold lanes together via the same NaN-handling min/max idioms the
      // vector three-same FMAX/FMIN path uses (lines ~1828):
      //   FMAXV/FMINV       — NaN-propagating: tmp=b; MAX/MIN tmp,a; MAX/MIN a,b;
      //                        POR a,tmp. (x86 MAX/MINPS returns src on NaN, so the
      //                        two-sided op + POR keeps a NaN exponent if either
      //                        input was NaN and makes +-0 order-independent.)
      //   FMAXNMV/FMINNMV   — NaN-suppressing: substitute each NaN lane with the
      //                        other operand via a CMPUNORDPS self-compare mask,
      //                        then MAX/MIN. Mirrors the interpreter FmaxScalar /
      //                        FmaxnmScalar reduction. Only the FP32 .4S form is
      //                        lowered here; the Armv8.2 FP16 (.8H) form needs an
      //                        F16C round-trip absent from the backend, so it bails.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFminv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv: {
        if (args.is_fp16) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_max =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv);
        const bool is_nm =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFmaxnmv) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFminnmv);
        const int32_t vd_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xt = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        // Fold b (=xt) into a (=xn) with FP32 packed NaN-aware min/max.
        auto EmitPairFp = [&](FpRegister a, FpRegister b) {
          if (!is_nm) {
            FpRegister tmp = AllocTempSimdReg();
            builder_.Gen<x86_64::MovdqaXRegXReg>(tmp.machine_reg(), b.machine_reg());
            if (is_max) {
              builder_.Gen<x86_64::MaxpsXRegXReg>(tmp.machine_reg(), a.machine_reg());
              builder_.Gen<x86_64::MaxpsXRegXReg>(a.machine_reg(), b.machine_reg());
            } else {
              builder_.Gen<x86_64::MinpsXRegXReg>(tmp.machine_reg(), a.machine_reg());
              builder_.Gen<x86_64::MinpsXRegXReg>(a.machine_reg(), b.machine_reg());
            }
            builder_.Gen<x86_64::PorXRegXReg>(a.machine_reg(), tmp.machine_reg());
          } else {
            FpRegister mask_a = AllocTempSimdReg();
            FpRegister mask_b = AllocTempSimdReg();
            FpRegister an_sub = AllocTempSimdReg();
            FpRegister bn_sub = AllocTempSimdReg();
            builder_.Gen<x86_64::MovdqaXRegXReg>(mask_a.machine_reg(), a.machine_reg());
            builder_.Gen<x86_64::MovdqaXRegXReg>(mask_b.machine_reg(), b.machine_reg());
            builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_a.machine_reg(), mask_a.machine_reg());
            builder_.Gen<x86_64::CmpunordpsXRegXReg>(mask_b.machine_reg(), mask_b.machine_reg());
            builder_.Gen<x86_64::MovdqaXRegXReg>(an_sub.machine_reg(), mask_a.machine_reg());
            builder_.Gen<x86_64::PandXRegXReg>(an_sub.machine_reg(), b.machine_reg());
            builder_.Gen<x86_64::MovdqaXRegXReg>(bn_sub.machine_reg(), mask_b.machine_reg());
            builder_.Gen<x86_64::PandXRegXReg>(bn_sub.machine_reg(), a.machine_reg());
            builder_.Gen<x86_64::PandnXRegXReg>(mask_a.machine_reg(), a.machine_reg());
            builder_.Gen<x86_64::PandnXRegXReg>(mask_b.machine_reg(), b.machine_reg());
            builder_.Gen<x86_64::PorXRegXReg>(mask_a.machine_reg(), an_sub.machine_reg());
            builder_.Gen<x86_64::PorXRegXReg>(mask_b.machine_reg(), bn_sub.machine_reg());
            if (is_max) {
              builder_.Gen<x86_64::MaxpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
            } else {
              builder_.Gen<x86_64::MinpsXRegXReg>(mask_a.machine_reg(), mask_b.machine_reg());
            }
            builder_.Gen<x86_64::MovdqaXRegXReg>(a.machine_reg(), mask_a.machine_reg());
          }
        };
        // Step 1: fold lanes {2,3} into {0,1}. PSHUFD 0x4E swaps the 64-bit halves.
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0x4E));
        EmitPairFp(xn, xt);
        // Step 2: fold lane 1 into lane 0. PSHUFD 0xB1 swaps adjacent dwords.
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            xt.machine_reg(), xn.machine_reg(), static_cast<int8_t>(0xB1));
        EmitPairFp(xn, xt);
        // Keep only the low 32-bit result lane; zero the upper 96 bits.
        builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
        builder_.GenSetSimd<16>(vd_off, xn.machine_reg());
        return;
      }

      // FCVTZS V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — floating-point convert to
      // signed integer, round toward zero. Branchless mirror of the validated
      // lite lowering (lite_translator.h::AdvSimdTwoRegMisc kFcvtzsV, FP32
      // path): CVTTPS2DQ does the truncating conversion, then a packed
      // saturation fix-up folds in the ARM out-of-range semantics that x86
      // CVTTPS2DQ gets wrong — NaN lanes must become 0 (x86 yields
      // 0x80000000), and positive-overflow lanes must become INT32_MAX (x86
      // yields the 0x80000000 "integer indefinite"). The FP64 .2D form (branchy
      // per-lane in lite) and the Armv8.2 FP16 form bail to lite→interp.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzsV: {
        if (args.is_fp16 || args.size != 0b10) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        FpRegister x_dst = AllocTempSimdReg();
        FpRegister x_mask = AllocTempSimdReg();
        FpRegister x_eqmin = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        // 1. Primary truncating conversion.
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
        // 2. NaN lanes -> 0.
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
        // 3. INT32_MIN (0x80000000) per lane.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
        // 4. result == INT32_MIN ?
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
        // 5. src non-negative ? PSRAD 31 -> 0 if non-neg, all-1s if neg.
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
        // 6. Flip INT_MIN -> INT_MAX in positive-overflow lanes.
        builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
        SetVRegFull(args.rd, x_dst, args.q);
        return;
      }

      // FCVTZU V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — floating-point convert to
      // unsigned integer, round toward zero. Branchless mirror of the lite
      // lowering (kFcvtzuV, FP32 path). x86 has no packed FP->u32, so the
      // classic "subtract 2^31" offset trick brings [2^31, 2^32) into signed
      // CVTTPS2DQ range, restores bit 31, and saturates too-big/Inf lanes to
      // UINT32_MAX; MAXPS(src, 0) collapses NaN/negative to 0. FP64 .2D / FP16
      // bail to lite.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtzuV: {
        if (args.is_fp16 || args.size != 0b10) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister x_dst = AllocTempSimdReg();       // src -> result
        FpRegister x_pow31 = AllocTempSimdReg();
        FpRegister x_needs_off = AllocTempSimdReg();
        // Zeroed scratch (AllocZeroedSimdReg gives it a defining write, avoiding
        // a PXOR-self read of an undefined register). Reused after step 1.
        FpRegister x_scratch = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(x_dst.machine_reg(), vn_off);
        // 1. Clamp neg/NaN to 0 via MAXPS with the zeroed scratch.
        builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        // 2. Broadcast 2^31 = 0x4F000000 to all four FP32 lanes.
        Register gp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
        builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
        // 3. needs_offset = (2^31 <= src_clamped).
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
        // 4. offset_amount = 2^31 where needs_offset.
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
        // 5. src_for_cvt = src_clamped - offset_amount.
        builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        // 6. too_big = (2^31 <= src_for_cvt).
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
        // 7. CVTTPS2DQ.
        builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
        // 8. Build 0x80000000 per lane; AND needs_offset; OR into result.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
        // 9. Saturate too-big lanes to 0xFFFFFFFF.
        builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        SetVRegFull(args.rd, x_dst, args.q);
        return;
      }

      // FCVTNS / FCVTPS / FCVTMS V Vd.<T>, Vn.<T> (FP32 .2S/.4S) —
      // floating-point convert to signed integer with an explicit rounding
      // mode (round-to-nearest-ties-even / toward +inf / toward -inf).
      // Branchless mirror of the validated lite lowering
      // (lite_translator.h::AdvSimdTwoRegMisc kFcvtnsV/kFcvtpsV/kFcvtmsV, FP32
      // path): ROUNDPS with the matching imm makes each finite lane an exact
      // integer-valued FP (NaN/±Inf/sign-of-zero pass through unchanged), then
      // the same CVTTPS2DQ + packed saturation fix-up used by FCVTZS V folds in
      // the ARM out-of-range semantics x86 gets wrong. `(args.size & 1) == 1`
      // is the FP64 .2D form (branchy per-lane in lite) and FP16 both bail to
      // lite→interp — mirroring the lite FP32-only fast path.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpsV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmsV: {
        if (args.is_fp16 || (args.size & 1) == 1) {
          UndefinedReturningVoid();
          return;
        }
        int8_t round_imm;
        if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnsV) {
          round_imm = int8_t{0x08};   // RNE + suppress-inexact
        } else if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpsV) {
          round_imm = int8_t{0x0A};   // toward +inf
        } else {
          round_imm = int8_t{0x09};   // toward -inf
        }
        FpRegister xn = AllocTempSimdReg();
        FpRegister x_dst = AllocTempSimdReg();
        FpRegister x_mask = AllocTempSimdReg();
        FpRegister x_eqmin = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.Gen<x86_64::RoundpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), round_imm);
        // FCVTZS V saturation fix-up (Roundps preserves NaN/±Inf/sign-of-zero).
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
        SetVRegFull(args.rd, x_dst, args.q);
        return;
      }

      // FCVTNU / FCVTPU / FCVTMU V Vd.<T>, Vn.<T> (FP32 .2S/.4S) —
      // floating-point convert to unsigned integer with an explicit rounding
      // mode. Branchless mirror of the lite lowering
      // (kFcvtnuV/kFcvtpuV/kFcvtmuV, FP32 path): ROUNDPS then the FCVTZU V
      // offset-by-2^31 trick (MAXPS(src,0) collapses NaN/negative to 0; the
      // [2^31,2^32) range is brought into signed CVTTPS2DQ range and bit 31 is
      // restored; too-big/Inf lanes saturate to UINT32_MAX). FP64 .2D / FP16
      // bail to lite.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtmuV: {
        if (args.is_fp16 || (args.size & 1) == 1) {
          UndefinedReturningVoid();
          return;
        }
        int8_t round_imm;
        if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtnuV) {
          round_imm = int8_t{0x08};   // RNE + suppress-inexact
        } else if (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtpuV) {
          round_imm = int8_t{0x0A};   // toward +inf
        } else {
          round_imm = int8_t{0x09};   // toward -inf
        }
        FpRegister x_dst = AllocTempSimdReg();
        FpRegister x_pow31 = AllocTempSimdReg();
        FpRegister x_needs_off = AllocTempSimdReg();
        FpRegister x_scratch = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(x_dst.machine_reg(), vn_off);
        builder_.Gen<x86_64::RoundpsXRegXRegImm>(x_dst.machine_reg(), x_dst.machine_reg(), round_imm);
        // 1. Clamp neg/NaN to 0 via MAXPS with the zeroed scratch.
        builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        // 2. Broadcast 2^31 = 0x4F000000 to all four FP32 lanes.
        Register gp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
        builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(
            x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
        // 3. needs_offset = (2^31 <= src_clamped).
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
        // 4. offset_amount = 2^31 where needs_offset.
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
        // 5. src_for_cvt = src_clamped - offset_amount.
        builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        // 6. too_big = (2^31 <= src_for_cvt).
        builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
        // 7. CVTTPS2DQ.
        builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
        // 8. Build 0x80000000 per lane; AND needs_offset; OR into result.
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
        builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
        builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
        // 9. Saturate too-big lanes to 0xFFFFFFFF.
        builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
        SetVRegFull(args.rd, x_dst, args.q);
        return;
      }

      // FCVTAS / FCVTAU V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — floating-point
      // convert to integer, round-to-nearest ties-away-from-zero. x86 ROUNDPS
      // has no ties-away mode, so mirror the lite FRINTA trick
      // (kFcvtasV/kFcvtauV, FP32 path): per lane addend = copysign(0.5, x),
      // gated to 0 when |x| >= 2^23 (already integer), ADDPS, ROUNDPS imm=3
      // (trunc). Then the FCVTZS V (signed) / FCVTZU V (unsigned) saturation
      // fix-up. FP64 .2D / FP16 bail to lite.
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtasV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV: {
        if (args.is_fp16 || (args.size & 1) == 1) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_unsigned =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kFcvtauV);
        // abs_bits and copysign are first written by a self-PCMPEQD (all-ones
        // idiom), so they need a defining write first (AllocZeroedSimdReg) to
        // satisfy lifetime analysis; half is first written by MovdXRegReg.
        FpRegister xn = AllocTempSimdReg();
        FpRegister copysign = AllocZeroedSimdReg();
        FpRegister half = AllocTempSimdReg();
        FpRegister abs_bits = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        // FRINTA dance (FP32 form).
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(abs_bits.machine_reg(), abs_bits.machine_reg());
        builder_.Gen<x86_64::PsrldXRegImm>(abs_bits.machine_reg(), int8_t{1});   // 0x7FFFFFFF
        builder_.Gen<x86_64::PandXRegXReg>(abs_bits.machine_reg(), xn.machine_reg());  // |bits|
        builder_.Gen<x86_64::PcmpeqdXRegXReg>(copysign.machine_reg(), copysign.machine_reg());
        builder_.Gen<x86_64::PslldXRegImm>(copysign.machine_reg(), int8_t{31});  // 0x80000000
        builder_.Gen<x86_64::PandXRegXReg>(copysign.machine_reg(), xn.machine_reg());  // sign bit
        Register gp_half = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x3F000000)));  // 0.5
        builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), gp_half);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(half.machine_reg(), half.machine_reg(), int8_t{0x00});
        builder_.Gen<x86_64::PorXRegXReg>(copysign.machine_reg(), half.machine_reg());  // sign|0.5
        Register gp_pow = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4B000000)));  // 2^23
        builder_.Gen<x86_64::MovdXRegReg>(half.machine_reg(), gp_pow);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(half.machine_reg(), half.machine_reg(), int8_t{0x00});
        builder_.Gen<x86_64::PcmpgtdXRegXReg>(half.machine_reg(), abs_bits.machine_reg());  // 1s where |x|<2^23
        builder_.Gen<x86_64::PandXRegXReg>(copysign.machine_reg(), half.machine_reg());  // zero addend if int
        builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), copysign.machine_reg());
        builder_.Gen<x86_64::RoundpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), int8_t{0x03});  // trunc
        if (!is_unsigned) {
          // FCVTZS V .2S/.4S saturation fix-up.
          FpRegister x_dst = AllocTempSimdReg();
          FpRegister x_mask = AllocTempSimdReg();
          FpRegister x_eqmin = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
          builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
          builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
          builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
          SetVRegFull(args.rd, x_dst, args.q);
        } else {
          // FCVTZU V .2S/.4S saturation fix-up (offset-by-2^31 trick).
          FpRegister x_dst = AllocTempSimdReg();
          FpRegister x_pow31 = AllocTempSimdReg();
          FpRegister x_needs_off = AllocTempSimdReg();
          FpRegister x_scratch = AllocZeroedSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
          builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
          Register gp2 = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
          builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp2);
          builder_.Gen<x86_64::PshufdXRegXRegImm>(
              x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
          builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
          builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
          builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
          builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
          builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
          builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
          builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
          builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
          builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
          SetVRegFull(args.rd, x_dst, args.q);
        }
        return;
      }

      // SCVTF / UCVTF V Vd.<T>, Vn.<T> (FP32 .2S/.4S) — integer to
      // floating-point convert (the inverse direction of FCVTZS/FCVTZU V).
      // Branchless mirror of the validated lite lowering
      // (lite_translator.h::AdvSimdTwoRegMisc kScvtfV/kUcvtfV, FP32 path):
      //   SCVTF: a single CVTDQ2PS — x86 signed int32 -> FP32 matches ARM.
      //   UCVTF: CVTDQ2PS treats the input as signed, so lanes with bit31 set
      //          come out negative; recover the unsigned value by adding 2^32
      //          (FP32 bits 0x4F800000) per lane wherever bit31 was set
      //          (PSRAD 31 mask), which is exact for the [2^31, 2^32) range.
      // The FP64 .2D form (branchy per-lane in lite) and FP16 bail to
      // lite→interp, mirroring the lite FP32-only fast path.
      case Decoder::AdvSimdTwoRegMiscOpcode::kScvtfV:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV: {
        if (args.is_fp16 || (args.size & 1) == 1) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_unsigned =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUcvtfV);
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        if (!is_unsigned) {
          // SCVTF V: signed int32 -> FP32 is native.
          builder_.Gen<x86_64::Cvtdq2psXRegXReg>(xn.machine_reg(), xn.machine_reg());
          SetVRegFull(args.rd, xn, args.q);
          return;
        }
        // UCVTF V: CVTDQ2PS + per-lane 2^32 addend for MSB-set lanes.
        FpRegister msb = AllocTempSimdReg();
        FpRegister addend = AllocTempSimdReg();
        builder_.Gen<x86_64::MovdqaXRegXReg>(msb.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PsradXRegImm>(msb.machine_reg(), int8_t{31});  // 0 or all-1s
        builder_.Gen<x86_64::Cvtdq2psXRegXReg>(xn.machine_reg(), xn.machine_reg());  // signed convert
        Register gp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F800000)));  // 2^32
        builder_.Gen<x86_64::MovdXRegReg>(addend.machine_reg(), gp);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(addend.machine_reg(), addend.machine_reg(), int8_t{0x00});
        builder_.Gen<x86_64::PandXRegXReg>(addend.machine_reg(), msb.machine_reg());  // 2^32 where MSB set
        builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), addend.machine_reg());
        SetVRegFull(args.rd, xn, args.q);
        return;
      }

      // SADDLP / UADDLP / SADALP / UADALP Vd.<Ta>, Vn.<Tb> — pairwise long
      // add / add-accumulate. Each adjacent pair of esize-wide source lanes is
      // widened (sign/zero) to 2*esize and summed; SADALP/UADALP accumulate the
      // pairwise sums into the existing Vd. Mirrors the validated lite lowering
      // (lite_translator.h::AdvSimdTwoRegMisc kSaddlp path) line-by-line:
      //   byte->half   signed:   PSLLW/PSRAW extract+sign-extend the low byte,
      //                          PSRAW the high byte, PADDW.
      //   byte->half   unsigned: 0x00FF mask (PCMPEQW+PSRLW 8) low byte, PSRLW 8
      //                          high byte, PADDW.
      //   half->word   signed:   PMADDWD against a per-half 0x0001 (exact signed
      //                          pair sum -> 32 bits per dword).
      //   half->word   unsigned: 0x0000FFFF mask (PCMPEQD+PSRLD 16) low half,
      //                          PSRLD 16 high half, PADDD.
      //   word->dword  signed:   PMOVSXDQ low/high dword pairs, PUNPCKL/HQDQ to
      //                          re-pair, PADDQ.
      //   word->dword  unsigned: 0xFFFFFFFF-per-qword mask (PCMPEQD+PSRLQ 32)
      //                          low dword, PSRLQ 32 high dword, PADDQ.
      // Accumulate forms PADD into Vd; Q=0 zeroes the upper 64 bits (mask_low64).
      case Decoder::AdvSimdTwoRegMiscOpcode::kSaddlp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUaddlp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kSadalp:
      case Decoder::AdvSimdTwoRegMiscOpcode::kUadalp: {
        if (args.size == 0b11) {
          UndefinedReturningVoid();
          return;
        }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSaddlp) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSadalp);
        const bool is_accum =
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kSadalp) ||
            (args.opcode == Decoder::AdvSimdTwoRegMiscOpcode::kUadalp);
        const int32_t vd_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xres = AllocTempSimdReg();
        // xtmp seeds the all-ones masks via a self-compare, so it needs a def
        // (AllocZeroedSimdReg) before the PCMPEQ; the Movdqa branches overwrite
        // it harmlessly.
        FpRegister xtmp = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        switch (args.size) {
          case 0b00: {  // byte -> half
            if (is_signed) {
              builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsllwXRegImm>(xres.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PsrawXRegImm>(xres.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrawXRegImm>(xtmp.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PaddwXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            } else {
              builder_.Gen<x86_64::PcmpeqwXRegXReg>(xtmp.machine_reg(), xtmp.machine_reg());
              builder_.Gen<x86_64::PsrlwXRegImm>(xtmp.machine_reg(), int8_t{8});  // 0x00FF
              builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
              builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrlwXRegImm>(xtmp.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PaddwXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            }
            break;
          }
          case 0b01: {  // half -> word
            if (is_signed) {
              builder_.Gen<x86_64::PcmpeqwXRegXReg>(xtmp.machine_reg(), xtmp.machine_reg());
              builder_.Gen<x86_64::PsrlwXRegImm>(xtmp.machine_reg(), int8_t{15});  // 0x0001
              builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PmaddwdXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            } else {
              builder_.Gen<x86_64::PcmpeqdXRegXReg>(xtmp.machine_reg(), xtmp.machine_reg());
              builder_.Gen<x86_64::PsrldXRegImm>(xtmp.machine_reg(), int8_t{16});  // 0x0000FFFF
              builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
              builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrldXRegImm>(xtmp.machine_reg(), int8_t{16});
              builder_.Gen<x86_64::PadddXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            }
            break;
          }
          case 0b10: {  // word -> dword
            if (is_signed) {
              FpRegister xhi = AllocTempSimdReg();
              builder_.Gen<x86_64::PmovsxdqXRegXReg>(xres.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::MovdqaXRegXReg>(xhi.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrldqXRegImm>(xhi.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PmovsxdqXRegXReg>(xhi.machine_reg(), xhi.machine_reg());
              builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xres.machine_reg());
              builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xtmp.machine_reg(), xhi.machine_reg());
              builder_.Gen<x86_64::PunpckhqdqXRegXReg>(xres.machine_reg(), xhi.machine_reg());
              builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            } else {
              builder_.Gen<x86_64::PcmpeqdXRegXReg>(xtmp.machine_reg(), xtmp.machine_reg());
              builder_.Gen<x86_64::PsrlqXRegImm>(xtmp.machine_reg(), int8_t{32});  // lo dword mask
              builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PandXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
              builder_.Gen<x86_64::MovdqaXRegXReg>(xtmp.machine_reg(), xn.machine_reg());
              builder_.Gen<x86_64::PsrlqXRegImm>(xtmp.machine_reg(), int8_t{32});
              builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
            }
            break;
          }
          default:
            UndefinedReturningVoid();
            return;
        }
        if (is_accum) {
          FpRegister xd = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
          switch (args.size) {
            case 0b00:
              builder_.Gen<x86_64::PaddwXRegXReg>(xres.machine_reg(), xd.machine_reg());
              break;
            case 0b01:
              builder_.Gen<x86_64::PadddXRegXReg>(xres.machine_reg(), xd.machine_reg());
              break;
            default:
              builder_.Gen<x86_64::PaddqXRegXReg>(xres.machine_reg(), xd.machine_reg());
              break;
          }
        }
        if (!args.q) {
          // mask_low64: zero the upper 64 bits (D-register semantics).
          builder_.Gen<x86_64::PslldqXRegImm>(xres.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PsrldqXRegImm>(xres.machine_reg(), int8_t{8});
        }
        builder_.GenSetSimd<16>(vd_off, xres.machine_reg());
        return;
      }

      default:
        UndefinedReturningVoid();
        return;
    }
  }

  // AdvSIMD scalar two-register misc.  The single-lane FP<->integer converts
  // FCVTZS / FCVTZU (float->int, round toward zero) and SCVTF / UCVTF
  // (int->float), all S form / FP32, are lowered here; every other scalar
  // two-reg-misc opcode bails to lite via UndefinedReturningVoid.
  //
  // Lowering is a single-lane application of the corresponding vector .2S/.4S
  // recipes in AdvSimdTwoRegMisc (kFcvtzsV / kFcvtzuV / kScvtfV / kUcvtfV).  The
  // source is loaded full-width and scrubbed to lane 0 (PSLLDQ+PSRLDQ keep the
  // low 32 bits and zero everything above), so lanes 1..3 become 0 and convert
  // to 0.  This is required for correctness: SetVRegFull q=false writes the low
  // 64 bits via MOVSD, so lane 1 (Vd[63:32]) survives — scrubbing forces it to
  // 0 as the scalar S result requires (Vd[31:0]=result, Vd[127:32]=0).  For the
  // int->float SCVTF/UCVTF forms the scrubbed lanes are integer 0, converting to
  // +0.0f = 0x00000000, which is exactly the required zero fill.
  //   * FCVTZS/FCVTZU/SCVTF/UCVTF with sz(bit0 of size)=0 (S/FP32): vector recipe.
  //   * sz=1 (D / FP64) is branchy per-lane in lite -> bail.
  //   * every other scalar two-reg-misc opcode -> bail.
  void AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs& args) {
    if (!success()) {
      return;
    }
    // Scalar saturating extract-narrow SQXTN/UQXTN/SQXTUN Vd.<Tb>, Vn.<Ta> —
    // single-lane collapse of the vector kSqxtn/kUqxtn/kSqxtun recipe (see
    // AdvSimdTwoRegMisc). size=00 (H->B), 01 (S->H), 10 (D->S); size=11 is
    // decoder-rejected. Scrub the source to lane 0 (keep the low 2<<size source
    // bytes, zero the rest) so the packed vector recipe processes only the
    // scalar element while lanes 1.. stay 0 (saturate(0)=0), then commit with
    // SetVRegNarrow q=false (MOVSD low-64, upper 64 zeroed) for the scalar
    // Vd[127:esize]=0 result. Mirrors interpreter kSqxtn/kUqxtn/kSqxtun.
    if (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtn ||
        args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kUqxtn ||
        args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtun) {
      const auto opc = args.opcode;
      const int32_t vn_off_narrow =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
      FpRegister x = AllocTempSimdReg();
      builder_.GenGetSimd<16>(x.machine_reg(), vn_off_narrow);
      // Scrub to lane 0: keep the low (2<<size) source bytes, zero everything
      // above (PSLLDQ N + PSRLDQ N, N = 16 - src_bytes).
      const int8_t scrub = static_cast<int8_t>(16 - (2 << args.size));  // 14,12,8
      builder_.Gen<x86_64::PslldqXRegImm>(x.machine_reg(), scrub);
      builder_.Gen<x86_64::PsrldqXRegImm>(x.machine_reg(), scrub);
      if (args.size == 0b10) {
        // 64->32: no 64->32 x86 pack, so clamp each 64-bit lane into the
        // destination range with PCMPGTQ/PCMPEQQ masked blends and gather the
        // two low dwords with PSHUFD — bit-exact mirror of the vector size=10
        // path. The scrubbed lane 1 is 0 -> clamp(0)=0 -> gathered dword=0.
        auto set_const = [&](FpRegister r, int64_t v) {
          Register gp = std::get<0>(Gen<x86_64::MovqRegImm>(v));
          builder_.Gen<x86_64::MovqXRegReg>(r.machine_reg(), gp);
          builder_.Gen<x86_64::PunpcklqdqXRegXReg>(r.machine_reg(), r.machine_reg());
        };
        // x = (x & ~mask) | (val & mask), via x ^= (x ^ val) & mask.
        auto blend = [&](FpRegister val, FpRegister mask) {
          FpRegister t = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), val.machine_reg());
          builder_.Gen<x86_64::PandXRegXReg>(t.machine_reg(), mask.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(x.machine_reg(), t.machine_reg());
        };
        if (opc == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtn) {
          FpRegister c = AllocTempSimdReg();
          FpRegister m = AllocTempSimdReg();
          set_const(c, int64_t{0x000000007FFFFFFFLL});  // INT32_MAX
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());  // x > MAX
          blend(c, m);
          set_const(c, static_cast<int64_t>(0xFFFFFFFF80000000ULL));  // INT32_MIN
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), c.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), x.machine_reg());  // x < MIN
          blend(c, m);
        } else if (opc == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtun) {
          FpRegister c = AllocTempSimdReg();
          FpRegister m = AllocTempSimdReg();
          set_const(c, int64_t{0x00000000FFFFFFFFLL});  // UINT32_MAX
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), c.machine_reg());  // x > UMAX
          blend(c, m);
          FpRegister z = AllocZeroedSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), z.machine_reg());
          builder_.Gen<x86_64::PcmpgtqXRegXReg>(m.machine_reg(), x.machine_reg());  // x < 0
          builder_.Gen<x86_64::PandnXRegXReg>(m.machine_reg(), x.machine_reg());     // neg -> 0
          builder_.Gen<x86_64::MovdqaXRegXReg>(x.machine_reg(), m.machine_reg());
        } else {  // kUqxtn: unsigned uint64 -> clamp to UINT32_MAX
          FpRegister c = AllocTempSimdReg();
          FpRegister m = AllocTempSimdReg();
          set_const(c, int64_t{0x00000000FFFFFFFFLL});  // UINT32_MAX
          builder_.Gen<x86_64::MovdqaXRegXReg>(m.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PsrlqXRegImm>(m.machine_reg(), int8_t{32});  // high 32 bits
          FpRegister z = AllocZeroedSimdReg();
          builder_.Gen<x86_64::PcmpeqqXRegXReg>(m.machine_reg(), z.machine_reg());  // high32==0
          FpRegister t = AllocTempSimdReg();
          builder_.Gen<x86_64::MovdqaXRegXReg>(t.machine_reg(), x.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(t.machine_reg(), c.machine_reg());
          builder_.Gen<x86_64::PandnXRegXReg>(m.machine_reg(), t.machine_reg());
          builder_.Gen<x86_64::PxorXRegXReg>(x.machine_reg(), m.machine_reg());
        }
        builder_.Gen<x86_64::PshufdXRegXRegImm>(x.machine_reg(), x.machine_reg(),
                                                int8_t{0b00001000});
        SetVRegNarrow(args.rd, x, /*q=*/false);
        return;
      }
      // 8<-16 / 16<-32: x86 narrowing packs against a zeroed high source.
      FpRegister xz = AllocZeroedSimdReg();
      if (opc == Decoder::AdvSimdScalarTwoRegMiscOpcode::kUqxtn) {
        FpRegister xm = AllocTempSimdReg();
        Register mlo = std::get<0>(Gen<x86_64::MovqRegImm>(
            args.size == 0b00 ? int64_t{0x00FF00FF00FF00FFLL}
                              : int64_t{0x0000FFFF0000FFFFLL}));
        builder_.Gen<x86_64::MovqXRegReg>(xm.machine_reg(), mlo);
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xm.machine_reg(), xm.machine_reg());
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PminuwXRegXReg>(x.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PackuswbXRegXReg>(x.machine_reg(), xz.machine_reg());
        } else {
          builder_.Gen<x86_64::PminudXRegXReg>(x.machine_reg(), xm.machine_reg());
          builder_.Gen<x86_64::PackusdwXRegXReg>(x.machine_reg(), xz.machine_reg());
        }
      } else if (opc == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqxtun) {
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PackuswbXRegXReg>(x.machine_reg(), xz.machine_reg());
        } else {
          builder_.Gen<x86_64::PackusdwXRegXReg>(x.machine_reg(), xz.machine_reg());
        }
      } else {  // kSqxtn
        if (args.size == 0b00) {
          builder_.Gen<x86_64::PacksswbXRegXReg>(x.machine_reg(), xz.machine_reg());
        } else {
          builder_.Gen<x86_64::PackssdwXRegXReg>(x.machine_reg(), xz.machine_reg());
        }
      }
      SetVRegNarrow(args.rd, x, /*q=*/false);
      return;
    }
    // Scalar SQABS/SQNEG Vd, Vn — single-lane saturating signed absolute
    // value / negate. Mirrors the vector kSqabs/kSqneg SSE recipe
    // (lite_translator.h) on a lane-0-scrubbed source: scrub Vn so lanes 1..
    // become the integer 0 (SQABS(0)=SQNEG(0)=0, never the saturating input),
    // apply the packed abs/neg + INT_MIN->INT_MAX saturation, then commit with
    // SetVRegFull q=false (MOVSD low-64, upper 64 zeroed). size=00/01/10
    // (B/H/S); size=11 (D) needs PCMPGTQ/PCMPEQQ 64-bit lanes and bails to lite
    // — the same boundary as the vector .2D form. Matches interpreter
    // AdvSimdScalarTwoRegMisc kSqabs/kSqneg.
    if (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqabs ||
        args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqneg) {
      if (args.size == 0b11) {
        UndefinedReturningVoid();  // D-width: bail to lite (64-bit-lane saturation).
        return;
      }
      const bool is_neg =
          (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kSqneg);
      const int32_t vn_off_sq =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off_sq);
      // Scrub to lane 0: keep the low (1<<size) bytes, zero everything above,
      // so lanes 1.. hold the integer 0.
      const int8_t scrub = static_cast<int8_t>(16 - (1 << args.size));  // 15,14,12
      builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), scrub);
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), scrub);
      FpRegister xres = AllocTempSimdReg();
      // xtmp is defined below by MovD (INT_MIN broadcast) — no self-op on a
      // fresh temp; the branches use AllocZeroedSimdReg for their zero source
      // to avoid the lifetime-analysis use-before-def on a self-PXOR.
      FpRegister xtmp = AllocTempSimdReg();
      if (is_neg) {
        // SQNEG: res = 0 - xn (packed).  xres starts at zero (copied from a
        // zeroed reg so the def is a MOVDQA, not a use-before-def self-PXOR).
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(),
                                             AllocZeroedSimdReg().machine_reg());
        switch (args.size) {
          case 0b00: builder_.Gen<x86_64::PsubbXRegXReg>(xres.machine_reg(), xn.machine_reg()); break;
          case 0b01: builder_.Gen<x86_64::PsubwXRegXReg>(xres.machine_reg(), xn.machine_reg()); break;
          default:   builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), xn.machine_reg()); break;
        }
      } else {
        // SQABS: sign = PCMPGT(0, xn); res = (xn ^ sign) - sign = |xn|.
        FpRegister xsign = AllocZeroedSimdReg();
        switch (args.size) {
          case 0b00: builder_.Gen<x86_64::PcmpgtbXRegXReg>(xsign.machine_reg(), xn.machine_reg()); break;
          case 0b01: builder_.Gen<x86_64::PcmpgtwXRegXReg>(xsign.machine_reg(), xn.machine_reg()); break;
          default:   builder_.Gen<x86_64::PcmpgtdXRegXReg>(xsign.machine_reg(), xn.machine_reg()); break;
        }
        builder_.Gen<x86_64::MovdqaXRegXReg>(xres.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xres.machine_reg(), xsign.machine_reg());
        switch (args.size) {
          case 0b00: builder_.Gen<x86_64::PsubbXRegXReg>(xres.machine_reg(), xsign.machine_reg()); break;
          case 0b01: builder_.Gen<x86_64::PsubwXRegXReg>(xres.machine_reg(), xsign.machine_reg()); break;
          default:   builder_.Gen<x86_64::PsubdXRegXReg>(xres.machine_reg(), xsign.machine_reg()); break;
        }
      }
      // Saturation: INT_MIN lanes still hold INT_MIN (0x80..0) after abs/neg;
      // XOR with an all-1s mask on those lanes turns INT_MIN -> INT_MAX. A
      // 32-bit broadcast pattern covers all three widths (the INT_MIN bit
      // repeats every 32 bits at byte/half/word granularity). xtmp is dead
      // here (sign mask consumed for SQABS; never set for SQNEG).
      int32_t int_min_bcast;
      switch (args.size) {
        case 0b00: int_min_bcast = static_cast<int32_t>(0x80808080u); break;
        case 0b01: int_min_bcast = static_cast<int32_t>(0x80008000u); break;
        default:   int_min_bcast = static_cast<int32_t>(0x80000000u); break;
      }
      Register gp_min = std::get<0>(Gen<x86_64::MovlRegImm>(int_min_bcast));
      builder_.Gen<x86_64::MovdXRegReg>(xtmp.machine_reg(), gp_min);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xtmp.machine_reg(), xtmp.machine_reg(), int8_t{0x00});
      switch (args.size) {
        case 0b00: builder_.Gen<x86_64::PcmpeqbXRegXReg>(xtmp.machine_reg(), xn.machine_reg()); break;
        case 0b01: builder_.Gen<x86_64::PcmpeqwXRegXReg>(xtmp.machine_reg(), xn.machine_reg()); break;
        default:   builder_.Gen<x86_64::PcmpeqdXRegXReg>(xtmp.machine_reg(), xn.machine_reg()); break;
      }
      builder_.Gen<x86_64::PxorXRegXReg>(xres.machine_reg(), xtmp.machine_reg());
      SetVRegFull(args.rd, xres, /*q=*/false);
      return;
    }
    const bool is_fcvtzs =
        (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtzs);
    const bool is_fcvtzu =
        (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kFcvtzu);
    const bool is_scvtf =
        (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kScvtf);
    const bool is_ucvtf =
        (args.opcode == Decoder::AdvSimdScalarTwoRegMiscOpcode::kUcvtf);
    if ((!is_fcvtzs && !is_fcvtzu && !is_scvtf && !is_ucvtf) ||
        (args.size & 1) != 0) {
      UndefinedReturningVoid();
      return;
    }
    const int32_t vn_off =
        static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
    // SCVTF / UCVTF scalar (S): int32 -> FP32.  Mirror the kScvtfV/kUcvtfV FP32
    // recipe on a lane-0-scrubbed source.
    if (is_scvtf || is_ucvtf) {
      FpRegister xn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // Scrub to lane 0: keep the low 32 bits (the scalar int), zero bytes [15:4].
      builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
      if (is_scvtf) {
        // SCVTF: signed int32 -> FP32 is native.
        builder_.Gen<x86_64::Cvtdq2psXRegXReg>(xn.machine_reg(), xn.machine_reg());
        SetVRegFull(args.rd, xn, /*q=*/false);
        return;
      }
      // UCVTF: CVTDQ2PS + per-lane 2^32 addend for MSB-set lanes.
      FpRegister msb = AllocTempSimdReg();
      FpRegister addend = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(msb.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(msb.machine_reg(), int8_t{31});  // 0 or all-1s
      builder_.Gen<x86_64::Cvtdq2psXRegXReg>(xn.machine_reg(), xn.machine_reg());  // signed convert
      Register gp =
          std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F800000)));  // 2^32
      builder_.Gen<x86_64::MovdXRegReg>(addend.machine_reg(), gp);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(
          addend.machine_reg(), addend.machine_reg(), int8_t{0x00});
      builder_.Gen<x86_64::PandXRegXReg>(addend.machine_reg(), msb.machine_reg());
      builder_.Gen<x86_64::AddpsXRegXReg>(xn.machine_reg(), addend.machine_reg());
      SetVRegFull(args.rd, xn, /*q=*/false);
      return;
    }
    const bool is_unsigned = is_fcvtzu;
    if (!is_unsigned) {
      // FCVTZS scalar (S): mirror the kFcvtzsV FP32 recipe on a lane-0-scrubbed
      // source (CVTTPS2DQ + NaN->0 + positive-overflow->INT32_MAX fix-up).
      FpRegister xn = AllocTempSimdReg();
      FpRegister x_dst = AllocTempSimdReg();
      FpRegister x_mask = AllocTempSimdReg();
      FpRegister x_eqmin = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      // Scrub to lane 0: keep the low 32 bits, zero bytes [15:4].
      builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
      // 1. Primary truncating conversion.
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
      // 2. NaN lanes -> 0.
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::CmpunordpsXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
      // 3. INT32_MIN (0x80000000) per lane.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_mask.machine_reg(), x_mask.machine_reg());
      builder_.Gen<x86_64::PslldXRegImm>(x_mask.machine_reg(), int8_t{31});
      // 4. result == INT32_MIN ?
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_eqmin.machine_reg(), x_dst.machine_reg());
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_eqmin.machine_reg(), x_mask.machine_reg());
      // 5. src non-negative ? PSRAD 31 -> 0 if non-neg, all-1s if neg.
      builder_.Gen<x86_64::MovdqaXRegXReg>(x_mask.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PsradXRegImm>(x_mask.machine_reg(), int8_t{31});
      builder_.Gen<x86_64::PandnXRegXReg>(x_mask.machine_reg(), x_eqmin.machine_reg());
      // 6. Flip INT_MIN -> INT_MAX in positive-overflow lanes.
      builder_.Gen<x86_64::PxorXRegXReg>(x_dst.machine_reg(), x_mask.machine_reg());
      SetVRegFull(args.rd, x_dst, /*q=*/false);
      return;
    }
    // FCVTZU scalar (S): mirror the kFcvtzuV FP32 recipe on a lane-0-scrubbed
    // source (subtract-2^31 offset trick + saturate too-big/Inf to UINT32_MAX).
    FpRegister x_dst = AllocTempSimdReg();
    FpRegister x_pow31 = AllocTempSimdReg();
    FpRegister x_needs_off = AllocTempSimdReg();
    FpRegister x_scratch = AllocZeroedSimdReg();
    builder_.GenGetSimd<16>(x_dst.machine_reg(), vn_off);
    // Scrub to lane 0: keep the low 32 bits, zero bytes [15:4].
    builder_.Gen<x86_64::PslldqXRegImm>(x_dst.machine_reg(), int8_t{12});
    builder_.Gen<x86_64::PsrldqXRegImm>(x_dst.machine_reg(), int8_t{12});
    // 1. Clamp neg/NaN to 0 via MAXPS with the zeroed scratch.
    builder_.Gen<x86_64::MaxpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
    // 2. Broadcast 2^31 = 0x4F000000 to all four FP32 lanes.
    Register gp = std::get<0>(Gen<x86_64::MovlRegImm>(static_cast<int32_t>(0x4F000000)));
    builder_.Gen<x86_64::MovdXRegReg>(x_pow31.machine_reg(), gp);
    builder_.Gen<x86_64::PshufdXRegXRegImm>(
        x_pow31.machine_reg(), x_pow31.machine_reg(), int8_t{0x00});
    // 3. needs_offset = (2^31 <= src_clamped).
    builder_.Gen<x86_64::MovdqaXRegXReg>(x_needs_off.machine_reg(), x_pow31.machine_reg());
    builder_.Gen<x86_64::CmplepsXRegXReg>(x_needs_off.machine_reg(), x_dst.machine_reg());
    // 4. offset_amount = 2^31 where needs_offset.
    builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(x_scratch.machine_reg(), x_needs_off.machine_reg());
    // 5. src_for_cvt = src_clamped - offset_amount.
    builder_.Gen<x86_64::SubpsXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
    // 6. too_big = (2^31 <= src_for_cvt).
    builder_.Gen<x86_64::MovdqaXRegXReg>(x_scratch.machine_reg(), x_pow31.machine_reg());
    builder_.Gen<x86_64::CmplepsXRegXReg>(x_scratch.machine_reg(), x_dst.machine_reg());
    // 7. CVTTPS2DQ.
    builder_.Gen<x86_64::Cvttps2dqXRegXReg>(x_dst.machine_reg(), x_dst.machine_reg());
    // 8. Build 0x80000000 per lane; AND needs_offset; OR into result.
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_pow31.machine_reg(), x_pow31.machine_reg());
    builder_.Gen<x86_64::PslldXRegImm>(x_pow31.machine_reg(), int8_t{31});
    builder_.Gen<x86_64::PandXRegXReg>(x_pow31.machine_reg(), x_needs_off.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_pow31.machine_reg());
    // 9. Saturate too-big lanes to 0xFFFFFFFF.
    builder_.Gen<x86_64::PorXRegXReg>(x_dst.machine_reg(), x_scratch.machine_reg());
    SetVRegFull(args.rd, x_dst, /*q=*/false);
  }

  // AdvSIMD scalar three-same.  Only the saturating-doubling-multiply-high
  // scalar forms (SQDMULH / SQRDMULH, H and S lanes) are lowered here; every
  // other scalar three-same opcode bails to lite via UndefinedReturningVoid.
  //
  // Lowering is a single-lane application of the vector .4H/.8H and .2S/.4S
  // SQDMULH/SQRDMULH recipes in AdvSimdThreeSame (opcode 0b10110).  The
  // operands are loaded full-width and scrubbed to lane 0 (PSLLDQ+PSRLDQ keep
  // the low 16 / 32 bits and zero everything above), so the packed recipe
  // processes only the scalar lane while lanes 1.. stay 0 (a corner-free 0
  // result).  SetVRegFull q=false then writes the scalar result with
  // Vd[127:lane] = 0.
  //   * size=01 (H): SQDMULH via PMULHW+PMULLW combine; SQRDMULH via PMULHRSW
  //     (SSSE3); both with the INT16_MIN*INT16_MIN corner fixup.
  //   * size=10 (S): two PMULDQ + PSLLQ double + optional 2^31 round + PSHUFD
  //     0xDD lift + INT32_MIN*INT32_MIN corner fixup (SSE4.1).
  //   * size=00/11 are unallocated for this opcode -> bail.
  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs& args) {
    if (!success()) {
      return;
    }
    if (args.opcode != Decoder::AdvSimdScalarThreeSameOpcode::kSqdmulhScalar &&
        args.opcode != Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmulhScalar) {
      UndefinedReturningVoid();
      return;
    }
    if (args.size != 0b01 && args.size != 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_round =
        (args.opcode == Decoder::AdvSimdScalarThreeSameOpcode::kSqrdmulhScalar);
    const int32_t vn_off =
        static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
    const int32_t vm_off =
        static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);
    if (args.size == 0b01) {
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      // Scrub to lane 0: keep the low 16 bits, zero bytes [15:2].
      builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{14});
      builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{14});
      builder_.Gen<x86_64::PslldqXRegImm>(xm.machine_reg(), int8_t{14});
      builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{14});
      if (is_round) {
        if (!host_platform::kHasSSSE3) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn_corner = AllocTempSimdReg();
        FpRegister xm_corner = AllocTempSimdReg();
        FpRegister x_min = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqw idiom
        builder_.Gen<x86_64::MovdqaXRegXReg>(xn_corner.machine_reg(), xn.machine_reg());
        builder_.Gen<x86_64::MovdqaXRegXReg>(xm_corner.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(x_min.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});  // INT16_MIN
        builder_.Gen<x86_64::PmulhrswXRegXReg>(xn.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_corner.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_corner.machine_reg(), x_min.machine_reg());
        builder_.Gen<x86_64::PandXRegXReg>(xn_corner.machine_reg(), xm_corner.machine_reg());
        builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_corner.machine_reg());
        SetVRegFull(args.rd, xn, /*q=*/false);
        return;
      }
      // SQDMULH .H via PMULHW + PMULLW combine + corner fixup (SSE2).
      FpRegister xn_lo = AllocTempSimdReg();
      FpRegister xn_corner = AllocTempSimdReg();
      FpRegister xm_corner = AllocTempSimdReg();
      FpRegister x_min = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqw idiom
      builder_.Gen<x86_64::MovdqaXRegXReg>(xn_corner.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xm_corner.machine_reg(), xm.machine_reg());
      builder_.Gen<x86_64::MovdqaXRegXReg>(xn_lo.machine_reg(), xn.machine_reg());
      builder_.Gen<x86_64::PmullwXRegXReg>(xn_lo.machine_reg(), xm.machine_reg());  // low16(a*b)
      builder_.Gen<x86_64::PmulhwXRegXReg>(xn.machine_reg(), xm.machine_reg());     // high16(a*b)
      builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{1});              // high<<1
      builder_.Gen<x86_64::PsrlwXRegImm>(xn_lo.machine_reg(), int8_t{15});          // low top bit
      builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xn_lo.machine_reg());     // high16(2*a*b)
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(x_min.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PsllwXRegImm>(x_min.machine_reg(), int8_t{15});          // INT16_MIN
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(xn_corner.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm_corner.machine_reg(), x_min.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xn_corner.machine_reg(), xm_corner.machine_reg());
      builder_.Gen<x86_64::PxorXRegXReg>(xn.machine_reg(), xn_corner.machine_reg());
      SetVRegFull(args.rd, xn, /*q=*/false);
      return;
    }
    // size == 0b10 (S form): PMULDQ widen (SSE4.1) + PSLLQ double + optional
    // round + corner fixup; PSHUFD 0xDD lifts each product's upper 32 bits.
    if (!host_platform::kHasSSE4_1) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    FpRegister x_const = AllocZeroedSimdReg();  // pre-defined: self-Pcmpeqd idiom
    FpRegister corner = AllocTempSimdReg();
    FpRegister xp_lo = AllocTempSimdReg();
    FpRegister xp_hi = AllocTempSimdReg();
    FpRegister xm_hi = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    // Scrub to lane 0: keep the low 32 bits, zero bytes [15:4].
    builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{12});
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{12});
    builder_.Gen<x86_64::PslldqXRegImm>(xm.machine_reg(), int8_t{12});
    builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{12});
    // x_const = INT32_MIN broadcast across 4 dwords.
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
    builder_.Gen<x86_64::PslldXRegImm>(x_const.machine_reg(), int8_t{31});
    // Corner: lane 0 where Vn.s[0] == INT32_MIN AND Vm.s[0] == INT32_MIN.
    builder_.Gen<x86_64::MovdqaXRegXReg>(corner.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(corner.machine_reg(), x_const.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(xp_lo.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(xp_lo.machine_reg(), x_const.machine_reg());
    builder_.Gen<x86_64::PandXRegXReg>(corner.machine_reg(), xp_lo.machine_reg());
    // Two PMULDQs reconstruct the signed 32x32 -> 64 products (even lanes in
    // xp_lo, odd lanes in xp_hi).
    builder_.Gen<x86_64::MovdqaXRegXReg>(xp_lo.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PmuldqXRegXReg>(xp_lo.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::MovdqaXRegXReg>(xp_hi.machine_reg(), xn.machine_reg());
    builder_.Gen<x86_64::PsrlqXRegImm>(xp_hi.machine_reg(), int8_t{32});
    builder_.Gen<x86_64::MovdqaXRegXReg>(xm_hi.machine_reg(), xm.machine_reg());
    builder_.Gen<x86_64::PsrlqXRegImm>(xm_hi.machine_reg(), int8_t{32});
    builder_.Gen<x86_64::PmuldqXRegXReg>(xp_hi.machine_reg(), xm_hi.machine_reg());
    // Double each 64-bit signed product.
    builder_.Gen<x86_64::PsllqXRegImm>(xp_lo.machine_reg(), int8_t{1});
    builder_.Gen<x86_64::PsllqXRegImm>(xp_hi.machine_reg(), int8_t{1});
    if (is_round) {
      // SQRDMULH: add rounding constant 2^31 = 0x80000000 per qword.
      builder_.Gen<x86_64::PcmpeqdXRegXReg>(x_const.machine_reg(), x_const.machine_reg());
      builder_.Gen<x86_64::PsllqXRegImm>(x_const.machine_reg(), int8_t{63});
      builder_.Gen<x86_64::PsrlqXRegImm>(x_const.machine_reg(), int8_t{32});
      builder_.Gen<x86_64::PaddqXRegXReg>(xp_lo.machine_reg(), x_const.machine_reg());
      builder_.Gen<x86_64::PaddqXRegXReg>(xp_hi.machine_reg(), x_const.machine_reg());
    }
    // Extract the upper 32 bits of each 64-bit lane and interleave.
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xp_lo.machine_reg(), xp_lo.machine_reg(), static_cast<int8_t>(0xDD));
    builder_.Gen<x86_64::PshufdXRegXRegImm>(xp_hi.machine_reg(), xp_hi.machine_reg(), static_cast<int8_t>(0xDD));
    builder_.Gen<x86_64::PunpckldqXRegXReg>(xp_lo.machine_reg(), xp_hi.machine_reg());
    // Apply corner mask: INT32_MIN ^ 0xFFFFFFFF = INT32_MAX.
    builder_.Gen<x86_64::PxorXRegXReg>(xp_lo.machine_reg(), corner.machine_reg());
    SetVRegFull(args.rd, xp_lo, /*q=*/false);
  }

  void AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  // AdvSIMD shift-by-immediate (vector and scalar).  Mirrors the high-value
  // subset of lite_translator.h::AdvSimdShiftByImm into the optimizing tier so
  // NEON widening/shift loops (calculate_gnu_hash_neon and every SIMD widening
  // expansion) stop bailing the heavy region wholesale to lite.  Covered here:
  //   * USHLL / SSHLL (incl. UXTL/SXTL == #0) — 8B->8H / 4H->4S / 2S->2D
  //     widening, both Q halves (long2 reads Vn[127:64]).
  //   * SHL / USHR / SSHR at H/S/D lanes (esize 16/32/64).
  //   * SSRA / USRA (shift-accumulate) at H/S/D lanes (USRA all three; SSRA
  //     esize 16/32 only — SSRA .2D needs PSRAQ and bails).
  //   * SLI / SRI (shift-and-insert) at H/S/D lanes (all three; PSLL/PSRL mask
  //     round trip + shifted Vn + POR).
  //   * SRSHR / URSHR / SRSRA / URSRA (rounding right shift + accumulate) at
  //     H/S/D lanes (URSHR/URSRA all three; SRSHR/SRSRA esize 16/32 only —
  //     signed .2D needs PSRAQ and bails). Round bit via a PSRL/PSLL/PSRL
  //     bit-bracket, PADD, then (accumulate) PADD into Vd.
  //   * SQSHL / UQSHL / SQSHLU (saturating shift left) at vector H/S lanes
  //     (all three; UQSHL/SQSHLU also .2D via PSLLQ/PSRLQ/PCMPEQQ; SQSHL .2D
  //     needs PSRAQ recovery and bails). Shift-left, recover via the inverse
  //     shift, PCMPEQ-vs-source for a per-lane no-overflow mask, blend the
  //     shifted value with the per-lane saturation limit.
  // Everything else (byte-lane SHL/USHR/SSHR/SSRA/USRA/SLI/SRI/rounding/
  // saturating, SSHR/SSRA/SRSHR/SRSRA/SQSHL .2D which need PSRAQ, scalar
  // saturating B/H/S/D, and narrow/fixed-point conversions) calls Undefined()
  // which sets success_=false and bails the
  // region to lite — the lite tier already lowers those correctly (a heavy
  // bail is correct-but-slow, acceptable for the rarer shift variants).
  void AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& args) {
    if (!success()) {
      return;
    }
    const uint8_t immh = args.immh;
    if (immh == 0) {
      UndefinedReturningVoid();
      return;
    }
    const int32_t vn_off =
        static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
    const uint16_t immh_immb = static_cast<uint16_t>((immh << 3) | args.immb);

    switch (args.opcode) {
      case Decoder::AdvSimdShiftImmOpcode::kUshll:
      case Decoder::AdvSimdShiftImmOpcode::kSshll: {
        // USHLL/SSHLL Vd.<wide>, Vn.<narrow>, #shift.  esize = 8 <<
        // highest-set-bit(immh); Q=1 ("long2") widens the upper 64 of Vn.
        // Left-shift is bit-identical signed/unsigned, so PSLL{W,D,Q} is
        // shared after the signed/unsigned widen.  Result is always a full
        // 128-bit register (SetVRegFull q=true).
        if (immh & 0b1000) {  // RESERVED for shift-left-long.
          UndefinedReturningVoid();
          return;
        }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSshll);
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        if (args.q) {
          // Bring Vn[127:64] into the low 64 so PMOVSX/PMOVZX widens it.
          builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{8});
        }
        uint8_t shift_imm;
        if (immh & 0b0100) {  // 2S -> 2D
          shift_imm = static_cast<uint8_t>(immh_immb - 32);
          if (is_signed) {
            builder_.Gen<x86_64::PmovsxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          } else {
            builder_.Gen<x86_64::PmovzxdqXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          if (shift_imm != 0) {
            builder_.Gen<x86_64::PsllqXRegImm>(xn.machine_reg(), static_cast<int8_t>(shift_imm));
          }
        } else if (immh & 0b0010) {  // 4H -> 4S
          shift_imm = static_cast<uint8_t>(immh_immb - 16);
          if (is_signed) {
            builder_.Gen<x86_64::PmovsxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          } else {
            builder_.Gen<x86_64::PmovzxwdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          if (shift_imm != 0) {
            builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), static_cast<int8_t>(shift_imm));
          }
        } else {  // immh & 0b0001: 8B -> 8H
          shift_imm = static_cast<uint8_t>(immh_immb - 8);
          if (is_signed) {
            builder_.Gen<x86_64::PmovsxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          } else {
            builder_.Gen<x86_64::PmovzxbwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          if (shift_imm != 0) {
            builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), static_cast<int8_t>(shift_imm));
          }
        }
        SetVRegFull(args.rd, xn, /*q=*/true);
        return;
      }
      case Decoder::AdvSimdShiftImmOpcode::kShl:
      case Decoder::AdvSimdShiftImmOpcode::kUshr:
      case Decoder::AdvSimdShiftImmOpcode::kSshr: {
        // esize from immh: bit3->64, bit2->32, bit1->16, bit0->8 (byte, no
        // x86 packed shift -> bail to lite, which widens via PMOVSX/PACKSS).
        if (immh == 0b0001) {  // byte lane
          UndefinedReturningVoid();
          return;
        }
        uint8_t esize_bits;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else {  // immh & 0b0010
          esize_bits = 16;
        }
        const bool is_left =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kShl);
        // SHL count = immh:immb - esize (0..esize-1); USHR/SSHR count =
        // 2*esize - immh:immb (1..esize).  PSLL/PSRL/PSRA saturate to
        // 0/sign-fill at count>=esize, matching ARM at the boundary.
        const uint8_t shift_count =
            is_left ? static_cast<uint8_t>(immh_immb - esize_bits)
                    : static_cast<uint8_t>(2 * esize_bits - immh_immb);
        const bool is_arith =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSshr);
        // SSHR .2D / scalar-D needs PSRAQ (AVX-512F-VL only) -> bail to lite,
        // whose GPR fallback ships each 64-bit lane through SARQ.
        if (is_arith && esize_bits == 64) {
          UndefinedReturningVoid();
          return;
        }
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        const int8_t cnt = static_cast<int8_t>(shift_count);
        if (is_left) {
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), cnt);
          } else if (esize_bits == 32) {
            builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsllqXRegImm>(xn.machine_reg(), cnt);
          }
        } else if (is_arith) {  // SSHR, esize 16 or 32
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), cnt);
          }
        } else {  // USHR
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
          } else if (esize_bits == 32) {
            builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
          }
        }
        // Q=0 (incl. scalar) forms zero Vd[127:64] via SetVRegFull.
        SetVRegFull(args.rd, xn, args.q);
        return;
      }
      case Decoder::AdvSimdShiftImmOpcode::kSsra:
      case Decoder::AdvSimdShiftImmOpcode::kUsra: {
        // Shift-accumulate: Vd<i> += (Vn<i> >>{arith,logical} shift) per lane.
        // Mirrors lite_translator.h::AdvSimdShiftByImm SSRA/USRA (H/S/D lanes):
        // shift the source, then PADD the packed result into Vd's current value.
        // Byte lane needs PMOVSX/PMOVZX widening + PACK (no x86 packed byte
        // shift); SSRA .2D needs PSRAQ (AVX-512F-VL). Both bail to lite, which
        // ships a widening / GPR fallback (correct-but-slow).
        if (immh == 0b0001) {  // byte lane
          UndefinedReturningVoid();
          return;
        }
        uint8_t esize_bits;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else {  // immh & 0b0010
          esize_bits = 16;
        }
        const bool is_arith =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSsra);
        if (is_arith && esize_bits == 64) {  // SSRA .2D / scalar-D: PSRAQ only.
          UndefinedReturningVoid();
          return;
        }
        // SSRA/USRA count = 2*esize - immh:immb, range [1, esize]. At the
        // esize boundary PSRL saturates to 0 (USRA: add 0 -> Vd unchanged) and
        // PSRA sign-fills (SSRA: add 0/-1 per lane) — both match ARM.
        const uint8_t shift_count =
            static_cast<uint8_t>(2 * esize_bits - immh_immb);
        const int8_t cnt = static_cast<int8_t>(shift_count);
        const int32_t vd_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        // Load Vd's current value up front; rd==rn is safe because xn/xd are
        // independent temps and the writeback (SetVRegFull) happens last.
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        if (is_arith) {  // SSRA, esize 16 or 32.
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
            builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
          } else {  // 32
            builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), cnt);
            builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
          }
        } else {  // USRA, esize 16/32/64.
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
            builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
          } else if (esize_bits == 32) {
            builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
            builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
          } else {  // 64
            builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
            builder_.Gen<x86_64::PaddqXRegXReg>(xd.machine_reg(), xn.machine_reg());
          }
        }
        // Q=0 (.4H/.2S, incl. scalar) zero Vd[127:64] via SetVRegFull's D-form
        // merge, discarding the accumulate's garbage in the upper lanes.
        SetVRegFull(args.rd, xd, args.q);
        return;
      }
      case Decoder::AdvSimdShiftImmOpcode::kSrshr:
      case Decoder::AdvSimdShiftImmOpcode::kUrshr:
      case Decoder::AdvSimdShiftImmOpcode::kSrsra:
      case Decoder::AdvSimdShiftImmOpcode::kUrsra: {
        // Rounding right shift (+ accumulate). Per lane:
        //   *RSHR Vd<i> = floor((Vn<i> + 2^(shift-1)) / 2^shift)
        //   *RSRA Vd<i> = Vd<i> + floor((Vn<i> + 2^(shift-1)) / 2^shift)
        // Mirrors lite_translator.h::AdvSimdShiftByImm rounding family (H/S/D
        // packed path): shift Vn right (arith for signed, logical for
        // unsigned) into xn, isolate bit (shift-1) of Vn as the +round term in
        // xr via a PSRL/PSLL/PSRL bit-bracket, PADD them, then (accumulate)
        // PADD into Vd. Byte lane (needs PMOVZXBW widening) and signed .2D
        // SRSHR/SRSRA (need PSRAQ, AVX-512F-VL only) bail to lite, which ships
        // a widening / GPR fallback (correct-but-slow).
        if (immh == 0b0001) {  // byte lane
          UndefinedReturningVoid();
          return;
        }
        uint8_t esize_bits;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else {  // immh & 0b0010
          esize_bits = 16;
        }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrshr ||
             args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra);
        const bool is_accumulate =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSrsra ||
             args.opcode == Decoder::AdvSimdShiftImmOpcode::kUrsra);
        // Signed .2D needs PSRAQ (AVX-512F-VL) -> bail; lite's GPR fallback
        // ships each 64-bit lane through SARQ + round.
        if (is_signed && esize_bits == 64) {
          UndefinedReturningVoid();
          return;
        }
        // count = 2*esize - immh:immb, range [1, esize]. At count==esize PSRL
        // saturates to 0 and the round bit isolates the MSB, giving the
        // ARM-correct result (floor((x + 2^(esize-1)) / 2^esize) = MSB).
        const uint8_t shift_count =
            static_cast<uint8_t>(2 * esize_bits - immh_immb);
        const int8_t cnt = static_cast<int8_t>(shift_count);
        const int8_t cnt_minus_1 = static_cast<int8_t>(shift_count - 1);
        const int8_t esize_minus_1 = static_cast<int8_t>(esize_bits - 1);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xr = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        builder_.GenGetSimd<16>(xr.machine_reg(), vn_off);
        // Main shifted value in xn (arith for signed H/S, logical otherwise).
        if (esize_bits == 16) {
          if (is_signed) {
            builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
          }
        } else if (esize_bits == 32) {
          if (is_signed) {
            builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
          }
        } else {  // 64 (unsigned only; signed .2D bailed above)
          builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
        }
        // Round bit in xr: isolate bit (shift-1) of Vn into bit 0 per lane via
        // PSRL(cnt-1) then a PSLL(esize-1)/PSRL(esize-1) bit-0 bracket.
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xr.machine_reg(), cnt_minus_1);
          builder_.Gen<x86_64::PsllwXRegImm>(xr.machine_reg(), esize_minus_1);
          builder_.Gen<x86_64::PsrlwXRegImm>(xr.machine_reg(), esize_minus_1);
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PsrldXRegImm>(xr.machine_reg(), cnt_minus_1);
          builder_.Gen<x86_64::PslldXRegImm>(xr.machine_reg(), esize_minus_1);
          builder_.Gen<x86_64::PsrldXRegImm>(xr.machine_reg(), esize_minus_1);
        } else {  // 64
          builder_.Gen<x86_64::PsrlqXRegImm>(xr.machine_reg(), cnt_minus_1);
          builder_.Gen<x86_64::PsllqXRegImm>(xr.machine_reg(), esize_minus_1);
          builder_.Gen<x86_64::PsrlqXRegImm>(xr.machine_reg(), esize_minus_1);
        }
        // shifted + round bit.
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xr.machine_reg());
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xr.machine_reg());
        } else {
          builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xr.machine_reg());
        }
        FpRegister result = xn;
        if (is_accumulate) {
          FpRegister xd = AllocTempSimdReg();
          const int32_t vd_off =
              static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
          // Load orig Vd before the writeback; rd==rn safe (independent temps).
          builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
          } else if (esize_bits == 32) {
            builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
          } else {
            builder_.Gen<x86_64::PaddqXRegXReg>(xd.machine_reg(), xn.machine_reg());
          }
          result = xd;
        }
        // Q=0 (.4H/.2S, incl. scalar D) zero Vd[127:64] via SetVRegFull.
        SetVRegFull(args.rd, result, args.q);
        return;
      }
      case Decoder::AdvSimdShiftImmOpcode::kSli:
      case Decoder::AdvSimdShiftImmOpcode::kSri: {
        // Shift-and-insert. SLI: Vd<i> = (Vn<i> << shift) with Vd's low `shift`
        // bits preserved. SRI: Vd<i> = USHR(Vn<i>, shift) with Vd's high
        // `shift` bits preserved. Mirrors lite_translator.h::AdvSimdShiftByImm
        // SLI/SRI (H/S/D lanes): clear Vd's inserted bits with a PSLL+PSRL (SLI)
        // or PSRL+PSLL (SRI) round trip, shift Vn into place, POR the two. Byte
        // lane has no x86 packed byte shift and bails to lite.
        if (immh == 0b0001) {  // byte lane
          UndefinedReturningVoid();
          return;
        }
        uint8_t esize_bits;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else {  // immh & 0b0010
          esize_bits = 16;
        }
        const bool is_sli = (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSli);
        // SLI count = immh:immb - esize, range [0, esize-1].
        // SRI count = 2*esize - immh:immb, range [1, esize].
        const uint8_t shift_count =
            is_sli ? static_cast<uint8_t>(immh_immb - esize_bits)
                   : static_cast<uint8_t>(2 * esize_bits - immh_immb);
        const int8_t cnt = static_cast<int8_t>(shift_count);
        // inv = esize - shift is the width of Vd's preserved field. At the
        // boundaries the PSLL/PSRL round trip collapses to the ARM-correct
        // extreme: SLI count 0 -> inv==esize clears all of Vd (POR yields Vn);
        // SRI count==esize -> inv==0 preserves all of Vd and USHR gives 0.
        const int8_t inv = static_cast<int8_t>(esize_bits - shift_count);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        // Load Vd up front; rd==rn is safe because xn/xd are independent temps
        // and the writeback (SetVRegFull) happens last.
        const int32_t vd_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
        builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
        if (is_sli) {
          // Vd = (Vd keep low `shift` bits) | (Vn << shift).
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsllwXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsrlwXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), cnt);
          } else if (esize_bits == 32) {
            builder_.Gen<x86_64::PslldXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsrldXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), cnt);
          } else {  // 64
            builder_.Gen<x86_64::PsllqXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsrlqXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsllqXRegImm>(xn.machine_reg(), cnt);
          }
        } else {
          // SRI: Vd = (Vd keep high `shift` bits) | USHR(Vn, shift).
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsrlwXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsllwXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
          } else if (esize_bits == 32) {
            builder_.Gen<x86_64::PsrldXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PslldXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
          } else {  // 64
            builder_.Gen<x86_64::PsrlqXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsllqXRegImm>(xd.machine_reg(), inv);
            builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
          }
        }
        builder_.Gen<x86_64::PorXRegXReg>(xd.machine_reg(), xn.machine_reg());
        // Q=0 (.4H/.2S, incl. scalar D) zero Vd[127:64] via SetVRegFull.
        SetVRegFull(args.rd, xd, args.q);
        return;
      }
      case Decoder::AdvSimdShiftImmOpcode::kSqshl:
      case Decoder::AdvSimdShiftImmOpcode::kUqshl:
      case Decoder::AdvSimdShiftImmOpcode::kSqshlu: {
        // Saturating shift left by immediate. Per lane, shift Vn left by
        // `shift`; if the result overflows the element's saturating range,
        // replace it with the saturation limit. Mirrors
        // lite_translator.h::AdvSimdShiftByImm SQSHL/UQSHL/SQSHLU vector packed
        // pipeline (H/S lanes for all three; UQSHL/SQSHLU also .2D via
        // PSLLQ/PSRLQ/PCMPEQQ; SQSHL .2D needs PSRAQ recovery and bails).
        // Mechanism: shift-left into xs, recover with the inverse shift into xm,
        // PCMPEQ against the source to build a per-lane "no-overflow" mask,
        // blend the shifted value with the per-lane saturation target. SQSHLU
        // pre-zeros negative source lanes so the unsigned clamp yields the
        // architectural 0.
        //
        // Scalar B/H/S/D forms (args.scalar — saturating shift accepts any
        // non-zero immh) and byte-lane vector forms bail to lite, which has
        // full scalar + byte coverage (correct-but-slow).
        if (args.scalar || immh == 0b0001) {
          UndefinedReturningVoid();
          return;
        }
        uint8_t esize_bits;
        if (immh & 0b1000) {
          esize_bits = 64;
        } else if (immh & 0b0100) {
          esize_bits = 32;
        } else {  // immh & 0b0010
          esize_bits = 16;
        }
        const bool is_signed =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshl);
        const bool is_sqshlu =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshlu);
        // SQSHL .2D needs PSRAQ (AVX-512F-VL) for the signed back-shift
        // recovery -> bail; lite ships each 64-bit lane through a GPR fallback.
        if (is_signed && esize_bits == 64) {
          UndefinedReturningVoid();
          return;
        }
        // SQSHL/UQSHL/SQSHLU count = immh:immb - esize, range [0, esize-1].
        const uint8_t shift_count =
            static_cast<uint8_t>(immh_immb - esize_bits);
        const int8_t cnt = static_cast<int8_t>(shift_count);
        FpRegister xn = AllocTempSimdReg();
        FpRegister xs = AllocTempSimdReg();
        FpRegister xm = AllocTempSimdReg();
        // xt starts zeroed so its self-referencing PCMPEQ/PXOR idioms below
        // (all-ones / re-zero) read a defined register — the heavy tier's SSA
        // lifetime analysis rejects a use-before-def, unlike the lite tier's
        // physical registers.
        FpRegister xt = AllocZeroedSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);

        if (is_sqshlu) {
          // pos_xn = (Vn < 0) ? 0 : Vn. Pre-zero negative source lanes so the
          // unsigned clamp below yields the architectural 0. xt is already zero
          // (AllocZeroedSimdReg), so PCMPGT(xt, xn) yields the neg-lane mask.
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PcmpgtwXRegXReg>(xt.machine_reg(), xn.machine_reg());
          } else if (esize_bits == 32) {
            builder_.Gen<x86_64::PcmpgtdXRegXReg>(xt.machine_reg(), xn.machine_reg());
          } else {
            builder_.Gen<x86_64::PcmpgtqXRegXReg>(xt.machine_reg(), xn.machine_reg());
          }
          builder_.Gen<x86_64::PandnXRegXReg>(xt.machine_reg(), xn.machine_reg());  // ~neg & xn
          builder_.Gen<x86_64::MovdqaXRegXReg>(xn.machine_reg(), xt.machine_reg());
        }

        // xs = xn << cnt.
        builder_.Gen<x86_64::MovdqaXRegXReg>(xs.machine_reg(), xn.machine_reg());
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PsllwXRegImm>(xs.machine_reg(), cnt);
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PslldXRegImm>(xs.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsllqXRegImm>(xs.machine_reg(), cnt);
        }
        // xm = recover(xs): arithmetic (signed) / logical (unsigned) shift-right
        // by the same count. If it reproduces the source, the shift did not
        // overflow that lane.
        builder_.Gen<x86_64::MovdqaXRegXReg>(xm.machine_reg(), xs.machine_reg());
        if (is_signed) {  // SQSHL, esize 16/32 (esize 64 bailed above).
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsrawXRegImm>(xm.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsradXRegImm>(xm.machine_reg(), cnt);
          }
        } else {  // UQSHL / SQSHLU, esize 16/32/64.
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), cnt);
          } else if (esize_bits == 32) {
            builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsrlqXRegImm>(xm.machine_reg(), cnt);
          }
        }
        // xm = eq_mask: per-lane all-ones where recover(shift) == source.
        if (esize_bits == 16) {
          builder_.Gen<x86_64::PcmpeqwXRegXReg>(xm.machine_reg(), xn.machine_reg());
        } else if (esize_bits == 32) {
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(xm.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PcmpeqqXRegXReg>(xm.machine_reg(), xn.machine_reg());
        }

        // Saturation target in xt.
        if (is_signed) {
          // SQSHL: sat = INT_MAX XOR neg_mask(xn). (esize 16/32 only.)
          builder_.Gen<x86_64::PxorXRegXReg>(xt.machine_reg(), xt.machine_reg());
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PcmpgtwXRegXReg>(xt.machine_reg(), xn.machine_reg());
          } else {
            builder_.Gen<x86_64::PcmpgtdXRegXReg>(xt.machine_reg(), xn.machine_reg());
          }
          // xn is now free (its value was consumed by the eq_mask + neg_mask);
          // reuse it to hold INT_MAX = PSRL(all-ones, 1).
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(xn.machine_reg(), xn.machine_reg());
          if (esize_bits == 16) {
            builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{1});
          } else {
            builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{1});
          }
          builder_.Gen<x86_64::PxorXRegXReg>(xt.machine_reg(), xn.machine_reg());
        } else {
          // UQSHL / SQSHLU: sat = UINT_MAX (all-ones) per lane.
          builder_.Gen<x86_64::PcmpeqdXRegXReg>(xt.machine_reg(), xt.machine_reg());
        }

        // Blend: result = (eq_mask & shifted) | (~eq_mask & sat).
        builder_.Gen<x86_64::PandXRegXReg>(xs.machine_reg(), xm.machine_reg());
        builder_.Gen<x86_64::PandnXRegXReg>(xm.machine_reg(), xt.machine_reg());
        builder_.Gen<x86_64::PorXRegXReg>(xs.machine_reg(), xm.machine_reg());
        // Q=0 (.4H/.2S) zero Vd[127:64] via SetVRegFull's D-form merge.
        SetVRegFull(args.rd, xs, args.q);
        return;
      }
      case Decoder::AdvSimdShiftImmOpcode::kShrn:
      case Decoder::AdvSimdShiftImmOpcode::kRshrn: {
        // SHRN/RSHRN Vd.<narrow>, Vn.<wide>, #shift — shift-right narrow.
        // Mirrors lite_translator.h::AdvSimdShiftByImm's non-saturating narrow
        // path: (RSHRN only) broadcast+add the per-lane rounding constant ->
        // logical right shift (PSRL{W,D,Q}) -> PSHUFB gather of each wide
        // lane's low half into the packed low 64. src_bits = 8<<highest-set-
        // bit(immh); dst = src/2. immh bit3 set is RESERVED (bail). SHRN/RSHRN
        // have no scalar form, so args.scalar is always false here. The
        // saturating narrows (SQSHRN/UQSHRN/SQRSHRN/UQRSHRN/SQSHRUN/SQRSHRUN)
        // still bail to lite via the default case below.
        if (immh & 0b1000) {  // RESERVED for narrowing shifts.
          UndefinedReturningVoid();
          return;
        }
        uint8_t src_bits;
        if (immh & 0b0100) {
          src_bits = 64;
        } else if (immh & 0b0010) {
          src_bits = 32;
        } else {  // immh == 0b0001
          src_bits = 16;
        }
        const uint8_t narrow_rshift = static_cast<uint8_t>(src_bits - immh_immb);
        const bool is_rounding =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kRshrn);
        const int8_t cnt = static_cast<int8_t>(narrow_rshift);
        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
        if (is_rounding) {
          // Broadcast the per-lane rounding constant (1 << (rshift-1)) across
          // every src-width lane, then add. Wrap-on-overflow is benign — the
          // shift range [1, dst_bits] keeps the add's carry inside the low
          // half PSHUFB gathers; any wrap at bit src_bits is dropped anyway.
          const uint64_t round_lane = uint64_t{1} << (narrow_rshift - 1);
          uint64_t round_pattern;
          if (src_bits == 16) {
            round_pattern = round_lane * uint64_t{0x0001000100010001ULL};
          } else if (src_bits == 32) {
            round_pattern = round_lane * uint64_t{0x0000000100000001ULL};
          } else {
            round_pattern = round_lane;
          }
          FpRegister xround = AllocTempSimdReg();
          Register gr = std::get<0>(
              Gen<x86_64::MovqRegImm>(static_cast<int64_t>(round_pattern)));
          builder_.Gen<x86_64::MovqXRegReg>(xround.machine_reg(), gr);
          builder_.Gen<x86_64::PinsrqXRegRegImm>(xround.machine_reg(), gr, int8_t{1});
          if (src_bits == 16) {
            builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(), xround.machine_reg());
          } else if (src_bits == 32) {
            builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(), xround.machine_reg());
          } else {
            builder_.Gen<x86_64::PaddqXRegXReg>(xn.machine_reg(), xround.machine_reg());
          }
        }
        // Logical right shift by cnt (in [1, dst_bits] <= src_bits/2), so the
        // high bits the narrow discards never reach the kept low half — SHRN's
        // untyped shift is bit-identical to a logical shift here.
        if (src_bits == 16) {
          builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
        } else if (src_bits == 32) {
          builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
        } else {
          builder_.Gen<x86_64::PsrlqXRegImm>(xn.machine_reg(), cnt);
        }
        // PSHUFB narrow: the low-8 selector entries gather each wide lane's low
        // half into the packed low 64; the high-8 entries (0x80) zero the upper
        // 64.  src 16 -> low byte of each .8H lane; src 32 -> low half of each
        // .4S lane; src 64 -> low word of each .2D lane.
        int64_t mask_lo;
        if (src_bits == 16) {
          mask_lo = static_cast<int64_t>(0x0E0C0A0806040200LL);
        } else if (src_bits == 32) {
          mask_lo = static_cast<int64_t>(0x0D0C090805040100LL);
        } else {
          mask_lo = static_cast<int64_t>(0x0B0A090803020100LL);
        }
        FpRegister xmask = AllocTempSimdReg();
        Register gm = std::get<0>(Gen<x86_64::MovqRegImm>(mask_lo));
        builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), gm);
        Register gmhi = std::get<0>(
            Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0x8080808080808080ULL)));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(xmask.machine_reg(), gmhi, int8_t{1});
        builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), xmask.machine_reg());
        if (!args.q) {
          // SHRN: narrowed lanes sit in the low 64; SetVRegFull zeroes
          // Vd[127:64] via its D-form merge.
          SetVRegFull(args.rd, xn, /*q=*/false);
        } else {
          // SHRN2: place the narrowed lanes in Vd[127:64], preserving Vd[63:0].
          const int32_t vd_off =
              static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
          FpRegister xd = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
          FpRegister xdlow = AllocZeroedSimdReg();
          builder_.Gen<x86_64::MovsdXRegXReg>(xdlow.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PorXRegXReg>(xdlow.machine_reg(), xn.machine_reg());
          builder_.GenSetSimd<16>(vd_off, xdlow.machine_reg());
        }
        return;
      }
      case Decoder::AdvSimdShiftImmOpcode::kSqshrn:
      case Decoder::AdvSimdShiftImmOpcode::kUqshrn:
      case Decoder::AdvSimdShiftImmOpcode::kSqshrun:
      case Decoder::AdvSimdShiftImmOpcode::kSqrshrn:
      case Decoder::AdvSimdShiftImmOpcode::kUqrshrn:
      case Decoder::AdvSimdShiftImmOpcode::kSqrshrun: {
        // (Rounding and non-rounding) saturating shift-right narrow (vector,
        // 16/32-bit source).  Mirrors lite_translator.h::AdvSimdShiftByImm's
        // saturating narrow path: (rounding only) add the per-lane rounding
        // bias -> shift each wide lane right (arithmetic for a signed source,
        // logical for unsigned) -> clamp to the destination range ->
        // PSHUFB-gather the low half of each lane into the packed low 64.
        //   SQSHRN/SQRSHRN   (signed->signed):     PSRA{W,D} + PMINS/PMAXS clamp.
        //   UQSHRN/UQRSHRN   (unsigned->unsigned): PSRL{W,D} + PMINU clamp.
        //   SQSHRUN/SQRSHRUN (signed->unsigned):   PSRA{W,D} + PMAXS-vs-0 + PMINS.
        // Rounding bias:
        //   SQRSHRN: signed-saturating pre-add of (1<<(shift-1)) — PADDSW for
        //     src16, PMINSD-preclamp+PADDD for src32 (no integer PADDSD).
        //   UQRSHRN: unsigned-saturating pre-add — PADDUSW for src16,
        //     PMINUD-preclamp+PADDD for src32.
        //   SQRSHRUN: the carry-bit identity (x+(1<<(s-1)))>>s == (x>>s) +
        //     bit(s-1 of x) for arithmetic >> — compute the carry pre-shift
        //     (PSLL then PSRL to bit 0), add it after the shift; avoids a
        //     premature signed saturation that would drop 1 LSB at the boundary.
        // src=64 needs PSRAQ (AVX-512F-VL) for the signed shift and a manual
        // hi32-nonzero rewrite for the unsigned clamp; UQRSHRN src=64 needs a
        // saturating PADDQ — all of which the lite tier also bails, so bail
        // here.  The scalar forms fall to lite via the bail below.
        if (immh & 0b1000) {  // RESERVED for narrowing shifts.
          UndefinedReturningVoid();
          return;
        }
        if (args.scalar) {  // Scalar saturating narrow -> lite handles it.
          UndefinedReturningVoid();
          return;
        }
        uint8_t src_bits;
        if (immh & 0b0100) {
          src_bits = 64;
        } else if (immh & 0b0010) {
          src_bits = 32;
        } else {  // immh == 0b0001
          src_bits = 16;
        }
        if (src_bits == 64) {  // needs PSRAQ / manual clamp -> bail to lite.
          UndefinedReturningVoid();
          return;
        }
        const bool is_rounding =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqrshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrun);
        const bool is_saturating_signed =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrn);
        const bool is_saturating_unsigned =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqshrn) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kUqrshrn);
        const bool is_signed_to_unsigned =
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqshrun) ||
            (args.opcode == Decoder::AdvSimdShiftImmOpcode::kSqrshrun);
        // Signed source (SQSHRN/SQSHRUN + rounding) uses an arithmetic right
        // shift; unsigned source (UQSHRN/UQRSHRN) uses a logical right shift.
        const bool uses_signed_shift =
            is_saturating_signed || is_signed_to_unsigned;
        const uint8_t narrow_rshift = static_cast<uint8_t>(src_bits - immh_immb);
        const int8_t cnt = static_cast<int8_t>(narrow_rshift);

        // Materialize a broadcast constant (`pattern` in both qwords) into a
        // fresh SIMD reg — the rounding bias and the clamp bounds need this.
        auto broadcast = [&](uint64_t pattern) -> FpRegister {
          FpRegister x = AllocTempSimdReg();
          Register gr =
              std::get<0>(Gen<x86_64::MovqRegImm>(static_cast<int64_t>(pattern)));
          builder_.Gen<x86_64::MovqXRegReg>(x.machine_reg(), gr);
          builder_.Gen<x86_64::PinsrqXRegRegImm>(x.machine_reg(), gr, int8_t{1});
          return x;
        };

        FpRegister xn = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);

        // xcarry holds the SQRSHRUN pre-shift rounding carry (0/1 per lane);
        // computed while xn is intact, added back after the arithmetic shift.
        FpRegister xcarry;
        if (is_rounding) {
          const uint64_t round_lane = uint64_t{1} << (narrow_rshift - 1);
          const uint64_t round_pattern =
              (src_bits == 16) ? round_lane * uint64_t{0x0001000100010001ULL}
                               : round_lane * uint64_t{0x0000000100000001ULL};
          if (is_saturating_unsigned) {
            FpRegister xround = broadcast(round_pattern);
            if (src_bits == 16) {
              // PADDUSW: unsigned-saturating word add (SSE2).
              builder_.Gen<x86_64::PadduswXRegXReg>(xn.machine_reg(),
                                                    xround.machine_reg());
            } else {
              // No unsigned-saturating PADDUSD in baseline SSE; pre-clamp xn to
              // (0xFFFFFFFF - round_lane) so the plain PADDD cannot overflow
              // past UINT32_MAX. Lanes hitting the pre-clamp settle at exactly
              // UINT32_MAX, which the post-shift PMINUD then narrows correctly.
              const uint32_t clamp_lane =
                  0xFFFFFFFFu - static_cast<uint32_t>(round_lane);
              const uint64_t clamp_pattern =
                  (uint64_t{clamp_lane} << 32) | uint64_t{clamp_lane};
              FpRegister xclamp = broadcast(clamp_pattern);
              builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(),
                                                   xclamp.machine_reg());
              builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(),
                                                  xround.machine_reg());
            }
          } else if (is_signed_to_unsigned) {
            // SQRSHRUN carry-bit identity: isolate bit (cnt-1) of each lane into
            // bit 0 (shift it to the top, then logically back). No wide add ->
            // no premature saturation.
            xcarry = AllocTempSimdReg();
            builder_.Gen<x86_64::MovdqaXRegXReg>(xcarry.machine_reg(),
                                                 xn.machine_reg());
            if (src_bits == 16) {
              builder_.Gen<x86_64::PsllwXRegImm>(
                  xcarry.machine_reg(), static_cast<int8_t>(16 - narrow_rshift));
              builder_.Gen<x86_64::PsrlwXRegImm>(xcarry.machine_reg(), int8_t{15});
            } else {
              builder_.Gen<x86_64::PslldXRegImm>(
                  xcarry.machine_reg(), static_cast<int8_t>(32 - narrow_rshift));
              builder_.Gen<x86_64::PsrldXRegImm>(xcarry.machine_reg(), int8_t{31});
            }
          } else {  // SQRSHRN (signed->signed).
            FpRegister xround = broadcast(round_pattern);
            if (src_bits == 16) {
              // PADDSW: signed-saturating word add (SSE2).
              builder_.Gen<x86_64::PaddswXRegXReg>(xn.machine_reg(),
                                                   xround.machine_reg());
            } else {
              // No integer PADDSD in baseline SSE; pre-clamp xn to
              // (0x7FFFFFFF - round_lane) so the plain PADDD cannot overflow
              // positively. The round constant is positive, so no negative
              // underflow is possible. Lanes hitting the pre-clamp settle at
              // exactly INT32_MAX after the add, which the post-shift
              // PMINSD-vs-signed-max then drives to the saturated dst value.
              const uint32_t clamp_lane =
                  0x7FFFFFFFu - static_cast<uint32_t>(round_lane);
              const uint64_t clamp_pattern =
                  (uint64_t{clamp_lane} << 32) | uint64_t{clamp_lane};
              FpRegister xclamp = broadcast(clamp_pattern);
              builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(),
                                                   xclamp.machine_reg());
              builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(),
                                                  xround.machine_reg());
            }
          }
        }

        if (uses_signed_shift) {
          if (src_bits == 16) {
            builder_.Gen<x86_64::PsrawXRegImm>(xn.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsradXRegImm>(xn.machine_reg(), cnt);
          }
        } else {
          if (src_bits == 16) {
            builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), cnt);
          } else {
            builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), cnt);
          }
        }

        if (is_rounding && is_signed_to_unsigned) {
          // SQRSHRUN: add the pre-shift rounding carry. Post-shift magnitudes
          // are small (|x>>cnt| <= INT*_MAX>>1), so a plain PADD cannot overflow
          // before the unsigned clamp below.
          if (src_bits == 16) {
            builder_.Gen<x86_64::PaddwXRegXReg>(xn.machine_reg(),
                                                xcarry.machine_reg());
          } else {
            builder_.Gen<x86_64::PadddXRegXReg>(xn.machine_reg(),
                                                xcarry.machine_reg());
          }
        }

        if (is_saturating_signed) {
          // Clamp each src-lane to the signed dst range:
          //   dst 8 : [-128, 127]   as signed 16 = [0xFF80, 0x007F].
          //   dst 16: [-32768,32767] as signed 32 = [0xFFFF8000, 0x00007FFF].
          uint64_t sat_max_pattern, sat_min_pattern;
          if (src_bits == 16) {
            sat_max_pattern = 0x007F007F007F007FULL;
            sat_min_pattern = 0xFF80FF80FF80FF80ULL;
          } else {
            sat_max_pattern = 0x00007FFF00007FFFULL;
            sat_min_pattern = 0xFFFF8000FFFF8000ULL;
          }
          FpRegister xsatmax = broadcast(sat_max_pattern);
          FpRegister xsatmin = broadcast(sat_min_pattern);
          if (src_bits == 16) {
            builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xsatmax.machine_reg());
            builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xsatmin.machine_reg());
          } else {
            builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xsatmax.machine_reg());
            builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xsatmin.machine_reg());
          }
        } else if (is_signed_to_unsigned) {
          // SQSHRUN: clamp the post-shift signed value to the unsigned dst
          // range [0, 2^dst_bits - 1].  PMAXS-vs-zero pins negatives at 0;
          // after that every lane is non-negative, so a signed PMINS against
          // the positive dst-max is equivalent to unsigned-min.
          const uint64_t sat_max_pattern =
              (src_bits == 16) ? 0x00FF00FF00FF00FFULL   // dst 8: 0xFF
                               : 0x0000FFFF0000FFFFULL;  // dst 16: 0xFFFF
          FpRegister xsatmax = broadcast(sat_max_pattern);
          FpRegister xzero = AllocZeroedSimdReg();
          if (src_bits == 16) {
            builder_.Gen<x86_64::PmaxswXRegXReg>(xn.machine_reg(), xzero.machine_reg());
            builder_.Gen<x86_64::PminswXRegXReg>(xn.machine_reg(), xsatmax.machine_reg());
          } else {
            builder_.Gen<x86_64::PmaxsdXRegXReg>(xn.machine_reg(), xzero.machine_reg());
            builder_.Gen<x86_64::PminsdXRegXReg>(xn.machine_reg(), xsatmax.machine_reg());
          }
        } else {  // UQSHRN: unsigned-min clamp to (1 << dst_bits) - 1.
          const uint64_t sat_pattern =
              (src_bits == 16) ? 0x00FF00FF00FF00FFULL
                               : 0x0000FFFF0000FFFFULL;
          FpRegister xsat = broadcast(sat_pattern);
          if (src_bits == 16) {
            builder_.Gen<x86_64::PminuwXRegXReg>(xn.machine_reg(), xsat.machine_reg());
          } else {
            builder_.Gen<x86_64::PminudXRegXReg>(xn.machine_reg(), xsat.machine_reg());
          }
        }

        // PSHUFB narrow gather: the low-8 selector entries gather each wide
        // lane's low half into the packed low 64; the high-8 entries (0x80)
        // zero the upper 64.  Same layout as the SHRN/RSHRN path above.
        const int64_t mask_lo =
            (src_bits == 16) ? static_cast<int64_t>(0x0E0C0A0806040200LL)
                             : static_cast<int64_t>(0x0D0C090805040100LL);
        FpRegister xmask = AllocTempSimdReg();
        Register gm = std::get<0>(Gen<x86_64::MovqRegImm>(mask_lo));
        builder_.Gen<x86_64::MovqXRegReg>(xmask.machine_reg(), gm);
        Register gmhi = std::get<0>(
            Gen<x86_64::MovqRegImm>(static_cast<int64_t>(0x8080808080808080ULL)));
        builder_.Gen<x86_64::PinsrqXRegRegImm>(xmask.machine_reg(), gmhi, int8_t{1});
        builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), xmask.machine_reg());
        if (!args.q) {
          // Q=0: narrowed lanes in the low 64; SetVRegFull zeroes Vd[127:64].
          SetVRegFull(args.rd, xn, /*q=*/false);
        } else {
          // Q=1 ("...2" form): place narrowed lanes in Vd[127:64], preserve
          // Vd[63:0].
          const int32_t vd_off =
              static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);
          FpRegister xd = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
          FpRegister xdlow = AllocZeroedSimdReg();
          builder_.Gen<x86_64::MovsdXRegXReg>(xdlow.machine_reg(), xd.machine_reg());
          builder_.Gen<x86_64::PslldqXRegImm>(xn.machine_reg(), int8_t{8});
          builder_.Gen<x86_64::PorXRegXReg>(xdlow.machine_reg(), xn.machine_reg());
          builder_.GenSetSimd<16>(vd_off, xdlow.machine_reg());
        }
        return;
      }
      default:
        UndefinedReturningVoid();
        return;
    }
  }

  // Heavy-tier mirror of lite_translator.h::AdvSimdVecXIndexedElement's
  // integer MUL/MLA/MLS by-element block (halfword .4h/.8h size=01, word
  // .2s/.4s size=10):
  //   MUL: Vd = Vn * broadcast(Vm.lane[index])
  //   MLA: Vd = Vd + Vn * broadcast(Vm.lane[index])
  //   MLS: Vd = Vd - Vn * broadcast(Vm.lane[index])
  // No saturation, no widening — the bottom esize bits of each host product
  // match the architectural result modulo 2^esize. Broadcast is Pshuflw+Pshufd
  // (halfword; Psrldq 8 first if index>=4) or Pshufd (word). Q=0 upper-64
  // discard via SetVRegFull(rd, res, q). All other by-element opcodes still
  // bail to the lite tier — emit NOTHING before a bail.
  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs& args) {
    if (!success()) {
      return;
    }
    using Op = Decoder::AdvSimdVecXIdxOpcode;

    // Widening MUL/MAC by element: SMULL/UMULL/SMLAL/UMLAL/SMLSL/UMLSL.
    // Heavy-tier mirror of lite_translator.h's widening by-element arms
    // (size=01: .4h/.8h -> .4s; size=10: .2s/.4s -> .2d). The destination is
    // always 128-bit regardless of Q — widening forms do NOT zero the upper
    // half, so SetVRegFull(rd, res, /*q=*/true) stores the full register.
    //   *MULL : Vd  = widen(Vn.selected) * widen(broadcast(Vm.lane[index]))
    //   *MLAL : Vd += widen(Vn.selected) * widen(broadcast(Vm.lane[index]))
    //   *MLSL : Vd -= widen(Vn.selected) * widen(broadcast(Vm.lane[index]))
    // Q=0 selects the low source half (Vn bytes 0..7); Q=1 selects the high
    // half (Vn bytes 8..15) — brought into the low quad by PSRLDQ 8 before the
    // register-form PMOVSX/PMOVZX widen (the lite tier uses a memory-operand
    // PMOVSX at vn_off+(q?8:0); heavy has no such op, so shift-then-widen).
    // Both signed 16x16 and unsigned 16x16 products fit in 32 bits, so PMULLD's
    // signed low-32 result is correct for both forms at size=01; size=10 uses
    // PMULDQ (signed) / PMULUDQ (unsigned) for the 32x32 -> 64 products.
    //
    // Verified encodings (ARM ARM C7.2):
    //   smull  v0.4s, v1.4h, v2.h[0] = 0x0F42A020
    //   umull  v0.4s, v1.4h, v2.h[0] = 0x2F42A020
    //   smlal  v0.4s, v1.4h, v2.h[0] = 0x0F422020
    //   smlsl  v0.4s, v1.4h, v2.h[0] = 0x0F426020
    //   smull  v0.2d, v1.2s, v2.s[0] = 0x0F82A020
    //   umlal  v0.2d, v1.2s, v2.s[0] = 0x2F822020
    if (args.opcode == Op::kSmullIdx || args.opcode == Op::kUmullIdx ||
        args.opcode == Op::kSmlalIdx || args.opcode == Op::kUmlalIdx ||
        args.opcode == Op::kSmlslIdx || args.opcode == Op::kUmlslIdx) {
      if (args.size != 0b01 && args.size != 0b10) {
        UndefinedReturningVoid();
        return;
      }
      const bool w_signed = (args.opcode == Op::kSmullIdx ||
                             args.opcode == Op::kSmlalIdx ||
                             args.opcode == Op::kSmlslIdx);
      const bool w_accum = (args.opcode == Op::kSmlalIdx ||
                            args.opcode == Op::kUmlalIdx ||
                            args.opcode == Op::kSmlslIdx ||
                            args.opcode == Op::kUmlslIdx);
      const bool w_sub = (args.opcode == Op::kSmlslIdx ||
                          args.opcode == Op::kUmlslIdx);

      const int32_t w_vn_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
      const int32_t w_vm_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);
      const int32_t w_vd_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);

      FpRegister wm = AllocTempSimdReg();
      FpRegister wn = AllocTempSimdReg();
      builder_.GenGetSimd<16>(wm.machine_reg(), w_vm_off);
      builder_.GenGetSimd<16>(wn.machine_reg(), w_vn_off);
      // Bring Vn's selected source half (Q=1 -> bytes 8..15) into the low quad.
      if (args.q) {
        builder_.Gen<x86_64::PsrldqXRegImm>(wn.machine_reg(), int8_t{8});
      }

      if (args.size == 0b01) {
        // Broadcast Vm.h[index] across all 8 halfword lanes.
        if (args.index >= 4) {
          builder_.Gen<x86_64::PsrldqXRegImm>(wm.machine_reg(), int8_t{8});
        }
        const uint8_t i = args.index & 0b11;
        const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        builder_.Gen<x86_64::PshuflwXRegXRegImm>(wm.machine_reg(), wm.machine_reg(), imm);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(wm.machine_reg(), wm.machine_reg(), int8_t{0x44});
        // Widen low 4 halfwords of each source to 4 x 32-bit lanes.
        if (w_signed) {
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(wm.machine_reg(), wm.machine_reg());
          builder_.Gen<x86_64::PmovsxwdXRegXReg>(wn.machine_reg(), wn.machine_reg());
        } else {
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(wm.machine_reg(), wm.machine_reg());
          builder_.Gen<x86_64::PmovzxwdXRegXReg>(wn.machine_reg(), wn.machine_reg());
        }
        builder_.Gen<x86_64::PmulldXRegXReg>(wn.machine_reg(), wm.machine_reg());
        FpRegister w_result = wn;
        if (w_accum) {
          FpRegister wd = AllocTempSimdReg();
          builder_.GenGetSimd<16>(wd.machine_reg(), w_vd_off);
          if (w_sub) {
            builder_.Gen<x86_64::PsubdXRegXReg>(wd.machine_reg(), wn.machine_reg());
          } else {
            builder_.Gen<x86_64::PadddXRegXReg>(wd.machine_reg(), wn.machine_reg());
          }
          w_result = wd;
        }
        SetVRegFull(args.rd, w_result, /*q=*/true);
        return;
      }

      // size=10: word sources -> .2d. Broadcast Vm.s[index] across all 4 dword
      // lanes; PMULDQ/PMULUDQ read dword positions 0 and 2, so the broadcast
      // suffices (no widen of Vm). Widen Vn's low 2 dwords to 2 qword lanes.
      {
        const uint8_t i = args.index & 0b11;
        const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
        builder_.Gen<x86_64::PshufdXRegXRegImm>(wm.machine_reg(), wm.machine_reg(), imm);
      }
      if (w_signed) {
        builder_.Gen<x86_64::PmovsxdqXRegXReg>(wn.machine_reg(), wn.machine_reg());
        builder_.Gen<x86_64::PmuldqXRegXReg>(wn.machine_reg(), wm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmovzxdqXRegXReg>(wn.machine_reg(), wn.machine_reg());
        builder_.Gen<x86_64::PmuludqXRegXReg>(wn.machine_reg(), wm.machine_reg());
      }
      FpRegister w_result = wn;
      if (w_accum) {
        FpRegister wd = AllocTempSimdReg();
        builder_.GenGetSimd<16>(wd.machine_reg(), w_vd_off);
        if (w_sub) {
          builder_.Gen<x86_64::PsubqXRegXReg>(wd.machine_reg(), wn.machine_reg());
        } else {
          builder_.Gen<x86_64::PaddqXRegXReg>(wd.machine_reg(), wn.machine_reg());
        }
        w_result = wd;
      }
      SetVRegFull(args.rd, w_result, /*q=*/true);
      return;
    }

    if (args.opcode != Op::kMul && args.opcode != Op::kMla &&
        args.opcode != Op::kMls) {
      UndefinedReturningVoid();
      return;
    }
    if (args.size != 0b01 && args.size != 0b10) {
      UndefinedReturningVoid();
      return;
    }
    const bool is_halfword = (args.size == 0b01);

    const int32_t vn_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
    const int32_t vm_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);
    const int32_t vd_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rd * 16);

    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);

    // Broadcast Vm.lane[index] across every destination lane.
    if (is_halfword) {
      if (args.index >= 4) {
        builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{8});
      }
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      builder_.Gen<x86_64::PshuflwXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), int8_t{0x44});
    } else {
      const uint8_t i = args.index & 0b11;
      const int8_t imm = static_cast<int8_t>((i << 6) | (i << 4) | (i << 2) | i);
      builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
    }

    FpRegister result = xn;
    if (args.opcode == Op::kMul) {
      if (is_halfword) {
        builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
    } else {
      FpRegister xd = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
      if (is_halfword) {
        builder_.Gen<x86_64::PmullwXRegXReg>(xn.machine_reg(), xm.machine_reg());
      } else {
        builder_.Gen<x86_64::PmulldXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      if (args.opcode == Op::kMla) {
        if (is_halfword) {
          builder_.Gen<x86_64::PaddwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PadddXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
      } else {
        if (is_halfword) {
          builder_.Gen<x86_64::PsubwXRegXReg>(xd.machine_reg(), xn.machine_reg());
        } else {
          builder_.Gen<x86_64::PsubdXRegXReg>(xd.machine_reg(), xn.machine_reg());
        }
      }
      result = xd;
    }

    SetVRegFull(args.rd, result, args.q);
  }

  void AdvSimdScalarXIndexedElement(const Decoder::AdvSimdScalarXIdxArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  //
  // Advanced SIMD (decomposed-primitive forms).
  //

  // EXT Vd.<T>, Vn.<T>, Vm.<T>, #index — extract a vector from the byte-wise
  // concatenation Vm:Vn (Vn is the low half) starting at byte `index`. Mirrors
  // lite_translator.h::AdvSimdExtract: for the 8-byte form (Q=0) PUNPCKLQDQ
  // packs Vn.low64 into bytes 0..7 and Vm.low64 into bytes 8..15 so a single
  // PSRLDQ walks a contiguous 16-byte window; for the 16-byte form (Q=1) the
  // window is (Vn >> index) | (Vm << (16-index)) via PSRLDQ/PSLLDQ + POR. The
  // index-out-of-range cases (imm4[3]=1 for Q=0, index>=16 for Q=1) are
  // UNDEFINED and bail, matching the lite tier.
  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    if (!success()) {
      return;
    }
    const int32_t vn_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + rn * 16);
    const int32_t vm_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + rm * 16);
    if (!q) {
      // 64-bit form: 8-byte window, index 0..7 (imm4[3] set is UNDEFINED).
      if (index >= 8) {
        UndefinedReturningVoid();
        return;
      }
      FpRegister xn = AllocTempSimdReg();
      FpRegister xm = AllocTempSimdReg();
      builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
      builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
      // Concatenate Vn.low64 (bytes 0..7) with Vm.low64 (bytes 8..15); the
      // upper 64 of Vn/Vm never participate in the Q=0 window.
      builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      if (index != 0) {
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), static_cast<int8_t>(index));
      }
      // The wanted 8-byte window sits in the low 64 bits; SetVRegFull(q=false)
      // zero-extends the upper 64.
      SetVRegFull(rd, xn, /*q=*/false);
      return;
    }
    // 128-bit form: 16-byte window, index 0..15.
    FpRegister xn = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    if (index == 0) {
      SetVRegFull(rd, xn, /*q=*/true);
      return;
    }
    if (index >= 16) {
      UndefinedReturningVoid();
      return;
    }
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);
    // result = (Vn >> index bytes) | (Vm << (16-index) bytes).
    builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), static_cast<int8_t>(index));
    builder_.Gen<x86_64::PslldqXRegImm>(xm.machine_reg(), static_cast<int8_t>(16 - index));
    builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xm.machine_reg());
    SetVRegFull(rd, xn, /*q=*/true);
  }

  // ZIP1/ZIP2, UZP1/UZP2, TRN1/TRN2 vector permute — heavy-tier mirror of
  // lite_translator.h::AdvSimdPermute. Register-domain-pure, straight-line SSE:
  //   ZIP  -> PUNPCKL/H{bw,wd,dq,qdq} (interleave lower/upper halves)
  //   UZP  -> PACKUSWB/PACKUSDW (byte/halfword) or SHUFPS (word); .2D == ZIP.2D
  //   TRN  -> PSHUFB + POR with materialized per-byte masks (byte/halfword) or
  //           PSHUFD + PUNPCKLDQ (word); .2D == ZIP.2D
  // opcode (3 bits, from Decoder::DecodeAdvSimd): 001=UZP1 010=TRN1 011=ZIP1
  // 101=UZP2 110=TRN2 111=ZIP2 (same encoding the lite tier consumes).
  // SetVRegFull(q=false) zeroes Vd[127:64], so no manual upper-zero tail is
  // needed (unlike the lite path). Q=0 .2D is reserved and bails; unknown
  // opcodes bail. The emulator host always has SSE4.1 (matching the sibling
  // USHLL/widening handlers, which use PACKUSDW/PMOVZX unguarded), so the
  // UZP .8H PACKUSDW form is not host-gated here.
  void AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size, uint8_t opcode, bool q) {
    if (!success()) {
      return;
    }
    const bool is_zip = (opcode == 0b011 || opcode == 0b111);
    const bool is_uzp = (opcode == 0b001 || opcode == 0b101);
    const bool is_trn = (opcode == 0b010 || opcode == 0b110);
    if (!is_zip && !is_uzp && !is_trn) {
      UndefinedReturningVoid();
      return;
    }
    if (size > 0b11) {
      UndefinedReturningVoid();
      return;
    }
    // Q=0 .2D is reserved by the ARM ARM (encoding restricted).
    if (!q && size == 0b11) {
      UndefinedReturningVoid();
      return;
    }
    const int32_t vn_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + rn * 16);
    const int32_t vm_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + rm * 16);
    FpRegister xn = AllocTempSimdReg();
    FpRegister xm = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xn.machine_reg(), vn_off);
    builder_.GenGetSimd<16>(xm.machine_reg(), vm_off);

    if (is_zip) {
      const bool is_zip2 = (opcode == 0b111);
      if (!q && is_zip2) {
        // Shift each source right by 4 bytes so the (originally upper-half)
        // bytes 4..7 land at positions 0..3; PUNPCKL then picks them up.
        builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{4});
        builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{4});
      }
      if (q && is_zip2) {
        switch (size) {
          case 0b00: builder_.Gen<x86_64::PunpckhbwXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
          case 0b01: builder_.Gen<x86_64::PunpckhwdXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
          case 0b10: builder_.Gen<x86_64::PunpckhdqXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
          case 0b11: builder_.Gen<x86_64::PunpckhqdqXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
        }
      } else {
        switch (size) {
          case 0b00: builder_.Gen<x86_64::PunpcklbwXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
          case 0b01: builder_.Gen<x86_64::PunpcklwdXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
          case 0b10: builder_.Gen<x86_64::PunpckldqXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
          case 0b11: builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg()); break;
        }
      }
    } else if (is_uzp) {
      const bool is_uzp2 = (opcode == 0b101);
      if (!q) {
        // Combine the lower 8 bytes of Vn and Vm into xn = [vn_lo | vm_lo] so
        // subsequent PACKUS/SHUFPS consumes a single source; the Q=0 tail then
        // discards the duplicated high half.
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
      }
      switch (size) {
        case 0b00: {  // .16B (q=1) or .8B (q=0). PACKUSWB.
          if (is_uzp2) {
            builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{8});
          } else {
            builder_.Gen<x86_64::PsllwXRegImm>(xn.machine_reg(), int8_t{8});
            builder_.Gen<x86_64::PsrlwXRegImm>(xn.machine_reg(), int8_t{8});
          }
          if (q) {
            if (is_uzp2) {
              builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});
            } else {
              builder_.Gen<x86_64::PsllwXRegImm>(xm.machine_reg(), int8_t{8});
              builder_.Gen<x86_64::PsrlwXRegImm>(xm.machine_reg(), int8_t{8});
            }
            builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PackuswbXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          break;
        }
        case 0b01: {  // .8H (q=1) or .4H (q=0). PACKUSDW (SSE4.1).
          if (is_uzp2) {
            builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{16});
          } else {
            builder_.Gen<x86_64::PslldXRegImm>(xn.machine_reg(), int8_t{16});
            builder_.Gen<x86_64::PsrldXRegImm>(xn.machine_reg(), int8_t{16});
          }
          if (q) {
            if (is_uzp2) {
              builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{16});
            } else {
              builder_.Gen<x86_64::PslldXRegImm>(xm.machine_reg(), int8_t{16});
              builder_.Gen<x86_64::PsrldXRegImm>(xm.machine_reg(), int8_t{16});
            }
            builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PackusdwXRegXReg>(xn.machine_reg(), xn.machine_reg());
          }
          break;
        }
        case 0b10: {  // .4S (q=1) or .2S (q=0). SHUFPS.
          const int8_t imm =
              is_uzp2 ? static_cast<int8_t>(0xDD) : static_cast<int8_t>(0x88);
          if (q) {
            builder_.Gen<x86_64::ShufpsXRegXRegImm>(xn.machine_reg(), xm.machine_reg(), imm);
          } else {
            // xn already holds [vn_lo | vm_lo] from PUNPCKLQDQ above.
            builder_.Gen<x86_64::ShufpsXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), imm);
          }
          break;
        }
        case 0b11: {  // .2D (q=1 only; q=0 caught above).
          // UZP1.2D == ZIP1.2D, UZP2.2D == ZIP2.2D (2-lane coincidence).
          if (is_uzp2) {
            builder_.Gen<x86_64::PunpckhqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
        }
      }
    } else {  // is_trn
      const bool is_trn2 = (opcode == 0b110);
      switch (size) {
        case 0b00:
        case 0b01: {
          // .16B/.8B and .8H/.4H use PSHUFB + POR with precomputed per-byte
          // masks (0x80 mask byte renders as 0 in PSHUFB). mask_n picks the
          // wanted Vn bytes into even output positions; mask_m picks Vm bytes
          // into odd positions; POR combines. Q=0 forms reuse the Q=1 masks —
          // the shared Q=0 upper-zero (SetVRegFull) discards bytes 8..15.
          int64_t mask_n_lo, mask_n_hi, mask_m_lo, mask_m_hi;
          if (size == 0b00) {
            if (is_trn2) {
              mask_n_lo = static_cast<int64_t>(0x8007800580038001LL);
              mask_n_hi = static_cast<int64_t>(0x800F800D800B8009LL);
              mask_m_lo = static_cast<int64_t>(0x0780058003800180LL);
              mask_m_hi = static_cast<int64_t>(0x0F800D800B800980LL);
            } else {
              mask_n_lo = static_cast<int64_t>(0x8006800480028000LL);
              mask_n_hi = static_cast<int64_t>(0x800E800C800A8008LL);
              mask_m_lo = static_cast<int64_t>(0x0680048002800080LL);
              mask_m_hi = static_cast<int64_t>(0x0E800C800A800880LL);
            }
          } else {  // size == 0b01: halfword granularity.
            if (is_trn2) {
              mask_n_lo = static_cast<int64_t>(0x8080070680800302LL);
              mask_n_hi = static_cast<int64_t>(0x80800F0E80800B0ALL);
              mask_m_lo = static_cast<int64_t>(0x0706808003028080LL);
              mask_m_hi = static_cast<int64_t>(0x0F0E80800B0A8080LL);
            } else {
              mask_n_lo = static_cast<int64_t>(0x8080050480800100LL);
              mask_n_hi = static_cast<int64_t>(0x80800D0C80800908LL);
              mask_m_lo = static_cast<int64_t>(0x0504808001008080LL);
              mask_m_hi = static_cast<int64_t>(0x0D0C808009088080LL);
            }
          }
          FpRegister mask_n = AllocTempSimdReg();
          FpRegister mask_m = AllocTempSimdReg();
          builder_.Gen<x86_64::MovqXRegReg>(mask_n.machine_reg(),
                                            GetImm(static_cast<uint64_t>(mask_n_lo)));
          builder_.Gen<x86_64::PinsrqXRegRegImm>(mask_n.machine_reg(),
                                                 GetImm(static_cast<uint64_t>(mask_n_hi)), int8_t{1});
          builder_.Gen<x86_64::MovqXRegReg>(mask_m.machine_reg(),
                                            GetImm(static_cast<uint64_t>(mask_m_lo)));
          builder_.Gen<x86_64::PinsrqXRegRegImm>(mask_m.machine_reg(),
                                                 GetImm(static_cast<uint64_t>(mask_m_hi)), int8_t{1});
          builder_.Gen<x86_64::PshufbXRegXReg>(xn.machine_reg(), mask_n.machine_reg());
          builder_.Gen<x86_64::PshufbXRegXReg>(xm.machine_reg(), mask_m.machine_reg());
          builder_.Gen<x86_64::PorXRegXReg>(xn.machine_reg(), xm.machine_reg());
          break;
        }
        case 0b10: {  // .4S (q=1) or .2S (q=0).
          if (!q) {
            // TRN1.2S == ZIP1.2S = [s1[0], s2[0]]; TRN2.2S == ZIP2.2S =
            // [s1[1], s2[1]] (PSRLDQ-4 prelude pulls lane 1 down to 0).
            if (is_trn2) {
              builder_.Gen<x86_64::PsrldqXRegImm>(xn.machine_reg(), int8_t{4});
              builder_.Gen<x86_64::PsrldqXRegImm>(xm.machine_reg(), int8_t{4});
            }
            builder_.Gen<x86_64::PunpckldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            // PSHUFD imm 0x88 (even) / 0xDD (odd) collapses each source's
            // wanted dwords into both halves; PUNPCKLDQ interleaves the lows:
            //   res = [s1[0], s2[0], s1[2], s2[2]] = TRN1.4S (0x88), etc.
            const int8_t imm = is_trn2 ? static_cast<int8_t>(0xDD)
                                       : static_cast<int8_t>(0x88);
            builder_.Gen<x86_64::PshufdXRegXRegImm>(xn.machine_reg(), xn.machine_reg(), imm);
            builder_.Gen<x86_64::PshufdXRegXRegImm>(xm.machine_reg(), xm.machine_reg(), imm);
            builder_.Gen<x86_64::PunpckldqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
        }
        case 0b11: {  // .2D (q=1 only; q=0 caught above).
          // TRN1.2D == ZIP1.2D, TRN2.2D == ZIP2.2D (2-lane coincidence).
          if (is_trn2) {
            builder_.Gen<x86_64::PunpckhqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          } else {
            builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xn.machine_reg(), xm.machine_reg());
          }
          break;
        }
      }
    }

    SetVRegFull(rd, xn, q);
  }

  void AdvSimdTableLookup(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t len, uint8_t op, bool q) {
    if (!success()) {
      return;
    }
    // TBL/TBX: per-byte gather across 1..4 consecutive table registers, mirroring
    // the lite lowering (register-domain-pure, no BB-splits). For each table reg r,
    // shift the index vector down by r*16 (bytewise) so PSHUFB picks lane (idx-r*16)
    // when it is in [0,15]; an in-range mask built via PSUBUSB/PCMPEQB gates the
    // result and forms the running "any table hit" mask. TBL zeroes misses; TBX
    // blends the original Vd back into the miss lanes.
    const uint8_t table_regs = static_cast<uint8_t>(len + 1);  // 1..4
    const int32_t vm_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + rm * 16);
    const int32_t vd_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + rd * 16);

    FpRegister xmm_idx        = AllocTempSimdReg();
    FpRegister xmm_acc        = AllocTempSimdReg();
    FpRegister xmm_in_range   = AllocTempSimdReg();
    FpRegister xmm_const15    = AllocTempSimdReg();
    FpRegister xmm_zero       = AllocTempSimdReg();
    FpRegister xmm_tmp_idx    = AllocTempSimdReg();
    FpRegister xmm_tmp_mask   = AllocTempSimdReg();
    FpRegister xmm_tmp_lookup = AllocTempSimdReg();

    // Load the index vector once. Vm is the per-byte selector.
    builder_.GenGetSimd<16>(xmm_idx.machine_reg(), vm_off);
    // acc = 0, in_range = 0, zero = 0. MOVQ from a zero GP reg zero-extends to a
    // full 128-bit zero and is a proper full-def (a self-PXOR would read an
    // undefined vreg and trip the lifetime analysis in this tier).
    builder_.Gen<x86_64::MovqXRegReg>(xmm_acc.machine_reg(), GetImm(uint64_t{0}));
    builder_.Gen<x86_64::MovqXRegReg>(xmm_in_range.machine_reg(), GetImm(uint64_t{0}));
    builder_.Gen<x86_64::MovqXRegReg>(xmm_zero.machine_reg(), GetImm(uint64_t{0}));
    // const15 = byte-broadcast(0x0F).
    builder_.Gen<x86_64::MovqXRegReg>(xmm_const15.machine_reg(),
                                      GetImm(uint64_t{0x0F0F0F0F0F0F0F0FULL}));
    builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmm_const15.machine_reg(), xmm_const15.machine_reg());

    for (uint8_t r = 0; r < table_regs; ++r) {
      const int32_t vn_off_r =
          static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + ((rn + r) & 31) * 16);

      // shifted_idx_r = Vm - r*16 (bytewise wrap). For r=0 this is just Vm.
      builder_.Gen<x86_64::MovdqaXRegXReg>(xmm_tmp_idx.machine_reg(), xmm_idx.machine_reg());
      if (r != 0) {
        const uint64_t broadcast =
            uint64_t{0x0101010101010101ULL} * static_cast<uint64_t>(r * 16);
        builder_.Gen<x86_64::MovqXRegReg>(xmm_tmp_mask.machine_reg(), GetImm(broadcast));
        builder_.Gen<x86_64::PunpcklqdqXRegXReg>(xmm_tmp_mask.machine_reg(),
                                                 xmm_tmp_mask.machine_reg());
        builder_.Gen<x86_64::PsubbXRegXReg>(xmm_tmp_idx.machine_reg(), xmm_tmp_mask.machine_reg());
      }

      // in_range_r = PCMPEQB(PSUBUSB(shifted_idx_r, 15), 0).
      builder_.Gen<x86_64::MovdqaXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_tmp_idx.machine_reg());
      builder_.Gen<x86_64::PsubusbXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_const15.machine_reg());
      builder_.Gen<x86_64::PcmpeqbXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_zero.machine_reg());

      // looked_up_r = PSHUFB(V[(rn+r)%32], shifted_idx_r), masked by in_range.
      builder_.GenGetSimd<16>(xmm_tmp_lookup.machine_reg(), vn_off_r);
      builder_.Gen<x86_64::PshufbXRegXReg>(xmm_tmp_lookup.machine_reg(), xmm_tmp_idx.machine_reg());
      builder_.Gen<x86_64::PandXRegXReg>(xmm_tmp_lookup.machine_reg(), xmm_tmp_mask.machine_reg());

      builder_.Gen<x86_64::PorXRegXReg>(xmm_acc.machine_reg(), xmm_tmp_lookup.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xmm_in_range.machine_reg(), xmm_tmp_mask.machine_reg());
    }

    if (op /* TBX */) {
      // result = acc | (Vd & ~in_range_all).
      builder_.GenGetSimd<16>(xmm_tmp_lookup.machine_reg(), vd_off);
      // xmm_tmp_mask := ~in_range_all & Vd
      builder_.Gen<x86_64::MovdqaXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_in_range.machine_reg());
      builder_.Gen<x86_64::PandnXRegXReg>(xmm_tmp_mask.machine_reg(), xmm_tmp_lookup.machine_reg());
      builder_.Gen<x86_64::PorXRegXReg>(xmm_acc.machine_reg(), xmm_tmp_mask.machine_reg());
    }

    // SetVRegFull zeroes Vd[127:64] when q=false (D-register semantics).
    SetVRegFull(rd, xmm_acc, q);
  }

  // AdvSIMD load/store multiple structures. Wave 1 covers the NON-interleaved
  // contiguous form (LD1/ST1 with 1..4 registers): a plain bulk transfer of
  // num_regs consecutive 16-byte (Q=1) or 8-byte (Q=0) vectors between guest
  // memory [Xn{, offset}] and V[rt .. rt+num_regs-1], with optional post-index
  // writeback (immediate = num_regs*vec_bytes, or register Xm). Each memory
  // access is the unaligned MOVDQU/MOVSD form wrapped in a recovery block so a
  // host fault reaches the guest signal handler; the 16-byte-aligned V[] slots
  // use GenGetSimd/GenSetSimd (MOVDQA). The Q=0 load uses MOVSD reg<-mem, which
  // zero-extends the upper 64 bits (D-register semantics). The interleaving
  // LD2/LD3/LD4 / ST2/ST3/ST4 de-interleave forms still bail to the lite tier
  // (their element-wise PINSR/PEXTR-from-memory has no heavy LIR op yet).
  // Mirrors lite_translator.h::AdvSimdMultiStruct (both the interleaved
  // LDn/STn de-/interleave path and the non-interleaved contiguous LD1/ST1).
  void AdvSimdMultiStruct(uint8_t rt,
                          uint8_t rn,
                          uint8_t num_regs,
                          uint8_t size,
                          bool q,
                          bool is_store,
                          bool postindex,
                          uint8_t rm,
                          bool is_interleaved) {
    if (!success()) {
      return;
    }
    const int32_t vec_bytes = q ? 16 : 8;

    // De-interleaving LD2/LD3/LD4 / interleaving ST2/ST3/ST4 (multiple-structure
    // form). Mirrors the lite translator's element-wise lowering: memory holds
    // num_regs * (vec_bytes/esize) elements laid out structure-major — element e
    // in memory belongs to register (e % num_regs), lane (e / num_regs). Because
    // the heavy tier has NO memory-operand PINSR/PEXTR (only the register forms),
    // each element is routed through a GP temp:
    //   load : MOV{zx,}* mem->gp (+recovery) then PINSR gp->lane; the destination
    //          register starts from a zeroed XMM so Q=0 zeroes the upper 64 bits.
    //   store: PEXTR lane->gp then MOV* gp->mem (+recovery).
    // The v[] register accesses stay full-width (GenGetSimd<16>/GenSetSimd<16>) so
    // the optimizer's 16-byte store/load forwarding on a v[] slot is never split
    // by a narrow sub-lane access (see the UMOV/INS-element note above); only the
    // guest-memory side is narrow. Exact for every (num_regs, esize, Q) combo.
    if (is_interleaved) {
      if (num_regs < 2 || num_regs > 4) {
        UndefinedReturningVoid();
        return;
      }
      if (size > 3) {
        UndefinedReturningVoid();
        return;
      }
      const int esize = 1 << size;
      const int num_lanes = vec_bytes / esize;

      Register ibase_orig = (rn == 31) ? GetSp() : GetReg(rn);
      Register ibase = ApplyTbi(ibase_orig);

      for (uint8_t r = 0; r < num_regs; r++) {
        const uint8_t vreg = (rt + r) & 31;
        const int32_t vt_off =
            static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + vreg * 16);
        if (is_store) {
          FpRegister xmm = AllocTempSimdReg();
          builder_.GenGetSimd<16>(xmm.machine_reg(), vt_off);
          for (int l = 0; l < num_lanes; l++) {
            const int32_t mem_off = static_cast<int32_t>((l * num_regs + r) * esize);
            const int8_t lane = static_cast<int8_t>(l);
            Register elem;
            switch (esize) {
              case 1:
                elem = std::get<0>(Gen<x86_64::PextrbRegXRegImm>(xmm.machine_reg(), lane));
                Gen<x86_64::MovbOpReg>({.base = ibase, .disp = mem_off}, elem);
                break;
              case 2:
                elem = std::get<0>(Gen<x86_64::PextrwRegXRegImm>(xmm.machine_reg(), lane));
                Gen<x86_64::MovwOpReg>({.base = ibase, .disp = mem_off}, elem);
                break;
              case 4:
                elem = std::get<0>(Gen<x86_64::PextrdRegXRegImm>(xmm.machine_reg(), lane));
                Gen<x86_64::MovlOpReg>({.base = ibase, .disp = mem_off}, elem);
                break;
              default:  // esize == 8
                elem = std::get<0>(Gen<x86_64::PextrqRegXRegImm>(xmm.machine_reg(), lane));
                Gen<x86_64::MovqOpReg>({.base = ibase, .disp = mem_off}, elem);
                break;
            }
            GenRecoveryBlockForLastInsn();
          }
        } else {
          FpRegister xmm = AllocZeroedSimdReg();
          for (int l = 0; l < num_lanes; l++) {
            const int32_t mem_off = static_cast<int32_t>((l * num_regs + r) * esize);
            const int8_t lane = static_cast<int8_t>(l);
            Register elem;
            switch (esize) {
              case 1:
                elem = std::get<0>(Gen<x86_64::MovzxblRegOp>({.base = ibase, .disp = mem_off}));
                GenRecoveryBlockForLastInsn();
                builder_.Gen<x86_64::PinsrbXRegRegImm>(xmm.machine_reg(), elem, lane);
                break;
              case 2:
                elem = std::get<0>(Gen<x86_64::MovzxwlRegOp>({.base = ibase, .disp = mem_off}));
                GenRecoveryBlockForLastInsn();
                builder_.Gen<x86_64::PinsrwXRegRegImm>(xmm.machine_reg(), elem, lane);
                break;
              case 4:
                elem = std::get<0>(Gen<x86_64::MovlRegOp>({.base = ibase, .disp = mem_off}));
                GenRecoveryBlockForLastInsn();
                builder_.Gen<x86_64::PinsrdXRegRegImm>(xmm.machine_reg(), elem, lane);
                break;
              default:  // esize == 8
                elem = std::get<0>(Gen<x86_64::MovqRegOp>({.base = ibase, .disp = mem_off}));
                GenRecoveryBlockForLastInsn();
                builder_.Gen<x86_64::PinsrqXRegRegImm>(xmm.machine_reg(), elem, lane);
                break;
            }
          }
          builder_.GenSetSimd<16>(vt_off, xmm.machine_reg());
        }
      }

      if (postindex) {
        // Writeback preserves the original (un-TBI-masked) top byte, so re-read
        // the base register rather than reusing the masked access address.
        Register reread_base = (rn == 31) ? GetSp() : GetReg(rn);
        Register new_base = Copy(reread_base);
        if (rm == 31) {
          new_base = std::get<0>(Gen<x86_64::AddqRegImm, kNoSSA>(
              new_base, static_cast<int32_t>(num_regs) * vec_bytes));
        } else {
          Register rm_val = GetReg(rm);
          new_base = std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(new_base, rm_val));
        }
        if (rn == 31) {
          SetSp(new_base);
        } else {
          SetReg(rn, new_base);
        }
      }
      return;
    }

    if (num_regs < 1 || num_regs > 4) {
      UndefinedReturningVoid();
      return;
    }

    Register base_orig = (rn == 31) ? GetSp() : GetReg(rn);
    Register base = ApplyTbi(base_orig);

    for (uint8_t r = 0; r < num_regs; r++) {
      const uint8_t vreg = (rt + r) & 31;
      const int32_t vt_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + vreg * 16);
      const int32_t mem_off = static_cast<int32_t>(r) * vec_bytes;
      if (is_store) {
        FpRegister xmm = AllocTempSimdReg();
        builder_.GenGetSimd<16>(xmm.machine_reg(), vt_off);
        if (q) {
          builder_.Gen<x86_64::MovdquOpXReg>({.base = base, .disp = mem_off}, xmm.machine_reg());
        } else {
          // Q=0: store the low 64 bits of V[vreg].
          builder_.Gen<x86_64::MovsdOpXReg>({.base = base, .disp = mem_off}, xmm.machine_reg());
        }
        GenRecoveryBlockForLastInsn();
      } else {
        FpRegister xmm =
            q ? FpRegister{std::get<0>(
                    Gen<x86_64::MovdquXRegOp>({.base = base, .disp = mem_off}))}
              // Q=0: MOVSD reg<-mem loads 8 bytes and zero-extends the upper 64.
              : FpRegister{std::get<0>(
                    Gen<x86_64::MovsdXRegOp>({.base = base, .disp = mem_off}))};
        GenRecoveryBlockForLastInsn();
        builder_.GenSetSimd<16>(vt_off, xmm.machine_reg());
      }
    }

    if (postindex) {
      // Writeback preserves the original (un-TBI-masked) top byte, so re-read the
      // base register rather than reusing the masked address used for the access.
      Register reread_base = (rn == 31) ? GetSp() : GetReg(rn);
      Register new_base = Copy(reread_base);
      if (rm == 31) {
        // Immediate post-index: total bytes accessed = num_regs * vec_bytes.
        new_base = std::get<0>(
            Gen<x86_64::AddqRegImm, kNoSSA>(new_base, static_cast<int32_t>(num_regs) * vec_bytes));
      } else {
        Register rm_val = GetReg(rm);
        new_base = std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(new_base, rm_val));
      }
      if (rn == 31) {
        SetSp(new_base);
      } else {
        SetReg(rn, new_base);
      }
    }
  }

  //
  // Crypto.
  //

  void Sha512(Decoder::Sha512Op op, uint8_t rd, uint8_t rn, uint8_t rm) {
    UndefinedReturningVoid();
    UNUSED_ARGS(op, rd, rn, rm);
  }

  void Eor3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, ra);
  }

  void Bcax(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, ra);
  }

  void Rax1(uint8_t rd, uint8_t rn, uint8_t rm) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm);
  }

  void Xar(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm6) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, imm6);
  }

  void Sm4e(uint8_t rd, uint8_t rn) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn);
  }

  void Sm4ekey(uint8_t rd, uint8_t rn, uint8_t rm) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm);
  }

  void Sm3ss1(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, ra);
  }

  void Sm3tt(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm2, uint8_t op) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, imm2, op);
  }

  void Sm3partw1(uint8_t rd, uint8_t rn, uint8_t rm) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm);
  }

  void Sm3partw2(uint8_t rd, uint8_t rn, uint8_t rm) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm);
  }

  void CryptoAes(uint8_t rd, uint8_t rn, uint8_t opcode) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, opcode);
  }

  void CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, opcode);
  }

  void CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, opcode);
  }

  //
  // Region machinery (not part of the SemanticsListener interface).
  //

  [[nodiscard]] bool IsRegionEndReached() const;
  void StartInsn();
  void Finalize(GuestAddr stop_pc);

  // Exported only for testing.
  [[nodiscard]] bool has_in_region_backedge() const { return has_in_region_backedge_; }

  [[nodiscard]] const ArenaMap<GuestAddr, MachineInsnPosition>& branch_targets() const {
    return branch_targets_;
  }

 private:
  // ThreadState offsets (computed directly; arm64 guest_state has no
  // GetThreadStateRegOffset helper like riscv64).
  static int32_t GetThreadStateRegOffset(uint8_t reg);
  static int32_t GetThreadStateSpOffset();

  // Syntax sugar mirroring riscv64's Gen<> adapter. It threads guest temp
  // registers through PseudoCopy for SSA form and dispatches to the
  // MachineIRBuilder.
  enum SSAMode { kSSA, kNoSSA };

  template <typename InsnType, enum SSAMode kSSAMode = kSSA, typename... Args>
  auto Gen(Args... args)
      -> std::enable_if_t<(std::is_same_v<std::remove_cvref_t<Args>, MachineReg> + ... + 0) ==
                              InsnType::kInfo.InputRegistersCount(),
                          std::array<MachineReg, InsnType::kInfo.OutputRegistersCount()>> {
    std::array<MachineReg, InsnType::kInfo.InputRegistersCount()> input;
    int input_index = 0;
    (
        [&input, &input_index]<typename Arg>(Arg arg) {
          if constexpr (std::is_same_v<std::remove_cvref_t<Arg>, MachineReg>) {
            input[input_index++] = arg;
          }
        }(args),
        ...);
    std::array<MachineReg, InsnType::kInfo.OutputRegistersCount()> output;
    input_index = 0;
    int output_index = 0;
    std::array<MachineReg, InsnType::kInfo.num_reg_operands> gen_args;
    for (int index = 0; index < InsnType::kInfo.num_reg_operands; index++) {
      if (InsnType::kInfo.reg_kinds[index].IsDef()) {
        if (InsnType::kInfo.reg_kinds[index].RegClass() == &x86_64::kFLAGS) {
          output[output_index] = GetFlagsRegister();
        } else {
          if (!InsnType::kInfo.reg_kinds[index].IsInput()) {
            output[output_index] = AllocTempReg();
          } else if (kSSAMode == kSSA) {
            output[output_index] = AllocTempReg();
            if (InsnType::kInfo.reg_kinds[index].IsInput()) {
              builder_.Gen<PseudoCopy>(output[output_index],
                                       input[input_index++],
                                       InsnType::kInfo.reg_kinds[index].RegClass()->reg_size);
            }
          } else {
            output[output_index] = input[input_index++];
          }
        }
        gen_args[index] = output[output_index++];
      } else {
        CHECK(InsnType::kInfo.reg_kinds[index].IsInput());
        if (kSSAMode == kSSA && InsnType::kInfo.reg_kinds[index].RegClass()->num_regs == 1) {
          CHECK(InsnType::kInfo.reg_kinds[index].RegClass() != &x86_64::kFLAGS);
          gen_args[index] = AllocTempReg();
          builder_.Gen<PseudoCopy>(gen_args[index],
                                   input[input_index++],
                                   InsnType::kInfo.reg_kinds[index].RegClass()->reg_size);
        } else {
          gen_args[index] = input[input_index++];
        }
      }
    }
    std::apply(
        InsnType::template kGenAutoFunc<x86_64::MachineIRBuilder>,
        std::tuple_cat(
            std::tuple<x86_64::MachineIRBuilder&>{builder_}, gen_args, []<typename Arg>(Arg arg) {
              if constexpr (std::is_same_v<std::remove_cvref_t<Arg>, MachineReg>) {
                return std::tuple{};
              } else {
                return std::tuple{arg};
              }
            }(args)...));
    return output;
  }

  BERBERIS_DECLARE_MACHINE_INSN_ADAPTER(
      /*may_discard*/ auto Gen,
      (, enum SSAMode kSSAMode = kSSA),
      MachineInsn,
      InputArgsTuple,
      typename x86_64::MachineInsn<
          typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>::OutputArgsTuple,
      Gen,
      (, kSSAMode))

  // Materialize ARM64 NZCV into ThreadState.cpu.flags from the host EFLAGS that
  // a preceding x86 ALU op left in `flags_vreg`. Bit-exact with
  // lite_translator.h::EmitStoreArmNZCV:
  //   PseudoReadFlags (LAHF + SETO) -> raw has N@15, Z@14, C@8, V@0
  //   AND 0xC101                    -> keep only N, Z, C, V
  //   if is_sub: XOR 0x0100         -> ARM borrow is inverted (ARM C = !x86 CF)
  //   MOVW [rbp + cpu.flags], raw   -> 16-bit store of the packed NZCV
  // cpu.flags is a uint16_t and the 2 bytes after it are alignment padding
  // before cpu.cached_fpcr, but the 16-bit store mirrors lite exactly and never
  // touches a neighbouring field.
  void EmitMaterializeNZCV(Register flags_vreg, bool is_sub) {
    Register raw = AllocTempReg();
    builder_.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, raw, flags_vreg);
    raw = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(raw, static_cast<int32_t>(0xC101)));
    if (is_sub) {
      raw = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(raw, static_cast<int32_t>(0x0100)));
    }
    builder_.Gen<x86_64::MovwOpReg>(
        {.base = x86_64::kMachineRegRBP,
         .disp = static_cast<int32_t>(offsetof(ThreadState, cpu.flags))},
        raw);
  }

  // ARM64 TBI (Top Byte Ignore): clear the top 8 bits of an address register
  // before using it as a host x86 memory operand. ARM64 ignores the top byte of
  // pointers in load/store; x86 does not, so we mask it ourselves. Returns a
  // fresh register holding (base & 0x00FF'FFFF'FFFF'FFFF). Mirrors
  // lite_translator.h::ApplyTbi (movq; shlq 8; shrq 8).
  [[nodiscard]] Register ApplyTbi(Register base) {
    Register tbi = Copy(base);
    tbi = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(tbi, int8_t{8}));
    tbi = std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(tbi, int8_t{8}));
    return tbi;
  }

  // Compute the address for a register-offset load/store:
  //   base + extend(offset_reg) << shift_amount
  // The extend applies the correct 32->64 widening (UXTW zero-extends, SXTW
  // sign-extends, LSL/UXTX/SXTX use the full 64-bit value), mirroring
  // lite_translator.h::ApplyOffsetExtend. Returns a fresh register; the base is
  // not modified (so an aliased base/dest stays correct).
  [[nodiscard]] Register EmitRegOffsetAddr(Register base,
                                           Register offset_reg,
                                           uint8_t extend_type,
                                           uint8_t shift_amount) {
    Register addr;
    switch (extend_type) {
      case 0b010:  // UXTW: zero-extend the low 32 bits (a 32-bit mov zero-extends).
        addr = std::get<0>(Gen<x86_64::MovlRegReg>(offset_reg));
        break;
      case 0b110:  // SXTW: sign-extend the low 32 bits to 64.
        addr = std::get<0>(Gen<x86_64::MovsxlqRegReg>(offset_reg));
        break;
      case 0b011:  // LSL / UXTX: full 64-bit value, no extension.
      case 0b111:  // SXTX: full 64-bit value, sign-extend is a no-op here.
      default:
        addr = Copy(offset_reg);
        break;
    }
    if (shift_amount != 0) {
      addr = std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(addr, static_cast<int8_t>(shift_amount)));
    }
    return std::get<0>(Gen<x86_64::AddqRegReg, kNoSSA>(addr, base));
  }

  // Emit `src` shifted by a constant amount (LSL/LSR/ASR/ROR) into a fresh
  // register, mirroring lite_translator.h::EmitShift. A 32-bit shift uses the
  // l-suffix forms (which zero-extend the result, matching ARM64 W-write
  // semantics). When shift_amount == 0 this is just a width-correct copy.
  [[nodiscard]] Register EmitShiftImm(Register src,
                                      Decoder::ShiftType shift_type,
                                      uint8_t shift_amount,
                                      bool is_64bit) {
    if (is_64bit) {
      Register dst = Copy(src);
      if (shift_amount == 0) {
        return dst;
      }
      int8_t amt = static_cast<int8_t>(shift_amount);
      switch (shift_type) {
        case Decoder::ShiftType::kLsl:
          return std::get<0>(Gen<x86_64::ShlqRegImm, kNoSSA>(dst, amt));
        case Decoder::ShiftType::kLsr:
          return std::get<0>(Gen<x86_64::ShrqRegImm, kNoSSA>(dst, amt));
        case Decoder::ShiftType::kAsr:
          return std::get<0>(Gen<x86_64::SarqRegImm, kNoSSA>(dst, amt));
        case Decoder::ShiftType::kRor:
          return std::get<0>(Gen<x86_64::RorqRegImm, kNoSSA>(dst, amt));
      }
      return dst;
    }
    Register dst = std::get<0>(Gen<x86_64::MovlRegReg>(src));
    if (shift_amount == 0) {
      return dst;
    }
    int8_t amt = static_cast<int8_t>(shift_amount);
    switch (shift_type) {
      case Decoder::ShiftType::kLsl:
        return std::get<0>(Gen<x86_64::ShllRegImm, kNoSSA>(dst, amt));
      case Decoder::ShiftType::kLsr:
        return std::get<0>(Gen<x86_64::ShrlRegImm, kNoSSA>(dst, amt));
      case Decoder::ShiftType::kAsr:
        return std::get<0>(Gen<x86_64::SarlRegImm, kNoSSA>(dst, amt));
      case Decoder::ShiftType::kRor:
        return std::get<0>(Gen<x86_64::RorlRegImm, kNoSSA>(dst, amt));
    }
    return dst;
  }

  //
  // Scalar floating-point helpers.
  //

  // Load the low scalar of guest V[reg] into a fresh SimdReg. MOVSS loads the
  // low 4 bytes (S) and MOVSD the low 8 bytes (D), each zeroing the rest of the
  // host XMM, so the value sits in lane 0 ready for an SSE op.
  //
  // A MOVSD (8-byte) load is used for BOTH S and D: it is one of the SIMD
  // opcodes RemoveLocalGuestContextAccesses recognizes as a guest-context GET
  // (MOVSS is not), so a prior 16-byte MOVDQA store to the same v[reg] forwards
  // correctly through the local optimizer instead of being dead-eliminated and
  // leaving a stale memory read. For S the extra 4 bytes loaded are harmless:
  // the scalar SSE op (ADDSS/MULSS/...) operates on lane 0 only.
  [[nodiscard]] FpRegister GetVRegScalar(uint8_t reg, bool /*is_double*/) {
    const int32_t off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + reg * 16);
    return FpRegister{
        std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = x86_64::kMachineRegRBP, .disp = off}))};
  }

  // Allocate a freshly-zeroed XMM. PXOR is dependency-breaking (zeroes
  // regardless of prior contents), but its operand is use_def, so a PseudoDefReg
  // first gives the vreg a lifetime for the data-flow analysis (mirrors the
  // riscv64 frontend's self-XOR zeroing idiom).
  [[nodiscard]] FpRegister AllocZeroedSimdReg() {
    FpRegister zero = AllocTempSimdReg();
    builder_.Gen<PseudoDefReg>(zero.machine_reg());
    builder_.Gen<x86_64::PxorXRegXReg>(zero.machine_reg(), zero.machine_reg());
    return zero;
  }

  // Write the scalar `value` (in lane 0) to guest V[reg] and ZERO the upper
  // bytes, matching ARM scalar-FP write semantics (a later vector read of the
  // same register must see a clean zero-extended value).
  //
  // The whole 16-byte slot is written with ONE aligned MOVDQA store of a
  // fully-formed XMM (lane 0 = the scalar, lanes above zero). A single store —
  // rather than a zero-store followed by a partial MOVSS/MOVSD — is required
  // because RemoveLocalGuestContextAccesses keys dead-store elimination on the
  // store displacement alone (not its width): a later MOVSD to the same v[reg]
  // offset would otherwise erase a preceding 16-byte zero-store and leave the
  // upper 8 bytes stale. We merge `value`'s lane 0 into a zeroed XMM (MOVSS/
  // MOVSD reg-reg preserve the dest's upper lanes), then store it once.
  void SetVRegScalar(uint8_t reg, FpRegister value, bool is_double) {
    if (!success()) {
      return;
    }
    const int32_t off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + reg * 16);
    FpRegister merged = AllocZeroedSimdReg();
    if (is_double) {
      builder_.Gen<x86_64::MovsdXRegXReg>(merged.machine_reg(), value.machine_reg());
    } else {
      builder_.Gen<x86_64::MovssXRegXReg>(merged.machine_reg(), value.machine_reg());
    }
    builder_.Gen<x86_64::MovdqaOpXReg>({.base = x86_64::kMachineRegRBP, .disp = off},
                                       merged.machine_reg());
  }

  // Write a scalar value held in a GP register (low 4 bytes for S, low 8 for D)
  // to guest V[reg], zeroing the upper bytes. Same single-MOVDQA-store rationale
  // as SetVRegScalar: move the GP value into lane 0 of a zeroed XMM, store once.
  void SetVRegScalarFromGp(uint8_t reg, Register value, bool is_double) {
    if (!success()) {
      return;
    }
    const int32_t off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + reg * 16);
    FpRegister merged = AllocZeroedSimdReg();
    if (is_double) {
      // MOVQ xmm, r64 zero-extends into the XMM (upper 64 cleared).
      builder_.Gen<x86_64::MovqXRegReg>(merged.machine_reg(), value);
    } else {
      // MOVD xmm, r32 zero-extends into the XMM (upper 96 cleared).
      builder_.Gen<x86_64::MovdXRegReg>(merged.machine_reg(), value);
    }
    builder_.Gen<x86_64::MovdqaOpXReg>({.base = x86_64::kMachineRegRBP, .disp = off},
                                       merged.machine_reg());
  }

  // Write a full or D-form vector result to guest V[reg]. For the Q-form
  // (q == true) all 128 bits of `value` are committed. For the D-form
  // (q == false) the upper 64 bits MUST be zeroed (ARM64 writes of a 64-bit
  // vector clear the rest of the register), so the low 64 bits of `value` are
  // merged (MOVSD reg-reg, which copies the low 8 bytes and preserves the
  // destination's upper lanes) into a PXOR-zeroed XMM before the store. Both
  // paths use ONE 16-byte MOVDQA store at the v[reg] displacement, matching the
  // single-store discipline of SetVRegScalar so RemoveLocalGuestContextAccesses
  // forwards and dead-store-eliminates correctly (it keys on the store
  // displacement, not its width).
  void SetVRegFull(uint8_t reg, FpRegister value, bool q) {
    if (!success()) {
      return;
    }
    const int32_t off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + reg * 16);
    if (q) {
      builder_.GenSetSimd<16>(off, value.machine_reg());
      return;
    }
    FpRegister merged = AllocZeroedSimdReg();
    builder_.Gen<x86_64::MovsdXRegXReg>(merged.machine_reg(), value.machine_reg());
    builder_.GenSetSimd<16>(off, merged.machine_reg());
  }

  // Commit a narrowing-op result: the narrowed lanes occupy the low 64 bits of
  // `narrowed`. Q=0 stores them to Vd.low with the upper 64 zeroed; Q=1 (the
  // "2" form) shifts them into Vd.high (PSLLDQ by 8) while preserving Vd's
  // existing low 64 (masked via MOVSD into a zeroed reg, then POR). Mirrors the
  // Q=0/Q2 store discipline of lite_translator.h's XTN/SQXTN family.
  void SetVRegNarrow(uint8_t rd, FpRegister narrowed, bool q) {
    if (!success()) {
      return;
    }
    if (!q) {
      SetVRegFull(rd, narrowed, /*q=*/false);
      return;
    }
    const int32_t vd_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + rd * 16);
    builder_.Gen<x86_64::PslldqXRegImm>(narrowed.machine_reg(), int8_t{8});
    FpRegister xd = AllocTempSimdReg();
    FpRegister xd_low = AllocZeroedSimdReg();
    builder_.GenGetSimd<16>(xd.machine_reg(), vd_off);
    builder_.Gen<x86_64::MovsdXRegXReg>(xd_low.machine_reg(), xd.machine_reg());
    builder_.Gen<x86_64::PorXRegXReg>(narrowed.machine_reg(), xd_low.machine_reg());
    SetVRegFull(rd, narrowed, /*q=*/true);
  }

  // Lower a scalar FP binary op through the guest-agnostic intrinsic layer.
  // kFunction is intrinsics::FAdd/FSub/FMul/FDiv<FloatN>; passing rm=DYN makes
  // InlineIntrinsic forward to the FXxxHostRounding variant bound to the SSE op
  // in machine_ir_intrinsic_binding.json. frm is unused on the host-rounding
  // path; a dummy temp register satisfies the signature.
  template <auto kFunction>
  void EmitFpBinop(FpRegister result, FpRegister src1, FpRegister src2) {
    if (!success()) {
      return;
    }
    Register frm = AllocTempReg();
    builder_.Gen<PseudoDefReg>(frm);
    InlineIntrinsicForHeavyOptimizer<kFunction>(
        &builder_, result, GetFlagsRegister(), int8_t{0b111}, frm, src1, src2);
  }

  // VFPExpandImm (ARM ARM, FMOV scalar/vector immediate). The FP constant is a
  // pure function of imm8, computed at translation time. Mirrors
  // lite_translator.h::VFPExpandImm32Jit / VFPExpandImm64Jit.
  static uint32_t VFPExpandImm32(uint8_t imm8) {
    uint32_t sign = (imm8 >> 7) & 1;
    uint32_t b = (imm8 >> 6) & 1;
    uint32_t exp = ((1 - b) << 7) | ((b ? 0x1Fu : 0u) << 2) | ((imm8 >> 4) & 0x3u);
    uint32_t mantissa = static_cast<uint32_t>(imm8 & 0xFu) << 19;
    return (sign << 31) | (exp << 23) | mantissa;
  }
  static uint64_t VFPExpandImm64(uint8_t imm8) {
    uint64_t sign = (imm8 >> 7) & 1;
    uint64_t b = (imm8 >> 6) & 1;
    uint64_t exp = ((1 - b) << 10) | ((b ? 0xFFull : 0ull) << 2) | ((imm8 >> 4) & 0x3ull);
    uint64_t mantissa = static_cast<uint64_t>(imm8 & 0xFull) << 48;
    return (sign << 63) | (exp << 52) | mantissa;
  }

  // AdvSIMDExpandImm (ARM ARM): the 128-bit MOVI/MVNI/FMOV(vector) immediate as
  // a pure function of (op, cmode, abc, defgh, q). Identical to
  // lite_translator.h::ExpandSimdModifiedImmJit; uses the VFPExpandImm helpers
  // above for the FMOV (cmode=1111) forms.
  static __uint128_t ExpandSimdModifiedImmJit(uint8_t op, uint8_t cmode, uint8_t abc,
                                              uint8_t defgh, bool q) {
    uint8_t imm8 = (abc << 5) | defgh;
    uint64_t imm64 = 0;
    if (op == 0) {
      switch (cmode >> 1) {
        case 0b000: imm64 = uint64_t{imm8} | (uint64_t{imm8} << 32); break;
        case 0b001: imm64 = (uint64_t{imm8} << 8) | (uint64_t{imm8} << 40); break;
        case 0b010: imm64 = (uint64_t{imm8} << 16) | (uint64_t{imm8} << 48); break;
        case 0b011: imm64 = (uint64_t{imm8} << 24) | (uint64_t{imm8} << 56); break;
        case 0b100: for (int i = 0; i < 4; i++) imm64 |= uint64_t{imm8} << (i * 16); break;
        case 0b101: for (int i = 0; i < 4; i++) imm64 |= uint64_t{imm8} << (i * 16 + 8); break;
        case 0b110:
          if (!(cmode & 1)) {
            uint32_t v = (uint32_t{imm8} << 8) | 0xFF;
            imm64 = uint64_t{v} | (uint64_t{v} << 32);
          } else {
            uint32_t v = (uint32_t{imm8} << 16) | 0xFFFF;
            imm64 = uint64_t{v} | (uint64_t{v} << 32);
          }
          break;
        case 0b111:
          if (!(cmode & 1)) {
            for (int i = 0; i < 8; i++) imm64 |= uint64_t{imm8} << (i * 8);
          } else {
            uint32_t f = VFPExpandImm32(imm8);
            imm64 = uint64_t{f} | (uint64_t{f} << 32);
          }
          break;
      }
    } else {
      if (cmode == 0b1110) {
        for (int i = 0; i < 8; i++)
          if (imm8 & (1 << i)) imm64 |= 0xFFULL << (i * 8);
      } else if (cmode == 0b1111) {
        imm64 = VFPExpandImm64(imm8);
      } else {
        return ~ExpandSimdModifiedImmJit(0, cmode, abc, defgh, q);
      }
    }
    __uint128_t result = static_cast<__uint128_t>(imm64);
    if (q) result |= static_cast<__uint128_t>(imm64) << 64;
    return result;
  }

  // Materialize a 0/1 predicate register that is 1 exactly when ARM64
  // condition `cond` is satisfied by the NZCV bits in ThreadState.cpu.flags
  // (N@15, Z@14, C@8, V@0). The bit extraction and boolean combination mirror
  // lite_translator.h::EmitJumpIfCondNotMet for every condition. `cond` must
  // not be kAl/kNv (those are unconditional and handled by the caller).
  [[nodiscard]] Register EmitArmCondPredicate(Decoder::Condition cond);

  // Emit a conditional branch to then_bb when `cond` is met, else_bb otherwise,
  // by testing the EmitArmCondPredicate result.
  void EmitCondBranch(Decoder::Condition cond,
                      MachineBasicBlock* then_bb,
                      MachineBasicBlock* else_bb);

  // Map the x86 EFLAGS a preceding UCOMIS{S,D} left in `flags_vreg` to ARM64 FP
  // NZCV and store the packed word into ThreadState.cpu.flags. Bit-exact with
  // lite_translator.h::EmitStoreArmFpNZCV: unordered(PF)=C,V; equal(ZF)=Z,C;
  // less(CF)=N; greater=C. Reads the flags into a GP register once (PseudoRead
  // Flags), then a PF>ZF>CF branch tree selects the leaf that writes cpu.flags.
  // Leaves the builder positioned at the tree's merge block.
  void EmitStoreArmFpNZCV(Register flags_vreg);

  [[nodiscard]] Register AllocTempReg() { return builder_.ir()->AllocVReg(); }
  [[nodiscard]] SimdReg AllocTempSimdReg() { return SimdReg{builder_.ir()->AllocVReg()}; }
  [[nodiscard]] Register GetFlagsRegister() const { return flag_register_; }

  // Set success_ = false and end the region path. Returning helpers add the
  // bail-out and keep the callbacks' return values type-correct.
  void UndefinedReturningVoid() { Undefined(); }
  void UndefinedReturningReg() { Undefined(); }

  void GenJump(GuestAddr target);
  void ExitGeneratedCode(GuestAddr target);
  void ExitRegionIndirect(Register target);

  // After a faulting host memory access, split off a recovery basic block that
  // exits the region so the guest signal handler runs. Guest-agnostic; copied
  // verbatim from heavy_optimizer/riscv64/frontend.cc.
  void GenRecoveryBlockForLastInsn();

  void ResolveJumps();
  void ReplaceJumpWithBranch(MachineBasicBlock* bb, MachineBasicBlock* target_bb);
  void UpdateBranchTargetsAfterSplit(GuestAddr addr,
                                     const MachineBasicBlock* old_bb,
                                     MachineBasicBlock* new_bb);

  void StartRegion() {
    auto* region_entry_bb = builder_.ir()->NewBasicBlock();
    auto* cont_bb = builder_.ir()->NewBasicBlock();
    builder_.ir()->AddEdge(region_entry_bb, cont_bb);
    builder_.StartBasicBlock(region_entry_bb);
    builder_.Gen<PseudoBranch>(cont_bb);
    builder_.StartBasicBlock(cont_bb);
  }

  GuestAddr pc_;
  bool success_;
  x86_64::MachineIRBuilder builder_;
  MachineReg flag_register_;
  bool is_uncond_branch_;
  // Set when ResolveJumps links a backward branch into an in-region loop
  // (a real hot loop captured in this region). Used by the runtime to decide
  // whether a small region — or a region that later bailed — is still worth
  // installing as heavy: an in-region loop avoids the per-iteration region-exit
  // dispatch the lite tier pays, which is the heavy tier's biggest win on tight
  // loops (e.g. integrity-check / CRC loops in real apps).
  bool has_in_region_backedge_ = false;
  // IR positions of all guest instructions of the current region, plus all
  // branch targets the region jumps to. A target outside the current region has
  // an uninitialized position (its basic block is nullptr).
  ArenaMap<GuestAddr, MachineInsnPosition> branch_targets_;

  template <typename... T>
  static constexpr void UNUSED_ARGS(const T&...) {}
};

}  // namespace berberis

#endif  // BERBERIS_HEAVY_OPTIMIZER_ARM64_FRONTEND_H_
