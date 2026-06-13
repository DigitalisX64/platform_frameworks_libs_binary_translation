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

  Register PcRelAddr(bool is_adrp, int64_t offset) {
    UndefinedReturningReg();
    UNUSED_ARGS(is_adrp, offset);
    return AllocTempReg();
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
      // Other SBFM (SBFX/SBFIZ) needs an arithmetic-shift-based extraction that
      // is fiddlier to lower correctly; bail conservatively.
      UndefinedReturningReg();
      return AllocTempReg();
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

  void LoadStoreExclusive(const Decoder::LoadStoreExclusiveArgs& args, Register base) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args, base);
  }

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

  Register Mrs(Decoder::SystemReg sysreg) {
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

  // LSLV/LSRV/ASRV/RORV (variable shifts). UDIV/SDIV/CRC32/PACGA bail: division
  // needs x86 RDX:RAX setup plus ARM divide-by-zero / INT_MIN-overflow handling
  // that is awkward in this MachineIR form, so those fall back to the lite
  // translator. The variable shifts use the x86 shift-by-CL forms; the backend
  // register allocator binds the count operand to RCX automatically.
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
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }

  // MADD/MSUB and the signed/unsigned widening multiply-accumulates
  // (SMADDL/SMSUBL/UMADDL/UMSUBL). SMULH/UMULH need the widening x86 MUL/IMUL
  // into RDX:RAX and bail. Mirrors lite_translator.h::DataProc3Src.
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
      default:
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }

  Register AddSubWithCarry(Register src1, Register src2, bool is_64bit, bool is_sub, bool set_flags) {
    UndefinedReturningReg();
    UNUSED_ARGS(src1, src2, is_64bit, is_sub, set_flags);
    return AllocTempReg();
  }

  // RBIT/REV16/REV32/REV/CLZ/CLS. Only REV16 and CLZ have a clean mapping here;
  // REV/REV32 would need x86 BSWAP (no MachineIR op available) and RBIT/CLS have
  // no direct mapping, so they bail. PAuth DP-1Src variants (opcode2 bit 0x40)
  // are treated as identity because Digitalis is PAC-blind. Mirrors
  // lite_translator.h::DataProc1Src.
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
      default:
        // RBIT (000000), REV32/REV (000010), REV (000011), CLS (000101): no
        // clean x86 MachineIR mapping here — bail to the lite translator.
        UndefinedReturningReg();
        return AllocTempReg();
    }
  }

  // EXTR Rd, Rn, Rm, #lsb: Rd = (Rn:Rm) >> lsb. lsb==0 is a copy of Rm. The
  // 32-bit non-zero case maps to x86 SHRD; the 64-bit non-zero case bails (the
  // 64-bit SHRD form is not available as a MachineIR op). Mirrors
  // lite_translator.h::Extr.
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
    if (is_64bit) {
      // No 64-bit SHRD MachineIR op; bail to lite.
      UndefinedReturningReg();
      return AllocTempReg();
    }
    // SHRD dest, src, imm: dest = (src:dest) >> imm.
    // ARM EXTR Wd = (Wn:Wm) >> lsb = SHRD(Wm, Wn, lsb): dest=Wm, src=Wn.
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

  // FCSEL: predicate-select between two scalar V regs. Not wired into the
  // optimizing tier yet (needs basic-block manipulation + scalar select);
  // bail to the lite translator/interpreter.
  void FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ftype, Decoder::Condition cond) {
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

  // FMADD/FMSUB/FNMADD/FNMSUB. Would need the FMA intrinsic bindings + scalar
  // V-reg plumbing for three operands; bail for now.
  void FpDataProc3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra, uint8_t ftype, bool o1, bool o0) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, ra, ftype, o1, o0);
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

  // FCVTZS/FCVTZU/SCVTF/UCVTF (FP<->int). The FCvtFloatToInteger* intrinsics +
  // cvtsi2ss SSE ops are not available to the ARM64 backend (riscv64-only macro
  // defs), so bail.
  void FpIntConversion(const Decoder::FpIntConvArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
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
  // FMAX/FMIN/FMAXNM/FMINNM/FNMUL (opcode >= 0b0100), FP16 (ftype=0b11), and the
  // reserved ftype=0b10 bail: their intrinsics/SSE ops are not wired here.
  void FpDataProc2(const Decoder::FpDataProc2Args& args) {
    if (!success()) {
      return;
    }
    if (args.ftype != 0b00 && args.ftype != 0b01) {
      UndefinedReturningVoid();
      return;
    }
    if (args.opcode > 0b0011) {
      // FMAX/FMIN/FMAXNM/FMINNM/FNMUL not wired into the optimizing tier.
      UndefinedReturningVoid();
      return;
    }
    const bool is_double = (args.ftype == 0b01);
    FpRegister src1 = GetVRegScalar(args.rn, is_double);
    FpRegister src2 = GetVRegScalar(args.rm, is_double);
    FpRegister result = AllocTempSimdReg();
    if (is_double) {
      switch (args.opcode) {
        case 0b0000:  // FMUL
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
    SetVRegScalar(args.rd, result, is_double);
  }

  // FCMP/FCMPE: needs an x86 UCOMIS{S,D} -> ARM FP NZCV mapping (the Ucomis SSE
  // ops are not allowlisted for the ARM64 backend); bail.
  void FpCompare(const Decoder::FpCompareArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  // FCCMP/FCCMPE: same UCOMIS dependency as FCMP plus a predicate; bail.
  void FpConditionalCompare(const Decoder::FpConditionalCompareArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

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

  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void SimdLoadLiteral(const Decoder::SimdLoadLiteralArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& args, Register base) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args, base);
  }

  void SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs& args, Register addr) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args, addr);
  }

  void SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs& args, Register base, Register offset) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args, base, offset);
  }

  void AdvSimdCopy(const Decoder::AdvSimdCopyArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
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
  //   ADD: Paddw (16), Paddd (32). 8-bit (Paddb) and 64-bit (Paddq) bail.
  //   SUB: Psubd (32). 8/16/64-bit (Psubb/Psubw/Psubq) bail.
  //   MUL: Pmullw (16), Pmulld (32). 8-bit and 64-bit have no packed op; bail.
  //   AND/ORR/EOR: Pand/Por/Pxor are element-size-independent (one op covers
  //     all). ORR with rn==rm is the AdvSIMD MOV (vector) alias and lowers the
  //     same way.
  //   CMEQ: Pcmpeqb/w/d are not in the ARM64 backend allowlist; bail (the lite
  //     translator/interpreter handles CMEQ).
  // Everything else (saturating, shifts, polynomial, FP, pairwise, widening,
  // CMGT/CMHI/etc.) bails to the lite translator/interpreter.
  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& args) {
    if (!success()) {
      return;
    }
    const int32_t vn_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rn * 16);
    const int32_t vm_off = static_cast<int32_t>(offsetof(ThreadState, cpu.v[0]) + args.rm * 16);

    // Validate the (opcode, size) pair up front and emit nothing on bail. After
    // this switch every reachable case has a single allowlisted packed op.
    switch (args.opcode) {
      case Decoder::AdvSimdThreeSameOpcode::kAdd:
        if (args.size != 0b01 && args.size != 0b10) {
          UndefinedReturningVoid();
          return;
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kSub:
        if (args.size != 0b10) {
          UndefinedReturningVoid();
          return;
        }
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
        if (args.size == 0b01) {
          builder_.Gen<x86_64::PaddwXRegXReg>(vn.machine_reg(), vm.machine_reg());
        } else {
          builder_.Gen<x86_64::PadddXRegXReg>(vn.machine_reg(), vm.machine_reg());
        }
        break;
      case Decoder::AdvSimdThreeSameOpcode::kSub:
        builder_.Gen<x86_64::PsubdXRegXReg>(vn.machine_reg(), vm.machine_reg());
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
      default:
        // Unreachable: the validation switch above already bailed.
        UndefinedReturningVoid();
        return;
    }

    SetVRegFull(args.rd, vn, args.q);
  }

  void AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void AdvSimdScalarXIndexedElement(const Decoder::AdvSimdScalarXIdxArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  //
  // Advanced SIMD (decomposed-primitive forms).
  //

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, index, q);
  }

  void AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size, uint8_t opcode, bool q) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, size, opcode, q);
  }

  void AdvSimdTableLookup(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t len, uint8_t op, bool q) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, len, op, q);
  }

  void AdvSimdMultiStruct(uint8_t rt,
                          uint8_t rn,
                          uint8_t num_regs,
                          uint8_t size,
                          bool q,
                          bool is_store,
                          bool postindex,
                          uint8_t rm,
                          bool is_interleaved) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rt, rn, num_regs, size, q, is_store, postindex, rm, is_interleaved);
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
  // IR positions of all guest instructions of the current region, plus all
  // branch targets the region jumps to. A target outside the current region has
  // an uninitialized position (its basic block is nullptr).
  ArenaMap<GuestAddr, MachineInsnPosition> branch_targets_;

  template <typename... T>
  static constexpr void UNUSED_ARGS(const T&...) {}
};

}  // namespace berberis

#endif  // BERBERIS_HEAVY_OPTIMIZER_ARM64_FRONTEND_H_
