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

// Why a heavy-optimizer region bailed to the lite tier. A bail is correct (lite
// re-translates the region) but slower; silent bails on common instructions have
// previously required a manual multi-app sweep to localize. Threaded through the
// Undefined() choke point so a berberis.tracing capture yields a bail histogram.
// Only the interesting classes are annotated at their call sites; the bulk keeps
// the kUnimplemented default and is still bucketable offline via the logged
// instruction encoding.
enum class BailReason {
  kUnimplemented = 0,  // instruction/form not yet lowered in the heavy tier (default)
  kUnsupportedSize,    // operand/access size the heavy path doesn't emit yet
  kUnsupportedFpType,  // ftype/precision guard (e.g. FP16, or a reserved ftype)
  kDefensive,          // validated-away illegal/reserved encoding (never a real gap)
  kOther,              // deliberately uncategorized
  kCount,
};

[[nodiscard]] inline const char* BailReasonName(BailReason reason) {
  switch (reason) {
    case BailReason::kUnimplemented:
      return "unimplemented";
    case BailReason::kUnsupportedSize:
      return "unsupported-size";
    case BailReason::kUnsupportedFpType:
      return "unsupported-fptype";
    case BailReason::kDefensive:
      return "defensive";
    case BailReason::kOther:
      return "other";
    case BailReason::kCount:
      return "?";
  }
  return "?";
}

// ARM64 optimizing-tier frontend: translates the ARM64 decoder's
// SemanticsPlayer callbacks into guest-agnostic x86_64 MachineIR. The generic
// region machinery (StartRegion / GenJump / ExitGeneratedCode / ResolveJumps /
// Finalize / StartInsn) is adapted from heavy_optimizer/riscv64/frontend.{h,cc}.
//
// Integer/branch/load-store instructions, scalar FP, and a broad NEON/SIMD
// subset (three-same, two-reg-misc, shifts, copies/permutes, indexed-element,
// atomics/exclusives, conversions) are translated to native x86_64. Scalar FP
// arithmetic (FADD/FSUB/FMUL/FDIV for S and D) is lowered through the
// guest-agnostic intrinsic layer (inline_intrinsic.h +
// machine_ir_intrinsic_binding.json); most other handlers emit MachineIR
// directly. Anything not handled calls Undefined() (sets success_ = false) so
// the two-gear runtime falls back to the lite translator/interpreter.
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

  void Undefined(BailReason reason = BailReason::kUnimplemented);
  void Nop() {}

  // DMB/DSB full barrier: lowers to MFENCE (x86 TSO lacks StoreLoad ordering).
  // The store-only/load-only barrier variants are NOPed in the decoder.
  void DataMemoryBarrier();

  //
  // Immediate-form data processing.
  //

  Register AddSubImm(bool is_sub, bool set_flags, bool is_64bit, Register src, uint32_t imm);

  Register AddSubImmTags(bool is_sub, Register src, uint8_t uimm6, uint8_t uimm4);

  Register LogicalImm(Decoder::LogicalImmOpcode opcode, bool is_64bit, Register src, uint64_t imm);

  Register MoveWide(Decoder::MoveWideOpcode opcode, bool is_64bit, uint16_t imm16, uint8_t shift);

  Register MoveWideKeep(Register current, uint16_t imm16, uint8_t shift, bool is_64bit);

  Register PcRelAddr(bool is_adrp, int64_t offset);

  Register LoadLiteral(Decoder::LoadStoreSize size, bool is_signed, int64_t offset);

  Register Bitfield(Decoder::BitfieldOpcode opcode,
                    bool is_64bit,
                    Register dst_val,
                    Register src,
                    uint8_t immr,
                    uint8_t imms);

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

  Register Load(Decoder::LoadStoreSize size,
                bool is_signed,
                bool is_64bit_target,
                Register base,
                int32_t offset);

  void Store(Decoder::LoadStoreSize size, Register base, int32_t offset, Register data);

  Register AddImm(Register base, int32_t offset);

  void LoadPair(Decoder::LoadStoreSize size,
                Register base,
                int32_t offset,
                uint8_t rt1,
                uint8_t rt2,
                uint8_t scale,
                bool is_signed);

  void StorePair(Decoder::LoadStoreSize size,
                 Register base,
                 int32_t offset,
                 Register data1,
                 Register data2,
                 uint8_t scale);

  Register LoadReg(Decoder::LoadStoreSize size,
                   bool is_signed,
                   bool is_64bit_target,
                   Register base,
                   Register offset_reg,
                   uint8_t extend_type,
                   uint8_t shift_amount);

  void StoreReg(Decoder::LoadStoreSize size,
                Register base,
                Register offset_reg,
                uint8_t extend_type,
                uint8_t shift_amount,
                Register data);

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

  void Svc(uint16_t imm);

  void Brk(uint16_t imm);

  Register Mrs(Decoder::SystemReg sysreg);

  void Msr(Decoder::SystemReg sysreg, Register src);

  void IcIvau(uint8_t rt);

  //
  // Register-form data processing.
  //

  Register LogicalShiftedReg(Decoder::LogicalShiftedRegOpcode opcode,
                             bool is_64bit,
                             bool invert,
                             Register src1,
                             Register src2,
                             Decoder::ShiftType shift_type,
                             uint8_t shift_amount);

  Register AddSubShiftedReg(bool is_sub,
                            bool set_flags,
                            bool is_64bit,
                            Register src1,
                            Register src2,
                            Decoder::ShiftType shift_type,
                            uint8_t shift_amount);

  Register AddSubExtendedReg(bool is_sub,
                             bool set_flags,
                             bool is_64bit,
                             Register src1,
                             Register src2,
                             uint8_t extend_type,
                             uint8_t shift_amount);

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

  Register DataProc2Src(Decoder::DataProc2SrcOpcode opcode,
                        bool is_64bit,
                        Register src1,
                        Register src2);

  Register DataProc3Src(Decoder::DataProc3SrcOpcode opcode,
                        bool is_64bit,
                        Register src1,
                        Register src2,
                        Register src3);

  Register AddSubWithCarry(Register src1, Register src2, bool is_64bit, bool is_sub, bool set_flags);

  Register DataProc1Src(Register src, uint8_t opcode2, bool is_64bit);

  Register Extr(Register src_n, Register src_m, uint8_t lsb, bool is_64bit);

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

  void MteDataProc(const Decoder::MteDataProcArgs& args);

  void MteLoadStore(const Decoder::MteLoadStoreArgs& args);

  //
  // Floating-point scalar.
  //

  void FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ftype, Decoder::Condition cond);

  void FpFixedPointConversion(const Decoder::FpFixedPointArgs& args);

  void FpDataProc3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra, uint8_t ftype, bool o1, bool o0);

  void FpMovImmediate(uint8_t rd, uint8_t imm8, uint8_t ftype);

  void FpIntConversion(const Decoder::FpIntConvArgs& args);

  void FpDataProc1(const Decoder::FpDataProc1Args& args);

  void FpDataProc2(const Decoder::FpDataProc2Args& args);

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
  // `fbits != 0` selects the fixed-point form (SCVTF/UCVTF fixed): the FP result
  // is scaled by 2^-fbits (an exact power-of-2 multiply that only biases the
  // exponent, so CVTSI2's single rounding carries through). `fbits == 0` is the
  // plain integer form. Mirrors lite_translator.h::FpFixedPointConversion.
  // `src_from_simd` (used by the scalar-SIMD `.d` SCVTF/UCVTF Dd,Dn forms):
  // read the source integer from V[rn]'s low 64-bit lane (via MOVQ) instead of
  // the general register X[rn]; also disables the rn==31/XZR special case (V31
  // is a valid SIMD source, not zero). The result still commits to V[rd] via
  // SetVRegScalar.
  void EmitScvtfUcvtf(const Decoder::FpIntConvArgs& args,
                      bool is_double,
                      uint8_t fbits = 0,
                      bool src_from_simd = false);

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
  // `fbits != 0` selects the fixed-point form (FCVTZS/FCVTZU fixed): the FP
  // source is pre-multiplied by 2^+fbits (exact power-of-2 scale) before the
  // truncating cvtt + saturation ladder. `fbits == 0` is the plain integer form.
  // Fixed-point never combines with the rounding/ties-away paths (round_imm < 0,
  // ties_away false). Mirrors lite_translator.h::FpFixedPointConversion.
  // `dest_to_simd` (used by the scalar-SIMD `.d` FCVTZS/FCVTZU Dd,Dn forms):
  // commit the integer result to V[rd]'s low 64-bit lane (via
  // SetVRegScalarFromGp, zeroing Vd[127:64]) instead of the general register
  // X[rd]; rd==31 is then V31, a valid destination, not discarded.
  void EmitFcvtz(const Decoder::FpIntConvArgs& args,
                 bool is_double,
                 int8_t round_imm = -1,
                 bool ties_away = false,
                 uint8_t fbits = 0,
                 bool dest_to_simd = false);

  //
  // Advanced SIMD (Args-struct forms).
  //

  void AdvSimdFcma(const Decoder::FcmaArgs& args);

  void AdvSimdFcmaIdx(const Decoder::FcmaIdxArgs& args);

  void AdvSimdBf16ThreeSame(const Decoder::Bf16ThreeSameArgs& args);

  void AdvSimdMatMul(const Decoder::MatMulArgs& args);

  void AdvSimdDotProduct(const Decoder::DotProductArgs& args);

  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs& args);

  void SimdLoadLiteral(const Decoder::SimdLoadLiteralArgs& args);

  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& args, Register base);

  void SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs& args, Register addr);

  void SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs& args, Register base, Register offset);

  void AdvSimdCopy(const Decoder::AdvSimdCopyArgs& args);

  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& args);

  void AdvSimdThreeDiff(const Decoder::AdvSimdThreeDiffArgs& args);

  void AdvSimdSingleStruct(const Decoder::AdvSimdSingleStructArgs& args);

  void AdvSimdTwoRegMisc(const Decoder::AdvSimdTwoRegMiscArgs& args);

  void AdvSimdScalarTwoRegMisc(const Decoder::AdvSimdScalarTwoRegMiscArgs& args);

  void AdvSimdScalarThreeSame(const Decoder::AdvSimdScalarThreeSameArgs& args);

  void AdvSimdScalarPairwise(const Decoder::AdvSimdScalarPairwiseArgs& args);

  void AdvSimdShiftByImm(const Decoder::AdvSimdShiftImmArgs& args);

  void AdvSimdVecXIndexedElement(const Decoder::AdvSimdVecXIdxArgs& args);

  void AdvSimdScalarXIndexedElement(const Decoder::AdvSimdScalarXIdxArgs& args);

  //
  // Advanced SIMD (decomposed-primitive forms).
  //

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q);

  void AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size, uint8_t opcode, bool q);

  void AdvSimdTableLookup(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t len, uint8_t op, bool q);

  void AdvSimdMultiStruct(uint8_t rt,
                          uint8_t rn,
                          uint8_t num_regs,
                          uint8_t size,
                          bool q,
                          bool is_store,
                          bool postindex,
                          uint8_t rm,
                          bool is_interleaved);

  //
  // Crypto.
  //

  void Sha512(Decoder::Sha512Op op, uint8_t rd, uint8_t rn, uint8_t rm);

  void Eor3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra);

  void Bcax(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra);

  void Rax1(uint8_t rd, uint8_t rn, uint8_t rm);

  void Xar(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm6);

  void Sm4e(uint8_t rd, uint8_t rn);

  void Sm4ekey(uint8_t rd, uint8_t rn, uint8_t rm);

  void Sm3ss1(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra);

  void Sm3tt(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm2, uint8_t op);

  void Sm3partw1(uint8_t rd, uint8_t rn, uint8_t rm);

  void Sm3partw2(uint8_t rd, uint8_t rn, uint8_t rm);

  void CryptoAes(uint8_t rd, uint8_t rn, uint8_t opcode);

  void CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode);

  void CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode);

  // SHA-256 arithmetic helpers (defined in frontend.cc). Nested so it can reach
  // the frontend's emit helpers while keeping the round math out of the header.
  struct Sha256Ops;

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
  // Out-of-line so the trace/atomic machinery never inlines into the many bail
  // sites. Called from Undefined() only when Tracing::IsOn().
  void RecordBail(BailReason reason);

  // ThreadState offsets (computed directly; arm64 guest_state has no
  // GetThreadStateRegOffset helper like riscv64).
  static int32_t GetThreadStateRegOffset(uint8_t reg);
  static int32_t GetThreadStateSpOffset();

  // Byte offset of guest V[reg] within ThreadState, i.e. the displacement used
  // by every GenGetSimd/GenSetSimd/MOVDQA against RBP. Equals
  // offsetof(ThreadState, cpu.v[0]) + reg * 16 (each v[] slot is 16 bytes).
  static int32_t GetVRegOffset(unsigned reg) {
    return static_cast<int32_t>(GetThreadStateSimdRegOffset(static_cast<int>(reg)));
  }

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
  //   AND kFlagsNZCVMask            -> keep only N, Z, C, V
  //   if is_sub: XOR kFlagCarry     -> ARM borrow is inverted (ARM C = !x86 CF)
  //   MOVW [rbp + cpu.flags], raw   -> 16-bit store of the packed NZCV
  // cpu.flags is a uint16_t and the 2 bytes after it are alignment padding
  // before cpu.cached_fpcr, but the 16-bit store mirrors lite exactly and never
  // touches a neighbouring field.
  void EmitMaterializeNZCV(Register flags_vreg, bool is_sub) {
    Register raw = AllocTempReg();
    builder_.Gen<PseudoReadFlags>(PseudoReadFlags::kWithOverflow, raw, flags_vreg);
    raw = std::get<0>(Gen<x86_64::AndlRegImm, kNoSSA>(raw, static_cast<int32_t>(kFlagsNZCVMask)));
    if (is_sub) {
      raw = std::get<0>(Gen<x86_64::XorlRegImm, kNoSSA>(raw, static_cast<int32_t>(CPUState::kFlagCarry)));
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

  [[nodiscard]] Register EmitRegOffsetAddr(Register base,
                                           Register offset_reg,
                                           uint8_t extend_type,
                                           uint8_t shift_amount);

  [[nodiscard]] Register EmitShiftImm(Register src,
                                      Decoder::ShiftType shift_type,
                                      uint8_t shift_amount,
                                      bool is_64bit);

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
    const int32_t off = GetVRegOffset(reg);
    return FpRegister{
        std::get<0>(Gen<x86_64::MovsdXRegOp>({.base = x86_64::kMachineRegRBP, .disp = off}))};
  }

  // Widen a scalar FP16 in V[reg] bits[15:0] to an FP32 in lane 0 of a fresh XMM,
  // forcing lanes 1..3 to +0.0. GetVRegScalar's 64-bit MOVSD may leave garbage in
  // bits[63:16], so PAND with [0x0000FFFF,0,0,0] isolates the half BEFORE
  // VCVTPH2PS (which reads each 16-bit lane independently). Keeping lanes 1..3 at
  // +0.0 makes the later VCVTPS2PH narrow yield [half,0,0,0] so SetVRegScalar's
  // MOVSS lane-0 copy is clean. Bit-exact with the lite tier's Pxor+Pinsrw widen.
  // Caller must have already checked host_platform::kHasF16C.
  [[nodiscard]] FpRegister EmitWidenHalfToF32(uint8_t reg) {
    FpRegister raw = GetVRegScalar(reg, /*is_double=*/false);  // MOVSD: [half,junk,0,0]
    FpRegister mask = AllocTempSimdReg();
    builder_.Gen<x86_64::MovdXRegReg>(mask.machine_reg(), GetImm(uint64_t{0x0000FFFF}));
    builder_.Gen<x86_64::PandXRegXReg>(raw.machine_reg(), mask.machine_reg());  // [half,0,0,0]
    builder_.Gen<x86_64::Vcvtph2psXRegXReg>(raw.machine_reg(), raw.machine_reg());  // [f32,+0,+0,+0]
    return raw;
  }

  // Narrow the FP32 in lane 0 of `val` (lanes 1..3 must be +0.0) to FP16 and
  // commit to V[reg] scalar with upper bytes zeroed. VCVTPS2PH RNE (imm=0) is the
  // single rounding; +0.0 upper lanes narrow to 0x0000 so bits[31:16] of lane 0
  // are zero when SetVRegScalar copies it. Caller must have checked F16C.
  void EmitNarrowF32ToHalfAndStore(uint8_t reg, FpRegister val) {
    builder_.Gen<x86_64::Vcvtps2phXRegXRegImm>(val.machine_reg(), val.machine_reg(),
                                               int8_t{0});
    SetVRegScalar(reg, val, /*is_double=*/false);
  }

  // Widen the FP16 lanes of the 16-byte guest V slot at `voff` to FP32 for a heavy
  // vector FP16 op. The low 4 halves (lanes 0..3) land as 4 FP32 in `*lo`; for q (.8H)
  // the high 4 halves (lanes 4..7) land as 4 FP32 in `*hi` (PSRLDQ 8 brings them down
  // first). Both are fresh temps. `*hi` is untouched when !q. GenGetSimd defines the
  // reg (no use-before-def). Caller must have checked host_platform::kHasF16C.
  void EmitWidenHalfVec(int32_t voff, bool q, FpRegister* lo, FpRegister* hi) {
    FpRegister xlo = AllocTempSimdReg();
    builder_.GenGetSimd<16>(xlo.machine_reg(), voff);
    if (q) {
      FpRegister xhi = AllocTempSimdReg();
      builder_.Gen<x86_64::MovdqaXRegXReg>(xhi.machine_reg(), xlo.machine_reg());
      builder_.Gen<x86_64::PsrldqXRegImm>(xhi.machine_reg(), int8_t{8});
      builder_.Gen<x86_64::Vcvtph2psXRegXReg>(xhi.machine_reg(), xhi.machine_reg());
      *hi = xhi;
    }
    // VCVTPH2PS reads only the low 64 bits (4 halves) of its source; lanes 4..7 in the
    // 128-bit load are ignored, so no pre-mask is needed for the low group.
    builder_.Gen<x86_64::Vcvtph2psXRegXReg>(xlo.machine_reg(), xlo.machine_reg());
    *lo = xlo;
  }

  // Narrow FP32 arithmetic results back to FP16 and commit to V[rd]. `lo` holds the 4
  // result FP32 for lanes 0..3; for q, `hi` holds lanes 4..7. VCVTPS2PH RNE (imm=0) is
  // the single rounding (exact for one FP16 op — FP32 mantissa strictly contains FP16's);
  // it packs 4 halves into the low 64 of its dst and zeroes the upper 64. For .8H the
  // high group is shifted into the upper 64 (PSLLDQ 8) and OR'd in. SetVRegFull zeroes
  // Vd[127:64] for q=0. Caller must have checked host_platform::kHasF16C.
  void EmitNarrowHalfVecAndStore(uint8_t rd, FpRegister lo, FpRegister hi, bool q) {
    builder_.Gen<x86_64::Vcvtps2phXRegXRegImm>(lo.machine_reg(), lo.machine_reg(), int8_t{0});
    if (q) {
      builder_.Gen<x86_64::Vcvtps2phXRegXRegImm>(hi.machine_reg(), hi.machine_reg(), int8_t{0});
      builder_.Gen<x86_64::PslldqXRegImm>(hi.machine_reg(), int8_t{8});
      builder_.Gen<x86_64::PorXRegXReg>(lo.machine_reg(), hi.machine_reg());
    }
    SetVRegFull(rd, lo, q);
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

  // Allocate an XMM preset to all-ones (every bit set), the common source for
  // sign/mask/-1 constants. Must start from AllocZeroedSimdReg(): a self-Pcmpeq
  // on a fresh undefined vreg is a use-before-def for the lifetime analysis
  // (reg_class_ CHECK), so the zeroing def makes the self-read legal. The
  // Pcmpeq width is irrelevant to the result (all-ones regardless of element
  // size); PCMPEQD is used as the single canonical form.
  [[nodiscard]] FpRegister AllocOnesSimdReg() {
    FpRegister ones = AllocZeroedSimdReg();
    builder_.Gen<x86_64::PcmpeqdXRegXReg>(ones.machine_reg(), ones.machine_reg());
    return ones;
  }

  // Packed-single FMULX saturation blend: FMUL except (+-0 * +-inf) lanes return
  // +-2.0 (sign = sign(a) XOR sign(b)) instead of NaN. Op-for-op mirror of the
  // scalar/by-element FP32 FMULX blends. Destroys `a`; returns the result reg.
  [[nodiscard]] FpRegister EmitFmulxF32Packed(FpRegister a, FpRegister b);

  // Packed-single pairwise-style min/max with ARM's +-0 tie and NaN rules:
  // FMAXP/FMINP (is_nm=false) propagate NaN; FMAXNMP/FMINNMP (is_nm=true)
  // suppress it; the +-0 tie is fixed by an explicit AND(max)/OR(min) blend over
  // lanes where both inputs are zero. Destroys xa/xb; returns the result reg.
  [[nodiscard]] FpRegister EmitFpPairwiseMinMaxF32Packed(FpRegister xa, FpRegister xb,
                                                         bool is_max, bool is_nm);

  void SetVRegScalar(uint8_t reg, FpRegister value, bool is_double);

  void SetVRegScalarFromGp(uint8_t reg, Register value, bool is_double);

  void SetVRegFull(uint8_t reg, FpRegister value, bool q);

  void SetVRegNarrow(uint8_t rd, FpRegister narrowed, bool q);

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
  // VFPExpandImm for half precision (E=5, F=10): FMOV Hd,#imm. Pure function of
  // imm8, computed at translation time. Mirrors lite_translator_fp_scalar.inc's
  // FpMovImmediate half expansion (which the lite tier currently bails on; the
  // interpreter implements it, so all tiers stay correct).
  static uint16_t VFPExpandImm16(uint8_t imm8) {
    uint16_t sign = (imm8 >> 7) & 1;
    uint16_t b = (imm8 >> 6) & 1;
    uint16_t exp = static_cast<uint16_t>(((1 - b) << 4) | ((b ? 0x3u : 0u) << 2) |
                                         ((imm8 >> 4) & 0x3u));  // 5 bits
    uint16_t mantissa = static_cast<uint16_t>(imm8 & 0xFu) << 6;  // 10 bits
    return static_cast<uint16_t>((sign << 15) | (exp << 10) | mantissa);
  }

  static __uint128_t ExpandSimdModifiedImmJit(uint8_t op, uint8_t cmode, uint8_t abc,
                                              uint8_t defgh, bool q);

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
  void UndefinedReturningVoid(BailReason reason = BailReason::kUnimplemented) {
    Undefined(reason);
  }
  void UndefinedReturningReg(BailReason reason = BailReason::kUnimplemented) {
    Undefined(reason);
  }

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
