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

#include "simd_register.h"

namespace berberis {

// ARM64 optimizing-tier frontend: translates the ARM64 decoder's
// SemanticsPlayer callbacks into guest-agnostic x86_64 MachineIR. The generic
// region machinery (StartRegion / GenJump / ExitGeneratedCode / ResolveJumps /
// Finalize / StartInsn) is adapted from heavy_optimizer/riscv64/frontend.{h,cc}.
//
// Currently only MoveWide (MOVZ/MOVN) and MoveWideKeep (MOVK) are translated to
// native x86_64; every other instruction calls Undefined() (sets success_ =
// false) so the two-gear runtime falls back to the lite translator/interpreter.
// New instructions are added here as the optimizing tier grows.
class HeavyOptimizerFrontend {
 public:
  using Decoder = Decoder<SemanticsPlayer<HeavyOptimizerFrontend>>;
  using Register = MachineReg;
  static constexpr Register no_register = MachineReg{};
  using FpRegister = SimdReg;
  static constexpr SimdReg no_fp_register = SimdReg{};

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

  Register AddSubImm(bool is_sub, bool set_flags, bool is_64bit, Register src, uint32_t imm) {
    UndefinedReturningReg();
    UNUSED_ARGS(is_sub, set_flags, is_64bit, src, imm);
    return AllocTempReg();
  }

  Register AddSubImmTags(bool is_sub, Register src, uint8_t uimm6, uint8_t uimm4) {
    UndefinedReturningReg();
    UNUSED_ARGS(is_sub, src, uimm6, uimm4);
    return AllocTempReg();
  }

  Register LogicalImm(Decoder::LogicalImmOpcode opcode, bool is_64bit, Register src, uint64_t imm) {
    UndefinedReturningReg();
    UNUSED_ARGS(opcode, is_64bit, src, imm);
    return AllocTempReg();
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

  Register LoadLiteral(Decoder::LoadStoreSize size, bool is_signed, int64_t offset) {
    UndefinedReturningReg();
    UNUSED_ARGS(size, is_signed, offset);
    return AllocTempReg();
  }

  Register Bitfield(Decoder::BitfieldOpcode opcode,
                    bool is_64bit,
                    Register dst_val,
                    Register src,
                    uint8_t immr,
                    uint8_t imms) {
    UndefinedReturningReg();
    UNUSED_ARGS(opcode, is_64bit, dst_val, src, immr, imms);
    return AllocTempReg();
  }

  //
  // Branches.
  //

  void Branch(int32_t offset) {
    UndefinedReturningVoid();
    UNUSED_ARGS(offset);
  }

  void BranchCond(Decoder::Condition cond, int32_t offset) {
    UndefinedReturningVoid();
    UNUSED_ARGS(cond, offset);
  }

  void BranchRegister(Register target) {
    UndefinedReturningVoid();
    UNUSED_ARGS(target);
  }

  void CompareAndBranch(bool is_nonzero, bool is_64bit, Register src, int32_t offset) {
    UndefinedReturningVoid();
    UNUSED_ARGS(is_nonzero, is_64bit, src, offset);
  }

  void TestAndBranch(bool is_nonzero, Register src, uint8_t bit, int32_t offset) {
    UndefinedReturningVoid();
    UNUSED_ARGS(is_nonzero, src, bit, offset);
  }

  //
  // Integer loads / stores.
  //

  Register Load(Decoder::LoadStoreSize size,
                bool is_signed,
                bool is_64bit_target,
                Register base,
                int32_t offset) {
    UndefinedReturningReg();
    UNUSED_ARGS(size, is_signed, is_64bit_target, base, offset);
    return AllocTempReg();
  }

  void Store(Decoder::LoadStoreSize size, Register base, int32_t offset, Register data) {
    UndefinedReturningVoid();
    UNUSED_ARGS(size, base, offset, data);
  }

  Register AddImm(Register base, int32_t offset) {
    UndefinedReturningReg();
    UNUSED_ARGS(base, offset);
    return AllocTempReg();
  }

  void LoadPair(Decoder::LoadStoreSize size,
                Register base,
                int32_t offset,
                uint8_t rt1,
                uint8_t rt2,
                uint8_t scale,
                bool is_signed) {
    UndefinedReturningVoid();
    UNUSED_ARGS(size, base, offset, rt1, rt2, scale, is_signed);
  }

  void StorePair(Decoder::LoadStoreSize size,
                 Register base,
                 int32_t offset,
                 Register data1,
                 Register data2,
                 uint8_t scale) {
    UndefinedReturningVoid();
    UNUSED_ARGS(size, base, offset, data1, data2, scale);
  }

  Register LoadReg(Decoder::LoadStoreSize size,
                   bool is_signed,
                   bool is_64bit_target,
                   Register base,
                   Register offset_reg,
                   uint8_t extend_type,
                   uint8_t shift_amount) {
    UndefinedReturningReg();
    UNUSED_ARGS(size, is_signed, is_64bit_target, base, offset_reg, extend_type, shift_amount);
    return AllocTempReg();
  }

  void StoreReg(Decoder::LoadStoreSize size,
                Register base,
                Register offset_reg,
                uint8_t extend_type,
                uint8_t shift_amount,
                Register data) {
    UndefinedReturningVoid();
    UNUSED_ARGS(size, base, offset_reg, extend_type, shift_amount, data);
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

  Register LogicalShiftedReg(Decoder::LogicalShiftedRegOpcode opcode,
                             bool is_64bit,
                             bool invert,
                             Register src1,
                             Register src2,
                             Decoder::ShiftType shift_type,
                             uint8_t shift_amount) {
    UndefinedReturningReg();
    UNUSED_ARGS(opcode, is_64bit, invert, src1, src2, shift_type, shift_amount);
    return AllocTempReg();
  }

  Register AddSubShiftedReg(bool is_sub,
                            bool set_flags,
                            bool is_64bit,
                            Register src1,
                            Register src2,
                            Decoder::ShiftType shift_type,
                            uint8_t shift_amount) {
    UndefinedReturningReg();
    UNUSED_ARGS(is_sub, set_flags, is_64bit, src1, src2, shift_type, shift_amount);
    return AllocTempReg();
  }

  Register AddSubExtendedReg(bool is_sub,
                             bool set_flags,
                             bool is_64bit,
                             Register src1,
                             Register src2,
                             uint8_t extend_type,
                             uint8_t shift_amount) {
    UndefinedReturningReg();
    UNUSED_ARGS(is_sub, set_flags, is_64bit, src1, src2, extend_type, shift_amount);
    return AllocTempReg();
  }

  Register ConditionalSelect(Decoder::ConditionalSelectOpcode opcode,
                             bool is_64bit,
                             Register src1,
                             Register src2,
                             Decoder::Condition cond) {
    UndefinedReturningReg();
    UNUSED_ARGS(opcode, is_64bit, src1, src2, cond);
    return AllocTempReg();
  }

  Register DataProc2Src(Decoder::DataProc2SrcOpcode opcode,
                        bool is_64bit,
                        Register src1,
                        Register src2) {
    UndefinedReturningReg();
    UNUSED_ARGS(opcode, is_64bit, src1, src2);
    return AllocTempReg();
  }

  Register DataProc3Src(Decoder::DataProc3SrcOpcode opcode,
                        bool is_64bit,
                        Register src1,
                        Register src2,
                        Register src3) {
    UndefinedReturningReg();
    UNUSED_ARGS(opcode, is_64bit, src1, src2, src3);
    return AllocTempReg();
  }

  Register AddSubWithCarry(Register src1, Register src2, bool is_64bit, bool is_sub, bool set_flags) {
    UndefinedReturningReg();
    UNUSED_ARGS(src1, src2, is_64bit, is_sub, set_flags);
    return AllocTempReg();
  }

  Register DataProc1Src(Register src, uint8_t opcode2, bool is_64bit) {
    UndefinedReturningReg();
    UNUSED_ARGS(src, opcode2, is_64bit);
    return AllocTempReg();
  }

  Register Extr(Register src_n, Register src_m, uint8_t lsb, bool is_64bit) {
    UndefinedReturningReg();
    UNUSED_ARGS(src_n, src_m, lsb, is_64bit);
    return AllocTempReg();
  }

  void ConditionalCompare(bool is_neg,
                          bool is_64bit,
                          Register rn,
                          Register rm,
                          Decoder::Condition cond,
                          uint8_t nzcv) {
    UndefinedReturningVoid();
    UNUSED_ARGS(is_neg, is_64bit, rn, rm, cond, nzcv);
  }

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

  void FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ftype, Decoder::Condition cond) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, ftype, cond);
  }

  void FpFixedPointConversion(const Decoder::FpFixedPointArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void FpDataProc3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra, uint8_t ftype, bool o1, bool o0) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, rn, rm, ra, ftype, o1, o0);
  }

  void FpMovImmediate(uint8_t rd, uint8_t imm8, uint8_t ftype) {
    UndefinedReturningVoid();
    UNUSED_ARGS(rd, imm8, ftype);
  }

  void FpIntConversion(const Decoder::FpIntConvArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void FpDataProc1(const Decoder::FpDataProc1Args& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void FpDataProc2(const Decoder::FpDataProc2Args& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

  void FpCompare(const Decoder::FpCompareArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
  }

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

  void AdvSimdThreeSame(const Decoder::AdvSimdThreeSameArgs& args) {
    UndefinedReturningVoid();
    UNUSED_ARGS(args);
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
