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

#ifndef BERBERIS_DECODER_ARM64_SEMANTICS_PLAYER_H_
#define BERBERIS_DECODER_ARM64_SEMANTICS_PLAYER_H_

#include "berberis/decoder/arm64/decoder.h"

namespace berberis {

// This class expresses the semantics of ARM64 instructions by calling a sequence of
// SemanticsListener callbacks.
template <class SemanticsListener>
class SemanticsPlayer {
 public:
  using Decoder = Decoder<SemanticsPlayer>;
  using Register = typename SemanticsListener::Register;

  explicit SemanticsPlayer(SemanticsListener* listener) : listener_(listener) {}

  // Decoder's InsnConsumer implementation.

  void AddSubImm(const typename Decoder::AddSubImmArgs& args) {
    // Rn=31 means SP for ADD/SUB (immediate).
    Register src = GetRegOrSp(args.src);
    Register result = listener_->AddSubImm(args.is_sub, args.set_flags, args.is_64bit,
                                           src, args.imm);
    if (args.set_flags) {
      // When S is set, Rd=31 means ZR (write to nowhere), flags are the real output.
      SetRegOrIgnore(args.dst, result);
    } else {
      // When S is not set, Rd=31 means SP.
      SetRegOrSp(args.dst, result);
    }
  }

  // ADDG/SUBG (add/subtract immediate, with tags).
  // Rn/Rd = 31 mean SP (not ZR). The listener does the address + tag-nibble
  // arithmetic on the value; the player handles the SP read/write.
  void AddSubImmTags(const typename Decoder::AddSubImmTagsArgs& args) {
    Register src = GetRegOrSp(args.src);
    Register result = listener_->AddSubImmTags(args.is_sub, src, args.uimm6, args.uimm4);
    SetRegOrSp(args.dst, result);
  }

  void LogicalImm(const typename Decoder::LogicalImmArgs& args) {
    Register src = GetRegOrZero(args.src);
    Register result = listener_->LogicalImm(args.opcode, args.is_64bit, src, args.imm);
    if (args.opcode == Decoder::LogicalImmOpcode::kAnds) {
      // ANDS: Rd=31 means ZR (write flags only).
      SetRegOrIgnore(args.dst, result);
    } else {
      // AND/ORR/EOR: Rd=31 means SP.
      SetRegOrSp(args.dst, result);
    }
  }

  void MoveWide(const typename Decoder::MoveWideArgs& args) {
    if (args.opcode == Decoder::MoveWideOpcode::kMovk) {
      // MOVK: keep other bits of the destination register, insert 16-bit immediate.
      Register current = (args.dst != 31) ? listener_->GetReg(args.dst) : listener_->GetImm(0);
      Register result = listener_->MoveWideKeep(current, args.imm, args.shift, args.is_64bit);
      SetRegOrIgnore(args.dst, result);
    } else {
      Register result = listener_->MoveWide(args.opcode, args.is_64bit, args.imm, args.shift);
      SetRegOrIgnore(args.dst, result);
    }
  }

  void PcRelAddr(const typename Decoder::PcRelAddrArgs& args) {
    Register result = listener_->PcRelAddr(args.is_adrp, args.offset);
    SetRegOrIgnore(args.dst, result);
  }

  // LDR/LDRSW (literal) — PC-relative integer load. The listener
  // computes the target address from `(insn_addr + offset)` and reads
  // `size` bytes; LDRSW sign-extends 32→64. PRFM literal is decoded as
  // Nop() upstream, so this handler is reached only for the load forms.
  void LoadLiteral(const typename Decoder::LoadLiteralArgs& args) {
    Register result = listener_->LoadLiteral(args.size, args.is_signed, args.offset);
    SetRegOrIgnore(args.rt, result);
  }

  void Bitfield(const typename Decoder::BitfieldArgs& args) {
    Register src = GetRegOrZero(args.src);
    Register dst_val = (args.opcode == Decoder::BitfieldOpcode::kBfm)
                           ? GetRegOrZero(args.dst)
                           : listener_->GetImm(0);
    Register result = listener_->Bitfield(args.opcode, args.is_64bit, dst_val, src,
                                          args.immr, args.imms);
    SetRegOrIgnore(args.dst, result);
  }

  void BranchImm(const typename Decoder::BranchImmArgs& args) {
    if (args.is_link) {
      // BL: save return address in X30.
      Register ret_addr = listener_->GetImm(listener_->GetInsnAddr() + 4);
      listener_->SetReg(30, ret_addr);
    }
    listener_->Branch(args.offset);
  }

  void BranchCond(const typename Decoder::BranchCondArgs& args) {
    listener_->BranchCond(args.cond, args.offset);
  }

  void BranchReg(const typename Decoder::BranchRegArgs& args) {
    Register target = GetRegOrZero(args.src);
    if (args.is_link) {
      // BLR: save return address in X30.
      // snapshot target before overwriting X30.
      // For BLR X30 (or any BLR Xn where Xn maps to the same host register as
      // X30), SetReg(30, ret_addr) would clobber the host register that still
      // holds the original branch target. Copy target to a fresh temp first so
      // ExitRegionIndirect jumps to the correct address.
      if (args.src == 30) {
        target = listener_->Copy(target);
      }
      Register ret_addr = listener_->GetImm(listener_->GetInsnAddr() + 4);
      listener_->SetReg(30, ret_addr);
    }
    listener_->BranchRegister(target);
  }

  void CompareAndBranch(const typename Decoder::CompareAndBranchArgs& args) {
    Register src = GetRegOrZero(args.src);
    listener_->CompareAndBranch(args.is_nonzero, args.is_64bit, src, args.offset);
  }

  void TestAndBranch(const typename Decoder::TestAndBranchArgs& args) {
    Register src = GetRegOrZero(args.src);
    listener_->TestAndBranch(args.is_nonzero, src, args.bit, args.offset);
  }

  void LoadStoreImm(const typename Decoder::LoadStoreImmArgs& args) {
    // Rn=31 means SP for load/store.
    Register base = GetRegOrSp(args.rn);
    if (args.is_store) {
      Register data = GetRegOrZero(args.rt);
      listener_->Store(args.size, base, args.offset, data);
    } else {
      Register result = listener_->Load(args.size, args.is_signed, args.is_64bit_target,
                                        base, args.offset);
      SetRegOrIgnore(args.rt, result);
    }
  }

  void LoadStoreImmPreIndex(const typename Decoder::LoadStoreImmArgs& args) {
    Register base = GetRegOrSp(args.rn);
    Register new_base = listener_->AddImm(base, args.offset);
    if (args.is_store) {
      Register data = GetRegOrZero(args.rt);
      listener_->Store(args.size, new_base, 0, data);
    } else {
      Register result = listener_->Load(args.size, args.is_signed, args.is_64bit_target,
                                        new_base, 0);
      SetRegOrIgnore(args.rt, result);
    }
    // Write back the new base address.
    SetRegOrSp(args.rn, new_base);
  }

  void LoadStoreImmPostIndex(const typename Decoder::LoadStoreImmArgs& args) {
    Register base = GetRegOrSp(args.rn);
    if (args.is_store) {
      Register data = GetRegOrZero(args.rt);
      listener_->Store(args.size, base, 0, data);
    } else {
      Register result = listener_->Load(args.size, args.is_signed, args.is_64bit_target,
                                        base, 0);
      SetRegOrIgnore(args.rt, result);
    }
    // Write back base + offset.
    Register new_base = listener_->AddImm(base, args.offset);
    SetRegOrSp(args.rn, new_base);
  }

  void LoadStorePair(const typename Decoder::LoadStorePairArgs& args) {
    Register base = GetRegOrSp(args.rn);

    // For pre-index and signed-offset, compute address with offset.
    // // For pre-index, compute address first and write back.
    // Register addr = base;
    // if (args.is_preindex) {
    //   addr = listener_->AddImm(base, args.offset);
    // }
    Register addr = base;
    if (args.is_preindex) {
      addr = listener_->AddImm(base, args.offset);
    } else if (!args.is_postindex && args.offset != 0) {
      // Signed-offset: apply offset without writeback.
      addr = listener_->AddImm(base, args.offset);
    }

    uint8_t scale = (args.size == Decoder::LoadStoreSize::k64bit) ? 8 : 4;

    if (args.is_store) {
      Register data1 = GetRegOrZero(args.rt1);
      Register data2 = GetRegOrZero(args.rt2);
      listener_->StorePair(args.size, addr, 0, data1, data2, scale);
    } else {
      listener_->LoadPair(args.size, addr, 0, args.rt1, args.rt2, scale, args.is_signed);
    }

    // Write back for pre-index and post-index.
    if (args.is_preindex) {
      SetRegOrSp(args.rn, addr);
    } else if (args.is_postindex) {
      Register new_base = listener_->AddImm(base, args.offset);
      SetRegOrSp(args.rn, new_base);
    }
  }

  void LoadStoreReg(const typename Decoder::LoadStoreRegArgs& args) {
    Register base = GetRegOrSp(args.rn);
    Register offset_reg = GetRegOrZero(args.rm);

    // Forward extend_type so the handler can apply
    // the correct 32->64 extension (UXTW vs SXTW vs LSL/SXTX) before
    // shift+add. Previously this collapsed to LSL and corrupted the
    // address whenever the offset W register had nonzero upper half or
    // SXTW was actually requested.
    if (args.is_store) {
      Register data = GetRegOrZero(args.rt);
      listener_->StoreReg(args.size, base, offset_reg, args.extend_type,
                          args.shift_amount, data);
    } else {
      Register result = listener_->LoadReg(args.size, args.is_signed, args.is_64bit_target,
                                           base, offset_reg, args.extend_type,
                                           args.shift_amount);
      SetRegOrIgnore(args.rt, result);
    }
  }

  void Svc(const typename Decoder::SvcArgs& args) {
    listener_->Svc(args.imm);
  }

  // BRK breakpoint (delivers SIGTRAP to the guest).
  void Brk(uint16_t imm) {
    listener_->Brk(imm);
  }

  void Mrs(const typename Decoder::MrsArgs& args) {
    Register result = listener_->Mrs(args.sysreg);
    SetRegOrIgnore(args.dst, result);
  }

  void Msr(const typename Decoder::MsrArgs& args) {
    Register src = GetRegOrZero(args.src);
    listener_->Msr(args.sysreg, src);
  }

  void LogicalShiftedReg(const typename Decoder::LogicalShiftedRegArgs& args) {
    Register src1 = GetRegOrZero(args.src1);
    Register src2 = GetRegOrZero(args.src2);
    Register result = listener_->LogicalShiftedReg(args.opcode, args.is_64bit, args.invert,
                                                   src1, src2, args.shift_type,
                                                   args.shift_amount);
    SetRegOrIgnore(args.dst, result);
  }

  void AddSubShiftedReg(const typename Decoder::AddSubShiftedRegArgs& args) {
    Register src1 = GetRegOrZero(args.src1);
    Register src2 = GetRegOrZero(args.src2);
    Register result = listener_->AddSubShiftedReg(args.is_sub, args.set_flags, args.is_64bit,
                                                  src1, src2, args.shift_type,
                                                  args.shift_amount);
    // Unlike the immediate/extended-register forms, the shifted-register form
    // never writes SP: Rd is the ZR form for both flag-setting and plain ops.
    SetRegOrIgnore(args.dst, result);
  }

  void AddSubExtendedReg(const typename Decoder::AddSubExtendedRegArgs& args) {
    Register src1 = GetRegOrSp(args.src1);
    Register src2 = GetRegOrZero(args.src2);
    Register result = listener_->AddSubExtendedReg(args.is_sub, args.set_flags, args.is_64bit,
                                                   src1, src2, args.extend_type,
                                                   args.shift_amount);
    if (args.set_flags) {
      SetRegOrIgnore(args.dst, result);
    } else {
      SetRegOrSp(args.dst, result);
    }
  }

  void ConditionalSelect(const typename Decoder::ConditionalSelectArgs& args) {
    Register src1 = GetRegOrZero(args.src1);
    Register src2 = GetRegOrZero(args.src2);
    Register result = listener_->ConditionalSelect(args.opcode, args.is_64bit,
                                                   src1, src2, args.cond);
    SetRegOrIgnore(args.dst, result);
  }

  void DataProc2Src(const typename Decoder::DataProc2SrcArgs& args) {
    Register src1 = GetRegOrZero(args.src1);
    Register src2 = GetRegOrZero(args.src2);
    Register result = listener_->DataProc2Src(args.opcode, args.is_64bit, src1, src2);
    SetRegOrIgnore(args.dst, result);
  }

  void DataProc3Src(const typename Decoder::DataProc3SrcArgs& args) {
    Register src1 = GetRegOrZero(args.src1);
    Register src2 = GetRegOrZero(args.src2);
    Register src3 = GetRegOrZero(args.src3);
    Register result = listener_->DataProc3Src(args.opcode, args.is_64bit, src1, src2, src3);
    SetRegOrIgnore(args.dst, result);
  }

  void Nop() { listener_->Nop(); }

  // DMB/DSB full barrier (SY/ISH/...): a StoreLoad fence the host must honor.
  void DataMemoryBarrier() { listener_->DataMemoryBarrier(); }

  // IC IVAU, Xt — invalidate the translation cache for the modified code line.
  void IcIvau(uint8_t rt) { listener_->IcIvau(rt); }

  void Undefined() { listener_->Undefined(); }

  // MTE DP-2src: IRG / GMI / SUBP / SUBPS. The listener owns the full
  // SP/XZR semantics + (for SUBPS) NZCV update, because those rules
  // differ between the four opcodes (e.g., IRG dst can be SP, the others
  // cannot). The pass-through here avoids having to expose all four
  // GetReg* / SetReg* variants from this template.
  void MteDataProc(const typename Decoder::MteDataProcArgs& args) {
    listener_->MteDataProc(args);
  }

  // MTE load/store memory tags. The listener owns SP/XZR rules + address
  // arithmetic + writeback because each opcode has slightly different
  // semantics and the SemanticsPlayer would otherwise need to expose four
  // variants (offset / pre / post / no-writeback) for one rare family.
  void MteLoadStore(const typename Decoder::MteLoadStoreArgs& args) {
    listener_->MteLoadStore(args);
  }

  // AdvSIMD complex floating-point (Armv8.3-FCMA): FCADD / FCMLA.
  // The listener owns the per-rot lane shuffling (FCADD's "rotate operand2
  // by ±90°" and FCMLA's four-rot table) since exposing every variant via
  // GetSimd*/SetSimd* would mean four method shapes on this template.
  void AdvSimdFcma(const typename Decoder::FcmaArgs& args) {
    listener_->AdvSimdFcma(args);
  }

  // indexed FCMLA
  // AdvSIMD complex floating-point by element (Armv8.3-FCMA): FCMLA-idx.
  // The listener owns broadcasting a single complex pair from Vm[index]
  // across every output pair; expressing the per-index broadcast through
  // GetSimd*/SetSimd* would require a separate by-element lane shape that
  // no other player path needs.
  void AdvSimdFcmaIdx(const typename Decoder::FcmaIdxArgs& args) {
    listener_->AdvSimdFcmaIdx(args);
  }

  // AdvSIMD BFloat16 three-same-extra (Armv8.6-BF16): BFDOT / BFMMLA.
  // The listener owns BF16-to-FP32 widening and the 2x4 * 4x2 matrix
  // iteration pattern because doing that through GetSimd*/SetSimd* would
  // force two distinct lane-layout templates onto this player.
  void AdvSimdBf16ThreeSame(const typename Decoder::Bf16ThreeSameArgs& args) {
    listener_->AdvSimdBf16ThreeSame(args);
  }

  // hello-dotprod
  // AdvSIMD integer dot product (Armv8.4-DotProd): SDOT / UDOT, vector
  // and by-element forms.  Lane layout (4 bytes packed per 32-bit lane)
  // and the per-form indexed broadcast belong in the listener since
  // expressing them through GetSimd*/SetSimd* would mean per-byte lane
  // accessors that no other player path needs.
  void AdvSimdDotProduct(const typename Decoder::DotProductArgs& args) {
    listener_->AdvSimdDotProduct(args);
  }

  // I8MM matrix multiply-accumulate (SMMLA/UMMLA/USMMLA).
  void AdvSimdMatMul(const typename Decoder::MatMulArgs& args) {
    listener_->AdvSimdMatMul(args);
  }

  void SimdModifiedImm(const typename Decoder::SimdModifiedImmArgs& args) {
    listener_->SimdModifiedImm(args);
  }

  void SimdLoadStoreImm(const typename Decoder::SimdLoadStoreImmArgs& args) {
    Register base = GetRegOrSp(args.rn);
    listener_->SimdLoadStoreImm(args, base);
  }

  // LDR (literal) — SIMD/FP form. The listener computes the address
  // from (insn_addr + offset) and loads `size` bytes into V[rt]
  // (32-bit S / 64-bit D / 128-bit Q), zero-extending the upper bits.
  void SimdLoadLiteral(const typename Decoder::SimdLoadLiteralArgs& args) {
    listener_->SimdLoadLiteral(args);
  }

  void SimdLoadStoreImmPreIndex(const typename Decoder::SimdLoadStoreImmArgs& args) {
    Register base = GetRegOrSp(args.rn);
    Register new_base = listener_->AddImm(base, args.offset);
    listener_->SimdLoadStoreImm(
        {.rt = args.rt, .rn = args.rn, .offset = 0, .size = args.size, .is_store = args.is_store},
        new_base);
    SetRegOrSp(args.rn, new_base);
  }

  void SimdLoadStoreImmPostIndex(const typename Decoder::SimdLoadStoreImmArgs& args) {
    Register base = GetRegOrSp(args.rn);
    listener_->SimdLoadStoreImm(
        {.rt = args.rt, .rn = args.rn, .offset = 0, .size = args.size, .is_store = args.is_store},
        base);
    Register new_base = listener_->AddImm(base, args.offset);
    SetRegOrSp(args.rn, new_base);
  }

  void SimdLoadStorePair(const typename Decoder::SimdLoadStorePairArgs& args) {
    Register base = GetRegOrSp(args.rn);

    Register addr = base;
    if (args.is_preindex || !args.is_postindex) {
      // Pre-index or signed-offset: apply offset before access.
      addr = listener_->AddImm(base, args.offset);
    }

    listener_->SimdLoadStorePair(args, addr);

    if (args.is_preindex) {
      SetRegOrSp(args.rn, addr);
    } else if (args.is_postindex) {
      Register new_base = listener_->AddImm(base, args.offset);
      SetRegOrSp(args.rn, new_base);
    }
  }

  void SimdLoadStoreReg(const typename Decoder::SimdLoadStoreRegArgs& args) {
    Register base = GetRegOrSp(args.rn);
    Register offset_reg = GetRegOrZero(args.rm);
    listener_->SimdLoadStoreReg(args, base, offset_reg);
  }

  void FpIntConversion(const typename Decoder::FpIntConvArgs& args) {
    listener_->FpIntConversion(args);
  }

  void FpMovImmediate(uint8_t rd, uint8_t imm8, uint8_t ftype) {
    listener_->FpMovImmediate(rd, imm8, ftype);
  }

  void FpDataProc3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra,
                   uint8_t ftype, bool o1, bool o0) {
    listener_->FpDataProc3(rd, rn, rm, ra, ftype, o1, o0);
  }

  void FpCondSelect(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ftype,
                    typename Decoder::Condition cond) {
    listener_->FpCondSelect(rd, rn, rm, ftype, cond);
  }

  void FpFixedPointConversion(const typename Decoder::FpFixedPointArgs& args) {
    listener_->FpFixedPointConversion(args);
  }

  void AdvSimdCopy(const typename Decoder::AdvSimdCopyArgs& args) {
    listener_->AdvSimdCopy(args);
  }

  void AdvSimdThreeSame(const typename Decoder::AdvSimdThreeSameArgs& args) {
    listener_->AdvSimdThreeSame(args);
  }

  void AdvSimdThreeDiff(const typename Decoder::AdvSimdThreeDiffArgs& args) {
    listener_->AdvSimdThreeDiff(args);
  }

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    listener_->AdvSimdExtract(rd, rn, rm, index, q);
  }

  void AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size,
                      uint8_t opcode, bool q) {
    listener_->AdvSimdPermute(rd, rn, rm, size, opcode, q);
  }

  void AdvSimdTableLookup(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t len,
                          uint8_t op, bool q) {
    listener_->AdvSimdTableLookup(rd, rn, rm, len, op, q);
  }

  void Sha512(typename Decoder::Sha512Op op, uint8_t rd, uint8_t rn,
              uint8_t rm) {
    listener_->Sha512(op, rd, rn, rm);
  }

  // SHA3 (FEAT_SHA3).
  void Eor3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    listener_->Eor3(rd, rn, rm, ra);
  }
  void Bcax(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    listener_->Bcax(rd, rn, rm, ra);
  }
  void Rax1(uint8_t rd, uint8_t rn, uint8_t rm) {
    listener_->Rax1(rd, rn, rm);
  }
  void Xar(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm6) {
    listener_->Xar(rd, rn, rm, imm6);
  }

  // SM4 (FEAT_SM4).
  void Sm4e(uint8_t rd, uint8_t rn) {
    listener_->Sm4e(rd, rn);
  }
  void Sm4ekey(uint8_t rd, uint8_t rn, uint8_t rm) {
    listener_->Sm4ekey(rd, rn, rm);
  }

  // SM3 (FEAT_SM3).
  void Sm3ss1(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    listener_->Sm3ss1(rd, rn, rm, ra);
  }
  void Sm3tt(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm2, uint8_t op) {
    listener_->Sm3tt(rd, rn, rm, imm2, op);
  }
  void Sm3partw1(uint8_t rd, uint8_t rn, uint8_t rm) {
    listener_->Sm3partw1(rd, rn, rm);
  }
  void Sm3partw2(uint8_t rd, uint8_t rn, uint8_t rm) {
    listener_->Sm3partw2(rd, rn, rm);
  }

  void CryptoAes(uint8_t rd, uint8_t rn, uint8_t opcode) {
    listener_->CryptoAes(rd, rn, opcode);
  }

  void CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode) {
    listener_->CryptoSha3Reg(rd, rn, rm, opcode);
  }
  void CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode) {
    listener_->CryptoSha2Reg(rd, rn, opcode);
  }

  void AdvSimdMultiStruct(uint8_t rt, uint8_t rn, uint8_t num_regs, uint8_t size,
                          bool q, bool is_store, bool postindex, uint8_t rm,
                          bool is_interleaved) {
    listener_->AdvSimdMultiStruct(rt, rn, num_regs, size, q, is_store, postindex, rm,
                                  is_interleaved);
  }

  void AdvSimdSingleStruct(const typename Decoder::AdvSimdSingleStructArgs& args) {
    listener_->AdvSimdSingleStruct(args);
  }

  void AddSubWithCarry(uint8_t rd, uint8_t rn, uint8_t rm, bool is_64bit, bool is_sub, bool set_flags) {
    Register src1 = GetRegOrZero(rn);
    Register src2 = GetRegOrZero(rm);
    Register result = listener_->AddSubWithCarry(src1, src2, is_64bit, is_sub, set_flags);
    SetRegOrIgnore(rd, result);
  }

  void DataProc1Src(uint8_t rd, uint8_t rn, uint8_t opcode2, bool is_64bit) {
    Register src = GetRegOrZero(rn);
    Register result = listener_->DataProc1Src(src, opcode2, is_64bit);
    SetRegOrIgnore(rd, result);
  }

  void Extr(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t lsb, bool is_64bit) {
    Register src_n = GetRegOrZero(rn);
    Register src_m = GetRegOrZero(rm);
    Register result = listener_->Extr(src_n, src_m, lsb, is_64bit);
    SetRegOrIgnore(rd, result);
  }

  void ConditionalCompare(const typename Decoder::ConditionalCompareArgs& args) {
    Register rn = GetRegOrZero(args.rn);
    Register rm = args.is_imm ? listener_->GetImm(args.rm_or_imm) : GetRegOrZero(args.rm_or_imm);
    listener_->ConditionalCompare(args.is_neg, args.is_64bit, rn, rm, args.cond, args.nzcv);
  }

  void LoadStoreExclusive(const typename Decoder::LoadStoreExclusiveArgs& args) {
    Register base = GetRegOrSp(args.rn);
    listener_->LoadStoreExclusive(args, base);
  }

  void FpDataProc1(const typename Decoder::FpDataProc1Args& args) {
    listener_->FpDataProc1(args);
  }

  void FpDataProc2(const typename Decoder::FpDataProc2Args& args) {
    listener_->FpDataProc2(args);
  }

  void FpCompare(const typename Decoder::FpCompareArgs& args) {
    listener_->FpCompare(args);
  }

  void FpConditionalCompare(const typename Decoder::FpConditionalCompareArgs& args) {
    listener_->FpConditionalCompare(args);
  }

  void AdvSimdTwoRegMisc(const typename Decoder::AdvSimdTwoRegMiscArgs& args) {
    listener_->AdvSimdTwoRegMisc(args);
  }

  void AdvSimdScalarTwoRegMisc(const typename Decoder::AdvSimdScalarTwoRegMiscArgs& args) {
    listener_->AdvSimdScalarTwoRegMisc(args);
  }

  void AdvSimdScalarThreeSame(const typename Decoder::AdvSimdScalarThreeSameArgs& args) {
    listener_->AdvSimdScalarThreeSame(args);
  }

  void AdvSimdScalarPairwise(const typename Decoder::AdvSimdScalarPairwiseArgs& args) {
    listener_->AdvSimdScalarPairwise(args);
  }

  void AdvSimdShiftByImm(const typename Decoder::AdvSimdShiftImmArgs& args) {
    listener_->AdvSimdShiftByImm(args);
  }

  void AdvSimdVecXIndexedElement(const typename Decoder::AdvSimdVecXIdxArgs& args) {
    listener_->AdvSimdVecXIndexedElement(args);
  }

  void AdvSimdScalarXIndexedElement(const typename Decoder::AdvSimdScalarXIdxArgs& args) {
    listener_->AdvSimdScalarXIndexedElement(args);
  }

 private:
  // ARM64: register 31 as zero register.
  Register GetRegOrZero(uint8_t reg) {
    return reg == 31 ? listener_->GetImm(0) : listener_->GetReg(reg);
  }

  // ARM64: register 31 as stack pointer.
  Register GetRegOrSp(uint8_t reg) {
    return reg == 31 ? listener_->GetSp() : listener_->GetReg(reg);
  }

  // ARM64: register 31 as zero register (writes to reg 31 are discarded).
  void SetRegOrIgnore(uint8_t reg, Register value) {
    if (reg != 31) {
      listener_->SetReg(reg, value);
    }
  }

  // ARM64: register 31 as stack pointer.
  void SetRegOrSp(uint8_t reg, Register value) {
    if (reg == 31) {
      listener_->SetSp(value);
    } else {
      listener_->SetReg(reg, value);
    }
  }

  SemanticsListener* listener_;
};

}  // namespace berberis

#endif  // BERBERIS_DECODER_ARM64_SEMANTICS_PLAYER_H_
