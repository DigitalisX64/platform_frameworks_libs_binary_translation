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

    // region digitalis
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
    // endregion

    uint8_t scale = (args.size == Decoder::LoadStoreSize::k64bit) ? 8 : 4;

    if (args.is_store) {
      Register data1 = GetRegOrZero(args.rt1);
      Register data2 = GetRegOrZero(args.rt2);
      listener_->StorePair(args.size, addr, 0, data1, data2, scale);
    } else {
      listener_->LoadPair(args.size, addr, 0, args.rt1, args.rt2, scale);
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

    // region digitalis - Forward extend_type so the handler can apply
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
    // endregion
  }

  void Svc(const typename Decoder::SvcArgs& args) {
    listener_->Svc(args.imm);
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
    if (args.set_flags) {
      SetRegOrIgnore(args.dst, result);
    } else {
      SetRegOrIgnore(args.dst, result);
    }
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

  void Undefined() { listener_->Undefined(); }

  // region digitalis
  void SimdModifiedImm(const typename Decoder::SimdModifiedImmArgs& args) {
    listener_->SimdModifiedImm(args);
  }

  void SimdLoadStoreImm(const typename Decoder::SimdLoadStoreImmArgs& args) {
    Register base = GetRegOrSp(args.rn);
    listener_->SimdLoadStoreImm(args, base);
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

  // region digitalis
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
  // endregion

  void AdvSimdCopy(const typename Decoder::AdvSimdCopyArgs& args) {
    listener_->AdvSimdCopy(args);
  }

  void AdvSimdThreeSame(const typename Decoder::AdvSimdThreeSameArgs& args) {
    listener_->AdvSimdThreeSame(args);
  }

  // region digitalis
  void AdvSimdThreeDiff(const typename Decoder::AdvSimdThreeDiffArgs& args) {
    listener_->AdvSimdThreeDiff(args);
  }
  // endregion

  void AdvSimdExtract(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t index, bool q) {
    listener_->AdvSimdExtract(rd, rn, rm, index, q);
  }

  // region digitalis
  void AdvSimdPermute(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t size,
                      uint8_t opcode, bool q) {
    listener_->AdvSimdPermute(rd, rn, rm, size, opcode, q);
  }
  // endregion

  // region digitalis
  void AdvSimdMultiStruct(uint8_t rt, uint8_t rn, uint8_t num_regs, uint8_t size,
                          bool q, bool is_store, bool postindex, uint8_t rm,
                          bool is_interleaved) {
    listener_->AdvSimdMultiStruct(rt, rn, num_regs, size, q, is_store, postindex, rm,
                                  is_interleaved);
  }

  void AdvSimdSingleStruct(const typename Decoder::AdvSimdSingleStructArgs& args) {
    listener_->AdvSimdSingleStruct(args);
  }
  // endregion

  // region digitalis
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
  // endregion

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

  void AdvSimdTwoRegMisc(const typename Decoder::AdvSimdTwoRegMiscArgs& args) {
    listener_->AdvSimdTwoRegMisc(args);
  }

  // region digitalis
  void AdvSimdScalarTwoRegMisc(const typename Decoder::AdvSimdScalarTwoRegMiscArgs& args) {
    listener_->AdvSimdScalarTwoRegMisc(args);
  }
  // endregion

  void AdvSimdShiftByImm(const typename Decoder::AdvSimdShiftImmArgs& args) {
    listener_->AdvSimdShiftByImm(args);
  }
  // endregion

  // region digitalis
  void AdvSimdVecXIndexedElement(const typename Decoder::AdvSimdVecXIdxArgs& args) {
    listener_->AdvSimdVecXIndexedElement(args);
  }
  // endregion

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
// endregion
