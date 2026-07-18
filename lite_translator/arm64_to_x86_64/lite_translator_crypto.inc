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

// Textual fragment of class LiteTranslator; included by lite_translator.h
// inside the class body. Not a standalone header -- no include guards on
// purpose (no #ifndef / #include / namespace). See lite_translator.h.
//
// Family: cryptographic extension (SHA/SM3/SM4/AES).
// Methods:
//   Sha512, Eor3, Bcax, Rax1, Xar, Sm4e, Sm4ekey, Sm3ss1, Sm3tt, Sm3partw1,
//   Sm3partw2, CryptoAes, CryptoSha3Reg, CryptoSha2Reg

  void Sha512(Decoder::Sha512Op op, uint8_t rd, uint8_t rn, uint8_t rm) {
    UNUSED(op, rd, rn, rm);
    Undefined();
  }

  // SHA3 (FEAT_SHA3): interpreter-only — bail.
  void Eor3(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    UNUSED(rd, rn, rm, ra);
    Undefined();
  }
  void Bcax(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    UNUSED(rd, rn, rm, ra);
    Undefined();
  }
  void Rax1(uint8_t rd, uint8_t rn, uint8_t rm) {
    UNUSED(rd, rn, rm);
    Undefined();
  }
  void Xar(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm6) {
    UNUSED(rd, rn, rm, imm6);
    Undefined();
  }

  // SM3/SM4 (interpreter only).
  void Sm4e(uint8_t rd, uint8_t rn) {
    UNUSED(rd, rn);
    Undefined();
  }
  void Sm4ekey(uint8_t rd, uint8_t rn, uint8_t rm) {
    UNUSED(rd, rn, rm);
    Undefined();
  }
  void Sm3ss1(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra) {
    UNUSED(rd, rn, rm, ra);
    Undefined();
  }
  void Sm3tt(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t imm2, uint8_t op) {
    UNUSED(rd, rn, rm, imm2, op);
    Undefined();
  }
  void Sm3partw1(uint8_t rd, uint8_t rn, uint8_t rm) {
    UNUSED(rd, rn, rm);
    Undefined();
  }
  void Sm3partw2(uint8_t rd, uint8_t rn, uint8_t rm) {
    UNUSED(rd, rn, rm);
    Undefined();
  }

  // AESE/AESD/AESMC/AESIMC via host AES-NI. The ARM ops map to AES-NI as:
  //   AESE  Vd = ShiftRows(SubBytes(Vd^Vn))       = AESENCLAST(Vd^Vn, 0)
  //   AESD  Vd = InvShiftRows(InvSubBytes(Vd^Vn)) = AESDECLAST(Vd^Vn, 0)
  //   AESMC Vd = MixColumns(Vn)                   = AESENC(AESDECLAST(Vn, 0), 0)
  //        (SubBytes/InvSubBytes and ShiftRows/InvShiftRows cancel, leaving Mix.)
  //   AESIMC Vd = InvMixColumns(Vn)               = AESIMC(Vn)
  // opcode: 00=AESE, 01=AESD, 10=AESMC, 11=AESIMC. Bail to the interpreter if the
  // host lacks AES-NI.
  void CryptoAes(uint8_t rd, uint8_t rn, uint8_t opcode) {
    if (!host_platform::kHasAES) { Undefined(); return; }
    const int32_t vd_off = VRegOffset(rd);
    const int32_t vn_off = VRegOffset(rn);
    SimdRegister xd = AllocTempSimdReg();
    SimdRegister xzero = AllocTempSimdReg();
    if (!success()) { return; }
    as_.Pxor(xzero, xzero);  // zero round key / round constant
    switch (opcode) {
      case 0b00:    // AESE
      case 0b01: {  // AESD
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vd_off});
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Pxor(xd, xn);
        if (opcode == 0b00) {
          as_.Aesenclast(xd, xzero);
        } else {
          as_.Aesdeclast(xd, xzero);
        }
        break;
      }
      case 0b10:    // AESMC = AESENC(AESDECLAST(Vn, 0), 0)
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = vn_off});
        as_.Aesdeclast(xd, xzero);
        as_.Aesenc(xd, xzero);
        break;
      default: {    // 0b11 AESIMC = AESIMC(Vn)
        SimdRegister xn = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xn, {.base = Assembler::rbp, .disp = vn_off});
        as_.Aesimc(xd, xn);
        break;
      }
    }
    as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xd);
  }

  void CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode) {
    UNUSED(rd, rn, rm, opcode);
    Undefined();  // interpreter fallback
  }

  void CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode) {
    UNUSED(rd, rn, opcode);
    Undefined();  // interpreter fallback
  }

