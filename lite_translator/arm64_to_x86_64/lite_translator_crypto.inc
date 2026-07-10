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

  void CryptoAes(uint8_t rd, uint8_t rn, uint8_t opcode) {
    UNUSED(rd, rn, opcode);
    Undefined();  // interpreter fallback
  }

  void CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode) {
    UNUSED(rd, rn, rm, opcode);
    Undefined();  // interpreter fallback
  }

  void CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode) {
    UNUSED(rd, rn, opcode);
    Undefined();  // interpreter fallback
  }

