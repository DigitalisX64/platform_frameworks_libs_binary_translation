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

  // SHA-256 (H/H2/SU0/SU1). The math mirrors the interpreter exactly (its
  // differential oracle). Because the lite temp allocator never frees a temp
  // mid-instruction, the 8-word working state for H/H2 cannot live in GP
  // registers; it lives in ThreadState.intrinsics_scratch_area (32 bytes = 8
  // uint32) and the round math reuses a fixed handful of GP temps via the
  // reg-op-[mem] forms. Whole 128-bit V-register loads/stores keep every
  // V-access at one width (no sub-lane write that a wider read could see
  // stale). ch/maj use the NOT-free identities ch=g^(e&(f^g)),
  // maj=(a&b)^(c&(a^b)); ROL(x,n) is ROR(x,32-n).
  //
  // Lowering these keeps crypto-heavy transport loops (TLS/QUIC per-packet
  // auth) from fragmenting into per-instruction interpreter round-trips.
  static constexpr int32_t kShaScr = offsetof(ThreadState, intrinsics_scratch_area);

  // Emits `dst = ROR(mem32[disp], n)`.
  void ShaRor(Register dst, int32_t disp, int8_t n) {
    as_.Movl(dst, {.base = Assembler::rbp, .disp = disp});
    if (n != 0) as_.Rorl(dst, n);
  }
  // acc = σ (rotr(a)^rotr(b)^third) where `third` is rotr(c) or shr; the source
  // word is at [disp]. Uses acc and one scratch temp `t`.
  void ShaBigSigma(Register acc, Register t, int32_t disp, int8_t r0, int8_t r1, int8_t r2) {
    ShaRor(acc, disp, r0);
    ShaRor(t, disp, r1);
    as_.Xorl(acc, t);
    ShaRor(t, disp, r2);
    as_.Xorl(acc, t);
  }
  void ShaLittleSigma(Register acc, Register t, int32_t disp, int8_t r0, int8_t r1, int8_t shr) {
    ShaRor(acc, disp, r0);
    ShaRor(t, disp, r1);
    as_.Xorl(acc, t);
    as_.Movl(t, {.base = Assembler::rbp, .disp = disp});
    as_.Shrl(t, shr);
    as_.Xorl(acc, t);
  }

  void CryptoSha3Reg(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t opcode) {
    if (opcode > 0b110) { Undefined(); return; }
    const int32_t vd_off = VRegOffset(rd);
    const int32_t vn_off = VRegOffset(rn);
    const int32_t vm_off = VRegOffset(rm);
    SimdRegister xtmp = AllocTempSimdReg();
    Register r0 = AllocTempReg(), r1 = AllocTempReg(), r2 = AllocTempReg(),
             r3 = AllocTempReg(), r4 = AllocTempReg();
    if (!success() || r0 == no_register || r1 == no_register || r2 == no_register ||
        r3 == no_register || r4 == no_register) {
      success_ = false;
      return;
    }
    auto m = [](int32_t off) { return Assembler::Operand{.base = Assembler::rbp, .disp = off}; };

    if (opcode == 0b011) {
      // SHA1SU0: t0 = d2^d0^m0; t1 = d3^d1^m1; t2 = n0^d2^m2; t3 = n1^d3^m3.
      // All inputs are read before any lane of Vd is written, so aliased
      // rd==rn / rd==rm stay correct.
      as_.Movl(r0, m(vd_off + 8));
      as_.Xorl(r0, m(vd_off + 0));
      as_.Xorl(r0, m(vm_off + 0));
      as_.Movl(r1, m(vd_off + 12));
      as_.Xorl(r1, m(vd_off + 4));
      as_.Xorl(r1, m(vm_off + 4));
      as_.Movl(r2, m(vn_off + 0));
      as_.Xorl(r2, m(vd_off + 8));
      as_.Xorl(r2, m(vm_off + 8));
      as_.Movl(r3, m(vn_off + 4));
      as_.Xorl(r3, m(vd_off + 12));
      as_.Xorl(r3, m(vm_off + 12));
      as_.Movl(m(vd_off + 0), r0);
      as_.Movl(m(vd_off + 4), r1);
      as_.Movl(m(vd_off + 8), r2);
      as_.Movl(m(vd_off + 12), r3);
      return;
    }
    if (opcode <= 0b010) {
      // SHA1C (000) / SHA1P (001) / SHA1M (010): four SHA-1 rounds.
      // {a,b,c,d} = Vd lanes in r0..r3, e = Sn (Vn lane 0) in r4, f/t staged
      // in one more temp plus one scratch word. Per round:
      //   f = Ch/Parity/Maj of (b,c,d);  t = ROL(a,5) + f + e + w[j];
      //   e=d; d=c; c=ROL(b,30); b=a; a=t.
      Register r5 = AllocTempReg();
      if (r5 == no_register) { success_ = false; return; }
      as_.Movl(r0, m(vd_off + 0));
      as_.Movl(r1, m(vd_off + 4));
      as_.Movl(r2, m(vd_off + 8));
      as_.Movl(r3, m(vd_off + 12));
      as_.Movl(r4, m(vn_off + 0));
      for (int j = 0; j < 4; j++) {
        // f -> r5.
        if (opcode == 0b000) {
          // Ch: d ^ (b & (c ^ d)).
          as_.Movl(r5, r2);
          as_.Xorl(r5, r3);
          as_.Andl(r5, r1);
          as_.Xorl(r5, r3);
        } else if (opcode == 0b001) {
          // Parity: b ^ c ^ d.
          as_.Movl(r5, r1);
          as_.Xorl(r5, r2);
          as_.Xorl(r5, r3);
        } else {
          // Maj: (b & c) | (d & (b | c)).
          as_.Movl(r5, r1);
          as_.Orl(r5, r2);
          as_.Andl(r5, r3);
          as_.Movl(m(kShaScr), r5);
          as_.Movl(r5, r1);
          as_.Andl(r5, r2);
          as_.Orl(r5, m(kShaScr));
        }
        // t = f + e + w[j] + ROL(a,5) -> r5.
        as_.Addl(r5, r4);
        as_.Addl(r5, m(vm_off + j * 4));
        as_.Movl(m(kShaScr), r5);
        as_.Movl(r5, r0);
        as_.Roll(r5, int8_t{5});
        as_.Addl(r5, m(kShaScr));
        // Rotate state.
        as_.Movl(r4, r3);           // e = d
        as_.Movl(r3, r2);           // d = c
        as_.Movl(r2, r1);
        as_.Roll(r2, int8_t{30});   // c = ROL(b, 30)
        as_.Movl(r1, r0);           // b = a
        as_.Movl(r0, r5);           // a = t
      }
      as_.Movl(m(vd_off + 0), r0);
      as_.Movl(m(vd_off + 4), r1);
      as_.Movl(m(vd_off + 8), r2);
      as_.Movl(m(vd_off + 12), r3);
      return;
    }

    if (opcode == 0b110) {
      // SHA256SU1: nd0=d0+σ1(m2)+n1; nd1=d1+σ1(m3)+n2;
      //            nd2=d2+σ1(nd0)+n3; nd3=d3+σ1(nd1)+m0. nd0/nd1 are consumed
      //            by nd2/nd3, so stage d[] in scratch[0..3] and overwrite.
      as_.Movdqu(xtmp, m(vd_off));
      as_.Movdqu(m(kShaScr), xtmp);  // scratch[0..3] = d[]
      // nd0 = d0 + σ1(m2) + n1.
      ShaLittleSigma(r0, r1, vm_off + 8, 17, 19, 10);
      as_.Addl(r0, m(kShaScr + 0));
      as_.Addl(r0, m(vn_off + 4));
      // nd1 = d1 + σ1(m3) + n2.
      ShaLittleSigma(r2, r1, vm_off + 12, 17, 19, 10);
      as_.Addl(r2, m(kShaScr + 4));
      as_.Addl(r2, m(vn_off + 8));
      as_.Movl(m(kShaScr + 0), r0);  // scratch[0] = nd0 (σ1 input for nd2)
      as_.Movl(m(kShaScr + 4), r2);  // scratch[1] = nd1
      // nd2 = d2 + σ1(nd0) + n3.
      ShaLittleSigma(r3, r1, kShaScr + 0, 17, 19, 10);
      as_.Addl(r3, m(kShaScr + 8));
      as_.Addl(r3, m(vn_off + 12));
      // nd3 = d3 + σ1(nd1) + m0.
      ShaLittleSigma(r4, r1, kShaScr + 4, 17, 19, 10);
      as_.Addl(r4, m(kShaScr + 12));
      as_.Addl(r4, m(vm_off + 0));
      as_.Movl(m(kShaScr + 8), r3);
      as_.Movl(m(kShaScr + 12), r4);
      as_.Movdqu(xtmp, m(kShaScr));
      as_.Movdqu(m(vd_off), xtmp);
      return;
    }

    // SHA256H (100): X=Vd, Y=Vn; SHA256H2 (101): X=Vn, Y=Vd.
    const int32_t vx = (opcode == 0b100) ? vd_off : vn_off;
    const int32_t vy = (opcode == 0b100) ? vn_off : vd_off;
    const int32_t X = kShaScr;       // x[0..3] at X+0,4,8,12
    const int32_t Y = kShaScr + 16;  // y[0..3] at Y+0,4,8,12
    as_.Movdqu(xtmp, m(vx));
    as_.Movdqu(m(X), xtmp);
    as_.Movdqu(xtmp, m(vy));
    as_.Movdqu(m(Y), xtmp);
    for (int e = 0; e < 4; e++) {
      // ch = y2 ^ (y0 & (y1 ^ y2))  -> r0.
      as_.Movl(r0, m(Y + 4));
      as_.Xorl(r0, m(Y + 8));
      as_.Andl(r0, m(Y + 0));
      as_.Xorl(r0, m(Y + 8));
      // t = y3 + Σ1(y0) + ch + w[e]  -> r1.
      ShaBigSigma(r1, r2, Y + 0, 6, 11, 25);
      as_.Addl(r1, r0);
      as_.Addl(r1, m(Y + 12));
      as_.Addl(r1, m(vm_off + e * 4));
      // new_x3 = t + x3  -> r0.
      as_.Movl(r0, r1);
      as_.Addl(r0, m(X + 12));
      // bs0 = Σ0(x0)  -> r2.
      ShaBigSigma(r2, r3, X + 0, 2, 13, 22);
      // maj = (x0 & x1) ^ (x2 & (x0 ^ x1))  -> r3.
      as_.Movl(r3, m(X + 0));
      as_.Xorl(r3, m(X + 4));
      as_.Andl(r3, m(X + 8));
      as_.Movl(r4, m(X + 0));
      as_.Andl(r4, m(X + 4));
      as_.Xorl(r3, r4);
      // new_y3 = t + bs0 + maj  -> r1.
      as_.Addl(r1, r2);
      as_.Addl(r1, r3);
      // Rotate state (read high->low before overwriting): now r0=new_x3, r1=new_y3.
      as_.Movl(r2, m(X + 8)); as_.Movl(m(X + 12), r2);
      as_.Movl(r2, m(X + 4)); as_.Movl(m(X + 8), r2);
      as_.Movl(r2, m(X + 0)); as_.Movl(m(X + 4), r2);
      as_.Movl(m(X + 0), r1);  // x[0] = new_y3
      as_.Movl(r2, m(Y + 8)); as_.Movl(m(Y + 12), r2);
      as_.Movl(r2, m(Y + 4)); as_.Movl(m(Y + 8), r2);
      as_.Movl(r2, m(Y + 0)); as_.Movl(m(Y + 4), r2);
      as_.Movl(m(Y + 0), r0);  // y[0] = new_x3
    }
    as_.Movdqu(xtmp, m((opcode == 0b100) ? X : Y));
    as_.Movdqu(m(vd_off), xtmp);
  }

  void CryptoSha2Reg(uint8_t rd, uint8_t rn, uint8_t opcode) {
    if (opcode > 0b10) { Undefined(); return; }
    const int32_t vd_off = VRegOffset(rd);
    const int32_t vn_off = VRegOffset(rn);
    SimdRegister xtmp = AllocTempSimdReg();
    Register r0 = AllocTempReg(), r1 = AllocTempReg(), r2 = AllocTempReg();
    if (!success() || r0 == no_register || r1 == no_register || r2 == no_register) {
      success_ = false;
      return;
    }
    auto m = [](int32_t off) { return Assembler::Operand{.base = Assembler::rbp, .disp = off}; };

    if (opcode == 0b00) {
      // SHA1H: Sd = ROL(Sn, 30), upper 96 bits of Vd zeroed.
      as_.Movl(r0, m(vn_off + 0));
      as_.Roll(r0, int8_t{30});
      as_.Pxor(xtmp, xtmp);
      as_.Movdqu(m(vd_off), xtmp);
      as_.Movl(m(vd_off + 0), r0);
      return;
    }
    if (opcode == 0b01) {
      // SHA1SU1: t_i = ROL(d_i ^ n_{i+1}, 1) for i = 0..2;
      //          t3  = ROL(d3 ^ t0, 1)  (t0 is the already-rotated word).
      // All Vd reads happen before any Vd write, so rd==rn stays correct.
      as_.Movl(r0, m(vd_off + 0));
      as_.Xorl(r0, m(vn_off + 4));
      as_.Roll(r0, int8_t{1});
      as_.Movl(r1, m(vd_off + 4));
      as_.Xorl(r1, m(vn_off + 8));
      as_.Roll(r1, int8_t{1});
      as_.Movl(r2, m(vd_off + 8));
      as_.Xorl(r2, m(vn_off + 12));
      as_.Roll(r2, int8_t{1});
      // t3 uses d3 and the rotated t0; read d3 before storing t0.
      as_.Movl(m(kShaScr), r2);
      as_.Movl(r2, m(vd_off + 12));
      as_.Xorl(r2, r0);
      as_.Roll(r2, int8_t{1});
      as_.Movl(m(vd_off + 0), r0);
      as_.Movl(m(vd_off + 4), r1);
      as_.Movl(m(vd_off + 12), r2);
      as_.Movl(r2, m(kShaScr));
      as_.Movl(m(vd_off + 8), r2);
      return;
    }
    // out_i = d_i + σ0(d_{i+1}); d_4 = Vn[0]. Stage d[] at scratch[0..3] and
    // Vn[0] at scratch[4]. Each out_i is written in place at scratch[i] AFTER
    // reading its two inputs (σ0(d_{i+1}) at [i+1], +d_i at [i]); d_i is last
    // read here, so the in-place store is safe and stays within the 32-byte
    // scratch area.
    as_.Movdqu(xtmp, m(vd_off));
    as_.Movdqu(m(kShaScr), xtmp);          // scratch[0..3] = d[]
    as_.Movl(r2, m(vn_off + 0));
    as_.Movl(m(kShaScr + 16), r2);         // scratch[4] = Vn[0] (== d_4)
    for (int i = 0; i < 4; i++) {
      ShaLittleSigma(r0, r1, kShaScr + (i + 1) * 4, 7, 18, 3);
      as_.Addl(r0, m(kShaScr + i * 4));
      as_.Movl(m(kShaScr + i * 4), r0);    // out_i in place
    }
    as_.Movdqu(xtmp, m(kShaScr));
    as_.Movdqu(m(vd_off), xtmp);
  }

