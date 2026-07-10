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
// Family: SIMD complex-FP / dot-product / modified-immediate / SIMD load-store.
// Methods:
//   AdvSimdFcma, AdvSimdFcmaIdx, AdvSimdBf16ThreeSame, AdvSimdMatMul,
//   AdvSimdDotProduct, VFPExpandImm32Jit, VFPExpandImm64Jit,
//   ExpandSimdModifiedImmJit, SimdModifiedImm, SimdLoadLiteral,
//   SimdLoadStoreImm, SimdLoadStorePair, SimdLoadStoreReg

  // FCADD/FCMLA JIT
  //
  // AdvSIMD complex floating-point (FCADD / FCMLA) JIT path for FP32
  // and FP64.  The interpreter (interpreter.h::AdvSimdFcma) is the
  // spec — see the per-pair scalar math there for the six rotations
  // (FCADD ±90/±270, FCMLA 0/90/180/270).  The JIT path here lowers
  // FP32 (size=0b10, .2S Q=0 and .4S Q=1) and FP64 (size=0b11, .2D
  // Q=1) to a 4-to-7 SSE-instruction sequence:
  //
  //   * FCADD .4S, rot=#rot:
  //       Vm' = shufps(Vm, Vm, 0xB1)           // pair-swap (re,im)->(im,re)
  //       sign = pslld(pcmpeqd self, 31)       // [0x80000000]*4
  //       lane = psrlq(pcmpeqd self, 32)       // rot=0:  [-1,0,-1,0] (negate real)
  //              psllq(pcmpeqd self, 32)       // rot=1:  [0,-1,0,-1] (negate imag)
  //       sign &= lane
  //       Vm' ^= sign                          // negate appropriate lanes
  //       Vd  = Vn + Vm'                       // ADDPS
  //
  //   * FCMLA .4S, rot=#rot:  Vd += n_broadcast * m_xformed where
  //       rot=#0   (rot=0): xform=identity,      broadcast=n_re
  //       rot=#90  (rot=1): xform=swap+~re,      broadcast=n_im
  //       rot=#180 (rot=2): xform=negate-all,    broadcast=n_re
  //       rot=#270 (rot=3): xform=swap+~im,      broadcast=n_im
  //     n_re_broadcast = pshufd(Vn, 0xA0) = [n_re0, n_re0, n_re1, n_re1]
  //     n_im_broadcast = pshufd(Vn, 0xF5) = [n_im0, n_im0, n_im1, n_im1]
  //
  //   * Q=0 form (.2S): same emit; the upper 64 bits are masked away with
  //     psrldq+pand at the end (matches AArch64 vector half-vector semantics
  //     and the interpreter's Q=0 zero-clear at the end of AdvSimdFcma).
  //
  // FP16 (size=0b01) and FP64 (size=0b11) bail to the interpreter via
  // `success_ = false` — the interpreter handles them through the same
  // FpHalfToSingle round-trip / double-precision arithmetic paths used by
  // the rest of. When AVX-512-FP16 (or F16C) is unconditionally
  // available on the emulator host CPU, the FP16 path becomes a 2-step
  // VCVTPH2PS round-trip; that's a follow-up perf row.
  void AdvSimdFcma(const Decoder::FcmaArgs& args) {
    if (args.size == 0b01) {
      // FP16 vector FCMA JIT
      //
      // Lower FP16 FCADD/FCMLA via F16C round-trip: widen each operand
      // half (4 FP16 lanes = 2 complex pairs) to FP32, run the FP32
      // FCMA lowering, narrow back to FP16. For .4H (Q=0) one pass
      // over the low 64 bits of each operand suffices; for .8H (Q=1)
      // we run the FP32 core twice (low + high halves), save the
      // low-half FP16 result in a temp, then recombine via PSLLDQ +
      // POR. Bit-exact for non-FMA FCADD/FCMLA semantics because
      // FP32's 24-bit mantissa contains FP16's 11; matches the
      // interpreter's per-pair FpHalfToSingle / FpSingleToHalf
      // round-trip path (interpreter.h::AdvSimdFcma size==0b01).
      if (!host_platform::kHasF16C) { success_ = false; return; }

      SimdRegister xmm_n_fp16 = AllocTempSimdReg();
      SimdRegister xmm_m_fp16 = AllocTempSimdReg();
      SimdRegister xmm_d_fp16 = AllocTempSimdReg();
      SimdRegister xmm_sign_fp16 = AllocTempSimdReg();
      SimdRegister xmm_lane_fp16 = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister xmm_lo_save_fp16 = no_simd_register;
      if (args.q) {
        xmm_lo_save_fp16 = AllocTempSimdReg();
        if (xmm_lo_save_fp16 == no_simd_register) { success_ = false; return; }
      }

      bool is_fcmla = (args.opcode == Decoder::FcmaOpcode::kFcmla);

      // Emit the FP32 FCMA core (same shape as the size==0b10 path
      // below) on already-widened operands. Returns the XMM holding
      // the 4 FP32 lanes of the result.
      auto emit_fp32_core = [&](SimdRegister rn, SimdRegister rm,
                                SimdRegister rd, SimdRegister rsign,
                                SimdRegister rlane) -> SimdRegister {
        as_.Pcmpeqd(rsign, rsign);
        as_.Pslld(rsign, static_cast<int8_t>(31));
        if (!is_fcmla) {
          // FCADD: rot==0 -> #90 (negate real); rot==1 -> #270 (negate imag).
          as_.Shufps(rm, rm, static_cast<int8_t>(0xB1));
          as_.Pcmpeqd(rlane, rlane);
          if (args.rot == 0) {
            as_.Psrlq(rlane, static_cast<int8_t>(32));
          } else {
            as_.Psllq(rlane, static_cast<int8_t>(32));
          }
          as_.Pand(rsign, rlane);
          as_.Xorps(rm, rsign);
          as_.Addps(rn, rm);
          return rn;
        }
        // FCMLA: result = Vd + n_broadcast * m_xformed.
        switch (args.rot) {
          case 0:
            break;
          case 1:
            as_.Shufps(rm, rm, static_cast<int8_t>(0xB1));
            as_.Pcmpeqd(rlane, rlane);
            as_.Psrlq(rlane, static_cast<int8_t>(32));
            as_.Pand(rsign, rlane);
            as_.Xorps(rm, rsign);
            break;
          case 2:
            as_.Xorps(rm, rsign);
            break;
          default:  // case 3
            as_.Shufps(rm, rm, static_cast<int8_t>(0xB1));
            as_.Pcmpeqd(rlane, rlane);
            as_.Psllq(rlane, static_cast<int8_t>(32));
            as_.Pand(rsign, rlane);
            as_.Xorps(rm, rsign);
            break;
        }
        if (args.rot == 0 || args.rot == 2) {
          as_.Pshufd(rn, rn, static_cast<int8_t>(0xA0));
        } else {
          as_.Pshufd(rn, rn, static_cast<int8_t>(0xF5));
        }
        as_.Mulps(rn, rm);
        as_.Addps(rd, rn);
        return rd;
      };

      int32_t src_n_off_fp16 = VRegOffset(args.rn);
      int32_t src_m_off_fp16 = VRegOffset(args.rm);
      int32_t dst_off_fp16 = VRegOffset(args.rd);

      if (!args.q) {
        // .4H: 4 FP16 lanes in low 64 bits of each operand = 2 pairs.
        as_.Movq(xmm_n_fp16, {.base = Assembler::rbp, .disp = src_n_off_fp16});
        as_.Vcvtph2ps(xmm_n_fp16, xmm_n_fp16);
        as_.Movq(xmm_m_fp16, {.base = Assembler::rbp, .disp = src_m_off_fp16});
        as_.Vcvtph2ps(xmm_m_fp16, xmm_m_fp16);
        if (is_fcmla) {
          as_.Movq(xmm_d_fp16, {.base = Assembler::rbp, .disp = dst_off_fp16});
          as_.Vcvtph2ps(xmm_d_fp16, xmm_d_fp16);
        }
        SimdRegister xmm_res = emit_fp32_core(xmm_n_fp16, xmm_m_fp16,
                                              xmm_d_fp16, xmm_sign_fp16,
                                              xmm_lane_fp16);
        as_.Vcvtps2ph(xmm_res, xmm_res, int8_t{0});
        // Vcvtps2ph(xmm,xmm,0) auto-zeroes the upper 64 bits.
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off_fp16}, xmm_res);
      } else {
        // .8H: process low 4 lanes, save narrowed FP16 result, then
        // process high 4 lanes, then recombine via PSLLDQ+POR.
        as_.Movq(xmm_n_fp16, {.base = Assembler::rbp, .disp = src_n_off_fp16});
        as_.Vcvtph2ps(xmm_n_fp16, xmm_n_fp16);
        as_.Movq(xmm_m_fp16, {.base = Assembler::rbp, .disp = src_m_off_fp16});
        as_.Vcvtph2ps(xmm_m_fp16, xmm_m_fp16);
        if (is_fcmla) {
          as_.Movq(xmm_d_fp16, {.base = Assembler::rbp, .disp = dst_off_fp16});
          as_.Vcvtph2ps(xmm_d_fp16, xmm_d_fp16);
        }
        SimdRegister xmm_res1 = emit_fp32_core(xmm_n_fp16, xmm_m_fp16,
                                               xmm_d_fp16, xmm_sign_fp16,
                                               xmm_lane_fp16);
        as_.Vcvtps2ph(xmm_res1, xmm_res1, int8_t{0});
        as_.Movdqa(xmm_lo_save_fp16, xmm_res1);

        // Pass 2 (high 4 FP16 lanes): re-read each operand at +8.
        as_.Movq(xmm_n_fp16, {.base = Assembler::rbp, .disp = src_n_off_fp16 + 8});
        as_.Vcvtph2ps(xmm_n_fp16, xmm_n_fp16);
        as_.Movq(xmm_m_fp16, {.base = Assembler::rbp, .disp = src_m_off_fp16 + 8});
        as_.Vcvtph2ps(xmm_m_fp16, xmm_m_fp16);
        if (is_fcmla) {
          as_.Movq(xmm_d_fp16, {.base = Assembler::rbp, .disp = dst_off_fp16 + 8});
          as_.Vcvtph2ps(xmm_d_fp16, xmm_d_fp16);
        }
        SimdRegister xmm_res2 = emit_fp32_core(xmm_n_fp16, xmm_m_fp16,
                                               xmm_d_fp16, xmm_sign_fp16,
                                               xmm_lane_fp16);
        as_.Vcvtps2ph(xmm_res2, xmm_res2, int8_t{0});
        as_.Pslldq(xmm_res2, int8_t{8});
        as_.Por(xmm_res2, xmm_lo_save_fp16);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off_fp16}, xmm_res2);
      }
      return;
    }

    SimdRegister xmm_n = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_sign = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_lane = AllocTempSimdReg();
    if (!success()) { return; }

    int32_t src_n_off = VRegOffset(args.rn);
    int32_t src_m_off = VRegOffset(args.rm);
    int32_t dst_off   = VRegOffset(args.rd);

    as_.Movdqu(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
    as_.Movdqu(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});

    SimdRegister xmm_result = no_simd_register;  // tracks the register holding the final value

    if (args.size == 0b10) {
      // FP32 path: 1 pair (.2S, Q=0) or 2 pairs (.4S, Q=1).
      // Sign-bit mask for 32-bit FP lanes: [0x80000000]*4.
      as_.Pcmpeqd(xmm_sign, xmm_sign);
      as_.Pslld(xmm_sign, static_cast<int8_t>(31));

      if (args.opcode == Decoder::FcmaOpcode::kFcadd) {
        // m_xformed = pair-swap(Vm), negate the lanes implied by rot.
        as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));

        as_.Pcmpeqd(xmm_lane, xmm_lane);
        if (args.rot == 0) {
          // rot=#90: negate real lanes (0, 2) — mask = [0xff..ff, 0, 0xff..ff, 0].
          as_.Psrlq(xmm_lane, static_cast<int8_t>(32));
        } else {
          // rot=#270: negate imag lanes (1, 3) — mask = [0, 0xff..ff, 0, 0xff..ff].
          as_.Psllq(xmm_lane, static_cast<int8_t>(32));
        }
        as_.Pand(xmm_sign, xmm_lane);
        as_.Xorps(xmm_m, xmm_sign);

        // Vn + m_xformed -> xmm_n.
        as_.Addps(xmm_n, xmm_m);
        xmm_result = xmm_n;
      } else {
        // FCMLA: result = Vd + n_broadcast * m_xformed.
        SimdRegister xmm_d = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xmm_d, {.base = Assembler::rbp, .disp = dst_off});

        // m_xformed depends on rotation.
        switch (args.rot) {
          case 0:
            // No transform.
            break;
          case 1:
            // Swap + negate real lanes.
            as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));
            as_.Pcmpeqd(xmm_lane, xmm_lane);
            as_.Psrlq(xmm_lane, static_cast<int8_t>(32));
            as_.Pand(xmm_sign, xmm_lane);
            as_.Xorps(xmm_m, xmm_sign);
            break;
          case 2:
            // Negate all lanes.
            as_.Xorps(xmm_m, xmm_sign);
            break;
          case 3:
            // Swap + negate imag lanes.
            as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));
            as_.Pcmpeqd(xmm_lane, xmm_lane);
            as_.Psllq(xmm_lane, static_cast<int8_t>(32));
            as_.Pand(xmm_sign, xmm_lane);
            as_.Xorps(xmm_m, xmm_sign);
            break;
          default:
            // Cannot happen: decoder only emits rot in 0..3 for FCMLA.
            success_ = false;
            return;
        }

        // Broadcast n_re (rot 0/2) or n_im (rot 1/3) across both pair slots.
        // PSHUFD imm=0xA0 = (10,10,00,00): [lane0, lane0, lane2, lane2] = n_re bcast.
        // PSHUFD imm=0xF5 = (11,11,01,01): [lane1, lane1, lane3, lane3] = n_im bcast.
        if (args.rot == 0 || args.rot == 2) {
          as_.Pshufd(xmm_n, xmm_n, static_cast<int8_t>(0xA0));
        } else {
          as_.Pshufd(xmm_n, xmm_n, static_cast<int8_t>(0xF5));
        }

        as_.Mulps(xmm_n, xmm_m);   // n_broadcast * m_xformed
        as_.Addps(xmm_d, xmm_n);   // Vd += ...
        xmm_result = xmm_d;
      }
    } else {
      // FP64 path: size==0b11, .2D only (decoder rejects Q=0 for FP64).
      // 1 complex pair (lane 0 = re, lane 1 = im); 8 bytes per lane.
      // Same shape as FP32 but with FP64-width lowerings:
      //   Shufpd 0x01 swaps the two doubles; Shufpd 0x00/0x03 broadcasts
      //   lane 0/1 across both. PSRLDQ/PSLLDQ 8 builds the lane-mask
      //   (single pair, 8-byte lane width).
      // Sign-bit mask for 64-bit FP lanes: [0x8000000000000000]*2.
      as_.Pcmpeqd(xmm_sign, xmm_sign);
      as_.Psllq(xmm_sign, static_cast<int8_t>(63));

      if (args.opcode == Decoder::FcmaOpcode::kFcadd) {
        // Pair-swap: SHUFPD imm=0x01 -> [m_im, m_re].
        as_.Shufpd(xmm_m, xmm_m, static_cast<int8_t>(0x01));

        as_.Pcmpeqd(xmm_lane, xmm_lane);
        if (args.rot == 0) {
          // rot=#90: negate real lane (lane 0).
          // mask = [0xff..ff (low 64), 0 (high 64)] = PSRLDQ 8 over [-1; 16].
          as_.Psrldq(xmm_lane, static_cast<int8_t>(8));
        } else {
          // rot=#270: negate imag lane (lane 1).
          // mask = [0 (low 64), 0xff..ff (high 64)] = PSLLDQ 8 over [-1; 16].
          as_.Pslldq(xmm_lane, static_cast<int8_t>(8));
        }
        as_.Pand(xmm_sign, xmm_lane);
        as_.Xorpd(xmm_m, xmm_sign);

        as_.Addpd(xmm_n, xmm_m);
        xmm_result = xmm_n;
      } else {
        // FCMLA: result = Vd + n_broadcast * m_xformed.
        SimdRegister xmm_d = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Movdqu(xmm_d, {.base = Assembler::rbp, .disp = dst_off});

        switch (args.rot) {
          case 0:
            // No transform.
            break;
          case 1:
            // Swap + negate real lane.
            as_.Shufpd(xmm_m, xmm_m, static_cast<int8_t>(0x01));
            as_.Pcmpeqd(xmm_lane, xmm_lane);
            as_.Psrldq(xmm_lane, static_cast<int8_t>(8));
            as_.Pand(xmm_sign, xmm_lane);
            as_.Xorpd(xmm_m, xmm_sign);
            break;
          case 2:
            // Negate both lanes.
            as_.Xorpd(xmm_m, xmm_sign);
            break;
          case 3:
            // Swap + negate imag lane.
            as_.Shufpd(xmm_m, xmm_m, static_cast<int8_t>(0x01));
            as_.Pcmpeqd(xmm_lane, xmm_lane);
            as_.Pslldq(xmm_lane, static_cast<int8_t>(8));
            as_.Pand(xmm_sign, xmm_lane);
            as_.Xorpd(xmm_m, xmm_sign);
            break;
          default:
            success_ = false;
            return;
        }

        // Broadcast n_re (rot 0/2) -> [n[0], n[0]] via SHUFPD imm=0x00.
        // Broadcast n_im (rot 1/3) -> [n[1], n[1]] via SHUFPD imm=0x03.
        if (args.rot == 0 || args.rot == 2) {
          as_.Shufpd(xmm_n, xmm_n, static_cast<int8_t>(0x00));
        } else {
          as_.Shufpd(xmm_n, xmm_n, static_cast<int8_t>(0x03));
        }

        as_.Mulpd(xmm_n, xmm_m);
        as_.Addpd(xmm_d, xmm_n);
        xmm_result = xmm_d;
      }
    }

    // Q=0 (.2S only — FP64 always has Q=1): zero the upper 64 bits.
    if (!args.q) {
      // Reuse xmm_lane to build a [0xff..ff (low 64), 0 (high 64)] mask.
      as_.Pcmpeqd(xmm_lane, xmm_lane);
      as_.Psrldq(xmm_lane, static_cast<int8_t>(8));
      as_.Pand(xmm_result, xmm_lane);
    }

    as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_result);
  }

  // indexed FCMLA
  //
  // AdvSIMD complex floating-point by element (FCMLA-idx) JIT path for
  // FP32 (.4s).  Same shape as AdvSimdFcma's FP32 FCMLA branch, except
  // Vm contributes a single broadcast complex pair (Vm.s[2*idx],
  // Vm.s[2*idx+1]) rather than per-pair distinct pairs.  The broadcast
  // is one PSHUFD over Vm before the existing per-rot transform:
  //   idx=0: imm=0x44 -> [Vm.s[0], Vm.s[1], Vm.s[0], Vm.s[1]]
  //   idx=1: imm=0xEE -> [Vm.s[2], Vm.s[3], Vm.s[2], Vm.s[3]]
  // After the broadcast, the per-rot Vm transform (Shufps/Pand/Xorps)
  // and the Vn-half broadcast (PSHUFD 0xA0 for n_re, 0xF5 for n_im)
  // match the non-indexed FCMLA path verbatim.  Vd is read-modify-write
  // (FCMLA always accumulates).
  //
  // FP32-indexed mandates Q=1: the decoder rejects L=1 or Q=0 for
  // size=0b10, so there is no .2s indexed form and the post-loop Q=0
  // zero-clear tail used by the vector path is unreachable here.
  //
  // FP16-indexed (size=0b01) bails to the interpreter — same F16C
  // round-trip follow-up as the FP16 FCMA-vector row.
  void AdvSimdFcmaIdx(const Decoder::FcmaIdxArgs& args) {
    if (args.size == 0b01) {
      // FP16 indexed FCMLA JIT
      //
      // Lower FP16-indexed FCMLA via F16C round-trip on each half: read
      // the indexed complex pair Vm[2*index : 2*index+1] (2 FP16) into
      // a working XMM, widen to 4 FP32 lanes via a single Vcvtph2ps
      // (covering the 4-FP16 chunk that contains the pair), broadcast
      // the indexed pair across both 4-FP32-lane halves via PSHUFD,
      // then run the FP32-indexed FCMLA core on each output half.
      //
      // Byte offset of the indexed pair in Vm is (index*4); the 8-byte
      // chunk containing the pair starts at ((index/2)*8). Within that
      // chunk after Vcvtph2ps, the pair is at FP32 lanes 0..1 (index
      // even) or 2..3 (index odd) — broadcast with PSHUFD imm=0x44 or
      // 0xEE respectively.
      if (!host_platform::kHasF16C) { success_ = false; return; }

      SimdRegister xmm_n_fp16 = AllocTempSimdReg();
      SimdRegister xmm_m_fp16 = AllocTempSimdReg();
      SimdRegister xmm_d_fp16 = AllocTempSimdReg();
      SimdRegister xmm_sign_fp16 = AllocTempSimdReg();
      SimdRegister xmm_lane_fp16 = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister xmm_lo_save_fp16 = no_simd_register;
      if (args.q) {
        xmm_lo_save_fp16 = AllocTempSimdReg();
        if (xmm_lo_save_fp16 == no_simd_register) { success_ = false; return; }
      }

      int8_t broadcast_imm =
          ((args.index & 1) == 0) ? int8_t{0x44}
                                  : static_cast<int8_t>(0xEE);
      int32_t src_n_off_fp16 = VRegOffset(args.rn);
      int32_t src_m_off_fp16 = VRegOffset(args.rm);
      int32_t dst_off_fp16 = VRegOffset(args.rd);
      int32_t vm_load_off = src_m_off_fp16 + (args.index / 2) * 8;

      auto emit_fp32_core = [&](SimdRegister rn, SimdRegister rm,
                                SimdRegister rd, SimdRegister rsign,
                                SimdRegister rlane) -> SimdRegister {
        as_.Pcmpeqd(rsign, rsign);
        as_.Pslld(rsign, static_cast<int8_t>(31));
        switch (args.rot) {
          case 0:
            break;
          case 1:
            as_.Shufps(rm, rm, static_cast<int8_t>(0xB1));
            as_.Pcmpeqd(rlane, rlane);
            as_.Psrlq(rlane, static_cast<int8_t>(32));
            as_.Pand(rsign, rlane);
            as_.Xorps(rm, rsign);
            break;
          case 2:
            as_.Xorps(rm, rsign);
            break;
          default:  // case 3
            as_.Shufps(rm, rm, static_cast<int8_t>(0xB1));
            as_.Pcmpeqd(rlane, rlane);
            as_.Psllq(rlane, static_cast<int8_t>(32));
            as_.Pand(rsign, rlane);
            as_.Xorps(rm, rsign);
            break;
        }
        if (args.rot == 0 || args.rot == 2) {
          as_.Pshufd(rn, rn, static_cast<int8_t>(0xA0));
        } else {
          as_.Pshufd(rn, rn, static_cast<int8_t>(0xF5));
        }
        as_.Mulps(rn, rm);
        as_.Addps(rd, rn);
        return rd;
      };

      if (!args.q) {
        // .4H indexed: 2 output pairs (lanes 0..3).
        as_.Movq(xmm_n_fp16, {.base = Assembler::rbp, .disp = src_n_off_fp16});
        as_.Vcvtph2ps(xmm_n_fp16, xmm_n_fp16);
        as_.Movq(xmm_d_fp16, {.base = Assembler::rbp, .disp = dst_off_fp16});
        as_.Vcvtph2ps(xmm_d_fp16, xmm_d_fp16);
        as_.Movq(xmm_m_fp16, {.base = Assembler::rbp, .disp = vm_load_off});
        as_.Vcvtph2ps(xmm_m_fp16, xmm_m_fp16);
        as_.Pshufd(xmm_m_fp16, xmm_m_fp16, broadcast_imm);
        SimdRegister xmm_res = emit_fp32_core(xmm_n_fp16, xmm_m_fp16,
                                              xmm_d_fp16, xmm_sign_fp16,
                                              xmm_lane_fp16);
        as_.Vcvtps2ph(xmm_res, xmm_res, int8_t{0});
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off_fp16}, xmm_res);
      } else {
        // .8H indexed: 4 output pairs (lanes 0..7). Two passes.
        as_.Movq(xmm_n_fp16, {.base = Assembler::rbp, .disp = src_n_off_fp16});
        as_.Vcvtph2ps(xmm_n_fp16, xmm_n_fp16);
        as_.Movq(xmm_d_fp16, {.base = Assembler::rbp, .disp = dst_off_fp16});
        as_.Vcvtph2ps(xmm_d_fp16, xmm_d_fp16);
        as_.Movq(xmm_m_fp16, {.base = Assembler::rbp, .disp = vm_load_off});
        as_.Vcvtph2ps(xmm_m_fp16, xmm_m_fp16);
        as_.Pshufd(xmm_m_fp16, xmm_m_fp16, broadcast_imm);
        SimdRegister xmm_res1 = emit_fp32_core(xmm_n_fp16, xmm_m_fp16,
                                               xmm_d_fp16, xmm_sign_fp16,
                                               xmm_lane_fp16);
        as_.Vcvtps2ph(xmm_res1, xmm_res1, int8_t{0});
        as_.Movdqa(xmm_lo_save_fp16, xmm_res1);

        as_.Movq(xmm_n_fp16, {.base = Assembler::rbp, .disp = src_n_off_fp16 + 8});
        as_.Vcvtph2ps(xmm_n_fp16, xmm_n_fp16);
        as_.Movq(xmm_d_fp16, {.base = Assembler::rbp, .disp = dst_off_fp16 + 8});
        as_.Vcvtph2ps(xmm_d_fp16, xmm_d_fp16);
        as_.Movq(xmm_m_fp16, {.base = Assembler::rbp, .disp = vm_load_off});
        as_.Vcvtph2ps(xmm_m_fp16, xmm_m_fp16);
        as_.Pshufd(xmm_m_fp16, xmm_m_fp16, broadcast_imm);
        SimdRegister xmm_res2 = emit_fp32_core(xmm_n_fp16, xmm_m_fp16,
                                               xmm_d_fp16, xmm_sign_fp16,
                                               xmm_lane_fp16);
        as_.Vcvtps2ph(xmm_res2, xmm_res2, int8_t{0});
        as_.Pslldq(xmm_res2, int8_t{8});
        as_.Por(xmm_res2, xmm_lo_save_fp16);
        as_.Movdqu({.base = Assembler::rbp, .disp = dst_off_fp16}, xmm_res2);
      }
      return;
    }

    if (args.size != 0b10) {
      // FP64-indexed (size=0b11) is reserved by the architecture;
      // size=0b00 is unreachable via the decoder. Bail to interpreter.
      success_ = false;
      return;
    }

    SimdRegister xmm_n = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_m = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_d = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_sign = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_lane = AllocTempSimdReg();
    if (!success()) { return; }

    int32_t src_n_off = VRegOffset(args.rn);
    int32_t src_m_off = VRegOffset(args.rm);
    int32_t dst_off   = VRegOffset(args.rd);

    as_.Movdqu(xmm_n, {.base = Assembler::rbp, .disp = src_n_off});
    as_.Movdqu(xmm_m, {.base = Assembler::rbp, .disp = src_m_off});
    as_.Movdqu(xmm_d, {.base = Assembler::rbp, .disp = dst_off});

    // Broadcast the indexed complex pair across all of xmm_m.  PSHUFD
    // imm bits (3:2:1:0) each pick a source dword for that dst dword.
    //   idx=0: imm=0b01_00_01_00 = 0x44 -> [src[0], src[1], src[0], src[1]]
    //   idx=1: imm=0b11_10_11_10 = 0xEE -> [src[2], src[3], src[2], src[3]]
    as_.Pshufd(xmm_m, xmm_m,
               static_cast<int8_t>(args.index == 0 ? 0x44 : 0xEE));

    // Sign-bit mask for 32-bit FP lanes: [0x80000000]*4.
    as_.Pcmpeqd(xmm_sign, xmm_sign);
    as_.Pslld(xmm_sign, static_cast<int8_t>(31));

    // Apply the per-rotation Vm transform — identical to the vector
    // FCMLA path (lite_translator::AdvSimdFcma FP32 FCMLA branch).
    switch (args.rot) {
      case 0:
        // No transform.
        break;
      case 1:
        // Swap pair + negate real lanes.
        as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));
        as_.Pcmpeqd(xmm_lane, xmm_lane);
        as_.Psrlq(xmm_lane, static_cast<int8_t>(32));
        as_.Pand(xmm_sign, xmm_lane);
        as_.Xorps(xmm_m, xmm_sign);
        break;
      case 2:
        // Negate all lanes.
        as_.Xorps(xmm_m, xmm_sign);
        break;
      case 3:
        // Swap pair + negate imag lanes.
        as_.Shufps(xmm_m, xmm_m, static_cast<int8_t>(0xB1));
        as_.Pcmpeqd(xmm_lane, xmm_lane);
        as_.Psllq(xmm_lane, static_cast<int8_t>(32));
        as_.Pand(xmm_sign, xmm_lane);
        as_.Xorps(xmm_m, xmm_sign);
        break;
      default:
        // Decoder only emits rot in 0..3 for FCMLA-idx.
        success_ = false;
        return;
    }

    // Broadcast n_re (rot 0/2) or n_im (rot 1/3) across both pair slots.
    if (args.rot == 0 || args.rot == 2) {
      as_.Pshufd(xmm_n, xmm_n, static_cast<int8_t>(0xA0));
    } else {
      as_.Pshufd(xmm_n, xmm_n, static_cast<int8_t>(0xF5));
    }

    as_.Mulps(xmm_n, xmm_m);   // n_broadcast * m_xformed
    as_.Addps(xmm_d, xmm_n);   // Vd += ...

    // FP32-indexed FCMLA mandates Q=1, so no Q=0 zero-clear is needed.
    as_.Movdqu({.base = Assembler::rbp, .disp = dst_off}, xmm_d);
  }

  // AdvSIMD BFloat16 three-same-extra (BFDOT / BFMMLA / BFMLAL{B,T}).
  //
  // Reference: ARM ARM C7.2.40 (BFDOT), C7.2.42 (BFMMLA), C7.2.39
  // (BFMLALB/BFMLALT).  Per-lane BF16-pair multiply-and-accumulate into
  // FP32 destination lanes.
  //
  // JIT covers BFDOT (vector + indexed) and BFMLAL{B,T} (vector +
  // indexed).  BFMMLA stays on the interpreter (`success_ = false`) — its
  // 2x2 matrix-mul shape (each output lane sums 4 BF16-pair products from
  // interleaved rows/columns) is enough larger that the per-cycle commit
  // budget keeps it as a follow-on.
  //
  // BFMLAL{B,T} semantics (interpreter at `interpreter.h:1515`,1542):
  //   off = (T ? 1 : 0)                         // B=even, T=odd
  //   Vector: for i in [0..4):
  //     Vd.s[i] += Bf16ToFloat(Vn.h[2i + off]) * Bf16ToFloat(Vm.h[2i + off])
  //   Indexed: m = Bf16ToFloat(Vm.h[idx])  (single BF16, broadcast),
  //     for i in [0..4):
  //       Vd.s[i] += Bf16ToFloat(Vn.h[2i + off]) * m
  //   Q is implicit 1 for BFMLAL (dest always .4s).
  //
  // BFMLAL{B,T} lowering: the BF16 we want lives at either the LOW half
  // (B, off=0) or the HIGH half (T, off=1) of each FP32 lane of the source
  // vector.  Bf16ToFloat(x) == (uint32_t(x) << 16) reinterpreted as FP32,
  // so the lane's FP32 representation is "BF16 in the high 16 bits, zero
  // in the low 16 bits".  Hence:
  //   B: PSLLD $16        — lifts the low BF16 to the high half of each
  //                          FP32 lane; low half becomes zero.
  //   T: PSRLD $16 then PSLLD $16
  //                      — equivalent to ANDing with 0xFFFF0000 per
  //                          dword: zeros the low half while keeping the
  //                          high BF16 in its existing high-half position.
  // After the widen pass, MULPS pairs lanes and ADDPS accumulates into Vd.
  // Indexed form broadcasts a single BF16 across all 4 lanes via
  // PXOR + PINSRW (memory operand) + PSHUFD — the PINSRW into word slot 1
  // places the BF16 directly in the high 16 bits of FP32 lane 0 (which
  // already has zero in the low half from the PXOR), so no shift is
  // needed before PSHUFD broadcasts that lane.  No B/T discriminator
  // path is taken for indexed M because the index field already picks
  // the exact BF16; off only affects the Vn widen.
  //
  // BFDOT semantics (interpreter at `interpreter.h:1487`):
  //   For each FP32 output lane i:
  //     Vd.s[i] = Vd.s[i] + Bf16ToFloat(Vn.h[2i])   * Bf16ToFloat(Vm.h[2i])
  //                       + Bf16ToFloat(Vn.h[2i+1]) * Bf16ToFloat(Vm.h[2i+1])
  //   Q=0: 2 output lanes, upper 64 bits of Vd zeroed.
  //   Q=1: 4 output lanes.
  //   Indexed form (kBfdotIdx): Vm reads a single BF16 pair at index
  //   args.index (0..3) and broadcasts it across all output lanes —
  //   the m operand for every lane is (Vm.h[2*idx], Vm.h[2*idx+1]).
  //
  // SSE3/SSE4.1 lowering (no AVX-512-BF16 — host baseline lacks it):
  //
  //   1.  Widen BF16 to FP32 in-register.  Bf16ToFloat(x) is just
  //       (uint32_t(x) << 16) reinterpreted as float, so the widen is
  //       PMOVZXWD (4 BF16 -> 4 zero-extended dwords) followed by
  //       PSLLD $16 (lifts the 16-bit value into the high half of each
  //       dword = the FP32 representation of Bf16ToFloat).  Bit-exact
  //       across normals, subnormals, zeros, infinities, and NaNs (the
  //       BF16 NaN encoding maps 1:1 to an FP32 NaN with the same
  //       quiet/signalling bit pattern in the high 16 bits).
  //
  //   2.  MULPS pairs the widened lanes: 4 FP32 products per half.
  //
  //   3.  HADDPS sums adjacent pairs.  For Q=1: HADDPS xmm_n_lo,
  //       xmm_n_hi produces [p0+p1, p2+p3, p4+p5, p6+p7] — exactly the
  //       4 output lanes.  For Q=0: HADDPS xmm_n_lo, zero produces
  //       [p0+p1, p2+p3, 0, 0] — 2 valid lanes in the low 64.
  //
  //   4.  Accumulate into Vd via ADDPS.  Q=1 reads Vd as memory operand
  //       and stores 16 bytes.  Q=0 reads only the low 8 bytes via MOVQ
  //       (zero-extending the upper 64) so the 16-byte MOVDQU store
  //       naturally zeros Vd's upper half per ARM AArch64 Q=0 layout.
  //
  // Indexed form: load a 32-bit dword from Vm at byte offset 4*idx
  // (containing the BF16 pair Vm.h[2*idx] | Vm.h[2*idx+1]<<16); PSHUFD
  // imm=0x00 broadcasts that dword across all 4 lanes; PMOVZXWD then
  // sees [Vm.h[2*idx], Vm.h[2*idx+1], Vm.h[2*idx], Vm.h[2*idx+1]] as
  // 4 BF16 lanes in the low 64; the widen produces the broadcasted
  // m operand for every output lane.  Q=1 reuses the same xmm against
  // both Vn halves.
  void AdvSimdBf16ThreeSame(const Decoder::Bf16ThreeSameArgs& args) {
    const bool is_bfdot = (args.opcode == Decoder::Bf16ThreeSameOpcode::kBfdot ||
                           args.opcode == Decoder::Bf16ThreeSameOpcode::kBfdotIdx);
    const bool is_bfmlal =
        (args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlalbVec ||
         args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlaltVec ||
         args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlalbIdx ||
         args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlaltIdx);
    const bool is_bfmmla =
        (args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmmla);
    if (!is_bfdot && !is_bfmlal && !is_bfmmla) {
      success_ = false;
      return;
    }

    const int32_t vn_off = VRegOffset(args.rn);
    const int32_t vm_off = VRegOffset(args.rm);
    const int32_t vd_off = VRegOffset(args.rd);

    if (is_bfmmla) {
      // BFMMLA: 2x2 FP32 output matrix from 2x4 BF16 row inputs.
      // Per ARM ARM C7.2.55 (and interpreter.h:1501):
      //   for i, j in [0..2):
      //     Vd.s[i*2 + j] += Σ_{k∈[0..4)} Bf16ToFloat(Vn.h[i*4 + k]) *
      //                                   Bf16ToFloat(Vm.h[j*4 + k])
      // Q is always 1 (decoder rejects Q=0).
      //
      // Output lane layout in Vd:
      //   lane0 = (0,0) = Vn_row0 · Vm_row0   (h[0..3] · h[0..3])
      //   lane1 = (0,1) = Vn_row0 · Vm_row1   (h[0..3] · h[4..7])
      //   lane2 = (1,0) = Vn_row1 · Vm_row0   (h[4..7] · h[0..3])
      //   lane3 = (1,1) = Vn_row1 · Vm_row1   (h[4..7] · h[4..7])
      //
      // SSE3 lowering (no AVX-512-BF16 dependency):
      //   1. Widen each row to 4 FP32 via PMOVZXWD + PSLLD $16 — 4 temps.
      //   2. Per row, compute two 4-FP32 product vectors via MULPS, then
      //      collapse each to a scalar dot via two HADDPS rounds:
      //        HADDPS prod_a, prod_b  -> [pair0_a, pair1_a, pair0_b, pair1_b]
      //        HADDPS that, itself   -> [dot_a, dot_b, dot_a, dot_b]
      //      The low 64 bits then hold (dot_a, dot_b).
      //   3. MOVLHPS packs the row-1 low-64 (dot_10, dot_11) into the
      //      high-64 of the row-0 register that already holds (dot_00,
      //      dot_01), yielding [dot_00, dot_01, dot_10, dot_11] — the
      //      exact Vd lane order required by the ARM ARM matrix layout.
      //   4. ADDPS against Vd, MOVDQU store.
      //
      // Temp register budget: 5 SimdRegisters
      //   (xmm_n0, xmm_n1, xmm_m0, xmm_m1, xmm_tmp).  AllocTempSimdReg
      //   failure routes through success_=false (region-aware spill
      //   fallback to interpreter).
      SimdRegister xmm_n0 = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister xmm_n1 = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister xmm_m0 = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister xmm_m1 = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister xmm_tmp = AllocTempSimdReg();
      if (!success()) { return; }

      // Widen Vn into row 0 / row 1 FP32 vectors.
      as_.Pmovzxwd(xmm_n0, {.base = Assembler::rbp, .disp = vn_off + 0});
      as_.Pslld(xmm_n0, static_cast<int8_t>(16));
      as_.Pmovzxwd(xmm_n1, {.base = Assembler::rbp, .disp = vn_off + 8});
      as_.Pslld(xmm_n1, static_cast<int8_t>(16));

      // Widen Vm into row 0 / row 1 FP32 vectors.
      as_.Pmovzxwd(xmm_m0, {.base = Assembler::rbp, .disp = vm_off + 0});
      as_.Pslld(xmm_m0, static_cast<int8_t>(16));
      as_.Pmovzxwd(xmm_m1, {.base = Assembler::rbp, .disp = vm_off + 8});
      as_.Pslld(xmm_m1, static_cast<int8_t>(16));

      // Row 0 dot products: (0,0) and (0,1).
      // xmm_tmp = xmm_n0 * xmm_m0 (preserving xmm_n0 for the (0,1) product).
      as_.Movdqa(xmm_tmp, xmm_n0);
      as_.Mulps(xmm_tmp, xmm_m0);
      // xmm_n0 = xmm_n0 * xmm_m1 (xmm_n0 source no longer needed after this).
      as_.Mulps(xmm_n0, xmm_m1);
      // HADDPS xmm_tmp, xmm_n0 -> [pair0_00, pair1_00, pair0_01, pair1_01].
      as_.Haddps(xmm_tmp, xmm_n0);
      // HADDPS xmm_tmp, xmm_tmp -> [dot_00, dot_01, dot_00, dot_01].
      as_.Haddps(xmm_tmp, xmm_tmp);
      // Now xmm_tmp[63:0] = (dot_00, dot_01).

      // Row 1 dot products: (1,0) and (1,1).  Reuse the now-free xmm_n0
      // as scratch.
      as_.Movdqa(xmm_n0, xmm_n1);
      as_.Mulps(xmm_n0, xmm_m0);
      as_.Mulps(xmm_n1, xmm_m1);
      as_.Haddps(xmm_n0, xmm_n1);
      as_.Haddps(xmm_n0, xmm_n0);
      // Now xmm_n0[63:0] = (dot_10, dot_11).

      // Pack the two row halves into one xmm: MOVLHPS xmm_tmp, xmm_n0
      // copies xmm_n0[63:0] (= (dot_10, dot_11)) into xmm_tmp[127:64]
      // while leaving xmm_tmp[63:0] (= (dot_00, dot_01)) untouched.
      // Result: xmm_tmp = [dot_00, dot_01, dot_10, dot_11].
      as_.Movlhps(xmm_tmp, xmm_n0);

      // Accumulate into Vd and store all 16 bytes.
      as_.Addps(xmm_tmp, {.base = Assembler::rbp, .disp = vd_off});
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_tmp);
      return;
    }

    if (is_bfmlal) {
      const bool is_bft =
          (args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlaltVec ||
           args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlaltIdx);
      const bool is_indexed_mlal =
          (args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlalbIdx ||
           args.opcode == Decoder::Bf16ThreeSameOpcode::kBfmlaltIdx);

      SimdRegister xmm_n = AllocTempSimdReg();
      if (!success()) { return; }
      SimdRegister xmm_m = AllocTempSimdReg();
      if (!success()) { return; }

      // Stage 1: widen Vn to FP32 across all 4 lanes.  For BFMLALB
      // (off=0) the wanted BF16 sits in the LOW 16 of each FP32 lane —
      // PSLLD $16 lifts it to the high half (zeroing the low).  For
      // BFMLALT (off=1) the wanted BF16 is already in the HIGH 16 — we
      // just need to clear the LOW 16, which is PSRLD $16 then PSLLD $16
      // (effective AND with 0xFFFF0000 per dword).
      as_.Movdqu(xmm_n, {.base = Assembler::rbp, .disp = vn_off});
      if (is_bft) {
        as_.Psrld(xmm_n, static_cast<int8_t>(16));
      }
      as_.Pslld(xmm_n, static_cast<int8_t>(16));

      // Stage 2: prepare widened/broadcasted Vm operand.
      if (is_indexed_mlal) {
        // Indexed: load the single BF16 at Vm.h[idx] (byte offset 2*idx),
        // place it directly into the high 16 of FP32 lane 0 via PINSRW
        // (word slot 1).  Lane 0 then holds [0:16]=0, [16:32]=bf16 —
        // exactly the FP32 representation of Bf16ToFloat(bf16).  PSHUFD
        // imm=0 broadcasts that lane to all 4 lanes.
        as_.Pxor(xmm_m, xmm_m);
        as_.Pinsrw(xmm_m,
                   {.base = Assembler::rbp, .disp = vm_off + 2 * args.index},
                   int8_t{1});
        as_.Pshufd(xmm_m, xmm_m, static_cast<int8_t>(0x00));
      } else {
        // Vector: mirror the Vn widen.
        as_.Movdqu(xmm_m, {.base = Assembler::rbp, .disp = vm_off});
        if (is_bft) {
          as_.Psrld(xmm_m, static_cast<int8_t>(16));
        }
        as_.Pslld(xmm_m, static_cast<int8_t>(16));
      }

      // Stage 3: multiply lane-by-lane.
      as_.Mulps(xmm_n, xmm_m);

      // Stage 4: accumulate into Vd (Q implicit 1, always 4 FP32 lanes)
      // and store 16 bytes.
      as_.Addps(xmm_n, {.base = Assembler::rbp, .disp = vd_off});
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_n);
      return;
    }

    const bool is_indexed = (args.opcode == Decoder::Bf16ThreeSameOpcode::kBfdotIdx);

    SimdRegister xmm_n_lo = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_m_lo = AllocTempSimdReg();
    if (!success()) { return; }

    SimdRegister xmm_n_hi = no_simd_register;
    SimdRegister xmm_m_hi = no_simd_register;
    if (args.q) {
      xmm_n_hi = AllocTempSimdReg();
      if (xmm_n_hi == no_simd_register) { success_ = false; return; }
      if (!is_indexed) {
        xmm_m_hi = AllocTempSimdReg();
        if (xmm_m_hi == no_simd_register) { success_ = false; return; }
      }
    }

    // Stage 1: widen Vm to FP32.
    if (is_indexed) {
      // Load the 32-bit BF16 pair at byte offset 4*idx; broadcast it
      // across all 4 dword lanes; widen low 4 BF16 words to FP32.
      as_.Movd(xmm_m_lo, {.base = Assembler::rbp,
                          .disp = vm_off + 4 * args.index});
      as_.Pshufd(xmm_m_lo, xmm_m_lo, static_cast<int8_t>(0x00));
      as_.Pmovzxwd(xmm_m_lo, xmm_m_lo);
      as_.Pslld(xmm_m_lo, static_cast<int8_t>(16));
    } else {
      // Vector form: widen low 4 BF16 of Vm.
      as_.Pmovzxwd(xmm_m_lo, {.base = Assembler::rbp, .disp = vm_off + 0});
      as_.Pslld(xmm_m_lo, static_cast<int8_t>(16));
      if (args.q) {
        as_.Pmovzxwd(xmm_m_hi, {.base = Assembler::rbp, .disp = vm_off + 8});
        as_.Pslld(xmm_m_hi, static_cast<int8_t>(16));
      }
    }

    // Stage 2: widen Vn to FP32.
    as_.Pmovzxwd(xmm_n_lo, {.base = Assembler::rbp, .disp = vn_off + 0});
    as_.Pslld(xmm_n_lo, static_cast<int8_t>(16));
    if (args.q) {
      as_.Pmovzxwd(xmm_n_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
      as_.Pslld(xmm_n_hi, static_cast<int8_t>(16));
    }

    // Stage 3: pair multiply.
    as_.Mulps(xmm_n_lo, xmm_m_lo);
    if (args.q) {
      // Vector form pairs with xmm_m_hi; indexed reuses xmm_m_lo
      // (broadcasted m pair is identical for both Vn halves).
      as_.Mulps(xmm_n_hi, is_indexed ? xmm_m_lo : xmm_m_hi);
    }

    // Stage 4: horizontal-add adjacent FP32 pairs into output lanes.
    if (args.q) {
      // [p0+p1, p2+p3, p4+p5, p6+p7] — the 4 BFDOT output lanes.
      as_.Haddps(xmm_n_lo, xmm_n_hi);
    } else {
      // Q=0: only 2 output lanes; HADDPS against zero clears the
      // upper 64 bits of the result so the MOVQ-then-MOVDQU path
      // below naturally writes zero into Vd's upper half.
      as_.Pxor(xmm_m_lo, xmm_m_lo);
      as_.Haddps(xmm_n_lo, xmm_m_lo);
    }

    // Stage 5: accumulate into Vd and store 16 bytes.
    if (args.q) {
      as_.Addps(xmm_n_lo, {.base = Assembler::rbp, .disp = vd_off});
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_n_lo);
    } else {
      // Read Vd's low 8 bytes (low 2 FP32 lanes) zero-extended into
      // xmm_m_lo, ADDPS in place (upper 64 of xmm_n_lo is already 0
      // from HADDPS-vs-zero), MOVDQU 16 bytes so Vd's upper half lands
      // as zero per ARM AArch64 Q=0 layout.
      as_.Movq(xmm_m_lo, {.base = Assembler::rbp, .disp = vd_off});
      as_.Addps(xmm_n_lo, xmm_m_lo);
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_n_lo);
    }
  }

  // SDOT/UDOT JIT
  //
  // AdvSIMD integer dot product: SDOT/UDOT (vector and indexed-by-element).
  // Reference: ARM ARM C7.2.397 (SDOT), C7.2.398 (UDOT).
  //
  // Each 32-bit lane of Vd accumulates the dot product of 4 byte products:
  //   Vd[i] = (int32_t)(Vd[i] + sum_{k=0..3}(Vn_byte[4*i+k] * Vm_byte[base+k]))
  // where base = 4*i for the vector form, 4*args.index for the indexed
  // form (the indexed form broadcasts a single 4-byte group of Vm across
  // every lane).  SDOT sign-extends both byte vectors; UDOT zero-extends.
  // Q=0 reads only the low 8 bytes of Vn (and Vm in vector form), writes
  // 2 lanes, and zeros the upper 64 bits of Vd.
  //
  // SSE4.1 / SSSE3 lowering (no AVX-VNNI dependency — the emulator host
  // baseline doesn't include VPDPBUSD):
  //
  //   1.  Widen each 8-byte half of Vn (and Vm, vector form) from bytes to
  //       16-bit signed (SDOT) or unsigned-zero-extended (UDOT) words via
  //       PMOVSXBW / PMOVZXBW.
  //   2.  PMADDWD pairs adjacent 16-bit words: 8 words -> 4 int32 lanes,
  //       each lane = w[2k]*v[2k] + w[2k+1]*v[2k+1].
  //   3.  Each ARM dot lane is the sum of 4 byte products, so two adjacent
  //       PMADDWD outputs must be horizontally added — PHADDD pairs.
  //   4.  PADDD accumulates the 4 lanes into Vd (read-modify-write).
  //
  // For Q=0 (2 lanes): apply (1)-(3) on the low 8 bytes only, PHADDD x,x
  // gives [lane0,lane1,lane0,lane1]; MOVQ x,x masks the duplicate upper
  // half; MOVQ-load Vd's low half (zero-upper), PADDD, MOVDQU 16-byte
  // store — the upper 8 bytes land as zero, matching Q=0 semantics.
  //
  // Indexed form: load 4 bytes from [Vm + 4*idx] via MOVD, PSHUFD
  // imm=0x00 broadcasts that dword across all four 32-bit lanes; the
  // resulting xmm has identical low and high 8 bytes, so one
  // PMOVSXBW/ZXBW serves both PMADDWD pairings.
  //
  // All eight DOT encodings — vec×{Q=0,Q=1} × idx×{Q=0,Q=1} ×
  // {SDOT,UDOT} — are lowered.  The interpreter (interpreter.h:1237)
  // remains the executable spec; this JIT path produces bit-exact
  // output (32-bit integer arithmetic with defined wraparound).
  void AdvSimdMatMul(const Decoder::MatMulArgs& args) {
    // I8MM 8-bit matrix multiply-accumulate (Q=1 only):
    // Vd is a 2x2 int32 matrix, Vn holds 2 rows of 8 int8, Vm holds 2 rows of
    // 8 int8 (the result's columns). Output lane (2*i+j) += dot(Vn row i, Vm
    // row j) over 8 bytes. Widen each 8-byte row to 8x16 (sign per operand,
    // as in USDOT), PMADDWD each (row,col) pairing, then three PHADDDs fold the
    // four partial vectors into [d00, d01, d10, d11].
    using Op = Decoder::MatMulOpcode;
    const bool n_signed = (args.opcode == Op::kSmmla);
    const bool m_signed = (args.opcode == Op::kSmmla || args.opcode == Op::kUsmmla);
    const int32_t vn_off = VRegOffset(args.rn);
    const int32_t vm_off = VRegOffset(args.rm);
    const int32_t vd_off = VRegOffset(args.rd);
    SimdRegister n0 = AllocTempSimdReg();
    SimdRegister n1 = AllocTempSimdReg();
    SimdRegister m0 = AllocTempSimdReg();
    SimdRegister m1 = AllocTempSimdReg();
    SimdRegister a = AllocTempSimdReg();
    SimdRegister b = AllocTempSimdReg();
    SimdRegister c = AllocTempSimdReg();
    if (!success()) { return; }
    auto widen = [&](SimdRegister r, int32_t off, bool sgn) {
      if (sgn) {
        as_.Pmovsxbw(r, {.base = Assembler::rbp, .disp = off});
      } else {
        as_.Pmovzxbw(r, {.base = Assembler::rbp, .disp = off});
      }
    };
    widen(n0, vn_off + 0, n_signed);  // Vn row 0
    widen(n1, vn_off + 8, n_signed);  // Vn row 1
    widen(m0, vm_off + 0, m_signed);  // Vm row 0
    widen(m1, vm_off + 8, m_signed);  // Vm row 1
    as_.Movdqa(a, n0); as_.Pmaddwd(a, m0);  // p00
    as_.Movdqa(b, n0); as_.Pmaddwd(b, m1);  // p01
    as_.Movdqa(c, n1); as_.Pmaddwd(c, m0);  // p10
    as_.Pmaddwd(n1, m1);                     // p11
    as_.Phaddd(a, b);   // [pair-sums of p00 | p01]
    as_.Phaddd(c, n1);  // [pair-sums of p10 | p11]
    as_.Phaddd(a, c);   // [d00, d01, d10, d11]
    as_.Paddd(a, {.base = Assembler::rbp, .disp = vd_off});
    as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, a);
  }

  void AdvSimdDotProduct(const Decoder::DotProductArgs& args) {
    // Per-operand signedness covers SDOT/UDOT and the I8MM
    // mixed-sign forms USDOT (Vn unsigned, Vm signed) and SUDOT (Vn signed, Vm
    // unsigned). An unsigned byte zero-extended to 16 bits stays < 32768, so
    // PMADDWD's signed 16x16 multiply yields the correct mixed-sign products.
    using Op = Decoder::DotProductOpcode;
    const bool n_signed = (args.opcode == Op::kSdot || args.opcode == Op::kSdotIdx ||
                           args.opcode == Op::kSudotIdx);
    const bool m_signed = (args.opcode == Op::kSdot || args.opcode == Op::kSdotIdx ||
                           args.opcode == Op::kUsdot || args.opcode == Op::kUsdotIdx);
    const bool is_indexed = (args.opcode == Op::kSdotIdx || args.opcode == Op::kUdotIdx ||
                             args.opcode == Op::kUsdotIdx || args.opcode == Op::kSudotIdx);

    const int32_t vn_off = VRegOffset(args.rn);
    const int32_t vm_off = VRegOffset(args.rm);
    const int32_t vd_off = VRegOffset(args.rd);

    SimdRegister xmm_n_lo = AllocTempSimdReg();
    if (!success()) { return; }
    SimdRegister xmm_m_lo = AllocTempSimdReg();
    if (!success()) { return; }

    // For Q=1 we additionally need 2 temps to widen the high halves
    // (vector form) or hold the second PMADDWD result (indexed form).
    SimdRegister xmm_n_hi = no_simd_register;
    SimdRegister xmm_m_hi = no_simd_register;
    if (args.q) {
      xmm_n_hi = AllocTempSimdReg();
      if (xmm_n_hi == no_simd_register) { success_ = false; return; }
      if (!is_indexed) {
        xmm_m_hi = AllocTempSimdReg();
        if (xmm_m_hi == no_simd_register) { success_ = false; return; }
      }
    }

    // Stage 1: prepare widened Vm operand.
    if (is_indexed) {
      // Load Vm[4*idx..4*idx+3] as a dword, broadcast across all 4
      // lanes; after PSHUFD the low and high 8 bytes are identical,
      // so a single PMOVSXBW/ZXBW from the low half suffices for both
      // PMADDWD pairings.
      as_.Movd(xmm_m_lo, {.base = Assembler::rbp,
                          .disp = vm_off + 4 * args.index});
      as_.Pshufd(xmm_m_lo, xmm_m_lo, static_cast<int8_t>(0x00));
      if (m_signed) {
        as_.Pmovsxbw(xmm_m_lo, xmm_m_lo);
      } else {
        as_.Pmovzxbw(xmm_m_lo, xmm_m_lo);
      }
    } else {
      // Vector form: widen low 8 bytes of Vm from memory.
      if (m_signed) {
        as_.Pmovsxbw(xmm_m_lo, {.base = Assembler::rbp, .disp = vm_off + 0});
      } else {
        as_.Pmovzxbw(xmm_m_lo, {.base = Assembler::rbp, .disp = vm_off + 0});
      }
      if (args.q) {
        if (m_signed) {
          as_.Pmovsxbw(xmm_m_hi, {.base = Assembler::rbp, .disp = vm_off + 8});
        } else {
          as_.Pmovzxbw(xmm_m_hi, {.base = Assembler::rbp, .disp = vm_off + 8});
        }
      }
    }

    // Stage 2: widen Vn.
    if (n_signed) {
      as_.Pmovsxbw(xmm_n_lo, {.base = Assembler::rbp, .disp = vn_off + 0});
    } else {
      as_.Pmovzxbw(xmm_n_lo, {.base = Assembler::rbp, .disp = vn_off + 0});
    }
    if (args.q) {
      if (n_signed) {
        as_.Pmovsxbw(xmm_n_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
      } else {
        as_.Pmovzxbw(xmm_n_hi, {.base = Assembler::rbp, .disp = vn_off + 8});
      }
    }

    // Stage 3: pair multiply-and-add into 4 int32 partial sums per half.
    as_.Pmaddwd(xmm_n_lo, xmm_m_lo);
    if (args.q) {
      // Vector form pairs with xmm_m_hi; indexed form reuses xmm_m_lo.
      as_.Pmaddwd(xmm_n_hi, is_indexed ? xmm_m_lo : xmm_m_hi);
    }

    // Stage 4: horizontal add adjacent pairs to get final lanes.
    if (args.q) {
      // Result lanes: [n_lo[0..1], n_lo[2..3], n_hi[0..1], n_hi[2..3]].
      as_.Phaddd(xmm_n_lo, xmm_n_hi);
    } else {
      // Q=0: only 2 lanes are meaningful.  PHADDD xmm,xmm duplicates
      // the low 64 bits into the upper 64; we mask via MOVQ below.
      as_.Phaddd(xmm_n_lo, xmm_n_lo);
    }

    // Stage 5: accumulate into Vd and store.
    if (args.q) {
      as_.Paddd(xmm_n_lo, {.base = Assembler::rbp, .disp = vd_off});
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_n_lo);
    } else {
      // Q=0: low 8 bytes of Vd are accumulated; upper 8 bytes are zeroed.
      // MOVQ x,x clears the duplicate upper half from PHADDD; MOVQ from
      // Vd zero-extends Vd[0..7] into a clean operand; PADDD adds; the
      // full-width MOVDQU lands the upper 8 bytes as zero.
      as_.Movq(xmm_n_lo, xmm_n_lo);
      as_.Movq(xmm_m_lo, {.base = Assembler::rbp, .disp = vd_off});
      as_.Paddd(xmm_n_lo, xmm_m_lo);
      as_.Movdqu({.base = Assembler::rbp, .disp = vd_off}, xmm_n_lo);
    }
  }

  // Mirror of the interpreter's ExpandSimdModifiedImm — pure function of the
  // (compile-time-known) encoding fields, so the JIT can materialise the full
  // 128-bit MOVI/MVNI value at translation time and emit a constant load.
  // VFPExpandImm (ARM ARM J1) for FMOV (vector, immediate). Single precision:
  // float = a : NOT(b) : b*5 : c:d : e:f:g:h:Zeros(19).
  static uint32_t VFPExpandImm32Jit(uint8_t imm8) {
    uint32_t sign = (imm8 >> 7) & 1;
    uint32_t b = (imm8 >> 6) & 1;
    uint32_t exp = ((1 - b) << 7) | ((b ? 0x1Fu : 0u) << 2) | ((imm8 >> 4) & 0x3u);
    uint32_t mantissa = static_cast<uint32_t>(imm8 & 0xFu) << 19;
    return (sign << 31) | (exp << 23) | mantissa;
  }
  // Double precision: double = a : NOT(b) : b*8 : c:d : e:f:g:h:Zeros(48).
  static uint64_t VFPExpandImm64Jit(uint8_t imm8) {
    uint64_t sign = (imm8 >> 7) & 1;
    uint64_t b = (imm8 >> 6) & 1;
    uint64_t exp = ((1 - b) << 10) | ((b ? 0xFFull : 0ull) << 2) | ((imm8 >> 4) & 0x3ull);
    uint64_t mantissa = static_cast<uint64_t>(imm8 & 0xFull) << 48;
    return (sign << 63) | (exp << 52) | mantissa;
  }

  static __uint128_t ExpandSimdModifiedImmJit(uint8_t op, uint8_t cmode,
                                              uint8_t abc, uint8_t defgh, bool q) {
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
            // MOVI 8-bit replicated (cmode=0b1110).
            for (int i = 0; i < 8; i++) imm64 |= uint64_t{imm8} << (i * 8);
          } else {
            // FMOV (vector, immediate) single-precision (cmode=0b1111, op=0).
            uint32_t f = VFPExpandImm32Jit(imm8);
            imm64 = uint64_t{f} | (uint64_t{f} << 32);
          }
          break;
      }
    } else {
      if (cmode == 0b1110) {
        for (int i = 0; i < 8; i++)
          if (imm8 & (1 << i)) imm64 |= 0xFFULL << (i * 8);
      } else if (cmode == 0b1111) {
        // FMOV (vector, immediate) double-precision (op=1, cmode=0b1111).
        imm64 = VFPExpandImm64Jit(imm8);
      } else {
        return ~ExpandSimdModifiedImmJit(0, cmode, abc, defgh, q);
      }
    }
    __uint128_t result = static_cast<__uint128_t>(imm64);
    if (q) result |= static_cast<__uint128_t>(imm64) << 64;
    return result;
  }

  void SimdModifiedImm(const Decoder::SimdModifiedImmArgs& args) {
    const uint8_t cmode = args.cmode;
    // ORR/BIC (vector, immediate) — cmode<0>==1 and cmode<3:2>!=11 — are
    // read-modify-write, unlike the MOVI/MVNI/FMOV replace forms. The immediate
    // is the MOVI-style (op=0) expansion; ORR (op=0) sets bits, BIC (op=1)
    // clears them.
    const bool is_orr_bic = (cmode & 1) && ((cmode & 0b1100) != 0b1100);
    __uint128_t value = is_orr_bic
        ? ExpandSimdModifiedImmJit(0, cmode, args.abc, args.defgh, args.q)
        : ExpandSimdModifiedImmJit(args.op, cmode, args.abc, args.defgh, args.q);
    uint64_t lo = static_cast<uint64_t>(value);
    // Q==0 operates on the low 64 bits and zeroes the upper 64 of Vd.
    uint64_t hi = args.q ? static_cast<uint64_t>(value >> 64) : 0;
    int32_t off = VRegOffset(args.rd);
    SimdRegister xd = AllocTempSimdReg();
    if (!success()) { return; }

    if (is_orr_bic) {
      // Materialise the immediate, load Vd (low 64 zero-extended when Q==0 so
      // the upper half ends up zeroed), then OR (set) or AND-NOT (clear).
      SimdRegister ximm = AllocTempSimdReg();
      if (!success()) { return; }
      Register tmp = AllocTempReg();
      if (tmp == no_register) { success_ = false; return; }
      as_.Movq(tmp, static_cast<int64_t>(lo));
      as_.Movq(ximm, tmp);  // zero-extends: upper 64 = 0
      if (hi != 0) {
        as_.Movq(tmp, static_cast<int64_t>(hi));
        as_.Pinsrq(ximm, tmp, int8_t{1});
      }
      if (args.q) {
        as_.Movdqu(xd, {.base = Assembler::rbp, .disp = off});
      } else {
        as_.Movsd(xd, {.base = Assembler::rbp, .disp = off});  // zero-extends upper 64
      }
      if (args.op == 0) {
        as_.Por(xd, ximm);
        as_.Movdqu({.base = Assembler::rbp, .disp = off}, xd);
      } else {
        as_.Pandn(ximm, xd);  // ximm = ~imm & Vd
        as_.Movdqu({.base = Assembler::rbp, .disp = off}, ximm);
      }
      return;
    }

    // MOVI/MVNI/FMOV: emit a constant load (replace).
    if (lo == 0 && hi == 0) {
      as_.Pxor(xd, xd);
    } else {
      Register tmp = AllocTempReg();
      if (tmp == no_register) { success_ = false; return; }
      as_.Movq(tmp, static_cast<int64_t>(lo));
      as_.Movq(xd, tmp);  // zero-extends: upper 64 = 0
      if (hi != 0) {
        as_.Movq(tmp, static_cast<int64_t>(hi));
        as_.Pinsrq(xd, tmp, int8_t{1});
      }
    }
    as_.Movdqu({.base = Assembler::rbp, .disp = off}, xd);
  }

  // LDR (literal) SIMD/FP: load 32/64/128 bits from [insn_addr + offset]
  // into V[rt]. The address is constant at JIT time, so we materialize
  // it via movabs into a temp register and reuse the standard SIMD
  // immediate-offset load path (rn=0, offset=0; the materialized addr is
  // passed as the base). The integer LDR (literal) JIT follows the same
  // pattern.
  void SimdLoadLiteral(const Decoder::SimdLoadLiteralArgs& args) {
    GuestAddr target = GetInsnAddr() + args.offset;
    Register addr = AllocTempReg();
    if (addr == no_register) { Undefined(); return; }
    as_.Movq(addr, target);
    SimdLoadStoreImm(
        {.rt = args.rt, .rn = 0, .offset = 0, .size = args.size, .is_store = false},
        addr);
  }

  void SimdLoadStoreImm(const Decoder::SimdLoadStoreImmArgs& args, Register base) {
    // apply TBI mask before using base as memory operand.
    base = ApplyTbi(base);
    // Handle 128-bit SIMD load/store with immediate offset (STR/LDR Q-register).
    if (args.size == Decoder::SimdLoadStoreSize::k128bit) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }

      int32_t offset = args.offset;
      if (args.is_store) {
        // Load from ThreadState SIMD register, then store to memory.
        int32_t vreg_offset = VRegOffset(args.rt);
        as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movdqu({.base = base, .disp = offset}, xmm);
      } else {
        // Load from memory, store to ThreadState SIMD register.
        as_.Movdqu(xmm, {.base = base, .disp = offset});
        int32_t vreg_offset = VRegOffset(args.rt);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      }
      return;
    }
    // 64-bit (D-register) load/store.
    if (args.size == Decoder::SimdLoadStoreSize::k64bit) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }

      int32_t offset = args.offset;
      if (args.is_store) {
        int32_t vreg_offset = VRegOffset(args.rt);
        as_.Movq(xmm, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movq({.base = base, .disp = offset}, xmm);
      } else {
        as_.Pxor(xmm, xmm);  // Zero upper 64 bits.
        as_.Movq(xmm, {.base = base, .disp = offset});
        int32_t vreg_offset = VRegOffset(args.rt);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      }
      return;
    }
    // 32-bit (S-register) load/store.
    if (args.size == Decoder::SimdLoadStoreSize::k32bit) {
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }

      int32_t offset = args.offset;
      int32_t vreg_offset = VRegOffset(args.rt);
      if (args.is_store) {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movl({.base = base, .disp = offset}, tmp);
      } else {
        // Zero the full 128-bit register, then load 32 bits.
        SimdRegister xmm = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
        as_.Movl(tmp, {.base = base, .disp = offset});
        as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
      }
      return;
    }
    // 16-bit (H-register) load/store.
    if (args.size == Decoder::SimdLoadStoreSize::k16bit) {
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }

      int32_t offset = args.offset;
      int32_t vreg_offset = VRegOffset(args.rt);
      if (args.is_store) {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movw({.base = base, .disp = offset}, tmp);
      } else {
        SimdRegister xmm = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
        as_.Movzxwl(tmp, {.base = base, .disp = offset});
        as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
      }
      return;
    }
    // 8-bit (B-register) load/store.
    if (args.size == Decoder::SimdLoadStoreSize::k8bit) {
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }

      int32_t offset = args.offset;
      int32_t vreg_offset = VRegOffset(args.rt);
      if (args.is_store) {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movb({.base = base, .disp = offset}, tmp);
      } else {
        SimdRegister xmm = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
        as_.Movzxbl(tmp, {.base = base, .disp = offset});
        as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
      }
      return;
    }
    Undefined();
  }

  void SimdLoadStorePair(const Decoder::SimdLoadStorePairArgs& args, Register addr) {
    // apply TBI mask before using addr as memory operand.
    addr = ApplyTbi(addr);
    // Handle 128-bit pair store/load (STP/LDP q-register).
    if (args.size == Decoder::SimdLoadStoreSize::k128bit) {
      SimdRegister xmm1 = AllocTempSimdReg();
      SimdRegister xmm2 = AllocTempSimdReg();
      if (!success()) { return; }

      if (args.is_store) {
        int32_t vreg1_off = VRegOffset(args.rt1);
        int32_t vreg2_off = VRegOffset(args.rt2);
        as_.Movdqu(xmm1, {.base = Assembler::rbp, .disp = vreg1_off});
        as_.Movdqu(xmm2, {.base = Assembler::rbp, .disp = vreg2_off});
        as_.Movdqu({.base = addr, .disp = 0}, xmm1);
        as_.Movdqu({.base = addr, .disp = 16}, xmm2);
      } else {
        as_.Movdqu(xmm1, {.base = addr, .disp = 0});
        as_.Movdqu(xmm2, {.base = addr, .disp = 16});
        int32_t vreg1_off = VRegOffset(args.rt1);
        int32_t vreg2_off = VRegOffset(args.rt2);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg1_off}, xmm1);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg2_off}, xmm2);
      }
      return;
    }
    // Handle 64-bit (D) and 32-bit (S) pairs (LDP/STP d/s registers). The two
    // elements sit at [addr] and [addr + element_size]. A load zero-extends the
    // unused upper lanes (MOVSD zeroes bits 127:64, MOVSS zeroes bits 127:32),
    // then the full 16-byte v[] slot is committed via MOVDQU so the guest
    // register's upper bits read as zero.
    if (args.size == Decoder::SimdLoadStoreSize::k64bit ||
        args.size == Decoder::SimdLoadStoreSize::k32bit) {
      SimdRegister xmm1 = AllocTempSimdReg();
      SimdRegister xmm2 = AllocTempSimdReg();
      if (!success()) { return; }
      int32_t vreg1_off = VRegOffset(args.rt1);
      int32_t vreg2_off = VRegOffset(args.rt2);
      const int32_t element_size =
          (args.size == Decoder::SimdLoadStoreSize::k64bit) ? 8 : 4;
      if (args.size == Decoder::SimdLoadStoreSize::k64bit) {
        if (args.is_store) {
          as_.Movsd(xmm1, {.base = Assembler::rbp, .disp = vreg1_off});
          as_.Movsd(xmm2, {.base = Assembler::rbp, .disp = vreg2_off});
          as_.Movsd({.base = addr, .disp = 0}, xmm1);
          as_.Movsd({.base = addr, .disp = element_size}, xmm2);
        } else {
          as_.Movsd(xmm1, {.base = addr, .disp = 0});  // zero-extends upper 64
          as_.Movsd(xmm2, {.base = addr, .disp = element_size});
          as_.Movdqu({.base = Assembler::rbp, .disp = vreg1_off}, xmm1);
          as_.Movdqu({.base = Assembler::rbp, .disp = vreg2_off}, xmm2);
        }
      } else {  // k32bit
        if (args.is_store) {
          as_.Movss(xmm1, {.base = Assembler::rbp, .disp = vreg1_off});
          as_.Movss(xmm2, {.base = Assembler::rbp, .disp = vreg2_off});
          as_.Movss({.base = addr, .disp = 0}, xmm1);
          as_.Movss({.base = addr, .disp = element_size}, xmm2);
        } else {
          as_.Movss(xmm1, {.base = addr, .disp = 0});  // zero-extends upper 96
          as_.Movss(xmm2, {.base = addr, .disp = element_size});
          as_.Movdqu({.base = Assembler::rbp, .disp = vreg1_off}, xmm1);
          as_.Movdqu({.base = Assembler::rbp, .disp = vreg2_off}, xmm2);
        }
      }
      return;
    }
    Undefined();
  }

  void SimdLoadStoreReg(const Decoder::SimdLoadStoreRegArgs& args,
                        Register base, Register offset_reg) {
    // Handle SIMD load/store with register offset:
    //   LDR/STR Vt, [Xn, (X|W)m{, extend{ #shift}}]
    // The offset register may need UXTW / SXTW / UXTX / SXTX extension
    // before shift+add, exactly like the GP load/store path.
    Register addr = AllocTempReg();
    if (addr == no_register) { Undefined(); return; }

    ApplyOffsetExtend(addr, offset_reg, args.extend_type);
    if (args.shift_amount > 0) {
      as_.Shlq(addr, static_cast<int8_t>(args.shift_amount));
    }
    as_.Addq(addr, base);
    // apply TBI mask (top byte ignored on ARM64).
    as_.Shlq(addr, static_cast<int8_t>(8));
    as_.Shrq(addr, static_cast<int8_t>(8));

    int32_t vreg_offset = VRegOffset(args.rt);

    if (args.size == Decoder::SimdLoadStoreSize::k128bit) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      if (args.is_store) {
        as_.Movdqu(xmm, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movdqu({.base = addr, .disp = 0}, xmm);
      } else {
        as_.Movdqu(xmm, {.base = addr, .disp = 0});
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      }
    } else if (args.size == Decoder::SimdLoadStoreSize::k64bit) {
      SimdRegister xmm = AllocTempSimdReg();
      if (!success()) { return; }
      if (args.is_store) {
        as_.Movq(xmm, {.base = Assembler::rbp, .disp = vreg_offset});
        as_.Movq({.base = addr, .disp = 0}, xmm);
      } else {
        as_.Pxor(xmm, xmm);
        as_.Movq(xmm, {.base = addr, .disp = 0});
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
      }
    } else {
      // 8-bit, 16-bit, 32-bit: use GP register for transfer
      Register tmp = AllocTempReg();
      if (tmp == no_register) { Undefined(); return; }
      if (args.is_store) {
        as_.Movl(tmp, {.base = Assembler::rbp, .disp = vreg_offset});
        if (args.size == Decoder::SimdLoadStoreSize::k8bit) {
          as_.Movb({.base = addr, .disp = 0}, tmp);
        } else if (args.size == Decoder::SimdLoadStoreSize::k16bit) {
          as_.Movw({.base = addr, .disp = 0}, tmp);
        } else {  // k32bit
          as_.Movl({.base = addr, .disp = 0}, tmp);
        }
      } else {
        // Zero the full 128-bit SIMD register first
        SimdRegister xmm = AllocTempSimdReg();
        if (!success()) { return; }
        as_.Pxor(xmm, xmm);
        as_.Movdqu({.base = Assembler::rbp, .disp = vreg_offset}, xmm);
        // Load only the relevant bytes
        if (args.size == Decoder::SimdLoadStoreSize::k8bit) {
          as_.Movzxbl(tmp, {.base = addr, .disp = 0});
        } else if (args.size == Decoder::SimdLoadStoreSize::k16bit) {
          as_.Movzxwl(tmp, {.base = addr, .disp = 0});
        } else {  // k32bit
          as_.Movl(tmp, {.base = addr, .disp = 0});
        }
        as_.Movl({.base = Assembler::rbp, .disp = vreg_offset}, tmp);
      }
    }
    return;
  }

