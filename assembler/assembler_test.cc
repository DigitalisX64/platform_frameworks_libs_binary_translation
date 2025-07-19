/*
 * Copyright (C) 2014 The Android Open Source Project
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

#include "gtest/gtest.h"

#include <sys/mman.h>

#include <cstring>
#include <iterator>
#include <string>

#include "berberis/assembler/machine_code.h"
#include "berberis/assembler/rv32e.h"
#include "berberis/assembler/rv32i.h"
#include "berberis/assembler/rv64i.h"
#include "berberis/assembler/x86_32.h"
#include "berberis/assembler/x86_64.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/tracing.h"
#include "berberis/test_utils/scoped_exec_region.h"

#if defined(__i386__)
using CodeEmitter = berberis::x86_32::Assembler;
#elif defined(__amd64__)
using CodeEmitter = berberis::x86_64::Assembler;
#elif defined(__riscv)
using CodeEmitter = berberis::riscv64::Assembler;
#else
#error "Unsupported platform"
#endif

namespace berberis {

enum class CPUArch {
  kX86_64,
  kRiscv64,
};

int Callee() {
  return 239;
}

float FloatFunc(float f1, float f2) {
  return f1 - f2;
}

inline bool IsInstructionEqual(std::string code_str1,
                               std::string code_str2,
                               uint32_t insn,
                               uint32_t insn_size) {
  return code_str1.compare(
             insn * (insn_size + 1), insn_size, code_str2, insn * (insn_size + 1), insn_size) == 0;
}

template <typename ParcelInt>
inline bool CompareCode(const ParcelInt* code_template_begin,
                        const ParcelInt* code_template_end,
                        const MachineCode& code,
                        CPUArch arch) {
  if ((code_template_end - code_template_begin) * sizeof(ParcelInt) != code.install_size()) {
    TRACE_AND_ALOGE(
        "Code size mismatch: %zd != %u",
        (code_template_end - code_template_begin) * static_cast<unsigned>(sizeof(ParcelInt)),
        code.install_size());
    return false;
  }

  if (memcmp(code_template_begin, code.AddrAs<uint8_t>(0), code.install_size()) != 0) {
    TRACE_AND_ALOGE("Code mismatch");
    MachineCode code2;
    code2.AddSequence(code_template_begin, code_template_end - code_template_begin);
    std::string code_str1, code_str2;
    uint32_t insn_size = 0;
    switch (arch) {
      case CPUArch::kRiscv64:
        insn_size = 8;
        code.AsString(&code_str1, InstructionSize::FourBytes);
        code2.AsString(&code_str2, InstructionSize::FourBytes);
        break;
      case CPUArch::kX86_64:
        insn_size = 2;
        code.AsString(&code_str1, InstructionSize::OneByte);
        code2.AsString(&code_str2, InstructionSize::OneByte);
        break;
    }
    CHECK_EQ(code_str1.size() % (insn_size + 1), 0);
    CHECK_EQ(code_str2.size() % (insn_size + 1), 0);
    uint32_t number_of_instructions = code_str1.size() / (insn_size + 1);
    // Skip identical part.
    uint32_t insn = 0;
    while (insn < number_of_instructions &&
           IsInstructionEqual(code_str1, code_str2, insn, insn_size)) {
      insn++;
    }
    for (uint32_t i = insn; i < insn + 20 && i < number_of_instructions; i++) {
      TRACE_AND_ALOGE("Assembler generated: %s, should be %s\n",
                      code_str1.substr(i * (insn_size + 1), insn_size).c_str(),
                      code_str2.substr(i * (insn_size + 1), insn_size).c_str());
    }

    return false;
  }
  return true;
}

namespace riscv32 {

bool AssemblerTest() {
  MachineCode code;
  Assembler assembler(&code);
  Assembler::Label data_begin, data_end, label;
  assembler.Bind(&data_begin);
  // We test loads and stores twice to ensure that both positive and negative immediates are
  // present both in auipc and in the follow-up load/store instructions.
  assembler.Fld(Assembler::f1, data_end, Assembler::x2);
  assembler.Flw(Assembler::f3, data_end, Assembler::x4);
  assembler.Fsd(Assembler::f5, data_end, Assembler::x6);
  assembler.Fsw(Assembler::f7, data_end, Assembler::x8);
  assembler.Lb(Assembler::x9, data_end);
  assembler.Lbu(Assembler::x10, data_end);
  assembler.Lh(Assembler::x11, data_end);
  assembler.Lhu(Assembler::x12, data_end);
  assembler.Lw(Assembler::x13, data_end);
  assembler.Sb(Assembler::x14, data_end, Assembler::x15);
  assembler.Sh(Assembler::x16, data_end, Assembler::x17);
  assembler.Sw(Assembler::x18, data_end, Assembler::x19);
  assembler.La(Assembler::x20, data_end);
  assembler.Bcc(Assembler::Condition::kEqual, Assembler::x1, Assembler::x2, label);
  assembler.Bcc(Assembler::Condition::kNotEqual, Assembler::x3, Assembler::x4, label);
  assembler.Bcc(Assembler::Condition::kLess, Assembler::x5, Assembler::x6, label);
  assembler.Bcc(Assembler::Condition::kGreaterEqual, Assembler::x7, Assembler::x8, label);
  assembler.Bcc(Assembler::Condition::kBelow, Assembler::x9, Assembler::x10, label);
  assembler.Bcc(Assembler::Condition::kAboveEqual, Assembler::x11, Assembler::x12, label);
  assembler.Jal(Assembler::x1, label);
  assembler.Add(Assembler::x1, Assembler::x2, Assembler::x3);
  assembler.Addi(Assembler::x1, Assembler::x2, 42);
  assembler.Bind(&label);
  // Jalr have two alternate forms.
  assembler.Jalr(Assembler::x1, Assembler::x2, 42);
  assembler.Jalr(Assembler::x3, {.base = Assembler::x4, .disp = 42});
  assembler.Sw(Assembler::x1, {.base = Assembler::x2, .disp = 42});
  assembler.Jal(Assembler::x2, label);
  assembler.Beq(Assembler::x1, Assembler::x2, label);
  assembler.Bne(Assembler::x3, Assembler::x4, label);
  assembler.Blt(Assembler::x5, Assembler::x6, label);
  assembler.Bge(Assembler::x7, Assembler::x8, label);
  assembler.Bltu(Assembler::x9, Assembler::x10, label);
  assembler.Bgeu(Assembler::x11, Assembler::x12, label);
  assembler.Csrrc(Assembler::x1, Assembler::Csr::kVl, Assembler::x2);
  assembler.Csrrs(Assembler::x3, Assembler::Csr::kVtype, Assembler::x4);
  assembler.Csrrw(Assembler::x5, Assembler::Csr::kVlenb, Assembler::x6);
  assembler.Slli(Assembler::x1, Assembler::x2, 3);
  assembler.Srai(Assembler::x4, Assembler::x5, 6);
  assembler.Srli(Assembler::x7, Assembler::x8, 9);
  assembler.FcvtSW(Assembler::f1, Assembler::x2, Assembler::Rounding::kRmm);
  assembler.FcvtSWu(Assembler::f3, Assembler::x4);
  assembler.FcvtWS(Assembler::x1, Assembler::f2, Assembler::Rounding::kRmm);
  assembler.FcvtWuS(Assembler::x3, Assembler::f4);
  assembler.FsqrtS(Assembler::f1, Assembler::f2, Assembler::Rounding::kRmm);
  assembler.FsqrtD(Assembler::f3, Assembler::f4);
  assembler.PrefetchI({.base = Assembler::x1, .disp = 32});
  assembler.PrefetchR({.base = Assembler::x2, .disp = 64});
  assembler.PrefetchW({.base = Assembler::x3, .disp = 96});
  assembler.Li(Assembler::x15, static_cast<int32_t>(0xaf));
  assembler.Seqz(Assembler::x20, Assembler::x10);
  assembler.Snez(Assembler::x2, Assembler::x9);
  assembler.Sltz(Assembler::x30, Assembler::x1);
  assembler.Sgtz(Assembler::x25, Assembler::x16);
  assembler.J(0x42);
  assembler.Jal(-0x26);
  assembler.Jr(Assembler::x19);
  assembler.Jalr(Assembler::x7);
  // Move target position for more than 2048 bytes down to ensure auipc would use non-zero
  // immediate.
  for (size_t index = 138; index < 1200; ++index) {
    assembler.TwoByte(uint16_t{0});
  }
  assembler.Fld(Assembler::f1, data_begin, Assembler::x2);
  assembler.Flw(Assembler::f3, data_begin, Assembler::x4);
  assembler.Fsd(Assembler::f5, data_begin, Assembler::x6);
  assembler.Fsw(Assembler::f7, data_begin, Assembler::x8);
  assembler.Lb(Assembler::x9, data_begin);
  assembler.Lbu(Assembler::x10, data_begin);
  assembler.Lh(Assembler::x11, data_begin);
  assembler.Lhu(Assembler::x12, data_begin);
  assembler.Lw(Assembler::x13, data_begin);
  assembler.Sb(Assembler::x14, data_begin, Assembler::x15);
  assembler.Sh(Assembler::x16, data_begin, Assembler::x17);
  assembler.Sw(Assembler::x18, data_begin, Assembler::x19);
  assembler.La(Assembler::x20, data_begin);
  assembler.Bind(&data_end);
  assembler.Bexti(Assembler::x16, Assembler::x1, 20);
  assembler.Rori(Assembler::x5, Assembler::x3, 5);
  assembler.Finalize();

  // clang-format off
  static const uint16_t kCodeTemplate[] = {
    0x1117, 0x0000,     // begin: auipc   x2, 4096
    0x3087, 0x9c81,     //        fld     f1, -1592(x2)
    0x1217, 0x0000,     //        auipc   x4, 4096
    0x2187, 0x9c02,     //        flw     f3, -1600(x4)
    0x1317, 0x0000,     //        auipc   x6, 4096
    0x3c27, 0x9a53,     //        fsd     f5, -1608(x6)
    0x1417, 0x0000,     //        auipc   x8, 4096
    0x2827, 0x9a74,     //        fsw     f7, -1616(x8)
    0x1497, 0x0000,     //        auipc   x9, 4096
    0x8483, 0x9a84,     //        lb      x9, -1624(x9)
    0x1517, 0x0000,     //        auipc   x10, 4096
    0x4503, 0x9a05,     //        lbu     x10, -1632(x10)
    0x1597, 0x0000,     //        auipc   x11, 4096
    0x9583, 0x9985,     //        lh      x11, -1640(x11)
    0x1617, 0x0000,     //        auipc   x12, 4096
    0x5603, 0x9906,     //        lhu     x12, -1648(x12)
    0x1697, 0x0000,     //        auipc   x13, 4096
    0xa683, 0x9886,     //        lw      x13, -1656(x13)
    0x1797, 0x0000,     //        auipc   x15, 4096
    0x8023, 0x98e7,     //        sb      x14, -1664(x15)
    0x1897, 0x0000,     //        auipc   x17, 4096
    0x9c23, 0x9708,     //        sh      x16, -1672(x17)
    0x1997, 0x0000,     //        auipc   x19, 4096
    0xa823, 0x9729,     //        sw      x18, -1680(x19)
    0x1a17, 0x0000,     //        auipc   x20, 4096
    0x0a13, 0x968a,     //        addi    x20, x20, -1688
    0x8263, 0x0220,     //        beq     x1, x2, label
    0x9063, 0x0241,     //        bne     x3, x4, label
    0xce63, 0x0062,     //        blt     x5, x6, label
    0xdc63, 0x0083,     //        bge     x7, x8, label
    0xea63, 0x00a4,     //        bltu    x9, x10, label
    0xf863, 0x00c5,     //        bgeu    x11, x12, label
    0x00ef, 0x00c0,     //        jal     x1, label
    0x00b3, 0x0031,     //        add     x1, x2, x3
    0x0093, 0x02a1,     //        addi    x1, x2, 42
    0x00e7, 0x02a1,     // label: jalr    x1, x2, 42
    0x01e7, 0x02a2,     //        jalr    x3, 42(x4)
    0x2523, 0x0211,     //        sw      x1, 42(x2)
    0xf16f, 0xff5f,     //        jal     x2, label
    0x88e3, 0xfe20,     //        beq     x1, x2, label
    0x96e3, 0xfe41,     //        bne     x3, x4, label
    0xc4e3, 0xfe62,     //        blt     x5, x6, label
    0xd2e3, 0xfe83,     //        bge     x7, x8, label
    0xe0e3, 0xfea4,     //        bltu    x9, x10, label
    0xfee3, 0xfcc5,     //        bgeu    x11, x12, label
    0x30f3, 0xc201,     //        csrrc   x1, vl, x2
    0x21f3, 0xc212,     //        csrrs   x3, vtype, x4
    0x12f3, 0xc223,     //        csrrw   x5, vlenb, x6
    0x1093, 0x0031,     //        slli    x1, x2, 3
    0xd213, 0x4062,     //        srai    x4, x5, 6
    0x5393, 0x0094,     //        srli    x7, x8, 9
    0x40d3, 0xd001,     //        fcvt.s.w f1, x2, rmm
    0x71d3, 0xd012,     //        fcvt.s.wu f3, x4
    0x40d3, 0xc001,     //        fcvt.w.s x1, f2, rmm
    0x71d3, 0xc012,     //        fcvt.wu.s x3, f4
    0x40d3, 0x5801,     //        fsqrt.s f1, f2, rmm
    0x71d3, 0x5a02,     //        fsqrt.d f3, f4
    0xe013, 0x0200,     //        prefetch.i 32(x1)
    0x6013, 0x0411,     //        prefetch.r 64(x2)
    0xe013, 0x0631,     //        prefetch.w 96(x3)
    0x0793, 0x0af0,     //        addi x15, x15, 0xaf
    0x3a13, 0x0015,     //        sltiu x20, x10, 1
    0x3133, 0x0090,     //        sltu x2, x0, x9
    0xaf33, 0x0000,     //        slt x30, x1, x0
    0x2cb3, 0x0100,     //        slt x25, x0, x16
    0x006f, 0x0420,     //        jal zero, 0x42
    0xf0ef, 0xfdbf,     //        jal x1, -0x26
    0x8067, 0x0009,     //        jalr zero, x19, 0
    0x80e7, 0x0003,     //        jalr x1, x7, 0
    [ 138 ... 1199 ] = 0,//       padding
    0xf117, 0xffff,     //        auipc   x2, -4096
    0x3087, 0x6a01,     //        fld     f1,1696(x2)
    0xf217, 0xffff,     //        auipc   x4, -4096
    0x2187, 0x6982,     //        flw     f3,1688(x4)
    0xf317, 0xffff,     //        auipc   x6, -4096
    0x3827, 0x6853,     //        fsd     f5,1680(x6)
    0xf417, 0xffff,     //        auipc   x8, -4096
    0x2427, 0x6874,     //        fsw     f7,1672(x8)
    0xf497, 0xffff,     //        auipc   x9, -4096
    0x8483, 0x6804,     //        lb      x9,1664(x9)
    0xf517, 0xffff,     //        auipc   x10, -4096
    0x4503, 0x6785,     //        lbu     x10,1656(x10)
    0xf597, 0xffff,     //        auipc   x11, -4096
    0x9583, 0x6705,     //        lh      x11,1648(x11)
    0xf617, 0xffff,     //        auipc   x12, -4096
    0x5603, 0x6686,     //        lhu     x12,1640(x12)
    0xf697, 0xffff,     //        auipc   x13, -4096
    0xa683, 0x6606,     //        lw      x13,1632(x13)
    0xf797, 0xffff,     //        auipc   x15, -4096
    0x8c23, 0x64e7,     //        sb      x14,1624(x15)
    0xf897, 0xffff,     //        auipc   x17, -4096
    0x9823, 0x6508,     //        sh      x16,1616(x17)
    0xf997, 0xffff,     //        auipc   x19, -4096
    0xa423, 0x6529,     //        sw      x18,1608(x19)
    0xfa17, 0xffff,     //        auipc   x20, -4096
    0x0a13, 0x640a,     //        addi    x20,x20,1600
    0xd813, 0x4940,     //        bexti    x16,x1,20
    0xd293, 0x6051,     //        rori    x5, x3, 5
  };                    // end:
  // clang-format on

  return CompareCode(std::begin(kCodeTemplate), std::end(kCodeTemplate), code, CPUArch::kRiscv64);
}

}  // namespace riscv32

namespace riscv64 {

bool AssemblerTest() {
  MachineCode code;
  Assembler assembler(&code);
  Assembler::Label data_begin, data_end;
  assembler.Bind(&data_begin);
  // We test loads and stores twice to ensure that both positive and negative immediates are
  // present both in auipc and in the follow-up load/store instructions.
  assembler.Ld(Assembler::x1, data_end);
  assembler.Lwu(Assembler::x2, data_end);
  assembler.Sd(Assembler::x3, data_end, Assembler::x4);
  assembler.Bcc(Assembler::Condition::kAlways, Assembler::x1, Assembler::x2, 48);
  assembler.Bcc(Assembler::Condition::kEqual, Assembler::x3, Assembler::x4, 44);
  assembler.Bcc(Assembler::Condition::kNotEqual, Assembler::x5, Assembler::x6, 40);
  assembler.Bcc(Assembler::Condition::kLess, Assembler::x7, Assembler::x8, 36);
  assembler.Bcc(Assembler::Condition::kGreaterEqual, Assembler::x9, Assembler::x10, 32);
  assembler.Bcc(Assembler::Condition::kBelow, Assembler::x11, Assembler::x12, 28);
  assembler.Bcc(Assembler::Condition::kAboveEqual, Assembler::x13, Assembler::x14, 24);
  assembler.Jal(Assembler::x1, 20);
  assembler.Add(Assembler::x1, Assembler::x2, Assembler::x3);
  assembler.Addw(Assembler::x1, Assembler::x2, Assembler::x3);
  assembler.Addi(Assembler::x1, Assembler::x2, 42);
  assembler.Addiw(Assembler::x1, Assembler::x2, 42);
  // Jalr have two alternate forms.
  assembler.Jalr(Assembler::x1, Assembler::x2, 42);
  assembler.Jalr(Assembler::x3, {.base = Assembler::x4, .disp = 42});
  assembler.Sw(Assembler::x1, {.base = Assembler::x2, .disp = 42});
  assembler.Sd(Assembler::x3, {.base = Assembler::x4, .disp = 42});
  assembler.Jal(Assembler::x2, -16);
  assembler.Beq(Assembler::x1, Assembler::x2, -20);
  assembler.Bne(Assembler::x3, Assembler::x4, -24);
  assembler.Blt(Assembler::x5, Assembler::x6, -28);
  assembler.Bge(Assembler::x7, Assembler::x8, -32);
  assembler.Bltu(Assembler::x9, Assembler::x10, -36);
  assembler.Bgeu(Assembler::x11, Assembler::x12, -40);
  assembler.Bcc(Assembler::Condition::kAlways, Assembler::x13, Assembler::x14, -44);
  assembler.Csrrc(Assembler::x1, Assembler::Csr::kVl, 2);
  assembler.Csrrs(Assembler::x3, Assembler::Csr::kVtype, 4);
  assembler.Csrrw(Assembler::x5, Assembler::Csr::kVlenb, 6);
  assembler.Csrrci(Assembler::x7, Assembler::Csr::kVl, 8);
  assembler.Csrrsi(Assembler::x9, Assembler::Csr::kVtype, 10);
  assembler.Csrrwi(Assembler::x11, Assembler::Csr::kVlenb, 12);
  assembler.Slliw(Assembler::x1, Assembler::x2, 3);
  assembler.Sraiw(Assembler::x4, Assembler::x5, 6);
  assembler.Srliw(Assembler::x7, Assembler::x8, 9);
  assembler.FcvtDL(Assembler::f1, Assembler::x2, Assembler::Rounding::kRmm);
  assembler.FcvtDLu(Assembler::f3, Assembler::x4);
  assembler.FcvtLD(Assembler::x1, Assembler::f2, Assembler::Rounding::kRmm);
  assembler.FcvtLuD(Assembler::x3, Assembler::f4);
  assembler.FsqrtS(Assembler::f1, Assembler::f2, Assembler::Rounding::kRmm);
  assembler.FsqrtD(Assembler::f3, Assembler::f4);
  assembler.PrefetchI({.base = Assembler::x1, .disp = 32});
  assembler.PrefetchR({.base = Assembler::x2, .disp = 64});
  assembler.PrefetchW({.base = Assembler::x3, .disp = 96});
  assembler.Li(Assembler::x10, static_cast<int64_t>(0xaaaa'0aa0'aaa0'0aaa));
  assembler.Ret();
  assembler.Call(data_end);
  assembler.Tail(data_end);
  assembler.Bgt(Assembler::x4, Assembler::x0, data_end);
  assembler.Bgtu(Assembler::x2, Assembler::x20, data_end);
  assembler.Ble(Assembler::x1, Assembler::x30, data_end);
  assembler.Bleu(Assembler::x8, Assembler::x16, data_end);
  assembler.Beqz(Assembler::x5, data_end);
  assembler.Bnez(Assembler::x4, data_end);
  assembler.Blez(Assembler::x2, data_end);
  assembler.Bgez(Assembler::x3, data_end);
  assembler.Bltz(Assembler::x9, data_end);
  assembler.Bgtz(Assembler::x12, data_end);
  // Move target position for more than 2048 bytes down to ensure auipc would use non-zero
  // immediate.
  for (size_t index = 142; index < 1200; ++index) {
    assembler.TwoByte(uint16_t{0});
  }
  assembler.Ld(Assembler::x1, data_begin);
  assembler.Lwu(Assembler::x2, data_begin);
  assembler.Sd(Assembler::x3, data_begin, Assembler::x4);
  assembler.Bind(&data_end);
  assembler.SextW(Assembler::x15, Assembler::x12);
  assembler.AddUW(Assembler::x14, Assembler::x22, Assembler::x29);
  assembler.ZextW(Assembler::x13, Assembler::x21);
  assembler.Sh3add(Assembler::x13, Assembler::x9, Assembler::x10);
  assembler.Bexti(Assembler::x16, Assembler::x1, 53);
  assembler.Rori(Assembler::x22, Assembler::x30, 43);
  assembler.Roriw(Assembler::x29, Assembler::x2, 30);
  assembler.Ror(Assembler::x14, Assembler::x1, Assembler::x10);
  assembler.Rorw(Assembler::x25, Assembler::x5, Assembler::x4);
  assembler.Not(Assembler::x10, Assembler::x4);
  assembler.Neg(Assembler::x11, Assembler::x3);
  assembler.Negw(Assembler::x12, Assembler::x2);
  assembler.SextB(Assembler::x22, Assembler::x7);
  assembler.SextH(Assembler::x23, Assembler::x8);
  assembler.Finalize();

  // clang-format off
  static const uint16_t kCodeTemplate[] = {
    0x1097, 0x0000,     // begin: auipc   x1, 4096
    0xb083, 0x9780,     //        ld,     x1, -1672(x1)
    0x1117, 0x0000,     //        auipc   x2, 4096
    0x6103, 0x9701,     //        lwu     x2,-1680(x2)
    0x1217, 0x0000,     //        auipc   x4, 4096
    0x3423, 0x9632,     //        sd      x3,-1688(x4)
    0x006f, 0x0300,     //        jal     x0, label
    0x8663, 0x0241,     //        beq     x1, x2, label
    0x9463, 0x0262,     //        bne     x3, x4, label
    0xc263, 0x0283,     //        blt     x5, x6, label
    0xd063, 0x02a4,     //        bge     x7, x8, label
    0xee63, 0x00c5,     //        bltu    x9, x10, label
    0xfc63, 0x00e6,     //        bgeu    x11, x12, label
    0x00ef, 0x0140,     //        jal     x1, label
    0x00b3, 0x0031,     //        add     x1, x2, x3
    0x00bb, 0x0031,     //        addw    x1, x2, x3
    0x0093, 0x02a1,     //        addi    x1, x2, 42
    0x009b, 0x02a1,     //        addiw   x1, x2, 42
    0x00e7, 0x02a1,     // label: jalr    x1, x2, 42
    0x01e7, 0x02a2,     //        jalr    x3, 42(x4)
    0x2523, 0x0211,     //        sw      x1, 42(x2)
    0x3523, 0x0232,     //        sd      x3, 42(x4)
    0xf16f, 0xff1f,     //        jal     x2, label
    0x86e3, 0xfe20,     //        beq     x1, x2, label
    0x94e3, 0xfe41,     //        bne     x3, x4, label
    0xc2e3, 0xfe62,     //        blt     x5, x6, label
    0xd0e3, 0xfe83,     //        bge     x7, x8, label
    0xeee3, 0xfca4,     //        bltu    x9, x10, label
    0xfce3, 0xfcc5,     //        bgeu    x11, x12, label
    0xf06f, 0xfd5f,     //        jal     x0, label
    0x70f3, 0xc201,     //        csrrc   x1, vl, 2
    0x61f3, 0xc212,     //        csrrs   x3, vtype, 4
    0x52f3, 0xc223,     //        csrrw   x5, vlenb, 6
    0x73f3, 0xc204,     //        csrrci  x7, vl, 8
    0x64f3, 0xc215,     //        csrrsi  x9, vtype, 10
    0x55f3, 0xc226,     //        csrrwi  x11, vlenb, 12
    0x109b, 0x0031,     //        slliw   x1, x2, 3
    0xd21b, 0x4062,     //        sraiw   x4, x5, 6
    0x539b, 0x0094,     //        srliw   x7, x8, 9
    0x40d3, 0xd221,     //        fcvt.d.l f1, x2, rmm
    0x71d3, 0xd232,     //        fcvt.d.lu f3, x4
    0x40d3, 0xc221,     //        fcvt.l.d x1, f2, rmm
    0x71d3, 0xc232,     //        fcvt.lu.d x3, f4
    0x40d3, 0x5801,     //        fsqrt.s f1, f2, rmm
    0x71d3, 0x5a02,     //        fsqrt.d f3, f4
    0xe013, 0x0200,     //        prefetch.i 32(x1)
    0x6013, 0x0411,     //        prefetch.r 64(x2)
    0xe013, 0x0631,     //        prefetch.w 96(x3)
    0x5537,0xfd55,      //        lui a0, 0xfd555
    0x0513, 0x0555,     //        addi a0, a0, 85
    0x1513, 0x00d5,     //        slli a0, a0, 0xd
    0x0513, 0x0ab5,     //        addi a0, a0, 171
    0x1513, 0x00c5,     //        slli a0, a0, 0xc
    0x0513, 0xa015,     //        addi a0, a0, -1535
    0x1513, 0x00c5,     //        slli a0, a0, 0xc
    0x0513, 0xaaa5,     //        addi a0,a0,-1366
    0x8067, 0x0000,     //        ret
    0x1317, 0x0000,     //        auipc x6, 0x1
    0x00e7, 0x8943,     //        jalr x1, x6, -1900
    0x1317, 0x0000,     //        auipc x6, 0x1
    0x0067, 0x88c3,     //        jalr x0, x6, -1908
    0x42e3, 0x0840,     //        blt x0, x4, 0x884
    0x60e3, 0x082a,     //        bltu x20, x2, 0x880
    0x5ee3, 0x061f,     //        bge x30, x1, 0x87c
    0x7ce3, 0x0688,     //        bgeu x16, x8, 0x878
    0x8ae3, 0x0602,     //        beq x5, 0x874
    0x18e3, 0x0602,     //        bne x4, 0x870
    0x56e3, 0x0620,     //        ble x2, 0x86c
    0xd4e3, 0x0601,     //        bge x3, 0x868
    0xc2e3, 0x0604,     //        blt x9, 0x864
    0x40e3, 0x06c0,     //        bgt x12, 0x860
    [ 142 ... 1199 ] = 0,//        padding
    0xf097, 0xffff,     //        auipc   x1, -4096
    0xb083, 0x6a00,     //        ld      x1, 1696(x1)
    0xf117, 0xffff,     //        auipc   x2, -4096
    0x6103, 0x6981,     //        lwu     x2, 1688(x2)
    0xf217, 0xffff,     //        auipc   x4, -4096
    0x3823, 0x6832,     //        sd      x3, 1680(x4)
    0x079b, 0x0006,     //        addi.w x15, x12, 0
    0x073b, 0x09db,     //        add.uw x14, x22, x29
    0x86bb, 0x080a,     //        add.uw x13, x21, zero
    0xe6b3, 0x20a4,     //        sh3add x13, x9, x10
    0xd813, 0x4b50,     //        bexti x16, x1, 53
    0x5b13, 0x62bf,     //        rori x22, x30, 43
    0x5e9b, 0x61e1,     //        roriw x29, x2, 30
    0xd733, 0x60a0,     //        ror x14, x1, x10
    0xdcbb, 0x6042,     //        rorw x25, x5, x4
    0x4513, 0xfff2,     //        xori x10, x4, -1
    0x05b3, 0x4030,     //        sub x11, zero, x3
    0x063b, 0x4020,     //        subw x12, zero, x2
    0x9b13, 0x6043,     //        sext.b x22, x7
    0x1b93, 0x6054,     //        sext.h x23, x8
  };                    // end:
  // clang-format on

  return CompareCode(std::begin(kCodeTemplate), std::end(kCodeTemplate), code, CPUArch::kRiscv64);
}

}  // namespace riscv64

namespace x86_32 {

bool AssemblerTest() {
  MachineCode code;
  Assembler assembler(&code);
  assembler.Movl(Assembler::eax, {.base = Assembler::esp, .disp = 4});
  assembler.CmpXchgl({.base = Assembler::esp, .disp = 4}, Assembler::eax);
  assembler.Subl(Assembler::esp, 16);
  assembler.Movl({.base = Assembler::esp}, Assembler::eax);
  assembler.Push(Assembler::esp);
  assembler.Push(0xcccccccc);
  assembler.Pushl({.base = Assembler::esp, .disp = 0x428});
  assembler.Popl({.base = Assembler::esp, .disp = 0x428});
  assembler.Movl(Assembler::ecx, 0xcccccccc);
  assembler.Call(Assembler::ecx);
  assembler.Movl(Assembler::eax, {.base = Assembler::esp, .disp = 8});
  assembler.Addl(Assembler::esp, 24);
  assembler.Ret();
  assembler.Finalize();

  // clang-format off
  static const uint8_t kCodeTemplate[] = {
    0x8b, 0x44, 0x24, 0x04,                    // mov     0x4(%esp),%eax
    0x0f, 0xb1, 0x44, 0x24, 0x04,              // cmpxchg 0x4(%esp),%eax
    0x83, 0xec, 0x10,                          // sub     $16, %esp
    0x89, 0x04, 0x24,                          // mov     %eax,(%esp)
    0x54,                                      // push    %esp
    0x68, 0xcc, 0xcc, 0xcc, 0xcc,              // push    $cccccccc
    0xff, 0xb4, 0x24, 0x28, 0x04, 0x00, 0x00,  // pushl   0x428(%esp)
    0x8f, 0x84, 0x24, 0x28, 0x04, 0x00, 0x00,  // popl    0x428(%esp)
    0xb9, 0xcc, 0xcc, 0xcc, 0xcc,              // mov     $cccccccc, %ecx
    0xff, 0xd1,                                // call    *%ecx
    0x8b, 0x44, 0x24, 0x08,                    // mov     0x8(%esp),%eax
    0x83, 0xc4, 0x18,                          // add     $24, %esp
    0xc3                                       //  ret
  };
  // clang-format on

  return CompareCode(std::begin(kCodeTemplate), std::end(kCodeTemplate), code, CPUArch::kX86_64);
}

}  // namespace x86_32

namespace x86_64 {

bool AssemblerTest() {
  MachineCode code;
  Assembler assembler(&code);
  assembler.Movq(Assembler::rax, Assembler::rdi);
  assembler.Subq(Assembler::rsp, 16);
  assembler.Movq({.base = Assembler::rsp}, Assembler::rax);
  assembler.Movq({.base = Assembler::rsp, .disp = 8}, Assembler::rax);
  assembler.Movl({.base = Assembler::rax, .disp = 16}, 239);
  assembler.Movq(Assembler::r11, {.base = Assembler::rsp});
  assembler.Addq(Assembler::rsp, 16);
  assembler.Ret();
  assembler.Finalize();

  // clang-format off
  static const uint8_t kCodeTemplate[] = {
    0x48, 0x89, 0xf8,               // mov %rdi, %rax
    0x48, 0x83, 0xec, 0x10,         // sub $0x10, %rsp
    0x48, 0x89, 0x04, 0x24,         // mov rax, (%rsp)
    0x48, 0x89, 0x44, 0x24, 0x08,   // mov rax, 8(%rsp)
    0xc7, 0x40, 0x10, 0xef, 0x00,   // movl $239, 0x10(%rax)
    0x00, 0x00,
    0x4c, 0x8b, 0x1c, 0x24,         // mov (%rsp), r11
    0x48, 0x83, 0xc4, 0x10,         // add $0x10, %rsp
    0xc3                            // ret
  };
  // clang-format on

  return CompareCode(std::begin(kCodeTemplate), std::end(kCodeTemplate), code, CPUArch::kX86_64);
}

}  // namespace x86_64

#if defined(__i386__)

namespace x86_32 {

bool LabelTest() {
  MachineCode code;
  CodeEmitter as(&code);
  Assembler::Label skip, skip2, back, end;
  as.Call(bit_cast<const void*>(&Callee));
  as.Jmp(skip);
  as.Movl(Assembler::eax, 2);
  as.Bind(&skip);
  as.Addl(Assembler::eax, 8);
  as.Jmp(skip2);
  as.Bind(&back);
  as.Addl(Assembler::eax, 12);
  as.Jmp(end);
  as.Bind(&skip2);
  as.Jmp(back);
  as.Bind(&end);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  int result = exec.get<int()>()();
  return result == 239 + 8 + 12;
}

bool CondTest1() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::eax, 0xcccccccc);
  as.Movl(Assembler::edx, {.base = Assembler::esp, .disp = 4});  // arg1.
  as.Movl(Assembler::ecx, {.base = Assembler::esp, .disp = 8});  // arg2.
  as.Cmpl(Assembler::edx, Assembler::ecx);
  as.Setcc(Assembler::Condition::kEqual, Assembler::eax);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t(int, int);
  auto target_func = exec.get<TestFunc>();
  uint32_t result = target_func(1, 2);
  if (result != 0xcccccc00) {
    TRACE_AND_ALOGE("Bug in seteq(not equal): %x", result);
    return false;
  }
  result = target_func(-1, -1);
  if (result != 0xcccccc01) {
    TRACE_AND_ALOGE("Bug in seteq(equal): %x", result);
    return false;
  }
  return true;
}

bool CondTest2() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::edx, {.base = Assembler::esp, .disp = 4});  // arg1.
  as.Movl(Assembler::ecx, {.base = Assembler::esp, .disp = 8});  // arg2.
  as.Xorl(Assembler::eax, Assembler::eax);
  as.Testb(Assembler::edx, Assembler::ecx);
  as.Setcc(Assembler::Condition::kNotZero, Assembler::eax);
  as.Xchgl(Assembler::eax, Assembler::ecx);
  as.Xchgl(Assembler::ecx, Assembler::eax);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t(int, int);
  auto target_func = exec.get<TestFunc>();
  uint32_t result = target_func(0x11, 1);
  if (result != 0x1) {
    TRACE_AND_ALOGE("Bug in testb(not zero): %x", result);
    return false;
  }
  result = target_func(0x11, 0x8);
  if (result != 0x0) {
    TRACE_AND_ALOGE("Bug in testb(zero): %x", result);
    return false;
  }
  return true;
}

bool JccTest() {
  MachineCode code;
  CodeEmitter as(&code);
  Assembler::Label equal, above, below, done;
  as.Movl(Assembler::edx, {.base = Assembler::esp, .disp = 4});  // arg1.
  as.Movl(Assembler::ecx, {.base = Assembler::esp, .disp = 8});  // arg2.
  as.Cmpl(Assembler::edx, Assembler::ecx);
  as.Jcc(Assembler::Condition::kEqual, equal);
  as.Jcc(Assembler::Condition::kBelow, below);
  as.Jcc(Assembler::Condition::kAbove, above);

  as.Movl(Assembler::eax, 13);
  as.Jmp(done);

  as.Bind(&equal);
  as.Movl(Assembler::eax, 0u);
  as.Jmp(done);

  as.Bind(&below);
  as.Movl(Assembler::eax, -1);
  as.Jmp(done);

  as.Bind(&above);
  as.Movl(Assembler::eax, 1);
  as.Jmp(done);

  as.Bind(&done);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = int(int, int);
  auto target_func = exec.get<TestFunc>();
  int result = target_func(1, 1);
  if (result != 0) {
    TRACE_AND_ALOGE("Bug in jcc(equal): %x", result);
    return false;
  }
  result = target_func(1, 0);
  if (result != 1) {
    TRACE_AND_ALOGE("Bug in jcc(above): %x", result);
    return false;
  }
  result = target_func(0, 1);
  if (result != -1) {
    TRACE_AND_ALOGE("Bug in jcc(below): %x", result);
    return false;
  }
  return true;
}

bool ShiftTest() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::eax, {.base = Assembler::esp, .disp = 4});
  as.Shll(Assembler::eax, int8_t{2});
  as.Shrl(Assembler::eax, int8_t{1});
  as.Movl(Assembler::ecx, 3);
  as.ShllByCl(Assembler::eax);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t(uint32_t);
  uint32_t result = exec.get<TestFunc>()(22);
  return result == (22 << 4);
}

bool LogicTest() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::eax, {.base = Assembler::esp, .disp = 4});
  as.Movl(Assembler::ecx, 0x1);
  as.Xorl(Assembler::eax, Assembler::ecx);
  as.Movl(Assembler::ecx, 0xf);
  as.Andl(Assembler::eax, Assembler::ecx);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t(uint32_t);
  uint32_t result = exec.get<TestFunc>()(239);
  return result == ((239 ^ 1) & 0xf);
}

bool BsrTest() {
  MachineCode code;
  CodeEmitter as(&code);

  as.Movl(Assembler::ecx, {.base = Assembler::esp, .disp = 4});
  as.Movl(Assembler::edx, 239);
  as.Bsrl(Assembler::eax, Assembler::ecx);
  as.Cmovl(Assembler::Condition::kZero, Assembler::eax, Assembler::edx);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t(uint32_t arg);
  auto func = exec.get<TestFunc>();
  return func(0) == 239 && func(1 << 15) == 15;
}

bool CallFPTest() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Push(0x3f800000);
  as.Push(0x40000000);
  as.Call(bit_cast<const void*>(&FloatFunc));
  as.Fstps({.base = Assembler::esp});
  as.Pop(Assembler::eax);
  as.Addl(Assembler::esp, 4);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t();
  uint32_t result = exec.get<TestFunc>()();
  return result == 0x3f800000;
}

bool XmmTest() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::eax, 0x3f800000);
  as.Movd(Assembler::xmm0, Assembler::eax);
  as.Movl(Assembler::eax, 0x40000000);
  as.Movd(Assembler::xmm5, Assembler::eax);
  as.Addss(Assembler::xmm0, Assembler::xmm5);
  as.Movd(Assembler::eax, Assembler::xmm0);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t();
  uint32_t result = exec.get<TestFunc>()();
  return result == 0x40400000;
}

bool ReadGlobalTest() {
  MachineCode code;
  CodeEmitter as(&code);
  static const uint32_t kData[] __attribute__((aligned(16))) =  // NOLINT
      {0x00112233, 0x44556677, 0x8899aabb, 0xccddeeff};
  as.Movsd(Assembler::xmm0, {.disp = bit_cast<int32_t>(&kData)});
  as.Movdqa(Assembler::xmm1, {.disp = bit_cast<int32_t>(&kData)});
  as.Movl(Assembler::eax, {.base = Assembler::esp, .disp = 4});
  as.Movl(Assembler::ecx, {.base = Assembler::esp, .disp = 8});
  as.Movsd({.base = Assembler::eax}, Assembler::xmm0);
  as.Movdqu({.base = Assembler::ecx}, Assembler::xmm1);

  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = void(void*, void*);
  uint8_t res1[8];
  uint8_t res2[16];
  exec.get<TestFunc>()(res1, res2);

  return (memcmp(res1, kData, 8) == 0) && (memcmp(res2, kData, 16) == 0);
}

}  // namespace x86_32

#elif defined(__amd64__)

namespace x86_64 {

bool LabelTest() {
  MachineCode code;
  CodeEmitter as(&code);
  Assembler::Label skip, skip2, back, end;
  as.Call(bit_cast<const void*>(&Callee));
  as.Jmp(skip);
  as.Movl(Assembler::rax, 2);
  as.Bind(&skip);
  as.Addb(Assembler::rax, {end});
  as.Jmp(skip2);
  as.Bind(&back);
  as.Addl(Assembler::rax, 12);
  as.Jmp(end);
  as.Bind(&skip2);
  as.Jmp(back);
  as.Bind(&end);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = int();
  int result = exec.get<TestFunc>()();
  return result == uint8_t(239 + 0xc3) + 12;
}

bool CondTest1() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::rax, 0xcccccccc);
  as.Cmpl(Assembler::rdi, Assembler::rsi);
  as.Setcc(Assembler::Condition::kEqual, Assembler::rax);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  std::string code_str;
  code.AsString(&code_str, InstructionSize::OneByte);
  using TestFunc = uint32_t(int, int);
  auto target_func = exec.get<TestFunc>();
  uint32_t result;
  result = target_func(1, 2);
  if (result != 0xcccccc00) {
    TRACE_AND_ALOGE("Bug in seteq(not equal): %x", result);
    return false;
  }
  result = target_func(-1, -1);
  if (result != 0xcccccc01) {
    TRACE_AND_ALOGE("Bug in seteq(equal): %x", result);
    return false;
  }
  return true;
}

bool CondTest2() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::rdx, Assembler::rdi);  // arg1.
  as.Movl(Assembler::rcx, Assembler::rsi);  // arg2.
  as.Xorl(Assembler::rax, Assembler::rax);
  as.Testb(Assembler::rdx, Assembler::rcx);
  as.Setcc(Assembler::Condition::kNotZero, Assembler::rax);
  as.Xchgq(Assembler::rax, Assembler::rcx);
  as.Xchgq(Assembler::rcx, Assembler::rax);
  as.Xchgq(Assembler::rcx, Assembler::r11);
  as.Xchgq(Assembler::r11, Assembler::rcx);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t(int, int);
  auto target_func = exec.get<TestFunc>();
  uint32_t result = target_func(0x11, 1);
  if (result != 0x1) {
    printf("Bug in testb(not zero): %x", result);
    return false;
  }
  result = target_func(0x11, 0x8);
  if (result != 0x0) {
    printf("Bug in testb(zero): %x", result);
    return false;
  }
  return true;
}

bool JccTest() {
  MachineCode code;
  CodeEmitter as(&code);
  Assembler::Label equal, above, below, done;
  as.Cmpl(Assembler::rdi, Assembler::rsi);
  as.Jcc(Assembler::Condition::kEqual, equal);
  as.Jcc(Assembler::Condition::kBelow, below);
  as.Jcc(Assembler::Condition::kAbove, above);

  as.Movl(Assembler::rax, 13);
  as.Jmp(done);

  as.Bind(&equal);
  as.Movq(Assembler::rax, 0);
  as.Jmp(done);

  as.Bind(&below);
  as.Movl(Assembler::rax, -1);
  as.Jmp(done);

  as.Bind(&above);
  as.Movl(Assembler::rax, 1);
  as.Jmp(done);

  as.Bind(&done);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = int(int, int);
  auto target_func = exec.get<TestFunc>();
  int result;
  result = target_func(1, 1);
  if (result != 0) {
    TRACE_AND_ALOGE("Bug in jcc(equal): %x", result);
    return false;
  }
  result = target_func(1, 0);
  if (result != 1) {
    TRACE_AND_ALOGE("Bug in jcc(above): %x", result);
    return false;
  }
  result = target_func(0, 1);
  if (result != -1) {
    TRACE_AND_ALOGE("Bug in jcc(below): %x", result);
    return false;
  }
  return true;
}

bool ReadWriteTest() {
  MachineCode code;
  CodeEmitter as(&code);

  as.Movq(Assembler::rax, 0);
  as.Movb(Assembler::rax, {.base = Assembler::rdi});
  as.Movl(Assembler::rcx, {.base = Assembler::rsi});
  as.Addl(Assembler::rax, Assembler::rcx);
  as.Movl({.base = Assembler::rsi}, Assembler::rax);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t(uint8_t*, uint32_t*);
  uint8_t p1[4] = {0x12, 0x34, 0x56, 0x78};
  uint32_t p2 = 0x239;
  uint32_t result = exec.get<TestFunc>()(p1, &p2);
  return (result == 0x239 + 0x12) && (p2 == result);
}

bool CallFPTest() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::rax, 0x40000000);
  as.Movd(Assembler::xmm0, Assembler::rax);
  as.Movl(Assembler::rax, 0x3f800000);
  as.Movd(Assembler::xmm1, Assembler::rax);
  as.Call(bit_cast<const void*>(&FloatFunc));
  as.Movd(Assembler::rax, Assembler::xmm0);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t();
  uint32_t result = exec.get<TestFunc>()();
  return result == 0x3f800000;
}

bool XmmTest() {
  MachineCode code;
  CodeEmitter as(&code);
  as.Movl(Assembler::rax, 0x40000000);
  as.Movd(Assembler::xmm0, Assembler::rax);
  as.Movl(Assembler::rax, 0x3f800000);
  as.Movd(Assembler::xmm11, Assembler::rax);
  as.Addss(Assembler::xmm0, Assembler::xmm11);
  as.Movaps(Assembler::xmm12, Assembler::xmm0);
  as.Addss(Assembler::xmm0, Assembler::xmm12);
  as.Movapd(Assembler::xmm14, Assembler::xmm1);
  as.Movd(Assembler::rax, Assembler::xmm0);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t();
  uint32_t result = exec.get<TestFunc>()();
  return result == 0x40c00000;
}

bool XmmMemTest() {
  MachineCode code;
  CodeEmitter as(&code);

  as.Movsd(Assembler::xmm0, {.base = Assembler::rdi});
  as.Movaps(Assembler::xmm12, Assembler::xmm0);
  as.Addsd(Assembler::xmm12, Assembler::xmm12);
  as.Movsd({.base = Assembler::rdi}, Assembler::xmm12);
  as.Movq(Assembler::rax, Assembler::xmm0);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  double d = 239.0;
  char bits[16], *p = bits + 5;
  memcpy(p, &d, sizeof(d));

  using TestFunc = uint64_t(char* p);
  uint64_t result = exec.get<TestFunc>()(p);
  uint64_t doubled = *bit_cast<uint64_t*>(p);
  return result == 0x406de00000000000ULL && doubled == 0x407de00000000000ULL;
}

bool MovsxblRexTest() {
  MachineCode code;
  CodeEmitter as(&code);

  as.Xorl(Assembler::rdx, Assembler::rdx);
  as.Movl(Assembler::rsi, 0xdeadff);
  // CodeEmitter should use REX prefix to encode SIL.
  // Without REX DH is used.
  as.Movsxbl(Assembler::rax, Assembler::rsi);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t();
  uint32_t result = exec.get<TestFunc>()();

  return result == 0xffffffff;
}

bool MovzxblRexTest() {
  MachineCode code;
  CodeEmitter as(&code);

  as.Xorl(Assembler::rdx, Assembler::rdx);
  as.Movl(Assembler::rsi, 0xdeadff);
  // CodeEmitter should use REX prefix to encode SIL.
  // Without REX DH is used.
  as.Movzxbl(Assembler::rax, Assembler::rsi);
  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t();
  uint32_t result = exec.get<TestFunc>()();

  return result == 0x000000ff;
}

bool ShldlRexTest() {
  MachineCode code;
  CodeEmitter as(&code);

  as.Movl(Assembler::rdx, 0x12345678);
  // If most-significant bit is not encoded correctly with REX
  // RAX can be used instead R8 and R10 can be used instead RDX.
  // Init them all:
  as.Xorl(Assembler::r8, Assembler::r8);
  as.Movl(Assembler::rax, 0xdeadbeef);
  as.Movl(Assembler::r10, 0xdeadbeef);

  as.Shldl(Assembler::r8, Assembler::rdx, int8_t{8});
  as.Movl(Assembler::rcx, 8);
  as.ShldlByCl(Assembler::r8, Assembler::rdx);

  as.Movl(Assembler::rax, Assembler::r8);

  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t();
  uint32_t result = exec.get<TestFunc>()();

  return result == 0x1212;
}

bool ShrdlRexTest() {
  MachineCode code;
  CodeEmitter as(&code);

  as.Movl(Assembler::rdx, 0x12345678);
  // If most-significant bit is not encoded correctly with REX
  // RAX can be used instead R8 and R10 can be used instead RDX.
  // Init them all:
  as.Xorl(Assembler::r8, Assembler::r8);
  as.Movl(Assembler::rax, 0xdeadbeef);
  as.Movl(Assembler::r10, 0xdeadbeef);

  as.Shrdl(Assembler::r8, Assembler::rdx, int8_t{8});
  as.Movl(Assembler::rcx, 8);
  as.ShrdlByCl(Assembler::r8, Assembler::rdx);

  as.Movl(Assembler::rax, Assembler::r8);

  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = uint32_t();
  uint32_t result = exec.get<TestFunc>()();

  return result == 0x78780000;
}

bool ReadGlobalTest() {
  MachineCode code;
  CodeEmitter as(&code);
  static const uint32_t kData[] __attribute__((aligned(16))) =  // NOLINT
      {0x00112233, 0x44556677, 0x8899aabb, 0xccddeeff};
  // We couldn't read data from arbitrary address on x86_64, need address in first 2GiB.
  void* data =
      mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
  // Copy our global there.
  memcpy(data, kData, 16);
  int32_t data_offset = static_cast<int32_t>(bit_cast<intptr_t>(data));
  as.Movsd(Assembler::xmm0, {.disp = data_offset});
  as.Movdqa(Assembler::xmm1, {.disp = data_offset});
  as.Movsd({.base = Assembler::rdi}, Assembler::xmm0);
  as.Movdqu({.base = Assembler::rsi}, Assembler::xmm1);

  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = void(void*, void*);
  uint8_t res1[8];
  uint8_t res2[16];
  exec.get<TestFunc>()(res1, res2);

  munmap(data, 4096);

  return (memcmp(res1, kData, 8) == 0) && (memcmp(res2, kData, 16) == 0);
}

bool MemShiftTest() {
  MachineCode code;
  CodeEmitter as(&code);

  as.Push(Assembler::rdi);
  as.Movl(Assembler::rcx, 1);
  as.ShrlByCl({.base = Assembler::rsp});
  as.Addl(Assembler::rcx, 1);
  as.Movq(Assembler::rdi, Assembler::rsp);
  as.ShllByCl({.base = Assembler::rdi});
  as.Pop(Assembler::rax);

  as.Ret();
  as.Finalize();

  ScopedExecRegion exec(&code);

  using TestFunc = int(int x);
  int result = exec.get<TestFunc>()(0x10);

  return result == 0x20;
}

}  // namespace x86_64

#endif

#if defined(__i386__) || defined(__amd64__)

#if defined(__i386__)

extern "C" const uint8_t berberis_gnu_as_output_start[] asm(
    "berberis_gnu_as_output_start_x86_32");
extern "C" const uint8_t berberis_gnu_as_output_end[] asm(
    "berberis_gnu_as_output_end_x86_32");

namespace x86_32 {
void GenInsnsCommon(CodeEmitter* as);
void GenInsnsArch(CodeEmitter* as);
}  // namespace x86_32

#else

extern "C" const uint8_t berberis_gnu_as_output_start[] asm(
    "berberis_gnu_as_output_start_x86_64");
extern "C" const uint8_t berberis_gnu_as_output_end[] asm(
    "berberis_gnu_as_output_end_x86_64");

namespace x86_64 {
void GenInsnsCommon(CodeEmitter* as);
void GenInsnsArch(CodeEmitter* as);
}  // namespace x86_64

#endif

bool ExhaustiveTest() {
  MachineCode code;
  CodeEmitter as(&code);

#if defined(__i386__)
  berberis::x86_32::GenInsnsCommon(&as);
  berberis::x86_32::GenInsnsArch(&as);
#else
  berberis::x86_64::GenInsnsCommon(&as);
  berberis::x86_64::GenInsnsArch(&as);
#endif
  as.Finalize();

  return CompareCode(
      berberis_gnu_as_output_start, berberis_gnu_as_output_end, code, CPUArch::kX86_64);
}

bool MixedAssembler() {
  MachineCode code;
  x86_32::Assembler as32(&code);
  x86_64::Assembler as64(&code);
  x86_32::Assembler::Label lbl32;
  x86_64::Assembler::Label lbl64;

  as32.Jmp(lbl32);
  as32.Xchgl(x86_32::Assembler::eax, x86_32::Assembler::eax);
  as64.Jmp(lbl64);
  as64.Xchgl(x86_64::Assembler::rax, x86_64::Assembler::rax);
  as32.Bind(&lbl32);
  as32.Movl(x86_32::Assembler::eax, {.disp = 0});
  as64.Bind(&lbl64);
  as32.Finalize();
  as64.Finalize();

  // clang-format off
  static const uint8_t kCodeTemplate[] = {
    0xe9, 0x08, 0x00, 0x00, 0x00,              // jmp lbl32
    0x90,                                      // xchg %eax, %eax == nop
    0xe9, 0x07, 0x00, 0x00, 0x00,              // jmp lbl64
    0x87, 0xc0,                                // xchg %eax, %eax != nop
                                               // lbl32:
    0xa1, 0x00, 0x00, 0x00, 0x00               // movabs %eax, 0x0
                                               // lbl64:
  };
  // clang-format on

  return CompareCode(std::begin(kCodeTemplate), std::end(kCodeTemplate), code, CPUArch::kX86_64);
}
#endif

}  // namespace berberis

TEST(Assembler, AssemblerTest) {
  EXPECT_TRUE(berberis::riscv32::AssemblerTest());
  EXPECT_TRUE(berberis::riscv64::AssemblerTest());
  EXPECT_TRUE(berberis::x86_32::AssemblerTest());
  EXPECT_TRUE(berberis::x86_64::AssemblerTest());
#if defined(__i386__)
  EXPECT_TRUE(berberis::x86_32::LabelTest());
  EXPECT_TRUE(berberis::x86_32::CondTest1());
  EXPECT_TRUE(berberis::x86_32::CondTest2());
  EXPECT_TRUE(berberis::x86_32::JccTest());
  EXPECT_TRUE(berberis::x86_32::ShiftTest());
  EXPECT_TRUE(berberis::x86_32::LogicTest());
  EXPECT_TRUE(berberis::x86_32::CallFPTest());
  EXPECT_TRUE(berberis::x86_32::XmmTest());
  EXPECT_TRUE(berberis::x86_32::BsrTest());
  EXPECT_TRUE(berberis::x86_32::ReadGlobalTest());
#elif defined(__amd64__)
  EXPECT_TRUE(berberis::x86_64::LabelTest());
  EXPECT_TRUE(berberis::x86_64::CondTest1());
  EXPECT_TRUE(berberis::x86_64::CondTest2());
  EXPECT_TRUE(berberis::x86_64::JccTest());
  EXPECT_TRUE(berberis::x86_64::ReadWriteTest());
  EXPECT_TRUE(berberis::x86_64::CallFPTest());
  EXPECT_TRUE(berberis::x86_64::XmmTest());
  EXPECT_TRUE(berberis::x86_64::XmmMemTest());
  EXPECT_TRUE(berberis::x86_64::MovsxblRexTest());
  EXPECT_TRUE(berberis::x86_64::MovzxblRexTest());
  EXPECT_TRUE(berberis::x86_64::ShldlRexTest());
  EXPECT_TRUE(berberis::x86_64::ShrdlRexTest());
  EXPECT_TRUE(berberis::x86_64::ReadGlobalTest());
  EXPECT_TRUE(berberis::x86_64::MemShiftTest());
#endif
  // Currently we don't support these tests for riscv.
  // TODO(b/352784623): Implement for riscv.
#if defined(__i386__) || defined(__x86_64__)
  EXPECT_TRUE(berberis::ExhaustiveTest());
  EXPECT_TRUE(berberis::MixedAssembler());
#endif
}
