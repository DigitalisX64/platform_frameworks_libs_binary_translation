/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include "berberis/assembler/x86_64.h"

#include "berberis/assembler/machine_code.h"
#include "berberis/base/bit_util.h"
#include "berberis/code_gen_lib/gen_adaptor.h"
#include "berberis/code_gen_lib/gen_wrapper.h"
#include "berberis/guest_abi/guest_arguments.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_exec_region.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

// Constant for NaN boxing and unboxing of 32-bit floats.
constexpr uint64_t kNanBoxFloat32 = 0xffff'ffff'0000'0000ULL;

bool g_called;
uint32_t g_arg;
ThreadState g_state{};
uint32_t g_insn;
uint32_t g_ret_insn;

void DummyTrampoline(void* arg, ThreadState* state) {
  g_called = true;
  ASSERT_EQ(&g_arg, arg);
  ASSERT_EQ(&g_state, state);
  ASSERT_EQ(g_state.cpu.insn_addr, ToGuestAddr(&g_insn));
  ASSERT_EQ(GetLinkRegister(g_state.cpu), ToGuestAddr(&g_ret_insn));
}

TEST(CodeGenLib, GenTrampolineAdaptor) {
  MachineCode machine_code;

  GenTrampolineAdaptor(
      &machine_code, ToGuestAddr(&g_insn), AsHostCode(DummyTrampoline), &g_arg, "DummyTrampoline");

  ScopedExecRegion exec(&machine_code);

  g_called = false;
  g_state.cpu.insn_addr = 0;
  SetLinkRegister(g_state.cpu, ToGuestAddr(&g_ret_insn));

  TestingRunGeneratedCode(&g_state, exec.get(), ToGuestAddr(&g_ret_insn));

  ASSERT_TRUE(g_called);
  ASSERT_EQ(g_state.cpu.insn_addr, ToGuestAddr(&g_ret_insn));
}

void GenMoveResidenceToReg(MachineCode* machine_code) {
  x86_64::Assembler as(machine_code);
  // Perform x0 = ThreadState::residence.
  as.Movq(as.rdx, {.base = as.rbp, .disp = offsetof(ThreadState, residence)});
  as.Movq({.base = as.rbp, .disp = offsetof(ThreadState, cpu.x[0])}, as.rdx);
  as.Jmp(kEntryExitGeneratedCode);
}

uint64_t GetResidenceReg(const ThreadState& state) {
  return state.cpu.x[0];
}

void CheckResidenceTrampoline(void*, ThreadState* state) {
  EXPECT_EQ(state->residence, kOutsideGeneratedCode);
}

void AddToTranslationCache(GuestAddr guest_addr, HostCodePiece host_code_piece) {
  auto* tc = TranslationCache::GetInstance();
  GuestCodeEntry* entry = tc->AddAndLockForTranslation(guest_addr, 0);
  ASSERT_NE(entry, nullptr);
  tc->SetTranslatedAndUnlock(
      guest_addr, entry, 1, GuestCodeEntry::Kind::kSpecialHandler, host_code_piece);
}

TEST(CodeGenLib, GenTrampolineAdaptorResidence) {
  MachineCode trampoline_adaptor;
  GenTrampolineAdaptor(&trampoline_adaptor,
                       ToGuestAddr(&g_insn),
                       AsHostCode(CheckResidenceTrampoline),
                       nullptr,
                       nullptr);
  ScopedExecRegion trampoline_exec(&trampoline_adaptor);

  // Trampoline returns to generated code, so we generate some.
  MachineCode generated_code;
  GenMoveResidenceToReg(&generated_code);
  ScopedExecRegion generated_code_exec(&generated_code);

  AddToTranslationCache(ToGuestAddr(&g_ret_insn),
                        {generated_code_exec.GetHostCodeAddr(), generated_code.install_size()});

  g_state.cpu.insn_addr = 0;
  SetLinkRegister(g_state.cpu, ToGuestAddr(&g_ret_insn));
  EXPECT_EQ(g_state.residence, kOutsideGeneratedCode);

  berberis_RunGeneratedCode(&g_state, trampoline_exec.get());

  EXPECT_EQ(g_state.residence, kOutsideGeneratedCode);
  EXPECT_EQ(g_state.cpu.insn_addr, ToGuestAddr(&g_ret_insn));
  EXPECT_EQ(GetResidenceReg(g_state), kInsideGeneratedCode);

  TranslationCache::GetInstance()->InvalidateGuestRange(ToGuestAddr(&g_ret_insn),
                                                        ToGuestAddr(&g_ret_insn) + 1);
}

void DummyRunner2(GuestAddr pc, GuestArgumentBuffer* buf) {
  g_called = true;
  ASSERT_EQ(pc, ToGuestAddr(&g_insn));
  ASSERT_NE(nullptr, buf);
  ASSERT_EQ(1, buf->argc);
  ASSERT_EQ(0, buf->resc);
  ASSERT_EQ(1234u, buf->argv[0]);
}

TEST(CodeGenLib, GenWrapGuestFunction) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "vi", AsHostCode(DummyRunner2), "DummyRunner2");

  ScopedExecRegion exec(&machine_code);

  g_called = false;
  exec.get<void(int)>()(1234);

  ASSERT_TRUE(g_called);
}

void Run10UInt8(GuestAddr pc, GuestArgumentBuffer* buf) {
  ASSERT_EQ(ToGuestAddr(&g_insn), pc);
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(buf->argc, 8);
  ASSERT_EQ(buf->stack_argc, 16);
  ASSERT_EQ(buf->resc, 1);
  ASSERT_EQ(buf->argv[0], 0U);
  ASSERT_EQ(buf->argv[1], 0xffU);
  ASSERT_EQ(buf->argv[2], 2U);
  ASSERT_EQ(buf->argv[3], 3U);
  ASSERT_EQ(buf->argv[4], 4U);
  ASSERT_EQ(buf->argv[5], 5U);
  ASSERT_EQ(buf->argv[6], 6U);
  ASSERT_EQ(buf->argv[7], 0xf9U);
  ASSERT_EQ(buf->stack_argv[0], 0xf8U);
  ASSERT_EQ(buf->stack_argv[1], 9U);
  buf->argv[0] = 0xf6;
}

TEST(CodeGenLib, GenWrapGuestFunction_Run10UInt8) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "zzzzzzzzzzz", AsHostCode(Run10UInt8), "Run10UInt8");

  ScopedExecRegion exec(&machine_code);

  using Func = uint8_t(
      uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t);
  uint8_t res = exec.get<Func>()(0, 0xff, 2, 3, 4, 5, 6, 0xf9, 0xf8, 9);
  ASSERT_EQ(res, 0xf6U);
}

void Run10Int8(GuestAddr pc, GuestArgumentBuffer* buf) {
  ASSERT_EQ(ToGuestAddr(&g_insn), pc);
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(buf->argc, 8);
  ASSERT_EQ(buf->stack_argc, 16);
  ASSERT_EQ(buf->resc, 1);
  ASSERT_EQ(buf->argv[0], 0U);
  ASSERT_EQ(buf->argv[1], 0xffff'ffff'ffff'ffffULL);
  ASSERT_EQ(buf->argv[2], 2U);
  ASSERT_EQ(buf->argv[3], 3U);
  ASSERT_EQ(buf->argv[4], 4U);
  ASSERT_EQ(buf->argv[5], 5U);
  ASSERT_EQ(buf->argv[6], 6U);
  ASSERT_EQ(buf->argv[7], 0xffff'ffff'ffff'fff9ULL);
  ASSERT_EQ(buf->stack_argv[0], 0xffff'ffff'ffff'fff8ULL);
  ASSERT_EQ(buf->stack_argv[1], 9U);
  buf->argv[0] = 0xffff'ffff'ffff'fff6;
}

TEST(CodeGenLib, GenWrapGuestFunction_Run10Int8) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "bbbbbbbbbbb", AsHostCode(Run10Int8), "Run10Int8");

  ScopedExecRegion exec(&machine_code);

  using Func =
      int8_t(int8_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t);
  int8_t res = exec.get<Func>()(0, -1, 2, 3, 4, 5, 6, -7, -8, 9);
  ASSERT_EQ(res, -10);
}

void Run10UInt16(GuestAddr pc, GuestArgumentBuffer* buf) {
  ASSERT_EQ(ToGuestAddr(&g_insn), pc);
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(buf->argc, 8);
  ASSERT_EQ(buf->stack_argc, 16);
  ASSERT_EQ(buf->resc, 1);
  ASSERT_EQ(buf->argv[0], 0U);
  ASSERT_EQ(buf->argv[1], 0xffffU);
  ASSERT_EQ(buf->argv[2], 2U);
  ASSERT_EQ(buf->argv[3], 3U);
  ASSERT_EQ(buf->argv[4], 4U);
  ASSERT_EQ(buf->argv[5], 5U);
  ASSERT_EQ(buf->argv[6], 6U);
  ASSERT_EQ(buf->argv[7], 0xfff9U);
  ASSERT_EQ(buf->stack_argv[0], 0xfff8U);
  ASSERT_EQ(buf->stack_argv[1], 9U);
  buf->argv[0] = 0xfff6;
}

TEST(CodeGenLib, GenWrapGuestFunction_Run10UInt16) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "ccccccccccc", AsHostCode(Run10UInt16), "Run10UInt16");

  ScopedExecRegion exec(&machine_code);

  using Func = uint16_t(uint16_t,
                        uint16_t,
                        uint16_t,
                        uint16_t,
                        uint16_t,
                        uint16_t,
                        uint16_t,
                        uint16_t,
                        uint16_t,
                        uint16_t);
  uint16_t res = exec.get<Func>()(0, 0xffff, 2, 3, 4, 5, 6, 0xfff9, 0xfff8, 9);
  ASSERT_EQ(res, 0xfff6U);
}

void Run10Int16(GuestAddr pc, GuestArgumentBuffer* buf) {
  ASSERT_EQ(ToGuestAddr(&g_insn), pc);
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(buf->argc, 8);
  ASSERT_EQ(buf->stack_argc, 16);
  ASSERT_EQ(buf->resc, 1);
  ASSERT_EQ(buf->argv[0], 0U);
  ASSERT_EQ(buf->argv[1], 0xffff'ffff'ffff'ffffULL);
  ASSERT_EQ(buf->argv[2], 2U);
  ASSERT_EQ(buf->argv[3], 3U);
  ASSERT_EQ(buf->argv[4], 4U);
  ASSERT_EQ(buf->argv[5], 5U);
  ASSERT_EQ(buf->argv[6], 6U);
  ASSERT_EQ(buf->argv[7], 0xffff'ffff'ffff'fff9ULL);
  ASSERT_EQ(buf->stack_argv[0], 0xffff'ffff'ffff'fff8ULL);
  ASSERT_EQ(buf->stack_argv[1], 9U);
  buf->argv[0] = 0xffff'ffff'ffff'fff6;
}

TEST(CodeGenLib, GenWrapGuestFunction_Run10Int16) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "sssssssssss", AsHostCode(Run10Int16), "Run10Int16");

  ScopedExecRegion exec(&machine_code);

  using Func = int16_t(
      int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t);
  int16_t res = exec.get<Func>()(0, -1, 2, 3, 4, 5, 6, -7, -8, 9);
  ASSERT_EQ(res, -10);
}

void Run10Int(GuestAddr pc, GuestArgumentBuffer* buf) {
  ASSERT_EQ(ToGuestAddr(&g_insn), pc);
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(buf->argc, 8);
  ASSERT_EQ(buf->stack_argc, 16);
  ASSERT_EQ(buf->resc, 1);
  ASSERT_EQ(buf->argv[0], 0U);
  ASSERT_EQ(buf->argv[1], 0xffff'ffff'ffff'ffffULL);
  ASSERT_EQ(buf->argv[2], 2U);
  ASSERT_EQ(buf->argv[3], 3U);
  ASSERT_EQ(buf->argv[4], 4U);
  ASSERT_EQ(buf->argv[5], 5U);
  ASSERT_EQ(buf->argv[6], 6U);
  ASSERT_EQ(buf->argv[7], 0xffff'ffff'ffff'fff9ULL);
  ASSERT_EQ(buf->stack_argv[0], 0xffff'ffff'ffff'fff8ULL);
  ASSERT_EQ(buf->stack_argv[1], 9U);
  buf->argv[0] = 0xffff'ffff'ffff'fff6;
}

TEST(CodeGenLib, GenWrapGuestFunction_Run10Int) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "iiiiiiiiiii", AsHostCode(Run10Int), "Run10Int");

  ScopedExecRegion exec(&machine_code);

  using Func = int(int, int, int, int, int, int, int, int, int, int);
  int res = exec.get<Func>()(0, -1, 2, 3, 4, 5, 6, -7, -8, 9);
  ASSERT_EQ(res, -10);
}

void Run18Fp(GuestAddr pc, GuestArgumentBuffer* buf) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  ASSERT_EQ(pc, ToGuestAddr(&g_insn));
  ASSERT_NE(nullptr, buf);
  // riscv verification
  ASSERT_EQ(8, buf->argc);
  ASSERT_EQ(8, buf->fp_argc);
  ASSERT_EQ(16, buf->stack_argc);
  ASSERT_EQ(0, buf->resc);
  ASSERT_EQ(1, buf->fp_resc);
  // 32-bit parameters passed in floating-point registers are 1-extended.
  // 32-bit parameters passed in general-purpose registers and on the stack are 0-extended.
  ASSERT_EQ(kNanBoxFloat32, buf->fp_argv[0] & kNanBoxFloat32);
  ASSERT_FLOAT_EQ(0.0f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[0])));
  ASSERT_EQ(kNanBoxFloat32, buf->fp_argv[1] & kNanBoxFloat32);
  ASSERT_FLOAT_EQ(1.1f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[1])));
  ASSERT_EQ(kNanBoxFloat32, buf->fp_argv[2] & kNanBoxFloat32);
  ASSERT_FLOAT_EQ(2.2f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[2])));
  ASSERT_EQ(kNanBoxFloat32, buf->fp_argv[3] & kNanBoxFloat32);
  ASSERT_FLOAT_EQ(3.3f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[3])));
  ASSERT_EQ(kNanBoxFloat32, buf->fp_argv[4] & kNanBoxFloat32);
  ASSERT_FLOAT_EQ(4.4f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[4])));
  ASSERT_EQ(kNanBoxFloat32, buf->fp_argv[5] & kNanBoxFloat32);
  ASSERT_FLOAT_EQ(5.5f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[5])));
  ASSERT_EQ(kNanBoxFloat32, buf->fp_argv[6] & kNanBoxFloat32);
  ASSERT_FLOAT_EQ(6.6f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[6])));
  ASSERT_EQ(kNanBoxFloat32, buf->fp_argv[7] & kNanBoxFloat32);
  ASSERT_FLOAT_EQ(7.7f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[7])));
  ASSERT_FLOAT_EQ(8.8f, bit_cast<float>(static_cast<uint32_t>(buf->argv[0])));
  ASSERT_FLOAT_EQ(9.9f, bit_cast<float>(static_cast<uint32_t>(buf->argv[1])));
  ASSERT_FLOAT_EQ(10.01f, bit_cast<float>(static_cast<uint32_t>(buf->argv[2])));
  ASSERT_FLOAT_EQ(20.02f, bit_cast<float>(static_cast<uint32_t>(buf->argv[3])));
  ASSERT_FLOAT_EQ(30.03f, bit_cast<float>(static_cast<uint32_t>(buf->argv[4])));
  ASSERT_FLOAT_EQ(40.04f, bit_cast<float>(static_cast<uint32_t>(buf->argv[5])));
  ASSERT_FLOAT_EQ(50.05f, bit_cast<float>(static_cast<uint32_t>(buf->argv[6])));
  ASSERT_FLOAT_EQ(60.06f, bit_cast<float>(static_cast<uint32_t>(buf->argv[7])));
  ASSERT_FLOAT_EQ(70.07f, bit_cast<float>(static_cast<uint32_t>(buf->stack_argv[0])));
  ASSERT_FLOAT_EQ(80.08f, bit_cast<float>(static_cast<uint32_t>(buf->stack_argv[1])));
  buf->fp_argv[0] = static_cast<uint64_t>(bit_cast<uint32_t>(45.45f)) | kNanBoxFloat32;
}

TEST(CodeGenLib, GenWrapGuestFunction_Run10Fp) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "fffffffffffffffffff", AsHostCode(Run18Fp), "Run18Fp");

  ScopedExecRegion exec(&machine_code);

  using Func = float(float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float);
  float res = exec.get<Func>()(0.0f,
                               1.1f,
                               2.2f,
                               3.3f,
                               4.4f,
                               5.5f,
                               6.6f,
                               7.7f,
                               8.8f,
                               9.9f,
                               10.01f,
                               20.02f,
                               30.03f,
                               40.04f,
                               50.05f,
                               60.06f,
                               70.07f,
                               80.08f);
  ASSERT_FLOAT_EQ(45.45f, res);
}

}  // namespace

}  // namespace berberis
