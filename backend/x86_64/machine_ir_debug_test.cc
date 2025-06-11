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

#include "berberis/backend/code_emitter.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/base/arena_alloc.h"

namespace berberis {

namespace {

constexpr auto kMachineRegFLAGS = x86_64::MachineRegs::kFLAGS;

constexpr auto kMachineRegRAX = x86_64::MachineRegs::kRAX;
constexpr auto kMachineRegRCX = x86_64::MachineRegs::kRCX;
constexpr auto kMachineRegRDX = x86_64::MachineRegs::kRDX;
constexpr auto kMachineRegRBX = x86_64::MachineRegs::kRBX;
constexpr auto kMachineRegRSI = x86_64::MachineRegs::kRSI;
constexpr auto kMachineRegRDI = x86_64::MachineRegs::kRDI;
constexpr auto kMachineRegR8 = x86_64::MachineRegs::kR8;

constexpr auto kMachineRegXMM0 = x86_64::MachineRegs::kXMM0;
constexpr auto kMachineRegXMM1 = x86_64::MachineRegs::kXMM1;

class TestCodeEmitter : public CodeEmitter {
 public:
  TestCodeEmitter(MachineCode* mc,
                  uint32_t frame_size,
                  size_t max_ids,
                  Arena* arena,
                  Condition cond,
                  Register reg1,
                  XMMRegister xmm0,
                  Operand op0,
                  int64_t imm,
                  Register reg2,
                  XMMRegister xmm1,
                  Operand op1)
      : CodeEmitter(mc, frame_size, max_ids, arena),
        cond_(cond),
        reg1_(reg1),
        reg2_(reg2),
        xmm0_(xmm0),
        xmm1_(xmm1),
        op0_(op0),
        op1_(op1),
        imm_(imm) {}

  void TestInsn(Condition cond,
                Register reg1,
                XMMRegister xmm0,
                Operand op0,
                int64_t imm,
                Register reg2,
                XMMRegister xmm1,
                Operand op1) {
    EXPECT_EQ(cond, cond_);
    EXPECT_EQ(reg1, reg1_);
    EXPECT_EQ(reg2, reg2_);
    EXPECT_EQ(xmm0, xmm0_);
    EXPECT_EQ(xmm1, xmm1_);
    EXPECT_EQ(op0, op0_);
    EXPECT_EQ(op1, op1_);
    EXPECT_EQ(imm, imm_);
  }

 private:
  Condition cond_;
  Register reg1_;
  Register reg2_;
  XMMRegister xmm0_;
  XMMRegister xmm1_;
  Operand op0_;
  Operand op1_;
  int64_t imm_;
};

template <typename MacroAssemblers>
class TestInsn {
 public:
  using DeviceInsnInfo = device_arch_info::DeviceInsnInfo<
      [](CodeEmitter& code_emitter,
         CodeEmitter::Condition cond,
         CodeEmitter::Register reg1,
         CodeEmitter::XMMRegister xmm0,
         CodeEmitter::Operand op0,
         int64_t imm,
         CodeEmitter::Register reg2,
         CodeEmitter::XMMRegister xmm1,
         CodeEmitter::Operand op1) {
        static_cast<TestCodeEmitter&>(code_emitter)
            .TestInsn(cond, reg1, xmm0, op0, imm, reg2, xmm1, op1);
      },
      "TEST_INSN",
      true,
      []<typename Opcode> { return static_cast<Opcode>(0x00ff'ffff); },
      device_arch_info::NoCPUIDRestriction,
      std::tuple<
          device_arch_info::OperandInfo<x86_64::device_arch_info::RBX, device_arch_info::kUseDef>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::Cond, device_arch_info::kUse>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::GeneralReg64,
                                        device_arch_info::kDef>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::FpReg64, device_arch_info::kDef>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::Mem64, device_arch_info::kDef>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::Imm64, device_arch_info::kUse>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::GeneralReg32,
                                        device_arch_info::kUse>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::FpReg32, device_arch_info::kUse>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::Mem32, device_arch_info::kUse>,
          device_arch_info::OperandInfo<x86_64::device_arch_info::FLAGS, device_arch_info::kDef>>>;
};

class Verifier {
 public:
  Verifier() = default;

  void Init(x86_64::MachineIR& machine_ir,
            const char* debug_string,
            CodeEmitter::Condition cond,
            CodeEmitter::Register reg1,
            CodeEmitter::XMMRegister xmm0,
            CodeEmitter::Operand op0,
            int64_t imm,
            CodeEmitter::Register reg2,
            CodeEmitter::XMMRegister xmm1,
            CodeEmitter::Operand op1) {
    MachineCode machine_code;
    TestCodeEmitter as{&machine_code,
                       machine_ir.FrameSize(),
                       machine_ir.bb_list().size(),
                       machine_ir.arena(),
                       cond,
                       reg1,
                       xmm0,
                       op0,
                       imm,
                       reg2,
                       xmm1,
                       op1};
    EXPECT_EQ(machine_ir.GetDebugString(), debug_string);
    machine_ir.Emit(&as);
  }
};

TEST(MachineIRDebugTest, AbsoluteAbsoluteOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kOverflow,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), O, rcx, xmm0, [0x1], 0x4242424242424242, rbx, xmm1, [0x2], (eflags)
)",
                CodeEmitter::Condition::kOverflow,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.disp = 2});
}

TEST(MachineIRDebugTest, AbsoluteBaseOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kNoOverflow,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.base = kMachineRegR8, .disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), NO, rcx, xmm0, [0x1], 0x4242424242424242, rbx, xmm1, [r8 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kNoOverflow,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.base = CodeEmitter::r8, .disp = 2});
}

TEST(MachineIRDebugTest, AbsoluteBaseIndexOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kBelow,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.base = kMachineRegR8,
                                              .index = kMachineRegRDI,
                                              .scale = CodeEmitter::kTimesFour,
                                              .disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), B, rcx, xmm0, [0x1], 0x4242424242424242, rbx, xmm1, [r8 + rdi * 4 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kBelow,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.base = CodeEmitter::r8,
                 .index = CodeEmitter::rdi,
                 .scale = CodeEmitter::kTimesFour,
                 .disp = 2});
}

TEST(MachineIRDebugTest, AbsoluteIndexOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(
      kMachineRegRAX,
      CodeEmitter::Condition::kAboveEqual,
      kMachineRegRCX,
      kMachineRegXMM0,
      x86_64::MemoryOperand{.disp = 1},
      0x4242'4242'4242'4242,
      kMachineRegRBX,
      kMachineRegXMM1,
      x86_64::MemoryOperand{.index = kMachineRegRDI, .scale = CodeEmitter::kTimesFour, .disp = 2},
      kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), AE, rcx, xmm0, [0x1], 0x4242424242424242, rbx, xmm1, [rdi * 4 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kAboveEqual,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.index = CodeEmitter::rdi, .scale = CodeEmitter::kTimesFour, .disp = 2});
}

TEST(MachineIRDebugTest, BaseAbsoluteOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kEqual,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.base = kMachineRegRDX, .disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), Z, rcx, xmm0, [rdx + 0x1], 0x4242424242424242, rbx, xmm1, [0x2], (eflags)
)",
                CodeEmitter::Condition::kEqual,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.base = CodeEmitter::rdx, .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.disp = 2});
}

TEST(MachineIRDebugTest, BaseBaseOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kNotEqual,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.base = kMachineRegRDX, .disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.base = kMachineRegR8, .disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), NZ, rcx, xmm0, [rdx + 0x1], 0x4242424242424242, rbx, xmm1, [r8 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kNotEqual,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.base = CodeEmitter::rdx, .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.base = CodeEmitter::r8, .disp = 2});
}

TEST(MachineIRDebugTest, BaseBaseIndexOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kBelowEqual,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.base = kMachineRegRDX, .disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.base = kMachineRegR8,
                                              .index = kMachineRegRDI,
                                              .scale = CodeEmitter::kTimesFour,
                                              .disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), BE, rcx, xmm0, [rdx + 0x1], 0x4242424242424242, rbx, xmm1, [r8 + rdi * 4 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kBelowEqual,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.base = CodeEmitter::rdx, .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.base = CodeEmitter::r8,
                 .index = CodeEmitter::rdi,
                 .scale = CodeEmitter::kTimesFour,
                 .disp = 2});
}

TEST(MachineIRDebugTest, BaseIndexOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(
      kMachineRegRAX,
      CodeEmitter::Condition::kAbove,
      kMachineRegRCX,
      kMachineRegXMM0,
      x86_64::MemoryOperand{.base = kMachineRegRDX, .disp = 1},
      0x4242'4242'4242'4242,
      kMachineRegRBX,
      kMachineRegXMM1,
      x86_64::MemoryOperand{.index = kMachineRegRDI, .scale = CodeEmitter::kTimesFour, .disp = 2},
      kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), A, rcx, xmm0, [rdx + 0x1], 0x4242424242424242, rbx, xmm1, [rdi * 4 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kAbove,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.base = CodeEmitter::rdx, .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.index = CodeEmitter::rdi, .scale = CodeEmitter::kTimesFour, .disp = 2});
}

TEST(MachineIRDebugTest, IndexAbsoluteOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(
      kMachineRegRAX,
      CodeEmitter::Condition::kNegative,
      kMachineRegRCX,
      kMachineRegXMM0,
      x86_64::MemoryOperand{.index = kMachineRegRSI, .scale = CodeEmitter::kTimesTwo, .disp = 1},
      0x4242'4242'4242'4242,
      kMachineRegRBX,
      kMachineRegXMM1,
      x86_64::MemoryOperand{.disp = 2},
      kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), N, rcx, xmm0, [rsi * 2 + 0x1], 0x4242424242424242, rbx, xmm1, [0x2], (eflags)
)",
                CodeEmitter::Condition::kNegative,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.index = CodeEmitter::rsi, .scale = CodeEmitter::kTimesTwo, .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.disp = 2});
}

TEST(MachineIRDebugTest, IndexBaseOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(
      kMachineRegRAX,
      CodeEmitter::Condition::kPositiveOrZero,
      kMachineRegRCX,
      kMachineRegXMM0,
      x86_64::MemoryOperand{.index = kMachineRegRSI, .scale = CodeEmitter::kTimesTwo, .disp = 1},
      0x4242'4242'4242'4242,
      kMachineRegRBX,
      kMachineRegXMM1,
      x86_64::MemoryOperand{.disp = 2},
      kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), PL, rcx, xmm0, [rsi * 2 + 0x1], 0x4242424242424242, rbx, xmm1, [0x2], (eflags)
)",
                CodeEmitter::Condition::kPositiveOrZero,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.index = CodeEmitter::rsi, .scale = CodeEmitter::kTimesTwo, .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.disp = 2});
}

TEST(MachineIRDebugTest, IndexBaseIndexOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(
      kMachineRegRAX,
      CodeEmitter::Condition::kParityEven,
      kMachineRegRCX,
      kMachineRegXMM0,
      x86_64::MemoryOperand{.index = kMachineRegRSI, .scale = CodeEmitter::kTimesTwo, .disp = 1},
      0x4242'4242'4242'4242,
      kMachineRegRBX,
      kMachineRegXMM1,
      x86_64::MemoryOperand{.base = kMachineRegR8,
                            .index = kMachineRegRDI,
                            .scale = CodeEmitter::kTimesFour,
                            .disp = 2},
      kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), PE, rcx, xmm0, [rsi * 2 + 0x1], 0x4242424242424242, rbx, xmm1, [r8 + rdi * 4 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kParityEven,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.index = CodeEmitter::rsi, .scale = CodeEmitter::kTimesTwo, .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.base = CodeEmitter::r8,
                 .index = CodeEmitter::rdi,
                 .scale = CodeEmitter::kTimesFour,
                 .disp = 2});
}

TEST(MachineIRDebugTest, IndexIndexOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(
      kMachineRegRAX,
      CodeEmitter::Condition::kParityOdd,
      kMachineRegRCX,
      kMachineRegXMM0,
      x86_64::MemoryOperand{.index = kMachineRegRSI, .scale = CodeEmitter::kTimesTwo, .disp = 1},
      0x4242'4242'4242'4242,
      kMachineRegRBX,
      kMachineRegXMM1,
      x86_64::MemoryOperand{.index = kMachineRegRDI, .scale = CodeEmitter::kTimesFour, .disp = 2},
      kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), PO, rcx, xmm0, [rsi * 2 + 0x1], 0x4242424242424242, rbx, xmm1, [rdi * 4 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kParityOdd,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.index = CodeEmitter::rsi, .scale = CodeEmitter::kTimesTwo, .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.index = CodeEmitter::rdi, .scale = CodeEmitter::kTimesFour, .disp = 2});
}

TEST(MachineIRDebugTest, BaseIndexAbsoluteOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kLess,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.base = kMachineRegRDX,
                                              .index = kMachineRegRSI,
                                              .scale = CodeEmitter::kTimesTwo,
                                              .disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), LS, rcx, xmm0, [rdx + rsi * 2 + 0x1], 0x4242424242424242, rbx, xmm1, [0x2], (eflags)
)",
                CodeEmitter::Condition::kLess,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.base = CodeEmitter::rdx,
                 .index = CodeEmitter::rsi,
                 .scale = CodeEmitter::kTimesTwo,
                 .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.disp = 2});
}

TEST(MachineIRDebugTest, BaseIndexBaseOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kGreaterEqual,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.base = kMachineRegRDX,
                                              .index = kMachineRegRSI,
                                              .scale = CodeEmitter::kTimesTwo,
                                              .disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.base = kMachineRegR8, .disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), GE, rcx, xmm0, [rdx + rsi * 2 + 0x1], 0x4242424242424242, rbx, xmm1, [r8 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kGreaterEqual,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.base = CodeEmitter::rdx,
                 .index = CodeEmitter::rsi,
                 .scale = CodeEmitter::kTimesTwo,
                 .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.base = CodeEmitter::r8, .disp = 2});
}

TEST(MachineIRDebugTest, BaseIndexBaseIndexOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(kMachineRegRAX,
                        CodeEmitter::Condition::kLessEqual,
                        kMachineRegRCX,
                        kMachineRegXMM0,
                        x86_64::MemoryOperand{.base = kMachineRegRDX,
                                              .index = kMachineRegRSI,
                                              .scale = CodeEmitter::kTimesTwo,
                                              .disp = 1},
                        0x4242'4242'4242'4242,
                        kMachineRegRBX,
                        kMachineRegXMM1,
                        x86_64::MemoryOperand{.base = kMachineRegR8,
                                              .index = kMachineRegRDI,
                                              .scale = CodeEmitter::kTimesFour,
                                              .disp = 2},
                        kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), LE, rcx, xmm0, [rdx + rsi * 2 + 0x1], 0x4242424242424242, rbx, xmm1, [r8 + rdi * 4 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kLessEqual,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.base = CodeEmitter::rdx,
                 .index = CodeEmitter::rsi,
                 .scale = CodeEmitter::kTimesTwo,
                 .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.base = CodeEmitter::r8,
                 .index = CodeEmitter::rdi,
                 .scale = CodeEmitter::kTimesFour,
                 .disp = 2});
}

TEST(MachineIRDebugTest, BaseIndexIndexOperands) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  builder.Gen<TestInsn>(
      kMachineRegRAX,
      CodeEmitter::Condition::kGreater,
      kMachineRegRCX,
      kMachineRegXMM0,
      x86_64::MemoryOperand{.base = kMachineRegRDX,
                            .index = kMachineRegRSI,
                            .scale = CodeEmitter::kTimesTwo,
                            .disp = 1},
      0x4242'4242'4242'4242,
      kMachineRegRBX,
      kMachineRegXMM1,
      x86_64::MemoryOperand{.index = kMachineRegRDI, .scale = CodeEmitter::kTimesFour, .disp = 2},
      kMachineRegFLAGS);
  Verifier verifier;
  verifier.Init(machine_ir,
                R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (rax), GT, rcx, xmm0, [rdx + rsi * 2 + 0x1], 0x4242424242424242, rbx, xmm1, [rdi * 4 + 0x2], (eflags)
)",
                CodeEmitter::Condition::kGreater,
                CodeEmitter::rcx,
                CodeEmitter::xmm0,
                {.base = CodeEmitter::rdx,
                 .index = CodeEmitter::rsi,
                 .scale = CodeEmitter::kTimesTwo,
                 .disp = 1},
                0x4242'4242'4242'4242,
                CodeEmitter::rbx,
                CodeEmitter::xmm1,
                {.index = CodeEmitter::rdi, .scale = CodeEmitter::kTimesFour, .disp = 2});
}

TEST(MachineIRDebugTest, VirtualRegisters) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);

  x86_64::MachineIRBuilder builder(&machine_ir);
  builder.StartBasicBlock(machine_ir.NewBasicBlock());

  MachineReg reg0 = machine_ir.AllocVReg();
  MachineReg reg1 = machine_ir.AllocVReg();
  MachineReg reg2 = machine_ir.AllocVReg();
  MachineReg xmm0 = machine_ir.AllocVReg();
  MachineReg xmm1 = machine_ir.AllocVReg();
  MachineReg base0 = machine_ir.AllocVReg();
  MachineReg index0 = machine_ir.AllocVReg();
  MachineReg base1 = machine_ir.AllocVReg();
  MachineReg index1 = machine_ir.AllocVReg();
  MachineReg flags = machine_ir.AllocVReg();

  builder.Gen<TestInsn>(
      reg0,
      CodeEmitter::Condition::kEqual,
      reg1,
      xmm0,
      x86_64::MemoryOperand{
          .base = base0, .index = index0, .scale = CodeEmitter::kTimesOne, .disp = 1},
      0x4242'4242'4242'4242,
      reg2,
      xmm1,
      x86_64::MemoryOperand{.base = base1, .index = index1, CodeEmitter::kTimesEight, .disp = 2},
      flags);
  EXPECT_EQ(machine_ir.GetDebugString(), R"( 0 MachineBasicBlock live_in=[] live_out=[]
    TEST_INSN (RBX v0), Z, GeneralReg64 v1, FpReg64 v3, [GeneralReg64 v5 + GeneralReg64 v6 * 1 + 0x1], 0x4242424242424242, GeneralReg32 v2, FpReg32 v4, [GeneralReg64 v7 + GeneralReg64 v8 * 8 + 0x2], (FLAGS v9)
)");
}

}  // namespace

}  // namespace berberis
