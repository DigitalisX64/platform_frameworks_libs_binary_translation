#include "berberis/backend/x86_64/use_vex_three_operand_forms.h"

#include <cstdint>
#include <iterator>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/checks.h"

namespace berberis::x86_64 {

namespace {

// The two-address SSE operations with a three-operand AVX form taking the
// same operands in the same order.
#define DRION_VEX_FORMS(REG_FORM, IMM_FORM) \
  REG_FORM(Addpd, Vaddpd) \
  REG_FORM(Addps, Vaddps) \
  REG_FORM(Haddpd, Vhaddpd) \
  REG_FORM(Haddps, Vhaddps) \
  REG_FORM(Addsd, Vaddsd) \
  REG_FORM(Addss, Vaddss) \
  REG_FORM(Cmpeqpd, Vcmpeqpd) \
  REG_FORM(Cmpeqps, Vcmpeqps) \
  REG_FORM(Cmplepd, Vcmplepd) \
  REG_FORM(Cmpleps, Vcmpleps) \
  REG_FORM(Cmpltpd, Vcmpltpd) \
  REG_FORM(Cmpltps, Vcmpltps) \
  REG_FORM(Cmpunordpd, Vcmpunordpd) \
  REG_FORM(Cmpunordps, Vcmpunordps) \
  REG_FORM(Divpd, Vdivpd) \
  REG_FORM(Divps, Vdivps) \
  REG_FORM(Divsd, Vdivsd) \
  REG_FORM(Divss, Vdivss) \
  REG_FORM(Subpd, Vsubpd) \
  REG_FORM(Subps, Vsubps) \
  REG_FORM(Subsd, Vsubsd) \
  REG_FORM(Subss, Vsubss) \
  REG_FORM(Maxpd, Vmaxpd) \
  REG_FORM(Maxps, Vmaxps) \
  REG_FORM(Minpd, Vminpd) \
  REG_FORM(Minps, Vminps) \
  REG_FORM(Mulpd, Vmulpd) \
  REG_FORM(Mulps, Vmulps) \
  REG_FORM(Mulsd, Vmulsd) \
  REG_FORM(Mulss, Vmulss) \
  REG_FORM(Packssdw, Vpackssdw) \
  REG_FORM(Packsswb, Vpacksswb) \
  REG_FORM(Packusdw, Vpackusdw) \
  REG_FORM(Packuswb, Vpackuswb) \
  REG_FORM(Paddb, Vpaddb) \
  REG_FORM(Paddd, Vpaddd) \
  REG_FORM(Paddq, Vpaddq) \
  REG_FORM(Paddsb, Vpaddsb) \
  REG_FORM(Paddsw, Vpaddsw) \
  REG_FORM(Paddusb, Vpaddusb) \
  REG_FORM(Paddusw, Vpaddusw) \
  REG_FORM(Paddw, Vpaddw) \
  REG_FORM(Pand, Vpand) \
  REG_FORM(Pandn, Vpandn) \
  REG_FORM(Pcmpeqb, Vpcmpeqb) \
  REG_FORM(Pcmpeqd, Vpcmpeqd) \
  REG_FORM(Pcmpeqq, Vpcmpeqq) \
  REG_FORM(Pcmpeqw, Vpcmpeqw) \
  REG_FORM(Pcmpgtb, Vpcmpgtb) \
  REG_FORM(Pcmpgtd, Vpcmpgtd) \
  REG_FORM(Pcmpgtq, Vpcmpgtq) \
  REG_FORM(Pcmpgtw, Vpcmpgtw) \
  REG_FORM(Phaddd, Vphaddd) \
  REG_FORM(Phaddw, Vphaddw) \
  REG_FORM(Pmaddwd, Vpmaddwd) \
  REG_FORM(Pmaxsb, Vpmaxsb) \
  REG_FORM(Pmaxsd, Vpmaxsd) \
  REG_FORM(Pmaxsw, Vpmaxsw) \
  REG_FORM(Pmaxub, Vpmaxub) \
  REG_FORM(Pmaxud, Vpmaxud) \
  REG_FORM(Pmaxuw, Vpmaxuw) \
  REG_FORM(Pminsb, Vpminsb) \
  REG_FORM(Pminsd, Vpminsd) \
  REG_FORM(Pminsw, Vpminsw) \
  REG_FORM(Pminub, Vpminub) \
  REG_FORM(Pminud, Vpminud) \
  REG_FORM(Pminuw, Vpminuw) \
  REG_FORM(Pmuldq, Vpmuldq) \
  REG_FORM(Pmulhrsw, Vpmulhrsw) \
  REG_FORM(Pmulhw, Vpmulhw) \
  REG_FORM(Pmulld, Vpmulld) \
  REG_FORM(Pmullw, Vpmullw) \
  REG_FORM(Pmuludq, Vpmuludq) \
  REG_FORM(Por, Vpor) \
  REG_FORM(Psadbw, Vpsadbw) \
  REG_FORM(Pshufb, Vpshufb) \
  REG_FORM(Psubb, Vpsubb) \
  REG_FORM(Psubd, Vpsubd) \
  REG_FORM(Psubq, Vpsubq) \
  REG_FORM(Psubsb, Vpsubsb) \
  REG_FORM(Psubsw, Vpsubsw) \
  REG_FORM(Psubusb, Vpsubusb) \
  REG_FORM(Psubusw, Vpsubusw) \
  REG_FORM(Psubw, Vpsubw) \
  REG_FORM(Punpckhbw, Vpunpckhbw) \
  REG_FORM(Punpckhdq, Vpunpckhdq) \
  REG_FORM(Punpckhqdq, Vpunpckhqdq) \
  REG_FORM(Punpckhwd, Vpunpckhwd) \
  REG_FORM(Punpcklbw, Vpunpcklbw) \
  REG_FORM(Punpckldq, Vpunpckldq) \
  REG_FORM(Punpcklqdq, Vpunpcklqdq) \
  REG_FORM(Punpcklwd, Vpunpcklwd) \
  REG_FORM(Pxor, Vpxor) \
  REG_FORM(Xorpd, Vxorpd) \
  IMM_FORM(Pslld, Vpslld) \
  IMM_FORM(Pslldq, Vpslldq) \
  IMM_FORM(Psllq, Vpsllq) \
  IMM_FORM(Psllw, Vpsllw) \
  IMM_FORM(Psrad, Vpsrad) \
  IMM_FORM(Psraw, Vpsraw) \
  IMM_FORM(Psrld, Vpsrld) \
  IMM_FORM(Psrldq, Vpsrldq) \
  IMM_FORM(Psrlq, Vpsrlq) \
  IMM_FORM(Psrlw, Vpsrlw)

bool HasVexForm(MachineOpcode opcode) {
  switch (opcode) {
#define DRION_VEX_REG_CASE(sse, vex) case kMachineOp##sse##XRegXReg:
#define DRION_VEX_IMM_CASE(sse, vex) case kMachineOp##sse##XRegImm:
    DRION_VEX_FORMS(DRION_VEX_REG_CASE, DRION_VEX_IMM_CASE)
#undef DRION_VEX_REG_CASE
#undef DRION_VEX_IMM_CASE
    return true;
    default:
      return false;
  }
}

bool References(const berberis::MachineInsn* insn, MachineReg reg) {
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegAt(i) == reg) {
      return true;
    }
  }
  return false;
}

bool Defines(const berberis::MachineInsn* insn, MachineReg reg) {
  for (int i = 0; i < insn->NumRegOperands(); i++) {
    if (insn->RegAt(i) == reg && insn->RegKindAt(i).IsDef()) {
      return true;
    }
  }
  return false;
}

// Returns the AVX form of the two-address SSE operation op, reading src in
// place of op's destination. Each AVX form takes
// its operands in the SSE order, with src as the first source, so the result
// is the same, including the lanes a scalar operation copies through.
berberis::MachineInsn* NewVexForm(MachineIR* machine_ir,
                                  const berberis::MachineInsn* op,
                                  MachineReg src) {
  MachineReg dst = op->RegAt(0);
  // An operation reading its own destination as its second operand reads src.
  auto second = [&]() { return op->RegAt(1) == dst ? src : op->RegAt(1); };
  auto imm = [&]() { return static_cast<int8_t>(AsMachineInsnX86_64(op)->imm()); };
  switch (op->opcode()) {
#define DRION_VEX_REG_FORM(sse, vex) \
  case kMachineOp##sse##XRegXReg:    \
    return machine_ir->NewInsn<vex##XRegXRegXReg>(dst, src, second());
#define DRION_VEX_IMM_FORM(sse, vex) \
  case kMachineOp##sse##XRegImm:     \
    return machine_ir->NewInsn<vex##XRegXRegImm>(dst, src, imm());
    DRION_VEX_FORMS(DRION_VEX_REG_FORM, DRION_VEX_IMM_FORM)
#undef DRION_VEX_REG_FORM
#undef DRION_VEX_IMM_FORM
    default:
      FATAL("no AVX form for opcode %d", op->opcode());
  }
}

// Merges the copy that starts op's destination into op. Returns true if it did.
bool TryMerge(MachineIR* machine_ir, MachineBasicBlock* bb, MachineInsnList::iterator op_it) {
  berberis::MachineInsn* op = *op_it;
  if (!HasVexForm(op->opcode())) {
    return false;
  }
  MachineReg dst = op->RegAt(0);
  if (!dst.IsVReg()) {
    return false;
  }
  // The latest earlier instruction that touches dst must be the copy into it,
  // and the copied register must keep its value up to op.
  auto it = op_it;
  while (it != bb->insn_list().begin()) {
    --it;
    if (!References(*it, dst)) {
      continue;
    }
    berberis::MachineInsn* copy = *it;
    if (copy->opcode() != kMachineOpPseudoCopy || copy->RegAt(0) != dst) {
      return false;
    }
    MachineReg src = copy->RegAt(1);
    if (!src.IsVReg() || src == dst) {
      return false;
    }
    for (auto between = std::next(it); between != op_it; ++between) {
      if (Defines(*between, src)) {
        return false;
      }
    }
    bb->insn_list().erase(it);
    *op_it = NewVexForm(machine_ir, op, src);
    return true;
  }
  return false;
}

}  // namespace

void UseVexThreeOperandForms(MachineIR* machine_ir) {
  for (auto* bb : machine_ir->bb_list()) {
    for (auto op_it = bb->insn_list().begin(); op_it != bb->insn_list().end(); ++op_it) {
      TryMerge(machine_ir, bb, op_it);
    }
  }
}

}  // namespace berberis::x86_64
