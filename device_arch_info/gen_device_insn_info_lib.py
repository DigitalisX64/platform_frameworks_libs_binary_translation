#!/usr/bin/python
#
# Copyright (C) 2025 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate device_insn_info files out of the definition file.

* Operand usage

Register allocator needs operand usage to learn which operands can share the
same register.

To understand register sharing options, register allocator assumes insn works
in these steps:
- read input operands
- do the job
- write output operands

So, input-output operands should have dedicated registers, while input-only
operands can share registers with output-only operands.

There might be an exception when output-only operand is written before all
input-only operands are read, so its register can't be shared. Such operands
are usually referred as output-only-early-clobber operands.

For register sharing, output-only-early-clobber operand is the same as
input-output operand, but it is unnatural to describe output-only as
input-output, so we use a special keyword for it.

Finally, keywords are:
use - input-only
def - output-only
def_early_clobber - output-only-early-clobber
use_def - input-output

* Scratch operands

Scratch operands are actually output operands - indeed, their original value
is not used and they get some new value after the insn is done. However, they
are usually written before all input operands are read, so it makes sense to
describe scratch operands as output-only-early-clobber.
"""

import asm_defs
import json
import sys


def _gen_device_insn_info(f, insns):
  for insn in insns:
    print ("""
template <typename MacroAssemblers>
class %s {
 public:
  using DeviceInsnInfo = device_arch_info::DeviceInsnInfo<%s>;
};""" % (
      insn['name'],
      ',\n                  '.join(
        [_get_asm_reference(insn),
       '"%s"' % insn['mnemo'],
       # Int3, Lfence, Mfence, Sfence, and UD2 have side effects not related to arguments.
       # TODO: decide if we still need it (currently MachineIR treats all instructions without
       # operands as volatile).
       'true' if insn['name'] in ('Int3', 'Lfence', 'Mfence', 'Sfence', 'UD2') else 'false',
       _get_opcode_reference(insn),
       _get_cpuid_restriction(insn),
       _get_reg_operands_info(insn['args'])])),  file=f)


def _get_asm_type(asm, prefix):
  args = filter(
    lambda arg: not asm_defs.is_implicit_reg(arg['class']), asm['args'])
  return ', '.join(_get_asm_operand_type(arg, prefix) for arg in args)


def _get_asm_operand_type(arg, prefix):
  cls = arg.get('class')
  if asm_defs.is_cond(cls):
    return prefix + 'Condition'
  if asm_defs.is_label(cls):
    return prefix + 'Label'
  if asm_defs.is_x87reg(cls):
    return prefix + 'X87Register'
  if asm_defs.is_greg(cls):
    return prefix + 'Register'
  if asm_defs.is_xreg(cls):
    return prefix + 'XMMRegister'
  if asm_defs.is_yreg(cls):
    return prefix + 'YMMRegister'
  if asm_defs.is_mem_op(cls):
    return 'const ' + prefix + 'Operand&'
  if asm_defs.is_imm(cls):
    if cls == 'Imm2':
      return 'int8_t'
    return 'int' + cls[3:] + '_t'
  assert False, f"Unknown asm operand type: {arg}"


def _get_asm_reference(asm):
  # Because of misfeature of Itanium C++ ABI we couldn't just use MacroAssembler
  # to static cast these references if we want to use them as template argument:
  # https://ibob.bg/blog/2018/08/18/a-bug-in-the-cpp-standard/

  # Thankfully there are usually no need to use the same trick for MacroInstructions
  # since we may always rename these, except when immediates are involved.

  # But for assembler we need to use actual type from where these
  # instructions come from!
  #
  # E.g. LZCNT have to be processed like this:
  #   static_cast<void (Assembler_common_x86::*)(
  #     typename Assembler_common_x86::Register,
  #     typename Assembler_common_x86::Register)>(
  #       &Assembler_common_x86::Lzcntl)
  assembler = 'std::tuple_element_t<%s, MacroAssemblers>' % asm['macroassembler']
  return 'static_cast<void (%s::*)(%s)>(%s&%s::%s%s)' % (
      assembler,
      _get_asm_type(asm, 'typename %s::' % assembler),
      '\n                  ',
      assembler,
      'template ' if '<' in asm['asm'] else '',
      asm['asm'])


def _get_cpuid_restriction(asm):
  cpuid_restriction = 'device_arch_info::NoCPUIDRestriction'
  if 'feature' in asm:
    if asm['feature'] == 'AuthenticAMD':
      cpuid_restriction = 'device_arch_info::IsAuthenticAMD'
    else:
      cpuid_restriction = 'device_arch_info::Has%s' % asm['feature']
  return cpuid_restriction


def _get_opcode_reference(asm):
  return f"[]<typename Opcode>{{ return Opcode::kMachineOp{asm['name']}; }}"


def _get_reg_operands_info(args):
  return 'std::tuple<%s>' % ', '.join(_get_reg_operand_info(arg) for arg in args)


def _get_reg_operand_info(arg):
  class_info = 'device_arch_info::%s' % arg['class']
  if arg['class'] in ('Cond', 'Imm2', 'Imm8', 'Imm16', 'Imm32', 'Imm64', 'Label'):
    return 'device_arch_info::OperandInfo<%s, device_arch_info::kUse>' % class_info
  assert 'usage' in arg, f"Unknown asm operand without 'usage'"
  using_info = 'device_arch_info::%s' % {
      'def': 'kDef',
      'def_early_clobber': 'kDefEarlyClobber',
      'use': 'kUse',
      'use_def': 'kUseDef'
  }[arg['usage']]
  return 'device_arch_info::OperandInfo<%s, %s>' % (class_info, using_info)


def _load_lir_def(asm_def, macroassembler):
  _, insns = asm_defs.load_asm_defs(asm_def)
  for insn in insns:
    insn['macroassembler'] = macroassembler
  return insns
