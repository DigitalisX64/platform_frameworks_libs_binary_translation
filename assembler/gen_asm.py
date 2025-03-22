#!/usr/bin/python
#
# Copyright (C) 2014 The Android Open Source Project
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
#

"""Generate assembler files out of the definition file."""

import asm_defs
import os
import re
import sys

from enum import Enum

INDENT = '  '

ROUNDING_MODES = ['FE_TONEAREST', 'FE_DOWNWARD', 'FE_UPWARD', 'FE_TOWARDZERO', 'FE_TIESAWAY']

_imm_types = {
    # x86 immediates
    'Imm2': 'int8_t',
    'Imm8': 'int8_t',
    'Imm16': 'int16_t',
    'Imm32': 'int32_t',
    'Imm64': 'int64_t',
    # Official RISC-V immediates
    'B-Imm': 'BImmediate',
    'I-Imm': 'IImmediate',
    'J-Imm': 'JImmediate',
    'P-Imm': 'PImmediate',
    'S-Imm': 'SImmediate',
    'U-Imm': 'UImmediate',
    # Extra RISC-V immediates
    'Csr-Imm' : 'CsrImmediate',
    'Shift32-Imm': 'Shift32Immediate',
    'Shift64-Imm': 'Shift64Immediate'
}

class AssemblerMode(Enum):
  BINARY_ASSEMBLER = 0
  TEXT_ASSEMBLER = 1
  VERIFIER_ASSEMBLER = 3

def _get_arg_type_name(arg, insn_type):
  cls = arg.get('class')
  if asm_defs.is_x87reg(cls):
    return 'X87Register'
  if asm_defs.is_greg(cls):
    return 'Register'
  if asm_defs.is_freg(cls):
    return 'FpRegister'
  if asm_defs.is_xreg(cls):
    return 'XMMRegister'
  if asm_defs.is_yreg(cls):
    return 'YMMRegister'
  if asm_defs.is_imm(cls):
    return _imm_types[cls]
  if asm_defs.is_disp(cls):
    return 'int32_t'
  if asm_defs.is_label(cls):
    return 'const Label&'
  if asm_defs.is_cond(cls):
    return 'Condition'
  if asm_defs.is_csr(cls):
    return 'Csr'
  if asm_defs.is_rm(cls):
    return 'Rounding'
  if asm_defs.is_mem_op(cls):
    if insn_type is not None and insn_type.endswith('-type'):
      return 'const Operand<Register, %sImmediate>&' % insn_type[:-5]
    return 'const Operand&'
  raise Exception('class %s is not supported' % (cls))


def _get_immediate_type(insn):
  imm_type = None
  for arg in insn.get('args'):
    cls = arg.get('class')
    if asm_defs.is_imm(cls):
      assert imm_type is None
      imm_type = _imm_types[cls]
  return imm_type


def _get_params(insn, filter=None):
  result = []
  arg_count = 0
  for arg in insn.get('args'):
    if asm_defs.is_implicit_reg(arg.get('class')):
      continue
    if filter is not None and filter(arg):
      continue
    result.append("[[maybe_unused]] %s arg%d" % (
      _get_arg_type_name(arg, insn.get('type', None)), arg_count))
    arg_count += 1
  return ', '.join(result)


def _contains_mem(insn):
  return any(asm_defs.is_mem_op(arg['class']) for arg in insn.get('args'))


def _get_template_name(insn):
  name = insn.get('asm')
  if '<' not in name:
    return None, name
  return 'template <%s>' % ', '.join(
      'int' if param.strip() in ROUNDING_MODES else
      'bool' if param.strip() in ('true', 'false') else
      'typename' if re.search('[_a-zA-Z]', param) else 'int'
      for param in name.split('<',1)[1][:-1].split(',')), name.split('<')[0]

def _handle_jump(insn, jump_is_conditional, f):
  arg_count = 0
  for arg in insn.get('args'):
    if _get_arg_type_name(arg, insn.get('type', None)) ==  "const Label&":
      if jump_is_conditional:
        print('  HandleConditionalJump(arg%d);' %
        arg_count, file=f)
        return
      print('  HandleUnconditionalJump(arg%d);' %
        arg_count, file=f)
      return
    if _get_arg_type_name(arg, insn.get('type', None)) ==  "Register":
      assert(not jump_is_conditional)
      print('  HandleUnconditionalJumpRegister();', file=f)
    arg_count += 1

def _check_insn_is_dependency_breaking(insn):
  if "dependency_breaking" in insn:
    return True

def _handle_def_register_reset(name, insn, arch, f):
  """
  'def'/'def_early_clobber' registers in an intrinsic must be defined before they can be used.
  Verifier assembler checks intrinsics to ensure that this is the case.
  However, there are a number of special instructions where it is valid for a 'def'
  register to be both read and written when it's first defined.
  Example: A Xor instruction, using the same register as both input and output arguments sets this
  register to 0, regardless of its initial value, effectively resetting the register.
  Thus, it is valid for this instruction to read and write a 'def' register, even if it hasn't
  been written to yet in the intrinsic.
  """
  if not _check_insn_is_dependency_breaking(insn):
    return
  if name.startswith("P") or name.startswith("V") or name.endswith("pd") or name.endswith("ps"):
    register_type = "XMMRegister"
  else:
    register_type = "Register"
  num_registers = 3 if name.startswith("V") else 2
  arg_count = 0
  registers = []
  for arg in insn.get('args'):
      if asm_defs.is_implicit_reg(arg.get('class')):
        continue
      if (_get_arg_type_name(arg, insn.get('type', None)) == register_type
          and 'x86' in arch):
        registers.append(arg_count)
      arg_count += 1
  if (len(registers) != num_registers):
    return
  if num_registers == 2:
    print('  HandleDefOrDefEarlyClobberRegisterReset(arg%d, arg%d);' %
    (registers[0], registers[1]), file=f)
  else:
    print('  HandleDefOrDefEarlyClobberRegisterReset(arg%d, arg%d, arg%d);' %
    (registers[0], registers[1], registers[2]), file=f)

def _get_implicit_fixed_register(arg_class):
  if arg_class in ["AL", "AX", "EAX", "RAX"]:
    return "gpr_a"
  if arg_class in ["EBX", "RBX"]:
    return "gpr_b"
  if arg_class in ["CL", "CX", "ECX", "RCX"]:
    return "gpr_c"
  if arg_class in ["DL", "DX", "EDX", "RDX"]:
    return "gpr_d"
  return False

def _gen_register_read_write_info(insn, arch):
  # Process register uses before register defs. This ensures valid register uses are verified
  # against register definitions that occurred only before the current instruction.
  register_types_to_gen = ['Register', 'XMMRegister']
  for usage in ('use', 'def'):
    arg_count = 0
    for arg in insn.get('args'):
      if asm_defs.is_implicit_reg(arg.get('class')):
        implicit_fixed_reg = _get_implicit_fixed_register(arg.get('class'))
        if implicit_fixed_reg and (arg.get('usage') == usage or arg.get('usage') == "use_def"):
          yield '  Register%s(%s);' % (usage.capitalize(), implicit_fixed_reg)
        continue
      if (_get_arg_type_name(arg, insn.get('type', None)) in register_types_to_gen
          and 'x86' in arch):
        if arg.get('usage') == usage or arg.get('usage') == "use_def":
          yield '  Register%s(arg%d);' % (usage.capitalize(), arg_count)
      arg_count += 1

def _check_insn_uses_xmm(insn, arch):
  for arg in insn.get('args'):
    if (asm_defs.is_xreg(arg.get('class')) and 'x86' in arch):
      return True
  return False

def _gen_generic_functions_h(f, insns, arch, assembler_mode):
  template_names = set()
  for insn in insns:
    template, name = _get_template_name(insn)
    params = _get_params(insn)
    imm_type = _get_immediate_type(insn)
    if template:
      # We could only describe each template function once, or that would be
      # compilation error.  Yet functions with the same name but different
      # arguments are different (e.g. MacroVTbl<15> and MacroVTbl<23>).
      # But different types of arguments could map to the same C++ type.
      # For example MacroCmpFloat<Float32> and MacroCmpFloat<Float64> have
      # different IR arguments (FpReg32 vs FpReg64), but both map to the
      # same C++ type: XMMRegister.
      #
      # Use function name + parameters (as described by _get_params) to get
      # full description of template function.
      template_name = str({
          'name': name,
          'params': params
      })
      if template_name in template_names:
        continue
      template_names.add(template_name)
      print(template, file=f)
    # If this is binary assembler then we only generate header and then actual
    # implementation is written manually.
    #
    # Text assembled passes "real" work down to GNU as, this works fine with
    # just a simple generic implementation.
    if assembler_mode == AssemblerMode.BINARY_ASSEMBLER:
      if 'opcode' in insn:
        assert '' not in insn
        insn['opcodes'] = [insn['opcode']]
      if 'opcodes' in insn:
        opcodes = []
        for opcode in insn['opcodes']:
          if re.match('^[0-9a-fA-F]{2}$', opcode):
            opcodes.append('uint8_t{0x%s}' % opcode)
          elif re.match('^[0-9a-fA-F]{4}$', opcode):
            opcodes.append('uint16_t{0x%s}' % opcode)
          elif re.match('^[0-9a-fA-F]{8}$', opcode):
            opcodes.append('uint32_t{0x%s}' % opcode)
          elif re.match('^[0-9a-fA-F]{4}_[0-9a-fA-F]{4}$', opcode):
            opcodes.append('uint32_t{0x%s}' % re.sub('_', '\'', opcode))
          elif re.match('^[0-7]$', opcode):
            opcodes.append('uint8_t{%s}' % opcode)
          else:
            assert False
        insn['processed_opcodes'] = opcodes
        print('constexpr void %s(%s) {' % (name, params), file=f)
        if 'x86' in arch:
          _gen_emit_shortcut(f, insn, insns)
        _gen_emit_instruction(f, insn, arch)
        print('}', file=f)
        # If we have a memory operand (there may be at most one) then we also
        # have a special x86-64 exclusive form which accepts Label (it can be
        # emulated on x86-32, too, if needed).
        if 'const Operand&' in params and 'x86' in arch:
          print("", file=f)
          print('constexpr void %s(%s) {' % (
              name, params.replace('const Operand&', 'const LabelOperand')), file=f)
          _gen_emit_shortcut(f, insn, insns)
          _gen_emit_instruction(f, insn, arch, rip_operand=True)
          print('}\n', file=f)
        if 'Rounding' in params:
          print("", file=f)
          print('constexpr void %s(%s) {' % (
              name, _get_params(insn, lambda arg: arg.get('class', '') == 'Rm')), file=f)
          _gen_emit_instruction(f, insn, arch, dyn_rm=True)
          print('}\n', file=f)
      else:
        print('constexpr void %s(%s);' % (name, params), file=f)
      # If immediate type is integer then we want to prevent automatic
      # conversions from integers of larger sizes.
      if imm_type is not None and "int" in imm_type:
        if template:
          print(template[:-1] + ", typename ImmType>", file=f)
        else:
          print('template<typename ImmType>', file=f)
        print(('auto %s(%s) -> '
                    'std::enable_if_t<std::is_integral_v<ImmType> && '
                    'sizeof(%s) < sizeof(ImmType)> = delete;') % (
                        name, params.replace(imm_type, 'ImmType'), imm_type), file=f)

    elif assembler_mode == AssemblerMode.TEXT_ASSEMBLER:
      print('constexpr void %s(%s) {' % (name, params), file=f)
      print('  Instruction(%s);' % ', '.join(
          ['"%s"' % insn.get('native-asm', name)] +
          list(_gen_instruction_args(insn, arch))), file=f)
      print('}', file=f)

    else: # verifier_assembler
      print('constexpr void %s(%s) {' % (name, params), file=f)
      if (name == "Jcc" or name == "Jmp") and 'x86' in arch:
        jump_is_conditional = name == "Jcc"
        _handle_jump(insn, jump_is_conditional, f)
        print(' EndInstruction();', file=f)
        print('}', file=f)
        continue
      if 'feature' in insn:
        print('  SetRequiredFeature%s();' % insn['feature'], file=f)
      else:
        if _check_insn_uses_xmm(insn, arch):
          print('  SetRequiredFeatureSSEOrSSE2();', file=f)
      for arg in insn.get('args'):
        if arg["class"] == "FLAGS":
          print('  SetDefinesFLAGS();', file=f)
          break
      _handle_def_register_reset(name, insn, arch, f)
      for register_read_write in _gen_register_read_write_info(insn, arch):
        print(register_read_write, file=f)
      print('  EndInstruction();', file=f)
      print('}', file=f)


def _gen_instruction_args(insn, arch):
  arg_count = 0
  for arg in insn.get('args'):
    if asm_defs.is_implicit_reg(arg.get('class')):
      continue
    if (_get_arg_type_name(arg, insn.get('type', None)) == 'Register'
        and 'x86' in arch):
      yield 'typename DerivedAssemblerType::%s(arg%d)' % (
          _ARGUMENT_FORMATS_TO_SIZES[arg['class']], arg_count)
    else:
      yield 'arg%d' % arg_count
    arg_count += 1


def _gen_emit_shortcut(f, insn, insns):
  # If we have one 'Imm8' argument then it could be shift, try too see if
  # ShiftByOne with the same arguments exist.
  if asm_defs.exactly_one_of(arg['class'] == 'Imm8' for arg in insn['args']):
    _gen_emit_shortcut_shift(f, insn, insns)
  if asm_defs.exactly_one_of(arg['class'] in ('Imm16', 'Imm32') for arg in insn['args']):
    if insn['asm'].endswith('Accumulator'):
      _gen_emit_shortcut_accumulator_imm8(f, insn, insns)
    else:
      _gen_emit_shortcut_generic_imm8(f, insn, insns)
  if len(insn['args']) > 1 and insn['args'][0]['class'].startswith('GeneralReg'):
    _gen_emit_shortcut_accumulator(f, insn, insns)


def _gen_emit_shortcut_shift(f, insn, insns):
  # Replace Imm8 argument with '1' argument.
  non_imm_args = [arg for arg in insn['args'] if arg['class'] != 'Imm8']
  imm_arg_index = insn['args'].index({'class': 'Imm8'})
  for maybe_shift_by_1_insn in insns:
    if not _is_insn_match(maybe_shift_by_1_insn,
                          insn['asm'] + 'ByOne',
                          non_imm_args):
      continue
    # Now call that version if immediate is 1.
    args = []
    arg_count = 0
    for arg in non_imm_args:
      if asm_defs.is_implicit_reg(arg['class']):
        continue
      args.append('arg%d' % arg_count)
      arg_count += 1
    print('  if (arg%d == 1) return %sByOne(%s);' % (
        imm_arg_index, insn['asm'], ', '.join(args)), file=f)


def _gen_emit_shortcut_accumulator_imm8(f, insn, insns):
  insn_name = insn['asm'][:-11]
  args = insn['args']
  assert len(args) == 3 and args[2]['class'] == 'FLAGS'
  acc_class = args[0]['class']
  # Note: AL is accumulator, too, but but imm is always 8-bit for it which means
  # it shouldn't be encountered here and if it *does* appear here - it's an error
  # and we should fail.
  assert acc_class in ('AX', 'EAX', 'RAX')
  greg_class = {
      'AX': 'GeneralReg16',
      'EAX': 'GeneralReg32',
      'RAX': 'GeneralReg64'
  }[acc_class]
  maybe_8bit_imm_args = [
    { 'class': greg_class, 'usage': args[0]['usage'] },
    { 'class': 'Imm8' },
    { 'class': 'FLAGS', 'usage': insn['args'][2]['usage'] }
  ]
  for maybe_imm8_insn in insns:
    if not _is_insn_match(maybe_imm8_insn,
                          insn_name + 'Imm8',
                          maybe_8bit_imm_args):
      continue
    print('  if (IsInRange<int8_t>(arg0)) {', file=f)
    print(('    return %s(DerivedAssemblerType::Accumulator(), '
                 'static_cast<int8_t>(arg0));') % (
                     maybe_imm8_insn['asm'],), file=f)
    print('  }', file=f)

def _gen_emit_shortcut_generic_imm8(f, insn, insns):
  maybe_8bit_imm_args = [{ 'class': 'Imm8' } if arg['class'].startswith('Imm') else arg
                         for arg in insn['args']]
  imm_arg_index = maybe_8bit_imm_args.index({'class': 'Imm8'})
  for maybe_imm8_insn in insns:
    if not _is_insn_match(maybe_imm8_insn,
                          insn['asm'] + 'Imm8',
                          maybe_8bit_imm_args):
      continue
    # Now call that version if immediate fits into 8-bit.
    arg_count = len(_get_params(insn).split(','))
    print('  if (IsInRange<int8_t>(arg%d)) {' % (arg_count - 1), file=f)
    print('   return %s(%s);' % (maybe_imm8_insn['asm'], ', '.join(
        ('static_cast<int8_t>(arg%d)' if n == arg_count - 1 else 'arg%d') % n
        for n in range(arg_count))), file=f)
    print('  }', file=f)


def _gen_emit_shortcut_accumulator(f, insn, insns):
  accumulator_name = {
      'GeneralReg8': 'AL',
      'GeneralReg16': 'AX',
      'GeneralReg32': 'EAX',
      'GeneralReg64': 'RAX'
  }[insn['args'][0]['class']]
  maybe_accumulator_args = [
      { 'class': accumulator_name, 'usage': insn['args'][0]['usage']}
  ] + insn['args'][1:]
  for maybe_accumulator_insn in insns:
    if not _is_insn_match(maybe_accumulator_insn,
                          insn['asm'] + 'Accumulator',
                          maybe_accumulator_args):
      continue
    # Now call that version if register is an Accumulator.
    arg_count = len(_get_params(insn).split(','))
    print('  if (DerivedAssemblerType::IsAccumulator(arg0)) {', file=f)
    print('  return %s(%s);' % (
      maybe_accumulator_insn['asm'],
      ', '.join('arg%d' % n for n in range(1, arg_count))), file=f)
    print('}', file=f)


def _is_insn_match(insn, expected_name, expected_args):
  # Note: usually there are more than one instruction with the same name
  # but different arguments because they could accept either GeneralReg
  # or Memory or Immediate argument.
  #   Instructions:
  #     Addl %eax, $1
  #     Addl (%eax), $1
  #     Addl %eax, %eax
  #     Addl (%eax), %eax
  #   are all valid.
  #
  # Yet not all instruction have all kinds of optimizations: TEST only have
  # version with accumulator (which is shorter than usual) - but does not
  # have while version with short immediate. Imul have version with short
  # immediate - but not version with accumulator.
  #
  # We want to ensure that we have the exact match - expected name plus
  # expected arguments.

  return insn['asm'] == expected_name and insn['args'] == expected_args


_ARGUMENT_FORMATS_TO_SIZES = {
  'X87Reg' : 'RegisterDefaultBit',
  'Cond': '',
  'FpReg32' : 'VectorRegister128Bit',
  'FpReg64' : 'VectorRegister128Bit',
  'GeneralReg' : 'RegisterDefaultBit',
  'GeneralReg8' : 'Register8Bit',
  'GeneralReg16' : 'Register16Bit',
  'GeneralReg32' : 'Register32Bit',
  'GeneralReg64' : 'Register64Bit',
  'Imm2': '',
  'Imm8': '',
  'Imm16': '',
  'Imm32': '',
  'Imm64': '',
  'Mem': 'MemoryDefaultBit',
  'Mem8' : 'Memory8Bit',
  'Mem16' : 'Memory16Bit',
  'Mem32' : 'Memory32Bit',
  'Mem64' : 'Memory64Bit',
  'Mem128' : 'Memory128Bit',
  'MemX87': 'MemoryX87',
  'MemX8716': 'MemoryX8716Bit',
  'MemX8732': 'MemoryX8732Bit',
  'MemX8764': 'MemoryX8764Bit',
  'MemX8780': 'MemoryX8780Bit',
  'RegX87': 'X87Register',
  'XmmReg' : 'VectorRegister128Bit',
  'YmmReg' : 'VectorRegister256Bit',
  'VecMem32': 'VectorMemory32Bit',
  'VecMem64': 'VectorMemory64Bit',
  'VecMem128': 'VectorMemory128Bit',
  'VecMem256': 'VectorMemory256Bit',
  'VecReg128' : 'VectorRegister128Bit',
  'VecReg256' : 'VectorRegister256Bit'
}


# On x86-64 each instruction which accepts explicit memory operant (there may at most be one)
# can also accept $rip-relative addressing (where distance to operand is specified by 32-bit
# difference between end of instruction and operand address).
#
# We use it to support Label operands - only name of class is changed from MemoryXXX to LabelXXX,
# e.g. VectorMemory32Bit becomes VectorLabel32Bit.
#
# Note: on x86-32 that mode can also be emulated using regular instruction form, if needed.
def _gen_emit_instruction(f, insn, arch, rip_operand=False, dyn_rm=False):
  result = []
  arg_count = 0
  for arg in insn['args']:
    if asm_defs.is_implicit_reg(arg['class']):
      continue
    # Note: in RISC-V there is never any ambiguity about whether full register or its part is used.
    # Instead size of operand is always encoded in the name, e.g. addw vs add or fadd.s vs fadd.d
    if arch in ['common_riscv', 'rv32', 'rv64']:
      if dyn_rm and arg['class'] == 'Rm':
        result.append('Rounding::kDyn')
      else:
        result.append('arg%d' % arg_count)
    else:
      result.append('%s(arg%d)' % (_ARGUMENT_FORMATS_TO_SIZES[arg['class']], arg_count))
    arg_count += 1
  # If we want %rip--operand then we need to replace 'Memory' with 'Labal'
  if rip_operand:
    result = [arg.replace('Memory', 'Label') for arg in result]
  print('  Emit%sInstruction<%s>(%s);' % (
      asm_defs._get_cxx_name(insn.get('type', '')),
      ', '.join(insn['processed_opcodes']),
      ', '.join(result)), file=f)


def _gen_memory_function_specializations_h(f, insns, arch):
  for insn in insns:
    # Only build additional definitions needed for memory access in LIR if there
    # are memory arguments and instruction is intended for use in LIR
    if not _contains_mem(insn) or insn.get('skip_lir'):
      continue
    template, _ = _get_template_name(insn)
    params = _get_params(insn)
    for addr_mode in ('Absolute', 'BaseDisp', 'IndexDisp', 'BaseIndexDisp'):
      # Generate a function to expand a macro and emit a corresponding
      # assembly instruction with a memory operand.
      macro_name = asm_defs.get_mem_macro_name(insn, addr_mode)
      incoming_args = []
      outgoing_args = []
      for i, arg in enumerate(insn.get('args')):
        if asm_defs.is_implicit_reg(arg.get('class')):
          continue
        arg_name = 'arg%d' % (i)
        if asm_defs.is_mem_op(arg.get('class')):
          if addr_mode == 'Absolute':
            incoming_args.append('int32_t %s' % (arg_name))
            outgoing_args.append('{.disp = %s}' % (arg_name))
            continue
          mem_args = []
          if addr_mode in ('BaseDisp', 'BaseIndexDisp'):
            mem_args.append(['Register', 'base', arg_name + '_base'])
          if addr_mode in ('IndexDisp', 'BaseIndexDisp'):
            mem_args.append(['Register', 'index', arg_name + '_index'])
            mem_args.append(['ScaleFactor', 'scale', arg_name + '_scale'])
          mem_args.append(['int32_t', 'disp', arg_name + '_disp'])
          incoming_args.extend(['%s %s' % (pair[0], pair[2]) for pair in mem_args])
          outgoing_args.append('{%s}' % (
              ', '.join(['.%s = %s' % (pair[1], pair[2]) for pair in mem_args])))
        else:
          incoming_args.append('%s %s' % (_get_arg_type_name(arg, None), arg_name))
          outgoing_args.append(arg_name)
      if template:
        print(template, file=f)
      print('constexpr void %s(%s) {' % (macro_name, ', '.join(incoming_args)), file=f)
      print('  %s(%s);' % (insn.get('asm'), ', '.join(outgoing_args)), file=f)
      print('}', file=f)


def _is_for_asm(insn):
  if insn.get('skip_asm'):
    return False
  return True


def _load_asm_defs(asm_def):
  arch, insns = asm_defs.load_asm_defs(asm_def)
  # Filter out explicitly disabled instructions.
  return arch, [i for i in insns if _is_for_asm(i)]


def main(argv):
  # Usage: gen_asm.py --binary-assembler|--text_assembler
  #                   <assembler_common-inl.h>
  #                   <assembler_<arch>-inl.h>
  #                   ...
  #                   <def_common>
  #                   <def_arch>
  #                   ...
  #
  # Usage: gen_asm.py --using
  #                   <def_common>
  #                   <def_arch>
  #                   ...
  mode = argv[1]

  if (mode != '--binary-assembler' and
      mode != '--text-assembler' and
      mode != '--verifier-assembler' and
      mode != '--using'):
    assert False, 'unknown option %s' % (mode)

  if mode == '--binary-assembler' or mode == '--text-assembler' or mode == "--verifier-assembler":
    if mode == '--binary-assembler':
      assembler_mode = AssemblerMode.BINARY_ASSEMBLER
    elif mode == '--text-assembler':
      assembler_mode = AssemblerMode.TEXT_ASSEMBLER
    else:
      assembler_mode = AssemblerMode.VERIFIER_ASSEMBLER

    assert len(argv) % 2 == 0
    filenames = argv[2:]
    filename_pairs = ((filenames[i], filenames[len(filenames)//2 + i])
                      for i in range(0, len(filenames)//2))

    for out_filename, input_filename in filename_pairs:
      arch, loaded_defs = _load_asm_defs(input_filename)
      with open(out_filename, 'w') as out_file:
        _gen_generic_functions_h(out_file, loaded_defs, arch, assembler_mode)
        if assembler_mode == AssemblerMode.BINARY_ASSEMBLER and arch is not None and 'x86' in arch:
          _gen_memory_function_specializations_h(out_file, loaded_defs, arch)
  else:
    assert mode == '--using'

    instruction_names = set()
    for input_filename in argv[3:]:
      arch, loaded_defs = _load_asm_defs(input_filename)
      for insn in loaded_defs:
        instruction_names.add(insn['asm'])

    with open(argv[2], 'w') as out_file:
      print("""
#ifndef IMPORT_ASSEMBLER_FUNCTIONS
#error This file is supposed to be included from berberis/intrinsics/macro_assembler-inl.h
#endif
""", file=out_file)
      for name in instruction_names:
        print('using Assembler::%s;' % name, file=out_file)

if __name__ == '__main__':
  sys.exit(main(sys.argv))
