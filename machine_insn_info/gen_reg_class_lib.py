#!/usr/bin/python3
#
# Copyright (C) 2023 The Android Open Source Project
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

"""Generate machine IR register class definitions from data file."""

def gen_machine_reg_class_inc(f, reg_classes):
  for reg_class in reg_classes:
    print(f'class {reg_class.get('name')};', file=f)
  for reg_class in reg_classes:
    name = reg_class.get('name')
    regs = reg_class.get('regs')
    size = reg_class.get('size') * 8
    print(f'class {name} {{', file=f)
    print(' public:', file=f)
    print(f'  static constexpr const char* kName = "{name}";', file=f)
    print('  static constexpr size_t kSizeInBits = %d;' % size, file=f)
    print('  using RegistersList = std::tuple<%s>;' % ', '.join(regs), file=f)
    if 'gcc_asm_name' in reg_class:
      if 'type' in reg_class:
        print('  using Type = %s;' % reg_class['type'], file=f)
      elif size == 128:
        print('  using Type = __m128;', file=f)
      elif size == 256:
        print('#ifdef __AVX__', file=f)
        print('  using Type = __m256;', file=f)
        print('#endif', file=f)
      else:
        print('  using Type = uint%d_t;' % size, file=f)
      gcc_asm_name = reg_class.get('gcc_asm_name')
      print(f'  static constexpr char kAsRegister = \'{gcc_asm_name}\';', file=f)
    else:
      # std::conditional_t requires type even for branch that wouldn't be taken.
      # Use of `void` as type here means it would be compatible with that logic,
      # but would exclude most accidental uses of it because `void` can not be used
      # to declare arguments of functions, or local variables.
      print('  using Type = void;', file=f)
    if len(regs) == 1:
      print('  template <typename Assembler>', file=f)
      print('  static constexpr auto kAssemblerRegisterPointer = '
            f'&Assembler::gpr_{gcc_asm_name};', file=f)
    print('  template <typename MachineRegDefinitions>', file=f)
    print('  static constexpr auto kMachineRegId = '
          f'MachineRegDefinitions::k{name};', file=f)
    print('};', file=f)


def expand_aliases(reg_classes):
  expanded = {}
  for reg_class in reg_classes:
    expanded_regs = []
    for r in reg_class.get('regs'):
      expanded_regs.extend(expanded.get(r, [r]))
    reg_class['regs'] = expanded_regs
    expanded[reg_class.get('name')] = expanded_regs
