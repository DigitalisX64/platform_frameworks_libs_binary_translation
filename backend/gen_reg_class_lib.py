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
    print(f'class {name} {{', file=f)
    print(' public:', file=f)
    print(f'  static constexpr const char* kName = "{name}";', file=f)
    print('  static constexpr size_t kSizeInBits = %d;' %
        (reg_class.get('size') * 8), file=f)
    print('  using RegistersList = std::tuple<%s>;' % ', '.join(regs), file=f)
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
