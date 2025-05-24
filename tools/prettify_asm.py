#!/usr/bin/python
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
#

import json
import re
import sys

# Add numbers to the names of fields to rearrange them better.
# They would be removed from final file.
field_substitute = {
  'name' : '|000|name',
  'encodings' : '|000|encodings',
  'stems' : '|000|stems',
  'feature' : '|001|feature',
  'args' : '|010|args',
  'comment' : '|015|comment',
  'asm' : '|020|asm',
  'opcodes' : '|021|opcodes',
  'reg_to_rm' : '|022|reg_to_rm',
  'mnemo' : '|030|mnemo',
}

def Version(str):
  result = []
  isdigit = False
  word = ''
  for char in str + ('a' if str[-1:].isdigit() else '0'):
    if char.isdigit() == isdigit:
      word += char
    else:
      if isdigit:
        result.append(('0' * 1000 + word)[-1000:])
      else:
        result.append((word + ' ' * 1000)[:1000])
      isdigit = not isdigit
      word = char
  return '.'.join(result)

def main(argv):
  # Usage: prettify_asm_def.py <file.json>

  with open(argv[1]) as file:
    obj = json.load(file)

  insns = {}
  for insn in obj['insns']:
    if 'stems' in insn:
      sorted_stems = sorted(insn['stems'])
      insn['stems'] = sorted_stems
      name = Version(', '.join(sorted_stems) + '; ' + str(insn['args']))
    elif 'encodings' in insn:
      sorted_stems = sorted(insn['encodings'])
      name = Version(', '.join(sorted_stems) + '; ' + str(insn['args']))
    else:
      name = Version(insn['name'] + '; ' + str(insn['args']))
    new_insn = {}
    for field, value in insn.items():
      new_insn[field_substitute[field]] = value
    assert name not in insns
    insns[name] = new_insn

  obj['insns'] = [insn[1] for insn in sorted(iter(insns.items()))]

  text = json.dumps(obj, indent=2, sort_keys=True)

  # Remove numbers from names of fields
  text = re.sub('[|][0-9][0-9][0-9][|]', '', text)

  def replace_if_short(match):
    match = match.group()
    replace = ' '.join(match.split())
    if len(replace) < 100 or (
       len(replace) < 120 and 'optimizable_using_commutation' in replace):
      return replace
    else:
      return match

  # Make short lists one-liners
  text = re.sub('[\\[{][^][{}]*[]}]', replace_if_short, text)
  # Allow opcodes list.
  text = re.sub('[\\[{][^][{}]*"opcodes"[^][{}]*[\\[{][^][{}]*[]}][^][{}]*[]}]', replace_if_short, text)

  # Remove trailing spaces
  text = re.sub(' $', '', text, flags=re.MULTILINE)

  # Fix the license
  text = re.sub('\\\\u201c', '“', text, flags=re.MULTILINE)
  text = re.sub('\\\\u201d', '”', text, flags=re.MULTILINE)

  with open(argv[1], 'w') as file:
    print(text, file=file)

if __name__ == '__main__':
  sys.exit(main(sys.argv))
