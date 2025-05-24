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
  'variants' : '|001|variants',
  'insn' : '|002|insn',
  'feature' : '|003|feature',
  'nan' : '|003|nan',
  'usage' : '|003|usage',
  'in' : '|004|in',
  'out' : '|005|out',
  'comment' : '|010|comment'
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
  # Usage: prettify_ir_binding.py <file.json>

  with open(argv[1]) as file:
    json_bindings = json.load(file)

  out_bindings = []
  license_text = []
  for binding in json_bindings:
    new_binding = {}
    if isinstance(binding, str):
      license_text.append(binding)
    else:
      for field, value in binding.items():
        new_binding[field_substitute[field]] = value
      out_bindings.append(new_binding)

  out_bindings = license_text + sorted(out_bindings,
      key=lambda binding:
          Version(binding[field_substitute['name']] +
                  str(binding.get(field_substitute['variants'], '')) +
                  str(binding.get(field_substitute['usage'], '')) +
                  str(binding.get(field_substitute['nan'], '')) +
                  str(binding.get(field_substitute['feature'], ''))))

  text = json.dumps(out_bindings, indent=2, sort_keys=True)

  # Remove numbers from names of fields
  text = re.sub('[|][0-9][0-9][0-9][|]', '', text)

  def replace_if_short(match):
    match = match.group()
    replace = ' '.join(match.split())
    if len(replace) < 90:
      return replace
    else:
      return match

  # Make short lists one-liners
  text = re.sub('[\\[{][^][{}]*[]}]', replace_if_short, text)

  # Remove trailing spaces
  text = re.sub(' $', '', text, flags=re.MULTILINE)

  # Fix the license
  text = re.sub('\\\\u201c', '“', text, flags=re.MULTILINE)
  text = re.sub('\\\\u201d', '”', text, flags=re.MULTILINE)

  with open(argv[1], 'w') as file:
    print(text, file=file)

if __name__ == '__main__':
  sys.exit(main(sys.argv))
