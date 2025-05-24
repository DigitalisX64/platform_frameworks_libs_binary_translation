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

from collections import OrderedDict
import json
import re
import sys

# Add numbers to the names of fields to rearrange them better.
# They will be removed from final file.
field_substitute = {
  "comment": "|000|comment",
  "class": "|001|class",
  "note": "|001|note",
  "precise_nans": "|001|precise_nans",
  "variants": "|001|variants",
  "in": "|002|in",
  "out": "|003|out",
  "side_effects_comment": "|004|side_effects_comment",
  "has_side_effects": "|005|has_side_effects"
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
  # Usage: prettify_intrinsics.py <file.json>

  with open(argv[1]) as file:
    json_intrinsics = json.load(file)

  out_intrinsics = OrderedDict()
  license = None
  for intrinsic_name, intrinsic_body in json_intrinsics.items():
    new_intrinsic = {}
    if intrinsic_name == 'License':
      license = intrinsic_body
    else:
      for field, value in intrinsic_body.items():
        if field == 'variants':
          new_intrinsic[field_substitute[field]] = sorted(value, key=Version)
        else:
          new_intrinsic[field_substitute[field]] = value
      out_intrinsics[intrinsic_name] = new_intrinsic

  text = json.dumps(out_intrinsics, indent=2, sort_keys=True)

  # Add license back if present
  if license:
    license = json.dumps([license], indent=2)
    text = '{\n  "License":' + license[3:-2] + ',\n' + text[2:]

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
