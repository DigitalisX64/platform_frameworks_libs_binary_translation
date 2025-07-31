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
#

import argparse
import os
import re
import sys

# The core regex to find C-style hexadecimal literals.
# It's case-insensitive and captures the prefix, digits, and suffix.
# Group 1: '0x' prefix
# Group 2: The hexadecimal digits (and any existing separators)
# Group 3: The type suffix (e.g., 'u', 'll', 'ull')
HEX_LITERAL_REGEX = re.compile(
    r"""
    \b          # Word boundary, to not match things like 'my0xvar'
    (0x)        # Group 1: The '0x' prefix
    ([0-9a-f']+) # Group 2: Hex digits, allowing for existing ' separators
    ([ul]*)?    # Group 3: Optional suffix (U, L, UL, ULL, etc.)
    \b          # Word boundary
    """,
    re.IGNORECASE | re.VERBOSE,
)

def format_hex_literal(match: re.Match) -> str:
    """A replacement function for re.sub to format a matched hex literal."""
    # Per the rules: prefix and digits are lowercase.
    prefix = match.group(1)
    digits = match.group(2)
    suffix = match.group(3) or ''

    assert(prefix == '0x')

    # Sanitize the digits by lowercasing and removing any existing separators.
    clean_digits = digits.lower().replace("'", "")

    # If the number has more than 4 digits, add separators.
    # This also re-formats existing numbers to ensure consistency.
    if len(clean_digits) > 4:
        # Insert separators every 4 characters from the right.
        parts = []
        temp_digits = clean_digits
        while temp_digits:
            parts.append(temp_digits[-4:])
            temp_digits = temp_digits[:-4]
        formatted_digits = "'".join(reversed(parts))
    else:
        formatted_digits = clean_digits

    # Per the rules: suffix must be uppercase.
    formatted_suffix = suffix.upper()

    return f"{prefix}{formatted_digits}{formatted_suffix}"

def process_file(file_path: str, dry_run: bool):
    """Reads a file, formats hex literals, and writes it back if changed."""
    try:
        # Use 'ignore' to handle files that may not be UTF-8.
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()

        new_content = HEX_LITERAL_REGEX.sub(format_hex_literal, original_content)

        if original_content != new_content:
            if dry_run:
                print(f"Changes needed to: {file_path}")
            else:
                print(f"Applying changes to: {file_path}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

    except IOError as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)

def main():
    """Parses command-line arguments and processes files."""
    parser = argparse.ArgumentParser(
        description="""
A script to format hexadecimal literals in source code files to comply with:
1. Lowercase literals (0x...).
2. Uppercase suffixes (...U, ...ULL).
3. Digit separators for numbers > 32 bits (0x....'....).
""",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Process a single file in the current directory
  python format_literals.py my_file.c

  # Process multiple specific files
  python format_literals.py ./src/main.c ./include/config.h

  # Recursively process all default file types in a directory
  python format_literals.py -r ./src

  # Do a 'dry run' to see which files would be changed, without saving
  python format_literals.py -r --dry-run ./src

  # Recursively process only .c and .h files
  python format_literals.py -r --ext .c --ext .h ./src
"""
    )
    parser.add_argument(
        "paths",
        metavar="PATH",
        nargs='+',
        help="One or more file or directory paths to process."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search for files in any specified directories."
    )
    parser.add_argument(
        "--ext",
        dest="extensions",
        action="append",
        help="File extension to process (e.g., .c). Can be specified multiple times. "
             "If used, this overrides the default list."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be modified without actually changing them."
    )

    args = parser.parse_args()

    # Determine which file extensions to target.
    if args.extensions:
        # User specified their own list of extensions.
        target_extensions = set(args.extensions)
    else:
        # Use the default list of common C/C++ extensions.
        target_extensions = {'.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.inl'}

    # Collect all file paths to process.
    files_to_process = []
    for path in args.paths:
        if os.path.isfile(path):
            files_to_process.append(path)
        elif os.path.isdir(path):
            if not args.recursive:
                print(f"Warning: '{path}' is a directory. Use -r or --recursive to process its contents.", file=sys.stderr)
                continue
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if any(filename.endswith(ext) for ext in target_extensions):
                        files_to_process.append(os.path.join(root, filename))
        else:
            print(f"Error: Path not found '{path}'", file=sys.stderr)

    if not files_to_process:
        print("No files found to process. Check your path and --ext arguments.")
        return

    print(f"Processing {len(files_to_process)} files...")
    for file_path in files_to_process:
        process_file(file_path, args.dry_run)

    print("\nFormatting complete.")

if __name__ == "__main__":
    main()
