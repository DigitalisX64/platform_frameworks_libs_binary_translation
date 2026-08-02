#!/usr/bin/env python3
# Copyright (C) 2026 utzcoz
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

"""Harvest (encoding, objdump-mnemonic) pairs from real ARM64 binaries.

The decoder differential test needs ground truth for what an encoding *is*.
objdump is that ground truth: it is an independent implementation of the same
ARM ARM tables the decoder implements, so a disagreement is a decoder bug in
almost every case.

Real shipped code is used rather than synthetic encodings on purpose --- the
historical decoder bugs (LDPSW zero-extending, SWP routed to LDADD, CMGT mapped
to SMAX) were all in encodings that occur constantly in real libraries and not
at all in a hand-written test list.

Usage:
  harvest_encodings.py --objdump <llvm-objdump> --out encodings.txt <lib.so>...

Output is one `<hex8> <mnemonic>` pair per line, deduplicated by encoding and
sorted, so regenerating from a different library set produces a reviewable diff
rather than a reshuffle.
"""

import argparse
import re
import subprocess
import sys

# `    8004: 1003ffe0     	adr	x0, 0x10000`
INSN_RE = re.compile(r"^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+(\S+)")


def is_sve(enc):
    """True for the SVE/SVE2 encoding space (A64 top-level op0 == 0b0010).

    Digitalis implements A64 base plus AdvSIMD/NEON and crypto; SVE is out of
    scope and the decoder correctly reports such encodings as Undefined. objdump
    still disassembles them, so without this filter a single string constant that
    happens to sit in .text ("gome", 0x656d6f67, decodes as SVE FNMLS) shows up
    as a decoder mismatch. Exactly one such word appears across the whole system
    image --- this is noise removal, not the suppression of a real gap. Drop the
    filter if SVE is ever implemented.
    """
    return (enc >> 25) & 0xF == 0b0010


def harvest(objdump, path):
    """Return {encoding: mnemonic} for one binary."""
    out = {}
    try:
        proc = subprocess.run(
            [objdump, "-d", path],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        print(f"warning: cannot disassemble {path}: {e}", file=sys.stderr)
        return out
    for line in proc.stdout.splitlines():
        m = INSN_RE.match(line)
        if not m:
            continue
        enc, mnemonic = m.group(1), m.group(2)
        # objdump emits the raw word for data and for encodings it cannot
        # decode; both are useless as ground truth. `<unknown>` is its marker.
        if mnemonic.startswith("<") or mnemonic == "...":
            continue
        if is_sve(int(enc, 16)):
            continue
        # llvm-objdump prints the bytes in memory order; ARM64 is little-endian
        # so the printed 8 hex digits are already the instruction word.
        out[enc] = mnemonic
    return out


def cap_per_mnemonic(pairs, cap):
    """Keep at most `cap` encodings per mnemonic, spread across the range.

    The decoder dispatches on opcode/size/option fields, not on register numbers,
    so the thousands of `add` encodings in a system image exercise one decode
    path with different register bits. Capping keeps the corpus committable while
    losing almost no decode coverage.

    Selection is evenly spaced across the sorted encodings rather than the first
    `cap`, because taking the first N clusters in low encoding space --- which
    for most formats means low register numbers and small immediates, exactly the
    fields that do NOT vary the decode path. Even spacing samples the whole
    immediate/option range instead.
    """
    by_mnemonic = {}
    for enc, mnemonic in pairs.items():
        by_mnemonic.setdefault(mnemonic, []).append(enc)
    kept = {}
    for mnemonic, encs in by_mnemonic.items():
        encs.sort()
        if len(encs) <= cap:
            chosen = encs
        else:
            step = len(encs) / cap
            chosen = [encs[int(i * step)] for i in range(cap)]
        for e in chosen:
            kept[e] = mnemonic
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objdump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cap",
        type=int,
        default=128,
        help="max encodings kept per mnemonic (0 = keep all)",
    )
    ap.add_argument("libs", nargs="+")
    args = ap.parse_args()

    merged = {}
    for lib in args.libs:
        got = harvest(args.objdump, lib)
        merged.update(got)
    raw = len(merged)
    if args.cap:
        merged = cap_per_mnemonic(merged, args.cap)

    with open(args.out, "w") as f:
        f.write("# ARM64 (encoding, objdump mnemonic) pairs harvested from real\n")
        f.write("# shipped binaries. Regenerate with harvest_encodings.py.\n")
        f.write("# One `<hex8> <mnemonic>` per line, deduplicated and sorted.\n")
        for enc in sorted(merged):
            f.write(f"{enc} {merged[enc]}\n")
    mnemonics = len(set(merged.values()))
    print(
        f"wrote {len(merged)} encodings ({mnemonics} mnemonics) to {args.out}"
        f" [from {raw} unique, cap={args.cap}]"
    )


if __name__ == "__main__":
    main()
