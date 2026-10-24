/*
 * Copyright (C) 2026 utzcoz
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Per-tier coverage table over the committed decoder corpus.
//
// ARM64 guest code runs through three tiers, and they are allowed to have
// different coverage: the interpreter is the fallback, the lite translator is
// the first gear, the heavy optimizer is the second. A gap in the JIT tiers is
// not a crash --- a heavy bail just falls back to lite, and a lite decline just
// falls back to the interpreter. That is exactly what makes coverage
// regressions invisible: they are correct-but-slow, so no correctness test can
// see them.
//
// The cost is real. The heavy frontend silently bailing on ADRP, MRS TPIDR_EL0
// and SBFX -- all long since handled by the interpreter and the lite tier --
// made it bail out of nearly every real-application region, so the second gear
// never engaged on real apps and a security SDK's CRC loop ran at lite speed
// until it tripped the app's own watchdog. Nothing failed; it just got slow.
//
// So this test measures, per mnemonic, how many of the corpus encodings each
// JIT tier can translate, and compares that against a committed table. It is
// asymmetric on purpose:
//   * a tier translating FEWER encodings of a mnemonic than the table records
//     fails --- that is coverage lost;
//   * a tier translating MORE is reported, not failed --- that is someone
//     implementing something, and the table is regenerated in the same commit.
//
// The existing `...Bails` assertions in frontend_tests.cc pin individual
// intentional bails. They are the opposite of this test: they lock in the
// current bail set for specific encodings, while this one tracks the aggregate
// and notices drift. Both are useful; neither replaces the other.
//
// Nothing here EXECUTES guest code. The corpus is harvested from real binaries,
// so its encodings reference arbitrary registers and addresses; they are only
// ever handed to the two translators, which are pure code generators.
//
// The interpreter is not a column. Its dispatch goes through the same Decoder,
// so an encoding the decoder routes to Undefined is exactly the set that
// DecoderObjdumpDiff.UndefinedCoverageBaseline already tracks, and determining
// whether a decoded instruction is implemented in the interpreter body would
// require running it.

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "android-base/file.h"

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/heavy_optimizer/arm64/heavy_optimize_region.h"
#include "berberis/lite_translator/lite_translate_region.h"

namespace berberis {

namespace {

constexpr const char* kCorpusFile = "decoder/arm64_test/arm64_decoder_encodings.txt";
constexpr const char* kTableFile = "heavy_optimizer/arm64/arm64_tier_coverage.txt";

std::string DataPath(const char* name) {
  return android::base::GetExecutableDirectory() + "/" + name;
}

struct Counts {
  int total = 0;
  int lite = 0;
  int heavy = 0;
};

// Single-instruction region: can the lite tier translate it?
bool LiteTranslates(uint32_t insn) {
  static uint32_t code[1];
  code[0] = insn;
  GuestAddr start = ToGuestAddr(code);
  GuestAddr end = start + 4;
  MachineCode mc;
  auto [ok, stop] = TryLiteTranslateRegion(
      start, &mc, LiteTranslateParams{.end_pc = end, .allow_dispatch = false});
  return ok && stop == end;
}

// Single-instruction region: can the heavy tier translate it WHOLE? A partial
// or bailed region falls back to lite on-device, so it does not count.
bool HeavyTranslates(uint32_t insn) {
  static uint32_t code[1];
  code[0] = insn;
  GuestAddr start = ToGuestAddr(code);
  GuestAddr end = start + 4;
  MachineCode mc;
  auto [stop, ok, num] = HeavyOptimizeRegion(start, &mc, HeavyOptimizeParams{.end_pc = end});
  return ok && stop == end && num == 1;
}

std::map<std::string, Counts> MeasureCoverage() {
  std::map<std::string, Counts> out;
  std::ifstream f(DataPath(kCorpusFile));
  if (!f.good()) return out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string enc_hex, mnemonic;
    if (!(ss >> enc_hex >> mnemonic)) continue;
    uint32_t enc = static_cast<uint32_t>(std::strtoul(enc_hex.c_str(), nullptr, 16));
    Counts& c = out[mnemonic];
    c.total++;
    if (LiteTranslates(enc)) c.lite++;
    if (HeavyTranslates(enc)) c.heavy++;
  }
  return out;
}

std::map<std::string, Counts> LoadTable() {
  std::map<std::string, Counts> out;
  std::ifstream f(DataPath(kTableFile));
  if (!f.good()) return out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string mnemonic;
    Counts c;
    if (!(ss >> mnemonic >> c.total >> c.lite >> c.heavy)) continue;
    out[mnemonic] = c;
  }
  return out;
}

TEST(Arm64TierCoverage, NoTierLosesCoverage) {
  auto measured = MeasureCoverage();
  ASSERT_FALSE(measured.empty()) << "corpus not found next to the test binary";
  auto table = LoadTable();
  ASSERT_FALSE(table.empty()) << "coverage table not found next to the test binary";

  std::vector<std::string> regressions;
  int improved = 0;
  for (const auto& [mnemonic, want] : table) {
    auto it = measured.find(mnemonic);
    if (it == measured.end()) {
      // The corpus no longer contains this mnemonic: it was regenerated from a
      // different library set. Not a translator regression.
      fprintf(stderr, "Arm64TierCoverage: '%s' absent from corpus\n", mnemonic.c_str());
      continue;
    }
    const Counts& got = it->second;
    char buf[256];
    if (got.lite < want.lite) {
      snprintf(buf, sizeof(buf), "%s: lite translates %d/%d, table records %d", mnemonic.c_str(),
               got.lite, got.total, want.lite);
      regressions.push_back(buf);
    }
    if (got.heavy < want.heavy) {
      snprintf(buf, sizeof(buf), "%s: heavy translates %d/%d, table records %d", mnemonic.c_str(),
               got.heavy, got.total, want.heavy);
      regressions.push_back(buf);
    }
    if (got.lite > want.lite || got.heavy > want.heavy) improved++;
  }

  if (improved > 0) {
    fprintf(stderr,
            "Arm64TierCoverage: %d mnemonic(s) now translate more than the table records"
            " -- regenerate %s\n",
            improved,
            kTableFile);
  }

  std::string detail;
  for (const auto& r : regressions) detail += "\n  " + r;
  EXPECT_TRUE(regressions.empty()) << "tier coverage regressed:" << detail;
}

// Writes a fresh table to stdout when asked. Keeping the generator in the test
// binary means the table can never be produced by a different notion of
// "translates" than the assertion uses.
TEST(Arm64TierCoverage, DISABLED_RegenerateTable) {
  auto measured = MeasureCoverage();
  ASSERT_FALSE(measured.empty());
  printf("# Per-mnemonic tier coverage over arm64_decoder_encodings.txt.\n");
  printf("# Columns: <mnemonic> <encodings> <lite-translates> <heavy-translates>\n");
  printf("# Regenerate:\n");
  printf("#   berberis_arm64_host_tests \\\n");
  printf("#     --gtest_also_run_disabled_tests \\\n");
  printf("#     --gtest_filter=Arm64TierCoverage.DISABLED_RegenerateTable\n");
  printf("#\n");
  printf("# A tier translating FEWER encodings than recorded here fails\n");
  printf("# Arm64TierCoverage.NoTierLosesCoverage. Translating more is reported, not\n");
  printf("# failed -- regenerate this file in the commit that adds the coverage.\n");
  for (const auto& [mnemonic, c] : measured) {
    printf("%s %d %d %d\n", mnemonic.c_str(), c.total, c.lite, c.heavy);
  }
}

}  // namespace

}  // namespace berberis
