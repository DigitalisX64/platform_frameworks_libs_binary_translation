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

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_guest_exec_region.h"
#include "berberis/test_utils/translation_test.h"

namespace berberis {

// runtime/arm64/translator_x86_64.cc defines these in namespace berberis with
// external linkage ("Exported for testing only") but has no companion header
// (unlike runtime/riscv64/translator_x86_64.h). Forward-declare them here with
// the exact signatures so the linker resolves them against libberberis_runtime_arm64.
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> TryLiteTranslateAndInstallRegion(
    GuestAddr pc,
    LiteTranslateParams params = LiteTranslateParams());
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> HeavyOptimizeAndInstallRegion(
    GuestAddr pc);

namespace {

class Arm64RuntimeTranslator : public TranslationTest {};

// A region the lite translator fully handles installs a kLiteTranslated entry.
TEST_F(Arm64RuntimeTranslator, LiteTranslateSupportedRegion) {
  static const uint32_t code[] = {
      0x8b020023,  // add x3, x1, x2
      0x14000001,  // b .+4  (unconditional branch ends the region)
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      TryLiteTranslateAndInstallRegion(ToGuestAddr(code));

  EXPECT_TRUE(success);
  EXPECT_NE(host_code_piece.code, kNullHostCodeAddr);
  EXPECT_GT(host_code_piece.size, 0U);
  EXPECT_EQ(guest_size, 8U);
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kLiteTranslated);
}

// A region whose very first instruction the lite translator cannot handle (BRK
// is delivered by the interpreter, which raises the synchronous SIGTRAP) fails
// to translate at all — nothing to install. Mirrors the riscv64 `ecall` case.
TEST_F(Arm64RuntimeTranslator, LiteTranslateUnsupportedRegion) {
  static const uint32_t code[] = {
      0xd4200000,  // brk #0
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      TryLiteTranslateAndInstallRegion(ToGuestAddr(code));

  EXPECT_FALSE(success);
}

// A supported prefix followed by an unsupported instruction installs the prefix
// only: the region is truncated at the bail point (here, after the single ADD).
TEST_F(Arm64RuntimeTranslator, LiteTranslatePartiallySupportedRegion) {
  static const uint32_t code[] = {
      0x8b020023,  // add x3, x1, x2
      0xd4200000,  // brk #0  (interpreter-only, bails the lite translator)
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      TryLiteTranslateAndInstallRegion(ToGuestAddr(code));

  EXPECT_TRUE(success);
  EXPECT_NE(host_code_piece.code, kNullHostCodeAddr);
  EXPECT_GT(host_code_piece.size, 0U);
  EXPECT_EQ(guest_size, 4U);
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kLiteTranslated);
}

// The heavy optimizer bails on BRK exactly like the lite tier; with no prefix
// translated and no in-region back-edge, the install wrapper reports failure so
// the runtime re-lite-translates (and ultimately interprets) the region.
TEST_F(Arm64RuntimeTranslator, HeavyOptimizeUnsupportedRegion) {
  static const uint32_t code[] = {
      0xd4200000,  // brk #0
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      HeavyOptimizeAndInstallRegion(ToGuestAddr(code));

  EXPECT_FALSE(success);
  EXPECT_EQ(guest_size, 0U);
}

// A region the heavy optimizer fully translates but that is smaller than the
// gear-up threshold (GetGearUpMinInsns, default 20) is declined: gearing up a
// tiny region cannot recoup the optimizing tier's codegen overhead, so the
// wrapper returns failure and the runtime keeps the lite region. This exercises
// runtime/arm64/translator_x86_64.cc's gear-up policy, which riscv64 lacks.
TEST_F(Arm64RuntimeTranslator, HeavyOptimizeDeclinesSmallSupportedRegion) {
  static const uint32_t code[] = {
      0xd2800220,  // movz x0, #0x11
      0xd2800441,  // movz x1, #0x22
      0xd65f03c0,  // ret  (indirect branch: definitively ends the region)
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      HeavyOptimizeAndInstallRegion(ToGuestAddr(code));

  // Fully heavy-translatable, but 3 < 20 instructions -> gear-up declined.
  EXPECT_FALSE(success);
}

// A region above the gear-up threshold that the heavy optimizer fully translates
// installs a kHeavyOptimized entry.
TEST_F(Arm64RuntimeTranslator, HeavyOptimizeLargeSupportedRegion) {
  static const uint32_t code[] = {
      0x91000400, 0x91000400, 0x91000400, 0x91000400, 0x91000400,  // add x0, x0, #1
      0x91000400, 0x91000400, 0x91000400, 0x91000400, 0x91000400,  // (x24)
      0x91000400, 0x91000400, 0x91000400, 0x91000400, 0x91000400,
      0x91000400, 0x91000400, 0x91000400, 0x91000400, 0x91000400,
      0x91000400, 0x91000400, 0x91000400, 0x91000400,
      0xd65f03c0,  // ret  (indirect branch: definitively ends the region)
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      HeavyOptimizeAndInstallRegion(ToGuestAddr(code));

  EXPECT_TRUE(success);
  EXPECT_NE(host_code_piece.code, kNullHostCodeAddr);
  EXPECT_GT(host_code_piece.size, 0U);
  EXPECT_EQ(guest_size, sizeof(code));
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kHeavyOptimized);
}

}  // namespace

}  // namespace berberis
