/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_guest_exec_region.h"
#include "berberis/test_utils/translation_test.h"

#include "translator_x86_64.h"

namespace berberis {

namespace {

class TranslatorRiscv64ToX86_64 : public TranslationTest {};

TEST_F(TranslatorRiscv64ToX86_64, LiteTranslateSupportedRegion) {
  static const uint32_t code[] = {
      0x0020'81b3,  // add x3, x1, x2
      0x0080'00ef,  // jal x1, 8
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

TEST_F(TranslatorRiscv64ToX86_64, LiteTranslateUnsupportedRegion) {
  static const uint32_t code[] = {
      0x0000'0073,  // ecall #0x0
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] =
      TryLiteTranslateAndInstallRegion(ToGuestAddr(code));

  EXPECT_FALSE(success);
}

TEST_F(TranslatorRiscv64ToX86_64, LiteTranslatePartiallySupportedRegion) {
  static const uint32_t code[] = {
      0x0020'81b3,  // add x3, x1, x2
      0x0000'0073,  // ecall #0x0
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

TEST_F(TranslatorRiscv64ToX86_64, HeavyOptimizeSupportedRegion) {
  static const uint32_t code[] = {
      0x0080'00ef,  // jal x1, 8
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] = HeavyOptimizeRegion(ToGuestAddr(code));

  EXPECT_TRUE(success);
  EXPECT_NE(host_code_piece.code, kEntryInterpret);
  EXPECT_GT(host_code_piece.size, 0U);
  EXPECT_EQ(guest_size, 4U);
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kHeavyOptimized);
}

TEST_F(TranslatorRiscv64ToX86_64, HeavyOptimizeUnsupportedRegion) {
  static const uint32_t code[] = {
      0x0000'100f,  // fence.i
  };
  ScopedGuestExecRegion exec_region(ToGuestAddr(code), sizeof(code));

  auto [success, host_code_piece, guest_size, kind] = HeavyOptimizeRegion(ToGuestAddr(code));

  EXPECT_FALSE(success);
  EXPECT_NE(host_code_piece.code, kEntryInterpret);
  EXPECT_EQ(host_code_piece.size, 0U);
  EXPECT_EQ(guest_size, 0U);
  EXPECT_EQ(kind, GuestCodeEntry::Kind::kInterpreted);
}

}  // namespace

}  // namespace berberis
