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

// IC IVAU must drop the cached translation for the modified code.
//
// ARM64 has no flush_icache syscall: a user-space JIT that writes new code
// issues `ic ivau` per cache line and then executes it. A binary translator
// that ignores this keeps running the translation of the code that USED to be
// at that address. The failure is not a crash but stale behaviour, and it is
// address-reuse dependent, which makes it very hard to find from the symptom --
// it presented as a wedged Qt fill loop.
//
// Both JIT tiers deliberately decline this instruction (lite sets success_ =
// false, the heavy frontend bails), so the invalidation is performed by the
// interpreter and the contract lives there. These tests drive it through
// InterpretInsn exactly as a region would on-device.
//
// The negative case matters as much as the positive one: invalidating the whole
// cache would satisfy the positive test while destroying performance, so an
// unrelated translation must survive.

#include "gtest/gtest.h"

#include <cstdint>
#include <tuple>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/arm64/interpreter.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_guest_exec_region.h"
#include "berberis/test_utils/translation_test.h"

namespace berberis {

// Defined in runtime/arm64/translator_x86_64.cc with external linkage but no
// companion header; declared here exactly as translator_x86_64_test.cc does.
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> TryLiteTranslateAndInstallRegion(
    GuestAddr pc,
    LiteTranslateParams params = LiteTranslateParams());

namespace {

// `ic ivau, x<rt>` == 0xD50B7520 | rt.
constexpr uint32_t kIcIvauBase = 0xD50B7520;

// A region the lite translator fully handles, so installing it succeeds.
// add x3, x1, x2 ; b .+4 (the branch ends the region).
alignas(64) const uint32_t kRegionA[] = {0x8b020023, 0x14000001};
alignas(64) const uint32_t kRegionB[] = {0x8b020023, 0x14000001};

class Arm64IcIvauInvalidation : public TranslationTest {
 protected:
  // The translation cache is process-wide and these regions are static, so a
  // test that leaves an entry installed would make the NEXT test's install fail
  // (AddAndLockForTranslation only transitions out of NotTranslated). Start
  // every test from a clean slate rather than relying on the previous test's
  // IC IVAU to have done the cleanup -- otherwise a broken IC IVAU shows up as
  // a cascade of unrelated setup failures instead of one clear one.
  void SetUp() override {
    TranslationTest::SetUp();
    TranslationCache* cache = TranslationCache::GetInstance();
    cache->InvalidateGuestRange(ToGuestAddr(kRegionA),
                                ToGuestAddr(kRegionA) + sizeof(kRegionA));
    cache->InvalidateGuestRange(ToGuestAddr(kRegionB),
                                ToGuestAddr(kRegionB) + sizeof(kRegionB));
  }

  static bool IsTranslated(GuestAddr pc) {
    return TranslationCache::GetInstance()->GetHostCodePtr(pc)->load() != kEntryNotTranslated;
  }

  // Translates the region at `pc` and registers it in the translation cache.
  // TryLiteTranslateAndInstallRegion only emits the host code and places it in
  // the code pool; publishing the entry is the caller's job on-device
  // (TranslateRegion does it), and without it there is nothing for
  // InvalidateGuestRange to find.
  static bool InstallRegion(GuestAddr pc) {
    auto [success, host_code_piece, guest_size, kind] = TryLiteTranslateAndInstallRegion(pc);
    if (!success) return false;
    TranslationCache* cache = TranslationCache::GetInstance();
    GuestCodeEntry* entry = cache->AddAndLockForTranslation(pc, /*counter_threshold=*/0);
    if (entry == nullptr) return false;
    cache->SetTranslatedAndUnlock(
        pc, entry, static_cast<uint32_t>(guest_size), kind, host_code_piece);
    return true;
  }

  // Runs a single `ic ivau, x<rt>` through the interpreter with x<rt> = addr.
  void RunIcIvau(GuestAddr addr, uint8_t rt = 5) {
    ThreadState state{};
    static uint32_t insn;
    insn = kIcIvauBase | rt;
    state.cpu.x[rt] = addr;
    state.cpu.insn_addr = ToGuestAddr(&insn);
    InterpretInsn(&state);
  }
};

TEST_F(Arm64IcIvauInvalidation, EncodingMatchesArmArm) {
  // `ic ivau, x0` and `ic ivau, x5` as emitted by the assembler.
  EXPECT_EQ(kIcIvauBase | 0u, 0xD50B7520u);
  EXPECT_EQ(kIcIvauBase | 5u, 0xD50B7525u);
}

TEST_F(Arm64IcIvauInvalidation, InvalidatesTranslationAtTheAddress) {
  GuestAddr pc = ToGuestAddr(kRegionA);
  ScopedGuestExecRegion exec_region(pc, sizeof(kRegionA));

  ASSERT_TRUE(InstallRegion(pc));
  ASSERT_TRUE(IsTranslated(pc)) << "region did not install, test would be vacuous";

  RunIcIvau(pc);

  EXPECT_FALSE(IsTranslated(pc)) << "IC IVAU left a stale translation at the modified address";
}

// The guest passes any address within the line, not necessarily the region
// start, because it walks its buffer in cache-line steps.
TEST_F(Arm64IcIvauInvalidation, InvalidatesFromAnyAddressInTheSameCacheLine) {
  GuestAddr pc = ToGuestAddr(kRegionA);
  ScopedGuestExecRegion exec_region(pc, sizeof(kRegionA));

  ASSERT_TRUE(InstallRegion(pc));
  ASSERT_TRUE(IsTranslated(pc));

  // kRegionA is 64-byte aligned, so +4 is inside the same line.
  RunIcIvau(pc + 4);

  EXPECT_FALSE(IsTranslated(pc)) << "IC IVAU only honoured the exact region start address";
}

// Negative control. Invalidating everything would pass the tests above while
// throwing away every translation on the device, so a translation in an
// unrelated cache line must survive.
TEST_F(Arm64IcIvauInvalidation, LeavesOtherCacheLinesAlone) {
  GuestAddr pc_a = ToGuestAddr(kRegionA);
  GuestAddr pc_b = ToGuestAddr(kRegionB);
  ASSERT_NE(pc_a & ~static_cast<GuestAddr>(63), pc_b & ~static_cast<GuestAddr>(63))
      << "regions share a cache line; the control proves nothing";

  ScopedGuestExecRegion exec_a(pc_a, sizeof(kRegionA));
  ScopedGuestExecRegion exec_b(pc_b, sizeof(kRegionB));

  ASSERT_TRUE(InstallRegion(pc_a));
  ASSERT_TRUE(InstallRegion(pc_b));
  ASSERT_TRUE(IsTranslated(pc_a));
  ASSERT_TRUE(IsTranslated(pc_b));

  RunIcIvau(pc_a);

  EXPECT_FALSE(IsTranslated(pc_a));
  EXPECT_TRUE(IsTranslated(pc_b)) << "IC IVAU invalidated an unrelated cache line";
}

// XZR is not an address. `ic ivau, xzr` must not invalidate anything.
TEST_F(Arm64IcIvauInvalidation, XzrOperandIsIgnored) {
  GuestAddr pc = ToGuestAddr(kRegionA);
  ScopedGuestExecRegion exec_region(pc, sizeof(kRegionA));

  ASSERT_TRUE(InstallRegion(pc));
  ASSERT_TRUE(IsTranslated(pc));

  RunIcIvau(pc, /*rt=*/31);

  EXPECT_TRUE(IsTranslated(pc)) << "ic ivau, xzr invalidated a translation";
}

}  // namespace

}  // namespace berberis
