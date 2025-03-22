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

#include <chrono>  // chrono_literals::operator""ms
#include <initializer_list>
#include <string>
#include <thread>  // this_thread::sleep_for

#include "berberis/base/config.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/runtime_library.h"  // kEntry*
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {

namespace {

using std::chrono_literals::operator""ms;
// A test guest pc that is valid in both 32bit and 64bit modes.
constexpr GuestAddr kGuestPC = 0x12345678;

TEST(TranslationCacheTest, DefaultNotTranslated) {
  TranslationCache tc;

  EXPECT_EQ(tc.GetHostCodePtr(kGuestPC)->load(), kEntryNotTranslated);
  EXPECT_EQ(tc.GetHostCodePtr(kGuestPC + 1024)->load(), kEntryNotTranslated);
  EXPECT_EQ(tc.GetInvocationCounter(kGuestPC), 0U);
}

TEST(TranslationCacheTest, UpdateInvocationCounter) {
  TranslationCache tc;

  // Create entry
  GuestCodeEntry* entry = tc.AddAndLockForTranslation(kGuestPC, 0);
  ASSERT_TRUE(entry);
  EXPECT_EQ(entry->invocation_counter, 0U);
  entry->invocation_counter = 42;
  tc.SetTranslatedAndUnlock(
      kGuestPC, entry, 1, GuestCodeEntry::Kind::kSpecialHandler, {kEntryNoExec, 0});

  EXPECT_EQ(tc.GetInvocationCounter(kGuestPC), 42U);
}

TEST(TranslationCacheTest, AddAndLockForTranslation) {
  TranslationCache tc;

  // Cannot lock if counter is below the threshold, but entry is created anyway.
  ASSERT_FALSE(tc.AddAndLockForTranslation(kGuestPC, 1));
  GuestCodeEntry* entry = tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC);
  ASSERT_TRUE(entry);
  EXPECT_EQ(tc.GetHostCodePtr(kGuestPC)->load(), kEntryNotTranslated);
  EXPECT_EQ(entry->kind, GuestCodeEntry::Kind::kInterpreted);
  EXPECT_EQ(tc.GetInvocationCounter(kGuestPC), 1U);

  // Lock when counter is equal or above the threshold.
  entry = tc.AddAndLockForTranslation(kGuestPC, 1);
  ASSERT_TRUE(entry);
  EXPECT_EQ(tc.GetHostCodePtr(kGuestPC)->load(), kEntryTranslating);
  EXPECT_EQ(entry->kind, GuestCodeEntry::Kind::kUnderProcessing);
  EXPECT_EQ(tc.GetInvocationCounter(kGuestPC), 1U);

  // Cannot lock locked.
  ASSERT_FALSE(tc.AddAndLockForTranslation(kGuestPC, 0));

  // Unlock.
  tc.SetTranslatedAndUnlock(
      kGuestPC, entry, 1, GuestCodeEntry::Kind::kSpecialHandler, {kEntryNoExec, 0});
  EXPECT_EQ(tc.GetHostCodePtr(kGuestPC)->load(), kEntryNoExec);
  EXPECT_EQ(entry->kind, GuestCodeEntry::Kind::kSpecialHandler);

  // Cannot lock translated.
  ASSERT_FALSE(tc.AddAndLockForTranslation(kGuestPC, 0));
}

constexpr bool kWrappedHostFunc = true;

TEST(TranslationCacheTest, AddAndLockForWrapping) {
  TranslationCache tc;

  // Add and lock nonexistent.
  GuestCodeEntry* entry = tc.AddAndLockForWrapping(kGuestPC);
  ASSERT_TRUE(entry);
  ASSERT_EQ(kEntryWrapping, tc.GetHostCodePtr(kGuestPC)->load());
  ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kUnderProcessing);

  // Cannot lock locked.
  ASSERT_FALSE(tc.AddAndLockForWrapping(kGuestPC));

  // Unlock.
  tc.SetWrappedAndUnlock(kGuestPC, entry, kWrappedHostFunc, {kEntryNoExec, 0});
  ASSERT_EQ(kEntryNoExec, tc.GetHostCodePtr(kGuestPC)->load());
  ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kHostWrapped);

  // Cannot lock wrapped.
  ASSERT_FALSE(tc.AddAndLockForWrapping(kGuestPC));

  // Cannot lock not translated but already interpreted.
  ASSERT_FALSE(tc.AddAndLockForTranslation(kGuestPC + 64, 1));
  entry = tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC + 64);
  ASSERT_TRUE(entry);
  ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kInterpreted);
  ASSERT_FALSE(tc.AddAndLockForWrapping(kGuestPC + 64));
}

HostCodeAddr kHostCodeStub = AsHostCodeAddr(AsHostCode(0xdeadbeef));

void TestWrappingWorker(TranslationCache* tc, GuestAddr pc) {
  while (true) {
    GuestCodeEntry* entry = tc->AddAndLockForWrapping(pc);
    if (entry) {
      EXPECT_EQ(entry->host_code->load(), kEntryWrapping);
      EXPECT_EQ(entry->kind, GuestCodeEntry::Kind::kUnderProcessing);
      // Give other threads some time to run the loop. Typical test
      // time is 1 ms, make sleep order of magnitude longer - 10ms.
      std::this_thread::sleep_for(10ms);
      tc->SetWrappedAndUnlock(pc, entry, kWrappedHostFunc, {kHostCodeStub, 0});
      EXPECT_EQ(entry->host_code->load(), kHostCodeStub);
      EXPECT_EQ(entry->kind, GuestCodeEntry::Kind::kHostWrapped);
      return;
    }

    auto host_code = tc->GetHostCodePtr(pc)->load();

    // Warning: the order of comparisons here is
    // important since code can change in between.
    if (host_code == kEntryWrapping) {
      continue;
    }

    EXPECT_EQ(host_code, kHostCodeStub);
    break;
  }

  return;
}

void TestTranslationWorker(TranslationCache* tc, GuestAddr pc) {
  while (true) {
    GuestCodeEntry* entry = tc->AddAndLockForTranslation(pc, 0);
    if (entry) {
      EXPECT_EQ(entry->host_code->load(), kEntryTranslating);
      EXPECT_EQ(entry->kind, GuestCodeEntry::Kind::kUnderProcessing);
      // Give other threads some time to run the loop. Typical test
      // time is 1 ms, make sleep order of magnitude longer - 10ms.
      std::this_thread::sleep_for(10ms);
      tc->SetTranslatedAndUnlock(
          pc, entry, 1, GuestCodeEntry::Kind::kSpecialHandler, {kHostCodeStub, 0});
      EXPECT_EQ(entry->host_code->load(), kHostCodeStub);
      return;
    }

    auto host_code = tc->GetHostCodePtr(pc)->load();
    if (host_code == kEntryTranslating) {
      continue;
    }
    EXPECT_EQ(host_code, kHostCodeStub);
    break;
  }

  return;
}

template <void(WorkerFunc)(TranslationCache*, GuestAddr)>
void TranslationCacheTestRunThreads() {
  TranslationCache tc;
  constexpr uint32_t kNumThreads = 16;
  std::array<std::thread, kNumThreads> threads;

  // First test situation, when every thread has its own pc.
  for (uint32_t i = 0; i < kNumThreads; i++) {
    threads[i] = std::thread(WorkerFunc, &tc, i);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Now introduce heavy contention.
  for (auto& thread : threads) {
    thread = std::thread(WorkerFunc, &tc, kGuestPC);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(TranslationCacheTest, InvalidateNotTranslated) {
  TranslationCache tc;

  ASSERT_EQ(kEntryNotTranslated, tc.GetHostCodePtr(kGuestPC)->load());

  tc.InvalidateGuestRange(kGuestPC, kGuestPC + 1);

  // Not translated stays not translated
  ASSERT_EQ(kEntryNotTranslated, tc.GetHostCodePtr(kGuestPC)->load());
  ASSERT_FALSE(tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC));
}

TEST(TranslationCacheTest, InvalidateTranslated) {
  TranslationCache tc;

  GuestCodeEntry* entry = tc.AddAndLockForTranslation(kGuestPC, 0);
  ASSERT_TRUE(entry);
  ASSERT_EQ(kEntryTranslating, tc.GetHostCodePtr(kGuestPC)->load());

  tc.SetTranslatedAndUnlock(
      kGuestPC, entry, 1, GuestCodeEntry::Kind::kHeavyOptimized, {kHostCodeStub, 4});
  ASSERT_EQ(kHostCodeStub, tc.GetHostCodePtr(kGuestPC)->load());

  tc.InvalidateGuestRange(kGuestPC, kGuestPC + 1);

  ASSERT_EQ(kEntryNotTranslated, tc.GetHostCodePtr(kGuestPC)->load());
  ASSERT_FALSE(tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC));
}

TEST(TranslationCacheTest, InvalidateTranslating) {
  TranslationCache tc;

  GuestCodeEntry* entry = tc.AddAndLockForTranslation(kGuestPC, 0);
  ASSERT_TRUE(entry);
  ASSERT_EQ(kEntryTranslating, tc.GetHostCodePtr(kGuestPC)->load());

  tc.InvalidateGuestRange(kGuestPC, kGuestPC + 1);
  ASSERT_EQ(kEntryInvalidating, tc.GetHostCodePtr(kGuestPC)->load());
  entry = tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC);
  ASSERT_TRUE(entry);
  ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kUnderProcessing);

  tc.SetTranslatedAndUnlock(
      kGuestPC, entry, 1, GuestCodeEntry::Kind::kSpecialHandler, {kHostCodeStub, 4});
  ASSERT_EQ(kEntryNotTranslated, tc.GetHostCodePtr(kGuestPC)->load());
  ASSERT_FALSE(tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC));
}

TEST(TranslationCacheTest, InvalidateTranslatingOutOfRange) {
  TranslationCache tc;

  GuestCodeEntry* entry = tc.AddAndLockForTranslation(kGuestPC, 0);
  ASSERT_TRUE(entry);
  ASSERT_EQ(kEntryTranslating, tc.GetHostCodePtr(kGuestPC)->load());

  // Invalidate range that does *not* contain translating address.
  // The entry should still be invalidated, as translated region is only known after translation,
  // and it might overlap with the invalidated range.
  tc.InvalidateGuestRange(kGuestPC + 100, kGuestPC + 101);
  ASSERT_EQ(kEntryInvalidating, tc.GetHostCodePtr(kGuestPC)->load());

  tc.SetTranslatedAndUnlock(
      kGuestPC, entry, 1, GuestCodeEntry::Kind::kSpecialHandler, {kHostCodeStub, 4});
  ASSERT_EQ(kEntryNotTranslated, tc.GetHostCodePtr(kGuestPC)->load());
}

bool Translate(TranslationCache* tc, GuestAddr pc, uint32_t size, HostCodeAddr host_code) {
  GuestCodeEntry* entry = tc->AddAndLockForTranslation(pc, 0);
  if (!entry) {
    return false;
  }
  tc->SetTranslatedAndUnlock(
      pc, entry, size, GuestCodeEntry::Kind::kSpecialHandler, {host_code, 4});
  return true;
}

TEST(TranslationCacheTest, LockForGearUpTranslation) {
  TranslationCache tc;

  // Cannot lock if not yet added.
  ASSERT_FALSE(tc.LockForGearUpTranslation(kGuestPC));

  ASSERT_TRUE(Translate(&tc, kGuestPC + 0, 1, kHostCodeStub));
  GuestCodeEntry* entry = tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC);
  ASSERT_TRUE(entry);
  ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kSpecialHandler);

  // Cannot lock if kind is not kLiteTranslated.
  ASSERT_FALSE(tc.LockForGearUpTranslation(kGuestPC));

  entry->kind = GuestCodeEntry::Kind::kLiteTranslated;

  entry = tc.LockForGearUpTranslation(kGuestPC);
  ASSERT_TRUE(entry);
  ASSERT_EQ(kEntryTranslating, tc.GetHostCodePtr(kGuestPC)->load());
  ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kUnderProcessing);

  // Unlock.
  tc.SetTranslatedAndUnlock(
      kGuestPC, entry, 1, GuestCodeEntry::Kind::kHeavyOptimized, {kEntryNoExec, 0});
  ASSERT_EQ(kEntryNoExec, tc.GetHostCodePtr(kGuestPC)->load());
  ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kHeavyOptimized);

  // Cannot lock translated.
  ASSERT_FALSE(tc.AddAndLockForTranslation(kGuestPC, 0));
}

TEST(TranslationCacheTest, InvalidateRange) {
  TranslationCache tc;

  ASSERT_TRUE(Translate(&tc, kGuestPC + 0, 1, kHostCodeStub));
  ASSERT_TRUE(Translate(&tc, kGuestPC + 1, 1, kHostCodeStub));
  ASSERT_TRUE(Translate(&tc, kGuestPC + 2, 1, kHostCodeStub));

  ASSERT_EQ(kHostCodeStub, tc.GetHostCodePtr(kGuestPC + 0)->load());
  ASSERT_EQ(kHostCodeStub, tc.GetHostCodePtr(kGuestPC + 1)->load());
  ASSERT_EQ(kHostCodeStub, tc.GetHostCodePtr(kGuestPC + 2)->load());

  tc.InvalidateGuestRange(kGuestPC + 1, kGuestPC + 2);

  ASSERT_EQ(kHostCodeStub, tc.GetHostCodePtr(kGuestPC + 0)->load());
  ASSERT_EQ(kEntryNotTranslated, tc.GetHostCodePtr(kGuestPC + 1)->load());
  ASSERT_EQ(kHostCodeStub, tc.GetHostCodePtr(kGuestPC + 2)->load());
}

bool Wrap(TranslationCache* tc, GuestAddr pc, HostCodeAddr host_code) {
  GuestCodeEntry* entry = tc->AddAndLockForWrapping(pc);
  if (!entry) {
    return false;
  }
  tc->SetWrappedAndUnlock(pc, entry, kWrappedHostFunc, {host_code, 0});
  return true;
}

TEST(TranslationCacheTest, InvalidateWrapped) {
  TranslationCache tc;

  ASSERT_TRUE(Wrap(&tc, kGuestPC, kEntryNoExec));

  tc.InvalidateGuestRange(kGuestPC, kGuestPC + 1);

  ASSERT_EQ(kEntryNotTranslated, tc.GetHostCodePtr(kGuestPC)->load());
}

TEST(TranslationCacheTest, InvalidateWrappingWrap) {
  TranslationCache tc;

  GuestCodeEntry* entry = tc.AddAndLockForWrapping(kGuestPC);
  ASSERT_TRUE(entry);

  tc.InvalidateGuestRange(kGuestPC, kGuestPC + 1);
  ASSERT_EQ(kEntryInvalidating, tc.GetHostCodePtr(kGuestPC)->load());

  tc.SetWrappedAndUnlock(kGuestPC, entry, kWrappedHostFunc, {kEntryNoExec, 0});
  ASSERT_EQ(kEntryNotTranslated, tc.GetHostCodePtr(kGuestPC)->load());

  ASSERT_TRUE(Wrap(&tc, kGuestPC, kEntryNoExec));
}

TEST(TranslationCacheTest, WrapInvalidateWrap) {
  TranslationCache tc;

  ASSERT_TRUE(Wrap(&tc, kGuestPC, kEntryNoExec));

  tc.InvalidateGuestRange(kGuestPC, kGuestPC + 1);

  ASSERT_TRUE(Wrap(&tc, kGuestPC, kEntryNoExec));
}

TEST(TranslationCacheTest, WrapInvalidateTranslate) {
  TranslationCache tc;

  ASSERT_TRUE(Wrap(&tc, kGuestPC, kEntryNoExec));

  tc.InvalidateGuestRange(kGuestPC, kGuestPC + 1);

  ASSERT_TRUE(Translate(&tc, kGuestPC, 1, kEntryNoExec));
}

TEST(TranslationCacheTest, WrappingStatesTest) {
  TranslationCacheTestRunThreads<TestWrappingWorker>();
}

TEST(TranslationCacheTest, TranslationStatesTest) {
  TranslationCacheTestRunThreads<TestTranslationWorker>();
}

constexpr size_t kGuestGearShiftRange = 64;

void TestTriggerGearShiftForAddresses(
    GuestAddr pc,
    std::initializer_list<std::tuple<GuestAddr, uint32_t>> addr_and_expected_counter_list) {
  TranslationCache tc;
  // Lite translate interesting addresses.
  for (auto [pc, unused_counter] : addr_and_expected_counter_list) {
    ASSERT_TRUE(Translate(&tc, pc, 1, kHostCodeStub));
    GuestCodeEntry* entry = tc.LookupGuestCodeEntryUnsafeForTesting(pc);
    ASSERT_TRUE(entry);
    ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kSpecialHandler);
    ASSERT_EQ(entry->invocation_counter, 0u);
    entry->kind = GuestCodeEntry::Kind::kLiteTranslated;
  }

  tc.TriggerGearShift(pc, kGuestGearShiftRange);

  for (auto [pc, expected_counter] : addr_and_expected_counter_list) {
    ASSERT_EQ(tc.LookupGuestCodeEntryUnsafeForTesting(pc)->invocation_counter, expected_counter)
        << "pc=" << pc;
  }
}

TEST(TranslationCacheTest, TriggerGearShift) {
  TestTriggerGearShiftForAddresses(kGuestPC,
                                   {{kGuestPC, config::kGearSwitchThreshold},
                                    {kGuestPC - kGuestGearShiftRange, config::kGearSwitchThreshold},
                                    {kGuestPC - kGuestGearShiftRange - 1, 0},
                                    {kGuestPC + kGuestGearShiftRange, config::kGearSwitchThreshold},
                                    {kGuestPC + kGuestGearShiftRange + 1, 0}});
}

TEST(TranslationCacheTest, TriggerGearShiftTargetLessThanRange) {
  constexpr GuestAddr kSmallGuestPC = kGuestGearShiftRange / 2;
  TestTriggerGearShiftForAddresses(
      kSmallGuestPC,
      {{kSmallGuestPC, config::kGearSwitchThreshold},
       {kNullGuestAddr, config::kGearSwitchThreshold},
       {kSmallGuestPC + kGuestGearShiftRange, config::kGearSwitchThreshold}});
}

TEST(TranslationCacheTest, TriggerGearShiftDoesNotAffectNotLiteTranslated) {
  TranslationCache tc;
  ASSERT_TRUE(Translate(&tc, kGuestPC, 1, kHostCodeStub));
  GuestCodeEntry* entry = tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC);
  ASSERT_TRUE(entry);
  ASSERT_EQ(entry->kind, GuestCodeEntry::Kind::kSpecialHandler);
  ASSERT_EQ(entry->invocation_counter, 0u);

  tc.TriggerGearShift(kGuestPC, kGuestGearShiftRange);

  ASSERT_EQ(tc.LookupGuestCodeEntryUnsafeForTesting(kGuestPC)->invocation_counter, 0u);
}

}  // namespace

}  // namespace berberis
