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

#include "berberis/digitalis_extra_proxy/named_trampoline_override.h"

namespace berberis {

namespace {

// The registry is append-only process-global state with no reset API (it is
// only ever fed by library constructors), so every test here uses names of its
// own and the fixture registers one table once.

void UpstreamTrampolineA(HostCode, ThreadState*) {}
void UpstreamTrampolineB(HostCode, ThreadState*) {}
void ReplacementTrampolineA(HostCode, ThreadState*) {}
void ReplacementTrampolineB(HostCode, ThreadState*) {}

const NamedTrampolineOverride kTestOverrides[] = {
    {"NamedTrampolineOverrideTest::A", ReplacementTrampolineA},
    {"NamedTrampolineOverrideTest::B", ReplacementTrampolineB},
};

class NamedTrampolineOverrideTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    RegisterDigitalisNamedTrampolineOverrides(
        kTestOverrides, sizeof(kTestOverrides) / sizeof(kTestOverrides[0]));
  }
};

TEST_F(NamedTrampolineOverrideTest, UnknownNamePassesThrough) {
  EXPECT_EQ(nullptr,
            FindDigitalisNamedTrampolineOverride("NamedTrampolineOverrideTest::Unknown",
                                                 UpstreamTrampolineA));
  EXPECT_EQ(nullptr, GetDigitalisNamedTrampolineOriginal("NamedTrampolineOverrideTest::Unknown"));
}

TEST_F(NamedTrampolineOverrideTest, MatchInstallsOverrideAndRecordsOriginal) {
  EXPECT_EQ(ReplacementTrampolineA,
            FindDigitalisNamedTrampolineOverride("NamedTrampolineOverrideTest::A",
                                                 UpstreamTrampolineA));
  EXPECT_EQ(UpstreamTrampolineA,
            GetDigitalisNamedTrampolineOriginal("NamedTrampolineOverrideTest::A"));

  // A re-wrap of the same name (a second object exposing the same vtable)
  // records the same original again and keeps returning the override.
  EXPECT_EQ(ReplacementTrampolineA,
            FindDigitalisNamedTrampolineOverride("NamedTrampolineOverrideTest::A",
                                                 UpstreamTrampolineA));
  EXPECT_EQ(UpstreamTrampolineA,
            GetDigitalisNamedTrampolineOriginal("NamedTrampolineOverrideTest::A"));
}

TEST_F(NamedTrampolineOverrideTest, SelfWrapNeverBecomesItsOwnOriginal) {
  // An override file that (directly or by replaying an upstream registration
  // helper) wraps its own overridden name presents the override itself as the
  // incoming trampoline. Recording that would make delegation to the original
  // recurse forever, so the registry must refuse it — both before any real
  // original exists and after one was recorded.
  EXPECT_EQ(ReplacementTrampolineB,
            FindDigitalisNamedTrampolineOverride("NamedTrampolineOverrideTest::B",
                                                 ReplacementTrampolineB));
  EXPECT_EQ(nullptr, GetDigitalisNamedTrampolineOriginal("NamedTrampolineOverrideTest::B"));

  EXPECT_EQ(ReplacementTrampolineB,
            FindDigitalisNamedTrampolineOverride("NamedTrampolineOverrideTest::B",
                                                 UpstreamTrampolineB));
  EXPECT_EQ(UpstreamTrampolineB,
            GetDigitalisNamedTrampolineOriginal("NamedTrampolineOverrideTest::B"));

  EXPECT_EQ(ReplacementTrampolineB,
            FindDigitalisNamedTrampolineOverride("NamedTrampolineOverrideTest::B",
                                                 ReplacementTrampolineB));
  EXPECT_EQ(UpstreamTrampolineB,
            GetDigitalisNamedTrampolineOriginal("NamedTrampolineOverrideTest::B"));
}

}  // namespace

}  // namespace berberis
