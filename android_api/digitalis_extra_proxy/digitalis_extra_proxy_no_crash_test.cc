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

// Ground-truth checks for the "no Bad '<sym>' call abort" guarantee:
//   - the loud arm64 net DoGracefulBadTrampoline zeroes x0 and returns (no abort);
//   - each of the three contract stubs returns its API's "unavailable" value.
// All exercised by hand-building a ThreadState, setting the relevant guest
// registers, calling the handler directly, and inspecting the result — no guest
// execution and no device library needed.

#include "gtest/gtest.h"

#include <cstdint>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/proxy_loader/proxy_library_builder.h"
#include "berberis/runtime_primitives/host_code.h"

#include "digitalis_extra_stubs.h"

namespace berberis {
namespace {

TEST(Arm64ProxyNoCrash, GracefulBadTrampolineZeroesX0AndReturns) {
  ThreadState state{};
  GetCPUState(state).x[0] = 0xdeadbeefdeadbeefULL;
  // callee carries the symbol name string, exactly as DoBadTrampoline expects.
  DoGracefulBadTrampoline(reinterpret_cast<HostCode>("SomeUnreachableSym"), &state);
  EXPECT_EQ(GetCPUState(state).x[0], 0u);  // returned 0, did not abort
}

TEST(Arm64ProxyNoCrash, AIBinderToPlatformBinderReturnsNullSp) {
  ThreadState state{};
  uint64_t sret_buf = 0x1111111122222222ULL;  // pre-filled garbage
  GetCPUState(state).x[8] = ToGuestAddr(&sret_buf);                          // indirect result loc
  GetCPUState(state).x[0] = ToGuestAddr(reinterpret_cast<void*>(0xABCDULL));  // fake AIBinder*
  DoStub_AIBinder_toPlatformBinder(static_cast<HostCode>(nullptr), &state);
  EXPECT_EQ(sret_buf, 0u);                                        // null sp written into sret buffer
  EXPECT_EQ(GetCPUState(state).x[0], GetCPUState(state).x[8]);    // x0 == sret ptr per AAPCS64
}

TEST(Arm64ProxyNoCrash, GlGetVkProcAddrNvReturnsNull) {
  ThreadState state{};
  GetCPUState(state).x[0] = ToGuestAddr(reinterpret_cast<void*>(0xF00DULL));  // fake name ptr
  DoStub_glGetVkProcAddrNV(static_cast<HostCode>(nullptr), &state);
  EXPECT_EQ(GetCPUState(state).x[0], 0u);  // returned NULL proc addr
}

TEST(Arm64ProxyNoCrash, SetPerformInterceptorIsNoOpNoAbort) {
  ThreadState state{};
  GetCPUState(state).x[0] = ToGuestAddr(reinterpret_cast<void*>(0xAAULL));  // fake window
  GetCPUState(state).x[1] = ToGuestAddr(reinterpret_cast<void*>(0xBBULL));  // fake interceptor
  GetCPUState(state).x[2] = 0;
  // Must simply return without aborting; void function, no output asserted.
  DoStub_ANativeWindow_setPerformInterceptor(static_cast<HostCode>(nullptr), &state);
  SUCCEED();
}

}  // namespace
}  // namespace berberis
