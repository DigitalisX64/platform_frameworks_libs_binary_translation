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

#include "berberis/runtime/execute_guest.h"

#include <cstdint>

#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime/berberis.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {

namespace {

TEST(Arm64ExecuteGuest, Basic) {
  const uint32_t code[] = {
      0x8b030041,  // add x1, x2, x3
      0x9ac42021,  // lsl x1, x1, x4  (LSLV — variable shift)
      0x14000001,  // b .+4  (exit the region to stop_pc)
  };

  InitBerberis();

  GuestMapShadow::GetInstance()->SetExecutable(ToGuestAddr(&code[0]), sizeof(code));

  GuestThread* thread = GetCurrentGuestThread();
  auto& cpu_state = thread->state()->cpu;
  cpu_state.insn_addr = ToGuestAddr(&code[0]);
  cpu_state.x[2] = 10;
  cpu_state.x[3] = 11;
  cpu_state.x[4] = 1;
  GuestAddr stop_pc = ToGuestAddr(&code[0]) + 12;
  auto cache = TranslationCache::GetInstance();
  cache->SetStop(stop_pc);
  ExecuteGuest(thread->state());
  cache->TestingClearStop(stop_pc);
  // x1 = (x2 + x3) << x4 = (10 + 11) << 1 = 42.
  EXPECT_EQ(cpu_state.x[1], 42u);

  GuestMapShadow::GetInstance()->ClearExecutable(ToGuestAddr(&code[0]), sizeof(code));
}

}  // namespace

}  // namespace berberis
