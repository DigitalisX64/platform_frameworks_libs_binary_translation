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

#ifndef BERBERIS_LITE_TRANSLATOR_LITE_TRANSLATE_REGION_H_
#define BERBERIS_LITE_TRANSLATOR_LITE_TRANSLATE_REGION_H_

#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/config.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_library/runtime_library.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

struct LiteTranslateParams {
  GuestAddr end_pc = GetGuestAddrRangeEnd();
  bool allow_dispatch = true;
  bool enable_reg_mapping = true;
  bool enable_self_profiling = false;
  // Let a branch back to the region's start loop inside the region, with a
  // pending-signals check per iteration, instead of exiting through dispatch.
  // Translation takes extra passes, so only regions known to be hot and
  // staying in the lite tier set it. Ignored with enable_self_profiling: the
  // hotness counter at the region start must count every iteration.
  bool allow_in_region_loops = false;
  uint32_t* counter_location = nullptr;
  uint32_t counter_threshold = config::kGearSwitchThreshold;
  HostCode counter_threshold_callback =
      AsHostCode(berberis::berberis_entry_HandleLiteCounterThresholdReached);
};

std::tuple<bool, GuestAddr> TryLiteTranslateRegion(GuestAddr start_pc,
                                                   MachineCode* machine_code,
                                                   LiteTranslateParams params);

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_LITE_TRANSLATE_REGION_H_
