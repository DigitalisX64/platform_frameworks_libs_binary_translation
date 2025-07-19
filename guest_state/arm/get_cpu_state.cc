/*
 * Copyright (C) 2024 The Android Open Source Project
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

#include "berberis/guest_state/get_cpu_state_opaque.h"
#include "native_bridge_support/guest_state_accessor/accessor.h"

#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_state_arch.h"

#include <cstddef>
#include <cstring>

namespace berberis {

int GetCpuState(NativeBridgeGuestRegs* guest_regs, const CPUState* state) {
  if (guest_regs->guest_arch != NATIVE_BRIDGE_ARCH_ARM) {
    TRACE_AND_ALOGE("The guest architecture is unmatched: %llu", guest_regs->guest_arch);
    return NATIVE_BRIDGE_GUEST_STATE_ACCESSOR_ERROR_UNSUPPORTED_ARCH;
  }
  memcpy(&guest_regs->regs_arm.r, &state->r, sizeof(state->r));
  guest_regs->regs_arm.r[15] = state->insn_addr;
  memcpy(&guest_regs->regs_arm.q, &state->d, sizeof(state->d));
  return 0;
}

extern "C" __attribute__((visibility("default"))) int LoadGuestStateRegisters(
    const void* guest_state_data,
    size_t guest_state_data_size,
    NativeBridgeGuestRegs* guest_regs) {
  if (guest_state_data_size < sizeof(ThreadState)) {
    TRACE_AND_ALOGE("The guest state data size is invalid: %zu", guest_state_data_size);
    return NATIVE_BRIDGE_GUEST_STATE_ACCESSOR_ERROR_INVALID_STATE;
  }
  guest_regs->guest_arch = NATIVE_BRIDGE_ARCH_ARM;
  return GetCpuState(guest_regs, &(static_cast<const ThreadState*>(guest_state_data))->cpu);
}

}  // namespace berberis
