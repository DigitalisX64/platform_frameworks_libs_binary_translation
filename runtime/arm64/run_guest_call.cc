/*
 * Copyright (C) 2019 The Android Open Source Project
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

#include "berberis/runtime_library/runtime_library.h"

#include <cstdint>
#include <cstring>

#include "berberis/base/bit_util.h"
#include "berberis/calling_conventions/calling_conventions_arm64.h"
#include "berberis/guest_abi/guest_arguments_arch.h"
#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"  // GetCurrentGuestThread
#include "berberis/guest_os_primitives/scoped_pending_signals.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/instrument/guest_call.h"
#include "berberis/runtime_primitives/virtual_guest_call_frame.h"

namespace berberis {

void RunGuestCall(GuestAddr pc, GuestArgumentBuffer* buf) {
  GuestThread* thread = GetCurrentGuestThread();
  ThreadState* state = thread->state();

  ScopedPendingSignalsEnabler scoped_pending_signals_enabler(thread);

  ScopedVirtualGuestCallFrame virtual_guest_call_frame(&state->cpu, pc);

  memcpy(state->cpu.x, buf->argv, buf->argc * sizeof(buf->argv[0]));
  memcpy(state->cpu.v, buf->simd_argv, buf->simd_argc * sizeof(buf->simd_argv[0]));

  state->cpu.sp -= buf->stack_argc;
  state->cpu.sp = AlignDown(state->cpu.sp, arm64::CallingConventions::kStackAlignmentBeforeCall);

  memcpy(ToHostAddr<void>(state->cpu.sp), buf->stack_argv, buf->stack_argc);

  if (kInstrumentWrappers) {
    OnWrappedGuestCall(state, pc);
  }

  ExecuteGuestCall(state);

  if (kInstrumentWrappers) {
    OnWrappedGuestReturn(state, pc);
  }

  memcpy(buf->argv, state->cpu.x, buf->resc * sizeof(buf->argv[0]));
  memcpy(buf->simd_argv, state->cpu.v, buf->simd_resc * sizeof(buf->simd_argv[0]));
}

}  // namespace berberis