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

#include "berberis/runtime_primitives/runtime_library.h"

#include <cstdint>
#include <cstring>

#include "berberis/calling_conventions/calling_conventions_arm64.h"
#include "berberis/guest_abi/guest_arguments.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_os_primitives/scoped_pending_signals.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_arch.h"
#include "berberis/instrument/guest_call.h"
#include "berberis/runtime_primitives/virtual_guest_call_frame.h"

namespace berberis {

// Call a guest function from the host with the given arguments.
// ARM64 AAPCS64: x0-x7 for integer args, v0-v7 for FP/SIMD args.
// SP is a separate register in CPUState (not part of x[]).
void RunGuestCall(GuestAddr pc, GuestArgumentBuffer* buf) {
  GuestThread* thread = GetCurrentGuestThread();
  ThreadState* state = thread->state();

  ScopedPendingSignalsEnabler scoped_pending_signals_enabler(thread);

  ScopedVirtualGuestCallFrame virtual_guest_call_frame(&state->cpu, pc);

  // Copy argc int registers for the arguments into the cpu state.
  // ARM64 uses x0-x7 for integer args.
  memcpy(&(state->cpu.x[0]), buf->argv, buf->argc * sizeof(buf->argv[0]));

  // Copy simd_argc SIMD registers for the arguments into the cpu state.
  // ARM64 uses v0-v7 for FP/SIMD args.
  memcpy(&(state->cpu.v[0]), buf->simd_argv, buf->simd_argc * sizeof(buf->simd_argv[0]));

  // sp -= stack_argc
  state->cpu.sp -= buf->stack_argc;
  // sp = align_down(sp, ...)
  state->cpu.sp =
      AlignDown(state->cpu.sp, arm64::CallingConventions::kStackAlignmentBeforeCall);

  memcpy(ToHostAddr<void>(state->cpu.sp), buf->stack_argv, buf->stack_argc);

  if (kInstrumentWrappers) {
    OnWrappedGuestCall(state, pc);
  }

  ExecuteGuestCall(state);

  if (kInstrumentWrappers) {
    OnWrappedGuestReturn(state, pc);
  }

  // Copy result int and SIMD registers from the cpu state back.
  memcpy(buf->argv, &(state->cpu.x[0]), buf->resc * sizeof(buf->argv[0]));
  memcpy(buf->simd_argv, &(state->cpu.v[0]), buf->simd_resc * sizeof(buf->simd_argv[0]));
}

}  // namespace berberis
