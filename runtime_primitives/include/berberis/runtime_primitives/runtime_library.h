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

#ifndef BERBERIS_RUNTIME_PRIMITIVES_RUNTIME_LIBRARY_H_
#define BERBERIS_RUNTIME_PRIMITIVES_RUNTIME_LIBRARY_H_

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

extern "C" {

void berberis_RunGeneratedCode(ThreadState* state, HostCode code);
void berberis_entry_Interpret();
void berberis_entry_ExitGeneratedCode();
void berberis_entry_Stop();
void berberis_entry_NoExec();
void berberis_entry_HandleLiteCounterThresholdReached();

// TODO(b/232598137): use status variable instead?
void berberis_entry_NotTranslated();
void berberis_entry_Translating();
void berberis_entry_Invalidating();
void berberis_entry_Wrapping();

static_assert(berberis_entry_NotTranslated != berberis_entry_Translating,
              "code to distinguish entry status got optimized");

__attribute__((__visibility__("hidden"))) void berberis_HandleNoExec(ThreadState* state);

}  // extern "C"

// These constants are initialized by InitHostEntries()
extern HostCodeAddr kEntryInterpret;
extern HostCodeAddr kEntryExitGeneratedCode;
extern HostCodeAddr kEntryStop;
extern HostCodeAddr kEntryNoExec;
extern HostCodeAddr kEntryNotTranslated;
extern HostCodeAddr kEntryTranslating;
extern HostCodeAddr kEntryInvalidating;
extern HostCodeAddr kEntryWrapping;

void InitHostEntries();

void InvalidateGuestRange(GuestAddr start, GuestAddr end);

// Don't pull in the dependency on guest_abi to runtime_primitives, since GuestArgumentBuffer is
// used strictly in opaque manner here.
struct GuestArgumentBuffer;

void RunGuestCall(GuestAddr pc, GuestArgumentBuffer* buf);
void ExecuteGuestCall(ThreadState* state);

// region digitalis
// Optional hook invoked by berberis_HandleNoExec before it raises SIGSEGV. A
// higher layer that can resolve guest symbols may register a handler which
// services the fault — e.g. redirecting a guest call that landed in a host
// system library (because a hardened library resolved a host symbol from
// /proc/self/maps and called it directly, bypassing the loader) to the guest's
// own translatable counterpart of that library — and return true to resume the
// guest. Returning false, or registering no hook, delivers the default SIGSEGV.
using HandleNoExecHook = bool (*)(ThreadState* state);
void SetHandleNoExecHook(HandleNoExecHook hook);
// endregion

}  // namespace berberis

#endif  // BERBERIS_RUNTIME_PRIMITIVES_RUNTIME_LIBRARY_H_
