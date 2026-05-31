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

#include "berberis/runtime_primitives/interpret_helpers.h"

#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>

#include "berberis/base/checks.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

void UndefinedInsn(GuestAddr pc) {
  auto* addr = ToHostAddr<const uint32_t>(pc);
  // ARM64 instructions are always 4 bytes.
  uint32_t code;
  memcpy(&code, addr, sizeof(code));
  TRACE_AND_ALOGE("Undefined arm64 instruction 0x%" PRIx32 " at %p", code, addr);
#ifdef __GLIBC__
  // Our old 2.17 glibc has a bug resulting in raise() failing after any CLONE_VM.
  // Work around by calling tgkill directly.
  pid_t pid = syscall(__NR_getpid);
  pid_t tid = syscall(__NR_gettid);
  syscall(SYS_tgkill, pid, tid, SIGILL);
#else
  raise(SIGILL);
#endif
}

void BreakpointInsn(GuestAddr pc) {
  // BRK #imm delivers a synchronous SIGTRAP. Mirror UndefinedInsn's self-tgkill,
  // but with SIGTRAP: the Berberis host signal handler catches it and routes it
  // to the guest's SIGTRAP action with the guest PC at the BRK instruction, so a
  // guest debugger or sanitizer trap-handler can read the breakpoint immediate
  // from the faulting instruction. With no guest handler installed the default
  // SIGTRAP action terminates — the architecturally-correct outcome for BRK
  // (and strictly better than the previous illegal-instruction SIGILL).
  auto* addr = ToHostAddr<const uint32_t>(pc);
  uint32_t code;
  memcpy(&code, addr, sizeof(code));
  TRACE("BRK arm64 instruction 0x%" PRIx32 " at %p -> SIGTRAP", code, addr);
#ifdef __GLIBC__
  pid_t pid = syscall(__NR_getpid);
  pid_t tid = syscall(__NR_gettid);
  syscall(SYS_tgkill, pid, tid, SIGTRAP);
#else
  raise(SIGTRAP);
#endif
}

}  // namespace berberis
