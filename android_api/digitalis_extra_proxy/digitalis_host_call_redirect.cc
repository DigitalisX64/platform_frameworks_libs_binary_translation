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

// Redirect a guest call that landed in a host system library to the guest's own
// translatable copy of that library.
//
// Hardened third-party libraries resolve a system-library symbol by parsing
// /proc/self/maps + the ELF export table and call it directly, bypassing the
// guest loader/PLT (an anti-hook technique). Under translation the symbol they
// find is the HOST x86_64 implementation (present in the process because, e.g.,
// the in-process system WebView pulled libsqlite.so in), so the guest `blr`
// lands in non-guest-executable host code and traps in berberis_HandleNoExec.
// Alibaba SecurityGuard's libsgmainso (shipped in AliExpress) does exactly this,
// using the host libsqlite.so as a real SQLite engine (sqlite3_libversion_number,
// sqlite3_open_v2, ...).
//
// Berberis ships guest (arm64) copies of these system libraries
// (/system/lib64/arm64/...). When the faulting host PC is exactly an exported
// symbol of a host system library that has such a guest counterpart, load the
// guest copy and resolve the same symbol in it, then redirect the guest PC to
// the guest function — leaving all argument and link registers untouched. The
// guest function then runs under translation like any other guest code, so its
// callbacks stay guest->guest and no per-function ABI marshalling is needed.
// Bounded so it cannot mask a wild branch: the fault address must be exactly an
// exported symbol entry of a library under a system path that also has a guest
// copy resolvable by the guest loader.

#include <dlfcn.h>

#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "berberis/base/tracing.h"
#include "berberis/guest_loader/guest_loader.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/runtime_library.h"

namespace berberis {

namespace {

#if defined(__x86_64__)

bool IsHostSystemLibPath(const char* path) {
  return std::strncmp(path, "/system/", 8) == 0 || std::strncmp(path, "/apex/", 6) == 0 ||
         std::strncmp(path, "/vendor/", 8) == 0 || std::strncmp(path, "/product/", 9) == 0;
}

const char* Basename(const char* path) {
  const char* slash = std::strrchr(path, '/');
  return slash != nullptr ? slash + 1 : path;
}

// Resolve `symbol` in the guest copy of the library named `soname`, loading it
// once and caching the handle and every looked-up symbol. Returns 0 if the
// guest copy or the symbol is unavailable.
GuestAddr ResolveGuestEquivalent(const char* soname, const char* symbol) {
  static std::mutex mutex;
  static std::map<std::string, void*> handles;
  static std::map<std::string, GuestAddr> symbols;
  std::lock_guard<std::mutex> lock(mutex);

  std::string sym_key = std::string(soname) + '|' + symbol;
  auto cached = symbols.find(sym_key);
  if (cached != symbols.end()) {
    return cached->second;
  }

  auto* loader = GuestLoader::GetInstance();
  if (loader == nullptr) {
    return 0;
  }

  void* handle;
  auto handle_it = handles.find(soname);
  if (handle_it != handles.end()) {
    handle = handle_it->second;
  } else {
    handle = loader->DlOpen(soname, RTLD_NOW | RTLD_LOCAL);
    handles[soname] = handle;
  }

  GuestAddr addr = (handle != nullptr) ? loader->DlSym(handle, symbol) : 0;
  symbols[sym_key] = addr;
  return addr;
}

bool RedirectHostSystemLibCallToGuest(ThreadState* state) {
  CPUState& cpu = GetCPUState(*state);
  void* host_pc = ToHostAddr<void>(GetInsnAddr(cpu));
  Dl_info info;
  if (dladdr(host_pc, &info) == 0 || info.dli_sname == nullptr || info.dli_fname == nullptr ||
      info.dli_saddr != host_pc) {
    return false;
  }
  if (!IsHostSystemLibPath(info.dli_fname)) {
    return false;
  }
  // ResolveGuestEquivalent runs guest linker code on THIS thread's CPUState the
  // first (uncached) time a library/symbol is resolved: DlOpen relocates the
  // guest copy and runs its constructors, DlSym walks its symbol table. That
  // nested guest call is a normal AAPCS64 call, so it is free to clobber the
  // caller-saved registers x0-x18 and v0-v7/v16-v31 — and here those registers
  // still hold the *arguments* of the redirected function, because the guest is
  // mid-`br <target>` (a tail branch), not at a call-return boundary.
  // ScopedVirtualGuestCallFrame only preserves sp/x29/x30/pc, so without this
  // snapshot the redirected function would execute with the linker's leftover
  // garbage in its argument registers. Observed: Kuaishou's libAemonPlayer
  // reaches libgui Surface::hook_query with x0=0 / x2=0x10 and faults writing
  // through the bogus out-parameter — flaky (~17% of launches) exactly because
  // only the first, uncached resolution runs guest code; cached resolutions
  // touch no registers. Snapshot the full guest register file and restore it
  // after resolution so the redirect is transparent: only the PC is rerouted.
  const CPUState saved_cpu = cpu;
  GuestAddr guest_pc = ResolveGuestEquivalent(Basename(info.dli_fname), info.dli_sname);
  cpu = saved_cpu;
  if (guest_pc == 0) {
    return false;
  }
  TRACE("Redirecting guest call into host %s '%s' to guest %s",
        info.dli_fname,
        info.dli_sname,
        Basename(info.dli_fname));
  SetInsnAddr(cpu, guest_pc);
  return true;
}

__attribute__((constructor(102))) void RegisterHostCallRedirect() {
  SetHandleNoExecHook(RedirectHostSystemLibCallToGuest);
}

#endif  // defined(__x86_64__)

}  // namespace

}  // namespace berberis
