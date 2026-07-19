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

#include "berberis/proxy_loader/proxy_library_builder.h"

#include <dlfcn.h>
// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
#include <sys/mman.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#endif
// endregion

#include <cstring>

#include "berberis/base/checks.h"
#include "berberis/base/logging.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/host_function_wrapper_impl.h"

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
// Full CPUState definition — DoGracefulBadTrampoline writes the guest return
// register (x0) directly. arm64-guarded so the guest-agnostic proxy loader
// (riscv64/arm) keeps only the opaque forward declaration.
#include "berberis/guest_state/guest_state.h"
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
// endregion

namespace berberis {

void DoBadThunk() {
  LOG_ALWAYS_FATAL("Bad thunk call before %p", __builtin_return_address(0));
}

void DoBadTrampoline(HostCode callee, ThreadState* state) {
  CHECK(state);
  const char* name = static_cast<const char*>(callee);
  LOG_ALWAYS_FATAL("Bad '%s' call from %p",
                   name ? name : "[unknown name]",
                   ToHostAddr<void>(GetLinkRegister(GetCPUState(*state))));
}

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
// LOUD, non-fatal replacement for DoBadTrampoline used ONLY on the arm64 guest.
// A DoBadTrampoline symbol that is actually reached — which should never happen
// for the unreachable framework-internal bulk, since every NDK-stable bad symbol
// is covered or contract-stubbed (machine-audited by the Digitalis project's
// bad-symbol enumerator) — degrades to a greppable warning and a
// zeroed integer return instead of a SIGABRT. riscv64/arm keep the fatal
// DoBadTrampoline above (the upstream bug-detector). This is a last-resort
// backstop, not a fix: it names the symbol and returns 0; a symbol whose caller
// dereferences a returned pointer may still fault downstream. The guarantee it
// backs is "no NDK API call aborts".
void DoGracefulBadTrampoline(HostCode callee, ThreadState* state) {
  CHECK(state);
  const char* name = static_cast<const char*>(callee);
  TRACE("berberis: BAD-TRAMPOLINE '%s' called from %p — returning 0 (graceful, arm64)",
        name ? name : "[unknown name]",
        ToHostAddr<void>(GetLinkRegister(GetCPUState(*state))));
  GetCPUState(*state).x[0] = 0;
}
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
// endregion

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
namespace {

struct ExtraRegistry {
  const char* library_name;
  const KnownTrampoline* trampolines;
  size_t count;
};

// One slot per RegisterExtraTrampolines call (one per proxied library, and a
// file may register several — e.g. libGLESv2.so + libGLESv3.so). This MUST stay
// safely above the number of REGISTER_DIGITALIS_EXTRA_TRAMPOLINES sites under
// android_api/digitalis_extra_proxy/; overflow is silently dropped below, and a
// dropped registration un-covers that library's symbols (they fall through to
// DoGracefulBadTrampoline and return 0), corrupting any app that calls them —
// tier-independently, and non-deterministically by constructor init order.
// There are currently 9 registration sites; keep generous headroom.
constexpr size_t kMaxExtraRegistries = 32;
ExtraRegistry g_extra_registries[kMaxExtraRegistries];
size_t g_num_extra_registries = 0;

// Linear search the extra-trampoline registry for a symbol name in a library.
const KnownTrampoline* FindExtraTrampoline(const char* library_name, const char* name) {
  for (size_t i = 0; i < g_num_extra_registries; ++i) {
    const auto& reg = g_extra_registries[i];
    if (strcmp(reg.library_name, library_name) != 0) {
      continue;
    }
    for (size_t j = 0; j < reg.count; ++j) {
      if (strcmp(reg.trampolines[j].name, name) == 0) {
        return &reg.trampolines[j];
      }
    }
  }
  return nullptr;
}

// Overrides of symbols whose primary entry has a WORKING trampoline (see
// RegisterExtraTrampolineOverrides in the header). Kept separate from
// g_extra_registries so plain extras can never accidentally shadow a healthy
// upstream trampoline.
ExtraRegistry g_override_registries[kMaxExtraRegistries];
size_t g_num_override_registries = 0;

const KnownTrampoline* FindExtraTrampolineOverride(const char* library_name, const char* name) {
  for (size_t i = 0; i < g_num_override_registries; ++i) {
    const auto& reg = g_override_registries[i];
    if (strcmp(reg.library_name, library_name) != 0) {
      continue;
    }
    for (size_t j = 0; j < reg.count; ++j) {
      if (strcmp(reg.trampolines[j].name, name) == 0) {
        return &reg.trampolines[j];
      }
    }
  }
  return nullptr;
}

// Storage for the {primary marshal, primary thunk} chains handed to override
// trampolines as their callee. One slot per overridden symbol per proxy-library
// load; proxy libraries load once per process, so a small fixed pool suffices.
// Slots are never freed — the installed trampoline references them for the
// process lifetime.
constexpr size_t kMaxChainedOverrides = 16;
ChainedTrampoline g_chained_overrides[kMaxChainedOverrides];
size_t g_num_chained_overrides = 0;

ChainedTrampoline* AllocChainedOverride(TrampolineFunc marshal_and_call, void* thunk) {
  // Reuse an identical chain: a symbol can be re-intercepted once per guest
  // linker namespace (Chromium creates many), always with the same resolved
  // primary, and must not consume a fresh slot each time.
  for (size_t i = 0; i < g_num_chained_overrides; ++i) {
    if (g_chained_overrides[i].marshal_and_call == marshal_and_call &&
        g_chained_overrides[i].thunk == thunk) {
      return &g_chained_overrides[i];
    }
  }
  if (g_num_chained_overrides >= kMaxChainedOverrides) {
    TRACE("ProxyLibraryBuilder: chained-override pool full (%zu); override dropped",
          g_num_chained_overrides);
    return nullptr;
  }
  ChainedTrampoline* chain = &g_chained_overrides[g_num_chained_overrides++];
  chain->marshal_and_call = marshal_and_call;
  chain->thunk = thunk;
  return chain;
}

// Current protection (PROT_* mask) of the mapping containing `addr`, or -1 if
// unknown. Parses /proc/self/maps; only called on the cold variable-interception
// path, so the scan cost is irrelevant.
int GuestPageProt(const void* addr) {
  FILE* f = fopen("/proc/self/maps", "re");
  if (f == nullptr) {
    return -1;
  }
  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  char line[512];
  int prot = -1;
  while (fgets(line, sizeof(line), f) != nullptr) {
    uintptr_t start = 0, end = 0;
    char r = '-', w = '-', x = '-', p = '-';
    if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR " %c%c%c%c", &start, &end, &r, &w, &x, &p) == 6 &&
        a >= start && a < end) {
      prot = 0;
      if (r == 'r') prot |= PROT_READ;
      if (w == 'w') prot |= PROT_WRITE;
      if (x == 'x') prot |= PROT_EXEC;
      break;
    }
  }
  fclose(f);
  return prot;
}

// Store `size` bytes from `src` into the guest destination `dst`. A proxied
// variable's guest slot (an imported-variable GOT entry in .data.rel.ro) can
// already be GNU-RELRO read-only by the time this interception runs — the guest
// linker mprotects it after relocation — so a plain store faults SEGV_ACCERR
// (observed crashing Baidu Maps sub-processes at libmediandk/libc variable
// slots; /proc/self/mem FOLL_FORCE writes are also refused on the relro page).
// If the destination page is currently non-writable, temporarily restore write
// access, store, and re-apply the EXACT original protection — never leave a
// pre-relro writable page read-only, which would break the guest linker's later
// relocations into the same page. Reached only under InterceptGuestSymbol's mutex.
void StoreGuestVariable(void* dst, const void* src, size_t size) {
  int prot = GuestPageProt(dst);
  if (prot >= 0 && !(prot & PROT_WRITE)) {
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    uintptr_t a = reinterpret_cast<uintptr_t>(dst);
    void* page = reinterpret_cast<void*>(a & ~(page_size - 1));
    size_t span = (a + size) - reinterpret_cast<uintptr_t>(page);
    if (mprotect(page, span, prot | PROT_WRITE) == 0) {
      memcpy(dst, src, size);
      mprotect(page, span, prot);
      return;
    }
  }
  memcpy(dst, src, size);
}

}  // namespace

void ProxyLibraryBuilder::RegisterExtraTrampolines(const char* library_name,
                                                   const KnownTrampoline* trampolines,
                                                   size_t count) {
  if (g_num_extra_registries >= kMaxExtraRegistries) {
    TRACE("ProxyLibraryBuilder: extra-trampoline registry full (%zu/%zu); dropping registration "
          "for \"%s\" (%zu trampolines)",
          g_num_extra_registries,
          kMaxExtraRegistries,
          library_name,
          count);
    return;
  }
  g_extra_registries[g_num_extra_registries++] = {library_name, trampolines, count};
}

void ProxyLibraryBuilder::RegisterExtraTrampolineOverrides(const char* library_name,
                                                           const KnownTrampoline* trampolines,
                                                           size_t count) {
  if (g_num_override_registries >= kMaxExtraRegistries) {
    TRACE("ProxyLibraryBuilder: override registry full (%zu/%zu); dropping registration "
          "for \"%s\" (%zu trampolines)",
          g_num_override_registries,
          kMaxExtraRegistries,
          library_name,
          count);
    return;
  }
  g_override_registries[g_num_override_registries++] = {library_name, trampolines, count};
}
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
// endregion

void ProxyLibraryBuilder::InterceptSymbol(GuestAddr guest_addr, const char* name) {
  CHECK(guest_addr);

  // TODO(b/287342829): functions_ are sorted, use binary search!
  for (size_t i = 0; i < num_functions_; ++i) {
    const auto& function = functions_[i];
    if (strcmp(name, function.name) == 0) {
      void* thunk = function.thunk;
      if (!thunk) {
        // Default thunk.
        thunk = dlsym(handle_, name);
      }
      if (!thunk) {
        // Assume no thunk needed, all work is done by trampoline.
        thunk = reinterpret_cast<void*>(DoBadThunk);
      }
      if (function.marshal_and_call == DoBadTrampoline) {
        // region digitalis - a primary-table entry marked DoBadTrampoline
        // (incompatible signature, no upstream custom trampoline) can be
        // overridden by a Digitalis-side extra trampoline registered for this
        // library via RegisterExtraTrampolines. Consult the extras registry
        // before falling back to the fatal DoBadTrampoline. arm64-only: the
        // extras registry exists solely in the arm64-translation proxy loader,
        // so this override is byte-identical-absent for riscv64/arm builds.
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
        if (const KnownTrampoline* extra = FindExtraTrampoline(library_name_, name);
            extra != nullptr && extra->marshal_and_call != DoBadTrampoline) {
          void* extra_thunk = extra->thunk;
          if (!extra_thunk) {
            extra_thunk = dlsym(handle_, name);
          }
          if (!extra_thunk) {
            extra_thunk = reinterpret_cast<void*>(DoBadThunk);
          }
          MakeTrampolineCallable(guest_addr, false, extra->marshal_and_call, extra_thunk, name);
          return;
        }
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
        // endregion
        // HACK: DoBadTrampoline needs function name passed as callee!
        // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
        MakeTrampolineCallable(guest_addr, false, DoGracefulBadTrampoline, name, name);
#else
        // endregion
        MakeTrampolineCallable(guest_addr, false, DoBadTrampoline, name, name);
        // region digitalis
#endif
        // endregion
      } else {
        // region digitalis - a Digitalis-side override registered via
        // RegisterExtraTrampolineOverrides replaces this healthy primary
        // trampoline while keeping it callable: the override runs with a
        // ChainedTrampoline callee holding the primary's resolved
        // {marshal_and_call, thunk}, so it can execute the upstream behavior
        // first and only post-process guest state (e.g. libEGL's
        // eglGetProcAddress adds wrapping for ANGLE extension procs the
        // upstream table cannot marshal). arm64-only, like the extras registry.
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
        if (const KnownTrampoline* override_entry =
                FindExtraTrampolineOverride(library_name_, name);
            override_entry != nullptr) {
          if (ChainedTrampoline* chain =
                  AllocChainedOverride(function.marshal_and_call, thunk);
              chain != nullptr) {
            MakeTrampolineCallable(
                guest_addr, false, override_entry->marshal_and_call, chain, name);
            return;
          }
        }
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
        // endregion
        MakeTrampolineCallable(guest_addr, false, function.marshal_and_call, thunk, name);
      }
      return;
    }
  }

  // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  // Search Digitalis-side extra trampolines registered for this library.
  // Same dispatch shape as the primary loop, factored above to share between
  // primary and extras. Only present in the arm64-translation build of the
  // proxy loader (libberberis_proxy_loader_arm64); the upstream guest-agnostic
  // libberberis_proxy_loader omits this block entirely.
  if (const KnownTrampoline* extra = FindExtraTrampoline(library_name_, name); extra != nullptr) {
    void* thunk = extra->thunk;
    if (!thunk) {
      thunk = dlsym(handle_, name);
    }
    if (!thunk) {
      thunk = reinterpret_cast<void*>(DoBadThunk);
    }
    if (extra->marshal_and_call == DoBadTrampoline) {
      // Already inside the arm64-only block: a bad extra entry degrades loudly.
      MakeTrampolineCallable(guest_addr, false, DoGracefulBadTrampoline, name, name);
    } else {
      MakeTrampolineCallable(guest_addr, false, extra->marshal_and_call, thunk, name);
    }
    return;
  }
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
  // endregion

  // TODO(b/287342829): variables_ are sorted, use binary search!
  for (size_t i = 0; i < num_variables_; ++i) {
    const auto& variable = variables_[i];
    if (strcmp(name, variable.name) == 0) {
      if (variable.size != sizeof(GuestAddr)) {
        // TODO(b/287342829): at the moment, all intercepted variables are assumed to be pointers!
        TRACE("proxy library \"%s\": size mismatch for variable \"%s\"", library_name_, name);
      }
      void* addr = dlsym(handle_, name);
      if (!addr) {
        TRACE("proxy library \"%s\": symbol for variable \"%s\" is NULL", library_name_, name);
      } else {
        // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
        // The guest variable slot may already be GNU-RELRO read-only by now; a
        // plain store faults SEGV_ACCERR. Write via /proc/self/mem to bypass page
        // write protection. See StoreGuestVariable.
        StoreGuestVariable(ToHostAddr<void>(guest_addr), addr, sizeof(GuestAddr));
#else
        // endregion
        // TODO(b/287342829): copy variable.size bytes instead!
        memcpy(ToHostAddr<void>(guest_addr), addr, sizeof(GuestAddr));
        // region digitalis
#endif
        // endregion
      }
      return;
    }
  }

  TRACE("proxy library \"%s\": symbol \"%s\" not found", library_name_, name);
}

void ProxyLibraryBuilder::Build(const char* library_name,
                                size_t size_translation,
                                const KnownTrampoline* translations,
                                size_t size_data_symbols,
                                const KnownVariable* variables) {
  handle_ = dlopen(library_name, RTLD_GLOBAL);
  if (!handle_) {
    LOG_ALWAYS_FATAL("dlopen failed: %s: %s", library_name, dlerror());
  }

  library_name_ = library_name;
  num_functions_ = size_translation;
  functions_ = translations;
  num_variables_ = size_data_symbols;
  variables_ = variables;
}

}  // namespace berberis
