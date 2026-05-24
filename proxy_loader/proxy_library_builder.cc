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

#include <cstring>

#include "berberis/base/checks.h"
#include "berberis/base/logging.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/host_function_wrapper_impl.h"

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
namespace {

struct ExtraRegistry {
  const char* library_name;
  const KnownTrampoline* trampolines;
  size_t count;
};

constexpr size_t kMaxExtraRegistries = 8;
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
        // HACK: DoBadTrampoline needs function name passed as callee!
        MakeTrampolineCallable(guest_addr, false, DoBadTrampoline, name, name);
      } else {
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
      MakeTrampolineCallable(guest_addr, false, DoBadTrampoline, name, name);
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
        // TODO(b/287342829): copy variable.size bytes instead!
        memcpy(ToHostAddr<void>(guest_addr), addr, sizeof(GuestAddr));
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
