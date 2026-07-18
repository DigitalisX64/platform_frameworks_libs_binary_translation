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

#ifndef BERBERIS_LOADER_PROXY_LIBRARY_BUILDER_H_
#define BERBERIS_LOADER_PROXY_LIBRARY_BUILDER_H_

#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_function_wrapper_impl.h"  // TrampolineFunc

namespace berberis {

struct ThreadState;

struct KnownTrampoline {
  const char* name;
  TrampolineFunc marshal_and_call;
  void* thunk;
};

struct KnownVariable {
  const char* name;
  size_t size;
};

// TODO(eaeltsin): these are stubs used by known trampolines, consider killing!
void DoBadThunk();
void DoBadTrampoline(HostCode callee, ThreadState* state);

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
// arm64-only loud, non-fatal replacement for DoBadTrampoline. Installed by
// InterceptSymbol at the DoBadTrampoline fall-through sites so an uncovered bad
// symbol degrades to a greppable trace + zeroed x0 instead of a SIGABRT. See
// the definition in proxy_library_builder.cc.
void DoGracefulBadTrampoline(HostCode callee, ThreadState* state);

// Callee payload for an extra-trampoline OVERRIDE of a symbol that has a
// working primary trampoline (see RegisterExtraTrampolineOverrides). The
// override's marshal function receives a pointer to this struct as its
// HostCode callee and can run the primary first:
//   chain->marshal_and_call(chain->thunk, state);
struct ChainedTrampoline {
  TrampolineFunc marshal_and_call;
  void* thunk;
};
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
// endregion

class ProxyLibraryBuilder {
 public:
  ProxyLibraryBuilder() = default;

  void Build(const char* library_name,
             size_t size_translation,
             const KnownTrampoline* translations,
             size_t size_data_symbols,
             const KnownVariable* variables);

  void InterceptSymbol(GuestAddr guest_addr, const char* name);

  // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  // Append additional KnownTrampoline entries for `library_name` that will be
  // consulted by InterceptSymbol after the primary table search misses.
  //
  // Built only for the arm64-to-x86_64 translation configuration. The proxy
  // loader is otherwise guest-arch-agnostic, but the extras-registry is a
  // Digitalis-specific addition and is gated to its target build to avoid
  // bloating libberberis_riscv64 (or any future guest-arch lib) with code it
  // never exercises. The arm64 build of the proxy loader is shipped as the
  // separate cc_library_static `libberberis_proxy_loader_arm64`; the original
  // upstream `libberberis_proxy_loader` is unchanged.
  //
  // The primary table is built once via `Build()` from a single static array
  // shipped by upstream proxy_libc / proxy_libm. This API exists so that the
  // Digitalis-side android_api/libc/ and android_api/libm/ source can ship a
  // *parallel* trampoline array for symbols not covered by the upstream JSON
  // manifest (e.g. LFS-64 aliases, isnan/isinf family, memrchr, strchrnul,
  // ldexpf, cospi, sinpi …). A prior audit found 121 libc + 34 libm
  // such symbols; they all resolve correctly today via translation through the
  // guest libc/libm (LD_DEBUG=symbols shows zero UNRESOLVED warnings), but they
  // miss the fast-path host-passthrough trampoline and are interpreted instead.
  //
  // `trampolines` must remain valid for the lifetime of the proxy process —
  // typically a static const array in a .cc file with a __constructor that
  // calls this API at module-load time.
  //
  // Thread-safety: callers must serialize via single-thread initialization
  // ordering (gcc constructor priority). The registry is fixed-size; if more
  // than kMaxExtraRegistries (8) registrations land for one proxy lib, the
  // overflow is silently dropped.
  static void RegisterExtraTrampolines(const char* library_name,
                                       const KnownTrampoline* trampolines,
                                       size_t count);

  // Register KnownTrampoline entries that OVERRIDE a symbol whose primary
  // table entry already carries a working (non-DoBadTrampoline) trampoline.
  // Unlike RegisterExtraTrampolines (a fallback for symbols the primary table
  // misses or marks bad), an override replaces the primary's marshal function
  // while keeping it reachable: InterceptSymbol installs the override with a
  // ChainedTrampoline as its callee, holding the primary's resolved
  // {marshal_and_call, thunk}. The override runs the primary via the chain and
  // then post-processes guest state — e.g. the libEGL eglGetProcAddress
  // override reuses the upstream wrap table and only adds handling for
  // host-present procs the upstream table cannot marshal.
  //
  // Same lifetime/thread-safety contract as RegisterExtraTrampolines.
  static void RegisterExtraTrampolineOverrides(const char* library_name,
                                               const KnownTrampoline* trampolines,
                                               size_t count);
#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
  // endregion

 private:
  const char* library_name_ = nullptr;
  size_t num_functions_ = 0;
  const KnownTrampoline* functions_ = nullptr;
  size_t num_variables_ = 0;
  const KnownVariable* variables_ = nullptr;
  void* handle_ = nullptr;
};

// Assume kKnownTrampolines and kKnownVariables defined.
#define DEFINE_INIT_PROXY_LIBRARY(soname)                                    \
  extern "C" void InitProxyLibrary(ProxyLibraryBuilder* builder) {           \
    builder->Build(soname,                                                   \
                   sizeof(kKnownTrampolines) / sizeof(kKnownTrampolines[0]), \
                   kKnownTrampolines,                                        \
                   sizeof(kKnownVariables) / sizeof(kKnownVariables[0]),     \
                   kKnownVariables);                                         \
  }

}  // namespace berberis

#endif  // BERBERIS_LOADER_PROXY_LIBRARY_BUILDER_H_
