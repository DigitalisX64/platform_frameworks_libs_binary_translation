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

#ifndef BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_REGISTER_EXTRA_TRAMPOLINES_H_
#define BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_REGISTER_EXTRA_TRAMPOLINES_H_

#include <cstddef>

#include "berberis/proxy_loader/proxy_library_builder.h"

// Registers a digitalis_extra trampoline table for `lib_name` at static-init
// time. `table` must be a `const KnownTrampoline[]` array in scope; its element
// count is derived from the array type, so no separate count constant is needed.
//
// Constructor priority 101 runs these registrations before the upstream proxy
// tables' constructors. Upstream libc_translation.cc (and the other proxy libs)
// declare their init constructors with no priority (= 65535 / default), while
// the early proxy-library prep work lives in priorities 0-100; gcc/clang run
// constructors in ascending priority order (lower = earlier). Priority 101
// therefore lands after the early prep but before the late default-priority
// upstream tables, so these entries beat a DoBadTrampoline primary via the
// arm64 InterceptSymbol override in proxy_library_builder.cc.
//
// `table` is pasted into the generated function name to keep the symbol unique
// per translation unit. Invoke at namespace scope, once per table.
#define REGISTER_DIGITALIS_EXTRA_TRAMPOLINES(lib_name, table)                 \
  __attribute__((constructor(101))) static void RegisterDigitalisExtra_##table() { \
    ::berberis::ProxyLibraryBuilder::RegisterExtraTrampolines(                \
        (lib_name), (table), sizeof(table) / sizeof((table)[0]));            \
  }

// Same as above but for OVERRIDES of symbols whose primary table entry already
// has a working trampoline. The override's marshal function receives a
// `const ChainedTrampoline*` as its HostCode callee (the primary's resolved
// {marshal_and_call, thunk}) so it can run the upstream behavior first and only
// post-process guest state. See RegisterExtraTrampolineOverrides.
#define REGISTER_DIGITALIS_EXTRA_TRAMPOLINE_OVERRIDES(lib_name, table)        \
  __attribute__((constructor(101))) static void RegisterDigitalisOverride_##table() { \
    ::berberis::ProxyLibraryBuilder::RegisterExtraTrampolineOverrides(        \
        (lib_name), (table), sizeof(table) / sizeof((table)[0]));            \
  }

#endif  // BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_REGISTER_EXTRA_TRAMPOLINES_H_
