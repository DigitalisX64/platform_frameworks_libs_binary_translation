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

// Digitalis-side fast-path trampolines for libc symbols that the upstream
// proxy_libc JSON manifest does not cover.
//
// Background: the upstream proxy_libc trampoline table at
// frameworks/libs/native_bridge_support/android_api/libc/proxy/trampolines_arm64_to_x86_64-inl.h
// is auto-generated from api_arm64.json and covers ~100 of the ~1441 symbols
// in the arm64 guest libc.so. The other ~1340 symbols still resolve correctly
// — they just go through full instruction-by-instruction translation of the
// guest libc.so code instead of dispatching directly to the host x86_64 libc.
// A prior audit enumerated 121 specific symbols (e.g. memrchr,
// strchrnul, the LFS-64 aliases, the isnan/isinf family) where adding a
// host-passthrough fast-path is a strict speedup with no correctness change.
//
// Restricted to ABI-compatible cases: all symbols here have identical struct
// layouts and argument types between arm64 and x86_64 bionic on LP64. Long-
// double-returning math (e.g. nexttowardl), jmp_buf-touching, and the
// _Unwind_* family are intentionally NOT covered here because their host-vs-
// guest ABI differs and they need custom marshaling beyond the scope of this
// file.

#if defined(__x86_64__)

#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/proxy_loader/proxy_library_builder.h"

#include "register_extra_trampolines.h"

namespace berberis {
namespace {

// Pointer-only signatures use `void*` so the trampoline doesn't unpack the
// pointee — the proxy machinery passes pointers through verbatim, which is
// correct as long as the pointed-to struct layout is identical on both
// arches. Verified for arm64 vs x86_64 bionic LP64 on:
//   FILE, struct rlimit, struct stat, struct statfs, struct statvfs,
//   struct dirent, fd_set, sigset_t, struct timespec.
//
// All ssize_t / off_t are int64_t under LP64.
// File-offset-64 aliases (creat64, open64-not-here-because-varargs, stat64,
// mmap64, etc.) are identical to their bare names on LP64.
const KnownTrampoline kDigitalisExtraLibcTrampolines[] = {
    // --- Math classification wrappers (double / float; long-double variants
    // intentionally skipped due to arm64 128-bit vs x86_64 80-bit ABI gap).
    {"isnan", GetTrampolineFunc<auto(double) -> int32_t>(), nullptr},
    {"isnanf", GetTrampolineFunc<auto(float) -> int32_t>(), nullptr},
    {"isinf", GetTrampolineFunc<auto(double) -> int32_t>(), nullptr},
    {"isinff", GetTrampolineFunc<auto(float) -> int32_t>(), nullptr},
    {"isfinite", GetTrampolineFunc<auto(double) -> int32_t>(), nullptr},
    {"isfinitef", GetTrampolineFunc<auto(float) -> int32_t>(), nullptr},
    {"isnormal", GetTrampolineFunc<auto(double) -> int32_t>(), nullptr},
    {"isnormalf", GetTrampolineFunc<auto(float) -> int32_t>(), nullptr},
    {"__fpclassify", GetTrampolineFunc<auto(double) -> int32_t>(), nullptr},

    // --- Memory primitives (existing trampolines cover memchr/memcpy/memset;
    // these are the remaining ones from the audit). _chk variants take an
    // extra trailing destination-size argument.
    {"memrchr", GetTrampolineFunc<auto(void*, int32_t, size_t) -> void*>(), nullptr},
    {"__memcpy_chk",
     GetTrampolineFunc<auto(void*, void*, size_t, size_t) -> void*>(),
     nullptr},
    {"__memset_chk",
     GetTrampolineFunc<auto(void*, int32_t, size_t, size_t) -> void*>(),
     nullptr},

    // --- String primitives.
    {"strchrnul", GetTrampolineFunc<auto(void*, int32_t) -> void*>(), nullptr},
    {"stpcpy", GetTrampolineFunc<auto(void*, void*) -> void*>(), nullptr},

    // --- Misc POSIX (no struct unpacking needed: opaque pointers + ints).
    // ftell returns long (int64_t on LP64).
    {"ftell", GetTrampolineFunc<auto(void*) -> int64_t>(), nullptr},
    {"lockf", GetTrampolineFunc<auto(int32_t, int32_t, int64_t) -> int32_t>(), nullptr},
    {"prlimit",
     GetTrampolineFunc<auto(int32_t, int32_t, void*, void*) -> int32_t>(),
     nullptr},
    {"pselect",
     GetTrampolineFunc<auto(int32_t, void*, void*, void*, void*, void*) -> int32_t>(),
     nullptr},

    // --- LFS-64 aliases. On LP64 (both arm64 and x86_64), these are
    // strictly equivalent to their unsuffixed counterparts; bionic exports
    // both names but the underlying implementations are identical. Pass
    // through to the host's equivalent symbol via dlsym (which will resolve
    // them as the same address).
    {"stat64", GetTrampolineFunc<auto(void*, void*) -> int32_t>(), nullptr},
    {"lstat64", GetTrampolineFunc<auto(void*, void*) -> int32_t>(), nullptr},
    {"fstatfs64", GetTrampolineFunc<auto(int32_t, void*) -> int32_t>(), nullptr},
    {"fstatvfs64", GetTrampolineFunc<auto(int32_t, void*) -> int32_t>(), nullptr},
    {"statfs64", GetTrampolineFunc<auto(void*, void*) -> int32_t>(), nullptr},
    {"statvfs64", GetTrampolineFunc<auto(void*, void*) -> int32_t>(), nullptr},
    {"ftruncate64", GetTrampolineFunc<auto(int32_t, int64_t) -> int32_t>(), nullptr},
    {"truncate64", GetTrampolineFunc<auto(void*, int64_t) -> int32_t>(), nullptr},
    {"getrlimit64", GetTrampolineFunc<auto(int32_t, void*) -> int32_t>(), nullptr},
    {"setrlimit64", GetTrampolineFunc<auto(int32_t, void*) -> int32_t>(), nullptr},
    {"sendfile64",
     GetTrampolineFunc<auto(int32_t, int32_t, void*, size_t) -> int64_t>(),
     nullptr},
    {"creat64", GetTrampolineFunc<auto(void*, uint32_t) -> int32_t>(), nullptr},
    {"alphasort64", GetTrampolineFunc<auto(void*, void*) -> int32_t>(), nullptr},

    // --- mmap64 takes (addr, length, prot, flags, fd, offset). LP64 off_t
    // is 64 bits, identical to off64_t.
    {"mmap64",
     GetTrampolineFunc<auto(void*, size_t, int32_t, int32_t, int32_t, int64_t) -> void*>(),
     nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libc.so", kDigitalisExtraLibcTrampolines)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
