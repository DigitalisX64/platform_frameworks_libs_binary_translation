/*
 * Copyright (C) 2020 The Android Open Source Project
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

#include "berberis/base/config_globals.h"

#if defined(__BIONIC__)
#include <sys/system_properties.h>
#endif

#include <bitset>
#include <cerrno>
#include <cstdint>
#include <cstdlib>  // strtoull
#include <cstring>
#include <string>
#include <string_view>

#include "berberis/base/checks.h"
#include "berberis/base/forever_alloc.h"
#include "berberis/base/strings.h"  // Split
#include "berberis/base/tracing.h"

namespace berberis {

namespace {

const char* g_main_executable_real_path = nullptr;

const char* g_app_package_name = nullptr;
const char* g_app_private_dir = nullptr;

const char* MakeForeverCStr(std::string_view view) {
  auto str = std::bit_cast<char*>(AllocateForever(view.size() + 1, alignof(char)));
  memcpy(str, view.data(), view.size());
  str[view.size()] = '\0';
  return str;
}

#if defined(__BIONIC__)

bool TryReadBionicSystemPropertyImpl(const std::string_view prop_name, const char** value_ptr) {
  auto pi = __system_property_find(prop_name.data());
  if (!pi) {
    return false;
  }
  __system_property_read_callback(
      pi,
      [](void* cookie, const char*, const char* value, unsigned) {
        *std::bit_cast<const char**>(cookie) = MakeForeverCStr(value);
      },
      value_ptr);
  return true;
}

bool TryReadBionicSystemProperty(const std::string_view prop_name, const char** value_ptr) {
  // Allow properties without ro. prefix (they override ro. properties).
  if (prop_name.size() > 3 && prop_name.substr(0, 3) == "ro." &&
      TryReadBionicSystemPropertyImpl(prop_name.substr(3), value_ptr)) {
    return true;
  }

  return TryReadBionicSystemPropertyImpl(prop_name, value_ptr);
}

#endif

bool TryReadConfig(const char* env_name,
                   [[maybe_unused]] const std::string_view prop_name,
                   const char** value_ptr) {
  if (auto env = getenv(env_name)) {
    *value_ptr = MakeForeverCStr(env);
    return true;
  }
#if defined(__BIONIC__)
  return TryReadBionicSystemProperty(prop_name, value_ptr);
#else
  return false;
#endif
}

}  // namespace

ConfigStr::ConfigStr(const char* env_name, const char* prop_name) {
  TryReadConfig(env_name, prop_name, &value_);
}

void SetMainExecutableRealPath(std::string_view path) {
  CHECK(!path.empty());
  CHECK_EQ('/', path[0]);
  g_main_executable_real_path = MakeForeverCStr(path);
}

const char* GetMainExecutableRealPath() {
  return g_main_executable_real_path;
}

const char** GetMainExecutableRealPathPointer() {
  return &g_main_executable_real_path;
}

void SetAppPackageName(std::string_view name) {
  CHECK(!name.empty());
  g_app_package_name = MakeForeverCStr(name);
}

const char* GetAppPackageName() {
  return g_app_package_name;
}

void SetAppPrivateDir(std::string_view name) {
  CHECK(!name.empty());
  g_app_private_dir = MakeForeverCStr(name);
}

const char* GetAppPrivateDir() {
  return g_app_private_dir;
}

namespace {

std::string ToString(ConfigFlag flag) {
  switch (flag) {
    case kVerboseTranslation:
      return "verbose-translation";
    case kAccurateSigsegv:
      return "accurate-sigsegv";
    case kTopByteIgnore:
      return "top-byte-ignore";
    case kDisableHeavyOptimizations:
      return "disable-heavy-opts";
    case kDisableRegMap:
      return "disable-reg-map";
    case kDisableAdjacentRegionsTranslation:
      return "disable-adjacent-regions-translation";
    case kDisableIntrinsicInlining:
      return "disable-intrinsic-inlining";
    case kMergeProfilesForSameModeRegions:
      return "merge-profiles-for-same-mode-regions";
    case kPrintTranslatedAddrs:
      return "print-translated-addrs";
    case kDeterministicTracing:
      return "deterministic-tracing";
    case kPrintIRs:
      return "print-irs";
    case kPrintIRsAsDot:
      return "print-irs-as-dot";
    case kPrintCodePoolSize:
      return "print-code-pool-size";
    case kLocalExperiment:
      return "local-experiment";
    case kPlatformCustomCPUCapability:
      return "platform-custom-cpu-capability";
    // region digitalis
    case kDisableIrCheck:
      return "disable-ir-check";
    // endregion
    case kNumConfigFlags:
      break;
  }
  return "<unknown-config-flag>";
}

std::bitset<kNumConfigFlags> MakeConfigFlagsSet() {
  ConfigStr var("BERBERIS_FLAGS", "ro.berberis.flags");
  std::bitset<kNumConfigFlags> flags_set;
  if (!var.get()) {
    return flags_set;
  }
  auto token_vector = Split(var.get(), ",");
  for (const auto& token : token_vector) {
    bool found = false;
    for (int flag = 0; flag < kNumConfigFlags; flag++) {
      if (token == ToString(ConfigFlag(flag))) {
        flags_set.set(flag);
        found = true;
        break;
      }
    }
    if (!found) {
      TRACE_AND_ALOGE("Unrecognized config flag '%s' - ignoring", token.c_str());
    }
  }
  return flags_set;
}

uintptr_t ParseAddr(const char* addr_cstr) {
  if (!addr_cstr) {
    return 0;
  }
  char* end_ptr = nullptr;
  errno = 0;
  uintptr_t addr = static_cast<uintptr_t>(strtoull(addr_cstr, &end_ptr, 16));

  // Warning: setting errno on failure is implementation defined. So we also use extra heuristics.
  if (errno != 0 || (*end_ptr != '\n' && *end_ptr != '\0')) {
    TRACE_AND_ALOGE("Cannot convert \"%s\" to integer: %s\n",
                    addr_cstr,
                    errno != 0 ? strerror(errno) : "unexpected end of string");
    return 0;
  }
  return addr;
}

}  // namespace

const char* GetTracingConfig() {
  static ConfigStr var("BERBERIS_TRACING", "berberis.tracing");
  return var.get();
}

const char* GetTranslationModeConfig() {
  static ConfigStr var("BERBERIS_MODE", "berberis.mode");
  return var.get();
}

const char* GetProfilingConfig() {
  static ConfigStr var("BERBERIS_PROFILING", "berberis.profiling");
  return var.get();
}

uintptr_t GetEntryPointOverride() {
  static ConfigStr var("BERBERIS_ENTRY_POINT", "berberis.entry_point");
  static uintptr_t entry_point = ParseAddr(var.get());
  return entry_point;
}

bool IsConfigFlagSet(ConfigFlag flag) {
  static auto flags_set = MakeConfigFlagsSet();
  return flags_set.test(flag);
}

}  // namespace berberis
