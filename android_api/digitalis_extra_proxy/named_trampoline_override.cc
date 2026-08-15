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

#include "berberis/digitalis_extra_proxy/named_trampoline_override.h"

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)

#include <string.h>

#include <atomic>

#include "berberis/base/logging.h"

namespace berberis {

namespace {

struct NamedOverrideEntry {
  const NamedTrampolineOverride* overrides;
  size_t count;
};

// One slot per RegisterDigitalisNamedTrampolineOverrides call (one per proxy
// library override file). Overflow is checked fatally below: unlike the
// symbol-level extras registry, a dropped registration here would silently
// reinstate an upstream LOG_ALWAYS_FATAL trampoline, so failing loudly at
// library load beats failing in an app's audio path.
constexpr size_t kMaxNamedOverrideRegistries = 16;
NamedOverrideEntry g_named_override_registries[kMaxNamedOverrideRegistries];
size_t g_num_named_override_registries = 0;

// Originals are recorded at wrap time, which can race across guest threads
// wrapping different objects concurrently; a name's original is always the
// same upstream function, so a relaxed atomic pointer per lookup result is
// sufficient. Sized for the total override count across all registries.
constexpr size_t kMaxNamedOverrides = 64;
struct RecordedOriginal {
  const char* name;
  std::atomic<TrampolineFunc> original;
};
RecordedOriginal g_recorded_originals[kMaxNamedOverrides];
std::atomic<size_t> g_num_recorded_originals{0};

std::atomic<TrampolineFunc>* FindOrAddRecordedOriginal(const char* name) {
  size_t count = g_num_recorded_originals.load(std::memory_order_acquire);
  for (size_t i = 0; i < count; ++i) {
    if (strcmp(g_recorded_originals[i].name, name) == 0) {
      return &g_recorded_originals[i].original;
    }
  }
  // Append. Concurrent first-wrap of two different names is possible; claim a
  // slot with a CAS on the count. Re-scan is unnecessary: duplicate rows for
  // one name would still all hold the same upstream function, and lookups
  // return the first match.
  for (;;) {
    size_t index = g_num_recorded_originals.load(std::memory_order_acquire);
    LOG_ALWAYS_FATAL_IF(index >= kMaxNamedOverrides,
                        "Digitalis named-trampoline original table overflow (%zu)", index);
    if (g_num_recorded_originals.compare_exchange_weak(
            index, index + 1, std::memory_order_acq_rel)) {
      g_recorded_originals[index].name = name;
      return &g_recorded_originals[index].original;
    }
  }
}

}  // namespace

void RegisterDigitalisNamedTrampolineOverrides(const NamedTrampolineOverride* overrides,
                                               size_t count) {
  LOG_ALWAYS_FATAL_IF(g_num_named_override_registries >= kMaxNamedOverrideRegistries,
                      "Digitalis named-trampoline override registry overflow");
  g_named_override_registries[g_num_named_override_registries++] = {overrides, count};
}

TrampolineFunc GetDigitalisNamedTrampolineOriginal(const char* name) {
  size_t count = g_num_recorded_originals.load(std::memory_order_acquire);
  for (size_t i = 0; i < count; ++i) {
    if (strcmp(g_recorded_originals[i].name, name) == 0) {
      return g_recorded_originals[i].original.load(std::memory_order_acquire);
    }
  }
  return nullptr;
}

TrampolineFunc FindDigitalisNamedTrampolineOverride(const char* name, TrampolineFunc original) {
  for (size_t i = 0; i < g_num_named_override_registries; ++i) {
    const auto& registry = g_named_override_registries[i];
    for (size_t j = 0; j < registry.count; ++j) {
      if (strcmp(registry.overrides[j].name, name) == 0) {
        // Self-wrap guard: if an override file (directly or by replaying an
        // upstream registration helper) wraps its own overridden name, the
        // incoming "original" is the override itself; recording it would make
        // delegation to the original recurse forever. Keep the previously
        // recorded upstream trampoline instead.
        if (original != registry.overrides[j].override_trampoline) {
          FindOrAddRecordedOriginal(registry.overrides[j].name)
              ->store(original, std::memory_order_release);
        }
        return registry.overrides[j].override_trampoline;
      }
    }
  }
  return nullptr;
}

}  // namespace berberis

#endif  // NATIVE_BRIDGE_GUEST_ARCH_ARM64
