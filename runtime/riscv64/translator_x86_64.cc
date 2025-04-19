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

#include "translator_x86_64.h"
#include "berberis/base/config.h"  // kGuestPageSize;
#include "berberis/runtime/translator.h"
#include "translator.h"

#include <cstdint>
#include <cstdlib>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/checks.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/heavy_optimizer/riscv64/heavy_optimize_region.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {

namespace {

// Syntax sugar.
GuestCodeEntry::Kind kSpecialHandler = GuestCodeEntry::Kind::kSpecialHandler;
GuestCodeEntry::Kind kInterpreted = GuestCodeEntry::Kind::kInterpreted;
GuestCodeEntry::Kind kLiteTranslated = GuestCodeEntry::Kind::kLiteTranslated;
GuestCodeEntry::Kind kHeavyOptimized = GuestCodeEntry::Kind::kHeavyOptimized;

enum class TranslationMode {
  kInterpretOnly,
  kLiteTranslateOrFallbackToInterpret,
  kHeavyOptimizeOrFallbackToInterpret,
  kHeavyOptimizeOrFallbackToLiteTranslator,
  kLiteTranslateThenHeavyOptimize,
  kTwoGear = kLiteTranslateThenHeavyOptimize,
  kNumModes
};

TranslationMode g_translation_mode = TranslationMode::kTwoGear;

void UpdateTranslationMode() {
  // Indices must match TranslationMode enum.
  constexpr const char* kTranslationModeNames[] = {"interpret-only",
                                                   "lite-translate-or-interpret",
                                                   "heavy-optimize-or-interpret",
                                                   "heavy-optimize-or-lite-translate",
                                                   "two-gear"};
  static_assert(static_cast<int>(TranslationMode::kNumModes) ==
                sizeof(kTranslationModeNames) / sizeof(char*));

  const char* config_mode = GetTranslationModeConfig();
  if (!config_mode) {
    return;
  }

  for (int i = 0; i < static_cast<int>(TranslationMode::kNumModes); i++) {
    if (0 == strcmp(config_mode, kTranslationModeNames[i])) {
      g_translation_mode = TranslationMode(i);
      TRACE("translation mode is manually set to '%s'", config_mode);
      return;
    }
  }

  LOG_ALWAYS_FATAL("Unrecognized translation mode '%s'", config_mode);
}

enum class TranslationGear {
  kFirst,
  kSecond,
};

size_t GetExecutableRegionSize(GuestAddr pc) {
  // With kGuestPageSize>=4k we scan at least 1k instructions, which should be enough for a single
  // region.
  auto [is_exec, exec_size] =
      GuestMapShadow::GetInstance()->GetExecutableRegionSize(pc, config::kGuestPageSize);
  // Must be called on pc which is already proven to be executable.
  CHECK(is_exec);
  return exec_size;
}

}  // namespace

void InitTranslatorArch() {
  UpdateTranslationMode();
}

// Exported for testing only.
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> TryLiteTranslateAndInstallRegion(
    GuestAddr pc,
    LiteTranslateParams params) {
  MachineCode machine_code;

  params.end_pc = pc + GetExecutableRegionSize(pc);
  auto [success, stop_pc] = TryLiteTranslateRegion(pc, &machine_code, params);

  size_t size = stop_pc - pc;

  if (success) {
    return {true, InstallTranslated(&machine_code, pc, size, "lite"), size, kLiteTranslated};
  }

  if (size == 0) {
    // Cannot translate even single instruction - the attempt failed.
    return {false, {}, 0, {}};
  }

  MachineCode another_machine_code;
  params.end_pc = stop_pc;
  std::tie(success, stop_pc) = TryLiteTranslateRegion(pc, &another_machine_code, params);
  CHECK(success);
  CHECK_EQ(stop_pc, params.end_pc);

  return {true,
          InstallTranslated(&another_machine_code, pc, size, "lite_range"),
          size,
          kLiteTranslated};
}

// Exported for testing only.
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> HeavyOptimizeRegion(GuestAddr pc) {
  MachineCode machine_code;
  auto [stop_pc, success, unused_number_of_processed_instructions] =
      HeavyOptimizeRegion(pc, &machine_code, {.end_pc = pc + GetExecutableRegionSize(pc)});
  size_t size = stop_pc - pc;
  if (!success && (size == 0)) {
    // Cannot translate even single instruction - the attempt failed.
    return {false, {}, 0, {}};
  }

  // Report success because we at least translated some instructions.
  return {true, InstallTranslated(&machine_code, pc, size, "heavy"), size, kHeavyOptimized};
}

template <TranslationGear kGear = TranslationGear::kFirst>
void TranslateRegion(GuestAddr pc) {
  TranslationCache* cache = TranslationCache::GetInstance();

  GuestCodeEntry* entry;
  if constexpr (kGear == TranslationGear::kFirst) {
    entry = cache->AddAndLockForTranslation(pc, 0);
  } else {
    CHECK(g_translation_mode == TranslationMode::kTwoGear);
    entry = cache->LockForGearUpTranslation(pc);
  }
  if (!entry) {
    return;
  }

  GuestMapShadow* guest_map_shadow = GuestMapShadow::GetInstance();
  auto [is_executable, first_insn_size] = IsPcExecutable(pc, guest_map_shadow);
  if (!is_executable) {
    cache->SetTranslatedAndUnlock(pc, entry, first_insn_size, kSpecialHandler, {kEntryNoExec, 0});
    return;
  }

  bool success;
  HostCodePiece host_code_piece;
  size_t size;
  GuestCodeEntry::Kind kind;
  if (g_translation_mode == TranslationMode::kInterpretOnly) {
    std::tie(host_code_piece, size, kind) =
        std::make_tuple(HostCodePiece{kEntryInterpret, 0}, first_insn_size, kInterpreted);
  } else if (g_translation_mode == TranslationMode::kLiteTranslateOrFallbackToInterpret) {
    std::tie(success, host_code_piece, size, kind) = TryLiteTranslateAndInstallRegion(pc);
    if (!success) {
      std::tie(host_code_piece, size, kind) =
          std::make_tuple(HostCodePiece{kEntryInterpret, 0}, first_insn_size, kInterpreted);
    }
  } else if (g_translation_mode == TranslationMode::kTwoGear && kGear == TranslationGear::kFirst) {
    std::tie(success, host_code_piece, size, kind) = TryLiteTranslateAndInstallRegion(
        pc, {.enable_self_profiling = true, .counter_location = &(entry->invocation_counter)});
    if (!success) {
      // Heavy supports more insns than lite, so try to heavy optimize. If that fails, then
      // fallback to interpret.
      std::tie(success, host_code_piece, size, kind) = HeavyOptimizeRegion(pc);
      if (!success) {
        std::tie(host_code_piece, size, kind) =
            std::make_tuple(HostCodePiece{kEntryInterpret, 0}, first_insn_size, kInterpreted);
      }
    }
  } else if (g_translation_mode == TranslationMode::kHeavyOptimizeOrFallbackToInterpret ||
             g_translation_mode == TranslationMode::kHeavyOptimizeOrFallbackToLiteTranslator ||
             (g_translation_mode == TranslationMode::kTwoGear &&
              kGear == TranslationGear::kSecond)) {
    std::tie(success, host_code_piece, size, kind) = HeavyOptimizeRegion(pc);
    if (!success) {
      if (g_translation_mode == TranslationMode::kHeavyOptimizeOrFallbackToInterpret ||
          // Lite might fail since not all insns are implemented. Fallback to interpret.
          (g_translation_mode == TranslationMode::kTwoGear && kGear == TranslationGear::kSecond)) {
        std::tie(host_code_piece, size, kind) =
            std::make_tuple(HostCodePiece{kEntryInterpret, 0}, first_insn_size, kInterpreted);
      } else if (g_translation_mode == TranslationMode::kHeavyOptimizeOrFallbackToLiteTranslator) {
        std::tie(success, host_code_piece, size, kind) =
            TryLiteTranslateAndInstallRegion(pc, {.enable_self_profiling = false});
        // Lite might fail since not all insns are implemented. Fallback to interpret.
        if (!success) {
          std::tie(host_code_piece, size, kind) =
              std::make_tuple(HostCodePiece{kEntryInterpret, 0}, first_insn_size, kInterpreted);
        }
      }
    }
  } else {
    LOG_ALWAYS_FATAL("Unsupported translation mode %u", g_translation_mode);
  }

  cache->SetTranslatedAndUnlock(pc, entry, size, kind, host_code_piece);
}

// A wrapper to export a template function.
void TranslateRegionAtFirstGear(GuestAddr pc) {
  TranslateRegion<TranslationGear::kFirst>(pc);
}

// ATTENTION: This symbol gets called directly, without PLT. To keep text
// sharable we should prevent preemption of this symbol, so do not export it!
// TODO(b/232598137): may be set default visibility to protected instead?
extern "C" __attribute__((used, __visibility__("hidden"))) void berberis_HandleNotTranslated(
    ThreadState* state) {
  TranslateRegion(state->cpu.insn_addr);
}

extern "C" __attribute__((used, __visibility__("hidden"))) void berberis_HandleInterpret(
    ThreadState* state) {
  InterpretInsn(state);
}

extern "C" __attribute__((used, __visibility__("hidden"))) const void* berberis_GetDispatchAddress(
    ThreadState* state) {
  CHECK(state);
  if (ArePendingSignalsPresent(*state)) {
    return AsHostCode(kEntryExitGeneratedCode);
  }
  return AsHostCode(TranslationCache::GetInstance()->GetHostCodePtr(state->cpu.insn_addr)->load());
}

extern "C" __attribute__((used, __visibility__("hidden"))) void
berberis_HandleLiteCounterThresholdReached(ThreadState* state) {
  CHECK(g_translation_mode == TranslationMode::kTwoGear);
  TranslateRegion<TranslationGear::kSecond>(state->cpu.insn_addr);
}

}  // namespace berberis
