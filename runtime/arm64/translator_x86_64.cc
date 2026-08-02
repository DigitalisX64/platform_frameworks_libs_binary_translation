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

#include "translator.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_os_primitives/guest_signal.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/heavy_optimizer/arm64/heavy_optimize_region.h"
#include "berberis/interpreter/arm64/interpreter.h"
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

// Translation strategy. The default is the two-gear tier: lite-translate every
// region first, then re-translate hot regions (those crossing the hotness
// counter) with the heavy optimizer (integer, branch, load/store, scalar FP,
// NEON integer). Instructions the heavy frontend does not yet handle make it
// bail, and a bailed region re-lite-translates rather than dropping to the
// interpreter. The single-gear lite tier and interpret-only remain available,
// opt-in via `berberis.mode=<name>` / BERBERIS_MODE (e.g. for bisection).
enum class TranslationMode {
  kInterpretOnly,
  kLiteTranslateOrFallbackToInterpret,
  kTwoGear,
  kNumModes,
};

TranslationMode g_translation_mode = TranslationMode::kTwoGear;

void UpdateTranslationMode() {
  // Indices must match the TranslationMode enum order.
  constexpr const char* kTranslationModeNames[] = {
      "interpret-only",
      "lite-translate-or-interpret",
      "two-gear",
  };
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
  // Install Berberis's host SIGSEGV/SIGBUS handler. This used to live in
  // runtime/berberis.cc inside `#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)`,
  // but that file is compiled into the host-agnostic libberberis_runtime which
  // does not get the ARM64 cflag — the call was silently elided. Hooking it
  // here (the arm64-specific translator) ensures the host fault signals are
  // claimed for arm64 guest processes.
  ClaimHostFaultSignals();
  UpdateTranslationMode();
}

// Exported for testing only.
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> TryLiteTranslateAndInstallRegion(
    GuestAddr pc,
    LiteTranslateParams params = LiteTranslateParams()) {
  MachineCode machine_code;

  params.end_pc = pc + GetExecutableRegionSize(pc);
  // enable direct dispatch
  // JIT is stable (16 breaks total, all register pressure). Enable direct
  // dispatch so regions chain directly through the translation cache instead
  // of returning to the ExecuteGuest loop on every region boundary.
  params.allow_dispatch = true;
  // Diagnostic knob: BERBERIS_FLAGS=disable-reg-map (or
  // `setprop berberis.flags disable-reg-map`) turns off cross-instruction
  // guest->host register mapping so every guest register read/write goes
  // through ThreadState memory. Mirrors the riscv64 translator; used to bisect
  // register-mapping codegen bugs from the interpreter-equivalent path.
  if (IsConfigFlagSet(kDisableRegMap)) {
    params.enable_reg_mapping = false;
  }
  auto [success, stop_pc] = TryLiteTranslateRegion(pc, &machine_code, params);

  size_t size = stop_pc - pc;


  if (success) {
    // region digitalis
    // A null install means the code pool could not get an executable region
    // (memory pressure). Report failure so the caller installs an interpreted
    // entry -- never kill the app over a JIT allocation.
    HostCodePiece piece = InstallTranslated(&machine_code, pc, size, "lite");
    if (piece.code == kNullHostCodeAddr) {
      return {false, {}, 0, {}};
    }
    return {true, piece, size, kLiteTranslated};
    // endregion
  }

  if (size == 0) {
    // Cannot translate even single instruction - the attempt failed.
    return {false, {}, 0, {}};
  }

  // Partial success: re-translate up to the point of failure with end_pc clamped.
  MachineCode another_machine_code;
  params.end_pc = stop_pc;
  std::tie(success, stop_pc) = TryLiteTranslateRegion(pc, &another_machine_code, params);
  CHECK(success);
  CHECK_EQ(stop_pc, params.end_pc);

  // region digitalis
  HostCodePiece range_piece = InstallTranslated(&another_machine_code, pc, size, "lite_range");
  if (range_piece.code == kNullHostCodeAddr) {
    return {false, {}, 0, {}};
  }
  // endregion
  return {true, range_piece, size, kLiteTranslated};
}

// translation profiling counters
static struct TranslationStats {
  uint64_t total_translations = 0;
  uint64_t jit_successes = 0;
  uint64_t jit_failures = 0;  // fell back to interpreter
  uint64_t total_jit_insns = 0;  // sum of region sizes (in ARM64 instructions)
  uint64_t interpret_invocations = 0;
} g_translation_stats;

// Minimum guest-instruction count for a region to be worth gearing up to the
// heavy optimizer. Below this, the optimizing tier's global passes (register
// allocation, loop-invariant hoisting, flag folding) have too little scope to
// recoup their higher per-instruction codegen overhead, so the heavy region
// runs no faster — and on tiny flag-heavy regions, slower — than the lite tier's
// tight single-pass output. Declining gear-up keeps the lite region, which is
// always correct and never slower than lite. Tunable via BERBERIS_GEARUP_MIN_INSNS
// for measurement sweeps.
constexpr size_t kDefaultGearUpMinInsns = 20;

size_t GetGearUpMinInsns() {
  static const size_t value = []() -> size_t {
    static ConfigStr config("BERBERIS_GEARUP_MIN_INSNS", "berberis.gearup_min_insns");
    const char* str = config.get();
    if (str) {
      char* end = nullptr;
      unsigned long parsed = strtoul(str, &end, 10);
      if (end != str && (*end == '\0' || *end == '\n')) {
        return static_cast<size_t>(parsed);
      }
    }
    return kDefaultGearUpMinInsns;
  }();
  return value;
}

// Optimizing (second-gear) translation install. The heavy optimizer translates
// the region; if it bails on an instruction it does not yet handle, this returns
// {false, ...} and the caller re-lite-translates the whole region.
std::tuple<bool, HostCodePiece, size_t, GuestCodeEntry::Kind> HeavyOptimizeAndInstallRegion(
    GuestAddr pc) {
  MachineCode machine_code;
  bool has_in_region_backedge = false;
  auto [stop_pc, success, number_of_instructions] = HeavyOptimizeRegion(
      pc, &machine_code, {.end_pc = pc + GetExecutableRegionSize(pc)}, &has_in_region_backedge);
  size_t size = stop_pc - pc;
  // When the heavy frontend bails partway through, the prefix it did translate is
  // still a valid region ending in an exit to the bailing PC. Install that prefix
  // ONLY when it captured an in-region loop back-edge — i.e. the hot loop is
  // fully inside the prefix and runs without the per-iteration region-exit
  // dispatch the lite tier pays. That is the case a real-app integrity/CRC loop
  // sitting in a function whose tail uses an unsupported instruction hits, and
  // it is the heavy tier's biggest win. Without a captured back-edge, a partial
  // prefix would just fragment a still-incomplete loop into extra region
  // boundaries (slower than the single lite region) — so discard and re-lite.
  if (!success) {
    if (has_in_region_backedge && size > 0) {
      // region digitalis
      HostCodePiece heavy_piece = InstallTranslated(&machine_code, pc, size, "heavy");
      if (heavy_piece.code == kNullHostCodeAddr) {
        return {false, {}, 0, {}};
      }
      return {true, heavy_piece, size, kHeavyOptimized};
      // endregion
    }
    return {false, {}, 0, {}};
  }
  // Decline gear-up for fully-translated regions too small for the heavy
  // optimizer to help. The size threshold (not the back-edge signal) governs the
  // full-success path on purpose: a small loop the heavy tier translates in full
  // is not automatically a win — e.g. a tight data-dependent CSEL loop is *slower*
  // heavy than lite, so keep it lite. The back-edge signal only rescues a region
  // that BAILED (above): there the alternative is re-liting and losing an
  // already-captured in-region loop. The caller re-lite-translates a declined
  // gear-up without self-profiling, so the region settles on the lite tier
  // permanently instead of re-attempting gear-up every threshold.
  if (number_of_instructions < GetGearUpMinInsns()) {
    return {false, {}, 0, {}};
  }
  // region digitalis
  HostCodePiece heavy_piece2 = InstallTranslated(&machine_code, pc, size, "heavy");
  if (heavy_piece2.code == kNullHostCodeAddr) {
    return {false, {}, 0, {}};
  }
  return {true, heavy_piece2, size, kHeavyOptimized};
  // endregion
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

  bool success = false;
  HostCodePiece host_code_piece{kEntryInterpret, 0};
  size_t size = first_insn_size;
  GuestCodeEntry::Kind kind = kInterpreted;

  if (g_translation_mode == TranslationMode::kInterpretOnly) {
    // berberis.mode=interpret-only: install the interpreter for every region,
    // bypassing the JIT. Diagnostic knob to bisect a JIT codegen bug (correct
    // under interpret-only) from an interpreter/syscall/proxy bug (still wrong).
    // success stays false -> counts as a jit_failure below, as before.
  } else if (g_translation_mode == TranslationMode::kTwoGear &&
             kGear == TranslationGear::kSecond) {
    // Hot region gearing up: try the optimizing tier; if the heavy frontend
    // bails on an instruction it does not yet handle, re-lite so the hot region
    // stays JIT-compiled; only then interpret.
    std::tie(success, host_code_piece, size, kind) = HeavyOptimizeAndInstallRegion(pc);
    if (!success) {
      std::tie(success, host_code_piece, size, kind) = TryLiteTranslateAndInstallRegion(pc);
    }
  } else {
    // First gear: lite-translate, falling back to the interpreter. In two-gear
    // mode enable self-profiling so the hotness counter can trigger the gear-up
    // (berberis_HandleLiteCounterThresholdReached); the default mode leaves it
    // off, preserving the historical single-gear behaviour exactly.
    LiteTranslateParams params;
    if (g_translation_mode == TranslationMode::kTwoGear) {
      params.enable_self_profiling = true;
      params.counter_location = &entry->invocation_counter;
    }
    std::tie(success, host_code_piece, size, kind) = TryLiteTranslateAndInstallRegion(pc, params);
  }

  if (!success) {
    host_code_piece = {kEntryInterpret, 0};
    size = first_insn_size;
    kind = kInterpreted;
  }
  cache->SetTranslatedAndUnlock(pc, entry, size, kind, host_code_piece);

  // profiling
  if (success) {
    g_translation_stats.jit_successes++;
    g_translation_stats.total_jit_insns += size / 4;
  } else {
    g_translation_stats.jit_failures++;
  }
  g_translation_stats.total_translations++;
  // Log first 20, then every 100th translation
  if (g_translation_stats.total_translations <= 20 || g_translation_stats.total_translations % 100 == 0) {
    TRACE_AND_ALOGD("berberis: trans#%lu pc=0x%lx size=%lu %s jit=%lu interp=%lu avg=%lu",
                    (unsigned long)g_translation_stats.total_translations,
                    (unsigned long)pc,
                    (unsigned long)(success ? size / 4 : 0),
                    kind == kHeavyOptimized ? "HEAVY" : (success ? "JIT" : "INTERP"),
                    (unsigned long)g_translation_stats.jit_successes,
                    (unsigned long)g_translation_stats.jit_failures,
                    g_translation_stats.jit_successes > 0
                        ? (unsigned long)(g_translation_stats.total_jit_insns / g_translation_stats.jit_successes)
                        : 0UL);
  }
}

// ATTENTION: This symbol gets called directly, without PLT. To keep text
// sharable we should prevent preemption of this symbol, so do not export it!
extern "C" __attribute__((used, __visibility__("hidden"))) void berberis_HandleNotTranslated(
    ThreadState* state) {
  TranslateRegion(state->cpu.insn_addr);
}

extern "C" __attribute__((used, __visibility__("hidden"))) void berberis_HandleInterpret(
    ThreadState* state) {
  // interpreter invocation counter with syscall diagnostics
  g_translation_stats.interpret_invocations++;
  bool should_log = g_translation_stats.interpret_invocations <= 25 ||
                    g_translation_stats.interpret_invocations % 5000000 == 0;
  uint64_t pre_x0 = state->cpu.x[0];
  if (should_log) {
    TRACE_AND_ALOGD("berberis: interp #%lu pc=0x%lx x8=%lu x0=0x%lx x1=0x%lx x2=0x%lx",
                    (unsigned long)g_translation_stats.interpret_invocations,
                    (unsigned long)state->cpu.insn_addr,
                    (unsigned long)state->cpu.x[8],
                    (unsigned long)state->cpu.x[0],
                    (unsigned long)state->cpu.x[1],
                    (unsigned long)state->cpu.x[2]);
  }
  InterpretBatch(state, 500, TranslationCache::GetInstance());
  // post-SVC diagnostics
  if (should_log && state->cpu.x[0] != pre_x0) {
    TRACE_AND_ALOGD("berberis: post-interp x0=0x%lx (was 0x%lx) pc=0x%lx",
                    (unsigned long)state->cpu.x[0],
                    (unsigned long)pre_x0,
                    (unsigned long)state->cpu.insn_addr);
  }
}

extern "C" __attribute__((used, __visibility__("hidden"))) const void* berberis_GetDispatchAddress(
    ThreadState* state) {
  CHECK(state);
  // dispatch watchdog for hang diagnosis
  static thread_local uint64_t dispatch_count = 0;
  dispatch_count++;
  if (dispatch_count % 10000 == 0) {
    TRACE_AND_ALOGD("berberis: dispatch#%lu pc=0x%lx x0=0x%lx x29=0x%lx x30=0x%lx sp=0x%lx",
                    (unsigned long)dispatch_count,
                    (unsigned long)state->cpu.insn_addr,
                    (unsigned long)state->cpu.x[0],
                    (unsigned long)state->cpu.x[29],
                    (unsigned long)state->cpu.x[30],
                    (unsigned long)state->cpu.x[1]);  // x1 for context
  }
  if (ArePendingSignalsPresent(*state)) {
    return AsHostCode(kEntryExitGeneratedCode);
  }
  return AsHostCode(TranslationCache::GetInstance()->GetHostCodePtr(state->cpu.insn_addr)->load());
}

extern "C" __attribute__((used, __visibility__("hidden"))) void
berberis_HandleLiteCounterThresholdReached(ThreadState* state) {
  // Only reachable in two-gear mode: the hotness counter is only emitted when
  // self-profiling is enabled, which the default (single-gear) mode never does.
  CHECK(g_translation_mode == TranslationMode::kTwoGear);
  TranslateRegion<TranslationGear::kSecond>(state->cpu.insn_addr);
}

}  // namespace berberis
