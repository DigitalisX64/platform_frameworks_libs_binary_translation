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
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_os_primitives/guest_signal.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_opaque.h"
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
  auto [success, stop_pc] = TryLiteTranslateRegion(pc, &machine_code, params);

  size_t size = stop_pc - pc;

  if (success) {
    return {true, InstallTranslated(&machine_code, pc, size, "lite"), size, kLiteTranslated};
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

  return {true,
          InstallTranslated(&another_machine_code, pc, size, "lite_range"),
          size,
          kLiteTranslated};
}

// translation profiling counters
static struct TranslationStats {
  uint64_t total_translations = 0;
  uint64_t jit_successes = 0;
  uint64_t jit_failures = 0;  // fell back to interpreter
  uint64_t total_jit_insns = 0;  // sum of region sizes (in ARM64 instructions)
  uint64_t interpret_invocations = 0;
} g_translation_stats;

void TranslateRegion(GuestAddr pc) {
  TranslationCache* cache = TranslationCache::GetInstance();

  GuestCodeEntry* entry = cache->AddAndLockForTranslation(pc, 0);
  if (!entry) {
    return;
  }

  GuestMapShadow* guest_map_shadow = GuestMapShadow::GetInstance();
  auto [is_executable, first_insn_size] = IsPcExecutable(pc, guest_map_shadow);
  if (!is_executable) {
    cache->SetTranslatedAndUnlock(pc, entry, first_insn_size, kSpecialHandler, {kEntryNoExec, 0});
    return;
  }

  // Try lite translation first; fall back to interpreter on failure.
  auto [success, host_code_piece, size, kind] = TryLiteTranslateAndInstallRegion(pc);
  if (success) {
    cache->SetTranslatedAndUnlock(pc, entry, size, kind, host_code_piece);
    // profiling
    g_translation_stats.jit_successes++;
    g_translation_stats.total_jit_insns += size / 4;
  } else {
    cache->SetTranslatedAndUnlock(
        pc, entry, first_insn_size, kInterpreted, {kEntryInterpret, 0});
    g_translation_stats.jit_failures++;
  }
  g_translation_stats.total_translations++;
  // Log first 20, then every 100th translation
  if (g_translation_stats.total_translations <= 20 || g_translation_stats.total_translations % 100 == 0) {
    TRACE_AND_ALOGD("berberis: trans#%lu pc=0x%lx size=%lu %s jit=%lu interp=%lu avg=%lu",
                    (unsigned long)g_translation_stats.total_translations,
                    (unsigned long)pc,
                    (unsigned long)(success ? size / 4 : 0),
                    success ? "JIT" : "INTERP",
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
  // Re-translate — stays as lite-translated or falls back to interpreted.
  TranslateRegion(state->cpu.insn_addr);
}

}  // namespace berberis
