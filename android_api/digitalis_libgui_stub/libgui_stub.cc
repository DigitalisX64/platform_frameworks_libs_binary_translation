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

// Guest libgui.so stub — see Android.bp for the rationale. Symbols that real
// guest code dlsym()s from libgui are added here as benign no-ops as they are
// observed; the stub's purpose is only to let dlopen("libgui.so")/dlsym succeed
// so the caller does not fault, while the real surface/present path stays on the
// host-proxied GPU route.

// A single placeholder so the produced .so is non-empty and has a dynamic symbol
// table.
extern "C" int __digitalis_libgui_stub_placeholder() {
  return 0;
}

// Symbols real guest code dlsym()s from libgui, provided as benign no-ops.

// android::Surface::hook_perform(ANativeWindow*, int operation, ...) — the static
// implementation behind ANativeWindow's `perform` hook (NATIVE_WINDOW_SET_* buffer
// /swap-interval/scaling setters). Google Filament's Android platform dlsym()s and
// calls it on the SwapChain present path. Under Digitalis the actual buffer/present
// path is proxied to the host GPU stack, so these guest-side Surface operations are
// no-ops here; return 0 (success) and ignore the varargs so the caller proceeds and
// keeps rendering through the proxied route. Mangles to
// _ZN7android7Surface12hook_performEP13ANativeWindowiz.
struct ANativeWindow;
namespace android {
struct Surface {
  static int hook_perform(ANativeWindow* window, int operation, ...);
  static int hook_query(const ANativeWindow* window, int what, int* value);
};
}  // namespace android

int android::Surface::hook_perform(ANativeWindow* /*window*/, int /*operation*/, ...) {
  return 0;
}

// android::Surface::hook_query(const ANativeWindow*, int what, int* value) — the
// static implementation behind ANativeWindow's `query` hook (NATIVE_WINDOW_WIDTH/
// HEIGHT/FORMAT/MIN_UNDEQUEUED_BUFFERS/... getters). Kuaishou's Kwai media player
// (libAemonPlayer.so) resolves this host libgui symbol directly by parsing
// /proc/self/maps + the ELF export table and calls it, so the guest `blr` lands in
// host x86_64 code and traps in berberis_HandleNoExec. The host-call redirect then
// reroutes it to this guest copy; before this stub exported the symbol the redirect
// failed and the launch SIGSEGV'd (~25% of launches). Under Digitalis the real
// surface/present path stays on the host-proxied GPU route, so this guest-side query
// is a no-op: zero the caller's out-parameter (avoid an uninitialised read) and
// return 0 (success) so the caller proceeds. Mangles to
// _ZN7android7Surface10hook_queryEPK13ANativeWindowiPi.
int android::Surface::hook_query(const ANativeWindow* /*window*/, int /*what*/, int* value) {
  if (value != nullptr) {
    *value = 0;
  }
  return 0;
}
