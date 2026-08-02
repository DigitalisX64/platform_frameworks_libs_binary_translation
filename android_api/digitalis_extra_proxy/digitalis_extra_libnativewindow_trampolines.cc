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

// Digitalis-side contract stub for ANativeWindow_setPerformInterceptor, which
// the upstream proxy leaves as DoBadTrampoline. It is a private/system debug
// hook whose interceptor callback takes a per-op va_list that cannot be
// forwarded without per-op interpretation; no known app calls it. The stub is a
// crash-free no-op (install no interceptor) instead of aborting. See
// DoStub_ANativeWindow_setPerformInterceptor in digitalis_extra_stubs.h and
// a per-op va_list dispatcher would be needed for live interception.

// ANativeWindow_lock's ANativeWindow_Buffer::stride is repaired for planar YUV
// windows -- see DoDigitalisANativeWindowLock below.

#if defined(__x86_64__)

#include <cstdint>

#include "berberis/base/tracing.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/proxy_loader/proxy_library_builder.h"

#include "digitalis_extra_stubs.h"
#include "register_extra_trampolines.h"

namespace berberis {
namespace {

// ANativeWindow_Buffer, from android/native_window.h. Identical layout for the
// arm64 guest and the x86_64 host under LP64, which is why the upstream
// trampoline can pass the pointer straight through.
struct ANativeWindowBuffer {
  int32_t width;
  int32_t height;
  int32_t stride;
  int32_t format;
  void* bits;
  uint32_t reserved[6];
};

// HAL_PIXEL_FORMAT values (system/graphics.h) for the planar/semiplanar YUV
// layouts a window can be configured with.
constexpr int32_t kHalPixelFormatYV12 = 0x32315659;
constexpr int32_t kHalPixelFormatYCrCb420SP = 0x11;
constexpr int32_t kHalPixelFormatY8 = 0x20203859;
constexpr int32_t kHalPixelFormatY16 = 0x20363159;

// The luma row stride a gralloc allocation is required to have for `format`, or
// 0 when we cannot state it. YV12 is specified with a 16-pixel-aligned luma
// stride; the single-plane Y formats are tightly packed.
int32_t LumaStrideFor(int32_t format, int32_t width) {
  if (width <= 0) {
    return 0;
  }
  switch (format) {
    case kHalPixelFormatYV12:
      return (width + 15) & ~15;
    case kHalPixelFormatYCrCb420SP:
    case kHalPixelFormatY8:
    case kHalPixelFormatY16:
      return width;
    default:
      return 0;
  }
}

using PFN_ANativeWindowLock = int32_t (*)(void*, void*, void*);

// Chained override of the upstream ANativeWindow_lock trampoline.
//
// A planar-YUV window comes back from the host with `stride == 0`. The single
// `ANativeWindow_Buffer::stride` field cannot describe a planar layout -- those
// strides belong in `android_ycbcr` (ystride/cstride) -- so the emulator's
// gralloc simply leaves it zero, whereas the gralloc implementations these apps
// are written against report the luma stride there.
//
// A CPU-side renderer addresses row y at `bits + y * stride`, so a zero stride
// collapses every row onto row 0 and the posted buffer stays as allocated. An
// untouched YUV buffer is all zeros, and Y=U=V=0 converts to RGB(0,135,0) --
// solid green video, which is exactly how this surfaced.
//
// Repair only the contract violation: when the host reports success but leaves
// the stride unset, substitute the stride the format mandates. A host that does
// fill the field is left completely alone.
void DoDigitalisANativeWindowLock(HostCode callee, ThreadState* state) {
  const auto* chain = static_cast<const ChainedTrampoline*>(callee);

  // Capture the out-parameter BEFORE running the primary: x0 doubles as the
  // return register, so the primary trampoline overwrites the argument regs.
  auto [window, out_buffer, dirty_bounds] = GuestParamsValues<PFN_ANativeWindowLock>(state);
  UNUSED(window, dirty_bounds);
  auto* buffer = static_cast<ANativeWindowBuffer*>(out_buffer);

  chain->marshal_and_call(chain->thunk, state);

  auto&& [ret] = GuestReturnReference<PFN_ANativeWindowLock>(state);
  if (ret != 0 || buffer == nullptr || buffer->stride != 0) {
    return;
  }
  int32_t stride = LumaStrideFor(buffer->format, buffer->width);
  if (stride == 0) {
    return;  // Not a layout we can state a stride for; leave it as the host set it.
  }
  TRACE("digitalis ANativeWindow_lock: host left stride unset for format 0x%x (%dx%d); "
        "using mandated luma stride %d",
        buffer->format,
        buffer->width,
        buffer->height,
        stride);
  buffer->stride = stride;
}

const KnownTrampoline kDigitalisANativeWindowLockOverride[] = {
    {"ANativeWindow_lock", DoDigitalisANativeWindowLock, nullptr},
};

const KnownTrampoline kDigitalisANativeWindowLockOverrideForLibandroid[] = {
    {"ANativeWindow_lock", DoDigitalisANativeWindowLock, nullptr},
};

const KnownTrampoline kDigitalisExtraLibnativewindowTrampolines[] = {
    {"ANativeWindow_setPerformInterceptor", DoStub_ANativeWindow_setPerformInterceptor, nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libnativewindow.so",
                                     kDigitalisExtraLibnativewindowTrampolines)

// ANativeWindow_lock is exported from both libnativewindow.so and the libandroid.so
// NDK aggregate, and an app may resolve it from either; override it in each. The
// tables are separate arrays because the registration macro derives its generated
// symbol name from the table's name.
REGISTER_DIGITALIS_EXTRA_TRAMPOLINE_OVERRIDES("libnativewindow.so",
                                              kDigitalisANativeWindowLockOverride)
REGISTER_DIGITALIS_EXTRA_TRAMPOLINE_OVERRIDES("libandroid.so",
                                              kDigitalisANativeWindowLockOverrideForLibandroid)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
