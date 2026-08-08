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

#include <dlfcn.h>

#include <cstdint>
#include <mutex>

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

// Size of one luma sample in bytes, or 0 for a format whose stride we do not
// repair. AHardwareBuffer_lockPlanes reports row strides in bytes while
// ANativeWindow_Buffer::stride counts pixels, so the two differ for Y16.
int32_t LumaBytesPerPixel(int32_t format) {
  switch (format) {
    case kHalPixelFormatYV12:
    case kHalPixelFormatYCrCb420SP:
    case kHalPixelFormatY8:
      return 1;
    case kHalPixelFormatY16:
      return 2;
    default:
      return 0;
  }
}

// The luma row stride the format's own layout rules call for. YV12 specifies a
// 16-pixel-aligned luma stride; the other planar/single-plane Y layouts are
// tightly packed. This is only a last resort -- a gralloc is free to pad rows
// further than the format requires, so it is a guess where asking is not.
int32_t SpecLumaStride(int32_t format, int32_t width) {
  return (format == kHalPixelFormatYV12) ? ((width + 15) & ~15) : width;
}

// AHardwareBuffer_Desc, _Plane and _Planes from android/hardware_buffer.h.
// Declared here rather than included because the functions below are reached by
// dlsym -- this library must keep working on a host that does not export them.
struct AHardwareBufferDesc {
  uint32_t width;
  uint32_t height;
  uint32_t layers;
  uint32_t format;
  uint64_t usage;
  uint32_t stride;
  uint32_t rfu0;
  uint64_t rfu1;
};

struct AHardwareBufferPlane {
  void* data;
  uint32_t pixel_stride;
  uint32_t row_stride;
};

struct AHardwareBufferPlanes {
  uint32_t plane_count;
  AHardwareBufferPlane planes[4];
};

// AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN:
// the usage ANativeWindow_lock itself forces onto the window before dequeuing.
constexpr uint64_t kUsageCpuReadWriteOften = 0x3ULL | (0x3ULL << 4);

// NATIVE_WINDOW_CONSUMER_USAGE_BITS, from system/window.h.
constexpr int kNativeWindowConsumerUsageBits = 10;

// The usage a buffer of this window is allocated with: what the consumer on the
// far end of the BufferQueue asks for, plus the CPU access ANativeWindow_lock
// adds on the producer side. Included because gralloc picks row alignment from
// the whole description, not the geometry alone, so a scratch buffer that omits
// the consumer's bits is not necessarily laid out like the real one. Windows
// whose consumer usage cannot be queried fall back to the CPU bits.
//
// The consumer bits live on the far side of the BufferQueue, so this is a
// binder round trip -- call it once per window, never per frame.
uint64_t WindowBufferUsage(void* window) {
  static auto query = reinterpret_cast<int (*)(const void*, int, int*)>(  //
      dlsym(RTLD_DEFAULT, "ANativeWindow_query"));
  int consumer_usage = 0;
  if (query == nullptr || query(window, kNativeWindowConsumerUsageBits, &consumer_usage) != 0) {
    return kUsageCpuReadWriteOften;
  }
  return kUsageCpuReadWriteOften | static_cast<uint32_t>(consumer_usage);
}

// Ask the host gralloc what luma row stride it really gives a buffer of this
// format and geometry, in pixels, or 0 if it will not say.
//
// The stride is a property of the allocator, not of the format: how far a row
// is padded past `width` varies with the gralloc and with the buffer's usage,
// and guessing wrong shears every row. So allocate a throwaway buffer with the
// same description and read the stride the allocator chose for it.
int32_t QueryHostLumaStride(int32_t format, int32_t width, int32_t height, uint64_t usage) {
  static auto allocate = reinterpret_cast<int (*)(const AHardwareBufferDesc*, void**)>(
      dlsym(RTLD_DEFAULT, "AHardwareBuffer_allocate"));
  static auto release = reinterpret_cast<void (*)(void*)>(  //
      dlsym(RTLD_DEFAULT, "AHardwareBuffer_release"));
  static auto lock_planes =
      reinterpret_cast<int (*)(void*, uint64_t, int32_t, const void*, AHardwareBufferPlanes*)>(
          dlsym(RTLD_DEFAULT, "AHardwareBuffer_lockPlanes"));
  static auto unlock = reinterpret_cast<int (*)(void*, int*)>(  //
      dlsym(RTLD_DEFAULT, "AHardwareBuffer_unlock"));
  static auto describe = reinterpret_cast<void (*)(const void*, AHardwareBufferDesc*)>(
      dlsym(RTLD_DEFAULT, "AHardwareBuffer_describe"));

  if (allocate == nullptr || release == nullptr) {
    return 0;
  }
  AHardwareBufferDesc desc = {};
  desc.width = static_cast<uint32_t>(width);
  desc.height = static_cast<uint32_t>(height);
  desc.layers = 1;
  desc.format = static_cast<uint32_t>(format);
  desc.usage = usage;
  void* buffer = nullptr;
  if (allocate(&desc, &buffer) != 0 || buffer == nullptr) {
    return 0;
  }

  int32_t stride = 0;
  // Prefer the per-plane row stride: it is the one number gralloc always knows
  // for a planar layout, and it is precisely what the buffer's single `stride`
  // field fails to carry (which is why it arrives here as zero).
  if (lock_planes != nullptr && unlock != nullptr) {
    AHardwareBufferPlanes planes = {};
    if (lock_planes(buffer, desc.usage, -1, nullptr, &planes) == 0) {
      if (planes.plane_count > 0) {
        stride = static_cast<int32_t>(planes.planes[0].row_stride) / LumaBytesPerPixel(format);
      }
      unlock(buffer, nullptr);
    }
  }
  if (stride == 0 && describe != nullptr) {
    AHardwareBufferDesc allocated = {};
    describe(buffer, &allocated);
    stride = static_cast<int32_t>(allocated.stride);
  }
  release(buffer);
  return stride;
}

// The luma row stride to report for a buffer this window just handed back, in
// pixels, or 0 when we cannot state one.
//
// Cached, keyed by the window as well as the buffer description: this runs once
// per locked frame, and both of the questions behind the answer -- the window's
// usage and the allocator's stride for it -- cost far too much to ask per frame.
int32_t LumaStrideFor(void* window, int32_t format, int32_t width, int32_t height) {
  if (width <= 0 || height <= 0 || LumaBytesPerPixel(format) == 0) {
    return 0;
  }
  struct Entry {
    void* window;
    int32_t format;
    int32_t width;
    int32_t height;
    int32_t stride;
  };
  // A window changes geometry rarely; a handful of entries covers every surface
  // an app has open, and the table is only appended to.
  constexpr size_t kCacheCapacity = 16;
  static Entry cache[kCacheCapacity];
  static size_t cache_size = 0;
  static std::mutex mutex;

  std::lock_guard<std::mutex> lock(mutex);
  for (size_t i = 0; i < cache_size; ++i) {
    if (cache[i].window == window && cache[i].format == format && cache[i].width == width &&
        cache[i].height == height) {
      return cache[i].stride;
    }
  }

  const uint64_t usage = WindowBufferUsage(window);
  int32_t stride = QueryHostLumaStride(format, width, height, usage);
  if (stride >= width) {
    TRACE(
        "digitalis ANativeWindow_lock: host gralloc uses luma stride %d for format 0x%x (%dx%d) "
        "usage 0x%llx",
        stride,
        format,
        width,
        height,
        static_cast<unsigned long long>(usage));
  } else {
    // Either the host does not export the AHardwareBuffer entry points or it
    // refused this description. The format's own rule is the best guess left.
    stride = SpecLumaStride(format, width);
    TRACE(
        "digitalis ANativeWindow_lock: host gralloc would not report a luma stride for format 0x%x "
        "(%dx%d) usage 0x%llx; assuming the %d the format specifies",
        format,
        width,
        height,
        static_cast<unsigned long long>(usage),
        stride);
  }
  if (cache_size < kCacheCapacity) {
    cache[cache_size++] = {window, format, width, height, stride};
  }
  return stride;
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
// the stride unset, substitute the one that host's own allocator uses. A host
// that does fill the field is left completely alone.
void DoDigitalisANativeWindowLock(HostCode callee, ThreadState* state) {
  const auto* chain = static_cast<const ChainedTrampoline*>(callee);

  // Capture the out-parameter BEFORE running the primary: x0 doubles as the
  // return register, so the primary trampoline overwrites the argument regs.
  auto [window, out_buffer, dirty_bounds] = GuestParamsValues<PFN_ANativeWindowLock>(state);
  UNUSED(dirty_bounds);
  void* locked_window = window;
  auto* buffer = static_cast<ANativeWindowBuffer*>(out_buffer);

  chain->marshal_and_call(chain->thunk, state);

  auto&& [ret] = GuestReturnReference<PFN_ANativeWindowLock>(state);
  if (ret != 0 || buffer == nullptr || buffer->stride != 0) {
    return;
  }
  int32_t stride = LumaStrideFor(locked_window, buffer->format, buffer->width, buffer->height);
  if (stride == 0) {
    return;  // Not a layout we can state a stride for; leave it as the host set it.
  }
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
