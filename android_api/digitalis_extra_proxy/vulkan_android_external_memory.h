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

#ifndef BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_VULKAN_ANDROID_EXTERNAL_MEMORY_H_
#define BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_VULKAN_ANDROID_EXTERNAL_MEMORY_H_

#include <cstdint>

// Detection helper for the Vulkan proxy override in
// digitalis_extra_libvulkan_trampolines.cc, split into its own header so it can
// be unit-tested on the host without a device (it is pure pointer arithmetic
// over a VkBaseInStructure-shaped chain; no guest-state translation).
//
// Background: the generated libvulkan proxy
// (native_bridge_support/.../libvulkan/proxy) knows only the Vulkan structs its
// registry was generated from. The entire
// VK_ANDROID_external_memory_android_hardware_buffer extension was filtered out
// of that registry, so none of its pNext structs are known. When a pNext chain
// contains any struct the proxy does not know, ConvertOptionalStructures takes
// its rebuild path and its `default: continue;` silently DROPS the unknown
// struct from the chain handed to the host driver. Against gfxstream's encoder
// that is masked; against a real in-process ICD (e.g. Mesa ANV on a bare-metal
// device) the driver then allocates/creates without the AHB import/external
// format and later faults dereferencing uninitialized external-image state.
//
// This helper lets the override recognize the exact structs that get dropped so
// it can forward the affected call with its chain intact.

namespace berberis {

// VkBaseInStructure layout: a VkStructureType (uint32_t) followed by a pointer
// to the next struct. Under LP64 the pointer is 8-byte aligned, so p_next sits
// at offset 8 with 4 bytes of padding after s_type — matched here implicitly by
// the compiler's natural alignment of the pointer member.
struct VulkanStructHeader {
  uint32_t s_type;
  const VulkanStructHeader* p_next;
};

// VkStructureType values for VK_ANDROID_external_memory_android_hardware_buffer.
// Kept as a plain enum of the numeric values from vulkan_core.h so the header
// pulls in no Vulkan SDK dependency.
enum : uint32_t {
  kVkStructureTypeAndroidHardwareBufferUsageAndroid = 1000129000,
  kVkStructureTypeAndroidHardwareBufferPropertiesAndroid = 1000129001,
  kVkStructureTypeAndroidHardwareBufferFormatPropertiesAndroid = 1000129002,
  kVkStructureTypeImportAndroidHardwareBufferInfoAndroid = 1000129003,
  kVkStructureTypeMemoryGetAndroidHardwareBufferInfoAndroid = 1000129004,
  kVkStructureTypeExternalFormatAndroid = 1000129005,
  kVkStructureTypeAndroidHardwareBufferFormatProperties2Android = 1000129006,
};

// True if any struct in the chain rooted at `header` (inclusive) is one of the
// AndroidHardwareBuffer external-memory structs the proxy does not know and
// would therefore drop. `header` may be nullptr (empty chain -> false).
inline bool VulkanChainHasAndroidExternalMemoryStruct(const VulkanStructHeader* header) {
  for (const VulkanStructHeader* s = header; s != nullptr; s = s->p_next) {
    switch (s->s_type) {
      case kVkStructureTypeAndroidHardwareBufferUsageAndroid:
      case kVkStructureTypeAndroidHardwareBufferPropertiesAndroid:
      case kVkStructureTypeAndroidHardwareBufferFormatPropertiesAndroid:
      case kVkStructureTypeImportAndroidHardwareBufferInfoAndroid:
      case kVkStructureTypeMemoryGetAndroidHardwareBufferInfoAndroid:
      case kVkStructureTypeExternalFormatAndroid:
      case kVkStructureTypeAndroidHardwareBufferFormatProperties2Android:
        return true;
      default:
        break;
    }
  }
  return false;
}

}  // namespace berberis

#endif  // BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_VULKAN_ANDROID_EXTERNAL_MEMORY_H_
