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

// Digitalis-side override of the libvulkan proxy calls that carry
// VK_ANDROID_external_memory_android_hardware_buffer structs in a pNext chain.
//
// Why this exists: the generated libvulkan proxy
// (native_bridge_support/.../libvulkan/proxy) was generated from a Vulkan
// registry with the AndroidHardwareBuffer external-memory extension filtered
// out, so it knows none of that extension's structs. Its pNext walker
// (ConvertOptionalStructures) rebuilds any chain that contains an unknown struct
// and, in the rebuild, its `default: continue;` SILENTLY DROPS the unknown
// struct. So on `vkAllocateMemory` the guest's VkImportAndroidHardwareBufferInfo-
// ANDROID is stripped, on `vkCreateImage` the VkExternalFormatANDROID is
// stripped, and on `vkGetAndroidHardwareBufferPropertiesANDROID` the output
// VkAndroidHardwareBufferFormatProperties[2]ANDROID is stripped before/after the
// host driver sees it.
//
// Against gfxstream's guest ICD (a command encoder that re-serializes structs)
// the drop is masked. Against a real in-process ICD — e.g. Mesa ANV on a
// bare-metal x86 device, the Drion configuration — the driver allocates/creates
// without the AHB import/external format and then faults dereferencing
// uninitialized external-image state on first use (observed as a SIGSEGV deep
// inside vulkan.intel.so during Unity's first frame).
//
// The fix forwards the affected calls with their pNext chain INTACT. Under LP64
// every struct on these chains is layout-identical between the arm64 guest and
// the x86-64 host and carries only plain data or host handles (the
// AHardwareBuffer* inside VkImportAndroidHardwareBufferInfoANDROID is already a
// host object, produced by the proxied AHardwareBuffer_allocate), and guest
// memory shares the host address space, so handing the host driver the guest
// pointer directly is correct.
//
// Blast radius is kept minimal: vkAllocateMemory / vkCreateImage /
// vkCreateSamplerYcbcrConversion are called by every Vulkan app, so the override
// only takes its own path when the chain actually carries a dropped AHB struct
// AND the app used the default allocator (pAllocator == nullptr, the Unity
// case); every other call delegates unchanged to the proven upstream trampoline.
// The two AHB-only getters always forward directly — they exist solely for this
// extension, so there is no non-AHB path to protect.

#if defined(__x86_64__)

#include <cstdint>

#include "berberis/base/tracing.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/proxy_loader/proxy_library_builder.h"
#include "berberis/runtime_primitives/host_code.h"

#include "register_extra_trampolines.h"
#include "vulkan_android_external_memory.h"

namespace berberis {
namespace {

using VkResult_t = int32_t;

// Handles and struct pointers all marshal as void* — valid under LP64 since the
// guest and host share the address space and the pointee layouts are identical;
// only the register class matters to the trampoline dispatch.
using PFN_vkAllocateMemory =
    VkResult_t (*)(void* device, const void* pAllocateInfo, const void* pAllocator, void* pMemory);
using PFN_vkCreateImage =
    VkResult_t (*)(void* device, const void* pCreateInfo, const void* pAllocator, void* pImage);
using PFN_vkCreateSamplerYcbcrConversion = VkResult_t (*)(void* device,
                                                          const void* pCreateInfo,
                                                          const void* pAllocator,
                                                          void* pYcbcrConversion);
using PFN_vkGetAndroidHardwareBufferPropertiesANDROID =
    VkResult_t (*)(void* device, const void* buffer, void* pProperties);
using PFN_vkGetMemoryAndroidHardwareBufferANDROID =
    VkResult_t (*)(void* device, const void* pInfo, void* pBuffer);

// True if `head` (a VkBaseInStructure-shaped struct) carries an AHB external-
// memory struct the upstream proxy would drop.
bool ChainNeedsAndroidExternalMemoryPassthrough(const void* head) {
  return VulkanChainHasAndroidExternalMemoryStruct(static_cast<const VulkanStructHeader*>(head));
}

// Shared body for the three create/allocate calls that share the signature
// (device, pInfo, pAllocator, pOut). Delegates to the upstream trampoline for
// the common case and only forwards directly when an AHB struct would otherwise
// be dropped and the default allocator is in use.
template <typename PFN>
void ForwardCreateWithAhbChain(const ChainedTrampoline* chain,
                               ThreadState* state,
                               const char* name) {
  auto [device, p_info, p_allocator, p_out] = GuestParamsValues<PFN>(state);
  if (p_allocator != nullptr || !ChainNeedsAndroidExternalMemoryPassthrough(p_info)) {
    // Common/safe path: no dropped AHB struct, or a custom allocator whose guest
    // callbacks the upstream trampoline already wraps. Behavior is unchanged.
    chain->marshal_and_call(chain->thunk, state);
    return;
  }
  // Default allocator + a dropped AHB struct: hand the host driver the full
  // guest chain. pAllocator is nullptr here, so no callback wrapping is needed.
  PFN host_fn = AsFuncPtr(chain->thunk);
  VkResult_t result = host_fn(device, p_info, nullptr, p_out);
  auto&& [ret] = GuestReturnReference<PFN>(state);
  ret = result;
  TRACE("digitalis %s: forwarded AHB pNext chain intact", name);
}

void DoTrampoline_vkAllocateMemory(HostCode callee, ThreadState* state) {
  ForwardCreateWithAhbChain<PFN_vkAllocateMemory>(
      static_cast<const ChainedTrampoline*>(callee), state, "vkAllocateMemory");
}

void DoTrampoline_vkCreateImage(HostCode callee, ThreadState* state) {
  ForwardCreateWithAhbChain<PFN_vkCreateImage>(
      static_cast<const ChainedTrampoline*>(callee), state, "vkCreateImage");
}

void DoTrampoline_vkCreateSamplerYcbcrConversion(HostCode callee, ThreadState* state) {
  ForwardCreateWithAhbChain<PFN_vkCreateSamplerYcbcrConversion>(
      static_cast<const ChainedTrampoline*>(callee), state, "vkCreateSamplerYcbcrConversion");
}

// vkGetAndroidHardwareBufferPropertiesANDROID(device, buffer, pProperties):
// pProperties is an OUTPUT struct whose pNext may carry the proxy-unknown
// VkAndroidHardwareBufferFormatProperties[2]ANDROID that the host fills. If the
// upstream trampoline drops it, the guest's format info stays uninitialized and
// the subsequent vkCreateImage builds a bad external image. Always forward
// directly (AHB-only call, no allocator, no non-AHB path to protect).
void DoTrampoline_vkGetAndroidHardwareBufferPropertiesANDROID(HostCode callee, ThreadState* state) {
  const auto* chain = static_cast<const ChainedTrampoline*>(callee);
  auto [device, buffer, p_properties] =
      GuestParamsValues<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(state);
  PFN_vkGetAndroidHardwareBufferPropertiesANDROID host_fn = AsFuncPtr(chain->thunk);
  VkResult_t result = host_fn(device, buffer, p_properties);
  auto&& [ret] = GuestReturnReference<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(state);
  ret = result;
}

// vkGetMemoryAndroidHardwareBufferANDROID(device, pInfo, pBuffer): pInfo is the
// proxy-unknown VkMemoryGetAndroidHardwareBufferInfoANDROID (plain: a
// VkDeviceMemory handle); pBuffer receives a host AHardwareBuffer*. Always
// forward directly.
void DoTrampoline_vkGetMemoryAndroidHardwareBufferANDROID(HostCode callee, ThreadState* state) {
  const auto* chain = static_cast<const ChainedTrampoline*>(callee);
  auto [device, p_info, p_buffer] =
      GuestParamsValues<PFN_vkGetMemoryAndroidHardwareBufferANDROID>(state);
  PFN_vkGetMemoryAndroidHardwareBufferANDROID host_fn = AsFuncPtr(chain->thunk);
  VkResult_t result = host_fn(device, p_info, p_buffer);
  auto&& [ret] = GuestReturnReference<PFN_vkGetMemoryAndroidHardwareBufferANDROID>(state);
  ret = result;
}

const KnownTrampoline kDigitalisLibVulkanOverrides[] = {
    {"vkAllocateMemory", DoTrampoline_vkAllocateMemory, nullptr},
    {"vkCreateImage", DoTrampoline_vkCreateImage, nullptr},
    {"vkCreateSamplerYcbcrConversion", DoTrampoline_vkCreateSamplerYcbcrConversion, nullptr},
    {"vkGetAndroidHardwareBufferPropertiesANDROID",
     DoTrampoline_vkGetAndroidHardwareBufferPropertiesANDROID, nullptr},
    {"vkGetMemoryAndroidHardwareBufferANDROID",
     DoTrampoline_vkGetMemoryAndroidHardwareBufferANDROID, nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINE_OVERRIDES("libvulkan.so", kDigitalisLibVulkanOverrides)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
