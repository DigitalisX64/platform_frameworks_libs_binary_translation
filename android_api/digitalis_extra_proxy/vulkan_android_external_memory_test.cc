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

#include "gtest/gtest.h"

#include <cstdint>

#include "vulkan_android_external_memory.h"

namespace berberis {

namespace {

// A few real Vulkan core sTypes the proxy DOES know, used to build "compatible"
// chains that must not be mistaken for AHB chains.
constexpr uint32_t kVkStructureTypeMemoryAllocateInfo = 5;
constexpr uint32_t kVkStructureTypeImageCreateInfo = 14;
constexpr uint32_t kVkStructureTypeMemoryDedicatedAllocateInfo = 1000127001;
constexpr uint32_t kVkStructureTypeExportMemoryAllocateInfo = 1000072002;

TEST(VulkanAndroidExternalMemory, EmptyChainIsNotAhb) {
  EXPECT_FALSE(VulkanChainHasAndroidExternalMemoryStruct(nullptr));
}

TEST(VulkanAndroidExternalMemory, SingleNonAhbStructIsNotAhb) {
  VulkanStructHeader only{kVkStructureTypeMemoryAllocateInfo, nullptr};
  EXPECT_FALSE(VulkanChainHasAndroidExternalMemoryStruct(&only));
}

TEST(VulkanAndroidExternalMemory, ChainOfOnlyKnownStructsIsNotAhb) {
  // VkMemoryAllocateInfo -> VkMemoryDedicatedAllocateInfo -> VkExportMemoryAllocateInfo
  VulkanStructHeader exp{kVkStructureTypeExportMemoryAllocateInfo, nullptr};
  VulkanStructHeader dedicated{kVkStructureTypeMemoryDedicatedAllocateInfo, &exp};
  VulkanStructHeader alloc{kVkStructureTypeMemoryAllocateInfo, &dedicated};
  EXPECT_FALSE(VulkanChainHasAndroidExternalMemoryStruct(&alloc));
}

TEST(VulkanAndroidExternalMemory, ImportAhbInPNextIsDetected) {
  // The vkAllocateMemory case: VkMemoryAllocateInfo -> VkImportAndroidHardwareBufferInfoANDROID.
  VulkanStructHeader import{kVkStructureTypeImportAndroidHardwareBufferInfoAndroid, nullptr};
  VulkanStructHeader alloc{kVkStructureTypeMemoryAllocateInfo, &import};
  EXPECT_TRUE(VulkanChainHasAndroidExternalMemoryStruct(&alloc));
}

TEST(VulkanAndroidExternalMemory, ExternalFormatInPNextIsDetected) {
  // The vkCreateImage case: VkImageCreateInfo -> VkExternalFormatANDROID.
  VulkanStructHeader external_format{kVkStructureTypeExternalFormatAndroid, nullptr};
  VulkanStructHeader image{kVkStructureTypeImageCreateInfo, &external_format};
  EXPECT_TRUE(VulkanChainHasAndroidExternalMemoryStruct(&image));
}

TEST(VulkanAndroidExternalMemory, AhbStructDeepInChainIsDetected) {
  // Known head, known middle, AHB tail: the walker must reach the end.
  VulkanStructHeader usage{kVkStructureTypeAndroidHardwareBufferUsageAndroid, nullptr};
  VulkanStructHeader dedicated{kVkStructureTypeMemoryDedicatedAllocateInfo, &usage};
  VulkanStructHeader image{kVkStructureTypeImageCreateInfo, &dedicated};
  EXPECT_TRUE(VulkanChainHasAndroidExternalMemoryStruct(&image));
}

TEST(VulkanAndroidExternalMemory, AhbStructAtHeadIsDetected) {
  VulkanStructHeader head{kVkStructureTypeAndroidHardwareBufferPropertiesAndroid, nullptr};
  EXPECT_TRUE(VulkanChainHasAndroidExternalMemoryStruct(&head));
}

TEST(VulkanAndroidExternalMemory, EveryAhbSTypeIsDetected) {
  const uint32_t kAhbTypes[] = {
      kVkStructureTypeAndroidHardwareBufferUsageAndroid,
      kVkStructureTypeAndroidHardwareBufferPropertiesAndroid,
      kVkStructureTypeAndroidHardwareBufferFormatPropertiesAndroid,
      kVkStructureTypeImportAndroidHardwareBufferInfoAndroid,
      kVkStructureTypeMemoryGetAndroidHardwareBufferInfoAndroid,
      kVkStructureTypeExternalFormatAndroid,
      kVkStructureTypeAndroidHardwareBufferFormatProperties2Android,
  };
  for (uint32_t s_type : kAhbTypes) {
    VulkanStructHeader head{s_type, nullptr};
    EXPECT_TRUE(VulkanChainHasAndroidExternalMemoryStruct(&head))
        << "sType " << s_type << " should be detected as an AHB external-memory struct";
  }
}

}  // namespace

}  // namespace berberis
