/*
 * Copyright (C) 2022 The Android Open Source Project
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

#include "berberis/runtime_primitives/exec_region_elf_backed.h"
#include "berberis/tiny_loader/tiny_loader.h"

#include <android/dlext.h>
#include <dlfcn.h>
#include <sys/mman.h>

#include "berberis/base/bit_util.h"
#include "berberis/base/fd.h"
#include "berberis/base/mmap.h"

// Note that we have to use absolute path for ANDROID_DLEXT_FORCE_LOAD to work correctly
// otherwise searching by soname will trigger and the flag will have no effect.
#if defined(__LP64__)
const constexpr char* kExecRegionLibraryPath = "/system/lib64/libberberis_exec_region.so";
#else
const constexpr char* kExecRegionLibraryPath = "/system/lib/libberberis_exec_region.so";
#endif

const constexpr char* kRegionStartSymbolName = "exec_region_start";
const constexpr char* kRegionEndSymbolName = "exec_region_end";

namespace berberis {

ExecRegion ExecRegionElfBackedFactory::Create(size_t size) {
  size = AlignUpPageSize(size);

  // Since we cannot force android loader to map library in lower 2G memory we will need
  // to reserve the space first and then direct the loader to load the library at that address.
  size_t load_size = TinyLoader::CalculateLoadSize(kExecRegionLibraryPath, nullptr);
  CHECK_NE(load_size, 0);

  void* load_addr = MmapImplOrDie({.addr = nullptr,
                                   .size = load_size,
                                   .prot = PROT_NONE,
                                   .flags = MAP_ANONYMOUS | MAP_PRIVATE,
                                   .berberis_flags = kMmapBerberis32Bit});

  android_dlextinfo dlextinfo{.flags = ANDROID_DLEXT_FORCE_LOAD | ANDROID_DLEXT_RESERVED_ADDRESS,
                              .reserved_addr = load_addr,
                              .reserved_size = load_size};

  void* handle = android_dlopen_ext(kExecRegionLibraryPath, RTLD_NOW, &dlextinfo);
  if (handle == nullptr) {
    FATAL("Couldn't load \"%s\": %s", kExecRegionLibraryPath, dlerror());
  }
  void* region_start = dlsym(handle, kRegionStartSymbolName);
  CHECK(region_start != nullptr);
  auto region_start_addr = bit_cast<uintptr_t>(region_start);
  CHECK(region_start_addr % kPageSize == 0);

  void* region_end = dlsym(handle, kRegionEndSymbolName);
  CHECK(region_end != nullptr);
  auto region_end_addr = bit_cast<uintptr_t>(region_end);
  CHECK(region_end_addr % kPageSize == 0);
  size_t region_size = region_end_addr - region_start_addr;
  CHECK_GE(region_size, size);

  auto fd = CreateMemfdOrDie("exec");
  // region digitalis - tag as host-owned so a concurrent guest fd sweep can't
  // close the fd between here and the maps below; see fd.h.
  TagHostOwnedFdUnsafe(fd);
  // endregion
  FtruncateOrDie(fd, static_cast<off64_t>(region_size));

  ExecRegion result{
      static_cast<uint8_t*>(MmapImplOrDie({.addr = region_start,
                                           .size = region_size,
                                           .prot = PROT_READ | PROT_EXEC,
                                           .flags = MAP_FIXED | MAP_SHARED,
                                           .fd = fd})),
      static_cast<uint8_t*>(MmapImplOrDie(
          {.size = region_size, .prot = PROT_READ | PROT_WRITE, .flags = MAP_SHARED, .fd = fd})),
      region_size};

  // region digitalis - close with the tag so the fdsan slot is left clean.
  // CloseUnsafe(fd);
  CloseHostOwnedFdUnsafe(fd);
  // endregion
  return result;
}

}  // namespace berberis
