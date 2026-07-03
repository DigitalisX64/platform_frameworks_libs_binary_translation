/*
 * Copyright (C) 2015 The Android Open Source Project
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

#include "berberis/runtime_primitives/exec_region_anonymous.h"

#include <sys/mman.h>

#include "berberis/base/fd.h"
#include "berberis/base/mmap.h"

namespace berberis {

ExecRegion ExecRegionAnonymousFactory::Create(size_t size) {
  size = AlignUpPageSize(size);

  auto fd = CreateMemfdOrDie("exec");
  // region digitalis - tag as host-owned so a concurrent guest fd sweep can't
  // close the fd between here and the maps below; see fd.h.
  TagHostOwnedFdUnsafe(fd);
  // endregion
  FtruncateOrDie(fd, static_cast<off64_t>(size));

#if defined(__x86_64__)
  constexpr int kBerberisFlags = kMmapBerberis32Bit;
#else
  // TODO(b/363611588): enable for other backends (arm64/riscv64)
  constexpr int kBerberisFlags = 0;
#endif  // defined(__x86_64__)

  ExecRegion result{
      static_cast<uint8_t*>(MmapImplOrDie({.size = size,
                                           .prot = PROT_READ | PROT_EXEC,
                                           .flags = MAP_SHARED,
                                           .fd = fd,
                                           .berberis_flags = kBerberisFlags})),
      static_cast<uint8_t*>(MmapImplOrDie(
          {.size = size, .prot = PROT_READ | PROT_WRITE, .flags = MAP_SHARED, .fd = fd})),
      size};

  // region digitalis - close with the tag so the fdsan slot is left clean.
  // CloseUnsafe(fd);
  CloseHostOwnedFdUnsafe(fd);
  // endregion
  return result;
}

}  // namespace berberis
