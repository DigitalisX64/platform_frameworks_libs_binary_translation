/*
 * Copyright (C) 2025 The Android Open Source Project
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

#ifndef BERBERIS_MACHINE_INSN_INFO_COMMON_MACHINE_INSN_INFO_H_
#define BERBERIS_MACHINE_INSN_INFO_COMMON_MACHINE_INSN_INFO_H_

#include <cstdint>

namespace berberis::machine_insn_info_backend {

class Mem8 {
 public:
  using Type = uint8_t;
};

class Mem16 {
 public:
  using Type = uint16_t;
};

class Mem32 {
 public:
  using Type = uint32_t;
};

class Mem64 {
 public:
  using Type = uint64_t;
};

}  // namespace berberis::machine_insn_info_backend

#endif  // BERBERIS_MACHINE_INSN_INFO_COMMON_MACHINE_INSN_INFO_H_
