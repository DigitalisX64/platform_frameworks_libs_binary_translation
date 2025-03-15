/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include "berberis/runtime_primitives/code_pool.h"

#include <cstring>
#include <mutex>

#include "berberis/base/forever_alloc.h"
#include "berberis/runtime_primitives/exec_region_anonymous.h"

#if defined(__BIONIC__)
#include "berberis/runtime_primitives/exec_region_elf_backed.h"
#endif

namespace berberis {

void* DataPool::AddRaw(const void* ptr, uint32_t size) {
  void* result;
  // Take the lock to allocate in arena.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    result = arena_.Alloc(size, /* align = */ 16);
  }
  memcpy(result, ptr, size);
  return result;
}

void ResetAllExecRegions() {
  GetDefaultCodePoolInstance()->ResetExecRegion();
  GetColdCodePoolInstance()->ResetExecRegion();
  GetFunctionWrapperCodePoolInstance()->ResetExecRegion();
}

CodePool<ExecRegionAnonymousFactory>* GetDefaultCodePoolInstance() {
  static auto* g_code_pool = NewForever<CodePool<ExecRegionAnonymousFactory>>();
  return g_code_pool;
}

CodePool<ExecRegionAnonymousFactory>* GetColdCodePoolInstance() {
  static auto* g_cold_code_pool = NewForever<CodePool<ExecRegionAnonymousFactory>>();
  return g_cold_code_pool;
}

#if defined(__BIONIC__)
CodePool<ExecRegionElfBackedFactory>* GetFunctionWrapperCodePoolInstance() {
  static auto* g_code_pool = NewForever<CodePool<ExecRegionElfBackedFactory>>();
  return g_code_pool;
}
#else
CodePool<ExecRegionAnonymousFactory>* GetFunctionWrapperCodePoolInstance() {
  return GetDefaultCodePoolInstance();
}
#endif

DataPool* DataPool::GetInstance() {
  static auto* g_data_pool = NewForever<DataPool>();
  return g_data_pool;
}

}  // namespace berberis
