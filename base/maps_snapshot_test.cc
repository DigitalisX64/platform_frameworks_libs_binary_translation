/*
 * Copyright (C) 2024 The Android Open Source Project
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

#include "sys/mman.h"

#include <cstdint>
#include <cstdio>
#include <cstring>  // strncmp
#include <memory>
#include <string>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/base/maps_snapshot.h"

namespace berberis {

namespace {

void Foo() {};

TEST(MapsSnapshot, Basic) {
  auto* maps_snapshot = MapsSnapshot::GetInstance();

  maps_snapshot->ClearForTesting();

  // No mappings can be found before snapshot is taken by Update().
  auto no_mappings_result = maps_snapshot->FindMappedObjectName(bit_cast<uintptr_t>(&Foo));
  ASSERT_FALSE(no_mappings_result.has_value());

  maps_snapshot->Update();

  auto result = maps_snapshot->FindMappedObjectName(bit_cast<uintptr_t>(&Foo));
  ASSERT_TRUE(result.has_value());
  ASSERT_FALSE(result.value().empty());
}

TEST(MapsSnapshot, AnonymousMapping) {
  auto* maps_snapshot = MapsSnapshot::GetInstance();

  void* addr = mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  ASSERT_NE(addr, MAP_FAILED);
  maps_snapshot->Update();
  auto result = maps_snapshot->FindMappedObjectName(bit_cast<uintptr_t>(addr));
  munmap(addr, 4096);

  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result.value().empty());
}

std::tuple<uintptr_t, std::string> GetAddressOfFirstMappingWithSubstring(std::string substr) {
  std::unique_ptr<FILE, decltype(&fclose)> maps_file(fopen("/proc/self/maps", "r"), fclose);
  if (maps_file == nullptr) {
    ADD_FAILURE() << "Cannot open /proc/self/maps";
    return {0, ""};
  }

  char line[512], pathname[256];
  uintptr_t start;
  while (fgets(line, sizeof(line), maps_file.get())) {
    // Maximum string size 255 so that we have space for the terminating '\0'.
    int match_count = sscanf(
        line, "%" SCNxPTR "-%*" SCNxPTR " %*s %*lx %*x:%*x %*lu%*[ ]%255s", &start, pathname);
    if (match_count == 2) {
      std::string current_pathname(pathname);
      if (current_pathname.find(substr) != current_pathname.npos) {
        return {start, current_pathname};
      }
    }
  }
  ADD_FAILURE() << "Cannot find " << substr << " in /proc/self/maps";
  return {0, ""};
}

TEST(MapsSnapshot, ExactFilenameMatch) {
  auto* maps_snapshot = MapsSnapshot::GetInstance();

  // Take some object that must be mapped already and is unlikely to be suddenly unmapped. "libc.so"
  // may have a version suffix like "libc-2.19.so", which would make parsing too challenging for
  // what this test requires. We don't want to search just for "libc" either since it's likely to
  // match an unrelated library. "libc++.so" is taken from the local build
  // (out/host/linux-x86/lib64/libc++.so) so we should be able to find it.
  auto [addr, pathname] = GetAddressOfFirstMappingWithSubstring("libc++.so");
  ASSERT_GT(addr, 0u);

  maps_snapshot->Update();
  auto result = maps_snapshot->FindMappedObjectName(bit_cast<uintptr_t>(addr));

  ASSERT_TRUE(result.has_value());
  // MapsSnapshot only stores first 255 symbols plus terminating null.
  ASSERT_TRUE(strncmp(result.value().c_str(), pathname.c_str(), 255) == 0);
}

}  // namespace

}  // namespace berberis
