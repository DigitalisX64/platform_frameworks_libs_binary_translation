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

#include "gtest/gtest.h"

#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <thread>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/table_of_tables.h"

namespace {

// Collects the fds in this process that back a TableOfTables "child" default
// memfd (memfd_create("child", ...) shows as "/memfd:child (deleted)" in
// /proc/self/fd). The test process may already hold one such fd from a global
// translation cache, so callers diff a before/after snapshot to isolate the fd
// belonging to a freshly-constructed table.
std::set<int> ChildMemfdSnapshot() {
  std::set<int> fds;
  DIR* dir = opendir("/proc/self/fd");
  if (dir == nullptr) {
    return fds;
  }
  struct dirent* ent;
  while ((ent = readdir(dir)) != nullptr) {
    int fd = atoi(ent->d_name);
    if (fd <= 0) {
      continue;
    }
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    char target[256];
    ssize_t n = readlink(path, target, sizeof(target) - 1);
    if (n <= 0) {
      continue;
    }
    target[n] = '\0';
    if (strstr(target, "memfd:child") != nullptr) {
      fds.insert(fd);
    }
  }
  closedir(dir);
  return fds;
}

TEST(TableOfTables, Smoke) {
  berberis::TableOfTables<berberis::GuestAddr, uintptr_t> tot(42);
  ASSERT_EQ(42U, tot.Get(25));
  ASSERT_EQ(1729U, *tot.Put(25, 1729));
  ASSERT_EQ(1729U, tot.Get(25));
  ASSERT_EQ(42U, tot.Get(255));
  ASSERT_EQ(42U, tot.Get((25 << 16) | 25));
}

TEST(TableOfTables, GetPointer) {
  berberis::TableOfTables<berberis::GuestAddr, uintptr_t> tot(42);
  ASSERT_EQ(42U, tot.Get(25));
  auto* addr = tot.GetPointer(25);
  ASSERT_EQ(42U, *addr);
  ASSERT_EQ(42U, tot.Get(25));
  ASSERT_EQ(1729U, *tot.Put(25, 1729));
  ASSERT_EQ(1729U, *addr);
  ASSERT_EQ(1729U, tot.Get(25));
  ASSERT_EQ(42U, tot.Get(255));
}

TEST(TableOfTables, Stress) {
  berberis::TableOfTables<berberis::GuestAddr, uintptr_t> tot(42);

  std::thread threads[64];

  for (size_t i = 0; i < 64; ++i) {
    uint32_t base_num = (i % 2 == 0) ? 0 : 65520;
    threads[i] = std::thread(
        [](berberis::TableOfTables<berberis::GuestAddr, uintptr_t>* tot, uint32_t base) {
          for (uint32_t page_num = 0; page_num < 4098; ++page_num) {
            uint32_t page = (page_num << 17);
            ASSERT_EQ(42U, tot->Get(page | (base + 4)));
            auto* addr = tot->GetPointer(page | (base + 5));
            ASSERT_EQ(42U, tot->Get(page | (base + 4)));
            ASSERT_EQ(1729U, *tot->Put(page | (base + 5), 1729));
            ASSERT_EQ(1U, *tot->Put(page | (base + 6), 1));
            ASSERT_EQ(42U, tot->Get(page | (base + 4)));
            ASSERT_EQ(1729U, *addr);
          }
        },
        &tot,
        base_num);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (uint32_t page_num = 0; page_num < 4098; ++page_num) {
    uint32_t page = (page_num << 17);
    ASSERT_EQ(1729U, tot.Get(page | 5));
    ASSERT_EQ(1U, tot.Get(page | 6));
    ASSERT_EQ(42U, tot.Get(page | 4));
    ASSERT_EQ(42U, tot.Get(page | 255));

    ASSERT_EQ(1729U, tot.Get(page | 65525));
    ASSERT_EQ(1U, tot.Get(page | 65526));
    ASSERT_EQ(42U, tot.Get(page | 65524));
    ASSERT_EQ(42U, tot.Get(page | 65535));
  }
}

// Simulates a legal guest dup2/dup3 landing on the translator's cached default
// memfd number: the kernel silently replaces our memfd with an unrelated file
// description. Without self-validation the next child-table allocation would
// mmap that file's contents as a translation table (here: a zero-filled file,
// so the child would read 0 instead of the default value). The fix re-fstats
// the cached memfd and heals by recreating it when its (st_dev, st_ino)
// identity no longer matches, so the child table stays correctly default-filled.
TEST(TableOfTables, HealsReplacedDefaultMemfd) {
  constexpr uintptr_t kDefault = 42;

  // Isolate the child memfd of *this* table: snapshot before/after construction
  // (the process may already hold a global translation-cache child memfd).
  std::set<int> before = ChildMemfdSnapshot();
  berberis::TableOfTables<berberis::GuestAddr, uintptr_t> tot(kDefault);
  std::set<int> after = ChildMemfdSnapshot();

  int child_fd = -1;
  for (int fd : after) {
    if (before.find(fd) == before.end()) {
      child_fd = fd;
      break;
    }
  }
  ASSERT_GT(child_fd, 0) << "could not locate this table's cached default memfd";

  // Learn the memfd's size so the replacement is large enough to back a full
  // child table (avoids SIGBUS in the unhealed path, giving a clean wrong-value
  // failure instead of a crash).
  struct stat st;
  ASSERT_EQ(fstat(child_fd, &st), 0);
  const off_t region_size = st.st_size;
  ASSERT_GT(region_size, 0);

  // Steal the fd number with an unrelated, zero-filled file.
  int repl = syscall(SYS_memfd_create, "stolen", 0u);
  ASSERT_GT(repl, 0);
  ASSERT_EQ(ftruncate(repl, region_size), 0);
  ASSERT_EQ(dup2(repl, child_fd), child_fd);
  close(repl);  // child_fd now solely references the zero-filled file.

  // First child-table allocation after the yank. With the fix this heals and
  // fills with kDefault; without it the child reads the zero-filled stolen file.
  ASSERT_EQ(kDefault, tot.Get(0x5));
  ASSERT_EQ(kDefault, *tot.GetPointer(0x5));
  // A subsequent write/read through the healed table stays coherent.
  ASSERT_EQ(1729U, *tot.Put(0x5, 1729));
  ASSERT_EQ(1729U, tot.Get(0x5));
}

TEST(TableOfTables_DeathTest, InvalidAddress) {
#ifdef BERBERIS_GUEST_LP64
  berberis::TableOfTables<berberis::GuestAddr, uintptr_t> tot(42);

  // region digitalis - Digitalis masks PAC/TBI pointer tags (bits 48-63) out of
  // the key before splitting (see SplitKey in table_of_tables.h) so a guest PC
  // carrying pointer tags does not trip the table-range check. An address with
  // nonzero top bits therefore no longer dies; it aliases to the masked slot and
  // reads the default value. (Upstream expected EXPECT_DEATH here.)
  EXPECT_EQ(42U, tot.Get(0xdead'beef'1234'5678ULL));
  // endregion
#endif
}

}  // namespace
