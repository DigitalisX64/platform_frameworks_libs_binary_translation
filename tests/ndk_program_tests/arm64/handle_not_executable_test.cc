/*
 * Copyright (C) 2016 The Android Open Source Project
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

#include <setjmp.h>
#include <sys/mman.h>
#include <unistd.h>  // sysconf(_SC_PAGESIZE)

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "berberis/ndk_program_tests/scoped_sigaction.h"

// Make sure compiler doesn't recognize undefined behavior and doesn't optimize out call to nullptr.
volatile void* g_null_addr = nullptr;

namespace {

TEST(HandleNotExecutable, NotExecutable) {
  uint32_t* code = reinterpret_cast<uint32_t*>(mmap(0,
                                                    sysconf(_SC_PAGESIZE),
                                                    PROT_READ | PROT_WRITE,  // No PROT_EXEC!
                                                    MAP_PRIVATE | MAP_ANONYMOUS,
                                                    -1,
                                                    0));
  using Func = void (*)();
  ASSERT_EXIT((reinterpret_cast<Func>(code))(), testing::KilledBySignal(SIGSEGV), "");
  munmap(code, sysconf(_SC_PAGESIZE));
}

TEST(HandleNotExecutable, PcLessThan4096) {
  using Func = void (*)();
  ASSERT_EXIT((reinterpret_cast<Func>(const_cast<void*>(g_null_addr)))(),
              testing::KilledBySignal(SIGSEGV),
              "");
  ASSERT_EXIT((reinterpret_cast<Func>(4095))(), testing::KilledBySignal(SIGSEGV), "");
}

// Add some valid code to the end of the first page and graceful failure rescue at the beginning of
// the second page.
constexpr uint32_t kPageCrossingCode[] = {
    // First page
    // mov x0, x0
    0xaa00'03e0,
    // Second page
    // If SIGSEGV doesn't happen, make sure we return cleanly.
    // ret
    0xd65f'03c0,
};

constexpr size_t kFirstPageCodeSize = 4;
sigjmp_buf g_jmpbuf;
uint8_t* g_noexec_page_addr = nullptr;

void SigsegvHandler(int /* sig */, siginfo_t* /* info */, void* ctx) {
  fprintf(stderr, "SIGSEGV caught\n");
  // Warning: do not use ASSERT, so that we recover with longjump unconditionally.
  // Otherwise we'll be calling the handler in infinite loop.
  EXPECT_EQ(static_cast<ucontext*>(ctx)->uc_mcontext.pc,
            reinterpret_cast<uintptr_t>(g_noexec_page_addr));
  longjmp(g_jmpbuf, 1);
}

TEST(HandleNotExecutable, ExecutableToNotExecutablePageCrossing) {
  const long kPageSize = sysconf(_SC_PAGESIZE);
  // Allocate two pages.
  uint8_t* first_page = static_cast<uint8_t*>(
      mmap(0, kPageSize * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  // Make first page executable.
  mprotect(first_page, kPageSize, PROT_READ | PROT_WRITE | PROT_EXEC);

  g_noexec_page_addr = first_page + kPageSize;
  uint8_t* start_addr = g_noexec_page_addr - kFirstPageCodeSize;
  memcpy(start_addr, kPageCrossingCode, sizeof(kPageCrossingCode));

  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = SigsegvHandler;
  ScopedSigaction scoped_sa(SIGSEGV, &sa);

  if (setjmp(g_jmpbuf) == 0) {
    fprintf(stderr, "Jumping to executable page before non-executable page\n");
    reinterpret_cast<void (*)()>(start_addr)();
    ADD_FAILURE() << "Function call should not have returned";
  } else {
    fprintf(stderr, "Successful recovery\n");
  }

  munmap(first_page, kPageSize * 2);
}

}  // namespace
