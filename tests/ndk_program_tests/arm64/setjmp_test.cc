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

#include "gtest/gtest.h"

#include <setjmp.h>

#include <bit>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint64_t kClobberVal{0xdead'beef'dead'beef};

jmp_buf g_jmp_buf;

// Use naked assembly since otherwise compiler may save/restore callee saved before longjmp instead
// of making sure they are clobbered.
[[gnu::naked]] void ClobberAndLongjmp(void* jmp_buf_ptr, int ret_val, void* longjmp_ptr) {
  // clang-format off
  asm volatile(
      "mov x19, %0\n\t" "mov x20, %0\n\t" "mov x21, %0\n\t"
      "mov x22, %0\n\t" "mov x23, %0\n\t" "mov x24, %0\n\t"
      "mov x25, %0\n\t" "mov x26, %0\n\t" "mov x27, %0\n\t" "mov x28, %0\n\t"
      "fmov d8,  %d1\n\t" "fmov d9,  %d1\n\t" "fmov d10, %d1\n\t" "fmov d11, %d1\n\t"
      "fmov d12, %d1\n\t" "fmov d13, %d1\n\t" "fmov d14, %d1\n\t" "fmov d15, %d1\n\t"
      // jmp_buf_ptr and ret_val are already in x0 and x1.
      "blr x2"
      :
      : "r"(kClobberVal), "w"(std::bit_cast<double>(kClobberVal))
      // Make sure kClobberVal is not allocated to x0-x2 since they carry function arguments.
      : "x0", "x1", "x2",
      // Clobbers.
      "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
      "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
  // clang-format on
}

TEST(SetJmp, CalleeSavedRegistersRestoredArm64) {
  // x20-x28 (9 registers). x19 is used and tested in a special way.
  // d8-d15 (8 registers).
  constexpr size_t kNumSavedRegs = 9 + 8;
  using RegisterState = uint64_t[kNumSavedRegs];

  // 1. SETUP
  // Has to be volatile, so that compiler doesn't optimize it out around the setjmp.
  volatile RegisterState magic_values;
  volatile RegisterState restored_values{};
  for (size_t i = 0; i < kNumSavedRegs; i++) {
    magic_values[i] = uint64_t{0x0101'0101'0101'0101} * i;
  }

  // Group everything in one structure, so that we can pass it into the assembly in one register.
  volatile struct AsmParams {
    volatile RegisterState* input;
    volatile RegisterState* output;
    jmp_buf* jmp_buf_ptr;
    void* setjmp_ptr;
    int setjmp_ret_val;
  } asm_params{
      .input = &magic_values,
      .output = &restored_values,
      .jmp_buf_ptr = &g_jmp_buf,
      .setjmp_ptr = reinterpret_cast<void*>(&setjmp),
  };

  volatile bool longjmp_happeded = false;

  register uintptr_t asm_params_ptr asm("x19") = std::bit_cast<uintptr_t>(&asm_params);
  // 2. EXECUTION
  asm volatile(
      // asm_params_ptr comes in x19.
      // Also x19 is callee-saved so it's preserved over setjmp/longjmp.
      // Warning: we cannot allocate more stack space here to carry values over setjmp
      // since on the first return stack will be deallocated and clobbered.
      "ldr x9, [x19, %[input_offset]]\n\t"
      // Load magic values into registers.
      "ldr x20, [x9], #8\n\t"
      "ldp x21, x22, [x9], #16\n\t"
      "ldp x23, x24, [x9], #16\n\t"
      "ldp x25, x26, [x9], #16\n\t"
      "ldp x27, x28, [x9], #16\n\t"
      "ldp d8,  d9,  [x9], #16\n\t"
      "ldp d10, d11, [x9], #16\n\t"
      "ldp d12, d13, [x9], #16\n\t"
      "ldp d14, d15, [x9], #16\n\t"

      "ldr x0, [x19, %[jmp_buf_ptr_offset]]\n\t"
      "ldr x1, [x19, %[setjmp_ptr_offset]]\n\t"
      "blr x1\n\t"
      "str x0, [x19, %[setjmp_ret_val_offset]]\n\t"
      "cbz x0, .L_done_%= \n\t"

      // --- RESTORED PATH (only runs after longjmp) ---
      "ldr x9, [x19, %[output_offset]]\n\t"
      // Store registers to the output buffer.
      "str x20, [x9], #8\n\t"
      "stp x21, x22, [x9], #16\n\t"
      "stp x23, x24, [x9], #16\n\t"
      "stp x25, x26, [x9], #16\n\t"
      "stp x27, x28, [x9], #16\n\t"
      "stp d8,  d9,  [x9], #16\n\t"
      "stp d10, d11, [x9], #16\n\t"
      "stp d12, d13, [x9], #16\n\t"
      "stp d14, d15, [x9], #16\n\t"

      ".L_done_%=: \n\t"
      :
      : "r"(asm_params_ptr),
        [input_offset] "i"(offsetof(AsmParams, input)),
        [output_offset] "i"(offsetof(AsmParams, output)),
        [setjmp_ptr_offset] "i"(offsetof(AsmParams, setjmp_ptr)),
        [jmp_buf_ptr_offset] "i"(offsetof(AsmParams, jmp_buf_ptr)),
        [setjmp_ret_val_offset] "i"(offsetof(AsmParams, setjmp_ret_val))
      // We initialize callee-saved registers except x19 with magic values and setjmp call
      // clobbers caller-saved. This means that we clobber everything except x19. x18 (scs)
      // and x29 (fp) are reserved and cannot be used on the clobber list. sp is restored by
      // setjmp.
      // clang-format off
    : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
      "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", /*x18*/ /*x19*/
      "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", /*x29*/
      "x30", "memory", "cc",
    // Vector registers.
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
    "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
       "v30", "v31");
  // clang-format on

  printf("setjmp returned %d\n", asm_params.setjmp_ret_val);

  if (asm_params.setjmp_ret_val == 0) {
    longjmp_happeded = true;
    ClobberAndLongjmp(&g_jmp_buf, 1, reinterpret_cast<void*>(&longjmp));
    FAIL() << "longjmp() did not execute. This line should not be reached.";
  }

  // 3. VERIFICATION
  ASSERT_TRUE(longjmp_happeded);

  for (size_t i = 0; i < kNumSavedRegs; i++) {
    EXPECT_EQ(magic_values[i], restored_values[i])
        << "register isn't properly restored for index " << i;
  }
}

}  // namespace
