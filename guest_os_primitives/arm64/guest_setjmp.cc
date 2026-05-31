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

#include "berberis/guest_os_primitives/guest_setjmp.h"

#include <csetjmp>
#include <cstdint>
#include <cstring>  // memcpy

#include "berberis/base/host_signal.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

namespace {

// jmp_buf format is totally Bionic-private (see bionic/libc/arch-arm64/bionic/setjmp.S)
// We don't have to use the original format as save/restore is only done here.
// Still, let's keep it compatible with some release as it might help debugging...
//
// ARM64 jmp_buf layout (bionic):
// word   name            description
// 0      sigflag/cookie  setjmp cookie in top 63 bits, signal mask flag in low bit
// 1      sigmask         64-bit signal mask
// 2      x19
// 3      x20
// 4      x21
// 5      x22
// 6      x23
// 7      x24
// 8      x25
// 9      x26
// 10     x27
// 11     x28
// 12     x29 (FP)
// 13     x30 (LR)
// 14     SP
// 15     d8
// 16     d9
// 17     d10
// 18     d11
// 19     d12
// 20     d13
// 21     d14
// 22     d15
// 23     checksum
// _JBLEN: defined in bionic/libc/include/setjmp.h (32 for arm64)

const int kJmpBufSigFlagAndCookieWord = 0;
const int kJmpBufSigMaskWord = 1;
const int kJmpBufCoreBaseWord = 2;     // x19-x28 start here (10 regs)
const int kJmpBufFPWord = 12;          // x29
const int kJmpBufLRWord = 13;          // x30
const int kJmpBufSPWord = 14;          // SP
const int kJmpBufFloatingPointBaseWord = 15;  // d8-d15 (8 regs)
const int kJmpBufChecksumWord = 23;
// jmp_buf should be at least 32 words long.
// Use the last word to store the address of the host jmp_buf.
const int kJmpBufHostBufWord = 31;

// jmp_buf cookie can be anything but 0 (see bionic/tests/setjmp_test.cpp: setjmp_cookie)
// ATTENTION: Keep low bit 0 for signal mask flag.
const uint64_t kJmpBufCookie = 0x12'3210ULL;

uint64_t CalcJumpBufChecksum(const uint64_t* buf) {
  uint64_t res = 0;
  for (int i = 0; i < kJmpBufChecksumWord; ++i) {
    res ^= buf[i];
  }
  return res;
}

}  // namespace

void SaveRegsToJumpBuf(const ThreadState* state, void* guest_jmp_buf, int save_sig_mask) {
  uint64_t* buf = reinterpret_cast<uint64_t*>(guest_jmp_buf);

  // Clear the buffer in case the format has gaps.
  memset(buf, 0, kJmpBufChecksumWord * sizeof(uint64_t));

  // Cookie, signal flag, signal mask
  buf[kJmpBufSigFlagAndCookieWord] = kJmpBufCookie;
  if (save_sig_mask) {
    buf[kJmpBufSigFlagAndCookieWord] |= 0x1;
    RTSigprocmaskSyscallOrDie(
        SIG_SETMASK, nullptr, reinterpret_cast<HostSigset*>(buf + kJmpBufSigMaskWord));
  }

  // x19-x28 (callee-saved general registers)
  for (int i = 0; i < 10; ++i) {
    buf[kJmpBufCoreBaseWord + i] = state->cpu.x[19 + i];
  }

  // x29 (FP), x30 (LR)
  buf[kJmpBufFPWord] = state->cpu.x[29];
  buf[kJmpBufLRWord] = state->cpu.x[30];

  // SP
  buf[kJmpBufSPWord] = state->cpu.sp;

  // d8-d15 (callee-saved FP registers, lower 64 bits of v8-v15)
  for (int i = 0; i < 8; ++i) {
    uint64_t low64;
    memcpy(&low64, &state->cpu.v[8 + i], sizeof(low64));
    buf[kJmpBufFloatingPointBaseWord + i] = low64;
  }

  // Checksum
  buf[kJmpBufChecksumWord] = CalcJumpBufChecksum(buf);
}

void RestoreRegsFromJumpBuf(ThreadState* state, void* guest_jmp_buf, int retval) {
  const uint64_t* buf = reinterpret_cast<const uint64_t*>(guest_jmp_buf);

  // Checksum
  if (buf[kJmpBufChecksumWord] != CalcJumpBufChecksum(buf)) {
    LOG_ALWAYS_FATAL("setjmp checksum mismatch");
  }

  // Cookie
  if ((buf[kJmpBufSigFlagAndCookieWord] & ~0x1ULL) != kJmpBufCookie) {
    LOG_ALWAYS_FATAL("setjmp cookie mismatch");
  }

  // Signal mask
  if (buf[kJmpBufSigFlagAndCookieWord] & 0x1) {
    RTSigprocmaskSyscallOrDie(
        SIG_SETMASK, reinterpret_cast<const HostSigset*>(buf + kJmpBufSigMaskWord), nullptr);
  }

  // x19-x28
  for (int i = 0; i < 10; ++i) {
    state->cpu.x[19 + i] = buf[kJmpBufCoreBaseWord + i];
  }

  // x29 (FP), x30 (LR)
  state->cpu.x[29] = buf[kJmpBufFPWord];
  state->cpu.x[30] = buf[kJmpBufLRWord];

  // SP
  state->cpu.sp = buf[kJmpBufSPWord];

  // d8-d15 (restore lower 64 bits, zero upper)
  for (int i = 0; i < 8; ++i) {
    __uint128_t val = buf[kJmpBufFloatingPointBaseWord + i];
    state->cpu.v[8 + i] = val;
  }

  // Function return: set x0 = retval, pc = lr
  CPUState& cpu = state->cpu;
  SetInsnAddr(cpu, GetLinkRegister(cpu));
  SetReturnValueRegister(cpu, retval);
}

jmp_buf** GetHostJmpBufPtr(void* guest_jmp_buf) {
  uint64_t* buf = reinterpret_cast<uint64_t*>(guest_jmp_buf);
  return reinterpret_cast<jmp_buf**>(buf + kJmpBufHostBufWord);
}

}  // namespace berberis
