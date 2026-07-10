/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef BERBERIS_GUEST_STATE_GUEST_STATE_ARCH_H_
#define BERBERIS_GUEST_STATE_GUEST_STATE_ARCH_H_

#include <array>
#include <atomic>
#include <cstdint>

#include "berberis/base/config.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "native_bridge_support/arm64/guest_state/guest_state_cpu_state.h"

namespace berberis {

// Guest CPU state + interface to access guest memory.
struct ThreadState {
  CPUState cpu;

  // Scratch space for host x87/MXCSR use by the inline-intrinsic lowering: some
  // host ops can only read/write memory operands. Mirrors the riscv64
  // ThreadState so the guest-agnostic inline_intrinsic.h scratch path
  // type-checks for the ARM64 optimizing backend too.
  alignas(config::kScratchAreaAlign) uint8_t intrinsics_scratch_area[config::kScratchAreaSize];

  // Guest thread pointer.
  GuestThread* thread;

  // Guest TLS pointer.
  // It can be read using MRC instruction.
  // Statically linked ARM executable initializes it by set_tls syscall.
  // For PIC objects, InitThreadState sets it either to host TLS or to (stub) thread-id.
  // TODO(b/36890513): guest should have its own TLS area for PIC objects too.
  GuestAddr tls;

  // Keep pending signals status here for fast checking in generated code.
  // TODO(b/28058920): move to GuestThread!
  std::atomic_uint_least8_t pending_signals_status;

  GuestThreadResidence residence;

  // Arbitrary per-thread data added by instrumentation.
  void* instrument_data;

  // Point to the guest thread memory start position.
  void* thread_state_storage;
};

inline constexpr unsigned kNumGuestRegs = std::size(CPUState{}.x);
inline constexpr unsigned kNumGuestSimdRegs = std::size(CPUState{}.v);

inline constexpr unsigned kGuestCacheLineSize = 64;

// ---- ARM64 NZCV flag-word layout ------------------------------------------
// CPUState::flags stores NZCV in a host-aligned layout chosen so x86_64 LAHF
// (AH: CF@8, ZF@14, SF@15) + SETO (bit 0) splat them into place (see
// CPUState::FlagMask). These derived constants give the lite translator, the
// heavy optimizer, and the differential fuzzers one source of truth for that
// layout instead of re-hardcoding 0xC101 and bit indices 15/14/8/0 at every
// flag-materialization site. All derived from CPUState::kFlag* so they track
// the host layout; the static_asserts pin the x86_64 values.

// Bit index of a single-bit flag mask (e.g. 1<<15 -> 15).
constexpr int8_t FlagBitIndex(uint32_t single_bit_mask) {
  int8_t i = 0;
  for (; (single_bit_mask >> i) > 1u; ++i) {
  }
  return i;
}

inline constexpr int8_t kFlagNegativeBit = FlagBitIndex(CPUState::kFlagNegative);
inline constexpr int8_t kFlagZeroBit = FlagBitIndex(CPUState::kFlagZero);
inline constexpr int8_t kFlagCarryBit = FlagBitIndex(CPUState::kFlagCarry);
inline constexpr int8_t kFlagOverflowBit = FlagBitIndex(CPUState::kFlagOverflow);

// Mask selecting the four NZCV bits within cpu.flags (== 0xC101 on x86_64).
inline constexpr uint16_t kFlagsNZCVMask =
    static_cast<uint16_t>(CPUState::kFlagNegative | CPUState::kFlagZero | CPUState::kFlagCarry |
                          CPUState::kFlagOverflow);

// ARM FCMP result encodings (ARM ARM C6.2.70) as cpu.flags words:
//   greater   N0 Z0 C1 V0 -> C
//   less      N1 Z0 C0 V0 -> N
//   equal     N0 Z1 C1 V0 -> Z,C
//   unordered N0 Z0 C1 V1 -> C,V
inline constexpr uint16_t kFlagsFpGreater = CPUState::kFlagCarry;
inline constexpr uint16_t kFlagsFpLess = CPUState::kFlagNegative;
inline constexpr uint16_t kFlagsFpEqual = CPUState::kFlagZero | CPUState::kFlagCarry;
inline constexpr uint16_t kFlagsFpUnordered = CPUState::kFlagCarry | CPUState::kFlagOverflow;

#if defined(__x86_64__)
static_assert(kFlagsNZCVMask == 0xC101);
static_assert(kFlagNegativeBit == 15 && (1 << kFlagNegativeBit) == CPUState::kFlagNegative);
static_assert(kFlagZeroBit == 14 && (1 << kFlagZeroBit) == CPUState::kFlagZero);
static_assert(kFlagCarryBit == 8 && (1 << kFlagCarryBit) == CPUState::kFlagCarry);
static_assert(kFlagOverflowBit == 0 && (1 << kFlagOverflowBit) == CPUState::kFlagOverflow);
static_assert(kFlagsFpGreater == 0x0100);
static_assert(kFlagsFpLess == 0x8000);
static_assert(kFlagsFpEqual == 0x4100);
static_assert(kFlagsFpUnordered == 0x0101);
#endif

}  // namespace berberis

#endif  // BERBERIS_GUEST_STATE_GUEST_STATE_ARCH_H_
