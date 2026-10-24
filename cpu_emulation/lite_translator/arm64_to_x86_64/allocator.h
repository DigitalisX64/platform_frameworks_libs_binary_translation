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

#ifndef BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_ALLOCATOR_H_
#define BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_ALLOCATOR_H_

#include <algorithm>  // std::max
#include <optional>

#include "berberis/assembler/x86_64.h"
#include "berberis/base/dependent_false.h"

namespace berberis {

template <typename RegType>
inline constexpr auto kAllocatableRegisters = []() {
  static_assert(kDependentTypeFalse<RegType>,
                "kAllocatableRegisters is only usable with x86_64::Assembler::Register or "
                "x86_64::Assembler::XMMRegister");
  return true;
};

// add rcx+rdx to register pool (13 GP regs)
// RCX is saved/restored around variable shifts (SHL/SHR/SAR/ROR by CL) and NZCV flag emission.
// RDX is saved/restored around DIV/IDIV/widening MUL instructions that clobber it.
// RCX placed early (index 1) so it's used for permanent mappings, not temps.
template <>
inline constexpr x86_64::Assembler::Register kAllocatableRegisters<x86_64::Assembler::Register>[] =
    {x86_64::Assembler::rbx,
     x86_64::Assembler::rcx,
     x86_64::Assembler::rsi,
     x86_64::Assembler::rdi,
     x86_64::Assembler::r8,
     x86_64::Assembler::r9,
     x86_64::Assembler::r10,
     x86_64::Assembler::r11,
     x86_64::Assembler::r12,
     x86_64::Assembler::r13,
     x86_64::Assembler::r14,
     x86_64::Assembler::r15,
     x86_64::Assembler::rdx};

// Enforce the invariants the comment above and the rest of the lite translator rely on:
//   * exactly 13 allocatable GP registers (temp-vs-permanent accounting in Alloc/AllocTemp
//     assumes this count);
//   * rax, rbp, rsp are NOT in the pool — rax holds the guest PC, rbp the ThreadState pointer,
//     and rsp is the host stack; handing any of them out would corrupt dispatch;
//   * rcx sits at index 1 so it is consumed early for permanent guest-register mappings rather
//     than as a temp, per the save/restore-around-shifts note above.
static_assert(std::size(kAllocatableRegisters<x86_64::Assembler::Register>) == 13,
              "lite translator GP pool must have exactly 13 registers");
static_assert(kAllocatableRegisters<x86_64::Assembler::Register>[1] == x86_64::Assembler::rcx,
              "rcx must stay at index 1 (used early for permanent mappings, not as a temp)");
static_assert(
    [] {
      for (auto reg : kAllocatableRegisters<x86_64::Assembler::Register>) {
        if (reg == x86_64::Assembler::rax || reg == x86_64::Assembler::rbp ||
            reg == x86_64::Assembler::rsp) {
          return false;
        }
      }
      return true;
    }(),
    "allocatable GP pool must exclude rax (guest PC), rbp (ThreadState ptr), and rsp (host stack)");

template <>
inline constexpr x86_64::Assembler::XMMRegister
    kAllocatableRegisters<x86_64::Assembler::XMMRegister>[] = {x86_64::Assembler::xmm0,
                                                               x86_64::Assembler::xmm1,
                                                               x86_64::Assembler::xmm2,
                                                               x86_64::Assembler::xmm3,
                                                               x86_64::Assembler::xmm4,
                                                               x86_64::Assembler::xmm5,
                                                               x86_64::Assembler::xmm6,
                                                               x86_64::Assembler::xmm7,
                                                               x86_64::Assembler::xmm8,
                                                               x86_64::Assembler::xmm9,
                                                               x86_64::Assembler::xmm10,
                                                               x86_64::Assembler::xmm11,
                                                               x86_64::Assembler::xmm12,
                                                               x86_64::Assembler::xmm13,
                                                               x86_64::Assembler::xmm14,
                                                               x86_64::Assembler::xmm15};

template <typename RegType>
class Allocator {
 public:
  // Fixed number of back-of-pool registers reserved for per-instruction temps.
  // Permanent guest-register mappings (Alloc) grow from the front of the pool
  // and may never intrude into this reserve, so any instruction whose temp
  // appetite fits the reserve can be translated at ANY point in a region —
  // register pressure makes guest-register accesses spill through ThreadState
  // memory (see GetReg/SetReg) instead of failing the region.
  //
  // Sized by the worst-case temp appetite of the hottest instruction class:
  // an SP-based LDP/STP needs exactly 6 GP temps (base + address + a TBI mask
  // and a data/result temp per element). Measured on a 14-app prebuilt sweep,
  // 94% of all lite-translation failures were STP/LDP starving for temps at
  // the pressure wall; with the reserve they always fit. An instruction
  // needing MORE than the reserve still fails cleanly under pressure and
  // falls back to the clamp-and-retranslate path (rare: <1% of failures).
  //
  // Only the GP allocator ever calls Alloc() — guest V registers are not
  // permanently mapped — so for the XMM pool this reserve is inert.
  static constexpr uint32_t kReservedTempRegs = 6;

  std::optional<RegType> Alloc() {
    // std::max(temp_regs_allocated, kReservedTempRegs): temps live right now
    // (mid-instruction) may already reach below the reserve line; never hand
    // out a register a live temp occupies.
    if (regs_allocated_ + std::max(temp_regs_allocated, kReservedTempRegs) >= kNumRegister) {
      return std::nullopt;
    }
    return std::optional<RegType>(kAllocatableRegisters<RegType>[regs_allocated_++]);
  }

  std::optional<RegType> AllocTemp() {
    if (regs_allocated_ + temp_regs_allocated >= kNumRegister) {
      return std::nullopt;
    }
    auto res = std::optional<RegType>(
        kAllocatableRegisters<RegType>[kNumRegister - 1 - temp_regs_allocated]);
    temp_regs_allocated++;
    return res;
  }

  void FreeTemps() { temp_regs_allocated = 0; }

 private:
  inline static const uint32_t kNumRegister = std::size(kAllocatableRegisters<RegType>);

  uint32_t regs_allocated_ = 0;
  uint32_t temp_regs_allocated = 0;
};

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_ARM64_TO_X86_64_ALLOCATOR_H_
