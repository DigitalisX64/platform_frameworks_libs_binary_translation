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

#include <array>
#include <cstdint>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/virtual_guest_call_frame.h"

namespace berberis {

namespace {

TEST(VirtualGuestFrame, InitReturnAddress) {
  constexpr GuestAddr kVirtualGuestFrameReturnAddress = 0xbeef'face;
  ScopedVirtualGuestCallFrame::SetReturnAddress(kVirtualGuestFrameReturnAddress);

  CPUState cpu{};

  alignas(uint64_t) std::array<char, 128> stack;
  SetXReg<SP>(cpu, ToGuestAddr(stack.data() + stack.size()));

  ScopedVirtualGuestCallFrame virtual_guest_call_frame(&cpu, 0xdead'beef);

  EXPECT_EQ(kVirtualGuestFrameReturnAddress, GetXReg<RA>(cpu));

  // Pretend guest code executed up to return address.
  cpu.insn_addr = GetXReg<RA>(cpu);
}

void RunGuestCall(CPUState* cpu) {
  ScopedVirtualGuestCallFrame virtual_guest_call_frame(cpu, 0xbaaa'aaad);

  // Pretend guest code executed up to return address.
  cpu->insn_addr = GetXReg<RA>(*cpu);

  // ScopedVirtualGuestCallFrame creates a stack frame to represent the host function
  // that is calling guest code. That pseudo-function can make arbitrary
  // adjustments to sp and ra because those are callee-saved registers that are
  // restored when the function returns.
  SetXReg<SP>(*cpu, 0x000f'f1ce);
  SetXReg<RA>(*cpu, 0xbaad'f00d);
}

TEST(VirtualGuestFrame, Restore) {
  CPUState cpu{};

  alignas(uint64_t) std::array<char, 128> stack;
  const GuestAddr sp = ToGuestAddr(stack.data() + stack.size());
  const GuestAddr ra = 0xdead'beef;
  const GuestAddr fp = 0xdead'c0de;

  SetXReg<RA>(cpu, ra);
  SetXReg<SP>(cpu, sp);
  SetXReg<FP>(cpu, fp);

  RunGuestCall(&cpu);

  EXPECT_EQ(ra, GetXReg<RA>(cpu));
  EXPECT_EQ(sp, GetXReg<SP>(cpu));
  EXPECT_EQ(fp, GetXReg<FP>(cpu));
}

}  // namespace

}  // namespace berberis
