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

#include "berberis/runtime_primitives/virtual_guest_call_frame.h"

#include <cstdint>

#include "berberis/base/checks.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

GuestAddr ScopedVirtualGuestCallFrame::g_return_address_ = {};

// For ARM64, guest function preserves at least sp and returns by jumping
// to address provided in x30 (LR). So here ctor emulates the following code:
//
//   // save registers to be changed and maintain stack alignment
//   stp x29, x30, [sp, #-16]!
//   mov x29, sp
//
//   <parameters passing happens after ctor, adjusts x0-7, sp>
//
//   x30 = 'special-return-addr'  // ensure stop after return and call guest function, as
//   pc = 'pc'                    //   'special-return-addr': blr <pc>
//
// and dtor emulates the following code:
//
//   mov sp, x29
//   ldp x29, x30, [sp], #16
//
ScopedVirtualGuestCallFrame::ScopedVirtualGuestCallFrame(CPUState* cpu, GuestAddr pc) : cpu_(cpu) {
  // stp x29, x30, [sp, #-16]!
  cpu_->sp -= 16;
  uint64_t* saved_regs = ToHostAddr<uint64_t>(cpu_->sp);
  saved_regs[0] = cpu_->x[29];   // x29 (FP)
  saved_regs[1] = cpu_->x[30];   // x30 (LR)
  // mov x29, sp
  cpu_->x[29] = cpu_->sp;

  // For safety checks!
  stack_pointer_ = cpu_->x[29];
  link_register_ = cpu_->x[30];

  program_counter_ = cpu_->insn_addr;

  // Set pc and lr as for 'blr <guest>'.
  cpu_->x[30] = g_return_address_;
  cpu_->insn_addr = pc;
}

ScopedVirtualGuestCallFrame::~ScopedVirtualGuestCallFrame() {
  // Safety check - returned to correct pc?
  CHECK_EQ(g_return_address_, cpu_->insn_addr);
  // Safety check - guest call didn't preserve fp?
  CHECK_EQ(stack_pointer_, cpu_->x[29]);

  cpu_->sp = cpu_->x[29];

  const uint64_t* saved_regs = ToHostAddr<uint64_t>(cpu_->sp);
  // ldp x29, x30, [sp], #16
  cpu_->x[29] = saved_regs[0];
  cpu_->x[30] = saved_regs[1];
  cpu_->sp += 16;
  cpu_->insn_addr = program_counter_;

  // Safety checks - guest stack was smashed?
  CHECK_EQ(link_register_, cpu_->x[30]);
}

}  // namespace berberis
