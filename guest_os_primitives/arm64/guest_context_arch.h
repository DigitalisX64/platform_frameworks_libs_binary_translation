// region digitalis
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

#ifndef BERBERIS_GUEST_OS_PRIMITIVES_ARM64_GUEST_CONTEXT_ARCH_H_
#define BERBERIS_GUEST_OS_PRIMITIVES_ARM64_GUEST_CONTEXT_ARCH_H_

#include <cstdint>
#include <cstring>  // memcpy

#include "berberis/base/checks.h"
#include "berberis/base/struct_check.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

class GuestContext {
 public:
  GuestContext() = default;
  GuestContext(const GuestContext&) = delete;
  GuestContext& operator=(const GuestContext&) = delete;

  void Save(const CPUState* cpu) {
    // Save everything.
    cpu_ = *cpu;

    // Save context.
    memset(&ctx_, 0, sizeof(ctx_));
    // x0-x30
    static_assert(sizeof(cpu->x) == sizeof(ctx_.uc_mcontext.regs));
    memcpy(ctx_.uc_mcontext.regs, cpu->x, sizeof(ctx_.uc_mcontext.regs));
    ctx_.uc_mcontext.sp = cpu->sp;
    ctx_.uc_mcontext.pc = cpu->insn_addr;
    ctx_.uc_mcontext.pstate = cpu->flags;

    // Save FPSIMD context.
    fpsimd_.head.magic = FPSIMD_MAGIC;
    fpsimd_.head.size = sizeof(Guest_fpsimd_context);
    static_assert(sizeof(cpu->v) == sizeof(fpsimd_.vregs));
    memcpy(fpsimd_.vregs, cpu->v, sizeof(fpsimd_.vregs));
    // TODO: save fpsr/fpcr properly.
  }

  void Restore(CPUState* cpu) const {
    // Restore everything.
    *cpu = cpu_;

    // Overwrite from context.
    memcpy(cpu->x, ctx_.uc_mcontext.regs, sizeof(ctx_.uc_mcontext.regs));
    cpu->sp = ctx_.uc_mcontext.sp;
    cpu->insn_addr = ctx_.uc_mcontext.pc;
    cpu->flags = static_cast<uint16_t>(ctx_.uc_mcontext.pstate);

    // Restore FPSIMD.
    if (fpsimd_.head.magic == FPSIMD_MAGIC) {
      memcpy(cpu->v, fpsimd_.vregs, sizeof(fpsimd_.vregs));
    }
  }

  void* ptr() { return &ctx_; }

 private:
  static constexpr uint32_t FPSIMD_MAGIC = 0x46508001;

  // See bionic/libc/kernel/uapi/asm-arm64/asm/sigcontext.h
  struct Guest_sigcontext {
    uint64_t fault_address;
    uint64_t regs[31];
    uint64_t sp;
    uint64_t pc;
    uint64_t pstate;
    // 128 bytes for future expansion.
    uint8_t __reserved[4096] __attribute__((aligned(16)));
  };

  // Header for extended context blocks in __reserved area.
  struct Guest_aarch64_ctx {
    uint32_t magic;
    uint32_t size;
  };

  // FPSIMD context that appears in __reserved area.
  struct Guest_fpsimd_context {
    struct Guest_aarch64_ctx head;
    uint32_t fpsr;
    uint32_t fpcr;
    __uint128_t vregs[32];
  };

  // See bionic/libc/kernel/uapi/asm-arm64/asm/ucontext.h
  struct Guest_ucontext {
    uint64_t uc_flags;
    Guest_ucontext* uc_link;
    // We assume guest stack_t is compatible with host (see RunGuestSyscall___NR_sigaltstack).
    stack_t uc_stack;
    Guest_sigset_t uc_sigmask;
    uint8_t __linux_unused[1024 / 8 - sizeof(Guest_sigset_t)];
    Guest_sigcontext uc_mcontext;
  };

  Guest_ucontext ctx_;
  // FPSIMD context stored separately for easy access.
  Guest_fpsimd_context fpsimd_;
  CPUState cpu_;
};

}  // namespace berberis

#endif  // BERBERIS_GUEST_OS_PRIMITIVES_ARM64_GUEST_CONTEXT_ARCH_H_
// endregion
