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

#include <signal.h>  // stack_t

#include <cstdint>
#include <cstring>  // memcpy

#include "berberis/base/checks.h"
#include "berberis/base/struct_check.h"
#include "berberis/guest_os_primitives/guest_signal.h"  // Guest_sigset_t
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

    // Per the Linux arm64 ABI, FPSIMD state is delivered to the guest signal
    // handler as a chained _aarch64_ctx block at the start of
    // uc_mcontext.__reserved. A handler that walks this chain expects:
    //   {magic=FPSIMD_MAGIC=0x46508001, size=528}  followed by fpsr|fpcr|vregs
    //   {magic=0, size=0}                           NULL terminator
    // The trailing zero terminator is already in place because the memset
    // above zeroed all of __reserved before we write the FPSIMD block.
    static_assert(sizeof(Guest_fpsimd_context) <= sizeof(ctx_.uc_mcontext.__reserved));
    auto* fpsimd =
        reinterpret_cast<Guest_fpsimd_context*>(ctx_.uc_mcontext.__reserved);
    fpsimd->head.magic = FPSIMD_MAGIC;
    fpsimd->head.size = sizeof(Guest_fpsimd_context);
    // FPSR bit layout (ARM ARM C5.2.8): bits[31:27]=N,Z,C,V,QC; bit[7]=IDC;
    // bit[4]=IXC; etc. Bionic-built handlers reading uc_mcontext FPSIMD see
    // QC (saturation) — populated when SIMD ops saturate — as well as IDC
    // for the emulated input-denormal flag. We don't currently emulate the
    // arithmetic exception bits beyond IDC, so this is the full set of FPSR
    // bits we can faithfully report.
    fpsimd->fpsr = cpu->emulated_fpsr;
    fpsimd->fpcr = cpu->cached_fpcr;
    static_assert(sizeof(cpu->v) == sizeof(fpsimd->vregs));
    memcpy(fpsimd->vregs, cpu->v, sizeof(fpsimd->vregs));
  }

  void Restore(CPUState* cpu) const {
    // Restore everything.
    *cpu = cpu_;

    // Overwrite from context.
    memcpy(cpu->x, ctx_.uc_mcontext.regs, sizeof(ctx_.uc_mcontext.regs));
    cpu->sp = ctx_.uc_mcontext.sp;
    cpu->insn_addr = ctx_.uc_mcontext.pc;
    cpu->flags = static_cast<uint16_t>(ctx_.uc_mcontext.pstate);

    // Restore FPSIMD from the chained block at the start of __reserved.
    // If the guest handler clobbered or unchained the FPSIMD context, the
    // magic check fails and we leave cpu->v / fpsr / fpcr at the values
    // copied from cpu_.
    const auto* fpsimd =
        reinterpret_cast<const Guest_fpsimd_context*>(ctx_.uc_mcontext.__reserved);
    if (fpsimd->head.magic == FPSIMD_MAGIC) {
      memcpy(cpu->v, fpsimd->vregs, sizeof(fpsimd->vregs));
      cpu->emulated_fpsr = fpsimd->fpsr;
      cpu->cached_fpcr = fpsimd->fpcr;
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

  // §L2 box (1) audit: lock the byte layout to bionic's arm64 ABI so future
  // edits cannot silently drift. Offsets/sizes are in bits (CHAR_BIT * bytes)
  // because that is the macro contract. References:
  //   bionic/libc/kernel/uapi/asm-arm64/asm/sigcontext.h (struct sigcontext)
  //   bionic/libc/include/sys/ucontext.h                  (struct ucontext)
  // Guest_sigcontext: 31× uint64 GPRs at offset 8B + sp@256B + pc@264B +
  // pstate@272B + 8B implicit padding (because __reserved has __aligned__(16)
  // and offset 280 is not 16-aligned) + __reserved[4096]@288B = 4384 bytes.
  CHECK_STRUCT_LAYOUT(Guest_sigcontext, 35072, 128);
  CHECK_FIELD_LAYOUT(Guest_sigcontext, fault_address, 0, 64);
  CHECK_FIELD_LAYOUT(Guest_sigcontext, regs, 64, 1984);
  CHECK_FIELD_LAYOUT(Guest_sigcontext, sp, 2048, 64);
  CHECK_FIELD_LAYOUT(Guest_sigcontext, pc, 2112, 64);
  CHECK_FIELD_LAYOUT(Guest_sigcontext, pstate, 2176, 64);
  CHECK_FIELD_LAYOUT(Guest_sigcontext, __reserved, 2304, 32768);

  // Guest_fpsimd_context: 8B _aarch64_ctx head + fpsr(4) + fpcr(4) +
  // 32× __uint128_t vregs(@16B aligned) = 528 bytes.
  CHECK_STRUCT_LAYOUT(Guest_fpsimd_context, 4224, 128);
  CHECK_FIELD_LAYOUT(Guest_fpsimd_context, head, 0, 64);
  CHECK_FIELD_LAYOUT(Guest_fpsimd_context, fpsr, 64, 32);
  CHECK_FIELD_LAYOUT(Guest_fpsimd_context, fpcr, 96, 32);
  CHECK_FIELD_LAYOUT(Guest_fpsimd_context, vregs, 128, 4096);

  // Guest_ucontext: uc_flags@0 + uc_link@8 + uc_stack@16 + uc_sigmask@40 +
  // __linux_unused@48 (120B) + uc_mcontext@176 (8B implicit padding pushes
  // the 16B-aligned uc_mcontext from offset 168 to 176). Total 4560 bytes.
  CHECK_STRUCT_LAYOUT(Guest_ucontext, 36480, 128);
  CHECK_FIELD_LAYOUT(Guest_ucontext, uc_flags, 0, 64);
  CHECK_FIELD_LAYOUT(Guest_ucontext, uc_link, 64, 64);
  CHECK_FIELD_LAYOUT(Guest_ucontext, uc_stack, 128, 192);
  CHECK_FIELD_LAYOUT(Guest_ucontext, uc_sigmask, 320, 64);
  CHECK_FIELD_LAYOUT(Guest_ucontext, uc_mcontext, 1408, 35072);

  Guest_ucontext ctx_;
  CPUState cpu_;
};

}  // namespace berberis

#endif  // BERBERIS_GUEST_OS_PRIMITIVES_ARM64_GUEST_CONTEXT_ARCH_H_
