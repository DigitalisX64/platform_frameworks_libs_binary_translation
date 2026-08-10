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
#include "berberis/lite_translator/lite_translate_region.h"

#include <cstdint>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/checks.h"
#include "berberis/base/tracing.h"
#include "berberis/code_gen_lib/code_gen_lib.h"
#include "berberis/decoder/arm64/decoder.h"
#include "berberis/decoder/arm64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"

#include "arm64_to_x86_64/lite_translator.h"

namespace berberis {

namespace {

void Finalize(LiteTranslator* translator, GuestAddr pc) {
  translator->ExitRegion(pc);
  translator->as()->Finalize();
}

void GenIncrementProfileCounter(x86_64::Assembler* as, const LiteTranslateParams& params) {
  CHECK(params.counter_location);
  *params.counter_location = 0;

  as->Movq(as->rcx, bit_cast<int64_t>(params.counter_location));
  static_assert(sizeof(*params.counter_location) == 4);
  as->Addl({.base = as->rcx}, 1);
  as->Cmpl({.base = as->rcx}, params.counter_threshold);
  as->Jcc(x86_64::Assembler::Condition::kGreater, params.counter_threshold_callback);
}

}  // namespace

// Returns the success status and
// - in case of success, the pc of the next instruction past the translated region
// - in case of failure, the pc of the failed instruction
// Specifically, returns input pc if we cannot translate even the first instruction.
std::tuple<bool, GuestAddr> TryLiteTranslateRegion(GuestAddr start_pc,
                                                   MachineCode* machine_code,
                                                   LiteTranslateParams params) {
  CHECK_LT(start_pc, params.end_pc);
  LiteTranslator translator(machine_code, start_pc, params);
  SemanticsPlayer sem_player(&translator);
  Decoder decoder(&sem_player);

  if (params.enable_self_profiling) {
    GenIncrementProfileCounter(translator.as(), params);
  }

  while (translator.GetInsnAddr() < params.end_pc && !translator.is_region_end_reached()) {
    // No early termination on register pressure: the allocator reserves a
    // fixed block of temp registers (Allocator::kReservedTempRegs) that
    // permanent mappings can never consume, so once the mapping pool is full,
    // guest-register accesses spill through ThreadState memory (GetReg/SetReg)
    // and translation continues to the region's natural end. Ending the region
    // here instead would split hot straight-line code (function prologues are
    // the worst case) into fragments too small for the heavy tier to gear up.
    uint8_t insn_size = decoder.Decode(ToHostAddr<const uint16_t>(translator.GetInsnAddr()));
    if (!translator.success()) {
      // JIT break profiling
      static uint64_t break_count = 0;
      break_count++;
      if (break_count <= 20 || break_count % 50000 == 0) {
        GuestAddr fail_pc = translator.GetInsnAddr();
        uint32_t insn = *ToHostAddr<const uint32_t>(fail_pc);
        TRACE_AND_ALOGD("berberis: JIT break #%lu pc=0x%lx insn=0x%08x",
                        (unsigned long)break_count,
                        (unsigned long)fail_pc,
                        insn);
      }
      return {false, translator.GetInsnAddr()};
    }
    translator.FreeTempRegs();
    translator.IncrementInsnAddr(insn_size);
  }

  Finalize(&translator, translator.GetInsnAddr());

  return {translator.success(), translator.GetInsnAddr()};
}

}  // namespace berberis
