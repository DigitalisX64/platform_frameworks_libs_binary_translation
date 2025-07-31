/*
 * Copyright (C) 2024 The Android Open Source Project
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

#include "berberis/code_gen_lib/code_gen_lib.h"

#include "berberis/assembler/machine_code.h"
#include "berberis/assembler/rv64i.h"
#include "berberis/base/macros.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {

void GenTrampolineAdaptor(MachineCode* mc,
                          GuestAddr pc,
                          HostCode marshall,
                          const void* callee,
                          const char* name) {
  UNUSED(mc, pc, marshall, callee, name);
}

void EmitDirectDispatch(rv64i::Assembler* as, GuestAddr pc, bool check_pending_signals) {
  UNUSED(check_pending_signals);
  // insn_addr is passed between regions in s11.
  as->Li(as->s11, pc);

  if (!config::kLinkJumpsBetweenRegions) {
    as->Li(as->t1, reinterpret_cast<uint64_t>(kEntryExitGeneratedCode));
    as->Jr(as->t1);
    return;
  }

  // TODO(b/352784623): Check for pending signals.

  CHECK_EQ(pc & GuestAddr{0xffff'0000'0000'0000U}, 0);
  as->Li(as->t1, reinterpret_cast<uint64_t>(TranslationCache::GetInstance()->GetHostCodePtr(pc)));
  as->Ld(as->t1, {.base = rv64i::Assembler::t1, .disp = 0});
  as->Jr(as->t1);
}

void EmitIndirectDispatch(rv64i::Assembler* as, rv64i::Assembler::Register target) {
  // insn_addr is passed between regions in s11.
  if (target != as->s11) {
    as->Mv(as->s11, target);
  }

  if (!config::kLinkJumpsBetweenRegions) {
    as->Li(as->t1, reinterpret_cast<uint64_t>(kEntryExitGeneratedCode));
    as->Jr(as->t1);
    return;
  }

  // TODO(b/352784623): Add check for signals.

  auto main_table_ptr = TranslationCache::GetInstance()->main_table_ptr();

  as->Lui(as->t1, 0x100'0000);
  as->Addi(as->t1, as->t1, -1);
  as->Srli(as->t2, as->s11, 24);
  as->And(as->t2, as->t2, as->t1);
  as->Li(as->t3, reinterpret_cast<uint64_t>(main_table_ptr));
  as->Sh3add(as->t2, as->t2, as->t3);
  as->Ld(as->t2, {.base = rv64i::Assembler::t2, .disp = 0});

  as->And(as->t1, as->t1, as->s11);
  as->Sh3add(as->t1, as->t1, as->t2);
  as->Ld(as->t1, {.base = rv64i::Assembler::t1, .disp = 0});

  as->Jr(as->t1);
}

}  // namespace berberis
