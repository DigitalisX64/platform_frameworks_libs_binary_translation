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

#include "berberis/runtime_primitives/runtime_library.h"

#if defined(__x86_64__)
#include "berberis/assembler/machine_code.h"
#include "berberis/assembler/x86_64.h"
#include "berberis/runtime_primitives/code_pool.h"
#endif

namespace berberis {

HostCodeAddr kEntryInterpret;
HostCodeAddr kEntryExitGeneratedCode;
HostCodeAddr kEntryStop;
HostCodeAddr kEntryNoExec;
HostCodeAddr kEntryNotTranslated;
HostCodeAddr kEntryTranslating;
HostCodeAddr kEntryInvalidating;
HostCodeAddr kEntryWrapping;

namespace {
// This function installs a trampoline in the CodePool address space.
// This needed to ensure that all entries in the translation cache
// are always pointing to the memory allocated via CodePool.
HostCodeAddr InstallEntryTrampoline(HostCode target_function_ptr) {
#if defined(__x86_64__)
  MachineCode mc;
  x86_64::Assembler as(&mc);
  as.Jmp(target_function_ptr);
  as.Finalize();
  // TODO(b/232598137): maybe use ColdPool?
  return GetDefaultCodePoolInstance()->Add(&mc);
#else
  return AsHostCodeAddr(target_function_ptr);
#endif
}
}  // namespace

void InitHostEntries() {
  kEntryInterpret = InstallEntryTrampoline(AsHostCode(berberis_entry_Interpret));
  kEntryExitGeneratedCode = InstallEntryTrampoline(AsHostCode(berberis_entry_ExitGeneratedCode));
  kEntryStop = InstallEntryTrampoline(AsHostCode(berberis_entry_Stop));
  kEntryNoExec = InstallEntryTrampoline(AsHostCode(berberis_entry_NoExec));
  kEntryNotTranslated = InstallEntryTrampoline(AsHostCode(berberis_entry_NotTranslated));
  kEntryTranslating = InstallEntryTrampoline(AsHostCode(berberis_entry_Translating));
  kEntryInvalidating = InstallEntryTrampoline(AsHostCode(berberis_entry_Invalidating));
  kEntryWrapping = InstallEntryTrampoline(AsHostCode(berberis_entry_Wrapping));
}

}  // namespace berberis
