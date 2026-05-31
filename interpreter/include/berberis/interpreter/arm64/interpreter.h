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

#ifndef BERBERIS_INTERPRETER_ARM64_INTERPRETER_H_
#define BERBERIS_INTERPRETER_ARM64_INTERPRETER_H_

#include "berberis/guest_state/guest_state.h"

namespace berberis { class TranslationCache; }

namespace berberis {

void InitInterpreter();
void InterpretInsn(ThreadState* state);
// Batch interpreter — reuses Interpreter/Decoder objects across instructions
// to eliminate per-instruction construction overhead (~3x faster).
void InterpretBatch(ThreadState* state, int max_insns, TranslationCache* cache);

}  // namespace berberis

#endif  // BERBERIS_INTERPRETER_ARM64_INTERPRETER_H_
