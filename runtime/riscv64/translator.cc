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

#include "berberis/runtime/translator.h"
#include "translator.h"

#include <cstdint>
#include <cstdlib>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/profiler_interface.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/runtime_primitives/virtual_guest_call_frame.h"

namespace berberis {

namespace {

// Use aligned address of this variable as the default stop address for guest execution.
// It should never coincide with any guest address or address of a wrapped host symbol.
// Unwinder might examine nearby insns.
alignas(4) uint32_t g_native_bridge_call_guest[] = {
    // <native_bridge_call_guest>:
    0xd503'201f,  // nop
    0xd503'201f,  // nop  <--
    0xd503'201f,  // nop
};

uint8_t GetRiscv64InsnSize(GuestAddr pc) {
  constexpr uint16_t kInsnLenMask = uint16_t{0b11};
  if ((*ToHostAddr<uint16_t>(pc) & kInsnLenMask) != kInsnLenMask) {
    return 2;
  }
  return 4;
}

}  // namespace

HostCodePiece InstallTranslated(MachineCode* machine_code,
                                GuestAddr pc,
                                size_t size,
                                const char* prefix) {
  HostCodeAddr host_code = GetDefaultCodePoolInstance()->Add(machine_code);
  ProfilerLogGeneratedCode(AsHostCode(host_code), machine_code->install_size(), pc, size, prefix);
  return {host_code, machine_code->install_size()};
}

// Check whether the given guest program counter is executable, accounting for compressed
// instructions. Returns a tuple indicating whether the memory is executable and the size of the
// first instruction in bytes.
std::tuple<bool, uint8_t> IsPcExecutable(GuestAddr pc, GuestMapShadow* guest_map_shadow) {
  // First check if the instruction would be in executable memory if it is compressed.  This
  // prevents dereferencing unknown memory to determine the size of the instruction.
  constexpr uint8_t kMinimumInsnSize = 2;
  if (!guest_map_shadow->IsExecutable(pc, kMinimumInsnSize)) {
    return {false, kMinimumInsnSize};
  }

  // Now check the rest of the instruction based on its size.  It is now safe to dereference the
  // memory at pc because at least two bytes are within known executable memory.
  uint8_t first_insn_size = GetRiscv64InsnSize(pc);
  if (first_insn_size > kMinimumInsnSize &&
      !guest_map_shadow->IsExecutable(pc + kMinimumInsnSize, first_insn_size - kMinimumInsnSize)) {
    return {false, first_insn_size};
  }

  return {true, first_insn_size};
}

void InitTranslator() {
  InitTranslatorArch();
  InitVirtualGuestCallFrameReturnAddress(ToGuestAddr(g_native_bridge_call_guest + 1));
  InitInterpreter();
}

}  // namespace berberis
