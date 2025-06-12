/*
 * Copyright (C) 2025 The Android Open Source Project
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

#ifndef BERBERIS_BACKEND_X86_64_MEM_OPERAND_H_
#define BERBERIS_BACKEND_X86_64_MEM_OPERAND_H_

#include <cstdint>

#include "berberis/backend/common/machine_ir.h"

namespace berberis {

namespace x86_64 {

struct MemoryOperand {
  MachineReg base = kInvalidMachineReg;
  MachineReg index = kInvalidMachineReg;
  Assembler::ScaleFactor scale = Assembler::kTimesOne;
  // Note: x86-64 only supports 64bit offset in one instruction: movabs – and that one may only be
  // be used to move a value to or from RAX. We don't use it in our code anywhere and it would be
  // better to treat it as a special case, rather than pretend that other instruction may support
  // 64bit offset.
  int32_t disp = 0;
};

// To be able to write
//   Gen<x86_64::MovssXRegOp>(res.machine_reg(), {.base = arg, .disp = offset});
// wea need a set of instructions, because “Args... args” argument couldn't accept it.
//
// See: “Perfect forwarding forwards objects, not braced things that are trying to become objects”:
// https://devblogs.microsoft.com/oldnewthing/20230727-00/?p=108494
//
// But that means that we need to duplicate the same tedious logic in many places: MachineIR,
// MachineIRBuilder, HeavyOptimizerFrontend, etc.
//
// BERBERIS_DECLARE_MACHINE_INSN_ADAPTER makes that work easy (on the assumption that these
// helper functions only exist to solve this particular problem and delegate actual work to
// another functions).

#define BERBERIS_DECLARE_MACHINE_INSN_ADAPTER(                                                     \
    AdapterName, TupleClass, TupleName, ResultType, ForwarderName)                                 \
  template <template <typename> typename InsnType>                                                 \
  AdapterName()                                                                                    \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 0,  \
                         ResultType> {                                                             \
    return ForwarderName<x86_64::MachineInsn<                                                      \
        typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>>();                   \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 1,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName)>(arg0);                      \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 2,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName)>(arg0, arg1);                \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 3,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName)>(arg0, arg1, arg2);          \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 4,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName)>(arg0, arg1, arg2, arg3);    \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 5,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4);                                                             \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 6,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5);                                                       \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(6, TupleClass, TupleName) arg6)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 7,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(6, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5, arg6);                                                 \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(6, TupleClass, TupleName) arg6,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(7, TupleClass, TupleName) arg7)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 8,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(6, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(7, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);                                           \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(6, TupleClass, TupleName) arg6,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(7, TupleClass, TupleName) arg7,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(8, TupleClass, TupleName) arg8)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 9,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(6, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(7, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(8, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);                                     \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType>                                                 \
  AdapterName(BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(6, TupleClass, TupleName) arg6,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(7, TupleClass, TupleName) arg7,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(8, TupleClass, TupleName) arg8,                  \
              BERBERIS_DECLARE_MACHINE_INSN_TUPLE(9, TupleClass, TupleName) arg9)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName> == 10, \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(5, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(6, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(7, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(8, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_MACHINE_INSN_TUPLE(9, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);                               \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType, typename... Args>                               \
  AdapterName(Args... args)->ResultType {                                                          \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>,  \
        Args...>(args...);                                                                         \
  }

#define BERBERIS_DECLARE_MACHINE_INSN_TUPLE(N, TupleClass, TupleName) \
  std::tuple_element_t<N,                                             \
                       typename x86_64::TupleClass<typename InsnType< \
                           typename CodeEmitter::Assemblers>::DeviceInsnInfo>::TupleName>

}  // namespace x86_64

}  // namespace berberis

#endif  // BERBERIS_BACKEND_X86_64_MEM_OPERAND_H_
