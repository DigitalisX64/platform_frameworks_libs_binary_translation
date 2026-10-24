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

#ifndef BERBERIS_HEAVY_OPTIMIZER_ARM64_MACHINE_INSN_ADAPTER_H_
#define BERBERIS_HEAVY_OPTIMIZER_ARM64_MACHINE_INSN_ADAPTER_H_

#include <tuple>
#include <type_traits>

#include "berberis/backend/x86_64/machine_ir.h"

// Generates the Gen<InsnTemplate>(args...) overloads for the ARM64 heavy
// optimizer's frontend. A braced argument such as
//   Gen<x86_64::MovssXRegOp>(reg, {.base = arg, .disp = offset})
// cannot reach a plain `Args&&...` parameter pack, so one overload per operand
// count has to be declared; this macro writes them out.
//
// The backend has its own BERBERIS_DECLARE_MACHINE_INSN_ADAPTER, but it hands
// the forwarder a single type, x86_64::MachineInsn<DeviceInsnInfo, kSSAMode>,
// and a std::tuple -- upstream moved SSA splitting into MachineInsn and the
// MachineIRBuilder, so its forwarder needs no SSA argument. The ARM64 frontend
// still splits by hand in its own Gen(), driven by a kSSAMode template
// argument that 263 of its call sites set explicitly, and MachineInsn must do
// no splitting of its own underneath that. Expressing those two requirements
// needs the pre-split shape of the macro, which upstream no longer has, so it
// lives here. It pins the backend's kSSAMode to x86_64::kForceNoSSA, which
// keeps a kUseDef operand in a single register the way Android 16 did.
//
// Should the frontend ever adopt the backend's splitting -- dropping its own
// Gen() for a forwarder to MachineIRBuilder::GenMachineInsn, the way the
// riscv64 frontend does -- this header goes away with it. That is a codegen
// change across every Gen<> call site and wants the benchmark and sample gates
// of its own, not a port.

// Android 16's MachineInsn carried no SSA mode, so a kUseDef operand occupied a
// single register. kForceNoSSA reproduces that layout and satisfies the
// backend's static_assert at the same time: it yields the enum kNoSSA for an
// instruction that has kUseDef operands and the defaulted bool false for one
// that does not, which is exactly what the assert demands of each.
#define BERBERIS_ARM64_INSN_SSA_MODE \
  x86_64::kForceNoSSA<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo>

// The adapter's ResultType argument has to arrive as one preprocessor token:
// spelling MachineInsn<DeviceInsnInfo, mode>::OutputArgsTuple at the call site
// would be split at that comma and arrive as two arguments.
#define BERBERIS_ARM64_INSN_OUTPUT_TUPLE                                                 \
  typename x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, \
                               BERBERIS_ARM64_INSN_SSA_MODE>::OutputArgsTuple

#define BERBERIS_DECLARE_ARM64_MACHINE_INSN_ADAPTER(                                                     \
    AdapterName, ExtraArgs, TupleClass, TupleName, ResultType, ForwarderName, ExtraForward)        \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName()                                                                                    \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 0,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward>();                                 \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 1,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName)>(arg0);                      \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 2,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName)>(arg0, arg1);                \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 3,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName)>(arg0, arg1, arg2);          \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 4,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName)>(arg0, arg1, arg2, arg3);    \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 5,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4);                                                             \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 6,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5);                                                       \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(6, TupleClass, TupleName) arg6)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 7,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(6, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5, arg6);                                                 \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(6, TupleClass, TupleName) arg6,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(7, TupleClass, TupleName) arg7)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 8,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(6, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(7, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);                                           \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(6, TupleClass, TupleName) arg6,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(7, TupleClass, TupleName) arg7,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(8, TupleClass, TupleName) arg8)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 9,  \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(6, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(7, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(8, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);                                     \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs> \
  AdapterName(BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName) arg0,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName) arg1,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName) arg2,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName) arg3,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName) arg4,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName) arg5,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(6, TupleClass, TupleName) arg6,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(7, TupleClass, TupleName) arg7,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(8, TupleClass, TupleName) arg8,                  \
              BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(9, TupleClass, TupleName) arg9)                  \
      ->std::enable_if_t<std::tuple_size_v<typename x86_64::TupleClass<typename InsnType<          \
                             typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>::TupleName> == 10, \
                         ResultType> {                                                             \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(0, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(1, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(2, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(3, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(4, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(5, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(6, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(7, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(8, TupleClass, TupleName),                             \
        BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(9, TupleClass, TupleName)>(                            \
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);                               \
  }                                                                                                \
                                                                                                   \
  template <template <typename> typename InsnType BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraArgs, \
            typename... Args>                                                                      \
  AdapterName(Args... args)->ResultType {                                                          \
    return ForwarderName<                                                                          \
        x86_64::MachineInsn<typename InsnType<typename CodeEmitter::Assemblers>::DeviceInsnInfo, BERBERIS_ARM64_INSN_SSA_MODE>   \
            BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE ExtraForward,                                    \
        Args...>(args...);                                                                         \
  }

#define BERBERIS_DECLARE_ARM64_MACHINE_INSN_TUPLE(N, TupleClass, TupleName)  \
  std::tuple_element_t<N,                                                    \
                       typename x86_64::TupleClass<                          \
                           typename InsnType<                                \
                               typename CodeEmitter::Assemblers>::DeviceInsnInfo, \
                           BERBERIS_ARM64_INSN_SSA_MODE>::TupleName>

#define BERBERIS_DECLARE_ARM64_MACHINE_INSN_UNBRACE(...) __VA_ARGS__

#endif  // BERBERIS_HEAVY_OPTIMIZER_ARM64_MACHINE_INSN_ADAPTER_H_
