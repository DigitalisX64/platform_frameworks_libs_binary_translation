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

#include "gtest/gtest.h"

#include "berberis/backend/x86_64/machine_insn_intrinsics.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/intrinsics_bindings.h"

namespace berberis {

namespace {

// TEST(MachineInsnIntrinsicsTest, HasNMem)
static_assert(x86_64::has_n_mem_v<
              1,
              ArgTraits<TmpArg,
                        machine_insn_info::OperandInfo<machine_insn_info::Mem32,
                                                       machine_insn_info::kDefEarlyClobber>>>);
static_assert(!x86_64::has_n_mem_v<1>);
static_assert(!x86_64::has_n_mem_v<
              1,
              ArgTraits<TmpArg,
                        machine_insn_info::OperandInfo<machine_insn_info::GeneralReg32,
                                                       machine_insn_info::kDefEarlyClobber>>>);
static_assert(
    x86_64::has_n_mem_v<2,
                        ArgTraits<TmpArg,
                                  machine_insn_info::OperandInfo<machine_insn_info::Mem32,
                                                                 machine_insn_info::kUse>>,
                        ArgTraits<TmpArg,
                                  machine_insn_info::OperandInfo<machine_insn_info::Mem32,
                                                                 machine_insn_info::kDef>>>);
static_assert(!x86_64::has_n_mem_v<
              2,
              ArgTraits<TmpArg,
                        machine_insn_info::OperandInfo<machine_insn_info::Mem32,
                                                       machine_insn_info::kDefEarlyClobber>>>);

// TEST(MachineInsnIntrinsicsTest, ConstructorArgs)
static_assert(
    std::is_same_v<x86_64::constructor_args_t<ArgTraits<
                       TmpArg,
                       machine_insn_info::OperandInfo<machine_insn_info::Mem64,
                                                      machine_insn_info::kDefEarlyClobber>>>,
                   std::tuple<MachineReg, int32_t>>);
static_assert(
    std::is_same_v<x86_64::constructor_args_t<ArgTraits<
                       TmpArg,
                       machine_insn_info::OperandInfo<machine_insn_info::GeneralReg64,
                                                      machine_insn_info::kDefEarlyClobber>>>,
                   std::tuple<MachineReg>>);
static_assert(
    std::is_same_v<
        x86_64::constructor_args_t<ArgTraits<
            InArg<0>,
            machine_insn_info::OperandInfo<machine_insn_info::Imm32, machine_insn_info::kUse>>>,
        std::tuple<int32_t>>);
static_assert(std::is_same_v<
              x86_64::constructor_args_t<
                  ArgTraits<InArg<0>,
                            machine_insn_info::OperandInfo<machine_insn_info::Imm16,
                                                           machine_insn_info::kUse>>,
                  ArgTraits<TmpArg,
                            machine_insn_info::OperandInfo<machine_insn_info::Mem64,
                                                           machine_insn_info::kDefEarlyClobber>>,
                  ArgTraits<TmpArg,
                            machine_insn_info::OperandInfo<machine_insn_info::GeneralReg64,
                                                           machine_insn_info::kDefEarlyClobber>>>,
              std::tuple<int16_t, MachineReg, int32_t, MachineReg>>);

}  // namespace

}  // namespace berberis
