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

// This inline-header is intended be included into a source file testing the correctness of
// riscv64 instructions execution by an interpreter or a jit-translator.
//
// Assumptions list:
//
// 1. Includes
//
// #include "gtest/gtest.h"
//
// #include <cstdint>
// #include <initializer_list>
// #include <tuple>
// #include <vector>
//
// #include "berberis/base/bit_util.h"
// #include "berberis/guest_state/guest_addr.h"
// #include "berberis/guest_state/guest_state_riscv64.h"
//
// 2. RunOneInstruction is defined and implemented
//
// 3. TESTSUITE macro is defined

#ifndef TESTSUITE
#error "TESTSUITE is undefined"
#endif

#if !(defined(TESTING_INTERPRETER) || defined(TESTING_LITE_TRANSLATOR) || \
      defined(TESTING_HEAVY_OPTIMIZER))
#error \
    "One of TESTING_INTERPRETER, TESTING_LITE_TRANSLATOR, TESTING_HEAVY_OPTIMIZER must be defined"
#endif

namespace {

// TODO(b/276787675): remove these files from interpreter when they are no longer needed there.
// Maybe extract FPvalueToFPReg and TupleMap to a separate header?
inline constexpr class FPValueToFPReg {
 public:
  uint64_t operator()(uint64_t value) const { return value; }
  uint64_t operator()(float value) const {
    return bit_cast<uint32_t>(value) | 0xffff'ffff'0000'0000;
  }
  uint64_t operator()(double value) const { return bit_cast<uint64_t>(value); }
} kFPValueToFPReg;

// Helper function for the unit tests. Can be used to normalize values before processing.
//
// “container” is supposed to be container of tuples, e.g. std::initializer_list<std::tuple<…>>.
// “transformer” would be applied to the individual elements of tuples in the following loop:
//
//   for (auto& [value1, value2, value3] : TupleMap(container, [](auto value){ return …; })) {
//     …
//   }
//
// Returns vector of tuples where each tuple element is processed by transformer.
template <typename ContainerType, typename Transformer>
decltype(auto) TupleMap(const ContainerType& container, const Transformer& transformer) {
  using std::begin;

  auto transform_tuple_func = [&transformer](auto&&... value) {
    return std::tuple{transformer(value)...};
  };

  std::vector<decltype(std::apply(transform_tuple_func, *begin(container)))> result;

  for (const auto& tuple : container) {
    result.push_back(std::apply(transform_tuple_func, tuple));
  }

  return result;
}

void RaiseFeExceptForGuestFlags(uint8_t riscv_fflags) {
  EXPECT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
  if (riscv_fflags & FPFlags::NX) {
    EXPECT_EQ(feraiseexcept(FE_INEXACT), 0);
  }
  if (riscv_fflags & FPFlags::UF) {
    EXPECT_EQ(feraiseexcept(FE_UNDERFLOW), 0);
  }
  if (riscv_fflags & FPFlags::OF) {
    EXPECT_EQ(feraiseexcept(FE_OVERFLOW), 0);
  }
  if (riscv_fflags & FPFlags::DZ) {
    EXPECT_EQ(feraiseexcept(FE_DIVBYZERO), 0);
  }
  if (riscv_fflags & FPFlags::NV) {
    EXPECT_EQ(feraiseexcept(FE_INVALID), 0);
  }
}

void TestFeExceptForGuestFlags(uint8_t riscv_fflags) {
  EXPECT_EQ(bool(riscv_fflags & FPFlags::NX), bool(fetestexcept(FE_INEXACT)));
  EXPECT_EQ(bool(riscv_fflags & FPFlags::UF), bool(fetestexcept(FE_UNDERFLOW)));
  EXPECT_EQ(bool(riscv_fflags & FPFlags::OF), bool(fetestexcept(FE_OVERFLOW)));
  EXPECT_EQ(bool(riscv_fflags & FPFlags::DZ), bool(fetestexcept(FE_DIVBYZERO)));
  EXPECT_EQ(bool(riscv_fflags & FPFlags::NV), bool(fetestexcept(FE_INVALID)));
}

}  // namespace

class TESTSUITE : public ::testing::Test {
 public:
  TESTSUITE()
      : state_{
            .cpu = {.vtype = uint64_t{1} << 63, .frm = intrinsics::GuestModeFromHostRounding()}} {}

  template <uint8_t kInsnSize = 4>
  void RunInstruction(const uint32_t& insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    EXPECT_TRUE(RunOneInstruction<kInsnSize>(&state_, state_.cpu.insn_addr + kInsnSize));
  }

  // Compressed Instructions.

  template <RegisterType register_type, uint64_t expected_result, uint8_t kTargetReg>
  void TestCompressedStore(uint16_t insn_bytes, uint64_t offset) {
    store_area_ = 0;
    SetXReg<kTargetReg>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - offset));
    SetReg<register_type, 9>(state_.cpu, kDataToLoad);
    RunInstruction<2>(insn_bytes);
    EXPECT_EQ(store_area_, expected_result);
  }

  template <RegisterType register_type, uint64_t expected_result, uint8_t kSourceReg>
  void TestCompressedLoad(uint16_t insn_bytes, uint64_t offset) {
    SetXReg<kSourceReg>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - offset));
    RunInstruction<2>(insn_bytes);
    EXPECT_EQ((GetReg<register_type, 9>(state_.cpu)), expected_result);
  }

  void TestCAddi(uint16_t insn_bytes, uint64_t expected_increment) {
    SetXReg<2>(state_.cpu, 1);
    RunInstruction<2>(insn_bytes);
    EXPECT_EQ(GetXReg<2>(state_.cpu), 1 + expected_increment);
  }

  void TestCAddi16sp(uint16_t insn_bytes, uint64_t expected_offset) {
    SetXReg<2>(state_.cpu, 1);
    RunInstruction<2>(insn_bytes);
    EXPECT_EQ(GetXReg<2>(state_.cpu), 1 + expected_offset);
  }

  void TestLi(uint32_t insn_bytes, uint64_t expected_result) {
    RunInstruction<2>(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  void TestCAddi4spn(uint16_t insn_bytes, uint64_t expected_offset) {
    SetXReg<2>(state_.cpu, 1);
    RunInstruction<2>(insn_bytes);
    EXPECT_EQ(GetXReg<9>(state_.cpu), 1 + expected_offset);
  }

  void TestCBeqzBnez(uint16_t insn_bytes, uint64_t value, int16_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    if (expected_offset == 0) {
      // Emit pending signal so we don't get stuck in an infinite loop.
      state_.pending_signals_status = kPendingSignalsPresent;
    } else {
      state_.pending_signals_status = kPendingSignalsDisabled;
    }
    SetXReg<9>(state_.cpu, value);
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + expected_offset));
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
  }

  void TestCMiscAlu(uint16_t insn_bytes,
                    std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      SetXReg<8>(state_.cpu, arg1);
      SetXReg<9>(state_.cpu, arg2);
      RunInstruction<2>(insn_bytes);
      EXPECT_EQ(GetXReg<8>(state_.cpu), expected_result);
    }
  }

  void TestCMiscAluSingleInput(uint16_t insn_bytes,
                               std::initializer_list<std::tuple<uint64_t, uint64_t>> args) {
    for (auto [arg1, expected_result] : args) {
      SetXReg<8>(state_.cpu, arg1);
      RunInstruction<2>(insn_bytes);
      EXPECT_EQ(GetXReg<8>(state_.cpu), expected_result);
    }
  }

  void TestCMiscAluImm(uint16_t insn_bytes, uint64_t value, uint64_t expected_result) {
    SetXReg<9>(state_.cpu, value);
    RunInstruction<2>(insn_bytes);
    EXPECT_EQ(GetXReg<9>(state_.cpu), expected_result);
  }

  void TestCJ(uint16_t insn_bytes, int16_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    if (expected_offset == 0) {
      // Emit pending signal so we don't get stuck in an infinite loop.
      state_.pending_signals_status = kPendingSignalsPresent;
    } else {
      state_.pending_signals_status = kPendingSignalsDisabled;
    }
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + expected_offset));
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
  }

  void TestCOp(uint32_t insn_bytes,
               std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      SetXReg<1>(state_.cpu, arg1);
      SetXReg<2>(state_.cpu, arg2);
      RunInstruction<2>(insn_bytes);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  // Non-Compressed Instructions.

  void TestFFlagsOnGuestAndHost(uint8_t expected_guest_fflags) {
    // Read fflags register.
    RunInstruction(0x0010'2173);  // frflags x2
    EXPECT_EQ(GetXReg<2>(state_.cpu), expected_guest_fflags);

    // Check corresponding fenv exception flags on host.
    TestFeExceptForGuestFlags(expected_guest_fflags);
  }

  void TestFCsr(uint32_t insn_bytes,
                uint8_t fcsr_to_set,
                uint8_t expected_fcsr,
                uint8_t expected_cpustate_frm) {
    state_.cpu.frm =
        0b100u;  // Pass non-zero frm to ensure that we don't accidentally rely on it being zero.
    SetXReg<3>(state_.cpu, fcsr_to_set);
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<2>(state_.cpu), 0b1000'0000ULL | expected_fcsr);
    EXPECT_EQ(state_.cpu.frm, expected_cpustate_frm);
  }

  void TestFrm(uint32_t insn_bytes, uint8_t frm_to_set, uint8_t expected_rm) {
    state_.cpu.frm = 0b001u;
    SetXReg<3>(state_.cpu, frm_to_set);
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<2>(state_.cpu), 0b001u);
    EXPECT_EQ(state_.cpu.frm, expected_rm);
  }

  void TestOp(uint32_t insn_bytes,
              std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      SetXReg<2>(state_.cpu, arg1);
      SetXReg<3>(state_.cpu, arg2);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void TestOpFp(uint32_t insn_bytes, std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg1, arg2, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      SetFReg<2>(state_.cpu, arg1);
      SetFReg<3>(state_.cpu, arg2);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
    }
  }

  void TestOpImm(uint32_t insn_bytes,
                 std::initializer_list<std::tuple<uint64_t, uint16_t, uint64_t>> args) {
    for (auto [arg1, imm, expected_result] : args) {
      CHECK_LE(imm, 63);
      uint32_t insn_bytes_with_immediate = insn_bytes | imm << 20;
      SetXReg<2>(state_.cpu, arg1);
      RunInstruction(insn_bytes_with_immediate);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  void TestAuipc(uint32_t insn_bytes, uint64_t expected_offset) {
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_offset + ToGuestAddr(&insn_bytes));
  }

  void TestLui(uint32_t insn_bytes, uint64_t expected_result) {
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  void TestBranch(uint32_t insn_bytes,
                  std::initializer_list<std::tuple<uint64_t, uint64_t, int8_t>> args) {
    auto code_start = ToGuestAddr(&insn_bytes);
    for (auto [arg1, arg2, expected_offset] : args) {
      state_.cpu.insn_addr = code_start;
      SetXReg<1>(state_.cpu, arg1);
      SetXReg<2>(state_.cpu, arg2);
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + expected_offset));
      EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    }
  }

  void TestJumpAndLink(uint32_t insn_bytes, int8_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + expected_offset));
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + 4);
  }

  void TestLoad(uint32_t insn_bytes, uint64_t expected_result) {
    // Offset is always 8.
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  // kLinkRegisterOffsetIfUsed is size of instruction or 0 if instruction does not link register.
  template <uint8_t kLinkRegisterOffsetIfUsed>
  void TestJumpAndLinkRegister(uint32_t insn_bytes, uint64_t base_disp, int64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<1>(state_.cpu, 0);
    SetXReg<2>(state_.cpu, code_start + base_disp);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + expected_offset));
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    if constexpr (kLinkRegisterOffsetIfUsed == 0) {
      EXPECT_EQ(GetXReg<1>(state_.cpu), 0UL);
    } else {
      EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + kLinkRegisterOffsetIfUsed);
    }
  }

  void TestStore(uint32_t insn_bytes, uint64_t expected_result) {
    // Offset is always 8.
    SetXReg<1>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - 8));
    SetXReg<2>(state_.cpu, kDataToStore);
    store_area_ = 0;
    RunInstruction(insn_bytes);
    EXPECT_EQ(store_area_, expected_result);
  }

  template <typename... Types>
  void TestFma(uint32_t insn_bytes, std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg1, arg2, arg3, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      SetFReg<2>(state_.cpu, arg1);
      SetFReg<3>(state_.cpu, arg2);
      SetFReg<4>(state_.cpu, arg3);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
    }
  }

#if (defined(TESTING_INTERPRETER) || defined(TESTING_HEAVY_OPTIMIZER))

  void TestAtomicLoad(uint32_t insn_bytes,
                      const uint64_t* const data_to_load,
                      uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(data_to_load));
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetXReg<2>(state_.cpu), expected_result);
    EXPECT_EQ(state_.cpu.reservation_address, ToGuestAddr(data_to_load));
    // We always reserve the full 64-bit range of the reservation address.
    EXPECT_EQ(state_.cpu.reservation_value, *data_to_load);
  }

  template <typename T>
  void TestAtomicStore(uint32_t insn_bytes, T expected_result) {
    store_area_ = ~uint64_t{0};
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdead'beef);
    state_.cpu.reservation_address = ToGuestAddr(&store_area_);
    state_.cpu.reservation_value = store_area_;
    MemoryRegionReservation::SetOwner(ToGuestAddr(&store_area_), &state_.cpu);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(static_cast<T>(store_area_), expected_result);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 0u);
  }

  void TestAtomicStoreNoLoadFailure(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdead'beef);
    store_area_ = 0;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(store_area_, 0u);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 1u);
  }

  void TestAtomicStoreDifferentLoadFailure(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdead'beef);
    state_.cpu.reservation_address = ToGuestAddr(&kDataToStore);
    state_.cpu.reservation_value = 0;
    MemoryRegionReservation::SetOwner(ToGuestAddr(&kDataToStore), &state_.cpu);
    store_area_ = 0;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(store_area_, 0u);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 1u);
  }

#endif  // (defined(TESTING_INTERPRETER) || defined(TESTING_HEAVY_OPTIMIZER))

  void TestAmo(uint32_t insn_bytes,
               uint64_t arg1,
               uint64_t arg2,
               uint64_t expected_result,
               uint64_t expected_memory) {
    // Copy arg1 into store_area_
    store_area_ = arg1;
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_)));
    SetXReg<3>(state_.cpu, arg2);
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    EXPECT_EQ(store_area_, expected_memory);
  }

  void TestAmo(uint32_t insn_bytes32, uint32_t insn_bytes64, uint64_t expected_memory) {
    TestAmo(insn_bytes32,
            0xffff'eeee'dddd'ccccULL,
            0xaaaa'bbbb'cccc'ddddULL,
            0xffff'ffff'dddd'ccccULL,
            0xffff'eeee'0000'0000 | uint32_t(expected_memory));
    TestAmo(insn_bytes64,
            0xffff'eeee'dddd'ccccULL,
            0xaaaa'bbbb'cccc'ddddULL,
            0xffff'eeee'dddd'ccccULL,
            expected_memory);
  }

  template <typename... Types>
  void TestFmvFloatToInteger(uint32_t insn_bytes,
                             std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      SetFReg<1>(state_.cpu, arg);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void TestFmvIntegerToFloat(uint32_t insn_bytes,
                             std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : args) {
      SetXReg<1>(state_.cpu, arg);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetFReg<1>(state_.cpu), kFPValueToFPReg(expected_result));
    }
  }

  template <typename... Types>
  void TestOpFpGpRegisterTarget(uint32_t insn_bytes,
                                std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg1, arg2, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      SetFReg<2>(state_.cpu, arg1);
      SetFReg<3>(state_.cpu, arg2);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void TestOpFpGpRegisterTargetSingleInput(uint32_t insn_bytes,
                                           std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      SetFReg<2>(state_.cpu, arg);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void TestOpFpGpRegisterSourceSingleInput(uint32_t insn_bytes,
                                           std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      SetXReg<2>(state_.cpu, arg);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void TestOpFpSingleInput(uint32_t insn_bytes, std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      SetFReg<2>(state_.cpu, arg);
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
    }
  }

  void TestLoadFp(uint32_t insn_bytes, uint64_t expected_result) {
    // Offset is always 8.
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
  }

  void TestStoreFp(uint32_t insn_bytes, uint64_t expected_result) {
    // Offset is always 8.
    SetXReg<1>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - 8));
    SetFReg<2>(state_.cpu, kDataToStore);
    store_area_ = 0;
    RunInstruction(insn_bytes);
    EXPECT_EQ(store_area_, expected_result);
  }

  void TestVsetvl(
      uint32_t insn_bytes,
      std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>>
          args) {
    for (auto [vl_orig, vtype_orig, avl, vtype_new, vl_expected, vtype_expected] : args) {
      state_.cpu.vl = vl_orig;
      state_.cpu.vtype = vtype_orig;
      SetXReg<1>(state_.cpu, ~0ULL);
      SetXReg<2>(state_.cpu, avl);
      SetXReg<3>(state_.cpu, vtype_new);
      RunInstruction(insn_bytes);
      if (insn_bytes & 0b11111'0000000) {
        EXPECT_EQ(GetXReg<1>(state_.cpu), vl_expected);
      } else {
        EXPECT_EQ(GetXReg<1>(state_.cpu), ~0ULL);
      }
      EXPECT_EQ(state_.cpu.vl, vl_expected);
      EXPECT_EQ(state_.cpu.vtype, vtype_expected);
    }
  }

 protected:
  static constexpr uint64_t kDataToLoad{0xffff'eeee'dddd'ccccULL};
  static constexpr uint64_t kDataToStore = kDataToLoad;
  uint64_t store_area_;
  ThreadState state_;
};

// Tests for Compressed Instructions.
template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoadOrStore32bit(TESTSUITE* that) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 2;
      uint8_t i2 : 1;
      uint8_t i3_i5 : 3;
      uint8_t i6 : 1;
    } i_bits;
  };
  for (offset = uint8_t{0}; offset < uint8_t{128}; offset += 4) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rd : 3;
        uint8_t i6 : 1;
        uint8_t i2 : 1;
        uint8_t rs : 3;
        uint8_t i3_i5 : 3;
        uint8_t high_opcode : 3;
      } __attribute__((__packed__));
    } o_bits = {
        .low_opcode = 0b00,
        .rd = 1,
        .i6 = i_bits.i6,
        .i2 = i_bits.i2,
        .rs = 0,
        .i3_i5 = i_bits.i3_i5,
        .high_opcode = 0b000,
    };
    (that->*execute_instruction_func)(o_bits.parcel | opcode, offset);
  }
}

TEST_F(TESTSUITE, CompressedLoadAndStores32bit) {
  // c.Lw
  TestCompressedLoadOrStore32bit<
      0b010'000'000'00'000'00,
      &TESTSUITE::TestCompressedLoad<RegisterType::kReg,
                                     static_cast<uint64_t>(static_cast<int32_t>(kDataToLoad)),
                                     8>>(this);
  // c.Sw
  TestCompressedLoadOrStore32bit<
      0b110'000'000'00'000'00,
      &TESTSUITE::TestCompressedStore<RegisterType::kReg,
                                      static_cast<uint64_t>(static_cast<uint32_t>(kDataToLoad)),
                                      8>>(this);
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoadOrStore64bit(TESTSUITE* that) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 3;
      uint8_t i3_i5 : 3;
      uint8_t i6_i7 : 2;
    } i_bits;
  };
  for (offset = int16_t{0}; offset < int16_t{256}; offset += 8) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rd : 3;
        uint8_t i6_i7 : 2;
        uint8_t rs : 3;
        uint8_t i3_i5 : 3;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b00,
        .rd = 1,
        .i6_i7 = i_bits.i6_i7,
        .rs = 0,
        .i3_i5 = i_bits.i3_i5,
        .high_opcode = 0b000,
    };
    (that->*execute_instruction_func)(o_bits.parcel | opcode, offset);
  }
}

TEST_F(TESTSUITE, CompressedLoadAndStores) {
  // c.Ld
  TestCompressedLoadOrStore64bit<
      0b011'000'000'00'000'00,
      &TESTSUITE::TestCompressedLoad<RegisterType::kReg, kDataToLoad, 8>>(this);
  // c.Sd
  TestCompressedLoadOrStore64bit<
      0b111'000'000'00'000'00,
      &TESTSUITE::TestCompressedStore<RegisterType::kReg, kDataToLoad, 8>>(this);
  // c.Fld
  TestCompressedLoadOrStore64bit<
      0b001'000'000'00'000'00,
      &TESTSUITE::TestCompressedLoad<RegisterType::kFpReg, kDataToLoad, 8>>(this);
  // c.Fsd
  TestCompressedLoadOrStore64bit<
      0b101'000'000'00'000'00,
      &TESTSUITE::TestCompressedStore<RegisterType::kFpReg, kDataToLoad, 8>>(this);
}

TEST_F(TESTSUITE, TestCompressedStore32bitsp) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 2;
      uint8_t i2_i5 : 4;
      uint8_t i6_i7 : 2;
    } i_bits;
  };
  for (offset = uint16_t{0}; offset < uint16_t{256}; offset += 4) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rs2 : 5;
        uint8_t i6_i7 : 2;
        uint8_t i2_i5 : 4;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b10,
        .rs2 = 9,
        .i6_i7 = i_bits.i6_i7,
        .i2_i5 = i_bits.i2_i5,
        .high_opcode = 0b110,
    };
    // c.Swsp
    TestCompressedStore<RegisterType::kReg,
                        static_cast<uint64_t>(static_cast<uint32_t>(kDataToStore)),
                        2>(o_bits.parcel, offset);
  }
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedStore64bitsp(TESTSUITE* that) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 3;
      uint8_t i3_i5 : 3;
      uint8_t i6_i8 : 3;
    } i_bits;
  };
  for (offset = uint16_t{0}; offset < uint16_t{512}; offset += 8) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rs2 : 5;
        uint8_t i6_i8 : 3;
        uint8_t i3_i5 : 3;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b10,
        .rs2 = 9,
        .i6_i8 = i_bits.i6_i8,
        .i3_i5 = i_bits.i3_i5,
        .high_opcode = 0b101,
    };
    (that->*execute_instruction_func)(o_bits.parcel | opcode, offset);
  }
}

TEST_F(TESTSUITE, TestCompressedStore64bitsp) {
  // c.Sdsp
  TestCompressedStore64bitsp<0b011'000'000'00'000'00,
                             &TESTSUITE::TestCompressedStore<RegisterType::kReg, kDataToStore, 2>>(
      this);
  // c.Fsdsp
  TestCompressedStore64bitsp<
      0b001'000'000'00'000'00,
      &TESTSUITE::TestCompressedStore<RegisterType::kFpReg, kDataToStore, 2>>(this);
}

TEST_F(TESTSUITE, TestCompressedLoad32bitsp) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 2;
      uint8_t i2_i4 : 3;
      uint8_t i5 : 1;
      uint8_t i6_i7 : 2;
    } i_bits;
  };
  for (offset = uint16_t{0}; offset < uint16_t{256}; offset += 4) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i6_i7 : 2;
        uint8_t i2_i4 : 3;
        uint8_t rd : 5;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b10,
        .i6_i7 = i_bits.i6_i7,
        .i2_i4 = i_bits.i2_i4,
        .rd = 9,
        .i5 = i_bits.i5,
        .high_opcode = 0b010,
    };
    // c.Lwsp
    TestCompressedLoad<RegisterType::kReg,
                       static_cast<uint64_t>(static_cast<int32_t>(kDataToLoad)),
                       2>(o_bits.parcel, offset);
  }
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoad64bitsp(TESTSUITE* that) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 3;
      uint8_t i3_i4 : 2;
      uint8_t i5 : 1;
      uint8_t i6_i8 : 3;
    } i_bits;
  };
  for (offset = uint16_t{0}; offset < uint16_t{512}; offset += 8) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i6_i8 : 3;
        uint8_t i3_i4 : 2;
        uint8_t rd : 5;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b10,
        .i6_i8 = i_bits.i6_i8,
        .i3_i4 = i_bits.i3_i4,
        .rd = 9,
        .i5 = i_bits.i5,
        .high_opcode = 0b001,
    };
    (that->*execute_instruction_func)(o_bits.parcel | opcode, offset);
  }
}

TEST_F(TESTSUITE, TestCompressedLoad64bitsp) {
  // c.Ldsp
  TestCompressedLoad64bitsp<0b011'000'000'00'000'00,
                            &TESTSUITE::TestCompressedLoad<RegisterType::kReg, kDataToLoad, 2>>(
      this);
  // c.Fldsp
  TestCompressedLoad64bitsp<0b001'000'000'00'000'00,
                            &TESTSUITE::TestCompressedLoad<RegisterType::kFpReg, kDataToLoad, 2>>(
      this);
}

TEST_F(TESTSUITE, CAddi) {
  union {
    int8_t offset;
    struct [[gnu::packed]] {
      uint8_t i4_i0 : 5;
      uint8_t i5 : 1;
    } i_bits;
  };
  for (offset = int8_t{-32}; offset < int8_t{31}; offset++) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i4_i0 : 5;
        uint8_t r : 5;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      } __attribute__((__packed__));
    } o_bits = {
        .low_opcode = 0,
        .i4_i0 = i_bits.i4_i0,
        .r = 2,
        .i5 = i_bits.i5,
        .high_opcode = 0,
    };
    // c.Addi
    TestCAddi(o_bits.parcel | 0b0000'0000'0000'0001, offset);
    // c.Addiw
    TestCAddi(o_bits.parcel | 0b0010'0000'0000'0001, offset);
  }
}

TEST_F(TESTSUITE, CAddi16sp) {
  union {
    int16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 4;
      uint8_t i4 : 1;
      uint8_t i5 : 1;
      uint8_t i6 : 1;
      uint8_t i7 : 1;
      uint8_t i8 : 1;
      uint8_t i9 : 1;
    } i_bits;
  };
  for (offset = int16_t{-512}; offset < int16_t{512}; offset += 16) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i5 : 1;
        uint8_t i7 : 1;
        uint8_t i8 : 1;
        uint8_t i6 : 1;
        uint8_t i4 : 1;
        uint8_t rd : 5;
        uint8_t i9 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b01,
        .i5 = i_bits.i5,
        .i7 = i_bits.i7,
        .i8 = i_bits.i8,
        .i6 = i_bits.i6,
        .i4 = i_bits.i4,
        .rd = 2,
        .i9 = i_bits.i9,
        .high_opcode = 0b011,
    };
    TestCAddi16sp(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CLui) {
  union {
    int32_t offset;
    struct [[gnu::packed]] {
      uint8_t : 12;
      uint8_t i12_i16 : 5;
      uint8_t i17 : 1;
    } i_bits;
  };
  for (offset = int32_t{-131072}; offset < int32_t{131072}; offset += 4096) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i12_i16 : 5;
        uint8_t rd : 5;
        uint8_t i17 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b01,
        .i12_i16 = i_bits.i12_i16,
        .rd = 1,
        .i17 = i_bits.i17,
        .high_opcode = 0b011,
    };
    TestLi(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CLi) {
  union {
    int8_t offset;
    struct [[gnu::packed]] {
      uint8_t i0_i4 : 5;
      uint8_t i5 : 1;
    } i_bits;
  };
  for (offset = int8_t{-32}; offset < int8_t{32}; offset++) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i0_i4 : 5;
        uint8_t rd : 5;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b01,
        .i0_i4 = i_bits.i0_i4,
        .rd = 1,
        .i5 = i_bits.i5,
        .high_opcode = 0b010,
    };
    TestLi(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CAddi4spn) {
  union {
    int16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 2;
      uint8_t i2 : 1;
      uint8_t i3 : 1;
      uint8_t i4 : 1;
      uint8_t i5 : 1;
      uint8_t i6 : 1;
      uint8_t i7 : 1;
      uint8_t i8 : 1;
      uint8_t i9 : 1;
    } i_bits;
  };
  for (offset = int16_t{4}; offset < int16_t{1024}; offset += 4) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rd : 3;
        uint8_t i3 : 1;
        uint8_t i2 : 1;
        uint8_t i6 : 1;
        uint8_t i7 : 1;
        uint8_t i8 : 1;
        uint8_t i9 : 1;
        uint8_t i4 : 1;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b00,
        .rd = 1,
        .i3 = i_bits.i3,
        .i2 = i_bits.i2,
        .i6 = i_bits.i6,
        .i7 = i_bits.i7,
        .i8 = i_bits.i8,
        .i9 = i_bits.i9,
        .i4 = i_bits.i4,
        .i5 = i_bits.i5,
        .high_opcode = 0b000,
    };
    TestCAddi4spn(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CBeqzBnez) {
  union {
    int16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 1;
      uint8_t i1 : 1;
      uint8_t i2 : 1;
      uint8_t i3 : 1;
      uint8_t i4 : 1;
      uint8_t i5 : 1;
      uint8_t i6 : 1;
      uint8_t i7 : 1;
      uint8_t i8 : 1;
    } i_bits;
  };
  for (offset = int16_t{-256}; offset < int16_t{256}; offset += 8) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i5 : 1;
        uint8_t i1 : 1;
        uint8_t i2 : 1;
        uint8_t i6 : 1;
        uint8_t i7 : 1;
        uint8_t rs : 3;
        uint8_t i3 : 1;
        uint8_t i4 : 1;
        uint8_t i8 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0,
        .i5 = i_bits.i5,
        .i1 = i_bits.i1,
        .i2 = i_bits.i2,
        .i6 = i_bits.i6,
        .i7 = i_bits.i7,
        .rs = 1,
        .i3 = i_bits.i3,
        .i4 = i_bits.i4,
        .i8 = i_bits.i8,
        .high_opcode = 0,
    };
    TestCBeqzBnez(o_bits.parcel | 0b1100'0000'0000'0001, 0, offset);
    TestCBeqzBnez(o_bits.parcel | 0b1110'0000'0000'0001, 1, offset);
  }
}

TEST_F(TESTSUITE, CMiscAluInstructions) {
  // c.Sub
  TestCMiscAlu(0x8c05, {{42, 23, 19}});
  // c.Xor
  TestCMiscAlu(0x8c25, {{0b0101, 0b0011, 0b0110}});
  // c.Or
  TestCMiscAlu(0x8c45, {{0b0101, 0b0011, 0b0111}});
  // c.And
  TestCMiscAlu(0x8c65, {{0b0101, 0b0011, 0b0001}});
  // c.SubW
  TestCMiscAlu(0x9c05, {{42, 23, 19}});
  // c.AddW
  TestCMiscAlu(0x9c25, {{19, 23, 42}});
}

TEST_F(TESTSUITE, CBitManipInstructions) {
  // c.zext.h
  TestCMiscAluSingleInput(0x9c69, {{0xffff'ffff'ffff'fffe, 0xfffe}});
  // c.zext.w
  TestCMiscAluSingleInput(0x9c71, {{0xffff'ffff'ffff'fffe, 0xffff'fffe}});
  // c.zext.b
  TestCMiscAluSingleInput(0x9c61, {{0xffff'ffff'ffff'fffe, 0xfe}});
  // c.sext.b
  TestCMiscAluSingleInput(0x9c65, {{0b1111'1110, 0xffff'ffff'ffff'fffe}});
  // c.sext.h
  TestCMiscAluSingleInput(0x9c6d, {{0b1111'1111'1111'1110, 0xffff'ffff'ffff'fffe}});
}

TEST_F(TESTSUITE, CMiscAluImm) {
  union {
    uint8_t uimm;
    // Note: c.Andi uses sign-extended immediate while c.Srli/c.cSrain need zero-extended one.
    // If we store the value into uimm and read from imm compiler would do correct conversion.
    int8_t imm : 6;
    struct [[gnu::packed]] {
      uint8_t i0_i4 : 5;
      uint8_t i5 : 1;
    } i_bits;
  };
  for (uimm = uint8_t{0}; uimm < uint8_t{64}; uimm++) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i0_i4 : 5;
        uint8_t r : 3;
        uint8_t mid_opcode : 2;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0,
        .i0_i4 = i_bits.i0_i4,
        .r = 1,
        .mid_opcode = 0,
        .i5 = i_bits.i5,
        .high_opcode = 0,
    };
    // The o_bits.parcel here doesn't include opcodes and we are adding it in the function call.
    // c.Srli
    TestCMiscAluImm(o_bits.parcel | 0b1000'0000'0000'0001,
                    0x8000'0000'0000'0000ULL,
                    0x8000'0000'0000'0000ULL >> uimm);
    // c.Srai
    TestCMiscAluImm(o_bits.parcel | 0b1000'0100'0000'0001,
                    0x8000'0000'0000'0000LL,
                    ~0 ^ ((0x8000'0000'0000'0000 ^ ~0) >>
                          uimm));  // Avoid shifting negative numbers to avoid UB
    // c.Andi
    TestCMiscAluImm(o_bits.parcel | 0b1000'1000'0000'0001,
                    0xffff'ffff'ffff'ffffULL,
                    0xffff'ffff'ffff'ffffULL & imm);

    // Previous instructions use 3-bit register encoding where 0b000 is r8, 0b001 is r9, etc.
    // c.Slli uses 5-bit register encoding. Since we want it to also work with r9 in the test body
    // we add 0b01000 to register bits to mimic that shift-by-8.
    // c.Slli                                   vvvvvv adds 8 to r to handle rd' vs rd difference.
    TestCMiscAluImm(o_bits.parcel | 0b0000'0100'0000'0010,
                    0x0000'0000'0000'0001ULL,
                    0x0000'0000'0000'0001ULL << uimm);
  }
}

TEST_F(TESTSUITE, CJ) {
  union {
    int16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 1;
      uint8_t i1 : 1;
      uint8_t i2 : 1;
      uint8_t i3 : 1;
      uint8_t i4 : 1;
      uint8_t i5 : 1;
      uint8_t i6 : 1;
      uint8_t i7 : 1;
      uint8_t i8 : 1;
      uint8_t i9 : 1;
      uint8_t i10 : 1;
      uint8_t i11 : 1;
    } i_bits;
  };
  for (offset = int16_t{-2048}; offset < int16_t{2048}; offset += 2) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i5 : 1;
        uint8_t i1 : 1;
        uint8_t i2 : 1;
        uint8_t i3 : 1;
        uint8_t i7 : 1;
        uint8_t i6 : 1;
        uint8_t i10 : 1;
        uint8_t i8 : 1;
        uint8_t i9 : 1;
        uint8_t i4 : 1;
        uint8_t i11 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b01,
        .i5 = i_bits.i5,
        .i1 = i_bits.i1,
        .i2 = i_bits.i2,
        .i3 = i_bits.i3,
        .i7 = i_bits.i7,
        .i6 = i_bits.i6,
        .i10 = i_bits.i10,
        .i8 = i_bits.i8,
        .i9 = i_bits.i9,
        .i4 = i_bits.i4,
        .i11 = i_bits.i11,
        .high_opcode = 0b101,
    };
    TestCJ(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CJalr) {
  // C.Jr
  TestJumpAndLinkRegister<0>(0x8102, 42, 42);
  // C.Mv
  TestCOp(0x808a, {{0, 1, 1}});
  // C.Jalr
  TestJumpAndLinkRegister<2>(0x9102, 42, 42);
  // C.Add
  TestCOp(0x908a, {{1, 2, 3}});
}

// Tests for Non-Compressed Instructions.

TEST_F(TESTSUITE, CsrInstructions) {
  ScopedRoundingMode scoped_rounding_mode;
  // Csrrw x2, frm, 2
  TestFrm(0x0021'5173, 0, 2);
  // Csrrsi x2, frm, 2
  TestFrm(0x0021'6173, 0, 3);
  // Csrrci x2, frm, 1
  TestFrm(0x0020'f173, 0, 0);
}

constexpr uint8_t kFPFlagsAll = FPFlags::NX | FPFlags::UF | FPFlags::OF | FPFlags::DZ | FPFlags::NV;
// Ensure all trailing bits are set in kFPFlagsAll so that all combinations are possible.
static_assert(__builtin_ctz(~kFPFlagsAll) == 5);

// Automatically saves and restores fenv throughout the lifetime of a parent scope.
class ScopedFenv {
 public:
  ScopedFenv() { EXPECT_EQ(fegetenv(&env_), 0); }
  ~ScopedFenv() { EXPECT_EQ(fesetenv(&env_), 0); }

 private:
  fenv_t env_;
};

TEST_F(TESTSUITE, FFlagsRead) {
  ScopedFenv fenv;
  for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
    RaiseFeExceptForGuestFlags(fflags);
    RunInstruction(0x0010'2173);  // frflags x2
    EXPECT_EQ(GetXReg<2>(state_.cpu), fflags);
  }
}

TEST_F(TESTSUITE, FFlagsSwap) {
  ScopedFenv fenv;
  for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
    RaiseFeExceptForGuestFlags(fflags);
    // After swapping in 0 for flags, read fflags to verify.
    SetXReg<3>(state_.cpu, 0);
    RunInstruction(0x0011'9173);  // fsflags x2, x3
    EXPECT_EQ(GetXReg<2>(state_.cpu), fflags);
    TestFFlagsOnGuestAndHost(0u);
  }
}

TEST_F(TESTSUITE, FFlagsSwapImmediate) {
  ScopedFenv fenv;
  for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
    RaiseFeExceptForGuestFlags(fflags);
    // After swapping in 0 for flags, read fflags to verify.
    RunInstruction(0x0010'5173);  // fsflags x2, 0
    EXPECT_EQ(GetXReg<2>(state_.cpu), fflags);
    TestFFlagsOnGuestAndHost(0u);
  }
}

TEST_F(TESTSUITE, FFlagsWrite) {
  ScopedFenv fenv;
  for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
    SetXReg<3>(state_.cpu, fflags);
    RunInstruction(0x0011'9073);  // fsflags x3
    TestFFlagsOnGuestAndHost(fflags);
  }
}

TEST_F(TESTSUITE, FFlagsWriteImmediate) {
  ScopedFenv fenv;
  for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
    RunInstruction(0x0010'5073 | fflags << 15);  // fsflagsi 0 (+ fflags)
    TestFFlagsOnGuestAndHost(fflags);
  }
}

TEST_F(TESTSUITE, FFlagsClearBits) {
  ScopedFenv fenv;
  for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
    RaiseFeExceptForGuestFlags(kFPFlagsAll);
    SetXReg<3>(state_.cpu, fflags);
    RunInstruction(0x0011'b073);  // csrc fflags, x3
    // Read fflags to verify previous bitwise clear operation.
    TestFFlagsOnGuestAndHost(static_cast<uint8_t>(~fflags & kFPFlagsAll));
  }
}

TEST_F(TESTSUITE, FFlagsClearBitsImmediate) {
  ScopedFenv fenv;
  for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
    RaiseFeExceptForGuestFlags(kFPFlagsAll);
    RunInstruction(0x0010'7073 | fflags << 15);  // csrci fflags, 0 (+ fflags)
    // Read fflags to verify previous bitwise clear operation.
    TestFFlagsOnGuestAndHost(static_cast<uint8_t>(~fflags & kFPFlagsAll));
  }
}

TEST_F(TESTSUITE, FCsrRegister) {
  ScopedFenv fenv;
  for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
    RaiseFeExceptForGuestFlags(fflags);

    // Read and verify fflags, then replace with all flags.
    TestFCsr(0x0031'9173 /* fscsr x2,x3 */, fflags, fflags, 0);

    // Only read fcsr and verify fflags.
    TestFCsr(0x0030'2173 /* frcsr x2 */, /* ignored */ 0, fflags, /* expected_frm= */ 0b100u);
  }

  for (bool immediate_source : {true, false}) {
    for (uint8_t fflags = 0; fflags <= kFPFlagsAll; fflags++) {
      EXPECT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
      if (immediate_source) {
        TestFCsr(0x0030'5173 /* csrrwi x2,fcsr,0 */ | (fflags << 15), 0, 0, 0);
      } else {
        TestFCsr(0x0031'9173 /* fscsr x2,x3 */, 0b100'0000 | fflags, 0, /* expected_frm= */ 0b010u);
      }
      TestFFlagsOnGuestAndHost(fflags);
    }
  }
}

TEST_F(TESTSUITE, FsrRegister) {
  ScopedRoundingMode scoped_rounding_mode;
  int rounding[][2] = {{0, FE_TONEAREST},
                       {1, FE_TOWARDZERO},
                       {2, FE_DOWNWARD},
                       {3, FE_UPWARD},
                       {4, FE_TOWARDZERO},
                       // Only low three bits must be affecting output (for forward compatibility).
                       {8, FE_TONEAREST},
                       {9, FE_TOWARDZERO},
                       {10, FE_DOWNWARD},
                       {11, FE_UPWARD},
                       {12, FE_TOWARDZERO}};
  for (bool immediate_source : {true, false}) {
    for (auto [guest_rounding, host_rounding] : rounding) {
      if (immediate_source) {
        TestFrm(0x0020'5173 | (guest_rounding << 15), 0, guest_rounding & 0b111);
      } else {
        TestFrm(0x0021'9173, guest_rounding, guest_rounding & 0b111);
      }
      EXPECT_EQ(std::fegetround(), host_rounding);
    }
  }
}

TEST_F(TESTSUITE, OpInstructions) {
  // Add
  TestOp(0x0031'00b3, {{19, 23, 42}});
  // Sub
  TestOp(0x4031'00b3, {{42, 23, 19}});
  // And
  TestOp(0x0031'70b3, {{0b0101, 0b0011, 0b0001}});
  // Or
  TestOp(0x0031'60b3, {{0b0101, 0b0011, 0b0111}});
  // Xor
  TestOp(0x0031'40b3, {{0b0101, 0b0011, 0b0110}});
  // Sll
  TestOp(0x0031'10b3, {{0b1010, 3, 0b0101'0000}});
  // Srl
  TestOp(0x0031'50b3, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Sra
  TestOp(0x4031'50b3, {{0xf000'0000'0000'0000ULL, 12, 0xffff'0000'0000'0000ULL}});
  // Slt
  TestOp(0x0031'20b3,
         {
             {19, 23, 1},
             {23, 19, 0},
             {~0ULL, 0, 1},
         });
  // Sltu
  TestOp(0x0031'30b3,
         {
             {19, 23, 1},
             {23, 19, 0},
             {~0ULL, 0, 0},
         });
  // Mul
  TestOp(0x0231'00b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x0a3d'70a3'd70a'3d71}});
  // Mulh
  TestOp(0x231'10b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x28f5'c28f'5c28'f5c3}});
  // Mulhsu
  TestOp(0x231'20b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0xc28f'5c28'f5c2'8f5c}});
  // Mulhu
  TestOp(0x231'30b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x5c28'f5c2'8f5c'28f5}});
  // Div
  TestOp(0x231'40b3, {{0x9999'9999'9999'9999, 0x3333, 0xfffd'fffd'fffd'fffe}});
  TestOp(0x231'40b3, {{42, 2, 21}});
  TestOp(0x231'40b3, {{42, 0, -1}});
  TestOp(0x231'40b3, {{-2147483648, -1, 2147483648}});
  TestOp(0x231'40b3, {{0x8000'0000'0000'0000, -1, 0x8000'0000'0000'0000}});

  // Divu
  TestOp(0x231'50b3, {{0x9999'9999'9999'9999, 0x3333, 0x0003'0003'0003'0003}});
  TestOp(0x231'50b3, {{42, 2, 21}});
  TestOp(0x231'50b3, {{42, 0, 0xffff'ffff'ffff'ffffULL}});
  // Rem
  TestOp(0x231'60b3, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'ffff'ffff}});
  TestOp(0x231'60b3, {{0x9999'9999'9999'9999, 0, 0x9999'9999'9999'9999}});
  // Remu
  TestOp(0x231'70b3, {{0x9999'9999'9999'9999, 0x3333, 0}});
  TestOp(0x231'70b3, {{0x9999'9999'9999'9999, 0, 0x9999'9999'9999'9999}});
  // Andn
  TestOp(0x4031'70b3, {{0b0101, 0b0011, 0b0100}});
  // Orn
  TestOp(0x4031'60b3, {{0b0101, 0b0011, 0xffff'ffff'ffff'fffd}});
  // Xnor
  TestOp(0x4031'40b3, {{0b0101, 0b0011, 0xffff'ffff'ffff'fff9}});
  // Max
  TestOp(0x0a31'60b3, {{bit_cast<uint64_t>(int64_t{-5}), 4, 4}});
  TestOp(0x0a31'60b3,
         {{bit_cast<uint64_t>(int64_t{-5}),
           bit_cast<uint64_t>(int64_t{-10}),
           bit_cast<uint64_t>(int64_t{-5})}});
  // Maxu
  TestOp(0x0a31'70b3, {{50, 1, 50}});
  // Min
  TestOp(0x0a31'40b3, {{bit_cast<uint64_t>(int64_t{-5}), 4, bit_cast<uint64_t>(int64_t{-5})}});
  TestOp(0x0a31'40b3,
         {{bit_cast<uint64_t>(int64_t{-5}),
           bit_cast<uint64_t>(int64_t{-10}),
           bit_cast<uint64_t>(int64_t{-10})}});
  // Minu
  TestOp(0x0a31'50b3, {{50, 1, 1}});

  // Ror
  TestOp(0x6031'50b3, {{0xf000'0000'0000'000fULL, 4, 0xff00'0000'0000'0000ULL}});
  TestOp(0x6031'50b3, {{0xf000'0000'0000'000fULL, 8, 0x0ff0'0000'0000'0000ULL}});
  // Rol
  TestOp(0x6031'10b3, {{0xff00'0000'0000'0000ULL, 4, 0xf000'0000'0000'000fULL}});
  TestOp(0x6031'10b3, {{0x000f'ff00'0000'000fULL, 8, 0x0fff'0000'0000'0f00ULL}});
  // Sh1add
  TestOp(0x2031'20b3,
         {{0x0008'0000'0000'0001, 0x1001'0001'0000'0000ULL, 0x1011'0001'0000'0002ULL}});
  // Sh2add
  TestOp(0x2031'40b3,
         {{0x0008'0000'0000'0001, 0x0001'0001'0000'0000ULL, 0x0021'0001'0000'0004ULL}});
  // Sh3add
  TestOp(0x2031'60b3,
         {{0x0008'0000'0000'0001, 0x1001'0011'0000'0000ULL, 0x1041'0011'0000'0008ULL}});
  // Bclr
  TestOp(0x4831'10b3, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0000ULL}});
  TestOp(0x4831'10b3, {{0b1000'0001'0000'0001ULL, 8, 0b1000'0000'0000'0001ULL}});
  // Bext
  TestOp(0x4831'50b3, {{0b1000'0001'0000'0001ULL, 0, 0b0000'0000'0000'0001ULL}});
  TestOp(0x4831'50b3, {{0b1000'0001'0000'0001ULL, 8, 0b0000'0000'0000'0001ULL}});
  TestOp(0x4831'50b3, {{0b1000'0001'0000'0001ULL, 7, 0b0000'0000'0000'0000ULL}});
  // Binv
  TestOp(0x6831'10b3, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0000ULL}});
  TestOp(0x6831'10b3, {{0b1000'0001'0000'0001ULL, 1, 0b1000'0001'0000'0011ULL}});
  // Bset
  TestOp(0x2831'10b3, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0001ULL}});
  TestOp(0x2831'10b3, {{0b1000'0001'0000'0001ULL, 1, 0b1000'0001'0000'0011ULL}});
}

TEST_F(TESTSUITE, Op32Instructions) {
  // Addw
  TestOp(0x0031'00bb, {{19, 23, 42}, {0x8000'0000, 0, 0xffff'ffff'8000'0000}});
  // Add.uw
  TestOp(0x0831'00bb, {{19, 23, 42}, {0x8000'0000'8000'0000, 1, 0x0000'0000'8000'0001}});
  // Subw
  TestOp(0x4031'00bb, {{42, 23, 19}, {0x8000'0000, 0, 0xffff'ffff'8000'0000}});
  // Sllw
  TestOp(0x0031'10bb, {{0b1010, 3, 0b1010'000}});
  // Srlw
  TestOp(0x0031'50bb, {{0x0000'0000'f000'0000ULL, 12, 0x0000'0000'000f'0000ULL}});
  // Sraw
  TestOp(0x4031'50bb, {{0x0000'0000'f000'0000ULL, 12, 0xffff'ffff'ffff'0000ULL}});
  // Mulw
  TestOp(0x0231'00bb, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0xffff'ffff'd70a'3d71}});
  // Divw
  TestOp(0x231'40bb, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'fffd'fffe}});
  TestOp(0x231'40bb, {{0x9999'9999'9999'9999, 0, -1}});
  TestOp(0x231'40bb, {{-2147483648, -1, -2147483648}});

  // Divuw
  TestOp(0x231'50bb,
         {{0x9999'9999'9999'9999, 0x3333, 0x0000'0000'0003'0003},
          {0xffff'ffff'8000'0000, 1, 0xffff'ffff'8000'0000}});
  // Remw
  TestOp(0x231'60bb, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'ffff'ffff}});
  // Remuw
  TestOp(0x231'70bb,
         {{0x9999'9999'9999'9999, 0x3333, 0},
          {0xffff'ffff'8000'0000, 0xffff'ffff'8000'0001, 0xffff'ffff'8000'0000}});
  // Zext.h
  TestOp(0x0801'40bb, {{0xffff'ffff'ffff'fffeULL, 0, 0xfffe}});
  // Zext.b
  TestOp(0x0ff1'7093, {{0xffff'ffff'ffff'fffeULL, 0, 0xfe}});
  // Zext.w
  TestOp(0x0801'00bb, {{0xffff'ffff'ffff'fffeULL, 0, 0xffff'fffe}});
  // Rorw
  TestOp(0x6031'50bb, {{0x0000'0000'f000'000fULL, 4, 0xffff'ffff'ff00'0000}});
  TestOp(0x6031'50bb, {{0x0000'0000'f000'0000ULL, 4, 0x0000'0000'0f00'0000}});
  TestOp(0x6031'50bb, {{0x0000'0000'0f00'000fULL, 4, 0xffff'ffff'f0f0'0000}});
  // Rolw
  TestOp(0x6031'10bb, {{0x0000'0000'f000'000fULL, 4, 0x0000'0000'0000'00ff}});
  TestOp(0x6031'10bb, {{0x0000'0000'0ff0'0000ULL, 4, 0xffff'ffff'ff00'0000}});
  // Sh1add.uw
  TestOp(0x2031'20bb, {{0xf0ff'0000'8000'0001, 0x8000'0000, 0x0000'0001'8000'0002}});
  // Sh2add.uw
  TestOp(0x2031'40bb, {{0xf0ff'00ff'8000'0001, 0x8000'0000, 0x0000'0002'8000'0004}});
  // Sh3add.uw
  TestOp(0x2031'60bb, {{0xf0ff'0f00'8000'0001, 0x8000'0000, 0x0000'0004'8000'0008}});
}

TEST_F(TESTSUITE, OpImmInstructions) {
  // Addi
  TestOpImm(0x0001'0093, {{19, 23, 42}});
  // Slti
  TestOpImm(0x0001'2093,
            {
                {19, 23, 1},
                {23, 19, 0},
                {~0ULL, 0, 1},
            });
  // Sltiu
  TestOpImm(0x0001'3093,
            {
                {19, 23, 1},
                {23, 19, 0},
                {~0ULL, 0, 0},
            });
  // Xori
  TestOpImm(0x0001'4093, {{0b0101, 0b0011, 0b0110}});
  // Ori
  TestOpImm(0x0001'6093, {{0b0101, 0b0011, 0b0111}});
  // Andi
  TestOpImm(0x0001'7093, {{0b0101, 0b0011, 0b0001}});
  // Slli
  TestOpImm(0x0001'1093, {{0b1010, 3, 0b1010'000}});
  // Srli
  TestOpImm(0x0001'5093, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Srai
  TestOpImm(0x4001'5093, {{0xf000'0000'0000'0000ULL, 12, 0xffff'0000'0000'0000ULL}});
  // Rori
  TestOpImm(0x6001'5093, {{0xf000'0000'0000'000fULL, 4, 0xff00'0000'0000'0000ULL}});
  // Clz
  TestOpImm(0x6001'1093, {{0, 0, 64}});
  TestOpImm(0x6001'1093, {{123, 0, 57}});
  // Ctz
  TestOpImm(0x6011'1093, {{0, 0, 64}});
  TestOpImm(0x6011'1093, {{0x0100'0000'0000, 0, 40}});
  // Cpop
  TestOpImm(0x6021'1093, {{0xf000'0000'0000'000fULL, 0, 8}});
  // Rev8
  TestOpImm(0x6b81'5093, {{0x0000'0000'0000'000fULL, 0, 0x0f00'0000'0000'0000ULL}});
  TestOpImm(0x6b81'5093, {{0xf000'0000'0000'0000ULL, 0, 0x0000'0000'0000'00f0ULL}});
  TestOpImm(0x6b81'5093, {{0x00f0'0000'0000'0000ULL, 0, 0x0000'0000'0000'f000ULL}});
  TestOpImm(0x6b81'5093, {{0x0000'000f'0000'0000ULL, 0, 0x0000'0000'0f00'0000ULL}});

  // Sext.b
  TestOpImm(0x6041'1093, {{0b1111'1110, 0, 0xffff'ffff'ffff'fffe}});  // -2
  // Sext.h
  TestOpImm(0x6051'1093, {{0b1111'1110, 0, 0xfe}});
  TestOpImm(0x6051'1093, {{0b1111'1111'1111'1110, 0, 0xffff'ffff'ffff'fffe}});
  // Orc.b
  TestOpImm(0x2871'5093, {{0xfe00'f0ff'fa00'fffb, 0, 0xff00'ffff'ff00'ffff}});
  TestOpImm(0x2871'5093, {{0xfa00, 0, 0xff00}});
  // Bclri
  TestOpImm(0x4801'1093, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0000ULL}});
  TestOpImm(0x4801'1093, {{0b1000'0001'0000'0001ULL, 8, 0b1000'0000'0000'0001ULL}});
  // Bexti
  TestOpImm(0x4801'5093, {{0b1000'0001'0000'0001ULL, 0, 0b0000'0000'0000'0001ULL}});
  TestOpImm(0x4801'5093, {{0b1000'0001'0000'0001ULL, 8, 0b0000'0000'0000'0001ULL}});
  TestOpImm(0x4801'5093, {{0b1000'0001'0000'0001ULL, 7, 0b0000'0000'0000'0000ULL}});
  // Binvi
  TestOpImm(0x6801'1093, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0000ULL}});
  TestOpImm(0x6801'1093, {{0b1000'0001'0000'0001ULL, 1, 0b1000'0001'0000'0011ULL}});
  // Bseti
  TestOpImm(0x2801'1093, {{0b1000'0001'0000'0001ULL, 0, 0b1000'0001'0000'0001ULL}});
  TestOpImm(0x2801'1093, {{0b1000'0001'0000'0001ULL, 1, 0b1000'0001'0000'0011ULL}});
}

TEST_F(TESTSUITE, OpImm32Instructions) {
  // Addiw
  TestOpImm(0x0001'009b, {{19, 23, 42}, {0x8000'0000, 0, 0xffff'ffff'8000'0000}});
  // Slliw
  TestOpImm(0x0001'109b, {{0b1010, 3, 0b1010'000}});
  // Srliw
  TestOpImm(0x0001'509b, {{0x0000'0000'f000'0000ULL, 12, 0x0000'0000'000f'0000ULL}});
  // Sraiw
  TestOpImm(0x4001'509b, {{0x0000'0000'f000'0000ULL, 12, 0xffff'ffff'ffff'0000ULL}});
  // Roriw
  TestOpImm(0x6001'509b, {{0x0000'0000'f000'000fULL, 4, 0xffff'ffff'ff00'0000}});
  TestOpImm(0x6001'509b, {{0x0000'0000'f000'0000ULL, 4, 0x0000'0000'0f00'0000}});
  TestOpImm(0x6001'509b, {{0x0000'0000'0f00'000fULL, 4, 0xffff'ffff'f0f0'0000}});
  // Clzw
  TestOpImm(0x6001'109b, {{0, 0, 32}});
  TestOpImm(0x6001'109b, {{123, 0, 25}});
  // Ctzw
  TestOpImm(0x6011'109b, {{0, 0, 32}});
  TestOpImm(0x6011'109b, {{0x0000'0000'0000'0010, 0, 4}});
  // Cpopw
  TestOpImm(0x6021'109b, {{0xf000'0000'0000'000f, 0, 4}});
  // Slli.uw
  TestOpImm(0x0801'109b, {{0x0000'0000'f000'000fULL, 4, 0x0000'000f'0000'00f0}});
}

TEST_F(TESTSUITE, OpFpInstructions) {
  // FAdd.S
  TestOpFp(0x0031'00d3, {std::tuple{1.0f, 2.0f, 3.0f}});
  // FAdd.D
  TestOpFp(0x0231'00d3, {std::tuple{1.0, 2.0, 3.0}});
  // FSub.S
  TestOpFp(0x0831'00d3, {std::tuple{3.0f, 2.0f, 1.0f}});
  // FSub.D
  TestOpFp(0x0a31'00d3, {std::tuple{3.0, 2.0, 1.0}});
  // FMul.S
  TestOpFp(0x1031'00d3, {std::tuple{3.0f, 2.0f, 6.0f}});
  // FMul.D
  TestOpFp(0x1231'00d3, {std::tuple{3.0, 2.0, 6.0}});
  // FDiv.S
  TestOpFp(0x1831'00d3, {std::tuple{6.0f, 2.0f, 3.0f}});
  // FDiv.D
  TestOpFp(0x1a31'00d3, {std::tuple{6.0, 2.0, 3.0}});
  // FSgnj.S
  TestOpFp(0x2031'00d3,
           {std::tuple{1.0f, 2.0f, 1.0f},
            {-1.0f, 2.0f, 1.0f},
            {1.0f, -2.0f, -1.0f},
            {-1.0f, -2.0f, -1.0f}});
  // FSgnj.D
  TestOpFp(0x2231'00d3,
           {
               std::tuple{1.0, 2.0, 1.0},
               {-1.0, 2.0, 1.0},
               {1.0, -2.0, -1.0},
               {-1.0, -2.0, -1.0},
           });
  // FSgnjn.S
  TestOpFp(0x2031'10d3,
           {
               std::tuple{1.0f, 2.0f, -1.0f},
               {1.0f, 2.0f, -1.0f},
               {1.0f, -2.0f, 1.0f},
               {-1.0f, -2.0f, 1.0f},
           });
  // FSgnjn.D
  TestOpFp(0x2231'10d3,
           {
               std::tuple{1.0, 2.0, -1.0},
               {1.0, 2.0, -1.0},
               {1.0, -2.0, 1.0},
               {-1.0, -2.0, 1.0},
           });
  // FSgnjx.S
  TestOpFp(0x2031'20d3,
           {
               std::tuple{1.0f, 2.0f, 1.0f},
               {-1.0f, 2.0f, -1.0f},
               {1.0f, -2.0f, -1.0f},
               {-1.0f, -2.0f, 1.0f},
           });
  // FSgnjx.D
  TestOpFp(0x2231'20d3,
           {
               std::tuple{1.0, 2.0, 1.0},
               {-1.0, 2.0, -1.0},
               {1.0, -2.0, -1.0},
               {-1.0, -2.0, 1.0},
           });
  // FMin.S
  TestOpFp(0x2831'00d3,
           {std::tuple{+0.f, +0.f, +0.f},
            {+0.f, -0.f, -0.f},
            {-0.f, +0.f, -0.f},
            {-0.f, -0.f, -0.f},
            {+0.f, 1.f, +0.f},
            {-0.f, 1.f, -0.f}});
  // FMin.D
  TestOpFp(0x2a31'00d3,
           {std::tuple{+0.0, +0.0, +0.0},
            {+0.0, -0.0, -0.0},
            {-0.0, +0.0, -0.0},
            {-0.0, -0.0, -0.0},
            {+0.0, 1.0, +0.0},
            {-0.0, 1.0, -0.0}});
  // FMax.S
  TestOpFp(0x2831'10d3,
           {std::tuple{+0.f, +0.f, +0.f},
            {+0.f, -0.f, +0.f},
            {-0.f, +0.f, +0.f},
            {-0.f, -0.f, -0.f},
            {+0.f, 1.f, 1.f},
            {-0.f, 1.f, 1.f}});
  // FMax.D
  TestOpFp(0x2a31'10d3,
           {std::tuple{+0.0, +0.0, +0.0},
            {+0.0, -0.0, +0.0},
            {-0.0, +0.0, +0.0},
            {-0.0, -0.0, -0.0},
            {+0.0, 1.0, 1.0},
            {-0.0, 1.0, 1.0}});
}

TEST_F(TESTSUITE, UpperImmInstructions) {
  // Auipc
  TestAuipc(0xfedc'b097, 0xffff'ffff'fedc'b000);
  // Lui
  TestLui(0xfedc'b0b7, 0xffff'ffff'fedc'b000);
}

TEST_F(TESTSUITE, TestBranchInstructions) {
  // Beq
  TestBranch(0x0020'8463,
             {
                 {42, 42, 8},
                 {41, 42, 4},
                 {42, 41, 4},
             });
  // Bne
  TestBranch(0x0020'9463,
             {
                 {42, 42, 4},
                 {41, 42, 8},
                 {42, 41, 8},
             });
  // Bltu
  TestBranch(0x0020'e463,
             {
                 {41, 42, 8},
                 {42, 42, 4},
                 {42, 41, 4},
                 {0xf000'0000'0000'0000ULL, 42, 4},
                 {42, 0xf000'0000'0000'0000ULL, 8},
             });
  // Bgeu
  TestBranch(0x0020'f463,
             {
                 {42, 41, 8},
                 {42, 42, 8},
                 {41, 42, 4},
                 {0xf000'0000'0000'0000ULL, 42, 8},
                 {42, 0xf000'0000'0000'0000ULL, 4},
             });
  // Blt
  TestBranch(0x0020'c463,
             {
                 {41, 42, 8},
                 {42, 42, 4},
                 {42, 41, 4},
                 {0xf000'0000'0000'0000ULL, 42, 8},
                 {42, 0xf000'0000'0000'0000ULL, 4},
             });
  // Bge
  TestBranch(0x0020'd463,
             {
                 {42, 41, 8},
                 {42, 42, 8},
                 {41, 42, 4},
                 {0xf000'0000'0000'0000ULL, 42, 4},
                 {42, 0xf000'0000'0000'0000ULL, 8},
             });
  // Beq with negative offset.
  TestBranch(0xfe20'8ee3,
             {
                 {42, 42, -4},
             });
}

TEST_F(TESTSUITE, JumpAndLinkInstructions) {
  // Jal
  TestJumpAndLink(0x0080'00ef, 8);
  // Jal with negative offset.
  TestJumpAndLink(0xffdf'f0ef, -4);
}

TEST_F(TESTSUITE, JumpAndLinkWithReturnAddressRegisterAsTarget) {
  uint32_t insn_bytes{// jalr   ra
                      0x0000'80e7};
  auto code_start = ToGuestAddr(&insn_bytes);
  state_.cpu.insn_addr = code_start;
  // Translation cache requires upper bits to be zero.
  constexpr GuestAddr kJumpTargetAddr = 0x0000'f00d'cafe'b0baULL;
  SetXReg<RA>(state_.cpu, kJumpTargetAddr);

  EXPECT_TRUE(RunOneInstruction(&state_, kJumpTargetAddr));
  EXPECT_EQ(state_.cpu.insn_addr, kJumpTargetAddr);
  EXPECT_EQ(GetXReg<RA>(state_.cpu), code_start + 4);
}

TEST_F(TESTSUITE, JumpAndLinkRegisterInstructions) {
  // Jalr offset=4.
  TestJumpAndLinkRegister<4>(0x0041'00e7, 38, 42);
  // Jalr offset=-4.
  TestJumpAndLinkRegister<4>(0xffc1'00e7, 42, 38);
  // Jalr offset=5 - must properly align the target to even.
  TestJumpAndLinkRegister<4>(0x0051'00e7, 38, 42);
  // Jr offset=4.
  TestJumpAndLinkRegister<0>(0x0041'0067, 38, 42);
  // Jr offset=-4.
  TestJumpAndLinkRegister<0>(0xffc1'0067, 42, 38);
  // Jr offset=5 - must properly align the target to even.
  TestJumpAndLinkRegister<0>(0x0051'0067, 38, 42);
}

TEST_F(TESTSUITE, LoadInstructions) {
  // Offset is always 8.
  // Lbu
  TestLoad(0x0081'4083, kDataToLoad & 0xffULL);
  // Lhu
  TestLoad(0x0081'5083, kDataToLoad & 0xffffULL);
  // Lwu
  TestLoad(0x0081'6083, kDataToLoad & 0xffff'ffffULL);
  // Ldu
  TestLoad(0x0081'3083, kDataToLoad);
  // Lb
  TestLoad(0x0081'0083, int64_t{int8_t(kDataToLoad)});
  // Lh
  TestLoad(0x0081'1083, int64_t{int16_t(kDataToLoad)});
  // Lw
  TestLoad(0x0081'2083, int64_t{int32_t(kDataToLoad)});
}

TEST_F(TESTSUITE, StoreInstructions) {
  // Offset is always 8.
  // Sb
  TestStore(0x0020'8423, kDataToStore & 0xffULL);
  // Sh
  TestStore(0x0020'9423, kDataToStore & 0xffffULL);
  // Sw
  TestStore(0x0020'a423, kDataToStore & 0xffff'ffffULL);
  // Sd
  TestStore(0x0020'b423, kDataToStore);
}

TEST_F(TESTSUITE, FmaInstructions) {
  // Fmadd.S
  TestFma(0x2031'70c3, {std::tuple{1.0f, 2.0f, 3.0f, 5.0f}});
  // Fmadd.D
  TestFma(0x2231'70c3, {std::tuple{1.0, 2.0, 3.0, 5.0}});
  // Fmsub.S
  TestFma(0x2031'70c7, {std::tuple{1.0f, 2.0f, 3.0f, -1.0f}});
  // Fmsub.D
  TestFma(0x2231'70c7, {std::tuple{1.0, 2.0, 3.0, -1.0}});
  // Fnmsub.S
  TestFma(0x2031'70cb, {std::tuple{1.0f, 2.0f, 3.0f, 1.0f}});
  // Fnmsub.D
  TestFma(0x2231'70cb, {std::tuple{1.0, 2.0, 3.0, 1.0}});
  // Fnmadd.S
  TestFma(0x2031'70cf, {std::tuple{1.0f, 2.0f, 3.0f, -5.0f}});
  // Fnmadd.D
  TestFma(0x2231'70cf, {std::tuple{1.0, 2.0, 3.0, -5.0}});
}

#if (defined(TESTING_INTERPRETER) || defined(TESTING_HEAVY_OPTIMIZER))

TEST_F(TESTSUITE, AtomicLoadInstructions) {
  // Validate sign-extension of returned value.
  const uint64_t kNegative32BitValue = 0x0000'0000'8000'0000ULL;
  const uint64_t kSignExtendedNegative = 0xffff'ffff'8000'0000ULL;
  const uint64_t kPositive32BitValue = 0xffff'ffff'0000'0000ULL;
  const uint64_t kSignExtendedPositive = 0ULL;
  static_assert(static_cast<int32_t>(kSignExtendedPositive) >= 0);
  static_assert(static_cast<int32_t>(kSignExtendedNegative) < 0);

  // Lrw - sign extends from 32 to 64.
  TestAtomicLoad(0x1000'a12f, &kPositive32BitValue, kSignExtendedPositive);
  TestAtomicLoad(0x1000'a12f, &kNegative32BitValue, kSignExtendedNegative);

  // Lrd
  TestAtomicLoad(0x1000'b12f, &kDataToLoad, kDataToLoad);
}

TEST_F(TESTSUITE, AtomicStoreInstructions) {
  // Scw
  TestAtomicStore(0x1820'a1af, static_cast<uint32_t>(kDataToStore));

  // Scd
  TestAtomicStore(0x1820'b1af, kDataToStore);
}

TEST_F(TESTSUITE, AtomicStoreInstructionNoLoadFailure) {
  // Scw
  TestAtomicStoreNoLoadFailure(0x1820'a1af);

  // Scd
  TestAtomicStoreNoLoadFailure(0x1820'b1af);
}

TEST_F(TESTSUITE, AtomicStoreInstructionDifferentLoadFailure) {
  // Scw
  TestAtomicStoreDifferentLoadFailure(0x1820'a1af);

  // Scd
  TestAtomicStoreDifferentLoadFailure(0x1820'b1af);
}

#endif  // (defined(TESTING_INTERPRETER) || defined(TESTING_HEAVY_OPTIMIZER))

TEST_F(TESTSUITE, AmoInstructions) {
  // Verifying that all aq and rl combinations work for Amoswap, but only test relaxed one for most
  // other instructions for brevity.

  // AmoswaoW/AmoswaoD
  TestAmo(0x0831'20af, 0x0831'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWAq/AmoswapDAq
  TestAmo(0x0c31'20af, 0x0c31'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWRl/AmoswapDRl
  TestAmo(0x0a31'20af, 0x0a31'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWAqrl/AmoswapDAqrl
  TestAmo(0x0e31'20af, 0x0e31'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoaddW/AmoaddD
  TestAmo(0x0031'20af, 0x0031'30af, 0xaaaa'aaaa'aaaa'aaa9);

  // AmoxorW/AmoxorD
  TestAmo(0x2031'20af, 0x2031'30af, 0x5555'5555'1111'1111);

  // AmoandW/AmoandD
  TestAmo(0x6031'20af, 0x6031'30af, 0xaaaa'aaaa'cccc'cccc);

  // AmoorW/AmoorD
  TestAmo(0x4031'20af, 0x4031'30af, 0xffff'ffff'dddd'dddd);

  // AmominW/AmominD
  TestAmo(0x8031'20af, 0x8031'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmomaxW/AmomaxD
  TestAmo(0xa031'20af, 0xa031'30af, 0xffff'eeee'dddd'ccccULL);

  // AmominuW/AmominuD
  TestAmo(0xc031'20af, 0xc031'30af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmomaxuW/AmomaxuD
  TestAmo(0xe031'20af, 0xe031'30af, 0xffff'eeee'dddd'ccccULL);
}

TEST_F(TESTSUITE, OpFpSingleInputInstructions) {
  // FSqrt.S
  TestOpFpSingleInput(0x5801'70d3, {std::tuple{4.0f, 2.0f}});
  // FSqrt.D
  TestOpFpSingleInput(0x5a01'70d3, {std::tuple{16.0, 4.0}});
}

TEST_F(TESTSUITE, Fmv) {
  // Fmv.X.W
  TestFmvFloatToInteger(0xe000'80d3,
                        {std::tuple{1.0f, static_cast<uint64_t>(bit_cast<uint32_t>(1.0f))},
                         {-1.0f, static_cast<int64_t>(bit_cast<int32_t>(-1.0f))}});
  // Fmv.W.X
  TestFmvIntegerToFloat(
      0xf000'80d3,
      {std::tuple{bit_cast<uint32_t>(1.0f), 1.0f}, {bit_cast<uint32_t>(-1.0f), -1.0f}});
  // Fmv.X.D
  TestFmvFloatToInteger(
      0xe200'80d3, {std::tuple{1.0, bit_cast<uint64_t>(1.0)}, {-1.0, bit_cast<uint64_t>(-1.0)}});
  // Fmv.D.X
  TestFmvIntegerToFloat(
      0xf200'80d3, {std::tuple{bit_cast<uint64_t>(1.0), 1.0}, {bit_cast<uint64_t>(-1.0), -1.0}});
  // Fmv.S
  TestOpFpSingleInput(0x2021'00d3, {std::tuple{1.0f, 1.0f}, {-1.0f, -1.0f}});
  // Fmv.D
  TestOpFpSingleInput(0x2221'00d3,
                      {std::tuple{bit_cast<uint64_t>(1.0), 1.0}, {bit_cast<uint64_t>(-1.0), -1.0}});
}

const uint32_t kPosNanFloat = kFPValueToFPReg(std::numeric_limits<float>::quiet_NaN());
const uint32_t kNegNanFloat = kFPValueToFPReg(-std::numeric_limits<float>::quiet_NaN());
const uint64_t kPosNanDouble = kFPValueToFPReg(std::numeric_limits<double>::quiet_NaN());
const uint64_t kNegNanDouble = kFPValueToFPReg(-std::numeric_limits<double>::quiet_NaN());
constexpr uint64_t kMaskFloatBits = (uint64_t{1} << 32) - 1;

TEST_F(TESTSUITE, FabsSinglePrecisionNanPosToPos) {
  SetFReg<2>(state_.cpu, kPosNanFloat);
  RunInstruction(0x2021'20d3);  // fabs.s f1, f2
  EXPECT_EQ(GetFReg<1>(state_.cpu) & kMaskFloatBits, kPosNanFloat);
}

TEST_F(TESTSUITE, FabsSinglePrecisionNanNegToPos) {
  SetFReg<2>(state_.cpu, kNegNanFloat);
  RunInstruction(0x2021'20d3);  // fabs.s f1, f2
  EXPECT_EQ(GetFReg<1>(state_.cpu) & kMaskFloatBits, kPosNanFloat);
}

TEST_F(TESTSUITE, FabsDoublePrecisionNanPosToPos) {
  SetFReg<2>(state_.cpu, kPosNanDouble);
  RunInstruction(0x2221'20d3);  // fabs.d f1, f2
  EXPECT_EQ(GetFReg<1>(state_.cpu), kPosNanDouble);
}

TEST_F(TESTSUITE, FabsDoublePrecisionNanNegToPos) {
  SetFReg<2>(state_.cpu, kNegNanDouble);
  RunInstruction(0x2221'20d3);  // fabs.d f1, f2
  EXPECT_EQ(GetFReg<1>(state_.cpu), kPosNanDouble);
}

TEST_F(TESTSUITE, FnegSinglePrecisionNanPosToNeg) {
  SetFReg<2>(state_.cpu, kPosNanFloat);
  RunInstruction(0x2021'10d3);  // fneg.s f1, f2
  EXPECT_EQ(GetFReg<1>(state_.cpu) & kMaskFloatBits, kNegNanFloat);
}

TEST_F(TESTSUITE, FnegSinglePrecisionNanNegToPos) {
  SetFReg<2>(state_.cpu, kNegNanFloat);
  RunInstruction(0x2021'10d3);  // fneg.s f1, f2
  EXPECT_EQ(GetFReg<1>(state_.cpu) & kMaskFloatBits, kPosNanFloat);
}

TEST_F(TESTSUITE, FnegDoublePrecisionNanPosToNeg) {
  SetFReg<2>(state_.cpu, kPosNanDouble);
  RunInstruction(0x2221'10d3);  // fneg.s f1, f2
  EXPECT_EQ(GetFReg<1>(state_.cpu), kNegNanDouble);
}

TEST_F(TESTSUITE, FnegDoublePrecisionNanNegToPos) {
  SetFReg<2>(state_.cpu, kNegNanDouble);
  RunInstruction(0x2221'10d3);  // fneg.s f1, f2
  EXPECT_EQ(GetFReg<1>(state_.cpu), kPosNanDouble);
}

TEST_F(TESTSUITE, OpFpFcvt) {
  // Fcvt.S.D
  TestOpFpSingleInput(0x4011'70d3, {std::tuple{1.0, 1.0f}});
  // Fcvt.D.S
  TestOpFpSingleInput(0x4201'00d3, {std::tuple{2.0f, 2.0}});
  // Fcvt.W.S
  TestOpFpGpRegisterTargetSingleInput(0xc001'70d3, {std::tuple{3.0f, 3UL}});
  // Fcvt.WU.S
  TestOpFpGpRegisterTargetSingleInput(0xc011'70d3, {std::tuple{3.0f, 3UL}});
  // Fcvt.L.S
  TestOpFpGpRegisterTargetSingleInput(0xc021'70d3, {std::tuple{3.0f, 3UL}});
  // Fcvt.LU.S
  TestOpFpGpRegisterTargetSingleInput(0xc031'70d3, {std::tuple{3.0f, 3UL}});
  // Fcvt.W.D
  TestOpFpGpRegisterTargetSingleInput(0xc201'70d3, {std::tuple{3.0, 3UL}});
  // Fcvt.WU.D
  TestOpFpGpRegisterTargetSingleInput(0xc211'70d3, {std::tuple{3.0, 3UL}});
  // Fcvt.L.D
  TestOpFpGpRegisterTargetSingleInput(0xc221'70d3, {std::tuple{3.0, 3UL}});
  // Fcvt.LU.D
  TestOpFpGpRegisterTargetSingleInput(0xc231'70d3, {std::tuple{3.0, 3UL}});
  // Fcvt.S.W
  TestOpFpGpRegisterSourceSingleInput(0xd001'70d3, {std::tuple{3UL, 3.0f}});
  // Fcvt.S.WU
  TestOpFpGpRegisterSourceSingleInput(0xd011'70d3, {std::tuple{3UL, 3.0f}});
  // Fcvt.S.L
  TestOpFpGpRegisterSourceSingleInput(0xd021'70d3, {std::tuple{3UL, 3.0f}});
  // Fcvt.S.LU
  TestOpFpGpRegisterSourceSingleInput(0xd031'70d3, {std::tuple{3UL, 3.0f}});
  // Fcvt.D.W
  TestOpFpGpRegisterSourceSingleInput(0xd201'70d3, {std::tuple{3UL, 3.0}});
  // Fcvt.D.Wu
  TestOpFpGpRegisterSourceSingleInput(0xd211'70d3, {std::tuple{3UL, 3.0}});
  // Fcvt.D.L
  TestOpFpGpRegisterSourceSingleInput(0xd221'70d3, {std::tuple{3UL, 3.0}});
  // Fcvt.D.LU
  TestOpFpGpRegisterSourceSingleInput(0xd231'70d3, {std::tuple{3UL, 3.0}});
}

TEST_F(TESTSUITE, OpFpGpRegisterTargetInstructions) {
  // Fle.S
  TestOpFpGpRegisterTarget(0xa031'00d3,
                           {std::tuple{1.0f, 2.0f, 1UL}, {2.0f, 1.0f, 0UL}, {0.0f, 0.0f, 1UL}});
  // Fle.D
  TestOpFpGpRegisterTarget(0xa231'00d3,
                           {std::tuple{1.0, 2.0, 1UL}, {2.0, 1.0, 0UL}, {0.0, 0.0, 1UL}});
  // Flt.S
  TestOpFpGpRegisterTarget(0xa031'10d3,
                           {std::tuple{1.0f, 2.0f, 1UL}, {2.0f, 1.0f, 0UL}, {0.0f, 0.0f, 0UL}});
  // Flt.D
  TestOpFpGpRegisterTarget(0xa231'10d3,
                           {std::tuple{1.0, 2.0, 1UL}, {2.0, 1.0, 0UL}, {0.0, 0.0, 0UL}});
  // Feq.S
  TestOpFpGpRegisterTarget(0xa031'20d3,
                           {std::tuple{1.0f, 2.0f, 0UL}, {2.0f, 1.0f, 0UL}, {0.0f, 0.0f, 1UL}});
  // Feq.D
  TestOpFpGpRegisterTarget(0xa231'20d3,
                           {std::tuple{1.0, 2.0, 0UL}, {2.0, 1.0, 0UL}, {0.0, 0.0, 1UL}});
}

TEST_F(TESTSUITE, TestOpFpGpRegisterTargetSingleInput) {
  // Fclass.S
  TestOpFpGpRegisterTargetSingleInput(
      0xe001'10d3,
      {std::tuple{-std::numeric_limits<float>::infinity(), 0b00'0000'0001UL},
       {-1.0f, 0b00'0000'0010UL},
       {-std::numeric_limits<float>::denorm_min(), 0b00'0000'0100UL},
       {-0.0f, 0b00'0000'1000UL},
       {0.0f, 0b00'0001'0000UL},
       {std::numeric_limits<float>::denorm_min(), 0b00'0010'0000UL},
       {1.0f, 0b00'0100'0000UL},
       {std::numeric_limits<float>::infinity(), 0b00'1000'0000UL},
       {std::numeric_limits<float>::signaling_NaN(), 0b01'0000'0000UL},
       {std::numeric_limits<float>::quiet_NaN(), 0b10'0000'0000UL}});
  // Fclass.D
  TestOpFpGpRegisterTargetSingleInput(
      0xe201'10d3,
      {std::tuple{-std::numeric_limits<double>::infinity(), 0b00'0000'0001UL},
       {-1.0, 0b00'0000'0010UL},
       {-std::numeric_limits<double>::denorm_min(), 0b00'0000'0100UL},
       {-0.0, 0b00'0000'1000UL},
       {0.0, 0b00'0001'0000UL},
       {std::numeric_limits<double>::denorm_min(), 0b00'0010'0000UL},
       {1.0, 0b00'0100'0000UL},
       {std::numeric_limits<double>::infinity(), 0b00'1000'0000UL},
       {std::numeric_limits<double>::signaling_NaN(), 0b01'0000'0000UL},
       {std::numeric_limits<double>::quiet_NaN(), 0b10'0000'0000UL}});
}

TEST_F(TESTSUITE, RoundingModeTest) {
  // FAdd.S
  TestOpFp(0x0031'00d3,
           // Test RNE
           {std::tuple{1.0000001f, 0.000000059604645f, 1.0000002f},
            {1.0000002f, 0.000000059604645f, 1.0000002f},
            {1.0000004f, 0.000000059604645f, 1.0000005f},
            {-1.0000001f, -0.000000059604645f, -1.0000002f},
            {-1.0000002f, -0.000000059604645f, -1.0000002f},
            {-1.0000004f, -0.000000059604645f, -1.0000005f}});
  // FAdd.S
  TestOpFp(0x0031'10d3,
           // Test RTZ
           {std::tuple{1.0000001f, 0.000000059604645f, 1.0000001f},
            {1.0000002f, 0.000000059604645f, 1.0000002f},
            {1.0000004f, 0.000000059604645f, 1.0000004f},
            {-1.0000001f, -0.000000059604645f, -1.0000001f},
            {-1.0000002f, -0.000000059604645f, -1.0000002f},
            {-1.0000004f, -0.000000059604645f, -1.0000004f}});
  // FAdd.S
  TestOpFp(0x0031'20d3,
           // Test RDN
           {std::tuple{1.0000001f, 0.000000059604645f, 1.0000001f},
            {1.0000002f, 0.000000059604645f, 1.0000002f},
            {1.0000004f, 0.000000059604645f, 1.0000004f},
            {-1.0000001f, -0.000000059604645f, -1.0000002f},
            {-1.0000002f, -0.000000059604645f, -1.0000004f},
            {-1.0000004f, -0.000000059604645f, -1.0000005f}});
  // FAdd.S
  TestOpFp(0x0031'30d3,
           // Test RUP
           {std::tuple{1.0000001f, 0.000000059604645f, 1.0000002f},
            {1.0000002f, 0.000000059604645f, 1.0000004f},
            {1.0000004f, 0.000000059604645f, 1.0000005f},
            {-1.0000001f, -0.000000059604645f, -1.0000001f},
            {-1.0000002f, -0.000000059604645f, -1.0000002f},
            {-1.0000004f, -0.000000059604645f, -1.0000004f}});
  // FAdd.S
  TestOpFp(0x0031'40d3,
           // Test RMM
           {std::tuple{1.0000001f, 0.000000059604645f, 1.0000002f},
            {1.0000002f, 0.000000059604645f, 1.0000004f},
            {1.0000004f, 0.000000059604645f, 1.0000005f},
            {-1.0000001f, -0.000000059604645f, -1.0000002f},
            {-1.0000002f, -0.000000059604645f, -1.0000004f},
            {-1.0000004f, -0.000000059604645f, -1.0000005f}});

  // FAdd.D
  TestOpFp(0x0231'00d3,
           // Test RNE
           {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000004},
            {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000004},
            {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000009},
            {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000004},
            {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000004},
            {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000009}});
  // FAdd.D
  TestOpFp(0x0231'10d3,
           // Test RTZ
           {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000002},
            {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000004},
            {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000007},
            {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000002},
            {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000004},
            {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000007}});
  // FAdd.D
  TestOpFp(0x0231'20d3,
           // Test RDN
           {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000002},
            {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000004},
            {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000007},
            {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000004},
            {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000007},
            {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000009}});
  // FAdd.D
  TestOpFp(0x0231'30d3,
           // Test RUP
           {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000004},
            {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000007},
            {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000009},
            {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000002},
            {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000004},
            {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000007}});
  // FAdd.D
  TestOpFp(0x0231'40d3,
           // Test RMM
           {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000004},
            {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000007},
            {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000009},
            {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000004},
            {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000007},
            {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000009}});
}

TEST_F(TESTSUITE, LoadFpInstructions) {
  // Offset is always 8.
  TestLoadFp(0x0081'2087, kDataToLoad | 0xffff'ffff'0000'0000ULL);
  TestLoadFp(0x0081'3087, kDataToLoad);
}

TEST_F(TESTSUITE, StoreFpInstructions) {
  // Offset is always 8.
  // Fsw
  TestStoreFp(0x0020'a427, kDataToStore & 0xffff'ffffULL);
  // Fsd
  TestStoreFp(0x0020'b427, kDataToStore);
}

#if defined(TESTING_INTERPRETER) || defined(TESTING_LITE_TRANSLATOR)

TEST_F(TESTSUITE, TestVsetvl) {
  constexpr uint64_t kVill =
      0b1'0000000'00000000'00000000'00000000'00000000'00000000'00000000'00000000;
  // Vsetvl, rs1 != x0
  TestVsetvl(0x8031'70d7,
             {
                 // Valid combinations.
                 {~0ULL, ~0ULL, ~0ULL, 005, 2, 005},
                 {~0ULL, ~0ULL, ~0ULL, 006, 4, 006},
                 {~0ULL, ~0ULL, ~0ULL, 007, 8, 007},
                 {~0ULL, ~0ULL, ~0ULL, 000, 16, 000},
                 {~0ULL, ~0ULL, ~0ULL, 001, 32, 001},
                 {~0ULL, ~0ULL, ~0ULL, 002, 64, 002},
                 {~0ULL, ~0ULL, ~0ULL, 003, 128, 003},
                 {~0ULL, ~0ULL, ~0ULL, 015, 1, 015},
                 {~0ULL, ~0ULL, ~0ULL, 016, 2, 016},
                 {~0ULL, ~0ULL, ~0ULL, 017, 4, 017},
                 {~0ULL, ~0ULL, ~0ULL, 010, 8, 010},
                 {~0ULL, ~0ULL, ~0ULL, 011, 16, 011},
                 {~0ULL, ~0ULL, ~0ULL, 012, 32, 012},
                 {~0ULL, ~0ULL, ~0ULL, 013, 64, 013},
                 {~0ULL, ~0ULL, ~0ULL, 026, 1, 026},
                 {~0ULL, ~0ULL, ~0ULL, 027, 2, 027},
                 {~0ULL, ~0ULL, ~0ULL, 020, 4, 020},
                 {~0ULL, ~0ULL, ~0ULL, 021, 8, 021},
                 {~0ULL, ~0ULL, ~0ULL, 022, 16, 022},
                 {~0ULL, ~0ULL, ~0ULL, 023, 32, 023},
                 {~0ULL, ~0ULL, ~0ULL, 037, 1, 037},
                 {~0ULL, ~0ULL, ~0ULL, 030, 2, 030},
                 {~0ULL, ~0ULL, ~0ULL, 031, 4, 031},
                 {~0ULL, ~0ULL, ~0ULL, 032, 8, 032},
                 {~0ULL, ~0ULL, ~0ULL, 033, 16, 033},
                 // Invalid combinations.
                 {~0ULL, ~0ULL, ~0ULL, 004, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 014, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 024, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 025, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 034, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 035, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 036, 0, kVill},
                 // Invalid sizes.
                 {~0ULL, ~0ULL, ~0ULL, 040, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 041, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 042, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 043, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 044, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 045, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 046, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 047, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 050, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 051, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 052, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 053, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 054, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 055, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 056, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 057, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 060, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 061, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 062, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 063, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 064, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 065, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 066, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 067, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 070, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 071, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 072, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 073, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 074, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 075, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 076, 0, kVill},
                 {~0ULL, ~0ULL, ~0ULL, 077, 0, kVill},
                 // Vma/vta bits.
                 {~0ULL, ~0ULL, ~0ULL, 0100, 16, 0100},
                 {~0ULL, ~0ULL, ~0ULL, 0200, 16, 0200},
                 {~0ULL, ~0ULL, ~0ULL, 0300, 16, 0300},
                 // Extra bits ignored as permitted by RISC-V specification.
                 {~0ULL, ~0ULL, ~0ULL, 0400, 16, 0000},
                 {~0ULL, ~0ULL, ~0ULL, 0500, 16, 0100},
                 {~0ULL, ~0ULL, ~0ULL, 0600, 16, 0200},
                 {~0ULL, ~0ULL, ~0ULL, 0700, 16, 0300},
                 // Avl handling.
                 {~0ULL, ~0ULL, 67, 003, 67, 003},
                 {~0ULL, ~0ULL, 151, 003, 76, 003},
                 {~0ULL, ~0ULL, 256, 003, 128, 003},
                 {~0ULL, ~0ULL, 257, 003, 128, 003},
             });
  // vsetvl rs1 == x0, rd != x0
  TestVsetvl(0x8030'70d7, {{~0ULL, ~0ULL, 42, 000, 16, 000}});
  // vsetvl rs1 == x0, rd == x0
  TestVsetvl(0x8030'7057,
             {// Valid change of vtype.
              {9, 000, 128, 022, 9, 022},
              // Invalid change of vtype.
              {8, 001, 128, 022, 0, kVill}});
  // vsetvli rs1 != x0
  TestVsetvl(0x121'70d7, {{~0ULL, ~0ULL, 128, 0, 16, 022}});
  // vsetvli rs1 == x0, rd != x0
  TestVsetvl(0x120'70d7, {{~0ULL, ~0ULL, 42, 000, 16, 022}});
  // vsetvli, rs1 == x0, rd == x0
  TestVsetvl(0x120'7057,
             {// Valid change of vtype.
              {9, 000, 128, ~0ULL, 9, 022},
              // Invalid change of vtype.
              {8, 001, 128, ~0ULL, 0, kVill}});
  // vsetivli rs1 != x0
  TestVsetvl(0xc128'70d7, {{~0ULL, ~0ULL, 128, 0, 16, 022}});
  // vsetivli rs1 == x0, rd != x0
  TestVsetvl(0xc120'70d7, {{~0ULL, ~0ULL, 42, 000, 16, 022}});
  // vsetivli, rs1 == x0, rd == x0
  TestVsetvl(0xc120'7057,
             {// Valid change of vtype.
              {9, 000, 128, ~0ULL, 9, 022},
              // Invalid change of vtype.
              {8, 001, 128, ~0ULL, 0, kVill}});
}

#endif  // defined(TESTING_INTERPRETER) || defined(TESTING_LITE_TRANSLATOR)
