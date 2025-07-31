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

#include <malloc.h>  // memalign
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>  // copy_n, fill_n
#include <cfenv>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <memory>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/intrinsics/guest_cpu_flags.h"       // GuestModeFromHostRounding
#include "berberis/intrinsics/guest_rounding_modes.h"  // ScopedRoundingMode
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/vector_intrinsics.h"
#include "berberis/runtime_primitives/memory_region_reservation.h"

#include "../faulty_memory_accesses.h"

namespace berberis {

namespace {

#if defined(__i386__)
constexpr size_t kRegIP = REG_EIP;
#elif defined(__x86_64__)
constexpr size_t kRegIP = REG_RIP;
#else
#error "Unsupported arch"
#endif

bool sighandler_called = false;

void FaultHandler(int /* sig */, siginfo_t* /* info */, void* ctx) {
  ucontext_t* ucontext = reinterpret_cast<ucontext_t*>(ctx);
  static_assert(sizeof(void*) == sizeof(greg_t), "Unsupported type sizes");
  void* fault_addr = reinterpret_cast<void*>(ucontext->uc_mcontext.gregs[kRegIP]);
  void* recovery_addr = FindFaultyMemoryAccessRecoveryAddrForTesting(fault_addr);
  sighandler_called = true;
  CHECK(recovery_addr);
  ucontext->uc_mcontext.gregs[kRegIP] = reinterpret_cast<greg_t>(recovery_addr);
}

class ScopedFaultySigaction {
 public:
  ScopedFaultySigaction() {
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = FaultHandler;
    CHECK_EQ(sigaction(SIGSEGV, &sa, &old_sa_), 0);
  }

  ~ScopedFaultySigaction() { CHECK_EQ(sigaction(SIGSEGV, &old_sa_, nullptr), 0); }

 private:
  struct sigaction old_sa_;
};

//  Interpreter decodes the size itself, but we need to accept this template parameter to share
//  tests with translators.
template <uint8_t kInsnSize = 4>
bool RunOneInstruction(ThreadState* state, GuestAddr stop_pc) {
  InterpretInsn(state);
  return state->cpu.insn_addr == stop_pc;
}

class Riscv64InterpreterTest : public ::testing::Test {
 public:
  // Non-Compressed Instructions.
  Riscv64InterpreterTest()
      : state_{
            .cpu = {.vtype = uint64_t{1} << 63, .frm = intrinsics::GuestModeFromHostRounding()}} {}

  void InterpretFence(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    InterpretInsn(&state_);
  }

  // Vector instructions.
  template <typename ElementType>
  void TestFPExceptions(uint32_t insn_bytes) {
    // Install the following arguments: NaN, 1.0, 1.0, NaN.
    // Then try to mask out NaNs with vstart, vl, and mask.
    state_.cpu.v[24] = state_.cpu.v[16] = state_.cpu.v[8] =
        SIMD128Register{__v2du{0x7ff4'0000'0000'0000, 0x3ff0'0000'0000'0000}}.Get<__uint128_t>();
    state_.cpu.v[25] = state_.cpu.v[17] = state_.cpu.v[9] =
        SIMD128Register{__v2du{0x3ff0'0000'0000'0000, 0x7ff4'0000'0000'0000}}.Get<__uint128_t>();
    state_.cpu.v[26] = state_.cpu.v[18] = state_.cpu.v[10] = SIMD128Register{
        __v4su{
            0x7fa0'0000, 0x3f80'0000, 0x3f80'0000, 0x7fa0'0000}}.Get<__uint128_t>();
    state_.cpu.f[1] = 0xffff'ffff'3f80'0000;
    state_.cpu.f[2] = 0x3ff0'0000'0000'0000;
    ;
    state_.cpu.vtype = (BitUtilLog2(sizeof(ElementType)) << 3) | /*vlmul=*/1;
    // First clear exceptions and execute the instruction with vstart = 0, vl = 4.
    // This should produce FE_INVALID with most floating point instructions.
    EXPECT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    state_.cpu.vstart = 0;
    state_.cpu.vl = 4;
    state_.cpu.v[0] = 0b1111;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_TRUE(bool(fetestexcept(FE_ALL_EXCEPT)));
    // Mask NaNs using vstart and vl.
    EXPECT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    state_.cpu.vstart = 1;
    state_.cpu.vl = 3;
    state_.cpu.v[0] = 0b1111;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_FALSE(bool(fetestexcept(FE_ALL_EXCEPT)));
    // Mask NaNs using mask.
    EXPECT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    state_.cpu.vstart = 0;
    state_.cpu.vl = 4;
    state_.cpu.v[0] = 0b0110;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_FALSE(bool(fetestexcept(FE_ALL_EXCEPT)));
  }

  template <size_t kNFfields>
  void TestVlXreXX(uint32_t insn_bytes) {
    const auto kUndisturbedValue = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&kVectorComparisonSource));
    for (size_t index = 0; index < 8; index++) {
      state_.cpu.v[8 + index] = kUndisturbedValue;
    }
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    for (size_t index = 0; index < 8; index++) {
      EXPECT_EQ(state_.cpu.v[8 + index],
                (index >= kNFfields
                     ? kUndisturbedValue
                     : SIMD128Register{kVectorComparisonSource[index]}.Get<__uint128_t>()));
    }
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul>
  void VlxsegXeiXX(
      uint32_t insn_bytes,
      const ElementType::BaseType (&expected_results)[kNFfields * kLmul][static_cast<int>(
          sizeof(SIMD128Register) / sizeof(ElementType))]) {
    VlxsegXeiXX<ElementType, UInt8, kNFfields, kLmul>(insn_bytes, expected_results);
    VlxsegXeiXX<ElementType, UInt8, kNFfields, kLmul>(insn_bytes | 0x0800'0000, expected_results);
    VlxsegXeiXX<ElementType, UInt16, kNFfields, kLmul>(insn_bytes | 0x5000, expected_results);
    VlxsegXeiXX<ElementType, UInt16, kNFfields, kLmul>(insn_bytes | 0x0800'5000, expected_results);
    VlxsegXeiXX<ElementType, UInt32, kNFfields, kLmul>(insn_bytes | 0x6000, expected_results);
    VlxsegXeiXX<ElementType, UInt32, kNFfields, kLmul>(insn_bytes | 0x0800'6000, expected_results);
    VlxsegXeiXX<ElementType, UInt64, kNFfields, kLmul>(insn_bytes | 0x7000, expected_results);
    VlxsegXeiXX<ElementType, UInt64, kNFfields, kLmul>(insn_bytes | 0x0800'7000, expected_results);
  }

  template <typename DataElementType, typename IndexElementType, size_t kNFfields, size_t kLmul>
  void VlxsegXeiXX(
      uint32_t insn_bytes,
      const DataElementType::BaseType (&expected_results)[kNFfields * kLmul][static_cast<int>(
          sizeof(SIMD128Register) / sizeof(DataElementType))]) {
    constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(IndexElementType);
    constexpr size_t kTotalElements =
        sizeof(SIMD128Register) / sizeof(DataElementType) * kLmul * kNFfields;
    // If we need more indexes than may fit into 8 vector registers then such operation is
    // impossible on RISC-V and we should skip that combination.
    if constexpr (sizeof(IndexElementType) * kTotalElements <= sizeof(SIMD128Register) * 8) {
      TestVectorLoad<DataElementType, kNFfields, kLmul>(insn_bytes, expected_results, [this] {
        for (size_t reg_no = 0; reg_no < AlignUp<kElementsCount>(kTotalElements) / kElementsCount;
             ++reg_no) {
          SIMD128Register index_register;
          for (size_t index = 0; index < kElementsCount; ++index) {
            index_register.Set(IndexElementType{static_cast<typename IndexElementType::BaseType>(
                                   kPermutedIndexes[index + reg_no * kElementsCount] *
                                   sizeof(DataElementType) * kNFfields)},
                               index);
          }
          state_.cpu.v[16 + reg_no] = index_register.Get<__uint128_t>();
        }
      });
    }
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul>
  void TestVlsegXeXX(
      uint32_t insn_bytes,
      const ElementType::BaseType (
          &expected_results)[kNFfields * kLmul][static_cast<int>(16 / sizeof(ElementType))]) {
    TestVectorLoad<ElementType, kNFfields, kLmul>(insn_bytes, expected_results, [] {});
    // Turn Vector Unit-Stride Segment Loads and Stores into Fault-Only-First Load.
    TestVectorLoad<ElementType, kNFfields, kLmul>(
        insn_bytes | 0x0100'0000, expected_results, [] {});
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul>
  void TestVlssegXeXX(
      uint32_t insn_bytes,
      uint64_t stride,
      const ElementType::BaseType (
          &expected_results)[kNFfields * kLmul][static_cast<int>(16 / sizeof(ElementType))]) {
    TestVectorLoad<ElementType, kNFfields, kLmul>(
        insn_bytes, expected_results, [stride, this] { SetXReg<2>(state_.cpu, stride); });
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul, typename ExtraCPUInitType>
  void TestVectorLoad(
      uint32_t insn_bytes,
      const ElementType::BaseType (
          &expected_results)[kNFfields * kLmul][static_cast<int>(16 / sizeof(ElementType))],
      ExtraCPUInitType ExtraCPUInit) {
    constexpr int kElementsCount = static_cast<int>(16 / sizeof(ElementType));
    const auto kUndisturbedValue = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    for (uint8_t vstart = 0; vstart <= kElementsCount * kLmul; ++vstart) {
      for (uint8_t vl = 0; vl <= kElementsCount * kLmul; ++vl) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          // Handle three masking cases:
          //   no masking (vma == 0), agnostic (vma == 1), undisturbed (vma == 2)
          for (uint8_t vma = 0; vma < 3; ++vma) {
            state_.cpu.vtype = (vma & 1) << 7 | (vta << 6) |
                               (BitUtilLog2(sizeof(ElementType)) << 3) | BitUtilLog2(kLmul);
            state_.cpu.vstart = vstart;
            state_.cpu.vl = vl;
            uint32_t insn_bytes_with_vm = insn_bytes;
            // If masking is supposed to be disabled then we need to set vm bit (#25).
            if (!vma) {
              insn_bytes_with_vm |= (1 << 25);
            }
            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes_with_vm);
            SetXReg<1>(state_.cpu, ToGuestAddr(&kVectorCalculationsSource));
            state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
            for (size_t index = 0; index < 8; index++) {
              state_.cpu.v[8 + index] = kUndisturbedValue;
            }
            ExtraCPUInit();
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
            EXPECT_EQ(state_.cpu.vstart, 0);
            EXPECT_EQ(state_.cpu.vl, vl);
            for (size_t field = 0; field < kNFfields; field++) {
              for (size_t index = 0; index < kLmul; index++) {
                for (size_t element = 0; element < kElementsCount; ++element) {
                  ElementType expected_element;
                  if (vstart >= vl) {
                    // When vstart ⩾ vl, there are no body elements, and no elements are updated in
                    // any destinationvector register group, including that no tail elements are
                    // updated with agnostic values.
                    expected_element =
                        SIMD128Register{kUndisturbedResult}.Get<ElementType>(element);
                  } else if (element + index * kElementsCount < std::min(vstart, vl)) {
                    // Elements before vstart are undisturbed if vstart is less than vl.
                    expected_element =
                        SIMD128Register{kUndisturbedResult}.Get<ElementType>(element);
                  } else if (element + index * kElementsCount >= vl) {
                    // Element after vl have to be processed with vta policy.
                    if (vta == 1) {
                      expected_element = ~ElementType{0};
                    } else {
                      expected_element =
                          SIMD128Register{kUndisturbedResult}.Get<ElementType>(element);
                    }
                  } else if (vma &&
                             (~(state_.cpu.v[0]) >> (element + index * kElementsCount)) & 1) {
                    // If element is inactive it's processed with vma policy.
                    if (vma == 1) {
                      expected_element = ~ElementType{0};
                    } else {
                      expected_element =
                          SIMD128Register{kUndisturbedResult}.Get<ElementType>(element);
                    }
                  } else {
                    expected_element =
                        ElementType{expected_results[index + field * kLmul][element]};
                  }
                  EXPECT_EQ(
                      SIMD128Register{state_.cpu.v[8 + index + field * kLmul]}.Get<ElementType>(
                          element),
                      expected_element);
                }
              }
            }
            for (size_t index = kNFfields * kLmul; index < 8; index++) {
              EXPECT_EQ(state_.cpu.v[8 + index], kUndisturbedValue);
            }
          }
        }
      }
    }
  }

  void TestVlm(uint32_t insn_bytes, __v16qu expected_results) {
    const auto kUndisturbedValue = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    // Vlm.v is special form of normal vector load which mostly ignores vtype.
    // The only bit that it honors is vill: https://github.com/riscv/riscv-v-spec/pull/877
    // Verify that changes to vtype don't affect the execution (but vstart and vl do).
    for (uint8_t sew = 0; sew < 4; ++sew) {
      for (uint8_t vlmul = 0; vlmul < 4; ++vlmul) {
        const uint8_t kElementsCount = (16 >> sew) << vlmul;
        for (uint8_t vstart = 0; vstart <= kElementsCount; ++vstart) {
          for (uint8_t vl = 0; vl <= kElementsCount; ++vl) {
            const uint8_t kVlmVl = AlignUp<CHAR_BIT>(vl) / CHAR_BIT;
            for (uint8_t vta = 0; vta < 2; ++vta) {
              for (uint8_t vma = 0; vma < 2; ++vma) {
                state_.cpu.vtype = (vma << 7) | (vta << 6) | (sew << 3) | vlmul;
                state_.cpu.vstart = vstart;
                state_.cpu.vl = vl;
                state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
                SetXReg<1>(state_.cpu, ToGuestAddr(&kVectorCalculationsSource));
                state_.cpu.v[8] = kUndisturbedValue;
                EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
                EXPECT_EQ(state_.cpu.vstart, 0);
                EXPECT_EQ(state_.cpu.vl, vl);
                for (size_t element = 0; element < 16; ++element) {
                  UInt8 expected_element;
                  if (element < vstart || vstart >= kVlmVl) {
                    expected_element = SIMD128Register{kUndisturbedResult}.Get<UInt8>(element);
                  } else if (element >= kVlmVl) {
                    expected_element = ~UInt8{0};
                  } else {
                    expected_element = UInt8{expected_results[element]};
                  }
                  EXPECT_EQ(SIMD128Register{state_.cpu.v[8]}.Get<UInt8>(element), expected_element);
                }
              }
            }
          }
        }
      }
    }
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul, size_t kResultSize>
  void VsxsegXeiXX(uint32_t insn_bytes, const uint64_t (&expected_results)[kResultSize]) {
    VsxsegXeiXX<ElementType, UInt8, kNFfields, kLmul>(insn_bytes, expected_results);
    VsxsegXeiXX<ElementType, UInt8, kNFfields, kLmul>(insn_bytes | 0x0800'0000, expected_results);
    VsxsegXeiXX<ElementType, UInt16, kNFfields, kLmul>(insn_bytes | 0x5000, expected_results);
    VsxsegXeiXX<ElementType, UInt16, kNFfields, kLmul>(insn_bytes | 0x0800'5000, expected_results);
    VsxsegXeiXX<ElementType, UInt32, kNFfields, kLmul>(insn_bytes | 0x6000, expected_results);
    VsxsegXeiXX<ElementType, UInt32, kNFfields, kLmul>(insn_bytes | 0x0800'6000, expected_results);
    VsxsegXeiXX<ElementType, UInt64, kNFfields, kLmul>(insn_bytes | 0x7000, expected_results);
    VsxsegXeiXX<ElementType, UInt64, kNFfields, kLmul>(insn_bytes | 0x0800'7000, expected_results);
  }

  template <typename DataElementType,
            typename IndexElementType,
            size_t kNFfields,
            size_t kLmul,
            size_t kResultSize>
  void VsxsegXeiXX(uint32_t insn_bytes, const uint64_t (&expected_results)[kResultSize]) {
    constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(IndexElementType);
    constexpr size_t kTotalElements =
        sizeof(SIMD128Register) / sizeof(DataElementType) * kLmul * kNFfields;
    // If we need more indexes than may fit into 8 vector registers then such operation is
    // impossible on RISC-V and we should skip that combination.
    if constexpr (sizeof(IndexElementType) * kTotalElements <= sizeof(SIMD128Register) * 8) {
      TestVectorStore<DataElementType, kNFfields, kLmul>(insn_bytes, expected_results, [this] {
        for (size_t reg_no = 0; reg_no < AlignUp<kElementsCount>(kTotalElements) / kElementsCount;
             ++reg_no) {
          SIMD128Register index_register;
          for (size_t index = 0; index < kElementsCount; ++index) {
            index_register.Set(IndexElementType{static_cast<typename IndexElementType::BaseType>(
                                   kPermutedIndexes[index + reg_no * kElementsCount] *
                                   sizeof(DataElementType) * kNFfields)},
                               index);
          }
          state_.cpu.v[16 + reg_no] = index_register.Get<__uint128_t>();
        }
      });
    }
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul, size_t kResultSize>
  void TestVssegXeXX(uint32_t insn_bytes, const uint64_t (&expected_results)[kResultSize]) {
    TestVectorStore<ElementType, kNFfields, kLmul>(insn_bytes, expected_results, [] {});
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul, size_t kResultSize>
  void TestVsssegXeXX(uint32_t insn_bytes,
                      uint64_t stride,
                      const uint64_t (&expected_results)[kResultSize]) {
    TestVectorStore<ElementType, kNFfields, kLmul>(
        insn_bytes, expected_results, [stride, this] { SetXReg<2>(state_.cpu, stride); });
  }

  template <typename ElementType,
            size_t kNFfields,
            size_t kLmul,
            size_t kResultSize,
            typename ExtraCPUInitType>
  void TestVectorStore(uint32_t insn_bytes,
                       const uint64_t (&expected_results)[kResultSize],
                       ExtraCPUInitType ExtraCPUInit) {
    constexpr size_t kElementsCount = 16 / sizeof(ElementType);
    const SIMD128Register kUndisturbedValue = SIMD128Register{kUndisturbedResult};
    state_.cpu.vtype = (BitUtilLog2(sizeof(ElementType)) << 3) | BitUtilLog2(kLmul);
    state_.cpu.vstart = 0;
    state_.cpu.vl = kElementsCount * kLmul;
    // First verify that store works with no inactive elements and no masking.
    uint32_t insn_bytes_with_vm = insn_bytes | (1 << 25);
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes_with_vm);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    for (size_t index = 0; index < 8; ++index) {
      state_.cpu.v[8 + index] =
          SIMD128Register{kVectorCalculationsSource[index]}.Get<__uint128_t>();
    }
    for (uint64_t& element : store_area_) {
      element = kUndisturbedResult[0];
    }
    ExtraCPUInit();
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    for (size_t index = 0; index < std::size(store_area_); ++index) {
      if (index >= std::size(expected_results)) {
        EXPECT_EQ(store_area_[index], static_cast<uint64_t>(kUndisturbedResult[0]));
      } else {
        EXPECT_EQ(store_area_[index], expected_results[index]);
      }
    }
    // Most vector instruction apply masks to *target* operands which means we may apply the exact
    // same, well-tested logic to verify how they work. But store instructions apply masks to *data
    // source* operand which makes generation of expected target problematic (especially for
    // complicated store with segments and strides and/or indexes). To sidestep the issue we are
    // first testing that full version with all elements active works (above) and then reuse it to
    // verify that `vstart`, `vl` 𝖺𝗇𝖽 `mask` operands work as expected. This wouldn't work if some
    // elements are overlapping, but these instructions, while technically permitted, already are
    // allowed to produce many possible different results, thus we are not testing them.
    for (uint8_t vstart = 0; vstart <= kElementsCount * kLmul; ++vstart) {
      for (uint8_t vl = 0; vl <= kElementsCount * kLmul; ++vl) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          // Handle three masking cases:
          //   no masking (vma == 0), agnostic (vma == 1), undisturbed (vma == 2)
          // Note: vta and vma settings are ignored by store instructions, we are just verifying
          // this is what actually happens.
          for (uint8_t vma = 0; vma < 3; ++vma) {
            // Use instruction in the mode tested above to generate expected results.
            state_.cpu.vtype = (BitUtilLog2(sizeof(ElementType)) << 3) | BitUtilLog2(kLmul);
            state_.cpu.vstart = 0;
            state_.cpu.vl = kElementsCount * kLmul;
            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes_with_vm);
            decltype(store_area_) expected_store_area;
            SetXReg<1>(state_.cpu, ToGuestAddr(&expected_store_area));
            for (size_t index = 0; index < kLmul; ++index) {
              for (size_t field = 0; field < kNFfields; ++field) {
                SIMD128Register register_value =
                    SIMD128Register{kVectorCalculationsSource[index + field * kLmul]};
                if (vma) {
                  auto ApplyMask = [&register_value, &kUndisturbedValue, index](auto mask) {
                    register_value =
                        (register_value & mask[index]) | (kUndisturbedValue & ~mask[index]);
                  };
                  if constexpr (sizeof(ElementType) == sizeof(Int8)) {
                    ApplyMask(kMaskInt8);
                  } else if constexpr (sizeof(ElementType) == sizeof(Int16)) {
                    ApplyMask(kMaskInt16);
                  } else if constexpr (sizeof(ElementType) == sizeof(Int32)) {
                    ApplyMask(kMaskInt32);
                  } else if constexpr (sizeof(ElementType) == sizeof(Int64)) {
                    ApplyMask(kMaskInt64);
                  } else {
                    static_assert(kDependentTypeFalse<ElementType>);
                  }
                }
                if (vstart > kElementsCount * index) {
                  for (size_t prefix_id = 0;
                       prefix_id < std::min(kElementsCount, vstart - kElementsCount * index);
                       ++prefix_id) {
                    register_value.Set(kUndisturbedValue.Get<ElementType>(0), prefix_id);
                  }
                }
                if (vl <= kElementsCount * index) {
                  register_value = kUndisturbedValue.Get<__uint128_t>();
                } else if (vl < kElementsCount * (index + 1)) {
                  for (size_t suffix_id = vl - kElementsCount * index; suffix_id < kElementsCount;
                       ++suffix_id) {
                    register_value.Set(kUndisturbedValue.Get<ElementType>(0), suffix_id);
                  }
                }
                state_.cpu.v[8 + index + field * kLmul] = register_value.Get<__uint128_t>();
              }
            }
            for (uint64_t& element : expected_store_area) {
              element = kUndisturbedResult[0];
            }
            ExtraCPUInit();
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
            // Now execute instruction with mode that we want to actually test.
            state_.cpu.vtype = (vma & 1) << 7 | (vta << 6) |
                               (BitUtilLog2(sizeof(ElementType)) << 3) | BitUtilLog2(kLmul);
            state_.cpu.vstart = vstart;
            state_.cpu.vl = vl;
            uint32_t insn_bytes_with_vm = insn_bytes;
            // If masking is supposed to be disabled then we need to set vm bit (#25).
            if (!vma) {
              insn_bytes_with_vm |= (1 << 25);
            }
            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes_with_vm);
            SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
            state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
            for (size_t index = 0; index < 8; ++index) {
              state_.cpu.v[8 + index] =
                  SIMD128Register{kVectorCalculationsSource[index]}.Get<__uint128_t>();
            }
            for (uint64_t& element : store_area_) {
              element = kUndisturbedResult[0];
            }
            ExtraCPUInit();
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
            for (size_t index = 0; index < std::size(store_area_); ++index) {
              EXPECT_EQ(store_area_[index], expected_store_area[index]);
            }
          }
        }
      }
    }
  }

  void TestVsm(uint32_t insn_bytes, __v16qu expected_results) {
    const auto kUndisturbedValue = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    // Vlm.v is special form of normal vector load which mostly ignores vtype.
    // The only bit that it honors is vill: https://github.com/riscv/riscv-v-spec/pull/877
    // Verify that changes to vtype don't affect the execution (but vstart and vl do).
    for (uint8_t sew = 0; sew < 4; ++sew) {
      for (uint8_t vlmul = 0; vlmul < 4; ++vlmul) {
        const uint8_t kElementsCount = (16 >> sew) << vlmul;
        for (uint8_t vstart = 0; vstart <= kElementsCount; ++vstart) {
          for (uint8_t vl = 0; vl <= kElementsCount; ++vl) {
            const uint8_t kVlmVl = AlignUp<CHAR_BIT>(vl) / CHAR_BIT;
            for (uint8_t vta = 0; vta < 2; ++vta) {
              for (uint8_t vma = 0; vma < 2; ++vma) {
                state_.cpu.vtype = (vma << 7) | (vta << 6) | (sew << 3) | vlmul;
                state_.cpu.vstart = vstart;
                state_.cpu.vl = vl;
                state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
                SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
                store_area_[0] = kUndisturbedResult[0];
                store_area_[1] = kUndisturbedResult[1];
                state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
                EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
                EXPECT_EQ(state_.cpu.vstart, 0);
                EXPECT_EQ(state_.cpu.vl, vl);
                SIMD128Register memory_result =
                    SIMD128Register{__v2du{store_area_[0], store_area_[1]}};
                for (size_t element = 0; element < 16; ++element) {
                  UInt8 expected_element;
                  if (element < vstart || element >= kVlmVl) {
                    expected_element = SIMD128Register{kUndisturbedResult}.Get<UInt8>(element);
                  } else {
                    expected_element = UInt8{expected_results[element]};
                  }
                  EXPECT_EQ(memory_result.Get<UInt8>(element), expected_element);
                }
              }
            }
          }
        }
      }
    }
  }

  // Vector instructions.
  void TestVleXXff(uint32_t insn_bytes,
                   uint8_t loadable_bytes,
                   uint8_t vsew,
                   uint8_t expected_vl,
                   bool fail_on_first = false) {
    ScopedFaultySigaction scoped_sa;
    sighandler_called = false;
    char* buffer;
    const size_t kPageSize = sysconf(_SC_PAGE_SIZE);
    buffer = (char*)memalign(kPageSize, 2 * kPageSize);
    mprotect(buffer, kPageSize * 2, PROT_WRITE);
    char* p = buffer + kPageSize - loadable_bytes;
    std::memcpy(p, &kVectorCalculationsSource[0], 128);
    mprotect(buffer + kPageSize, kPageSize, PROT_NONE);

    auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, (vsew << 3) | 3);
    state_.cpu.vstart = 0;
    state_.cpu.vl = vlmax;
    state_.cpu.vtype = vtype;
    insn_bytes = insn_bytes | (1 << 25);
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(p));

    if (fail_on_first) {
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr));
    } else {
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    }
    if (loadable_bytes < 128) {
      EXPECT_TRUE(sighandler_called);
    } else {
      EXPECT_FALSE(sighandler_called);
    }
    EXPECT_EQ(state_.cpu.vl, expected_vl);
  }

  // Vector instructions.
  void TestVleXX(uint32_t insn_bytes,
                 const __v16qu (&expected_result_int8)[8],
                 const __v8hu (&expected_result_int16)[8],
                 const __v4su (&expected_result_int32)[8],
                 const __v2du (&expected_result_int64)[8],
                 uint8_t veew,
                 const __v2du (&source)[16]) {
    auto Verify = [this, &source](uint32_t insn_bytes,
                                  uint8_t vsew,
                                  uint8_t veew,
                                  uint8_t vlmul_max,
                                  const auto& expected_result,
                                  auto mask) {
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        int vemul = SignExtend<3>(vlmul);
        vemul += vsew;  // Multiply by SEW.
        vemul -= veew;  // Divide by EEW.
        if (vemul < -3 || vemul > 3) {
          // Incompatible vlmul
          continue;
        }

        for (uint8_t vta = 0; vta < 2; ++vta) {
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
            if (vlmax == 0) {
              continue;
            }
            // Test vstart/vl changes with only with vemul == 2 (4 registers)
            if (vemul == 2) {
              state_.cpu.vstart = vlmax / 8;
              state_.cpu.vl = (vlmax * 5) / 8;
            } else {
              state_.cpu.vstart = 0;
              state_.cpu.vl = vlmax;
            }
            state_.cpu.vtype = vtype;

            // Set expected_result vector registers into 0b01010101… pattern.
            for (size_t index = 0; index < 8; ++index) {
              state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
            }
            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
            SetXReg<1>(state_.cpu, ToGuestAddr(&source));
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
            // Values for inactive elements (i.e. corresponding mask bit is 0).
            const size_t n = std::size(source);
            __m128i expected_inactive[n];
            std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));
            if (vemul >= 0) {
              for (size_t index = 0; index < 1 << vemul; ++index) {
                if (index == 0 && vemul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            SIMD128Register{
                                (kUndisturbedResult & kFractionMaskInt8[3]) |
                                (expected_result[index] & mask[index] & ~kFractionMaskInt8[3]) |
                                (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3])}
                                .Get<__uint128_t>());
                } else if (index == 2 && vemul == 2) {
                  EXPECT_EQ(
                      state_.cpu.v[8 + index],
                      SIMD128Register{
                          (expected_result[index] & mask[index] & kFractionMaskInt8[3]) |
                          (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                          ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])}
                          .Get<__uint128_t>());
                } else if (index == 3 && vemul == 2 && vta) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
                } else if (index == 3 && vemul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kUndisturbedResult});
                } else {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            SIMD128Register{(expected_result[index] & mask[index]) |
                                            (expected_inactive[index] & ~mask[index])}
                                .Get<__uint128_t>());
                }
              }

            } else {
              EXPECT_EQ(state_.cpu.v[8],
                        SIMD128Register{
                            (expected_result[0] & mask[0] & kFractionMaskInt8[(vemul + 4)]) |
                            (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[(vemul + 4)]) |
                            ((vta ? kAgnosticResult : kUndisturbedResult) &
                             ~kFractionMaskInt8[(vemul + 4)])}
                            .Get<__uint128_t>());
            }
          }
        }
      }
    };
    switch (veew) {
      case 0:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int8, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int8, kMaskInt8);
        break;
      case 1:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int16, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int16, kMaskInt16);
        break;
      case 2:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int32, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int32, kMaskInt32);
        break;
      case 3:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int64, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int64, kMaskInt64);
        break;
      default:
        break;
    }
  }

  // Vector instructions.
  void TestVseXX(uint32_t insn_bytes,
                 const __v16qu (&expected_result_int8)[8],
                 const __v8hu (&expected_result_int16)[8],
                 const __v4su (&expected_result_int32)[8],
                 const __v2du (&expected_result_int64)[8],
                 uint8_t veew,
                 const __v2du (&source)[16]) {
    auto Verify = [this, &source](uint32_t insn_bytes,
                                  uint8_t vsew,
                                  uint8_t veew,
                                  uint8_t vlmul_max,
                                  const auto& expected_result,
                                  auto mask) {
      SIMD128Register expected_result_in_register;
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        int vemul = SignExtend<3>(vlmul);
        vemul += vsew;  // Multiply by SEW.
        vemul -= veew;  // Divide by EEW.
        if (vemul < -3 || vemul > 3) {
          // Incompatible vlmul
          continue;
        }

        for (uint8_t vma = 0; vma < 2; ++vma) {
          auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, (vma << 7) | (vsew << 3) | vlmul);
          if (vlmax == 0) {
            continue;
          }
          // Test vstart/vl changes with only with vemul == 2 (4 registers)
          if (vemul == 2) {
            state_.cpu.vstart = vlmax / 8;
            state_.cpu.vl = (vlmax * 5) / 8;
          } else {
            state_.cpu.vstart = 0;
            state_.cpu.vl = vlmax;
          }
          state_.cpu.vtype = vtype;

          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
          for (size_t index = 0; index < 8; index++) {
            state_.cpu.v[8 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
            store_area_[index * 2] = kUndisturbedResult[0];
            store_area_[index * 2 + 1] = kUndisturbedResult[1];
          }
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
          // Values for inactive elements (i.e. corresponding mask bit is 0).
          const size_t n = std::size(source);
          __m128i expected_inactive[n];
          std::fill_n(expected_inactive, n, kUndisturbedResult);
          if (vemul >= 0) {
            for (size_t index = 0; index < 1 << vemul; ++index) {
              if (index == 0 && vemul == 2) {
                expected_result_in_register = SIMD128Register{
                    (kUndisturbedResult & kFractionMaskInt8[3]) |
                    (expected_result[index] & mask[index] & ~kFractionMaskInt8[3]) |
                    (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3])};
              } else if (index == 2 && vemul == 2) {
                expected_result_in_register = SIMD128Register{
                    (expected_result[index] & mask[index] & kFractionMaskInt8[3]) |
                    (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                    ((kUndisturbedResult) & ~kFractionMaskInt8[3])};
              } else if (index == 3 && vemul == 2) {
                expected_result_in_register = SIMD128Register{kUndisturbedResult};
              } else {
                expected_result_in_register =
                    SIMD128Register{(expected_result[index] & mask[index]) |
                                    (expected_inactive[index] & ~mask[index])};
              }

              EXPECT_EQ(store_area_[index * 2], expected_result_in_register.Get<uint64_t>(0));
              EXPECT_EQ(store_area_[index * 2 + 1], expected_result_in_register.Get<uint64_t>(1));
            }

          } else {
            expected_result_in_register =
                SIMD128Register{(expected_result[0] & mask[0] & kFractionMaskInt8[(vemul + 4)]) |
                                (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[(vemul + 4)]) |
                                ((kUndisturbedResult) & ~kFractionMaskInt8[(vemul + 4)])};
            EXPECT_EQ(store_area_[0], expected_result_in_register.Get<uint64_t>(0));
            EXPECT_EQ(store_area_[1], expected_result_in_register.Get<uint64_t>(1));
          }
        }
      }
    };
    switch (veew) {
      case 0:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int8, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int8, kMaskInt8);
        break;
      case 1:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int16, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int16, kMaskInt16);
        break;
      case 2:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int32, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int32, kMaskInt32);
        break;
      case 3:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int64, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int64, kMaskInt64);
        break;
      default:
        break;
    }
  }

  template <int kNFfields>
  void TestVmvXr(uint32_t insn_bytes) {
    TestVmvXr<Int8, kNFfields>(insn_bytes);
    TestVmvXr<Int16, kNFfields>(insn_bytes);
    TestVmvXr<Int32, kNFfields>(insn_bytes);
    TestVmvXr<Int64, kNFfields>(insn_bytes);
  }

  template <typename ElementType, int kNFfields>
  void TestVmvXr(uint32_t insn_bytes) {
    // Note that VmvXr actually DOES depend on vtype, contrary to what RISC-V V 1.0 manual says:
    //   https://github.com/riscv/riscv-v-spec/pull/872
    state_.cpu.vtype = BitUtilLog2(sizeof(ElementType)) << 3;
    state_.cpu.vl = 0;
    constexpr int kElementsCount = static_cast<int>(sizeof(SIMD128Register) / sizeof(ElementType));
    for (int vstart = 0; vstart <= kElementsCount * kNFfields; ++vstart) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      state_.cpu.vstart = vstart;
      for (size_t index = 0; index < 16; ++index) {
        state_.cpu.v[8 + index] =
            SIMD128Register{kVectorComparisonSource[index]}.Get<__uint128_t>();
      }
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
      for (int index = 0; index < 8; ++index) {
        SIMD128Register expected_state{kVectorComparisonSource[index]};
        SIMD128Register source_value{kVectorComparisonSource[index + 8]};
        if ((vstart < kElementsCount * kNFfields) && index < kNFfields) {
          // The usual property that no elements are written if vstart >= vl does not apply to these
          // instructions. Instead, no elements are written if vstart >= evl.
          for (int element_index = 0; element_index < kElementsCount; ++element_index) {
            if (element_index + index * kElementsCount >= vstart) {
              expected_state.Set(source_value.Get<ElementType>(element_index), element_index);
            }
          }
        }
        EXPECT_EQ(state_.cpu.v[8 + index], expected_state.Get<__uint128_t>());
      }
      EXPECT_EQ(state_.cpu.vstart, 0);
    }
  }

  template <typename ElementType>
  void TestVfmvfs(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.vtype = BitUtilLog2(sizeof(ElementType)) << 3;
    state_.cpu.vstart = 0;
    state_.cpu.vl = 0;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
    SetFReg<1>(state_.cpu, 0x5555'5555'5555'5555);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
  }

  template <typename ElementType>
  void TestVfmvsf(uint32_t insn_bytes, uint64_t boxed_value, ElementType unboxed_value) {
    for (uint8_t vstart = 0; vstart < 2; ++vstart) {
      for (uint8_t vl = 0; vl < 2; ++vl) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          state_.cpu.vtype = (vta << 6) | (BitUtilLog2(sizeof(ElementType)) << 3);
          state_.cpu.vstart = vstart;
          state_.cpu.vl = vl;
          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
          SetFReg<1>(state_.cpu, boxed_value);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
          if (vstart == 0 && vl != 0) {
            SIMD128Register expected_result =
                vta ? ~SIMD128Register{} : SIMD128Register{kVectorCalculationsSource[0]};
            expected_result.Set<ElementType>(unboxed_value, 0);
            EXPECT_EQ(state_.cpu.v[8], expected_result);
          } else {
            EXPECT_EQ(state_.cpu.v[8],
                      SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>());
          }
        }
      }
    }
  }

  template <typename ElementType>
  void TestVmvsx(uint32_t insn_bytes) {
    for (uint8_t vstart = 0; vstart < 2; ++vstart) {
      for (uint8_t vl = 0; vl < 2; ++vl) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          state_.cpu.vtype = (vta << 6) | (BitUtilLog2(sizeof(ElementType)) << 3);
          state_.cpu.vstart = vstart;
          state_.cpu.vl = vl;
          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
          SetXReg<1>(state_.cpu, 0x5555'5555'5555'5555);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
          if (vstart == 0 && vl != 0) {
            SIMD128Register expected_result =
                vta ? ~SIMD128Register{} : SIMD128Register{kVectorCalculationsSource[0]};
            expected_result.Set<ElementType>(MaybeTruncateTo<ElementType>(0x5555'5555'5555'5555),
                                             0);
            EXPECT_EQ(state_.cpu.v[8], expected_result);
          } else {
            EXPECT_EQ(state_.cpu.v[8],
                      SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>());
          }
        }
      }
    }
  }

  template <typename ElementType>
  void TestVmvxs(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.vtype = BitUtilLog2(sizeof(ElementType)) << 3;
    state_.cpu.vstart = 0;
    state_.cpu.vl = 0;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
    SetXReg<1>(state_.cpu, 0x5555'5555'5555'5555);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  template <size_t kNFfields>
  void TestVsX(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    for (size_t index = 0; index < 8; index++) {
      state_.cpu.v[8 + index] = SIMD128Register{kVectorComparisonSource[index]}.Get<__uint128_t>();
      store_area_[index * 2] = kUndisturbedResult[0];
      store_area_[index * 2 + 1] = kUndisturbedResult[1];
    }
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    for (size_t index = 0; index < 8; index++) {
      EXPECT_EQ(
          store_area_[index * 2],
          (index >= kNFfields ? kUndisturbedResult[0]
                              : SIMD128Register{kVectorComparisonSource[index]}.Get<uint64_t>(0)));
      EXPECT_EQ(
          store_area_[index * 2 + 1],
          (index >= kNFfields ? kUndisturbedResult[1]
                              : SIMD128Register{kVectorComparisonSource[index]}.Get<uint64_t>(1)));
    }
  }

  void TestVectorRegisterGather(uint32_t insn_bytes,
                                const __v16qu (&expected_result_int8)[8],
                                const __v8hu (&expected_result_int16)[8],
                                const __v4su (&expected_result_int32)[8],
                                const __v2du (&expected_result_int64)[8],
                                const uint8_t vlmul,
                                const __v2du (&source)[16]) {
    auto Verify = [this, &source, &vlmul](
                      uint32_t insn_bytes, uint8_t vsew, const auto& expected_result, auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < 8; ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }

      for (uint8_t vta = 0; vta < 2; ++vta) {
        for (uint8_t vma = 0; vma < 2; ++vma) {
          auto [vlmax, vtype] =
              intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
          // Incompatible vsew and vlmax. Skip it.
          if (vlmax == 0) {
            continue;
          }

          // Make sure indexes in src2 fall within vlmax.
          uint64_t src2_mask{0};
          const size_t kElementSize = 8 << vsew;
          const uint64_t kIndexMask = (1 << BitUtilLog2(vlmax)) - 1;
          for (uint8_t index = 0; index < 64 / kElementSize; index++) {
            src2_mask |= kIndexMask << (kElementSize * index);
          }
          for (size_t index = 0; index < 8; ++index) {
            __v2du masked_register = {source[8 + index][0] & src2_mask, source[8 + index][1]};
            state_.cpu.v[24 + index] = SIMD128Register{masked_register}.Get<__uint128_t>();
          }
          SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaaULL & kIndexMask);

          // To make tests quick enough we don't test vstart and vl change with small register
          // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of
          // first
          // register and half of last register.
          if (vlmul == 2) {
            state_.cpu.vstart = vlmax / 8;
            state_.cpu.vl = (vlmax * 5) / 8;
          } else {
            state_.cpu.vstart = 0;
            state_.cpu.vl = vlmax;
          }
          state_.cpu.vtype = vtype;

          // Set expected_result vector registers into 0b01010101… pattern.
          for (size_t index = 0; index < 8; ++index) {
            state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
          }

          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

          // Values for inactive elements (i.e. corresponding mask bit is 0).
          const size_t n = std::size(source) * 2;
          __m128i expected_inactive[n];
          // For most instructions, follow basic inactive processing rules based on vma flag.
          std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));

          if (vlmul < 4) {
            for (size_t index = 0; index < 1 << vlmul; ++index) {
              if (index == 0 && vlmul == 2) {
                EXPECT_EQ(state_.cpu.v[8 + index],
                          SIMD128Register{
                              (kUndisturbedResult & kFractionMaskInt8[3]) |
                              (expected_result[index] & mask[index] & ~kFractionMaskInt8[3]) |
                              (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3])}
                              .Get<__uint128_t>());
              } else if (index == 2 && vlmul == 2) {
                EXPECT_EQ(
                    state_.cpu.v[8 + index],
                    SIMD128Register{
                        (expected_result[index] & mask[index] & kFractionMaskInt8[3]) |
                        (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                        ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])}
                        .Get<__uint128_t>());
              } else if (index == 3 && vlmul == 2 && vta) {
                EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
              } else if (index == 3 && vlmul == 2) {
                EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kUndisturbedResult});
              } else {
                EXPECT_EQ(state_.cpu.v[8 + index],
                          SIMD128Register{(expected_result[index] & mask[index]) |
                                          (expected_inactive[index] & ~mask[index])}
                              .Get<__uint128_t>());
              }
            }
          } else {
            EXPECT_EQ(
                state_.cpu.v[8],
                SIMD128Register{
                    (expected_result[0] & mask[0] & kFractionMaskInt8[vlmul - 4]) |
                    (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[vlmul - 4]) |
                    ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[vlmul - 4])}
                    .Get<__uint128_t>());
          }

          if (vlmul == 2) {
            // Every vector instruction must set vstart to 0, but shouldn't touch vl.
            EXPECT_EQ(state_.cpu.vstart, 0);
            EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
          }
        }
      }
    };
    Verify(insn_bytes, 0, expected_result_int8, kMaskInt8);
    Verify(insn_bytes | (1 << 25), 0, expected_result_int8, kNoMask);
    Verify(insn_bytes, 1, expected_result_int16, kMaskInt16);
    Verify(insn_bytes | (1 << 25), 1, expected_result_int16, kNoMask);
    Verify(insn_bytes, 2, expected_result_int32, kMaskInt32);
    Verify(insn_bytes | (1 << 25), 2, expected_result_int32, kNoMask);
    Verify(insn_bytes, 3, expected_result_int64, kMaskInt64);
    Verify(insn_bytes | (1 << 25), 3, expected_result_int64, kNoMask);
  }

  void TestVectorFloatInstruction(uint32_t insn_bytes,
                                  const UInt32x4Tuple (&expected_result_int32)[8],
                                  const UInt64x2Tuple (&expected_result_int64)[8],
                                  const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kDefault>(
        insn_bytes, source, expected_result_int32, expected_result_int64);
  }

  void TestVectorInstruction(uint32_t insn_bytes,
                             const UInt8x16Tuple (&expected_result_int8)[8],
                             const UInt16x8Tuple (&expected_result_int16)[8],
                             const UInt32x4Tuple (&expected_result_int32)[8],
                             const UInt64x2Tuple (&expected_result_int64)[8],
                             const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kDefault>(
        insn_bytes,
        source,
        expected_result_int8,
        expected_result_int16,
        expected_result_int32,
        expected_result_int64);
  }

  void TestVectorMergeFloatInstruction(uint32_t insn_bytes,
                                       const UInt32x4Tuple (&expected_result_int32)[8],
                                       const UInt64x2Tuple (&expected_result_int64)[8],
                                       const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kVMerge>(
        insn_bytes, source, expected_result_int32, expected_result_int64);
  }

  void TestVectorMergeInstruction(uint32_t insn_bytes,
                                  const UInt8x16Tuple (&expected_result_int8)[8],
                                  const UInt16x8Tuple (&expected_result_int16)[8],
                                  const UInt32x4Tuple (&expected_result_int32)[8],
                                  const UInt64x2Tuple (&expected_result_int64)[8],
                                  const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kVMerge>(
        insn_bytes,
        source,
        expected_result_int8,
        expected_result_int16,
        expected_result_int32,
        expected_result_int64);
  }

  void TestWideningVectorFloatInstruction(uint32_t insn_bytes,
                                          const UInt64x2Tuple (&expected_result_int64)[8],
                                          const __v2du (&source)[16],
                                          __m128i dst_result = kUndisturbedResult) {
    TestVectorInstructionInternal<TestVectorInstructionKind::kFloat,
                                  TestVectorInstructionMode::kWidening>(
        insn_bytes, dst_result, source, expected_result_int64);
  }

  void TestWideningVectorFloatInstruction(uint32_t insn_bytes,
                                          const UInt32x4Tuple (&expected_result_int32)[8],
                                          const UInt64x2Tuple (&expected_result_int64)[8],
                                          const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kWidening>(
        insn_bytes, source, expected_result_int32, expected_result_int64);
  }

  enum class TestVectorInstructionKind { kInteger, kFloat };
  enum class TestVectorInstructionMode { kDefault, kWidening, kNarrowing, kVMerge };

  template <TestVectorInstructionKind kTestVectorInstructionKind,
            TestVectorInstructionMode kTestVectorInstructionMode,
            typename... ExpectedResultType>
  void TestVectorInstruction(uint32_t insn_bytes,
                             const __v2du (&source)[16],
                             const ExpectedResultType&... expected_result) {
    TestVectorInstructionInternal<kTestVectorInstructionKind, kTestVectorInstructionMode>(
        insn_bytes, kUndisturbedResult, source, expected_result...);
  }

  template <TestVectorInstructionKind kTestVectorInstructionKind,
            TestVectorInstructionMode kTestVectorInstructionMode,
            typename... ExpectedResultType>
  void TestVectorInstructionInternal(uint32_t insn_bytes,
                                     __m128i dst_result,
                                     const __v2du (&source)[16],
                                     const ExpectedResultType&... expected_result) {
    auto Verify = [this, &source, dst_result](uint32_t insn_bytes,
                                              uint8_t vsew,
                                              uint8_t vlmul_max,
                                              const auto& expected_result,
                                              auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }
      if (kTestVectorInstructionKind == TestVectorInstructionKind::kInteger) {
        // Set x1 for vx instructions.
        SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);
      } else {
        // We only support Float32/Float64 for float instructions, but there are conversion
        // instructions that work with double width floats.
        // These instructions never use float registers though and thus we don't need to store
        // anything into f1 register, if they are used.
        // For Float32/Float64 case we load 5.625 of the appropriate type into f1.
        ASSERT_LE(vsew, 3);
        if (vsew == 2) {
          SetFReg<1>(state_.cpu, 0xffff'ffff'40b4'0000);  // float 5.625
        } else if (vsew == 3) {
          SetFReg<1>(state_.cpu, 0x4016'8000'0000'0000);  // double 5.625
        }
      }
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kNarrowing ||
                      kTestVectorInstructionMode == TestVectorInstructionMode::kWidening) {
          // Incompatible vlmul for narrowing.
          if (vlmul == 3) {
            continue;
          }
        }
        for (uint8_t vta = 0; vta < 2; ++vta) {
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
            // Incompatible vsew and vlmax. Skip it.
            if (vlmax == 0) {
              continue;
            }
            uint8_t emul =
                (vlmul + (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening)) &
                0b111;

            // To make tests quick enough we don't test vstart and vl change with small register
            // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
            // register, last register and half of next-to last register.
            // Don't use vlmul == 3 because that one may not be supported if instruction widens the
            // result.
            if (emul == 2) {
              state_.cpu.vstart = vlmax / 8;
              state_.cpu.vl = (vlmax * 5) / 8;
            } else {
              state_.cpu.vstart = 0;
              state_.cpu.vl = vlmax;
            }
            state_.cpu.vtype = vtype;

            // Set expected_result vector registers into 0b01010101… pattern.
            for (size_t index = 0; index < 8; ++index) {
              state_.cpu.v[8 + index] = SIMD128Register{dst_result}.Get<__uint128_t>();
            }

            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

            // Values for inactive elements (i.e. corresponding mask bit is 0).
            __m128i expected_inactive[8];
            if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kVMerge) {
              // vs2 is the start of the source vector register group.
              // Note: copy_n input/output args are backwards compared to fill_n below.
              std::copy_n(source, 8, expected_inactive);
            } else {
              // For most instructions, follow basic inactive processing rules based on vma flag.
              std::fill_n(expected_inactive, 8, (vma ? kAgnosticResult : dst_result));
            }

            if (emul < 4) {
              for (size_t index = 0; index < 1 << emul; ++index) {
                if (index == 0 && emul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            ((dst_result & kFractionMaskInt8[3]) |
                             (SIMD128Register{expected_result[index]} & mask[index] &
                              ~kFractionMaskInt8[3]) |
                             (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3]))
                                .template Get<__uint128_t>());
                } else if (index == 2 && emul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            ((SIMD128Register{expected_result[index]} & mask[index] &
                              kFractionMaskInt8[3]) |
                             (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                             ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[3]))
                                .template Get<__uint128_t>());
                } else if (index == 3 && emul == 2 && vta) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
                } else if (index == 3 && emul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{dst_result});
                } else {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            ((SIMD128Register{expected_result[index]} & mask[index]) |
                             ((expected_inactive[index] & ~mask[index])))
                                .template Get<__uint128_t>());
                }
              }
            } else {
              EXPECT_EQ(
                  state_.cpu.v[8],
                  ((SIMD128Register{expected_result[0]} & mask[0] & kFractionMaskInt8[emul - 4]) |
                   (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[emul - 4]) |
                   ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[emul - 4]))
                      .template Get<__uint128_t>());
            }

            if (emul == 2) {
              // Every vector instruction must set vstart to 0, but shouldn't touch vl.
              EXPECT_EQ(state_.cpu.vstart, 0);
              EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
            }
          }
        }
      }
    };

    // Some instructions don't support use of mask register, but in these instructions bit
    // #25 is set.  This function doesn't support these. Verify that vm bit is not set.
    EXPECT_EQ(insn_bytes & (1 << 25), 0U);
    // Every insruction is tested with vm bit not set (and mask register used) and with vm bit
    // set (and mask register is not used).
    ((Verify(insn_bytes,
             BitUtilLog2(sizeof(std::remove_cvref_t<decltype(std::get<0>(expected_result[0]))>)) -
                 (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening),
             8,
             expected_result,
             MaskForElem<std::remove_cvref_t<decltype(std::get<0>(expected_result[0]))>>()),
      Verify((insn_bytes &
              ~(0x01f0'0000 * (kTestVectorInstructionMode == TestVectorInstructionMode::kVMerge))) |
                 (1 << 25),
             BitUtilLog2(sizeof(std::remove_cvref_t<decltype(std::get<0>(expected_result[0]))>)) -
                 (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening),
             8,
             expected_result,
             kNoMask)),
     ...);
  }

  void TestVectorMaskInstruction(uint8_t max_vstart,
                                 intrinsics::InactiveProcessing vma,
                                 uint32_t insn_bytes,
                                 const __v2du expected_result) {
    // Mask instructions don't look on vtype directly, but they still require valid one because it
    // affects vlmax;
    auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, 3 | (static_cast<uint8_t>(vma) << 7));
    // We need mask with a few bits set for Vmsₓf instructions.  Inverse of normal kMask works.
    const __uint128_t mask = SIMD128Register{~kMask}.Get<__uint128_t>();
    const __uint128_t undisturbed = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    const __uint128_t src1 = SIMD128Register{kVectorCalculationsSourceLegacy[0]}.Get<__uint128_t>();
    const __uint128_t src2 = SIMD128Register{kVectorCalculationsSourceLegacy[8]}.Get<__uint128_t>();
    const __uint128_t expected = SIMD128Register{expected_result}.Get<__uint128_t>();
    state_.cpu.vtype = vtype;
    for (uint8_t vl = 0; vl <= vlmax; ++vl) {
      state_.cpu.vl = vl;
      for (uint8_t vstart = 0; vstart <= max_vstart; ++vstart) {
        state_.cpu.vstart = vstart;
        // Set expected_result vector registers into 0b01010101… pattern.
        state_.cpu.v[0] = mask;
        state_.cpu.v[8] = undisturbed;
        state_.cpu.v[16] = src1;
        state_.cpu.v[24] = src2;

        state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
        EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

        for (uint8_t bit_pos = 0; bit_pos < 128; ++bit_pos) {
          __uint128_t bit = __uint128_t{1} << bit_pos;
          // When vstart ⩾ vl, there are no body elements, and no elements are updated in any
          // destinationvector register group, including that no tail elements are updated with
          // agnostic values.
          if (bit_pos < vstart || vstart >= vl) {
            EXPECT_EQ(state_.cpu.v[8] & bit, undisturbed & bit);
          } else if (bit_pos >= vl) {
            EXPECT_EQ(state_.cpu.v[8] & bit, bit);
          } else {
            EXPECT_EQ(state_.cpu.v[8] & bit, expected & bit);
          }
        }
      }
    }
  }

  template <typename ElementType>
  auto MaskForElem() {
    if constexpr (std::is_same_v<ElementType, uint8_t>) {
      return kMaskInt8;
    } else if constexpr (std::is_same_v<ElementType, uint16_t>) {
      return kMaskInt16;
    } else if constexpr (std::is_same_v<ElementType, uint32_t>) {
      return kMaskInt32;
    } else if constexpr (std::is_same_v<ElementType, uint64_t>) {
      return kMaskInt64;
    } else {
      static_assert(kDependentTypeFalse<ElementType>);
    }
  }

  void TestVectorMaskTargetInstruction(uint32_t insn_bytes,
                                       const uint32_t expected_result_int32,
                                       const uint16_t expected_result_int64,
                                       const __v2du (&source)[16]) {
    TestVectorMaskTargetInstruction(
        insn_bytes, source, expected_result_int32, expected_result_int64);
  }

  void TestVectorMaskTargetInstruction(uint32_t insn_bytes,
                                       const UInt8x16Tuple(&expected_result_int8),
                                       const uint64_t expected_result_int16,
                                       const uint32_t expected_result_int32,
                                       const uint16_t expected_result_int64,
                                       const __v2du (&source)[16]) {
    TestVectorMaskTargetInstruction(insn_bytes,
                                    source,
                                    expected_result_int8,
                                    expected_result_int16,
                                    expected_result_int32,
                                    expected_result_int64);
  }

  template <typename... ExpectedResultType>
  void TestVectorMaskTargetInstruction(uint32_t insn_bytes,
                                       const __v2du (&source)[16],
                                       const ExpectedResultType(&... expected_result)) {
    auto Verify = [this, &source](
                      uint32_t insn_bytes, uint8_t vsew, const auto& expected_result, auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }
      // Set x1 for vx instructions.
      SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);
      // Set f1 for vf instructions.
      if (vsew == 2) {
        SetFReg<1>(state_.cpu, 0xffff'ffff'40b4'0000);  // float 5.625
      } else if (vsew == 3) {
        SetFReg<1>(state_.cpu, 0x4016'8000'0000'0000);  // double 5.625
      }
      for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
        for (uint8_t vta = 0; vta < 2; ++vta) {  // vta should be ignored but we test both values!
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
            // Incompatible vsew and vlmax. Skip it.
            if (vlmax == 0) {
              continue;
            }

            // To make tests quick enough we don't test vstart and vl change with small register
            // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
            // register, last register and half of next-to last register.
            // Don't use vlmul == 3 because that one may not be supported if instruction widens the
            // result.
            if (vlmul == 2) {
              state_.cpu.vstart = vlmax / 8;
              state_.cpu.vl = (vlmax * 5) / 8;
            } else {
              state_.cpu.vstart = 0;
              state_.cpu.vl = vlmax;
            }
            state_.cpu.vtype = vtype;

            // Set expected_result vector registers into 0b01010101… pattern.
            for (size_t index = 0; index < 8; ++index) {
              state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
            }

            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

            SIMD128Register expected_result_in_register(expected_result);
            if (vma == 0) {
              expected_result_in_register = (expected_result_in_register & SIMD128Register{mask}) |
                                            (kUndisturbedResult & ~SIMD128Register{mask});
            } else {
              expected_result_in_register = expected_result_in_register | ~SIMD128Register{mask};
            }
            // Mask registers are always processing tail like vta is set.
            if (vlmax != 128)
              expected_result_in_register |= std::get<0>(
                  intrinsics::MakeBitmaskFromVl((vlmul == 2) ? (vlmax * 5) / 8 : vlmax));
            if (vlmul == 2) {
              const auto [start_mask] = intrinsics::MakeBitmaskFromVl(vlmax / 8);
              expected_result_in_register = (SIMD128Register{kUndisturbedResult} & ~start_mask) |
                                            (expected_result_in_register & start_mask);
            }
            EXPECT_EQ(state_.cpu.v[8], expected_result_in_register);

            if (vlmul == 2) {
              // Every vector instruction must set vstart to 0, but shouldn't touch vl.
              EXPECT_EQ(state_.cpu.vstart, 0);
              EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
            }
          }
        }
      }
    };

    ((Verify(insn_bytes,
             BitUtilLog2(sizeof(SIMD128Register) / sizeof(ExpectedResultType)),
             expected_result,
             kMask),
      Verify(insn_bytes | (1 << 25),
             BitUtilLog2(sizeof(SIMD128Register) / sizeof(ExpectedResultType)),
             expected_result,
             kNoMask[0])),
     ...);
  }

  void TestVectorCarryInstruction(uint32_t insn_bytes,
                                  const UInt8x16Tuple (&expected_result_int8)[8],
                                  const UInt16x8Tuple (&expected_result_int16)[8],
                                  const UInt32x4Tuple (&expected_result_int32)[8],
                                  const UInt64x2Tuple (&expected_result_int64)[8],
                                  const __v2du (&source)[16]) {
    auto Verify = [this, &source](uint32_t insn_bytes, uint8_t vsew, const auto& expected_result) {
      __m128i dst_result = kUndisturbedResult;

      // Set mask register
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();

      // Set source registers
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }

      // Set x1 for vx instructions.
      SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);

      for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          uint8_t vma = 0;
          auto [vlmax, vtype] =
              intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
          // Incompatible vsew and vlmax. Skip it.
          if (vlmax == 0) {
            continue;
          }
          uint8_t emul = vlmul & 0b111;

          // To make tests quick enough we don't test vstart and vl change with small register
          // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
          // register, last register and half of next-to last register.
          // Don't use vlmul == 3 because that one may not be supported if instruction widens the
          // result.
          if (vlmul == 2) {
            state_.cpu.vstart = vlmax / 8;
            state_.cpu.vl = (vlmax * 5) / 8;
          } else {
            state_.cpu.vstart = 0;
            state_.cpu.vl = vlmax;
          }
          state_.cpu.vtype = vtype;

          // Set expected_result vector registers into 0b01010101… pattern.
          for (size_t index = 0; index < 8; ++index) {
            state_.cpu.v[8 + index] = SIMD128Register{dst_result}.Get<__uint128_t>();
          }

          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

          if (emul < 4) {
            for (size_t index = 0; index < 1 << emul; ++index) {
              if (index == 0 && emul == 2) {
                EXPECT_EQ(state_.cpu.v[8 + index],
                          ((dst_result & kFractionMaskInt8[3]) |
                           (SIMD128Register{expected_result[index]} & ~kFractionMaskInt8[3]))
                              .template Get<__uint128_t>());
              } else if (index == 2 && emul == 2) {
                EXPECT_EQ(state_.cpu.v[8 + index],
                          ((SIMD128Register{expected_result[index]} & kFractionMaskInt8[3]) |
                           ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[3]))
                              .template Get<__uint128_t>());
              } else if (index == 3 && emul == 2 && vta) {
                EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
              } else if (index == 3 && emul == 2) {
                EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{dst_result});
              } else {
                EXPECT_EQ(state_.cpu.v[8 + index],
                          (SIMD128Register{expected_result[index]}).template Get<__uint128_t>());
              }
            }
          } else {
            EXPECT_EQ(state_.cpu.v[8],
                      ((SIMD128Register{expected_result[0]} & kFractionMaskInt8[emul - 4]) |
                       ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[emul - 4]))
                          .template Get<__uint128_t>());
          }

          if (emul == 2) {
            // Every vector instruction must set vstart to 0, but shouldn't touch vl.
            EXPECT_EQ(state_.cpu.vstart, 0);
            EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
          }
        }
      }
    };

    // Some instructions don't support use of mask register, but in these instructions bit
    // #25 is set.  This function doesn't support these. Verify that vm bit is not set.
    EXPECT_EQ(insn_bytes & (1 << 25), 0U);

    Verify(insn_bytes, 0, expected_result_int8);
    Verify(insn_bytes, 1, expected_result_int16);
    Verify(insn_bytes, 2, expected_result_int32);
    Verify(insn_bytes, 3, expected_result_int64);
  }

  void TestVXmXXsInstruction(uint32_t insn_bytes,
                            const uint64_t (&expected_result_no_mask)[129],
                            const uint64_t (&expected_result_with_mask)[129],
                            const __v2du source) {
    auto Verify = [this, &source](uint32_t insn_bytes,
                                  const uint64_t (&expected_result)[129]) {
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();

      auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, 3);
      state_.cpu.vtype = vtype;
      state_.cpu.vstart = 0;
      state_.cpu.v[16] = SIMD128Register{source}.Get<__uint128_t>();

      for (uint8_t vl = 0; vl <= vlmax; ++vl) {
        state_.cpu.vl = vl;
        SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);

        state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
        EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
        EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result[vl]) << std::to_string(vl);
      }
    };

    Verify(insn_bytes, expected_result_with_mask);
    Verify(insn_bytes | (1 << 25), expected_result_no_mask);
  }

  void TestVectorFloatPermutationInstruction(uint32_t insn_bytes,
                                             const UInt32x4Tuple (&expected_result_int32)[8],
                                             const UInt64x2Tuple (&expected_result_int64)[8],
                                             const __v2du (&source)[16],
                                             uint8_t vlmul,
                                             uint64_t skip = 0,
                                             bool ignore_vma_for_last = false,
                                             bool last_elem_is_f1 = false) {
    TestVectorPermutationInstruction<TestVectorInstructionKind::kFloat>(insn_bytes,
                                                                        source,
                                                                        vlmul,
                                                                        skip,
                                                                        ignore_vma_for_last,
                                                                        last_elem_is_f1,
                                                                        /* regx1 */ 0x0,
                                                                        expected_result_int32,
                                                                        expected_result_int64);
  }

  void TestVectorPermutationInstruction(uint32_t insn_bytes,
                                        const UInt8x16Tuple (&expected_result_int8)[8],
                                        const UInt16x8Tuple (&expected_result_int16)[8],
                                        const UInt32x4Tuple (&expected_result_int32)[8],
                                        const UInt64x2Tuple (&expected_result_int64)[8],
                                        const __v2du (&source)[16],
                                        uint8_t vlmul,
                                        uint64_t regx1 = 0x0,
                                        uint64_t skip = 0,
                                        bool ignore_vma_for_last = false,
                                        bool last_elem_is_x1 = false) {
    TestVectorPermutationInstruction<TestVectorInstructionKind::kInteger>(insn_bytes,
                                                                          source,
                                                                          vlmul,
                                                                          skip,
                                                                          ignore_vma_for_last,
                                                                          last_elem_is_x1,
                                                                          regx1,
                                                                          expected_result_int8,
                                                                          expected_result_int16,
                                                                          expected_result_int32,
                                                                          expected_result_int64);
  }

  // Unlike regular arithmetic instructions, the result of a permutation
  // instruction depends also on vlmul.  Also, the vslideup specs mention that
  // the destination vector remains unchanged the first |offset| elements (in
  // effect, the offset acts akin to vstart), in those cases skip can be used
  // to specify how many elements' mask will be skipped (counting from the
  // beginning, should be the same as the offset).
  //
  // If |ignore_vma_for_last| is true, an inactive element at vl-1 will be
  // treated as if vma=0 (Undisturbed).
  // If |last_elem_is_reg1| is true, the last element of the vector in
  // expected_result (that is, at vl-1) will be expected to be the same as
  // |regx1| when VL < VMAX and said element is active.
  template <TestVectorInstructionKind kTestVectorInstructionKind,
            typename... ExpectedResultType,
            size_t... kResultsCount>
  void TestVectorPermutationInstruction(uint32_t insn_bytes,
                                        const __v2du (&source)[16],
                                        uint8_t vlmul,
                                        uint64_t skip,
                                        bool ignore_vma_for_last,
                                        bool last_elem_is_reg1,
                                        uint64_t regx1,
                                        const ExpectedResultType&... expected_result) {
    auto Verify = [this, &source, vlmul, regx1, skip, ignore_vma_for_last, last_elem_is_reg1](
                      uint32_t insn_bytes,
                      uint8_t vsew,
                      const auto& expected_result_raw,
                      auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }

      if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
        UNUSED(regx1);
        // We only support Float32/Float64 for float instructions, but there are conversion
        // instructions that work with double width floats.
        // These instructions never use float registers though and thus we don't need to store
        // anything into f1 register, if they are used.
        // For Float32/Float64 case we load 5.625 of the appropriate type into f1.
        ASSERT_LE(vsew, 3);
        if (vsew == 2) {
          SetFReg<1>(state_.cpu, 0xffff'ffff'40b4'0000);  // float 5.625
        } else if (vsew == 3) {
          SetFReg<1>(state_.cpu, 0x4016'8000'0000'0000);  // double 5.625
        }
      } else {
        // Set x1 for vx instructions.
        SetXReg<1>(state_.cpu, regx1);
      }

      const size_t kElementSize = 1 << vsew;
      size_t num_regs = 1 << vlmul;
      if (vlmul > 3) {
        num_regs = 1;
      }
      // Values for which the mask is not applied due to being before the offset when doing
      // vslideup.
      SIMD128Register skip_mask[num_regs];
      int64_t toskip = skip;
      for (size_t index = 0; index < num_regs && toskip > 0; ++index) {
        size_t skip_bits = toskip * kElementSize * 8;
        skip_mask[index] =
            ~std::get<0>(intrinsics::MakeBitmaskFromVl(skip_bits > 128 ? 128 : skip_bits));
        toskip -= 16 / kElementSize;
      }

      for (uint8_t vta = 0; vta < 2; ++vta) {
        for (uint8_t vma = 0; vma < 2; ++vma) {
          auto [vlmax, vtype] =
              intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
          // Incompatible vsew and vlmax. Skip it.
          if (vlmax == 0) {
            continue;
          }

          // To make tests quick enough we don't test vstart and vl change with small register
          // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
          // register, last register and half of next-to last register.
          // Don't use vlmul == 3 because that one may not be supported if instruction widens the
          // result.
          if (vlmul == 2) {
            state_.cpu.vstart = vlmax / 8;
            state_.cpu.vl = (vlmax * 5) / 8;
          } else {
            state_.cpu.vstart = 0;
            state_.cpu.vl = vlmax;
          }
          state_.cpu.vtype = vtype;

          // Set dst vector registers into 0b01010101… pattern.
          for (size_t index = 0; index < 8; ++index) {
            state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
          }

          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

          const size_t n = std::size(source);
          // Values for inactive elements (i.e. corresponding mask bit is 0).
          __m128i expected_inactive[n];
          // For most instructions, follow basic inactive processing rules based on vma flag.
          std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));

          const size_t kElementsPerRegister = 16 / kElementSize;
          const size_t last_reg = (state_.cpu.vl - 1) / kElementsPerRegister;
          const size_t last_elem = (state_.cpu.vl - 1) % kElementsPerRegister;
          const auto [mask_for_vl] = intrinsics::MakeBitmaskFromVl(last_elem * kElementSize * 8);
          if (vma && ignore_vma_for_last) {
            // Set expected value for inactive element at vl-1 to Undisturbed.
            expected_inactive[last_reg] =
                ((expected_inactive[last_reg] & ~mask_for_vl) | (kUndisturbedResult & mask_for_vl))
                    .Get<__m128i>();
          }

          SIMD128Register expected_result[std::size(expected_result_raw)];
          for (size_t index = 0; index < std::size(expected_result_raw); ++index) {
            expected_result[index] = SIMD128Register{expected_result_raw[index]};
          }

          if (vlmul == 2 && last_elem_is_reg1) {
            switch (kElementSize) {
              case 1:
                expected_result[last_reg].template Set<uint8_t>(
                    static_cast<uint8_t>(GetXReg<1>(state_.cpu)), last_elem);
                break;
              case 2:
                expected_result[last_reg].template Set<uint16_t>(
                    static_cast<uint16_t>(GetXReg<1>(state_.cpu)), last_elem);
                break;
              case 4:
                if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
                  expected_result[last_reg].template Set<uint32_t>(
                      static_cast<uint32_t>(GetFReg<1>(state_.cpu)), last_elem);
                } else {
                  expected_result[last_reg].template Set<uint32_t>(
                      static_cast<uint32_t>(GetXReg<1>(state_.cpu)), last_elem);
                }
                break;
              case 8:
                if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
                  expected_result[last_reg].template Set<uint64_t>(
                      static_cast<uint64_t>(GetFReg<1>(state_.cpu)), last_elem);
                } else {
                  expected_result[last_reg].template Set<uint64_t>(
                      static_cast<uint64_t>(GetXReg<1>(state_.cpu)), last_elem);
                }
                break;
              default:
                FAIL() << "Element size is " << kElementSize;
            }
          }

          if (vlmul < 4) {
            for (size_t index = 0; index < num_regs; ++index) {
              if (index == 0 && vlmul == 2) {
                EXPECT_EQ(
                    state_.cpu.v[8 + index],
                    SIMD128Register{(kUndisturbedResult & kFractionMaskInt8[3]) |
                                    (expected_result[index] & (mask[index] | skip_mask[index]) &
                                     ~kFractionMaskInt8[3]) |
                                    (expected_inactive[index] & ~mask[index] & ~skip_mask[index] &
                                     ~kFractionMaskInt8[3])}
                        .Get<__uint128_t>());
              } else if (index == 2 && vlmul == 2) {
                EXPECT_EQ(
                    state_.cpu.v[8 + index],
                    SIMD128Register{
                        (expected_result[index] & (mask[index] | skip_mask[index]) &
                         kFractionMaskInt8[3]) |
                        (expected_inactive[index] & ~mask[index] & ~skip_mask[index] &
                         kFractionMaskInt8[3]) |
                        ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])}
                        .Get<__uint128_t>());
              } else if (index == 3 && vlmul == 2 && vta) {
                EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
              } else if (index == 3 && vlmul == 2) {
                EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kUndisturbedResult});
              } else {
                EXPECT_EQ(
                    state_.cpu.v[8 + index],
                    SIMD128Register{(expected_result[index] & (mask[index] | skip_mask[index])) |
                                    (expected_inactive[index] & ~(mask[index] | skip_mask[index]))}
                        .Get<__uint128_t>());
              }
            }
          } else {
            __uint128_t v8 = state_.cpu.v[8];
            SIMD128Register affected_part{
                expected_result[0] & ((mask[0] & kFractionMaskInt8[vlmul - 4]) | skip_mask[0])};
            SIMD128Register masked_part{expected_inactive[0] & ~mask[0] & ~skip_mask[0] &
                                        kFractionMaskInt8[vlmul - 4]};
            SIMD128Register tail_part{(vta ? kAgnosticResult : kUndisturbedResult) &
                                      ~kFractionMaskInt8[vlmul - 4]};

            EXPECT_EQ(v8, (affected_part | masked_part | tail_part).Get<__uint128_t>());
          }

          if (vlmul == 2) {
            // Every vector instruction must set vstart to 0, but shouldn't touch vl.
            EXPECT_EQ(state_.cpu.vstart, 0);
            EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
          }
        }
      }
    };

    // Test with and without masking enabled.
    (Verify(insn_bytes,
            BitUtilLog2(sizeof(std::remove_cvref_t<decltype(std::get<0>(expected_result[0]))>)),
            expected_result,
            MaskForElem<std::remove_cvref_t<decltype(std::get<0>(expected_result[0]))>>()),
     ...);
    (Verify(insn_bytes | (1 << 25),
            BitUtilLog2(sizeof(std::remove_cvref_t<decltype(std::get<0>(expected_result[0]))>)),
            expected_result,
            kNoMask),
     ...);
  }

 protected:
  static constexpr __v2du kVectorCalculationsSource[16] = {
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
      {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
      {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
      {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978},

      {0x9e0c'9a09'9604'9200, 0x8e1c'8a18'8614'8211},
      {0xbe2c'ba29'b624'b220, 0xae3c'aa38'a634'a231},
      {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
      {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
      {0x1e8c'1a89'1684'1280, 0x0e9c'0a98'0694'0291},
      {0x3eac'3aa9'36a4'32a0, 0x2ebc'2ab8'26b4'22b1},
      {0x5ecc'5ac9'56c4'52c0, 0x4edc'4ad8'46d4'42d1},
      {0x7eec'7ae9'76e4'72e0, 0x6efc'6af8'66f4'62f1},
  };

  static constexpr __v2du kVfClassSource[16] = {
      {0x8000'0000'0000'0000, 0x8e1c'8a18'8614'8211},
      {0x0000'0000'0000'0000, 0xae3c'aa38'a634'a231},
      {0x7ff0'0000'0000'0000, 0xff80'0000'7f80'0000},
      {0xfff0'0000'0000'0000, 0xee7c'ea78'e674'e271},
      {0x1e8c'1a89'1684'1280, 0x0e9c'0a98'0694'0291},
      {0x3eac'3aa9'36a4'32a0, 0x2ebc'2ab8'26b4'22b1},
      {0x5ecc'5ac9'56c4'52c0, 0x4edc'4ad8'46d4'42d1},
      {0x7fb7'ffff'7fb7'ffff, 0x7ff7'ffff'ffff'ffff},

      {0x8000'0000'0000'0000, 0x8e1c'8a18'8614'8211},
      {0x0000'0000'0000'0000, 0xae3c'aa38'a634'a231},
      {0x7ff0'0000'0000'0000, 0xff80'0000'7f80'0000},
      {0xfff0'0000'0000'0000, 0xee7c'ea78'e674'e271},
      {0x1e8c'1a89'1684'1280, 0x0e9c'0a98'0694'0291},
      {0x3eac'3aa9'36a4'32a0, 0x2ebc'2ab8'26b4'22b1},
      {0x5ecc'5ac9'56c4'52c0, 0x4edc'4ad8'46d4'42d1},
      {0x7fb7'ffff'7fb7'ffff, 0x7ff7'ffff'ffff'ffff},
  };

  static constexpr __v2du kVectorCalculationsSourceLegacy[16] = {
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
      {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
      {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
      {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978},

      {0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
      {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
      {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
      {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
      {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
      {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
      {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
      {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1},
  };

  static constexpr __v2du kVectorComparisonSource[16] = {
      {0xf005'f005'f005'f005, 0xffff'ffff'4040'4040},
      {0xffff'ffff'40b4'40b4, 0xffff'ffff'40b4'0000},
      {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
      {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
      {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
      {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
      {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
      {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9},

      {0xf005'f005'f005'f005, 0xffff'ffff'4040'4040},
      {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
      {0xfff1'fff1'fff1'fff1, 0xfff1'fff1'fff1'fff1},
      {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
      {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
      {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
      {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
      {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1},
  };

  // Mask in form suitable for storing in v0 and use in v0.t form.
  static constexpr __v2du kMask = {0xd5ad'd6b5'ad6b'b5ad, 0x6af7'57bb'deed'7bb5};
  // Mask used with vsew = 0 (8bit) elements.
  static constexpr __v16qu kMaskInt8[8] = {
      {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255},
      {255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255},
      {255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255},
      {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 0, 255, 255},
      {255, 0, 255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0},
      {255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 255},
      {255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0},
      {255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 0},
  };
  // Mask used with vsew = 1 (16bit) elements.
  static constexpr __v8hu kMaskInt16[8] = {
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
      {0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000},
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
      {0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff},
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff},
  };
  // Mask used with vsew = 2 (32bit) elements.
  static constexpr __v4su kMaskInt32[8] = {
      {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
      {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
      {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000},
      {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
      {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
      {0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0x0000'0000},
      {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
      {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
  };
  // Mask used with vsew = 3 (64bit) elements.
  static constexpr __v2du kMaskInt64[8] = {
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
  };
  // To verify operations without masking.
  static constexpr __v16qu kNoMask[8] = {
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
  };
  // Half of sub-register lmul.
  static constexpr __v16qu kFractionMaskInt8[5] = {
      {255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},        // Half of ⅛ reg = ¹⁄₁₆
      {255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // Half of ¼ reg = ⅛
      {255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // Half of ½ reg = ¼
      {255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0},  // Half of full reg = ½
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},  // Full reg
  };
  // Agnostic result is -1 on RISC-V, not 0.
  static constexpr __m128i kAgnosticResult = {-1, -1};
  // Undisturbed result is put in registers v8, v9, …, v15 and is expected to get read back.
  static constexpr __m128i kUndisturbedResult = {0x5555'5555'5555'5555, 0x5555'5555'5555'5555};
  // Note: permutation of indexes here is not entirely random. First 32 indexes are limited to 31
  // maximum and first 64 indexes are limited to 63. That way we can guarantee that 8byte elements
  // and 4byte elements wouldn't need to access area outside of our 256-byte buffer.
  static constexpr uint8_t kPermutedIndexes[128] = {
      1,   0,   3,  2,   7,   5,   4,   6,   9,   14,  15,  11,  13,  12, 8,  10,  30,  31,  17,
      22,  18,  26, 25,  19,  29,  28,  16,  21,  27,  24,  20,  23,  44, 50, 52,  34,  61,  38,
      54,  43,  42, 63,  57,  40,  36,  46,  39,  47,  35,  41,  62,  59, 60, 51,  55,  53,  33,
      32,  58,  49, 56,  37,  45,  48,  124, 92,  78,  101, 114, 89,  75, 64, 98,  112, 111, 118,
      121, 102, 73, 105, 109, 68,  103, 72,  110, 79,  119, 96,  123, 85, 90, 126, 66,  69,  120,
      97,  113, 76, 100, 67,  125, 117, 65,  84,  104, 122, 71,  81,  99, 70, 91,  86,  115, 127,
      77,  107, 74, 93,  80,  106, 87,  94,  83,  95,  116, 108, 82,  88};

  // Store area for store instructions.  We need at least 16 uint64_t to handle 8×128bit registers,
  // plus 2× of that to test strided instructions.
  alignas(16) uint64_t store_area_[32];

  ThreadState state_;
};

#define TESTSUITE Riscv64InterpretInsnTest
#define TESTING_INTERPRETER

#include "berberis/test_utils/insn_tests_riscv64-inl.h"

#undef TESTING_INTERPRETER
#undef TESTSUITE

// Tests for Non-Compressed Instructions.

TEST_F(Riscv64InterpreterTest, FenceInstructions) {
  // Fence
  InterpretFence(0x0ff0'000f);
  // FenceTso
  InterpretFence(0x8330'000f);

  // FenceI explicitly not supported.
}

TEST_F(Riscv64InterpreterTest, SyscallWrite) {
  const char message[] = "Hello";
  // Prepare a pipe to write to.
  int pipefd[2];
  ASSERT_EQ(0, pipe(pipefd));

  // Only ecall instruction needs guest thread, since it involves pending signals manipulations.
  std::unique_ptr<GuestThread, decltype(&GuestThread::Destroy)> guest_thread(
      GuestThread::CreateForTest(&state_), GuestThread::Destroy);
  state_.thread = guest_thread.get();

  // SYS_write
  SetXReg<17>(state_.cpu, 0x40);
  // File descriptor
  SetXReg<10>(state_.cpu, pipefd[1]);
  // String
  SetXReg<11>(state_.cpu, bit_cast<uint64_t>(&message[0]));
  // Size
  SetXReg<12>(state_.cpu, sizeof(message));

  uint32_t insn_bytes = 0x0000'0073;
  state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
  InterpretInsn(&state_);

  // Check number of bytes written.
  EXPECT_EQ(GetXReg<10>(state_.cpu), sizeof(message));

  // Check the message was written to the pipe.
  char buf[sizeof(message)] = {};
  ssize_t read_size = read(pipefd[0], &buf, sizeof(buf));
  EXPECT_NE(read_size, -1);
  EXPECT_EQ(0, strcmp(message, buf));
  close(pipefd[0]);
  close(pipefd[1]);
}

TEST_F(Riscv64InterpreterTest, TestFPExceptions) {
  // Keep the same sort as Section 19 "Vector Instruction Listing".
  TestFPExceptions<intrinsics::Float32>(0x012d'1557);  // Vfadd.vv v10, v18, v26, v0.t
  TestFPExceptions<intrinsics::Float64>(0x010c'1457);  // Vfadd.vv v8, v16, v24, v0.t
  TestFPExceptions<intrinsics::Float32>(0x0120'd557);  // Vfadd.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x0101'5457);  // Vfadd.vf v8, v16, f2, v0.t
  TestFPExceptions<intrinsics::Float32>(0x092d'1557);  // Vfsub.vv v10, v18, v26, v0.t
  TestFPExceptions<intrinsics::Float64>(0x090c'1457);  // Vfsub.vv v8, v16, v24, v0.t
  TestFPExceptions<intrinsics::Float32>(0x0920'd557);  // Vfsub.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x0901'5457);  // Vfsub.vf v8, v16, f2, v0.t
  TestFPExceptions<intrinsics::Float32>(0x812d'1557);  // Vfdiv.vv v10, v18, v26, v0.t
  TestFPExceptions<intrinsics::Float64>(0x810c'1457);  // Vfdiv.vv v8, v16, v24, v0.t
  TestFPExceptions<intrinsics::Float32>(0x8120'd557);  // Vfdiv.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x8101'5457);  // Vfdiv.vf v8, v16, f2, v0.t
  TestFPExceptions<intrinsics::Float32>(0x912d'1557);  // Vfmul.vv v10, v18, v26, v0.t
  TestFPExceptions<intrinsics::Float64>(0x910c'1457);  // Vfmul.vv v8, v16, v24, v0.t
  TestFPExceptions<intrinsics::Float32>(0x9120'd557);  // Vfmul.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x9101'5457);  // Vfmul.vf v8, v16, f2, v0.t
  TestFPExceptions<intrinsics::Float32>(0x9d20'd557);  // Vfrsub.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x9d01'5457);  // Vfrsub.vf v8, v16, f2, v0.t
}

TEST_F(Riscv64InterpreterTest, TestVlXreXX) {
  TestVlXreXX<1>(0x0280'8407);  // vl1re8.v v8, (x1)
  TestVlXreXX<2>(0x2280'8407);  // vl2re8.v v8, (x1)
  TestVlXreXX<4>(0x6280'8407);  // vl4re8.v v8, (x1)
  TestVlXreXX<8>(0xe280'8407);  // vl8re8.v v8, (x1)

  TestVlXreXX<1>(0x0280'd407);  // vl1re16.v v8, (x1)
  TestVlXreXX<2>(0x2280'd407);  // vl2re16.v v8, (x1)
  TestVlXreXX<4>(0x6280'd407);  // vl4re16.v v8, (x1)
  TestVlXreXX<8>(0xe280'd407);  // vl8re16.v v8, (x1)

  TestVlXreXX<1>(0x0280'e407);  // vl1re32.v v8, (x1)
  TestVlXreXX<2>(0x2280'e407);  // vl2re32.v v8, (x1)
  TestVlXreXX<4>(0x6280'e407);  // vl4re32.v v8, (x1)
  TestVlXreXX<8>(0xe280'e407);  // vl8re32.v v8, (x1)

  TestVlXreXX<1>(0x0280'f407);  // vl1re64.v v8, (x1)
  TestVlXreXX<2>(0x2280'f407);  // vl2re64.v v8, (x1)
  TestVlXreXX<4>(0x6280'f407);  // vl4re64.v v8, (x1)
  TestVlXreXX<8>(0xe280'f407);  // vl8re64.v v8, (x1)
}

TEST_F(Riscv64InterpreterTest, TestVmXr) {
  TestVmvXr<1>(0x9f00'3457);  // Vmv1r.v v8, v16
  TestVmvXr<2>(0x9f00'b457);  // Vmv2r.v v8, v16
  TestVmvXr<4>(0x9f01'b457);  // Vmv4r.v v8, v16
  TestVmvXr<8>(0x9f03'b457);  // Vmv8r.v v8, v16
}

TEST_F(Riscv64InterpreterTest, TestVfrsqrt7) {
  TestVectorFloatInstruction(0x4d82'1457,  // Vfrsqrt7.v v8, v24, v0.t
                             {{0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x53fb'8000, 0x4ff4'8000, 0x5bed'8000, 0x57e7'8000},
                              {0x43e2'0000, 0x3fdc'8000, 0x4bd7'8000, 0x47d3'0000},
                              {0x33ce'8000, 0x2fca'8000, 0x3bc6'8000, 0x37c3'0000},
                              {0x23bf'8000, 0x1fbc'8000, 0x2bb9'0000, 0x27b6'8000}},
                             {{0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x50a1'1000'0000'0000, 0x5898'3000'0000'0000},
                              {0x4091'1000'0000'0000, 0x4888'2000'0000'0000},
                              {0x3081'0000'0000'0000, 0x3878'1000'0000'0000},
                              {0x2071'0000'0000'0000, 0x2868'0000'0000'0000}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfclass) {
  TestVectorFloatInstruction(0x4d88'1457,  // Vfclass.v v8, v24, v0.t
                             {{0x0000'0010, 0x0000'0008, 0x0000'0002, 0x0000'0002},
                              {0x0000'0010, 0x0000'0010, 0x0000'0002, 0x0000'0002},
                              {0x0000'0010, 0x0000'0200, 0x0000'0080, 0x0000'0001},
                              {0x0000'0010, 0x0000'0200, 0x0000'0002, 0x0000'0002},
                              {0x0000'0040, 0x0000'0040, 0x0000'0040, 0x0000'0040},
                              {0x0000'0040, 0x0000'0040, 0x0000'0040, 0x0000'0040},
                              {0x0000'0040, 0x0000'0040, 0x0000'0040, 0x0000'0040},
                              {0x0000'0100, 0x0000'0100, 0x0000'0200, 0x0000'0200}},
                             {{0x0000'0000'0000'0008, 0x0000'0000'0000'0002},
                              {0x0000'0000'0000'0010, 0x0000'0000'0000'0002},
                              {0x0000'0000'0000'0080, 0x0000'0000'0000'0002},
                              {0x0000'0000'0000'0001, 0x0000'0000'0000'0002},
                              {0x0000'0000'0000'0040, 0x0000'0000'0000'0040},
                              {0x0000'0000'0000'0040, 0x0000'0000'0000'0040},
                              {0x0000'0000'0000'0040, 0x0000'0000'0000'0040},
                              {0x0000'0000'0000'0040, 0x0000'0000'0000'0100}},
                             kVfClassSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmvfs) {
  TestVfmvfs<intrinsics::Float32>(0x4280'10d7, 0xffff'ffff'8302'8100);  // Vfmv.f.s f1, v8
  TestVfmvfs<intrinsics::Float64>(0x4280'10d7, 0x8706'8504'8302'8100);  // Vfmv.f.s f1, v8
}

TEST_F(Riscv64InterpreterTest, TestVfmvsf) {
  TestVfmvsf<intrinsics::Float32>(0x4200'd457,  // Vfmv.s.f v8, f1
                                  0xffff'ffff'40b4'0000,
                                  intrinsics::Float32{5.625f});
  TestVfmvsf<intrinsics::Float64>(0x4200'd457,  // Vfmv.s.f v8, f1
                                  0x4016'8000'0000'0000,
                                  intrinsics::Float64{5.625});
}

TEST_F(Riscv64InterpreterTest, TestVmvsx) {
  TestVmvsx<Int8>(0x4200'e457);   // Vmv.s.x v8, x1
  TestVmvsx<Int16>(0x4200'e457);  // Vmv.s.x v8, x1
  TestVmvsx<Int32>(0x4200'e457);  // Vmv.s.x v8, x1
  TestVmvsx<Int64>(0x4200'e457);  // Vmv.s.x v8, x1
}

TEST_F(Riscv64InterpreterTest, TestVmvxs) {
  TestVmvxs<Int8>(0x4280'20d7, 0);                       // Vmv.x.s x1, v8
  TestVmvxs<Int16>(0x4280'20d7, 0xffff'ffff'ffff'8100);  // Vmv.x.s x1, v8
  TestVmvxs<Int32>(0x4280'20d7, 0xffff'ffff'8302'8100);  // Vmv.x.s x1, v8
  TestVmvxs<Int64>(0x4280'20d7, 0x8706'8504'8302'8100);  // Vmv.x.s x1, v8
}

TEST_F(Riscv64InterpreterTest, TestVsX) {
  TestVsX<1>(0x0280'8427);  // vs1r.v v8, (x1)
  TestVsX<2>(0x2280'8427);  // vs2r.v v8, (x1)
  TestVsX<4>(0x6280'8427);  // vs4r.v v8, (x1)
  TestVsX<8>(0xe280'8427);  // vs8r.v v8, (x1)
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew8_vlmul1) {
  VlxsegXeiXX<UInt8, 1, 1>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                           {{129, 0, 131, 2, 135, 133, 4, 6, 137, 14, 143, 139, 141, 12, 8, 10}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew8_vlmul2) {
  VlxsegXeiXX<UInt8, 1, 2>(
      0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
      {{129, 0, 131, 2, 135, 133, 4, 6, 137, 14, 143, 139, 141, 12, 8, 10},
       {30, 159, 145, 22, 18, 26, 153, 147, 157, 28, 16, 149, 155, 24, 20, 151}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew8_vlmul4) {
  VlxsegXeiXX<UInt8, 1, 4>(
      0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
      {{129, 0, 131, 2, 135, 133, 4, 6, 137, 14, 143, 139, 141, 12, 8, 10},
       {30, 159, 145, 22, 18, 26, 153, 147, 157, 28, 16, 149, 155, 24, 20, 151},
       {44, 50, 52, 34, 189, 38, 54, 171, 42, 191, 185, 40, 36, 46, 167, 175},
       {163, 169, 62, 187, 60, 179, 183, 181, 161, 32, 58, 177, 56, 165, 173, 48}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew8_vlmul8) {
  VlxsegXeiXX<UInt8, 1, 8>(
      0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
      {{129, 0, 131, 2, 135, 133, 4, 6, 137, 14, 143, 139, 141, 12, 8, 10},
       {30, 159, 145, 22, 18, 26, 153, 147, 157, 28, 16, 149, 155, 24, 20, 151},
       {44, 50, 52, 34, 189, 38, 54, 171, 42, 191, 185, 40, 36, 46, 167, 175},
       {163, 169, 62, 187, 60, 179, 183, 181, 161, 32, 58, 177, 56, 165, 173, 48},
       {124, 92, 78, 229, 114, 217, 203, 64, 98, 112, 239, 118, 249, 102, 201, 233},
       {237, 68, 231, 72, 110, 207, 247, 96, 251, 213, 90, 126, 66, 197, 120, 225},
       {241, 76, 100, 195, 253, 245, 193, 84, 104, 122, 199, 209, 227, 70, 219, 86},
       {243, 255, 205, 235, 74, 221, 80, 106, 215, 94, 211, 223, 116, 108, 82, 88}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew8_vlmul1) {
  VlxsegXeiXX<UInt8, 2, 1>(
      0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
      {{2, 0, 6, 4, 14, 10, 8, 12, 18, 28, 30, 22, 26, 24, 16, 20},
       {131, 129, 135, 133, 143, 139, 137, 141, 147, 157, 159, 151, 155, 153, 145, 149}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew8_vlmul2) {
  VlxsegXeiXX<UInt8, 2, 2>(
      0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
      {{2, 0, 6, 4, 14, 10, 8, 12, 18, 28, 30, 22, 26, 24, 16, 20},
       {60, 62, 34, 44, 36, 52, 50, 38, 58, 56, 32, 42, 54, 48, 40, 46},
       {131, 129, 135, 133, 143, 139, 137, 141, 147, 157, 159, 151, 155, 153, 145, 149},
       {189, 191, 163, 173, 165, 181, 179, 167, 187, 185, 161, 171, 183, 177, 169, 175}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew8_vlmul4) {
  VlxsegXeiXX<UInt8, 2, 4>(
      0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
      {{2, 0, 6, 4, 14, 10, 8, 12, 18, 28, 30, 22, 26, 24, 16, 20},
       {60, 62, 34, 44, 36, 52, 50, 38, 58, 56, 32, 42, 54, 48, 40, 46},
       {88, 100, 104, 68, 122, 76, 108, 86, 84, 126, 114, 80, 72, 92, 78, 94},
       {70, 82, 124, 118, 120, 102, 110, 106, 66, 64, 116, 98, 112, 74, 90, 96},
       {131, 129, 135, 133, 143, 139, 137, 141, 147, 157, 159, 151, 155, 153, 145, 149},
       {189, 191, 163, 173, 165, 181, 179, 167, 187, 185, 161, 171, 183, 177, 169, 175},
       {217, 229, 233, 197, 251, 205, 237, 215, 213, 255, 243, 209, 201, 221, 207, 223},
       {199, 211, 253, 247, 249, 231, 239, 235, 195, 193, 245, 227, 241, 203, 219, 225}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg3eiXX_sew8_vlmul1) {
  VlxsegXeiXX<UInt8, 3, 1>(
      0x4500'8407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
      {{131, 0, 137, 6, 149, 143, 12, 18, 155, 42, 173, 161, 167, 36, 24, 30},
       {4, 129, 10, 135, 22, 16, 141, 147, 28, 171, 46, 34, 40, 165, 153, 159},
       {133, 2, 139, 8, 151, 145, 14, 20, 157, 44, 175, 163, 169, 38, 26, 32}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg3eiXX_sew8_vlmul2) {
  VlxsegXeiXX<UInt8, 3, 2>(
      0x4500'8407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
      {{131, 0, 137, 6, 149, 143, 12, 18, 155, 42, 173, 161, 167, 36, 24, 30},
       {90, 221, 179, 66, 54, 78, 203, 185, 215, 84, 48, 191, 209, 72, 60, 197},
       {4, 129, 10, 135, 22, 16, 141, 147, 28, 171, 46, 34, 40, 165, 153, 159},
       {219, 94, 52, 195, 183, 207, 76, 58, 88, 213, 177, 64, 82, 201, 189, 70},
       {133, 2, 139, 8, 151, 145, 14, 20, 157, 44, 175, 163, 169, 38, 26, 32},
       {92, 223, 181, 68, 56, 80, 205, 187, 217, 86, 50, 193, 211, 74, 62, 199}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg4eiXX_sew8_vlmul1) {
  VlxsegXeiXX<UInt8, 4, 1>(
      0x6500'8407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
      {{4, 0, 12, 8, 28, 20, 16, 24, 36, 56, 60, 44, 52, 48, 32, 40},
       {133, 129, 141, 137, 157, 149, 145, 153, 165, 185, 189, 173, 181, 177, 161, 169},
       {6, 2, 14, 10, 30, 22, 18, 26, 38, 58, 62, 46, 54, 50, 34, 42},
       {135, 131, 143, 139, 159, 151, 147, 155, 167, 187, 191, 175, 183, 179, 163, 171}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew4_vlmul2) {
  VlxsegXeiXX<UInt8, 4, 2>(
      0x6500'8407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
      {{4, 0, 12, 8, 28, 20, 16, 24, 36, 56, 60, 44, 52, 48, 32, 40},
       {120, 124, 68, 88, 72, 104, 100, 76, 116, 112, 64, 84, 108, 96, 80, 92},
       {133, 129, 141, 137, 157, 149, 145, 153, 165, 185, 189, 173, 181, 177, 161, 169},
       {249, 253, 197, 217, 201, 233, 229, 205, 245, 241, 193, 213, 237, 225, 209, 221},
       {6, 2, 14, 10, 30, 22, 18, 26, 38, 58, 62, 46, 54, 50, 34, 42},
       {122, 126, 70, 90, 74, 106, 102, 78, 118, 114, 66, 86, 110, 98, 82, 94},
       {135, 131, 143, 139, 159, 151, 147, 155, 167, 187, 191, 175, 183, 179, 163, 171},
       {251, 255, 199, 219, 203, 235, 231, 207, 247, 243, 195, 215, 239, 227, 211, 223}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg5eiXX_sew8) {
  VlxsegXeiXX<UInt8, 5, 1>(
      0x8500'8407,  // Vluxseg5ei8.v v8, (x1), v16, v0.t
      {{133, 0, 143, 10, 163, 153, 20, 30, 173, 70, 203, 183, 193, 60, 40, 50},
       {6, 129, 16, 139, 36, 26, 149, 159, 46, 199, 76, 56, 66, 189, 169, 179},
       {135, 2, 145, 12, 165, 155, 22, 32, 175, 72, 205, 185, 195, 62, 42, 52},
       {8, 131, 18, 141, 38, 28, 151, 161, 48, 201, 78, 58, 68, 191, 171, 181},
       {137, 4, 147, 14, 167, 157, 24, 34, 177, 74, 207, 187, 197, 64, 44, 54}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg6eiXX_sew8) {
  VlxsegXeiXX<UInt8, 6, 1>(
      0xa500'8407,  // Vluxseg6ei8.v v8, (x1), v16, v0.t
      {{6, 0, 18, 12, 42, 30, 24, 36, 54, 84, 90, 66, 78, 72, 48, 60},
       {135, 129, 147, 141, 171, 159, 153, 165, 183, 213, 219, 195, 207, 201, 177, 189},
       {8, 2, 20, 14, 44, 32, 26, 38, 56, 86, 92, 68, 80, 74, 50, 62},
       {137, 131, 149, 143, 173, 161, 155, 167, 185, 215, 221, 197, 209, 203, 179, 191},
       {10, 4, 22, 16, 46, 34, 28, 40, 58, 88, 94, 70, 82, 76, 52, 64},
       {139, 133, 151, 145, 175, 163, 157, 169, 187, 217, 223, 199, 211, 205, 181, 193}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg7eiXX_sew8) {
  VlxsegXeiXX<UInt8, 7, 1>(
      0xc500'8407,  // Vluxseg7ei8.v v8, (x1), v16, v0.t
      {{135, 0, 149, 14, 177, 163, 28, 42, 191, 98, 233, 205, 219, 84, 56, 70},
       {8, 129, 22, 143, 50, 36, 157, 171, 64, 227, 106, 78, 92, 213, 185, 199},
       {137, 2, 151, 16, 179, 165, 30, 44, 193, 100, 235, 207, 221, 86, 58, 72},
       {10, 131, 24, 145, 52, 38, 159, 173, 66, 229, 108, 80, 94, 215, 187, 201},
       {139, 4, 153, 18, 181, 167, 32, 46, 195, 102, 237, 209, 223, 88, 60, 74},
       {12, 133, 26, 147, 54, 40, 161, 175, 68, 231, 110, 82, 96, 217, 189, 203},
       {141, 6, 155, 20, 183, 169, 34, 48, 197, 104, 239, 211, 225, 90, 62, 76}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg8eiXX_sew8) {
  VlxsegXeiXX<UInt8, 8, 1>(
      0xe500'8407,  // Vluxseg8ei8.v v8, (x1), v16, v0.t
      {{8, 0, 24, 16, 56, 40, 32, 48, 72, 112, 120, 88, 104, 96, 64, 80},
       {137, 129, 153, 145, 185, 169, 161, 177, 201, 241, 249, 217, 233, 225, 193, 209},
       {10, 2, 26, 18, 58, 42, 34, 50, 74, 114, 122, 90, 106, 98, 66, 82},
       {139, 131, 155, 147, 187, 171, 163, 179, 203, 243, 251, 219, 235, 227, 195, 211},
       {12, 4, 28, 20, 60, 44, 36, 52, 76, 116, 124, 92, 108, 100, 68, 84},
       {141, 133, 157, 149, 189, 173, 165, 181, 205, 245, 253, 221, 237, 229, 197, 213},
       {14, 6, 30, 22, 62, 46, 38, 54, 78, 118, 126, 94, 110, 102, 70, 86},
       {143, 135, 159, 151, 191, 175, 167, 183, 207, 247, 255, 223, 239, 231, 199, 215}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew16_vlmul1) {
  VlxsegXeiXX<UInt16, 1, 1>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8302, 0x8100, 0x8706, 0x8504, 0x8f0e, 0x8b0a, 0x8908, 0x8d0c}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew16_vlmul2) {
  VlxsegXeiXX<UInt16, 1, 2>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8302, 0x8100, 0x8706, 0x8504, 0x8f0e, 0x8b0a, 0x8908, 0x8d0c},
                             {0x9312, 0x9d1c, 0x9f1e, 0x9716, 0x9b1a, 0x9918, 0x9110, 0x9514}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew16_vlmul4) {
  VlxsegXeiXX<UInt16, 1, 4>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8302, 0x8100, 0x8706, 0x8504, 0x8f0e, 0x8b0a, 0x8908, 0x8d0c},
                             {0x9312, 0x9d1c, 0x9f1e, 0x9716, 0x9b1a, 0x9918, 0x9110, 0x9514},
                             {0xbd3c, 0xbf3e, 0xa322, 0xad2c, 0xa524, 0xb534, 0xb332, 0xa726},
                             {0xbb3a, 0xb938, 0xa120, 0xab2a, 0xb736, 0xb130, 0xa928, 0xaf2e}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew16_vlmul8) {
  VlxsegXeiXX<UInt16, 1, 8>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8302, 0x8100, 0x8706, 0x8504, 0x8f0e, 0x8b0a, 0x8908, 0x8d0c},
                             {0x9312, 0x9d1c, 0x9f1e, 0x9716, 0x9b1a, 0x9918, 0x9110, 0x9514},
                             {0xbd3c, 0xbf3e, 0xa322, 0xad2c, 0xa524, 0xb534, 0xb332, 0xa726},
                             {0xbb3a, 0xb938, 0xa120, 0xab2a, 0xb736, 0xb130, 0xa928, 0xaf2e},
                             {0xd958, 0xe564, 0xe968, 0xc544, 0xfb7a, 0xcd4c, 0xed6c, 0xd756},
                             {0xd554, 0xff7e, 0xf372, 0xd150, 0xc948, 0xdd5c, 0xcf4e, 0xdf5e},
                             {0xc746, 0xd352, 0xfd7c, 0xf776, 0xf978, 0xe766, 0xef6e, 0xeb6a},
                             {0xc342, 0xc140, 0xf574, 0xe362, 0xf170, 0xcb4a, 0xdb5a, 0xe160}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew16_vlmul1) {
  VlxsegXeiXX<UInt16, 2, 1>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8504, 0x8100, 0x8d0c, 0x8908, 0x9d1c, 0x9514, 0x9110, 0x9918},
                             {0x8706, 0x8302, 0x8f0e, 0x8b0a, 0x9f1e, 0x9716, 0x9312, 0x9b1a}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew16_vlmul2) {
  VlxsegXeiXX<UInt16, 2, 2>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8504, 0x8100, 0x8d0c, 0x8908, 0x9d1c, 0x9514, 0x9110, 0x9918},
                             {0xa524, 0xb938, 0xbd3c, 0xad2c, 0xb534, 0xb130, 0xa120, 0xa928},
                             {0x8706, 0x8302, 0x8f0e, 0x8b0a, 0x9f1e, 0x9716, 0x9312, 0x9b1a},
                             {0xa726, 0xbb3a, 0xbf3e, 0xaf2e, 0xb736, 0xb332, 0xa322, 0xab2a}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew16_vlmul4) {
  VlxsegXeiXX<UInt16, 2, 4>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8504, 0x8100, 0x8d0c, 0x8908, 0x9d1c, 0x9514, 0x9110, 0x9918},
                             {0xa524, 0xb938, 0xbd3c, 0xad2c, 0xb534, 0xb130, 0xa120, 0xa928},
                             {0xf978, 0xfd7c, 0xc544, 0xd958, 0xc948, 0xe968, 0xe564, 0xcd4c},
                             {0xf574, 0xf170, 0xc140, 0xd554, 0xed6c, 0xe160, 0xd150, 0xdd5c},
                             {0x8706, 0x8302, 0x8f0e, 0x8b0a, 0x9f1e, 0x9716, 0x9312, 0x9b1a},
                             {0xa726, 0xbb3a, 0xbf3e, 0xaf2e, 0xb736, 0xb332, 0xa322, 0xab2a},
                             {0xfb7a, 0xff7e, 0xc746, 0xdb5a, 0xcb4a, 0xeb6a, 0xe766, 0xcf4e},
                             {0xf776, 0xf372, 0xc342, 0xd756, 0xef6e, 0xe362, 0xd352, 0xdf5e}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg3eiXX_sew16_vlmul1) {
  VlxsegXeiXX<UInt16, 3, 1>(0x4500'8407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x8706, 0x8100, 0x9312, 0x8d0c, 0xab2a, 0x9f1e, 0x9918, 0xa524},
                             {0x8908, 0x8302, 0x9514, 0x8f0e, 0xad2c, 0xa120, 0x9b1a, 0xa726},
                             {0x8b0a, 0x8504, 0x9716, 0x9110, 0xaf2e, 0xa322, 0x9d1c, 0xa928}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg3eiXX_sew16_vlmul2) {
  VlxsegXeiXX<UInt16, 3, 2>(0x4500'8407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x8706, 0x8100, 0x9312, 0x8d0c, 0xab2a, 0x9f1e, 0x9918, 0xa524},
                             {0xb736, 0xd554, 0xdb5a, 0xc342, 0xcf4e, 0xc948, 0xb130, 0xbd3c},
                             {0x8908, 0x8302, 0x9514, 0x8f0e, 0xad2c, 0xa120, 0x9b1a, 0xa726},
                             {0xb938, 0xd756, 0xdd5c, 0xc544, 0xd150, 0xcb4a, 0xb332, 0xbf3e},
                             {0x8b0a, 0x8504, 0x9716, 0x9110, 0xaf2e, 0xa322, 0x9d1c, 0xa928},
                             {0xbb3a, 0xd958, 0xdf5e, 0xc746, 0xd352, 0xcd4c, 0xb534, 0xc140}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg4eiXX_sew16_vlmul1) {
  VlxsegXeiXX<UInt16, 4, 1>(0x6500'8407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0x8908, 0x8100, 0x9918, 0x9110, 0xb938, 0xa928, 0xa120, 0xb130},
                             {0x8b0a, 0x8302, 0x9b1a, 0x9312, 0xbb3a, 0xab2a, 0xa322, 0xb332},
                             {0x8d0c, 0x8504, 0x9d1c, 0x9514, 0xbd3c, 0xad2c, 0xa524, 0xb534},
                             {0x8f0e, 0x8706, 0x9f1e, 0x9716, 0xbf3e, 0xaf2e, 0xa726, 0xb736}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg4eiXX_sew16_vlmul2) {
  VlxsegXeiXX<UInt16, 4, 2>(0x6500'8407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0x8908, 0x8100, 0x9918, 0x9110, 0xb938, 0xa928, 0xa120, 0xb130},
                             {0xc948, 0xf170, 0xf978, 0xd958, 0xe968, 0xe160, 0xc140, 0xd150},
                             {0x8b0a, 0x8302, 0x9b1a, 0x9312, 0xbb3a, 0xab2a, 0xa322, 0xb332},
                             {0xcb4a, 0xf372, 0xfb7a, 0xdb5a, 0xeb6a, 0xe362, 0xc342, 0xd352},
                             {0x8d0c, 0x8504, 0x9d1c, 0x9514, 0xbd3c, 0xad2c, 0xa524, 0xb534},
                             {0xcd4c, 0xf574, 0xfd7c, 0xdd5c, 0xed6c, 0xe564, 0xc544, 0xd554},
                             {0x8f0e, 0x8706, 0x9f1e, 0x9716, 0xbf3e, 0xaf2e, 0xa726, 0xb736},
                             {0xcf4e, 0xf776, 0xff7e, 0xdf5e, 0xef6e, 0xe766, 0xc746, 0xd756}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg5eiXX_sew16) {
  VlxsegXeiXX<UInt16, 5, 1>(0x8500'8407,  // Vluxseg5ei8.v v8, (x1), v16, v0.t
                            {{0x8b0a, 0x8100, 0x9f1e, 0x9514, 0xc746, 0xb332, 0xa928, 0xbd3c},
                             {0x8d0c, 0x8302, 0xa120, 0x9716, 0xc948, 0xb534, 0xab2a, 0xbf3e},
                             {0x8f0e, 0x8504, 0xa322, 0x9918, 0xcb4a, 0xb736, 0xad2c, 0xc140},
                             {0x9110, 0x8706, 0xa524, 0x9b1a, 0xcd4c, 0xb938, 0xaf2e, 0xc342},
                             {0x9312, 0x8908, 0xa726, 0x9d1c, 0xcf4e, 0xbb3a, 0xb130, 0xc544}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg6eiXX_sew16) {
  VlxsegXeiXX<UInt16, 6, 1>(0xa500'8407,  // Vluxseg6ei8.v v8, (x1), v16, v0.t
                            {{0x8d0c, 0x8100, 0xa524, 0x9918, 0xd554, 0xbd3c, 0xb130, 0xc948},
                             {0x8f0e, 0x8302, 0xa726, 0x9b1a, 0xd756, 0xbf3e, 0xb332, 0xcb4a},
                             {0x9110, 0x8504, 0xa928, 0x9d1c, 0xd958, 0xc140, 0xb534, 0xcd4c},
                             {0x9312, 0x8706, 0xab2a, 0x9f1e, 0xdb5a, 0xc342, 0xb736, 0xcf4e},
                             {0x9514, 0x8908, 0xad2c, 0xa120, 0xdd5c, 0xc544, 0xb938, 0xd150},
                             {0x9716, 0x8b0a, 0xaf2e, 0xa322, 0xdf5e, 0xc746, 0xbb3a, 0xd352}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg7eiXX_sew16) {
  VlxsegXeiXX<UInt16, 7, 1>(0xc500'8407,  // Vluxseg7ei8.v v8, (x1), v16, v0.t
                            {{0x8f0e, 0x8100, 0xab2a, 0x9d1c, 0xe362, 0xc746, 0xb938, 0xd554},
                             {0x9110, 0x8302, 0xad2c, 0x9f1e, 0xe564, 0xc948, 0xbb3a, 0xd756},
                             {0x9312, 0x8504, 0xaf2e, 0xa120, 0xe766, 0xcb4a, 0xbd3c, 0xd958},
                             {0x9514, 0x8706, 0xb130, 0xa322, 0xe968, 0xcd4c, 0xbf3e, 0xdb5a},
                             {0x9716, 0x8908, 0xb332, 0xa524, 0xeb6a, 0xcf4e, 0xc140, 0xdd5c},
                             {0x9918, 0x8b0a, 0xb534, 0xa726, 0xed6c, 0xd150, 0xc342, 0xdf5e},
                             {0x9b1a, 0x8d0c, 0xb736, 0xa928, 0xef6e, 0xd352, 0xc544, 0xe160}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg8eiXX_sew16) {
  VlxsegXeiXX<UInt16, 8, 1>(0xe500'8407,  // Vluxseg8ei8.v v8, (x1), v16, v0.t
                            {{0x9110, 0x8100, 0xb130, 0xa120, 0xf170, 0xd150, 0xc140, 0xe160},
                             {0x9312, 0x8302, 0xb332, 0xa322, 0xf372, 0xd352, 0xc342, 0xe362},
                             {0x9514, 0x8504, 0xb534, 0xa524, 0xf574, 0xd554, 0xc544, 0xe564},
                             {0x9716, 0x8706, 0xb736, 0xa726, 0xf776, 0xd756, 0xc746, 0xe766},
                             {0x9918, 0x8908, 0xb938, 0xa928, 0xf978, 0xd958, 0xc948, 0xe968},
                             {0x9b1a, 0x8b0a, 0xbb3a, 0xab2a, 0xfb7a, 0xdb5a, 0xcb4a, 0xeb6a},
                             {0x9d1c, 0x8d0c, 0xbd3c, 0xad2c, 0xfd7c, 0xdd5c, 0xcd4c, 0xed6c},
                             {0x9f1e, 0x8f0e, 0xbf3e, 0xaf2e, 0xff7e, 0xdf5e, 0xcf4e, 0xef6e}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew32_vlmul1) {
  VlxsegXeiXX<UInt32, 1, 1>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8706'8504, 0x8302'8100, 0x8f0e'8d0c, 0x8b0a'8908}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew32_vlmul2) {
  VlxsegXeiXX<UInt32, 1, 2>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8706'8504, 0x8302'8100, 0x8f0e'8d0c, 0x8b0a'8908},
                             {0x9f1e'9d1c, 0x9716'9514, 0x9312'9110, 0x9b1a'9918}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew32_vlmul4) {
  VlxsegXeiXX<UInt32, 1, 4>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8706'8504, 0x8302'8100, 0x8f0e'8d0c, 0x8b0a'8908},
                             {0x9f1e'9d1c, 0x9716'9514, 0x9312'9110, 0x9b1a'9918},
                             {0xa726'a524, 0xbb3a'b938, 0xbf3e'bd3c, 0xaf2e'ad2c},
                             {0xb736'b534, 0xb332'b130, 0xa322'a120, 0xab2a'a928}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew32_vlmul8) {
  VlxsegXeiXX<UInt32, 1, 8>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8706'8504, 0x8302'8100, 0x8f0e'8d0c, 0x8b0a'8908},
                             {0x9f1e'9d1c, 0x9716'9514, 0x9312'9110, 0x9b1a'9918},
                             {0xa726'a524, 0xbb3a'b938, 0xbf3e'bd3c, 0xaf2e'ad2c},
                             {0xb736'b534, 0xb332'b130, 0xa322'a120, 0xab2a'a928},
                             {0xfb7a'f978, 0xff7e'fd7c, 0xc746'c544, 0xdb5a'd958},
                             {0xcb4a'c948, 0xeb6a'e968, 0xe766'e564, 0xcf4e'cd4c},
                             {0xf776'f574, 0xf372'f170, 0xc342'c140, 0xd756'd554},
                             {0xef6e'ed6c, 0xe362'e160, 0xd352'd150, 0xdf5e'dd5c}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew32_vlmul1) {
  VlxsegXeiXX<UInt32, 2, 1>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8b0a'8908, 0x8302'8100, 0x9b1a'9918, 0x9312'9110},
                             {0x8f0e'8d0c, 0x8706'8504, 0x9f1e'9d1c, 0x9716'9514}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew32_vlmul2) {
  VlxsegXeiXX<UInt32, 2, 2>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8b0a'8908, 0x8302'8100, 0x9b1a'9918, 0x9312'9110},
                             {0xbb3a'b938, 0xab2a'a928, 0xa322'a120, 0xb332'b130},
                             {0x8f0e'8d0c, 0x8706'8504, 0x9f1e'9d1c, 0x9716'9514},
                             {0xbf3e'bd3c, 0xaf2e'ad2c, 0xa726'a524, 0xb736'b534}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew32_vlmul4) {
  VlxsegXeiXX<UInt32, 2, 4>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8b0a'8908, 0x8302'8100, 0x9b1a'9918, 0x9312'9110},
                             {0xbb3a'b938, 0xab2a'a928, 0xa322'a120, 0xb332'b130},
                             {0xcb4a'c948, 0xf372'f170, 0xfb7a'f978, 0xdb5a'd958},
                             {0xeb6a'e968, 0xe362'e160, 0xc342'c140, 0xd352'd150},
                             {0x8f0e'8d0c, 0x8706'8504, 0x9f1e'9d1c, 0x9716'9514},
                             {0xbf3e'bd3c, 0xaf2e'ad2c, 0xa726'a524, 0xb736'b534},
                             {0xcf4e'cd4c, 0xf776'f574, 0xff7e'fd7c, 0xdf5e'dd5c},
                             {0xef6e'ed6c, 0xe766'e564, 0xc746'c544, 0xd756'd554}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg3eiXX_sew32_vlmul1) {
  VlxsegXeiXX<UInt32, 3, 1>(0x4500'8407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c, 0x8302'8100, 0xa726'a524, 0x9b1a'9918},
                             {0x9312'9110, 0x8706'8504, 0xab2a'a928, 0x9f1e'9d1c},
                             {0x9716'9514, 0x8b0a'8908, 0xaf2e'ad2c, 0xa322'a120}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg3eiXX_sew32_vlmul2) {
  VlxsegXeiXX<UInt32, 3, 2>(0x4500'8407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c, 0x8302'8100, 0xa726'a524, 0x9b1a'9918},
                             {0xd756'd554, 0xbf3e'bd3c, 0xb332'b130, 0xcb4a'c948},
                             {0x9312'9110, 0x8706'8504, 0xab2a'a928, 0x9f1e'9d1c},
                             {0xdb5a'd958, 0xc342'c140, 0xb736'b534, 0xcf4e'cd4c},
                             {0x9716'9514, 0x8b0a'8908, 0xaf2e'ad2c, 0xa322'a120},
                             {0xdf5e'dd5c, 0xc746'c544, 0xbb3a'b938, 0xd352'd150}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg4eiXX_sew32_vlmul1) {
  VlxsegXeiXX<UInt32, 4, 1>(0x6500'8407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0x9312'9110, 0x8302'8100, 0xb332'b130, 0xa322'a120},
                             {0x9716'9514, 0x8706'8504, 0xb736'b534, 0xa726'a524},
                             {0x9b1a'9918, 0x8b0a'8908, 0xbb3a'b938, 0xab2a'a928},
                             {0x9f1e'9d1c, 0x8f0e'8d0c, 0xbf3e'bd3c, 0xaf2e'ad2c}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg4eiXX_sew32_vlmul2) {
  VlxsegXeiXX<UInt32, 4, 2>(0x6500'8407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0x9312'9110, 0x8302'8100, 0xb332'b130, 0xa322'a120},
                             {0xf372'f170, 0xd352'd150, 0xc342'c140, 0xe362'e160},
                             {0x9716'9514, 0x8706'8504, 0xb736'b534, 0xa726'a524},
                             {0xf776'f574, 0xd756'd554, 0xc746'c544, 0xe766'e564},
                             {0x9b1a'9918, 0x8b0a'8908, 0xbb3a'b938, 0xab2a'a928},
                             {0xfb7a'f978, 0xdb5a'd958, 0xcb4a'c948, 0xeb6a'e968},
                             {0x9f1e'9d1c, 0x8f0e'8d0c, 0xbf3e'bd3c, 0xaf2e'ad2c},
                             {0xff7e'fd7c, 0xdf5e'dd5c, 0xcf4e'cd4c, 0xef6e'ed6c}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg5eiXX_sew32) {
  VlxsegXeiXX<UInt32, 5, 1>(0x8500'8407,  // Vluxseg5ei8.v v8, (x1), v16, v0.t
                            {{0x9716'9514, 0x8302'8100, 0xbf3e'bd3c, 0xab2a'a928},
                             {0x9b1a'9918, 0x8706'8504, 0xc342'c140, 0xaf2e'ad2c},
                             {0x9f1e'9d1c, 0x8b0a'8908, 0xc746'c544, 0xb332'b130},
                             {0xa322'a120, 0x8f0e'8d0c, 0xcb4a'c948, 0xb736'b534},
                             {0xa726'a524, 0x9312'9110, 0xcf4e'cd4c, 0xbb3a'b938}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg6eiXX_sew32) {
  VlxsegXeiXX<UInt32, 6, 1>(0xa500'8407,  // Vluxseg6ei8.v v8, (x1), v16, v0.t
                            {{0x9b1a'9918, 0x8302'8100, 0xcb4a'c948, 0xb332'b130},
                             {0x9f1e'9d1c, 0x8706'8504, 0xcf4e'cd4c, 0xb736'b534},
                             {0xa322'a120, 0x8b0a'8908, 0xd352'd150, 0xbb3a'b938},
                             {0xa726'a524, 0x8f0e'8d0c, 0xd756'd554, 0xbf3e'bd3c},
                             {0xab2a'a928, 0x9312'9110, 0xdb5a'd958, 0xc342'c140},
                             {0xaf2e'ad2c, 0x9716'9514, 0xdf5e'dd5c, 0xc746'c544}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg7eiXX_sew32) {
  VlxsegXeiXX<UInt32, 7, 1>(0xc500'8407,  // Vluxseg7ei8.v v8, (x1), v16, v0.t
                            {{0x9f1e'9d1c, 0x8302'8100, 0xd756'd554, 0xbb3a'b938},
                             {0xa322'a120, 0x8706'8504, 0xdb5a'd958, 0xbf3e'bd3c},
                             {0xa726'a524, 0x8b0a'8908, 0xdf5e'dd5c, 0xc342'c140},
                             {0xab2a'a928, 0x8f0e'8d0c, 0xe362'e160, 0xc746'c544},
                             {0xaf2e'ad2c, 0x9312'9110, 0xe766'e564, 0xcb4a'c948},
                             {0xb332'b130, 0x9716'9514, 0xeb6a'e968, 0xcf4e'cd4c},
                             {0xb736'b534, 0x9b1a'9918, 0xef6e'ed6c, 0xd352'd150}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg8eiXX_sew32) {
  VlxsegXeiXX<UInt32, 8, 1>(0xe500'8407,  // Vluxseg8ei8.v v8, (x1), v16, v0.t
                            {{0xa322'a120, 0x8302'8100, 0xe362'e160, 0xc342'c140},
                             {0xa726'a524, 0x8706'8504, 0xe766'e564, 0xc746'c544},
                             {0xab2a'a928, 0x8b0a'8908, 0xeb6a'e968, 0xcb4a'c948},
                             {0xaf2e'ad2c, 0x8f0e'8d0c, 0xef6e'ed6c, 0xcf4e'cd4c},
                             {0xb332'b130, 0x9312'9110, 0xf372'f170, 0xd352'd150},
                             {0xb736'b534, 0x9716'9514, 0xf776'f574, 0xd756'd554},
                             {0xbb3a'b938, 0x9b1a'9918, 0xfb7a'f978, 0xdb5a'd958},
                             {0xbf3e'bd3c, 0x9f1e'9d1c, 0xff7e'fd7c, 0xdf5e'dd5c}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew64_vlmul1) {
  VlxsegXeiXX<UInt64, 1, 1>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew64_vlmul2) {
  VlxsegXeiXX<UInt64, 1, 2>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100},
                             {0x9f1e'9d1c'9b1a'9918, 0x9716'9514'9312'9110}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew64_vlmul4) {
  VlxsegXeiXX<UInt64, 1, 4>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100},
                             {0x9f1e'9d1c'9b1a'9918, 0x9716'9514'9312'9110},
                             {0xbf3e'bd3c'bb3a'b938, 0xaf2e'ad2c'ab2a'a928},
                             {0xa726'a524'a322'a120, 0xb736'b534'b332'b130}});
}

TEST_F(Riscv64InterpreterTest, TestVlxeiXX_sew64_vlmul8) {
  VlxsegXeiXX<UInt64, 1, 8>(0x0500'8407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100},
                             {0x9f1e'9d1c'9b1a'9918, 0x9716'9514'9312'9110},
                             {0xbf3e'bd3c'bb3a'b938, 0xaf2e'ad2c'ab2a'a928},
                             {0xa726'a524'a322'a120, 0xb736'b534'b332'b130},
                             {0xcf4e'cd4c'cb4a'c948, 0xf776'f574'f372'f170},
                             {0xff7e'fd7c'fb7a'f978, 0xdf5e'dd5c'db5a'd958},
                             {0xef6e'ed6c'eb6a'e968, 0xe766'e564'e362'e160},
                             {0xc746'c544'c342'c140, 0xd756'd554'd352'd150}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew64_vlmul1) {
  VlxsegXeiXX<UInt64, 2, 1>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x9716'9514'9312'9110, 0x8706'8504'8302'8100},
                             {0x9f1e'9d1c'9b1a'9918, 0x8f0e'8d0c'8b0a'8908}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew64_vlmul2) {
  VlxsegXeiXX<UInt64, 2, 2>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x9716'9514'9312'9110, 0x8706'8504'8302'8100},
                             {0xb736'b534'b332'b130, 0xa726'a524'a322'a120},
                             {0x9f1e'9d1c'9b1a'9918, 0x8f0e'8d0c'8b0a'8908},
                             {0xbf3e'bd3c'bb3a'b938, 0xaf2e'ad2c'ab2a'a928}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg2eiXX_sew64_vlmul4) {
  VlxsegXeiXX<UInt64, 2, 4>(0x2500'8407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x9716'9514'9312'9110, 0x8706'8504'8302'8100},
                             {0xb736'b534'b332'b130, 0xa726'a524'a322'a120},
                             {0xf776'f574'f372'f170, 0xd756'd554'd352'd150},
                             {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                             {0x9f1e'9d1c'9b1a'9918, 0x8f0e'8d0c'8b0a'8908},
                             {0xbf3e'bd3c'bb3a'b938, 0xaf2e'ad2c'ab2a'a928},
                             {0xff7e'fd7c'fb7a'f978, 0xdf5e'dd5c'db5a'd958},
                             {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg3eiXX_sew64_vlmul1) {
  VlxsegXeiXX<UInt64, 3, 1>(0x4500'8407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x9f1e'9d1c'9b1a'9918, 0x8706'8504'8302'8100},
                             {0xa726'a524'a322'a120, 0x8f0e'8d0c'8b0a'8908},
                             {0xaf2e'ad2c'ab2a'a928, 0x9716'9514'9312'9110}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg3eiXX_sew64_vlmul2) {
  VlxsegXeiXX<UInt64, 3, 2>(0x4500'8407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x9f1e'9d1c'9b1a'9918, 0x8706'8504'8302'8100},
                             {0xcf4e'cd4c'cb4a'c948, 0xb736'b534'b332'b130},
                             {0xa726'a524'a322'a120, 0x8f0e'8d0c'8b0a'8908},
                             {0xd756'd554'd352'd150, 0xbf3e'bd3c'bb3a'b938},
                             {0xaf2e'ad2c'ab2a'a928, 0x9716'9514'9312'9110},
                             {0xdf5e'dd5c'db5a'd958, 0xc746'c544'c342'c140}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg4eiXX_sew64_vlmul1) {
  VlxsegXeiXX<UInt64, 4, 1>(0x6500'8407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0xa726'a524'a322'a120, 0x8706'8504'8302'8100},
                             {0xaf2e'ad2c'ab2a'a928, 0x8f0e'8d0c'8b0a'8908},
                             {0xb736'b534'b332'b130, 0x9716'9514'9312'9110},
                             {0xbf3e'bd3c'bb3a'b938, 0x9f1e'9d1c'9b1a'9918}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg4eiXX_sew64_vlmul2) {
  VlxsegXeiXX<UInt64, 4, 2>(0x6500'8407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0xa726'a524'a322'a120, 0x8706'8504'8302'8100},
                             {0xe766'e564'e362'e160, 0xc746'c544'c342'c140},
                             {0xaf2e'ad2c'ab2a'a928, 0x8f0e'8d0c'8b0a'8908},
                             {0xef6e'ed6c'eb6a'e968, 0xcf4e'cd4c'cb4a'c948},
                             {0xb736'b534'b332'b130, 0x9716'9514'9312'9110},
                             {0xf776'f574'f372'f170, 0xd756'd554'd352'd150},
                             {0xbf3e'bd3c'bb3a'b938, 0x9f1e'9d1c'9b1a'9918},
                             {0xff7e'fd7c'fb7a'f978, 0xdf5e'dd5c'db5a'd958}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg5eiXX_sew64) {
  VlxsegXeiXX<UInt64, 5, 1>(0x8500'8407,  // Vluxseg5ei8.v v8, (x1), v16, v0.t
                            {{0xaf2e'ad2c'ab2a'a928, 0x8706'8504'8302'8100},
                             {0xb736'b534'b332'b130, 0x8f0e'8d0c'8b0a'8908},
                             {0xbf3e'bd3c'bb3a'b938, 0x9716'9514'9312'9110},
                             {0xc746'c544'c342'c140, 0x9f1e'9d1c'9b1a'9918},
                             {0xcf4e'cd4c'cb4a'c948, 0xa726'a524'a322'a120}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg6eiXX_sew64) {
  VlxsegXeiXX<UInt64, 6, 1>(0xa500'8407,  // Vluxseg6ei8.v v8, (x1), v16, v0.t
                            {{0xb736'b534'b332'b130, 0x8706'8504'8302'8100},
                             {0xbf3e'bd3c'bb3a'b938, 0x8f0e'8d0c'8b0a'8908},
                             {0xc746'c544'c342'c140, 0x9716'9514'9312'9110},
                             {0xcf4e'cd4c'cb4a'c948, 0x9f1e'9d1c'9b1a'9918},
                             {0xd756'd554'd352'd150, 0xa726'a524'a322'a120},
                             {0xdf5e'dd5c'db5a'd958, 0xaf2e'ad2c'ab2a'a928}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg7eiXX_sew64) {
  VlxsegXeiXX<UInt64, 7, 1>(0xc500'8407,  // Vluxseg7ei8.v v8, (x1), v16, v0.t
                            {{0xbf3e'bd3c'bb3a'b938, 0x8706'8504'8302'8100},
                             {0xc746'c544'c342'c140, 0x8f0e'8d0c'8b0a'8908},
                             {0xcf4e'cd4c'cb4a'c948, 0x9716'9514'9312'9110},
                             {0xd756'd554'd352'd150, 0x9f1e'9d1c'9b1a'9918},
                             {0xdf5e'dd5c'db5a'd958, 0xa726'a524'a322'a120},
                             {0xe766'e564'e362'e160, 0xaf2e'ad2c'ab2a'a928},
                             {0xef6e'ed6c'eb6a'e968, 0xb736'b534'b332'b130}});
}

TEST_F(Riscv64InterpreterTest, TestVlxseg8eiXX_sew64) {
  VlxsegXeiXX<UInt64, 8, 1>(0xe500'8407,  // Vluxseg8ei8.v v8, (x1), v16, v0.t
                            {{0xc746'c544'c342'c140, 0x8706'8504'8302'8100},
                             {0xcf4e'cd4c'cb4a'c948, 0x8f0e'8d0c'8b0a'8908},
                             {0xd756'd554'd352'd150, 0x9716'9514'9312'9110},
                             {0xdf5e'dd5c'db5a'd958, 0x9f1e'9d1c'9b1a'9918},
                             {0xe766'e564'e362'e160, 0xa726'a524'a322'a120},
                             {0xef6e'ed6c'eb6a'e968, 0xaf2e'ad2c'ab2a'a928},
                             {0xf776'f574'f372'f170, 0xb736'b534'b332'b130},
                             {0xff7e'fd7c'fb7a'f978, 0xbf3e'bd3c'bb3a'b938}});
}

TEST_F(Riscv64InterpreterTest, TestVle8_vlmul1) {
  TestVlsegXeXX<UInt8, 1, 1>(0x0000'8407,  // vlse8.v v8, (x1), v0.t
                             {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143}});
}

TEST_F(Riscv64InterpreterTest, TestVle8_vlmul2) {
  TestVlsegXeXX<UInt8, 1, 2>(
      0x0000'8407,  // vlse8.v v8, (x1), v0.t
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159}});
}

TEST_F(Riscv64InterpreterTest, TestVle8_vlmul4) {
  TestVlsegXeXX<UInt8, 1, 4>(
      0x0000'8407,  // vlse8.v v8, (x1), v0.t
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
       {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
       {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191}});
}

TEST_F(Riscv64InterpreterTest, TestVle8_vlmul8) {
  TestVlsegXeXX<UInt8, 1, 8>(
      0x0000'8407,  // vlse8.v v8, (x1), v0.t
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
       {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
       {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
       {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
       {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
       {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
       {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e8_vlmul1) {
  TestVlsegXeXX<UInt8, 2, 1>(
      0x2000'8407,  // vlseg2e8.v v8, (x1), v0.t
      {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
       {129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e8_vlmul2) {
  TestVlsegXeXX<UInt8, 2, 2>(
      0x2000'8407,  // vlseg2e8.v v8, (x1), v0.t
      {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
       {129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159},
       {161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e8_vlmul4) {
  TestVlsegXeXX<UInt8, 2, 4>(
      0x2000'8407,  // vlseg2e8.v v8, (x1), v0.t
      {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126},
       {129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159},
       {161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191},
       {193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223},
       {225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg3e8_vlmul1) {
  TestVlsegXeXX<UInt8, 3, 1>(
      0x4000'8407,  // vlseg3e8.v v8, (x1), v0.t
      {{0, 131, 6, 137, 12, 143, 18, 149, 24, 155, 30, 161, 36, 167, 42, 173},
       {129, 4, 135, 10, 141, 16, 147, 22, 153, 28, 159, 34, 165, 40, 171, 46},
       {2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32, 163, 38, 169, 44, 175}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg3e8_vlmul2) {
  TestVlsegXeXX<UInt8, 3, 2>(
      0x4000'8407,  // vlseg3e8.v v8, (x1), v0.t
      {{0, 131, 6, 137, 12, 143, 18, 149, 24, 155, 30, 161, 36, 167, 42, 173},
       {48, 179, 54, 185, 60, 191, 66, 197, 72, 203, 78, 209, 84, 215, 90, 221},
       {129, 4, 135, 10, 141, 16, 147, 22, 153, 28, 159, 34, 165, 40, 171, 46},
       {177, 52, 183, 58, 189, 64, 195, 70, 201, 76, 207, 82, 213, 88, 219, 94},
       {2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32, 163, 38, 169, 44, 175},
       {50, 181, 56, 187, 62, 193, 68, 199, 74, 205, 80, 211, 86, 217, 92, 223}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg4e8_vlmul1) {
  TestVlsegXeXX<UInt8, 4, 1>(
      0x6000'8407,  // vlseg4e8.v v8, (x1), v0.t
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg4e8_vlmul2) {
  TestVlsegXeXX<UInt8, 4, 2>(
      0x6000'8407,  // vlseg4e8.v v8, (x1), v0.t
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126},
       {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191},
       {195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg5e8) {
  TestVlsegXeXX<UInt8, 5, 1>(
      0x8000'8407,  // vlseg5e8.v v8, (x1), v0.t
      {{0, 133, 10, 143, 20, 153, 30, 163, 40, 173, 50, 183, 60, 193, 70, 203},
       {129, 6, 139, 16, 149, 26, 159, 36, 169, 46, 179, 56, 189, 66, 199, 76},
       {2, 135, 12, 145, 22, 155, 32, 165, 42, 175, 52, 185, 62, 195, 72, 205},
       {131, 8, 141, 18, 151, 28, 161, 38, 171, 48, 181, 58, 191, 68, 201, 78},
       {4, 137, 14, 147, 24, 157, 34, 167, 44, 177, 54, 187, 64, 197, 74, 207}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg6e8) {
  TestVlsegXeXX<UInt8, 6, 1>(
      0xa000'8407,  // vlseg6e8.v v8, (x1), v0.t
      {{0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90},
       {129, 135, 141, 147, 153, 159, 165, 171, 177, 183, 189, 195, 201, 207, 213, 219},
       {2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92},
       {131, 137, 143, 149, 155, 161, 167, 173, 179, 185, 191, 197, 203, 209, 215, 221},
       {4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94},
       {133, 139, 145, 151, 157, 163, 169, 175, 181, 187, 193, 199, 205, 211, 217, 223}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg7e8) {
  TestVlsegXeXX<UInt8, 7, 1>(
      0xc000'8407,  // vlseg7e8.v v8, (x1), v0.t
      {{0, 135, 14, 149, 28, 163, 42, 177, 56, 191, 70, 205, 84, 219, 98, 233},
       {129, 8, 143, 22, 157, 36, 171, 50, 185, 64, 199, 78, 213, 92, 227, 106},
       {2, 137, 16, 151, 30, 165, 44, 179, 58, 193, 72, 207, 86, 221, 100, 235},
       {131, 10, 145, 24, 159, 38, 173, 52, 187, 66, 201, 80, 215, 94, 229, 108},
       {4, 139, 18, 153, 32, 167, 46, 181, 60, 195, 74, 209, 88, 223, 102, 237},
       {133, 12, 147, 26, 161, 40, 175, 54, 189, 68, 203, 82, 217, 96, 231, 110},
       {6, 141, 20, 155, 34, 169, 48, 183, 62, 197, 76, 211, 90, 225, 104, 239}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg8e8) {
  TestVlsegXeXX<UInt8, 8, 1>(
      0xe000'8407,  // vlseg8e8.v v8, (x1), v0.t
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124},
       {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253},
       {6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126},
       {135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255}});
}

TEST_F(Riscv64InterpreterTest, TestVle16_vlmul1) {
  TestVlsegXeXX<UInt16, 1, 1>(0x0000'd407,  // vle16.v v8, (x1), v0.t
                              {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e}});
}

TEST_F(Riscv64InterpreterTest, TestVle16_vlmul2) {
  TestVlsegXeXX<UInt16, 1, 2>(0x0000'd407,  // vle16.v v8, (x1), v0.t
                              {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
                               {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e}});
}

TEST_F(Riscv64InterpreterTest, TestVle16_vlmul4) {
  TestVlsegXeXX<UInt16, 1, 4>(0x0000'd407,  // vle16.v v8, (x1), v0.t
                              {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
                               {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
                               {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
                               {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e}});
}

TEST_F(Riscv64InterpreterTest, TestVle16_vlmul8) {
  TestVlsegXeXX<UInt16, 1, 8>(0x0000'd407,  // vle16.v v8, (x1), v0.t
                              {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
                               {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
                               {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
                               {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
                               {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
                               {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
                               {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
                               {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e16_vlmul1) {
  TestVlsegXeXX<UInt16, 2, 1>(0x2000'd407,  // vlseg2e16.v v8, (x1), v0.t
                              {{0x8100, 0x8504, 0x8908, 0x8d0c, 0x9110, 0x9514, 0x9918, 0x9d1c},
                               {0x8302, 0x8706, 0x8b0a, 0x8f0e, 0x9312, 0x9716, 0x9b1a, 0x9f1e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e16_vlmul2) {
  TestVlsegXeXX<UInt16, 2, 2>(0x2000'd407,  // vlseg2e16.v v8, (x1), v0.t
                              {{0x8100, 0x8504, 0x8908, 0x8d0c, 0x9110, 0x9514, 0x9918, 0x9d1c},
                               {0xa120, 0xa524, 0xa928, 0xad2c, 0xb130, 0xb534, 0xb938, 0xbd3c},
                               {0x8302, 0x8706, 0x8b0a, 0x8f0e, 0x9312, 0x9716, 0x9b1a, 0x9f1e},
                               {0xa322, 0xa726, 0xab2a, 0xaf2e, 0xb332, 0xb736, 0xbb3a, 0xbf3e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e16_vlmul4) {
  TestVlsegXeXX<UInt16, 2, 4>(0x2000'd407,  // vlseg2e16.v v8, (x1), v0.t
                              {{0x8100, 0x8504, 0x8908, 0x8d0c, 0x9110, 0x9514, 0x9918, 0x9d1c},
                               {0xa120, 0xa524, 0xa928, 0xad2c, 0xb130, 0xb534, 0xb938, 0xbd3c},
                               {0xc140, 0xc544, 0xc948, 0xcd4c, 0xd150, 0xd554, 0xd958, 0xdd5c},
                               {0xe160, 0xe564, 0xe968, 0xed6c, 0xf170, 0xf574, 0xf978, 0xfd7c},
                               {0x8302, 0x8706, 0x8b0a, 0x8f0e, 0x9312, 0x9716, 0x9b1a, 0x9f1e},
                               {0xa322, 0xa726, 0xab2a, 0xaf2e, 0xb332, 0xb736, 0xbb3a, 0xbf3e},
                               {0xc342, 0xc746, 0xcb4a, 0xcf4e, 0xd352, 0xd756, 0xdb5a, 0xdf5e},
                               {0xe362, 0xe766, 0xeb6a, 0xef6e, 0xf372, 0xf776, 0xfb7a, 0xff7e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg3e16_vlmul1) {
  TestVlsegXeXX<UInt16, 3, 1>(0x4000'd407,  // vlseg3e16.v v8, (x1), v0.t
                              {{0x8100, 0x8706, 0x8d0c, 0x9312, 0x9918, 0x9f1e, 0xa524, 0xab2a},
                               {0x8302, 0x8908, 0x8f0e, 0x9514, 0x9b1a, 0xa120, 0xa726, 0xad2c},
                               {0x8504, 0x8b0a, 0x9110, 0x9716, 0x9d1c, 0xa322, 0xa928, 0xaf2e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg3e16_vlmul2) {
  TestVlsegXeXX<UInt16, 3, 2>(0x4000'd407,  // vlseg3e16.v v8, (x1), v0.t
                              {{0x8100, 0x8706, 0x8d0c, 0x9312, 0x9918, 0x9f1e, 0xa524, 0xab2a},
                               {0xb130, 0xb736, 0xbd3c, 0xc342, 0xc948, 0xcf4e, 0xd554, 0xdb5a},
                               {0x8302, 0x8908, 0x8f0e, 0x9514, 0x9b1a, 0xa120, 0xa726, 0xad2c},
                               {0xb332, 0xb938, 0xbf3e, 0xc544, 0xcb4a, 0xd150, 0xd756, 0xdd5c},
                               {0x8504, 0x8b0a, 0x9110, 0x9716, 0x9d1c, 0xa322, 0xa928, 0xaf2e},
                               {0xb534, 0xbb3a, 0xc140, 0xc746, 0xcd4c, 0xd352, 0xd958, 0xdf5e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg4e16_vlmul1) {
  TestVlsegXeXX<UInt16, 4, 1>(0x6000'd407,  // vlseg4e16.v v8, (x1), v0.t
                              {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                               {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                               {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                               {0x8706, 0x8f0e, 0x9716, 0x9f1e, 0xa726, 0xaf2e, 0xb736, 0xbf3e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg4e16_vlmul2) {
  TestVlsegXeXX<UInt16, 4, 2>(0x6000'd407,  // vlseg4e16.v v8, (x1), v0.t
                              {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                               {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                               {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                               {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a},
                               {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                               {0xc544, 0xcd4c, 0xd554, 0xdd5c, 0xe564, 0xed6c, 0xf574, 0xfd7c},
                               {0x8706, 0x8f0e, 0x9716, 0x9f1e, 0xa726, 0xaf2e, 0xb736, 0xbf3e},
                               {0xc746, 0xcf4e, 0xd756, 0xdf5e, 0xe766, 0xef6e, 0xf776, 0xff7e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg5e16) {
  TestVlsegXeXX<UInt16, 5, 1>(0x8000'd407,  // vlseg5e16.v v8, (x1), v0.t
                              {{0x8100, 0x8b0a, 0x9514, 0x9f1e, 0xa928, 0xb332, 0xbd3c, 0xc746},
                               {0x8302, 0x8d0c, 0x9716, 0xa120, 0xab2a, 0xb534, 0xbf3e, 0xc948},
                               {0x8504, 0x8f0e, 0x9918, 0xa322, 0xad2c, 0xb736, 0xc140, 0xcb4a},
                               {0x8706, 0x9110, 0x9b1a, 0xa524, 0xaf2e, 0xb938, 0xc342, 0xcd4c},
                               {0x8908, 0x9312, 0x9d1c, 0xa726, 0xb130, 0xbb3a, 0xc544, 0xcf4e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg6e16) {
  TestVlsegXeXX<UInt16, 6, 1>(0xa000'd407,  // vlseg6e16.v v8, (x1), v0.t
                              {{0x8100, 0x8d0c, 0x9918, 0xa524, 0xb130, 0xbd3c, 0xc948, 0xd554},
                               {0x8302, 0x8f0e, 0x9b1a, 0xa726, 0xb332, 0xbf3e, 0xcb4a, 0xd756},
                               {0x8504, 0x9110, 0x9d1c, 0xa928, 0xb534, 0xc140, 0xcd4c, 0xd958},
                               {0x8706, 0x9312, 0x9f1e, 0xab2a, 0xb736, 0xc342, 0xcf4e, 0xdb5a},
                               {0x8908, 0x9514, 0xa120, 0xad2c, 0xb938, 0xc544, 0xd150, 0xdd5c},
                               {0x8b0a, 0x9716, 0xa322, 0xaf2e, 0xbb3a, 0xc746, 0xd352, 0xdf5e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg7e16) {
  TestVlsegXeXX<UInt16, 7, 1>(0xc000'd407,  // vlseg7e16.v v8, (x1), v0.t
                              {{0x8100, 0x8f0e, 0x9d1c, 0xab2a, 0xb938, 0xc746, 0xd554, 0xe362},
                               {0x8302, 0x9110, 0x9f1e, 0xad2c, 0xbb3a, 0xc948, 0xd756, 0xe564},
                               {0x8504, 0x9312, 0xa120, 0xaf2e, 0xbd3c, 0xcb4a, 0xd958, 0xe766},
                               {0x8706, 0x9514, 0xa322, 0xb130, 0xbf3e, 0xcd4c, 0xdb5a, 0xe968},
                               {0x8908, 0x9716, 0xa524, 0xb332, 0xc140, 0xcf4e, 0xdd5c, 0xeb6a},
                               {0x8b0a, 0x9918, 0xa726, 0xb534, 0xc342, 0xd150, 0xdf5e, 0xed6c},
                               {0x8d0c, 0x9b1a, 0xa928, 0xb736, 0xc544, 0xd352, 0xe160, 0xef6e}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg8e16) {
  TestVlsegXeXX<UInt16, 8, 1>(0xe000'd407,  // vlseg8e16.v v8, (x1), v0.t
                              {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                               {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                               {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                               {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                               {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978},
                               {0x8b0a, 0x9b1a, 0xab2a, 0xbb3a, 0xcb4a, 0xdb5a, 0xeb6a, 0xfb7a},
                               {0x8d0c, 0x9d1c, 0xad2c, 0xbd3c, 0xcd4c, 0xdd5c, 0xed6c, 0xfd7c},
                               {0x8f0e, 0x9f1e, 0xaf2e, 0xbf3e, 0xcf4e, 0xdf5e, 0xef6e, 0xff7e}});
}

TEST_F(Riscv64InterpreterTest, TestVle32_vlmul1) {
  TestVlsegXeXX<UInt32, 1, 1>(0x0000'e407,  // vle32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c}});
}

TEST_F(Riscv64InterpreterTest, TestVle32_vlmul2) {
  TestVlsegXeXX<UInt32, 1, 2>(0x0000'e407,  // vle32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                               {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c}});
}

TEST_F(Riscv64InterpreterTest, TestVle32_vlmul4) {
  TestVlsegXeXX<UInt32, 1, 4>(0x0000'e407,  // vle32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                               {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
                               {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
                               {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c}});
}

TEST_F(Riscv64InterpreterTest, TestVle32_vlmul8) {
  TestVlsegXeXX<UInt32, 1, 8>(0x0000'e407,  // vle32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                               {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
                               {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
                               {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
                               {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                               {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                               {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                               {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e32_vlmul1) {
  TestVlsegXeXX<UInt32, 2, 1>(0x2000'e407,  // vlseg2e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8b0a'8908, 0x9312'9110, 0x9b1a'9918},
                               {0x8706'8504, 0x8f0e'8d0c, 0x9716'9514, 0x9f1e'9d1c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e32_vlmul2) {
  TestVlsegXeXX<UInt32, 2, 2>(0x2000'e407,  // vlseg2e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8b0a'8908, 0x9312'9110, 0x9b1a'9918},
                               {0xa322'a120, 0xab2a'a928, 0xb332'b130, 0xbb3a'b938},
                               {0x8706'8504, 0x8f0e'8d0c, 0x9716'9514, 0x9f1e'9d1c},
                               {0xa726'a524, 0xaf2e'ad2c, 0xb736'b534, 0xbf3e'bd3c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e32_vlmul4) {
  TestVlsegXeXX<UInt32, 2, 4>(0x2000'e407,  // vlseg2e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8b0a'8908, 0x9312'9110, 0x9b1a'9918},
                               {0xa322'a120, 0xab2a'a928, 0xb332'b130, 0xbb3a'b938},
                               {0xc342'c140, 0xcb4a'c948, 0xd352'd150, 0xdb5a'd958},
                               {0xe362'e160, 0xeb6a'e968, 0xf372'f170, 0xfb7a'f978},
                               {0x8706'8504, 0x8f0e'8d0c, 0x9716'9514, 0x9f1e'9d1c},
                               {0xa726'a524, 0xaf2e'ad2c, 0xb736'b534, 0xbf3e'bd3c},
                               {0xc746'c544, 0xcf4e'cd4c, 0xd756'd554, 0xdf5e'dd5c},
                               {0xe766'e564, 0xef6e'ed6c, 0xf776'f574, 0xff7e'fd7c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg3e32_vlmul1) {
  TestVlsegXeXX<UInt32, 3, 1>(0x4000'e407,  // vlseg3e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8f0e'8d0c, 0x9b1a'9918, 0xa726'a524},
                               {0x8706'8504, 0x9312'9110, 0x9f1e'9d1c, 0xab2a'a928},
                               {0x8b0a'8908, 0x9716'9514, 0xa322'a120, 0xaf2e'ad2c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg3e32_vlmul2) {
  TestVlsegXeXX<UInt32, 3, 2>(0x4000'e407,  // vlseg3e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8f0e'8d0c, 0x9b1a'9918, 0xa726'a524},
                               {0xb332'b130, 0xbf3e'bd3c, 0xcb4a'c948, 0xd756'd554},
                               {0x8706'8504, 0x9312'9110, 0x9f1e'9d1c, 0xab2a'a928},
                               {0xb736'b534, 0xc342'c140, 0xcf4e'cd4c, 0xdb5a'd958},
                               {0x8b0a'8908, 0x9716'9514, 0xa322'a120, 0xaf2e'ad2c},
                               {0xbb3a'b938, 0xc746'c544, 0xd352'd150, 0xdf5e'dd5c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg4e32_vlmul1) {
  TestVlsegXeXX<UInt32, 4, 1>(0x6000'e407,  // vlseg4e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                               {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                               {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                               {0x8f0e'8d0c, 0x9f1e'9d1c, 0xaf2e'ad2c, 0xbf3e'bd3c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg4e32_vlmul2) {
  TestVlsegXeXX<UInt32, 4, 2>(0x6000'e407,  // vlseg4e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                               {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                               {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                               {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574},
                               {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                               {0xcb4a'c948, 0xdb5a'd958, 0xeb6a'e968, 0xfb7a'f978},
                               {0x8f0e'8d0c, 0x9f1e'9d1c, 0xaf2e'ad2c, 0xbf3e'bd3c},
                               {0xcf4e'cd4c, 0xdf5e'dd5c, 0xef6e'ed6c, 0xff7e'fd7c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg5e32) {
  TestVlsegXeXX<UInt32, 5, 1>(0x8000'e407,  // vlseg5e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9716'9514, 0xab2a'a928, 0xbf3e'bd3c},
                               {0x8706'8504, 0x9b1a'9918, 0xaf2e'ad2c, 0xc342'c140},
                               {0x8b0a'8908, 0x9f1e'9d1c, 0xb332'b130, 0xc746'c544},
                               {0x8f0e'8d0c, 0xa322'a120, 0xb736'b534, 0xcb4a'c948},
                               {0x9312'9110, 0xa726'a524, 0xbb3a'b938, 0xcf4e'cd4c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg6e32) {
  TestVlsegXeXX<UInt32, 6, 1>(0xa000'e407,  // vlseg6e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9b1a'9918, 0xb332'b130, 0xcb4a'c948},
                               {0x8706'8504, 0x9f1e'9d1c, 0xb736'b534, 0xcf4e'cd4c},
                               {0x8b0a'8908, 0xa322'a120, 0xbb3a'b938, 0xd352'd150},
                               {0x8f0e'8d0c, 0xa726'a524, 0xbf3e'bd3c, 0xd756'd554},
                               {0x9312'9110, 0xab2a'a928, 0xc342'c140, 0xdb5a'd958},
                               {0x9716'9514, 0xaf2e'ad2c, 0xc746'c544, 0xdf5e'dd5c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg7e32) {
  TestVlsegXeXX<UInt32, 7, 1>(0xc000'e407,  // vlseg7e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9f1e'9d1c, 0xbb3a'b938, 0xd756'd554},
                               {0x8706'8504, 0xa322'a120, 0xbf3e'bd3c, 0xdb5a'd958},
                               {0x8b0a'8908, 0xa726'a524, 0xc342'c140, 0xdf5e'dd5c},
                               {0x8f0e'8d0c, 0xab2a'a928, 0xc746'c544, 0xe362'e160},
                               {0x9312'9110, 0xaf2e'ad2c, 0xcb4a'c948, 0xe766'e564},
                               {0x9716'9514, 0xb332'b130, 0xcf4e'cd4c, 0xeb6a'e968},
                               {0x9b1a'9918, 0xb736'b534, 0xd352'd150, 0xef6e'ed6c}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg8e32) {
  TestVlsegXeXX<UInt32, 8, 1>(0xe000'e407,  // vlseg8e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                               {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                               {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                               {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                               {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170},
                               {0x9716'9514, 0xb736'b534, 0xd756'd554, 0xf776'f574},
                               {0x9b1a'9918, 0xbb3a'b938, 0xdb5a'd958, 0xfb7a'f978},
                               {0x9f1e'9d1c, 0xbf3e'bd3c, 0xdf5e'dd5c, 0xff7e'fd7c}});
}

TEST_F(Riscv64InterpreterTest, TestVle64_vlmul1) {
  TestVlsegXeXX<UInt64, 1, 1>(0x0000'f407,  // vle64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908}});
}

TEST_F(Riscv64InterpreterTest, TestVle64_vlmul2) {
  TestVlsegXeXX<UInt64, 1, 2>(0x0000'f407,  // vle64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                               {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918}});
}

TEST_F(Riscv64InterpreterTest, TestVle64_vlmul4) {
  TestVlsegXeXX<UInt64, 1, 4>(0x0000'f407,  // vle64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                               {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
                               {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
                               {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938}});
}

TEST_F(Riscv64InterpreterTest, TestVle64_vlmul8) {
  TestVlsegXeXX<UInt64, 1, 8>(0x0000'f407,  // vle64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                               {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
                               {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
                               {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
                               {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                               {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                               {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                               {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e64_vlmul1) {
  TestVlsegXeXX<UInt64, 2, 1>(0x2000'f407,  // vlseg2e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9716'9514'9312'9110},
                               {0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e64_vlmul2) {
  TestVlsegXeXX<UInt64, 2, 2>(0x2000'f407,  // vlseg2e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9716'9514'9312'9110},
                               {0xa726'a524'a322'a120, 0xb736'b534'b332'b130},
                               {0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918},
                               {0xaf2e'ad2c'ab2a'a928, 0xbf3e'bd3c'bb3a'b938}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg2e64_vlmul4) {
  TestVlsegXeXX<UInt64, 2, 4>(0x2000'f407,  // vlseg2e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9716'9514'9312'9110},
                               {0xa726'a524'a322'a120, 0xb736'b534'b332'b130},
                               {0xc746'c544'c342'c140, 0xd756'd554'd352'd150},
                               {0xe766'e564'e362'e160, 0xf776'f574'f372'f170},
                               {0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918},
                               {0xaf2e'ad2c'ab2a'a928, 0xbf3e'bd3c'bb3a'b938},
                               {0xcf4e'cd4c'cb4a'c948, 0xdf5e'dd5c'db5a'd958},
                               {0xef6e'ed6c'eb6a'e968, 0xff7e'fd7c'fb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg3e64_vlmul1) {
  TestVlsegXeXX<UInt64, 3, 1>(0x4000'f407,  // vlseg3e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9f1e'9d1c'9b1a'9918},
                               {0x8f0e'8d0c'8b0a'8908, 0xa726'a524'a322'a120},
                               {0x9716'9514'9312'9110, 0xaf2e'ad2c'ab2a'a928}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg3e64_vlmul2) {
  TestVlsegXeXX<UInt64, 3, 2>(0x4000'f407,  // vlseg3e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9f1e'9d1c'9b1a'9918},
                               {0xb736'b534'b332'b130, 0xcf4e'cd4c'cb4a'c948},
                               {0x8f0e'8d0c'8b0a'8908, 0xa726'a524'a322'a120},
                               {0xbf3e'bd3c'bb3a'b938, 0xd756'd554'd352'd150},
                               {0x9716'9514'9312'9110, 0xaf2e'ad2c'ab2a'a928},
                               {0xc746'c544'c342'c140, 0xdf5e'dd5c'db5a'd958}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg4e64_vlmul1) {
  TestVlsegXeXX<UInt64, 4, 1>(0x6000'f407,  // vlseg4e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                               {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                               {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                               {0x9f1e'9d1c'9b1a'9918, 0xbf3e'bd3c'bb3a'b938}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg4e64_vlmul2) {
  TestVlsegXeXX<UInt64, 4, 2>(0x6000'f407,  // vlseg4e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                               {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                               {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                               {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968},
                               {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                               {0xd756'd554'd352'd150, 0xf776'f574'f372'f170},
                               {0x9f1e'9d1c'9b1a'9918, 0xbf3e'bd3c'bb3a'b938},
                               {0xdf5e'dd5c'db5a'd958, 0xff7e'fd7c'fb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg5e64) {
  TestVlsegXeXX<UInt64, 5, 1>(0x8000'f407,  // vlseg5e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xaf2e'ad2c'ab2a'a928},
                               {0x8f0e'8d0c'8b0a'8908, 0xb736'b534'b332'b130},
                               {0x9716'9514'9312'9110, 0xbf3e'bd3c'bb3a'b938},
                               {0x9f1e'9d1c'9b1a'9918, 0xc746'c544'c342'c140},
                               {0xa726'a524'a322'a120, 0xcf4e'cd4c'cb4a'c948}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg6e64) {
  TestVlsegXeXX<UInt64, 6, 1>(0xa000'f407,  // vlseg6e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xb736'b534'b332'b130},
                               {0x8f0e'8d0c'8b0a'8908, 0xbf3e'bd3c'bb3a'b938},
                               {0x9716'9514'9312'9110, 0xc746'c544'c342'c140},
                               {0x9f1e'9d1c'9b1a'9918, 0xcf4e'cd4c'cb4a'c948},
                               {0xa726'a524'a322'a120, 0xd756'd554'd352'd150},
                               {0xaf2e'ad2c'ab2a'a928, 0xdf5e'dd5c'db5a'd958}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg7e64) {
  TestVlsegXeXX<UInt64, 7, 1>(0xc000'f407,  // vlseg7e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xbf3e'bd3c'bb3a'b938},
                               {0x8f0e'8d0c'8b0a'8908, 0xc746'c544'c342'c140},
                               {0x9716'9514'9312'9110, 0xcf4e'cd4c'cb4a'c948},
                               {0x9f1e'9d1c'9b1a'9918, 0xd756'd554'd352'd150},
                               {0xa726'a524'a322'a120, 0xdf5e'dd5c'db5a'd958},
                               {0xaf2e'ad2c'ab2a'a928, 0xe766'e564'e362'e160},
                               {0xb736'b534'b332'b130, 0xef6e'ed6c'eb6a'e968}});
}

TEST_F(Riscv64InterpreterTest, TestVlseg8e64) {
  TestVlsegXeXX<UInt64, 8, 1>(0xe000'f407,  // vlseg8e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                               {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                               {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                               {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                               {0xa726'a524'a322'a120, 0xe766'e564'e362'e160},
                               {0xaf2e'ad2c'ab2a'a928, 0xef6e'ed6c'eb6a'e968},
                               {0xb736'b534'b332'b130, 0xf776'f574'f372'f170},
                               {0xbf3e'bd3c'bb3a'b938, 0xff7e'fd7c'fb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlse8_vlmul1) {
  TestVlssegXeXX<UInt8, 1, 1>(0x0820'8407,  // vlse8.v v8, (x1), x2, v0.t
                              4,
                              {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60}});
}

TEST_F(Riscv64InterpreterTest, TestVlse8_vlmul2) {
  TestVlssegXeXX<UInt8, 1, 2>(
      0x0820'8407,  // vlse8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124}});
}

TEST_F(Riscv64InterpreterTest, TestVlse8_vlmul4) {
  TestVlssegXeXX<UInt8, 1, 4>(
      0x0820'8407,  // vlse8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {0, 9, 17, 24, 32, 41, 49, 56, 64, 73, 81, 88, 96, 105, 113, 120},
       {128, 137, 145, 152, 160, 169, 177, 184, 192, 201, 209, 216, 224, 233, 241, 248}});
}

TEST_F(Riscv64InterpreterTest, TestVlse8_vlmul8) {
  TestVlssegXeXX<UInt8, 1, 8>(
      0x0820'8407,  // vlse8.v v8, (x1), x2, v0.t
      2,
      {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126},
       {0, 4, 9, 12, 17, 20, 24, 28, 32, 36, 41, 44, 49, 52, 56, 60},
       {64, 68, 73, 76, 81, 84, 88, 92, 96, 100, 105, 108, 113, 116, 120, 124},
       {128, 132, 137, 140, 145, 148, 152, 156, 160, 164, 169, 172, 177, 180, 184, 188},
       {192, 196, 201, 204, 209, 212, 216, 220, 224, 228, 233, 236, 241, 244, 248, 252}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e8_vlmul1) {
  TestVlssegXeXX<UInt8, 2, 1>(
      0x2820'8407,  // vlsseg2e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e8_vlmul2) {
  TestVlssegXeXX<UInt8, 2, 2>(
      0x2820'8407,  // vlsseg2e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e8_vlmul4) {
  TestVlssegXeXX<UInt8, 2, 4>(
      0x2820'8407,  // vlsseg2e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {0, 9, 17, 24, 32, 41, 49, 56, 64, 73, 81, 88, 96, 105, 113, 120},
       {128, 137, 145, 152, 160, 169, 177, 184, 192, 201, 209, 216, 224, 233, 241, 248},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253},
       {146, 154, 130, 138, 178, 186, 162, 170, 210, 218, 194, 202, 242, 250, 226, 234},
       {18, 26, 2, 10, 50, 58, 34, 42, 82, 90, 66, 74, 114, 122, 98, 106}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg3e8_vlmul1) {
  TestVlssegXeXX<UInt8, 3, 1>(
      0x4820'8407,  // vlsseg3e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg3e8_vlmul2) {
  TestVlssegXeXX<UInt8, 3, 2>(
      0x4820'8407,  // vlsseg3e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg4e8_vlmul1) {
  TestVlssegXeXX<UInt8, 4, 1>(
      0x6820'8407,  // vlsseg4e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg4e8_vlmul2) {
  TestVlssegXeXX<UInt8, 4, 2>(
      0x6820'8407,  // vlsseg4e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126},
       {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191},
       {195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg5e8) {
  TestVlssegXeXX<UInt8, 5, 1>(
      0x8820'8407,  // vlsseg5e8.v v8, (x1), x2, v0.t
      8,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg6e8) {
  TestVlssegXeXX<UInt8, 6, 1>(
      0xa820'8407,  // vlsseg6e8.v v8, (x1), x2, v0.t
      8,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124},
       {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg7e8) {
  TestVlssegXeXX<UInt8, 7, 1>(
      0xc820'8407,  // vlsseg7e8.v v8, (x1), x2, v0.t
      8,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124},
       {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253},
       {6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg8e8) {
  TestVlssegXeXX<UInt8, 8, 1>(
      0xe820'8407,  // vlsseg8e8.v v8, (x1), x2, v0.t
      8,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124},
       {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253},
       {6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126},
       {135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255}});
}

TEST_F(Riscv64InterpreterTest, TestVlse16_vlmul1) {
  TestVlssegXeXX<UInt16, 1, 1>(0x0820'd407,  // vlse16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938}});
}

TEST_F(Riscv64InterpreterTest, TestVlse16_vlmul2) {
  TestVlssegXeXX<UInt16, 1, 2>(0x0820'd407,  // vlse16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978}});
}

TEST_F(Riscv64InterpreterTest, TestVlse16_vlmul4) {
  TestVlssegXeXX<UInt16, 1, 4>(0x0820'd407,  // vlse16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x9200, 0x8211, 0xb220, 0xa231, 0xd240, 0xc251, 0xf260, 0xe271},
                                {0x1280, 0x0291, 0x32a0, 0x22b1, 0x52c0, 0x42d1, 0x72e0, 0x62f1}});
}

TEST_F(Riscv64InterpreterTest, TestVlse16_vlmul8) {
  TestVlssegXeXX<UInt16, 1, 8>(0x0820'd407,  // vlse16.v v8, (x1), x2, v0.t
                               4,
                               {{0x8100, 0x8504, 0x8908, 0x8d0c, 0x9110, 0x9514, 0x9918, 0x9d1c},
                                {0xa120, 0xa524, 0xa928, 0xad2c, 0xb130, 0xb534, 0xb938, 0xbd3c},
                                {0xc140, 0xc544, 0xc948, 0xcd4c, 0xd150, 0xd554, 0xd958, 0xdd5c},
                                {0xe160, 0xe564, 0xe968, 0xed6c, 0xf170, 0xf574, 0xf978, 0xfd7c},
                                {0x9200, 0x9a09, 0x8211, 0x8a18, 0xb220, 0xba29, 0xa231, 0xaa38},
                                {0xd240, 0xda49, 0xc251, 0xca58, 0xf260, 0xfa69, 0xe271, 0xea78},
                                {0x1280, 0x1a89, 0x0291, 0x0a98, 0x32a0, 0x3aa9, 0x22b1, 0x2ab8},
                                {0x52c0, 0x5ac9, 0x42d1, 0x4ad8, 0x72e0, 0x7ae9, 0x62f1, 0x6af8}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e16_vlmul1) {
  TestVlssegXeXX<UInt16, 2, 1>(0x2820'd407,  // vlsseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e16_vlmul2) {
  TestVlssegXeXX<UInt16, 2, 2>(0x2820'd407,  // vlsseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e16_vlmul4) {
  TestVlssegXeXX<UInt16, 2, 4>(0x2820'd407,  // vlsseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x9200, 0x8211, 0xb220, 0xa231, 0xd240, 0xc251, 0xf260, 0xe271},
                                {0x1280, 0x0291, 0x32a0, 0x22b1, 0x52c0, 0x42d1, 0x72e0, 0x62f1},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a},
                                {0x9604, 0x8614, 0xb624, 0xa634, 0xd644, 0xc654, 0xf664, 0xe674},
                                {0x1684, 0x0694, 0x36a4, 0x26b4, 0x56c4, 0x46d4, 0x76e4, 0x66f4}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg3e16_vlmul1) {
  TestVlssegXeXX<UInt16, 3, 1>(0x4820'd407,  // vlsseg3e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg3e16_vlmul2) {
  TestVlssegXeXX<UInt16, 3, 2>(0x4820'd407,  // vlsseg3e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a},
                                {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                                {0xc544, 0xcd4c, 0xd554, 0xdd5c, 0xe564, 0xed6c, 0xf574, 0xfd7c}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg4e16_vlmul1) {
  TestVlssegXeXX<UInt16, 4, 1>(0x6820'd407,  // vlsseg4e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                                {0x8706, 0x8f0e, 0x9716, 0x9f1e, 0xa726, 0xaf2e, 0xb736, 0xbf3e}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg4e16_vlmul2) {
  TestVlssegXeXX<UInt16, 4, 2>(0x6820'd407,  // vlsseg4e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a},
                                {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                                {0xc544, 0xcd4c, 0xd554, 0xdd5c, 0xe564, 0xed6c, 0xf574, 0xfd7c},
                                {0x8706, 0x8f0e, 0x9716, 0x9f1e, 0xa726, 0xaf2e, 0xb736, 0xbf3e},
                                {0xc746, 0xcf4e, 0xd756, 0xdf5e, 0xe766, 0xef6e, 0xf776, 0xff7e}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg5e16) {
  TestVlssegXeXX<UInt16, 5, 1>(0x8820'd407,  // vlsseg5e16.v v8, (x1), x2, v0.t
                               16,
                               {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                                {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                                {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                                {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                                {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg6e16) {
  TestVlssegXeXX<UInt16, 6, 1>(0xa820'd407,  // vlsseg6e16.v v8, (x1), x2, v0.t
                               16,
                               {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                                {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                                {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                                {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                                {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978},
                                {0x8b0a, 0x9b1a, 0xab2a, 0xbb3a, 0xcb4a, 0xdb5a, 0xeb6a, 0xfb7a}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg7e16) {
  TestVlssegXeXX<UInt16, 7, 1>(0xc820'd407,  // vlsseg7e16.v v8, (x1), x2, v0.t
                               16,
                               {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                                {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                                {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                                {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                                {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978},
                                {0x8b0a, 0x9b1a, 0xab2a, 0xbb3a, 0xcb4a, 0xdb5a, 0xeb6a, 0xfb7a},
                                {0x8d0c, 0x9d1c, 0xad2c, 0xbd3c, 0xcd4c, 0xdd5c, 0xed6c, 0xfd7c}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg8e16) {
  TestVlssegXeXX<UInt16, 8, 1>(0xe820'd407,  // vlsseg8e16.v v8, (x1), x2, v0.t
                               16,
                               {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                                {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                                {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                                {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                                {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978},
                                {0x8b0a, 0x9b1a, 0xab2a, 0xbb3a, 0xcb4a, 0xdb5a, 0xeb6a, 0xfb7a},
                                {0x8d0c, 0x9d1c, 0xad2c, 0xbd3c, 0xcd4c, 0xdd5c, 0xed6c, 0xfd7c},
                                {0x8f0e, 0x9f1e, 0xaf2e, 0xbf3e, 0xcf4e, 0xdf5e, 0xef6e, 0xff7e}});
}

TEST_F(Riscv64InterpreterTest, TestVlse32_vlmul1) {
  TestVlssegXeXX<UInt32, 1, 1>(0x0820'e407,  // vlse32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130}});
}

TEST_F(Riscv64InterpreterTest, TestVlse32_vlmul2) {
  TestVlssegXeXX<UInt32, 1, 2>(0x0820'e407,  // vlse32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170}});
}

TEST_F(Riscv64InterpreterTest, TestVlse32_vlmul4) {
  TestVlssegXeXX<UInt32, 1, 4>(0x0820'e407,  // vlse32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x9604'9200, 0xb624'b220, 0xd644'd240, 0xf664'f260},
                                {0x1684'1280, 0x36a4'32a0, 0x56c4'52c0, 0x76e4'72e0}});
}

TEST_F(Riscv64InterpreterTest, TestVlse32_vlmul8) {
  TestVlssegXeXX<UInt32, 1, 8>(0x0820'e407,  // vlse32.v v8, (x1), x2, v0.t
                               8,
                               {{0x8302'8100, 0x8b0a'8908, 0x9312'9110, 0x9b1a'9918},
                                {0xa322'a120, 0xab2a'a928, 0xb332'b130, 0xbb3a'b938},
                                {0xc342'c140, 0xcb4a'c948, 0xd352'd150, 0xdb5a'd958},
                                {0xe362'e160, 0xeb6a'e968, 0xf372'f170, 0xfb7a'f978},
                                {0x9604'9200, 0x8614'8211, 0xb624'b220, 0xa634'a231},
                                {0xd644'd240, 0xc654'c251, 0xf664'f260, 0xe674'e271},
                                {0x1684'1280, 0x0694'0291, 0x36a4'32a0, 0x26b4'22b1},
                                {0x56c4'52c0, 0x46d4'42d1, 0x76e4'72e0, 0x66f4'62f1}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e32_vlmul1) {
  TestVlssegXeXX<UInt32, 2, 1>(0x2820'e407,  // vlsseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e32_vlmul2) {
  TestVlssegXeXX<UInt32, 2, 2>(0x2820'e407,  // vlsseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e32_vlmul4) {
  TestVlssegXeXX<UInt32, 2, 4>(0x2820'e407,  // vlsseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x9604'9200, 0xb624'b220, 0xd644'd240, 0xf664'f260},
                                {0x1684'1280, 0x36a4'32a0, 0x56c4'52c0, 0x76e4'72e0},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574},
                                {0x9e0c'9a09, 0xbe2c'ba29, 0xde4c'da49, 0xfe6c'fa69},
                                {0x1e8c'1a89, 0x3eac'3aa9, 0x5ecc'5ac9, 0x7eec'7ae9}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg3e32_vlmul1) {
  TestVlssegXeXX<UInt32, 3, 1>(0x4820'e407,  // vlsseg3e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg3e32_vlmul2) {
  TestVlssegXeXX<UInt32, 3, 2>(0x4820'e407,  // vlsseg3e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574},
                                {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                                {0xcb4a'c948, 0xdb5a'd958, 0xeb6a'e968, 0xfb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg4e32_vlmul1) {
  TestVlssegXeXX<UInt32, 4, 1>(0x6820'e407,  // vlsseg4e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                                {0x8f0e'8d0c, 0x9f1e'9d1c, 0xaf2e'ad2c, 0xbf3e'bd3c}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg4e32_vlmul2) {
  TestVlssegXeXX<UInt32, 4, 2>(0x6820'e407,  // vlsseg4e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574},
                                {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                                {0xcb4a'c948, 0xdb5a'd958, 0xeb6a'e968, 0xfb7a'f978},
                                {0x8f0e'8d0c, 0x9f1e'9d1c, 0xaf2e'ad2c, 0xbf3e'bd3c},
                                {0xcf4e'cd4c, 0xdf5e'dd5c, 0xef6e'ed6c, 0xff7e'fd7c}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg5e32) {
  TestVlssegXeXX<UInt32, 5, 1>(0x8820'e407,  // vlsseg5e32.v v8, (x1), x2, v0.t
                               32,
                               {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                                {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                                {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                                {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                                {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg6e32) {
  TestVlssegXeXX<UInt32, 6, 1>(0xa820'e407,  // vlsseg6e32.v v8, (x1), x2, v0.t
                               32,
                               {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                                {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                                {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                                {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                                {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170},
                                {0x9716'9514, 0xb736'b534, 0xd756'd554, 0xf776'f574}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg7e32) {
  TestVlssegXeXX<UInt32, 7, 1>(0xc820'e407,  // vlsseg7e32.v v8, (x1), x2, v0.t
                               32,
                               {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                                {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                                {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                                {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                                {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170},
                                {0x9716'9514, 0xb736'b534, 0xd756'd554, 0xf776'f574},
                                {0x9b1a'9918, 0xbb3a'b938, 0xdb5a'd958, 0xfb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg8e32) {
  TestVlssegXeXX<UInt32, 8, 1>(0xe820'e407,  // vlsseg8e32.v v8, (x1), x2, v0.t
                               32,
                               {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                                {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                                {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                                {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                                {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170},
                                {0x9716'9514, 0xb736'b534, 0xd756'd554, 0xf776'f574},
                                {0x9b1a'9918, 0xbb3a'b938, 0xdb5a'd958, 0xfb7a'f978},
                                {0x9f1e'9d1c, 0xbf3e'bd3c, 0xdf5e'dd5c, 0xff7e'fd7c}});
}

TEST_F(Riscv64InterpreterTest, TestVlse64_vlmul1) {
  TestVlssegXeXX<UInt64, 1, 1>(0x0820'f407,  // vlse64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120}});
}

TEST_F(Riscv64InterpreterTest, TestVlse64_vlmul2) {
  TestVlssegXeXX<UInt64, 1, 2>(0x0820'f407,  // vlse64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160}});
}

TEST_F(Riscv64InterpreterTest, TestVlse64_vlmul4) {
  TestVlssegXeXX<UInt64, 1, 4>(0x0820'f407,  // vlse64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x9e0c'9a09'9604'9200, 0xde4c'da49'd644'd240},
                                {0x1e8c'1a89'1684'1280, 0x5ecc'5ac9'56c4'52c0}});
}

TEST_F(Riscv64InterpreterTest, TestVlse64_vlmul8) {
  TestVlssegXeXX<UInt64, 1, 8>(0x0820'f407,  // vlse64.v v8, (x1), x2, v0.t
                               16,
                               {{0x8706'8504'8302'8100, 0x9716'9514'9312'9110},
                                {0xa726'a524'a322'a120, 0xb736'b534'b332'b130},
                                {0xc746'c544'c342'c140, 0xd756'd554'd352'd150},
                                {0xe766'e564'e362'e160, 0xf776'f574'f372'f170},
                                {0x9e0c'9a09'9604'9200, 0xbe2c'ba29'b624'b220},
                                {0xde4c'da49'd644'd240, 0xfe6c'fa69'f664'f260},
                                {0x1e8c'1a89'1684'1280, 0x3eac'3aa9'36a4'32a0},
                                {0x5ecc'5ac9'56c4'52c0, 0x7eec'7ae9'76e4'72e0}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e64_vlmul1) {
  TestVlssegXeXX<UInt64, 2, 1>(0x2820'f407,  // vlsseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e64_vlmul2) {
  TestVlssegXeXX<UInt64, 2, 2>(0x2820'f407,  // vlsseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg2e64_vlmul4) {
  TestVlssegXeXX<UInt64, 2, 4>(0x2820'f407,  // vlsseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x9e0c'9a09'9604'9200, 0xde4c'da49'd644'd240},
                                {0x1e8c'1a89'1684'1280, 0x5ecc'5ac9'56c4'52c0},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968},
                                {0x8e1c'8a18'8614'8211, 0xce5c'ca58'c654'c251},
                                {0x0e9c'0a98'0694'0291, 0x4edc'4ad8'46d4'42d1}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg3e64_vlmul1) {
  TestVlssegXeXX<UInt64, 3, 1>(0x4820'f407,  // vlsseg3e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0x9716'9514'9312'9110, 0xb736'b534'b332'b130}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg3e64_vlmul2) {
  TestVlssegXeXX<UInt64, 3, 2>(0x4820'f407,  // vlsseg3e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968},
                                {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                                {0xd756'd554'd352'd150, 0xf776'f574'f372'f170}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg4e64_vlmul1) {
  TestVlssegXeXX<UInt64, 4, 1>(0x6820'f407,  // vlsseg4e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                                {0x9f1e'9d1c'9b1a'9918, 0xbf3e'bd3c'bb3a'b938}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg4e64_vlmul2) {
  TestVlssegXeXX<UInt64, 4, 2>(0x6820'f407,  // vlsseg4e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968},
                                {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                                {0xd756'd554'd352'd150, 0xf776'f574'f372'f170},
                                {0x9f1e'9d1c'9b1a'9918, 0xbf3e'bd3c'bb3a'b938},
                                {0xdf5e'dd5c'db5a'd958, 0xff7e'fd7c'fb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg5e64) {
  TestVlssegXeXX<UInt64, 5, 1>(0x8820'f407,  // vlsseg5e64.v v8, (x1), x2, v0.t
                               64,
                               {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                                {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                                {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                                {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                                {0xa726'a524'a322'a120, 0xe766'e564'e362'e160}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg6e64) {
  TestVlssegXeXX<UInt64, 6, 1>(0xa820'f407,  // vlsseg6e64.v v8, (x1), x2, v0.t
                               64,
                               {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                                {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                                {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                                {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                                {0xa726'a524'a322'a120, 0xe766'e564'e362'e160},
                                {0xaf2e'ad2c'ab2a'a928, 0xef6e'ed6c'eb6a'e968}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg7e64) {
  TestVlssegXeXX<UInt64, 7, 1>(0xc820'f407,  // vlsseg7e64.v v8, (x1), x2, v0.t
                               64,
                               {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                                {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                                {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                                {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                                {0xa726'a524'a322'a120, 0xe766'e564'e362'e160},
                                {0xaf2e'ad2c'ab2a'a928, 0xef6e'ed6c'eb6a'e968},
                                {0xb736'b534'b332'b130, 0xf776'f574'f372'f170}});
}

TEST_F(Riscv64InterpreterTest, TestVlsseg8e64) {
  TestVlssegXeXX<UInt64, 8, 1>(0xe820'f407,  // vlsseg8e64.v v8, (x1), x2, v0.t
                               64,
                               {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                                {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                                {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                                {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                                {0xa726'a524'a322'a120, 0xe766'e564'e362'e160},
                                {0xaf2e'ad2c'ab2a'a928, 0xef6e'ed6c'eb6a'e968},
                                {0xb736'b534'b332'b130, 0xf776'f574'f372'f170},
                                {0xbf3e'bd3c'bb3a'b938, 0xff7e'fd7c'fb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlm) {
  TestVlm(0x02b0'8407,  // vlm.v v8, (x1)
          {0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143});
}

TEST_F(Riscv64InterpreterTest, Vsxei8_sew8_vlmul1) {
  VsxsegXeiXX<UInt8, 1, 1>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                           {0x0487'8506'0283'0081, 0x0a89'0c8d'8b8f'080e});
}

TEST_F(Riscv64InterpreterTest, Vsxei8_sew8_vlmul2) {
  VsxsegXeiXX<UInt8, 1, 2>(
      0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
      {0x0487'8506'0283'0081, 0x0a89'0c8d'8b8f'080e, 0x9f93'9b1e'9714'121a, 0x9110'1899'1c95'169d});
}

TEST_F(Riscv64InterpreterTest, Vsxei8_sew8_vlmul4) {
  VsxsegXeiXX<UInt8, 1, 4>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                           {0x0487'8506'0283'0081,
                            0x0a89'0c8d'8b8f'080e,
                            0x9f93'9b1e'9714'121a,
                            0x9110'1899'1c95'169d,
                            0x2ea5'bd2c'30a3'38b9,
                            0xafad'3e20'a728'b1ab,
                            0x3626'b722'b5a1'bbbf,
                            0xa932'2434'b33a'2a3c});
}

TEST_F(Riscv64InterpreterTest, Vsxei8_sew8_vlmul8) {
  VsxsegXeiXX<UInt8, 1, 8>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                           {0x0487'8506'0283'0081,
                            0x0a89'0c8d'8b8f'080e,
                            0x9f93'9b1e'9714'121a,
                            0x9110'1899'1c95'169d,
                            0x2ea5'bd2c'30a3'38b9,
                            0xafad'3e20'a728'b1ab,
                            0x3626'b722'b5a1'bbbf,
                            0xa932'2434'b33a'2a3c,
                            0x6aed'ddd1'e35c'66c7,
                            0xd542'72e1'4674'4ed3,
                            0x78ef'd9e7'7a7e'eb76,
                            0xfbf9'f5c1'6e5a'c5ff,
                            0x52cd'c362'6c48'dfd7,
                            0x4a54'50fd'f3f7'cf68,
                            0x56cb'e57c'7044'60c9,
                            0xf1db'6440'58e9'4c5e});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew8_vlmul1) {
  VsxsegXeiXX<UInt8, 2, 1>(
      0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
      {0x1202'9383'1000'9181, 0x1404'9787'9585'1606, 0x9b8b'9f8f'1808'1e0e, 0x1a0a'9989'1c0c'9d8d});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew8_vlmul2) {
  VsxsegXeiXX<UInt8, 2, 2>(0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                           {0x2202'a383'2000'a181,
                            0x2404'a787'a585'2606,
                            0xab8b'af8f'2808'2e0e,
                            0x2a0a'a989'2c0c'ad8d,
                            0xb797'3414'3212'3a1a,
                            0xbf9f'b393'bb9b'3e1e,
                            0x3c1c'b595'3616'bd9d,
                            0xb191'3010'3818'b999});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew8_vlmul4) {
  VsxsegXeiXX<UInt8, 2, 4>(0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                           {0x4202'c383'4000'c181,
                            0x4404'c787'c585'4606,
                            0xcb8b'cf8f'4808'4e0e,
                            0x4a0a'c989'4c0c'cd8d,
                            0xd797'5414'5212'5a1a,
                            0xdf9f'd393'db9b'5e1e,
                            0x5c1c'd595'5616'dd9d,
                            0xd191'5010'5818'd999,
                            0x7030'e3a3'7838'f9b9,
                            0x6e2e'e5a5'fdbd'6c2c,
                            0xe7a7'6828'f1b1'ebab,
                            0xefaf'edad'7e3e'6020,
                            0xf5b5'e1a1'fbbb'ffbf,
                            0x7636'6626'f7b7'6222,
                            0xf3b3'7a3a'6a2a'7c3c,
                            0xe9a9'7232'6424'7434});
}

TEST_F(Riscv64InterpreterTest, Vsxseg3eiXX_sew8_vlmul1) {
  VsxsegXeiXX<UInt8, 3, 1>(0x4500'8427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                           {0x9383'2010'00a1'9181,
                            0x8526'1606'2212'02a3,
                            0x2414'04a7'9787'a595,
                            0x9f8f'2818'082e'1e0e,
                            0x0cad'9d8d'ab9b'8baf,
                            0x2a1a'0aa9'9989'2c1c});
}

TEST_F(Riscv64InterpreterTest, Vsxseg3eiXX_sew8_vlmul2) {
  VsxsegXeiXX<UInt8, 3, 2>(0x4500'8427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                           {0xa383'4020'00c1'a181,
                            0x8546'2606'4222'02c3,
                            0x4424'04c7'a787'c5a5,
                            0xaf8f'4828'084e'2e0e,
                            0x0ccd'ad8d'cbab'8bcf,
                            0x4a2a'0ac9'a989'4c2c,
                            0x3414'5232'125a'3a1a,
                            0x9b5e'3e1e'd7b7'9754,
                            0xdfbf'9fd3'b393'dbbb,
                            0xb595'5636'16dd'bd9d,
                            0x18d9'b999'5c3c'1cd5,
                            0xd1b1'9150'3010'5838});
}

TEST_F(Riscv64InterpreterTest, Vsxseg4eiXX_sew8_vlmul1) {
  VsxsegXeiXX<UInt8, 4, 1>(0x6500'8427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                           {0x3020'1000'b1a1'9181,
                            0x3222'1202'b3a3'9383,
                            0xb5a5'9585'3626'1606,
                            0x3424'1404'b7a7'9787,
                            0x3828'1808'3e2e'1e0e,
                            0xbbab'9b8b'bfaf'9f8f,
                            0x3c2c'1c0c'bdad'9d8d,
                            0x3a2a'1a0a'b9a9'9989});
}

TEST_F(Riscv64InterpreterTest, Vsxseg4eiXX_sew8_vlmul2) {
  VsxsegXeiXX<UInt8, 4, 2>(0x6500'8427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                           {0x6040'2000'e1c1'a181,
                            0x6242'2202'e3c3'a383,
                            0xe5c5'a585'6646'2606,
                            0x6444'2404'e7c7'a787,
                            0x6848'2808'6e4e'2e0e,
                            0xebcb'ab8b'efcf'af8f,
                            0x6c4c'2c0c'edcd'ad8d,
                            0x6a4a'2a0a'e9c9'a989,
                            0x7252'3212'7a5a'3a1a,
                            0xf7d7'b797'7454'3414,
                            0xfbdb'bb9b'7e5e'3e1e,
                            0xffdf'bf9f'f3d3'b393,
                            0x7656'3616'fddd'bd9d,
                            0x7c5c'3c1c'f5d5'b595,
                            0x7858'3818'f9d9'b999,
                            0xf1d1'b191'7050'3010});
}

TEST_F(Riscv64InterpreterTest, Vsxseg5eiXX_sew8) {
  VsxsegXeiXX<UInt8, 5, 1>(0x8500'8427,  // Vsuxseg5ei8.v v8, (x1), v16, v0.t
                           {0x2010'00c1'b1a1'9181,
                            0x02c3'b3a3'9383'4030,
                            0x3626'1606'4232'2212,
                            0x9787'c5b5'a595'8546,
                            0x4434'2414'04c7'b7a7,
                            0x2818'084e'3e2e'1e0e,
                            0x8bcf'bfaf'9f8f'4838,
                            0xbdad'9d8d'cbbb'ab9b,
                            0x9989'4c3c'2c1c'0ccd,
                            0x4a3a'2a1a'0ac9'b9a9});
}

TEST_F(Riscv64InterpreterTest, Vsxseg6eiXX_sew8) {
  VsxsegXeiXX<UInt8, 6, 1>(0xa500'8427,  // Vsuxseg6ei8.v v8, (x1), v16, v0.t
                           {0x1000'd1c1'b1a1'9181,
                            0xb3a3'9383'5040'3020,
                            0x5242'3222'1202'd3c3,
                            0x9585'5646'3626'1606,
                            0xb7a7'9787'd5c5'b5a5,
                            0x5444'3424'1404'd7c7,
                            0x1808'5e4e'3e2e'1e0e,
                            0xbfaf'9f8f'5848'3828,
                            0xdbcb'bbab'9b8b'dfcf,
                            0x1c0c'ddcd'bdad'9d8d,
                            0xb9a9'9989'5c4c'3c2c,
                            0x5a4a'3a2a'1a0a'd9c9});
}

TEST_F(Riscv64InterpreterTest, Vsxseg7eiXX_sew8) {
  VsxsegXeiXX<UInt8, 7, 1>(0xc500'8427,  // Vsuxseg7ei8.v v8, (x1), v16, v0.t
                           {0x00e1'd1c1'b1a1'9181,
                            0x9383'6050'4030'2010,
                            0x2212'02e3'd3c3'b3a3,
                            0x3626'1606'6252'4232,
                            0xc5b5'a595'8566'5646,
                            0xd7c7'b7a7'9787'e5d5,
                            0x6454'4434'2414'04e7,
                            0x086e'5e4e'3e2e'1e0e,
                            0x9f8f'6858'4838'2818,
                            0xab9b'8bef'dfcf'bfaf,
                            0xbdad'9d8d'ebdb'cbbb,
                            0x4c3c'2c1c'0ced'ddcd,
                            0xd9c9'b9a9'9989'6c5c,
                            0x6a5a'4a3a'2a1a'0ae9});
}

TEST_F(Riscv64InterpreterTest, Vsxseg8eiXX_sew8) {
  VsxsegXeiXX<UInt8, 8, 1>(0xe500'8427,  // Vsuxseg8ei8.v v8, (x1), v16, v0.t
                           {0xf1e1'd1c1'b1a1'9181,
                            0x7060'5040'3020'1000,
                            0xf3e3'd3c3'b3a3'9383,
                            0x7262'5242'3222'1202,
                            0x7666'5646'3626'1606,
                            0xf5e5'd5c5'b5a5'9585,
                            0xf7e7'd7c7'b7a7'9787,
                            0x7464'5444'3424'1404,
                            0x7e6e'5e4e'3e2e'1e0e,
                            0x7868'5848'3828'1808,
                            0xffef'dfcf'bfaf'9f8f,
                            0xfbeb'dbcb'bbab'9b8b,
                            0xfded'ddcd'bdad'9d8d,
                            0x7c6c'5c4c'3c2c'1c0c,
                            0xf9e9'd9c9'b9a9'9989,
                            0x7a6a'5a4a'3a2a'1a0a});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew16_vlmul1) {
  VsxsegXeiXX<UInt16, 1, 1>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8504'8706'8100'8302, 0x8908'8f0e'8b0a'8d0c});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew16_vlmul2) {
  VsxsegXeiXX<UInt16, 1, 2>(
      0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
      {0x8504'8706'8100'8302, 0x8908'8f0e'8b0a'8d0c, 0x9716'9f1e'9110'9d1c, 0x9514'9312'9918'9b1a});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew16_vlmul4) {
  VsxsegXeiXX<UInt16, 1, 4>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8504'8706'8100'8302,
                             0x8908'8f0e'8b0a'8d0c,
                             0x9716'9f1e'9110'9d1c,
                             0x9514'9312'9918'9b1a,
                             0xaf2e'a928'a524'b534,
                             0xbf3e'a726'b736'bd3c,
                             0xb938'ab2a'ad2c'bb3a,
                             0xa322'a120'b130'b332});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew16_vlmul8) {
  VsxsegXeiXX<UInt16, 1, 8>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8504'8706'8100'8302,
                             0x8908'8f0e'8b0a'8d0c,
                             0x9716'9f1e'9110'9d1c,
                             0x9514'9312'9918'9b1a,
                             0xaf2e'a928'a524'b534,
                             0xbf3e'a726'b736'bd3c,
                             0xb938'ab2a'ad2c'bb3a,
                             0xa322'a120'b130'b332,
                             0xe160'c746'f170'f372,
                             0xdd5c'cb4a'fb7a'd958,
                             0xcf4e'd150'e362'd756,
                             0xdf5e'db5a'fd7c'c140,
                             0xeb6a'c342'f776'ff7e,
                             0xed6c'cd4c'ef6e'c544,
                             0xe766'f574'd554'f978,
                             0xd352'e564'c948'e968});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew16_vlmul1) {
  VsxsegXeiXX<UInt16, 2, 1>(
      0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
      {0x9110'8100'9312'8302, 0x9514'8504'9716'8706, 0x9b1a'8b0a'9d1c'8d0c, 0x9918'8908'9f1e'8f0e});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew16_vlmul2) {
  VsxsegXeiXX<UInt16, 2, 2>(0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0xa120'8100'a322'8302,
                             0xa524'8504'a726'8706,
                             0xab2a'8b0a'ad2c'8d0c,
                             0xa928'8908'af2e'8f0e,
                             0xb130'9110'bd3c'9d1c,
                             0xb736'9716'bf3e'9f1e,
                             0xb938'9918'bb3a'9b1a,
                             0xb534'9514'b332'9312});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew16_vlmul4) {
  VsxsegXeiXX<UInt16, 2, 4>(0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0xc140'8100'c342'8302,
                             0xc544'8504'c746'8706,
                             0xcb4a'8b0a'cd4c'8d0c,
                             0xc948'8908'cf4e'8f0e,
                             0xd150'9110'dd5c'9d1c,
                             0xd756'9716'df5e'9f1e,
                             0xd958'9918'db5a'9b1a,
                             0xd554'9514'd352'9312,
                             0xe564'a524'f574'b534,
                             0xef6e'af2e'e968'a928,
                             0xf776'b736'fd7c'bd3c,
                             0xff7e'bf3e'e766'a726,
                             0xed6c'ad2c'fb7a'bb3a,
                             0xf978'b938'eb6a'ab2a,
                             0xf170'b130'f372'b332,
                             0xe362'a322'e160'a120});
}

TEST_F(Riscv64InterpreterTest, Vsxseg3eiXX_sew16_vlmul1) {
  VsxsegXeiXX<UInt16, 3, 1>(0x4500'8427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x8100'a322'9312'8302,
                             0x9716'8706'a120'9110,
                             0xa524'9514'8504'a726,
                             0x8b0a'ad2c'9d1c'8d0c,
                             0x9f1e'8f0e'ab2a'9b1a,
                             0xa928'9918'8908'af2e});
}

TEST_F(Riscv64InterpreterTest, Vsxseg3eiXX_sew16_vlmul2) {
  VsxsegXeiXX<UInt16, 3, 2>(0x4500'8427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x8100'c342'a322'8302,
                             0xa726'8706'c140'a120,
                             0xc544'a524'8504'c746,
                             0x8b0a'cd4c'ad2c'8d0c,
                             0xaf2e'8f0e'cb4a'ab2a,
                             0xc948'a928'8908'cf4e,
                             0x9110'dd5c'bd3c'9d1c,
                             0xbf3e'9f1e'd150'b130,
                             0xd756'b736'9716'df5e,
                             0x9918'db5a'bb3a'9b1a,
                             0xb332'9312'd958'b938,
                             0xd554'b534'9514'd352});
}

TEST_F(Riscv64InterpreterTest, Vsxseg4eiXX_sew16_vlmul1) {
  VsxsegXeiXX<UInt16, 4, 1>(0x6500'8427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0xb130'a120'9110'8100,
                             0xb736'a726'9716'8706,
                             0xb534'a524'9514'8504,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0xbb3a'ab2a'9b1a'8b0a,
                             0xbf3e'af2e'9f1e'8f0e,
                             0xb938'a928'9918'8908});
}

TEST_F(Riscv64InterpreterTest, Vsxseg4eiXX_sew16_vlmul2) {
  VsxsegXeiXX<UInt16, 4, 2>(0x6500'8427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0xe362'c342'a322'8302,
                             0xe160'c140'a120'8100,
                             0xe766'c746'a726'8706,
                             0xe564'c544'a524'8504,
                             0xed6c'cd4c'ad2c'8d0c,
                             0xeb6a'cb4a'ab2a'8b0a,
                             0xef6e'cf4e'af2e'8f0e,
                             0xe968'c948'a928'8908,
                             0xfd7c'dd5c'bd3c'9d1c,
                             0xf170'd150'b130'9110,
                             0xff7e'df5e'bf3e'9f1e,
                             0xf776'd756'b736'9716,
                             0xfb7a'db5a'bb3a'9b1a,
                             0xf978'd958'b938'9918,
                             0xf372'd352'b332'9312,
                             0xf574'd554'b534'9514});
}

TEST_F(Riscv64InterpreterTest, Vsxseg5eiXX_sew16) {
  VsxsegXeiXX<UInt16, 5, 1>(0x8500'8427,  // Vsuxseg5ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0xa120'9110'8100'c342,
                             0x9716'8706'c140'b130,
                             0x8504'c746'b736'a726,
                             0xc544'b534'a524'9514,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0xab2a'9b1a'8b0a'cd4c,
                             0x9f1e'8f0e'cb4a'bb3a,
                             0x8908'cf4e'bf3e'af2e,
                             0xc948'b938'a928'9918});
}

TEST_F(Riscv64InterpreterTest, Vsxseg6eiXX_sew16) {
  VsxsegXeiXX<UInt16, 6, 1>(0xa500'8427,  // Vsuxseg6ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0x9110'8100'd352'c342,
                             0xd150'c140'b130'a120,
                             0xb736'a726'9716'8706,
                             0x9514'8504'd756'c746,
                             0xd554'c544'b534'a524,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0x9b1a'8b0a'dd5c'cd4c,
                             0xdb5a'cb4a'bb3a'ab2a,
                             0xbf3e'af2e'9f1e'8f0e,
                             0x9918'8908'df5e'cf4e,
                             0xd958'c948'b938'a928});
}

TEST_F(Riscv64InterpreterTest, Vsxseg7eiXX_sew16) {
  VsxsegXeiXX<UInt16, 7, 1>(0xc500'8427,  // Vsuxseg7ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0x8100'e362'd352'c342,
                             0xc140'b130'a120'9110,
                             0x9716'8706'e160'd150,
                             0xd756'c746'b736'a726,
                             0xa524'9514'8504'e766,
                             0xe564'd554'c544'b534,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0x8b0a'ed6c'dd5c'cd4c,
                             0xcb4a'bb3a'ab2a'9b1a,
                             0x9f1e'8f0e'eb6a'db5a,
                             0xdf5e'cf4e'bf3e'af2e,
                             0xa928'9918'8908'ef6e,
                             0xe968'd958'c948'b938});
}

TEST_F(Riscv64InterpreterTest, Vsxseg8eiXX_sew16) {
  VsxsegXeiXX<UInt16, 8, 1>(0xe500'8427,  // Vsuxseg8ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0xf372'e362'd352'c342,
                             0xb130'a120'9110'8100,
                             0xf170'e160'd150'c140,
                             0xb736'a726'9716'8706,
                             0xf776'e766'd756'c746,
                             0xb534'a524'9514'8504,
                             0xf574'e564'd554'c544,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0xfd7c'ed6c'dd5c'cd4c,
                             0xbb3a'ab2a'9b1a'8b0a,
                             0xfb7a'eb6a'db5a'cb4a,
                             0xbf3e'af2e'9f1e'8f0e,
                             0xff7e'ef6e'df5e'cf4e,
                             0xb938'a928'9918'8908,
                             0xf978'e968'd958'c948});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew32_vlmul1) {
  VsxsegXeiXX<UInt32, 1, 1>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8302'8100'8706'8504, 0x8b0a'8908'8f0e'8d0c});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew32_vlmul2) {
  VsxsegXeiXX<UInt32, 1, 2>(
      0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
      {0x8302'8100'8706'8504, 0x8b0a'8908'8f0e'8d0c, 0x9716'9514'9b1a'9918, 0x9312'9110'9f1e'9d1c});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew32_vlmul4) {
  VsxsegXeiXX<UInt32, 1, 4>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8302'8100'8706'8504,
                             0x8b0a'8908'8f0e'8d0c,
                             0x9716'9514'9b1a'9918,
                             0x9312'9110'9f1e'9d1c,
                             0xa322'a120'bb3a'b938,
                             0xaf2e'ad2c'bf3e'bd3c,
                             0xb332'b130'b736'b534,
                             0xab2a'a928'a726'a524});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew32_vlmul8) {
  VsxsegXeiXX<UInt32, 1, 8>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8302'8100'8706'8504,
                             0x8b0a'8908'8f0e'8d0c,
                             0x9716'9514'9b1a'9918,
                             0x9312'9110'9f1e'9d1c,
                             0xa322'a120'bb3a'b938,
                             0xaf2e'ad2c'bf3e'bd3c,
                             0xb332'b130'b736'b534,
                             0xab2a'a928'a726'a524,
                             0xcb4a'c948'eb6a'e968,
                             0xdf5e'dd5c'd352'd150,
                             0xef6e'ed6c'fb7a'f978,
                             0xff7e'fd7c'cf4e'cd4c,
                             0xdb5a'd958'f776'f574,
                             0xf372'f170'd756'd554,
                             0xe362'e160'e766'e564,
                             0xc746'c544'c342'c140});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew32_vlmul1) {
  VsxsegXeiXX<UInt32, 2, 1>(
      0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
      {0x9716'9514'8706'8504, 0x9312'9110'8302'8100, 0x9f1e'9d1c'8f0e'8d0c, 0x9b1a'9918'8b0a'8908});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew32_vlmul2) {
  VsxsegXeiXX<UInt32, 2, 2>(0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0xa726'a524'8706'8504,
                             0xa322'a120'8302'8100,
                             0xaf2e'ad2c'8f0e'8d0c,
                             0xab2a'a928'8b0a'8908,
                             0xbb3a'b938'9b1a'9918,
                             0xb736'b534'9716'9514,
                             0xbf3e'bd3c'9f1e'9d1c,
                             0xb332'b130'9312'9110});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew32_vlmul4) {
  VsxsegXeiXX<UInt32, 2, 4>(0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0xc746'c544'8706'8504,
                             0xc342'c140'8302'8100,
                             0xcf4e'cd4c'8f0e'8d0c,
                             0xcb4a'c948'8b0a'8908,
                             0xdb5a'd958'9b1a'9918,
                             0xd756'd554'9716'9514,
                             0xdf5e'dd5c'9f1e'9d1c,
                             0xd352'd150'9312'9110,
                             0xfb7a'f978'bb3a'b938,
                             0xe362'e160'a322'a120,
                             0xff7e'fd7c'bf3e'bd3c,
                             0xef6e'ed6c'af2e'ad2c,
                             0xf776'f574'b736'b534,
                             0xf372'f170'b332'b130,
                             0xe766'e564'a726'a524,
                             0xeb6a'e968'ab2a'a928});
}

TEST_F(Riscv64InterpreterTest, Vsxseg3eiXX_sew32_vlmul1) {
  VsxsegXeiXX<UInt32, 3, 1>(0x4500'8427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0x8302'8100'a726'a524,
                             0xa322'a120'9312'9110,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0x8b0a'8908'af2e'ad2c,
                             0xab2a'a928'9b1a'9918});
}

TEST_F(Riscv64InterpreterTest, Vsxseg3eiXX_sew32_vlmul2) {
  VsxsegXeiXX<UInt32, 3, 2>(0x4500'8427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0xa726'a524'8706'8504,
                             0x8302'8100'c746'c544,
                             0xc342'c140'a322'a120,
                             0xaf2e'ad2c'8f0e'8d0c,
                             0x8b0a'8908'cf4e'cd4c,
                             0xcb4a'c948'ab2a'a928,
                             0xbb3a'b938'9b1a'9918,
                             0x9716'9514'db5a'd958,
                             0xd756'd554'b736'b534,
                             0xbf3e'bd3c'9f1e'9d1c,
                             0x9312'9110'df5e'dd5c,
                             0xd352'd150'b332'b130});
}

TEST_F(Riscv64InterpreterTest, Vsxseg4eiXX_sew32_vlmul1) {
  VsxsegXeiXX<UInt32, 4, 1>(0x6500'8427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0x9312'9110'8302'8100,
                             0xb332'b130'a322'a120,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0x9b1a'9918'8b0a'8908,
                             0xbb3a'b938'ab2a'a928});
}

TEST_F(Riscv64InterpreterTest, Vsxseg4eiXX_sew32_vlmul2) {
  VsxsegXeiXX<UInt32, 4, 2>(0x6500'8427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0xa726'a524'8706'8504,
                             0xe766'e564'c746'c544,
                             0xa322'a120'8302'8100,
                             0xe362'e160'c342'c140,
                             0xaf2e'ad2c'8f0e'8d0c,
                             0xef6e'ed6c'cf4e'cd4c,
                             0xab2a'a928'8b0a'8908,
                             0xeb6a'e968'cb4a'c948,
                             0xbb3a'b938'9b1a'9918,
                             0xfb7a'f978'db5a'd958,
                             0xb736'b534'9716'9514,
                             0xf776'f574'd756'd554,
                             0xbf3e'bd3c'9f1e'9d1c,
                             0xff7e'fd7c'df5e'dd5c,
                             0xb332'b130'9312'9110,
                             0xf372'f170'd352'd150});
}

TEST_F(Riscv64InterpreterTest, Vsxseg5eiXX_sew32) {
  VsxsegXeiXX<UInt32, 5, 1>(0x8500'8427,  // Vsuxseg5ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0x8302'8100'c746'c544,
                             0xa322'a120'9312'9110,
                             0xc342'c140'b332'b130,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0x8b0a'8908'cf4e'cd4c,
                             0xab2a'a928'9b1a'9918,
                             0xcb4a'c948'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, Vsxseg6eiXX_sew32) {
  VsxsegXeiXX<UInt32, 6, 1>(0xa500'8427,  // Vsuxseg6ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0xd756'd554'c746'c544,
                             0x9312'9110'8302'8100,
                             0xb332'b130'a322'a120,
                             0xd352'd150'c342'c140,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0xdf5e'dd5c'cf4e'cd4c,
                             0x9b1a'9918'8b0a'8908,
                             0xbb3a'b938'ab2a'a928,
                             0xdb5a'd958'cb4a'c948});
}

TEST_F(Riscv64InterpreterTest, Vsxseg7eiXX_sew32) {
  VsxsegXeiXX<UInt32, 7, 1>(0xc500'8427,  // Vsuxseg7ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0xd756'd554'c746'c544,
                             0x8302'8100'e766'e564,
                             0xa322'a120'9312'9110,
                             0xc342'c140'b332'b130,
                             0xe362'e160'd352'd150,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0xdf5e'dd5c'cf4e'cd4c,
                             0x8b0a'8908'ef6e'ed6c,
                             0xab2a'a928'9b1a'9918,
                             0xcb4a'c948'bb3a'b938,
                             0xeb6a'e968'db5a'd958});
}

TEST_F(Riscv64InterpreterTest, Vsxseg8eiXX_sew32) {
  VsxsegXeiXX<UInt32, 8, 1>(0xe500'8427,  // Vsuxseg8ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0xd756'd554'c746'c544,
                             0xf776'f574'e766'e564,
                             0x9312'9110'8302'8100,
                             0xb332'b130'a322'a120,
                             0xd352'd150'c342'c140,
                             0xf372'f170'e362'e160,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0xdf5e'dd5c'cf4e'cd4c,
                             0xff7e'fd7c'ef6e'ed6c,
                             0x9b1a'9918'8b0a'8908,
                             0xbb3a'b938'ab2a'a928,
                             0xdb5a'd958'cb4a'c948,
                             0xfb7a'f978'eb6a'e968});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew64_vlmul1) {
  VsxsegXeiXX<UInt64, 1, 1>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew64_vlmul2) {
  VsxsegXeiXX<UInt64, 1, 2>(
      0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
      {0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100, 0x9f1e'9d1c'9b1a'9918, 0x9716'9514'9312'9110});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew64_vlmul4) {
  VsxsegXeiXX<UInt64, 1, 4>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x8706'8504'8302'8100,
                             0x9f1e'9d1c'9b1a'9918,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xa726'a524'a322'a120});
}

TEST_F(Riscv64InterpreterTest, VsxeiXX_sew64_vlmul8) {
  VsxsegXeiXX<UInt64, 1, 8>(0x0500'8427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x8706'8504'8302'8100,
                             0x9f1e'9d1c'9b1a'9918,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xa726'a524'a322'a120,
                             0xf776'f574'f372'f170,
                             0xc746'c544'c342'c140,
                             0xff7e'fd7c'fb7a'f978,
                             0xdf5e'dd5c'db5a'd958,
                             0xef6e'ed6c'eb6a'e968,
                             0xe766'e564'e362'e160,
                             0xcf4e'cd4c'cb4a'c948,
                             0xd756'd554'd352'd150});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew64_vlmul1) {
  VsxsegXeiXX<UInt64, 2, 1>(
      0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
      {0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918, 0x8706'8504'8302'8100, 0x9716'9514'9312'9110});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew64_vlmul2) {
  VsxsegXeiXX<UInt64, 2, 2>(0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0xaf2e'ad2c'ab2a'a928,
                             0x8706'8504'8302'8100,
                             0xa726'a524'a322'a120,
                             0x9f1e'9d1c'9b1a'9918,
                             0xbf3e'bd3c'bb3a'b938,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130});
}

TEST_F(Riscv64InterpreterTest, Vsxseg2eiXX_sew64_vlmul4) {
  VsxsegXeiXX<UInt64, 2, 4>(0x2500'8427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0xcf4e'cd4c'cb4a'c948,
                             0x8706'8504'8302'8100,
                             0xc746'c544'c342'c140,
                             0x9f1e'9d1c'9b1a'9918,
                             0xdf5e'dd5c'db5a'd958,
                             0x9716'9514'9312'9110,
                             0xd756'd554'd352'd150,
                             0xb736'b534'b332'b130,
                             0xf776'f574'f372'f170,
                             0xaf2e'ad2c'ab2a'a928,
                             0xef6e'ed6c'eb6a'e968,
                             0xbf3e'bd3c'bb3a'b938,
                             0xff7e'fd7c'fb7a'f978,
                             0xa726'a524'a322'a120,
                             0xe766'e564'e362'e160});
}

TEST_F(Riscv64InterpreterTest, Vsxseg3eiXX_sew64_vlmul1) {
  VsxsegXeiXX<UInt64, 3, 1>(0x4500'8427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120});
}

TEST_F(Riscv64InterpreterTest, Vsxseg3eiXX_sew64_vlmul2) {
  VsxsegXeiXX<UInt64, 3, 2>(0x4500'8427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0xaf2e'ad2c'ab2a'a928,
                             0xcf4e'cd4c'cb4a'c948,
                             0x8706'8504'8302'8100,
                             0xa726'a524'a322'a120,
                             0xc746'c544'c342'c140,
                             0x9f1e'9d1c'9b1a'9918,
                             0xbf3e'bd3c'bb3a'b938,
                             0xdf5e'dd5c'db5a'd958,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130,
                             0xd756'd554'd352'd150});
}

TEST_F(Riscv64InterpreterTest, Vsxseg4eiXX_sew64_vlmul1) {
  VsxsegXeiXX<UInt64, 4, 1>(0x6500'8427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130});
}

TEST_F(Riscv64InterpreterTest, Vsxseg4eiXX_sew64_vlmul2) {
  VsxsegXeiXX<UInt64, 4, 2>(0x6500'8427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0xaf2e'ad2c'ab2a'a928,
                             0xcf4e'cd4c'cb4a'c948,
                             0xef6e'ed6c'eb6a'e968,
                             0x8706'8504'8302'8100,
                             0xa726'a524'a322'a120,
                             0xc746'c544'c342'c140,
                             0xe766'e564'e362'e160,
                             0x9f1e'9d1c'9b1a'9918,
                             0xbf3e'bd3c'bb3a'b938,
                             0xdf5e'dd5c'db5a'd958,
                             0xff7e'fd7c'fb7a'f978,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130,
                             0xd756'd554'd352'd150,
                             0xf776'f574'f372'f170});
}

TEST_F(Riscv64InterpreterTest, Vsxseg5eiXX_sew64) {
  VsxsegXeiXX<UInt64, 5, 1>(0x8500'8427,  // Vsuxseg5ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xcf4e'cd4c'cb4a'c948,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130,
                             0xc746'c544'c342'c140});
}

TEST_F(Riscv64InterpreterTest, Vsxseg6eiXX_sew64) {
  VsxsegXeiXX<UInt64, 6, 1>(0xa500'8427,  // Vsuxseg6ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xcf4e'cd4c'cb4a'c948,
                             0xdf5e'dd5c'db5a'd958,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130,
                             0xc746'c544'c342'c140,
                             0xd756'd554'd352'd150});
}

TEST_F(Riscv64InterpreterTest, Vsxseg7eiXX_sew64) {
  VsxsegXeiXX<UInt64, 7, 1>(0xc500'8427,  // Vsuxseg7ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xcf4e'cd4c'cb4a'c948,
                             0xdf5e'dd5c'db5a'd958,
                             0xef6e'ed6c'eb6a'e968,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130,
                             0xc746'c544'c342'c140,
                             0xd756'd554'd352'd150,
                             0xe766'e564'e362'e160});
}

TEST_F(Riscv64InterpreterTest, Vsxseg8eiXX_sew64) {
  VsxsegXeiXX<UInt64, 8, 1>(0xe500'8427,  // Vsuxseg8ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xcf4e'cd4c'cb4a'c948,
                             0xdf5e'dd5c'db5a'd958,
                             0xef6e'ed6c'eb6a'e968,
                             0xff7e'fd7c'fb7a'f978,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130,
                             0xc746'c544'c342'c140,
                             0xd756'd554'd352'd150,
                             0xe766'e564'e362'e160,
                             0xf776'f574'f372'f170});
}

TEST_F(Riscv64InterpreterTest, TestVse8_vlmul1) {
  TestVssegXeXX<UInt8, 1, 1>(0x0000'8427,  // vsse8.v v8, (x1), v0.t
                             {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908});
}

TEST_F(Riscv64InterpreterTest, TestVse8_vlmul2) {
  TestVssegXeXX<UInt8, 1, 2>(
      0x0000'8427,  // vsse8.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908, 0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918});
}

TEST_F(Riscv64InterpreterTest, TestVse8_vlmul4) {
  TestVssegXeXX<UInt8, 1, 4>(0x0000'8427,  // vsse8.v v8, (x1), v0.t
                             {0x8706'8504'8302'8100,
                              0x8f0e'8d0c'8b0a'8908,
                              0x9716'9514'9312'9110,
                              0x9f1e'9d1c'9b1a'9918,
                              0xa726'a524'a322'a120,
                              0xaf2e'ad2c'ab2a'a928,
                              0xb736'b534'b332'b130,
                              0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVse8_vlmul8) {
  TestVssegXeXX<UInt8, 1, 8>(0x0000'8427,  // vsse8.v v8, (x1), v0.t
                             {0x8706'8504'8302'8100,
                              0x8f0e'8d0c'8b0a'8908,
                              0x9716'9514'9312'9110,
                              0x9f1e'9d1c'9b1a'9918,
                              0xa726'a524'a322'a120,
                              0xaf2e'ad2c'ab2a'a928,
                              0xb736'b534'b332'b130,
                              0xbf3e'bd3c'bb3a'b938,
                              0xc746'c544'c342'c140,
                              0xcf4e'cd4c'cb4a'c948,
                              0xd756'd554'd352'd150,
                              0xdf5e'dd5c'db5a'd958,
                              0xe766'e564'e362'e160,
                              0xef6e'ed6c'eb6a'e968,
                              0xf776'f574'f372'f170,
                              0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e8_vlmul1) {
  TestVssegXeXX<UInt8, 2, 1>(
      0x2000'8427,  // vsseg2e8.v v8, (x1), v0.t
      {0x9383'1202'9181'1000, 0x9787'1606'9585'1404, 0x9b8b'1a0a'9989'1808, 0x9f8f'1e0e'9d8d'1c0c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e8_vlmul2) {
  TestVssegXeXX<UInt8, 2, 2>(0x2000'8427,  // vsseg2e8.v v8, (x1), v0.t
                             {0xa383'2202'a181'2000,
                              0xa787'2606'a585'2404,
                              0xab8b'2a0a'a989'2808,
                              0xaf8f'2e0e'ad8d'2c0c,
                              0xb393'3212'b191'3010,
                              0xb797'3616'b595'3414,
                              0xbb9b'3a1a'b999'3818,
                              0xbf9f'3e1e'bd9d'3c1c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e8_vlmul4) {
  TestVssegXeXX<UInt8, 2, 4>(0x2000'8427,  // vsseg2e8.v v8, (x1), v0.t
                             {0xc383'4202'c181'4000,
                              0xc787'4606'c585'4404,
                              0xcb8b'4a0a'c989'4808,
                              0xcf8f'4e0e'cd8d'4c0c,
                              0xd393'5212'd191'5010,
                              0xd797'5616'd595'5414,
                              0xdb9b'5a1a'd999'5818,
                              0xdf9f'5e1e'dd9d'5c1c,
                              0xe3a3'6222'e1a1'6020,
                              0xe7a7'6626'e5a5'6424,
                              0xebab'6a2a'e9a9'6828,
                              0xefaf'6e2e'edad'6c2c,
                              0xf3b3'7232'f1b1'7030,
                              0xf7b7'7636'f5b5'7434,
                              0xfbbb'7a3a'f9b9'7838,
                              0xffbf'7e3e'fdbd'7c3c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg3e8_vlmul1) {
  TestVssegXeXX<UInt8, 3, 1>(0x4000'8427,  // vsseg3e8.v v8, (x1), v0.t
                             {0x1202'a191'8120'1000,
                              0x8524'1404'a393'8322,
                              0xa797'8726'1606'a595,
                              0x1a0a'a999'8928'1808,
                              0x8d2c'1c0c'ab9b'8b2a,
                              0xaf9f'8f2e'1e0e'ad9d});
}

TEST_F(Riscv64InterpreterTest, TestVsseg3e8_vlmul2) {
  TestVssegXeXX<UInt8, 3, 2>(0x4000'8427,  // vsseg3e8.v v8, (x1), v0.t
                             {0x2202'c1a1'8140'2000,
                              0x8544'2404'c3a3'8342,
                              0xc7a7'8746'2606'c5a5,
                              0x2a0a'c9a9'8948'2808,
                              0x8d4c'2c0c'cbab'8b4a,
                              0xcfaf'8f4e'2e0e'cdad,
                              0x3212'd1b1'9150'3010,
                              0x9554'3414'd3b3'9352,
                              0xd7b7'9756'3616'd5b5,
                              0x3a1a'd9b9'9958'3818,
                              0x9d5c'3c1c'dbbb'9b5a,
                              0xdfbf'9f5e'3e1e'ddbd});
}

TEST_F(Riscv64InterpreterTest, TestVsseg4e8_vlmul1) {
  TestVssegXeXX<UInt8, 4, 1>(0x6000'8427,  // vsseg4e8.v v8, (x1), v0.t
                             {0xb1a1'9181'3020'1000,
                              0xb3a3'9383'3222'1202,
                              0xb5a5'9585'3424'1404,
                              0xb7a7'9787'3626'1606,
                              0xb9a9'9989'3828'1808,
                              0xbbab'9b8b'3a2a'1a0a,
                              0xbdad'9d8d'3c2c'1c0c,
                              0xbfaf'9f8f'3e2e'1e0e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg4e8_vlmul2) {
  TestVssegXeXX<UInt8, 4, 2>(0x6000'8427,  // vsseg4e8.v v8, (x1), v0.t
                             {0xe1c1'a181'6040'2000,
                              0xe3c3'a383'6242'2202,
                              0xe5c5'a585'6444'2404,
                              0xe7c7'a787'6646'2606,
                              0xe9c9'a989'6848'2808,
                              0xebcb'ab8b'6a4a'2a0a,
                              0xedcd'ad8d'6c4c'2c0c,
                              0xefcf'af8f'6e4e'2e0e,
                              0xf1d1'b191'7050'3010,
                              0xf3d3'b393'7252'3212,
                              0xf5d5'b595'7454'3414,
                              0xf7d7'b797'7656'3616,
                              0xf9d9'b999'7858'3818,
                              0xfbdb'bb9b'7a5a'3a1a,
                              0xfddd'bd9d'7c5c'3c1c,
                              0xffdf'bf9f'7e5e'3e1e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg5e8) {
  TestVssegXeXX<UInt8, 5, 1>(0x8000'8427,  // vsseg5e8.v v8, (x1), v0.t
                             {0xa191'8140'3020'1000,
                              0x8342'3222'1202'c1b1,
                              0x3424'1404'c3b3'a393,
                              0x1606'c5b5'a595'8544,
                              0xc7b7'a797'8746'3626,
                              0xa999'8948'3828'1808,
                              0x8b4a'3a2a'1a0a'c9b9,
                              0x3c2c'1c0c'cbbb'ab9b,
                              0x1e0e'cdbd'ad9d'8d4c,
                              0xcfbf'af9f'8f4e'3e2e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg6e8) {
  TestVssegXeXX<UInt8, 6, 1>(0xa000'8427,  // vsseg6e8.v v8, (x1), v0.t
                             {0x9181'5040'3020'1000,
                              0x3222'1202'd1c1'b1a1,
                              0xd3c3'b3a3'9383'5242,
                              0x9585'5444'3424'1404,
                              0x3626'1606'd5c5'b5a5,
                              0xd7c7'b7a7'9787'5646,
                              0x9989'5848'3828'1808,
                              0x3a2a'1a0a'd9c9'b9a9,
                              0xdbcb'bbab'9b8b'5a4a,
                              0x9d8d'5c4c'3c2c'1c0c,
                              0x3e2e'1e0e'ddcd'bdad,
                              0xdfcf'bfaf'9f8f'5e4e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg7e8) {
  TestVssegXeXX<UInt8, 7, 1>(0xc000'8427,  // vsseg7e8.v v8, (x1), v0.t
                             {0x8160'5040'3020'1000,
                              0x1202'e1d1'c1b1'a191,
                              0xa393'8362'5242'3222,
                              0x3424'1404'e3d3'c3b3,
                              0xc5b5'a595'8564'5444,
                              0x5646'3626'1606'e5d5,
                              0xe7d7'c7b7'a797'8766,
                              0x8968'5848'3828'1808,
                              0x1a0a'e9d9'c9b9'a999,
                              0xab9b'8b6a'5a4a'3a2a,
                              0x3c2c'1c0c'ebdb'cbbb,
                              0xcdbd'ad9d'8d6c'5c4c,
                              0x5e4e'3e2e'1e0e'eddd,
                              0xefdf'cfbf'af9f'8f6e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg8e8) {
  TestVssegXeXX<UInt8, 8, 1>(0xe000'8427,  // vsseg8e8.v v8, (x1), v0.t
                             {0x7060'5040'3020'1000,
                              0xf1e1'd1c1'b1a1'9181,
                              0x7262'5242'3222'1202,
                              0xf3e3'd3c3'b3a3'9383,
                              0x7464'5444'3424'1404,
                              0xf5e5'd5c5'b5a5'9585,
                              0x7666'5646'3626'1606,
                              0xf7e7'd7c7'b7a7'9787,
                              0x7868'5848'3828'1808,
                              0xf9e9'd9c9'b9a9'9989,
                              0x7a6a'5a4a'3a2a'1a0a,
                              0xfbeb'dbcb'bbab'9b8b,
                              0x7c6c'5c4c'3c2c'1c0c,
                              0xfded'ddcd'bdad'9d8d,
                              0x7e6e'5e4e'3e2e'1e0e,
                              0xffef'dfcf'bfaf'9f8f});
}

TEST_F(Riscv64InterpreterTest, TestVse16_vlmul1) {
  TestVssegXeXX<UInt16, 1, 1>(0x0000'd427,  // vse16.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908});
}

TEST_F(Riscv64InterpreterTest, TestVse16_vlmul2) {
  TestVssegXeXX<UInt16, 1, 2>(
      0x0000'd427,  // vse16.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908, 0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918});
}

TEST_F(Riscv64InterpreterTest, TestVse16_vlmul4) {
  TestVssegXeXX<UInt16, 1, 4>(0x0000'd427,  // vse16.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVse16_vlmul8) {
  TestVssegXeXX<UInt16, 1, 8>(0x0000'd427,  // vse16.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938,
                               0xc746'c544'c342'c140,
                               0xcf4e'cd4c'cb4a'c948,
                               0xd756'd554'd352'd150,
                               0xdf5e'dd5c'db5a'd958,
                               0xe766'e564'e362'e160,
                               0xef6e'ed6c'eb6a'e968,
                               0xf776'f574'f372'f170,
                               0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e16_vlmul1) {
  TestVssegXeXX<UInt16, 2, 1>(
      0x2000'd427,  // vsseg2e16.v v8, (x1), v0.t
      {0x9312'8302'9110'8100, 0x9716'8706'9514'8504, 0x9b1a'8b0a'9918'8908, 0x9f1e'8f0e'9d1c'8d0c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e16_vlmul2) {
  TestVssegXeXX<UInt16, 2, 2>(0x2000'd427,  // vsseg2e16.v v8, (x1), v0.t
                              {0xa322'8302'a120'8100,
                               0xa726'8706'a524'8504,
                               0xab2a'8b0a'a928'8908,
                               0xaf2e'8f0e'ad2c'8d0c,
                               0xb332'9312'b130'9110,
                               0xb736'9716'b534'9514,
                               0xbb3a'9b1a'b938'9918,
                               0xbf3e'9f1e'bd3c'9d1c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e16_vlmul4) {
  TestVssegXeXX<UInt16, 2, 4>(0x2000'd427,  // vsseg2e16.v v8, (x1), v0.t
                              {0xc342'8302'c140'8100,
                               0xc746'8706'c544'8504,
                               0xcb4a'8b0a'c948'8908,
                               0xcf4e'8f0e'cd4c'8d0c,
                               0xd352'9312'd150'9110,
                               0xd756'9716'd554'9514,
                               0xdb5a'9b1a'd958'9918,
                               0xdf5e'9f1e'dd5c'9d1c,
                               0xe362'a322'e160'a120,
                               0xe766'a726'e564'a524,
                               0xeb6a'ab2a'e968'a928,
                               0xef6e'af2e'ed6c'ad2c,
                               0xf372'b332'f170'b130,
                               0xf776'b736'f574'b534,
                               0xfb7a'bb3a'f978'b938,
                               0xff7e'bf3e'fd7c'bd3c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg3e16_vlmul1) {
  TestVssegXeXX<UInt16, 3, 1>(0x4000'd427,  // vsseg3e16.v v8, (x1), v0.t
                              {0x8302'a120'9110'8100,
                               0x9514'8504'a322'9312,
                               0xa726'9716'8706'a524,
                               0x8b0a'a928'9918'8908,
                               0x9d1c'8d0c'ab2a'9b1a,
                               0xaf2e'9f1e'8f0e'ad2c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg3e16_vlmul2) {
  TestVssegXeXX<UInt16, 3, 2>(0x4000'd427,  // vsseg3e16.v v8, (x1), v0.t
                              {0x8302'c140'a120'8100,
                               0xa524'8504'c342'a322,
                               0xc746'a726'8706'c544,
                               0x8b0a'c948'a928'8908,
                               0xad2c'8d0c'cb4a'ab2a,
                               0xcf4e'af2e'8f0e'cd4c,
                               0x9312'd150'b130'9110,
                               0xb534'9514'd352'b332,
                               0xd756'b736'9716'd554,
                               0x9b1a'd958'b938'9918,
                               0xbd3c'9d1c'db5a'bb3a,
                               0xdf5e'bf3e'9f1e'dd5c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg4e16_vlmul1) {
  TestVssegXeXX<UInt16, 4, 1>(0x6000'd427,  // vsseg4e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0xb332'a322'9312'8302,
                               0xb534'a524'9514'8504,
                               0xb736'a726'9716'8706,
                               0xb938'a928'9918'8908,
                               0xbb3a'ab2a'9b1a'8b0a,
                               0xbd3c'ad2c'9d1c'8d0c,
                               0xbf3e'af2e'9f1e'8f0e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg4e16_vlmul2) {
  TestVssegXeXX<UInt16, 4, 2>(0x6000'd427,  // vsseg4e16.v v8, (x1), v0.t
                              {0xe160'c140'a120'8100,
                               0xe362'c342'a322'8302,
                               0xe564'c544'a524'8504,
                               0xe766'c746'a726'8706,
                               0xe968'c948'a928'8908,
                               0xeb6a'cb4a'ab2a'8b0a,
                               0xed6c'cd4c'ad2c'8d0c,
                               0xef6e'cf4e'af2e'8f0e,
                               0xf170'd150'b130'9110,
                               0xf372'd352'b332'9312,
                               0xf574'd554'b534'9514,
                               0xf776'd756'b736'9716,
                               0xf978'd958'b938'9918,
                               0xfb7a'db5a'bb3a'9b1a,
                               0xfd7c'dd5c'bd3c'9d1c,
                               0xff7e'df5e'bf3e'9f1e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg5e16) {
  TestVssegXeXX<UInt16, 5, 1>(0x8000'd427,  // vsseg5e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0xa322'9312'8302'c140,
                               0x9514'8504'c342'b332,
                               0x8706'c544'b534'a524,
                               0xc746'b736'a726'9716,
                               0xb938'a928'9918'8908,
                               0xab2a'9b1a'8b0a'c948,
                               0x9d1c'8d0c'cb4a'bb3a,
                               0x8f0e'cd4c'bd3c'ad2c,
                               0xcf4e'bf3e'af2e'9f1e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg6e16) {
  TestVssegXeXX<UInt16, 6, 1>(0xa000'd427,  // vsseg6e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0x9312'8302'd150'c140,
                               0xd352'c342'b332'a322,
                               0xb534'a524'9514'8504,
                               0x9716'8706'd554'c544,
                               0xd756'c746'b736'a726,
                               0xb938'a928'9918'8908,
                               0x9b1a'8b0a'd958'c948,
                               0xdb5a'cb4a'bb3a'ab2a,
                               0xbd3c'ad2c'9d1c'8d0c,
                               0x9f1e'8f0e'dd5c'cd4c,
                               0xdf5e'cf4e'bf3e'af2e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg7e16) {
  TestVssegXeXX<UInt16, 7, 1>(0xc000'd427,  // vsseg7e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0x8302'e160'd150'c140,
                               0xc342'b332'a322'9312,
                               0x9514'8504'e362'd352,
                               0xd554'c544'b534'a524,
                               0xa726'9716'8706'e564,
                               0xe766'd756'c746'b736,
                               0xb938'a928'9918'8908,
                               0x8b0a'e968'd958'c948,
                               0xcb4a'bb3a'ab2a'9b1a,
                               0x9d1c'8d0c'eb6a'db5a,
                               0xdd5c'cd4c'bd3c'ad2c,
                               0xaf2e'9f1e'8f0e'ed6c,
                               0xef6e'df5e'cf4e'bf3e});
}

TEST_F(Riscv64InterpreterTest, TestVsseg8e16) {
  TestVssegXeXX<UInt16, 8, 1>(0xe000'd427,  // vsseg8e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0xf170'e160'd150'c140,
                               0xb332'a322'9312'8302,
                               0xf372'e362'd352'c342,
                               0xb534'a524'9514'8504,
                               0xf574'e564'd554'c544,
                               0xb736'a726'9716'8706,
                               0xf776'e766'd756'c746,
                               0xb938'a928'9918'8908,
                               0xf978'e968'd958'c948,
                               0xbb3a'ab2a'9b1a'8b0a,
                               0xfb7a'eb6a'db5a'cb4a,
                               0xbd3c'ad2c'9d1c'8d0c,
                               0xfd7c'ed6c'dd5c'cd4c,
                               0xbf3e'af2e'9f1e'8f0e,
                               0xff7e'ef6e'df5e'cf4e});
}

TEST_F(Riscv64InterpreterTest, TestVse32_vlmul1) {
  TestVssegXeXX<UInt32, 1, 1>(0x0000'e427,  // vse32.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908});
}

TEST_F(Riscv64InterpreterTest, TestVse32_vlmul2) {
  TestVssegXeXX<UInt32, 1, 2>(
      0x0000'e427,  // vse32.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908, 0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918});
}

TEST_F(Riscv64InterpreterTest, TestVse32_vlmul4) {
  TestVssegXeXX<UInt32, 1, 4>(0x0000'e427,  // vse32.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVse32_vlmul8) {
  TestVssegXeXX<UInt32, 1, 8>(0x0000'e427,  // vse32.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938,
                               0xc746'c544'c342'c140,
                               0xcf4e'cd4c'cb4a'c948,
                               0xd756'd554'd352'd150,
                               0xdf5e'dd5c'db5a'd958,
                               0xe766'e564'e362'e160,
                               0xef6e'ed6c'eb6a'e968,
                               0xf776'f574'f372'f170,
                               0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e32_vlmul1) {
  TestVssegXeXX<UInt32, 2, 1>(
      0x2000'e427,  // vsseg2e32.v v8, (x1), v0.t
      {0x9312'9110'8302'8100, 0x9716'9514'8706'8504, 0x9b1a'9918'8b0a'8908, 0x9f1e'9d1c'8f0e'8d0c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e32_vlmul2) {
  TestVssegXeXX<UInt32, 2, 2>(0x2000'e427,  // vsseg2e32.v v8, (x1), v0.t
                              {0xa322'a120'8302'8100,
                               0xa726'a524'8706'8504,
                               0xab2a'a928'8b0a'8908,
                               0xaf2e'ad2c'8f0e'8d0c,
                               0xb332'b130'9312'9110,
                               0xb736'b534'9716'9514,
                               0xbb3a'b938'9b1a'9918,
                               0xbf3e'bd3c'9f1e'9d1c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e32_vlmul4) {
  TestVssegXeXX<UInt32, 2, 4>(0x2000'e427,  // vsseg2e32.v v8, (x1), v0.t
                              {0xc342'c140'8302'8100,
                               0xc746'c544'8706'8504,
                               0xcb4a'c948'8b0a'8908,
                               0xcf4e'cd4c'8f0e'8d0c,
                               0xd352'd150'9312'9110,
                               0xd756'd554'9716'9514,
                               0xdb5a'd958'9b1a'9918,
                               0xdf5e'dd5c'9f1e'9d1c,
                               0xe362'e160'a322'a120,
                               0xe766'e564'a726'a524,
                               0xeb6a'e968'ab2a'a928,
                               0xef6e'ed6c'af2e'ad2c,
                               0xf372'f170'b332'b130,
                               0xf776'f574'b736'b534,
                               0xfb7a'f978'bb3a'b938,
                               0xff7e'fd7c'bf3e'bd3c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg3e32_vlmul1) {
  TestVssegXeXX<UInt32, 3, 1>(0x4000'e427,  // vsseg3e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0x8706'8504'a322'a120,
                               0xa726'a524'9716'9514,
                               0x9b1a'9918'8b0a'8908,
                               0x8f0e'8d0c'ab2a'a928,
                               0xaf2e'ad2c'9f1e'9d1c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg3e32_vlmul2) {
  TestVssegXeXX<UInt32, 3, 2>(0x4000'e427,  // vsseg3e32.v v8, (x1), v0.t
                              {0xa322'a120'8302'8100,
                               0x8706'8504'c342'c140,
                               0xc746'c544'a726'a524,
                               0xab2a'a928'8b0a'8908,
                               0x8f0e'8d0c'cb4a'c948,
                               0xcf4e'cd4c'af2e'ad2c,
                               0xb332'b130'9312'9110,
                               0x9716'9514'd352'd150,
                               0xd756'd554'b736'b534,
                               0xbb3a'b938'9b1a'9918,
                               0x9f1e'9d1c'db5a'd958,
                               0xdf5e'dd5c'bf3e'bd3c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg4e32_vlmul1) {
  TestVssegXeXX<UInt32, 4, 1>(0x6000'e427,  // vsseg4e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0x9716'9514'8706'8504,
                               0xb736'b534'a726'a524,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0x9f1e'9d1c'8f0e'8d0c,
                               0xbf3e'bd3c'af2e'ad2c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg4e32_vlmul2) {
  TestVssegXeXX<UInt32, 4, 2>(0x6000'e427,  // vsseg4e32.v v8, (x1), v0.t
                              {0xa322'a120'8302'8100,
                               0xe362'e160'c342'c140,
                               0xa726'a524'8706'8504,
                               0xe766'e564'c746'c544,
                               0xab2a'a928'8b0a'8908,
                               0xeb6a'e968'cb4a'c948,
                               0xaf2e'ad2c'8f0e'8d0c,
                               0xef6e'ed6c'cf4e'cd4c,
                               0xb332'b130'9312'9110,
                               0xf372'f170'd352'd150,
                               0xb736'b534'9716'9514,
                               0xf776'f574'd756'd554,
                               0xbb3a'b938'9b1a'9918,
                               0xfb7a'f978'db5a'd958,
                               0xbf3e'bd3c'9f1e'9d1c,
                               0xff7e'fd7c'df5e'dd5c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg5e32) {
  TestVssegXeXX<UInt32, 5, 1>(0x8000'e427,  // vsseg5e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0x8706'8504'c342'c140,
                               0xa726'a524'9716'9514,
                               0xc746'c544'b736'b534,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0x8f0e'8d0c'cb4a'c948,
                               0xaf2e'ad2c'9f1e'9d1c,
                               0xcf4e'cd4c'bf3e'bd3c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg6e32) {
  TestVssegXeXX<UInt32, 6, 1>(0xa000'e427,  // vsseg6e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0xd352'd150'c342'c140,
                               0x9716'9514'8706'8504,
                               0xb736'b534'a726'a524,
                               0xd756'd554'c746'c544,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0xdb5a'd958'cb4a'c948,
                               0x9f1e'9d1c'8f0e'8d0c,
                               0xbf3e'bd3c'af2e'ad2c,
                               0xdf5e'dd5c'cf4e'cd4c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg7e32) {
  TestVssegXeXX<UInt32, 7, 1>(0xc000'e427,  // vsseg7e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0xd352'd150'c342'c140,
                               0x8706'8504'e362'e160,
                               0xa726'a524'9716'9514,
                               0xc746'c544'b736'b534,
                               0xe766'e564'd756'd554,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0xdb5a'd958'cb4a'c948,
                               0x8f0e'8d0c'eb6a'e968,
                               0xaf2e'ad2c'9f1e'9d1c,
                               0xcf4e'cd4c'bf3e'bd3c,
                               0xef6e'ed6c'df5e'dd5c});
}

TEST_F(Riscv64InterpreterTest, TestVsseg8e32) {
  TestVssegXeXX<UInt32, 8, 1>(0xe000'e427,  // vsseg8e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0xd352'd150'c342'c140,
                               0xf372'f170'e362'e160,
                               0x9716'9514'8706'8504,
                               0xb736'b534'a726'a524,
                               0xd756'd554'c746'c544,
                               0xf776'f574'e766'e564,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0xdb5a'd958'cb4a'c948,
                               0xfb7a'f978'eb6a'e968,
                               0x9f1e'9d1c'8f0e'8d0c,
                               0xbf3e'bd3c'af2e'ad2c,
                               0xdf5e'dd5c'cf4e'cd4c,
                               0xff7e'fd7c'ef6e'ed6c});
}

TEST_F(Riscv64InterpreterTest, TestVse64_vlmul1) {
  TestVssegXeXX<UInt64, 1, 1>(0x0000'f427,  // vse64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908});
}

TEST_F(Riscv64InterpreterTest, TestVse64_vlmul2) {
  TestVssegXeXX<UInt64, 1, 2>(
      0x0000'f427,  // vse64.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908, 0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918});
}

TEST_F(Riscv64InterpreterTest, TestVse64_vlmul4) {
  TestVssegXeXX<UInt64, 1, 4>(0x0000'f427,  // vse64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVse64_vlmul8) {
  TestVssegXeXX<UInt64, 1, 8>(0x0000'f427,  // vse64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938,
                               0xc746'c544'c342'c140,
                               0xcf4e'cd4c'cb4a'c948,
                               0xd756'd554'd352'd150,
                               0xdf5e'dd5c'db5a'd958,
                               0xe766'e564'e362'e160,
                               0xef6e'ed6c'eb6a'e968,
                               0xf776'f574'f372'f170,
                               0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e64_vlmul1) {
  TestVssegXeXX<UInt64, 2, 1>(
      0x2000'f427,  // vsseg2e64.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x9716'9514'9312'9110, 0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e64_vlmul2) {
  TestVssegXeXX<UInt64, 2, 2>(0x2000'f427,  // vsseg2e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0xa726'a524'a322'a120,
                               0x8f0e'8d0c'8b0a'8908,
                               0xaf2e'ad2c'ab2a'a928,
                               0x9716'9514'9312'9110,
                               0xb736'b534'b332'b130,
                               0x9f1e'9d1c'9b1a'9918,
                               0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVsseg2e64_vlmul4) {
  TestVssegXeXX<UInt64, 2, 4>(0x2000'f427,  // vsseg2e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0xc746'c544'c342'c140,
                               0x8f0e'8d0c'8b0a'8908,
                               0xcf4e'cd4c'cb4a'c948,
                               0x9716'9514'9312'9110,
                               0xd756'd554'd352'd150,
                               0x9f1e'9d1c'9b1a'9918,
                               0xdf5e'dd5c'db5a'd958,
                               0xa726'a524'a322'a120,
                               0xe766'e564'e362'e160,
                               0xaf2e'ad2c'ab2a'a928,
                               0xef6e'ed6c'eb6a'e968,
                               0xb736'b534'b332'b130,
                               0xf776'f574'f372'f170,
                               0xbf3e'bd3c'bb3a'b938,
                               0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsseg3e64_vlmul1) {
  TestVssegXeXX<UInt64, 3, 1>(0x4000'f427,  // vsseg3e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928});
}

TEST_F(Riscv64InterpreterTest, TestVsseg3e64_vlmul2) {
  TestVssegXeXX<UInt64, 3, 2>(0x4000'f427,  // vsseg3e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0xa726'a524'a322'a120,
                               0xc746'c544'c342'c140,
                               0x8f0e'8d0c'8b0a'8908,
                               0xaf2e'ad2c'ab2a'a928,
                               0xcf4e'cd4c'cb4a'c948,
                               0x9716'9514'9312'9110,
                               0xb736'b534'b332'b130,
                               0xd756'd554'd352'd150,
                               0x9f1e'9d1c'9b1a'9918,
                               0xbf3e'bd3c'bb3a'b938,
                               0xdf5e'dd5c'db5a'd958});
}

TEST_F(Riscv64InterpreterTest, TestVsseg4e64_vlmul1) {
  TestVssegXeXX<UInt64, 4, 1>(0x6000'f427,  // vsseg4e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVsseg4e64_vlmul2) {
  TestVssegXeXX<UInt64, 4, 2>(0x6000'f427,  // vsseg4e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0xa726'a524'a322'a120,
                               0xc746'c544'c342'c140,
                               0xe766'e564'e362'e160,
                               0x8f0e'8d0c'8b0a'8908,
                               0xaf2e'ad2c'ab2a'a928,
                               0xcf4e'cd4c'cb4a'c948,
                               0xef6e'ed6c'eb6a'e968,
                               0x9716'9514'9312'9110,
                               0xb736'b534'b332'b130,
                               0xd756'd554'd352'd150,
                               0xf776'f574'f372'f170,
                               0x9f1e'9d1c'9b1a'9918,
                               0xbf3e'bd3c'bb3a'b938,
                               0xdf5e'dd5c'db5a'd958,
                               0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsseg5e64) {
  TestVssegXeXX<UInt64, 5, 1>(0x8000'f427,  // vsseg5e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0xc746'c544'c342'c140,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938,
                               0xcf4e'cd4c'cb4a'c948});
}

TEST_F(Riscv64InterpreterTest, TestVsseg6e64) {
  TestVssegXeXX<UInt64, 6, 1>(0xa000'f427,  // vsseg6e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0xc746'c544'c342'c140,
                               0xd756'd554'd352'd150,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938,
                               0xcf4e'cd4c'cb4a'c948,
                               0xdf5e'dd5c'db5a'd958});
}

TEST_F(Riscv64InterpreterTest, TestVsseg7e64) {
  TestVssegXeXX<UInt64, 7, 1>(0xc000'f427,  // vsseg7e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0xc746'c544'c342'c140,
                               0xd756'd554'd352'd150,
                               0xe766'e564'e362'e160,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938,
                               0xcf4e'cd4c'cb4a'c948,
                               0xdf5e'dd5c'db5a'd958,
                               0xef6e'ed6c'eb6a'e968});
}

TEST_F(Riscv64InterpreterTest, TestVsseg8e64) {
  TestVssegXeXX<UInt64, 8, 1>(0xe000'f427,  // vsseg8e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0xc746'c544'c342'c140,
                               0xd756'd554'd352'd150,
                               0xe766'e564'e362'e160,
                               0xf776'f574'f372'f170,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938,
                               0xcf4e'cd4c'cb4a'c948,
                               0xdf5e'dd5c'db5a'd958,
                               0xef6e'ed6c'eb6a'e968,
                               0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsse8_vlmul1) {
  TestVsssegXeXX<UInt8, 1, 1>(0x0820'8427,  // vsse8.v v8, (x1), x2, v0.t
                              4,
                              {0x5555'5581'5555'5500,
                               0x5555'5583'5555'5502,
                               0x5555'5585'5555'5504,
                               0x5555'5587'5555'5506,
                               0x5555'5589'5555'5508,
                               0x5555'558b'5555'550a,
                               0x5555'558d'5555'550c,
                               0x5555'558f'5555'550e});
}

TEST_F(Riscv64InterpreterTest, TestVsse8_vlmul2) {
  TestVsssegXeXX<UInt8, 1, 2>(0x0820'8427,  // vsse8.v v8, (x1), x2, v0.t
                              4,
                              {0x5555'5581'5555'5500,
                               0x5555'5583'5555'5502,
                               0x5555'5585'5555'5504,
                               0x5555'5587'5555'5506,
                               0x5555'5589'5555'5508,
                               0x5555'558b'5555'550a,
                               0x5555'558d'5555'550c,
                               0x5555'558f'5555'550e,
                               0x5555'5591'5555'5510,
                               0x5555'5593'5555'5512,
                               0x5555'5595'5555'5514,
                               0x5555'5597'5555'5516,
                               0x5555'5599'5555'5518,
                               0x5555'559b'5555'551a,
                               0x5555'559d'5555'551c,
                               0x5555'559f'5555'551e});
}

TEST_F(Riscv64InterpreterTest, TestVsse8_vlmul4) {
  TestVsssegXeXX<UInt8, 1, 4>(
      0x0820'8427,  // vlse8.v v8, (x1), x2, v0.t
      4,
      {0x5555'5581'5555'5500, 0x5555'5583'5555'5502, 0x5555'5585'5555'5504, 0x5555'5587'5555'5506,
       0x5555'5589'5555'5508, 0x5555'558b'5555'550a, 0x5555'558d'5555'550c, 0x5555'558f'5555'550e,
       0x5555'5591'5555'5510, 0x5555'5593'5555'5512, 0x5555'5595'5555'5514, 0x5555'5597'5555'5516,
       0x5555'5599'5555'5518, 0x5555'559b'5555'551a, 0x5555'559d'5555'551c, 0x5555'559f'5555'551e,
       0x5555'55a1'5555'5520, 0x5555'55a3'5555'5522, 0x5555'55a5'5555'5524, 0x5555'55a7'5555'5526,
       0x5555'55a9'5555'5528, 0x5555'55ab'5555'552a, 0x5555'55ad'5555'552c, 0x5555'55af'5555'552e,
       0x5555'55b1'5555'5530, 0x5555'55b3'5555'5532, 0x5555'55b5'5555'5534, 0x5555'55b7'5555'5536,
       0x5555'55b9'5555'5538, 0x5555'55bb'5555'553a, 0x5555'55bd'5555'553c, 0x5555'55bf'5555'553e});
}

TEST_F(Riscv64InterpreterTest, TestVsse8_vlmul8) {
  TestVsssegXeXX<UInt8, 1, 8>(
      0x0820'8427,  // vsse8.v v8, (x1), x2, v0.t
      2,
      {0x5583'5502'5581'5500, 0x5587'5506'5585'5504, 0x558b'550a'5589'5508, 0x558f'550e'558d'550c,
       0x5593'5512'5591'5510, 0x5597'5516'5595'5514, 0x559b'551a'5599'5518, 0x559f'551e'559d'551c,
       0x55a3'5522'55a1'5520, 0x55a7'5526'55a5'5524, 0x55ab'552a'55a9'5528, 0x55af'552e'55ad'552c,
       0x55b3'5532'55b1'5530, 0x55b7'5536'55b5'5534, 0x55bb'553a'55b9'5538, 0x55bf'553e'55bd'553c,
       0x55c3'5542'55c1'5540, 0x55c7'5546'55c5'5544, 0x55cb'554a'55c9'5548, 0x55cf'554e'55cd'554c,
       0x55d3'5552'55d1'5550, 0x55d7'5556'55d5'5554, 0x55db'555a'55d9'5558, 0x55df'555e'55dd'555c,
       0x55e3'5562'55e1'5560, 0x55e7'5566'55e5'5564, 0x55eb'556a'55e9'5568, 0x55ef'556e'55ed'556c,
       0x55f3'5572'55f1'5570, 0x55f7'5576'55f5'5574, 0x55fb'557a'55f9'5578, 0x55ff'557e'55fd'557c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e8_vlmul1) {
  TestVsssegXeXX<UInt8, 2, 1>(0x2820'8427,  // vssseg2e8.v v8, (x1), x2, v0.t
                              4,
                              {0x5555'9181'5555'1000,
                               0x5555'9383'5555'1202,
                               0x5555'9585'5555'1404,
                               0x5555'9787'5555'1606,
                               0x5555'9989'5555'1808,
                               0x5555'9b8b'5555'1a0a,
                               0x5555'9d8d'5555'1c0c,
                               0x5555'9f8f'5555'1e0e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e8_vlmul2) {
  TestVsssegXeXX<UInt8, 2, 2>(0x2820'8427,  // vssseg2e8.v v8, (x1), x2, v0.t
                              4,
                              {0x5555'a181'5555'2000,
                               0x5555'a383'5555'2202,
                               0x5555'a585'5555'2404,
                               0x5555'a787'5555'2606,
                               0x5555'a989'5555'2808,
                               0x5555'ab8b'5555'2a0a,
                               0x5555'ad8d'5555'2c0c,
                               0x5555'af8f'5555'2e0e,
                               0x5555'b191'5555'3010,
                               0x5555'b393'5555'3212,
                               0x5555'b595'5555'3414,
                               0x5555'b797'5555'3616,
                               0x5555'b999'5555'3818,
                               0x5555'bb9b'5555'3a1a,
                               0x5555'bd9d'5555'3c1c,
                               0x5555'bf9f'5555'3e1e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e8_vlmul4) {
  TestVsssegXeXX<UInt8, 2, 4>(
      0x2820'8427,  // vssseg4e8.v v8, (x1), x2, v0.t
      4,
      {0x5555'c181'5555'4000, 0x5555'c383'5555'4202, 0x5555'c585'5555'4404, 0x5555'c787'5555'4606,
       0x5555'c989'5555'4808, 0x5555'cb8b'5555'4a0a, 0x5555'cd8d'5555'4c0c, 0x5555'cf8f'5555'4e0e,
       0x5555'd191'5555'5010, 0x5555'd393'5555'5212, 0x5555'd595'5555'5414, 0x5555'd797'5555'5616,
       0x5555'd999'5555'5818, 0x5555'db9b'5555'5a1a, 0x5555'dd9d'5555'5c1c, 0x5555'df9f'5555'5e1e,
       0x5555'e1a1'5555'6020, 0x5555'e3a3'5555'6222, 0x5555'e5a5'5555'6424, 0x5555'e7a7'5555'6626,
       0x5555'e9a9'5555'6828, 0x5555'ebab'5555'6a2a, 0x5555'edad'5555'6c2c, 0x5555'efaf'5555'6e2e,
       0x5555'f1b1'5555'7030, 0x5555'f3b3'5555'7232, 0x5555'f5b5'5555'7434, 0x5555'f7b7'5555'7636,
       0x5555'f9b9'5555'7838, 0x5555'fbbb'5555'7a3a, 0x5555'fdbd'5555'7c3c, 0x5555'ffbf'5555'7e3e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg3e8_vlmul1) {
  TestVsssegXeXX<UInt8, 3, 1>(0x4820'8427,  // vssseg3e8.v v8, (x1), x2, v0.t
                              4,
                              {0x55a1'9181'5520'1000,
                               0x55a3'9383'5522'1202,
                               0x55a5'9585'5524'1404,
                               0x55a7'9787'5526'1606,
                               0x55a9'9989'5528'1808,
                               0x55ab'9b8b'552a'1a0a,
                               0x55ad'9d8d'552c'1c0c,
                               0x55af'9f8f'552e'1e0e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg3e8_vlmul2) {
  TestVsssegXeXX<UInt8, 3, 2>(0x4820'8427,  // vssseg3e8.v v8, (x1), x2, v0.t
                              4,
                              {0x55c1'a181'5540'2000,
                               0x55c3'a383'5542'2202,
                               0x55c5'a585'5544'2404,
                               0x55c7'a787'5546'2606,
                               0x55c9'a989'5548'2808,
                               0x55cb'ab8b'554a'2a0a,
                               0x55cd'ad8d'554c'2c0c,
                               0x55cf'af8f'554e'2e0e,
                               0x55d1'b191'5550'3010,
                               0x55d3'b393'5552'3212,
                               0x55d5'b595'5554'3414,
                               0x55d7'b797'5556'3616,
                               0x55d9'b999'5558'3818,
                               0x55db'bb9b'555a'3a1a,
                               0x55dd'bd9d'555c'3c1c,
                               0x55df'bf9f'555e'3e1e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg4e8_vlmul1) {
  TestVsssegXeXX<UInt8, 4, 1>(0x6820'8427,  // vssseg4e8.v v8, (x1), x2, v0.t
                              4,
                              {0xb1a1'9181'3020'1000,
                               0xb3a3'9383'3222'1202,
                               0xb5a5'9585'3424'1404,
                               0xb7a7'9787'3626'1606,
                               0xb9a9'9989'3828'1808,
                               0xbbab'9b8b'3a2a'1a0a,
                               0xbdad'9d8d'3c2c'1c0c,
                               0xbfaf'9f8f'3e2e'1e0e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg4e8_vlmul2) {
  TestVsssegXeXX<UInt8, 4, 2>(0x6820'8427,  // vssseg4e8.v v8, (x1), x2, v0.t
                              4,
                              {0xe1c1'a181'6040'2000,
                               0xe3c3'a383'6242'2202,
                               0xe5c5'a585'6444'2404,
                               0xe7c7'a787'6646'2606,
                               0xe9c9'a989'6848'2808,
                               0xebcb'ab8b'6a4a'2a0a,
                               0xedcd'ad8d'6c4c'2c0c,
                               0xefcf'af8f'6e4e'2e0e,
                               0xf1d1'b191'7050'3010,
                               0xf3d3'b393'7252'3212,
                               0xf5d5'b595'7454'3414,
                               0xf7d7'b797'7656'3616,
                               0xf9d9'b999'7858'3818,
                               0xfbdb'bb9b'7a5a'3a1a,
                               0xfddd'bd9d'7c5c'3c1c,
                               0xffdf'bf9f'7e5e'3e1e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg5e8) {
  TestVsssegXeXX<UInt8, 5, 1>(0x8820'8427,  // vssseg5e8.v v8, (x1), x2, v0.t
                              8,
                              {0x5555'5540'3020'1000,
                               0x5555'55c1'b1a1'9181,
                               0x5555'5542'3222'1202,
                               0x5555'55c3'b3a3'9383,
                               0x5555'5544'3424'1404,
                               0x5555'55c5'b5a5'9585,
                               0x5555'5546'3626'1606,
                               0x5555'55c7'b7a7'9787,
                               0x5555'5548'3828'1808,
                               0x5555'55c9'b9a9'9989,
                               0x5555'554a'3a2a'1a0a,
                               0x5555'55cb'bbab'9b8b,
                               0x5555'554c'3c2c'1c0c,
                               0x5555'55cd'bdad'9d8d,
                               0x5555'554e'3e2e'1e0e,
                               0x5555'55cf'bfaf'9f8f});
}

TEST_F(Riscv64InterpreterTest, TestVssseg6e8) {
  TestVsssegXeXX<UInt8, 6, 1>(0xa820'8427,  // vssseg6e8.v v8, (x1), x2, v0.t
                              8,
                              {0x5555'5040'3020'1000,
                               0x5555'd1c1'b1a1'9181,
                               0x5555'5242'3222'1202,
                               0x5555'd3c3'b3a3'9383,
                               0x5555'5444'3424'1404,
                               0x5555'd5c5'b5a5'9585,
                               0x5555'5646'3626'1606,
                               0x5555'd7c7'b7a7'9787,
                               0x5555'5848'3828'1808,
                               0x5555'd9c9'b9a9'9989,
                               0x5555'5a4a'3a2a'1a0a,
                               0x5555'dbcb'bbab'9b8b,
                               0x5555'5c4c'3c2c'1c0c,
                               0x5555'ddcd'bdad'9d8d,
                               0x5555'5e4e'3e2e'1e0e,
                               0x5555'dfcf'bfaf'9f8f});
}

TEST_F(Riscv64InterpreterTest, TestVssseg7e8) {
  TestVsssegXeXX<UInt8, 7, 1>(0xc820'8427,  // vssseg7e8.v v8, (x1), x2, v0.t
                              8,
                              {0x5560'5040'3020'1000,
                               0x55e1'd1c1'b1a1'9181,
                               0x5562'5242'3222'1202,
                               0x55e3'd3c3'b3a3'9383,
                               0x5564'5444'3424'1404,
                               0x55e5'd5c5'b5a5'9585,
                               0x5566'5646'3626'1606,
                               0x55e7'd7c7'b7a7'9787,
                               0x5568'5848'3828'1808,
                               0x55e9'd9c9'b9a9'9989,
                               0x556a'5a4a'3a2a'1a0a,
                               0x55eb'dbcb'bbab'9b8b,
                               0x556c'5c4c'3c2c'1c0c,
                               0x55ed'ddcd'bdad'9d8d,
                               0x556e'5e4e'3e2e'1e0e,
                               0x55ef'dfcf'bfaf'9f8f});
}

TEST_F(Riscv64InterpreterTest, TestVssseg8e8) {
  TestVsssegXeXX<UInt8, 8, 1>(0xe820'8427,  // vssseg8e8.v v8, (x1), x2, v0.t
                              8,
                              {0x7060'5040'3020'1000,
                               0xf1e1'd1c1'b1a1'9181,
                               0x7262'5242'3222'1202,
                               0xf3e3'd3c3'b3a3'9383,
                               0x7464'5444'3424'1404,
                               0xf5e5'd5c5'b5a5'9585,
                               0x7666'5646'3626'1606,
                               0xf7e7'd7c7'b7a7'9787,
                               0x7868'5848'3828'1808,
                               0xf9e9'd9c9'b9a9'9989,
                               0x7a6a'5a4a'3a2a'1a0a,
                               0xfbeb'dbcb'bbab'9b8b,
                               0x7c6c'5c4c'3c2c'1c0c,
                               0xfded'ddcd'bdad'9d8d,
                               0x7e6e'5e4e'3e2e'1e0e,
                               0xffef'dfcf'bfaf'9f8f});
}

TEST_F(Riscv64InterpreterTest, TestVsse16_vlmul1) {
  TestVsssegXeXX<UInt16, 1, 1>(0x0820'd427,  // vsse16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'5555'5555'8100,
                                0x5555'5555'5555'8302,
                                0x5555'5555'5555'8504,
                                0x5555'5555'5555'8706,
                                0x5555'5555'5555'8908,
                                0x5555'5555'5555'8b0a,
                                0x5555'5555'5555'8d0c,
                                0x5555'5555'5555'8f0e});
}

TEST_F(Riscv64InterpreterTest, TestVsse16_vlmul2) {
  TestVsssegXeXX<UInt16, 1, 2>(0x0820'd427,  // vsse16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'5555'5555'8100,
                                0x5555'5555'5555'8302,
                                0x5555'5555'5555'8504,
                                0x5555'5555'5555'8706,
                                0x5555'5555'5555'8908,
                                0x5555'5555'5555'8b0a,
                                0x5555'5555'5555'8d0c,
                                0x5555'5555'5555'8f0e,
                                0x5555'5555'5555'9110,
                                0x5555'5555'5555'9312,
                                0x5555'5555'5555'9514,
                                0x5555'5555'5555'9716,
                                0x5555'5555'5555'9918,
                                0x5555'5555'5555'9b1a,
                                0x5555'5555'5555'9d1c,
                                0x5555'5555'5555'9f1e});
}

TEST_F(Riscv64InterpreterTest, TestVsse16_vlmul4) {
  TestVsssegXeXX<UInt16, 1, 4>(
      0x0820'd427,  // vsse16.v v8, (x1), x2, v0.t
      8,
      {0x5555'5555'5555'8100, 0x5555'5555'5555'8302, 0x5555'5555'5555'8504, 0x5555'5555'5555'8706,
       0x5555'5555'5555'8908, 0x5555'5555'5555'8b0a, 0x5555'5555'5555'8d0c, 0x5555'5555'5555'8f0e,
       0x5555'5555'5555'9110, 0x5555'5555'5555'9312, 0x5555'5555'5555'9514, 0x5555'5555'5555'9716,
       0x5555'5555'5555'9918, 0x5555'5555'5555'9b1a, 0x5555'5555'5555'9d1c, 0x5555'5555'5555'9f1e,
       0x5555'5555'5555'a120, 0x5555'5555'5555'a322, 0x5555'5555'5555'a524, 0x5555'5555'5555'a726,
       0x5555'5555'5555'a928, 0x5555'5555'5555'ab2a, 0x5555'5555'5555'ad2c, 0x5555'5555'5555'af2e,
       0x5555'5555'5555'b130, 0x5555'5555'5555'b332, 0x5555'5555'5555'b534, 0x5555'5555'5555'b736,
       0x5555'5555'5555'b938, 0x5555'5555'5555'bb3a, 0x5555'5555'5555'bd3c, 0x5555'5555'5555'bf3e});
}

TEST_F(Riscv64InterpreterTest, TestVsse16_vlmul8) {
  TestVsssegXeXX<UInt16, 1, 8>(
      0x0820'd427,  // vsse16.v v8, (x1), x2, v0.t
      4,
      {0x5555'8302'5555'8100, 0x5555'8706'5555'8504, 0x5555'8b0a'5555'8908, 0x5555'8f0e'5555'8d0c,
       0x5555'9312'5555'9110, 0x5555'9716'5555'9514, 0x5555'9b1a'5555'9918, 0x5555'9f1e'5555'9d1c,
       0x5555'a322'5555'a120, 0x5555'a726'5555'a524, 0x5555'ab2a'5555'a928, 0x5555'af2e'5555'ad2c,
       0x5555'b332'5555'b130, 0x5555'b736'5555'b534, 0x5555'bb3a'5555'b938, 0x5555'bf3e'5555'bd3c,
       0x5555'c342'5555'c140, 0x5555'c746'5555'c544, 0x5555'cb4a'5555'c948, 0x5555'cf4e'5555'cd4c,
       0x5555'd352'5555'd150, 0x5555'd756'5555'd554, 0x5555'db5a'5555'd958, 0x5555'df5e'5555'dd5c,
       0x5555'e362'5555'e160, 0x5555'e766'5555'e564, 0x5555'eb6a'5555'e968, 0x5555'ef6e'5555'ed6c,
       0x5555'f372'5555'f170, 0x5555'f776'5555'f574, 0x5555'fb7a'5555'f978, 0x5555'ff7e'5555'fd7c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e16_vlmul1) {
  TestVsssegXeXX<UInt16, 2, 1>(0x2820'd427,  // vssseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'5555'9110'8100,
                                0x5555'5555'9312'8302,
                                0x5555'5555'9514'8504,
                                0x5555'5555'9716'8706,
                                0x5555'5555'9918'8908,
                                0x5555'5555'9b1a'8b0a,
                                0x5555'5555'9d1c'8d0c,
                                0x5555'5555'9f1e'8f0e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e16_vlmul2) {
  TestVsssegXeXX<UInt16, 2, 2>(0x2820'd427,  // vssseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'5555'a120'8100,
                                0x5555'5555'a322'8302,
                                0x5555'5555'a524'8504,
                                0x5555'5555'a726'8706,
                                0x5555'5555'a928'8908,
                                0x5555'5555'ab2a'8b0a,
                                0x5555'5555'ad2c'8d0c,
                                0x5555'5555'af2e'8f0e,
                                0x5555'5555'b130'9110,
                                0x5555'5555'b332'9312,
                                0x5555'5555'b534'9514,
                                0x5555'5555'b736'9716,
                                0x5555'5555'b938'9918,
                                0x5555'5555'bb3a'9b1a,
                                0x5555'5555'bd3c'9d1c,
                                0x5555'5555'bf3e'9f1e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e16_vlmul4) {
  TestVsssegXeXX<UInt16, 2, 4>(
      0x2820'd427,  // vssseg2e16.v v8, (x1), x2, v0.t
      8,
      {0x5555'5555'c140'8100, 0x5555'5555'c342'8302, 0x5555'5555'c544'8504, 0x5555'5555'c746'8706,
       0x5555'5555'c948'8908, 0x5555'5555'cb4a'8b0a, 0x5555'5555'cd4c'8d0c, 0x5555'5555'cf4e'8f0e,
       0x5555'5555'd150'9110, 0x5555'5555'd352'9312, 0x5555'5555'd554'9514, 0x5555'5555'd756'9716,
       0x5555'5555'd958'9918, 0x5555'5555'db5a'9b1a, 0x5555'5555'dd5c'9d1c, 0x5555'5555'df5e'9f1e,
       0x5555'5555'e160'a120, 0x5555'5555'e362'a322, 0x5555'5555'e564'a524, 0x5555'5555'e766'a726,
       0x5555'5555'e968'a928, 0x5555'5555'eb6a'ab2a, 0x5555'5555'ed6c'ad2c, 0x5555'5555'ef6e'af2e,
       0x5555'5555'f170'b130, 0x5555'5555'f372'b332, 0x5555'5555'f574'b534, 0x5555'5555'f776'b736,
       0x5555'5555'f978'b938, 0x5555'5555'fb7a'bb3a, 0x5555'5555'fd7c'bd3c, 0x5555'5555'ff7e'bf3e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg3e16_vlmul1) {
  TestVsssegXeXX<UInt16, 3, 1>(0x4820'd427,  // vssseg3e16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'a120'9110'8100,
                                0x5555'a322'9312'8302,
                                0x5555'a524'9514'8504,
                                0x5555'a726'9716'8706,
                                0x5555'a928'9918'8908,
                                0x5555'ab2a'9b1a'8b0a,
                                0x5555'ad2c'9d1c'8d0c,
                                0x5555'af2e'9f1e'8f0e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg3e16_vlmul2) {
  TestVsssegXeXX<UInt16, 3, 2>(0x4820'd427,  // vssseg3e16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'c140'a120'8100,
                                0x5555'c342'a322'8302,
                                0x5555'c544'a524'8504,
                                0x5555'c746'a726'8706,
                                0x5555'c948'a928'8908,
                                0x5555'cb4a'ab2a'8b0a,
                                0x5555'cd4c'ad2c'8d0c,
                                0x5555'cf4e'af2e'8f0e,
                                0x5555'd150'b130'9110,
                                0x5555'd352'b332'9312,
                                0x5555'd554'b534'9514,
                                0x5555'd756'b736'9716,
                                0x5555'd958'b938'9918,
                                0x5555'db5a'bb3a'9b1a,
                                0x5555'dd5c'bd3c'9d1c,
                                0x5555'df5e'bf3e'9f1e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg4e16_vlmul1) {
  TestVsssegXeXX<UInt16, 4, 1>(0x6820'd427,  // vssseg4e16.v v8, (x1), x2, v0.t
                               8,
                               {0xb130'a120'9110'8100,
                                0xb332'a322'9312'8302,
                                0xb534'a524'9514'8504,
                                0xb736'a726'9716'8706,
                                0xb938'a928'9918'8908,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0xbf3e'af2e'9f1e'8f0e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg4e16_vlmul2) {
  TestVsssegXeXX<UInt16, 4, 2>(0x6820'd427,  // vssseg4e16.v v8, (x1), x2, v0.t
                               8,
                               {0xe160'c140'a120'8100,
                                0xe362'c342'a322'8302,
                                0xe564'c544'a524'8504,
                                0xe766'c746'a726'8706,
                                0xe968'c948'a928'8908,
                                0xeb6a'cb4a'ab2a'8b0a,
                                0xed6c'cd4c'ad2c'8d0c,
                                0xef6e'cf4e'af2e'8f0e,
                                0xf170'd150'b130'9110,
                                0xf372'd352'b332'9312,
                                0xf574'd554'b534'9514,
                                0xf776'd756'b736'9716,
                                0xf978'd958'b938'9918,
                                0xfb7a'db5a'bb3a'9b1a,
                                0xfd7c'dd5c'bd3c'9d1c,
                                0xff7e'df5e'bf3e'9f1e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg5e16) {
  TestVsssegXeXX<UInt16, 5, 1>(0x8820'd427,  // vssseg5e16.v v8, (x1), x2, v0.t
                               16,
                               {0xb130'a120'9110'8100,
                                0x5555'5555'5555'c140,
                                0xb332'a322'9312'8302,
                                0x5555'5555'5555'c342,
                                0xb534'a524'9514'8504,
                                0x5555'5555'5555'c544,
                                0xb736'a726'9716'8706,
                                0x5555'5555'5555'c746,
                                0xb938'a928'9918'8908,
                                0x5555'5555'5555'c948,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0x5555'5555'5555'cb4a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0x5555'5555'5555'cd4c,
                                0xbf3e'af2e'9f1e'8f0e,
                                0x5555'5555'5555'cf4e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg6e16) {
  TestVsssegXeXX<UInt16, 6, 1>(0xa820'd427,  // vssseg6e16.v v8, (x1), x2, v0.t
                               16,
                               {0xb130'a120'9110'8100,
                                0x5555'5555'd150'c140,
                                0xb332'a322'9312'8302,
                                0x5555'5555'd352'c342,
                                0xb534'a524'9514'8504,
                                0x5555'5555'd554'c544,
                                0xb736'a726'9716'8706,
                                0x5555'5555'd756'c746,
                                0xb938'a928'9918'8908,
                                0x5555'5555'd958'c948,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0x5555'5555'db5a'cb4a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0x5555'5555'dd5c'cd4c,
                                0xbf3e'af2e'9f1e'8f0e,
                                0x5555'5555'df5e'cf4e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg7e16) {
  TestVsssegXeXX<UInt16, 7, 1>(0xc820'd427,  // vssseg7e16.v v8, (x1), x2, v0.t
                               16,
                               {0xb130'a120'9110'8100,
                                0x5555'e160'd150'c140,
                                0xb332'a322'9312'8302,
                                0x5555'e362'd352'c342,
                                0xb534'a524'9514'8504,
                                0x5555'e564'd554'c544,
                                0xb736'a726'9716'8706,
                                0x5555'e766'd756'c746,
                                0xb938'a928'9918'8908,
                                0x5555'e968'd958'c948,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0x5555'eb6a'db5a'cb4a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0x5555'ed6c'dd5c'cd4c,
                                0xbf3e'af2e'9f1e'8f0e,
                                0x5555'ef6e'df5e'cf4e});
}

TEST_F(Riscv64InterpreterTest, TestVssseg8e16) {
  TestVsssegXeXX<UInt16, 8, 1>(0xe820'd427,  // vssseg8e16.v v8, (x1), x2, v0.t
                               16,
                               {0xb130'a120'9110'8100,
                                0xf170'e160'd150'c140,
                                0xb332'a322'9312'8302,
                                0xf372'e362'd352'c342,
                                0xb534'a524'9514'8504,
                                0xf574'e564'd554'c544,
                                0xb736'a726'9716'8706,
                                0xf776'e766'd756'c746,
                                0xb938'a928'9918'8908,
                                0xf978'e968'd958'c948,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0xfb7a'eb6a'db5a'cb4a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0xfd7c'ed6c'dd5c'cd4c,
                                0xbf3e'af2e'9f1e'8f0e,
                                0xff7e'ef6e'df5e'cf4e});
}

TEST_F(Riscv64InterpreterTest, TestVsse32_vlmul1) {
  TestVsssegXeXX<UInt32, 1, 1>(0x0820'e427,  // vsse32.v v8, (x1), x2, v0.t
                               16,
                               {0x5555'5555'8302'8100,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8706'8504,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8f0e'8d0c});
}

TEST_F(Riscv64InterpreterTest, TestVsse32_vlmul2) {
  TestVsssegXeXX<UInt32, 1, 2>(0x0820'e427,  // vsse32.v v8, (x1), x2, v0.t
                               16,
                               {0x5555'5555'8302'8100,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8706'8504,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8f0e'8d0c,
                                0x5555'5555'5555'5555,
                                0x5555'5555'9312'9110,
                                0x5555'5555'5555'5555,
                                0x5555'5555'9716'9514,
                                0x5555'5555'5555'5555,
                                0x5555'5555'9b1a'9918,
                                0x5555'5555'5555'5555,
                                0x5555'5555'9f1e'9d1c});
}

TEST_F(Riscv64InterpreterTest, TestVsse32_vlmul4) {
  TestVsssegXeXX<UInt32, 1, 4>(
      0x0820'e427,  // vsse32.v v8, (x1), x2, v0.t
      16,
      {0x5555'5555'8302'8100, 0x5555'5555'5555'5555, 0x5555'5555'8706'8504, 0x5555'5555'5555'5555,
       0x5555'5555'8b0a'8908, 0x5555'5555'5555'5555, 0x5555'5555'8f0e'8d0c, 0x5555'5555'5555'5555,
       0x5555'5555'9312'9110, 0x5555'5555'5555'5555, 0x5555'5555'9716'9514, 0x5555'5555'5555'5555,
       0x5555'5555'9b1a'9918, 0x5555'5555'5555'5555, 0x5555'5555'9f1e'9d1c, 0x5555'5555'5555'5555,
       0x5555'5555'a322'a120, 0x5555'5555'5555'5555, 0x5555'5555'a726'a524, 0x5555'5555'5555'5555,
       0x5555'5555'ab2a'a928, 0x5555'5555'5555'5555, 0x5555'5555'af2e'ad2c, 0x5555'5555'5555'5555,
       0x5555'5555'b332'b130, 0x5555'5555'5555'5555, 0x5555'5555'b736'b534, 0x5555'5555'5555'5555,
       0x5555'5555'bb3a'b938, 0x5555'5555'5555'5555, 0x5555'5555'bf3e'bd3c});
}

TEST_F(Riscv64InterpreterTest, TestVsse32_vlmul8) {
  TestVsssegXeXX<UInt32, 1, 8>(
      0x0820'e427,  // vsse32.v v8, (x1), x2, v0.t
      8,
      {0x5555'5555'8302'8100, 0x5555'5555'8706'8504, 0x5555'5555'8b0a'8908, 0x5555'5555'8f0e'8d0c,
       0x5555'5555'9312'9110, 0x5555'5555'9716'9514, 0x5555'5555'9b1a'9918, 0x5555'5555'9f1e'9d1c,
       0x5555'5555'a322'a120, 0x5555'5555'a726'a524, 0x5555'5555'ab2a'a928, 0x5555'5555'af2e'ad2c,
       0x5555'5555'b332'b130, 0x5555'5555'b736'b534, 0x5555'5555'bb3a'b938, 0x5555'5555'bf3e'bd3c,
       0x5555'5555'c342'c140, 0x5555'5555'c746'c544, 0x5555'5555'cb4a'c948, 0x5555'5555'cf4e'cd4c,
       0x5555'5555'd352'd150, 0x5555'5555'd756'd554, 0x5555'5555'db5a'd958, 0x5555'5555'df5e'dd5c,
       0x5555'5555'e362'e160, 0x5555'5555'e766'e564, 0x5555'5555'eb6a'e968, 0x5555'5555'ef6e'ed6c,
       0x5555'5555'f372'f170, 0x5555'5555'f776'f574, 0x5555'5555'fb7a'f978, 0x5555'5555'ff7e'fd7c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e32_vlmul1) {
  TestVsssegXeXX<UInt32, 2, 1>(0x2820'e427,  // vssseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {0x9312'9110'8302'8100,
                                0x5555'5555'5555'5555,
                                0x9716'9514'8706'8504,
                                0x5555'5555'5555'5555,
                                0x9b1a'9918'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'8f0e'8d0c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e32_vlmul2) {
  TestVsssegXeXX<UInt32, 2, 2>(0x2820'e427,  // vssseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {0xa322'a120'8302'8100,
                                0x5555'5555'5555'5555,
                                0xa726'a524'8706'8504,
                                0x5555'5555'5555'5555,
                                0xab2a'a928'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0xaf2e'ad2c'8f0e'8d0c,
                                0x5555'5555'5555'5555,
                                0xb332'b130'9312'9110,
                                0x5555'5555'5555'5555,
                                0xb736'b534'9716'9514,
                                0x5555'5555'5555'5555,
                                0xbb3a'b938'9b1a'9918,
                                0x5555'5555'5555'5555,
                                0xbf3e'bd3c'9f1e'9d1c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e32_vlmul4) {
  TestVsssegXeXX<UInt32, 2, 4>(
      0x2820'e427,  // vssseg2e32.v v8, (x1), x2, v0.t
      16,
      {0xc342'c140'8302'8100, 0x5555'5555'5555'5555, 0xc746'c544'8706'8504, 0x5555'5555'5555'5555,
       0xcb4a'c948'8b0a'8908, 0x5555'5555'5555'5555, 0xcf4e'cd4c'8f0e'8d0c, 0x5555'5555'5555'5555,
       0xd352'd150'9312'9110, 0x5555'5555'5555'5555, 0xd756'd554'9716'9514, 0x5555'5555'5555'5555,
       0xdb5a'd958'9b1a'9918, 0x5555'5555'5555'5555, 0xdf5e'dd5c'9f1e'9d1c, 0x5555'5555'5555'5555,
       0xe362'e160'a322'a120, 0x5555'5555'5555'5555, 0xe766'e564'a726'a524, 0x5555'5555'5555'5555,
       0xeb6a'e968'ab2a'a928, 0x5555'5555'5555'5555, 0xef6e'ed6c'af2e'ad2c, 0x5555'5555'5555'5555,
       0xf372'f170'b332'b130, 0x5555'5555'5555'5555, 0xf776'f574'b736'b534, 0x5555'5555'5555'5555,
       0xfb7a'f978'bb3a'b938, 0x5555'5555'5555'5555, 0xff7e'fd7c'bf3e'bd3c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg3e32_vlmul1) {
  TestVsssegXeXX<UInt32, 3, 1>(0x4820'e427,  // vssseg3e32.v v8, (x1), x2, v0.t
                               16,
                               {0x9312'9110'8302'8100,
                                0x5555'5555'a322'a120,
                                0x9716'9514'8706'8504,
                                0x5555'5555'a726'a524,
                                0x9b1a'9918'8b0a'8908,
                                0x5555'5555'ab2a'a928,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0x5555'5555'af2e'ad2c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg3e32_vlmul2) {
  TestVsssegXeXX<UInt32, 3, 2>(0x4820'e427,  // vssseg3e32.v v8, (x1), x2, v0.t
                               16,
                               {0xa322'a120'8302'8100,
                                0x5555'5555'c342'c140,
                                0xa726'a524'8706'8504,
                                0x5555'5555'c746'c544,
                                0xab2a'a928'8b0a'8908,
                                0x5555'5555'cb4a'c948,
                                0xaf2e'ad2c'8f0e'8d0c,
                                0x5555'5555'cf4e'cd4c,
                                0xb332'b130'9312'9110,
                                0x5555'5555'd352'd150,
                                0xb736'b534'9716'9514,
                                0x5555'5555'd756'd554,
                                0xbb3a'b938'9b1a'9918,
                                0x5555'5555'db5a'd958,
                                0xbf3e'bd3c'9f1e'9d1c,
                                0x5555'5555'df5e'dd5c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg4e32_vlmul1) {
  TestVsssegXeXX<UInt32, 4, 1>(0x6820'e427,  // vssseg4e32.v v8, (x1), x2, v0.t
                               16,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg4e32_vlmul2) {
  TestVsssegXeXX<UInt32, 4, 2>(0x6820'e427,  // vssseg4e32.v v8, (x1), x2, v0.t
                               16,
                               {0xa322'a120'8302'8100,
                                0xe362'e160'c342'c140,
                                0xa726'a524'8706'8504,
                                0xe766'e564'c746'c544,
                                0xab2a'a928'8b0a'8908,
                                0xeb6a'e968'cb4a'c948,
                                0xaf2e'ad2c'8f0e'8d0c,
                                0xef6e'ed6c'cf4e'cd4c,
                                0xb332'b130'9312'9110,
                                0xf372'f170'd352'd150,
                                0xb736'b534'9716'9514,
                                0xf776'f574'd756'd554,
                                0xbb3a'b938'9b1a'9918,
                                0xfb7a'f978'db5a'd958,
                                0xbf3e'bd3c'9f1e'9d1c,
                                0xff7e'fd7c'df5e'dd5c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg5e32) {
  TestVsssegXeXX<UInt32, 5, 1>(0x8820'e427,  // vssseg5e32.v v8, (x1), x2, v0.t
                               32,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0x5555'5555'c342'c140,
                                0x5555'5555'5555'5555,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0x5555'5555'c746'c544,
                                0x5555'5555'5555'5555,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0x5555'5555'cb4a'c948,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c,
                                0x5555'5555'cf4e'cd4c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg6e32) {
  TestVsssegXeXX<UInt32, 6, 1>(0xa820'e427,  // vssseg6e32.v v8, (x1), x2, v0.t
                               32,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0xd352'd150'c342'c140,
                                0x5555'5555'5555'5555,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0xd756'd554'c746'c544,
                                0x5555'5555'5555'5555,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0xdb5a'd958'cb4a'c948,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c,
                                0xdf5e'dd5c'cf4e'cd4c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg7e32) {
  TestVsssegXeXX<UInt32, 7, 1>(0xc820'e427,  // vssseg7e32.v v8, (x1), x2, v0.t
                               32,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0xd352'd150'c342'c140,
                                0x5555'5555'e362'e160,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0xd756'd554'c746'c544,
                                0x5555'5555'e766'e564,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0xdb5a'd958'cb4a'c948,
                                0x5555'5555'eb6a'e968,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c,
                                0xdf5e'dd5c'cf4e'cd4c,
                                0x5555'5555'ef6e'ed6c});
}

TEST_F(Riscv64InterpreterTest, TestVssseg8e32) {
  TestVsssegXeXX<UInt32, 8, 1>(0xe820'e427,  // vssseg8e32.v v8, (x1), x2, v0.t
                               32,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0xd352'd150'c342'c140,
                                0xf372'f170'e362'e160,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0xd756'd554'c746'c544,
                                0xf776'f574'e766'e564,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0xdb5a'd958'cb4a'c948,
                                0xfb7a'f978'eb6a'e968,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c,
                                0xdf5e'dd5c'cf4e'cd4c,
                                0xff7e'fd7c'ef6e'ed6c});
}

TEST_F(Riscv64InterpreterTest, TestVsse64_vlmul1) {
  TestVsssegXeXX<UInt64, 1, 1>(0x0820'f427,  // vsse64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908});
}

TEST_F(Riscv64InterpreterTest, TestVsse64_vlmul2) {
  TestVsssegXeXX<UInt64, 1, 2>(0x0820'f427,  // vsse64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x9716'9514'9312'9110,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'9b1a'9918});
}

TEST_F(Riscv64InterpreterTest, TestVsse64_vlmul4) {
  TestVsssegXeXX<UInt64, 1, 4>(
      0x0820'f427,  // vsse64.v v8, (x1), x2, v0.t
      32,
      {0x8706'8504'8302'8100, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0x8f0e'8d0c'8b0a'8908, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0x9716'9514'9312'9110, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0x9f1e'9d1c'9b1a'9918, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0xa726'a524'a322'a120, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0xaf2e'ad2c'ab2a'a928, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0xb736'b534'b332'b130, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVsse64_vlmul8) {
  TestVsssegXeXX<UInt64, 1, 8>(
      0x0820'f427,  // vsse64.v v8, (x1), x2, v0.t
      16,
      {0x8706'8504'8302'8100, 0x5555'5555'5555'5555, 0x8f0e'8d0c'8b0a'8908, 0x5555'5555'5555'5555,
       0x9716'9514'9312'9110, 0x5555'5555'5555'5555, 0x9f1e'9d1c'9b1a'9918, 0x5555'5555'5555'5555,
       0xa726'a524'a322'a120, 0x5555'5555'5555'5555, 0xaf2e'ad2c'ab2a'a928, 0x5555'5555'5555'5555,
       0xb736'b534'b332'b130, 0x5555'5555'5555'5555, 0xbf3e'bd3c'bb3a'b938, 0x5555'5555'5555'5555,
       0xc746'c544'c342'c140, 0x5555'5555'5555'5555, 0xcf4e'cd4c'cb4a'c948, 0x5555'5555'5555'5555,
       0xd756'd554'd352'd150, 0x5555'5555'5555'5555, 0xdf5e'dd5c'db5a'd958, 0x5555'5555'5555'5555,
       0xe766'e564'e362'e160, 0x5555'5555'5555'5555, 0xef6e'ed6c'eb6a'e968, 0x5555'5555'5555'5555,
       0xf776'f574'f372'f170, 0x5555'5555'5555'5555, 0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e64_vlmul1) {
  TestVsssegXeXX<UInt64, 2, 1>(0x2820'f427,  // vssseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e64_vlmul2) {
  TestVsssegXeXX<UInt64, 2, 2>(0x2820'f427,  // vssseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0xa726'a524'a322'a120,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0xaf2e'ad2c'ab2a'a928,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x9716'9514'9312'9110,
                                0xb736'b534'b332'b130,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'9b1a'9918,
                                0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVssseg2e64_vlmul4) {
  TestVsssegXeXX<UInt64, 2, 4>(0x2820'f427,  // vssseg2e64.v v8, (x1), x2, v0.t
                               16,
                               {0x8706'8504'8302'8100,
                                0xc746'c544'c342'c140,
                                0x8f0e'8d0c'8b0a'8908,
                                0xcf4e'cd4c'cb4a'c948,
                                0x9716'9514'9312'9110,
                                0xd756'd554'd352'd150,
                                0x9f1e'9d1c'9b1a'9918,
                                0xdf5e'dd5c'db5a'd958,
                                0xa726'a524'a322'a120,
                                0xe766'e564'e362'e160,
                                0xaf2e'ad2c'ab2a'a928,
                                0xef6e'ed6c'eb6a'e968,
                                0xb736'b534'b332'b130,
                                0xf776'f574'f372'f170,
                                0xbf3e'bd3c'bb3a'b938,
                                0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVssseg3e64_vlmul1) {
  TestVsssegXeXX<UInt64, 3, 1>(0x4820'f427,  // vssseg3e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928});
}

TEST_F(Riscv64InterpreterTest, TestVssseg3e64_vlmul2) {
  TestVsssegXeXX<UInt64, 3, 2>(0x4820'f427,  // vssseg3e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0xa726'a524'a322'a120,
                                0xc746'c544'c342'c140,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0xaf2e'ad2c'ab2a'a928,
                                0xcf4e'cd4c'cb4a'c948,
                                0x5555'5555'5555'5555,
                                0x9716'9514'9312'9110,
                                0xb736'b534'b332'b130,
                                0xd756'd554'd352'd150,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'9b1a'9918,
                                0xbf3e'bd3c'bb3a'b938,
                                0xdf5e'dd5c'db5a'd958});
}

TEST_F(Riscv64InterpreterTest, TestVssseg4e64_vlmul1) {
  TestVsssegXeXX<UInt64, 4, 1>(0x6820'f427,  // vssseg4e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938});
}

TEST_F(Riscv64InterpreterTest, TestVssseg4e64_vlmul2) {
  TestVsssegXeXX<UInt64, 4, 2>(0x6820'f427,  // vssseg4e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0xa726'a524'a322'a120,
                                0xc746'c544'c342'c140,
                                0xe766'e564'e362'e160,
                                0x8f0e'8d0c'8b0a'8908,
                                0xaf2e'ad2c'ab2a'a928,
                                0xcf4e'cd4c'cb4a'c948,
                                0xef6e'ed6c'eb6a'e968,
                                0x9716'9514'9312'9110,
                                0xb736'b534'b332'b130,
                                0xd756'd554'd352'd150,
                                0xf776'f574'f372'f170,
                                0x9f1e'9d1c'9b1a'9918,
                                0xbf3e'bd3c'bb3a'b938,
                                0xdf5e'dd5c'db5a'd958,
                                0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVssseg5e64) {
  TestVsssegXeXX<UInt64, 5, 1>(0x8820'f427,  // vssseg5e64.v v8, (x1), x2, v0.t
                               64,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0xc746'c544'c342'c140,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938,
                                0xcf4e'cd4c'cb4a'c948});
}

TEST_F(Riscv64InterpreterTest, TestVssseg6e64) {
  TestVsssegXeXX<UInt64, 6, 1>(0xa820'f427,  // vssseg6e64.v v8, (x1), x2, v0.t
                               64,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0xc746'c544'c342'c140,
                                0xd756'd554'd352'd150,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938,
                                0xcf4e'cd4c'cb4a'c948,
                                0xdf5e'dd5c'db5a'd958});
}

TEST_F(Riscv64InterpreterTest, TestVssseg7e64) {
  TestVsssegXeXX<UInt64, 7, 1>(0xc820'f427,  // vssseg7e64.v v8, (x1), x2, v0.t
                               64,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0xc746'c544'c342'c140,
                                0xd756'd554'd352'd150,
                                0xe766'e564'e362'e160,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938,
                                0xcf4e'cd4c'cb4a'c948,
                                0xdf5e'dd5c'db5a'd958,
                                0xef6e'ed6c'eb6a'e968});
}

TEST_F(Riscv64InterpreterTest, TestVssseg8e64) {
  TestVsssegXeXX<UInt64, 8, 1>(0xe820'f427,  // vssseg8e64.v v8, (x1), x2, v0.t
                               64,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0xc746'c544'c342'c140,
                                0xd756'd554'd352'd150,
                                0xe766'e564'e362'e160,
                                0xf776'f574'f372'f170,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938,
                                0xcf4e'cd4c'cb4a'c948,
                                0xdf5e'dd5c'db5a'd958,
                                0xef6e'ed6c'eb6a'e968,
                                0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsm) {
  TestVsm(0x02b0'8427,  // vsm.v v8, (x1)
          {0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143});
}

TEST_F(Riscv64InterpreterTest, TestVectorMaskInstructions) {
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x630c'2457,  // vmandn.mm v8, v16, v24
                            {0x8102'8504'8102'8100, 0x8102'8504'890a'8908});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x670c'2457,  // vmand.mm v8, v16, v24
                            {0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x6b0c'2457,  // vmor.mm v8, v16, v24
                            {0x8f0e'8f0d'8706'8300, 0x9f1e'9f1c'9f1e'9b19});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x6f0c'2457,  // vmxor.mm v8, v16, v24
                            {0x890a'8f0d'8506'8300, 0x9112'9714'9d1e'9b19});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x730c'2457,  // vmorn.mm v8, v16, v24
                            {0xf7f7'f5f6'fbfb'fdff, 0xefef'edef'ebeb'edee});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x770c'2457,  // vmnand.mm v8, v16, v24
                            {0xf9fb'ffff'fdff'ffff, 0xf1f3'f7f7'fdff'ffff});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x7b0c'2457,  // vmnor.mm v8, v16, v24
                            {0x70f1'70f2'78f9'7cff, 0x60e1'60e3'60e1'64e6});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x7f0c'2457,  // vmxnor.mm v8, v16, v24
                            {0x76f5'70f2'7af9'7cff, 0x6eed'68eb'62e1'64e6});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5300'a457,  // vmsbf.m v8, v16
                            {0x0000'0000'0000'00ff, 0x0000'0000'0000'0000});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5301'2457,  // vmsof.m v8, v16
                            {0x0000'0000'0000'0100, 0x0000'0000'0000'0000});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5301'a457,  // vmsif.m v8, v16
                            {0x0000'0000'0000'01ff, 0x0000'0000'0000'0000});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5100'a457,  // vmsbf.m v8, v16, v0.t
                            {0xd5ad'd6b5'adff'ffff, 0x6af7'57bb'deed'7bb5});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5101'2457,  // vmsof.m v8, v16, v0.t
                            {0xd5ad'd6b5'af6b'b5ad, 0x6af7'57bb'deed'7bb5});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5101'a457,  // vmsif.m v8, v16, v0.t
                            {0xd5ad'd6b5'afff'ffff, 0x6af7'57bb'deed'7bb5});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kUndisturbed,
                            0x5100'a457,  // vmsbf.m v8, v16, v0.t
                            {0x5505'5415'05d5'5f57, 0x4055'5511'5445'5115});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kUndisturbed,
                            0x5101'2457,  // vmsof.m v8, v16, v0.t
                            {0x5505'5415'0741'1505, 0x4055'5511'5445'5115});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kUndisturbed,
                            0x5101'a457,  // vmsif.m v8, v16, v0.t
                            {0x5505'5415'07d5'5f57, 0x4055'5511'5445'5115});
}

TEST_F(Riscv64InterpreterTest, TestVmseq) {
  TestVectorMaskTargetInstruction(0x610c'0457,  // Vmseq.vv v8, v16, v24, v0.t
                                  {255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  0x0000'0000'0000'00ff,
                                  0x0000'000f,
                                  0x0003,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6100'c457,  // Vmseq.vx v8, v16, x1, v0.t
                                  {0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  0x0000'0000'0f00'0000,
                                  0x0000'3000,
                                  0x0040,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x610a'b457,  // Vmseq.vi  v8, v16, -0xb, v0.t
                                  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 0, 0, 0, 0},
                                  0x0000'f000'0000'0000,
                                  0x0000'0000,
                                  0x0000,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmfne) {
  TestVectorMaskTargetInstruction(0x710c'1457,  // Vmfne.vv v8, v16, v24, v0.t
                                  0xffff'fff8,
                                  0xfffe,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x7100'd457,  // Vmfne.vf v8, v16, f1, v0.t
                                  0xffff'ffbf,
                                  0xffdf,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsne) {
  TestVectorMaskTargetInstruction(
      0x650c'0457,  // Vmsne.vv v8, v16, v24, v0.t
      {0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      0xffff'ffff'ffff'ff00,
      0xffff'fff0,
      0xfffc,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x6500'c457,  // Vmsne.vx v8, v16, x1, v0.t
      {255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      0xffff'ffff'f0ff'ffff,
      0xffff'cfff,
      0xffbf,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x650a'b457,  // Vmsne.vi  v8, v16, -0xb, v0.t
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 170, 255, 255, 255, 255},
      0xffff'0fff'ffff'ffff,
      0xffff'ffff,
      0xffff,
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmflt) {
  TestVectorMaskTargetInstruction(0x6d0c'1457,  // Vmflt.vv v8, v16, v24, v0.t
                                  0x0000'f000,
                                  0x00c0,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6d00'd457,  // Vmflt.vf v8, v16, f1, v0.t
                                  0xff00'ff07,
                                  0xf0d1,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsltu) {
  TestVectorMaskTargetInstruction(0x690c'0457,  // Vmsltu.vv v8, v16, v24, v0.t
                                  {0, 0, 0, 3, 255, 255, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255},
                                  0xffff'0000'f0ff'1000,
                                  0xff00'cf00,
                                  0xf0b0,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x6900'c457,  // Vmsltu.vx v8, v16, x1, v0.t
      {85, 15, 10, 11, 255, 255, 0, 255, 0, 0, 0, 0, 136, 136, 255, 255},
      0xffaa'0000'f0ff'3330,
      0xff00'cf54,
      0xf0b0,
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmslt) {
  TestVectorMaskTargetInstruction(0x6d0c'0457,  // Vmslt.vv v8, v16, v24, v0.t
                                  {0, 0, 245, 247, 0, 32, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255},
                                  0xffff'0000'ff40'dc00,
                                  0xff00'f0a0,
                                  0xf0cc,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6d00'c457,  // Vmslt.vx v8, v16, x1, v0.t
                                  {0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 136, 136, 255, 255},
                                  0xffaa'0000'0040'0000,
                                  0xff00'0000,
                                  0xf000,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmfle) {
  TestVectorMaskTargetInstruction(0x650c'1457,  // Vmfle.vv v8, v16, v24, v0.t
                                  0x0000'f007,
                                  0x00c1,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6500'd457,  // Vmfle.vf v8, v16, f1, v0.t
                                  0xff00'ff47,
                                  0xf0f1,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsleu) {
  TestVectorMaskTargetInstruction(
      0x710c'0457,  // Vmsleu.vv v8, v16, v24, v0.t
      {255, 255, 0, 3, 255, 255, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255},
      0xffff'0000'f0ff'10ff,
      0xff00'cf0f,
      0xf0b3,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x7100'c457,  // Vmsleu.vx v8, v16, x1, v0.t
      {85, 15, 10, 11, 255, 255, 255, 255, 0, 0, 0, 0, 136, 136, 255, 255},
      0xffaa'0000'ffff'3330,
      0xff00'ff54,
      0xf0f0,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x710a'b457,  // Vmsleu.vi  v8, v16, -0xb, v0.t
      {255, 15, 15, 15, 255, 255, 255, 255, 85, 0, 5, 85, 255, 255, 255, 255},
      0xffff'f30f'ffff'333f,
      0xffff'ff57,
      0xffff,
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsle) {
  TestVectorMaskTargetInstruction(
      0x750c'0457,  // Vmsle.vv v8, v16, v24, v0.t
      {255, 255, 245, 247, 0, 32, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255},
      0xffff'0000'ff40'dcff,
      0xff00'f0af,
      0xf0cf,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x7500'c457,  // Vmsle.vx v8, v16, x1, v0.t
                                  {0, 0, 0, 0, 0, 32, 255, 0, 0, 0, 0, 0, 136, 136, 255, 255},
                                  0xffaa'0000'0f40'0000,
                                  0xff00'3000,
                                  0xf040,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x750a'b457,  // Vmsle.vi  v8, v16, -0xb
                                  {170, 0, 5, 4, 0, 32, 255, 0, 85, 0, 5, 85, 255, 255, 255, 255},
                                  0xffff'f30f'0f40'000f,
                                  0xffff'3003,
                                  0xff4f,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmfgt) {
  TestVectorMaskTargetInstruction(0x7500'd457,  // Vmfgt.vf v8, v16, f1, v0.t
                                  0x0000'0010,
                                  0x0000,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsgtu) {
  TestVectorMaskTargetInstruction(
      0x7900'c457,  // Vmsgtu.vx v8, v16, x1, v0.t
      {170, 240, 245, 244, 0, 0, 0, 0, 255, 255, 255, 255, 119, 119, 0, 0},
      0x0055'ffff'0000'cccf,
      0x00ff'00ab,
      0x0f0f,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x790a'b457,  // Vmsgtu.vi  v8, v16, -0xb, v0.t
                                  {0, 240, 240, 240, 0, 0, 0, 0, 170, 255, 250, 170, 0, 0, 0, 0},
                                  0x0000'0cf0'0000'ccc0,
                                  0x0000'00a8,
                                  0x0000,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsgt) {
  TestVectorMaskTargetInstruction(
      0x7d00'c457,  // Vmsgt.vx v8, v16, x1, v0.t
      {255, 255, 255, 255, 255, 223, 0, 255, 255, 255, 255, 255, 119, 119, 0, 0},
      0x0055'ffff'f0bf'ffff,
      0x00ff'cfff,
      0x0fbf,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x7d0a'b457,  // Vmsgt.vi  v8, v16, -0xb, v0.t
      {85, 255, 250, 251, 255, 223, 0, 255, 170, 255, 250, 170, 0, 0, 0, 0},
      0x0000'0cf0'f0bf'fff0,
      0x0000'cffc,
      0x00b0,
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmfge) {
  TestVectorMaskTargetInstruction(0x7d00'd457,  // Vmfge.vf v8, v16, f1, v0.t
                                  0x0000'0050,
                                  0x0020,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmacc) {
  TestVectorFloatInstruction(0xb188'1457,  // vfmacc.vv v8, v16, v24, v0.t
                             {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x6a1f'cefd, 0x7629'21c4, 0x6232'9db4, 0x6e3c'70f9},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5551, 0xd66b'bbc8, 0x5555'5555, 0x5555'5037},
                              {0xfaad'fde4, 0xff80'0000, 0xf2c2'c69a, 0xfecd'99e3},
                              {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000}},
                             {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x75b4'9040'f9f1'ea75, 0x6dcb'c6d1'12f0'a99b},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0xd614'2330'4af7'4c90, 0x5555'5555'5555'5555},
                              {0xfff0'0000'0000'0000, 0xfe5b'5815'60f1'ac51},
                              {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0xb100'd457,  // vfmacc.vf v8, f1, v16, v0.t
                             {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'550e, 0x5555'0ca1},
                              {0x550b'37bf, 0xd895'6354, 0xdc99'df27, 0xe09c'b3a3},
                              {0xe49f'8677, 0xe8a2'594a, 0xeca5'2c1d, 0xf0a7'fef0},
                              {0xf4aa'd1c3, 0xf8ad'a496, 0xfcb0'7768, 0xff80'0000}},
                             {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0xd780'0dff'a493'9082, 0xdf85'b3a5'4a3b'e0d2},
                              {0xe790'194a'efe1'8677, 0xef95'bef0'9587'2c1d},
                              {0xf7a0'2496'3b2c'd1c3, 0xffa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfnmacc) {
  TestVectorFloatInstruction(0xb588'1457,  // vfnmacc.vv v8, v16, v24, v0.t
                             {{0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xea1f'cefd, 0xf629'21c4, 0xe232'9db4, 0xee3c'70f9},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5551, 0x566b'bbc8, 0xd555'5555, 0xd555'5037},
                              {0x7aad'fde4, 0x7f80'0000, 0x72c2'c69a, 0x7ecd'99e3},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xf5b4'9040'f9f1'ea75, 0xedcb'c6d1'12f0'a99b},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0x5614'2330'4af7'4c90, 0xd555'5555'5555'5555},
                              {0x7ff0'0000'0000'0000, 0x7e5b'5815'60f1'ac51},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0xb500'd457,  // vfnmacc.vf v8, f1, v16, v0.t
                             {{0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'550e, 0xd555'0ca1},
                              {0xd50b'37bf, 0x5895'6354, 0x5c99'df27, 0x609c'b3a3},
                              {0x649f'8677, 0x68a2'594a, 0x6ca5'2c1d, 0x70a7'fef0},
                              {0x74aa'd1c3, 0x78ad'a496, 0x7cb0'7768, 0x7f80'0000}},
                             {{0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0x5780'0dff'a493'9082, 0x5f85'b3a5'4a3b'e0d2},
                              {0x6790'194a'efe1'8677, 0x6f95'bef0'9587'2c1d},
                              {0x77a0'2496'3b2c'd1c3, 0x7fa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfwmacc) {
  __m128i dst_result = {0x0000'0000'0000'0000, 0x0000'0000'0000'0000};
  TestWideningVectorFloatInstruction(0xf188'1457,  // vfwmacc.vv v8, v16, v24, v0.t
                                     {{0x3330'e53c'6480'0000, 0x34b2'786b'bbc5'4900},
                                      {0x3234'1766'da4a'6200, 0x33b5'cab6'2d6c'4800},
                                      {0x3937'92ba'5bd0'8000, 0x3ab9'666a'779a'0d00},
                                      {0x383b'4565'd61f'6600, 0x39bd'3935'e5bd'8800},
                                      {0x3f3f'423b'5522'0000, 0x40c0'ab36'1ab7'e880},
                                      {0x3e41'bab3'e9fa'b500, 0x3fc2'd4dc'5007'e400},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf100'd457,  // vfwmacc.vf v8, f1, v16, v0.t
                                     {{0xb886'f0ad'0000'0000, 0xb907'a561'b400'0000},
                                      {0xb988'5a16'6800'0000, 0xba09'0ecb'1c00'0000},
                                      {0xba89'c37f'd000'0000, 0xbb0a'7834'8400'0000},
                                      {0xbb8b'2ce9'3800'0000, 0xbc0b'e19d'ec00'0000},
                                      {0xbc8c'9652'a000'0000, 0xbd0d'4b07'5400'0000},
                                      {0xbd8d'ffbc'0800'0000, 0xbe0e'b470'bc00'0000},
                                      {0xbe8f'6925'7000'0000, 0xbf10'0eed'1200'0000},
                                      {0xbf90'6947'6c00'0000, 0xc010'c3a1'c600'0000}},
                                     kVectorCalculationsSource,
                                     dst_result);

  dst_result = {0x401c'6666'6666'6666, 0x401c'6666'6666'6666};
  TestWideningVectorFloatInstruction(0xf188'1457,  // vfwmacc.vv v8, v16, v24, v0.t
                                     {{0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'66e3'6f53'baee, 0x40c0'aec2'e784'b54d},
                                      {0x401c'6666'66f4'3c05, 0x401c'fd0d'48e6'a586},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf100'd457,  // vfwmacc.vf v8, f1, v16, v0.t
                                     {{0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6657},
                                      {0x401c'6666'6666'5766, 0x401c'6666'6657'0c2e},
                                      {0x401c'6666'56b1'd3ae, 0x401c'6656'5779'5466},
                                      {0x401c'55fd'1efa'6666, 0x4007'4589'40cc'cccc}},
                                     kVectorCalculationsSource,
                                     dst_result);
}

TEST_F(Riscv64InterpreterTest, TestVfwnmacc) {
  __m128i dst_result = {0x0000'0000'0000'0000, 0x0000'0000'0000'0000};
  TestWideningVectorFloatInstruction(0xf588'1457,  // vfwnmacc.vv v8, v16, v24, v0.t
                                     {{0xb330'e53c'6480'0000, 0xb4b2'786b'bbc5'4900},
                                      {0xb234'1766'da4a'6200, 0xb3b5'cab6'2d6c'4800},
                                      {0xb937'92ba'5bd0'8000, 0xbab9'666a'779a'0d00},
                                      {0xb83b'4565'd61f'6600, 0xb9bd'3935'e5bd'8800},
                                      {0xbf3f'423b'5522'0000, 0xc0c0'ab36'1ab7'e880},
                                      {0xbe41'bab3'e9fa'b500, 0xbfc2'd4dc'5007'e400},
                                      {0xc543'f9df'a83a'4000, 0xc6c5'2438'7aa3'4a80},
                                      {0xc446'53b6'69e6'3700, 0xc5c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf500'd457,  // vfwnmacc.vf v8, f1, v16, v0.t
                                     {{0x3886'f0ad'0000'0000, 0x3907'a561'b400'0000},
                                      {0x3988'5a16'6800'0000, 0x3a09'0ecb'1c00'0000},
                                      {0x3a89'c37f'd000'0000, 0x3b0a'7834'8400'0000},
                                      {0x3b8b'2ce9'3800'0000, 0x3c0b'e19d'ec00'0000},
                                      {0x3c8c'9652'a000'0000, 0x3d0d'4b07'5400'0000},
                                      {0x3d8d'ffbc'0800'0000, 0x3e0e'b470'bc00'0000},
                                      {0x3e8f'6925'7000'0000, 0x3f10'0eed'1200'0000},
                                      {0x3f90'6947'6c00'0000, 0x4010'c3a1'c600'0000}},
                                     kVectorCalculationsSource,
                                     dst_result);

  dst_result = {0x401c'6666'6666'6666, 0x401c'6666'6666'6666};
  TestWideningVectorFloatInstruction(0xf588'1457,  // vfwnmacc.vv v8, v16, v24, v0.t
                                     {{0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'66e3'6f53'baee, 0xc0c0'aec2'e784'b54d},
                                      {0xc01c'6666'66f4'3c05, 0xc01c'fd0d'48e6'a586},
                                      {0xc543'f9df'a83a'4000, 0xc6c5'2438'7aa3'4a80},
                                      {0xc446'53b6'69e6'3700, 0xc5c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf500'd457,  // vfwnmacc.vf v8, f1, v16, v0.t
                                     {{0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6657},
                                      {0xc01c'6666'6666'5766, 0xc01c'6666'6657'0c2e},
                                      {0xc01c'6666'56b1'd3ae, 0xc01c'6656'5779'5466},
                                      {0xc01c'55fd'1efa'6666, 0xc007'4589'40cc'cccc}},
                                     kVectorCalculationsSource,
                                     dst_result);
}

TEST_F(Riscv64InterpreterTest, TestVfwmsac) {
  __m128i dst_result = {0x0000'0000'0000'0000, 0x0000'0000'0000'0000};
  TestWideningVectorFloatInstruction(0xf988'1457,  // vfwmsac.vv v8, v16, v24, v0.t
                                     {{0x3330'e53c'6480'0000, 0x34b2'786b'bbc5'4900},
                                      {0x3234'1766'da4a'6200, 0x33b5'cab6'2d6c'4800},
                                      {0x3937'92ba'5bd0'8000, 0x3ab9'666a'779a'0d00},
                                      {0x383b'4565'd61f'6600, 0x39bd'3935'e5bd'8800},
                                      {0x3f3f'423b'5522'0000, 0x40c0'ab36'1ab7'e880},
                                      {0x3e41'bab3'e9fa'b500, 0x3fc2'd4dc'5007'e400},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf900'd457,  // vfwmsac.vf v8, f1, v16, v0.t
                                     {{0xb886'f0ad'0000'0000, 0xb907'a561'b400'0000},
                                      {0xb988'5a16'6800'0000, 0xba09'0ecb'1c00'0000},
                                      {0xba89'c37f'd000'0000, 0xbb0a'7834'8400'0000},
                                      {0xbb8b'2ce9'3800'0000, 0xbc0b'e19d'ec00'0000},
                                      {0xbc8c'9652'a000'0000, 0xbd0d'4b07'5400'0000},
                                      {0xbd8d'ffbc'0800'0000, 0xbe0e'b470'bc00'0000},
                                      {0xbe8f'6925'7000'0000, 0xbf10'0eed'1200'0000},
                                      {0xbf90'6947'6c00'0000, 0xc010'c3a1'c600'0000}},
                                     kVectorCalculationsSource,
                                     dst_result);

  dst_result = {0x401c'6666'6666'6666, 0x401c'6666'6666'6666};
  TestWideningVectorFloatInstruction(0xf988'1457,  // vfwmsac.vv v8, v16, v24, v0.t
                                     {{0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'65e9'5d79'11de, 0x40c0'a7a9'4deb'1bb3},
                                      {0xc01c'6666'65d8'90c7, 0xc01b'cfbf'83e6'2746},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf900'd457,  // vfwmsac.vf v8, f1, v16, v0.t
                                     {{0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6675},
                                      {0xc01c'6666'6666'7566, 0xc01c'6666'6675'c09e},
                                      {0xc01c'6666'761a'f91e, 0xc01c'6676'7553'7866},
                                      {0xc01c'76cf'add2'6666, 0xc026'9504'1633'3333}},
                                     kVectorCalculationsSource,
                                     dst_result);
}

TEST_F(Riscv64InterpreterTest, TestVfwnmsac) {
  __m128i dst_result = {0x0000'0000'0000'0000, 0x0000'0000'0000'0000};
  TestWideningVectorFloatInstruction(0xfd88'1457,  // vfwnmsac.vv v8, v16, v24, v0.t
                                     {{0xb330'e53c'6480'0000, 0xb4b2'786b'bbc5'4900},
                                      {0xb234'1766'da4a'6200, 0xb3b5'cab6'2d6c'4800},
                                      {0xb937'92ba'5bd0'8000, 0xbab9'666a'779a'0d00},
                                      {0xb83b'4565'd61f'6600, 0xb9bd'3935'e5bd'8800},
                                      {0xbf3f'423b'5522'0000, 0xc0c0'ab36'1ab7'e880},
                                      {0xbe41'bab3'e9fa'b500, 0xbfc2'd4dc'5007'e400},
                                      {0xc543'f9df'a83a'4000, 0xc6c5'2438'7aa3'4a80},
                                      {0xc446'53b6'69e6'3700, 0xc5c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xfd00'd457,  // vfwnmsac.vf v8, f1, v16, v0.t
                                     {{0x3886'f0ad'0000'0000, 0x3907'a561'b400'0000},
                                      {0x3988'5a16'6800'0000, 0x3a09'0ecb'1c00'0000},
                                      {0x3a89'c37f'd000'0000, 0x3b0a'7834'8400'0000},
                                      {0x3b8b'2ce9'3800'0000, 0x3c0b'e19d'ec00'0000},
                                      {0x3c8c'9652'a000'0000, 0x3d0d'4b07'5400'0000},
                                      {0x3d8d'ffbc'0800'0000, 0x3e0e'b470'bc00'0000},
                                      {0x3e8f'6925'7000'0000, 0x3f10'0eed'1200'0000},
                                      {0x3f90'6947'6c00'0000, 0x4010'c3a1'c600'0000}},
                                     kVectorCalculationsSource,
                                     dst_result);

  dst_result = {0x401c'6666'6666'6666, 0x401c'6666'6666'6666};
  TestWideningVectorFloatInstruction(0xfd88'1457,  // vfwnmsac.vv v8, v16, v24, v0.t
                                     {{0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'65e9'5d79'11de, 0xc0c0'a7a9'4deb'1bb3},
                                      {0x401c'6666'65d8'90c7, 0x401b'cfbf'83e6'2746},
                                      {0xc543'f9df'a83a'4000, 0xc6c5'2438'7aa3'4a80},
                                      {0xc446'53b6'69e6'3700, 0xc5c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xfd00'd457,  // vfwnmsac.vf v8, f1, v16, v0.t
                                     {{0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6675},
                                      {0x401c'6666'6666'7566, 0x401c'6666'6675'c09e},
                                      {0x401c'6666'761a'f91e, 0x401c'6676'7553'7866},
                                      {0x401c'76cf'add2'6666, 0x4026'9504'1633'3333}},
                                     kVectorCalculationsSource,
                                     dst_result);
}

TEST_F(Riscv64InterpreterTest, TestVfmsac) {
  TestVectorFloatInstruction(0xb988'1457,  // vfmsac.vv v8, v16, v24, v0.t
                             {{0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0x6a1f'cefd, 0x7629'21c4, 0x6232'9db3, 0x6e3c'70f9},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5559, 0xd6ab'3339, 0xd555'5555, 0xd555'5a73},
                              {0xfaad'fde4, 0xff80'0000, 0xf2c2'c69a, 0xfecd'99e3},
                              {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000}},
                             {{0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0x75b4'9040'f9f1'ea75, 0x6dcb'c6d1'12f0'a99b},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd614'25da'f5a1'f73b, 0xd555'5555'5555'5555},
                              {0xfff0'0000'0000'0000, 0xfe5b'5815'60f1'ac51},
                              {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0xb900'd457,  // vfmsac.vf v8, f1, v16, v0.t
                             {{0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'559c, 0xd555'9e09},
                              {0xd58f'b976, 0xd898'b8aa, 0xdc99'e27d, 0xe09c'b3a6},
                              {0xe49f'8678, 0xe8a2'594a, 0xeca5'2c1d, 0xf0a7'fef0},
                              {0xf4aa'd1c3, 0xf8ad'a496, 0xfcb0'7768, 0xff80'0000}},
                             {{0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd780'0dff'a498'e5d7, 0xdf85'b3a5'4a3b'e0d2},
                              {0xe790'194a'efe1'8678, 0xef95'bef0'9587'2c1d},
                              {0xf7a0'2496'3b2c'd1c3, 0xffa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfnmsac) {
  TestVectorFloatInstruction(0xbd88'1457,  // vfnmsac.vv v8, v16, v24, v0.t
                             {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0xea1f'cefd, 0xf629'21c4, 0xe232'9db3, 0xee3c'70f9},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5559, 0x56ab'3339, 0x5555'5555, 0x5555'5a73},
                              {0x7aad'fde4, 0x7f80'0000, 0x72c2'c69a, 0x7ecd'99e3},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0xf5b4'9040'f9f1'ea75, 0xedcb'c6d1'12f0'a99b},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5614'25da'f5a1'f73b, 0x5555'5555'5555'5555},
                              {0x7ff0'0000'0000'0000, 0x7e5b'5815'60f1'ac51},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0xbd00'd457,  // vfnmsac.vf v8, f1, v16, v0.t
                             {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'559c, 0x5555'9e09},
                              {0x558f'b976, 0x5898'b8aa, 0x5c99'e27d, 0x609c'b3a6},
                              {0x649f'8678, 0x68a2'594a, 0x6ca5'2c1d, 0x70a7'fef0},
                              {0x74aa'd1c3, 0x78ad'a496, 0x7cb0'7768, 0x7f80'0000}},
                             {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5780'0dff'a498'e5d7, 0x5f85'b3a5'4a3b'e0d2},
                              {0x6790'194a'efe1'8678, 0x6f95'bef0'9587'2c1d},
                              {0x77a0'2496'3b2c'd1c3, 0x7fa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmadd) {
  TestVectorFloatInstruction(0xa188'1457,  // vfmadd.vv v8, v16, v24, v0.t
                             {{0x98dd'a63a, 0x9e28'a06a, 0xa0e6'e462, 0xa4ed'95be},
                              {0xb624'b220, 0xbe2c'ba29, 0xb100'd4ec, 0xb504'308a},
                              {0xd644'd240, 0xde4c'da49, 0xc654'e5df, 0xce5c'ca7c},
                              {0xf664'f260, 0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78},
                              {0xd922'4bb5, 0xdd25'a463, 0xe128'fd11, 0xe52c'55bf},
                              {0xe92f'ae6d, 0xed33'071b, 0xf136'5fc9, 0xf539'b877},
                              {0xf93d'1125, 0xfd40'69d3, 0xff80'0000, 0xff80'0000},
                              {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000}},
                             {{0x9e0c'9a09'9d86'3e2c, 0xa474'5e08'5cb1'b0b0},
                              {0xbe2c'ba29'b624'b220, 0xb484'68bd'bcbc'6610},
                              {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
                              {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
                              {0xdcae'5c5b'af03'ac55, 0xe4b4'88dd'dcdc'8630},
                              {0xecbe'71c6'6f19'1715, 0xf4c4'9393'3ce7'3b90},
                              {0xfcce'8731'2f2e'81d5, 0xfff0'0000'0000'0000},
                              {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000}},

                             kVectorCalculationsSource);

  TestVectorFloatInstruction(0xa100'd457,  // vfmadd.vf v8, f1, v16, v0.t
                             {{0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5695'fffe, 0x5695'fe62},
                              {0x5694'5a5d, 0xd70b'd554, 0xdb5a'8e58, 0xdf5e'dd11},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0xd756'd554'd2da'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},

                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfnmadd) {
  TestVectorFloatInstruction(0xa588'1457,  // vfnmadd.vv v8, v16, v24, v0.t
                             {{0x18dd'a63a, 0x1e28'a06a, 0x20e6'e462, 0x24ed'95be},
                              {0x3624'b220, 0x3e2c'ba29, 0x3100'd4ec, 0x3504'308a},
                              {0x5644'd240, 0x5e4c'da49, 0x4654'e5df, 0x4e5c'ca7c},
                              {0x7664'f260, 0x7e6c'fa69, 0x6674'e271, 0x6e7c'ea78},
                              {0x5922'4bb5, 0x5d25'a463, 0x6128'fd11, 0x652c'55bf},
                              {0x692f'ae6d, 0x6d33'071b, 0x7136'5fc9, 0x7539'b877},
                              {0x793d'1125, 0x7d40'69d3, 0x7f80'0000, 0x7f80'0000},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0x1e0c'9a09'9d86'3e2c, 0x2474'5e08'5cb1'b0b0},
                              {0x3e2c'ba29'b624'b220, 0x3484'68bd'bcbc'6610},
                              {0x5e4c'da49'd644'd240, 0x4e5c'ca58'c654'c251},
                              {0x7e6c'fa69'f664'f260, 0x6e7c'ea78'e674'e271},
                              {0x5cae'5c5b'af03'ac55, 0x64b4'88dd'dcdc'8630},
                              {0x6cbe'71c6'6f19'1715, 0x74c4'9393'3ce7'3b90},
                              {0x7cce'8731'2f2e'81d5, 0x7ff0'0000'0000'0000},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSource);

  TestVectorFloatInstruction(0xa500'd457,  // vfmadd.vf v8, f1, v16, v0.t
                             {{0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd695'fffe, 0xd695'fe62},
                              {0xd694'5a5d, 0x570b'd554, 0x5b5a'8e58, 0x5f5e'dd11},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0x5756'd554'd2da'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmsub) {
  TestVectorFloatInstruction(0xa988'1457,  // vfmsub.vv v8, v16, v24, v0.t
                             {{0x98d5'5d1a, 0x1de1'2750, 0xa0e6'e462, 0xa4ed'95be},
                              {0x3624'b220, 0x3e2c'ba29, 0xb100'd4e6, 0xb504'2aa4},
                              {0x5644'd240, 0x5e4c'da49, 0x4654'9ec3, 0x4e5c'ca34},
                              {0x7664'f260, 0x7e6c'fa69, 0x6674'e271, 0x6e7c'ea78},
                              {0xd922'4bb5, 0xdd25'a463, 0xe128'fd11, 0xe52c'55bf},
                              {0xe92f'ae6d, 0xed33'071b, 0xf136'5fc9, 0xf539'b877},
                              {0xf93d'1125, 0xfd40'69d3, 0xff80'0000, 0xff80'0000},
                              {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000}},
                             {{0x1e0c'9a09'8e82'e5d4, 0xa474'5e08'5cb1'b0b0},
                              {0x3e2c'ba29'b624'b220, 0xb484'68bd'bcbc'6610},
                              {0x5e4c'da49'd644'd240, 0x4e5c'ca58'c654'c251},
                              {0x7e6c'fa69'f664'f260, 0x6e7c'ea78'e674'e271},
                              {0xdcae'5c5b'af03'ac55, 0xe4b4'88dd'dcdc'8630},
                              {0xecbe'71c6'6f19'1715, 0xf4c4'9393'3ce7'3b90},
                              {0xfcce'8731'2f2e'81d5, 0xfff0'0000'0000'0000},
                              {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000}},
                             kVectorCalculationsSource);

  TestVectorFloatInstruction(0xa900'd457,  // vfmsub.vf v8, f1, v16, v0.t
                             {{0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0001, 0x5696'019d},
                              {0x5697'a5a2, 0x5790'eaaa, 0x5b5b'2458, 0x5f5e'dda7},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x5756'd554'd3ca'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfnmsub) {
  TestVectorFloatInstruction(0xad88'1457,  // vfnmsub.vv v8, v16, v24, v0.t
                             {{0x18d5'5d1a, 0x9de1'2750, 0x20e6'e462, 0x24ed'95be},
                              {0xb624'b220, 0xbe2c'ba29, 0x3100'd4e6, 0x3504'2aa4},
                              {0xd644'd240, 0xde4c'da49, 0xc654'9ec3, 0xce5c'ca34},
                              {0xf664'f260, 0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78},
                              {0x5922'4bb5, 0x5d25'a463, 0x6128'fd11, 0x652c'55bf},
                              {0x692f'ae6d, 0x6d33'071b, 0x7136'5fc9, 0x7539'b877},
                              {0x793d'1125, 0x7d40'69d3, 0x7f80'0000, 0x7f80'0000},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0x9e0c'9a09'8e82'e5d4, 0x2474'5e08'5cb1'b0b0},
                              {0xbe2c'ba29'b624'b220, 0x3484'68bd'bcbc'6610},
                              {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
                              {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
                              {0x5cae'5c5b'af03'ac55, 0x64b4'88dd'dcdc'8630},
                              {0x6cbe'71c6'6f19'1715, 0x74c4'9393'3ce7'3b90},
                              {0x7cce'8731'2f2e'81d5, 0x7ff0'0000'0000'0000},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSource);

  TestVectorFloatInstruction(0xad00'd457,  // vfnmsub.vf v8, f1, v16, v0.t
                             {{0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0001, 0xd696'019d},
                              {0xd697'a5a2, 0xd790'eaaa, 0xdb5b'2458, 0xdf5e'dda7},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd756'd554'd3ca'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVbrev8) {
  TestVectorInstruction(
      0x4984'2457,  // vbrev8.v v8, v24, v0.t
      {{160, 15, 160, 15, 160, 15, 160, 15, 2, 2, 2, 2, 255, 255, 255, 255},
       {136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136},
       {143, 255, 143, 255, 143, 255, 143, 255, 143, 255, 143, 255, 143, 255, 143, 255},
       {6, 70, 38, 102, 150, 86, 54, 118, 142, 78, 46, 110, 30, 94, 62, 126},
       {1, 65, 33, 97, 145, 81, 49, 113, 137, 73, 41, 105, 25, 89, 57, 121},
       {5, 69, 37, 101, 149, 85, 53, 117, 141, 77, 45, 109, 29, 93, 61, 125},
       {3, 67, 35, 99, 147, 83, 51, 115, 139, 75, 43, 107, 27, 91, 59, 123},
       {7, 71, 39, 103, 151, 87, 55, 119, 143, 79, 47, 111, 31, 95, 63, 127}},
      {{0x0fa0, 0x0fa0, 0x0fa0, 0x0fa0, 0x0202, 0x0202, 0xffff, 0xffff},
       {0x8888, 0x8888, 0x8888, 0x8888, 0x8888, 0x8888, 0x8888, 0x8888},
       {0xff8f, 0xff8f, 0xff8f, 0xff8f, 0xff8f, 0xff8f, 0xff8f, 0xff8f},
       {0x4606, 0x6626, 0x5696, 0x7636, 0x4e8e, 0x6e2e, 0x5e1e, 0x7e3e},
       {0x4101, 0x6121, 0x5191, 0x7131, 0x4989, 0x6929, 0x5919, 0x7939},
       {0x4505, 0x6525, 0x5595, 0x7535, 0x4d8d, 0x6d2d, 0x5d1d, 0x7d3d},
       {0x4303, 0x6323, 0x5393, 0x7333, 0x4b8b, 0x6b2b, 0x5b1b, 0x7b3b},
       {0x4707, 0x6727, 0x5797, 0x7737, 0x4f8f, 0x6f2f, 0x5f1f, 0x7f3f}},
      {{0x0fa0'0fa0, 0x0fa0'0fa0, 0x0202'0202, 0xffff'ffff},
       {0x8888'8888, 0x8888'8888, 0x8888'8888, 0x8888'8888},
       {0xff8f'ff8f, 0xff8f'ff8f, 0xff8f'ff8f, 0xff8f'ff8f},
       {0x6626'4606, 0x7636'5696, 0x6e2e'4e8e, 0x7e3e'5e1e},
       {0x6121'4101, 0x7131'5191, 0x6929'4989, 0x7939'5919},
       {0x6525'4505, 0x7535'5595, 0x6d2d'4d8d, 0x7d3d'5d1d},
       {0x6323'4303, 0x7333'5393, 0x6b2b'4b8b, 0x7b3b'5b1b},
       {0x6727'4707, 0x7737'5797, 0x6f2f'4f8f, 0x7f3f'5f1f}},
      {{0x0fa0'0fa0'0fa0'0fa0, 0xffff'ffff'0202'0202},
       {0x8888'8888'8888'8888, 0x8888'8888'8888'8888},
       {0xff8f'ff8f'ff8f'ff8f, 0xff8f'ff8f'ff8f'ff8f},
       {0x7636'5696'6626'4606, 0x7e3e'5e1e'6e2e'4e8e},
       {0x7131'5191'6121'4101, 0x7939'5919'6929'4989},
       {0x7535'5595'6525'4505, 0x7d3d'5d1d'6d2d'4d8d},
       {0x7333'5393'6323'4303, 0x7b3b'5b1b'6b2b'4b8b},
       {0x7737'5797'6727'4707, 0x7f3f'5f1f'6f2f'4f8f}},
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmin) {
  TestVectorFloatInstruction(0x1100'd457,  // vfmin.vf v8, v16, f1, v0.t
                             {{0xf005'f005, 0xf005'f005, 0x4040'4040, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
                              {0xaaaa'aaaa, 0xaaaa'aaaa, 0x1111'1111, 0x1111'1111},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb},
                              {0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9}},
                             {{0xf005'f005'f005'f005, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
                              {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
                              {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},
                             kVectorComparisonSource);
  TestVectorFloatInstruction(0x110c'1457,  // vfmin.vv v8,v16,v24,v0.t
                             {{0xf005'f005, 0xf005'f005, 0x4040'4040, 0x7fc0'0000},
                              {0x1111'1111, 0x1111'1111, 0x1111'1111, 0x1111'1111},
                              {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
                              {0xaaaa'aaaa, 0xaaaa'aaaa, 0x1111'1111, 0x1111'1111},
                              {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
                              {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
                              {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
                              {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
                             {{0xf005'f005'f005'f005, 0x7ff8'0000'0000'0000},
                              {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
                              {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
                              {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
                              {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
                              {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
                              {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
                              {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
                             kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmax) {
  TestVectorFloatInstruction(0x1900'd457,  // vfmax.vf v8, v16, f1, v0.t
                             {{0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'40b4, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000}},
                             {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000}},
                             kVectorComparisonSource);
  TestVectorFloatInstruction(0x190c'1457,  // vfmax.vv v8,v16,v24,v0.t
                             {{0xf005'f005, 0xf005'f005, 0x4040'4040, 0x7fc0'0000},
                              {0x40b4'40b4, 0x1111'1111, 0x40b4'0000, 0x1111'1111},
                              {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
                              {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
                              {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
                              {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
                              {0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb},
                              {0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9}},
                             {{0xf005'f005'f005'f005, 0x7ff8'0000'0000'0000},
                              {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
                              {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
                              {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
                              {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
                              {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
                              {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
                              {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},
                             kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVfsgnj) {
  TestVectorFloatInstruction(0x210c'1457,  // vfsgnj.vv v8, v16, v24, v0.t
                             {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                              {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
                              {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
                              {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
                              {0x4342'c140, 0x4746'c544, 0x4b4a'c948, 0x4f4e'cd4c},
                              {0x5352'd150, 0x5756'd554, 0x5b5a'd958, 0x5f5e'dd5c},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                              {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
                              {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
                              {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
                              {0x4746'c544'c342'c140, 0x4f4e'cd4c'cb4a'c948},
                              {0x5756'd554'd352'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x2100'd457,  // vfsgnj.vf v8, v16, f1, v0.t
                             {{0x0302'8100, 0x0706'8504, 0x0b0a'8908, 0x0f0e'8d0c},
                              {0x1312'9110, 0x1716'9514, 0x1b1a'9918, 0x1f1e'9d1c},
                              {0x2322'a120, 0x2726'a524, 0x2b2a'a928, 0x2f2e'ad2c},
                              {0x3332'b130, 0x3736'b534, 0x3b3a'b938, 0x3f3e'bd3c},
                              {0x4342'c140, 0x4746'c544, 0x4b4a'c948, 0x4f4e'cd4c},
                              {0x5352'd150, 0x5756'd554, 0x5b5a'd958, 0x5f5e'dd5c},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0x0706'8504'8302'8100, 0x0f0e'8d0c'8b0a'8908},
                              {0x1716'9514'9312'9110, 0x1f1e'9d1c'9b1a'9918},
                              {0x2726'a524'a322'a120, 0x2f2e'ad2c'ab2a'a928},
                              {0x3736'b534'b332'b130, 0x3f3e'bd3c'bb3a'b938},
                              {0x4746'c544'c342'c140, 0x4f4e'cd4c'cb4a'c948},
                              {0x5756'd554'd352'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x250c'1457,  // vfsgnjn.vv v8, v16, v24, v0.t
                             {{0x0302'8100, 0x0706'8504, 0x0b0a'8908, 0x0f0e'8d0c},
                              {0x1312'9110, 0x1716'9514, 0x1b1a'9918, 0x1f1e'9d1c},
                              {0x2322'a120, 0x2726'a524, 0x2b2a'a928, 0x2f2e'ad2c},
                              {0x3332'b130, 0x3736'b534, 0x3b3a'b938, 0x3f3e'bd3c},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x0706'8504'8302'8100, 0x0f0e'8d0c'8b0a'8908},
                              {0x1716'9514'9312'9110, 0x1f1e'9d1c'9b1a'9918},
                              {0x2726'a524'a322'a120, 0x2f2e'ad2c'ab2a'a928},
                              {0x3736'b534'b332'b130, 0x3f3e'bd3c'bb3a'b938},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x2500'd457,  // vfsgnjn.vf v8, v16, f1, v0.t
                             {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                              {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
                              {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
                              {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                              {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
                              {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
                              {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x290c'1457,  // vfsgnjx.vv v8, v16, v24, v0.t
                             {{0x0302'8100, 0x0706'8504, 0x0b0a'8908, 0x0f0e'8d0c},
                              {0x1312'9110, 0x1716'9514, 0x1b1a'9918, 0x1f1e'9d1c},
                              {0x2322'a120, 0x2726'a524, 0x2b2a'a928, 0x2f2e'ad2c},
                              {0x3332'b130, 0x3736'b534, 0x3b3a'b938, 0x3f3e'bd3c},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x0706'8504'8302'8100, 0x0f0e'8d0c'8b0a'8908},
                              {0x1716'9514'9312'9110, 0x1f1e'9d1c'9b1a'9918},
                              {0x2726'a524'a322'a120, 0x2f2e'ad2c'ab2a'a928},
                              {0x3736'b534'b332'b130, 0x3f3e'bd3c'bb3a'b938},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x2900'd457,  // vfsgnjx.vf v8, v16, f1, v0.t
                             {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                              {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
                              {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
                              {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                              {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
                              {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
                              {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
}

// Note that the expected test outputs for v[f]merge.vXm are identical to those for v[f]mv.v.X.
// This happens because v[f]merge.vXm is just a v[f]mv.v.X with mask (second operand is not used
// by v[f]mv.v.X but the difference between v[f]merge.vXm and v[f]mv.v.X is captured in masking
// logic within TestVectorInstruction itself via the parameter TestVectorInstructionMode::kVMerge
// for V[f]merge/V[f]mv).
TEST_F(Riscv64InterpreterTest, TestVmerge) {
  TestVectorMergeFloatInstruction(0x5d00'd457,  // Vfmerge.vfm v8, v16, f1, v0
                                  {{0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000}},
                                  {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000}},
                                  kVectorCalculationsSource);
  TestVectorMergeInstruction(
      0x5d0c'0457,  // Vmerge.vvm v8, v16, v24, v0
      {{0, 146, 4, 150, 9, 154, 12, 158, 17, 130, 20, 134, 24, 138, 28, 142},
       {32, 178, 36, 182, 41, 186, 44, 190, 49, 162, 52, 166, 56, 170, 60, 174},
       {64, 210, 68, 214, 73, 218, 76, 222, 81, 194, 84, 198, 88, 202, 92, 206},
       {96, 242, 100, 246, 105, 250, 108, 254, 113, 226, 116, 230, 120, 234, 124, 238},
       {128, 18, 132, 22, 137, 26, 140, 30, 145, 2, 148, 6, 152, 10, 156, 14},
       {160, 50, 164, 54, 169, 58, 172, 62, 177, 34, 180, 38, 184, 42, 188, 46},
       {192, 82, 196, 86, 201, 90, 204, 94, 209, 66, 212, 70, 216, 74, 220, 78},
       {224, 114, 228, 118, 233, 122, 236, 126, 241, 98, 244, 102, 248, 106, 252, 110}},
      {{0x9200, 0x9604, 0x9a09, 0x9e0c, 0x8211, 0x8614, 0x8a18, 0x8e1c},
       {0xb220, 0xb624, 0xba29, 0xbe2c, 0xa231, 0xa634, 0xaa38, 0xae3c},
       {0xd240, 0xd644, 0xda49, 0xde4c, 0xc251, 0xc654, 0xca58, 0xce5c},
       {0xf260, 0xf664, 0xfa69, 0xfe6c, 0xe271, 0xe674, 0xea78, 0xee7c},
       {0x1280, 0x1684, 0x1a89, 0x1e8c, 0x0291, 0x0694, 0x0a98, 0x0e9c},
       {0x32a0, 0x36a4, 0x3aa9, 0x3eac, 0x22b1, 0x26b4, 0x2ab8, 0x2ebc},
       {0x52c0, 0x56c4, 0x5ac9, 0x5ecc, 0x42d1, 0x46d4, 0x4ad8, 0x4edc},
       {0x72e0, 0x76e4, 0x7ae9, 0x7eec, 0x62f1, 0x66f4, 0x6af8, 0x6efc}},
      {{0x9604'9200, 0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18},
       {0xb624'b220, 0xbe2c'ba29, 0xa634'a231, 0xae3c'aa38},
       {0xd644'd240, 0xde4c'da49, 0xc654'c251, 0xce5c'ca58},
       {0xf664'f260, 0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78},
       {0x1684'1280, 0x1e8c'1a89, 0x0694'0291, 0x0e9c'0a98},
       {0x36a4'32a0, 0x3eac'3aa9, 0x26b4'22b1, 0x2ebc'2ab8},
       {0x56c4'52c0, 0x5ecc'5ac9, 0x46d4'42d1, 0x4edc'4ad8},
       {0x76e4'72e0, 0x7eec'7ae9, 0x66f4'62f1, 0x6efc'6af8}},
      {{0x9e0c'9a09'9604'9200, 0x8e1c'8a18'8614'8211},
       {0xbe2c'ba29'b624'b220, 0xae3c'aa38'a634'a231},
       {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
       {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
       {0x1e8c'1a89'1684'1280, 0x0e9c'0a98'0694'0291},
       {0x3eac'3aa9'36a4'32a0, 0x2ebc'2ab8'26b4'22b1},
       {0x5ecc'5ac9'56c4'52c0, 0x4edc'4ad8'46d4'42d1},
       {0x7eec'7ae9'76e4'72e0, 0x6efc'6af8'66f4'62f1}},
      kVectorCalculationsSource);
  TestVectorMergeInstruction(
      0x5d00'c457,  // Vmerge.vxm v8, v16, x1, v0
      {{170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170}},
      {{0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa}},
      {{0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSource);
  TestVectorMergeInstruction(
      0x5d0a'b457,  // Vmerge.vim v8, v16, -0xb, v0
      {{245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245}},
      {{0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5}},
      {{0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5}},
      {{0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVslide1down) {
  // Where the element at the top gets inserted will depend on VLMUL so we use
  // TestVectorPermutationInstruction instead of TestVectorInstruction.

  // VLMUL = 0
  TestVectorPermutationInstruction(
      0x3d80'e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 0xaa}, {}, {}, {}, {}, {}, {}, {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0xaaaa},
       {},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0xaaaa'aaaa}, {}, {}, {}, {}, {}, {}, {}},
      {{0x1e1c'1a18'1614'1211, 0xaaaa'aaaa'aaaa'aaaa}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 1
  TestVectorPermutationInstruction(
      0x3d80'e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 0xaa},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0xaaaa},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0xaaaa'aaaa},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0xaaaa'aaaa'aaaa'aaaa},
       {},
       {},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 2
  TestVectorPermutationInstruction(
      0x3d80'e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 64},
       {66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94, 96},
       {98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126, 0xaa},
       {},
       {},
       {},
       {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x4240},
       {0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c, 0x6260},
       {0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c, 0xaaaa},
       {},
       {},
       {},
       {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x4644'4240},
       {0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58, 0x6664'6260},
       {0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78, 0xaaaa'aaaa},
       {},
       {},
       {},
       {}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0xaaaa'aaaa'aaaa'aaaa},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 3
  TestVectorPermutationInstruction(
      0x3d80'e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 64},
       {66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94, 96},
       {98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126, 128},
       {130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158, 160},
       {162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190, 192},
       {194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222, 224},
       {226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254, 0xaa}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x4240},
       {0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c, 0x6260},
       {0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c, 0x8280},
       {0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c, 0xa2a0},
       {0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc, 0xc2c0},
       {0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc, 0xe2e0},
       {0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc, 0xaaaa}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x4644'4240},
       {0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58, 0x6664'6260},
       {0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78, 0x8684'8280},
       {0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98, 0xa6a4'a2a0},
       {0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8, 0xc6c4'c2c0},
       {0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8, 0xe6e4'e2e0},
       {0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8, 0xaaaa'aaaa}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x8e8c'8a89'8684'8280},
       {0x9e9c'9a98'9694'9291, 0xaeac'aaa9'a6a4'a2a0},
       {0xbebc'bab8'b6b4'b2b1, 0xcecc'cac9'c6c4'c2c0},
       {0xdedc'dad8'd6d4'd2d1, 0xeeec'eae9'e6e4'e2e0},
       {0xfefc'faf8'f6f4'f2f1, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 4
  TestVectorPermutationInstruction(0x3d80'e457,  // vslide1down.vx v8, v24, x1, v0.t
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/4,
                                   /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
                                   /*skip=*/0,
                                   /*ignore_vma_for_last=*/true,
                                   /*last_elem_is_x1=*/true);

  // VLMUL = 5
  TestVectorPermutationInstruction(
      0x3d80'e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 0xaa, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0xaaaa, 0, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/5,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 6
  TestVectorPermutationInstruction(
      0x3d80'e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 0xaa, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0x0604, 0xaaaa, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0xaaaa'aaaa, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/6,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 7
  TestVectorPermutationInstruction(
      0x3d80'e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 0xaa, 0, 0, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0x0604, 0x0a09, 0x0e0c, 0xaaaa, 0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0x0e0c'0a09, 0xaaaa'aaaa, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/7,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);
}

TEST_F(Riscv64InterpreterTest, TestVfslide1up) {
  TestVectorFloatInstruction(0x3980'd457,  // vfslide1up.vf v8, v24, f1, v0.t
                             {{0x40b4'0000, 0x9604'9200, 0x9e0c'9a09, 0x8614'8211},
                              {0x8e1c'8a18, 0xb624'b220, 0xbe2c'ba29, 0xa634'a231},
                              {0xae3c'aa38, 0xd644'd240, 0xde4c'da49, 0xc654'c251},
                              {0xce5c'ca58, 0xf664'f260, 0xfe6c'fa69, 0xe674'e271},
                              {0xee7c'ea78, 0x1684'1280, 0x1e8c'1a89, 0x0694'0291},
                              {0x0e9c'0a98, 0x36a4'32a0, 0x3eac'3aa9, 0x26b4'22b1},
                              {0x2ebc'2ab8, 0x56c4'52c0, 0x5ecc'5ac9, 0x46d4'42d1},
                              {0x4edc'4ad8, 0x76e4'72e0, 0x7eec'7ae9, 0x66f4'62f1}},
                             {{0x4016'8000'0000'0000, 0x9e0c'9a09'9604'9200},
                              {0x8e1c'8a18'8614'8211, 0xbe2c'ba29'b624'b220},
                              {0xae3c'aa38'a634'a231, 0xde4c'da49'd644'd240},
                              {0xce5c'ca58'c654'c251, 0xfe6c'fa69'f664'f260},
                              {0xee7c'ea78'e674'e271, 0x1e8c'1a89'1684'1280},
                              {0x0e9c'0a98'0694'0291, 0x3eac'3aa9'36a4'32a0},
                              {0x2ebc'2ab8'26b4'22b1, 0x5ecc'5ac9'56c4'52c0},
                              {0x4edc'4ad8'46d4'42d1, 0x7eec'7ae9'76e4'72e0}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfslide1down) {
  // Where the element at the top gets inserted will depend on VLMUL so we use
  // TestVectorFloatPermutationInstruction instead of TestVectorFloatInstruction.

  // VLMUL = 0
  TestVectorFloatPermutationInstruction(
      0x3d80'd457,  // vfslide1down.vf v8, v24, f1, v0.t
      {{0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18, 0x40b4'0000}, {}, {}, {}, {}, {}, {}, {}},
      {{0x8e1c'8a18'8614'8211, 0x4016'8000'0000'0000}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSource,
      /*vlmul=*/0,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_f1=*/true);

  // VLMUL = 1
  TestVectorFloatPermutationInstruction(0x3d80'd457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18, 0xb624'b220},
                                         {0xbe2c'ba29, 0xa634'a231, 0xae3c'aa38, 0x40b4'0000},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {}},
                                        {{0x8e1c'8a18'8614'8211, 0xbe2c'ba29'b624'b220},
                                         {0xae3c'aa38'a634'a231, 0x4016'8000'0000'0000},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/1,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 2
  TestVectorFloatPermutationInstruction(0x3d80'd457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18, 0xb624'b220},
                                         {0xbe2c'ba29, 0xa634'a231, 0xae3c'aa38, 0xd644'd240},
                                         {0xde4c'da49, 0xc654'c251, 0xce5c'ca58, 0xf664'f260},
                                         {0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78, 0x40b4'0000},
                                         {},
                                         {},
                                         {},
                                         {}},
                                        {{0x8e1c'8a18'8614'8211, 0xbe2c'ba29'b624'b220},
                                         {0xae3c'aa38'a634'a231, 0xde4c'da49'd644'd240},
                                         {0xce5c'ca58'c654'c251, 0xfe6c'fa69'f664'f260},
                                         {0xee7c'ea78'e674'e271, 0x4016'8000'0000'0000},
                                         {},
                                         {},
                                         {},
                                         {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/2,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 3
  TestVectorFloatPermutationInstruction(0x3d80'd457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18, 0xb624'b220},
                                         {0xbe2c'ba29, 0xa634'a231, 0xae3c'aa38, 0xd644'd240},
                                         {0xde4c'da49, 0xc654'c251, 0xce5c'ca58, 0xf664'f260},
                                         {0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78, 0x1684'1280},
                                         {0x1e8c'1a89, 0x0694'0291, 0x0e9c'0a98, 0x36a4'32a0},
                                         {0x3eac'3aa9, 0x26b4'22b1, 0x2ebc'2ab8, 0x56c4'52c0},
                                         {0x5ecc'5ac9, 0x46d4'42d1, 0x4edc'4ad8, 0x76e4'72e0},
                                         {0x7eec'7ae9, 0x66f4'62f1, 0x6efc'6af8, 0x40b4'0000}},
                                        {{0x8e1c'8a18'8614'8211, 0xbe2c'ba29'b624'b220},
                                         {0xae3c'aa38'a634'a231, 0xde4c'da49'd644'd240},
                                         {0xce5c'ca58'c654'c251, 0xfe6c'fa69'f664'f260},
                                         {0xee7c'ea78'e674'e271, 0x1e8c'1a89'1684'1280},
                                         {0x0e9c'0a98'0694'0291, 0x3eac'3aa9'36a4'32a0},
                                         {0x2ebc'2ab8'26b4'22b1, 0x5ecc'5ac9'56c4'52c0},
                                         {0x4edc'4ad8'46d4'42d1, 0x7eec'7ae9'76e4'72e0},
                                         {0x6efc'6af8'66f4'62f1, 0x4016'8000'0000'0000}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/3,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 4
  TestVectorFloatPermutationInstruction(0x3d80'd457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/4,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 5
  TestVectorFloatPermutationInstruction(0x3d80'd457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/5,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 6
  TestVectorFloatPermutationInstruction(0x3d80'd457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x40b4'0000, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/6,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 7
  TestVectorFloatPermutationInstruction(
      0x3d80'd457,  // vfslide1down.vf v8, v24, f1, v0.t
      {{0x9e0c'9a09, 0x40b4'0000, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0x4016'8000'0000'0000, 0}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSource,
      /*vlmul=*/7,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_f1=*/true);
}

TEST_F(Riscv64InterpreterTest, TestVseXX) {
  TestVseXX(0x8427,  // vse8.v v8, (x1), v0.t
            {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
             {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
             {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
             {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
             {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
             {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
             {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
             {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}},
            {},
            {},
            {},
            0,
            kVectorCalculationsSourceLegacy);
  TestVseXX(0xd427,  // vse16.v v8, (x1), v0.t
            {},
            {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
             {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
             {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
             {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
             {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
             {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
             {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
             {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
            {},
            {},
            1,
            kVectorCalculationsSourceLegacy);
  TestVseXX(0xe427,  // vse32.v v8, (x1), v0.t
            {},
            {},
            {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
             {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
             {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
             {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
             {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
             {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
             {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
             {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
            {},
            2,
            kVectorCalculationsSourceLegacy);
  TestVseXX(0xf427,  // vse64.v v8, (x1), v0.t
            {},
            {},
            {},
            {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
             {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
             {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
             {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
             {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
             {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
             {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
             {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
            3,
            kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVleXX) {
  TestVleXX(0x8407,  // vle8.v v8, (x1), v0.t
            {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
             {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
             {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
             {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
             {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
             {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
             {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
             {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}},
            {},
            {},
            {},
            0,
            kVectorCalculationsSourceLegacy);
  TestVleXX(0xd407,  // vle16.v v8, (x1), v0.t
            {},
            {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
             {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
             {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
             {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
             {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
             {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
             {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
             {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
            {},
            {},
            1,
            kVectorCalculationsSourceLegacy);
  TestVleXX(0xe407,  // vle32.v v8, (x1), v0.t
            {},
            {},
            {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
             {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
             {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
             {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
             {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
             {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
             {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
             {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
            {},
            2,
            kVectorCalculationsSourceLegacy);
  TestVleXX(0xf407,  // vle64.v v8, (x1), v0.t
            {},
            {},
            {},
            {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
             {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
             {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
             {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
             {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
             {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
             {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
             {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
            3,
            kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVleXXff) {
  TestVleXXff(0x0100'8407, 6, 0, 6);  // vle8ff.v v8, (x1), v0.t
  TestVleXXff(0x0100'8407, 8, 0, 8);
  TestVleXXff(0x0100'8407, 16, 0, 16);
  TestVleXXff(0x0100'8407, 32, 0, 32);
  TestVleXXff(0x0100'8407, 255, 0, 128);  // All 128 bytes accessible.

  TestVleXXff(0x0100'd407, 6, 1, 3);  // vle16ff.v v8, (x1), v0.t
  TestVleXXff(0x0100'd407, 8, 1, 4);
  TestVleXXff(0x0100'd407, 16, 1, 8);
  TestVleXXff(0x0100'd407, 32, 1, 16);

  TestVleXXff(0x0100'e407, 6, 2, 1);  // vle32ff.v v8, (x1), v0.t
  TestVleXXff(0x0100'e407, 8, 2, 2);
  TestVleXXff(0x0100'e407, 16, 2, 4);
  TestVleXXff(0x0100'e407, 32, 2, 8);
  TestVleXXff(0x0100'e407, 64, 2, 16);

  TestVleXXff(0x0100'f407, 8, 3, 1);  // vle64ff.v v8, (x1), v0.t
  TestVleXXff(0x0100'f407, 16, 3, 2);
  TestVleXXff(0x0100'f407, 32, 3, 4);
  TestVleXXff(0x0100'f407, 64, 3, 8);

  TestVleXXff(0x0100'f407, 6, 3, 16, true);  // Should raise exception and not change VL
}

TEST_F(Riscv64InterpreterTest, TestVcpopm) {
  TestVXmXXsInstruction(
      0x4108'20d7,  // vcpop.m x1, v16, v0.t
      {0,           /* default value when vl=0 */
       0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  2,  2,  3,  3,  3,  3,  3,
       3,  3,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  7,  7,  7,  7,  7,  7,  8,  8,  9,  9,
       9,  9,  9,  10, 10, 11, 12, 12, 12, 12, 12, 12, 13, 14, 15, 15, 15, 15, 15, 16, 16, 16,
       16, 17, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20, 20, 21, 21, 22, 22, 22, 22, 22,
       23, 24, 24, 25, 25, 25, 25, 26, 26, 26, 27, 28, 28, 28, 28, 28, 29, 29, 30, 31, 31, 31,
       31, 32, 32, 33, 34, 35, 35, 35, 35, 35, 36, 37, 38, 39, 39, 39, 39, 40},
      {0, /* default value when vl=0 */
       0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  2,  2,  3,  3,  3,  3,  3,
       3,  3,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,
       7,  7,  7,  8,  8,  8,  9,  9,  9,  9,  9,  9,  10, 10, 11, 11, 11, 11, 11, 12, 12, 12,
       12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15,
       15, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 21, 21, 21, 21,
       21, 21, 21, 22, 23, 23, 23, 23, 23, 23, 23, 24, 24, 25, 25, 25, 25, 25},
      kVectorCalculationsSourceLegacy[0]);
}

TEST_F(Riscv64InterpreterTest, TestVfirstm) {
  TestVXmXXsInstruction(0x4108'a0d7,  // vfirst.m x1, v16, v0.t
                        {[0 ... 8] = ~0ULL, [9 ... 128] = 8},
                        {[0 ... 8] = ~0ULL, [9 ... 128] = 8},
                        kVectorCalculationsSourceLegacy[0]);
}

// For Vrgather the expectations for different VLMULs are very different, i.e. different vlmax
// values produce different results for the same src. So we have to call each of them explicitly.
// To produce none-all-zero expected_results, all registers index vectors(v24) have first half
// masked to have valid indexes (< vlmax); x1 is masked to be < vlmax; uimm is accordingly to be 0,
// 1 or 3.
TEST_F(Riscv64InterpreterTest, TestVrgather) {
  // VLMUL = 0
  TestVectorRegisterGather(0x310c'0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 2, 0, 6, 0, 10, 0, 14},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/0,
                           kVectorCalculationsSource);

  TestVectorRegisterGather(0x3100'c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}},
                           {{0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504}},
                           {{0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908}},
                           {{0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100}},
                           /*vlmul=*/0,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100'b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504}},
      {{0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908}},
      /*vlmul=*/0,
      kVectorCalculationsSource);

  // VLMUL = 1
  TestVectorRegisterGather(0x310c'0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 18, 4, 22, 137, 26, 12, 30, 145, 0, 20, 0, 24, 0, 28, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 2, 0, 6, 0, 10, 0, 14},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/1,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100'c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}},
                           {{0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514}},
                           {{0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908}},
                           {{0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110}},
                           /*vlmul=*/1,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100'b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504}},
      {{0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908}},
      /*vlmul=*/1,
      kVectorCalculationsSource);

  // VLMUL = 2
  TestVectorRegisterGather(0x310c'0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 18, 4, 22, 137, 26, 12, 30, 145, 0, 20, 0, 24, 0, 28, 0},
                            {32, 50, 36, 54, 169, 58, 44, 62, 177, 0, 52, 0, 56, 0, 60, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {32, 50, 36, 54, 169, 58, 44, 62, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 2, 0, 6, 0, 10, 0, 14},
                            {32, 50, 36, 54, 169, 58, 44, 62, 0, 34, 0, 38, 0, 42, 0, 46},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {32, 50, 36, 54, 169, 58, 44, 62, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/2,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100'c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}},
                           {{0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514}},
                           {{0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928}},
                           {{0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110}},
                           /*vlmul=*/2,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100'b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504}},
      {{0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908}},
      /*vlmul=*/2,
      kVectorCalculationsSource);

  // VLMUL = 3
  TestVectorRegisterGather(0x310c'0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 18, 4, 22, 137, 26, 12, 30, 145, 0, 20, 0, 24, 0, 28, 0},
                            {32, 50, 36, 54, 169, 58, 44, 62, 177, 0, 52, 0, 56, 0, 60, 0},
                            {64, 82, 68, 86, 201, 90, 76, 94, 209, 0, 84, 0, 88, 0, 92, 0},
                            {96, 114, 100, 118, 233, 122, 108, 126, 241, 0, 116, 0, 120, 0, 124, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 2, 0, 6, 0, 10, 0, 14},
                            {32, 50, 36, 54, 169, 58, 44, 62, 0, 34, 0, 38, 0, 42, 0, 46},
                            {64, 82, 68, 86, 201, 90, 76, 94, 0, 66, 0, 70, 0, 74, 0, 78},
                            {96, 114, 100, 118, 233, 122, 108, 126, 0, 98, 0, 102, 0, 106, 0, 110}},
                           {{0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0xc140, 0xc948, 0xd352, 0xd958, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0xc140, 0xc948, 0xd352, 0xd958, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0xc140, 0xc948, 0xd352, 0xd958, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0xc140, 0xc948, 0xd352, 0xd958, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/3,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100'c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}},
                           {{0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554}},
                           {{0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928}},
                           {{0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150}},
                           /*vlmul=*/3,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3101'b457,  // vrgather.vi v8,v16,3,v0.t
      {{131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131}},
      {{0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706}},
      {{0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c}},
      {{0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918}},
      /*vlmul=*/3,
      kVectorCalculationsSource);

  // VLMUL = 5
  TestVectorRegisterGather(0x310c'0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {},
                           {},
                           /*vlmul=*/5,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100'c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100}},
                           {},
                           {},
                           /*vlmul=*/5,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100'b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {},
      {},
      /*vlmul=*/5,
      kVectorCalculationsSource);

  // VLMUL = 6
  TestVectorRegisterGather(0x310c'0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000}},
                           {},
                           /*vlmul=*/6,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100'c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
                           {{0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100}},
                           {{0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100}},
                           {},
                           /*vlmul=*/6,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100'b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {},
      /*vlmul=*/6,
      kVectorCalculationsSource);

  // VLMUL = 7
  TestVectorRegisterGather(0x310c'0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 2, 0, 6, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/7,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100'c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
                           {{0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504}},
                           {{0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100}},
                           {{0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100}},
                           /*vlmul=*/7,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100'b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      /*vlmul=*/7,
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVadc) {
  TestVectorCarryInstruction(
      0x410c'0457,  //  vadc.vvm v8,v16,v24,v0
      {{1, 19, 7, 26, 13, 32, 18, 38, 26, 11, 31, 17, 37, 24, 42, 30},
       {49, 68, 54, 74, 61, 80, 67, 85, 74, 59, 79, 66, 84, 72, 90, 78},
       {97, 115, 103, 121, 110, 128, 114, 134, 121, 108, 127, 113, 133, 119, 139, 126},
       {145, 163, 151, 170, 157, 176, 162, 182, 170, 155, 175, 161, 181, 167, 187, 174},
       {193, 211, 199, 217, 206, 224, 210, 230, 218, 204, 222, 210, 229, 216, 235, 221},
       {241, 3, 247, 10, 253, 16, 3, 22, 9, 252, 15, 2, 21, 7, 27, 14},
       {33, 52, 38, 58, 46, 64, 50, 70, 58, 44, 63, 49, 69, 55, 75, 61},
       {81, 100, 87, 105, 94, 112, 99, 118, 105, 92, 110, 98, 116, 104, 123, 109}},
      {{0x1301, 0x1906, 0x1f0e, 0x2513, 0x0b19, 0x111f, 0x1724, 0x1d2b},
       {0x4331, 0x4936, 0x4f3e, 0x5542, 0x3b4a, 0x414f, 0x4754, 0x4d5b},
       {0x7361, 0x7967, 0x7f6d, 0x8573, 0x6b79, 0x717f, 0x7785, 0x7d8a},
       {0xa391, 0xa996, 0xaf9e, 0xb5a3, 0x9ba9, 0xa1af, 0xa7b4, 0xadbb},
       {0xd3c1, 0xd9c6, 0xdfce, 0xe5d2, 0xcbda, 0xd1df, 0xd7e4, 0xddeb},
       {0x03f0, 0x09f7, 0x0ffe, 0x1602, 0xfc0a, 0x020e, 0x0815, 0x0e1b},
       {0x3421, 0x3a26, 0x402e, 0x4633, 0x2c39, 0x323f, 0x3844, 0x3e4b},
       {0x6451, 0x6a56, 0x705e, 0x7662, 0x5c6a, 0x626e, 0x6875, 0x6e7b}},
      {{0x1907'1301, 0x2513'1f0d, 0x111f'0b1a, 0x1d2b'1725},
       {0x4937'4330, 0x5543'4f3e, 0x414f'3b49, 0x4d5b'4755},
       {0x7967'7361, 0x8573'7f6d, 0x717f'6b7a, 0x7d8b'7784},
       {0xa997'a391, 0xb5a3'af9e, 0xa1af'9ba9, 0xadbb'a7b5},
       {0xd9c6'd3c1, 0xe5d2'dfce, 0xd1de'cbd9, 0xddea'd7e5},
       {0x09f7'03f0, 0x1603'0ffe, 0x020e'fc0a, 0x0e1b'0814},
       {0x3a27'3421, 0x4633'402d, 0x323f'2c3a, 0x3e4b'3845},
       {0x6a57'6450, 0x7663'705e, 0x626f'5c69, 0x6e7b'6875}},
      {{0x2513'1f0e'1907'1301, 0x1d2b'1725'111f'0b19},
       {0x5543'4f3e'4937'4331, 0x4d5b'4755'414f'3b4a},
       {0x8573'7f6e'7967'7360, 0x7d8b'7785'717f'6b7a},
       {0xb5a3'af9e'a997'a390, 0xadbb'a7b5'a1af'9baa},
       {0xe5d2'dfcd'd9c6'd3c1, 0xddea'd7e4'd1de'cbd9},
       {0x1603'0ffe'09f7'03f1, 0x0e1b'0815'020e'fc09},
       {0x4633'402e'3a27'3421, 0x3e4b'3845'323f'2c3a},
       {0x7663'705e'6a57'6450, 0x6e7b'6875'626f'5c6a}},
      kVectorCalculationsSource);

  TestVectorCarryInstruction(
      0x4100'c457,  // vadc.vxm	v8,v16,x1,v0
      {{171, 43, 173, 46, 174, 48, 176, 50, 179, 51, 181, 53, 183, 56, 184, 58},
       {187, 60, 188, 62, 190, 64, 193, 65, 195, 67, 197, 70, 198, 72, 200, 74},
       {203, 75, 205, 77, 207, 80, 208, 82, 210, 84, 213, 85, 215, 87, 217, 90},
       {219, 91, 221, 94, 222, 96, 224, 98, 227, 99, 229, 101, 231, 103, 233, 106},
       {235, 107, 237, 109, 239, 112, 240, 114, 243, 116, 244, 118, 247, 120, 249, 121},
       {251, 123, 253, 126, 254, 128, 1, 130, 2, 132, 5, 134, 7, 135, 9, 138},
       {11, 140, 12, 142, 15, 144, 16, 146, 19, 148, 21, 149, 23, 151, 25, 153},
       {27, 156, 29, 157, 31, 160, 33, 162, 34, 164, 36, 166, 38, 168, 41, 169}},
      {{0x2bab, 0x2dac, 0x2faf, 0x31b1, 0x33b2, 0x35b5, 0x37b6, 0x39b9},
       {0x3bbb, 0x3dbc, 0x3fbf, 0x41c0, 0x43c3, 0x45c5, 0x47c6, 0x49c9},
       {0x4bcb, 0x4dcd, 0x4fce, 0x51d1, 0x53d2, 0x55d5, 0x57d7, 0x59d8},
       {0x5bdb, 0x5ddc, 0x5fdf, 0x61e1, 0x63e2, 0x65e5, 0x67e6, 0x69e9},
       {0x6beb, 0x6dec, 0x6fef, 0x71f0, 0x73f3, 0x75f5, 0x77f6, 0x79f9},
       {0x7bfa, 0x7dfd, 0x7fff, 0x8200, 0x8403, 0x8604, 0x8807, 0x8a09},
       {0x8c0b, 0x8e0c, 0x900f, 0x9211, 0x9412, 0x9615, 0x9816, 0x9a19},
       {0x9c1b, 0x9e1c, 0xa01f, 0xa220, 0xa423, 0xa624, 0xa827, 0xaa29}},
      {{0x2dad'2bab, 0x31b1'2fae, 0x35b5'33b3, 0x39b9'37b7},
       {0x3dbd'3bba, 0x41c1'3fbf, 0x45c5'43c2, 0x49c9'47c7},
       {0x4dcd'4bcb, 0x51d1'4fce, 0x55d5'53d3, 0x59d9'57d6},
       {0x5ddd'5bdb, 0x61e1'5fdf, 0x65e5'63e2, 0x69e9'67e7},
       {0x6ded'6beb, 0x71f1'6fef, 0x75f5'73f2, 0x79f9'77f7},
       {0x7dfd'7bfa, 0x8201'7fff, 0x8605'8403, 0x8a09'8806},
       {0x8e0d'8c0b, 0x9211'900e, 0x9615'9413, 0x9a19'9817},
       {0x9e1d'9c1a, 0xa221'a01f, 0xa625'a422, 0xaa29'a827}},
      {{0x31b1'2faf'2dad'2bab, 0x39b9'37b7'35b5'33b2},
       {0x41c1'3fbf'3dbd'3bbb, 0x49c9'47c7'45c5'43c3},
       {0x51d1'4fcf'4dcd'4bca, 0x59d9'57d7'55d5'53d3},
       {0x61e1'5fdf'5ddd'5bda, 0x69e9'67e7'65e5'63e3},
       {0x71f1'6fef'6ded'6beb, 0x79f9'77f7'75f5'73f2},
       {0x8201'7fff'7dfd'7bfb, 0x8a09'8807'8605'8402},
       {0x9211'900f'8e0d'8c0b, 0x9a19'9817'9615'9413},
       {0xa221'a01f'9e1d'9c1a, 0xaa29'a827'a625'a423}},
      kVectorCalculationsSource);

  TestVectorCarryInstruction(
      0x4105'b457,  // vadc.vim v8,v16,0xb,v0
      {{12, 140, 14, 143, 15, 145, 17, 147, 20, 148, 22, 150, 24, 153, 25, 155},
       {28, 157, 29, 159, 31, 161, 34, 162, 36, 164, 38, 167, 39, 169, 41, 171},
       {44, 172, 46, 174, 48, 177, 49, 179, 51, 181, 54, 182, 56, 184, 58, 187},
       {60, 188, 62, 191, 63, 193, 65, 195, 68, 196, 70, 198, 72, 200, 74, 203},
       {76, 204, 78, 206, 80, 209, 81, 211, 84, 213, 85, 215, 88, 217, 90, 218},
       {92, 220, 94, 223, 95, 225, 98, 227, 99, 229, 102, 231, 104, 232, 106, 235},
       {108, 237, 109, 239, 112, 241, 113, 243, 116, 245, 118, 246, 120, 248, 122, 250},
       {124, 253, 126, 254, 128, 1, 130, 3, 131, 5, 133, 7, 135, 9, 138, 10}},
      {{0x810c, 0x830d, 0x8510, 0x8712, 0x8913, 0x8b16, 0x8d17, 0x8f1a},
       {0x911c, 0x931d, 0x9520, 0x9721, 0x9924, 0x9b26, 0x9d27, 0x9f2a},
       {0xa12c, 0xa32e, 0xa52f, 0xa732, 0xa933, 0xab36, 0xad38, 0xaf39},
       {0xb13c, 0xb33d, 0xb540, 0xb742, 0xb943, 0xbb46, 0xbd47, 0xbf4a},
       {0xc14c, 0xc34d, 0xc550, 0xc751, 0xc954, 0xcb56, 0xcd57, 0xcf5a},
       {0xd15b, 0xd35e, 0xd560, 0xd761, 0xd964, 0xdb65, 0xdd68, 0xdf6a},
       {0xe16c, 0xe36d, 0xe570, 0xe772, 0xe973, 0xeb76, 0xed77, 0xef7a},
       {0xf17c, 0xf37d, 0xf580, 0xf781, 0xf984, 0xfb85, 0xfd88, 0xff8a}},
      {{0x8302'810c, 0x8706'850f, 0x8b0a'8914, 0x8f0e'8d18},
       {0x9312'911b, 0x9716'9520, 0x9b1a'9923, 0x9f1e'9d28},
       {0xa322'a12c, 0xa726'a52f, 0xab2a'a934, 0xaf2e'ad37},
       {0xb332'b13c, 0xb736'b540, 0xbb3a'b943, 0xbf3e'bd48},
       {0xc342'c14c, 0xc746'c550, 0xcb4a'c953, 0xcf4e'cd58},
       {0xd352'd15b, 0xd756'd560, 0xdb5a'd964, 0xdf5e'dd67},
       {0xe362'e16c, 0xe766'e56f, 0xeb6a'e974, 0xef6e'ed78},
       {0xf372'f17b, 0xf776'f580, 0xfb7a'f983, 0xff7e'fd88}},
      {{0x8706'8504'8302'810c, 0x8f0e'8d0c'8b0a'8913},
       {0x9716'9514'9312'911c, 0x9f1e'9d1c'9b1a'9924},
       {0xa726'a524'a322'a12b, 0xaf2e'ad2c'ab2a'a934},
       {0xb736'b534'b332'b13b, 0xbf3e'bd3c'bb3a'b944},
       {0xc746'c544'c342'c14c, 0xcf4e'cd4c'cb4a'c953},
       {0xd756'd554'd352'd15c, 0xdf5e'dd5c'db5a'd963},
       {0xe766'e564'e362'e16c, 0xef6e'ed6c'eb6a'e974},
       {0xf776'f574'f372'f17b, 0xff7e'fd7c'fb7a'f984}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVsbc) {
  TestVectorCarryInstruction(
      0x490c'0457,  // vsb.vvm	v8,v16,v24,v0
      {{255, 17, 1, 18, 5, 20, 6, 22, 8, 249, 9, 251, 11, 252, 14, 254},
       {15, 32, 18, 34, 21, 36, 21, 39, 24, 9, 25, 10, 28, 12, 30, 14},
       {31, 49, 33, 51, 36, 52, 38, 54, 41, 24, 41, 27, 43, 29, 45, 30},
       {47, 65, 49, 66, 53, 68, 54, 70, 56, 41, 57, 43, 59, 45, 61, 46},
       {63, 81, 65, 83, 68, 84, 70, 86, 72, 56, 74, 58, 75, 60, 77, 63},
       {79, 97, 81, 98, 85, 100, 85, 102, 89, 72, 89, 74, 91, 77, 93, 78},
       {95, 112, 98, 114, 100, 116, 102, 118, 104, 88, 105, 91, 107, 93, 109, 95},
       {111, 128, 113, 131, 116, 132, 117, 134, 121, 104, 122, 106, 124, 108, 125, 111}},
      {{0x10ff, 0x1302, 0x1504, 0x1705, 0xf909, 0xfb09, 0xfd0c, 0xff0d},
       {0x210f, 0x2312, 0x2514, 0x2716, 0x0918, 0x0b19, 0x0d1c, 0x0f1d},
       {0x311f, 0x3321, 0x3525, 0x3725, 0x1929, 0x1b29, 0x1d2b, 0x1f2e},
       {0x412f, 0x4332, 0x4534, 0x4735, 0x2939, 0x2b39, 0x2d3c, 0x2f3d},
       {0x513f, 0x5342, 0x5544, 0x5746, 0x3948, 0x3b49, 0x3d4c, 0x3f4d},
       {0x6150, 0x6351, 0x6554, 0x6756, 0x4958, 0x4b5a, 0x4d5b, 0x4f5d},
       {0x715f, 0x7362, 0x7564, 0x7765, 0x5969, 0x5b69, 0x5d6c, 0x5f6d},
       {0x816f, 0x8372, 0x8574, 0x8776, 0x6978, 0x6b7a, 0x6d7b, 0x6f7d}},
      {{0x1302'10ff, 0x1706'1505, 0xfb09'f908, 0xff0d'fd0b},
       {0x2312'2110, 0x2716'2514, 0x0b1a'0919, 0x0f1e'0d1b},
       {0x3322'311f, 0x3726'3525, 0x1b2a'1928, 0x1f2e'1d2c},
       {0x4332'412f, 0x4736'4534, 0x2b3a'2939, 0x2f3e'2d3b},
       {0x5341'513f, 0x5745'5544, 0x3b49'3949, 0x3f4d'3d4b},
       {0x6351'6150, 0x6755'6554, 0x4b59'4958, 0x4f5d'4d5c},
       {0x7361'715f, 0x7765'7565, 0x5b69'5968, 0x5f6d'5d6b},
       {0x8371'8170, 0x8775'8574, 0x6b79'6979, 0x6f7d'6d7b}},
      {{0x1706'1505'1302'10ff, 0xff0d'fd0b'fb09'f909},
       {0x2716'2515'2312'210f, 0x0f1e'0d1c'0b1a'0918},
       {0x3726'3525'3322'3120, 0x1f2e'1d2c'1b2a'1928},
       {0x4736'4535'4332'4130, 0x2f3e'2d3c'2b3a'2938},
       {0x5745'5544'5341'513f, 0x3f4d'3d4b'3b49'3949},
       {0x6755'6554'6351'614f, 0x4f5d'4d5b'4b59'4959},
       {0x7765'7564'7361'715f, 0x5f6d'5d6b'5b69'5968},
       {0x8775'8574'8371'8170, 0x6f7d'6d7b'6b79'6978}},
      kVectorCalculationsSource);

  TestVectorCarryInstruction(
      0x4900'c457,  // vsbc.vxm	v8,v16,x1,v0
      {{169, 41, 167, 38, 166, 36, 164, 34, 161, 33, 159, 31, 157, 28, 156, 26},
       {153, 24, 152, 22, 150, 20, 147, 19, 145, 17, 143, 14, 142, 12, 140, 10},
       {137, 9, 135, 7, 133, 4, 132, 2, 130, 0, 127, 255, 125, 253, 123, 250},
       {121, 249, 119, 246, 118, 244, 116, 242, 113, 241, 111, 239, 109, 237, 107, 234},
       {105, 233, 103, 231, 101, 228, 100, 226, 97, 224, 96, 222, 93, 220, 91, 219},
       {89, 217, 87, 214, 86, 212, 83, 210, 82, 208, 79, 206, 77, 205, 75, 202},
       {73, 200, 72, 198, 69, 196, 68, 194, 65, 192, 63, 191, 61, 189, 59, 187},
       {57, 184, 55, 183, 53, 180, 51, 178, 50, 176, 48, 174, 46, 172, 43, 171}},
      {{0x29a9, 0x27a8, 0x25a5, 0x23a3, 0x21a2, 0x1f9f, 0x1d9e, 0x1b9b},
       {0x1999, 0x1798, 0x1595, 0x1394, 0x1191, 0x0f8f, 0x0d8e, 0x0b8b},
       {0x0989, 0x0787, 0x0586, 0x0383, 0x0182, 0xff7f, 0xfd7d, 0xfb7c},
       {0xf979, 0xf778, 0xf575, 0xf373, 0xf172, 0xef6f, 0xed6e, 0xeb6b},
       {0xe969, 0xe768, 0xe565, 0xe364, 0xe161, 0xdf5f, 0xdd5e, 0xdb5b},
       {0xd95a, 0xd757, 0xd555, 0xd354, 0xd151, 0xcf50, 0xcd4d, 0xcb4b},
       {0xc949, 0xc748, 0xc545, 0xc343, 0xc142, 0xbf3f, 0xbd3e, 0xbb3b},
       {0xb939, 0xb738, 0xb535, 0xb334, 0xb131, 0xaf30, 0xad2d, 0xab2b}},
      {{0x27a8'29a9, 0x23a4'25a6, 0x1fa0'21a1, 0x1b9c'1d9d},
       {0x1798'199a, 0x1394'1595, 0x0f90'1192, 0x0b8c'0d8d},
       {0x0788'0989, 0x0384'0586, 0xff80'0181, 0xfb7b'fd7e},
       {0xf777'f979, 0xf373'f575, 0xef6f'f172, 0xeb6b'ed6d},
       {0xe767'e969, 0xe363'e565, 0xdf5f'e162, 0xdb5b'dd5d},
       {0xd757'd95a, 0xd353'd555, 0xcf4f'd151, 0xcb4b'cd4e},
       {0xc747'c949, 0xc343'c546, 0xbf3f'c141, 0xbb3b'bd3d},
       {0xb737'b93a, 0xb333'b535, 0xaf2f'b132, 0xab2b'ad2d}},
      {{0x23a4'25a6'27a8'29a9, 0x1b9c'1d9e'1fa0'21a2},
       {0x1394'1596'1798'1999, 0x0b8c'0d8e'0f90'1191},
       {0x0384'0586'0788'098a, 0xfb7b'fd7d'ff80'0181},
       {0xf373'f575'f777'f97a, 0xeb6b'ed6d'ef6f'f171},
       {0xe363'e565'e767'e969, 0xdb5b'dd5d'df5f'e162},
       {0xd353'd555'd757'd959, 0xcb4b'cd4d'cf4f'd152},
       {0xc343'c545'c747'c949, 0xbb3b'bd3d'bf3f'c141},
       {0xb333'b535'b737'b93a, 0xab2b'ad2d'af2f'b131}},
      kVectorCalculationsSource);
}
}  // namespace

}  // namespace berberis
