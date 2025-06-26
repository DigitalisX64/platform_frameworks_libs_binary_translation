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
#ifndef BERBERIS_RUNTIME_PRIMITIVES_HOST_CODE_H_
#define BERBERIS_RUNTIME_PRIMITIVES_HOST_CODE_H_

#include <cstdint>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"

namespace berberis {

// Pointer to host executable machine code.
using HostCode = const void*;

// Type used in translation cache and for host_entries
#if defined(__x86_64__)
using HostCodeAddr = uint32_t;

inline HostCodeAddr AsHostCodeAddr(HostCode host_code) {
  CHECK(IsInRange<HostCodeAddr>(bit_cast<uintptr_t>(host_code)));
  return static_cast<HostCodeAddr>(bit_cast<uintptr_t>(host_code));
}

inline HostCode AsHostCode(HostCodeAddr host_code_addr) {
  return bit_cast<HostCode>(uintptr_t{host_code_addr});
}
#else
// TODO(b/363611588): use uint32_t for other 64bit backends (arm64/riscv64)
using HostCodeAddr = uintptr_t;

inline HostCodeAddr AsHostCodeAddr(HostCode host_code) {
  return bit_cast<HostCodeAddr>(host_code);
}

inline HostCode AsHostCode(HostCodeAddr host_code_addr) {
  return bit_cast<HostCode>(host_code_addr);
}
#endif  // defined(__x86_64__)

constexpr HostCodeAddr kNullHostCodeAddr = 0;

template <typename T>
inline HostCode AsHostCode(T ptr) {
  return bit_cast<HostCode>(ptr);
}

template <typename T>
inline T AsFuncPtr(HostCode ptr) {
  return bit_cast<T>(const_cast<void*>(ptr));
}

// Note: ideally we would like the class to be a local class in the AsFuncPtr function below, but
// C++ doesn't allow that: local classes are not supposed to have template members.
class AsFuncPtrAdaptor;
AsFuncPtrAdaptor AsFuncPtr(HostCode ptr);
class [[nodiscard]] AsFuncPtrAdaptor {
  // Note: we need this helper to describe the operator on the next line. Otherwise the C++ syntax
  // parser becomes very confused. It has to come before the operator to help the parser, though.
  template <typename Result, typename... Args>
  using MakeFunctionType = Result (*)(Args...);

 public:
  template <typename Result, typename... Args>
  operator MakeFunctionType<Result, Args...>() {
    return bit_cast<MakeFunctionType<Result, Args...>>(ptr_);
  }

 private:
  AsFuncPtrAdaptor() = delete;
  AsFuncPtrAdaptor(const AsFuncPtrAdaptor&) = delete;
  AsFuncPtrAdaptor(AsFuncPtrAdaptor&&) = delete;
  AsFuncPtrAdaptor& operator=(const AsFuncPtrAdaptor&) = delete;
  AsFuncPtrAdaptor& operator=(AsFuncPtrAdaptor&&) = delete;
  constexpr explicit AsFuncPtrAdaptor(HostCode ptr) noexcept : ptr_(const_cast<void*>(ptr)) {}
  friend AsFuncPtrAdaptor AsFuncPtr(HostCode);
  void* ptr_;
};

// The result of this function is assumed to be assigned to a non-auto variable type, which will
// involve the conversion operator of AsFuncPtrAdaptor.
inline AsFuncPtrAdaptor AsFuncPtr(HostCode ptr) {
  return AsFuncPtrAdaptor{ptr};
}

struct HostCodePiece {
  HostCodeAddr code;
  uint32_t size;
};

}  // namespace berberis

#endif  // BERBERIS_RUNTIME_PRIMITIVES_HOST_CODE_H_
