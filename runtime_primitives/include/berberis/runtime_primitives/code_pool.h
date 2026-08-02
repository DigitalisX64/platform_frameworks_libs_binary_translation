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

#ifndef BERBERIS_RUNTIME_PRIMITIVES_CODE_POOL_H_
#define BERBERIS_RUNTIME_PRIMITIVES_CODE_POOL_H_

// region digitalis
#include <unistd.h>
// endregion

#include <cstdint>
#include <mutex>
#include <type_traits>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/exec_region.h"
#include "berberis/base/tracing.h"
#include "berberis/runtime_primitives/exec_region_anonymous.h"
#include "berberis/runtime_primitives/host_code.h"

#if defined(__BIONIC__)
#include "berberis/runtime_primitives/exec_region_elf_backed.h"
#endif

namespace berberis {

// Code pool is an arena used to store fragments of generated code.
// TODO(b/232598137): Consider freeing allocated regions.
template <typename ExecRegionFactory>
class CodePool {
 public:
  CodePool()
      : exec_(ExecRegionFactory::Create(ExecRegionFactory::kExecRegionSize)),
        current_address_{exec_.begin()},
        detached_size_{0},
        // region digitalis
        owner_pid_{getpid()} {};
        // endregion

  // Not copyable or movable
  CodePool(const CodePool&) = delete;
  CodePool& operator=(const CodePool&) = delete;
  CodePool(CodePool&&) = delete;
  CodePool& operator=(CodePool&&) = delete;

  [[nodiscard]] HostCodeAddr Add(MachineCode* code) {
    std::lock_guard<std::mutex> lock(mutex_);

    // region digitalis
    // Fork safety. The executable region is a MAP_SHARED memfd, so it is shared
    // with every process forked from this one. The guest-clone path resets exec
    // regions in the child (ResetCurrentGuestThreadAfterFork -> ResetAllExecRegions),
    // but a host/framework-initiated fork -- in particular the Android zygote
    // (and Chromium's app-zygote) forking renderer processes -- does not go
    // through that path, so the child keeps sharing the parent's region. Multiple
    // renderer siblings then translate into the SAME shared offsets (each starts
    // from the parent's current_address_) and overwrite each other's code, so a
    // later dispatch jumps into corrupted/mid-instruction bytes and crashes
    // (observed as Chromium renderers dying with SIGSEGV/SIGILL "NO-RECOVERY").
    // Detect the fork lazily on the first install after the pid changes and give
    // this process a fresh private region. Parent translations stay valid: their
    // executable alias remains mapped read-only and their recovery_map_ entries
    // (inherited copy-on-write) still resolve.
    pid_t cur_pid = getpid();
    if (cur_pid != owner_pid_) {
      ResetExecRegion();
      owner_pid_ = cur_pid;
    }
    // endregion

    uint32_t size = code->install_size();

    // Align region start on 64-byte cache line to facilite more stable instruction fetch
    // performance on benchmarks. Region start is always a branch target, so this also ensures
    // 16-bytes alignment for branch targets recommended by Intel.
    // TODO(b/200327919): Try only doing this for heavy-optimized code to avoid extra gaps between
    // lite-translated regions.
    current_address_ = AlignUp(current_address_, 64);

    // Note that pointer arithmetic on nullptr is undefined behavior.
    CHECK_NE(current_address_, nullptr);
    if (exec_.end() < current_address_ + size) {
      // region digitalis
      // A failed exec-region allocation must not kill the guest app: the
      // interpreter can always run the region. Return the null code address so
      // the caller installs an interpreted entry instead. Keep the old region
      // attached (current_address_ still points into it) so previously
      // installed code stays valid and a later Add can retry.
      if (!TryResetExecRegion(size)) {
        return kNullHostCodeAddr;
      }
      // endregion
    }

    const uint8_t* result = current_address_;
    current_address_ += size;

    code->Install(&exec_, result, &recovery_map_);

    if (IsConfigFlagSet(kPrintCodePoolSize)) {
      TRACE("Code pool %p: new size %zu", this, GetTotalSize());
    }

    return AsHostCodeAddr(result);
  }

  [[nodiscard]] uintptr_t FindRecoveryCode(uintptr_t fault_addr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = recovery_map_.find(fault_addr);
    if (it != recovery_map_.end()) {
      return it->second;
    }
    return 0;
  }

  void ResetExecRegion(uint32_t size = ExecRegionFactory::kExecRegionSize) {
    detached_size_ += exec_.size();
    exec_.Detach();
    exec_ = ExecRegionFactory::Create(std::max(size, ExecRegionFactory::kExecRegionSize));
    current_address_ = exec_.begin();
  }

  // region digitalis
  // Non-fatal variant: returns false and leaves the pool untouched when the
  // host cannot provide a new executable region (memory pressure). See the
  // call site in Add().
  // Detects a factory that offers a non-fatal TryCreate. Factories without one
  // (test mocks, the ELF-backed prebuilt region) keep using Create.
  template <typename F, typename = void>
  struct HasTryCreate : std::false_type {};
  template <typename F>
  struct HasTryCreate<F, std::void_t<decltype(F::TryCreate(std::size_t{}))>> : std::true_type {};

  [[nodiscard]] bool TryResetExecRegion(uint32_t size = ExecRegionFactory::kExecRegionSize) {
    const uint32_t want = std::max(size, ExecRegionFactory::kExecRegionSize);
    ExecRegion fresh;
    if constexpr (HasTryCreate<ExecRegionFactory>::value) {
      fresh = ExecRegionFactory::TryCreate(want);
    } else {
      fresh = ExecRegionFactory::Create(want);
    }
    if (fresh.begin() == nullptr) {
      return false;
    }
    detached_size_ += exec_.size();
    exec_.Detach();
    exec_ = std::move(fresh);
    current_address_ = exec_.begin();
    return true;
  }
  // endregion

  size_t GetTotalSize() const { return detached_size_ + (current_address_ - exec_.begin()); }

 private:
  ExecRegion exec_;
  const uint8_t* current_address_;
  // TODO(b/232598137): have recovery map for each region instead!
  RecoveryMap recovery_map_;
  mutable std::mutex mutex_;
  size_t detached_size_;
  // region digitalis
  // The pid that owns the current exec region. A mismatch on Add() means this
  // process forked since the region was created; see the fork-safety note there.
  pid_t owner_pid_;
  // endregion
};

// Stored data for generated code.
class DataPool {
 public:
  // Returns default data pool.
  static DataPool* GetInstance();

  DataPool() = default;

  template <typename T>
  T* Add(const T& v) {
    return bit_cast<T*>(AddRaw(&v, sizeof(T)));
  }

  void* AddRaw(const void* ptr, uint32_t size);

 private:
  Arena arena_;
  std::mutex mutex_;
};

// Resets exec regions for all CodePools
void ResetAllExecRegions();

// Returns default code pool.
[[nodiscard]] CodePool<ExecRegionAnonymousFactory>* GetDefaultCodePoolInstance();
// Use cold code pool to avoid interleaving cold code with hot code, as it induces more cache and
// TLB misses (see go/ndkt-two-gear-overhead).
[[nodiscard]] CodePool<ExecRegionAnonymousFactory>* GetColdCodePoolInstance();

#if defined(__BIONIC__)
[[nodiscard]] CodePool<ExecRegionElfBackedFactory>* GetFunctionWrapperCodePoolInstance();
#else
[[nodiscard]] CodePool<ExecRegionAnonymousFactory>* GetFunctionWrapperCodePoolInstance();
#endif

}  // namespace berberis

#endif  // BERBERIS_RUNTIME_PRIMITIVES_CODE_POOL_H_
