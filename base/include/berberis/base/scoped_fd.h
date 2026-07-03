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

#ifndef BERBERIS_BASE_SCOPED_FD_H
#define BERBERIS_BASE_SCOPED_FD_H

#include <unistd.h>

// region digitalis
#if defined(__ANDROID__)
#include <sys/syscall.h>

#include <android/fdsan.h>

#include "berberis/base/tracing.h"
#endif
// endregion

namespace berberis {

class ScopedFd {
 public:
  ScopedFd(int fd) : fd_{fd} {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd(ScopedFd&&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ScopedFd& operator=(ScopedFd&&) = delete;
  ~ScopedFd() { reset(-1); }

 private:
  void reset(int fd) {
    if (fd_ != -1) {
      // region digitalis
#if defined(__ANDROID__)
      // Berberis host-opens short-lived fds (e.g. TinyLoader::OpenFile during
      // ResetAllExecRegions in CloneGuestThread). Two fdsan hazards on close:
      //  - The fd number can carry a stale owner tag left behind by an earlier
      //    raw-syscall close of a tagged fd. A bare close() is
      //    android_fdsan_close_with_tag(fd, 0), which aborts on that stale tag
      //    ("expected to be unowned, actually owned by Parcel ...").
      //  - Our kernel fd can have been yanked (a wild guest close/close_range)
      //    and the number REUSED by a live host owner (Fence, Parcel,
      //    unique_fd). Closing here would destroy the new owner's fd, and
      //    closing with the CURRENT tag would also strip its tag, making the
      //    owner's own close abort later ("... actually unowned").
      // A nonzero tag on our number always means a host object owns the slot
      // now — abandon instead of closing. (ScopedFd itself never tags, and
      // Berberis' tagged internal fds — see fd.h — are always closed with
      // their tag, so no stale Berberis tag can sit on a reused slot.) An
      // untagged fd gets a raw syscall close, which never enters fdsan's
      // error paths.
      if (android_fdsan_get_owner_tag(fd_) == 0) {
        syscall(__NR_close, fd_);
      } else {
        TRACE("ScopedFd %d reused by a tagged host owner; abandoning close", fd_);
      }
#else
      // endregion
      close(fd_);
      // region digitalis
#endif
      // endregion
    }

    fd_ = fd;
  }

  int fd_;
};

}  // namespace berberis

#endif  // BERBERIS_BASE_SCOPED_FD_H
