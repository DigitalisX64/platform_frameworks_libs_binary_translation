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
#include <android/fdsan.h>
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
      // Close with the fd's CURRENT fdsan owner tag instead of a bare close().
      // Berberis host-opens short-lived fds (e.g. TinyLoader::OpenFile during
      // ResetAllExecRegions in CloneGuestThread) that can land on an fd number
      // still carrying a stale fdsan owner tag — left by a previously-closed
      // owner (an Android Parcel / unique_fd) whose close did not clear the host
      // libc fdsan table. A bare close() is android_fdsan_close_with_tag(fd, 0),
      // which then aborts with "expected to be unowned, actually owned by
      // Parcel ...". Closing with the fd's current tag always passes the fdsan
      // check and clears the table entry, so the fd can be safely reused.
      android_fdsan_close_with_tag(fd_, android_fdsan_get_owner_tag(fd_));
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
