/*
 * Copyright (C) 2017 The Android Open Source Project
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

#include "berberis/base/prctl_helpers.h"

#include <sys/prctl.h>

// These are not defined in our glibc, but are supported in kernel since 5.17 (but not necessarily
// enabled). It's always enabled in Android kernerls, but otherwise on Linux may be disabled
// depending on CONFIG_ANON_VMA_NAME boot config flag. So the caller needs to check the result to
// see if it actually worked.
#if !defined(PR_SET_VMA)
#define PR_SET_VMA 0x5356'4d41
#endif
#if !defined(PR_SET_VMA_ANON_NAME)
#define PR_SET_VMA_ANON_NAME 0
#endif

namespace berberis {

int SetVmaAnonName(void* addr, size_t size, const char* name) {
  return prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, addr, size, name);
}

}  // namespace berberis
