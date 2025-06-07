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

#ifndef BERBERIS_BASE_STRING_LITERAL_H_
#define BERBERIS_BASE_STRING_LITERAL_H_

#include <algorithm>
#include <cstddef>

namespace berberis {

// Note: we use that type as argument of template which means that “all base classes and non-static
// data members should be public and non-mutable”.
template <size_t N>
struct StringLiteral {
  constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }
  constexpr operator const char*() const { return value; }

  char value[N];
};

}  // namespace berberis

#endif  // BERBERIS_BASE_STRING_LITERAL_H_
