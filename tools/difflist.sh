#!/bin/bash
#
#
# Copyright (C) 2018 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Note: have to be run from the root of the repository and you must ensure that
# both remotes goog/mirror-aosp-main and goog/main exists.

aosp_branch=goog/mirror-aosp-main
local_branch=goog/main

set -eu

if [[ -d "frameworks/libs/binary_translation" ]]; then
  cd "frameworks/libs/binary_translation"
else
  while ! [[ -d ".git" ]]; do
    cd ..
    if [[ "$PWD" == "/" ]]; then
      echo "Couldn't find working directory"
      exit 1
    fi
  done
fi

readarray -t files < <(
  git diff "$aosp_branch" "$local_branch" |
  grep '^diff --git' |
  while read d g a b ; do
    echo "${b:2}"
  done
)
declare -A aosp_cls=() goog_cls=()
for file in "${files[@]}"; do
  readarray -t aosp_changes < <(
    git log "$aosp_branch" "$file" |
    grep '^commit ' |
    cut -b 8-
  )
  declare -A aosp_changes_map
  for aosp_change in "${aosp_changes[@]}"; do
    aosp_change_id="$(
      git log -n 1 "$aosp_change" | grep Change-Id: || true
    )"
    if ! [[ -z "${aosp_change_id}" ]]; then
      aosp_changes_map["$aosp_change_id"]=https://r.android.com/q/commit:"$aosp_change"
    fi
  done
  readarray -t goog_changes < <(
    git log "$local_branch" "$file" |
    grep '^commit ' |
    cut -b 8-
  )
  declare -A goog_changes_map
  for goog_change in "${goog_changes[@]}"; do
    goog_change_id="$(
      git log -n 1 "$goog_change" | grep Change-Id: || true
    )"
    if ! [[ -z "${goog_change_id}" ]]; then
      goog_changes_map["$goog_change_id"]=https://googleplex-android-review.googlesource.com/q/commit:"$goog_change"
    fi
  done

  for aosp_change_id in "${!aosp_changes_map[@]}"; do
    if [[ "${goog_changes_map["$aosp_change_id"]:-absent}" = "absent" ]] ; then
      aosp_cls[$aosp_change_id]="${aosp_changes_map[$aosp_change_id]}"
    fi
  done
  for goog_change_id in "${!goog_changes_map[@]}"; do
    if [[ "${aosp_changes_map["$goog_change_id"]:-absent}" = "absent" ]] ; then
       goog_cls[$goog_change_id]="${goog_changes_map[$goog_change_id]}"
    fi
  done
done
if ((${#aosp_cls[@]}>0)); then
  echo Only in AOSP:
  for cl in "${!aosp_cls[@]}" ; do
    echo "$cl => ${aosp_cls[$cl]}"
  done
fi
if ((${#goog_cls[@]}>0)); then
  echo Only in GOOG:
  for cl in "${!goog_cls[@]}" ; do
    echo "$cl => ${goog_cls[$cl]}"
  done
fi
