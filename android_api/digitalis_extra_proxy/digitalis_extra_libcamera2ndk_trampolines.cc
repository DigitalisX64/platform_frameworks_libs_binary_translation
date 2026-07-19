/*
 * Copyright (C) 2026 utzcoz
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

// Digitalis-side extra trampolines for the two libcamera2ndk symbols the
// upstream proxy leaves as DoBadTrampoline:
//
//   ACameraCaptureSessionShared_startStreaming(
//       ACameraCaptureSession* sharedSession,
//       ACameraCaptureSession_captureCallbacksV2* callbacks,
//       int numOutputWindows, ANativeWindow** windows, int* captureSequenceId)
//       -> camera_status_t
//
//   ACameraCaptureSessionShared_logicalCamera_startStreaming(... same shape,
//       with ACameraCaptureSession_logicalCamera_captureCallbacksV2* callbacks)
//
// Both take a callbacks struct of one void* context + seven distinct guest
// callback function pointers. The auto-generator can't marshal it because the
// pointers are guest code addresses the host camera service will call back. We
// wrap each one with WrapGuestFunction (routing host->guest through
// RunGuestCall, which auto-attaches a guest thread to any host callback
// thread), build a host-callable callbacks struct, and forward to the dlsym'd
// host symbol. The remaining parameters are plain scalars / opaque pointers
// passed verbatim under LP64 (ANativeWindow** holds host-proxied window
// handles; int* captureSequenceId is an output the host writes directly).
//
// The per-field marshalling — the only part that can be wrong silently — is
// verified against the NdkCameraCaptureSession.h signatures in
// camera2ndk_shared_capture_callbacks_test.cc (run under berberis_arm64_host_tests).
// The shared-camera session itself cannot be opened on the Digitalis emulator
// (SYSTEM_CAMERA permission + no HAL shared-session support), so end-to-end
// exercise is impossible there; the host recorder test below is the
// ground-truth verification instead.
//
// Host functions are reached via the dlsym'd `callee` (not by name) so this
// static lib adds no libcamera2ndk link dependency to libberberis_arm64.so.

#if defined(__x86_64__)

#include "berberis/guest_abi/guest_params.h"
#include "berberis/proxy_loader/proxy_library_builder.h"
#include "berberis/runtime_primitives/host_code.h"

#include "register_extra_trampolines.h"

#include "camera2ndk_shared_capture_callbacks.h"

namespace berberis {
namespace {

// camera_status_t is a plain enum (int-backed); int is the ABI-equivalent
// return type and avoids pulling the bionic-only <camera/NdkCameraError.h>.
using SharedStartStreamingSig = int(ACameraCaptureSession*,
                                    ACameraCaptureSession_captureCallbacksV2*,
                                    int,
                                    ANativeWindow**,
                                    int*);

using SharedLogicalStartStreamingSig =
    int(ACameraCaptureSession*,
        ACameraCaptureSession_logicalCamera_captureCallbacksV2*,
        int,
        ANativeWindow**,
        int*);

void DoCustomTrampoline_ACameraCaptureSessionShared_startStreaming(HostCode callee,
                                                                   ProcessState* state) {
  auto [session, guest_cbs, num_windows, windows, capture_sequence_id] =
      GuestParamsValues<SharedStartStreamingSig>(state);
  ACameraCaptureSession_captureCallbacksV2 host_cbs = {};
  ACameraCaptureSession_captureCallbacksV2* host_cbs_ptr = nullptr;
  if (guest_cbs != nullptr) {
    MarshalCaptureCallbacksV2(guest_cbs, &host_cbs);
    host_cbs_ptr = &host_cbs;
  }
  auto&& [ret] = GuestReturnReference<SharedStartStreamingSig>(state);
  ret = reinterpret_cast<SharedStartStreamingSig*>(const_cast<void*>(callee))(
      session, host_cbs_ptr, num_windows, windows, capture_sequence_id);
}

void DoCustomTrampoline_ACameraCaptureSessionShared_logicalCamera_startStreaming(
    HostCode callee,
    ProcessState* state) {
  auto [session, guest_cbs, num_windows, windows, capture_sequence_id] =
      GuestParamsValues<SharedLogicalStartStreamingSig>(state);
  ACameraCaptureSession_logicalCamera_captureCallbacksV2 host_cbs = {};
  ACameraCaptureSession_logicalCamera_captureCallbacksV2* host_cbs_ptr = nullptr;
  if (guest_cbs != nullptr) {
    MarshalLogicalCameraCaptureCallbacksV2(guest_cbs, &host_cbs);
    host_cbs_ptr = &host_cbs;
  }
  auto&& [ret] = GuestReturnReference<SharedLogicalStartStreamingSig>(state);
  ret = reinterpret_cast<SharedLogicalStartStreamingSig*>(const_cast<void*>(callee))(
      session, host_cbs_ptr, num_windows, windows, capture_sequence_id);
}

const KnownTrampoline kDigitalisExtraLibcamera2ndkTrampolines[] = {
    {"ACameraCaptureSessionShared_startStreaming",
     DoCustomTrampoline_ACameraCaptureSessionShared_startStreaming, nullptr},
    {"ACameraCaptureSessionShared_logicalCamera_startStreaming",
     DoCustomTrampoline_ACameraCaptureSessionShared_logicalCamera_startStreaming, nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libcamera2ndk.so", kDigitalisExtraLibcamera2ndkTrampolines)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
