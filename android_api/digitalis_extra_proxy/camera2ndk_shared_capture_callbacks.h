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

#ifndef BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_CAMERA2NDK_SHARED_CAPTURE_CALLBACKS_H_
#define BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_CAMERA2NDK_SHARED_CAPTURE_CALLBACKS_H_

#include <cstddef>
#include <cstdint>

#include "berberis/base/bit_util.h"
#include "berberis/guest_abi/guest_function_wrapper.h"
#include "berberis/guest_abi/guest_type.h"

// Self-contained mirror of the libcamera2ndk shared-camera V2 capture-callback
// structs and their callback function-pointer typedefs, transcribed verbatim
// from frameworks/av/camera/ndk/include/camera/NdkCameraCaptureSession.h.
//
// It is deliberately *self-contained* (opaque forward-declarations + scalar
// types only) rather than #include <camera/NdkCameraCaptureSession.h>: the real
// NDK header relies on bionic-only availability macros (__INTRODUCED_IN, etc.)
// that do not compile in the host gtest, and these structs are a flat array of
// eight 8-byte pointers whose layout the static_asserts below pin against the
// header. The verbatim callback signatures (with their source line numbers) are
// the ground truth the host marshalling test checks each wrapped callback
// against — see camera2ndk_shared_capture_callbacks_test.cc.
//
// Header reference (NdkCameraCaptureSession.h):
//   captureCallback_startV2        :855
//   captureCallback_result         :217
//   captureCallback_failed         :235
//   captureCallback_sequenceEnd    :248
//   captureCallback_sequenceAbort  :260
//   captureCallback_bufferLost     :277
//   logicalCamera_captureCallback_result :704
//   logicalCamera_captureCallback_failed :741
//   ACameraCaptureSession_captureCallbacksV2               :863
//   ACameraCaptureSession_logicalCamera_captureCallbacksV2 :914

namespace berberis {

// Opaque NDK handle types — only ever handled by pointer; layout is irrelevant
// to the callback ABI (every pointer is 8 bytes under LP64). Mirror the NDK
// typedef-struct spelling so the function-pointer signatures read identically.
typedef struct ACameraCaptureSession ACameraCaptureSession;
typedef struct ACaptureRequest ACaptureRequest;
typedef struct ACameraMetadata ACameraMetadata;
typedef struct ACameraCaptureFailure ACameraCaptureFailure;
typedef struct ALogicalCameraCaptureFailure ALogicalCameraCaptureFailure;
typedef struct ANativeWindow ANativeWindow;

// Callback function-pointer typedefs (verbatim parameter lists from the header).
typedef void (*ACameraCaptureSession_captureCallback_startV2)(void* context,
                                                              ACameraCaptureSession* session,
                                                              const ACaptureRequest* request,
                                                              int64_t timestamp,
                                                              int64_t frameNumber);

typedef void (*ACameraCaptureSession_captureCallback_result)(void* context,
                                                             ACameraCaptureSession* session,
                                                             ACaptureRequest* request,
                                                             const ACameraMetadata* result);

typedef void (*ACameraCaptureSession_captureCallback_failed)(void* context,
                                                             ACameraCaptureSession* session,
                                                             ACaptureRequest* request,
                                                             ACameraCaptureFailure* failure);

typedef void (*ACameraCaptureSession_captureCallback_sequenceEnd)(void* context,
                                                                 ACameraCaptureSession* session,
                                                                 int sequenceId,
                                                                 int64_t frameNumber);

typedef void (*ACameraCaptureSession_captureCallback_sequenceAbort)(void* context,
                                                                   ACameraCaptureSession* session,
                                                                   int sequenceId);

typedef void (*ACameraCaptureSession_captureCallback_bufferLost)(void* context,
                                                                ACameraCaptureSession* session,
                                                                ACaptureRequest* request,
                                                                ANativeWindow* window,
                                                                int64_t frameNumber);

typedef void (*ACameraCaptureSession_logicalCamera_captureCallback_result)(
    void* context,
    ACameraCaptureSession* session,
    ACaptureRequest* request,
    const ACameraMetadata* result,
    size_t physicalResultCount,
    const char** physicalCameraIds,
    const ACameraMetadata** physicalResults);

typedef void (*ACameraCaptureSession_logicalCamera_captureCallback_failed)(
    void* context,
    ACameraCaptureSession* session,
    ACaptureRequest* request,
    ALogicalCameraCaptureFailure* failure);

typedef struct ACameraCaptureSession_captureCallbacksV2 {
  void* context;
  ACameraCaptureSession_captureCallback_startV2 onCaptureStarted;
  ACameraCaptureSession_captureCallback_result onCaptureProgressed;
  ACameraCaptureSession_captureCallback_result onCaptureCompleted;
  ACameraCaptureSession_captureCallback_failed onCaptureFailed;
  ACameraCaptureSession_captureCallback_sequenceEnd onCaptureSequenceCompleted;
  ACameraCaptureSession_captureCallback_sequenceAbort onCaptureSequenceAborted;
  ACameraCaptureSession_captureCallback_bufferLost onCaptureBufferLost;
} ACameraCaptureSession_captureCallbacksV2;

typedef struct ACameraCaptureSession_logicalCamera_captureCallbacksV2 {
  void* context;
  ACameraCaptureSession_captureCallback_startV2 onCaptureStarted;
  ACameraCaptureSession_captureCallback_result onCaptureProgressed;
  ACameraCaptureSession_logicalCamera_captureCallback_result onLogicalCameraCaptureCompleted;
  ACameraCaptureSession_logicalCamera_captureCallback_failed onLogicalCameraCaptureFailed;
  ACameraCaptureSession_captureCallback_sequenceEnd onCaptureSequenceCompleted;
  ACameraCaptureSession_captureCallback_sequenceAbort onCaptureSequenceAborted;
  ACameraCaptureSession_captureCallback_bufferLost onCaptureBufferLost;
} ACameraCaptureSession_logicalCamera_captureCallbacksV2;

// Layout pins: a flat array of one context pointer + seven callback pointers.
// A reordering / dropped / extra field fails to compile here.
static_assert(sizeof(ACameraCaptureSession_captureCallbacksV2) == 8 * sizeof(void*));
static_assert(offsetof(ACameraCaptureSession_captureCallbacksV2, context) == 0);
static_assert(offsetof(ACameraCaptureSession_captureCallbacksV2, onCaptureStarted) == 8);
static_assert(offsetof(ACameraCaptureSession_captureCallbacksV2, onCaptureProgressed) == 16);
static_assert(offsetof(ACameraCaptureSession_captureCallbacksV2, onCaptureCompleted) == 24);
static_assert(offsetof(ACameraCaptureSession_captureCallbacksV2, onCaptureFailed) == 32);
static_assert(offsetof(ACameraCaptureSession_captureCallbacksV2, onCaptureSequenceCompleted) == 40);
static_assert(offsetof(ACameraCaptureSession_captureCallbacksV2, onCaptureSequenceAborted) == 48);
static_assert(offsetof(ACameraCaptureSession_captureCallbacksV2, onCaptureBufferLost) == 56);

static_assert(sizeof(ACameraCaptureSession_logicalCamera_captureCallbacksV2) == 8 * sizeof(void*));
static_assert(offsetof(ACameraCaptureSession_logicalCamera_captureCallbacksV2, context) == 0);
static_assert(
    offsetof(ACameraCaptureSession_logicalCamera_captureCallbacksV2, onCaptureStarted) == 8);
static_assert(
    offsetof(ACameraCaptureSession_logicalCamera_captureCallbacksV2, onCaptureProgressed) == 16);
static_assert(offsetof(ACameraCaptureSession_logicalCamera_captureCallbacksV2,
                       onLogicalCameraCaptureCompleted) == 24);
static_assert(offsetof(ACameraCaptureSession_logicalCamera_captureCallbacksV2,
                       onLogicalCameraCaptureFailed) == 32);
static_assert(offsetof(ACameraCaptureSession_logicalCamera_captureCallbacksV2,
                       onCaptureSequenceCompleted) == 40);
static_assert(offsetof(ACameraCaptureSession_logicalCamera_captureCallbacksV2,
                       onCaptureSequenceAborted) == 48);
static_assert(
    offsetof(ACameraCaptureSession_logicalCamera_captureCallbacksV2, onCaptureBufferLost) == 56);

// Wraps a guest callback function pointer (a guest code address stored in a
// host-typed field) into a host-callable thunk that routes back through
// RunGuestCall. WrapGuestFunction returns nullptr for a null guest address, so
// optional callbacks need no special-casing.
template <typename FnPtr>
inline FnPtr WrapCallback(FnPtr guest_fn, const char* name) {
  return WrapGuestFunction(bit_cast<GuestType<FnPtr>>(guest_fn), name);
}

inline void MarshalCaptureCallbacksV2(const ACameraCaptureSession_captureCallbacksV2* guest,
                                      ACameraCaptureSession_captureCallbacksV2* host) {
  host->context = guest->context;
  host->onCaptureStarted = WrapCallback(
      guest->onCaptureStarted, "ACameraCaptureSession_captureCallbacksV2::onCaptureStarted");
  host->onCaptureProgressed = WrapCallback(
      guest->onCaptureProgressed, "ACameraCaptureSession_captureCallbacksV2::onCaptureProgressed");
  host->onCaptureCompleted = WrapCallback(
      guest->onCaptureCompleted, "ACameraCaptureSession_captureCallbacksV2::onCaptureCompleted");
  host->onCaptureFailed = WrapCallback(
      guest->onCaptureFailed, "ACameraCaptureSession_captureCallbacksV2::onCaptureFailed");
  host->onCaptureSequenceCompleted =
      WrapCallback(guest->onCaptureSequenceCompleted,
                   "ACameraCaptureSession_captureCallbacksV2::onCaptureSequenceCompleted");
  host->onCaptureSequenceAborted =
      WrapCallback(guest->onCaptureSequenceAborted,
                   "ACameraCaptureSession_captureCallbacksV2::onCaptureSequenceAborted");
  host->onCaptureBufferLost = WrapCallback(
      guest->onCaptureBufferLost, "ACameraCaptureSession_captureCallbacksV2::onCaptureBufferLost");
}

inline void MarshalLogicalCameraCaptureCallbacksV2(
    const ACameraCaptureSession_logicalCamera_captureCallbacksV2* guest,
    ACameraCaptureSession_logicalCamera_captureCallbacksV2* host) {
  host->context = guest->context;
  host->onCaptureStarted =
      WrapCallback(guest->onCaptureStarted,
                   "ACameraCaptureSession_logicalCamera_captureCallbacksV2::onCaptureStarted");
  host->onCaptureProgressed =
      WrapCallback(guest->onCaptureProgressed,
                   "ACameraCaptureSession_logicalCamera_captureCallbacksV2::onCaptureProgressed");
  host->onLogicalCameraCaptureCompleted = WrapCallback(
      guest->onLogicalCameraCaptureCompleted,
      "ACameraCaptureSession_logicalCamera_captureCallbacksV2::onLogicalCameraCaptureCompleted");
  host->onLogicalCameraCaptureFailed = WrapCallback(
      guest->onLogicalCameraCaptureFailed,
      "ACameraCaptureSession_logicalCamera_captureCallbacksV2::onLogicalCameraCaptureFailed");
  host->onCaptureSequenceCompleted = WrapCallback(
      guest->onCaptureSequenceCompleted,
      "ACameraCaptureSession_logicalCamera_captureCallbacksV2::onCaptureSequenceCompleted");
  host->onCaptureSequenceAborted = WrapCallback(
      guest->onCaptureSequenceAborted,
      "ACameraCaptureSession_logicalCamera_captureCallbacksV2::onCaptureSequenceAborted");
  host->onCaptureBufferLost = WrapCallback(
      guest->onCaptureBufferLost,
      "ACameraCaptureSession_logicalCamera_captureCallbacksV2::onCaptureBufferLost");
}

}  // namespace berberis

#endif  // BERBERIS_ANDROID_API_DIGITALIS_EXTRA_PROXY_CAMERA2NDK_SHARED_CAPTURE_CALLBACKS_H_
