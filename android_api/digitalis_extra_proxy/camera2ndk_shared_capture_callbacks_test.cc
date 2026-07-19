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

// Verifies the per-field guest->host marshalling of the shared-camera V2
// capture-callback structs that back the libcamera2ndk
// ACameraCaptureSessionShared_startStreaming /
// ACameraCaptureSessionShared_logicalCamera_startStreaming trampolines.
//
// The shared-camera API itself cannot be exercised end-to-end on the Digitalis
// emulator (SYSTEM_CAMERA permission + no HAL shared-session support — see
// a shared-session AIDL HAL), so this is the ground-truth check
// for the only risky part of the trampolines: that each of the callback
// function pointers is wrapped with the exact signature declared in
// <camera/NdkCameraCaptureSession.h> and therefore receives the host->guest
// arguments in the right registers, with the right widths, in the right order.
//
// Strategy: a single hand-assembled ARM64 "recorder" guest function stores
// x1..x6 (the up-to-six arguments that follow the void* context, which is
// arg0/x0) into a buffer it receives in x0. We point every callback field at
// it, marshal the struct, then fire each wrapped host thunk with distinct
// sentinel values and confirm the guest saw them. A wrong field type/order
// would land a value in the wrong slot (or truncate it) and fail here.

#include "gtest/gtest.h"

#include <sys/mman.h>

#include <cstdint>

#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/kernel_api/sys_mman_emulation.h"
#include "berberis/test_utils/guest_exec_region.h"
#include "berberis/test_utils/translation_test.h"

#include "camera2ndk_shared_capture_callbacks.h"

namespace berberis {
namespace {

// Sentinels, distinct per argument slot so a misrouted value is obvious.
constexpr uint64_t kSession = 0x1111'1111'1111'1111ULL;
constexpr uint64_t kRequest = 0x2222'2222'2222'2222ULL;
constexpr uint64_t kResult = 0x3333'3333'3333'3333ULL;
constexpr uint64_t kFailure = 0x4444'4444'4444'4444ULL;
constexpr uint64_t kWindow = 0x5555'5555'5555'5555ULL;
constexpr uint64_t kPhysIds = 0x6666'6666'6666'6666ULL;
constexpr uint64_t kPhysResults = 0x7777'7777'7777'7777ULL;
constexpr int64_t kTimestamp = 0x1234'5678'9ABC'DEF0LL;
constexpr int64_t kFrameNumber = 0x0FED'CBA9'8765'4321LL;
constexpr int kSequenceId = 0x6789'ABCD;
constexpr size_t kPhysCount = 0x00AB'CDEF'0000'0042ULL;

template <typename T>
T As(uint64_t v) {
  return reinterpret_cast<T>(v);
}

class Arm64Camera2NdkCaptureCallbacksV2Test : public TranslationTest {
 protected:
  void SetUp() override {
    TranslationTest::SetUp();  // InitBerberis()
    // Recorder: str x1..x6 into [x0 + 8*k]; ret. Encodings confirmed with
    // llvm-mc/objdump (aarch64).
    recorder_ = MakeGuestExecRegion<uint32_t>({
        0xf900'0001,  // str x1, [x0]
        0xf900'0402,  // str x2, [x0, #8]
        0xf900'0803,  // str x3, [x0, #16]
        0xf900'0c04,  // str x4, [x0, #24]
        0xf900'1005,  // str x5, [x0, #32]
        0xf900'1406,  // str x6, [x0, #40]
        0xd65f'03c0,  // ret
    });
    record_ = static_cast<uint64_t*>(MmapForGuest(nullptr,
                                                  sizeof(uint64_t) * 8,
                                                  PROT_READ | PROT_WRITE,
                                                  MAP_PRIVATE | MAP_ANONYMOUS,
                                                  -1,
                                                  0));
  }

  // record_[k] holds the value the guest saw in x(k+1): Arg(0) == x1, etc.
  uint64_t Arg(int k) const { return record_[k]; }
  void ResetRecord() {
    for (int i = 0; i < 8; ++i) {
      record_[i] = 0;
    }
  }

  template <typename FnPtr>
  FnPtr Recorder() const {
    return reinterpret_cast<FnPtr>(recorder_);
  }

  ACameraCaptureSession_captureCallbacksV2 MarshalSimple() {
    ACameraCaptureSession_captureCallbacksV2 guest = {};
    guest.context = record_;
    guest.onCaptureStarted = Recorder<ACameraCaptureSession_captureCallback_startV2>();
    guest.onCaptureProgressed = Recorder<ACameraCaptureSession_captureCallback_result>();
    guest.onCaptureCompleted = Recorder<ACameraCaptureSession_captureCallback_result>();
    guest.onCaptureFailed = Recorder<ACameraCaptureSession_captureCallback_failed>();
    guest.onCaptureSequenceCompleted = Recorder<ACameraCaptureSession_captureCallback_sequenceEnd>();
    guest.onCaptureSequenceAborted = Recorder<ACameraCaptureSession_captureCallback_sequenceAbort>();
    guest.onCaptureBufferLost = Recorder<ACameraCaptureSession_captureCallback_bufferLost>();
    ACameraCaptureSession_captureCallbacksV2 host = {};
    MarshalCaptureCallbacksV2(&guest, &host);
    return host;
  }

  ACameraCaptureSession_logicalCamera_captureCallbacksV2 MarshalLogical() {
    ACameraCaptureSession_logicalCamera_captureCallbacksV2 guest = {};
    guest.context = record_;
    guest.onCaptureStarted = Recorder<ACameraCaptureSession_captureCallback_startV2>();
    guest.onCaptureProgressed = Recorder<ACameraCaptureSession_captureCallback_result>();
    guest.onLogicalCameraCaptureCompleted =
        Recorder<ACameraCaptureSession_logicalCamera_captureCallback_result>();
    guest.onLogicalCameraCaptureFailed =
        Recorder<ACameraCaptureSession_logicalCamera_captureCallback_failed>();
    guest.onCaptureSequenceCompleted = Recorder<ACameraCaptureSession_captureCallback_sequenceEnd>();
    guest.onCaptureSequenceAborted = Recorder<ACameraCaptureSession_captureCallback_sequenceAbort>();
    guest.onCaptureBufferLost = Recorder<ACameraCaptureSession_captureCallback_bufferLost>();
    ACameraCaptureSession_logicalCamera_captureCallbacksV2 host = {};
    MarshalLogicalCameraCaptureCallbacksV2(&guest, &host);
    return host;
  }

  GuestAddr recorder_ = 0;
  uint64_t* record_ = nullptr;
};

TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, ContextIsPassedThroughVerbatim) {
  auto host = MarshalSimple();
  EXPECT_EQ(record_, host.context);
}

// onCaptureStarted: void(void* ctx, ACameraCaptureSession*, const ACaptureRequest*,
//                        int64_t timestamp, int64_t frameNumber)  [startV2]
TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, OnCaptureStartedDeliversFiveArgs) {
  auto host = MarshalSimple();
  ASSERT_NE(nullptr, host.onCaptureStarted);
  ResetRecord();
  host.onCaptureStarted(host.context,
                        As<ACameraCaptureSession*>(kSession),
                        As<const ACaptureRequest*>(kRequest),
                        kTimestamp,
                        kFrameNumber);
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(kRequest, Arg(1));
  EXPECT_EQ(static_cast<uint64_t>(kTimestamp), Arg(2));
  EXPECT_EQ(static_cast<uint64_t>(kFrameNumber), Arg(3));
}

// onCaptureProgressed: void(void* ctx, ACameraCaptureSession*, ACaptureRequest*,
//                          const ACameraMetadata* result)  [result]
TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, OnCaptureProgressedDeliversResult) {
  auto host = MarshalSimple();
  ASSERT_NE(nullptr, host.onCaptureProgressed);
  ResetRecord();
  host.onCaptureProgressed(host.context,
                           As<ACameraCaptureSession*>(kSession),
                           As<ACaptureRequest*>(kRequest),
                           As<const ACameraMetadata*>(kResult));
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(kRequest, Arg(1));
  EXPECT_EQ(kResult, Arg(2));
}

TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, OnCaptureCompletedDeliversResult) {
  auto host = MarshalSimple();
  ASSERT_NE(nullptr, host.onCaptureCompleted);
  ResetRecord();
  host.onCaptureCompleted(host.context,
                          As<ACameraCaptureSession*>(kSession),
                          As<ACaptureRequest*>(kRequest),
                          As<const ACameraMetadata*>(kResult));
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(kRequest, Arg(1));
  EXPECT_EQ(kResult, Arg(2));
}

// onCaptureFailed: void(void* ctx, ACameraCaptureSession*, ACaptureRequest*,
//                       ACameraCaptureFailure*)  [failed]
TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, OnCaptureFailedDeliversFailure) {
  auto host = MarshalSimple();
  ASSERT_NE(nullptr, host.onCaptureFailed);
  ResetRecord();
  host.onCaptureFailed(host.context,
                       As<ACameraCaptureSession*>(kSession),
                       As<ACaptureRequest*>(kRequest),
                       As<ACameraCaptureFailure*>(kFailure));
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(kRequest, Arg(1));
  EXPECT_EQ(kFailure, Arg(2));
}

// onCaptureSequenceCompleted: void(void* ctx, ACameraCaptureSession*,
//                                  int sequenceId, int64_t frameNumber)  [sequenceEnd]
TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, OnCaptureSequenceCompletedDeliversIdAndFrame) {
  auto host = MarshalSimple();
  ASSERT_NE(nullptr, host.onCaptureSequenceCompleted);
  ResetRecord();
  host.onCaptureSequenceCompleted(host.context,
                                  As<ACameraCaptureSession*>(kSession),
                                  kSequenceId,
                                  kFrameNumber);
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(static_cast<uint32_t>(kSequenceId), Arg(1) & 0xFFFF'FFFFu);
  EXPECT_EQ(static_cast<uint64_t>(kFrameNumber), Arg(2));
}

// onCaptureSequenceAborted: void(void* ctx, ACameraCaptureSession*, int sequenceId)
//                           [sequenceAbort]
TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, OnCaptureSequenceAbortedDeliversId) {
  auto host = MarshalSimple();
  ASSERT_NE(nullptr, host.onCaptureSequenceAborted);
  ResetRecord();
  host.onCaptureSequenceAborted(host.context,
                                As<ACameraCaptureSession*>(kSession),
                                kSequenceId);
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(static_cast<uint32_t>(kSequenceId), Arg(1) & 0xFFFF'FFFFu);
}

// onCaptureBufferLost: void(void* ctx, ACameraCaptureSession*, ACaptureRequest*,
//                          ANativeWindow*, int64_t frameNumber)  [bufferLost]
TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, OnCaptureBufferLostDeliversWindowAndFrame) {
  auto host = MarshalSimple();
  ASSERT_NE(nullptr, host.onCaptureBufferLost);
  ResetRecord();
  host.onCaptureBufferLost(host.context,
                           As<ACameraCaptureSession*>(kSession),
                           As<ACaptureRequest*>(kRequest),
                           As<ANativeWindow*>(kWindow),
                           kFrameNumber);
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(kRequest, Arg(1));
  EXPECT_EQ(kWindow, Arg(2));
  EXPECT_EQ(static_cast<uint64_t>(kFrameNumber), Arg(3));
}

TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, NullGuestCallbacksStayNull) {
  ACameraCaptureSession_captureCallbacksV2 guest = {};
  guest.context = record_;  // every callback field left null
  ACameraCaptureSession_captureCallbacksV2 host = {};
  MarshalCaptureCallbacksV2(&guest, &host);
  EXPECT_EQ(record_, host.context);
  EXPECT_EQ(nullptr, host.onCaptureStarted);
  EXPECT_EQ(nullptr, host.onCaptureProgressed);
  EXPECT_EQ(nullptr, host.onCaptureCompleted);
  EXPECT_EQ(nullptr, host.onCaptureFailed);
  EXPECT_EQ(nullptr, host.onCaptureSequenceCompleted);
  EXPECT_EQ(nullptr, host.onCaptureSequenceAborted);
  EXPECT_EQ(nullptr, host.onCaptureBufferLost);
}

// Logical-multi-camera variant: the unique field is onLogicalCameraCaptureCompleted,
// the widest callback (6 args after context: session, request, result,
// size_t physicalResultCount, const char** physicalCameraIds,
// const ACameraMetadata** physicalResults).
TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, LogicalOnCaptureCompletedDeliversSevenArgs) {
  auto host = MarshalLogical();
  ASSERT_NE(nullptr, host.onLogicalCameraCaptureCompleted);
  ResetRecord();
  host.onLogicalCameraCaptureCompleted(host.context,
                                       As<ACameraCaptureSession*>(kSession),
                                       As<ACaptureRequest*>(kRequest),
                                       As<const ACameraMetadata*>(kResult),
                                       kPhysCount,
                                       As<const char**>(kPhysIds),
                                       As<const ACameraMetadata**>(kPhysResults));
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(kRequest, Arg(1));
  EXPECT_EQ(kResult, Arg(2));
  EXPECT_EQ(static_cast<uint64_t>(kPhysCount), Arg(3));
  EXPECT_EQ(kPhysIds, Arg(4));
  EXPECT_EQ(kPhysResults, Arg(5));
}

// onLogicalCameraCaptureFailed: void(void* ctx, ACameraCaptureSession*,
//     ACaptureRequest*, ALogicalCameraCaptureFailure*)
TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, LogicalOnCaptureFailedDeliversFailure) {
  auto host = MarshalLogical();
  ASSERT_NE(nullptr, host.onLogicalCameraCaptureFailed);
  ResetRecord();
  host.onLogicalCameraCaptureFailed(host.context,
                                    As<ACameraCaptureSession*>(kSession),
                                    As<ACaptureRequest*>(kRequest),
                                    As<ALogicalCameraCaptureFailure*>(kFailure));
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(kRequest, Arg(1));
  EXPECT_EQ(kFailure, Arg(2));
}

TEST_F(Arm64Camera2NdkCaptureCallbacksV2Test, LogicalSharedFieldsAlsoMarshal) {
  auto host = MarshalLogical();
  ASSERT_NE(nullptr, host.onCaptureStarted);
  ASSERT_NE(nullptr, host.onCaptureBufferLost);
  ResetRecord();
  host.onCaptureBufferLost(host.context,
                           As<ACameraCaptureSession*>(kSession),
                           As<ACaptureRequest*>(kRequest),
                           As<ANativeWindow*>(kWindow),
                           kFrameNumber);
  EXPECT_EQ(kSession, Arg(0));
  EXPECT_EQ(kRequest, Arg(1));
  EXPECT_EQ(kWindow, Arg(2));
  EXPECT_EQ(static_cast<uint64_t>(kFrameNumber), Arg(3));
}

}  // namespace
}  // namespace berberis
