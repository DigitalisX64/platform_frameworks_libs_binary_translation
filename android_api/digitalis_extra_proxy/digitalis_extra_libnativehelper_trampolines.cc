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

// Digitalis-side extra trampolines for libnativehelper JNI-helper symbols the
// upstream proxy leaves as DoBadTrampoline. They take a JNIEnv* first argument,
// so a plain pointer pass-through is WRONG (the guest's JNIEnv has a guest-
// callable function table; the host would call guest function pointers). Each
// trampoline translates the guest JNIEnv to the host JNIEnv via ToHostJNIEnv and
// forwards the remaining (flat, layout-identical under LP64) arguments, mirroring
// native_bridge_support's nativehelper_trampolines.cc.
//
// The host function is reached via `callee` (the proxy dlsym's it from host
// libnativehelper at registration) rather than by name, because this static lib
// is whole_static_lib'd into libberberis_arm64.so, which does not link
// libnativehelper. If host libnativehelper does not export the symbol, callee is
// DoBadThunk and the call still aborts cleanly — verify reachability per symbol.
//
// jniThrowExceptionFmt is the lone variadic helper. It cannot be forwarded as a
// host varargs call, but its documented semantics are exactly
// `vsnprintf(msgBuf, 512, fmt, args); jniThrowException(env, className, msgBuf)`,
// so it is covered in-surface by walking the guest variadic tail per AAPCS64 with
// GuestVAListParams and formatting host-side (see its trampoline below). It does
// not use `callee` — the host inlines jniThrowExceptionFmt as a static header
// function, so no host symbol exists to dlsym.

#if defined(__x86_64__)

#include <jni.h>

#include "berberis/base/format_buffer.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/jni/jni_trampolines.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/proxy_loader/proxy_library_builder.h"

#include "register_extra_trampolines.h"

namespace berberis {
namespace {

// Host signatures (function types) for GuestParamsValues/GuestReturnReference
// and for casting `callee`. JNIEnv* is the translated first argument.
using Sig_ThrowException = int(JNIEnv*, const char*, const char*);
using Sig_ThrowMsg = int(JNIEnv*, const char*);
using Sig_ThrowIO = int(JNIEnv*, int);
using Sig_ThrowErrno = int(JNIEnv*, const char*, int);
using Sig_LogException = void(JNIEnv*, int, const char*, jthrowable);
using Sig_CreateString = jstring(JNIEnv*, const jchar*, jsize);
using Sig_NioFields = jlong(JNIEnv*, jobject, jint*, jint*, jint*);
using Sig_NioPointer = jlong(JNIEnv*, jobject);
using Sig_NioBaseArray = jarray(JNIEnv*, jobject);
using Sig_NioBaseArrayOffset = jint(JNIEnv*, jobject);
using Sig_RegisterNatives = jint(JNIEnv*, const char*, const JNINativeMethod*, jint);

// HostCode is `const void*`; a function pointer cannot carry the const, so drop
// it before reinterpreting. `callee` is the host libnativehelper function the
// proxy dlsym'd at registration.
template <typename Sig>
Sig* HostFn(HostCode callee) {
  return reinterpret_cast<Sig*>(const_cast<void*>(callee));
}

void DoCustomTrampoline_jniThrowException(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_class, arg_msg] = GuestParamsValues<Sig_ThrowException>(state);
  auto&& [ret] = GuestReturnReference<Sig_ThrowException>(state);
  ret = HostFn<Sig_ThrowException>(callee)(ToHostJNIEnv(guest_env), arg_class, arg_msg);
}

void DoCustomTrampoline_jniThrowNullPointerException(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_msg] = GuestParamsValues<Sig_ThrowMsg>(state);
  auto&& [ret] = GuestReturnReference<Sig_ThrowMsg>(state);
  ret = HostFn<Sig_ThrowMsg>(callee)(ToHostJNIEnv(guest_env), arg_msg);
}

void DoCustomTrampoline_jniThrowRuntimeException(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_msg] = GuestParamsValues<Sig_ThrowMsg>(state);
  auto&& [ret] = GuestReturnReference<Sig_ThrowMsg>(state);
  ret = HostFn<Sig_ThrowMsg>(callee)(ToHostJNIEnv(guest_env), arg_msg);
}

void DoCustomTrampoline_jniThrowIOException(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_errno] = GuestParamsValues<Sig_ThrowIO>(state);
  auto&& [ret] = GuestReturnReference<Sig_ThrowIO>(state);
  ret = HostFn<Sig_ThrowIO>(callee)(ToHostJNIEnv(guest_env), arg_errno);
}

void DoCustomTrampoline_jniThrowErrnoException(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_fn, arg_errno] = GuestParamsValues<Sig_ThrowErrno>(state);
  auto&& [ret] = GuestReturnReference<Sig_ThrowErrno>(state);
  ret = HostFn<Sig_ThrowErrno>(callee)(ToHostJNIEnv(guest_env), arg_fn, arg_errno);
}

void DoCustomTrampoline_jniLogException(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_prio, arg_tag, arg_throwable] = GuestParamsValues<Sig_LogException>(state);
  HostFn<Sig_LogException>(callee)(ToHostJNIEnv(guest_env), arg_prio, arg_tag, arg_throwable);
}

void DoCustomTrampoline_jniCreateString(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_chars, arg_len] = GuestParamsValues<Sig_CreateString>(state);
  auto&& [ret] = GuestReturnReference<Sig_CreateString>(state);
  ret = HostFn<Sig_CreateString>(callee)(ToHostJNIEnv(guest_env), arg_chars, arg_len);
}

// jniGetNioBuffer* take a java.nio.Buffer and write its position/limit/etc. The
// out-parameters and the jobject pass through verbatim (guest memory is host-
// addressable in the proxied call); only the JNIEnv* needs translation.
void DoCustomTrampoline_jniGetNioBufferFields(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_buf, arg_pos, arg_limit, arg_shift] =
      GuestParamsValues<Sig_NioFields>(state);
  auto&& [ret] = GuestReturnReference<Sig_NioFields>(state);
  ret = HostFn<Sig_NioFields>(callee)(
      ToHostJNIEnv(guest_env), arg_buf, arg_pos, arg_limit, arg_shift);
}

void DoCustomTrampoline_jniGetNioBufferPointer(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_buf] = GuestParamsValues<Sig_NioPointer>(state);
  auto&& [ret] = GuestReturnReference<Sig_NioPointer>(state);
  ret = HostFn<Sig_NioPointer>(callee)(ToHostJNIEnv(guest_env), arg_buf);
}

void DoCustomTrampoline_jniGetNioBufferBaseArray(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_buf] = GuestParamsValues<Sig_NioBaseArray>(state);
  auto&& [ret] = GuestReturnReference<Sig_NioBaseArray>(state);
  ret = HostFn<Sig_NioBaseArray>(callee)(ToHostJNIEnv(guest_env), arg_buf);
}

void DoCustomTrampoline_jniGetNioBufferBaseArrayOffset(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_buf] = GuestParamsValues<Sig_NioBaseArrayOffset>(state);
  auto&& [ret] = GuestReturnReference<Sig_NioBaseArrayOffset>(state);
  ret = HostFn<Sig_NioBaseArrayOffset>(callee)(ToHostJNIEnv(guest_env), arg_buf);
}

// jniRegisterNativeMethods does FindClass(className) then env->RegisterNatives.
// className passes through verbatim; the JNINativeMethod array passes through
// verbatim too (identical 3-pointer LP64 layout) — the guest function pointers
// it carries are NOT wrapped here, exactly as the native bridge's own
// JNIEnv::RegisterNatives trampoline leaves them: the ART native-bridge layer
// recognizes guest native fnPtrs and routes their later invocation through the
// translator. Only the JNIEnv* needs translation.
void DoCustomTrampoline_jniRegisterNativeMethods(HostCode callee, ProcessState* state) {
  auto [guest_env, arg_class, arg_methods, arg_num] = GuestParamsValues<Sig_RegisterNatives>(state);
  auto&& [ret] = GuestReturnReference<Sig_RegisterNatives>(state);
  ret = HostFn<Sig_RegisterNatives>(callee)(
      ToHostJNIEnv(guest_env), arg_class, arg_methods, arg_num);
}

// jniThrowExceptionFmt(JNIEnv*, const char* className, const char* fmt, ...) is the
// one true-varargs JNIHelp helper. A host varargs call cannot be reconstructed from
// a dynamic argument list portably, but its documented behavior is precisely:
//     char msgBuf[512]; vsnprintf(msgBuf, sizeof(msgBuf), fmt, args);
//     return jniThrowException(env, className, msgBuf);
// We reproduce it entirely in-surface:
//   1. Read the three named params (JNIEnv* translated, two const char* pass-through).
//   2. Walk the guest variadic tail per AAPCS64 with GuestVAListParams — the same
//      path Berberis uses for guest-side tracing (guest_loader.cc's TraceCallback).
//   3. Format host-side with FormatBufferImpl into a fixed 512-byte buffer (matching
//      the header's msgBuf[512]).
//   4. Throw on the host VM via the translated JNIEnv (FindClass + ThrowNew), which
//      is what jniThrowException does — so no host libnativehelper symbol is needed.
//
// Specifier coverage equals FormatBufferImpl: %s %d %u %x %c %p, the l / ll integer
// length forms, and %z size_t. Floating-point specifiers (%f/%g/%e) are NOT handled
// by FormatBufferImpl (the shared guest-trace formatter) and are not emitted by
// libnativehelper's exception callers; a %f stops formatting at that point with the
// preceding text preserved (callers use integer/string formats only).
using Sig_ThrowExceptionFmt = int(JNIEnv*, const char*, const char*, ...);

// Bridges GuestVAListParams to FormatBufferImpl's argument-extraction contract.
// Mirrors guest_loader.cc's file-local FormatBufferGuestParamsArgs.
class GuestFmtArgs {
 public:
  explicit GuestFmtArgs(GuestVAListParams&& params) : params_(params) {}

  const char* GetCStr() { return params_.GetPointerParam<const char>(); }
  uintmax_t GetPtrAsUInt() { return params_.GetParam<GuestAddr>(); }
  intmax_t GetInt() { return params_.GetParam<int>(); }
  intmax_t GetLong() { return params_.GetParam<long>(); }
  intmax_t GetLongLong() { return params_.GetParam<long long>(); }
  uintmax_t GetUInt() { return params_.GetParam<unsigned int>(); }
  uintmax_t GetULong() { return params_.GetParam<unsigned long>(); }
  uintmax_t GetULongLong() { return params_.GetParam<unsigned long long>(); }
  intmax_t GetChar() { return params_.GetParam<int>(); }
  uintmax_t GetSizeT() { return params_.GetParam<GuestAddr>(); }

 private:
  GuestVAListParams params_;
};

void DoCustomTrampoline_jniThrowExceptionFmt(HostCode /*callee*/, ProcessState* state) {
  auto [guest_env, arg_class, arg_fmt] = GuestParamsValues<Sig_ThrowExceptionFmt>(state);
  auto&& [ret] = GuestReturnReference<Sig_ThrowExceptionFmt>(state);

  char msg_buf[512];
  if (arg_fmt != nullptr) {
    GuestFmtArgs fmt_args{GuestParamsValues<Sig_ThrowExceptionFmt>(state)};
    CStrBuffer out(msg_buf, sizeof(msg_buf) - 1);
    FormatBufferImpl(&out, arg_fmt, &fmt_args);
    msg_buf[out.Size()] = '\0';
  } else {
    msg_buf[0] = '\0';
  }

  JNIEnv* env = ToHostJNIEnv(guest_env);
  if (env == nullptr) {
    ret = -1;
    return;
  }
  // Mirror jniThrowException: a pending exception is logged-and-cleared upstream;
  // here we just clear it so the new throw is the live one.
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }
  jclass exc_class = env->FindClass(arg_class);
  if (exc_class == nullptr) {
    // FindClass left a pending NoClassDefFoundError; clear it and report failure,
    // matching jniThrowException's nonzero return when the class can't be found.
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
    }
    ret = -1;
    return;
  }
  ret = (env->ThrowNew(exc_class, msg_buf) == JNI_OK) ? 0 : -1;
  env->DeleteLocalRef(exc_class);
}

const KnownTrampoline kDigitalisExtraLibnativehelperTrampolines[] = {
    {"jniThrowException", DoCustomTrampoline_jniThrowException, nullptr},
    {"jniThrowExceptionFmt", DoCustomTrampoline_jniThrowExceptionFmt, nullptr},
    {"jniThrowNullPointerException", DoCustomTrampoline_jniThrowNullPointerException, nullptr},
    {"jniThrowRuntimeException", DoCustomTrampoline_jniThrowRuntimeException, nullptr},
    {"jniThrowIOException", DoCustomTrampoline_jniThrowIOException, nullptr},
    {"jniThrowErrnoException", DoCustomTrampoline_jniThrowErrnoException, nullptr},
    {"jniLogException", DoCustomTrampoline_jniLogException, nullptr},
    {"jniCreateString", DoCustomTrampoline_jniCreateString, nullptr},
    {"jniGetNioBufferFields", DoCustomTrampoline_jniGetNioBufferFields, nullptr},
    {"jniGetNioBufferPointer", DoCustomTrampoline_jniGetNioBufferPointer, nullptr},
    {"jniGetNioBufferBaseArray", DoCustomTrampoline_jniGetNioBufferBaseArray, nullptr},
    {"jniGetNioBufferBaseArrayOffset", DoCustomTrampoline_jniGetNioBufferBaseArrayOffset, nullptr},
    {"jniRegisterNativeMethods", DoCustomTrampoline_jniRegisterNativeMethods, nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINES("libnativehelper.so",
                                     kDigitalisExtraLibnativehelperTrampolines)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
