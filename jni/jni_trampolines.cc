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

#include "berberis/jni/jni_trampolines.h"

#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include <jni.h>  // NOLINT [build/include_order]

#include "berberis/base/checks.h"
#include "berberis/base/gettid.h"
#include "berberis/base/logging.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_abi/guest_arguments.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_abi/guest_type.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/native_bridge/jmethod_shorty.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/runtime_library.h"

#include "guest_jni_trampolines.h"

// #define LOG_JNI(...) ALOGE(__VA_ARGS__)
#define LOG_JNI(...)

namespace berberis {

namespace {

char ConvertDalvikTypeCharToWrapperTypeChar(char c) {
  switch (c) {
    case 'V':  // void
      return 'v';
    case 'Z':  // boolean
      return 'z';
    case 'B':  // byte
      return 'b';
    case 'S':  // short
      return 's';
    case 'C':  // char
      return 'c';
    case 'I':  // int
      return 'i';
    case 'L':  // class object - pointer
      return 'p';
    case 'J':  // long
      return 'l';
    case 'F':  // float
      return 'f';
    case 'D':  // double
      return 'd';
    default:
      FATAL("Failed to convert Dalvik char '%c'", c);
  }
}

void ConvertDalvikShortyToWrapperSignature(char* dst,
                                           int size,
                                           const char* src,
                                           bool add_jnienv_and_jobject) {
  // return type, env and clazz.
  CHECK_GT(size, 3);
  char* cur = dst;
  *cur++ = ConvertDalvikTypeCharToWrapperTypeChar(*src++);

  if (add_jnienv_and_jobject) {
    *cur++ = 'p';
    *cur++ = 'p';
  }

  while (*src) {
    CHECK_LT(cur, dst + (size - 1));
    *cur++ = ConvertDalvikTypeCharToWrapperTypeChar(*src++);
  }

  *cur = '\0';
}

void RunGuestJNIFunction(GuestAddr pc, GuestArgumentBuffer* buf) {
  auto [host_jni_env] = HostArgumentsValues<void(JNIEnv*)>(buf);
  {
    auto&& [guest_jni_env] = GuestArgumentsReferences<void(JNIEnv*)>(buf);
    guest_jni_env = ToGuestJNIEnv(host_jni_env);
  }
  RunGuestCall(pc, buf);
}

void RunGuestJNIOnLoad(GuestAddr pc, GuestArgumentBuffer* buf) {
  auto [host_java_vm, reserved] = HostArgumentsValues<decltype(JNI_OnLoad)>(buf);
  {
    auto&& [guest_java_vm, reserved] = GuestArgumentsReferences<decltype(JNI_OnLoad)>(buf);
    guest_java_vm = ToGuestJavaVM(host_java_vm);
  }
  RunGuestCall(pc, buf);
}

}  // namespace

HostCode WrapGuestJNIFunction(GuestAddr pc,
                              const char* shorty,
                              const char* name,
                              bool has_jnienv_and_jobject) {
  const size_t size = strlen(shorty);
  char signature[size + /* env, clazz and trailing zero */ 3];
  ConvertDalvikShortyToWrapperSignature(
      signature, sizeof(signature), shorty, has_jnienv_and_jobject);
  auto guest_runner = has_jnienv_and_jobject ? RunGuestJNIFunction : RunGuestCall;
  return WrapGuestFunctionImpl(pc, signature, guest_runner, name);
}

HostCode WrapGuestJNIOnLoad(GuestAddr pc) {
  return WrapGuestFunctionImpl(pc, "ipp", RunGuestJNIOnLoad, "JNI_OnLoad");
}

namespace {

std::vector<jvalue> ConvertVAList(JNIEnv* env, jmethodID methodID, GuestVAListParams&& params) {
  std::vector<jvalue> result;
  const char* short_signature = GetJMethodShorty(env, methodID);
  CHECK(short_signature);
  short_signature++;  // skip return value
  int len = strlen(short_signature);
  result.resize(len);
  for (int i = 0; i < len; i++) {
    jvalue& arg = result[i];
    char c = short_signature[i];
    switch (c) {
      case 'Z':  // boolean (u8)
        arg.z = params.GetParam<uint8_t>();
        break;
      case 'B':  // byte (i8)
        arg.b = params.GetParam<int8_t>();
        break;
      case 'S':  // short (i16)
        arg.s = params.GetParam<int16_t>();
        break;
      case 'C':  // char (u16)
        arg.c = params.GetParam<uint16_t>();
        break;
      case 'I':  // int (i32)
        arg.i = params.GetParam<int32_t>();
        break;
      case 'J':  // long (i64)
        arg.j = params.GetParam<int64_t>();
        break;
      case 'F':  // float - passed as double
        arg.f = params.GetParam<double>();
        break;
      case 'D':  // double
        arg.d = params.GetParam<double>();
        break;
      case 'L':  // class object (pointer)
        arg.l = params.GetParam<jobject>();
        break;
      default:
        FATAL("Failed to convert Dalvik char '%c'", c);
        break;
    }
  }
  return result;
}

// jint RegisterNatives(
//     JNIEnv *env, jclass clazz,
//     const JNINativeMethod *methods, jint nMethods);
void DoTrampoline_JNIEnv_RegisterNatives(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JNIEnv>().functions->RegisterNatives);
  auto [guest_env, arg_clazz, arg_methods, arg_n] = GuestParamsValues<PFN_callee>(state);
  JNIEnv* arg_env = ToHostJNIEnv(guest_env);

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_env->functions)->RegisterNatives(arg_env, arg_clazz, arg_methods, arg_n);
}

// jint GetJavaVM(
//     JNIEnv *env, JavaVM **vm);
void DoTrampoline_JNIEnv_GetJavaVM(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JNIEnv>().functions->GetJavaVM);
  auto [guest_env, arg_vm] = GuestParamsValues<PFN_callee>(state);
  JNIEnv* arg_env = ToHostJNIEnv(guest_env);
  JavaVM* host_vm;

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_env->functions)->GetJavaVM(arg_env, &host_vm);
  if (ret == 0) {
    *bit_cast<GuestType<JavaVM*>*>(arg_vm) = ToGuestJavaVM(host_vm);
  }
}

void DoTrampoline_JNIEnv_CallStaticVoidMethodV(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JNIEnv>().functions->CallStaticVoidMethodV);
  auto [arg_env, arg_1, arg_2, arg_va] = GuestParamsValues<PFN_callee>(state);
  JNIEnv* arg_0 = ToHostJNIEnv(arg_env);
  std::vector<jvalue> arg_vector = ConvertVAList(arg_0, arg_2, ToGuestAddr(arg_va));
  jvalue* arg_3 = &arg_vector[0];

  // Note, this call is the only difference from the auto-generated trampoline.
  JNIEnv_CallStaticVoidMethodV_ForGuest(arg_0, arg_1, arg_2, arg_3);

  (arg_0->functions)->CallStaticVoidMethodA(arg_0, arg_1, arg_2, arg_3);
}

struct KnownMethodTrampoline {
  unsigned index;
  TrampolineFunc marshal_and_call;
};

#include "jni_trampolines-inl.h"  // NOLINT(build/include)

// According to our observations there is only one instance of JavaVM
// and there are 1 or sometimes more instances of JNIEnv per thread created
// by Java Runtime (JNIEnv instances are not shared between different threads).
//
// This is why we store one global mapping for JavaVM for the app.
// And multiple mappings of JNIEnv per thread. There is often only one JNIEnv
// per thread, but we have seen examples where 2 instances where created.
//
// It is likely that the new JNIEnv instance for the thread supersedes the
// previous one but the code below does not make this assumption.
struct JNIEnvMapping {
  std::deque<JNIEnv> guest_jni_envs;
  std::map<GuestType<JNIEnv*>, JNIEnv*> guest_to_host_jni_env;
  std::map<JNIEnv*, GuestType<JNIEnv*>> host_to_guest_jni_env;
};

std::mutex g_jni_guard_mutex;

JavaVM g_guest_java_vm;
JavaVM* g_host_java_vm;

// TODO(b/399909631): Add a callback from GuestThread::Destroy to remove entries
// from this map.
std::map<pid_t, JNIEnvMapping> g_jni_env_mappings;

void DoJavaVMTrampoline_DestroyJavaVM(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->DestroyJavaVM);
  auto [arg_vm] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->DestroyJavaVM(arg_java_vm);
}

// jint AttachCurrentThread(JavaVM*, JNIEnv**, void*);
void DoJavaVMTrampoline_AttachCurrentThread(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->AttachCurrentThread);
  auto [arg_vm, arg_env_ptr, arg_args] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);
  JNIEnv* env = nullptr;

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->AttachCurrentThread(arg_java_vm, &env, arg_args);

  GuestType<JNIEnv*> guest_jni_env = ToGuestJNIEnv(env);
  memcpy(arg_env_ptr, &guest_jni_env, sizeof(guest_jni_env));
}

// jint DetachCurrentThread(JavaVM*);
void DoJavaVMTrampoline_DetachCurrentThread(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->DetachCurrentThread);
  auto [arg_vm] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->DetachCurrentThread(arg_java_vm);
}

// jint GetEnv(JavaVM*, void**, jint);
void DoJavaVMTrampoline_GetEnv(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->GetEnv);
  auto [arg_vm, arg_env_ptr, arg_version] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);

  LOG_JNI("JavaVM::GetEnv(%p, %p, %d)", arg_java_vm, arg_env_ptr, arg_version);

  void* env = nullptr;
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->GetEnv(arg_java_vm, &env, arg_version);

  GuestType<JNIEnv*> guest_jni_env = ToGuestJNIEnv(static_cast<JNIEnv*>(env));
  memcpy(arg_env_ptr, &guest_jni_env, sizeof(guest_jni_env));

  LOG_JNI("= jint(%d)", ret);
}

// jint AttachCurrentThreadAsDaemon(JavaVM* vm, void** penv, void* args);
void DoJavaVMTrampoline_AttachCurrentThreadAsDaemon(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->AttachCurrentThreadAsDaemon);
  auto [arg_vm, arg_env_ptr, arg_args] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);

  JNIEnv* env = nullptr;
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->AttachCurrentThreadAsDaemon(arg_java_vm, &env, arg_args);

  GuestType<JNIEnv*> guest_jni_env = ToGuestJNIEnv(env);
  memcpy(arg_env_ptr, &guest_jni_env, sizeof(guest_jni_env));
}

void WrapJavaVM(void* java_vm) {
  HostCode* vtable = *reinterpret_cast<HostCode**>(java_vm);
  // vtable[0] is NULL
  // vtable[1] is NULL
  // vtable[2] is NULL

  WrapHostFunctionImpl(vtable[3], DoJavaVMTrampoline_DestroyJavaVM, "JavaVM::DestroyJavaVM");

  WrapHostFunctionImpl(
      vtable[4], DoJavaVMTrampoline_AttachCurrentThread, "JavaVM::AttachCurrentThread");

  WrapHostFunctionImpl(
      vtable[5], DoJavaVMTrampoline_DetachCurrentThread, "JavaVM::DetachCurrentThread");

  WrapHostFunctionImpl(vtable[6], DoJavaVMTrampoline_GetEnv, "JavaVM::GetEnv");

  WrapHostFunctionImpl(vtable[7],
                       DoJavaVMTrampoline_AttachCurrentThreadAsDaemon,
                       "JavaVM::AttachCurrentThreadAsDaemon");
}

// We set this to 1 when host JNIEnv/JavaVM functions are wrapped.
std::atomic<uint32_t> g_jni_env_wrapped = {0};
std::atomic<uint32_t> g_java_vm_wrapped = {0};

}  // namespace

GuestType<JNIEnv*> ToGuestJNIEnv(JNIEnv* host_jni_env) {
  if (!host_jni_env) {
    return 0;
  }
  // We need to wrap host JNI functions only once. We use an atomic variable
  // to guard this initialization. Since we use very simple logic without
  // waiting here, multiple threads can wrap host JNI functions simultaneously.
  // This is OK since wrapping is thread-safe and later wrappings override
  // previous ones atomically.
  // TODO(halyavin) Consider creating a general mechanism for thread-safe
  // initialization with parameters, if we need it in more than one place.
  if (std::atomic_load_explicit(&g_jni_env_wrapped, std::memory_order_acquire) == 0U) {
    WrapJNIEnv(host_jni_env);
    std::atomic_store_explicit(&g_jni_env_wrapped, 1U, std::memory_order_release);
  }

  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  pid_t thread_id = GettidSyscall();
  JNIEnvMapping& mapping = g_jni_env_mappings[thread_id];

  auto it = mapping.host_to_guest_jni_env.find(host_jni_env);
  if (it != mapping.host_to_guest_jni_env.end()) {
    return it->second;
  }

  mapping.guest_jni_envs.emplace_back(*host_jni_env);
  JNIEnv* guest_jni_env = &mapping.guest_jni_envs.back();
  auto [unused_it1, host_to_guest_inserted] =
      mapping.host_to_guest_jni_env.try_emplace(host_jni_env, guest_jni_env);
  CHECK(host_to_guest_inserted);

  auto [unused_it2, guest_to_host_inserted] =
      mapping.guest_to_host_jni_env.try_emplace(guest_jni_env, host_jni_env);
  CHECK(guest_to_host_inserted);

  return guest_jni_env;
}

JNIEnv* ToHostJNIEnv(GuestType<JNIEnv*> guest_jni_env) {
  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  pid_t thread_id = GettidSyscall();
  JNIEnvMapping& mapping = g_jni_env_mappings[thread_id];

  auto it = mapping.guest_to_host_jni_env.find(guest_jni_env);

  if (it == mapping.guest_to_host_jni_env.end()) {
    ALOGE("Unexpected guest JNIEnv: %p (it was never passed to guest), passing to host 'as is'",
          ToHostAddr(guest_jni_env));
    TRACE("Unexpected guest JNIEnv: %p (it was never passed to guest), passing to host 'as is'",
          ToHostAddr(guest_jni_env));
    return ToHostAddr(guest_jni_env);
  }

  return it->second;
}

GuestType<JavaVM*> ToGuestJavaVM(JavaVM* host_java_vm) {
  CHECK(host_java_vm);
  if (std::atomic_load_explicit(&g_java_vm_wrapped, std::memory_order_acquire) == 0U) {
    WrapJavaVM(host_java_vm);
    std::atomic_store_explicit(&g_java_vm_wrapped, 1U, std::memory_order_release);
  }

  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  if (g_host_java_vm == nullptr) {
    g_guest_java_vm = *host_java_vm;
    g_host_java_vm = host_java_vm;
  }

  if (g_host_java_vm != host_java_vm) {
    TRACE("Warning: Unexpected host JavaVM: %p (expecting %p), passing as is",
          host_java_vm,
          g_host_java_vm);
    return host_java_vm;
  }

  return &g_guest_java_vm;
}

JavaVM* ToHostJavaVM(GuestType<JavaVM*> guest_java_vm) {
  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  if (ToHostAddr(guest_java_vm) == &g_guest_java_vm) {
    return g_host_java_vm;
  }

  TRACE("Warning: Unexpected guest JavaVM: %p (expecting %p), passing as is",
        ToHostAddr(guest_java_vm),
        &g_guest_java_vm);

  return ToHostAddr(guest_java_vm);
}

}  // namespace berberis
