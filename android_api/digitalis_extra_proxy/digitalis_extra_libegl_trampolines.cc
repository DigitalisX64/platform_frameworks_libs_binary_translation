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

// Digitalis-side override of libEGL's eglGetProcAddress that adds guest-callable
// wrapping for the ANGLE/CHROMIUM extension entry points the upstream proxy's
// wrap table does not know.
//
// Why this exists: ANGLE's host eglGetProcAddress returns a non-NULL dispatch
// stub for every proc it knows, and the upstream proxy trampoline
// (native_bridge_support .../egl_trampolines.cc) NULLs the guest return when its
// generated kOpenGLTrampolines table can't marshal that proc. That is fine for
// procs whose extension the driver does not advertise — but for advertised
// extensions it is a landmine: Chromium's GL bindings gate calls on the
// GL_EXTENSIONS string, not on the probed pointer being non-NULL. With ANGLE as
// the host GLES driver, GL_ANGLE_robust_client_memory IS advertised, Chromium
// enables the robust paths, and the very first state query calls the NULL that
// our proxy returned for glGetIntegervRobustANGLE — the guest jumps to address
// 0, berberis_HandleNoExec raises SIGSEGV(SEGV_ACCERR, addr=0), and the
// Chromium GPU process dies in a loop until the browser aborts with "Timed out
// waiting for GPU channel" (observed as the Helium crash on any heavy page).
//
// The override chains to the upstream trampoline first (keeping its several
// hundred core-GL wrappers), and only when upstream returned NULL for a
// host-present proc does it consult the table below. The table is demand-driven:
// it covers exactly the procs a Chromium 149 GPU process probes and that
// ANGLE-on-Android can advertise (GL_ANGLE_robust_client_memory,
// GL_ANGLE_get_tex_level_parameter, GL_ANGLE_multi_draw, GL_ANGLE_polygon_mode,
// GL_ANGLE_request_extension, GL_ANGLE_shader_pixel_local_storage,
// GL_CHROMIUM_copy_texture, GL_CHROMIUM_bind_uniform_location,
// GL_ANGLE_memory_object_flags, GL_ANGLE_vulkan_image, GL_ANGLE_blob_cache,
// EGL_KHR_debug, EGL_CHROMIUM_sync_control, and friends). Probed procs of
// extensions ANGLE never advertises on Android (D3D stream textures, Metal
// shared events, macOS GPU-power switching, external contexts) are deliberately
// NOT covered: the upstream NULL stays correct and unreachable for them.
//
// Pointer parameters marshal as void* — valid under LP64 since guest and host
// share the address space and the pointee layout is identical; only the
// register class matters to the trampoline generator. Integer widths and float
// parameters follow the ANGLE prototypes exactly.

#if defined(__x86_64__)

#include <cstdint>
#include <cstring>

#include "berberis/base/tracing.h"
#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_abi/guest_function_wrapper.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/proxy_loader/proxy_library_builder.h"
#include "berberis/runtime_primitives/host_code.h"

#include "register_extra_trampolines.h"

namespace berberis {
namespace {

// Self-contained GL/EGL scalar typedefs (no EGL/GLES headers needed).
using GLboolean_t = uint8_t;
using GLenum_t = uint32_t;
using GLuint_t = uint32_t;
using GLint_t = int32_t;
using GLsizei_t = int32_t;
using GLfloat_t = float;
using GLuint64_t = uint64_t;
using GLbitfield_t = uint32_t;
using GLsizeiptr_t = intptr_t;
using EGLBoolean_t = uint32_t;
using EGLint_t = int32_t;

using EglGenericFn = void (*)();
using PFN_eglGetProcAddress = EglGenericFn (*)(const char*);

// Registers `fn` as guest-callable with the auto-generated marshaller for the
// exact host signature FuncPtr.
template <typename FuncPtr>
void WrapAs(HostCode fn, const char* name) {
  WrapHostFunctionImpl(fn, GetTrampolineFunc<FuncPtr>(), name);
}

// --- Custom marshallers for the two procs that take guest callbacks. ---

// EGLint eglDebugMessageControlKHR(EGLDEBUGPROCKHR callback,
//                                  const EGLAttrib* attrib_list);
// EGLDEBUGPROCKHR: void (*)(EGLenum error, const char* command,
//                           EGLint messageType, EGLLabelKHR threadLabel,
//                           EGLLabelKHR objectLabel, const char* message)
void DoTrampoline_eglDebugMessageControlKHR(HostCode callee, ThreadState* state) {
  using DebugProc =
      void (*)(GLenum_t, const char*, EGLint_t, void*, void*, const char*);
  using PFN_callee = EGLint_t (*)(DebugProc, const void*);
  PFN_callee callee_function = AsFuncPtr(callee);

  auto [guest_callback, attrib_list] = GuestParamsValues<PFN_callee>(state);
  DebugProc host_callback =
      ToGuestAddr(guest_callback) == 0
          ? nullptr
          : WrapGuestFunction(guest_callback, "EGLDEBUGPROCKHR");

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = callee_function(host_callback, attrib_list);
}

// void glBlobCacheCallbacksANGLE(GLSETBLOBPROCANGLE set, GLGETBLOBPROCANGLE get,
//                                const void* userParam);
// set: void (*)(const void* key, GLsizeiptr keySize,
//               const void* value, GLsizeiptr valueSize, const void* userParam)
// get: GLsizeiptr (*)(const void* key, GLsizeiptr keySize,
//                     void* value, GLsizeiptr valueSize, const void* userParam)
void DoTrampoline_glBlobCacheCallbacksANGLE(HostCode callee, ThreadState* state) {
  using SetProc =
      void (*)(const void*, GLsizeiptr_t, const void*, GLsizeiptr_t, const void*);
  using GetProc =
      GLsizeiptr_t (*)(const void*, GLsizeiptr_t, void*, GLsizeiptr_t, const void*);
  using PFN_callee = void (*)(SetProc, GetProc, const void*);
  PFN_callee callee_function = AsFuncPtr(callee);

  auto [guest_set, guest_get, user_param] = GuestParamsValues<PFN_callee>(state);
  SetProc host_set = ToGuestAddr(guest_set) == 0
                         ? nullptr
                         : WrapGuestFunction(guest_set, "GLSETBLOBPROCANGLE");
  GetProc host_get = ToGuestAddr(guest_get) == 0
                         ? nullptr
                         : WrapGuestFunction(guest_get, "GLGETBLOBPROCANGLE");

  callee_function(host_set, host_get, user_param);
}

template <TrampolineFunc kFunc>
void WrapAsCustom(HostCode fn, const char* name) {
  WrapHostFunctionImpl(fn, kFunc, name);
}

// --- Demand-driven wrap table (names Chromium probes; ANGLE prototypes). ---

struct AngleProc {
  const char* name;
  void (*wrap)(HostCode fn, const char* name);
};

constexpr AngleProc kAngleProcs[] = {
    // GL_ANGLE_robust_client_memory — Chromium routes ALL state queries through
    // these when the extension is advertised; glGetIntegervRobustANGLE is the
    // proven Helium GPU-process crasher.
    {"glGetBooleanvRobustANGLE", WrapAs<void (*)(GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetBufferParameteri64vRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetBufferParameterivRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetBufferPointervRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetFloatvRobustANGLE", WrapAs<void (*)(GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetFramebufferAttachmentParameterivRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetInteger64i_vRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLuint_t, GLsizei_t, void*, void*)>},
    {"glGetInteger64vRobustANGLE", WrapAs<void (*)(GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetIntegeri_vRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLuint_t, GLsizei_t, void*, void*)>},
    {"glGetIntegervRobustANGLE", WrapAs<void (*)(GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetInternalformativRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetMultisamplefvRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLuint_t, GLsizei_t, void*, void*)>},
    {"glGetProgramivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetQueryObjecti64vRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetQueryObjectivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetQueryObjectui64vRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetQueryObjectuivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetQueryivRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetRenderbufferParameterivRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetSamplerParameterfvRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetSamplerParameterivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetShaderivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetTexParameterfvRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetTexParameterivRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetUniformfvRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLint_t, GLsizei_t, void*, void*)>},
    {"glGetUniformivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLint_t, GLsizei_t, void*, void*)>},
    {"glGetUniformuivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLint_t, GLsizei_t, void*, void*)>},
    {"glGetVertexAttribIivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetVertexAttribIuivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetVertexAttribPointervRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetVertexAttribfvRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetVertexAttribivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetActiveUniformBlockivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLuint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glReadPixelsRobustANGLE",
     WrapAs<void (*)(GLint_t, GLint_t, GLsizei_t, GLsizei_t, GLenum_t, GLenum_t,
                     GLsizei_t, void*, void*, void*, void*)>},
    {"glSamplerParameterfvRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, const void*)>},
    {"glSamplerParameterivRobustANGLE",
     WrapAs<void (*)(GLuint_t, GLenum_t, GLsizei_t, const void*)>},
    {"glTexImage2DRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLint_t, GLint_t, GLsizei_t, GLsizei_t, GLint_t,
                     GLenum_t, GLenum_t, GLsizei_t, const void*)>},
    {"glTexImage3DRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLint_t, GLint_t, GLsizei_t, GLsizei_t, GLsizei_t,
                     GLint_t, GLenum_t, GLenum_t, GLsizei_t, const void*)>},
    {"glTexParameterfvRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, const void*)>},
    {"glTexParameterivRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLenum_t, GLsizei_t, const void*)>},
    {"glTexSubImage2DRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLint_t, GLint_t, GLint_t, GLsizei_t, GLsizei_t,
                     GLenum_t, GLenum_t, GLsizei_t, const void*)>},
    {"glTexSubImage3DRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLint_t, GLint_t, GLint_t, GLint_t, GLsizei_t,
                     GLsizei_t, GLsizei_t, GLenum_t, GLenum_t, GLsizei_t,
                     const void*)>},

    // GL_ANGLE_get_tex_level_parameter
    {"glGetTexLevelParameterfvANGLE",
     WrapAs<void (*)(GLenum_t, GLint_t, GLenum_t, void*)>},
    {"glGetTexLevelParameterfvRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetTexLevelParameterivANGLE",
     WrapAs<void (*)(GLenum_t, GLint_t, GLenum_t, void*)>},
    {"glGetTexLevelParameterivRobustANGLE",
     WrapAs<void (*)(GLenum_t, GLint_t, GLenum_t, GLsizei_t, void*, void*)>},

    // GL_ANGLE_multi_draw
    {"glMultiDrawArraysANGLE",
     WrapAs<void (*)(GLenum_t, const void*, const void*, GLsizei_t)>},
    {"glMultiDrawArraysInstancedANGLE",
     WrapAs<void (*)(GLenum_t, const void*, const void*, const void*, GLsizei_t)>},
    {"glMultiDrawElementsANGLE",
     WrapAs<void (*)(GLenum_t, const void*, GLenum_t, const void*, GLsizei_t)>},
    {"glMultiDrawElementsInstancedANGLE",
     WrapAs<void (*)(GLenum_t, const void*, GLenum_t, const void*, const void*,
                     GLsizei_t)>},

    // GL_ANGLE_polygon_mode / GL_ANGLE_request_extension
    {"glPolygonModeANGLE", WrapAs<void (*)(GLenum_t, GLenum_t)>},
    {"glRequestExtensionANGLE", WrapAs<void (*)(const char*)>},

    // GL_ANGLE_shader_pixel_local_storage
    {"glBeginPixelLocalStorageANGLE", WrapAs<void (*)(GLsizei_t, const void*)>},
    {"glEndPixelLocalStorageANGLE", WrapAs<void (*)(GLsizei_t, const void*)>},
    {"glFramebufferMemorylessPixelLocalStorageANGLE",
     WrapAs<void (*)(GLint_t, GLenum_t)>},
    {"glFramebufferPixelLocalClearValuefvANGLE",
     WrapAs<void (*)(GLint_t, const void*)>},
    {"glFramebufferPixelLocalClearValueivANGLE",
     WrapAs<void (*)(GLint_t, const void*)>},
    {"glFramebufferPixelLocalClearValueuivANGLE",
     WrapAs<void (*)(GLint_t, const void*)>},
    {"glFramebufferPixelLocalStorageInterruptANGLE", WrapAs<void (*)()>},
    {"glFramebufferPixelLocalStorageRestoreANGLE", WrapAs<void (*)()>},
    {"glFramebufferTexturePixelLocalStorageANGLE",
     WrapAs<void (*)(GLint_t, GLuint_t, GLint_t, GLint_t)>},
    {"glGetFramebufferPixelLocalStorageParameterfvRobustANGLE",
     WrapAs<void (*)(GLint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glGetFramebufferPixelLocalStorageParameterivRobustANGLE",
     WrapAs<void (*)(GLint_t, GLenum_t, GLsizei_t, void*, void*)>},
    {"glPixelLocalStorageBarrierANGLE", WrapAs<void (*)()>},

    // GL_CHROMIUM_copy_texture / GL_CHROMIUM_bind_uniform_location
    {"glBindUniformLocationCHROMIUM",
     WrapAs<void (*)(GLuint_t, GLint_t, const char*)>},
    {"glCopySubTextureCHROMIUM",
     WrapAs<void (*)(GLuint_t, GLint_t, GLenum_t, GLuint_t, GLint_t, GLint_t,
                     GLint_t, GLint_t, GLint_t, GLsizei_t, GLsizei_t, GLboolean_t,
                     GLboolean_t, GLboolean_t)>},
    {"glCopyTextureCHROMIUM",
     WrapAs<void (*)(GLuint_t, GLint_t, GLenum_t, GLuint_t, GLint_t, GLint_t,
                     GLenum_t, GLboolean_t, GLboolean_t, GLboolean_t)>},

    // GL_ANGLE_memory_object_flags / GL_ANGLE_vulkan_image
    {"glAcquireTexturesANGLE",
     WrapAs<void (*)(GLuint_t, const void*, const void*)>},
    {"glReleaseTexturesANGLE", WrapAs<void (*)(GLuint_t, const void*, void*)>},
    {"glTexStorageMemFlags2DANGLE",
     WrapAs<void (*)(GLenum_t, GLsizei_t, GLenum_t, GLsizei_t, GLsizei_t, GLuint_t,
                     GLuint64_t, GLbitfield_t, GLbitfield_t, const void*)>},

    // GL_ANGLE_blob_cache (guest callbacks — custom marshaller)
    {"glBlobCacheCallbacksANGLE", WrapAsCustom<DoTrampoline_glBlobCacheCallbacksANGLE>},

    // EGL_KHR_debug (guest callback — custom marshaller)
    {"eglDebugMessageControlKHR", WrapAsCustom<DoTrampoline_eglDebugMessageControlKHR>},

    // EGL_CHROMIUM_sync_control / EGL_ANGLE_sync_control_rate
    {"eglGetSyncValuesCHROMIUM",
     WrapAs<EGLBoolean_t (*)(void*, void*, void*, void*, void*)>},
    {"eglGetMscRateANGLE", WrapAs<EGLBoolean_t (*)(void*, void*, void*, void*)>},

    // Assorted ANGLE EGL extensions reachable on the Android Vulkan backend.
    {"eglExportVkImageANGLE", WrapAs<EGLBoolean_t (*)(void*, void*, void*, void*)>},
    {"eglHandleGPUSwitchANGLE", WrapAs<void (*)(void*)>},
    {"eglLockVulkanQueueANGLE", WrapAs<void (*)(void*)>},
    {"eglQueryDisplayAttribANGLE", WrapAs<EGLBoolean_t (*)(void*, EGLint_t, void*)>},
    {"eglQueryStringiANGLE", WrapAs<const char* (*)(void*, EGLint_t, EGLint_t)>},
    {"eglSetValidationEnabledANGLE", WrapAs<void (*)(EGLBoolean_t)>},
    {"eglUnlockVulkanQueueANGLE", WrapAs<void (*)(void*)>},
    {"eglWaitUntilWorkScheduledANGLE", WrapAs<void (*)(void*)>},
};

const AngleProc* FindAngleProc(const char* name) {
  for (const auto& proc : kAngleProcs) {
    if (strcmp(proc.name, name) == 0) {
      return &proc;
    }
  }
  return nullptr;
}

// Chained override of the upstream eglGetProcAddress custom trampoline. The
// callee is a ChainedTrampoline holding the primary's {marshal_and_call, thunk};
// the thunk is the host eglGetProcAddress itself (the primary table binds it).
void DoDigitalisEglGetProcAddress(HostCode callee, ThreadState* state) {
  const auto* chain = static_cast<const ChainedTrampoline*>(callee);

  // Capture the proc name BEFORE running the primary: x0 doubles as the return
  // register, so the primary trampoline overwrites it with the result.
  auto [proc_name_param] = GuestParamsValues<PFN_eglGetProcAddress>(state);
  const char* proc_name = static_cast<const char*>(proc_name_param);

  // Full upstream behavior first: host lookup + kOpenGLTrampolines wrapping.
  chain->marshal_and_call(chain->thunk, state);

  auto&& [ret] = GuestReturnReference<PFN_eglGetProcAddress>(state);
  if (ToGuestAddr(ret) != 0 || proc_name == nullptr) {
    return;  // Upstream wrapped it (or nothing to retry).
  }

  const AngleProc* proc = FindAngleProc(proc_name);
  if (proc == nullptr) {
    return;  // Not covered here either; keep the upstream NULL.
  }

  PFN_eglGetProcAddress host_get_proc_address = AsFuncPtr(chain->thunk);
  EglGenericFn host_fn = host_get_proc_address(proc_name);
  if (host_fn == nullptr) {
    return;  // Host driver does not have it; NULL is the correct answer.
  }

  proc->wrap(reinterpret_cast<HostCode>(host_fn), proc->name);
  TRACE("digitalis eglGetProcAddress: wrapped \"%s\"", proc->name);
  ret = host_fn;
}

const KnownTrampoline kDigitalisLibEGLOverrides[] = {
    {"eglGetProcAddress", DoDigitalisEglGetProcAddress, nullptr},
};

REGISTER_DIGITALIS_EXTRA_TRAMPOLINE_OVERRIDES("libEGL.so", kDigitalisLibEGLOverrides)

}  // namespace
}  // namespace berberis

#endif  // defined(__x86_64__)
