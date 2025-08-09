#
# Copyright (C) 2023 The Android Open Source Project
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

# This file defines:
#   BERBERIS_PRODUCT_PACKAGES_RISCV64_TO_X86_64 - list of main product packages for riscv64 to
#                                                 x86_64 translation.
#

include frameworks/libs/native_bridge_support/native_bridge_support.mk

# Note: Keep in sync with `berberis_all_riscv64_to_x86_64_defaults` in Android.bp.
BERBERIS_PRODUCT_PACKAGES_RISCV64_TO_X86_64 := \
    libberberis_exec_region \
    libberberis_proxy_libEGL \
    libberberis_proxy_libGLESv1_CM \
    libberberis_proxy_libGLESv2 \
    libberberis_proxy_libGLESv3 \
    libberberis_proxy_libOpenMAXAL \
    libberberis_proxy_libOpenSLES \
    libberberis_proxy_libaaudio \
    libberberis_proxy_libamidi \
    libberberis_proxy_libandroid \
    libberberis_proxy_libandroid_runtime \
    libberberis_proxy_libbinder_ndk \
    libberberis_proxy_libc \
    libberberis_proxy_libcamera2ndk \
    libberberis_proxy_libjnigraphics \
    libberberis_proxy_libm \
    libberberis_proxy_libmediandk \
    libberberis_proxy_libnativehelper \
    libberberis_proxy_libnativewindow \
    libberberis_proxy_libneuralnetworks \
    libberberis_proxy_libvulkan \
    libberberis_proxy_libwebviewchromium_plat_support \
    berberis_prebuilt_riscv64 \
    berberis_program_runner_binfmt_misc_riscv64 \
    berberis_program_runner_riscv64 \
    libberberis_riscv64

# TODO(b/277625560): Include $(NATIVE_BRIDGE_PRODUCT_PACKAGES) instead
# when all its bits are ready for riscv64.
BERBERIS_PRODUCT_PACKAGES_RISCV64_TO_X86_64 += $(NATIVE_BRIDGE_PRODUCT_PACKAGES_RISCV64_READY)

BERBERIS_DISTRIBUTION_ARTIFACTS_RISCV64 := \
    system/bin/berberis_program_runner_binfmt_misc_riscv64 \
    system/bin/berberis_program_runner_riscv64 \
    system/bin/riscv64/app_process64 \
    system/bin/riscv64/linker64 \
    system/etc/binfmt_misc/riscv64_dyn \
    system/etc/binfmt_misc/riscv64_exe \
    system/etc/init/berberis.rc \
    system/etc/ld.config.riscv64.txt \
    system/lib64/libberberis_exec_region.so \
    system/lib64/libberberis_proxy_libEGL.so \
    system/lib64/libberberis_proxy_libGLESv1_CM.so \
    system/lib64/libberberis_proxy_libGLESv2.so \
    system/lib64/libberberis_proxy_libGLESv3.so \
    system/lib64/libberberis_proxy_libOpenMAXAL.so \
    system/lib64/libberberis_proxy_libOpenSLES.so \
    system/lib64/libberberis_proxy_libaaudio.so \
    system/lib64/libberberis_proxy_libamidi.so \
    system/lib64/libberberis_proxy_libandroid.so \
    system/lib64/libberberis_proxy_libandroid_runtime.so \
    system/lib64/libberberis_proxy_libbinder_ndk.so \
    system/lib64/libberberis_proxy_libc.so \
    system/lib64/libberberis_proxy_libcamera2ndk.so \
    system/lib64/libberberis_proxy_libjnigraphics.so \
    system/lib64/libberberis_proxy_libm.so \
    system/lib64/libberberis_proxy_libmediandk.so \
    system/lib64/libberberis_proxy_libnativehelper.so \
    system/lib64/libberberis_proxy_libnativewindow.so \
    system/lib64/libberberis_proxy_libneuralnetworks.so \
    system/lib64/libberberis_proxy_libvulkan.so \
    system/lib64/libberberis_proxy_libwebviewchromium_plat_support.so \
    system/lib64/libberberis_riscv64.so \
    system/lib64/riscv64/ld-android.so \
    system/lib64/riscv64/libEGL.so \
    system/lib64/riscv64/libGLESv1_CM.so \
    system/lib64/riscv64/libGLESv2.so \
    system/lib64/riscv64/libGLESv3.so \
    system/lib64/riscv64/libOpenMAXAL.so \
    system/lib64/riscv64/libOpenSLES.so \
    system/lib64/riscv64/libaaudio.so \
    system/lib64/riscv64/libamidi.so \
    system/lib64/riscv64/libandroid.so \
    system/lib64/riscv64/libandroid_runtime.so \
    system/lib64/riscv64/libandroidicu.so \
    system/lib64/riscv64/libbase.so \
    system/lib64/riscv64/libbinder_ndk.so \
    system/lib64/riscv64/libc++.so \
    system/lib64/riscv64/libc.so \
    system/lib64/riscv64/libcamera2ndk.so \
    system/lib64/riscv64/libcompiler_rt.so \
    system/lib64/riscv64/libcrypto.so \
    system/lib64/riscv64/libcutils.so \
    system/lib64/riscv64/libdl.so \
    system/lib64/riscv64/libdl_android.so \
    system/lib64/riscv64/libicu.so \
    system/lib64/riscv64/libicui18n.so \
    system/lib64/riscv64/libicuuc.so \
    system/lib64/riscv64/libjnigraphics.so \
    system/lib64/riscv64/liblog.so \
    system/lib64/riscv64/libm.so \
    system/lib64/riscv64/libmediandk.so \
    system/lib64/riscv64/libnative_bridge_vdso.so \
    system/lib64/riscv64/libnativehelper.so \
    system/lib64/riscv64/libnativewindow.so \
    system/lib64/riscv64/libneuralnetworks.so \
    system/lib64/riscv64/libsqlite.so \
    system/lib64/riscv64/libssl.so \
    system/lib64/riscv64/libstdc++.so \
    system/lib64/riscv64/libsync.so \
    system/lib64/riscv64/libutils.so \
    system/lib64/riscv64/libvndksupport.so \
    system/lib64/riscv64/libvulkan.so \
    system/lib64/riscv64/libwebviewchromium_plat_support.so \
    system/lib64/riscv64/libz.so
