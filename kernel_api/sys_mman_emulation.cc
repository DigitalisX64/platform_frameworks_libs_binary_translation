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

#include "berberis/kernel_api/sys_mman_emulation.h"

#include <sys/mman.h>

#include <cerrno>
// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
#include <cstdint>
#include <elf.h>
#include <unistd.h>

#include <android/log.h>
#endif
// endregion

#include "berberis/base/mmap.h"
#include "berberis/base/prctl_helpers.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

namespace {

int ToHostProt(int guest_prot) {
  if (guest_prot & PROT_EXEC) {
    // Guest EXEC should _not_ be host EXEC but should be host READ!
    return (guest_prot & ~PROT_EXEC) | PROT_READ;
  }
  return guest_prot;
}

// Clobbers errno.
void UpdateGuestProt(int guest_prot, void* addr, size_t length) {
  GuestAddr guest_addr = ToGuestAddr(addr);
  GuestMapShadow* shadow = GuestMapShadow::GetInstance();
  if (guest_prot & PROT_EXEC) {
    shadow->SetExecutable(guest_addr, length);
  } else {
    shadow->ClearExecutable(guest_addr, length);
  }
}

}  // namespace

// ATTENTION: the order of mmap/mprotect/munmap and SetExecutable/ClearExecutable is essential!
//
// The issue here is that threads might be executing the code being munmap'ed or mprotect'ed.
// SetExecutable/ClearExecutable should flush code cache and notify threads to restart.
// If other thread starts translation after actual mmap/mprotect/munmap but before xbit update,
// it might pick up an already obsolete code.

// region digitalis - mmap diagnostic counters
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
static uint64_t g_mmap_count = 0;
static uint64_t g_mmap_fail_count = 0;
#endif
// endregion

void* MmapForGuest(void* addr, size_t length, int prot, int flags, int fd, off64_t offset) {
  // region digitalis - log mmap calls
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  uint64_t n = ++g_mmap_count;
#endif
  // endregion
  void* result = mmap64(addr, length, ToHostProt(prot), flags, fd, offset);
  if (result != MAP_FAILED) {
    UpdateGuestProt(prot, result, length);
  }
  // region digitalis - zero .bss partial pages for ELF segment mappings
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  // When the guest linker mmaps a file-backed page for a LOAD segment that has
  // .bss (p_memsz > p_filesz), the kernel maps the full page from the file,
  // including bytes beyond p_filesz (e.g., section headers). The linker should
  // memset these to zero, but the ARM64 memset under translation may not execute
  // correctly. As a safety net, we detect ELF segments with .bss and zero the
  // partial page ourselves after the mmap.
  if (result != MAP_FAILED && fd >= 0 &&
      (flags & MAP_FIXED) && (flags & MAP_PRIVATE) && (prot & PROT_WRITE)) {
    Elf64_Ehdr ehdr;
    if (pread(fd, &ehdr, sizeof(ehdr), 0) == sizeof(ehdr) &&
        ehdr.e_ident[EI_MAG0] == ELFMAG0 && ehdr.e_ident[EI_MAG1] == ELFMAG1 &&
        ehdr.e_ident[EI_MAG2] == ELFMAG2 && ehdr.e_ident[EI_MAG3] == ELFMAG3 &&
        ehdr.e_phnum > 0 && ehdr.e_phnum <= 64) {
      Elf64_Phdr phdrs[64];
      ssize_t phdr_bytes = ehdr.e_phnum * sizeof(Elf64_Phdr);
      if (pread(fd, phdrs, phdr_bytes, ehdr.e_phoff) == phdr_bytes) {
        // The kernel rounds the mapping up to page size, so bytes beyond the
        // guest's requested length but within the page are still mapped from
        // the file. Use page-aligned length to cover the full mapped region.
        size_t page_size = static_cast<size_t>(getpagesize());
        size_t mapped_length = (length + page_size - 1) & ~(page_size - 1);
        for (int i = 0; i < ehdr.e_phnum; i++) {
          if (phdrs[i].p_type != PT_LOAD) continue;
          if (phdrs[i].p_memsz <= phdrs[i].p_filesz) continue;
          // This segment has .bss (memsz > filesz).
          // Only process this segment if this mmap belongs to it.
          // The linker maps each segment with offset = floor(p_offset, page_size).
          // Two segments may share overlapping file ranges but have different
          // page-aligned offsets, so only match the one this mmap is actually for.
          off64_t page_aligned_seg_offset = phdrs[i].p_offset & ~(off64_t)(page_size - 1);
          if (offset != page_aligned_seg_offset) continue;
          // Check if our mapping covers the boundary between file data and .bss.
          off64_t seg_file_end = phdrs[i].p_offset + phdrs[i].p_filesz;
          if (seg_file_end >= offset && seg_file_end < offset + (off64_t)mapped_length) {
            size_t bss_start = (size_t)(seg_file_end - offset);
            size_t bytes_to_zero = mapped_length - bss_start;
            if (bytes_to_zero > 0 && bss_start < mapped_length) {
              memset(static_cast<char*>(result) + bss_start, 0, bytes_to_zero);
              static uint64_t bss_zero_count = 0;
              if (++bss_zero_count <= 10) {
                __android_log_print(ANDROID_LOG_INFO, "berberis",
                    "bss-zero#%lu: %p+0x%lx len=0x%lx (seg off=0x%lx filesz=0x%lx memsz=0x%lx)",
                    (unsigned long)bss_zero_count,
                    result, (unsigned long)bss_start, (unsigned long)bytes_to_zero,
                    (unsigned long)phdrs[i].p_offset,
                    (unsigned long)phdrs[i].p_filesz, (unsigned long)phdrs[i].p_memsz);
              }
            }
          }
        }
      }
    }
  }
#endif
  // endregion
  // region digitalis - log all executable mmaps and first 30
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  if (n <= 30 || n % 500 == 0 || (prot & 4)) {
    __android_log_print(ANDROID_LOG_ERROR, "berberis",
        "mmap#%lu addr=%p→%p len=0x%lx prot=%d flags=0x%x fd=%d off=0x%lx",
        (unsigned long)n, addr, result, (unsigned long)length, prot, flags, fd, (unsigned long)offset);
  }
#endif
  // endregion
  // region digitalis - log failures
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  if (result == MAP_FAILED) {
    int saved_errno = errno;
    ++g_mmap_fail_count;
    __android_log_print(ANDROID_LOG_ERROR, "berberis",
        "mmap FAILED #%lu errno=%d addr=%p len=0x%lx prot=%d flags=0x%x fd=%d off=0x%lx (total_fail=%lu)",
        (unsigned long)n, saved_errno, addr, (unsigned long)length, prot, flags, fd,
        (unsigned long)offset, (unsigned long)g_mmap_fail_count);
    errno = saved_errno;
  }
#endif
  // endregion
  return result;
}

int MunmapForGuest(void* addr, size_t length) {
  GuestMapShadow::GetInstance()->ClearExecutable(ToGuestAddr(addr), length);
  return munmap(addr, length);
}

int MprotectForGuest(void* addr, size_t length, int prot) {
  // In b/218772975 the app is scanning "/proc/self/maps" and tries to mprotect
  // mappings for some libraries found there (for unknown reason) effectively removing
  // execution permission. GuestMapShadow is pre-populated with such mappings, so we
  // suppress guest mprotect for them.
  if (GuestMapShadow::GetInstance()->IntersectsWithProtectedMapping(
          addr, static_cast<char*>(addr) + length)) {
    TRACE("Suppressing guest mprotect(%p, %zu) on a mapping protected from guest", addr, length);
    errno = EACCES;
    return -1;
  }

  UpdateGuestProt(prot, addr, length);
  return mprotect(addr, length, ToHostProt(prot));
}

void* MremapForGuest(void* old_addr, size_t old_size, size_t new_size, int flags, void* new_addr) {
  // As we drop xbit for host mmap calls, host mappings might differ from guest
  // mappings, and host mremap might work when guest mremap should not. Check in
  // advance to avoid that. Rules for checks:
  // 1. Shrink without MREMAP_FIXED - always Ok.
  // 2. Shrink with MREMAP_FIXED - needs consistent permissions within new_size.
  // 3. Grow - needs consistent permissions within old_size.
  GuestMapShadow* shadow = GuestMapShadow::GetInstance();
  if (new_size <= old_size) {
    if ((flags & MREMAP_FIXED) &&
        shadow->GetExecutable(ToGuestAddr(old_addr), new_size) == kBitMixed) {
      errno = EFAULT;
      return MAP_FAILED;
    }
  } else {
    if (shadow->GetExecutable(ToGuestAddr(old_addr), old_size) == kBitMixed) {
      errno = EFAULT;
      return MAP_FAILED;
    }
  }

  void* result = mremap(old_addr, old_size, new_size, flags, new_addr);

  if (result != MAP_FAILED) {
    shadow->RemapExecutable(ToGuestAddr(old_addr), old_size, ToGuestAddr(result), new_size);
  }
  return result;
}

}  // namespace berberis
