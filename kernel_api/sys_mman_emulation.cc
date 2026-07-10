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
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <elf.h>
#include <unistd.h>
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

void* MmapForGuest(void* addr, size_t length, int prot, int flags, int fd, off64_t offset) {
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
  //
  // For APK-contained libraries, the fd points to the APK file and the offset
  // is non-zero. The ELF header is NOT at offset 0 of the fd. We find it by
  // searching backward from the current mapping offset.
  if (result != MAP_FAILED && fd >= 0 &&
      (flags & MAP_FIXED) && (flags & MAP_PRIVATE) && (prot & PROT_WRITE)) {
    Elf64_Ehdr ehdr;
    off64_t elf_base = -1;
    // Try offset 0 first (standalone ELF files).
    if (pread(fd, &ehdr, sizeof(ehdr), 0) == sizeof(ehdr) &&
        ehdr.e_ident[EI_MAG0] == ELFMAG0 && ehdr.e_ident[EI_MAG1] == ELFMAG1 &&
        ehdr.e_ident[EI_MAG2] == ELFMAG2 && ehdr.e_ident[EI_MAG3] == ELFMAG3) {
      elf_base = 0;
    }
    // If not found at 0, search backward from the mapping offset for the ELF
    // magic. This handles libraries stored inside APKs where the ELF starts at
    // a non-zero offset within the container file.
    if (elf_base < 0 && offset > 0) {
      size_t page_size = static_cast<size_t>(getpagesize());
      // The data segment is typically within 1MB of the ELF header. Search back
      // in page increments up to 2MB.
      off64_t search_limit = (offset > 0x200000) ? offset - 0x200000 : 0;
      for (off64_t candidate = (offset & ~(off64_t)(page_size - 1)) - page_size;
           candidate >= search_limit;
           candidate -= page_size) {
        if (pread(fd, &ehdr, sizeof(ehdr), candidate) == sizeof(ehdr) &&
            ehdr.e_ident[EI_MAG0] == ELFMAG0 && ehdr.e_ident[EI_MAG1] == ELFMAG1 &&
            ehdr.e_ident[EI_MAG2] == ELFMAG2 && ehdr.e_ident[EI_MAG3] == ELFMAG3 &&
            ehdr.e_type == ET_DYN && ehdr.e_machine == EM_AARCH64) {
          elf_base = candidate;
          break;
        }
      }
    }
    if (elf_base >= 0 && ehdr.e_phnum > 0 && ehdr.e_phnum <= 64) {
      Elf64_Phdr phdrs[64];
      ssize_t phdr_bytes = ehdr.e_phnum * sizeof(Elf64_Phdr);
      if (pread(fd, phdrs, phdr_bytes, elf_base + ehdr.e_phoff) == phdr_bytes) {
        size_t page_size = static_cast<size_t>(getpagesize());
        size_t mapped_length = (length + page_size - 1) & ~(page_size - 1);
        for (int i = 0; i < ehdr.e_phnum; i++) {
          if (phdrs[i].p_type != PT_LOAD) continue;
          if (phdrs[i].p_memsz <= phdrs[i].p_filesz) continue;
          // This segment has .bss (memsz > filesz).
          // Adjust p_offset to be relative to the container file (APK).
          off64_t seg_file_offset = elf_base + (off64_t)phdrs[i].p_offset;
          off64_t page_aligned_seg_offset = seg_file_offset & ~(off64_t)(page_size - 1);
          if (offset != page_aligned_seg_offset) continue;
          // Two PT_LOAD segments can share a page-aligned file offset (when
          // both p_offset values fall in the same file page, e.g. one segment
          // ends in the page and the next starts in the same page). In that
          // case the page_aligned_seg_offset check above matches BOTH segments
          // for whichever mmap the linker is doing. Disambiguate by length:
          // this mapping belongs to segment i only if it is small enough to be
          // explained by segment i's own memsz from its start in this page.
          size_t seg_in_mapping_offset = (size_t)(seg_file_offset - offset);
          size_t seg_max_mapping =
              (seg_in_mapping_offset + phdrs[i].p_memsz + page_size - 1) &
              ~(page_size - 1);
          if (mapped_length > seg_max_mapping) continue;
          off64_t seg_file_end = seg_file_offset + phdrs[i].p_filesz;
          if (seg_file_end >= offset && seg_file_end < offset + (off64_t)mapped_length) {
            size_t bss_start = (size_t)(seg_file_end - offset);
            // Cap at the segment's actual BSS size (memsz - filesz).
            size_t seg_bss_size = phdrs[i].p_memsz - phdrs[i].p_filesz;
            size_t bytes_to_zero = std::min(mapped_length - bss_start, seg_bss_size);
            if (bytes_to_zero > 0 && bss_start < mapped_length) {
              // When two .bss-bearing PT_LOADs share a file page, the checks
              // above can accept segment i against the OTHER segment's
              // mapping (offset and rounded length are identical for both),
              // and zeroing segment i's virtual bss range would then corrupt
              // the sibling's file-backed bytes. The signals available here
              // cannot distinguish the two mappings, so be conservative: if
              // the would-be-zeroed window overlaps any other PT_LOAD's file
              // extent, skip the safety-net zeroing rather than risk
              // corrupting real content.
              off64_t zero_file_start = offset + (off64_t)bss_start;
              off64_t zero_file_end = zero_file_start + (off64_t)bytes_to_zero;
              bool overlaps_other_segment = false;
              for (int j = 0; j < ehdr.e_phnum; j++) {
                if (j == i || phdrs[j].p_type != PT_LOAD || phdrs[j].p_filesz == 0) continue;
                off64_t j_start = elf_base + (off64_t)phdrs[j].p_offset;
                off64_t j_end = j_start + (off64_t)phdrs[j].p_filesz;
                if (j_start < zero_file_end && zero_file_start < j_end) {
                  overlaps_other_segment = true;
                  break;
                }
              }
              if (!overlaps_other_segment) {
                memset(static_cast<char*>(result) + bss_start, 0, bytes_to_zero);
              }
            }
          }
        }
      }
    }
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
