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

#include "gtest/gtest.h"

#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "berberis/kernel_api/open_emulation.h"
#include "berberis/kernel_api/sys_mman_emulation.h"

namespace berberis {

namespace {

// ---------------------------------------------------------------------------
// Target 1: MmapForGuest's ARM64 .bss partial-page zeroing safety net.
//
// When the guest linker maps a file-backed page for a PT_LOAD segment that has
// .bss (p_memsz > p_filesz), the kernel maps the whole page from the file --
// including the section headers / padding beyond p_filesz. The guest linker
// would normally memset those to zero, but the ARM64 memset under translation
// may not run correctly, so MmapForGuest re-does the zeroing itself: it locates
// the ELF header (at fd offset 0, or by a backward page search for APK-embedded
// libraries), scans the phdr table, and zeroes [p_filesz .. page_end) of the
// segment's file-backed page.
//
// These tests build REAL small ELF images in a memfd and map them exactly the
// way the guest linker would (MAP_FIXED | MAP_PRIVATE, PROT_WRITE, page-aligned
// file offset), then observe which bytes of the resulting mapping were zeroed
// vs. survived. Every would-be-zeroed byte is pre-filled with a sentinel so the
// zeroing is directly observable.
// ---------------------------------------------------------------------------

constexpr uint8_t kSentinel = 0xAB;

size_t PageSize() {
  return static_cast<size_t>(getpagesize());
}

// Fabricates a minimal, well-formed ELF64 program header for a PT_LOAD segment.
Elf64_Phdr MakeLoad(uint64_t offset, uint64_t filesz, uint64_t memsz, uint64_t align) {
  Elf64_Phdr p{};
  p.p_type = PT_LOAD;
  p.p_flags = PF_R | PF_W;
  p.p_offset = offset;
  p.p_vaddr = offset;
  p.p_paddr = offset;
  p.p_filesz = filesz;
  p.p_memsz = memsz;
  p.p_align = align;
  return p;
}

// Fabricates a minimal ELF64 header. The zeroing code accepts the header at fd
// offset 0 on ELF-magic alone, but the backward-search path additionally
// requires ET_DYN + EM_AARCH64, so we always set those to exercise both paths
// with the same builder.
Elf64_Ehdr MakeEhdr(uint16_t phnum) {
  Elf64_Ehdr e{};
  e.e_ident[EI_MAG0] = ELFMAG0;
  e.e_ident[EI_MAG1] = ELFMAG1;
  e.e_ident[EI_MAG2] = ELFMAG2;
  e.e_ident[EI_MAG3] = ELFMAG3;
  e.e_ident[EI_CLASS] = ELFCLASS64;
  e.e_ident[EI_DATA] = ELFDATA2LSB;
  e.e_ident[EI_VERSION] = EV_CURRENT;
  e.e_type = ET_DYN;
  e.e_machine = EM_AARCH64;
  e.e_version = EV_CURRENT;
  e.e_phoff = sizeof(Elf64_Ehdr);
  e.e_phentsize = sizeof(Elf64_Phdr);
  e.e_phnum = phnum;
  return e;
}

// Builds an in-memory file image of `total` bytes, filled with the sentinel,
// with an ELF header + phdr table placed at byte offset `elf_base`. Passing an
// empty phdr list with elf_base < 0 leaves the whole image as sentinel bytes
// (the non-ELF negative case).
std::vector<uint8_t> BuildImage(size_t total,
                                off64_t elf_base,
                                const std::vector<Elf64_Phdr>& phdrs) {
  std::vector<uint8_t> img(total, kSentinel);
  if (elf_base >= 0) {
    Elf64_Ehdr e = MakeEhdr(static_cast<uint16_t>(phdrs.size()));
    memcpy(img.data() + elf_base, &e, sizeof(e));
    memcpy(img.data() + elf_base + e.e_phoff, phdrs.data(),
           phdrs.size() * sizeof(Elf64_Phdr));
  }
  return img;
}

// Creates an anonymous in-memory file (memfd) holding `image`. Using the raw
// syscall avoids depending on memfd_create() being declared by the host libc
// headers. Returns -1 on failure.
int CreateImageFd(const std::vector<uint8_t>& image) {
  int fd = static_cast<int>(syscall(SYS_memfd_create, "digitalis-elf-test", 0u));
  if (fd < 0) {
    return -1;
  }
  ssize_t written = write(fd, image.data(), image.size());
  if (written != static_cast<ssize_t>(image.size())) {
    close(fd);
    return -1;
  }
  return fd;
}

// Asserts that [from, to) of `p` are all `val`; on mismatch, reports the first
// offending byte so a failure localizes the mis-zeroed region.
::testing::AssertionResult BytesAllEqual(const uint8_t* p,
                                         size_t from,
                                         size_t to,
                                         uint8_t val) {
  for (size_t i = from; i < to; ++i) {
    if (p[i] != val) {
      return ::testing::AssertionFailure()
             << "byte[" << i << "] = 0x" << std::hex << static_cast<int>(p[i])
             << ", expected 0x" << static_cast<int>(val) << std::dec
             << " (checking range [" << from << ", " << to << "))";
    }
  }
  return ::testing::AssertionSuccess();
}

// Reserves a single page of address space so a subsequent MAP_FIXED mapping has
// a stable, valid target address (mirrors how the guest linker reserves the
// segment range up front, then MAP_FIXED-maps each PT_LOAD into it).
uint8_t* ReservePage() {
  void* r = mmap(nullptr, PageSize(), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (r == MAP_FAILED) {
    return nullptr;
  }
  return static_cast<uint8_t*>(r);
}

// Case (a): a single PT_LOAD with .bss, ELF header at fd offset 0.
// filesz=100, memsz=page. Mapping the segment's page must zero [100, page)
// while the first 100 file-content bytes survive.
TEST(Arm64MmapBssZeroTest, SingleSegmentZeroesTailBeyondFilesz) {
  const size_t page = PageSize();
  ASSERT_GE(page, 4096u);
  const uint64_t kFilesz = 100;

  std::vector<Elf64_Phdr> phdrs = {MakeLoad(page, kFilesz, page, page)};
  std::vector<uint8_t> image = BuildImage(2 * page, /*elf_base=*/0, phdrs);
  int fd = CreateImageFd(image);
  ASSERT_GE(fd, 0);

  uint8_t* base = ReservePage();
  ASSERT_NE(base, nullptr);
  void* got = MmapForGuest(base, page, PROT_READ | PROT_WRITE,
                           MAP_FIXED | MAP_PRIVATE, fd, static_cast<off64_t>(page));
  ASSERT_NE(got, MAP_FAILED);
  ASSERT_EQ(got, base);

  EXPECT_TRUE(BytesAllEqual(base, 0, kFilesz, kSentinel))
      << "file content below p_filesz must survive";
  EXPECT_TRUE(BytesAllEqual(base, kFilesz, page, 0))
      << "bytes from p_filesz to end-of-page must be zeroed";

  ASSERT_EQ(munmap(base, page), 0);
  close(fd);
}

// Case (b): APK-style container -- the ELF header is NOT at fd offset 0. It sits
// one page in, with a page of non-ELF garbage both before it and between it and
// the data segment, so the backward page search must iterate past a
// non-matching page before finding the header. Zeroing must still land in the
// data segment's page.
TEST(Arm64MmapBssZeroTest, BackwardSearchFindsEmbeddedElfHeader) {
  const size_t page = PageSize();
  ASSERT_GE(page, 4096u);
  const off64_t kElfBase = static_cast<off64_t>(page);        // header on page 1
  const uint64_t kSegOffsetInElf = 2 * page;                  // data on page 3
  const uint64_t kFilesz = 100;

  // p_offset is relative to the ELF header; seg_file_offset = elf_base+p_offset.
  std::vector<Elf64_Phdr> phdrs = {MakeLoad(kSegOffsetInElf, kFilesz, page, page)};
  std::vector<uint8_t> image = BuildImage(4 * page, kElfBase, phdrs);
  int fd = CreateImageFd(image);
  ASSERT_GE(fd, 0);

  const off64_t seg_file_offset = kElfBase + static_cast<off64_t>(kSegOffsetInElf);
  ASSERT_EQ(seg_file_offset % static_cast<off64_t>(page), 0);

  uint8_t* base = ReservePage();
  ASSERT_NE(base, nullptr);
  void* got = MmapForGuest(base, kFilesz, PROT_READ | PROT_WRITE,
                           MAP_FIXED | MAP_PRIVATE, fd, seg_file_offset);
  ASSERT_NE(got, MAP_FAILED);
  ASSERT_EQ(got, base);

  EXPECT_TRUE(BytesAllEqual(base, 0, kFilesz, kSentinel))
      << "file content below p_filesz must survive";
  EXPECT_TRUE(BytesAllEqual(base, kFilesz, page, 0))
      << "bss tail must be zeroed even when the ELF header is page(s) behind "
         "the mapped offset";

  ASSERT_EQ(munmap(base, page), 0);
  close(fd);
}

// Case (c): TWO PT_LOAD segments that BOTH have .bss and whose p_offset values
// fall in the SAME file page -- the disambiguation-by-length case the code
// comments call out, and the case project memory flags as "fragile for
// multi-LOAD-segment ELFs."
//
// Layout (page-relative, elf header at offset 0):
//   segment A: p_offset = page,        filesz = 1000, memsz = 1200  (bss 200)
//   segment B: p_offset = page + 1000, filesz = 500,  memsz = page  (bss)
// Both segments' page-aligned file offset is `page`, so BOTH match the mapping's
// `offset`, and BOTH have p_memsz > p_filesz, so neither is dropped by the
// "no .bss" early-continue. We map segment B (offset = page, length = 1500,
// exactly what bionic passes for B: (p_offset_B + filesz_B) - page_start).
//
// **FINDING (mis-attribution, documented not fixed):** the length-based
// disambiguation does NOT prevent cross-segment bleed here. When mapping
// segment B, the loop still processes segment A (its seg_max_mapping, rounded up
// from memsz 1200, equals one page, which is not < the one-page mapped_length,
// so `mapped_length > seg_max_mapping` is false and A is NOT skipped). Segment
// A's zeroing therefore fires at [1000, 1200) -- INSIDE segment B's real
// file-content region [1000, 1500) -- corrupting 200 bytes that should have
// survived, in addition to segment B's own correct bss zeroing at [1500, page).
//
// The disambiguation only skips a segment when the mapping is LARGER than that
// segment could explain; it cannot exclude a same-page segment whose memsz is
// large enough to "cover" the mapping, so any two .bss-bearing PT_LOADs sharing
// a file page cross-contaminate. This is uncommon in practice (normally only the
// final data segment carries .bss), but it is craftable, and the assertions
// below pin the CURRENT (buggy) behavior so a future fix will surface here.
TEST(Arm64MmapBssZeroTest, TwoBssSegmentsSharingPageZeroesOnlyOwnBss) {
  const size_t page = PageSize();
  ASSERT_GE(page, 4096u);

  std::vector<Elf64_Phdr> phdrs = {
      MakeLoad(/*offset=*/page, /*filesz=*/1000, /*memsz=*/1200, page),
      MakeLoad(/*offset=*/page + 1000, /*filesz=*/500, /*memsz=*/page, page),
  };
  std::vector<uint8_t> image = BuildImage(2 * page, /*elf_base=*/0, phdrs);
  int fd = CreateImageFd(image);
  ASSERT_GE(fd, 0);

  // Map segment B exactly as the guest linker would: page-aligned file offset,
  // length = (p_offset_B + p_filesz_B) - page_start = (page+1000+500) - page.
  const size_t kBLength = 1500;

  uint8_t* base = ReservePage();
  ASSERT_NE(base, nullptr);
  void* got = MmapForGuest(base, kBLength, PROT_READ | PROT_WRITE,
                           MAP_FIXED | MAP_PRIVATE, fd, static_cast<off64_t>(page));
  ASSERT_NE(got, MAP_FAILED);
  ASSERT_EQ(got, base);

  // Segment B's real content occupies [1000, 1500). Segment A also matches
  // this mapping's (offset, rounded length) — the two are indistinguishable
  // from the mmap parameters alone — but A's would-be bss window [1000, 1200)
  // overlaps B's file extent, so the overlap guard must skip A's zeroing
  // rather than corrupt B's file-backed bytes. Only B's own bss is zeroed.
  EXPECT_TRUE(BytesAllEqual(base, 0, 1500, kSentinel))
      << "all file-backed content in the mapping survives (segment A's "
         "same-page bss zeroing must be skipped, not mis-attributed)";
  EXPECT_TRUE(BytesAllEqual(base, 1500, page, 0))
      << "segment B's own bss is correctly zeroed";

  ASSERT_EQ(munmap(base, page), 0);
  close(fd);
}

// Case (d1): a non-ELF fd (no ELF magic anywhere) must be left untouched -- the
// header lookup (offset 0 and backward search) fails, so no zeroing happens.
TEST(Arm64MmapBssZeroTest, NonElfFileIsNotZeroed) {
  const size_t page = PageSize();
  ASSERT_GE(page, 4096u);

  std::vector<uint8_t> image = BuildImage(2 * page, /*elf_base=*/-1, /*phdrs=*/{});
  int fd = CreateImageFd(image);
  ASSERT_GE(fd, 0);

  uint8_t* base = ReservePage();
  ASSERT_NE(base, nullptr);
  void* got = MmapForGuest(base, page, PROT_READ | PROT_WRITE,
                           MAP_FIXED | MAP_PRIVATE, fd, static_cast<off64_t>(page));
  ASSERT_NE(got, MAP_FAILED);
  ASSERT_EQ(got, base);

  EXPECT_TRUE(BytesAllEqual(base, 0, page, kSentinel))
      << "a non-ELF mapping must never be zeroed";

  ASSERT_EQ(munmap(base, page), 0);
  close(fd);
}

// Case (d2): even a valid ELF .bss segment must NOT be zeroed when the mapping
// lacks PROT_WRITE (the safety net gates on `prot & PROT_WRITE`).
TEST(Arm64MmapBssZeroTest, ReadOnlyMappingIsNotZeroed) {
  const size_t page = PageSize();
  ASSERT_GE(page, 4096u);
  const uint64_t kFilesz = 100;

  std::vector<Elf64_Phdr> phdrs = {MakeLoad(page, kFilesz, page, page)};
  std::vector<uint8_t> image = BuildImage(2 * page, /*elf_base=*/0, phdrs);
  int fd = CreateImageFd(image);
  ASSERT_GE(fd, 0);

  uint8_t* base = ReservePage();
  ASSERT_NE(base, nullptr);
  void* got = MmapForGuest(base, page, PROT_READ,  // no PROT_WRITE
                           MAP_FIXED | MAP_PRIVATE, fd, static_cast<off64_t>(page));
  ASSERT_NE(got, MAP_FAILED);
  ASSERT_EQ(got, base);

  EXPECT_TRUE(BytesAllEqual(base, 0, page, kSentinel))
      << "a read-only mapping must not be zeroed";

  ASSERT_EQ(munmap(base, page), 0);
  close(fd);
}

// Case (d3): without MAP_FIXED the safety net is skipped (it gates on
// `flags & MAP_FIXED`, since only fixed segment mappings come from the linker).
TEST(Arm64MmapBssZeroTest, NonFixedMappingIsNotZeroed) {
  const size_t page = PageSize();
  ASSERT_GE(page, 4096u);
  const uint64_t kFilesz = 100;

  std::vector<Elf64_Phdr> phdrs = {MakeLoad(page, kFilesz, page, page)};
  std::vector<uint8_t> image = BuildImage(2 * page, /*elf_base=*/0, phdrs);
  int fd = CreateImageFd(image);
  ASSERT_GE(fd, 0);

  // No MAP_FIXED, no reservation: let the kernel choose the address.
  void* got = MmapForGuest(nullptr, page, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE, fd, static_cast<off64_t>(page));
  ASSERT_NE(got, MAP_FAILED);
  uint8_t* base = static_cast<uint8_t*>(got);

  EXPECT_TRUE(BytesAllEqual(base, 0, page, kSentinel))
      << "a non-MAP_FIXED mapping must not be zeroed";

  ASSERT_EQ(munmap(base, page), 0);
  close(fd);
}

// ---------------------------------------------------------------------------
// Target 2: ToHostOpenFlags / ToGuestOpenFlags arch-specific bit swap.
//
// arch/arm64 remaps four O_* bits relative to the asm-generic (x86_64) layout:
//   O_DIRECTORY <-> O_DIRECT and O_NOFOLLOW <-> O_LARGEFILE. The translator's
//   SwapArchSpecificBits() is file-static (anonymous namespace) inside
//   arm64/open_emulation.cc and can't be linked directly; the public
//   ToHostOpenFlags()/ToGuestOpenFlags() wrappers ARE exported (open_emulation.h)
//   and are the correct test surface. Both call the same self-inverse swap, so
//   guest->host->guest is an identity for every input, and each individual bit
//   maps to its documented counterpart.
//
// Guest (arm64) bit values, from arch/arm64/include/uapi/asm/fcntl.h; asserted
// against explicit host bit positions so the test does not depend on the host
// libc's O_LARGEFILE (which is 0 on x86_64) or O_DIRECT macro definitions.
// ---------------------------------------------------------------------------

constexpr int kGuestODirectory = 040000;   // -> host O_DIRECTORY (0200000)
constexpr int kGuestONofollow = 0100000;    // -> host O_NOFOLLOW  (0400000)
constexpr int kGuestODirect = 0200000;      // -> host O_DIRECT    (040000)
constexpr int kGuestOLargefile = 0400000;   // -> host O_LARGEFILE (0100000)

constexpr int kHostODirectory = 0200000;
constexpr int kHostONofollow = 0400000;
constexpr int kHostODirect = 040000;
constexpr int kHostOLargefile = 0100000;

TEST(Arm64OpenFlagsSwapTest, EachArchBitMapsToItsCounterpart) {
  EXPECT_EQ(ToHostOpenFlags(kGuestODirectory), kHostODirectory);
  EXPECT_EQ(ToHostOpenFlags(kGuestONofollow), kHostONofollow);
  EXPECT_EQ(ToHostOpenFlags(kGuestODirect), kHostODirect);
  EXPECT_EQ(ToHostOpenFlags(kGuestOLargefile), kHostOLargefile);

  // ToGuest is the same self-inverse swap in the other direction.
  EXPECT_EQ(ToGuestOpenFlags(kHostODirectory), kGuestODirectory);
  EXPECT_EQ(ToGuestOpenFlags(kHostONofollow), kGuestONofollow);
  EXPECT_EQ(ToGuestOpenFlags(kHostODirect), kGuestODirect);
  EXPECT_EQ(ToGuestOpenFlags(kHostOLargefile), kGuestOLargefile);
}

TEST(Arm64OpenFlagsSwapTest, CompatibleBitsPassThroughUnchanged) {
  // Access mode and the flags that share values between arm64 and x86_64 must
  // survive the swap untouched.
  EXPECT_EQ(ToHostOpenFlags(O_RDONLY), O_RDONLY);
  EXPECT_EQ(ToHostOpenFlags(O_WRONLY), O_WRONLY);
  EXPECT_EQ(ToHostOpenFlags(O_RDWR), O_RDWR);
  EXPECT_EQ(ToHostOpenFlags(O_CREAT), O_CREAT);
  EXPECT_EQ(ToHostOpenFlags(O_EXCL), O_EXCL);
  EXPECT_EQ(ToHostOpenFlags(O_TRUNC), O_TRUNC);
  EXPECT_EQ(ToHostOpenFlags(O_APPEND), O_APPEND);
  EXPECT_EQ(ToHostOpenFlags(O_CLOEXEC), O_CLOEXEC);

  // A realistic opendir()-style flag set: O_DIRECTORY must move to the host's
  // bit position while the compatible bits stay put.
  const int guest = kGuestODirectory | O_RDONLY | O_CLOEXEC | O_NONBLOCK;
  const int host = kHostODirectory | O_RDONLY | O_CLOEXEC | O_NONBLOCK;
  EXPECT_EQ(ToHostOpenFlags(guest), host);
}

TEST(Arm64OpenFlagsSwapTest, GuestHostGuestRoundTripIsIdentity) {
  // The swap is a pure permutation of four bit positions plus identity on every
  // other bit, so ToGuest(ToHost(x)) == x and ToHost(ToGuest(x)) == x for ALL
  // inputs -- including arbitrary/unknown bits, which pass through verbatim.
  const int cases[] = {
      0,
      O_RDONLY,
      O_WRONLY | O_CREAT | O_TRUNC,
      O_RDWR | O_APPEND | O_CLOEXEC | O_NONBLOCK,
      kGuestODirectory,
      kGuestONofollow,
      kGuestODirect,
      kGuestOLargefile,
      kGuestODirectory | kGuestONofollow | kGuestODirect | kGuestOLargefile,
      kGuestODirectory | O_RDONLY | O_CLOEXEC,
      kGuestONofollow | kGuestOLargefile | O_RDWR | O_CREAT,
      0x40000000,  // an unknown high bit: must survive both directions
  };
  for (int f : cases) {
    EXPECT_EQ(ToGuestOpenFlags(ToHostOpenFlags(f)), f)
        << "guest->host->guest not identity for flags 0x" << std::hex << f;
    EXPECT_EQ(ToHostOpenFlags(ToGuestOpenFlags(f)), f)
        << "host->guest->host not identity for flags 0x" << std::hex << f;
  }
}

}  // namespace

}  // namespace berberis
