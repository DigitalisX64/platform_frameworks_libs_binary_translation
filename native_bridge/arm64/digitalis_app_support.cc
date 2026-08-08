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

#include "arm64/digitalis_app_support.h"

#include <android/log.h>
#define DIGITALIS_LOG(...) __android_log_print(ANDROID_LOG_DEBUG, "berberis", __VA_ARGS__)

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "ziparchive/zip_archive.h"

#include "berberis/base/config_globals.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

// Extract <apk_path>!/<entry> to /data/data/<pkg>/cache/berberis_extract/<basename>
// and return the extracted path on success. Used as a fallback when the guest
// dynamic linker fails to dlopen an in-APK library path — happens with Qt 6
// apps (e.g. VulkanCapsViewer) whose deployment APKs ship Qt libs in the
// `base.apk!/lib/arm64-v8a/` form and whose JVM-side QtLoader hits a
// `dlopen failed: library not found` from inside an instrumented `!/` path
// after libziparchive logs a META-INF/...properties duplicate-entry warning.
//
// The extracted file is dropped in the app's own cache (per-uid, persists
// across launches, cleaned up by the OS), so the next launch reuses it.
// Returns "" if anything fails — caller falls through to its existing
// host-loader path.
//
// arm64-guest only: relies on libziparchive, which is not linked into the
// riscv64 native bridge. The call site is guarded by NATIVE_BRIDGE_GUEST_ARCH_ARM64.
std::string ExtractInApkLibToCache(const char* libpath) {
  const char* bang = strstr(libpath, "!/");
  if (bang == nullptr) {
    return {};
  }
  std::string apk_path(libpath, static_cast<size_t>(bang - libpath));
  std::string entry_name(bang + 2);  // skip "!/"
  const char* basename = strrchr(entry_name.c_str(), '/');
  basename = basename ? basename + 1 : entry_name.c_str();

  const char* private_dir = berberis::GetAppPrivateDir();
  if (private_dir == nullptr || private_dir[0] == '\0') {
    DIGITALIS_LOG("ExtractInApkLibToCache: no app private dir; can't cache %s", libpath);
    return {};
  }
  std::string cache_dir = std::string(private_dir) + "/cache/berberis_extract";
  // Best-effort mkdir; ignore EEXIST.
  mkdir((std::string(private_dir) + "/cache").c_str(), 0700);
  mkdir(cache_dir.c_str(), 0700);

  std::string out_path = cache_dir + "/" + basename;

  // Reuse a previously extracted copy only if it is still the copy this APK
  // contains. The cache lives in the app's data directory, which survives an
  // app update, while the APK is rewritten by one -- so a cache keyed on the
  // library name alone would keep serving the previous version's code to the
  // new version's Java, which surfaces as an UnsatisfiedLinkError for a
  // newly added JNI method, or worse as old native code running silently
  // against new callers.
  //
  // An update always writes a newer APK than any extract taken from the
  // previous one, so requiring the cached copy to be newer than the APK is
  // enough to catch it, and costs one stat rather than opening the archive.
  struct stat st_out;
  struct stat st_apk;
  if (stat(out_path.c_str(), &st_out) == 0 && st_out.st_size > 0) {
    if (stat(apk_path.c_str(), &st_apk) != 0 || st_out.st_mtime > st_apk.st_mtime) {
      return out_path;
    }
    DIGITALIS_LOG("ExtractInApkLibToCache: %s is stale (apk is newer); re-extracting",
                  out_path.c_str());
  }

  ZipArchiveHandle zip = nullptr;
  if (OpenArchive(apk_path.c_str(), &zip) != 0) {
    DIGITALIS_LOG("ExtractInApkLibToCache: OpenArchive(%s) failed", apk_path.c_str());
    if (zip) CloseArchive(zip);
    return {};
  }
  ZipEntry entry;
  if (FindEntry(zip, entry_name, &entry) != 0) {
    DIGITALIS_LOG("ExtractInApkLibToCache: entry %s not found in %s",
                  entry_name.c_str(), apk_path.c_str());
    CloseArchive(zip);
    return {};
  }
  // Extract to a temp file and rename so partial reads from concurrent
  // launches don't see a half-written .so.
  std::string tmp_path = out_path + ".tmp";
  int out_fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (out_fd < 0) {
    DIGITALIS_LOG("ExtractInApkLibToCache: open(%s) failed: %s", tmp_path.c_str(), strerror(errno));
    CloseArchive(zip);
    return {};
  }
  int32_t rc = ExtractEntryToFile(zip, &entry, out_fd);
  close(out_fd);
  CloseArchive(zip);
  if (rc != 0) {
    unlink(tmp_path.c_str());
    DIGITALIS_LOG("ExtractInApkLibToCache: ExtractEntryToFile rc=%d for %s", rc, entry_name.c_str());
    return {};
  }
  if (rename(tmp_path.c_str(), out_path.c_str()) != 0) {
    unlink(tmp_path.c_str());
    DIGITALIS_LOG("ExtractInApkLibToCache: rename(%s,%s) failed: %s",
                  tmp_path.c_str(), out_path.c_str(), strerror(errno));
    return {};
  }
  DIGITALIS_LOG("ExtractInApkLibToCache: %s -> %s", libpath, out_path.c_str());
  return out_path;
}

// Inject QT_PLUGIN_PATH / QT_QPA_PLATFORM_PLUGIN_PATH into guest libc's
// environ for Qt-for-Android prebuilt apps. Berberis extracts arm64-v8a .so
// files to <app>/cache/berberis_extract/ instead of populating Android's
// standard nativeLibraryDir (/data/app/.../lib/arm64/). Qt-for-Android's
// libraryPaths-init relies on nativeLibraryDir being populated; with it
// empty, QGuiApplication's QPluginLoader fails with `Could not find the Qt
// platform plugin "android" in ""` SIGABRT. Pointing Qt's env-var-based
// plugin search at the extract dir gives it somewhere to look. We also
// create a `<extract>/<category>/lib<name>.so` symlink layout (Qt's
// QFactoryLoader expects this category/plugin path structure) for each
// `libplugins_<category>_<name>_arm64-v8a.so` in the extract dir.
//
// We patch guest libc's `environ` symbol directly (analogous to the
// __progname patch in FinalizeInit) because the host process's setenv()
// has no effect on guest libc: the guest has its own environ pointer
// initialized from the envp on the guest stack at zygote startup, before
// the app package is known.
//
// arm64-guest only: Qt-for-Android is an arm64 prebuilt-app scenario and this
// pairs with the libziparchive extract fallback above. The call site in
// FinalizeInit is guarded by NATIVE_BRIDGE_GUEST_ARCH_ARM64.
//
// `dlsym` resolves a symbol in the given guest libc handle; the caller passes
// NdktNativeBridge::DlSym (local to native_bridge.cc, not visible here).
void DigitalisInjectQtPluginPath(
    void* guest_libc,
    const std::function<GuestAddr(void* handle, const char* name)>& dlsym) {
  const char* private_dir = berberis::GetAppPrivateDir();
  if (private_dir == nullptr || private_dir[0] == '\0') {
    return;
  }

  char extract_dir[768];
  int len = snprintf(extract_dir, sizeof(extract_dir),
                     "%s/cache/berberis_extract", private_dir);
  if (len <= 0 || len >= static_cast<int>(sizeof(extract_dir))) {
    return;
  }
  // This runs at FinalizeInit, BEFORE any app library is extracted to the
  // cache. On a COLD cache the extract dir does not exist yet, so we must NOT
  // bail here: the QT_*_PLUGIN_PATH env injection below has to happen on the
  // first launch too. Without it, Qt's platform-plugin search falls back to
  // loading the plugin from its in-APK `base.apk!/lib/...` path — a second,
  // partially-initialised copy of the plugin whose Q_GLOBAL_STATIC mutexes are
  // null, which then SIGSEGVs on first use. Pointing QT_PLUGIN_PATH /
  // QT_QPA_PLATFORM_PLUGIN_PATH at the cache dir (where ExtractInApkLibToCache
  // drops the plugin moments later, by its native arm64-v8a name) makes Qt
  // find and load the good cache copy instead. Create the dir now so the env
  // var resolves; the symlink walk below finds nothing on a cold first launch
  // (it adds the Qt-friendly symlinks on later warm launches once the plugin
  // is in the cache) — QT_PLUGIN_PATH alone is enough to steer the first
  // launch to the extracted plugin.
  mkdir((std::string(private_dir) + "/cache").c_str(), 0700);
  mkdir(extract_dir, 0700);

  // Walk extract dir and add multiple Qt-friendly symlinks for every
  // libplugins_<cat>_<name>_<src-abi>.so found:
  //
  //   <extract>/<cat>/lib<name>.so  (Qt's standard category/plugin layout)
  //   <extract>/libplugins_<cat>_<name>_<host-abi>.so  (Qt-Android pattern;
  //       Qt-Android's QFactoryLoader checks libraryPath ROOT and looks for
  //       the plugin filename with the HOST abi suffix — see
  //       QT_DEBUG_PLUGINS trace — so an arm64-v8a-only APK extracted on an
  //       x86_64 host needs a libplugins_<cat>_<name>_x86_64.so alias)
  //   <extract>/lib<name>.so  (fallback standard Qt plugin name)
  //
  // Qt reads each .so's Q_PLUGIN_METADATA to pick the right one, so only
  // the directory placement and a Qt-recognised filename matter; the link
  // target's actual filename is irrelevant.
  DIR* dir = opendir(extract_dir);
  if (dir != nullptr) {
    static constexpr char kPrefix[] = "libplugins_";
    static constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
    static constexpr char kSrcSuffix[] = "_arm64-v8a.so";
    static constexpr size_t kSrcSuffixLen = sizeof(kSrcSuffix) - 1;
    // Host-side aliases Qt may look for. Berberis pretends to be arm64-v8a
    // to the guest, but Qt-on-Android may probe the host ABI via JNI; cover
    // both bases.
    static constexpr const char* kAliasAbiSuffixes[] = {
        "_x86_64.so", "_x86.so", "_armeabi-v7a.so"};
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
      const char* name = ent->d_name;
      size_t name_len = strlen(name);
      if (name_len <= kPrefixLen + kSrcSuffixLen) continue;
      if (memcmp(name, kPrefix, kPrefixLen) != 0) continue;
      if (memcmp(name + name_len - kSrcSuffixLen, kSrcSuffix, kSrcSuffixLen) != 0) continue;
      const char* cat_start = name + kPrefixLen;
      const char* cat_end = strchr(cat_start, '_');
      if (cat_end == nullptr || cat_end >= name + name_len - kSrcSuffixLen) continue;
      size_t cat_len = static_cast<size_t>(cat_end - cat_start);
      const char* plugin_start = cat_end + 1;
      size_t plugin_len =
          static_cast<size_t>((name + name_len - kSrcSuffixLen) - plugin_start);
      if (cat_len == 0 || plugin_len == 0) continue;

      // (1) <extract>/<cat>/lib<name>.so
      char cat_dir[896];
      snprintf(cat_dir, sizeof(cat_dir),
               "%s/%.*s", extract_dir, static_cast<int>(cat_len), cat_start);
      mkdir(cat_dir, 0700);
      char target[1024];
      snprintf(target, sizeof(target), "%s/lib%.*s.so",
               cat_dir, static_cast<int>(plugin_len), plugin_start);
      char nested_source[896];
      snprintf(nested_source, sizeof(nested_source), "../%s", name);
      unlink(target);
      if (symlink(nested_source, target) != 0 && errno != EEXIST) {
        DIGITALIS_LOG("InjectQtPluginPath: symlink(%s <- %s) failed: %s",
                      target, nested_source, strerror(errno));
      }

      // (2) <extract>/libplugins_<cat>_<name>_<alias-abi>.so
      for (const char* alias_suffix : kAliasAbiSuffixes) {
        char alias_target[1024];
        size_t base_len = name_len - kSrcSuffixLen;  // strip "_arm64-v8a.so"
        snprintf(alias_target, sizeof(alias_target), "%s/%.*s%s",
                 extract_dir, static_cast<int>(base_len), name, alias_suffix);
        unlink(alias_target);
        if (symlink(name, alias_target) != 0 && errno != EEXIST) {
          DIGITALIS_LOG("InjectQtPluginPath: symlink(%s <- %s) failed: %s",
                        alias_target, name, strerror(errno));
        }
      }

      // (3) <extract>/lib<name>.so (standard Qt plugin name)
      char std_target[1024];
      snprintf(std_target, sizeof(std_target), "%s/lib%.*s.so",
               extract_dir, static_cast<int>(plugin_len), plugin_start);
      unlink(std_target);
      if (symlink(name, std_target) != 0 && errno != EEXIST) {
        DIGITALIS_LOG("InjectQtPluginPath: symlink(%s <- %s) failed: %s",
                      std_target, name, strerror(errno));
      }
    }
    closedir(dir);
  }

  // Patch guest libc's environ to include QT_PLUGIN_PATH /
  // QT_QPA_PLATFORM_PLUGIN_PATH. Allocate a new env array in host memory
  // (host/guest share address space, so guest libc can read it directly);
  // copy the existing env strings; append our two entries; atomically swap
  // the guest's environ pointer.
  berberis::GuestAddr environ_addr = dlsym(guest_libc, "environ");
  if (environ_addr == berberis::kNullGuestAddr) {
    DIGITALIS_LOG("InjectQtPluginPath: guest libc 'environ' symbol not found");
    return;
  }

  // The symbol environ is `char** environ;`, so the symbol address is a
  // char***. ToHostAddr<char**>(addr) returns char*** as desired.
  char*** environ_ptr = berberis::ToHostAddr<char**>(environ_addr);
  char** old_environ = *environ_ptr;
  size_t old_count = 0;
  if (old_environ != nullptr) {
    while (old_environ[old_count] != nullptr) ++old_count;
  }

  // Build the QT_* entries; static storage so they live for the process'
  // lifetime (guest libc keeps pointers into our array indefinitely).
  static char qt_plugin_path_buf[1024];
  static char qt_qpa_buf[1024];
  snprintf(qt_plugin_path_buf, sizeof(qt_plugin_path_buf),
           "QT_PLUGIN_PATH=%s", extract_dir);
  snprintf(qt_qpa_buf, sizeof(qt_qpa_buf),
           "QT_QPA_PLATFORM_PLUGIN_PATH=%s", extract_dir);

  // Skip already-present entries (e.g. user / app set their own).
  bool has_plugin = false;
  bool has_qpa = false;
  for (size_t i = 0; i < old_count; ++i) {
    if (strncmp(old_environ[i], "QT_PLUGIN_PATH=", 15) == 0) has_plugin = true;
    if (strncmp(old_environ[i], "QT_QPA_PLATFORM_PLUGIN_PATH=", 28) == 0) has_qpa = true;
  }
  if (has_plugin && has_qpa) {
    return;
  }

  size_t extra = static_cast<size_t>(!has_plugin) + static_cast<size_t>(!has_qpa);
  char** new_environ =
      static_cast<char**>(calloc(old_count + extra + 1, sizeof(char*)));
  if (new_environ == nullptr) {
    return;
  }
  size_t idx = 0;
  for (size_t i = 0; i < old_count; ++i) {
    new_environ[idx++] = old_environ[i];
  }
  if (!has_plugin) new_environ[idx++] = qt_plugin_path_buf;
  if (!has_qpa) new_environ[idx++] = qt_qpa_buf;
  new_environ[idx] = nullptr;

  *environ_ptr = new_environ;
  DIGITALIS_LOG(
      "InjectQtPluginPath: patched guest environ (old_count=%zu, added=%zu) "
      "QT_PLUGIN_PATH=%s",
      old_count, extra, extract_dir);
}

}  // namespace berberis
