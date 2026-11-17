#ifndef BERBERIS_NATIVE_BRIDGE_GUEST_LOAD_ERROR_H_
#define BERBERIS_NATIVE_BRIDGE_GUEST_LOAD_ERROR_H_

#include <string>

namespace berberis {

// Keeps the guest linker's error from a failed library load until the
// caller asks for it. The guest loader's dlerror() is one-shot, and the
// host-loader fallback that follows a failed guest load sets an error of its
// own, so reading either one directly reports the fallback, not the guest
// cause (a missing dependency or an unresolved symbol). Like dlerror(), the
// saved error is returned once.
class GuestLoadError {
 public:
  // Forgets an earlier load's error.
  void Clear() { saved_ = false; }

  // Saves `error` (a null error clears) and returns the saved text, or null.
  const char* Save(const char* error) {
    saved_ = error != nullptr;
    if (!saved_) {
      return nullptr;
    }
    pending_ = error;
    return pending_.c_str();
  }

  // Returns the saved error once, then null until the next Save().
  const char* Take() {
    if (!saved_) {
      return nullptr;
    }
    saved_ = false;
    returned_.swap(pending_);
    return returned_.c_str();
  }

 private:
  bool saved_ = false;
  std::string pending_;
  // Owns the text Take() returned until the next Take().
  std::string returned_;
};

}  // namespace berberis

#endif  // BERBERIS_NATIVE_BRIDGE_GUEST_LOAD_ERROR_H_
