#include "guest_load_error.h"

#include <string>

#include "gtest/gtest.h"

namespace berberis {
namespace {

TEST(GuestLoadError, SavedGuestErrorSurvivesUntilTaken) {
  auto error = GuestLoadError();
  auto guest = std::string("dlopen failed: library \"libdep.so\" needed by \"libapp.so\"");
  EXPECT_STREQ(guest.c_str(), error.Save(guest.c_str()));
  // The guest loader reuses its buffer; the saved copy must not follow it.
  guest.assign("later guest error");

  EXPECT_STREQ("dlopen failed: library \"libdep.so\" needed by \"libapp.so\"", error.Take());
  EXPECT_EQ(nullptr, error.Take());
}

TEST(GuestLoadError, NullErrorClearsAnEarlierOne) {
  auto error = GuestLoadError();
  error.Save("first");
  EXPECT_EQ(nullptr, error.Save(nullptr));
  EXPECT_EQ(nullptr, error.Take());
}

TEST(GuestLoadError, ClearForgetsTheSavedError) {
  auto error = GuestLoadError();
  error.Save("first");
  error.Clear();
  EXPECT_EQ(nullptr, error.Take());
}

TEST(GuestLoadError, TakenTextStaysValidUntilTheNextTake) {
  auto error = GuestLoadError();
  error.Save("first");
  const auto* first = error.Take();
  error.Save("second");
  EXPECT_STREQ("first", first);
  EXPECT_STREQ("second", error.Take());
}

}  // namespace
}  // namespace berberis
