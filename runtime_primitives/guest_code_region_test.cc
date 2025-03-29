/*
 * Copyright (C) 2025 The Android Open Source Project
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "berberis/runtime_primitives/guest_code_region.h"

namespace berberis {

namespace {

using testing::ElementsAre;

TEST(GuestCodeRegion, Smoke) {
  Arena arena;
  GuestCodeRegion region(&arena);

  EXPECT_TRUE(region.branch_targets().empty());

  {
    // 42 - 50 ->{8, 100}
    auto* bb = region.NewBasicBlock(42, 8, ArenaVector<GuestAddr>({8, 100}, &arena));
    EXPECT_EQ(bb->start_addr(), 42u);
    EXPECT_EQ(bb->size(), 8u);
    EXPECT_EQ(bb->end_addr(), 50u);
    EXPECT_THAT(bb->out_edges(), ElementsAre(8, 100));
    EXPECT_TRUE(bb->in_edges().empty());
  }

  EXPECT_THAT(region.branch_targets(), ElementsAre(8, 100));

  {
    // 56 - 60 -> {42, 120}
    auto* bb = region.NewBasicBlock(56, 4, ArenaVector<GuestAddr>({42, 50}, &arena));
    EXPECT_EQ(bb->start_addr(), 56u);
    EXPECT_EQ(bb->size(), 4u);
    EXPECT_EQ(bb->end_addr(), 60u);
    EXPECT_THAT(bb->out_edges(), ElementsAre(42, 50));
    EXPECT_TRUE(bb->in_edges().empty());
  }

  EXPECT_THAT(region.branch_targets(), ElementsAre(8, 42, 50, 100));

  region.ResolveEdges();

  auto& basic_blocks = region.basic_blocks();

  EXPECT_EQ(basic_blocks.size(), 2u);

  ASSERT_TRUE(basic_blocks.contains(42));
  ASSERT_TRUE(basic_blocks.contains(56));

  {
    auto& bb = basic_blocks.at(42);
    EXPECT_THAT(bb.in_edges(), ElementsAre(56));
  }

  {
    auto& bb = basic_blocks.at(56);
    EXPECT_TRUE(bb.in_edges().empty());
  }
}

TEST(GuestCodeRegion, ResolveEdges) {
  Arena arena;
  GuestCodeRegion region(&arena);

  // 42 - 54
  region.NewBasicBlock(42, 12, ArenaVector<GuestAddr>({100, 150, 200}, &arena));

  EXPECT_THAT(region.branch_targets(), ElementsAre(100, 150, 200));

  // 100 - 120
  region.NewBasicBlock(100, 20, ArenaVector<GuestAddr>({8, 200, 1000}, &arena));

  EXPECT_THAT(region.branch_targets(), ElementsAre(8, 100, 150, 200, 1000));

  // 200 - 240
  region.NewBasicBlock(200, 40, ArenaVector<GuestAddr>({80, 120}, &arena));

  EXPECT_THAT(region.branch_targets(), ElementsAre(8, 80, 100, 120, 150, 200, 1000));

  region.ResolveEdges();

  auto& basic_blocks = region.basic_blocks();
  ASSERT_EQ(basic_blocks.size(), 3u);
  ASSERT_TRUE(basic_blocks.contains(42));
  ASSERT_TRUE(basic_blocks.contains(100));
  ASSERT_TRUE(basic_blocks.contains(200));

  {
    auto bb = basic_blocks.at(42);
    EXPECT_TRUE(bb.in_edges().empty());
  }

  {
    auto bb = basic_blocks.at(100);
    EXPECT_THAT(bb.in_edges(), ElementsAre(42));
  }

  {
    auto bb = basic_blocks.at(200);
    EXPECT_THAT(bb.in_edges(), ElementsAre(42, 100));
  }
}

TEST(GuestCodeRegion, SplitBasicBlock) {
  Arena arena;
  GuestCodeRegion region(&arena);

  // 42 - 54
  region.NewBasicBlock(42, 12, ArenaVector<GuestAddr>({110, 150, 220}, &arena));

  EXPECT_THAT(region.branch_targets(), ElementsAre(110, 150, 220));

  // 100 - 120
  region.NewBasicBlock(100, 20, ArenaVector<GuestAddr>({8, 50, 1000}, &arena));

  EXPECT_THAT(region.branch_targets(), ElementsAre(8, 50, 110, 150, 220, 1000));

  // 200 - 240
  region.NewBasicBlock(200, 40, ArenaVector<GuestAddr>({80, 120, 240}, &arena));

  EXPECT_THAT(region.branch_targets(), ElementsAre(8, 50, 80, 110, 120, 150, 220, 240, 1000));

  // 240 - 250
  region.NewBasicBlock(240, 50, ArenaVector<GuestAddr>({10, 210, 230}, &arena));

  EXPECT_THAT(region.branch_targets(),
              ElementsAre(8, 10, 50, 80, 110, 120, 150, 210, 220, 230, 240, 1000));

  region.ResolveEdges();

  auto& basic_blocks = region.basic_blocks();
  ASSERT_EQ(basic_blocks.size(), 9u);
  ASSERT_TRUE(basic_blocks.contains(42));
  ASSERT_TRUE(basic_blocks.contains(50));
  ASSERT_TRUE(basic_blocks.contains(100));
  ASSERT_TRUE(basic_blocks.contains(110));
  ASSERT_TRUE(basic_blocks.contains(200));
  ASSERT_TRUE(basic_blocks.contains(210));
  ASSERT_TRUE(basic_blocks.contains(220));
  ASSERT_TRUE(basic_blocks.contains(230));
  ASSERT_TRUE(basic_blocks.contains(240));

  {
    auto bb = basic_blocks.at(42);
    EXPECT_EQ(bb.start_addr(), 42u);
    EXPECT_EQ(bb.size(), 8u);
    EXPECT_EQ(bb.end_addr(), 50u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(50));
    EXPECT_TRUE(bb.in_edges().empty());
  }

  {
    auto bb = basic_blocks.at(50);
    EXPECT_EQ(bb.start_addr(), 50u);
    EXPECT_EQ(bb.size(), 4u);
    EXPECT_EQ(bb.end_addr(), 54u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(110, 150, 220));
    EXPECT_THAT(bb.in_edges(), ElementsAre(42, 110));
  }

  {
    auto bb = basic_blocks.at(100);
    EXPECT_EQ(bb.start_addr(), 100u);
    EXPECT_EQ(bb.size(), 10u);
    EXPECT_EQ(bb.end_addr(), 110u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(110));
    EXPECT_TRUE(bb.in_edges().empty());
  }

  {
    auto bb = basic_blocks.at(110);
    EXPECT_EQ(bb.start_addr(), 110u);
    EXPECT_EQ(bb.size(), 10u);
    EXPECT_EQ(bb.end_addr(), 120u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(8, 50, 1000));
    EXPECT_THAT(bb.in_edges(), ElementsAre(50, 100));
  }

  {
    auto bb = basic_blocks.at(200);
    EXPECT_EQ(bb.start_addr(), 200u);
    EXPECT_EQ(bb.size(), 10u);
    EXPECT_EQ(bb.end_addr(), 210u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(210));
    EXPECT_TRUE(bb.in_edges().empty());
  }

  {
    auto bb = basic_blocks.at(210);
    EXPECT_EQ(bb.start_addr(), 210u);
    EXPECT_EQ(bb.size(), 10u);
    EXPECT_EQ(bb.end_addr(), 220u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(220));
    EXPECT_THAT(bb.in_edges(), ElementsAre(200, 240));
  }

  {
    auto bb = basic_blocks.at(220);
    EXPECT_EQ(bb.start_addr(), 220u);
    EXPECT_EQ(bb.size(), 10u);
    EXPECT_EQ(bb.end_addr(), 230u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(230));
    EXPECT_THAT(bb.in_edges(), ElementsAre(50, 210));
  }

  {
    auto bb = basic_blocks.at(230);
    EXPECT_EQ(bb.start_addr(), 230u);
    EXPECT_EQ(bb.size(), 10u);
    EXPECT_EQ(bb.end_addr(), 240u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(80, 120, 240));
    EXPECT_THAT(bb.in_edges(), ElementsAre(220, 240));
  }

  {
    auto bb = basic_blocks.at(240);
    EXPECT_EQ(bb.start_addr(), 240u);
    EXPECT_EQ(bb.size(), 50u);
    EXPECT_EQ(bb.end_addr(), 290u);
    EXPECT_THAT(bb.out_edges(), ElementsAre(10, 210, 230));
    EXPECT_THAT(bb.in_edges(), ElementsAre(230));
  }
}

TEST(GuestCodeRegion, InvalidRegion) {
  Arena arena;
  GuestCodeRegion region(&arena);

  // Overlapping code blocks are not allowed
  region.NewBasicBlock(100, 60, ArenaVector<GuestAddr>({}, &arena));
  region.NewBasicBlock(150, 50, ArenaVector<GuestAddr>({}, &arena));

  EXPECT_DEATH(region.ResolveEdges(), "");
}

TEST(GuestCodeRegion, NoResolveEdgesTwice) {
  Arena arena;
  GuestCodeRegion region(&arena);

  region.NewBasicBlock(100, 60, ArenaVector<GuestAddr>({}, &arena));

  region.ResolveEdges();

  EXPECT_DEATH(region.ResolveEdges(), "");
}

TEST(GuestCodeRegion, ResolveEdgesExpectsNoInEdges) {
  Arena arena;
  GuestCodeRegion region(&arena);

  auto* bb = region.NewBasicBlock(100, 60, ArenaVector<GuestAddr>({}, &arena));
  bb->AddInEdge(5);

  EXPECT_DEATH(region.ResolveEdges(), "");
}

TEST(GuestCodeRegion, NoNewBasicBlockAfterResolveRegion) {
  Arena arena;
  GuestCodeRegion region(&arena);

  region.NewBasicBlock(100, 60, ArenaVector<GuestAddr>({}, &arena));

  region.ResolveEdges();

  EXPECT_DEATH(region.NewBasicBlock(200, 20, ArenaVector<GuestAddr>({}, &arena)), "");
}

}  // namespace

}  // namespace berberis
