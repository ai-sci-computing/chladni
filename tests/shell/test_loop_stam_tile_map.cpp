/**
 * @file test_loop_stam_tile_map.cpp
 * @brief S.2 — tile-mapping routine for Stam's parameter-space evaluator.
 *
 * Pins @ref chladni::shell::loop::stam_tile_map for representative
 * (v, w) samples in each tile, the level-boundary and tile-boundary
 * edge cases, and the round-trip identity @f$ t_{n,k}^{-1}(v_p, w_p) =
 * (v, w) @f$ on a grid of interior points.
 *
 * Reference: @cite stam_1999_loop_evaluation Sec 3 (tile partition,
 * Fig. 5) and Sec 4 (`EvalSurf` algorithm).
 */

#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

/// Inverse of the affine tile transform t_{n,k}: given (n, k, v_p, w_p),
/// reconstruct the original (v, w) in the unit triangle.
std::pair<double, double>
inverse_tile_transform(int n, int k, double v_p, double w_p)
{
    const double pow2 = std::pow(2.0, n);  // 2^n (NOT 2^(n-1))
    switch (k) {
        case 1: return { (v_p + 1.0) / pow2,        w_p          / pow2 };
        case 2: return { (1.0 - v_p)  / pow2,       (1.0 - w_p)  / pow2 };
        case 3: return {  v_p         / pow2,       (w_p + 1.0)  / pow2 };
    }
    // Unreachable: stam_tile_map only returns k in {1, 2, 3}.
    throw std::logic_error("inverse_tile_transform: bad k");
}

constexpr double kTol = 1.0e-12;

}  // namespace

TEST_CASE("stam_tile_map: pinned samples at n=1 (v+w in (1/2, 1])",
          "[shell][loop][stam][tile_map]")
{
    using chladni::shell::loop::stam_tile_map;

    // Tile k=1 (v > 1/2 after rescale): near the v=1 corner.
    {
        const auto tm = stam_tile_map(0.7, 0.1);
        CAPTURE(tm.n, tm.k, tm.v_p, tm.w_p);
        REQUIRE(tm.n == 1);
        REQUIRE(tm.k == 1);
        REQUIRE(std::abs(tm.v_p - 0.4) < kTol);
        REQUIRE(std::abs(tm.w_p - 0.2) < kTol);
    }
    // Tile k=3 (w > 1/2 after rescale): near the w=1 corner.
    {
        const auto tm = stam_tile_map(0.1, 0.7);
        REQUIRE(tm.n == 1);
        REQUIRE(tm.k == 3);
        REQUIRE(std::abs(tm.v_p - 0.2) < kTol);
        REQUIRE(std::abs(tm.w_p - 0.4) < kTol);
    }
    // Tile k=2 (middle, flipped): neither v nor w > 1/2 after rescale.
    {
        const auto tm = stam_tile_map(0.3, 0.3);
        REQUIRE(tm.n == 1);
        REQUIRE(tm.k == 2);
        REQUIRE(std::abs(tm.v_p - 0.4) < kTol);
        REQUIRE(std::abs(tm.w_p - 0.4) < kTol);
    }
}

TEST_CASE("stam_tile_map: pinned samples at deeper levels",
          "[shell][loop][stam][tile_map]")
{
    using chladni::shell::loop::stam_tile_map;

    // v + w = 0.2 -> n = floor(1 - log2(0.2)) = floor(3.32) = 3.
    {
        const auto tm = stam_tile_map(0.1, 0.1);
        CAPTURE(tm.n, tm.k, tm.v_p, tm.w_p);
        REQUIRE(tm.n == 3);
        REQUIRE(tm.k == 2);  // 4*0.1 = 0.4 < 0.5 for both -> middle
        REQUIRE(std::abs(tm.v_p - 0.2) < kTol);
        REQUIRE(std::abs(tm.w_p - 0.2) < kTol);
    }
    // v + w = 0.5 exactly -> log2(0.5) = -1 -> n = 2.
    // v_s = 2*0.49 = 0.98 -> k=1.
    {
        const auto tm = stam_tile_map(0.49, 0.01);
        CAPTURE(tm.n, tm.k, tm.v_p, tm.w_p);
        REQUIRE(tm.n == 2);
        REQUIRE(tm.k == 1);
        REQUIRE(std::abs(tm.v_p - 0.96) < kTol);
        REQUIRE(std::abs(tm.w_p - 0.04) < kTol);
    }
}

TEST_CASE("stam_tile_map: level boundary v+w = 1/2",
          "[shell][loop][stam][tile_map][boundary]")
{
    using chladni::shell::loop::stam_tile_map;

    // Just above 1/2 (still in n=1 band).
    {
        const auto tm = stam_tile_map(0.30, 0.21);  // sum = 0.51
        REQUIRE(tm.n == 1);
    }
    // Just below 1/2 (drops to n=2).
    {
        const auto tm = stam_tile_map(0.30, 0.19);  // sum = 0.49
        REQUIRE(tm.n == 2);
    }
    // At 1/2 exactly -> n=2 (log2(0.5) = -1, 1 - (-1) = 2, floor = 2).
    {
        const auto tm = stam_tile_map(0.25, 0.25);  // sum = 0.50
        REQUIRE(tm.n == 2);
    }
}

TEST_CASE("stam_tile_map: corners",
          "[shell][loop][stam][tile_map][boundary]")
{
    using chladni::shell::loop::stam_tile_map;

    // (v=1, w=0): the v=1 corner of Omega -> tile 1, unit-triangle (1, 0).
    {
        const auto tm = stam_tile_map(1.0, 0.0);
        REQUIRE(tm.n == 1);
        REQUIRE(tm.k == 1);
        REQUIRE(std::abs(tm.v_p - 1.0) < kTol);
        REQUIRE(std::abs(tm.w_p - 0.0) < kTol);
    }
    // (v=0, w=1): the w=1 corner of Omega -> tile 3, unit-triangle (0, 1).
    {
        const auto tm = stam_tile_map(0.0, 1.0);
        REQUIRE(tm.n == 1);
        REQUIRE(tm.k == 3);
        REQUIRE(std::abs(tm.v_p - 0.0) < kTol);
        REQUIRE(std::abs(tm.w_p - 1.0) < kTol);
    }
    // (v=0.5, w=0.5): on the hypotenuse, opposite the extraordinary vertex.
    // sum = 1, n = 1. Neither > 0.5 -> k=2. mapped: (1-1, 1-1) = (0, 0).
    // (0, 0) is the extraordinary-vertex corner of the unit triangle.
    {
        const auto tm = stam_tile_map(0.5, 0.5);
        REQUIRE(tm.n == 1);
        REQUIRE(tm.k == 2);
        REQUIRE(std::abs(tm.v_p - 0.0) < kTol);
        REQUIRE(std::abs(tm.w_p - 0.0) < kTol);
    }
}

TEST_CASE("stam_tile_map: round-trip identity over interior grid",
          "[shell][loop][stam][tile_map][roundtrip]")
{
    using chladni::shell::loop::stam_tile_map;

    // Strict-interior grid samples — round trip must reproduce input.
    for (double v : {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}) {
        for (double w : {0.02, 0.05, 0.1, 0.15, 0.25, 0.4}) {
            if (v + w > 0.99) continue;       // stay strictly inside Omega
            if (v + w < 1.0e-3) continue;     // stay away from origin
            CAPTURE(v, w);
            const auto tm = stam_tile_map(v, w);
            const auto [v_back, w_back] =
                inverse_tile_transform(tm.n, tm.k, tm.v_p, tm.w_p);
            INFO("n=" << tm.n << " k=" << tm.k
                 << " v_p=" << tm.v_p << " w_p=" << tm.w_p
                 << " ; recovered v=" << v_back << " w=" << w_back);
            REQUIRE(std::abs(v_back - v) < 1.0e-12);
            REQUIRE(std::abs(w_back - w) < 1.0e-12);
        }
    }
}

TEST_CASE("stam_tile_map: round-trip on deep tiles near the origin",
          "[shell][loop][stam][tile_map][roundtrip]")
{
    using chladni::shell::loop::stam_tile_map;

    // Probe many levels — n = 1 through ~20 — by halving v + w each pass.
    double v = 0.4;
    double w = 0.3;
    for (int trial = 0; trial < 20; ++trial) {
        const auto tm = stam_tile_map(v, w);
        CAPTURE(trial, v, w, tm.n, tm.k);
        REQUIRE(tm.n >= 1);
        REQUIRE(tm.k >= 1);
        REQUIRE(tm.k <= 3);
        // Mapped (v_p, w_p) must lie in the unit triangle (with some slack
        // for floating-point round-off at the boundary).
        REQUIRE(tm.v_p >= -1.0e-12);
        REQUIRE(tm.w_p >= -1.0e-12);
        REQUIRE(tm.v_p + tm.w_p <= 1.0 + 1.0e-12);
        // Round-trip.
        const auto [v_back, w_back] =
            inverse_tile_transform(tm.n, tm.k, tm.v_p, tm.w_p);
        REQUIRE(std::abs(v_back - v) < 1.0e-12);
        REQUIRE(std::abs(w_back - w) < 1.0e-12);

        v *= 0.5;
        w *= 0.5;
    }
}

TEST_CASE("stam_tile_map: tile-of-(v_p, w_p)-mapped tile spans unit triangle",
          "[shell][loop][stam][tile_map][coverage]")
{
    using chladni::shell::loop::stam_tile_map;

    // For each tile k in {1, 2, 3} at n=1, exercise a corner of the
    // tile's source region and verify the mapped (v_p, w_p) lands at
    // a corner of the unit triangle. This pins each t_{1,k} explicitly.
    // Source corners for each n=1 tile (from Stam Sec 3 Fig. 5):
    //   k=1: (1/2, 0), (1, 0), (1/2, 1/2)     ->  (0, 0), (1, 0), (0, 1)
    //   k=2: (0, 0)*, (1/2, 0), (0, 1/2)      ->  (1, 1)*, (0, 1), (1, 0)
    //   k=3: (0, 1/2), (1/2, 1/2), (0, 1)     ->  (0, 0), (1, 0), (0, 1)
    // (Starred entries are degenerate corners we exclude.)

    // k=1 corner (1, 0) -> mapped (1, 0).
    {
        const auto tm = stam_tile_map(1.0, 0.0);
        REQUIRE(tm.k == 1);
        REQUIRE(std::abs(tm.v_p - 1.0) < kTol);
        REQUIRE(std::abs(tm.w_p - 0.0) < kTol);
    }
    // k=1 corner (1/2 + eps, 1/2 - eps) -> mapped (2*eps, 1 - 2*eps), close to (0, 1).
    {
        const double eps = 1.0e-6;
        const auto tm = stam_tile_map(0.5 + eps, 0.5 - eps);
        REQUIRE(tm.n == 1);
        REQUIRE(tm.k == 1);
        REQUIRE(std::abs(tm.v_p - 2.0 * eps) < 1.0e-10);
        REQUIRE(std::abs(tm.w_p - (1.0 - 2.0 * eps)) < 1.0e-10);
    }
    // k=3 corner (0, 1) -> mapped (0, 1).
    {
        const auto tm = stam_tile_map(0.0, 1.0);
        REQUIRE(tm.k == 3);
        REQUIRE(std::abs(tm.v_p - 0.0) < kTol);
        REQUIRE(std::abs(tm.w_p - 1.0) < kTol);
    }
}

TEST_CASE("stam_tile_map: invalid args throw",
          "[shell][loop][stam][tile_map][validation]")
{
    using chladni::shell::loop::stam_tile_map;

    REQUIRE_THROWS_AS(stam_tile_map(-0.1, 0.5), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_tile_map(0.5, -0.1), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_tile_map(0.6, 0.5), std::invalid_argument);  // v + w = 1.1
    REQUIRE_THROWS_AS(stam_tile_map(0.0, 0.0), std::invalid_argument);  // origin
}

TEST_CASE("stam_tile_map: tiny-but-nonzero v+w yields finite tile (review7 I8)",
          "[shell][loop][stam][tile_map][validation]")
{
    using chladni::shell::loop::stam_tile_map;

    // A near-apex sum just above the kMinSum reject threshold drives the
    // level index n ~ 1000; without the n cap, pow(2, n-1) overflows to +inf
    // and the rescaled tile parameters are non-finite. The cap keeps the
    // public-API output finite (the eigenstructure is numerically dead this
    // deep anyway).
    const auto tm = stam_tile_map(5.0e-301, 5.0e-301);  // sum = 1e-300
    REQUIRE(tm.n <= 60);
    REQUIRE(std::isfinite(tm.v_p));
    REQUIRE(std::isfinite(tm.w_p));
}
