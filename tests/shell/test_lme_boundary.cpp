/**
 * @file test_lme_boundary.cpp
 * @brief Unit tests for @ref chladni::shell::lme::collect_boundary_edges,
 *        the boundary-detection utility used by the ghost-node
 *        construction (Millán 2011 §4.1.2).
 *
 * Each test mesh exercises one of the structural properties:
 *
 *  1. **Closed manifold has zero boundary edges.** Icosphere k=2 is
 *     watertight; the boundary-edge list must be empty.
 *  2. **Single triangle has three boundary edges, one per side.** All
 *     three edges are incident to exactly one triangle, the same one;
 *     the third corner is the opposite vertex for each.
 *  3. **Polar disk has one boundary edge per rim segment.** A 32x4
 *     polar disk has 32 outer-rim segments; each yields a boundary
 *     edge whose opposite interior vertex is the inner-ring partner.
 *
 * In every case the @c v_int field must be distinct from @c v0 and
 * @c v1 (the third corner).
 */

#include <chladni/mesh.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <set>
#include <unordered_set>

using chladni::mesh::generate_circular_disk;
using chladni::mesh::generate_icosphere;
using chladni::shell::lme::BoundaryEdge;
using chladni::shell::lme::build_ghost_positions;
using chladni::shell::lme::collect_boundary_edges;
using chladni::shell::lme::reflect_across_edge_line;

TEST_CASE("collect_boundary_edges: closed icosphere has zero boundary edges",
          "[lme][boundary][validation]")
{
    const auto mesh = generate_icosphere(/*radius=*/1.0, /*n_subdivisions=*/2);
    const auto bdry = collect_boundary_edges(mesh.F);
    REQUIRE(bdry.empty());
}

TEST_CASE("collect_boundary_edges: single triangle yields three edges",
          "[lme][boundary][validation]")
{
    // Three vertices forming one triangle. Every edge belongs to
    // exactly that triangle, so all three are boundary edges and the
    // opposite-corner vertex for each is the third triangle corner.
    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 2;

    const auto bdry = collect_boundary_edges(F);
    REQUIRE(bdry.size() == 3);

    std::set<std::pair<int, int>> seen_edges;
    for (const auto& e : bdry) {
        REQUIRE(e.v0 < e.v1);
        REQUIRE(e.v_int != e.v0);
        REQUIRE(e.v_int != e.v1);
        REQUIRE(e.face == 0);
        seen_edges.insert({e.v0, e.v1});
    }
    REQUIRE(seen_edges.size() == 3);
    REQUIRE(seen_edges.count({0, 1}) == 1);
    REQUIRE(seen_edges.count({0, 2}) == 1);
    REQUIRE(seen_edges.count({1, 2}) == 1);
}

TEST_CASE("collect_boundary_edges: 32x4 polar disk has 32 rim edges",
          "[lme][boundary][validation]")
{
    constexpr double R = 0.10;
    constexpr int    n_az  = 32;
    constexpr int    n_rad = 4;
    const auto       mesh  = generate_circular_disk(R, n_az, n_rad);

    const auto bdry = collect_boundary_edges(mesh.F);
    REQUIRE(bdry.size() == static_cast<std::size_t>(n_az));

    // Every boundary endpoint must be a rim vertex (radius = R).
    // generate_circular_disk's vertex 0 is the centre; rim vertices
    // are the outermost ring. We test directly via radius rather than
    // index convention so this stays robust to mesh re-numbering.
    constexpr double tol = 1e-9;
    std::unordered_set<int> bdry_vertices;
    for (const auto& e : bdry) {
        REQUIRE(e.v0 < e.v1);
        REQUIRE(e.v_int != e.v0);
        REQUIRE(e.v_int != e.v1);

        const double r0 = mesh.V.row(e.v0).head<2>().norm();
        const double r1 = mesh.V.row(e.v1).head<2>().norm();
        REQUIRE(std::abs(r0 - R) < tol);
        REQUIRE(std::abs(r1 - R) < tol);

        // v_int is the wing inside the disk → strictly inside the rim.
        const double r_int = mesh.V.row(e.v_int).head<2>().norm();
        REQUIRE(r_int < R - tol);

        bdry_vertices.insert(e.v0);
        bdry_vertices.insert(e.v1);
    }
    // 32 rim vertices, each appearing on two adjacent edges.
    REQUIRE(bdry_vertices.size() == static_cast<std::size_t>(n_az));
}

TEST_CASE("reflect_across_edge_line: planar reflection lies in the plane",
          "[lme][boundary][validation]")
{
    // Edge along the x-axis, v_int above the y-axis. Ghost must land
    // at the mirror position below the x-axis.
    Eigen::RowVector3d v0(0.0, 0.0, 0.0);
    Eigen::RowVector3d v1(1.0, 0.0, 0.0);
    Eigen::RowVector3d v_int(0.3, 0.7, 0.0);

    const Eigen::RowVector3d ghost = reflect_across_edge_line(v0, v1, v_int);
    REQUIRE(std::abs(ghost.x() - 0.3) < 1e-15);
    REQUIRE(std::abs(ghost.y() + 0.7) < 1e-15);
    REQUIRE(std::abs(ghost.z())       < 1e-15);
}

TEST_CASE("reflect_across_edge_line: reflection is involutory",
          "[lme][boundary][validation]")
{
    // Reflecting the ghost back across the same edge line must
    // recover the original v_int to floating-point.
    Eigen::RowVector3d v0(1.2, -0.3, 0.5);
    Eigen::RowVector3d v1(-0.4, 0.8, 1.1);
    Eigen::RowVector3d v_int(0.7, 0.1, -0.2);

    const Eigen::RowVector3d ghost  = reflect_across_edge_line(v0, v1, v_int);
    const Eigen::RowVector3d v_back = reflect_across_edge_line(v0, v1, ghost);
    REQUIRE((v_back - v_int).norm() < 1e-14);
}

TEST_CASE("reflect_across_edge_line: foot of perpendicular is invariant",
          "[lme][boundary][validation]")
{
    // The midpoint of (v_int, ghost) must lie on the edge line through
    // (v0, v1). This is the defining geometric property of reflection
    // across the edge line.
    Eigen::RowVector3d v0(0.5, 1.0, 0.0);
    Eigen::RowVector3d v1(2.5, 1.0, 0.0);
    Eigen::RowVector3d v_int(1.2, 1.4, 0.6);

    const Eigen::RowVector3d ghost = reflect_across_edge_line(v0, v1, v_int);
    const Eigen::RowVector3d mid   = 0.5 * (v_int + ghost);

    // Edge is parallel to x-axis at y=1, z=0. The midpoint must have
    // y == 1.0 and z == 0.0; x can be anything on the line.
    REQUIRE(std::abs(mid.y() - 1.0) < 1e-15);
    REQUIRE(std::abs(mid.z())       < 1e-15);
}

TEST_CASE("reflect_across_edge_line: throws on zero-length edge",
          "[lme][boundary][validation]")
{
    Eigen::RowVector3d v0(0.0, 0.0, 0.0);
    Eigen::RowVector3d v_int(1.0, 1.0, 1.0);
    REQUIRE_THROWS_AS(reflect_across_edge_line(v0, v0, v_int),
                      std::invalid_argument);
}

TEST_CASE("build_ghost_positions: polar disk ghosts lie outside the rim",
          "[lme][boundary][validation]")
{
    // The 32x4 polar disk has 32 rim segments. The ghost row (one per
    // boundary edge, at the chord midpoint, offset by the first
    // interior row's normal spacing) must lie OUTSIDE the disk
    // (radius > R) at roughly one radial spacing dr past the rim.
    constexpr double R = 0.10;
    constexpr int    n_az  = 32;
    constexpr int    n_rad = 4;
    const auto       mesh  = generate_circular_disk(R, n_az, n_rad);
    const double     dr    = R / n_rad;

    const auto bdry   = collect_boundary_edges(mesh.F);
    const auto ghosts = build_ghost_positions(mesh.V, mesh.F, bdry);

    REQUIRE(ghosts.rows() == static_cast<Eigen::Index>(bdry.size()));
    REQUIRE(ghosts.cols() == 3);

    constexpr double tol = 1e-12;
    for (Eigen::Index i = 0; i < ghosts.rows(); ++i) {
        const double r_ghost = ghosts.row(i).head<2>().norm();
        REQUIRE(r_ghost > R + tol);
        // Offset comparable to the interior radial spacing (the row
        // mirrors the first interior ring; chord sagitta gives the
        // small slack below dr).
        REQUIRE(r_ghost - R > 0.5 * dr);
        REQUIRE(r_ghost - R < 1.5 * dr);

        // Disk is planar (z=0); ghosts must stay in the plane.
        REQUIRE(std::abs(ghosts(i, 2)) < 1e-14);
    }
}

TEST_CASE("build_ghost_positions: uniform row regardless of QuadSplit rim "
          "valence",
          "[lme][boundary][quadsplit][validation]")
{
    // REGRESSION (2026-06-03, QuadSplit-rim bug): the previous recipe
    // reflected the boundary-adjacent triangle's interior corner
    // (v_int) across the rim chord, which inherits TRIANGULATION
    // artifacts rather than the boundary's intrinsic geometry:
    //  - Checkerboard: adjacent boundary edges share the same v_int →
    //    near-coincident ghost PAIRS (~0.2 dr apart) at every other
    //    rim vertex — nearly linearly dependent LME basis pairs that
    //    produced a spurious low-frequency m=0 mode (45 Hz vs the
    //    physical 234 Hz breathing mode on the 32x8 Leissa disk).
    //  - UnionJack: v_int is the quad-CENTRE vertex at half the radial
    //    spacing → a ghost row only ~0.4 dr outside the rim, again
    //    nearly dependent against the rim row (spurious mode at 86 Hz).
    // The cure: one ghost per boundary edge at the chord midpoint,
    // offset along the in-plane outward normal by the FIRST INTERIOR
    // ROW's directional spacing (max |(x_w - m)·n_out| over interior
    // one-ring neighbours of the edge endpoints) — the same
    // max-adjacent directional-spacing rule the SME gaps use (RMA13
    // §3.2.1). All three splits share the same rim chords and the same
    // ~dr-deep first interior row, so the ghost row must come out
    // (nearly) IDENTICAL across splits.
    constexpr double R     = 0.10;
    constexpr int    n_az  = 32;
    constexpr int    n_rad = 8;
    const double     dr    = R / n_rad;

    using chladni::mesh::QuadSplit;
    const auto run = [&](QuadSplit split) {
        const auto mesh = generate_circular_disk(R, n_az, n_rad, split);
        const auto bdry = collect_boundary_edges(mesh.F);
        return build_ghost_positions(mesh.V, mesh.F, bdry);
    };
    const Eigen::MatrixXd g_cons = run(QuadSplit::Consistent);
    const Eigen::MatrixXd g_cb   = run(QuadSplit::Checkerboard);
    const Eigen::MatrixXd g_uj   = run(QuadSplit::UnionJack);

    REQUIRE(g_cons.rows() == n_az);
    REQUIRE(g_cb.rows()   == n_az);
    REQUIRE(g_uj.rows()   == n_az);

    // Property 1: NO near-coincident ghost pairs — min pairwise ghost
    // distance stays on the rim-spacing scale for every split.
    const auto min_pair_dist = [](const Eigen::MatrixXd& g) {
        double d2_min = std::numeric_limits<double>::infinity();
        for (Eigen::Index i = 0; i < g.rows(); ++i) {
            for (Eigen::Index j = i + 1; j < g.rows(); ++j) {
                d2_min = std::min(
                    d2_min, (g.row(i) - g.row(j)).squaredNorm());
            }
        }
        return std::sqrt(d2_min);
    };
    REQUIRE(min_pair_dist(g_cons) > 0.5 * dr);
    REQUIRE(min_pair_dist(g_cb)   > 0.5 * dr);
    REQUIRE(min_pair_dist(g_uj)   > 0.5 * dr);

    // Property 2: every ghost sits roughly one interior-row spacing
    // outside the rim (NOT the v_int lottery's 0.4 dr).
    const auto offsets_ok = [&](const Eigen::MatrixXd& g) {
        for (Eigen::Index i = 0; i < g.rows(); ++i) {
            const double off = g.row(i).head<2>().norm() - R;
            if (off < 0.5 * dr || off > 1.5 * dr) return false;
        }
        return true;
    };
    REQUIRE(offsets_ok(g_cons));
    REQUIRE(offsets_ok(g_cb));
    REQUIRE(offsets_ok(g_uj));

    // Property 3: the three splits produce rows on the SAME SCALE.
    // Exact identity is too strict — the max-adjacent probe sees the
    // diagonal interior neighbours on Consistent/Checkerboard (whose
    // distance below a curved rim's chord is inflated by the sagitta
    // of their azimuthal offset, ~27% here) but only the radial one
    // on UnionJack. That jitter is harmless for hull extension; the
    // bug regime was rows at 2-3x CLOSER spacing. Compare as SETS
    // (row ordering follows the boundary-edge enumeration).
    const auto set_dist = [](const Eigen::MatrixXd& a,
                             const Eigen::MatrixXd& b) {
        double worst = 0.0;
        for (Eigen::Index i = 0; i < a.rows(); ++i) {
            double best = std::numeric_limits<double>::infinity();
            for (Eigen::Index j = 0; j < b.rows(); ++j) {
                best = std::min(
                    best, (a.row(i) - b.row(j)).squaredNorm());
            }
            worst = std::max(worst, std::sqrt(best));
        }
        return worst;
    };
    REQUIRE(set_dist(g_cb, g_cons) < 0.35 * dr);
    REQUIRE(set_dist(g_uj, g_cons) < 0.35 * dr);
}
