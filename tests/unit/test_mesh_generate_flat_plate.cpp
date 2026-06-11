/**
 * @file test_mesh_generate_flat_plate.cpp
 * @brief Topology and geometry tests for chladni::mesh::generate_flat_plate.
 *
 * Pins the chamfered-rectangle structure that the L.5c.1 augmentation
 * relies on: all-valence-6 interior, all-valence-4 mid-boundary, and
 * exactly 6 valence-3 corners (no valence-2 corners). Two cell sizes
 * exercised: 2x2 (smallest legal — degenerates into the hex-fan
 * fixture from test_loop_boundary.cpp) and 4x3 (general rectangle).
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>

#include <map>
#include <set>
#include <utility>

using Catch::Matchers::WithinAbs;

namespace {

/// Tally the valences of all vertices in a TriMesh.
std::vector<int> mesh_valences(const chladni::mesh::TriMesh& m)
{
    const auto edges = chladni::shell::build_edges(m.F);
    return chladni::shell::loop::vertex_valences(m.V.rows(), edges);
}

/// Count how many vertices have each (valence, on-boundary) class.
struct ValenceTally {
    int n_valence3_boundary = 0;
    int n_valence4_boundary = 0;
    int n_valence_other_boundary = 0;
    int n_valence6_interior = 0;
    int n_valence_other_interior = 0;
};

ValenceTally tally(const chladni::mesh::TriMesh& m)
{
    const auto edges = chladni::shell::build_edges(m.F);
    const auto vals  =
        chladni::shell::loop::vertex_valences(m.V.rows(), edges);
    std::vector<bool> is_bdry(m.V.rows(), false);
    for (const auto& e : edges) {
        if (e.is_boundary()) {
            is_bdry[e.v0] = true;
            is_bdry[e.v1] = true;
        }
    }

    ValenceTally t;
    for (Eigen::Index v = 0; v < m.V.rows(); ++v) {
        const int val = vals[v];
        if (is_bdry[v]) {
            if      (val == 3) ++t.n_valence3_boundary;
            else if (val == 4) ++t.n_valence4_boundary;
            else               ++t.n_valence_other_boundary;
        } else {
            if (val == 6) ++t.n_valence6_interior;
            else          ++t.n_valence_other_interior;
        }
    }
    return t;
}

}  // namespace

TEST_CASE("generate_flat_plate: argument validation",
          "[mesh][plate][validation]")
{
    using chladni::mesh::generate_flat_plate;
    REQUIRE_THROWS_AS(generate_flat_plate(0.0, 1.0, 2, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(generate_flat_plate(1.0, 0.0, 2, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(generate_flat_plate(1.0, 1.0, 1, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(generate_flat_plate(1.0, 1.0, 2, 1), std::invalid_argument);
    REQUIRE_NOTHROW   (generate_flat_plate(1.0, 1.0, 2, 2));
}

TEST_CASE("generate_flat_plate: 2x2 — vertex / face counts",
          "[mesh][plate][counts]")
{
    const auto m = chladni::mesh::generate_flat_plate(1.0, 1.0, 2, 2);

    // (n_x+1)(n_y+1) - 2 omitted corners.
    REQUIRE(m.V.rows() == 9 - 2);
    REQUIRE(m.V.cols() == 3);
    // 2 * n_x * n_y - 2 chamfer-removed triangles.
    REQUIRE(m.F.rows() == 2 * 2 * 2 - 2);
    REQUIRE(m.F.cols() == 3);
}

TEST_CASE("generate_flat_plate: 2x2 — all corners valence 3, one valence-6 interior",
          "[mesh][plate][valence][2x2]")
{
    const auto m = chladni::mesh::generate_flat_plate(1.0, 1.0, 2, 2);
    const auto t = tally(m);

    REQUIRE(t.n_valence3_boundary       == 6);
    REQUIRE(t.n_valence4_boundary       == 0);  // 2x2 has no mid-boundary
    REQUIRE(t.n_valence_other_boundary  == 0);
    REQUIRE(t.n_valence6_interior       == 1);
    REQUIRE(t.n_valence_other_interior  == 0);
}

TEST_CASE("generate_flat_plate: 4x3 — interior all valence-6, boundary mix of 3 and 4",
          "[mesh][plate][valence][4x3]")
{
    const auto m = chladni::mesh::generate_flat_plate(2.0, 1.5, 4, 3);
    const auto t = tally(m);

    // (n_x+1)(n_y+1) - 2 = 5*4 - 2 = 18 vertices.
    REQUIRE(m.V.rows() == 18);
    // Boundary loop length = 2*(n_x + n_y) - 2 = 12 vertices.
    // Interior count = 18 - 12 = 6 vertices, all valence-6.
    // Of the 12 boundary vertices: 6 are valence-3 (the chamfered hex
    // corners), the rest valence-4.
    REQUIRE(t.n_valence6_interior       == 6);
    REQUIRE(t.n_valence_other_interior  == 0);
    REQUIRE(t.n_valence3_boundary       == 6);
    REQUIRE(t.n_valence4_boundary       == 12 - 6);
    REQUIRE(t.n_valence_other_boundary  == 0);
}

TEST_CASE("generate_flat_plate: faces are CCW seen from +z",
          "[mesh][plate][winding]")
{
    const auto m = chladni::mesh::generate_flat_plate(2.0, 1.5, 4, 3);
    for (Eigen::Index f = 0; f < m.F.rows(); ++f) {
        const auto& V0 = m.V.row(m.F(f, 0));
        const auto& V1 = m.V.row(m.F(f, 1));
        const auto& V2 = m.V.row(m.F(f, 2));
        const double s = (V1(0) - V0(0)) * (V2(1) - V0(1))
                       - (V1(1) - V0(1)) * (V2(0) - V0(0));
        CAPTURE(f, s);
        REQUIRE(s > 0.0);
    }
}

TEST_CASE("generate_flat_plate: every vertex lies in the z=0 plane",
          "[mesh][plate][flat]")
{
    const auto m = chladni::mesh::generate_flat_plate(2.0, 1.5, 4, 3);
    for (Eigen::Index v = 0; v < m.V.rows(); ++v) {
        CAPTURE(v);
        REQUIRE_THAT(m.V(v, 2), WithinAbs(0.0, 0.0));
    }
}

TEST_CASE("generate_flat_plate: the generated mesh runs through assemble_stiffness_loop",
          "[mesh][plate][global_k]")
{
    const auto m = chladni::mesh::generate_flat_plate(2.0, 1.5, 4, 3);
    chladni::shell::ShellMaterial mat;
    mat.k_L           = 1.0e6;
    mat.k_B           = 1.0e3;
    mat.poisson_ratio = 0.3;
    REQUIRE_NOTHROW(
        chladni::shell::loop::assemble_stiffness_loop(m.V, m.F, mat));
}