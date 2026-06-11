/**
 * @file test_mesh_generate_icosphere.cpp
 * @brief Topology and geometry tests for chladni::mesh::generate_icosphere.
 *
 * Pins:
 *  - Combinatorics at n_subdivisions = 0, 1, 2, 3 against the
 *    closed-form formulas V_k = 12 + 30 (4^k - 1) / 3 and F_k = 20 4^k.
 *  - All vertices on the sphere of the requested radius.
 *  - Mesh closed (every edge has both face_left and face_right).
 *  - CCW outward winding (face normals point away from the origin).
 *  - Valence distribution: 12 valence-5 vertices (the original
 *    icosahedron corners) + the rest valence-6, for any n_subdivisions
 *    >= 1 (Loop subdivision preserves valence at the original-vertex
 *    slots; new edge-midpoint vertices have valence 6 once both their
 *    incident triangles get subdivided).
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

using Catch::Matchers::WithinRel;

namespace {

Eigen::Index expected_V_count(int k)
{
    // V_k = 12 + 30 * (4^k - 1) / 3
    long long four_k = 1;
    for (int i = 0; i < k; ++i) four_k *= 4;
    return 12 + 30LL * (four_k - 1) / 3;
}

Eigen::Index expected_F_count(int k)
{
    long long four_k = 1;
    for (int i = 0; i < k; ++i) four_k *= 4;
    return 20LL * four_k;
}

}  // namespace

TEST_CASE("generate_icosphere: argument validation",
          "[mesh][icosphere][validation]")
{
    REQUIRE_THROWS_AS(chladni::mesh::generate_icosphere(0.0,  0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(chladni::mesh::generate_icosphere(1.0, -1),
                      std::invalid_argument);
    REQUIRE_NOTHROW (chladni::mesh::generate_icosphere(1.0,  0));
}

TEST_CASE("generate_icosphere: combinatorics at multiple subdivisions",
          "[mesh][icosphere][counts]")
{
    for (int k : {0, 1, 2, 3}) {
        CAPTURE(k);
        const auto m = chladni::mesh::generate_icosphere(1.0, k);
        REQUIRE(m.V.rows() == expected_V_count(k));
        REQUIRE(m.F.rows() == expected_F_count(k));
        REQUIRE(m.V.cols() == 3);
        REQUIRE(m.F.cols() == 3);
    }
}

TEST_CASE("generate_icosphere: all vertices on the sphere of the given radius",
          "[mesh][icosphere][geometry]")
{
    const double R = 0.137;
    for (int k : {0, 1, 2, 3}) {
        CAPTURE(k);
        const auto m = chladni::mesh::generate_icosphere(R, k);
        for (Eigen::Index i = 0; i < m.V.rows(); ++i) {
            CAPTURE(i);
            const double r = m.V.row(i).norm();
            REQUIRE_THAT(r, WithinRel(R, 1e-12));
        }
    }
}

TEST_CASE("generate_icosphere: faces are CCW seen from outside",
          "[mesh][icosphere][winding]")
{
    for (int k : {0, 1, 2}) {
        CAPTURE(k);
        const auto m = chladni::mesh::generate_icosphere(1.0, k);
        for (Eigen::Index f = 0; f < m.F.rows(); ++f) {
            const Eigen::Vector3d v0 = m.V.row(m.F(f, 0));
            const Eigen::Vector3d v1 = m.V.row(m.F(f, 1));
            const Eigen::Vector3d v2 = m.V.row(m.F(f, 2));
            const Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
            const Eigen::Vector3d c = (v0 + v1 + v2) / 3.0;
            CAPTURE(f, n.dot(c));
            REQUIRE(n.dot(c) > 0.0);  // outward
        }
    }
}

TEST_CASE("generate_icosphere: mesh is closed (no boundary edges)",
          "[mesh][icosphere][closed]")
{
    for (int k : {0, 1, 2}) {
        CAPTURE(k);
        const auto m = chladni::mesh::generate_icosphere(1.0, k);
        const auto edges = chladni::shell::build_edges(m.F);
        for (const auto& e : edges) {
            CAPTURE(e.v0, e.v1);
            REQUIRE(e.is_interior());
        }
    }
}

TEST_CASE("generate_icosphere: valence distribution = 12 v5 + (V-12) v6",
          "[mesh][icosphere][valence]")
{
    // n=0 (bare icosahedron) has all 12 vertices at valence 5. Subdivide
    // and the 12 originals stay valence-5, the new edge-midpoint
    // vertices come out valence-6.
    for (int k : {1, 2, 3}) {
        CAPTURE(k);
        const auto m = chladni::mesh::generate_icosphere(1.0, k);
        const auto edges = chladni::shell::build_edges(m.F);
        const auto vals  =
            chladni::shell::loop::vertex_valences(m.V.rows(), edges);
        int n5 = 0, n6 = 0, n_other = 0;
        for (int v : vals) {
            if      (v == 5) ++n5;
            else if (v == 6) ++n6;
            else             ++n_other;
        }
        REQUIRE(n5 == 12);
        REQUIRE(n6 == m.V.rows() - 12);
        REQUIRE(n_other == 0);
    }

    // n=0 special: ALL 12 vertices are valence-5 (no valence-6 yet).
    {
        const auto m = chladni::mesh::generate_icosphere(1.0, 0);
        const auto edges = chladni::shell::build_edges(m.F);
        const auto vals  =
            chladni::shell::loop::vertex_valences(m.V.rows(), edges);
        for (int v : vals) {
            CAPTURE(v);
            REQUIRE(v == 5);
        }
    }
}

TEST_CASE("generate_icosphere: ships through assemble_stiffness_loop at k>=1",
          "[mesh][icosphere][global_k]")
{
    chladni::shell::ShellMaterial mat;
    mat.k_L           = 1.0e6;
    mat.k_B           = 1.0e3;
    mat.poisson_ratio = 0.3;

    // n_subdivisions >= 1 satisfies Cirak-Ortiz §4.6 step-1's
    // "at most one extraordinary corner per triangle" prerequisite.
    for (int k : {1, 2}) {
        CAPTURE(k);
        const auto m = chladni::mesh::generate_icosphere(0.10, k);
        REQUIRE_NOTHROW(
            chladni::shell::loop::assemble_stiffness_loop(m.V, m.F, mat));
    }
}
