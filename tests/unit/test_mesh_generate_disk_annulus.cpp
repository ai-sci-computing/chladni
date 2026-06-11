/**
 * @file test_mesh_generate_disk_annulus.cpp
 * @brief Topology / winding / manifoldness tests for the disk and annulus
 *        generators (R22): generate_disk_iso, generate_disk_hex,
 *        generate_annulus.
 *
 * These generators carry off-by-one-prone ring/strip logic and documented
 * exact vertex/triangle counts, but had no direct unit coverage — only
 * indirect use inside the modal-validation shell tests. We pin the counts,
 * confirm CCW winding (positive +z signed area on every face), confirm the
 * mesh is edge-manifold (build_edges does not throw), and count the
 * boundary loops, plus the documented invalid-argument rejections.
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>

namespace {

using chladni::mesh::TriMesh;

/// Signed area of triangle f projected onto the z=0 plane; positive iff the
/// winding is CCW seen from +z.
double signed_z_area(const TriMesh& m, Eigen::Index f)
{
    const Eigen::Vector3d a = m.V.row(m.F(f, 0));
    const Eigen::Vector3d b = m.V.row(m.F(f, 1));
    const Eigen::Vector3d c = m.V.row(m.F(f, 2));
    return 0.5 * ((b.x() - a.x()) * (c.y() - a.y())
                - (c.x() - a.x()) * (b.y() - a.y()));
}

/// Number of boundary edges (an edge bounding exactly one face).
int boundary_edge_count(const TriMesh& m)
{
    const auto edges = chladni::shell::build_edges(m.F);
    int n = 0;
    for (const auto& e : edges) {
        if (!e.is_interior()) ++n;
    }
    return n;
}

/// Assert every face winds CCW from +z and all vertices lie in z = 0.
void require_flat_ccw(const TriMesh& m)
{
    for (Eigen::Index f = 0; f < m.F.rows(); ++f) {
        INFO("face " << f << " signed z-area = " << signed_z_area(m, f));
        REQUIRE(signed_z_area(m, f) > 0.0);
    }
    for (Eigen::Index v = 0; v < m.V.rows(); ++v) {
        REQUIRE(std::abs(m.V(v, 2)) < 1e-12);
    }
}

}  // namespace

TEST_CASE("generate_disk_iso: counts, winding, manifold rim",
          "[mesh][disk][generator]")
{
    constexpr double R = 0.5;
    const int n_boundary = 24;
    const auto m = chladni::mesh::generate_disk_iso(R, n_boundary);

    // The first n_boundary vertices are the rim, ordered CCW from angle 0.
    REQUIRE(m.V.rows() >= n_boundary);
    for (int i = 0; i < n_boundary; ++i) {
        INFO("rim vertex " << i);
        REQUIRE(std::abs(m.V.row(i).head<2>().norm() - R) < 1e-9);
    }

    require_flat_ccw(m);
    // Edge-manifold (and vertex-manifold): build_edges must not throw.
    REQUIRE_NOTHROW(chladni::shell::build_edges(m.F));
    // Exactly one boundary loop (the rim) of n_boundary edges.
    REQUIRE(boundary_edge_count(m) == n_boundary);

    SECTION("invalid args throw") {
        REQUIRE_THROWS_AS(chladni::mesh::generate_disk_iso(0.0, 8),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(chladni::mesh::generate_disk_iso(R, 2),
                          std::invalid_argument);
    }
}

TEST_CASE("generate_disk_hex: closed-form counts, winding, manifold",
          "[mesh][disk][generator]")
{
    constexpr double R = 1.0;
    // After chamfering the 6 hex corners (see generate_disk_hex doc), the
    // counts are V = 3L^2 + 3L - 5, F = 6(L^2 - 1), rim = 6(L - 1). L >= 2.
    for (int L : {2, 3, 4}) {
        INFO("n_layers = " << L);
        const auto m = chladni::mesh::generate_disk_hex(R, L);

        REQUIRE(m.V.rows() == 3 * L * L + 3 * L - 5);
        REQUIRE(m.F.rows() == 6 * (L * L - 1));

        require_flat_ccw(m);
        REQUIRE_NOTHROW(chladni::shell::build_edges(m.F));
        // One chamfered-hexagon rim loop: 6(L-1) boundary edges.
        REQUIRE(boundary_edge_count(m) == 6 * (L - 1));
        // Rim vertices are radially snapped onto the circle r = R.
        const auto edges = chladni::shell::build_edges(m.F);
        for (const auto& e : edges) {
            if (e.is_interior()) continue;
            for (Eigen::Index v : {e.v0, e.v1}) {
                REQUIRE(std::abs(m.V.row(v).head<2>().norm() - R) < 1e-9);
            }
        }
    }

    SECTION("invalid args throw") {
        REQUIRE_THROWS_AS(chladni::mesh::generate_disk_hex(-1.0, 2),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(chladni::mesh::generate_disk_hex(R, 0),
                          std::invalid_argument);
        // L=1 chamfers away every rim vertex -> degenerate, now rejected.
        REQUIRE_THROWS_AS(chladni::mesh::generate_disk_hex(R, 1),
                          std::invalid_argument);
    }
}

TEST_CASE("generate_annulus: counts, winding, two rims, manifold",
          "[mesh][annulus][generator]")
{
    constexpr double a = 1.0;  // outer
    constexpr double b = 0.4;  // inner
    const int n_az = 16, n_r = 4;

    SECTION("Consistent split — base polar grid") {
        const auto m = chladni::mesh::generate_annulus(
            a, b, n_az, n_r, chladni::mesh::QuadSplit::Consistent);
        REQUIRE(m.V.rows() == n_r * n_az);
        REQUIRE(m.F.rows() == 2 * n_az * (n_r - 1));
        require_flat_ccw(m);
        REQUIRE_NOTHROW(chladni::shell::build_edges(m.F));
        // Two boundary loops (inner + outer rim), n_az edges each.
        REQUIRE(boundary_edge_count(m) == 2 * n_az);
        // Inner ring on r = b, outer ring on r = a.
        for (int i = 0; i < n_az; ++i) {
            REQUIRE(std::abs(m.V.row(i).head<2>().norm() - b) < 1e-9);
            REQUIRE(std::abs(m.V.row((n_r - 1) * n_az + i).head<2>().norm() - a)
                    < 1e-9);
        }
    }

    SECTION("UnionJack split — centre vertex per quad") {
        const auto m = chladni::mesh::generate_annulus(
            a, b, n_az, n_r, chladni::mesh::QuadSplit::UnionJack);
        REQUIRE(m.V.rows() == (2 * n_r - 1) * n_az);
        REQUIRE(m.F.rows() == 4 * n_az * (n_r - 1));
        require_flat_ccw(m);
        REQUIRE_NOTHROW(chladni::shell::build_edges(m.F));
        REQUIRE(boundary_edge_count(m) == 2 * n_az);
    }

    SECTION("invalid args throw") {
        REQUIRE_THROWS_AS(chladni::mesh::generate_annulus(a, b, 16, 1),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(chladni::mesh::generate_annulus(a, b, 2, 4),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(chladni::mesh::generate_annulus(a, a, 16, 4),
                          std::invalid_argument);  // inner == outer
        // Checkerboard requires even n_azimuthal.
        REQUIRE_THROWS_AS(
            chladni::mesh::generate_annulus(
                a, b, 15, 4, chladni::mesh::QuadSplit::Checkerboard),
            std::invalid_argument);
    }
}
