/**
 * @file test_mesh_generate_cylinder.cpp
 * @brief Pin
 *   @ref chladni::mesh::generate_cylinder
 *   to its documented topology and geometry.
 */

#include <chladni/mesh.hpp>

#include <Eigen/Geometry>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace {

constexpr double kAbsTol = 1.0e-12;

}  // namespace

TEST_CASE("generate_cylinder: vertex and face counts match the documented topology",
          "[mesh][procedural][cylinder]")
{
    const int n_around = 16;
    const int n_along  = 8;
    const auto m = chladni::mesh::generate_cylinder(1.0, 4.0, n_around, n_along);

    REQUIRE(m.num_vertices() == n_around * (n_along + 1));
    REQUIRE(m.num_faces()    == 2 * n_around * n_along);
}

TEST_CASE("generate_cylinder: every vertex sits on the parametric mid-surface",
          "[mesh][procedural][cylinder]")
{
    using Catch::Matchers::WithinAbs;
    const double R = 0.13;
    const double L = 1.7;
    const int n_around = 12;
    const int n_along  = 5;
    const auto m = chladni::mesh::generate_cylinder(R, L, n_around, n_along);

    constexpr double pi = std::numbers::pi_v<double>;
    for (int j = 0; j <= n_along; ++j) {
        for (int i = 0; i < n_around; ++i) {
            const Eigen::Index idx = j * n_around + i;
            const double x = R * std::cos(2.0 * pi * i / n_around);
            const double y = R * std::sin(2.0 * pi * i / n_around);
            const double z = L * static_cast<double>(j) / n_along;
            INFO("ring j=" << j << " around i=" << i << " (linear index " << idx << ")");
            REQUIRE_THAT(m.V(idx, 0), WithinAbs(x, kAbsTol));
            REQUIRE_THAT(m.V(idx, 1), WithinAbs(y, kAbsTol));
            REQUIRE_THAT(m.V(idx, 2), WithinAbs(z, kAbsTol));
        }
    }
}

TEST_CASE("generate_cylinder: topologically a tube — no seam, two open boundary loops",
          "[mesh][procedural][cylinder]")
{
    // Two assertions equivalent to seamlessness:
    //  (1) every interior edge has exactly two adjacent faces (manifold);
    //  (2) the only boundary edges are the two end loops at z = 0 and z = L
    //      — i.e. boundary_edges == 2 * n_around exactly.
    // A naive flat-rectangular-strip generator that duplicates vertices at
    // the wrap-around (a "seam") would have 2 * n_around + 2 * n_along
    // boundary edges; this test catches that.
    // Equivalent Euler-characteristic statement: V - E + F = 0 (chi(tube) = 0
    // versus chi(disk) = 1 for a strip with seam).
    const double R = 0.7;
    const double L = 1.3;
    const int n_around = 8;
    const int n_along  = 4;
    const auto m = chladni::mesh::generate_cylinder(R, L, n_around, n_along);

    std::map<std::pair<int, int>, int> edge_count;
    for (Eigen::Index f = 0; f < m.num_faces(); ++f) {
        const std::array<int, 3> v = {m.F(f, 0), m.F(f, 1), m.F(f, 2)};
        for (std::size_t e = 0; e < 3; ++e) {
            const int a = v[e];
            const int b = v[(e + 1u) % 3u];
            const auto key = std::pair<int, int>{std::min(a, b), std::max(a, b)};
            ++edge_count[key];
        }
    }

    int boundary_edges = 0;
    int interior_edges = 0;
    for (const auto& [key, count] : edge_count) {
        if (count == 1) {
            ++boundary_edges;
            // Boundary edges must lie at z=0 or z=L (the two end loops);
            // an internal seam would put boundary edges at intermediate z.
            const double z_a = m.V(key.first,  2);
            const double z_b = m.V(key.second, 2);
            const bool at_bottom = std::abs(z_a) < kAbsTol
                                && std::abs(z_b) < kAbsTol;
            const bool at_top    = std::abs(z_a - L) < kAbsTol
                                && std::abs(z_b - L) < kAbsTol;
            INFO("boundary edge (" << key.first << ", " << key.second
                 << ") z_a=" << z_a << " z_b=" << z_b);
            REQUIRE((at_bottom || at_top));
        } else if (count == 2) {
            ++interior_edges;
        } else {
            INFO("edge (" << key.first << ", " << key.second
                 << ") has " << count << " adjacent faces");
            FAIL("non-manifold edge");
        }
    }

    REQUIRE(boundary_edges == 2 * n_around);

    // Euler characteristic of a cylindrical tube is 0.
    const int V = static_cast<int>(m.num_vertices());
    const int F = static_cast<int>(m.num_faces());
    const int E = boundary_edges + interior_edges;
    REQUIRE((V - E + F) == 0);
}

TEST_CASE("generate_cylinder: face normals point outward from the cylinder axis",
          "[mesh][procedural][cylinder]")
{
    const double R = 0.5;
    const double L = 2.0;
    const int n_around = 16;
    const int n_along  = 8;
    const auto m = chladni::mesh::generate_cylinder(R, L, n_around, n_along);

    for (Eigen::Index f = 0; f < m.num_faces(); ++f) {
        const Eigen::RowVector3d v0 = m.V.row(m.F(f, 0));
        const Eigen::RowVector3d v1 = m.V.row(m.F(f, 1));
        const Eigen::RowVector3d v2 = m.V.row(m.F(f, 2));
        const Eigen::RowVector3d n = (v1 - v0).cross(v2 - v0);
        const Eigen::RowVector3d centroid = (v0 + v1 + v2) / 3.0;
        // Outward radial direction at the centroid: (cx, cy, 0).
        const double radial_dot = n(0) * centroid(0) + n(1) * centroid(1);
        INFO("face " << f << " centroid=" << centroid << " normal=" << n
             << " radial_dot=" << radial_dot);
        REQUIRE(radial_dot > 0.0);
    }
}

TEST_CASE("generate_cylinder: rejects degenerate arguments",
          "[mesh][procedural][cylinder]")
{
    REQUIRE_THROWS_AS(chladni::mesh::generate_cylinder(0.0, 1.0, 8, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(chladni::mesh::generate_cylinder(1.0, 0.0, 8, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(chladni::mesh::generate_cylinder(1.0, 1.0, 2, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(chladni::mesh::generate_cylinder(1.0, 1.0, 8, 0),
                      std::invalid_argument);
}
