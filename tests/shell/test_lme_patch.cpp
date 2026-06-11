/**
 * @file test_lme_patch.cpp
 * @brief Unit tests for @ref chladni::shell::lme::Patch and
 *        @ref chladni::shell::lme::build_patch — the weighted-PCA tangent
 *        chart that the curved-shell LME assembler will build on each
 *        patch anchor.
 *
 * Tests the three properties of the wPCA chart described in
 * @cite millan_rosolen_arroyo_2011_thin_shell_maxent §2.1:
 *
 *  1. **Orthonormal tangent frame.** @f$ V_A^\top V_A = I @f$ — the
 *     columns of @c V are two orthonormal vectors. Independent of the
 *     mesh shape; failure indicates the eigendecomposition step is
 *     broken.
 *  2. **Flat-plane reduction.** A patch on a planar (z=0) point cloud
 *     has zero out-of-plane eigenvalue, and the projected coordinates
 *     @c xi preserve the in-plane geometry exactly (isometry).
 *  3. **Curved-surface non-planarity.** On a unit sphere or cylinder
 *     patch, the third (out-of-plane) eigenvalue is small relative to
 *     the in-plane spread — quantifies the local-curvature deviation
 *     from a flat tangent plane.
 *  4. **Tangent-direction sanity.** On a unit cylinder at anchor
 *     (1, 0, 0) the tangent plane contains the @c z axis and the
 *     azimuthal direction; the radial direction lies outside.
 */

#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>

#include <cmath>
#include <vector>

using chladni::shell::lme::build_patch;
using chladni::shell::lme::Patch;
using Catch::Matchers::WithinAbs;

namespace {

/// Build the 3x3 xy-grid centred at the origin in the z=0 plane. The
/// anchor is the centre vertex (id 4); the eight neighbours are at the
/// corners and edge midpoints of the 2h×2h square.
Eigen::MatrixXd flat_3x3_grid(double h)
{
    Eigen::MatrixXd X(9, 3);
    int row = 0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            X(row, 0) = i * h;
            X(row, 1) = j * h;
            X(row, 2) = 0.0;
            ++row;
        }
    }
    return X;
}

/// Sample 19 points from a unit sphere covering a small spherical cap
/// around the north pole — anchor at the north pole plus 6 nearest
/// neighbours plus 12 second-ring neighbours.
Eigen::MatrixXd spherical_cap_19(double cap_angle = 0.3)
{
    std::vector<Eigen::Vector3d> rows;
    rows.reserve(19);
    rows.emplace_back(0.0, 0.0, 1.0);  // north pole = anchor
    for (int ring = 1; ring <= 2; ++ring) {
        const double theta = ring * cap_angle / 2.0;
        const double z     = std::cos(theta);
        const double r     = std::sin(theta);
        const int    n_phi = (ring == 1) ? 6 : 12;
        for (int k = 0; k < n_phi; ++k) {
            const double phi = 2.0 * M_PI * k / n_phi;
            rows.emplace_back(r * std::cos(phi), r * std::sin(phi), z);
        }
    }
    Eigen::MatrixXd X(static_cast<Eigen::Index>(rows.size()), 3);
    for (std::size_t k = 0; k < rows.size(); ++k) {
        X.row(static_cast<Eigen::Index>(k)) = rows[k];
    }
    return X;
}

/// Sample 9 points on a unit cylinder (axis = z, radius = 1) around the
/// anchor (1, 0, 0): the anchor plus 8 neighbours in a 3x3 stencil over
/// (azimuth, axial) coordinates.
Eigen::MatrixXd cylinder_3x3_patch(double dphi, double dz)
{
    Eigen::MatrixXd X(9, 3);
    int row = 0;
    for (int jz = -1; jz <= 1; ++jz) {
        for (int ip = -1; ip <= 1; ++ip) {
            const double phi = ip * dphi;
            X(row, 0) = std::cos(phi);
            X(row, 1) = std::sin(phi);
            X(row, 2) = jz * dz;
            ++row;
        }
    }
    return X;
}

/// All node indices [0, n) — the "use every node as a neighbour" case.
std::vector<int> all_indices(int n)
{
    std::vector<int> ids(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        ids[static_cast<std::size_t>(k)] = k;
    }
    return ids;
}

}  // namespace

TEST_CASE("lme::build_patch: tangent frame V is orthonormal (3x3 grid)",
          "[shell][lme][patch]")
{
    const double h    = 0.1;
    const auto   X    = flat_3x3_grid(h);
    const auto   ids  = all_indices(static_cast<int>(X.rows()));
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(X.rows(), 1.6 / (h * h));

    const Patch P = build_patch(/*anchor_id=*/4, X, ids, beta);

    // V is 3x2; V^T V should equal the 2x2 identity to machine precision.
    const Eigen::Matrix2d VtV = P.V.transpose() * P.V;
    REQUIRE_THAT(VtV(0, 0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(VtV(1, 1), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(VtV(0, 1), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(VtV(1, 0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("lme::build_patch: flat plane has zero out-of-plane eigenvalue",
          "[shell][lme][patch]")
{
    const double h    = 0.1;
    const auto   X    = flat_3x3_grid(h);
    const auto   ids  = all_indices(static_cast<int>(X.rows()));
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(X.rows(), 1.6 / (h * h));

    const Patch P = build_patch(/*anchor_id=*/4, X, ids, beta);

    // The 3rd (smallest) eigenvalue of the wPCA covariance is the
    // out-of-plane spread. For coplanar nodes it must be 0 (within
    // floating-point noise of the eigensolver).
    REQUIRE_THAT(P.out_of_plane_eig, WithinAbs(0.0, 1e-14));

    // V's column space must be the z=0 plane: the third axis (0, 0, 1)
    // must be orthogonal to both columns.
    const Eigen::Vector3d ez(0.0, 0.0, 1.0);
    REQUIRE_THAT(ez.dot(P.V.col(0)), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(ez.dot(P.V.col(1)), WithinAbs(0.0, 1e-12));
}

TEST_CASE("lme::build_patch: flat plane ξ projection is isometric",
          "[shell][lme][patch]")
{
    const double h    = 0.1;
    const auto   X    = flat_3x3_grid(h);
    const auto   ids  = all_indices(static_cast<int>(X.rows()));
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(X.rows(), 1.6 / (h * h));

    const Patch P = build_patch(/*anchor_id=*/4, X, ids, beta);

    // For every neighbour, |ξ_a|² should equal |P_a - Q̄|² since the
    // points are exactly in the V's column span (the z=0 plane).
    REQUIRE(P.xi.rows() == X.rows());
    for (Eigen::Index k = 0; k < X.rows(); ++k) {
        const Eigen::Vector3d delta = X.row(k).transpose() - P.Qbar;
        const double          n3    = delta.squaredNorm();
        const double          n2    = P.xi.row(k).squaredNorm();
        REQUIRE_THAT(n2, WithinAbs(n3, 1e-12));
    }
}

TEST_CASE("lme::build_patch: spherical cap has small out-of-plane eigenvalue",
          "[shell][lme][patch]")
{
    const auto X   = spherical_cap_19(0.30);
    const auto ids = all_indices(static_cast<int>(X.rows()));
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(X.rows(), 4.0);

    const Patch P = build_patch(/*anchor_id=*/0, X, ids, beta);

    // The cap radius (in z) is roughly r = 1 - cos(cap_angle/2) ≈ 0.011.
    // Out-of-plane spread is bounded by r² ≈ 1.2e-4; the in-plane spread
    // scales like sin²(cap_angle/2) ≈ 0.022. We require the eigenvalue
    // ratio to be at least one order of magnitude apart — the surface is
    // nearly tangent-planar.
    REQUIRE(P.out_of_plane_eig < 1e-3);
}

TEST_CASE("lme::build_patch: cylinder patch tangent contains z-axis",
          "[shell][lme][patch]")
{
    // Anchor at (1, 0, 0) on a unit cylinder. Tangent plane at that point
    // is the xz-half-plane near x=1 (locally), i.e. the column space of V
    // must contain the z axis (and the azimuthal direction near (0, 1, 0)).
    const auto X   = cylinder_3x3_patch(0.10, 0.10);
    const auto ids = all_indices(static_cast<int>(X.rows()));
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(X.rows(), 100.0);

    const Patch P = build_patch(/*anchor_id=*/4, X, ids, beta);

    // The radial direction at anchor (1, 0, 0) is +x; the local normal.
    // It should be very nearly orthogonal to V's column space — the
    // projection of e_x onto V must have squared norm ≪ 1.
    const Eigen::Vector3d ex(1.0, 0.0, 0.0);
    const Eigen::Vector2d ex_in_V = P.V.transpose() * ex;
    REQUIRE(ex_in_V.squaredNorm() < 1e-2);

    // Conversely the z axis lies inside the tangent plane: the in-plane
    // component should account for essentially all of its norm.
    const Eigen::Vector3d ez(0.0, 0.0, 1.0);
    const Eigen::Vector2d ez_in_V = P.V.transpose() * ez;
    REQUIRE(ez_in_V.squaredNorm() > 0.99);
}

TEST_CASE("lme::build_patch: empty neighbour list throws",
          "[shell][lme][patch]")
{
    const auto X    = flat_3x3_grid(0.1);
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(X.rows(), 1.0);

    REQUIRE_THROWS_AS(
        build_patch(/*anchor_id=*/4, X, /*neighbor_ids=*/{}, beta),
        std::invalid_argument);
}

TEST_CASE("lme::build_patch: anchor out of range throws",
          "[shell][lme][patch]")
{
    const auto X    = flat_3x3_grid(0.1);
    const auto ids  = all_indices(static_cast<int>(X.rows()));
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(X.rows(), 1.0);

    REQUIRE_THROWS_AS(
        build_patch(/*anchor_id=*/99, X, ids, beta),
        std::invalid_argument);
}
