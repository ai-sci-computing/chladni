/**
 * @file test_lme_basis_curved.cpp
 * @brief Unit tests for the curved-shell LME basis evaluator
 *        @ref chladni::shell::lme::evaluate_basis_curved.
 *
 * Implements the closed form of
 * @cite millan_rosolen_arroyo_2011_thin_shell_maxent eq. (8):
 *
 *   @f[
 *     T_a(y) = \sum_{A \in \mathcal N_y^Q}
 *              w_A^Q(y)\;
 *              p_a\!\bigl(\Pi_A(y)\bigr).
 *   @f]
 *
 * The returned weights @f$ T_a(y) @f$ produce
 * @f$ u_h(y) = \sum_a T_a(y)\,u_a @f$ for any nodal field
 * @f$ \{u_a\} @f$. The test invariants follow from this:
 *
 *  1. **Single-patch flat-plate identity.** With a single patch
 *     covering every node on a planar mesh the curved evaluator must
 *     reduce exactly to the 2D LME basis (after the tangent
 *     projection, which is an isometry for in-plane points).
 *  2. **Partition of unity.** @f$ \sum_a T_a(y) = 1 @f$ — Shepard PoU
 *     and LME PoU compose multiplicatively into a global PoU on the
 *     reconstructed displacement.
 *  3. **Constant-field reproduction.** Setting @f$ u_a \equiv c @f$
 *     yields @f$ u_h(y) = c @f$ — follows from (2) but worth a direct
 *     check on a curved (sphere) fixture.
 *  4. **Two-patch flat-plate consistency.** Building two charts that
 *     share the same node set on a planar mesh still recovers the
 *     same vertex displacements as the 2D LME — the tangent-frame
 *     sign ambiguity wash through (|ξ−ξ_a|² is invariant).
 */

#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>

#include <cmath>
#include <numeric>
#include <vector>

using chladni::shell::lme::build_patch;
using chladni::shell::lme::CurvedBasisWeights;
using chladni::shell::lme::evaluate_basis;
using chladni::shell::lme::evaluate_basis_curved;
using chladni::shell::lme::Patch;
using Catch::Matchers::WithinAbs;

namespace {

/// 5x5 xy-grid on [-1, 1]^2 in the z=0 plane, row-major.
Eigen::MatrixXd flat_5x5_grid()
{
    constexpr int    N = 5;
    constexpr double L = 1.0;
    constexpr double h = 2.0 * L / (N - 1);
    Eigen::MatrixXd  X(N * N, 3);
    int              row = 0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            X(row, 0) = -L + i * h;
            X(row, 1) = -L + j * h;
            X(row, 2) = 0.0;
            ++row;
        }
    }
    return X;
}

/// All [0, n) indices.
std::vector<int> all_ids(int n)
{
    std::vector<int> ids(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) ids[static_cast<std::size_t>(k)] = k;
    return ids;
}

/// 19-point spherical cap around the north pole, copied from
/// test_lme_patch.cpp for sphere-fixture symmetry.
Eigen::MatrixXd spherical_cap_19(double cap_angle = 0.30)
{
    std::vector<Eigen::Vector3d> rows;
    rows.reserve(19);
    rows.emplace_back(0.0, 0.0, 1.0);
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

/// Sum the basis weights at a query for the PoU check.
double total_weight(const CurvedBasisWeights& w)
{
    return std::accumulate(w.values.begin(), w.values.end(), 0.0);
}

/// Look up the weight at global node id; returns 0 if not present.
double weight_at(const CurvedBasisWeights& w, int node_id)
{
    for (std::size_t k = 0; k < w.indices.size(); ++k) {
        if (w.indices[k] == node_id) return w.values[k];
    }
    return 0.0;
}

}  // namespace

TEST_CASE("lme::evaluate_basis_curved: single-patch flat plate reduces to 2D LME",
          "[shell][lme][basis_curved]")
{
    // Build a 5x5 z=0 grid and put a single patch with all 25 nodes as
    // neighbours, anchored at the central node (id 12). The curved
    // evaluator at any interior query point must agree with a direct 2D
    // LME basis evaluation on the (x, y) projections.
    const auto X        = flat_5x5_grid();
    const auto ids_all  = all_ids(static_cast<int>(X.rows()));
    const double h      = 0.5;  // grid spacing
    Eigen::VectorXd beta_lme =
        Eigen::VectorXd::Constant(X.rows(), 1.6 / (h * h));

    std::vector<Patch> patches;
    patches.push_back(build_patch(/*anchor_id=*/12, X, ids_all, beta_lme));

    Eigen::MatrixXd patch_points(1, 3);
    patch_points.row(0) = X.row(12);
    Eigen::VectorXd beta_patches = Eigen::VectorXd::Constant(1, 4.0);

    Eigen::Vector3d y(0.27, -0.13, 0.0);  // interior query

    const CurvedBasisWeights cw = evaluate_basis_curved(
        patches, patch_points, beta_patches,
        X, beta_lme, y,
        /*tol_shepard=*/1.0e-12,
        /*r_cut=*/100.0,
        /*newton_tol=*/1.0e-12,
        /*newton_max_iters=*/30);

    // Direct 2D evaluation: drop z, run evaluate_basis at the 2D query.
    Eigen::MatrixXd X2d = X.leftCols(2);
    Eigen::VectorXd y2d(2);  y2d << y.x(), y.y();
    const auto bv2d = evaluate_basis(X2d, beta_lme, y2d, /*r_cut=*/100.0);

    REQUIRE(cw.indices.size() == bv2d.indices.size());
    for (std::size_t k = 0; k < bv2d.indices.size(); ++k) {
        const int    node_id = bv2d.indices[k];
        const double p2d     = bv2d.values[k];
        const double pc      = weight_at(cw, node_id);
        REQUIRE_THAT(pc, WithinAbs(p2d, 1.0e-10));
    }
}

TEST_CASE("lme::evaluate_basis_curved: PoU identity on a sphere patch",
          "[shell][lme][basis_curved]")
{
    const auto X        = spherical_cap_19(0.30);
    const auto ids_all  = all_ids(static_cast<int>(X.rows()));
    Eigen::VectorXd beta_lme = Eigen::VectorXd::Constant(X.rows(), 50.0);

    std::vector<Patch> patches;
    patches.push_back(build_patch(/*anchor_id=*/0, X, ids_all, beta_lme));

    Eigen::MatrixXd patch_points(1, 3);
    patch_points.row(0) = X.row(0);
    Eigen::VectorXd beta_patches = Eigen::VectorXd::Constant(1, 4.0);

    // Query a point on the sphere just away from the north pole.
    const double    theta = 0.05;
    Eigen::Vector3d y(std::sin(theta), 0.0, std::cos(theta));

    const CurvedBasisWeights cw = evaluate_basis_curved(
        patches, patch_points, beta_patches,
        X, beta_lme, y,
        /*tol_shepard=*/1.0e-12,
        /*r_cut=*/100.0,
        /*newton_tol=*/1.0e-12,
        /*newton_max_iters=*/30);

    REQUIRE_THAT(total_weight(cw), WithinAbs(1.0, 1.0e-10));
    for (double v : cw.values) REQUIRE(v >= -1.0e-12);
}

TEST_CASE("lme::evaluate_basis_curved: reproduces a constant field on a sphere",
          "[shell][lme][basis_curved]")
{
    const auto X        = spherical_cap_19(0.30);
    const auto ids_all  = all_ids(static_cast<int>(X.rows()));
    Eigen::VectorXd beta_lme = Eigen::VectorXd::Constant(X.rows(), 50.0);

    std::vector<Patch> patches;
    patches.push_back(build_patch(/*anchor_id=*/0, X, ids_all, beta_lme));

    Eigen::MatrixXd patch_points(1, 3);
    patch_points.row(0) = X.row(0);
    Eigen::VectorXd beta_patches = Eigen::VectorXd::Constant(1, 4.0);

    // u_a ≡ (1.5, -0.5, 2.0)  for every node.
    const Eigen::Vector3d u_const(1.5, -0.5, 2.0);

    // Query at the centroid of a few ring-1 neighbours.
    Eigen::Vector3d y =
        0.5 * (X.row(1).transpose() + X.row(2).transpose());

    const CurvedBasisWeights cw = evaluate_basis_curved(
        patches, patch_points, beta_patches,
        X, beta_lme, y,
        /*tol_shepard=*/1.0e-12,
        /*r_cut=*/100.0,
        /*newton_tol=*/1.0e-12,
        /*newton_max_iters=*/30);

    Eigen::Vector3d u_h = Eigen::Vector3d::Zero();
    for (std::size_t k = 0; k < cw.indices.size(); ++k) {
        u_h += cw.values[k] * u_const;
    }
    REQUIRE_THAT((u_h - u_const).norm(), WithinAbs(0.0, 1.0e-10));
}

TEST_CASE("lme::evaluate_basis_curved: two-patch flat plate matches 2D LME",
          "[shell][lme][basis_curved]")
{
    // Two patches on the same flat 5x5 grid; both cover every node. The
    // Shepard PoU blends two charts that may differ only in V's sign
    // convention. The result must still equal the canonical 2D LME.
    const auto X        = flat_5x5_grid();
    const auto ids_all  = all_ids(static_cast<int>(X.rows()));
    const double h      = 0.5;
    Eigen::VectorXd beta_lme =
        Eigen::VectorXd::Constant(X.rows(), 1.6 / (h * h));

    std::vector<Patch> patches;
    patches.push_back(build_patch(/*anchor_id=*/12, X, ids_all, beta_lme));
    patches.push_back(build_patch(/*anchor_id=*/0,  X, ids_all, beta_lme));

    Eigen::MatrixXd patch_points(2, 3);
    patch_points.row(0) = X.row(12);
    patch_points.row(1) = X.row(0);
    Eigen::VectorXd beta_patches(2);
    beta_patches << 1.0, 1.0;

    Eigen::Vector3d y(0.18, 0.07, 0.0);

    const CurvedBasisWeights cw = evaluate_basis_curved(
        patches, patch_points, beta_patches,
        X, beta_lme, y,
        /*tol_shepard=*/1.0e-12,
        /*r_cut=*/100.0,
        /*newton_tol=*/1.0e-12,
        /*newton_max_iters=*/30);

    Eigen::MatrixXd X2d = X.leftCols(2);
    Eigen::VectorXd y2d(2);  y2d << y.x(), y.y();
    const auto bv2d = evaluate_basis(X2d, beta_lme, y2d, /*r_cut=*/100.0);

    REQUIRE(cw.indices.size() == bv2d.indices.size());
    for (std::size_t k = 0; k < bv2d.indices.size(); ++k) {
        const int    node_id = bv2d.indices[k];
        const double p2d     = bv2d.values[k];
        const double pc      = weight_at(cw, node_id);
        REQUIRE_THAT(pc, WithinAbs(p2d, 1.0e-10));
    }
}

TEST_CASE("lme::evaluate_basis_curved: empty patch vector throws",
          "[shell][lme][basis_curved]")
{
    const auto X        = flat_5x5_grid();
    Eigen::VectorXd beta_lme = Eigen::VectorXd::Constant(X.rows(), 6.4);
    Eigen::Vector3d y(0.0, 0.0, 0.0);
    std::vector<Patch> empty;
    Eigen::MatrixXd    pq(0, 3);
    Eigen::VectorXd    bq(0);

    REQUIRE_THROWS_AS(
        evaluate_basis_curved(empty, pq, bq, X, beta_lme, y),
        std::invalid_argument);
}
