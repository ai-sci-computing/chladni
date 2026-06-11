/**
 * @file test_lme_nodal_gaps.cpp
 * @brief Unit tests for @ref chladni::shell::lme::compute_nodal_gaps,
 *        the per-node 2nd-order SME slack-matrix builder (Rosolen-Millán-
 *        Arroyo 2013 §3.2.2).
 *
 * Coverage:
 *
 *  1. Per-kind formula checks — one synthetic @ref NodalGap per
 *     @ref NodalGapKind, each compared against the paper's exact
 *     expression. Verifies the dispatch table.
 *
 *  2. Topology checks on real meshes:
 *      - **Closed mesh**: every node @ref NodalGapKind::Interior →
 *        @f$ (\alpha/4) h^2 I @f$.
 *      - **Strip mesh**: long-edge nodes @ref BoundaryEdgeMid → tangent
 *        slack only; one ring inward @ref NearOneBoundaryEdge → tangent
 *        slack + boundary-normal slack.
 *      - **Anisotropic spacing**: eigendecomposition of
 *        @f$ d_a @f$ recovers the supplied principal axes
 *        @f$ (h_a^i, v_a^i) @f$.
 *
 *  3. Structural invariants — all outputs symmetric and PSD; output
 *     length matches input length; empty input → empty output.
 *
 *  4. Input validation — @f$ \alpha \le 1 @f$, @f$ \beta < 1 @f$,
 *     non-positive spacings, non-unit vectors, non-orthogonal
 *     anisotropic basis.
 *
 * Tests use modest absolute tolerances (1e-12) — the construction is
 * pure linear algebra and has no Newton/quadrature error.
 */

#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <stdexcept>
#include <vector>

using chladni::shell::lme::NodalGap;
using chladni::shell::lme::NodalGapKind;
using chladni::shell::lme::compute_nodal_gaps;

namespace {

constexpr double kTol = 1e-12;

/// True iff @c M is symmetric to absolute tolerance @c tol.
bool is_symmetric(const Eigen::Matrix2d& M, double tol = kTol) {
    return (M - M.transpose()).cwiseAbs().maxCoeff() <= tol;
}

/// True iff @c M is symmetric and has both eigenvalues @c >= -tol.
bool is_psd(const Eigen::Matrix2d& M, double tol = kTol) {
    if (!is_symmetric(M, tol)) return false;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(M);
    return es.eigenvalues().minCoeff() >= -tol;
}

}  // namespace

TEST_CASE("compute_nodal_gaps: empty input yields empty output",
          "[lme][nodal_gaps][validation]")
{
    const auto out = compute_nodal_gaps({}, /*alpha=*/4.0, /*beta=*/1.0);
    REQUIRE(out.empty());
}

TEST_CASE("compute_nodal_gaps: Interior — (alpha/4) h^2 I",
          "[lme][nodal_gaps]")
{
    const double alpha = 4.0;
    const double beta  = 1.0;
    const double h     = 0.25;

    NodalGap g;
    g.kind = NodalGapKind::Interior;
    g.h    = h;

    const auto out = compute_nodal_gaps({g}, alpha, beta);
    REQUIRE(out.size() == 1);

    const Eigen::Matrix2d expected =
        (alpha / 4.0) * h * h * Eigen::Matrix2d::Identity();
    REQUIRE((out[0] - expected).cwiseAbs().maxCoeff() < kTol);
    REQUIRE(is_psd(out[0]));
}

TEST_CASE("compute_nodal_gaps: BoundaryCorner — d = 0",
          "[lme][nodal_gaps]")
{
    NodalGap g;
    g.kind = NodalGapKind::BoundaryCorner;
    g.h    = 0.5;  // ignored

    const auto out = compute_nodal_gaps({g}, /*alpha=*/4.0, /*beta=*/1.0);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].cwiseAbs().maxCoeff() == 0.0);
    REQUIRE(is_psd(out[0]));
}

TEST_CASE("compute_nodal_gaps: BoundaryEdgeMid — (alpha/4) h^2 t t^T",
          "[lme][nodal_gaps]")
{
    const double alpha = 4.0;
    const double beta  = 1.0;
    const double h     = 0.2;
    const Eigen::Vector2d t(std::cos(0.3), std::sin(0.3));

    NodalGap g;
    g.kind = NodalGapKind::BoundaryEdgeMid;
    g.h    = h;
    g.t    = t;

    const auto out = compute_nodal_gaps({g}, alpha, beta);
    REQUIRE(out.size() == 1);

    const Eigen::Matrix2d expected = (alpha / 4.0) * h * h * (t * t.transpose());
    REQUIRE((out[0] - expected).cwiseAbs().maxCoeff() < kTol);
    REQUIRE(is_psd(out[0]));

    // Rank-1: one eigenvalue should be zero, the other (alpha/4) h^2.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(out[0]);
    REQUIRE(std::abs(es.eigenvalues()(0)) < kTol);
    REQUIRE(std::abs(es.eigenvalues()(1) - (alpha / 4.0) * h * h) < kTol);
}

TEST_CASE("compute_nodal_gaps: BoundaryEdgeNearCorner — beta h^2 t t^T",
          "[lme][nodal_gaps]")
{
    const double alpha = 4.0;
    const double beta  = 2.5;
    const double h     = 0.1;
    const Eigen::Vector2d t(0.0, 1.0);

    NodalGap g;
    g.kind = NodalGapKind::BoundaryEdgeNearCorner;
    g.h    = h;
    g.t    = t;

    const auto out = compute_nodal_gaps({g}, alpha, beta);
    REQUIRE(out.size() == 1);

    const Eigen::Matrix2d expected = beta * h * h * (t * t.transpose());
    REQUIRE((out[0] - expected).cwiseAbs().maxCoeff() < kTol);
    REQUIRE(is_psd(out[0]));
}

TEST_CASE("compute_nodal_gaps: NearOneBoundaryEdge — beta h^2 n n^T + (alpha/4) h^2 t t^T",
          "[lme][nodal_gaps]")
{
    const double alpha = 4.0;
    const double beta  = 1.5;
    const double h     = 0.3;
    // Pick a tangent and a perpendicular normal.
    const Eigen::Vector2d t(1.0, 0.0);
    const Eigen::Vector2d n(0.0, 1.0);

    NodalGap g;
    g.kind = NodalGapKind::NearOneBoundaryEdge;
    g.h    = h;
    g.t    = t;
    g.n    = n;

    const auto out = compute_nodal_gaps({g}, alpha, beta);
    REQUIRE(out.size() == 1);

    const Eigen::Matrix2d expected =
          beta            * h * h * (n * n.transpose())
        + (alpha / 4.0)   * h * h * (t * t.transpose());
    REQUIRE((out[0] - expected).cwiseAbs().maxCoeff() < kTol);
    REQUIRE(is_psd(out[0]));

    // With t perpendicular to n, eigenvalues should be exactly beta h^2
    // (along n) and (alpha/4) h^2 (along t).
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(out[0]);
    const double e_lo = es.eigenvalues()(0);
    const double e_hi = es.eigenvalues()(1);
    const double v_n  = beta            * h * h;
    const double v_t  = (alpha / 4.0)   * h * h;
    const double lo   = std::min(v_n, v_t);
    const double hi   = std::max(v_n, v_t);
    REQUIRE(std::abs(e_lo - lo) < kTol);
    REQUIRE(std::abs(e_hi - hi) < kTol);
}

TEST_CASE("compute_nodal_gaps: NearTwoBoundaryEdges — beta h^2 (n n^T + n' n'^T)",
          "[lme][nodal_gaps]")
{
    const double alpha = 4.0;
    const double beta  = 1.0;
    const double h     = 0.2;
    // A 90-degree wedge: two outward normals at right angles.
    const Eigen::Vector2d n (1.0, 0.0);
    const Eigen::Vector2d n2(0.0, 1.0);

    NodalGap g;
    g.kind = NodalGapKind::NearTwoBoundaryEdges;
    g.h    = h;
    g.n    = n;
    g.n2   = n2;

    const auto out = compute_nodal_gaps({g}, alpha, beta);
    REQUIRE(out.size() == 1);

    const Eigen::Matrix2d expected =
        beta * h * h * (n * n.transpose() + n2 * n2.transpose());
    REQUIRE((out[0] - expected).cwiseAbs().maxCoeff() < kTol);
    REQUIRE(is_psd(out[0]));

    // n, n' perpendicular → d = beta h^2 I, both eigenvalues equal.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(out[0]);
    REQUIRE(std::abs(es.eigenvalues()(0) - beta * h * h) < kTol);
    REQUIRE(std::abs(es.eigenvalues()(1) - beta * h * h) < kTol);
}

TEST_CASE("compute_nodal_gaps: NearTwoBoundaryEdges — non-perpendicular normals",
          "[lme][nodal_gaps]")
{
    // 60-degree wedge: normals at 0 and pi/3 radians (interior angle
    // 120 degrees, exterior angle 60). Eigenstructure is non-trivial
    // but the construction still yields a symmetric PSD matrix.
    const double alpha = 4.0;
    const double beta  = 1.0;
    const double h     = 0.1;
    const Eigen::Vector2d n (1.0, 0.0);
    const Eigen::Vector2d n2(std::cos(M_PI / 3.0), std::sin(M_PI / 3.0));

    NodalGap g;
    g.kind = NodalGapKind::NearTwoBoundaryEdges;
    g.h    = h;
    g.n    = n;
    g.n2   = n2;

    const auto out = compute_nodal_gaps({g}, alpha, beta);
    REQUIRE(out.size() == 1);
    REQUIRE(is_psd(out[0]));
    REQUIRE(is_symmetric(out[0]));

    // Trace equals beta h^2 (|n|^2 + |n2|^2) = 2 beta h^2.
    REQUIRE(std::abs(out[0].trace() - 2.0 * beta * h * h) < kTol);
}

TEST_CASE("compute_nodal_gaps: InteriorAnisotropic — eigendecomposition recovery",
          "[lme][nodal_gaps]")
{
    // Plan-mandated: feed an arbitrary anisotropic spacing tensor and
    // verify the output's eigendecomposition recovers the supplied
    // principal axes.
    const double alpha = 4.0;
    const double beta  = 1.0;
    const double h1    = 0.4;
    const double h2    = 0.1;
    const double theta = 0.7;  // arbitrary tilt of the principal axes
    const Eigen::Vector2d v1( std::cos(theta), std::sin(theta));
    const Eigen::Vector2d v2(-std::sin(theta), std::cos(theta));

    NodalGap g;
    g.kind = NodalGapKind::InteriorAnisotropic;
    g.h1   = h1;
    g.h2   = h2;
    g.v1   = v1;
    g.v2   = v2;

    const auto out = compute_nodal_gaps({g}, alpha, beta);
    REQUIRE(out.size() == 1);
    REQUIRE(is_psd(out[0]));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(out[0]);
    // Eigen sorts eigenvalues ascending; h2 < h1 here so e(0) is from v2.
    const double lam_lo = es.eigenvalues()(0);
    const double lam_hi = es.eigenvalues()(1);
    REQUIRE(std::abs(lam_lo - (alpha / 4.0) * h2 * h2) < kTol);
    REQUIRE(std::abs(lam_hi - (alpha / 4.0) * h1 * h1) < kTol);

    // The eigenvector for lam_hi must be v1 up to sign.
    const Eigen::Vector2d ev_hi = es.eigenvectors().col(1);
    REQUIRE(std::min((ev_hi - v1).norm(), (ev_hi + v1).norm()) < kTol);
    const Eigen::Vector2d ev_lo = es.eigenvectors().col(0);
    REQUIRE(std::min((ev_lo - v2).norm(), (ev_lo + v2).norm()) < kTol);
}

TEST_CASE("compute_nodal_gaps: closed mesh — every node interior-isotropic",
          "[lme][nodal_gaps]")
{
    // Plan-mandated: a closed-manifold chart has no boundary, so every
    // node is interior-isotropic. Build a synthetic batch of 25 nodes
    // with varying h_a and verify the output matches the formula
    // index-wise.
    const double alpha = 4.0;
    const double beta  = 1.0;

    std::vector<NodalGap> info;
    info.reserve(25);
    for (int i = 0; i < 25; ++i) {
        NodalGap g;
        g.kind = NodalGapKind::Interior;
        g.h    = 0.05 + 0.01 * static_cast<double>(i);  // 0.05 .. 0.29
        info.push_back(g);
    }

    const auto out = compute_nodal_gaps(info, alpha, beta);
    REQUIRE(out.size() == info.size());

    for (std::size_t k = 0; k < info.size(); ++k) {
        const double h = info[k].h;
        const Eigen::Matrix2d expected =
            (alpha / 4.0) * h * h * Eigen::Matrix2d::Identity();
        REQUIRE((out[k] - expected).cwiseAbs().maxCoeff() < kTol);
        REQUIRE(is_psd(out[k]));
    }
}

TEST_CASE("compute_nodal_gaps: strip mesh — boundary + near-boundary classification",
          "[lme][nodal_gaps]")
{
    // Plan-mandated: idealised "strip" mesh — a long rectangle infinitely
    // extended in y, finite in x. The boundary consists of two parallel
    // edges at x = +/- L/2 with outward normals +/- e_x and tangents
    // +/- e_y. We model:
    //   - 5 nodes on the bottom (y = -L/2) boundary edge: BoundaryEdgeMid
    //     with t = +e_x (tangent along the edge).
    //   - 5 nodes one ring inward (y = -L/2 + h): NearOneBoundaryEdge
    //     with n = -e_y (outward normal of the bottom edge), t = +e_x.
    //
    // Verify the formulas dispatch correctly across the population.
    const double alpha = 4.0;
    const double beta  = 1.5;
    const double h     = 0.1;
    const Eigen::Vector2d t( 1.0,  0.0);  // along the strip
    const Eigen::Vector2d n( 0.0, -1.0);  // outward from the bottom edge

    constexpr int kBdryNodes  = 5;
    constexpr int kNearNodes  = 5;
    std::vector<NodalGap> info;
    info.reserve(kBdryNodes + kNearNodes);

    for (int i = 0; i < kBdryNodes; ++i) {
        NodalGap g;
        g.kind = NodalGapKind::BoundaryEdgeMid;
        g.h    = h;
        g.t    = t;
        info.push_back(g);
    }
    for (int i = 0; i < kNearNodes; ++i) {
        NodalGap g;
        g.kind = NodalGapKind::NearOneBoundaryEdge;
        g.h    = h;
        g.t    = t;
        g.n    = n;
        info.push_back(g);
    }

    const auto out = compute_nodal_gaps(info, alpha, beta);
    REQUIRE(out.size() == info.size());

    const Eigen::Matrix2d d_bdry =
        (alpha / 4.0) * h * h * (t * t.transpose());
    const Eigen::Matrix2d d_near =
          beta          * h * h * (n * n.transpose())
        + (alpha / 4.0) * h * h * (t * t.transpose());

    for (int i = 0; i < kBdryNodes; ++i) {
        REQUIRE((out[static_cast<std::size_t>(i)] - d_bdry)
                    .cwiseAbs().maxCoeff() < kTol);
        REQUIRE(is_psd(out[static_cast<std::size_t>(i)]));
    }
    for (int i = 0; i < kNearNodes; ++i) {
        const std::size_t k = static_cast<std::size_t>(kBdryNodes + i);
        REQUIRE((out[k] - d_near).cwiseAbs().maxCoeff() < kTol);
        REQUIRE(is_psd(out[k]));
    }
}

TEST_CASE("compute_nodal_gaps: mixed kinds preserve index alignment",
          "[lme][nodal_gaps]")
{
    // Build a batch covering all 7 kinds. Verify each output matches the
    // formula for its corresponding input index — guards against any
    // accidental reordering or off-by-one in the dispatch loop.
    const double alpha = 4.0;
    const double beta  = 1.5;
    const double h     = 0.2;

    std::vector<NodalGap> info;
    info.reserve(7);

    NodalGap g_int;     g_int.kind = NodalGapKind::Interior;          g_int.h = h;
    NodalGap g_aniso;   g_aniso.kind = NodalGapKind::InteriorAnisotropic;
                        g_aniso.h1 = 0.3; g_aniso.h2 = 0.1;
                        g_aniso.v1 = Eigen::Vector2d(1.0, 0.0);
                        g_aniso.v2 = Eigen::Vector2d(0.0, 1.0);
    NodalGap g_corner;  g_corner.kind = NodalGapKind::BoundaryCorner; g_corner.h = h;
    NodalGap g_emid;    g_emid.kind = NodalGapKind::BoundaryEdgeMid;  g_emid.h = h;
                        g_emid.t = Eigen::Vector2d(1.0, 0.0);
    NodalGap g_enear;   g_enear.kind = NodalGapKind::BoundaryEdgeNearCorner;
                        g_enear.h = h; g_enear.t = Eigen::Vector2d(0.0, 1.0);
    NodalGap g_n1;      g_n1.kind = NodalGapKind::NearOneBoundaryEdge;
                        g_n1.h = h; g_n1.t = Eigen::Vector2d(1.0, 0.0);
                        g_n1.n = Eigen::Vector2d(0.0, 1.0);
    NodalGap g_n2;      g_n2.kind = NodalGapKind::NearTwoBoundaryEdges;
                        g_n2.h = h; g_n2.n  = Eigen::Vector2d(1.0, 0.0);
                        g_n2.n2 = Eigen::Vector2d(0.0, 1.0);

    info.push_back(g_int);
    info.push_back(g_aniso);
    info.push_back(g_corner);
    info.push_back(g_emid);
    info.push_back(g_enear);
    info.push_back(g_n1);
    info.push_back(g_n2);

    const auto out = compute_nodal_gaps(info, alpha, beta);
    REQUIRE(out.size() == 7);
    for (const auto& M : out) {
        REQUIRE(is_psd(M));
    }

    // Spot-check each by trace (full matrix already tested elsewhere).
    REQUIRE(std::abs(out[0].trace() - 2.0 * (alpha / 4.0) * h * h) < kTol);
    REQUIRE(std::abs(out[1].trace()
                      - (alpha / 4.0) * (0.3 * 0.3 + 0.1 * 0.1)) < kTol);
    REQUIRE(out[2].cwiseAbs().maxCoeff() == 0.0);
    REQUIRE(std::abs(out[3].trace() - (alpha / 4.0) * h * h) < kTol);
    REQUIRE(std::abs(out[4].trace() - beta * h * h) < kTol);
    REQUIRE(std::abs(out[5].trace()
                      - (beta + alpha / 4.0) * h * h) < kTol);
    REQUIRE(std::abs(out[6].trace() - 2.0 * beta * h * h) < kTol);
}

// =====================================================================
// Input validation
// =====================================================================

TEST_CASE("compute_nodal_gaps: alpha must be > 1",
          "[lme][nodal_gaps][validation]")
{
    NodalGap g;
    g.kind = NodalGapKind::Interior;
    g.h    = 0.1;

    REQUIRE_THROWS_AS(compute_nodal_gaps({g}, /*alpha=*/1.0, /*beta=*/1.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(compute_nodal_gaps({g}, /*alpha=*/0.5, /*beta=*/1.0),
                      std::invalid_argument);
}

TEST_CASE("compute_nodal_gaps: beta must be >= 1",
          "[lme][nodal_gaps][validation]")
{
    NodalGap g;
    g.kind = NodalGapKind::Interior;
    g.h    = 0.1;

    REQUIRE_THROWS_AS(compute_nodal_gaps({g}, /*alpha=*/4.0, /*beta=*/0.9),
                      std::invalid_argument);
    REQUIRE_NOTHROW(compute_nodal_gaps({g}, /*alpha=*/4.0, /*beta=*/1.0));
}

TEST_CASE("compute_nodal_gaps: non-positive spacing throws",
          "[lme][nodal_gaps][validation]")
{
    {
        NodalGap g;
        g.kind = NodalGapKind::Interior;
        g.h    = 0.0;
        REQUIRE_THROWS_AS(compute_nodal_gaps({g}, 4.0, 1.0),
                          std::invalid_argument);
    }
    {
        NodalGap g;
        g.kind = NodalGapKind::Interior;
        g.h    = -0.1;
        REQUIRE_THROWS_AS(compute_nodal_gaps({g}, 4.0, 1.0),
                          std::invalid_argument);
    }
    // BoundaryCorner ignores h, so zero h is fine.
    {
        NodalGap g;
        g.kind = NodalGapKind::BoundaryCorner;
        g.h    = 0.0;
        REQUIRE_NOTHROW(compute_nodal_gaps({g}, 4.0, 1.0));
    }
}

TEST_CASE("compute_nodal_gaps: non-unit tangent throws",
          "[lme][nodal_gaps][validation]")
{
    NodalGap g;
    g.kind = NodalGapKind::BoundaryEdgeMid;
    g.h    = 0.1;
    g.t    = Eigen::Vector2d(2.0, 0.0);  // length 2, not unit

    REQUIRE_THROWS_AS(compute_nodal_gaps({g}, 4.0, 1.0),
                      std::invalid_argument);
}

TEST_CASE("compute_nodal_gaps: non-unit normal throws",
          "[lme][nodal_gaps][validation]")
{
    NodalGap g;
    g.kind = NodalGapKind::NearOneBoundaryEdge;
    g.h    = 0.1;
    g.t    = Eigen::Vector2d(1.0, 0.0);
    g.n    = Eigen::Vector2d(0.0, 0.5);  // length 0.5, not unit

    REQUIRE_THROWS_AS(compute_nodal_gaps({g}, 4.0, 1.0),
                      std::invalid_argument);
}

TEST_CASE("compute_nodal_gaps: anisotropic non-orthogonal basis throws",
          "[lme][nodal_gaps][validation]")
{
    NodalGap g;
    g.kind = NodalGapKind::InteriorAnisotropic;
    g.h1   = 0.3;
    g.h2   = 0.1;
    g.v1   = Eigen::Vector2d(1.0, 0.0);
    // v2 not orthogonal to v1.
    g.v2   = Eigen::Vector2d(std::cos(0.1), std::sin(0.1));

    REQUIRE_THROWS_AS(compute_nodal_gaps({g}, 4.0, 1.0),
                      std::invalid_argument);
}

TEST_CASE("compute_nodal_gaps: anisotropic non-positive h_i throws",
          "[lme][nodal_gaps][validation]")
{
    {
        NodalGap g;
        g.kind = NodalGapKind::InteriorAnisotropic;
        g.h1   = 0.0;
        g.h2   = 0.1;
        g.v1   = Eigen::Vector2d(1.0, 0.0);
        g.v2   = Eigen::Vector2d(0.0, 1.0);
        REQUIRE_THROWS_AS(compute_nodal_gaps({g}, 4.0, 1.0),
                          std::invalid_argument);
    }
    {
        NodalGap g;
        g.kind = NodalGapKind::InteriorAnisotropic;
        g.h1   = 0.3;
        g.h2   = -0.1;
        g.v1   = Eigen::Vector2d(1.0, 0.0);
        g.v2   = Eigen::Vector2d(0.0, 1.0);
        REQUIRE_THROWS_AS(compute_nodal_gaps({g}, 4.0, 1.0),
                          std::invalid_argument);
    }
}
