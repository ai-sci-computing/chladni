/**
 * @file test_lme_sme_basis.cpp
 * @brief Unit tests for @ref chladni::shell::lme::evaluate_sme_basis,
 *        the 2D 2nd-order SME basis evaluator (Rosolen-Millán-Arroyo
 *        2013 §3.3).
 *
 * Fixture: a 5x5 uniform grid on @f$[-1, 1]^2@f$ with @f$ h = 0.5 @f$,
 * classified per paper Fig. 6 — corners get @f$ d_a = 0 @f$, boundary
 * edge nodes get rank-1 tangential slack, near-boundary nodes get slack
 * with @f$ \beta @f$ along the normal + @f$ \alpha/4 @f$ along the
 * tangent, and the single pure-interior centre gets isotropic
 * @f$ (\alpha/4) h^2 I @f$. The classification is built directly with
 * @ref chladni::shell::lme::compute_nodal_gaps (already shipped in
 * Phase A.1) so we test the full SME pipeline at once.
 *
 * Coverage:
 *
 *  1. **Partition of unity** (PoU): @f$ \sum_a s_a = 1 @f$ to @c 1e-12
 *     at several interior query points.
 *  2. **1st-order moment**: @f$ \sum_a s_a (x - x_a) = 0 @f$ at the
 *     Newton tolerance.
 *  3. **2nd-order moment with @f$ d_a @f$**: @f$ \sum_a s_a D_a = 0 @f$
 *     at the Newton tolerance, where
 *     @f$ D_a = (x_a - x) \otimes (x_a - x) - d_a @f$.
 *  4. **Weak Kronecker-delta at corners**: at @f$ x = x_b @f$ with
 *     @f$ d_b = 0 @f$, @f$ s_a(x_b) = \delta_{ab} @f$.
 *  5. **Nonnegativity**: every @f$ s_a \ge 0 @f$ at every query point.
 *  6. **Reproduction of an arbitrary quadratic via least-squares**
 *     (paper §3.4.2): the SME basis can fit @f$ u(x) = c + b \cdot x +
 *     x^\top C x @f$ to machine precision; the LSQ residual on a dense
 *     grid of query points is @c < 1e-10 in relative norm.
 *  7. **Degenerate all-zero @f$ d_a @f$ throws**: Newton diverges
 *     because the moment-space convex hull collapses to a measure-zero
 *     set (paper §3.1 — the canonical 2nd-order constraint is
 *     infeasible).
 *  8. **Input validation**: empty nodes, mismatched @c d.size(), wrong
 *     query dimension, non-symmetric @f$ d_a @f$, non-positive
 *     @c r_cut.
 */

#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <vector>

using chladni::shell::LMEBasisAndGrad;
using chladni::shell::LMEBasisGradHess;
using chladni::shell::LMEBasisValues;
using chladni::shell::lme::NodalGap;
using chladni::shell::lme::NodalGapKind;
using chladni::shell::lme::compute_nodal_gaps;
using chladni::shell::lme::evaluate_sme_basis;
using chladni::shell::lme::evaluate_sme_basis_and_grad;
using chladni::shell::lme::evaluate_sme_basis_grad_and_hess;
using chladni::shell::lme::evaluate_sme_basis_grad_and_hess_closed_form;

namespace {

constexpr int    kN     = 5;          ///< Grid side (5x5 = 25 nodes).
constexpr double kH     = 0.5;        ///< Spacing.
constexpr double kAlpha = 4.0;        ///< Paper §3.4.1 recommended interior slack.
constexpr double kBeta  = 1.0;        ///< Paper default boundary slack factor.

/// Linear index into a kN-by-kN grid.
inline int idx(int i, int j) { return j * kN + i; }

/// Build a uniform kN x kN grid on @f$[-1, 1]^2@f$ as an N×2 matrix
/// in column-major-of-rows order (row k is node k).
Eigen::MatrixXd make_grid() {
    Eigen::MatrixXd nodes(kN * kN, 2);
    for (int j = 0; j < kN; ++j) {
        for (int i = 0; i < kN; ++i) {
            nodes(idx(i, j), 0) = -1.0 + static_cast<double>(i) * kH;
            nodes(idx(i, j), 1) = -1.0 + static_cast<double>(j) * kH;
        }
    }
    return nodes;
}

/// Build the per-paper Fig. 6 classification for the 5x5 grid.
/// Returns a vector of NodalGap that compute_nodal_gaps turns into
/// the d_a matrices. The grid corners are at (i, j) ∈ {0, 4}².
std::vector<NodalGap> classify_grid() {
    std::vector<NodalGap> info(kN * kN);

    const Eigen::Vector2d ex(1.0, 0.0);
    const Eigen::Vector2d ey(0.0, 1.0);

    for (int j = 0; j < kN; ++j) {
        for (int i = 0; i < kN; ++i) {
            NodalGap& g = info[static_cast<std::size_t>(idx(i, j))];
            g.h = kH;

            const bool on_left   = (i == 0);
            const bool on_right  = (i == kN - 1);
            const bool on_bottom = (j == 0);
            const bool on_top    = (j == kN - 1);
            const bool on_bdry_x = on_left || on_right;
            const bool on_bdry_y = on_bottom || on_top;

            if (on_bdry_x && on_bdry_y) {
                // Boundary corner.
                g.kind = NodalGapKind::BoundaryCorner;
            } else if (on_bdry_x) {
                // Left or right edge — tangent along y.
                g.t = ey;
                // Adjacent-to-corner cells are j == 1 or j == kN-2.
                if (j == 1 || j == kN - 2) {
                    g.kind = NodalGapKind::BoundaryEdgeNearCorner;
                } else {
                    g.kind = NodalGapKind::BoundaryEdgeMid;
                }
            } else if (on_bdry_y) {
                // Bottom or top edge — tangent along x.
                g.t = ex;
                if (i == 1 || i == kN - 2) {
                    g.kind = NodalGapKind::BoundaryEdgeNearCorner;
                } else {
                    g.kind = NodalGapKind::BoundaryEdgeMid;
                }
            } else {
                // Interior (i, j ∈ {1, 2, 3} for a 5x5 grid).
                const bool near_left   = (i == 1);
                const bool near_right  = (i == kN - 2);
                const bool near_bottom = (j == 1);
                const bool near_top    = (j == kN - 2);
                const bool near_x      = near_left || near_right;
                const bool near_y      = near_bottom || near_top;

                if (near_x && near_y) {
                    g.kind = NodalGapKind::NearTwoBoundaryEdges;
                    g.n  = near_left   ? -ex : ex;   // outward of x-bdry
                    g.n2 = near_bottom ? -ey : ey;   // outward of y-bdry
                } else if (near_x) {
                    g.kind = NodalGapKind::NearOneBoundaryEdge;
                    g.n = near_left ? -ex : ex;
                    g.t = ey;
                } else if (near_y) {
                    g.kind = NodalGapKind::NearOneBoundaryEdge;
                    g.n = near_bottom ? -ey : ey;
                    g.t = ex;
                } else {
                    // Pure interior — only the centre on a 5x5 grid.
                    g.kind = NodalGapKind::Interior;
                }
            }
        }
    }
    return info;
}

/// Bundle: nodes + d_a vector built from the paper-Fig. 6 classification.
struct GridFixture {
    Eigen::MatrixXd               nodes;  ///< 25 x 2
    std::vector<Eigen::Matrix2d>  d;      ///< length 25
};

GridFixture build_fixture() {
    GridFixture f;
    f.nodes = make_grid();
    f.d     = compute_nodal_gaps(classify_grid(), kAlpha, kBeta);
    return f;
}

/// Sum of values in a sparse basis: PoU check helper.
double basis_sum(const LMEBasisValues& bv) {
    double s = 0.0;
    for (double v : bv.values) s += v;
    return s;
}

/// 1st-order moment Σ s_a (x - x_a). Returns the residual vector.
Eigen::Vector2d first_moment(
    const LMEBasisValues& bv,
    const Eigen::MatrixXd& nodes,
    const Eigen::Vector2d& x)
{
    Eigen::Vector2d m = Eigen::Vector2d::Zero();
    for (std::size_t k = 0; k < bv.indices.size(); ++k) {
        const int a = bv.indices[k];
        m += bv.values[k] * (x - nodes.row(a).transpose());
    }
    return m;
}

/// 2nd-order centered moment Σ s_a [(x_a - x)⊗(x_a - x) - d_a].
Eigen::Matrix2d second_moment_residual(
    const LMEBasisValues& bv,
    const Eigen::MatrixXd& nodes,
    const std::vector<Eigen::Matrix2d>& d,
    const Eigen::Vector2d& x)
{
    Eigen::Matrix2d M = Eigen::Matrix2d::Zero();
    for (std::size_t k = 0; k < bv.indices.size(); ++k) {
        const int a = bv.indices[k];
        const Eigen::Vector2d u_a = nodes.row(a).transpose() - x;
        M += bv.values[k] *
             (u_a * u_a.transpose() - d[static_cast<std::size_t>(a)]);
    }
    return M;
}

}  // namespace

// =====================================================================
// Convergence + per-query property tests
// =====================================================================

TEST_CASE("evaluate_sme_basis: partition of unity",
          "[lme][sme_basis]")
{
    const auto f = build_fixture();
    const double r_cut = 10.0;   // generous; whole grid in active set

    // Five interior query points.
    const std::vector<Eigen::Vector2d> queries = {
        {0.0, 0.0},
        {0.3, 0.2},
        {-0.4, 0.1},
        {0.7, -0.6},
        {-0.25, -0.75},
    };

    for (const auto& x : queries) {
        const auto bv = evaluate_sme_basis(f.nodes, f.d, x, r_cut);
        REQUIRE(std::abs(basis_sum(bv) - 1.0) < 1e-12);
    }
}

TEST_CASE("evaluate_sme_basis: 1st-order moment is zero",
          "[lme][sme_basis]")
{
    const auto f = build_fixture();
    const double r_cut = 10.0;
    const double tol   = 1e-9;  // Newton default is 1e-10

    const std::vector<Eigen::Vector2d> queries = {
        {0.0, 0.0},
        {0.3, 0.2},
        {-0.4, 0.1},
        {0.7, -0.6},
    };

    for (const auto& x : queries) {
        const auto bv = evaluate_sme_basis(f.nodes, f.d, x, r_cut);
        const auto m  = first_moment(bv, f.nodes, x);
        REQUIRE(m.lpNorm<Eigen::Infinity>() < tol);
    }
}

TEST_CASE("evaluate_sme_basis: 2nd-order moment with d_a is zero",
          "[lme][sme_basis]")
{
    const auto f = build_fixture();
    const double r_cut = 10.0;
    const double tol   = 1e-9;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0, 0.0},
        {0.3, 0.2},
        {-0.4, 0.1},
        {0.7, -0.6},
    };

    for (const auto& x : queries) {
        const auto bv = evaluate_sme_basis(f.nodes, f.d, x, r_cut);
        const auto R  = second_moment_residual(bv, f.nodes, f.d, x);
        REQUIRE(R.cwiseAbs().maxCoeff() < tol);
    }
}

TEST_CASE("evaluate_sme_basis: basis values are nonnegative",
          "[lme][sme_basis]")
{
    const auto f = build_fixture();
    const double r_cut = 10.0;

    for (double y = -0.9; y <= 0.9; y += 0.3) {
        for (double x = -0.9; x <= 0.9; x += 0.3) {
            const Eigen::Vector2d q(x, y);
            const auto bv = evaluate_sme_basis(f.nodes, f.d, q, r_cut);
            for (double v : bv.values) {
                REQUIRE(v >= 0.0);
            }
        }
    }
}

TEST_CASE("evaluate_sme_basis: weak Kronecker-delta at boundary corners",
          "[lme][sme_basis]")
{
    // The four grid corners have d_b = 0 by the SME boundary recipe.
    // Evaluating at x = x_b must yield s_a(x_b) = δ_{ab}.
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const std::vector<int> corner_ids = {
        idx(0, 0),
        idx(kN - 1, 0),
        idx(0, kN - 1),
        idx(kN - 1, kN - 1),
    };

    for (int b : corner_ids) {
        // Sanity: this is indeed a zero-gap corner in our fixture.
        REQUIRE(f.d[static_cast<std::size_t>(b)].cwiseAbs().maxCoeff()
                == 0.0);

        const Eigen::Vector2d xb = f.nodes.row(b).transpose();
        const auto bv = evaluate_sme_basis(f.nodes, f.d, xb, r_cut);

        // Expect exactly one active index, with value 1, at index b.
        REQUIRE(bv.indices.size() == 1);
        REQUIRE(bv.indices[0]  == b);
        REQUIRE(std::abs(bv.values[0] - 1.0) < 1e-14);
    }
}

// =====================================================================
// Polynomial reproduction (paper §3.4.2)
// =====================================================================

TEST_CASE("evaluate_sme_basis: LSQ-reproduces an arbitrary quadratic",
          "[lme][sme_basis]")
{
    // Paper §3.4.2 test: the SME basis spans the space of quadratics
    // exactly. Sample a known quadratic on a dense interior grid, fit
    // nodal coefficients by least squares, and verify the residual
    // vanishes to ~ machine precision.
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const int N_nodes = static_cast<int>(f.nodes.rows());

    // Known target quadratic (paper §3.4.2 uses similar form):
    //   u(x) = c0 + b·x + x^T C x   with arbitrary nonzero coefficients.
    const double           c0 = 0.7;
    const Eigen::Vector2d  bb(0.3, -0.4);
    Eigen::Matrix2d        CC;
    CC <<  0.5, -0.2,
          -0.2,  0.8;

    auto target = [&](const Eigen::Vector2d& x) {
        return c0 + bb.dot(x) + x.transpose() * CC * x;
    };

    // 7x7 sampling grid in (-0.8, 0.8)^2 — safely interior, away from
    // the boundary where the SME slack matrices degenerate.
    constexpr int M  = 7;
    constexpr int MM = M * M;
    Eigen::MatrixXd B(MM, N_nodes);
    Eigen::VectorXd u_q(MM);
    int row = 0;
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < M; ++i) {
            const double s = -0.8 + 1.6 * static_cast<double>(i) /
                                   static_cast<double>(M - 1);
            const double t = -0.8 + 1.6 * static_cast<double>(j) /
                                   static_cast<double>(M - 1);
            const Eigen::Vector2d xq(s, t);
            const auto bv = evaluate_sme_basis(f.nodes, f.d, xq, r_cut);

            B.row(row).setZero();
            for (std::size_t k = 0; k < bv.indices.size(); ++k) {
                B(row, bv.indices[k]) = bv.values[k];
            }
            u_q(row) = target(xq);
            ++row;
        }
    }

    // Solve least-squares B u_a = u_q via thin SVD (robust to the
    // rank-deficient pure-Newton-basis-matrix that some boundary-class
    // columns might present).
    const Eigen::VectorXd u_a =
        B.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(u_q);
    const Eigen::VectorXd residual = B * u_a - u_q;

    // Paper §3.4.2 reports "error within machine precision". On a
    // 5x5 grid with M=49 sample points the residual is dominated by
    // the basis condition number rather than the polynomial fit; we
    // require < 1e-10 relative to ||u_q||.
    const double rel_err = residual.norm() / u_q.norm();
    REQUIRE(rel_err < 1e-10);
}

// =====================================================================
// Degenerate + invalid inputs
// =====================================================================

TEST_CASE("evaluate_sme_basis: all-zero d_a → Newton diverges (throws)",
          "[lme][sme_basis][validation]")
{
    // The canonical 2nd-order constraint Σ s_a (x_a - x)⊗(x_a - x) = 0
    // is infeasible at any x distinct from a node (paper §3.1). Newton
    // on the dual is unbounded; the function must reach its iteration
    // cap and throw std::runtime_error.
    Eigen::MatrixXd nodes = make_grid();
    std::vector<Eigen::Matrix2d> d(static_cast<std::size_t>(nodes.rows()),
                                    Eigen::Matrix2d::Zero());

    REQUIRE_THROWS_AS(
        evaluate_sme_basis(nodes, d, Eigen::Vector2d(0.13, 0.07),
                           /*r_cut=*/10.0,
                           /*newton_tol=*/1e-10,
                           /*newton_max_iters=*/30),
        std::runtime_error);
}

TEST_CASE("evaluate_sme_basis: empty nodes throws",
          "[lme][sme_basis][validation]")
{
    Eigen::MatrixXd nodes(0, 2);
    std::vector<Eigen::Matrix2d> d;
    REQUIRE_THROWS_AS(
        evaluate_sme_basis(nodes, d, Eigen::Vector2d(0.0, 0.0), 1.0),
        std::invalid_argument);
}

TEST_CASE("evaluate_sme_basis: d.size() mismatch throws",
          "[lme][sme_basis][validation]")
{
    const auto f = build_fixture();
    auto d_short = f.d;
    d_short.pop_back();

    REQUIRE_THROWS_AS(
        evaluate_sme_basis(f.nodes, d_short,
                           Eigen::Vector2d(0.0, 0.0), 10.0),
        std::invalid_argument);
}

TEST_CASE("evaluate_sme_basis: 3D nodes throws (only 2D supported)",
          "[lme][sme_basis][validation]")
{
    Eigen::MatrixXd nodes_3d(4, 3);
    nodes_3d << 0, 0, 0,
                1, 0, 0,
                0, 1, 0,
                1, 1, 0;
    std::vector<Eigen::Matrix2d> d(4, Eigen::Matrix2d::Identity());

    REQUIRE_THROWS_AS(
        evaluate_sme_basis(nodes_3d, d,
                           Eigen::Vector2d(0.5, 0.5), 10.0),
        std::invalid_argument);
}

TEST_CASE("evaluate_sme_basis: query-x dimension mismatch throws",
          "[lme][sme_basis][validation]")
{
    const auto f = build_fixture();
    Eigen::Vector3d x_3d(0.0, 0.0, 0.0);
    REQUIRE_THROWS_AS(
        evaluate_sme_basis(f.nodes, f.d, x_3d, 10.0),
        std::invalid_argument);
}

TEST_CASE("evaluate_sme_basis: non-positive r_cut throws",
          "[lme][sme_basis][validation]")
{
    const auto f = build_fixture();
    REQUIRE_THROWS_AS(
        evaluate_sme_basis(f.nodes, f.d,
                           Eigen::Vector2d(0.0, 0.0), 0.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        evaluate_sme_basis(f.nodes, f.d,
                           Eigen::Vector2d(0.0, 0.0), -1.0),
        std::invalid_argument);
}

TEST_CASE("evaluate_sme_basis: asymmetric d_a throws",
          "[lme][sme_basis][validation]")
{
    Eigen::MatrixXd nodes = make_grid();
    auto d = compute_nodal_gaps(classify_grid(), kAlpha, kBeta);
    // Break symmetry on an interior matrix.
    d[static_cast<std::size_t>(idx(2, 2))](0, 1) += 0.001;

    REQUIRE_THROWS_AS(
        evaluate_sme_basis(nodes, d,
                           Eigen::Vector2d(0.0, 0.0), 10.0),
        std::invalid_argument);
}

// =====================================================================
// Phase A.3 — evaluate_sme_basis_and_grad
// =====================================================================

TEST_CASE("evaluate_sme_basis_and_grad: agrees with evaluate_sme_basis on indices/values",
          "[lme][sme_basis_and_grad]")
{
    // Same Newton solve as the value-only entry point — indices and
    // values must match to machine precision.
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.3,  0.2},
        {-0.4, 0.1},
        {0.7, -0.6},
    };

    for (const auto& x : queries) {
        const auto bv  = evaluate_sme_basis(f.nodes, f.d, x, r_cut);
        const auto bvg = evaluate_sme_basis_and_grad(f.nodes, f.d, x, r_cut);
        REQUIRE(bvg.indices == bv.indices);
        REQUIRE(bvg.values.size() == bv.values.size());
        for (std::size_t k = 0; k < bv.values.size(); ++k) {
            REQUIRE(std::abs(bvg.values[k] - bv.values[k]) < 1e-14);
        }
        REQUIRE(bvg.gradients.size() == bv.values.size());
        for (const auto& g : bvg.gradients) {
            REQUIRE(g.size() == 2);
        }
    }
}

TEST_CASE("evaluate_sme_basis_and_grad: sum of gradients is zero (PoU derivative)",
          "[lme][sme_basis_and_grad]")
{
    // Differentiating Σ s_a = 1 w.r.t. x gives Σ ∇s_a = 0 exactly.
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.3,  0.2},
        {-0.4, 0.1},
        {0.7, -0.6},
    };

    for (const auto& x : queries) {
        const auto bvg = evaluate_sme_basis_and_grad(f.nodes, f.d, x, r_cut);
        Eigen::Vector2d sum = Eigen::Vector2d::Zero();
        for (const auto& g : bvg.gradients) sum += g;
        // Inherits the Newton convergence tolerance (1e-10) through
        // the IFT solve; 1e-9 absorbs the propagation slack.
        REQUIRE(sum.lpNorm<Eigen::Infinity>() < 1e-9);
    }
}

TEST_CASE("evaluate_sme_basis_and_grad: Σ x_a ⊗ ∇s_a = I (linear-reproduction derivative)",
          "[lme][sme_basis_and_grad]")
{
    // Differentiating Σ s_a x_a = x w.r.t. x gives Σ x_a (∇s_a)^T = I.
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.3,  0.2},
        {-0.4, 0.1},
        {0.7, -0.6},
    };

    for (const auto& x : queries) {
        const auto bvg = evaluate_sme_basis_and_grad(f.nodes, f.d, x, r_cut);
        Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
        for (std::size_t k = 0; k < bvg.indices.size(); ++k) {
            const int a = bvg.indices[k];
            const Eigen::Vector2d x_a = f.nodes.row(a).transpose();
            S += x_a * bvg.gradients[k].transpose();
        }
        const Eigen::Matrix2d err = S - Eigen::Matrix2d::Identity();
        REQUIRE(err.cwiseAbs().maxCoeff() < 1e-9);
    }
}

TEST_CASE("evaluate_sme_basis_and_grad: matches central finite differences",
          "[lme][sme_basis_and_grad]")
{
    // Central FD on s_a(x ± h e_k) is O(h²) accurate; with h = 1e-4
    // on a smooth basis the FD error is ~ h² · ||∇²s_a|| ≈ 1e-8 for
    // well-conditioned interior queries. Compare to 1e-6 relative on
    // the largest component (loosen near corners where curvature is
    // larger).
    const auto f = build_fixture();
    const double r_cut = 10.0;
    const double h     = 1e-4;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.15, 0.27},
        {-0.31, 0.42},
        {0.45, -0.35},
    };

    auto values_at = [&](const Eigen::Vector2d& xq) {
        return evaluate_sme_basis(f.nodes, f.d, xq, r_cut).values;
    };

    for (const auto& x : queries) {
        const auto bvg = evaluate_sme_basis_and_grad(f.nodes, f.d, x, r_cut);

        // We must use evaluate_sme_basis's index ordering; it matches
        // bvg.indices by construction (compute_sme_state shared).
        const auto vals_pm_x =
            std::pair(values_at(x + Eigen::Vector2d(h, 0)),
                      values_at(x - Eigen::Vector2d(h, 0)));
        const auto vals_pm_y =
            std::pair(values_at(x + Eigen::Vector2d(0, h)),
                      values_at(x - Eigen::Vector2d(0, h)));

        for (std::size_t k = 0; k < bvg.indices.size(); ++k) {
            const double grad_x_fd =
                (vals_pm_x.first[k] - vals_pm_x.second[k]) / (2.0 * h);
            const double grad_y_fd =
                (vals_pm_y.first[k] - vals_pm_y.second[k]) / (2.0 * h);
            const Eigen::Vector2d grad_an = bvg.gradients[k];
            REQUIRE(std::abs(grad_an(0) - grad_x_fd) < 1e-6);
            REQUIRE(std::abs(grad_an(1) - grad_y_fd) < 1e-6);
        }
    }
}

TEST_CASE("evaluate_sme_basis_and_grad: throws at boundary-corner exact match",
          "[lme][sme_basis_and_grad][validation]")
{
    // At x = x_b with d_b = 0, evaluate_sme_basis returns δ_b; the
    // gradient is not defined there (basis is non-differentiable —
    // same as 1st-order LME at conv-hull corners).
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const Eigen::Vector2d xb = f.nodes.row(idx(0, 0)).transpose();
    REQUIRE_THROWS_AS(
        evaluate_sme_basis_and_grad(f.nodes, f.d, xb, r_cut),
        std::domain_error);
}

TEST_CASE("evaluate_sme_basis_and_grad: input validation",
          "[lme][sme_basis_and_grad][validation]")
{
    const auto f = build_fixture();

    // Empty nodes.
    {
        Eigen::MatrixXd empty(0, 2);
        std::vector<Eigen::Matrix2d> empty_d;
        REQUIRE_THROWS_AS(
            evaluate_sme_basis_and_grad(empty, empty_d,
                                        Eigen::Vector2d(0.0, 0.0), 1.0),
            std::invalid_argument);
    }
    // d.size() mismatch.
    {
        auto d_short = f.d;
        d_short.pop_back();
        REQUIRE_THROWS_AS(
            evaluate_sme_basis_and_grad(f.nodes, d_short,
                                        Eigen::Vector2d(0.0, 0.0), 10.0),
            std::invalid_argument);
    }
    // r_cut ≤ 0.
    {
        REQUIRE_THROWS_AS(
            evaluate_sme_basis_and_grad(f.nodes, f.d,
                                        Eigen::Vector2d(0.0, 0.0), -1.0),
            std::invalid_argument);
    }
}

// =====================================================================
// Phase A.4 — evaluate_sme_basis_grad_and_hess
// =====================================================================

TEST_CASE("evaluate_sme_basis_grad_and_hess: values + gradients agree with Phase A.3",
          "[lme][sme_basis_grad_hess]")
{
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.3,  0.2},
        {-0.4, 0.1},
        {0.45, -0.35},
    };

    for (const auto& x : queries) {
        const auto bvg = evaluate_sme_basis_and_grad(f.nodes, f.d, x, r_cut);
        const auto gh  = evaluate_sme_basis_grad_and_hess(f.nodes, f.d, x, r_cut);

        REQUIRE(gh.indices == bvg.indices);
        REQUIRE(gh.values.size()    == bvg.values.size());
        REQUIRE(gh.gradients.size() == bvg.gradients.size());
        REQUIRE(gh.hessians.size()  == bvg.values.size());

        for (std::size_t k = 0; k < bvg.values.size(); ++k) {
            REQUIRE(std::abs(gh.values[k] - bvg.values[k]) < 1e-14);
            REQUIRE((gh.gradients[k] - bvg.gradients[k]).lpNorm<Eigen::Infinity>()
                    < 1e-14);
            REQUIRE(gh.hessians[k].rows() == 2);
            REQUIRE(gh.hessians[k].cols() == 2);
        }
    }
}

TEST_CASE("evaluate_sme_basis_grad_and_hess: per-node Hessians are symmetric",
          "[lme][sme_basis_grad_hess]")
{
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.3,  0.2},
        {-0.4, 0.1},
        {0.45, -0.35},
    };

    for (const auto& x : queries) {
        const auto gh = evaluate_sme_basis_grad_and_hess(f.nodes, f.d, x, r_cut);
        for (const auto& H : gh.hessians) {
            const Eigen::Matrix2d asym = H - H.transpose();
            REQUIRE(asym.cwiseAbs().maxCoeff() < 1e-14);
        }
    }
}

TEST_CASE("evaluate_sme_basis_grad_and_hess: Σ ∇²s_a = 0",
          "[lme][sme_basis_grad_hess]")
{
    // Hessian of PoU. Bounded by FD truncation (h=1e-5, expect ~1e-8
    // error per component) inflated by the IFT propagation in each
    // gradient evaluation.
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.3,  0.2},
        {-0.4, 0.1},
        {0.45, -0.35},
    };

    for (const auto& x : queries) {
        const auto gh = evaluate_sme_basis_grad_and_hess(f.nodes, f.d, x, r_cut);
        Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
        for (const auto& H : gh.hessians) S += H;
        REQUIRE(S.cwiseAbs().maxCoeff() < 1e-6);
    }
}

TEST_CASE("evaluate_sme_basis_grad_and_hess: Σ x_{a,k} ∇²s_a = 0",
          "[lme][sme_basis_grad_hess]")
{
    // Hessian of Σ s_a x_a = x; the LHS is linear in x so its
    // 2nd derivative vanishes. Verify component-wise.
    const auto f = build_fixture();
    const double r_cut = 10.0;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.3,  0.2},
        {-0.4, 0.1},
    };

    for (const auto& x : queries) {
        const auto gh = evaluate_sme_basis_grad_and_hess(f.nodes, f.d, x, r_cut);

        for (int comp = 0; comp < 2; ++comp) {
            Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
            for (std::size_t k = 0; k < gh.indices.size(); ++k) {
                const int a = gh.indices[k];
                S += f.nodes(a, comp) * gh.hessians[k];
            }
            REQUIRE(S.cwiseAbs().maxCoeff() < 1e-6);
        }
    }
}

TEST_CASE("evaluate_sme_basis_grad_and_hess: matches a second-pass FD against gradient",
          "[lme][sme_basis_grad_hess]")
{
    // Two independent FD passes on the analytical gradient — same
    // implementation strategy but with a different step size — must
    // agree to better than the per-pass FD truncation accuracy.
    // Implementation uses h_impl = 1e-5; check against h_test = 1e-4.
    // Truncation residual scales as h²: 1e-10 (impl) vs 1e-8 (test).
    const auto f = build_fixture();
    const double r_cut = 10.0;
    const double h     = 1e-4;

    const std::vector<Eigen::Vector2d> queries = {
        {0.0,  0.0},
        {0.3,  0.2},
        {-0.31, 0.42},
    };

    for (const auto& x : queries) {
        const auto gh = evaluate_sme_basis_grad_and_hess(f.nodes, f.d, x, r_cut);

        // Build "manual" FD Hessian on the gradient with our own h.
        const auto gp_x = evaluate_sme_basis_and_grad(
            f.nodes, f.d, x + Eigen::Vector2d(h, 0), r_cut);
        const auto gm_x = evaluate_sme_basis_and_grad(
            f.nodes, f.d, x - Eigen::Vector2d(h, 0), r_cut);
        const auto gp_y = evaluate_sme_basis_and_grad(
            f.nodes, f.d, x + Eigen::Vector2d(0, h), r_cut);
        const auto gm_y = evaluate_sme_basis_and_grad(
            f.nodes, f.d, x - Eigen::Vector2d(0, h), r_cut);

        for (std::size_t k = 0; k < gh.indices.size(); ++k) {
            // FD column 0 = ∂(grad)/∂x ; column 1 = ∂(grad)/∂y .
            const Eigen::Vector2d col_x =
                (gp_x.gradients[k] - gm_x.gradients[k]) / (2.0 * h);
            const Eigen::Vector2d col_y =
                (gp_y.gradients[k] - gm_y.gradients[k]) / (2.0 * h);
            Eigen::Matrix2d H_fd;
            H_fd.col(0) = col_x;
            H_fd.col(1) = col_y;
            H_fd = 0.5 * (H_fd + H_fd.transpose().eval());

            const Eigen::Matrix2d H_ana = gh.hessians[k];
            const double err = (H_ana - H_fd).cwiseAbs().maxCoeff();
            INFO("node " << gh.indices[k] << " err = " << err);
            REQUIRE(err < 1e-5);
        }
    }
}

TEST_CASE("evaluate_sme_basis_grad_and_hess: throws at boundary-corner exact match",
          "[lme][sme_basis_grad_hess][validation]")
{
    const auto f = build_fixture();
    const Eigen::Vector2d xb = f.nodes.row(idx(0, 0)).transpose();
    REQUIRE_THROWS_AS(
        evaluate_sme_basis_grad_and_hess(f.nodes, f.d, xb, 10.0),
        std::domain_error);
}

TEST_CASE("evaluate_sme_basis_grad_and_hess_closed_form: verification harness",
          "[.diag][lme][sme_basis_grad_hess][diag_sme_closed_form]")
{
    // Verification harness for the closed-form Hessian
    // (Rosolen-Millán-Arroyo 2013 Appendix C, derived in our θ packing
    // — see build_grad_and_hess_from_state). The closed form is the
    // *exact* spatial Hessian; the FD path is a finite-difference of
    // the gradient and carries a noise floor of ~1e-6 (its own
    // self-consistency tests above gate at 1e-6/1e-5). So "match FD" is
    // a weak check. The decisive correctness evidence is:
    //
    //   (A) the closed form satisfies the two analytic Hessian
    //       sum-identities to machine precision (the FD path only
    //       reaches ~1e-6), and
    //   (B) it matches a Richardson-extrapolated FD reference (O(h⁴),
    //       built here in the truncation-dominated regime) far below
    //       the raw single-h FD noise floor.
    const auto f = build_fixture();
    const std::vector<Eigen::Vector2d> qs = {
        Eigen::Vector2d(0.0, 0.0),
        Eigen::Vector2d(0.25, -0.25),
        Eigen::Vector2d(-0.5, 0.5),
        Eigen::Vector2d(0.1, 0.7),
        Eigen::Vector2d(-0.7, -0.1),
    };
    constexpr double r_cut = 10.0;

    // Richardson central-FD Hessian of the gradient: combine h and 2h
    // in the truncation regime to cancel the O(h²) term, leaving O(h⁴).
    auto fd_hessian = [&](const Eigen::Vector2d& x, double h) {
        const auto gp_x = evaluate_sme_basis_and_grad(
            f.nodes, f.d, x + Eigen::Vector2d(h, 0), r_cut);
        const auto gm_x = evaluate_sme_basis_and_grad(
            f.nodes, f.d, x - Eigen::Vector2d(h, 0), r_cut);
        const auto gp_y = evaluate_sme_basis_and_grad(
            f.nodes, f.d, x + Eigen::Vector2d(0, h), r_cut);
        const auto gm_y = evaluate_sme_basis_and_grad(
            f.nodes, f.d, x - Eigen::Vector2d(0, h), r_cut);
        std::vector<Eigen::Matrix2d> H(gp_x.indices.size());
        for (std::size_t k = 0; k < H.size(); ++k) {
            Eigen::Matrix2d M;
            M.col(0) = (gp_x.gradients[k] - gm_x.gradients[k]) / (2.0 * h);
            M.col(1) = (gp_y.gradients[k] - gm_y.gradients[k]) / (2.0 * h);
            H[k] = 0.5 * (M + M.transpose().eval());
        }
        return H;
    };

    double max_diff_raw  = 0.0;   // closed-form vs raw single-h FD
    double max_diff_rich = 0.0;   // closed-form vs Richardson FD
    double max_sum_pou   = 0.0;   // Σ_a ∇²s_a
    double max_sum_lin   = 0.0;   // Σ_a x_{a,c} ∇²s_a
    double max_asym      = 0.0;   // symmetry of closed-form Hessian

    for (const auto& xq : qs) {
        const auto cf = evaluate_sme_basis_grad_and_hess_closed_form(
            f.nodes, f.d, xq, r_cut);
        const auto fd = evaluate_sme_basis_grad_and_hess(
            f.nodes, f.d, xq, r_cut);

        // Richardson: H ≈ (4·FD(h) − FD(2h)) / 3, in the truncation
        // regime (h = 1e-3, well above the ~1e-12 gradient-noise floor).
        const double h = 1e-3;
        const auto fd_h  = fd_hessian(xq, h);
        const auto fd_2h = fd_hessian(xq, 2.0 * h);

        REQUIRE(cf.indices.size() == fd.indices.size());
        Eigen::Matrix2d sum_pou = Eigen::Matrix2d::Zero();
        Eigen::Matrix2d sum_lin0 = Eigen::Matrix2d::Zero();
        Eigen::Matrix2d sum_lin1 = Eigen::Matrix2d::Zero();
        for (std::size_t k = 0; k < cf.indices.size(); ++k) {
            REQUIRE(cf.indices[k] == fd.indices[k]);
            const Eigen::Matrix2d& Hcf = cf.hessians[k];
            const Eigen::Matrix2d rich = (4.0 * fd_h[k] - fd_2h[k]) / 3.0;

            max_diff_raw  = std::max(max_diff_raw,
                (Hcf - fd.hessians[k]).cwiseAbs().maxCoeff());
            max_diff_rich = std::max(max_diff_rich,
                (Hcf - rich).cwiseAbs().maxCoeff());
            max_asym = std::max(max_asym,
                (Hcf - Hcf.transpose()).cwiseAbs().maxCoeff());

            const int a = cf.indices[k];
            sum_pou  += Hcf;
            sum_lin0 += f.nodes(a, 0) * Hcf;
            sum_lin1 += f.nodes(a, 1) * Hcf;
        }
        max_sum_pou = std::max(max_sum_pou, sum_pou.cwiseAbs().maxCoeff());
        max_sum_lin = std::max({max_sum_lin,
            sum_lin0.cwiseAbs().maxCoeff(),
            sum_lin1.cwiseAbs().maxCoeff()});
    }
    std::fprintf(stderr,
        "[sme_closed_form_diag] %zu queries: Σ∇²s=%.3e  Σx∇²s=%.3e  "
        "asym=%.3e  vs-Richardson-FD=%.3e  vs-raw-FD=%.3e\n",
        qs.size(), max_sum_pou, max_sum_lin, max_asym,
        max_diff_rich, max_diff_raw);

    // (A) Analytic identities — exact for the closed form (the FD path
    //     only reaches 1e-6 on these). Σ∇²s and symmetry are exact for
    //     *any* θ (softmax normalisation + symmetric construction), so
    //     they hit machine precision. Σx·∇²s = 0 instead follows from
    //     linear reproduction Σs·x = x, which holds only at the *true*
    //     θ*; at the Newton-converged θ* (residual ≤ Newton tol 1e-10)
    //     it and its Hessian inherit that tolerance → ~1e-9, not eps.
    REQUIRE(max_sum_pou < 1e-12);
    REQUIRE(max_sum_lin < 1e-8);
    REQUIRE(max_asym    < 1e-14);
    // (B) High-accuracy Richardson reference: the closed form is the
    //     limit the FD path converges toward, so this is far tighter
    //     than the raw single-h FD noise floor (~1e-6).
    REQUIRE(max_diff_rich < 1e-7);
    // Raw single-h FD agrees within its own noise floor.
    REQUIRE(max_diff_raw < 1e-6);
}

