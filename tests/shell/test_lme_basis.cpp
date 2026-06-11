/**
 * @file test_lme_basis.cpp
 * @brief Unit tests for the 1st-order Local Max-Ent basis evaluator.
 *
 * Tests the three properties of @ref chladni::shell::lme::evaluate_basis
 * proved in @cite arroyo_ortiz_2006_local_maximum_entropy:
 *
 * 1. **Partition of unity** — @f$ \sum_a p_a(x) = 1 @f$ to machine
 *    precision. Built into the basis construction
 *    (@f$ p_a = q_a / Z @f$); failure indicates a Newton convergence
 *    or numerical-overflow bug.
 * 2. **Affine reproduction** — @f$ \sum_a p_a(x)\, x_a = x @f$. The
 *    constraint that the dual Newton solver enforces; an exact-to-tol
 *    pass certifies that @f$ \lambda^\star @f$ was reached.
 * 3. **Weak Kronecker-delta at conv-hull corners** — at a corner
 *    @f$ x_b @f$ of @f$ \mathrm{conv}\,X @f$, @f$ p_a(x_b) = \delta_{ab} @f$.
 *    Handled by the exact-node-match shortcut in @c lme.cpp (the dual
 *    Hessian degenerates at corners; the shortcut bypasses Newton).
 *
 * Auxiliary checks:
 *
 * - Nonnegativity @f$ p_a(x) \ge 0 @f$ at every query point.
 * - Input validation: empty node set / mismatched @c beta /
 *   nonpositive @c beta_a / empty active set after truncation must
 *   throw @c std::invalid_argument.
 *
 * Fixture is a regular @f$ N \times N @f$ square grid on @f$ [0, L]^2 @f$
 * with uniform @f$ \beta_a = \gamma / h^2 @f$ at the Millan 2011
 * default @f$ \gamma = 1.6 @f$.
 */

#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>

using chladni::shell::LMEBasisAndGrad;
using chladni::shell::LMEBasisGradHess;
using chladni::shell::LMEBasisValues;
using chladni::shell::lme::evaluate_basis;
using chladni::shell::lme::evaluate_basis_and_grad;
using chladni::shell::lme::evaluate_basis_grad_and_hess;
using Catch::Matchers::WithinAbs;

namespace {

/// Build a uniform @f$ N \times N @f$ grid on @f$ [0, L]^2 @f$ in
/// row-major order: node id @c i*N+j has coordinates
/// @f$ (i\,h,\, j\,h) @f$ with @f$ h = L / (N-1) @f$. Conv-hull
/// corners are at ids @c 0, @c N-1, @c N*(N-1), @c N*N-1.
Eigen::MatrixXd square_grid(int n, double L = 1.0)
{
    Eigen::MatrixXd X(static_cast<Eigen::Index>(n) * n, 2);
    const double h = L / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const Eigen::Index row = static_cast<Eigen::Index>(i) * n + j;
            X(row, 0) = static_cast<double>(i) * h;
            X(row, 1) = static_cast<double>(j) * h;
        }
    }
    return X;
}

/// Convenience: build @c beta = γ / h² constant across all nodes.
Eigen::VectorXd uniform_beta(Eigen::Index n_nodes, double gamma, double h)
{
    return Eigen::VectorXd::Constant(n_nodes, gamma / (h * h));
}

/// Reconstruct the expansion @f$ \sum_a p_a x_a @f$ for affine
/// reproduction checks.
Eigen::Vector2d reconstruct_x(const LMEBasisValues& bv,
                              const Eigen::MatrixXd& nodes)
{
    Eigen::Vector2d x = Eigen::Vector2d::Zero();
    for (std::size_t k = 0; k < bv.indices.size(); ++k) {
        x += bv.values[k] * nodes.row(bv.indices[k]).transpose();
    }
    return x;
}

}  // namespace

TEST_CASE("LME basis: partition of unity at interior point", "[lme][basis]")
{
    // 5x5 grid on [0,1]^2; h = 0.25. Wide r_cut so all 25 nodes
    // participate — strict PoU should hold to machine precision.
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    Eigen::VectorXd x(2);
    x << 0.37, 0.42;  // generic interior, off-grid

    const auto bv = evaluate_basis(nodes, beta, x, /*r_cut=*/10.0);

    REQUIRE(bv.indices.size() == bv.values.size());

    double sum = 0.0;
    for (const double v : bv.values) {
        REQUIRE(v >= 0.0);
        sum += v;
    }
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-12));
}

TEST_CASE("LME basis: affine reproduction Sum p_a x_a = x",
          "[lme][basis]")
{
    // Same fixture; verify the linear constraint enforced by Newton.
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    // Battery of interior query points, each in conv(X) = [0,1]^2.
    const std::vector<Eigen::Vector2d> probes = {
        {0.31, 0.59},  // generic
        {0.50, 0.50},  // centre
        {0.13, 0.27},  // near boundary
        {0.88, 0.46},  // near opposite boundary
        {0.62, 0.74},
    };

    for (const auto& xref : probes) {
        Eigen::VectorXd x = xref;
        const auto bv = evaluate_basis(nodes, beta, x, /*r_cut=*/10.0);

        const Eigen::Vector2d xrec = reconstruct_x(bv, nodes);
        const double err = (xrec - xref).norm();
        INFO("query x = (" << xref(0) << ", " << xref(1) << ")");
        REQUIRE_THAT(err, WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("LME basis: weak Kronecker-delta at conv-hull corners",
          "[lme][basis]")
{
    // At a corner x_b of conv(X), p_a(x_b) = delta_{ab}.
    // (Arroyo-Ortiz 2006 §3.1; doc §3 second bullet.)
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    // Row-major layout: corners are at indices 0, N-1, N*(N-1), N*N-1.
    const std::vector<int> corner_ids = {
        0,
        N - 1,
        N * (N - 1),
        N * N - 1,
    };

    for (const int b : corner_ids) {
        Eigen::VectorXd x = nodes.row(b).transpose();
        const auto bv = evaluate_basis(nodes, beta, x, /*r_cut=*/10.0);

        INFO("corner node id = " << b);
        // Find p_b(x_b) and confirm it is 1; all others 0.
        bool found_b = false;
        for (std::size_t k = 0; k < bv.indices.size(); ++k) {
            const int    a = bv.indices[k];
            const double p = bv.values[k];
            if (a == b) {
                REQUIRE_THAT(p, WithinAbs(1.0, 1e-12));
                found_b = true;
            } else {
                REQUIRE_THAT(p, WithinAbs(0.0, 1e-12));
            }
        }
        REQUIRE(found_b);
    }
}

TEST_CASE("LME basis: Newton tolerance is met at probes near boundary",
          "[lme][basis]")
{
    // Stress Newton near the conv-hull boundary (without sitting on a
    // node, which would trigger the exact-match shortcut). Tightens
    // the affine-reproduction tolerance to match the requested Newton
    // tol — confirms the solver actually drives ||grad ln Z|| down to
    // the spec.
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    Eigen::VectorXd x(2);
    x << 0.04, 0.06;  // close to corner (0,0) but not on it

    const auto bv = evaluate_basis(nodes, beta, x,
                                    /*r_cut=*/10.0,
                                    /*newton_tol=*/1e-12,
                                    /*newton_max_iters=*/60);

    double sum = 0.0;
    for (const double v : bv.values) sum += v;
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-12));

    const Eigen::Vector2d xrec = reconstruct_x(bv, nodes);
    REQUIRE_THAT((xrec - x).norm(), WithinAbs(0.0, 1e-11));
}

TEST_CASE("LME basis gradient: finite-difference regression",
          "[lme][basis][grad]")
{
    // Closed-form ∇p_a from Arroyo-Ortiz 2006 eq. 44 (uniform β):
    //     ∇p_a = -p_a · J^{-1} · (x - x_a)
    // Compare against central-differences of evaluate_basis at the
    // same x. The active sets must coincide at x and at x ± h*e_j —
    // a wide r_cut ensures every node is in every set.
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    // All probes deliberately off-node: with h=0.25 on a 5x5 grid,
    // every node sits on the (0.25 k_1, 0.25 k_2) lattice — picking
    // odd .xy decimals avoids coincidence (and the gradient-undefined
    // domain_error that goes with it).
    const std::vector<Eigen::Vector2d> probes = {
        {0.37, 0.42},
        {0.31, 0.59},
        {0.62, 0.27},
        {0.13, 0.74},
    };

    constexpr double fd_step = 1e-5;
    // Newton tol kept tight so the FD residual is dominated by the
    // O(h^2) truncation of central differences (~1e-10) rather than
    // by Newton noise on the underlying p_a samples.
    constexpr double tol     = 1e-12;

    for (const auto& xref : probes) {
        Eigen::VectorXd x = xref;
        const auto bg = evaluate_basis_and_grad(nodes, beta, x,
                                                 /*r_cut=*/10.0,
                                                 tol, /*max_iters=*/60);

        INFO("query x = (" << xref(0) << ", " << xref(1) << ")");
        REQUIRE(bg.indices.size() == bg.values.size());
        REQUIRE(bg.indices.size() == bg.gradients.size());

        // Reference: full N-vector of FD gradients, indexed by node id.
        Eigen::MatrixXd grad_fd =
            Eigen::MatrixXd::Zero(nodes.rows(), 2);
        for (int j = 0; j < 2; ++j) {
            Eigen::VectorXd xp = x;
            Eigen::VectorXd xm = x;
            xp(j) += fd_step;
            xm(j) -= fd_step;
            const auto bp = evaluate_basis(nodes, beta, xp,
                                            /*r_cut=*/10.0, tol, 60);
            const auto bm = evaluate_basis(nodes, beta, xm,
                                            /*r_cut=*/10.0, tol, 60);
            // Wide r_cut → active set is all nodes in row-major order.
            for (std::size_t k = 0; k < bp.indices.size(); ++k) {
                grad_fd(bp.indices[k], j) += bp.values[k] / (2.0 * fd_step);
            }
            for (std::size_t k = 0; k < bm.indices.size(); ++k) {
                grad_fd(bm.indices[k], j) -= bm.values[k] / (2.0 * fd_step);
            }
        }

        // Compare per-active-node.
        for (std::size_t k = 0; k < bg.indices.size(); ++k) {
            const int a = bg.indices[k];
            const Eigen::Vector2d g_ana = bg.gradients[k];
            const Eigen::Vector2d g_num = grad_fd.row(a).transpose();
            const double err = (g_ana - g_num).norm();
            INFO("node " << a << "  ana=(" << g_ana(0) << "," << g_ana(1)
                 << ")  num=(" << g_num(0) << "," << g_num(1) << ")");
            // FD truncation error scales like (fd_step)^2 * |∇^3 p| /
            // 6; on this fixture |∇^3 p| stays modest (~50) so the FD
            // floor is around 1e-8 with fd_step=1e-5.
            REQUIRE_THAT(err, WithinAbs(0.0, 5e-7));
        }
    }
}

TEST_CASE("LME basis gradient: consistency identities",
          "[lme][basis][grad]")
{
    // From differentiating PoU Σ p_a = 1 and linear reproduction
    // Σ p_a x_a = x:
    //   Σ_a ∇p_a(x) = 0           (in R^d)
    //   Σ_a x_a ⊗ (∇p_a)^T = I    (in R^{d x d})
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    // Off-node probes (see note in the FD regression test).
    const std::vector<Eigen::Vector2d> probes = {
        {0.37, 0.62},
        {0.42, 0.18},
        {0.71, 0.83},
    };

    for (const auto& xref : probes) {
        Eigen::VectorXd x = xref;
        // Tight Newton tol — the residual Σ ∇p_a equals -J^{-1} g
        // where g is the converged Newton gradient (norm ≤ tol). Near
        // the boundary ||J^{-1}|| can reach ~50, so newton_tol=1e-13
        // keeps the consistency residual under 1e-11.
        const auto bg = evaluate_basis_and_grad(nodes, beta, x,
                                                 /*r_cut=*/10.0,
                                                 /*newton_tol=*/1e-13,
                                                 /*max_iters=*/60);

        INFO("query x = (" << xref(0) << ", " << xref(1) << ")");

        Eigen::Vector2d sum_grad = Eigen::Vector2d::Zero();
        Eigen::Matrix2d sum_xa_grad = Eigen::Matrix2d::Zero();
        for (std::size_t k = 0; k < bg.indices.size(); ++k) {
            sum_grad    += bg.gradients[k];
            const Eigen::Vector2d xa = nodes.row(bg.indices[k]).transpose();
            sum_xa_grad += xa * bg.gradients[k].transpose();
        }

        // ∑ ∇p_a = 0 ∈ R^2
        REQUIRE_THAT(sum_grad.norm(), WithinAbs(0.0, 1e-10));

        // ∑ x_a ⊗ (∇p_a)^T = I ∈ R^{2x2}
        const double err = (sum_xa_grad
                            - Eigen::Matrix2d::Identity()).norm();
        REQUIRE_THAT(err, WithinAbs(0.0, 1e-10));
    }
}

TEST_CASE("LME basis Hessian: finite-difference regression against gradients",
          "[lme][basis][hess]")
{
    // Closed-form ∇²p_a should match central differences of the
    // closed-form gradients. (Two analytic formulas compared against
    // each other through one FD layer — the second-derivative formula
    // can't be wrong without ALSO breaking the gradient one this way.)
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    const std::vector<Eigen::Vector2d> probes = {
        {0.37, 0.42},
        {0.31, 0.59},
        {0.62, 0.27},
    };

    constexpr double fd_step = 1e-5;
    constexpr double tol     = 1e-12;

    for (const auto& xref : probes) {
        Eigen::VectorXd x = xref;
        const auto gh = evaluate_basis_grad_and_hess(
            nodes, beta, x, /*r_cut=*/10.0, tol, /*max_iters=*/60);

        INFO("query x = (" << xref(0) << ", " << xref(1) << ")");
        REQUIRE(gh.indices.size() == gh.values.size());
        REQUIRE(gh.indices.size() == gh.gradients.size());
        REQUIRE(gh.indices.size() == gh.hessians.size());

        // FD reference: ∂(∇p_a)_i / ∂x_j ≈ [(∇p_a at x+h e_j)_i -
        // (∇p_a at x-h e_j)_i] / (2h). For each direction j, collect a
        // full per-node gradient vector via evaluate_basis_and_grad.
        Eigen::MatrixXd hess_fd =
            Eigen::MatrixXd::Zero(nodes.rows() * 2, 2);  // row 2a+i, col j
        for (int j = 0; j < 2; ++j) {
            Eigen::VectorXd xp = x;
            Eigen::VectorXd xm = x;
            xp(j) += fd_step;
            xm(j) -= fd_step;
            const auto gp = evaluate_basis_and_grad(nodes, beta, xp,
                                                     /*r_cut=*/10.0,
                                                     tol, 60);
            const auto gm = evaluate_basis_and_grad(nodes, beta, xm,
                                                     /*r_cut=*/10.0,
                                                     tol, 60);
            for (std::size_t k = 0; k < gp.indices.size(); ++k) {
                const int a = gp.indices[k];
                for (int i = 0; i < 2; ++i) {
                    hess_fd(2 * a + i, j) +=
                        gp.gradients[k](i) / (2.0 * fd_step);
                }
            }
            for (std::size_t k = 0; k < gm.indices.size(); ++k) {
                const int a = gm.indices[k];
                for (int i = 0; i < 2; ++i) {
                    hess_fd(2 * a + i, j) -=
                        gm.gradients[k](i) / (2.0 * fd_step);
                }
            }
        }

        for (std::size_t k = 0; k < gh.indices.size(); ++k) {
            const int a = gh.indices[k];
            const Eigen::Matrix2d H_ana = gh.hessians[k];
            Eigen::Matrix2d H_num;
            H_num << hess_fd(2 * a + 0, 0), hess_fd(2 * a + 0, 1),
                     hess_fd(2 * a + 1, 0), hess_fd(2 * a + 1, 1);

            // Symmetry of the analytic Hessian.
            REQUIRE_THAT((H_ana - H_ana.transpose()).norm(),
                         WithinAbs(0.0, 1e-12));

            const double err = (H_ana - H_num).norm();
            INFO("node " << a << "  ||H_ana - H_num|| = " << err);
            // Hessian FD floor is roughly fd_step^2 * |∇^3 grad| ~
            // 1e-4 at γ=1.6 on this fixture — pick a margin that's
            // 10× that to catch sign / coefficient bugs without
            // tracking FD truncation drift.
            REQUIRE_THAT(err, WithinAbs(0.0, 1e-3));
        }
    }
}

TEST_CASE("LME basis Hessian: consistency identities",
          "[lme][basis][hess]")
{
    // From the higher-order reproduction identities:
    //   Σ_a ∇²p_a(x) = 0           (∇² of PoU)
    //   Σ_a x_{a,k} ∇²p_a(x) = 0   (∇² of linear reproduction)
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    const std::vector<Eigen::Vector2d> probes = {
        {0.37, 0.62},
        {0.42, 0.18},
        {0.71, 0.83},
    };

    for (const auto& xref : probes) {
        Eigen::VectorXd x = xref;
        // Newton tol tight enough that the consistency residuals are
        // dominated by floating-point round-off, not by Newton noise.
        const auto gh = evaluate_basis_grad_and_hess(
            nodes, beta, x, /*r_cut=*/10.0,
            /*newton_tol=*/1e-13, /*max_iters=*/60);

        INFO("query x = (" << xref(0) << ", " << xref(1) << ")");

        Eigen::Matrix2d sum_hess    = Eigen::Matrix2d::Zero();
        Eigen::Matrix2d sum_xa0_h   = Eigen::Matrix2d::Zero();  // k=0
        Eigen::Matrix2d sum_xa1_h   = Eigen::Matrix2d::Zero();  // k=1
        for (std::size_t k = 0; k < gh.indices.size(); ++k) {
            const Eigen::Vector2d xa = nodes.row(gh.indices[k]).transpose();
            const Eigen::Matrix2d H  = gh.hessians[k];
            sum_hess  += H;
            sum_xa0_h += xa(0) * H;
            sum_xa1_h += xa(1) * H;
        }

        REQUIRE_THAT(sum_hess.norm(),  WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(sum_xa0_h.norm(), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(sum_xa1_h.norm(), WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("LME basis gradient + Hessian: finite-difference regression with "
          "NON-UNIFORM per-node beta",
          "[lme][basis][grad][hess][nonuniform_beta]")
{
    // REGRESSION (2026-06-03, QuadSplit spurious-mode root cause): the
    // closed-form derivatives previously used the Arroyo-Ortiz 2006
    // eq. 44 UNIFORM-β specialisation ∇p_a = -p_a J^{-1}(x - x_a),
    // while the assemblers pass PER-NODE β_a = γ/h_a². For non-uniform
    // β the correct formulas are Millán 2011 Appendix A (A5)-(A7),
    // with the extra terms r̄ = 2 Σ_b β_b p_b (x - x_b) and
    // Dk* = (J̄ - I)(J*)^{-1}, J̄ = 2 Σ_b β_b p_b m_b ⊗ m_b. On
    // quasi-uniform meshes the β spread is small and the missing terms
    // were invisible; on two-sublattice node sets (QuadSplit UnionJack
    // / Checkerboard, ~30-70% β alternation) the wrong Hessians made
    // the assembled K a non-variational energy → a spurious m=0 mode
    // BELOW the analytic spectrum (32x8 Leissa disk: 83 Hz vs the
    // physical 224 Hz breathing mode).
    //
    // Fixture: 5x5 grid with a strong checkerboard β modulation
    // (±40%), FD ground truth exactly as in the uniform-β tests above.
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    Eigen::VectorXd beta(nodes.rows());
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double mod = ((i + j) % 2 == 0) ? 1.4 : 0.6;
            beta(static_cast<Eigen::Index>(i) * N + j) =
                mod * gamma / (h * h);
        }
    }

    const std::vector<Eigen::Vector2d> probes = {
        {0.37, 0.42},
        {0.31, 0.59},
        {0.62, 0.27},
        {0.13, 0.74},
    };

    constexpr double fd_step = 1e-5;
    constexpr double tol     = 1e-12;

    for (const auto& xref : probes) {
        Eigen::VectorXd x = xref;
        const auto gh = evaluate_basis_grad_and_hess(
            nodes, beta, x, /*r_cut=*/10.0, tol, /*max_iters=*/60);

        INFO("query x = (" << xref(0) << ", " << xref(1) << ")");

        // --- gradient FD reference (on basis VALUES, which come from
        // the dual Newton solve and are exact for any per-node β) ----
        Eigen::MatrixXd grad_fd =
            Eigen::MatrixXd::Zero(nodes.rows(), 2);
        // --- Hessian FD reference (2nd central differences of the
        // VALUES — independent of the gradient closed form) ----------
        Eigen::MatrixXd hess_fd =
            Eigen::MatrixXd::Zero(nodes.rows() * 2, 2);
        for (int j = 0; j < 2; ++j) {
            Eigen::VectorXd xp = x, xm = x;
            xp(j) += fd_step;
            xm(j) -= fd_step;
            const auto bp = evaluate_basis(nodes, beta, xp,
                                           /*r_cut=*/10.0, tol, 60);
            const auto bm = evaluate_basis(nodes, beta, xm,
                                           /*r_cut=*/10.0, tol, 60);
            for (std::size_t k = 0; k < bp.indices.size(); ++k) {
                grad_fd(bp.indices[k], j) +=
                    bp.values[k] / (2.0 * fd_step);
            }
            for (std::size_t k = 0; k < bm.indices.size(); ++k) {
                grad_fd(bm.indices[k], j) -=
                    bm.values[k] / (2.0 * fd_step);
            }
            // Hessian columns via FD of values: ∂²p/∂x_i∂x_j from
            // central differences of one-sided gradient FD would
            // compound error; instead FD the analytic-free VALUES on a
            // 4-point stencil per (i, j) below.
        }
        // Full 2x2 value-based Hessian stencil per node:
        //   H(i,j) = [p(x+hi ei+hj ej) - p(x+hi ei-hj ej)
        //            - p(x-hi ei+hj ej) + p(x-hi ei-hj ej)] / (4 h²)
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                Eigen::VectorXd xpp = x, xpm = x, xmp = x, xmm = x;
                xpp(i) += fd_step; xpp(j) += fd_step;
                xpm(i) += fd_step; xpm(j) -= fd_step;
                xmp(i) -= fd_step; xmp(j) += fd_step;
                xmm(i) -= fd_step; xmm(j) -= fd_step;
                const auto bpp = evaluate_basis(nodes, beta, xpp, 10.0, tol, 60);
                const auto bpm = evaluate_basis(nodes, beta, xpm, 10.0, tol, 60);
                const auto bmp = evaluate_basis(nodes, beta, xmp, 10.0, tol, 60);
                const auto bmm = evaluate_basis(nodes, beta, xmm, 10.0, tol, 60);
                auto scatter = [&](const LMEBasisValues& bv, double s) {
                    for (std::size_t k = 0; k < bv.indices.size(); ++k) {
                        hess_fd(2 * bv.indices[k] + i, j) +=
                            s * bv.values[k] /
                            (4.0 * fd_step * fd_step);
                    }
                };
                scatter(bpp, +1.0);
                scatter(bpm, -1.0);
                scatter(bmp, -1.0);
                scatter(bmm, +1.0);
            }
        }

        for (std::size_t k = 0; k < gh.indices.size(); ++k) {
            const int a = gh.indices[k];
            const Eigen::Vector2d g_ana = gh.gradients[k];
            const Eigen::Vector2d g_num = grad_fd.row(a).transpose();
            INFO("node " << a
                 << "  grad ana=(" << g_ana(0) << "," << g_ana(1)
                 << ")  num=(" << g_num(0) << "," << g_num(1) << ")");
            REQUIRE_THAT((g_ana - g_num).norm(), WithinAbs(0.0, 5e-7));

            const Eigen::Matrix2d H_ana = gh.hessians[k];
            Eigen::Matrix2d H_num;
            H_num << hess_fd(2 * a + 0, 0), hess_fd(2 * a + 0, 1),
                     hess_fd(2 * a + 1, 0), hess_fd(2 * a + 1, 1);
            const double err = (H_ana - H_num).norm();
            INFO("node " << a << "  ||H_ana - H_num|| = " << err
                 << "  H_ana=[" << H_ana(0,0) << "," << H_ana(0,1)
                 << ";" << H_ana(1,0) << "," << H_ana(1,1)
                 << "]  H_num=[" << H_num(0,0) << "," << H_num(0,1)
                 << ";" << H_num(1,0) << "," << H_num(1,1) << "]");
            // Value-based 2nd-difference FD floor ~1e-4·|p| on this
            // fixture; gate at 1e-2 absolute (the uniform-β formula
            // applied to ±40% modulated β is wrong by O(1..100)).
            REQUIRE_THAT(err, WithinAbs(0.0, 1e-2));
        }
    }
}

TEST_CASE("LME basis gradient: throws at exact node coincidence",
          "[lme][basis][grad]")
{
    // At conv-hull corners the basis is non-differentiable; gradient
    // evaluation throws std::domain_error so callers fail loudly
    // instead of returning a fabricated value.
    constexpr int    N     = 5;
    constexpr double L     = 1.0;
    constexpr double gamma = 1.6;
    const double h = L / static_cast<double>(N - 1);

    const Eigen::MatrixXd nodes = square_grid(N, L);
    const Eigen::VectorXd beta  = uniform_beta(nodes.rows(), gamma, h);

    Eigen::VectorXd x_corner = nodes.row(0).transpose();
    REQUIRE_THROWS_AS(
        evaluate_basis_and_grad(nodes, beta, x_corner, /*r_cut=*/10.0),
        std::domain_error);

    // Same for an interior node — the gradient is well-defined there
    // in the limit but the implementation conservatively throws.
    Eigen::VectorXd x_mid = nodes.row(N * N / 2).transpose();
    REQUIRE_THROWS_AS(
        evaluate_basis_and_grad(nodes, beta, x_mid, /*r_cut=*/10.0),
        std::domain_error);
}

TEST_CASE("LME basis: input validation", "[lme][basis]")
{
    const Eigen::MatrixXd nodes_ok = square_grid(3);
    const Eigen::VectorXd beta_ok  = uniform_beta(nodes_ok.rows(), 1.6, 0.5);
    Eigen::VectorXd       x_ok(2);
    x_ok << 0.4, 0.4;

    SECTION("empty node set")
    {
        Eigen::MatrixXd empty(0, 2);
        Eigen::VectorXd empty_beta(0);
        REQUIRE_THROWS_AS(
            evaluate_basis(empty, empty_beta, x_ok, /*r_cut=*/10.0),
            std::invalid_argument);
    }

    SECTION("beta size mismatch")
    {
        Eigen::VectorXd beta_short(nodes_ok.rows() - 1);
        beta_short.setOnes();
        REQUIRE_THROWS_AS(
            evaluate_basis(nodes_ok, beta_short, x_ok, /*r_cut=*/10.0),
            std::invalid_argument);
    }

    SECTION("nonpositive beta")
    {
        Eigen::VectorXd beta_bad = beta_ok;
        beta_bad(0) = -1.0;
        REQUIRE_THROWS_AS(
            evaluate_basis(nodes_ok, beta_bad, x_ok, /*r_cut=*/10.0),
            std::invalid_argument);
    }

    SECTION("empty active set after truncation")
    {
        // Query point far from every node, r_cut tiny.
        Eigen::VectorXd x_far(2);
        x_far << 1000.0, 1000.0;
        REQUIRE_THROWS_AS(
            evaluate_basis(nodes_ok, beta_ok, x_far, /*r_cut=*/1e-3),
            std::invalid_argument);
    }
}
