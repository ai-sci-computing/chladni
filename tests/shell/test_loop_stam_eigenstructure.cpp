/**
 * @file test_loop_stam_eigenstructure.cpp
 * @brief S.1 — exact eigendecomposition of the extended subdivision
 *        matrix A around an extraordinary vertex of valence N.
 *
 * Pins
 * @ref chladni::shell::loop::stam_eigenstructure to the algebraic
 * identity V * Lambda * V^{-1} = A (with a Jordan-block correction
 * for N = 3, see @cite stam_1999_loop_evaluation Appendix C).
 *
 * Coverage:
 *  - Reconstruction A = V Lambda V^{-1} for N in {4, 5, 6, 7, 8, 12, 16}.
 *  - Reconstruction A = V J V^{-1} for N = 3 (Jordan block at lambda=1/16).
 *  - Eigenvalue spot-checks: mu_1 = 1, mu_2 = 5/8 - alpha(N), and a
 *    cyclic eigenvalue f(k) per Stam App. B.
 *  - V is non-singular (V_inv * V ~ I).
 *  - All eigenvalues lie in (0, 1] (Loop subdivision is a contraction
 *    on the eigenspace of non-affine modes).
 *  - Input validation: N < 3 throws.
 */

#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kRecoTol = 1.0e-9;
constexpr double kEigTol  = 1.0e-12;

double alpha_N(int N)
{
    const double c = std::cos(2.0 * std::numbers::pi_v<double>
                              / static_cast<double>(N));
    const double t = 3.0 + 2.0 * c;
    return 0.625 - (t * t) / 64.0;
}

double fk(int N, int k)
{
    return 0.375 + 0.25 *
        std::cos(2.0 * std::numbers::pi_v<double>
                 * static_cast<double>(k) / static_cast<double>(N));
}

/// Reconstruct A from V, V_inv, lambda — accounting for the N=3 Jordan
/// super-diagonal entry at position (K-2, K-1).
Eigen::MatrixXd reconstruct_A(
    const chladni::shell::loop::StamEigenstructure& es)
{
    const Eigen::Index K = es.lambda.size();
    Eigen::MatrixXd J = es.lambda.asDiagonal();
    if (es.has_jordan_block) {
        J(K - 2, K - 1) += 1.0;
    }
    return es.V * J * es.V_inv;
}

}  // namespace

TEST_CASE("build_extended_subdivision_matrix: shape, row sums, sanity",
          "[shell][loop][stam][subdivision_matrix]")
{
    using chladni::shell::loop::build_extended_subdivision_matrix;

    for (int N : {3, 4, 5, 6, 7, 8, 12, 16}) {
        CAPTURE(N);
        const Eigen::MatrixXd A = build_extended_subdivision_matrix(N);

        // Shape.
        REQUIRE(A.rows() == N + 6);
        REQUIRE(A.cols() == N + 6);

        // Row sums == 1: Loop subdivision is affine-invariant; each new
        // control vertex is a convex/affine combination of the old ones.
        for (int i = 0; i < A.rows(); ++i) {
            const double row_sum = A.row(i).sum();
            INFO("row " << i << " sum = " << row_sum);
            REQUIRE(std::abs(row_sum - 1.0) < 1.0e-12);
        }

        // Block-upper-triangular: top-right (N+1) x 5 block must be 0.
        REQUIRE(A.block(0, N + 1, N + 1, 5).cwiseAbs().maxCoeff() < 1.0e-15);
    }
}

TEST_CASE("build_extended_subdivision_matrix: throws on N < 3",
          "[shell][loop][stam][subdivision_matrix][validation]")
{
    using chladni::shell::loop::build_extended_subdivision_matrix;
    REQUIRE_THROWS_AS(build_extended_subdivision_matrix(2), std::invalid_argument);
    REQUIRE_THROWS_AS(build_extended_subdivision_matrix(0), std::invalid_argument);
    REQUIRE_THROWS_AS(build_extended_subdivision_matrix(-1), std::invalid_argument);
}

TEST_CASE("build_extended_subdivision_matrix_bar: shape, row sums, A consistency",
          "[shell][loop][stam][subdivision_matrix][bar]")
{
    using chladni::shell::loop::build_extended_subdivision_matrix;
    using chladni::shell::loop::build_extended_subdivision_matrix_bar;

    for (int N : {3, 4, 5, 6, 7, 8, 12, 16}) {
        CAPTURE(N);
        const Eigen::MatrixXd A    = build_extended_subdivision_matrix(N);
        const Eigen::MatrixXd Abar = build_extended_subdivision_matrix_bar(N);

        // Shape: (N+12) x (N+6).
        REQUIRE(Abar.rows() == N + 12);
        REQUIRE(Abar.cols() == N + 6);

        // Top K = N+6 rows of A_bar must equal A (the unbarred matrix).
        const double topblock_residual = (Abar.topRows(N + 6) - A).cwiseAbs().maxCoeff();
        INFO("top-block residual = " << topblock_residual);
        REQUIRE(topblock_residual < 1.0e-15);

        // All row sums == 1 (affine reproduction of Loop subdivision).
        for (int i = 0; i < Abar.rows(); ++i) {
            const double rs = Abar.row(i).sum();
            INFO("Abar row " << i << " sum = " << rs);
            REQUIRE(std::abs(rs - 1.0) < 1.0e-12);
        }
    }
}

TEST_CASE("build_extended_subdivision_matrix_bar: throws on N < 3",
          "[shell][loop][stam][subdivision_matrix][bar][validation]")
{
    using chladni::shell::loop::build_extended_subdivision_matrix_bar;
    REQUIRE_THROWS_AS(build_extended_subdivision_matrix_bar(2), std::invalid_argument);
    REQUIRE_THROWS_AS(build_extended_subdivision_matrix_bar(-1), std::invalid_argument);
}

TEST_CASE("stam_eigenstructure: reconstruction V Lambda V^{-1} = A, regular N>=4",
          "[shell][loop][stam][eigenstructure]")
{
    using chladni::shell::loop::stam_eigenstructure;
    using chladni::shell::loop::build_extended_subdivision_matrix;

    for (int N : {4, 5, 6, 7, 8, 12, 16}) {
        CAPTURE(N);
        const auto es = stam_eigenstructure(N);
        REQUIRE(es.N == N);
        REQUIRE_FALSE(es.has_jordan_block);
        REQUIRE(es.lambda.size() == N + 6);
        REQUIRE(es.V.rows() == N + 6);
        REQUIRE(es.V.cols() == N + 6);
        REQUIRE(es.V_inv.rows() == N + 6);
        REQUIRE(es.V_inv.cols() == N + 6);

        const Eigen::MatrixXd A      = build_extended_subdivision_matrix(N);
        const Eigen::MatrixXd A_reco = reconstruct_A(es);
        const double residual = (A - A_reco).cwiseAbs().maxCoeff();
        INFO("reconstruction max-abs residual = " << residual);
        REQUIRE(residual < kRecoTol);

        // V * V_inv ~ I.
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N + 6, N + 6);
        const double inv_residual = (es.V * es.V_inv - I).cwiseAbs().maxCoeff();
        INFO("V * V_inv - I  residual = " << inv_residual);
        REQUIRE(inv_residual < kRecoTol);
    }
}

TEST_CASE("stam_eigenstructure: reconstruction with Jordan block, N=3",
          "[shell][loop][stam][eigenstructure][jordan]")
{
    using chladni::shell::loop::stam_eigenstructure;
    using chladni::shell::loop::build_extended_subdivision_matrix;

    const auto es = stam_eigenstructure(3);
    REQUIRE(es.N == 3);
    REQUIRE(es.has_jordan_block);
    REQUIRE(es.lambda.size() == 9);

    // App. C eigenvalues: (1, 1/4, 1/4, 1/8, 1/8, 1/8, 1/16, 1/16, 1/16).
    const std::vector<double> expected_lambda{
        1.0, 0.25, 0.25, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625};
    for (int i = 0; i < 9; ++i) {
        CAPTURE(i);
        INFO("got = " << es.lambda(i) << "  expected = " << expected_lambda[i]);
        REQUIRE(std::abs(es.lambda(i) - expected_lambda[i]) < kEigTol);
    }

    const Eigen::MatrixXd A      = build_extended_subdivision_matrix(3);
    const Eigen::MatrixXd A_reco = reconstruct_A(es);
    const double residual = (A - A_reco).cwiseAbs().maxCoeff();
    INFO("Jordan reconstruction max-abs residual = " << residual);
    REQUIRE(residual < kRecoTol);

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(9, 9);
    REQUIRE((es.V * es.V_inv - I).cwiseAbs().maxCoeff() < kRecoTol);
}

TEST_CASE("stam_eigenstructure: eigenvalues land in (0, 1]",
          "[shell][loop][stam][eigenstructure][spectral]")
{
    using chladni::shell::loop::stam_eigenstructure;

    // Loop subdivision is non-expansive: |lambda| <= 1, with the unique
    // dominant eigenvalue lambda=1 (affine reproduction). All other
    // eigenvalues are strictly positive on the lower branches.
    for (int N : {3, 4, 5, 6, 7, 8, 12, 16}) {
        CAPTURE(N);
        const auto es = stam_eigenstructure(N);
        for (int i = 0; i < es.lambda.size(); ++i) {
            CAPTURE(i);
            const double lam = es.lambda(i);
            INFO("lambda[" << i << "] = " << lam);
            REQUIRE(lam > 0.0);
            REQUIRE(lam <= 1.0 + kEigTol);
        }
        // Exactly one eigenvalue equals 1 (the affine/constant mode).
        int n_ones = 0;
        for (int i = 0; i < es.lambda.size(); ++i) {
            if (std::abs(es.lambda(i) - 1.0) < kEigTol) ++n_ones;
        }
        REQUIRE(n_ones == 1);
    }
}

TEST_CASE("stam_eigenstructure: eigenvalue spot-checks vs Stam App. B",
          "[shell][loop][stam][eigenstructure][spectral]")
{
    using chladni::shell::loop::stam_eigenstructure;

    // For each tested N, verify that the multiset of eigenvalues
    // contains {mu_2 = 5/8 - alpha(N), f(1)}. We only check membership
    // (within tol) — the ordering inside lambda is an implementation
    // detail of stam_eigenstructure.
    auto contains = [](const Eigen::VectorXd& v, double x, double tol) {
        for (int i = 0; i < v.size(); ++i) {
            if (std::abs(v(i) - x) < tol) return true;
        }
        return false;
    };

    for (int N : {4, 5, 6, 7, 8, 12, 16}) {
        CAPTURE(N);
        const auto es = stam_eigenstructure(N);

        const double mu_2 = 0.625 - alpha_N(N);
        const double f_1  = fk(N, 1);

        INFO("mu_2 = " << mu_2 << "  f(1) = " << f_1);
        REQUIRE(contains(es.lambda, mu_2, 1.0e-10));
        REQUIRE(contains(es.lambda, f_1, 1.0e-10));
        REQUIRE(contains(es.lambda, 1.0,  1.0e-10));     // mu_1 = 1
        REQUIRE(contains(es.lambda, 0.125, 1.0e-10));    // S_12 has 1/8 mult 3
        REQUIRE(contains(es.lambda, 0.0625, 1.0e-10));   // S_12 has 1/16 mult 2
    }
}

TEST_CASE("stam_eigenstructure: regular case N=6 against known Loop spectrum",
          "[shell][loop][stam][eigenstructure][regular]")
{
    using chladni::shell::loop::stam_eigenstructure;

    // Regular interior vertex (N=6): closed-form Loop spectrum (Stam App B).
    // S eigenvalues for N=6: {mu_1=1, mu_2=1/4, f(1)=1/2, f(2)=1/4,
    //                         f(3)=1/8, f(4)=1/4, f(5)=1/2}
    //   = {1, 1/2, 1/2, 1/4, 1/4, 1/4, 1/8}.
    // Combined with S_12 eigenvalues {1/8, 1/8, 1/8, 1/16, 1/16}:
    //   = {1, 1/2, 1/2, 1/4, 1/4, 1/4, 1/8, 1/8, 1/8, 1/8, 1/16, 1/16}.
    const auto es = stam_eigenstructure(6);

    std::vector<double> sorted_eigs(es.lambda.data(),
                                    es.lambda.data() + es.lambda.size());
    std::sort(sorted_eigs.begin(), sorted_eigs.end(), std::greater<double>());

    const std::vector<double> expected{
        1.0,
        0.5, 0.5,
        0.25, 0.25, 0.25,
        0.125, 0.125, 0.125, 0.125,
        0.0625, 0.0625,
    };
    REQUIRE(sorted_eigs.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i, sorted_eigs[i], expected[i]);
        REQUIRE(std::abs(sorted_eigs[i] - expected[i]) < 1.0e-10);
    }
}

TEST_CASE("stam_eigenstructure: throws on N < 3",
          "[shell][loop][stam][eigenstructure][validation]")
{
    using chladni::shell::loop::stam_eigenstructure;
    REQUIRE_THROWS_AS(stam_eigenstructure(2), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_eigenstructure(0), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_eigenstructure(-1), std::invalid_argument);
}
