/**
 * @file test_loop_stam_phi.cpp
 * @brief S.3c — Stam eigenbasis Phi(v, w) at points in the central
 *        irregular triangle's unit triangle.
 *
 * Pins
 * @ref chladni::shell::loop::stam_phi (combining @ref stam_eigenstructure,
 * @ref build_extended_subdivision_matrix_bar, @ref stam_picking_matrices,
 * @ref stam_tile_map and @ref regular_basis) to:
 *
 *  - **Constant reproduction**: @f$ \Phi_0(v, w) = 1 @f$ at every
 *    (v, w), since @f$ \lambda_0 = 1 @f$ and column 0 of V is the
 *    constant eigenmode (combined with row-sum-1 of @f$ \bar A @f$
 *    and partition-of-unity of @c regular_basis).
 *  - **Equivalent direct evaluation**: for arbitrary scalar control
 *    values @f$ C_0 @f$ at the K = N + 6 patch vertices, the surface
 *    @f$ s(v, w) = \hat C_0^\top \Phi(v, w) @f$ (with
 *    @f$ \hat C_0 = V^{-1} C_0 @f$, Stam Eq. 6) reproduces the value
 *    obtained by subdividing the patch @c n times and evaluating the
 *    matching regular sub-element with @c regular_basis. Tested at
 *    several (v, w) sweeping all 3 tiles and depths n = 1, 2, 3.
 *  - **Throws**: invalid N, invalid (v, w).
 *
 * N = 3 covered after S.5 (Jordan-block correction in
 * apply_lambda_pow_and_jordan_in_place).
 */

#include <chladni/shell/loop.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

// Mirror of build_irregular_patch in loop_stam.cpp — see the matching
// helper in test_loop_stam_picking_matrices.cpp for the topology spec.
void build_irregular_patch(int N,
                           Eigen::MatrixXd& V,
                           Eigen::MatrixXi& F)
{
    const int K = N + 6;
    V.resize(K, 3);
    V.row(0).setZero();
    for (int k = 1; k <= N; ++k) {
        const double angle = -2.0 * pi * static_cast<double>(k - 1)
                                       / static_cast<double>(N);
        V.row(k) << std::cos(angle), std::sin(angle), 0.0;
    }
    constexpr double R_outer = 2.0;
    auto place = [&](int idx, double angle) {
        V.row(idx) << R_outer * std::cos(angle),
                      R_outer * std::sin(angle),
                      0.0;
    };
    const double Nf = static_cast<double>(N);
    place(N + 1,  pi / Nf);
    place(N + 2, -pi / (2.0 * Nf));
    place(N + 3, -pi / Nf);
    place(N + 4,  2.0 * pi / Nf + pi / (2.0 * Nf));
    place(N + 5,  3.0 * pi / Nf);

    F.resize(N + 7, 3);
    F.row(0) << 0, 1, N;
    for (int k = 1; k <= N - 1; ++k) {
        F.row(k) << 0, k + 1, k;
    }
    F.row(N + 0) << 1, N + 1, N;
    F.row(N + 1) << 1, N + 2, N + 1;
    F.row(N + 2) << 1, N + 3, N + 2;
    F.row(N + 3) << 1, 2,     N + 3;
    F.row(N + 4) << N, N + 1, N + 4;
    F.row(N + 5) << N, N + 4, N + 5;
    F.row(N + 6) << N, N + 5, N - 1;
}

}  // namespace

TEST_CASE("stam_phi: constant reproduction Phi_0(v, w) == 1",
          "[shell][loop][stam][phi]")
{
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::stam_phi;

    // Sweep a grid of (v, w) inside the unit triangle (avoiding the
    // extraordinary vertex at the origin where stam_tile_map throws).
    const std::vector<std::pair<double, double>> samples = {
        {0.7, 0.2},   {0.5, 0.4},  {0.3, 0.6},   // n = 1, all tiles
        {0.4, 0.05},  {0.05, 0.4}, {0.2, 0.2},   // n = 2 mix
        {0.15, 0.05}, {0.05, 0.15}, {0.1, 0.1},  // n = 3 mix
    };
    for (int N : {3, 4, 5, 6, 7, 8, 12}) {
        CAPTURE(N);
        const auto ev = make_stam_evaluator(N);
        for (const auto& [v, w] : samples) {
            CAPTURE(v, w);
            const Eigen::VectorXd phi = stam_phi(ev, v, w);
            REQUIRE(phi.size() == N + 6);
            // Phi_0 corresponds to the lambda_0 = 1 mode (constant
            // eigenvector). Combined with row-sum-1 of Abar and
            // regular_basis partition of unity it must hit 1 exactly
            // (up to floating-point round-off).
            REQUIRE(phi(0) == Catch::Approx(1.0).epsilon(0).margin(1.0e-12));
        }
    }
}

TEST_CASE("stam_phi: matches direct multi-pass subdivision",
          "[shell][loop][stam][phi]")
{
    using chladni::shell::loop::build_patch_stencils;
    using chladni::shell::loop::canonical_regular_dofs;
    using chladni::shell::loop::loop_subdivide_n_times;
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::regular_basis;
    using chladni::shell::loop::stam_phi;
    using chladni::shell::loop::stam_tile_map;

    // For each tile k and each depth n, pick a (v, w) that lands in
    // tile k at level n. Then the regular sub-element at depth n in
    // tile k is row sub_face_idx[k-1] of F_sub^n (the corner-0
    // recursion keeps the irregular sub-face at row 0 across passes,
    // so the 3 regular sub-faces of the depth-(n-1) irregular always
    // sit at rows 1, 2, 3 of F_sub^n).
    constexpr std::array<Eigen::Index, 3> sub_face_idx = {1, 3, 2};

    struct Sample { double v; double w; int expect_n; int expect_k; };
    const std::vector<Sample> samples = {
        // n = 1 (v + w in (0.5, 1])
        {0.7, 0.2, 1, 1}, {0.6, 0.3, 1, 1},             // tile 1
        {0.3, 0.4, 1, 2}, {0.2, 0.4, 1, 2},             // tile 2 medial
        {0.2, 0.7, 1, 3}, {0.1, 0.6, 1, 3},             // tile 3
        // n = 2 (v + w in (0.25, 0.5])
        {0.35, 0.05, 2, 1}, {0.05, 0.35, 2, 3}, {0.15, 0.15, 2, 2},
        // n = 3
        {0.18, 0.04, 3, 1}, {0.04, 0.18, 3, 3}, {0.07, 0.07, 3, 2},
    };

    // N = 3 covered: the Jordan-block correction at lambda = 1/16
    // (Stam App. C) is applied inside stam_phi via
    // apply_lambda_pow_and_jordan_in_place — the (n-1) lambda^(n-2)
    // off-diagonal contribution from the J^{n-1} block at indices
    // (K-2, K-1) lifts phi(K-2) by jord_coef * u(K-1, pre-scaling).
    for (int N : {3, 4, 5, 6, 7, 8, 12}) {
        CAPTURE(N);
        const int K = N + 6;
        const auto ev = make_stam_evaluator(N);

        // Pre-pose a scalar test field on the K control vertices.
        Eigen::VectorXd c0(K);
        for (int i = 0; i < K; ++i) {
            c0(i) = std::sin(0.7 * (i + 1))
                  + 0.31 * std::cos(1.13 * (i + 2));
        }

        // Per Stam Eq. (6): s(v, w) = (V^{-1} c0)^T Phi(v, w). The
        // StamEvaluator caches (P_k Abar V)^T but not V itself, so
        // re-fetch the eigenstructure to get V_inv.
        const auto eig = chladni::shell::loop::stam_eigenstructure(N);
        const Eigen::VectorXd c_hat = eig.V_inv * c0;

        // Build the patch and pre-subdivide enough times for the
        // deepest sample (n_max).
        Eigen::MatrixXd V_geom;
        Eigen::MatrixXi F;
        build_irregular_patch(N, V_geom, F);
        // Stuff the scalar field into V_geom.col(2).
        Eigen::MatrixXd V_test = V_geom;
        V_test.col(2) = c0;

        int n_max = 0;
        for (const auto& s : samples) n_max = std::max(n_max, s.expect_n);

        const auto subN = loop_subdivide_n_times(V_test, F, n_max);
        const auto stencils = build_patch_stencils(subN.V_sub, subN.F_sub);

        for (const auto& s : samples) {
            CAPTURE(s.v, s.w, s.expect_n, s.expect_k);
            // Sanity: tile map agrees with our hand-picked (n, k).
            const auto tm = stam_tile_map(s.v, s.w);
            REQUIRE(tm.n == s.expect_n);
            REQUIRE(tm.k == s.expect_k);

            // Stam evaluation.
            const Eigen::VectorXd phi = stam_phi(ev, s.v, s.w);
            const double s_stam = c_hat.dot(phi);

            // Direct evaluation: locate the depth-n regular sub-face
            // in F_sub^{n_max} (we subdivided more than necessary; the
            // depth-n regular sub-face at tile k still has the same
            // index pattern as long as we descend through the corner-0
            // irregular at each level — which keeps the relevant
            // sub-face at row sub_face_idx[k-1] of the *depth-n*
            // F_sub^n. After n_max subdivisions, the depth-n regular
            // sub-face has been further subdivided 4 times per extra
            // level, so we descend further by repeating the corner-0
            // path... but that defeats the purpose).
            //
            // Simpler: subdivide *exactly* n times for each sample.
            const auto sub_n = loop_subdivide_n_times(V_test, F, s.expect_n);
            const auto sten_n = build_patch_stencils(sub_n.V_sub, sub_n.F_sub);

            const Eigen::Index face_row = sub_face_idx[
                static_cast<std::size_t>(s.expect_k - 1)];
            const auto dofs = canonical_regular_dofs(
                sten_n[static_cast<std::size_t>(face_row)],
                sub_n.F_sub);

            const Eigen::Matrix<double, 12, 1> b
                = regular_basis(tm.v_p, tm.w_p);
            double s_direct = 0.0;
            for (int slot = 0; slot < 12; ++slot) {
                s_direct += b(slot) * sub_n.V_sub(
                    dofs[static_cast<std::size_t>(slot)], 2);
            }

            INFO("s_stam = " << s_stam << "  s_direct = " << s_direct);
            REQUIRE(s_stam
                    == Catch::Approx(s_direct).epsilon(0).margin(1.0e-10));
        }
    }
}

TEST_CASE("stam_phi: throws on invalid N or (v, w)",
          "[shell][loop][stam][phi][validation]")
{
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::stam_phi;

    REQUIRE_THROWS_AS(make_stam_evaluator(2),  std::invalid_argument);
    REQUIRE_THROWS_AS(make_stam_evaluator(0),  std::invalid_argument);
    REQUIRE_THROWS_AS(make_stam_evaluator(-1), std::invalid_argument);

    const auto ev = make_stam_evaluator(6);
    REQUIRE_THROWS_AS(stam_phi(ev, -0.1,  0.5),  std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi(ev,  0.5, -0.1),  std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi(ev,  0.7,  0.5),  std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi(ev,  0.0,  0.0),  std::invalid_argument);
}
