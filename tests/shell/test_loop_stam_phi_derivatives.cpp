/**
 * @file test_loop_stam_phi_derivatives.cpp
 * @brief S.4 — derivatives of the Stam eigenbasis Phi(v, w).
 *
 * Pins
 * @ref chladni::shell::loop::stam_phi_grad and
 * @ref chladni::shell::loop::stam_phi_hess to:
 *
 *  - **Central differences vs analytic** at fixed (v, w):
 *      stam_phi_grad[:, 0] ≈ (stam_phi(v+h, w) - stam_phi(v-h, w)) / (2h)
 *      stam_phi_grad[:, 1] ≈ (stam_phi(v, w+h) - stam_phi(v, w-h)) / (2h)
 *      stam_phi_hess[:, 0] ≈ (stam_phi(v+h, w) - 2 phi(v, w) + stam_phi(v-h, w)) / h^2
 *      stam_phi_hess[:, 1] ≈ mixed central second difference
 *      stam_phi_hess[:, 2] ≈ ditto for w^2
 *    Picks (v, w) STRICTLY inside one dyadic tile so that h-perturbed
 *    points stay in the same tile (otherwise the affine map signs
 *    flip and central differences are meaningless).
 *
 *  - **End-to-end vs direct multi-pass subdivision** for the surface
 *    gradient: with arbitrary scalar control values c0,
 *      s_grad_v(v, w)  =  (V_inv c0)^T · stam_phi_grad(...)[: , 0]
 *    matches sigma_k * 2^n * (regular_basis_grad(v_p, w_p)[:, 0])^T ·
 *    (control values at the depth-n regular sub-element's canonical
 *    DOFs). Same comparison for w-partial.
 *
 *  - **Throws**: invalid (v, w) propagated from stam_tile_map.
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

// Mirror of build_irregular_patch in loop_stam.cpp.
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

TEST_CASE("stam_phi_grad / stam_phi_hess: shapes and constant-mode invariants",
          "[shell][loop][stam][phi][derivatives]")
{
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::stam_phi_grad;
    using chladni::shell::loop::stam_phi_hess;

    for (int N : {3, 4, 5, 6, 7, 8, 12}) {
        CAPTURE(N);
        const auto ev = make_stam_evaluator(N);
        const int K = N + 6;

        const auto pg = stam_phi_grad(ev, 0.6, 0.3);
        const auto ph = stam_phi_hess(ev, 0.6, 0.3);
        REQUIRE(pg.rows() == K); REQUIRE(pg.cols() == 2);
        REQUIRE(ph.rows() == K); REQUIRE(ph.cols() == 3);

        // Phi_0 is the constant mode (lambda_0 = 1, eigenvector =
        // (1, 1, ..., 1)^T). Its derivatives must be exactly 0
        // everywhere. Pin this for a few sample points.
        for (auto [v, w] : std::vector<std::pair<double, double>>{
                 {0.7, 0.2}, {0.4, 0.3}, {0.2, 0.6},
                 {0.3, 0.05}, {0.05, 0.3}, {0.15, 0.15}}) {
            CAPTURE(v, w);
            const auto pgi = stam_phi_grad(ev, v, w);
            const auto phi_h = stam_phi_hess(ev, v, w);
            REQUIRE(pgi(0, 0) == Catch::Approx(0.0).margin(1.0e-12));
            REQUIRE(pgi(0, 1) == Catch::Approx(0.0).margin(1.0e-12));
            REQUIRE(phi_h(0, 0) == Catch::Approx(0.0).margin(1.0e-12));
            REQUIRE(phi_h(0, 1) == Catch::Approx(0.0).margin(1.0e-12));
            REQUIRE(phi_h(0, 2) == Catch::Approx(0.0).margin(1.0e-12));
        }
    }
}

TEST_CASE("stam_phi_grad: central-difference of stam_phi (within one tile)",
          "[shell][loop][stam][phi][derivatives]")
{
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::stam_phi;
    using chladni::shell::loop::stam_phi_grad;
    using chladni::shell::loop::stam_tile_map;

    // Choose interior (v, w) STRICTLY inside one dyadic tile (away
    // from tile boundaries), so that (v +/- h, w) and (v, w +/- h)
    // stay in the same tile for the chosen h. The affine tile map
    // changes sign across tiles, so central differences across a
    // boundary would be meaningless.
    constexpr double h = 1.0e-6;
    const std::vector<std::pair<double, double>> samples = {
        {0.7, 0.2},    // n=1, tile k=1 (v=1 corner sub-tile)
        {0.3, 0.4},    // n=1, tile k=2 (medial)
        {0.2, 0.7},    // n=1, tile k=3
        {0.35, 0.05},  // n=2, tile k=1
        {0.07, 0.07},  // n=3, tile k=2
    };

    for (int N : {3, 4, 5, 6, 7, 8}) {
        CAPTURE(N);
        const auto ev = make_stam_evaluator(N);
        for (auto [v, w] : samples) {
            CAPTURE(v, w);
            // Sanity: tile of (v +/- h) matches tile of (v, w).
            const auto tm0  = stam_tile_map(v, w);
            const auto tmvm = stam_tile_map(v - h, w);
            const auto tmvp = stam_tile_map(v + h, w);
            const auto tmwm = stam_tile_map(v, w - h);
            const auto tmwp = stam_tile_map(v, w + h);
            REQUIRE(tmvm.k == tm0.k); REQUIRE(tmvm.n == tm0.n);
            REQUIRE(tmvp.k == tm0.k); REQUIRE(tmvp.n == tm0.n);
            REQUIRE(tmwm.k == tm0.k); REQUIRE(tmwm.n == tm0.n);
            REQUIRE(tmwp.k == tm0.k); REQUIRE(tmwp.n == tm0.n);

            const auto pg = stam_phi_grad(ev, v, w);
            const Eigen::VectorXd phi_vp = stam_phi(ev, v + h, w);
            const Eigen::VectorXd phi_vm = stam_phi(ev, v - h, w);
            const Eigen::VectorXd phi_wp = stam_phi(ev, v, w + h);
            const Eigen::VectorXd phi_wm = stam_phi(ev, v, w - h);

            const Eigen::VectorXd dphi_dv_fd = (phi_vp - phi_vm) / (2.0 * h);
            const Eigen::VectorXd dphi_dw_fd = (phi_wp - phi_wm) / (2.0 * h);

            for (int i = 0; i < pg.rows(); ++i) {
                CAPTURE(i);
                REQUIRE(pg(i, 0)
                        == Catch::Approx(dphi_dv_fd(i))
                            .epsilon(0).margin(1.0e-5));
                REQUIRE(pg(i, 1)
                        == Catch::Approx(dphi_dw_fd(i))
                            .epsilon(0).margin(1.0e-5));
            }
        }
    }
}

TEST_CASE("stam_phi_hess: central-difference of stam_phi (within one tile)",
          "[shell][loop][stam][phi][derivatives]")
{
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::stam_phi;
    using chladni::shell::loop::stam_phi_hess;
    using chladni::shell::loop::stam_tile_map;

    // Same tile-interior constraint as above. Use a slightly larger h
    // for second-difference stability.
    constexpr double h = 1.0e-4;
    const std::vector<std::pair<double, double>> samples = {
        {0.7, 0.2}, {0.3, 0.4}, {0.2, 0.7}, {0.35, 0.05},
    };

    for (int N : {3, 4, 5, 6, 7, 8}) {
        CAPTURE(N);
        const auto ev = make_stam_evaluator(N);
        for (auto [v, w] : samples) {
            CAPTURE(v, w);
            const auto tm0 = stam_tile_map(v, w);
            for (auto [dv, dw] :
                 std::vector<std::pair<double, double>>{
                     {-h, 0}, {h, 0}, {0, -h}, {0, h},
                     {-h, -h}, {h, -h}, {-h, h}, {h, h}}) {
                const auto tmp = stam_tile_map(v + dv, w + dw);
                REQUIRE(tmp.k == tm0.k);
                REQUIRE(tmp.n == tm0.n);
            }

            const auto ph = stam_phi_hess(ev, v, w);
            const Eigen::VectorXd phi_0  = stam_phi(ev, v,     w    );
            const Eigen::VectorXd phi_vp = stam_phi(ev, v + h, w    );
            const Eigen::VectorXd phi_vm = stam_phi(ev, v - h, w    );
            const Eigen::VectorXd phi_wp = stam_phi(ev, v,     w + h);
            const Eigen::VectorXd phi_wm = stam_phi(ev, v,     w - h);
            const Eigen::VectorXd phi_pp = stam_phi(ev, v + h, w + h);
            const Eigen::VectorXd phi_pm = stam_phi(ev, v + h, w - h);
            const Eigen::VectorXd phi_mp = stam_phi(ev, v - h, w + h);
            const Eigen::VectorXd phi_mm = stam_phi(ev, v - h, w - h);

            const Eigen::VectorXd d2_dv2 = (phi_vp - 2.0 * phi_0 + phi_vm) / (h * h);
            const Eigen::VectorXd d2_dw2 = (phi_wp - 2.0 * phi_0 + phi_wm) / (h * h);
            const Eigen::VectorXd d2_dvdw
                = (phi_pp - phi_pm - phi_mp + phi_mm) / (4.0 * h * h);

            for (int i = 0; i < ph.rows(); ++i) {
                CAPTURE(i);
                REQUIRE(ph(i, 0)
                        == Catch::Approx(d2_dv2(i))
                            .epsilon(0).margin(1.0e-3));
                REQUIRE(ph(i, 1)
                        == Catch::Approx(d2_dvdw(i))
                            .epsilon(0).margin(1.0e-3));
                REQUIRE(ph(i, 2)
                        == Catch::Approx(d2_dw2(i))
                            .epsilon(0).margin(1.0e-3));
            }
        }
    }
}

TEST_CASE("stam_phi_grad: end-to-end vs direct subdivision + regular_basis_grad",
          "[shell][loop][stam][phi][derivatives]")
{
    using chladni::shell::loop::build_patch_stencils;
    using chladni::shell::loop::canonical_regular_dofs;
    using chladni::shell::loop::loop_subdivide_n_times;
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::regular_basis_grad;
    using chladni::shell::loop::stam_eigenstructure;
    using chladni::shell::loop::stam_phi_grad;
    using chladni::shell::loop::stam_tile_map;

    constexpr std::array<Eigen::Index, 3> sub_face_idx = {1, 3, 2};
    auto tile_sign = [](int k) { return (k == 2) ? -1.0 : 1.0; };

    struct Sample { double v; double w; int n; int k; };
    const std::vector<Sample> samples = {
        {0.7, 0.2, 1, 1}, {0.3, 0.4, 1, 2}, {0.2, 0.7, 1, 3},
        {0.35, 0.05, 2, 1}, {0.05, 0.35, 2, 3}, {0.15, 0.15, 2, 2},
        {0.18, 0.04, 3, 1}, {0.04, 0.18, 3, 3}, {0.07, 0.07, 3, 2},
    };

    for (int N : {3, 4, 5, 6, 7, 8}) {
        CAPTURE(N);
        const int K = N + 6;
        const auto ev = make_stam_evaluator(N);
        const auto eig = stam_eigenstructure(N);

        Eigen::VectorXd c0(K);
        for (int i = 0; i < K; ++i) {
            c0(i) = std::sin(0.7 * (i + 1))
                  + 0.31 * std::cos(1.13 * (i + 2));
        }
        const Eigen::VectorXd c_hat = eig.V_inv * c0;

        Eigen::MatrixXd V_geom;
        Eigen::MatrixXi F;
        build_irregular_patch(N, V_geom, F);
        Eigen::MatrixXd V_test = V_geom;
        V_test.col(2) = c0;

        for (const auto& s : samples) {
            CAPTURE(s.v, s.w, s.n, s.k);
            const auto tm = stam_tile_map(s.v, s.w);
            REQUIRE(tm.n == s.n);
            REQUIRE(tm.k == s.k);

            // Stam-derived global gradient.
            const auto pg = stam_phi_grad(ev, s.v, s.w);
            const double sg_dv_stam = c_hat.dot(pg.col(0));
            const double sg_dw_stam = c_hat.dot(pg.col(1));

            // Direct: subdivide n times, locate the depth-n regular
            // sub-face, evaluate regular_basis_grad at (v_p, w_p),
            // then apply the chain-rule factor sigma_k * 2^n.
            const auto sub_n = loop_subdivide_n_times(V_test, F, s.n);
            const auto stencils = build_patch_stencils(sub_n.V_sub,
                                                       sub_n.F_sub);
            const Eigen::Index face_row = sub_face_idx[
                static_cast<std::size_t>(s.k - 1)];
            const auto dofs = canonical_regular_dofs(
                stencils[static_cast<std::size_t>(face_row)],
                sub_n.F_sub);

            const Eigen::Matrix<double, 12, 2> bg
                = regular_basis_grad(tm.v_p, tm.w_p);
            double sg_dv_local = 0.0;
            double sg_dw_local = 0.0;
            for (int slot = 0; slot < 12; ++slot) {
                const double c_slot = sub_n.V_sub(
                    dofs[static_cast<std::size_t>(slot)], 2);
                sg_dv_local += bg(slot, 0) * c_slot;
                sg_dw_local += bg(slot, 1) * c_slot;
            }
            const double factor = tile_sign(s.k) * std::pow(2.0, s.n);
            const double sg_dv_direct = factor * sg_dv_local;
            const double sg_dw_direct = factor * sg_dw_local;

            REQUIRE(sg_dv_stam
                    == Catch::Approx(sg_dv_direct)
                        .epsilon(0).margin(1.0e-9));
            REQUIRE(sg_dw_stam
                    == Catch::Approx(sg_dw_direct)
                        .epsilon(0).margin(1.0e-9));
        }
    }
}

TEST_CASE("stam_phi_hess: end-to-end vs direct subdivision + regular_basis_hess",
          "[shell][loop][stam][phi][derivatives]")
{
    using chladni::shell::loop::build_patch_stencils;
    using chladni::shell::loop::canonical_regular_dofs;
    using chladni::shell::loop::loop_subdivide_n_times;
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::regular_basis_hess;
    using chladni::shell::loop::stam_eigenstructure;
    using chladni::shell::loop::stam_phi_hess;
    using chladni::shell::loop::stam_tile_map;

    constexpr std::array<Eigen::Index, 3> sub_face_idx = {1, 3, 2};

    struct Sample { double v; double w; int n; int k; };
    const std::vector<Sample> samples = {
        {0.7, 0.2, 1, 1}, {0.3, 0.4, 1, 2}, {0.2, 0.7, 1, 3},
        {0.35, 0.05, 2, 1}, {0.05, 0.35, 2, 3}, {0.15, 0.15, 2, 2},
    };

    for (int N : {3, 4, 5, 6, 7, 8}) {
        CAPTURE(N);
        const int K = N + 6;
        const auto ev = make_stam_evaluator(N);
        const auto eig = stam_eigenstructure(N);

        Eigen::VectorXd c0(K);
        for (int i = 0; i < K; ++i) {
            c0(i) = std::sin(0.7 * (i + 1))
                  + 0.31 * std::cos(1.13 * (i + 2));
        }
        const Eigen::VectorXd c_hat = eig.V_inv * c0;

        Eigen::MatrixXd V_geom;
        Eigen::MatrixXi F;
        build_irregular_patch(N, V_geom, F);
        Eigen::MatrixXd V_test = V_geom;
        V_test.col(2) = c0;

        for (const auto& s : samples) {
            CAPTURE(s.v, s.w, s.n, s.k);
            const auto tm = stam_tile_map(s.v, s.w);

            const auto ph = stam_phi_hess(ev, s.v, s.w);
            const double sh_vv_stam = c_hat.dot(ph.col(0));
            const double sh_vw_stam = c_hat.dot(ph.col(1));
            const double sh_ww_stam = c_hat.dot(ph.col(2));

            const auto sub_n = loop_subdivide_n_times(V_test, F, s.n);
            const auto stencils = build_patch_stencils(sub_n.V_sub,
                                                       sub_n.F_sub);
            const Eigen::Index face_row = sub_face_idx[
                static_cast<std::size_t>(s.k - 1)];
            const auto dofs = canonical_regular_dofs(
                stencils[static_cast<std::size_t>(face_row)],
                sub_n.F_sub);

            const Eigen::Matrix<double, 12, 3> bh
                = regular_basis_hess(tm.v_p, tm.w_p);
            double sh_vv_local = 0.0, sh_vw_local = 0.0, sh_ww_local = 0.0;
            for (int slot = 0; slot < 12; ++slot) {
                const double c_slot = sub_n.V_sub(
                    dofs[static_cast<std::size_t>(slot)], 2);
                sh_vv_local += bh(slot, 0) * c_slot;
                sh_vw_local += bh(slot, 1) * c_slot;
                sh_ww_local += bh(slot, 2) * c_slot;
            }
            const double factor = std::pow(4.0, s.n);  // no sign for hessian

            REQUIRE(sh_vv_stam
                    == Catch::Approx(factor * sh_vv_local)
                        .epsilon(0).margin(1.0e-8));
            REQUIRE(sh_vw_stam
                    == Catch::Approx(factor * sh_vw_local)
                        .epsilon(0).margin(1.0e-8));
            REQUIRE(sh_ww_stam
                    == Catch::Approx(factor * sh_ww_local)
                        .epsilon(0).margin(1.0e-8));
        }
    }
}

TEST_CASE("stam_phi_grad / stam_phi_hess: throws on invalid (v, w)",
          "[shell][loop][stam][phi][derivatives][validation]")
{
    using chladni::shell::loop::make_stam_evaluator;
    using chladni::shell::loop::stam_phi_grad;
    using chladni::shell::loop::stam_phi_hess;

    const auto ev = make_stam_evaluator(6);
    REQUIRE_THROWS_AS(stam_phi_grad(ev, -0.1,  0.5), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi_grad(ev,  0.5, -0.1), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi_grad(ev,  0.7,  0.5), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi_grad(ev,  0.0,  0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi_hess(ev, -0.1,  0.5), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi_hess(ev,  0.5, -0.1), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi_hess(ev,  0.7,  0.5), std::invalid_argument);
    REQUIRE_THROWS_AS(stam_phi_hess(ev,  0.0,  0.0), std::invalid_argument);
}
