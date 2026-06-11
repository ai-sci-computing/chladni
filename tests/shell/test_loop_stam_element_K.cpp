/**
 * @file test_loop_stam_element_K.cpp
 * @brief S.6 — per-element stiffness K_e on irregular Stam patch.
 *
 * Pins
 * @ref chladni::shell::loop::evaluate_patch_stam and
 * @ref chladni::shell::loop::element_stiffness_stam to:
 *
 *  - **Shape**: K_e is square @f$ 3K \times 3K @f$, with @f$ K = N + 6 @f$.
 *  - **Symmetry**: K_e == K_e^T (within floating-point round-off).
 *  - **Rigid-translation null space**: applying a global 3D
 *    translation @f$ \delta @f$ to every patch control vertex (i.e.
 *    setting the displacement vector @c u to @f$ (\delta_x, \delta_y,
 *    \delta_z, \delta_x, \delta_y, \delta_z, \ldots) @f$) gives zero
 *    elastic strain energy: @f$ u^\top K_e u = 0 @f$. Both membrane
 *    and bending energies must vanish — the regular kernel encodes
 *    this via @c sum_I dN_I = 0 (gradient partition of unity), and
 *    the Stam basis inherits the same property through the constant
 *    eigenmode (see @c stam_phi: Phi_0 = 1 with zero gradient and
 *    Hessian, plus the K-1 non-constant modes' control-vertex
 *    contractions cancel under uniform shifts).
 *  - **Membrane and bending decompose**: zeroing @c k_L turns off the
 *    membrane block; zeroing @c k_B turns off the bending block;
 *    setting both gives K_e = 0.
 *  - **Throws**: invalid V, mismatched patch_dofs.size(), degenerate
 *    parametrisation.
 *
 * The end-to-end pin against the regular path's K_e for valence-6
 * patches lives outside S.6 (it requires a basis-equivalence
 * transformation that S.7 will expose); for now we verify the
 * structural invariants only.
 */

#include <chladni/shell/loop.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chladni/material.hpp>
#include <chladni/shell/assembler.hpp>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

// Mirror of build_irregular_patch in loop_stam.cpp (and the matching
// helper in the picking-matrix / phi tests).
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

// Identity DOF list (0, 1, ..., K-1) for the canonical irregular patch.
std::vector<Eigen::Index> identity_dofs(int N)
{
    const int K = N + 6;
    std::vector<Eigen::Index> dofs(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) dofs[static_cast<std::size_t>(i)] = i;
    return dofs;
}

chladni::shell::ShellMaterial make_test_material()
{
    return {/*k_L=*/1.0e9, /*k_B=*/1.0e3, /*poisson_ratio=*/0.3};
}

}  // namespace

// Probe 2 of the Stam-bug investigation: K_stam vs K_L34 on a flat
// valence-5 patch, via the full LoopAssembler. If the eigenvalue
// spectra disagree on flat geometry (where bending decomposes
// cleanly), the bug is in the K-assembly kernel; if they agree on
// flat geometry, the bug surfaces only under curvature.
TEST_CASE("Stam vs L.3.4 K spectrum on a flat valence-5 patch",
          "[.experiment][shell][loop][stam][element_K]")
{
    using chladni::shell::loop::build_extended_subdivision_matrix;
    namespace cs = chladni::shell;

    const int N = 5;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    build_irregular_patch(N, V, F);   // flat, z=0

    constexpr double h = 1.0e-3;
    chladni::IsotropicMaterial mat{
        .youngs_modulus = 200.0e9, .poisson_ratio = 0.30,
        .density = 7850.0};
    const auto sm = cs::shell_material_from_isotropic(mat, h);

    auto run = [&](bool use_stam) {
        cs::LoopAssembler::Params p;
        p.use_stam = use_stam;
        p.m_lump   = cs::MassLumping::None;
        // Use the assemble_K API directly so we can inspect K.
        const cs::LoopAssembler asm_(p);
        return asm_.assemble_K(V, F, sm);
    };
    const auto K_l34  = run(/*use_stam=*/false);
    const auto K_stam = run(/*use_stam=*/true);

    REQUIRE(K_l34.rows()  == 3 * V.rows());
    REQUIRE(K_stam.rows() == 3 * V.rows());

    // Compare eigenvalues. For a flat patch the lowest 6 should be
    // ~0 (rigid body modes). The next modes (bending) should agree
    // between the two paths.
    const Eigen::MatrixXd K_l34_d  = Eigen::MatrixXd(K_l34);
    const Eigen::MatrixXd K_stam_d = Eigen::MatrixXd(K_stam);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_l34(K_l34_d);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_stam(K_stam_d);

    std::cout << "[flat valence-5 patch — V=" << V.rows()
              << "  3V=" << 3*V.rows() << "]\n"
              << "  ||K_stam - K_l34||_F = "
              << (K_stam_d - K_l34_d).norm() << '\n'
              << "  K_l34 eigvals (first 12): ";
    for (int i = 0; i < 12 && i < es_l34.eigenvalues().size(); ++i) {
        std::cout << es_l34.eigenvalues()(i) << " ";
    }
    std::cout << "\n  K_stam eigvals (first 12): ";
    for (int i = 0; i < 12 && i < es_stam.eigenvalues().size(); ++i) {
        std::cout << es_stam.eigenvalues()(i) << " ";
    }
    std::cout << '\n';
}

TEST_CASE("element_stiffness_stam: shape and symmetry",
          "[shell][loop][stam][element_K]")
{
    using chladni::shell::loop::element_stiffness_stam;
    using chladni::shell::loop::evaluate_patch_stam;
    using chladni::shell::loop::make_stam_evaluator;

    const auto material = make_test_material();

    for (int N : {3, 4, 5, 6, 7, 8, 12}) {
        CAPTURE(N);
        const int K = N + 6;
        const auto ev = make_stam_evaluator(N);

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        build_irregular_patch(N, V, F);

        const auto dofs = identity_dofs(N);

        // Sanity check that evaluate_patch_stam returns the expected
        // shape at one point — the actual K assembly evaluates at
        // 7 Dunavant points internally.
        const auto pe = evaluate_patch_stam(
            ev, dofs, V, 1.0 / 3.0, 1.0 / 3.0);
        REQUIRE(pe.N.size() == K);
        REQUIRE(pe.N_grad.rows() == K); REQUIRE(pe.N_grad.cols() == 2);
        REQUIRE(pe.N_hess.rows() == K); REQUIRE(pe.N_hess.cols() == 3);

        const auto K_e = element_stiffness_stam(ev, dofs, V, material);
        REQUIRE(K_e.rows() == 3 * K);
        REQUIRE(K_e.cols() == 3 * K);

        const double sym_err = (K_e - K_e.transpose()).cwiseAbs().maxCoeff();
        const double max_abs = K_e.cwiseAbs().maxCoeff();
        // Allow O(eps * max_abs) asymmetry from FP non-associativity.
        REQUIRE(sym_err <= 1.0e-9 * std::max(1.0, max_abs));
    }
}

TEST_CASE("element_stiffness_stam: rigid-translation null space",
          "[shell][loop][stam][element_K]")
{
    using chladni::shell::loop::element_stiffness_stam;
    using chladni::shell::loop::evaluate_patch_stam;
    using chladni::shell::loop::make_stam_evaluator;

    const auto material = make_test_material();

    for (int N : {3, 4, 5, 6, 7, 8, 12}) {
        CAPTURE(N);
        const int K = N + 6;
        const auto ev = make_stam_evaluator(N);

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        build_irregular_patch(N, V, F);

        const auto dofs = identity_dofs(N);
        const auto K_e = element_stiffness_stam(ev, dofs, V, material);

        // Build the 3 rigid-translation displacement modes.
        for (int axis = 0; axis < 3; ++axis) {
            CAPTURE(axis);
            Eigen::VectorXd u = Eigen::VectorXd::Zero(3 * K);
            for (int i = 0; i < K; ++i) u(3 * i + axis) = 1.0;

            const Eigen::VectorXd Ku = K_e * u;
            const double e_strain = u.dot(Ku);

            // Both membrane and bending energies must vanish under a
            // pure global translation. Tolerance scaled by k_L *
            // typical-area to absorb FP round-off in the rank-deficient
            // K_e.
            const double tol = 1.0e-6
                * std::max(material.k_L, material.k_B);
            REQUIRE(std::abs(e_strain) < tol);
            // K * u as an L2 vector — also small (each component is a
            // sum of N_grad * a contributions that should cancel).
            REQUIRE(Ku.cwiseAbs().maxCoeff() < tol);
        }
    }
}

TEST_CASE("element_stiffness_stam: membrane / bending decompose",
          "[shell][loop][stam][element_K]")
{
    using chladni::shell::loop::element_stiffness_stam;
    using chladni::shell::loop::evaluate_patch_stam;
    using chladni::shell::loop::make_stam_evaluator;

    const int N = 5;
    const int K = N + 6;
    const auto ev = make_stam_evaluator(N);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    build_irregular_patch(N, V, F);
    const auto dofs = identity_dofs(N);
    const auto pe = evaluate_patch_stam(
        ev, dofs, V, 1.0 / 3.0, 1.0 / 3.0);

    const chladni::shell::ShellMaterial mat_full = {1.0e9, 1.0e3, 0.3};
    const chladni::shell::ShellMaterial mat_memb = {1.0e9, 0.0,   0.3};
    const chladni::shell::ShellMaterial mat_bend = {0.0,   1.0e3, 0.3};
    const chladni::shell::ShellMaterial mat_zero = {0.0,   0.0,   0.3};

    const auto K_full = element_stiffness_stam(ev, dofs, V, mat_full);
    const auto K_memb = element_stiffness_stam(ev, dofs, V, mat_memb);
    const auto K_bend = element_stiffness_stam(ev, dofs, V, mat_bend);
    const auto K_zero = element_stiffness_stam(ev, dofs, V, mat_zero);

    REQUIRE(K_zero.cwiseAbs().maxCoeff() < 1.0e-12);
    const double sum_err = (K_full - (K_memb + K_bend))
                                .cwiseAbs().maxCoeff();
    REQUIRE(sum_err < 1.0e-9 * K_full.cwiseAbs().maxCoeff());
}

TEST_CASE("element_stiffness_stam: positive semi-definite",
          "[shell][loop][stam][element_K]")
{
    using chladni::shell::loop::element_stiffness_stam;
    using chladni::shell::loop::evaluate_patch_stam;
    using chladni::shell::loop::make_stam_evaluator;

    const auto material = make_test_material();

    for (int N : {3, 4, 5, 6, 7, 8}) {
        CAPTURE(N);
        const auto ev = make_stam_evaluator(N);

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        build_irregular_patch(N, V, F);
        const auto dofs = identity_dofs(N);
        const auto pe = evaluate_patch_stam(
            ev, dofs, V, 1.0 / 3.0, 1.0 / 3.0);
        Eigen::MatrixXd K_e = element_stiffness_stam(ev, dofs, V, material);
        // Symmetrise to avoid SelfAdjointEigenSolver tripping on FP
        // asymmetry that's negligible against |K_e|.
        K_e = 0.5 * (K_e + K_e.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K_e);
        REQUIRE(es.info() == Eigen::Success);
        const double smallest = es.eigenvalues().minCoeff();
        const double largest  = es.eigenvalues().maxCoeff();
        // Non-negative within FP tolerance scaled by the matrix norm.
        REQUIRE(smallest > -1.0e-7 * std::max(1.0, largest));
    }
}

TEST_CASE("evaluate_patch_stam / element_stiffness_stam: throws",
          "[shell][loop][stam][element_K][validation]")
{
    using chladni::shell::loop::element_stiffness_stam;
    using chladni::shell::loop::evaluate_patch_stam;
    using chladni::shell::loop::make_stam_evaluator;

    const int N = 6;
    const auto ev = make_stam_evaluator(N);
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    build_irregular_patch(N, V, F);

    // wrong dofs size
    REQUIRE_THROWS_AS(
        evaluate_patch_stam(ev, std::vector<Eigen::Index>{0, 1}, V,
                            1.0 / 3.0, 1.0 / 3.0),
        std::invalid_argument);

    // V too narrow
    Eigen::MatrixXd V_narrow(V.rows(), 2);
    V_narrow << V.col(0), V.col(1);
    REQUIRE_THROWS_AS(
        evaluate_patch_stam(ev, identity_dofs(N), V_narrow,
                            1.0 / 3.0, 1.0 / 3.0),
        std::invalid_argument);

    // (v, w) outside the unit triangle (propagated from stam_tile_map).
    REQUIRE_THROWS_AS(
        evaluate_patch_stam(ev, identity_dofs(N), V, 0.7, 0.5),
        std::invalid_argument);

    // Degenerate parametrisation: zero out the patch positions so cov_basis = 0.
    // element_stiffness_stam now evaluates the patch internally at every
    // quadrature point and throws at the first degenerate point.
    Eigen::MatrixXd V_degen = Eigen::MatrixXd::Zero(N + 6, 3);
    REQUIRE_THROWS_AS(
        element_stiffness_stam(
            ev, identity_dofs(N), V_degen, make_test_material()),
        std::runtime_error);
}
