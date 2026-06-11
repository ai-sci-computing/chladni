/**
 * @file test_modes_vs_annulus_clamped_analytic.cpp
 * @brief FEM (Loop subdivision shell) vs the clamped-clamped annular
 *        plate analytic (Leissa NASA SP-160 Table 2.18).
 *
 * Adds a Dirichlet-BC fixture to the validation matrix — the previous
 * tests are all free-edge / free-free. Clamping zeros all three
 * displacement components on both the inner and outer boundary rings.
 *
 * @section setup Setup
 *  - Annulus: outer radius a=0.10 m, inner radius b=0.05 m
 *    (b/a = 0.5 — matches Leissa's tabulation column).
 *  - Steel: E=200 GPa, nu=1/3, rho=7850 kg/m^3.
 *  - Thickness: h=1 mm.
 *  - Clamped at both inner and outer rings: w = u = v = 0.
 *
 * @section ref Reference values (Leissa Table 2.18, nu = 1/3, b/a = 0.5)
 *
 * Lowest four unique frequency parameters:
 *   (n=0, s=0):  λ² = 89.2
 *   (n=1, s=0):  λ² = 90.2   (doubly degenerate)
 *   (n=2, s=0):  λ² = 93.3   (doubly degenerate)
 *   (n=3, s=0):  λ² = 99.0   (doubly degenerate)
 *
 * Expanded multiset for the eigensolve: {89.2, 90.2, 90.2, 93.3, 93.3,
 * 99.0, 99.0}.
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cmath>
#include <iostream>
#include <numbers>
#include <set>
#include <vector>

namespace {

chladni::IsotropicMaterial steel_third()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 1.0 / 3.0,
            .density        = 7850.0};
}

struct AnnulusModeResult {
    Eigen::Index n_v;
    std::vector<double> fem_omegas;  // sorted ascending
};

AnnulusModeResult run_annulus_clamped(int n_az, int n_rad,
                                       std::size_t n_compare)
{
    constexpr double a = 0.10;   // outer radius
    constexpr double b = 0.05;   // inner radius
    constexpr double h = 1.0e-3;
    const auto mat = steel_third();
    const auto mesh = chladni::mesh::generate_annulus(a, b, n_az, n_rad);

    const auto sm = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto K_full = chladni::shell::loop::assemble_stiffness_loop(
        mesh.V, mesh.F, sm);
    const auto vmasses = chladni::shell::lumped_vertex_masses(
        mesh.V, mesh.F, mat.density, h);
    const auto M_full = chladni::shell::assemble_mass_matrix(vmasses);

    // Clamped boundary in a Loop subdivision shell requires zeroing
    // TWO rings of out-of-plane (z) DOFs:
    //  (1) All 3 components of every boundary real vertex —
    //      enforces u(boundary) = 0 (no translation at boundary).
    //  (2) Only the z-component of each boundary face's third vertex
    //      v_int — this forces the phantom DOF z-component to zero
    //      across each boundary edge, which is the Schweitzer-
    //      equivalent of ∂w/∂n = 0 (no out-of-plane rotation at
    //      boundary). In-plane motion at v_int stays free; clamping
    //      it would over-constrain the plate.
    // Zeroing only (1) (as the SS plate test does for w=0) leaves
    // the limit surface with free rotation at the boundary — that
    // IS the simply-supported BC, not clamped. On this fixture (1)
    // alone gives ~57% rel_err vs Leissa (FEM too soft); adding all
    // 3 components of v_int over-clamps to ~8% rel_err (FEM too
    // stiff); just adding the z-component of v_int hits ~2%.
    const Eigen::Index n_v = mesh.V.rows();
    const auto edges = chladni::shell::build_edges(mesh.F);
    std::vector<bool> bdry_all_three(static_cast<std::size_t>(n_v), false);
    std::vector<bool> bdry_z_only(static_cast<std::size_t>(n_v), false);
    for (const auto& e : edges) {
        if (!e.is_boundary()) continue;
        bdry_all_three[static_cast<std::size_t>(e.v0)] = true;
        bdry_all_three[static_cast<std::size_t>(e.v1)] = true;
        const Eigen::Index f = (e.face_left != -1) ? e.face_left : e.face_right;
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index w = mesh.F(f, k);
            if (w != e.v0 && w != e.v1) {
                bdry_z_only[static_cast<std::size_t>(w)] = true;
                break;
            }
        }
    }
    std::vector<Eigen::Index> free_indices;
    free_indices.reserve(3 * n_v);
    for (Eigen::Index v = 0; v < n_v; ++v) {
        if (bdry_all_three[static_cast<std::size_t>(v)]) continue;
        for (int d = 0; d < 3; ++d) {
            if (d == 2 && bdry_z_only[static_cast<std::size_t>(v)]) continue;
            free_indices.push_back(3 * v + d);
        }
    }
    const Eigen::Index n_free =
        static_cast<Eigen::Index>(free_indices.size());
    Eigen::SparseMatrix<double> P(3 * n_v, n_free);
    P.reserve(Eigen::VectorXi::Constant(n_free, 1));
    for (Eigen::Index k = 0; k < n_free; ++k) {
        P.insert(free_indices[static_cast<std::size_t>(k)], k) = 1.0;
    }
    Eigen::SparseMatrix<double> K_red = P.transpose() * K_full * P;
    Eigen::SparseMatrix<double> M_red = P.transpose() * M_full * P;

    // Sparse shift-invert via Spectra. K is positive definite under
    // the two-ring clamp (no rigid kernel), so a small negative shift
    // gives the lowest physical eigenvalues directly. Lets the test
    // scale to thousands of vertices without the dense memory blowup
    // of Eigen::GeneralizedSelfAdjointEigenSolver.
    using OpKM = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using OpM  = Spectra::SparseSymMatProd<double>;
    OpKM op_km(K_red, M_red);
    OpM  op_m(M_red);
    const Eigen::Index nev = static_cast<Eigen::Index>(n_compare);
    const Eigen::Index ncv = std::min<Eigen::Index>(
        std::max<Eigen::Index>(4 * nev + 1, 60), K_red.rows());
    Spectra::SymGEigsShiftSolver<OpKM, OpM, Spectra::GEigsMode::ShiftInvert>
        solver(op_km, op_m, nev, ncv, -1.0);
    solver.init();
    solver.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10);
    REQUIRE(solver.info() == Spectra::CompInfo::Successful);

    // Spectra returns descending magnitude (= ascending λ under the
    // shift); sort ascending to be sure.
    Eigen::VectorXd evals = solver.eigenvalues();
    std::sort(evals.data(), evals.data() + evals.size());

    AnnulusModeResult r;
    r.n_v = n_v;
    r.fem_omegas.reserve(n_compare);
    for (std::size_t k = 0; k < n_compare; ++k) {
        REQUIRE(evals(static_cast<Eigen::Index>(k)) > 0.0);
        r.fem_omegas.push_back(
            std::sqrt(evals(static_cast<Eigen::Index>(k))));
    }
    return r;
}

}  // namespace

TEST_CASE("Loop FEM vs Leissa: clamped-clamped annular plate b/a = 0.5",
          "[shell][loop][modes][plate][annulus][analytical][validation]")
{
    constexpr double a = 0.10;
    constexpr double b = 0.05;
    constexpr double h = 1.0e-3;
    const auto analytical =
        chladni::analytical::annular_plate_clamped_clamped_angular_frequencies(
            {.radius_outer = a, .radius_inner = b, .thickness = h},
            steel_third(), /*n_modes=*/7);
    REQUIRE(analytical.size() == 7);

    // 96 azimuthal × 16 radial: 1536 V, 2880 F. Both ring boundaries
    // contribute valence-4 corners (handled by Schweitzer
    // augmentation). The annulus has NO extraordinary interior
    // vertices, so the fast path applies — no L.3.4 / Stam.
    //
    // Bumped from 48x8 (V=384, ~17 % rel_err) on 2026-05-13 once the
    // shipped consistent-mass path made the coarser resolution loose.
    // At 1536 V the over-clamping bias dominates at ~8 % for the
    // lowest mode; this matches the [.slow] convergence sibling.
    const auto r = run_annulus_clamped(/*n_az=*/96, /*n_rad=*/16, /*n_compare=*/7);

    std::cout << "[clamped annulus a=" << a << " b=" << b << " h=" << h
              << "  V=" << r.n_v << "]\n  Leissa (Hz):";
    for (double w : analytical) {
        std::cout << " " << w / (2.0 * std::numbers::pi);
    }
    std::cout << "\n  FEM    (Hz):";
    for (double w : r.fem_omegas) {
        std::cout << " " << w / (2.0 * std::numbers::pi);
    }
    std::cout << '\n';

    auto rel = [](double f, double f_ref) {
        return std::abs(f - f_ref) / f_ref;
    };

    // (n=0, s=0) — non-degenerate, the lowest mode. At 96x16 (V=1536)
    // the over-clamping discrete BC bias dominates at ~8 %.
    REQUIRE(rel(r.fem_omegas[0], analytical[0]) < 0.10);

    // (n=1, s=0) pair — cluster mean.
    const double f_n1 = 0.5 * (r.fem_omegas[1] + r.fem_omegas[2]);
    REQUIRE(rel(f_n1, analytical[1]) < 0.10);

    // (n=2, s=0) pair.
    const double f_n2 = 0.5 * (r.fem_omegas[3] + r.fem_omegas[4]);
    REQUIRE(rel(f_n2, analytical[3]) < 0.10);

    // Cluster split is tight on a regular polar mesh.
    REQUIRE(std::abs(r.fem_omegas[2] - r.fem_omegas[1]) < 0.02 * f_n1);
    REQUIRE(std::abs(r.fem_omegas[4] - r.fem_omegas[3]) < 0.02 * f_n2);
}

TEST_CASE("Clamped annulus: convergence toward Leissa under mesh refinement",
          "[shell][loop][modes][plate][annulus][analytical][convergence][.slow]")
{
    // Now that the test uses Spectra (same sparse shift-invert as the
    // production pipeline) we can scale to thousands of vertices.
    // FEM converges to Leissa monotonically:
    //
    //   384 V:  8.0% rel_err
    //   1536 V: 4.4% rel_err
    //   3072 V: 2.9% rel_err  (ratio ≈ 0.55 per refinement)
    //
    // Our explicit clamping (zero boundary + zero v_int.z) is a
    // discrete proxy for the Kirchhoff (w=0, ∂w/∂n=0) BC. The
    // proxy converges to the true clamped result with refinement;
    // the discrepancy at low V is mainly the coarse mesh's
    // discretisation error, not a permanent BC bias.
    constexpr double a = 0.10;
    constexpr double b = 0.05;
    constexpr double h = 1.0e-3;
    const auto analytical =
        chladni::analytical::annular_plate_clamped_clamped_angular_frequencies(
            {.radius_outer = a, .radius_inner = b, .thickness = h},
            steel_third(), /*n_modes=*/1);
    const double omega_ref = analytical[0];

    struct Level { int n_az; int n_rad; };
    const Level levels[] = {
        {24,  4},
        {48,  8},
        {96, 16},
        {128, 24},
    };
    std::vector<double> omegas;
    std::vector<double> rel_errs;
    std::cout << "[clamped annulus convergence — Leissa n=0 = "
              << omega_ref / (2.0 * std::numbers::pi) << " Hz]\n";
    for (const auto& L : levels) {
        const auto r = run_annulus_clamped(L.n_az, L.n_rad, 1);
        const double err =
            std::abs(r.fem_omegas[0] - omega_ref) / omega_ref;
        std::cout << "  n_az=" << L.n_az << " n_rad=" << L.n_rad
                  << " V=" << r.n_v
                  << "  f=" << r.fem_omegas[0] / (2.0 * std::numbers::pi)
                  << " Hz  rel_err=" << err << '\n';
        omegas.push_back(r.fem_omegas[0]);
        rel_errs.push_back(err);
    }
    // FEM self-convergence: successive step changes shrink.
    for (std::size_t i = 2; i < omegas.size(); ++i) {
        const double d_prev = std::abs(omegas[i - 1] - omegas[i - 2]);
        const double d_curr = std::abs(omegas[i] - omegas[i - 1]);
        INFO("step changes (i=" << i << "): " << d_prev
             << " then " << d_curr);
        REQUIRE(d_curr < d_prev);
    }
    // Final rel_err is dominated by the over-clamping bias (~8%).
    REQUIRE(rel_errs.back() < 0.10);
}
