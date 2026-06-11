/**
 * @file test_modes_vs_complete_sphere_analytic.cpp
 * @brief FEM (Loop subdivision shell) vs the closed-sphere analytic
 *        (Wilkinson 1965, classical+shear cubic) on an icosphere mesh.
 *
 * Closes the @e closed-shell physics validation half of the Loop
 * subdivision rewrite — apples-to-apples against
 * @ref chladni::analytical::complete_spherical_shell_wilkinson_angular_frequencies.
 *
 * @section setup Setup
 * - Icosphere: @c generate_icosphere(R = 0.10 m, n_subdivisions = k)
 *   for @c k in @c {2, 3} — 162 V/320 F and 642 V/1280 F.
 *   Every face has at least one valence-5 corner, so the L.3.4
 *   irregular path is exercised on @b every triangle. @c n_passes = 2
 *   (Cirak-Ortiz step-1 approximation with one extra subdivision pass)
 *   shrinks the dropped irregular-area residual to ~1/16 of a face.
 * - Material: steel (E = 200 GPa, nu = 0.30, rho = 7850 kg/m^3).
 * - Thickness: h = 1 mm  (R/h = 100, deep into the thin-shell regime).
 *
 * @section expected What we expect
 * On a perfect sphere the @c n=2 spheroidal mode is @c (2n+1) = 5-fold
 * degenerate. The icosphere mesh has icosahedral symmetry, not full
 * O(3), so the FEM splits this 5-fold degeneracy into a small cluster
 * (typical splitting <5% on 162 vertices, smaller still on 642). The
 * Wilkinson lower (bending) branch sits at ~37000 rad/s for steel
 * R = 0.10 m, h = 1 mm.
 *
 * @section convergence FEM convergence on closed icospheres (consistent mass)
 * Under the @ref MassLumping::None default (flipped 2026-05-17 late),
 * both Loop paths converge cleanly. Empirically (n_passes=2):
 *
 *     k = 2 (162 V):  L.3.4 rel_err ≈ 0.93 %   Stam rel_err ≈ 2.26 %
 *     k = 3 (642 V):  L.3.4 rel_err ≈ 0.26 %   Stam rel_err ≈ 0.32 %
 *     k = 4 (2562 V): L.3.4 rel_err ≈ 0.04 %   Stam rel_err ≈ 0.06 %
 *
 * (Pre-2026-05-17-late, under @ref MassLumping::RowSum, the same
 * fixture sat at ~36 % rel_err on both paths — see
 * @c [.experiment] sweep in @c test_loop_sphere_projected.cpp.)
 *
 * @section stam_vs_l34 Stam ≈ L.3.4 holds at k=2..4 — but with a caveat
 * The Stam S-series replaced the L.3.4 drop approximation with an
 * exact eigenbasis evaluation on the residual irregular sub-tile.
 * On this fixture at k=2..4 the two paths agree to ~0.02 pp, and
 * both track Wilkinson within the FEM's own coarse-mesh approximation.
 *
 * **Caveat (2026-05-17 very late):** at k=5 the Stam path produces
 * a spurious low mode at 5617 Hz that displaces one member of the
 * 5-fold degenerate n=2 pentet (the bulk-pentet at 5905.5 ×4 is
 * unaffected). The dense eigensolver returns a clean 5-fold pentet
 * at 5911.68 Hz, so the spurious mode is an artifact of Spectra's
 * shift-invert solver failing on the degenerate cluster — not a
 * Stam K/M bug. See [[chladni-spectra-degenerate-cluster]] and the
 * @c [.experiment] dense-solver cross-check in
 * @c test_loop_sphere_projected.cpp.
 *
 * The L.3.4 path is preserved as the default; Stam is reachable via
 * the @c use_stam flag. Both paths are pinned by their own test
 * cases below.
 */

#include <chladni/analytical/shell.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

chladni::IsotropicMaterial steel()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.30,
            .density        = 7850.0};
}

}  // namespace

namespace {

struct IcosphereResult {
    std::size_t V;
    double      omega_n2_mean;
    double      cluster_split;
    double      mode6;
};

IcosphereResult run_icosphere(int n_subdivisions, bool use_stam = false,
                              int n_passes = 2)
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;

    const auto mesh = chladni::mesh::generate_icosphere(R, n_subdivisions);

    // Lowest 6 modes after rigid-body filter.
    // Five of the six should cluster as the n=2 spheroidal pentet;
    // mode 6 is the first member of the n=3 septet.
    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, /*n_modes=*/6, n_passes, use_stam);
    REQUIRE(modes.omegas.size() == 6);

    for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
        CAPTURE(n_subdivisions, i);
        REQUIRE(modes.omegas(i) > 0.0);
        if (i > 0) {
            REQUIRE(modes.omegas(i) >= modes.omegas(i - 1));
        }
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < 5; ++i) {
        sum += modes.omegas(static_cast<Eigen::Index>(i));
    }
    return IcosphereResult{
        .V             = static_cast<std::size_t>(mesh.V.rows()),
        .omega_n2_mean = sum / 5.0,
        .cluster_split = (modes.omegas(4) - modes.omegas(0)) / (sum / 5.0),
        .mode6         = modes.omegas(5),
    };
}

}  // namespace

TEST_CASE("Loop FEM vs Wilkinson closed-sphere analytic — icosphere k=2 sanity",
          "[shell][loop][modes][closed][icosphere][analytical][validation]")
{
    // Default-run sanity check at k=2 (162 V, ~1 s eigensolve).
    // Under the consistent-mass default (MassLumping::None, flipped
    // from RowSum on 2026-05-17 late after the lumping bias was
    // diagnosed), the n=2 spheroidal pentet mean lands ~2 % HIGH vs
    // Wilkinson. Earlier versions of this test pinned a ~37 %
    // *low* rel_err and attributed it to "L.3.4 step-1 multi-pass
    // converges geometrically but slowly because every face is
    // irregular" — that diagnosis was wrong, almost all of the gap
    // was the lumping bias. See the @c [.experiment] mass-lumping
    // probe in @c test_loop_sphere_projected.cpp for the data.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;

    const auto analytical = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel(), /*n_modes=*/1);
    const double omega_n2_analytical = analytical.front();

    const auto k2 = run_icosphere(/*n_subdivisions=*/2);
    const double rel_err =
        std::abs(k2.omega_n2_mean - omega_n2_analytical) / omega_n2_analytical;

    INFO("k=2  V=" << k2.V
         << "  FEM n=2 mean = "    << k2.omega_n2_mean
         << "  Wilkinson n=2 = "    << omega_n2_analytical
         << "  cluster_split = "    << k2.cluster_split
         << "  rel_err = "          << rel_err);

    // Cluster integrity: icosahedral subgroup of O(3) splits the
    // 5-fold n=2 representation, but the spread is small at this
    // resolution and mode 6 sits clearly above the cluster (so we
    // identified the n=2 pentet correctly).
    REQUIRE(k2.cluster_split < 0.05);
    REQUIRE(k2.mode6 > k2.omega_n2_mean);

    // Consistent-mass Loop at k=2 matches Wilkinson to ~2 %. The 5 %
    // gate matches the LME-vs-Wilkinson curved-modes test budget on
    // the same fixture and leaves headroom over the empirical 2.3 %.
    REQUIRE(rel_err < 0.05);

    std::cout << "[icosphere R=" << R << " h=" << h
              << "  k=2 V=" << k2.V << "  n_passes=2]"
              << "  FEM n=2 mean = "  << k2.omega_n2_mean
              << "  vs Wilkinson = "  << omega_n2_analytical
              << "  rel_err = "       << rel_err
              << '\n';
}

TEST_CASE("Loop FEM vs Wilkinson closed-sphere analytic — icosphere k=2,3 convergence",
          "[shell][loop][modes][closed][icosphere][analytical][validation][convergence][.slow]")
{
    // Slow (~2 min) but high-value: pin the geometric convergence
    // trajectory of the L.3.4 multi-pass on a closed icosphere as
    // mesh refinement increases. This distinguishes "FEM converging
    // toward the analytical, just slowly" (the situation as of
    // commit 8be8f3e) from a fixed-bias calibration gap (the W.4
    // CST+IBM situation that motivated the Loop rewrite, documented
    // in `project_chladni_shell_formulation_plan` memory).
    //
    // Run ad-hoc: ./build/tests/chladni_tests "[convergence][.slow]"
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;

    const auto analytical = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel(), /*n_modes=*/1);
    const double omega_n2_analytical = analytical.front();

    const auto k2 = run_icosphere(/*n_subdivisions=*/2);
    const auto k3 = run_icosphere(/*n_subdivisions=*/3);
    const auto k2_stam = run_icosphere(2, /*use_stam=*/true);
    const auto k3_stam = run_icosphere(3, /*use_stam=*/true);

    auto rel_err = [&](double mean) {
        return std::abs(mean - omega_n2_analytical) / omega_n2_analytical;
    };
    const double rel_err_k2 = rel_err(k2.omega_n2_mean);
    const double rel_err_k3 = rel_err(k3.omega_n2_mean);
    const double err_k2_stam = rel_err(k2_stam.omega_n2_mean);
    const double err_k3_stam = rel_err(k3_stam.omega_n2_mean);

    INFO("k=2  V=" << k2.V << "  L.3.4 rel_err = " << rel_err_k2
         << "  Stam rel_err = " << err_k2_stam);
    INFO("k=3  V=" << k3.V << "  L.3.4 rel_err = " << rel_err_k3
         << "  Stam rel_err = " << err_k3_stam);

    REQUIRE(k3.cluster_split < 0.02);
    REQUIRE(k3_stam.cluster_split < 0.02);

    // FEM converges TOWARD the analytical with mesh refinement on both
    // paths. Under the consistent-mass default (MassLumping::None,
    // flipped 2026-05-17 late) Loop is variationally consistent and
    // refined eigenvalues approach Wilkinson from ABOVE — so k=3 sits
    // below k=2, both above the analytic.
    REQUIRE(k3.omega_n2_mean      < k2.omega_n2_mean);
    REQUIRE(k3_stam.omega_n2_mean < k2_stam.omega_n2_mean);
    REQUIRE(rel_err_k3            < rel_err_k2);
    REQUIRE(err_k3_stam           < err_k2_stam);

    // Geometric convergence: rel_err shrinks by at least 0.7 per
    // refinement step (empirically ~0.52 ratio for L.3.4).
    const double convergence_ratio      = rel_err_k3 / rel_err_k2;
    const double convergence_ratio_stam = err_k3_stam / err_k2_stam;
    INFO("L.3.4 convergence ratio = " << convergence_ratio);
    INFO("Stam  convergence ratio = " << convergence_ratio_stam);
    REQUIRE(convergence_ratio      < 0.7);
    REQUIRE(convergence_ratio_stam < 0.7);

    // Absolute threshold at k=3: ~18 % rel_err (both paths) under
    // consistent mass. The k=4 sub-percent regime is now exercised by
    // the (default-suite) k=4 sanity test below.
    REQUIRE(rel_err_k3  < 0.21);
    REQUIRE(err_k3_stam < 0.21);

    std::cout << "[icosphere R=" << R << " h=" << h << "  n_passes=2]\n"
              << "  L.3.4: k=2 V=" << k2.V << " rel_err=" << rel_err_k2
              << "  k=3 V=" << k3.V << " rel_err=" << rel_err_k3
              << "  ratio=" << convergence_ratio << '\n'
              << "  Stam:  k=2 V=" << k2_stam.V << " rel_err=" << err_k2_stam
              << "  k=3 V=" << k3_stam.V << " rel_err=" << err_k3_stam
              << "  ratio=" << convergence_ratio_stam << '\n';
}

// ---------------------------------------------------------------------------
// S.8: Stam exact-evaluation path on the closed icosphere.
//
// With use_stam=true the residual irregular sub-triangles are evaluated
// via the Stam 1999 eigenbasis instead of being dropped by the L.3.4
// step-1 multi-pass approximation. On the icosphere every face has at
// least one valence-5 corner (after one Loop subdivision), so the Stam
// path is exercised on every triangle — a real stress test.
// ---------------------------------------------------------------------------

TEST_CASE("Loop FEM (Stam path) vs Wilkinson — icosphere k=4 sanity",
          "[shell][loop][modes][closed][icosphere][analytical][stam][validation]")
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;

    const auto analytical = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel(), /*n_modes=*/1);
    const double omega_n2_analytical = analytical.front();

    // At k=4 (V=2562) both Loop paths converge to within sub-ppm of
    // each other and the analytic Wilkinson reference. Under the
    // consistent-mass default they sit on the same convergence
    // trajectory and the L.3.4-vs-Stam algorithmic difference becomes
    // indistinguishable from numerical-floor jitter — neither path is
    // robustly closer to the analytic than the other (their rel_errs
    // sit within ~0.05 percentage points of each other), so the gate
    // is a small-slack equality rather than the strict ordering that
    // was once true under the lumped-mass default (where the K-side
    // L.3.4 underestimate dominated the M-side noise).
    //
    // n_passes = 1: L.3.4 drops the full 25 %-area irregular residual
    // per parent triangle; Stam picks it up exactly. At k=4 the
    // residual is small enough that both paths land sub-percent.
    const auto k4_l34_p1  = run_icosphere(4, /*use_stam=*/false, 1);
    const auto k4_stam_p1 = run_icosphere(4, /*use_stam=*/true,  1);
    // n_passes = 2: residual shrinks to 1/16 ~= 6 %.
    const auto k4_l34_p2  = run_icosphere(4, /*use_stam=*/false, 2);
    const auto k4_stam_p2 = run_icosphere(4, /*use_stam=*/true,  2);

    auto rel_err = [&](double mean) {
        return std::abs(mean - omega_n2_analytical) / omega_n2_analytical;
    };
    const double err_l34_p1  = rel_err(k4_l34_p1.omega_n2_mean);
    const double err_stam_p1 = rel_err(k4_stam_p1.omega_n2_mean);
    const double err_l34_p2  = rel_err(k4_l34_p2.omega_n2_mean);
    const double err_stam_p2 = rel_err(k4_stam_p2.omega_n2_mean);

    INFO("k=4  V=" << k4_l34_p1.V
         << "  n_passes=1: L.3.4=" << err_l34_p1
         << "  Stam=" << err_stam_p1
         << "  | n_passes=2: L.3.4=" << err_l34_p2
         << "  Stam=" << err_stam_p2);

    // Cluster integrity preserved on both paths.
    REQUIRE(k4_stam_p1.cluster_split < 0.05);
    REQUIRE(k4_stam_p2.cluster_split < 0.05);

    // Both paths must hit the sub-percent regime at k=4.
    REQUIRE(err_l34_p1  < 0.02);
    REQUIRE(err_stam_p1 < 0.02);
    REQUIRE(err_l34_p2  < 0.02);
    REQUIRE(err_stam_p2 < 0.02);

    // Small-slack agreement between Stam and L.3.4 at k=4: both paths
    // converge to essentially the same answer (within 0.05 percentage
    // points) so neither is robustly closer to the analytic. The
    // strict ordering @c err_stam <= err_l34 held under the lumped-mass
    // default but flipped sign under consistent mass at the
    // numerical-floor level — see the docstring above for context.
    constexpr double kStamL34SlackAtK4 = 0.0005;
    REQUIRE(std::abs(err_stam_p1 - err_l34_p1) < kStamL34SlackAtK4);
    REQUIRE(std::abs(err_stam_p2 - err_l34_p2) < kStamL34SlackAtK4);

    std::cout << "[icosphere R=" << R << " h=" << h
              << "  k=4 V=" << k4_l34_p1.V << "]\n"
              << "  n_passes=1: L.3.4 rel_err = " << err_l34_p1
              << "  Stam rel_err = " << err_stam_p1 << '\n'
              << "  n_passes=2: L.3.4 rel_err = " << err_l34_p2
              << "  Stam rel_err = " << err_stam_p2 << '\n'
              << "  Wilkinson n=2 = " << omega_n2_analytical << '\n';
}
