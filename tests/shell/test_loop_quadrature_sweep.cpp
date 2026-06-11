/**
 * @file test_loop_quadrature_sweep.cpp
 * @brief A/B characterization of K / M quadrature rules on the existing
 *        analytic-fixture suite.
 *
 * The ShellAssembler framework introduced @ref chladni::shell::QuadratureRule
 * with three concrete rules:
 *  - @c OnePointCentroid    — Cirak-Ortiz Sec 4.6 statics baseline (degree 1)
 *  - @c ThreePointEdgeMid   — classical degree-2 rule
 *  - @c SevenPointDunavant  — the shipped default (degree-5)
 *
 * This benchmark sweeps the diagonal @c (k_quad, m_quad) ∈ {(1,1), (3,3), (7,7)}
 * across two fixtures with known analytic spectra, prints a frequency
 * table, and asserts only a coarse sanity bound (every rule lands within
 * an order of magnitude of the analytic). The actual numerical
 * characterisation lives in the printed output — the test is meant to
 * be re-run after any change that touches the FEM bending energy.
 *
 * Tagged @c [.benchmark] so it does not run in the default suite. Invoke
 * with @c "./build/tests/chladni_tests [.benchmark]" to execute.
 *
 * Fixtures:
 *  1. Wilkinson n=2 closed-sphere mode on icosphere k=2 (162 V). Every
 *     face is irregular (valence-5 corner) so the Stam path is exercised
 *     on every triangle. Under the consistent-mass default (post
 *     2026-05-17 late) the L.3.4 path lands sub-percent at k=2; earlier
 *     versions of this file cited "~37 % rel_err at k=2 under defaults"
 *     which reflected the @ref chladni::shell::MassLumping::RowSum
 *     default in force at the time, not a quadrature-rule effect.
 *  2. Leissa free-edge circular plate, lowest doublet (n=2, s=0) and
 *     ring mode (n=0, s=1), on a polar disk (32 × 8 = 257 V). The
 *     central vertex is valence-32 extraordinary; the rim is valence-4
 *     Schweitzer boundary. Convergence is sub-5% under defaults.
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/analytical/shell.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

namespace cs   = chladni::shell;
namespace cmsh = chladni::mesh;

namespace {

chladni::IsotropicMaterial steel_030()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.30,
            .density        = 7850.0};
}

chladni::IsotropicMaterial steel_033()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.33,
            .density        = 7850.0};
}

const char* rule_label(cs::QuadratureRule r)
{
    switch (r) {
    case cs::QuadratureRule::OnePointCentroid:  return "1-pt";
    case cs::QuadratureRule::ThreePointEdgeMid: return "3-pt";
    case cs::QuadratureRule::SevenPointDunavant: return "7-pt";
    }
    return "?";
}

cs::ShellModes compute_with_rule(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    const chladni::IsotropicMaterial& mat, double h, std::size_t n_modes,
    cs::QuadratureRule rule, int n_passes = 1, bool use_stam = true)
{
    cs::LoopAssembler::Params p;
    p.k_quad   = rule;
    p.m_quad   = rule;
    p.n_passes = n_passes;
    p.use_stam = use_stam;
    const auto sm = cs::shell_material_from_isotropic(mat, h);
    return cs::compute_shell_modes(
        V, F, mat, sm, h, n_modes, cs::LoopAssembler{p});
}

}  // namespace

TEST_CASE("quadrature sweep: Wilkinson n=2 on icosphere k=2",
          "[.benchmark][shell][quadrature][sweep]")
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;

    const auto mesh = cmsh::generate_icosphere(R, /*n_subdivisions=*/2);
    const auto analytical = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel_030(), /*n_modes=*/1);
    const double w_an = analytical.front();

    std::cout << "\n[quadrature sweep] Wilkinson n=2 closed sphere"
                 ", icosphere k=2 (V = " << mesh.V.rows() << ")\n"
                 "  analytic angular freq = " << w_an << " rad/s\n"
                 "  rule  | cluster mean (rad/s) | rel_err vs analytic\n"
                 "  ------+----------------------+--------------------\n";

    for (const auto rule : {cs::QuadratureRule::OnePointCentroid,
                            cs::QuadratureRule::ThreePointEdgeMid,
                            cs::QuadratureRule::SevenPointDunavant}) {
        const auto modes = compute_with_rule(
            mesh.V, mesh.F, steel_030(), h, /*n_modes=*/6, rule,
            /*n_passes=*/2, /*use_stam=*/true);

        // Lowest 5 modes form the n=2 spheroidal pentet under icosahedral
        // symmetry. Take the cluster mean.
        double sum = 0.0;
        for (Eigen::Index i = 0; i < 5; ++i) sum += modes.omegas(i);
        const double w_mean  = sum / 5.0;
        const double rel_err = std::abs(w_mean - w_an) / w_an;

        std::cout << "  " << rule_label(rule)
                  << "  | " << std::setw(20) << w_mean
                  << " | " << rel_err << '\n';

        // Coarse sanity — every rule should land within an order of
        // magnitude of the analytic. Tightening this is the FEM team's
        // call; the point of the sweep is the printed table.
        REQUIRE(w_mean > 0.0);
        REQUIRE(rel_err < 1.0);
    }
}

TEST_CASE("quadrature sweep: Leissa free-edge disk, lowest 3 modes",
          "[.benchmark][shell][quadrature][sweep]")
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;

    const auto mesh =
        cmsh::generate_circular_disk(R, /*n_az=*/32, /*n_rad=*/8);
    const auto analytical = chladni::analytical::
        free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, steel_033(), n_modes);

    // Leissa expected pattern: (2,0) doublet, (0,1) singleton, (3,0) doublet
    // (Chladni saddle, breathing ring, trefoil).
    const double w_an_2_0 = 0.5 * (analytical[0] + analytical[1]);  // (2,0) mean
    const double w_an_0_1 = analytical[2];                          // (0,1)
    const double w_an_3_0 = 0.5 * (analytical[3] + analytical[4]);  // (3,0) mean

    std::cout << "\n[quadrature sweep] Leissa free-edge circular plate"
                 ", polar 32x8 (V = " << mesh.V.rows() << ")\n"
                 "  analytic (Hz): (2,0)=" << (w_an_2_0 / (2.0 * std::numbers::pi))
              << "  (0,1)=" << (w_an_0_1 / (2.0 * std::numbers::pi))
              << "  (3,0)=" << (w_an_3_0 / (2.0 * std::numbers::pi)) << '\n'
              << "  rule  |  (2,0) Hz |  err  |  (0,1) Hz |  err  |  (3,0) Hz |  err\n"
                 "  ------+-----------+-------+-----------+-------+-----------+------\n";

    constexpr double tau = 2.0 * std::numbers::pi;

    for (const auto rule : {cs::QuadratureRule::OnePointCentroid,
                            cs::QuadratureRule::ThreePointEdgeMid,
                            cs::QuadratureRule::SevenPointDunavant}) {
        const auto modes = compute_with_rule(
            mesh.V, mesh.F, steel_033(), h, n_modes, rule);
        REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n_modes));

        const double w_fem_2_0 =
            0.5 * (modes.omegas(0) + modes.omegas(1));
        const double w_fem_0_1 = modes.omegas(2);
        const double w_fem_3_0 =
            0.5 * (modes.omegas(3) + modes.omegas(4));

        auto rerr = [](double f, double a) {
            return std::abs(f - a) / a;
        };

        std::cout << "  " << rule_label(rule)
                  << "  | " << std::setw(9) << (w_fem_2_0 / tau)
                  << " | " << std::setw(5) << rerr(w_fem_2_0, w_an_2_0)
                  << " | " << std::setw(9) << (w_fem_0_1 / tau)
                  << " | " << std::setw(5) << rerr(w_fem_0_1, w_an_0_1)
                  << " | " << std::setw(9) << (w_fem_3_0 / tau)
                  << " | " << std::setw(5) << rerr(w_fem_3_0, w_an_3_0)
                  << '\n';

        REQUIRE(modes.omegas.allFinite());
        REQUIRE(w_fem_2_0 > 0.0);
        REQUIRE(rerr(w_fem_2_0, w_an_2_0) < 0.5);
    }
}
